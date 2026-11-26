#include "omlx_m5/dflash_forward.h"
#include "affine_nax_cache.h"
#include "bf16_nax.h"
#include "forward_plan.h"
#include "mxfp4_nax_cache.h"
#include "silicon_tuning.h"

#include <array>
#include <deque>
#include <numeric>

namespace omlx::m5 {
using namespace detail::forward;

namespace {
struct SelectorParams {
  U32 vocab, hidden, rank, top_k, blocks, groups, width, anchor, position;
  float temperature;
  float uniforms[7];
};
static_assert(sizeof(SelectorParams) == 68);
struct ContextGatherOffsetHost {
  U32 tokens = 0, width = 0, position = 0, ring_rows = 0;
  U32 dst_row_offset = 0, pad0 = 0, pad1 = 0, pad2 = 0;
};
static_assert(sizeof(ContextGatherOffsetHost) == 32);
struct ContextRopeAppendOffsetHost {
  U32 tokens = 0, kv_heads = 0, dim = 0, position = 0, capacity = 0;
  float epsilon = 0.0f, theta = 0.0f;
  U32 src_row_offset = 0;
};
static_assert(sizeof(ContextRopeAppendOffsetHost) == 32);
struct BatchedConvParamsHost {
  U32 batch_size = 0, hidden = 0, group_size = 0, output_half = 0;
  std::array<U32, 8> slot_widths{};
};
static_assert(sizeof(BatchedConvParamsHost) == 48);
struct BatchedConvAddNormParamsHost {
  U32 batch_size = 0, hidden = 0, group_size = 0;
  float epsilon = 0.0f;
  std::array<U32, 8> slot_widths{};
};
static_assert(sizeof(BatchedConvAddNormParamsHost) == 48);
struct BatchedQkRopeStridedParamsHost {
  U32 batch_size = 0, q_heads = 0, kv_heads = 0, dim = 0, qkv_stride = 0;
  float epsilon = 0.0f, theta = 0.0f;
  U32 pad = 0;
  std::array<U32, 8> slot_widths{};
  std::array<U32, 8> slot_positions{};
};
static_assert(sizeof(BatchedQkRopeStridedParamsHost) == 96);
struct BatchedRopeParamsHost {
  U32 batch_size = 0, heads = 0, dim = 0, pad = 0;
  float epsilon = 0.0f, theta = 0.0f;
  std::array<U32, 8> slot_widths{};
  std::array<U32, 8> slot_positions{};
};
static_assert(sizeof(BatchedRopeParamsHost) == 88);

template <class T> std::vector<std::byte> pod_bytes(const T &v) {
  const auto b = std::as_bytes(std::span(&v, 1));
  return {b.begin(), b.end()};
}
template <class T> std::vector<std::byte> pod_span_bytes(std::span<const T> s) {
  const auto b = std::as_bytes(s);
  return {b.begin(), b.end()};
}
Ref df_row_slice_ref(Ref base, U32 row_off, U32 rows, U32 cols,
                     U32 elem_bytes = 2U) {
  const U64 off = mul(mul(U64(row_off), U64(cols)), U64(elem_bytes));
  const U64 len = mul(mul(U64(rows), U64(cols)), U64(elem_bytes));
  if (!base.buffer || off + len > base.bytes)
    throw std::out_of_range("DFlash packed batch row slice out of bounds");
  return Ref{base.buffer, base.offset + off, len};
}
void push_unique_offset(std::vector<U64> &dst, U64 off) {
  if (std::ranges::find(dst, off) == dst.end())
    dst.push_back(off);
}
constexpr U32 selector_chunk = 256;
constexpr U32 selector_stage1_chunks = 64;
constexpr U32 selector_max_top_k = 32;

class DFlashBf16QkvCache {
public:
  struct LayerEntry {
    Ref qkv_data;
    Ref kv_data;
    U32 q = 0;
    U32 kv = 0;
    U32 h = 0;
  };

  [[nodiscard]] bool active() const noexcept { return !layers_.empty(); }

  [[nodiscard]] const LayerEntry *find(U32 layer) const noexcept {
    return (layer < layers_.size() && layers_[layer].qkv_data.buffer != nullptr)
               ? &layers_[layer]
               : nullptr;
  }

  static std::shared_ptr<DFlashBf16QkvCache>
  acquire(const void *owner_key, const MetalContext &ctx,
          const std::filesystem::path &libs, const MaterializedModel &weights,
          const DFlash2ModelConfig &cfg) {
    return nax_acquire_owner_cache<DFlashBf16QkvCache>(
        owner_key, [&](DFlashBf16QkvCache &cache) {
          cache.build(ctx, libs, weights, cfg);
        });
  }

private:
  void build(const MetalContext &ctx, const std::filesystem::path &libs,
             const MaterializedModel &weights, const DFlash2ModelConfig &cfg) {
    const U32 h = cfg.hidden_size;
    const U32 q = cfg.query_heads * cfg.head_dim;
    const U32 kv = cfg.key_value_heads * cfg.head_dim;
    if (cfg.layer_count == 0u || h < 256u || (h % 64u) != 0u || q < 64u ||
        (q % 64u) != 0u || kv < 64u || (kv % 64u) != 0u) {
      return;
    }
    for (U32 l = 0; l < cfg.layer_count; ++l) {
      const auto p = "dflash2.layers." + std::to_string(l) + ".";
      const auto qw = weights.parent_view(p + "self_attn.q_proj.weight");
      const auto kw = weights.parent_view(p + "self_attn.k_proj.weight");
      const auto vw = weights.parent_view(p + "self_attn.v_proj.weight");
      if (!bf16_nax_weight_eligible(qw, q, h) ||
          !bf16_nax_weight_eligible(kw, kv, h) ||
          !bf16_nax_weight_eligible(vw, kv, h)) {
        return;
      }
    }

    NaxSegmentArena arena;
    std::vector<NaxSegmentArena::Placement> placements;
    placements.reserve(cfg.layer_count);
    const U64 total_rows = U64(q) + 2ULL * U64(kv);
    const U64 layer_bytes = mul(total_rows, mul(U64(h), 2ULL));
    for (U32 l = 0; l < cfg.layer_count; ++l) {
      placements.push_back(arena.plan_alloc(layer_bytes, "DFlash2 BF16 QKV"));
    }
    arena.materialize(ctx, segments_);
    layers_.resize(cfg.layer_count);

    static std::atomic<U64> pack_id{0};
    Encoder enc(ctx, libs, 1,
                "dflash.bf16.qkv.pack." + std::to_string(++pack_id));
    enc.begin_stage("pack_qkv_bf16");
    const U32 vec_cols = (h & 7u) == 0u ? (h >> 3u) : h;
    const U32 tg_x = std::min<U32>(64u, vec_cols);
    const U32 tg_y = std::max<U32>(1u, std::min<U32>(4u, U32(total_rows)));
    const U32 grid_x = ((vec_cols + tg_x - 1u) / tg_x) * tg_x;
    const U32 grid_y = ((U32(total_rows) + tg_y - 1u) / tg_y) * tg_y;
    const U64 q_bytes = mul(U64(q), mul(U64(h), 2ULL));
    const U64 kv2_bytes = mul(2ULL * U64(kv), mul(U64(h), 2ULL));

    for (U32 l = 0; l < cfg.layer_count; ++l) {
      const Ref qkv_ref = NaxSegmentArena::resolve(segments_, placements[l]);
      const auto p = "dflash2.layers." + std::to_string(l) + ".";
      auto &d = enc.emit(
          "dflash_forward", "omlx_m5_draft_pack_qkv_bf16",
          {.width = grid_x, .height = grid_y}, {.width = tg_x, .height = tg_y},
          {plane(weights.parent_view(p + "self_attn.q_proj.weight"), "data"),
           plane(weights.parent_view(p + "self_attn.k_proj.weight"), "data"),
           plane(weights.parent_view(p + "self_attn.v_proj.weight"), "data"),
           qkv_ref});
      enc.dynamic(d, 4, words({q, kv, h}));
      layers_[l] = LayerEntry{
          .qkv_data = qkv_ref,
          .kv_data = Ref{qkv_ref.buffer, qkv_ref.offset + q_bytes, kv2_bytes},
          .q = q,
          .kv = kv,
          .h = h,
      };
    }
    auto pack_prof = enc.finish();
    const std::array calls = {
        MetalCommandInvocation{.plan = &pack_prof.plan,
                               .bindings = pack_prof.bindings,
                               .parameters = &pack_prof.parameters}};
    static_cast<void>(ctx.execute_batch(calls));
  }

  std::vector<std::unique_ptr<MetalBuffer>> segments_;
  std::vector<LayerEntry> layers_;
};

void fnv1a_bytes(U64 &hash, const void *data, std::size_t length) {
  const auto *ptr = static_cast<const unsigned char *>(data);
  for (std::size_t i = 0; i < length; ++i) {
    hash ^= U64(ptr[i]);
    hash *= 1099511628211ULL;
  }
}

void fnv1a_str(U64 &hash, std::string_view text) {
  const U64 len = text.size();
  fnv1a_bytes(hash, &len, sizeof(len));
  fnv1a_bytes(hash, text.data(), text.size());
}

template <class T> void fnv1a_val(U64 &hash, const T &value) {
  fnv1a_bytes(hash, &value, sizeof(value));
}

std::string hex64(U64 value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i) {
    out[std::size_t(i)] = digits[value & 0xFU];
    value >>= 4U;
  }
  return out;
}

#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

std::pair<U64, U64> hash_feature_row(std::span<const std::byte> row) {
#if defined(__ARM_FEATURE_CRC32)
  std::uint32_t c0 = 0xFFFFFFFFU, c1 = 0x9E3779B9U;
  std::uint32_t c2 = 0x85EBCA6BU, c3 = 0xC2B2AE35U;
  const auto *ptr = reinterpret_cast<const unsigned char *>(row.data());
  std::size_t offset = 0;
  const std::size_t n = row.size();
  for (; offset + 32 <= n; offset += 32) {
    U64 w0, w1, w2, w3;
    std::memcpy(&w0, ptr + offset, 8);
    std::memcpy(&w1, ptr + offset + 8, 8);
    std::memcpy(&w2, ptr + offset + 16, 8);
    std::memcpy(&w3, ptr + offset + 24, 8);
    c0 = __crc32cd(c0, w0);
    c1 = __crc32cd(c1, w1);
    c2 = __crc32cd(c2, w2);
    c3 = __crc32cd(c3, w3);
  }
  for (; offset + 8 <= n; offset += 8) {
    U64 w0;
    std::memcpy(&w0, ptr + offset, 8);
    c0 = __crc32cd(c0, w0);
    c1 = __crc32d(c1, w0);
  }
  for (; offset < n; ++offset) {
    c0 = __crc32cb(c0, ptr[offset]);
    c2 = __crc32b(c2, ptr[offset]);
  }
  return {(U64(c0) << 32U) | U64(c1), (U64(c2) << 32U) | U64(c3)};
#else
  U64 lo = 14695981039346656037ULL;
  U64 hi = 0x9E3779B97F4A7C15ULL;
  std::size_t offset = 0;
  while (offset + 8 <= row.size()) {
    U64 chunk = 0;
    std::memcpy(&chunk, row.data() + offset, 8);
    lo = (lo ^ chunk) * 1099511628211ULL;
    hi ^= chunk + 0x9E3779B97F4A7C15ULL + (hi << 6U) + (hi >> 2U);
    offset += 8;
  }
  if (offset < row.size()) {
    fnv1a_bytes(lo, row.data() + offset, row.size() - offset);
    fnv1a_bytes(hi, row.data() + offset, row.size() - offset);
  }
  return {lo, hi};
#endif
}
} // namespace

QwenForwardMemoryPlan plan_dflash_forward_memory(const DFlash2ModelConfig &c,
                                                 U32 max_context) {
  if (!max_context || !c.hidden_size || c.head_dim > 256 || c.head_dim % 2 ||
      c.hidden_size % 64 || c.intermediate_size % 64 || !c.sliding_window ||
      !c.selector_top_k || c.selector_top_k > selector_max_top_k ||
      !c.selector_rank || !c.vocabulary_size)
    throw std::invalid_argument("DFlash execution geometry unsupported");
  QwenForwardMemoryPlan plan;
  const auto add = [&](std::string name, U64 rows, U64 columns, U64 bytes = 2) {
    U64 count = mul(mul(rows, columns), bytes);
    if (count > UINT32_MAX * 2ULL)
      throw std::overflow_error("DFlash Metal indexing overflow");
    plan.buffers.push_back({std::move(name), count, false, false});
    plan.workspace_bytes += count;
    plan.largest_buffer_bytes = std::max(plan.largest_buffer_bytes, count);
  };
  U64 h = c.hidden_size, q = c.query_heads * c.head_dim,
      kv = c.key_value_heads * c.head_dim;
  add("tokens", 8, 1, 4);
  add("hidden", 8, h);
  add("norm", 8, h);
  add("context_input", std::min(max_context, 1024U),
      h * c.target_layer_ids.size());
  add("context_raw", std::min(max_context, 1024U), h);
  add("context_norm", std::min(max_context, 1024U), h);
  add("dynamic", 8, 4 * h / c.conv_group_size);
  add("prepared", 8, h);
  add("qkv", 8, q + 2 * kv);
  add("context_kv", std::min(max_context, 1024U), 2 * kv);
  for (U32 layer = 0; layer < c.layer_count; ++layer) {
    const auto capacity = std::min(max_context, c.sliding_window);
    add("cache_key." + std::to_string(layer), capacity, kv);
    add("cache_value." + std::to_string(layer), capacity, kv);
  }
  add("attention", 8, q);
  add("projected", 8, h);
  add("gate", 8, c.intermediate_size);
  add("up", 8, c.intermediate_size);
  add("product", 8, c.intermediate_size);
  add("logits", 8, c.vocabulary_size);
  add("selector_partial_ids", 8ULL * selector_stage1_chunks, c.selector_top_k,
      4);
  add("selector_partial_scores", 8ULL * selector_stage1_chunks,
      c.selector_top_k, 4);
  add("selector_ids", 8, c.selector_top_k, 4);
  add("selector_scores", 8, c.selector_top_k, 4);
  add("selector_projected", 8, c.selector_rank, 4);
  add("selector_tokens", 8, 1, 4);
  add("selector_probabilities", 8, c.selector_top_k, 4);
  add("selector_fallback", 1, 1, 4);
  add("sums", 8, (std::max({h, q, U64(c.intermediate_size)}) + 31) / 32, 4);
  add("zero", 1, 256, 1);
  return plan;
}

class DFlashForward::Impl {
public:
  struct ContextProfile {
    Profile profile;
    U64 gather_ring_offset = 0;
  };
  struct FusedProfile {
    Profile profile;
    U32 context_capacity = 0;
    std::size_t context_patch_count = 0;
    U64 gather_ring_offset = 0;
    U64 keyed_sampling_offset = 0;
  };

  Impl(const MetalContext &ctx, const MaterializedModel &w,
       DFlash2ModelConfig c, U32 max, const MaterializedWeightView &emb,
       const MaterializedWeightView &head, const MetalBuffer &features,
       const std::filesystem::path &directory, bool enable_m4_short_routes,
       LinearNumericPolicy linear_policy, SamplingContract sampling_contract)
      : context(ctx), weights(w), config(std::move(c)), max_context(max),
        embedding(emb), head(head), features{&features, 0, features.size()},
        directory(directory), m4_short_routes(enable_m4_short_routes),
        linear_policy(linear_policy), sampling_contract(sampling_contract) {
    (void)sampling_contract_name(sampling_contract);
    const auto &cfg = config;
    h = cfg.hidden_size;
    q = cfg.query_heads * cfg.head_dim;
    kv = cfg.key_value_heads * cfg.head_dim;
    feature_row_bytes = mul(U64(cfg.target_layer_ids.size()), U64(h) * 2ULL);
    const U64 min_required_rows = std::min<U64>(max, cfg.sliding_window);
    if (embedding.parent_shape != std::vector<U64>{cfg.vocabulary_size, h} ||
        head.parent_shape != embedding.parent_shape ||
        (feature_row_bytes > 0 &&
         (features.size() < mul(min_required_rows, feature_row_bytes) ||
          (features.size() % feature_row_bytes) != 0)))
      throw std::invalid_argument("DFlash shared target contract mismatch");
    if (feature_row_bytes > 0) {
      const U64 physical_rows = features.size() / feature_row_bytes;
      if (physical_rows < max_context) {
        feature_ring_rows = U32(physical_rows);
      }
    }
    for (const auto &item : plan_dflash_forward_memory(cfg, max).buffers) {
      auto buffer =
          std::make_unique<MetalBuffer>(context.make_buffer(item.bytes));
      refs.emplace(item.name, Ref{buffer.get(), 0, item.bytes});
      buffers.push_back(std::move(buffer));
    }
    const_cast<MetalBuffer *>(get("zero").buffer)
        ->write(std::vector<std::byte>(256));
    cached_slots.resize(std::min(max_context, cfg.sliding_window));
    gpu_cores = nax_estimate_gpu_cores(context.capabilities().device_name,
                                       detail::probe_host_agx_gpu_cores());
    bf16_nax_enabled =
        (linear_policy == LinearNumericPolicy::RowExactPortableV1 ||
         linear_policy == LinearNumericPolicy::Legacy) &&
        bf16_nax_runtime_eligible(context, directory);
    selector_nax_f32_enabled =
        bf16_nax_enabled && h >= 256u && (h % 64u) == 0u &&
        cfg.selector_rank >= 16u &&
        std::filesystem::is_regular_file(
            directory / "omlx_m5_attention_nax_prefill.metallib");
    if (bf16_nax_enabled && h >= 256u) {
      dflash_qkv_cache =
          DFlashBf16QkvCache::acquire(&weights, context, directory, weights, cfg);
    }
    if (linear_policy == LinearNumericPolicy::RowExactPortableV1 &&
        mxfp4_nax_runtime_eligible(context, directory)) {
      std::vector<MaterializedWeightView> candidates;
      candidates.push_back(head);
      candidates.push_back(weight("fc.weight"));
      for (U32 layer = 0; layer < config.layer_count; ++layer) {
        const auto p = "layers." + std::to_string(layer) + ".";
        for (const char *suffix :
             {"attention_conv.kernel_projection.weight",
              "self_attn.q_proj.weight", "self_attn.k_proj.weight",
              "self_attn.v_proj.weight", "self_attn.o_proj.weight",
              "mlp_conv.kernel_projection.weight", "mlp.gate_proj.weight",
              "mlp.up_proj.weight", "mlp.down_proj.weight"}) {
          candidates.push_back(weight(p + suffix));
        }
      }
      mxfp4_nax_cache =
          Mxfp4NaxCache::acquire(&weights, context, directory, candidates);
      if (mxfp4_nax_cache && mxfp4_nax_cache->active()) {
        const U32 max_mp = mxfp4_nax_mp(std::min(max_context, 1024U));
        const U32 max_k =
            std::max({h, q, config.intermediate_size,
                      U32(config.target_layer_ids.size()) * h});
        const U64 f16_bytes = mul(mul(max_mp, max_k), 2u);
        const U64 rs_bytes = mul(max_mp, 4u);
        auto f16_buf = std::make_unique<MetalBuffer>(
            context.make_private_buffer(f16_bytes));
        auto f16_mid_buf = std::make_unique<MetalBuffer>(
            context.make_private_buffer(f16_bytes));
        auto rs_buf =
            std::make_unique<MetalBuffer>(context.make_private_buffer(rs_bytes));
        auto ones_buf =
            std::make_unique<MetalBuffer>(context.make_buffer(rs_bytes));
        std::vector<float> ones_init(max_mp, 1.0f);
        ones_buf->write(std::as_bytes(std::span(ones_init)));
        f16_scratch = Ref{f16_buf.get(), 0, f16_bytes};
        f16_mid = Ref{f16_mid_buf.get(), 0, f16_bytes};
        f16_row_scale = Ref{rs_buf.get(), 0, rs_bytes};
        f16_ones_scale = Ref{ones_buf.get(), 0, rs_bytes};
        buffers.push_back(std::move(f16_buf));
        buffers.push_back(std::move(f16_mid_buf));
        buffers.push_back(std::move(rs_buf));
        buffers.push_back(std::move(ones_buf));
      }
    }
    if ((linear_policy == LinearNumericPolicy::RowExactPortableV1 ||
         head.layout == "oq_nax_tiled_g64_v1" ||
         head.layout == "oq_nax_tiled_g128_v1") &&
        affine_nax_runtime_eligible(context, directory) &&
        affine_nax_weight_eligible(head)) {
      const void *head_key =
          head.planes.empty()
              ? static_cast<const void *>(&weights)
              : static_cast<const void *>(head.planes.front().buffer);
      affine_nax_head_cache = AffineNaxCache::acquire(
          head_key, context, directory, get("zero"), {head});
      if (affine_nax_head_cache && affine_nax_head_cache->active()) {
        const U32 max_mp = affine_nax_mp(std::min(max_context, 1024U));
        const U64 xs_bytes = mul(mul(max_mp, h / 64u), 4u);
        auto xs_buf = std::make_unique<MetalBuffer>(
            context.make_private_buffer(xs_bytes));
        affine_nax_xs = Ref{xs_buf.get(), 0, xs_bytes};
        buffers.push_back(std::move(xs_buf));
      }
    }
    const auto install_icb = [&](Profile &installed) {
      if (context.capabilities().profile.compute_icb_eligible)
        installed.icb.emplace(context.prepare_icb_profile(
            installed.plan.key(), {.plan = &installed.plan,
                                   .bindings = installed.bindings,
                                   .parameters = &installed.parameters}));
    };
    profile = compile(true, keyed_sampling_offset);
    install_icb(profile);
    U64 unused_cpu_keyed = 0;
    cpu_profile = compile(false, unused_cpu_keyed);
    install_icb(cpu_profile);

    context_profiles.reserve(11);
    for (U32 capacity :
         {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U, 256U, 512U, 1024U}) {
      if (capacity > max_context)
        break;
      context_profiles.push_back(compile_context(capacity));
      install_icb(context_profiles.back().profile);
    }
    const U32 final_capacity = std::min(max_context, 1024U);
    if (context_profiles.back().profile.capacity < final_capacity) {
      context_profiles.push_back(compile_context(final_capacity));
      install_icb(context_profiles.back().profile);
    }

    fused_profiles.reserve(4);
    fused_cpu_profiles.reserve(4);
    for (U32 ctx_cap : {1U, 2U, 4U, 8U}) {
      if (ctx_cap > max_context)
        break;
      fused_profiles.push_back(compile_fused(ctx_cap, true));
      install_icb(fused_profiles.back().profile);
      fused_cpu_profiles.push_back(compile_fused(ctx_cap, false));
      install_icb(fused_cpu_profiles.back().profile);
    }
    const U32 max_small_ctx = std::min(max_context, 8U);
    if (!fused_profiles.empty() &&
        fused_profiles.back().context_capacity < max_small_ctx) {
      fused_profiles.push_back(compile_fused(max_small_ctx, true));
      install_icb(fused_profiles.back().profile);
      fused_cpu_profiles.push_back(compile_fused(max_small_ctx, false));
      install_icb(fused_cpu_profiles.back().profile);
    }
  }
  Ref get(const std::string &name) const { return refs.at(name); }
  MaterializedWeightView weight(const std::string &name) const {
    return weights.parent_view("dflash2." + name);
  }
  Ref data(const std::string &name) const {
    return plane(weight(name), "data");
  }
  std::vector<float> read_weight(const std::string &name) {
    auto ref = data(name);
    auto bytes = ref.buffer->read(ref.offset, ref.bytes);
    std::vector<float> values(bytes.size() / 2);
    for (U64 i = 0; i < values.size(); ++i) {
      std::uint16_t code;
      std::memcpy(&code, bytes.data() + 2 * i, 2);
      values[i] = std::bit_cast<float>(U32(code) << 16);
    }
    return values;
  }
  U64 ensure_cpu_selector() {
    if (!selector.empty())
      return 0;
    selector = read_weight("candidate_selector.hidden_projection.weight");
    predecessor = read_weight("candidate_selector.predecessor_codebook");
    successor = read_weight("candidate_selector.successor_codebook");
    return (selector.size() + predecessor.size() + successor.size()) * 2;
  }
  std::string plan_key_suffix() const {
    const auto fnv_weight = [](U64 &hash, const MaterializedWeightView &w) {
      fnv1a_val(hash, w.object_id);
      fnv1a_str(hash, w.format);
      fnv1a_str(hash, w.layout);
      fnv1a_val(hash, w.padded_k);
      fnv1a_val(hash, w.row_start);
      fnv1a_val(hash, w.row_count);
      fnv1a_val(hash, w.in_features);
      for (const auto &p : w.planes) {
        fnv1a_str(hash, p.kind);
        fnv1a_val(hash, p.bytes);
      }
    };
    U64 wsig = 14695981039346656037ULL;
    fnv1a_val(wsig, features.bytes);
    fnv_weight(wsig, embedding);
    fnv_weight(wsig, head);
    const bool tied_head =
        !embedding.planes.empty() && !head.planes.empty() &&
        embedding.planes.front().buffer == head.planes.front().buffer &&
        embedding.planes.front().offset == head.planes.front().offset;
    fnv1a_val(wsig, tied_head);
    fnv_weight(wsig, weight("fc.weight"));
    fnv_weight(wsig, weight("hidden_norm.weight"));
    fnv_weight(wsig, weight("norm.weight"));
    fnv_weight(wsig, weight("candidate_selector.hidden_projection.weight"));
    fnv_weight(wsig, weight("candidate_selector.predecessor_codebook"));
    fnv_weight(wsig, weight("candidate_selector.successor_codebook"));
    for (U32 layer = 0; layer < config.layer_count; ++layer) {
      const auto p = "layers." + std::to_string(layer) + ".";
      for (const char *suffix :
           {"input_layernorm.weight", "attention_conv.kernel_projection.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp_conv.kernel_projection.weight", "mlp.gate_proj.weight",
            "mlp.up_proj.weight", "mlp.down_proj.weight"}) {
        fnv_weight(wsig, weight(p + suffix));
      }
    }
    std::string taps = config.target_layer_ids.empty() ? "none" : "";
    for (std::size_t i = 0; i < config.target_layer_ids.size(); ++i) {
      if (i)
        taps.push_back('-');
      taps += std::to_string(config.target_layer_ids[i]);
    }
    return ".L" + std::to_string(config.layer_count) + ".h" +
           std::to_string(h) + ".f" +
           std::to_string(config.intermediate_size) + ".v" +
           std::to_string(config.vocabulary_size) + ".qh" +
           std::to_string(config.query_heads) + ".kvh" +
           std::to_string(config.key_value_heads) + ".hd" +
           std::to_string(config.head_dim) + ".cg" +
           std::to_string(config.conv_group_size) + ".sw" +
           std::to_string(config.sliding_window) + ".bs" +
           std::to_string(config.block_size) + ".sr" +
           std::to_string(config.selector_rank) + ".sk" +
           std::to_string(config.selector_top_k) + ".eps" +
           std::to_string(word(config.epsilon)) + ".th" +
           std::to_string(word(config.rope_theta)) + ".ctx" +
           std::to_string(max_context) + ".m4sr" +
           (m4_short_routes ? "1" : "0") + ".bf16nax" +
           (bf16_nax_enabled ? "1" : "0") + ".tl-" + taps + ".ws" +
           hex64(wsig) + ".v2";
  }
  void emit_bf16_nax_fused(
      Encoder &e, Ref w_data,
      std::initializer_list<std::pair<MaterializedWeightView, U32>> parts,
      Ref x, Ref y, U32 n, U32 k) {
    const U32 t = e.profile.capacity;
    const U32 mp = bf16_nax_mp(t);
    const U32 sel_t = nax_effective_select_t(e, std::nullopt);
    const auto sel_exp = nax_effective_select_explicit_t(e, std::nullopt);
    const U32 bk = bf16_nax_bk(k);
    const std::optional<U32> tokens_patch(3u);
    const U32 default_padded_n = ((n + 15u) / 16u) * 16u;
    const NaxWidePrefillTile wide_tile =
        (std::getenv("OMLX_M5_BF16_NAX_BN") == nullptr &&
         std::getenv("OMLX_M5_BF16_NAX_SK") == nullptr)
            ? nax_select_wide_prefill_tile(default_padded_n, sel_t, sel_exp,
                                           gpu_cores)
            : NaxWidePrefillTile::none;
    std::string grp;
    if (wide_tile != NaxWidePrefillTile::none) {
      const bool is_64 = (wide_tile == NaxWidePrefillTile::bm64_bn64);
      const U32 tile_mn = is_64 ? 64u : 32u;
      const U32 tptg = is_64 ? 128u : 32u;
      const U32 x_tiles = (n + tile_mn - 1u) / tile_mn;
      const U32 y_tiles = (t + tile_mn - 1u) / tile_mn;
      const std::string tag = is_64 ? "bm64_bn64" : "bm32_bn32";
      const std::string fn =
          "omlx_m5_bf16_nax_bk" + std::to_string(bk) + "_" + tag + "_sk1";
      const MetalDispatchSize grid{.width = U64(x_tiles) * tptg,
                                   .height = y_tiles};
      const MetalDispatchSize threads{.width = tptg};
      e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
      auto &d = e.emit("bf16_nax", fn, grid, threads, {x, w_data, y});
      e.dynamic(d, 3, words({n, k, mp, t}), tokens_patch);
      grp = "bf16.nax.bk" + std::to_string(bk) + "." + tag + ".sk1.v1";
    } else {
      const U32 bn = bf16_nax_bn(n, bk);
      const U32 sk = bf16_nax_split_k(n, k, bk, bn);
      const U32 padded_n = ((n + bn - 1u) / bn) * bn;
      const U32 rf = nax_select_row_factor(sel_t);
      const U32 bm =
          (rf == 1u && sel_t >= 9u &&
           ((bn == 16u && (sk == 2u || sk == 4u || sk == 8u)) ||
            (bn == 32u && sk == 2u)))
              ? ((sel_t >= 17u) ? 32u : 16u)
              : 8u;
      const U32 sel_y_tiles =
          (bm > 8u) ? ((sel_t + bm - 1u) / bm)
                    : nax_y_tiles(bf16_nax_mp(sel_t), rf);
      const U32 y_tiles =
          (bm > 8u) ? ((t + bm - 1u) / bm) : nax_y_tiles(mp, rf);
      const U32 nsg =
          (bm == 8u && bn == 16u)
              ? nax_select_nsg_bn16(padded_n, sel_y_tiles, sk, gpu_cores, true)
              : 1u;
      std::string fn = "omlx_m5_bf16_nax_bk" + std::to_string(bk) +
                       (bm > 8u ? ("_bm" + std::to_string(bm)) : std::string{}) +
                       "_n" + std::to_string(bn) + "_sk" + std::to_string(sk);
      if (bm == 8u && bn == 16u && (rf > 1u || nsg > 1u)) {
        fn += "_rf" + std::to_string(rf) + "_nsg" + std::to_string(nsg);
      } else if (bm == 8u && rf > 1u) {
        fn += "_rf" + std::to_string(rf);
      }
      const U32 x_tiles = (padded_n / bn + nsg - 1u) / nsg;
      const MetalDispatchSize grid{.width = U64(x_tiles) * 32u * sk * nsg,
                                   .height = y_tiles};
      const MetalDispatchSize threads{.width = 32u * sk * nsg};
      e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
      auto &d = e.emit("bf16_nax", fn, grid, threads, {x, w_data, y});
      e.dynamic(d, 3, words({n, k, mp, t}), tokens_patch);
      grp = "bf16.nax.bk" + std::to_string(bk) + ".n" + std::to_string(bn) +
            ".sk" + std::to_string(sk) + ".v1";
    }
    for (const auto &[pw, pn] : parts) {
      e.record_weight(pw, pn, k, "bf16_nax_m5_v1", "bf16_nax_m5_v1", {},
                      "linear_row_exact_portable_v1", grp, true);
    }
  }
  void linear_with_sums(Encoder &e, const std::string &name, Ref x, Ref y,
                        U32 n, U32 k, Ref sums_ref) {
    const auto w = weight(name);
    if (linear_policy == LinearNumericPolicy::RowExactPortableV1 &&
        mxfp4_nax_active()) {
      if (const auto *tw = mxfp4_nax_cache->find(w.object_id);
           tw && tw->n == n && tw->k == k) {
        linear_mxfp4_nax(e, *tw, x, y, f16_scratch, f16_row_scale);
        e.record_weight(
            w, n, k, "mxfp4_nax_m5_v1", "mxfp4_nax_m5_v1", {},
            "linear_row_exact_portable_v1",
            "mxfp4.nax.f16_bk128.sk" + std::to_string(tw->split_k) + ".v1",
            true);
        return;
      }
    }
    if (bf16_nax_enabled && bf16_nax_weight_eligible(w, n, k)) {
      linear_bf16_nax(e, w, x, y, n, k, std::nullopt, gpu_cores);
      return;
    }
    const auto decision = resolve_linear_route(
        w, n, k, e.profile.capacity, context.capabilities().device_name,
        m4_short_routes, {}, linear_policy,
        context.capabilities().profile.probe.apple_gpu_family);
    detail::forward::linear(e, w, x, y, n, k, sums_ref, get("zero"),
                            decision.route);
    e.record_weight(w, n, k, decision.route.family, decision.source,
                    decision.fallback_reason, decision.route.numeric_contract_id,
                    decision.route.row_exact_group, decision.route.row_exact);
  }
  void linear(Encoder &e, const std::string &name, Ref x, Ref y, U32 n, U32 k) {
    linear_with_sums(e, name, x, y, n, k, get("sums"));
  }
  void norm(Encoder &e, const std::string &name, Ref x, Ref y) {
    U32 rows = e.profile.capacity;
    auto &d = e.emit("dflash_forward", "omlx_m5_draft_rmsnorm",
                     {.width = 256, .height = rows}, {.width = 256},
                     {x, data(name), y});
    e.dynamic(d, 3, words({h, rows, word(config.epsilon), 0}), 1);
  }
  void conv(Encoder &e, const std::string &name, Ref x, Ref out, U32 half) {
    if ((h & 3u) == 0u && (config.conv_group_size & 3u) == 0u) {
      const U32 vec_cols = h >> 2u;
      const U32 grid_x = ((vec_cols + 127u) / 128u) * 128u;
      auto &d = e.emit("dflash_forward", "omlx_m5_draft_conv_vec4",
                       {.width = grid_x, .height = 8}, {.width = 128},
                       {x, get("dynamic"), data(name + ".base_kernel"), out});
      e.dynamic(d, 4, words({8, h, config.conv_group_size, half}), 0);
      return;
    }
    auto &d = e.emit("dflash_forward", "omlx_m5_draft_conv",
                     {.width = U64(8) * h}, {.width = 128},
                     {x, get("dynamic"), data(name + ".base_kernel"), out});
    e.dynamic(d, 4, words({8, h, config.conv_group_size, half}), 0);
  }
  void conv_add_norm(Encoder &e, const std::string &conv_name,
                     const std::string &norm_name, Ref projected, Ref hidden,
                     Ref norm_out) {
    auto &d = e.emit("dflash_forward", "omlx_m5_draft_conv_add_rmsnorm",
                     {.width = 256, .height = 8}, {.width = 256},
                     {projected, get("dynamic"),
                      data(conv_name + ".base_kernel"), hidden,
                      data(norm_name), norm_out});
    e.dynamic(d, 6, words({8, h, config.conv_group_size, word(config.epsilon)}),
              0);
  }
  void rope(Encoder &e, Ref x, const std::string &name, U32 heads) {
    auto &d = e.emit("dflash_forward", "omlx_m5_draft_rope",
                     {.width = 32, .height = U64(e.profile.capacity) * heads},
                     {.width = 32}, {x, data(name)});
    e.dynamic(d, 2,
              words({e.profile.capacity, heads, config.head_dim, 0,
                     word(config.epsilon), word(config.rope_theta)}),
              0, 3);
  }
  U64 encode_context_body(Encoder &e, U32 capacity) {
    e.begin_stage("draft_context_projection");
    const U32 feature_width = h * U32(config.target_layer_ids.size());
    if (weight("fc.weight").format != "direct_bf16_v1")
      throw std::invalid_argument(
          "DFlash incremental FC requires BF16 weights");
    U64 gather_offset = 0;
    if ((feature_width & 7u) == 0u) {
      const U32 vec_width = feature_width >> 3u;
      const U32 grid_x = ((vec_width + 127u) / 128u) * 128u;
      auto &gather =
          e.emit("dflash_forward", "omlx_m5_draft_context_gather_vec8",
                 {.width = grid_x, .height = capacity}, {.width = 128},
                 {features, get("context_input")});
      gather_offset = e.dynamic(
          gather, 2, words({capacity, feature_width, 0, feature_ring_rows}), 0,
          2);
    } else {
      auto &gather = e.emit("dflash_forward", "omlx_m5_draft_context_gather",
                            {.width = U64(capacity) * feature_width},
                            {.width = 128}, {features, get("context_input")});
      gather_offset = e.dynamic(
          gather, 2, words({capacity, feature_width, 0, feature_ring_rows}), 0,
          2);
    }
    linear(e, "fc.weight", get("context_input"), get("context_raw"), h,
           feature_width);
    norm(e, "hidden_norm.weight", get("context_raw"), get("context_norm"));
    const U32 ctx_buf_rows = std::min(max_context, 1024U);
    const U64 ctx_kv_slice_bytes = mul(mul(ctx_buf_rows, kv), 2ULL);
    const Ref ctx_kv = get("context_kv");
    const Ref ctx_k_fallback{ctx_kv.buffer, ctx_kv.offset, ctx_kv_slice_bytes};
    const Ref ctx_v_fallback{ctx_kv.buffer, ctx_kv.offset + ctx_kv_slice_bytes,
                             ctx_kv_slice_bytes};
    const U32 ring_cap = std::min(max_context, config.sliding_window);
    for (U32 layer = 0; layer < config.layer_count; ++layer) {
      const auto prefix = "layers." + std::to_string(layer) + ".";
      const auto kw = weight(prefix + "self_attn.k_proj.weight");
      const auto vw = weight(prefix + "self_attn.v_proj.weight");
      if (kw.format != "direct_bf16_v1" || vw.format != "direct_bf16_v1")
        throw std::invalid_argument(
            "DFlash incremental KV requires BF16 weights");
      if (const auto *qkv_entry =
              (dflash_qkv_cache ? dflash_qkv_cache->find(layer) : nullptr);
          qkv_entry && (!mxfp4_nax_active() ||
                        (!mxfp4_nax_cache->find(kw.object_id) &&
                         !mxfp4_nax_cache->find(vw.object_id)))) {
        emit_bf16_nax_fused(e, qkv_entry->kv_data, {{kw, kv}, {vw, kv}},
                            get("context_norm"), ctx_kv, 2u * kv, h);
        auto &rope_append = e.emit(
            "dflash_forward", "omlx_m5_draft_context_rope_cache_append",
            {.width = 32, .height = U64(capacity) * config.key_value_heads},
            {.width = 32},
            {ctx_kv, data(prefix + "self_attn.k_norm.weight"),
             get("cache_key." + std::to_string(layer)),
             get("cache_value." + std::to_string(layer))});
        e.dynamic(rope_append, 4,
                  words({capacity, config.key_value_heads, config.head_dim, 0,
                         ring_cap, word(config.epsilon),
                         word(config.rope_theta), 0}),
                  0, 3);
      } else {
        linear(e, prefix + "self_attn.k_proj.weight", get("context_norm"),
               ctx_k_fallback, kv, h);
        linear(e, prefix + "self_attn.v_proj.weight", get("context_norm"),
               ctx_v_fallback, kv, h);
        rope(e, ctx_k_fallback, prefix + "self_attn.k_norm.weight",
             config.key_value_heads);
        auto &append = e.emit("dflash_forward", "omlx_m5_draft_cache_append",
                              {.width = U64(capacity) * kv}, {.width = 128},
                              {ctx_k_fallback, ctx_v_fallback,
                               get("cache_key." + std::to_string(layer)),
                               get("cache_value." + std::to_string(layer))});
        e.dynamic(append, 4, words({capacity, kv, 0, ring_cap}), 0, 2);
      }
    }
    return gather_offset + 12ULL;
  }
  U64 encode_backbone_body(Encoder &e, bool gpu_selector) {
    e.begin_stage("draft_backbone");
    embedding_lookup(e, embedding, get("tokens"), get("hidden"), get("zero"),
                     config.vocabulary_size, h);
    if (config.layer_count > 0) {
      norm(e, "layers.0.input_layernorm.weight", get("hidden"), get("norm"));
    } else {
      norm(e, "norm.weight", get("hidden"), get("norm"));
    }
    const Ref qkv_buf = get("qkv");
    const U64 q_slice_bytes = mul(8ULL, mul(q, 2ULL));
    const U64 kv_slice_bytes = mul(8ULL, mul(kv, 2ULL));
    const Ref q_fallback{qkv_buf.buffer, qkv_buf.offset, q_slice_bytes};
    const Ref nk_fallback{qkv_buf.buffer, qkv_buf.offset + q_slice_bytes,
                          kv_slice_bytes};
    const Ref nv_fallback{qkv_buf.buffer,
                          qkv_buf.offset + q_slice_bytes + kv_slice_bytes,
                          kv_slice_bytes};
    for (U32 layer = 0; layer < config.layer_count; ++layer) {
      const auto p = "layers." + std::to_string(layer) + ".";
      linear(e, p + "attention_conv.kernel_projection.weight", get("norm"),
             get("dynamic"), 4 * h / config.conv_group_size, h);
      conv(e, p + "attention_conv", get("norm"), get("prepared"), 0);
      const auto qw_v = weight(p + "self_attn.q_proj.weight");
      const auto kw_v = weight(p + "self_attn.k_proj.weight");
      const auto vw_v = weight(p + "self_attn.v_proj.weight");
      if (const auto *qkv_entry =
              (dflash_qkv_cache ? dflash_qkv_cache->find(layer) : nullptr);
          qkv_entry && (!mxfp4_nax_active() ||
                        (!mxfp4_nax_cache->find(qw_v.object_id) &&
                         !mxfp4_nax_cache->find(kw_v.object_id) &&
                         !mxfp4_nax_cache->find(vw_v.object_id)))) {
        const U32 qkv_stride = q + 2u * kv;
        emit_bf16_nax_fused(e, qkv_entry->qkv_data,
                            {{qw_v, q}, {kw_v, kv}, {vw_v, kv}},
                            get("prepared"), qkv_buf, qkv_stride, h);
        const U64 total_qk_heads =
            U64(8) * (config.query_heads + config.key_value_heads);
        const U64 qk_rope_tgs = (total_qk_heads + 3ULL) / 4ULL;
        auto &qk_rope = e.emit(
            "dflash_forward", "omlx_m5_draft_qk_rope_strided",
            {.width = 128, .height = qk_rope_tgs}, {.width = 128},
            {qkv_buf, data(p + "self_attn.q_norm.weight"),
             data(p + "self_attn.k_norm.weight")});
        e.dynamic(qk_rope, 3,
                  words({8, config.query_heads, config.key_value_heads,
                         config.head_dim, qkv_stride, 0, word(config.epsilon),
                         word(config.rope_theta)}),
                  0, 5);
        const U64 attn_tg_w =
            (config.sliding_window > 64U &&
             (config.head_dim == 128U || config.head_dim == 256U))
                ? 256ULL
                : 32ULL;
        const Ref nk_strided = sub(qkv_buf, mul(q, 2ULL));
        const Ref nv_strided = sub(qkv_buf, mul(q + kv, 2ULL));
        auto &a = e.emit(
            "dflash_forward", "omlx_m5_draft_attention",
            {.width = attn_tg_w, .height = U64(8) * config.query_heads},
            {.width = attn_tg_w},
            {qkv_buf, nk_strided, nv_strided,
             get("cache_key." + std::to_string(layer)),
             get("cache_value." + std::to_string(layer)), get("attention")});
        e.dynamic(a, 6,
                  words({8, 0, config.query_heads, config.key_value_heads,
                         config.head_dim, config.sliding_window,
                         std::min(max_context, config.sliding_window),
                         qkv_stride, qkv_stride}),
                  0, 1);
      } else {
        const U64 attn_tg_w =
            (config.sliding_window > 64U &&
             (config.head_dim == 128U || config.head_dim == 256U))
                ? 256ULL
                : 32ULL;
        linear(e, p + "self_attn.q_proj.weight", get("prepared"), q_fallback, q,
               h);
        linear(e, p + "self_attn.k_proj.weight", get("prepared"), nk_fallback,
               kv, h);
        linear(e, p + "self_attn.v_proj.weight", get("prepared"), nv_fallback,
               kv, h);
        rope(e, q_fallback, p + "self_attn.q_norm.weight", config.query_heads);
        rope(e, nk_fallback, p + "self_attn.k_norm.weight",
             config.key_value_heads);
        auto &a = e.emit(
            "dflash_forward", "omlx_m5_draft_attention",
            {.width = attn_tg_w, .height = U64(8) * config.query_heads},
            {.width = attn_tg_w},
            {q_fallback, nk_fallback, nv_fallback,
             get("cache_key." + std::to_string(layer)),
             get("cache_value." + std::to_string(layer)), get("attention")});
        e.dynamic(a, 6,
                  words({8, 0, config.query_heads, config.key_value_heads,
                         config.head_dim, config.sliding_window,
                         std::min(max_context, config.sliding_window), q, kv}),
                  0, 1);
      }
      linear(e, p + "self_attn.o_proj.weight", get("attention"),
             get("projected"), h, q);
      conv_add_norm(e, p + "attention_conv",
                    p + "post_attention_layernorm.weight", get("projected"),
                    get("hidden"), get("norm"));
      linear(e, p + "mlp_conv.kernel_projection.weight", get("norm"),
             get("dynamic"), 4 * h / config.conv_group_size, h);
      conv(e, p + "mlp_conv", get("norm"), get("prepared"), 0);
      const auto gw_v = weight(p + "mlp.gate_proj.weight");
      const auto uw_v = weight(p + "mlp.up_proj.weight");
      const auto dw_v = weight(p + "mlp.down_proj.weight");
      bool did_fused_mlp = false;
      if (linear_policy == LinearNumericPolicy::RowExactPortableV1 &&
          mxfp4_nax_active()) {
        const auto *gw = mxfp4_nax_cache->find(gw_v.object_id);
        const auto *uw = mxfp4_nax_cache->find(uw_v.object_id);
        const auto *dw = mxfp4_nax_cache->find(dw_v.object_id);
        if (gw && uw && dw && gw->bn == 16u && uw->bn == 16u &&
            gw->n == config.intermediate_size && gw->k == h &&
            uw->n == config.intermediate_size && uw->k == h &&
            gw->split_k == uw->split_k && dw->n == h &&
            dw->k == config.intermediate_size) {
          ensure_f16_cast(e, get("prepared"), f16_scratch, f16_row_scale, h);
          linear_mxfp4_nax_swiglu_direct_f16(e, *gw, *uw, f16_scratch,
                                             f16_row_scale, f16_mid, true);
          const std::string grp = "mxfp4.nax.f16_bk128.n16.sk" +
                                  std::to_string(gw->split_k) + ".v1";
          e.record_weight(gw_v, config.intermediate_size, h, "mxfp4_nax_m5_v1",
                          "mxfp4_nax_m5_v1", {},
                          "linear_row_exact_portable_v1", grp, true);
          e.record_weight(uw_v, config.intermediate_size, h, "mxfp4_nax_m5_v1",
                          "mxfp4_nax_m5_v1", {},
                          "linear_row_exact_portable_v1", grp, true);
          linear_mxfp4_nax_direct_f16(e, *dw, f16_mid, f16_ones_scale,
                                      get("projected"));
          e.record_weight(dw_v, h, config.intermediate_size, "mxfp4_nax_m5_v1",
                          "mxfp4_nax_m5_v1", {}, "linear_row_exact_portable_v1",
                          "mxfp4.nax.f16_bk128.n" + std::to_string(dw->bn) +
                              ".sk" + std::to_string(dw->split_k) + ".v1",
                          true);
          did_fused_mlp = true;
        }
      }
      if (!did_fused_mlp && bf16_nax_enabled &&
          bf16_nax_weight_eligible(gw_v, config.intermediate_size, h) &&
          bf16_nax_weight_eligible(uw_v, config.intermediate_size, h)) {
        linear_bf16_nax_swiglu(e, gw_v, uw_v, get("prepared"), get("product"),
                               config.intermediate_size, h, gpu_cores);
        linear(e, p + "mlp.down_proj.weight", get("product"), get("projected"),
               h, config.intermediate_size);
        did_fused_mlp = true;
      }
      if (!did_fused_mlp) {
        linear(e, p + "mlp.gate_proj.weight", get("prepared"), get("gate"),
               config.intermediate_size, h);
        linear(e, p + "mlp.up_proj.weight", get("prepared"), get("up"),
               config.intermediate_size, h);
        auto &activation = e.emit(
            "mlp", "omlx_m5_swiglu_a16",
            {.width = 32, .height = U64(8) * config.intermediate_size / 32},
            {.width = 32}, {get("gate"), get("up"), get("product")});
        e.dynamic(activation, 3,
                  words({8, config.intermediate_size,
                         config.intermediate_size / 32, 0}),
                  0);
        linear(e, p + "mlp.down_proj.weight", get("product"), get("projected"),
               h, config.intermediate_size);
      }
      const std::string next_norm_name =
          (layer + 1 < config.layer_count)
              ? ("layers." + std::to_string(layer + 1) +
                 ".input_layernorm.weight")
              : std::string("norm.weight");
      conv_add_norm(e, p + "mlp_conv", next_norm_name, get("projected"),
                    get("hidden"), get("norm"));
    }
    bool did_nax_head = false;
    if (!did_nax_head &&
        linear_policy == LinearNumericPolicy::RowExactPortableV1 &&
        mxfp4_nax_active()) {
      if (const auto *tw = mxfp4_nax_cache->find(head.object_id);
          tw && tw->n == config.vocabulary_size && tw->k == h) {
        linear_mxfp4_nax(e, *tw, get("norm"), get("logits"), f16_scratch,
                         f16_row_scale);
        e.record_weight(
            head, config.vocabulary_size, h, "mxfp4_nax_m5_v1",
            "mxfp4_nax_m5_v1", {}, "linear_row_exact_portable_v1",
            "mxfp4.nax.f16_bk128.n" + std::to_string(tw->bn) + ".sk" +
                std::to_string(tw->split_k) + ".v1",
            true);
        did_nax_head = true;
      }
    }
    if (!did_nax_head &&
        (linear_policy == LinearNumericPolicy::RowExactPortableV1 ||
         head.layout == "oq_nax_tiled_g64_v1" ||
         head.layout == "oq_nax_tiled_g128_v1") &&
        affine_nax_head_cache && affine_nax_head_cache->active()) {
      if (const auto *tw = affine_nax_head_cache->find(head.object_id);
          tw && tw->n == config.vocabulary_size && tw->k == h) {
        linear_affine_nax(e, *tw, get("norm"), affine_nax_xs, get("logits"));
        e.record_weight(
            head, config.vocabulary_size, h, "affine_nax_m5_v1",
            "affine_nax_m5_v1", {}, "linear_row_exact_portable_v1",
            "oq.nax.q" + std::to_string(tw->bits) + "_g64.n" +
                std::to_string(tw->bn) + ".sk" + std::to_string(tw->split_k) +
                ".v1",
            true);
        did_nax_head = true;
      }
    }
    if (!did_nax_head) {
      const auto head_route = resolve_linear_route(
          head, config.vocabulary_size, h, e.profile.capacity,
          context.capabilities().device_name, m4_short_routes, {},
          linear_policy, context.capabilities().profile.probe.apple_gpu_family);
      detail::forward::linear(e, head, get("norm"), get("logits"),
                              config.vocabulary_size, h, get("sums"),
                              get("zero"), head_route.route);
      e.record_weight(head, config.vocabulary_size, h, head_route.route.family,
                      head_route.source, head_route.fallback_reason,
                      head_route.route.numeric_contract_id,
                      head_route.route.row_exact_group,
                      head_route.route.row_exact);
    }
    if (!gpu_selector)
      return 0;
    const U32 blocks =
        (U64(config.vocabulary_size) + selector_chunk - 1) / selector_chunk;
    const U32 groups = (U64(blocks) + 31) / 32;
    const SelectorParams selector_params{.vocab = config.vocabulary_size,
                                         .hidden = h,
                                         .rank = config.selector_rank,
                                         .top_k = config.selector_top_k,
                                         .blocks = blocks,
                                         .groups = groups,
                                         .width = 8};
    const auto bytes = std::as_bytes(std::span(&selector_params, 1));
    const std::vector<std::byte> params(bytes.begin(), bytes.end());
    // Emit selector_projection immediately after lm_head: it reads `norm`
    // (not `logits`), overlapping concurrently behind lm_head with 0 barrier!
    e.begin_stage("selector_projection");
    const auto proj_w = weight("candidate_selector.hidden_projection.weight");
    if (selector_nax_f32_enabled && proj_w.format == "direct_bf16_v1" &&
        proj_w.row_count == config.selector_rank && proj_w.in_features == h &&
        (proj_w.padded_k == 0u || proj_w.padded_k == h)) {
      const U32 bk = (h % 128u) == 0u ? 128u : 64u;
      const U32 padded_n = ((config.selector_rank + 15u) / 16u) * 16u;
      const std::string fn = bk == 128u
                                 ? "omlx_m5_bf16_nax_f32_bk128_n16_sk4"
                                 : "omlx_m5_bf16_nax_f32_bk64_n16_sk4";
      auto &project = e.emit(
          "attention_nax_prefill", fn,
          {.width = U64(padded_n / 16u) * 128u, .height = 1u}, {.width = 128u},
          {get("norm"), plane(proj_w, "data"), get("selector_projected")});
      e.dynamic(project, 3, words({config.selector_rank, h, 8u, 8u}), 3u);
    } else {
      auto &project = e.emit(
          "dflash_forward", "omlx_m5_selector_project_hidden",
          {.width = 32, .height = config.selector_rank}, {.width = 32},
          {get("norm"), data("candidate_selector.hidden_projection.weight"),
           get("selector_projected")});
      e.dynamic(project, 3, params);
    }
    e.begin_stage("selector_topk_chunks");
    auto &stage1 = e.emit("dflash_forward", "omlx_m5_selector_topk16_stage1",
                          {.width = selector_stage1_chunks * 256u, .height = 7u},
                          {.width = 256u},
                          {get("logits"), get("selector_partial_ids"),
                           get("selector_partial_scores")});
    e.profile.sampling_offset = e.dynamic(stage1, 3, params);
    e.begin_stage("selector_topk_merge");
    auto &stage2 = e.emit(
        "dflash_forward", "omlx_m5_selector_topk16_stage2",
        {.width = selector_stage1_chunks, .height = 7u},
        {.width = selector_stage1_chunks},
        {get("selector_partial_ids"), get("selector_partial_scores"),
         get("selector_ids"), get("selector_scores")});
    e.dynamic(stage2, 4, params);
    e.begin_stage("selector_finish");
    auto &select = e.emit(
        "dflash_forward",
        sampling_contract == SamplingContract::PositionCoupledV1
            ? "omlx_m5_selector_finish_position"
            : "omlx_m5_selector_finish",
        {.width = 1024}, {.width = 1024},
        {get("selector_ids"), get("selector_scores"), get("selector_projected"),
         data("candidate_selector.predecessor_codebook"),
         data("candidate_selector.successor_codebook"), get("selector_tokens"),
         get("selector_probabilities"), get("selector_fallback")});
    e.dynamic(select, 8, params);
    U64 keyed_off = 0;
    if (sampling_contract == SamplingContract::PositionCoupledV1)
      keyed_off = e.dynamic(select, 9, std::vector<std::byte>(24));
    return keyed_off;
  }
  Profile compile(bool gpu_selector, U64 &out_keyed_offset) {
    Encoder e(
        context, directory, 8,
        std::string(gpu_selector ? "dflash.forward.gpu"
                                 : "dflash.forward.cpu") +
            (sampling_contract == SamplingContract::PositionCoupledV1
                 ? ".position-coupled-v1"
                 : "") +
            (linear_policy == LinearNumericPolicy::RowExactPortableV1
                 ? ".linear-row-exact-v1"
                 : "") +
            plan_key_suffix());
    out_keyed_offset = encode_backbone_body(e, gpu_selector);
    return e.finish();
  }
  ContextProfile compile_context(U32 capacity) {
    Encoder e(
        context, directory, capacity,
        std::string("dflash.context") +
            (sampling_contract == SamplingContract::PositionCoupledV1
                 ? ".position-coupled-v1"
                 : "") +
            (linear_policy == LinearNumericPolicy::RowExactPortableV1
                 ? ".linear-row-exact-v1"
                 : "") +
            plan_key_suffix());
    const U64 gather_ring_off = encode_context_body(e, capacity);
    return {.profile = e.finish(), .gather_ring_offset = gather_ring_off};
  }
  FusedProfile compile_fused(U32 ctx_capacity, bool gpu_selector) {
    Encoder e(
        context, directory, ctx_capacity,
        std::string(gpu_selector ? "dflash.fused.gpu" : "dflash.fused.cpu") +
            ".ctx" + std::to_string(ctx_capacity) +
            (sampling_contract == SamplingContract::PositionCoupledV1
                 ? ".position-coupled-v1"
                 : "") +
            (linear_policy == LinearNumericPolicy::RowExactPortableV1
                 ? ".linear-row-exact-v1"
                 : "") +
            plan_key_suffix());
    const U64 gather_ring_off = encode_context_body(e, ctx_capacity);
    const std::size_t ctx_patches = e.profile.patches.size();
    e.clear_dynamic_parameter_cache();
    e.profile.capacity = 8;
    const U64 keyed_off = encode_backbone_body(e, gpu_selector);
    return {.profile = e.finish(),
            .context_capacity = ctx_capacity,
            .context_patch_count = ctx_patches,
            .gather_ring_offset = gather_ring_off,
            .keyed_sampling_offset = keyed_off};
  }
  std::pair<MetalSubmission, U32> update_context(U32 count, U32 position,
                                                 bool use_icb) {
    auto found = std::find_if(context_profiles.begin(), context_profiles.end(),
                              [count](const ContextProfile &p) {
                                return count <= p.profile.capacity;
                              });
    if (found == context_profiles.end())
      throw std::out_of_range("DFlash context update capacity");
    auto &selected = found->profile;
    for (const auto &patch : selected.patches) {
      const U32 value = patch.position ? position : count * patch.multiplier;
      std::memcpy(selected.parameter_data.data() + patch.offset +
                      patch.word * 4,
                  &value, 4);
    }
    std::memcpy(selected.parameter_data.data() + found->gather_ring_offset,
                &feature_ring_rows, 4);
    selected.parameters.write("step", selected.parameter_data);
    if (use_icb && selected.icb)
      return {context.replay_icb(*selected.icb), selected.capacity};
    const std::array calls = {
        MetalCommandInvocation{.plan = &selected.plan,
                               .bindings = selected.bindings,
                               .parameters = &selected.parameters}};
    return {context.execute_batch(calls), selected.capacity};
  }
  struct PackedBatchKey {
    std::vector<const Impl *> slots;
    bool has_ctx = false;
    bool cpu_selector = false;
    bool operator==(const PackedBatchKey &) const = default;
  };
  struct PackedDraftProfile {
    PackedBatchKey key;
    Profile profile;
    std::size_t ctx_patch_count = 0;
    std::vector<U64> gather_offsets;
    std::vector<std::vector<U64>> ctx_append_offsets;
    std::vector<U64> conv_offsets;
    std::vector<U64> conv_add_norm_offsets;
    std::vector<U64> qk_rope_offsets;
    std::vector<U64> rope_q_offsets;
    std::vector<U64> rope_k_offsets;
    std::vector<std::vector<U64>> attn_offsets;
    std::vector<U64> selector_params_offsets;
    U64 selector_keys_offset = 0;
  };

  Ref pb_get(const std::string &name) const { return pb_refs.at(name); }

  void ensure_packed_workspace(U32 min_b) {
    if (pb_max_b >= min_b)
      return;
    packed_cache_.clear();
    pb_refs.clear();
    pb_buffers.clear();
    const U32 alloc_b = (min_b <= 4U) ? 4U : 8U;
    const U64 rows = U64(alloc_b) * 8ULL;
    const auto add_pb = [&](std::string name, U64 r, U64 cols, U64 elem = 2ULL) {
      const U64 bytes = std::max<U64>(16ULL, mul(mul(r, cols), elem));
      auto buf = std::make_unique<MetalBuffer>(context.make_buffer(bytes));
      buf->zero();
      pb_refs.emplace(std::move(name), Ref{buf.get(), 0, bytes});
      pb_buffers.push_back(std::move(buf));
    };
    const U64 feat_w = U64(h) * U64(config.target_layer_ids.size());
    add_pb("pb_tokens", rows, 1, 4);
    add_pb("pb_hidden", rows, h);
    add_pb("pb_norm", rows, h);
    add_pb("pb_dynamic", rows, 4ULL * h / config.conv_group_size);
    add_pb("pb_prepared", rows, h);
    add_pb("pb_qkv", rows, U64(q) + 2ULL * U64(kv));
    add_pb("pb_attention", rows, q);
    add_pb("pb_projected", rows, h);
    add_pb("pb_gate", rows, config.intermediate_size);
    add_pb("pb_up", rows, config.intermediate_size);
    add_pb("pb_product", rows, config.intermediate_size);
    add_pb("pb_logits", rows, config.vocabulary_size);
    add_pb("pb_selector_partial_ids", rows * selector_stage1_chunks,
           config.selector_top_k, 4);
    add_pb("pb_selector_partial_scores", rows * selector_stage1_chunks,
           config.selector_top_k, 4);
    add_pb("pb_selector_ids", rows, config.selector_top_k, 4);
    add_pb("pb_selector_scores", rows, config.selector_top_k, 4);
    add_pb("pb_selector_projected", rows, config.selector_rank, 4);
    add_pb("pb_selector_tokens", rows, 1, 4);
    add_pb("pb_selector_probabilities", rows, config.selector_top_k, 4);
    add_pb("pb_selector_fallback", alloc_b, 1, 4);
    add_pb("pb_sums", rows,
           (std::max({U64(h), U64(q), U64(config.intermediate_size), feat_w}) +
            31ULL) /
               32ULL,
           4);
    pb_max_b = alloc_b;
  }

  PackedDraftProfile compile_packed_batch(const PackedBatchKey &skey) {
    const U32 B = U32(skey.slots.size());
    const U32 total_rows = 8U * B;
    Fnv1a64 kh;
    kh.u64(B);
    kh.u64(skey.has_ctx ? 1ULL : 0ULL);
    kh.u64(skey.cpu_selector ? 1ULL : 0ULL);
    for (const auto *sp : skey.slots)
      kh.u64(sp->owner_id);

    Encoder e(context, directory, total_rows,
              "dflash.packed." + kh.hex() +
                  (skey.has_ctx ? ".ctx" : ".noctx") +
                  (skey.cpu_selector ? ".cpu" : ".gpu") +
                  (sampling_contract == SamplingContract::PositionCoupledV1
                       ? ".position-coupled-v1"
                       : "") +
                  (linear_policy == LinearNumericPolicy::RowExactPortableV1
                       ? ".linear-row-exact-v1"
                       : "") +
                  plan_key_suffix());

    PackedDraftProfile out;
    out.key = skey;
    const Ref sums_ref = pb_get("pb_sums");

    // Stage A: Packed context projection across all B slots (sum(rem_b) <= 8*B).
    if (skey.has_ctx) {
      e.begin_stage("packed_draft_context_projection");
      const U32 feature_width = h * U32(config.target_layer_ids.size());
      if (weight("fc.weight").format != "direct_bf16_v1")
        throw std::invalid_argument(
            "DFlash incremental FC requires BF16 weights");
      out.gather_offsets.resize(B);
      for (U32 b = 0; b < B; ++b) {
        const auto &sp = *skey.slots[b];
        const ContextGatherOffsetHost gp{.tokens = 0U,
                                         .width = feature_width,
                                         .position = 0xFF000000U | b,
                                         .ring_rows = sp.feature_ring_rows,
                                         .dst_row_offset = 0U};
        if ((feature_width & 7u) == 0u) {
          const U32 vec_width = feature_width >> 3u;
          const U32 grid_x = ((vec_width + 127u) / 128u) * 128u;
          auto &gather = e.emit("dflash_forward",
                                "omlx_m5_draft_context_gather_vec8_offset",
                                {.width = grid_x, .height = 8U}, {.width = 128},
                                {sp.features, get("context_input")});
          out.gather_offsets[b] = e.dynamic(gather, 2, pod_bytes(gp));
        } else {
          auto &gather =
              e.emit("dflash_forward", "omlx_m5_draft_context_gather_offset",
                     {.width = U64(8U) * feature_width}, {.width = 128},
                     {sp.features, get("context_input")});
          out.gather_offsets[b] = e.dynamic(gather, 2, pod_bytes(gp));
        }
      }
      linear_with_sums(e, "fc.weight", get("context_input"), get("context_raw"),
                       h, feature_width, sums_ref);
      norm(e, "hidden_norm.weight", get("context_raw"), get("context_norm"));

      const U32 ctx_buf_rows = std::min(max_context, 1024U);
      const U64 ctx_kv_slice_bytes = mul(mul(ctx_buf_rows, kv), 2ULL);
      const Ref ctx_kv = get("context_kv");
      const Ref ctx_k_fallback{ctx_kv.buffer, ctx_kv.offset, ctx_kv_slice_bytes};
      const Ref ctx_v_fallback{ctx_kv.buffer, ctx_kv.offset + ctx_kv_slice_bytes,
                               ctx_kv_slice_bytes};
      const U32 ring_cap = std::min(max_context, config.sliding_window);
      out.ctx_append_offsets.assign(config.layer_count, std::vector<U64>(B, 0));

      for (U32 layer = 0; layer < config.layer_count; ++layer) {
        const auto prefix = "layers." + std::to_string(layer) + ".";
        const auto kw = weight(prefix + "self_attn.k_proj.weight");
        const auto vw = weight(prefix + "self_attn.v_proj.weight");
        if (kw.format != "direct_bf16_v1" || vw.format != "direct_bf16_v1")
          throw std::invalid_argument(
              "DFlash incremental KV requires BF16 weights");
        const auto *qkv_entry =
            (dflash_qkv_cache ? dflash_qkv_cache->find(layer) : nullptr);
        const bool fused_kv =
            qkv_entry && (!mxfp4_nax_active() ||
                          (!mxfp4_nax_cache->find(kw.object_id) &&
                           !mxfp4_nax_cache->find(vw.object_id)));
        if (fused_kv) {
          emit_bf16_nax_fused(e, qkv_entry->kv_data, {{kw, kv}, {vw, kv}},
                              get("context_norm"), ctx_kv, 2u * kv, h);
        } else {
          linear_with_sums(e, prefix + "self_attn.k_proj.weight",
                           get("context_norm"), ctx_k_fallback, kv, h, sums_ref);
          linear_with_sums(e, prefix + "self_attn.v_proj.weight",
                           get("context_norm"), ctx_v_fallback, kv, h, sums_ref);
        }
        for (U32 b = 0; b < B; ++b) {
          const auto &sp = *skey.slots[b];
          const ContextRopeAppendOffsetHost ap{
              .tokens = 0U,
              .kv_heads = config.key_value_heads,
              .dim = config.head_dim,
              .position = 0xFF000000U | (layer << 8U) | b,
              .capacity = ring_cap,
              .epsilon = config.epsilon,
              .theta = config.rope_theta,
              .src_row_offset = 0U};
          if (fused_kv) {
            auto &ra = e.emit(
                "dflash_forward",
                "omlx_m5_draft_context_rope_cache_append_offset",
                {.width = 32, .height = U64(8U) * config.key_value_heads},
                {.width = 32},
                {ctx_kv, data(prefix + "self_attn.k_norm.weight"),
                 sp.get("cache_key." + std::to_string(layer)),
                 sp.get("cache_value." + std::to_string(layer))});
            out.ctx_append_offsets[layer][b] = e.dynamic(ra, 4, pod_bytes(ap));
          } else {
            auto &ra = e.emit(
                "dflash_forward",
                "omlx_m5_draft_context_rope_cache_append_split_offset",
                {.width = 32, .height = U64(8U) * config.key_value_heads},
                {.width = 32},
                {ctx_k_fallback, ctx_v_fallback,
                 data(prefix + "self_attn.k_norm.weight"),
                 sp.get("cache_key." + std::to_string(layer)),
                 sp.get("cache_value." + std::to_string(layer))});
            out.ctx_append_offsets[layer][b] = e.dynamic(ra, 5, pod_bytes(ap));
          }
        }
      }
      out.ctx_patch_count = e.profile.patches.size();
      e.clear_dynamic_parameter_cache();
    }

    // Stage B: Packed [8*B, H] 5-layer backbone + shared target lm_head + selector.
    e.profile.capacity = total_rows;
    e.begin_stage("packed_draft_backbone");
    embedding_lookup(e, embedding, pb_get("pb_tokens"), pb_get("pb_hidden"),
                     get("zero"), config.vocabulary_size, h);
    if (config.layer_count > 0) {
      norm(e, "layers.0.input_layernorm.weight", pb_get("pb_hidden"),
           pb_get("pb_norm"));
    } else {
      norm(e, "norm.weight", pb_get("pb_hidden"), pb_get("pb_norm"));
    }

    const Ref qkv_buf = pb_get("pb_qkv");
    const U64 q_slice_bytes = mul(U64(total_rows), mul(q, 2ULL));
    const U64 kv_slice_bytes = mul(U64(total_rows), mul(kv, 2ULL));
    const Ref q_fallback{qkv_buf.buffer, qkv_buf.offset, q_slice_bytes};
    const Ref nk_fallback{qkv_buf.buffer, qkv_buf.offset + q_slice_bytes,
                          kv_slice_bytes};
    const Ref nv_fallback{qkv_buf.buffer,
                          qkv_buf.offset + q_slice_bytes + kv_slice_bytes,
                          kv_slice_bytes};
    const BatchedConvParamsHost conv_init{
        .batch_size = B,
        .hidden = h,
        .group_size = config.conv_group_size,
        .output_half = 0U,
        .slot_widths = {8, 8, 8, 8, 8, 8, 8, 8}};
    const BatchedConvAddNormParamsHost conv_an_init{
        .batch_size = B,
        .hidden = h,
        .group_size = config.conv_group_size,
        .epsilon = config.epsilon,
        .slot_widths = {8, 8, 8, 8, 8, 8, 8, 8}};

    auto emit_conv_batched = [&](const std::string &name, Ref x, Ref dst) {
      if ((h & 3u) == 0u && (config.conv_group_size & 3u) == 0u) {
        const U32 vec_cols = h >> 2u;
        const U32 grid_x = ((vec_cols + 127u) / 128u) * 128u;
        auto &d = e.emit("dflash_forward", "omlx_m5_draft_conv_vec4_batched",
                         {.width = grid_x, .height = total_rows},
                         {.width = 128},
                         {x, pb_get("pb_dynamic"),
                          data(name + ".base_kernel"), dst});
        push_unique_offset(out.conv_offsets,
                           e.dynamic(d, 4, pod_bytes(conv_init)));
      } else {
        auto &d = e.emit("dflash_forward", "omlx_m5_draft_conv_batched",
                         {.width = h, .height = total_rows}, {.width = 128},
                         {x, pb_get("pb_dynamic"),
                          data(name + ".base_kernel"), dst});
        push_unique_offset(out.conv_offsets,
                           e.dynamic(d, 4, pod_bytes(conv_init)));
      }
    };
    auto emit_conv_add_norm_batched = [&](const std::string &conv_name,
                                          const std::string &norm_name) {
      auto &d = e.emit(
          "dflash_forward", "omlx_m5_draft_conv_add_rmsnorm_batched",
          {.width = 256, .height = total_rows}, {.width = 256},
          {pb_get("pb_projected"), pb_get("pb_dynamic"),
           data(conv_name + ".base_kernel"), pb_get("pb_hidden"),
           data(norm_name), pb_get("pb_norm")});
      push_unique_offset(out.conv_add_norm_offsets,
                         e.dynamic(d, 6, pod_bytes(conv_an_init)));
    };

    out.attn_offsets.assign(config.layer_count, std::vector<U64>(B, 0));
    for (U32 layer = 0; layer < config.layer_count; ++layer) {
      const auto p = "layers." + std::to_string(layer) + ".";
      linear_with_sums(e, p + "attention_conv.kernel_projection.weight",
                       pb_get("pb_norm"), pb_get("pb_dynamic"),
                       4 * h / config.conv_group_size, h, sums_ref);
      emit_conv_batched(p + "attention_conv", pb_get("pb_norm"),
                        pb_get("pb_prepared"));

      const auto qw_v = weight(p + "self_attn.q_proj.weight");
      const auto kw_v = weight(p + "self_attn.k_proj.weight");
      const auto vw_v = weight(p + "self_attn.v_proj.weight");
      if (const auto *qkv_entry =
              (dflash_qkv_cache ? dflash_qkv_cache->find(layer) : nullptr);
          qkv_entry && (!mxfp4_nax_active() ||
                        (!mxfp4_nax_cache->find(qw_v.object_id) &&
                         !mxfp4_nax_cache->find(kw_v.object_id) &&
                         !mxfp4_nax_cache->find(vw_v.object_id)))) {
        const U32 qkv_stride = q + 2u * kv;
        emit_bf16_nax_fused(e, qkv_entry->qkv_data,
                            {{qw_v, q}, {kw_v, kv}, {vw_v, kv}},
                            pb_get("pb_prepared"), qkv_buf, qkv_stride, h);
        const U64 total_qk_heads =
            U64(total_rows) * (config.query_heads + config.key_value_heads);
        const U64 qk_rope_tgs = (total_qk_heads + 3ULL) / 4ULL;
        const BatchedQkRopeStridedParamsHost qkr_init{
            .batch_size = B,
            .q_heads = config.query_heads,
            .kv_heads = config.key_value_heads,
            .dim = config.head_dim,
            .qkv_stride = qkv_stride,
            .epsilon = config.epsilon,
            .theta = config.rope_theta,
            .pad = 0U,
            .slot_widths = {8, 8, 8, 8, 8, 8, 8, 8},
            .slot_positions = {0, 0, 0, 0, 0, 0, 0, 0}};
        auto &qk_rope = e.emit(
            "dflash_forward", "omlx_m5_draft_qk_rope_strided_batched",
            {.width = 128, .height = qk_rope_tgs}, {.width = 128},
            {qkv_buf, data(p + "self_attn.q_norm.weight"),
             data(p + "self_attn.k_norm.weight")});
        push_unique_offset(out.qk_rope_offsets,
                           e.dynamic(qk_rope, 3, pod_bytes(qkr_init)));
        const U64 attn_tg_w =
            (config.sliding_window > 64U &&
             (config.head_dim == 128U || config.head_dim == 256U))
                ? 256ULL
                : 32ULL;
        for (U32 b = 0; b < B; ++b) {
          const auto &sp = *skey.slots[b];
          const Ref slot_q = df_row_slice_ref(qkv_buf, 8U * b, 8U, qkv_stride);
          const Ref slot_nk = sub(slot_q, mul(q, 2ULL));
          const Ref slot_nv = sub(slot_q, mul(q + kv, 2ULL));
          const Ref slot_out =
              df_row_slice_ref(pb_get("pb_attention"), 8U * b, 8U, q);
          auto &a = e.emit(
              "dflash_forward", "omlx_m5_draft_attention",
              {.width = attn_tg_w, .height = U64(8U) * config.query_heads},
              {.width = attn_tg_w},
              {slot_q, slot_nk, slot_nv,
               sp.get("cache_key." + std::to_string(layer)),
               sp.get("cache_value." + std::to_string(layer)), slot_out});
          out.attn_offsets[layer][b] = e.dynamic(
              a, 6,
              words({8U, 0xFF000000U | (layer << 8U) | b, config.query_heads,
                     config.key_value_heads, config.head_dim,
                     config.sliding_window,
                     std::min(max_context, config.sliding_window), qkv_stride,
                     qkv_stride}));
        }
      } else {
        linear_with_sums(e, p + "self_attn.q_proj.weight",
                         pb_get("pb_prepared"), q_fallback, q, h, sums_ref);
        linear_with_sums(e, p + "self_attn.k_proj.weight",
                         pb_get("pb_prepared"), nk_fallback, kv, h, sums_ref);
        linear_with_sums(e, p + "self_attn.v_proj.weight",
                         pb_get("pb_prepared"), nv_fallback, kv, h, sums_ref);
        const BatchedRopeParamsHost rq_init{
            .batch_size = B,
            .heads = config.query_heads,
            .dim = config.head_dim,
            .pad = 0U,
            .epsilon = config.epsilon,
            .theta = config.rope_theta,
            .slot_widths = {8, 8, 8, 8, 8, 8, 8, 8},
            .slot_positions = {0, 0, 0, 0, 0, 0, 0, 0}};
        auto &rq = e.emit(
            "dflash_forward", "omlx_m5_draft_rope_batched",
            {.width = 32, .height = U64(total_rows) * config.query_heads},
            {.width = 32}, {q_fallback, data(p + "self_attn.q_norm.weight")});
        push_unique_offset(out.rope_q_offsets,
                           e.dynamic(rq, 2, pod_bytes(rq_init)));
        BatchedRopeParamsHost rk_init = rq_init;
        rk_init.heads = config.key_value_heads;
        auto &rk = e.emit(
            "dflash_forward", "omlx_m5_draft_rope_batched",
            {.width = 32, .height = U64(total_rows) * config.key_value_heads},
            {.width = 32}, {nk_fallback, data(p + "self_attn.k_norm.weight")});
        push_unique_offset(out.rope_k_offsets,
                           e.dynamic(rk, 2, pod_bytes(rk_init)));
        const U64 attn_tg_w =
            (config.sliding_window > 64U &&
             (config.head_dim == 128U || config.head_dim == 256U))
                ? 256ULL
                : 32ULL;
        for (U32 b = 0; b < B; ++b) {
          const auto &sp = *skey.slots[b];
          auto &a = e.emit(
              "dflash_forward", "omlx_m5_draft_attention",
              {.width = attn_tg_w, .height = U64(8U) * config.query_heads},
              {.width = attn_tg_w},
              {df_row_slice_ref(q_fallback, 8U * b, 8U, q),
               df_row_slice_ref(nk_fallback, 8U * b, 8U, kv),
               df_row_slice_ref(nv_fallback, 8U * b, 8U, kv),
               sp.get("cache_key." + std::to_string(layer)),
               sp.get("cache_value." + std::to_string(layer)),
               df_row_slice_ref(pb_get("pb_attention"), 8U * b, 8U, q)});
          out.attn_offsets[layer][b] = e.dynamic(
              a, 6,
              words({8U, 0xFF000000U | (layer << 8U) | b, config.query_heads,
                     config.key_value_heads, config.head_dim,
                     config.sliding_window,
                     std::min(max_context, config.sliding_window), q, kv}));
        }
      }
      linear_with_sums(e, p + "self_attn.o_proj.weight", pb_get("pb_attention"),
                       pb_get("pb_projected"), h, q, sums_ref);
      emit_conv_add_norm_batched(p + "attention_conv",
                                 p + "post_attention_layernorm.weight");
      linear_with_sums(e, p + "mlp_conv.kernel_projection.weight",
                       pb_get("pb_norm"), pb_get("pb_dynamic"),
                       4 * h / config.conv_group_size, h, sums_ref);
      emit_conv_batched(p + "mlp_conv", pb_get("pb_norm"),
                        pb_get("pb_prepared"));

      const auto gw_v = weight(p + "mlp.gate_proj.weight");
      const auto uw_v = weight(p + "mlp.up_proj.weight");
      const auto dw_v = weight(p + "mlp.down_proj.weight");
      bool did_fused_mlp = false;
      if (linear_policy == LinearNumericPolicy::RowExactPortableV1 &&
          mxfp4_nax_active()) {
        const auto *gw = mxfp4_nax_cache->find(gw_v.object_id);
        const auto *uw = mxfp4_nax_cache->find(uw_v.object_id);
        const auto *dw = mxfp4_nax_cache->find(dw_v.object_id);
        if (gw && uw && dw && gw->bn == 16u && uw->bn == 16u &&
            gw->n == config.intermediate_size && gw->k == h &&
            uw->n == config.intermediate_size && uw->k == h &&
            gw->split_k == uw->split_k && dw->n == h &&
            dw->k == config.intermediate_size) {
          ensure_f16_cast(e, pb_get("pb_prepared"), f16_scratch, f16_row_scale,
                          h);
          linear_mxfp4_nax_swiglu_direct_f16(e, *gw, *uw, f16_scratch,
                                             f16_row_scale, f16_mid, true);
          const std::string grp = "mxfp4.nax.f16_bk128.n16.sk" +
                                  std::to_string(gw->split_k) + ".v1";
          e.record_weight(gw_v, config.intermediate_size, h, "mxfp4_nax_m5_v1",
                          "mxfp4_nax_m5_v1", {}, "linear_row_exact_portable_v1",
                          grp, true);
          e.record_weight(uw_v, config.intermediate_size, h, "mxfp4_nax_m5_v1",
                          "mxfp4_nax_m5_v1", {}, "linear_row_exact_portable_v1",
                          grp, true);
          linear_mxfp4_nax_direct_f16(e, *dw, f16_mid, f16_ones_scale,
                                      pb_get("pb_projected"));
          e.record_weight(dw_v, h, config.intermediate_size, "mxfp4_nax_m5_v1",
                          "mxfp4_nax_m5_v1", {}, "linear_row_exact_portable_v1",
                          "mxfp4.nax.f16_bk128.n" + std::to_string(dw->bn) +
                              ".sk" + std::to_string(dw->split_k) + ".v1",
                          true);
          did_fused_mlp = true;
        }
      }
      if (!did_fused_mlp && bf16_nax_enabled &&
          bf16_nax_weight_eligible(gw_v, config.intermediate_size, h) &&
          bf16_nax_weight_eligible(uw_v, config.intermediate_size, h)) {
        linear_bf16_nax_swiglu(e, gw_v, uw_v, pb_get("pb_prepared"),
                               pb_get("pb_product"), config.intermediate_size, h,
                               gpu_cores);
        linear_with_sums(e, p + "mlp.down_proj.weight", pb_get("pb_product"),
                         pb_get("pb_projected"), h, config.intermediate_size,
                         sums_ref);
        did_fused_mlp = true;
      }
      if (!did_fused_mlp) {
        linear_with_sums(e, p + "mlp.gate_proj.weight", pb_get("pb_prepared"),
                         pb_get("pb_gate"), config.intermediate_size, h,
                         sums_ref);
        linear_with_sums(e, p + "mlp.up_proj.weight", pb_get("pb_prepared"),
                         pb_get("pb_up"), config.intermediate_size, h, sums_ref);
        auto &activation = e.emit(
            "mlp", "omlx_m5_swiglu_a16",
            {.width = 32,
             .height = U64(total_rows) * config.intermediate_size / 32},
            {.width = 32},
            {pb_get("pb_gate"), pb_get("pb_up"), pb_get("pb_product")});
        e.dynamic(activation, 3,
                  words({total_rows, config.intermediate_size,
                         config.intermediate_size / 32, 0}));
        linear_with_sums(e, p + "mlp.down_proj.weight", pb_get("pb_product"),
                         pb_get("pb_projected"), h, config.intermediate_size,
                         sums_ref);
      }
      const std::string next_norm_name =
          (layer + 1 < config.layer_count)
              ? ("layers." + std::to_string(layer + 1) +
                 ".input_layernorm.weight")
              : std::string("norm.weight");
      emit_conv_add_norm_batched(p + "mlp_conv", next_norm_name);
    }

    // Shared target lm_head over all [8*B, H] rows in ONE weight read!
    bool did_nax_head = false;
    if (!did_nax_head &&
        linear_policy == LinearNumericPolicy::RowExactPortableV1 &&
        mxfp4_nax_active()) {
      if (const auto *tw = mxfp4_nax_cache->find(head.object_id);
          tw && tw->n == config.vocabulary_size && tw->k == h) {
        linear_mxfp4_nax(e, *tw, pb_get("pb_norm"), pb_get("pb_logits"),
                         f16_scratch, f16_row_scale, total_rows, gpu_cores);
        e.record_weight(
            head, config.vocabulary_size, h, "mxfp4_nax_m5_v1",
            "mxfp4_nax_m5_v1", {}, "linear_row_exact_portable_v1",
            "mxfp4.nax.f16_bk128.n" + std::to_string(tw->bn) + ".sk" +
                std::to_string(tw->split_k) + ".v1",
            true, total_rows);
        did_nax_head = true;
      }
    }
    if (!did_nax_head &&
        (linear_policy == LinearNumericPolicy::RowExactPortableV1 ||
         head.layout == "oq_nax_tiled_g64_v1" ||
         head.layout == "oq_nax_tiled_g128_v1") &&
        affine_nax_head_cache && affine_nax_head_cache->active()) {
      if (const auto *tw = affine_nax_head_cache->find(head.object_id);
          tw && tw->n == config.vocabulary_size && tw->k == h) {
        linear_affine_nax(e, *tw, pb_get("pb_norm"), affine_nax_xs,
                          pb_get("pb_logits"), total_rows, gpu_cores);
        e.record_weight(
            head, config.vocabulary_size, h, "affine_nax_m5_v1",
            "affine_nax_m5_v1", {}, "linear_row_exact_portable_v1",
            "oq.nax.q" + std::to_string(tw->bits) + "_g64.n" +
                std::to_string(tw->bn) + ".sk" + std::to_string(tw->split_k) +
                ".v1",
            true, total_rows);
        did_nax_head = true;
      }
    }
    if (!did_nax_head) {
      const auto head_route = resolve_linear_route(
          head, config.vocabulary_size, h, total_rows,
          context.capabilities().device_name, m4_short_routes, {},
          linear_policy, context.capabilities().profile.probe.apple_gpu_family);
      detail::forward::linear(e, head, pb_get("pb_norm"), pb_get("pb_logits"),
                              config.vocabulary_size, h, sums_ref, get("zero"),
                              head_route.route, total_rows);
      e.record_weight(head, config.vocabulary_size, h, head_route.route.family,
                      head_route.source, head_route.fallback_reason,
                      head_route.route.numeric_contract_id,
                      head_route.route.row_exact_group,
                      head_route.route.row_exact, total_rows);
    }

    if (!skey.cpu_selector) {
      const U32 blocks =
          (U64(config.vocabulary_size) + selector_chunk - 1) / selector_chunk;
      const U32 groups = (U64(blocks) + 31) / 32;
      std::vector<SelectorParams> sp_init(
          B, SelectorParams{.vocab = config.vocabulary_size,
                            .hidden = h,
                            .rank = config.selector_rank,
                            .top_k = config.selector_top_k,
                            .blocks = blocks,
                            .groups = groups,
                            .width = 8});
      const auto sp_bytes = pod_span_bytes(std::span<const SelectorParams>(sp_init));

      e.begin_stage("packed_selector_projection");
      const auto proj_w = weight("candidate_selector.hidden_projection.weight");
      if (selector_nax_f32_enabled && proj_w.format == "direct_bf16_v1" &&
          proj_w.row_count == config.selector_rank && proj_w.in_features == h &&
          (proj_w.padded_k == 0u || proj_w.padded_k == h)) {
        const U32 bk = (h % 128u) == 0u ? 128u : 64u;
        const U32 padded_n = ((config.selector_rank + 15u) / 16u) * 16u;
        const std::string fn = bk == 128u
                                   ? "omlx_m5_bf16_nax_f32_bk128_n16_sk4"
                                   : "omlx_m5_bf16_nax_f32_bk64_n16_sk4";
        auto &project = e.emit(
            "attention_nax_prefill", fn,
            {.width = U64(padded_n / 16u) * 128u, .height = B}, {.width = 128u},
            {pb_get("pb_norm"), plane(proj_w, "data"),
             pb_get("pb_selector_projected")});
        e.dynamic(project, 3,
                  words({config.selector_rank, h, total_rows, total_rows}));
      } else {
        auto &project = e.emit(
            "dflash_forward", "omlx_m5_selector_project_hidden_batched",
            {.width = 32, .height = config.selector_rank, .depth = B},
            {.width = 32, .height = 1, .depth = 1},
            {pb_get("pb_norm"),
             data("candidate_selector.hidden_projection.weight"),
             pb_get("pb_selector_projected")});
        push_unique_offset(out.selector_params_offsets,
                           e.dynamic(project, 3, sp_bytes));
      }

      e.begin_stage("packed_selector_topk_chunks");
      auto &stage1 = e.emit(
          "dflash_forward", "omlx_m5_selector_topk16_stage1_batched",
          {.width = selector_stage1_chunks * 256u, .height = 7u * B},
          {.width = 256u},
          {pb_get("pb_logits"), pb_get("pb_selector_partial_ids"),
           pb_get("pb_selector_partial_scores")});
      push_unique_offset(out.selector_params_offsets,
                         e.dynamic(stage1, 3, sp_bytes));

      e.begin_stage("packed_selector_topk_merge");
      auto &stage2 = e.emit(
          "dflash_forward", "omlx_m5_selector_topk16_stage2_batched",
          {.width = selector_stage1_chunks, .height = 7u * B},
          {.width = selector_stage1_chunks},
          {pb_get("pb_selector_partial_ids"),
           pb_get("pb_selector_partial_scores"), pb_get("pb_selector_ids"),
           pb_get("pb_selector_scores")});
      push_unique_offset(out.selector_params_offsets,
                         e.dynamic(stage2, 4, sp_bytes));

      e.begin_stage("packed_selector_finish");
      auto &select = e.emit(
          "dflash_forward",
          sampling_contract == SamplingContract::PositionCoupledV1
              ? "omlx_m5_selector_finish_position_batched"
              : "omlx_m5_selector_finish_batched",
          {.width = 1024, .height = B}, {.width = 1024, .height = 1},
          {pb_get("pb_selector_ids"), pb_get("pb_selector_scores"),
           pb_get("pb_selector_projected"),
           data("candidate_selector.predecessor_codebook"),
           data("candidate_selector.successor_codebook"),
           pb_get("pb_selector_tokens"), pb_get("pb_selector_probabilities"),
           pb_get("pb_selector_fallback")});
      push_unique_offset(out.selector_params_offsets,
                         e.dynamic(select, 8, sp_bytes));
      if (sampling_contract == SamplingContract::PositionCoupledV1) {
        out.selector_keys_offset =
            e.dynamic(select, 9, std::vector<std::byte>(U64(B) * 24ULL));
      }
    }

    out.profile = e.finish();
    ensure_profile_icb(context, out.profile);
    return out;
  }

  PackedDraftProfile &acquire_packed_batch(const PackedBatchKey &key) {
    for (auto it = packed_cache_.begin(); it != packed_cache_.end(); ++it) {
      if (it->key == key) {
        if (it != packed_cache_.begin()) {
          auto hit = std::move(*it);
          packed_cache_.erase(it);
          packed_cache_.push_front(std::move(hit));
        }
        return packed_cache_.front();
      }
    }
    if (packed_cache_.size() >= 8U)
      packed_cache_.pop_back();
    packed_cache_.push_front(compile_packed_batch(key));
    return packed_cache_.front();
  }

  static std::atomic<U64> next_owner_id_;
  U64 owner_id = next_owner_id_.fetch_add(1ULL, std::memory_order_relaxed);
  const MetalContext &context;
  const MaterializedModel &weights;
  DFlash2ModelConfig config;
  U32 max_context, h, q, kv;
  U64 feature_row_bytes = 0;
  U32 feature_ring_rows = 0;
  U64 target_epoch = 0;
  U64 cached_epoch = 0;
  MaterializedWeightView embedding, head;
  Ref features;
  std::filesystem::path directory;
  bool m4_short_routes = false;
  LinearNumericPolicy linear_policy = LinearNumericPolicy::Legacy;
  SamplingContract sampling_contract = SamplingContract::LegacyRejectionV1;
  U64 keyed_sampling_offset = 0;
  std::vector<std::unique_ptr<MetalBuffer>> buffers;
  std::map<std::string, Ref> refs;
  U32 pb_max_b = 0;
  std::vector<std::unique_ptr<MetalBuffer>> pb_buffers;
  std::map<std::string, Ref> pb_refs;
  std::deque<PackedDraftProfile> packed_cache_;
  std::vector<float> selector, predecessor, successor;
  Profile profile, cpu_profile;
  std::vector<ContextProfile> context_profiles;
  std::vector<FusedProfile> fused_profiles;
  std::vector<FusedProfile> fused_cpu_profiles;
  struct CachedContextSlot {
    bool valid = false;
    U32 position = 0;
    U64 lo = 0;
    U64 hi = 0;
  };
  std::vector<CachedContextSlot> cached_slots;
  U32 frontier = 0;
  bool mxfp4_nax_active() const {
    return mxfp4_nax_cache && mxfp4_nax_cache->active();
  }
  bool bf16_nax_enabled = false;
  bool selector_nax_f32_enabled = false;
  U32 gpu_cores = 64u;
  std::shared_ptr<DFlashBf16QkvCache> dflash_qkv_cache;
  std::shared_ptr<Mxfp4NaxCache> mxfp4_nax_cache;
  Ref f16_scratch, f16_mid, f16_row_scale, f16_ones_scale;
  std::shared_ptr<AffineNaxCache> affine_nax_head_cache;
  Ref affine_nax_xs;
};
std::atomic<U64> DFlashForward::Impl::next_owner_id_{1ULL};
DFlashForward::DFlashForward(const MetalContext &context,
                             const MaterializedModel &weights,
                             DFlash2ModelConfig config, U32 max_context,
                             const MaterializedWeightView &embedding,
                             const MaterializedWeightView &head,
                             const MetalBuffer &features,
                             const std::filesystem::path &directory,
                             bool enable_m4_short_routes,
                             LinearNumericPolicy linear_policy,
                             SamplingContract sampling_contract)
    : impl_(std::make_unique<Impl>(context, weights, std::move(config),
                                   max_context, embedding, head, features,
                                   directory, enable_m4_short_routes,
                                   linear_policy, sampling_contract)) {}
DFlashForward::~DFlashForward() = default;
const DFlash2ModelConfig &DFlashForward::config() const {
  return impl_->config;
}
void DFlashForward::set_target_feature_epoch(U64 epoch) noexcept {
  impl_->target_epoch = epoch;
}
U64 DFlashForward::target_feature_epoch() const noexcept {
  return impl_->target_epoch;
}
void DFlashForward::set_target_feature_ring(U32 ring_rows) {
  auto &p = *impl_;
  if (ring_rows != 0) {
    const U32 min_rows = std::min(p.max_context, p.config.sliding_window);
    if (ring_rows < min_rows ||
        (p.feature_row_bytes > 0 &&
         U64(ring_rows) * p.feature_row_bytes > p.features.bytes))
      throw std::invalid_argument("DFlash feature_ring_rows out of bounds");
  }
  if (p.feature_ring_rows != ring_rows) {
    p.feature_ring_rows = ring_rows;
    p.frontier = 0;
    p.cached_epoch = 0;
    std::ranges::fill(p.cached_slots, Impl::CachedContextSlot{});
  }
}
U32 DFlashForward::target_feature_ring() const noexcept {
  return impl_->feature_ring_rows;
}
QwenProfileDescription DFlashForward::inspect_profile() const {
  const auto &p = impl_->profile;
  QwenProfileDescription out{.capacity = p.capacity,
                             .dispatch_count = p.plan.dispatch_count(),
                             .barrier_count = p.plan.barrier_count()};
  out.numeric_contract_id = p.numeric_contract_id;
  out.sampling_contract_id =
      std::string(sampling_contract_name(impl_->sampling_contract));
  for (const auto &r : p.weight_routes)
    out.weights.push_back(
        {r.stage, r.object, r.format, r.library, r.function, r.family, r.source,
         r.fallback_reason, r.n, r.k, r.t, r.grid, r.threads,
         r.numeric_contract_id, r.row_exact_group, r.row_exact_eligible});
  return out;
}
std::vector<QwenStageTiming> DFlashForward::profile_stages() {
  auto &p = *impl_;
  if (!p.profile.icb)
    throw std::logic_error("draft stage diagnosis requires recorded ICB");
  std::vector<QwenStageTiming> timings;
  for (const auto &stage : p.profile.stages) {
    const auto run = p.context.replay_icb(*p.profile.icb, stage.name);
    timings.push_back({stage.name, run.timing.gpu_ms, run.timing.wall_ms,
                       run.dispatch_count});
  }
  return timings;
}
DFlashPreparedProposal DFlashForward::prepare_propose(
    U32 anchor, U32 context_tokens, U32 draft_tokens, float temperature,
    U64 seed, bool use_icb, bool inspect_tensors, bool cpu_selector,
    bool tokens_only) {
  auto &p = *impl_;
  const auto &c = p.config;
  if (anchor >= c.vocabulary_size || !context_tokens ||
      context_tokens > p.max_context || !draft_tokens || draft_tokens > 7 ||
      draft_tokens + 1 > c.block_size || !std::isfinite(temperature) ||
      temperature < 0)
    throw std::invalid_argument("DFlash proposal request invalid");

  const bool use_epoch_fast_path = (p.target_epoch != 0 && !inspect_tensors);
  if (use_epoch_fast_path) {
    if (p.cached_epoch != 0 && p.cached_epoch != p.target_epoch) {
      p.frontier = 0;
    }
    p.cached_epoch = p.target_epoch;
  }
  if (context_tokens < p.frontier)
    throw std::invalid_argument("DFlash context rewound without reset");

  U32 width = draft_tokens + 1;
  DFlashPreparedProposal prep;
  prep.anchor = anchor;
  prep.context_tokens = context_tokens;
  prep.draft_tokens = draft_tokens;
  prep.width = width;
  prep.temperature = temperature;
  prep.seed = seed;
  prep.inspect_tensors = inspect_tensors;
  prep.cpu_selector = cpu_selector;
  prep.tokens_only = tokens_only;
  auto &result = prep.partial;
  const U32 first = context_tokens >= c.sliding_window
                        ? context_tokens - c.sliding_window + 1
                        : 0;
  const U64 row_bytes = p.feature_row_bytes;
  const U32 cache_cap = U32(p.cached_slots.size());
  const auto physical_row = [&](U32 pos) -> U32 {
    return p.feature_ring_rows ? (pos % p.feature_ring_rows) : pos;
  };

  if (!use_epoch_fast_path && p.frontier > first && cache_cap > 0 &&
      row_bytes > 0) {
    const U32 verify_end = std::min(p.frontier, context_tokens);
    for (U32 pos = first; pos < verify_end; ++pos) {
      const auto &slot = p.cached_slots[pos % cache_cap];
      const U32 phys = physical_row(pos);
      const auto row_span = p.features.buffer->view(
          p.features.offset + U64(phys) * row_bytes, row_bytes);
      const auto [lo, hi] = hash_feature_row(row_span);
      if (!slot.valid || slot.position != pos || slot.lo != lo ||
          slot.hi != hi) {
        p.frontier = pos;
        break;
      }
    }
  }

  U32 begin = std::max(first, p.frontier);
  result.context_rows_projected = context_tokens - begin;
  result.context_rows_reused = begin - first;

  const auto record_crc_range = [&](U32 start_pos, U32 count) {
    if (!use_epoch_fast_path && cache_cap > 0 && row_bytes > 0) {
      for (U32 r = 0; r < count; ++r) {
        const U32 pos = start_pos + r;
        const U32 phys = physical_row(pos);
        const auto row_span = p.features.buffer->view(
            p.features.offset + U64(phys) * row_bytes, row_bytes);
        const auto [lo, hi] = hash_feature_row(row_span);
        p.cached_slots[pos % cache_cap] = {true, pos, lo, hi};
      }
    }
  };

  // Drain any cold-start prefix chunks larger than 8 rows; leave a final 1..8
  // row tail to fuse into the single-ICB (update_context + draft_propose) plan.
  auto &fused_table = cpu_selector ? p.fused_cpu_profiles : p.fused_profiles;
  const U32 max_fused_cap =
      fused_table.empty() ? 0U : fused_table.back().context_capacity;
  while (begin < context_tokens &&
         (context_tokens - begin > max_fused_cap || max_fused_cap == 0U)) {
    const U32 remaining = context_tokens - begin;
    U32 count = std::min({1024U, remaining, c.sliding_window});
    if (remaining > 1024U && remaining - count > 0U &&
        remaining - count <= max_fused_cap) {
      // Natural leftover <= 8 will be fused below.
    }
    auto [submission, capacity] = p.update_context(count, begin, use_icb);
    result.submission.timing.gpu_ms += submission.timing.gpu_ms;
    result.submission.dispatch_count += submission.dispatch_count;
    result.context_rows_launched += capacity;
    record_crc_range(begin, count);
    begin += count;
    p.frontier = begin;
  }

  std::array<U32, 8> tokens;
  tokens.fill(c.mask_token_id);
  tokens[0] = anchor;
  const_cast<MetalBuffer *>(p.get("tokens").buffer)
      ->write(std::as_bytes(std::span(tokens)));

  SelectorParams selector_params{
      .vocab = c.vocabulary_size,
      .hidden = c.hidden_size,
      .rank = c.selector_rank,
      .top_k = c.selector_top_k,
      .blocks =
          U32((U64(c.vocabulary_size) + selector_chunk - 1) / selector_chunk),
      .groups = U32((U64(c.vocabulary_size) + selector_chunk * 32 - 1) /
                    (selector_chunk * 32)),
      .width = width,
      .anchor = anchor,
      .position = context_tokens,
      .temperature = temperature};
  for (U32 t = 0; t < draft_tokens; ++t)
    selector_params.uniforms[t] =
        float(candidate_selector_uniform(seed, U64(context_tokens) + t));

  if (begin < context_tokens) {
    const U32 rem = context_tokens - begin;
    auto found =
        std::find_if(fused_table.begin(), fused_table.end(),
                     [rem](const Impl::FusedProfile &fp) {
                       return rem <= fp.context_capacity;
                     });
    if (found == fused_table.end())
      throw std::out_of_range("DFlash fused context update capacity");
    auto &fp = *found;
    auto &fprof = fp.profile;
    for (std::size_t idx = 0; idx < fp.context_patch_count; ++idx) {
      const auto &patch = fprof.patches[idx];
      const U32 value = patch.position ? begin : rem * patch.multiplier;
      std::memcpy(fprof.parameter_data.data() + patch.offset + patch.word * 4,
                  &value, 4);
    }
    std::memcpy(fprof.parameter_data.data() + fp.gather_ring_offset,
                &p.feature_ring_rows, 4);
    for (std::size_t idx = fp.context_patch_count; idx < fprof.patches.size();
         ++idx) {
      const auto &patch = fprof.patches[idx];
      const U32 value =
          patch.position ? context_tokens : width * patch.multiplier;
      std::memcpy(fprof.parameter_data.data() + patch.offset + patch.word * 4,
                  &value, 4);
    }
    if (!cpu_selector) {
      std::memcpy(fprof.parameter_data.data() + fprof.sampling_offset,
                  &selector_params, sizeof(selector_params));
      if (p.sampling_contract == SamplingContract::PositionCoupledV1) {
        const std::array<U64, 3> key{seed, U64(context_tokens) + 1,
                                     draft_sampling_purpose};
        std::memcpy(fprof.parameter_data.data() + fp.keyed_sampling_offset,
                    key.data(), 24);
      }
    }
    fprof.parameters.write("step", fprof.parameter_data);
    record_crc_range(begin, rem);
    result.context_rows_launched += fp.context_capacity;
    p.frontier = context_tokens;
    result.short_matrix_calls = fprof.short_matrix_calls;
    result.short_lane_calls = fprof.short_lane_calls;
    prep.execution_profile = &fprof;
    if (use_icb && fprof.icb) {
      prep.icb = &(*fprof.icb);
    }
    return prep;
  }

  auto &profile = cpu_selector ? p.cpu_profile : p.profile;
  for (const auto &patch : profile.patches) {
    U32 value = patch.position ? context_tokens : width * patch.multiplier;
    std::memcpy(profile.parameter_data.data() + patch.offset + patch.word * 4,
                &value, 4);
  }
  if (!cpu_selector)
    std::memcpy(profile.parameter_data.data() + profile.sampling_offset,
                &selector_params, sizeof(selector_params));
  if (!cpu_selector &&
      p.sampling_contract == SamplingContract::PositionCoupledV1) {
    const std::array<U64, 3> key{seed, U64(context_tokens) + 1,
                                 draft_sampling_purpose};
    std::memcpy(profile.parameter_data.data() + p.keyed_sampling_offset,
                key.data(), 24);
  }
  profile.parameters.write("step", profile.parameter_data);
  result.short_matrix_calls = profile.short_matrix_calls;
  result.short_lane_calls = profile.short_lane_calls;
  prep.execution_profile = &profile;
  if (use_icb && profile.icb) {
    prep.icb = &(*profile.icb);
  }
  return prep;
}

DFlash2DraftResult DFlashForward::finish_propose(
    DFlashPreparedProposal &&prep, bool already_executed,
    MetalSubmission batched_submission) {
  auto &p = *impl_;
  const auto &c = p.config;
  const U32 width = prep.width;
  const U32 draft_tokens = prep.draft_tokens;
  const U32 anchor = prep.anchor;
  const U32 context_tokens = prep.context_tokens;
  const float temperature = prep.temperature;
  const U64 seed = prep.seed;
  const bool inspect_tensors = prep.inspect_tensors;
  const bool cpu_selector = prep.cpu_selector;
  const bool tokens_only = prep.tokens_only && !inspect_tensors;
  const auto &exec_profile =
      prep.execution_profile
          ? *static_cast<const Profile *>(prep.execution_profile)
          : (cpu_selector ? p.cpu_profile : p.profile);
  DFlash2DraftResult result = std::move(prep.partial);
  MetalSubmission proposal;
  if (already_executed) {
    proposal = batched_submission;
  } else if (prep.icb) {
    proposal = p.context.replay_icb(*prep.icb);
  } else {
    const std::array calls = {
        MetalCommandInvocation{.plan = &exec_profile.plan,
                               .bindings = exec_profile.bindings,
                               .parameters = &exec_profile.parameters}};
    proposal = p.context.execute_batch(calls);
  }
  result.submission.timing.gpu_ms += proposal.timing.gpu_ms;
  result.submission.dispatch_count += proposal.dispatch_count;
  if (inspect_tensors || cpu_selector) {
    result.hidden =
        bf16_values(*p.get("norm").buffer, U64(width) * c.hidden_size);
    result.logits =
        bf16_values(*p.get("logits").buffer, U64(width) * c.vocabulary_size);
    result.host_read_bytes +=
        U64(width) * (c.hidden_size + c.vocabulary_size) * 2;
  }
  if (cpu_selector) {
    result.host_read_bytes += p.ensure_cpu_selector();
    for (U32 t = 1; t < width; ++t) {
      CandidateRow row;
      row.token_ids.resize(c.vocabulary_size);
      std::iota(row.token_ids.begin(), row.token_ids.end(), 0);
      const auto logits =
          std::span(result.logits)
              .subspan(U64(t) * c.vocabulary_size, c.vocabulary_size);
      std::partial_sort(
          row.token_ids.begin(), row.token_ids.begin() + c.selector_top_k,
          row.token_ids.end(), [&](auto a, auto b) {
            return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
          });
      row.token_ids.resize(c.selector_top_k);
      for (auto id : row.token_ids)
        row.unary_scores.push_back(logits[id]);
      row.projected_hidden.resize(c.selector_rank);
      for (U32 r = 0; r < c.selector_rank; ++r)
        for (U32 f = 0; f < c.hidden_size; ++f)
          row.projected_hidden[r] =
              std::fma(p.selector[U64(r) * c.hidden_size + f],
                       result.hidden[U64(t) * c.hidden_size + f],
                       row.projected_hidden[r]);
      result.candidate_rows.push_back(std::move(row));
    }
    result.selection = select_candidate_path(
        anchor, result.candidate_rows, p.predecessor, p.successor,
        c.vocabulary_size, c.selector_rank, temperature, seed,
        U64(context_tokens) +
            U64(p.sampling_contract == SamplingContract::PositionCoupledV1),
        p.sampling_contract);
  } else {
    const auto read = [&](std::string_view name, U64 offset, U64 bytes) {
      result.host_read_bytes += bytes;
      return p.get(std::string(name)).buffer->read(offset, bytes);
    };
    U32 needs_fallback = 0;
    const auto flag = read("selector_fallback", 0, 4);
    std::memcpy(&needs_fallback, flag.data(), 4);
    if (needs_fallback) {
      result.compact_cpu_fallback = true;
      result.host_read_bytes += p.ensure_cpu_selector();
      const auto ids = read("selector_ids", U64(c.selector_top_k) * 4,
                            U64(draft_tokens) * c.selector_top_k * 4);
      const auto scores = read("selector_scores", U64(c.selector_top_k) * 4,
                               U64(draft_tokens) * c.selector_top_k * 4);
      const auto projected =
          read("selector_projected", U64(c.selector_rank) * 4,
               U64(draft_tokens) * c.selector_rank * 4);
      for (U32 t = 0; t < draft_tokens; ++t) {
        CandidateRow row;
        row.token_ids.resize(c.selector_top_k);
        row.unary_scores.resize(c.selector_top_k);
        row.projected_hidden.resize(c.selector_rank);
        std::memcpy(row.token_ids.data(),
                    ids.data() + U64(t) * c.selector_top_k * 4,
                    U64(c.selector_top_k) * 4);
        std::memcpy(row.unary_scores.data(),
                    scores.data() + U64(t) * c.selector_top_k * 4,
                    U64(c.selector_top_k) * 4);
        std::memcpy(row.projected_hidden.data(),
                    projected.data() + U64(t) * c.selector_rank * 4,
                    U64(c.selector_rank) * 4);
        result.candidate_rows.push_back(std::move(row));
      }
      result.selection = select_candidate_path(
          anchor, result.candidate_rows, p.predecessor, p.successor,
          c.vocabulary_size, c.selector_rank, temperature, seed,
          U64(context_tokens) +
              U64(p.sampling_contract == SamplingContract::PositionCoupledV1),
          p.sampling_contract);
    } else if (tokens_only) {
      const auto tokens = read("selector_tokens", 4, U64(draft_tokens) * 4);
      result.selection.tokens.reserve(draft_tokens);
      result.selection.steps.reserve(draft_tokens);
      for (U32 t = 0; t < draft_tokens; ++t) {
        U32 token = 0;
        std::memcpy(&token, tokens.data() + U64(t) * 4, 4);
        if (token >= c.vocabulary_size)
          throw std::runtime_error("GPU selector produced invalid token");
        const auto token_i32 = std::int32_t(token);
        result.selection.tokens.push_back(token_i32);
        result.selection.steps.push_back(SelectorStep{.token = token_i32});
      }
      result.gpu_selector = true;
    } else {
      const auto ids = read("selector_ids", U64(c.selector_top_k) * 4,
                            U64(draft_tokens) * c.selector_top_k * 4);
      const auto tokens = read("selector_tokens", 4, U64(draft_tokens) * 4);
      const auto probabilities =
          read("selector_probabilities", U64(c.selector_top_k) * 4,
               U64(draft_tokens) * c.selector_top_k * 4);
      result.selection.tokens.reserve(draft_tokens);
      result.selection.steps.reserve(draft_tokens);
      for (U32 t = 0; t < draft_tokens; ++t) {
        SelectorStep step;
        U32 token = 0;
        std::memcpy(&token, tokens.data() + U64(t) * 4, 4);
        if (token >= c.vocabulary_size)
          throw std::runtime_error("GPU selector produced invalid token");
        step.token = std::int32_t(token);
        step.candidate_ids.resize(c.selector_top_k);
        step.probabilities.resize(c.selector_top_k);
        std::memcpy(step.candidate_ids.data(),
                    ids.data() + U64(t) * c.selector_top_k * 4,
                    U64(c.selector_top_k) * 4);
        std::memcpy(step.probabilities.data(),
                    probabilities.data() + U64(t) * c.selector_top_k * 4,
                    U64(c.selector_top_k) * 4);
        result.selection.tokens.push_back(step.token);
        result.selection.steps.push_back(std::move(step));
      }
      result.gpu_selector = true;
    }
  }
  result.width = width;
  result.draft_dispatches = result.submission.dispatch_count;
  result.position_parallel_threads = 128;
  result.parallel_backbone = true;
  result.dynamic_convolution = true;
  result.conditional_selector = true;
  return result;
}
DFlash2DraftResult DFlashForward::propose(U32 anchor, U32 context_tokens,
                                          U32 draft_tokens, float temperature,
                                          U64 seed, bool use_icb,
                                          bool inspect_tensors,
                                          bool cpu_selector,
                                          bool tokens_only) {
  auto prep =
      prepare_propose(anchor, context_tokens, draft_tokens, temperature, seed,
                      use_icb, inspect_tensors, cpu_selector, tokens_only);
  return finish_propose(std::move(prep), false);
}

DFlashPackedBatchResult DFlashForward::propose_packed_batch(
    std::span<const DFlashPackedSlotSlice> slices, bool use_icb,
    bool cpu_selector) {
  auto &p = *impl_;
  const auto &c = p.config;
  const U32 B = U32(slices.size());
  if (B < 2U || B > 8U)
    throw std::invalid_argument(
        "DFlashForward::propose_packed_batch requires 2..8 slices");
  const U32 total_rows = 8U * B;
  if (total_rows > std::min(p.max_context, 1024U))
    throw std::out_of_range(
        "DFlashForward::propose_packed_batch exceeds context buffer capacity");

  DFlashPackedBatchResult batch_out;
  batch_out.slot_results.resize(B);

  Impl::PackedBatchKey skey;
  skey.slots.reserve(B);
  skey.cpu_selector = cpu_selector;

  std::array<U32, 8> widths{};
  std::array<U32, 8> positions{};
  std::array<U32, 8> begins{};
  std::array<U32, 8> rems{};
  std::array<U32, 8> ctx_offs{};
  U32 total_ctx = 0U;

  for (U32 b = 0; b < B; ++b) {
    const auto &sl = slices[b];
    if (!sl.slot_draft)
      throw std::invalid_argument("DFlash packed slice missing slot_draft");
    auto &sp = *sl.slot_draft->impl_;
    if (sp.h != p.h || sp.q != p.q || sp.kv != p.kv ||
        sp.max_context != p.max_context ||
        sp.config.sliding_window != c.sliding_window ||
        sp.config.vocabulary_size != c.vocabulary_size ||
        sp.config.layer_count != c.layer_count ||
        sp.sampling_contract != p.sampling_contract ||
        sp.linear_policy != p.linear_policy)
      throw std::invalid_argument("DFlash packed slice contract mismatch");
    if (std::find(skey.slots.begin(), skey.slots.end(), &sp) !=
        skey.slots.end())
      throw std::invalid_argument("DFlash packed slice duplicate slot_draft");
    if (sl.anchor >= c.vocabulary_size || !sl.context_tokens ||
        sl.context_tokens > sp.max_context || !sl.draft_tokens ||
        sl.draft_tokens > 7 || sl.draft_tokens + 1 > c.block_size ||
        !std::isfinite(sl.temperature) || sl.temperature < 0)
      throw std::invalid_argument("DFlash packed proposal request invalid");

    skey.slots.push_back(&sp);
    widths[b] = sl.draft_tokens + 1U;
    positions[b] = sl.context_tokens;

    const bool use_epoch_fast_path = (sp.target_epoch != 0);
    if (use_epoch_fast_path) {
      if (sp.cached_epoch != 0 && sp.cached_epoch != sp.target_epoch) {
        sp.frontier = 0;
      }
      sp.cached_epoch = sp.target_epoch;
    }
    if (sl.context_tokens < sp.frontier)
      throw std::invalid_argument("DFlash context rewound without reset");

    auto &res = batch_out.slot_results[b];
    const U32 first = sl.context_tokens >= c.sliding_window
                          ? sl.context_tokens - c.sliding_window + 1
                          : 0;
    const U64 row_bytes = sp.feature_row_bytes;
    const U32 cache_cap = U32(sp.cached_slots.size());
    const auto physical_row = [&](U32 pos) -> U32 {
      return sp.feature_ring_rows ? (pos % sp.feature_ring_rows) : pos;
    };

    if (!use_epoch_fast_path && sp.frontier > first && cache_cap > 0 &&
        row_bytes > 0) {
      const U32 verify_end = std::min(sp.frontier, sl.context_tokens);
      for (U32 pos = first; pos < verify_end; ++pos) {
        const auto &cslot = sp.cached_slots[pos % cache_cap];
        const U32 phys = physical_row(pos);
        const auto row_span = sp.features.buffer->view(
            sp.features.offset + U64(phys) * row_bytes, row_bytes);
        const auto [lo, hi] = hash_feature_row(row_span);
        if (!cslot.valid || cslot.position != pos || cslot.lo != lo ||
            cslot.hi != hi) {
          sp.frontier = pos;
          break;
        }
      }
    }

    U32 begin = std::max(first, sp.frontier);
    res.context_rows_projected = sl.context_tokens - begin;
    res.context_rows_reused = begin - first;

    const auto record_crc = [&](U32 start_pos, U32 count) {
      if (!use_epoch_fast_path && cache_cap > 0 && row_bytes > 0) {
        for (U32 r = 0; r < count; ++r) {
          const U32 pos = start_pos + r;
          const U32 phys = physical_row(pos);
          const auto row_span = sp.features.buffer->view(
              sp.features.offset + U64(phys) * row_bytes, row_bytes);
          const auto [lo, hi] = hash_feature_row(row_span);
          sp.cached_slots[pos % cache_cap] = {true, pos, lo, hi};
        }
      }
    };

    while (begin < sl.context_tokens && sl.context_tokens - begin > 8U) {
      const U32 remaining = sl.context_tokens - begin;
      const U32 count = std::min({1024U, remaining, c.sliding_window});
      auto [submission, capacity] = sp.update_context(count, begin, use_icb);
      res.submission.timing.gpu_ms += submission.timing.gpu_ms;
      res.submission.dispatch_count += submission.dispatch_count;
      res.context_rows_launched += capacity;
      record_crc(begin, count);
      begin += count;
      sp.frontier = begin;
    }

    begins[b] = begin;
    rems[b] = sl.context_tokens - begin;
    ctx_offs[b] = total_ctx;
    total_ctx += rems[b];
  }

  skey.has_ctx = (total_ctx > 0U);
  p.ensure_packed_workspace(B);
  auto &entry = p.acquire_packed_batch(skey);
  auto &prof = entry.profile;

  // Stage [8*B] tokens (anchor + mask tokens per slot).
  std::vector<U32> packed_tokens(total_rows, c.mask_token_id);
  for (U32 b = 0; b < B; ++b) {
    packed_tokens[8U * b] = slices[b].anchor;
  }
  const_cast<MetalBuffer *>(p.pb_get("pb_tokens").buffer)
      ->write(std::as_bytes(std::span(packed_tokens)));

  // Patch Stage A (packed context projection) dynamic parameters.
  if (skey.has_ctx) {
    for (std::size_t idx = 0; idx < entry.ctx_patch_count; ++idx) {
      const auto &patch = prof.patches[idx];
      const U32 value = patch.position ? 0U : total_ctx * patch.multiplier;
      std::memcpy(prof.parameter_data.data() + patch.offset + patch.word * 4,
                  &value, 4);
    }
    const U32 feature_width = p.h * U32(c.target_layer_ids.size());
    const U32 ring_cap = std::min(p.max_context, c.sliding_window);
    for (U32 b = 0; b < B; ++b) {
      const auto &sp = *skey.slots[b];
      const ContextGatherOffsetHost gp{.tokens = rems[b],
                                       .width = feature_width,
                                       .position = begins[b],
                                       .ring_rows = sp.feature_ring_rows,
                                       .dst_row_offset = ctx_offs[b]};
      std::memcpy(prof.parameter_data.data() + entry.gather_offsets[b], &gp,
                  sizeof(gp));
      for (U32 layer = 0; layer < c.layer_count; ++layer) {
        const ContextRopeAppendOffsetHost ap{
            .tokens = rems[b],
            .kv_heads = c.key_value_heads,
            .dim = c.head_dim,
            .position = begins[b],
            .capacity = ring_cap,
            .epsilon = c.epsilon,
            .theta = c.rope_theta,
            .src_row_offset = ctx_offs[b]};
        std::memcpy(
            prof.parameter_data.data() + entry.ctx_append_offsets[layer][b],
            &ap, sizeof(ap));
      }
    }
  }

  // Patch Stage B (packed [8*B, H] backbone + selector) dynamic parameters.
  for (std::size_t idx = entry.ctx_patch_count; idx < prof.patches.size();
       ++idx) {
    const auto &patch = prof.patches[idx];
    const U32 value = patch.position ? 0U : total_rows * patch.multiplier;
    std::memcpy(prof.parameter_data.data() + patch.offset + patch.word * 4,
                &value, 4);
  }
  {
    const BatchedConvParamsHost cp{.batch_size = B,
                                   .hidden = p.h,
                                   .group_size = c.conv_group_size,
                                   .output_half = 0U,
                                   .slot_widths = widths};
    for (U64 off : entry.conv_offsets) {
      std::memcpy(prof.parameter_data.data() + off, &cp, sizeof(cp));
    }
    const BatchedConvAddNormParamsHost canp{.batch_size = B,
                                            .hidden = p.h,
                                            .group_size = c.conv_group_size,
                                            .epsilon = c.epsilon,
                                            .slot_widths = widths};
    for (U64 off : entry.conv_add_norm_offsets) {
      std::memcpy(prof.parameter_data.data() + off, &canp, sizeof(canp));
    }
    const BatchedQkRopeStridedParamsHost qkrp{.batch_size = B,
                                              .q_heads = c.query_heads,
                                              .kv_heads = c.key_value_heads,
                                              .dim = c.head_dim,
                                              .qkv_stride = p.q + 2U * p.kv,
                                              .epsilon = c.epsilon,
                                              .theta = c.rope_theta,
                                              .pad = 0U,
                                              .slot_widths = widths,
                                              .slot_positions = positions};
    for (U64 off : entry.qk_rope_offsets) {
      std::memcpy(prof.parameter_data.data() + off, &qkrp, sizeof(qkrp));
    }
    if (!entry.rope_q_offsets.empty()) {
      const BatchedRopeParamsHost rqp{.batch_size = B,
                                      .heads = c.query_heads,
                                      .dim = c.head_dim,
                                      .pad = 0U,
                                      .epsilon = c.epsilon,
                                      .theta = c.rope_theta,
                                      .slot_widths = widths,
                                      .slot_positions = positions};
      for (U64 off : entry.rope_q_offsets) {
        std::memcpy(prof.parameter_data.data() + off, &rqp, sizeof(rqp));
      }
    }
    if (!entry.rope_k_offsets.empty()) {
      const BatchedRopeParamsHost rkp{.batch_size = B,
                                      .heads = c.key_value_heads,
                                      .dim = c.head_dim,
                                      .pad = 0U,
                                      .epsilon = c.epsilon,
                                      .theta = c.rope_theta,
                                      .slot_widths = widths,
                                      .slot_positions = positions};
      for (U64 off : entry.rope_k_offsets) {
        std::memcpy(prof.parameter_data.data() + off, &rkp, sizeof(rkp));
      }
    }
    for (U32 layer = 0; layer < c.layer_count; ++layer) {
      for (U32 b = 0; b < B; ++b) {
        const U64 off = entry.attn_offsets[layer][b];
        std::memcpy(prof.parameter_data.data() + off + 0U, &widths[b], 4);
        std::memcpy(prof.parameter_data.data() + off + 4U, &positions[b], 4);
      }
    }
    if (!cpu_selector) {
      const U32 blocks =
          U32((U64(c.vocabulary_size) + selector_chunk - 1) / selector_chunk);
      const U32 groups = U32((U64(c.vocabulary_size) + selector_chunk * 32 - 1) /
                             (selector_chunk * 32));
      std::vector<SelectorParams> sp_arr(B);
      std::vector<std::array<U64, 3>> keys_arr(B);
      for (U32 b = 0; b < B; ++b) {
        const auto &sl = slices[b];
        SelectorParams sp_b{.vocab = c.vocabulary_size,
                            .hidden = c.hidden_size,
                            .rank = c.selector_rank,
                            .top_k = c.selector_top_k,
                            .blocks = blocks,
                            .groups = groups,
                            .width = widths[b],
                            .anchor = sl.anchor,
                            .position = sl.context_tokens,
                            .temperature = sl.temperature};
        for (U32 t = 0; t < sl.draft_tokens; ++t) {
          sp_b.uniforms[t] = float(
              candidate_selector_uniform(sl.seed, U64(sl.context_tokens) + t));
        }
        sp_arr[b] = sp_b;
        keys_arr[b] = {sl.seed, U64(sl.context_tokens) + 1ULL,
                       draft_sampling_purpose};
      }
      for (U64 off : entry.selector_params_offsets) {
        std::memcpy(prof.parameter_data.data() + off, sp_arr.data(),
                    std::span(sp_arr).size_bytes());
      }
      if (p.sampling_contract == SamplingContract::PositionCoupledV1) {
        std::memcpy(prof.parameter_data.data() + entry.selector_keys_offset,
                    keys_arr.data(), std::span(keys_arr).size_bytes());
      }
    }
  }
  prof.parameters.write("step", prof.parameter_data);

  // Update each slot's context frontier & CRC cache.
  for (U32 b = 0; b < B; ++b) {
    auto &sp = *slices[b].slot_draft->impl_;
    if (rems[b] > 0U) {
      const bool use_epoch_fast_path = (sp.target_epoch != 0);
      const U64 row_bytes = sp.feature_row_bytes;
      const U32 cache_cap = U32(sp.cached_slots.size());
      if (!use_epoch_fast_path && cache_cap > 0 && row_bytes > 0) {
        for (U32 r = 0; r < rems[b]; ++r) {
          const U32 pos = begins[b] + r;
          const U32 phys =
              sp.feature_ring_rows ? (pos % sp.feature_ring_rows) : pos;
          const auto row_span = sp.features.buffer->view(
              sp.features.offset + U64(phys) * row_bytes, row_bytes);
          const auto [lo, hi] = hash_feature_row(row_span);
          sp.cached_slots[pos % cache_cap] = {true, pos, lo, hi};
        }
      }
      batch_out.slot_results[b].context_rows_launched += rems[b];
      sp.frontier = slices[b].context_tokens;
    }
  }

  // Single CB submission for ALL B slots (packed context + 5L backbone + lm_head + selector)!
  if (use_icb && prof.icb) {
    batch_out.submission = p.context.replay_icb(*prof.icb);
  } else {
    const std::array calls = {
        MetalCommandInvocation{.plan = &prof.plan,
                               .bindings = prof.bindings,
                               .parameters = &prof.parameters}};
    batch_out.submission = p.context.execute_batch(calls);
  }

  const double gpu_share = batch_out.submission.timing.gpu_ms / double(B);
  const double wall_share = batch_out.submission.timing.wall_ms / double(B);
  const U64 disp_share =
      std::max<U64>(1ULL, batch_out.submission.dispatch_count / B);

  if (cpu_selector) {
    const auto all_hidden =
        bf16_values(*p.pb_get("pb_norm").buffer, U64(total_rows) * c.hidden_size);
    const auto all_logits = bf16_values(*p.pb_get("pb_logits").buffer,
                                        U64(total_rows) * c.vocabulary_size);
    const U64 sel_w_bytes = p.ensure_cpu_selector();
    for (U32 b = 0; b < B; ++b) {
      const auto &sl = slices[b];
      auto &res = batch_out.slot_results[b];
      res.host_read_bytes +=
          U64(widths[b]) * (c.hidden_size + c.vocabulary_size) * 2 +
          (b == 0U ? sel_w_bytes : 0ULL);
      for (U32 t = 1; t < widths[b]; ++t) {
        const U64 grow = U64(8U * b + t);
        CandidateRow row;
        row.token_ids.resize(c.vocabulary_size);
        std::iota(row.token_ids.begin(), row.token_ids.end(), 0);
        const auto logits =
            std::span(all_logits)
                .subspan(grow * c.vocabulary_size, c.vocabulary_size);
        std::partial_sort(
            row.token_ids.begin(), row.token_ids.begin() + c.selector_top_k,
            row.token_ids.end(), [&](auto a_id, auto b_id) {
              return logits[a_id] > logits[b_id] ||
                     (logits[a_id] == logits[b_id] && a_id < b_id);
            });
        row.token_ids.resize(c.selector_top_k);
        for (auto id : row.token_ids)
          row.unary_scores.push_back(logits[id]);
        row.projected_hidden.resize(c.selector_rank);
        for (U32 r = 0; r < c.selector_rank; ++r)
          for (U32 f_i = 0; f_i < c.hidden_size; ++f_i)
            row.projected_hidden[r] = std::fma(
                p.selector[U64(r) * c.hidden_size + f_i],
                all_hidden[grow * c.hidden_size + f_i], row.projected_hidden[r]);
        res.candidate_rows.push_back(std::move(row));
      }
      res.selection = select_candidate_path(
          sl.anchor, res.candidate_rows, p.predecessor, p.successor,
          c.vocabulary_size, c.selector_rank, sl.temperature, sl.seed,
          U64(sl.context_tokens) +
              U64(p.sampling_contract == SamplingContract::PositionCoupledV1),
          p.sampling_contract);
    }
  } else {
    const auto fb_raw =
        p.pb_get("pb_selector_fallback").buffer->read(0, U64(B) * 4U);
    const auto tok_raw =
        p.pb_get("pb_selector_tokens").buffer->read(0, U64(total_rows) * 4U);
    std::vector<U32> fallbacks(B, 0U);
    std::memcpy(fallbacks.data(), fb_raw.data(), fb_raw.size());
    std::vector<U32> all_tokens(total_rows, 0U);
    std::memcpy(all_tokens.data(), tok_raw.data(), tok_raw.size());

    bool need_probs = false;
    for (U32 b = 0; b < B; ++b) {
      if (!fallbacks[b] && !slices[b].tokens_only) {
        need_probs = true;
        break;
      }
    }
    std::vector<std::byte> all_ids_raw, all_probs_raw;
    if (need_probs) {
      all_ids_raw = p.pb_get("pb_selector_ids")
                        .buffer->read(0, U64(total_rows) * c.selector_top_k * 4);
      all_probs_raw =
          p.pb_get("pb_selector_probabilities")
              .buffer->read(0, U64(total_rows) * c.selector_top_k * 4);
    }

    for (U32 b = 0; b < B; ++b) {
      const auto &sl = slices[b];
      auto &res = batch_out.slot_results[b];
      res.host_read_bytes += 4U + U64(sl.draft_tokens) * 4U;
      if (fallbacks[b]) {
        res.compact_cpu_fallback = true;
        res.host_read_bytes += p.ensure_cpu_selector();
        const U64 row_start = U64(8U * b + 1U);
        const auto ids = p.pb_get("pb_selector_ids")
                             .buffer->read(row_start * c.selector_top_k * 4,
                                           U64(sl.draft_tokens) *
                                               c.selector_top_k * 4);
        const auto scores =
            p.pb_get("pb_selector_scores")
                .buffer->read(row_start * c.selector_top_k * 4,
                              U64(sl.draft_tokens) * c.selector_top_k * 4);
        const auto projected =
            p.pb_get("pb_selector_projected")
                .buffer->read(row_start * c.selector_rank * 4,
                              U64(sl.draft_tokens) * c.selector_rank * 4);
        res.host_read_bytes += ids.size() + scores.size() + projected.size();
        for (U32 t = 0; t < sl.draft_tokens; ++t) {
          CandidateRow row;
          row.token_ids.resize(c.selector_top_k);
          row.unary_scores.resize(c.selector_top_k);
          row.projected_hidden.resize(c.selector_rank);
          std::memcpy(row.token_ids.data(),
                      ids.data() + U64(t) * c.selector_top_k * 4,
                      U64(c.selector_top_k) * 4);
          std::memcpy(row.unary_scores.data(),
                      scores.data() + U64(t) * c.selector_top_k * 4,
                      U64(c.selector_top_k) * 4);
          std::memcpy(row.projected_hidden.data(),
                      projected.data() + U64(t) * c.selector_rank * 4,
                      U64(c.selector_rank) * 4);
          res.candidate_rows.push_back(std::move(row));
        }
        res.selection = select_candidate_path(
            sl.anchor, res.candidate_rows, p.predecessor, p.successor,
            c.vocabulary_size, c.selector_rank, sl.temperature, sl.seed,
            U64(sl.context_tokens) +
                U64(p.sampling_contract == SamplingContract::PositionCoupledV1),
            p.sampling_contract);
      } else if (sl.tokens_only) {
        res.selection.tokens.reserve(sl.draft_tokens);
        res.selection.steps.reserve(sl.draft_tokens);
        for (U32 t = 0; t < sl.draft_tokens; ++t) {
          const U32 token = all_tokens[8U * b + 1U + t];
          if (token >= c.vocabulary_size)
            throw std::runtime_error("GPU selector produced invalid token");
          const auto token_i32 = std::int32_t(token);
          res.selection.tokens.push_back(token_i32);
          res.selection.steps.push_back(SelectorStep{.token = token_i32});
        }
        res.gpu_selector = true;
      } else {
        res.host_read_bytes += 2ULL * U64(sl.draft_tokens) * c.selector_top_k * 4;
        res.selection.tokens.reserve(sl.draft_tokens);
        res.selection.steps.reserve(sl.draft_tokens);
        for (U32 t = 0; t < sl.draft_tokens; ++t) {
          const U32 token = all_tokens[8U * b + 1U + t];
          if (token >= c.vocabulary_size)
            throw std::runtime_error("GPU selector produced invalid token");
          SelectorStep step;
          step.token = std::int32_t(token);
          step.candidate_ids.resize(c.selector_top_k);
          step.probabilities.resize(c.selector_top_k);
          const U64 byte_off =
              U64(8U * b + 1U + t) * U64(c.selector_top_k) * 4ULL;
          std::memcpy(step.candidate_ids.data(), all_ids_raw.data() + byte_off,
                      U64(c.selector_top_k) * 4);
          std::memcpy(step.probabilities.data(),
                      all_probs_raw.data() + byte_off,
                      U64(c.selector_top_k) * 4);
          res.selection.tokens.push_back(step.token);
          res.selection.steps.push_back(std::move(step));
        }
        res.gpu_selector = true;
      }
    }
  }

  for (U32 b = 0; b < B; ++b) {
    auto &res = batch_out.slot_results[b];
    res.submission.timing.gpu_ms += gpu_share;
    res.submission.timing.wall_ms += wall_share;
    res.submission.dispatch_count += disp_share;
    res.short_matrix_calls = prof.short_matrix_calls;
    res.short_lane_calls = prof.short_lane_calls;
    res.width = widths[b];
    res.draft_dispatches = res.submission.dispatch_count;
    res.position_parallel_threads = 128;
    res.parallel_backbone = true;
    res.dynamic_convolution = true;
    res.conditional_selector = true;
  }
  return batch_out;
}
void DFlashForward::retain_frontier(U32 frontier_rows, U64 epoch) noexcept {
  impl_->frontier = frontier_rows;
  impl_->target_epoch = epoch;
  impl_->cached_epoch = epoch;
}
void DFlashForward::reset() {
  impl_->frontier = 0;
  impl_->cached_epoch = 0;
  std::ranges::fill(impl_->cached_slots, Impl::CachedContextSlot{});
}
U32 DFlashForward::cached_context() const { return impl_->frontier; }
} // namespace omlx::m5
