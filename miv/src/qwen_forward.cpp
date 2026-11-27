#include "omlx_m5/qwen_forward.h"
#include "affine_nax_cache.h"
#include "ane/ane_dual_offload.h"
#include "attention_nax.h"
#include "attention_sage2.h"
#include "bf16_nax.h"
#include "chunk_blend_cache.h"
#include "forward_plan.h"
#include "gdn_prefill.h"
#include "lora_runtime.h"
#include "moe_nax.h"
#include "mxfp4_nax_cache.h"
#include "silicon_tuning.h"
#include "w4a4_nax.h"
#include "w4a8_nax.h"
#include "omlx_m5/execution_contract.h"
#include "omlx_m5/kv_page_pool.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace omlx::m5 {
using namespace detail::forward;

namespace {

RowExactPlaneSlice to_slice(Ref r) noexcept {
  return {.buffer = r.buffer, .offset = r.offset, .bytes = r.bytes};
}

bool sibling_concat_disabled() noexcept {
  if (const char *env = std::getenv("OMLX_M5_ENABLE_SIBLING_CONCAT");
      env != nullptr && std::string_view(env) == "1") {
    return false;
  }
  return true;
}

Ref row_slice_ref(Ref base, U32 row_offset, U32 row_count, U32 cols,
                  U32 elem_bytes = 2U) {
  const U64 off = mul(U64(row_offset), mul(U64(cols), elem_bytes));
  const U64 len = mul(U64(row_count), mul(U64(cols), elem_bytes));
  if (!base.buffer || off + len > base.bytes) {
    throw std::out_of_range("Qwen packed row slice exceeds workspace buffer");
  }
  return Ref{base.buffer, base.offset + off, len};
}

std::vector<std::byte> words_span(std::span<const U32> data) {
  auto bytes = std::as_bytes(data);
  return {bytes.begin(), bytes.end()};
}

} // namespace

struct Q5High1Cache::Impl {
  void ensure_built(const MetalContext &ctx, const std::filesystem::path &libs,
                    std::span<const MaterializedWeightView> candidates,
                    const AffineNaxCache *affine_cache) {
    if (!std::filesystem::is_regular_file(libs / "omlx_m5_row_exact.metallib")) {
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (affine_cache != nullptr) {
      affine_cache_ = affine_cache;
    }
    struct PendingItem {
      U32 object_id = 0;
      Ref src_high2{};
      U64 h1_bytes = 0;
      NaxSegmentArena::Placement place{};
    };
    struct PendingSbItem {
      U32 object_id = 0;
      Ref src_sb{};
      U32 padded_n = 0;
      U32 groups = 0;
      U32 bn = 16;
      NaxSegmentArena::Placement place{};
    };
    NaxSegmentArena arena;
    std::vector<PendingItem> items;
    std::vector<PendingSbItem> sb_items;
    for (const auto &w : candidates) {
      const U32 bits = affine_nax_bits(w.format);
      if (bits == 0U || w.parent_shape.size() != 2U || w.row_count == 0U ||
          w.in_features < 512U || (w.in_features % 64U) != 0U) {
        continue;
      }
      const U32 bn = affine_nax_bn(U32(w.row_count));
      const U32 padded_n = ((U32(w.row_count) + bn - 1U) / bn) * bn;
      if (bits == 5U && !entries_.contains(w.object_id)) {
        const auto *tw_existing =
            affine_cache ? affine_cache->find(w.object_id) : nullptr;
        if (!(tw_existing && tw_existing->high1.buffer)) {
          Ref src_h2{};
          if (tw_existing && tw_existing->bits == 5U &&
              tw_existing->bn == bn && tw_existing->n == w.row_count &&
              tw_existing->k == w.in_features) {
            src_h2 = tw_existing->high2;
          } else if (is_oq_nax_pretiled_layout(w.layout)) {
            src_h2 = plane(w, "nax_high2");
          }
          const U64 h1_bytes = mul(padded_n, U32(w.in_features) / 8U);
          if (src_h2.buffer && src_h2.bytes >= 2ULL * h1_bytes &&
              (h1_bytes % 4ULL) == 0ULL) {
            items.push_back({w.object_id, src_h2, h1_bytes,
                             arena.plan_alloc(
                                 h1_bytes, "Q5 T=1 compacted 1-bit high plane")});
          }
        }
      }
      if (is_oq_nax_pretiled_layout(w.layout) &&
          !sb_tiled_entries_.contains(w.object_id)) {
        const auto *tw =
            affine_cache ? affine_cache->find(w.object_id) : nullptr;
        if (!(tw && tw->sb_tiled.buffer)) {
          const U32 groups = U32(w.in_features) / 64U;
          const U64 sb_bytes = mul(mul(padded_n, groups), 4U);
          const Ref src_sb = plane(w, "nax_sb");
          if (src_sb.buffer && src_sb.bytes >= sb_bytes) {
            sb_items.push_back(
                {w.object_id, src_sb, padded_n, groups, bn,
                 arena.plan_alloc(sb_bytes, "Affine KL32 sb_tiled")});
          }
        }
      }
    }
    if (items.empty() && sb_items.empty()) {
      return;
    }
    const std::size_t base_seg = segments_.size();
    arena.materialize(ctx, segments_);
    static std::atomic<U64> compact_id{0};
    Encoder enc(ctx, libs, 1U,
                "q5.high1.compact." + std::to_string(++compact_id));
    enc.begin_stage("compact_q5_high1");
    for (const auto &it : items) {
      const Ref dst = NaxSegmentArena::resolve(segments_, it.place, base_seg);
      const U32 num_u32 = U32(it.h1_bytes / 4ULL);
      auto &d = enc.emit("row_exact", "omlx_m5_compact_q5_high2_to_high1",
                         {.width = num_u32}, {.width = 256U},
                         {it.src_high2, dst});
      enc.dynamic(d, 2, words({num_u32}));
      entries_.emplace(it.object_id, dst);
    }
    for (const auto &si : sb_items) {
      const Ref dst = NaxSegmentArena::resolve(segments_, si.place, base_seg);
      auto &d = enc.emit("row_exact", "omlx_m5_transpose_affine_sb_to_tiled",
                         {.width = si.padded_n, .height = si.groups},
                         {.width = 16U, .height = std::min(si.groups, 16U)},
                         {si.src_sb, dst});
      enc.dynamic(d, 2, words({si.groups, si.padded_n, si.bn, 0U}));
      sb_tiled_entries_.emplace(si.object_id, dst);
    }
    auto prof = enc.finish();
    const std::array calls{MetalCommandInvocation{.plan = &prof.plan,
                                                  .bindings = prof.bindings,
                                                  .parameters = &prof.parameters}};
    (void)ctx.execute_batch(calls);
  }

  const Ref *find(U32 object_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = entries_.find(object_id);
    if (it != entries_.end()) {
      return &it->second;
    }
    if (affine_cache_ != nullptr) {
      if (const auto *tw = affine_cache_->find(object_id);
          tw != nullptr && tw->high1.buffer != nullptr) {
        return &tw->high1;
      }
    }
    return nullptr;
  }

  const Ref *find_sb_tiled(U32 object_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = sb_tiled_entries_.find(object_id);
    if (it != sb_tiled_entries_.end()) {
      return &it->second;
    }
    if (affine_cache_ != nullptr) {
      if (const auto *tw = affine_cache_->find(object_id);
          tw != nullptr && tw->sb_tiled.buffer != nullptr) {
        return &tw->sb_tiled;
      }
    }
    return nullptr;
  }

  const AffineNaxCache *affine_cache() const {
    std::lock_guard<std::mutex> lk(mu_);
    return affine_cache_;
  }

  mutable std::mutex mu_;
  const AffineNaxCache *affine_cache_ = nullptr;
  std::vector<std::unique_ptr<MetalBuffer>> segments_;
  std::unordered_map<U32, Ref> entries_;
  std::unordered_map<U32, Ref> sb_tiled_entries_;
};

Q5High1Cache::Q5High1Cache() : impl_(std::make_unique<Impl>()) {}
Q5High1Cache::~Q5High1Cache() = default;

std::optional<RowExactPlaneSlice>
Q5High1Cache::find_q5_high1(std::uint32_t object_id) const {
  if (!impl_)
    return std::nullopt;
  if (const Ref *r = impl_->find(object_id); r && r->buffer) {
    return to_slice(*r);
  }
  return std::nullopt;
}

std::optional<RowExactPlaneSlice>
Q5High1Cache::find_sb_tiled(std::uint32_t object_id) const {
  if (!impl_)
    return std::nullopt;
  if (const Ref *r = impl_->find_sb_tiled(object_id); r && r->buffer) {
    return to_slice(*r);
  }
  return std::nullopt;
}

namespace {

std::shared_ptr<Q5High1Cache>
acquire_q5_high1_cache(const void *owner_key, const MetalContext &ctx,
                       const std::filesystem::path &libs,
                       std::span<const MaterializedWeightView> candidates,
                       const AffineNaxCache *affine_cache) {
  auto cache =
      nax_acquire_owner_cache<Q5High1Cache>(owner_key, [](Q5High1Cache &) {});
  cache->impl_->ensure_built(ctx, libs, candidates, affine_cache);
  return cache;
}

} // namespace

std::shared_ptr<Q5High1Cache>
Q5High1Cache::acquire(const void *owner_key, const MetalContext &ctx,
                      const std::filesystem::path &libs,
                      std::span<const MaterializedWeightView> candidates) {
  return acquire_q5_high1_cache(owner_key, ctx, libs, candidates, nullptr);
}

std::optional<ResolvedKl32Mxfp4Planes>
resolve_pretiled_kl32_mxfp4_planes(const MaterializedWeightView &w) {
  if (w.format != "mxfp4_e2m1_g32_e8m0_v1" || w.parent_shape.size() != 2U ||
      w.row_count == 0U || w.in_features < 512U ||
      (w.in_features % 128U) != 0U) {
    return std::nullopt;
  }
  if (w.layout == "mx_nax_tiled_bk128_v1") {
    const auto d = try_plane(w, "nax_data");
    const auto s = try_plane(w, "nax_scale");
    if (d && s && d->buffer && s->buffer) {
      return ResolvedKl32Mxfp4Planes{
          .codes = to_slice(*d),
          .scales = to_slice(*s),
          .n = U32(w.row_count),
          .k = U32(w.in_features),
          .bn = mxfp4_nax_bn(U32(w.row_count)),
          .has_fwht64 = w.has_fwht64(),
      };
    }
  }
  return std::nullopt;
}

std::optional<ResolvedKl32AffinePlanes>
resolve_pretiled_kl32_affine_planes(const MaterializedWeightView &w,
                                    const Q5High1Cache *cache,
                                    RowExactPlaneSlice zero_plane) {
  const U32 bits = affine_nax_bits(w.format);
  if (bits == 0U || w.parent_shape.size() != 2U || w.row_count == 0U ||
      w.in_features < 512U || (w.in_features % 64U) != 0U || !zero_plane) {
    return std::nullopt;
  }
  const U32 exp_bn = affine_nax_bn(U32(w.row_count));
  std::optional<RowExactPlaneSlice> q5_h1;
  if (bits == 5U) {
    if (!cache)
      return std::nullopt;
    q5_h1 = cache->find_q5_high1(w.object_id);
    if (!q5_h1)
      return std::nullopt;
  }
  if (cache && cache->impl_) {
    if (const AffineNaxCache *aff = cache->impl_->affine_cache();
        aff != nullptr) {
      if (const auto *tw = aff->find(w.object_id);
          tw && tw->n == w.row_count && tw->k == w.in_features &&
          tw->bn == exp_bn && tw->bits == bits && tw->sb_tiled.buffer) {
        const RowExactPlaneSlice high_sl =
            (bits == 5U) ? *q5_h1
            : affine_nax_has_high_plane(bits) ? to_slice(tw->high2)
                                              : zero_plane;
        return ResolvedKl32AffinePlanes{
            .codes = to_slice(tw->codes),
            .high = high_sl,
            .sb_tiled = to_slice(tw->sb_tiled),
            .n = U32(w.row_count),
            .k = U32(w.in_features),
            .bn = exp_bn,
            .bits = bits,
            .is_lane_v2 = true,
            .has_fwht64 = w.has_fwht64(),
        };
      }
    }
  }
  if (is_oq_nax_pretiled_layout(w.layout)) {
    if (!cache)
      return std::nullopt;
    const auto sb_tl = cache->find_sb_tiled(w.object_id);
    const auto codes = try_plane(w, "nax_codes");
    if (!sb_tl || !codes || !codes->buffer)
      return std::nullopt;
    RowExactPlaneSlice high_sl = zero_plane;
    if (bits == 5U) {
      high_sl = *q5_h1;
    } else if (bits == 3U) {
      const auto h1 = try_plane(w, "nax_high1");
      if (!h1 || !h1->buffer)
        return std::nullopt;
      high_sl = to_slice(*h1);
    } else if (bits == 6U) {
      const auto h2 = try_plane(w, "nax_high2");
      if (!h2 || !h2->buffer)
        return std::nullopt;
      high_sl = to_slice(*h2);
    }
    return ResolvedKl32AffinePlanes{
        .codes = to_slice(*codes),
        .high = high_sl,
        .sb_tiled = *sb_tl,
        .n = U32(w.row_count),
        .k = U32(w.in_features),
        .bn = exp_bn,
        .bits = bits,
        .is_lane_v2 = is_oq_nax_lane_v2_layout(w.layout),
        .has_fwht64 = w.has_fwht64(),
    };
  }
  return std::nullopt;
}

U32 resolve_qwen_primary_workspace_rows(U32 max_context, bool inspect_tensors,
                                        bool sparse_moe) noexcept {
  U32 cap = inspect_tensors ? 1024U : (sparse_moe ? 4096U : 2048U);
  if (const char *env = std::getenv("OMLX_M5_PREFILL_CHUNK_SIZE")) {
    const unsigned long parsed = std::strtoul(env, nullptr, 10);
    if (parsed >= 64UL && parsed <= 4096UL) {
      cap = static_cast<U32>(parsed);
    }
  }
  if (inspect_tensors && cap > 1024U) {
    cap = 1024U;
  }
  return std::min<U32>(max_context, cap);
}

class QwenForward::Impl {
public:
  static U64 new_owner_id() {
    static std::atomic<U64> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
  }
  const U64 owner_id = new_owner_id();
  std::string execution_identity;
  std::string shared_contract_identity;
  std::string base_execution_identity_;
  std::string base_shared_contract_identity_;
  std::string bound_context_identity_;
  bool execution_identity_bound = false, execution_started = false;

  Impl(const MetalContext &context, const MaterializedModel &weights,
       const QwenLayerCatalog &layers, QwenModelConfig config, U32 max_context,
       const std::filesystem::path &libraries, std::span<const U32> taps,
       bool inspect_routes, bool speculative, bool enable_m4_short_routes,
       std::optional<QwenLinearRouteOverride> diagnostic_override,
       bool enable_attention_split, bool enable_batch,
       std::optional<QwenMoeRouteOverride> moe_override,
        LinearNumericPolicy linear_policy, SamplingContract sampling_contract,
       omlx::m5::AttentionNumericPolicy attention_policy,
       bool inspect_tensors_workspace, std::string_view kv_dtype,
       QwenForward *shared_primary, bool enable_flash_nax_prefill,
       U32 kv_pool_pages = 0U,
       std::shared_ptr<const QwenMergedLoraOverrides> merged_lora = nullptr,
       std::shared_ptr<const QwenBypassLoraRuntime> bypass_lora = nullptr,
       QwenPrefillOptions prefill_options = {})
      : context(context), weights(weights), config(std::move(config)),
        max_context(max_context), libraries(libraries),
        silicon_tuning(detail::resolve_silicon_tuning(context)),
        kv_format_(parse_kv_storage_format(kv_dtype)),
        shared_primary_(shared_primary),
        workspace_rows_(shared_primary
                            ? std::min<U32>(max_context, 16U)
                            : resolve_qwen_primary_workspace_rows(
                                  max_context, inspect_tensors_workspace,
                                  this->config.num_experts > 1)),
        feature_layers(taps.begin(), taps.end()),
        inspect_routes_enabled(inspect_routes),
        inspect_tensors_workspace(inspect_tensors_workspace),
        speculative_enabled(speculative),
        m4_short_routes(enable_m4_short_routes), linear_policy(linear_policy),
        sampling_contract(sampling_contract),
        attention_policy(attention_policy),
        strict_two_pass_attn_(
            detail::forward::strict_two_pass_attention_enabled()),
        flash_nax_prefill_(
            (attention_policy != omlx::m5::AttentionNumericPolicy::RowExactV1 ||
             this->config.hidden_size >= 1024) &&
            !detail::forward::env_flag_enabled(
                "OMLX_M5_DISABLE_FLASH_NAX_PREFILL") &&
            (enable_flash_nax_prefill ||
             detail::forward::flash_nax_prefill_enabled_by_env())),
        split_attention_enabled(enable_attention_split),
        batch_enabled(enable_batch),
        diagnostic_override(std::move(diagnostic_override)),
        moe_override(std::move(moe_override)),
        prefill_options_(std::move(prefill_options)) {
    merged_lora_ = std::move(merged_lora);
    bypass_lora_ = std::move(bypass_lora);
    (void)sampling_contract_name(sampling_contract);
    if (attention_policy != omlx::m5::AttentionNumericPolicy::Legacy &&
        attention_policy != omlx::m5::AttentionNumericPolicy::RowExactV1)
      throw std::invalid_argument("unknown Attention numeric policy");
    if (this->moe_override &&
        context.capabilities().profile.probe.apple_gpu_family < 9)
      throw std::invalid_argument("MoE A/B override requires Apple GPU family 9+");
    if (this->moe_override &&
        (this->moe_override->object.empty() ||
         this->moe_override->format.empty() ||
         this->moe_override->layout.empty() || !this->moe_override->n ||
         !this->moe_override->k || !this->moe_override->capacity ||
         (this->moe_override->choice != "grouped" &&
          this->moe_override->choice != "fallback")))
      throw std::invalid_argument("MoE A/B override key/choice invalid");
    if (this->diagnostic_override &&
        context.capabilities().profile.probe.apple_gpu_family < 9)
      throw std::invalid_argument(
          "linear A/B override requires Apple GPU family 9+");
    const auto &c = this->config;
    if (enable_attention_split &&
        ((context.capabilities().profile.probe.apple_gpu_family < 9 &&
          attention_policy == omlx::m5::AttentionNumericPolicy::Legacy) ||
         c.full_attention_layers() == 0))
      throw std::invalid_argument(
          "split Attention diagnostic requires Apple GPU family 9+ Attention");
    if (c.num_experts != 0 &&
        (c.num_experts != 256 || c.num_experts_per_token != 8 ||
         c.shared_expert_intermediate_size <= 0 ||
         c.shared_expert_intermediate_size % 64))
      throw std::invalid_argument(
          "Qwen MoE requires 256 experts, top-8 and aligned shared FFN");
    if (c.hidden_size <= 0 || c.vocab_size <= 0 || c.intermediate_size <= 0 ||
        c.num_hidden_layers != static_cast<int>(layers.layers().size()) ||
        c.hidden_size % 64 || c.intermediate_size % 64 || max_context == 0 ||
        c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 ||
        c.num_attention_heads % c.num_key_value_heads || c.head_dim <= 0 ||
        c.head_dim > 256 || c.linear_num_key_heads <= 0 ||
        c.linear_num_value_heads <= 0 || c.linear_key_head_dim <= 0 ||
        c.linear_value_head_dim <= 0 || c.linear_conv_kernel_dim <= 1 ||
        c.linear_num_value_heads % c.linear_num_key_heads)
      throw std::invalid_argument("unsupported Qwen execution geometry");

    h = U32(c.hidden_size);
    f = U32(c.intermediate_size);
    v = U32(c.vocab_size);
    q = U32(c.num_attention_heads * c.head_dim);
    kv = U32(c.num_key_value_heads * c.head_dim);
    gk = U32(c.linear_num_key_heads * c.linear_key_head_dim);
    gv = U32(c.linear_num_value_heads * c.linear_value_head_dim);
    conv = 2 * gk + gv;

    const bool affine4_kv = (kv_format_ == KvStorageFormat::Affine4Group128);
    const bool w4a4_keep_qk_a8 =
        (prefill_options_.linear_mode == PrefillLinearMode::W4A4 &&
         w4a4_keep_qk_a8_enabled());
    const bool enable_w4a4_ws =
        (prefill_options_.linear_mode == PrefillLinearMode::W4A4);
    const bool enable_w4a8_ws =
        (prefill_options_.linear_mode == PrefillLinearMode::W4A8 ||
         w4a4_keep_qk_a8);
    const bool enable_minf_ws =
        (shared_primary_ == nullptr &&
         prefill_options_.attention_mode == PrefillAttentionMode::Sage2 &&
         prefill_options_.sparse_minference && c.full_attention_layers() > 0 &&
         c.head_dim == 256);
    const bool enable_sage2_kv_ws =
        (shared_primary_ == nullptr &&
         prefill_options_.attention_mode == PrefillAttentionMode::Sage2 &&
         c.full_attention_layers() > 0 && c.head_dim == 256);
    allocation_plan = plan_qwen_forward_memory(
        c, max_context, U32(feature_layers.size()), inspect_routes, speculative,
        enable_attention_split,
        sampling_contract == SamplingContract::PositionCoupledV1,
        attention_policy == omlx::m5::AttentionNumericPolicy::RowExactV1,
        inspect_tensors_workspace, 0U, workspace_rows_, affine4_kv, 0U,
        flash_nax_prefill_, enable_w4a8_ws, enable_minf_ws,
        enable_sage2_kv_ws, enable_w4a4_ws);
    feature_ring_rows_ = allocation_plan.feature_ring_rows;

    if (allocation_plan.largest_buffer_bytes >
        context.capabilities().max_buffer_length)
      throw std::invalid_argument("Qwen planned buffer exceeds device limit");

    token_ids = allocate("token_ids");
    hidden = allocate("hidden");
    norm = allocate("norm");
    query = allocate("query");
    key = allocate("key");
    value = allocate("value");
    qkv = allocate("qkv");
    z = allocate("z");
    a = allocate("a");
    b = allocate("b");
    convolved = allocate("convolved");
    recurrent = allocate("recurrent");
    mixer = allocate("mixer");
    if (enable_attention_split) {
      split_max = allocate("attention_split_max");
      const bool bf16_two_pass =
          attention_policy == omlx::m5::AttentionNumericPolicy::RowExactV1 &&
          !affine4_kv && strict_two_pass_attn_;
      if (!bf16_two_pass) {
        split_sum = allocate("attention_split_sum");
        split_output = allocate("attention_split_output");
      }
    }
    delta = allocate("delta");
    gate = allocate("gate");
    up = allocate("up");
    product = allocate("product");
    sums = allocate("sums");
    if (c.num_experts) {
      router = allocate("router");
      route_ids = allocate("route_ids");
      route_scores = allocate("route_scores");
      expert_output = allocate("expert_output");
      routed = allocate("routed");
      shared = allocate("shared");
      shared_gate = allocate("shared_gate");
    }
    logits = allocate("logits");
    last_logits = allocate("last_logits");
    sampled = allocate("sampled");
    if (sampling_contract == SamplingContract::PositionCoupledV1) {
      position_ids = allocate("position_sample_ids");
      position_scores = allocate("position_sample_scores");
    }
    for (const auto layer : feature_layers)
      if (layer >= U32(c.num_hidden_layers))
        throw std::invalid_argument("Qwen feature layer outside model");
    if (!feature_layers.empty())
      features = allocate("features");
    if (speculative && !feature_layers.empty())
      feature_history = allocate("feature_history");
    if (workspace_rows_ >= 64U && c.linear_attention_layers() > 0 &&
        c.linear_key_head_dim == 128 && c.linear_value_head_dim == 128)
      gdn_chunk_ws = allocate("qwen.workspace.gdn_chunk");
    if (enable_w4a8_ws) {
      a8_qa = allocate_private("a8_qa");
      a8_sa = allocate_private("a8_sa");
      a8_ra = allocate_private("a8_ra");
      a8_workspace_mp_ = w4a8_nax_mp(workspace_rows_);
    }
    if (enable_w4a4_ws) {
      w4a4_codes = allocate_private("w4a4_codes");
      w4a4_scales = allocate_private("w4a4_scales");
      w4a4_workspace_mp_ = w4a4_nax_mp(workspace_rows_);
    }
    if (enable_minf_ws) {
      minference_scratch = allocate("minference_scratch");
    }
    if (enable_sage2_kv_ws) {
      sage2_kv_ws_ = allocate_private("sage2_kv_ws");
      sage2_kv_ws_layout_ =
          plan_sage2_kv_workspace(U32(c.num_key_value_heads), max_context);
    }
    zero = allocate("zero");
    const_cast<MetalBuffer *>(zero.buffer)->zero();
    argmax_scratch = allocate_private("argmax_partials");
    rms_sq_buf = allocate_private("rms_sq_buf");
    moe_tg_done_buf = allocate_private("moe_tg_done_buf");
    if (c.full_attention_layers() > 0) {
      if (attention_policy == omlx::m5::AttentionNumericPolicy::RowExactV1 ||
          affine4_kv || flash_nax_prefill_) {
        attention_positions = allocate("attention_positions");
      }
      if (!allocation_names.insert("kv_page_table").second)
        throw std::logic_error("duplicate Qwen KV page table");
      const U32 logical_kv_pages = (max_context + 63U) / 64U;
      const U64 kv_page_b = kv_page_bytes(
          kv_format_, 64U, U32(c.num_key_value_heads), U32(c.head_dim));
      if (shared_primary_ != nullptr && shared_primary_->impl_->page_pool &&
          shared_primary_->impl_->page_pool->shared_allocator()) {
        page_pool = std::make_unique<KvPagePool>(
            context, max_context, 64,
            shared_primary_->impl_->page_pool->shared_allocator(), kv_format_);
      } else {
        U32 eff_kv_pages = logical_kv_pages + 1U;
        if (const char *env_tok = std::getenv("OMLX_M5_KV_POOL_TOKENS")) {
          const U64 req_tok = std::strtoull(env_tok, nullptr, 10);
          if (req_tok > 0) {
            eff_kv_pages = std::max<U32>(logical_kv_pages + 1U,
                                         U32((req_tok + 63ULL) / 64ULL));
          }
        } else if (kv_pool_pages > 0U) {
          eff_kv_pages = std::max<U32>(logical_kv_pages + 1U, kv_pool_pages);
        }
        if (kv_page_b > 0 && eff_kv_pages > logical_kv_pages + 1U) {
          const U64 max_safe_pages =
              std::min<U64>(context.capabilities().max_buffer_length,
                            8ULL << 30) /
              kv_page_b;
          if (max_safe_pages >= logical_kv_pages + 1U) {
            eff_kv_pages = std::min<U32>(eff_kv_pages, U32(max_safe_pages));
          }
        }
        page_pool = std::make_unique<KvPagePool>(
            context, max_context, 64, eff_kv_pages, kv_format_);
      }
    }
    if (inspect_routes && c.num_experts) {
      route_trace = allocate("route_trace");
      router_logits_trace = allocate("router_logits_trace");
      router_scores_trace = allocate("router_scores_trace");
    }
    token_staging.resize(workspace_rows_);

    for (const auto &stage : layers.global_stages()) {
      if (stage.role == "embedding")
        embedding = weights.parent_view(stage.object_id);
      if (stage.role == "final_norm")
        final_norm = weights.parent_view(stage.object_id);
      if (stage.role == "lm_head")
        head = weights.parent_view(stage.object_id);
    }
    std::optional<bool> source_norm_offset;
    const U32 logical_kv_pages_plan = (max_context + 63U) / 64U;
    const U64 kv_page_b_plan = kv_page_bytes(
        kv_format_, 64U, U32(c.num_key_value_heads), U32(c.head_dim));
    for (const auto &layer : layers.layers()) {
      Block block{.type = layer.type, .layer_index = layer.layer};
      for (const auto &stage : layer.stages)
        if (stage.object_id != UINT32_MAX)
          block.operands.emplace(stage.role,
                                 weights.parent_view(stage.object_id));
      if (layer.type == LayerType::full_attention) {
        const std::string s0_name = "state." + std::to_string(layer.layer) + ".0";
        const std::string s1_name = "state." + std::to_string(layer.layer) + ".1";
        if (shared_primary_ != nullptr &&
            blocks.size() < shared_primary_->impl_->blocks.size() &&
            page_pool && shared_primary_->impl_->page_pool &&
            page_pool->shared_allocator() ==
                shared_primary_->impl_->page_pool->shared_allocator()) {
          if (!allocation_names.insert(s0_name).second ||
              !allocation_names.insert(s1_name).second)
            throw std::logic_error("duplicate Qwen shared KV state");
          const auto &pb = shared_primary_->impl_->blocks[blocks.size()];
          block.state0 = pb.state0;
          block.state1 = pb.state1;
          const U64 orig_slab =
              mul(U64(logical_kv_pages_plan + 1U), kv_page_b_plan) * 2ULL;
          if (allocation_plan.state_bytes >= orig_slab)
            allocation_plan.state_bytes -= orig_slab;
        } else {
          const U64 kv_slab_bytes =
              mul(U64(page_pool->physical_pages()), kv_page_b_plan);
          block.state0 = allocate_sized(s0_name, kv_slab_bytes);
          block.state1 = allocate_sized(s1_name, kv_slab_bytes);
          const U64 orig_slab =
              mul(U64(logical_kv_pages_plan + 1U), kv_page_b_plan);
          if (kv_slab_bytes > orig_slab) {
            allocation_plan.state_bytes += 2ULL * (kv_slab_bytes - orig_slab);
            allocation_plan.largest_buffer_bytes =
                std::max(allocation_plan.largest_buffer_bytes, kv_slab_bytes);
          }
        }
      } else {
        const auto &shape = block.get("gdn_conv").parent_shape;
        const auto kernel = U64(c.linear_conv_kernel_dim);
        const bool hf_layout = shape == std::vector<U64>{conv, 1, kernel};
        const bool mlx_layout = shape == std::vector<U64>{conv, kernel, 1};
        if ((!hf_layout && !mlx_layout) ||
            (source_norm_offset && *source_norm_offset != hf_layout))
          throw std::invalid_argument(
              "Qwen source convolution/norm conventions disagree");
        source_norm_offset = hf_layout;
        block.state0 = allocate("state." + std::to_string(layer.layer) + ".0");
        block.state1 = allocate("state." + std::to_string(layer.layer) + ".1");
      }
      if (speculative && layer.type != LayerType::full_attention) {
        const auto prefix = "state." + std::to_string(layer.layer);
        block.history = allocate(prefix + ".history");
        block.replay_qkv = allocate(prefix + ".qkv");
        block.replay_keys = allocate(prefix + ".keys");
        block.replay_decays = allocate(prefix + ".decays");
        block.replay_deltas = allocate(prefix + ".deltas");
      }
      blocks.push_back(std::move(block));
    }
    if (embedding.name.empty() || final_norm.name.empty() || head.name.empty())
      throw std::invalid_argument("Qwen global parameters missing");
    if (merged_lora_ != nullptr &&
        !merged_lora_->views_by_object_id.empty()) {
      if (const auto it =
              merged_lora_->views_by_object_id.find(head.object_id);
          it != merged_lora_->views_by_object_id.end()) {
        head = it->second;
      }
      for (auto &block : blocks) {
        for (auto &[role, w] : block.operands) {
          (void)role;
          if (const auto it =
                  merged_lora_->views_by_object_id.find(w.object_id);
              it != merged_lora_->views_by_object_id.end()) {
            w = it->second;
          }
        }
      }
    }
    if (!source_norm_offset.has_value()) {
      const auto is_delta_norm_family = [](std::string_view type) {
        return type.starts_with("qwen3_5") || type.starts_with("qwen3_6") ||
               type.starts_with("qwen3_8") || type.starts_with("qwen3.5") ||
               type.starts_with("qwen3.6") || type.starts_with("qwen3.8");
      };
      const bool has_model_type =
          !c.model_type.empty() || !c.text_model_type.empty();
      const bool delta_family = is_delta_norm_family(c.model_type) ||
                                is_delta_norm_family(c.text_model_type);
      if (has_model_type && !delta_family) {
        source_norm_offset = false;
      } else {
        const auto norm_mean = [&](const MaterializedWeightView &w, U64 count) {
          const auto ref = direct(w, count);
          const auto raw = ref.buffer->read(ref.offset, ref.bytes);
          double total = 0.0;
          for (U64 i = 0; i < count; ++i) {
            std::uint16_t code = 0;
            std::memcpy(&code, raw.data() + 2 * i, 2);
            const float val = std::bit_cast<float>(U32(code) << 16);
            if (!std::isfinite(val))
              throw std::invalid_argument("Qwen RMSNorm weight is non-finite");
            total += double(val);
          }
          return total / double(count);
        };
        const double mean_final = norm_mean(final_norm, h);
        const double mean_input =
            norm_mean(blocks.front().get("input_norm"), h);
        const bool final_zero_centered =
            std::abs(mean_final) < std::abs(mean_final - 1.0);
        const bool input_zero_centered =
            std::abs(mean_input) < std::abs(mean_input - 1.0);
        if (final_zero_centered != input_zero_centered)
          throw std::invalid_argument(
              "Qwen source RMSNorm conventions disagree");
        source_norm_offset = final_zero_centered;
      }
    }
    normalization_weight_offset =
        source_norm_offset.value_or(false) ? 1.0F : 0.0F;
    if (allocation_names.size() != allocation_plan.buffers.size())
      throw std::logic_error("Qwen allocation plan not fully consumed");

    // Private packed verify logits workspace on primary slot (for B>=2 verify M>8).
    if (!shared_primary_ && batch_enabled && speculative &&
        logits.bytes < mul(64U, mul(v, 2U))) {
      const U64 pv_bytes = mul(64U, mul(v, 2U));
      auto pv_buf =
          std::make_unique<MetalBuffer>(context.make_private_buffer(pv_bytes));
      packed_verify_logits = Ref{pv_buf.get(), 0, pv_bytes};
      track_extra_workspace("packed_verify_logits", *pv_buf);
      buffers.push_back(std::move(pv_buf));
    }

    bf16_nax_bypass_ok_ = bf16_nax_runtime_eligible(context, libraries);
    bf16_nax_enabled =
        linear_policy == LinearNumericPolicy::RowExactPortableV1 &&
        bf16_nax_bypass_ok_;
    if (config.num_experts &&
        (linear_policy == LinearNumericPolicy::RowExactPortableV1 ||
         linear_policy == LinearNumericPolicy::Legacy) &&
        moe_nax_runtime_eligible(context, libraries) && !blocks.empty() &&
        moe_nax_weight_eligible(blocks.front().get("moe_expert_gate"),
                                blocks.front().get("moe_expert_up"),
                                blocks.front().get("moe_expert_down"), h, f)) {
      const std::size_t moe_buf_start = buffers.size();
      moe_nax_ws.allocate(context, workspace_rows_, h, f, buffers,
                          U32(config.num_experts),
                          U32(config.num_experts_per_token));
      for (std::size_t bi = moe_buf_start; bi < buffers.size(); ++bi) {
        track_extra_workspace("moe_nax_ws_" + std::to_string(bi - moe_buf_start),
                              *buffers[bi]);
      }
    }

    std::vector<MaterializedWeightView> candidates;
    bool has_pretiled_mxfp4 = (head.layout == "mx_nax_tiled_bk128_v1");
    bool has_pretiled_affine = is_oq_nax_pretiled_layout(head.layout);
    bool any_fwht64 = head.has_fwht64();
    candidates.push_back(head);
    for (const auto &block : blocks) {
      for (const auto &[role, w] : block.operands) {
        (void)role;
        has_pretiled_mxfp4 |= (w.layout == "mx_nax_tiled_bk128_v1");
        has_pretiled_affine |= is_oq_nax_pretiled_layout(w.layout);
        any_fwht64 |= w.has_fwht64();
        candidates.push_back(w);
      }
    }

    has_row_exact_metallib_ =
        std::filesystem::is_regular_file(libraries / "omlx_m5_row_exact.metallib");
    const void *cache_owner_key =
        merged_lora_ ? static_cast<const void *>(merged_lora_.get())
                     : static_cast<const void *>(&weights);

    const bool w4a4_requested =
        (prefill_options_.linear_mode == PrefillLinearMode::W4A4 &&
         w4a4_codes.buffer != nullptr &&
         w4a4_nax_runtime_eligible(context, libraries));
    w4a4_min_tokens_ = w4a4_nax_min_tokens();
    w4a4_keep_qk_a8_ = w4a4_keep_qk_a8;
    w4a4_down_proj_ = []() {
      const char *env = std::getenv("OMLX_M5_W4A4_DOWN_BF16");
      return !(env != nullptr && std::string_view(env) == "1");
    }();
    const bool w4a8_requested =
        ((prefill_options_.linear_mode == PrefillLinearMode::W4A8 ||
          w4a4_keep_qk_a8) &&
         a8_qa.buffer != nullptr &&
         w4a8_nax_runtime_eligible(context, libraries));
    w4a8_min_tokens_ = w4a8_nax_min_tokens();
    w4a8_down_proj_ = w4a8_down_proj_enabled();

    if (mxfp4_nax_runtime_eligible(context, libraries) &&
        (linear_policy == LinearNumericPolicy::RowExactPortableV1 ||
         w4a8_requested || w4a4_requested || mxfp4_w4a4_prefill_enabled() ||
         has_pretiled_mxfp4 ||
         (t_domain_active() && pretiled_qmv_t1_enabled()))) {
      mxfp4_nax_cache = Mxfp4NaxCache::acquire(
          cache_owner_key, context, libraries, candidates,
          silicon_tuning.linear_split_k_target_slices);
      if (mxfp4_nax_cache && !sibling_concat_disabled()) {
        std::lock_guard<std::recursive_mutex> cache_lk(
            mxfp4_nax_cache->build_mutex());
        bool any_concat = false;
        for (auto &block : blocks) {
          if (block.type == LayerType::full_attention) {
            const std::array<MaterializedWeightView, 3> parts{
                block.get("attention_q"), block.get("attention_k"),
                block.get("attention_v")};
            const U32 cid = 0x80000000U | (block.layer_index * 4U + 0U);
            any_concat |= mxfp4_nax_cache->register_concat(cid, parts);
            block.concat_qkv_id = cid;
          } else {
            const std::array<MaterializedWeightView, 2> qkvz_parts{
                block.get("gdn_qkv"), block.get("gdn_z")};
            const U32 cid_qkvz = 0x80000000U | (block.layer_index * 4U + 1U);
            any_concat |= mxfp4_nax_cache->register_concat(cid_qkvz, qkvz_parts);
            block.concat_gdn_qkvz_id = cid_qkvz;

            const std::array<MaterializedWeightView, 2> ab_parts{
                block.get("gdn_a"), block.get("gdn_b")};
            const U32 cid_ab = 0x80000000U | (block.layer_index * 4U + 2U);
            any_concat |= mxfp4_nax_cache->register_concat(cid_ab, ab_parts);
            block.concat_gdn_ab_id = cid_ab;
          }
        }
        if (any_concat) {
          mxfp4_nax_cache->build(context, libraries);
        }
      }
      if (mxfp4_nax_cache && mxfp4_nax_cache->active()) {
        const U32 max_mp = mxfp4_w4a4_mp(workspace_rows_);
        const U32 max_k = std::max(
            {h, f, q, gv, U32(config.shared_expert_intermediate_size)});
        const U64 f16_norm_bytes = mul(mul(max_mp, h), 2u);
        const U64 f16_bytes = mul(mul(max_mp, max_k), 2u);
        const U64 rs_bytes = mul(max_mp, 4u);
        auto f16_norm_buf = std::make_unique<MetalBuffer>(
            context.make_private_buffer(f16_norm_bytes));
        auto f16_buf =
            std::make_unique<MetalBuffer>(context.make_private_buffer(f16_bytes));
        auto rs_buf =
            std::make_unique<MetalBuffer>(context.make_private_buffer(rs_bytes));
        auto ones_buf =
            std::make_unique<MetalBuffer>(context.make_buffer(rs_bytes));
        std::vector<float> ones_init(max_mp, 1.0f);
        ones_buf->write(std::as_bytes(std::span(ones_init)));
        f16_norm = Ref{f16_norm_buf.get(), 0, f16_norm_bytes};
        f16_scratch = Ref{f16_buf.get(), 0, f16_bytes};
        f16_row_scale = Ref{rs_buf.get(), 0, rs_bytes};
        f16_ones_scale = Ref{ones_buf.get(), 0, rs_bytes};
        track_extra_workspace("mxfp4_nax_f16_norm", *f16_norm_buf);
        track_extra_workspace("mxfp4_nax_f16_ws", *f16_buf);
        track_extra_workspace("mxfp4_nax_f16_row_scale", *rs_buf);
        track_extra_workspace("mxfp4_nax_f16_ones_scale", *ones_buf);
        buffers.push_back(std::move(f16_norm_buf));
        buffers.push_back(std::move(f16_buf));
        buffers.push_back(std::move(rs_buf));
        buffers.push_back(std::move(ones_buf));
        if (mxfp4_w4a4_prefill_enabled() ||
            linear_policy == LinearNumericPolicy::Legacy) {
          const U64 a4_code_bytes = mul(max_mp, max_k / 2u);
          const U64 a4_scale_bytes = mul(max_mp, max_k / 32u);
          auto a4_c_buf = std::make_unique<MetalBuffer>(
              context.make_private_buffer(a4_code_bytes));
          auto a4_s_buf = std::make_unique<MetalBuffer>(
              context.make_private_buffer(a4_scale_bytes));
          a4_codes = Ref{a4_c_buf.get(), 0, a4_code_bytes};
          a4_scales = Ref{a4_s_buf.get(), 0, a4_scale_bytes};
          track_extra_workspace("mxfp4_nax_a4_codes", *a4_c_buf);
          track_extra_workspace("mxfp4_nax_a4_scales", *a4_s_buf);
          buffers.push_back(std::move(a4_c_buf));
          buffers.push_back(std::move(a4_s_buf));
        }
      }
    }

    if (affine_nax_runtime_eligible(context, libraries) &&
        (linear_policy == LinearNumericPolicy::RowExactPortableV1 ||
         w4a8_requested || has_pretiled_affine ||
         (t_domain_active() && pretiled_qmv_t1_enabled()))) {
      affine_nax_cache = AffineNaxCache::acquire(
          cache_owner_key, context, libraries, zero, candidates,
          silicon_tuning.linear_split_k_target_slices);
      if (affine_nax_cache && !sibling_concat_disabled()) {
        std::lock_guard<std::recursive_mutex> cache_lk(
            affine_nax_cache->build_mutex());
        bool any_concat = false;
        for (auto &block : blocks) {
          if (block.type == LayerType::full_attention) {
            const std::array<MaterializedWeightView, 3> parts{
                block.get("attention_q"), block.get("attention_k"),
                block.get("attention_v")};
            const U32 cid = 0x80000000U | (block.layer_index * 4U + 0U);
            any_concat |= affine_nax_cache->register_concat(cid, parts);
            block.concat_qkv_id = cid;
          } else {
            const std::array<MaterializedWeightView, 2> qkvz_parts{
                block.get("gdn_qkv"), block.get("gdn_z")};
            const U32 cid_qkvz = 0x80000000U | (block.layer_index * 4U + 1U);
            any_concat |= affine_nax_cache->register_concat(cid_qkvz, qkvz_parts);
            block.concat_gdn_qkvz_id = cid_qkvz;

            const std::array<MaterializedWeightView, 2> ab_parts{
                block.get("gdn_a"), block.get("gdn_b")};
            const U32 cid_ab = 0x80000000U | (block.layer_index * 4U + 2U);
            any_concat |= affine_nax_cache->register_concat(cid_ab, ab_parts);
            block.concat_gdn_ab_id = cid_ab;
          }
        }
        if (any_concat) {
          affine_nax_cache->build(context, libraries, zero);
        }
      }
      if (affine_nax_cache && affine_nax_cache->active()) {
        const U32 max_mp = affine_nax_mp(workspace_rows_);
        const U32 max_k = std::max(
            {h, f, q, gv, U32(config.shared_expert_intermediate_size)});
        const U64 xs_bytes = mul(mul(max_mp, max_k / 64u), 4u);
        auto xs_buf =
            std::make_unique<MetalBuffer>(context.make_private_buffer(xs_bytes));
        affine_nax_xs = Ref{xs_buf.get(), 0, xs_bytes};
        track_extra_workspace("affine_nax_xs", *xs_buf);
        buffers.push_back(std::move(xs_buf));
      }
    }

    if (any_fwht64) {
      if (f16_scratch.buffer != nullptr) {
        fwht_scratch = f16_scratch;
      } else {
        const U32 max_k = std::max(
            {h, f, q, gv, U32(config.shared_expert_intermediate_size)});
        const U64 fw_bytes = mul(mul(workspace_rows_, max_k), 2ULL);
        auto fw_buf = std::make_unique<MetalBuffer>(
            context.make_private_buffer(fw_bytes));
        fwht_scratch = Ref{fw_buf.get(), 0, fw_bytes};
        track_extra_workspace("fwht64_bf16_scratch", *fw_buf);
        buffers.push_back(std::move(fw_buf));
      }
    }
    if (w4a8_requested && any_fwht64) {
      const U32 max_mp = w4a8_nax_mp(workspace_rows_);
      const U32 max_mid_k =
          std::max({f, U32(config.shared_expert_intermediate_size)});
      if (max_mid_k > 0U && (max_mid_k % 64U) == 0U) {
        const U64 qa_bytes = mul(max_mp, max_mid_k);
        const U64 s_bytes = mul(mul(max_mp, max_mid_k / 64U), 4ULL);
        auto qa_buf = std::make_unique<MetalBuffer>(
            context.make_private_buffer(qa_bytes));
        auto sa_buf = std::make_unique<MetalBuffer>(
            context.make_private_buffer(s_bytes));
        auto ra_buf = std::make_unique<MetalBuffer>(
            context.make_private_buffer(s_bytes));
        a8_mid_qa = Ref{qa_buf.get(), 0, qa_bytes};
        a8_mid_sa = Ref{sa_buf.get(), 0, s_bytes};
        a8_mid_ra = Ref{ra_buf.get(), 0, s_bytes};
        track_extra_workspace("w4a8_mid_qa", *qa_buf);
        track_extra_workspace("w4a8_mid_sa", *sa_buf);
        track_extra_workspace("w4a8_mid_ra", *ra_buf);
        buffers.push_back(std::move(qa_buf));
        buffers.push_back(std::move(sa_buf));
        buffers.push_back(std::move(ra_buf));
      }
    }
    if (w4a4_requested && any_fwht64) {
      const U32 max_mp = w4a4_nax_mp(workspace_rows_);
      const U32 max_mid_k =
          std::max({f, U32(config.shared_expert_intermediate_size)});
      if (max_mid_k > 0U && (max_mid_k % 128U) == 0U) {
        const U64 tiles_m = U64(max_mp / 16U);
        const U64 tiles_k = U64(max_mid_k / 128U);
        const U64 c_bytes = mul(mul(tiles_m, tiles_k), 1024ULL);
        const U64 s_bytes = mul(mul(tiles_m, tiles_k), 64ULL);
        auto c_buf = std::make_unique<MetalBuffer>(
            context.make_private_buffer(c_bytes));
        auto s_buf = std::make_unique<MetalBuffer>(
            context.make_private_buffer(s_bytes));
        w4a4_mid_codes = Ref{c_buf.get(), 0, c_bytes};
        w4a4_mid_scales = Ref{s_buf.get(), 0, s_bytes};
        track_extra_workspace("w4a4_mid_codes", *c_buf);
        track_extra_workspace("w4a4_mid_scales", *s_buf);
        buffers.push_back(std::move(c_buf));
        buffers.push_back(std::move(s_buf));
      }
    }

    if (!speculative_enabled && h >= 512U &&
        has_row_exact_metallib_ && t_domain_active() &&
        pretiled_qmv_t1_enabled()) {
      q5_high1_cache = acquire_q5_high1_cache(
          cache_owner_key, context, libraries, candidates, affine_nax_cache.get());
    }

    if ((mxfp4_nax_active() || affine_nax_active()) &&
        !sibling_concat_disabled()) {
      const U32 max_concat_n = std::max(
          2U * q + 2U * kv, conv + gv + 2U * U32(c.linear_num_value_heads));
      const U64 concat_bytes = mul(mul(workspace_rows_, max_concat_n), 2U);
      auto c_buf = std::make_unique<MetalBuffer>(
          context.make_private_buffer(concat_bytes));
      concat_out = Ref{c_buf.get(), 0, concat_bytes};
      track_extra_workspace("sibling_concat_out", *c_buf);
      buffers.push_back(std::move(c_buf));
    }

    if (!shared_primary_) {
      auto ps_parts =
          std::make_unique<MetalBuffer>(context.make_private_buffer(8192U));
      auto ps_meta = std::make_unique<MetalBuffer>(context.make_buffer(64U));
      auto ps_ids = std::make_unique<MetalBuffer>(context.make_buffer(2048U));
      auto ps_scores =
          std::make_unique<MetalBuffer>(context.make_buffer(2048U));
      packed_samp_parts = Ref{ps_parts.get(), 0, 8192U};
      packed_samp_meta = Ref{ps_meta.get(), 0, 64U};
      packed_samp_ids = Ref{ps_ids.get(), 0, 2048U};
      packed_samp_scores = Ref{ps_scores.get(), 0, 2048U};
      track_extra_workspace("packed_samp_parts", *ps_parts);
      track_extra_workspace("packed_samp_meta", *ps_meta);
      track_extra_workspace("packed_samp_ids", *ps_ids);
      track_extra_workspace("packed_samp_scores", *ps_scores);
      buffers.push_back(std::move(ps_parts));
      buffers.push_back(std::move(ps_meta));
      buffers.push_back(std::move(ps_ids));
      buffers.push_back(std::move(ps_scores));
      last_packed_logits_buf_ = logits.buffer;
    }

    gdn_chunk_nax_enabled_ =
        gdn_chunk_ws.buffer != nullptr && c.linear_attention_layers() > 0 &&
        gdn_chunk_nax_runtime_eligible(
            context, libraries, U32(c.linear_num_key_heads),
            U32(c.linear_num_value_heads), U32(c.linear_key_head_dim),
            U32(c.linear_value_head_dim));

    w4a8_nax_enabled_ =
        w4a8_requested &&
        (mxfp4_nax_active() || affine_nax_active() || moe_nax_ws.active);
    w4a4_nax_enabled_ =
        w4a4_requested && (mxfp4_nax_active() || moe_nax_ws.active);

    const bool affine4_planar_ok =
        (kv_format_ != KvStorageFormat::Affine4Group128) ||
        affine4_kv_runtime_eligible(context, libraries,
                                    U32(c.num_attention_heads),
                                    U32(c.num_key_value_heads),
                                    U32(c.head_dim));
    sage2_enabled_ =
        (prefill_options_.attention_mode == PrefillAttentionMode::Sage2) &&
        c.full_attention_layers() > 0 && affine4_planar_ok &&
        sage2_kv_ws_.buffer != nullptr &&
        sage2_runtime_eligible(context, libraries, U32(c.num_attention_heads),
                               U32(c.num_key_value_heads), U32(c.head_dim));
    minference_enabled_ =
        sage2_enabled_ && prefill_options_.sparse_minference &&
        minference_scratch.buffer != nullptr &&
        minference_runtime_eligible(context, libraries,
                                    U32(c.num_attention_heads),
                                    U32(c.num_key_value_heads),
                                    U32(c.head_dim));
    sage2_min_end_kv_ = kDefaultSage2MinEndKv;
    if (const char *env_s2 = std::getenv("OMLX_M5_SAGE2_MIN_KV_TOKENS")) {
      const auto parsed = std::strtoul(env_s2, nullptr, 10);
      sage2_min_end_kv_ = U32(std::min<unsigned long>(parsed, 1UL << 24));
    }
    minf_min_mean_kv_ = kDefaultMInferenceMinMeanKv;
    if (const char *env_mf = std::getenv("OMLX_M5_MINFERENCE_MIN_MEAN_KV")) {
      const auto parsed = std::strtoul(env_mf, nullptr, 10);
      minf_min_mean_kv_ = U32(std::min<unsigned long>(parsed, 1UL << 24));
    }

    if (attention_positions.buffer == nullptr && c.full_attention_layers() > 0 &&
        (sage2_enabled_ || prefill_options_.cache_blend)) {
      const U64 ap_bytes = mul(workspace_rows_, 4U);
      auto ap_buf = std::make_unique<MetalBuffer>(context.make_buffer(ap_bytes));
      attention_positions = Ref{ap_buf.get(), 0, ap_bytes};
      track_extra_workspace("prefill_attention_positions", *ap_buf);
      buffers.push_back(std::move(ap_buf));
    }

    cache_blend_kernels_ready_ =
        affine4_planar_ok &&
        std::filesystem::is_regular_file(libraries / "omlx_m5_cache_blend.metallib") &&
        context.capabilities().profile.probe.runtime_os_major >= 27 &&
        context.capabilities().profile.probe.metal4_compiler &&
        context.capabilities().profile.probe.tensor_api;

    // Module 4b / WP 3C: Graded Dual-ANE Collaboration (`compact_int8` + `ultra_fp16_swiglu`).
    // Mode resolution & legacy precedence:
    //   1. Explicit prefill_options_.ane_mode (or OMLX_M5_ANE_MODE diagnostic override)
    //   2. Legacy prefill_options_.ane_offload == true -> PrefillAneMode::Auto
    //   3. Auto grading:
    //      - MoE (num_experts > 0) -> off ("moe_unsupported")
    //      - ANE unavailable -> off ("ane_unavailable")
    //      - gpu_cores >= 64 && intermediate_size >= 4096 && workspace_rows >= 2048
    //        -> ultra_fp16_swiglu + 1-layer canary auto-guard (<= 0.98 * GPU-solo)
    //      - Else (gpu_cores < 64 or tiny synthetic fixtures) -> compact_int8
    {
      PrefillAneMode req_ane_mode = effective_requested_prefill_ane_mode(
          prefill_options_.ane_mode, prefill_options_.ane_offload);
      if (const char *env_mode = std::getenv("OMLX_M5_ANE_MODE");
          env_mode != nullptr && *env_mode != '\0') {
        if (const auto parsed = try_parse_prefill_ane_mode(env_mode)) {
          req_ane_mode = *parsed;
        }
      }

      U32 effective_gpu_cores = silicon_tuning.gpu_cores;
      if (const char *fake_gc = std::getenv("OMLX_M5_FAKE_GPU_CORES");
          fake_gc != nullptr && *fake_gc != '\0') {
        const auto parsed = std::strtoul(fake_gc, nullptr, 10);
        if (parsed > 0UL && parsed <= 512UL) {
          effective_gpu_cores = static_cast<U32>(parsed);
        }
      }

      if (!shared_primary_ && req_ane_mode != PrefillAneMode::Off) {
        const auto metallib_path = libraries / "omlx_m5_ane_pack_unpack.metallib";
        if (c.num_experts > 0) {
          ane_mode_effective_ = PrefillAneMode::Off;
          ane_inactive_reason_ = "moe_unsupported";
        } else {
          const auto ane_hw = ane::probe_ane_hardware();
          if (!ane_hw.is_available() ||
              !std::filesystem::is_regular_file(metallib_path)) {
            ane_mode_effective_ = PrefillAneMode::Off;
            ane_inactive_reason_ = "ane_unavailable";
          } else {
            PrefillAneMode target_mode = req_ane_mode;
            if (target_mode == PrefillAneMode::Auto) {
              const bool ultra_eligible =
                  (effective_gpu_cores >= 64U && c.intermediate_size >= 4096 &&
                   workspace_rows_ >= ane::kAneUltraDefaultChunkTokens);
              target_mode = ultra_eligible ? PrefillAneMode::UltraFp16SwiGlu
                                           : PrefillAneMode::CompactInt8;
            }

            ane::AneDualOffloadConfig ane_cfg;
            ane_cfg.linear_route =
                (prefill_options_.linear_mode == PrefillLinearMode::W4A4)
                    ? ane::AneLinearRoute::W4A4
                    : ((prefill_options_.linear_mode == PrefillLinearMode::W4A8)
                           ? ane::AneLinearRoute::W4A8
                           : ane::AneLinearRoute::BF16);
            ane_cfg.artifact_digest = prefill_options_.artifact_digest;
            ane_cfg.metallib_path = metallib_path;
            ane_cfg.external_mtl_device = context.native_mtl_device();
            ane_cfg.external_mtl_command_queue =
                context.native_mtl_command_queue();
            ane_cfg.share_input_surface = true;

            if (target_mode == PrefillAneMode::UltraFp16SwiGlu) {
              ane_cfg.chunk_tokens =
                  prefill_options_.ane_chunk_tokens
                      ? prefill_options_.ane_chunk_tokens
                      : std::min<U32>(ane::kAneUltraDefaultChunkTokens,
                                      workspace_rows_);
              ane_cfg.alpha_bf16 = ane::kAneUltraDefaultAlphaBf16;
              ane_cfg.alpha_w4a8 = ane::kAneUltraDefaultAlphaW4A8;
              ane_cfg.alpha_w4a4 = ane::kAneUltraDefaultAlphaW4A4;
              ane_cfg.use_fp16_weights = true;
              ane_cfg.mlp_swiglu_only = true;
              ane_cfg.max_in_features = 8192U;
              ane_cfg.max_bank_bytes_per_die = ane::kAneUltraMaxBankBytesPerDie;
              ane_cfg.max_procedures_per_bank =
                  ane::kAneUltraMaxProceduresPerBank;
              ane_cfg.calibration_bypass_ratio =
                  ane::kAneDefaultCalibrationBypassRatio;
            } else {
              // CompactInt8 (G=64 INT8 multi-projection offload for gpu_cores < 64
              // and explicit compact_int8).
              ane_cfg.chunk_tokens =
                  prefill_options_.ane_chunk_tokens
                      ? prefill_options_.ane_chunk_tokens
                      : std::min<U32>(ane::kAneDefaultChunkTokens,
                                      workspace_rows_);
              ane_cfg.alpha_bf16 = ane::kAneDefaultAlphaBf16;
              ane_cfg.alpha_w4a8 = ane::kAneDefaultAlphaW4A8;
              ane_cfg.alpha_w4a4 = ane::kAneDefaultAlphaW4A8;
              ane_cfg.use_fp16_weights = false;
              ane_cfg.mlp_swiglu_only = false;
              ane_cfg.max_in_features = 0U;
            }

            if (const char *env_chunk = std::getenv("OMLX_M5_ANE_CHUNK_TOKENS")) {
              const auto parsed = std::strtoul(env_chunk, nullptr, 10);
              if (parsed >= 64UL && parsed <= workspace_rows_) {
                ane_cfg.chunk_tokens = static_cast<U32>(parsed);
              }
            }
            if (const char *env_alpha = std::getenv("OMLX_M5_ANE_ALPHA")) {
              const double parsed = std::strtod(env_alpha, nullptr);
              if (std::isfinite(parsed) && parsed > 0.0 &&
                  parsed <= ane::kAneMaxAlpha) {
                ane_cfg.alpha_override = parsed;
              }
            }
            if (const char *env_fp16 = std::getenv("OMLX_M5_ANE_FP16")) {
              ane_cfg.use_fp16_weights = (std::string_view(env_fp16) == "1");
            }
            if (const char *env_swiglu = std::getenv("OMLX_M5_ANE_SWIGLU_ONLY")) {
              ane_cfg.mlp_swiglu_only = (std::string_view(env_swiglu) == "1");
            }
            if (const char *env_maxk = std::getenv("OMLX_M5_ANE_MAX_IN_FEATURES")) {
              ane_cfg.max_in_features =
                  static_cast<U32>(std::strtoul(env_maxk, nullptr, 10));
            }
            if (const char *env_procs =
                    std::getenv("OMLX_M5_ANE_MAX_PROCS_PER_BANK")) {
              const auto parsed = std::strtoul(env_procs, nullptr, 10);
              if (parsed >= 1UL && parsed <= 250UL) {
                ane_cfg.max_procedures_per_bank = static_cast<U32>(parsed);
              }
            }
            if (const char *env_mib = std::getenv("OMLX_M5_ANE_MAX_BANK_MIB")) {
              const auto parsed = std::strtoul(env_mib, nullptr, 10);
              if (parsed >= 32UL && parsed <= 2560UL) {
                ane_cfg.max_bank_bytes_per_die =
                    static_cast<U64>(parsed) * 1024ULL * 1024ULL;
              }
            }
            if (const char *env_share =
                    std::getenv("OMLX_M5_ANE_SHARE_INPUT_SURFACE")) {
              ane_cfg.share_input_surface = (std::string_view(env_share) != "0");
            }
            if (const char *env_ratio = std::getenv("OMLX_M5_ANE_CAL_RATIO")) {
              const double parsed = std::strtod(env_ratio, nullptr);
              if (std::isfinite(parsed) && parsed > 0.0 && parsed <= 2.0) {
                ane_cfg.calibration_bypass_ratio = parsed;
              }
            }

            if (ane_cfg.chunk_tokens >= 32U &&
                (ane_cfg.chunk_tokens % 32U) == 0U &&
                ane_cfg.chunk_tokens <= workspace_rows_) {
              bool allow_full_build = true;
              if (target_mode == PrefillAneMode::UltraFp16SwiGlu) {
                const U32 active_dies =
                    (!ane_cfg.force_single_die && ane_hw.physical_die_count >= 2U)
                        ? 2U
                        : 1U;
                double calibrated_alpha = ane_cfg.alpha_bf16;
                std::string cal_reason;
                const bool cal_ok = calibrate_ultra_fp16_swiglu_canary(
                    ane_cfg, effective_gpu_cores, active_dies, ane_hw.os_build,
                    calibrated_alpha, cal_reason);
                if (cal_ok) {
                  ane_cfg.alpha_override = calibrated_alpha;
                } else {
                  allow_full_build = false;
                  ane_mode_effective_ = PrefillAneMode::Off;
                  ane_inactive_reason_ =
                      cal_reason.empty() ? "calibration_bypassed" : cal_reason;
                }
              }

              if (allow_full_build) {
                try {
                  ane_engine_ = ane::AneDualOffloadEngine::build_from_model(
                      weights, layers, ane_cfg);
                } catch (...) {
                  ane_engine_.reset();
                }
                if (ane_engine_ && ane_engine_->is_active()) {
                  ane_active_ = true;
                  ane_mode_effective_ = target_mode;
                  ane_inactive_reason_.clear();
                  ane_chunk_tokens_ = ane_engine_->stats().chunk_tokens;
                  ane_split_alpha_ = ane_engine_->stats().target_alpha;
                  if (ane_mode_effective_ == PrefillAneMode::UltraFp16SwiGlu) {
                    const auto default_route =
                        select_chunk_attention_route(ane_chunk_tokens_, 0U);
                    const U32 default_env =
                        (default_route == ChunkAttentionRoute::Sage2MInference ||
                         (default_route == ChunkAttentionRoute::Sage2Dense &&
                          max_context > 4096U))
                            ? select_long_ctx_envelope(ane_chunk_tokens_)
                            : 0U;
                    static_cast<void>(get_or_build_ultra_swiglu_chunk_plan(
                        default_env, default_route, false));
                  }
                } else if (ane_inactive_reason_.empty()) {
                  ane_mode_effective_ = PrefillAneMode::Off;
                  ane_inactive_reason_ = "ane_unavailable";
                }
              }
            } else {
              ane_mode_effective_ = PrefillAneMode::Off;
              ane_inactive_reason_ = "chunk_tokens_unavailable";
            }
          }
        }
      }
    }
    prefill_telemetry_.prefill_ane_active = ane_active_;
    prefill_telemetry_.prefill_ane_mode_effective =
        std::string(prefill_ane_mode_to_string(ane_mode_effective_));
    prefill_telemetry_.prefill_ane_inactive_reason = ane_inactive_reason_;
    prefill_telemetry_.prefill_ane_split_alpha = ane_split_alpha_;

    reset();

    if (!speculative_enabled && !inspect_routes_enabled &&
        !inspect_tensors_workspace && feature_layers.empty() &&
        context.capabilities().profile.compute_icb_eligible) {
      for (U32 slot_i = 0; slot_i < 2U; ++slot_i) {
        auto &ap = ar_pipe_[slot_i];
        auto samp_buf = std::make_unique<MetalBuffer>(context.make_buffer(8U));
        ap.sampled = Ref{samp_buf.get(), 0, 8U};
        track_extra_workspace("ar_pipe_sampled_" + std::to_string(slot_i),
                              *samp_buf);
        buffers.push_back(std::move(samp_buf));
        if (attention_positions.buffer) {
          auto pos_buf = std::make_unique<MetalBuffer>(context.make_buffer(4U));
          ap.attn_pos = Ref{pos_buf.get(), 0, 4U};
          track_extra_workspace("ar_pipe_attn_pos_" + std::to_string(slot_i),
                                *pos_buf);
          buffers.push_back(std::move(pos_buf));
        }
      }
    }
    if (!shared_primary_ && !inspect_routes_enabled &&
        !inspect_tensors_workspace && workspace_rows_ >= 64U) {
      for (U32 slot_i = 0; slot_i < 2U; ++slot_i) {
        auto &pp = prefill_pipe_[slot_i];
        const U64 tok_bytes = mul(workspace_rows_, 4U);
        auto tok_buf =
            std::make_unique<MetalBuffer>(context.make_buffer(tok_bytes));
        pp.token_ids = Ref{tok_buf.get(), 0, tok_bytes};
        track_extra_workspace("prefill_pipe_tokens_" + std::to_string(slot_i),
                              *tok_buf);
        buffers.push_back(std::move(tok_buf));
        if (attention_positions.buffer) {
          auto pos_buf =
              std::make_unique<MetalBuffer>(context.make_buffer(tok_bytes));
          pp.attn_pos = Ref{pos_buf.get(), 0, tok_bytes};
          track_extra_workspace(
              "prefill_pipe_attn_pos_" + std::to_string(slot_i), *pos_buf);
          buffers.push_back(std::move(pos_buf));
        }
      }
    }

    ensure_bypass_scratch();
    rebuild_profiles();

    ExecutionIdentityFields base_fields{
        {"source", std::string(execution_source_id())},
        {"kernels", execution_kernel_bundle_digest(libraries)},
        {"linear",
         linear_policy == LinearNumericPolicy::RowExactPortableV1
             ? (mxfp4_nax_active()
                    ? "linear_row_exact_mxfp4_nax_v1"
                    : affine_nax_active() ? "linear_row_exact_affine_nax_v1"
                                          : "linear_row_exact_portable_v1")
             : "legacy_route_v1"},
        {"attention",
         attention_policy == omlx::m5::AttentionNumericPolicy::RowExactV1
             ? (strict_two_pass_attn_ ? "attention.row_exact.2pass.v1"
                                      : "attention.row_exact.v1")
             : "legacy_attention_v1"},
        {"gdn", "gdn.sequential.simd32.f32_state.conv_bf16.v1"},
        {"sampling", std::string(sampling_contract_name(sampling_contract))},
        {"chunk", std::string(prefill_chunk_policy_id(sampling_contract))},
        {"scope", "process-local immutable Program owner"}};
    if (kv_format_ != KvStorageFormat::BFloat16) {
      base_fields.emplace("kv_dtype",
                          std::string(kv_storage_format_name(kv_format_)));
    }
    if (flash_nax_prefill_ && this->config.hidden_size >= 1024) {
      base_fields.emplace("prefill_attn", "flash_nax_v1");
    }
    if (merged_lora_ && !merged_lora_->fingerprint.empty()) {
      base_fields.emplace("merged_lora", merged_lora_->fingerprint);
    }
    if (prefill_options_.linear_mode != PrefillLinearMode::BFloat16) {
      base_fields.emplace(
          "prefill_linear_mode",
          std::string(prefill_linear_mode_to_string(prefill_options_.linear_mode)));
    }
    if (prefill_options_.attention_mode != PrefillAttentionMode::ExactNax) {
      base_fields.emplace(
          "prefill_attention_mode",
          std::string(
              prefill_attention_mode_to_string(prefill_options_.attention_mode)));
    }
    if (prefill_options_.sparse_minference) {
      base_fields.emplace("prefill_sparse_minference", "1");
    }
    base_shared_contract_identity_ = execution_identity_digest(base_fields);
    // CacheBlend and Dual-ANE are single-slot prefill-only paths disabled on
    // secondary slots (Slot 1+) to avoid duplicating per-slot chunk/IOSurface
    // pools; keep them out of shared_contract_identity so B>=2 packed decode
    // across Slot 0 and Slot 1+ remains contract-compatible.
    if (prefill_options_.cache_blend) {
      base_fields.emplace("prefill_cache_blend", "1");
    }
    const PrefillAneMode req_ane_identity = effective_requested_prefill_ane_mode(
        prefill_options_.ane_mode, prefill_options_.ane_offload);
    if (req_ane_identity != PrefillAneMode::Off) {
      base_fields.emplace(
          "prefill_ane_mode",
          std::string(prefill_ane_mode_to_string(req_ane_identity)));
    }
    base_fields.emplace("owner", std::to_string(owner_id));
    base_execution_identity_ = execution_identity_digest(base_fields);
    refresh_lora_execution_identity();

    if (this->diagnostic_override && !diagnostic_override_matched)
      throw std::invalid_argument(
          "linear A/B override key matches no weight/profile");
    if (this->moe_override && !moe_override_matched)
      throw std::invalid_argument(
          "MoE A/B override key matches no weight/profile");
  }

  void ensure_bypass_scratch() {
    if (bypass_lora_ && !bypass_lora_->entries_by_object_id.empty()) {
      const U32 mp = ((workspace_rows_ + 7U) / 8U) * 8U;
      const U32 max_r = std::max<U32>(16U, bypass_lora_->max_r_slot);
      const U64 need_bytes =
          detail::lora_rt::align_256(mul(mul(mp, max_r), 2U));
      if (!lora_bypass_tmp_buf_ || lora_bypass_tmp_buf_->size() < need_bytes) {
        lora_bypass_tmp_buf_ =
            std::make_unique<MetalBuffer>(context.make_private_buffer(need_bytes));
      }
      lora_bypass_tmp_ = Ref{lora_bypass_tmp_buf_.get(), 0, need_bytes};
    } else {
      lora_bypass_tmp_ = {};
      lora_bypass_tmp_buf_.reset();
    }
  }

  void refresh_lora_execution_identity() {
    std::string sh = base_shared_contract_identity_;
    std::string ex = base_execution_identity_;
    if (bypass_lora_ && !bypass_lora_->fingerprint.empty()) {
      sh = execution_identity_digest(
          {{"program", sh}, {"bypass_lora", bypass_lora_->fingerprint}});
      ex = execution_identity_digest(
          {{"program", ex}, {"bypass_lora", bypass_lora_->fingerprint}});
    }
    if (execution_identity_bound && !bound_context_identity_.empty()) {
      sh = execution_identity_digest(
          {{"program", sh}, {"context", bound_context_identity_}});
      ex = execution_identity_digest(
          {{"program", ex}, {"context", bound_context_identity_}});
    }
    shared_contract_identity = std::move(sh);
    execution_identity = std::move(ex);
  }

  void rebuild_profiles() {
    profiles.clear();
    batch_profiles.clear();
    verify_profiles.clear();
    split_profiles.clear();
    split_batch_profiles.clear();
    split_verify_profiles.clear();
    long_ctx_profiles_.clear();
    ane_layer_plans_.clear();
    ane_ultra_plans_.clear();
    packed_cache_.clear();
    verify_async_profile_ = nullptr;
    for (auto &ap : ar_pipe_)
      ap.profiles.clear();
    for (auto &pp : prefill_pipe_)
      pp.profiles.clear();

    for (U32 capacity : {1U, 2U, 4U, 8U, 12U, 16U, 24U, 32U, 64U, 128U, 256U,
                         512U, 1024U, 2048U, 4096U}) {
      if (capacity > workspace_rows_ || capacity > max_context)
        break;
      profiles.push_back(compile(capacity, false));
      ensure_profile_icb(context, profiles.back());
    }
    const U32 final_capacity = std::min(max_context, workspace_rows_);
    if (profiles.empty() || profiles.back().capacity < final_capacity) {
      profiles.push_back(compile(final_capacity, false));
      ensure_profile_icb(context, profiles.back());
    }
    if (batch_enabled) {
      for (U32 capacity : {1U, 2U, 4U, 8U}) {
        if (capacity <= workspace_rows_ && capacity <= max_context) {
          batch_profiles.push_back(compile(capacity, false, false, 0, true));
          ensure_profile_icb(context, batch_profiles.back());
        }
      }
      const U32 batch_tail = std::min({max_context, workspace_rows_, 8U});
      if (batch_profiles.empty() || batch_profiles.back().capacity < batch_tail) {
        batch_profiles.push_back(compile(batch_tail, false, false, 0, true));
        ensure_profile_icb(context, batch_profiles.back());
      }
    }
    if (speculative_enabled) {
      for (U32 capacity : {1U, 2U, 3U, 4U, 8U}) {
        if (capacity > workspace_rows_ || capacity > max_context)
          break;
        verify_profiles.push_back(compile(capacity, true));
        ensure_profile_icb(context, verify_profiles.back());
      }
      const U32 verify_tail = std::min({max_context, workspace_rows_, 8U});
      if (verify_profiles.empty() ||
          verify_profiles.back().capacity < verify_tail) {
        verify_profiles.push_back(compile(verify_tail, true));
        ensure_profile_icb(context, verify_profiles.back());
      }
      commit_profile = compile_commit();
      ensure_profile_icb(context, commit_profile);
    }
    if (page_pool) {
      kv_cow_profile = compile_kv_cow();
      ensure_profile_icb(context, kv_cow_profile);
    }
    if (!inspect_tensors_workspace && workspace_rows_ > 8U) {
      inspect_norm_profile = compile_inspect_norm();
      inspect_head_chunk_profile = compile_inspect_head_chunk();
      ensure_profile_icb(context, inspect_norm_profile);
      ensure_profile_icb(context, inspect_head_chunk_profile);
    }
    if (split_attention_enabled) {
      std::vector<U32> envelopes{std::min(2048U, max_context),
                                 std::min(8192U, max_context),
                                 std::min(32768U, max_context),
                                 std::min(65536U, max_context), max_context};
      std::ranges::sort(envelopes);
      envelopes.erase(std::unique(envelopes.begin(), envelopes.end()),
                      envelopes.end());
      for (U32 capacity : {1U, 2U, 3U, 4U, 8U}) {
        if (capacity > workspace_rows_ || capacity > max_context)
          break;
        for (const U32 envelope : envelopes) {
          if (envelope < capacity)
            continue;
          if (capacity != 3U) {
            split_profiles.push_back(compile(capacity, false, true, envelope));
            ensure_profile_icb(context, split_profiles.back());
            if (batch_enabled) {
              split_batch_profiles.push_back(
                  compile(capacity, false, true, envelope, true));
              ensure_profile_icb(context, split_batch_profiles.back());
            }
          }
          if (speculative_enabled) {
            split_verify_profiles.push_back(
                compile(capacity, true, true, envelope));
            ensure_profile_icb(context, split_verify_profiles.back());
          }
        }
      }
    }
    if (!shared_primary_) {
      delegated_sampler_profile = compile_single_sampler();
      ensure_profile_icb(context, delegated_sampler_profile);
    }
    if (!speculative_enabled && !inspect_routes_enabled &&
        !inspect_tensors_workspace && feature_layers.empty() &&
        context.capabilities().profile.compute_icb_eligible &&
        ar_pipe_[0].sampled.buffer != nullptr) {
      const Ref feed_tok{token_ids.buffer, 0, 4U};
      const U32 split_min_ctx =
          std::max<U32>(512U, silicon_tuning.split_kv_min_context);
      std::vector<U32> ar_envelopes{max_context};
      if (split_attention_enabled) {
        ar_envelopes = {std::min(512U, max_context),
                        std::min(split_min_ctx, max_context),
                        std::min(2048U, max_context),
                        std::min(8192U, max_context),
                        std::min(16384U, max_context),
                        std::min(32768U, max_context),
                        max_context};
        std::ranges::sort(ar_envelopes);
        ar_envelopes.erase(
            std::unique(ar_envelopes.begin(), ar_envelopes.end()),
            ar_envelopes.end());
      }
      ScopedTdomOverride ar_tdom_guard{tdom_batch_override_, true};
      struct ScopedSplitPartCap {
        U32 &slot;
        U32 prev;
        ScopedSplitPartCap(U32 &s, U32 v) : slot(s), prev(s) { slot = v; }
        ~ScopedSplitPartCap() { slot = prev; }
      } ar_split_cap_guard{
          split_kv_partition_cap_override_,
          attention_policy == omlx::m5::AttentionNumericPolicy::RowExactV1
              ? 0U
              : silicon_tuning.split_kv_partitions};
      for (U32 slot_i = 0; slot_i < 2U; ++slot_i) {
        auto &ap = ar_pipe_[slot_i];
        const std::string tag = ".ar_pipe" + std::to_string(slot_i) +
                                (batch_enabled ? ".td1" : "");
        if (split_attention_enabled) {
          const bool affine4_split_at_512 =
              (kv_format_ == KvStorageFormat::Affine4Group128) &&
              (attention_positions.buffer != nullptr) &&
              (config.head_dim == 256) &&
              affine4_kv_runtime_eligible(
                  context, libraries, U32(config.num_attention_heads),
                  U32(config.num_key_value_heads), U32(config.head_dim)) &&
              affine4_split_scratch_layout(
                  1U, U32(config.num_attention_heads),
                  U32(config.num_key_value_heads), 1U, split_output.bytes)
                  .fits;
          for (const U32 env : ar_envelopes) {
            if (env < 1U)
              continue;
            const bool use_split =
                (env >= split_min_ctx && env > 512U) || affine4_split_at_512;
            ap.profiles.push_back(compile(1U, false, use_split, env, false,
                                          ap.sampled, ap.attn_pos, feed_tok,
                                          tag));
            ensure_profile_icb(context, ap.profiles.back());
          }
        } else {
          ap.profiles.push_back(compile(1U, false, false, 0U, false, ap.sampled,
                                        ap.attn_pos, feed_tok, tag));
          ensure_profile_icb(context, ap.profiles.back());
        }
      }
    }
    if (!shared_primary_ && !inspect_routes_enabled &&
        !inspect_tensors_workspace && workspace_rows_ >= 64U &&
        prefill_pipe_[0].token_ids.buffer != nullptr) {
      for (U32 slot_i = 0; slot_i < 2U; ++slot_i) {
        auto &pp = prefill_pipe_[slot_i];
        const std::string tag = ".pf_pipe" + std::to_string(slot_i);
        for (U32 capacity : {64U, 128U, 256U, 512U, 1024U, 2048U, 4096U}) {
          if (capacity > workspace_rows_ || capacity > max_context)
            break;
          pp.profiles.push_back(compile(capacity, false, false, 0U, false, {},
                                        pp.attn_pos, {}, tag,
                                        /*trunk_only=*/true, pp.token_ids));
          ensure_profile_icb(context, pp.profiles.back());
        }
        const U32 final_pf_cap = std::min(max_context, workspace_rows_);
        if (pp.profiles.empty() || pp.profiles.back().capacity < final_pf_cap) {
          pp.profiles.push_back(compile(final_pf_cap, false, false, 0U, false,
                                        {}, pp.attn_pos, {}, tag,
                                        /*trunk_only=*/true, pp.token_ids));
          ensure_profile_icb(context, pp.profiles.back());
        }
      }
    }
    if (!inspect_tensors_workspace) {
      const U32 max_cap = std::min(max_context, workspace_rows_);
      auto precompile_lc = [&](U32 cap, U32 env, ChunkAttentionRoute route,
                               bool include_pipe_slot1) {
        const LongCtxProfileKey fk{cap, env, route, false, false, 0U};
        if (!long_ctx_profiles_.contains(fk)) {
          auto prof = compile(cap, false, false, env, false, {}, {}, {}, {},
                              false, {}, route);
          ensure_profile_icb(context, prof);
          long_ctx_profiles_.emplace(fk, std::move(prof));
        }
        if (!shared_primary_ && !inspect_routes_enabled &&
            prefill_pipe_[0].token_ids.buffer != nullptr && cap >= 64U) {
          const U32 max_slot =
              (include_pipe_slot1 &&
               prefill_pipe_[1].token_ids.buffer != nullptr)
                  ? 2U
                  : 1U;
          for (U32 slot_i = 0U; slot_i < max_slot; ++slot_i) {
            const LongCtxProfileKey tk{cap, env, route, true, true,
                                       std::uint8_t(slot_i)};
            if (!long_ctx_profiles_.contains(tk)) {
              auto tprof = compile(
                  cap, false, false, env, false, {},
                  prefill_pipe_[slot_i].attn_pos, {},
                  ".pf_pipe" + std::to_string(slot_i), /*trunk_only=*/true,
                  prefill_pipe_[slot_i].token_ids, route);
              ensure_profile_icb(context, tprof);
              long_ctx_profiles_.emplace(tk, std::move(tprof));
            }
          }
        }
      };

      // WP4: Precompile tight Sage2Dense prefix envelopes (2K, 4K) with both
      // double-buffered async pipe slots when max_context > 4K so prepare_k
      // launches <= 64/128 blocks instead of max_context/32.
      if (sage2_enabled_ && max_context > 4096U &&
          max_cap >= kFlashNaxPrefillMinTokens) {
        for (const U32 env : {2048U, 4096U}) {
          precompile_lc(max_cap, env, ChunkAttentionRoute::Sage2Dense,
                        /*include_pipe_slot1=*/true);
        }
      }

      if ((minference_enabled_ ||
           (flash_nax_prefill_ && split_attention_enabled &&
            config.head_dim == 256)) &&
          max_context >= kMInferenceMinContextTokens) {
        std::vector<U32> lc_envs{max_context};
        if (max_context > kMInferenceMinContextTokens) {
          lc_envs.push_back(kMInferenceMinContextTokens);
        }
        std::ranges::sort(lc_envs);
        lc_envs.erase(std::unique(lc_envs.begin(), lc_envs.end()),
                      lc_envs.end());
        std::vector<U32> lc_caps;
        for (U32 cap : {64U, 256U, 1024U, max_cap}) {
          if (cap >= kFlashNaxPrefillMinTokens && cap <= max_cap) {
            lc_caps.push_back(cap);
          }
        }
        std::ranges::sort(lc_caps);
        lc_caps.erase(std::unique(lc_caps.begin(), lc_caps.end()),
                      lc_caps.end());
        for (const U32 env : lc_envs) {
          for (const U32 cap : lc_caps) {
            if (minference_enabled_) {
              precompile_lc(cap, env, ChunkAttentionRoute::Sage2MInference,
                            /*include_pipe_slot1=*/false);
            } else if (gqa_flash_prefill_tail_needs_split_kv(
                           cap, U32(config.num_attention_heads),
                           U32(config.num_key_value_heads), env,
                           silicon_tuning.gpu_cores)) {
              precompile_lc(cap, env, ChunkAttentionRoute::ExactNax,
                            /*include_pipe_slot1=*/false);
            }
          }
        }
      }
    }
  }

  Ref allocate(std::string_view name) {
    const auto found = std::find_if(
        allocation_plan.buffers.begin(), allocation_plan.buffers.end(),
        [&](const auto &item) { return item.name == name; });
    if (found == allocation_plan.buffers.end() ||
        !allocation_names.insert(std::string(name)).second)
      throw std::logic_error("Qwen allocation is absent or duplicated: " +
                             std::string(name));
    auto buffer =
        std::make_unique<MetalBuffer>(context.make_buffer(found->bytes));
    Ref ref{buffer.get(), 0, found->bytes};
    if (found->clear_on_reset)
      states.push_back(buffer.get());
    buffers.push_back(std::move(buffer));
    return ref;
  }

  Ref allocate_private(std::string_view name) {
    const auto found = std::find_if(
        allocation_plan.buffers.begin(), allocation_plan.buffers.end(),
        [&](const auto &item) { return item.name == name; });
    if (found == allocation_plan.buffers.end() ||
        !allocation_names.insert(std::string(name)).second)
      throw std::logic_error("Qwen allocation is absent or duplicated: " +
                             std::string(name));
    auto buffer = std::make_unique<MetalBuffer>(
        context.make_private_buffer(found->bytes));
    Ref ref{buffer.get(), 0, found->bytes};
    if (found->clear_on_reset)
      states.push_back(buffer.get());
    buffers.push_back(std::move(buffer));
    return ref;
  }

  void track_extra_workspace(std::string name, const MetalBuffer &buf) {
    const U64 bytes = buf.size();
    allocation_plan.workspace_bytes += bytes;
    allocation_plan.largest_buffer_bytes =
        std::max(allocation_plan.largest_buffer_bytes, bytes);
    allocation_plan.buffers.push_back(
        {std::move(name), bytes, /*state=*/false, /*clear_on_reset=*/false});
  }

  Ref allocate_sized(std::string_view name, U64 explicit_bytes) {
    const auto found = std::find_if(
        allocation_plan.buffers.begin(), allocation_plan.buffers.end(),
        [&](const auto &item) { return item.name == name; });
    if (found == allocation_plan.buffers.end() ||
        !allocation_names.insert(std::string(name)).second)
      throw std::logic_error("Qwen allocation is absent or duplicated: " +
                             std::string(name));
    auto buffer =
        std::make_unique<MetalBuffer>(context.make_buffer(explicit_bytes));
    Ref ref{buffer.get(), 0, explicit_bytes};
    if (found->clear_on_reset)
      states.push_back(buffer.get());
    buffers.push_back(std::move(buffer));
    return ref;
  }

  Ref direct(const MaterializedWeightView &w, U64 count) {
    if (w.format != "direct_bf16_v1" || plane(w, "data").bytes != mul(count, 2))
      throw std::invalid_argument("Qwen BF16 parameter geometry: " + w.name);
    return plane(w, "data");
  }

  bool is_overridden(const MaterializedWeightView &w) const {
    return diagnostic_override.has_value() &&
           diagnostic_override->object == w.name;
  }

  bool has_bypass_lora(const MaterializedWeightView &w) const noexcept {
    return bypass_lora_ != nullptr && bypass_lora_->has(w.object_id);
  }

  const QwenBypassLoraEntry *
  find_bypass_lora(const MaterializedWeightView &w) const noexcept {
    return bypass_lora_ != nullptr ? bypass_lora_->find(w.object_id) : nullptr;
  }

  const QwenBypassSiblingGroup *
  find_bypass_group(U32 layer_index, QwenBypassGroupKind kind) const noexcept {
    return bypass_lora_ != nullptr ? bypass_lora_->find_group(layer_index, kind)
                                   : nullptr;
  }

  template <typename BlockT>
  bool block_has_any_bypass(
      const BlockT &block,
      std::initializer_list<const char *> roles) const noexcept {
    if (bypass_lora_ == nullptr)
      return false;
    for (const char *role : roles) {
      const auto it = block.operands.find(role);
      if (it != block.operands.end() &&
          bypass_lora_->has(it->second.object_id)) {
        return true;
      }
    }
    return false;
  }

  void emit_bypass_for_weight(Encoder &e, const MaterializedWeightView &w,
                              Ref x, Ref y, U32 n,
                              std::optional<U32> explicit_t = {}) {
    if (const auto *entry = find_bypass_lora(w); entry != nullptr) {
      detail::lora_rt::emit_bypass_single_entry(
          e, *entry, x, y, lora_bypass_tmp_, n, 0U, bf16_nax_bypass_ok_,
          silicon_tuning.gpu_cores, explicit_t);
    }
  }

  static bool pretiled_qmv_t1_enabled() {
    if (const char *env = std::getenv("OMLX_M5_ENABLE_PRETILED_QMV_T1")) {
      return std::string_view(env) != "0" && !std::string_view(env).empty();
    }
    return true;
  }

  Ref affine_high_plane(const MaterializedWeightView &w, U32 bits) const {
    return bits == 3U                     ? plane(w, "nax_high1")
           : (bits == 5U || bits == 6U)   ? plane(w, "nax_high2")
                                          : zero;
  }

  struct ScopedTdomOverride {
    bool &flag;
    bool prev;
    ScopedTdomOverride(bool &f, bool v) noexcept : flag(f), prev(f) {
      flag = v;
    }
    ~ScopedTdomOverride() { flag = prev; }
  };

  bool allow_tdom_t1() const noexcept {
    return !batch_enabled || tdom_batch_override_;
  }

  bool t_domain_active() const noexcept {
    const bool apple10 =
        context.capabilities().device_name.starts_with("Apple M5") ||
        context.capabilities().profile.probe.apple_gpu_family >= 10;
    return t_domain_dispatch_enabled(!speculative_enabled && h >= 1024U &&
                                     apple10);
  }

  bool resolve_pretiled_mxfp4_t1(const MaterializedWeightView &w, U32 t,
                                 Ref *out_codes = nullptr,
                                 Ref *out_scales = nullptr) const {
    if (!(t == 1U && !speculative_enabled && allow_tdom_t1() &&
          t_domain_active() && pretiled_qmv_t1_enabled() &&
          !is_overridden(w) && w.format == "mxfp4_e2m1_g32_e8m0_v1" &&
          w.parent_shape.size() == 2 && w.row_count > 0 && h >= 512U &&
          w.in_features >= 512U && (w.in_features % 128U) == 0U)) {
      return false;
    }
    if (w.layout == "mx_nax_tiled_bk128_v1") {
      if (out_codes)
        *out_codes = plane(w, "nax_data");
      if (out_scales)
        *out_scales = plane(w, "nax_scale");
      return true;
    }
    if (mxfp4_nax_cache) {
      const U32 exp_bn = mxfp4_nax_bn(U32(w.row_count));
      if (const auto *tw = mxfp4_nax_cache->find(w.object_id);
          tw && tw->n == w.row_count && tw->k == w.in_features &&
          tw->bn == exp_bn) {
        if (out_codes)
          *out_codes = tw->codes;
        if (out_scales)
          *out_scales = tw->scales;
        return true;
      }
    }
    return false;
  }

  static U32 pretiled_k_flags(U32 k, bool lane_v2, bool fwht64) noexcept {
    return (k & 0x3FFFFFFFU) | (lane_v2 ? 0x40000000U : 0U) |
           (fwht64 ? 0x80000000U : 0U);
  }

  bool resolve_pretiled_affine_t1(const MaterializedWeightView &w, U32 t,
                                  Ref *out_codes = nullptr,
                                  Ref *out_high = nullptr,
                                  Ref *out_sb = nullptr,
                                  bool *out_lane_v2 = nullptr) const {
    const U32 bits = affine_nax_bits(w.format);
    if (!(t == 1U && !speculative_enabled && allow_tdom_t1() &&
          t_domain_active() && pretiled_qmv_t1_enabled() &&
          !is_overridden(w) && bits != 0U && w.parent_shape.size() == 2 &&
          w.row_count > 0 && h >= 512U && w.in_features >= 512U &&
          (w.in_features % 64U) == 0U)) {
      return false;
    }
    const Ref *q5_h1 =
        (bits == 5U && q5_high1_cache && q5_high1_cache->impl_)
            ? q5_high1_cache->impl_->find(w.object_id)
            : nullptr;
    if (bits == 5U && q5_h1 == nullptr) {
      return false;
    }
    if (affine_nax_cache) {
      const U32 exp_bn = affine_nax_bn(U32(w.row_count));
      if (const auto *tw = affine_nax_cache->find(w.object_id);
          tw && tw->n == w.row_count && tw->k == w.in_features &&
          tw->bn == exp_bn && tw->bits == bits && tw->sb_tiled.buffer) {
        if (out_codes)
          *out_codes = tw->codes;
        if (out_high)
          *out_high = (bits == 5U)
                          ? *q5_h1
                          : (affine_nax_has_high_plane(bits) ? tw->high2 : zero);
        if (out_sb)
          *out_sb = tw->sb_tiled;
        if (out_lane_v2)
          *out_lane_v2 = true;
        return true;
      }
    }
    if (is_oq_nax_pretiled_layout(w.layout)) {
      const Ref *sb_tl =
          (q5_high1_cache && q5_high1_cache->impl_)
              ? q5_high1_cache->impl_->find_sb_tiled(w.object_id)
              : nullptr;
      if (!sb_tl || !sb_tl->buffer) {
        return false;
      }
      if (out_codes)
        *out_codes = plane(w, "nax_codes");
      if (out_high)
        *out_high = (bits == 5U) ? *q5_h1 : affine_high_plane(w, bits);
      if (out_sb)
        *out_sb = *sb_tl;
      if (out_lane_v2)
        *out_lane_v2 = is_oq_nax_lane_v2_layout(w.layout);
      return true;
    }
    return false;
  }

  bool resolve_mxfp4_planes_any_t(const MaterializedWeightView &w,
                                  Ref *out_codes = nullptr,
                                  Ref *out_scales = nullptr) const {
    if (w.format != "mxfp4_e2m1_g32_e8m0_v1" || w.parent_shape.size() != 2U ||
        w.row_count == 0U || w.in_features < 512U ||
        (w.in_features % 128U) != 0U) {
      return false;
    }
    if (w.layout == "mx_nax_tiled_bk128_v1") {
      if (out_codes)
        *out_codes = plane(w, "nax_data");
      if (out_scales)
        *out_scales = plane(w, "nax_scale");
      return true;
    }
    if (mxfp4_nax_cache) {
      const U32 exp_bn = mxfp4_nax_bn(U32(w.row_count));
      if (const auto *tw = mxfp4_nax_cache->find(w.object_id);
          tw && tw->n == w.row_count && tw->k == w.in_features &&
          tw->bn == exp_bn) {
        if (out_codes)
          *out_codes = tw->codes;
        if (out_scales)
          *out_scales = tw->scales;
        return true;
      }
    }
    return false;
  }

  bool resolve_affine_planes_any_t(const MaterializedWeightView &w,
                                   Ref *out_codes = nullptr,
                                   Ref *out_high = nullptr,
                                   Ref *out_sb = nullptr,
                                   bool *out_lane_v2 = nullptr) const {
    const U32 bits = affine_nax_bits(w.format);
    if (bits == 0U || w.parent_shape.size() != 2U || w.row_count == 0U ||
        w.in_features < 512U || (w.in_features % 64U) != 0U) {
      return false;
    }
    const Ref *q5_h1 =
        (bits == 5U && q5_high1_cache && q5_high1_cache->impl_)
            ? q5_high1_cache->impl_->find(w.object_id)
            : nullptr;
    if (bits == 5U && q5_h1 == nullptr) {
      return false;
    }
    if (affine_nax_cache) {
      const U32 exp_bn = affine_nax_bn(U32(w.row_count));
      if (const auto *tw = affine_nax_cache->find(w.object_id);
          tw && tw->n == w.row_count && tw->k == w.in_features &&
          tw->bn == exp_bn && tw->bits == bits && tw->sb_tiled.buffer) {
        if (out_codes)
          *out_codes = tw->codes;
        if (out_high)
          *out_high = (bits == 5U)
                          ? *q5_h1
                          : (affine_nax_has_high_plane(bits) ? tw->high2 : zero);
        if (out_sb)
          *out_sb = tw->sb_tiled;
        if (out_lane_v2)
          *out_lane_v2 = true;
        return true;
      }
    }
    if (is_oq_nax_pretiled_layout(w.layout)) {
      const Ref *sb_tl =
          (q5_high1_cache && q5_high1_cache->impl_)
              ? q5_high1_cache->impl_->find_sb_tiled(w.object_id)
              : nullptr;
      if (!sb_tl || !sb_tl->buffer) {
        return false;
      }
      if (out_codes)
        *out_codes = plane(w, "nax_codes");
      if (out_high)
        *out_high = (bits == 5U) ? *q5_h1 : affine_high_plane(w, bits);
      if (out_sb)
        *out_sb = *sb_tl;
      if (out_lane_v2)
        *out_lane_v2 = is_oq_nax_lane_v2_layout(w.layout);
      return true;
    }
    return false;
  }

  bool use_pretiled_mxfp4_t1(const MaterializedWeightView &w, U32 t) const {
    return resolve_pretiled_mxfp4_t1(w, t);
  }

  bool use_pretiled_affine_t1(const MaterializedWeightView &w, U32 t) const {
    return resolve_pretiled_affine_t1(w, t);
  }

  bool use_mxfp4_w4a4_for(U32 t) const {
    if (!mxfp4_nax_active() || !a4_codes.buffer || t < 32u ||
        diagnostic_override)
      return false;
    return mxfp4_w4a4_prefill_enabled();
  }

  bool use_mxfp4_nax_for_weight(const MaterializedWeightView &w, U32 t) const {
    if (use_pretiled_mxfp4_t1(w, t))
      return false;
    if (!mxfp4_nax_active() || !f16_scratch.buffer || use_mxfp4_w4a4_for(t) ||
        is_overridden(w))
      return false;
    // Bug #1 fix: keep NAX active unconditionally on pre-tiled weights even
    // under LinearNumericPolicy::Legacy or OMLX_M5_T_DOMAIN_DISPATCH=1.
    if (w.layout == "mx_nax_tiled_bk128_v1")
      return true;
    if (linear_policy != LinearNumericPolicy::RowExactPortableV1)
      return false;
    if (t == 1u && t_domain_active())
      return false;
    return true;
  }

  bool use_affine_nax_for_weight(const MaterializedWeightView &w, U32 t) const {
    if (use_pretiled_affine_t1(w, t))
      return false;
    if (!affine_nax_active() || !affine_nax_xs.buffer || is_overridden(w))
      return false;
    // Bug #1 fix: keep NAX active unconditionally on pre-tiled weights.
    if (is_oq_nax_pretiled_layout(w.layout))
      return true;
    if (linear_policy != LinearNumericPolicy::RowExactPortableV1)
      return false;
    if (t == 1u && t_domain_active())
      return false;
    return true;
  }

  void linear_from_f16(Encoder &e, const MaterializedWeightView &w, Ref x_f16,
                       Ref rs, Ref y, U32 n, U32 k,
                       std::optional<U32> explicit_t = {}) {
    const auto *tw = mxfp4_nax_cache->find(w.object_id);
    if (!tw || tw->n != n || tw->k != k)
      throw std::logic_error("linear_from_f16 called on non-NAX weight");
    const bool unit_scale =
        (rs.buffer != nullptr && rs.buffer == f16_ones_scale.buffer);
    linear_mxfp4_nax_direct_f16(e, *tw, x_f16, rs, y, explicit_t,
                                silicon_tuning.gpu_cores, unit_scale);
    e.record_weight(
        w, n, k, "mxfp4_nax_m5_v1", "mxfp4_nax_m5_v1", {},
        "linear_row_exact_portable_v1",
        mxfp4_nax_group(e, *tw, explicit_t, silicon_tuning.gpu_cores),
        true, explicit_t);
  }

  void linear_base(Encoder &e, const MaterializedWeightView &w, Ref x, Ref y,
                   U32 n, U32 k, std::optional<U32> explicit_t = {}) {
    const U32 t = explicit_t.value_or(e.profile.capacity);
    const U32 sel_t = nax_effective_select_t(e, explicit_t);
    if (use_mxfp4_w4a4_for(sel_t) && !is_overridden(w)) {
      if (const auto *tw = mxfp4_nax_cache->find(w.object_id);
          tw && tw->n == n && tw->k == k) {
        linear_mxfp4_w4a4_nax(e, *tw, x, y, a4_codes, a4_scales, explicit_t,
                              silicon_tuning.gpu_cores, w.has_fwht64());
        e.record_weight(
            w, n, k, "mxfp4_w4a4_nax_m5_v1", "mxfp4_w4a4_nax_m5_v1", {},
            "legacy_route_v1",
            mxfp4_w4a4_group(e, *tw, explicit_t, silicon_tuning.gpu_cores),
            false, explicit_t);
        return;
      }
    }
    if (use_mxfp4_nax_for_weight(w, sel_t)) {
      if (const auto *tw = mxfp4_nax_cache->find(w.object_id);
          tw && tw->n == n && tw->k == k) {
        linear_mxfp4_nax(e, *tw, x, y, f16_scratch, f16_row_scale, explicit_t,
                         silicon_tuning.gpu_cores, 0u, 0u, 0u, w.has_fwht64());
        e.record_weight(
            w, n, k, "mxfp4_nax_m5_v1", "mxfp4_nax_m5_v1", {},
            "linear_row_exact_portable_v1",
            mxfp4_nax_group(e, *tw, explicit_t, silicon_tuning.gpu_cores),
            true, explicit_t);
        return;
      }
    }
    if (use_affine_nax_for_weight(w, sel_t)) {
      if (const auto *tw = affine_nax_cache->find(w.object_id);
          tw && tw->n == n && tw->k == k) {
        if (w.has_fwht64()) {
          linear_affine_nax_fwht64(e, *tw, x, fwht_scratch, affine_nax_xs, y,
                                   explicit_t, silicon_tuning.gpu_cores);
        } else {
          linear_affine_nax(e, *tw, x, affine_nax_xs, y, explicit_t,
                            silicon_tuning.gpu_cores);
        }
        e.record_weight(
            w, n, k, "affine_nax_m5_v1", "affine_nax_m5_v1", {},
            "linear_row_exact_portable_v1",
            affine_nax_group(e, *tw, explicit_t, silicon_tuning.gpu_cores),
            true, explicit_t);
        return;
      }
    }
    if (bf16_nax_enabled && !is_overridden(w) &&
        bf16_nax_weight_eligible(w, n, k)) {
      Ref x_in = x;
      if (w.has_fwht64()) {
        detail::forward::emit_fwht64_bf16(e, x, fwht_scratch, k, explicit_t);
        x_in = fwht_scratch;
      }
      linear_bf16_nax(e, w, x_in, y, n, k, explicit_t, silicon_tuning.gpu_cores);
      return;
    }
    if (e.profile.capacity == 1U && w.row_count == n && w.in_features == k) {
      Ref mx_c{}, mx_s{};
      if (resolve_pretiled_mxfp4_t1(w, t, &mx_c, &mx_s)) {
        const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
        const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
        const char *fn =
            (n >= 65536U)  ? "omlx_m5_pretiled_qmv_mxfp4_bn32_r4_t1"
            : (n >= 2048U) ? "omlx_m5_pretiled_qmv_mxfp4_bn16_r4_t1"
                           : "omlx_m5_pretiled_qmv_mxfp4_bn16_r2_t1";
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U}, {x, mx_c, mx_s, y});
        e.dynamic(d, 4,
                  words({n, pretiled_k_flags(k, false, w.has_fwht64()),
                         U32(w.padded_k), t}),
                  explicit_t.has_value() ? std::optional<U32>{}
                                         : std::optional<U32>{3U});
        e.record_weight(w, n, k, "pretiled_qmv_t1", "pretiled_mxfp4_qmv_t1_v3",
                        {}, "legacy_route_v1", {}, false, explicit_t);
        return;
      }
      Ref aff_c{}, aff_h{}, aff_sb{};
      bool lane_v2 = false;
      if (resolve_pretiled_affine_t1(w, t, &aff_c, &aff_h, &aff_sb, &lane_v2)) {
        const U32 bits = affine_nax_bits(w.format);
        const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
        const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
        const char *fn = "omlx_m5_pretiled_qmv_affine_t1";
        if (bits == 4U) {
          fn = (n >= 65536U)  ? "omlx_m5_pretiled_qmv_affine_q4_bn32_r4_t1"
               : (n >= 2048U) ? "omlx_m5_pretiled_qmv_affine_q4_bn16_r4_t1"
                              : "omlx_m5_pretiled_qmv_affine_q4_bn16_r2_t1";
        } else if (bits == 5U) {
          fn = (n >= 65536U)  ? "omlx_m5_pretiled_qmv_affine_q5_bn32_r4_t1"
               : (n >= 2048U) ? "omlx_m5_pretiled_qmv_affine_q5_bn16_r4_t1"
                              : "omlx_m5_pretiled_qmv_affine_q5_bn16_r2_t1";
        } else if (bits == 6U) {
          fn = (n >= 65536U)  ? "omlx_m5_pretiled_qmv_affine_q6_bn32_r4_t1"
               : (n >= 2048U) ? "omlx_m5_pretiled_qmv_affine_q6_bn16_r4_t1"
                              : "omlx_m5_pretiled_qmv_affine_q6_bn16_r2_t1";
        } else if (bits == 8U) {
          fn = (n >= 65536U)  ? "omlx_m5_pretiled_qmv_affine_q8_bn32_r4_t1"
               : (n >= 2048U) ? "omlx_m5_pretiled_qmv_affine_q8_bn16_r4_t1"
                              : "omlx_m5_pretiled_qmv_affine_q8_bn16_r2_t1";
        }
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U}, {x, aff_c, aff_h, aff_sb, y});
        e.dynamic(d, 5,
                  words({n, t, 0U, pretiled_k_flags(k, lane_v2, w.has_fwht64()),
                         bits}),
                  explicit_t.has_value() ? std::optional<U32>{}
                                         : std::optional<U32>{1U});
        e.record_weight(w, n, k, "pretiled_qmv_t1", "pretiled_affine_qmv_t1_v3",
                        {}, "legacy_route_v1", {}, false, explicit_t);
        return;
      }
    }
    auto decision = resolve_linear_route(
        w, n, k, t, context.capabilities().device_name, m4_short_routes,
        diagnostic_override, linear_policy,
        context.capabilities().profile.probe.apple_gpu_family);
    diagnostic_override_matched |= decision.override_matched;
    Ref x_leg = x;
    if (w.has_fwht64()) {
      detail::forward::emit_fwht64_bf16(e, x, fwht_scratch, k, explicit_t);
      x_leg = fwht_scratch;
    }
    detail::forward::linear(e, w, x_leg, y, n, k, sums, zero, decision.route,
                            explicit_t);
    e.record_weight(w, n, k, decision.route.family, decision.source,
                    decision.fallback_reason, decision.route.numeric_contract_id,
                    decision.route.row_exact_group, decision.route.row_exact,
                    explicit_t);
  }

  void linear(Encoder &e, const MaterializedWeightView &w, Ref x, Ref y, U32 n,
              U32 k, std::optional<U32> explicit_t = {}) {
    linear_base(e, w, x, y, n, k, explicit_t);
    emit_bypass_for_weight(e, w, x, y, n, explicit_t);
  }

  bool use_w4a8_for_t(U32 t, bool verify) const noexcept {
    return w4a8_nax_enabled_ && !verify && t >= w4a8_min_tokens_ &&
           a8_qa.buffer != nullptr && (h % 64U) == 0U && !diagnostic_override;
  }

  bool use_w4a8_prefill(const Encoder &e, bool verify,
                        std::optional<U32> explicit_t = {}) const noexcept {
    return use_w4a8_for_t(nax_effective_select_t(e, explicit_t), verify);
  }

  Encoder::A8QuantKey make_a8_key_for(const Encoder &e, Ref input, U32 k,
                                      Ref qa, Ref sa, Ref ra,
                                      std::optional<U32> explicit_t = {},
                                      bool fwht64 = false) const {
    const U32 t = explicit_t.value_or(e.profile.capacity);
    const bool dyn_rows = !explicit_t.has_value() && t != 1U;
    return Encoder::A8QuantKey{
        .input_buffer = input.buffer,
        .input_offset = input.offset,
        .qa_buffer = qa.buffer,
        .qa_offset = qa.offset,
        .sa_buffer = sa.buffer,
        .sa_offset = sa.offset,
        .ra_buffer = ra.buffer,
        .ra_offset = ra.offset,
        .k = k,
        .rows = t,
        .mp = w4a8_nax_mp(t),
        .dynamic_tokens = dyn_rows,
        .fwht64 = fwht64,
    };
  }

  Encoder::A8QuantKey make_a8_key(const Encoder &e, Ref input, U32 k,
                                  std::optional<U32> explicit_t = {},
                                  bool fwht64 = false) const {
    return make_a8_key_for(e, input, k, a8_qa, a8_sa, a8_ra, explicit_t,
                           fwht64);
  }

  bool ensure_a8_quant(Encoder &e, Ref input, U32 k,
                       std::optional<U32> explicit_t = {},
                       bool fwht64 = false) {
    if (a8_qa.buffer == nullptr || a8_sa.buffer == nullptr ||
        a8_ra.buffer == nullptr || (k % 64U) != 0U || k == 0U) {
      return false;
    }
    const auto key = make_a8_key(e, input, k, explicit_t, fwht64);
    if (e.has_a8_quant(key)) {
      return true;
    }
    emit_quantize_a8_g64(e, input, a8_qa, a8_sa, a8_ra, k, explicit_t,
                         std::nullopt, fwht64);
    e.set_a8_quant(key);
    return true;
  }

  bool can_w4a8_weight(const MaterializedWeightView &w, U32 n, U32 k) const {
    if (!w4a8_nax_enabled_ || is_overridden(w) || (k % 64U) != 0U ||
        a8_qa.buffer == nullptr) {
      return false;
    }
    if (mxfp4_nax_active() && (k % 128U) == 0U) {
      if (const auto *tw = mxfp4_nax_cache->find(w.object_id);
          tw != nullptr && tw->n == n && tw->k == k) {
        return true;
      }
    }
    if (affine_nax_active()) {
      if (const auto *tw = affine_nax_cache->find(w.object_id);
          tw != nullptr && tw->n == n && tw->k == k) {
        return true;
      }
    }
    return false;
  }

  bool try_linear_w4a8_base(Encoder &e, const MaterializedWeightView &w, Ref x,
                            Ref y, U32 n, U32 k,
                            std::optional<U32> explicit_t = {},
                            U32 row_offset = 0U, U32 out_stride = 0U,
                            U32 slice_n = 0U) {
    if (!w4a8_nax_enabled_ || is_overridden(w) || (k % 64U) != 0U ||
        a8_qa.buffer == nullptr) {
      return false;
    }
    const bool use_mid =
        w.has_fwht64() && a8_mid_qa.buffer != nullptr &&
        e.has_a8_quant(make_a8_key_for(e, x, k, a8_mid_qa, a8_mid_sa, a8_mid_ra,
                                       explicit_t, /*fwht64=*/true));
    const Ref qa_use = use_mid ? a8_mid_qa : a8_qa;
    const Ref sa_use = use_mid ? a8_mid_sa : a8_sa;
    const Ref ra_use = use_mid ? a8_mid_ra : a8_ra;
    if (mxfp4_nax_active() && (k % 128U) == 0U) {
      if (const auto *tw = mxfp4_nax_cache->find(w.object_id);
          tw != nullptr && tw->n == n && tw->k == k) {
        if (!use_mid && !ensure_a8_quant(e, x, k, explicit_t, w.has_fwht64())) {
          return false;
        }
        linear_mxfp4_w4a8_nax(e, *tw, qa_use, sa_use, y, explicit_t, row_offset,
                              out_stride, slice_n);
        e.record_weight(w, n, k, "mxfp4_w4a8_nax_m5_v1", "mxfp4_w4a8_nax_m5_v1",
                        {}, "w4a8_prefill_v1", "mxfp4.w4a8.bm64_bn64.sk1.v1",
                        false, explicit_t);
        return true;
      }
    }
    if (affine_nax_active()) {
      if (const auto *tw = affine_nax_cache->find(w.object_id);
          tw != nullptr && tw->n == n && tw->k == k) {
        if (!use_mid && !ensure_a8_quant(e, x, k, explicit_t, w.has_fwht64())) {
          return false;
        }
        linear_affine_w4a8_nax(e, *tw, qa_use, sa_use, ra_use, y, explicit_t,
                               row_offset, out_stride, slice_n);
        e.record_weight(
            w, n, k, "affine_w4a8_nax_m5_v1", "affine_w4a8_nax_m5_v1", {},
            "w4a8_prefill_v1",
            "oq.w4a8.q" + std::to_string(tw->bits) + "_g" +
                std::to_string(tw->group_size) + ".bm64_bn64.sk1.v1",
            false, explicit_t);
        return true;
      }
    }
    return false;
  }

  bool try_linear_w4a8_swiglu(Encoder &e, const MaterializedWeightView &gate_w,
                              const MaterializedWeightView &up_w, Ref x, Ref y,
                              U32 n, U32 k, std::optional<U32> explicit_t = {},
                              U32 row_offset = 0U, U32 out_stride = 0U,
                              U32 slice_n = 0U) {
    if (!w4a8_nax_enabled_ || is_overridden(gate_w) || is_overridden(up_w) ||
        has_bypass_lora(gate_w) || has_bypass_lora(up_w) || (k % 64U) != 0U ||
        a8_qa.buffer == nullptr) {
      return false;
    }
    if (mxfp4_nax_active() && (k % 128U) == 0U) {
      const auto *gw = mxfp4_nax_cache->find(gate_w.object_id);
      const auto *uw = mxfp4_nax_cache->find(up_w.object_id);
      if (gw && uw && gw->bn == 16U && uw->bn == 16U && gw->n == n &&
          gw->k == k && uw->n == n && uw->k == k) {
        if (!ensure_a8_quant(e, x, k, explicit_t)) {
          return false;
        }
        linear_mxfp4_w4a8_nax_swiglu(e, *gw, *uw, a8_qa, a8_sa, y, explicit_t,
                                     row_offset, out_stride, slice_n);
        e.record_weight(gate_w, n, k, "mxfp4_w4a8_nax_m5_v1",
                        "mxfp4_w4a8_swiglu_out64_v1", {}, "w4a8_prefill_v1",
                        "mxfp4.w4a8.swiglu_out64.sk1.v1", false, explicit_t);
        e.record_weight(up_w, n, k, "mxfp4_w4a8_nax_m5_v1",
                        "mxfp4_w4a8_swiglu_out64_v1", {}, "w4a8_prefill_v1",
                        "mxfp4.w4a8.swiglu_out64.sk1.v1", false, explicit_t);
        return true;
      }
    }
    if (affine_nax_active()) {
      const auto *gw = affine_nax_cache->find(gate_w.object_id);
      const auto *uw = affine_nax_cache->find(up_w.object_id);
      if (gw && uw && gw->bn == 16U && uw->bn == 16U && gw->n == n &&
          gw->k == k && uw->n == n && uw->k == k) {
        const std::string tag_g = affine_nax_mode_tag(gw->bits);
        const std::string tag_u = affine_nax_mode_tag(uw->bits);
        const bool pair_ok = (tag_g == tag_u) ||
                             (tag_g == "q4" && tag_u == "q56") ||
                             (tag_g == "q56" && tag_u == "q4");
        if (pair_ok && ensure_a8_quant(e, x, k, explicit_t)) {
          linear_affine_w4a8_nax_swiglu(e, *gw, *uw, a8_qa, a8_sa, a8_ra, y,
                                        explicit_t, row_offset, out_stride,
                                        slice_n);
          const std::string grp = "oq.w4a8." + tag_g + "_" + tag_u +
                                  ".swiglu_out64.sk1.v1";
          e.record_weight(gate_w, n, k, "affine_w4a8_nax_m5_v1",
                          "affine_w4a8_swiglu_out64_v1", {}, "w4a8_prefill_v1",
                          grp, false, explicit_t);
          e.record_weight(up_w, n, k, "affine_w4a8_nax_m5_v1",
                          "affine_w4a8_swiglu_out64_v1", {}, "w4a8_prefill_v1",
                          grp, false, explicit_t);
          return true;
        }
      }
    }
    return false;
  }

  bool try_linear_w4a8_swiglu_fwht64_a8(Encoder &e,
                                        const MaterializedWeightView &gate_w,
                                        const MaterializedWeightView &up_w,
                                        Ref x, Ref y_bf16, U32 n, U32 k,
                                        bool also_write_bf16,
                                        std::optional<U32> explicit_t = {},
                                        U32 row_offset = 0U,
                                        U32 out_stride = 0U,
                                        U32 slice_n = 0U) {
    if (!w4a8_nax_enabled_ || is_overridden(gate_w) || is_overridden(up_w) ||
        has_bypass_lora(gate_w) || has_bypass_lora(up_w) ||
        gate_w.has_fwht64() || up_w.has_fwht64() || (k % 64U) != 0U ||
        (n % 64U) != 0U || (row_offset % 64U) != 0U || row_offset >= n ||
        a8_qa.buffer == nullptr || a8_mid_qa.buffer == nullptr ||
        a8_mid_sa.buffer == nullptr || a8_mid_ra.buffer == nullptr) {
      return false;
    }
    const U32 eff_n = (slice_n > 0U) ? slice_n : (n - row_offset);
    if (row_offset + eff_n > n || (eff_n % 64U) != 0U) {
      return false;
    }
    const bool is_slice = (row_offset > 0U) ||
                          (out_stride > 0U && out_stride != n) ||
                          (slice_n > 0U && slice_n != n);
    if (mxfp4_nax_active() && (k % 128U) == 0U) {
      const auto *gw = mxfp4_nax_cache->find(gate_w.object_id);
      const auto *uw = mxfp4_nax_cache->find(up_w.object_id);
      if (gw && uw && gw->bn == 16U && uw->bn == 16U && gw->n == n &&
          gw->k == k && uw->n == n && uw->k == k) {
        if (!ensure_a8_quant(e, x, k, explicit_t, /*fwht64=*/false)) {
          return false;
        }
        linear_mxfp4_w4a8_nax_swiglu_fwht64_a8(
            e, *gw, *uw, a8_qa, a8_sa, a8_mid_qa, a8_mid_sa, a8_mid_ra, y_bf16,
            also_write_bf16, explicit_t, row_offset, out_stride, slice_n);
        if (!is_slice) {
          e.set_a8_quant(make_a8_key_for(e, y_bf16, n, a8_mid_qa, a8_mid_sa,
                                         a8_mid_ra, explicit_t,
                                         /*fwht64=*/true));
        }
        e.record_weight(gate_w, n, k, "mxfp4_w4a8_nax_m5_v1",
                        "mxfp4_w4a8_swiglu_out64_fwht64_a8_v2", {},
                        "w4a8_prefill_v1", "mxfp4.w4a8.swiglu_out64.fwht64.v2",
                        false, explicit_t);
        e.record_weight(up_w, n, k, "mxfp4_w4a8_nax_m5_v1",
                        "mxfp4_w4a8_swiglu_out64_fwht64_a8_v2", {},
                        "w4a8_prefill_v1", "mxfp4.w4a8.swiglu_out64.fwht64.v2",
                        false, explicit_t);
        return true;
      }
    }
    if (affine_nax_active()) {
      const auto *gw = affine_nax_cache->find(gate_w.object_id);
      const auto *uw = affine_nax_cache->find(up_w.object_id);
      if (gw && uw && gw->bn == 16U && uw->bn == 16U && gw->n == n &&
          gw->k == k && uw->n == n && uw->k == k) {
        const std::string tag_g = affine_nax_mode_tag(gw->bits);
        const std::string tag_u = affine_nax_mode_tag(uw->bits);
        const bool pair_ok = (tag_g == tag_u) ||
                             (tag_g == "q4" && tag_u == "q56") ||
                             (tag_g == "q56" && tag_u == "q4");
        if (pair_ok &&
            ensure_a8_quant(e, x, k, explicit_t, /*fwht64=*/false)) {
          linear_affine_w4a8_nax_swiglu_fwht64_a8(
              e, *gw, *uw, a8_qa, a8_sa, a8_ra, a8_mid_qa, a8_mid_sa, a8_mid_ra,
              y_bf16, also_write_bf16, explicit_t, row_offset, out_stride,
              slice_n);
          if (!is_slice) {
            e.set_a8_quant(make_a8_key_for(e, y_bf16, n, a8_mid_qa, a8_mid_sa,
                                           a8_mid_ra, explicit_t,
                                           /*fwht64=*/true));
          }
          const std::string grp = "oq.w4a8." + tag_g + "_" + tag_u +
                                  ".swiglu_out64.fwht64.v2";
          e.record_weight(gate_w, n, k, "affine_w4a8_nax_m5_v1",
                          "affine_w4a8_swiglu_out64_fwht64_a8_v2", {},
                          "w4a8_prefill_v1", grp, false, explicit_t);
          e.record_weight(up_w, n, k, "affine_w4a8_nax_m5_v1",
                          "affine_w4a8_swiglu_out64_fwht64_a8_v2", {},
                          "w4a8_prefill_v1", grp, false, explicit_t);
          return true;
        }
      }
    }
    return false;
  }

  bool try_concat_w4a8_nax(Encoder &e, U32 concat_id,
                           std::span<const MaterializedWeightView> parts,
                           std::span<const U32> part_ns, U32 k, Ref x_bf16,
                           Ref dst_concat,
                           std::optional<U32> explicit_t = {}) {
    if (!w4a8_nax_enabled_ || !concat_id || !dst_concat.buffer ||
        parts.empty() || (k % 64U) != 0U || a8_qa.buffer == nullptr) {
      return false;
    }
    U32 total_n = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
      if (is_overridden(parts[i])) {
        return false;
      }
      total_n += part_ns[i];
    }
    if (mxfp4_nax_active() && (k % 128U) == 0U) {
      if (const auto *tw = mxfp4_nax_cache->find(concat_id);
          tw != nullptr && tw->n == total_n && tw->k == k) {
        if (!ensure_a8_quant(e, x_bf16, k, explicit_t)) {
          return false;
        }
        linear_mxfp4_w4a8_nax(e, *tw, a8_qa, a8_sa, dst_concat, explicit_t);
        for (std::size_t i = 0; i < parts.size(); ++i) {
          e.record_weight(parts[i], part_ns[i], k, "mxfp4_w4a8_nax_m5_v1",
                          "mxfp4_w4a8_nax_m5_v1", {}, "w4a8_prefill_v1",
                          "mxfp4.w4a8.bm64_bn64.sk1.v1", false, explicit_t);
        }
        return true;
      }
    }
    if (affine_nax_active()) {
      if (const auto *tw = affine_nax_cache->find(concat_id);
          tw != nullptr && tw->n == total_n && tw->k == k) {
        if (!ensure_a8_quant(e, x_bf16, k, explicit_t)) {
          return false;
        }
        linear_affine_w4a8_nax(e, *tw, a8_qa, a8_sa, a8_ra, dst_concat,
                               explicit_t);
        const std::string grp = "oq.w4a8.q" + std::to_string(tw->bits) + "_g" +
                                std::to_string(tw->group_size) +
                                ".bm64_bn64.sk1.v1";
        for (std::size_t i = 0; i < parts.size(); ++i) {
          e.record_weight(parts[i], part_ns[i], k, "affine_w4a8_nax_m5_v1",
                          "affine_w4a8_nax_m5_v1", {}, "w4a8_prefill_v1", grp,
                          false, explicit_t);
        }
        return true;
      }
    }
    return false;
  }

  bool use_w4a4_for_t(U32 t, bool verify) const noexcept {
    return w4a4_nax_enabled_ && !verify && t >= w4a4_min_tokens_ &&
           w4a4_codes.buffer != nullptr && (h % 128U) == 0U &&
           !diagnostic_override;
  }

  bool use_w4a4_prefill(const Encoder &e, bool verify,
                        std::optional<U32> explicit_t = {}) const noexcept {
    return use_w4a4_for_t(nax_effective_select_t(e, explicit_t), verify);
  }

  Encoder::A4QuantKey make_a4_key_for(const Encoder &e, Ref input, U32 k,
                                      Ref codes, Ref scales,
                                      std::optional<U32> explicit_t = {},
                                      bool fwht64 = false) const {
    const U32 t = explicit_t.value_or(e.profile.capacity);
    const bool dyn_rows = !explicit_t.has_value() && t != 1U;
    return Encoder::A4QuantKey{
        .input_buffer = input.buffer,
        .input_offset = input.offset,
        .codes_buffer = codes.buffer,
        .codes_offset = codes.offset,
        .scales_buffer = scales.buffer,
        .scales_offset = scales.offset,
        .k = k,
        .rows = t,
        .mp = w4a4_nax_mp(t),
        .dynamic_tokens = dyn_rows,
        .fwht64 = fwht64,
    };
  }

  Encoder::A4QuantKey make_a4_key(const Encoder &e, Ref input, U32 k,
                                  std::optional<U32> explicit_t = {},
                                  bool fwht64 = false) const {
    return make_a4_key_for(e, input, k, w4a4_codes, w4a4_scales, explicit_t,
                           fwht64);
  }

  bool ensure_a4_quant(Encoder &e, Ref input, U32 k,
                       std::optional<U32> explicit_t = {},
                       bool fwht64 = false) {
    if (w4a4_codes.buffer == nullptr || w4a4_scales.buffer == nullptr ||
        (k % 128U) != 0U || k == 0U) {
      return false;
    }
    const auto key = make_a4_key(e, input, k, explicit_t, fwht64);
    if (e.has_a4_quant(key)) {
      return true;
    }
    emit_quantize_a4_bm16_bk128(e, input, w4a4_codes, w4a4_scales, k,
                                explicit_t, std::nullopt, fwht64);
    e.set_a4_quant(key);
    return true;
  }

  bool can_w4a4_weight(const MaterializedWeightView &w, U32 n, U32 k) const {
    if (!w4a4_nax_enabled_ || is_overridden(w) || (k % 128U) != 0U ||
        w4a4_codes.buffer == nullptr || !mxfp4_nax_active()) {
      return false;
    }
    if (const auto *tw = mxfp4_nax_cache->find(w.object_id);
        tw != nullptr && tw->n == n && tw->k == k) {
      return true;
    }
    return false;
  }

  bool try_linear_w4a4_base(Encoder &e, const MaterializedWeightView &w, Ref x,
                            Ref y, U32 n, U32 k,
                            std::optional<U32> explicit_t = {},
                            U32 row_offset = 0U, U32 out_stride = 0U,
                            U32 slice_n = 0U) {
    if (!w4a4_nax_enabled_ || is_overridden(w) || (k % 128U) != 0U ||
        w4a4_codes.buffer == nullptr || !mxfp4_nax_active()) {
      return false;
    }
    const auto *tw = mxfp4_nax_cache->find(w.object_id);
    if (tw == nullptr || tw->n != n || tw->k != k) {
      return false;
    }
    const bool use_mid =
        w.has_fwht64() && w4a4_mid_codes.buffer != nullptr &&
        e.has_a4_quant(make_a4_key_for(e, x, k, w4a4_mid_codes, w4a4_mid_scales,
                                       explicit_t, /*fwht64=*/true));
    const Ref codes_use = use_mid ? w4a4_mid_codes : w4a4_codes;
    const Ref scales_use = use_mid ? w4a4_mid_scales : w4a4_scales;
    if (!use_mid && !ensure_a4_quant(e, x, k, explicit_t, w.has_fwht64())) {
      return false;
    }
    emit_mxfp4_w4a4_nax_linear(e, *tw, codes_use, scales_use, y, explicit_t,
                               row_offset, out_stride, slice_n);
    e.record_weight(w, n, k, "mxfp4_w4a4_nax_m5_v2", "mxfp4_w4a4_nax_m5_v2", {},
                    "w4a4_prefill_v1", "mxfp4.w4a4.bm64_bn64.sk1.v2", false,
                    explicit_t);
    return true;
  }

  bool try_linear_w4a4_swiglu(Encoder &e, const MaterializedWeightView &gate_w,
                              const MaterializedWeightView &up_w, Ref x, Ref y,
                              U32 n, U32 k, std::optional<U32> explicit_t = {},
                              U32 row_offset = 0U, U32 out_stride = 0U,
                              U32 slice_n = 0U) {
    if (!w4a4_nax_enabled_ || is_overridden(gate_w) || is_overridden(up_w) ||
        has_bypass_lora(gate_w) || has_bypass_lora(up_w) || (k % 128U) != 0U ||
        w4a4_codes.buffer == nullptr || !mxfp4_nax_active()) {
      return false;
    }
    const auto *gw = mxfp4_nax_cache->find(gate_w.object_id);
    const auto *uw = mxfp4_nax_cache->find(up_w.object_id);
    if (!gw || !uw || gw->bn != 16U || uw->bn != 16U || gw->n != n ||
        gw->k != k || uw->n != n || uw->k != k) {
      return false;
    }
    if (!ensure_a4_quant(e, x, k, explicit_t)) {
      return false;
    }
    emit_mxfp4_w4a4_nax_swiglu(e, *gw, *uw, w4a4_codes, w4a4_scales, y,
                               explicit_t, row_offset, out_stride, slice_n);
    e.record_weight(gate_w, n, k, "mxfp4_w4a4_nax_m5_v2",
                    "mxfp4_w4a4_swiglu_out64_v2", {}, "w4a4_prefill_v1",
                    "mxfp4.w4a4.swiglu_out64.sk1.v2", false, explicit_t);
    e.record_weight(up_w, n, k, "mxfp4_w4a4_nax_m5_v2",
                    "mxfp4_w4a4_swiglu_out64_v2", {}, "w4a4_prefill_v1",
                    "mxfp4.w4a4.swiglu_out64.sk1.v2", false, explicit_t);
    return true;
  }

  bool try_linear_w4a4_swiglu_fwht64_a4(Encoder &e,
                                        const MaterializedWeightView &gate_w,
                                        const MaterializedWeightView &up_w,
                                        Ref x, Ref y_bf16, U32 n, U32 k,
                                        bool also_write_bf16,
                                        std::optional<U32> explicit_t = {},
                                        U32 row_offset = 0U,
                                        U32 out_stride = 0U, U32 slice_n = 0U) {
    if (!w4a4_nax_enabled_ || is_overridden(gate_w) || is_overridden(up_w) ||
        has_bypass_lora(gate_w) || has_bypass_lora(up_w) ||
        gate_w.has_fwht64() || up_w.has_fwht64() || (k % 128U) != 0U ||
        (n % 128U) != 0U || (row_offset % 128U) != 0U ||
        w4a4_codes.buffer == nullptr || w4a4_mid_codes.buffer == nullptr ||
        w4a4_mid_scales.buffer == nullptr || !mxfp4_nax_active()) {
      return false;
    }
    const auto *gw = mxfp4_nax_cache->find(gate_w.object_id);
    const auto *uw = mxfp4_nax_cache->find(up_w.object_id);
    if (!gw || !uw || gw->bn != 16U || uw->bn != 16U || gw->n != n ||
        gw->k != k || uw->n != n || uw->k != k) {
      return false;
    }
    if (!ensure_a4_quant(e, x, k, explicit_t, /*fwht64=*/false)) {
      return false;
    }
    emit_mxfp4_w4a4_nax_swiglu_fwht64_a4(
        e, *gw, *uw, w4a4_codes, w4a4_scales, w4a4_mid_codes, w4a4_mid_scales,
        y_bf16, also_write_bf16, explicit_t, row_offset, out_stride, slice_n);
    if (row_offset == 0U && slice_n == 0U) {
      e.set_a4_quant(make_a4_key_for(e, y_bf16, n, w4a4_mid_codes,
                                     w4a4_mid_scales, explicit_t,
                                     /*fwht64=*/true));
    }
    e.record_weight(gate_w, n, k, "mxfp4_w4a4_nax_m5_v2",
                    "mxfp4_w4a4_swiglu_out64_fwht64_a4_v2", {},
                    "w4a4_prefill_v1", "mxfp4.w4a4.swiglu_out64.fwht64.v2",
                    false, explicit_t);
    e.record_weight(up_w, n, k, "mxfp4_w4a4_nax_m5_v2",
                    "mxfp4_w4a4_swiglu_out64_fwht64_a4_v2", {},
                    "w4a4_prefill_v1", "mxfp4.w4a4.swiglu_out64.fwht64.v2",
                    false, explicit_t);
    return true;
  }

  bool try_concat_w4a4_nax(Encoder &e, U32 concat_id,
                           std::span<const MaterializedWeightView> parts,
                           std::span<const U32> part_ns, U32 k, Ref x_bf16,
                           Ref dst_concat,
                           std::optional<U32> explicit_t = {}) {
    if (!w4a4_nax_enabled_ || !concat_id || !dst_concat.buffer ||
        parts.empty() || (k % 128U) != 0U || w4a4_codes.buffer == nullptr ||
        !mxfp4_nax_active()) {
      return false;
    }
    U32 total_n = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
      if (is_overridden(parts[i])) {
        return false;
      }
      total_n += part_ns[i];
    }
    if (const auto *tw = mxfp4_nax_cache->find(concat_id);
        tw != nullptr && tw->n == total_n && tw->k == k) {
      if (!ensure_a4_quant(e, x_bf16, k, explicit_t)) {
        return false;
      }
      emit_mxfp4_w4a4_nax_linear(e, *tw, w4a4_codes, w4a4_scales, dst_concat,
                                 explicit_t);
      for (std::size_t i = 0; i < parts.size(); ++i) {
        e.record_weight(parts[i], part_ns[i], k, "mxfp4_w4a4_nax_m5_v2",
                        "mxfp4_w4a4_nax_m5_v2", {}, "w4a4_prefill_v1",
                        "mxfp4.w4a4.bm64_bn64.sk1.v2", false, explicit_t);
      }
      return true;
    }
    return false;
  }

  bool can_gpu_suffix_linear(const MaterializedWeightView &w, U32 n, U32 k,
                             bool use_w4a8) const {
    if (is_overridden(w) || has_bypass_lora(w)) {
      return false;
    }
    if (use_w4a8 && can_w4a8_weight(w, n, k)) {
      return true;
    }
    if (mxfp4_nax_active()) {
      if (const auto *tw = mxfp4_nax_cache->find(w.object_id);
          tw != nullptr && tw->n == n && tw->k == k) {
        return true;
      }
    }
    if (affine_nax_active()) {
      if (const auto *tw = affine_nax_cache->find(w.object_id);
          tw != nullptr && tw->n == n && tw->k == k) {
        return true;
      }
    }
    return false;
  }

  bool emit_gpu_suffix_linear(Encoder &e, const MaterializedWeightView &w,
                              Ref x, Ref y, U32 n, U32 k, bool use_w4a8,
                              U32 row_offset, U32 out_stride, U32 slice_n,
                              std::optional<U32> explicit_t = {}) {
    if (use_w4a8 && try_linear_w4a8_base(e, w, x, y, n, k, explicit_t,
                                         row_offset, out_stride, slice_n)) {
      return true;
    }
    if (mxfp4_nax_active()) {
      if (const auto *tw = mxfp4_nax_cache->find(w.object_id);
          tw != nullptr && tw->n == n && tw->k == k) {
        linear_mxfp4_nax(e, *tw, x, y, f16_scratch, f16_row_scale, explicit_t,
                         silicon_tuning.gpu_cores, row_offset, out_stride,
                         slice_n, w.has_fwht64());
        return true;
      }
    }
    if (affine_nax_active()) {
      if (const auto *tw = affine_nax_cache->find(w.object_id);
          tw != nullptr && tw->n == n && tw->k == k) {
        if (w.has_fwht64()) {
          linear_affine_nax_fwht64(e, *tw, x, fwht_scratch, affine_nax_xs, y,
                                   explicit_t, silicon_tuning.gpu_cores,
                                   row_offset, out_stride, slice_n);
        } else {
          linear_affine_nax(e, *tw, x, affine_nax_xs, y, explicit_t,
                            silicon_tuning.gpu_cores, row_offset, out_stride,
                            slice_n);
        }
        return true;
      }
    }
    return false;
  }

  bool can_gpu_suffix_swiglu(const MaterializedWeightView &gate_w,
                             const MaterializedWeightView &up_w, U32 n, U32 k,
                             bool use_w4a8) const {
    if (is_overridden(gate_w) || is_overridden(up_w) ||
        has_bypass_lora(gate_w) || has_bypass_lora(up_w)) {
      return false;
    }
    if (use_w4a8 && can_w4a8_weight(gate_w, n, k) &&
        can_w4a8_weight(up_w, n, k)) {
      if (mxfp4_nax_active() && (k % 128U) == 0U) {
        const auto *gw = mxfp4_nax_cache->find(gate_w.object_id);
        const auto *uw = mxfp4_nax_cache->find(up_w.object_id);
        if (gw && uw && gw->bn == 16U && uw->bn == 16U) {
          return true;
        }
      }
      if (affine_nax_active()) {
        const auto *gw = affine_nax_cache->find(gate_w.object_id);
        const auto *uw = affine_nax_cache->find(up_w.object_id);
        if (gw && uw && gw->bn == 16U && uw->bn == 16U) {
          const std::string tg = affine_nax_mode_tag(gw->bits);
          const std::string tu = affine_nax_mode_tag(uw->bits);
          if (tg == tu || (tg == "q4" && tu == "q56") ||
              (tg == "q56" && tu == "q4")) {
            return true;
          }
        }
      }
    }
    if (mxfp4_nax_active()) {
      const auto *gw = mxfp4_nax_cache->find(gate_w.object_id);
      const auto *uw = mxfp4_nax_cache->find(up_w.object_id);
      if (gw && uw && gw->bn == 16U && uw->bn == 16U && gw->n == n &&
          gw->k == k && uw->n == n && uw->k == k) {
        return true;
      }
    }
    if (affine_nax_active()) {
      const auto *gw = affine_nax_cache->find(gate_w.object_id);
      const auto *uw = affine_nax_cache->find(up_w.object_id);
      if (gw && uw && gw->bn == 16U && uw->bn == 16U && gw->n == n &&
          gw->k == k && uw->n == n && uw->k == k) {
        const std::string tg = affine_nax_mode_tag(gw->bits);
        const std::string tu = affine_nax_mode_tag(uw->bits);
        if (tg == tu || (tg == "q4" && tu == "q56") ||
            (tg == "q56" && tu == "q4")) {
          return true;
        }
      }
    }
    return false;
  }

  bool can_gpu_suffix_swiglu_fwht64_a8(const MaterializedWeightView &gate_w,
                                       const MaterializedWeightView &up_w,
                                       const MaterializedWeightView &down_w,
                                       U32 n, U32 k, bool use_w4a8) const {
    if (!use_w4a8 || !w4a8_nax_enabled_ || !down_w.has_fwht64() ||
        gate_w.has_fwht64() || up_w.has_fwht64() || (n % 64U) != 0U ||
        (k % 64U) != 0U || a8_qa.buffer == nullptr ||
        a8_mid_qa.buffer == nullptr || a8_mid_sa.buffer == nullptr ||
        a8_mid_ra.buffer == nullptr || has_bypass_lora(down_w) ||
        inspect_tensors_workspace || !can_w4a8_weight(down_w, h, n)) {
      return false;
    }
    return can_gpu_suffix_swiglu(gate_w, up_w, n, k, /*use_w4a8=*/true);
  }

  bool emit_gpu_suffix_swiglu(Encoder &e, const MaterializedWeightView &gate_w,
                              const MaterializedWeightView &up_w, Ref x, Ref y,
                              U32 n, U32 k, bool use_w4a8, U32 row_offset,
                              U32 out_stride, U32 slice_n,
                              std::optional<U32> explicit_t = {}) {
    if (use_w4a8 &&
        try_linear_w4a8_swiglu(e, gate_w, up_w, x, y, n, k, explicit_t,
                               row_offset, out_stride, slice_n)) {
      return true;
    }
    if (mxfp4_nax_active()) {
      const auto *gw = mxfp4_nax_cache->find(gate_w.object_id);
      const auto *uw = mxfp4_nax_cache->find(up_w.object_id);
      if (gw && uw && gw->bn == 16U && uw->bn == 16U && gw->n == n &&
          gw->k == k && uw->n == n && uw->k == k) {
        linear_mxfp4_nax_swiglu(e, *gw, *uw, x, f16_scratch, f16_row_scale, y,
                                silicon_tuning.gpu_cores, explicit_t,
                                row_offset, out_stride, slice_n);
        return true;
      }
    }
    if (affine_nax_active()) {
      const auto *gw = affine_nax_cache->find(gate_w.object_id);
      const auto *uw = affine_nax_cache->find(up_w.object_id);
      if (gw && uw && gw->bn == 16U && uw->bn == 16U && gw->n == n &&
          gw->k == k && uw->n == n && uw->k == k) {
        linear_affine_nax_swiglu(e, *gw, *uw, x, affine_nax_xs, y,
                                 silicon_tuning.gpu_cores, explicit_t,
                                 row_offset, out_stride, slice_n);
        return true;
      }
    }
    return false;
  }

  // 9e163eee N-concatenated sibling projection helper (2-way or 3-way).
  bool try_concat_nax(Encoder &e, U32 concat_id,
                      std::span<const MaterializedWeightView> parts,
                      std::span<const U32> part_ns, U32 k, bool input_is_f16,
                      Ref x_bf16, Ref dst_concat,
                      std::optional<U32> explicit_t = {}) {
    if (!concat_id || !dst_concat.buffer || parts.empty())
      return false;
    const U32 sel_t = nax_effective_select_t(e, explicit_t);
    U32 total_n = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
      if (is_overridden(parts[i]))
        return false;
      total_n += part_ns[i];
    }
    if (use_mxfp4_nax_for_weight(parts.front(), sel_t)) {
      if (const auto *tw = mxfp4_nax_cache->find(concat_id);
          tw && tw->n == total_n && tw->k == k) {
        if (input_is_f16) {
          linear_mxfp4_nax_direct_f16(e, *tw, f16_norm, f16_ones_scale,
                                      dst_concat, explicit_t,
                                      silicon_tuning.gpu_cores,
                                      /*unit_scale=*/true);
        } else {
          linear_mxfp4_nax(e, *tw, x_bf16, dst_concat, f16_scratch,
                           f16_row_scale, explicit_t, silicon_tuning.gpu_cores);
        }
        const std::string grp =
            mxfp4_nax_group(e, *tw, explicit_t, silicon_tuning.gpu_cores);
        for (std::size_t i = 0; i < parts.size(); ++i) {
          e.record_weight(parts[i], part_ns[i], k, "mxfp4_nax_m5_v1",
                          "mxfp4_nax_m5_v1", {}, "linear_row_exact_portable_v1",
                          grp, true, explicit_t);
        }
        return true;
      }
    }
    if (!input_is_f16 && use_affine_nax_for_weight(parts.front(), sel_t)) {
      if (const auto *tw = affine_nax_cache->find(concat_id);
          tw && tw->n == total_n && tw->k == k) {
        linear_affine_nax(e, *tw, x_bf16, affine_nax_xs, dst_concat, explicit_t,
                          silicon_tuning.gpu_cores);
        const std::string grp =
            affine_nax_group(e, *tw, explicit_t, silicon_tuning.gpu_cores);
        for (std::size_t i = 0; i < parts.size(); ++i) {
          e.record_weight(parts[i], part_ns[i], k, "affine_nax_m5_v1",
                          "affine_nax_m5_v1", {}, "linear_row_exact_portable_v1",
                          grp, true, explicit_t);
        }
        return true;
      }
    }
    return false;
  }

  void emit_split2(Encoder &e, Ref src, Ref dst0, Ref dst1, U32 n0, U32 n1,
                   std::optional<U32> explicit_t = {}) {
    const U32 t = explicit_t.value_or(e.profile.capacity);
    auto &d = e.emit("qwen_forward", "omlx_m5_forward_split2_cols",
                     {.width = U64(t) * (n0 + n1)}, {.width = 128},
                     {src, dst0, dst1});
    e.dynamic(d, 3, words({t, n0, n1}),
              explicit_t.has_value() ? std::optional<U32>{}
                                     : std::optional<U32>{0});
  }

  void emit_split3(Encoder &e, Ref src, Ref dst0, Ref dst1, Ref dst2, U32 n0,
                   U32 n1, U32 n2, std::optional<U32> explicit_t = {}) {
    const U32 t = explicit_t.value_or(e.profile.capacity);
    auto &d = e.emit("qwen_forward", "omlx_m5_forward_split3_cols",
                     {.width = U64(t) * (n0 + n1 + n2)}, {.width = 128},
                     {src, dst0, dst1, dst2});
    e.dynamic(d, 4, words({t, n0, n1, n2}),
              explicit_t.has_value() ? std::optional<U32>{}
                                     : std::optional<U32>{0});
  }

  void normalize(Encoder &e, const MaterializedWeightView &w, Ref x, Ref y,
                 std::optional<U32> explicit_rows = {}) {
    const U32 rows = explicit_rows.value_or(e.profile.capacity);
    auto &d = e.emit("rmsnorm_projection", "omlx_m5_rmsnorm_a16",
                     {.width = 128, .height = rows}, {.width = 128},
                     {x, direct(w, h), y});
    e.dynamic(d, 3,
              words({h, rows, word(float(config.rms_norm_eps)),
                     word(normalization_weight_offset)}),
              explicit_rows.has_value() || rows == 1 ? std::optional<U32>{}
                                                     : std::optional<U32>{1});
  }
  void add_normalize(Encoder &e, const MaterializedWeightView &w, Ref x,
                     Ref res_delta, Ref y,
                     std::optional<U32> explicit_rows = {}) {
    const U32 rows = explicit_rows.value_or(e.profile.capacity);
    auto &d = e.emit("rmsnorm_projection", "omlx_m5_add_rmsnorm_a16",
                     {.width = 128, .height = rows}, {.width = 128},
                     {x, res_delta, direct(w, h), y});
    e.dynamic(d, 4,
              words({h, rows, word(float(config.rms_norm_eps)),
                     word(normalization_weight_offset)}),
              explicit_rows.has_value() || rows == 1 ? std::optional<U32>{}
                                                     : std::optional<U32>{1});
  }
  void normalize_f16(Encoder &e, const MaterializedWeightView &w, Ref x,
                     Ref y_f16, std::optional<U32> explicit_rows = {}) {
    const U32 rows = explicit_rows.value_or(e.profile.capacity);
    const U32 mp = mxfp4_nax_mp(rows);
    auto &d = e.emit("rmsnorm_projection", "omlx_m5_rmsnorm_a16_f16",
                     {.width = 128, .height = mp}, {.width = 128},
                     {x, direct(w, h), y_f16});
    e.dynamic(d, 3,
              words({h, rows, word(float(config.rms_norm_eps)),
                     word(normalization_weight_offset)}),
              explicit_rows.has_value() || rows == 1 ? std::optional<U32>{}
                                                     : std::optional<U32>{1});
  }
  void add_normalize_f16(Encoder &e, const MaterializedWeightView &w, Ref x,
                         Ref res_delta, Ref y_f16,
                         std::optional<U32> explicit_rows = {}) {
    const U32 rows = explicit_rows.value_or(e.profile.capacity);
    const U32 mp = mxfp4_nax_mp(rows);
    auto &d = e.emit("rmsnorm_projection", "omlx_m5_add_rmsnorm_a16_f16",
                     {.width = 128, .height = mp}, {.width = 128},
                     {x, res_delta, direct(w, h), y_f16});
    e.dynamic(d, 4,
              words({h, rows, word(float(config.rms_norm_eps)),
                     word(normalization_weight_offset)}),
              explicit_rows.has_value() || rows == 1 ? std::optional<U32>{}
                                                     : std::optional<U32>{1});
  }
  void add(Encoder &e, std::optional<U32> explicit_rows = {}) {
    const U32 rows = explicit_rows.value_or(e.profile.capacity);
    auto &d = e.emit("qwen_forward", "omlx_m5_forward_add",
                     {.width = U64(rows) * h}, {.width = 128}, {hidden, delta});
    e.dynamic(d, 2, words({rows, h}),
              explicit_rows.has_value() ? std::optional<U32>{}
                                        : std::optional<U32>{0});
  }

  void normalize_init_zero_t1(Encoder &e, const MaterializedWeightView &w,
                              Ref x, Ref y, U32 num_sq_words,
                              U32 num_moe_words) {
    auto &d = e.emit("rmsnorm_projection", "omlx_m5_rmsnorm_a16_init_zero_t1",
                     {.width = 128, .height = 1}, {.width = 128},
                     {x, direct(w, h), y, rms_sq_buf, moe_tg_done_buf});
    e.dynamic(d, 5,
              words({h, num_sq_words, word(float(config.rms_norm_eps)),
                     word(normalization_weight_offset), num_moe_words}));
  }

  void normalize_init_zero_m2(Encoder &e, const MaterializedWeightView &w,
                              Ref x, Ref y, U32 num_sq_words,
                              U32 num_moe_words) {
    auto &d = e.emit("row_exact", "omlx_m5_rmsnorm_init_zero_m2",
                     {.width = 128, .height = 2}, {.width = 128},
                     {x, direct(w, h), y, rms_sq_buf, moe_tg_done_buf});
    e.dynamic(d, 5,
              words({h, num_sq_words, word(float(config.rms_norm_eps)),
                     word(normalization_weight_offset), num_moe_words}));
  }

  bool resolve_pretiled_mxfp4_m2(const MaterializedWeightView &w,
                                 Ref *out_codes = nullptr,
                                 Ref *out_scales = nullptr) const {
    if (!(!speculative_enabled && t_domain_active() &&
          pretiled_qmv_t1_enabled() && has_row_exact_metallib_ &&
          !is_overridden(w) && h >= 512U)) {
      return false;
    }
    return resolve_mxfp4_planes_any_t(w, out_codes, out_scales);
  }

  bool resolve_pretiled_affine_m2(const MaterializedWeightView &w,
                                  Ref *out_codes = nullptr,
                                  Ref *out_high = nullptr,
                                  Ref *out_sb = nullptr,
                                  bool *out_lane_v2 = nullptr) const {
    if (!(!speculative_enabled && t_domain_active() &&
          pretiled_qmv_t1_enabled() && has_row_exact_metallib_ &&
          !is_overridden(w) && h >= 512U)) {
      return false;
    }
    return resolve_affine_planes_any_t(w, out_codes, out_high, out_sb,
                                       out_lane_v2);
  }

  bool try_linear_m2_base(Encoder &e, const MaterializedWeightView &w, Ref x,
                          Ref y, U32 n, U32 k) {
    if (w.row_count != n || w.in_features != k)
      return false;
    Ref mx_c{}, mx_s{};
    if (resolve_pretiled_mxfp4_m2(w, &mx_c, &mx_s)) {
      const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
      const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
      const char *fn = (n >= 65536U)  ? "omlx_m5_pretiled_qmv_mxfp4_bn32_r4_m2"
                       : (n >= 2048U) ? "omlx_m5_pretiled_qmv_mxfp4_bn16_r4_m2"
                                      : "omlx_m5_pretiled_qmv_mxfp4_bn16_r2_m2";
      auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                       {.width = 64U}, {x, mx_c, mx_s, y});
      e.dynamic(d, 4,
                words({n, pretiled_k_flags(k, false, w.has_fwht64()),
                       U32(w.padded_k), 2U}));
      e.record_weight(w, n, k, "pretiled_qmv_m2", "pretiled_mxfp4_qmv_m2_v1",
                      {}, "legacy_route_v1", {}, false, 2U);
      return true;
    }
    Ref aff_c{}, aff_h{}, aff_sb{};
    bool lane_v2 = false;
    if (resolve_pretiled_affine_m2(w, &aff_c, &aff_h, &aff_sb, &lane_v2)) {
      const U32 bits = affine_nax_bits(w.format);
      const U32 tgs = (n + 3U) / 4U;
      const char *fn = "omlx_m5_pretiled_qmv_affine_r2_m2";
      if (bits == 4U)
        fn = (n >= 65536U) ? "omlx_m5_pretiled_qmv_affine_q4_bn32_r2_m2"
                           : "omlx_m5_pretiled_qmv_affine_q4_bn16_r2_m2";
      else if (bits == 5U)
        fn = (n >= 65536U) ? "omlx_m5_pretiled_qmv_affine_q5_bn32_r2_m2"
                           : "omlx_m5_pretiled_qmv_affine_q5_bn16_r2_m2";
      else if (bits == 6U)
        fn = (n >= 65536U) ? "omlx_m5_pretiled_qmv_affine_q6_bn32_r2_m2"
                           : "omlx_m5_pretiled_qmv_affine_q6_bn16_r2_m2";
      else if (bits == 8U)
        fn = (n >= 65536U) ? "omlx_m5_pretiled_qmv_affine_q8_bn32_r2_m2"
                           : "omlx_m5_pretiled_qmv_affine_q8_bn16_r2_m2";
      auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                       {.width = 64U}, {x, aff_c, aff_h, aff_sb, y});
      e.dynamic(d, 5,
                words({n, 2U, 0U, pretiled_k_flags(k, lane_v2, w.has_fwht64()),
                       bits}));
      e.record_weight(w, n, k, "pretiled_qmv_m2", "pretiled_affine_qmv_m2_v1",
                      {}, "legacy_route_v1", {}, false, 2U);
      return true;
    }
    return false;
  }

  bool try_linear_m2(Encoder &e, const MaterializedWeightView &w, Ref x, Ref y,
                     U32 n, U32 k) {
    if (!try_linear_m2_base(e, w, x, y, n, k))
      return false;
    emit_bypass_for_weight(e, w, x, y, n, 2U);
    return true;
  }

  bool try_linear_add_sq_m2(Encoder &e, const MaterializedWeightView &w, Ref x,
                            Ref hidden_io, U32 n, U32 k, U32 sq_slot) {
    if (has_bypass_lora(w) || n >= 65536U || !rms_sq_buf.buffer ||
        w.row_count != n || w.in_features != k) {
      return false;
    }
    Ref mx_c{}, mx_s{};
    if (resolve_pretiled_mxfp4_m2(w, &mx_c, &mx_s)) {
      const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
      const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
      const char *fn = (n >= 2048U)
                           ? "omlx_m5_pretiled_qmv_mxfp4_bn16_r4_add_sq_m2"
                           : "omlx_m5_pretiled_qmv_mxfp4_bn16_r2_add_sq_m2";
      auto &d =
          e.emit("row_exact", fn, {.width = U64(tgs) * 64U}, {.width = 64U},
                 {x, mx_c, mx_s, hidden_io, rms_sq_buf});
      e.dynamic(d, 5,
                fused_norm_params(n, 0U, 0U, 0U,
                                  pretiled_k_flags(k, false, w.has_fwht64()),
                                  0U, sq_slot));
      e.record_weight(w, n, k, "pretiled_qmv_m2",
                      "pretiled_mxfp4_qmv_add_sq_m2_v1", {}, "legacy_route_v1",
                      {}, false, 2U);
      return true;
    }
    Ref aff_c{}, aff_h{}, aff_sb{};
    bool lane_v2 = false;
    if (resolve_pretiled_affine_m2(w, &aff_c, &aff_h, &aff_sb, &lane_v2)) {
      const U32 bits = affine_nax_bits(w.format);
      const U32 tgs = (n + 3U) / 4U;
      const char *fn = "omlx_m5_pretiled_qmv_affine_r2_add_sq_m2";
      if (bits == 4U)
        fn = "omlx_m5_pretiled_qmv_affine_q4_bn16_r2_add_sq_m2";
      else if (bits == 5U)
        fn = "omlx_m5_pretiled_qmv_affine_q5_bn16_r2_add_sq_m2";
      else if (bits == 6U)
        fn = "omlx_m5_pretiled_qmv_affine_q6_bn16_r2_add_sq_m2";
      else if (bits == 8U)
        fn = "omlx_m5_pretiled_qmv_affine_q8_bn16_r2_add_sq_m2";
      auto &d =
          e.emit("row_exact", fn, {.width = U64(tgs) * 64U}, {.width = 64U},
                 {x, aff_c, aff_h, aff_sb, hidden_io, rms_sq_buf});
      e.dynamic(d, 6,
                fused_norm_params(n, 0U, 0U, 0U,
                                  pretiled_k_flags(k, lane_v2, w.has_fwht64()),
                                  bits, sq_slot));
      e.record_weight(w, n, k, "pretiled_qmv_m2",
                      "pretiled_affine_qmv_add_sq_m2_v1", {}, "legacy_route_v1",
                      {}, false, 2U);
      return true;
    }
    return false;
  }

  bool try_linear_norm_m2(Encoder &e, const MaterializedWeightView &w,
                          const MaterializedWeightView &norm_w, Ref hidden_in,
                          Ref y, U32 n, U32 k, U32 sq_slot) {
    if (has_bypass_lora(w) || !rms_sq_buf.buffer || w.row_count != n ||
        w.in_features != k) {
      return false;
    }
    Ref mx_c{}, mx_s{};
    if (n >= 2048U && resolve_pretiled_mxfp4_m2(w, &mx_c, &mx_s)) {
      const Ref nw_ref = direct(norm_w, k);
      const U32 tgs = (n + 7U) / 8U;
      const char *fn = (n >= 65536U)
                           ? "omlx_m5_pretiled_qmv_mxfp4_bn32_r4_norm_m2"
                           : "omlx_m5_pretiled_qmv_mxfp4_bn16_r4_norm_m2";
      auto &d =
          e.emit("row_exact", fn, {.width = U64(tgs) * 64U}, {.width = 64U},
                 {hidden_in, nw_ref, rms_sq_buf, mx_c, mx_s, y});
      e.dynamic(d, 6,
                fused_norm_params(n, 0U, 0U, 0U,
                                  pretiled_k_flags(k, false, w.has_fwht64()),
                                  0U, sq_slot));
      e.record_weight(w, n, k, "pretiled_qmv_m2",
                      "pretiled_mxfp4_qmv_norm_m2_v1", {}, "legacy_route_v1",
                      {}, false, 2U);
      return true;
    }
    Ref aff_c{}, aff_h{}, aff_sb{};
    bool lane_v2 = false;
    if (resolve_pretiled_affine_m2(w, &aff_c, &aff_h, &aff_sb, &lane_v2)) {
      const Ref nw_ref = direct(norm_w, k);
      const U32 bits = affine_nax_bits(w.format);
      const U32 tgs = (n + 3U) / 4U;
      const char *fn = "omlx_m5_pretiled_qmv_affine_r2_norm_m2";
      if (bits == 4U)
        fn = (n >= 65536U) ? "omlx_m5_pretiled_qmv_affine_q4_bn32_r2_norm_m2"
                           : "omlx_m5_pretiled_qmv_affine_q4_bn16_r2_norm_m2";
      else if (bits == 5U)
        fn = (n >= 65536U) ? "omlx_m5_pretiled_qmv_affine_q5_bn32_r2_norm_m2"
                           : "omlx_m5_pretiled_qmv_affine_q5_bn16_r2_norm_m2";
      else if (bits == 6U)
        fn = (n >= 65536U) ? "omlx_m5_pretiled_qmv_affine_q6_bn32_r2_norm_m2"
                           : "omlx_m5_pretiled_qmv_affine_q6_bn16_r2_norm_m2";
      else if (bits == 8U)
        fn = (n >= 65536U) ? "omlx_m5_pretiled_qmv_affine_q8_bn32_r2_norm_m2"
                           : "omlx_m5_pretiled_qmv_affine_q8_bn16_r2_norm_m2";
      auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                       {.width = 64U},
                       {hidden_in, nw_ref, rms_sq_buf, aff_c, aff_h, aff_sb, y});
      e.dynamic(d, 7,
                fused_norm_params(n, 0U, 0U, 0U,
                                  pretiled_k_flags(k, lane_v2, w.has_fwht64()),
                                  bits, sq_slot));
      e.record_weight(w, n, k, "pretiled_qmv_m2",
                      "pretiled_affine_qmv_norm_m2_v1", {}, "legacy_route_v1",
                      {}, false, 2U);
      return true;
    }
    return false;
  }

  bool try_linear_swiglu_m2(Encoder &e, const MaterializedWeightView &gate_w,
                            const MaterializedWeightView &up_w, Ref x, Ref y,
                            U32 n, U32 k,
                            const MaterializedWeightView *norm_w = nullptr,
                            std::optional<U32> sq_slot = std::nullopt) {
    if (n >= 65536U || is_overridden(gate_w) || is_overridden(up_w) ||
        has_bypass_lora(gate_w) || has_bypass_lora(up_w) ||
        gate_w.padded_k != up_w.padded_k || gate_w.row_count != n ||
        gate_w.in_features != k || up_w.row_count != n ||
        up_w.in_features != k) {
      return false;
    }
    const bool fused = (norm_w != nullptr && sq_slot.has_value() &&
                        rms_sq_buf.buffer != nullptr);
    if (gate_w.has_fwht64() != up_w.has_fwht64()) {
      return false;
    }
    Ref gc{}, gs{}, uc{}, us{};
    if (resolve_pretiled_mxfp4_m2(gate_w, &gc, &gs) &&
        resolve_pretiled_mxfp4_m2(up_w, &uc, &us)) {
      const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
      const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
      if (fused) {
        const Ref nw_ref = direct(*norm_w, k);
        const char *fn =
            (n >= 2048U) ? "omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r4_norm_m2"
                         : "omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r2_norm_m2";
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U},
                         {x, nw_ref, rms_sq_buf, gc, gs, uc, us, y});
        e.dynamic(d, 8,
                  fused_norm_params(n, 0U, 0U, 0U,
                                    pretiled_k_flags(k, false, gate_w.has_fwht64()),
                                    0U, *sq_slot));
      } else {
        const char *fn =
            (n >= 2048U) ? "omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r4_m2"
                         : "omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r2_m2";
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U}, {x, gc, gs, uc, us, y});
        e.dynamic(d, 6,
                  words({n, pretiled_k_flags(k, false, gate_w.has_fwht64()),
                         U32(gate_w.padded_k), 2U}));
      }
      const char *src = fused ? "pretiled_mxfp4_qmv_swiglu_norm_m2_v1"
                              : "pretiled_mxfp4_qmv_swiglu_m2_v1";
      e.record_weight(gate_w, n, k, "pretiled_qmv_m2", src, {},
                      "legacy_route_v1", {}, false, 2U);
      e.record_weight(up_w, n, k, "pretiled_qmv_m2", src, {},
                      "legacy_route_v1", {}, false, 2U);
      return true;
    }
    Ref agc{}, agh{}, agsb{}, auc{}, auh{}, ausb{};
    bool g_v2 = false, u_v2 = false;
    if (resolve_pretiled_affine_m2(gate_w, &agc, &agh, &agsb, &g_v2) &&
        resolve_pretiled_affine_m2(up_w, &auc, &auh, &ausb, &u_v2) &&
        g_v2 == u_v2) {
      const U32 g_bits = affine_nax_bits(gate_w.format);
      const U32 u_bits = affine_nax_bits(up_w.format);
      const U32 tgs = (n + 3U) / 4U;
      const U32 packed_bits = g_bits | (u_bits << 8U);
      const U32 kf = pretiled_k_flags(k, g_v2, gate_w.has_fwht64());
      if (fused) {
        const Ref nw_ref = direct(*norm_w, k);
        const char *fn = "omlx_m5_pretiled_qmv_affine_swiglu_r2_norm_m2";
        if (g_bits == u_bits) {
          if (g_bits == 4U)
            fn = "omlx_m5_pretiled_qmv_affine_swiglu_q4_r2_norm_m2";
          else if (g_bits == 5U)
            fn = "omlx_m5_pretiled_qmv_affine_swiglu_q5_r2_norm_m2";
          else if (g_bits == 6U)
            fn = "omlx_m5_pretiled_qmv_affine_swiglu_q6_r2_norm_m2";
          else if (g_bits == 8U)
            fn = "omlx_m5_pretiled_qmv_affine_swiglu_q8_r2_norm_m2";
        }
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U},
                         {x, nw_ref, rms_sq_buf, agc, agh, agsb, auc, auh, ausb,
                          y});
        e.dynamic(d, 10,
                  fused_norm_params(n, 0U, 0U, 0U, kf, packed_bits, *sq_slot));
      } else {
        const char *fn = "omlx_m5_pretiled_qmv_affine_swiglu_r2_m2";
        if (g_bits == u_bits) {
          if (g_bits == 4U)
            fn = "omlx_m5_pretiled_qmv_affine_swiglu_q4_r2_m2";
          else if (g_bits == 5U)
            fn = "omlx_m5_pretiled_qmv_affine_swiglu_q5_r2_m2";
          else if (g_bits == 6U)
            fn = "omlx_m5_pretiled_qmv_affine_swiglu_q6_r2_m2";
          else if (g_bits == 8U)
            fn = "omlx_m5_pretiled_qmv_affine_swiglu_q8_r2_m2";
        }
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U},
                         {x, agc, agh, agsb, auc, auh, ausb, y});
        e.dynamic(d, 8, words({n, 2U, 0U, kf, packed_bits}));
      }
      const char *src = fused ? "pretiled_affine_qmv_swiglu_norm_m2_v1"
                              : "pretiled_affine_qmv_swiglu_m2_v1";
      e.record_weight(gate_w, n, k, "pretiled_qmv_m2", src, {},
                      "legacy_route_v1", {}, false, 2U);
      e.record_weight(up_w, n, k, "pretiled_qmv_m2", src, {},
                      "legacy_route_v1", {}, false, 2U);
      return true;
    }
    return false;
  }

  bool try_concat3_m2(Encoder &e,
                      const std::array<MaterializedWeightView, 3> &parts,
                      const std::array<U32, 3> &ns, U32 k, Ref x, Ref y0,
                      Ref y1, Ref y2,
                      const MaterializedWeightView *norm_w = nullptr,
                      std::optional<U32> sq_slot = std::nullopt) {
    if (ns[0] >= 65536U || ns[1] >= 65536U || ns[2] >= 65536U ||
        parts[0].row_count != ns[0] || parts[1].row_count != ns[1] ||
        parts[2].row_count != ns[2] || parts[0].in_features != k ||
        parts[1].in_features != k || parts[2].in_features != k) {
      return false;
    }
    const bool fused = (norm_w != nullptr && sq_slot.has_value() &&
                        rms_sq_buf.buffer != nullptr);
    if (fused && (has_bypass_lora(parts[0]) || has_bypass_lora(parts[1]) ||
                  has_bypass_lora(parts[2]))) {
      return false;
    }
    Ref d0{}, s0{}, d1{}, s1{}, d2{}, s2{};
    if (resolve_pretiled_mxfp4_m2(parts[0], &d0, &s0) &&
        resolve_pretiled_mxfp4_m2(parts[1], &d1, &s1) &&
        resolve_pretiled_mxfp4_m2(parts[2], &d2, &s2)) {
      const bool use_r4 = ((ns[0] + ns[1] + ns[2]) >= 2048U);
      const U32 rptg = use_r4 ? 8U : 4U;
      const U32 tgs = ((ns[0] + rptg - 1U) / rptg) +
                      ((ns[1] + rptg - 1U) / rptg) +
                      ((ns[2] + rptg - 1U) / rptg);
      if (fused) {
        const Ref nw_ref = direct(*norm_w, k);
        const char *fn =
            use_r4 ? "omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r4_norm_m2"
                   : "omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r2_norm_m2";
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U},
                         {x, nw_ref, rms_sq_buf, d0, s0, d1, s1, d2, s2, y0, y1,
                          y2});
        e.dynamic(d, 12,
                  fused_norm_params(ns[0], ns[1], ns[2], 0U, k, 0U, *sq_slot));
      } else {
        const char *fn =
            use_r4 ? "omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r4_m2"
                   : "omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r2_m2";
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U},
                         {x, d0, s0, d1, s1, d2, s2, y0, y1, y2});
        e.dynamic(d, 10, words({ns[0], ns[1], ns[2], k}));
      }
      const char *src = fused ? "pretiled_mxfp4_qmv_concat3_norm_m2_v1"
                              : "pretiled_mxfp4_qmv_concat3_m2_v1";
      for (std::size_t i = 0; i < 3; ++i) {
        e.record_weight(parts[i], ns[i], k, "pretiled_qmv_m2", src, {},
                        "legacy_route_v1", {}, false, 2U);
      }
      return true;
    }
    Ref h0{}, h1{}, h2{};
    bool lv0 = false, lv1 = false, lv2 = false;
    if (resolve_pretiled_affine_m2(parts[0], &d0, &h0, &s0, &lv0) &&
        resolve_pretiled_affine_m2(parts[1], &d1, &h1, &s1, &lv1) &&
        resolve_pretiled_affine_m2(parts[2], &d2, &h2, &s2, &lv2) &&
        lv0 == lv1 && lv1 == lv2 &&
        parts[0].has_fwht64() == parts[1].has_fwht64() &&
        parts[1].has_fwht64() == parts[2].has_fwht64()) {
      const U32 b0 = affine_nax_bits(parts[0].format);
      const U32 b1 = affine_nax_bits(parts[1].format);
      const U32 b2 = affine_nax_bits(parts[2].format);
      const U32 packed_bits = b0 | (b1 << 8U) | (b2 << 16U);
      const U32 kf = pretiled_k_flags(k, lv0, parts[0].has_fwht64());
      const U32 tgs =
          ((ns[0] + 3U) / 4U) + ((ns[1] + 3U) / 4U) + ((ns[2] + 3U) / 4U);
      const bool all_q456 = (b0 >= 4U && b0 <= 6U) && (b1 >= 4U && b1 <= 6U) &&
                            (b2 >= 4U && b2 <= 6U);
      if (fused) {
        const Ref nw_ref = direct(*norm_w, k);
        const char *fn = "omlx_m5_pretiled_qmv_affine_concat3_r2_norm_m2";
        if (b0 == b1 && b1 == b2 && (b0 == 4U || b0 == 5U || b0 == 6U)) {
          fn = (b0 == 5U)
                   ? "omlx_m5_pretiled_qmv_affine_concat3_q5_r2_norm_m2"
               : (b0 == 4U)
                   ? "omlx_m5_pretiled_qmv_affine_concat3_q4_r2_norm_m2"
                   : "omlx_m5_pretiled_qmv_affine_concat3_q6_r2_norm_m2";
        } else if (all_q456) {
          fn = "omlx_m5_pretiled_qmv_affine_concat3_q456_r2_norm_m2";
        }
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U},
                         {x, nw_ref, rms_sq_buf, d0, h0, s0, d1, h1, s1, d2, h2,
                          s2, y0, y1, y2});
        e.dynamic(d, 15,
                  fused_norm_params(ns[0], ns[1], ns[2], 0U, kf, packed_bits,
                                    *sq_slot));
      } else {
        const char *fn = "omlx_m5_pretiled_qmv_affine_concat3_r2_m2";
        if (b0 == b1 && b1 == b2 && (b0 == 4U || b0 == 5U || b0 == 6U)) {
          fn = (b0 == 5U)   ? "omlx_m5_pretiled_qmv_affine_concat3_q5_r2_m2"
               : (b0 == 4U) ? "omlx_m5_pretiled_qmv_affine_concat3_q4_r2_m2"
                            : "omlx_m5_pretiled_qmv_affine_concat3_q6_r2_m2";
        } else if (all_q456) {
          fn = "omlx_m5_pretiled_qmv_affine_concat3_q456_r2_m2";
        }
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U},
                         {x, d0, h0, s0, d1, h1, s1, d2, h2, s2, y0, y1, y2});
        e.dynamic(d, 13, words({ns[0], ns[1], ns[2], kf, packed_bits}));
      }
      const char *src = fused ? "pretiled_affine_qmv_concat3_norm_m2_v1"
                              : "pretiled_affine_qmv_concat3_m2_v1";
      for (std::size_t i = 0; i < 3; ++i) {
        e.record_weight(parts[i], ns[i], k, "pretiled_qmv_m2", src, {},
                        "legacy_route_v1", {}, false, 2U);
      }
      return true;
    }
    return false;
  }

  bool try_concat4_m2(Encoder &e,
                      const std::array<MaterializedWeightView, 4> &parts,
                      const std::array<U32, 4> &ns, U32 k, Ref x, Ref y0,
                      Ref y1, Ref y2, Ref y3,
                      const MaterializedWeightView *norm_w = nullptr,
                      std::optional<U32> sq_slot = std::nullopt) {
    if (ns[0] >= 65536U || ns[1] >= 65536U || ns[2] >= 65536U ||
        ns[3] >= 65536U || parts[0].row_count != ns[0] ||
        parts[1].row_count != ns[1] || parts[2].row_count != ns[2] ||
        parts[3].row_count != ns[3] || parts[0].in_features != k ||
        parts[1].in_features != k || parts[2].in_features != k ||
        parts[3].in_features != k) {
      return false;
    }
    const bool fused = (norm_w != nullptr && sq_slot.has_value() &&
                        rms_sq_buf.buffer != nullptr);
    if (fused && (has_bypass_lora(parts[0]) || has_bypass_lora(parts[1]) ||
                  has_bypass_lora(parts[2]) || has_bypass_lora(parts[3]))) {
      return false;
    }
    Ref d0{}, s0{}, d1{}, s1{}, d2{}, s2{}, d3{}, s3{};
    if (resolve_pretiled_mxfp4_m2(parts[0], &d0, &s0) &&
        resolve_pretiled_mxfp4_m2(parts[1], &d1, &s1) &&
        resolve_pretiled_mxfp4_m2(parts[2], &d2, &s2) &&
        resolve_pretiled_mxfp4_m2(parts[3], &d3, &s3)) {
      const bool use_r4 = ((ns[0] + ns[1] + ns[2] + ns[3]) >= 2048U);
      const U32 rptg = use_r4 ? 8U : 4U;
      const U32 tgs = ((ns[0] + rptg - 1U) / rptg) +
                      ((ns[1] + rptg - 1U) / rptg) +
                      ((ns[2] + rptg - 1U) / rptg) +
                      ((ns[3] + rptg - 1U) / rptg);
      if (fused) {
        const Ref nw_ref = direct(*norm_w, k);
        const char *fn = use_r4 ? "omlx_m5_pretiled_qmv_mxfp4_concat4_r4_norm_m2"
                                : "omlx_m5_pretiled_qmv_mxfp4_concat4_r2_norm_m2";
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U},
                         {x, nw_ref, rms_sq_buf, d0, s0, d1, s1, d2, s2, d3, s3,
                          y0, y1, y2, y3});
        e.dynamic(d, 15,
                  fused_norm_params(ns[0], ns[1], ns[2], ns[3], k, 0U,
                                    *sq_slot));
      } else {
        const char *fn = use_r4 ? "omlx_m5_pretiled_qmv_mxfp4_concat4_r4_m2"
                                : "omlx_m5_pretiled_qmv_mxfp4_concat4_r2_m2";
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U},
                         {x, d0, s0, d1, s1, d2, s2, d3, s3, y0, y1, y2, y3});
        e.dynamic(d, 13, words({ns[0], ns[1], ns[2], ns[3], k}));
      }
      const char *src = fused ? "pretiled_mxfp4_qmv_concat4_norm_m2_v1"
                              : "pretiled_mxfp4_qmv_concat4_m2_v1";
      for (std::size_t i = 0; i < 4; ++i) {
        e.record_weight(parts[i], ns[i], k, "pretiled_qmv_m2", src, {},
                        "legacy_route_v1", {}, false, 2U);
      }
      return true;
    }
    Ref h0{}, h1{}, h2{}, h3{};
    bool lv0 = false, lv1 = false, lv2 = false, lv3 = false;
    if (resolve_pretiled_affine_m2(parts[0], &d0, &h0, &s0, &lv0) &&
        resolve_pretiled_affine_m2(parts[1], &d1, &h1, &s1, &lv1) &&
        resolve_pretiled_affine_m2(parts[2], &d2, &h2, &s2, &lv2) &&
        resolve_pretiled_affine_m2(parts[3], &d3, &h3, &s3, &lv3) &&
        lv0 == lv1 && lv1 == lv2 && lv2 == lv3 &&
        parts[0].has_fwht64() == parts[1].has_fwht64() &&
        parts[1].has_fwht64() == parts[2].has_fwht64() &&
        parts[2].has_fwht64() == parts[3].has_fwht64()) {
      const U32 b0 = affine_nax_bits(parts[0].format);
      const U32 b1 = affine_nax_bits(parts[1].format);
      const U32 b2 = affine_nax_bits(parts[2].format);
      const U32 b3 = affine_nax_bits(parts[3].format);
      const U32 packed_bits = b0 | (b1 << 8U) | (b2 << 16U) | (b3 << 24U);
      const U32 kf = pretiled_k_flags(k, lv0, parts[0].has_fwht64());
      const U32 tgs = ((ns[0] + 3U) / 4U) + ((ns[1] + 3U) / 4U) +
                      ((ns[2] + 3U) / 4U) + ((ns[3] + 3U) / 4U);
      const bool all_q456 = (b0 >= 4U && b0 <= 6U) && (b1 >= 4U && b1 <= 6U) &&
                            (b2 >= 4U && b2 <= 6U) && (b3 >= 4U && b3 <= 6U);
      if (fused) {
        const Ref nw_ref = direct(*norm_w, k);
        const char *fn =
            all_q456 ? "omlx_m5_pretiled_qmv_affine_concat4_q456_r2_norm_m2"
                     : "omlx_m5_pretiled_qmv_affine_concat4_r2_norm_m2";
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U},
                         {x, nw_ref, rms_sq_buf, d0, h0, s0, d1, h1, s1, d2, h2,
                          s2, d3, h3, s3, y0, y1, y2, y3});
        e.dynamic(d, 19,
                  fused_norm_params(ns[0], ns[1], ns[2], ns[3], kf, packed_bits,
                                    *sq_slot));
      } else {
        const char *fn =
            all_q456 ? "omlx_m5_pretiled_qmv_affine_concat4_q456_r2_m2"
                     : "omlx_m5_pretiled_qmv_affine_concat4_r2_m2";
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U},
                         {x, d0, h0, s0, d1, h1, s1, d2, h2, s2, d3, h3, s3, y0,
                          y1, y2, y3});
        e.dynamic(d, 17, words({ns[0], ns[1], ns[2], ns[3], kf, packed_bits}));
      }
      const char *src = fused ? "pretiled_affine_qmv_concat4_norm_m2_v1"
                              : "pretiled_affine_qmv_concat4_m2_v1";
      for (std::size_t i = 0; i < 4; ++i) {
        e.record_weight(parts[i], ns[i], k, "pretiled_qmv_m2", src, {},
                        "legacy_route_v1", {}, false, 2U);
      }
      return true;
    }
    return false;
  }

  std::vector<std::byte> fused_norm_params(U32 n0, U32 n1, U32 n2, U32 n3,
                                           U32 k_in, U32 bits_packed,
                                           U32 sq_slot) const {
    const U32 k_eff = k_in & 0x3FFFFFFFU;
    return words({n0, n1, n2, n3, k_in, bits_packed, sq_slot,
                  word(1.0F / float(k_eff)), word(float(config.rms_norm_eps)),
                  word(normalization_weight_offset)});
  }

  bool try_linear_add_sq_t1(Encoder &e, const MaterializedWeightView &w, Ref x,
                            Ref hidden_io, U32 n, U32 k, U32 sq_slot) {
    if (has_bypass_lora(w) || e.profile.capacity != 1U || n >= 65536U ||
        !rms_sq_buf.buffer || w.row_count != n || w.in_features != k) {
      return false;
    }
    Ref mx_c{}, mx_s{};
    if (resolve_pretiled_mxfp4_t1(w, 1U, &mx_c, &mx_s)) {
      const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
      const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
      const char *fn = (n >= 2048U)
                           ? "omlx_m5_pretiled_qmv_mxfp4_bn16_r4_add_sq_t1"
                           : "omlx_m5_pretiled_qmv_mxfp4_bn16_r2_add_sq_t1";
      auto &d =
          e.emit("row_exact", fn, {.width = U64(tgs) * 64U}, {.width = 64U},
                 {x, mx_c, mx_s, hidden_io, rms_sq_buf});
      e.dynamic(d, 5,
                fused_norm_params(n, 0U, 0U, 0U,
                                  pretiled_k_flags(k, false, w.has_fwht64()),
                                  0U, sq_slot));
      e.record_weight(w, n, k, "pretiled_qmv_t1",
                      "pretiled_mxfp4_qmv_add_sq_t1_v4", {}, "legacy_route_v1",
                      {}, false);
      return true;
    }
    Ref aff_c{}, aff_h{}, aff_sb{};
    bool lane_v2 = false;
    if (resolve_pretiled_affine_t1(w, 1U, &aff_c, &aff_h, &aff_sb, &lane_v2)) {
      const U32 bits = affine_nax_bits(w.format);
      const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
      const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
      const char *fn = "omlx_m5_pretiled_qmv_affine_add_sq_t1";
      if (n >= 2048U) {
        if (bits == 4U)
          fn = "omlx_m5_pretiled_qmv_affine_q4_bn16_r4_add_sq_t1";
        else if (bits == 5U)
          fn = "omlx_m5_pretiled_qmv_affine_q5_bn16_r4_add_sq_t1";
        else if (bits == 6U)
          fn = "omlx_m5_pretiled_qmv_affine_q6_bn16_r4_add_sq_t1";
        else if (bits == 8U)
          fn = "omlx_m5_pretiled_qmv_affine_q8_bn16_r4_add_sq_t1";
      }
      auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                       {.width = 64U},
                       {x, aff_c, aff_h, aff_sb, hidden_io, rms_sq_buf});
      e.dynamic(d, 6,
                fused_norm_params(n, 0U, 0U, 0U,
                                  pretiled_k_flags(k, lane_v2, w.has_fwht64()),
                                  bits, sq_slot));
      e.record_weight(w, n, k, "pretiled_qmv_t1",
                      "pretiled_affine_qmv_add_sq_t1_v4", {}, "legacy_route_v1",
                      {}, false);
      return true;
    }
    return false;
  }

  bool try_linear_norm_t1(Encoder &e, const MaterializedWeightView &w,
                          const MaterializedWeightView &norm_w, Ref hidden_in,
                          Ref y, U32 n, U32 k, U32 sq_slot) {
    if (has_bypass_lora(w) || e.profile.capacity != 1U || !rms_sq_buf.buffer ||
        w.row_count != n || w.in_features != k) {
      return false;
    }
    Ref mx_c{}, mx_s{};
    if (resolve_pretiled_mxfp4_t1(w, 1U, &mx_c, &mx_s)) {
      const Ref nw_ref = direct(norm_w, k);
      const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
      const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
      const char *fn =
          (n >= 65536U)  ? "omlx_m5_pretiled_qmv_mxfp4_bn32_r4_norm_t1"
          : (n >= 2048U) ? "omlx_m5_pretiled_qmv_mxfp4_bn16_r4_norm_t1"
                         : "omlx_m5_pretiled_qmv_mxfp4_bn16_r2_norm_t1";
      auto &d =
          e.emit("row_exact", fn, {.width = U64(tgs) * 64U}, {.width = 64U},
                 {hidden_in, nw_ref, rms_sq_buf, mx_c, mx_s, y});
      e.dynamic(d, 6,
                fused_norm_params(n, 0U, 0U, 0U,
                                  pretiled_k_flags(k, false, w.has_fwht64()),
                                  0U, sq_slot));
      e.record_weight(w, n, k, "pretiled_qmv_t1",
                      "pretiled_mxfp4_qmv_norm_t1_v4", {}, "legacy_route_v1",
                      {}, false);
      return true;
    }
    Ref aff_c{}, aff_h{}, aff_sb{};
    bool lane_v2 = false;
    if (resolve_pretiled_affine_t1(w, 1U, &aff_c, &aff_h, &aff_sb, &lane_v2)) {
      const Ref nw_ref = direct(norm_w, k);
      const U32 bits = affine_nax_bits(w.format);
      const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
      const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
      const char *fn = "omlx_m5_pretiled_qmv_affine_norm_t1";
      if (n >= 65536U) {
        if (bits == 4U)
          fn = "omlx_m5_pretiled_qmv_affine_q4_bn32_r4_norm_t1";
        else if (bits == 5U)
          fn = "omlx_m5_pretiled_qmv_affine_q5_bn32_r4_norm_t1";
        else if (bits == 6U)
          fn = "omlx_m5_pretiled_qmv_affine_q6_bn32_r4_norm_t1";
        else if (bits == 8U)
          fn = "omlx_m5_pretiled_qmv_affine_q8_bn32_r4_norm_t1";
      } else if (n >= 2048U) {
        if (bits == 4U)
          fn = "omlx_m5_pretiled_qmv_affine_q4_bn16_r4_norm_t1";
        else if (bits == 5U)
          fn = "omlx_m5_pretiled_qmv_affine_q5_bn16_r4_norm_t1";
        else if (bits == 6U)
          fn = "omlx_m5_pretiled_qmv_affine_q6_bn16_r4_norm_t1";
        else if (bits == 8U)
          fn = "omlx_m5_pretiled_qmv_affine_q8_bn16_r4_norm_t1";
      }
      auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                       {.width = 64U},
                       {hidden_in, nw_ref, rms_sq_buf, aff_c, aff_h, aff_sb, y});
      e.dynamic(d, 7,
                fused_norm_params(n, 0U, 0U, 0U,
                                  pretiled_k_flags(k, lane_v2, w.has_fwht64()),
                                  bits, sq_slot));
      e.record_weight(w, n, k, "pretiled_qmv_t1",
                      "pretiled_affine_qmv_norm_t1_v4", {}, "legacy_route_v1",
                      {}, false);
      return true;
    }
    return false;
  }

  bool try_linear_swiglu_norm_t1(Encoder &e,
                                 const MaterializedWeightView &gate_w,
                                 const MaterializedWeightView &up_w,
                                 const MaterializedWeightView &norm_w,
                                 Ref hidden_in, Ref y, U32 n, U32 k,
                                 U32 sq_slot) {
    if (e.profile.capacity != 1U || n >= 65536U || !rms_sq_buf.buffer ||
        is_overridden(gate_w) || is_overridden(up_w) ||
        has_bypass_lora(gate_w) || has_bypass_lora(up_w) ||
        gate_w.padded_k != up_w.padded_k || gate_w.row_count != n ||
        gate_w.in_features != k || up_w.row_count != n ||
        up_w.in_features != k) {
      return false;
    }
    Ref gc{}, gs{}, uc{}, us{};
    if (resolve_pretiled_mxfp4_t1(gate_w, 1U, &gc, &gs) &&
        resolve_pretiled_mxfp4_t1(up_w, 1U, &uc, &us)) {
      const Ref nw_ref = direct(norm_w, k);
      const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
      const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
      const char *fn =
          (n >= 2048U) ? "omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r4_norm_t1"
                       : "omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r2_norm_t1";
      auto &d =
          e.emit("row_exact", fn, {.width = U64(tgs) * 64U}, {.width = 64U},
                 {hidden_in, nw_ref, rms_sq_buf, gc, gs, uc, us, y});
      e.dynamic(d, 8, fused_norm_params(n, 0U, 0U, 0U, k, 0U, sq_slot));
      e.record_weight(gate_w, n, k, "pretiled_qmv_t1",
                      "pretiled_mxfp4_qmv_swiglu_norm_t1_v4", {},
                      "legacy_route_v1", {}, false);
      e.record_weight(up_w, n, k, "pretiled_qmv_t1",
                      "pretiled_mxfp4_qmv_swiglu_norm_t1_v4", {},
                      "legacy_route_v1", {}, false);
      return true;
    }
    Ref agc{}, agh{}, agsb{}, auc{}, auh{}, ausb{};
    bool g_v2 = false, u_v2 = false;
    if (resolve_pretiled_affine_t1(gate_w, 1U, &agc, &agh, &agsb, &g_v2) &&
        resolve_pretiled_affine_t1(up_w, 1U, &auc, &auh, &ausb, &u_v2) &&
        g_v2 == u_v2 && gate_w.has_fwht64() == up_w.has_fwht64()) {
      const Ref nw_ref = direct(norm_w, k);
      const U32 g_bits = affine_nax_bits(gate_w.format);
      const U32 u_bits = affine_nax_bits(up_w.format);
      const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
      const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
      const U32 packed_bits = g_bits | (u_bits << 8U);
      const char *fn = "omlx_m5_pretiled_qmv_affine_swiglu_norm_t1";
      if (g_bits == u_bits && n >= 2048U) {
        if (g_bits == 4U)
          fn = "omlx_m5_pretiled_qmv_affine_swiglu_q4_r4_norm_t1";
        else if (g_bits == 5U)
          fn = "omlx_m5_pretiled_qmv_affine_swiglu_q5_r4_norm_t1";
        else if (g_bits == 6U)
          fn = "omlx_m5_pretiled_qmv_affine_swiglu_q6_r4_norm_t1";
        else if (g_bits == 8U)
          fn = "omlx_m5_pretiled_qmv_affine_swiglu_q8_r4_norm_t1";
      }
      auto &d = e.emit(
          "row_exact", fn, {.width = U64(tgs) * 64U}, {.width = 64U},
          {hidden_in, nw_ref, rms_sq_buf, agc, agh, agsb, auc, auh, ausb, y});
      e.dynamic(
          d, 10,
          fused_norm_params(n, 0U, 0U, 0U,
                            pretiled_k_flags(k, g_v2, gate_w.has_fwht64()),
                            packed_bits, sq_slot));
      e.record_weight(gate_w, n, k, "pretiled_qmv_t1",
                      "pretiled_affine_qmv_swiglu_norm_t1_v4", {},
                      "legacy_route_v1", {}, false);
      e.record_weight(up_w, n, k, "pretiled_qmv_t1",
                      "pretiled_affine_qmv_swiglu_norm_t1_v4", {},
                      "legacy_route_v1", {}, false);
      return true;
    }
    return false;
  }

  bool try_linear_swiglu(Encoder &e, const MaterializedWeightView &gate_w,
                         const MaterializedWeightView &up_w, Ref x, Ref y,
                         U32 n, U32 k, std::optional<U32> explicit_t = {}) {
    if (is_overridden(gate_w) || is_overridden(up_w) ||
        has_bypass_lora(gate_w) || has_bypass_lora(up_w) ||
        gate_w.padded_k != up_w.padded_k) {
      return false;
    }
    const U32 t = explicit_t.value_or(e.profile.capacity);
    const U32 sel_t = nax_effective_select_t(e, explicit_t);
    if (e.profile.capacity == 1U && gate_w.row_count == n &&
        gate_w.in_features == k && up_w.row_count == n &&
        up_w.in_features == k) {
      Ref gc{}, gs{}, uc{}, us{};
      if (resolve_pretiled_mxfp4_t1(gate_w, t, &gc, &gs) &&
          resolve_pretiled_mxfp4_t1(up_w, t, &uc, &us)) {
        const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
        const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
        const char *fn =
            (n >= 65536U)  ? "omlx_m5_pretiled_qmv_mxfp4_swiglu_t1"
            : (n >= 2048U) ? "omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r4_t1"
                           : "omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r2_t1";
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U}, {x, gc, gs, uc, us, y});
        e.dynamic(d, 6, words({n, k, U32(gate_w.padded_k), t}),
                  explicit_t.has_value() ? std::optional<U32>{}
                                         : std::optional<U32>{3U});
        e.record_weight(gate_w, n, k, "pretiled_qmv_t1",
                        "pretiled_mxfp4_qmv_swiglu_t1_v3", {},
                        "legacy_route_v1", {}, false, explicit_t);
        e.record_weight(up_w, n, k, "pretiled_qmv_t1",
                        "pretiled_mxfp4_qmv_swiglu_t1_v3", {},
                        "legacy_route_v1", {}, false, explicit_t);
        return true;
      }
      Ref agc{}, agh{}, agsb{}, auc{}, auh{}, ausb{};
      bool g_v2 = false, u_v2 = false;
      if (resolve_pretiled_affine_t1(gate_w, t, &agc, &agh, &agsb, &g_v2) &&
          resolve_pretiled_affine_t1(up_w, t, &auc, &auh, &ausb, &u_v2) &&
          g_v2 == u_v2 && gate_w.has_fwht64() == up_w.has_fwht64()) {
        const U32 g_bits = affine_nax_bits(gate_w.format);
        const U32 u_bits = affine_nax_bits(up_w.format);
        const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
        const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
        const U32 packed_bits = g_bits | (u_bits << 8U);
        const char *fn = "omlx_m5_pretiled_qmv_affine_swiglu_t1";
        if (g_bits == u_bits && n < 65536U) {
          if (g_bits == 4U) {
            fn = (n >= 2048U) ? "omlx_m5_pretiled_qmv_affine_swiglu_q4_r4_t1"
                              : "omlx_m5_pretiled_qmv_affine_swiglu_q4_r2_t1";
          } else if (g_bits == 5U) {
            fn = (n >= 2048U) ? "omlx_m5_pretiled_qmv_affine_swiglu_q5_r4_t1"
                              : "omlx_m5_pretiled_qmv_affine_swiglu_q5_r2_t1";
          } else if (g_bits == 6U) {
            fn = (n >= 2048U) ? "omlx_m5_pretiled_qmv_affine_swiglu_q6_r4_t1"
                              : "omlx_m5_pretiled_qmv_affine_swiglu_q6_r2_t1";
          } else if (g_bits == 8U) {
            fn = (n >= 2048U) ? "omlx_m5_pretiled_qmv_affine_swiglu_q8_r4_t1"
                              : "omlx_m5_pretiled_qmv_affine_swiglu_q8_r2_t1";
          }
        }
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U}, {x, agc, agh, agsb, auc, auh, ausb, y});
        e.dynamic(d, 8,
                  words({n, t, 0U,
                         pretiled_k_flags(k, g_v2, gate_w.has_fwht64()),
                         packed_bits}),
                  explicit_t.has_value() ? std::optional<U32>{}
                                         : std::optional<U32>{1U});
        e.record_weight(gate_w, n, k, "pretiled_qmv_t1",
                        "pretiled_affine_qmv_swiglu_t1_v3", {},
                        "legacy_route_v1", {}, false, explicit_t);
        e.record_weight(up_w, n, k, "pretiled_qmv_t1",
                        "pretiled_affine_qmv_swiglu_t1_v3", {},
                        "legacy_route_v1", {}, false, explicit_t);
        return true;
      }
    }
    if (explicit_t.has_value() && t > e.profile.capacity) {
      return false;
    }
    if (use_affine_nax_for_weight(gate_w, sel_t) &&
        use_affine_nax_for_weight(up_w, sel_t)) {
      const auto *gw = affine_nax_cache->find(gate_w.object_id);
      const auto *uw = affine_nax_cache->find(up_w.object_id);
      if (gw && uw && gw->bn == 16u && uw->bn == 16u && gw->n == n &&
          gw->k == k && uw->n == n && uw->k == k && gw->split_k == uw->split_k) {
        const std::string tg = affine_nax_mode_tag(gw->bits);
        const std::string tu = affine_nax_mode_tag(uw->bits);
        const bool mixed_q4_q56 = (tg == "q4" && tu == "q56") ||
                                  (tg == "q56" && tu == "q4");
        if (gw->bits == uw->bits || mixed_q4_q56) {
          linear_affine_nax_swiglu(e, *gw, *uw, x, affine_nax_xs, y,
                                   silicon_tuning.gpu_cores, explicit_t);
          const std::string grp = affine_nax_group(
              e, *gw, explicit_t, silicon_tuning.gpu_cores, uw->group_size);
          e.record_weight(gate_w, n, k, "affine_nax_m5_v1", "affine_nax_m5_v1",
                          {}, "linear_row_exact_portable_v1", grp, true,
                          explicit_t);
          e.record_weight(up_w, n, k, "affine_nax_m5_v1", "affine_nax_m5_v1",
                          {}, "linear_row_exact_portable_v1", grp, true,
                          explicit_t);
          return true;
        }
      }
    }
    if (gate_w.format != up_w.format) {
      return false;
    }
    if (gate_w.format == "direct_bf16_v1") {
      if (bf16_nax_enabled && bf16_nax_weight_eligible(gate_w, n, k) &&
          bf16_nax_weight_eligible(up_w, n, k)) {
        linear_bf16_nax_swiglu(e, gate_w, up_w, x, y, n, k,
                               silicon_tuning.gpu_cores, explicit_t);
        return true;
      }
      return false;
    }
    if (use_mxfp4_w4a4_for(sel_t)) {
      const auto *gw = mxfp4_nax_cache->find(gate_w.object_id);
      const auto *uw = mxfp4_nax_cache->find(up_w.object_id);
      if (gw && uw && gw->bn == 16u && uw->bn == 16u && gw->n == n &&
          gw->k == k && uw->n == n && uw->k == k) {
        linear_mxfp4_w4a4_nax_swiglu(e, *gw, *uw, x, y, a4_codes, a4_scales,
                                     silicon_tuning.gpu_cores, explicit_t);
        const std::string grp =
            mxfp4_w4a4_group(e, *gw, explicit_t, silicon_tuning.gpu_cores);
        e.record_weight(gate_w, n, k, "mxfp4_w4a4_nax_m5_v1",
                        "mxfp4_w4a4_nax_m5_v1", {}, "legacy_route_v1", grp,
                        false, explicit_t);
        e.record_weight(up_w, n, k, "mxfp4_w4a4_nax_m5_v1",
                        "mxfp4_w4a4_nax_m5_v1", {}, "legacy_route_v1", grp,
                        false, explicit_t);
        return true;
      }
    }
    if (use_mxfp4_nax_for_weight(gate_w, sel_t) &&
        use_mxfp4_nax_for_weight(up_w, sel_t)) {
      const auto *gw = mxfp4_nax_cache->find(gate_w.object_id);
      const auto *uw = mxfp4_nax_cache->find(up_w.object_id);
      if (gw && uw && gw->bn == 16u && uw->bn == 16u && gw->n == n &&
          gw->k == k && uw->n == n && uw->k == k &&
          gw->split_k == uw->split_k) {
        linear_mxfp4_nax_swiglu(e, *gw, *uw, x, f16_scratch, f16_row_scale, y,
                                silicon_tuning.gpu_cores, explicit_t);
        const std::string grp =
            mxfp4_nax_group(e, *gw, explicit_t, silicon_tuning.gpu_cores, true);
        e.record_weight(gate_w, n, k, "mxfp4_nax_m5_v1", "mxfp4_nax_m5_v1", {},
                        "linear_row_exact_portable_v1", grp, true, explicit_t);
        e.record_weight(up_w, n, k, "mxfp4_nax_m5_v1", "mxfp4_nax_m5_v1", {},
                        "linear_row_exact_portable_v1", grp, true, explicit_t);
        return true;
      }
    }
    if (explicit_t.has_value() &&
        (t != e.profile.capacity ||
         (!e.delegated_prefill && speculative_enabled &&
          e.profile.capacity >= 64U))) {
      return false;
    }
    if (linear_policy != LinearNumericPolicy::RowExactPortableV1) {
      return false;
    }
    auto d_gate = resolve_linear_route(
        gate_w, n, k, t, context.capabilities().device_name, m4_short_routes,
        diagnostic_override, linear_policy,
        context.capabilities().profile.probe.apple_gpu_family);
    auto d_up = resolve_linear_route(
        up_w, n, k, t, context.capabilities().device_name, m4_short_routes,
        diagnostic_override, linear_policy,
        context.capabilities().profile.probe.apple_gpu_family);
    if (!d_gate.route.row_exact || !d_up.route.row_exact ||
        d_gate.route.function != d_up.route.function ||
        d_gate.route.bits == 3U ||
        gate_w.layout == "mx_nax_tiled_bk128_v1" ||
        is_oq_nax_pretiled_layout(gate_w.layout)) {
      return false;
    }
    detail::forward::linear_swiglu(e, gate_w, up_w, x, y, n, k, sums, zero,
                                   d_gate.route);
    e.record_weight(gate_w, n, k, d_gate.route.family, d_gate.source,
                    d_gate.fallback_reason, d_gate.route.numeric_contract_id,
                    d_gate.route.row_exact_group, d_gate.route.row_exact);
    e.record_weight(up_w, n, k, d_up.route.family, d_up.source,
                    d_up.fallback_reason, d_up.route.numeric_contract_id,
                    d_up.route.row_exact_group, d_up.route.row_exact);
    return true;
  }

  struct Block {
    LayerType type;
    U32 layer_index = 0;
    U32 concat_qkv_id = 0;
    U32 concat_gdn_qkvz_id = 0;
    U32 concat_gdn_ab_id = 0;
    std::map<std::string, MaterializedWeightView> operands;
    Ref state0, state1, history, replay_qkv, replay_keys, replay_decays,
        replay_deltas;
    const MaterializedWeightView &get(const char *role) const {
      return operands.at(role);
    }
  };

  bool block_input_nax(const Block &block, U32 t,
                       std::optional<U32> explicit_t = {}) const {
    if (block.type == LayerType::full_attention) {
      if (block_has_any_bypass(block,
                               {"attention_q", "attention_k", "attention_v"})) {
        return false;
      }
    } else if (block_has_any_bypass(block,
                                    {"gdn_qkv", "gdn_z", "gdn_a", "gdn_b"})) {
      return false;
    }
    auto has = [&](const char *role, U32 n, U32 k) {
      const auto &w = block.get(role);
      if (!use_mxfp4_nax_for_weight(w, t))
        return false;
      const auto *tw = mxfp4_nax_cache->find(w.object_id);
      return tw && tw->n == n && tw->k == k &&
             mxfp4_nax_select_wide(*tw, t, explicit_t,
                                   silicon_tuning.gpu_cores, false) ==
                 NaxWidePrefillTile::none;
    };
    if (block.type == LayerType::full_attention)
      return has("attention_q", 2 * q, h) && has("attention_k", kv, h) &&
             has("attention_v", kv, h);
    return has("gdn_qkv", conv, h) && has("gdn_z", gv, h) &&
           has("gdn_a", U32(config.linear_num_value_heads), h) &&
           has("gdn_b", U32(config.linear_num_value_heads), h);
  }

  bool block_mlp_input_nax(const Block &block, U32 t,
                           std::optional<U32> explicit_t = {}) const {
    if (config.num_experts ||
        block_has_any_bypass(block, {"mlp_gate", "mlp_up", "mlp_down"}))
      return false;
    const auto &gw_v = block.get("mlp_gate");
    const auto &uw_v = block.get("mlp_up");
    const auto &dw_v = block.get("mlp_down");
    if (dw_v.has_fwht64() || !use_mxfp4_nax_for_weight(gw_v, t) ||
        !use_mxfp4_nax_for_weight(uw_v, t) ||
        !use_mxfp4_nax_for_weight(dw_v, t))
      return false;
    const auto *gw = mxfp4_nax_cache->find(gw_v.object_id);
    const auto *uw = mxfp4_nax_cache->find(uw_v.object_id);
    const auto *dw = mxfp4_nax_cache->find(dw_v.object_id);
    if (!(gw && uw && dw && gw->bn == 16u && uw->bn == 16u && gw->n == f &&
          gw->k == h && uw->n == f && uw->k == h &&
          gw->split_k == uw->split_k && dw->n == h && dw->k == f))
      return false;
    const bool gw_wide =
        mxfp4_nax_select_wide(*gw, t, explicit_t, silicon_tuning.gpu_cores,
                              true) != NaxWidePrefillTile::none;
    const bool dw_wide =
        mxfp4_nax_select_wide(*dw, t, explicit_t, silicon_tuning.gpu_cores,
                              false) != NaxWidePrefillTile::none;
    if (explicit_t.has_value() ? (gw_wide || dw_wide) : (gw_wide && dw_wide))
      return false;
    return true;
  }

  enum class ChunkAttentionRoute : std::uint8_t {
    ExactNax = 0,
    Sage2Dense = 1,
    Sage2MInference = 2,
  };

  void emit_attention_state_step(
      Encoder &e, const Block &block, const Block &state_block,
      KvPagePool &slot_pool, U32 slot_max_context, KvStorageFormat slot_kv_fmt,
      Ref q_ref, Ref k_ref, Ref v_ref, Ref mix_ref, Ref attn_pos_ref, U32 t,
      bool split, U32 context_envelope, bool dynamic_tokens,
      std::optional<U32> pos_placeholder = std::nullopt,
      std::vector<std::pair<U64, U32>> *pos_patches = nullptr,
      std::optional<ChunkAttentionRoute> attn_route_override = std::nullopt) {
    const auto &c = config;
    const bool exact_attention =
        attention_policy == omlx::m5::AttentionNumericPolicy::RowExactV1;
    const U32 pos_init = pos_placeholder.value_or(0U);
    const std::optional<U32> tok_word_post =
        dynamic_tokens ? std::optional<U32>{4U} : std::optional<U32>{};
    const std::optional<U32> pos_word_post =
        pos_placeholder.has_value() ? std::optional<U32>{}
                                    : std::optional<U32>{9U};

    Ref table_ref{&slot_pool.table_buffer(), 0, slot_pool.table_buffer().size()};
    Ref gate_ref = sub(q_ref, U64(c.head_dim) * 2);
    const bool affine4_nax_ok =
        (slot_kv_fmt == KvStorageFormat::Affine4Group128) &&
        (attn_pos_ref.buffer != nullptr) &&
        affine4_kv_runtime_eligible(context, libraries,
                                    U32(c.num_attention_heads),
                                    U32(c.num_key_value_heads),
                                    U32(c.head_dim));
    const bool fuse_qk_kv_append_t1 =
        (t == 1U) && dynamic_tokens && !pos_placeholder.has_value() &&
        exact_attention && (slot_kv_fmt == KvStorageFormat::BFloat16) &&
        (attn_pos_ref.buffer != nullptr) && !inspect_tensors_workspace &&
        ((c.head_dim % 2U) == 0U);
    const bool fuse_qk_kv_append_affine4_t1 =
        affine4_nax_ok && (t == 1U) && dynamic_tokens &&
        !pos_placeholder.has_value() && exact_attention &&
        !inspect_tensors_workspace &&
        (c.head_dim == 256 || c.head_dim == 128);
    bool affine4_q_prepacked = false;
    if (fuse_qk_kv_append_t1) {
      auto &post_kv = e.emit(
          "attention_exact", "omlx_m5_attention_qk_norm_rope_kv_append_exact",
          {.width = 32,
           .height =
               U64(t) * (c.num_attention_heads + c.num_key_value_heads)},
          {.width = 32},
          {q_ref, k_ref, v_ref, direct(block.get("q_norm"), c.head_dim),
           direct(block.get("k_norm"), c.head_dim), state_block.state0,
           state_block.state1, table_ref, attn_pos_ref});
      const U64 off_post = e.dynamic(
          post_kv, 9,
          words({U32(c.num_attention_heads), U32(c.num_key_value_heads),
                 U32(c.head_dim),
                 U32(std::llround(c.head_dim * c.partial_rotary_factor)), t,
                 q * 2, U32(c.head_dim * 2), kv, U32(c.head_dim), pos_init,
                 word(float(c.rms_norm_eps)), word(float(c.rope_theta)),
                 word(normalization_weight_offset)}),
          tok_word_post, pos_word_post);
      if (pos_patches)
        pos_patches->emplace_back(off_post, 9U);
    } else if (fuse_qk_kv_append_affine4_t1) {
      const U32 splits_t1 = split ? ((context_envelope + 511U) / 512U) : 0U;
      const auto lo_t1 =
          (split && c.head_dim == 256)
              ? affine4_split_scratch_layout(
                    1U, U32(c.num_attention_heads),
                    U32(c.num_key_value_heads), splits_t1, split_output.bytes)
              : Affine4SplitScratchLayout{};
      auto post_words = words(
          {U32(c.num_attention_heads), U32(c.num_key_value_heads),
           U32(c.head_dim),
           U32(std::llround(c.head_dim * c.partial_rotary_factor)), t, q * 2,
           U32(c.head_dim * 2), kv, U32(c.head_dim), pos_init,
           word(float(c.rms_norm_eps)), word(float(c.rope_theta)),
           word(normalization_weight_offset)});
      if (split && c.head_dim == 256 && lo_t1.fits) {
        const Ref q_half_ref{split_output.buffer,
                             split_output.offset + lo_t1.q_half_offset,
                             lo_t1.q_half_bytes};
        const Ref q_sum_ref{split_output.buffer,
                            split_output.offset + lo_t1.q_sum_offset,
                            lo_t1.q_sum_bytes};
        auto &post_kv = e.emit(
            "attention_affine4",
            "omlx_m5_attention_qk_norm_rope_kv_append_pack_q_affine4_g128",
            {.width = 32,
             .height =
                 U64(t) * (c.num_attention_heads + c.num_key_value_heads)},
            {.width = 32},
            {q_ref, k_ref, v_ref, direct(block.get("q_norm"), c.head_dim),
             direct(block.get("k_norm"), c.head_dim), state_block.state0,
             state_block.state1, table_ref, attn_pos_ref, q_half_ref,
             q_sum_ref});
        const U64 off_post = e.dynamic(post_kv, 11, std::move(post_words),
                                       tok_word_post, pos_word_post);
        if (pos_patches)
          pos_patches->emplace_back(off_post, 9U);
        affine4_q_prepacked = true;
      } else {
        auto &post_kv = e.emit(
            "attention_affine4",
            "omlx_m5_attention_qk_norm_rope_kv_append_affine4_g128",
            {.width = 32,
             .height =
                 U64(t) * (c.num_attention_heads + c.num_key_value_heads)},
            {.width = 32},
            {q_ref, k_ref, v_ref, direct(block.get("q_norm"), c.head_dim),
             direct(block.get("k_norm"), c.head_dim), state_block.state0,
             state_block.state1, table_ref, attn_pos_ref});
        const U64 off_post = e.dynamic(post_kv, 9, std::move(post_words),
                                       tok_word_post, pos_word_post);
        if (pos_patches)
          pos_patches->emplace_back(off_post, 9U);
      }
    } else {
      auto &post =
          exact_attention
              ? e.emit("attention_exact", "omlx_m5_attention_qk_norm_rope_exact",
                       {.width = 32,
                        .height = U64(t) *
                                  (c.num_attention_heads + c.num_key_value_heads)},
                       {.width = 32},
                       {q_ref, k_ref, direct(block.get("q_norm"), c.head_dim),
                        direct(block.get("k_norm"), c.head_dim), attn_pos_ref})
              : e.emit("attention_postprocess", "omlx_m5_attention_qk_norm_rope",
                       {.width = 32,
                        .height = U64(t) *
                                  (c.num_attention_heads + c.num_key_value_heads)},
                       {.width = 32},
                       {q_ref, k_ref, direct(block.get("q_norm"), c.head_dim),
                        direct(block.get("k_norm"), c.head_dim)});
      const U64 off_post = e.dynamic(
          post, exact_attention ? 5 : 4,
          words({U32(c.num_attention_heads), U32(c.num_key_value_heads),
                 U32(c.head_dim),
                 U32(std::llround(c.head_dim * c.partial_rotary_factor)), t,
                 q * 2, U32(c.head_dim * 2), kv, U32(c.head_dim), pos_init,
                 word(float(c.rms_norm_eps)), word(float(c.rope_theta)),
                 word(normalization_weight_offset)}),
          tok_word_post, pos_word_post);
      if (pos_patches)
        pos_patches->emplace_back(off_post, 9U);
    }

    auto params = words({U32(c.num_attention_heads),
                         U32(c.num_key_value_heads),
                         U32(c.head_dim),
                         t,
                         pos_init,
                         slot_max_context,
                         64,
                         slot_pool.physical_pages(),
                         q * 2,
                         U32(c.head_dim * 2),
                         kv,
                         kv,
                         q * 2,
                         U32(c.head_dim * 2)});
    const std::optional<U32> tok_word_attn =
        dynamic_tokens ? std::optional<U32>{3U} : std::optional<U32>{};
    const std::optional<U32> pos_word_attn =
        pos_placeholder.has_value() ? std::optional<U32>{}
                                    : std::optional<U32>{4U};

    const ChunkAttentionRoute default_attn_route =
        (sage2_enabled_ && t >= kFlashNaxPrefillMinTokens && c.head_dim == 256 &&
         attn_pos_ref.buffer != nullptr && sage2_kv_ws_.buffer != nullptr)
            ? ((minference_enabled_ &&
                context_envelope >= kMInferenceMinContextTokens)
                   ? ChunkAttentionRoute::Sage2MInference
                   : ChunkAttentionRoute::Sage2Dense)
            : ChunkAttentionRoute::ExactNax;
    const ChunkAttentionRoute eff_attn_route =
        attn_route_override.value_or(default_attn_route);

    if (slot_kv_fmt == KvStorageFormat::Affine4Group128) {
      if (affine4_nax_ok) {
        if (!fuse_qk_kv_append_affine4_t1) {
          emit_paged_kv_append_affine4_g128(
              e, c, t, k_ref, v_ref, state_block.state0, state_block.state1,
              table_ref, params, tok_word_attn, pos_word_attn, pos_patches);
        }
        if (!split && !pos_placeholder.has_value() && dynamic_tokens &&
            eff_attn_route != ChunkAttentionRoute::ExactNax &&
            sage2_enabled_ && sage2_kv_ws_.buffer != nullptr &&
            t >= kFlashNaxPrefillMinTokens && c.head_dim == 256 &&
            attn_pos_ref.buffer != nullptr) {
          const bool use_minf =
              (eff_attn_route == ChunkAttentionRoute::Sage2MInference) &&
              minference_enabled_ &&
              context_envelope >= kMInferenceMinContextTokens &&
              minference_scratch.buffer != nullptr;
          emit_paged_flash_prefill_sage2(
              e, c, t, context_envelope, /*is_affine4=*/true, use_minf, q_ref,
              gate_ref, state_block.state0, state_block.state1, mix_ref,
              table_ref, attn_pos_ref, params, sage2_kv_ws_,
              sage2_kv_ws_layout_,
              use_minf ? std::optional<Ref>{minference_scratch} : std::nullopt);
          e.profile.used_sage2 = true;
          if (use_minf) {
            e.profile.used_minference = true;
            e.profile.minf_context_envelope = context_envelope;
          }
        } else if (!split && (!exact_attention || c.hidden_size >= 1024) &&
                   !pos_placeholder.has_value() && dynamic_tokens &&
                   flash_nax_prefill_ && t >= kFlashNaxPrefillMinTokens &&
                   c.head_dim == 256) {
          const bool short_tail_split =
              split_attention_enabled && split_max.buffer != nullptr &&
              split_sum.buffer != nullptr && split_output.buffer != nullptr &&
              gqa_flash_prefill_tail_needs_split_kv(
                  t, U32(c.num_attention_heads), U32(c.num_key_value_heads),
                  context_envelope, silicon_tuning.gpu_cores) &&
              affine4_short_tail_split_tokens(
                  t, U32(c.num_attention_heads), U32(c.num_key_value_heads),
                  context_envelope, split_max.bytes, split_sum.bytes,
                  split_output.bytes) > 0U;
          if (short_tail_split) {
            emit_paged_attention_affine4_g128(
                e, c, t, /*split=*/true, context_envelope, slot_max_context,
                slot_pool.physical_pages(), q_ref, gate_ref, state_block.state0,
                state_block.state1, mix_ref, table_ref, attn_pos_ref, split_max,
                split_sum, split_output, std::move(params), tok_word_attn,
                pos_word_attn, pos_patches, affine4_q_prepacked);
          } else if (flash_nax_prefill_runtime_eligible(context, libraries,
                                                        U32(c.head_dim))) {
            emit_paged_flash_prefill_affine4_g128(
                e, c, t, q_ref, gate_ref, state_block.state0,
                state_block.state1, mix_ref, table_ref, attn_pos_ref,
                std::move(params));
          } else {
            emit_paged_attention_affine4_g128(
                e, c, t, /*split=*/false, context_envelope, slot_max_context,
                slot_pool.physical_pages(), q_ref, gate_ref, state_block.state0,
                state_block.state1, mix_ref, table_ref, attn_pos_ref, split_max,
                split_sum, split_output, std::move(params), tok_word_attn,
                pos_word_attn, pos_patches, affine4_q_prepacked);
          }
        } else {
          emit_paged_attention_affine4_g128(
              e, c, t, split, context_envelope, slot_max_context,
              slot_pool.physical_pages(), q_ref, gate_ref, state_block.state0,
              state_block.state1, mix_ref, table_ref, attn_pos_ref, split_max,
              split_sum, split_output, std::move(params), tok_word_attn,
              pos_word_attn, pos_patches, affine4_q_prepacked);
        }
      } else {
        auto &app = e.emit(
            "qwen_forward", "omlx_m5_forward_paged_kv_append_affine4",
            {.width = 32U, .height = U64(t) * c.num_key_value_heads},
            {.width = 32U},
            {k_ref, v_ref, state_block.state0, state_block.state1, table_ref});
        const U64 off_app =
            e.dynamic(app, 5, params, tok_word_attn, pos_word_attn);
        auto &att = e.emit(
            "qwen_forward",
            "omlx_m5_forward_paged_causal_attention_affine4_exact",
            {.width = 32U, .height = U64(t) * c.num_attention_heads},
            {.width = 32U},
            {q_ref, gate_ref, state_block.state0, state_block.state1, mix_ref,
             table_ref});
        const U64 off_att =
            e.dynamic(att, 6, std::move(params), tok_word_attn, pos_word_attn);
        e.profile.attention_kernel =
            "omlx_m5_forward_paged_causal_attention_affine4_exact";
        e.profile.attention_grid = att.grid;
        e.profile.attention_numeric_contract_id =
            "paged.simd32.affine4_g128.ordered_softmax.v1";
        e.profile.attention_row_exact_eligible = true;
        if (pos_patches) {
          pos_patches->emplace_back(off_app, 4U);
          pos_patches->emplace_back(off_att, 4U);
        }
      }
      return;
    }

    if (!fuse_qk_kv_append_t1) {
      auto &append = e.emit("attention_postprocess", "omlx_m5_paged_kv_append",
                            {.width = U64(t) * kv}, {.width = 128},
                            {k_ref, v_ref, state_block.state0,
                             state_block.state1, table_ref});
      const U64 off_app =
          e.dynamic(append, 5, params, tok_word_attn, pos_word_attn);
      if (pos_patches)
        pos_patches->emplace_back(off_app, 4U);
    }

    if (!pos_placeholder.has_value() && dynamic_tokens && !split &&
        eff_attn_route != ChunkAttentionRoute::ExactNax && sage2_enabled_ &&
        sage2_kv_ws_.buffer != nullptr && t >= kFlashNaxPrefillMinTokens &&
        c.head_dim == 256 && attn_pos_ref.buffer != nullptr) {
      const bool use_minf =
          (eff_attn_route == ChunkAttentionRoute::Sage2MInference) &&
          minference_enabled_ &&
          context_envelope >= kMInferenceMinContextTokens &&
          minference_scratch.buffer != nullptr;
      emit_paged_flash_prefill_sage2(
          e, c, t, context_envelope, /*is_affine4=*/false, use_minf, q_ref,
          gate_ref, state_block.state0, state_block.state1, mix_ref, table_ref,
          attn_pos_ref, params, sage2_kv_ws_, sage2_kv_ws_layout_,
          use_minf ? std::optional<Ref>{minference_scratch} : std::nullopt);
      e.profile.used_sage2 = true;
      if (use_minf) {
        e.profile.used_minference = true;
        e.profile.minf_context_envelope = context_envelope;
      }
      return;
    }
    if (!pos_placeholder.has_value() && dynamic_tokens && !split &&
        (!exact_attention || c.hidden_size >= 1024) && flash_nax_prefill_ &&
        t >= kFlashNaxPrefillMinTokens && attn_pos_ref.buffer != nullptr &&
        flash_nax_prefill_runtime_eligible(context, libraries,
                                           U32(c.head_dim))) {
      if (c.head_dim == 256 && split_attention_enabled && exact_attention &&
          !strict_two_pass_attn_ && split_max.buffer != nullptr &&
          split_sum.buffer != nullptr && split_output.buffer != nullptr &&
          gqa_flash_prefill_tail_needs_split_kv(
              t, U32(c.num_attention_heads), U32(c.num_key_value_heads),
              context_envelope, silicon_tuning.gpu_cores)) {
        U32 tail_chunk = 0U;
        U32 tail_splits = 0U;
        for (U32 cand_chunk : {512U, 1024U, 2048U, 4096U, 8192U}) {
          const U32 cand_splits =
              (context_envelope + cand_chunk - 1U) / cand_chunk;
          if (cand_splits >= 2U &&
              U64(t) * c.num_attention_heads * cand_splits * 4ULL <=
                  split_max.bytes &&
              U64(t) * c.num_attention_heads * cand_splits * 4ULL <=
                  split_sum.bytes &&
              U64(t) * c.num_attention_heads * cand_splits * c.head_dim *
                      4ULL <=
                  split_output.bytes) {
            tail_chunk = cand_chunk;
            tail_splits = cand_splits;
            break;
          }
        }
        if (tail_splits >= 2U) {
          e.profile.split_attention = true;
          e.profile.split_count = tail_splits;
          e.profile.split_workspace_bytes =
              U64(t) * c.num_attention_heads * tail_splits *
              (8ULL + U64(c.head_dim) * 4ULL);
          auto split_params = words(
              {U32(c.num_attention_heads), U32(c.num_key_value_heads),
               U32(c.head_dim), t, pos_init, slot_max_context, 64,
               slot_pool.physical_pages(), q * 2, U32(c.head_dim * 2), kv, kv,
               q * 2, U32(c.head_dim * 2), tail_splits, tail_chunk});
          emit_paged_split_attention_exact_1pass_gqa(
              e, context, libraries, c, t, tail_splits, q_ref, gate_ref,
              state_block.state0, state_block.state1, mix_ref, table_ref,
              split_max, split_sum, split_output, attn_pos_ref,
              std::move(split_params), tok_word_attn, pos_word_attn);
          return;
        }
      }
      emit_paged_flash_prefill_nax(e, c, t, q_ref, gate_ref,
                                   state_block.state0, state_block.state1,
                                   mix_ref, table_ref, attn_pos_ref,
                                   std::move(params));
      return;
    }
    if (split && exact_attention && attn_pos_ref.buffer != nullptr) {
      if (strict_two_pass_attn_) {
        const U32 splits = (context_envelope + 511) / 512;
        if (!splits || U64(t) * c.num_attention_heads * splits * 4 >
                           split_max.bytes)
          throw std::invalid_argument("split Attention workspace exceeded");
        e.profile.split_attention = true;
        e.profile.split_count = splits;
        e.profile.split_workspace_bytes =
            U64(t) * c.num_attention_heads * splits * 4;
        auto split_params =
            words({U32(c.num_attention_heads), U32(c.num_key_value_heads),
                   U32(c.head_dim), t, pos_init, slot_max_context, 64,
                   slot_pool.physical_pages(), q * 2, U32(c.head_dim * 2), kv,
                   kv, q * 2, U32(c.head_dim * 2), splits, 512});
        emit_paged_split_attention_exact_gqa(
            e, c, t, splits, q_ref, gate_ref, state_block.state0,
            state_block.state1, mix_ref, table_ref, split_max, attn_pos_ref,
            std::move(split_params), tok_word_attn, pos_word_attn);
      } else {
        const U32 chunk_tokens =
            exact_attention_split_chunk_tokens(slot_max_context, 0U);
        const U32 splits = exact_attention_split_count(
            context_envelope, slot_max_context, 0U);
        if (!splits ||
            U64(t) * c.num_attention_heads * splits * 4 > split_max.bytes ||
            U64(t) * c.num_attention_heads * splits * 4 > split_sum.bytes ||
            U64(t) * c.num_attention_heads * splits * c.head_dim * 4 >
                split_output.bytes)
          throw std::invalid_argument("split Attention workspace exceeded");
        e.profile.split_attention = true;
        e.profile.split_count = splits;
        e.profile.split_workspace_bytes =
            U64(t) * c.num_attention_heads * splits * (8 + c.head_dim * 4);
        auto split_params =
            words({U32(c.num_attention_heads), U32(c.num_key_value_heads),
                   U32(c.head_dim), t, pos_init, slot_max_context, 64,
                   slot_pool.physical_pages(), q * 2, U32(c.head_dim * 2), kv,
                   kv, q * 2, U32(c.head_dim * 2), splits, chunk_tokens});
        emit_paged_split_attention_exact_1pass_gqa(
            e, context, libraries, c, t, splits, q_ref, gate_ref,
            state_block.state0, state_block.state1, mix_ref, table_ref,
            split_max, split_sum, split_output, attn_pos_ref,
            std::move(split_params), tok_word_attn, pos_word_attn);
      }
      return;
    }
    if (!pos_placeholder.has_value() && dynamic_tokens && !split &&
        exact_attention) {
      emit_paged_causal_attention_exact_gqa(
          e, context, libraries, c, t, strict_two_pass_attn_, q_ref, gate_ref,
          state_block.state0, state_block.state1, mix_ref, table_ref,
          attn_pos_ref, std::move(params));
      return;
    }

    if (split) {
      const U32 chunk_tokens =
          exact_attention
              ? 512U
              : exact_attention_split_chunk_tokens(
                    slot_max_context, split_kv_partition_cap_override_);
      const U32 splits =
          exact_attention
              ? (context_envelope + 511U) / 512U
              : exact_attention_split_count(
                    context_envelope, slot_max_context,
                    split_kv_partition_cap_override_);
      if (!splits ||
          (exact_attention
               ? U64(t) * c.num_attention_heads * splits * 4 > split_max.bytes
               : (U64(t) * c.num_attention_heads * splits * 4 >
                      split_max.bytes ||
                  U64(t) * c.num_attention_heads * splits * 4 >
                      split_sum.bytes ||
                  U64(t) * c.num_attention_heads * splits * c.head_dim * 4 >
                      split_output.bytes)))
        throw std::invalid_argument("split Attention workspace exceeded");
      e.profile.split_attention = true;
      e.profile.split_count = splits;
      e.profile.split_workspace_bytes =
          U64(t) * c.num_attention_heads * splits *
          (exact_attention ? 4 : (8 + c.head_dim * 4));
      auto split_params = words({U32(c.num_attention_heads),
                                 U32(c.num_key_value_heads),
                                 U32(c.head_dim),
                                 t,
                                 pos_init,
                                 slot_max_context,
                                 64,
                                 slot_pool.physical_pages(),
                                 q * 2,
                                 U32(c.head_dim * 2),
                                 kv,
                                 kv,
                                 q * 2,
                                 U32(c.head_dim * 2),
                                 splits,
                                 chunk_tokens});
      auto &partial = e.emit(
          "attention_split_kv", "omlx_m5_paged_attention_split_partial",
          {.width = U64(splits) * 32, .height = U64(t) * c.num_attention_heads},
          {.width = 32},
          {q_ref, state_block.state0, state_block.state1, split_max, split_sum,
           split_output, table_ref});
      e.profile.attention_kernel = "omlx_m5_paged_attention_split_partial";
      e.profile.attention_grid = partial.grid;
      const U64 off_p =
          e.dynamic(partial, 7, split_params, tok_word_attn, pos_word_attn);
      auto &merge =
          e.emit("attention_split_kv", "omlx_m5_paged_attention_split_merge",
                 {.width = 32, .height = U64(t) * c.num_attention_heads},
                 {.width = 32},
                 {gate_ref, split_max, split_sum, split_output, mix_ref});
      e.profile.attention_merge_kernel = "omlx_m5_paged_attention_split_merge";
      e.profile.attention_merge_grid = merge.grid;
      const U64 off_m = e.dynamic(merge, 5, std::move(split_params),
                                  tok_word_attn, pos_word_attn);
      if (pos_patches) {
        pos_patches->emplace_back(off_p, 4U);
        pos_patches->emplace_back(off_m, 4U);
      }
    } else {
      const char *exact_fn = strict_two_pass_attn_
                                 ? "omlx_m5_paged_causal_attention_exact"
                                 : "omlx_m5_paged_causal_attention_exact_1pass";
      const U32 tptg_y =
          exact_attention
              ? (strict_two_pass_attn_
                     ? exact_attention_gqa_threads_y(
                           t, U32(c.num_attention_heads),
                           U32(c.num_key_value_heads), U32(c.head_dim))
                     : exact_attention_1pass_gqa_threads_y(
                           context, libraries, exact_fn, t,
                           U32(c.num_attention_heads),
                           U32(c.num_key_value_heads), U32(c.head_dim)))
              : 1U;
      auto &attn =
          exact_attention
              ? e.emit("attention_exact", exact_fn,
                       {.width = 32, .height = U64(t) * c.num_attention_heads},
                       {.width = 32, .height = tptg_y},
                       {q_ref, gate_ref, state_block.state0, state_block.state1,
                        mix_ref, table_ref, attn_pos_ref})
              : e.emit("attention_postprocess", "omlx_m5_paged_causal_attention",
                       {.width = 32, .height = U64(t) * c.num_attention_heads},
                       {.width = 32},
                       {q_ref, gate_ref, state_block.state0, state_block.state1,
                        mix_ref, table_ref});
      e.profile.attention_kernel =
          exact_attention ? exact_fn : "omlx_m5_paged_causal_attention";
      e.profile.attention_grid = attn.grid;
      const U64 off_att = e.dynamic(attn, exact_attention ? 7 : 6, params,
                                    tok_word_attn, pos_word_attn);
      if (pos_patches)
        pos_patches->emplace_back(off_att, 4U);
      if (exact_attention) {
        e.profile.attention_numeric_contract_id =
            strict_two_pass_attn_
                ? "paged.simd32.globalmax.ordered_softmax.f32_gate.v1"
                : "paged.simd32.online_sb512.ordered_softmax.f32_gate.v1";
        e.profile.attention_row_exact_eligible = true;
      }
    }
  }

  void attention(Encoder &e, const Block &block, bool split,
                 U32 context_envelope, bool input_is_f16 = false,
                 Ref custom_attn_pos = {},
                 std::optional<U32> *io_sq_slot = nullptr,
                 std::optional<U32> out_sq_slot = std::nullopt,
                 bool input_is_a8 = false,
                 std::optional<ChunkAttentionRoute> attn_route_override =
                     std::nullopt,
                 bool input_is_a4 = false) {
    const U32 t = e.profile.capacity;
    Ref q_ref = query, k_ref = key, v_ref = value;
    const std::array<MaterializedWeightView, 3> parts{
        block.get("attention_q"), block.get("attention_k"),
        block.get("attention_v")};
    const std::array<U32, 3> part_ns{2 * q, kv, kv};
    const auto *qkv_bypass =
        find_bypass_group(block.layer_index, QwenBypassGroupKind::attn_qkv);
    Ref d0{}, s0{}, d1{}, s1{}, d2{}, s2{};
    Ref h0{}, h1{}, h2{};
    bool fused_qkv_norm = false;
    if (io_sq_slot != nullptr && io_sq_slot->has_value()) {
      const U32 sq_in = **io_sq_slot;
      *io_sq_slot = std::nullopt;
      if (qkv_bypass == nullptr && t == 1U && !input_is_f16 && 2 * q < 65536U &&
          kv < 65536U && parts[0].row_count == 2 * q &&
          parts[1].row_count == kv && parts[2].row_count == kv &&
          parts[0].in_features == h && parts[1].in_features == h &&
          parts[2].in_features == h) {
        if (resolve_pretiled_mxfp4_t1(parts[0], 1U, &d0, &s0) &&
            resolve_pretiled_mxfp4_t1(parts[1], 1U, &d1, &s1) &&
            resolve_pretiled_mxfp4_t1(parts[2], 1U, &d2, &s2)) {
          const Ref nw_ref = direct(block.get("input_norm"), h);
          const bool use_r4 = ((2 * q + 2 * kv) >= 2048U);
          const U32 rows_per_tg = use_r4 ? 8U : 4U;
          const U32 tgs = ((2 * q + rows_per_tg - 1U) / rows_per_tg) +
                          2U * ((kv + rows_per_tg - 1U) / rows_per_tg);
          const char *fn =
              use_r4 ? "omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r4_norm_t1"
                     : "omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r2_norm_t1";
          auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                           {.width = 64U},
                           {hidden, nw_ref, rms_sq_buf, d0, s0, d1, s1, d2, s2,
                            query, key, value});
          e.dynamic(d, 12, fused_norm_params(2 * q, kv, kv, 0U, h, 0U, sq_in));
          for (std::size_t i = 0; i < 3; ++i) {
            e.record_weight(parts[i], part_ns[i], h, "pretiled_qmv_t1",
                            "pretiled_mxfp4_qmv_concat3_norm_t1_v4", {},
                            "legacy_route_v1", {}, false);
          }
          fused_qkv_norm = true;
        } else if (bool lv0 = false, lv1 = false, lv2 = false;
                   resolve_pretiled_affine_t1(parts[0], 1U, &d0, &h0, &s0, &lv0) &&
                   resolve_pretiled_affine_t1(parts[1], 1U, &d1, &h1, &s1, &lv1) &&
                   resolve_pretiled_affine_t1(parts[2], 1U, &d2, &h2, &s2, &lv2) &&
                   lv0 == lv1 && lv1 == lv2) {
          const Ref nw_ref = direct(block.get("input_norm"), h);
          const U32 b0 = affine_nax_bits(parts[0].format);
          const U32 b1 = affine_nax_bits(parts[1].format);
          const U32 b2 = affine_nax_bits(parts[2].format);
          const U32 packed_bits = b0 | (b1 << 8U) | (b2 << 16U);
          const bool use_r4 = ((2 * q + 2 * kv) >= 2048U);
          const U32 rows_per_tg = use_r4 ? 8U : 4U;
          const U32 tgs = ((2 * q + rows_per_tg - 1U) / rows_per_tg) +
                          2U * ((kv + rows_per_tg - 1U) / rows_per_tg);
          const bool all_q456 = (b0 >= 4U && b0 <= 6U) &&
                                (b1 >= 4U && b1 <= 6U) &&
                                (b2 >= 4U && b2 <= 6U);
          const char *fn = use_r4
                               ? "omlx_m5_pretiled_qmv_affine_concat3_r4_norm_t1"
                               : "omlx_m5_pretiled_qmv_affine_concat3_r2_norm_t1";
          if (use_r4 && b0 == b1 && b1 == b2 &&
              (b0 == 4U || b0 == 5U || b0 == 6U)) {
            fn = (b0 == 5U)
                     ? "omlx_m5_pretiled_qmv_affine_concat3_q5_r4_norm_t1"
                 : (b0 == 4U)
                     ? "omlx_m5_pretiled_qmv_affine_concat3_q4_r4_norm_t1"
                     : "omlx_m5_pretiled_qmv_affine_concat3_q6_r4_norm_t1";
          } else if (all_q456) {
            fn = use_r4 ? "omlx_m5_pretiled_qmv_affine_concat3_q456_r4_norm_t1"
                        : "omlx_m5_pretiled_qmv_affine_concat3_q456_r2_norm_t1";
          }
          auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                           {.width = 64U},
                           {hidden, nw_ref, rms_sq_buf, d0, h0, s0, d1, h1, s1,
                            d2, h2, s2, query, key, value});
          e.dynamic(
              d, 15,
              fused_norm_params(2 * q, kv, kv, 0U,
                                pretiled_k_flags(h, lv0, false), packed_bits,
                                sq_in));
          for (std::size_t i = 0; i < 3; ++i) {
            e.record_weight(parts[i], part_ns[i], h, "pretiled_qmv_t1",
                            "pretiled_affine_qmv_concat3_norm_t1_v4", {},
                            "legacy_route_v1", {}, false);
          }
          fused_qkv_norm = true;
        }
      }
      if (!fused_qkv_norm) {
        normalize(e, block.get("input_norm"), hidden, norm);
      }
    }
    if (!fused_qkv_norm && t == 1U && !input_is_f16 &&
        parts[0].row_count == 2 * q &&
        parts[1].row_count == kv && parts[2].row_count == kv &&
        parts[0].in_features == h && parts[1].in_features == h &&
        parts[2].in_features == h &&
        resolve_pretiled_mxfp4_t1(parts[0], 1U, &d0, &s0) &&
        resolve_pretiled_mxfp4_t1(parts[1], 1U, &d1, &s1) &&
        resolve_pretiled_mxfp4_t1(parts[2], 1U, &d2, &s2)) {
      const bool use_r4 = ((2 * q + 2 * kv) >= 2048U);
      const U32 rows_per_tg = use_r4 ? 8U : 4U;
      const U32 tgs = ((2 * q + rows_per_tg - 1U) / rows_per_tg) +
                      2U * ((kv + rows_per_tg - 1U) / rows_per_tg);
      const char *fn = use_r4 ? "omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r4_t1"
                              : "omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r2_t1";
      auto &d =
          e.emit("row_exact", fn, {.width = U64(tgs) * 64U}, {.width = 64U},
                 {norm, d0, s0, d1, s1, d2, s2, query, key, value});
      e.dynamic(d, 10, words({2 * q, kv, kv, h}));
      for (std::size_t i = 0; i < 3; ++i) {
        e.record_weight(parts[i], part_ns[i], h, "pretiled_qmv_t1",
                        "pretiled_mxfp4_qmv_concat3_t1_v3", {},
                        "legacy_route_v1", {}, false);
      }
    } else if (bool lv0 = false, lv1 = false, lv2 = false;
               !fused_qkv_norm && t == 1U && !input_is_f16 &&
               parts[0].row_count == 2 * q && parts[1].row_count == kv &&
               parts[2].row_count == kv && parts[0].in_features == h &&
               parts[1].in_features == h && parts[2].in_features == h &&
               resolve_pretiled_affine_t1(parts[0], 1U, &d0, &h0, &s0, &lv0) &&
               resolve_pretiled_affine_t1(parts[1], 1U, &d1, &h1, &s1, &lv1) &&
               resolve_pretiled_affine_t1(parts[2], 1U, &d2, &h2, &s2, &lv2) &&
               lv0 == lv1 && lv1 == lv2) {
      const U32 b0 = affine_nax_bits(parts[0].format);
      const U32 b1 = affine_nax_bits(parts[1].format);
      const U32 b2 = affine_nax_bits(parts[2].format);
      const U32 packed_bits = b0 | (b1 << 8U) | (b2 << 16U);
      const bool use_r4 = ((2 * q + 2 * kv) >= 2048U);
      const U32 rows_per_tg = use_r4 ? 8U : 4U;
      const U32 tgs = ((2 * q + rows_per_tg - 1U) / rows_per_tg) +
                      2U * ((kv + rows_per_tg - 1U) / rows_per_tg);
      const bool all_q456 = (b0 >= 4U && b0 <= 6U) && (b1 >= 4U && b1 <= 6U) &&
                            (b2 >= 4U && b2 <= 6U);
      const char *fn = "omlx_m5_pretiled_qmv_affine_concat3_t1";
      if (use_r4 && b0 == b1 && b1 == b2 && (b0 == 4U || b0 == 5U || b0 == 6U)) {
        fn = (b0 == 5U)   ? "omlx_m5_pretiled_qmv_affine_concat3_q5_r4_t1"
             : (b0 == 4U) ? "omlx_m5_pretiled_qmv_affine_concat3_q4_r4_t1"
                          : "omlx_m5_pretiled_qmv_affine_concat3_q6_r4_t1";
      } else if (all_q456) {
        fn = use_r4 ? "omlx_m5_pretiled_qmv_affine_concat3_q456_r4_t1"
                    : "omlx_m5_pretiled_qmv_affine_concat3_q456_r2_t1";
      }
      auto &d = e.emit(
          "row_exact", fn, {.width = U64(tgs) * 64U}, {.width = 64U},
          {norm, d0, h0, s0, d1, h1, s1, d2, h2, s2, query, key, value});
      e.dynamic(d, 13,
                words({2 * q, kv, kv, pretiled_k_flags(h, lv0, false),
                       packed_bits}));
      for (std::size_t i = 0; i < 3; ++i) {
        e.record_weight(parts[i], part_ns[i], h, "pretiled_qmv_t1",
                        "pretiled_affine_qmv_concat3_t1_v3", {},
                        "legacy_route_v1", {}, false);
      }
    } else if (!fused_qkv_norm && input_is_a4 && !w4a4_keep_qk_a8_ &&
               try_concat_w4a4_nax(e, block.concat_qkv_id, parts, part_ns, h,
                                   norm, concat_out)) {
      emit_split3(e, concat_out, query, key, value, 2 * q, kv, kv);
    } else if (!fused_qkv_norm && input_is_a4) {
      if (w4a4_keep_qk_a8_) {
        if (!try_linear_w4a8_base(e, block.get("attention_q"), norm, query,
                                  2 * q, h) &&
            !try_linear_w4a4_base(e, block.get("attention_q"), norm, query,
                                  2 * q, h))
          linear_base(e, block.get("attention_q"), norm, query, 2 * q, h);
        if (!try_linear_w4a8_base(e, block.get("attention_k"), norm, key, kv,
                                  h) &&
            !try_linear_w4a4_base(e, block.get("attention_k"), norm, key, kv,
                                  h))
          linear_base(e, block.get("attention_k"), norm, key, kv, h);
      } else {
        if (!try_linear_w4a4_base(e, block.get("attention_q"), norm, query,
                                  2 * q, h))
          linear_base(e, block.get("attention_q"), norm, query, 2 * q, h);
        if (!try_linear_w4a4_base(e, block.get("attention_k"), norm, key, kv,
                                  h))
          linear_base(e, block.get("attention_k"), norm, key, kv, h);
      }
      if (!try_linear_w4a4_base(e, block.get("attention_v"), norm, value, kv,
                                h))
        linear_base(e, block.get("attention_v"), norm, value, kv, h);
    } else if (!fused_qkv_norm && input_is_a8 &&
               try_concat_w4a8_nax(e, block.concat_qkv_id, parts, part_ns, h,
                                   norm, concat_out)) {
      emit_split3(e, concat_out, query, key, value, 2 * q, kv, kv);
    } else if (!fused_qkv_norm && input_is_a8) {
      if (!try_linear_w4a8_base(e, block.get("attention_q"), norm, query, 2 * q,
                                h))
        linear_base(e, block.get("attention_q"), norm, query, 2 * q, h);
      if (!try_linear_w4a8_base(e, block.get("attention_k"), norm, key, kv, h))
        linear_base(e, block.get("attention_k"), norm, key, kv, h);
      if (!try_linear_w4a8_base(e, block.get("attention_v"), norm, value, kv,
                                h))
        linear_base(e, block.get("attention_v"), norm, value, kv, h);
    } else if (!fused_qkv_norm &&
               try_concat_nax(e, block.concat_qkv_id, parts, part_ns, h,
                              input_is_f16, norm, concat_out)) {
      if (t == 1U) {
        q_ref = Ref{concat_out.buffer, 0, mul(2 * q, 2)};
        k_ref = Ref{concat_out.buffer, mul(2 * q, 2), mul(kv, 2)};
        v_ref = Ref{concat_out.buffer, mul(2 * q + kv, 2), mul(kv, 2)};
      } else {
        emit_split3(e, concat_out, query, key, value, 2 * q, kv, kv);
      }
    } else if (!fused_qkv_norm && input_is_f16) {
      linear_from_f16(e, block.get("attention_q"), f16_norm, f16_ones_scale,
                      query, 2 * q, h);
      linear_from_f16(e, block.get("attention_k"), f16_norm, f16_ones_scale,
                      key, kv, h);
      linear_from_f16(e, block.get("attention_v"), f16_norm, f16_ones_scale,
                      value, kv, h);
    } else if (!fused_qkv_norm) {
      linear_base(e, block.get("attention_q"), norm, query, 2 * q, h);
      linear_base(e, block.get("attention_k"), norm, key, kv, h);
      linear_base(e, block.get("attention_v"), norm, value, kv, h);
    }
    if (qkv_bypass != nullptr) {
      const std::array<detail::lora_rt::BypassPartDest, 3> dests{{
          {q_ref, 2 * q, 0U},
          {k_ref, kv, 0U},
          {v_ref, kv, 0U},
      }};
      detail::lora_rt::emit_bypass_sibling_group(
          e, *qkv_bypass, norm, dests, lora_bypass_tmp_, bf16_nax_bypass_ok_,
          silicon_tuning.gpu_cores);
    }

    emit_attention_state_step(
        e, block, block, *page_pool, max_context, kv_format_, q_ref, k_ref,
        v_ref, mixer,
        custom_attn_pos.buffer ? custom_attn_pos : attention_positions, t,
        split, context_envelope, true, std::nullopt, nullptr,
        attn_route_override);
    if (input_is_a4 &&
        try_linear_w4a4_base(e, block.get("attention_o"), mixer, delta, h, q)) {
      emit_bypass_for_weight(e, block.get("attention_o"), mixer, delta, h);
      if (io_sq_slot != nullptr)
        *io_sq_slot = std::nullopt;
    } else if (input_is_a8 &&
               try_linear_w4a8_base(e, block.get("attention_o"), mixer, delta,
                                    h, q)) {
      emit_bypass_for_weight(e, block.get("attention_o"), mixer, delta, h);
      if (io_sq_slot != nullptr)
        *io_sq_slot = std::nullopt;
    } else if (io_sq_slot != nullptr && out_sq_slot.has_value() &&
               try_linear_add_sq_t1(e, block.get("attention_o"), mixer, hidden,
                                    h, q, *out_sq_slot)) {
      *io_sq_slot = out_sq_slot;
    } else {
      linear(e, block.get("attention_o"), mixer, delta, h, q);
      if (io_sq_slot != nullptr)
        *io_sq_slot = std::nullopt;
    }
  }

  U32 select_forward_bucket_capacity(U32 count, U32 slot_max_context) const {
    const U32 limit = std::min(slot_max_context, workspace_rows_);
    for (U32 cap : {1U, 2U, 4U, 8U, 12U, 16U, 24U, 32U, 64U, 128U, 256U, 512U,
                    1024U, 2048U, 4096U}) {
      if (cap > limit)
        break;
      if (count <= cap)
        return cap;
    }
    return limit;
  }

  void emit_gdn_state_step(Encoder &e, const Block &block,
                           const Block &state_block, Ref qkv_ref, Ref z_ref,
                           Ref a_ref, Ref b_ref, Ref conv_ref, Ref rec_ref,
                           Ref mix_ref, U32 t, bool verify, bool dynamic_tokens,
                           bool allow_f16_out, U32 slot_max_context = 0U,
                           std::optional<U32> out_sq_slot = std::nullopt,
                           bool *out_fused_add_sq = nullptr,
                           bool out_w4a8 = false,
                           bool out_w4a8_write_bf16 = false,
                           bool emit_gdn_out_linear = true,
                           bool *out_wrote_a8 = nullptr,
                           bool out_w4a4 = false,
                           bool out_w4a4_write_bf16 = false,
                           bool *out_wrote_a4 = nullptr) {
    if (out_fused_add_sq)
      *out_fused_add_sq = false;
    if (out_wrote_a8)
      *out_wrote_a8 = false;
    if (out_wrote_a4)
      *out_wrote_a4 = false;
    const auto &c = config;
    auto params =
        words({U32(c.linear_num_key_heads), U32(c.linear_num_value_heads),
               U32(c.linear_key_head_dim), U32(c.linear_value_head_dim), conv,
               U32(c.linear_conv_kernel_dim), t, word(float(c.rms_norm_eps))});
    const std::optional<U32> tok_w6 =
        dynamic_tokens ? std::optional<U32>{6U} : std::optional<U32>{};
    const std::optional<U32> tok_w0 =
        dynamic_tokens ? std::optional<U32>{0U} : std::optional<U32>{};

    const bool fused_verify_conv =
        verify && t >= 1U && t <= 8U &&
        !detail::forward::env_flag_enabled("OMLX_M5_DISABLE_GDN_SPEC_FUSION");
    if (verify && !fused_verify_conv) {
      copy(e, state_block.state1, state_block.history, state_block.state1.bytes);
      auto &save = e.emit("qwen_forward", "omlx_m5_forward_copy_rows",
                          {.width = U64(t) * conv}, {.width = 128},
                          {qkv_ref, state_block.replay_qkv});
      e.dynamic(save, 2, words({t, conv, 0}), tok_w0);
    }

    Ref conv_w =
        direct(block.get("gdn_conv"), U64(conv) * c.linear_conv_kernel_dim);
    Ref hist_ref = (verify && !fused_verify_conv) ? state_block.history
                                                  : state_block.state1;
    Ref a_log = direct(block.get("gdn_a_log"), c.linear_num_value_heads);
    Ref dt_bias = direct(block.get("gdn_dt_bias"), c.linear_num_value_heads);
    Ref gdn_norm = direct(block.get("gdn_norm"), c.linear_value_head_dim);
    const U32 kd = U32(c.linear_key_head_dim);
    const U32 vd = U32(c.linear_value_head_dim);

    const auto &out_w = block.get("gdn_out");
    const bool eff_out_w4a4 =
        out_w4a4 && !verify && can_w4a4_weight(out_w, h, gv);
    const bool eff_out_w4a8 =
        !eff_out_w4a4 && out_w4a8 && !verify && can_w4a8_weight(out_w, h, gv);
    const auto *gdn_out_nax =
        (!eff_out_w4a4 && !eff_out_w4a8 && allow_f16_out &&
         emit_gdn_out_linear && !has_bypass_lora(out_w) &&
         use_mxfp4_nax_for_weight(out_w, t))
            ? mxfp4_nax_cache->find(out_w.object_id)
            : nullptr;
    const bool out_f16 =
        (!out_w.has_fwht64() && gdn_out_nax != nullptr &&
         gdn_out_nax->n == h && gdn_out_nax->k == gv);
    const bool mega_fuse_a4 = eff_out_w4a4 && !out_w.has_fwht64();
    const bool mega_fuse_a8 = eff_out_w4a8 && !out_w.has_fwht64();
    const U32 mp = out_f16 ? mxfp4_nax_mp(t) : 0U;
    auto emit_gdn_out_bf16 = [&]() {
      if (!emit_gdn_out_linear) {
        return;
      }
      if (eff_out_w4a4 && try_linear_w4a4_base(e, out_w, mix_ref, delta, h, gv)) {
        emit_bypass_for_weight(e, out_w, mix_ref, delta, h);
        return;
      }
      if (eff_out_w4a8 && try_linear_w4a8_base(e, out_w, mix_ref, delta, h, gv)) {
        emit_bypass_for_weight(e, out_w, mix_ref, delta, h);
        return;
      }
      if (out_sq_slot.has_value() &&
          try_linear_add_sq_t1(e, out_w, mix_ref, hidden, h, gv,
                               *out_sq_slot)) {
        if (out_fused_add_sq)
          *out_fused_add_sq = true;
      } else {
        linear(e, out_w, mix_ref, delta, h, gv);
      }
    };

    if (!verify) {
      const U32 gdn_cap =
          dynamic_tokens
              ? t
              : select_forward_bucket_capacity(
                    t, slot_max_context ? slot_max_context : max_context);
      auto mega_params = words(
          {U32(c.linear_num_key_heads), U32(c.linear_num_value_heads), kd, vd,
           conv, U32(c.linear_conv_kernel_dim), t, word(float(c.rms_norm_eps)),
           mp});
      if (try_emit_gdn_mega_prefill(
              e, gdn_cap, gv, U32(c.linear_num_key_heads),
              U32(c.linear_num_value_heads), kd, vd, conv, qkv_ref, conv_w,
              hist_ref, a_ref, b_ref, a_log, dt_bias, state_block.state0, z_ref,
              gdn_norm, out_f16 ? f16_scratch : mix_ref, out_f16, gdn_chunk_ws,
              gdn_chunk_nax_enabled_, params, mega_params, dynamic_tokens,
              silicon_tuning.gpu_cores, mega_fuse_a8 ? a8_qa : Ref{},
              mega_fuse_a8 ? a8_sa : Ref{}, mega_fuse_a8 ? a8_ra : Ref{},
              mega_fuse_a8 ? w4a8_nax_mp(t) : 0U,
              mega_fuse_a8 && (out_w4a8_write_bf16 || has_bypass_lora(out_w)),
              float(c.rms_norm_eps), mega_fuse_a4 ? w4a4_codes : Ref{},
              mega_fuse_a4 ? w4a4_scales : Ref{},
              mega_fuse_a4 ? w4a4_nax_mp(t) : 0U,
              mega_fuse_a4 && (out_w4a4_write_bf16 || has_bypass_lora(out_w)))) {
        if (mega_fuse_a4) {
          if (out_wrote_a4)
            *out_wrote_a4 = true;
          e.set_a4_quant(make_a4_key(
              e, mix_ref, gv,
              dynamic_tokens ? std::nullopt : std::optional<U32>(t)));
          if (emit_gdn_out_linear) {
            if (!try_linear_w4a4_base(e, out_w, mix_ref, delta, h, gv)) {
              linear_base(e, out_w, mix_ref, delta, h, gv);
            }
            emit_bypass_for_weight(e, out_w, mix_ref, delta, h);
          }
        } else if (mega_fuse_a8) {
          if (out_wrote_a8)
            *out_wrote_a8 = true;
          e.set_a8_quant(make_a8_key(
              e, mix_ref, gv,
              dynamic_tokens ? std::nullopt : std::optional<U32>(t)));
          if (emit_gdn_out_linear) {
            if (!try_linear_w4a8_base(e, out_w, mix_ref, delta, h, gv)) {
              linear_base(e, out_w, mix_ref, delta, h, gv);
            }
            emit_bypass_for_weight(e, out_w, mix_ref, delta, h);
          }
        } else if (eff_out_w4a4 || eff_out_w4a8) {
          emit_gdn_out_bf16();
        } else if (out_f16) {
          linear_from_f16(e, out_w, f16_scratch, f16_ones_scale, delta, h, gv);
        } else if (allow_f16_out) {
          emit_gdn_out_bf16();
        }
        e.profile.gdn_row_exact_eligible = true;
        return;
      }
    }

    if (allow_f16_out && !verify && !inspect_tensors_workspace && t == 1U &&
        kd == 128U && vd == 128U && c.linear_conv_kernel_dim == 4U &&
        (conv % 4U) == 0U && c.linear_num_key_heads > 0 &&
        c.linear_num_value_heads >= c.linear_num_key_heads &&
        (c.linear_num_value_heads % c.linear_num_key_heads) == 0 &&
        gv == U32(c.linear_num_value_heads) * 128U) {
      if (out_f16) {
        auto &fused = e.emit(
            "gdn_layer", "omlx_m5_gdn_t1_conv_recurrent_norm_gate_f16",
            {.width = 256U, .height = U32(c.linear_num_value_heads)},
            {.width = 256U},
            {qkv_ref, conv_w, hist_ref, a_ref, b_ref, a_log, dt_bias,
             state_block.state0, z_ref, gdn_norm, f16_scratch});
        e.dynamic(fused, 11, params, tok_w6);
        auto &commit_h = e.emit(
            "gdn_layer", "omlx_m5_gdn_commit_history_vec4_t1",
            {.width = conv / 4U}, {.width = 256U}, {hist_ref, qkv_ref});
        e.dynamic(commit_h, 2, params, tok_w6);
        linear_from_f16(e, out_w, f16_scratch, f16_ones_scale, delta, h, gv);
      } else {
        auto &fused = e.emit(
            "gdn_layer", "omlx_m5_gdn_t1_conv_recurrent_norm_gate",
            {.width = 256U, .height = U32(c.linear_num_value_heads)},
            {.width = 256U},
            {qkv_ref, conv_w, hist_ref, a_ref, b_ref, a_log, dt_bias,
             state_block.state0, z_ref, gdn_norm, mix_ref});
        e.dynamic(fused, 11, params, tok_w6);
        auto &commit_h = e.emit(
            "gdn_layer", "omlx_m5_gdn_commit_history_vec4_t1",
            {.width = conv / 4U}, {.width = 256U}, {hist_ref, qkv_ref});
        e.dynamic(commit_h, 2, params, tok_w6);
        emit_gdn_out_bf16();
      }
      e.profile.gdn_row_exact_eligible = true;
      return;
    }

    if (fused_verify_conv) {
      auto &conv_vs = e.emit(
          "gdn_layer", "omlx_m5_gdn_conv_silu_vec4_t8_verify_save",
          {.width = (conv + 3U) / 4U, .height = (t + 7U) / 8U},
          {.width = 64U, .height = 2U},
          {qkv_ref, conv_w, state_block.state1, conv_ref,
           state_block.replay_qkv});
      e.dynamic(conv_vs, 5, params, tok_w6);
    } else {
      emit_gdn_conv_silu(e, t, conv, qkv_ref, conv_w, hist_ref, conv_ref,
                         params, dynamic_tokens);
    }

    if (!verify && !inspect_tensors_workspace && t == 1U && kd == 128U &&
        vd == 128U && c.linear_num_key_heads > 0 &&
        c.linear_num_value_heads >= c.linear_num_key_heads &&
        (c.linear_num_value_heads % c.linear_num_key_heads) == 0 &&
        gv == U32(c.linear_num_value_heads) * 128U) {
      if (out_f16) {
        auto &fused = e.emit(
            "gdn_layer", "omlx_m5_gdn_t1_recurrent_norm_gate_f16",
            {.width = 256U, .height = U32(c.linear_num_value_heads)},
            {.width = 256U},
            {conv_ref, a_ref, b_ref, a_log, dt_bias, state_block.state0, z_ref,
             gdn_norm, f16_scratch});
        e.dynamic(fused, 9, params, tok_w6);
        linear_from_f16(e, out_w, f16_scratch, f16_ones_scale, delta, h, gv);
      } else {
        auto &fused = e.emit(
            "gdn_layer", "omlx_m5_gdn_t1_recurrent_norm_gate",
            {.width = 256U, .height = U32(c.linear_num_value_heads)},
            {.width = 256U},
            {conv_ref, a_ref, b_ref, a_log, dt_bias, state_block.state0, z_ref,
             gdn_norm, mix_ref});
        e.dynamic(fused, 9, params, tok_w6);
        if (allow_f16_out) {
          emit_gdn_out_bf16();
        }
      }
      e.profile.gdn_row_exact_eligible = true;
      return;
    }

    if (verify) {
      if (dynamic_tokens) {
        emit_gdn_recurrent_verify(
            e, gv, kd, vd, conv_ref, a_ref, b_ref, a_log, dt_bias,
            state_block.state0, rec_ref, state_block.replay_keys,
            state_block.replay_decays, state_block.replay_deltas, params);
      } else {
        const bool use_4row = (kd == 128U && (vd % 4U) == 0U && (gv % 4U) == 0U);
        auto &rec = e.emit(
            "gdn_layer",
            use_4row ? "omlx_m5_gdn_recurrent_record_4row"
                     : "omlx_m5_gdn_recurrent_record",
            {.width = 32, .height = use_4row ? (gv / 4U) : gv}, {.width = 32},
            {conv_ref, a_ref, b_ref, a_log, dt_bias, state_block.state0,
             rec_ref, state_block.replay_keys, state_block.replay_decays,
             state_block.replay_deltas});
        e.dynamic(rec, 10, params);
      }
    } else {
      const U32 gdn_cap =
          dynamic_tokens
              ? t
              : select_forward_bucket_capacity(
                    t, slot_max_context ? slot_max_context : max_context);
      emit_gdn_recurrent_forward(
          e, gdn_cap, gv, kd, vd, conv_ref, a_ref, b_ref, a_log, dt_bias,
          state_block.state0, rec_ref, params,
          silicon_tuning.gdn_tb8_min_tokens, gdn_chunk_ws,
          gdn_chunk_nax_enabled_, U32(c.linear_num_key_heads),
          U32(c.linear_num_value_heads), dynamic_tokens,
          silicon_tuning.gpu_cores);
    }

    if (out_f16) {
      auto &norm_gate = e.emit(
          "gdn_layer", "omlx_m5_gdn_norm_gate_f16",
          {.width = 32, .height = U64(mp) * c.linear_num_value_heads},
          {.width = 32},
          {rec_ref, z_ref, gdn_norm, f16_scratch});
      e.dynamic(norm_gate, 4, params, tok_w6);
      linear_from_f16(e, out_w, f16_scratch, f16_ones_scale, delta, h, gv);
    } else {
      auto &norm_gate = e.emit(
          "gdn_layer", "omlx_m5_gdn_norm_gate",
          {.width = 32, .height = U64(t) * c.linear_num_value_heads},
          {.width = 32},
          {rec_ref, z_ref, gdn_norm, mix_ref});
      e.dynamic(norm_gate, 4, params, tok_w6);
      if (allow_f16_out) {
        emit_gdn_out_bf16();
      }
    }
    e.profile.gdn_row_exact_eligible = true;
  }

  void gdn(Encoder &e, const Block &block, bool verify,
           bool input_is_f16 = false, std::optional<U32> *io_sq_slot = nullptr,
           std::optional<U32> out_sq_slot = std::nullopt,
           bool input_is_a8 = false, bool input_is_a4 = false) {
    const auto &c = config;
    const U32 t = e.profile.capacity;
    const U32 hv = U32(c.linear_num_value_heads);
    Ref qkv_ref = qkv, z_ref = z, a_ref = a, b_ref = b;

    const std::array<MaterializedWeightView, 2> qkvz_parts{
        block.get("gdn_qkv"), block.get("gdn_z")};
    const std::array<U32, 2> qkvz_ns{conv, gv};
    const std::array<MaterializedWeightView, 2> ab_parts{
        block.get("gdn_a"), block.get("gdn_b")};
    const std::array<U32, 2> ab_ns{hv, hv};
    const auto *gdn_bypass =
        find_bypass_group(block.layer_index, QwenBypassGroupKind::gdn_inputs);
    Ref qz_c0{}, qz_s0{}, qz_c1{}, qz_s1{}, qz_h0{}, qz_h1{};
    Ref ab_c0{}, ab_s0{}, ab_c1{}, ab_s1{}, ab_h0{}, ab_h1{};
    bool fused_qkvzab = false;
    if (io_sq_slot != nullptr && io_sq_slot->has_value()) {
      const U32 sq_in = **io_sq_slot;
      *io_sq_slot = std::nullopt;
      if (gdn_bypass == nullptr && t == 1U && !input_is_f16 && conv < 65536U &&
          gv < 65536U && hv < 65536U && qkvz_parts[0].row_count == conv &&
          qkvz_parts[1].row_count == gv && ab_parts[0].row_count == hv &&
          ab_parts[1].row_count == hv && qkvz_parts[0].in_features == h &&
          qkvz_parts[1].in_features == h && ab_parts[0].in_features == h &&
          ab_parts[1].in_features == h) {
        if (resolve_pretiled_mxfp4_t1(qkvz_parts[0], 1U, &qz_c0, &qz_s0) &&
            resolve_pretiled_mxfp4_t1(qkvz_parts[1], 1U, &qz_c1, &qz_s1) &&
            resolve_pretiled_mxfp4_t1(ab_parts[0], 1U, &ab_c0, &ab_s0) &&
            resolve_pretiled_mxfp4_t1(ab_parts[1], 1U, &ab_c1, &ab_s1)) {
          const Ref nw_ref = direct(block.get("input_norm"), h);
          const bool use_r4 = ((conv + gv + 2U * hv) >= 2048U);
          const U32 rptg = use_r4 ? 8U : 4U;
          const U32 tgs = ((conv + rptg - 1U) / rptg) +
                          ((gv + rptg - 1U) / rptg) +
                          2U * ((hv + rptg - 1U) / rptg);
          const char *fn = use_r4 ? "omlx_m5_pretiled_qmv_mxfp4_concat4_r4_norm_t1"
                                  : "omlx_m5_pretiled_qmv_mxfp4_concat4_r2_norm_t1";
          auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                           {.width = 64U},
                           {hidden, nw_ref, rms_sq_buf, qz_c0, qz_s0, qz_c1,
                            qz_s1, ab_c0, ab_s0, ab_c1, ab_s1, qkv, z, a, b});
          e.dynamic(d, 15, fused_norm_params(conv, gv, hv, hv, h, 0U, sq_in));
          e.record_weight(qkvz_parts[0], conv, h, "pretiled_qmv_t1",
                          "pretiled_mxfp4_qmv_concat4_norm_t1_v4", {},
                          "legacy_route_v1", {}, false);
          e.record_weight(qkvz_parts[1], gv, h, "pretiled_qmv_t1",
                          "pretiled_mxfp4_qmv_concat4_norm_t1_v4", {},
                          "legacy_route_v1", {}, false);
          e.record_weight(ab_parts[0], hv, h, "pretiled_qmv_t1",
                          "pretiled_mxfp4_qmv_concat4_norm_t1_v4", {},
                          "legacy_route_v1", {}, false);
          e.record_weight(ab_parts[1], hv, h, "pretiled_qmv_t1",
                          "pretiled_mxfp4_qmv_concat4_norm_t1_v4", {},
                          "legacy_route_v1", {}, false);
          fused_qkvzab = true;
        } else if (bool lv0 = false, lv1 = false, lv2 = false, lv3 = false;
                   resolve_pretiled_affine_t1(qkvz_parts[0], 1U, &qz_c0, &qz_h0,
                                              &qz_s0, &lv0) &&
                   resolve_pretiled_affine_t1(qkvz_parts[1], 1U, &qz_c1, &qz_h1,
                                              &qz_s1, &lv1) &&
                   resolve_pretiled_affine_t1(ab_parts[0], 1U, &ab_c0, &ab_h0,
                                              &ab_s0, &lv2) &&
                   resolve_pretiled_affine_t1(ab_parts[1], 1U, &ab_c1, &ab_h1,
                                              &ab_s1, &lv3) &&
                   lv0 == lv1 && lv1 == lv2 && lv2 == lv3) {
          const Ref nw_ref = direct(block.get("input_norm"), h);
          const U32 b0 = affine_nax_bits(qkvz_parts[0].format);
          const U32 b1 = affine_nax_bits(qkvz_parts[1].format);
          const U32 b2 = affine_nax_bits(ab_parts[0].format);
          const U32 b3 = affine_nax_bits(ab_parts[1].format);
          const U32 packed_bits =
              b0 | (b1 << 8U) | (b2 << 16U) | (b3 << 24U);
          const bool use_r4 = ((conv + gv + 2U * hv) >= 2048U);
          const U32 rptg = use_r4 ? 8U : 4U;
          const U32 tgs = ((conv + rptg - 1U) / rptg) +
                          ((gv + rptg - 1U) / rptg) +
                          2U * ((hv + rptg - 1U) / rptg);
          const bool all_q456 = (b0 >= 4U && b0 <= 6U) &&
                                (b1 >= 4U && b1 <= 6U) &&
                                (b2 >= 4U && b2 <= 6U) &&
                                (b3 >= 4U && b3 <= 6U);
          const char *fn =
              all_q456
                  ? (use_r4 ? "omlx_m5_pretiled_qmv_affine_concat4_q456_r4_norm_t1"
                            : "omlx_m5_pretiled_qmv_affine_concat4_q456_r2_norm_t1")
                  : (use_r4 ? "omlx_m5_pretiled_qmv_affine_concat4_r4_norm_t1"
                            : "omlx_m5_pretiled_qmv_affine_concat4_r2_norm_t1");
          auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                           {.width = 64U},
                           {hidden, nw_ref, rms_sq_buf, qz_c0, qz_h0, qz_s0,
                            qz_c1, qz_h1, qz_s1, ab_c0, ab_h0, ab_s0, ab_c1,
                            ab_h1, ab_s1, qkv, z, a, b});
          e.dynamic(d, 19,
                    fused_norm_params(conv, gv, hv, hv,
                                      pretiled_k_flags(h, lv0, false),
                                      packed_bits, sq_in));
          e.record_weight(qkvz_parts[0], conv, h, "pretiled_qmv_t1",
                          "pretiled_affine_qmv_concat4_norm_t1_v4", {},
                          "legacy_route_v1", {}, false);
          e.record_weight(qkvz_parts[1], gv, h, "pretiled_qmv_t1",
                          "pretiled_affine_qmv_concat4_norm_t1_v4", {},
                          "legacy_route_v1", {}, false);
          e.record_weight(ab_parts[0], hv, h, "pretiled_qmv_t1",
                          "pretiled_affine_qmv_concat4_norm_t1_v4", {},
                          "legacy_route_v1", {}, false);
          e.record_weight(ab_parts[1], hv, h, "pretiled_qmv_t1",
                          "pretiled_affine_qmv_concat4_norm_t1_v4", {},
                          "legacy_route_v1", {}, false);
          fused_qkvzab = true;
        }
      }
      if (!fused_qkvzab) {
        normalize(e, block.get("input_norm"), hidden, norm);
      }
    }
    if (!fused_qkvzab && t == 1U && !input_is_f16 &&
        qkvz_parts[0].row_count == conv &&
        qkvz_parts[1].row_count == gv && ab_parts[0].row_count == hv &&
        ab_parts[1].row_count == hv && qkvz_parts[0].in_features == h &&
        qkvz_parts[1].in_features == h && ab_parts[0].in_features == h &&
        ab_parts[1].in_features == h) {
      if (resolve_pretiled_mxfp4_t1(qkvz_parts[0], 1U, &qz_c0, &qz_s0) &&
          resolve_pretiled_mxfp4_t1(qkvz_parts[1], 1U, &qz_c1, &qz_s1) &&
          resolve_pretiled_mxfp4_t1(ab_parts[0], 1U, &ab_c0, &ab_s0) &&
          resolve_pretiled_mxfp4_t1(ab_parts[1], 1U, &ab_c1, &ab_s1)) {
        const bool use_r4 = ((conv + gv + 2U * hv) >= 2048U);
        const U32 rptg = use_r4 ? 8U : 4U;
        const U32 tgs = ((conv + rptg - 1U) / rptg) +
                        ((gv + rptg - 1U) / rptg) +
                        2U * ((hv + rptg - 1U) / rptg);
        const char *fn = use_r4 ? "omlx_m5_pretiled_qmv_mxfp4_concat4_r4_t1"
                                : "omlx_m5_pretiled_qmv_mxfp4_concat4_r2_t1";
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U},
                         {norm, qz_c0, qz_s0, qz_c1, qz_s1, ab_c0, ab_s0, ab_c1,
                          ab_s1, qkv, z, a, b});
        e.dynamic(d, 13, words({conv, gv, hv, hv, h}));
        e.record_weight(qkvz_parts[0], conv, h, "pretiled_qmv_t1",
                        "pretiled_mxfp4_qmv_concat4_t1_v3", {},
                        "legacy_route_v1", {}, false);
        e.record_weight(qkvz_parts[1], gv, h, "pretiled_qmv_t1",
                        "pretiled_mxfp4_qmv_concat4_t1_v3", {},
                        "legacy_route_v1", {}, false);
        e.record_weight(ab_parts[0], hv, h, "pretiled_qmv_t1",
                        "pretiled_mxfp4_qmv_concat4_t1_v3", {},
                        "legacy_route_v1", {}, false);
        e.record_weight(ab_parts[1], hv, h, "pretiled_qmv_t1",
                        "pretiled_mxfp4_qmv_concat4_t1_v3", {},
                        "legacy_route_v1", {}, false);
        fused_qkvzab = true;
      } else if (bool lv0 = false, lv1 = false, lv2 = false, lv3 = false;
                 resolve_pretiled_affine_t1(qkvz_parts[0], 1U, &qz_c0, &qz_h0,
                                            &qz_s0, &lv0) &&
                 resolve_pretiled_affine_t1(qkvz_parts[1], 1U, &qz_c1, &qz_h1,
                                            &qz_s1, &lv1) &&
                 resolve_pretiled_affine_t1(ab_parts[0], 1U, &ab_c0, &ab_h0,
                                            &ab_s0, &lv2) &&
                 resolve_pretiled_affine_t1(ab_parts[1], 1U, &ab_c1, &ab_h1,
                                            &ab_s1, &lv3) &&
                 lv0 == lv1 && lv1 == lv2 && lv2 == lv3) {
        const U32 b0 = affine_nax_bits(qkvz_parts[0].format);
        const U32 b1 = affine_nax_bits(qkvz_parts[1].format);
        const U32 b2 = affine_nax_bits(ab_parts[0].format);
        const U32 b3 = affine_nax_bits(ab_parts[1].format);
        const U32 packed_bits =
            b0 | (b1 << 8U) | (b2 << 16U) | (b3 << 24U);
        const bool use_r4 = ((conv + gv + 2U * hv) >= 2048U);
        const U32 rptg = use_r4 ? 8U : 4U;
        const U32 tgs = ((conv + rptg - 1U) / rptg) +
                        ((gv + rptg - 1U) / rptg) +
                        2U * ((hv + rptg - 1U) / rptg);
        const bool all_q456 = (b0 >= 4U && b0 <= 6U) && (b1 >= 4U && b1 <= 6U) &&
                              (b2 >= 4U && b2 <= 6U) && (b3 >= 4U && b3 <= 6U);
        const char *fn =
            all_q456
                ? (use_r4 ? "omlx_m5_pretiled_qmv_affine_concat4_q456_r4_t1"
                          : "omlx_m5_pretiled_qmv_affine_concat4_q456_r2_t1")
                : (use_r4 ? "omlx_m5_pretiled_qmv_affine_concat4_r4_t1"
                          : "omlx_m5_pretiled_qmv_affine_concat4_r2_t1");
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U},
                         {norm, qz_c0, qz_h0, qz_s0, qz_c1, qz_h1, qz_s1, ab_c0,
                          ab_h0, ab_s0, ab_c1, ab_h1, ab_s1, qkv, z, a, b});
        e.dynamic(d, 17,
                  words({conv, gv, hv, hv, pretiled_k_flags(h, lv0, false),
                         packed_bits}));
        e.record_weight(qkvz_parts[0], conv, h, "pretiled_qmv_t1",
                        "pretiled_affine_qmv_concat4_t1_v3", {},
                        "legacy_route_v1", {}, false);
        e.record_weight(qkvz_parts[1], gv, h, "pretiled_qmv_t1",
                        "pretiled_affine_qmv_concat4_t1_v3", {},
                        "legacy_route_v1", {}, false);
        e.record_weight(ab_parts[0], hv, h, "pretiled_qmv_t1",
                        "pretiled_affine_qmv_concat4_t1_v3", {},
                        "legacy_route_v1", {}, false);
        e.record_weight(ab_parts[1], hv, h, "pretiled_qmv_t1",
                        "pretiled_affine_qmv_concat4_t1_v3", {},
                        "legacy_route_v1", {}, false);
        fused_qkvzab = true;
      }
    }
    if (!fused_qkvzab) {
      if (t == 1U && !input_is_f16 && qkvz_parts[0].row_count == conv &&
          qkvz_parts[1].row_count == gv && qkvz_parts[0].in_features == h &&
          qkvz_parts[1].in_features == h &&
          resolve_pretiled_mxfp4_t1(qkvz_parts[0], 1U, &qz_c0, &qz_s0) &&
          resolve_pretiled_mxfp4_t1(qkvz_parts[1], 1U, &qz_c1, &qz_s1)) {
        const U32 rows_per_tg = ((conv + gv) >= 4096U) ? 8U : 4U;
        const U32 tgs = ((conv + rows_per_tg - 1U) / rows_per_tg) +
                        ((gv + rows_per_tg - 1U) / rows_per_tg);
        auto &d = e.emit("row_exact", "omlx_m5_pretiled_qmv_mxfp4_concat2_t1",
                         {.width = U64(tgs) * 64U}, {.width = 64U},
                         {norm, qz_c0, qz_s0, qz_c1, qz_s1, qkv, z});
        e.dynamic(d, 7, words({conv, gv, 0U, h}));
        for (std::size_t i = 0; i < 2; ++i) {
          e.record_weight(qkvz_parts[i], qkvz_ns[i], h, "pretiled_qmv_t1",
                          "pretiled_mxfp4_qmv_concat2_t1_v3", {},
                          "legacy_route_v1", {}, false);
        }
      } else if (bool lv0 = false, lv1 = false;
                 t == 1U && !input_is_f16 &&
                 qkvz_parts[0].row_count == conv &&
                 qkvz_parts[1].row_count == gv &&
                 qkvz_parts[0].in_features == h &&
                 qkvz_parts[1].in_features == h &&
                 resolve_pretiled_affine_t1(qkvz_parts[0], 1U, &qz_c0, &qz_h0,
                                            &qz_s0, &lv0) &&
                 resolve_pretiled_affine_t1(qkvz_parts[1], 1U, &qz_c1, &qz_h1,
                                            &qz_s1, &lv1) &&
                 lv0 == lv1) {
        const U32 b0 = affine_nax_bits(qkvz_parts[0].format);
        const U32 b1 = affine_nax_bits(qkvz_parts[1].format);
        const U32 packed_bits = b0 | (b1 << 8U);
        const U32 rows_per_tg = ((conv + gv) >= 4096U) ? 8U : 4U;
        const U32 tgs = ((conv + rows_per_tg - 1U) / rows_per_tg) +
                        ((gv + rows_per_tg - 1U) / rows_per_tg);
        auto &d = e.emit("row_exact", "omlx_m5_pretiled_qmv_affine_concat2_t1",
                         {.width = U64(tgs) * 64U}, {.width = 64U},
                         {norm, qz_c0, qz_h0, qz_s0, qz_c1, qz_h1, qz_s1, qkv, z});
        e.dynamic(d, 9,
                  words({conv, gv, 0U, pretiled_k_flags(h, lv0, false),
                         packed_bits}));
        for (std::size_t i = 0; i < 2; ++i) {
          e.record_weight(qkvz_parts[i], qkvz_ns[i], h, "pretiled_qmv_t1",
                          "pretiled_affine_qmv_concat2_t1_v3", {},
                          "legacy_route_v1", {}, false);
        }
      } else if (input_is_a4 &&
                 try_concat_w4a4_nax(e, block.concat_gdn_qkvz_id, qkvz_parts,
                                     qkvz_ns, h, norm, concat_out)) {
        emit_split2(e, concat_out, qkv, z, conv, gv);
      } else if (input_is_a4) {
        if (!try_linear_w4a4_base(e, block.get("gdn_qkv"), norm, qkv, conv, h))
          linear_base(e, block.get("gdn_qkv"), norm, qkv, conv, h);
        if (!try_linear_w4a4_base(e, block.get("gdn_z"), norm, z, gv, h))
          linear_base(e, block.get("gdn_z"), norm, z, gv, h);
      } else if (input_is_a8 &&
                 try_concat_w4a8_nax(e, block.concat_gdn_qkvz_id, qkvz_parts,
                                     qkvz_ns, h, norm, concat_out)) {
        emit_split2(e, concat_out, qkv, z, conv, gv);
      } else if (input_is_a8) {
        if (!try_linear_w4a8_base(e, block.get("gdn_qkv"), norm, qkv, conv, h))
          linear_base(e, block.get("gdn_qkv"), norm, qkv, conv, h);
        if (!try_linear_w4a8_base(e, block.get("gdn_z"), norm, z, gv, h))
          linear_base(e, block.get("gdn_z"), norm, z, gv, h);
      } else if (try_concat_nax(e, block.concat_gdn_qkvz_id, qkvz_parts, qkvz_ns,
                                h, input_is_f16, norm, concat_out)) {
        if (t == 1U && !inspect_tensors_workspace) {
          qkv_ref = Ref{concat_out.buffer, 0, mul(conv, 2)};
          z_ref = Ref{concat_out.buffer, mul(conv, 2), mul(gv, 2)};
        } else {
          emit_split2(e, concat_out, qkv, z, conv, gv);
        }
      } else if (input_is_f16) {
        linear_from_f16(e, block.get("gdn_qkv"), f16_norm, f16_ones_scale, qkv,
                        conv, h);
        linear_from_f16(e, block.get("gdn_z"), f16_norm, f16_ones_scale, z, gv,
                        h);
      } else {
        linear_base(e, block.get("gdn_qkv"), norm, qkv, conv, h);
        linear_base(e, block.get("gdn_z"), norm, z, gv, h);
      }

      const Ref ab_concat =
          (t == 1U && !inspect_tensors_workspace && concat_out.buffer != nullptr)
              ? Ref{concat_out.buffer, mul(conv + gv, 2), mul(2U * hv, 2)}
              : concat_out;
      if (t == 1U && !input_is_f16 && ab_parts[0].row_count == hv &&
          ab_parts[1].row_count == hv && ab_parts[0].in_features == h &&
          ab_parts[1].in_features == h &&
          resolve_pretiled_mxfp4_t1(ab_parts[0], 1U, &ab_c0, &ab_s0) &&
          resolve_pretiled_mxfp4_t1(ab_parts[1], 1U, &ab_c1, &ab_s1)) {
        const U32 rows_per_tg = ((2U * hv) >= 4096U) ? 8U : 4U;
        const U32 tgs = 2U * ((hv + rows_per_tg - 1U) / rows_per_tg);
        auto &d = e.emit("row_exact", "omlx_m5_pretiled_qmv_mxfp4_concat2_t1",
                         {.width = U64(tgs) * 64U}, {.width = 64U},
                         {norm, ab_c0, ab_s0, ab_c1, ab_s1, a, b});
        e.dynamic(d, 7, words({hv, hv, 0U, h}));
        for (std::size_t i = 0; i < 2; ++i) {
          e.record_weight(ab_parts[i], ab_ns[i], h, "pretiled_qmv_t1",
                          "pretiled_mxfp4_qmv_concat2_t1_v3", {},
                          "legacy_route_v1", {}, false);
        }
      } else if (bool lv0 = false, lv1 = false;
                 t == 1U && !input_is_f16 &&
                 ab_parts[0].row_count == hv && ab_parts[1].row_count == hv &&
                 ab_parts[0].in_features == h && ab_parts[1].in_features == h &&
                 resolve_pretiled_affine_t1(ab_parts[0], 1U, &ab_c0, &ab_h0,
                                            &ab_s0, &lv0) &&
                 resolve_pretiled_affine_t1(ab_parts[1], 1U, &ab_c1, &ab_h1,
                                            &ab_s1, &lv1) &&
                 lv0 == lv1) {
        const U32 b0 = affine_nax_bits(ab_parts[0].format);
        const U32 b1 = affine_nax_bits(ab_parts[1].format);
        const U32 packed_bits = b0 | (b1 << 8U);
        const U32 rows_per_tg = ((2U * hv) >= 4096U) ? 8U : 4U;
        const U32 tgs = 2U * ((hv + rows_per_tg - 1U) / rows_per_tg);
        auto &d = e.emit("row_exact", "omlx_m5_pretiled_qmv_affine_concat2_t1",
                         {.width = U64(tgs) * 64U}, {.width = 64U},
                         {norm, ab_c0, ab_h0, ab_s0, ab_c1, ab_h1, ab_s1, a, b});
        e.dynamic(d, 9,
                  words({hv, hv, 0U, pretiled_k_flags(h, lv0, false),
                         packed_bits}));
        for (std::size_t i = 0; i < 2; ++i) {
          e.record_weight(ab_parts[i], ab_ns[i], h, "pretiled_qmv_t1",
                          "pretiled_affine_qmv_concat2_t1_v3", {},
                          "legacy_route_v1", {}, false);
        }
      } else if (input_is_a4 &&
                 try_concat_w4a4_nax(e, block.concat_gdn_ab_id, ab_parts, ab_ns,
                                     h, norm, ab_concat)) {
        emit_split2(e, ab_concat, a, b, hv, hv);
      } else if (input_is_a4) {
        if (!try_linear_w4a4_base(e, block.get("gdn_a"), norm, a, hv, h))
          linear_base(e, block.get("gdn_a"), norm, a, hv, h);
        if (!try_linear_w4a4_base(e, block.get("gdn_b"), norm, b, hv, h))
          linear_base(e, block.get("gdn_b"), norm, b, hv, h);
      } else if (input_is_a8 &&
                 try_concat_w4a8_nax(e, block.concat_gdn_ab_id, ab_parts, ab_ns,
                                     h, norm, ab_concat)) {
        emit_split2(e, ab_concat, a, b, hv, hv);
      } else if (input_is_a8) {
        if (!try_linear_w4a8_base(e, block.get("gdn_a"), norm, a, hv, h))
          linear_base(e, block.get("gdn_a"), norm, a, hv, h);
        if (!try_linear_w4a8_base(e, block.get("gdn_b"), norm, b, hv, h))
          linear_base(e, block.get("gdn_b"), norm, b, hv, h);
      } else if (try_concat_nax(e, block.concat_gdn_ab_id, ab_parts, ab_ns, h,
                                input_is_f16, norm, ab_concat)) {
        if (t == 1U && !inspect_tensors_workspace) {
          a_ref = Ref{ab_concat.buffer, ab_concat.offset, mul(hv, 2)};
          b_ref =
              Ref{ab_concat.buffer, ab_concat.offset + mul(hv, 2), mul(hv, 2)};
        } else {
          emit_split2(e, concat_out, a, b, hv, hv);
        }
      } else if (input_is_f16) {
        linear_from_f16(e, block.get("gdn_a"), f16_norm, f16_ones_scale, a, hv,
                        h);
        linear_from_f16(e, block.get("gdn_b"), f16_norm, f16_ones_scale, b, hv,
                        h);
      } else {
        linear_base(e, block.get("gdn_a"), norm, a, hv, h);
        linear_base(e, block.get("gdn_b"), norm, b, hv, h);
      }
    }
    if (gdn_bypass != nullptr) {
      const std::array<detail::lora_rt::BypassPartDest, 4> dests{{
          {qkv_ref, conv, 0U},
          {z_ref, gv, 0U},
          {a_ref, hv, 0U},
          {b_ref, hv, 0U},
      }};
      detail::lora_rt::emit_bypass_sibling_group(
          e, *gdn_bypass, norm, dests, lora_bypass_tmp_, bf16_nax_bypass_ok_,
          silicon_tuning.gpu_cores);
    }

    bool gdn_fused_sq = false;
    emit_gdn_state_step(
        e, block, block, qkv_ref, z_ref, a_ref, b_ref, convolved, recurrent,
        mixer, t, verify, true, true, 0U,
        io_sq_slot != nullptr ? out_sq_slot : std::optional<U32>{},
        &gdn_fused_sq, /*out_w4a8=*/input_is_a8, false, true, nullptr,
        /*out_w4a4=*/input_is_a4);
    if (io_sq_slot != nullptr) {
      *io_sq_slot = gdn_fused_sq ? out_sq_slot : std::nullopt;
    }
  }

  void swiglu(Encoder &e, U32 rows, U32 features_count,
              std::optional<U32> explicit_base_rows = {}) {
    auto &d = e.emit("mlp", "omlx_m5_swiglu_a16",
                     {.width = 32, .height = U64(rows) * (features_count / 32)},
                     {.width = 32}, {gate, up, product});
    if (explicit_base_rows.has_value()) {
      e.dynamic(d, 3, words({rows, features_count, features_count / 32, 0}));
    } else {
      e.dynamic(d, 3, words({rows, features_count, features_count / 32, 0}), 0,
                {}, rows / e.profile.capacity);
    }
  }

  void expert_linear(Encoder &e, const MaterializedWeightView &w, Ref x, Ref y,
                     U32 n, U32 k, bool down,
                     std::optional<U32> explicit_t = {}) {
    if (w.parent_shape != std::vector<U64>{256, n, k})
      throw std::invalid_argument("Qwen expert bank shape mismatch: " + w.name);
    const U32 t = explicit_t.value_or(e.profile.capacity);
    const bool diagnostic =
        moe_override && moe_override->object == w.name &&
        moe_override->format == w.format && moe_override->layout == w.layout &&
        moe_override->n == n && moe_override->k == k &&
        moe_override->capacity == t;
    moe_override_matched |= diagnostic;
    const bool fallback = diagnostic && moe_override->choice == "fallback";
    const auto function = [&] {
      auto kernel = select_grouped_expert_kernel(w.format);
      if (fallback)
        kernel.replace(kernel.size() - std::string("grouped").size(),
                       std::string("grouped").size(), "fallback");
      return kernel;
    }();
    const auto family = fallback ? "fallback_a16" : "grouped_a16";
    const auto source = diagnostic ? "p12_exact_ab" : "baseline";
    const auto params =
        words({256, 8, n, k, U32(w.padded_k), t, U32(down), U32(down)});
    const std::optional<U32> tok_w5 =
        explicit_t.has_value() ? std::optional<U32>{} : std::optional<U32>{5U};
    const MetalDispatchSize grid{
        .width = fallback ? U64(t) * 8 * n * 32
                          : U64(t) * 8 * ((n + 3) / 4) * 128};
    const MetalDispatchSize threads{.width = fallback ? 32U : 128U};
    if (w.format == "mxfp4_e2m1_g32_e8m0_v1") {
      auto &d = e.emit(
          "moe", function, grid, threads,
          {x, route_ids, route_scores, plane(w, "data"), plane(w, "scale"), y});
      e.dynamic(d, 6, params, tok_w5);
      e.record_weight(w, n, k, family, source,
                      diagnostic ? "explicit_p12_ab"
                                 : "expert_candidate_unqualified",
                      "legacy_route_v1", {}, false, explicit_t);
      return;
    }
    if (!w.format.starts_with("oq_affine_q"))
      throw std::invalid_argument("expert bank format unsupported");
    U32 bits = std::stoul(w.format.substr(11));
    U32 group = std::stoul(w.format.substr(w.format.find("_g") + 2));
    Ref high = bits == 5   ? plane(w, "high_bit")
               : bits == 6 ? plane(w, "high_2bit")
                           : zero;
    const U32 mult = down ? 8U : 1U;
    const bool dyn_rows = !explicit_t.has_value() && (t != 1 || mult != 1);
    const Encoder::GroupSumKey sum_key{sums.buffer, sums.offset, x.buffer,
                                       x.offset,    k,           group,
                                       t * mult,    dyn_rows,    mult};
    if (!e.has_group_sum(sum_key)) {
      auto &sum =
          e.emit("moe", "omlx_m5_moe_group_sum_g" + std::to_string(group),
                 {.width = U64(t) * mult * (k / group) * 32}, {.width = 32},
                 {x, sums});
      e.dynamic(sum, 2, params, tok_w5);
      e.set_group_sum(sum_key);
    }
    auto &d = e.emit(
        "moe", function, grid, threads,
        {x, route_ids, route_scores, plane(w, bits == 8 ? "code" : "base_code"),
         high, plane(w, "scale"), plane(w, "bias"), sums, y});
    e.dynamic(d, 9, params, tok_w5);
    e.record_weight(w, n, k, family, source,
                    diagnostic ? "explicit_p12_ab"
                               : "expert_candidate_unqualified",
                    "legacy_route_v1", {}, false, explicit_t);
  }

  bool can_moe_nax_for(const Block &block, U32 t,
                       std::optional<U32> explicit_t = {}) const {
    if (moe_override || !moe_nax_ws.active || t > moe_nax_ws.max_t ||
        t < moe_nax_min_tokens())
      return false;
    const auto &gw = block.get("moe_expert_gate");
    const auto &uw = block.get("moe_expert_up");
    const auto &dw = block.get("moe_expert_down");
    if (gw.format != uw.format || gw.padded_k != uw.padded_k ||
        gw.parent_shape != std::vector<U64>{256, f, h} ||
        uw.parent_shape != std::vector<U64>{256, f, h} ||
        dw.parent_shape != std::vector<U64>{256, h, f})
      return false;
    const bool has_moe_min_env =
        (std::getenv("OMLX_M5_MOE_NAX_MIN_T") != nullptr);
    const bool force_t1_moe_qmv =
        gw.format == "mxfp4_e2m1_g32_e8m0_v1" && !speculative_enabled &&
        allow_tdom_t1() && explicit_t.value_or(0U) < 2U && t_domain_active() &&
        !has_moe_min_env;
    const bool unlock_moe_nax =
        !force_t1_moe_qmv &&
        (speculative_enabled ||
         linear_policy == LinearNumericPolicy::RowExactPortableV1 ||
         explicit_t.value_or(0U) >= 2U || has_moe_min_env);
    const U32 eff_moe_min_t =
        has_moe_min_env
            ? moe_nax_min_tokens()
            : (unlock_moe_nax
                   ? silicon_tuning.moe_nax_min_tokens
                   : std::max<U32>(silicon_tuning.moe_nax_min_tokens, 8U));
    return t >= eff_moe_min_t && moe_nax_weight_eligible(gw, uw, dw, h, f);
  }

  bool try_fused_moe_ffn(Encoder &e, const Block &block,
                         bool &wrote_expert_output,
                         Ref fused_router_logits = {},
                         std::optional<U32> explicit_t = {},
                         bool verify = false, bool input_is_a8 = false,
                         bool input_is_a4 = false) {
    wrote_expert_output = false;
    if (moe_override)
      return false;
    const auto &gw = block.get("moe_expert_gate");
    const auto &uw = block.get("moe_expert_up");
    const auto &dw = block.get("moe_expert_down");
    if (gw.format != uw.format || gw.padded_k != uw.padded_k ||
        gw.parent_shape != std::vector<U64>{256, f, h} ||
        uw.parent_shape != std::vector<U64>{256, f, h} ||
        dw.parent_shape != std::vector<U64>{256, h, f}) {
      return false;
    }
    const U32 t = explicit_t.value_or(e.profile.capacity);
    const U32 sel_t = nax_effective_select_t(e, explicit_t);
    if (input_is_a4 && !verify &&
        moe_w4a4_nax_runtime_eligible(context, libraries) &&
        ensure_a4_quant(e, norm, h, explicit_t) &&
        emit_moe_w4a4_nax_ffn(e, moe_nax_ws, gw, uw, dw, w4a4_codes,
                              w4a4_scales, w4a4_nax_mp(t), route_ids,
                              route_scores, expert_output, routed, t, h, f,
                              8u, 256u, explicit_t, fused_router_logits,
                              /*emit_reduce=*/false, w4a4_down_proj_)) {
      wrote_expert_output = true;
      return true;
    }
    if (input_is_a8 && !verify &&
        moe_w4a8_nax_runtime_eligible(context, libraries) &&
        ensure_a8_quant(e, norm, h, explicit_t) &&
        emit_moe_w4a8_nax_ffn(e, moe_nax_ws, gw, uw, dw, a8_qa, a8_sa, a8_ra,
                              w4a8_nax_mp(t), route_ids, route_scores,
                              expert_output, routed, t, h, f, 8u, 256u,
                              explicit_t, fused_router_logits,
                              /*emit_reduce=*/false, w4a8_down_proj_)) {
      wrote_expert_output = true;
      return true;
    }
    const bool has_moe_min_env =
        (std::getenv("OMLX_M5_MOE_NAX_MIN_T") != nullptr);
    const bool force_t1_moe_qmv =
        gw.format == "mxfp4_e2m1_g32_e8m0_v1" && !speculative_enabled &&
        allow_tdom_t1() && explicit_t.value_or(0U) < 2U && t_domain_active() &&
        !has_moe_min_env;
    const bool unlock_moe_nax =
        !force_t1_moe_qmv &&
        (speculative_enabled ||
         linear_policy == LinearNumericPolicy::RowExactPortableV1 ||
         explicit_t.value_or(0U) >= 2U || has_moe_min_env);
    const U32 eff_moe_min_t =
        has_moe_min_env
            ? moe_nax_min_tokens()
            : (unlock_moe_nax
                   ? silicon_tuning.moe_nax_min_tokens
                   : std::max<U32>(silicon_tuning.moe_nax_min_tokens, 8U));
    const bool use_out64 =
        !verify && sel_t >= 32U &&
        !detail::forward::env_flag_enabled("OMLX_M5_DISABLE_SWIGLU_OUT64");
    if (t >= eff_moe_min_t &&
        emit_moe_nax_ffn(e, moe_nax_ws, gw, uw, dw, norm, route_ids,
                         route_scores, expert_output, routed, t, h, f, 8u, 256u,
                         silicon_tuning.gpu_cores, explicit_t,
                         fused_router_logits, /*emit_reduce=*/false,
                         use_out64)) {
      wrote_expert_output = true;
      return true;
    }

    const auto up_params = words({256, 8, f, h, U32(gw.padded_k), t, 0U, 0U});
    const auto down_params = words({256, 8, h, f, U32(dw.padded_k), t, 1U, 1U});
    const std::optional<U32> tok_w5 =
        explicit_t.has_value() ? std::optional<U32>{} : std::optional<U32>{5U};
    const MetalDispatchSize up_grid{.width = U64(t) * 8 * ((f + 15) / 16) * 128};
    const MetalDispatchSize up_threads{.width = 128U};
    const MetalDispatchSize down_grid{.width = U64(t) * h * 256U};
    const MetalDispatchSize down_threads{.width = 256U};

    if (gw.format == "mxfp4_e2m1_g32_e8m0_v1" &&
        dw.format == "mxfp4_e2m1_g32_e8m0_v1") {
      const MetalDispatchSize mxfp4_down_grid{
          .width = U64(t) * ((h + 7U) / 8U) * 256U};
      auto &d1 = e.emit("moe", "omlx_m5_mxfp4_moe_w4a16_swiglu_grouped",
                        up_grid, up_threads,
                        {norm, route_ids, plane(gw, "data"), plane(gw, "scale"),
                         plane(uw, "data"), plane(uw, "scale"), product});
      e.dynamic(d1, 7, up_params, tok_w5);
      e.record_weight(gw, f, h, "grouped_a16", "fused_swiglu_v3",
                      "expert_candidate_unqualified", "legacy_route_v1", {},
                      false, explicit_t);
      e.record_weight(uw, f, h, "grouped_a16", "fused_swiglu_v3",
                      "expert_candidate_unqualified", "legacy_route_v1", {},
                      false, explicit_t);
      if (dw.has_fwht64()) {
        detail::forward::emit_fwht64_bf16(e, product, product, f, explicit_t,
                                          8U);
      }
      auto &d2 = e.emit("moe", "omlx_m5_mxfp4_moe_w4a16_down_reduce",
                        mxfp4_down_grid, down_threads,
                        {product, route_ids, route_scores, plane(dw, "data"),
                         plane(dw, "scale"), routed});
      e.dynamic(d2, 6, down_params, tok_w5);
      e.record_weight(dw, h, f, "grouped_a16", "fused_down_reduce_v3",
                      "expert_candidate_unqualified", "legacy_route_v1", {},
                      false, explicit_t);
      return true;
    }

    if (gw.format.starts_with("oq_affine_q") &&
        dw.format.starts_with("oq_affine_q")) {
      const U32 u_bits = std::stoul(gw.format.substr(11));
      const U32 u_group =
          std::stoul(gw.format.substr(gw.format.find("_g") + 2));
      const U32 d_bits = std::stoul(dw.format.substr(11));
      const U32 d_group =
          std::stoul(dw.format.substr(dw.format.find("_g") + 2));
      if (u_bits < 4U || d_bits < 4U)
        return false;
      Ref g_high = u_bits == 5   ? plane(gw, "high_bit")
                   : u_bits == 6 ? plane(gw, "high_2bit")
                                 : zero;
      Ref u_high = u_bits == 5   ? plane(uw, "high_bit")
                   : u_bits == 6 ? plane(uw, "high_2bit")
                                 : zero;
      Ref d_high = d_bits == 5   ? plane(dw, "high_bit")
                   : d_bits == 6 ? plane(dw, "high_2bit")
                                 : zero;
      const Encoder::GroupSumKey u_sum_key{
          sums.buffer, sums.offset, norm.buffer,
          norm.offset, h,           u_group,
          t,           !explicit_t.has_value() && t != 1,
          1U};
      if (!e.has_group_sum(u_sum_key)) {
        auto &sum =
            e.emit("moe", "omlx_m5_moe_group_sum_g" + std::to_string(u_group),
                   {.width = U64(t) * (h / u_group) * 32}, {.width = 32},
                   {norm, sums});
        e.dynamic(sum, 2, up_params, tok_w5);
        e.set_group_sum(u_sum_key);
      }
      const std::string fn1 = "omlx_m5_oq_q" + std::to_string(u_bits) + "_g" +
                              std::to_string(u_group) + "_moe_swiglu_grouped";
      auto &d1 = e.emit(
          "moe", fn1, up_grid, up_threads,
          {norm, route_ids, plane(gw, u_bits == 8 ? "code" : "base_code"),
           g_high, plane(gw, "scale"), plane(gw, "bias"),
           plane(uw, u_bits == 8 ? "code" : "base_code"), u_high,
           plane(uw, "scale"), plane(uw, "bias"), sums, product});
      e.dynamic(d1, 12, up_params, tok_w5);
      e.record_weight(gw, f, h, "grouped_a16", "fused_swiglu_v3",
                      "expert_candidate_unqualified", "legacy_route_v1", {},
                      false, explicit_t);
      e.record_weight(uw, f, h, "grouped_a16", "fused_swiglu_v3",
                      "expert_candidate_unqualified", "legacy_route_v1", {},
                      false, explicit_t);

      if (dw.has_fwht64()) {
        detail::forward::emit_fwht64_bf16(e, product, product, f, explicit_t,
                                          8U);
      }
      const Encoder::GroupSumKey d_sum_key{
          sums.buffer,    sums.offset,    product.buffer,
          product.offset, f,              d_group,
          t * 8U,         !explicit_t.has_value(),
          8U,             dw.has_fwht64()};
      if (!e.has_group_sum(d_sum_key)) {
        auto &sum =
            e.emit("moe", "omlx_m5_moe_group_sum_g" + std::to_string(d_group),
                   {.width = U64(t) * 8U * (f / d_group) * 32}, {.width = 32},
                   {product, sums});
        e.dynamic(sum, 2, down_params, tok_w5);
        e.set_group_sum(d_sum_key);
      }
      const std::string fn2 = "omlx_m5_oq_q" + std::to_string(d_bits) + "_g" +
                              std::to_string(d_group) + "_moe_down_reduce";
      auto &d2 = e.emit(
          "moe", fn2, down_grid, down_threads,
          {product, route_ids, route_scores,
           plane(dw, d_bits == 8 ? "code" : "base_code"), d_high,
           plane(dw, "scale"), plane(dw, "bias"), sums, routed});
      e.dynamic(d2, 9, down_params, tok_w5);
      e.record_weight(dw, h, f, "grouped_a16", "fused_down_reduce_v3",
                      "expert_candidate_unqualified", "legacy_route_v1", {},
                      false, explicit_t);
      return true;
    }
    return false;
  }

  void mlp(Encoder &e, const Block &block, U32 layer_index,
           bool input_is_f16 = false, std::optional<U32> explicit_t = {},
           std::optional<U32> *io_sq_slot = nullptr,
           std::optional<U32> out_sq_slot = std::nullopt,
           bool allow_moe9_bt = false, bool input_is_a8 = false,
           bool verify = false, bool input_is_a4 = false) {
    const auto t = explicit_t.value_or(e.profile.capacity);
    if (!config.num_experts) {
      if (input_is_a4) {
        const auto &dw_v = block.get("mlp_down");
        const bool down_w4a4 = dw_v.has_fwht64() || w4a4_down_proj_;
        const bool mode_b_ok =
            down_w4a4 && dw_v.has_fwht64() && can_w4a4_weight(dw_v, h, f) &&
            try_linear_w4a4_swiglu_fwht64_a4(
                e, block.get("mlp_gate"), block.get("mlp_up"), norm, product, f,
                h, has_bypass_lora(dw_v) || inspect_tensors_workspace,
                explicit_t);
        if (!mode_b_ok &&
            !try_linear_w4a4_swiglu(e, block.get("mlp_gate"),
                                    block.get("mlp_up"), norm, product, f, h,
                                    explicit_t)) {
          if (!try_linear_w4a4_base(e, block.get("mlp_gate"), norm, gate, f, h,
                                    explicit_t))
            linear_base(e, block.get("mlp_gate"), norm, gate, f, h, explicit_t);
          if (!try_linear_w4a4_base(e, block.get("mlp_up"), norm, up, f, h,
                                    explicit_t))
            linear_base(e, block.get("mlp_up"), norm, up, f, h, explicit_t);
          if (const auto *mlp_gu_bypass = find_bypass_group(
                  layer_index, QwenBypassGroupKind::mlp_gate_up);
              mlp_gu_bypass != nullptr) {
            const std::array<detail::lora_rt::BypassPartDest, 2> dests{{
                {gate, f, 0U},
                {up, f, 0U},
            }};
            detail::lora_rt::emit_bypass_sibling_group(
                e, *mlp_gu_bypass, norm, dests, lora_bypass_tmp_,
                bf16_nax_bypass_ok_, silicon_tuning.gpu_cores, explicit_t);
          }
          swiglu(e, t, f, explicit_t);
        }
        if (down_w4a4 &&
            try_linear_w4a4_base(e, dw_v, product, delta, h, f, explicit_t)) {
          emit_bypass_for_weight(e, dw_v, product, delta, h, explicit_t);
        } else {
          linear(e, dw_v, product, delta, h, f, explicit_t);
        }
        if (io_sq_slot != nullptr)
          *io_sq_slot = std::nullopt;
        return;
      }
      if (input_is_a8) {
        const auto &dw_v = block.get("mlp_down");
        const bool down_w4a8 = dw_v.has_fwht64() || w4a8_down_proj_;
        const bool mode_b_ok =
            down_w4a8 && dw_v.has_fwht64() && can_w4a8_weight(dw_v, h, f) &&
            try_linear_w4a8_swiglu_fwht64_a8(
                e, block.get("mlp_gate"), block.get("mlp_up"), norm, product, f,
                h, has_bypass_lora(dw_v) || inspect_tensors_workspace,
                explicit_t);
        if (!mode_b_ok &&
            !try_linear_w4a8_swiglu(e, block.get("mlp_gate"),
                                    block.get("mlp_up"), norm, product, f, h,
                                    explicit_t)) {
          if (!try_linear_w4a8_base(e, block.get("mlp_gate"), norm, gate, f, h,
                                    explicit_t))
            linear_base(e, block.get("mlp_gate"), norm, gate, f, h, explicit_t);
          if (!try_linear_w4a8_base(e, block.get("mlp_up"), norm, up, f, h,
                                    explicit_t))
            linear_base(e, block.get("mlp_up"), norm, up, f, h, explicit_t);
          if (const auto *mlp_gu_bypass = find_bypass_group(
                  layer_index, QwenBypassGroupKind::mlp_gate_up);
              mlp_gu_bypass != nullptr) {
            const std::array<detail::lora_rt::BypassPartDest, 2> dests{{
                {gate, f, 0U},
                {up, f, 0U},
            }};
            detail::lora_rt::emit_bypass_sibling_group(
                e, *mlp_gu_bypass, norm, dests, lora_bypass_tmp_,
                bf16_nax_bypass_ok_, silicon_tuning.gpu_cores, explicit_t);
          }
          swiglu(e, t, f, explicit_t);
        }
        if (down_w4a8 &&
            try_linear_w4a8_base(e, dw_v, product, delta, h, f, explicit_t)) {
          emit_bypass_for_weight(e, dw_v, product, delta, h, explicit_t);
        } else {
          linear(e, dw_v, product, delta, h, f, explicit_t);
        }
        if (io_sq_slot != nullptr)
          *io_sq_slot = std::nullopt;
        return;
      }
      if (input_is_f16) {
        const auto &gw_v = block.get("mlp_gate");
        const auto &uw_v = block.get("mlp_up");
        const auto *gw = mxfp4_nax_cache->find(gw_v.object_id);
        const auto *uw = mxfp4_nax_cache->find(uw_v.object_id);
        linear_mxfp4_nax_swiglu_direct_f16(e, *gw, *uw, f16_norm,
                                           f16_ones_scale, f16_scratch, true,
                                           silicon_tuning.gpu_cores,
                                           explicit_t,
                                           /*unit_scale=*/true);
        const std::string grp = mxfp4_nax_group(
            e, *gw, explicit_t, silicon_tuning.gpu_cores, true);
        e.record_weight(gw_v, f, h, "mxfp4_nax_m5_v1", "mxfp4_nax_m5_v1", {},
                        "linear_row_exact_portable_v1", grp, true, explicit_t);
        e.record_weight(uw_v, f, h, "mxfp4_nax_m5_v1", "mxfp4_nax_m5_v1", {},
                        "linear_row_exact_portable_v1", grp, true, explicit_t);
        linear_from_f16(e, block.get("mlp_down"), f16_scratch, f16_ones_scale,
                        delta, h, f, explicit_t);
        return;
      }
      bool swiglu_done = false;
      if (io_sq_slot != nullptr && io_sq_slot->has_value()) {
        const U32 sq_in = **io_sq_slot;
        *io_sq_slot = std::nullopt;
        if (try_linear_swiglu_norm_t1(e, block.get("mlp_gate"),
                                      block.get("mlp_up"),
                                      block.get("post_attention_norm"), hidden,
                                      product, f, h, sq_in)) {
          swiglu_done = true;
        } else {
          normalize(e, block.get("post_attention_norm"), hidden, norm);
        }
      }
      if (!swiglu_done &&
          !try_linear_swiglu(e, block.get("mlp_gate"), block.get("mlp_up"),
                             norm, product, f, h, explicit_t)) {
        linear_base(e, block.get("mlp_gate"), norm, gate, f, h, explicit_t);
        linear_base(e, block.get("mlp_up"), norm, up, f, h, explicit_t);
        if (const auto *mlp_gu_bypass =
                find_bypass_group(layer_index, QwenBypassGroupKind::mlp_gate_up);
            mlp_gu_bypass != nullptr) {
          const std::array<detail::lora_rt::BypassPartDest, 2> dests{{
              {gate, f, 0U},
              {up, f, 0U},
          }};
          detail::lora_rt::emit_bypass_sibling_group(
              e, *mlp_gu_bypass, norm, dests, lora_bypass_tmp_,
              bf16_nax_bypass_ok_, silicon_tuning.gpu_cores, explicit_t);
        }
        swiglu(e, t, f, explicit_t);
      }
      if (io_sq_slot != nullptr && out_sq_slot.has_value() &&
          try_linear_add_sq_t1(e, block.get("mlp_down"), product, hidden, h, f,
                               *out_sq_slot)) {
        *io_sq_slot = out_sq_slot;
      } else {
        linear(e, block.get("mlp_down"), product, delta, h, f, explicit_t);
        if (io_sq_slot != nullptr)
          *io_sq_slot = std::nullopt;
      }
      return;
    }

    const std::optional<U32> tok_w0 =
        explicit_t.has_value() ? std::optional<U32>{} : std::optional<U32>{0U};
    const bool trace_routes = route_trace.buffer && !explicit_t.has_value();

    const bool moe_has_bypass = block_has_any_bypass(
        block, {"moe_shared_mlp_gate", "moe_shared_mlp_up",
                "moe_shared_mlp_down", "moe_shared_gate", "moe_router"});

    if (!moe_has_bypass && e.profile.capacity == 1U && t == 1U &&
        !speculative_enabled && allow_tdom_t1() && t_domain_active() &&
        !moe_override) {
      const U32 shared_f_t1 = U32(config.shared_expert_intermediate_size);
      const auto &gw = block.get("moe_expert_gate");
      const auto &uw = block.get("moe_expert_up");
      const auto &dw = block.get("moe_expert_down");
      const auto &sgw = block.get("moe_shared_mlp_gate");
      const auto &suw = block.get("moe_shared_mlp_up");
      const auto &sdw = block.get("moe_shared_mlp_down");
      const auto &rw = block.get("moe_router");
      const auto &sgate_w = block.get("moe_shared_gate");
      if (!is_overridden(gw) && !is_overridden(uw) && !is_overridden(dw) &&
          !is_overridden(sgw) && !is_overridden(suw) && !is_overridden(sdw) &&
          !is_overridden(rw) && !is_overridden(sgate_w) &&
          can_moe9_mxfp4_t1(
              moe_nax_ws, gw, uw, dw, sgw, suw, sdw,
              mxfp4_nax_active() ? mxfp4_nax_cache.get() : nullptr, t, h, f,
              shared_f_t1, 8u, 256u)) {
        bool fused_sgate = false;
        bool router_done = false;
        const bool has_sq_in = (io_sq_slot != nullptr && io_sq_slot->has_value());
        const U32 sq_in = has_sq_in ? **io_sq_slot : 0U;
        if (has_sq_in)
          *io_sq_slot = std::nullopt;
        if (io_sq_slot != nullptr && moe_tg_done_buf.buffer != nullptr) {
          router_done = emit_moe9_router_fused_top8_t1(
              e, rw, &sgate_w,
              affine_nax_active() ? affine_nax_cache.get() : nullptr, has_sq_in,
              has_sq_in ? hidden : norm,
              has_sq_in ? direct(block.get("post_attention_norm"), h) : Ref{},
              has_sq_in ? rms_sq_buf : Ref{}, has_sq_in ? norm : Ref{}, zero,
              router, route_ids, route_scores, shared_gate, moe_tg_done_buf, h,
              sq_in, layer_index, float(config.rms_norm_eps),
              normalization_weight_offset, explicit_t, &fused_sgate);
          if (!router_done && has_sq_in) {
            normalize(e, block.get("post_attention_norm"), hidden, norm);
          }
        } else if (has_sq_in) {
          normalize(e, block.get("post_attention_norm"), hidden, norm);
        }
        if (!router_done &&
            !emit_moe9_router_top8_t1(
                e, rw, &sgate_w,
                affine_nax_active() ? affine_nax_cache.get() : nullptr, norm,
                zero, router, route_ids, route_scores, shared_gate, h,
                explicit_t, &fused_sgate)) {
          linear(e, rw, norm, router, 256, h, explicit_t);
          auto &route =
              e.emit("qwen_forward", "omlx_m5_forward_router_top8_logits",
                     {.width = 32, .height = 1}, {.width = 32},
                     {router, route_ids, route_scores});
          e.dynamic(route, 3, words({1U, 256U}), tok_w0);
        }
        if (trace_routes) {
          auto &trace =
              e.emit("qwen_forward", "omlx_m5_forward_capture_routes",
                     {.width = 8}, {.width = 128}, {route_ids, route_trace});
          e.dynamic(trace, 2,
                    words({1U, 8, U32(config.num_hidden_layers), layer_index}),
                    0);
          auto &logits_trace = e.emit("qwen_forward", "omlx_m5_forward_capture",
                                      {.width = 256}, {.width = 128},
                                      {router, router_logits_trace});
          e.dynamic(logits_trace, 2,
                    words({1U, 256, U32(config.num_hidden_layers), layer_index}),
                    0);
          auto &scores_trace = e.emit("qwen_forward", "omlx_m5_forward_capture",
                                      {.width = 8}, {.width = 128},
                                      {route_scores, router_scores_trace});
          e.dynamic(scores_trace, 2,
                    words({1U, 8, U32(config.num_hidden_layers), layer_index}),
                    0);
        }
        if (!fused_sgate) {
          linear(e, sgate_w, norm, shared_gate, 1, h, explicit_t);
        }
        if (io_sq_slot != nullptr && out_sq_slot.has_value() &&
            emit_moe9_mxfp4_add_sq_t1(
                e, moe_nax_ws, gw, uw, dw, sgw, suw, sdw,
                mxfp4_nax_active() ? mxfp4_nax_cache.get() : nullptr, norm,
                route_ids, route_scores, shared_gate, hidden, rms_sq_buf, h, f,
                *out_sq_slot, explicit_t)) {
          *io_sq_slot = out_sq_slot;
        } else {
          (void)emit_moe9_mxfp4_t1(
              e, moe_nax_ws, gw, uw, dw, sgw, suw, sdw,
              mxfp4_nax_active() ? mxfp4_nax_cache.get() : nullptr, norm,
              route_ids, route_scores, shared_gate, delta, h, f, explicit_t);
          if (io_sq_slot != nullptr)
            *io_sq_slot = std::nullopt;
        }
        return;
      }
    }

    if (!moe_has_bypass && allow_moe9_bt && explicit_t.has_value() && t >= 2U &&
        t <= 8U && !speculative_enabled && t_domain_active() &&
        !moe_override) {
      const U32 shared_f_bt = U32(config.shared_expert_intermediate_size);
      const auto &gw = block.get("moe_expert_gate");
      const auto &uw = block.get("moe_expert_up");
      const auto &dw = block.get("moe_expert_down");
      const auto &sgw = block.get("moe_shared_mlp_gate");
      const auto &suw = block.get("moe_shared_mlp_up");
      const auto &sdw = block.get("moe_shared_mlp_down");
      const auto &rw = block.get("moe_router");
      const auto &sgate_w = block.get("moe_shared_gate");
      if (!is_overridden(gw) && !is_overridden(uw) && !is_overridden(dw) &&
          !is_overridden(sgw) && !is_overridden(suw) && !is_overridden(sdw) &&
          !is_overridden(rw) && !is_overridden(sgate_w) &&
          can_moe9_mxfp4_bt(
              moe_nax_ws, gw, uw, dw, sgw, suw, sdw,
              mxfp4_nax_active() ? mxfp4_nax_cache.get() : nullptr, t, h, f,
              shared_f_bt, 8u, 256u)) {
        bool fused_sgate = false;
        bool router_done = false;
        const bool has_sq_in = (io_sq_slot != nullptr && io_sq_slot->has_value());
        const U32 sq_in = has_sq_in ? **io_sq_slot : 0U;
        if (has_sq_in)
          *io_sq_slot = std::nullopt;
        // Self-resetting moe_slot disjoint from moe9_t1's [0..blocks+3].
        const U32 bt_moe_slot = U32(blocks.size()) + 8U;
        if (moe_tg_done_buf.buffer != nullptr &&
            can_moe9_router_fused_bt(
                rw, affine_nax_active() ? affine_nax_cache.get() : nullptr, t,
                h, 256u)) {
          router_done = emit_moe9_router_fused_top8_bt(
              e, rw, &sgate_w,
              affine_nax_active() ? affine_nax_cache.get() : nullptr, has_sq_in,
              has_sq_in ? hidden : norm,
              has_sq_in ? direct(block.get("post_attention_norm"), h) : Ref{},
              has_sq_in ? rms_sq_buf : Ref{}, has_sq_in ? norm : Ref{}, zero,
              router, route_ids, route_scores, shared_gate, moe_tg_done_buf, t,
              h, sq_in, bt_moe_slot, float(config.rms_norm_eps),
              normalization_weight_offset, explicit_t, &fused_sgate);
          if (!router_done && has_sq_in) {
            normalize(e, block.get("post_attention_norm"), hidden, norm,
                      explicit_t);
          }
        } else if (has_sq_in) {
          normalize(e, block.get("post_attention_norm"), hidden, norm,
                    explicit_t);
        }
        if (!router_done) {
          linear(e, rw, norm, router, 256, h, explicit_t);
          auto &route =
              e.emit("qwen_forward", "omlx_m5_forward_router_top8_logits",
                     {.width = 32, .height = t}, {.width = 32},
                     {router, route_ids, route_scores});
          e.dynamic(route, 3, words({t, 256U}), tok_w0);
        }
        if (!fused_sgate) {
          linear(e, sgate_w, norm, shared_gate, 1, h, explicit_t);
        }
        if (io_sq_slot != nullptr && out_sq_slot.has_value() &&
            emit_moe9_mxfp4_add_sq_bt(
                e, moe_nax_ws, gw, uw, dw, sgw, suw, sdw,
                mxfp4_nax_active() ? mxfp4_nax_cache.get() : nullptr, norm,
                route_ids, route_scores, shared_gate, hidden, rms_sq_buf, t, h,
                f, *out_sq_slot, explicit_t)) {
          *io_sq_slot = out_sq_slot;
        } else {
          (void)emit_moe9_mxfp4_bt(
              e, moe_nax_ws, gw, uw, dw, sgw, suw, sdw,
              mxfp4_nax_active() ? mxfp4_nax_cache.get() : nullptr, norm,
              route_ids, route_scores, shared_gate, delta, t, h, f, explicit_t);
          if (io_sq_slot != nullptr)
            *io_sq_slot = std::nullopt;
        }
        return;
      }
    }

    if (io_sq_slot != nullptr && io_sq_slot->has_value()) {
      normalize(e, block.get("post_attention_norm"), hidden, norm, explicit_t);
      *io_sq_slot = std::nullopt;
    }
    linear(e, block.get("moe_router"), norm, router, 256, h, explicit_t);
    const bool fuse_router_nax =
        !trace_routes && can_moe_nax_for(block, t, explicit_t);
    if (!fuse_router_nax) {
      auto &route = e.emit("qwen_forward", "omlx_m5_forward_router_top8_logits",
                           {.width = 32, .height = t}, {.width = 32},
                           {router, route_ids, route_scores});
      e.dynamic(route, 3, words({t, 256}), tok_w0);
    }
    if (trace_routes) {
      auto &trace = e.emit("qwen_forward", "omlx_m5_forward_capture_routes",
                           {.width = U64(t) * 8}, {.width = 128},
                           {route_ids, route_trace});
      e.dynamic(trace, 2,
                words({t, 8, U32(config.num_hidden_layers), layer_index}), 0);
      auto &logits_trace = e.emit("qwen_forward", "omlx_m5_forward_capture",
                                  {.width = U64(t) * 256}, {.width = 128},
                                  {router, router_logits_trace});
      e.dynamic(logits_trace, 2,
                words({t, 256, U32(config.num_hidden_layers), layer_index}), 0);
      auto &scores_trace = e.emit("qwen_forward", "omlx_m5_forward_capture",
                                  {.width = U64(t) * 8}, {.width = 128},
                                  {route_scores, router_scores_trace});
      e.dynamic(scores_trace, 2,
                words({t, 8, U32(config.num_hidden_layers), layer_index}), 0);
    }
    bool wrote_expert_output = false;
    if (!try_fused_moe_ffn(e, block, wrote_expert_output,
                           fuse_router_nax ? router : Ref{}, explicit_t,
                           verify, input_is_a8, input_is_a4)) {
      expert_linear(e, block.get("moe_expert_gate"), norm, gate, f, h, false,
                    explicit_t);
      expert_linear(e, block.get("moe_expert_up"), norm, up, f, h, false,
                    explicit_t);
      swiglu(e, t * 8, f, explicit_t);
      if (block.get("moe_expert_down").has_fwht64()) {
        detail::forward::emit_fwht64_bf16(e, product, product, f, explicit_t,
                                          8U);
      }
      expert_linear(e, block.get("moe_expert_down"), product, expert_output, h,
                    f, true, explicit_t);
      wrote_expert_output = true;
    }
    U32 shared_f = U32(config.shared_expert_intermediate_size);
    const auto &sdw_v = block.get("moe_shared_mlp_down");
    const bool shared_down_w4a4 = sdw_v.has_fwht64() || w4a4_down_proj_;
    const bool shared_down_w4a8 = sdw_v.has_fwht64() || w4a8_down_proj_;
    const bool shared_mode_b_a4 =
        input_is_a4 && shared_down_w4a4 && sdw_v.has_fwht64() &&
        can_w4a4_weight(sdw_v, h, shared_f) &&
        try_linear_w4a4_swiglu_fwht64_a4(
            e, block.get("moe_shared_mlp_gate"), block.get("moe_shared_mlp_up"),
            norm, product, shared_f, h,
            has_bypass_lora(sdw_v) || inspect_tensors_workspace, explicit_t);
    const bool shared_mode_b =
        shared_mode_b_a4 ||
        (input_is_a8 && shared_down_w4a8 && sdw_v.has_fwht64() &&
         can_w4a8_weight(sdw_v, h, shared_f) &&
         try_linear_w4a8_swiglu_fwht64_a8(
             e, block.get("moe_shared_mlp_gate"), block.get("moe_shared_mlp_up"),
             norm, product, shared_f, h,
             has_bypass_lora(sdw_v) || inspect_tensors_workspace, explicit_t));
    const bool shared_swiglu_done =
        shared_mode_b ||
        (input_is_a4 &&
         try_linear_w4a4_swiglu(e, block.get("moe_shared_mlp_gate"),
                                block.get("moe_shared_mlp_up"), norm, product,
                                shared_f, h, explicit_t)) ||
        (input_is_a8 &&
         try_linear_w4a8_swiglu(e, block.get("moe_shared_mlp_gate"),
                                block.get("moe_shared_mlp_up"), norm, product,
                                shared_f, h, explicit_t)) ||
        (!input_is_a4 && !input_is_a8 &&
         try_linear_swiglu(e, block.get("moe_shared_mlp_gate"),
                           block.get("moe_shared_mlp_up"), norm, product,
                           shared_f, h, explicit_t));
    if (!shared_swiglu_done) {
      if (!(input_is_a4 &&
            try_linear_w4a4_base(e, block.get("moe_shared_mlp_gate"), norm,
                                 gate, shared_f, h, explicit_t)) &&
          !(input_is_a8 &&
            try_linear_w4a8_base(e, block.get("moe_shared_mlp_gate"), norm,
                                 gate, shared_f, h, explicit_t))) {
        linear_base(e, block.get("moe_shared_mlp_gate"), norm, gate, shared_f,
                    h, explicit_t);
      }
      if (!(input_is_a4 &&
            try_linear_w4a4_base(e, block.get("moe_shared_mlp_up"), norm, up,
                                 shared_f, h, explicit_t)) &&
          !(input_is_a8 &&
            try_linear_w4a8_base(e, block.get("moe_shared_mlp_up"), norm, up,
                                 shared_f, h, explicit_t))) {
        linear_base(e, block.get("moe_shared_mlp_up"), norm, up, shared_f, h,
                    explicit_t);
      }
      if (const auto *sgu_bypass = find_bypass_group(
              layer_index, QwenBypassGroupKind::moe_shared_gate_up);
          sgu_bypass != nullptr) {
        const std::array<detail::lora_rt::BypassPartDest, 2> dests{{
            {gate, shared_f, 0U},
            {up, shared_f, 0U},
        }};
        detail::lora_rt::emit_bypass_sibling_group(
            e, *sgu_bypass, norm, dests, lora_bypass_tmp_, bf16_nax_bypass_ok_,
            silicon_tuning.gpu_cores, explicit_t);
      }
      swiglu(e, t, shared_f, explicit_t);
    }
    if (input_is_a4 && shared_down_w4a4 &&
        try_linear_w4a4_base(e, sdw_v, product, shared, h, shared_f,
                             explicit_t)) {
      emit_bypass_for_weight(e, sdw_v, product, shared, h, explicit_t);
    } else if (input_is_a8 && shared_down_w4a8 &&
               try_linear_w4a8_base(e, sdw_v, product, shared, h, shared_f,
                                    explicit_t)) {
      emit_bypass_for_weight(e, sdw_v, product, shared, h, explicit_t);
    } else {
      linear(e, sdw_v, product, shared, h, shared_f, explicit_t);
    }
    linear(e, block.get("moe_shared_gate"), norm, shared_gate, 1, h, explicit_t);
    if (wrote_expert_output) {
      auto &combine =
          e.emit("mlp", "omlx_m5_moe_topk_shared_combine",
                 {.width = U64(t) * h}, {.width = 128},
                 {expert_output, shared, shared_gate, delta});
      e.dynamic(combine, 4, words({t, 8, h, 0}), tok_w0);
    } else {
      auto &combine =
          e.emit("mlp", "omlx_m5_moe_shared_combine", {.width = U64(t) * h},
                 {.width = 128}, {routed, shared, shared_gate, delta});
      e.dynamic(combine, 4, words({t, 8, h, 0}), tok_w0);
    }
  }

  std::string plan_key_suffix() const {
    Fnv1a64 wsig;
    wsig.weight(embedding);
    wsig.weight(final_norm);
    wsig.weight(head);
    const bool tied_head =
        !embedding.planes.empty() && !head.planes.empty() &&
        embedding.planes.front().buffer == head.planes.front().buffer &&
        embedding.planes.front().offset == head.planes.front().offset;
    wsig.u64(tied_head ? 1ULL : 0ULL);
    if (merged_lora_ && !merged_lora_->fingerprint.empty()) {
      wsig.str("merged_lora");
      wsig.str(merged_lora_->fingerprint);
    }
    if (bypass_lora_ && !bypass_lora_->fingerprint.empty()) {
      wsig.str("bypass_lora");
      wsig.str(bypass_lora_->fingerprint);
    }
    std::string layer_sig;
    layer_sig.reserve(blocks.size());
    for (const auto &block : blocks) {
      layer_sig.push_back(block.type == LayerType::full_attention ? 'a' : 'g');
      for (const auto &[role, w] : block.operands) {
        wsig.str(role);
        wsig.weight(w);
      }
    }
    const U32 rotary_dim =
        U32(std::llround(config.head_dim * config.partial_rotary_factor));
    std::string taps_str = feature_layers.empty() ? "none" : "";
    for (std::size_t i = 0; i < feature_layers.size(); ++i) {
      if (i)
        taps_str.push_back('-');
      taps_str += std::to_string(feature_layers[i]);
    }
    std::string key =
        ".L" + std::to_string(blocks.size()) + "-" + layer_sig + ".h" +
        std::to_string(h) + ".f" + std::to_string(f) + ".v" +
        std::to_string(v) + ".ah" + std::to_string(config.num_attention_heads) +
        ".kvh" + std::to_string(config.num_key_value_heads) + ".hd" +
        std::to_string(config.head_dim) + ".rd" + std::to_string(rotary_dim) +
        ".gkh" + std::to_string(config.linear_num_key_heads) + ".gvh" +
        std::to_string(config.linear_num_value_heads) + ".gkd" +
        std::to_string(config.linear_key_head_dim) + ".gvd" +
        std::to_string(config.linear_value_head_dim) + ".ck" +
        std::to_string(config.linear_conv_kernel_dim) + ".exp" +
        std::to_string(config.num_experts) + "x" +
        std::to_string(config.num_experts_per_token) + ".sf" +
        std::to_string(config.shared_expert_intermediate_size) + ".eps" +
        std::to_string(word(float(config.rms_norm_eps))) + ".th" +
        std::to_string(word(float(config.rope_theta))) + ".no" +
        std::to_string(word(normalization_weight_offset)) + ".ctx" +
        std::to_string(max_context) + ".kvp" +
        std::to_string(page_pool ? page_pool->physical_pages() : 0U) + ".wsr" +
        std::to_string(workspace_rows_) +
        ".ring" + std::to_string(feature_ring_rows_) + ".kv-" +
        std::string(kv_storage_format_name(kv_format_)) + ".gc" +
        std::to_string(silicon_tuning.gpu_cores) + ".lp" +
        (linear_policy == LinearNumericPolicy::RowExactPortableV1 ? "1" : "0") +
        ".sc" +
        (sampling_contract == SamplingContract::PositionCoupledV1 ? "1" : "0") +
        ".ap" +
        (attention_policy == omlx::m5::AttentionNumericPolicy::RowExactV1 ? "1"
                                                                          : "0") +
        (strict_two_pass_attn_ ? ".s2p1" : "") +
        (flash_nax_prefill_ ? ".fnax1" : "") +
        (gdn_chunk_nax_enabled_
             ? ".gdnck2m" + std::to_string(gdn_chunk_nax_min_tokens())
             : "") +
        ((mxfp4_nax_active() || affine_nax_active() || bf16_nax_enabled)
             ? ".naxwp" + std::to_string(nax_wide_prefill_min_tokens()) +
                   (std::getenv("OMLX_M5_WIDE_PREFILL_TILE")
                        ? ".wpt" + std::string(std::getenv("OMLX_M5_WIDE_PREFILL_TILE"))
                        : "")
             : "") +
        (pretiled_qmv_t1_enabled()
             ? ".ptqmv4"
             : ".ptqmv0") +
        (moe_nax_ws.active ? ".mnax" + std::to_string(moe_nax_min_tokens()) +
                                 ".mw" +
                                 std::to_string(moe_nax_wide_min_tokens()) +
                                 ".mb64_" +
                                 std::to_string(moe_nax_bm64_min_tokens()) +
                                 (moe9_t1_enabled() ? ".m9t3" : ".m9t0") +
                                 (moe9_bt_enabled() ? ".m9b1" : ".m9b0") +
                                 (std::getenv("OMLX_M5_MOE_NAX_MIN_T")
                                      ? ".mne1"
                                      : "")
                           : "") +
        (detail::forward::env_flag_enabled("OMLX_M5_DISABLE_GDN_SPEC_FUSION")
             ? ".gdnf2.td"
             : ".gdnf3.td") +
        (t_domain_active() ? "1" : "0") + ".be" +
        (batch_enabled ? "1" : "0") + ".m4sr" +
        (m4_short_routes ? "1" : "0") + ".spec" +
        (speculative_enabled ? "1" : "0") + ".ir" +
        (inspect_routes_enabled ? "1" : "0") + ".itw" +
        (inspect_tensors_workspace ? "1" : "0") + ".sa" +
        (split_attention_enabled ? "1" : "0") + ".taps-" + taps_str + ".ws" +
        wsig.hex();
    if (diagnostic_override) {
      key += ".lin-ab." + diagnostic_override->object + "." +
             diagnostic_override->format + ".n" +
             std::to_string(diagnostic_override->n) + ".k" +
             std::to_string(diagnostic_override->k) + ".t" +
             std::to_string(diagnostic_override->capacity) + "." +
             diagnostic_override->choice;
    }
    if (moe_override) {
      key += ".moe-ab." + moe_override->object + "." + moe_override->format +
             "." + moe_override->layout + ".n" +
             std::to_string(moe_override->n) + ".k" +
             std::to_string(moe_override->k) + ".t" +
             std::to_string(moe_override->capacity) + "." +
             moe_override->choice;
    }
    if (w4a8_nax_enabled_) {
      key += ".plm1.w4m" + std::to_string(w4a8_min_tokens_);
      if (w4a8_down_proj_) {
        key += ".w4dp1";
      }
    }
    if (w4a4_nax_enabled_) {
      key += ".plm2.w4a4m" + std::to_string(w4a4_min_tokens_);
      if (w4a4_keep_qk_a8_) {
        key += ".qka8";
      }
      if (!w4a4_down_proj_) {
        key += ".d16";
      }
    }
    if (sage2_enabled_) {
      key += ".pam1";
      if (resolve_sage2_pv_mode() == Sage2PvMode::Mxfp8) {
        key += ".s2mx8";
      }
    }
    if (minference_enabled_) {
      key += ".minf1";
    }
    return key;
  }

  Profile compile(U32 t, bool verify, bool split = false,
                  U32 context_envelope = 0, bool defer_sample = false,
                  Ref custom_sampled = {}, Ref custom_attn_pos = {},
                  Ref feed_next_token = {}, std::string key_tag = {},
                  bool trunk_only = false, Ref custom_token_ids = {},
                  std::optional<ChunkAttentionRoute> attn_route_override =
                      std::nullopt) {
    Encoder e(
        context, libraries, t,
        (verify ? "qwen.verify" : "qwen.forward") +
            std::string(linear_policy == LinearNumericPolicy::RowExactPortableV1
                            ? ".linear-row-exact-v1"
                            : "") +
            std::string(sampling_contract == SamplingContract::PositionCoupledV1
                            ? ".position-coupled-v1"
                            : "") +
            std::string(
                attention_policy == omlx::m5::AttentionNumericPolicy::RowExactV1
                    ? ".attention-row-exact-v1"
                    : "") +
            std::string(defer_sample ? ".batch" : "") +
            std::string(trunk_only ? ".trunk" : "") +
            std::string(split ? ".split.l" + std::to_string(context_envelope)
                              : (context_envelope
                                     ? ".nosplit.l" +
                                           std::to_string(context_envelope)
                                     : "")) +
            (attn_route_override.has_value()
                 ? (".ar" + std::to_string(U32(*attn_route_override)))
                 : std::string{}) +
            key_tag + plan_key_suffix());
    e.profile.context_envelope = context_envelope;
    e.profile.sampling_contract_id =
        std::string(sampling_contract_name(sampling_contract));
    if (linear_policy == LinearNumericPolicy::RowExactPortableV1) {
      e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    }
    e.begin_stage("embedding");
    embedding_lookup(e, embedding,
                     custom_token_ids.buffer ? custom_token_ids : token_ids,
                     hidden, zero, v, h);
    const bool extract_last_row_head =
        !verify && !defer_sample && !inspect_tensors_workspace && t > 8U;
    const bool nax_head =
        !has_bypass_lora(head) &&
        use_mxfp4_nax_for_weight(head, extract_last_row_head ? 1U : t) &&
        [&] {
          const auto *tw = mxfp4_nax_cache->find(head.object_id);
          return tw && tw->n == v && tw->k == h;
        }();
    const bool layer_w4a4 = use_w4a4_prefill(e, verify);
    const bool layer_w4a8 = !layer_w4a4 && use_w4a8_prefill(e, verify);
    if (layer_w4a4) {
      e.profile.used_w4a4 = true;
    }
    if (layer_w4a8) {
      e.profile.used_w4a8 = true;
    }
    const bool allow_t1_fused_norms =
        (t == 1U && !verify && !defer_sample && !trunk_only &&
         !speculative_enabled && allow_tdom_t1() && t_domain_active() &&
         pretiled_qmv_t1_enabled() && !inspect_tensors_workspace &&
         rms_sq_buf.buffer != nullptr && moe_tg_done_buf.buffer != nullptr &&
         h < 65536U);
    const bool affine_xs_eligible =
        (!layer_w4a4 && !layer_w4a8 && !allow_t1_fused_norms && t > 1U &&
         affine_nax_active() && affine_nax_xs.buffer != nullptr &&
         (h % 64U) == 0U);
    const U32 num_sq_words = U32(blocks.size()) * 2U + 4U;
    const U32 num_moe_words =
        config.num_experts ? (U32(blocks.size()) + 4U) : 0U;
    std::optional<U32> pending_sq_slot;
    std::optional<Encoder::GroupSumKey> carried_affine_xs;
    std::optional<Encoder::A8QuantKey> carried_a8_quant;
    std::optional<Encoder::A4QuantKey> carried_a4_quant;
    auto restore_carried_xs = [&]() {
      if (carried_affine_xs.has_value()) {
        e.set_group_sum(*carried_affine_xs);
        carried_affine_xs.reset();
      }
      if (carried_a8_quant.has_value()) {
        e.set_a8_quant(*carried_a8_quant);
        carried_a8_quant.reset();
      }
      if (carried_a4_quant.has_value()) {
        e.set_a4_quant(*carried_a4_quant);
        carried_a4_quant.reset();
      }
    };
    auto block_input_needs_bf16 = [&](const Block &blk) -> bool {
      if (inspect_tensors_workspace)
        return true;
      if (blk.type == LayerType::full_attention) {
        return block_has_any_bypass(
                   blk, {"attention_q", "attention_k", "attention_v"}) ||
               !can_w4a8_weight(blk.get("attention_q"), 2 * q, h) ||
               !can_w4a8_weight(blk.get("attention_k"), kv, h) ||
               !can_w4a8_weight(blk.get("attention_v"), kv, h);
      }
      const U32 hv = U32(config.linear_num_value_heads);
      return block_has_any_bypass(blk, {"gdn_qkv", "gdn_z", "gdn_a", "gdn_b"}) ||
             !can_w4a8_weight(blk.get("gdn_qkv"), conv, h) ||
             !can_w4a8_weight(blk.get("gdn_z"), gv, h) ||
             !can_w4a8_weight(blk.get("gdn_a"), hv, h) ||
             !can_w4a8_weight(blk.get("gdn_b"), hv, h);
    };
    auto block_mlp_needs_bf16 = [&](const Block &blk) -> bool {
      if (inspect_tensors_workspace || config.num_experts > 0)
        return true;
      return block_has_any_bypass(blk, {"mlp_gate", "mlp_up"}) ||
             !can_w4a8_weight(blk.get("mlp_gate"), f, h) ||
             !can_w4a8_weight(blk.get("mlp_up"), f, h);
    };
    auto block_input_needs_bf16_a4 = [&](const Block &blk) -> bool {
      if (inspect_tensors_workspace)
        return true;
      if (blk.type == LayerType::full_attention) {
        return w4a4_keep_qk_a8_ ||
               block_has_any_bypass(
                   blk, {"attention_q", "attention_k", "attention_v"}) ||
               !can_w4a4_weight(blk.get("attention_q"), 2 * q, h) ||
               !can_w4a4_weight(blk.get("attention_k"), kv, h) ||
               !can_w4a4_weight(blk.get("attention_v"), kv, h);
      }
      const U32 hv = U32(config.linear_num_value_heads);
      return block_has_any_bypass(blk, {"gdn_qkv", "gdn_z", "gdn_a", "gdn_b"}) ||
             !can_w4a4_weight(blk.get("gdn_qkv"), conv, h) ||
             !can_w4a4_weight(blk.get("gdn_z"), gv, h) ||
             !can_w4a4_weight(blk.get("gdn_a"), hv, h) ||
             !can_w4a4_weight(blk.get("gdn_b"), hv, h);
    };
    auto block_mlp_needs_bf16_a4 = [&](const Block &blk) -> bool {
      if (inspect_tensors_workspace || config.num_experts > 0)
        return true;
      return block_has_any_bypass(blk, {"mlp_gate", "mlp_up"}) ||
             !can_w4a4_weight(blk.get("mlp_gate"), f, h) ||
             !can_w4a4_weight(blk.get("mlp_up"), f, h);
    };
    auto make_norm_xs_key = [&]() -> Encoder::GroupSumKey {
      return Encoder::GroupSumKey{affine_nax_xs.buffer,
                                  affine_nax_xs.offset,
                                  norm.buffer,
                                  norm.offset,
                                  h,
                                  64U,
                                  t,
                                  true,
                                  2U};
    };
    U32 layer_index = 0;
    for (const auto &block : blocks) {
      const bool in_f16 =
          !layer_w4a4 && !layer_w4a8 && block_input_nax(block, t);
      const bool mlp_f16 =
          !layer_w4a4 && !layer_w4a8 && block_mlp_input_nax(block, t);
      const bool in_affine_xs =
          (!layer_w4a4 && !layer_w4a8 && !in_f16 && affine_xs_eligible);
      const bool mlp_affine_xs =
          (!layer_w4a4 && !layer_w4a8 && !mlp_f16 && affine_xs_eligible);
      const bool layer_fuse = allow_t1_fused_norms && !layer_w4a4 &&
                              !layer_w4a8 && !in_f16 && !mlp_f16;
      e.begin_stage(
          "layer." + std::to_string(layer_index) +
          (block.type == LayerType::full_attention ? ".attention" : ".gdn"));
      restore_carried_xs();
      if (layer_index == 0) {
        if (layer_w4a4) {
          emit_rmsnorm_a4_bm16_bk128(e, hidden,
                                     direct(block.get("input_norm"), h), norm,
                                     w4a4_codes, w4a4_scales, h,
                                     float(config.rms_norm_eps),
                                     normalization_weight_offset,
                                     block_input_needs_bf16_a4(block));
          e.set_a4_quant(make_a4_key(e, norm, h));
        } else if (layer_w4a8) {
          emit_rmsnorm_a8_g64(e, hidden, direct(block.get("input_norm"), h),
                              norm, a8_qa, a8_sa, a8_ra, h,
                              float(config.rms_norm_eps),
                              normalization_weight_offset,
                              block_input_needs_bf16(block));
          e.set_a8_quant(make_a8_key(e, norm, h));
        } else if (in_f16)
          normalize_f16(e, block.get("input_norm"), hidden, f16_norm);
        else if (allow_t1_fused_norms)
          normalize_init_zero_t1(e, block.get("input_norm"), hidden, norm,
                                 num_sq_words, num_moe_words);
        else if (in_affine_xs)
          emit_affine_rmsnorm_xsum(e, hidden, direct(block.get("input_norm"), h),
                                   norm, affine_nax_xs, h,
                                   float(config.rms_norm_eps),
                                   normalization_weight_offset);
        else
          normalize(e, block.get("input_norm"), hidden, norm);
      } else if (!layer_fuse && pending_sq_slot.has_value()) {
        if (layer_w4a4) {
          emit_rmsnorm_a4_bm16_bk128(e, hidden,
                                     direct(block.get("input_norm"), h), norm,
                                     w4a4_codes, w4a4_scales, h,
                                     float(config.rms_norm_eps),
                                     normalization_weight_offset,
                                     block_input_needs_bf16_a4(block));
          e.set_a4_quant(make_a4_key(e, norm, h));
        } else if (layer_w4a8) {
          emit_rmsnorm_a8_g64(e, hidden, direct(block.get("input_norm"), h),
                              norm, a8_qa, a8_sa, a8_ra, h,
                              float(config.rms_norm_eps),
                              normalization_weight_offset,
                              block_input_needs_bf16(block));
          e.set_a8_quant(make_a8_key(e, norm, h));
        } else if (in_f16)
          normalize_f16(e, block.get("input_norm"), hidden, f16_norm);
        else if (in_affine_xs)
          emit_affine_rmsnorm_xsum(e, hidden, direct(block.get("input_norm"), h),
                                   norm, affine_nax_xs, h,
                                   float(config.rms_norm_eps),
                                   normalization_weight_offset);
        else
          normalize(e, block.get("input_norm"), hidden, norm);
        pending_sq_slot.reset();
      }
      if (block.type == LayerType::full_attention)
        attention(e, block, split, context_envelope, in_f16, custom_attn_pos,
                  layer_fuse ? &pending_sq_slot : nullptr,
                  layer_fuse ? std::optional<U32>{2U * layer_index}
                             : std::optional<U32>{},
                  layer_w4a8, attn_route_override, layer_w4a4);
      else
        gdn(e, block, verify, in_f16, layer_fuse ? &pending_sq_slot : nullptr,
            layer_fuse ? std::optional<U32>{2U * layer_index}
                       : std::optional<U32>{},
            layer_w4a8, layer_w4a4);
      e.begin_stage("layer." + std::to_string(layer_index) + ".mlp");
      if (!pending_sq_slot.has_value()) {
        if (layer_w4a4) {
          emit_add_rmsnorm_a4_bm16_bk128(
              e, hidden, delta, direct(block.get("post_attention_norm"), h),
              norm, w4a4_codes, w4a4_scales, h, float(config.rms_norm_eps),
              normalization_weight_offset, block_mlp_needs_bf16_a4(block));
          e.set_a4_quant(make_a4_key(e, norm, h));
        } else if (layer_w4a8) {
          emit_add_rmsnorm_a8_g64(
              e, hidden, delta, direct(block.get("post_attention_norm"), h),
              norm, a8_qa, a8_sa, a8_ra, h, float(config.rms_norm_eps),
              normalization_weight_offset, block_mlp_needs_bf16(block));
          e.set_a8_quant(make_a8_key(e, norm, h));
        } else if (mlp_f16)
          add_normalize_f16(e, block.get("post_attention_norm"), hidden, delta,
                            f16_norm);
        else if (mlp_affine_xs)
          emit_affine_add_rmsnorm_xsum(
              e, hidden, delta, direct(block.get("post_attention_norm"), h),
              norm, affine_nax_xs, h, float(config.rms_norm_eps),
              normalization_weight_offset);
        else
          add_normalize(e, block.get("post_attention_norm"), hidden, delta,
                        norm);
      }
      mlp(e, block, layer_index, mlp_f16, {},
          layer_fuse ? &pending_sq_slot : nullptr,
          layer_fuse ? std::optional<U32>{2U * layer_index + 1U}
                     : std::optional<U32>{},
          false, layer_w4a8, verify, layer_w4a4);
      if (!pending_sq_slot.has_value()) {
        if (layer_index + 1 < blocks.size()) {
          if (layer_w4a4) {
            emit_add_rmsnorm_a4_bm16_bk128(
                e, hidden, delta,
                direct(blocks[layer_index + 1].get("input_norm"), h), norm,
                w4a4_codes, w4a4_scales, h, float(config.rms_norm_eps),
                normalization_weight_offset,
                block_input_needs_bf16_a4(blocks[layer_index + 1]));
            carried_a4_quant = make_a4_key(e, norm, h);
          } else if (layer_w4a8) {
            emit_add_rmsnorm_a8_g64(
                e, hidden, delta,
                direct(blocks[layer_index + 1].get("input_norm"), h), norm,
                a8_qa, a8_sa, a8_ra, h, float(config.rms_norm_eps),
                normalization_weight_offset,
                block_input_needs_bf16(blocks[layer_index + 1]));
            carried_a8_quant = make_a8_key(e, norm, h);
          } else {
            const bool next_in_f16 = block_input_nax(blocks[layer_index + 1], t);
            if (next_in_f16)
              add_normalize_f16(e, blocks[layer_index + 1].get("input_norm"),
                                hidden, delta, f16_norm);
            else if (affine_xs_eligible) {
              emit_affine_add_rmsnorm_xsum(
                  e, hidden, delta,
                  direct(blocks[layer_index + 1].get("input_norm"), h), norm,
                  affine_nax_xs, h, float(config.rms_norm_eps),
                  normalization_weight_offset);
              carried_affine_xs = make_norm_xs_key();
            } else
              add_normalize(e, blocks[layer_index + 1].get("input_norm"), hidden,
                            delta, norm);
          }
        } else if (!extract_last_row_head) {
          if (nax_head)
            add_normalize_f16(e, final_norm, hidden, delta, f16_norm);
          else if (affine_xs_eligible) {
            emit_affine_add_rmsnorm_xsum(
                e, hidden, delta, direct(final_norm, h), norm, affine_nax_xs, h,
                float(config.rms_norm_eps), normalization_weight_offset);
            carried_affine_xs = make_norm_xs_key();
          } else
            add_normalize(e, final_norm, hidden, delta, norm);
        } else {
          add(e);
        }
      }
      for (U32 tap = 0; tap < feature_layers.size(); ++tap) {
        if (feature_layers[tap] != layer_index)
          continue;
        auto &capture =
            e.emit("qwen_forward", "omlx_m5_forward_capture",
                   {.width = U64(t) * h}, {.width = 128}, {hidden, features});
        e.dynamic(capture, 2, words({t, h, U32(feature_layers.size()), tap}),
                  0);
      }
      ++layer_index;
    }
    if (trunk_only) {
      if (feature_history.buffer && !verify) {
        e.begin_stage("head");
        append_features(e);
      }
      return e.finish();
    }
    e.begin_stage("head");
    restore_carried_xs();
    if (feature_history.buffer && !verify)
      append_features(e);
    Ref sample_logits = last_logits;
    if (extract_last_row_head) {
      Ref last_hidden{delta.buffer, 0, mul(h, 2)};
      Ref last_norm{norm.buffer, 0, mul(h, 2)};
      Ref row0_logits{logits.buffer, 0, mul(v, 2)};
      auto &extract =
          e.emit("qwen_forward", "omlx_m5_forward_last_row", {.width = h},
                 {.width = 256}, {hidden, last_hidden});
      e.dynamic(extract, 2, words({t, h}), 0);
      if (nax_head) {
        normalize_f16(e, final_norm, last_hidden, f16_norm, 1U);
        linear_from_f16(e, head, f16_norm, f16_ones_scale, row0_logits, v, h,
                        1U);
      } else {
        normalize(e, final_norm, last_hidden, last_norm, 1U);
        linear(e, head, last_norm, row0_logits, v, h, 1U);
      }
      sample_logits = row0_logits;
    } else {
      bool head_done = false;
      if (pending_sq_slot.has_value()) {
        const U32 sq_final = *pending_sq_slot;
        pending_sq_slot.reset();
        if (!nax_head &&
            try_linear_norm_t1(e, head, final_norm, hidden, logits, v, h,
                               sq_final)) {
          e.profile.fused_final_norm_t1 = true;
          head_done = true;
        } else if (nax_head) {
          normalize_f16(e, final_norm, hidden, f16_norm);
        } else {
          normalize(e, final_norm, hidden, norm);
        }
      }
      if (!head_done) {
        if (nax_head)
          linear_from_f16(e, head, f16_norm, f16_ones_scale, logits, v, h);
        else
          linear(e, head, norm, logits, v, h);
      }
      if (verify)
        return e.finish();
      if (defer_sample) {
        auto &last = e.emit("qwen_forward", "omlx_m5_forward_last_row",
                            {.width = v}, {.width = 256}, {logits, last_logits});
        e.dynamic(last, 2, words({t, v}), 0);
        return e.finish();
      }
      if (t == 1U) {
        sample_logits = Ref{logits.buffer, 0, mul(v, 2)};
      } else {
        auto &last = e.emit("qwen_forward", "omlx_m5_forward_last_row",
                            {.width = v}, {.width = 256}, {logits, last_logits});
        e.dynamic(last, 2, words({t, v}), 0);
      }
    }
    e.begin_stage("sample");
    const Ref out_sampled = custom_sampled.buffer ? custom_sampled : sampled;
    const bool use_64tg_argmax =
        (v >= 256U) && (argmax_scratch.buffer != nullptr) &&
        !detail::forward::env_flag_enabled("OMLX_M5_DISABLE_64TG_ARGMAX");
    if (sampling_contract == SamplingContract::PositionCoupledV1) {
      if (use_64tg_argmax) {
        auto &st1 = e.emit("position_sampling",
                           "omlx_m5_position_argmax_stage1_64tg",
                           {.width = 64U * 256U, .height = 1U}, {.width = 256U},
                           {sample_logits, argmax_scratch});
        e.constant(st1, 2, words({v, 1, 1, 0}));
        if (feed_next_token.buffer) {
          auto &sample =
              e.emit("position_sampling",
                     "omlx_m5_position_sample_logits_feed_stage2",
                     {.width = 256U}, {.width = 256U},
                     {argmax_scratch, out_sampled, position_ids,
                      position_scores, feed_next_token});
          e.profile.sampling_offset = e.dynamic(
              sample, 5, std::vector<std::byte>(sizeof(PositionSamplingRow)));
          e.constant(sample, 6, words({v, 1, 1, 0}));
          return e.finish();
        }
        auto &sample =
            e.emit("position_sampling", "omlx_m5_position_sample_logits_stage2",
                   {.width = 256U}, {.width = 256U},
                   {sample_logits, argmax_scratch, out_sampled, position_ids,
                    position_scores});
        e.profile.sampling_offset = e.dynamic(
            sample, 5, std::vector<std::byte>(sizeof(PositionSamplingRow)));
        e.constant(sample, 6, words({v, 1, 1, 0}));
        return e.finish();
      }
      if (feed_next_token.buffer) {
        auto &sample = e.emit(
            "position_sampling", "omlx_m5_position_sample_logits_feed",
            {.width = 256}, {.width = 256},
            {sample_logits, out_sampled, position_ids, position_scores,
             feed_next_token});
        e.profile.sampling_offset = e.dynamic(
            sample, 5, std::vector<std::byte>(sizeof(PositionSamplingRow)));
        e.constant(sample, 6, words({v, 1, 1, 0}));
        return e.finish();
      }
      auto &sample = e.emit("position_sampling", "omlx_m5_position_sample_logits",
                            {.width = 256}, {.width = 256},
                            {sample_logits, out_sampled, position_ids, position_scores});
      e.profile.sampling_offset = e.dynamic(
          sample, 4, std::vector<std::byte>(sizeof(PositionSamplingRow)));
      e.constant(sample, 5, words({v, 1, 1, 0}));
      return e.finish();
    }
    if (use_64tg_argmax) {
      Ref vocab_parts{argmax_scratch.buffer, 0, 512U};
      auto &st1 =
          e.emit("vocab", "omlx_m5_vocab_argmax_stage1_64tg",
                 {.width = 64U * 256U}, {.width = 256U},
                 {sample_logits, vocab_parts});
      e.constant(st1, 2, words({v}));
      if (feed_next_token.buffer) {
        auto &sample = e.emit(
            "vocab", "omlx_m5_sample_logits_feed_stage2", {.width = 256U},
            {.width = 256U},
            {sample_logits, vocab_parts, out_sampled, feed_next_token});
        e.profile.sampling_offset =
            e.dynamic(sample, 4, std::vector<std::byte>(sizeof(SampleParams)));
        return e.finish();
      }
      auto &sample =
          e.emit("vocab", "omlx_m5_sample_logits_stage2", {.width = 256U},
                 {.width = 256U}, {sample_logits, vocab_parts, out_sampled});
      e.profile.sampling_offset =
          e.dynamic(sample, 3, std::vector<std::byte>(sizeof(SampleParams)));
      return e.finish();
    }
    if (feed_next_token.buffer) {
      auto &sample =
          e.emit("vocab", "omlx_m5_sample_logits_feed", {.width = 256},
                 {.width = 256}, {sample_logits, out_sampled, feed_next_token});
      e.profile.sampling_offset =
          e.dynamic(sample, 3, std::vector<std::byte>(sizeof(SampleParams)));
      return e.finish();
    }
    auto &sample = e.emit("vocab", "omlx_m5_sample_logits", {.width = 256},
                          {.width = 256}, {sample_logits, out_sampled});
    e.profile.sampling_offset =
        e.dynamic(sample, 2, std::vector<std::byte>(sizeof(SampleParams)));
    return e.finish();
  }

  Profile compile_single_sampler() {
    Encoder e(context, libraries, 1, "qwen.delegated_sample" + plan_key_suffix());
    e.begin_stage("sample");
    Ref row0_logits{logits.buffer, 0, mul(v, 2)};
    const bool use_64tg_argmax =
        (v >= 256U) && (argmax_scratch.buffer != nullptr) &&
        !detail::forward::env_flag_enabled("OMLX_M5_DISABLE_64TG_ARGMAX");
    if (sampling_contract == SamplingContract::PositionCoupledV1) {
      if (use_64tg_argmax) {
        auto &st1 = e.emit("position_sampling",
                           "omlx_m5_position_argmax_stage1_64tg",
                           {.width = 64U * 256U, .height = 1U}, {.width = 256U},
                           {row0_logits, argmax_scratch});
        e.constant(st1, 2, words({v, 1, 1, 0}));
        auto &sample =
            e.emit("position_sampling", "omlx_m5_position_sample_logits_stage2",
                   {.width = 256U}, {.width = 256U},
                   {row0_logits, argmax_scratch, sampled, position_ids,
                    position_scores});
        e.profile.sampling_offset = e.dynamic(
            sample, 5, std::vector<std::byte>(sizeof(PositionSamplingRow)));
        e.constant(sample, 6, words({v, 1, 1, 0}));
        return e.finish();
      }
      auto &sample = e.emit("position_sampling", "omlx_m5_position_sample_logits",
                            {.width = 256}, {.width = 256},
                            {row0_logits, sampled, position_ids, position_scores});
      e.profile.sampling_offset = e.dynamic(
          sample, 4, std::vector<std::byte>(sizeof(PositionSamplingRow)));
      e.constant(sample, 5, words({v, 1, 1, 0}));
      return e.finish();
    }
    if (use_64tg_argmax) {
      Ref vocab_parts{argmax_scratch.buffer, 0, 512U};
      auto &st1 =
          e.emit("vocab", "omlx_m5_vocab_argmax_stage1_64tg",
                 {.width = 64U * 256U}, {.width = 256U},
                 {row0_logits, vocab_parts});
      e.constant(st1, 2, words({v}));
      auto &sample =
          e.emit("vocab", "omlx_m5_sample_logits_stage2", {.width = 256U},
                 {.width = 256U}, {row0_logits, vocab_parts, sampled});
      e.profile.sampling_offset =
          e.dynamic(sample, 3, std::vector<std::byte>(sizeof(SampleParams)));
      return e.finish();
    }
    auto &sample = e.emit("vocab", "omlx_m5_sample_logits", {.width = 256},
                          {.width = 256}, {row0_logits, sampled});
    e.profile.sampling_offset =
        e.dynamic(sample, 2, std::vector<std::byte>(sizeof(SampleParams)));
    return e.finish();
  }

  void copy(Encoder &e, Ref from, Ref to, U64 bytes) {
    auto &d = e.emit("qwen_forward", "omlx_m5_forward_copy_bytes",
                     {.width = (bytes + 15) / 16}, {.width = 256}, {from, to});
    e.constant(d, 2, words({U32(bytes)}));
  }

  Profile compile_kv_cow() {
    const U32 words16_per_page = kv_copy_words16_per_page(
        kv_format_, 64U, U32(config.num_key_value_heads), U32(config.head_dim));
    Encoder e(context, libraries, 1, "qwen.kv_cow" + plan_key_suffix());
    U64 offset = 0;
    for (const auto &block : blocks) {
      if (block.type != LayerType::full_attention)
        continue;
      auto &d = e.emit("qwen_forward", "omlx_m5_forward_kv_page_copy_words16",
                       {.width = words16_per_page}, {.width = 128},
                       {block.state0, block.state1});
      offset = e.dynamic(d, 2, words({0, 0, words16_per_page}));
    }
    e.profile.sampling_offset = offset;
    return e.finish();
  }

  void execute_cow_copies(const std::vector<KvPageCopy> &copies) {
    if (copies.empty() || !page_pool)
      return;
    const U32 physical = page_pool->physical_pages();
    const U32 words16_per_page = kv_copy_words16_per_page(
        kv_format_, 64U, U32(config.num_key_value_heads), U32(config.head_dim));
    for (const auto move : copies) {
      if (move.source >= physical || move.destination >= physical ||
          move.source == move.destination)
        throw std::invalid_argument("KV copy page ID outside pool");
      const std::array<U32, 3> params{move.source, move.destination,
                                      words16_per_page};
      std::memcpy(kv_cow_profile.parameter_data.data() +
                      kv_cow_profile.sampling_offset,
                  params.data(), sizeof(params));
      kv_cow_profile.parameters.write("step", kv_cow_profile.parameter_data);
      if (kv_cow_profile.icb)
        static_cast<void>(context.replay_icb(*kv_cow_profile.icb));
      else {
        const std::array calls = {
            MetalCommandInvocation{.plan = &kv_cow_profile.plan,
                                   .bindings = kv_cow_profile.bindings,
                                   .parameters = &kv_cow_profile.parameters}};
        static_cast<void>(context.execute_batch(calls));
      }
    }
  }

  Profile compile_inspect_norm() {
    const U32 cap = workspace_rows_;
    Encoder e(context, libraries, cap, "qwen.inspect_norm" + plan_key_suffix());
    auto &d = e.emit("rmsnorm_projection", "omlx_m5_rmsnorm_a16",
                     {.width = 128, .height = cap}, {.width = 128},
                     {hidden, direct(final_norm, h), norm});
    e.dynamic(d, 3,
              words({h, cap, word(float(config.rms_norm_eps)),
                     word(normalization_weight_offset)}),
              1);
    return e.finish();
  }

  Profile compile_inspect_head_chunk() {
    Encoder e(
        context, libraries, 8,
        std::string("qwen.inspect_head_chunk") +
            (linear_policy == LinearNumericPolicy::RowExactPortableV1
                 ? ".linear-row-exact-v1"
                 : "") +
            plan_key_suffix());
    e.begin_stage("head_chunk");
    Ref chunk_norm{delta.buffer, 0, mul(8, mul(h, 2))};
    auto &slice =
        e.emit("qwen_forward", "omlx_m5_forward_slice_rows",
               {.width = U64(8) * h}, {.width = 128}, {norm, chunk_norm});
    e.dynamic(slice, 2, words({8, h, 0}), 0, 2);
    linear(e, head, chunk_norm, logits, v, h);
    return e.finish();
  }

  void append_features(Encoder &e) {
    const U32 width = h * U32(feature_layers.size());
    auto &d = e.emit("qwen_forward", "omlx_m5_forward_copy_rows_ring",
                     {.width = U64(e.profile.capacity) * width}, {.width = 128},
                     {features, feature_history});
    e.dynamic(d, 2, words({e.profile.capacity, width, 0, feature_ring_rows_}),
              0, 2);
  }

  Profile compile_commit() {
    Encoder e(
        context, libraries, 8,
        std::string("qwen.commit") +
            (linear_policy == LinearNumericPolicy::RowExactPortableV1
                 ? ".linear-row-exact-v1"
                 : "") +
            plan_key_suffix());
    const bool disable_gdn_spec_fusion =
        detail::forward::env_flag_enabled("OMLX_M5_DISABLE_GDN_SPEC_FUSION");
    const U32 kd = U32(config.linear_key_head_dim);
    for (const auto &block : blocks) {
      if (block.type == LayerType::full_attention)
        continue;
      auto params = words(
          {U32(config.linear_num_key_heads), U32(config.linear_num_value_heads),
           kd, U32(config.linear_value_head_dim), conv,
           U32(config.linear_conv_kernel_dim), 8,
           word(float(config.rms_norm_eps))});
      if (disable_gdn_spec_fusion) {
        auto &rec = e.emit("gdn_layer", "omlx_m5_gdn_replay_prefix",
                           {.width = U64(gv) * kd}, {.width = 128},
                           {block.state0, block.replay_keys,
                            block.replay_decays, block.replay_deltas});
        e.dynamic(rec, 4, params, 6);
        auto &history =
            e.emit("gdn_layer", "omlx_m5_gdn_commit_history", {.width = conv},
                   {.width = 128}, {block.state1, block.replay_qkv});
        e.dynamic(history, 2, params, 6);
      } else {
        U64 state_tgs = 0;
        U64 hist_tgs = 0;
        if ((kd & 3U) == 0U) {
          state_tgs = (U64(gv) * (kd >> 2U) + 127ULL) >> 7ULL;
          const U32 hist_items = ((conv & 3U) == 0U) ? (conv >> 2U) : conv;
          hist_tgs = (U64(hist_items) + 127ULL) >> 7ULL;
        } else {
          state_tgs = (U64(gv) * kd + 127ULL) >> 7ULL;
          hist_tgs = (U64(conv) + 127ULL) >> 7ULL;
        }
        auto &commit_d = e.emit(
            "gdn_layer", "omlx_m5_gdn_replay_prefix_commit_history",
            {.width = (state_tgs + hist_tgs) * 128ULL}, {.width = 128U},
            {block.state0, block.replay_keys, block.replay_decays,
             block.replay_deltas, block.state1, block.replay_qkv});
        e.dynamic(commit_d, 6, params, 6);
      }
    }
    if (feature_history.buffer)
      append_features(e);
    return e.finish();
  }

  // Cross-slot shared-weight packed batch profile & LRU cache (up to 8 shapes).
  struct PackedSlotPosPatch {
    U32 slice_index = 0;
    U64 offset = 0;
    U32 word = 0;
    bool operator==(const PackedSlotPosPatch &) const = default;
  };
  struct PackedShapeKey {
    bool verify = false;
    bool sample_in_icb = false;
    bool scatter_slot_logits = true;
    U32 affine4_split_env = 0;
    U32 bf16_split_env = 0;
    std::vector<const Impl *> slots;
    std::vector<U32> widths;
    bool operator==(const PackedShapeKey &) const = default;
  };
  struct CachedPackedProfile {
    PackedShapeKey key;
    Profile profile;
    std::vector<PackedSlotPosPatch> pos_patches;
    std::vector<U64> sample_opt_offsets;
  };

  CachedPackedProfile compile_packed_batch_profile(const PackedShapeKey &skey) {
    const std::size_t B = skey.slots.size();
    U32 total_m = 0;
    U32 max_w = 0;
    bool all_w_eq_1 = true;
    std::vector<U32> row_offs(B, 0U);
    std::array<U32, 8> row_offs_8{};
    std::array<U32, 8> widths_8{};
    std::array<U32, 8> caps_8{};
    for (std::size_t b_idx = 0; b_idx < B; ++b_idx) {
      row_offs[b_idx] = total_m;
      row_offs_8[b_idx] = total_m;
      widths_8[b_idx] = skey.widths[b_idx];
      caps_8[b_idx] = skey.slots[b_idx]->max_context;
      all_w_eq_1 &= (skey.widths[b_idx] == 1U);
      max_w = std::max(max_w, skey.widths[b_idx]);
      total_m += skey.widths[b_idx];
    }
    U32 cap_m = workspace_rows_;
    for (U32 cand : {1U, 2U, 4U, 8U, 12U, 16U, 24U, 32U, 64U, 128U, 256U, 512U,
                     1024U, 2048U, 4096U}) {
      if (cand >= total_m && cand <= workspace_rows_) {
        cap_m = cand;
        break;
      }
    }

    const bool single_t1_tdom =
        (!skey.verify && B == 1U && total_m == 1U && !speculative_enabled &&
         t_domain_active() && pretiled_qmv_t1_enabled() &&
         !inspect_tensors_workspace);
    const bool use_m2 =
        (!skey.verify && total_m == 2U && !speculative_enabled &&
         t_domain_active() && pretiled_qmv_t1_enabled() &&
         !inspect_tensors_workspace && has_row_exact_metallib_);
    const bool allow_t1_fused =
        (single_t1_tdom && rms_sq_buf.buffer != nullptr &&
         moe_tg_done_buf.buffer != nullptr && h < 65536U);
    const bool allow_m2_fused =
        (use_m2 && rms_sq_buf.buffer != nullptr &&
         moe_tg_done_buf.buffer != nullptr && h < 65536U);
    const U32 m2_sq_mul = config.num_experts ? 4U : 1U;
    const bool allow_moe9_bt =
        (config.num_experts > 0 && !skey.verify && !speculative_enabled &&
         t_domain_active() && total_m >= 2U && total_m <= 8U);

    ScopedTdomOverride tdom_guard{tdom_batch_override_, single_t1_tdom};

    Fnv1a64 sk_hash;
    sk_hash.u64(skey.verify ? 1ULL : 0ULL);
    sk_hash.u64(skey.sample_in_icb ? 2ULL : (skey.scatter_slot_logits ? 1ULL : 0ULL));
    sk_hash.u64(skey.affine4_split_env);
    sk_hash.u64(skey.bf16_split_env);
    for (std::size_t b_idx = 0; b_idx < B; ++b_idx) {
      sk_hash.u64(skey.slots[b_idx]->owner_id);
      sk_hash.u64(skey.widths[b_idx]);
    }
    const char *mode_tag =
        skey.verify ? "qwen.packed_verify."
                    : (skey.sample_in_icb
                           ? "qwen.packed_ar_samp."
                           : (skey.scatter_slot_logits ? "qwen.packed_ar."
                                                       : "qwen.packed_ar_ns."));
    Encoder e(context, libraries, cap_m,
              std::string(mode_tag) + sk_hash.hex() + plan_key_suffix());
    e.profile.sampling_contract_id =
        std::string(sampling_contract_name(sampling_contract));

    std::vector<PackedSlotPosPatch> pos_patches;
    std::vector<U64> sample_opt_offsets;
    auto add_slot_patches =
        [&](U32 b_idx, const std::vector<std::pair<U64, U32>> &raw) {
          for (const auto &[off, wd] : raw) {
            const PackedSlotPosPatch patch{b_idx, off, wd};
            if (std::ranges::find(pos_patches, patch) == pos_patches.end()) {
              pos_patches.push_back(patch);
            }
          }
        };

    // 1. Packed embedding across all M tokens.
    e.begin_stage("embedding");
    embedding_lookup(e, embedding, token_ids, hidden, zero, v, h);

    // 2. Shared-weight transformer layers across all M tokens.
    e.delegated_prefill = (!skey.verify && B == 1U && total_m > 16U);
    const U32 hv = U32(config.linear_num_value_heads);
    const U32 kh = U32(config.linear_num_key_heads);
    const U32 kd = U32(config.linear_key_head_dim);
    const U32 vd = U32(config.linear_value_head_dim);
    const bool affine_xs_eligible =
        (!use_m2 && !single_t1_tdom && affine_nax_active() &&
         affine_nax_xs.buffer != nullptr && (h % 64U) == 0U && total_m > 1U);
    const bool can_batched_gdn_t1 =
        (!skey.verify && B >= 2U && B <= 8U && all_w_eq_1 &&
         !inspect_tensors_workspace && kd == 128U && vd == 128U &&
         config.linear_conv_kernel_dim == 4U && (conv % 4U) == 0U && kh > 0U &&
         hv >= kh && (hv % kh) == 0U && gv == hv * 128U);
    const bool can_batched_gdn_verify =
        (skey.verify && B >= 2U && B <= 8U && max_w >= 1U && max_w <= 8U &&
         !inspect_tensors_workspace &&
         !detail::forward::env_flag_enabled("OMLX_M5_DISABLE_GDN_SPEC_FUSION") &&
         kd == 128U && (vd % 4U) == 0U && (gv % 4U) == 0U && kh > 0U &&
         hv >= kh && (hv % kh) == 0U);
    const bool batched_attn_base =
        (B >= 2U && B <= 8U && !inspect_tensors_workspace &&
         attention_policy == omlx::m5::AttentionNumericPolicy::RowExactV1 &&
         attention_positions.buffer != nullptr);

    const bool verify_head_f16 =
        (skey.verify && !use_m2 && !single_t1_tdom &&
         !has_bypass_lora(head) &&
         use_mxfp4_nax_for_weight(head, total_m) &&
         [&] {
           const auto *tw = mxfp4_nax_cache->find(head.object_id);
           return tw && tw->n == v && tw->k == h &&
                  mxfp4_nax_select_wide(*tw, total_m, total_m,
                                        silicon_tuning.gpu_cores, false) ==
                      NaxWidePrefillTile::none;
         }());
    const bool verify_head_affine_xs =
        (skey.verify && !verify_head_f16 && affine_xs_eligible);

    const U32 num_sq_words =
        allow_m2_fused ? ((U32(blocks.size()) * 2U + 4U) * 2U * m2_sq_mul)
                       : (U32(blocks.size()) * 2U + 4U);
    const U32 num_moe_words =
        (config.num_experts && allow_t1_fused) ? (U32(blocks.size()) + 4U) : 0U;
    std::optional<U32> pending_sq_slot;
    std::optional<Encoder::GroupSumKey> carried_affine_xs;
    auto restore_carried_xs = [&]() {
      if (carried_affine_xs.has_value()) {
        e.set_group_sum(*carried_affine_xs);
        carried_affine_xs.reset();
      }
    };
    auto make_norm_xs_key = [&](U32 rows) -> Encoder::GroupSumKey {
      return Encoder::GroupSumKey{affine_nax_xs.buffer,
                                  affine_nax_xs.offset,
                                  norm.buffer,
                                  norm.offset,
                                  h,
                                  64U,
                                  rows,
                                  false,
                                  2U};
    };

    U32 layer_index = 0;
    for (const auto &block : blocks) {
      const bool in_f16 =
          (!use_m2 && !single_t1_tdom &&
           block_input_nax(block, total_m, total_m));
      const bool mlp_f16 =
          (!use_m2 && !single_t1_tdom &&
           block_mlp_input_nax(block, total_m, total_m));
      const bool in_affine_xs = (!in_f16 && affine_xs_eligible);
      const bool mlp_affine_xs = (!mlp_f16 && affine_xs_eligible);
      const bool layer_fuse_t1 = allow_t1_fused && !in_f16 && !mlp_f16;
      const bool layer_fuse_m2 = allow_m2_fused && !in_f16 && !mlp_f16;

      e.begin_stage("layer." + std::to_string(layer_index) +
                    (block.type == LayerType::full_attention ? ".attention"
                                                             : ".gdn"));
      restore_carried_xs();

      if (layer_index == 0U) {
        if (in_f16) {
          normalize_f16(e, block.get("input_norm"), hidden, f16_norm, total_m);
        } else if (allow_t1_fused) {
          normalize_init_zero_t1(e, block.get("input_norm"), hidden, norm,
                                 num_sq_words, num_moe_words);
        } else if (allow_m2_fused) {
          normalize_init_zero_m2(e, block.get("input_norm"), hidden, norm,
                                 num_sq_words, num_moe_words);
        } else if (in_affine_xs) {
          emit_affine_rmsnorm_xsum(
              e, hidden, direct(block.get("input_norm"), h), norm,
              affine_nax_xs, h, float(config.rms_norm_eps),
              normalization_weight_offset, total_m);
        } else {
          normalize(e, block.get("input_norm"), hidden, norm, total_m);
        }
      } else if (!(layer_fuse_t1 || layer_fuse_m2) &&
                 pending_sq_slot.has_value()) {
        if (in_f16)
          normalize_f16(e, block.get("input_norm"), hidden, f16_norm, total_m);
        else if (in_affine_xs)
          emit_affine_rmsnorm_xsum(
              e, hidden, direct(block.get("input_norm"), h), norm,
              affine_nax_xs, h, float(config.rms_norm_eps),
              normalization_weight_offset, total_m);
        else
          normalize(e, block.get("input_norm"), hidden, norm, total_m);
        pending_sq_slot.reset();
      }

      if (block.type == LayerType::full_attention) {
        const std::array<MaterializedWeightView, 3> parts{
            block.get("attention_q"), block.get("attention_k"),
            block.get("attention_v")};
        const std::array<U32, 3> part_ns{2 * q, kv, kv};

        bool can_batched_attn =
            batched_attn_base && (skey.bf16_split_env == 0U);
        bool can_batched_affine4_attn =
            batched_attn_base && config.head_dim == 256 &&
            affine4_kv_runtime_eligible(
                context, libraries, U32(config.num_attention_heads),
                U32(config.num_key_value_heads), U32(config.head_dim));
        if (can_batched_attn || can_batched_affine4_attn) {
          const auto &s0_blk = skey.slots[0]->blocks[layer_index];
          for (std::size_t b_idx = 0; b_idx < B; ++b_idx) {
            const auto &sp = *skey.slots[b_idx];
            const auto &sb = sp.blocks[layer_index];
            const bool shared_kv =
                sp.page_pool != nullptr &&
                sb.state0.buffer == s0_blk.state0.buffer &&
                sb.state0.offset == s0_blk.state0.offset &&
                sb.state1.buffer == s0_blk.state1.buffer &&
                sb.state1.offset == s0_blk.state1.offset;
            if (!shared_kv || sp.kv_format_ != KvStorageFormat::BFloat16) {
              can_batched_attn = false;
            }
            if (!shared_kv ||
                sp.kv_format_ != KvStorageFormat::Affine4Group128) {
              can_batched_affine4_attn = false;
            }
            if (!can_batched_attn && !can_batched_affine4_attn) {
              break;
            }
          }
        }
        const bool use_batched_affine4_flash =
            can_batched_affine4_attn && flash_nax_prefill_ &&
            flash_nax_prefill_runtime_eligible(context, libraries,
                                               U32(config.head_dim)) &&
            !skey.verify && max_w >= 16U;
        BatchedAffine4SplitPlan batched_affine4_split{};
        if (can_batched_affine4_attn && !use_batched_affine4_flash) {
          if (skey.affine4_split_env > 0U && split_max.buffer &&
              split_sum.buffer && split_output.buffer) {
            batched_affine4_split = plan_batched_affine4_split(
                total_m, U32(B), max_w, U32(config.num_attention_heads),
                U32(config.num_key_value_heads), skey.affine4_split_env,
                split_max.bytes, split_sum.bytes, split_output.bytes);
          }
          if (!batched_affine4_split.valid) {
            can_batched_affine4_attn = false;
          }
        }

        const auto *qkv_bypass =
            find_bypass_group(layer_index, QwenBypassGroupKind::attn_qkv);
        bool qkv_done = false;
        bool used_strided_qkv = false;
        if (layer_fuse_t1 && pending_sq_slot.has_value()) {
          const U32 sq_in = *pending_sq_slot;
          pending_sq_slot.reset();
          Ref d0{}, s0{}, d1{}, s1{}, d2{}, s2{}, h0{}, h1{}, h2{};
          if (qkv_bypass == nullptr && 2 * q < 65536U && kv < 65536U &&
              parts[0].row_count == 2 * q && parts[1].row_count == kv &&
              parts[2].row_count == kv && parts[0].in_features == h &&
              parts[1].in_features == h && parts[2].in_features == h) {
            if (resolve_pretiled_mxfp4_t1(parts[0], 1U, &d0, &s0) &&
                resolve_pretiled_mxfp4_t1(parts[1], 1U, &d1, &s1) &&
                resolve_pretiled_mxfp4_t1(parts[2], 1U, &d2, &s2)) {
              const Ref nw_ref = direct(block.get("input_norm"), h);
              const bool use_r4 = ((2 * q + 2 * kv) >= 2048U);
              const U32 rptg = use_r4 ? 8U : 4U;
              const U32 tgs = ((2 * q + rptg - 1U) / rptg) +
                              2U * ((kv + rptg - 1U) / rptg);
              const char *fn =
                  use_r4 ? "omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r4_norm_t1"
                         : "omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r2_norm_t1";
              auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                               {.width = 64U},
                               {hidden, nw_ref, rms_sq_buf, d0, s0, d1, s1, d2,
                                s2, query, key, value});
              e.dynamic(d, 12,
                        fused_norm_params(2 * q, kv, kv, 0U, h, 0U, sq_in));
              for (std::size_t i = 0; i < 3; ++i)
                e.record_weight(parts[i], part_ns[i], h, "pretiled_qmv_t1",
                                "pretiled_mxfp4_qmv_concat3_norm_t1_v4", {},
                                "legacy_route_v1", {}, false);
              qkv_done = true;
            } else if (bool lv0 = false, lv1 = false, lv2 = false;
                       resolve_pretiled_affine_t1(parts[0], 1U, &d0, &h0,
                                                  &s0, &lv0) &&
                       resolve_pretiled_affine_t1(parts[1], 1U, &d1, &h1,
                                                  &s1, &lv1) &&
                       resolve_pretiled_affine_t1(parts[2], 1U, &d2, &h2,
                                                  &s2, &lv2) &&
                       lv0 == lv1 && lv1 == lv2) {
              const Ref nw_ref = direct(block.get("input_norm"), h);
              const U32 b0 = affine_nax_bits(parts[0].format);
              const U32 b1 = affine_nax_bits(parts[1].format);
              const U32 b2 = affine_nax_bits(parts[2].format);
              const U32 packed_bits = b0 | (b1 << 8U) | (b2 << 16U);
              const bool use_r4 = ((2 * q + 2 * kv) >= 2048U);
              const U32 rptg = use_r4 ? 8U : 4U;
              const U32 tgs = ((2 * q + rptg - 1U) / rptg) +
                              2U * ((kv + rptg - 1U) / rptg);
              const bool all_q456 = (b0 >= 4U && b0 <= 6U) &&
                                    (b1 >= 4U && b1 <= 6U) &&
                                    (b2 >= 4U && b2 <= 6U);
              const char *fn =
                  use_r4 ? "omlx_m5_pretiled_qmv_affine_concat3_r4_norm_t1"
                         : "omlx_m5_pretiled_qmv_affine_concat3_r2_norm_t1";
              if (use_r4 && b0 == b1 && b1 == b2 &&
                  (b0 == 4U || b0 == 5U || b0 == 6U)) {
                fn = (b0 == 5U)
                         ? "omlx_m5_pretiled_qmv_affine_concat3_q5_r4_norm_t1"
                     : (b0 == 4U)
                         ? "omlx_m5_pretiled_qmv_affine_concat3_q4_r4_norm_t1"
                         : "omlx_m5_pretiled_qmv_affine_concat3_q6_r4_norm_t1";
              } else if (all_q456) {
                fn = use_r4
                         ? "omlx_m5_pretiled_qmv_affine_concat3_q456_r4_norm_t1"
                         : "omlx_m5_pretiled_qmv_affine_concat3_q456_r2_norm_t1";
              }
              auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                               {.width = 64U},
                               {hidden, nw_ref, rms_sq_buf, d0, h0, s0, d1, h1,
                                s1, d2, h2, s2, query, key, value});
              e.dynamic(d, 15,
                        fused_norm_params(2 * q, kv, kv, 0U,
                                          pretiled_k_flags(h, lv0, false),
                                          packed_bits, sq_in));
              for (std::size_t i = 0; i < 3; ++i)
                e.record_weight(parts[i], part_ns[i], h, "pretiled_qmv_t1",
                                "pretiled_affine_qmv_concat3_norm_t1_v4", {},
                                "legacy_route_v1", {}, false);
              qkv_done = true;
            }
          }
          if (!qkv_done)
            normalize(e, block.get("input_norm"), hidden, norm, total_m);
        } else if (layer_fuse_m2 && pending_sq_slot.has_value()) {
          const U32 sq_in = *pending_sq_slot;
          pending_sq_slot.reset();
          qkv_done =
              try_concat3_m2(e, parts, part_ns, h, hidden, query, key, value,
                             &block.get("input_norm"), sq_in * m2_sq_mul);
          if (!qkv_done)
            normalize(e, block.get("input_norm"), hidden, norm, total_m);
        }

        if (!qkv_done && single_t1_tdom && !in_f16) {
          Ref d0{}, s0{}, d1{}, s1{}, d2{}, s2{}, h0{}, h1{}, h2{};
          if (parts[0].row_count == 2 * q && parts[1].row_count == kv &&
              parts[2].row_count == kv && parts[0].in_features == h &&
              parts[1].in_features == h && parts[2].in_features == h) {
            if (resolve_pretiled_mxfp4_t1(parts[0], 1U, &d0, &s0) &&
                resolve_pretiled_mxfp4_t1(parts[1], 1U, &d1, &s1) &&
                resolve_pretiled_mxfp4_t1(parts[2], 1U, &d2, &s2)) {
              const bool use_r4 = ((2 * q + 2 * kv) >= 2048U);
              const U32 rptg = use_r4 ? 8U : 4U;
              const U32 tgs = ((2 * q + rptg - 1U) / rptg) +
                              2U * ((kv + rptg - 1U) / rptg);
              const char *fn =
                  use_r4 ? "omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r4_t1"
                         : "omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r2_t1";
              auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                               {.width = 64U},
                               {norm, d0, s0, d1, s1, d2, s2, query, key,
                                value});
              e.dynamic(d, 10, words({2 * q, kv, kv, h}));
              for (std::size_t i = 0; i < 3; ++i)
                e.record_weight(parts[i], part_ns[i], h, "pretiled_qmv_t1",
                                "pretiled_mxfp4_qmv_concat3_t1_v3", {},
                                "legacy_route_v1", {}, false);
              qkv_done = true;
            } else if (bool lv0 = false, lv1 = false, lv2 = false;
                       resolve_pretiled_affine_t1(parts[0], 1U, &d0, &h0,
                                                  &s0, &lv0) &&
                       resolve_pretiled_affine_t1(parts[1], 1U, &d1, &h1,
                                                  &s1, &lv1) &&
                       resolve_pretiled_affine_t1(parts[2], 1U, &d2, &h2,
                                                  &s2, &lv2) &&
                       lv0 == lv1 && lv1 == lv2) {
              const U32 b0 = affine_nax_bits(parts[0].format);
              const U32 b1 = affine_nax_bits(parts[1].format);
              const U32 b2 = affine_nax_bits(parts[2].format);
              const U32 packed_bits = b0 | (b1 << 8U) | (b2 << 16U);
              const bool use_r4 = ((2 * q + 2 * kv) >= 2048U);
              const U32 rptg = use_r4 ? 8U : 4U;
              const U32 tgs = ((2 * q + rptg - 1U) / rptg) +
                              2U * ((kv + rptg - 1U) / rptg);
              const bool all_q456 = (b0 >= 4U && b0 <= 6U) &&
                                    (b1 >= 4U && b1 <= 6U) &&
                                    (b2 >= 4U && b2 <= 6U);
              const char *fn = "omlx_m5_pretiled_qmv_affine_concat3_t1";
              if (use_r4 && b0 == b1 && b1 == b2 &&
                  (b0 == 4U || b0 == 5U || b0 == 6U)) {
                fn = (b0 == 5U) ? "omlx_m5_pretiled_qmv_affine_concat3_q5_r4_t1"
                     : (b0 == 4U)
                         ? "omlx_m5_pretiled_qmv_affine_concat3_q4_r4_t1"
                         : "omlx_m5_pretiled_qmv_affine_concat3_q6_r4_t1";
              } else if (all_q456) {
                fn = use_r4 ? "omlx_m5_pretiled_qmv_affine_concat3_q456_r4_t1"
                            : "omlx_m5_pretiled_qmv_affine_concat3_q456_r2_t1";
              }
              auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                               {.width = 64U},
                               {norm, d0, h0, s0, d1, h1, s1, d2, h2, s2, query,
                                key, value});
              e.dynamic(d, 13,
                        words({2 * q, kv, kv, pretiled_k_flags(h, lv0, false),
                               packed_bits}));
              for (std::size_t i = 0; i < 3; ++i)
                e.record_weight(parts[i], part_ns[i], h, "pretiled_qmv_t1",
                                "pretiled_affine_qmv_concat3_t1_v3", {},
                                "legacy_route_v1", {}, false);
              qkv_done = true;
            }
          }
        }
        if (!qkv_done && use_m2) {
          qkv_done =
              try_concat3_m2(e, parts, part_ns, h, norm, query, key, value);
        }
        if (!qkv_done &&
            try_concat_nax(e, block.concat_qkv_id, parts, part_ns, h, in_f16,
                           norm, concat_out, total_m)) {
          if (can_batched_attn || can_batched_affine4_attn || total_m == 1U) {
            used_strided_qkv = true;
          } else {
            emit_split3(e, concat_out, query, key, value, 2 * q, kv, kv,
                        total_m);
          }
          qkv_done = true;
        }
        if (!qkv_done) {
          if (in_f16) {
            linear_from_f16(e, block.get("attention_q"), f16_norm,
                            f16_ones_scale, query, 2 * q, h, total_m);
            linear_from_f16(e, block.get("attention_k"), f16_norm,
                            f16_ones_scale, key, kv, h, total_m);
            linear_from_f16(e, block.get("attention_v"), f16_norm,
                            f16_ones_scale, value, kv, h, total_m);
          } else if (use_m2) {
            if (!try_linear_m2_base(e, block.get("attention_q"), norm, query,
                                    2 * q, h))
              linear_base(e, block.get("attention_q"), norm, query, 2 * q, h,
                          total_m);
            if (!try_linear_m2_base(e, block.get("attention_k"), norm, key, kv,
                                    h))
              linear_base(e, block.get("attention_k"), norm, key, kv, h,
                          total_m);
            if (!try_linear_m2_base(e, block.get("attention_v"), norm, value,
                                    kv, h))
              linear_base(e, block.get("attention_v"), norm, value, kv, h,
                          total_m);
          } else {
            linear_base(e, block.get("attention_q"), norm, query, 2 * q, h,
                        total_m);
            linear_base(e, block.get("attention_k"), norm, key, kv, h, total_m);
            linear_base(e, block.get("attention_v"), norm, value, kv, h,
                        total_m);
          }
        }
        if (qkv_bypass != nullptr) {
          const U32 qkv_tot = 2 * q + 2 * kv;
          const std::array<detail::lora_rt::BypassPartDest, 3> dests =
              used_strided_qkv
                  ? std::array<detail::lora_rt::BypassPartDest, 3>{{
                        {concat_out, qkv_tot, 0U},
                        {concat_out, qkv_tot, 2 * q},
                        {concat_out, qkv_tot, 2 * q + kv},
                    }}
                  : std::array<detail::lora_rt::BypassPartDest, 3>{{
                        {query, 2 * q, 0U},
                        {key, kv, 0U},
                        {value, kv, 0U},
                    }};
          detail::lora_rt::emit_bypass_sibling_group(
              e, *qkv_bypass, norm, dests, lora_bypass_tmp_,
              bf16_nax_bypass_ok_, silicon_tuning.gpu_cores, total_m);
        }

        if (can_batched_attn) {
          const auto &c = config;
          const U32 hd = U32(c.head_dim);
          const U32 qh = U32(c.num_attention_heads);
          const U32 kvh = U32(c.num_key_value_heads);
          const U32 qkv_stride = used_strided_qkv ? (2 * q + 2 * kv) : 0U;
          Ref q_ref = used_strided_qkv ? concat_out : query;
          Ref gate_ref =
              used_strided_qkv
                  ? Ref{concat_out.buffer, concat_out.offset + mul(hd, 2U),
                        concat_out.bytes - mul(hd, 2U)}
                  : Ref{query.buffer, query.offset + mul(hd, 2U),
                        query.bytes - mul(hd, 2U)};
          Ref k_ref =
              used_strided_qkv
                  ? Ref{concat_out.buffer, concat_out.offset + mul(2 * q, 2U),
                        concat_out.bytes - mul(2 * q, 2U)}
                  : key;
          Ref v_ref =
              used_strided_qkv
                  ? Ref{concat_out.buffer,
                        concat_out.offset + mul(2 * q + kv, 2U),
                        concat_out.bytes - mul(2 * q + kv, 2U)}
                  : value;
          const U32 q_tok_stride = used_strided_qkv ? qkv_stride : (2 * q);
          const U32 k_tok_stride = used_strided_qkv ? qkv_stride : kv;
          const U32 v_tok_stride = used_strided_qkv ? qkv_stride : kv;

          std::array<Ref, 8> pt{};
          for (std::size_t i = 0; i < 8; ++i) {
            const auto &tb = skey.slots[i < B ? i : 0]->page_pool->table_buffer();
            pt[i] = Ref{&tb, 0, tb.size()};
          }
          const auto &s0_blk = skey.slots[0]->blocks[layer_index];
          const U32 rot_dim =
              U32(std::llround(c.head_dim * c.partial_rotary_factor));

          std::vector<U32> post_w{qh,
                                  kvh,
                                  hd,
                                  rot_dim,
                                  total_m,
                                  q_tok_stride,
                                  hd * 2U,
                                  k_tok_stride,
                                  hd,
                                  v_tok_stride,
                                  U32(B),
                                  word(float(c.rms_norm_eps)),
                                  word(float(c.rope_theta)),
                                  word(normalization_weight_offset)};
          post_w.insert(post_w.end(), row_offs_8.begin(), row_offs_8.end());
          post_w.insert(post_w.end(), widths_8.begin(), widths_8.end());
          auto &post = e.emit(
              "attention_exact",
              "omlx_m5_attention_qk_norm_rope_kv_append_batched_exact",
              {.width = 32U, .height = U64(total_m) * (qh + kvh)},
              {.width = 32U},
              {q_ref, k_ref, v_ref, direct(block.get("q_norm"), hd),
               direct(block.get("k_norm"), hd), s0_blk.state0, s0_blk.state1,
               attention_positions, pt[0], pt[1], pt[2], pt[3], pt[4], pt[5],
               pt[6], pt[7]});
          e.dynamic(post, 16, words_span(post_w));

          const char *attn_fn =
              strict_two_pass_attn_
                  ? "omlx_m5_paged_causal_attention_batched_exact"
                  : "omlx_m5_paged_causal_attention_batched_exact_1pass";
          const U32 repeat = (kvh > 0U) ? (qh / kvh) : 1U;
          U32 tptg_y = 1U;
          if (!detail::forward::env_flag_enabled(
                  "OMLX_M5_DISABLE_GQA_EXACT_ATTENTION") &&
              hd == 256U && kvh > 0U && (qh % kvh) == 0U && repeat > 0U &&
              repeat <= 32U) {
            const std::string lib_key = "qwen.forward.library.attention_exact";
            context.register_library(
                {.key = lib_key,
                 .path = libraries / "omlx_m5_attention_exact.metallib"});
            const auto pinfo = context.warmup_pipeline(
                {.key = std::string("qwen.forward.") + attn_fn,
                 .library_key = lib_key,
                 .function_name = attn_fn,
                 .variant = "fixed-qwen",
                 .support_indirect_command_buffers =
                     context.capabilities().profile.compute_icb_eligible});
            const U32 max_pipe_sgs = U32(std::min<U64>(
                32ULL, pinfo.max_threads_per_threadgroup / 32ULL));
            const U32 max_sgs =
                (repeat <= 16U) ? std::min<U32>(16U, max_pipe_sgs)
                                : max_pipe_sgs;
            for (U32 bq : {8U, 4U, 2U, 1U}) {
              if (bq <= max_w && repeat * bq <= max_sgs) {
                const U32 t_tiles = (max_w + bq - 1U) / bq;
                if (U64(B) * t_tiles * kvh >= 16ULL || bq == 1U) {
                  tptg_y = repeat * bq;
                  break;
                }
              }
            }
          }
          const U32 bq = (tptg_y > 1U) ? (tptg_y / repeat) : 1U;
          const U64 grid_h =
              (tptg_y > 1U) ? (U64((max_w + bq - 1U) / bq) * kvh * tptg_y)
                            : (U64(max_w) * qh);
          std::vector<U32> attn_w{qh,
                                  kvh,
                                  hd,
                                  U32(B),
                                  64U,
                                  skey.slots[0]->page_pool->physical_pages(),
                                  q_tok_stride,
                                  hd * 2U,
                                  q_tok_stride,
                                  hd * 2U};
          attn_w.insert(attn_w.end(), row_offs_8.begin(), row_offs_8.end());
          attn_w.insert(attn_w.end(), widths_8.begin(), widths_8.end());
          attn_w.insert(attn_w.end(), caps_8.begin(), caps_8.end());
          auto &attn = e.emit(
              "attention_exact", attn_fn,
              {.width = 32U, .height = grid_h, .depth = U64(B)},
              {.width = 32U, .height = tptg_y, .depth = 1U},
              {q_ref, gate_ref, s0_blk.state0, s0_blk.state1, mixer,
               attention_positions, pt[0], pt[1], pt[2], pt[3], pt[4], pt[5],
               pt[6], pt[7]});
          e.dynamic(attn, 14, words_span(attn_w));
          e.profile.attention_kernel = attn_fn;
          e.profile.attention_grid = attn.grid;
          e.profile.attention_numeric_contract_id =
              strict_two_pass_attn_
                  ? "paged.simd32.globalmax.ordered_softmax.f32_gate.v1"
                  : "paged.simd32.online_sb512.ordered_softmax.f32_gate.v1";
          e.profile.attention_row_exact_eligible = true;
        } else if (can_batched_affine4_attn) {
          const auto &c = config;
          const U32 hd = U32(c.head_dim);
          const U32 qkv_stride = used_strided_qkv ? (2 * q + 2 * kv) : 0U;
          Ref q_ref = used_strided_qkv ? concat_out : query;
          Ref gate_ref =
              used_strided_qkv
                  ? Ref{concat_out.buffer, concat_out.offset + mul(hd, 2U),
                        concat_out.bytes - mul(hd, 2U)}
                  : Ref{query.buffer, query.offset + mul(hd, 2U),
                        query.bytes - mul(hd, 2U)};
          Ref k_ref =
              used_strided_qkv
                  ? Ref{concat_out.buffer, concat_out.offset + mul(2 * q, 2U),
                        concat_out.bytes - mul(2 * q, 2U)}
                  : key;
          Ref v_ref =
              used_strided_qkv
                  ? Ref{concat_out.buffer,
                        concat_out.offset + mul(2 * q + kv, 2U),
                        concat_out.bytes - mul(2 * q + kv, 2U)}
                  : value;
          const U32 q_tok_stride = used_strided_qkv ? qkv_stride : (2 * q);
          const U32 k_tok_stride = used_strided_qkv ? qkv_stride : kv;
          const U32 v_tok_stride = used_strided_qkv ? qkv_stride : kv;

          std::array<Ref, 8> pt{};
          for (std::size_t i = 0; i < 8; ++i) {
            const auto &tb =
                skey.slots[i < B ? i : 0]->page_pool->table_buffer();
            pt[i] = Ref{&tb, 0, tb.size()};
          }
          const auto &s0_blk = skey.slots[0]->blocks[layer_index];
          emit_batched_attention_affine4_g128(
              e, c, U32(B), total_m, max_w, all_w_eq_1,
              use_batched_affine4_flash, batched_affine4_split,
              skey.slots[0]->max_context,
              skey.slots[0]->page_pool->physical_pages(), q_tok_stride,
              k_tok_stride, v_tok_stride, normalization_weight_offset, q_ref,
              gate_ref, k_ref, v_ref, direct(block.get("q_norm"), hd),
              direct(block.get("k_norm"), hd), s0_blk.state0, s0_blk.state1,
              mixer, attention_positions, pt, row_offs_8, widths_8, split_max,
              split_sum, split_output);
        } else {
          for (std::size_t b_idx = 0; b_idx < B; ++b_idx) {
            const auto &sp = *skey.slots[b_idx];
            const U32 r_off = row_offs[b_idx];
            const U32 w_b = skey.widths[b_idx];
            std::vector<std::pair<U64, U32>> raw_patches;
            Ref slice_attn_pos =
                attention_positions.buffer
                    ? row_slice_ref(attention_positions, r_off, w_b, 1U, 4U)
                    : Ref{};
            Ref q_s = used_strided_qkv
                          ? Ref{concat_out.buffer, 0, mul(2 * q, 2)}
                          : row_slice_ref(query, r_off, w_b, 2 * q);
            Ref k_s = used_strided_qkv
                          ? Ref{concat_out.buffer, mul(2 * q, 2), mul(kv, 2)}
                          : row_slice_ref(key, r_off, w_b, kv);
            Ref v_s =
                used_strided_qkv
                    ? Ref{concat_out.buffer, mul(2 * q + kv, 2), mul(kv, 2)}
                    : row_slice_ref(value, r_off, w_b, kv);
            const bool slot_split =
                (skey.bf16_split_env > 0U) &&
                (sp.kv_format_ == KvStorageFormat::BFloat16) &&
                split_attention_enabled;
            const U32 slot_env =
                slot_split ? std::min(skey.bf16_split_env, sp.max_context) : 0U;
            emit_attention_state_step(
                e, block, sp.blocks[layer_index], *sp.page_pool, sp.max_context,
                sp.kv_format_, q_s, k_s, v_s,
                row_slice_ref(mixer, r_off, w_b, q), slice_attn_pos, w_b,
                slot_split, slot_env, false, 0xFF000000U | U32(b_idx),
                &raw_patches);
            add_slot_patches(U32(b_idx), raw_patches);
          }
        }

        if (layer_fuse_t1 &&
            try_linear_add_sq_t1(e, block.get("attention_o"), mixer, hidden, h,
                                 q, 2U * layer_index)) {
          pending_sq_slot = 2U * layer_index;
        } else if (layer_fuse_m2 &&
                   try_linear_add_sq_m2(e, block.get("attention_o"), mixer,
                                        hidden, h, q,
                                        (2U * layer_index) * m2_sq_mul)) {
          pending_sq_slot = 2U * layer_index;
        } else if (use_m2 &&
                   try_linear_m2(e, block.get("attention_o"), mixer, delta, h,
                                 q)) {
        } else {
          linear(e, block.get("attention_o"), mixer, delta, h, q, total_m);
        }
      } else {
        const std::array<MaterializedWeightView, 2> qkvz_parts{
            block.get("gdn_qkv"), block.get("gdn_z")};
        const std::array<U32, 2> qkvz_ns{conv, gv};
        const std::array<MaterializedWeightView, 2> ab_parts{
            block.get("gdn_a"), block.get("gdn_b")};
        const std::array<U32, 2> ab_ns{hv, hv};
        const std::array<MaterializedWeightView, 4> qkvzab_parts{
            qkvz_parts[0], qkvz_parts[1], ab_parts[0], ab_parts[1]};
        const std::array<U32, 4> qkvzab_ns{conv, gv, hv, hv};
        const auto *gdn_bypass =
            find_bypass_group(layer_index, QwenBypassGroupKind::gdn_inputs);

        Ref qkv_in = qkv, z_in = z, a_in = a, b_in = b;
        Ref ab_strided_base{};
        U32 qkv_stride = conv, z_stride = gv, a_stride = hv, b_stride = hv;
        bool qkvzab_done = false;

        if (layer_fuse_t1 && pending_sq_slot.has_value()) {
          const U32 sq_in = *pending_sq_slot;
          pending_sq_slot.reset();
          Ref qz_c0{}, qz_s0{}, qz_c1{}, qz_s1{}, qz_h0{}, qz_h1{};
          Ref ab_c0{}, ab_s0{}, ab_c1{}, ab_s1{}, ab_h0{}, ab_h1{};
          if (gdn_bypass == nullptr && conv < 65536U && gv < 65536U &&
              hv < 65536U && qkvz_parts[0].row_count == conv &&
              qkvz_parts[1].row_count == gv && ab_parts[0].row_count == hv &&
              ab_parts[1].row_count == hv && qkvz_parts[0].in_features == h &&
              qkvz_parts[1].in_features == h && ab_parts[0].in_features == h &&
              ab_parts[1].in_features == h) {
            if (resolve_pretiled_mxfp4_t1(qkvz_parts[0], 1U, &qz_c0, &qz_s0) &&
                resolve_pretiled_mxfp4_t1(qkvz_parts[1], 1U, &qz_c1, &qz_s1) &&
                resolve_pretiled_mxfp4_t1(ab_parts[0], 1U, &ab_c0, &ab_s0) &&
                resolve_pretiled_mxfp4_t1(ab_parts[1], 1U, &ab_c1, &ab_s1)) {
              const Ref nw_ref = direct(block.get("input_norm"), h);
              const bool use_r4 = ((conv + gv + 2U * hv) >= 2048U);
              const U32 rptg = use_r4 ? 8U : 4U;
              const U32 tgs = ((conv + rptg - 1U) / rptg) +
                              ((gv + rptg - 1U) / rptg) +
                              2U * ((hv + rptg - 1U) / rptg);
              const char *fn =
                  use_r4 ? "omlx_m5_pretiled_qmv_mxfp4_concat4_r4_norm_t1"
                         : "omlx_m5_pretiled_qmv_mxfp4_concat4_r2_norm_t1";
              auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                               {.width = 64U},
                               {hidden, nw_ref, rms_sq_buf, qz_c0, qz_s0, qz_c1,
                                qz_s1, ab_c0, ab_s0, ab_c1, ab_s1, qkv, z, a,
                                b});
              e.dynamic(d, 15,
                        fused_norm_params(conv, gv, hv, hv, h, 0U, sq_in));
              for (std::size_t i = 0; i < 4; ++i)
                e.record_weight(qkvzab_parts[i], qkvzab_ns[i], h,
                                "pretiled_qmv_t1",
                                "pretiled_mxfp4_qmv_concat4_norm_t1_v4", {},
                                "legacy_route_v1", {}, false);
              qkvzab_done = true;
            } else if (bool lv0 = false, lv1 = false, lv2 = false, lv3 = false;
                       resolve_pretiled_affine_t1(qkvz_parts[0], 1U, &qz_c0,
                                                  &qz_h0, &qz_s0, &lv0) &&
                       resolve_pretiled_affine_t1(qkvz_parts[1], 1U, &qz_c1,
                                                  &qz_h1, &qz_s1, &lv1) &&
                       resolve_pretiled_affine_t1(ab_parts[0], 1U, &ab_c0,
                                                  &ab_h0, &ab_s0, &lv2) &&
                       resolve_pretiled_affine_t1(ab_parts[1], 1U, &ab_c1,
                                                  &ab_h1, &ab_s1, &lv3) &&
                       lv0 == lv1 && lv1 == lv2 && lv2 == lv3) {
              const Ref nw_ref = direct(block.get("input_norm"), h);
              const U32 b0 = affine_nax_bits(qkvz_parts[0].format);
              const U32 b1 = affine_nax_bits(qkvz_parts[1].format);
              const U32 b2 = affine_nax_bits(ab_parts[0].format);
              const U32 b3 = affine_nax_bits(ab_parts[1].format);
              const U32 packed_bits =
                  b0 | (b1 << 8U) | (b2 << 16U) | (b3 << 24U);
              const bool use_r4 = ((conv + gv + 2U * hv) >= 2048U);
              const U32 rptg = use_r4 ? 8U : 4U;
              const U32 tgs = ((conv + rptg - 1U) / rptg) +
                              ((gv + rptg - 1U) / rptg) +
                              2U * ((hv + rptg - 1U) / rptg);
              const bool all_q456 = (b0 >= 4U && b0 <= 6U) &&
                                    (b1 >= 4U && b1 <= 6U) &&
                                    (b2 >= 4U && b2 <= 6U) &&
                                    (b3 >= 4U && b3 <= 6U);
              const char *fn =
                  all_q456
                      ? (use_r4
                             ? "omlx_m5_pretiled_qmv_affine_concat4_q456_r4_norm_t1"
                             : "omlx_m5_pretiled_qmv_affine_concat4_q456_r2_norm_t1")
                      : (use_r4
                             ? "omlx_m5_pretiled_qmv_affine_concat4_r4_norm_t1"
                             : "omlx_m5_pretiled_qmv_affine_concat4_r2_norm_t1");
              auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                               {.width = 64U},
                               {hidden, nw_ref, rms_sq_buf, qz_c0, qz_h0, qz_s0,
                                qz_c1, qz_h1, qz_s1, ab_c0, ab_h0, ab_s0, ab_c1,
                                ab_h1, ab_s1, qkv, z, a, b});
              e.dynamic(d, 19,
                        fused_norm_params(conv, gv, hv, hv,
                                          pretiled_k_flags(h, lv0, false),
                                          packed_bits, sq_in));
              for (std::size_t i = 0; i < 4; ++i)
                e.record_weight(qkvzab_parts[i], qkvzab_ns[i], h,
                                "pretiled_qmv_t1",
                                "pretiled_affine_qmv_concat4_norm_t1_v4", {},
                                "legacy_route_v1", {}, false);
              qkvzab_done = true;
            }
          }
          if (!qkvzab_done)
            normalize(e, block.get("input_norm"), hidden, norm, total_m);
        } else if (layer_fuse_m2 && pending_sq_slot.has_value()) {
          const U32 sq_in = *pending_sq_slot;
          pending_sq_slot.reset();
          qkvzab_done =
              try_concat4_m2(e, qkvzab_parts, qkvzab_ns, h, hidden, qkv, z, a,
                             b, &block.get("input_norm"), sq_in * m2_sq_mul);
          if (!qkvzab_done)
            normalize(e, block.get("input_norm"), hidden, norm, total_m);
        }

        if (!qkvzab_done && single_t1_tdom && !in_f16) {
          Ref qz_c0{}, qz_s0{}, qz_c1{}, qz_s1{}, qz_h0{}, qz_h1{};
          Ref ab_c0{}, ab_s0{}, ab_c1{}, ab_s1{}, ab_h0{}, ab_h1{};
          if (qkvz_parts[0].row_count == conv &&
              qkvz_parts[1].row_count == gv && ab_parts[0].row_count == hv &&
              ab_parts[1].row_count == hv && qkvz_parts[0].in_features == h &&
              qkvz_parts[1].in_features == h && ab_parts[0].in_features == h &&
              ab_parts[1].in_features == h) {
            if (resolve_pretiled_mxfp4_t1(qkvz_parts[0], 1U, &qz_c0, &qz_s0) &&
                resolve_pretiled_mxfp4_t1(qkvz_parts[1], 1U, &qz_c1, &qz_s1) &&
                resolve_pretiled_mxfp4_t1(ab_parts[0], 1U, &ab_c0, &ab_s0) &&
                resolve_pretiled_mxfp4_t1(ab_parts[1], 1U, &ab_c1, &ab_s1)) {
              const bool use_r4 = ((conv + gv + 2U * hv) >= 2048U);
              const U32 rptg = use_r4 ? 8U : 4U;
              const U32 tgs = ((conv + rptg - 1U) / rptg) +
                              ((gv + rptg - 1U) / rptg) +
                              2U * ((hv + rptg - 1U) / rptg);
              const char *fn =
                  use_r4 ? "omlx_m5_pretiled_qmv_mxfp4_concat4_r4_t1"
                         : "omlx_m5_pretiled_qmv_mxfp4_concat4_r2_t1";
              auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                               {.width = 64U},
                               {norm, qz_c0, qz_s0, qz_c1, qz_s1, ab_c0, ab_s0,
                                ab_c1, ab_s1, qkv, z, a, b});
              e.dynamic(d, 13, words({conv, gv, hv, hv, h}));
              for (std::size_t i = 0; i < 4; ++i)
                e.record_weight(qkvzab_parts[i], qkvzab_ns[i], h,
                                "pretiled_qmv_t1",
                                "pretiled_mxfp4_qmv_concat4_t1_v3", {},
                                "legacy_route_v1", {}, false);
              qkvzab_done = true;
            } else if (bool lv0 = false, lv1 = false, lv2 = false, lv3 = false;
                       resolve_pretiled_affine_t1(qkvz_parts[0], 1U, &qz_c0,
                                                  &qz_h0, &qz_s0, &lv0) &&
                       resolve_pretiled_affine_t1(qkvz_parts[1], 1U, &qz_c1,
                                                  &qz_h1, &qz_s1, &lv1) &&
                       resolve_pretiled_affine_t1(ab_parts[0], 1U, &ab_c0,
                                                  &ab_h0, &ab_s0, &lv2) &&
                       resolve_pretiled_affine_t1(ab_parts[1], 1U, &ab_c1,
                                                  &ab_h1, &ab_s1, &lv3) &&
                       lv0 == lv1 && lv1 == lv2 && lv2 == lv3) {
              const U32 b0 = affine_nax_bits(qkvz_parts[0].format);
              const U32 b1 = affine_nax_bits(qkvz_parts[1].format);
              const U32 b2 = affine_nax_bits(ab_parts[0].format);
              const U32 b3 = affine_nax_bits(ab_parts[1].format);
              const U32 packed_bits =
                  b0 | (b1 << 8U) | (b2 << 16U) | (b3 << 24U);
              const bool use_r4 = ((conv + gv + 2U * hv) >= 2048U);
              const U32 rptg = use_r4 ? 8U : 4U;
              const U32 tgs = ((conv + rptg - 1U) / rptg) +
                              ((gv + rptg - 1U) / rptg) +
                              2U * ((hv + rptg - 1U) / rptg);
              const bool all_q456 = (b0 >= 4U && b0 <= 6U) &&
                                    (b1 >= 4U && b1 <= 6U) &&
                                    (b2 >= 4U && b2 <= 6U) &&
                                    (b3 >= 4U && b3 <= 6U);
              const char *fn =
                  all_q456
                      ? (use_r4
                             ? "omlx_m5_pretiled_qmv_affine_concat4_q456_r4_t1"
                             : "omlx_m5_pretiled_qmv_affine_concat4_q456_r2_t1")
                      : (use_r4
                             ? "omlx_m5_pretiled_qmv_affine_concat4_r4_t1"
                             : "omlx_m5_pretiled_qmv_affine_concat4_r2_t1");
              auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                               {.width = 64U},
                               {norm, qz_c0, qz_h0, qz_s0, qz_c1, qz_h1, qz_s1,
                                ab_c0, ab_h0, ab_s0, ab_c1, ab_h1, ab_s1, qkv,
                                z, a, b});
              e.dynamic(d, 17,
                        words({conv, gv, hv, hv, pretiled_k_flags(h, lv0, false),
                               packed_bits}));
              for (std::size_t i = 0; i < 4; ++i)
                e.record_weight(qkvzab_parts[i], qkvzab_ns[i], h,
                                "pretiled_qmv_t1",
                                "pretiled_affine_qmv_concat4_t1_v3", {},
                                "legacy_route_v1", {}, false);
              qkvzab_done = true;
            }
          }
        }
        if (!qkvzab_done && use_m2) {
          qkvzab_done = try_concat4_m2(e, qkvzab_parts, qkvzab_ns, h, norm, qkv,
                                       z, a, b);
        }
        if (!qkvzab_done) {
          const bool direct_strided_gdn =
              can_batched_gdn_t1 || can_batched_gdn_verify ||
              (total_m == 1U && !inspect_tensors_workspace);
          if (try_concat_nax(e, block.concat_gdn_qkvz_id, qkvz_parts, qkvz_ns,
                             h, in_f16, norm, concat_out, total_m)) {
            if (direct_strided_gdn) {
              qkv_in = concat_out;
              z_in = Ref{concat_out.buffer, concat_out.offset + mul(conv, 2U),
                         concat_out.bytes - mul(conv, 2U)};
              qkv_stride = conv + gv;
              z_stride = conv + gv;
            } else {
              emit_split2(e, concat_out, qkv, z, conv, gv, total_m);
            }
          } else if (in_f16) {
            linear_from_f16(e, block.get("gdn_qkv"), f16_norm, f16_ones_scale,
                            qkv, conv, h, total_m);
            linear_from_f16(e, block.get("gdn_z"), f16_norm, f16_ones_scale, z,
                            gv, h, total_m);
          } else if (use_m2) {
            if (!try_linear_m2_base(e, block.get("gdn_qkv"), norm, qkv, conv, h))
              linear_base(e, block.get("gdn_qkv"), norm, qkv, conv, h, total_m);
            if (!try_linear_m2_base(e, block.get("gdn_z"), norm, z, gv, h))
              linear_base(e, block.get("gdn_z"), norm, z, gv, h, total_m);
          } else {
            linear_base(e, block.get("gdn_qkv"), norm, qkv, conv, h, total_m);
            linear_base(e, block.get("gdn_z"), norm, z, gv, h, total_m);
          }

          const U64 ab_bytes = mul(mul(total_m, 2U * hv), 2U);
          Ref ab_dst = concat_out;
          bool ab_direct_strided = false;
          if (can_batched_gdn_t1 && convolved.buffer &&
              convolved.bytes >= ab_bytes) {
            ab_dst = convolved;
            ab_direct_strided = true;
          } else if ((can_batched_gdn_verify ||
                      (total_m == 1U && !inspect_tensors_workspace)) &&
                     query.buffer && query.bytes >= ab_bytes) {
            ab_dst = query;
            ab_direct_strided = true;
          } else if (qkv_in.buffer == concat_out.buffer) {
            ab_dst = Ref{};
          }
          if (ab_dst.buffer &&
              try_concat_nax(e, block.concat_gdn_ab_id, ab_parts, ab_ns, h,
                             in_f16, norm, ab_dst, total_m)) {
            if (ab_direct_strided) {
              ab_strided_base = ab_dst;
              a_in = ab_dst;
              b_in = Ref{ab_dst.buffer, ab_dst.offset + mul(hv, 2U),
                         ab_dst.bytes - mul(hv, 2U)};
              a_stride = 2U * hv;
              b_stride = 2U * hv;
            } else {
              emit_split2(e, ab_dst, a, b, hv, hv, total_m);
            }
          } else if (in_f16) {
            linear_from_f16(e, block.get("gdn_a"), f16_norm, f16_ones_scale, a,
                            hv, h, total_m);
            linear_from_f16(e, block.get("gdn_b"), f16_norm, f16_ones_scale, b,
                            hv, h, total_m);
          } else if (use_m2) {
            if (!try_linear_m2_base(e, block.get("gdn_a"), norm, a, hv, h))
              linear_base(e, block.get("gdn_a"), norm, a, hv, h, total_m);
            if (!try_linear_m2_base(e, block.get("gdn_b"), norm, b, hv, h))
              linear_base(e, block.get("gdn_b"), norm, b, hv, h, total_m);
          } else {
            linear_base(e, block.get("gdn_a"), norm, a, hv, h, total_m);
            linear_base(e, block.get("gdn_b"), norm, b, hv, h, total_m);
          }
        }
        if (gdn_bypass != nullptr) {
          const bool qz_strided = (qkv_in.buffer == concat_out.buffer);
          const bool ab_strided = (ab_strided_base.buffer != nullptr);
          const std::array<detail::lora_rt::BypassPartDest, 4> dests{{
              qz_strided
                  ? detail::lora_rt::BypassPartDest{concat_out, conv + gv, 0U}
                  : detail::lora_rt::BypassPartDest{qkv, conv, 0U},
              qz_strided
                  ? detail::lora_rt::BypassPartDest{concat_out, conv + gv, conv}
                  : detail::lora_rt::BypassPartDest{z, gv, 0U},
              ab_strided
                  ? detail::lora_rt::BypassPartDest{ab_strided_base, 2U * hv,
                                                    0U}
                  : detail::lora_rt::BypassPartDest{a, hv, 0U},
              ab_strided
                  ? detail::lora_rt::BypassPartDest{ab_strided_base, 2U * hv,
                                                    hv}
                  : detail::lora_rt::BypassPartDest{b, hv, 0U},
          }};
          detail::lora_rt::emit_bypass_sibling_group(
              e, *gdn_bypass, norm, dests, lora_bypass_tmp_,
              bf16_nax_bypass_ok_, silicon_tuning.gpu_cores, total_m);
        }

        const auto &out_w = block.get("gdn_out");
        const auto *gdn_out_nax =
            (!use_m2 && !single_t1_tdom && !has_bypass_lora(out_w) &&
             use_mxfp4_nax_for_weight(out_w, total_m))
                ? mxfp4_nax_cache->find(out_w.object_id)
                : nullptr;
        const bool can_gdn_out_f16 =
            (!out_w.has_fwht64() && gdn_out_nax != nullptr &&
             gdn_out_nax->n == h && gdn_out_nax->k == gv &&
             mxfp4_nax_select_wide(*gdn_out_nax, total_m, total_m,
                                   silicon_tuning.gpu_cores, false) ==
                 NaxWidePrefillTile::none);
        Ref conv_w = direct(block.get("gdn_conv"),
                            U64(conv) * config.linear_conv_kernel_dim);
        Ref a_log = direct(block.get("gdn_a_log"), hv);
        Ref dt_bias = direct(block.get("gdn_dt_bias"), hv);
        Ref gdn_norm = direct(block.get("gdn_norm"), vd);

        if (can_batched_gdn_t1) {
          std::array<Ref, 8> hs{}, sts{};
          for (std::size_t i = 0; i < 8; ++i) {
            const auto &sb = skey.slots[i < B ? i : 0]->blocks[layer_index];
            hs[i] = sb.state1;
            sts[i] = sb.state0;
          }
          const U32 mp = can_gdn_out_f16 ? mxfp4_nax_mp(total_m) : 0U;
          auto t1_params = words(
              {kh, hv, kd, vd, conv, U32(config.linear_conv_kernel_dim), U32(B),
               word(float(config.rms_norm_eps)), qkv_stride, z_stride, a_stride,
               b_stride, mp});
          auto &fused = e.emit(
              "gdn_layer",
              can_gdn_out_f16
                  ? "omlx_m5_gdn_t1_batched_conv_recurrent_norm_gate_f16"
                  : "omlx_m5_gdn_t1_batched_conv_recurrent_norm_gate",
              {.width = 256U, .height = hv, .depth = U64(B)},
              {.width = 256U, .height = 1U, .depth = 1U},
              {qkv_in, conv_w, a_in, b_in, a_log, dt_bias, z_in, gdn_norm,
               can_gdn_out_f16 ? f16_scratch : mixer, hs[0], hs[1], hs[2],
               hs[3], hs[4], hs[5], hs[6], hs[7], sts[0], sts[1], sts[2],
               sts[3], sts[4], sts[5], sts[6], sts[7]});
          e.dynamic(fused, 25, t1_params);
          auto &commit_h = e.emit(
              "gdn_layer", "omlx_m5_gdn_commit_history_batched_vec4_t1",
              {.width = conv / 4U, .height = U64(B)}, {.width = 256U},
              {qkv_in, hs[0], hs[1], hs[2], hs[3], hs[4], hs[5], hs[6], hs[7]});
          e.dynamic(commit_h, 9, std::move(t1_params));
          e.profile.gdn_row_exact_eligible = true;
          if (can_gdn_out_f16) {
            linear_from_f16(e, out_w, f16_scratch, f16_ones_scale, delta, h, gv,
                            total_m);
          } else if (layer_fuse_m2 &&
                     try_linear_add_sq_m2(e, out_w, mixer, hidden, h, gv,
                                          (2U * layer_index) * m2_sq_mul)) {
            pending_sq_slot = 2U * layer_index;
          } else if (use_m2 && try_linear_m2(e, out_w, mixer, delta, h, gv)) {
          } else {
            linear(e, out_w, mixer, delta, h, gv, total_m);
          }
        } else if (can_batched_gdn_verify) {
          std::array<Ref, 8> hs{}, rqks{};
          for (std::size_t i = 0; i < 8; ++i) {
            const auto &sb = skey.slots[i < B ? i : 0]->blocks[layer_index];
            hs[i] = sb.state1;
            rqks[i] = sb.replay_qkv;
          }
          const U32 mp = can_gdn_out_f16 ? mxfp4_nax_mp(total_m) : 0U;
          auto make_verify_params = [&](U32 sbase) {
            std::vector<U32> vw{kh,
                                hv,
                                kd,
                                vd,
                                conv,
                                U32(config.linear_conv_kernel_dim),
                                U32(B),
                                total_m,
                                word(float(config.rms_norm_eps)),
                                qkv_stride,
                                z_stride,
                                a_stride,
                                b_stride,
                                sbase,
                                mp,
                                0U};
            vw.insert(vw.end(), row_offs_8.begin(), row_offs_8.end());
            vw.insert(vw.end(), widths_8.begin(), widths_8.end());
            return words_span(vw);
          };
          auto vp0 = make_verify_params(0U);
          auto &conv_vs = e.emit(
              "gdn_layer", "omlx_m5_gdn_conv_silu_vec4_t8_verify_save_batched",
              {.width = (conv + 3U) / 4U,
               .height = (max_w + 7U) / 8U,
               .depth = U64(B)},
              {.width = 64U, .height = 1U, .depth = 1U},
              {qkv_in, conv_w, convolved, hs[0], hs[1], hs[2], hs[3], hs[4],
               hs[5], hs[6], hs[7], rqks[0], rqks[1], rqks[2], rqks[3], rqks[4],
               rqks[5], rqks[6], rqks[7]});
          e.dynamic(conv_vs, 19, vp0);

          for (U32 sbase = 0U; sbase < U32(B); sbase += 4U) {
            const U32 sub_b = std::min<U32>(4U, U32(B) - sbase);
            std::array<const Block *, 4> sb4{};
            for (U32 j = 0U; j < 4U; ++j) {
              const U32 si = (sbase + j < U32(B)) ? (sbase + j) : sbase;
              sb4[j] = &skey.slots[si]->blocks[layer_index];
            }
            auto &rec = e.emit(
                "gdn_layer", "omlx_m5_gdn_recurrent_record_4row_batched4",
                {.width = 32U, .height = gv / 4U, .depth = U64(sub_b)},
                {.width = 32U, .height = 1U, .depth = 1U},
                {convolved,      a_in,           b_in,
                 a_log,          dt_bias,        recurrent,
                 sb4[0]->state0, sb4[0]->replay_keys, sb4[0]->replay_decays,
                 sb4[0]->replay_deltas,
                 sb4[1]->state0, sb4[1]->replay_keys, sb4[1]->replay_decays,
                 sb4[1]->replay_deltas,
                 sb4[2]->state0, sb4[2]->replay_keys, sb4[2]->replay_decays,
                 sb4[2]->replay_deltas,
                 sb4[3]->state0, sb4[3]->replay_keys, sb4[3]->replay_decays,
                 sb4[3]->replay_deltas});
            if (sbase > 0U)
              rec.barrier_before = false;
            e.dynamic(rec, 22, sbase == 0U ? vp0 : make_verify_params(sbase));
          }

          if (can_gdn_out_f16) {
            auto &ng = e.emit("gdn_layer", "omlx_m5_gdn_norm_gate_f16_strided",
                              {.width = 32U, .height = U64(mp) * hv},
                              {.width = 32U},
                              {recurrent, z_in, gdn_norm, f16_scratch});
            e.dynamic(ng, 4, std::move(vp0));
            linear_from_f16(e, out_w, f16_scratch, f16_ones_scale, delta, h, gv,
                            total_m);
          } else {
            auto &ng = e.emit("gdn_layer", "omlx_m5_gdn_norm_gate_strided",
                              {.width = 32U, .height = U64(total_m) * hv},
                              {.width = 32U},
                              {recurrent, z_in, gdn_norm, mixer});
            e.dynamic(ng, 4, std::move(vp0));
            linear(e, out_w, mixer, delta, h, gv, total_m);
          }
          e.profile.gdn_row_exact_eligible = true;
        } else if (single_t1_tdom) {
          const auto &sp = *skey.slots[0];
          bool gdn_fused_sq = false;
          emit_gdn_state_step(
              e, block, sp.blocks[layer_index], qkv_in, z_in, a_in, b_in,
              convolved, recurrent, mixer, 1U, false, false, true,
              sp.max_context,
              layer_fuse_t1 ? std::optional<U32>{2U * layer_index}
                            : std::nullopt,
              &gdn_fused_sq);
          if (gdn_fused_sq)
            pending_sq_slot = 2U * layer_index;
        } else {
          for (std::size_t b_idx = 0; b_idx < B; ++b_idx) {
            const auto &sp = *skey.slots[b_idx];
            const U32 r_off = row_offs[b_idx];
            const U32 w_b = skey.widths[b_idx];
            emit_gdn_state_step(
                e, block, sp.blocks[layer_index],
                row_slice_ref(qkv_in, r_off, w_b, conv),
                row_slice_ref(z_in, r_off, w_b, gv),
                row_slice_ref(a_in, r_off, w_b, hv),
                row_slice_ref(b_in, r_off, w_b, hv),
                row_slice_ref(convolved, r_off, w_b, conv),
                row_slice_ref(recurrent, r_off, w_b, gv),
                row_slice_ref(mixer, r_off, w_b, gv), w_b, skey.verify, false,
                false, sp.max_context);
          }
          if (layer_fuse_m2 &&
              try_linear_add_sq_m2(e, out_w, mixer, hidden, h, gv,
                                   (2U * layer_index) * m2_sq_mul)) {
            pending_sq_slot = 2U * layer_index;
          } else if (use_m2 && try_linear_m2(e, out_w, mixer, delta, h, gv)) {
          } else {
            linear(e, out_w, mixer, delta, h, gv, total_m);
          }
        }
      }

      e.begin_stage("layer." + std::to_string(layer_index) + ".mlp");
      if (!pending_sq_slot.has_value()) {
        if (mlp_f16) {
          add_normalize_f16(e, block.get("post_attention_norm"), hidden, delta,
                            f16_norm, total_m);
        } else if (mlp_affine_xs) {
          emit_affine_add_rmsnorm_xsum(
              e, hidden, delta, direct(block.get("post_attention_norm"), h),
              norm, affine_nax_xs, h, float(config.rms_norm_eps),
              normalization_weight_offset, total_m);
        } else {
          add_normalize(e, block.get("post_attention_norm"), hidden, delta,
                        norm, total_m);
        }
      }

      if (single_t1_tdom) {
        mlp(e, block, layer_index, false, 1U,
            layer_fuse_t1 ? &pending_sq_slot : nullptr,
            layer_fuse_t1 ? std::optional<U32>{2U * layer_index + 1U}
                          : std::nullopt);
      } else if (use_m2 && !config.num_experts) {
        bool sw_done = false;
        if (pending_sq_slot.has_value()) {
          const U32 sq_in = *pending_sq_slot;
          pending_sq_slot.reset();
          sw_done = try_linear_swiglu_m2(
              e, block.get("mlp_gate"), block.get("mlp_up"), hidden, product, f,
              h, &block.get("post_attention_norm"), sq_in);
          if (!sw_done)
            normalize(e, block.get("post_attention_norm"), hidden, norm, 2U);
        }
        if (!sw_done &&
            !try_linear_swiglu_m2(e, block.get("mlp_gate"), block.get("mlp_up"),
                                  norm, product, f, h)) {
          if (!try_linear_swiglu(e, block.get("mlp_gate"), block.get("mlp_up"),
                                 norm, product, f, h, 2U)) {
            if (!try_linear_m2_base(e, block.get("mlp_gate"), norm, gate, f, h))
              linear_base(e, block.get("mlp_gate"), norm, gate, f, h, 2U);
            if (!try_linear_m2_base(e, block.get("mlp_up"), norm, up, f, h))
              linear_base(e, block.get("mlp_up"), norm, up, f, h, 2U);
            if (const auto *mlp_gu_bypass = find_bypass_group(
                    layer_index, QwenBypassGroupKind::mlp_gate_up);
                mlp_gu_bypass != nullptr) {
              const std::array<detail::lora_rt::BypassPartDest, 2> dests{{
                  {gate, f, 0U},
                  {up, f, 0U},
              }};
              detail::lora_rt::emit_bypass_sibling_group(
                  e, *mlp_gu_bypass, norm, dests, lora_bypass_tmp_,
                  bf16_nax_bypass_ok_, silicon_tuning.gpu_cores, 2U);
            }
            swiglu(e, 2U, f, 2U);
          }
        }
        if (layer_fuse_m2 &&
            try_linear_add_sq_m2(e, block.get("mlp_down"), product, hidden, h,
                                 f, 2U * layer_index + 1U)) {
          pending_sq_slot = 2U * layer_index + 1U;
        } else if (!try_linear_m2(e, block.get("mlp_down"), product, delta, h,
                                  f)) {
          linear(e, block.get("mlp_down"), product, delta, h, f, 2U);
        }
      } else {
        mlp(e, block, layer_index, mlp_f16, total_m,
            (allow_moe9_bt && layer_fuse_m2) ? &pending_sq_slot : nullptr,
            (allow_moe9_bt && layer_fuse_m2)
                ? std::optional<U32>{2U * layer_index + 1U}
                : std::nullopt,
            allow_moe9_bt);
      }

      if (!pending_sq_slot.has_value()) {
        if (layer_index + 1 < blocks.size()) {
          const bool next_in_f16 =
              (!use_m2 && !single_t1_tdom &&
               block_input_nax(blocks[layer_index + 1], total_m, total_m));
          const bool next_in_affine_xs = (!next_in_f16 && affine_xs_eligible);
          if (next_in_f16) {
            add_normalize_f16(e, blocks[layer_index + 1].get("input_norm"),
                              hidden, delta, f16_norm, total_m);
          } else if (next_in_affine_xs) {
            emit_affine_add_rmsnorm_xsum(
                e, hidden, delta,
                direct(blocks[layer_index + 1].get("input_norm"), h), norm,
                affine_nax_xs, h, float(config.rms_norm_eps),
                normalization_weight_offset, total_m);
            carried_affine_xs = make_norm_xs_key(total_m);
          } else {
            add_normalize(e, blocks[layer_index + 1].get("input_norm"), hidden,
                          delta, norm, total_m);
          }
        } else if (skey.verify) {
          if (verify_head_f16) {
            add_normalize_f16(e, final_norm, hidden, delta, f16_norm, total_m);
          } else if (verify_head_affine_xs) {
            emit_affine_add_rmsnorm_xsum(
                e, hidden, delta, direct(final_norm, h), norm, affine_nax_xs, h,
                float(config.rms_norm_eps), normalization_weight_offset,
                total_m);
            carried_affine_xs = make_norm_xs_key(total_m);
          } else {
            add_normalize(e, final_norm, hidden, delta, norm, total_m);
          }
        } else {
          add(e, total_m);
        }
      }

      for (U32 tap = 0; tap < feature_layers.size(); ++tap) {
        if (feature_layers[tap] != layer_index)
          continue;
        auto &cap_d = e.emit(
            "qwen_forward", "omlx_m5_forward_capture",
            {.width = U64(total_m) * h}, {.width = 128}, {hidden, features});
        e.dynamic(cap_d, 2,
                  words({total_m, h, U32(feature_layers.size()), tap}));
      }
      ++layer_index;
    }

    // 3. Per-slot target feature scatter + trunk hidden scatter + lm_head (+ optional in-ICB sample).
    e.delegated_prefill = false;
    e.begin_stage("head");
    restore_carried_xs();
    if (!feature_layers.empty() && features.buffer) {
      const U32 feat_w = h * U32(feature_layers.size());
      for (std::size_t b_idx = 0; b_idx < B; ++b_idx) {
        const auto &sp = *skey.slots[b_idx];
        const U32 r_off = row_offs[b_idx];
        const U32 w_b = skey.widths[b_idx];
        Ref src_feat = row_slice_ref(features, r_off, w_b, feat_w);
        if (skey.verify) {
          if (sp.features.buffer && &sp != this) {
            auto &cp = e.emit("qwen_forward", "omlx_m5_forward_copy_rows",
                              {.width = U64(w_b) * feat_w}, {.width = 128},
                              {src_feat, sp.features});
            e.dynamic(cp, 2, words({w_b, feat_w, 0U}));
          }
        } else if (sp.feature_history.buffer) {
          const U32 pos_ph = 0xFF000000U | U32(b_idx);
          auto &ring_cp =
              e.emit("qwen_forward", "omlx_m5_forward_copy_rows_ring",
                     {.width = U64(w_b) * feat_w}, {.width = 128},
                     {src_feat, sp.feature_history});
          const U64 off_ring = e.dynamic(
              ring_cp, 2, words({w_b, feat_w, pos_ph, sp.feature_ring_rows_}));
          pos_patches.push_back({U32(b_idx), off_ring, 2U});
        }
      }
      if (skey.verify) {
        for (std::size_t b_idx = 0; b_idx < B; ++b_idx) {
          const auto &sp = *skey.slots[b_idx];
          const U32 r_off = row_offs[b_idx];
          const U32 w_b = skey.widths[b_idx];
          if (&sp == this && r_off != 0U && sp.features.buffer) {
            const U64 need = mul(U64(w_b), mul(feat_w, 2U));
            if (!packed_verify_features.buffer ||
                packed_verify_features.bytes < need) {
              const U64 alloc_bytes =
                  std::max(need, mul(8U, mul(feat_w, 2U)));
              auto pf_buf = std::make_unique<MetalBuffer>(
                  context.make_private_buffer(alloc_bytes));
              packed_verify_features = Ref{pf_buf.get(), 0, alloc_bytes};
              buffers.push_back(std::move(pf_buf));
            }
            Ref src_feat = row_slice_ref(features, r_off, w_b, feat_w);
            Ref tmp_feat =
                row_slice_ref(packed_verify_features, 0U, w_b, feat_w);
            auto &cp1 = e.emit("qwen_forward", "omlx_m5_forward_copy_rows",
                               {.width = U64(w_b) * feat_w}, {.width = 128},
                               {src_feat, tmp_feat});
            e.dynamic(cp1, 2, words({w_b, feat_w, 0U}));
            auto &cp2 = e.emit("qwen_forward", "omlx_m5_forward_copy_rows",
                               {.width = U64(w_b) * feat_w}, {.width = 128},
                               {tmp_feat, sp.features});
            e.dynamic(cp2, 2, words({w_b, feat_w, 0U}));
          }
        }
      }
    }

    if (!skey.verify) {
      // Extract only the B final rows (1 per slot) -> single B-row final_norm + lm_head!
      // Primary's hidden[0..total_m-1] is preserved untouched (never overwritten).
      const U32 b_count = U32(B);
      Ref b_logits = row_slice_ref(logits, 0U, b_count, v);
      bool head_done = false;
      if (pending_sq_slot.has_value()) {
        const U32 sq_final = *pending_sq_slot;
        pending_sq_slot.reset();
        if (all_w_eq_1 && b_count == 1U && single_t1_tdom &&
            try_linear_norm_t1(e, head, final_norm, hidden, b_logits, v, h,
                               sq_final)) {
          head_done = true;
        } else if (all_w_eq_1 && b_count == 2U && allow_m2_fused &&
                   try_linear_norm_m2(e, head, final_norm, hidden, b_logits, v,
                                      h, sq_final * m2_sq_mul)) {
          head_done = true;
        }
      }
      if (!head_done) {
        Ref b_hidden = all_w_eq_1 ? row_slice_ref(hidden, 0U, b_count, h)
                                  : row_slice_ref(delta, 0U, b_count, h);
        if (!all_w_eq_1) {
          for (std::size_t b_idx = 0; b_idx < B; ++b_idx) {
            const U32 last_r = row_offs[b_idx] + skey.widths[b_idx] - 1U;
            Ref dst_h = row_slice_ref(delta, U32(b_idx), 1U, h);
            auto &ex = e.emit("qwen_forward", "omlx_m5_forward_slice_rows",
                              {.width = h}, {.width = 128}, {hidden, dst_h});
            e.dynamic(ex, 2, words({1U, h, last_r}));
          }
        }
        const bool ar_head_m2 =
            (b_count == 2U && !speculative_enabled &&
             t_domain_active() && pretiled_qmv_t1_enabled() &&
             !inspect_tensors_workspace && has_row_exact_metallib_);
        const bool ar_head_f16 =
            (!ar_head_m2 && !single_t1_tdom && !has_bypass_lora(head) &&
             use_mxfp4_nax_for_weight(head, b_count) &&
             [&] {
               const auto *tw = mxfp4_nax_cache->find(head.object_id);
               return tw && tw->n == v && tw->k == h &&
                      mxfp4_nax_select_wide(*tw, b_count, b_count,
                                            silicon_tuning.gpu_cores, false) ==
                          NaxWidePrefillTile::none;
             }());
        const bool ar_head_affine_xs =
            (!ar_head_f16 && !ar_head_m2 && !single_t1_tdom &&
             affine_nax_active() && affine_nax_xs.buffer != nullptr &&
             (h % 64U) == 0U && b_count > 1U);
        Ref b_norm = row_slice_ref(norm, 0U, b_count, h);
        if (ar_head_f16) {
          normalize_f16(e, final_norm, b_hidden, f16_norm, b_count);
          linear_from_f16(e, head, f16_norm, f16_ones_scale, b_logits, v, h,
                          b_count);
        } else if (ar_head_m2) {
          normalize(e, final_norm, b_hidden, b_norm, 2U);
          if (!try_linear_m2(e, head, b_norm, b_logits, v, h)) {
            linear(e, head, b_norm, b_logits, v, h, 2U);
          }
        } else if (ar_head_affine_xs) {
          emit_affine_rmsnorm_xsum(e, b_hidden, direct(final_norm, h), b_norm,
                                   affine_nax_xs, h, float(config.rms_norm_eps),
                                   normalization_weight_offset, b_count);
          linear(e, head, b_norm, b_logits, v, h, b_count);
        } else {
          normalize(e, final_norm, b_hidden, b_norm, b_count);
          linear(e, head, b_norm, b_logits, v, h, b_count);
        }
      }

      // Batched MTP hook for non-verify AR steps: scatter each 1-row slot's
      // pre-final_norm trunk hidden state into sp.hidden[0] so MTP's next K>0
      // step (cond_mode==0) reads the exact trunk hidden instead of stale rows.
      if (speculative_enabled) {
        for (std::size_t b_idx = 0; b_idx < B; ++b_idx) {
          const auto &sp = *skey.slots[b_idx];
          if (&sp == this || !sp.hidden.buffer || skey.widths[b_idx] != 1U)
            continue;
          const U32 r_off = row_offs[b_idx];
          Ref src_h = row_slice_ref(hidden, r_off, 1U, h);
          Ref dst_h = row_slice_ref(sp.hidden, 0U, 1U, h);
          auto &cp_h = e.emit("qwen_forward", "omlx_m5_forward_copy_rows",
                              {.width = h}, {.width = 128}, {src_h, dst_h});
          e.dynamic(cp_h, 2, words({1U, h, 0U}));
        }
        if (all_w_eq_1) {
          for (std::size_t b_idx = 0; b_idx < B; ++b_idx) {
            const auto &sp = *skey.slots[b_idx];
            const U32 r_off = row_offs[b_idx];
            if (&sp == this && r_off != 0U && sp.hidden.buffer) {
              Ref src_h = row_slice_ref(hidden, r_off, 1U, h);
              Ref dst_h = row_slice_ref(sp.hidden, 0U, 1U, h);
              auto &cp_h = e.emit("qwen_forward", "omlx_m5_forward_copy_rows",
                                  {.width = h}, {.width = 128}, {src_h, dst_h});
              e.dynamic(cp_h, 2, words({1U, h, 0U}));
            }
          }
        }
      }

      if (skey.sample_in_icb) {
        e.begin_stage("sample");
        const bool use_64tg_argmax =
            (v >= 256U) && (packed_samp_parts.buffer != nullptr) &&
            !detail::forward::env_flag_enabled("OMLX_M5_DISABLE_64TG_ARGMAX");
        if (sampling_contract == SamplingContract::PositionCoupledV1) {
          std::vector<PositionSamplingRow> init_rows(b_count);
          for (U32 bi = 0U; bi < b_count; ++bi) {
            init_rows[bi].seed = 0x51A00000ULL | U64(bi);
            init_rows[bi].position = U64(bi) + 1ULL;
          }
          const auto init_raw =
              std::as_bytes(std::span<const PositionSamplingRow>(init_rows));
          const std::vector<std::byte> init_opts(init_raw.begin(),
                                                 init_raw.end());
          if (use_64tg_argmax) {
            auto &st1 = e.emit("position_sampling",
                               "omlx_m5_position_sample_logits_stage1",
                               {.width = 64U * 256U, .height = b_count},
                               {.width = 256U}, {b_logits, packed_samp_parts});
            const U64 off1 = e.dynamic(st1, 2, init_opts);
            e.constant(st1, 3, words({v, b_count, b_count, 0U}));
            auto &st2 = e.emit(
                "position_sampling", "omlx_m5_position_sample_logits_stage2",
                {.width = 256U, .height = b_count}, {.width = 256U},
                {b_logits, packed_samp_parts, packed_samp_meta, packed_samp_ids,
                 packed_samp_scores});
            const U64 off2 = e.dynamic(st2, 5, init_opts);
            e.constant(st2, 6, words({v, b_count, b_count, 0U}));
            sample_opt_offsets.push_back(off1);
            if (off2 != off1)
              sample_opt_offsets.push_back(off2);
          } else {
            auto &samp = e.emit(
                "position_sampling", "omlx_m5_position_sample_logits",
                {.width = 256U, .height = b_count}, {.width = 256U},
                {b_logits, packed_samp_meta, packed_samp_ids,
                 packed_samp_scores});
            const U64 off = e.dynamic(samp, 4, init_opts);
            e.constant(samp, 5, words({v, b_count, b_count, 0U}));
            sample_opt_offsets.push_back(off);
          }
        } else {
          for (U32 b_idx = 0U; b_idx < b_count; ++b_idx) {
            Ref row_logits = row_slice_ref(logits, b_idx, 1U, v);
            Ref row_out{packed_samp_meta.buffer, U64(b_idx) * 8U, 4U};
            // Distinct per-slot initializer so Encoder::dynamic never
            // deduplicates different slots onto the same parameter offset.
            SampleParams init_sp{};
            init_sp.vocabulary = v;
            init_sp.seed = 0x51B00000ULL | U64(b_idx);
            init_sp.absolute_position = b_idx;
            const auto sp_raw =
                std::as_bytes(std::span<const SampleParams>(&init_sp, 1));
            const std::vector<std::byte> sp_bytes(sp_raw.begin(), sp_raw.end());
            if (use_64tg_argmax) {
              Ref row_parts{packed_samp_parts.buffer, U64(b_idx) * 512U, 512U};
              auto &st1 = e.emit("vocab", "omlx_m5_vocab_argmax_stage1_64tg",
                                 {.width = 64U * 256U}, {.width = 256U},
                                 {row_logits, row_parts});
              e.constant(st1, 2, words({v}));
              auto &st2 = e.emit("vocab", "omlx_m5_sample_logits_stage2",
                                 {.width = 256U}, {.width = 256U},
                                 {row_logits, row_parts, row_out});
              sample_opt_offsets.push_back(e.dynamic(st2, 3, sp_bytes));
            } else {
              auto &samp = e.emit("vocab", "omlx_m5_sample_logits",
                                  {.width = 256U}, {.width = 256U},
                                  {row_logits, row_out});
              sample_opt_offsets.push_back(e.dynamic(samp, 2, sp_bytes));
            }
          }
        }
      } else if (skey.scatter_slot_logits) {
        for (std::size_t b_idx = 0; b_idx < B; ++b_idx) {
          const auto &sp = *skey.slots[b_idx];
          Ref src_row = row_slice_ref(logits, U32(b_idx), 1U, v);
          auto &cp_last = e.emit("qwen_forward", "omlx_m5_forward_copy_rows",
                                 {.width = v}, {.width = 256},
                                 {src_row, sp.last_logits});
          e.dynamic(cp_last, 2, words({1U, v, 0U}));
          if (&sp != this || b_idx != 0U) {
            Ref sp_row0 = row_slice_ref(sp.logits, 0U, 1U, v);
            auto &cp_r0 = e.emit("qwen_forward", "omlx_m5_forward_copy_rows",
                                 {.width = v}, {.width = 256},
                                 {src_row, sp_row0});
            e.dynamic(cp_r0, 2, words({1U, v, 0U}));
          }
        }
      }
    } else {
      // Batched MTP hook: scatter each slot's [w_b, H] pre-final_norm trunk
      // hidden rows into sp.hidden[0..w_b] (secondary slots first to avoid
      // overlap; primary copied via delta only if placed at r_off != 0).
      for (std::size_t b_idx = 0; b_idx < B; ++b_idx) {
        const auto &sp = *skey.slots[b_idx];
        if (&sp == this || !sp.hidden.buffer)
          continue;
        const U32 r_off = row_offs[b_idx];
        const U32 w_b = skey.widths[b_idx];
        Ref src_h = row_slice_ref(hidden, r_off, w_b, h);
        auto &cp_h = e.emit("qwen_forward", "omlx_m5_forward_copy_rows",
                            {.width = U64(w_b) * h}, {.width = 128},
                            {src_h, sp.hidden});
        e.dynamic(cp_h, 2, words({w_b, h, 0U}));
      }
      for (std::size_t b_idx = 0; b_idx < B; ++b_idx) {
        const auto &sp = *skey.slots[b_idx];
        const U32 r_off = row_offs[b_idx];
        const U32 w_b = skey.widths[b_idx];
        if (&sp == this && r_off != 0U && sp.hidden.buffer) {
          Ref src_h = row_slice_ref(hidden, r_off, w_b, h);
          Ref tmp_h = row_slice_ref(delta, 0U, w_b, h);
          auto &cp1 = e.emit("qwen_forward", "omlx_m5_forward_copy_rows",
                             {.width = U64(w_b) * h}, {.width = 128},
                             {src_h, tmp_h});
          e.dynamic(cp1, 2, words({w_b, h, 0U}));
          auto &cp2 = e.emit("qwen_forward", "omlx_m5_forward_copy_rows",
                             {.width = U64(w_b) * h}, {.width = 128},
                             {tmp_h, sp.hidden});
          e.dynamic(cp2, 2, words({w_b, h, 0U}));
        }
      }

      bool primary_nonzero_offset = false;
      for (std::size_t b_idx = 0; b_idx < B; ++b_idx) {
        if (skey.slots[b_idx] == this && row_offs[b_idx] != 0U) {
          primary_nonzero_offset = true;
          break;
        }
      }
      const bool use_packed_verify_logits =
          (mul(total_m, mul(v, 2U)) > logits.bytes) ||
          (skey.scatter_slot_logits && primary_nonzero_offset);
      if (use_packed_verify_logits && !packed_verify_logits.buffer) {
        const U64 pv_bytes = mul(64U, mul(v, 2U));
        auto pv_buf =
            std::make_unique<MetalBuffer>(context.make_private_buffer(pv_bytes));
        packed_verify_logits = Ref{pv_buf.get(), 0, pv_bytes};
        buffers.push_back(std::move(pv_buf));
      }
      Ref dst_all_logits =
          use_packed_verify_logits ? packed_verify_logits : logits;
      if (!dst_all_logits.buffer ||
          mul(total_m, mul(v, 2U)) > dst_all_logits.bytes) {
        throw std::out_of_range(
            "Qwen packed verify batch exceeds 64-token logits scratch");
      }
      if (verify_head_f16) {
        linear_from_f16(e, head, f16_norm, f16_ones_scale, dst_all_logits, v, h,
                        total_m);
      } else {
        linear(e, head, norm, dst_all_logits, v, h, total_m);
      }
      if (skey.scatter_slot_logits) {
        for (std::size_t b_idx = 0; b_idx < B; ++b_idx) {
          const auto &sp = *skey.slots[b_idx];
          const U32 r_off = row_offs[b_idx];
          const U32 w_b = skey.widths[b_idx];
          if (dst_all_logits.buffer == sp.logits.buffer && r_off == 0U)
            continue;
          auto &sl = e.emit("qwen_forward", "omlx_m5_forward_slice_rows",
                            {.width = U64(w_b) * v}, {.width = 256},
                            {dst_all_logits, sp.logits});
          e.dynamic(sl, 2, words({w_b, v, r_off}));
        }
      }
    }

    CachedPackedProfile entry{skey, e.finish(), std::move(pos_patches),
                              std::move(sample_opt_offsets)};
    ensure_profile_icb(context, entry.profile);
    return entry;
  }

  CachedPackedProfile &acquire_packed_profile(const PackedShapeKey &key) {
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
    if (packed_cache_.size() >= 16U) {
      packed_cache_.pop_back();
    }
    packed_cache_.push_front(compile_packed_batch_profile(key));
    return packed_cache_.front();
  }

  QwenForwardResult
  execute_delegated_prefill(QwenForward &secondary_slot,
                            std::span<const U32> tokens,
                            const GpuSamplingOptions &sampling, bool use_icb) {
    const std::array<PackedSlotSlice, 1> slice{{
        PackedSlotSlice{.slot_program = &secondary_slot,
                        .slot_index = 0U,
                        .tokens = tokens,
                        .sampling = sampling,
                        .verify = false},
    }};
    const U32 start_pos = secondary_slot.position();
    auto packed = run_packed_batch_impl(slice, use_icb);
    QwenForwardResult res = std::move(packed.slot_results.front());
    res.submission = packed.submission;

    // Sample token from row0 of primary's logits (which already holds secondary's last row!).
    auto &sprof = delegated_sampler_profile;
    if (sampling_contract == SamplingContract::PositionCoupledV1) {
      const PositionSamplingRow row{.seed = sampling.seed,
                                    .position = start_pos + tokens.size(),
                                    .top_k = sampling.top_k,
                                    .temperature = sampling.temperature,
                                    .top_p = sampling.top_p};
      std::memcpy(sprof.parameter_data.data() + sprof.sampling_offset, &row,
                  sizeof(row));
    } else {
      const SampleParams sp{v,
                            sampling.top_k,
                            sampling.temperature,
                            sampling.top_p,
                            sampling.seed,
                            sampling.stream,
                            start_pos + tokens.size() - 1};
      std::memcpy(sprof.parameter_data.data() + sprof.sampling_offset, &sp,
                  sizeof(sp));
    }
    sprof.parameters.write("step", sprof.parameter_data);
    MetalSubmission samp_sub;
    if (use_icb && sprof.icb) {
      samp_sub = context.replay_icb(*sprof.icb);
    } else {
      const std::array calls{MetalCommandInvocation{.plan = &sprof.plan,
                                                    .bindings = sprof.bindings,
                                                    .parameters =
                                                        &sprof.parameters}};
      samp_sub = context.execute_batch(calls);
    }
    res.submission.timing.gpu_ms += samp_sub.timing.gpu_ms;
    res.submission.timing.wall_ms += samp_sub.timing.wall_ms;
    res.submission.dispatch_count += samp_sub.dispatch_count;

    const auto tok_raw = sampled.buffer->read(
        0, sampling_contract == SamplingContract::PositionCoupledV1 ? 8 : 4);
    std::memcpy(&res.token, tok_raw.data(), 4);
    res.host_read_bytes = tok_raw.size();
    if (sampling_contract == SamplingContract::PositionCoupledV1) {
      U32 flag = 0;
      std::memcpy(&flag, tok_raw.data() + 4, 4);
      if (flag & 2)
        throw std::runtime_error("nonfinite position target logits");
      if (flag & 1) {
        auto ib = position_ids.buffer->read(0, U64(sampling.top_k) * 4);
        auto sb = position_scores.buffer->read(0, U64(sampling.top_k) * 4);
        std::vector<U32> ids(sampling.top_k);
        std::vector<float> scores(sampling.top_k);
        std::memcpy(ids.data(), ib.data(), ib.size());
        std::memcpy(scores.data(), sb.data(), sb.size());
        res.token = position_sample_candidates(
            ids, scores,
            {.seed = sampling.seed,
             .position = start_pos + tokens.size(),
             .top_k = sampling.top_k,
             .temperature = sampling.temperature,
             .top_p = sampling.top_p});
        res.host_read_bytes += ib.size() + sb.size();
      }
    } else if (res.token >= v) {
      throw std::runtime_error("nonfinite or invalid legacy sampled token");
    }
    return res;
  }

  PackedBatchResult
  run_packed_batch_impl(std::span<const PackedSlotSlice> slices, bool use_icb,
                        bool sample_in_icb = false,
                        bool scatter_slot_logits = true) {
    if (shared_primary_ != nullptr) {
      return shared_primary_->impl_->run_packed_batch_impl(
          slices, use_icb, sample_in_icb, scatter_slot_logits);
    }
    if (slices.empty() || slices.size() > 8U) {
      throw std::invalid_argument("Qwen packed batch requires 1..8 slot slices");
    }
    require_no_async_in_flight("run_packed_batch");
    execution_started = true;
    const bool verify = slices.front().verify;
    if (sample_in_icb && verify) {
      throw std::invalid_argument(
          "Qwen run_packed_batch_and_sample does not support verify slices");
    }
    PackedShapeKey shape_key{.verify = verify,
                             .sample_in_icb = sample_in_icb,
                             .scatter_slot_logits =
                                 sample_in_icb ? false : scatter_slot_logits};
    shape_key.slots.reserve(slices.size());
    shape_key.widths.reserve(slices.size());

    U32 total_m = 0;
    U32 max_w_run = 0;
    U32 max_end_pos = 0;
    U32 max_slot_ctx = 0;
    bool all_affine4 =
        (slices.size() >= 2U && slices.size() <= 8U && config.head_dim == 256 &&
         attention_policy == omlx::m5::AttentionNumericPolicy::RowExactV1 &&
         !inspect_tensors_workspace && attention_positions.buffer != nullptr &&
         split_max.buffer != nullptr && split_sum.buffer != nullptr &&
         split_output.buffer != nullptr);
    bool all_bf16 =
        (slices.size() >= 2U && slices.size() <= 8U &&
         split_attention_enabled && !inspect_tensors_workspace &&
         attention_positions.buffer != nullptr && split_max.buffer != nullptr &&
         (strict_two_pass_attn_ ||
          (split_sum.buffer != nullptr && split_output.buffer != nullptr)));
    for (const auto &s : slices) {
      if (!s.slot_program)
        throw std::invalid_argument("Qwen packed batch slice missing slot_program");
      auto &sp = *s.slot_program->impl_;
      sp.require_no_async_in_flight("run_packed_batch");
      sp.execution_started = true;
      if (s.verify != verify || sp.pending_verify || s.tokens.empty() ||
          (verify && (!sp.speculative_enabled || s.tokens.size() > 8U)) ||
          s.tokens.size() > sp.max_context - sp.next_position ||
          s.sampling.contract != sp.sampling_contract ||
          std::any_of(s.tokens.begin(), s.tokens.end(),
                      [&](U32 id) { return id >= sp.v; }) ||
          s.sampling.top_k > 64 ||
          (s.sampling.temperature > 0 && s.sampling.top_k == 0) ||
          s.sampling.temperature < 0 || !std::isfinite(s.sampling.temperature) ||
          !std::isfinite(s.sampling.top_p) || s.sampling.top_p < 0 ||
          s.sampling.top_p > 1) {
        throw std::invalid_argument("Qwen packed batch slice contract invalid");
      }
      if (sp.shared_contract_identity != shared_contract_identity) {
        throw std::invalid_argument(
            "Qwen packed batch slot differs in execution contract");
      }
      const U32 w_b = U32(s.tokens.size());
      total_m += w_b;
      max_w_run = std::max(max_w_run, w_b);
      max_end_pos = std::max(max_end_pos, sp.next_position + w_b);
      max_slot_ctx = std::max(max_slot_ctx, sp.max_context);
      all_affine4 &= (sp.kv_format_ == KvStorageFormat::Affine4Group128);
      all_bf16 &= (sp.kv_format_ == KvStorageFormat::BFloat16);
      shape_key.slots.push_back(&sp);
      shape_key.widths.push_back(w_b);
    }
    if (total_m == 0U || total_m > workspace_rows_) {
      throw std::out_of_range("Qwen packed batch total tokens exceed workspace");
    }
    const bool will_use_batched_affine4_flash =
        all_affine4 && flash_nax_prefill_ && !verify && max_w_run >= 16U;
    if (all_affine4 && !will_use_batched_affine4_flash) {
      U32 env = max_slot_ctx;
      for (U32 cand : {2048U, 4096U, 8192U, 16384U, 32768U, 65536U}) {
        if (cand >= max_end_pos) {
          env = std::min(cand, max_slot_ctx);
          break;
        }
      }
      shape_key.affine4_split_env = std::max(512U, env);
    }
    const U32 split_min_ctx =
        std::max<U32>(512U, silicon_tuning.split_kv_min_context);
    if (all_bf16 && max_w_run <= 8U && max_end_pos >= split_min_ctx) {
      U32 env = max_slot_ctx;
      for (U32 cand : {2048U, 4096U, 8192U, 16384U, 32768U, 65536U}) {
        if (cand >= max_end_pos) {
          env = std::min(cand, max_slot_ctx);
          break;
        }
      }
      shape_key.bf16_split_env = std::max(512U, env);
    }

    auto &entry = acquire_packed_profile(shape_key);
    auto &prof = entry.profile;

    // Stage packed token_ids into primary workspace.
    std::fill_n(token_staging.begin(), prof.capacity, 0U);
    U32 cursor = 0;
    for (const auto &s : slices) {
      std::copy(s.tokens.begin(), s.tokens.end(),
                token_staging.begin() + cursor);
      cursor += U32(s.tokens.size());
    }
    const_cast<MetalBuffer *>(token_ids.buffer)
        ->write(std::as_bytes(std::span(token_staging).first(prof.capacity)));

    // Patch global packed M words, per-slot position words, and optional in-ICB sampling rows.
    for (const auto &patch : prof.patches) {
      const U32 val = patch.position ? 0U : total_m * patch.multiplier;
      std::memcpy(prof.parameter_data.data() + patch.offset + patch.word * 4,
                  &val, 4);
    }
    for (const auto &spatch : entry.pos_patches) {
      const U32 pos_val = shape_key.slots[spatch.slice_index]->next_position;
      std::memcpy(prof.parameter_data.data() + spatch.offset + spatch.word * 4,
                  &pos_val, 4);
    }
    if (sample_in_icb) {
      if (sampling_contract == SamplingContract::PositionCoupledV1) {
        std::array<PositionSamplingRow, 8> srows{};
        for (std::size_t b_idx = 0; b_idx < slices.size(); ++b_idx) {
          const auto &sp = *slices[b_idx].slot_program->impl_;
          const U32 w_b = U32(slices[b_idx].tokens.size());
          srows[b_idx] = PositionSamplingRow{
              .seed = slices[b_idx].sampling.seed,
              .position = sp.next_position + w_b,
              .top_k = slices[b_idx].sampling.top_k,
              .temperature = slices[b_idx].sampling.temperature,
              .top_p = slices[b_idx].sampling.top_p,
              .valid = 1U};
        }
        for (const U64 opt_off : entry.sample_opt_offsets) {
          std::memcpy(prof.parameter_data.data() + opt_off, srows.data(),
                      slices.size() * sizeof(PositionSamplingRow));
        }
      } else {
        for (std::size_t b_idx = 0; b_idx < slices.size(); ++b_idx) {
          const auto &sp = *slices[b_idx].slot_program->impl_;
          const U32 w_b = U32(slices[b_idx].tokens.size());
          const SampleParams sparams{v,
                                     slices[b_idx].sampling.top_k,
                                     slices[b_idx].sampling.temperature,
                                     slices[b_idx].sampling.top_p,
                                     slices[b_idx].sampling.seed,
                                     slices[b_idx].sampling.stream,
                                     sp.next_position + w_b - 1U};
          std::memcpy(prof.parameter_data.data() +
                          entry.sample_opt_offsets[b_idx],
                      &sparams, sizeof(sparams));
        }
      }
    }
    prof.parameters.write("step", prof.parameter_data);

    // Stage per-slot attention_positions & begin KV page appends with RAII rollback.
    struct ActivePoolGuard {
      std::vector<KvPagePool *> pools;
      bool active = true;
      ~ActivePoolGuard() {
        if (!active)
          return;
        for (auto *pool : pools) {
          if (pool) {
            try {
              pool->abort();
            } catch (...) {
            }
          }
        }
      }
    } guard;

    std::vector<U32> packed_positions;
    if (attention_positions.buffer) {
      packed_positions.reserve(total_m);
    }
    for (std::size_t b_idx = 0; b_idx < slices.size(); ++b_idx) {
      auto &sp = *slices[b_idx].slot_program->impl_;
      const U32 w_b = U32(slices[b_idx].tokens.size());
      sp.last_trace_tokens = 0;
      if (attention_positions.buffer) {
        for (U32 i = 0; i < w_b; ++i)
          packed_positions.push_back(sp.next_position + i);
      }
      if (sp.page_pool) {
        if (sp.page_pool->tokens() != sp.next_position)
          throw std::logic_error("Qwen packed batch KV frontier mismatch");
        const auto copies = sp.page_pool->begin_append(w_b);
        guard.pools.push_back(sp.page_pool.get());
        sp.execute_cow_copies(copies);
      }
    }
    if (attention_positions.buffer && !packed_positions.empty()) {
      const_cast<MetalBuffer *>(attention_positions.buffer)
          ->write(std::as_bytes(std::span(packed_positions)));
    }

    bool primary_in_slices = false;
    bool primary_nonzero_offset = false;
    {
      U32 off = 0U;
      for (const auto &s : slices) {
        if (s.slot_program->impl_.get() == this) {
          primary_in_slices = true;
          if (off != 0U)
            primary_nonzero_offset = true;
        }
        off += U32(s.tokens.size());
      }
    }
    const bool use_packed_verify_logits =
        verify && ((mul(total_m, mul(v, 2U)) > logits.bytes) ||
                   (shape_key.scatter_slot_logits && primary_nonzero_offset));
    if (!primary_in_slices && (!verify || !use_packed_verify_logits)) {
      has_last_logits = false;
    }

    PackedBatchResult out;
    if (use_icb && prof.icb) {
      out.submission = context.replay_icb(*prof.icb);
    } else {
      const std::array calls{
          MetalCommandInvocation{.plan = &prof.plan,
                                 .bindings = prof.bindings,
                                 .parameters = &prof.parameters}};
      out.submission = context.execute_batch(calls);
    }

    last_packed_logits_buf_ =
        use_packed_verify_logits ? packed_verify_logits.buffer : logits.buffer;

    std::array<U32, 8> sampled_tokens{};
    std::array<U64, 8> sampled_read_bytes{};
    if (sample_in_icb) {
      const auto meta_bytes =
          packed_samp_meta.buffer->read(0, U64(slices.size()) * 8U);
      out.sampling_host_read_bytes = meta_bytes.size();
      if (sampling_contract == SamplingContract::PositionCoupledV1) {
        for (std::size_t b_idx = 0; b_idx < slices.size(); ++b_idx) {
          const auto &sp = *slices[b_idx].slot_program->impl_;
          const U32 w_b = U32(slices[b_idx].tokens.size());
          U32 tok = 0U, flag = 0U;
          std::memcpy(&tok, meta_bytes.data() + b_idx * 8U, 4);
          std::memcpy(&flag, meta_bytes.data() + b_idx * 8U + 4U, 4);
          U64 row_rb = 8U;
          if (flag & 2U)
            throw std::runtime_error("nonfinite position target logits");
          if (flag & 1U) {
            const U32 tk = slices[b_idx].sampling.top_k;
            auto ib = packed_samp_ids.buffer->read(U64(b_idx) * 64U * 4U,
                                                   U64(tk) * 4U);
            auto sb = packed_samp_scores.buffer->read(U64(b_idx) * 64U * 4U,
                                                      U64(tk) * 4U);
            std::vector<U32> ids(tk);
            std::vector<float> scores(tk);
            std::memcpy(ids.data(), ib.data(), ib.size());
            std::memcpy(scores.data(), sb.data(), sb.size());
            tok = position_sample_candidates(
                ids, scores,
                {.seed = slices[b_idx].sampling.seed,
                 .position = sp.next_position + w_b,
                 .top_k = tk,
                 .temperature = slices[b_idx].sampling.temperature,
                 .top_p = slices[b_idx].sampling.top_p});
            row_rb += ib.size() + sb.size();
            out.sampling_host_read_bytes += ib.size() + sb.size();
            out.compact_fallback_rows += 1U;
          }
          sampled_tokens[b_idx] = tok;
          sampled_read_bytes[b_idx] = row_rb;
        }
      } else {
        for (std::size_t b_idx = 0; b_idx < slices.size(); ++b_idx) {
          U32 tok = 0U;
          std::memcpy(&tok, meta_bytes.data() + b_idx * 8U, 4);
          if (tok >= v)
            throw std::runtime_error(
                "nonfinite or invalid legacy sampled token");
          sampled_tokens[b_idx] = tok;
          sampled_read_bytes[b_idx] = 8U;
        }
      }
    }

    out.slot_results.reserve(slices.size());
    for (std::size_t b_idx = 0; b_idx < slices.size(); ++b_idx) {
      auto &sp = *slices[b_idx].slot_program->impl_;
      const U32 w_b = U32(slices[b_idx].tokens.size());
      if (verify) {
        sp.pending_verify = w_b;
        sp.has_last_logits = false;
      } else {
        if (sp.page_pool)
          sp.page_pool->commit(w_b);
        sp.next_position += w_b;
        sp.last_logits_row = 0U;
        sp.has_last_logits = (!sample_in_icb && scatter_slot_logits);
      }
      QwenForwardResult r{.token = sampled_tokens[b_idx],
                          .tokens = w_b,
                          .executed_layers = U32(blocks.size()),
                          .host_read_bytes = sampled_read_bytes[b_idx],
                          .short_matrix_calls = prof.short_matrix_calls,
                          .short_lane_calls = prof.short_lane_calls};
      out.slot_results.push_back(std::move(r));
    }
    guard.active = false;
    return out;
  }

  void require_no_async_in_flight(const char *op) const {
    if (ar_in_flight_count_ != 0U || prefill_in_flight_count_ != 0U) {
      throw std::logic_error(std::string("Qwen ") + op +
                             " cannot run while async steps are in flight");
    }
  }

  void drain_and_invalidate_async_steps() {
    if (ar_in_flight_count_ != 0U || prefill_in_flight_count_ != 0U) {
      static_cast<void>(context.submit_barrier());
      ar_in_flight_count_ = 0U;
      ar_pipe_cursor_ = 0U;
      prefill_in_flight_count_ = 0U;
      prefill_pipe_cursor_ = 0U;
      ++async_generation_;
    }
  }

  void reset() {
    drain_and_invalidate_async_steps();
    if (page_pool)
      page_pool->reset();
    for (auto *buffer : states)
      buffer->zero();
    next_position = 0;
    ar_in_flight_count_ = 0;
    ar_pipe_cursor_ = 0;
    prefill_in_flight_count_ = 0;
    prefill_pipe_cursor_ = 0;
    pending_verify = 0;
    verify_async_profile_ = nullptr;
    last_trace_tokens = 0;
    last_logits_row = 0;
    has_last_logits = false;
    ++feature_epoch_;
  }

  struct LongCtxProfileKey {
    U32 capacity = 0;
    U32 context_envelope = 0;
    ChunkAttentionRoute attn_route = ChunkAttentionRoute::ExactNax;
    bool trunk_only = false;
    bool uses_pipe_slot0 = false;
    std::uint8_t pipe_slot = 0;
    auto operator<=>(const LongCtxProfileKey &) const = default;
  };

  struct AneChunkPlanKey {
    U32 context_envelope = 0;
    ChunkAttentionRoute attn_route = ChunkAttentionRoute::ExactNax;
    bool trunk_only = false;
    auto operator<=>(const AneChunkPlanKey &) const = default;
  };

  struct AneLayerSubPlan {
    bool is_full_attn = false;
    std::optional<Profile> attn_pre_o;
    std::optional<U32> attn_o_op_id;
    std::optional<Profile> attn_o_gpu;
    std::optional<Profile> gdn_qkv_ab;
    std::optional<U32> gdn_z_op_id;
    std::optional<Profile> gdn_z_gpu;
    std::optional<Profile> gdn_rec;
    std::optional<U32> gdn_out_op_id;
    std::optional<Profile> gdn_out_gpu;
    std::optional<Profile> post_norm;
    std::optional<U32> mlp_gu_op_id;
    std::optional<Profile> mlp_gu_gpu;
    std::optional<U32> mlp_down_op_id;
    std::optional<Profile> mlp_down_gpu;
    std::optional<Profile> next_norm;
  };

  struct AneChunkPlan {
    U32 chunk_t = 0;
    U32 context_envelope = 0;
    ChunkAttentionRoute attn_route = ChunkAttentionRoute::ExactNax;
    bool trunk_only = false;
    bool used_w4a8 = false;
    bool used_sage2 = false;
    bool used_minference = false;
    std::optional<Profile> embed_norm0;
    std::vector<AneLayerSubPlan> layers;
    std::optional<Profile> head_sample;
  };

  struct UltraSwiGluLayerSubPlan {
    std::optional<Profile> pre_mlp;    // attn/gdn (full concat + GDN Stage-3) + post_norm
    std::optional<U32> mlp_gu_op_id;   // MlpGateUpSwiGlu ANE op_id
    ane::AneUnpackTarget unpack_target{};
    std::optional<Profile> mlp_gu_gpu; // GPU suffix [N_ane, f) in BF16 or W4A8 Mode B
    std::optional<Profile> post_mlp;   // mlp_down + add + next input_norm (or final add)
  };

  struct UltraSwiGluChunkPlan {
    U32 chunk_t = 0;
    U32 context_envelope = 0;
    ChunkAttentionRoute attn_route = ChunkAttentionRoute::ExactNax;
    bool trunk_only = false;
    bool used_w4a8 = false;
    bool used_sage2 = false;
    bool used_minference = false;
    std::optional<Profile> embed_norm0;
    std::vector<UltraSwiGluLayerSubPlan> layers;
    std::optional<Profile> head_sample;
  };

  template <typename T>
  static std::vector<std::byte> pod_bytes(const T &v) {
    const auto raw = std::as_bytes(std::span<const T, 1>(&v, 1));
    return std::vector<std::byte>(raw.begin(), raw.end());
  }

  ChunkAttentionRoute select_chunk_attention_route(U32 count,
                                                   U32 start_pos) const noexcept {
    if (!sage2_enabled_ || count < kFlashNaxPrefillMinTokens ||
        config.head_dim != 256) {
      return ChunkAttentionRoute::ExactNax;
    }
    const U32 end_pos = start_pos + count;
    if (end_pos < sage2_min_end_kv_) {
      return ChunkAttentionRoute::ExactNax;
    }
    // Tiny tail over deep KV cache: use Split-KV ExactNax to saturate GPU cores.
    if (count < 64U && end_pos >= 4096U && flash_nax_prefill_ &&
        split_attention_enabled && !strict_two_pass_attn_ &&
        gqa_flash_prefill_tail_needs_split_kv(
            count, U32(config.num_attention_heads),
            U32(config.num_key_value_heads), end_pos,
            silicon_tuning.gpu_cores)) {
      return ChunkAttentionRoute::ExactNax;
    }
    const U32 mean_causal_kv = start_pos + (count >> 1U);
    if (minference_enabled_ && end_pos >= kMInferenceMinContextTokens &&
        mean_causal_kv >= minf_min_mean_kv_) {
      return ChunkAttentionRoute::Sage2MInference;
    }
    return ChunkAttentionRoute::Sage2Dense;
  }

  bool needs_long_ctx_prefill_profile(U32 count, U32 start_pos) const noexcept {
    if (inspect_tensors_workspace || count < kFlashNaxPrefillMinTokens) {
      return false;
    }
    const U32 end_pos = start_pos + count;
    const auto route = select_chunk_attention_route(count, start_pos);
    if (route == ChunkAttentionRoute::Sage2MInference) {
      return true;
    }
    if (route == ChunkAttentionRoute::Sage2Dense && max_context > 4096U) {
      return true;
    }
    if (route == ChunkAttentionRoute::ExactNax && flash_nax_prefill_ &&
        split_attention_enabled && config.head_dim == 256 && end_pos >= 4096U &&
        gqa_flash_prefill_tail_needs_split_kv(
            count, U32(config.num_attention_heads),
            U32(config.num_key_value_heads), end_pos, silicon_tuning.gpu_cores)) {
      return true;
    }
    return false;
  }

  U32 select_long_ctx_envelope(U32 end_pos) const noexcept {
    for (U32 env :
         {2048U, 4096U, 8192U, 16384U, 32768U, 65536U, 131072U, 262144U}) {
      if (env >= max_context) {
        return max_context;
      }
      if (end_pos <= env) {
        return env;
      }
    }
    return max_context;
  }

  Profile *select_long_ctx_prefill_profile(U32 count, U32 start_pos,
                                           bool trunk_only,
                                           bool use_pipe_slot0,
                                           U32 pipe_slot = 0U) {
    if (!needs_long_ctx_prefill_profile(count, start_pos)) {
      return nullptr;
    }
    const U32 cap = select_forward_bucket_capacity(count, max_context);
    const U32 end_pos = start_pos + count;
    const U32 env = select_long_ctx_envelope(end_pos);
    const auto route = select_chunk_attention_route(count, start_pos);
    const std::uint8_t slot_id =
        use_pipe_slot0 ? std::uint8_t(pipe_slot & 1U) : std::uint8_t(0U);
    const LongCtxProfileKey exact_key{cap, env, route, trunk_only,
                                      use_pipe_slot0, slot_id};
    if (auto it = long_ctx_profiles_.find(exact_key);
        it != long_ctx_profiles_.end()) {
      return &it->second;
    }
    // Check if an existing precompiled profile covers (cap, env, route) when warmup is sealed.
    Profile *best = nullptr;
    for (auto &[k, prof] : long_ctx_profiles_) {
      if (k.attn_route == route && k.trunk_only == trunk_only &&
          k.uses_pipe_slot0 == use_pipe_slot0 && k.pipe_slot == slot_id &&
          k.capacity >= count && k.context_envelope >= end_pos) {
        if (!best || k.capacity < best->capacity ||
            (k.capacity == best->capacity &&
             k.context_envelope < best->context_envelope)) {
          best = &prof;
        }
      }
    }
    if (context.warmup_sealed()) {
      return best;
    }
    const Ref tok_ref =
        use_pipe_slot0 ? prefill_pipe_[slot_id].token_ids : Ref{};
    const Ref pos_ref =
        use_pipe_slot0 ? prefill_pipe_[slot_id].attn_pos : Ref{};
    const std::string tag =
        use_pipe_slot0 ? (".pf_pipe" + std::to_string(slot_id)) : "";
    auto prof = compile(cap, false, false, env, false, {}, pos_ref, {}, tag,
                        trunk_only, tok_ref, route);
    ensure_profile_icb(context, prof);
    auto [it, _] = long_ctx_profiles_.emplace(exact_key, std::move(prof));
    return &it->second;
  }

  void record_prefill_telemetry(U32 count, U32 start_pos, bool used_w4a8,
                                bool used_sage2, bool used_minf, U32 minf_cap,
                                U32 minf_env, bool used_ane,
                                QwenPrefillTelemetry *out_delta,
                                std::optional<U32> ane_offload_override = {},
                                bool used_w4a4 = false) {
    QwenPrefillTelemetry delta;
    delta.prefill_ane_active = ane_active_;
    delta.prefill_ane_mode_effective =
        std::string(prefill_ane_mode_to_string(ane_mode_effective_));
    delta.prefill_ane_inactive_reason = ane_inactive_reason_;
    delta.prefill_ane_split_alpha = ane_split_alpha_;
    if (used_w4a8) {
      prefill_telemetry_.prefill_w4a8_tokens += count;
      delta.prefill_w4a8_tokens += count;
    }
    if (used_w4a4) {
      prefill_telemetry_.prefill_w4a4_tokens += count;
      delta.prefill_w4a4_tokens += count;
    }
    if (used_sage2) {
      prefill_telemetry_.prefill_sage2_tokens += count;
      delta.prefill_sage2_tokens += count;
    }
    if (used_minf && minference_scratch.buffer != nullptr &&
        config.full_attention_layers() > 0 && count > 0U && minf_cap > 0U &&
        minf_env >= kMInferenceMinContextTokens) {
      const U32 qh = U32(config.num_attention_heads);
      const U32 kvh = U32(config.num_key_value_heads);
      const auto lo = plan_minference_scratch(minf_cap, qh, kvh, minf_env);
      const auto tier = select_sage2_tier(minf_cap, qh, kvh);
      const U32 q_blocks_actual = (count + tier.bm_tok - 1U) / tier.bm_tok;
      const U64 need_bytes = U64(kvh) * U64(lo.max_q_blocks) * sizeof(U32);
      if (lo.active_counts_off + need_bytes <= minference_scratch.bytes) {
        const auto raw = minference_scratch.buffer->read(
            minference_scratch.offset + lo.active_counts_off, need_bytes);
        std::vector<U32> counts(U64(kvh) * U64(lo.max_q_blocks), 0U);
        std::memcpy(counts.data(), raw.data(), raw.size());
        U64 layer_total = 0ULL;
        U64 layer_skipped = 0ULL;
        for (U32 kh_i = 0; kh_i < kvh; ++kh_i) {
          for (U32 qb = 0; qb < q_blocks_actual; ++qb) {
            const U32 max_q_pos =
                start_pos + std::min<U32>(count - 1U,
                                          (qb + 1U) * tier.bm_tok - 1U);
            const U32 total_blks = (max_q_pos + 32U) / 32U;
            const U32 active_blks = std::min<U32>(
                total_blks, counts[U64(kh_i) * lo.max_q_blocks + qb]);
            layer_total += total_blks;
            layer_skipped += (total_blks - active_blks);
          }
        }
        const U64 full_layers = U64(config.full_attention_layers());
        const U64 tot = layer_total * full_layers;
        const U64 skip = layer_skipped * full_layers;
        prefill_telemetry_.prefill_minference_blocks_total += tot;
        prefill_telemetry_.prefill_minference_blocks_skipped += skip;
        delta.prefill_minference_blocks_total += tot;
        delta.prefill_minference_blocks_skipped += skip;
      }
    }
    if (ane_active_) {
      if (ane_offload_override.has_value()) {
        const U32 off_tok = std::min<U32>(*ane_offload_override, count);
        const U32 fb_tok = count - off_tok;
        prefill_telemetry_.prefill_ane_offload_tokens += off_tok;
        delta.prefill_ane_offload_tokens += off_tok;
        if (fb_tok > 8U) {
          prefill_telemetry_.prefill_ane_fallback_tokens += fb_tok;
          delta.prefill_ane_fallback_tokens += fb_tok;
        }
      } else if (used_ane) {
        prefill_telemetry_.prefill_ane_offload_tokens += count;
        delta.prefill_ane_offload_tokens += count;
      } else if (count > 8U) {
        prefill_telemetry_.prefill_ane_fallback_tokens += count;
        delta.prefill_ane_fallback_tokens += count;
      }
    }
    if (out_delta != nullptr) {
      *out_delta = delta;
    }
  }

  bool can_run_ane_chunk(U32 count, bool verify, bool inspect_tensors,
                         bool profile_stages) const noexcept {
    return !verify && !inspect_tensors && !profile_stages &&
           shared_primary_ == nullptr && ane_active_ && ane_engine_ != nullptr &&
           ane_engine_->can_offload_chunk(count) && bypass_lora_ == nullptr &&
           !inspect_routes_enabled && !inspect_tensors_workspace &&
           feature_layers.empty();
  }

  // WP 3C Startup 1-Layer Canary Micro-Calibration Auto-Guard:
  // Benchmarks Layer-0 (post_norm + mlp_gu + mlp_down) GPU-solo (ICB) vs
  // concurrent collapsed GPU+Dual-ANE across candidate alphas BEFORE compiling
  // all 64 layers, caching the result on disk and requiring
  // t_Hybrid(alpha*) <= bypass_ratio * t_GPU_solo (default 0.98) for zero regression.
  bool calibrate_ultra_fp16_swiglu_canary(
      const ane::AneDualOffloadConfig &base_cfg,
      U32 effective_gpu_cores,
      U32 active_dies,
      const std::string &os_build,
      double &out_best_alpha,
      std::string &out_reason) {
    out_best_alpha = base_cfg.alpha_override.value_or(
        base_cfg.linear_route == ane::AneLinearRoute::W4A4
            ? base_cfg.alpha_w4a4
            : (base_cfg.linear_route == ane::AneLinearRoute::W4A8
                   ? base_cfg.alpha_w4a8
                   : base_cfg.alpha_bf16));
    out_reason.clear();
    if (blocks.empty()) {
      out_reason = "no_layers";
      return false;
    }

    const auto &blk0 = blocks[0];
    const auto &gw = blk0.get("mlp_gate");
    const auto &uw = blk0.get("mlp_up");
    const auto &dw = blk0.get("mlp_down");
    const U32 chunk_t = base_cfg.chunk_tokens;
    const bool use_w4a8 = use_w4a8_for_t(chunk_t, false);
    if (!can_gpu_suffix_swiglu(gw, uw, f, h, use_w4a8)) {
      out_reason = "gpu_suffix_unsupported";
      return false;
    }

    const bool force_active =
        detail::forward::env_flag_enabled("OMLX_M5_ANE_FORCE_ACTIVE");
    const bool recalibrate =
        detail::forward::env_flag_enabled("OMLX_M5_ANE_RECALIBRATE");
    const bool verbose =
        detail::forward::env_flag_enabled("OMLX_M5_ANE_CAL_VERBOSE");
    const double bypass_ratio = base_cfg.calibration_bypass_ratio;

    std::vector<double> raw_candidates;
    if (base_cfg.alpha_override.has_value()) {
      raw_candidates.push_back(*base_cfg.alpha_override);
    } else if (base_cfg.linear_route == ane::AneLinearRoute::W4A4) {
      raw_candidates = {0.10, 0.09, 0.08, 0.06};
    } else {
      raw_candidates = {0.12, 0.10, 0.09, 0.08, 0.06};
    }

    std::vector<double> candidates;
    std::vector<U32> seen_die_rows;
    for (double a : raw_candidates) {
      const auto sp = ane::compute_ane_channel_split(
          f, a, active_dies, base_cfg.max_alpha);
      if (!sp.enabled()) {
        continue;
      }
      if (std::find(seen_die_rows.begin(), seen_die_rows.end(), sp.die_rows) ==
          seen_die_rows.end()) {
        seen_die_rows.push_back(sp.die_rows);
        candidates.push_back(a);
      }
    }
    if (candidates.empty()) {
      out_reason = "no_valid_split";
      return false;
    }

    std::string cal_key;
    const bool use_disk_cache =
        base_cfg.enable_disk_cache && !base_cfg.artifact_digest.empty();
    if (use_disk_cache) {
      cal_key = ane::compute_calibration_cache_key(
          os_build, effective_gpu_cores, active_dies, base_cfg.artifact_digest,
          base_cfg.linear_route, chunk_t, h, f, bypass_ratio,
          base_cfg.alpha_override);
      if (!recalibrate && !force_active) {
        const auto cached =
            ane::load_cached_calibration(base_cfg.cache_root, cal_key);
        if (cached.valid) {
          out_best_alpha = cached.best_alpha;
          out_reason = cached.reason;
          if (verbose) {
            std::fprintf(
                stderr,
                "[ane-cal] cache hit key=%s active=%d alpha=%.4f gpu_solo=%.3fms hybrid=%.3fms reason=%s\n",
                cal_key.substr(0, 12).c_str(), cached.active ? 1 : 0,
                cached.best_alpha, cached.gpu_solo_ms, cached.best_hybrid_ms,
                cached.reason.c_str());
          }
          return cached.active;
        }
      }
    }

    // Seed hidden & delta with real finite activations via embedding lookup on token IDs.
    {
      std::vector<U32> seed_ids(chunk_t, 0U);
      const U32 mod_v = std::max<U32>(1U, std::min<U32>(v, 64U));
      for (U32 i = 0U; i < chunk_t; ++i) {
        seed_ids[i] = i % mod_v;
      }
      const_cast<MetalBuffer *>(token_ids.buffer)
          ->write(std::as_bytes(std::span(seed_ids)));
      Encoder e(context, libraries, chunk_t,
                "qwen.ane.cal.seed.c" + std::to_string(chunk_t) +
                    plan_key_suffix());
      e.begin_stage("cal_seed");
      embedding_lookup(e, embedding, token_ids, hidden, zero, v, h);
      copy(e, hidden, delta, mul(mul(chunk_t, h), 2ULL));
      Profile seed_prof = e.finish();
      const std::array calls{MetalCommandInvocation{
          .plan = &seed_prof.plan,
          .bindings = seed_prof.bindings,
          .parameters = &seed_prof.parameters}};
      static_cast<void>(context.execute_batch(calls));
    }

    constexpr U32 kBurst = 4U;
    constexpr U32 kWarmups = 2U;
    constexpr U32 kSamples = 5U;
    const bool mode_b_a8 =
        can_gpu_suffix_swiglu_fwht64_a8(gw, uw, dw, f, h, use_w4a8);
    const bool down_w4a8 = use_w4a8 && (dw.has_fwht64() || w4a8_down_proj_);
    const bool affine_xs_eligible =
        (!use_w4a8 && affine_nax_active() && affine_nax_xs.buffer != nullptr &&
         (h % 64U) == 0U);
    const bool mlp_f16 = !use_w4a8 && block_mlp_input_nax(blk0, chunk_t);

    auto median_of = [](std::vector<double> vals) -> double {
      if (vals.empty()) {
        return 0.0;
      }
      std::sort(vals.begin(), vals.end());
      return vals[vals.size() / 2U];
    };

    // 1. Measure GPU-solo steady-state Layer-0 MLP (kBurst reps in 1 ICB profile).
    double gpu_solo_ms = 0.0;
    {
      Encoder e(context, libraries, chunk_t,
                "qwen.ane.cal.gpu_solo.c" + std::to_string(chunk_t) +
                    plan_key_suffix());
      e.begin_stage("cal_gpu_solo");
      for (U32 bi = 0U; bi < kBurst; ++bi) {
        if (use_w4a8) {
          emit_add_rmsnorm_a8_g64(
              e, hidden, delta, direct(blk0.get("post_attention_norm"), h),
              norm, a8_qa, a8_sa, a8_ra, h, float(config.rms_norm_eps),
              normalization_weight_offset, /*write_bf16=*/false);
          e.set_a8_quant(make_a8_key(e, norm, h));
        } else if (mlp_f16) {
          add_normalize_f16(e, blk0.get("post_attention_norm"), hidden, delta,
                            f16_norm);
        } else if (affine_xs_eligible) {
          emit_affine_add_rmsnorm_xsum(
              e, hidden, delta, direct(blk0.get("post_attention_norm"), h),
              norm, affine_nax_xs, h, float(config.rms_norm_eps),
              normalization_weight_offset);
          e.set_group_sum(Encoder::GroupSumKey{
              affine_nax_xs.buffer, affine_nax_xs.offset, norm.buffer,
              norm.offset, h, 64U, chunk_t, true, 2U});
        } else {
          add_normalize(e, blk0.get("post_attention_norm"), hidden, delta, norm);
        }
        mlp(e, blk0, 0U, mlp_f16, {}, nullptr, std::nullopt, false, use_w4a8,
            false);
      }
      Profile solo_prof = e.finish();
      ensure_profile_icb(context, solo_prof);
      auto run_solo = [&]() -> double {
        const auto t0 = std::chrono::steady_clock::now();
        if (solo_prof.icb) {
          static_cast<void>(context.replay_icb(*solo_prof.icb));
        } else {
          const std::array calls{MetalCommandInvocation{
              .plan = &solo_prof.plan,
              .bindings = solo_prof.bindings,
              .parameters = &solo_prof.parameters}};
          static_cast<void>(context.execute_batch(calls));
        }
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() /
               double(kBurst);
      };
      for (U32 w = 0U; w < kWarmups; ++w) {
        (void)run_solo();
      }
      std::vector<double> samples;
      samples.reserve(kSamples);
      for (U32 s = 0U; s < kSamples; ++s) {
        samples.push_back(run_solo());
      }
      gpu_solo_ms = median_of(std::move(samples));
    }

    // 2. Sweep candidate alphas using a 1-layer canary engine on the active MTLCommandQueue.
    const std::array<ane::AneOffloadOpSpec, 1> layer0_specs{{
        ane::AneOffloadOpSpec{
            .layer_index = 0U,
            .role = ane::AneProjectionRole::MlpGateUpSwiGlu,
            .name = "mlp_gate_up_swiglu",
            .weight = gw,
            .up_weight = uw,
        },
    }};

    // Shared Layer-0 post_norm and mlp_down sub-profiles across alpha candidates.
    Profile p_post_norm = [&] {
      Encoder e(context, libraries, chunk_t,
                "qwen.ane.cal.post_norm.c" + std::to_string(chunk_t) +
                    plan_key_suffix());
      e.begin_stage("cal_post_norm");
      if (use_w4a8) {
        emit_add_rmsnorm_a8_g64(
            e, hidden, delta, direct(blk0.get("post_attention_norm"), h), norm,
            a8_qa, a8_sa, a8_ra, h, float(config.rms_norm_eps),
            normalization_weight_offset, /*write_bf16=*/true);
      } else if (affine_xs_eligible) {
        emit_affine_add_rmsnorm_xsum(
            e, hidden, delta, direct(blk0.get("post_attention_norm"), h), norm,
            affine_nax_xs, h, float(config.rms_norm_eps),
            normalization_weight_offset);
      } else {
        add_normalize(e, blk0.get("post_attention_norm"), hidden, delta, norm);
      }
      return e.finish();
    }();

    Profile p_mlp_down = [&] {
      Encoder e(context, libraries, chunk_t,
                "qwen.ane.cal.mlp_dn.c" + std::to_string(chunk_t) +
                    (mode_b_a8 ? ".mb" : ".ma") + plan_key_suffix());
      e.begin_stage("cal_mlp_dn");
      if (mode_b_a8) {
        e.set_a8_quant(make_a8_key_for(e, product, f, a8_mid_qa, a8_mid_sa,
                                       a8_mid_ra, std::nullopt,
                                       /*fwht64=*/true));
      }
      if (down_w4a8 && try_linear_w4a8_base(e, dw, product, delta, h, f)) {
      } else {
        linear(e, dw, product, delta, h, f);
      }
      return e.finish();
    }();

    auto encode_prof = [&](void *mtl_cb, Profile &pr) {
      const std::array calls{MetalCommandInvocation{
          .plan = &pr.plan, .bindings = pr.bindings, .parameters = &pr.parameters}};
      context.encode_into_command_buffer(mtl_cb, calls);
    };

    double best_alpha = candidates.front();
    double best_hybrid_ms = std::numeric_limits<double>::infinity();
    std::vector<std::pair<double, double>> cand_timings;
    cand_timings.reserve(candidates.size());

    for (double cand_alpha : candidates) {
      ane::AneDualOffloadConfig c_cfg = base_cfg;
      c_cfg.alpha_override = cand_alpha;
      c_cfg.enable_disk_cache = use_disk_cache;
      std::unique_ptr<ane::AneDualOffloadEngine> canary;
      try {
        canary = ane::AneDualOffloadEngine::build(layer0_specs, c_cfg);
      } catch (...) {
        continue;
      }
      if (!canary || !canary->is_active()) {
        continue;
      }
      const auto *op0 =
          canary->find_op(0U, ane::AneProjectionRole::MlpGateUpSwiGlu);
      if (op0 == nullptr || !op0->split.enabled()) {
        continue;
      }

      Profile p_mlp_gu_gpu = [&] {
        Encoder e(context, libraries, chunk_t,
                  "qwen.ane.cal.mlp_gu.c" + std::to_string(chunk_t) + ".off" +
                      std::to_string(op0->split.gpu_row_offset) +
                      (mode_b_a8 ? ".mb" : ".ma") + plan_key_suffix());
        e.begin_stage("cal_mlp_gu");
        if (use_w4a8) {
          e.set_a8_quant(make_a8_key(e, norm, h));
        } else if (affine_xs_eligible) {
          e.set_group_sum(Encoder::GroupSumKey{
              affine_nax_xs.buffer, affine_nax_xs.offset, norm.buffer,
              norm.offset, h, 64U, chunk_t, true, 2U});
        }
        if (mode_b_a8) {
          (void)try_linear_w4a8_swiglu_fwht64_a8(
              e, gw, uw, norm, product, f, h, /*also_write_bf16=*/false,
              std::nullopt, op0->split.gpu_row_offset, f, op0->split.gpu_rows);
        } else {
          (void)emit_gpu_suffix_swiglu(
              e, gw, uw, norm, product, f, h, use_w4a8,
              op0->split.gpu_row_offset, f, op0->split.gpu_rows);
        }
        return e.finish();
      }();

      ane::AneUnpackTarget unpack_target{};
      if (mode_b_a8) {
        unpack_target.mode = ane::AneUnpackMode::Fwht64A8;
        unpack_target.primary_buf = a8_mid_qa.buffer->native_mtl_buffer();
        unpack_target.primary_offset = a8_mid_qa.offset;
        unpack_target.aux0_buf = a8_mid_sa.buffer->native_mtl_buffer();
        unpack_target.aux0_offset = a8_mid_sa.offset;
        unpack_target.aux1_buf = a8_mid_ra.buffer->native_mtl_buffer();
        unpack_target.aux1_offset = a8_mid_ra.offset;
        unpack_target.out_stride = f;
        unpack_target.dst_col_offset = 0U;
        unpack_target.mp = w4a8_nax_mp(chunk_t);
      } else {
        unpack_target.mode = ane::AneUnpackMode::BF16;
        unpack_target.primary_buf = product.buffer->native_mtl_buffer();
        unpack_target.primary_offset = product.offset;
        unpack_target.out_stride = f;
        unpack_target.dst_col_offset = 0U;
        unpack_target.mp = chunk_t;
      }

      auto run_hybrid = [&]() -> double {
        const auto t0 = std::chrono::steady_clock::now();
        void *cb = canary->begin_collapsed_cb();
        for (U32 bi = 0U; bi < kBurst; ++bi) {
          encode_prof(cb, p_post_norm);
          const U64 seq = canary->encode_pack_and_commit(
              cb, op0->op_id, norm.buffer->native_mtl_buffer(), norm.offset, h);
          canary->submit_gpu_suffix_cb(
              [&](void *cb_b1) { encode_prof(cb_b1, p_mlp_gu_gpu); });
          cb = canary->begin_unpack_cb(seq, op0->op_id, unpack_target);
          encode_prof(cb, p_mlp_down);
        }
        canary->commit_collapsed_cb(cb);
        canary->flush_and_wait();
        static_cast<void>(context.submit_classic_barrier());
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() /
               double(kBurst);
      };

      try {
        for (U32 w = 0U; w < kWarmups; ++w) {
          (void)run_hybrid();
        }
        std::vector<double> samples;
        samples.reserve(kSamples);
        for (U32 s = 0U; s < kSamples; ++s) {
          samples.push_back(run_hybrid());
        }
        const double med_ms = median_of(std::move(samples));
        cand_timings.emplace_back(cand_alpha, med_ms);
        if (verbose) {
          auto measure_prof_ms = [&](Profile &pr) -> double {
            const auto s0 = std::chrono::steady_clock::now();
            void *mcb = canary->begin_collapsed_cb();
            for (U32 bi = 0U; bi < kBurst; ++bi) {
              encode_prof(mcb, pr);
            }
            canary->commit_collapsed_cb(mcb);
            canary->flush_and_wait();
            const auto s1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(s1 - s0).count() /
                   double(kBurst);
          };
          const double gu_ms = measure_prof_ms(p_mlp_gu_gpu);
          const double dn_ms = measure_prof_ms(p_mlp_down);
          const auto a0 = std::chrono::steady_clock::now();
          void *acb = canary->begin_collapsed_cb();
          for (U32 bi = 0U; bi < kBurst; ++bi) {
            const U64 seq = canary->encode_pack_and_commit(
                acb, op0->op_id, norm.buffer->native_mtl_buffer(), norm.offset,
                h);
            acb = canary->begin_unpack_cb(seq, op0->op_id, unpack_target);
          }
          canary->commit_collapsed_cb(acb);
          canary->flush_and_wait();
          const auto a1 = std::chrono::steady_clock::now();
          const double ane_ms =
              std::chrono::duration<double, std::milli>(a1 - a0).count() /
              double(kBurst);
          std::fprintf(
              stderr,
              "[ane-cal] route=%s T=%u alpha=%.3f (ane=%u/%u shared_in=%d modeB=%d) "
              "gpu_solo=%.3fms hybrid=%.3fms ratio=%.4f "
              "[gu_sfx=%.3fms dn=%.3fms ane_solo=%.3fms]\n",
              std::string(ane::to_string(base_cfg.linear_route)).c_str(),
              chunk_t, cand_alpha, op0->split.ane_total_rows, f,
              canary->stats().shared_input_surface_active ? 1 : 0,
              mode_b_a8 ? 1 : 0, gpu_solo_ms, med_ms,
              gpu_solo_ms > 0.0 ? (med_ms / gpu_solo_ms) : 0.0,
              gu_ms, dn_ms, ane_ms);
        }
        if (med_ms < best_hybrid_ms) {
          best_hybrid_ms = med_ms;
          best_alpha = cand_alpha;
        }
      } catch (...) {
        // Skip failed candidate cleanly.
      }
      canary.reset();
    }

    const bool has_valid_timing = std::isfinite(best_hybrid_ms);
    const bool active =
        force_active ||
        (has_valid_timing && gpu_solo_ms > 0.0 &&
         best_hybrid_ms <= bypass_ratio * gpu_solo_ms);
    out_best_alpha = best_alpha;
    out_reason = active ? std::string{} : std::string("calibration_bypassed");

    if (use_disk_cache && !force_active && has_valid_timing) {
      ane::AneCachedCalibrationRecord rec{};
      rec.valid = true;
      rec.active = active;
      rec.best_alpha = best_alpha;
      rec.gpu_solo_ms = gpu_solo_ms;
      rec.best_hybrid_ms = best_hybrid_ms;
      rec.reason = out_reason;
      rec.candidate_timings_ms = cand_timings;
      ane::save_cached_calibration(base_cfg.cache_root, cal_key, rec);
    }
    return active;
  }

  UltraSwiGluChunkPlan &get_or_build_ultra_swiglu_chunk_plan(
      U32 context_envelope,
      ChunkAttentionRoute attn_route,
      bool /*trunk_only*/) {
    // Share the 64-layer plan across trunk_only=true and trunk_only=false;
    // head_sample is built once and only executed when !trunk_only.
    const AneChunkPlanKey plan_key{context_envelope, attn_route, false};
    if (auto it = ane_ultra_plans_.find(plan_key);
        it != ane_ultra_plans_.end()) {
      return it->second;
    }
    const U32 chunk_t = ane_chunk_tokens_;
    const bool use_w4a8 = use_w4a8_for_t(chunk_t, false);
    const bool affine_xs_eligible =
        (!use_w4a8 && affine_nax_active() && affine_nax_xs.buffer != nullptr &&
         (h % 64U) == 0U);
    const std::string pkey_sfx = ".ane_ultra.c" + std::to_string(chunk_t) +
                                 ".e" + std::to_string(context_envelope) +
                                 ".ar" + std::to_string(U32(attn_route)) +
                                 plan_key_suffix();

    auto block_input_needs_bf16 = [&](const Block &blk) -> bool {
      if (inspect_tensors_workspace)
        return true;
      if (blk.type == LayerType::full_attention) {
        return block_has_any_bypass(
                   blk, {"attention_q", "attention_k", "attention_v"}) ||
               !can_w4a8_weight(blk.get("attention_q"), 2 * q, h) ||
               !can_w4a8_weight(blk.get("attention_k"), kv, h) ||
               !can_w4a8_weight(blk.get("attention_v"), kv, h);
      }
      const U32 hv = U32(config.linear_num_value_heads);
      return block_has_any_bypass(blk, {"gdn_qkv", "gdn_z", "gdn_a", "gdn_b"}) ||
             !can_w4a8_weight(blk.get("gdn_qkv"), conv, h) ||
             !can_w4a8_weight(blk.get("gdn_z"), gv, h) ||
             !can_w4a8_weight(blk.get("gdn_a"), hv, h) ||
             !can_w4a8_weight(blk.get("gdn_b"), hv, h);
    };

    UltraSwiGluChunkPlan plan;
    plan.chunk_t = chunk_t;
    plan.context_envelope = context_envelope;
    plan.attn_route = attn_route;
    plan.trunk_only = false;
    plan.used_w4a8 = use_w4a8;

    // 1. Embedding + Layer-0 input RMSNorm.
    {
      Encoder e(context, libraries, chunk_t, "qwen.ultra.emb0" + pkey_sfx);
      e.begin_stage("embedding");
      embedding_lookup(e, embedding, token_ids, hidden, zero, v, h);
      const bool in0_f16 = !use_w4a8 && block_input_nax(blocks[0], chunk_t);
      if (use_w4a8) {
        emit_rmsnorm_a8_g64(e, hidden, direct(blocks[0].get("input_norm"), h),
                            norm, a8_qa, a8_sa, a8_ra, h,
                            float(config.rms_norm_eps),
                            normalization_weight_offset,
                            block_input_needs_bf16(blocks[0]));
      } else if (in0_f16) {
        normalize_f16(e, blocks[0].get("input_norm"), hidden, f16_norm);
      } else if (affine_xs_eligible) {
        emit_affine_rmsnorm_xsum(e, hidden,
                                 direct(blocks[0].get("input_norm"), h), norm,
                                 affine_nax_xs, h, float(config.rms_norm_eps),
                                 normalization_weight_offset);
      } else {
        normalize(e, blocks[0].get("input_norm"), hidden, norm);
      }
      plan.embed_norm0 = e.finish();
    }

    plan.layers.resize(blocks.size());
    for (U32 li = 0U; li < blocks.size(); ++li) {
      const auto &block = blocks[li];
      auto &lp = plan.layers[li];
      const std::string ltag = ".L" + std::to_string(li) + pkey_sfx;
      const bool in_f16 = !use_w4a8 && block_input_nax(block, chunk_t);
      const bool in_affine_xs = (!use_w4a8 && !in_f16 && affine_xs_eligible);

      // 2. Unbroken per-layer pre-MLP GPU trunk:
      //    Full-Attn (QKV concat + Sage2/MInf/Exact + o_proj) OR
      //    GDN (QKVZ concat + AB concat + GDN Mega Stage-3 fusion + gdn_out)
      //    + post_attention_norm (always writes BF16 norm for ANE pack).
      {
        Encoder e(context, libraries, chunk_t, "qwen.ultra.pre_mlp" + ltag);
        e.begin_stage("pre_mlp");
        if (use_w4a8) {
          e.set_a8_quant(make_a8_key(e, norm, h));
        } else if (in_affine_xs) {
          e.set_group_sum(Encoder::GroupSumKey{
              affine_nax_xs.buffer, affine_nax_xs.offset, norm.buffer,
              norm.offset, h, 64U, chunk_t, true, 2U});
        }
        if (block.type == LayerType::full_attention) {
          attention(e, block, /*split=*/false, context_envelope, in_f16, {},
                    nullptr, std::nullopt, use_w4a8, attn_route);
        } else {
          gdn(e, block, /*verify=*/false, in_f16, nullptr, std::nullopt,
              use_w4a8);
        }
        plan.used_sage2 |= e.profile.used_sage2;
        plan.used_minference |= e.profile.used_minference;

        if (use_w4a8) {
          emit_add_rmsnorm_a8_g64(
              e, hidden, delta, direct(block.get("post_attention_norm"), h),
              norm, a8_qa, a8_sa, a8_ra, h, float(config.rms_norm_eps),
              normalization_weight_offset, /*write_bf16=*/true);
        } else if (affine_xs_eligible) {
          emit_affine_add_rmsnorm_xsum(
              e, hidden, delta, direct(block.get("post_attention_norm"), h),
              norm, affine_nax_xs, h, float(config.rms_norm_eps),
              normalization_weight_offset);
        } else {
          add_normalize(e, block.get("post_attention_norm"), hidden, delta,
                        norm);
        }
        lp.pre_mlp = e.finish();
      }

      // 3. CB_B1: MLP Gate+Up+SwiGLU GPU suffix [N_ane, f) overlapping Dual-ANE [0, N_ane),
      //    with Mode B (also_write_bf16=false) FWHT-64 + A8 splice when down_proj uses FWHT-64!
      const auto &gw = block.get("mlp_gate");
      const auto &uw = block.get("mlp_up");
      const auto &dw = block.get("mlp_down");
      const auto *op =
          ane_engine_->find_op(li, ane::AneProjectionRole::MlpGateUpSwiGlu);
      const bool offload_gu =
          (op != nullptr && op->split.enabled() &&
           can_gpu_suffix_swiglu(gw, uw, f, h, use_w4a8));
      const bool down_w4a8 = use_w4a8 && (dw.has_fwht64() || w4a8_down_proj_);
      bool used_mode_b_a8 = false;
      {
        Encoder e(context, libraries, chunk_t, "qwen.ultra.mlp_gu" + ltag);
        e.begin_stage("mlp_gu");
        if (use_w4a8) {
          e.set_a8_quant(make_a8_key(e, norm, h));
        } else if (affine_xs_eligible) {
          e.set_group_sum(Encoder::GroupSumKey{
              affine_nax_xs.buffer, affine_nax_xs.offset, norm.buffer,
              norm.offset, h, 64U, chunk_t, true, 2U});
        }
        if (offload_gu) {
          lp.mlp_gu_op_id = op->op_id;
          if (can_gpu_suffix_swiglu_fwht64_a8(gw, uw, dw, f, h, use_w4a8) &&
              try_linear_w4a8_swiglu_fwht64_a8(
                  e, gw, uw, norm, product, f, h, /*also_write_bf16=*/false,
                  std::nullopt, op->split.gpu_row_offset, f,
                  op->split.gpu_rows)) {
            used_mode_b_a8 = true;
            lp.unpack_target.mode = ane::AneUnpackMode::Fwht64A8;
            lp.unpack_target.primary_buf =
                a8_mid_qa.buffer->native_mtl_buffer();
            lp.unpack_target.primary_offset = a8_mid_qa.offset;
            lp.unpack_target.aux0_buf = a8_mid_sa.buffer->native_mtl_buffer();
            lp.unpack_target.aux0_offset = a8_mid_sa.offset;
            lp.unpack_target.aux1_buf = a8_mid_ra.buffer->native_mtl_buffer();
            lp.unpack_target.aux1_offset = a8_mid_ra.offset;
            lp.unpack_target.out_stride = f;
            lp.unpack_target.dst_col_offset = 0U;
            lp.unpack_target.mp = w4a8_nax_mp(chunk_t);
          } else {
            (void)emit_gpu_suffix_swiglu(e, gw, uw, norm, product, f, h,
                                         use_w4a8, op->split.gpu_row_offset, f,
                                         op->split.gpu_rows);
            lp.unpack_target.mode = ane::AneUnpackMode::BF16;
            lp.unpack_target.primary_buf = product.buffer->native_mtl_buffer();
            lp.unpack_target.primary_offset = product.offset;
            lp.unpack_target.out_stride = f;
            lp.unpack_target.dst_col_offset = 0U;
            lp.unpack_target.mp = chunk_t;
          }
        } else if (down_w4a8 && dw.has_fwht64() && can_w4a8_weight(dw, h, f) &&
                   try_linear_w4a8_swiglu_fwht64_a8(
                       e, gw, uw, norm, product, f, h,
                       /*also_write_bf16=*/false)) {
          used_mode_b_a8 = true;
        } else if (use_w4a8 &&
                   try_linear_w4a8_swiglu(e, gw, uw, norm, product, f, h)) {
        } else if (!try_linear_swiglu(e, gw, uw, norm, product, f, h)) {
          linear_base(e, gw, norm, gate, f, h);
          linear_base(e, uw, norm, up, f, h);
          swiglu(e, chunk_t, f);
        }
        lp.mlp_gu_gpu = e.finish();
      }

      // 4. Post-MLP GPU work fused into the unpack CB:
      //    mlp_down (consuming Mode B a8_mid directly!) + residual add + next layer input_norm.
      {
        Encoder e(context, libraries, chunk_t, "qwen.ultra.post_mlp" + ltag);
        e.begin_stage("post_mlp");
        if (used_mode_b_a8) {
          e.set_a8_quant(make_a8_key_for(e, product, f, a8_mid_qa, a8_mid_sa,
                                         a8_mid_ra, std::nullopt,
                                         /*fwht64=*/true));
        }
        if (down_w4a8 && try_linear_w4a8_base(e, dw, product, delta, h, f)) {
        } else {
          linear(e, dw, product, delta, h, f);
        }
        if (li + 1U < blocks.size()) {
          const auto &next_blk = blocks[li + 1U];
          const bool next_in_f16 = !use_w4a8 && block_input_nax(next_blk, chunk_t);
          if (use_w4a8) {
            emit_add_rmsnorm_a8_g64(
                e, hidden, delta, direct(next_blk.get("input_norm"), h), norm,
                a8_qa, a8_sa, a8_ra, h, float(config.rms_norm_eps),
                normalization_weight_offset, block_input_needs_bf16(next_blk));
          } else if (next_in_f16) {
            add_normalize_f16(e, next_blk.get("input_norm"), hidden, delta,
                              f16_norm);
          } else if (affine_xs_eligible) {
            emit_affine_add_rmsnorm_xsum(
                e, hidden, delta, direct(next_blk.get("input_norm"), h), norm,
                affine_nax_xs, h, float(config.rms_norm_eps),
                normalization_weight_offset);
          } else {
            add_normalize(e, next_blk.get("input_norm"), hidden, delta, norm);
          }
        } else {
          add(e);
        }
        lp.post_mlp = e.finish();
      }
    }

    // 5. Final head + sample (built once, executed only when !trunk_only).
    {
      Encoder e(context, libraries, chunk_t, "qwen.ultra.head" + pkey_sfx);
      e.begin_stage("head");
      Ref last_hidden{delta.buffer, 0, mul(h, 2)};
      Ref last_norm{norm.buffer, 0, mul(h, 2)};
      Ref row0_logits{logits.buffer, 0, mul(v, 2)};
      auto &extract =
          e.emit("qwen_forward", "omlx_m5_forward_last_row", {.width = h},
                 {.width = 256}, {hidden, last_hidden});
      e.dynamic(extract, 2, words({chunk_t, h}));
      normalize(e, final_norm, last_hidden, last_norm, 1U);
      linear(e, head, last_norm, row0_logits, v, h, 1U);
      e.begin_stage("sample");
      const bool use_64tg_argmax =
          (v >= 256U) && (argmax_scratch.buffer != nullptr) &&
          !detail::forward::env_flag_enabled("OMLX_M5_DISABLE_64TG_ARGMAX");
      if (sampling_contract == SamplingContract::PositionCoupledV1) {
        if (use_64tg_argmax) {
          auto &st1 = e.emit("position_sampling",
                             "omlx_m5_position_argmax_stage1_64tg",
                             {.width = 64U * 256U, .height = 1U},
                             {.width = 256U}, {row0_logits, argmax_scratch});
          e.constant(st1, 2, words({v, 1, 1, 0}));
          auto &sample = e.emit(
              "position_sampling", "omlx_m5_position_sample_logits_stage2",
              {.width = 256U}, {.width = 256U},
              {row0_logits, argmax_scratch, sampled, position_ids,
               position_scores});
          e.profile.sampling_offset = e.dynamic(
              sample, 5, std::vector<std::byte>(sizeof(PositionSamplingRow)));
          e.constant(sample, 6, words({v, 1, 1, 0}));
        } else {
          auto &sample =
              e.emit("position_sampling", "omlx_m5_position_sample_logits",
                     {.width = 256}, {.width = 256},
                     {row0_logits, sampled, position_ids, position_scores});
          e.profile.sampling_offset = e.dynamic(
              sample, 4, std::vector<std::byte>(sizeof(PositionSamplingRow)));
          e.constant(sample, 5, words({v, 1, 1, 0}));
        }
      } else if (use_64tg_argmax) {
        Ref vocab_parts{argmax_scratch.buffer, 0, 512U};
        auto &st1 =
            e.emit("vocab", "omlx_m5_vocab_argmax_stage1_64tg",
                   {.width = 64U * 256U}, {.width = 256U},
                   {row0_logits, vocab_parts});
        e.constant(st1, 2, words({v}));
        auto &sample =
            e.emit("vocab", "omlx_m5_sample_logits_stage2", {.width = 256U},
                   {.width = 256U}, {row0_logits, vocab_parts, sampled});
        e.profile.sampling_offset =
            e.dynamic(sample, 3, std::vector<std::byte>(sizeof(SampleParams)));
      } else {
        auto &sample = e.emit("vocab", "omlx_m5_sample_logits", {.width = 256},
                              {.width = 256}, {row0_logits, sampled});
        e.profile.sampling_offset =
            e.dynamic(sample, 2, std::vector<std::byte>(sizeof(SampleParams)));
      }
      plan.head_sample = e.finish();
    }

    auto [it, _] = ane_ultra_plans_.emplace(plan_key, std::move(plan));
    return it->second;
  }

  QwenForwardResult execute_ultra_swiglu_chunk(
      std::span<const U32> tokens,
      const GpuSamplingOptions &sampling,
      bool trunk_only) {
    const U32 count = U32(tokens.size());
    const U32 start_pos = next_position;
    const U32 end_pos = start_pos + count;
    const auto route = select_chunk_attention_route(count, start_pos);
    const U32 env =
        (route == ChunkAttentionRoute::Sage2MInference ||
         (route == ChunkAttentionRoute::Sage2Dense && max_context > 4096U))
            ? select_long_ctx_envelope(end_pos)
            : 0U;
    auto &plan = get_or_build_ultra_swiglu_chunk_plan(env, route, trunk_only);

    std::fill_n(token_staging.begin(), count, 0U);
    std::copy(tokens.begin(), tokens.end(), token_staging.begin());
    const_cast<MetalBuffer *>(token_ids.buffer)
        ->write(std::as_bytes(std::span(token_staging).first(count)));
    if (attention_positions.buffer) {
      std::vector<U32> positions(count);
      for (U32 i = 0U; i < count; ++i)
        positions[i] = start_pos + i;
      const_cast<MetalBuffer *>(attention_positions.buffer)
          ->write(std::as_bytes(std::span(positions)));
    }

    auto patch_prof = [&](std::optional<Profile> &opt_prof) {
      if (!opt_prof.has_value() || opt_prof->patches.empty()) {
        return;
      }
      auto &prof = *opt_prof;
      for (const auto &patch : prof.patches) {
        const U32 val =
            patch.position ? start_pos : count * patch.multiplier;
        std::memcpy(prof.parameter_data.data() + patch.offset + patch.word * 4,
                    &val, 4);
      }
      prof.parameters.write("step", prof.parameter_data);
    };
    patch_prof(plan.embed_norm0);
    for (auto &lp : plan.layers) {
      patch_prof(lp.pre_mlp);
      patch_prof(lp.mlp_gu_gpu);
      patch_prof(lp.post_mlp);
    }
    if (!trunk_only && plan.head_sample.has_value()) {
      auto &hp = *plan.head_sample;
      if (sampling_contract == SamplingContract::PositionCoupledV1) {
        const PositionSamplingRow row{.seed = sampling.seed,
                                      .position = start_pos + count,
                                      .top_k = sampling.top_k,
                                      .temperature = sampling.temperature,
                                      .top_p = sampling.top_p};
        std::memcpy(hp.parameter_data.data() + hp.sampling_offset, &row,
                    sizeof(row));
      } else {
        const SampleParams sp{v,
                              sampling.top_k,
                              sampling.temperature,
                              sampling.top_p,
                              sampling.seed,
                              sampling.stream,
                              start_pos + count - 1U};
        std::memcpy(hp.parameter_data.data() + hp.sampling_offset, &sp,
                    sizeof(sp));
      }
      hp.parameters.write("step", hp.parameter_data);
    }

    auto *pool = page_pool.get();
    if (pool && pool->tokens() != start_pos)
      throw std::logic_error("Qwen ANE KV page frontier disagrees with model");
    const auto page_copies =
        pool ? pool->begin_append(count) : std::vector<KvPageCopy>{};
    struct PageGuard {
      KvPagePool *pool;
      bool active;
      ~PageGuard() {
        if (active)
          try {
            pool->abort();
          } catch (...) {
          }
      }
    } page_guard{pool, pool != nullptr};
    execute_cow_copies(page_copies);

    auto encode_one = [&](void *mtl_cb, Profile &pr) {
      const std::array calls{MetalCommandInvocation{
          .plan = &pr.plan, .bindings = pr.bindings, .parameters = &pr.parameters}};
      context.encode_into_command_buffer(mtl_cb, calls);
    };
    auto encode_two = [&](void *mtl_cb, Profile &p0, Profile &p1) {
      const std::array calls{
          MetalCommandInvocation{
              .plan = &p0.plan, .bindings = p0.bindings, .parameters = &p0.parameters},
          MetalCommandInvocation{
              .plan = &p1.plan, .bindings = p1.bindings, .parameters = &p1.parameters}};
      context.encode_into_command_buffer(mtl_cb, calls);
    };

    // Collapsed minimal-CB overlap schedule (~65 trunk CBs/chunk):
    //   CB_0:   embed_norm0 + pre_mlp[0] + ane_pack[0] + signal(ane_ready, 1)
    //   CB_B1:  mlp_gu_gpu[li] [N_ane, f) (overlaps Dual-ANE [0, N_ane)!)
    //   CB_k:   wait(ane_done, seq) + fused_unpack[li] + signal(unpack_done, seq)
    //           + post_mlp[li] + pre_mlp[li+1] + ane_pack[li+1] + signal(ane_ready, seq+1)
    void *cb = ane_engine_->begin_collapsed_cb();
    if (!plan.layers.empty()) {
      encode_two(cb, *plan.embed_norm0, *plan.layers[0].pre_mlp);
    } else {
      encode_one(cb, *plan.embed_norm0);
    }

    for (U32 li = 0U; li < plan.layers.size(); ++li) {
      auto &lp = plan.layers[li];
      if (lp.mlp_gu_op_id.has_value()) {
        const U64 seq = ane_engine_->encode_pack_and_commit(
            cb, *lp.mlp_gu_op_id, norm.buffer->native_mtl_buffer(), norm.offset,
            h);
        ane_engine_->submit_gpu_suffix_cb(
            [&](void *cb_b1) { encode_one(cb_b1, *lp.mlp_gu_gpu); });
        cb = ane_engine_->begin_unpack_cb(seq, *lp.mlp_gu_op_id,
                                          lp.unpack_target);
      } else {
        encode_one(cb, *lp.mlp_gu_gpu);
      }
      if (li + 1U < plan.layers.size()) {
        encode_two(cb, *lp.post_mlp, *plan.layers[li + 1U].pre_mlp);
      } else if (!trunk_only && plan.head_sample.has_value()) {
        encode_two(cb, *lp.post_mlp, *plan.head_sample);
      } else {
        encode_one(cb, *lp.post_mlp);
      }
    }
    ane_engine_->commit_collapsed_cb(cb);

    ane_engine_->flush_and_wait();
    MetalSubmission sub{.timing = context.submit_classic_barrier()};

    QwenForwardResult result{
        .tokens = count,
        .executed_layers = U32(blocks.size()),
        .submission = std::move(sub),
    };
    if (!trunk_only) {
      const auto tok_bytes = sampled.buffer->read(
          0, sampling_contract == SamplingContract::PositionCoupledV1 ? 8 : 4);
      std::memcpy(&result.token, tok_bytes.data(), 4);
      result.host_read_bytes = tok_bytes.size();
      if (sampling_contract == SamplingContract::PositionCoupledV1) {
        U32 flag = 0;
        std::memcpy(&flag, tok_bytes.data() + 4, 4);
        if (flag & 2U)
          throw std::runtime_error("nonfinite position target logits");
        if (flag & 1U) {
          auto ib = position_ids.buffer->read(0, U64(sampling.top_k) * 4);
          auto sb = position_scores.buffer->read(0, U64(sampling.top_k) * 4);
          std::vector<U32> ids(sampling.top_k);
          std::vector<float> scores(sampling.top_k);
          std::memcpy(ids.data(), ib.data(), ib.size());
          std::memcpy(scores.data(), sb.data(), sb.size());
          result.token = position_sample_candidates(
              ids, scores,
              {.seed = sampling.seed,
               .position = start_pos + count,
               .top_k = sampling.top_k,
               .temperature = sampling.temperature,
               .top_p = sampling.top_p});
          result.host_read_bytes += ib.size() + sb.size();
        }
      } else if (result.token >= v) {
        throw std::runtime_error("nonfinite or invalid legacy sampled token");
      }
    }

    if (pool)
      pool->commit(count);
    page_guard.active = false;
    next_position += count;
    last_logits_row = 0U;
    has_last_logits = !trunk_only;
    record_prefill_telemetry(count, start_pos, plan.used_w4a8, plan.used_sage2,
                             plan.used_minference, plan.chunk_t,
                             plan.context_envelope, /*used_ane=*/true,
                             &result.prefill_delta);
    return result;
  }

  AneChunkPlan &get_or_build_ane_chunk_plan(U32 context_envelope,
                                            ChunkAttentionRoute attn_route,
                                            bool trunk_only) {
    const AneChunkPlanKey plan_key{context_envelope, attn_route, trunk_only};
    if (auto it = ane_layer_plans_.find(plan_key);
        it != ane_layer_plans_.end()) {
      return it->second;
    }
    const U32 chunk_t = ane_chunk_tokens_;
    const bool use_w4a8 = use_w4a8_for_t(chunk_t, false);
    const std::string pkey_sfx = ".ane.c" + std::to_string(chunk_t) + ".e" +
                                 std::to_string(context_envelope) + ".ar" +
                                 std::to_string(U32(attn_route)) +
                                 (trunk_only ? ".tr" : ".fl") +
                                 plan_key_suffix();

    AneChunkPlan plan;
    plan.chunk_t = chunk_t;
    plan.context_envelope = context_envelope;
    plan.attn_route = attn_route;
    plan.trunk_only = trunk_only;
    plan.used_w4a8 = use_w4a8;

    // 1. Embedding + Layer-0 input RMSNorm (always write BF16 norm for ANE!).
    {
      Encoder e(context, libraries, chunk_t, "qwen.ane.emb0" + pkey_sfx);
      e.begin_stage("embedding");
      embedding_lookup(e, embedding, token_ids, hidden, zero, v, h);
      if (use_w4a8) {
        emit_rmsnorm_a8_g64(e, hidden, direct(blocks[0].get("input_norm"), h),
                            norm, a8_qa, a8_sa, a8_ra, h,
                            float(config.rms_norm_eps),
                            normalization_weight_offset, /*write_bf16=*/true);
      } else {
        normalize(e, blocks[0].get("input_norm"), hidden, norm);
      }
      plan.embed_norm0 = e.finish();
    }

    plan.layers.resize(blocks.size());
    const U32 hv = U32(config.linear_num_value_heads);

    for (U32 li = 0; li < blocks.size(); ++li) {
      const auto &block = blocks[li];
      auto &lp = plan.layers[li];
      lp.is_full_attn = (block.type == LayerType::full_attention);
      const std::string ltag = ".L" + std::to_string(li) + pkey_sfx;

      if (lp.is_full_attn) {
        // Full-Attn Q/K/V + Attention state step (100% GPU: active Route A + Route B).
        {
          Encoder e(context, libraries, chunk_t, "qwen.ane.attn_pre" + ltag);
          e.begin_stage("attn_pre");
          if (use_w4a8) {
            e.set_a8_quant(make_a8_key(e, norm, h));
            const std::array<MaterializedWeightView, 3> parts{
                block.get("attention_q"), block.get("attention_k"),
                block.get("attention_v")};
            const std::array<U32, 3> part_ns{2 * q, kv, kv};
            if (try_concat_w4a8_nax(e, block.concat_qkv_id, parts, part_ns, h,
                                    norm, concat_out)) {
              emit_split3(e, concat_out, query, key, value, 2 * q, kv, kv);
            } else {
              if (!try_linear_w4a8_base(e, block.get("attention_q"), norm,
                                        query, 2 * q, h))
                linear_base(e, block.get("attention_q"), norm, query, 2 * q, h);
              if (!try_linear_w4a8_base(e, block.get("attention_k"), norm, key,
                                        kv, h))
                linear_base(e, block.get("attention_k"), norm, key, kv, h);
              if (!try_linear_w4a8_base(e, block.get("attention_v"), norm,
                                        value, kv, h))
                linear_base(e, block.get("attention_v"), norm, value, kv, h);
            }
          } else {
            const std::array<MaterializedWeightView, 3> parts{
                block.get("attention_q"), block.get("attention_k"),
                block.get("attention_v")};
            const std::array<U32, 3> part_ns{2 * q, kv, kv};
            if (try_concat_nax(e, block.concat_qkv_id, parts, part_ns, h, false,
                               norm, concat_out)) {
              emit_split3(e, concat_out, query, key, value, 2 * q, kv, kv);
            } else {
              linear_base(e, block.get("attention_q"), norm, query, 2 * q, h);
              linear_base(e, block.get("attention_k"), norm, key, kv, h);
              linear_base(e, block.get("attention_v"), norm, value, kv, h);
            }
          }
          emit_attention_state_step(
              e, block, block, *page_pool, max_context, kv_format_, query, key,
              value, mixer, attention_positions, chunk_t, false,
              context_envelope, true, std::nullopt, nullptr, attn_route);
          plan.used_sage2 |= e.profile.used_sage2;
          plan.used_minference |= e.profile.used_minference;
          lp.attn_pre_o = e.finish();
        }
        // Full-Attn o_proj: Dual-ANE [0..N_ane) || GPU suffix [N_ane..h).
        {
          const auto &o_w = block.get("attention_o");
          const auto *op =
              ane_engine_->find_op(li, ane::AneProjectionRole::AttnOProj);
          const bool offload_o =
              (op != nullptr && op->split.enabled() &&
               can_gpu_suffix_linear(o_w, h, q, use_w4a8));
          Encoder e(context, libraries, chunk_t, "qwen.ane.attn_o" + ltag);
          e.begin_stage("attn_o");
          if (offload_o) {
            lp.attn_o_op_id = op->op_id;
            (void)emit_gpu_suffix_linear(e, o_w, mixer, delta, h, q, use_w4a8,
                                         op->split.gpu_row_offset, h,
                                         op->split.gpu_rows);
          } else if (use_w4a8 &&
                     try_linear_w4a8_base(e, o_w, mixer, delta, h, q)) {
          } else {
            linear(e, o_w, mixer, delta, h, q);
          }
          lp.attn_o_gpu = e.finish();
        }
      } else {
        // GDN in_proj_qkv/a/b: STRICTLY 100% GPU (NEVER ANE!).
        {
          Encoder e(context, libraries, chunk_t, "qwen.ane.gdn_qkvab" + ltag);
          e.begin_stage("gdn_qkvab");
          if (use_w4a8) {
            e.set_a8_quant(make_a8_key(e, norm, h));
            if (!try_linear_w4a8_base(e, block.get("gdn_qkv"), norm, qkv, conv,
                                      h))
              linear_base(e, block.get("gdn_qkv"), norm, qkv, conv, h);
            const std::array<MaterializedWeightView, 2> ab_parts{
                block.get("gdn_a"), block.get("gdn_b")};
            const std::array<U32, 2> ab_ns{hv, hv};
            if (try_concat_w4a8_nax(e, block.concat_gdn_ab_id, ab_parts, ab_ns,
                                    h, norm, concat_out)) {
              emit_split2(e, concat_out, a, b, hv, hv);
            } else {
              if (!try_linear_w4a8_base(e, block.get("gdn_a"), norm, a, hv, h))
                linear_base(e, block.get("gdn_a"), norm, a, hv, h);
              if (!try_linear_w4a8_base(e, block.get("gdn_b"), norm, b, hv, h))
                linear_base(e, block.get("gdn_b"), norm, b, hv, h);
            }
          } else {
            linear_base(e, block.get("gdn_qkv"), norm, qkv, conv, h);
            const std::array<MaterializedWeightView, 2> ab_parts{
                block.get("gdn_a"), block.get("gdn_b")};
            const std::array<U32, 2> ab_ns{hv, hv};
            if (try_concat_nax(e, block.concat_gdn_ab_id, ab_parts, ab_ns, h,
                               false, norm, concat_out)) {
              emit_split2(e, concat_out, a, b, hv, hv);
            } else {
              linear_base(e, block.get("gdn_a"), norm, a, hv, h);
              linear_base(e, block.get("gdn_b"), norm, b, hv, h);
            }
          }
          lp.gdn_qkv_ab = e.finish();
        }
        // GDN in_proj_z: Dual-ANE [0..N_ane) || GPU suffix [N_ane..gv).
        {
          const auto &z_w = block.get("gdn_z");
          const auto *op =
              ane_engine_->find_op(li, ane::AneProjectionRole::GdnInProjZ);
          const bool offload_z =
              (op != nullptr && op->split.enabled() &&
               can_gpu_suffix_linear(z_w, gv, h, use_w4a8));
          Encoder e(context, libraries, chunk_t, "qwen.ane.gdn_z" + ltag);
          e.begin_stage("gdn_z");
          if (use_w4a8) {
            e.set_a8_quant(make_a8_key(e, norm, h));
          }
          if (offload_z) {
            lp.gdn_z_op_id = op->op_id;
            (void)emit_gpu_suffix_linear(e, z_w, norm, z, gv, h, use_w4a8,
                                         op->split.gpu_row_offset, gv,
                                         op->split.gpu_rows);
          } else if (use_w4a8 &&
                     try_linear_w4a8_base(e, z_w, norm, z, gv, h)) {
          } else {
            linear(e, z_w, norm, z, gv, h);
          }
          lp.gdn_z_gpu = e.finish();
        }
        // GDN Conv1D + Recurrence + NormGate (writes BF16 mixer + A8 if W4A8).
        const bool stage3_req_a8 =
            use_w4a8 && gdn_chunk_nax_enabled_ &&
            chunk_t >= gdn_chunk_nax_min_tokens() &&
            can_w4a8_weight(block.get("gdn_out"), h, gv);
        bool stage3_wrote_a8 = false;
        {
          Encoder e(context, libraries, chunk_t, "qwen.ane.gdn_rec" + ltag);
          e.begin_stage("gdn_rec");
          emit_gdn_state_step(
              e, block, block, qkv, z, a, b, convolved, recurrent, mixer,
              chunk_t, /*verify=*/false, /*dynamic_tokens=*/true,
              /*allow_f16_out=*/false, 0U, std::nullopt, nullptr,
              /*out_w4a8=*/stage3_req_a8,
              /*out_w4a8_write_bf16=*/true, /*emit_gdn_out_linear=*/false,
              &stage3_wrote_a8);
          lp.gdn_rec = e.finish();
        }
        // GDN out_proj: Dual-ANE [0..N_ane) || GPU suffix [N_ane..h).
        {
          const auto &out_w = block.get("gdn_out");
          const auto *op =
              ane_engine_->find_op(li, ane::AneProjectionRole::GdnOutProj);
          const bool offload_out =
              (op != nullptr && op->split.enabled() &&
               can_gpu_suffix_linear(out_w, h, gv, use_w4a8));
          Encoder e(context, libraries, chunk_t, "qwen.ane.gdn_out" + ltag);
          e.begin_stage("gdn_out");
          if (stage3_wrote_a8) {
            e.set_a8_quant(make_a8_key(e, mixer, gv));
          }
          if (offload_out) {
            lp.gdn_out_op_id = op->op_id;
            (void)emit_gpu_suffix_linear(e, out_w, mixer, delta, h, gv, use_w4a8,
                                         op->split.gpu_row_offset, h,
                                         op->split.gpu_rows);
          } else if (use_w4a8 &&
                     try_linear_w4a8_base(e, out_w, mixer, delta, h, gv)) {
          } else {
            linear(e, out_w, mixer, delta, h, gv);
          }
          lp.gdn_out_gpu = e.finish();
        }
      }

      // Post-attention residual add + RMSNorm (always write BF16 norm for ANE!).
      {
        Encoder e(context, libraries, chunk_t, "qwen.ane.post_norm" + ltag);
        e.begin_stage("post_norm");
        if (use_w4a8) {
          emit_add_rmsnorm_a8_g64(
              e, hidden, delta, direct(block.get("post_attention_norm"), h),
              norm, a8_qa, a8_sa, a8_ra, h, float(config.rms_norm_eps),
              normalization_weight_offset, /*write_bf16=*/true);
        } else {
          add_normalize(e, block.get("post_attention_norm"), hidden, delta,
                        norm);
        }
        lp.post_norm = e.finish();
      }

      // Dense MLP Fused Gate+Up+SwiGLU: Dual-ANE [0..N_ane) || GPU suffix [N_ane..f).
      {
        const auto &gw = block.get("mlp_gate");
        const auto &uw = block.get("mlp_up");
        const auto *op =
            ane_engine_->find_op(li, ane::AneProjectionRole::MlpGateUpSwiGlu);
        const bool offload_gu =
            (op != nullptr && op->split.enabled() &&
             can_gpu_suffix_swiglu(gw, uw, f, h, use_w4a8));
        Encoder e(context, libraries, chunk_t, "qwen.ane.mlp_gu" + ltag);
        e.begin_stage("mlp_gu");
        if (use_w4a8) {
          e.set_a8_quant(make_a8_key(e, norm, h));
        }
        if (offload_gu) {
          lp.mlp_gu_op_id = op->op_id;
          (void)emit_gpu_suffix_swiglu(e, gw, uw, norm, product, f, h, use_w4a8,
                                       op->split.gpu_row_offset, f,
                                       op->split.gpu_rows);
        } else if (use_w4a8 &&
                   try_linear_w4a8_swiglu(e, gw, uw, norm, product, f, h)) {
        } else if (!try_linear_swiglu(e, gw, uw, norm, product, f, h)) {
          linear_base(e, gw, norm, gate, f, h);
          linear_base(e, uw, norm, up, f, h);
          swiglu(e, chunk_t, f);
        }
        lp.mlp_gu_gpu = e.finish();
      }

      // Dense MLP Down: Dual-ANE [0..N_ane) || GPU suffix [N_ane..h) (W4A16 default).
      {
        const auto &dw = block.get("mlp_down");
        const bool down_w4a8 = use_w4a8 && (dw.has_fwht64() || w4a8_down_proj_);
        const auto *op =
            ane_engine_->find_op(li, ane::AneProjectionRole::MlpDown);
        const bool offload_dn =
            (op != nullptr && op->split.enabled() &&
             can_gpu_suffix_linear(dw, h, f, down_w4a8));
        Encoder e(context, libraries, chunk_t, "qwen.ane.mlp_dn" + ltag);
        e.begin_stage("mlp_dn");
        if (offload_dn) {
          lp.mlp_down_op_id = op->op_id;
          (void)emit_gpu_suffix_linear(e, dw, product, delta, h, f, down_w4a8,
                                       op->split.gpu_row_offset, h,
                                       op->split.gpu_rows);
        } else if (down_w4a8 &&
                   try_linear_w4a8_base(e, dw, product, delta, h, f)) {
        } else {
          linear(e, dw, product, delta, h, f);
        }
        lp.mlp_down_gpu = e.finish();
      }

      // Residual add + next layer input RMSNorm (or final add).
      {
        Encoder e(context, libraries, chunk_t, "qwen.ane.next_norm" + ltag);
        e.begin_stage("next_norm");
        if (li + 1U < blocks.size()) {
          if (use_w4a8) {
            emit_add_rmsnorm_a8_g64(
                e, hidden, delta,
                direct(blocks[li + 1U].get("input_norm"), h), norm, a8_qa, a8_sa,
                a8_ra, h, float(config.rms_norm_eps),
                normalization_weight_offset, /*write_bf16=*/true);
          } else {
            add_normalize(e, blocks[li + 1U].get("input_norm"), hidden, delta,
                          norm);
          }
        } else {
          add(e);
        }
        lp.next_norm = e.finish();
      }
    }

    // Final head + sample on last row (when !trunk_only).
    if (!trunk_only) {
      Encoder e(context, libraries, chunk_t, "qwen.ane.head" + pkey_sfx);
      e.begin_stage("head");
      Ref last_hidden{delta.buffer, 0, mul(h, 2)};
      Ref last_norm{norm.buffer, 0, mul(h, 2)};
      Ref row0_logits{logits.buffer, 0, mul(v, 2)};
      auto &extract =
          e.emit("qwen_forward", "omlx_m5_forward_last_row", {.width = h},
                 {.width = 256}, {hidden, last_hidden});
      e.dynamic(extract, 2, words({chunk_t, h}));
      normalize(e, final_norm, last_hidden, last_norm, 1U);
      linear(e, head, last_norm, row0_logits, v, h, 1U);
      e.begin_stage("sample");
      const bool use_64tg_argmax =
          (v >= 256U) && (argmax_scratch.buffer != nullptr) &&
          !detail::forward::env_flag_enabled("OMLX_M5_DISABLE_64TG_ARGMAX");
      if (sampling_contract == SamplingContract::PositionCoupledV1) {
        if (use_64tg_argmax) {
          auto &st1 = e.emit("position_sampling",
                             "omlx_m5_position_argmax_stage1_64tg",
                             {.width = 64U * 256U, .height = 1U},
                             {.width = 256U}, {row0_logits, argmax_scratch});
          e.constant(st1, 2, words({v, 1, 1, 0}));
          auto &sample = e.emit(
              "position_sampling", "omlx_m5_position_sample_logits_stage2",
              {.width = 256U}, {.width = 256U},
              {row0_logits, argmax_scratch, sampled, position_ids,
               position_scores});
          e.profile.sampling_offset = e.dynamic(
              sample, 5, std::vector<std::byte>(sizeof(PositionSamplingRow)));
          e.constant(sample, 6, words({v, 1, 1, 0}));
        } else {
          auto &sample =
              e.emit("position_sampling", "omlx_m5_position_sample_logits",
                     {.width = 256}, {.width = 256},
                     {row0_logits, sampled, position_ids, position_scores});
          e.profile.sampling_offset = e.dynamic(
              sample, 4, std::vector<std::byte>(sizeof(PositionSamplingRow)));
          e.constant(sample, 5, words({v, 1, 1, 0}));
        }
      } else if (use_64tg_argmax) {
        Ref vocab_parts{argmax_scratch.buffer, 0, 512U};
        auto &st1 =
            e.emit("vocab", "omlx_m5_vocab_argmax_stage1_64tg",
                   {.width = 64U * 256U}, {.width = 256U},
                   {row0_logits, vocab_parts});
        e.constant(st1, 2, words({v}));
        auto &sample =
            e.emit("vocab", "omlx_m5_sample_logits_stage2", {.width = 256U},
                   {.width = 256U}, {row0_logits, vocab_parts, sampled});
        e.profile.sampling_offset =
            e.dynamic(sample, 3, std::vector<std::byte>(sizeof(SampleParams)));
      } else {
        auto &sample = e.emit("vocab", "omlx_m5_sample_logits", {.width = 256},
                              {.width = 256}, {row0_logits, sampled});
        e.profile.sampling_offset =
            e.dynamic(sample, 2, std::vector<std::byte>(sizeof(SampleParams)));
      }
      plan.head_sample = e.finish();
    }

    auto [it, _] = ane_layer_plans_.emplace(plan_key, std::move(plan));
    return it->second;
  }

  QwenForwardResult execute_ane_chunk(std::span<const U32> tokens,
                                      const GpuSamplingOptions &sampling,
                                      bool trunk_only) {
    if (ane_mode_effective_ == PrefillAneMode::UltraFp16SwiGlu) {
      return execute_ultra_swiglu_chunk(tokens, sampling, trunk_only);
    }
    const U32 count = U32(tokens.size());
    const U32 start_pos = next_position;
    const U32 end_pos = start_pos + count;
    const auto route = select_chunk_attention_route(count, start_pos);
    const U32 env =
        (route == ChunkAttentionRoute::Sage2MInference ||
         (route == ChunkAttentionRoute::Sage2Dense && max_context > 4096U))
            ? select_long_ctx_envelope(end_pos)
            : 0U;
    auto &plan = get_or_build_ane_chunk_plan(env, route, trunk_only);

    std::fill_n(token_staging.begin(), count, 0U);
    std::copy(tokens.begin(), tokens.end(), token_staging.begin());
    const_cast<MetalBuffer *>(token_ids.buffer)
        ->write(std::as_bytes(std::span(token_staging).first(count)));
    if (attention_positions.buffer) {
      std::vector<U32> positions(count);
      for (U32 i = 0; i < count; ++i)
        positions[i] = start_pos + i;
      const_cast<MetalBuffer *>(attention_positions.buffer)
          ->write(std::as_bytes(std::span(positions)));
    }
    for (auto &lp : plan.layers) {
      if (lp.is_full_attn && lp.attn_pre_o.has_value()) {
        auto &prof = *lp.attn_pre_o;
        for (const auto &patch : prof.patches) {
          const U32 val =
              patch.position ? start_pos : count * patch.multiplier;
          std::memcpy(prof.parameter_data.data() + patch.offset +
                          patch.word * 4,
                      &val, 4);
        }
        prof.parameters.write("step", prof.parameter_data);
      }
    }
    if (!trunk_only && plan.head_sample.has_value()) {
      auto &hp = *plan.head_sample;
      if (sampling_contract == SamplingContract::PositionCoupledV1) {
        const PositionSamplingRow row{.seed = sampling.seed,
                                      .position = start_pos + count,
                                      .top_k = sampling.top_k,
                                      .temperature = sampling.temperature,
                                      .top_p = sampling.top_p};
        std::memcpy(hp.parameter_data.data() + hp.sampling_offset, &row,
                    sizeof(row));
      } else {
        const SampleParams sp{v,
                              sampling.top_k,
                              sampling.temperature,
                              sampling.top_p,
                              sampling.seed,
                              sampling.stream,
                              start_pos + count - 1U};
        std::memcpy(hp.parameter_data.data() + hp.sampling_offset, &sp,
                    sizeof(sp));
      }
      hp.parameters.write("step", hp.parameter_data);
    }

    auto *pool = page_pool.get();
    if (pool && pool->tokens() != start_pos)
      throw std::logic_error("Qwen ANE KV page frontier disagrees with model");
    const auto page_copies =
        pool ? pool->begin_append(count) : std::vector<KvPageCopy>{};
    struct PageGuard {
      KvPagePool *pool;
      bool active;
      ~PageGuard() {
        if (active)
          try {
            pool->abort();
          } catch (...) {
          }
      }
    } page_guard{pool, pool != nullptr};
    execute_cow_copies(page_copies);

    auto submit_prof = [&](Profile &pr) {
      const std::array calls{MetalCommandInvocation{
          .plan = &pr.plan, .bindings = pr.bindings, .parameters = &pr.parameters}};
      context.submit_classic_async(calls);
    };
    auto run_hybrid = [&](std::optional<U32> op_id, Ref src, U32 src_stride,
                          Ref dst, U32 out_stride, Profile &gpu_prof) {
      if (op_id.has_value()) {
        ane_engine_->enqueue_op(
            *op_id, src.buffer->native_mtl_buffer(), src.offset,
            dst.buffer->native_mtl_buffer(), dst.offset, out_stride,
            [&](void *mtl_cb) {
              const std::array calls{
                  MetalCommandInvocation{.plan = &gpu_prof.plan,
                                         .bindings = gpu_prof.bindings,
                                         .parameters = &gpu_prof.parameters}};
              context.encode_into_command_buffer(mtl_cb, calls);
            },
            src_stride);
      } else {
        submit_prof(gpu_prof);
      }
    };

    // Zero per-layer waitUntilCompleted(): stream entire T_chunk across GPU || Dual-ANE!
    submit_prof(*plan.embed_norm0);
    for (auto &lp : plan.layers) {
      if (lp.is_full_attn) {
        submit_prof(*lp.attn_pre_o);
        run_hybrid(lp.attn_o_op_id, mixer, q, delta, h, *lp.attn_o_gpu);
      } else {
        submit_prof(*lp.gdn_qkv_ab);
        run_hybrid(lp.gdn_z_op_id, norm, h, z, gv, *lp.gdn_z_gpu);
        submit_prof(*lp.gdn_rec);
        run_hybrid(lp.gdn_out_op_id, mixer, gv, delta, h, *lp.gdn_out_gpu);
      }
      submit_prof(*lp.post_norm);
      run_hybrid(lp.mlp_gu_op_id, norm, h, product, f, *lp.mlp_gu_gpu);
      run_hybrid(lp.mlp_down_op_id, product, f, delta, h, *lp.mlp_down_gpu);
      submit_prof(*lp.next_norm);
    }
    if (!trunk_only && plan.head_sample.has_value()) {
      submit_prof(*plan.head_sample);
    }

    ane_engine_->flush_and_wait();
    MetalSubmission sub{.timing = context.submit_classic_barrier()};

    QwenForwardResult result{
        .tokens = count,
        .executed_layers = U32(blocks.size()),
        .submission = std::move(sub),
    };
    if (!trunk_only) {
      const auto tok_bytes = sampled.buffer->read(
          0, sampling_contract == SamplingContract::PositionCoupledV1 ? 8 : 4);
      std::memcpy(&result.token, tok_bytes.data(), 4);
      result.host_read_bytes = tok_bytes.size();
      if (sampling_contract == SamplingContract::PositionCoupledV1) {
        U32 flag = 0;
        std::memcpy(&flag, tok_bytes.data() + 4, 4);
        if (flag & 2U)
          throw std::runtime_error("nonfinite position target logits");
        if (flag & 1U) {
          auto ib = position_ids.buffer->read(0, U64(sampling.top_k) * 4);
          auto sb = position_scores.buffer->read(0, U64(sampling.top_k) * 4);
          std::vector<U32> ids(sampling.top_k);
          std::vector<float> scores(sampling.top_k);
          std::memcpy(ids.data(), ib.data(), ib.size());
          std::memcpy(scores.data(), sb.data(), sb.size());
          result.token = position_sample_candidates(
              ids, scores,
              {.seed = sampling.seed,
               .position = start_pos + count,
               .top_k = sampling.top_k,
               .temperature = sampling.temperature,
               .top_p = sampling.top_p});
          result.host_read_bytes += ib.size() + sb.size();
        }
      } else if (result.token >= v) {
        throw std::runtime_error("nonfinite or invalid legacy sampled token");
      }
    }

    if (pool)
      pool->commit(count);
    page_guard.active = false;
    next_position += count;
    last_logits_row = 0U;
    has_last_logits = !trunk_only;
    record_prefill_telemetry(count, start_pos, plan.used_w4a8, plan.used_sage2,
                             plan.used_minference, plan.chunk_t,
                             plan.context_envelope, /*used_ane=*/true,
                             &result.prefill_delta);
    return result;
  }

  bool cache_blend_eligible() const noexcept {
    if (!prefill_options_.cache_blend || !cache_blend_kernels_ready_ ||
        kv_format_ != KvStorageFormat::Affine4Group128 || config.head_dim != 256 ||
        config.full_attention_layers() <= 0 || page_pool == nullptr ||
        shared_primary_ != nullptr || speculative_enabled ||
        inspect_routes_enabled || bypass_lora_ != nullptr || (h % 128U) != 0U ||
        (q % 128U) != 0U) {
      return false;
    }
    if (config.linear_attention_layers() > 0 &&
        (config.linear_key_head_dim != 128 ||
         config.linear_value_head_dim != 128 || gdn_chunk_ws.buffer == nullptr ||
         !gdn_chunk_nax_enabled_)) {
      return false;
    }
    return true;
  }

  U32 blend_probe_layer_index() const noexcept {
    for (U32 i = 0; i < blocks.size(); ++i) {
      if (blocks[i].type == LayerType::full_attention) {
        return i;
      }
    }
    return 0U;
  }

  void run_blend_exec_profile(Profile &prof) {
    const std::array calls{MetalCommandInvocation{
        .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
    static_cast<void>(context.execute_batch(calls));
  }

  void emit_blend_gdn_slice(Encoder &e, const Block &block, Ref state0_ref,
                            Ref state1_ref, U32 row_off, U32 slice_len) {
    if (slice_len == 0U)
      return;
    const auto &c = config;
    const U32 kh = U32(c.linear_num_key_heads);
    const U32 hv = U32(c.linear_num_value_heads);
    const U32 kd = U32(c.linear_key_head_dim);
    const U32 vd = U32(c.linear_value_head_dim);
    const Ref qkv_s{qkv.buffer, qkv.offset + mul(row_off, mul(conv, 2)),
                    mul(slice_len, mul(conv, 2))};
    const Ref z_s{z.buffer, z.offset + mul(row_off, mul(gv, 2)),
                  mul(slice_len, mul(gv, 2))};
    const Ref a_s{a.buffer, a.offset + mul(row_off, mul(hv, 2)),
                  mul(slice_len, mul(hv, 2))};
    const Ref b_s{b.buffer, b.offset + mul(row_off, mul(hv, 2)),
                  mul(slice_len, mul(hv, 2))};
    const Ref mix_s{mixer.buffer, mixer.offset + mul(row_off, mul(gv, 2)),
                    mul(slice_len, mul(gv, 2))};
    const Ref conv_w =
        direct(block.get("gdn_conv"), U64(conv) * c.linear_conv_kernel_dim);
    const Ref a_log = direct(block.get("gdn_a_log"), hv);
    const Ref dt_bias = direct(block.get("gdn_dt_bias"), hv);
    const Ref gdn_norm = direct(block.get("gdn_norm"), vd);
    auto params =
        words({kh, hv, kd, vd, conv, U32(c.linear_conv_kernel_dim), slice_len,
               word(float(c.rms_norm_eps))});
    emit_gdn_conv_silu(e, slice_len, conv, qkv_s, conv_w, state1_ref, convolved,
                       params, /*dynamic_tokens=*/false);
    emit_gdn_recurrent_forward(
        e, slice_len, gv, kd, vd, convolved, a_s, b_s, a_log, dt_bias,
        state0_ref, recurrent, params, silicon_tuning.gdn_tb8_min_tokens,
        gdn_chunk_ws, gdn_chunk_nax_enabled_, kh, hv, /*dynamic_tokens=*/false,
        silicon_tuning.gpu_cores);
    auto &ng = e.emit("gdn_layer", "omlx_m5_gdn_norm_gate",
                      {.width = 32U, .height = U64(slice_len) * hv},
                      {.width = 32U}, {recurrent, z_s, gdn_norm, mix_s});
    e.dynamic(ng, 4, params);
  }

  void emit_blend_attn_route_b(Encoder &e, const Block &block, U32 m_rows,
                               U32 start_pos_for_params, U32 context_envelope,
                               Ref attn_pos_ref) {
    const auto &c = config;
    const Ref table_ref{&page_pool->table_buffer(), 0,
                        page_pool->table_buffer().size()};
    const Ref gate_ref = sub(query, U64(c.head_dim) * 2);
    auto params = words({U32(c.num_attention_heads),
                         U32(c.num_key_value_heads),
                         U32(c.head_dim),
                         m_rows,
                         start_pos_for_params,
                         max_context,
                         64U,
                         page_pool->physical_pages(),
                         q * 2,
                         U32(c.head_dim * 2),
                         kv,
                         kv,
                         q * 2,
                         U32(c.head_dim * 2)});
    const auto route =
        select_chunk_attention_route(m_rows, start_pos_for_params);
    if (route != ChunkAttentionRoute::ExactNax && sage2_enabled_ &&
        sage2_kv_ws_.buffer != nullptr && m_rows >= kFlashNaxPrefillMinTokens) {
      const bool use_minf =
          (route == ChunkAttentionRoute::Sage2MInference) &&
          minference_enabled_ &&
          context_envelope >= kMInferenceMinContextTokens &&
          minference_scratch.buffer != nullptr;
      emit_paged_flash_prefill_sage2(
          e, c, m_rows, context_envelope, /*is_affine4=*/true, use_minf, query,
          gate_ref, block.state0, block.state1, mixer, table_ref, attn_pos_ref,
          params, sage2_kv_ws_, sage2_kv_ws_layout_,
          use_minf ? std::optional<Ref>{minference_scratch} : std::nullopt);
      e.profile.used_sage2 = true;
      if (use_minf) {
        e.profile.used_minference = true;
        e.profile.minf_context_envelope = context_envelope;
      }
      return;
    }
    const bool affine4_nax_ok = affine4_kv_runtime_eligible(
        context, libraries, U32(c.num_attention_heads),
        U32(c.num_key_value_heads), U32(c.head_dim));
    if (affine4_nax_ok) {
      const bool short_tail_split =
          split_attention_enabled && split_max.buffer != nullptr &&
          split_sum.buffer != nullptr && split_output.buffer != nullptr &&
          gqa_flash_prefill_tail_needs_split_kv(
              m_rows, U32(c.num_attention_heads), U32(c.num_key_value_heads),
              context_envelope, silicon_tuning.gpu_cores) &&
          affine4_short_tail_split_tokens(
              m_rows, U32(c.num_attention_heads), U32(c.num_key_value_heads),
              context_envelope, split_max.bytes, split_sum.bytes,
              split_output.bytes) > 0U;
      if (short_tail_split) {
        emit_paged_attention_affine4_g128(
            e, c, m_rows, /*split=*/true, context_envelope, max_context,
            page_pool->physical_pages(), query, gate_ref, block.state0,
            block.state1, mixer, table_ref, attn_pos_ref, split_max, split_sum,
            split_output, std::move(params), std::nullopt, std::nullopt);
      } else if (flash_nax_prefill_ && m_rows >= kFlashNaxPrefillMinTokens &&
                 flash_nax_prefill_runtime_eligible(context, libraries,
                                                    U32(c.head_dim))) {
        emit_paged_flash_prefill_affine4_g128(
            e, c, m_rows, query, gate_ref, block.state0, block.state1, mixer,
            table_ref, attn_pos_ref, std::move(params));
      } else {
        emit_paged_attention_affine4_g128(
            e, c, m_rows, /*split=*/false, context_envelope, max_context,
            page_pool->physical_pages(), query, gate_ref, block.state0,
            block.state1, mixer, table_ref, attn_pos_ref, split_max, split_sum,
            split_output, std::move(params), std::nullopt, std::nullopt);
      }
      return;
    }
    auto &att = e.emit("qwen_forward",
                       "omlx_m5_forward_paged_causal_attention_affine4_exact",
                       {.width = 32U, .height = U64(m_rows) * c.num_attention_heads},
                       {.width = 32U},
                       {query, gate_ref, block.state0, block.state1, mixer,
                        table_ref});
    e.dynamic(att, 6, std::move(params));
  }

  void capture_and_insert_blend_chunk(std::span<const U32> chunk_tokens,
                                      U32 start_pos, U32 w_bnd,
                                      detail::ChunkBlendCache &blend_cache,
                                      U64 model_exec_hash,
                                      QwenPrefillTelemetry &accum_delta) {
    const U32 lm = U32(chunk_tokens.size());
    const U32 probe_idx = blend_probe_layer_index();
    const U32 num_attn_layers = U32(config.full_attention_layers());
    const U32 num_gdn_layers = U32(config.linear_attention_layers());
    const U32 kvh = U32(config.num_key_value_heads);
    const U32 qh = U32(config.num_attention_heads);
    const U32 rot_dim =
        U32(std::llround(config.head_dim * config.partial_rotary_factor));
    const U32 kh = U32(config.linear_num_key_heads);
    const U32 hv = U32(config.linear_num_value_heads);
    const U32 eff_bnd = std::min<U32>(w_bnd, lm);

    detail::BlendChunkGeometry geom;
    geom.num_full_attn_layers = num_attn_layers;
    geom.num_gdn_layers = num_gdn_layers;
    geom.q_heads = qh;
    geom.kv_heads = kvh;
    geom.head_dim = U32(config.head_dim);
    geom.rotary_dim = rot_dim;
    geom.hidden_size = h;
    geom.gdn_key_heads = kh;
    geom.gdn_value_heads = hv;
    geom.gdn_key_dim = U32(config.linear_key_head_dim);
    geom.gdn_value_dim = U32(config.linear_value_head_dim);
    geom.gdn_conv_width = conv;
    geom.gdn_conv_kernel = U32(config.linear_conv_kernel_dim);
    geom.token_length = lm;
    geom.interior_start = eff_bnd;
    geom.probe_layer_index = probe_idx;
    geom.gdn_op_format = detail::BlendGdnOperatorFormat::BFloat16;

    auto entry = std::make_shared<detail::BlendChunkEntry>();
    entry->key = detail::make_blend_chunk_key(model_exec_hash, chunk_tokens);
    entry->tokens.assign(chunk_tokens.begin(), chunk_tokens.end());
    entry->geometry = geom;
    entry->payload = detail::allocate_blend_chunk_payload(context, geom);

    // Scratch FP32 state buffers for extracting M_C and B_C without disturbing live state0.
    const U64 gdn_op_f32_bytes =
        num_gdn_layers ? mul(mul(U64(hv), 128ULL * 128ULL), 4ULL) : 256ULL;
    MetalBuffer m_tmp_buf = context.make_private_buffer(gdn_op_f32_bytes);
    MetalBuffer b_tmp_buf = context.make_private_buffer(gdn_op_f32_bytes);
    MetalBuffer zero_state_buf = context.make_buffer(gdn_op_f32_bytes);
    zero_state_buf.zero();
    const Ref m_tmp_ref{&m_tmp_buf, 0, gdn_op_f32_bytes};
    const Ref b_tmp_ref{&b_tmp_buf, 0, gdn_op_f32_bytes};
    const Ref zero_state_ref{&zero_state_buf, 0, gdn_op_f32_bytes};

    // Upload tokens and positions.
    std::fill_n(token_staging.begin(), lm, 0U);
    std::copy(chunk_tokens.begin(), chunk_tokens.end(), token_staging.begin());
    const_cast<MetalBuffer *>(token_ids.buffer)
        ->write(std::as_bytes(std::span(token_staging).first(lm)));
    std::vector<U32> positions(lm);
    for (U32 i = 0; i < lm; ++i)
      positions[i] = start_pos + i;
    const_cast<MetalBuffer *>(attention_positions.buffer)
        ->write(std::as_bytes(std::span(positions)));

    const auto page_copies = page_pool->begin_append(lm);
    struct PageGuard {
      KvPagePool *pool;
      bool active;
      ~PageGuard() {
        if (active)
          try {
            pool->abort();
          } catch (...) {
          }
      }
    } page_guard{page_pool.get(), true};
    execute_cow_copies(page_copies);

    const U32 end_pos = start_pos + lm;
    const auto cold_route = select_chunk_attention_route(lm, start_pos);
    const U32 ctx_env =
        (cold_route != ChunkAttentionRoute::ExactNax)
            ? select_long_ctx_envelope(end_pos)
            : end_pos;
    const bool use_w4a4 = use_w4a4_for_t(lm, false);
    const bool use_w4a8 = !use_w4a4 && use_w4a8_for_t(lm, false);
    const U64 seq_id = blend_plan_seq_.fetch_add(1ULL, std::memory_order_relaxed);
    Encoder e(context, libraries, lm,
              "qwen.blend.cold." + std::to_string(seq_id) + plan_key_suffix());
    if (use_w4a4) {
      e.profile.used_w4a4 = true;
    }
    if (use_w4a8) {
      e.profile.used_w4a8 = true;
    }

    e.begin_stage("embedding");
    embedding_lookup(e, embedding, token_ids, hidden, zero, v, h);

    const U64 attn_layer_kv_bytes =
        U64(lm) * U64(kvh) * geom.attn_vec_bytes();
    const U64 attn_layer_rot_bytes =
        U64(lm) * U64(kvh) * U64(rot_dim) * 2ULL;
    const U64 gdn_layer_tail_bytes =
        U64(config.linear_conv_kernel_dim - 1) * U64(conv) * 2ULL;
    const U64 gdn_layer_op_bytes = U64(hv) * 128ULL * 128ULL * 2ULL;
    const U64 gdn_layer_sc_bytes = U64(hv) * 128ULL * 2ULL;
    const U64 gdn_layer_cum_bytes = U64(lm) * U64(hv) * 2ULL;
    const U32 l_int = (lm > eff_bnd) ? (lm - eff_bnd) : 0U;

    U32 attn_idx = 0U;
    U32 gdn_idx = 0U;
    for (U32 li = 0; li < blocks.size(); ++li) {
      const auto &block = blocks[li];
      e.begin_stage("blend_cold.L" + std::to_string(li));
      if (li == probe_idx) {
        // Capture pre-Layer-3 residual hidden H3_res in int8_g128.
        detail::BlendInt8QuantParams qp{.tokens = lm, .width = h};
        auto &qh3 = e.emit(
            "cache_blend", "omlx_m5_blend_quantize_int8_g128_bf16",
            {.width = 32U, .height = U64(lm) * (h / 128U)}, {.width = 32U},
            {hidden, Ref{&entry->payload.h3_res_i8, 0, geom.h3_res_i8_bytes()},
             Ref{&entry->payload.h3_res_scales, 0, geom.h3_res_scale_bytes()}});
        e.dynamic(qh3, 3, pod_bytes(qp));
      }

      if (use_w4a4) {
        emit_rmsnorm_a4_bm16_bk128(e, hidden, direct(block.get("input_norm"), h),
                                   norm, w4a4_codes, w4a4_scales, h,
                                   float(config.rms_norm_eps),
                                   normalization_weight_offset,
                                   /*write_bf16=*/true);
        e.set_a4_quant(make_a4_key(e, norm, h));
      } else if (use_w4a8) {
        emit_rmsnorm_a8_g64(e, hidden, direct(block.get("input_norm"), h), norm,
                            a8_qa, a8_sa, a8_ra, h, float(config.rms_norm_eps),
                            normalization_weight_offset, /*write_bf16=*/true);
        e.set_a8_quant(make_a8_key(e, norm, h));
      } else {
        normalize(e, block.get("input_norm"), hidden, norm);
      }

      if (block.type == LayerType::full_attention) {
        const std::array<MaterializedWeightView, 3> parts{
            block.get("attention_q"), block.get("attention_k"),
            block.get("attention_v")};
        const std::array<U32, 3> part_ns{2 * q, kv, kv};
        if (use_w4a4 && !w4a4_keep_qk_a8_ &&
            try_concat_w4a4_nax(e, block.concat_qkv_id, parts, part_ns, h, norm,
                                concat_out)) {
          emit_split3(e, concat_out, query, key, value, 2 * q, kv, kv);
        } else if (use_w4a4) {
          if (w4a4_keep_qk_a8_) {
            if (!try_linear_w4a8_base(e, parts[0], norm, query, 2 * q, h) &&
                !try_linear_w4a4_base(e, parts[0], norm, query, 2 * q, h))
              linear_base(e, parts[0], norm, query, 2 * q, h);
            if (!try_linear_w4a8_base(e, parts[1], norm, key, kv, h) &&
                !try_linear_w4a4_base(e, parts[1], norm, key, kv, h))
              linear_base(e, parts[1], norm, key, kv, h);
          } else {
            if (!try_linear_w4a4_base(e, parts[0], norm, query, 2 * q, h))
              linear_base(e, parts[0], norm, query, 2 * q, h);
            if (!try_linear_w4a4_base(e, parts[1], norm, key, kv, h))
              linear_base(e, parts[1], norm, key, kv, h);
          }
          if (!try_linear_w4a4_base(e, parts[2], norm, value, kv, h))
            linear_base(e, parts[2], norm, value, kv, h);
        } else if (use_w4a8 &&
            try_concat_w4a8_nax(e, block.concat_qkv_id, parts, part_ns, h, norm,
                                concat_out)) {
          emit_split3(e, concat_out, query, key, value, 2 * q, kv, kv);
        } else if (use_w4a8) {
          if (!try_linear_w4a8_base(e, parts[0], norm, query, 2 * q, h))
            linear_base(e, parts[0], norm, query, 2 * q, h);
          if (!try_linear_w4a8_base(e, parts[1], norm, key, kv, h))
            linear_base(e, parts[1], norm, key, kv, h);
          if (!try_linear_w4a8_base(e, parts[2], norm, value, kv, h))
            linear_base(e, parts[2], norm, value, kv, h);
        } else if (try_concat_nax(e, block.concat_qkv_id, parts, part_ns, h,
                                  false, norm, concat_out)) {
          emit_split3(e, concat_out, query, key, value, 2 * q, kv, kv);
        } else {
          linear_base(e, parts[0], norm, query, 2 * q, h);
          linear_base(e, parts[1], norm, key, kv, h);
          linear_base(e, parts[2], norm, value, kv, h);
        }

        // Capture pre-RoPE K (post k_norm) + V BEFORE emit_attention_state_step mutates key!
        detail::BlendExtractPreRopeKvParams ep{};
        ep.key_value_heads = kvh;
        ep.head_dim = U32(config.head_dim);
        ep.rotary_dim = rot_dim;
        ep.tokens = lm;
        ep.key_token_stride = kv;
        ep.value_token_stride = kv;
        ep.apply_k_norm = 1U;
        ep.write_rotary_bf16 = 1U;
        ep.epsilon = float(config.rms_norm_eps);
        ep.weight_offset = normalization_weight_offset;
        const Ref dst_k_pre{&entry->payload.attn_k_pre,
                            U64(attn_idx) * attn_layer_kv_bytes,
                            attn_layer_kv_bytes};
        const Ref dst_k_rot{&entry->payload.attn_k_rotary_bf16,
                            U64(attn_idx) * attn_layer_rot_bytes,
                            attn_layer_rot_bytes};
        const Ref dst_v{&entry->payload.attn_v,
                        U64(attn_idx) * attn_layer_kv_bytes,
                        attn_layer_kv_bytes};
        auto &ext = e.emit(
            "cache_blend", "omlx_m5_blend_extract_kv_prerope_affine4_g128",
            {.width = 32U, .height = U64(lm) * kvh}, {.width = 32U},
            {key, value, direct(block.get("k_norm"), config.head_dim),
             dst_k_pre, dst_k_rot, dst_v});
        e.dynamic(ext, 6, pod_bytes(ep));

        // Run live attention (populates KvPagePool & produces mixer).
        emit_attention_state_step(
            e, block, block, *page_pool, max_context, kv_format_, query, key,
            value, mixer, attention_positions, lm, /*split=*/false, ctx_env,
            /*dynamic_tokens=*/true, std::nullopt, nullptr, cold_route);

        if (li == probe_idx) {
          detail::BlendInt8QuantParams qm{.tokens = lm, .width = q};
          auto &qmix = e.emit(
              "cache_blend", "omlx_m5_blend_quantize_int8_g128_bf16",
              {.width = 32U, .height = U64(lm) * (q / 128U)}, {.width = 32U},
              {mixer,
               Ref{&entry->payload.mix3_cached_i8, 0,
                   geom.mix3_cached_i8_bytes()},
               Ref{&entry->payload.mix3_cached_scales, 0,
                   geom.mix3_cached_scale_bytes()}});
          e.dynamic(qmix, 3, pod_bytes(qm));
        }
        if (!(use_w4a4 && try_linear_w4a4_base(e, block.get("attention_o"),
                                               mixer, delta, h, q)) &&
            !(use_w4a8 && try_linear_w4a8_base(e, block.get("attention_o"),
                                               mixer, delta, h, q))) {
          linear(e, block.get("attention_o"), mixer, delta, h, q);
        }
        ++attn_idx;
      } else {
        const std::array<MaterializedWeightView, 2> qkvz_parts{
            block.get("gdn_qkv"), block.get("gdn_z")};
        const std::array<U32, 2> qkvz_ns{conv, gv};
        const std::array<MaterializedWeightView, 2> ab_parts{
            block.get("gdn_a"), block.get("gdn_b")};
        const std::array<U32, 2> ab_ns{hv, hv};
        if (use_w4a4 &&
            try_concat_w4a4_nax(e, block.concat_gdn_qkvz_id, qkvz_parts,
                                qkvz_ns, h, norm, concat_out)) {
          emit_split2(e, concat_out, qkv, z, conv, gv);
        } else if (use_w4a4) {
          if (!try_linear_w4a4_base(e, qkvz_parts[0], norm, qkv, conv, h))
            linear_base(e, qkvz_parts[0], norm, qkv, conv, h);
          if (!try_linear_w4a4_base(e, qkvz_parts[1], norm, z, gv, h))
            linear_base(e, qkvz_parts[1], norm, z, gv, h);
        } else if (use_w4a8 &&
            try_concat_w4a8_nax(e, block.concat_gdn_qkvz_id, qkvz_parts,
                                qkvz_ns, h, norm, concat_out)) {
          emit_split2(e, concat_out, qkv, z, conv, gv);
        } else if (use_w4a8) {
          if (!try_linear_w4a8_base(e, qkvz_parts[0], norm, qkv, conv, h))
            linear_base(e, qkvz_parts[0], norm, qkv, conv, h);
          if (!try_linear_w4a8_base(e, qkvz_parts[1], norm, z, gv, h))
            linear_base(e, qkvz_parts[1], norm, z, gv, h);
        } else if (try_concat_nax(e, block.concat_gdn_qkvz_id, qkvz_parts,
                                  qkvz_ns, h, false, norm, concat_out)) {
          emit_split2(e, concat_out, qkv, z, conv, gv);
        } else {
          linear_base(e, qkvz_parts[0], norm, qkv, conv, h);
          linear_base(e, qkvz_parts[1], norm, z, gv, h);
        }
        if (use_w4a4 &&
            try_concat_w4a4_nax(e, block.concat_gdn_ab_id, ab_parts, ab_ns, h,
                                norm, concat_out)) {
          emit_split2(e, concat_out, a, b, hv, hv);
        } else if (use_w4a4) {
          if (!try_linear_w4a4_base(e, ab_parts[0], norm, a, hv, h))
            linear_base(e, ab_parts[0], norm, a, hv, h);
          if (!try_linear_w4a4_base(e, ab_parts[1], norm, b, hv, h))
            linear_base(e, ab_parts[1], norm, b, hv, h);
        } else if (use_w4a8 &&
            try_concat_w4a8_nax(e, block.concat_gdn_ab_id, ab_parts, ab_ns, h,
                                norm, concat_out)) {
          emit_split2(e, concat_out, a, b, hv, hv);
        } else if (use_w4a8) {
          if (!try_linear_w4a8_base(e, ab_parts[0], norm, a, hv, h))
            linear_base(e, ab_parts[0], norm, a, hv, h);
          if (!try_linear_w4a8_base(e, ab_parts[1], norm, b, hv, h))
            linear_base(e, ab_parts[1], norm, b, hv, h);
        } else if (try_concat_nax(e, block.concat_gdn_ab_id, ab_parts, ab_ns, h,
                                  false, norm, concat_out)) {
          emit_split2(e, concat_out, a, b, hv, hv);
        } else {
          linear_base(e, ab_parts[0], norm, a, hv, h);
          linear_base(e, ab_parts[1], norm, b, hv, h);
        }

        // Capture tail_qkv (final conv_kernel-1 rows of qkv).
        const U32 tail_rows = U32(config.linear_conv_kernel_dim - 1);
        const U32 tail_start_row = (lm >= tail_rows) ? (lm - tail_rows) : 0U;
        const Ref qkv_tail_src{qkv.buffer,
                               qkv.offset + mul(tail_start_row, mul(conv, 2)),
                               gdn_layer_tail_bytes};
        const Ref tail_dst{&entry->payload.gdn_tail_qkv,
                           U64(gdn_idx) * gdn_layer_tail_bytes,
                           gdn_layer_tail_bytes};
        copy(e, qkv_tail_src, tail_dst, gdn_layer_tail_bytes);

        // Capture cumulative log-decay G_t[lm, Hv].
        const Ref a_log = direct(block.get("gdn_a_log"), hv);
        const Ref dt_bias = direct(block.get("gdn_dt_bias"), hv);
        const Ref cum_dst{&entry->payload.gdn_cum_decay_f16,
                          U64(gdn_idx) * gdn_layer_cum_bytes,
                          gdn_layer_cum_bytes};
        detail::BlendGdnCumDecayParams cdp{.value_heads = hv, .tokens = lm};
        auto &cd = e.emit(
            "cache_blend", "omlx_m5_gdn_compute_cum_decay_f16",
            {.width = std::max<U64>(32ULL, ((U64(hv) + 31ULL) / 32ULL) * 32ULL)},
            {.width = 32U}, {a, a_log, dt_bias, cum_dst});
        e.dynamic(cd, 4, pod_bytes(cdp));

        // Live GDN conv + recurrence + norm_gate -> live state0, state1, mixer.
        emit_blend_gdn_slice(e, block, block.state0, block.state1, 0U, lm);

        // Extract exact interior [eff_bnd, lm) transition (M_C, B_C) via NAX!
        if (l_int > 0U) {
          const Ref conv_int{convolved.buffer,
                             convolved.offset + mul(eff_bnd, mul(conv, 2)),
                             mul(l_int, mul(conv, 2))};
          const Ref a_int{a.buffer, a.offset + mul(eff_bnd, mul(hv, 2)),
                          mul(l_int, mul(hv, 2))};
          const Ref b_int{b.buffer, b.offset + mul(eff_bnd, mul(hv, 2)),
                          mul(l_int, mul(hv, 2))};
          const auto ws = gdn_chunk_nax_workspace_layout(l_int, kh, hv);
          const Ref qk_ref{gdn_chunk_ws.buffer, gdn_chunk_ws.offset,
                           ws.qk_bytes};
          const Ref ctrl_ref{gdn_chunk_ws.buffer,
                             gdn_chunk_ws.offset + ws.qk_bytes, ws.ctrl_bytes};
          auto int_params = words({kh, hv, 128U, 128U, conv,
                                   U32(config.linear_conv_kernel_dim), l_int,
                                   word(float(config.rms_norm_eps))});
          auto &prep = e.emit(
              "gdn_chunk_nax", "omlx_m5_gdn_chunk_prepare_c64",
              {.width = U64(kh) * 128U, .height = ws.max_chunks},
              {.width = 128U},
              {conv_int, a_int, b_int, a_log, dt_bias, qk_ref, ctrl_ref});
          e.dynamic(prep, 7, int_params);

          // Extract M_C (S_init = I_128, V = 0) and pack into payload.gdn_m_*.
          detail::BlendGdnExtractMParams emp{
              .key_heads = kh, .value_heads = hv, .key_dim = 128U,
              .value_dim = 128U, .tokens = l_int};
          auto &em = e.emit("cache_blend",
                            "omlx_m5_gdn_extract_transition_m_c64",
                            {.width = 128U, .height = hv, .depth = 2U},
                            {.width = 128U}, {qk_ref, ctrl_ref, m_tmp_ref});
          e.dynamic(em, 3, pod_bytes(emp));

          detail::BlendGdnPackOpParams pop{
              .value_heads = hv,
              .dim = 128U,
              .op_format = static_cast<U32>(geom.gdn_op_format)};
          const Ref m_dst{&entry->payload.gdn_m_data,
                          U64(gdn_idx) * gdn_layer_op_bytes,
                          gdn_layer_op_bytes};
          const Ref m_sc{&entry->payload.gdn_m_row_scales,
                         U64(gdn_idx) * gdn_layer_sc_bytes, gdn_layer_sc_bytes};
          auto &pm = e.emit("cache_blend", "omlx_m5_gdn_pack_operator",
                            {.width = 32U, .height = U64(hv) * 128U},
                            {.width = 32U}, {m_tmp_ref, m_dst, m_sc});
          e.dynamic(pm, 3, pod_bytes(pop));

          // Extract B_C (S_init = 0, V = V_int) via recurrence into zeroed b_tmp_ref.
          copy(e, zero_state_ref, b_tmp_ref, gdn_op_f32_bytes);
          auto &rec_b = e.emit(
              "gdn_chunk_nax", "omlx_m5_gdn_chunk_recurrence_c64",
              {.width = U64(hv) * 128U, .height = 4U}, {.width = 128U},
              {conv_int, qk_ref, ctrl_ref, b_tmp_ref, recurrent});
          e.dynamic(rec_b, 5, int_params);
          const Ref b_dst{&entry->payload.gdn_b_data,
                          U64(gdn_idx) * gdn_layer_op_bytes,
                          gdn_layer_op_bytes};
          const Ref b_sc{&entry->payload.gdn_b_row_scales,
                         U64(gdn_idx) * gdn_layer_sc_bytes, gdn_layer_sc_bytes};
          auto &pb = e.emit("cache_blend", "omlx_m5_gdn_pack_operator",
                            {.width = 32U, .height = U64(hv) * 128U},
                            {.width = 32U}, {b_tmp_ref, b_dst, b_sc});
          e.dynamic(pb, 3, pod_bytes(pop));
        }

        if (!(use_w4a4 && try_linear_w4a4_base(e, block.get("gdn_out"), mixer,
                                               delta, h, gv)) &&
            !(use_w4a8 && try_linear_w4a8_base(e, block.get("gdn_out"), mixer,
                                               delta, h, gv))) {
          linear(e, block.get("gdn_out"), mixer, delta, h, gv);
        }
        ++gdn_idx;
      }

      if (use_w4a4) {
        emit_add_rmsnorm_a4_bm16_bk128(
            e, hidden, delta, direct(block.get("post_attention_norm"), h), norm,
            w4a4_codes, w4a4_scales, h, float(config.rms_norm_eps),
            normalization_weight_offset,
            /*write_bf16=*/(config.num_experts > 0));
        e.set_a4_quant(make_a4_key(e, norm, h));
      } else if (use_w4a8) {
        emit_add_rmsnorm_a8_g64(
            e, hidden, delta, direct(block.get("post_attention_norm"), h), norm,
            a8_qa, a8_sa, a8_ra, h, float(config.rms_norm_eps),
            normalization_weight_offset,
            /*write_bf16=*/(config.num_experts > 0));
        e.set_a8_quant(make_a8_key(e, norm, h));
      } else {
        add_normalize(e, block.get("post_attention_norm"), hidden, delta, norm);
      }
      mlp(e, block, li, false, {}, nullptr, std::nullopt, false, use_w4a8,
          /*verify=*/false, use_w4a4);
      add(e);
    }

    auto prof = e.finish();
    for (const auto &patch : prof.patches) {
      const U32 val = patch.position ? start_pos : lm * patch.multiplier;
      std::memcpy(prof.parameter_data.data() + patch.offset + patch.word * 4,
                  &val, 4);
    }
    prof.parameters.write("step", prof.parameter_data);
    run_blend_exec_profile(prof);

    page_pool->commit(lm);
    page_guard.active = false;
    next_position += lm;
    has_last_logits = false;

    QwenPrefillTelemetry seg_delta;
    record_prefill_telemetry(lm, start_pos, prof.used_w4a8, prof.used_sage2,
                             prof.used_minference, lm, prof.minf_context_envelope,
                             /*used_ane=*/false, &seg_delta, std::nullopt,
                             prof.used_w4a4);
    accum_delta.prefill_w4a8_tokens += seg_delta.prefill_w4a8_tokens;
    accum_delta.prefill_w4a4_tokens += seg_delta.prefill_w4a4_tokens;
    accum_delta.prefill_sage2_tokens += seg_delta.prefill_sage2_tokens;
    accum_delta.prefill_minference_blocks_total +=
        seg_delta.prefill_minference_blocks_total;
    accum_delta.prefill_minference_blocks_skipped +=
        seg_delta.prefill_minference_blocks_skipped;
    accum_delta.prefill_ane_fallback_tokens +=
        seg_delta.prefill_ane_fallback_tokens;

    (void)blend_cache.insert(std::move(entry));
  }

  QwenForwardResult run_blend_prefill_impl(
      QwenForward &owner, std::span<const U32> prompt, U32 prefix_tokens,
      detail::ChunkBlendCache &blend_cache, U64 model_exec_hash,
      const GpuSamplingOptions &sampling, bool inspect_tensors,
      const detail::BlendSegmentConfig *seg_cfg) {
    require_no_async_in_flight("run_blend_prefill");
    execution_started = true;
    if (pending_verify)
      throw std::logic_error("Qwen verification must commit or abort first");
    if (prefix_tokens != next_position || prompt.size() <= prefix_tokens ||
        prompt.size() > max_context)
      throw std::invalid_argument("Qwen blend prefill frontier/length invalid");
    if (std::any_of(prompt.begin(), prompt.end(),
                    [&](U32 id) { return id >= v; }))
      throw std::invalid_argument("Qwen blend prefill token outside vocabulary");

    const U32 total_tokens = U32(prompt.size());
    const U32 suffix_len = total_tokens - prefix_tokens;

    auto run_standard_chunked =
        [&](U32 start_pos, U32 end_pos, bool sample_last,
            QwenPrefillTelemetry &accum) -> QwenForwardResult {
      QwenForwardResult last_res{};
      const U32 chunk_cap =
          (ane_active_ && ane_chunk_tokens_ > 0U)
              ? std::min<U32>(workspace_rows_, ane_chunk_tokens_)
              : workspace_rows_;
      U32 cur = start_pos;
      while (cur < end_pos) {
        const U32 rem = end_pos - cur;
        const U32 chunk = std::min<U32>(rem, chunk_cap);
        const bool is_final = (cur + chunk == end_pos);
        if (is_final && sample_last) {
          last_res = owner.run(prompt.subspan(cur, chunk), sampling,
                               inspect_tensors, /*use_icb=*/true);
          if (inspect_tensors && last_res.logits.size() > v) {
            last_res.logits.assign(
                last_res.logits.end() - static_cast<std::ptrdiff_t>(v),
                last_res.logits.end());
          }
          if (inspect_tensors && last_res.hidden.size() > h) {
            last_res.hidden.assign(
                last_res.hidden.end() - static_cast<std::ptrdiff_t>(h),
                last_res.hidden.end());
          }
        } else {
          last_res = owner.run_trunk(prompt.subspan(cur, chunk), true);
        }
        accum.prefill_w4a8_tokens += last_res.prefill_delta.prefill_w4a8_tokens;
        accum.prefill_w4a4_tokens += last_res.prefill_delta.prefill_w4a4_tokens;
        accum.prefill_sage2_tokens +=
            last_res.prefill_delta.prefill_sage2_tokens;
        accum.prefill_minference_blocks_total +=
            last_res.prefill_delta.prefill_minference_blocks_total;
        accum.prefill_minference_blocks_skipped +=
            last_res.prefill_delta.prefill_minference_blocks_skipped;
        accum.prefill_ane_offload_tokens +=
            last_res.prefill_delta.prefill_ane_offload_tokens;
        accum.prefill_ane_fallback_tokens +=
            last_res.prefill_delta.prefill_ane_fallback_tokens;
        cur += chunk;
      }
      return last_res;
    };

    if (!cache_blend_eligible() || !blend_cache.enabled()) {
      QwenPrefillTelemetry accum;
      accum.prefill_ane_active = ane_active_;
      accum.prefill_ane_mode_effective =
          std::string(prefill_ane_mode_to_string(ane_mode_effective_));
      accum.prefill_ane_inactive_reason = ane_inactive_reason_;
      accum.prefill_ane_split_alpha = ane_split_alpha_;
      auto res = run_standard_chunked(prefix_tokens, total_tokens, true, accum);
      res.tokens = suffix_len;
      res.prefill_delta = accum;
      return res;
    }

    detail::BlendSegmentConfig eff_cfg =
        seg_cfg ? *seg_cfg : detail::BlendSegmentConfig{};
    // Clamp chunk upper bound to workspace_rows_ so every cacheable segment can
    // be captured and replayed across multi-window 16K-256K prompts without drop.
    eff_cfg.l_max = std::min<U32>(eff_cfg.l_max, workspace_rows_);
    if (eff_cfg.l_min > eff_cfg.l_max) {
      eff_cfg.l_min = eff_cfg.l_max;
    }
    if (eff_cfg.gear_min_len > eff_cfg.l_max) {
      eff_cfg.gear_min_len = eff_cfg.l_max;
    }
    const auto plan =
        blend_cache.plan_prompt(model_exec_hash, prompt, prefix_tokens, eff_cfg);

    const U32 probe_idx = blend_probe_layer_index();
    const U32 kvh = U32(config.num_key_value_heads);
    const U32 qh = U32(config.num_attention_heads);
    const U32 rot_dim =
        U32(std::llround(config.head_dim * config.partial_rotary_factor));
    const U32 kh = U32(config.linear_num_key_heads);
    const U32 hv = U32(config.linear_num_value_heads);
    const U64 gdn_tail_bytes =
        U64(config.linear_conv_kernel_dim - 1) * U64(conv) * 2ULL;

    const auto emit_gdn_jump = [&](Encoder &e, const Block &blk, U32 gdn_idx,
                                   const detail::BlendChunkEntry &he) {
      const auto &g = he.geometry;
      const U64 elem_b =
          (g.gdn_op_format == detail::BlendGdnOperatorFormat::Fp8E4M3RowScale)
              ? 1ULL
              : 2ULL;
      const U64 op_bytes = U64(hv) * 128ULL * 128ULL * elem_b;
      const U64 sc_bytes = U64(hv) * 128ULL * 2ULL;
      const U64 cum_bytes = U64(g.token_length) * U64(hv) * 2ULL;
      detail::BlendGdnStateComposeParams cp{};
      cp.key_heads = kh;
      cp.value_heads = hv;
      cp.dim = 128U;
      cp.op_format = static_cast<U32>(g.gdn_op_format);
      cp.chunk_tokens = g.token_length;
      cp.num_hkvd_perturb = 0U;
      cp.apply_bias = 1U;
      auto &sc = e.emit(
          "cache_blend", "omlx_m5_gdn_state_compose_nax",
          {.width = 128U, .height = hv, .depth = 4U}, {.width = 128U},
          {blk.state0,
           Ref{&he.payload.gdn_m_data, U64(gdn_idx) * op_bytes, op_bytes},
           Ref{&he.payload.gdn_m_row_scales, U64(gdn_idx) * sc_bytes, sc_bytes},
           Ref{&he.payload.gdn_b_data, U64(gdn_idx) * op_bytes, op_bytes},
           Ref{&he.payload.gdn_b_row_scales, U64(gdn_idx) * sc_bytes, sc_bytes},
           Ref{&he.payload.gdn_cum_decay_f16, U64(gdn_idx) * cum_bytes,
               cum_bytes},
           zero, zero, zero, zero, zero, zero, zero, blk.state0});
      e.dynamic(sc, 14, pod_bytes(cp));
      copy(e,
           Ref{&he.payload.gdn_tail_qkv, U64(gdn_idx) * gdn_tail_bytes,
               gdn_tail_bytes},
           blk.state1, gdn_tail_bytes);
    };

    auto execute_warm_blend_window =
        [&](std::size_t seg_begin, std::size_t seg_end, bool is_final_window,
            QwenPrefillTelemetry &accum) -> QwenForwardResult {
      const U32 win_prefix = plan.segments[seg_begin].segment.start;
      U32 win_len = 0U;
      U32 win_hit_tokens = 0U;
      for (std::size_t si = seg_begin; si < seg_end; ++si) {
        win_len += plan.segments[si].segment.length;
        if (plan.segments[si].is_cache_hit()) {
          win_hit_tokens += plan.segments[si].segment.length;
        }
      }
      const U32 win_end = win_prefix + win_len;
      const U32 win_rel_base = win_prefix - prefix_tokens;
      const U32 ctx_env =
          sage2_enabled_ ? select_long_ctx_envelope(win_end) : win_end;

      const auto page_copies = page_pool->begin_append(win_len);
      struct PageGuard {
        KvPagePool *pool;
        bool active;
        ~PageGuard() {
          if (active)
            try {
              pool->abort();
            } catch (...) {
            }
        }
      } page_guard{page_pool.get(), true};
      execute_cow_copies(page_copies);

      // Scratch buffers for stitched hidden/mixer and HKVD selection in this window.
      MetalBuffer full_hidden_buf =
          context.make_private_buffer(mul(win_len, mul(h, 2ULL)));
      MetalBuffer full_mixer_buf =
          context.make_private_buffer(mul(win_len, mul(q, 2ULL)));
      MetalBuffer mix3_cached_i8_buf =
          context.make_private_buffer(mul(win_len, U64(q)));
      MetalBuffer mix3_cached_sc_buf =
          context.make_private_buffer(mul(win_len, mul(q / 128U, 2ULL)));
      MetalBuffer flags_buf = context.make_buffer(mul(win_len, 4ULL));
      MetalBuffer prompt_pos_buf = context.make_buffer(mul(win_len, 4ULL));
      MetalBuffer l2_scores_buf = context.make_buffer(mul(win_len, 4ULL));
      MetalBuffer active_idx_buf = context.make_buffer(mul(win_len, 4ULL));
      MetalBuffer active_pos_buf = context.make_buffer(mul(win_len, 4ULL));
      MetalBuffer active_cnt_buf = context.make_buffer(16ULL);
      const std::span<const U32> win_flags(
          plan.suffix_token_flags.data() + win_rel_base, win_len);
      const std::span<const U32> win_pos(
          plan.suffix_prompt_positions.data() + win_rel_base, win_len);
      flags_buf.write(std::as_bytes(win_flags));
      prompt_pos_buf.write(std::as_bytes(win_pos));

      std::vector<U32> mand_idx;
      std::vector<U32> mand_pos;
      mand_idx.reserve(win_len);
      mand_pos.reserve(win_len);
      for (U32 i = 0; i < win_len; ++i) {
        if (win_flags[i] == detail::kBlendTokenMandatory) {
          mand_idx.push_back(i);
          mand_pos.push_back(win_pos[i]);
        }
      }
      const U32 m_mand = U32(mand_idx.size());
      MetalBuffer mand_idx_buf =
          context.make_buffer(std::max<U64>(16ULL, mul(m_mand, 4ULL)));
      MetalBuffer mand_pos_buf =
          context.make_buffer(std::max<U64>(16ULL, mul(m_mand, 4ULL)));
      if (m_mand > 0U) {
        mand_idx_buf.write(std::as_bytes(std::span(mand_idx)));
        mand_pos_buf.write(std::as_bytes(std::span(mand_pos)));
      }

      const Ref full_hidden_ref{&full_hidden_buf, 0, full_hidden_buf.size()};
      const Ref full_mixer_ref{&full_mixer_buf, 0, full_mixer_buf.size()};
      const Ref mix3_i8_ref{&mix3_cached_i8_buf, 0, mix3_cached_i8_buf.size()};
      const Ref mix3_sc_ref{&mix3_cached_sc_buf, 0, mix3_cached_sc_buf.size()};
      const Ref flags_ref{&flags_buf, 0, flags_buf.size()};
      const Ref prompt_pos_ref{&prompt_pos_buf, 0, prompt_pos_buf.size()};
      const Ref l2_ref{&l2_scores_buf, 0, l2_scores_buf.size()};
      const Ref act_idx_ref{&active_idx_buf, 0, active_idx_buf.size()};
      const Ref act_pos_ref{&active_pos_buf, 0, active_pos_buf.size()};
      const Ref act_cnt_ref{&active_cnt_buf, 0, active_cnt_buf.size()};
      const Ref mand_idx_ref{&mand_idx_buf, 0, mand_idx_buf.size()};
      const Ref mand_pos_ref{&mand_pos_buf, 0, mand_pos_buf.size()};
      const Ref table_ref{&page_pool->table_buffer(), 0,
                          page_pool->table_buffer().size()};

      const U32 first_page = win_prefix / 64U;
      const U32 last_page = (win_end - 1U) / 64U;
      const U32 num_touched_pages = last_page - first_page + 1U;

      const auto emit_refresh_pages = [&](Encoder &e, const Block &blk) {
        detail::BlendRefreshPageSummaryParams rp{};
        rp.key_value_heads = kvh;
        rp.head_dim = U32(config.head_dim);
        rp.page_tokens = 64U;
        rp.first_logical_page = first_page;
        rp.num_logical_pages = num_touched_pages;
        rp.total_valid_tokens = win_end;
        auto &rf = e.emit(
            "cache_blend", "omlx_m5_blend_refresh_kv_page_summary_affine4_g128",
            {.width = 32U, .height = U64(num_touched_pages) * kvh},
            {.width = 32U}, {blk.state0, blk.state1, table_ref});
        e.dynamic(rf, 3, pod_bytes(rp));
      };

      // Upload window tokens once to token_ids.
      std::fill_n(token_staging.begin(), win_len, 0U);
      std::copy(prompt.begin() + win_prefix, prompt.begin() + win_end,
                token_staging.begin());
      const_cast<MetalBuffer *>(token_ids.buffer)
          ->write(std::as_bytes(std::span(token_staging).first(win_len)));
      const_cast<MetalBuffer *>(attention_positions.buffer)
          ->write(std::as_bytes(win_pos));

      // ====================================================================
      // Stage 1: KV Instantiation + Pre-Probe + Layer-3 Probe + HKVD Select
      // ====================================================================
      const bool use_w4a4_win = use_w4a4_for_t(win_len, false);
      const bool use_w4a8_win = !use_w4a4_win && use_w4a8_for_t(win_len, false);
      const U64 seq1 =
          blend_plan_seq_.fetch_add(1ULL, std::memory_order_relaxed);
      Encoder e1(context, libraries, win_len,
                 "qwen.blend.warm1." + std::to_string(seq1) + plan_key_suffix());
      if (use_w4a4_win) {
        e1.profile.used_w4a4 = true;
      }
      if (use_w4a8_win) {
        e1.profile.used_w4a8 = true;
      }

      // 1A. Instantiate cached KV for all hit segments across all full-attn layers.
      e1.begin_stage("blend_instantiate_kv");
      {
        U32 attn_idx = 0U;
        for (const auto &blk : blocks) {
          if (blk.type != LayerType::full_attention)
            continue;
          for (std::size_t si = seg_begin; si < seg_end; ++si) {
            const auto &ps = plan.segments[si];
            if (!ps.is_cache_hit())
              continue;
            const auto &he = *ps.hit_entry;
            const U64 lkv = U64(he.geometry.token_length) * U64(kvh) *
                            he.geometry.attn_vec_bytes();
            const U64 lrot =
                U64(he.geometry.token_length) * U64(kvh) * U64(rot_dim) * 2ULL;
            detail::BlendInstantiateKvParams ip{};
            ip.key_value_heads = kvh;
            ip.head_dim = U32(config.head_dim);
            ip.rotary_dim = rot_dim;
            ip.chunk_tokens = ps.segment.length;
            ip.dst_start_pos = ps.segment.start;
            ip.page_tokens = 64U;
            ip.theta = float(config.rope_theta);
            ip.src_token_offset = 0U;
            ip.use_rotary_bf16 = 1U;
            auto &inst = e1.emit(
                "cache_blend", "omlx_m5_blend_instantiate_kv_affine4_g128",
                {.width = 32U, .height = U64(ps.segment.length) * kvh},
                {.width = 32U},
                {Ref{&he.payload.attn_k_pre, U64(attn_idx) * lkv, lkv},
                 Ref{&he.payload.attn_k_rotary_bf16, U64(attn_idx) * lrot, lrot},
                 Ref{&he.payload.attn_v, U64(attn_idx) * lkv, lkv}, blk.state0,
                 blk.state1, table_ref});
            e1.dynamic(inst, 6, pod_bytes(ip));
          }
          emit_refresh_pages(e1, blk);
          ++attn_idx;
        }
      }

      // 1B. Populate full_hidden from cached H3_res for hit segments & mix3_cached.
      e1.begin_stage("blend_dequant_h3");
      for (std::size_t si = seg_begin; si < seg_end; ++si) {
        const auto &ps = plan.segments[si];
        if (!ps.is_cache_hit())
          continue;
        const auto &he = *ps.hit_entry;
        const U32 rel_start = ps.segment.start - win_prefix;
        const U32 lm = ps.segment.length;
        detail::BlendInt8QuantParams dq{.tokens = lm, .width = h};
        const Ref dst_h3{full_hidden_ref.buffer,
                         full_hidden_ref.offset + mul(rel_start, mul(h, 2ULL)),
                         mul(lm, mul(h, 2ULL))};
        auto &dh = e1.emit(
            "cache_blend", "omlx_m5_blend_dequantize_int8_g128_bf16",
            {.width = 32U, .height = U64(lm) * (h / 128U)}, {.width = 32U},
            {Ref{&he.payload.h3_res_i8, 0, he.geometry.h3_res_i8_bytes()},
             Ref{&he.payload.h3_res_scales, 0, he.geometry.h3_res_scale_bytes()},
             dst_h3});
        e1.dynamic(dh, 3, pod_bytes(dq));
        copy(e1,
             Ref{&he.payload.mix3_cached_i8, 0,
                 he.geometry.mix3_cached_i8_bytes()},
             Ref{mix3_i8_ref.buffer, mix3_i8_ref.offset + mul(rel_start, U64(q)),
                 he.geometry.mix3_cached_i8_bytes()},
             he.geometry.mix3_cached_i8_bytes());
        copy(e1,
             Ref{&he.payload.mix3_cached_scales, 0,
                 he.geometry.mix3_cached_scale_bytes()},
             Ref{mix3_sc_ref.buffer,
                 mix3_sc_ref.offset + mul(rel_start, mul(q / 128U, 2ULL)),
                 he.geometry.mix3_cached_scale_bytes()},
             he.geometry.mix3_cached_scale_bytes());
      }

      // 1C. Pre-probe layers (0 .. probe_idx - 1): dense boundary/miss + GDN jump!
      for (std::size_t si = seg_begin; si < seg_end; ++si) {
        const auto &ps = plan.segments[si];
        const U32 rel_start = ps.segment.start - win_prefix;
        const U32 run_len = ps.is_cache_hit()
                                ? std::min<U32>(ps.segment.boundary_window,
                                                ps.segment.length)
                                : ps.segment.length;
        if (run_len > 0U) {
          e1.begin_stage("blend_pre_probe.seg" + std::to_string(rel_start));
          e1.delegated_prefill = (run_len >= 16U);
          const Ref seg_tok{token_ids.buffer,
                            token_ids.offset + mul(rel_start, 4ULL),
                            mul(run_len, 4ULL)};
          embedding_lookup(e1, embedding, seg_tok, hidden, zero, v, h, run_len);
          const bool seg_w4a4 = use_w4a4_for_t(run_len, false);
          const bool seg_w4a8 = !seg_w4a4 && use_w4a8_for_t(run_len, false);
          for (U32 li = 0; li < probe_idx; ++li) {
            const auto &blk = blocks[li];
            if (seg_w4a4) {
              emit_rmsnorm_a4_bm16_bk128(e1, hidden,
                                         direct(blk.get("input_norm"), h), norm,
                                         w4a4_codes, w4a4_scales, h,
                                         float(config.rms_norm_eps),
                                         normalization_weight_offset, true,
                                         run_len);
              e1.set_a4_quant(make_a4_key(e1, norm, h, run_len));
            } else if (seg_w4a8) {
              emit_rmsnorm_a8_g64(e1, hidden, direct(blk.get("input_norm"), h),
                                  norm, a8_qa, a8_sa, a8_ra, h,
                                  float(config.rms_norm_eps),
                                  normalization_weight_offset, true, run_len);
              e1.set_a8_quant(make_a8_key(e1, norm, h, run_len));
            } else {
              normalize(e1, blk.get("input_norm"), hidden, norm, run_len);
            }
            const std::array<MaterializedWeightView, 2> qkvz_parts{
                blk.get("gdn_qkv"), blk.get("gdn_z")};
            const std::array<U32, 2> qkvz_ns{conv, gv};
            const std::array<MaterializedWeightView, 2> ab_parts{
                blk.get("gdn_a"), blk.get("gdn_b")};
            const std::array<U32, 2> ab_ns{hv, hv};
            if (seg_w4a4 &&
                try_concat_w4a4_nax(e1, blk.concat_gdn_qkvz_id, qkvz_parts,
                                    qkvz_ns, h, norm, concat_out, run_len)) {
              emit_split2(e1, concat_out, qkv, z, conv, gv, run_len);
            } else if (seg_w4a4) {
              if (!try_linear_w4a4_base(e1, qkvz_parts[0], norm, qkv, conv, h,
                                        run_len))
                linear_base(e1, qkvz_parts[0], norm, qkv, conv, h, run_len);
              if (!try_linear_w4a4_base(e1, qkvz_parts[1], norm, z, gv, h,
                                        run_len))
                linear_base(e1, qkvz_parts[1], norm, z, gv, h, run_len);
            } else if (seg_w4a8 &&
                try_concat_w4a8_nax(e1, blk.concat_gdn_qkvz_id, qkvz_parts,
                                    qkvz_ns, h, norm, concat_out, run_len)) {
              emit_split2(e1, concat_out, qkv, z, conv, gv, run_len);
            } else if (seg_w4a8) {
              if (!try_linear_w4a8_base(e1, qkvz_parts[0], norm, qkv, conv, h,
                                        run_len))
                linear_base(e1, qkvz_parts[0], norm, qkv, conv, h, run_len);
              if (!try_linear_w4a8_base(e1, qkvz_parts[1], norm, z, gv, h,
                                        run_len))
                linear_base(e1, qkvz_parts[1], norm, z, gv, h, run_len);
            } else if (try_concat_nax(e1, blk.concat_gdn_qkvz_id, qkvz_parts,
                                      qkvz_ns, h, false, norm, concat_out,
                                      run_len)) {
              emit_split2(e1, concat_out, qkv, z, conv, gv, run_len);
            } else {
              linear_base(e1, qkvz_parts[0], norm, qkv, conv, h, run_len);
              linear_base(e1, qkvz_parts[1], norm, z, gv, h, run_len);
            }
            if (seg_w4a4 &&
                try_concat_w4a4_nax(e1, blk.concat_gdn_ab_id, ab_parts, ab_ns,
                                    h, norm, concat_out, run_len)) {
              emit_split2(e1, concat_out, a, b, hv, hv, run_len);
            } else if (seg_w4a4) {
              if (!try_linear_w4a4_base(e1, ab_parts[0], norm, a, hv, h,
                                        run_len))
                linear_base(e1, ab_parts[0], norm, a, hv, h, run_len);
              if (!try_linear_w4a4_base(e1, ab_parts[1], norm, b, hv, h,
                                        run_len))
                linear_base(e1, ab_parts[1], norm, b, hv, h, run_len);
            } else if (seg_w4a8 &&
                try_concat_w4a8_nax(e1, blk.concat_gdn_ab_id, ab_parts, ab_ns,
                                    h, norm, concat_out, run_len)) {
              emit_split2(e1, concat_out, a, b, hv, hv, run_len);
            } else if (seg_w4a8) {
              if (!try_linear_w4a8_base(e1, ab_parts[0], norm, a, hv, h,
                                        run_len))
                linear_base(e1, ab_parts[0], norm, a, hv, h, run_len);
              if (!try_linear_w4a8_base(e1, ab_parts[1], norm, b, hv, h,
                                        run_len))
                linear_base(e1, ab_parts[1], norm, b, hv, h, run_len);
            } else if (try_concat_nax(e1, blk.concat_gdn_ab_id, ab_parts, ab_ns,
                                      h, false, norm, concat_out, run_len)) {
              emit_split2(e1, concat_out, a, b, hv, hv, run_len);
            } else {
              linear_base(e1, ab_parts[0], norm, a, hv, h, run_len);
              linear_base(e1, ab_parts[1], norm, b, hv, h, run_len);
            }
            emit_blend_gdn_slice(e1, blk, blk.state0, blk.state1, 0U, run_len);
            if (!(seg_w4a4 &&
                  try_linear_w4a4_base(e1, blk.get("gdn_out"), mixer, delta, h,
                                       gv, run_len)) &&
                !(seg_w4a8 &&
                  try_linear_w4a8_base(e1, blk.get("gdn_out"), mixer, delta, h,
                                       gv, run_len))) {
              linear(e1, blk.get("gdn_out"), mixer, delta, h, gv, run_len);
            }
            if (seg_w4a4) {
              emit_add_rmsnorm_a4_bm16_bk128(
                  e1, hidden, delta, direct(blk.get("post_attention_norm"), h),
                  norm, w4a4_codes, w4a4_scales, h, float(config.rms_norm_eps),
                  normalization_weight_offset, (config.num_experts > 0),
                  run_len);
              e1.set_a4_quant(make_a4_key(e1, norm, h, run_len));
            } else if (seg_w4a8) {
              emit_add_rmsnorm_a8_g64(
                  e1, hidden, delta, direct(blk.get("post_attention_norm"), h),
                  norm, a8_qa, a8_sa, a8_ra, h, float(config.rms_norm_eps),
                  normalization_weight_offset, (config.num_experts > 0),
                  run_len);
              e1.set_a8_quant(make_a8_key(e1, norm, h, run_len));
            } else {
              add_normalize(e1, blk.get("post_attention_norm"), hidden, delta,
                            norm, run_len);
            }
            mlp(e1, blk, li, false, run_len, nullptr, std::nullopt, false,
                seg_w4a8, /*verify=*/false, seg_w4a4);
            add(e1, run_len);
          }
          copy(e1, hidden,
               Ref{full_hidden_ref.buffer,
                   full_hidden_ref.offset + mul(rel_start, mul(h, 2ULL)),
                   mul(run_len, mul(h, 2ULL))},
               mul(run_len, mul(h, 2ULL)));
        }
        if (ps.is_cache_hit() && ps.segment.length > run_len) {
          U32 gdn_i = 0U;
          for (U32 li = 0; li < probe_idx; ++li) {
            if (blocks[li].type != LayerType::full_attention) {
              emit_gdn_jump(e1, blocks[li], gdn_i, *ps.hit_entry);
              ++gdn_i;
            }
          }
        }
      }

      // 1D. Layer probe_idx (Layer 3): scatter mandatory K/V, cross-chunk Attn, HKVD!
      {
        const auto &pblk = blocks[probe_idx];
        e1.begin_stage("blend_probe_layer");
        e1.delegated_prefill = false;
        if (use_w4a4_win) {
          emit_rmsnorm_a4_bm16_bk128(
              e1, full_hidden_ref, direct(pblk.get("input_norm"), h), norm,
              w4a4_codes, w4a4_scales, h, float(config.rms_norm_eps),
              normalization_weight_offset, /*write_bf16=*/true);
          e1.set_a4_quant(make_a4_key(e1, norm, h));
        } else if (use_w4a8_win) {
          emit_rmsnorm_a8_g64(
              e1, full_hidden_ref, direct(pblk.get("input_norm"), h), norm,
              a8_qa, a8_sa, a8_ra, h, float(config.rms_norm_eps),
              normalization_weight_offset, /*write_bf16=*/true);
          e1.set_a8_quant(make_a8_key(e1, norm, h));
        } else {
          normalize(e1, pblk.get("input_norm"), full_hidden_ref, norm);
        }

        if (m_mand > 0U) {
          e1.delegated_prefill = (m_mand >= 16U);
          detail::BlendGatherScatterParams gp{
              .max_active_tokens = m_mand, .width = h, .use_device_count = 0U,
              .residual_add = 0U};
          const U32 gx = (h % 4U == 0U) ? (h / 4U) : h;
          auto &gm = e1.emit("cache_blend", "omlx_m5_blend_gather_rows_bf16",
                             {.width = gx, .height = m_mand},
                             {.width = std::min<U32>(128U, gx)},
                             {norm, mand_idx_ref, zero, delta});
          e1.dynamic(gm, 4, pod_bytes(gp));
          const bool mand_w4a4 = use_w4a4_for_t(m_mand, false);
          const bool mand_w4a8 = !mand_w4a4 && use_w4a8_for_t(m_mand, false);
          const bool mand_k_a8 =
              (mand_w4a8 || (mand_w4a4 && w4a4_keep_qk_a8_)) &&
              try_linear_w4a8_base(e1, pblk.get("attention_k"), delta, key, kv,
                                   h, m_mand);
          if (!mand_k_a8 &&
              !(mand_w4a4 && try_linear_w4a4_base(e1, pblk.get("attention_k"),
                                                  delta, key, kv, h, m_mand))) {
            linear_base(e1, pblk.get("attention_k"), delta, key, kv, h, m_mand);
          }
          if (!(mand_w4a4 && try_linear_w4a4_base(e1, pblk.get("attention_v"),
                                                  delta, value, kv, h, m_mand)) &&
              !(mand_w4a8 &&
                try_linear_w4a8_base(e1, pblk.get("attention_v"), delta, value,
                                     kv, h, m_mand))) {
            linear_base(e1, pblk.get("attention_v"), delta, value, kv, h,
                        m_mand);
          }
          e1.delegated_prefill = false;
          detail::BlendQkNormRopeScatterParams sp{};
          sp.query_heads = 0U;
          sp.key_value_heads = kvh;
          sp.head_dim = U32(config.head_dim);
          sp.rotary_dim = rot_dim;
          sp.max_active_tokens = m_mand;
          sp.query_token_stride = q * 2U;
          sp.query_head_stride = U32(config.head_dim * 2);
          sp.key_token_stride = kv;
          sp.key_head_stride = U32(config.head_dim);
          sp.value_token_stride = kv;
          sp.page_tokens = 64U;
          sp.use_device_count = 0U;
          sp.epsilon = float(config.rms_norm_eps);
          sp.theta = float(config.rope_theta);
          sp.weight_offset = normalization_weight_offset;
          auto &skv = e1.emit(
              "cache_blend",
              "omlx_m5_attention_qk_norm_rope_kv_scatter_affine4_g128",
              {.width = 32U, .height = U64(m_mand) * kvh}, {.width = 32U},
              {query, key, value, direct(pblk.get("q_norm"), config.head_dim),
               direct(pblk.get("k_norm"), config.head_dim), pblk.state0,
               pblk.state1, table_ref, mand_pos_ref, zero});
          e1.dynamic(skv, 10, pod_bytes(sp));
          emit_refresh_pages(e1, pblk);
        }

        const bool probe_q_a8 =
            (use_w4a8_win || (use_w4a4_win && w4a4_keep_qk_a8_)) &&
            try_linear_w4a8_base(e1, pblk.get("attention_q"), norm, query,
                                 2 * q, h);
        if (!probe_q_a8 &&
            !(use_w4a4_win && try_linear_w4a4_base(e1, pblk.get("attention_q"),
                                                   norm, query, 2 * q, h))) {
          linear_base(e1, pblk.get("attention_q"), norm, query, 2 * q, h);
        }
        {
          auto &post = e1.emit(
              "attention_exact", "omlx_m5_attention_qk_norm_rope_exact",
              {.width = 32U, .height = U64(win_len) * (qh + kvh)},
              {.width = 32U},
              {query, key, direct(pblk.get("q_norm"), config.head_dim),
               direct(pblk.get("k_norm"), config.head_dim), prompt_pos_ref});
          e1.dynamic(post, 5,
                     words({qh, kvh, U32(config.head_dim), rot_dim, win_len,
                            q * 2, U32(config.head_dim * 2), kv,
                            U32(config.head_dim), win_prefix,
                            word(float(config.rms_norm_eps)),
                            word(float(config.rope_theta)),
                            word(normalization_weight_offset)}));
        }
        emit_blend_attn_route_b(e1, pblk, win_len, win_prefix, ctx_env,
                                prompt_pos_ref);
        copy(e1, mixer, full_mixer_ref, mul(win_len, mul(q, 2ULL)));

        detail::BlendHkvdSelectParams hp{};
        hp.total_tokens = win_len;
        hp.mix_width = q;
        hp.cached_is_int8 = 1U;
        hp.compute_l2_inline = 0U;
        hp.explicit_hkvd_k = 0U;
        hp.hkvd_ratio = detail::kBlendDefaultHkvdRatio;
        auto &l2 = e1.emit("cache_blend", "omlx_m5_blend_hkvd_l2_score",
                           {.width = 128U, .height = win_len},
                           {.width = 128U},
                           {full_mixer_ref, mix3_i8_ref, mix3_sc_ref, flags_ref,
                            l2_ref});
        e1.dynamic(l2, 5, pod_bytes(hp));
        auto &tk = e1.emit("cache_blend", "omlx_m5_blend_hkvd_topk_select",
                           {.width = 256U}, {.width = 256U},
                           {full_mixer_ref, mix3_i8_ref, mix3_sc_ref, flags_ref,
                            prompt_pos_ref, l2_ref, act_idx_ref, act_pos_ref,
                            act_cnt_ref});
        e1.dynamic(tk, 9, pod_bytes(hp));
      }

      auto prof1 = e1.finish();
      prof1.parameters.write("step", prof1.parameter_data);
      run_blend_exec_profile(prof1);

      U32 m_active = 0U;
      {
        const auto cnt_b = active_cnt_buf.read(0, 4);
        std::memcpy(&m_active, cnt_b.data(), 4);
      }
      if (m_active == 0U || m_active > win_len) {
        throw std::runtime_error("CacheBlend HKVD produced invalid M_active");
      }
      std::vector<U32> active_idx_host(m_active);
      std::vector<U32> active_pos_host(m_active);
      {
        const auto ib = active_idx_buf.read(0, mul(m_active, 4ULL));
        const auto pb = active_pos_buf.read(0, mul(m_active, 4ULL));
        std::memcpy(active_idx_host.data(), ib.data(), ib.size());
        std::memcpy(active_pos_host.data(), pb.data(), pb.size());
      }
      const_cast<MetalBuffer *>(attention_positions.buffer)
          ->write(std::as_bytes(std::span(active_pos_host)));

      U32 recomputed_hit_tokens = 0U;
      for (const U32 rel_i : active_idx_host) {
        const U32 abs_p = win_prefix + rel_i;
        for (std::size_t si = seg_begin; si < seg_end; ++si) {
          const auto &ps = plan.segments[si];
          if (ps.is_cache_hit() && abs_p >= ps.segment.start &&
              abs_p < ps.segment.start + ps.segment.length) {
            ++recomputed_hit_tokens;
            break;
          }
        }
      }

      // ====================================================================
      // Stage 2: Deep Layers (Layer 3 finish + Layers 4..L-1) on [M_active, H]
      // Composes active Route A (BF16/W4A8) + Route B (Exact/Sage2/MInf) +
      // Dual-ANE co-execution whenever ane_active_ && m_active >= ane_chunk!
      // ====================================================================
      const bool use_w4a8_active = use_w4a8_for_t(m_active, false);
      const U64 gdn_state_bytes =
          (config.linear_attention_layers() > 0)
              ? mul(mul(U64(hv), 128ULL * 128ULL), 4ULL)
              : 256ULL;
      const U64 gdn_hist_bytes =
          (config.linear_attention_layers() > 0) ? gdn_tail_bytes : 256ULL;
      MetalBuffer gdn_tmp_state = context.make_private_buffer(gdn_state_bytes);
      MetalBuffer gdn_tmp_hist = context.make_private_buffer(gdn_hist_bytes);
      const Ref gdn_tmp_s_ref{&gdn_tmp_state, 0, gdn_state_bytes};
      const Ref gdn_tmp_h_ref{&gdn_tmp_hist, 0, gdn_hist_bytes};

      U32 pre_probe_gdn_count = 0U;
      for (U32 li = 0; li <= probe_idx; ++li) {
        if (blocks[li].type != LayerType::full_attention) {
          ++pre_probe_gdn_count;
        }
      }

      struct ActiveSegSlice {
        const detail::BlendPlannedSegment *planned = nullptr;
        U32 a_begin = 0U;
        U32 w_bnd_count = 0U;
        U32 a_end = 0U;
      };
      std::vector<ActiveSegSlice> seg_slices;
      seg_slices.reserve(seg_end - seg_begin);
      {
        U32 cursor = 0U;
        for (std::size_t si = seg_begin; si < seg_end; ++si) {
          const auto &ps = plan.segments[si];
          const U32 seg_rel_start = ps.segment.start - win_prefix;
          const U32 seg_rel_end = seg_rel_start + ps.segment.length;
          const U32 a_begin = cursor;
          U32 w_bnd_cnt = 0U;
          const U32 bnd_end =
              seg_rel_start +
              (ps.is_cache_hit()
                   ? std::min<U32>(ps.segment.boundary_window, ps.segment.length)
                   : 0U);
          while (cursor < m_active && active_idx_host[cursor] < seg_rel_end) {
            if (active_idx_host[cursor] < bnd_end) {
              ++w_bnd_cnt;
            }
            ++cursor;
          }
          seg_slices.push_back({&ps, a_begin, w_bnd_cnt, cursor});
        }
      }

      auto emit_stage2_head_and_sample = [&](Encoder &eh) {
        eh.begin_stage("blend_head");
        Ref last_hidden{delta.buffer, 0, mul(h, 2)};
        Ref last_norm{norm.buffer, 0, mul(h, 2)};
        Ref row0_logits{logits.buffer, 0, mul(v, 2)};
        {
          auto &extract =
              eh.emit("qwen_forward", "omlx_m5_forward_last_row", {.width = h},
                      {.width = 256}, {hidden, last_hidden});
          eh.dynamic(extract, 2, words({m_active, h}));
        }
        normalize(eh, final_norm, last_hidden, last_norm, 1U);
        linear(eh, head, last_norm, row0_logits, v, h, 1U);
        eh.begin_stage("blend_sample");
        const bool use_64tg_argmax =
            (v >= 256U) && (argmax_scratch.buffer != nullptr) &&
            !detail::forward::env_flag_enabled("OMLX_M5_DISABLE_64TG_ARGMAX");
        if (sampling_contract == SamplingContract::PositionCoupledV1) {
          if (use_64tg_argmax) {
            auto &st1 = eh.emit("position_sampling",
                                "omlx_m5_position_argmax_stage1_64tg",
                                {.width = 64U * 256U, .height = 1U},
                                {.width = 256U}, {row0_logits, argmax_scratch});
            eh.constant(st1, 2, words({v, 1, 1, 0}));
            auto &sample = eh.emit(
                "position_sampling", "omlx_m5_position_sample_logits_stage2",
                {.width = 256U}, {.width = 256U},
                {row0_logits, argmax_scratch, sampled, position_ids,
                 position_scores});
            eh.profile.sampling_offset = eh.dynamic(
                sample, 5, std::vector<std::byte>(sizeof(PositionSamplingRow)));
            eh.constant(sample, 6, words({v, 1, 1, 0}));
          } else {
            auto &sample =
                eh.emit("position_sampling", "omlx_m5_position_sample_logits",
                        {.width = 256}, {.width = 256},
                        {row0_logits, sampled, position_ids, position_scores});
            eh.profile.sampling_offset = eh.dynamic(
                sample, 4, std::vector<std::byte>(sizeof(PositionSamplingRow)));
            eh.constant(sample, 5, words({v, 1, 1, 0}));
          }
        } else if (use_64tg_argmax) {
          Ref vocab_parts{argmax_scratch.buffer, 0, 512U};
          auto &st1 =
              eh.emit("vocab", "omlx_m5_vocab_argmax_stage1_64tg",
                      {.width = 64U * 256U}, {.width = 256U},
                      {row0_logits, vocab_parts});
          eh.constant(st1, 2, words({v}));
          auto &sample =
              eh.emit("vocab", "omlx_m5_sample_logits_stage2", {.width = 256U},
                      {.width = 256U}, {row0_logits, vocab_parts, sampled});
          eh.profile.sampling_offset =
              eh.dynamic(sample, 3, std::vector<std::byte>(sizeof(SampleParams)));
        } else {
          auto &sample = eh.emit("vocab", "omlx_m5_sample_logits", {.width = 256},
                                 {.width = 256}, {row0_logits, sampled});
          eh.profile.sampling_offset =
              eh.dynamic(sample, 2, std::vector<std::byte>(sizeof(SampleParams)));
        }
      };

      auto patch_stage2_sampling = [&](Profile &hp) {
        if (sampling_contract == SamplingContract::PositionCoupledV1) {
          const PositionSamplingRow row{.seed = sampling.seed,
                                        .position = win_end,
                                        .top_k = sampling.top_k,
                                        .temperature = sampling.temperature,
                                        .top_p = sampling.top_p};
          std::memcpy(hp.parameter_data.data() + hp.sampling_offset, &row,
                      sizeof(row));
        } else {
          const SampleParams sp{v,
                                sampling.top_k,
                                sampling.temperature,
                                sampling.top_p,
                                sampling.seed,
                                sampling.stream,
                                win_end - 1U};
          std::memcpy(hp.parameter_data.data() + hp.sampling_offset, &sp,
                      sizeof(sp));
        }
      };

      const bool stage2_ane =
          ane_active_ && ane_engine_ != nullptr && ane_chunk_tokens_ > 0U &&
          m_active >= ane_chunk_tokens_ && config.num_experts == 0 &&
          !inspect_routes_enabled && !inspect_tensors_workspace &&
          bypass_lora_ == nullptr && feature_layers.empty();

      bool stage2_used_w4a8 = use_w4a8_active;
      bool stage2_used_sage2 = false;
      bool stage2_used_minf = false;
      U32 stage2_minf_env = 0U;
      U32 stage2_ane_offload_rows = 0U;

      if (stage2_ane) {
        const U32 ane_blk_t = ane_chunk_tokens_;
        const U32 n_ane_blks = m_active / ane_blk_t;
        const U32 ane_rows_total = n_ane_blks * ane_blk_t;
        const U32 tail_rows = m_active - ane_rows_total;
        stage2_ane_offload_rows = ane_rows_total;

        // Pointer-stable profile storage kept alive until after classic barrier.
        std::deque<Profile> stage2_profiles;
        const U64 seq2 =
            blend_plan_seq_.fetch_add(1ULL, std::memory_order_relaxed);
        const std::string base_tag =
            "qwen.blend.warm2.ane." + std::to_string(seq2) + plan_key_suffix();

        auto submit_stage2_prof = [&](Profile &&pr) {
          stage2_profiles.push_back(std::move(pr));
          auto &stored = stage2_profiles.back();
          if (stored.used_w4a8)
            stage2_used_w4a8 = true;
          if (stored.used_sage2)
            stage2_used_sage2 = true;
          if (stored.used_minference) {
            stage2_used_minf = true;
            stage2_minf_env = stored.minf_context_envelope;
          }
          stored.parameters.write("step", stored.parameter_data);
          const std::array calls{MetalCommandInvocation{
              .plan = &stored.plan,
              .bindings = stored.bindings,
              .parameters = &stored.parameters}};
          context.submit_classic_async(calls);
        };

        auto run_stage2_ane_linear =
            [&](U32 li, ane::AneProjectionRole role,
                const MaterializedWeightView &w, Ref src, Ref dst, U32 n_out,
                U32 k_in, bool op_w4a8, bool carried_a8,
                const std::string &tag) {
              const auto *op = ane_engine_->find_op(li, role);
              const bool offload =
                  (op != nullptr && op->split.enabled() &&
                   can_gpu_suffix_linear(w, n_out, k_in, op_w4a8));
              if (!offload) {
                Encoder e(context, libraries, m_active, base_tag + tag);
                e.begin_stage("blend_ane_fb");
                if (op_w4a8 && carried_a8) {
                  e.set_a8_quant(make_a8_key(e, src, k_in));
                }
                if (!(op_w4a8 &&
                      try_linear_w4a8_base(e, w, src, dst, n_out, k_in))) {
                  linear(e, w, src, dst, n_out, k_in);
                }
                submit_stage2_prof(e.finish());
                return;
              }
              for (U32 bi = 0U; bi < n_ane_blks; ++bi) {
                const U32 row0 = bi * ane_blk_t;
                const Ref src_s{src.buffer,
                                src.offset + mul(row0, mul(k_in, 2ULL)),
                                mul(ane_blk_t, mul(k_in, 2ULL))};
                const Ref dst_s{dst.buffer,
                                dst.offset + mul(row0, mul(n_out, 2ULL)),
                                mul(ane_blk_t, mul(n_out, 2ULL))};
                Encoder e(context, libraries, ane_blk_t,
                          base_tag + tag + ".b" + std::to_string(bi));
                e.begin_stage("blend_ane_gpu_sfx");
                if (op_w4a8 && carried_a8 && row0 == 0U &&
                    w4a8_nax_mp(m_active) == w4a8_nax_mp(ane_blk_t)) {
                  e.set_a8_quant(make_a8_key(e, src_s, k_in));
                }
                (void)emit_gpu_suffix_linear(
                    e, w, src_s, dst_s, n_out, k_in, op_w4a8,
                    op->split.gpu_row_offset, n_out, op->split.gpu_rows);
                stage2_profiles.push_back(e.finish());
                auto &stored = stage2_profiles.back();
                if (stored.used_w4a8)
                  stage2_used_w4a8 = true;
                stored.parameters.write("step", stored.parameter_data);
                ane_engine_->enqueue_op(
                    op->op_id, src_s.buffer->native_mtl_buffer(), src_s.offset,
                    dst_s.buffer->native_mtl_buffer(), dst_s.offset, n_out,
                    [&](void *mtl_cb) {
                      const std::array calls{MetalCommandInvocation{
                          .plan = &stored.plan,
                          .bindings = stored.bindings,
                          .parameters = &stored.parameters}};
                      context.encode_into_command_buffer(mtl_cb, calls);
                    },
                    k_in);
              }
              if (tail_rows > 0U) {
                const Ref src_t{src.buffer,
                                src.offset +
                                    mul(ane_rows_total, mul(k_in, 2ULL)),
                                mul(tail_rows, mul(k_in, 2ULL))};
                const Ref dst_t{dst.buffer,
                                dst.offset +
                                    mul(ane_rows_total, mul(n_out, 2ULL)),
                                mul(tail_rows, mul(n_out, 2ULL))};
                const bool tail_w4a8 =
                    op_w4a8 && use_w4a8_for_t(tail_rows, false);
                Encoder e(context, libraries, tail_rows,
                          base_tag + tag + ".tail");
                e.begin_stage("blend_ane_tail");
                if (!(tail_w4a8 &&
                      try_linear_w4a8_base(e, w, src_t, dst_t, n_out, k_in))) {
                  linear(e, w, src_t, dst_t, n_out, k_in);
                }
                submit_stage2_prof(e.finish());
              }
            };

        auto run_stage2_ane_swiglu =
            [&](U32 li, const MaterializedWeightView &gw,
                const MaterializedWeightView &uw, Ref src, Ref dst, U32 n_out,
                U32 k_in, bool op_w4a8, bool carried_a8,
                const std::string &tag) {
              const auto *op = ane_engine_->find_op(
                  li, ane::AneProjectionRole::MlpGateUpSwiGlu);
              const bool offload =
                  (op != nullptr && op->split.enabled() &&
                   can_gpu_suffix_swiglu(gw, uw, n_out, k_in, op_w4a8));
              if (!offload) {
                Encoder e(context, libraries, m_active, base_tag + tag);
                e.begin_stage("blend_ane_gu_fb");
                if (op_w4a8 && carried_a8) {
                  e.set_a8_quant(make_a8_key(e, src, k_in));
                }
                if (!(op_w4a8 &&
                      try_linear_w4a8_swiglu(e, gw, uw, src, dst, n_out,
                                             k_in))) {
                  if (!try_linear_swiglu(e, gw, uw, src, dst, n_out, k_in)) {
                    linear_base(e, gw, src, gate, n_out, k_in);
                    linear_base(e, uw, src, up, n_out, k_in);
                    swiglu(e, m_active, n_out);
                  }
                }
                submit_stage2_prof(e.finish());
                return;
              }
              for (U32 bi = 0U; bi < n_ane_blks; ++bi) {
                const U32 row0 = bi * ane_blk_t;
                const Ref src_s{src.buffer,
                                src.offset + mul(row0, mul(k_in, 2ULL)),
                                mul(ane_blk_t, mul(k_in, 2ULL))};
                const Ref dst_s{dst.buffer,
                                dst.offset + mul(row0, mul(n_out, 2ULL)),
                                mul(ane_blk_t, mul(n_out, 2ULL))};
                Encoder e(context, libraries, ane_blk_t,
                          base_tag + tag + ".b" + std::to_string(bi));
                e.begin_stage("blend_ane_gu_sfx");
                if (op_w4a8 && carried_a8 && row0 == 0U &&
                    w4a8_nax_mp(m_active) == w4a8_nax_mp(ane_blk_t)) {
                  e.set_a8_quant(make_a8_key(e, src_s, k_in));
                }
                (void)emit_gpu_suffix_swiglu(
                    e, gw, uw, src_s, dst_s, n_out, k_in, op_w4a8,
                    op->split.gpu_row_offset, n_out, op->split.gpu_rows);
                stage2_profiles.push_back(e.finish());
                auto &stored = stage2_profiles.back();
                if (stored.used_w4a8)
                  stage2_used_w4a8 = true;
                stored.parameters.write("step", stored.parameter_data);
                ane_engine_->enqueue_op(
                    op->op_id, src_s.buffer->native_mtl_buffer(), src_s.offset,
                    dst_s.buffer->native_mtl_buffer(), dst_s.offset, n_out,
                    [&](void *mtl_cb) {
                      const std::array calls{MetalCommandInvocation{
                          .plan = &stored.plan,
                          .bindings = stored.bindings,
                          .parameters = &stored.parameters}};
                      context.encode_into_command_buffer(mtl_cb, calls);
                    },
                    k_in);
              }
              if (tail_rows > 0U) {
                const Ref src_t{src.buffer,
                                src.offset +
                                    mul(ane_rows_total, mul(k_in, 2ULL)),
                                mul(tail_rows, mul(k_in, 2ULL))};
                const Ref dst_t{dst.buffer,
                                dst.offset +
                                    mul(ane_rows_total, mul(n_out, 2ULL)),
                                mul(tail_rows, mul(n_out, 2ULL))};
                const bool tail_w4a8 =
                    op_w4a8 && use_w4a8_for_t(tail_rows, false);
                Encoder e(context, libraries, tail_rows,
                          base_tag + tag + ".tail");
                e.begin_stage("blend_ane_gu_tail");
                if (!(tail_w4a8 &&
                      try_linear_w4a8_swiglu(e, gw, uw, src_t, dst_t, n_out,
                                             k_in))) {
                  if (!try_linear_swiglu(e, gw, uw, src_t, dst_t, n_out,
                                         k_in)) {
                    linear_base(e, gw, src_t, gate, n_out, k_in);
                    linear_base(e, uw, src_t, up, n_out, k_in);
                    auto &sw = e.emit("mlp_layer", "omlx_m5_mlp_swiglu",
                                      {.width = U64(tail_rows) * n_out},
                                      {.width = 256}, {gate, up, dst_t});
                    e.dynamic(sw, 3, words({tail_rows, n_out}));
                  }
                }
                submit_stage2_prof(e.finish());
              }
            };

        auto run_stage2_ane_mlp_and_add = [&](U32 li, const Block &blk) {
          const std::string ltag = ".L" + std::to_string(li);
          {
            Encoder e(context, libraries, m_active,
                      base_tag + ltag + ".post_norm");
            e.begin_stage("post_norm");
            if (use_w4a8_active) {
              emit_add_rmsnorm_a8_g64(
                  e, hidden, delta, direct(blk.get("post_attention_norm"), h),
                  norm, a8_qa, a8_sa, a8_ra, h, float(config.rms_norm_eps),
                  normalization_weight_offset, /*write_bf16=*/true);
            } else {
              add_normalize(e, blk.get("post_attention_norm"), hidden, delta,
                            norm);
            }
            submit_stage2_prof(e.finish());
          }
          const auto &gw = blk.get("mlp_gate");
          const auto &uw = blk.get("mlp_up");
          const auto &dw = blk.get("mlp_down");
          const auto *op_gu = ane_engine_->find_op(
              li, ane::AneProjectionRole::MlpGateUpSwiGlu);
          const auto *op_dn =
              ane_engine_->find_op(li, ane::AneProjectionRole::MlpDown);
          const bool stage2_mode_b_a8 =
              (n_ane_blks == 1U && tail_rows == 0U && op_gu != nullptr &&
               op_gu->split.enabled() && op_dn == nullptr &&
               can_gpu_suffix_swiglu_fwht64_a8(gw, uw, dw, f, h,
                                               use_w4a8_active));
          if (stage2_mode_b_a8) {
            // Mode B W4A8 splice for CacheBlend Stage 2: GPU suffix [N_ane, f)
            // + fused ANE unpack [0, N_ane) directly into a8_mid_{qa,sa,ra}.
            Encoder e_gu(context, libraries, ane_blk_t,
                         base_tag + ltag + ".mlp_gu.mb");
            e_gu.begin_stage("blend_ane_gu_sfx_mb");
            e_gu.set_a8_quant(make_a8_key(e_gu, norm, h));
            (void)try_linear_w4a8_swiglu_fwht64_a8(
                e_gu, gw, uw, norm, product, f, h, /*also_write_bf16=*/false,
                std::nullopt, op_gu->split.gpu_row_offset, f,
                op_gu->split.gpu_rows);
            stage2_profiles.push_back(e_gu.finish());
            auto &stored_gu = stage2_profiles.back();
            if (stored_gu.used_w4a8)
              stage2_used_w4a8 = true;
            stored_gu.parameters.write("step", stored_gu.parameter_data);

            ane::AneUnpackTarget mb_target{};
            mb_target.mode = ane::AneUnpackMode::Fwht64A8;
            mb_target.primary_buf = a8_mid_qa.buffer->native_mtl_buffer();
            mb_target.primary_offset = a8_mid_qa.offset;
            mb_target.aux0_buf = a8_mid_sa.buffer->native_mtl_buffer();
            mb_target.aux0_offset = a8_mid_sa.offset;
            mb_target.aux1_buf = a8_mid_ra.buffer->native_mtl_buffer();
            mb_target.aux1_offset = a8_mid_ra.offset;
            mb_target.out_stride = f;
            mb_target.dst_col_offset = 0U;
            mb_target.mp = w4a8_nax_mp(ane_blk_t);
            ane_engine_->enqueue_op_ex(
                op_gu->op_id, norm.buffer->native_mtl_buffer(), norm.offset, h,
                mb_target, [&](void *mtl_cb) {
                  const std::array calls{MetalCommandInvocation{
                      .plan = &stored_gu.plan,
                      .bindings = stored_gu.bindings,
                      .parameters = &stored_gu.parameters}};
                  context.encode_into_command_buffer(mtl_cb, calls);
                });

            Encoder e_dn(context, libraries, ane_blk_t,
                         base_tag + ltag + ".mlp_dn.mb");
            e_dn.begin_stage("blend_ane_dn_mb");
            e_dn.set_a8_quant(make_a8_key_for(e_dn, product, f, a8_mid_qa,
                                              a8_mid_sa, a8_mid_ra,
                                              std::nullopt, /*fwht64=*/true));
            if (!try_linear_w4a8_base(e_dn, dw, product, delta, h, f)) {
              linear(e_dn, dw, product, delta, h, f);
            }
            add(e_dn);
            submit_stage2_prof(e_dn.finish());
            return;
          }
          run_stage2_ane_swiglu(li, gw, uw, norm, product, f, h,
                                use_w4a8_active, use_w4a8_active,
                                ltag + ".mlp_gu");
          run_stage2_ane_linear(
              li, ane::AneProjectionRole::MlpDown, dw, product, delta, h, f,
              use_w4a8_active && (dw.has_fwht64() || w4a8_down_proj_), false,
              ltag + ".mlp_dn");
          {
            Encoder e(context, libraries, m_active, base_tag + ltag + ".add");
            e.begin_stage("add");
            add(e);
            submit_stage2_prof(e.finish());
          }
        };

        // Gather active rows for Layer probe_idx finish.
        {
          Encoder e(context, libraries, m_active, base_tag + ".gather");
          e.begin_stage("blend_gather_active");
          detail::BlendGatherScatterParams gh{
              .max_active_tokens = m_active, .width = h, .use_device_count = 0U,
              .residual_add = 0U};
          const U32 gxh = (h % 4U == 0U) ? (h / 4U) : h;
          auto &g1 = e.emit("cache_blend", "omlx_m5_blend_gather_rows_bf16",
                            {.width = gxh, .height = m_active},
                            {.width = std::min<U32>(128U, gxh)},
                            {full_hidden_ref, act_idx_ref, act_cnt_ref, hidden});
          e.dynamic(g1, 4, pod_bytes(gh));
          detail::BlendGatherScatterParams gm{
              .max_active_tokens = m_active, .width = q, .use_device_count = 0U,
              .residual_add = 0U};
          const U32 gxq = (q % 4U == 0U) ? (q / 4U) : q;
          auto &g2 = e.emit("cache_blend", "omlx_m5_blend_gather_rows_bf16",
                            {.width = gxq, .height = m_active},
                            {.width = std::min<U32>(128U, gxq)},
                            {full_mixer_ref, act_idx_ref, act_cnt_ref, mixer});
          e.dynamic(g2, 4, pod_bytes(gm));
          submit_stage2_prof(e.finish());
        }

        // Finish Layer probe_idx with Dual-ANE on o_proj + MLP.
        {
          const auto &pblk = blocks[probe_idx];
          run_stage2_ane_linear(probe_idx, ane::AneProjectionRole::AttnOProj,
                                pblk.get("attention_o"), mixer, delta, h, q,
                                use_w4a8_active, false, ".probe.o");
          run_stage2_ane_mlp_and_add(probe_idx, pblk);
        }

        // Deep layers probe_idx + 1 .. blocks.size() - 1 with Dual-ANE streaming!
        U32 gdn_idx = pre_probe_gdn_count;
        for (U32 li = probe_idx + 1U; li < blocks.size(); ++li) {
          const auto &blk = blocks[li];
          const std::string ltag = ".L" + std::to_string(li);
          if (blk.type == LayerType::full_attention) {
            {
              Encoder e(context, libraries, m_active,
                        base_tag + ltag + ".attn_pre");
              e.begin_stage("blend_deep_attn");
              if (use_w4a8_active) {
                emit_rmsnorm_a8_g64(e, hidden, direct(blk.get("input_norm"), h),
                                    norm, a8_qa, a8_sa, a8_ra, h,
                                    float(config.rms_norm_eps),
                                    normalization_weight_offset, true);
                e.set_a8_quant(make_a8_key(e, norm, h));
              } else {
                normalize(e, blk.get("input_norm"), hidden, norm);
              }
              const std::array<MaterializedWeightView, 3> parts{
                  blk.get("attention_q"), blk.get("attention_k"),
                  blk.get("attention_v")};
              const std::array<U32, 3> part_ns{2 * q, kv, kv};
              if (use_w4a8_active &&
                  try_concat_w4a8_nax(e, blk.concat_qkv_id, parts, part_ns, h,
                                      norm, concat_out)) {
                emit_split3(e, concat_out, query, key, value, 2 * q, kv, kv);
              } else if (use_w4a8_active) {
                if (!try_linear_w4a8_base(e, parts[0], norm, query, 2 * q, h))
                  linear_base(e, parts[0], norm, query, 2 * q, h);
                if (!try_linear_w4a8_base(e, parts[1], norm, key, kv, h))
                  linear_base(e, parts[1], norm, key, kv, h);
                if (!try_linear_w4a8_base(e, parts[2], norm, value, kv, h))
                  linear_base(e, parts[2], norm, value, kv, h);
              } else if (try_concat_nax(e, blk.concat_qkv_id, parts, part_ns, h,
                                        false, norm, concat_out)) {
                emit_split3(e, concat_out, query, key, value, 2 * q, kv, kv);
              } else {
                linear_base(e, parts[0], norm, query, 2 * q, h);
                linear_base(e, parts[1], norm, key, kv, h);
                linear_base(e, parts[2], norm, value, kv, h);
              }
              detail::BlendQkNormRopeScatterParams sp{};
              sp.query_heads = qh;
              sp.key_value_heads = kvh;
              sp.head_dim = U32(config.head_dim);
              sp.rotary_dim = rot_dim;
              sp.max_active_tokens = m_active;
              sp.query_token_stride = q * 2U;
              sp.query_head_stride = U32(config.head_dim * 2);
              sp.key_token_stride = kv;
              sp.key_head_stride = U32(config.head_dim);
              sp.value_token_stride = kv;
              sp.page_tokens = 64U;
              sp.use_device_count = 0U;
              sp.epsilon = float(config.rms_norm_eps);
              sp.theta = float(config.rope_theta);
              sp.weight_offset = normalization_weight_offset;
              auto &sc = e.emit(
                  "cache_blend",
                  "omlx_m5_attention_qk_norm_rope_kv_scatter_affine4_g128",
                  {.width = 32U, .height = U64(m_active) * (qh + kvh)},
                  {.width = 32U},
                  {query, key, value, direct(blk.get("q_norm"), config.head_dim),
                   direct(blk.get("k_norm"), config.head_dim), blk.state0,
                   blk.state1, table_ref, attention_positions, zero});
              e.dynamic(sc, 10, pod_bytes(sp));
              emit_refresh_pages(e, blk);
              emit_blend_attn_route_b(e, blk, m_active, active_pos_host.front(),
                                      ctx_env, attention_positions);
              submit_stage2_prof(e.finish());
            }
            run_stage2_ane_linear(li, ane::AneProjectionRole::AttnOProj,
                                  blk.get("attention_o"), mixer, delta, h, q,
                                  use_w4a8_active, false, ltag + ".o");
          } else {
            // GDN input_norm + in_proj_qkv/a/b strictly 100% GPU (NEVER ANE!).
            {
              Encoder e(context, libraries, m_active,
                        base_tag + ltag + ".gdn_qkvab");
              e.begin_stage("blend_deep_gdn_qkvab");
              if (use_w4a8_active) {
                emit_rmsnorm_a8_g64(e, hidden, direct(blk.get("input_norm"), h),
                                    norm, a8_qa, a8_sa, a8_ra, h,
                                    float(config.rms_norm_eps),
                                    normalization_weight_offset,
                                    /*write_bf16=*/true);
                e.set_a8_quant(make_a8_key(e, norm, h));
              } else {
                normalize(e, blk.get("input_norm"), hidden, norm);
              }
              const std::array<MaterializedWeightView, 2> ab_parts{
                  blk.get("gdn_a"), blk.get("gdn_b")};
              const std::array<U32, 2> ab_ns{hv, hv};
              if (!(use_w4a8_active &&
                    try_linear_w4a8_base(e, blk.get("gdn_qkv"), norm, qkv, conv,
                                         h))) {
                linear_base(e, blk.get("gdn_qkv"), norm, qkv, conv, h);
              }
              if (use_w4a8_active &&
                  try_concat_w4a8_nax(e, blk.concat_gdn_ab_id, ab_parts, ab_ns,
                                      h, norm, concat_out)) {
                emit_split2(e, concat_out, a, b, hv, hv);
              } else if (use_w4a8_active) {
                if (!try_linear_w4a8_base(e, ab_parts[0], norm, a, hv, h))
                  linear_base(e, ab_parts[0], norm, a, hv, h);
                if (!try_linear_w4a8_base(e, ab_parts[1], norm, b, hv, h))
                  linear_base(e, ab_parts[1], norm, b, hv, h);
              } else if (try_concat_nax(e, blk.concat_gdn_ab_id, ab_parts,
                                        ab_ns, h, false, norm, concat_out)) {
                emit_split2(e, concat_out, a, b, hv, hv);
              } else {
                linear_base(e, ab_parts[0], norm, a, hv, h);
                linear_base(e, ab_parts[1], norm, b, hv, h);
              }
              submit_stage2_prof(e.finish());
            }
            // GDN in_proj_z on Dual-ANE || GPU suffix!
            run_stage2_ane_linear(li, ane::AneProjectionRole::GdnInProjZ,
                                  blk.get("gdn_z"), norm, z, gv, h,
                                  use_w4a8_active, use_w4a8_active,
                                  ltag + ".gdn_z");
            // GDN recurrence slices + state jump on GPU.
            {
              Encoder e(context, libraries, m_active,
                        base_tag + ltag + ".gdn_rec");
              e.begin_stage("blend_deep_gdn_rec");
              for (const auto &ss : seg_slices) {
                const auto &ps = *ss.planned;
                const U32 seg_m = ss.a_end - ss.a_begin;
                if (ps.is_cache_hit()) {
                  if (ss.w_bnd_count > 0U) {
                    emit_blend_gdn_slice(e, blk, blk.state0, blk.state1,
                                         ss.a_begin, ss.w_bnd_count);
                  }
                  const U32 hkvd_m = seg_m - ss.w_bnd_count;
                  if (hkvd_m > 0U) {
                    copy(e, blk.state0, gdn_tmp_s_ref, gdn_state_bytes);
                    copy(e, blk.state1, gdn_tmp_h_ref, gdn_hist_bytes);
                    emit_blend_gdn_slice(e, blk, gdn_tmp_s_ref, gdn_tmp_h_ref,
                                         ss.a_begin + ss.w_bnd_count, hkvd_m);
                  }
                  if (ps.segment.length > ss.w_bnd_count) {
                    emit_gdn_jump(e, blk, gdn_idx, *ps.hit_entry);
                  }
                } else if (seg_m > 0U) {
                  emit_blend_gdn_slice(e, blk, blk.state0, blk.state1,
                                       ss.a_begin, seg_m);
                }
              }
              submit_stage2_prof(e.finish());
            }
            // GDN out_proj on Dual-ANE || GPU suffix!
            run_stage2_ane_linear(li, ane::AneProjectionRole::GdnOutProj,
                                  blk.get("gdn_out"), mixer, delta, h, gv,
                                  use_w4a8_active, false, ltag + ".gdn_out");
            ++gdn_idx;
          }
          run_stage2_ane_mlp_and_add(li, blk);
        }

        if (is_final_window) {
          Encoder eh(context, libraries, m_active, base_tag + ".head");
          emit_stage2_head_and_sample(eh);
          auto hp = eh.finish();
          patch_stage2_sampling(hp);
          submit_stage2_prof(std::move(hp));
        }
        ane_engine_->flush_and_wait();
        static_cast<void>(context.submit_classic_barrier());
      } else {
        const bool use_w4a4_active = use_w4a4_for_t(m_active, false);
        const bool use_w4a8_active_gpu = !use_w4a4_active && use_w4a8_active;
        const U64 seq2 =
            blend_plan_seq_.fetch_add(1ULL, std::memory_order_relaxed);
        Encoder e2(context, libraries, m_active,
                   "qwen.blend.warm2." + std::to_string(seq2) +
                       plan_key_suffix());
        if (use_w4a4_active) {
          e2.profile.used_w4a4 = true;
        }
        if (use_w4a8_active_gpu) {
          e2.profile.used_w4a8 = true;
        }

        // Gather hidden[0..M_active, H] and mixer[0..M_active, q] for probe finish.
        e2.begin_stage("blend_gather_active");
        {
          detail::BlendGatherScatterParams gh{
              .max_active_tokens = m_active, .width = h, .use_device_count = 0U,
              .residual_add = 0U};
          const U32 gxh = (h % 4U == 0U) ? (h / 4U) : h;
          auto &g1 = e2.emit("cache_blend", "omlx_m5_blend_gather_rows_bf16",
                             {.width = gxh, .height = m_active},
                             {.width = std::min<U32>(128U, gxh)},
                             {full_hidden_ref, act_idx_ref, act_cnt_ref, hidden});
          e2.dynamic(g1, 4, pod_bytes(gh));

          detail::BlendGatherScatterParams gm{
              .max_active_tokens = m_active, .width = q, .use_device_count = 0U,
              .residual_add = 0U};
          const U32 gxq = (q % 4U == 0U) ? (q / 4U) : q;
          auto &g2 = e2.emit("cache_blend", "omlx_m5_blend_gather_rows_bf16",
                             {.width = gxq, .height = m_active},
                             {.width = std::min<U32>(128U, gxq)},
                             {full_mixer_ref, act_idx_ref, act_cnt_ref, mixer});
          e2.dynamic(g2, 4, pod_bytes(gm));
        }

        // Finish Layer probe_idx on [M_active, H].
        {
          const auto &pblk = blocks[probe_idx];
          e2.begin_stage("blend_probe_finish");
          if (!(use_w4a4_active &&
                try_linear_w4a4_base(e2, pblk.get("attention_o"), mixer, delta,
                                     h, q)) &&
              !(use_w4a8_active_gpu &&
                try_linear_w4a8_base(e2, pblk.get("attention_o"), mixer, delta,
                                     h, q))) {
            linear(e2, pblk.get("attention_o"), mixer, delta, h, q);
          }
          if (use_w4a4_active) {
            emit_add_rmsnorm_a4_bm16_bk128(
                e2, hidden, delta, direct(pblk.get("post_attention_norm"), h),
                norm, w4a4_codes, w4a4_scales, h, float(config.rms_norm_eps),
                normalization_weight_offset, (config.num_experts > 0));
            e2.set_a4_quant(make_a4_key(e2, norm, h));
          } else if (use_w4a8_active_gpu) {
            emit_add_rmsnorm_a8_g64(
                e2, hidden, delta, direct(pblk.get("post_attention_norm"), h),
                norm, a8_qa, a8_sa, a8_ra, h, float(config.rms_norm_eps),
                normalization_weight_offset, (config.num_experts > 0));
            e2.set_a8_quant(make_a8_key(e2, norm, h));
          } else {
            add_normalize(e2, pblk.get("post_attention_norm"), hidden, delta,
                          norm);
          }
          mlp(e2, pblk, probe_idx, false, {}, nullptr, std::nullopt, false,
              use_w4a8_active_gpu, /*verify=*/false, use_w4a4_active);
          add(e2);
        }

        U32 gdn_idx = pre_probe_gdn_count;
        for (U32 li = probe_idx + 1U; li < blocks.size(); ++li) {
          const auto &blk = blocks[li];
          e2.begin_stage("blend_deep.L" + std::to_string(li));
          if (use_w4a4_active) {
            emit_rmsnorm_a4_bm16_bk128(e2, hidden,
                                       direct(blk.get("input_norm"), h), norm,
                                       w4a4_codes, w4a4_scales, h,
                                       float(config.rms_norm_eps),
                                       normalization_weight_offset, true);
            e2.set_a4_quant(make_a4_key(e2, norm, h));
          } else if (use_w4a8_active_gpu) {
            emit_rmsnorm_a8_g64(e2, hidden, direct(blk.get("input_norm"), h),
                                norm, a8_qa, a8_sa, a8_ra, h,
                                float(config.rms_norm_eps),
                                normalization_weight_offset, true);
            e2.set_a8_quant(make_a8_key(e2, norm, h));
          } else {
            normalize(e2, blk.get("input_norm"), hidden, norm);
          }

          if (blk.type == LayerType::full_attention) {
            const std::array<MaterializedWeightView, 3> parts{
                blk.get("attention_q"), blk.get("attention_k"),
                blk.get("attention_v")};
            const std::array<U32, 3> part_ns{2 * q, kv, kv};
            if (use_w4a4_active && !w4a4_keep_qk_a8_ &&
                try_concat_w4a4_nax(e2, blk.concat_qkv_id, parts, part_ns, h,
                                    norm, concat_out)) {
              emit_split3(e2, concat_out, query, key, value, 2 * q, kv, kv);
            } else if (use_w4a4_active) {
              if (w4a4_keep_qk_a8_) {
                if (!try_linear_w4a8_base(e2, parts[0], norm, query, 2 * q, h) &&
                    !try_linear_w4a4_base(e2, parts[0], norm, query, 2 * q, h))
                  linear_base(e2, parts[0], norm, query, 2 * q, h);
                if (!try_linear_w4a8_base(e2, parts[1], norm, key, kv, h) &&
                    !try_linear_w4a4_base(e2, parts[1], norm, key, kv, h))
                  linear_base(e2, parts[1], norm, key, kv, h);
              } else {
                if (!try_linear_w4a4_base(e2, parts[0], norm, query, 2 * q, h))
                  linear_base(e2, parts[0], norm, query, 2 * q, h);
                if (!try_linear_w4a4_base(e2, parts[1], norm, key, kv, h))
                  linear_base(e2, parts[1], norm, key, kv, h);
              }
              if (!try_linear_w4a4_base(e2, parts[2], norm, value, kv, h))
                linear_base(e2, parts[2], norm, value, kv, h);
            } else if (use_w4a8_active_gpu &&
                try_concat_w4a8_nax(e2, blk.concat_qkv_id, parts, part_ns, h,
                                    norm, concat_out)) {
              emit_split3(e2, concat_out, query, key, value, 2 * q, kv, kv);
            } else if (use_w4a8_active_gpu) {
              if (!try_linear_w4a8_base(e2, parts[0], norm, query, 2 * q, h))
                linear_base(e2, parts[0], norm, query, 2 * q, h);
              if (!try_linear_w4a8_base(e2, parts[1], norm, key, kv, h))
                linear_base(e2, parts[1], norm, key, kv, h);
              if (!try_linear_w4a8_base(e2, parts[2], norm, value, kv, h))
                linear_base(e2, parts[2], norm, value, kv, h);
            } else if (try_concat_nax(e2, blk.concat_qkv_id, parts, part_ns, h,
                                      false, norm, concat_out)) {
              emit_split3(e2, concat_out, query, key, value, 2 * q, kv, kv);
            } else {
              linear_base(e2, parts[0], norm, query, 2 * q, h);
              linear_base(e2, parts[1], norm, key, kv, h);
              linear_base(e2, parts[2], norm, value, kv, h);
            }

            detail::BlendQkNormRopeScatterParams sp{};
            sp.query_heads = qh;
            sp.key_value_heads = kvh;
            sp.head_dim = U32(config.head_dim);
            sp.rotary_dim = rot_dim;
            sp.max_active_tokens = m_active;
            sp.query_token_stride = q * 2U;
            sp.query_head_stride = U32(config.head_dim * 2);
            sp.key_token_stride = kv;
            sp.key_head_stride = U32(config.head_dim);
            sp.value_token_stride = kv;
            sp.page_tokens = 64U;
            sp.use_device_count = 0U;
            sp.epsilon = float(config.rms_norm_eps);
            sp.theta = float(config.rope_theta);
            sp.weight_offset = normalization_weight_offset;
            auto &sc = e2.emit(
                "cache_blend",
                "omlx_m5_attention_qk_norm_rope_kv_scatter_affine4_g128",
                {.width = 32U, .height = U64(m_active) * (qh + kvh)},
                {.width = 32U},
                {query, key, value, direct(blk.get("q_norm"), config.head_dim),
                 direct(blk.get("k_norm"), config.head_dim), blk.state0,
                 blk.state1, table_ref, attention_positions, zero});
            e2.dynamic(sc, 10, pod_bytes(sp));
            emit_refresh_pages(e2, blk);

            emit_blend_attn_route_b(e2, blk, m_active, active_pos_host.front(),
                                    ctx_env, attention_positions);
            if (!(use_w4a4_active &&
                  try_linear_w4a4_base(e2, blk.get("attention_o"), mixer, delta,
                                       h, q)) &&
                !(use_w4a8_active_gpu &&
                  try_linear_w4a8_base(e2, blk.get("attention_o"), mixer, delta,
                                       h, q))) {
              linear(e2, blk.get("attention_o"), mixer, delta, h, q);
            }
          } else {
            const std::array<MaterializedWeightView, 2> qkvz_parts{
                blk.get("gdn_qkv"), blk.get("gdn_z")};
            const std::array<U32, 2> qkvz_ns{conv, gv};
            const std::array<MaterializedWeightView, 2> ab_parts{
                blk.get("gdn_a"), blk.get("gdn_b")};
            const std::array<U32, 2> ab_ns{hv, hv};
            if (use_w4a4_active &&
                try_concat_w4a4_nax(e2, blk.concat_gdn_qkvz_id, qkvz_parts,
                                    qkvz_ns, h, norm, concat_out)) {
              emit_split2(e2, concat_out, qkv, z, conv, gv);
            } else if (use_w4a4_active) {
              if (!try_linear_w4a4_base(e2, qkvz_parts[0], norm, qkv, conv, h))
                linear_base(e2, qkvz_parts[0], norm, qkv, conv, h);
              if (!try_linear_w4a4_base(e2, qkvz_parts[1], norm, z, gv, h))
                linear_base(e2, qkvz_parts[1], norm, z, gv, h);
            } else if (use_w4a8_active_gpu &&
                try_concat_w4a8_nax(e2, blk.concat_gdn_qkvz_id, qkvz_parts,
                                    qkvz_ns, h, norm, concat_out)) {
              emit_split2(e2, concat_out, qkv, z, conv, gv);
            } else if (use_w4a8_active_gpu) {
              if (!try_linear_w4a8_base(e2, qkvz_parts[0], norm, qkv, conv, h))
                linear_base(e2, qkvz_parts[0], norm, qkv, conv, h);
              if (!try_linear_w4a8_base(e2, qkvz_parts[1], norm, z, gv, h))
                linear_base(e2, qkvz_parts[1], norm, z, gv, h);
            } else if (try_concat_nax(e2, blk.concat_gdn_qkvz_id, qkvz_parts,
                                      qkvz_ns, h, false, norm, concat_out)) {
              emit_split2(e2, concat_out, qkv, z, conv, gv);
            } else {
              linear_base(e2, qkvz_parts[0], norm, qkv, conv, h);
              linear_base(e2, qkvz_parts[1], norm, z, gv, h);
            }
            if (use_w4a4_active &&
                try_concat_w4a4_nax(e2, blk.concat_gdn_ab_id, ab_parts, ab_ns,
                                    h, norm, concat_out)) {
              emit_split2(e2, concat_out, a, b, hv, hv);
            } else if (use_w4a4_active) {
              if (!try_linear_w4a4_base(e2, ab_parts[0], norm, a, hv, h))
                linear_base(e2, ab_parts[0], norm, a, hv, h);
              if (!try_linear_w4a4_base(e2, ab_parts[1], norm, b, hv, h))
                linear_base(e2, ab_parts[1], norm, b, hv, h);
            } else if (use_w4a8_active_gpu &&
                try_concat_w4a8_nax(e2, blk.concat_gdn_ab_id, ab_parts, ab_ns,
                                    h, norm, concat_out)) {
              emit_split2(e2, concat_out, a, b, hv, hv);
            } else if (use_w4a8_active_gpu) {
              if (!try_linear_w4a8_base(e2, ab_parts[0], norm, a, hv, h))
                linear_base(e2, ab_parts[0], norm, a, hv, h);
              if (!try_linear_w4a8_base(e2, ab_parts[1], norm, b, hv, h))
                linear_base(e2, ab_parts[1], norm, b, hv, h);
            } else if (try_concat_nax(e2, blk.concat_gdn_ab_id, ab_parts, ab_ns,
                                      h, false, norm, concat_out)) {
              emit_split2(e2, concat_out, a, b, hv, hv);
            } else {
              linear_base(e2, ab_parts[0], norm, a, hv, h);
              linear_base(e2, ab_parts[1], norm, b, hv, h);
            }

            for (const auto &ss : seg_slices) {
              const auto &ps = *ss.planned;
              const U32 seg_m = ss.a_end - ss.a_begin;
              if (ps.is_cache_hit()) {
                if (ss.w_bnd_count > 0U) {
                  emit_blend_gdn_slice(e2, blk, blk.state0, blk.state1,
                                       ss.a_begin, ss.w_bnd_count);
                }
                const U32 hkvd_m = seg_m - ss.w_bnd_count;
                if (hkvd_m > 0U) {
                  copy(e2, blk.state0, gdn_tmp_s_ref, gdn_state_bytes);
                  copy(e2, blk.state1, gdn_tmp_h_ref, gdn_hist_bytes);
                  emit_blend_gdn_slice(e2, blk, gdn_tmp_s_ref, gdn_tmp_h_ref,
                                       ss.a_begin + ss.w_bnd_count, hkvd_m);
                }
                if (ps.segment.length > ss.w_bnd_count) {
                  emit_gdn_jump(e2, blk, gdn_idx, *ps.hit_entry);
                }
              } else if (seg_m > 0U) {
                emit_blend_gdn_slice(e2, blk, blk.state0, blk.state1,
                                     ss.a_begin, seg_m);
              }
            }

            if (!(use_w4a4_active &&
                  try_linear_w4a4_base(e2, blk.get("gdn_out"), mixer, delta, h,
                                       gv)) &&
                !(use_w4a8_active_gpu &&
                  try_linear_w4a8_base(e2, blk.get("gdn_out"), mixer, delta, h,
                                       gv))) {
              linear(e2, blk.get("gdn_out"), mixer, delta, h, gv);
            }
            ++gdn_idx;
          }

          if (use_w4a4_active) {
            emit_add_rmsnorm_a4_bm16_bk128(
                e2, hidden, delta, direct(blk.get("post_attention_norm"), h),
                norm, w4a4_codes, w4a4_scales, h, float(config.rms_norm_eps),
                normalization_weight_offset, (config.num_experts > 0));
            e2.set_a4_quant(make_a4_key(e2, norm, h));
          } else if (use_w4a8_active_gpu) {
            emit_add_rmsnorm_a8_g64(
                e2, hidden, delta, direct(blk.get("post_attention_norm"), h),
                norm, a8_qa, a8_sa, a8_ra, h, float(config.rms_norm_eps),
                normalization_weight_offset, (config.num_experts > 0));
            e2.set_a8_quant(make_a8_key(e2, norm, h));
          } else {
            add_normalize(e2, blk.get("post_attention_norm"), hidden, delta,
                          norm);
          }
          mlp(e2, blk, li, false, {}, nullptr, std::nullopt, false,
              use_w4a8_active_gpu, /*verify=*/false, use_w4a4_active);
          add(e2);
        }

        if (is_final_window) {
          emit_stage2_head_and_sample(e2);
        }
        auto prof2 = e2.finish();
        if (is_final_window) {
          patch_stage2_sampling(prof2);
        }
        prof2.parameters.write("step", prof2.parameter_data);
        run_blend_exec_profile(prof2);
        stage2_used_w4a8 = prof2.used_w4a8;
        stage2_used_sage2 = prof2.used_sage2;
        stage2_used_minf = prof2.used_minference;
        stage2_minf_env = prof2.minf_context_envelope;
      }

      QwenForwardResult result{
          .tokens = win_len,
          .executed_layers = U32(blocks.size()),
      };
      if (is_final_window) {
        const auto tok_bytes = sampled.buffer->read(
            0, sampling_contract == SamplingContract::PositionCoupledV1 ? 8 : 4);
        std::memcpy(&result.token, tok_bytes.data(), 4);
        result.host_read_bytes = tok_bytes.size();
        if (sampling_contract == SamplingContract::PositionCoupledV1) {
          U32 flag = 0;
          std::memcpy(&flag, tok_bytes.data() + 4, 4);
          if (flag & 2U)
            throw std::runtime_error("nonfinite position target logits");
          if (flag & 1U) {
            auto ib = position_ids.buffer->read(0, U64(sampling.top_k) * 4);
            auto sb = position_scores.buffer->read(0, U64(sampling.top_k) * 4);
            std::vector<U32> ids(sampling.top_k);
            std::vector<float> scores(sampling.top_k);
            std::memcpy(ids.data(), ib.data(), ib.size());
            std::memcpy(scores.data(), sb.data(), sb.size());
            result.token = position_sample_candidates(
                ids, scores,
                {.seed = sampling.seed,
                 .position = win_end,
                 .top_k = sampling.top_k,
                 .temperature = sampling.temperature,
                 .top_p = sampling.top_p});
            result.host_read_bytes += ib.size() + sb.size();
          }
        } else if (result.token >= v) {
          throw std::runtime_error("nonfinite or invalid legacy sampled token");
        }
        if (inspect_tensors) {
          result.hidden = bf16_values(*norm.buffer, h);
          result.logits = bf16_values(*logits.buffer, v);
          result.host_read_bytes += mul(U64(h) + U64(v), 2ULL);
        }
      }

      page_pool->commit(win_len);
      page_guard.active = false;
      next_position += win_len;
      last_logits_row = 0U;
      has_last_logits = is_final_window;

      const bool any_w4a4 =
          prof1.used_w4a4 || (!stage2_ane && use_w4a4_for_t(m_active, false));
      const bool any_w4a8 = prof1.used_w4a8 || stage2_used_w4a8;
      const bool any_sage2 = prof1.used_sage2 || stage2_used_sage2;
      const bool any_minf = stage2_used_minf || prof1.used_minference;
      const U32 minf_cap = stage2_used_minf ? m_active : win_len;
      const U32 minf_env =
          stage2_used_minf ? stage2_minf_env : prof1.minf_context_envelope;
      QwenPrefillTelemetry win_delta;
      record_prefill_telemetry(
          m_active, active_pos_host.front(), any_w4a8, any_sage2, any_minf,
          minf_cap, minf_env, /*used_ane=*/stage2_ane, &win_delta,
          stage2_ane ? std::optional<U32>{stage2_ane_offload_rows}
                     : std::nullopt,
          any_w4a4);
      prefill_telemetry_.prefill_cache_blend_hit_tokens += win_hit_tokens;
      prefill_telemetry_.prefill_cache_blend_recomputed_tokens +=
          recomputed_hit_tokens;
      accum.prefill_w4a8_tokens += win_delta.prefill_w4a8_tokens;
      accum.prefill_w4a4_tokens += win_delta.prefill_w4a4_tokens;
      accum.prefill_sage2_tokens += win_delta.prefill_sage2_tokens;
      accum.prefill_minference_blocks_total +=
          win_delta.prefill_minference_blocks_total;
      accum.prefill_minference_blocks_skipped +=
          win_delta.prefill_minference_blocks_skipped;
      accum.prefill_ane_offload_tokens += win_delta.prefill_ane_offload_tokens;
      accum.prefill_ane_fallback_tokens +=
          win_delta.prefill_ane_fallback_tokens;
      accum.prefill_cache_blend_hit_tokens += win_hit_tokens;
      accum.prefill_cache_blend_recomputed_tokens += recomputed_hit_tokens;
      return result;
    };

    // ======================================================================
    // Unified Multi-Window Execution across arbitrary 16K-256K prompts!
    // Groups consecutive warm-hit + dense-tail segments into <= workspace_rows_
    // windows while capturing every cold CacheableChunk along the way.
    // ======================================================================
    QwenPrefillTelemetry accum;
    accum.prefill_ane_active = ane_active_;
    accum.prefill_ane_split_alpha = ane_split_alpha_;
    accum.prefill_ane_mode_effective =
        std::string(prefill_ane_mode_to_string(ane_mode_effective_));
    accum.prefill_ane_inactive_reason = ane_inactive_reason_;
    QwenForwardResult final_res{};

    std::size_t si = 0U;
    while (si < plan.segments.size()) {
      const auto &ps = plan.segments[si];
      if (!ps.is_cache_hit() || ps.segment.length > workspace_rows_) {
        const bool is_last_seg = (si + 1U == plan.segments.size());
        if (!is_last_seg && !ps.is_cache_hit() &&
            ps.segment.kind == detail::BlendSegmentKind::CacheableChunk &&
            ps.segment.length <= workspace_rows_) {
          capture_and_insert_blend_chunk(
              prompt.subspan(ps.segment.start, ps.segment.length),
              ps.segment.start, eff_cfg.boundary_window, blend_cache,
              model_exec_hash, accum);
        } else {
          final_res = run_standard_chunked(
              ps.segment.start, ps.segment.start + ps.segment.length,
              is_last_seg, accum);
        }
        ++si;
        continue;
      }

      // Warm window starting at a hit segment: pack consecutive hit/dense-tail
      // segments up to workspace_rows_ (stopping before uncached CacheableChunks
      // so newly-seen cacheable documents are still captured into the cache).
      std::size_t w_end = si;
      U32 win_len = 0U;
      while (w_end < plan.segments.size()) {
        const auto &cand = plan.segments[w_end];
        if (!cand.is_cache_hit() &&
            cand.segment.kind == detail::BlendSegmentKind::CacheableChunk) {
          break;
        }
        if (win_len > 0U && win_len + cand.segment.length > workspace_rows_) {
          break;
        }
        win_len += cand.segment.length;
        ++w_end;
      }
      const bool is_final_window = (w_end == plan.segments.size());
      auto w_res = execute_warm_blend_window(si, w_end, is_final_window, accum);
      if (is_final_window) {
        final_res = std::move(w_res);
      }
      si = w_end;
    }

    final_res.tokens = suffix_len;
    final_res.prefill_delta = accum;
    return final_res;
  }

  const MetalContext &context;
  const MaterializedModel &weights;
  QwenModelConfig config;
  U32 max_context, next_position = 0;
  std::filesystem::path libraries;
  detail::SiliconTuningConfig silicon_tuning;
  KvStorageFormat kv_format_ = KvStorageFormat::BFloat16;
  QwenForward *shared_primary_ = nullptr;
  U32 workspace_rows_ = 1024U;
  U32 feature_ring_rows_ = 0U;
  U64 feature_epoch_ = 0ULL;
  U32 h = 0, f = 0, v = 0, q = 0, kv = 0, gk = 0, gv = 0, conv = 0;
  float normalization_weight_offset = 0.0F;
  QwenForwardMemoryPlan allocation_plan;
  std::set<std::string> allocation_names;
  std::vector<std::unique_ptr<MetalBuffer>> buffers;
  std::vector<MetalBuffer *> states;
  Ref token_ids, hidden, norm, query, key, value, qkv, z, a, b, convolved,
      recurrent, mixer, delta, gate, up, product, sums, logits, last_logits,
      sampled, zero, features;
  Ref packed_verify_logits, packed_verify_features, concat_out;
  Ref split_max, split_sum, split_output;
  Ref route_trace, router_logits_trace, router_scores_trace;
  Ref router, route_ids, route_scores, expert_output, routed, shared,
      shared_gate;
  MaterializedWeightView embedding, final_norm, head;
  std::vector<Block> blocks;
  std::unique_ptr<KvPagePool> page_pool;
  mutable std::map<U64, std::vector<U32>> lease_pages_;
  std::vector<Profile> profiles, batch_profiles;
  std::vector<Profile> split_profiles, split_verify_profiles,
      split_batch_profiles;
  std::deque<CachedPackedProfile> packed_cache_;
  std::vector<U32> token_staging;
  std::vector<U32> feature_layers;
  bool inspect_routes_enabled = false;
  bool inspect_tensors_workspace = false;
  bool speculative_enabled = false;
  bool m4_short_routes = false;
  LinearNumericPolicy linear_policy = LinearNumericPolicy::Legacy;
  SamplingContract sampling_contract = SamplingContract::LegacyRejectionV1;
  omlx::m5::AttentionNumericPolicy attention_policy =
      omlx::m5::AttentionNumericPolicy::Legacy;
  bool strict_two_pass_attn_ = false;
  bool flash_nax_prefill_ = false;
  Ref attention_positions;
  Ref position_ids, position_scores;
  bool split_attention_enabled = false;
  bool batch_enabled = false;
  std::optional<QwenLinearRouteOverride> diagnostic_override;
  bool diagnostic_override_matched = false;
  std::optional<QwenMoeRouteOverride> moe_override;
  bool moe_override_matched = false;
  U32 pending_verify = 0;
  U32 last_trace_tokens = 0;
  U32 last_logits_row = 0;
  bool has_last_logits = false;
  Ref feature_history;
  std::vector<Profile> verify_profiles;
  Profile commit_profile;
  Profile kv_cow_profile;
  Profile inspect_norm_profile;
  Profile inspect_head_chunk_profile;
  Profile delegated_sampler_profile;

  struct ArPipeSlot {
    Ref sampled;
    Ref attn_pos;
    std::vector<Profile> profiles;
  };
  std::array<ArPipeSlot, 2> ar_pipe_{};
  U32 ar_in_flight_count_ = 0U;
  U32 ar_pipe_cursor_ = 0U;
  U64 async_generation_ = 0U;

  struct PrefillPipeSlot {
    Ref token_ids;
    Ref attn_pos;
    std::vector<Profile> profiles;
  };
  std::array<PrefillPipeSlot, 2> prefill_pipe_{};
  U32 prefill_in_flight_count_ = 0U;
  U32 prefill_pipe_cursor_ = 0U;

  bool mxfp4_nax_active() const {
    return mxfp4_nax_cache && mxfp4_nax_cache->active();
  }
  bool affine_nax_active() const {
    return affine_nax_cache && affine_nax_cache->active();
  }
  bool bf16_nax_enabled = false;
  bool bf16_nax_bypass_ok_ = false;
  bool has_row_exact_metallib_ = false;
  const Profile *verify_async_profile_ = nullptr;
  MoeNaxWorkspace moe_nax_ws;
  std::shared_ptr<Mxfp4NaxCache> mxfp4_nax_cache;
  Ref f16_norm, f16_scratch, f16_row_scale, f16_ones_scale;
  Ref a4_codes, a4_scales;
  std::shared_ptr<AffineNaxCache> affine_nax_cache;
  std::shared_ptr<Q5High1Cache> q5_high1_cache;
  Ref affine_nax_xs;
  Ref argmax_scratch;
  Ref rms_sq_buf, moe_tg_done_buf;
  Ref packed_samp_parts, packed_samp_meta, packed_samp_ids, packed_samp_scores;
  const MetalBuffer *last_packed_logits_buf_ = nullptr;
  bool tdom_batch_override_ = false;
  U32 split_kv_partition_cap_override_ = 0U;
  Ref gdn_chunk_ws;
  bool gdn_chunk_nax_enabled_ = false;
  QwenPrefillOptions prefill_options_{};
  U32 w4a8_min_tokens_ = kW4A8DefaultMinTokens;
  bool w4a8_nax_enabled_ = false;
  bool w4a8_down_proj_ = false;
  U32 w4a4_min_tokens_ = kW4A4DefaultMinTokens;
  bool w4a4_nax_enabled_ = false;
  bool w4a4_keep_qk_a8_ = false;
  bool w4a4_down_proj_ = true;
  Ref fwht_scratch;
  Ref a8_qa, a8_sa, a8_ra;
  Ref a8_mid_qa, a8_mid_sa, a8_mid_ra;
  U32 a8_workspace_mp_ = 0U;
  Ref w4a4_codes, w4a4_scales;
  Ref w4a4_mid_codes, w4a4_mid_scales;
  U32 w4a4_workspace_mp_ = 0U;
  static constexpr U32 kDefaultSage2MinEndKv = 0U;
  static constexpr U32 kDefaultMInferenceMinMeanKv = 4096U;
  bool sage2_enabled_ = false;
  bool minference_enabled_ = false;
  U32 sage2_min_end_kv_ = kDefaultSage2MinEndKv;
  U32 minf_min_mean_kv_ = kDefaultMInferenceMinMeanKv;
  Ref minference_scratch;
  Ref sage2_kv_ws_;
  Sage2KvWorkspaceLayout sage2_kv_ws_layout_{};
  std::unique_ptr<ane::AneDualOffloadEngine> ane_engine_;
  U32 ane_chunk_tokens_ = 0U;
  bool ane_active_ = false;
  PrefillAneMode ane_mode_effective_ = PrefillAneMode::Off;
  std::string ane_inactive_reason_;
  double ane_split_alpha_ = 0.0;
  bool cache_blend_kernels_ready_ = false;
  std::map<LongCtxProfileKey, Profile> long_ctx_profiles_;
  std::map<AneChunkPlanKey, AneChunkPlan> ane_layer_plans_;
  std::map<AneChunkPlanKey, UltraSwiGluChunkPlan> ane_ultra_plans_;
  QwenPrefillTelemetry prefill_telemetry_{};
  std::atomic<U64> blend_plan_seq_{1ULL};
  std::shared_ptr<const QwenMergedLoraOverrides> merged_lora_;
  std::shared_ptr<const QwenBypassLoraRuntime> bypass_lora_;
  std::unique_ptr<MetalBuffer> lora_bypass_tmp_buf_;
  Ref lora_bypass_tmp_{};

  struct UmaCheckpointPool {
    static constexpr std::size_t kDefaultMaxPooledSlabs = 16U;
    std::mutex mutex;
    std::vector<std::shared_ptr<MetalBuffer>> free_slabs;
    std::atomic<U64> next_epoch{1ULL};
    std::size_t max_pooled_slabs = kDefaultMaxPooledSlabs;

    std::shared_ptr<MetalBuffer> acquire(const MetalContext &ctx,
                                         U64 min_bytes,
                                         U64 alloc_quantum = 4096ULL) {
      if (min_bytes == 0ULL)
        return {};
      const U64 quantum = std::max<U64>(4096ULL, alloc_quantum);
      const U64 aligned_bytes =
          ((min_bytes + quantum - 1ULL) / quantum) * quantum;
      std::shared_ptr<MetalBuffer> evict_undersized;
      {
        std::scoped_lock lock(mutex);
        std::size_t best_idx = free_slabs.size();
        U64 best_size = UINT64_MAX;
        for (std::size_t i = 0; i < free_slabs.size(); ++i) {
          if (free_slabs[i] && free_slabs[i]->size() >= aligned_bytes &&
              free_slabs[i]->size() < best_size) {
            best_idx = i;
            best_size = free_slabs[i]->size();
          }
        }
        if (best_idx < free_slabs.size()) {
          auto slab = std::move(free_slabs[best_idx]);
          free_slabs.erase(free_slabs.begin() + std::ptrdiff_t(best_idx));
          return slab;
        }
        // Orphan-free resize: pop and destroy one undersized slab before
        // allocating a larger replacement so monotonic multi-turn growth leaves
        // zero dead undersized slabs in free_slabs.
        if (!free_slabs.empty()) {
          auto smallest = std::min_element(
              free_slabs.begin(), free_slabs.end(),
              [](const auto &a, const auto &b) {
                return (a ? a->size() : 0ULL) < (b ? b->size() : 0ULL);
              });
          evict_undersized = std::move(*smallest);
          free_slabs.erase(smallest);
        }
      }
      evict_undersized.reset();
      return std::make_shared<MetalBuffer>(ctx.make_buffer(aligned_bytes));
    }

    void recycle(std::shared_ptr<MetalBuffer> slab) noexcept {
      if (!slab)
        return;
      std::scoped_lock lock(mutex);
      if (free_slabs.size() >= max_pooled_slabs) {
        auto smallest = std::min_element(
            free_slabs.begin(), free_slabs.end(),
            [](const auto &a, const auto &b) {
              return (a ? a->size() : 0ULL) < (b ? b->size() : 0ULL);
            });
        if (smallest != free_slabs.end() && *smallest &&
            (*smallest)->size() < slab->size()) {
          *smallest = std::move(slab);
        }
        return;
      }
      free_slabs.push_back(std::move(slab));
    }

    void trim(std::size_t keep_max = 0U) noexcept {
      std::scoped_lock lock(mutex);
      while (free_slabs.size() > keep_max)
        free_slabs.pop_back();
    }
  };

  std::shared_ptr<UmaCheckpointPool> uma_pool_shared() const {
    if (!uma_ckpt_pool_) {
      if (shared_primary_ != nullptr) {
        uma_ckpt_pool_ = shared_primary_->impl_->uma_pool_shared();
      } else {
        uma_ckpt_pool_ = std::make_shared<UmaCheckpointPool>();
      }
    }
    return uma_ckpt_pool_;
  }

  mutable std::shared_ptr<UmaCheckpointPool> uma_ckpt_pool_;
  mutable std::optional<MetalCommandPlan> uma_gdn_copy_plan_;
  mutable std::map<U64, MetalCommandPlan> uma_feat_copy_plans_;
  mutable std::optional<MetalParameterTable> uma_feat_copy_params_;
  mutable std::set<U64> uma_live_epochs_;
};

struct QwenAsyncArStep::Impl {
  QwenForward *owner = nullptr;
  QwenForward::Impl *owner_impl = nullptr;
  Profile *profile = nullptr;
  U32 pipe_slot = 0U;
  U32 step_position = 0U;
  U64 generation = 0U;
  GpuSamplingOptions sampling{};
  MetalPendingSubmission pending;
  bool active = false;
  ~Impl() {
    if (active && owner_impl != nullptr) {
      try {
        (void)owner_impl->context.wait_submission(std::move(pending));
        if (generation == owner_impl->async_generation_ &&
            owner_impl->ar_in_flight_count_ > 0U) {
          --owner_impl->ar_in_flight_count_;
          ++owner_impl->next_position;
        }
      } catch (...) {
      }
      active = false;
    }
  }
};
QwenAsyncArStep::QwenAsyncArStep() noexcept = default;
QwenAsyncArStep::QwenAsyncArStep(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
QwenAsyncArStep::QwenAsyncArStep(QwenAsyncArStep &&) noexcept = default;
QwenAsyncArStep &QwenAsyncArStep::operator=(QwenAsyncArStep &&) noexcept =
    default;
QwenAsyncArStep::~QwenAsyncArStep() = default;
bool QwenAsyncArStep::valid() const noexcept {
  return impl_ != nullptr && impl_->active;
}

struct QwenAsyncPrefillStep::Impl {
  QwenForward *owner = nullptr;
  QwenForward::Impl *owner_impl = nullptr;
  Profile *profile = nullptr;
  U32 count = 0U;
  U64 generation = 0U;
  MetalPendingSubmission pending;
  bool active = false;
  ~Impl() {
    if (active && owner_impl != nullptr) {
      try {
        (void)owner_impl->context.wait_submission(std::move(pending));
        if (generation == owner_impl->async_generation_ &&
            owner_impl->prefill_in_flight_count_ > 0U) {
          --owner_impl->prefill_in_flight_count_;
        }
      } catch (...) {
      }
      active = false;
    }
  }
};
QwenAsyncPrefillStep::QwenAsyncPrefillStep() noexcept = default;
QwenAsyncPrefillStep::QwenAsyncPrefillStep(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
QwenAsyncPrefillStep::QwenAsyncPrefillStep(QwenAsyncPrefillStep &&) noexcept =
    default;
QwenAsyncPrefillStep &
QwenAsyncPrefillStep::operator=(QwenAsyncPrefillStep &&) noexcept = default;
QwenAsyncPrefillStep::~QwenAsyncPrefillStep() = default;
bool QwenAsyncPrefillStep::valid() const noexcept {
  return impl_ != nullptr && impl_->active;
}
U32 QwenAsyncPrefillStep::width() const noexcept {
  return impl_ ? impl_->count : 0U;
}

struct QwenPreparedStep::Impl {
  QwenForward *owner = nullptr;
  Profile *profile = nullptr;
  KvPagePool *page_pool = nullptr;
  U32 count = 0;
  bool verify = false, active = false;
  ~Impl() {
    if (active && page_pool)
      try {
        page_pool->abort();
      } catch (...) {
      }
  }
};
QwenPreparedStep::QwenPreparedStep(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
QwenPreparedStep::QwenPreparedStep(QwenPreparedStep &&) noexcept = default;
QwenPreparedStep &QwenPreparedStep::operator=(QwenPreparedStep &&) noexcept =
    default;
QwenPreparedStep::~QwenPreparedStep() = default;
MetalCommandInvocation QwenPreparedStep::invocation() const {
  if (!impl_ || !impl_->active)
    throw std::logic_error("Qwen batch step is not prepared");
  return {.plan = &impl_->profile->plan,
          .bindings = impl_->profile->bindings,
          .parameters = &impl_->profile->parameters};
}
MetalIcbProfile *QwenPreparedStep::icb_profile() const {
  if (!impl_ || !impl_->active || !impl_->profile || !impl_->profile->icb)
    return nullptr;
  return &(*impl_->profile->icb);
}
U32 QwenPreparedStep::width() const { return impl_->count; }
bool QwenPreparedStep::verifying() const { return impl_->verify; }

QwenForward::QwenForward(
    const MetalContext &context, const MaterializedModel &weights,
    const QwenLayerCatalog &layers, QwenModelConfig config, U32 max_context,
    const std::filesystem::path &libraries, std::span<const U32> feature_layers,
    bool inspect_routes, bool speculative, bool enable_m4_short_routes,
    std::optional<QwenLinearRouteOverride> diagnostic_override,
    bool enable_attention_split, bool enable_batch,
    std::optional<QwenMoeRouteOverride> moe_override,
    LinearNumericPolicy linear_policy, SamplingContract sampling_contract,
    omlx::m5::AttentionNumericPolicy attention_policy,
    bool inspect_tensors_workspace, std::string_view kv_dtype,
    QwenForward *shared_primary, bool enable_flash_nax_prefill,
    U32 kv_pool_pages,
    std::shared_ptr<const QwenMergedLoraOverrides> merged_lora,
    std::shared_ptr<const QwenBypassLoraRuntime> bypass_lora,
    QwenPrefillOptions prefill_options)
    : impl_(std::make_unique<Impl>(
          context, weights, layers, std::move(config), max_context, libraries,
          feature_layers, inspect_routes, speculative, enable_m4_short_routes,
          std::move(diagnostic_override), enable_attention_split, enable_batch,
          std::move(moe_override), linear_policy, sampling_contract,
          attention_policy, inspect_tensors_workspace, kv_dtype,
          shared_primary, enable_flash_nax_prefill, kv_pool_pages,
          std::move(merged_lora), std::move(bypass_lora),
          std::move(prefill_options))) {}
QwenForward::~QwenForward() = default;
QwenForward::QwenForward(QwenForward &&) noexcept = default;
QwenForward &QwenForward::operator=(QwenForward &&) noexcept = default;
void QwenForward::reset() { impl_->reset(); }
U32 QwenForward::position() const { return impl_->next_position; }
U32 QwenForward::ar_virtual_position() const noexcept {
  return impl_->next_position + impl_->ar_in_flight_count_;
}
bool QwenForward::can_pipeline_ar(
    const GpuSamplingOptions &sampling) const noexcept {
  const auto &p = *impl_;
  if (detail::forward::env_flag_enabled("OMLX_M5_DISABLE_ASYNC_AR"))
    return false;
  if (p.speculative_enabled || p.inspect_routes_enabled ||
      p.inspect_tensors_workspace || !p.feature_layers.empty() ||
      p.pending_verify != 0U || p.prefill_in_flight_count_ != 0U ||
      p.ar_pipe_[0].profiles.empty() ||
      p.ar_pipe_[1].profiles.empty() ||
      !p.ar_pipe_[0].profiles.front().icb.has_value() ||
      !p.ar_pipe_[1].profiles.front().icb.has_value())
    return false;
  if (sampling.contract != p.sampling_contract || sampling.top_k > 64U ||
      (sampling.temperature > 0.0F && sampling.top_k == 0U) ||
      sampling.temperature < 0.0F || !std::isfinite(sampling.temperature) ||
      !std::isfinite(sampling.top_p) || sampling.top_p < 0.0F ||
      sampling.top_p > 1.0F)
    return false;
  // Require GPU-authoritative sampling so device-fed token_ids[0] is 0-ULP exact:
  // LegacyRejectionV1 is 100% GPU-authoritative at all temperatures;
  // PositionCoupledV1 is 100% GPU-authoritative when greedy (temp <= 0 or top_k <= 1).
  if (sampling.contract == SamplingContract::PositionCoupledV1 &&
      sampling.temperature > 0.0F && sampling.top_k > 1U)
    return false;
  return true;
}

QwenAsyncArStep QwenForward::enqueue_ar_step(const GpuSamplingOptions &sampling,
                                             bool seed_first,
                                             U32 first_token) {
  auto &p = *impl_;
  p.execution_started = true;
  if (!can_pipeline_ar(sampling))
    throw std::invalid_argument("Qwen async AR step ineligible");
  if (p.ar_in_flight_count_ >= 2U)
    throw std::logic_error("Qwen async AR 2-slot pipeline saturated");
  if (seed_first && p.ar_in_flight_count_ != 0U)
    throw std::logic_error("Qwen async AR seed_first with active in-flight step");
  if (seed_first && first_token >= p.v)
    throw std::invalid_argument("Qwen async AR first_token outside vocabulary");

  const U32 vpos = p.next_position + p.ar_in_flight_count_;
  if (vpos >= p.max_context)
    throw std::out_of_range("Qwen async AR context exhausted");

  const U32 slot_i = p.ar_pipe_cursor_ & 1U;
  auto &ap = p.ar_pipe_[slot_i];
  auto found = std::find_if(
      ap.profiles.begin(), ap.profiles.end(), [&](const Profile &prof) {
        return (prof.context_envelope == 0U) ||
               (vpos + 1U <= prof.context_envelope);
      });
  if (found == ap.profiles.end() || !found->icb.has_value())
    throw std::out_of_range("Qwen async AR profile envelope unavailable");
  auto &profile = *found;

  if (seed_first) {
    const_cast<MetalBuffer *>(p.token_ids.buffer)
        ->write(std::as_bytes(std::span(&first_token, 1)));
  }
  for (const auto &patch : profile.patches) {
    const U32 val = patch.position ? vpos : patch.multiplier;
    std::memcpy(profile.parameter_data.data() + patch.offset + patch.word * 4,
                &val, 4);
  }
  if (ap.attn_pos.buffer) {
    const_cast<MetalBuffer *>(ap.attn_pos.buffer)
        ->write(std::as_bytes(std::span(&vpos, 1)));
  }
  if (p.sampling_contract == SamplingContract::PositionCoupledV1) {
    const PositionSamplingRow row{.seed = sampling.seed,
                                  .position = vpos + 1U,
                                  .top_k = sampling.top_k,
                                  .temperature = sampling.temperature,
                                  .top_p = sampling.top_p};
    std::memcpy(profile.parameter_data.data() + profile.sampling_offset, &row,
                sizeof(row));
  } else {
    const SampleParams params{p.v,          sampling.top_k, sampling.temperature,
                              sampling.top_p, sampling.seed,  sampling.stream,
                              vpos};
    std::memcpy(profile.parameter_data.data() + profile.sampling_offset,
                &params, sizeof(params));
  }
  profile.parameters.write("step", profile.parameter_data);

  auto *page_pool = p.page_pool.get();
  if (page_pool && page_pool->tokens() != vpos)
    throw std::logic_error("Qwen async AR KV frontier disagrees with vpos");
  const auto copies =
      page_pool ? page_pool->begin_append(1U) : std::vector<KvPageCopy>{};
  struct PageGuard {
    KvPagePool *pool;
    bool active;
    ~PageGuard() {
      if (active)
        try {
          pool->abort();
        } catch (...) {
        }
    }
  } page_guard{page_pool, page_pool != nullptr};
  p.execute_cow_copies(copies);

  auto handle = std::make_unique<QwenAsyncArStep::Impl>();
  handle->owner = this;
  handle->owner_impl = &p;
  handle->profile = &profile;
  handle->pipe_slot = slot_i;
  handle->step_position = vpos;
  handle->generation = p.async_generation_;
  handle->sampling = sampling;
  handle->pending = p.context.replay_icb_async(*profile.icb);
  if (page_pool) {
    page_pool->commit(1U);
    page_guard.active = false;
  }
  handle->active = true;
  ++p.ar_in_flight_count_;
  p.ar_pipe_cursor_ = (slot_i + 1U) & 1U;
  p.last_trace_tokens = 0U;
  return QwenAsyncArStep(std::move(handle));
}

QwenForwardResult QwenForward::finish_ar_step(QwenAsyncArStep &&step) {
  auto &p = *impl_;
  if (!step.impl_ || !step.impl_->active || step.impl_->owner != this)
    throw std::invalid_argument("Qwen async AR step invalid or wrong owner");
  auto &st = *step.impl_;
  st.active = false;
  MetalSubmission sub = p.context.wait_submission(std::move(st.pending));
  if (st.generation != p.async_generation_)
    throw std::logic_error(
        "Qwen async AR step invalidated by reset or restore");
  if (p.ar_in_flight_count_ == 0U)
    throw std::logic_error("Qwen async AR finish underflow");
  --p.ar_in_flight_count_;
  ++p.next_position;
  p.last_logits_row = 0U;
  p.has_last_logits = (p.ar_in_flight_count_ == 0U);

  const auto &ap = p.ar_pipe_[st.pipe_slot];
  const auto tok_bytes = ap.sampled.buffer->read(
      0, p.sampling_contract == SamplingContract::PositionCoupledV1 ? 8 : 4);
  QwenForwardResult result{
      .tokens = 1U,
      .executed_layers = U32(p.blocks.size()),
      .host_read_bytes = tok_bytes.size(),
      .short_matrix_calls = st.profile->short_matrix_calls,
      .short_lane_calls = st.profile->short_lane_calls,
      .submission = std::move(sub),
  };
  std::memcpy(&result.token, tok_bytes.data(), 4);
  if (p.sampling_contract == SamplingContract::PositionCoupledV1) {
    U32 flag = 0;
    std::memcpy(&flag, tok_bytes.data() + 4, 4);
    if (flag & 2U)
      throw std::runtime_error("nonfinite position target logits");
  } else if (result.token >= p.v) {
    throw std::runtime_error("nonfinite or invalid legacy sampled token");
  }
  return result;
}

void QwenForward::drain_ar_step(QwenAsyncArStep &&step) {
  auto &p = *impl_;
  if (!step.impl_ || !step.impl_->active || step.impl_->owner != this)
    return;
  auto &st = *step.impl_;
  st.active = false;
  (void)p.context.wait_submission(std::move(st.pending));
  if (st.generation != p.async_generation_)
    return;
  if (p.ar_in_flight_count_ > 0U) {
    --p.ar_in_flight_count_;
    ++p.next_position;
  }
  p.has_last_logits = false;
}

bool QwenForward::can_pipeline_prefill(std::size_t count) const noexcept {
  const auto &p = *impl_;
  if (detail::forward::env_flag_enabled("OMLX_M5_DISABLE_ASYNC_PREFILL"))
    return false;
  if (p.shared_primary_ != nullptr || p.inspect_routes_enabled ||
      p.inspect_tensors_workspace || p.pending_verify != 0U ||
      p.ar_in_flight_count_ != 0U || p.prefill_in_flight_count_ >= 2U)
    return false;
  if (count < 64U || count > p.workspace_rows_ ||
      count > p.max_context - p.next_position)
    return false;
  if ((p.next_position % 64U) != 0U)
    return false;
  if (p.can_run_ane_chunk(U32(count), false, false, false))
    return false;
  const auto route =
      p.select_chunk_attention_route(U32(count), p.next_position);
  // WP4: Unlock async double-buffered prefill on Dense Sage2 prefix chunks;
  // only synchronous MInference (CPU active_counts readback) and ExactNax
  // long-ctx split tails stay off the async pipe.
  if (route == QwenForward::Impl::ChunkAttentionRoute::Sage2MInference)
    return false;
  if (p.needs_long_ctx_prefill_profile(U32(count), p.next_position)) {
    if (route != QwenForward::Impl::ChunkAttentionRoute::Sage2Dense)
      return false;
    if (p.context.warmup_sealed()) {
      const std::uint8_t slot_id = std::uint8_t(p.prefill_pipe_cursor_ & 1U);
      const U32 end_pos = p.next_position + U32(count);
      bool found_icb = false;
      for (const auto &[k, prof] : p.long_ctx_profiles_) {
        if (k.attn_route == route && k.trunk_only && k.uses_pipe_slot0 &&
            k.pipe_slot == slot_id && k.capacity >= count &&
            k.context_envelope >= end_pos && prof.icb.has_value()) {
          found_icb = true;
          break;
        }
      }
      if (!found_icb)
        return false;
    }
  }
  if (p.prefill_pipe_[0].profiles.empty() ||
      p.prefill_pipe_[1].profiles.empty() ||
      !p.prefill_pipe_[0].profiles.front().icb.has_value() ||
      !p.prefill_pipe_[1].profiles.front().icb.has_value())
    return false;
  return true;
}

QwenAsyncPrefillStep
QwenForward::enqueue_prefill_step(std::span<const U32> tokens) {
  auto &p = *impl_;
  p.execution_started = true;
  if (!can_pipeline_prefill(tokens.size()))
    throw std::invalid_argument("Qwen async prefill step ineligible");
  if (std::any_of(tokens.begin(), tokens.end(),
                  [&](U32 id) { return id >= p.v; }))
    throw std::invalid_argument("Qwen async prefill token outside vocabulary");

  const U32 slot_i = p.prefill_pipe_cursor_ & 1U;
  auto &pp = p.prefill_pipe_[slot_i];
  Profile *selected_ptr = nullptr;
  if (p.needs_long_ctx_prefill_profile(U32(tokens.size()), p.next_position)) {
    selected_ptr = p.select_long_ctx_prefill_profile(
        U32(tokens.size()), p.next_position, /*trunk_only=*/true,
        /*use_pipe_slot0=*/true, slot_i);
  } else {
    auto found = std::find_if(pp.profiles.begin(), pp.profiles.end(),
                              [&](const Profile &prof) {
                                return tokens.size() <= prof.capacity;
                              });
    if (found != pp.profiles.end()) {
      selected_ptr = &(*found);
    }
  }
  if (selected_ptr == nullptr || !selected_ptr->icb.has_value())
    throw std::out_of_range("Qwen async prefill profile capacity unavailable");
  auto &profile = *selected_ptr;

  const U32 start_pos = p.next_position;
  const U32 count = U32(tokens.size());
  std::fill_n(p.token_staging.begin(), profile.capacity, 0U);
  std::copy(tokens.begin(), tokens.end(), p.token_staging.begin());
  const_cast<MetalBuffer *>(pp.token_ids.buffer)
      ->write(std::as_bytes(std::span(p.token_staging).first(profile.capacity)));
  for (const auto &patch : profile.patches) {
    const U32 val =
        patch.position ? start_pos : count * patch.multiplier;
    std::memcpy(profile.parameter_data.data() + patch.offset + patch.word * 4,
                &val, 4);
  }
  if (pp.attn_pos.buffer) {
    std::array<U32, 4096> positions{};
    for (U32 i = 0; i < count; ++i)
      positions[i] = start_pos + i;
    const_cast<MetalBuffer *>(pp.attn_pos.buffer)
        ->write(std::as_bytes(std::span(positions).first(count)));
  }
  profile.parameters.write("step", profile.parameter_data);

  auto *page_pool = p.page_pool.get();
  if (page_pool && page_pool->tokens() != start_pos)
    throw std::logic_error("Qwen async prefill KV frontier mismatch");
  const auto copies =
      page_pool ? page_pool->begin_append(count) : std::vector<KvPageCopy>{};
  struct PageGuard {
    KvPagePool *pool;
    bool active;
    ~PageGuard() {
      if (active)
        try {
          pool->abort();
        } catch (...) {
        }
    }
  } page_guard{page_pool, page_pool != nullptr};
  p.execute_cow_copies(copies);

  auto handle = std::make_unique<QwenAsyncPrefillStep::Impl>();
  handle->owner = this;
  handle->owner_impl = &p;
  handle->profile = &profile;
  handle->count = count;
  handle->generation = p.async_generation_;
  handle->pending = p.context.replay_icb_async(*profile.icb);
  if (page_pool) {
    page_pool->commit(count);
    page_guard.active = false;
  }
  handle->active = true;
  p.next_position += count;
  ++p.prefill_in_flight_count_;
  p.prefill_pipe_cursor_ = (slot_i + 1U) & 1U;
  p.last_trace_tokens = 0U;
  p.has_last_logits = false;
  return QwenAsyncPrefillStep(std::move(handle));
}

QwenForwardResult
QwenForward::finish_prefill_step(QwenAsyncPrefillStep &&step) {
  auto &p = *impl_;
  if (!step.impl_ || !step.impl_->active || step.impl_->owner != this)
    throw std::invalid_argument(
        "Qwen async prefill step invalid or wrong owner");
  auto &st = *step.impl_;
  st.active = false;
  MetalSubmission sub = p.context.wait_submission(std::move(st.pending));
  if (st.generation != p.async_generation_)
    throw std::logic_error(
        "Qwen async prefill step invalidated by reset or restore");
  if (p.prefill_in_flight_count_ == 0U)
    throw std::logic_error("Qwen async prefill finish underflow");
  --p.prefill_in_flight_count_;
  QwenForwardResult res{
      .token = 0U,
      .tokens = st.count,
      .executed_layers = U32(p.blocks.size()),
      .host_read_bytes = 0U,
      .short_matrix_calls = st.profile->short_matrix_calls,
      .short_lane_calls = st.profile->short_lane_calls,
      .submission = std::move(sub),
  };
  p.record_prefill_telemetry(st.count, p.next_position - st.count,
                             st.profile->used_w4a8, st.profile->used_sage2,
                             st.profile->used_minference, st.profile->capacity,
                             st.profile->minf_context_envelope,
                             /*used_ane=*/false, &res.prefill_delta,
                             std::nullopt, st.profile->used_w4a4);
  return res;
}

void QwenForward::drain_prefill_step(QwenAsyncPrefillStep &&step) {
  auto &p = *impl_;
  if (!step.impl_ || !step.impl_->active || step.impl_->owner != this)
    return;
  auto &st = *step.impl_;
  st.active = false;
  (void)p.context.wait_submission(std::move(st.pending));
  if (st.generation != p.async_generation_)
    return;
  if (p.prefill_in_flight_count_ > 0U) {
    --p.prefill_in_flight_count_;
  }
}

QwenForwardResult QwenForward::run_trunk(std::span<const U32> tokens,
                                         bool use_icb) {
  auto &p = *impl_;
  p.require_no_async_in_flight("run_trunk");
  if (p.shared_primary_ != nullptr || p.inspect_routes_enabled ||
      p.inspect_tensors_workspace || p.prefill_in_flight_count_ != 0U ||
      tokens.size() <= 8U || p.prefill_pipe_[0].profiles.empty()) {
    GpuSamplingOptions fallback_samp{.temperature = 0.0F,
                                     .top_k = 1U,
                                     .top_p = 1.0F,
                                     .contract = p.sampling_contract};
    return run(tokens, fallback_samp, false, use_icb);
  }
  p.execution_started = true;
  if (p.pending_verify)
    throw std::logic_error("Qwen verification must commit or abort first");
  if (tokens.empty() || tokens.size() > p.workspace_rows_ ||
      tokens.size() > p.max_context - p.next_position ||
      std::any_of(tokens.begin(), tokens.end(),
                  [&](U32 id) { return id >= p.v; }))
    throw std::invalid_argument("Qwen trunk prefill token/context invalid");

  const U32 count = U32(tokens.size());
  if (p.can_run_ane_chunk(count, false, false, false)) {
    GpuSamplingOptions unused_samp{.contract = p.sampling_contract};
    return p.execute_ane_chunk(tokens, unused_samp, /*trunk_only=*/true);
  }
  if (p.ane_chunk_tokens_ > 0U && count > p.ane_chunk_tokens_ &&
      (count % p.ane_chunk_tokens_) == 0U &&
      p.can_run_ane_chunk(p.ane_chunk_tokens_, false, false, false)) {
    GpuSamplingOptions unused_samp{.contract = p.sampling_contract};
    QwenForwardResult total_res{};
    total_res.tokens = count;
    total_res.executed_layers = U32(p.blocks.size());
    for (U32 off = 0U; off < count; off += p.ane_chunk_tokens_) {
      auto sub_res = p.execute_ane_chunk(
          tokens.subspan(off, p.ane_chunk_tokens_), unused_samp,
          /*trunk_only=*/true);
      total_res.submission.timing.gpu_ms += sub_res.submission.timing.gpu_ms;
      total_res.submission.timing.wall_ms += sub_res.submission.timing.wall_ms;
      total_res.submission.dispatch_count += sub_res.submission.dispatch_count;
      total_res.submission.shared_event_value =
          sub_res.submission.shared_event_value;
      total_res.prefill_delta.prefill_w4a8_tokens +=
          sub_res.prefill_delta.prefill_w4a8_tokens;
      total_res.prefill_delta.prefill_w4a4_tokens +=
          sub_res.prefill_delta.prefill_w4a4_tokens;
      total_res.prefill_delta.prefill_sage2_tokens +=
          sub_res.prefill_delta.prefill_sage2_tokens;
      total_res.prefill_delta.prefill_minference_blocks_skipped +=
          sub_res.prefill_delta.prefill_minference_blocks_skipped;
      total_res.prefill_delta.prefill_minference_blocks_total +=
          sub_res.prefill_delta.prefill_minference_blocks_total;
      total_res.prefill_delta.prefill_ane_active =
          sub_res.prefill_delta.prefill_ane_active;
      total_res.prefill_delta.prefill_ane_mode_effective =
          sub_res.prefill_delta.prefill_ane_mode_effective;
      total_res.prefill_delta.prefill_ane_inactive_reason =
          sub_res.prefill_delta.prefill_ane_inactive_reason;
      total_res.prefill_delta.prefill_ane_split_alpha =
          sub_res.prefill_delta.prefill_ane_split_alpha;
      total_res.prefill_delta.prefill_ane_offload_tokens +=
          sub_res.prefill_delta.prefill_ane_offload_tokens;
      total_res.prefill_delta.prefill_ane_fallback_tokens +=
          sub_res.prefill_delta.prefill_ane_fallback_tokens;
    }
    return total_res;
  }

  auto &pp = p.prefill_pipe_[0];
  Profile *lc_prof =
      p.select_long_ctx_prefill_profile(count, p.next_position, true, true);
  Profile *selected = lc_prof;
  if (selected == nullptr) {
    auto found = std::find_if(pp.profiles.begin(), pp.profiles.end(),
                              [&](const Profile &prof) {
                                return tokens.size() <= prof.capacity;
                              });
    if (found == pp.profiles.end())
      throw std::out_of_range("Qwen trunk prefill capacity unavailable");
    selected = &(*found);
  }
  auto &profile = *selected;

  const U32 start_pos = p.next_position;
  p.last_trace_tokens = 0U;
  std::fill_n(p.token_staging.begin(), profile.capacity, 0U);
  std::copy(tokens.begin(), tokens.end(), p.token_staging.begin());
  const_cast<MetalBuffer *>(pp.token_ids.buffer)
      ->write(std::as_bytes(std::span(p.token_staging).first(profile.capacity)));
  for (const auto &patch : profile.patches) {
    const U32 val =
        patch.position ? p.next_position : count * patch.multiplier;
    std::memcpy(profile.parameter_data.data() + patch.offset + patch.word * 4,
                &val, 4);
  }
  if (pp.attn_pos.buffer) {
    std::array<U32, 4096> positions{};
    for (U32 i = 0; i < count; ++i)
      positions[i] = p.next_position + i;
    const_cast<MetalBuffer *>(pp.attn_pos.buffer)
        ->write(std::as_bytes(std::span(positions).first(count)));
  }
  profile.parameters.write("step", profile.parameter_data);

  auto *page_pool = p.page_pool.get();
  if (page_pool && page_pool->tokens() != p.next_position)
    throw std::logic_error("Qwen KV page frontier disagrees with model");
  const auto page_copies =
      page_pool ? page_pool->begin_append(count) : std::vector<KvPageCopy>{};
  struct PageGuard {
    KvPagePool *pool;
    bool active;
    ~PageGuard() {
      if (active)
        try {
          pool->abort();
        } catch (...) {
        }
    }
  } page_guard{page_pool, page_pool != nullptr};
  p.execute_cow_copies(page_copies);

  MetalSubmission sub;
  if (use_icb && profile.icb) {
    sub = p.context.replay_icb(*profile.icb);
  } else {
    const std::array invocation = {
        MetalCommandInvocation{.plan = &profile.plan,
                               .bindings = profile.bindings,
                               .parameters = &profile.parameters}};
    sub = p.context.execute_batch(invocation);
  }
  if (page_pool)
    page_pool->commit(count);
  page_guard.active = false;
  p.next_position += count;
  p.has_last_logits = false;
  QwenForwardResult res{
      .token = 0U,
      .tokens = count,
      .executed_layers = U32(p.blocks.size()),
      .host_read_bytes = 0U,
      .short_matrix_calls = profile.short_matrix_calls,
      .short_lane_calls = profile.short_lane_calls,
      .submission = std::move(sub),
  };
  p.record_prefill_telemetry(count, start_pos, profile.used_w4a8,
                             profile.used_sage2, profile.used_minference,
                             profile.capacity, profile.minf_context_envelope,
                             /*used_ane=*/false, &res.prefill_delta,
                             std::nullopt, profile.used_w4a4);
  return res;
}

QwenForwardResult QwenForward::run_blend_prefill(
    std::span<const U32> prompt, U32 prefix_tokens,
    detail::ChunkBlendCache &blend_cache, U64 model_exec_hash,
    const GpuSamplingOptions &sampling, bool inspect_tensors,
    const detail::BlendSegmentConfig *seg_cfg) {
  return impl_->run_blend_prefill_impl(*this, prompt, prefix_tokens, blend_cache,
                                       model_exec_hash, sampling,
                                       inspect_tensors, seg_cfg);
}

QwenPrefillTelemetry QwenForward::prefill_telemetry() const noexcept {
  return impl_->prefill_telemetry_;
}

bool QwenForward::ane_offload_active() const noexcept {
  return impl_->ane_active_;
}

PrefillAneMode QwenForward::ane_mode_effective() const noexcept {
  return impl_->ane_mode_effective_;
}

std::string_view QwenForward::ane_inactive_reason() const noexcept {
  return impl_->ane_inactive_reason_;
}

double QwenForward::ane_split_alpha() const noexcept {
  return impl_->ane_split_alpha_;
}

U32 QwenForward::ane_chunk_tokens() const noexcept {
  return impl_->ane_chunk_tokens_;
}

bool QwenForward::cache_blend_eligible() const noexcept {
  return impl_->cache_blend_eligible();
}
const QwenForwardMemoryPlan &QwenForward::memory_plan() const {
  return impl_->allocation_plan;
}

namespace {
QwenProfileDescription describe_profile(const Profile &profile, bool verify) {
  QwenProfileDescription result{.capacity = profile.capacity,
                                .verify = verify,
                                .dispatch_count = profile.plan.dispatch_count(),
                                .barrier_count = profile.plan.barrier_count()};
  result.split_attention = profile.split_attention;
  result.split_count = profile.split_count;
  result.context_envelope = profile.context_envelope;
  result.split_workspace_bytes = profile.split_workspace_bytes;
  result.attention_kernel = profile.attention_kernel;
  result.attention_merge_kernel = profile.attention_merge_kernel;
  result.attention_grid = profile.attention_grid;
  result.attention_merge_grid = profile.attention_merge_grid;
  result.numeric_contract_id = profile.numeric_contract_id;
  result.sampling_contract_id = profile.sampling_contract_id;
  result.attention_numeric_contract_id = profile.attention_numeric_contract_id;
  result.attention_row_exact_eligible = profile.attention_row_exact_eligible;
  result.gdn_numeric_contract_id = profile.gdn_numeric_contract_id;
  result.gdn_row_exact_eligible = profile.gdn_row_exact_eligible;
  for (const auto &stage : profile.stages)
    result.stages.push_back({stage.name, stage.length});
  for (const auto &route : profile.weight_routes)
    result.weights.push_back({route.stage, route.object, route.format,
                              route.library, route.function, route.family,
                              route.source, route.fallback_reason, route.n,
                              route.k, route.t, route.grid, route.threads,
                              route.numeric_contract_id, route.row_exact_group,
                              route.row_exact_eligible});
  return result;
}
} // namespace

QwenProfileDescription QwenForward::inspect_profile(U32 tokens,
                                                    bool verify) const {
  const auto &profiles =
      impl_->split_attention_enabled && tokens <= 8
          ? (verify ? impl_->split_verify_profiles : impl_->split_profiles)
          : (verify ? impl_->verify_profiles : impl_->profiles);
  const auto found = std::find_if(
      profiles.begin(), profiles.end(), [&](const Profile &profile) {
        return tokens > 0 && tokens <= profile.capacity &&
               (!profile.split_attention ||
                impl_->next_position + tokens <= profile.context_envelope);
      });
  if (found == profiles.end())
    throw std::out_of_range("Qwen diagnostic profile unavailable");
  return describe_profile(*found, verify);
}

QwenProfileDescription
QwenForward::inspect_split_profile(U32 tokens, U32 context, bool verify) const {
  if (!impl_->split_attention_enabled || !tokens || context < tokens ||
      context > impl_->max_context)
    throw std::invalid_argument("Qwen split profile inspection unavailable");
  const auto &profiles =
      verify ? impl_->split_verify_profiles : impl_->split_profiles;
  const auto found = std::find_if(profiles.begin(), profiles.end(),
                                  [&](const Profile &profile) {
                                    return tokens <= profile.capacity &&
                                           context <= profile.context_envelope;
                                  });
  if (found == profiles.end())
    throw std::out_of_range("Qwen split context envelope unavailable");
  return describe_profile(*found, verify);
}

std::vector<std::byte> QwenForward::inspect_state(U64 max_bytes) const {
  U64 total = 0;
  for (const auto *state : impl_->states) {
    if (state->size() > max_bytes - total)
      throw std::length_error("Qwen diagnostic state exceeds readback limit");
    total += state->size();
  }
  std::vector<std::byte> result;
  result.reserve(total);
  for (const auto *state : impl_->states) {
    const auto bytes = state->read(0, state->size());
    result.insert(result.end(), bytes.begin(), bytes.end());
  }
  return result;
}

QwenPreparedStep QwenForward::prepare_batch(std::span<const U32> tokens,
                                            const GpuSamplingOptions &sampling,
                                            bool verify) {
  auto &p = *impl_;
  p.require_no_async_in_flight("prepare_batch");
  p.execution_started = true;
  if (sampling.contract != p.sampling_contract || !p.batch_enabled ||
      p.pending_verify || tokens.empty() || tokens.size() > 8 ||
      tokens.size() > p.max_context - p.next_position ||
      std::any_of(tokens.begin(), tokens.end(),
                  [&](U32 id) { return id >= p.v; }) ||
      sampling.top_k > 64 ||
      (sampling.temperature > 0 && sampling.top_k == 0) ||
      sampling.temperature < 0 || !std::isfinite(sampling.temperature) ||
      !std::isfinite(sampling.top_p) || sampling.top_p < 0 ||
      sampling.top_p > 1)
    throw std::invalid_argument("Qwen batch token/context/sampling invalid");
  auto &profiles = p.split_attention_enabled
                       ? (verify ? p.split_verify_profiles
                                 : p.split_batch_profiles)
                       : (verify ? p.verify_profiles : p.batch_profiles);
  if (p.attention_positions.buffer) {
    std::array<U32, 4096> positions{};
    for (U32 i = 0; i < tokens.size(); ++i)
      positions[i] = p.next_position + i;
    const_cast<MetalBuffer *>(p.attention_positions.buffer)
        ->write(std::as_bytes(std::span(positions).first(tokens.size())));
  }
  const auto found = std::find_if(
      profiles.begin(), profiles.end(), [&](const Profile &profile) {
        return tokens.size() <= profile.capacity &&
               (!profile.split_attention ||
                p.next_position + tokens.size() <= profile.context_envelope);
      });
  if (found == profiles.end())
    throw std::out_of_range("Qwen batch profile envelope unavailable");
  p.last_trace_tokens = 0;
  auto &profile = *found;
  std::fill_n(p.token_staging.begin(), profile.capacity, 0);
  std::copy(tokens.begin(), tokens.end(), p.token_staging.begin());
  const_cast<MetalBuffer *>(p.token_ids.buffer)
      ->write(std::as_bytes(std::span(p.token_staging).first(profile.capacity)));
  for (const auto &patch : profile.patches) {
    const U32 value = patch.position ? p.next_position
                                     : U32(tokens.size()) * patch.multiplier;
    std::memcpy(profile.parameter_data.data() + patch.offset + patch.word * 4,
                &value, 4);
  }
  profile.parameters.write("step", profile.parameter_data);
  auto prepared = std::make_unique<QwenPreparedStep::Impl>();
  prepared->owner = this;
  prepared->profile = &profile;
  prepared->page_pool = p.page_pool.get();
  prepared->count = U32(tokens.size());
  prepared->verify = verify;
  if (p.page_pool && p.page_pool->tokens() != p.next_position)
    throw std::logic_error("Qwen batch KV frontier differs from model");
  const auto copies = p.page_pool ? p.page_pool->begin_append(prepared->count)
                                  : std::vector<KvPageCopy>{};
  prepared->active = p.page_pool != nullptr;
  p.execute_cow_copies(copies);
  if (!p.page_pool)
    prepared->active = true;
  return QwenPreparedStep(std::move(prepared));
}

QwenForwardResult QwenForward::finish_batch(QwenPreparedStep &&step,
                                            U32 sampled_token) {
  auto &p = *impl_;
  if (!step.impl_ || !step.impl_->active || step.impl_->owner != this)
    throw std::invalid_argument("Qwen batch finish belongs to another row");
  const auto count = step.impl_->count;
  auto *profile = step.impl_->profile;
  QwenForwardResult result{.token = sampled_token,
                           .tokens = count,
                           .executed_layers = U32(p.blocks.size()),
                           .host_read_bytes = step.impl_->verify ? 0U : 4U,
                           .short_matrix_calls = profile->short_matrix_calls,
                           .short_lane_calls = profile->short_lane_calls};
  if (step.impl_->verify) {
    p.pending_verify = count;
    p.has_last_logits = false;
  } else {
    if (p.page_pool)
      p.page_pool->commit(count);
    p.next_position += count;
    p.last_trace_tokens = p.route_trace.buffer ? count : 0;
    p.last_logits_row = count - 1;
    p.has_last_logits = true;
  }
  step.impl_->active = false;
  return result;
}

PackedBatchResult
QwenForward::run_packed_batch(std::span<const PackedSlotSlice> slices,
                              bool use_icb, bool scatter_slot_logits) {
  return impl_->run_packed_batch_impl(slices, use_icb, false,
                                      scatter_slot_logits);
}

PackedBatchResult
QwenForward::run_packed_batch_and_sample(
    std::span<const PackedSlotSlice> slices, bool use_icb) {
  return impl_->run_packed_batch_impl(slices, use_icb, true, false);
}

const MetalBuffer &QwenForward::packed_logits_buffer() const {
  if (impl_->shared_primary_ != nullptr) {
    return impl_->shared_primary_->packed_logits_buffer();
  }
  return *(impl_->last_packed_logits_buf_ ? impl_->last_packed_logits_buf_
                                          : impl_->logits.buffer);
}

const MetalBuffer &QwenForward::batch_logits_buffer() const {
  return *impl_->last_logits.buffer;
}
const MetalBuffer &QwenForward::batch_projection_input_buffer() const {
  return *impl_->norm.buffer;
}
const MetalBuffer &QwenForward::batch_projection_output_buffer() const {
  return *impl_->logits.buffer;
}
U32 QwenForward::vocabulary_size() const { return impl_->v; }

QwenForwardResult QwenForward::run(std::span<const U32> tokens,
                                   const GpuSamplingOptions &sampling,
                                   bool inspect_tensors, bool use_icb,
                                   bool profile_stages) {
  return execute(tokens, sampling, inspect_tensors, use_icb, false,
                 profile_stages);
}

QwenForwardResult QwenForward::verify(std::span<const U32> tokens,
                                      const GpuSamplingOptions &sampling,
                                      bool inspect_tensors,
                                      bool profile_stages) {
  if (!impl_->speculative_enabled || tokens.size() > 8)
    throw std::invalid_argument(
        "Qwen verification unavailable or wider than 8");
  return execute(tokens, sampling, inspect_tensors, true, true, profile_stages);
}

bool QwenForward::can_enqueue_verify(std::size_t count) const noexcept {
  const auto &p = *impl_;
  if (!p.speculative_enabled || p.pending_verify != 0U ||
      p.ar_in_flight_count_ != 0U || p.prefill_in_flight_count_ != 0U ||
      count == 0U || count > 8U) {
    return false;
  }
  if (p.shared_primary_ != nullptr && count > p.workspace_rows_) {
    return false;
  }
  if (count > p.max_context - p.next_position) {
    return false;
  }
  const auto &profiles =
      p.split_attention_enabled ? p.split_verify_profiles : p.verify_profiles;
  const auto found = std::find_if(
      profiles.begin(), profiles.end(), [&](const Profile &profile) {
        return count <= profile.capacity &&
               (!profile.split_attention ||
                p.next_position + count <= profile.context_envelope);
      });
  return found != profiles.end() && found->icb.has_value();
}

MetalPendingSubmission
QwenForward::enqueue_verify(std::span<const U32> tokens,
                            const GpuSamplingOptions &sampling) {
  auto &p = *impl_;
  p.execution_started = true;
  if (p.pending_verify)
    throw std::logic_error("Qwen verification must commit or abort first");
  if (!p.speculative_enabled || tokens.empty() || tokens.size() > 8U ||
      !can_enqueue_verify(tokens.size()) ||
      std::any_of(tokens.begin(), tokens.end(),
                  [&](U32 id) { return id >= p.v; }) ||
      sampling.top_k > 64 ||
      (sampling.temperature > 0 && sampling.top_k == 0) ||
      sampling.temperature < 0 || !std::isfinite(sampling.temperature) ||
      !std::isfinite(sampling.top_p) || sampling.top_p < 0 ||
      sampling.top_p > 1 || sampling.contract != p.sampling_contract) {
    throw std::invalid_argument(
        "Qwen async verify token/context/sampling invalid");
  }
  auto &profiles =
      p.split_attention_enabled ? p.split_verify_profiles : p.verify_profiles;
  auto found = std::find_if(
      profiles.begin(), profiles.end(), [&](const Profile &profile) {
        return tokens.size() <= profile.capacity &&
               (!profile.split_attention ||
                p.next_position + tokens.size() <= profile.context_envelope);
      });
  if (found == profiles.end() || !found->icb.has_value())
    throw std::out_of_range("Qwen async verify ICB profile unavailable");
  auto &profile = *found;

  p.last_trace_tokens = 0U;
  std::fill_n(p.token_staging.begin(), profile.capacity, 0U);
  std::copy(tokens.begin(), tokens.end(), p.token_staging.begin());
  const_cast<MetalBuffer *>(p.token_ids.buffer)
      ->write(std::as_bytes(std::span(p.token_staging).first(profile.capacity)));
  for (const auto &patch : profile.patches) {
    const U32 value = patch.position ? p.next_position
                                     : U32(tokens.size()) * patch.multiplier;
    std::memcpy(profile.parameter_data.data() + patch.offset + patch.word * 4,
                &value, 4);
  }
  if (p.attention_positions.buffer) {
    std::array<U32, 8> positions{};
    for (U32 i = 0; i < tokens.size(); ++i)
      positions[i] = p.next_position + i;
    const_cast<MetalBuffer *>(p.attention_positions.buffer)
        ->write(std::as_bytes(std::span(positions).first(tokens.size())));
  }
  profile.parameters.write("step", profile.parameter_data);

  auto *page_pool = p.page_pool.get();
  if (page_pool && page_pool->tokens() != p.next_position)
    throw std::logic_error("Qwen KV page frontier disagrees with model");
  const auto page_copies = page_pool
                               ? page_pool->begin_append(U32(tokens.size()))
                               : std::vector<KvPageCopy>{};
  struct PageGuard {
    KvPagePool *pool;
    bool active;
    ~PageGuard() {
      if (active)
        try {
          pool->abort();
        } catch (...) {
        }
    }
  } page_guard{page_pool, page_pool != nullptr};
  p.execute_cow_copies(page_copies);

  MetalPendingSubmission pending = p.context.replay_icb_async(*profile.icb);
  p.pending_verify = U32(tokens.size());
  p.verify_async_profile_ = &profile;
  p.has_last_logits = false;
  page_guard.active = false;
  return pending;
}

QwenForwardResult QwenForward::finish_verify(MetalPendingSubmission &&pending,
                                             U32 count) {
  auto &p = *impl_;
  MetalSubmission sub = p.context.wait_submission(std::move(pending));
  if (p.pending_verify == 0U) {
    p.pending_verify = count;
  } else if (count != 0U && p.pending_verify != count) {
    throw std::invalid_argument("Qwen finish_verify width mismatch");
  }
  const Profile *prof = p.verify_async_profile_;
  p.verify_async_profile_ = nullptr;
  return QwenForwardResult{
      .token = 0U,
      .tokens = p.pending_verify,
      .executed_layers = U32(p.blocks.size()),
      .host_read_bytes = 0U,
      .short_matrix_calls = prof ? prof->short_matrix_calls : 0U,
      .short_lane_calls = prof ? prof->short_lane_calls : 0U,
      .submission = std::move(sub),
  };
}

QwenForwardResult QwenForward::execute(std::span<const U32> tokens,
                                       const GpuSamplingOptions &sampling,
                                       bool inspect_tensors, bool use_icb,
                                       bool verify, bool profile_stages) {
  auto &p = *impl_;
  p.require_no_async_in_flight("execute");
  p.execution_started = true;
  if (p.pending_verify)
    throw std::logic_error("Qwen verification must commit or abort first");
  const U32 max_forward_rows = p.shared_primary_ != nullptr
                                   ? p.shared_primary_->impl_->workspace_rows_
                                   : p.workspace_rows_;
  if (tokens.empty() || tokens.size() > max_forward_rows ||
      tokens.size() > p.max_context - p.next_position ||
      std::any_of(tokens.begin(), tokens.end(),
                  [&](U32 id) { return id >= p.v; }) ||
      sampling.top_k > 64 ||
      (sampling.temperature > 0 && sampling.top_k == 0) ||
      sampling.temperature < 0 || !std::isfinite(sampling.temperature) ||
      !std::isfinite(sampling.top_p) || sampling.top_p < 0 ||
      sampling.top_p > 1)
    throw std::invalid_argument(
        "Qwen forward token/context/sampling request invalid");
  if (sampling.contract != p.sampling_contract)
    throw std::invalid_argument(
        "sampling invocation contract differs from installed Program");

  // Secondary 16-row slot prefill delegation (>16 tokens) via primary's 1024-row workspace!
  if (p.shared_primary_ != nullptr && tokens.size() > p.workspace_rows_) {
    if (verify || inspect_tensors || profile_stages)
      throw std::out_of_range(
          "Qwen secondary slot diagnostic/verify width exceeds local scratch");
    return p.shared_primary_->impl_->execute_delegated_prefill(
        *this, tokens, sampling, use_icb);
  }

  // Module 4b: Dual-ANE streaming co-execution for fixed T_chunk prefill steps
  // (and exact multiples of T_chunk so callers passing 2048+ never silently bypass ANE).
  if (p.can_run_ane_chunk(U32(tokens.size()), verify, inspect_tensors,
                          profile_stages)) {
    return p.execute_ane_chunk(tokens, sampling, /*trunk_only=*/false);
  }
  if (p.ane_chunk_tokens_ > 0U && tokens.size() > p.ane_chunk_tokens_ &&
      (tokens.size() % p.ane_chunk_tokens_) == 0U &&
      p.can_run_ane_chunk(p.ane_chunk_tokens_, verify, inspect_tensors,
                          profile_stages)) {
    QwenForwardResult total_res{};
    total_res.tokens = U32(tokens.size());
    total_res.executed_layers = U32(p.blocks.size());
    for (std::size_t off = 0U; off < tokens.size(); off += p.ane_chunk_tokens_) {
      const bool is_last = (off + p.ane_chunk_tokens_ == tokens.size());
      auto sub_res = p.execute_ane_chunk(
          tokens.subspan(off, p.ane_chunk_tokens_), sampling,
          /*trunk_only=*/!is_last);
      total_res.submission.timing.gpu_ms += sub_res.submission.timing.gpu_ms;
      total_res.submission.timing.wall_ms += sub_res.submission.timing.wall_ms;
      total_res.submission.dispatch_count += sub_res.submission.dispatch_count;
      total_res.submission.shared_event_value =
          sub_res.submission.shared_event_value;
      total_res.prefill_delta.prefill_w4a8_tokens +=
          sub_res.prefill_delta.prefill_w4a8_tokens;
      total_res.prefill_delta.prefill_w4a4_tokens +=
          sub_res.prefill_delta.prefill_w4a4_tokens;
      total_res.prefill_delta.prefill_sage2_tokens +=
          sub_res.prefill_delta.prefill_sage2_tokens;
      total_res.prefill_delta.prefill_minference_blocks_skipped +=
          sub_res.prefill_delta.prefill_minference_blocks_skipped;
      total_res.prefill_delta.prefill_minference_blocks_total +=
          sub_res.prefill_delta.prefill_minference_blocks_total;
      total_res.prefill_delta.prefill_ane_active =
          sub_res.prefill_delta.prefill_ane_active;
      total_res.prefill_delta.prefill_ane_mode_effective =
          sub_res.prefill_delta.prefill_ane_mode_effective;
      total_res.prefill_delta.prefill_ane_inactive_reason =
          sub_res.prefill_delta.prefill_ane_inactive_reason;
      total_res.prefill_delta.prefill_ane_split_alpha =
          sub_res.prefill_delta.prefill_ane_split_alpha;
      total_res.prefill_delta.prefill_ane_offload_tokens +=
          sub_res.prefill_delta.prefill_ane_offload_tokens;
      total_res.prefill_delta.prefill_ane_fallback_tokens +=
          sub_res.prefill_delta.prefill_ane_fallback_tokens;
      if (is_last) {
        total_res.token = sub_res.token;
        total_res.host_read_bytes = sub_res.host_read_bytes;
      }
    }
    return total_res;
  }

  Profile *selected_profile = nullptr;
  if (!verify && tokens.size() > 8U) {
    selected_profile = p.select_long_ctx_prefill_profile(
        U32(tokens.size()), p.next_position, false, false);
  }
  if (selected_profile == nullptr) {
    auto &profiles = p.split_attention_enabled && tokens.size() <= 8
                         ? (verify ? p.split_verify_profiles : p.split_profiles)
                         : (verify ? p.verify_profiles : p.profiles);
    auto found = std::find_if(
        profiles.begin(), profiles.end(), [&](const Profile &profile) {
          return tokens.size() <= profile.capacity &&
                 (!profile.split_attention ||
                  p.next_position + tokens.size() <= profile.context_envelope);
        });
    if (found == profiles.end())
      throw std::out_of_range("Qwen forward capacity");
    selected_profile = &(*found);
  }
  const U32 start_pos = p.next_position;
  p.last_trace_tokens = 0;
  auto &profile = *selected_profile;
  std::fill_n(p.token_staging.begin(), profile.capacity, 0);
  std::copy(tokens.begin(), tokens.end(), p.token_staging.begin());
  const_cast<MetalBuffer *>(p.token_ids.buffer)
      ->write(std::as_bytes(std::span(p.token_staging).first(profile.capacity)));
  for (const auto &patch : profile.patches) {
    const U32 value = patch.position ? p.next_position
                                     : U32(tokens.size()) * patch.multiplier;
    std::memcpy(profile.parameter_data.data() + patch.offset + patch.word * 4,
                &value, 4);
  }
  SampleParams params{p.v,
                      sampling.top_k,
                      sampling.temperature,
                      sampling.top_p,
                      sampling.seed,
                      sampling.stream,
                      p.next_position + tokens.size() - 1};
  if (p.attention_positions.buffer) {
    std::array<U32, 4096> positions{};
    for (U32 i = 0; i < tokens.size(); ++i)
      positions[i] = p.next_position + i;
    const_cast<MetalBuffer *>(p.attention_positions.buffer)
        ->write(std::as_bytes(std::span(positions).first(tokens.size())));
  }
  if (!verify) {
    if (p.sampling_contract == SamplingContract::PositionCoupledV1) {
      const PositionSamplingRow row{.seed = sampling.seed,
                                    .position = p.next_position + tokens.size(),
                                    .top_k = sampling.top_k,
                                    .temperature = sampling.temperature,
                                    .top_p = sampling.top_p};
      std::memcpy(profile.parameter_data.data() + profile.sampling_offset, &row,
                  sizeof(row));
    } else {
      std::memcpy(profile.parameter_data.data() + profile.sampling_offset,
                  &params, sizeof(params));
    }
  }
  profile.parameters.write("step", profile.parameter_data);
  auto *page_pool = p.page_pool.get();
  if (page_pool && page_pool->tokens() != p.next_position)
    throw std::logic_error("Qwen KV page frontier disagrees with model");
  const auto page_copies = page_pool
                               ? page_pool->begin_append(U32(tokens.size()))
                               : std::vector<KvPageCopy>{};
  struct PageGuard {
    KvPagePool *pool;
    bool active;
    ~PageGuard() {
      if (active)
        try {
          pool->abort();
        } catch (...) {
        }
    }
  } page_guard{page_pool, page_pool != nullptr};
  p.execute_cow_copies(page_copies);
  QwenForwardResult result{.tokens = U32(tokens.size()),
                           .executed_layers = U32(p.blocks.size()),
                           .short_matrix_calls = profile.short_matrix_calls,
                           .short_lane_calls = profile.short_lane_calls};
  if (profile_stages) {
    if (!use_icb || !profile.icb)
      throw std::invalid_argument(
          "Qwen stage timing requires a recorded ICB profile");
    for (const auto &stage : profile.stages) {
      const auto measurement = p.context.replay_icb(*profile.icb, stage.name);
      result.stage_timings.push_back({stage.name, measurement.timing.gpu_ms,
                                      measurement.timing.wall_ms,
                                      measurement.dispatch_count});
      result.submission.timing.gpu_ms += measurement.timing.gpu_ms;
      result.submission.timing.wall_ms += measurement.timing.wall_ms;
      result.submission.dispatch_count += measurement.dispatch_count;
      result.submission.shared_event_value = measurement.shared_event_value;
    }
  } else if (use_icb && profile.icb) {
    result.submission = p.context.replay_icb(*profile.icb);
  } else {
    const std::array invocation = {
        MetalCommandInvocation{.plan = &profile.plan,
                               .bindings = profile.bindings,
                               .parameters = &profile.parameters}};
    result.submission = p.context.execute_batch(invocation);
  }
  if (!verify) {
    const auto token = p.sampled.buffer->read(
        0, p.sampling_contract == SamplingContract::PositionCoupledV1 ? 8 : 4);
    std::memcpy(&result.token, token.data(), 4);
    result.host_read_bytes = token.size();
    if (p.sampling_contract == SamplingContract::PositionCoupledV1) {
      U32 flag = 0;
      std::memcpy(&flag, token.data() + 4, 4);
      if (flag & 2)
        throw std::runtime_error("nonfinite position target logits");
      if (flag & 1) {
        auto ib = p.position_ids.buffer->read(0, U64(sampling.top_k) * 4);
        auto sb = p.position_scores.buffer->read(0, U64(sampling.top_k) * 4);
        std::vector<U32> ids(sampling.top_k);
        std::vector<float> scores(sampling.top_k);
        std::memcpy(ids.data(), ib.data(), ib.size());
        std::memcpy(scores.data(), sb.data(), sb.size());
        result.token = position_sample_candidates(
            ids, scores,
            {.seed = sampling.seed,
             .position = p.next_position + tokens.size(),
             .top_k = sampling.top_k,
             .temperature = sampling.temperature,
             .top_p = sampling.top_p});
        result.host_read_bytes += ib.size() + sb.size();
      }
    } else if (result.token >= p.v) {
      throw std::runtime_error("nonfinite or invalid legacy sampled token");
    }
  }
  if (inspect_tensors) {
    if (p.route_trace.buffer) {
      const auto count = mul(tokens.size(), mul(p.blocks.size(), 8));
      const auto bytes = p.route_trace.buffer->read(0, count * 4);
      result.layer_routes.resize(count);
      std::memcpy(result.layer_routes.data(), bytes.data(), bytes.size());
      result.host_read_bytes += bytes.size();
      result.router_logits =
          bf16_values(*p.router_logits_trace.buffer,
                      mul(tokens.size(), mul(p.blocks.size(), 256)));
      result.router_scores = bf16_values(*p.router_scores_trace.buffer, count);
      result.host_read_bytes +=
          2 * (result.router_logits.size() + result.router_scores.size());
    }
    if (!p.inspect_tensors_workspace && tokens.size() > 8) {
      auto &norm_prof = p.inspect_norm_profile;
      for (const auto &patch : norm_prof.patches) {
        const U32 val = U32(tokens.size()) * patch.multiplier;
        std::memcpy(norm_prof.parameter_data.data() + patch.offset +
                        patch.word * 4,
                    &val, 4);
      }
      norm_prof.parameters.write("step", norm_prof.parameter_data);
      if (use_icb && norm_prof.icb)
        static_cast<void>(p.context.replay_icb(*norm_prof.icb));
      else {
        const std::array calls = {
            MetalCommandInvocation{.plan = &norm_prof.plan,
                                   .bindings = norm_prof.bindings,
                                   .parameters = &norm_prof.parameters}};
        static_cast<void>(p.context.execute_batch(calls));
      }
      result.hidden = bf16_values(*p.norm.buffer, mul(tokens.size(), p.h));
      result.logits.clear();
      result.logits.reserve(mul(tokens.size(), p.v));
      auto &chunk_prof = p.inspect_head_chunk_profile;
      for (U32 row_offset = 0; row_offset < tokens.size(); row_offset += 8) {
        const U32 chunk = std::min<U32>(8U, U32(tokens.size()) - row_offset);
        for (const auto &patch : chunk_prof.patches) {
          const U32 val =
              patch.position ? row_offset : chunk * patch.multiplier;
          std::memcpy(chunk_prof.parameter_data.data() + patch.offset +
                          patch.word * 4,
                      &val, 4);
        }
        chunk_prof.parameters.write("step", chunk_prof.parameter_data);
        if (use_icb && chunk_prof.icb)
          static_cast<void>(p.context.replay_icb(*chunk_prof.icb));
        else {
          const std::array calls = {
              MetalCommandInvocation{.plan = &chunk_prof.plan,
                                     .bindings = chunk_prof.bindings,
                                     .parameters = &chunk_prof.parameters}};
          static_cast<void>(p.context.execute_batch(calls));
        }
        auto chunk_logits = bf16_values(*p.logits.buffer, mul(chunk, p.v));
        result.logits.insert(result.logits.end(), chunk_logits.begin(),
                             chunk_logits.end());
      }
    } else if (profile.fused_final_norm_t1 && tokens.size() == 1U) {
      const auto raw_h = bf16_values(*p.hidden.buffer, p.h);
      const auto fn_ref = p.direct(p.final_norm, p.h);
      const auto fn_bytes = fn_ref.buffer->read(fn_ref.offset, fn_ref.bytes);
      float sq = 0.0F;
      for (const float hv : raw_h)
        sq += hv * hv;
      const float inv =
          1.0F / std::sqrt(sq / float(p.h) + float(p.config.rms_norm_eps));
      result.hidden.resize(p.h);
      for (U32 i = 0; i < p.h; ++i) {
        std::uint16_t wc = 0;
        std::memcpy(&wc, fn_bytes.data() + 2 * i, 2);
        const float wf =
            std::bit_cast<float>(U32(wc) << 16) + p.normalization_weight_offset;
        const float nv = raw_h[i] * inv * wf;
        const U32 b32 = std::bit_cast<U32>(nv);
        const U32 lsb = (b32 >> 16) & 1U;
        const std::uint16_t bf =
            static_cast<std::uint16_t>((b32 + 0x7FFFU + lsb) >> 16);
        result.hidden[i] = std::bit_cast<float>(U32(bf) << 16);
      }
      result.logits = bf16_values(*p.logits.buffer, mul(tokens.size(), p.v));
    } else {
      result.hidden = bf16_values(*p.norm.buffer, mul(tokens.size(), p.h));
      result.logits = bf16_values(*p.logits.buffer, mul(tokens.size(), p.v));
    }
    result.host_read_bytes += mul(tokens.size(), mul(U64(p.h) + p.v, 2));
    if (p.features.buffer) {
      result.layer_features =
          bf16_values(*p.features.buffer,
                      mul(tokens.size(), mul(p.feature_layers.size(), p.h)));
      result.host_read_bytes += mul(result.layer_features.size(), 2);
    }
    if (p.config.linear_attention_layers() > 0) {
      for (const auto &[name, tensor, width] :
           {std::tuple{"qkv", p.qkv, p.conv},
            std::tuple{"convolved", p.convolved, p.conv},
            std::tuple{"recurrent", p.recurrent, p.gv},
            std::tuple{"z", p.z, p.gv},
            std::tuple{"a", p.a, U32(p.config.linear_num_value_heads)},
            std::tuple{"b", p.b, U32(p.config.linear_num_value_heads)}}) {
        result.gdn_inspection[name] =
            bf16_values(*tensor.buffer, mul(tokens.size(), width));
        result.host_read_bytes += mul(tokens.size(), mul(width, 2));
      }
    }
  }
  if (verify) {
    p.last_trace_tokens = 0;
    p.pending_verify = U32(tokens.size());
    p.has_last_logits = false;
    page_guard.active = false;
  } else {
    if (page_pool)
      page_pool->commit(U32(tokens.size()));
    p.next_position += U32(tokens.size());
    p.last_trace_tokens = p.route_trace.buffer ? U32(tokens.size()) : 0;
    if (!p.inspect_tensors_workspace && profile.capacity > 8U) {
      p.last_logits_row =
          inspect_tensors ? U32((tokens.size() - 1) % 8U) : 0U;
    } else {
      p.last_logits_row = U32(tokens.size() - 1);
    }
    p.has_last_logits = true;
    page_guard.active = false;
    p.record_prefill_telemetry(U32(tokens.size()), start_pos, profile.used_w4a8,
                               profile.used_sage2, profile.used_minference,
                               profile.capacity, profile.minf_context_envelope,
                               /*used_ane=*/false, &result.prefill_delta,
                               std::nullopt, profile.used_w4a4);
  }
  return result;
}

MetalIcbProfile *QwenForward::prepare_commit_prefix(U32 count) {
  auto &p = *impl_;
  if (!p.pending_verify || count > p.pending_verify)
    throw std::invalid_argument("Qwen commit outside pending prefix");
  if (!count)
    return nullptr;
  auto &profile = p.commit_profile;
  for (const auto &patch : profile.patches) {
    U32 value = patch.position ? p.next_position : count * patch.multiplier;
    std::memcpy(profile.parameter_data.data() + patch.offset + patch.word * 4,
                &value, 4);
  }
  profile.parameters.write("step", profile.parameter_data);
  return profile.icb ? &(*profile.icb) : nullptr;
}

MetalSubmission QwenForward::finish_commit_prefix(U32 count,
                                                  bool already_executed) {
  auto &p = *impl_;
  if (!p.pending_verify || count > p.pending_verify)
    throw std::invalid_argument("Qwen commit outside pending prefix");
  MetalSubmission result;
  if (count && !already_executed) {
    auto &profile = p.commit_profile;
    if (profile.icb)
      result = p.context.replay_icb(*profile.icb);
    else {
      const std::array calls = {
          MetalCommandInvocation{.plan = &profile.plan,
                                 .bindings = profile.bindings,
                                 .parameters = &profile.parameters}};
      result = p.context.execute_batch(calls);
    }
  }
  if (p.page_pool)
    p.page_pool->commit(count);
  p.next_position += count;
  p.pending_verify = 0;
  p.has_last_logits = false;
  return result;
}

MetalSubmission QwenForward::commit_prefix(U32 count) {
  auto *icb = prepare_commit_prefix(count);
  (void)icb;
  return finish_commit_prefix(count, false);
}
void QwenForward::abort_verify() {
  if (impl_->pending_verify && impl_->page_pool)
    impl_->page_pool->abort();
  impl_->pending_verify = 0;
  impl_->verify_async_profile_ = nullptr;
  impl_->has_last_logits = false;
}
std::shared_ptr<const Q5High1Cache>
QwenForward::q5_high1_cache() const noexcept {
  return impl_->q5_high1_cache;
}
std::optional<RowExactPlaneSlice>
QwenForward::q5_high1_plane(U32 object_id) const {
  if (!impl_->q5_high1_cache)
    return std::nullopt;
  return impl_->q5_high1_cache->find_q5_high1(object_id);
}
std::optional<RowExactPlaneSlice>
QwenForward::affine_sb_tiled_plane(U32 object_id) const {
  if (impl_->affine_nax_cache) {
    if (const auto *tw = impl_->affine_nax_cache->find(object_id);
        tw && tw->sb_tiled.buffer) {
      return to_slice(tw->sb_tiled);
    }
  }
  if (impl_->q5_high1_cache) {
    return impl_->q5_high1_cache->find_sb_tiled(object_id);
  }
  return std::nullopt;
}
std::optional<ResolvedKl32Mxfp4Planes>
QwenForward::resolve_kl32_mxfp4(const MaterializedWeightView &w) const {
  Ref c{}, s{};
  if (!impl_->resolve_mxfp4_planes_any_t(w, &c, &s))
    return std::nullopt;
  return ResolvedKl32Mxfp4Planes{
      .codes = to_slice(c),
      .scales = to_slice(s),
      .n = U32(w.row_count),
      .k = U32(w.in_features),
      .bn = mxfp4_nax_bn(U32(w.row_count)),
      .has_fwht64 = w.has_fwht64(),
  };
}
std::optional<ResolvedKl32AffinePlanes>
QwenForward::resolve_kl32_affine(const MaterializedWeightView &w) const {
  Ref c{}, h_pl{}, sb{};
  bool lane_v2 = false;
  if (!impl_->resolve_affine_planes_any_t(w, &c, &h_pl, &sb, &lane_v2))
    return std::nullopt;
  return ResolvedKl32AffinePlanes{
      .codes = to_slice(c),
      .high = to_slice(h_pl),
      .sb_tiled = to_slice(sb),
      .n = U32(w.row_count),
      .k = U32(w.in_features),
      .bn = affine_nax_bn(U32(w.row_count)),
      .bits = affine_nax_bits(w.format),
      .is_lane_v2 = lane_v2,
      .has_fwht64 = w.has_fwht64(),
  };
}
const MetalBuffer &QwenForward::feature_history() const {
  if (!impl_->feature_history.buffer)
    throw std::logic_error("Qwen features not installed");
  return *impl_->feature_history.buffer;
}
U64 QwenForward::feature_epoch() const noexcept {
  return impl_->feature_epoch_;
}
U32 QwenForward::feature_ring_rows() const noexcept {
  return impl_->feature_ring_rows_;
}
KvPagePool *QwenForward::kv_page_pool() noexcept {
  return impl_->page_pool.get();
}
const KvPagePool *QwenForward::kv_page_pool() const noexcept {
  return impl_->page_pool.get();
}
void QwenForward::share_prefill_workspace(QwenForward &primary) {
  if (&primary != this)
    impl_->shared_primary_ = &primary;
}
void QwenForward::share_kv_arena(QwenForward &primary) {
  if (&primary != this)
    impl_->shared_primary_ = &primary;
}
const MaterializedWeightView &QwenForward::embedding_weight() const {
  return impl_->embedding;
}
const MaterializedWeightView &QwenForward::head_weight() const {
  return impl_->head;
}
std::vector<float> QwenForward::read_logits(U32 count) const {
  const U64 max_rows = impl_->logits.bytes / mul(impl_->v, 2);
  if (count > max_rows)
    throw std::invalid_argument("Qwen logits count");
  return bf16_values(*impl_->logits.buffer, U64(count) * impl_->v);
}
std::uint32_t QwenForward::last_logits_row() const {
  const auto &p = *impl_;
  if (p.pending_verify || !p.has_last_logits)
    throw std::logic_error("Qwen committed last logits unavailable");
  return p.last_logits_row;
}
std::vector<std::byte> QwenForward::read_last_logits_bf16() const {
  const auto &p = *impl_;
  const U64 row_bytes = mul(p.v, 2);
  const U64 offset = mul(last_logits_row(), row_bytes);
  if (offset + row_bytes > p.logits.bytes)
    throw std::out_of_range("Qwen last logits row outside buffer");
  return p.logits.buffer->read(offset, row_bytes);
}
const MetalBuffer &QwenForward::verification_logits_buffer() const {
  return *impl_->logits.buffer;
}
const MetalBuffer &QwenForward::normed_hidden_buffer() const {
  return *impl_->norm.buffer;
}
const MetalBuffer &QwenForward::trunk_hidden_buffer() const {
  return *impl_->hidden.buffer;
}
const MaterializedWeightView &QwenForward::final_norm_weight() const {
  return impl_->final_norm;
}
float QwenForward::normalization_weight_offset() const noexcept {
  return impl_->normalization_weight_offset;
}
std::vector<std::byte> QwenForward::inspect_recurrent_state() const {
  std::vector<std::byte> result;
  for (const auto &block : impl_->blocks) {
    if (block.type == LayerType::full_attention)
      continue;
    for (const auto ref : {block.state0, block.state1}) {
      const auto data = ref.buffer->read(ref.offset, ref.bytes);
      result.insert(result.end(), data.begin(), data.end());
    }
  }
  return result;
}
std::vector<U32> QwenForward::inspect_kv_page_table() const {
  if (!impl_->page_pool)
    return {};
  const auto bytes = impl_->page_pool->table_buffer().read();
  std::vector<U32> pages(bytes.size() / sizeof(U32));
  std::memcpy(pages.data(), bytes.data(), bytes.size());
  return pages;
}
QwenMoeRouteSnapshot QwenForward::read_moe_route_trace(U32 layer) const {
  const auto &p = *impl_;
  if (!p.route_trace.buffer || !p.router_scores_trace.buffer ||
      p.pending_verify || !p.last_trace_tokens ||
      layer >= U32(p.config.num_hidden_layers) ||
      p.last_trace_tokens > p.next_position)
    throw std::logic_error("Qwen committed MoE route trace unavailable");
  const U32 t = p.last_trace_tokens;
  const U64 layers = U64(p.config.num_hidden_layers);
  const auto id_bytes = p.route_trace.buffer->read(0, U64(t) * layers * 8 * 4);
  const auto score_bytes =
      p.router_scores_trace.buffer->read(0, U64(t) * layers * 8 * 2);
  QwenMoeRouteSnapshot result{.layer = layer,
                              .tokens = t,
                              .start_position = p.next_position - t,
                              .readback_bytes =
                                  id_bytes.size() + score_bytes.size()};
  result.expert_ids.resize(U64(t) * 8);
  result.scores_bf16.resize(U64(t) * 8);
  for (U32 token = 0; token < t; ++token) {
    const U64 source = (U64(token) * layers + layer) * 8;
    std::memcpy(result.expert_ids.data() + U64(token) * 8,
                id_bytes.data() + source * 4, 8 * 4);
    std::memcpy(result.scores_bf16.data() + U64(token) * 8,
                score_bytes.data() + source * 2, 8 * 2);
  }
  return result;
}

namespace {
std::string continuation_payload_digest(const QwenContinuation &snapshot) {
  ExecutionIdentityFields fields{
      {"execution", snapshot.execution_identity},
      {"owner", std::to_string(snapshot.owner_id)},
      {"position", std::to_string(snapshot.position)},
      {"kv_lease", std::to_string(snapshot.kv_lease)},
      {"features", artifact_sha256_hex(snapshot.feature_history)}};
  for (std::size_t index = 0; index < snapshot.recurrent.size(); ++index)
    fields.emplace("recurrent." + std::to_string(index),
                   artifact_sha256_hex(snapshot.recurrent[index]));
  return execution_identity_digest(fields);
}
} // namespace

void QwenForward::bind_execution_identity(std::string identity) {
  auto &p = *impl_;
  if (identity.empty() || p.execution_started || p.execution_identity_bound ||
      p.next_position || p.pending_verify)
    throw std::invalid_argument("Qwen execution identity must bind once before use");
  p.bound_context_identity_ = std::move(identity);
  p.execution_identity_bound = true;
  p.refresh_lora_execution_identity();
}

void QwenForward::set_bypass_lora(
    std::shared_ptr<const QwenBypassLoraRuntime> bypass_lora,
    std::string context_execution_identity) {
  auto &p = *impl_;
  if (p.merged_lora_ != nullptr && !p.merged_lora_->views_by_object_id.empty()) {
    throw std::logic_error(
        "QwenForward cannot hot-swap bypass LoRA while merge LoRA is active");
  }
  p.drain_and_invalidate_async_steps();
  p.reset();
  p.bypass_lora_ = std::move(bypass_lora);
  p.ensure_bypass_scratch();
  p.rebuild_profiles();
  if (!context_execution_identity.empty()) {
    p.bound_context_identity_ = std::move(context_execution_identity);
    p.execution_identity_bound = true;
  }
  p.refresh_lora_execution_identity();
  p.execution_started = false;
}

std::uint64_t QwenForward::lora_scratch_bytes() const noexcept {
  const auto &p = *impl_;
  return p.lora_bypass_tmp_buf_ ? p.lora_bypass_tmp_buf_->size() : 0ULL;
}

std::string_view QwenForward::execution_identity() const {
  return impl_->execution_identity;
}

U64 QwenForward::non_kv_checkpoint_bytes() const {
  const auto &p = *impl_;
  U64 bytes = p.execution_identity.size() + 64;
  for (const auto &block : p.blocks)
    if (block.type != LayerType::full_attention)
      bytes += block.state0.bytes + block.state1.bytes;
  if (p.feature_history.buffer && p.feature_ring_rows_) {
    const U32 rows = std::min<U32>(p.next_position, p.feature_ring_rows_);
    bytes += U64(rows) * p.feature_layers.size() * p.h * 2;
  }
  return bytes;
}

U64 QwenForward::kv_page_all_layers_bytes() const {
  const auto &p = *impl_;
  if (!p.page_pool || p.config.full_attention_layers() <= 0)
    return 0;
  const U64 per_page = kv_page_bytes(p.kv_format_, 64U,
                                     U32(p.config.num_key_value_heads),
                                     U32(p.config.head_dim));
  return per_page * 2ULL * U64(p.config.full_attention_layers());
}

U64 QwenForward::checkpoint_bytes() const {
  const auto &p = *impl_;
  U64 bytes = non_kv_checkpoint_bytes();
  if (p.page_pool) {
    const U64 pages = (U64(p.next_position) + 63) / 64;
    bytes += pages * kv_page_all_layers_bytes();
  }
  return bytes;
}

QwenContinuation QwenForward::checkpoint() const {
  const auto &p = *impl_;
  p.require_no_async_in_flight("checkpoint");
  if (p.pending_verify || !p.next_position)
    throw std::logic_error("Qwen checkpoint requires committed prefill");
  QwenContinuation snapshot;
  snapshot.execution_identity = p.execution_identity;
  snapshot.owner_id = p.owner_id;
  snapshot.position = p.next_position;
  snapshot.accounted_bytes = checkpoint_bytes();
  for (const auto &block : p.blocks) {
    if (block.type == LayerType::full_attention)
      continue;
    for (const auto ref : {block.state0, block.state1}) {
      snapshot.recurrent.push_back(ref.buffer->read(ref.offset, ref.bytes));
    }
  }
  if (p.feature_history.buffer && p.feature_ring_rows_) {
    const U32 rows = std::min<U32>(snapshot.position, p.feature_ring_rows_);
    const U64 bytes = U64(rows) * p.feature_layers.size() * p.h * 2;
    snapshot.feature_history = p.feature_history.buffer->read(0, bytes);
  }
  if (p.page_pool) {
    snapshot.kv_lease = p.page_pool->retain_prefix();
    p.lease_pages_[snapshot.kv_lease] = p.page_pool->mapped_slots();
  }
  snapshot.payload_digest = continuation_payload_digest(snapshot);
  return snapshot;
}

void QwenForward::restore(const QwenContinuation &snapshot) {
  auto &p = *impl_;
  if (p.pending_verify || snapshot.owner_id != p.owner_id ||
      snapshot.execution_identity != p.execution_identity ||
      snapshot.payload_digest != continuation_payload_digest(snapshot) ||
      !snapshot.position || snapshot.position > p.max_context ||
      bool(snapshot.kv_lease) != bool(p.page_pool))
    throw std::invalid_argument("Qwen continuation identity/frontier invalid");
  std::size_t index = 0;
  for (const auto &block : p.blocks) {
    if (block.type == LayerType::full_attention)
      continue;
    for (const auto ref : {block.state0, block.state1}) {
      if (index >= snapshot.recurrent.size() ||
          snapshot.recurrent[index++].size() != ref.bytes)
        throw std::invalid_argument("Qwen continuation recurrent geometry");
    }
  }
  const U32 feat_rows =
      p.feature_ring_rows_
          ? std::min<U32>(snapshot.position, p.feature_ring_rows_)
          : 0U;
  const U64 feature_bytes =
      p.feature_history.buffer
          ? U64(feat_rows) * p.feature_layers.size() * p.h * 2
          : 0;
  if (index != snapshot.recurrent.size() ||
      snapshot.feature_history.size() != feature_bytes)
    throw std::invalid_argument("Qwen continuation feature geometry");
  try {
    p.drain_and_invalidate_async_steps();
    if (p.page_pool)
      p.page_pool->restore_prefix(snapshot.kv_lease);
    index = 0;
    for (const auto &block : p.blocks) {
      if (block.type == LayerType::full_attention)
        continue;
      for (const auto ref : {block.state0, block.state1})
        const_cast<MetalBuffer *>(ref.buffer)
            ->write(snapshot.recurrent[index++], ref.offset);
    }
    if (p.feature_history.buffer && feature_bytes)
      const_cast<MetalBuffer *>(p.feature_history.buffer)
          ->write(snapshot.feature_history);
    p.next_position = snapshot.position;
    p.last_logits_row = 0;
    p.has_last_logits = false;
    ++p.feature_epoch_;
  } catch (...) {
    p.reset();
    throw;
  }
}

void QwenForward::restore_shared(const QwenContinuation &snapshot,
                                 QwenForward *source_owner) {
  if (source_owner == this) {
    restore(snapshot);
    return;
  }
  auto &p = *impl_;
  if (!source_owner)
    throw std::invalid_argument("Qwen cross-slot restore requires source_owner");
  const auto &src = *source_owner->impl_;
  if (p.pending_verify || snapshot.owner_id != src.owner_id ||
      snapshot.execution_identity != src.execution_identity ||
      p.shared_contract_identity != src.shared_contract_identity ||
      snapshot.payload_digest != continuation_payload_digest(snapshot) ||
      !snapshot.position || snapshot.position > p.max_context ||
      bool(snapshot.kv_lease) != bool(p.page_pool))
    throw std::invalid_argument("Qwen cross-slot continuation identity invalid");

  std::size_t index = 0;
  for (const auto &block : p.blocks) {
    if (block.type == LayerType::full_attention)
      continue;
    for (const auto ref : {block.state0, block.state1}) {
      if (index >= snapshot.recurrent.size() ||
          snapshot.recurrent[index++].size() != ref.bytes)
        throw std::invalid_argument("Qwen cross-slot recurrent geometry");
    }
  }
  const U32 feat_rows =
      p.feature_ring_rows_
          ? std::min<U32>(snapshot.position, p.feature_ring_rows_)
          : 0U;
  const U64 feature_bytes =
      p.feature_history.buffer
          ? U64(feat_rows) * p.feature_layers.size() * p.h * 2
          : 0;
  if (index != snapshot.recurrent.size() ||
      snapshot.feature_history.size() != feature_bytes)
    throw std::invalid_argument("Qwen cross-slot feature geometry");

  try {
    p.drain_and_invalidate_async_steps();
    if (p.page_pool) {
      const auto lease_it = src.lease_pages_.find(snapshot.kv_lease);
      if (lease_it == src.lease_pages_.end())
        throw std::invalid_argument("Qwen source KV lease no longer active");
      const auto &src_pages = lease_it->second;
      const U32 num_pages = (snapshot.position + 63U) / 64U;
      if (src_pages.size() != num_pages)
        throw std::invalid_argument("Qwen source KV lease page count mismatch");

      if (p.page_pool->shared_allocator() != nullptr && src.page_pool &&
          p.page_pool->shared_allocator() == src.page_pool->shared_allocator()) {
        p.page_pool->install_shared_prefix(snapshot.position, src_pages);
      } else {
        p.page_pool->reset();
        static_cast<void>(p.page_pool->begin_append(snapshot.position));
        p.page_pool->commit(snapshot.position);
        const auto dst_pages = p.page_pool->mapped_slots();

        const U64 page_bytes = kv_page_bytes(p.kv_format_, 64U,
                                             U32(p.config.num_key_value_heads),
                                             U32(p.config.head_dim));
        static std::atomic<U64> restore_seq{1};
        Encoder enc(p.context, p.libraries, 1,
                    "qwen.restore_shared_kv." +
                        std::to_string(
                            restore_seq.fetch_add(1, std::memory_order_relaxed)));
        enc.begin_stage("copy_kv");
        for (std::size_t l = 0; l < p.blocks.size(); ++l) {
          if (p.blocks[l].type != LayerType::full_attention)
            continue;
          for (U32 pg = 0; pg < num_pages; ++pg) {
            const U64 s_off = U64(src_pages[pg]) * page_bytes;
            const U64 d_off = U64(dst_pages[pg]) * page_bytes;
            auto &d0 = enc.emit(
                "qwen_forward", "omlx_m5_forward_copy_bytes",
                {.width = (page_bytes + 15) / 16}, {.width = 256},
                {Ref{src.blocks[l].state0.buffer,
                     src.blocks[l].state0.offset + s_off, page_bytes},
                 Ref{p.blocks[l].state0.buffer,
                     p.blocks[l].state0.offset + d_off, page_bytes}});
            enc.dynamic(d0, 2, words({U32(page_bytes)}));
            auto &d1 = enc.emit(
                "qwen_forward", "omlx_m5_forward_copy_bytes",
                {.width = (page_bytes + 15) / 16}, {.width = 256},
                {Ref{src.blocks[l].state1.buffer,
                     src.blocks[l].state1.offset + s_off, page_bytes},
                 Ref{p.blocks[l].state1.buffer,
                     p.blocks[l].state1.offset + d_off, page_bytes}});
            enc.dynamic(d1, 2, words({U32(page_bytes)}));
          }
        }
        auto cp_prof = enc.finish();
        const std::array calls{
            MetalCommandInvocation{.plan = &cp_prof.plan,
                                   .bindings = cp_prof.bindings,
                                   .parameters = &cp_prof.parameters}};
        static_cast<void>(p.context.execute_batch(calls));
      }
    }

    index = 0;
    for (const auto &block : p.blocks) {
      if (block.type == LayerType::full_attention)
        continue;
      for (const auto ref : {block.state0, block.state1})
        const_cast<MetalBuffer *>(ref.buffer)
            ->write(snapshot.recurrent[index++], ref.offset);
    }
    if (p.feature_history.buffer && feature_bytes)
      const_cast<MetalBuffer *>(p.feature_history.buffer)
          ->write(snapshot.feature_history);
    p.next_position = snapshot.position;
    p.last_logits_row = 0;
    p.has_last_logits = false;
    ++p.feature_epoch_;
  } catch (...) {
    p.reset();
    throw;
  }
}

QwenSharedPrefixState
QwenForward::export_shared_prefix(bool include_recurrent) const {
  auto ckpt = checkpoint();
  QwenSharedPrefixState st;
  st.execution_identity = ckpt.execution_identity;
  st.position = ckpt.position;
  st.accounted_bytes = ckpt.accounted_bytes;
  if (impl_->page_pool) {
    st.kv_pages = impl_->page_pool->mapped_slots();
    impl_->page_pool->release_prefix(ckpt.kv_lease);
    impl_->lease_pages_.erase(ckpt.kv_lease);
  }
  if (include_recurrent) {
    st.recurrent = std::move(ckpt.recurrent);
  }
  st.feature_history = std::move(ckpt.feature_history);
  return st;
}

void QwenForward::import_shared_prefix(const QwenSharedPrefixState &state) {
  auto &p = *impl_;
  if (p.pending_verify || !state.position || state.position > p.max_context)
    throw std::invalid_argument("Qwen shared prefix import frontier invalid");
  p.drain_and_invalidate_async_steps();
  if (p.page_pool) {
    p.page_pool->install_shared_prefix(state.position, state.kv_pages);
  }
  if (!state.recurrent.empty()) {
    std::size_t index = 0;
    for (const auto &block : p.blocks) {
      if (block.type == LayerType::full_attention)
        continue;
      for (const auto ref : {block.state0, block.state1}) {
        if (index >= state.recurrent.size() ||
            state.recurrent[index].size() != ref.bytes)
          throw std::invalid_argument("Qwen shared prefix recurrent geometry");
        const_cast<MetalBuffer *>(ref.buffer)
            ->write(state.recurrent[index++], ref.offset);
      }
    }
  }
  if (p.feature_history.buffer && !state.feature_history.empty()) {
    const_cast<MetalBuffer *>(p.feature_history.buffer)
        ->write(state.feature_history);
  }
  p.next_position = state.position;
  p.last_logits_row = 0;
  p.has_last_logits = false;
  ++p.feature_epoch_;
}

void QwenForward::release(QwenContinuation &snapshot) {
  if (snapshot.owner_id != impl_->owner_id)
    throw std::invalid_argument("Qwen continuation belongs to another owner");
  if (snapshot.kv_lease) {
    if (!impl_->page_pool)
      throw std::logic_error("Qwen continuation belongs to a different owner");
    impl_->page_pool->release_prefix(snapshot.kv_lease);
    impl_->lease_pages_.erase(snapshot.kv_lease);
    snapshot.kv_lease = 0;
  }
  snapshot.owner_id = 0;
  snapshot.position = 0;
}

namespace {

constexpr U64 align_uma_slice_256(U64 bytes) noexcept {
  return (bytes + 255ULL) & ~255ULL;
}

struct UmaSlabLayout {
  std::vector<Ref> gdn_refs;
  std::vector<U64> gdn_slab_offsets;
  U64 recurrent_bytes = 0;
  U64 feature_bytes = 0;
  U64 feature_slab_offset = 0;
  U64 total_slab_bytes = 0;
};

template <class ImplType>
UmaSlabLayout compute_uma_slab_layout(const ImplType &p, U32 position) {
  UmaSlabLayout layout;
  layout.gdn_refs.reserve(p.blocks.size() * 2);
  layout.gdn_slab_offsets.reserve(p.blocks.size() * 2);
  U64 cursor = 0;
  for (const auto &block : p.blocks) {
    if (block.type == LayerType::full_attention)
      continue;
    for (const auto ref : {block.state0, block.state1}) {
      layout.gdn_refs.push_back(ref);
      layout.gdn_slab_offsets.push_back(cursor);
      layout.recurrent_bytes += ref.bytes;
      cursor += align_uma_slice_256(ref.bytes);
    }
  }
  const U32 feat_rows =
      (p.feature_history.buffer && p.feature_ring_rows_)
          ? std::min<U32>(position, p.feature_ring_rows_)
          : 0U;
  layout.feature_bytes =
      feat_rows ? U64(feat_rows) * p.feature_layers.size() * p.h * 2ULL : 0ULL;
  layout.feature_slab_offset = cursor;
  layout.total_slab_bytes =
      cursor + (layout.feature_bytes ? align_uma_slice_256(layout.feature_bytes)
                                     : 0ULL);
  return layout;
}

template <class ImplType>
void execute_uma_slab_copy(const ImplType &p, const UmaSlabLayout &layout,
                           const MetalBuffer &slab, bool pack_to_slab) {
  if (layout.gdn_refs.empty() && layout.feature_bytes == 0ULL)
    return;
  if (layout.total_slab_bytes > slab.size())
    throw std::out_of_range("Qwen UMA checkpoint slab too small");

  const auto ensure_copy_pso = [&]() {
    const std::string lib_key = "qwen.forward.library.qwen_forward";
    p.context.register_library(
        {.key = lib_key,
         .path = p.libraries / "omlx_m5_qwen_forward.metallib"});
    static_cast<void>(p.context.warmup_pipeline(
        {.key = "qwen.forward.omlx_m5_forward_copy_bytes",
         .library_key = lib_key,
         .function_name = "omlx_m5_forward_copy_bytes",
         .variant = "fixed-qwen",
         .support_indirect_command_buffers =
             p.context.capabilities().profile.compute_icb_eligible}));
  };

  std::vector<MetalCommandInvocation> invocations;
  invocations.reserve(2);

  if (!layout.gdn_refs.empty()) {
    if (!p.uma_gdn_copy_plan_.has_value()) {
      ensure_copy_pso();
      MetalCommandPlanDescriptor desc;
      desc.key = "qwen.uma.gdn_copy." + std::to_string(p.owner_id);
      desc.icb_qualified = false;
      desc.dispatches.reserve(layout.gdn_refs.size());
      for (std::size_t i = 0; i < layout.gdn_refs.size(); ++i) {
        const U64 sz = layout.gdn_refs[i].bytes;
        if (sz == 0ULL || sz > UINT32_MAX)
          throw std::overflow_error(
              "Qwen UMA recurrent state buffer exceeds 4GiB copy limit");
        MetalDispatchDescriptor d;
        d.pipeline_key = "qwen.forward.omlx_m5_forward_copy_bytes";
        d.kind = MetalDispatchKind::threads;
        d.barrier_before = false;
        d.grid = {.width = (sz + 15ULL) / 16ULL, .height = 1, .depth = 1};
        d.threads_per_threadgroup = {.width = 256, .height = 1, .depth = 1};
        d.resources = {
            {.argument_index = 0,
             .resource_slot = U32(2 * i),
             .minimum_bytes = sz},
            {.argument_index = 1,
             .resource_slot = U32(2 * i + 1),
             .minimum_bytes = sz},
        };
        d.constant_arguments = {{.argument_index = 2, .bytes = words({U32(sz)})}};
        desc.dispatches.push_back(std::move(d));
      }
      p.uma_gdn_copy_plan_ = p.context.prepare_command_plan(std::move(desc));
    }

    std::vector<MetalBufferBinding> gdn_bindings;
    gdn_bindings.reserve(layout.gdn_refs.size() * 2);
    for (std::size_t i = 0; i < layout.gdn_refs.size(); ++i) {
      const auto &ref = layout.gdn_refs[i];
      const U64 slab_off = layout.gdn_slab_offsets[i];
      if (pack_to_slab) {
        gdn_bindings.push_back({U32(2 * i), ref.buffer, ref.offset});
        gdn_bindings.push_back({U32(2 * i + 1), &slab, slab_off});
      } else {
        gdn_bindings.push_back({U32(2 * i), &slab, slab_off});
        gdn_bindings.push_back({U32(2 * i + 1), ref.buffer, ref.offset});
      }
    }
    invocations.push_back({.plan = &*p.uma_gdn_copy_plan_,
                           .bindings = std::move(gdn_bindings),
                           .parameters = nullptr});
  }

  if (layout.feature_bytes > 0ULL) {
    if (layout.feature_bytes > UINT32_MAX || !p.feature_history.buffer ||
        p.feature_history.offset + layout.feature_bytes >
            p.feature_history.buffer->size()) {
      throw std::overflow_error(
          "Qwen UMA feature_history exceeds copy bounds");
    }
    const U64 exact_threads = (layout.feature_bytes + 15ULL) / 16ULL;
    const U64 bucket_threads =
        std::max<U64>(256ULL, std::bit_ceil(exact_threads));
    auto plan_it = p.uma_feat_copy_plans_.find(bucket_threads);
    if (plan_it == p.uma_feat_copy_plans_.end()) {
      ensure_copy_pso();
      MetalCommandPlanDescriptor desc;
      desc.key = "qwen.uma.feat_copy." + std::to_string(p.owner_id) + ".g" +
                 std::to_string(bucket_threads);
      desc.icb_qualified = false;
      MetalDispatchDescriptor d;
      d.pipeline_key = "qwen.forward.omlx_m5_forward_copy_bytes";
      d.kind = MetalDispatchKind::threads;
      d.barrier_before = false;
      d.grid = {.width = bucket_threads, .height = 1, .depth = 1};
      d.threads_per_threadgroup = {.width = 256, .height = 1, .depth = 1};
      d.resources = {
          {.argument_index = 0, .resource_slot = 0, .minimum_bytes = 1},
          {.argument_index = 1, .resource_slot = 1, .minimum_bytes = 1},
      };
      d.dynamic_arguments = {
          {.argument_index = 2, .parameter_offset = 0, .minimum_bytes = 4}};
      desc.dispatches.push_back(std::move(d));
      plan_it = p.uma_feat_copy_plans_
                    .emplace(bucket_threads,
                             p.context.prepare_command_plan(std::move(desc)))
                    .first;
    }
    if (!p.uma_feat_copy_params_.has_value()) {
      p.uma_feat_copy_params_ = p.context.make_parameter_table(
          {.key = "qwen.uma.feat_params." + std::to_string(p.owner_id),
           .byte_size = 256,
           .fields = {{.key = "count", .offset = 0, .size = 4, .alignment = 4}}});
    }
    p.uma_feat_copy_params_->write("count", words({U32(layout.feature_bytes)}));
    std::vector<MetalBufferBinding> feat_bindings;
    feat_bindings.reserve(2);
    if (pack_to_slab) {
      feat_bindings.push_back(
          {0U, p.feature_history.buffer, p.feature_history.offset});
      feat_bindings.push_back({1U, &slab, layout.feature_slab_offset});
    } else {
      feat_bindings.push_back({0U, &slab, layout.feature_slab_offset});
      feat_bindings.push_back(
          {1U, p.feature_history.buffer, p.feature_history.offset});
    }
    invocations.push_back({.plan = &plan_it->second,
                           .bindings = std::move(feat_bindings),
                           .parameters = &*p.uma_feat_copy_params_});
  }

  if (!invocations.empty()) {
    static_cast<void>(p.context.execute_batch(invocations));
  }
}

} // namespace

QwenUmaContinuation QwenForward::checkpoint_uma() const {
  const auto &p = *impl_;
  p.require_no_async_in_flight("checkpoint_uma");
  if (p.pending_verify || !p.next_position)
    throw std::logic_error("Qwen checkpoint requires committed prefill");
  const auto layout = compute_uma_slab_layout(p, p.next_position);
  auto pool = p.uma_pool_shared();

  QwenUmaContinuation snapshot;
  snapshot.execution_identity = p.execution_identity;
  snapshot.owner_id = p.owner_id;
  snapshot.position = p.next_position;
  snapshot.recurrent_bytes = layout.recurrent_bytes;
  snapshot.feature_bytes = layout.feature_bytes;
  snapshot.slab_bytes = layout.total_slab_bytes;
  snapshot.accounted_bytes = non_kv_checkpoint_bytes();

  if (layout.total_slab_bytes > 0ULL) {
    // When DFlash2 feature_history is present, align slab capacity to a coarse
    // quantum (32 MiB on real models, 64 KiB on small test fixtures) so
    // monotonic multi-turn growth resizes at most 1-2 times instead of every
    // turn while preserving exact slab recycling.
    const U64 slab_quantum =
        layout.feature_bytes == 0ULL
            ? 4096ULL
            : (p.h >= 1024U ? (32ULL << 20) : (64ULL << 10));
    snapshot.uma_slab =
        pool->acquire(p.context, layout.total_slab_bytes, slab_quantum);
    try {
      execute_uma_slab_copy(p, layout, *snapshot.uma_slab, true);
    } catch (...) {
      pool->recycle(std::move(snapshot.uma_slab));
      throw;
    }
  }
  try {
    if (p.page_pool) {
      snapshot.kv_lease = p.page_pool->retain_prefix();
      snapshot.kv_pages = p.page_pool->mapped_slots();
      p.lease_pages_[snapshot.kv_lease] = snapshot.kv_pages;
    }
    const U64 epoch = pool->next_epoch.fetch_add(1ULL, std::memory_order_relaxed);
    snapshot.uma_epoch_id = epoch ? epoch : 1ULL;
    p.uma_live_epochs_.insert(snapshot.uma_epoch_id);
  } catch (...) {
    if (snapshot.kv_lease && p.page_pool) {
      try {
        p.page_pool->release_prefix(snapshot.kv_lease);
      } catch (...) {
      }
      p.lease_pages_.erase(snapshot.kv_lease);
    }
    pool->recycle(std::move(snapshot.uma_slab));
    throw;
  }
  return snapshot;
}

void QwenForward::restore_uma(const QwenUmaContinuation &snapshot) {
  auto &p = *impl_;
  if (p.pending_verify || snapshot.owner_id != p.owner_id ||
      snapshot.execution_identity != p.execution_identity ||
      snapshot.uma_epoch_id == 0ULL ||
      !p.uma_live_epochs_.contains(snapshot.uma_epoch_id) ||
      !snapshot.position || snapshot.position > p.max_context ||
      bool(snapshot.kv_lease) != bool(p.page_pool))
    throw std::invalid_argument(
        "Qwen UMA continuation identity/epoch/frontier invalid");
  const auto layout = compute_uma_slab_layout(p, snapshot.position);
  if (snapshot.recurrent_bytes != layout.recurrent_bytes ||
      snapshot.feature_bytes != layout.feature_bytes ||
      snapshot.slab_bytes != layout.total_slab_bytes)
    throw std::invalid_argument("Qwen UMA continuation geometry mismatch");
  if (layout.total_slab_bytes > 0ULL &&
      (!snapshot.uma_slab ||
       snapshot.uma_slab->size() < layout.total_slab_bytes))
    throw std::invalid_argument("Qwen UMA continuation slab invalid");

  try {
    p.drain_and_invalidate_async_steps();
    if (p.page_pool)
      p.page_pool->restore_prefix(snapshot.kv_lease);
    if (layout.total_slab_bytes > 0ULL)
      execute_uma_slab_copy(p, layout, *snapshot.uma_slab, false);
    p.ar_in_flight_count_ = 0U;
    p.ar_pipe_cursor_ = 0U;
    p.prefill_in_flight_count_ = 0U;
    p.prefill_pipe_cursor_ = 0U;
    p.verify_async_profile_ = nullptr;
    p.last_trace_tokens = 0U;
    p.next_position = snapshot.position;
    p.last_logits_row = 0;
    p.has_last_logits = false;
    ++p.feature_epoch_;
  } catch (...) {
    p.reset();
    throw;
  }
}

void QwenForward::restore_shared_uma(const QwenUmaContinuation &snapshot,
                                     QwenForward *source_owner) {
  if (source_owner == this) {
    restore_uma(snapshot);
    return;
  }
  auto &p = *impl_;
  if (!source_owner)
    throw std::invalid_argument(
        "Qwen UMA cross-slot restore requires source_owner");
  const auto &src = *source_owner->impl_;
  if (p.pending_verify || snapshot.owner_id != src.owner_id ||
      snapshot.execution_identity != src.execution_identity ||
      p.shared_contract_identity != src.shared_contract_identity ||
      snapshot.uma_epoch_id == 0ULL ||
      !src.uma_live_epochs_.contains(snapshot.uma_epoch_id) ||
      !snapshot.position || snapshot.position > p.max_context ||
      bool(snapshot.kv_lease) != bool(p.page_pool))
    throw std::invalid_argument(
        "Qwen UMA cross-slot continuation identity/epoch invalid");
  const auto layout = compute_uma_slab_layout(p, snapshot.position);
  if (snapshot.recurrent_bytes != layout.recurrent_bytes ||
      snapshot.feature_bytes != layout.feature_bytes ||
      snapshot.slab_bytes != layout.total_slab_bytes)
    throw std::invalid_argument(
        "Qwen UMA cross-slot continuation geometry mismatch");
  if (layout.total_slab_bytes > 0ULL &&
      (!snapshot.uma_slab ||
       snapshot.uma_slab->size() < layout.total_slab_bytes))
    throw std::invalid_argument("Qwen UMA cross-slot continuation slab invalid");

  try {
    p.drain_and_invalidate_async_steps();
    if (p.page_pool) {
      const auto lease_it = src.lease_pages_.find(snapshot.kv_lease);
      if (lease_it == src.lease_pages_.end())
        throw std::invalid_argument("Qwen source KV lease no longer active");
      const auto &src_pages = lease_it->second;
      const U32 num_pages = (snapshot.position + 63U) / 64U;
      if (src_pages.size() != num_pages)
        throw std::invalid_argument("Qwen source KV lease page count mismatch");

      if (p.page_pool->shared_allocator() != nullptr && src.page_pool &&
          p.page_pool->shared_allocator() == src.page_pool->shared_allocator()) {
        p.page_pool->install_shared_prefix(snapshot.position, src_pages);
      } else {
        p.page_pool->reset();
        static_cast<void>(p.page_pool->begin_append(snapshot.position));
        p.page_pool->commit(snapshot.position);
        const auto dst_pages = p.page_pool->mapped_slots();

        const U64 page_bytes = kv_page_bytes(p.kv_format_, 64U,
                                             U32(p.config.num_key_value_heads),
                                             U32(p.config.head_dim));
        static std::atomic<U64> restore_uma_seq{1};
        Encoder enc(
            p.context, p.libraries, 1,
            "qwen.restore_shared_uma_kv." +
                std::to_string(
                    restore_uma_seq.fetch_add(1, std::memory_order_relaxed)));
        enc.begin_stage("copy_kv");
        for (std::size_t l = 0; l < p.blocks.size(); ++l) {
          if (p.blocks[l].type != LayerType::full_attention)
            continue;
          for (U32 pg = 0; pg < num_pages; ++pg) {
            const U64 s_off = U64(src_pages[pg]) * page_bytes;
            const U64 d_off = U64(dst_pages[pg]) * page_bytes;
            auto &d0 = enc.emit(
                "qwen_forward", "omlx_m5_forward_copy_bytes",
                {.width = (page_bytes + 15) / 16}, {.width = 256},
                {Ref{src.blocks[l].state0.buffer,
                     src.blocks[l].state0.offset + s_off, page_bytes},
                 Ref{p.blocks[l].state0.buffer,
                     p.blocks[l].state0.offset + d_off, page_bytes}});
            enc.dynamic(d0, 2, words({U32(page_bytes)}));
            auto &d1 = enc.emit(
                "qwen_forward", "omlx_m5_forward_copy_bytes",
                {.width = (page_bytes + 15) / 16}, {.width = 256},
                {Ref{src.blocks[l].state1.buffer,
                     src.blocks[l].state1.offset + s_off, page_bytes},
                 Ref{p.blocks[l].state1.buffer,
                     p.blocks[l].state1.offset + d_off, page_bytes}});
            enc.dynamic(d1, 2, words({U32(page_bytes)}));
          }
        }
        auto cp_prof = enc.finish();
        const std::array calls{
            MetalCommandInvocation{.plan = &cp_prof.plan,
                                   .bindings = cp_prof.bindings,
                                   .parameters = &cp_prof.parameters}};
        static_cast<void>(p.context.execute_batch(calls));
      }
    }

    if (layout.total_slab_bytes > 0ULL)
      execute_uma_slab_copy(p, layout, *snapshot.uma_slab, false);
    p.ar_in_flight_count_ = 0U;
    p.ar_pipe_cursor_ = 0U;
    p.prefill_in_flight_count_ = 0U;
    p.prefill_pipe_cursor_ = 0U;
    p.verify_async_profile_ = nullptr;
    p.last_trace_tokens = 0U;
    p.next_position = snapshot.position;
    p.last_logits_row = 0;
    p.has_last_logits = false;
    ++p.feature_epoch_;
  } catch (...) {
    p.reset();
    throw;
  }
}

void QwenForward::release_uma(QwenUmaContinuation &snapshot) {
  auto &p = *impl_;
  if (snapshot.owner_id != p.owner_id)
    throw std::invalid_argument(
        "Qwen UMA continuation belongs to another owner");
  if (snapshot.kv_lease) {
    if (!p.page_pool)
      throw std::logic_error(
          "Qwen UMA continuation belongs to a different owner");
    p.page_pool->release_prefix(snapshot.kv_lease);
    p.lease_pages_.erase(snapshot.kv_lease);
    snapshot.kv_lease = 0;
  }
  if (snapshot.uma_epoch_id) {
    p.uma_live_epochs_.erase(snapshot.uma_epoch_id);
    snapshot.uma_epoch_id = 0;
  }
  if (snapshot.uma_slab) {
    p.uma_pool_shared()->recycle(std::move(snapshot.uma_slab));
    snapshot.uma_slab.reset();
  }
  snapshot.kv_pages.clear();
  snapshot.owner_id = 0;
  snapshot.position = 0;
  snapshot.recurrent_bytes = 0;
  snapshot.feature_bytes = 0;
  snapshot.slab_bytes = 0;
  snapshot.accounted_bytes = 0;
}

void QwenForward::trim_uma_checkpoint_pool(std::size_t max_pooled_slabs) {
  if (impl_->uma_ckpt_pool_) {
    impl_->uma_ckpt_pool_->trim(max_pooled_slabs);
  } else if (impl_->shared_primary_ && impl_->shared_primary_->impl_ &&
             impl_->shared_primary_->impl_->uma_ckpt_pool_) {
    impl_->shared_primary_->impl_->uma_ckpt_pool_->trim(max_pooled_slabs);
  }
}
U64 QwenForward::diagnostic_retain_kv_pages() {
  if (!impl_->page_pool)
    throw std::logic_error("Qwen has no KV page pool");
  return impl_->page_pool->retain_prefix();
}
void QwenForward::diagnostic_release_kv_pages(U64 lease) {
  if (!impl_->page_pool)
    throw std::logic_error("Qwen has no KV page pool");
  impl_->page_pool->release_prefix(lease);
}

} // namespace omlx::m5
