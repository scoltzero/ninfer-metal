#include "omlx_m5/qwen_forward.h"
#include "affine_nax_cache.h"
#include "attention_nax.h"
#include "bf16_nax.h"
#include "forward_plan.h"
#include "gdn_prefill.h"
#include "lora_runtime.h"
#include "moe_nax.h"
#include "mxfp4_nax_cache.h"
#include "silicon_tuning.h"
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
          if (w.layout == "oq_nax_tiled_g64_v1" ||
              w.layout == "oq_nax_tiled_g128_v1") {
            src_h2 = plane(w, "nax_high2");
          } else if (tw_existing && tw_existing->bits == 5U &&
                     tw_existing->bn == bn && tw_existing->n == w.row_count &&
                     tw_existing->k == w.in_features) {
            src_h2 = tw_existing->high2;
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
      if ((w.layout == "oq_nax_tiled_g64_v1" ||
           w.layout == "oq_nax_tiled_g128_v1") &&
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
        };
      }
    }
  }
  if (w.layout == "oq_nax_tiled_g64_v1" || w.layout == "oq_nax_tiled_g128_v1") {
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
       std::shared_ptr<const QwenBypassLoraRuntime> bypass_lora = nullptr)
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
        moe_override(std::move(moe_override)) {
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
    allocation_plan = plan_qwen_forward_memory(
        c, max_context, U32(feature_layers.size()), inspect_routes, speculative,
        enable_attention_split,
        sampling_contract == SamplingContract::PositionCoupledV1,
        attention_policy == omlx::m5::AttentionNumericPolicy::RowExactV1,
        inspect_tensors_workspace, 0U, workspace_rows_, affine4_kv, 0U,
        flash_nax_prefill_);
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
    bool has_pretiled_affine = (head.layout == "oq_nax_tiled_g64_v1" ||
                                head.layout == "oq_nax_tiled_g128_v1");
    candidates.push_back(head);
    for (const auto &block : blocks) {
      for (const auto &[role, w] : block.operands) {
        (void)role;
        has_pretiled_mxfp4 |= (w.layout == "mx_nax_tiled_bk128_v1");
        has_pretiled_affine |= (w.layout == "oq_nax_tiled_g64_v1" ||
                                w.layout == "oq_nax_tiled_g128_v1");
        candidates.push_back(w);
      }
    }

    has_row_exact_metallib_ =
        std::filesystem::is_regular_file(libraries / "omlx_m5_row_exact.metallib");
    const void *cache_owner_key =
        merged_lora_ ? static_cast<const void *>(merged_lora_.get())
                     : static_cast<const void *>(&weights);

    if (mxfp4_nax_runtime_eligible(context, libraries) &&
        (linear_policy == LinearNumericPolicy::RowExactPortableV1 ||
         mxfp4_w4a4_prefill_enabled() || has_pretiled_mxfp4 ||
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
         has_pretiled_affine ||
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
    base_shared_contract_identity_ = execution_identity_digest(base_fields);
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

  bool resolve_pretiled_affine_t1(const MaterializedWeightView &w, U32 t,
                                  Ref *out_codes = nullptr,
                                  Ref *out_high = nullptr,
                                  Ref *out_sb = nullptr) const {
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
        return true;
      }
    }
    if (w.layout == "oq_nax_tiled_g64_v1" ||
        w.layout == "oq_nax_tiled_g128_v1") {
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
                                   Ref *out_sb = nullptr) const {
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
        return true;
      }
    }
    if (w.layout == "oq_nax_tiled_g64_v1" ||
        w.layout == "oq_nax_tiled_g128_v1") {
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
    if (w.layout == "oq_nax_tiled_g64_v1" || w.layout == "oq_nax_tiled_g128_v1")
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
                              silicon_tuning.gpu_cores);
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
                         silicon_tuning.gpu_cores);
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
        linear_affine_nax(e, *tw, x, affine_nax_xs, y, explicit_t,
                          silicon_tuning.gpu_cores);
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
      linear_bf16_nax(e, w, x, y, n, k, explicit_t, silicon_tuning.gpu_cores);
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
        e.dynamic(d, 4, words({n, k, U32(w.padded_k), t}),
                  explicit_t.has_value() ? std::optional<U32>{}
                                         : std::optional<U32>{3U});
        e.record_weight(w, n, k, "pretiled_qmv_t1", "pretiled_mxfp4_qmv_t1_v3",
                        {}, "legacy_route_v1", {}, false, explicit_t);
        return;
      }
      Ref aff_c{}, aff_h{}, aff_sb{};
      if (resolve_pretiled_affine_t1(w, t, &aff_c, &aff_h, &aff_sb)) {
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
        e.dynamic(d, 5, words({n, t, 0U, k, bits}),
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
    detail::forward::linear(e, w, x, y, n, k, sums, zero, decision.route,
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
                                  Ref *out_sb = nullptr) const {
    if (!(!speculative_enabled && t_domain_active() &&
          pretiled_qmv_t1_enabled() && has_row_exact_metallib_ &&
          !is_overridden(w) && h >= 512U)) {
      return false;
    }
    return resolve_affine_planes_any_t(w, out_codes, out_high, out_sb);
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
      e.dynamic(d, 4, words({n, k, U32(w.padded_k), 2U}));
      e.record_weight(w, n, k, "pretiled_qmv_m2", "pretiled_mxfp4_qmv_m2_v1",
                      {}, "legacy_route_v1", {}, false, 2U);
      return true;
    }
    Ref aff_c{}, aff_h{}, aff_sb{};
    if (resolve_pretiled_affine_m2(w, &aff_c, &aff_h, &aff_sb)) {
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
      e.dynamic(d, 5, words({n, 2U, 0U, k, bits}));
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
      e.dynamic(d, 5, fused_norm_params(n, 0U, 0U, 0U, k, 0U, sq_slot));
      e.record_weight(w, n, k, "pretiled_qmv_m2",
                      "pretiled_mxfp4_qmv_add_sq_m2_v1", {}, "legacy_route_v1",
                      {}, false, 2U);
      return true;
    }
    Ref aff_c{}, aff_h{}, aff_sb{};
    if (resolve_pretiled_affine_m2(w, &aff_c, &aff_h, &aff_sb)) {
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
      e.dynamic(d, 6, fused_norm_params(n, 0U, 0U, 0U, k, bits, sq_slot));
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
      e.dynamic(d, 6, fused_norm_params(n, 0U, 0U, 0U, k, 0U, sq_slot));
      e.record_weight(w, n, k, "pretiled_qmv_m2",
                      "pretiled_mxfp4_qmv_norm_m2_v1", {}, "legacy_route_v1",
                      {}, false, 2U);
      return true;
    }
    Ref aff_c{}, aff_h{}, aff_sb{};
    if (resolve_pretiled_affine_m2(w, &aff_c, &aff_h, &aff_sb)) {
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
      e.dynamic(d, 7, fused_norm_params(n, 0U, 0U, 0U, k, bits, sq_slot));
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
        e.dynamic(d, 8, fused_norm_params(n, 0U, 0U, 0U, k, 0U, *sq_slot));
      } else {
        const char *fn =
            (n >= 2048U) ? "omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r4_m2"
                         : "omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r2_m2";
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U}, {x, gc, gs, uc, us, y});
        e.dynamic(d, 6, words({n, k, U32(gate_w.padded_k), 2U}));
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
    if (resolve_pretiled_affine_m2(gate_w, &agc, &agh, &agsb) &&
        resolve_pretiled_affine_m2(up_w, &auc, &auh, &ausb)) {
      const U32 g_bits = affine_nax_bits(gate_w.format);
      const U32 u_bits = affine_nax_bits(up_w.format);
      const U32 tgs = (n + 3U) / 4U;
      const U32 packed_bits = g_bits | (u_bits << 8U);
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
                  fused_norm_params(n, 0U, 0U, 0U, k, packed_bits, *sq_slot));
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
        e.dynamic(d, 8, words({n, 2U, 0U, k, packed_bits}));
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
    if (resolve_pretiled_affine_m2(parts[0], &d0, &h0, &s0) &&
        resolve_pretiled_affine_m2(parts[1], &d1, &h1, &s1) &&
        resolve_pretiled_affine_m2(parts[2], &d2, &h2, &s2)) {
      const U32 b0 = affine_nax_bits(parts[0].format);
      const U32 b1 = affine_nax_bits(parts[1].format);
      const U32 b2 = affine_nax_bits(parts[2].format);
      const U32 packed_bits = b0 | (b1 << 8U) | (b2 << 16U);
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
                  fused_norm_params(ns[0], ns[1], ns[2], 0U, k, packed_bits,
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
        e.dynamic(d, 13, words({ns[0], ns[1], ns[2], k, packed_bits}));
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
    if (resolve_pretiled_affine_m2(parts[0], &d0, &h0, &s0) &&
        resolve_pretiled_affine_m2(parts[1], &d1, &h1, &s1) &&
        resolve_pretiled_affine_m2(parts[2], &d2, &h2, &s2) &&
        resolve_pretiled_affine_m2(parts[3], &d3, &h3, &s3)) {
      const U32 b0 = affine_nax_bits(parts[0].format);
      const U32 b1 = affine_nax_bits(parts[1].format);
      const U32 b2 = affine_nax_bits(parts[2].format);
      const U32 b3 = affine_nax_bits(parts[3].format);
      const U32 packed_bits = b0 | (b1 << 8U) | (b2 << 16U) | (b3 << 24U);
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
                  fused_norm_params(ns[0], ns[1], ns[2], ns[3], k, packed_bits,
                                    *sq_slot));
      } else {
        const char *fn =
            all_q456 ? "omlx_m5_pretiled_qmv_affine_concat4_q456_r2_m2"
                     : "omlx_m5_pretiled_qmv_affine_concat4_r2_m2";
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U},
                         {x, d0, h0, s0, d1, h1, s1, d2, h2, s2, d3, h3, s3, y0,
                          y1, y2, y3});
        e.dynamic(d, 17, words({ns[0], ns[1], ns[2], ns[3], k, packed_bits}));
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
    return words({n0, n1, n2, n3, k_in, bits_packed, sq_slot,
                  word(1.0F / float(k_in)), word(float(config.rms_norm_eps)),
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
      e.dynamic(d, 5, fused_norm_params(n, 0U, 0U, 0U, k, 0U, sq_slot));
      e.record_weight(w, n, k, "pretiled_qmv_t1",
                      "pretiled_mxfp4_qmv_add_sq_t1_v4", {}, "legacy_route_v1",
                      {}, false);
      return true;
    }
    Ref aff_c{}, aff_h{}, aff_sb{};
    if (resolve_pretiled_affine_t1(w, 1U, &aff_c, &aff_h, &aff_sb)) {
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
      e.dynamic(d, 6, fused_norm_params(n, 0U, 0U, 0U, k, bits, sq_slot));
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
      e.dynamic(d, 6, fused_norm_params(n, 0U, 0U, 0U, k, 0U, sq_slot));
      e.record_weight(w, n, k, "pretiled_qmv_t1",
                      "pretiled_mxfp4_qmv_norm_t1_v4", {}, "legacy_route_v1",
                      {}, false);
      return true;
    }
    Ref aff_c{}, aff_h{}, aff_sb{};
    if (resolve_pretiled_affine_t1(w, 1U, &aff_c, &aff_h, &aff_sb)) {
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
      e.dynamic(d, 7, fused_norm_params(n, 0U, 0U, 0U, k, bits, sq_slot));
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
    if (resolve_pretiled_affine_t1(gate_w, 1U, &agc, &agh, &agsb) &&
        resolve_pretiled_affine_t1(up_w, 1U, &auc, &auh, &ausb)) {
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
      e.dynamic(d, 10,
                fused_norm_params(n, 0U, 0U, 0U, k, packed_bits, sq_slot));
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
      if (resolve_pretiled_affine_t1(gate_w, t, &agc, &agh, &agsb) &&
          resolve_pretiled_affine_t1(up_w, t, &auc, &auh, &ausb)) {
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
        e.dynamic(d, 8, words({n, t, 0U, k, packed_bits}),
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
    if (gate_w.format != up_w.format) {
      return false;
    }
    if (explicit_t.has_value() && t > e.profile.capacity) {
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
    if (use_affine_nax_for_weight(gate_w, sel_t) &&
        use_affine_nax_for_weight(up_w, sel_t)) {
      const auto *gw = affine_nax_cache->find(gate_w.object_id);
      const auto *uw = affine_nax_cache->find(up_w.object_id);
      if (gw && uw && gw->bits == uw->bits && gw->bn == 16u && uw->bn == 16u &&
          gw->n == n && gw->k == k && uw->n == n && uw->k == k &&
          gw->split_k == uw->split_k) {
        linear_affine_nax_swiglu(e, *gw, *uw, x, affine_nax_xs, y,
                                 silicon_tuning.gpu_cores, explicit_t);
        const std::string grp = affine_nax_group(
            e, *gw, explicit_t, silicon_tuning.gpu_cores, uw->group_size);
        e.record_weight(gate_w, n, k, "affine_nax_m5_v1", "affine_nax_m5_v1",
                        {}, "linear_row_exact_portable_v1", grp, true,
                        explicit_t);
        e.record_weight(up_w, n, k, "affine_nax_m5_v1", "affine_nax_m5_v1", {},
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
        gate_w.layout == "oq_nax_tiled_g64_v1" ||
        gate_w.layout == "oq_nax_tiled_g128_v1") {
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
    if (!use_mxfp4_nax_for_weight(gw_v, t) ||
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

  void emit_attention_state_step(
      Encoder &e, const Block &block, const Block &state_block,
      KvPagePool &slot_pool, U32 slot_max_context, KvStorageFormat slot_kv_fmt,
      Ref q_ref, Ref k_ref, Ref v_ref, Ref mix_ref, Ref attn_pos_ref, U32 t,
      bool split, U32 context_envelope, bool dynamic_tokens,
      std::optional<U32> pos_placeholder = std::nullopt,
      std::vector<std::pair<U64, U32>> *pos_patches = nullptr) {
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

    if (slot_kv_fmt == KvStorageFormat::Affine4Group128) {
      if (affine4_nax_ok) {
        if (!fuse_qk_kv_append_affine4_t1) {
          emit_paged_kv_append_affine4_g128(
              e, c, t, k_ref, v_ref, state_block.state0, state_block.state1,
              table_ref, params, tok_word_attn, pos_word_attn, pos_patches);
        }
        if (!split && (!exact_attention || c.hidden_size >= 1024) &&
            !pos_placeholder.has_value() && dynamic_tokens &&
            flash_nax_prefill_ && t >= kFlashNaxPrefillMinTokens &&
            c.head_dim == 256) {
          emit_paged_flash_prefill_affine4_g128(
              e, c, t, q_ref, gate_ref, state_block.state0, state_block.state1,
              mix_ref, table_ref, attn_pos_ref, std::move(params));
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
        (!exact_attention || c.hidden_size >= 1024) && flash_nax_prefill_ &&
        t >= kFlashNaxPrefillMinTokens && attn_pos_ref.buffer != nullptr &&
        flash_nax_prefill_runtime_eligible(context, libraries,
                                           U32(c.head_dim))) {
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
                 std::optional<U32> out_sq_slot = std::nullopt) {
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
        } else if (resolve_pretiled_affine_t1(parts[0], 1U, &d0, &h0, &s0) &&
                   resolve_pretiled_affine_t1(parts[1], 1U, &d1, &h1, &s1) &&
                   resolve_pretiled_affine_t1(parts[2], 1U, &d2, &h2, &s2)) {
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
              fused_norm_params(2 * q, kv, kv, 0U, h, packed_bits, sq_in));
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
    } else if (!fused_qkv_norm && t == 1U && !input_is_f16 &&
               parts[0].row_count == 2 * q && parts[1].row_count == kv &&
               parts[2].row_count == kv && parts[0].in_features == h &&
               parts[1].in_features == h && parts[2].in_features == h &&
               resolve_pretiled_affine_t1(parts[0], 1U, &d0, &h0, &s0) &&
               resolve_pretiled_affine_t1(parts[1], 1U, &d1, &h1, &s1) &&
               resolve_pretiled_affine_t1(parts[2], 1U, &d2, &h2, &s2)) {
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
      e.dynamic(d, 13, words({2 * q, kv, kv, h, packed_bits}));
      for (std::size_t i = 0; i < 3; ++i) {
        e.record_weight(parts[i], part_ns[i], h, "pretiled_qmv_t1",
                        "pretiled_affine_qmv_concat3_t1_v3", {},
                        "legacy_route_v1", {}, false);
      }
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
        split, context_envelope, true);
    if (io_sq_slot != nullptr && out_sq_slot.has_value() &&
        try_linear_add_sq_t1(e, block.get("attention_o"), mixer, hidden, h, q,
                             *out_sq_slot)) {
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
                           bool *out_fused_add_sq = nullptr) {
    if (out_fused_add_sq)
      *out_fused_add_sq = false;
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
    const auto *gdn_out_nax =
        (allow_f16_out && !has_bypass_lora(out_w) &&
         use_mxfp4_nax_for_weight(out_w, t))
            ? mxfp4_nax_cache->find(out_w.object_id)
            : nullptr;
    const bool out_f16 =
        (gdn_out_nax != nullptr && gdn_out_nax->n == h && gdn_out_nax->k == gv);
    const U32 mp = out_f16 ? mxfp4_nax_mp(t) : 0U;
    auto emit_gdn_out_bf16 = [&]() {
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
              gdn_chunk_nax_enabled_, params, mega_params, dynamic_tokens)) {
        if (out_f16) {
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
          U32(c.linear_num_value_heads), dynamic_tokens);
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
           std::optional<U32> out_sq_slot = std::nullopt) {
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
        } else if (resolve_pretiled_affine_t1(qkvz_parts[0], 1U, &qz_c0, &qz_h0,
                                              &qz_s0) &&
                   resolve_pretiled_affine_t1(qkvz_parts[1], 1U, &qz_c1, &qz_h1,
                                              &qz_s1) &&
                   resolve_pretiled_affine_t1(ab_parts[0], 1U, &ab_c0, &ab_h0,
                                              &ab_s0) &&
                   resolve_pretiled_affine_t1(ab_parts[1], 1U, &ab_c1, &ab_h1,
                                              &ab_s1)) {
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
                    fused_norm_params(conv, gv, hv, hv, h, packed_bits, sq_in));
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
      } else if (resolve_pretiled_affine_t1(qkvz_parts[0], 1U, &qz_c0, &qz_h0,
                                            &qz_s0) &&
                 resolve_pretiled_affine_t1(qkvz_parts[1], 1U, &qz_c1, &qz_h1,
                                            &qz_s1) &&
                 resolve_pretiled_affine_t1(ab_parts[0], 1U, &ab_c0, &ab_h0,
                                            &ab_s0) &&
                 resolve_pretiled_affine_t1(ab_parts[1], 1U, &ab_c1, &ab_h1,
                                            &ab_s1)) {
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
        e.dynamic(d, 17, words({conv, gv, hv, hv, h, packed_bits}));
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
      } else if (t == 1U && !input_is_f16 &&
                 qkvz_parts[0].row_count == conv &&
                 qkvz_parts[1].row_count == gv &&
                 qkvz_parts[0].in_features == h &&
                 qkvz_parts[1].in_features == h &&
                 resolve_pretiled_affine_t1(qkvz_parts[0], 1U, &qz_c0, &qz_h0,
                                            &qz_s0) &&
                 resolve_pretiled_affine_t1(qkvz_parts[1], 1U, &qz_c1, &qz_h1,
                                            &qz_s1)) {
        const U32 b0 = affine_nax_bits(qkvz_parts[0].format);
        const U32 b1 = affine_nax_bits(qkvz_parts[1].format);
        const U32 packed_bits = b0 | (b1 << 8U);
        const U32 rows_per_tg = ((conv + gv) >= 4096U) ? 8U : 4U;
        const U32 tgs = ((conv + rows_per_tg - 1U) / rows_per_tg) +
                        ((gv + rows_per_tg - 1U) / rows_per_tg);
        auto &d = e.emit("row_exact", "omlx_m5_pretiled_qmv_affine_concat2_t1",
                         {.width = U64(tgs) * 64U}, {.width = 64U},
                         {norm, qz_c0, qz_h0, qz_s0, qz_c1, qz_h1, qz_s1, qkv, z});
        e.dynamic(d, 9, words({conv, gv, 0U, h, packed_bits}));
        for (std::size_t i = 0; i < 2; ++i) {
          e.record_weight(qkvz_parts[i], qkvz_ns[i], h, "pretiled_qmv_t1",
                          "pretiled_affine_qmv_concat2_t1_v3", {},
                          "legacy_route_v1", {}, false);
        }
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
      } else if (t == 1U && !input_is_f16 &&
                 ab_parts[0].row_count == hv && ab_parts[1].row_count == hv &&
                 ab_parts[0].in_features == h && ab_parts[1].in_features == h &&
                 resolve_pretiled_affine_t1(ab_parts[0], 1U, &ab_c0, &ab_h0,
                                            &ab_s0) &&
                 resolve_pretiled_affine_t1(ab_parts[1], 1U, &ab_c1, &ab_h1,
                                            &ab_s1)) {
        const U32 b0 = affine_nax_bits(ab_parts[0].format);
        const U32 b1 = affine_nax_bits(ab_parts[1].format);
        const U32 packed_bits = b0 | (b1 << 8U);
        const U32 rows_per_tg = ((2U * hv) >= 4096U) ? 8U : 4U;
        const U32 tgs = 2U * ((hv + rows_per_tg - 1U) / rows_per_tg);
        auto &d = e.emit("row_exact", "omlx_m5_pretiled_qmv_affine_concat2_t1",
                         {.width = U64(tgs) * 64U}, {.width = 64U},
                         {norm, ab_c0, ab_h0, ab_s0, ab_c1, ab_h1, ab_s1, a, b});
        e.dynamic(d, 9, words({hv, hv, 0U, h, packed_bits}));
        for (std::size_t i = 0; i < 2; ++i) {
          e.record_weight(ab_parts[i], ab_ns[i], h, "pretiled_qmv_t1",
                          "pretiled_affine_qmv_concat2_t1_v3", {},
                          "legacy_route_v1", {}, false);
        }
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
        &gdn_fused_sq);
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
                         std::optional<U32> explicit_t = {}) {
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
    if (t >= eff_moe_min_t &&
        emit_moe_nax_ffn(e, moe_nax_ws, gw, uw, dw, norm, route_ids,
                         route_scores, expert_output, routed, t, h, f, 8u, 256u,
                         silicon_tuning.gpu_cores, explicit_t,
                         fused_router_logits, /*emit_reduce=*/false)) {
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

      const Encoder::GroupSumKey d_sum_key{
          sums.buffer,    sums.offset,    product.buffer,
          product.offset, f,              d_group,
          t * 8U,         !explicit_t.has_value(),
          8U};
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
           bool allow_moe9_bt = false) {
    const auto t = explicit_t.value_or(e.profile.capacity);
    if (!config.num_experts) {
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
                           fuse_router_nax ? router : Ref{}, explicit_t)) {
      expert_linear(e, block.get("moe_expert_gate"), norm, gate, f, h, false,
                    explicit_t);
      expert_linear(e, block.get("moe_expert_up"), norm, up, f, h, false,
                    explicit_t);
      swiglu(e, t * 8, f, explicit_t);
      expert_linear(e, block.get("moe_expert_down"), product, expert_output, h,
                    f, true, explicit_t);
      wrote_expert_output = true;
    }
    U32 shared_f = U32(config.shared_expert_intermediate_size);
    if (!try_linear_swiglu(e, block.get("moe_shared_mlp_gate"),
                           block.get("moe_shared_mlp_up"), norm, product,
                           shared_f, h, explicit_t)) {
      linear_base(e, block.get("moe_shared_mlp_gate"), norm, gate, shared_f, h,
                  explicit_t);
      linear_base(e, block.get("moe_shared_mlp_up"), norm, up, shared_f, h,
                  explicit_t);
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
    linear(e, block.get("moe_shared_mlp_down"), product, shared, h, shared_f,
           explicit_t);
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
    return key;
  }

  Profile compile(U32 t, bool verify, bool split = false,
                  U32 context_envelope = 0, bool defer_sample = false,
                  Ref custom_sampled = {}, Ref custom_attn_pos = {},
                  Ref feed_next_token = {}, std::string key_tag = {},
                  bool trunk_only = false, Ref custom_token_ids = {}) {
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
    const bool allow_t1_fused_norms =
        (t == 1U && !verify && !defer_sample && !trunk_only &&
         !speculative_enabled && allow_tdom_t1() && t_domain_active() &&
         pretiled_qmv_t1_enabled() && !inspect_tensors_workspace &&
         rms_sq_buf.buffer != nullptr && moe_tg_done_buf.buffer != nullptr &&
         h < 65536U);
    const bool affine_xs_eligible =
        (!allow_t1_fused_norms && t > 1U && affine_nax_active() &&
         affine_nax_xs.buffer != nullptr && (h % 64U) == 0U);
    const U32 num_sq_words = U32(blocks.size()) * 2U + 4U;
    const U32 num_moe_words =
        config.num_experts ? (U32(blocks.size()) + 4U) : 0U;
    std::optional<U32> pending_sq_slot;
    std::optional<Encoder::GroupSumKey> carried_affine_xs;
    auto restore_carried_xs = [&]() {
      if (carried_affine_xs.has_value()) {
        e.set_group_sum(*carried_affine_xs);
        carried_affine_xs.reset();
      }
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
      const bool in_f16 = block_input_nax(block, t);
      const bool mlp_f16 = block_mlp_input_nax(block, t);
      const bool in_affine_xs = (!in_f16 && affine_xs_eligible);
      const bool mlp_affine_xs = (!mlp_f16 && affine_xs_eligible);
      const bool layer_fuse = allow_t1_fused_norms && !in_f16 && !mlp_f16;
      e.begin_stage(
          "layer." + std::to_string(layer_index) +
          (block.type == LayerType::full_attention ? ".attention" : ".gdn"));
      restore_carried_xs();
      if (layer_index == 0) {
        if (in_f16)
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
        if (in_f16)
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
                             : std::optional<U32>{});
      else
        gdn(e, block, verify, in_f16, layer_fuse ? &pending_sq_slot : nullptr,
            layer_fuse ? std::optional<U32>{2U * layer_index}
                       : std::optional<U32>{});
      e.begin_stage("layer." + std::to_string(layer_index) + ".mlp");
      if (!pending_sq_slot.has_value()) {
        if (mlp_f16)
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
                     : std::optional<U32>{});
      if (!pending_sq_slot.has_value()) {
        if (layer_index + 1 < blocks.size()) {
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
            can_batched_affine4_attn && flash_nax_prefill_ && !skey.verify &&
            max_w >= 16U;
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
            } else if (resolve_pretiled_affine_t1(parts[0], 1U, &d0, &h0,
                                                  &s0) &&
                       resolve_pretiled_affine_t1(parts[1], 1U, &d1, &h1,
                                                  &s1) &&
                       resolve_pretiled_affine_t1(parts[2], 1U, &d2, &h2,
                                                  &s2)) {
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
                        fused_norm_params(2 * q, kv, kv, 0U, h, packed_bits,
                                          sq_in));
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
            } else if (resolve_pretiled_affine_t1(parts[0], 1U, &d0, &h0,
                                                  &s0) &&
                       resolve_pretiled_affine_t1(parts[1], 1U, &d1, &h1,
                                                  &s1) &&
                       resolve_pretiled_affine_t1(parts[2], 1U, &d2, &h2,
                                                  &s2)) {
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
              e.dynamic(d, 13, words({2 * q, kv, kv, h, packed_bits}));
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
            } else if (resolve_pretiled_affine_t1(qkvz_parts[0], 1U, &qz_c0,
                                                  &qz_h0, &qz_s0) &&
                       resolve_pretiled_affine_t1(qkvz_parts[1], 1U, &qz_c1,
                                                  &qz_h1, &qz_s1) &&
                       resolve_pretiled_affine_t1(ab_parts[0], 1U, &ab_c0,
                                                  &ab_h0, &ab_s0) &&
                       resolve_pretiled_affine_t1(ab_parts[1], 1U, &ab_c1,
                                                  &ab_h1, &ab_s1)) {
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
                        fused_norm_params(conv, gv, hv, hv, h, packed_bits,
                                          sq_in));
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
            } else if (resolve_pretiled_affine_t1(qkvz_parts[0], 1U, &qz_c0,
                                                  &qz_h0, &qz_s0) &&
                       resolve_pretiled_affine_t1(qkvz_parts[1], 1U, &qz_c1,
                                                  &qz_h1, &qz_s1) &&
                       resolve_pretiled_affine_t1(ab_parts[0], 1U, &ab_c0,
                                                  &ab_h0, &ab_s0) &&
                       resolve_pretiled_affine_t1(ab_parts[1], 1U, &ab_c1,
                                                  &ab_h1, &ab_s1)) {
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
              e.dynamic(d, 17, words({conv, gv, hv, hv, h, packed_bits}));
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
            (gdn_out_nax != nullptr && gdn_out_nax->n == h &&
             gdn_out_nax->k == gv &&
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
    std::shared_ptr<const QwenBypassLoraRuntime> bypass_lora)
    : impl_(std::make_unique<Impl>(
          context, weights, layers, std::move(config), max_context, libraries,
          feature_layers, inspect_routes, speculative, enable_m4_short_routes,
          std::move(diagnostic_override), enable_attention_split, enable_batch,
          std::move(moe_override), linear_policy, sampling_contract,
          attention_policy, inspect_tensors_workspace, kv_dtype,
          shared_primary, enable_flash_nax_prefill, kv_pool_pages,
          std::move(merged_lora), std::move(bypass_lora))) {}
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
  auto found = std::find_if(pp.profiles.begin(), pp.profiles.end(),
                            [&](const Profile &prof) {
                              return tokens.size() <= prof.capacity;
                            });
  if (found == pp.profiles.end() || !found->icb.has_value())
    throw std::out_of_range("Qwen async prefill profile capacity unavailable");
  auto &profile = *found;

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
  return QwenForwardResult{
      .token = 0U,
      .tokens = st.count,
      .executed_layers = U32(p.blocks.size()),
      .host_read_bytes = 0U,
      .short_matrix_calls = st.profile->short_matrix_calls,
      .short_lane_calls = st.profile->short_lane_calls,
      .submission = std::move(sub),
  };
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

  auto &pp = p.prefill_pipe_[0];
  auto found = std::find_if(pp.profiles.begin(), pp.profiles.end(),
                            [&](const Profile &prof) {
                              return tokens.size() <= prof.capacity;
                            });
  if (found == pp.profiles.end())
    throw std::out_of_range("Qwen trunk prefill capacity unavailable");
  auto &profile = *found;

  const U32 count = U32(tokens.size());
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
  return QwenForwardResult{
      .token = 0U,
      .tokens = count,
      .executed_layers = U32(p.blocks.size()),
      .host_read_bytes = 0U,
      .short_matrix_calls = profile.short_matrix_calls,
      .short_lane_calls = profile.short_lane_calls,
      .submission = std::move(sub),
  };
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
  };
}
std::optional<ResolvedKl32AffinePlanes>
QwenForward::resolve_kl32_affine(const MaterializedWeightView &w) const {
  Ref c{}, h_pl{}, sb{};
  if (!impl_->resolve_affine_planes_any_t(w, &c, &h_pl, &sb))
    return std::nullopt;
  return ResolvedKl32AffinePlanes{
      .codes = to_slice(c),
      .high = to_slice(h_pl),
      .sb_tiled = to_slice(sb),
      .n = U32(w.row_count),
      .k = U32(w.in_features),
      .bn = affine_nax_bn(U32(w.row_count)),
      .bits = affine_nax_bits(w.format),
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
