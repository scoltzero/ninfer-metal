#include "omlx_m5/mtp_forward.h"

#include "affine_nax_cache.h"
#include "attention_nax.h"
#include "bf16_nax.h"
#include "forward_plan.h"
#include "moe_nax.h"
#include "mxfp4_nax_cache.h"
#include "silicon_tuning.h"
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
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace omlx::m5 {
using namespace detail::forward;

namespace {

constexpr U32 kMtpConcatQkvId = 0x4D545130U;

U32 mtp_proposal_k(const GpuSamplingOptions &sampling, bool tokens_only,
                   U32 vocab) noexcept {
  if (tokens_only || sampling.temperature <= 0.0F) {
    return 0U;
  }
  const U32 contract_cap =
      (sampling.contract == SamplingContract::LegacyRejectionV1) ? 32U : 64U;
  return std::min<U32>(sampling.top_k, std::min<U32>(contract_cap, vocab));
}

void populate_mtp_proposal_step(SelectorStep &step, U32 proposal_k,
                                const U32 *cids, const float *cscores,
                                float temperature, float top_p) {
  step.candidate_ids.resize(proposal_k);
  step.probabilities.resize(proposal_k);
  float max_s = -INFINITY;
  for (U32 i = 0U; i < proposal_k; ++i) {
    step.candidate_ids[i] = std::int32_t(cids[i]);
    max_s = std::max(max_s, cscores[i]);
  }
  // Match omlx_m5_position_sample_logits_stage2 FP32 arithmetic so the host
  // top_p cutoff stays consistent with the GPU-sampled support set.
  float total = 0.0F;
  for (U32 i = 0U; i < proposal_k; ++i) {
    const float e = std::exp((cscores[i] - max_s) / temperature);
    step.probabilities[i] = e;
    total += e;
  }
  if (total > 0.0F) {
    for (U32 i = 0U; i < proposal_k; ++i) {
      step.probabilities[i] /= total;
    }
  }
  if (top_p < 1.0F) {
    float cum = 0.0F;
    U32 keep = proposal_k;
    for (U32 i = 0U; i < proposal_k; ++i) {
      cum += step.probabilities[i];
      if (cum >= top_p) {
        keep = i + 1U;
        break;
      }
    }
    // Guard against 1-ULP GPU exp() vs host std::exp() drift on razor-edge
    // top_p cutoffs so step.token is never zeroed out (q_token == 0).
    for (U32 i = keep; i < proposal_k; ++i) {
      if (step.candidate_ids[i] == step.token) {
        keep = i + 1U;
        break;
      }
    }
    float kept_sum = 0.0F;
    for (U32 i = 0U; i < proposal_k; ++i) {
      if (i >= keep) {
        step.probabilities[i] = 0.0F;
      } else {
        kept_sum += step.probabilities[i];
      }
    }
    if (kept_sum > 0.0F) {
      for (U32 i = 0U; i < keep; ++i) {
        step.probabilities[i] /= kept_sum;
      }
    }
  }
}

bool mtp_kl32_enabled() noexcept {
  if (const char *dis = std::getenv("OMLX_M5_DISABLE_MTP_KL32")) {
    if (std::string_view(dis) == "1" || std::string_view(dis) == "true") {
      return false;
    }
  }
  if (const char *en = std::getenv("OMLX_M5_ENABLE_PRETILED_QMV_T1")) {
    return std::string_view(en) != "0" && !std::string_view(en).empty();
  }
  return true;
}

class MtpQ5High1Cache {
public:
  void ensure_built(const MetalContext &ctx, const std::filesystem::path &libs,
                    const std::vector<MaterializedWeightView> &candidates,
                    const AffineNaxCache *affine_cache) {
    if (!std::filesystem::exists(libs / "omlx_m5_row_exact.metallib")) {
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
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
            items.push_back(
                {w.object_id, src_h2, h1_bytes,
                 arena.plan_alloc(h1_bytes, "MTP Q5 T=1 compacted 1-bit high")});
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
                 arena.plan_alloc(sb_bytes, "MTP Affine QMV T=1 sb_tiled")});
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
                "mtp.q5.high1.compact." + std::to_string(++compact_id));
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
    return it == entries_.end() ? nullptr : &it->second;
  }

  const Ref *find_sb_tiled(U32 object_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = sb_tiled_entries_.find(object_id);
    return it == sb_tiled_entries_.end() ? nullptr : &it->second;
  }

  static std::shared_ptr<MtpQ5High1Cache>
  acquire(const void *owner_key, const MetalContext &ctx,
          const std::filesystem::path &libs,
          const std::vector<MaterializedWeightView> &candidates,
          const AffineNaxCache *affine_cache) {
    auto cache = nax_acquire_owner_cache<MtpQ5High1Cache>(
        owner_key, [](MtpQ5High1Cache &) {});
    cache->ensure_built(ctx, libs, candidates, affine_cache);
    return cache;
  }

private:
  mutable std::mutex mu_;
  std::vector<std::unique_ptr<MetalBuffer>> segments_;
  std::unordered_map<U32, Ref> entries_;
  std::unordered_map<U32, Ref> sb_tiled_entries_;
};

struct MtpSaveTrunkParamsHost {
  U32 hidden = 0;
  U32 src_row = 0;
  float epsilon = 0.0F;
  float weight_offset = 0.0F;
};
static_assert(sizeof(MtpSaveTrunkParamsHost) == 16);

struct MtpStemNormsParamsHost {
  U32 hidden = 0;
  U32 rows = 0;
  U32 cond_mode = 0;
  U32 write_concat = 0;
  U32 trunk_row_base = 0;
  float epsilon = 0.0F;
  float target_norm_offset = 0.0F;
  float emb_norm_offset = 0.0F;
  float hid_norm_offset = 0.0F;
};
static_assert(sizeof(MtpStemNormsParamsHost) == 36);

struct MtpFcDualParamsHost {
  U32 n = 0;
  U32 h = 0;
  U32 rows = 0;
  U32 reserved = 0;
};
static_assert(sizeof(MtpFcDualParamsHost) == 16);

struct MtpLatchParamsHost {
  U32 hidden = 0;
  U32 src_row = 0;
  U32 step_index = 0;
  U32 sampled_mode = 0;
  U32 write_chain = 0;
  U32 norm_from_sq = 0;
  U32 sq_slot = 0;
  float inv_k = 0.0F;
  float eps = 0.0F;
  float w_off = 0.0F;
};
static_assert(sizeof(MtpLatchParamsHost) == 40);

struct MtpStemNormsOffsetParamsHost {
  U32 hidden = 0;
  U32 rows = 0;
  U32 cond_mode = 0;
  U32 write_concat = 0;
  U32 trunk_row_base = 0;
  U32 io_row_base = 0;
  float epsilon = 0.0F;
  float target_norm_offset = 0.0F;
  float emb_norm_offset = 0.0F;
  float hid_norm_offset = 0.0F;
};
static_assert(sizeof(MtpStemNormsOffsetParamsHost) == 40);

struct MtpSlotLayoutParamsHost {
  U32 batch_size = 0;
  U32 q_cols = 0;
  U32 kv_cols = 0;
  U32 mixer_cols = 0;
  U32 hidden = 0;
  U32 pad0 = 0;
  U32 pad1 = 0;
  U32 pad2 = 0;
  std::array<U32, 8> fold_counts{};
  std::array<U32, 8> fold_offsets{};
};
static_assert(sizeof(MtpSlotLayoutParamsHost) == 96);

struct MtpLatchBatchParamsHost {
  U32 batch_size = 0;
  U32 hidden = 0;
  U32 step_index = 0;
  U32 write_chain = 0;
};
static_assert(sizeof(MtpLatchBatchParamsHost) == 16);

Ref mtp_row_slice_ref(Ref base, U32 row_off, U32 rows, U32 cols,
                      U32 elem_bytes = 2U) {
  const U64 off = mul(mul(U64(row_off), U64(cols)), U64(elem_bytes));
  const U64 len = mul(mul(U64(rows), U64(cols)), U64(elem_bytes));
  if (!base.buffer || off + len > base.bytes) {
    throw std::out_of_range("MTP packed batch row slice out of bounds");
  }
  return Ref{base.buffer, base.offset + off, len};
}

void mtp_push_unique(std::vector<U64> &dst, U64 off) {
  if (std::ranges::find(dst, off) == dst.end()) {
    dst.push_back(off);
  }
}

template <class StructT> std::vector<std::byte> struct_bytes(const StructT &s) {
  const auto raw = std::as_bytes(std::span(&s, 1));
  return {raw.begin(), raw.end()};
}

double compute_bf16_norm_mean(const MaterializedWeightView &w, U64 expected_len) {
  if (w.format != "direct_bf16_v1") {
    throw std::invalid_argument("MTP RMSNorm weight must be direct_bf16_v1: " +
                                w.name);
  }
  const auto p = plane(w, "data");
  if (p.bytes != mul(expected_len, 2U)) {
    throw std::invalid_argument("MTP RMSNorm weight geometry mismatch: " +
                                w.name);
  }
  const auto raw = p.buffer->read(p.offset, p.bytes);
  double sum = 0.0;
  for (U64 i = 0; i < expected_len; ++i) {
    std::uint16_t code = 0;
    std::memcpy(&code, raw.data() + 2U * i, 2U);
    const float val = std::bit_cast<float>(U32(code) << 16U);
    if (!std::isfinite(val)) {
      throw std::invalid_argument("MTP RMSNorm weight contains nonfinite value: " +
                                  w.name);
    }
    sum += double(val);
  }
  return sum / double(expected_len);
}

MaterializedWeightView resolve_module_or_weight(const MaterializedModel &model,
                                                const std::string &base_name) {
  if (model.has_object(base_name)) {
    return model.parent_view(base_name);
  }
  const std::string weight_name = base_name + ".weight";
  if (model.has_object(weight_name)) {
    return model.parent_view(weight_name);
  }
  throw std::invalid_argument("MTP required weight missing: " + base_name);
}

bool has_module_or_weight(const MaterializedModel &model,
                          const std::string &base_name) noexcept {
  return model.has_object(base_name) ||
         model.has_object(base_name + ".weight");
}

std::string detect_mtp_prefix(const MaterializedModel &model) {
  if (model.has_object("language_model.mtp.norm.weight")) {
    return "language_model.mtp.";
  }
  if (model.has_object("mtp.norm.weight")) {
    return "mtp.";
  }
  throw std::invalid_argument(
      "MaterializedModel contains no Checkpoint-Embedded MTP head weights");
}

const MaterializedWeightView *
find_stage_weight(const QwenLayerCatalog *catalog,
                  const MaterializedModel *target_weights, U32 layer_idx,
                  std::string_view role,
                  std::optional<MaterializedWeightView> &storage) {
  if (!catalog || !target_weights ||
      layer_idx >= catalog->layers().size()) {
    return nullptr;
  }
  for (const auto &st : catalog->layer(layer_idx).stages) {
    if (st.role == role && target_weights->has_object(st.object_id)) {
      storage.emplace(target_weights->parent_view(st.object_id));
      return &(*storage);
    }
  }
  return nullptr;
}

} // namespace

bool has_mtp_weights(const ArtifactModel &artifact) noexcept {
  return std::ranges::any_of(artifact.objects(), [](const ArtifactObject &obj) {
    return classify_model_component(obj.name) == ModelComponent::mtp;
  });
}

bool has_mtp_weights(const MaterializedModel &model) noexcept {
  if (!model.valid()) {
    return false;
  }
  return model.has_object("language_model.mtp.norm.weight") ||
         model.has_object("mtp.norm.weight");
}

MtpNormOffsets repair_legacy_head_norms(
    const MaterializedModel &mtp_weights, std::string_view mtp_prefix,
    const QwenModelConfig &config, float target_norm_offset,
    const MaterializedWeightView *anchor_final_norm,
    const MaterializedWeightView *anchor_input_norm,
    const MaterializedWeightView *anchor_post_attn_norm,
    const MaterializedWeightView *anchor_q_norm,
    const MaterializedWeightView *anchor_k_norm) {
  MtpNormOffsets out{};
  if (target_norm_offset >= 0.5F) {
    out.pre_fc_norm_embedding = 1.0F;
    out.pre_fc_norm_hidden = 1.0F;
    out.input_layernorm = 1.0F;
    out.q_norm = 1.0F;
    out.k_norm = 1.0F;
    out.post_attention_layernorm = 1.0F;
    out.final_norm = 1.0F;
    out.repaired_any = false;
    return out;
  }

  const U64 h = U64(config.hidden_size);
  const U64 hd = U64(config.head_dim);
  const std::string p(mtp_prefix);

  const double mean_anchor_final =
      anchor_final_norm ? compute_bf16_norm_mean(*anchor_final_norm, h) : 1.0;
  const double mean_anchor_in =
      anchor_input_norm ? compute_bf16_norm_mean(*anchor_input_norm, h) : 1.0;
  const double mean_anchor_post =
      anchor_post_attn_norm ? compute_bf16_norm_mean(*anchor_post_attn_norm, h)
                            : mean_anchor_in;
  const double mean_anchor_q =
      anchor_q_norm ? compute_bf16_norm_mean(*anchor_q_norm, hd) : 1.0;
  const double mean_anchor_k =
      anchor_k_norm ? compute_bf16_norm_mean(*anchor_k_norm, hd) : 1.0;

  constexpr double kLegacyGapThreshold = 0.4;
  auto check_gap = [&](const std::string &suffix, U64 len,
                       double anchor_mean) -> float {
    const auto w = mtp_weights.parent_view(p + suffix);
    const double m = compute_bf16_norm_mean(w, len);
    if ((anchor_mean - m) > kLegacyGapThreshold) {
      out.repaired_any = true;
      return 1.0F;
    }
    return 0.0F;
  };

  // Match omlx/patches/mlx_lm_mtp/norm_repair.py::_REPAIR_GROUPS:
  // Only head norms whose raw HF mean can exceed the legacy 0.5 shift heuristic
  // (q_norm, k_norm, post_attention_layernorm, mtp.norm) are candidates for +1
  // repair. pre_fc_norm_embedding (shifted mean ~0.54 on Qwen3.8-27B),
  // pre_fc_norm_hidden, and input_layernorm have negative raw means and were
  // always shifted at conversion; shifting them again corrupts draft acceptance.
  (void)mean_anchor_in;
  out.pre_fc_norm_embedding = 0.0F;
  out.pre_fc_norm_hidden = 0.0F;
  out.input_layernorm = 0.0F;
  out.q_norm = check_gap("layers.0.self_attn.q_norm.weight", hd, mean_anchor_q);
  out.k_norm = check_gap("layers.0.self_attn.k_norm.weight", hd, mean_anchor_k);
  out.post_attention_layernorm =
      check_gap("layers.0.post_attention_layernorm.weight", h, mean_anchor_post);
  out.final_norm = check_gap("norm.weight", h, mean_anchor_final);
  return out;
}

// ---------------------------------------------------------------------------
// MtpDepthController
// ---------------------------------------------------------------------------

MtpDepthController::MtpDepthController(std::uint32_t max_depth,
                                       bool fixed_depth, double marginal_ms,
                                       double exit_margin)
    : max_depth_(std::clamp<std::uint32_t>(max_depth, 1U, kMaxSupportedDepth)),
      fixed_depth_(fixed_depth),
      marginal_ms_(marginal_ms > 0.0 ? marginal_ms : kDefaultMarginalMs),
      exit_margin_(std::clamp(exit_margin, 1.0, 2.0)) {
  reset();
}

void MtpDepthController::reset() {
  cur_ = max_depth_;
  p_.fill(std::nullopt);
  p_[0] = 0.6;
  t_.fill(std::nullopt);
  t_age_.fill(0.0);
  cycles_ = 0;
  probe_left_ = 0;
  exit_streak_ = 0;
  streak_spec_ = 0;
  ms_probe_ = 0.0;
  ms_explore_ = 0.0;
  warmup_.clear();
  if (!fixed_depth_) {
    for (std::uint32_t d = max_depth_; d >= 1U; --d) {
      warmup_.push_back(d);
    }
    if (max_depth_ > 1U) {
      warmup_.push_back(0U);
      warmup_.push_back(0U);
      warmup_.push_back(0U);
    }
  }
}

void MtpDepthController::on_new_request() noexcept {
  if (fixed_depth_) {
    return;
  }
  if (!warmup_.empty() || cycles_ == 0U) {
    reset();
    return;
  }
  p_.fill(std::nullopt);
  p_[0] = 0.6;
  probe_left_ = 0;
  exit_streak_ = 0;
  streak_spec_ = 0;
  ms_probe_ = 0.0;
  ms_explore_ = 0.0;
  cur_ = best_depth();
}

std::uint32_t
MtpDepthController::select_depth(std::uint32_t max_allowed_k) const noexcept {
  if (max_allowed_k == 0U) {
    return 0U;
  }
  if (fixed_depth_) {
    return std::min(max_allowed_k, max_depth_);
  }
  return std::min(cur_, max_allowed_k);
}

double MtpDepthController::p_eff(std::uint32_t j) const noexcept {
  std::uint32_t idx = std::min<std::uint32_t>(j, max_depth_ - 1U);
  while (idx > 0U && !p_[idx].has_value()) {
    --idx;
  }
  return p_[idx].value_or(0.6);
}

double MtpDepthController::marginal_est() const noexcept {
  std::optional<std::uint32_t> lo, hi;
  for (std::uint32_t d = 0U; d <= max_depth_; ++d) {
    if (t_[d].has_value()) {
      if (!lo.has_value()) {
        lo = d;
      }
      hi = d;
    }
  }
  if (lo.has_value() && hi.has_value() && *hi > *lo) {
    const double slope = (*t_[*hi] - *t_[*lo]) / double(*hi - *lo);
    if (slope > 0.0) {
      return slope;
    }
  }
  return marginal_ms_;
}

double MtpDepthController::t_est(std::uint32_t d) const noexcept {
  if (d <= max_depth_ && t_[d].has_value()) {
    return *t_[d];
  }
  bool any = false;
  double min_t = std::numeric_limits<double>::infinity();
  std::uint32_t closest_d = 0U;
  std::uint32_t best_dist = UINT32_MAX;
  for (std::uint32_t cand = 0U; cand <= max_depth_; ++cand) {
    if (!t_[cand].has_value()) {
      continue;
    }
    any = true;
    min_t = std::min(min_t, *t_[cand]);
    const std::uint32_t dist = (cand > d) ? (cand - d) : (d - cand);
    if (dist < best_dist) {
      best_dist = dist;
      closest_d = cand;
    }
  }
  if (!any) {
    return 30.0 + marginal_ms_ * double(d);
  }
  if (d == 0U) {
    return min_t;
  }
  return std::max(
      1e-3, *t_[closest_d] +
                marginal_est() * (double(d) - double(closest_d)));
}

double MtpDepthController::score(std::uint32_t d) const noexcept {
  double expected = 1.0;
  double run = 1.0;
  for (std::uint32_t j = 0U; j < d && j < max_depth_; ++j) {
    run *= p_eff(j);
    expected += run;
  }
  return expected / std::max(1e-6, t_est(d));
}

bool MtpDepthController::speculation_losing() const noexcept {
  if (!warmup_.empty() || !t_[0].has_value()) {
    return false;
  }
  const double base = score(0U);
  if (base <= 0.0) {
    return false;
  }
  double best = 0.0;
  for (std::uint32_t d = 1U; d <= max_depth_; ++d) {
    best = std::max(best, score(d));
  }
  return best < base * exit_margin_;
}

bool MtpDepthController::should_exit() const noexcept {
  return !fixed_depth_ && exit_streak_ >= kExitStreak &&
         streak_spec_ >= kExitConfirm;
}

std::uint32_t MtpDepthController::best_depth() const noexcept {
  std::uint32_t best_d = (cur_ <= max_depth_ && (cur_ > 0U || t_[0].has_value()))
                             ? cur_
                             : 1U;
  double best_s = score(best_d);
  const std::uint32_t first_d = t_[0].has_value() ? 0U : 1U;
  for (std::uint32_t d = first_d; d <= max_depth_; ++d) {
    if (d == best_d) {
      continue;
    }
    const double s = score(d);
    const double thresh = (d == cur_) ? best_s : (best_s * kHysteresis);
    if (s > thresh) {
      best_d = d;
      best_s = s;
    }
  }
  return best_d;
}

std::optional<std::uint32_t> MtpDepthController::best_rival() const noexcept {
  const double cur_s = score(cur_);
  if (cur_s <= 0.0) {
    return most_stale();
  }
  std::optional<std::uint32_t> rival;
  double rival_s = 0.0;
  for (std::uint32_t idx = 0U; idx <= max_depth_; ++idx) {
    const std::uint32_t d = (idx < max_depth_) ? (idx + 1U) : 0U;
    if (d == cur_) {
      continue;
    }
    const double s = score(d);
    if (s > rival_s) {
      rival = d;
      rival_s = s;
    }
  }
  if (rival.has_value() && rival_s >= cur_s / kProbeMargin) {
    return rival;
  }
  return std::nullopt;
}

std::optional<std::uint32_t> MtpDepthController::most_stale() const noexcept {
  std::optional<std::uint32_t> target;
  double max_age = -1.0;
  for (std::uint32_t idx = 0U; idx <= max_depth_; ++idx) {
    const std::uint32_t d = (idx < max_depth_) ? (idx + 1U) : 0U;
    if (d == cur_) {
      continue;
    }
    const double age =
        t_[d].has_value() ? t_age_[d] : std::numeric_limits<double>::infinity();
    if (age > max_age) {
      max_age = age;
      target = d;
    }
  }
  return target;
}

void MtpDepthController::update_time(std::uint32_t used,
                                     double cycle_ms) noexcept {
  cycle_ms = std::max(0.0, cycle_ms);
  if (!t_[used].has_value()) {
    t_[used] = cycle_ms;
    return;
  }
  const double prev = *t_[used];
  if (!warmup_.empty()) {
    t_[used] = std::min(prev, cycle_ms);
    return;
  }
  double a = 1.0 - std::exp(-cycle_ms / kTauMs);
  if (cycle_ms > kSpikeRatio * prev) {
    a *= kSpikeDamp;
  }
  t_[used] = (1.0 - a) * prev + a * cycle_ms;
}

void MtpDepthController::observe(std::uint32_t used, std::uint32_t accepted,
                                 double cycle_ms, bool time_sample) noexcept {
  ++cycles_;
  used = std::min(used, max_depth_);
  accepted = std::min(accepted, used);
  for (std::uint32_t j = 0U; j < used; ++j) {
    if (!p_[j].has_value()) {
      p_[j] = p_eff(j);
    }
    const double hit = (j < accepted) ? 1.0 : 0.0;
    p_[j] = (1.0 - kAlpha) * (*p_[j]) + kAlpha * hit;
    if (j >= accepted) {
      break;
    }
  }
  if (fixed_depth_) {
    return;
  }
  cycle_ms = std::max(0.0, cycle_ms);
  if (time_sample) {
    update_time(used, cycle_ms);
  }
  for (std::uint32_t d = 0U; d <= max_depth_; ++d) {
    if (t_[d].has_value()) {
      t_age_[d] += cycle_ms;
    }
  }
  if (time_sample && used <= max_depth_) {
    t_age_[used] = 0.0;
  }
  ms_probe_ += cycle_ms;
  ms_explore_ += cycle_ms;

  if (speculation_losing()) {
    ++exit_streak_;
    if (used > 0U && time_sample) {
      ++streak_spec_;
    }
  } else {
    exit_streak_ = 0;
    streak_spec_ = 0;
  }

  if (!warmup_.empty()) {
    if (!time_sample) {
      return;
    }
    warmup_.erase(warmup_.begin());
    if (!warmup_.empty()) {
      cur_ = warmup_.front();
      return;
    }
    cur_ = best_depth();
    ms_probe_ = 0.0;
    return;
  }

  if (probe_left_ == 0U && exit_streak_ >= kExitStreak &&
      streak_spec_ < kExitConfirm) {
    std::uint32_t best_spec = 1U;
    double best_s = score(1U);
    for (std::uint32_t d = 2U; d <= max_depth_; ++d) {
      const double s = score(d);
      if (s > best_s) {
        best_s = s;
        best_spec = d;
      }
    }
    cur_ = best_spec;
    probe_left_ = kExitConfirm - streak_spec_;
    ms_probe_ = 0.0;
    return;
  }

  if (probe_left_ > 0U) {
    --probe_left_;
    if (probe_left_ == 0U) {
      cur_ = best_depth();
      ms_probe_ = 0.0;
    }
    return;
  }

  cur_ = best_depth();
  if (max_depth_ > 1U) {
    const double period =
        std::max(kProbePeriodMs, double(kProbeLen) * cycle_ms / kProbeDuty);
    if (ms_probe_ >= period) {
      const bool explore_due =
          ms_explore_ >= std::max(kProbePeriodMaxMs, 2.0 * period);
      const auto target = explore_due ? most_stale() : best_rival();
      if (target.has_value()) {
        cur_ = *target;
        probe_left_ = kProbeLen;
        ms_probe_ = 0.0;
        if (explore_due) {
          ms_explore_ = 0.0;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Memory planner
// ---------------------------------------------------------------------------

QwenForwardMemoryPlan plan_mtp_forward_memory(const QwenModelConfig &c,
                                              std::uint32_t max_context,
                                              bool affine4_kv) {
  if (!max_context || c.hidden_size <= 0 || c.vocab_size <= 0 ||
      c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 ||
      c.head_dim <= 0 || (c.head_dim % 2) != 0) {
    throw std::invalid_argument("MTP execution geometry unsupported");
  }
  QwenForwardMemoryPlan plan;
  const bool is_moe = c.num_experts > 0;
  const U32 priming_rows =
      (!is_moe && c.hidden_size >= 1024)
          ? std::min<U32>(max_context, 256U)
          : std::min<U32>(max_context, 64U);
  plan.workspace_rows = priming_rows;

  const auto add_buf = [&](std::string name, U64 bytes, bool is_state = false,
                           bool clear_reset = false) {
    const U64 aligned = std::max<U64>(16ULL, bytes);
    if (aligned > (2ULL << 30)) {
      throw std::overflow_error("MTP buffer exceeds 2 GiB Apple10 ICB limit: " +
                                name);
    }
    plan.buffers.push_back({std::move(name), aligned, is_state, clear_reset});
    if (is_state) {
      plan.state_bytes += aligned;
    } else {
      plan.workspace_bytes += aligned;
    }
    plan.largest_buffer_bytes = std::max(plan.largest_buffer_bytes, aligned);
  };

  const U64 h = U64(c.hidden_size);
  const U64 f = U64(c.intermediate_size);
  const U64 v = U64(c.vocab_size);
  const U64 q = U64(c.num_attention_heads) * U64(c.head_dim);
  const U64 kv = U64(c.num_key_value_heads) * U64(c.head_dim);
  const U64 shared_f = U64(std::max<std::int32_t>(0, c.shared_expert_intermediate_size));
  const U64 mlp_rows = is_moe ? mul(priming_rows, 8ULL) : U64(priming_rows);
  const U64 max_ffn = std::max(f, shared_f);

  add_buf("fold_tokens", mul(priming_rows, 4U));
  add_buf("chain_token", 16U);
  add_buf("embed_out", mul(mul(priming_rows, h), 2U));
  add_buf("pending_trunk_normed", mul(h, 2U), true, true);
  add_buf("chain_hidden", mul(h, 2U));
  add_buf("stem_e", mul(mul(priming_rows, h), 2U));
  add_buf("stem_h", mul(mul(priming_rows, h), 2U));
  add_buf("stem_concat", mul(mul(priming_rows, 2U * h), 2U));
  add_buf("hidden", mul(mul(priming_rows, h), 2U));
  add_buf("norm", mul(mul(priming_rows, h), 2U));
  add_buf("last_norm", mul(h, 2U));
  add_buf("query", mul(mul(priming_rows, 2U * q), 2U));
  add_buf("key", mul(mul(priming_rows, kv), 2U));
  add_buf("value", mul(mul(priming_rows, kv), 2U));
  add_buf("mixer", mul(mul(priming_rows, q), 2U));
  add_buf("delta", mul(mul(priming_rows, h), 2U));
  add_buf("gate", mul(mul(mlp_rows, std::max<U64>(1ULL, max_ffn)), 2U));
  add_buf("up", mul(mul(mlp_rows, std::max<U64>(1ULL, max_ffn)), 2U));
  add_buf("product", mul(mul(mlp_rows, std::max<U64>(1ULL, max_ffn)), 2U));
  if (is_moe) {
    add_buf("router", mul(mul(priming_rows, 256U), 2U));
    add_buf("route_ids", mul(mul(priming_rows, 8U), 4U));
    add_buf("route_scores", mul(mul(priming_rows, 8U), 2U));
    add_buf("expert_output", mul(mul(mul(priming_rows, 8U), h), 2U));
    add_buf("routed", mul(mul(priming_rows, h), 2U));
    add_buf("shared", mul(mul(priming_rows, h), 2U));
    add_buf("shared_gate", mul(priming_rows, 2U));
  }
  add_buf("logits", mul(v, 2U));
  add_buf("qkv_concat", mul(mul(priming_rows, 2U * q + 2U * kv), 2U));
  add_buf("argmax_parts", 64U * 16U);
  add_buf("rms_sq", 64U);
  add_buf("rms_moe_dummy", 16U);
  add_buf("sampled", 4U * 8U);
  add_buf("compact_ids", 4U * 64U * 4U);
  add_buf("compact_scores", 4U * 64U * 4U);
  add_buf("proposal_out", 8U * 4U);
  const U64 max_sum_k = std::max({2U * h, q, max_ffn});
  add_buf("sums", mul(mul(mlp_rows, (max_sum_k + 31U) / 32U), 4U));
  add_buf("mtp_attn_pos",
          std::max<U64>(mul(priming_rows, 4U),
                        U64(MtpForward::kMaxDraftTokens) * 64ULL));
  if (c.hidden_size >= 1024) {
    constexpr U32 kMaxMtpSplitRows = 8U;
    const U32 qh_u = U32(c.num_attention_heads);
    const U32 kvh_u = U32(c.num_key_value_heads);
    const U32 hd_u = U32(c.head_dim);
    const U32 aff4_splits = (max_context + 511U) / 512U;
    const U32 bf16_splits =
        exact_attention_split_count(max_context, max_context, 32U);
    const U32 max_splits = std::max(aff4_splits, bf16_splits);
    const U64 split_meta_bytes =
        mul(mul(mul(kMaxMtpSplitRows, qh_u), max_splits), 4U);
    U64 split_out_bytes =
        mul(mul(mul(mul(kMaxMtpSplitRows, qh_u), bf16_splits), hd_u), 4U);
    if (affine4_kv && hd_u == 256U) {
      const auto lo = affine4_split_scratch_layout(
          kMaxMtpSplitRows, qh_u, kvh_u, aff4_splits,
          std::numeric_limits<U64>::max());
      split_out_bytes =
          std::max(split_out_bytes, lo.q_sum_offset + lo.q_sum_bytes);
    }
    add_buf("mtp_split_max", split_meta_bytes);
    add_buf("mtp_split_sum", split_meta_bytes);
    add_buf("mtp_split_out", split_out_bytes);
  }
  add_buf("zero", 256U);

  const KvStorageFormat kv_fmt =
      affine4_kv ? KvStorageFormat::Affine4Group128 : KvStorageFormat::BFloat16;
  const U32 pages = (max_context + 63U) / 64U;
  const U64 slab_bytes = kv_slab_bytes(
      kv_fmt, pages, 64U, U32(c.num_key_value_heads), U32(c.head_dim));
  add_buf("mtp_cache_key", slab_bytes, true, false);
  add_buf("mtp_cache_value", slab_bytes, true, false);
  return plan;
}

// ---------------------------------------------------------------------------
// MtpForward::Impl
// ---------------------------------------------------------------------------

class MtpForward::Impl {
public:
  struct StepPatchOffsets {
    U64 qk_rope_offset = 0; // word 4 = rows, word 9 = start_position
    std::vector<U64> attn_param_offsets{}; // word 3 = rows, word 4 = cache_start
    U64 sample_offset = 0;  // PositionSamplingRow or SampleParams
  };

  struct SaveTrunkProfile {
    Profile profile;
    U64 save_params_offset = 0; // word 1 = src_row
  };

  struct PrimingProfile {
    Profile profile;
    U32 capacity = 0;
    U32 cond_mode = 0;
    bool uses_delegated_trunk = false;
    U64 embed_offset = 0;       // word 5 = tokens
    U64 stem_offset = 0;        // word 1 = rows
    bool has_fc_dual = false;
    U64 fc_dual_offset = 0;     // word 2 = rows
    StepPatchOffsets attn_patches{};
  };

  struct FoldOnlyProfile {
    Profile profile;
    U32 fold_rows = 0;
    U32 cond_mode = 0;
    StepPatchOffsets attn_patches{};
  };

  struct ChainProfile {
    Profile profile;
    U32 fold_rows = 0;
    U32 draft_k = 0;
    U32 cond_mode = 0;
    std::array<StepPatchOffsets, kMaxDraftTokens> steps{};
  };

  Impl(const MetalContext &ctx, const MaterializedModel &mtp_w,
       QwenModelConfig cfg, U32 max_ctx, const MaterializedWeightView &emb,
       const MaterializedWeightView &hd_w,
       const MaterializedWeightView &tgt_final_norm,
       const MetalBuffer &tgt_trunk_hidden, float tgt_norm_offset,
       const std::filesystem::path &libs, bool m4_short,
       LinearNumericPolicy lin_pol, SamplingContract samp_contract,
       std::string_view kv_dtype, const QwenLayerCatalog *tgt_catalog,
       const MaterializedModel *tgt_weights,
       const MetalBuffer *shared_pf_trunk)
      : context(ctx), mtp_weights(mtp_w), config(std::move(cfg)),
        max_context(max_ctx), embedding(emb), head(hd_w),
        target_final_norm(tgt_final_norm),
        local_trunk_ref{&tgt_trunk_hidden, 0, tgt_trunk_hidden.size()},
        delegated_trunk_ref{shared_pf_trunk, 0,
                            shared_pf_trunk ? shared_pf_trunk->size() : 0},
        target_norm_offset(tgt_norm_offset), libraries(libs),
        m4_short_routes(m4_short), linear_policy(lin_pol),
        sampling_contract(samp_contract),
        kv_format(parse_kv_storage_format(kv_dtype)),
        gpu_cores(nax_estimate_gpu_cores(ctx.capabilities().device_name,
                                         detail::probe_host_agx_gpu_cores())) {
    (void)sampling_contract_name(sampling_contract);
    mtp_prefix = detect_mtp_prefix(mtp_weights);

    h = U32(config.hidden_size);
    v = U32(config.vocab_size);
    q = U32(config.num_attention_heads) * U32(config.head_dim);
    kv = U32(config.num_key_value_heads) * U32(config.head_dim);
    priming_cap = std::min<U32>(max_context, 64U);

    if (embedding.parent_shape != std::vector<U64>{v, h} ||
        head.parent_shape != std::vector<U64>{v, h}) {
      throw std::invalid_argument(
          "MtpForward shared target embedding/head shape mismatch");
    }

    // Resolve stem & Layer 0 weights.
    pre_fc_norm_embedding =
        mtp_weights.parent_view(mtp_prefix + "pre_fc_norm_embedding.weight");
    pre_fc_norm_hidden =
        mtp_weights.parent_view(mtp_prefix + "pre_fc_norm_hidden.weight");
    fc_weight = resolve_module_or_weight(mtp_weights, mtp_prefix + "fc");
    if (fc_weight.parent_shape != std::vector<U64>{h, 2ULL * h}) {
      throw std::invalid_argument("MtpForward fc shape mismatch: " +
                                  fc_weight.name);
    }

    const std::string l0 = mtp_prefix + "layers.0.";
    input_norm = mtp_weights.parent_view(l0 + "input_layernorm.weight");
    q_proj = resolve_module_or_weight(mtp_weights, l0 + "self_attn.q_proj");
    k_proj = resolve_module_or_weight(mtp_weights, l0 + "self_attn.k_proj");
    v_proj = resolve_module_or_weight(mtp_weights, l0 + "self_attn.v_proj");
    o_proj = resolve_module_or_weight(mtp_weights, l0 + "self_attn.o_proj");
    q_norm = mtp_weights.parent_view(l0 + "self_attn.q_norm.weight");
    k_norm = mtp_weights.parent_view(l0 + "self_attn.k_norm.weight");
    post_attn_norm =
        mtp_weights.parent_view(l0 + "post_attention_layernorm.weight");
    mtp_final_norm = mtp_weights.parent_view(mtp_prefix + "norm.weight");

    moe_layer = has_module_or_weight(mtp_weights,
                                     l0 + "mlp.switch_mlp.gate_proj");
    if (moe_layer) {
      moe_router = resolve_module_or_weight(mtp_weights, l0 + "mlp.gate");
      moe_gate = resolve_module_or_weight(mtp_weights,
                                          l0 + "mlp.switch_mlp.gate_proj");
      moe_up = resolve_module_or_weight(mtp_weights,
                                        l0 + "mlp.switch_mlp.up_proj");
      moe_down = resolve_module_or_weight(mtp_weights,
                                          l0 + "mlp.switch_mlp.down_proj");
      shared_gate_proj = resolve_module_or_weight(
          mtp_weights, l0 + "mlp.shared_expert.gate_proj");
      shared_up_proj = resolve_module_or_weight(
          mtp_weights, l0 + "mlp.shared_expert.up_proj");
      shared_down_proj = resolve_module_or_weight(
          mtp_weights, l0 + "mlp.shared_expert.down_proj");
      shared_expert_gate = resolve_module_or_weight(
          mtp_weights, l0 + "mlp.shared_expert_gate");
      if (moe_gate.parent_shape.size() != 3 || moe_gate.parent_shape[0] != 256 ||
          moe_gate.parent_shape[2] != h) {
        throw std::invalid_argument("MtpForward MoE expert gate shape invalid");
      }
      f = U32(moe_gate.parent_shape[1]);
      shared_f = U32(shared_gate_proj.row_count);
      config.num_experts = 256;
      config.num_experts_per_token = 8;
      config.intermediate_size = std::int32_t(f);
      config.shared_expert_intermediate_size = std::int32_t(shared_f);
    } else {
      mlp_gate = resolve_module_or_weight(mtp_weights, l0 + "mlp.gate_proj");
      mlp_up = resolve_module_or_weight(mtp_weights, l0 + "mlp.up_proj");
      mlp_down = resolve_module_or_weight(mtp_weights, l0 + "mlp.down_proj");
      if (mlp_gate.parent_shape.size() != 2 || mlp_gate.in_features != h) {
        throw std::invalid_argument("MtpForward dense MLP gate shape invalid");
      }
      f = U32(mlp_gate.row_count);
      config.num_experts = 0;
      config.intermediate_size = std::int32_t(f);
    }

    // Resolve backbone norm anchors for repair_legacy_head_norms.
    const MaterializedModel *anchor_model =
        tgt_weights ? tgt_weights : &mtp_weights;
    std::optional<MaterializedWeightView> s_in, s_post, s_q, s_k;
    const MaterializedWeightView *anc_in =
        find_stage_weight(tgt_catalog, anchor_model, 0U, "input_norm", s_in);
    const MaterializedWeightView *anc_post = find_stage_weight(
        tgt_catalog, anchor_model, 0U, "post_attention_norm", s_post);
    const MaterializedWeightView *anc_q = nullptr;
    const MaterializedWeightView *anc_k = nullptr;
    if (tgt_catalog) {
      for (const auto &ls : tgt_catalog->layers()) {
        if (ls.type == LayerType::full_attention) {
          anc_q =
              find_stage_weight(tgt_catalog, anchor_model, ls.layer, "q_norm", s_q);
          anc_k =
              find_stage_weight(tgt_catalog, anchor_model, ls.layer, "k_norm", s_k);
          break;
        }
      }
    }
    norm_offsets = repair_legacy_head_norms(
        mtp_weights, mtp_prefix, config, target_norm_offset, &target_final_norm,
        anc_in, anc_post, anc_q, anc_k);

    // Allocate workspace + 1-layer KvPagePool.
    const auto mem_plan = plan_mtp_forward_memory(
        config, max_context, kv_format == KvStorageFormat::Affine4Group128);
    priming_cap = mem_plan.workspace_rows;
    for (const auto &item : mem_plan.buffers) {
      auto buf = std::make_unique<MetalBuffer>(context.make_buffer(item.bytes));
      refs.emplace(item.name, Ref{buf.get(), 0, item.bytes});
      if (item.clear_on_reset) {
        clear_on_reset_buffers.push_back(buf.get());
      }
      buffers.push_back(std::move(buf));
    }
    const_cast<MetalBuffer *>(get("zero").buffer)
        ->write(std::vector<std::byte>(256));

    const U32 pages = (max_context + 63U) / 64U;
    mtp_kv_pool =
        std::make_unique<KvPagePool>(context, max_context, 64U, pages, kv_format);

    affine4_nax_ok =
        config.hidden_size >= 1024 &&
        kv_format == KvStorageFormat::Affine4Group128 &&
        affine4_kv_runtime_eligible(context, libraries,
                                    U32(config.num_attention_heads),
                                    U32(config.num_key_value_heads),
                                    U32(config.head_dim));
    flash_nax_prefill_ok =
        config.hidden_size >= 1024 &&
        flash_nax_prefill_runtime_eligible(context, libraries,
                                           U32(config.head_dim));
    split_attn_ok =
        config.hidden_size >= 1024 &&
        context.capabilities().profile.probe.apple_gpu_family >= 9 &&
        std::filesystem::is_regular_file(
            libraries / "omlx_m5_attention_split_kv.metallib");

    // Acquire NAX caches (reusing target's cache for shared head -> 0 B head duplication!).
    bf16_nax_enabled =
        linear_policy == LinearNumericPolicy::RowExactPortableV1 &&
        bf16_nax_runtime_eligible(context, libraries);

    if (moe_layer &&
        moe_nax_runtime_eligible(context, libraries) &&
        moe_nax_weight_eligible(moe_gate, moe_up, moe_down, h, f)) {
      moe_nax_ws.allocate(context, priming_cap, h, f, buffers, 256U, 8U);
    }

    std::vector<MaterializedWeightView> mtp_candidates{
        fc_weight, q_proj, k_proj, v_proj, o_proj};
    if (moe_layer) {
      mtp_candidates.push_back(moe_router);
      mtp_candidates.push_back(shared_gate_proj);
      mtp_candidates.push_back(shared_up_proj);
      mtp_candidates.push_back(shared_down_proj);
      mtp_candidates.push_back(shared_expert_gate);
    } else {
      mtp_candidates.push_back(mlp_gate);
      mtp_candidates.push_back(mlp_up);
      mtp_candidates.push_back(mlp_down);
    }
    bool any_pretiled_mx = (head.layout == "mx_nax_tiled_bk128_v1");
    bool any_pretiled_oq = (head.layout == "oq_nax_tiled_g64_v1" ||
                            head.layout == "oq_nax_tiled_g128_v1");
    for (const auto &cw : mtp_candidates) {
      any_pretiled_mx |= (cw.layout == "mx_nax_tiled_bk128_v1");
      any_pretiled_oq |= (cw.layout == "oq_nax_tiled_g64_v1" ||
                          cw.layout == "oq_nax_tiled_g128_v1");
    }

    const void *head_buf_ptr =
        head.planes.empty() ? nullptr : head.planes.front().buffer;
    bool head_is_overridden = false;
    if (tgt_weights != nullptr && head_buf_ptr != nullptr) {
      try {
        const auto orig_head = tgt_weights->parent_view(head.object_id);
        head_is_overridden =
            orig_head.planes.empty() ||
            orig_head.planes.front().buffer != head_buf_ptr;
      } catch (...) {
        head_is_overridden = true;
      }
    }
    const void *head_owner_key =
        head_is_overridden
            ? head_buf_ptr
            : (tgt_weights ? static_cast<const void *>(tgt_weights)
                           : static_cast<const void *>(&mtp_weights));
    const void *mtp_owner_key = reinterpret_cast<const void *>(
        reinterpret_cast<std::uintptr_t>(&mtp_weights) ^ 0x4D545035ULL);

    kl32_enabled = mtp_kl32_enabled() &&
                   std::filesystem::exists(libraries / "omlx_m5_row_exact.metallib");

    const std::array<MaterializedWeightView, 3> qkv_parts{q_proj, k_proj,
                                                          v_proj};

    const auto mtp_silicon_tuning = detail::resolve_silicon_tuning(context);
    if (mxfp4_nax_runtime_eligible(context, libraries) &&
        (linear_policy == LinearNumericPolicy::RowExactPortableV1 ||
         any_pretiled_mx || kl32_enabled)) {
      if (linear_policy == LinearNumericPolicy::RowExactPortableV1 ||
          head.layout == "mx_nax_tiled_bk128_v1" || kl32_enabled) {
        head_mxfp4_cache = Mxfp4NaxCache::acquire(
            head_owner_key, context, libraries, {head},
            mtp_silicon_tuning.linear_split_k_target_slices);
      }
      mtp_mxfp4_cache = nax_acquire_owner_cache<Mxfp4NaxCache>(
          mtp_owner_key, [&](Mxfp4NaxCache &c) {
            c.set_target_split_k(
                mtp_silicon_tuning.linear_split_k_target_slices);
            for (const auto &w : mtp_candidates) {
              c.register_weight(w);
            }
            c.register_concat(kMtpConcatQkvId, qkv_parts);
            c.build(context, libraries);
          });
      if ((head_mxfp4_cache && head_mxfp4_cache->active()) ||
          (mtp_mxfp4_cache && mtp_mxfp4_cache->active())) {
        const U32 max_mp = mxfp4_nax_mp(priming_cap);
        const U32 max_k = std::max({2U * h, f, q, shared_f});
        const U64 f16_bytes = mul(mul(max_mp, max_k), 2U);
        const U64 rs_bytes = mul(max_mp, 4U);
        auto f16_buf = std::make_unique<MetalBuffer>(
            context.make_private_buffer(f16_bytes));
        auto rs_buf = std::make_unique<MetalBuffer>(
            context.make_private_buffer(rs_bytes));
        f16_scratch = Ref{f16_buf.get(), 0, f16_bytes};
        f16_row_scale = Ref{rs_buf.get(), 0, rs_bytes};
        buffers.push_back(std::move(f16_buf));
        buffers.push_back(std::move(rs_buf));
      }
    }

    if (affine_nax_runtime_eligible(context, libraries) &&
        (linear_policy == LinearNumericPolicy::RowExactPortableV1 ||
         any_pretiled_oq || kl32_enabled)) {
      if (linear_policy == LinearNumericPolicy::RowExactPortableV1 ||
          head.layout == "oq_nax_tiled_g64_v1" ||
          head.layout == "oq_nax_tiled_g128_v1" || kl32_enabled) {
        head_affine_cache = AffineNaxCache::acquire(
            head_owner_key, context, libraries, get("zero"), {head},
            mtp_silicon_tuning.linear_split_k_target_slices);
      }
      mtp_affine_cache = nax_acquire_owner_cache<AffineNaxCache>(
          mtp_owner_key, [&](AffineNaxCache &c) {
            c.set_target_split_k(
                mtp_silicon_tuning.linear_split_k_target_slices);
            for (const auto &w : mtp_candidates) {
              c.register_weight(w, get("zero"));
            }
            c.register_concat(kMtpConcatQkvId, qkv_parts);
            c.build(context, libraries, get("zero"));
          });
      if ((head_affine_cache && head_affine_cache->active()) ||
          (mtp_affine_cache && mtp_affine_cache->active())) {
        const U32 max_mp = affine_nax_mp(priming_cap);
        const U32 max_k = std::max({2U * h, f, q, shared_f});
        const U64 xs_bytes = mul(mul(max_mp, std::max(1U, max_k / 64U)), 4U);
        auto xs_buf = std::make_unique<MetalBuffer>(
            context.make_private_buffer(xs_bytes));
        affine_nax_xs = Ref{xs_buf.get(), 0, xs_bytes};
        buffers.push_back(std::move(xs_buf));
      }
    }

    if (kl32_enabled) {
      head_q5_cache = MtpQ5High1Cache::acquire(head_owner_key, context,
                                               libraries, {head},
                                               head_affine_cache.get());
      mtp_q5_cache = MtpQ5High1Cache::acquire(mtp_owner_key, context, libraries,
                                              mtp_candidates,
                                              mtp_affine_cache.get());
    }

    reset();

    // Compile immutable ICB profiles once at load time.
    save_local_trunk = compile_save_trunk(local_trunk_ref, "local");
    if (delegated_trunk_ref.buffer != nullptr &&
        delegated_trunk_ref.buffer != local_trunk_ref.buffer) {
      save_delegated_trunk = compile_save_trunk(delegated_trunk_ref, "delegated");
    }

    const U32 local_trunk_max_rows =
        U32(std::min<U64>(priming_cap, local_trunk_ref.bytes / mul(h, 2U)));
    for (U32 cond_mode : {0U, 1U}) {
      for (U32 cap :
           {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U, 256U, 512U, 1024U}) {
        if (cap > priming_cap) {
          break;
        }
        if (cap <= local_trunk_max_rows) {
          local_priming_profiles[cond_mode].push_back(
              compile_priming(cap, cond_mode, local_trunk_ref, false));
        }
        if (delegated_trunk_ref.buffer != nullptr &&
            delegated_trunk_ref.buffer != local_trunk_ref.buffer) {
          delegated_priming_profiles[cond_mode].push_back(
              compile_priming(cap, cond_mode, delegated_trunk_ref, true));
        }
      }
      if (!local_priming_profiles[cond_mode].empty() &&
          local_priming_profiles[cond_mode].back().capacity <
              local_trunk_max_rows) {
        local_priming_profiles[cond_mode].push_back(compile_priming(
            local_trunk_max_rows, cond_mode, local_trunk_ref, false));
      }
      if (delegated_trunk_ref.buffer != nullptr &&
          delegated_trunk_ref.buffer != local_trunk_ref.buffer &&
          (delegated_priming_profiles[cond_mode].empty() ||
           delegated_priming_profiles[cond_mode].back().capacity < priming_cap)) {
        delegated_priming_profiles[cond_mode].push_back(
            compile_priming(priming_cap, cond_mode, delegated_trunk_ref, true));
      }
    }

    for (U32 c_fold = 1U; c_fold <= kMaxFoldTokens; ++c_fold) {
      if (c_fold > max_context) {
        break;
      }
      fold_only_profiles[0][c_fold - 1U] = compile_fold_only(c_fold, 0U);
      if (c_fold == 1U) {
        fold_only_profiles[1][0] = compile_fold_only(1U, 1U);
      }
      for (U32 k_draft = 1U; k_draft <= kMaxDraftTokens; ++k_draft) {
        if (c_fold + k_draft - 1U > max_context) {
          break;
        }
        chain_profiles[0][c_fold - 1U][k_draft - 1U] =
            compile_chain(c_fold, k_draft, 0U);
        if (c_fold == 1U) {
          chain_profiles[1][0][k_draft - 1U] = compile_chain(1U, k_draft, 1U);
        }
      }
    }
  }

  Ref get(std::string_view name) const { return refs.at(std::string(name)); }

  Ref direct(const MaterializedWeightView &w, U64 count) const {
    if (w.format != "direct_bf16_v1" || plane(w, "data").bytes != mul(count, 2U)) {
      throw std::invalid_argument("MTP direct BF16 weight mismatch: " + w.name);
    }
    return plane(w, "data");
  }

  const TiledMxfp4Weight *find_mxfp4_nax(const MaterializedWeightView &w) const {
    if (mtp_mxfp4_cache) {
      if (const auto *tw = mtp_mxfp4_cache->find(w.object_id)) {
        return tw;
      }
    }
    if (head_mxfp4_cache && w.object_id == head.object_id &&
        w.name == head.name) {
      return head_mxfp4_cache->find(w.object_id);
    }
    return nullptr;
  }

  const TiledAffineWeight *find_affine_nax(const MaterializedWeightView &w) const {
    if (mtp_affine_cache) {
      if (const auto *tw = mtp_affine_cache->find(w.object_id)) {
        return tw;
      }
    }
    if (head_affine_cache && w.object_id == head.object_id &&
        w.name == head.name) {
      return head_affine_cache->find(w.object_id);
    }
    return nullptr;
  }

  const Ref *find_q5_high1(const MaterializedWeightView &w) const {
    if (const auto *tw = find_affine_nax(w); tw && tw->high1.buffer) {
      return &tw->high1;
    }
    if (mtp_q5_cache) {
      if (const auto *r = mtp_q5_cache->find(w.object_id)) {
        return r;
      }
    }
    if (head_q5_cache && w.object_id == head.object_id && w.name == head.name) {
      return head_q5_cache->find(w.object_id);
    }
    return nullptr;
  }

  const Ref *find_sb_tiled_fallback(const MaterializedWeightView &w) const {
    if (mtp_q5_cache) {
      if (const auto *r = mtp_q5_cache->find_sb_tiled(w.object_id)) {
        return r;
      }
    }
    if (head_q5_cache && w.object_id == head.object_id && w.name == head.name) {
      return head_q5_cache->find_sb_tiled(w.object_id);
    }
    return nullptr;
  }

  Ref affine_high_plane(const MaterializedWeightView &w, U32 bits) const {
    return bits == 3U                   ? plane(w, "nax_high1")
           : (bits == 5U || bits == 6U) ? plane(w, "nax_high2")
                                        : get("zero");
  }

  bool resolve_pretiled_mxfp4_t1(const MaterializedWeightView &w, U32 t,
                                 Ref *out_codes = nullptr,
                                 Ref *out_scales = nullptr) const {
    if (!(kl32_enabled && t == 1U && h >= 512U &&
          w.format == "mxfp4_e2m1_g32_e8m0_v1" && w.parent_shape.size() == 2U &&
          w.row_count > 0U && w.in_features >= 512U &&
          (w.in_features % 128U) == 0U)) {
      return false;
    }
    if (w.layout == "mx_nax_tiled_bk128_v1") {
      if (out_codes)
        *out_codes = plane(w, "nax_data");
      if (out_scales)
        *out_scales = plane(w, "nax_scale");
      return true;
    }
    if (const auto *tw = find_mxfp4_nax(w)) {
      const U32 exp_bn = mxfp4_nax_bn(U32(w.row_count));
      if (tw->n == w.row_count && tw->k == w.in_features && tw->bn == exp_bn) {
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
    if (!(kl32_enabled && t == 1U && bits != 0U && h >= 512U &&
          w.parent_shape.size() == 2U && w.row_count > 0U &&
          w.in_features >= 512U && (w.in_features % 64U) == 0U)) {
      return false;
    }
    const Ref *q5_h1 = (bits == 5U) ? find_q5_high1(w) : nullptr;
    if (bits == 5U && q5_h1 == nullptr) {
      return false;
    }
    if (const auto *tw = find_affine_nax(w)) {
      const U32 exp_bn = affine_nax_bn(U32(w.row_count));
      if (tw->n == w.row_count && tw->k == w.in_features && tw->bn == exp_bn &&
          tw->bits == bits && tw->sb_tiled.buffer) {
        if (out_codes)
          *out_codes = tw->codes;
        if (out_high)
          *out_high = (bits == 5U)
                          ? *q5_h1
                          : (affine_nax_has_high_plane(bits) ? tw->high2
                                                             : get("zero"));
        if (out_sb)
          *out_sb = tw->sb_tiled;
        return true;
      }
    }
    if (w.layout == "oq_nax_tiled_g64_v1" ||
        w.layout == "oq_nax_tiled_g128_v1") {
      const Ref *sb_tl = find_sb_tiled_fallback(w);
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

  bool use_mxfp4_nax(const MaterializedWeightView &w) const {
    if (!f16_scratch.buffer) {
      return false;
    }
    if (w.layout == "mx_nax_tiled_bk128_v1") {
      return find_mxfp4_nax(w) != nullptr;
    }
    return linear_policy == LinearNumericPolicy::RowExactPortableV1 &&
           find_mxfp4_nax(w) != nullptr;
  }

  bool use_affine_nax(const MaterializedWeightView &w) const {
    if (!affine_nax_xs.buffer) {
      return false;
    }
    if (w.layout == "oq_nax_tiled_g64_v1" ||
        w.layout == "oq_nax_tiled_g128_v1") {
      return find_affine_nax(w) != nullptr;
    }
    return linear_policy == LinearNumericPolicy::RowExactPortableV1 &&
           find_affine_nax(w) != nullptr;
  }

  std::vector<std::byte> fused_norm_params(U32 n0, U32 n1, U32 n2, U32 n3,
                                           U32 k_in, U32 bits_packed,
                                           U32 sq_slot, float w_off) const {
    return words({n0, n1, n2, n3, k_in, bits_packed, sq_slot,
                  word(1.0F / float(k_in)), word(float(config.rms_norm_eps)),
                  word(w_off)});
  }

  bool can_linear_norm_t1(const MaterializedWeightView &w, U32 n, U32 k) const {
    return w.row_count == n && w.in_features == k &&
           (resolve_pretiled_mxfp4_t1(w, 1U) ||
            resolve_pretiled_affine_t1(w, 1U));
  }

  bool can_linear_swiglu_norm_t1(const MaterializedWeightView &gate_w,
                                 const MaterializedWeightView &up_w, U32 n,
                                 U32 k) const {
    if (n >= 65536U || gate_w.padded_k != up_w.padded_k ||
        gate_w.row_count != n || gate_w.in_features != k ||
        up_w.row_count != n || up_w.in_features != k) {
      return false;
    }
    return (resolve_pretiled_mxfp4_t1(gate_w, 1U) &&
            resolve_pretiled_mxfp4_t1(up_w, 1U)) ||
           (resolve_pretiled_affine_t1(gate_w, 1U) &&
            resolve_pretiled_affine_t1(up_w, 1U));
  }

  bool try_linear_add_sq_t1(Encoder &e, const MaterializedWeightView &w, Ref x,
                            Ref hidden_io, U32 n, U32 k, U32 sq_slot) {
    if (n >= 65536U || w.row_count != n || w.in_features != k) {
      return false;
    }
    Ref mx_c{}, mx_s{};
    if (resolve_pretiled_mxfp4_t1(w, 1U, &mx_c, &mx_s)) {
      const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
      const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
      const char *fn = (n >= 2048U)
                           ? "omlx_m5_pretiled_qmv_mxfp4_bn16_r4_add_sq_t1"
                           : "omlx_m5_pretiled_qmv_mxfp4_bn16_r2_add_sq_t1";
      auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                       {.width = 64U},
                       {x, mx_c, mx_s, hidden_io, get("rms_sq")});
      e.dynamic(d, 5, fused_norm_params(n, 0U, 0U, 0U, k, 0U, sq_slot, 0.0F));
      e.record_weight(w, n, k, "pretiled_qmv_t1",
                      "pretiled_mxfp4_qmv_add_sq_t1_v4", {}, "legacy_route_v1",
                      {}, false, 1U);
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
                       {x, aff_c, aff_h, aff_sb, hidden_io, get("rms_sq")});
      e.dynamic(d, 6, fused_norm_params(n, 0U, 0U, 0U, k, bits, sq_slot, 0.0F));
      e.record_weight(w, n, k, "pretiled_qmv_t1",
                      "pretiled_affine_qmv_add_sq_t1_v4", {}, "legacy_route_v1",
                      {}, false, 1U);
      return true;
    }
    return false;
  }

  bool try_linear_norm_t1(Encoder &e, const MaterializedWeightView &w,
                          const MaterializedWeightView &norm_w, float w_off,
                          Ref hidden_in, Ref y, U32 n, U32 k, U32 sq_slot) {
    if (w.row_count != n || w.in_features != k) {
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
      auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                       {.width = 64U},
                       {hidden_in, nw_ref, get("rms_sq"), mx_c, mx_s, y});
      e.dynamic(d, 6, fused_norm_params(n, 0U, 0U, 0U, k, 0U, sq_slot, w_off));
      e.record_weight(w, n, k, "pretiled_qmv_t1",
                      "pretiled_mxfp4_qmv_norm_t1_v4", {}, "legacy_route_v1",
                      {}, false, 1U);
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
      auto &d = e.emit(
          "row_exact", fn, {.width = U64(tgs) * 64U}, {.width = 64U},
          {hidden_in, nw_ref, get("rms_sq"), aff_c, aff_h, aff_sb, y});
      e.dynamic(d, 7,
                fused_norm_params(n, 0U, 0U, 0U, k, bits, sq_slot, w_off));
      e.record_weight(w, n, k, "pretiled_qmv_t1",
                      "pretiled_affine_qmv_norm_t1_v4", {}, "legacy_route_v1",
                      {}, false, 1U);
      return true;
    }
    return false;
  }

  bool try_linear_swiglu_norm_t1(Encoder &e,
                                 const MaterializedWeightView &gate_w,
                                 const MaterializedWeightView &up_w,
                                 const MaterializedWeightView &norm_w,
                                 float w_off, Ref hidden_in, Ref y, U32 n,
                                 U32 k, U32 sq_slot) {
    if (n >= 65536U || gate_w.padded_k != up_w.padded_k ||
        gate_w.row_count != n || gate_w.in_features != k ||
        up_w.row_count != n || up_w.in_features != k) {
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
      auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                       {.width = 64U},
                       {hidden_in, nw_ref, get("rms_sq"), gc, gs, uc, us, y});
      e.dynamic(d, 8, fused_norm_params(n, 0U, 0U, 0U, k, 0U, sq_slot, w_off));
      e.record_weight(gate_w, n, k, "pretiled_qmv_t1",
                      "pretiled_mxfp4_qmv_swiglu_norm_t1_v4", {},
                      "legacy_route_v1", {}, false, 1U);
      e.record_weight(up_w, n, k, "pretiled_qmv_t1",
                      "pretiled_mxfp4_qmv_swiglu_norm_t1_v4", {},
                      "legacy_route_v1", {}, false, 1U);
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
      auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                       {.width = 64U},
                       {hidden_in, nw_ref, get("rms_sq"), agc, agh, agsb, auc,
                        auh, ausb, y});
      e.dynamic(d, 10,
                fused_norm_params(n, 0U, 0U, 0U, k, packed_bits, sq_slot, w_off));
      e.record_weight(gate_w, n, k, "pretiled_qmv_t1",
                      "pretiled_affine_qmv_swiglu_norm_t1_v4", {},
                      "legacy_route_v1", {}, false, 1U);
      e.record_weight(up_w, n, k, "pretiled_qmv_t1",
                      "pretiled_affine_qmv_swiglu_norm_t1_v4", {},
                      "legacy_route_v1", {}, false, 1U);
      return true;
    }
    return false;
  }

  bool try_linear_swiglu(Encoder &e, const MaterializedWeightView &gate_w,
                         const MaterializedWeightView &up_w, Ref x, Ref y,
                         U32 n, U32 k, std::optional<U32> explicit_t) {
    if (gate_w.padded_k != up_w.padded_k || gate_w.row_count != n ||
        gate_w.in_features != k || up_w.row_count != n ||
        up_w.in_features != k) {
      return false;
    }
    const U32 t = explicit_t.value_or(e.profile.capacity);
    if (explicit_t.has_value() && *explicit_t == 1U) {
      Ref gc{}, gs{}, uc{}, us{};
      if (resolve_pretiled_mxfp4_t1(gate_w, 1U, &gc, &gs) &&
          resolve_pretiled_mxfp4_t1(up_w, 1U, &uc, &us)) {
        const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
        const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
        const char *fn =
            (n >= 65536U)  ? "omlx_m5_pretiled_qmv_mxfp4_swiglu_t1"
            : (n >= 2048U) ? "omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r4_t1"
                           : "omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r2_t1";
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U}, {x, gc, gs, uc, us, y});
        e.dynamic(d, 6, words({n, k, U32(gate_w.padded_k), 1U}));
        e.record_weight(gate_w, n, k, "pretiled_qmv_t1",
                        "pretiled_mxfp4_qmv_swiglu_t1_v3", {},
                        "legacy_route_v1", {}, false, 1U);
        e.record_weight(up_w, n, k, "pretiled_qmv_t1",
                        "pretiled_mxfp4_qmv_swiglu_t1_v3", {},
                        "legacy_route_v1", {}, false, 1U);
        return true;
      }
      Ref agc{}, agh{}, agsb{}, auc{}, auh{}, ausb{};
      if (resolve_pretiled_affine_t1(gate_w, 1U, &agc, &agh, &agsb) &&
          resolve_pretiled_affine_t1(up_w, 1U, &auc, &auh, &ausb)) {
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
        e.dynamic(d, 8, words({n, 1U, 0U, k, packed_bits}));
        e.record_weight(gate_w, n, k, "pretiled_qmv_t1",
                        "pretiled_affine_qmv_swiglu_t1_v3", {},
                        "legacy_route_v1", {}, false, 1U);
        e.record_weight(up_w, n, k, "pretiled_qmv_t1",
                        "pretiled_affine_qmv_swiglu_t1_v3", {},
                        "legacy_route_v1", {}, false, 1U);
        return true;
      }
    }
    if (gate_w.format != up_w.format ||
        (explicit_t.has_value() && t > e.profile.capacity)) {
      return false;
    }
    if (gate_w.format == "direct_bf16_v1" && bf16_nax_enabled &&
        bf16_nax_weight_eligible(gate_w, n, k) &&
        bf16_nax_weight_eligible(up_w, n, k)) {
      linear_bf16_nax_swiglu(e, gate_w, up_w, x, y, n, k, gpu_cores, explicit_t);
      return true;
    }
    if (use_mxfp4_nax(gate_w) && use_mxfp4_nax(up_w)) {
      const auto *gw = find_mxfp4_nax(gate_w);
      const auto *uw = find_mxfp4_nax(up_w);
      if (gw && uw && gw->bn == 16U && uw->bn == 16U && gw->n == n &&
          gw->k == k && uw->n == n && uw->k == k &&
          gw->split_k == uw->split_k) {
        linear_mxfp4_nax_swiglu(e, *gw, *uw, x, f16_scratch, f16_row_scale, y,
                                gpu_cores, explicit_t);
        const std::string grp =
            mxfp4_nax_group(e, *gw, explicit_t, gpu_cores, true);
        e.record_weight(gate_w, n, k, "mxfp4_nax_m5_v1", "mxfp4_nax_m5_v1", {},
                        "linear_row_exact_portable_v1", grp, true, explicit_t);
        e.record_weight(up_w, n, k, "mxfp4_nax_m5_v1", "mxfp4_nax_m5_v1", {},
                        "linear_row_exact_portable_v1", grp, true, explicit_t);
        return true;
      }
    }
    if (use_affine_nax(gate_w) && use_affine_nax(up_w)) {
      const auto *gw = find_affine_nax(gate_w);
      const auto *uw = find_affine_nax(up_w);
      if (gw && uw && gw->bits == uw->bits && gw->bn == 16U && uw->bn == 16U &&
          gw->n == n && gw->k == k && uw->n == n && uw->k == k &&
          gw->split_k == uw->split_k) {
        linear_affine_nax_swiglu(e, *gw, *uw, x, affine_nax_xs, y, gpu_cores,
                                 explicit_t);
        const std::string grp =
            affine_nax_group(e, *gw, explicit_t, gpu_cores, uw->group_size);
        e.record_weight(gate_w, n, k, "affine_nax_m5_v1", "affine_nax_m5_v1",
                        {}, "linear_row_exact_portable_v1", grp, true,
                        explicit_t);
        e.record_weight(up_w, n, k, "affine_nax_m5_v1", "affine_nax_m5_v1", {},
                        "linear_row_exact_portable_v1", grp, true, explicit_t);
        return true;
      }
    }
    return false;
  }

  bool try_concat3_qkv_t1(Encoder &e) {
    if (q_proj.row_count != 2U * q || k_proj.row_count != kv ||
        v_proj.row_count != kv || q_proj.in_features != h ||
        k_proj.in_features != h || v_proj.in_features != h ||
        2U * q >= 65536U || kv >= 65536U) {
      return false;
    }
    const std::array<MaterializedWeightView, 3> parts{q_proj, k_proj, v_proj};
    const std::array<U32, 3> part_ns{2U * q, kv, kv};
    Ref d0{}, s0{}, d1{}, s1{}, d2{}, s2{};
    if (resolve_pretiled_mxfp4_t1(q_proj, 1U, &d0, &s0) &&
        resolve_pretiled_mxfp4_t1(k_proj, 1U, &d1, &s1) &&
        resolve_pretiled_mxfp4_t1(v_proj, 1U, &d2, &s2)) {
      const bool use_r4 = ((2U * q + 2U * kv) >= 2048U);
      const U32 rows_per_tg = use_r4 ? 8U : 4U;
      const U32 tgs = ((2U * q + rows_per_tg - 1U) / rows_per_tg) +
                      2U * ((kv + rows_per_tg - 1U) / rows_per_tg);
      const char *fn = use_r4 ? "omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r4_t1"
                              : "omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r2_t1";
      auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                       {.width = 64U},
                       {get("norm"), d0, s0, d1, s1, d2, s2, get("query"),
                        get("key"), get("value")});
      e.dynamic(d, 10, words({2U * q, kv, kv, h}));
      for (std::size_t i = 0; i < 3; ++i) {
        e.record_weight(parts[i], part_ns[i], h, "pretiled_qmv_t1",
                        "pretiled_mxfp4_qmv_concat3_t1_v3", {},
                        "legacy_route_v1", {}, false, 1U);
      }
      return true;
    }
    Ref h0{}, h1{}, h2{};
    if (resolve_pretiled_affine_t1(q_proj, 1U, &d0, &h0, &s0) &&
        resolve_pretiled_affine_t1(k_proj, 1U, &d1, &h1, &s1) &&
        resolve_pretiled_affine_t1(v_proj, 1U, &d2, &h2, &s2)) {
      const U32 b0 = affine_nax_bits(q_proj.format);
      const U32 b1 = affine_nax_bits(k_proj.format);
      const U32 b2 = affine_nax_bits(v_proj.format);
      const U32 packed_bits = b0 | (b1 << 8U) | (b2 << 16U);
      const bool use_r4 = ((2U * q + 2U * kv) >= 2048U);
      const U32 rows_per_tg = use_r4 ? 8U : 4U;
      const U32 tgs = ((2U * q + rows_per_tg - 1U) / rows_per_tg) +
                      2U * ((kv + rows_per_tg - 1U) / rows_per_tg);
      const bool all_q456 = (b0 >= 4U && b0 <= 6U) && (b1 >= 4U && b1 <= 6U) &&
                            (b2 >= 4U && b2 <= 6U);
      const char *fn = "omlx_m5_pretiled_qmv_affine_concat3_t1";
      if (use_r4 && b0 == b1 && b1 == b2 &&
          (b0 == 4U || b0 == 5U || b0 == 6U)) {
        fn = (b0 == 5U)   ? "omlx_m5_pretiled_qmv_affine_concat3_q5_r4_t1"
             : (b0 == 4U) ? "omlx_m5_pretiled_qmv_affine_concat3_q4_r4_t1"
                          : "omlx_m5_pretiled_qmv_affine_concat3_q6_r4_t1";
      } else if (all_q456) {
        fn = use_r4 ? "omlx_m5_pretiled_qmv_affine_concat3_q456_r4_t1"
                    : "omlx_m5_pretiled_qmv_affine_concat3_q456_r2_t1";
      }
      auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                       {.width = 64U},
                       {get("norm"), d0, h0, s0, d1, h1, s1, d2, h2, s2,
                        get("query"), get("key"), get("value")});
      e.dynamic(d, 13, words({2U * q, kv, kv, h, packed_bits}));
      for (std::size_t i = 0; i < 3; ++i) {
        e.record_weight(parts[i], part_ns[i], h, "pretiled_qmv_t1",
                        "pretiled_affine_qmv_concat3_t1_v3", {},
                        "legacy_route_v1", {}, false, 1U);
      }
      return true;
    }
    return false;
  }

  bool try_concat_qkv_nax(Encoder &e, std::optional<U32> explicit_t) {
    const U32 total_n = 2U * q + 2U * kv;
    const std::array<MaterializedWeightView, 3> parts{q_proj, k_proj, v_proj};
    const std::array<U32, 3> part_ns{2U * q, kv, kv};
    if (use_mxfp4_nax(q_proj) && mtp_mxfp4_cache) {
      if (const auto *tw = mtp_mxfp4_cache->find(kMtpConcatQkvId);
          tw && tw->n == total_n && tw->k == h) {
        linear_mxfp4_nax(e, *tw, get("norm"), get("qkv_concat"), f16_scratch,
                         f16_row_scale, explicit_t, gpu_cores);
        const std::string grp = mxfp4_nax_group(e, *tw, explicit_t, gpu_cores);
        for (std::size_t i = 0; i < 3; ++i) {
          e.record_weight(parts[i], part_ns[i], h, "mxfp4_nax_m5_v1",
                          "mxfp4_nax_m5_v1", {}, "linear_row_exact_portable_v1",
                          grp, true, explicit_t);
        }
        return true;
      }
    }
    if (use_affine_nax(q_proj) && mtp_affine_cache) {
      if (const auto *tw = mtp_affine_cache->find(kMtpConcatQkvId);
          tw && tw->n == total_n && tw->k == h) {
        linear_affine_nax(e, *tw, get("norm"), affine_nax_xs, get("qkv_concat"),
                          explicit_t, gpu_cores);
        const std::string grp = affine_nax_group(e, *tw, explicit_t, gpu_cores);
        for (std::size_t i = 0; i < 3; ++i) {
          e.record_weight(parts[i], part_ns[i], h, "affine_nax_m5_v1",
                          "affine_nax_m5_v1", {},
                          "linear_row_exact_portable_v1", grp, true,
                          explicit_t);
        }
        return true;
      }
    }
    return false;
  }

  void emit_split3(Encoder &e, Ref src, Ref dst0, Ref dst1, Ref dst2, U32 n0,
                   U32 n1, U32 n2, std::optional<U32> explicit_t) {
    const U32 t = explicit_t.value_or(e.profile.capacity);
    auto &d = e.emit("qwen_forward", "omlx_m5_forward_split3_cols",
                     {.width = U64(t) * (n0 + n1 + n2)}, {.width = 128U},
                     {src, dst0, dst1, dst2});
    e.dynamic(d, 4, words({t, n0, n1, n2}),
              explicit_t.has_value() ? std::optional<U32>{}
                                     : std::optional<U32>{0U});
  }

  void emit_linear(Encoder &e, const MaterializedWeightView &w, Ref x, Ref y,
                   U32 n, U32 k, std::optional<U32> explicit_t) {
    const U32 t = explicit_t.value_or(e.profile.capacity);
    if (explicit_t.has_value() && *explicit_t == 1U && w.row_count == n &&
        w.in_features == k) {
      Ref mx_c{}, mx_s{};
      if (resolve_pretiled_mxfp4_t1(w, 1U, &mx_c, &mx_s)) {
        const U32 rows_per_tg = (n >= 2048U) ? 8U : 4U;
        const U32 tgs = (n + rows_per_tg - 1U) / rows_per_tg;
        const char *fn =
            (n >= 65536U)  ? "omlx_m5_pretiled_qmv_mxfp4_bn32_r4_t1"
            : (n >= 2048U) ? "omlx_m5_pretiled_qmv_mxfp4_bn16_r4_t1"
                           : "omlx_m5_pretiled_qmv_mxfp4_bn16_r2_t1";
        auto &d = e.emit("row_exact", fn, {.width = U64(tgs) * 64U},
                         {.width = 64U}, {x, mx_c, mx_s, y});
        e.dynamic(d, 4, words({n, k, U32(w.padded_k), 1U}));
        e.record_weight(w, n, k, "pretiled_qmv_t1", "pretiled_mxfp4_qmv_t1_v3",
                        {}, "legacy_route_v1", {}, false, explicit_t);
        return;
      }
      Ref aff_c{}, aff_h{}, aff_sb{};
      if (resolve_pretiled_affine_t1(w, 1U, &aff_c, &aff_h, &aff_sb)) {
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
        e.dynamic(d, 5, words({n, 1U, 0U, k, bits}));
        e.record_weight(w, n, k, "pretiled_qmv_t1", "pretiled_affine_qmv_t1_v3",
                        {}, "legacy_route_v1", {}, false, explicit_t);
        return;
      }
    }
    if (use_mxfp4_nax(w)) {
      if (const auto *tw = find_mxfp4_nax(w); tw && tw->n == n && tw->k == k) {
        linear_mxfp4_nax(e, *tw, x, y, f16_scratch, f16_row_scale, explicit_t,
                         gpu_cores);
        e.record_weight(
            w, n, k, "mxfp4_nax_m5_v1", "mxfp4_nax_m5_v1", {},
            "linear_row_exact_portable_v1",
            "mxfp4.nax.f16_bk128.n" + std::to_string(tw->bn) + ".sk" +
                std::to_string(tw->split_k) + ".v1",
            true, explicit_t);
        return;
      }
    }
    if (use_affine_nax(w)) {
      if (const auto *tw = find_affine_nax(w); tw && tw->n == n && tw->k == k) {
        linear_affine_nax(e, *tw, x, affine_nax_xs, y, explicit_t, gpu_cores);
        e.record_weight(
            w, n, k, "affine_nax_m5_v1", "affine_nax_m5_v1", {},
            "linear_row_exact_portable_v1",
            affine_nax_group(e, *tw, explicit_t, gpu_cores),
            true, explicit_t);
        return;
      }
    }
    if (bf16_nax_enabled && bf16_nax_weight_eligible(w, n, k)) {
      linear_bf16_nax(e, w, x, y, n, k, explicit_t, gpu_cores);
      return;
    }
    const auto decision = resolve_linear_route(
        w, n, k, t, context.capabilities().device_name, m4_short_routes, {},
        linear_policy, context.capabilities().profile.probe.apple_gpu_family);
    detail::forward::linear(e, w, x, y, n, k, get("sums"), get("zero"),
                            decision.route, explicit_t);
    e.record_weight(w, n, k, decision.route.family, decision.source,
                    decision.fallback_reason, decision.route.numeric_contract_id,
                    decision.route.row_exact_group, decision.route.row_exact,
                    explicit_t);
  }

  void emit_norm(Encoder &e, const MaterializedWeightView &w, float w_off, Ref x,
                 Ref y, U32 rows, bool dynamic_rows) {
    auto &d = e.emit("rmsnorm_projection", "omlx_m5_rmsnorm_a16",
                     {.width = 128U, .height = rows}, {.width = 128U},
                     {x, direct(w, h), y});
    e.dynamic(d, 3,
              words({h, rows, word(float(config.rms_norm_eps)), word(w_off)}),
              dynamic_rows ? std::optional<U32>(1U) : std::nullopt);
  }

  void emit_norm_init_zero_t1(Encoder &e, const MaterializedWeightView &w,
                              float w_off, Ref x, Ref y) {
    auto &d = e.emit("rmsnorm_projection", "omlx_m5_rmsnorm_a16_init_zero_t1",
                     {.width = 128U, .height = 1U}, {.width = 128U},
                     {x, direct(w, h), y, get("rms_sq"), get("rms_moe_dummy")});
    e.dynamic(d, 5,
              words({h, 2U, word(float(config.rms_norm_eps)), word(w_off), 0U}));
  }

  void emit_add_norm(Encoder &e, const MaterializedWeightView &w, float w_off,
                     Ref x, Ref res_delta, Ref y, U32 rows, bool dynamic_rows) {
    auto &d = e.emit("rmsnorm_projection", "omlx_m5_add_rmsnorm_a16",
                     {.width = 128U, .height = rows}, {.width = 128U},
                     {x, res_delta, direct(w, h), y});
    e.dynamic(d, 4,
              words({h, rows, word(float(config.rms_norm_eps)), word(w_off)}),
              dynamic_rows ? std::optional<U32>(1U) : std::nullopt);
  }

  StepPatchOffsets emit_mtp_attention_core(Encoder &e, const Impl &sp,
                                           Ref slot_q, Ref slot_k, Ref slot_v,
                                           Ref slot_mixer, Ref attn_pos_ref,
                                           U32 rows, bool dynamic_rows,
                                           U32 unique_tag) {
    StepPatchOffsets patches{};
    const U32 rotary_dim =
        U32(std::llround(config.head_dim * config.partial_rotary_factor));
    auto &post = e.emit(
        "mtp_forward", "omlx_m5_mtp_qk_norm_rope",
        {.width = 32U,
         .height = U64(rows) *
                   U64(config.num_attention_heads + config.num_key_value_heads)},
        {.width = 32U},
        {slot_q, slot_k, direct(q_norm, U64(config.head_dim)),
         direct(k_norm, U64(config.head_dim))});
    patches.qk_rope_offset = e.dynamic(
        post, 4,
        words({U32(config.num_attention_heads),
               U32(config.num_key_value_heads), U32(config.head_dim),
               rotary_dim, rows, q * 2U, U32(config.head_dim * 2), kv,
               U32(config.head_dim), unique_tag,
               word(float(config.rms_norm_eps)), word(float(config.rope_theta)),
               word(norm_offsets.q_norm), word(norm_offsets.k_norm)}),
        dynamic_rows ? std::optional<U32>(4U) : std::nullopt);

    Ref table_ref{&sp.mtp_kv_pool->table_buffer(), 0,
                  sp.mtp_kv_pool->table_buffer().size()};
    Ref gate_ref = sub(slot_q, U64(config.head_dim) * 2U);
    auto attn_params = words({U32(config.num_attention_heads),
                              U32(config.num_key_value_heads),
                              U32(config.head_dim),
                              rows,
                              unique_tag,
                              sp.max_context,
                              64U,
                              sp.mtp_kv_pool->physical_pages(),
                              q * 2U,
                              U32(config.head_dim * 2),
                              kv,
                              kv,
                              q * 2U,
                              U32(config.head_dim * 2)});
    const std::optional<U32> tok_w3 =
        dynamic_rows ? std::optional<U32>(3U) : std::nullopt;
    const std::optional<U32> pos_w4 =
        dynamic_rows ? std::optional<U32>(4U) : std::nullopt;

    std::vector<std::pair<U64, U32>> raw_pos_patches;
    auto *pos_patch_ptr = dynamic_rows ? nullptr : &raw_pos_patches;

    e.clear_dynamic_parameter_cache();
    if (sp.kv_format == KvStorageFormat::Affine4Group128 && sp.affine4_nax_ok) {
      emit_paged_kv_append_affine4_g128(
          e, config, rows, slot_k, slot_v, sp.get("mtp_cache_key"),
          sp.get("mtp_cache_value"), table_ref, attn_params, tok_w3, pos_w4,
          pos_patch_ptr);
      e.clear_dynamic_parameter_cache();
      if (dynamic_rows && rows >= 16U && config.head_dim == 256 &&
          sp.flash_nax_prefill_ok) {
        emit_paged_flash_prefill_affine4_g128(
            e, config, rows, slot_q, gate_ref, sp.get("mtp_cache_key"),
            sp.get("mtp_cache_value"), slot_mixer, table_ref, attn_pos_ref,
            std::move(attn_params));
      } else {
        const bool use_split = sp.split_attn_ok && rows <= 8U;
        emit_paged_attention_affine4_g128(
            e, config, rows, use_split, sp.max_context, sp.max_context,
            sp.mtp_kv_pool->physical_pages(), slot_q, gate_ref,
            sp.get("mtp_cache_key"), sp.get("mtp_cache_value"), slot_mixer,
            table_ref, attn_pos_ref,
            use_split ? sp.get("mtp_split_max") : Ref{},
            use_split ? sp.get("mtp_split_sum") : Ref{},
            use_split ? sp.get("mtp_split_out") : Ref{}, std::move(attn_params),
            tok_w3, pos_w4, pos_patch_ptr, false);
      }
    } else if (sp.kv_format == KvStorageFormat::Affine4Group128) {
      auto &app = e.emit(
          "qwen_forward", "omlx_m5_forward_paged_kv_append_affine4",
          {.width = 32U, .height = U64(rows) * U64(config.num_key_value_heads)},
          {.width = 32U},
          {slot_k, slot_v, sp.get("mtp_cache_key"), sp.get("mtp_cache_value"),
           table_ref});
      const U64 off_app = e.dynamic(app, 5, attn_params, tok_w3, pos_w4);
      if (!dynamic_rows) {
        mtp_push_unique(patches.attn_param_offsets, off_app);
      }
      e.clear_dynamic_parameter_cache();
      auto &att = e.emit(
          "qwen_forward", "omlx_m5_forward_paged_causal_attention_affine4_exact",
          {.width = 32U, .height = U64(rows) * U64(config.num_attention_heads)},
          {.width = 32U},
          {slot_q, gate_ref, sp.get("mtp_cache_key"), sp.get("mtp_cache_value"),
           slot_mixer, table_ref});
      const U64 off_att =
          e.dynamic(att, 6, std::move(attn_params), tok_w3, pos_w4);
      if (!dynamic_rows) {
        mtp_push_unique(patches.attn_param_offsets, off_att);
      }
    } else {
      auto &app = e.emit("attention_postprocess", "omlx_m5_paged_kv_append",
                         {.width = U64(rows) * kv}, {.width = 128U},
                         {slot_k, slot_v, sp.get("mtp_cache_key"),
                          sp.get("mtp_cache_value"), table_ref});
      const U64 off_app = e.dynamic(app, 5, attn_params, tok_w3, pos_w4);
      if (!dynamic_rows) {
        mtp_push_unique(patches.attn_param_offsets, off_app);
      }
      e.clear_dynamic_parameter_cache();
      if (dynamic_rows && rows >= 16U && sp.flash_nax_prefill_ok) {
        emit_paged_flash_prefill_nax(
            e, config, rows, slot_q, gate_ref, sp.get("mtp_cache_key"),
            sp.get("mtp_cache_value"), slot_mixer, table_ref, attn_pos_ref,
            std::move(attn_params));
      } else if (sp.split_attn_ok && rows <= 8U) {
        const U32 chunk_tokens =
            exact_attention_split_chunk_tokens(sp.max_context, 32U);
        const U32 splits =
            exact_attention_split_count(sp.max_context, sp.max_context, 32U);
        auto split_params = words({U32(config.num_attention_heads),
                                   U32(config.num_key_value_heads),
                                   U32(config.head_dim),
                                   rows,
                                   unique_tag,
                                   sp.max_context,
                                   64U,
                                   sp.mtp_kv_pool->physical_pages(),
                                   q * 2U,
                                   U32(config.head_dim * 2),
                                   kv,
                                   kv,
                                   q * 2U,
                                   U32(config.head_dim * 2),
                                   splits,
                                   chunk_tokens});
        auto &partial = e.emit(
            "attention_split_kv", "omlx_m5_paged_attention_split_partial",
            {.width = U64(splits) * 32U,
             .height = U64(rows) * U64(config.num_attention_heads)},
            {.width = 32U},
            {slot_q, sp.get("mtp_cache_key"), sp.get("mtp_cache_value"),
             sp.get("mtp_split_max"), sp.get("mtp_split_sum"),
             sp.get("mtp_split_out"), table_ref});
        const U64 off_p = e.dynamic(partial, 7, split_params, tok_w3, pos_w4);
        auto &merge = e.emit(
            "attention_split_kv", "omlx_m5_paged_attention_split_merge",
            {.width = 32U,
             .height = U64(rows) * U64(config.num_attention_heads)},
            {.width = 32U},
            {gate_ref, sp.get("mtp_split_max"), sp.get("mtp_split_sum"),
             sp.get("mtp_split_out"), slot_mixer});
        const U64 off_m =
            e.dynamic(merge, 5, std::move(split_params), tok_w3, pos_w4);
        if (!dynamic_rows) {
          mtp_push_unique(patches.attn_param_offsets, off_p);
          mtp_push_unique(patches.attn_param_offsets, off_m);
        }
      } else {
        auto &att = e.emit(
            "attention_postprocess", "omlx_m5_paged_causal_attention",
            {.width = 32U,
             .height = U64(rows) * U64(config.num_attention_heads)},
            {.width = 32U},
            {slot_q, gate_ref, sp.get("mtp_cache_key"),
             sp.get("mtp_cache_value"), slot_mixer, table_ref});
        const U64 off_att =
            e.dynamic(att, 6, std::move(attn_params), tok_w3, pos_w4);
        if (!dynamic_rows) {
          mtp_push_unique(patches.attn_param_offsets, off_att);
        }
      }
    }

    for (const auto &[off, _] : raw_pos_patches) {
      mtp_push_unique(patches.attn_param_offsets, off);
    }
    return patches;
  }

  StepPatchOffsets emit_mtp_layer0(Encoder &e, U32 rows, bool dynamic_rows,
                                   bool defer_final_norm = false,
                                   bool *out_deferred_final_norm = nullptr,
                                   Ref custom_attn_pos = {}) {
    if (out_deferred_final_norm) {
      *out_deferred_final_norm = false;
    }
    const std::optional<U32> explicit_t =
        dynamic_rows ? std::nullopt : std::optional<U32>(rows);

    if (rows == 1U && !dynamic_rows) {
      emit_norm_init_zero_t1(e, input_norm, norm_offsets.input_layernorm,
                             get("hidden"), get("norm"));
    } else {
      emit_norm(e, input_norm, norm_offsets.input_layernorm, get("hidden"),
                get("norm"), rows, dynamic_rows);
    }

    if (rows == 1U && !dynamic_rows && try_concat3_qkv_t1(e)) {
      // Q, K, V emitted in a single fused KL32 concat3 dispatch.
    } else if (try_concat_qkv_nax(e, explicit_t)) {
      emit_split3(e, get("qkv_concat"), get("query"), get("key"), get("value"),
                  2U * q, kv, kv, explicit_t);
    } else {
      emit_linear(e, q_proj, get("norm"), get("query"), 2U * q, h, explicit_t);
      emit_linear(e, k_proj, get("norm"), get("key"), kv, h, explicit_t);
      emit_linear(e, v_proj, get("norm"), get("value"), kv, h, explicit_t);
    }

    Ref attn_pos_ref =
        custom_attn_pos.buffer ? custom_attn_pos : get("mtp_attn_pos");
    StepPatchOffsets patches =
        emit_mtp_attention_core(e, *this, get("query"), get("key"),
                                get("value"), get("mixer"), attn_pos_ref, rows,
                                dynamic_rows, 0U);

    if (!moe_layer && rows == 1U && !dynamic_rows &&
        can_linear_swiglu_norm_t1(mlp_gate, mlp_up, f, h) &&
        try_linear_add_sq_t1(e, o_proj, get("mixer"), get("hidden"), h, q,
                             0U)) {
      (void)try_linear_swiglu_norm_t1(
          e, mlp_gate, mlp_up, post_attn_norm,
          norm_offsets.post_attention_layernorm, get("hidden"), get("product"),
          f, h, 0U);
      if (defer_final_norm && can_linear_norm_t1(head, v, h) &&
          try_linear_add_sq_t1(e, mlp_down, get("product"), get("hidden"), h, f,
                               1U)) {
        if (out_deferred_final_norm) {
          *out_deferred_final_norm = true;
        }
        return patches;
      }
      emit_linear(e, mlp_down, get("product"), get("delta"), h, f, 1U);
      emit_add_norm(e, mtp_final_norm, norm_offsets.final_norm, get("hidden"),
                    get("delta"), get("norm"), 1U, false);
      return patches;
    }

    emit_linear(e, o_proj, get("mixer"), get("delta"), h, q, explicit_t);
    emit_add_norm(e, post_attn_norm, norm_offsets.post_attention_layernorm,
                  get("hidden"), get("delta"), get("norm"), rows, dynamic_rows);

    // Layer 0 MLP (dense or MoE) + final mtp.norm -> get("norm")[0..rows-1, h].
    if (!moe_layer) {
      if (!try_linear_swiglu(e, mlp_gate, mlp_up, get("norm"), get("product"),
                             f, h, explicit_t)) {
        emit_linear(e, mlp_gate, get("norm"), get("gate"), f, h, explicit_t);
        emit_linear(e, mlp_up, get("norm"), get("up"), f, h, explicit_t);
        auto &sw = e.emit("mlp", "omlx_m5_swiglu_a16",
                          {.width = 32U, .height = U64(rows) * (f / 32U)},
                          {.width = 32U},
                          {get("gate"), get("up"), get("product")});
        if (dynamic_rows) {
          e.dynamic(sw, 3, words({rows, f, f / 32U, 0U}), 0U, {}, 1U);
        } else {
          e.dynamic(sw, 3, words({rows, f, f / 32U, 0U}));
        }
      }
      emit_linear(e, mlp_down, get("product"), get("delta"), h, f, explicit_t);
    } else {
      emit_moe_mlp(e, rows, dynamic_rows, explicit_t);
    }

    emit_add_norm(e, mtp_final_norm, norm_offsets.final_norm, get("hidden"),
                  get("delta"), get("norm"), rows, dynamic_rows);
    return patches;
  }

  void emit_moe_mlp(Encoder &e, U32 rows, bool dynamic_rows,
                    std::optional<U32> explicit_t) {
    const std::optional<U32> tok_w0 =
        dynamic_rows ? std::optional<U32>(0U) : std::nullopt;
    const std::optional<U32> tok_w5 =
        dynamic_rows ? std::optional<U32>(5U) : std::nullopt;

    emit_linear(e, moe_router, get("norm"), get("router"), 256U, h, explicit_t);
    auto &rt = e.emit("qwen_forward", "omlx_m5_forward_router_top8_logits",
                      {.width = 32U, .height = rows}, {.width = 32U},
                      {get("router"), get("route_ids"), get("route_scores")});
    e.dynamic(rt, 3, words({rows, 256U}), tok_w0);

    bool handled = false;
    if (dynamic_rows && moe_nax_ws.active &&
        emit_moe_nax_ffn(e, moe_nax_ws, moe_gate, moe_up, moe_down, get("norm"),
                         get("route_ids"), get("route_scores"),
                         get("expert_output"), get("routed"), rows, h, f, 8U,
                         256U, gpu_cores)) {
      handled = true;
    }
    if (!handled && moe_gate.format == "mxfp4_e2m1_g32_e8m0_v1" &&
        moe_down.format == "mxfp4_e2m1_g32_e8m0_v1") {
      const auto up_params =
          words({256U, 8U, f, h, U32(moe_gate.padded_k), rows, 0U, 0U});
      const auto down_params =
          words({256U, 8U, h, f, U32(moe_down.padded_k), rows, 1U, 1U});
      auto &d1 = e.emit("moe", "omlx_m5_mxfp4_moe_w4a16_swiglu_grouped",
                        {.width = U64(rows) * 8U * ((f + 3U) / 4U) * 128U},
                        {.width = 128U},
                        {get("norm"), get("route_ids"), plane(moe_gate, "data"),
                         plane(moe_gate, "scale"), plane(moe_up, "data"),
                         plane(moe_up, "scale"), get("product")});
      e.dynamic(d1, 7, up_params, tok_w5);
      auto &d2 = e.emit("moe", "omlx_m5_mxfp4_moe_w4a16_down_reduce",
                        {.width = U64(rows) * ((h + 7U) / 8U) * 256U},
                        {.width = 256U},
                        {get("product"), get("route_ids"), get("route_scores"),
                         plane(moe_down, "data"), plane(moe_down, "scale"),
                         get("routed")});
      e.dynamic(d2, 6, down_params, tok_w5);
      handled = true;
    }
    if (!handled && moe_gate.format.starts_with("oq_affine_q") &&
        moe_down.format.starts_with("oq_affine_q")) {
      const U32 u_bits = U32(std::stoul(moe_gate.format.substr(11)));
      const U32 u_group =
          U32(std::stoul(moe_gate.format.substr(moe_gate.format.find("_g") + 2)));
      const U32 d_bits = U32(std::stoul(moe_down.format.substr(11)));
      const U32 d_group =
          U32(std::stoul(moe_down.format.substr(moe_down.format.find("_g") + 2)));
      if (u_bits >= 4U && d_bits >= 4U) {
        const auto up_params =
            words({256U, 8U, f, h, U32(moe_gate.padded_k), rows, 0U, 0U});
        const auto down_params =
            words({256U, 8U, h, f, U32(moe_down.padded_k), rows, 1U, 1U});
        Ref g_high = u_bits == 5U   ? plane(moe_gate, "high_bit")
                     : u_bits == 6U ? plane(moe_gate, "high_2bit")
                                    : get("zero");
        Ref u_high = u_bits == 5U   ? plane(moe_up, "high_bit")
                     : u_bits == 6U ? plane(moe_up, "high_2bit")
                                    : get("zero");
        Ref d_high = d_bits == 5U   ? plane(moe_down, "high_bit")
                     : d_bits == 6U ? plane(moe_down, "high_2bit")
                                    : get("zero");
        auto &s1 =
            e.emit("moe", "omlx_m5_moe_group_sum_g" + std::to_string(u_group),
                   {.width = U64(rows) * (h / u_group) * 32U}, {.width = 32U},
                   {get("norm"), get("sums")});
        e.dynamic(s1, 2, up_params, tok_w5);
        auto &d1 = e.emit(
            "moe",
            "omlx_m5_oq_q" + std::to_string(u_bits) + "_g" +
                std::to_string(u_group) + "_moe_swiglu_grouped",
            {.width = U64(rows) * 8U * ((f + 3U) / 4U) * 128U}, {.width = 128U},
            {get("norm"), get("route_ids"),
             plane(moe_gate, u_bits == 8U ? "code" : "base_code"), g_high,
             plane(moe_gate, "scale"), plane(moe_gate, "bias"),
             plane(moe_up, u_bits == 8U ? "code" : "base_code"), u_high,
             plane(moe_up, "scale"), plane(moe_up, "bias"), get("sums"),
             get("product")});
        e.dynamic(d1, 12, up_params, tok_w5);

        auto &s2 =
            e.emit("moe", "omlx_m5_moe_group_sum_g" + std::to_string(d_group),
                   {.width = U64(rows) * 8U * (f / d_group) * 32U},
                   {.width = 32U}, {get("product"), get("sums")});
        e.dynamic(s2, 2, down_params, tok_w5);
        auto &d2 = e.emit(
            "moe",
            "omlx_m5_oq_q" + std::to_string(d_bits) + "_g" +
                std::to_string(d_group) + "_moe_down_reduce",
            {.width = U64(rows) * h * 256U}, {.width = 256U},
            {get("product"), get("route_ids"), get("route_scores"),
             plane(moe_down, d_bits == 8U ? "code" : "base_code"), d_high,
             plane(moe_down, "scale"), plane(moe_down, "bias"), get("sums"),
             get("routed")});
        e.dynamic(d2, 9, down_params, tok_w5);
        handled = true;
      }
    }
    if (!handled) {
      throw std::invalid_argument("Unsupported MTP MoE expert weight format: " +
                                  moe_gate.format);
    }

    if (!try_linear_swiglu(e, shared_gate_proj, shared_up_proj, get("norm"),
                           get("product"), shared_f, h, explicit_t)) {
      emit_linear(e, shared_gate_proj, get("norm"), get("gate"), shared_f, h,
                  explicit_t);
      emit_linear(e, shared_up_proj, get("norm"), get("up"), shared_f, h,
                  explicit_t);
      auto &sw = e.emit("mlp", "omlx_m5_swiglu_a16",
                        {.width = 32U, .height = U64(rows) * (shared_f / 32U)},
                        {.width = 32U},
                        {get("gate"), get("up"), get("product")});
      if (dynamic_rows) {
        e.dynamic(sw, 3, words({rows, shared_f, shared_f / 32U, 0U}), 0U, {},
                  1U);
      } else {
        e.dynamic(sw, 3, words({rows, shared_f, shared_f / 32U, 0U}));
      }
    }
    emit_linear(e, shared_down_proj, get("product"), get("shared"), h, shared_f,
                explicit_t);
    emit_linear(e, shared_expert_gate, get("norm"), get("shared_gate"), 1U, h,
                explicit_t);
    auto &comb = e.emit("mlp", "omlx_m5_moe_shared_combine",
                        {.width = U64(rows) * h}, {.width = 128U},
                        {get("routed"), get("shared"), get("shared_gate"),
                         get("delta")});
    e.dynamic(comb, 4, words({rows, 8U, h, 0U}), tok_w0);
  }

  bool use_fc_dual_bf16(U32 rows) const {
    if (fc_weight.format != "direct_bf16_v1") {
      return false;
    }
    if (bf16_nax_enabled && rows >= 8U &&
        bf16_nax_weight_eligible(fc_weight, h, 2U * h)) {
      return false;
    }
    return true;
  }

  void emit_embedding_only(Encoder &e, Ref tok_buf, U32 rows, bool dynamic_rows,
                           U64 *out_embed_off = nullptr) {
    std::string emb_fn = "omlx_m5_embedding_";
    auto emb_params =
        words({v, 0U, v, h, U32(embedding.padded_k), rows, 0U, 0U});
    const std::optional<U32> emb_tok_patch =
        dynamic_rows ? std::optional<U32>(5U) : std::nullopt;
    U64 emb_off = 0;
    if (embedding.format == "direct_bf16_v1") {
      auto &d = e.emit("vocab", emb_fn + "bf16", {.width = U64(rows) * h},
                       {.width = 256U},
                       {tok_buf, plane(embedding, "data"), get("embed_out")});
      emb_off = e.dynamic(d, 3, emb_params, emb_tok_patch);
    } else if (embedding.format == "mxfp4_e2m1_g32_e8m0_v1") {
      auto &d = e.emit("vocab", emb_fn + "mxfp4", {.width = U64(rows) * h},
                       {.width = 256U},
                       {tok_buf, plane(embedding, "data"),
                        plane(embedding, "scale"), get("embed_out")});
      emb_off = e.dynamic(d, 4, emb_params, emb_tok_patch);
    } else if (embedding.format.starts_with("oq_affine_q")) {
      const U32 bits = U32(std::stoul(embedding.format.substr(11)));
      const U32 group = U32(std::stoul(
          embedding.format.substr(embedding.format.find("_g") + 2)));
      if (bits == 3U) {
        auto &d =
            e.emit("qwen_forward",
                   "omlx_m5_forward_embedding_q3_g" + std::to_string(group),
                   {.width = U64(rows) * h}, {.width = 256U},
                   {tok_buf, plane(embedding, "low_2bit"),
                    plane(embedding, "high_bit"), plane(embedding, "scale"),
                    plane(embedding, "bias"), get("embed_out")});
        emb_off = e.dynamic(d, 6, emb_params, emb_tok_patch);
      } else {
        Ref high = bits == 5U   ? plane(embedding, "high_bit")
                   : bits == 6U ? plane(embedding, "high_2bit")
                                : get("zero");
        auto &d = e.emit(
            "vocab",
            emb_fn + "q" + std::to_string(bits) + "_g" + std::to_string(group),
            {.width = U64(rows) * h}, {.width = 256U},
            {tok_buf, plane(embedding, bits == 8U ? "code" : "base_code"), high,
             plane(embedding, "scale"), plane(embedding, "bias"),
             get("embed_out")});
        emb_off = e.dynamic(d, 6, emb_params, emb_tok_patch);
      }
    } else {
      throw std::invalid_argument("Unsupported MTP shared embedding format: " +
                                  embedding.format);
    }
    if (out_embed_off) {
      *out_embed_off = emb_off;
    }
  }

  void emit_fc_only(Encoder &e, U32 rows, bool dynamic_rows,
                    bool *out_has_fc_dual = nullptr,
                    U64 *out_fc_dual_off = nullptr) {
    const bool dual_fc = use_fc_dual_bf16(rows);
    if (dual_fc) {
      const bool use_2col_t1 =
          !dynamic_rows && rows == 1U && (h & 127U) == 0U;
      const MtpFcDualParamsHost fc_p{
          .n = h, .h = h, .rows = rows, .reserved = use_2col_t1 ? 1U : 0U};
      const U64 grid_w = use_2col_t1 ? U64((h + 7U) / 8U) * 128U
                                     : U64((h + 3U) / 4U) * 128U;
      const U64 grid_h = use_2col_t1 ? 1ULL : U64((rows + 7U) / 8U);
      auto &fcd = e.emit("mtp_forward", "omlx_m5_mtp_fc_dual_bf16",
                         {.width = grid_w, .height = grid_h}, {.width = 128U},
                         {get("stem_e"), get("stem_h"),
                          plane(fc_weight, "data"), get("hidden")});
      const U64 fc_off = e.dynamic(
          fcd, 4, struct_bytes(fc_p),
          dynamic_rows ? std::optional<U32>(2U) : std::nullopt);
      e.record_weight(fc_weight, h, 2U * h, "mtp_fc_dual_bf16",
                      "mtp_fc_dual_bf16", {}, "linear_row_exact_portable_v1",
                      "bf16.simd32.sequential_fma.v1", true,
                      dynamic_rows ? std::nullopt : std::optional<U32>(rows));
      if (out_has_fc_dual) {
        *out_has_fc_dual = true;
      }
      if (out_fc_dual_off) {
        *out_fc_dual_off = fc_off;
      }
    } else {
      if (out_has_fc_dual) {
        *out_has_fc_dual = false;
      }
      emit_linear(e, fc_weight, get("stem_concat"), get("hidden"), h, 2U * h,
                  dynamic_rows ? std::nullopt : std::optional<U32>(rows));
    }
  }

  void emit_stem(Encoder &e, Ref tok_buf, Ref trunk_buf, U32 rows,
                 U32 cond_mode, bool dynamic_rows, U64 *out_embed_off = nullptr,
                 U64 *out_stem_off = nullptr, bool *out_has_fc_dual = nullptr,
                 U64 *out_fc_dual_off = nullptr,
                 Ref chain_hidden_override = {}) {
    emit_embedding_only(e, tok_buf, rows, dynamic_rows, out_embed_off);

    // 2. Fused stem RMSNorms (pre_fc_norm_embedding + target_final_norm/pre_fc_norm_hidden).
    const bool dual_fc = use_fc_dual_bf16(rows);
    const MtpStemNormsParamsHost stem_p{
        .hidden = h,
        .rows = rows,
        .cond_mode = cond_mode,
        .write_concat = dual_fc ? 0U : 1U,
        .trunk_row_base = 0U,
        .epsilon = float(config.rms_norm_eps),
        .target_norm_offset = target_norm_offset,
        .emb_norm_offset = norm_offsets.pre_fc_norm_embedding,
        .hid_norm_offset = norm_offsets.pre_fc_norm_hidden,
    };
    const Ref chain_ref =
        chain_hidden_override.buffer ? chain_hidden_override : get("chain_hidden");
    auto &sn = e.emit("mtp_forward", "omlx_m5_mtp_stem_norms",
                      {.width = 128U, .height = rows, .depth = 2U},
                      {.width = 128U, .height = 1U, .depth = 1U},
                      {get("embed_out"), trunk_buf, get("pending_trunk_normed"),
                       chain_ref, direct(target_final_norm, h),
                       direct(pre_fc_norm_embedding, h),
                       direct(pre_fc_norm_hidden, h), get("stem_e"),
                       get("stem_h"), get("stem_concat")});
    const U64 st_off = e.dynamic(
        sn, 10, struct_bytes(stem_p),
        dynamic_rows ? std::optional<U32>(1U) : std::nullopt);
    if (out_stem_off) {
      *out_stem_off = st_off;
    }

    // 3. mtp.fc projection ([e, h] -> hidden).
    emit_fc_only(e, rows, dynamic_rows, out_has_fc_dual, out_fc_dual_off);
  }

  std::string plan_key_suffix() const {
    Fnv1a64 sig;
    auto add_w = [&](const MaterializedWeightView &w) {
      sig.weight(w);
      for (const auto &pl : w.planes) {
        sig.str(pl.kind);
        sig.u64(pl.bytes);
      }
    };
    sig.str(mtp_prefix);
    add_w(embedding);
    add_w(head);
    add_w(target_final_norm);
    add_w(pre_fc_norm_embedding);
    add_w(pre_fc_norm_hidden);
    add_w(fc_weight);
    add_w(input_norm);
    add_w(q_proj);
    add_w(k_proj);
    add_w(v_proj);
    add_w(o_proj);
    add_w(q_norm);
    add_w(k_norm);
    add_w(post_attn_norm);
    add_w(mtp_final_norm);
    if (moe_layer) {
      add_w(moe_router);
      add_w(moe_gate);
      add_w(moe_up);
      add_w(moe_down);
      add_w(shared_gate_proj);
      add_w(shared_up_proj);
      add_w(shared_down_proj);
      add_w(shared_expert_gate);
    } else {
      add_w(mlp_gate);
      add_w(mlp_up);
      add_w(mlp_down);
    }
    sig.u64(local_trunk_ref.bytes);
    sig.u64(delegated_trunk_ref.bytes);
    sig.u64(U64(priming_cap));
    sig.u64(U64(gpu_cores));
    sig.u64(m4_short_routes ? 1ULL : 0ULL);
    sig.u64(bf16_nax_enabled ? 1ULL : 0ULL);
    sig.u64(kl32_enabled ? 1ULL : 0ULL);
    sig.u64(moe_nax_ws.active ? 1ULL : 0ULL);
    sig.u64(f16_scratch.bytes);
    sig.u64(affine_nax_xs.bytes);
    sig.u64(U64(word(target_norm_offset)));
    sig.u64(U64(word(norm_offsets.pre_fc_norm_embedding)));
    sig.u64(U64(word(norm_offsets.pre_fc_norm_hidden)));
    sig.u64(U64(word(norm_offsets.input_layernorm)));
    sig.u64(U64(word(norm_offsets.q_norm)));
    sig.u64(U64(word(norm_offsets.k_norm)));
    sig.u64(U64(word(norm_offsets.post_attention_layernorm)));
    sig.u64(U64(word(norm_offsets.final_norm)));
    sig.u64(U64(max_context));
    sig.u64(U64(kv_format));
    sig.u64(U64(sampling_contract));
    sig.u64(U64(linear_policy));
    sig.u64(affine4_nax_ok ? 1ULL : 0ULL);
    sig.u64(flash_nax_prefill_ok ? 1ULL : 0ULL);
    sig.u64(split_attn_ok ? 1ULL : 0ULL);
    return ".mtp.v3." + sig.hex();
  }

  SaveTrunkProfile compile_save_trunk(Ref trunk_buf, std::string_view tag) {
    SaveTrunkProfile out;
    Encoder e(context, libraries, 1U,
              "mtp.save_trunk." + std::string(tag) + plan_key_suffix());
    e.begin_stage("save_trunk");
    const MtpSaveTrunkParamsHost p{
        .hidden = h,
        .src_row = 0U,
        .epsilon = float(config.rms_norm_eps),
        .weight_offset = target_norm_offset,
    };
    auto &d = e.emit("mtp_forward", "omlx_m5_mtp_save_trunk_row",
                     {.width = 128U, .height = 1U}, {.width = 128U},
                     {trunk_buf, direct(target_final_norm, h),
                      get("pending_trunk_normed")});
    out.save_params_offset = e.dynamic(d, 3, struct_bytes(p));
    out.profile = e.finish(true);
    return out;
  }

  PrimingProfile compile_priming(U32 cap, U32 cond_mode, Ref trunk_buf,
                                 bool is_delegated) {
    PrimingProfile out;
    out.capacity = cap;
    out.cond_mode = cond_mode;
    out.uses_delegated_trunk = is_delegated;
    Encoder e(context, libraries, cap,
              "mtp.prime.m" + std::to_string(cond_mode) +
                  (is_delegated ? ".del" : ".loc") + plan_key_suffix());
    e.begin_stage("prime_stem");
    emit_stem(e, get("fold_tokens"), trunk_buf, cap, cond_mode, true,
              &out.embed_offset, &out.stem_offset, &out.has_fc_dual,
              &out.fc_dual_offset);
    e.begin_stage("prime_layer0");
    out.attn_patches = emit_mtp_layer0(e, cap, true);
    out.profile = e.finish(true);
    return out;
  }

  FoldOnlyProfile compile_fold_only(U32 c_fold, U32 cond_mode) {
    FoldOnlyProfile out;
    out.fold_rows = c_fold;
    out.cond_mode = cond_mode;
    Encoder e(context, libraries, c_fold,
              "mtp.fold_only.m" + std::to_string(cond_mode) + ".c" +
                  std::to_string(c_fold) + plan_key_suffix());
    e.begin_stage("fold_stem");
    emit_stem(e, get("fold_tokens"), local_trunk_ref, c_fold, cond_mode, false);
    e.begin_stage("fold_layer0");
    out.attn_patches = emit_mtp_layer0(e, c_fold, false);
    out.profile = e.finish(true);
    return out;
  }

  ChainProfile compile_chain(U32 c_fold, U32 k_draft, U32 cond_mode) {
    ChainProfile out;
    out.fold_rows = c_fold;
    out.draft_k = k_draft;
    out.cond_mode = cond_mode;
    Encoder e(context, libraries, c_fold,
              "mtp.chain.m" + std::to_string(cond_mode) + ".c" +
                  std::to_string(c_fold) + ".k" + std::to_string(k_draft) +
                  plan_key_suffix());

    for (U32 d = 0U; d < k_draft; ++d) {
      const U32 step_rows = (d == 0U) ? c_fold : 1U;
      const U32 step_cond = (d == 0U) ? cond_mode : 2U;
      e.profile.capacity = step_rows;
      e.clear_dynamic_parameter_cache();

      e.begin_stage("chain.d" + std::to_string(d) + ".stem");
      Ref step_tok_buf = (d == 0U) ? get("fold_tokens") : get("chain_token");
      emit_stem(e, step_tok_buf, local_trunk_ref, step_rows, step_cond, false);

      e.begin_stage("chain.d" + std::to_string(d) + ".layer0");
      e.clear_dynamic_parameter_cache();
      bool deferred_final_norm = false;
      Ref step_attn_pos{get("mtp_attn_pos").buffer, U64(d) * 64ULL, 64ULL};
      out.steps[d] = emit_mtp_layer0(e, step_rows, false,
                                     /*defer_final_norm=*/true,
                                     &deferred_final_norm, step_attn_pos);

      e.begin_stage("chain.d" + std::to_string(d) + ".head_sample");
      e.profile.capacity = 1U;
      e.clear_dynamic_parameter_cache();
      if (deferred_final_norm) {
        (void)try_linear_norm_t1(
            e, head, mtp_final_norm, norm_offsets.final_norm,
            Ref{get("hidden").buffer, 0, mul(h, 2U)}, get("logits"), v, h, 1U);
      } else {
        Ref head_in = get("norm");
        if (step_rows > 1U) {
          auto &sl = e.emit("qwen_forward", "omlx_m5_forward_slice_rows",
                            {.width = h}, {.width = 128U},
                            {get("norm"), get("last_norm")});
          e.dynamic(sl, 2, words({1U, h, step_rows - 1U}));
          head_in = get("last_norm");
        } else {
          head_in = Ref{get("norm").buffer, 0, mul(h, 2U)};
        }
        emit_linear(e, head, head_in, get("logits"), v, h, 1U);
      }

      // 2-stage 64-TG greedy argmax + stochastic fallback on GPU:
      // Stage 1 fans out 64 TGs across all GPU cores when temperature == 0 or
      // top_k <= 1 (0.014 ms vs 0.28 ms 1-TG!), and early-exits in 0 DRAM reads
      // when stochastic so Stage 2 runs exact top-k/top-p sampling.
      Ref step_parts = get("argmax_parts");
      Ref step_sampled{get("sampled").buffer, U64(d) * 8U, 8U};
      Ref step_cids{get("compact_ids").buffer, U64(d) * 64U * 4U, 64U * 4U};
      Ref step_cscores{get("compact_scores").buffer, U64(d) * 64U * 4U,
                       64U * 4U};
      const std::vector<std::byte> srow_init(sizeof(PositionSamplingRow));
      auto &smp1 = e.emit(
          "position_sampling", "omlx_m5_position_sample_logits_stage1",
          {.width = 64U * 256U, .height = 1U}, {.width = 256U},
          {get("logits"), step_parts});
      out.steps[d].sample_offset = e.dynamic(smp1, 2, srow_init);
      e.constant(smp1, 3, words({v, 1U, 1U, 0U}));

      auto &smp2 = e.emit(
          "position_sampling", "omlx_m5_position_sample_logits_stage2",
          {.width = 256U, .height = 1U}, {.width = 256U},
          {get("logits"), step_parts, step_sampled, step_cids, step_cscores});
      (void)e.dynamic(smp2, 5, srow_init);
      e.constant(smp2, 6, words({v, 1U, 1U, 0U}));

      // Latch sampled draft token (+ seed chain_token & chain_hidden for d+1).
      const MtpLatchParamsHost latch_p{
          .hidden = h,
          .src_row = deferred_final_norm ? 0U : (step_rows - 1U),
          .step_index = d,
          .sampled_mode = 0U,
          .write_chain = (d + 1U < k_draft) ? 1U : 0U,
          .norm_from_sq = deferred_final_norm ? 1U : 0U,
          .sq_slot = 1U,
          .inv_k = 1.0F / float(h),
          .eps = float(config.rms_norm_eps),
          .w_off = norm_offsets.final_norm,
      };
      const U64 latch_w = (d + 1U < k_draft) ? U64(h) : 1ULL;
      Ref latch_src = deferred_final_norm ? get("hidden") : get("norm");
      auto &lt =
          e.emit("mtp_forward", "omlx_m5_mtp_latch_chain_step",
                 {.width = latch_w}, {.width = std::min<U64>(256ULL, latch_w)},
                 {get("sampled"), latch_src, direct(mtp_final_norm, h),
                  get("rms_sq"), get("proposal_out"), get("chain_token"),
                  get("chain_hidden")});
      e.dynamic(lt, 7, struct_bytes(latch_p));
    }
    e.profile.capacity = c_fold;
    out.profile = e.finish(true);
    return out;
  }

  static void patch_u32(Profile &prof, U64 base_off, U32 word_idx, U32 val) {
    std::memcpy(prof.parameter_data.data() + base_off + U64(word_idx) * 4U,
                &val, 4U);
  }

  static void patch_step_position(Profile &prof, const StepPatchOffsets &sp,
                                  U32 start_pos) {
    patch_u32(prof, sp.qk_rope_offset, 9U, start_pos);
    for (const U64 off : sp.attn_param_offsets) {
      patch_u32(prof, off, 4U, start_pos);
    }
  }

  static void patch_step_tokens_and_pos(Profile &prof,
                                        const StepPatchOffsets &sp, U32 rows,
                                        U32 start_pos) {
    patch_u32(prof, sp.qk_rope_offset, 4U, rows);
    patch_u32(prof, sp.qk_rope_offset, 9U, start_pos);
    for (const U64 off : sp.attn_param_offsets) {
      patch_u32(prof, off, 3U, rows);
      patch_u32(prof, off, 4U, start_pos);
    }
  }

  void rewind_speculative_tail() {
    if (mtp_kv_pool->tokens() > hist_offset) {
      mtp_kv_pool->truncate(hist_offset);
    }
  }

  void reserve_and_commit_kv(U32 append_tokens) {
    if (append_tokens == 0U) {
      return;
    }
    const auto copies = mtp_kv_pool->begin_append(append_tokens);
    if (!copies.empty()) {
      throw std::logic_error("MTP private KV pool triggered unexpected COW");
    }
    mtp_kv_pool->commit(append_tokens);
  }

  void reset() {
    mtp_kv_pool->reset();
    for (auto *buf : clear_on_reset_buffers) {
      buf->write(std::vector<std::byte>(buf->size()));
    }
    hist_offset = 0U;
    pending_trunk_valid = false;
    unprimed_broken = false;
  }

  struct PackedBatchKey {
    std::vector<const Impl *> slots;
    U32 max_k = 0;
    bool operator==(const PackedBatchKey &) const = default;
  };

  struct PackedBatchProfile {
    PackedBatchKey key;
    Profile profile;
    std::size_t d0_patch_count = 0;
    std::vector<U64> d0_stem_offsets;
    std::vector<U64> d0_layout_offsets;
    std::vector<StepPatchOffsets> d0_slot_attn;
    std::array<std::vector<StepPatchOffsets>, kMaxDraftTokens> chain_slot_attn;
    std::array<std::vector<U64>, kMaxDraftTokens> step_sample_offsets;
  };

  Ref pb_get(std::string_view name) const { return pb_refs.at(std::string(name)); }

  void ensure_packed_workspace() {
    if (!pb_refs.empty()) {
      return;
    }
    const auto add_pb = [&](std::string name, U64 bytes) {
      const U64 aligned = std::max<U64>(16ULL, bytes);
      auto buf = std::make_unique<MetalBuffer>(context.make_buffer(aligned));
      buf->zero();
      pb_refs.emplace(std::move(name), Ref{buf.get(), 0, aligned});
      pb_buffers.push_back(std::move(buf));
    };
    const U64 max_fold = 40ULL; // 8 slots * 5 fold tokens
    add_pb("pb_slot_query", mul(mul(max_fold, 2U * q), 2U));
    add_pb("pb_slot_key", mul(mul(max_fold, kv), 2U));
    add_pb("pb_slot_value", mul(mul(max_fold, kv), 2U));
    add_pb("pb_slot_mixer", mul(mul(max_fold, q), 2U));
    add_pb("pb_chain_tokens", 8ULL * 4ULL);
    add_pb("pb_chain_hidden", mul(mul(8ULL, h), 2U));
    add_pb("pb_last_norm", mul(mul(8ULL, h), 2U));
    add_pb("pb_logits", mul(mul(8ULL, v), 2U));
    add_pb("pb_argmax_parts", 8ULL * 64ULL * 16ULL);
    add_pb("pb_sampled", U64(kMaxDraftTokens) * 8ULL * 8ULL);
    add_pb("pb_compact_ids", U64(kMaxDraftTokens) * 8ULL * 64ULL * 4ULL);
    add_pb("pb_compact_scores", U64(kMaxDraftTokens) * 8ULL * 64ULL * 4ULL);
    add_pb("pb_proposal_out", 8ULL * 8ULL * 4ULL);
    add_pb("pb_attn_pos", U64(kMaxDraftTokens) * 8ULL * 64ULL);
  }

  StepPatchOffsets emit_slot_attention_only(Encoder &e, const Impl &sp,
                                            Ref slot_q, Ref slot_k, Ref slot_v,
                                            Ref slot_mixer, Ref slot_attn_pos,
                                            U32 max_rows, U32 unique_tag) {
    return emit_mtp_attention_core(e, sp, slot_q, slot_k, slot_v, slot_mixer,
                                   slot_attn_pos, max_rows, false, unique_tag);
  }

  PackedBatchProfile compile_packed_batch(const PackedBatchKey &skey) {
    const U32 B = U32(skey.slots.size());
    const U32 max_k = skey.max_k;
    const U32 fold_cap = kMaxFoldTokens * B;
    Fnv1a64 kh;
    kh.u64(B);
    kh.u64(max_k);
    for (const auto *sp : skey.slots) {
      kh.u64(sp->owner_id);
    }

    PackedBatchProfile out;
    out.key = skey;
    Encoder e(context, libraries, fold_cap,
              "mtp.packed.b" + std::to_string(B) + ".k" +
                  std::to_string(max_k) + "." + kh.hex() + plan_key_suffix());

    const MtpSlotLayoutParamsHost layout_init{
        .batch_size = B,
        .q_cols = 2U * q,
        .kv_cols = kv,
        .mixer_cols = q,
        .hidden = h};

    // ---------------------------------------------------------------------
    // Step d = 0: Packed dynamic fold (M0 = sum(c_b) <= 5*B) + [B, V] head
    // ---------------------------------------------------------------------
    e.profile.capacity = fold_cap;
    e.begin_stage("packed.d0.stem");
    emit_embedding_only(e, get("fold_tokens"), fold_cap, true);

    const bool d0_dual_fc = use_fc_dual_bf16(fold_cap);
    out.d0_stem_offsets.resize(B);
    for (U32 b = 0U; b < B; ++b) {
      const auto &sp = *skey.slots[b];
      const MtpStemNormsOffsetParamsHost stem_p{
          .hidden = h,
          .rows = 1U,
          .cond_mode = 0U,
          .write_concat = d0_dual_fc ? 0U : 1U,
          .trunk_row_base = 0U,
          .io_row_base = 0xFF000000U | b,
          .epsilon = float(config.rms_norm_eps),
          .target_norm_offset = target_norm_offset,
          .emb_norm_offset = norm_offsets.pre_fc_norm_embedding,
          .hid_norm_offset = norm_offsets.pre_fc_norm_hidden,
      };
      auto &sn = e.emit(
          "mtp_forward", "omlx_m5_mtp_stem_norms_offset",
          {.width = 128U, .height = kMaxFoldTokens, .depth = 2U},
          {.width = 128U, .height = 1U, .depth = 1U},
          {get("embed_out"), sp.local_trunk_ref,
           sp.get("pending_trunk_normed"), pb_get("pb_chain_hidden"),
           direct(target_final_norm, h), direct(pre_fc_norm_embedding, h),
           direct(pre_fc_norm_hidden, h), get("stem_e"), get("stem_h"),
           get("stem_concat")});
      out.d0_stem_offsets[b] = e.dynamic(sn, 10, struct_bytes(stem_p));
    }
    emit_fc_only(e, fold_cap, true);

    e.begin_stage("packed.d0.layer0");
    e.clear_dynamic_parameter_cache();
    emit_norm(e, input_norm, norm_offsets.input_layernorm, get("hidden"),
              get("norm"), fold_cap, true);
    if (try_concat_qkv_nax(e, std::nullopt)) {
      emit_split3(e, get("qkv_concat"), get("query"), get("key"), get("value"),
                  2U * q, kv, kv, std::nullopt);
    } else {
      emit_linear(e, q_proj, get("norm"), get("query"), 2U * q, h, std::nullopt);
      emit_linear(e, k_proj, get("norm"), get("key"), kv, h, std::nullopt);
      emit_linear(e, v_proj, get("norm"), get("value"), kv, h, std::nullopt);
    }

    {
      const U32 max_cols = std::max(2U * q, kv);
      const U64 grid_x = U64((max_cols + 127U) / 128U) * 128U;
      auto &sc = e.emit("mtp_forward", "omlx_m5_mtp_scatter_qkv_slots",
                        {.width = grid_x, .height = U64(fold_cap)},
                        {.width = 128U},
                        {get("query"), get("key"), get("value"),
                         pb_get("pb_slot_query"), pb_get("pb_slot_key"),
                         pb_get("pb_slot_value")});
      mtp_push_unique(out.d0_layout_offsets,
                      e.dynamic(sc, 6, struct_bytes(layout_init)));
    }
    out.d0_slot_attn.resize(B);
    for (U32 b = 0U; b < B; ++b) {
      Ref slot_pos{pb_get("pb_attn_pos").buffer, U64(b) * 64ULL, 64ULL};
      out.d0_slot_attn[b] = emit_slot_attention_only(
          e, *skey.slots[b],
          mtp_row_slice_ref(pb_get("pb_slot_query"), 5U * b, 5U, 2U * q),
          mtp_row_slice_ref(pb_get("pb_slot_key"), 5U * b, 5U, kv),
          mtp_row_slice_ref(pb_get("pb_slot_value"), 5U * b, 5U, kv),
          mtp_row_slice_ref(pb_get("pb_slot_mixer"), 5U * b, 5U, q), slot_pos,
          kMaxFoldTokens, 0xFF000000U | b);
    }
    {
      const U64 grid_x = U64((q + 127U) / 128U) * 128U;
      auto &ga = e.emit("mtp_forward", "omlx_m5_mtp_gather_mixer_slots",
                        {.width = grid_x, .height = U64(fold_cap)},
                        {.width = 128U},
                        {pb_get("pb_slot_mixer"), get("mixer")});
      mtp_push_unique(out.d0_layout_offsets,
                      e.dynamic(ga, 2, struct_bytes(layout_init)));
    }

    emit_linear(e, o_proj, get("mixer"), get("delta"), h, q, std::nullopt);
    emit_add_norm(e, post_attn_norm, norm_offsets.post_attention_layernorm,
                  get("hidden"), get("delta"), get("norm"), fold_cap, true);
    if (!moe_layer) {
      if (!try_linear_swiglu(e, mlp_gate, mlp_up, get("norm"), get("product"),
                             f, h, std::nullopt)) {
        emit_linear(e, mlp_gate, get("norm"), get("gate"), f, h, std::nullopt);
        emit_linear(e, mlp_up, get("norm"), get("up"), f, h, std::nullopt);
        auto &sw = e.emit("mlp", "omlx_m5_swiglu_a16",
                          {.width = 32U, .height = U64(fold_cap) * (f / 32U)},
                          {.width = 32U},
                          {get("gate"), get("up"), get("product")});
        e.dynamic(sw, 3, words({fold_cap, f, f / 32U, 0U}), 0U, {}, 1U);
      }
      emit_linear(e, mlp_down, get("product"), get("delta"), h, f, std::nullopt);
    } else {
      emit_moe_mlp(e, fold_cap, true, std::nullopt);
    }
    emit_add_norm(e, mtp_final_norm, norm_offsets.final_norm, get("hidden"),
                  get("delta"), get("norm"), fold_cap, true);
    out.d0_patch_count = e.profile.patches.size();

    auto emit_shared_head_sample_latch = [&](U32 d, Ref head_in) {
      e.begin_stage("packed.d" + std::to_string(d) + ".head_sample");
      e.profile.capacity = B;
      e.clear_dynamic_parameter_cache();
      emit_linear(e, head, head_in, pb_get("pb_logits"), v, h, B);

      Ref step_sampled{pb_get("pb_sampled").buffer, U64(d) * B * 8ULL,
                       U64(B) * 8ULL};
      Ref step_cids{pb_get("pb_compact_ids").buffer, U64(d) * B * 64ULL * 4ULL,
                    U64(B) * 64ULL * 4ULL};
      Ref step_cscores{pb_get("pb_compact_scores").buffer,
                       U64(d) * B * 64ULL * 4ULL, U64(B) * 64ULL * 4ULL};
      std::vector<PositionSamplingRow> srow_init(B);
      for (U32 b = 0U; b < B; ++b) {
        srow_init[b].seed = 0x10000ULL + U64(d) * 16ULL + U64(b);
      }
      const auto srow_raw = std::as_bytes(std::span<const PositionSamplingRow>(srow_init));
      const std::vector<std::byte> srow_bytes(srow_raw.begin(), srow_raw.end());

      auto &smp1 = e.emit("position_sampling",
                          "omlx_m5_position_sample_logits_stage1",
                          {.width = 64U * 256U, .height = B}, {.width = 256U},
                          {pb_get("pb_logits"), pb_get("pb_argmax_parts")});
      mtp_push_unique(out.step_sample_offsets[d], e.dynamic(smp1, 2, srow_bytes));
      e.constant(smp1, 3, words({v, B, B, 0U}));

      auto &smp2 = e.emit(
          "position_sampling", "omlx_m5_position_sample_logits_stage2",
          {.width = 256U, .height = B}, {.width = 256U},
          {pb_get("pb_logits"), pb_get("pb_argmax_parts"), step_sampled,
           step_cids, step_cscores});
      mtp_push_unique(out.step_sample_offsets[d], e.dynamic(smp2, 5, srow_bytes));
      e.constant(smp2, 6, words({v, B, B, 0U}));

      const bool write_chain = (d + 1U < max_k);
      const MtpLatchBatchParamsHost latch_p{
          .batch_size = B,
          .hidden = h,
          .step_index = d,
          .write_chain = write_chain ? 1U : 0U};
      const U64 latch_w = write_chain ? U64((h + 255U) / 256U) * 256ULL : 1ULL;
      auto &lt = e.emit("mtp_forward", "omlx_m5_mtp_latch_chain_batch",
                        {.width = latch_w, .height = B},
                        {.width = std::min<U64>(256ULL, latch_w), .height = 1U},
                        {step_sampled, head_in, pb_get("pb_proposal_out"),
                         pb_get("pb_chain_tokens"), pb_get("pb_chain_hidden")});
      e.dynamic(lt, 5, struct_bytes(latch_p));
    };

    e.profile.capacity = B;
    e.clear_dynamic_parameter_cache();
    {
      const U64 grid_x = U64((h + 127U) / 128U) * 128U;
      auto &ex = e.emit("mtp_forward", "omlx_m5_mtp_extract_last_rows",
                        {.width = grid_x, .height = B}, {.width = 128U},
                        {get("norm"), pb_get("pb_last_norm")});
      mtp_push_unique(out.d0_layout_offsets,
                      e.dynamic(ex, 2, struct_bytes(layout_init)));
    }
    emit_shared_head_sample_latch(0U, pb_get("pb_last_norm"));

    // ---------------------------------------------------------------------
    // Steps d = 1 .. max_k - 1: Fixed [B, H] chained drafts (NO scatter/gather!)
    // ---------------------------------------------------------------------
    for (U32 d = 1U; d < max_k; ++d) {
      e.profile.capacity = B;
      e.clear_dynamic_parameter_cache();
      e.begin_stage("packed.d" + std::to_string(d) + ".stem");
      emit_stem(e, pb_get("pb_chain_tokens"), local_trunk_ref, B, 2U, false,
                nullptr, nullptr, nullptr, nullptr, pb_get("pb_chain_hidden"));

      e.begin_stage("packed.d" + std::to_string(d) + ".layer0");
      e.clear_dynamic_parameter_cache();
      emit_norm(e, input_norm, norm_offsets.input_layernorm, get("hidden"),
                get("norm"), B, false);
      if (try_concat_qkv_nax(e, B)) {
        emit_split3(e, get("qkv_concat"), get("query"), get("key"), get("value"),
                    2U * q, kv, kv, B);
      } else {
        emit_linear(e, q_proj, get("norm"), get("query"), 2U * q, h, B);
        emit_linear(e, k_proj, get("norm"), get("key"), kv, h, B);
        emit_linear(e, v_proj, get("norm"), get("value"), kv, h, B);
      }

      out.chain_slot_attn[d].resize(B);
      for (U32 b = 0U; b < B; ++b) {
        Ref slot_pos{pb_get("pb_attn_pos").buffer,
                     U64(d * 8U + b) * 64ULL, 64ULL};
        out.chain_slot_attn[d][b] = emit_slot_attention_only(
            e, *skey.slots[b],
            mtp_row_slice_ref(get("query"), b, 1U, 2U * q),
            mtp_row_slice_ref(get("key"), b, 1U, kv),
            mtp_row_slice_ref(get("value"), b, 1U, kv),
            mtp_row_slice_ref(get("mixer"), b, 1U, q), slot_pos, 1U,
            0xFF000000U | (d << 8U) | b);
      }

      emit_linear(e, o_proj, get("mixer"), get("delta"), h, q, B);
      emit_add_norm(e, post_attn_norm, norm_offsets.post_attention_layernorm,
                    get("hidden"), get("delta"), get("norm"), B, false);
      if (!moe_layer) {
        if (!try_linear_swiglu(e, mlp_gate, mlp_up, get("norm"), get("product"),
                               f, h, B)) {
          emit_linear(e, mlp_gate, get("norm"), get("gate"), f, h, B);
          emit_linear(e, mlp_up, get("norm"), get("up"), f, h, B);
          auto &sw = e.emit("mlp", "omlx_m5_swiglu_a16",
                            {.width = 32U, .height = U64(B) * (f / 32U)},
                            {.width = 32U},
                            {get("gate"), get("up"), get("product")});
          e.dynamic(sw, 3, words({B, f, f / 32U, 0U}));
        }
        emit_linear(e, mlp_down, get("product"), get("delta"), h, f, B);
      } else {
        emit_moe_mlp(e, B, false, B);
      }
      emit_add_norm(e, mtp_final_norm, norm_offsets.final_norm, get("hidden"),
                    get("delta"), get("norm"), B, false);

      emit_shared_head_sample_latch(d, get("norm"));
    }

    e.profile.capacity = fold_cap;
    out.profile = e.finish(true);
    return out;
  }

  PackedBatchProfile &acquire_packed_batch(const PackedBatchKey &key) {
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
    if (packed_cache_.size() >= 8U) {
      packed_cache_.pop_back();
    }
    packed_cache_.push_front(compile_packed_batch(key));
    return packed_cache_.front();
  }

  static std::atomic<U64> next_owner_id_;
  U64 owner_id = next_owner_id_.fetch_add(1ULL, std::memory_order_relaxed);
  const MetalContext &context;
  const MaterializedModel &mtp_weights;
  QwenModelConfig config;
  U32 max_context = 0;
  MaterializedWeightView embedding, head, target_final_norm;
  Ref local_trunk_ref{}, delegated_trunk_ref{};
  float target_norm_offset = 0.0F;
  std::filesystem::path libraries;
  bool m4_short_routes = false;
  LinearNumericPolicy linear_policy = LinearNumericPolicy::Legacy;
  SamplingContract sampling_contract = SamplingContract::LegacyRejectionV1;
  KvStorageFormat kv_format = KvStorageFormat::BFloat16;
  U32 gpu_cores = 64U;

  std::string mtp_prefix;
  U32 h = 0, f = 0, shared_f = 0, v = 0, q = 0, kv = 0, priming_cap = 0;
  bool moe_layer = false;

  MaterializedWeightView pre_fc_norm_embedding, pre_fc_norm_hidden, fc_weight;
  MaterializedWeightView input_norm, q_proj, k_proj, v_proj, o_proj, q_norm,
      k_norm, post_attn_norm, mtp_final_norm;
  MaterializedWeightView mlp_gate, mlp_up, mlp_down;
  MaterializedWeightView moe_router, moe_gate, moe_up, moe_down,
      shared_gate_proj, shared_up_proj, shared_down_proj, shared_expert_gate;
  MtpNormOffsets norm_offsets{};

  std::vector<std::unique_ptr<MetalBuffer>> buffers;
  std::vector<MetalBuffer *> clear_on_reset_buffers;
  std::map<std::string, Ref> refs;
  std::vector<std::unique_ptr<MetalBuffer>> pb_buffers;
  std::map<std::string, Ref> pb_refs;
  std::deque<PackedBatchProfile> packed_cache_;
  std::unique_ptr<KvPagePool> mtp_kv_pool;

  bool affine4_nax_ok = false;
  bool flash_nax_prefill_ok = false;
  bool split_attn_ok = false;
  bool bf16_nax_enabled = false;
  bool kl32_enabled = false;
  MoeNaxWorkspace moe_nax_ws;
  std::shared_ptr<Mxfp4NaxCache> head_mxfp4_cache, mtp_mxfp4_cache;
  std::shared_ptr<AffineNaxCache> head_affine_cache, mtp_affine_cache;
  std::shared_ptr<MtpQ5High1Cache> head_q5_cache, mtp_q5_cache;
  Ref f16_scratch{}, f16_row_scale{}, affine_nax_xs{};

  SaveTrunkProfile save_local_trunk;
  std::optional<SaveTrunkProfile> save_delegated_trunk;
  std::array<std::vector<PrimingProfile>, 2> local_priming_profiles;
  std::array<std::vector<PrimingProfile>, 2> delegated_priming_profiles;
  std::array<std::array<FoldOnlyProfile, kMaxFoldTokens>, 2>
      fold_only_profiles{};
  std::array<std::array<std::array<ChainProfile, kMaxDraftTokens>,
                        kMaxFoldTokens>,
             2>
      chain_profiles{};

  U32 hist_offset = 0U;
  bool pending_trunk_valid = false;
  bool unprimed_broken = false;
};
std::atomic<U64> MtpForward::Impl::next_owner_id_{1ULL};

// ---------------------------------------------------------------------------
// MtpForward public interface
// ---------------------------------------------------------------------------

MtpForward::MtpForward(
    const MetalContext &context, const MaterializedModel &mtp_weights,
    QwenModelConfig config, std::uint32_t max_context,
    const MaterializedWeightView &embedding, const MaterializedWeightView &head,
    const MaterializedWeightView &target_final_norm,
    const MetalBuffer &target_trunk_hidden, float target_norm_offset,
    const std::filesystem::path &libraries, bool enable_m4_short_routes,
    LinearNumericPolicy linear_policy, SamplingContract sampling_contract,
    std::string_view kv_dtype, const QwenLayerCatalog *target_catalog,
    const MaterializedModel *target_weights,
    const MetalBuffer *shared_prefill_trunk)
    : impl_(std::make_unique<Impl>(
          context, mtp_weights, std::move(config), max_context, embedding, head,
          target_final_norm, target_trunk_hidden, target_norm_offset, libraries,
          enable_m4_short_routes, linear_policy, sampling_contract, kv_dtype,
          target_catalog, target_weights, shared_prefill_trunk)) {}

MtpForward::~MtpForward() = default;
MtpForward::MtpForward(MtpForward &&) noexcept = default;
MtpForward &MtpForward::operator=(MtpForward &&) noexcept = default;

void MtpForward::reset() { impl_->reset(); }

std::vector<std::byte> MtpForward::snapshot_pending_trunk() const {
  const auto &p = *impl_;
  if (!p.pending_trunk_valid || p.unprimed_broken) {
    return {};
  }
  const auto ref = p.get("pending_trunk_normed");
  return ref.buffer->read(ref.offset, mul(p.h, 2U));
}

void MtpForward::restore_frontier(
    std::uint32_t committed_hist_tokens,
    std::span<const std::byte> pending_trunk_bytes) {
  auto &p = *impl_;
  const U64 expected_bytes = mul(p.h, 2U);
  if (committed_hist_tokens > p.mtp_kv_pool->tokens() ||
      committed_hist_tokens > p.max_context ||
      pending_trunk_bytes.size() != expected_bytes) {
    throw std::invalid_argument("MtpForward::restore_frontier invalid snapshot");
  }
  p.mtp_kv_pool->truncate(committed_hist_tokens);
  const_cast<MetalBuffer *>(p.get("pending_trunk_normed").buffer)
      ->write(pending_trunk_bytes, 0U);
  p.hist_offset = committed_hist_tokens;
  p.pending_trunk_valid = true;
  p.unprimed_broken = false;
}

MetalSubmission MtpForward::seal_post_verify_frontier(
    std::span<const std::uint32_t> committed_round_tokens, bool use_icb) {
  auto &p = *impl_;
  if (committed_round_tokens.empty() ||
      committed_round_tokens.size() > kMaxFoldTokens) {
    throw std::invalid_argument(
        "MtpForward::seal_post_verify_frontier requires 1..5 tokens");
  }
  if (p.unprimed_broken) {
    return {};
  }
  const U32 chunk_len = U32(committed_round_tokens.size());
  const U32 local_max_rows = U32(p.local_trunk_ref.bytes / mul(p.h, 2U));
  if (chunk_len > local_max_rows) {
    throw std::out_of_range(
        "MtpForward::seal_post_verify_frontier exceeds local trunk capacity");
  }
  p.rewind_speculative_tail();

  const std::span<const U32> fold_toks = committed_round_tokens.subspan(1U);
  const U32 fold_total = U32(fold_toks.size());
  if (p.hist_offset + fold_total > p.max_context) {
    throw std::out_of_range(
        "MtpForward::seal_post_verify_frontier exceeds max_context");
  }

  MetalSubmission total_sub{};
  auto accumulate_sub = [&](const MetalSubmission &s) {
    total_sub.dispatch_count += s.dispatch_count;
    total_sub.timing.gpu_ms += s.timing.gpu_ms;
    total_sub.timing.wall_ms += s.timing.wall_ms;
  };

  if (fold_total > 0U) {
    const_cast<MetalBuffer *>(p.get("fold_tokens").buffer)
        ->write(std::as_bytes(fold_toks), 0U);
    std::array<U32, kMaxFoldTokens> pos_buf{};
    for (U32 r = 0U; r < fold_total; ++r) {
      pos_buf[r] = p.hist_offset + r;
    }
    const_cast<MetalBuffer *>(p.get("mtp_attn_pos").buffer)
        ->write(std::as_bytes(std::span<const U32>(pos_buf).first(fold_total)),
                0U);

    auto &fp = p.fold_only_profiles[0][fold_total - 1U];
    Impl::patch_step_position(fp.profile, fp.attn_patches, p.hist_offset);
    fp.profile.parameters.write("step", fp.profile.parameter_data);
    p.reserve_and_commit_kv(fold_total);
    if (use_icb && fp.profile.icb) {
      accumulate_sub(p.context.replay_icb(*fp.profile.icb));
    } else {
      const std::array calls{
          MetalCommandInvocation{.plan = &fp.profile.plan,
                                 .bindings = fp.profile.bindings,
                                 .parameters = &fp.profile.parameters}};
      accumulate_sub(p.context.execute_batch(calls));
    }
    p.hist_offset += fold_total;
  }

  auto &sv = p.save_local_trunk;
  Impl::patch_u32(sv.profile, sv.save_params_offset, 1U, chunk_len - 1U);
  sv.profile.parameters.write("step", sv.profile.parameter_data);
  if (use_icb && sv.profile.icb) {
    accumulate_sub(p.context.replay_icb(*sv.profile.icb));
  } else {
    const std::array calls{
        MetalCommandInvocation{.plan = &sv.profile.plan,
                               .bindings = sv.profile.bindings,
                               .parameters = &sv.profile.parameters}};
    accumulate_sub(p.context.execute_batch(calls));
  }
  p.pending_trunk_valid = true;
  return total_sub;
}

std::uint32_t MtpForward::committed_tokens() const noexcept {
  return impl_->hist_offset;
}

std::uint32_t MtpForward::kv_tokens() const noexcept {
  return impl_->mtp_kv_pool->tokens();
}

bool MtpForward::has_pending_trunk() const noexcept {
  return impl_->pending_trunk_valid;
}

bool MtpForward::ready_for_target_position(
    std::uint32_t target_pos) const noexcept {
  if (impl_->unprimed_broken || target_pos == 0U) {
    return false;
  }
  if (impl_->pending_trunk_valid) {
    return impl_->hist_offset + 1U == target_pos;
  }
  return impl_->hist_offset <= target_pos &&
         (target_pos - impl_->hist_offset) <= kMaxFoldTokens;
}

std::string_view MtpForward::prefix() const noexcept {
  return impl_->mtp_prefix;
}

bool MtpForward::is_moe_layer() const noexcept { return impl_->moe_layer; }

const MtpNormOffsets &MtpForward::norm_offsets() const noexcept {
  return impl_->norm_offsets;
}

MetalSubmission
MtpForward::prime_prefill_chunk(std::span<const std::uint32_t> chunk_tokens,
                                bool is_first_chunk, bool use_icb,
                                std::uint32_t trunk_row_offset,
                                bool force_delegated) {
  auto &p = *impl_;
  if (chunk_tokens.empty()) {
    throw std::invalid_argument("MtpForward::prime_prefill_chunk empty chunk");
  }
  if (is_first_chunk) {
    p.reset();
  } else if (!p.pending_trunk_valid || p.unprimed_broken) {
    p.unprimed_broken = true;
    return {};
  }

  const U32 chunk_len = U32(chunk_tokens.size());
  const U32 local_max_rows =
      U32(p.local_trunk_ref.bytes / mul(p.h, 2U));
  const U32 delegated_max_rows =
      p.delegated_trunk_ref.buffer
          ? U32(p.delegated_trunk_ref.bytes / mul(p.h, 2U))
          : 0U;
  const bool use_delegated =
      (force_delegated || trunk_row_offset + chunk_len > local_max_rows) &&
      p.save_delegated_trunk.has_value();
  const U32 trunk_max_rows = use_delegated ? delegated_max_rows : local_max_rows;
  if (trunk_row_offset + chunk_len > trunk_max_rows) {
    throw std::out_of_range("MtpForward prefill chunk exceeds trunk capacity");
  }

  p.rewind_speculative_tail();

  const std::span<const U32> fold_toks =
      is_first_chunk ? chunk_tokens.subspan(1U) : chunk_tokens;
  const U32 fold_total = U32(fold_toks.size());
  if (p.hist_offset + fold_total > p.max_context) {
    throw std::out_of_range("MtpForward priming exceeds max_context");
  }

  MetalSubmission total_sub{};
  auto accumulate_sub = [&](const MetalSubmission &s) {
    total_sub.dispatch_count += s.dispatch_count;
    total_sub.timing.gpu_ms += s.timing.gpu_ms;
    total_sub.timing.wall_ms += s.timing.wall_ms;
  };

  U32 sub_off = 0U;
  while (sub_off < fold_total) {
    const U32 sub_len = std::min<U32>(p.priming_cap, fold_total - sub_off);
    const U32 cond_mode = (!is_first_chunk && sub_off == 0U) ? 1U : 0U;
    const U32 trunk_row_base =
        trunk_row_offset +
        (is_first_chunk ? sub_off : (sub_off == 0U ? 0U : (sub_off - 1U)));
    const auto sub_toks = fold_toks.subspan(sub_off, sub_len);

    const_cast<MetalBuffer *>(p.get("fold_tokens").buffer)
        ->write(std::as_bytes(sub_toks), 0U);
    {
      std::array<U32, 1024> pos_buf{};
      for (U32 r = 0U; r < sub_len; ++r) {
        pos_buf[r] = p.hist_offset + r;
      }
      const_cast<MetalBuffer *>(p.get("mtp_attn_pos").buffer)
          ->write(std::as_bytes(std::span<const U32>(pos_buf).first(sub_len)),
                  0U);
    }

    auto &table = use_delegated ? p.delegated_priming_profiles[cond_mode]
                                : p.local_priming_profiles[cond_mode];
    auto it = std::find_if(table.begin(), table.end(), [&](const auto &entry) {
      return entry.capacity >= sub_len;
    });
    if (it == table.end()) {
      throw std::out_of_range("MtpForward priming bucket unavailable");
    }

    auto &prof = it->profile;
    for (const auto &patch : prof.patches) {
      const U32 val =
          patch.position ? p.hist_offset : sub_len * patch.multiplier;
      Impl::patch_u32(prof, patch.offset, patch.word, val);
    }
    Impl::patch_u32(prof, it->embed_offset, 5U, sub_len);
    Impl::patch_u32(prof, it->stem_offset, 1U, sub_len);
    Impl::patch_u32(prof, it->stem_offset, 4U, trunk_row_base);
    if (it->has_fc_dual) {
      Impl::patch_u32(prof, it->fc_dual_offset, 2U, sub_len);
    }
    Impl::patch_step_position(prof, it->attn_patches, p.hist_offset);
    prof.parameters.write("step", prof.parameter_data);

    p.reserve_and_commit_kv(sub_len);
    if (use_icb && prof.icb) {
      accumulate_sub(p.context.replay_icb(*prof.icb));
    } else {
      const std::array calls{
          MetalCommandInvocation{.plan = &prof.plan,
                                 .bindings = prof.bindings,
                                 .parameters = &prof.parameters}};
      accumulate_sub(p.context.execute_batch(calls));
    }
    p.hist_offset += sub_len;
    sub_off += sub_len;
  }

  // Latch pending_trunk_normed = norm(trunk[trunk_row_offset + chunk_len - 1])
  // for the next chunk or Cycle 1.
  auto &sv = use_delegated ? *p.save_delegated_trunk : p.save_local_trunk;
  Impl::patch_u32(sv.profile, sv.save_params_offset, 1U,
                  trunk_row_offset + chunk_len - 1U);
  sv.profile.parameters.write("step", sv.profile.parameter_data);
  if (use_icb && sv.profile.icb) {
    accumulate_sub(p.context.replay_icb(*sv.profile.icb));
  } else {
    const std::array calls{
        MetalCommandInvocation{.plan = &sv.profile.plan,
                               .bindings = sv.profile.bindings,
                               .parameters = &sv.profile.parameters}};
    accumulate_sub(p.context.execute_batch(calls));
  }
  p.pending_trunk_valid = true;
  return total_sub;
}

std::optional<MtpPreparedPrime> MtpForward::prepare_prime_prefill_chunk(
    std::span<const std::uint32_t> chunk_tokens, bool is_first_chunk,
    bool use_icb, std::uint32_t trunk_row_offset, bool force_delegated) {
  auto &p = *impl_;
  if (!use_icb || chunk_tokens.empty()) {
    return std::nullopt;
  }
  if (!is_first_chunk && (!p.pending_trunk_valid || p.unprimed_broken)) {
    return std::nullopt;
  }

  const U32 chunk_len = U32(chunk_tokens.size());
  const U32 local_max_rows = U32(p.local_trunk_ref.bytes / mul(p.h, 2U));
  const U32 delegated_max_rows =
      p.delegated_trunk_ref.buffer
          ? U32(p.delegated_trunk_ref.bytes / mul(p.h, 2U))
          : 0U;
  const bool use_delegated =
      (force_delegated || trunk_row_offset + chunk_len > local_max_rows) &&
      p.save_delegated_trunk.has_value();
  const U32 trunk_max_rows = use_delegated ? delegated_max_rows : local_max_rows;
  if (trunk_row_offset + chunk_len > trunk_max_rows) {
    return std::nullopt;
  }

  const std::span<const U32> fold_toks =
      is_first_chunk ? chunk_tokens.subspan(1U) : chunk_tokens;
  const U32 fold_total = U32(fold_toks.size());
  if (fold_total > p.priming_cap) {
    return std::nullopt;
  }
  const U32 base_hist = is_first_chunk ? 0U : p.hist_offset;
  if (base_hist + fold_total > p.max_context) {
    return std::nullopt;
  }

  auto &sv = use_delegated ? *p.save_delegated_trunk : p.save_local_trunk;
  if (!sv.profile.icb) {
    return std::nullopt;
  }

  Impl::PrimingProfile *prime_entry = nullptr;
  if (fold_total > 0U) {
    const U32 cond_mode = is_first_chunk ? 0U : 1U;
    auto &table = use_delegated ? p.delegated_priming_profiles[cond_mode]
                                : p.local_priming_profiles[cond_mode];
    auto it = std::find_if(table.begin(), table.end(), [&](const auto &entry) {
      return entry.capacity >= fold_total;
    });
    if (it == table.end() || !it->profile.icb) {
      return std::nullopt;
    }
    prime_entry = &(*it);
  }

  if (is_first_chunk) {
    p.reset();
  }
  p.rewind_speculative_tail();

  MetalIcbProfile *prime_icb = nullptr;
  if (prime_entry) {
    const_cast<MetalBuffer *>(p.get("fold_tokens").buffer)
        ->write(std::as_bytes(fold_toks), 0U);
    {
      std::array<U32, 1024> pos_buf{};
      for (U32 r = 0U; r < fold_total; ++r) {
        pos_buf[r] = p.hist_offset + r;
      }
      const_cast<MetalBuffer *>(p.get("mtp_attn_pos").buffer)
          ->write(std::as_bytes(std::span<const U32>(pos_buf).first(fold_total)),
                  0U);
    }
    auto &prof = prime_entry->profile;
    for (const auto &patch : prof.patches) {
      const U32 val =
          patch.position ? p.hist_offset : fold_total * patch.multiplier;
      Impl::patch_u32(prof, patch.offset, patch.word, val);
    }
    Impl::patch_u32(prof, prime_entry->embed_offset, 5U, fold_total);
    Impl::patch_u32(prof, prime_entry->stem_offset, 1U, fold_total);
    Impl::patch_u32(prof, prime_entry->stem_offset, 4U, trunk_row_offset);
    if (prime_entry->has_fc_dual) {
      Impl::patch_u32(prof, prime_entry->fc_dual_offset, 2U, fold_total);
    }
    Impl::patch_step_position(prof, prime_entry->attn_patches, p.hist_offset);
    prof.parameters.write("step", prof.parameter_data);
    p.reserve_and_commit_kv(fold_total);
    p.hist_offset += fold_total;
    prime_icb = &(*prof.icb);
  }

  Impl::patch_u32(sv.profile, sv.save_params_offset, 1U,
                  trunk_row_offset + chunk_len - 1U);
  sv.profile.parameters.write("step", sv.profile.parameter_data);
  p.pending_trunk_valid = true;
  return MtpPreparedPrime{.prime_icb = prime_icb,
                          .save_icb = &(*sv.profile.icb),
                          .fold_len = fold_total};
}

MetalSubmission MtpForward::fold_committed_only(
    std::span<const std::uint32_t> committed_tokens, bool use_icb) {
  auto &p = *impl_;
  const U32 c = U32(committed_tokens.size());
  if (c == 0U || c > kMaxFoldTokens) {
    throw std::invalid_argument("MtpForward::fold_committed_only requires 1..5 tokens");
  }
  if (p.unprimed_broken) {
    return {};
  }
  p.rewind_speculative_tail();
  const U32 cond_mode = p.pending_trunk_valid ? 1U : 0U;
  if (cond_mode == 1U && c != 1U) {
    throw std::logic_error("MtpForward pending trunk fold expects 1 anchor token");
  }
  if (p.hist_offset + c > p.max_context) {
    throw std::out_of_range("MtpForward fold exceeds max_context");
  }
  const_cast<MetalBuffer *>(p.get("fold_tokens").buffer)
      ->write(std::as_bytes(committed_tokens), 0U);
  {
    std::array<U32, kMaxFoldTokens> pos_buf{};
    for (U32 r = 0U; r < c; ++r) {
      pos_buf[r] = p.hist_offset + r;
    }
    const_cast<MetalBuffer *>(p.get("mtp_attn_pos").buffer)
        ->write(std::as_bytes(std::span<const U32>(pos_buf).first(c)), 0U);
  }

  auto &fp = p.fold_only_profiles[cond_mode][c - 1U];
  Impl::patch_step_position(fp.profile, fp.attn_patches, p.hist_offset);
  fp.profile.parameters.write("step", fp.profile.parameter_data);

  p.reserve_and_commit_kv(c);

  MetalSubmission sub;
  if (use_icb && fp.profile.icb) {
    sub = p.context.replay_icb(*fp.profile.icb);
  } else {
    const std::array calls{
        MetalCommandInvocation{.plan = &fp.profile.plan,
                               .bindings = fp.profile.bindings,
                               .parameters = &fp.profile.parameters}};
    sub = p.context.execute_batch(calls);
  }
  p.hist_offset += c;
  p.pending_trunk_valid = false;
  return sub;
}

MtpPreparedProposal
MtpForward::prepare_propose(std::span<const std::uint32_t> committed_tokens,
                            std::uint32_t target_position,
                            std::uint32_t draft_tokens,
                            const GpuSamplingOptions &sampling, bool use_icb,
                            bool tokens_only) {
  auto &p = *impl_;
  const U32 c = U32(committed_tokens.size());
  if (c == 0U || c > kMaxFoldTokens || draft_tokens == 0U ||
      draft_tokens > kMaxDraftTokens) {
    throw std::invalid_argument(
        "MtpForward::prepare_propose requires c in 1..5 and draft_tokens in 1..4");
  }
  if (p.unprimed_broken) {
    throw std::logic_error("MtpForward state is unprimed");
  }
  if (sampling.temperature < 0.0F || !std::isfinite(sampling.temperature) ||
      !std::isfinite(sampling.top_p) || sampling.top_p < 0.0F ||
      sampling.top_p > 1.0F || sampling.top_k > 64U ||
      (sampling.temperature > 0.0F && sampling.top_k == 0U)) {
    throw std::invalid_argument("MtpForward sampling options invalid");
  }

  // O(1) speculative KV rollback of any rejected drafts from the previous cycle!
  p.rewind_speculative_tail();

  const U32 cond_mode = p.pending_trunk_valid ? 1U : 0U;
  if (cond_mode == 1U && c != 1U) {
    throw std::logic_error("MtpForward first cycle with pending trunk requires c=1");
  }
  if (p.hist_offset + c != target_position) {
    throw std::logic_error(
        "MtpForward history frontier misaligned with target position: hist=" +
        std::to_string(p.hist_offset) + " c=" + std::to_string(c) +
        " target=" + std::to_string(target_position));
  }
  const U32 total_kv_append = c + (draft_tokens - 1U);
  if (p.hist_offset + total_kv_append > p.max_context) {
    throw std::out_of_range("MtpForward draft chain exceeds max_context");
  }

  const_cast<MetalBuffer *>(p.get("fold_tokens").buffer)
      ->write(std::as_bytes(committed_tokens), 0U);
  {
    std::array<U32, kMaxDraftTokens * 16U> pos_buf{};
    for (U32 r = 0U; r < c; ++r) {
      pos_buf[r] = p.hist_offset + r;
    }
    for (U32 d = 1U; d < draft_tokens; ++d) {
      pos_buf[d * 16U] = p.hist_offset + c + (d - 1U);
    }
    const_cast<MetalBuffer *>(p.get("mtp_attn_pos").buffer)
        ->write(std::as_bytes(
                    std::span<const U32>(pos_buf).first(draft_tokens * 16U)),
                0U);
  }

  auto &cp = p.chain_profiles[cond_mode][c - 1U][draft_tokens - 1U];
  auto &prof = cp.profile;

  const U32 eff_top_k =
      (sampling.temperature > 0.0F)
          ? (sampling.contract == SamplingContract::LegacyRejectionV1
                 ? std::min<U32>(sampling.top_k, 32U)
                 : sampling.top_k)
          : 1U;

  for (U32 d = 0U; d < draft_tokens; ++d) {
    const U32 kv_pos =
        (d == 0U) ? p.hist_offset : (p.hist_offset + c + (d - 1U));
    Impl::patch_step_position(prof, cp.steps[d], kv_pos);

    const PositionSamplingRow srow{
        .seed = sampling.seed,
        .position = U64(target_position) + U64(d) + 1ULL,
        .purpose = (sampling.contract == SamplingContract::PositionCoupledV1)
                       ? target_sampling_purpose
                       : draft_sampling_purpose,
        .top_k = eff_top_k,
        .temperature = sampling.temperature,
        .top_p = sampling.top_p,
        .valid = 1U,
    };
    std::memcpy(prof.parameter_data.data() + cp.steps[d].sample_offset, &srow,
                sizeof(srow));
  }
  prof.parameters.write("step", prof.parameter_data);

  // Reserve all c + (K - 1) KV positions; hist_offset only advances by c so
  // next cycle's truncate(hist_offset) rewinds the K - 1 speculative entries in O(1)!
  p.reserve_and_commit_kv(total_kv_append);
  p.hist_offset += c;
  p.pending_trunk_valid = false;

  MtpPreparedProposal prep;
  prep.partial.width = draft_tokens;
  prep.partial.short_matrix_calls = prof.short_matrix_calls;
  prep.partial.short_lane_calls = prof.short_lane_calls;
  prep.partial.gpu_selector = true;
  prep.icb = (use_icb && prof.icb) ? &(*prof.icb) : nullptr;
  prep.execution_profile = &cp;
  prep.committed_fold_count = c;
  prep.draft_tokens = draft_tokens;
  prep.target_position = target_position;
  prep.sampling = sampling;
  prep.tokens_only = tokens_only;
  return prep;
}

DFlash2DraftResult MtpForward::finish_propose(
    MtpPreparedProposal &&prepared, bool already_executed,
    MetalSubmission batched_submission) {
  auto &p = *impl_;
  const auto *cp =
      static_cast<const Impl::ChainProfile *>(prepared.execution_profile);
  if (!cp) {
    throw std::logic_error("MtpForward::finish_propose missing profile");
  }
  DFlash2DraftResult result = std::move(prepared.partial);
  if (already_executed) {
    result.submission = std::move(batched_submission);
  } else if (prepared.icb) {
    result.submission = p.context.replay_icb(*prepared.icb);
  } else {
    const std::array calls{
        MetalCommandInvocation{.plan = &cp->profile.plan,
                               .bindings = cp->profile.bindings,
                               .parameters = &cp->profile.parameters}};
    result.submission = p.context.execute_batch(calls);
  }
  result.draft_dispatches = result.submission.dispatch_count;

  const U32 k = prepared.draft_tokens;
  const auto raw = p.get("proposal_out").buffer->read(0U, 8U * 4U);
  result.host_read_bytes += raw.size();
  std::array<U32, 8> out_words{};
  std::memcpy(out_words.data(), raw.data(), raw.size());
  for (U32 d = 0U; d < k; ++d) {
    if ((out_words[4U + d] & 2U) != 0U) {
      throw std::runtime_error("nonfinite MTP draft logits");
    }
    if (out_words[d] >= p.v) {
      throw std::runtime_error("MTP draft token out of vocabulary bounds");
    }
  }

  std::vector<U32> all_cids;
  std::vector<float> all_cscores;
  const U32 proposal_k =
      mtp_proposal_k(prepared.sampling, prepared.tokens_only, p.v);
  if (proposal_k > 0U) {
    const auto raw_ids = p.get("compact_ids").buffer->read(0U, U64(k) * 64U * 4U);
    const auto raw_sc =
        p.get("compact_scores").buffer->read(0U, U64(k) * 64U * 4U);
    result.host_read_bytes += raw_ids.size() + raw_sc.size();
    all_cids.resize(U64(k) * 64U);
    all_cscores.resize(U64(k) * 64U);
    std::memcpy(all_cids.data(), raw_ids.data(), raw_ids.size());
    std::memcpy(all_cscores.data(), raw_sc.data(), raw_sc.size());
  }

  result.selection.tokens.resize(k);
  result.selection.steps.resize(k);
  for (U32 d = 0U; d < k; ++d) {
    const auto tok = std::int32_t(out_words[d]);
    result.selection.tokens[d] = tok;
    SelectorStep step{.token = tok};
    if (!prepared.tokens_only) {
      if (proposal_k == 0U) {
        step.candidate_ids = {tok};
        step.probabilities = {1.0F};
      } else {
        const U32 base = d * 64U;
        populate_mtp_proposal_step(step, proposal_k, all_cids.data() + base,
                                   all_cscores.data() + base,
                                   prepared.sampling.temperature,
                                   prepared.sampling.top_p);
      }
    }
    result.selection.steps[d] = std::move(step);
  }
  return result;
}

DFlash2DraftResult MtpForward::propose(
    std::span<const std::uint32_t> committed_tokens,
    std::uint32_t target_position, std::uint32_t draft_tokens,
    const GpuSamplingOptions &sampling, bool use_icb, bool tokens_only) {
  auto prep = prepare_propose(committed_tokens, target_position, draft_tokens,
                              sampling, use_icb, tokens_only);
  return finish_propose(std::move(prep), false);
}

bool MtpForward::can_propose_packed_batch(
    std::uint32_t batch_size) const noexcept {
  if (!impl_) {
    return false;
  }
  return batch_size >= 2U && batch_size <= 8U &&
         (kMaxFoldTokens * batch_size) <= impl_->priming_cap;
}

MtpPackedBatchResult MtpForward::propose_packed_batch(
    std::span<const MtpPackedSlotSlice> slices, bool use_icb) {
  auto &p = *impl_;
  const U32 B = U32(slices.size());
  if (!can_propose_packed_batch(B)) {
    throw std::invalid_argument(
        "MtpForward::propose_packed_batch batch size unsupported");
  }

  std::vector<const Impl *> slot_ptrs;
  slot_ptrs.reserve(B);
  std::array<U32, 8> fold_counts{};
  std::array<U32, 8> fold_offsets{};
  std::array<U32, 8> cond_modes{};
  std::array<U32, 8> hist_before{};
  std::vector<U32> flat_tokens;
  flat_tokens.reserve(U64(B) * kMaxFoldTokens);
  U32 total_fold = 0U;
  U32 max_k = 0U;
  bool any_probs = false;

  for (U32 b = 0U; b < B; ++b) {
    const auto &sl = slices[b];
    if (!sl.slot_mtp || !sl.slot_mtp->impl_) {
      throw std::invalid_argument(
          "MtpForward::propose_packed_batch null slot_mtp");
    }
    const auto &sp = *sl.slot_mtp->impl_;
    if (sp.h != p.h || sp.q != p.q || sp.kv != p.kv || sp.v != p.v ||
        sp.moe_layer != p.moe_layer || sp.kv_format != p.kv_format) {
      throw std::invalid_argument(
          "MtpForward::propose_packed_batch incompatible slot geometry");
    }
    for (const auto *prev : slot_ptrs) {
      if (prev == &sp) {
        throw std::invalid_argument(
            "MtpForward::propose_packed_batch duplicate slot");
      }
    }
    const U32 c = U32(sl.committed_tokens.size());
    const U32 k = sl.draft_tokens;
    if (c == 0U || c > kMaxFoldTokens || k == 0U || k > kMaxDraftTokens) {
      throw std::invalid_argument(
          "MtpForward::propose_packed_batch requires c in 1..5 and k in 1..4");
    }
    if (sp.unprimed_broken) {
      throw std::logic_error("MtpForward packed slot state is unprimed");
    }
    if (sl.sampling.temperature < 0.0F || !std::isfinite(sl.sampling.temperature) ||
        !std::isfinite(sl.sampling.top_p) || sl.sampling.top_p < 0.0F ||
        sl.sampling.top_p > 1.0F || sl.sampling.top_k > 64U ||
        (sl.sampling.temperature > 0.0F && sl.sampling.top_k == 0U)) {
      throw std::invalid_argument(
          "MtpForward::propose_packed_batch sampling options invalid");
    }
    const U32 cond_mode = sp.pending_trunk_valid ? 1U : 0U;
    if (cond_mode == 1U && c != 1U) {
      throw std::logic_error(
          "MtpForward packed first cycle with pending trunk requires c=1");
    }
    if (sp.hist_offset + c != sl.target_position) {
      throw std::logic_error(
          "MtpForward packed history frontier misaligned with target position");
    }
    if (sp.hist_offset + c + (k - 1U) > sp.max_context) {
      throw std::out_of_range("MtpForward packed draft chain exceeds max_context");
    }
    slot_ptrs.push_back(&sp);
    fold_counts[b] = c;
    fold_offsets[b] = total_fold;
    cond_modes[b] = cond_mode;
    hist_before[b] = sp.hist_offset;
    flat_tokens.insert(flat_tokens.end(), sl.committed_tokens.begin(),
                       sl.committed_tokens.end());
    total_fold += c;
    max_k = std::max(max_k, k);
    if (!sl.tokens_only && sl.sampling.temperature > 0.0F) {
      any_probs = true;
    }
  }

  p.ensure_packed_workspace();
  auto &pb = p.acquire_packed_batch({std::move(slot_ptrs), max_k});
  auto &prof = pb.profile;

  const_cast<MetalBuffer *>(p.get("fold_tokens").buffer)
      ->write(std::as_bytes(std::span<const U32>(flat_tokens)), 0U);

  for (std::size_t pi = 0; pi < pb.d0_patch_count; ++pi) {
    const auto &patch = prof.patches[pi];
    if (!patch.position) {
      Impl::patch_u32(prof, patch.offset, patch.word,
                      total_fold * patch.multiplier);
    }
  }

  const bool d0_dual_fc = p.use_fc_dual_bf16(kMaxFoldTokens * B);
  for (U32 b = 0U; b < B; ++b) {
    const MtpStemNormsOffsetParamsHost stem_p{
        .hidden = p.h,
        .rows = fold_counts[b],
        .cond_mode = cond_modes[b],
        .write_concat = d0_dual_fc ? 0U : 1U,
        .trunk_row_base = 0U,
        .io_row_base = fold_offsets[b],
        .epsilon = float(p.config.rms_norm_eps),
        .target_norm_offset = p.target_norm_offset,
        .emb_norm_offset = p.norm_offsets.pre_fc_norm_embedding,
        .hid_norm_offset = p.norm_offsets.pre_fc_norm_hidden,
    };
    std::memcpy(prof.parameter_data.data() + pb.d0_stem_offsets[b], &stem_p,
                sizeof(stem_p));
  }

  MtpSlotLayoutParamsHost layout_p{
      .batch_size = B,
      .q_cols = 2U * p.q,
      .kv_cols = p.kv,
      .mixer_cols = p.q,
      .hidden = p.h,
  };
  layout_p.fold_counts = fold_counts;
  layout_p.fold_offsets = fold_offsets;
  for (const U64 off : pb.d0_layout_offsets) {
    std::memcpy(prof.parameter_data.data() + off, &layout_p, sizeof(layout_p));
  }

  {
    std::array<U32, kMaxDraftTokens * 8U * 16U> pb_pos{};
    for (U32 b = 0U; b < B; ++b) {
      for (U32 r = 0U; r < fold_counts[b]; ++r) {
        pb_pos[b * 16U + r] = hist_before[b] + r;
      }
      Impl::patch_step_tokens_and_pos(prof, pb.d0_slot_attn[b], fold_counts[b],
                                      hist_before[b]);
    }
    for (U32 d = 1U; d < max_k; ++d) {
      for (U32 b = 0U; b < B; ++b) {
        const bool active = (d < slices[b].draft_tokens);
        const U32 tok_d = active ? 1U : 0U;
        const U32 pos_d =
            active ? (hist_before[b] + fold_counts[b] + (d - 1U)) : 0U;
        pb_pos[(d * 8U + b) * 16U] = pos_d;
        Impl::patch_step_tokens_and_pos(prof, pb.chain_slot_attn[d][b], tok_d,
                                        pos_d);
      }
    }
    const_cast<MetalBuffer *>(p.pb_get("pb_attn_pos").buffer)
        ->write(
            std::as_bytes(std::span<const U32>(pb_pos).first(max_k * 8U * 16U)),
            0U);
  }

  std::vector<PositionSamplingRow> srows(B);
  for (U32 d = 0U; d < max_k; ++d) {
    for (U32 b = 0U; b < B; ++b) {
      const auto &sl = slices[b];
      const bool active = (d < sl.draft_tokens);
      const U32 eff_top_k =
          (sl.sampling.temperature > 0.0F)
              ? (sl.sampling.contract == SamplingContract::LegacyRejectionV1
                     ? std::min<U32>(sl.sampling.top_k, 32U)
                     : sl.sampling.top_k)
              : 1U;
      srows[b] = PositionSamplingRow{
          .seed = sl.sampling.seed,
          .position = U64(sl.target_position) + U64(d) + 1ULL,
          .purpose =
              (sl.sampling.contract == SamplingContract::PositionCoupledV1)
                  ? target_sampling_purpose
                  : draft_sampling_purpose,
          .top_k = eff_top_k,
          .temperature = sl.sampling.temperature,
          .top_p = sl.sampling.top_p,
          .valid = active ? 1U : 0U,
      };
    }
    for (const U64 off : pb.step_sample_offsets[d]) {
      std::memcpy(prof.parameter_data.data() + off, srows.data(),
                  B * sizeof(PositionSamplingRow));
    }
  }
  prof.parameters.write("step", prof.parameter_data);

  for (U32 b = 0U; b < B; ++b) {
    auto &sp = *slices[b].slot_mtp->impl_;
    sp.rewind_speculative_tail();
    sp.reserve_and_commit_kv(fold_counts[b] + (slices[b].draft_tokens - 1U));
    sp.hist_offset += fold_counts[b];
    sp.pending_trunk_valid = false;
  }

  MtpPackedBatchResult out;
  if (use_icb && prof.icb) {
    out.submission = p.context.replay_icb(*prof.icb);
  } else {
    const std::array calls{
        MetalCommandInvocation{.plan = &prof.plan,
                               .bindings = prof.bindings,
                               .parameters = &prof.parameters}};
    out.submission = p.context.execute_batch(calls);
  }

  const auto raw_prop =
      p.pb_get("pb_proposal_out").buffer->read(0U, U64(B) * 8U * 4U);
  std::vector<U32> prop_words(U64(B) * 8U);
  std::memcpy(prop_words.data(), raw_prop.data(), raw_prop.size());

  std::vector<U32> all_cids;
  std::vector<float> all_cscores;
  std::size_t compact_read_bytes = 0U;
  if (any_probs) {
    const U64 n_entries = U64(max_k) * B * 64ULL;
    const auto raw_ids =
        p.pb_get("pb_compact_ids").buffer->read(0U, n_entries * 4ULL);
    const auto raw_sc =
        p.pb_get("pb_compact_scores").buffer->read(0U, n_entries * 4ULL);
    compact_read_bytes = raw_ids.size() + raw_sc.size();
    all_cids.resize(n_entries);
    all_cscores.resize(n_entries);
    std::memcpy(all_cids.data(), raw_ids.data(), raw_ids.size());
    std::memcpy(all_cscores.data(), raw_sc.data(), raw_sc.size());
  }

  const std::size_t total_read_bytes = raw_prop.size() + compact_read_bytes;
  out.slot_results.resize(B);
  for (U32 b = 0U; b < B; ++b) {
    const auto &sl = slices[b];
    const U32 k = sl.draft_tokens;
    auto &res = out.slot_results[b];
    res.width = k;
    res.short_matrix_calls = prof.short_matrix_calls;
    res.short_lane_calls = prof.short_lane_calls;
    res.gpu_selector = true;
    res.submission = out.submission;
    res.submission.timing.gpu_ms /= double(B);
    res.submission.timing.wall_ms /= double(B);
    res.submission.dispatch_count /= B;
    res.draft_dispatches = res.submission.dispatch_count;
    res.host_read_bytes = (b == 0U) ? total_read_bytes : 0U;

    const U32 prop_base = b * 8U;
    for (U32 d = 0U; d < k; ++d) {
      if ((prop_words[prop_base + 4U + d] & 2U) != 0U) {
        throw std::runtime_error("nonfinite packed MTP draft logits");
      }
      if (prop_words[prop_base + d] >= p.v) {
        throw std::runtime_error(
            "packed MTP draft token out of vocabulary bounds");
      }
    }

    const U32 proposal_k = mtp_proposal_k(sl.sampling, sl.tokens_only, p.v);

    res.selection.tokens.resize(k);
    res.selection.steps.resize(k);
    for (U32 d = 0U; d < k; ++d) {
      const auto tok = std::int32_t(prop_words[prop_base + d]);
      res.selection.tokens[d] = tok;
      SelectorStep step{.token = tok};
      if (!sl.tokens_only) {
        if (proposal_k == 0U) {
          step.candidate_ids = {tok};
          step.probabilities = {1.0F};
        } else {
          const U64 base = (U64(d) * B + U64(b)) * 64ULL;
          populate_mtp_proposal_step(step, proposal_k, all_cids.data() + base,
                                     all_cscores.data() + base,
                                     sl.sampling.temperature, sl.sampling.top_p);
        }
      }
      res.selection.steps[d] = std::move(step);
    }
  }
  return out;
}

QwenProfileDescription MtpForward::inspect_profile(std::uint32_t fold_tokens,
                                                   std::uint32_t draft_tokens) const {
  const auto &p = *impl_;
  if (fold_tokens == 0U || fold_tokens > kMaxFoldTokens || draft_tokens == 0U ||
      draft_tokens > kMaxDraftTokens) {
    throw std::invalid_argument("MtpForward::inspect_profile out of range");
  }
  const auto &prof =
      p.chain_profiles[0][fold_tokens - 1U][draft_tokens - 1U].profile;
  QwenProfileDescription desc;
  desc.capacity = fold_tokens;
  desc.verify = false;
  desc.dispatch_count = prof.plan.dispatch_count();
  desc.barrier_count = prof.plan.barrier_count();
  desc.numeric_contract_id = prof.numeric_contract_id;
  desc.sampling_contract_id =
      std::string(sampling_contract_name(p.sampling_contract));
  for (const auto &st : prof.stages) {
    desc.stages.push_back({st.name, st.length});
  }
  for (const auto &wr : prof.weight_routes) {
    desc.weights.push_back({wr.stage, wr.object, wr.format, wr.library,
                            wr.function, wr.family, wr.source,
                            wr.fallback_reason, wr.n, wr.k, wr.t, wr.grid,
                            wr.threads, wr.numeric_contract_id,
                            wr.row_exact_group, wr.row_exact_eligible});
  }
  return desc;
}

} // namespace omlx::m5
