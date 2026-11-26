#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/dflash2.h"
#include "omlx_m5/qwen_forward.h"

namespace omlx::m5 {

[[nodiscard]] bool has_mtp_weights(const ArtifactModel &artifact) noexcept;
[[nodiscard]] bool has_mtp_weights(const MaterializedModel &model) noexcept;

struct MtpNormOffsets {
  float pre_fc_norm_embedding = 0.0F;
  float pre_fc_norm_hidden = 0.0F;
  float input_layernorm = 0.0F;
  float q_norm = 0.0F;
  float k_norm = 0.0F;
  float post_attention_layernorm = 0.0F;
  float final_norm = 0.0F;
  bool repaired_any = false;
};

// Calibrates per-norm RMSNorm weight_offsets (+1.0f vs +0.0f) for MTP Layer 0.
// When target_norm_offset == 1.0f (HF zero-centered checkpoint), all MTP norms
// use +1.0f. When target_norm_offset == 0.0f (MLX/oQ shifted checkpoint),
// detects legacy unshifted MTP norms whose mean sits > 0.4 below their
// corresponding target backbone anchor norm mean and repairs them with +1.0f.
[[nodiscard]] MtpNormOffsets repair_legacy_head_norms(
    const MaterializedModel &mtp_weights, std::string_view mtp_prefix,
    const QwenModelConfig &config, float target_norm_offset,
    const MaterializedWeightView *anchor_final_norm = nullptr,
    const MaterializedWeightView *anchor_input_norm = nullptr,
    const MaterializedWeightView *anchor_post_attn_norm = nullptr,
    const MaterializedWeightView *anchor_q_norm = nullptr,
    const MaterializedWeightView *anchor_k_norm = nullptr);

// Pure host-side adaptive MTP depth controller (K in 0..4), ported from
// omlx/patches/mlx_lm_mtp/batch_generator.py::_DepthController with zero extra
// GPU syncs. Supports fixed_depth mode for deterministic tests / spec_search=off.
class MtpDepthController final {
public:
  static constexpr std::uint32_t kMaxSupportedDepth = 4;
  static constexpr std::uint32_t kDefaultMaxDepth = 4;
  static constexpr std::uint32_t kMaxBatchSize = 8;
  static constexpr double kAlpha = 0.08;
  static constexpr double kTauMs = 400.0;
  static constexpr double kProbePeriodMs = 1000.0;
  static constexpr double kProbePeriodMaxMs = 5000.0;
  static constexpr std::uint32_t kProbeLen = 4;
  static constexpr double kProbeDuty = 0.15;
  static constexpr double kProbeMargin = 1.15;
  static constexpr double kSpikeRatio = 2.0;
  static constexpr double kSpikeDamp = 0.25;
  static constexpr double kDefaultMarginalMs = 7.0;
  static constexpr double kHysteresis = 1.03;
  static constexpr double kDefaultExitMargin = 1.15;
  static constexpr std::uint32_t kExitStreak = 16;
  static constexpr std::uint32_t kExitConfirm = 16;

  explicit MtpDepthController(std::uint32_t max_depth = kDefaultMaxDepth,
                              bool fixed_depth = false,
                              double marginal_ms = kDefaultMarginalMs,
                              double exit_margin = kDefaultExitMargin);

  void reset();
  void on_new_request(std::uint32_t context_tokens = 0) noexcept;
  [[nodiscard]] std::uint32_t
  select_depth(std::uint32_t max_allowed_k,
               std::uint32_t batch_size = 1U) const noexcept;
  [[nodiscard]] bool is_probing(std::uint32_t batch_size = 1U) const noexcept;
  void observe(std::uint32_t used, std::uint32_t accepted, double cycle_ms,
               bool time_sample = true,
               std::uint32_t batch_size = 1U) noexcept;
  [[nodiscard]] bool should_exit() const noexcept;
  [[nodiscard]] bool fixed_depth() const noexcept { return fixed_depth_; }
  [[nodiscard]] std::uint32_t current_depth(std::uint32_t batch_size = 1U) const noexcept;
  [[nodiscard]] std::uint64_t cycles() const noexcept { return cycles_; }
  [[nodiscard]] double score(std::uint32_t depth,
                             std::uint32_t batch_size = 1U) const noexcept;

private:
  struct BatchTimingState {
    std::uint32_t cur = 4;
    std::array<std::optional<double>, kMaxSupportedDepth + 1> t{};
    std::array<double, kMaxSupportedDepth + 1> t_age{};
    std::uint32_t probe_left = 0;
    double ms_probe = 0.0;
    double ms_explore = 0.0;
    std::vector<std::uint32_t> warmup;
  };

  void reset_batch_state(BatchTimingState &bs);
  [[nodiscard]] double p_eff(std::uint32_t j) const noexcept;
  [[nodiscard]] double marginal_est(const BatchTimingState &bs) const noexcept;
  [[nodiscard]] double t_est(const BatchTimingState &bs,
                             std::uint32_t d) const noexcept;
  [[nodiscard]] double score_for(const BatchTimingState &bs,
                                 std::uint32_t d) const noexcept;
  [[nodiscard]] bool speculation_losing(const BatchTimingState &bs) const noexcept;
  [[nodiscard]] std::uint32_t best_depth(const BatchTimingState &bs) const noexcept;
  [[nodiscard]] std::optional<std::uint32_t>
  best_rival(const BatchTimingState &bs) const noexcept;
  [[nodiscard]] std::optional<std::uint32_t>
  most_stale(const BatchTimingState &bs) const noexcept;
  void update_time(BatchTimingState &bs, std::uint32_t used,
                   double cycle_ms) noexcept;

  std::uint32_t max_depth_ = 4;
  bool fixed_depth_ = false;
  double marginal_ms_ = kDefaultMarginalMs;
  double exit_margin_ = kDefaultExitMargin;

  std::array<std::optional<double>, kMaxSupportedDepth> p_{};
  std::array<BatchTimingState, kMaxBatchSize> batch_{};
  std::uint32_t active_batch_ = 1;
  std::uint32_t last_context_tokens_ = 0;
  std::uint64_t cycles_ = 0;
  std::uint32_t exit_streak_ = 0;
  std::uint32_t streak_spec_ = 0;
};

[[nodiscard]] QwenForwardMemoryPlan
plan_mtp_forward_memory(const QwenModelConfig &config,
                        std::uint32_t max_context, bool affine4_kv = false);

class MtpForward;

// Holds immutable committed MTP KV pages in the model's shared MTP arena.
struct MtpPrefixCheckpoint {
  std::uint32_t tokens = 0;
  std::uint64_t kv_page_bytes = 0;
  std::vector<std::uint32_t> kv_pages;
  std::vector<std::byte> pending_trunk;

  [[nodiscard]] const void *kv_allocator_id() const noexcept;
  [[nodiscard]] std::uint64_t non_kv_bytes() const noexcept;

private:
  struct Lease;
  std::shared_ptr<const Lease> lease_;
  friend class MtpForward;
};

struct MtpPreparedPrime {
  MetalIcbProfile *prime_icb = nullptr;
  MetalIcbProfile *save_icb = nullptr;
  std::uint32_t fold_len = 0;
};

struct MtpPackedSlotSlice {
  MtpForward *slot_mtp = nullptr;
  std::span<const std::uint32_t> committed_tokens{};
  std::uint32_t target_position = 0;
  std::uint32_t draft_tokens = 0;
  GpuSamplingOptions sampling{};
  bool tokens_only = false;
};

struct MtpPackedBatchResult {
  MetalSubmission submission;
  std::vector<DFlash2DraftResult> slot_results;
};

struct MtpPreparedProposal {
  DFlash2DraftResult partial;
  MetalIcbProfile *icb = nullptr;
  const void *execution_profile = nullptr;
  std::uint32_t committed_fold_count = 0;
  std::uint32_t draft_tokens = 0;
  std::uint32_t target_position = 0;
  GpuSamplingOptions sampling{};
  bool tokens_only = false;
};

// Native checkpoint-embedded Qwen3.5/3.6/3.8 Lightning MTP head executor.
// Shares target embedding_weight(), head_weight(), final_norm_weight(), and
// trunk_hidden_buffer() with 0 B vocabulary duplication, and executes both the
// committed history fold (c in 1..5) and chained draft steps (d = 1..K-1)
// inside ONE single Compute ICB submission.
class MtpForward final {
public:
  static constexpr std::uint32_t kDefaultMaxDraftTokens = 4;
  static constexpr std::uint32_t kMaxDraftTokens = 4;
  static constexpr std::uint32_t kMaxFoldTokens = 5;

  MtpForward(const MetalContext &context, const MaterializedModel &mtp_weights,
             QwenModelConfig config, std::uint32_t max_context,
             const MaterializedWeightView &embedding,
             const MaterializedWeightView &head,
             const MaterializedWeightView &target_final_norm,
             const MetalBuffer &target_trunk_hidden,
             float target_norm_offset,
             const std::filesystem::path &libraries,
             bool enable_m4_short_routes = false,
             LinearNumericPolicy linear_policy = LinearNumericPolicy::Legacy,
             SamplingContract sampling_contract =
                 SamplingContract::LegacyRejectionV1,
             std::string_view kv_dtype = "bf16",
             const QwenLayerCatalog *target_catalog = nullptr,
             const MaterializedModel *target_weights = nullptr,
             const MetalBuffer *shared_prefill_trunk = nullptr,
             const MtpForward *shared_kv_primary = nullptr,
             std::uint32_t shared_kv_pages = 0);
  ~MtpForward();
  MtpForward(MtpForward &&) noexcept;
  MtpForward &operator=(MtpForward &&) noexcept;
  MtpForward(const MtpForward &) = delete;
  MtpForward &operator=(const MtpForward &) = delete;

  // Folds prompt history (trunk_normed[t], chunk_tokens[t+1]) into mtp_kv_pool
  // immediately after QwenForward::run(chunk_tokens) and latches the final
  // row's trunk_normed into pending_trunk_normed for the next chunk or Cycle 1.
  MetalSubmission prime_prefill_chunk(std::span<const std::uint32_t> chunk_tokens,
                                      bool is_first_chunk,
                                      bool use_icb = true,
                                      std::uint32_t trunk_row_offset = 0U,
                                      bool force_delegated = false);

  // Stages single-chunk priming ICBs without submitting so a multi-slot batch
  // can replay all B slots' MTP priming + save_trunk in ONE CB. Returns
  // std::nullopt if the chunk requires multi-subchunk priming or non-ICB.
  [[nodiscard]] std::optional<MtpPreparedPrime>
  prepare_prime_prefill_chunk(std::span<const std::uint32_t> chunk_tokens,
                              bool is_first_chunk, bool use_icb = true,
                              std::uint32_t trunk_row_offset = 0U,
                              bool force_delegated = false);

  // Synchronizes MTP state when a prefix-cache restore jumps target position
  // without running prefill chunks. Rewinds/clears stale state cleanly.
  void reset();

  // Snapshots / restores the 1-row normalized target trunk state (`h * 2` B)
  // and truncates `mtp_kv_pool` to `committed_hist_tokens` so same-slot
  // Rolling Prefix Cache hits (including `<think>`-stripped turns and tool-call
  // turns) retain MTP KV history in O(1) without repriming from position 0.
  [[nodiscard]] std::vector<std::byte> snapshot_pending_trunk() const;
  void restore_frontier(std::uint32_t committed_hist_tokens,
                        std::span<const std::byte> pending_trunk_bytes);

  [[nodiscard]] std::optional<MtpPrefixCheckpoint> checkpoint_prefix();
  void restore_prefix(const MtpPrefixCheckpoint &checkpoint,
                      std::uint32_t committed_hist_tokens);
  void set_kv_reclaimer(std::function<bool()> reclaimer);

  // Seals the final verified round's committed tokens `input[0..count-1]`
  // (`count` in 1..5) against `target_trunk_hidden[0..count-1]` at decode-end
  // so `committed_tokens() == target.position() - 1` and `has_pending_trunk()`
  // holds ready for the next turn's prefix-cache hit.
  MetalSubmission
  seal_post_verify_frontier(std::span<const std::uint32_t> committed_round_tokens,
                            bool use_icb = true);

  // Folds committed_tokens[0..c-1] (c in 1..5) without running lm_head or
  // drafting (used when MtpDepthController probes/parks at K == 0 so MTP KV
  // history stays 100% aligned with target.position() at minimal cost).
  MetalSubmission
  fold_committed_only(std::span<const std::uint32_t> committed_tokens,
                      bool use_icb = true);

  // Single-CB fused history fold (c = committed_tokens.size() in 1..5) +
  // chained T=1 draft generation (K = draft_tokens in 1..4).
  // Automatically rewinds speculative tail via mtp_kv_pool.truncate(hist_offset)
  // in O(1) before appending the new cycle!
  [[nodiscard]] DFlash2DraftResult
  propose(std::span<const std::uint32_t> committed_tokens,
          std::uint32_t target_position, std::uint32_t draft_tokens,
          const GpuSamplingOptions &sampling, bool use_icb = true,
          bool tokens_only = false);

  [[nodiscard]] MtpPreparedProposal
  prepare_propose(std::span<const std::uint32_t> committed_tokens,
                  std::uint32_t target_position, std::uint32_t draft_tokens,
                  const GpuSamplingOptions &sampling, bool use_icb = true,
                  bool tokens_only = false);

  [[nodiscard]] DFlash2DraftResult
  finish_propose(MtpPreparedProposal &&prepared, bool already_executed,
                 MetalSubmission batched_submission = {});

  [[nodiscard]] bool
  can_propose_packed_batch(std::uint32_t batch_size) const noexcept;

  // Cross-slot packed multi-slot MTP drafting (B in 2..8): shares MTP Layer 0
  // and target lm_head weights ONCE per depth step d = 0..max_K-1 across all B
  // slots in ONE Compute ICB submission!
  [[nodiscard]] MtpPackedBatchResult
  propose_packed_batch(std::span<const MtpPackedSlotSlice> slices,
                       bool use_icb = true);

  [[nodiscard]] std::uint32_t committed_tokens() const noexcept;
  [[nodiscard]] std::uint32_t kv_tokens() const noexcept;
  [[nodiscard]] bool has_pending_trunk() const noexcept;
  [[nodiscard]] bool ready_for_target_position(std::uint32_t target_pos) const noexcept;
  [[nodiscard]] std::string_view prefix() const noexcept;
  [[nodiscard]] bool is_moe_layer() const noexcept;
  [[nodiscard]] const MtpNormOffsets &norm_offsets() const noexcept;
  [[nodiscard]] QwenProfileDescription
  inspect_profile(std::uint32_t fold_tokens = 1,
                  std::uint32_t draft_tokens = 4) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5
