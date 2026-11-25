#pragma once
#include "omlx_m5/position_sampling.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "omlx_m5/qwen_program.h"
#include "omlx_m5/state.h"

namespace omlx::m5 {

struct DynamicConvSpec {
  std::uint32_t width = 0;
  std::uint32_t group_width = 16;
};

[[nodiscard]] std::vector<float> dynamic_grouped_conv_two_tap(
    std::span<const float> input, std::uint32_t positions,
    const DynamicConvSpec &spec, std::span<const float> base,
    std::span<const float> delta);

[[nodiscard]] std::vector<float>
materialize_target_context(std::span<const float> concatenated_features,
                           std::uint32_t positions, std::uint32_t input_width,
                           std::span<const float> projection,
                           std::uint32_t hidden_width,
                           std::span<const float> norm_weight, float epsilon);

struct CandidateRow {
  std::vector<std::int32_t> token_ids;
  std::vector<float> unary_scores;
  std::vector<float> projected_hidden;
};

struct SelectorStep {
  std::int32_t token = -1;
  std::vector<std::int32_t> candidate_ids;
  std::vector<float> probabilities;
};

struct SelectorResult {
  std::vector<std::int32_t> tokens;
  std::vector<SelectorStep> steps;
};

// The same position-domain draw used by the CPU selector and GPU proposal.
[[nodiscard]] double candidate_selector_uniform(std::uint64_t seed,
                                                std::uint64_t position);
[[nodiscard]] double speculative_uniform(std::uint64_t seed,
                                         std::uint64_t position,
                                         std::uint64_t purpose);

struct DFlash2ModelConfig {
  std::uint32_t hidden_size = 0;
  std::uint32_t intermediate_size = 0;
  std::uint32_t vocabulary_size = 0;
  std::uint32_t layer_count = 0;
  std::uint32_t query_heads = 0;
  std::uint32_t key_value_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t max_position_embeddings = 0;
  std::uint32_t sliding_window = 0;
  std::uint32_t block_size = 0;
  std::uint32_t conv_group_size = 0;
  std::uint32_t conv_kernel_size = 0;
  std::uint32_t selector_rank = 0;
  std::uint32_t selector_top_k = 0;
  std::uint32_t mask_token_id = 0;
  float epsilon = 0.0F;
  float rope_theta = 0.0F;
  std::vector<std::uint32_t> target_layer_ids;
};

[[nodiscard]] DFlash2ModelConfig
load_dflash2_model_config(const std::filesystem::path &path);

struct DFlash2ArtifactValidation {
  std::uint64_t object_count = 0;
  std::uint64_t tensor_bytes = 0;
  bool companion_complete = false;
  bool shared_embedding_present = false;
  bool proposal_head_present = false;
};

[[nodiscard]] DFlash2ArtifactValidation
validate_dflash2_artifact(const ArtifactModel &artifact,
                          const DFlash2ModelConfig &config,
                          bool require_shared_weights);

struct DFlash2DraftBuildOptions {
  std::uint32_t max_context = 64;
  std::uint32_t max_draft_tokens = 7;
};

struct DFlash2DraftResult {
  SelectorResult selection;
  std::vector<float> hidden;
  std::vector<float> logits;
  std::vector<CandidateRow> candidate_rows;
  std::uint32_t width = 0;
  std::uint64_t draft_dispatches = 0;
  std::uint32_t short_matrix_calls = 0;
  std::uint32_t short_lane_calls = 0;
  std::uint32_t position_parallel_threads = 0;
  bool parallel_backbone = false;
  bool dynamic_convolution = false;
  bool conditional_selector = false;
  std::uint64_t host_read_bytes = 0;
  bool gpu_selector = false;
  bool compact_cpu_fallback = false;
  float oracle_max_abs_error = 0.0F;
  std::uint32_t context_rows_projected = 0;
  std::uint32_t context_rows_reused = 0;
  std::uint32_t context_rows_launched = 0;
  MetalSubmission submission;
};

struct DFlash2DraftStats {
  std::uint64_t rounds = 0;
  std::uint64_t metal_dispatches = 0;
  std::uint64_t context_tokens = 0;
  std::uint64_t proposal_tokens = 0;
};

class DFlash2DraftProgram final {
public:
  static DFlash2DraftProgram create(const std::filesystem::path &artifact_path,
                                    const std::filesystem::path &config_path,
                                    const std::filesystem::path &metallib_path,
                                    DFlash2DraftBuildOptions options = {});

  DFlash2DraftProgram(DFlash2DraftProgram &&) noexcept;
  DFlash2DraftProgram &operator=(DFlash2DraftProgram &&) noexcept;
  DFlash2DraftProgram(const DFlash2DraftProgram &) = delete;
  DFlash2DraftProgram &operator=(const DFlash2DraftProgram &) = delete;
  ~DFlash2DraftProgram();

  [[nodiscard]] const DFlash2ModelConfig &config() const;
  [[nodiscard]] const DFlash2DraftStats &stats() const;
  [[nodiscard]] DFlash2DraftResult
  propose(std::int32_t anchor_token, std::span<const float> target_features,
          std::uint32_t context_tokens, std::uint32_t draft_tokens,
          float temperature, std::uint64_t seed,
          std::uint64_t absolute_position);

private:
  class Impl;
  explicit DFlash2DraftProgram(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] SelectorResult select_candidate_path(
    std::int32_t anchor_token, std::span<const CandidateRow> rows,
    std::span<const float> predecessor_codebook,
    std::span<const float> successor_codebook, std::uint32_t vocab_size,
    std::uint32_t rank, float temperature, std::uint64_t seed,
    std::uint64_t absolute_position,
    SamplingContract contract=SamplingContract::LegacyRejectionV1);

struct TargetDistribution {
  std::vector<float> probabilities;
};

// Exact host fallback from the GPU's sorted top-k (BF16-represented scores).
// The old full-logits path uses the same normalization/top-p implementation.
[[nodiscard]] TargetDistribution target_distribution_from_topk(
    std::span<const std::uint32_t> ids, std::span<const float> scores,
    std::uint32_t vocabulary, float temperature, float top_p);

struct VerificationResult {
  std::vector<std::int32_t> licensed_outputs;
  std::size_t accepted_proposals = 0;
  bool used_correction = false;
};

[[nodiscard]] VerificationResult
verify_speculative(std::span<const SelectorStep> proposals,
                   std::span<const TargetDistribution> target,
                   float temperature, std::uint64_t seed,
                   std::uint64_t absolute_position);

enum class PreviewFinishReason {
  none,
  length,
  eos,
  stop,
  cancelled,
};

struct PreviewResult {
  std::vector<std::int32_t> tokens;
  std::size_t commit_length = 0;
  PreviewFinishReason finish_reason = PreviewFinishReason::none;
};

[[nodiscard]] PreviewResult preview_licensed_outputs(
    std::span<const std::int32_t> licensed, std::size_t remaining_budget,
    std::span<const std::int32_t> eos_token_ids,
    const std::vector<std::vector<std::int32_t>> &stop_sequences,
    bool cancelled);
[[nodiscard]] std::string_view finish_reason_name(PreviewFinishReason reason);

struct DFlashProgramOptions {
  std::uint32_t draft_tokens = 7;
};

struct DFlashRoundOptions {
  std::optional<std::uint32_t> forced_accepted_proposals;
  std::size_t remaining_budget = std::numeric_limits<std::size_t>::max();
  float temperature = 0.0F;
  std::uint64_t seed = 0;
  std::vector<std::int32_t> eos_token_ids;
  std::vector<std::vector<std::int32_t>> stop_sequences;
  bool cancelled = false;
  bool abort_after_verify = false;
};

struct DFlashRoundResult {
  std::uint32_t proposal_count = 0;
  std::uint32_t verify_width = 0;
  std::vector<std::int32_t> proposals;
  std::vector<std::int32_t> target_predictions;
  std::vector<std::int32_t> licensed_outputs;
  std::vector<std::int32_t> published_outputs;
  std::size_t accepted_proposals = 0;
  bool used_correction = false;
  bool used_bonus = false;
  bool teacher_forced_state_parity = false;
  PreviewFinishReason finish_reason = PreviewFinishReason::none;
  RoundStage stage = RoundStage::created;
  std::uint64_t target_state_before = 0;
  std::uint64_t target_verify_state = 0;
  std::uint64_t target_state_after = 0;
  std::uint64_t draft_state_before = 0;
  std::uint64_t draft_state_after = 0;
  std::uint64_t target_weight_publications = 0;
  std::uint64_t target_verify_calls = 0;
  std::uint64_t draft_dispatches = 0;
  std::uint64_t replay_dispatches = 0;
};

struct DFlashProgramStats {
  std::uint64_t rounds = 0;
  std::uint64_t committed_rounds = 0;
  std::uint64_t aborted_rounds = 0;
  std::uint64_t cancelled_rounds = 0;
  std::uint64_t proposal_tokens = 0;
  std::uint64_t accepted_proposals = 0;
  std::uint64_t correction_rounds = 0;
  std::uint64_t bonus_rounds = 0;
  std::uint64_t target_weight_publications = 0;
  std::uint64_t target_verify_calls = 0;
  std::uint64_t draft_dispatches = 0;
  std::uint64_t replay_dispatches = 0;
  std::vector<std::uint64_t> verify_width_histogram;
  std::vector<std::uint64_t> accepted_length_histogram;
};

class QwenDFlashProgram final {
public:
  static QwenDFlashProgram create(QwenProgram &&target,
                                  DFlash2DraftProgram &&draft,
                                  std::span<const std::uint32_t> prompt,
                                  DFlashProgramOptions options = {});

  QwenDFlashProgram(QwenDFlashProgram &&) noexcept;
  QwenDFlashProgram &operator=(QwenDFlashProgram &&) noexcept;
  QwenDFlashProgram(const QwenDFlashProgram &) = delete;
  QwenDFlashProgram &operator=(const QwenDFlashProgram &) = delete;
  ~QwenDFlashProgram();

  [[nodiscard]] std::int32_t pending_anchor() const;
  [[nodiscard]] std::uint32_t configured_draft_tokens() const;
  [[nodiscard]] const SequenceState &sequence_state() const;
  [[nodiscard]] const DFlashProgramStats &stats() const;
  [[nodiscard]] DFlashRoundResult
  run_round(const DFlashRoundOptions &options = {});
  void reset(std::span<const std::uint32_t> prompt);

private:
  class Impl;
  explicit QwenDFlashProgram(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5
