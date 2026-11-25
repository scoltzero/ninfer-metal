#pragma once

#include "omlx_m5/dflash2.h"
#include "omlx_m5/qwen_forward.h"

namespace omlx::m5 {

[[nodiscard]] QwenForwardMemoryPlan
plan_dflash_forward_memory(const DFlash2ModelConfig &,
                           std::uint32_t max_context);

struct DFlashPreparedProposal {
  DFlash2DraftResult partial;
  MetalIcbProfile *icb = nullptr;
  const void *execution_profile = nullptr;
  std::uint32_t anchor = 0;
  std::uint32_t context_tokens = 0;
  std::uint32_t draft_tokens = 0;
  std::uint32_t width = 0;
  float temperature = 0.0f;
  std::uint64_t seed = 0;
  bool inspect_tensors = false;
  bool cpu_selector = false;
  bool tokens_only = false;
};

class DFlashForward;

struct DFlashPackedSlotSlice {
  DFlashForward *slot_draft = nullptr;
  std::uint32_t anchor = 0;
  std::uint32_t context_tokens = 0;
  std::uint32_t draft_tokens = 0;
  float temperature = 0.0f;
  std::uint64_t seed = 0;
  bool tokens_only = false;
};

struct DFlashPackedBatchResult {
  MetalSubmission submission;
  std::vector<DFlash2DraftResult> slot_results;
};

// Real companion execution. The target owns shared embedding/head and the
// committed GPU feature history; all of those objects outlive this executor.
class DFlashForward final {
public:
  DFlashForward(const MetalContext &, const MaterializedModel &,
                DFlash2ModelConfig, std::uint32_t max_context,
                const MaterializedWeightView &embedding,
                const MaterializedWeightView &head,
                const MetalBuffer &target_features,
                const std::filesystem::path &libraries,
                bool enable_m4_short_routes = false,
                LinearNumericPolicy linear_policy = LinearNumericPolicy::Legacy,
                SamplingContract sampling_contract =
                    SamplingContract::LegacyRejectionV1);
  ~DFlashForward();
  [[nodiscard]] DFlash2DraftResult
  propose(std::uint32_t anchor, std::uint32_t context_tokens,
          std::uint32_t draft_tokens, float temperature, std::uint64_t seed,
          bool use_icb = true, bool inspect_tensors = false,
          bool cpu_selector = false, bool tokens_only = false);
  [[nodiscard]] DFlashPreparedProposal
  prepare_propose(std::uint32_t anchor, std::uint32_t context_tokens,
                  std::uint32_t draft_tokens, float temperature,
                  std::uint64_t seed, bool use_icb = true,
                  bool inspect_tensors = false, bool cpu_selector = false,
                  bool tokens_only = false);
  [[nodiscard]] DFlash2DraftResult
  finish_propose(DFlashPreparedProposal &&prepared, bool already_executed,
                 MetalSubmission batched_submission = {});
  [[nodiscard]] DFlashPackedBatchResult
  propose_packed_batch(std::span<const DFlashPackedSlotSlice> slices,
                       bool use_icb = true, bool cpu_selector = false);
  void set_target_feature_epoch(std::uint64_t epoch) noexcept;
  [[nodiscard]] std::uint64_t target_feature_epoch() const noexcept;
  void set_target_feature_ring(std::uint32_t ring_rows);
  [[nodiscard]] std::uint32_t target_feature_ring() const noexcept;
  void reset();
  [[nodiscard]] std::uint32_t cached_context() const;
  [[nodiscard]] const DFlash2ModelConfig &config() const;
  // Diagnostic range replay adds submissions; never use its sum as production
  // proposal latency.
  [[nodiscard]] std::vector<QwenStageTiming> profile_stages();
  [[nodiscard]] QwenProfileDescription inspect_profile() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5
