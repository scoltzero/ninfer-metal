#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "omlx_m5/dflash2.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/position_sampling.h"
#include "omlx_m5/vocab.h"

namespace omlx::m5 {

struct TargetGpuVerifyDecision {
  VerificationResult verification;
  MetalSubmission submission;
  std::uint64_t host_read_bytes = 0;
  bool compact_cpu_fallback = false;
};

struct TargetGpuPendingSelect {
  std::vector<PositionSamplingRow> rows;
  std::uint32_t width = 0;
  std::uint32_t proposal_count = 0;
  bool use_icb = true;
  bool async_icb = false;
  MetalPendingSubmission pending;
  TargetGpuVerifyDecision sync_result;
};

struct TargetGpuPreparedSelect {
  std::vector<PositionSamplingRow> rows;
  std::vector<SelectorStep> legacy_proposals;
  GpuSamplingOptions legacy_sampling{};
  std::uint64_t absolute_position = 0;
  std::uint32_t width = 0;
  std::uint32_t proposal_count = 0;
  std::uint32_t legacy_topk = 1;
  bool use_icb = true;
  bool coupled = false;
  MetalIcbProfile *icb = nullptr;
};

[[nodiscard]] std::uint64_t
target_gpu_verify_workspace_bytes(std::uint32_t vocabulary,
                                  SamplingContract contract=SamplingContract::LegacyRejectionV1);

// Consumes the pending target's BF16 logits without changing target state.
// Preview/commit/publication stay with the request owner.
class TargetGpuVerifier final {
public:
  TargetGpuVerifier(const MetalContext &, const MetalBuffer &logits,
                    std::uint32_t vocabulary,
                    const std::filesystem::path &libraries,
                    SamplingContract contract=SamplingContract::LegacyRejectionV1);
  ~TargetGpuVerifier();
  TargetGpuVerifier(const TargetGpuVerifier &) = delete;
  TargetGpuVerifier &operator=(const TargetGpuVerifier &) = delete;

  [[nodiscard]] bool can_enqueue_select(bool use_icb = true) const noexcept;

  [[nodiscard]] TargetGpuPendingSelect
  enqueue_select(std::span<const SelectorStep> proposals,
                 const GpuSamplingOptions &sampling,
                 std::uint64_t absolute_position, bool use_icb = true);

  [[nodiscard]] TargetGpuVerifyDecision
  finish_select(TargetGpuPendingSelect &&pending);

  [[nodiscard]] TargetGpuPreparedSelect
  prepare_select(std::span<const SelectorStep> proposals,
                 const GpuSamplingOptions &sampling,
                 std::uint64_t absolute_position, bool use_icb = true);

  [[nodiscard]] TargetGpuVerifyDecision
  finish_prepared_select(TargetGpuPreparedSelect &&prepared,
                         bool already_executed,
                         MetalSubmission batched_submission = {});

  [[nodiscard]] TargetGpuVerifyDecision
  select(std::span<const SelectorStep> proposals,
         const GpuSamplingOptions &sampling, std::uint64_t absolute_position,
         bool use_icb = true);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5
