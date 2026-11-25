#pragma once
#include "omlx_m5/metal.h"
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace omlx::m5 {
enum class SamplingContract { LegacyRejectionV1, PositionCoupledV1 };
[[nodiscard]] SamplingContract parse_sampling_contract(std::string_view);
[[nodiscard]] std::string_view sampling_contract_name(SamplingContract);
constexpr std::uint64_t target_sampling_purpose=0x5441524745540001ULL;
constexpr std::uint64_t draft_sampling_purpose=0x4452414654000001ULL;
struct PositionSamplingRow {
  std::uint64_t seed=0,position=0,purpose=target_sampling_purpose;
  std::uint32_t top_k=0;
  float temperature=0,top_p=1;
  std::uint32_t valid=1;
};
static_assert(sizeof(PositionSamplingRow)==40);
[[nodiscard]] float position_sampling_uniform(std::uint64_t seed,std::uint64_t position,
                                              std::uint32_t token,std::uint64_t purpose);
// Canonical compact host fallback: float64 filtering/Gumbel over represented
// input scores. It never consumes a sequential RNG or depends on row/slot ID.
[[nodiscard]] std::uint32_t position_sample_candidates(std::span<const std::uint32_t> ids,
                                                       std::span<const float> scores,
                                                       const PositionSamplingRow &);
struct PositionSamplingResult {
  std::vector<std::uint32_t> tokens;
  std::uint64_t host_read_bytes=0,compact_fallback_rows=0;
  MetalSubmission submission;
};
class PositionKeyedSampler final {
public:
  PositionKeyedSampler(const MetalContext &,const MetalBuffer &logits,
                       std::uint32_t vocabulary,std::uint32_t max_rows,
                       const std::filesystem::path &libraries);
  ~PositionKeyedSampler();
  [[nodiscard]] PositionSamplingResult sample(std::span<const PositionSamplingRow>,bool icb=true);
  // P04/P10 can keep normal tokens on GPU. Only a boundary decision requests
  // compact host resolution; it writes corrected tokens/cleared flags back.
  [[nodiscard]] MetalSubmission sample_device(std::span<const PositionSamplingRow>,bool icb=true);
  [[nodiscard]] PositionSamplingResult resolve_result(std::span<const PositionSamplingRow>);
  [[nodiscard]] const MetalBuffer &tokens_and_flags() const;
private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace omlx::m5
