#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/position_sampling.h"

namespace omlx::m5 {

struct EmbeddingInfo {
  std::string executor_key;
  std::string format_id;
  std::uint32_t tokens = 0;
  std::uint64_t vocabulary = 0;
  std::uint64_t row_start = 0;
  std::uint64_t row_count = 0;
  std::uint64_t hidden_size = 0;
  std::uint64_t token_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t weight_plane_bytes = 0;
  std::uint64_t dispatch_count = 1;
};

class EmbeddingPlan final {
public:
  EmbeddingPlan();
  EmbeddingPlan(EmbeddingPlan &&) noexcept;
  EmbeddingPlan &operator=(EmbeddingPlan &&) noexcept;
  EmbeddingPlan(const EmbeddingPlan &) = delete;
  EmbeddingPlan &operator=(const EmbeddingPlan &) = delete;
  ~EmbeddingPlan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const EmbeddingInfo &info() const;

private:
  class Impl;
  friend EmbeddingPlan prepare_embedding(const MetalContext &,
                                         const std::filesystem::path &,
                                         MaterializedWeightView, std::uint32_t);
  friend MetalSubmission execute_embedding(const MetalContext &,
                                           const EmbeddingPlan &,
                                           const MetalBuffer &, std::uint64_t,
                                           MetalBuffer &, std::uint64_t);

  explicit EmbeddingPlan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] EmbeddingPlan
prepare_embedding(const MetalContext &context,
                  const std::filesystem::path &metallib,
                  MaterializedWeightView weight, std::uint32_t tokens);

[[nodiscard]] MetalSubmission
execute_embedding(const MetalContext &context, const EmbeddingPlan &plan,
                  const MetalBuffer &token_ids, std::uint64_t token_offset,
                  MetalBuffer &output, std::uint64_t output_offset);

struct GpuSamplingOptions {
  float temperature = 0.0F;
  std::uint32_t top_k = 0;
  float top_p = 1.0F;
  std::uint64_t seed = 0;
  std::uint64_t stream = 0;
  std::uint64_t absolute_position = 0;
  SamplingContract contract = SamplingContract::LegacyRejectionV1;
};

struct SamplingInfo {
  std::string executor_key;
  std::uint64_t vocabulary = 0;
  std::uint64_t logits_bytes = 0;
  std::uint64_t output_bytes = 4;
  std::uint32_t maximum_top_k = 64;
  std::uint64_t dispatch_count = 1;
};

class SamplingPlan final {
public:
  SamplingPlan();
  SamplingPlan(SamplingPlan &&) noexcept;
  SamplingPlan &operator=(SamplingPlan &&) noexcept;
  SamplingPlan(const SamplingPlan &) = delete;
  SamplingPlan &operator=(const SamplingPlan &) = delete;
  ~SamplingPlan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const SamplingInfo &info() const;

private:
  class Impl;
  friend SamplingPlan prepare_sampling(const MetalContext &,
                                       const std::filesystem::path &,
                                       std::uint64_t);
  friend MetalSubmission execute_sampling(const MetalContext &,
                                          const SamplingPlan &,
                                          const MetalBuffer &, std::uint64_t,
                                          const GpuSamplingOptions &,
                                          MetalBuffer &, std::uint64_t);

  explicit SamplingPlan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] SamplingPlan
prepare_sampling(const MetalContext &context,
                 const std::filesystem::path &metallib,
                 std::uint64_t vocabulary);

// Greedy uses temperature=0. Sampled mode requires 1<=top_k<=64 and applies
// top_p inside that candidate set. Only the selected uint32 token leaves GPU.
[[nodiscard]] MetalSubmission
execute_sampling(const MetalContext &context, const SamplingPlan &plan,
                 const MetalBuffer &logits, std::uint64_t logits_offset,
                 const GpuSamplingOptions &options, MetalBuffer &token,
                 std::uint64_t token_offset = 0);

} // namespace omlx::m5
