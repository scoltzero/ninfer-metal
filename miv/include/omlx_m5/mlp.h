#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "omlx_m5/activation.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/projection.h"

namespace omlx::m5 {

enum class SwiGluProducerMode {
  a16,
  a4,
  a16_a4,
};

[[nodiscard]] std::string_view to_string(SwiGluProducerMode mode);

struct SwiGluActivationView {
  std::optional<ActivationPlaneView> a16;
  std::optional<Mxfp4ActivationView> a4;
  std::uint32_t rows = 0;
  std::uint64_t features = 0;
};

struct SwiGluInfo {
  std::string executor_key;
  std::uint32_t rows = 0;
  std::uint64_t features = 0;
  std::uint64_t input_bytes = 0;
  std::uint64_t a16_bytes = 0;
  std::uint64_t a4_code_bytes = 0;
  std::uint64_t a4_scale_bytes = 0;
  std::uint64_t dispatch_count = 1;
};

class SwiGluPlan final {
public:
  SwiGluPlan();
  SwiGluPlan(SwiGluPlan &&) noexcept;
  SwiGluPlan &operator=(SwiGluPlan &&) noexcept;
  SwiGluPlan(const SwiGluPlan &) = delete;
  SwiGluPlan &operator=(const SwiGluPlan &) = delete;
  ~SwiGluPlan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const SwiGluInfo &info() const;

private:
  class Impl;
  friend SwiGluPlan prepare_swiglu(const MetalContext &,
                                   const std::filesystem::path &, std::uint32_t,
                                   std::uint64_t);
  friend MetalSubmission
  produce_swiglu_activation(const MetalContext &, const SwiGluPlan &,
                            const ProjectionParentOutputView &,
                            const ProjectionParentOutputView &,
                            const SwiGluActivationView &);

  explicit SwiGluPlan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] SwiGluPlan prepare_swiglu(const MetalContext &context,
                                        const std::filesystem::path &metallib,
                                        std::uint32_t rows,
                                        std::uint64_t features);

[[nodiscard]] MetalSubmission
produce_swiglu_activation(const MetalContext &context, const SwiGluPlan &plan,
                          const ProjectionParentOutputView &gate,
                          const ProjectionParentOutputView &up,
                          const SwiGluActivationView &output);

struct MoeTopKReduceInfo {
  std::string executor_key;
  std::uint32_t tokens = 0;
  std::uint32_t top_k = 8;
  std::uint64_t features = 0;
  std::uint64_t input_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t dispatch_count = 1;
};

class MoeTopKReducePlan final {
public:
  MoeTopKReducePlan();
  MoeTopKReducePlan(MoeTopKReducePlan &&) noexcept;
  MoeTopKReducePlan &operator=(MoeTopKReducePlan &&) noexcept;
  MoeTopKReducePlan(const MoeTopKReducePlan &) = delete;
  MoeTopKReducePlan &operator=(const MoeTopKReducePlan &) = delete;
  ~MoeTopKReducePlan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const MoeTopKReduceInfo &info() const;

private:
  class Impl;
  friend MoeTopKReducePlan
  prepare_moe_topk_reduce(const MetalContext &, const std::filesystem::path &,
                          std::uint32_t, std::uint64_t, std::uint32_t);
  friend MetalSubmission execute_moe_topk_reduce(const MetalContext &,
                                                 const MoeTopKReducePlan &,
                                                 const MetalBuffer &,
                                                 std::uint64_t, MetalBuffer &,
                                                 std::uint64_t);

  explicit MoeTopKReducePlan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] MoeTopKReducePlan prepare_moe_topk_reduce(
    const MetalContext &context, const std::filesystem::path &metallib,
    std::uint32_t tokens, std::uint64_t features, std::uint32_t top_k = 8);

[[nodiscard]] MetalSubmission execute_moe_topk_reduce(
    const MetalContext &context, const MoeTopKReducePlan &plan,
    const MetalBuffer &assignment_output, std::uint64_t input_offset,
    MetalBuffer &token_output, std::uint64_t output_offset);

struct MoeSharedCombineInfo {
  std::string executor_key;
  std::uint32_t tokens = 0;
  std::uint64_t features = 0;
  std::uint64_t tensor_bytes = 0;
  std::uint64_t gate_bytes = 0;
  std::uint64_t dispatch_count = 1;
};

class MoeSharedCombinePlan final {
public:
  MoeSharedCombinePlan();
  MoeSharedCombinePlan(MoeSharedCombinePlan &&) noexcept;
  MoeSharedCombinePlan &operator=(MoeSharedCombinePlan &&) noexcept;
  MoeSharedCombinePlan(const MoeSharedCombinePlan &) = delete;
  MoeSharedCombinePlan &operator=(const MoeSharedCombinePlan &) = delete;
  ~MoeSharedCombinePlan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const MoeSharedCombineInfo &info() const;

private:
  class Impl;
  friend MoeSharedCombinePlan
  prepare_moe_shared_combine(const MetalContext &,
                             const std::filesystem::path &, std::uint32_t,
                             std::uint64_t);
  friend MetalSubmission execute_moe_shared_combine(
      const MetalContext &, const MoeSharedCombinePlan &, const MetalBuffer &,
      std::uint64_t, const MetalBuffer &, std::uint64_t, const MetalBuffer &,
      std::uint64_t, MetalBuffer &, std::uint64_t);

  explicit MoeSharedCombinePlan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] MoeSharedCombinePlan
prepare_moe_shared_combine(const MetalContext &context,
                           const std::filesystem::path &metallib,
                           std::uint32_t tokens, std::uint64_t features);

[[nodiscard]] MetalSubmission execute_moe_shared_combine(
    const MetalContext &context, const MoeSharedCombinePlan &plan,
    const MetalBuffer &routed, std::uint64_t routed_offset,
    const MetalBuffer &shared, std::uint64_t shared_offset,
    const MetalBuffer &shared_gate, std::uint64_t gate_offset,
    MetalBuffer &output, std::uint64_t output_offset);

} // namespace omlx::m5
