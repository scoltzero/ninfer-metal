#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "omlx_m5/activation.h"
#include "omlx_m5/metal.h"

namespace omlx::m5 {

enum class RmsNormProducerMode {
  none,
  a16,
  a4,
  a16_a4,
};

[[nodiscard]] std::string_view to_string(RmsNormProducerMode mode);

struct RmsNormProjectionInfo {
  std::string executor_key;
  std::uint64_t input_features = 0;
  std::uint32_t tokens = 0;
  float epsilon = 0.0F;
  std::uint64_t input_bytes = 0;
  std::uint64_t weight_bytes = 0;
  std::uint64_t a16_bytes = 0;
  std::uint64_t a4_code_bytes = 0;
  std::uint64_t a4_scale_bytes = 0;
};

struct RmsNormProduction {
  RmsNormProducerMode mode = RmsNormProducerMode::none;
  std::optional<MetalSubmission> submission;
};

class RmsNormProjectionPlan final {
public:
  RmsNormProjectionPlan();
  RmsNormProjectionPlan(RmsNormProjectionPlan &&) noexcept;
  RmsNormProjectionPlan &operator=(RmsNormProjectionPlan &&) noexcept;
  RmsNormProjectionPlan(const RmsNormProjectionPlan &) = delete;
  RmsNormProjectionPlan &operator=(const RmsNormProjectionPlan &) = delete;
  ~RmsNormProjectionPlan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const RmsNormProjectionInfo &info() const;

private:
  class Impl;
  friend RmsNormProjectionPlan
  prepare_rmsnorm_projection(const MetalContext &,
                             const std::filesystem::path &, std::uint64_t,
                             std::uint32_t, float);
  friend RmsNormProduction
  produce_rmsnorm_activation(const MetalContext &,
                             const RmsNormProjectionPlan &, const MetalBuffer &,
                             std::uint64_t, const MetalBuffer &, std::uint64_t,
                             const ActivationLease &);

  explicit RmsNormProjectionPlan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] RmsNormProjectionPlan prepare_rmsnorm_projection(
    const MetalContext &context, const std::filesystem::path &metallib,
    std::uint64_t input_features, std::uint32_t tokens, float epsilon);

[[nodiscard]] RmsNormProduction produce_rmsnorm_activation(
    const MetalContext &context, const RmsNormProjectionPlan &plan,
    const MetalBuffer &input, std::uint64_t input_offset,
    const MetalBuffer &weight, std::uint64_t weight_offset,
    const ActivationLease &lease);

} // namespace omlx::m5
