#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/activation.h"
#include "omlx_m5/dispatch.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"

namespace omlx::m5 {

struct Mxfp4A4QuantizeInfo {
  std::string executor_key;
  std::uint32_t tokens = 0;
  std::uint64_t input_features = 0;
  std::uint64_t input_bytes = 0;
  std::uint64_t code_bytes = 0;
  std::uint64_t scale_bytes = 0;
  std::uint64_t dispatch_count = 0;
};

class Mxfp4A4QuantizePlan final {
public:
  Mxfp4A4QuantizePlan();
  Mxfp4A4QuantizePlan(Mxfp4A4QuantizePlan &&) noexcept;
  Mxfp4A4QuantizePlan &operator=(Mxfp4A4QuantizePlan &&) noexcept;
  Mxfp4A4QuantizePlan(const Mxfp4A4QuantizePlan &) = delete;
  Mxfp4A4QuantizePlan &operator=(const Mxfp4A4QuantizePlan &) = delete;
  ~Mxfp4A4QuantizePlan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const Mxfp4A4QuantizeInfo &info() const;

private:
  class Impl;
  friend Mxfp4A4QuantizePlan
  prepare_mxfp4_a4_quantizer(const MetalContext &,
                             const std::filesystem::path &, std::uint64_t,
                             std::uint32_t);
  friend MetalSubmission quantize_mxfp4_a4(const MetalContext &,
                                           const Mxfp4A4QuantizePlan &,
                                           const MetalBuffer &, std::uint64_t,
                                           const Mxfp4ActivationView &);

  explicit Mxfp4A4QuantizePlan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] Mxfp4A4QuantizePlan
prepare_mxfp4_a4_quantizer(const MetalContext &context,
                           const std::filesystem::path &classic_metallib,
                           std::uint64_t input_features, std::uint32_t tokens);

[[nodiscard]] MetalSubmission
quantize_mxfp4_a4(const MetalContext &context, const Mxfp4A4QuantizePlan &plan,
                  const MetalBuffer &input, std::uint64_t input_offset,
                  const Mxfp4ActivationView &output);

enum class DenseMxfp4SmallTFamily {
  w4a16_simd,
  w4a4_simd,
  w4a16_tensorops,
  w4a4_tensorops,
};

[[nodiscard]] std::string_view to_string(DenseMxfp4SmallTFamily value);

struct DenseMxfp4SmallTExecutorInfo {
  DenseMxfp4SmallTFamily family = DenseMxfp4SmallTFamily::w4a16_simd;
  std::string executor_key;
  std::string function_name;
  ActivationProfile activation = ActivationProfile::a16;
  MetalExecutionRoute required_route = MetalExecutionRoute::classic_msl;
  std::uint32_t capacity = 0;
  std::uint64_t dispatch_count = 0;
  bool installed = false;
};

struct DenseMxfp4SmallTInfo {
  std::string format_id;
  std::string layout_id;
  std::uint64_t output_features = 0;
  std::uint64_t input_features = 0;
  std::uint32_t tokens = 0;
  std::uint32_t capacity = 0;
  std::uint64_t input_a16_bytes = 0;
  std::uint64_t input_a4_code_bytes = 0;
  std::uint64_t input_a4_scale_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t weight_plane_bytes = 0;
  bool native_tensorops_runtime_eligible = false;
  std::vector<DenseMxfp4SmallTExecutorInfo> executors;
};

class DenseMxfp4SmallTPlan final {
public:
  DenseMxfp4SmallTPlan();
  DenseMxfp4SmallTPlan(DenseMxfp4SmallTPlan &&) noexcept;
  DenseMxfp4SmallTPlan &operator=(DenseMxfp4SmallTPlan &&) noexcept;
  DenseMxfp4SmallTPlan(const DenseMxfp4SmallTPlan &) = delete;
  DenseMxfp4SmallTPlan &operator=(const DenseMxfp4SmallTPlan &) = delete;
  ~DenseMxfp4SmallTPlan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const DenseMxfp4SmallTInfo &info() const;
  [[nodiscard]] std::vector<std::string> installed_executor_keys() const;
  [[nodiscard]] std::vector<KernelCandidate> dispatch_candidates() const;
  [[nodiscard]] const DenseMxfp4SmallTExecutorInfo &
  executor(std::string_view executor_key) const;

private:
  class Impl;
  friend DenseMxfp4SmallTPlan prepare_dense_mxfp4_small_t(
      const MetalContext &, const std::filesystem::path &,
      const std::filesystem::path &, MaterializedWeightView, std::uint32_t);
  friend MetalSubmission execute_dense_mxfp4_small_t_a16(
      const MetalContext &, const DenseMxfp4SmallTPlan &, std::string_view,
      const MetalBuffer &, std::uint64_t, MetalBuffer &, std::uint64_t);
  friend MetalSubmission execute_dense_mxfp4_small_t_a4(
      const MetalContext &, const DenseMxfp4SmallTPlan &, std::string_view,
      const Mxfp4ActivationView &, MetalBuffer &, std::uint64_t);

  explicit DenseMxfp4SmallTPlan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] DenseMxfp4SmallTPlan prepare_dense_mxfp4_small_t(
    const MetalContext &context, const std::filesystem::path &classic_metallib,
    const std::filesystem::path &tensorops_metallib,
    MaterializedWeightView weight, std::uint32_t tokens);

[[nodiscard]] MetalSubmission execute_dense_mxfp4_small_t_a16(
    const MetalContext &context, const DenseMxfp4SmallTPlan &plan,
    std::string_view executor_key, const MetalBuffer &input,
    std::uint64_t input_offset, MetalBuffer &output,
    std::uint64_t output_offset);

[[nodiscard]] MetalSubmission execute_dense_mxfp4_small_t_a4(
    const MetalContext &context, const DenseMxfp4SmallTPlan &plan,
    std::string_view executor_key, const Mxfp4ActivationView &input,
    MetalBuffer &output, std::uint64_t output_offset);

} // namespace omlx::m5
