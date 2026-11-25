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

enum class DenseQmmBroadFamily {
  oq_cooperative,
  mxfp4_w4a16_simd,
  mxfp4_w4a4_simd,
  mxfp4_w4a16_tensorops,
  mxfp4_w4a4_tensorops,
};

[[nodiscard]] std::string_view to_string(DenseQmmBroadFamily value);

struct DenseQmmBroadExecutorInfo {
  DenseQmmBroadFamily family = DenseQmmBroadFamily::oq_cooperative;
  std::string executor_key;
  std::string function_name;
  ActivationProfile activation = ActivationProfile::a16;
  MetalExecutionRoute required_route = MetalExecutionRoute::classic_msl;
  std::uint64_t dispatch_count = 0;
  bool installed = false;
};

struct DenseQmmBroadInfo {
  std::string format_id;
  std::string layout_id;
  std::string domain;
  std::uint64_t output_features = 0;
  std::uint64_t input_features = 0;
  std::uint32_t tokens = 0;
  std::uint32_t token_tile = 16;
  std::uint32_t token_tiles = 0;
  std::uint64_t input_a16_bytes = 0;
  std::uint64_t input_a4_code_bytes = 0;
  std::uint64_t input_a4_scale_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t weight_plane_bytes = 0;
  std::uint64_t workspace_bytes = 0;
  bool native_tensorops_runtime_eligible = false;
  std::vector<DenseQmmBroadExecutorInfo> executors;
};

class DenseQmmBroadPlan final {
public:
  DenseQmmBroadPlan();
  DenseQmmBroadPlan(DenseQmmBroadPlan &&) noexcept;
  DenseQmmBroadPlan &operator=(DenseQmmBroadPlan &&) noexcept;
  DenseQmmBroadPlan(const DenseQmmBroadPlan &) = delete;
  DenseQmmBroadPlan &operator=(const DenseQmmBroadPlan &) = delete;
  ~DenseQmmBroadPlan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const DenseQmmBroadInfo &info() const;
  [[nodiscard]] std::vector<std::string> installed_executor_keys() const;
  [[nodiscard]] std::vector<KernelCandidate> dispatch_candidates() const;
  [[nodiscard]] const DenseQmmBroadExecutorInfo &
  executor(std::string_view executor_key) const;

private:
  class Impl;
  friend DenseQmmBroadPlan
  prepare_dense_qmm_broad(const MetalContext &, const std::filesystem::path &,
                          const std::filesystem::path &, MaterializedWeightView,
                          std::uint32_t);
  friend MetalSubmission
  execute_dense_qmm_broad_a16(const MetalContext &, const DenseQmmBroadPlan &,
                              std::string_view, const MetalBuffer &,
                              std::uint64_t, MetalBuffer &, std::uint64_t);
  friend MetalSubmission
  execute_dense_qmm_broad_a4(const MetalContext &, const DenseQmmBroadPlan &,
                             std::string_view, const Mxfp4ActivationView &,
                             MetalBuffer &, std::uint64_t);

  explicit DenseQmmBroadPlan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] bool dense_broad_tokens_supported(std::uint32_t tokens);

[[nodiscard]] DenseQmmBroadPlan
prepare_dense_qmm_broad(const MetalContext &context,
                        const std::filesystem::path &broad_metallib,
                        const std::filesystem::path &tensorops_metallib,
                        MaterializedWeightView weight, std::uint32_t tokens);

[[nodiscard]] MetalSubmission execute_dense_qmm_broad_a16(
    const MetalContext &context, const DenseQmmBroadPlan &plan,
    std::string_view executor_key, const MetalBuffer &input,
    std::uint64_t input_offset, MetalBuffer &output,
    std::uint64_t output_offset);

[[nodiscard]] MetalSubmission execute_dense_qmm_broad_a4(
    const MetalContext &context, const DenseQmmBroadPlan &plan,
    std::string_view executor_key, const Mxfp4ActivationView &input,
    MetalBuffer &output, std::uint64_t output_offset);

} // namespace omlx::m5
