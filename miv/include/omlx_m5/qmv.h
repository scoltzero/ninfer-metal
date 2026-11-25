#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "omlx_m5/dispatch.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"

namespace omlx::m5 {

struct DenseQmvT1Info {
  std::string executor_key;
  std::string pipeline_key;
  std::string function_name;
  std::string format_id;
  std::string layout_id;
  std::uint64_t output_features = 0;
  std::uint64_t input_features = 0;
  std::uint64_t input_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t weight_plane_bytes = 0;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t dispatch_count = 0;
};

class DenseQmvT1Plan final {
public:
  DenseQmvT1Plan();
  DenseQmvT1Plan(DenseQmvT1Plan &&) noexcept;
  DenseQmvT1Plan &operator=(DenseQmvT1Plan &&) noexcept;
  DenseQmvT1Plan(const DenseQmvT1Plan &) = delete;
  DenseQmvT1Plan &operator=(const DenseQmvT1Plan &) = delete;
  ~DenseQmvT1Plan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const DenseQmvT1Info &info() const;
  [[nodiscard]] std::vector<KernelCandidate>
  dispatch_candidates(std::uint32_t full_priority = 10,
                      std::uint32_t tail_priority = 1000) const;

private:
  class Impl;
  friend DenseQmvT1Plan prepare_dense_qmv_t1(const MetalContext &,
                                             const std::filesystem::path &,
                                             MaterializedWeightView);
  friend MetalSubmission execute_dense_qmv_t1(const MetalContext &,
                                              const DenseQmvT1Plan &,
                                              const MetalBuffer &,
                                              std::uint64_t, MetalBuffer &,
                                              std::uint64_t);

  explicit DenseQmvT1Plan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] DenseQmvT1Plan
prepare_dense_qmv_t1(const MetalContext &context,
                     const std::filesystem::path &metallib,
                     MaterializedWeightView weight);

[[nodiscard]] MetalSubmission
execute_dense_qmv_t1(const MetalContext &context, const DenseQmvT1Plan &plan,
                     const MetalBuffer &input, std::uint64_t input_offset,
                     MetalBuffer &output, std::uint64_t output_offset);

} // namespace omlx::m5
