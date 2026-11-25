#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/dispatch.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"

namespace omlx::m5 {

enum class DenseOqSmallTFamily {
  row_split,
  split_k,
  cooperative,
};

[[nodiscard]] std::string_view to_string(DenseOqSmallTFamily value);

struct DenseOqSmallTExecutorInfo {
  DenseOqSmallTFamily family = DenseOqSmallTFamily::row_split;
  std::string executor_key;
  std::string function_name;
  std::uint32_t capacity = 0;
  std::uint64_t partial_workspace_bytes = 0;
  std::uint64_t dispatch_count = 0;
  std::uint64_t static_threadgroup_memory_bytes = 0;
};

struct DenseOqSmallTInfo {
  std::string format_id;
  std::string layout_id;
  std::uint64_t output_features = 0;
  std::uint64_t input_features = 0;
  std::uint32_t tokens = 0;
  std::uint32_t capacity = 0;
  std::uint64_t input_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t weight_plane_bytes = 0;
  std::uint64_t group_sum_workspace_bytes = 0;
  std::vector<DenseOqSmallTExecutorInfo> executors;
};

class DenseOqSmallTPlan final {
public:
  DenseOqSmallTPlan();
  DenseOqSmallTPlan(DenseOqSmallTPlan &&) noexcept;
  DenseOqSmallTPlan &operator=(DenseOqSmallTPlan &&) noexcept;
  DenseOqSmallTPlan(const DenseOqSmallTPlan &) = delete;
  DenseOqSmallTPlan &operator=(const DenseOqSmallTPlan &) = delete;
  ~DenseOqSmallTPlan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const DenseOqSmallTInfo &info() const;
  [[nodiscard]] std::vector<std::string> installed_executor_keys() const;
  [[nodiscard]] std::vector<KernelCandidate> dispatch_candidates() const;
  [[nodiscard]] const DenseOqSmallTExecutorInfo &
  executor(std::string_view executor_key) const;

private:
  class Impl;
  friend DenseOqSmallTPlan
  prepare_dense_oq_small_t(const MetalContext &, const std::filesystem::path &,
                           MaterializedWeightView, std::uint32_t);
  friend MetalSubmission
  execute_dense_oq_small_t(const MetalContext &, const DenseOqSmallTPlan &,
                           std::string_view, const MetalBuffer &, std::uint64_t,
                           MetalBuffer &, std::uint64_t);

  explicit DenseOqSmallTPlan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] DenseOqSmallTPlan
prepare_dense_oq_small_t(const MetalContext &context,
                         const std::filesystem::path &metallib,
                         MaterializedWeightView weight, std::uint32_t tokens);

[[nodiscard]] MetalSubmission
execute_dense_oq_small_t(const MetalContext &context,
                         const DenseOqSmallTPlan &plan,
                         std::string_view executor_key,
                         const MetalBuffer &input, std::uint64_t input_offset,
                         MetalBuffer &output, std::uint64_t output_offset);

} // namespace omlx::m5
