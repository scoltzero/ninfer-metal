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

struct MoeRouteView {
  ActivationPlaneView indices;
  ActivationPlaneView scores;
  std::uint32_t tokens = 0;
  std::uint32_t experts = 256;
  std::uint32_t top_k = 8;
};

struct MoeRouterTop8Info {
  std::string executor_key;
  std::uint32_t tokens = 0;
  std::uint32_t experts = 256;
  std::uint32_t top_k = 8;
  std::uint64_t probability_bytes = 0;
  std::uint64_t index_bytes = 0;
  std::uint64_t score_bytes = 0;
  std::uint64_t dispatch_count = 1;
};

class MoeRouterTop8Plan final {
public:
  MoeRouterTop8Plan();
  MoeRouterTop8Plan(MoeRouterTop8Plan &&) noexcept;
  MoeRouterTop8Plan &operator=(MoeRouterTop8Plan &&) noexcept;
  MoeRouterTop8Plan(const MoeRouterTop8Plan &) = delete;
  MoeRouterTop8Plan &operator=(const MoeRouterTop8Plan &) = delete;
  ~MoeRouterTop8Plan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const MoeRouterTop8Info &info() const;
  [[nodiscard]] std::vector<KernelCandidate> dispatch_candidates() const;

private:
  class Impl;
  friend MoeRouterTop8Plan
  prepare_moe_router_top8(const MetalContext &, const std::filesystem::path &,
                          std::uint32_t);
  friend MetalSubmission execute_moe_router_top8(const MetalContext &,
                                                 const MoeRouterTop8Plan &,
                                                 const MetalBuffer &,
                                                 std::uint64_t,
                                                 const MoeRouteView &);

  explicit MoeRouterTop8Plan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] MoeRouterTop8Plan
prepare_moe_router_top8(const MetalContext &context,
                        const std::filesystem::path &metallib,
                        std::uint32_t tokens);

[[nodiscard]] MetalSubmission execute_moe_router_top8(
    const MetalContext &context, const MoeRouterTop8Plan &plan,
    const MetalBuffer &probabilities, std::uint64_t probability_offset,
    const MoeRouteView &output);

enum class MoeExpertFamily {
  grouped_a16,
  fallback_a16,
  grouped_a4,
  fallback_a4,
};

[[nodiscard]] std::string_view to_string(MoeExpertFamily family);

struct MoeExpertExecutorInfo {
  MoeExpertFamily family = MoeExpertFamily::grouped_a16;
  std::string executor_key;
  std::string function_name;
  ActivationProfile activation = ActivationProfile::a16;
  KernelFamily kernel_family = KernelFamily::grouped_moe;
  KernelAlignmentClass alignment = KernelAlignmentClass::full;
  std::uint64_t dispatch_count = 0;
};

struct MoeExpertInfo {
  std::string format_id;
  std::string layout_id;
  KernelFusedOp fused_op = KernelFusedOp::moe_gate_up;
  std::uint32_t experts = 256;
  std::uint32_t top_k = 8;
  std::uint32_t tokens = 0;
  std::uint64_t assignments = 0;
  std::uint64_t input_rows = 0;
  std::uint64_t output_features = 0;
  std::uint64_t input_features = 0;
  std::uint64_t input_a16_bytes = 0;
  std::uint64_t input_a4_code_bytes = 0;
  std::uint64_t input_a4_scale_bytes = 0;
  std::uint64_t route_index_bytes = 0;
  std::uint64_t route_score_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t weight_plane_bytes = 0;
  std::uint64_t workspace_bytes = 0;
  bool apply_route_scores = false;
  bool assignment_input = false;
  std::vector<MoeExpertExecutorInfo> executors;
};

class MoeExpertPlan final {
public:
  MoeExpertPlan();
  MoeExpertPlan(MoeExpertPlan &&) noexcept;
  MoeExpertPlan &operator=(MoeExpertPlan &&) noexcept;
  MoeExpertPlan(const MoeExpertPlan &) = delete;
  MoeExpertPlan &operator=(const MoeExpertPlan &) = delete;
  ~MoeExpertPlan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const MoeExpertInfo &info() const;
  [[nodiscard]] std::vector<std::string> installed_executor_keys() const;
  [[nodiscard]] std::vector<KernelCandidate> dispatch_candidates() const;
  [[nodiscard]] const MoeExpertExecutorInfo &
  executor(std::string_view executor_key) const;

private:
  class Impl;
  friend MoeExpertPlan prepare_moe_expert(const MetalContext &,
                                          const std::filesystem::path &,
                                          MaterializedWeightView, std::uint32_t,
                                          KernelFusedOp, bool);
  friend MetalSubmission
  execute_moe_expert_a16(const MetalContext &, const MoeExpertPlan &,
                         std::string_view, const MetalBuffer &, std::uint64_t,
                         const MoeRouteView &, MetalBuffer &, std::uint64_t);
  friend MetalSubmission
  execute_moe_expert_a4(const MetalContext &, const MoeExpertPlan &,
                        std::string_view, const Mxfp4ActivationView &,
                        const MoeRouteView &, MetalBuffer &, std::uint64_t);

  explicit MoeExpertPlan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] MoeExpertPlan
prepare_moe_expert(const MetalContext &context,
                   const std::filesystem::path &metallib,
                   MaterializedWeightView expert_parent, std::uint32_t tokens,
                   KernelFusedOp fused_op = KernelFusedOp::moe_gate_up,
                   bool apply_route_scores = false);

[[nodiscard]] MetalSubmission
execute_moe_expert_a16(const MetalContext &context, const MoeExpertPlan &plan,
                       std::string_view executor_key, const MetalBuffer &input,
                       std::uint64_t input_offset, const MoeRouteView &routes,
                       MetalBuffer &output, std::uint64_t output_offset);

[[nodiscard]] MetalSubmission
execute_moe_expert_a4(const MetalContext &context, const MoeExpertPlan &plan,
                      std::string_view executor_key,
                      const Mxfp4ActivationView &input,
                      const MoeRouteView &routes, MetalBuffer &output,
                      std::uint64_t output_offset);

} // namespace omlx::m5
