#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/activation.h"
#include "omlx_m5/capability.h"

namespace omlx::m5 {

enum class KernelGeometry {
  dense,
  moe_expert,
  moe_router,
};

enum class KernelDataType {
  bf16,
  fp16,
  fp32,
};

enum class KernelAlignmentClass {
  full,
  tail,
};

enum class KernelFusedOp {
  linear,
  attention_qkv,
  gdn_qkvz,
  dense_gate_up,
  moe_gate_up,
  moe_down,
  split_output,
};

enum class KernelFamily {
  qmv,
  row_split,
  split_k,
  cooperative,
  tensorops,
  grouped_moe,
  per_expert_fallback,
  tail_safe,
};

[[nodiscard]] std::string_view to_string(KernelGeometry value);
[[nodiscard]] std::string_view to_string(KernelDataType value);
[[nodiscard]] std::string_view to_string(KernelAlignmentClass value);
[[nodiscard]] std::string_view to_string(KernelFusedOp value);
[[nodiscard]] std::string_view to_string(KernelFamily value);

struct KernelKey {
  std::string format_id;
  std::string layout_id;
  ActivationPolicy activation_policy = ActivationPolicy::a16_only;
  ActivationProfile preferred_activation = ActivationProfile::a16;
  KernelDataType input_type = KernelDataType::bf16;
  KernelDataType output_type = KernelDataType::bf16;
  KernelDataType accumulator_type = KernelDataType::fp32;
  KernelGeometry geometry = KernelGeometry::dense;
  std::uint64_t output_features = 0;
  std::uint64_t input_features = 0;
  std::uint32_t tokens = 0;
  std::uint32_t experts = 0;
  std::uint32_t top_k = 0;
  KernelFusedOp fused_op = KernelFusedOp::linear;
  KernelAlignmentClass alignment = KernelAlignmentClass::full;

  bool operator==(const KernelKey &) const = default;
};

[[nodiscard]] KernelKey
make_kernel_key(const ArtifactBinding &binding, const ArtifactObject &object,
                std::uint32_t tokens, ActivationProfile preferred_activation,
                KernelFusedOp fused_op = KernelFusedOp::linear,
                KernelAlignmentClass alignment = KernelAlignmentClass::full,
                std::uint32_t moe_top_k = 8);
[[nodiscard]] KernelKey make_direct_kernel_key(
    const ArtifactObject &object, std::uint32_t tokens,
    KernelFusedOp fused_op = KernelFusedOp::linear,
    KernelAlignmentClass alignment = KernelAlignmentClass::full,
    KernelGeometry geometry = KernelGeometry::dense, std::uint32_t experts = 0,
    std::uint32_t top_k = 0);
void validate_kernel_key(const KernelKey &key);

struct KernelCandidate {
  std::string id;
  std::string executor_key;
  std::uint32_t priority = 0;
  KernelFamily family = KernelFamily::tail_safe;
  std::string format_id;
  std::string layout_id;
  ActivationProfile activation = ActivationProfile::a16;
  KernelDataType input_type = KernelDataType::bf16;
  KernelDataType output_type = KernelDataType::bf16;
  KernelDataType accumulator_type = KernelDataType::fp32;
  KernelGeometry geometry = KernelGeometry::dense;
  std::uint64_t output_features = 0;
  std::uint64_t input_features = 0;
  std::uint32_t min_tokens = 0;
  std::uint32_t max_tokens = 0;
  std::uint32_t experts = 0;
  std::uint32_t top_k = 0;
  KernelFusedOp fused_op = KernelFusedOp::linear;
  KernelAlignmentClass alignment = KernelAlignmentClass::full;
  MetalExecutionRoute required_route = MetalExecutionRoute::classic_msl;
  bool tail_safe = false;
};

void validate_kernel_candidate(const KernelCandidate &candidate);
[[nodiscard]] bool candidate_covers(const KernelCandidate &candidate,
                                    const KernelKey &key);
[[nodiscard]] bool
candidate_route_eligible(const KernelCandidate &candidate,
                         const MetalCapabilityProfile &capabilities);

struct KernelDispatchPlan {
  KernelKey requested;
  KernelCandidate candidate;
  ActivationProfile resolved_activation = ActivationProfile::a16;
  std::vector<std::string> fallback_reasons;

  [[nodiscard]] bool fallback() const { return !fallback_reasons.empty(); }
};

struct DispatchTableStats {
  std::uint64_t candidate_count = 0;
  std::uint64_t installed_candidate_count = 0;
  std::uint64_t planned_candidate_count = 0;
  bool sealed = false;
};

class DispatchTable final {
public:
  DispatchTable();
  DispatchTable(DispatchTable &&) noexcept;
  DispatchTable &operator=(DispatchTable &&) noexcept;
  DispatchTable(const DispatchTable &) = delete;
  DispatchTable &operator=(const DispatchTable &) = delete;
  ~DispatchTable();

  void register_candidate(KernelCandidate candidate);
  void seal(std::span<const std::string> installed_executor_keys);
  [[nodiscard]] bool sealed() const;
  [[nodiscard]] DispatchTableStats stats() const;
  [[nodiscard]] KernelDispatchPlan
  prepare(const KernelKey &key,
          const MetalCapabilityProfile &capabilities) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5
