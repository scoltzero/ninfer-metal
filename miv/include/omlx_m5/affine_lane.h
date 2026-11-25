#pragma once

#include "omlx_m5/materialization.h"
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace omlx::m5 {
struct AffineLaneGeometry {
  std::uint32_t n=0,k=0,padded_n=0,groups=0,split_k=0;
  std::uint64_t code_bytes=0,scale_bias_bytes=0;
  bool explicit_split_k=false;
  std::string layout_id="q4_tile32_group64_sbt_pair_bf16_v1";
};
[[nodiscard]] std::uint32_t affine_lane_split_k(std::uint32_t n,std::uint32_t k);
[[nodiscard]] bool affine_lane_fusion_compatible(const AffineLaneGeometry &,
                                                 const AffineLaneGeometry &);
// Eligibility for an explicit diagnostic only; not NAX/performance proof.
[[nodiscard]] bool affine_lane_m5_diagnostic_eligible(const MetalContext &);

// Separate immutable prepared representation; never changes the source Model.
class AffineLaneWeight final {
public:
  AffineLaneWeight(const MetalContext &,const MaterializedWeightView &,
                   std::optional<std::uint32_t> fixed_split_k={});
  ~AffineLaneWeight();
  AffineLaneWeight(AffineLaneWeight &&) noexcept;
  AffineLaneWeight &operator=(AffineLaneWeight &&) noexcept;
  [[nodiscard]] const AffineLaneGeometry &geometry() const;
  [[nodiscard]] const MetalBuffer &codes() const;
  [[nodiscard]] const MetalBuffer &scale_bias() const;
  [[nodiscard]] std::uint64_t preparation_readback_bytes() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
enum class AffineLaneExecution { PortableReference, M5TensorDiagnostic };

// Fixed max_rows <=128, stable input/output/weights/workspace and parameter
// addresses. Per-call updates are rows/MP only; no repack/compile/allocation.
class AffineLaneProjection final {
public:
  AffineLaneProjection(const MetalContext &,const AffineLaneWeight &,
                       const MetalBuffer &input,MetalBuffer &output,
                       const std::filesystem::path &libraries,
                       AffineLaneExecution,std::uint32_t max_rows=128);
  ~AffineLaneProjection();
  [[nodiscard]] MetalSubmission run(std::uint32_t rows,bool use_icb=true);
  [[nodiscard]] std::string numeric_contract_id() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace omlx::m5
