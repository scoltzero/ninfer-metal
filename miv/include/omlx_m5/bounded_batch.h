#pragma once
#include "omlx_m5/qwen_forward.h"
#include "omlx_m5/position_sampling.h"
#include <memory>
#include <span>
namespace omlx::m5 {
struct BatchRow { std::uint32_t slot=0,width=1,valid=1; };
class CoupledBatchSampler final {
public:
 CoupledBatchSampler(const MetalContext &,std::span<QwenForward *const>,const std::filesystem::path &);
 ~CoupledBatchSampler();
 [[nodiscard]] PositionSamplingResult sample(std::span<const BatchRow>,std::span<const PositionSamplingRow>,bool icb=true);
 [[nodiscard]] bool can_sample_contiguous() const noexcept;
 [[nodiscard]] PositionSamplingResult sample_contiguous(std::span<const PositionSamplingRow>,bool icb=true);
 [[nodiscard]] bool uses_icb() const;
 [[nodiscard]] std::vector<std::uint64_t> resource_addresses() const;
private: struct Impl;std::unique_ptr<Impl> impl_;
};
// Explicit diagnostic head rerun over actual final-normalized per-slot rows.
// It consumes shared weights once in a true M=sum(W_i) Linear invocation, then
// scatters results to real AR/verify consumers. The grouped head remains part
// of this prototype's measured cost; this is not a production speed claim.
class SharedHeadProjection final {
public:
 SharedHeadProjection(const MetalContext &,std::span<QwenForward *const>,const std::filesystem::path &);
 ~SharedHeadProjection();
 [[nodiscard]] MetalSubmission project(std::span<const BatchRow>,bool icb=true);
 [[nodiscard]] std::uint32_t last_rows() const;
 [[nodiscard]] QwenProfileWeightRoute execution_route() const;
 [[nodiscard]] std::vector<std::uint64_t> resource_addresses() const;
private:struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace omlx::m5
