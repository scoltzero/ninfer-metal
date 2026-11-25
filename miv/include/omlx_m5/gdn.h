#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/projection.h"

namespace omlx::m5 {

struct GdnDirectWeights {
  const MetalBuffer *conv = nullptr;
  std::uint64_t conv_offset = 0;
  std::uint64_t conv_bytes = 0;
  const MetalBuffer *a_log = nullptr;
  std::uint64_t a_log_offset = 0;
  std::uint64_t a_log_bytes = 0;
  const MetalBuffer *dt_bias = nullptr;
  std::uint64_t dt_bias_offset = 0;
  std::uint64_t dt_bias_bytes = 0;
  const MetalBuffer *norm = nullptr;
  std::uint64_t norm_offset = 0;
  std::uint64_t norm_bytes = 0;
};

struct GdnLayerStateInfo {
  std::uint32_t key_heads = 0;
  std::uint32_t value_heads = 0;
  std::uint32_t key_dim = 0;
  std::uint32_t value_dim = 0;
  std::uint32_t conv_width = 0;
  std::uint32_t conv_kernel = 0;
  std::uint64_t recurrent_state_bytes = 0;
  std::uint64_t conv_history_bytes = 0;
};

class GdnLayerState final {
public:
  GdnLayerState(const MetalContext &context, const QwenModelConfig &config);
  GdnLayerState(GdnLayerState &&) noexcept;
  GdnLayerState &operator=(GdnLayerState &&) noexcept;
  GdnLayerState(const GdnLayerState &) = delete;
  GdnLayerState &operator=(const GdnLayerState &) = delete;
  ~GdnLayerState();

  [[nodiscard]] const GdnLayerStateInfo &info() const;
  void reset();
  [[nodiscard]] const MetalBuffer &recurrent() const;
  [[nodiscard]] const MetalBuffer &history() const;

private:
  class Impl;
  friend MetalSubmission execute_gdn_layer(
      const MetalContext &, const class GdnLayerPlan &,
      const ProjectionParentOutputView &, const ProjectionParentOutputView &,
      const MetalBuffer &, std::uint64_t, const MetalBuffer &, std::uint64_t,
      const GdnDirectWeights &, GdnLayerState &, MetalBuffer &, std::uint64_t);
  std::unique_ptr<Impl> impl_;
};

struct GdnLayerInfo {
  std::string executor_key;
  std::uint32_t tokens = 0;
  std::uint32_t key_heads = 0;
  std::uint32_t value_heads = 0;
  std::uint32_t key_dim = 0;
  std::uint32_t value_dim = 0;
  std::uint32_t conv_width = 0;
  std::uint32_t conv_kernel = 0;
  float epsilon = 0.0F;
  std::uint64_t qkv_bytes = 0;
  std::uint64_t z_bytes = 0;
  std::uint64_t ab_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t dispatch_count = 3;
};

class GdnLayerPlan final {
public:
  GdnLayerPlan();
  GdnLayerPlan(GdnLayerPlan &&) noexcept;
  GdnLayerPlan &operator=(GdnLayerPlan &&) noexcept;
  GdnLayerPlan(const GdnLayerPlan &) = delete;
  GdnLayerPlan &operator=(const GdnLayerPlan &) = delete;
  ~GdnLayerPlan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const GdnLayerInfo &info() const;

private:
  class Impl;
  friend GdnLayerPlan prepare_gdn_layer(const MetalContext &,
                                        const std::filesystem::path &,
                                        const QwenModelConfig &, std::uint32_t);
  friend MetalSubmission execute_gdn_layer(
      const MetalContext &, const GdnLayerPlan &,
      const ProjectionParentOutputView &, const ProjectionParentOutputView &,
      const MetalBuffer &, std::uint64_t, const MetalBuffer &, std::uint64_t,
      const GdnDirectWeights &, GdnLayerState &, MetalBuffer &, std::uint64_t);

  explicit GdnLayerPlan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] GdnLayerPlan
prepare_gdn_layer(const MetalContext &context,
                  const std::filesystem::path &metallib,
                  const QwenModelConfig &config, std::uint32_t tokens);

[[nodiscard]] MetalSubmission execute_gdn_layer(
    const MetalContext &context, const GdnLayerPlan &plan,
    const ProjectionParentOutputView &qkv, const ProjectionParentOutputView &z,
    const MetalBuffer &a, std::uint64_t a_offset, const MetalBuffer &b,
    std::uint64_t b_offset, const GdnDirectWeights &weights,
    GdnLayerState &state, MetalBuffer &output, std::uint64_t output_offset);

} // namespace omlx::m5
