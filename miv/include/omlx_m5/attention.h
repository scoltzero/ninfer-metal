#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/projection.h"

namespace omlx::m5 {
enum class AttentionNumericPolicy { Legacy, RowExactV1 };

struct AttentionPostprocessInfo {
  std::string executor_key;
  std::uint32_t tokens = 0;
  std::uint32_t query_heads = 0;
  std::uint32_t key_value_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t rotary_dim = 0;
  float epsilon = 0.0F;
  double theta = 0.0;
  std::uint64_t query_span_bytes = 0;
  std::uint64_t key_span_bytes = 0;
  std::uint64_t norm_weight_bytes = 0;
  std::uint64_t dispatch_count = 1;
  std::uint64_t workspace_bytes = 0;
};

class AttentionPostprocessPlan final {
public:
  AttentionPostprocessPlan();
  AttentionPostprocessPlan(AttentionPostprocessPlan &&) noexcept;
  AttentionPostprocessPlan &operator=(AttentionPostprocessPlan &&) noexcept;
  AttentionPostprocessPlan(const AttentionPostprocessPlan &) = delete;
  AttentionPostprocessPlan &
  operator=(const AttentionPostprocessPlan &) = delete;
  ~AttentionPostprocessPlan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const AttentionPostprocessInfo &info() const;

private:
  class Impl;
  friend AttentionPostprocessPlan
  prepare_attention_postprocess(const MetalContext &,
                                const std::filesystem::path &,
                                const QwenModelConfig &, std::uint32_t);
  friend MetalSubmission execute_attention_postprocess(
      const MetalContext &, const AttentionPostprocessPlan &,
      const ProjectionSplitOutputView &, const ProjectionSplitOutputView &,
      const MetalBuffer &, std::uint64_t, const MetalBuffer &, std::uint64_t,
      std::uint64_t);

  explicit AttentionPostprocessPlan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] AttentionPostprocessPlan prepare_attention_postprocess(
    const MetalContext &context, const std::filesystem::path &metallib,
    const QwenModelConfig &config, std::uint32_t tokens);

// Applies per-head RMSNorm followed by the Qwen/MLX half-split RoPE layout.
// Query and key are updated in place; the interleaved query gate blocks are not
// touched.
[[nodiscard]] MetalSubmission execute_attention_postprocess(
    const MetalContext &context, const AttentionPostprocessPlan &plan,
    const ProjectionSplitOutputView &query,
    const ProjectionSplitOutputView &key, const MetalBuffer &query_norm,
    std::uint64_t query_norm_offset, const MetalBuffer &key_norm,
    std::uint64_t key_norm_offset, std::uint64_t start_position);

struct PagedKvCacheInfo {
  std::uint32_t key_value_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t capacity_tokens = 0;
  std::uint32_t page_tokens = 0;
  std::uint32_t page_count = 0;
  std::uint32_t physical_page_count = 0;
  std::uint64_t bytes_per_token = 0;
  std::uint64_t bytes_per_page = 0;
  std::uint64_t key_bytes = 0;
  std::uint64_t value_bytes = 0;
};

class PagedKvCache final {
public:
  PagedKvCache(const MetalContext &context, std::uint32_t key_value_heads,
               std::uint32_t head_dim, std::uint32_t capacity_tokens,
               std::uint32_t page_tokens = 64);
  PagedKvCache(PagedKvCache &&) noexcept;
  PagedKvCache &operator=(PagedKvCache &&) noexcept;
  PagedKvCache(const PagedKvCache &) = delete;
  PagedKvCache &operator=(const PagedKvCache &) = delete;
  ~PagedKvCache();

  [[nodiscard]] const PagedKvCacheInfo &info() const;
  [[nodiscard]] std::uint32_t tokens() const;
  void truncate(std::uint32_t tokens);
  void reset();
  [[nodiscard]] const MetalBuffer &keys() const;
  [[nodiscard]] const MetalBuffer &values() const;
  [[nodiscard]] const MetalBuffer &page_table() const;

private:
  class Impl;
  friend MetalSubmission execute_paged_attention(
      const MetalContext &, const class PagedAttentionPlan &,
      const ProjectionSplitOutputView &, const ProjectionSplitOutputView &,
      const ProjectionSplitOutputView &, const ProjectionSplitOutputView &,
      PagedKvCache &, MetalBuffer &, std::uint64_t);

  std::unique_ptr<Impl> impl_;
};

struct PagedAttentionInfo {
  std::string executor_key;
  std::uint32_t tokens = 0;
  std::uint32_t query_heads = 0;
  std::uint32_t key_value_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint64_t query_span_bytes = 0;
  std::uint64_t key_span_bytes = 0;
  std::uint64_t value_span_bytes = 0;
  std::uint64_t gate_span_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t dispatch_count = 2;
  std::uint64_t workspace_bytes = 0;
};

class PagedAttentionPlan final {
public:
  PagedAttentionPlan();
  PagedAttentionPlan(PagedAttentionPlan &&) noexcept;
  PagedAttentionPlan &operator=(PagedAttentionPlan &&) noexcept;
  PagedAttentionPlan(const PagedAttentionPlan &) = delete;
  PagedAttentionPlan &operator=(const PagedAttentionPlan &) = delete;
  ~PagedAttentionPlan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const PagedAttentionInfo &info() const;

private:
  class Impl;
  friend PagedAttentionPlan
  prepare_paged_attention(const MetalContext &, const std::filesystem::path &,
                          const QwenModelConfig &, std::uint32_t,
                          const PagedKvCacheInfo &);
  friend MetalSubmission execute_paged_attention(
      const MetalContext &, const PagedAttentionPlan &,
      const ProjectionSplitOutputView &, const ProjectionSplitOutputView &,
      const ProjectionSplitOutputView &, const ProjectionSplitOutputView &,
      PagedKvCache &, MetalBuffer &, std::uint64_t);

  explicit PagedAttentionPlan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] PagedAttentionPlan
prepare_paged_attention(const MetalContext &context,
                        const std::filesystem::path &metallib,
                        const QwenModelConfig &config, std::uint32_t tokens,
                        const PagedKvCacheInfo &cache);

// Appends key/value to the paged cache and computes causal attention. The
// sigmoid query gate is fused into the output write.
[[nodiscard]] MetalSubmission execute_paged_attention(
    const MetalContext &context, const PagedAttentionPlan &plan,
    const ProjectionSplitOutputView &query,
    const ProjectionSplitOutputView &key,
    const ProjectionSplitOutputView &value,
    const ProjectionSplitOutputView &gate, PagedKvCache &cache,
    MetalBuffer &output, std::uint64_t output_offset);

} // namespace omlx::m5
