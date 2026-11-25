#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "omlx_m5/quantization.h"

namespace omlx::m5 {

// Portable D2 reference operators. These routines deliberately favor a clear
// accumulation order and validation over throughput; tuned M5 kernels replace
// them only after their results have passed the same oracles.

[[nodiscard]] std::vector<float>
quantized_qmv(const QuantizedModule &module, std::span<const float> input);
[[nodiscard]] std::vector<float>
quantized_qmm(const QuantizedModule &module, std::span<const float> input,
              std::size_t rows);

[[nodiscard]] std::vector<float>
rms_norm(std::span<const float> input, std::size_t rows, std::size_t width,
         std::span<const float> weight, float epsilon);
void apply_rope(std::span<float> values, std::size_t positions,
                std::size_t heads, std::size_t head_dim,
                std::size_t rotary_dim, std::uint64_t start_position,
                double theta);
[[nodiscard]] std::vector<float>
silu_multiply(std::span<const float> gate,
              std::span<const float> values);

class CounterRng final {
public:
  explicit CounterRng(std::uint64_t seed);

  [[nodiscard]] std::uint64_t bits(std::uint64_t stream,
                                   std::uint64_t counter) const;
  [[nodiscard]] double uniform(std::uint64_t stream,
                               std::uint64_t counter) const;

private:
  std::uint64_t seed_;
};

struct SamplingOptions {
  float temperature = 0.0F;
  std::size_t top_k = 0;
  float top_p = 1.0F;
};

[[nodiscard]] std::int32_t
sample_logits(std::span<const float> logits, const SamplingOptions &options,
              const CounterRng &rng, std::uint64_t stream,
              std::uint64_t absolute_position);

class KvCache final {
public:
  KvCache(std::size_t kv_heads, std::size_t head_dim,
          std::size_t capacity_tokens);

  void append(std::span<const float> keys, std::span<const float> values);
  void truncate(std::size_t tokens);
  void clear();

  [[nodiscard]] std::size_t kv_heads() const;
  [[nodiscard]] std::size_t head_dim() const;
  [[nodiscard]] std::size_t capacity_tokens() const;
  [[nodiscard]] std::size_t tokens() const;
  [[nodiscard]] std::span<const float> key(std::size_t token,
                                           std::size_t head) const;
  [[nodiscard]] std::span<const float> value(std::size_t token,
                                             std::size_t head) const;

private:
  std::size_t kv_heads_;
  std::size_t head_dim_;
  std::size_t capacity_tokens_;
  std::size_t tokens_ = 0;
  std::vector<float> keys_;
  std::vector<float> values_;
};

[[nodiscard]] std::vector<float>
attention_decode(std::span<const float> query, std::size_t query_heads,
                 std::span<const float> key, std::span<const float> value,
                 KvCache &cache);
[[nodiscard]] std::vector<float>
attention_prefill(std::span<const float> queries,
                  std::span<const float> keys,
                  std::span<const float> values, std::size_t positions,
                  std::size_t query_heads, KvCache &cache,
                  std::size_t chunk_size);

struct GdnReferenceGeometry {
  std::size_t key_heads = 0;
  std::size_t value_heads = 0;
  std::size_t key_dim = 0;
  std::size_t value_dim = 0;
};

class GdnReferenceState final {
public:
  explicit GdnReferenceState(GdnReferenceGeometry geometry);
  GdnReferenceState(GdnReferenceGeometry geometry, std::vector<float> values);

  [[nodiscard]] const GdnReferenceGeometry &geometry() const;
  [[nodiscard]] const std::vector<float> &values() const;
  [[nodiscard]] std::vector<float> &mutable_values();
  void clear();

private:
  GdnReferenceGeometry geometry_;
  std::vector<float> values_;
};

[[nodiscard]] std::vector<float>
gdn_step(std::span<const float> query, std::span<const float> key,
         std::span<const float> value, std::span<const float> decay,
         std::span<const float> beta, GdnReferenceState &state);
[[nodiscard]] std::vector<float>
gdn_prefill(std::span<const float> queries, std::span<const float> keys,
            std::span<const float> values, std::span<const float> decays,
            std::span<const float> betas, std::size_t positions,
            GdnReferenceState &state, std::size_t chunk_size);

[[nodiscard]] std::vector<float> causal_depthwise_conv_silu(
    std::span<const float> input, std::size_t positions, std::size_t width,
    std::span<const float> kernel, std::size_t kernel_size,
    std::vector<float> &history);

} // namespace omlx::m5
