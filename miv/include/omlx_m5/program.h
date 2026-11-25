#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "omlx_m5/model_config.h"
#include "omlx_m5/reference.h"

namespace omlx::m5 {

struct ReferenceProgramConfig {
  std::size_t vocab_size = 0;
  std::size_t hidden_size = 0;
  std::size_t intermediate_size = 0;
  std::size_t attention_heads = 0;
  std::size_t kv_heads = 0;
  std::size_t head_dim = 0;
  std::size_t linear_key_heads = 0;
  std::size_t linear_value_heads = 0;
  std::size_t linear_key_dim = 0;
  std::size_t linear_value_dim = 0;
  std::size_t linear_conv_kernel = 0;
  std::size_t max_context = 0;
  std::size_t rotary_dim = 0;
  float rms_norm_eps = 1.0e-6F;
  double rope_theta = 10000.0;
  std::int32_t eos_token_id = -1;
  std::vector<LayerType> layer_types;

  [[nodiscard]] static ReferenceProgramConfig
  from_qwen(const QwenModelConfig &config, std::size_t max_context);
};

struct ReferenceRunResult {
  std::vector<std::int32_t> tokens;
  bool reached_eos = false;
  bool cancelled = false;
};

class ReferenceProgram final {
public:
  static ReferenceProgram create_deterministic(ReferenceProgramConfig config,
                                                std::uint64_t weight_seed);

  ReferenceProgram(ReferenceProgram &&) noexcept;
  ReferenceProgram &operator=(ReferenceProgram &&) noexcept;
  ReferenceProgram(const ReferenceProgram &) = delete;
  ReferenceProgram &operator=(const ReferenceProgram &) = delete;
  ~ReferenceProgram();

  [[nodiscard]] const ReferenceProgramConfig &config() const;
  [[nodiscard]] std::size_t position() const;
  [[nodiscard]] std::uint64_t parameter_count() const;
  void reset();

  [[nodiscard]] std::vector<float> step(std::int32_t token);
  [[nodiscard]] std::vector<float>
  prefill(std::span<const std::int32_t> tokens, std::size_t chunk_size);
  [[nodiscard]] ReferenceRunResult
  generate(std::span<const std::int32_t> prompt, std::size_t prefill_chunk,
           std::size_t max_new_tokens, const SamplingOptions &sampling,
           std::uint64_t sampling_seed,
           const std::function<bool()> &cancelled = {});

private:
  class Impl;

  explicit ReferenceProgram(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5
