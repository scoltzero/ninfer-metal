#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/model_config.h"
#include "omlx_m5/safetensors.h"

namespace omlx::m5 {

struct QuantizedModule {
  std::string name;
  std::int32_t bits = 0;
  std::int32_t group_size = 0;
  std::uint64_t out_features = 0;
  std::uint64_t in_features = 0;
  TensorView weights;
  TensorView scales;
  TensorView biases;
};

[[nodiscard]] QuantizedModule
bind_quantized_module(const SafeTensorModel &model,
                      const QwenModelConfig &config,
                      std::string_view module_name);
[[nodiscard]] std::vector<std::string>
quantized_module_names(const SafeTensorModel &model);
[[nodiscard]] std::vector<float> dequantize_row(const QuantizedModule &module,
                                                std::uint64_t row,
                                                std::uint64_t start,
                                                std::uint64_t count);
[[nodiscard]] std::vector<double>
dequantize_row_f64(const QuantizedModule &module, std::uint64_t row,
                   std::uint64_t start, std::uint64_t count);
[[nodiscard]] float bf16_to_float(std::uint16_t value);
[[nodiscard]] std::uint16_t float_to_bf16(float value);

} // namespace omlx::m5
