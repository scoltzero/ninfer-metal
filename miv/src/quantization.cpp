#include "omlx_m5/quantization.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace omlx::m5 {

namespace {

std::uint16_t load_u16(const std::byte *data) {
  std::uint16_t value = 0;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

std::uint32_t quantized_code(const std::byte *row, std::uint64_t row_bytes,
                             std::uint64_t column, std::int32_t bits) {
  const auto bit_offset = column * static_cast<std::uint64_t>(bits);
  const auto byte_offset = bit_offset / 8;
  const auto shift = static_cast<std::uint32_t>(bit_offset % 8);
  if (byte_offset >= row_bytes) {
    throw std::out_of_range("quantized column is outside the packed row");
  }
  std::uint32_t packed = static_cast<std::uint8_t>(row[byte_offset]);
  if (byte_offset + 1 < row_bytes) {
    packed |= static_cast<std::uint32_t>(
                  static_cast<std::uint8_t>(row[byte_offset + 1]))
              << 8;
  }
  return (packed >> shift) & ((1U << bits) - 1U);
}

void validate_parameter_tensor(const TensorView &tensor,
                               std::uint64_t out_features, std::uint64_t groups,
                               std::string_view label) {
  if (tensor.info == nullptr || tensor.info->dtype != TensorDType::bf16 ||
      tensor.info->shape !=
          std::vector<std::uint64_t>({out_features, groups})) {
    throw std::runtime_error(std::string(label) +
                             " must be BF16 [out_features, groups]");
  }
}

} // namespace

float bf16_to_float(std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
}

std::uint16_t float_to_bf16(float value) {
  if (std::isnan(value)) {
    return 0x7FC0U;
  }
  auto bits = std::bit_cast<std::uint32_t>(value);
  const auto rounding_bias = 0x7FFFU + ((bits >> 16) & 1U);
  bits += rounding_bias;
  return static_cast<std::uint16_t>(bits >> 16);
}

QuantizedModule bind_quantized_module(const SafeTensorModel &model,
                                      const QwenModelConfig &config,
                                      std::string_view module_name) {
  QuantizedModule module;
  module.name = module_name;
  module.weights = model.tensor(module.name + ".weight");
  module.scales = model.tensor(module.name + ".scales");
  module.biases = model.tensor(module.name + ".biases");
  if (module.weights.info == nullptr ||
      module.weights.info->dtype != TensorDType::u32 ||
      module.weights.info->shape.size() != 2) {
    throw std::runtime_error("quantized weight must be U32 rank 2");
  }

  const auto spec = config.quantization_for(module.name);
  module.bits = spec.bits;
  module.group_size = spec.group_size;
  module.out_features = module.weights.info->shape[0];
  if (module.scales.info == nullptr || module.scales.info->shape.size() != 2 ||
      module.scales.info->shape[0] != module.out_features) {
    throw std::runtime_error(
        "quantized scales must be rank 2 with matching rows");
  }
  const auto groups = module.scales.info->shape[1];
  if (groups > std::numeric_limits<std::uint64_t>::max() /
                   static_cast<std::uint64_t>(module.group_size)) {
    throw std::overflow_error("quantized input width overflow");
  }
  module.in_features = groups * static_cast<std::uint64_t>(module.group_size);
  if (module.in_features == 0 ||
      module.in_features * static_cast<std::uint64_t>(module.bits) % 32 != 0) {
    throw std::runtime_error("quantized input width cannot be packed into U32");
  }
  const auto expected_words =
      module.in_features * static_cast<std::uint64_t>(module.bits) / 32;
  if (module.weights.info->shape[1] != expected_words) {
    throw std::runtime_error(
        "packed weight width disagrees with config quantization");
  }
  const auto inferred_bits =
      module.weights.info->shape[1] * 32 / module.in_features;
  if (inferred_bits != static_cast<std::uint64_t>(module.bits)) {
    throw std::runtime_error(
        "packed weight bit width disagrees with config override");
  }
  validate_parameter_tensor(module.scales, module.out_features, groups,
                            "scales");
  validate_parameter_tensor(module.biases, module.out_features, groups,
                            "biases");
  return module;
}

std::vector<std::string> quantized_module_names(const SafeTensorModel &model) {
  std::vector<std::string> modules;
  constexpr std::string_view suffix = ".weight";
  for (const auto &name : model.tensor_names()) {
    if (!name.ends_with(suffix)) {
      continue;
    }
    const auto &info = model.tensor_info(name);
    if (info.dtype == TensorDType::u32) {
      modules.push_back(name.substr(0, name.size() - suffix.size()));
    }
  }
  return modules;
}

std::vector<float> dequantize_row(const QuantizedModule &module,
                                  std::uint64_t row, std::uint64_t start,
                                  std::uint64_t count) {
  if (row >= module.out_features || start > module.in_features ||
      count > module.in_features - start) {
    throw std::out_of_range(
        "requested dequantized range is outside the tensor");
  }
  const auto row_bytes =
      module.in_features * static_cast<std::uint64_t>(module.bits) / 8;
  const auto *packed_row = module.weights.data + row * row_bytes;
  const auto groups = module.in_features / module.group_size;
  const auto *scale_row =
      module.scales.data + row * groups * sizeof(std::uint16_t);
  const auto *bias_row =
      module.biases.data + row * groups * sizeof(std::uint16_t);

  std::vector<float> values;
  values.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t column = start; column < start + count; ++column) {
    const auto group = column / module.group_size;
    const auto scale =
        bf16_to_float(load_u16(scale_row + group * sizeof(std::uint16_t)));
    const auto bias =
        bf16_to_float(load_u16(bias_row + group * sizeof(std::uint16_t)));
    const auto code =
        quantized_code(packed_row, row_bytes, column, module.bits);
    const auto value = std::fma(scale, static_cast<float>(code), bias);
    if (!std::isfinite(value)) {
      throw std::runtime_error("dequantization produced a non-finite value");
    }
    values.push_back(value);
  }
  return values;
}

std::vector<double> dequantize_row_f64(const QuantizedModule &module,
                                       std::uint64_t row, std::uint64_t start,
                                       std::uint64_t count) {
  if (row >= module.out_features || start > module.in_features ||
      count > module.in_features - start) {
    throw std::out_of_range(
        "requested dequantized range is outside the tensor");
  }
  const auto row_bytes =
      module.in_features * static_cast<std::uint64_t>(module.bits) / 8;
  const auto *packed_row = module.weights.data + row * row_bytes;
  const auto groups = module.in_features / module.group_size;
  const auto *scale_row =
      module.scales.data + row * groups * sizeof(std::uint16_t);
  const auto *bias_row =
      module.biases.data + row * groups * sizeof(std::uint16_t);

  std::vector<double> values;
  values.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t column = start; column < start + count; ++column) {
    const auto group = column / module.group_size;
    const auto scale = static_cast<double>(
        bf16_to_float(load_u16(scale_row + group * sizeof(std::uint16_t))));
    const auto bias = static_cast<double>(
        bf16_to_float(load_u16(bias_row + group * sizeof(std::uint16_t))));
    const auto code =
        quantized_code(packed_row, row_bytes, column, module.bits);
    values.push_back(std::fma(scale, static_cast<double>(code), bias));
  }
  return values;
}

} // namespace omlx::m5
