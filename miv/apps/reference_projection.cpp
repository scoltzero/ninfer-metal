#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/model_config.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/reference.h"
#include "omlx_m5/safetensors.h"

namespace {

std::size_t parse_size(std::string_view value, const char *name) {
  std::size_t parsed = 0;
  const auto result = std::stoull(std::string(value), &parsed);
  if (parsed != value.size() || result == 0) {
    throw std::invalid_argument(std::string(name) +
                                " must be a positive integer");
  }
  return static_cast<std::size_t>(result);
}

std::vector<float> make_input(std::size_t rows, std::size_t width) {
  std::vector<float> input(rows * width);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < width; ++column) {
      const auto centered = static_cast<std::int32_t>(
                                (row * 17 + column * 13) % 257) -
                            128;
      input[row * width + column] =
          static_cast<float>(centered) / 1024.0F;
    }
  }
  return input;
}

} // namespace

int main(int argc, const char *const *argv) {
  try {
    if (argc != 4) {
      throw std::invalid_argument(
          "usage: omlx_m5_reference_projection MODEL_DIR MODULE ROWS");
    }
    const std::filesystem::path model_dir = argv[1];
    const std::string module_name = argv[2];
    const auto rows = parse_size(argv[3], "ROWS");
    if (rows > 8) {
      throw std::invalid_argument("ROWS must be within the D2 M=1..8 range");
    }
    const auto config =
        omlx::m5::load_qwen_model_config(model_dir / "config.json");
    const auto model = omlx::m5::SafeTensorModel::open(model_dir);
    const auto module =
        omlx::m5::bind_quantized_module(model, config, module_name);
    const auto input = make_input(rows, module.in_features);
    const auto output = omlx::m5::quantized_qmm(module, input, rows);
    const nlohmann::json result = {
        {"schema_version", 1},
        {"scope", "M4 correctness"},
        {"module", module_name},
        {"bits", module.bits},
        {"group_size", module.group_size},
        {"rows", rows},
        {"input_width", module.in_features},
        {"output_width", module.out_features},
        {"output", output},
    };
    std::cout << result.dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "omlx_m5_reference_projection: " << error.what() << '\n';
    return 1;
  }
}
