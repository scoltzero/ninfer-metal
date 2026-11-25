#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "omlx_m5/model_config.h"
#include "omlx_m5/program.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/safetensors.h"

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void validate(const std::filesystem::path &model_dir,
              std::string_view revision, std::int32_t hidden,
              std::int32_t layers, std::size_t tensors,
              std::size_t quantized_modules) {
  const auto config =
      omlx::m5::load_qwen_model_config(model_dir / "config.json");
  expect(config.hidden_size == hidden, "small-model hidden size mismatch");
  expect(config.num_hidden_layers == layers,
         "small-model layer count mismatch");
  expect(config.full_attention_layers() == layers / 4,
         "small-model full-attention count mismatch");
  const auto program =
      omlx::m5::ReferenceProgramConfig::from_qwen(config, 512);
  expect(program.layer_types.size() == static_cast<std::size_t>(layers),
         "Program lost small-model layer schedule");
  expect(program.rotary_dim == 64,
         "Program derived the wrong Qwen3.5 partial RoPE width");

  const auto model = omlx::m5::SafeTensorModel::open(model_dir, revision);
  expect(model.tensor_count() == tensors, "small-model tensor count mismatch");
  expect(model.declared_total_size() == model.total_tensor_bytes(),
         "standard safetensors total_size convention was not preserved");
  const auto names = omlx::m5::quantized_module_names(model);
  expect(names.size() == quantized_modules,
         "small-model quantized module count mismatch");
  for (const auto &name : names) {
    const auto module = omlx::m5::bind_quantized_module(model, config, name);
    expect(module.bits == 4 && module.group_size == 64,
           "small-model q4 binding mismatch");
  }
}

} // namespace

int main(int argc, const char *const *argv) {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "usage: d2_model_binding_tests MODEL_0_8B MODEL_4B");
    }
    validate(argv[1], "5d894f8cc4ef3e6c88537bf3746ed262f549da6a", 1024,
             24, 847, 187);
    validate(argv[2], "32f3e8ecf65426fc3306969496342d504bfa13f3", 2560,
             32, 1221, 249);
    std::cout << "D2 small-model bindings passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "D2 small-model binding test failed: " << error.what() << '\n';
    return 1;
  }
}
