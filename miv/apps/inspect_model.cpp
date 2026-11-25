#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "omlx_m5/memory_plan.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/safetensors.h"

namespace {

using Json = nlohmann::json;

struct Options {
  std::filesystem::path model_dir;
  std::string expected_revision;
  std::string sample_module;
  std::uint64_t sample_row = 0;
  std::uint64_t sample_start = 0;
  std::uint64_t sample_count = 0;
  std::uint64_t max_context = 32768;
};

std::uint64_t parse_u64(std::string_view text, std::string_view name) {
  std::size_t parsed = 0;
  const auto value = std::stoull(std::string(text), &parsed);
  if (parsed != text.size()) {
    throw std::invalid_argument(std::string(name) + " must be an integer");
  }
  return value;
}

Options parse_args(int argc, const char *const *argv) {
  if (argc < 2) {
    throw std::invalid_argument(
        "usage: omlx_m5_inspect_model MODEL_DIR [options]");
  }
  Options options;
  options.model_dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--expected-revision" && i + 1 < argc) {
      options.expected_revision = argv[++i];
    } else if (arg == "--sample-module" && i + 1 < argc) {
      options.sample_module = argv[++i];
    } else if (arg == "--sample-row" && i + 1 < argc) {
      options.sample_row = parse_u64(argv[++i], "--sample-row");
    } else if (arg == "--sample-start" && i + 1 < argc) {
      options.sample_start = parse_u64(argv[++i], "--sample-start");
    } else if (arg == "--sample-count" && i + 1 < argc) {
      options.sample_count = parse_u64(argv[++i], "--sample-count");
    } else if (arg == "--max-context" && i + 1 < argc) {
      options.max_context = parse_u64(argv[++i], "--max-context");
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  std::string(arg));
    }
  }
  return options;
}

} // namespace

int main(int argc, const char *const *argv) {
  try {
    const auto options = parse_args(argc, argv);
    const auto config =
        omlx::m5::load_qwen_model_config(options.model_dir / "config.json");
    omlx::m5::validate_qwen38_27b_target(config);
    const auto model = omlx::m5::SafeTensorModel::open(
        options.model_dir, options.expected_revision);
    const auto modules = omlx::m5::quantized_module_names(model);
    std::map<std::int32_t, std::size_t> modules_by_bits;
    for (const auto &name : modules) {
      const auto binding = omlx::m5::bind_quantized_module(model, config, name);
      ++modules_by_bits[binding.bits];
    }
    Json modules_by_bits_json = Json::object();
    for (const auto &[bits, count] : modules_by_bits) {
      modules_by_bits_json[std::to_string(bits)] = count;
    }
    const auto memory =
        omlx::m5::plan_sequence_memory(config, options.max_context);

    Json output = {
        {"schema_version", 1},
        {"model_dir", model.model_dir().string()},
        {"revision", model.revision()},
        {"shard_count", model.shard_count()},
        {"tensor_count", model.tensor_count()},
        {"total_tensor_bytes", model.total_tensor_bytes()},
        {"declared_total_size", model.declared_total_size()},
        {"quantized_module_count", modules.size()},
        {"quantized_modules_by_bits", modules_by_bits_json},
        {"geometry",
         {
             {"hidden_size", config.hidden_size},
             {"layers", config.num_hidden_layers},
             {"full_attention_layers", config.full_attention_layers()},
             {"linear_attention_layers", config.linear_attention_layers()},
             {"head_dim", config.head_dim},
             {"vocab_size", config.vocab_size},
         }},
        {"sequence_memory",
         {
             {"max_context", memory.max_context},
             {"kv_page_tokens", memory.kv_page_tokens},
             {"kv_page_count", memory.kv_page_count},
             {"kv_bytes_per_token", memory.kv_bytes_per_token},
             {"kv_total_bytes", memory.kv_total_bytes},
             {"gdn_state_bytes", memory.gdn_state_bytes},
             {"gdn_conv_history_bytes", memory.gdn_conv_history_bytes},
             {"total_bytes", memory.total_bytes},
         }},
    };

    if (!options.sample_module.empty()) {
      if (options.sample_count == 0) {
        throw std::invalid_argument(
            "--sample-count must be positive when sampling a module");
      }
      const auto module =
          omlx::m5::bind_quantized_module(model, config, options.sample_module);
      output["sample"] = {
          {"module", module.name},
          {"bits", module.bits},
          {"group_size", module.group_size},
          {"out_features", module.out_features},
          {"in_features", module.in_features},
          {"row", options.sample_row},
          {"start", options.sample_start},
          {"values", omlx::m5::dequantize_row(module, options.sample_row,
                                              options.sample_start,
                                              options.sample_count)},
      };
    }
    std::cout << output.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "omlx_m5_inspect_model: " << error.what() << '\n';
    return 1;
  }
}
