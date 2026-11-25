#include "omlx_m5/model_config.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace omlx::m5 {

namespace {

using Json = nlohmann::json;

QuantizationSpec parse_quantization_spec(const Json &value,
                                         const QuantizationSpec &fallback) {
  QuantizationSpec result = fallback;
  if (value.contains("bits")) {
    result.bits = value.at("bits").get<std::int32_t>();
  }
  if (value.contains("group_size")) {
    result.group_size = value.at("group_size").get<std::int32_t>();
  }
  if (value.contains("mode")) {
    result.mode = value.at("mode").get<std::string>();
  } else if (result.bits != 4 || result.group_size != 32) {
    // MLX mixed MXFP4 configs omit mode on protected affine overrides.
    result.mode = "affine";
  }
  const bool affine = result.mode == "affine" &&
                      (result.bits == 3 || result.bits == 4 ||
                       result.bits == 5 || result.bits == 6 ||
                       result.bits == 8) &&
                      (result.group_size == 64 || result.group_size == 128);
  const bool mxfp4 =
      result.mode == "mxfp4" && result.bits == 4 && result.group_size == 32;
  if (!affine && !mxfp4) {
    throw std::runtime_error("unsupported Qwen quantization specification");
  }
  return result;
}

std::int32_t required_positive_i32(const Json &value, std::string_view key) {
  const auto it = value.find(key);
  if (it == value.end() || it->is_null()) {
    throw std::runtime_error("missing model config field: " + std::string(key));
  }
  if (!it->is_number_integer()) {
    throw std::runtime_error("invalid integer model config field: " +
                             std::string(key));
  }
  const auto parsed = it->get<std::int64_t>();
  if (parsed <= 0 || parsed > std::numeric_limits<std::int32_t>::max()) {
    throw std::runtime_error("model config field must be positive: " +
                             std::string(key));
  }
  return static_cast<std::int32_t>(parsed);
}

std::int32_t optional_i32(const Json &value, std::string_view key,
                          std::int32_t fallback) {
  const auto it = value.find(key);
  if (it == value.end() || it->is_null()) {
    return fallback;
  }
  if (!it->is_number_integer()) {
    throw std::runtime_error("invalid integer model config field: " +
                             std::string(key));
  }
  const auto parsed = it->get<std::int64_t>();
  if (parsed < std::numeric_limits<std::int32_t>::min() ||
      parsed > std::numeric_limits<std::int32_t>::max()) {
    throw std::runtime_error("integer model config field out of range: " +
                             std::string(key));
  }
  return static_cast<std::int32_t>(parsed);
}

std::int32_t parse_eos_token_id(const Json &text) {
  const auto it = text.find("eos_token_id");
  if (it == text.end() || it->is_null()) {
    return -1;
  }
  if (it->is_number_integer()) {
    const auto parsed = it->get<std::int64_t>();
    if (parsed < std::numeric_limits<std::int32_t>::min() ||
        parsed > std::numeric_limits<std::int32_t>::max()) {
      throw std::runtime_error("eos_token_id is out of range");
    }
    return static_cast<std::int32_t>(parsed);
  }
  if (it->is_array()) {
    if (it->empty()) {
      return -1;
    }
    for (const auto &elem : *it) {
      if (!elem.is_number_integer()) {
        throw std::runtime_error("eos_token_id array must contain integers");
      }
      const auto parsed = elem.get<std::int64_t>();
      if (parsed < std::numeric_limits<std::int32_t>::min() ||
          parsed > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error("eos_token_id array value is out of range");
      }
    }
    return static_cast<std::int32_t>(it->front().get<std::int64_t>());
  }
  throw std::runtime_error(
      "eos_token_id must be an integer, integer array, or null");
}

} // namespace

QuantizationSpec
QwenModelConfig::quantization_for(std::string_view module_name) const {
  const auto it = quantization_overrides.find(std::string(module_name));
  return it == quantization_overrides.end() ? default_quantization : it->second;
}

std::int32_t QwenModelConfig::full_attention_layers() const {
  return static_cast<std::int32_t>(
      std::ranges::count(layer_types, LayerType::full_attention));
}

std::int32_t QwenModelConfig::linear_attention_layers() const {
  return static_cast<std::int32_t>(
      std::ranges::count(layer_types, LayerType::linear_attention));
}

QwenModelConfig
load_qwen_model_config(const std::filesystem::path &config_path) {
  std::ifstream input(config_path);
  if (!input) {
    throw std::runtime_error("unable to open config " + config_path.string());
  }
  Json root;
  try {
    root = Json::parse(input);
  } catch (const Json::exception &error) {
    throw std::runtime_error("invalid model config JSON: " +
                             std::string(error.what()));
  }
  if (!root.is_object() || !root.contains("text_config") ||
      !root.at("text_config").is_object()) {
    throw std::runtime_error("model config has no text_config object");
  }
  const auto &text = root.at("text_config");

  try {
    QwenModelConfig config;
    config.model_type = root.value("model_type", "");
    config.text_model_type = text.value("model_type", "");
    config.dtype = text.value("dtype", "");
    config.hidden_size = required_positive_i32(text, "hidden_size");
    config.num_hidden_layers = required_positive_i32(text, "num_hidden_layers");
    config.intermediate_size = optional_i32(
        text, "intermediate_size",
        optional_i32(text, "moe_intermediate_size", 0));
    if (config.intermediate_size <= 0) {
      throw std::runtime_error(
          "missing model config field: intermediate_size/moe_intermediate_size");
    }
    config.num_experts = optional_i32(text, "num_experts", 0);
    config.num_experts_per_token = optional_i32(text, "num_experts_per_tok", 0);
    config.shared_expert_intermediate_size =
        optional_i32(text, "shared_expert_intermediate_size", 0);
    if (config.num_experts < 0 || config.num_experts_per_token < 0 ||
        config.shared_expert_intermediate_size < 0 ||
        (config.num_experts > 0 &&
         (config.num_experts_per_token <= 0 ||
          config.num_experts_per_token > config.num_experts ||
          config.shared_expert_intermediate_size <= 0))) {
      throw std::runtime_error("invalid MoE model geometry");
    }
    config.vocab_size = required_positive_i32(text, "vocab_size");
    config.num_attention_heads =
        required_positive_i32(text, "num_attention_heads");
    config.num_key_value_heads =
        required_positive_i32(text, "num_key_value_heads");
    if (config.num_attention_heads % config.num_key_value_heads != 0) {
      throw std::runtime_error(
          "num_attention_heads must be divisible by num_key_value_heads");
    }
    config.head_dim = required_positive_i32(text, "head_dim");
    config.linear_num_key_heads =
        required_positive_i32(text, "linear_num_key_heads");
    config.linear_num_value_heads =
        required_positive_i32(text, "linear_num_value_heads");
    if (config.linear_num_value_heads % config.linear_num_key_heads != 0) {
      throw std::runtime_error(
          "linear_num_value_heads must be divisible by linear_num_key_heads");
    }
    config.linear_key_head_dim =
        required_positive_i32(text, "linear_key_head_dim");
    config.linear_value_head_dim =
        required_positive_i32(text, "linear_value_head_dim");
    config.linear_conv_kernel_dim =
        required_positive_i32(text, "linear_conv_kernel_dim");
    config.max_position_embeddings =
        required_positive_i32(text, "max_position_embeddings");
    config.eos_token_id = parse_eos_token_id(text);
    config.rms_norm_eps = text.value("rms_norm_eps", 1.0e-6);
    if (!std::isfinite(config.rms_norm_eps) || !(config.rms_norm_eps > 0.0)) {
      throw std::runtime_error("rms_norm_eps must be positive");
    }
    if (text.contains("rope_parameters")) {
      const auto &rope = text.at("rope_parameters");
      if (!rope.is_object()) {
        throw std::runtime_error("rope_parameters must be an object");
      }
      config.rope_theta = rope.value("rope_theta", 10000.0);
      config.partial_rotary_factor = rope.value("partial_rotary_factor", 1.0);
      config.rope_interleaved = rope.value("mrope_interleaved", true);
    }
    if (!std::isfinite(config.rope_theta) || !(config.rope_theta > 0.0) ||
        !std::isfinite(config.partial_rotary_factor) ||
        !(config.partial_rotary_factor > 0.0) ||
        config.partial_rotary_factor > 1.0) {
      throw std::runtime_error("invalid RoPE parameters");
    }

    const auto layer_types =
        text.at("layer_types").get<std::vector<std::string>>();
    if (layer_types.size() !=
        static_cast<std::size_t>(config.num_hidden_layers)) {
      throw std::runtime_error(
          "layer_types length does not match num_hidden_layers");
    }
    config.layer_types.reserve(layer_types.size());
    for (const auto &layer_type : layer_types) {
      if (layer_type == "linear_attention") {
        config.layer_types.push_back(LayerType::linear_attention);
      } else if (layer_type == "full_attention") {
        config.layer_types.push_back(LayerType::full_attention);
      } else {
        throw std::runtime_error("unsupported Qwen layer type: " + layer_type);
      }
    }

    if (root.contains("quantization") && root.at("quantization").is_object()) {
      const auto &quantization = root.at("quantization");
      config.default_quantization =
          parse_quantization_spec(quantization, QuantizationSpec{});
      for (const auto &[key, value] : quantization.items()) {
        if (!value.is_object()) {
          continue;
        }
        config.quantization_overrides.emplace(
            key, parse_quantization_spec(value, config.default_quantization));
      }
    }
    return config;
  } catch (const Json::exception &error) {
    throw std::runtime_error("invalid model config JSON: " +
                             std::string(error.what()));
  }
}

void validate_qwen38_27b_target(const QwenModelConfig &config) {
  const bool valid =
      config.model_type == "qwen3_5" &&
      config.text_model_type == "qwen3_5_text" && config.dtype == "bfloat16" &&
      config.hidden_size == 5120 && config.num_hidden_layers == 64 &&
      config.intermediate_size == 17408 && config.vocab_size == 248320 &&
      config.num_attention_heads == 24 && config.num_key_value_heads == 4 &&
      config.head_dim == 256 && config.linear_num_key_heads == 16 &&
      config.linear_num_value_heads == 48 &&
      config.linear_key_head_dim == 128 &&
      config.linear_value_head_dim == 128 &&
      config.linear_conv_kernel_dim == 4 &&
      config.full_attention_layers() == 16 &&
      config.linear_attention_layers() == 48 &&
      config.default_quantization.bits == 4 &&
      config.default_quantization.group_size == 64 &&
      config.default_quantization.mode == "affine";
  if (!valid) {
    throw std::runtime_error(
        "model config does not match the fixed Qwen3.8-27B oQ4e target");
  }
}

} // namespace omlx::m5
