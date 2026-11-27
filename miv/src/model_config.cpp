#include "omlx_m5/model_config.h"

#include "omlx_m5/artifact.h"
#include "omlx_m5/materialization.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace omlx::m5 {

namespace {

using Json = nlohmann::json;

const Json *find_nonnull_field(const Json &primary, const Json &secondary,
                               std::string_view key) {
  if (const auto it = primary.find(key);
      it != primary.end() && !it->is_null()) {
    return &(*it);
  }
  if (&primary != &secondary) {
    if (const auto it = secondary.find(key);
        it != secondary.end() && !it->is_null()) {
      return &(*it);
    }
  }
  return nullptr;
}

std::optional<std::string> optional_string(const Json &value,
                                           std::string_view key) {
  const auto it = value.find(key);
  if (it == value.end() || it->is_null()) {
    return std::nullopt;
  }
  if (!it->is_string()) {
    throw std::runtime_error("invalid string model config field: " +
                             std::string(key));
  }
  return it->get<std::string>();
}

QuantizationSpec parse_quantization_spec(const Json &value,
                                         const QuantizationSpec &fallback) {
  QuantizationSpec result = fallback;
  const bool has_bits = value.contains("bits") && !value.at("bits").is_null();
  const bool has_group =
      value.contains("group_size") && !value.at("group_size").is_null();
  const bool has_mode = value.contains("mode") && !value.at("mode").is_null();
  const bool has_quant_method =
      value.contains("quant_method") && !value.at("quant_method").is_null();
  if (has_bits) {
    result.bits = value.at("bits").get<std::int32_t>();
  }
  if (has_group) {
    result.group_size = value.at("group_size").get<std::int32_t>();
  }
  if (has_mode) {
    result.mode = value.at("mode").get<std::string>();
  } else if (has_quant_method) {
    result.mode = value.at("quant_method").get<std::string>();
  } else if (result.bits != 4 || result.group_size != 32) {
    // MLX mixed MXFP4 configs omit mode on protected affine overrides.
    result.mode = "affine";
  }
  if (result.mode == "mxfp4") {
    if (!has_bits) {
      result.bits = 4;
    }
    if (!has_group) {
      result.group_size = 32;
    }
  } else if (result.mode == "affine" && !has_group &&
             result.group_size != 64 && result.group_size != 128) {
    result.group_size = 64;
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

std::int32_t required_positive_i32_fallback(const Json &primary,
                                            const Json &secondary,
                                            std::string_view key) {
  const Json *found = find_nonnull_field(primary, secondary, key);
  if (found == nullptr) {
    throw std::runtime_error("missing model config field: " + std::string(key));
  }
  if (!found->is_number_integer()) {
    throw std::runtime_error("invalid integer model config field: " +
                             std::string(key));
  }
  const auto parsed = found->get<std::int64_t>();
  if (parsed <= 0 || parsed > std::numeric_limits<std::int32_t>::max()) {
    throw std::runtime_error("model config field must be positive: " +
                             std::string(key));
  }
  return static_cast<std::int32_t>(parsed);
}

std::int32_t optional_i32_fallback(const Json &primary, const Json &secondary,
                                   std::string_view key,
                                   std::int32_t fallback) {
  const Json *found = find_nonnull_field(primary, secondary, key);
  if (found == nullptr) {
    return fallback;
  }
  if (!found->is_number_integer()) {
    throw std::runtime_error("invalid integer model config field: " +
                             std::string(key));
  }
  const auto parsed = found->get<std::int64_t>();
  if (parsed < std::numeric_limits<std::int32_t>::min() ||
      parsed > std::numeric_limits<std::int32_t>::max()) {
    throw std::runtime_error("integer model config field out of range: " +
                             std::string(key));
  }
  return static_cast<std::int32_t>(parsed);
}

double optional_double_fallback(const Json &primary, const Json &secondary,
                                std::string_view key, double fallback) {
  const Json *found = find_nonnull_field(primary, secondary, key);
  if (found == nullptr) {
    return fallback;
  }
  if (!found->is_number()) {
    throw std::runtime_error("invalid numeric model config field: " +
                             std::string(key));
  }
  return found->get<double>();
}

bool optional_bool_fallback(const Json &primary, const Json &secondary,
                            std::string_view key, bool fallback) {
  const Json *found = find_nonnull_field(primary, secondary, key);
  if (found == nullptr) {
    return fallback;
  }
  if (!found->is_boolean()) {
    throw std::runtime_error("invalid boolean model config field: " +
                             std::string(key));
  }
  return found->get<bool>();
}

std::int32_t parse_eos_token_id_value(const Json &elem_or_val) {
  if (elem_or_val.is_null()) {
    return -1;
  }
  if (elem_or_val.is_number_integer()) {
    const auto parsed = elem_or_val.get<std::int64_t>();
    if (parsed < std::numeric_limits<std::int32_t>::min() ||
        parsed > std::numeric_limits<std::int32_t>::max()) {
      throw std::runtime_error("eos_token_id is out of range");
    }
    return static_cast<std::int32_t>(parsed);
  }
  if (elem_or_val.is_array()) {
    if (elem_or_val.empty()) {
      return -1;
    }
    for (const auto &elem : elem_or_val) {
      if (!elem.is_number_integer()) {
        throw std::runtime_error("eos_token_id array must contain integers");
      }
      const auto parsed = elem.get<std::int64_t>();
      if (parsed < std::numeric_limits<std::int32_t>::min() ||
          parsed > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error("eos_token_id array value is out of range");
      }
    }
    return static_cast<std::int32_t>(elem_or_val.front().get<std::int64_t>());
  }
  throw std::runtime_error(
      "eos_token_id must be an integer, integer array, or null");
}

std::int32_t parse_eos_token_id(const Json &text, const Json &root) {
  if (const auto it = text.find("eos_token_id"); it != text.end()) {
    return parse_eos_token_id_value(*it);
  }
  if (&text != &root) {
    if (const auto it = root.find("eos_token_id"); it != root.end()) {
      return parse_eos_token_id_value(*it);
    }
  }
  return -1;
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
  if (!root.is_object()) {
    throw std::runtime_error("model config root must be an object");
  }
  const bool has_text_config =
      root.contains("text_config") && !root.at("text_config").is_null();
  if (has_text_config && !root.at("text_config").is_object()) {
    throw std::runtime_error("model config text_config must be an object");
  }
  const auto &text = has_text_config ? root.at("text_config") : root;

  try {
    QwenModelConfig config;
    config.model_type = optional_string(root, "model_type")
                            .value_or(optional_string(text, "model_type").value_or(""));
    config.text_model_type =
        optional_string(text, "model_type").value_or(config.model_type);
    config.dtype = optional_string(text, "dtype")
                       .value_or(optional_string(text, "torch_dtype")
                                     .value_or(optional_string(root, "dtype")
                                                   .value_or(optional_string(root, "torch_dtype")
                                                                 .value_or("bfloat16"))));
    config.hidden_size =
        required_positive_i32_fallback(text, root, "hidden_size");
    config.num_hidden_layers =
        required_positive_i32_fallback(text, root, "num_hidden_layers");
    config.num_experts = optional_i32_fallback(
        text, root, "num_experts",
        optional_i32_fallback(
            text, root, "n_routed_experts",
            optional_i32_fallback(text, root, "num_local_experts", 0)));
    config.num_experts_per_token = optional_i32_fallback(
        text, root, "num_experts_per_tok",
        optional_i32_fallback(text, root, "num_experts_per_token", 0));
    config.shared_expert_intermediate_size =
        optional_i32_fallback(text, root, "shared_expert_intermediate_size", 0);
    if (config.num_experts < 0 || config.num_experts_per_token < 0 ||
        config.shared_expert_intermediate_size < 0) {
      throw std::runtime_error("invalid MoE model geometry");
    }
    if (config.num_experts == 0) {
      config.num_experts_per_token = 0;
      config.intermediate_size = optional_i32_fallback(
          text, root, "intermediate_size",
          optional_i32_fallback(text, root, "moe_intermediate_size", 0));
    } else if (config.num_experts < 2 || config.num_experts > 256 ||
               config.num_experts_per_token < 1 ||
               config.num_experts_per_token > 8 ||
               config.num_experts_per_token > config.num_experts) {
      throw std::runtime_error("invalid MoE model geometry");
    } else {
      config.intermediate_size = optional_i32_fallback(
          text, root, "moe_intermediate_size",
          optional_i32_fallback(text, root, "intermediate_size", 0));
    }
    if (config.intermediate_size <= 0) {
      throw std::runtime_error(
          "missing model config field: intermediate_size/moe_intermediate_size");
    }

    config.vocab_size =
        required_positive_i32_fallback(text, root, "vocab_size");
    config.num_attention_heads =
        required_positive_i32_fallback(text, root, "num_attention_heads");
    config.num_key_value_heads = optional_i32_fallback(
        text, root, "num_key_value_heads", config.num_attention_heads);
    if (config.num_key_value_heads <= 0 ||
        config.num_attention_heads % config.num_key_value_heads != 0) {
      throw std::runtime_error(
          "num_attention_heads must be divisible by num_key_value_heads");
    }
    if (find_nonnull_field(text, root, "head_dim") != nullptr) {
      config.head_dim = required_positive_i32_fallback(text, root, "head_dim");
    } else {
      if (config.hidden_size % config.num_attention_heads != 0) {
        throw std::runtime_error(
            "hidden_size must be divisible by num_attention_heads when head_dim is omitted");
      }
      config.head_dim = config.hidden_size / config.num_attention_heads;
    }

    if (const Json *lt_field = find_nonnull_field(text, root, "layer_types");
        lt_field != nullptr) {
      if (!lt_field->is_array()) {
        throw std::runtime_error("layer_types must be an array");
      }
      const auto layer_types = lt_field->get<std::vector<std::string>>();
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
          throw std::runtime_error("unsupported Qwen layer type: " +
                                   layer_type);
        }
      }
    } else {
      config.layer_types.assign(
          static_cast<std::size_t>(config.num_hidden_layers),
          LayerType::full_attention);
    }

    config.linear_num_key_heads =
        optional_i32_fallback(text, root, "linear_num_key_heads", 0);
    config.linear_num_value_heads =
        optional_i32_fallback(text, root, "linear_num_value_heads", 0);
    config.linear_key_head_dim =
        optional_i32_fallback(text, root, "linear_key_head_dim", 0);
    config.linear_value_head_dim =
        optional_i32_fallback(text, root, "linear_value_head_dim", 0);
    config.linear_conv_kernel_dim =
        optional_i32_fallback(text, root, "linear_conv_kernel_dim", 0);
    if (config.linear_attention_layers() > 0) {
      if (config.linear_num_key_heads <= 0 ||
          config.linear_num_value_heads <= 0 ||
          config.linear_key_head_dim <= 0 ||
          config.linear_value_head_dim <= 0 ||
          config.linear_conv_kernel_dim <= 0) {
        throw std::runtime_error(
            "linear_attention layers require positive linear_* geometry");
      }
      if (config.linear_num_value_heads % config.linear_num_key_heads != 0) {
        throw std::runtime_error(
            "linear_num_value_heads must be divisible by linear_num_key_heads");
      }
    } else if (config.linear_num_key_heads < 0 ||
               config.linear_num_value_heads < 0 ||
               config.linear_key_head_dim < 0 ||
               config.linear_value_head_dim < 0 ||
               config.linear_conv_kernel_dim < 0) {
      throw std::runtime_error("linear_* geometry fields cannot be negative");
    }

    if (find_nonnull_field(text, root, "max_position_embeddings") != nullptr) {
      config.max_position_embeddings =
          required_positive_i32_fallback(text, root, "max_position_embeddings");
    } else {
      config.max_position_embeddings = 32768;
    }
    config.eos_token_id = parse_eos_token_id(text, root);
    config.rms_norm_eps =
        optional_double_fallback(text, root, "rms_norm_eps", 1.0e-6);
    if (!std::isfinite(config.rms_norm_eps) || !(config.rms_norm_eps > 0.0)) {
      throw std::runtime_error("rms_norm_eps must be positive");
    }

    config.rope_theta =
        optional_double_fallback(text, root, "rope_theta", 10000.0);
    config.partial_rotary_factor =
        optional_double_fallback(text, root, "partial_rotary_factor", 1.0);
    if (const Json *rope_ptr = find_nonnull_field(text, root, "rope_parameters");
        rope_ptr != nullptr) {
      if (!rope_ptr->is_object()) {
        throw std::runtime_error("rope_parameters must be an object");
      }
      if (rope_ptr->contains("rope_theta") &&
          !rope_ptr->at("rope_theta").is_null()) {
        config.rope_theta = rope_ptr->at("rope_theta").get<double>();
      }
      if (rope_ptr->contains("partial_rotary_factor") &&
          !rope_ptr->at("partial_rotary_factor").is_null()) {
        config.partial_rotary_factor =
            rope_ptr->at("partial_rotary_factor").get<double>();
      }
      if (rope_ptr->contains("mrope_interleaved") &&
          !rope_ptr->at("mrope_interleaved").is_null()) {
        config.rope_interleaved = rope_ptr->at("mrope_interleaved").get<bool>();
      }
    } else if (const Json *rs_ptr =
                   find_nonnull_field(text, root, "rope_scaling");
               rs_ptr != nullptr && rs_ptr->is_object()) {
      if (rs_ptr->contains("rope_theta") &&
          !rs_ptr->at("rope_theta").is_null()) {
        config.rope_theta = rs_ptr->at("rope_theta").get<double>();
      }
      if (rs_ptr->contains("partial_rotary_factor") &&
          !rs_ptr->at("partial_rotary_factor").is_null()) {
        config.partial_rotary_factor =
            rs_ptr->at("partial_rotary_factor").get<double>();
      }
    }
    if (!std::isfinite(config.rope_theta) || !(config.rope_theta > 0.0) ||
        !std::isfinite(config.partial_rotary_factor) ||
        !(config.partial_rotary_factor > 0.0) ||
        config.partial_rotary_factor > 1.0) {
      throw std::runtime_error("invalid RoPE parameters");
    }

    config.tie_word_embeddings =
        optional_bool_fallback(text, root, "tie_word_embeddings", false);
    if (const Json *gate_f =
            find_nonnull_field(text, root, "has_attn_output_gate");
        gate_f != nullptr) {
      if (!gate_f->is_boolean()) {
        throw std::runtime_error(
            "invalid boolean model config field: has_attn_output_gate");
      }
      config.has_attn_output_gate = gate_f->get<bool>();
      config.has_attn_output_gate_explicit = true;
    } else if (const Json *gate_alt =
                   find_nonnull_field(text, root, "attn_output_gate");
               gate_alt != nullptr) {
      if (!gate_alt->is_boolean()) {
        throw std::runtime_error(
            "invalid boolean model config field: attn_output_gate");
      }
      config.has_attn_output_gate = gate_alt->get<bool>();
      config.has_attn_output_gate_explicit = true;
    }
    config.has_qk_norm =
        optional_bool_fallback(text, root, "has_qk_norm", true);
    config.has_shared_expert_gate =
        optional_bool_fallback(text, root, "has_shared_expert_gate", true);

    const Json *quant_ptr = find_nonnull_field(root, text, "quantization");
    if (quant_ptr == nullptr) {
      quant_ptr = find_nonnull_field(root, text, "quantization_config");
    }
    if (quant_ptr != nullptr && quant_ptr->is_object()) {
      const auto &quantization = *quant_ptr;
      const bool has_top_spec = quantization.contains("bits") ||
                                quantization.contains("group_size") ||
                                quantization.contains("mode") ||
                                quantization.contains("quant_method");
      if (has_top_spec) {
        const bool is_unquant_16 =
            quantization.contains("bits") &&
            quantization.at("bits").is_number_integer() &&
            quantization.at("bits").get<std::int64_t>() >= 16;
        if (!is_unquant_16) {
          config.default_quantization =
              parse_quantization_spec(quantization, QuantizationSpec{});
        }
      }
      for (const auto &[key, value] : quantization.items()) {
        if (!value.is_object()) {
          continue;
        }
        if (!value.contains("bits") && !value.contains("group_size") &&
            !value.contains("mode") && !value.contains("quant_method")) {
          continue;
        }
        if (value.contains("bits") && value.at("bits").is_number_integer() &&
            value.at("bits").get<std::int64_t>() >= 16) {
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

void reconcile_qwen_config_with_artifact(QwenModelConfig &config,
                                         const ArtifactModel &artifact) {
  const auto is_text_obj = [](const ArtifactObject &obj) {
    return classify_model_component(obj.name) == ModelComponent::text;
  };
  const auto find_text_suffix =
      [&](std::string_view suffix) -> const ArtifactObject * {
    for (const auto &obj : artifact.objects()) {
      if (!is_text_obj(obj)) {
        continue;
      }
      if (obj.name == suffix || obj.name.ends_with(suffix) ||
          obj.name == std::string(suffix) + ".weight" ||
          obj.name.ends_with(std::string(suffix) + ".weight")) {
        return &obj;
      }
    }
    return nullptr;
  };

  for (std::size_t layer = 0; layer < config.layer_types.size(); ++layer) {
    if (config.layer_types[layer] != LayerType::full_attention) {
      continue;
    }
    const std::string attn_prefix =
        "layers." + std::to_string(layer) + ".self_attn.";
    const ArtifactObject *q_obj = find_text_suffix(attn_prefix + "q_proj");
    if (q_obj != nullptr) {
      if (q_obj->shape.size() != 2) {
        throw std::invalid_argument("full-attention q_proj must be rank-2: " +
                                    q_obj->name);
      }
      const std::uint64_t q_base =
          static_cast<std::uint64_t>(config.num_attention_heads) *
          static_cast<std::uint64_t>(config.head_dim);
      if (q_obj->shape[0] == q_base) {
        config.has_attn_output_gate = false;
      } else if (q_obj->shape[0] == 2ULL * q_base) {
        config.has_attn_output_gate = true;
      } else {
        throw std::invalid_argument(
            "unexpected self_attn.q_proj out_features=" +
            std::to_string(q_obj->shape[0]) + " in " + q_obj->name +
            " (expected " + std::to_string(q_base) + " or " +
            std::to_string(2ULL * q_base) + ")");
      }
      const bool has_qn = find_text_suffix(attn_prefix + "q_norm") != nullptr;
      const bool has_kn = find_text_suffix(attn_prefix + "k_norm") != nullptr;
      if (has_qn != has_kn) {
        throw std::invalid_argument(
            "asymmetric self_attn q_norm/k_norm in artifact layer " +
            std::to_string(layer));
      }
      config.has_qk_norm = has_qn;
      break;
    }
  }

  if (config.num_experts > 0) {
    const ArtifactObject *exp_gate =
        find_text_suffix(".mlp.switch_mlp.gate_proj");
    if (exp_gate == nullptr) {
      exp_gate = find_text_suffix("layers.0.mlp.switch_mlp.gate_proj");
    }
    if (exp_gate != nullptr && exp_gate->shape.size() == 3 &&
        exp_gate->shape[1] > 0) {
      config.intermediate_size =
          static_cast<std::int32_t>(exp_gate->shape[1]);
    }
    const ArtifactObject *sh_gate_proj =
        find_text_suffix(".mlp.shared_expert.gate_proj");
    if (sh_gate_proj == nullptr) {
      sh_gate_proj = find_text_suffix("layers.0.mlp.shared_expert.gate_proj");
    }
    if (sh_gate_proj == nullptr) {
      config.shared_expert_intermediate_size = 0;
      config.has_shared_expert_gate = false;
    } else {
      if (config.shared_expert_intermediate_size <= 0 &&
          !sh_gate_proj->shape.empty()) {
        config.shared_expert_intermediate_size =
            static_cast<std::int32_t>(sh_gate_proj->shape[0]);
      }
      const bool has_seg =
          find_text_suffix(".mlp.shared_expert_gate") != nullptr ||
          find_text_suffix("layers.0.mlp.shared_expert_gate") != nullptr;
      config.has_shared_expert_gate = has_seg;
    }
  }

  bool has_dedicated_lm_head = false;
  for (const auto &obj : artifact.objects()) {
    if (is_text_obj(obj) &&
        (obj.name == "lm_head" || obj.name == "lm_head.weight" ||
         obj.name.ends_with(".lm_head") ||
         obj.name.ends_with(".lm_head.weight"))) {
      has_dedicated_lm_head = true;
      break;
    }
  }
  if (!has_dedicated_lm_head) {
    config.tie_word_embeddings = true;
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
