#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <span>
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

std::vector<float> tensor_floats(const omlx::m5::TensorView &tensor) {
  if (tensor.info == nullptr ||
      tensor.info->dtype != omlx::m5::TensorDType::bf16 ||
      tensor.nbytes % sizeof(std::uint16_t) != 0) {
    throw std::runtime_error("reference layer expects a BF16 tensor");
  }
  std::vector<float> result(tensor.nbytes / sizeof(std::uint16_t));
  for (std::size_t index = 0; index < result.size(); ++index) {
    std::uint16_t encoded = 0;
    std::memcpy(&encoded,
                tensor.data + index * sizeof(std::uint16_t),
                sizeof(encoded));
    result[index] = omlx::m5::bf16_to_float(encoded);
  }
  return result;
}

std::vector<float> deterministic_hidden(std::size_t width) {
  std::vector<float> result(width);
  for (std::size_t index = 0; index < width; ++index) {
    const auto centered = static_cast<std::int32_t>((index * 13) % 257) - 128;
    result[index] = static_cast<float>(centered) / 1024.0F;
  }
  return result;
}

void add_in_place(std::vector<float> &target,
                  std::span<const float> value) {
  if (target.size() != value.size()) {
    throw std::invalid_argument("reference layer residual geometry mismatch");
  }
  for (std::size_t index = 0; index < target.size(); ++index) {
    target[index] += value[index];
  }
}

std::vector<float> qlinear(const omlx::m5::SafeTensorModel &model,
                           const omlx::m5::QwenModelConfig &config,
                           std::string_view name,
                           std::span<const float> input) {
  return omlx::m5::quantized_qmv(
      omlx::m5::bind_quantized_module(model, config, name), input);
}

std::vector<float> mlp(const omlx::m5::SafeTensorModel &model,
                       const omlx::m5::QwenModelConfig &config,
                       std::string_view prefix,
                       std::span<const float> input) {
  const auto gate = qlinear(model, config, std::string(prefix) + ".gate_proj",
                            input);
  const auto up = qlinear(model, config, std::string(prefix) + ".up_proj",
                          input);
  const auto activated = omlx::m5::silu_multiply(gate, up);
  return qlinear(model, config, std::string(prefix) + ".down_proj",
                 activated);
}

std::vector<float>
run_attention(const omlx::m5::SafeTensorModel &model,
              const omlx::m5::QwenModelConfig &config,
              std::span<const float> input, std::size_t layer_index) {
  const auto prefix = "language_model.model.layers." +
                      std::to_string(layer_index);
  auto hidden = std::vector<float>(input.begin(), input.end());
  const auto input_norm_weight =
      tensor_floats(model.tensor(prefix + ".input_layernorm.weight"));
  auto normalized = omlx::m5::rms_norm(
      hidden, 1, config.hidden_size, input_norm_weight,
      static_cast<float>(config.rms_norm_eps));
  const auto attention_prefix = prefix + ".self_attn";
  const auto projected_q = qlinear(model, config,
                                   attention_prefix + ".q_proj", normalized);
  auto key =
      qlinear(model, config, attention_prefix + ".k_proj", normalized);
  const auto value =
      qlinear(model, config, attention_prefix + ".v_proj", normalized);
  const auto query_width = static_cast<std::size_t>(
      config.num_attention_heads * config.head_dim);
  std::vector<float> query(query_width);
  std::vector<float> gate(query_width);
  const auto per_head = static_cast<std::size_t>(config.head_dim) * 2;
  for (std::size_t head = 0;
       head < static_cast<std::size_t>(config.num_attention_heads); ++head) {
    const auto source = head * per_head;
    const auto destination = head * config.head_dim;
    std::ranges::copy(
        std::span<const float>(projected_q).subspan(source, config.head_dim),
        query.begin() + static_cast<std::ptrdiff_t>(destination));
    std::ranges::copy(std::span<const float>(projected_q)
                          .subspan(source + config.head_dim, config.head_dim),
                      gate.begin() +
                          static_cast<std::ptrdiff_t>(destination));
  }
  const auto query_norm_weight =
      tensor_floats(model.tensor(attention_prefix + ".q_norm.weight"));
  const auto key_norm_weight =
      tensor_floats(model.tensor(attention_prefix + ".k_norm.weight"));
  query = omlx::m5::rms_norm(query, config.num_attention_heads,
                             config.head_dim, query_norm_weight,
                             static_cast<float>(config.rms_norm_eps));
  key = omlx::m5::rms_norm(key, config.num_key_value_heads, config.head_dim,
                           key_norm_weight,
                           static_cast<float>(config.rms_norm_eps));
  const auto rotary_dim = static_cast<std::size_t>(
      config.head_dim * config.partial_rotary_factor);
  omlx::m5::apply_rope(query, 1, config.num_attention_heads, config.head_dim,
                       rotary_dim, 0, config.rope_theta);
  omlx::m5::apply_rope(key, 1, config.num_key_value_heads, config.head_dim,
                       rotary_dim, 0, config.rope_theta);
  omlx::m5::KvCache cache(config.num_key_value_heads, config.head_dim, 1);
  auto attended = omlx::m5::attention_decode(
      query, config.num_attention_heads, key, value, cache);
  for (std::size_t index = 0; index < attended.size(); ++index) {
    attended[index] *= 1.0F / (1.0F + std::exp(-gate[index]));
  }
  add_in_place(hidden, qlinear(model, config, attention_prefix + ".o_proj",
                               attended));
  const auto post_norm_weight = tensor_floats(
      model.tensor(prefix + ".post_attention_layernorm.weight"));
  normalized = omlx::m5::rms_norm(
      hidden, 1, config.hidden_size, post_norm_weight,
      static_cast<float>(config.rms_norm_eps));
  add_in_place(hidden, mlp(model, config, prefix + ".mlp", normalized));
  return hidden;
}

std::vector<float> run_gdn(const omlx::m5::SafeTensorModel &model,
                           const omlx::m5::QwenModelConfig &config,
                           std::span<const float> input,
                           std::size_t layer_index) {
  const auto prefix = "language_model.model.layers." +
                      std::to_string(layer_index);
  auto hidden = std::vector<float>(input.begin(), input.end());
  const auto input_norm_weight =
      tensor_floats(model.tensor(prefix + ".input_layernorm.weight"));
  auto normalized = omlx::m5::rms_norm(
      hidden, 1, config.hidden_size, input_norm_weight,
      static_cast<float>(config.rms_norm_eps));
  const auto gdn_prefix = prefix + ".linear_attn";
  const auto projected =
      qlinear(model, config, gdn_prefix + ".in_proj_qkv", normalized);
  const auto conv_weight =
      tensor_floats(model.tensor(gdn_prefix + ".conv1d.weight"));
  const auto conv_width = projected.size();
  std::vector<float> kernel(config.linear_conv_kernel_dim * conv_width);
  for (std::size_t channel = 0; channel < conv_width; ++channel) {
    for (std::size_t tap = 0;
         tap < static_cast<std::size_t>(config.linear_conv_kernel_dim); ++tap) {
      kernel[tap * conv_width + channel] =
          conv_weight[channel * config.linear_conv_kernel_dim + tap];
    }
  }
  std::vector<float> history(
      (config.linear_conv_kernel_dim - 1) * conv_width, 0.0F);
  const auto convolved = omlx::m5::causal_depthwise_conv_silu(
      projected, 1, conv_width, kernel, config.linear_conv_kernel_dim,
      history);
  const auto key_width = static_cast<std::size_t>(
      config.linear_num_key_heads * config.linear_key_head_dim);
  const auto value_width = static_cast<std::size_t>(
      config.linear_num_value_heads * config.linear_value_head_dim);
  auto query = std::vector<float>(
      convolved.begin(),
      convolved.begin() + static_cast<std::ptrdiff_t>(key_width));
  auto key = std::vector<float>(
      convolved.begin() + static_cast<std::ptrdiff_t>(key_width),
      convolved.begin() + static_cast<std::ptrdiff_t>(2 * key_width));
  const auto value = std::span<const float>(convolved).subspan(2 * key_width);
  const std::span<const float> no_weight;
  query = omlx::m5::rms_norm(query, config.linear_num_key_heads,
                             config.linear_key_head_dim, no_weight, 1.0e-6F);
  key = omlx::m5::rms_norm(key, config.linear_num_key_heads,
                           config.linear_key_head_dim, no_weight, 1.0e-6F);
  const auto inv_scale =
      1.0F / std::sqrt(static_cast<float>(config.linear_key_head_dim));
  for (auto &item : query) {
    item *= inv_scale * inv_scale;
  }
  for (auto &item : key) {
    item *= inv_scale;
  }
  const auto projected_a =
      qlinear(model, config, gdn_prefix + ".in_proj_a", normalized);
  const auto projected_b =
      qlinear(model, config, gdn_prefix + ".in_proj_b", normalized);
  const auto a_log = tensor_floats(model.tensor(gdn_prefix + ".A_log"));
  const auto dt_bias = tensor_floats(model.tensor(gdn_prefix + ".dt_bias"));
  std::vector<float> decay(config.linear_num_value_heads);
  std::vector<float> beta(config.linear_num_value_heads);
  for (std::size_t head = 0;
       head < static_cast<std::size_t>(config.linear_num_value_heads); ++head) {
    const auto activation = projected_a[head] + dt_bias[head];
    const auto softplus = activation > 20.0F
                              ? activation
                              : std::log1p(std::exp(activation));
    decay[head] = std::exp(-std::exp(a_log[head]) * softplus);
    beta[head] = 1.0F / (1.0F + std::exp(-projected_b[head]));
  }
  omlx::m5::GdnReferenceState state({
      .key_heads = static_cast<std::size_t>(config.linear_num_key_heads),
      .value_heads = static_cast<std::size_t>(config.linear_num_value_heads),
      .key_dim = static_cast<std::size_t>(config.linear_key_head_dim),
      .value_dim = static_cast<std::size_t>(config.linear_value_head_dim),
  });
  auto recurrent =
      omlx::m5::gdn_step(query, key, value, decay, beta, state);
  const auto recurrent_norm_weight =
      tensor_floats(model.tensor(gdn_prefix + ".norm.weight"));
  recurrent = omlx::m5::rms_norm(
      recurrent, config.linear_num_value_heads, config.linear_value_head_dim,
      recurrent_norm_weight, static_cast<float>(config.rms_norm_eps));
  const auto z =
      qlinear(model, config, gdn_prefix + ".in_proj_z", normalized);
  for (std::size_t index = 0; index < value_width; ++index) {
    recurrent[index] *= z[index] / (1.0F + std::exp(-z[index]));
  }
  add_in_place(hidden, qlinear(model, config, gdn_prefix + ".out_proj",
                               recurrent));
  const auto post_norm_weight = tensor_floats(
      model.tensor(prefix + ".post_attention_layernorm.weight"));
  normalized = omlx::m5::rms_norm(
      hidden, 1, config.hidden_size, post_norm_weight,
      static_cast<float>(config.rms_norm_eps));
  add_in_place(hidden, mlp(model, config, prefix + ".mlp", normalized));
  return hidden;
}

} // namespace

int main(int argc, const char *const *argv) {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "usage: omlx_m5_reference_layer MODEL_DIR {gdn|attention}");
    }
    const std::filesystem::path model_dir = argv[1];
    const std::string_view kind = argv[2];
    const auto config =
        omlx::m5::load_qwen_model_config(model_dir / "config.json");
    omlx::m5::validate_qwen38_27b_target(config);
    const auto model = omlx::m5::SafeTensorModel::open(model_dir);
    const auto input = deterministic_hidden(config.hidden_size);
    std::vector<float> output;
    std::size_t layer = 0;
    if (kind == "gdn") {
      output = run_gdn(model, config, input, layer);
    } else if (kind == "attention") {
      layer = 3;
      output = run_attention(model, config, input, layer);
    } else {
      throw std::invalid_argument("layer kind must be gdn or attention");
    }
    const nlohmann::json result = {
        {"schema_version", 1},
        {"scope", "M4 correctness"},
        {"kind", kind},
        {"layer", layer},
        {"width", output.size()},
        {"output", output},
    };
    std::cout << result.dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "omlx_m5_reference_layer: " << error.what() << '\n';
    return 1;
  }
}
