#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace omlx::m5 {

enum class LayerType {
  linear_attention,
  full_attention,
};

struct QuantizationSpec {
  std::int32_t bits = 4;
  std::int32_t group_size = 64;
  std::string mode = "affine";
};

struct QwenModelConfig {
  std::string model_type;
  std::string text_model_type;
  std::string dtype;
  std::int32_t hidden_size = 0;
  std::int32_t num_hidden_layers = 0;
  std::int32_t intermediate_size = 0;
  std::int32_t num_experts = 0;
  std::int32_t num_experts_per_token = 0;
  std::int32_t shared_expert_intermediate_size = 0;
  std::int32_t vocab_size = 0;
  std::int32_t num_attention_heads = 0;
  std::int32_t num_key_value_heads = 0;
  std::int32_t head_dim = 0;
  std::int32_t linear_num_key_heads = 0;
  std::int32_t linear_num_value_heads = 0;
  std::int32_t linear_key_head_dim = 0;
  std::int32_t linear_value_head_dim = 0;
  std::int32_t linear_conv_kernel_dim = 0;
  std::int32_t max_position_embeddings = 0;
  std::int32_t eos_token_id = -1;
  double rms_norm_eps = 1.0e-6;
  double rope_theta = 10000.0;
  double partial_rotary_factor = 1.0;
  bool rope_interleaved = true;
  std::vector<LayerType> layer_types;
  QuantizationSpec default_quantization;
  std::unordered_map<std::string, QuantizationSpec> quantization_overrides;

  [[nodiscard]] QuantizationSpec
  quantization_for(std::string_view module_name) const;
  [[nodiscard]] std::int32_t full_attention_layers() const;
  [[nodiscard]] std::int32_t linear_attention_layers() const;
};

[[nodiscard]] QwenModelConfig
load_qwen_model_config(const std::filesystem::path &config_path);
void validate_qwen38_27b_target(const QwenModelConfig &config);

} // namespace omlx::m5
