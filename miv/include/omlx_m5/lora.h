#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "omlx_m5/model_config.h"

namespace omlx::m5 {

enum class LoraFormat {
  peft,
  mlx,
};

struct LoraTargetKey {
  bool is_global = false;
  std::uint32_t layer_index = 0;
  std::string role;

  bool operator==(const LoraTargetKey &) const = default;
};

struct LoraProjectionUpdate {
  LoraTargetKey target;
  std::string source_module;
  std::uint32_t rank = 0;
  std::uint32_t out_features = 0;
  std::uint32_t in_features = 0;
  float scale = 1.0F;
  // Canonical row-major [rank, in_features] and [out_features, rank] FP32
  // factors such that delta_W = scale * (B @ A) of shape [out_features, in_features].
  std::vector<float> a_f32;
  std::vector<float> b_f32;
};

struct LoadedLoraAdapter {
  std::filesystem::path adapter_dir;
  LoraFormat format = LoraFormat::peft;
  std::string fingerprint;
  std::uint32_t adapted_layers = 0;
  std::vector<LoraProjectionUpdate> updates;
};

[[nodiscard]] std::uint16_t lora_float_to_bf16_rne(float value) noexcept;
[[nodiscard]] float lora_bf16_to_float(std::uint16_t bits) noexcept;

[[nodiscard]] LoadedLoraAdapter
load_lora_adapter_for_qwen(const std::filesystem::path &adapter_dir,
                           const QwenModelConfig &config,
                           bool lm_head_tied = false);

} // namespace omlx::m5
