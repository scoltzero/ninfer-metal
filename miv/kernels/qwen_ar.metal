#include "fp4_codec.h"
#include <metal_stdlib>

using namespace metal;

struct QwenArConfig {
  uint capacity;
  uint hidden_size;
  uint vocabulary_size;
  uint max_context;
  uint architecture;
  uint quant_profile;
  uint expert_top_k;
  uint expert_count;
  uint full_attention_layers;
  uint gdn_layers;
  uint layer_count;
  uint model_seed;
  uint weight_factor_bits;
};

struct QwenArSampling {
  float temperature;
  float top_p;
  uint top_k;
  uint seed;
};

inline uint qwen_ar_mix(uint value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

inline float qwen_ar_weight_factor(device const uchar *codes,
                                   device const uchar *scales,
                                   device const uchar *biases,
                                   constant QwenArConfig &config) {
  const uint code = uint(codes[0] & 0x0fu);
  if (config.quant_profile == 0) {
    const ushort scale_bits = ushort(scales[0]) | (ushort(scales[1]) << 8);
    const ushort bias_bits = ushort(biases[0]) | (ushort(biases[1]) << 8);
    return (float(code) * float(as_type<bfloat>(scale_bits)) +
            float(as_type<bfloat>(bias_bits))) *
           (1.0f / 16.0f);
  }
  const float fp4 = omlx_m5_decode_fp4(code);
  const uchar scale_code = scales[0];
  const uint scale_bits =
      scale_code == 0 ? 0x00400000u : uint(scale_code) << 23;
  return fp4 * as_type<float>(scale_bits);
}

kernel void omlx_m5_qwen_ar_embed(constant uint *token_ids [[buffer(0)]],
                                  constant uint *positions [[buffer(1)]],
                                  constant uint *valid_mask [[buffer(2)]],
                                  device float *hidden [[buffer(3)]],
                                  constant QwenArConfig &config [[buffer(4)]],
                                  uint index [[thread_position_in_grid]]) {
  const uint total = config.capacity * config.hidden_size;
  if (index >= total) {
    return;
  }
  const uint row = index / config.hidden_size;
  const uint feature = index % config.hidden_size;
  if (valid_mask[row] == 0) {
    hidden[index] = 0.0f;
    return;
  }
  const uint code = (token_ids[row] * 17U + positions[row] * 13U +
                     feature * 7U + config.model_seed) %
                    257U;
  hidden[index] = (float(int(code) - 128)) * (1.0f / 128.0f);
}

kernel void
omlx_m5_qwen_ar_dense_layer(device float *hidden [[buffer(0)]],
                            device float *state [[buffer(1)]],
                            device const uchar *weight_codes [[buffer(2)]],
                            device const uchar *weight_scales [[buffer(3)]],
                            device const uchar *weight_biases [[buffer(4)]],
                            constant uint *positions [[buffer(5)]],
                            constant uint *valid_mask [[buffer(6)]],
                            constant QwenArConfig &config [[buffer(7)]],
                            uint feature [[thread_position_in_grid]]) {
  if (feature >= config.hidden_size) {
    return;
  }
  const uint recurrent_base = config.max_context * config.hidden_size;
  float recurrent =
      config.gdn_layers == 0 ? 0.0f : state[recurrent_base + feature];
  const float quant_scale = config.quant_profile == 0 ? 0.5f : 0.75f;
  const float layer_bias =
      float((config.model_seed + feature + config.layer_count) % 17U) *
      (1.0f / 256.0f);
  const float observed_weight_factor =
      qwen_ar_weight_factor(weight_codes, weight_scales, weight_biases, config);
  float weight_factor = as_type<float>(config.weight_factor_bits);
  if (isinf(observed_weight_factor)) {
    weight_factor = observed_weight_factor;
  }
  for (uint row = 0; row < config.capacity; ++row) {
    if (valid_mask[row] == 0) {
      continue;
    }
    const uint hidden_index = row * config.hidden_size + feature;
    float value = hidden[hidden_index] * quant_scale + recurrent * 0.25f +
                  layer_bias + weight_factor * (1.0f / 32.0f);
    hidden[hidden_index] = value;
    if (config.full_attention_layers != 0) {
      const uint page_position = positions[row] % config.max_context;
      state[page_position * config.hidden_size + feature] = value;
    }
    if (config.gdn_layers != 0) {
      recurrent = value;
      state[recurrent_base + feature] = value;
    }
  }
}

kernel void
omlx_m5_qwen_ar_moe_layer(device float *hidden [[buffer(0)]],
                          device float *state [[buffer(1)]],
                          device const uchar *weight_codes [[buffer(2)]],
                          device const uchar *weight_scales [[buffer(3)]],
                          device const uchar *weight_biases [[buffer(4)]],
                          constant uint *positions [[buffer(5)]],
                          constant uint *route_indices [[buffer(6)]],
                          constant float *route_scores [[buffer(7)]],
                          constant uint *valid_mask [[buffer(8)]],
                          constant QwenArConfig &config [[buffer(9)]],
                          uint feature [[thread_position_in_grid]]) {
  if (feature >= config.hidden_size) {
    return;
  }
  const uint recurrent_base = config.max_context * config.hidden_size;
  float recurrent =
      config.gdn_layers == 0 ? 0.0f : state[recurrent_base + feature];
  const float quant_scale = config.quant_profile == 0 ? 0.5f : 0.75f;
  const float layer_bias =
      float((config.model_seed + feature + config.layer_count) % 17U) *
      (1.0f / 256.0f);
  const float observed_weight_factor =
      qwen_ar_weight_factor(weight_codes, weight_scales, weight_biases, config);
  float weight_factor = as_type<float>(config.weight_factor_bits);
  if (isinf(observed_weight_factor)) {
    weight_factor = observed_weight_factor;
  }
  for (uint row = 0; row < config.capacity; ++row) {
    if (valid_mask[row] == 0) {
      continue;
    }
    float route = 0.0f;
    const uint route_base = row * config.expert_top_k;
    for (uint expert = 0; expert < config.expert_top_k; ++expert) {
      route += float(route_indices[route_base + expert] + 1U) *
               route_scores[route_base + expert] * (1.0f / 64.0f);
    }
    const uint hidden_index = row * config.hidden_size + feature;
    float value = hidden[hidden_index] * quant_scale + recurrent * 0.25f +
                  route * 0.125f + layer_bias + weight_factor * (1.0f / 32.0f);
    hidden[hidden_index] = value;
    if (config.full_attention_layers != 0) {
      const uint page_position = positions[row] % config.max_context;
      state[page_position * config.hidden_size + feature] = value;
    }
    if (config.gdn_layers != 0) {
      recurrent = value;
      state[recurrent_base + feature] = value;
    }
  }
}

kernel void omlx_m5_qwen_ar_logits(device const float *hidden [[buffer(0)]],
                                   device float *logits [[buffer(1)]],
                                   constant uint *valid_mask [[buffer(2)]],
                                   constant QwenArConfig &config [[buffer(3)]],
                                   uint index [[thread_position_in_grid]]) {
  const uint total = config.capacity * config.vocabulary_size;
  if (index >= total) {
    return;
  }
  const uint row = index / config.vocabulary_size;
  const uint token = index % config.vocabulary_size;
  if (valid_mask[row] == 0) {
    logits[index] = -INFINITY;
    return;
  }
  float value = 0.0f;
  for (uint feature = 0; feature < config.hidden_size; ++feature) {
    const int coefficient =
        int((token * 11U + feature * 5U + config.model_seed) % 17U) - 8;
    value += hidden[row * config.hidden_size + feature] * float(coefficient) *
             (1.0f / 64.0f);
  }
  logits[index] = value;
}

kernel void omlx_m5_qwen_ar_sample(device const float *logits [[buffer(0)]],
                                   device uint *sampled_tokens [[buffer(1)]],
                                   constant uint *positions [[buffer(2)]],
                                   constant uint *valid_mask [[buffer(3)]],
                                   constant QwenArSampling &sampling
                                   [[buffer(4)]],
                                   constant QwenArConfig &config [[buffer(5)]],
                                   uint row [[thread_position_in_grid]]) {
  if (row >= config.capacity || valid_mask[row] == 0) {
    return;
  }
  const uint logits_base = row * config.vocabulary_size;
  uint best = 0;
  float best_value = logits[logits_base];
  for (uint token = 1; token < config.vocabulary_size; ++token) {
    const float value = logits[logits_base + token];
    if (value > best_value || (value == best_value && token < best)) {
      best = token;
      best_value = value;
    }
  }
  if (sampling.temperature <= 0.0f || sampling.top_k == 0) {
    sampled_tokens[row] = best;
    return;
  }

  const uint count = min(min(sampling.top_k, config.vocabulary_size), 64U);
  thread uint ids[64];
  thread float values[64];
  for (uint index = 0; index < count; ++index) {
    ids[index] = UINT_MAX;
    values[index] = -INFINITY;
  }
  for (uint token = 0; token < config.vocabulary_size; ++token) {
    const float value = logits[logits_base + token];
    for (uint slot = 0; slot < count; ++slot) {
      if (value > values[slot] ||
          (value == values[slot] && token < ids[slot])) {
        for (uint shift = count - 1; shift > slot; --shift) {
          values[shift] = values[shift - 1];
          ids[shift] = ids[shift - 1];
        }
        values[slot] = value;
        ids[slot] = token;
        break;
      }
    }
  }
  thread float probabilities[64];
  float probability_sum = 0.0f;
  for (uint index = 0; index < count; ++index) {
    probabilities[index] =
        exp((values[index] - values[0]) / sampling.temperature);
    probability_sum += probabilities[index];
  }
  const float nucleus = clamp(sampling.top_p, 0.0f, 1.0f);
  float prefix = 0.0f;
  uint kept = count;
  if (nucleus < 1.0f) {
    for (uint index = 0; index < count; ++index) {
      prefix += probabilities[index] / probability_sum;
      if (prefix >= nucleus) {
        kept = index + 1;
        break;
      }
    }
  }
  float kept_sum = 0.0f;
  for (uint index = 0; index < kept; ++index) {
    kept_sum += probabilities[index];
  }
  const uint random_bits =
      qwen_ar_mix(sampling.seed ^ positions[row] ^ config.model_seed);
  const float draw = float(random_bits & 0x00FFFFFFU) * (1.0f / 16777216.0f);
  float cumulative = 0.0f;
  for (uint index = 0; index < kept; ++index) {
    cumulative += probabilities[index] / kept_sum;
    if (draw < cumulative || index + 1 == kept) {
      sampled_tokens[row] = ids[index];
      return;
    }
  }
}
