#include <metal_stdlib>

using namespace metal;

struct DFlash2MetalConfig {
  uint hidden_size;
  uint intermediate_size;
  uint vocabulary_size;
  uint layer_count;
  uint query_heads;
  uint key_value_heads;
  uint head_dim;
  uint target_taps;
  uint context_tokens;
  uint width;
  uint conv_group_size;
  uint conv_kernel_size;
  uint selector_rank;
  uint selector_top_k;
  uint sliding_window;
  uint mask_token_id;
  float epsilon;
  float rope_theta;
};

constant uint kGlobalOffsetCount = 8;
constant uint kLayerOffsetCount = 15;

inline device const bfloat *weight_pointer(device const uchar *weights,
                                           device const ulong *offsets,
                                           uint index) {
  return reinterpret_cast<device const bfloat *>(weights + offsets[index]);
}

inline float weight_value(device const uchar *weights,
                          device const ulong *offsets, uint index,
                          ulong element) {
  return float(weight_pointer(weights, offsets, index)[element]);
}

inline uint layer_offset(uint layer, uint field) {
  return kGlobalOffsetCount + layer * kLayerOffsetCount + field;
}

inline float rms_inverse(device const float *values, ulong base, uint count,
                         float epsilon) {
  float sum = 0.0f;
  for (uint i = 0; i < count; ++i) {
    const float value = values[base + i];
    sum += value * value;
  }
  return rsqrt(sum / float(count) + epsilon);
}

inline float silu(float value) { return value / (1.0f + exp(-value)); }

inline void rope_pair(thread float &left, thread float &right, uint pair,
                      uint half_dim, uint position, float theta) {
  const float frequency = pow(theta, -float(pair) / float(half_dim));
  const float angle = float(position) * frequency;
  const float cosine = cos(angle);
  const float sine = sin(angle);
  const float a = left;
  const float b = right;
  left = a * cosine - b * sine;
  right = b * cosine + a * sine;
}

kernel void omlx_m5_dflash2_program(
    device const float *noise_embedding [[buffer(0)]],
    device const float *target_features [[buffer(1)]],
    device const uchar *weights [[buffer(2)]],
    device const ulong *offsets [[buffer(3)]],
    device float *workspace [[buffer(4)]],
    device float *hidden [[buffer(5)]],
    device float *logits [[buffer(6)]],
    device const DFlash2MetalConfig &config [[buffer(7)]],
    uint thread_index [[thread_position_in_grid]]) {
  const uint h = config.hidden_size;
  const uint intermediate = config.intermediate_size;
  const uint width = config.width;
  const uint context = config.context_tokens;
  const uint q_width = config.query_heads * config.head_dim;
  const uint kv_width = config.key_value_heads * config.head_dim;
  const uint groups = h / config.conv_group_size;
  const uint target_width = config.target_taps * h;

  ulong cursor = 0;
  device float *projected_context = workspace + cursor;
  cursor += ulong(context) * h;
  device float *normalized = workspace + cursor;
  cursor += ulong(width) * h;
  device float *dynamic = workspace + cursor;
  cursor += ulong(width) * 4UL * groups;
  device float *prepared = workspace + cursor;
  cursor += ulong(width) * h;
  device float *queries = workspace + cursor;
  cursor += ulong(width) * q_width;
  device float *noise_keys = workspace + cursor;
  cursor += ulong(width) * kv_width;
  device float *noise_values = workspace + cursor;
  cursor += ulong(width) * kv_width;
  device float *context_keys = workspace + cursor;
  cursor += ulong(context) * kv_width;
  device float *context_values = workspace + cursor;
  cursor += ulong(context) * kv_width;
  device float *attention = workspace + cursor;
  cursor += ulong(width) * q_width;
  device float *projected = workspace + cursor;
  cursor += ulong(width) * h;
  device float *gate = workspace + cursor;
  cursor += ulong(width) * intermediate;
  device float *up = workspace + cursor;
  cursor += ulong(width) * intermediate;
  device float *product = workspace + cursor;

  if (thread_index < width) {
    const uint token = thread_index;
    for (uint feature = 0; feature < h; ++feature) {
      hidden[token * h + feature] = noise_embedding[token * h + feature];
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  for (uint token = thread_index; token < context; token += 32u) {
    const ulong target_base = ulong(token) * target_width;
    const ulong projected_base = ulong(token) * h;
    for (uint output = 0; output < h; ++output) {
      float value = 0.0f;
      for (uint input = 0; input < target_width; ++input) {
        value += weight_value(weights, offsets, 0,
                              ulong(output) * target_width + input) *
                 target_features[target_base + input];
      }
      projected_context[projected_base + output] = value;
    }
    const float inverse =
        rms_inverse(projected_context, projected_base, h, config.epsilon);
    for (uint feature = 0; feature < h; ++feature) {
      projected_context[projected_base + feature] *=
          inverse * weight_value(weights, offsets, 1, feature);
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  for (uint layer = 0; layer < config.layer_count; ++layer) {
    const uint input_norm = layer_offset(layer, 0);
    const uint post_norm = layer_offset(layer, 1);
    const uint q_weight = layer_offset(layer, 2);
    const uint k_weight = layer_offset(layer, 3);
    const uint v_weight = layer_offset(layer, 4);
    const uint q_norm = layer_offset(layer, 5);
    const uint k_norm = layer_offset(layer, 6);
    const uint o_weight = layer_offset(layer, 7);
    const uint gate_weight = layer_offset(layer, 8);
    const uint up_weight = layer_offset(layer, 9);
    const uint down_weight = layer_offset(layer, 10);
    const uint attention_base = layer_offset(layer, 11);
    const uint attention_projection = layer_offset(layer, 12);
    const uint mlp_base = layer_offset(layer, 13);
    const uint mlp_projection = layer_offset(layer, 14);

    if (thread_index < width) {
      const uint token = thread_index;
      const float inverse = rms_inverse(hidden, token * h, h, config.epsilon);
      for (uint feature = 0; feature < h; ++feature) {
        normalized[token * h + feature] =
            hidden[token * h + feature] * inverse *
            weight_value(weights, offsets, input_norm, feature);
      }
      for (uint row = 0; row < 4 * groups; ++row) {
        float value = 0.0f;
        for (uint feature = 0; feature < h; ++feature) {
          value += weight_value(weights, offsets, attention_projection,
                                ulong(row) * h + feature) *
                   normalized[token * h + feature];
        }
        dynamic[token * 4 * groups + row] = value;
      }
      for (uint feature = 0; feature < h; ++feature) {
        const uint group = feature / config.conv_group_size;
        float value =
            (weight_value(weights, offsets, attention_base, feature) +
             dynamic[token * 4 * groups + group]) *
            normalized[token * h + feature];
        if (token > 0) {
          value +=
              (weight_value(weights, offsets, attention_base, h + feature) +
               dynamic[token * 4 * groups + groups + group]) *
              normalized[(token - 1) * h + feature];
        }
        prepared[token * h + feature] = value;
      }
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (thread_index < width) {
      const uint token = thread_index;
      for (uint output = 0; output < q_width; ++output) {
        float value = 0.0f;
        for (uint input = 0; input < h; ++input) {
          value += weight_value(weights, offsets, q_weight,
                                ulong(output) * h + input) *
                   prepared[token * h + input];
        }
        queries[token * q_width + output] = value;
      }
      for (uint output = 0; output < kv_width; ++output) {
        float key = 0.0f;
        float value = 0.0f;
        for (uint input = 0; input < h; ++input) {
          const float source = prepared[token * h + input];
          key += weight_value(weights, offsets, k_weight,
                              ulong(output) * h + input) *
                 source;
          value += weight_value(weights, offsets, v_weight,
                                ulong(output) * h + input) *
                   source;
        }
        noise_keys[token * kv_width + output] = key;
        noise_values[token * kv_width + output] = value;
      }
    }
    for (uint token = thread_index; token < context; token += 32u) {
      const ulong projected_base = ulong(token) * h;
      const ulong kv_base = ulong(token) * kv_width;
      for (uint output = 0; output < kv_width; ++output) {
        float key = 0.0f;
        float value = 0.0f;
        for (uint input = 0; input < h; ++input) {
          const float source = projected_context[projected_base + input];
          key += weight_value(weights, offsets, k_weight,
                              ulong(output) * h + input) *
                 source;
          value += weight_value(weights, offsets, v_weight,
                                ulong(output) * h + input) *
                   source;
        }
        context_keys[kv_base + output] = key;
        context_values[kv_base + output] = value;
      }
    }
    threadgroup_barrier(mem_flags::mem_device);

    const uint half_dim = config.head_dim / 2;
    if (thread_index < width) {
      const uint token = thread_index;
      const uint position = context + token;
      for (uint head = 0; head < config.query_heads; ++head) {
        const uint base = token * q_width + head * config.head_dim;
        const float inverse =
            rms_inverse(queries, base, config.head_dim, config.epsilon);
        for (uint feature = 0; feature < config.head_dim; ++feature) {
          queries[base + feature] *=
              inverse * weight_value(weights, offsets, q_norm, feature);
        }
        for (uint pair = 0; pair < half_dim; ++pair) {
          thread float left = queries[base + pair];
          thread float right = queries[base + half_dim + pair];
          rope_pair(left, right, pair, half_dim, position, config.rope_theta);
          queries[base + pair] = left;
          queries[base + half_dim + pair] = right;
        }
      }
      for (uint head = 0; head < config.key_value_heads; ++head) {
        const uint base = token * kv_width + head * config.head_dim;
        const float inverse =
            rms_inverse(noise_keys, base, config.head_dim, config.epsilon);
        for (uint feature = 0; feature < config.head_dim; ++feature) {
          noise_keys[base + feature] *=
              inverse * weight_value(weights, offsets, k_norm, feature);
        }
        for (uint pair = 0; pair < half_dim; ++pair) {
          thread float left = noise_keys[base + pair];
          thread float right = noise_keys[base + half_dim + pair];
          rope_pair(left, right, pair, half_dim, position, config.rope_theta);
          noise_keys[base + pair] = left;
          noise_keys[base + half_dim + pair] = right;
        }
      }
    }
    for (uint token = thread_index; token < context; token += 32u) {
      for (uint head = 0; head < config.key_value_heads; ++head) {
        const ulong base =
            ulong(token) * kv_width + ulong(head) * config.head_dim;
        const float inverse =
            rms_inverse(context_keys, base, config.head_dim, config.epsilon);
        for (uint feature = 0; feature < config.head_dim; ++feature) {
          context_keys[base + feature] *=
              inverse * weight_value(weights, offsets, k_norm, feature);
        }
        for (uint pair = 0; pair < half_dim; ++pair) {
          thread float left = context_keys[base + pair];
          thread float right = context_keys[base + half_dim + pair];
          rope_pair(left, right, pair, half_dim, token, config.rope_theta);
          context_keys[base + pair] = left;
          context_keys[base + half_dim + pair] = right;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_device);

    const float scale = rsqrt(float(config.head_dim));
    if (thread_index < width) {
      const uint token = thread_index;
      const uint query_position = context + token;
      for (uint q_head = 0; q_head < config.query_heads; ++q_head) {
        const uint kv_head =
            q_head * config.key_value_heads / config.query_heads;
        float maximum = -INFINITY;
        for (uint key_index = 0; key_index < context + width; ++key_index) {
          bool allowed = true;
          if (key_index < context) {
            allowed = query_position - key_index < config.sliding_window;
          }
          if (!allowed) {
            continue;
          }
          float score = 0.0f;
          for (uint feature = 0; feature < config.head_dim; ++feature) {
            const float key =
                key_index < context
                    ? context_keys[ulong(key_index) * kv_width +
                                   kv_head * config.head_dim + feature]
                    : noise_keys[(key_index - context) * kv_width +
                                 kv_head * config.head_dim + feature];
            score += queries[token * q_width + q_head * config.head_dim +
                             feature] *
                     key;
          }
          maximum = max(maximum, score * scale);
        }
        float denominator = 0.0f;
        for (uint key_index = 0; key_index < context + width; ++key_index) {
          bool allowed = true;
          if (key_index < context) {
            allowed = query_position - key_index < config.sliding_window;
          }
          if (!allowed) {
            continue;
          }
          float score = 0.0f;
          for (uint feature = 0; feature < config.head_dim; ++feature) {
            const float key =
                key_index < context
                    ? context_keys[ulong(key_index) * kv_width +
                                   kv_head * config.head_dim + feature]
                    : noise_keys[(key_index - context) * kv_width +
                                 kv_head * config.head_dim + feature];
            score += queries[token * q_width + q_head * config.head_dim +
                             feature] *
                     key;
          }
          denominator += exp(score * scale - maximum);
        }
        for (uint feature = 0; feature < config.head_dim; ++feature) {
          float output = 0.0f;
          for (uint key_index = 0; key_index < context + width; ++key_index) {
            bool allowed = true;
            if (key_index < context) {
              allowed = query_position - key_index < config.sliding_window;
            }
            if (!allowed) {
              continue;
            }
            float score = 0.0f;
            for (uint inner = 0; inner < config.head_dim; ++inner) {
              const float key =
                  key_index < context
                      ? context_keys[ulong(key_index) * kv_width +
                                     kv_head * config.head_dim + inner]
                      : noise_keys[(key_index - context) * kv_width +
                                   kv_head * config.head_dim + inner];
              score += queries[token * q_width + q_head * config.head_dim +
                               inner] *
                       key;
            }
            const float probability = exp(score * scale - maximum) / denominator;
            const float value =
                key_index < context
                    ? context_values[ulong(key_index) * kv_width +
                                     kv_head * config.head_dim + feature]
                    : noise_values[(key_index - context) * kv_width +
                                   kv_head * config.head_dim + feature];
            output += probability * value;
          }
          attention[token * q_width + q_head * config.head_dim + feature] =
              output;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (thread_index < width) {
      const uint token = thread_index;
      for (uint output = 0; output < h; ++output) {
        float value = 0.0f;
        for (uint input = 0; input < q_width; ++input) {
          value += weight_value(weights, offsets, o_weight,
                                ulong(output) * q_width + input) *
                   attention[token * q_width + input];
        }
        projected[token * h + output] = value;
      }
      for (uint feature = 0; feature < h; ++feature) {
        const uint group = feature / config.conv_group_size;
        float value =
            (weight_value(weights, offsets, attention_base, 2UL * h + feature) +
             dynamic[token * 4 * groups + 2 * groups + group]) *
            projected[token * h + feature];
        if (token > 0) {
          value +=
              (weight_value(weights, offsets, attention_base,
                            3UL * h + feature) +
               dynamic[token * 4 * groups + 3 * groups + group]) *
              projected[(token - 1) * h + feature];
        }
        hidden[token * h + feature] += value;
      }
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (thread_index < width) {
      const uint token = thread_index;
      const float inverse = rms_inverse(hidden, token * h, h, config.epsilon);
      for (uint feature = 0; feature < h; ++feature) {
        normalized[token * h + feature] =
            hidden[token * h + feature] * inverse *
            weight_value(weights, offsets, post_norm, feature);
      }
      for (uint row = 0; row < 4 * groups; ++row) {
        float value = 0.0f;
        for (uint feature = 0; feature < h; ++feature) {
          value += weight_value(weights, offsets, mlp_projection,
                                ulong(row) * h + feature) *
                   normalized[token * h + feature];
        }
        dynamic[token * 4 * groups + row] = value;
      }
      for (uint feature = 0; feature < h; ++feature) {
        const uint group = feature / config.conv_group_size;
        float value =
            (weight_value(weights, offsets, mlp_base, feature) +
             dynamic[token * 4 * groups + group]) *
            normalized[token * h + feature];
        if (token > 0) {
          value += (weight_value(weights, offsets, mlp_base, h + feature) +
                    dynamic[token * 4 * groups + groups + group]) *
                   normalized[(token - 1) * h + feature];
        }
        prepared[token * h + feature] = value;
      }
      for (uint output = 0; output < intermediate; ++output) {
        float gate_value = 0.0f;
        float up_value = 0.0f;
        for (uint input = 0; input < h; ++input) {
          const float source = prepared[token * h + input];
          gate_value += weight_value(weights, offsets, gate_weight,
                                     ulong(output) * h + input) *
                        source;
          up_value += weight_value(weights, offsets, up_weight,
                                   ulong(output) * h + input) *
                      source;
        }
        gate[token * intermediate + output] = gate_value;
        up[token * intermediate + output] = up_value;
        product[token * intermediate + output] = silu(gate_value) * up_value;
      }
      for (uint output = 0; output < h; ++output) {
        float value = 0.0f;
        for (uint input = 0; input < intermediate; ++input) {
          value += weight_value(weights, offsets, down_weight,
                                ulong(output) * intermediate + input) *
                   product[token * intermediate + input];
        }
        projected[token * h + output] = value;
      }
      for (uint feature = 0; feature < h; ++feature) {
        const uint group = feature / config.conv_group_size;
        float value =
            (weight_value(weights, offsets, mlp_base, 2UL * h + feature) +
             dynamic[token * 4 * groups + 2 * groups + group]) *
            projected[token * h + feature];
        if (token > 0) {
          value +=
              (weight_value(weights, offsets, mlp_base, 3UL * h + feature) +
               dynamic[token * 4 * groups + 3 * groups + group]) *
              projected[(token - 1) * h + feature];
        }
        hidden[token * h + feature] += value;
      }
    }
    threadgroup_barrier(mem_flags::mem_device);
  }

  if (thread_index < width) {
    const uint token = thread_index;
    const float inverse = rms_inverse(hidden, token * h, h, config.epsilon);
    for (uint feature = 0; feature < h; ++feature) {
      hidden[token * h + feature] *=
          inverse * weight_value(weights, offsets, 2, feature);
    }
    for (uint output = 0; output < config.vocabulary_size; ++output) {
      float value = 0.0f;
      for (uint input = 0; input < h; ++input) {
        value += weight_value(weights, offsets, 4,
                              ulong(output) * h + input) *
                 hidden[token * h + input];
      }
      logits[token * config.vocabulary_size + output] = value;
    }
  }
}
