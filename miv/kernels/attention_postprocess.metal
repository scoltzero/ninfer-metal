#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

struct AttentionPostprocessParams {
  uint query_heads;
  uint key_value_heads;
  uint head_dim;
  uint rotary_dim;
  uint tokens;
  uint query_token_stride;
  uint query_head_stride;
  uint key_token_stride;
  uint key_head_stride;
  uint start_position;
  float epsilon;
  float theta;
  float weight_offset;
};

struct PagedAttentionParams {
  uint query_heads;
  uint key_value_heads;
  uint head_dim;
  uint tokens;
  uint cache_start;
  uint cache_capacity;
  uint page_tokens;
  uint page_count;
  uint query_token_stride;
  uint query_head_stride;
  uint key_token_stride;
  uint value_token_stride;
  uint gate_token_stride;
  uint gate_head_stride;
};

kernel void omlx_m5_attention_qk_norm_rope(
    device bfloat *query [[buffer(0)]], device bfloat *key [[buffer(1)]],
    device const bfloat *query_norm [[buffer(2)]],
    device const bfloat *key_norm [[buffer(3)]],
    constant AttentionPostprocessParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint heads_per_token = params.query_heads + params.key_value_heads;
  const uint logical_head = tg_position.y;
  const uint token = logical_head / heads_per_token;
  if (token >= params.tokens)
    return;
  const uint local_head = logical_head - token * heads_per_token;
  const bool is_query = local_head < params.query_heads;
  const uint head = is_query ? local_head : local_head - params.query_heads;
  device bfloat *row =
      is_query ? query + ulong(token) * params.query_token_stride +
                     ulong(head) * params.query_head_stride
               : key + ulong(token) * params.key_token_stride +
                     ulong(head) * params.key_head_stride;
  device const bfloat *norm = is_query ? query_norm : key_norm;

  float square_sum = 0.0f;
  for (uint column = lane; column < params.head_dim; column += 32) {
    const float value = float(row[column]);
    square_sum = fma(value, value, square_sum);
  }
  const float inverse_rms =
      rsqrt(simd_sum(square_sum) / float(params.head_dim) + params.epsilon);
  const uint half_rotary = params.rotary_dim / 2;
  const float position = float(params.start_position + token);
  for (uint pair = lane; pair < half_rotary; pair += 32) {
    const uint first_index = pair;
    const uint second_index = pair + half_rotary;
    const float first = float(row[first_index]) * inverse_rms *
                        (float(norm[first_index]) + params.weight_offset);
    const float second = float(row[second_index]) * inverse_rms *
                         (float(norm[second_index]) + params.weight_offset);
    const float exponent = float(2 * pair) / float(params.rotary_dim);
    const float angle = position / pow(params.theta, exponent);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    row[first_index] = bfloat(fma(-second, sine, first * cosine));
    row[second_index] = bfloat(fma(first, sine, second * cosine));
  }
  for (uint column = params.rotary_dim + lane; column < params.head_dim;
       column += 32) {
    row[column] = bfloat(float(row[column]) * inverse_rms *
                         (float(norm[column]) + params.weight_offset));
  }
}

inline ulong paged_position(uint logical, device const uint *page_table,
                            constant PagedAttentionParams &params) {
  const uint page = logical / params.page_tokens;
  const uint within = logical - page * params.page_tokens;
  return ulong(page_table[page]) * params.page_tokens + within;
}

kernel void omlx_m5_paged_kv_append(device const bfloat *key [[buffer(0)]],
                                    device const bfloat *value [[buffer(1)]],
                                    device bfloat *cache_key [[buffer(2)]],
                                    device bfloat *cache_value [[buffer(3)]],
                                    device const uint *page_table [[buffer(4)]],
                                    constant PagedAttentionParams &params
                                    [[buffer(5)]],
                                    uint gid [[thread_position_in_grid]]) {
  const uint width = params.key_value_heads * params.head_dim;
  const ulong values = ulong(params.tokens) * width;
  if (ulong(gid) >= values) {
    return;
  }
  const uint token = gid / width;
  const uint feature = gid - token * width;
  const ulong position =
      paged_position(params.cache_start + token, page_table, params);
  cache_key[position * width + feature] =
      key[ulong(token) * params.key_token_stride + feature];
  cache_value[position * width + feature] =
      value[ulong(token) * params.value_token_stride + feature];
}

kernel void omlx_m5_paged_causal_attention(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const bfloat *cache_key [[buffer(2)]],
    device const bfloat *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *page_table [[buffer(5)]],
    constant PagedAttentionParams &params [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint logical_head = tg_position.y;
  const uint token = logical_head / params.query_heads;
  if (token >= params.tokens)
    return;
  const uint query_head = logical_head - token * params.query_heads;
  const uint repeat = params.query_heads / params.key_value_heads;
  const uint kv_head = query_head / repeat;
  device const bfloat *query_row = query +
                                   ulong(token) * params.query_token_stride +
                                   ulong(query_head) * params.query_head_stride;
  device const bfloat *gate_row = gate +
                                  ulong(token) * params.gate_token_stride +
                                  ulong(query_head) * params.gate_head_stride;
  const ulong kv_width = ulong(params.key_value_heads) * params.head_dim;
  const uint context_tokens = params.cache_start + token + 1;
  const float scale = rsqrt(float(params.head_dim));

  float q_reg[8];
#pragma clang loop unroll(full)
  for (uint index = 0; index < 8; ++index) {
    const uint feature = lane + index * 32;
    q_reg[index] = feature < params.head_dim ? float(query_row[feature]) : 0.0f;
  }

  float running_max = -INFINITY;
  float denominator = 0.0f;
  float accumulators[8];
#pragma clang loop unroll(full)
  for (uint index = 0; index < 8; ++index) {
    accumulators[index] = 0.0f;
  }

  const uint logical_pages =
      (context_tokens + params.page_tokens - 1) / params.page_tokens;
  const ulong head_offset = ulong(kv_head) * params.head_dim;

  if (params.head_dim == 256) {
    for (uint page = 0; page < logical_pages; ++page) {
      const ulong page_base = ulong(page_table[page]) * params.page_tokens;
      const uint valid =
          min(params.page_tokens, context_tokens - page * params.page_tokens);
      for (uint within = 0; within < valid; ++within) {
        const ulong kv_offset = (page_base + within) * kv_width + head_offset;
        device const bfloat *key_row = cache_key + kv_offset;
        device const bfloat *value_row = cache_value + kv_offset;
        float partial = 0.0f;
#pragma clang loop unroll(full)
        for (uint index = 0; index < 8; ++index) {
          partial = fma(q_reg[index], float(key_row[lane + index * 32]), partial);
        }
        const float score = simd_sum(partial) * scale;
        if (score > running_max) {
          const float rescale =
              (running_max == -INFINITY || score == INFINITY)
                  ? 0.0f
                  : exp(running_max - score);
          denominator = fma(denominator, rescale, 1.0f);
          running_max = score;
#pragma clang loop unroll(full)
          for (uint index = 0; index < 8; ++index) {
            accumulators[index] = fma(accumulators[index], rescale,
                                      float(value_row[lane + index * 32]));
          }
        } else if (score > -INFINITY) {
          const float probability =
              (score == running_max) ? 1.0f : exp(score - running_max);
          denominator += probability;
#pragma clang loop unroll(full)
          for (uint index = 0; index < 8; ++index) {
            accumulators[index] = fma(probability,
                                      float(value_row[lane + index * 32]),
                                      accumulators[index]);
          }
        }
      }
    }
  } else {
    for (uint page = 0; page < logical_pages; ++page) {
      const ulong page_base = ulong(page_table[page]) * params.page_tokens;
      const uint valid =
          min(params.page_tokens, context_tokens - page * params.page_tokens);
      for (uint within = 0; within < valid; ++within) {
        const ulong kv_offset = (page_base + within) * kv_width + head_offset;
        device const bfloat *key_row = cache_key + kv_offset;
        device const bfloat *value_row = cache_value + kv_offset;
        float partial = 0.0f;
#pragma clang loop unroll(full)
        for (uint index = 0; index < 8; ++index) {
          const uint feature = lane + index * 32;
          if (feature < params.head_dim) {
            partial = fma(q_reg[index], float(key_row[feature]), partial);
          }
        }
        const float score = simd_sum(partial) * scale;
        if (score > running_max) {
          const float rescale =
              (running_max == -INFINITY || score == INFINITY)
                  ? 0.0f
                  : exp(running_max - score);
          denominator = fma(denominator, rescale, 1.0f);
          running_max = score;
#pragma clang loop unroll(full)
          for (uint index = 0; index < 8; ++index) {
            const uint feature = lane + index * 32;
            if (feature < params.head_dim) {
              accumulators[index] =
                  fma(accumulators[index], rescale, float(value_row[feature]));
            }
          }
        } else if (score > -INFINITY) {
          const float probability =
              (score == running_max) ? 1.0f : exp(score - running_max);
          denominator += probability;
#pragma clang loop unroll(full)
          for (uint index = 0; index < 8; ++index) {
            const uint feature = lane + index * 32;
            if (feature < params.head_dim) {
              accumulators[index] =
                  fma(probability, float(value_row[feature]), accumulators[index]);
            }
          }
        }
      }
    }
  }

#pragma clang loop unroll(full)
  for (uint index = 0; index < 8; ++index) {
    const uint feature = lane + index * 32;
    if (feature < params.head_dim) {
      const float gate_value = float(gate_row[feature]);
      const float gated =
          (denominator > 0.0f ? (accumulators[index] / denominator) : 0.0f) /
          (1.0f + exp(-gate_value));
      output[(ulong(token) * params.query_heads + query_head) *
                 params.head_dim +
             feature] = bfloat(gated);
    }
  }
}

struct KvPageCopyParams {
  uint source_page;
  uint destination_page;
  uint values_per_page;
};
kernel void omlx_m5_kv_page_copy(device bfloat *cache_key [[buffer(0)]],
                                 device bfloat *cache_value [[buffer(1)]],
                                 constant KvPageCopyParams &params
                                 [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
  if (i >= params.values_per_page)
    return;
  const ulong source = ulong(params.source_page) * params.values_per_page + i;
  const ulong destination =
      ulong(params.destination_page) * params.values_per_page + i;
  cache_key[destination] = cache_key[source];
  cache_value[destination] = cache_value[source];
}

// P06 diagnostic control for the existing fused q/k normalization + RoPE.
// It intentionally materializes BF16 normalized q/k between two submissions.
kernel void omlx_m5_attention_qk_norm_only(
    device bfloat *query [[buffer(0)]], device bfloat *key [[buffer(1)]],
    device const bfloat *query_norm [[buffer(2)]],
    device const bfloat *key_norm [[buffer(3)]],
    constant AttentionPostprocessParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  const uint token = group.y / (params.query_heads + params.key_value_heads);
  if (token >= params.tokens)
    return;
  const uint local = group.y % (params.query_heads + params.key_value_heads);
  const bool is_query = local < params.query_heads;
  const uint head = is_query ? local : local - params.query_heads;
  device bfloat *row =
      is_query ? query + ulong(token) * params.query_token_stride +
                     ulong(head) * params.query_head_stride
               : key + ulong(token) * params.key_token_stride +
                     ulong(head) * params.key_head_stride;
  device const bfloat *norm = is_query ? query_norm : key_norm;
  float squares = 0.0f;
  for (uint feature = lane; feature < params.head_dim; feature += 32) {
    const float value = float(row[feature]);
    squares = fma(value, value, squares);
  }
  const float inverse =
      rsqrt(simd_sum(squares) / float(params.head_dim) + params.epsilon);
  for (uint feature = lane; feature < params.head_dim; feature += 32)
    row[feature] = bfloat(float(row[feature]) * inverse *
                          (float(norm[feature]) + params.weight_offset));
}
kernel void omlx_m5_attention_qk_rope_only(
    device bfloat *query [[buffer(0)]], device bfloat *key [[buffer(1)]],
    constant AttentionPostprocessParams &params [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  const uint token = group.y / (params.query_heads + params.key_value_heads);
  if (token >= params.tokens)
    return;
  const uint local = group.y % (params.query_heads + params.key_value_heads);
  const bool is_query = local < params.query_heads;
  const uint head = is_query ? local : local - params.query_heads;
  device bfloat *row =
      is_query ? query + ulong(token) * params.query_token_stride +
                     ulong(head) * params.query_head_stride
               : key + ulong(token) * params.key_token_stride +
                     ulong(head) * params.key_head_stride;
  const uint half_rotary = params.rotary_dim / 2;
  const float position = float(params.start_position + token);
  for (uint pair = lane; pair < half_rotary; pair += 32) {
    const float first = float(row[pair]);
    const float second = float(row[pair + half_rotary]);
    const float exponent = float(2 * pair) / float(params.rotary_dim);
    const float angle = position / pow(params.theta, exponent);
    const float cosine = cos(angle), sine = sin(angle);
    row[pair] = bfloat(fma(-second, sine, first * cosine));
    row[pair + half_rotary] = bfloat(fma(first, sine, second * cosine));
  }
}
