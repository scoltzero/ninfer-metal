// Diagnostic-only pre-P05 contiguous KV control from ed07474.
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

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

inline uint paged_position(uint logical,
                           constant PagedAttentionParams &params) {
  const uint page = logical / params.page_tokens;
  const uint within = logical - page * params.page_tokens;
  return page * params.page_tokens + within;
}

kernel void
omlx_m5_contiguous_kv_append_control(device const bfloat *key [[buffer(0)]],
                                     device const bfloat *value [[buffer(1)]],
                                     device bfloat *cache_key [[buffer(2)]],
                                     device bfloat *cache_value [[buffer(3)]],
                                     constant PagedAttentionParams &params
                                     [[buffer(4)]],
                                     uint gid [[thread_position_in_grid]]) {
  const uint width = params.key_value_heads * params.head_dim;
  const uint values = params.tokens * width;
  if (gid >= values) {
    return;
  }
  const uint token = gid / width;
  const uint feature = gid - token * width;
  const uint position = paged_position(params.cache_start + token, params);
  cache_key[position * width + feature] =
      key[token * params.key_token_stride + feature];
  cache_value[position * width + feature] =
      value[token * params.value_token_stride + feature];
}

kernel void omlx_m5_contiguous_attention_control(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const bfloat *cache_key [[buffer(2)]],
    device const bfloat *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant PagedAttentionParams &params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint logical_head = tg_position.y;
  const uint token = logical_head / params.query_heads;
  if (token >= params.tokens)
    return;
  const uint query_head = logical_head - token * params.query_heads;
  const uint repeat = params.query_heads / params.key_value_heads;
  const uint kv_head = query_head / repeat;
  device const bfloat *query_row = query + ulong(token) * params.query_token_stride +
                                   ulong(query_head) * params.query_head_stride;
  device const bfloat *gate_row = gate + ulong(token) * params.gate_token_stride +
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

  const ulong head_offset = ulong(kv_head) * params.head_dim;
  for (uint logical = 0; logical < context_tokens; ++logical) {
    const ulong position = paged_position(logical, params);
    const ulong kv_offset = position * kv_width + head_offset;
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
