#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

struct PagedSplitAttentionParams {
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
  uint split_count;
  uint split_tokens;
};

kernel void omlx_m5_paged_attention_split_partial(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *cache_key [[buffer(1)]],
    device const bfloat *cache_value [[buffer(2)]],
    device float *partial_max [[buffer(3)]],
    device float *partial_sum [[buffer(4)]],
    device float *partial_output [[buffer(5)]],
    device const uint *page_table [[buffer(6)]],
    constant PagedSplitAttentionParams &params [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint split = tg_position.x;
  const uint logical_head = tg_position.y;
  const uint token = logical_head / params.query_heads;
  if (split >= params.split_count || token >= params.tokens)
    return;
  const uint query_head = logical_head - token * params.query_heads;
  const uint kv_head =
      query_head / (params.query_heads / params.key_value_heads);
  const ulong kv_width = ulong(params.key_value_heads) * params.head_dim;
  const uint first = split * params.split_tokens;
  const uint last =
      min(first + params.split_tokens, params.cache_start + token + 1);
  const ulong record = ulong(logical_head) * params.split_count + split;
  if (first >= last) {
    if (lane == 0) {
      partial_max[record] = -INFINITY;
      partial_sum[record] = 0.0f;
    }
    for (uint feature = lane; feature < params.head_dim; feature += 32)
      partial_output[record * params.head_dim + feature] = 0.0f;
    return;
  }
  device const bfloat *q = query + ulong(token) * params.query_token_stride +
                           ulong(query_head) * params.query_head_stride;
  const float scale = rsqrt(float(params.head_dim));
  float q_reg[8];
#pragma clang loop unroll(full)
  for (uint index = 0; index < 8; ++index) {
    const uint feature = lane + index * 32;
    q_reg[index] = feature < params.head_dim ? float(q[feature]) : 0.0f;
  }

  float running_max = -INFINITY;
  float denominator = 0.0f;
  float accumulators[8];
#pragma clang loop unroll(full)
  for (uint index = 0; index < 8; ++index)
    accumulators[index] = 0.0f;

  const uint final_page = (last + params.page_tokens - 1) / params.page_tokens;
  const ulong head_offset = ulong(kv_head) * params.head_dim;

  if (params.head_dim == 256) {
    for (uint page = first / params.page_tokens; page < final_page; ++page) {
      const ulong base = ulong(page_table[page]) * params.page_tokens;
      const uint begin =
          max(first, page * params.page_tokens) - page * params.page_tokens;
      const uint end =
          min(last, (page + 1) * params.page_tokens) - page * params.page_tokens;
      for (uint within = begin; within < end; ++within) {
        const ulong kv_offset = (base + within) * kv_width + head_offset;
        device const bfloat *k = cache_key + kv_offset;
        device const bfloat *v = cache_value + kv_offset;
        float partial = 0.0f;
#pragma clang loop unroll(full)
        for (uint index = 0; index < 8; ++index) {
          partial = fma(q_reg[index], float(k[lane + index * 32]), partial);
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
            accumulators[index] =
                fma(accumulators[index], rescale, float(v[lane + index * 32]));
          }
        } else if (score > -INFINITY) {
          const float probability =
              (score == running_max) ? 1.0f : exp(score - running_max);
          denominator += probability;
#pragma clang loop unroll(full)
          for (uint index = 0; index < 8; ++index) {
            accumulators[index] = fma(probability, float(v[lane + index * 32]),
                                      accumulators[index]);
          }
        }
      }
    }
  } else {
    for (uint page = first / params.page_tokens; page < final_page; ++page) {
      const ulong base = ulong(page_table[page]) * params.page_tokens;
      const uint begin =
          max(first, page * params.page_tokens) - page * params.page_tokens;
      const uint end =
          min(last, (page + 1) * params.page_tokens) - page * params.page_tokens;
      for (uint within = begin; within < end; ++within) {
        const ulong kv_offset = (base + within) * kv_width + head_offset;
        device const bfloat *k = cache_key + kv_offset;
        device const bfloat *v = cache_value + kv_offset;
        float partial = 0.0f;
#pragma clang loop unroll(full)
        for (uint index = 0; index < 8; ++index) {
          const uint feature = lane + index * 32;
          if (feature < params.head_dim)
            partial = fma(q_reg[index], float(k[feature]), partial);
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
            if (feature < params.head_dim)
              accumulators[index] =
                  fma(accumulators[index], rescale, float(v[feature]));
          }
        } else if (score > -INFINITY) {
          const float probability =
              (score == running_max) ? 1.0f : exp(score - running_max);
          denominator += probability;
#pragma clang loop unroll(full)
          for (uint index = 0; index < 8; ++index) {
            const uint feature = lane + index * 32;
            if (feature < params.head_dim)
              accumulators[index] =
                  fma(probability, float(v[feature]), accumulators[index]);
          }
        }
      }
    }
  }
  if (lane == 0) {
    partial_max[record] = running_max;
    partial_sum[record] = denominator;
  }
#pragma clang loop unroll(full)
  for (uint index = 0; index < 8; ++index) {
    const uint feature = lane + index * 32;
    if (feature < params.head_dim)
      partial_output[record * params.head_dim + feature] = accumulators[index];
  }
}

kernel void omlx_m5_paged_attention_split_merge(
    device const bfloat *gate [[buffer(0)]],
    device const float *partial_max [[buffer(1)]],
    device const float *partial_sum [[buffer(2)]],
    device const float *partial_output [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant PagedSplitAttentionParams &params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint logical_head = tg_position.y;
  const uint token = logical_head / params.query_heads;
  if (token >= params.tokens)
    return;
  const uint query_head = logical_head - token * params.query_heads;
  const ulong begin = ulong(logical_head) * params.split_count;
  float maximum = -INFINITY;
  for (uint split = 0; split < params.split_count; ++split)
    maximum = max(maximum, partial_max[begin + split]);
  float denominator = 0.0f;
  for (uint split = 0; split < params.split_count; ++split) {
    const float chunk_sum = partial_sum[begin + split];
    if (chunk_sum > 0.0f)
      denominator += chunk_sum * exp(partial_max[begin + split] - maximum);
  }
  device const bfloat *g = gate + ulong(token) * params.gate_token_stride +
                           ulong(query_head) * params.gate_head_stride;
  for (uint feature = lane; feature < params.head_dim; feature += 32) {
    float total = 0.0f;
    for (uint split = 0; split < params.split_count; ++split) {
      const float chunk_sum = partial_sum[begin + split];
      if (chunk_sum > 0.0f)
        total = fma(exp(partial_max[begin + split] - maximum),
                    partial_output[(begin + split) * params.head_dim + feature],
                    total);
    }
    const float gated = total / denominator / (1.0f + exp(-float(g[feature])));
    output[ulong(logical_head) * params.head_dim + feature] = bfloat(gated);
  }
}

// Diagnostic unfused gate control: keep the FP32 softmax/value result until
// the second dispatch so the rounding boundary matches the fused path.
kernel void omlx_m5_paged_attention_split_merge_ungated(
    device const float *partial_max [[buffer(0)]],
    device const float *partial_sum [[buffer(1)]],
    device const float *partial_output [[buffer(2)]],
    device float *ungated [[buffer(3)]],
    constant PagedSplitAttentionParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint logical_head = tg_position.y;
  const uint token = logical_head / params.query_heads;
  if (token >= params.tokens)
    return;
  const ulong begin = ulong(logical_head) * params.split_count;
  float maximum = -INFINITY;
  for (uint split = 0; split < params.split_count; ++split)
    maximum = max(maximum, partial_max[begin + split]);
  float denominator = 0.0f;
  for (uint split = 0; split < params.split_count; ++split) {
    const float sum = partial_sum[begin + split];
    if (sum > 0.0f)
      denominator += sum * exp(partial_max[begin + split] - maximum);
  }
  for (uint feature = lane; feature < params.head_dim; feature += 32) {
    float total = 0.0f;
    for (uint split = 0; split < params.split_count; ++split)
      if (partial_sum[begin + split] > 0.0f)
        total += partial_output[(begin + split) * params.head_dim + feature] *
                 exp(partial_max[begin + split] - maximum);
    ungated[ulong(logical_head) * params.head_dim + feature] =
        total / denominator;
  }
}
kernel void
omlx_m5_attention_apply_gate_fp32(device const float *ungated [[buffer(0)]],
                                  device const bfloat *gate [[buffer(1)]],
                                  device bfloat *output [[buffer(2)]],
                                  constant PagedSplitAttentionParams &params
                                  [[buffer(3)]],
                                  uint i [[thread_position_in_grid]]) {
  if (ulong(i) >= ulong(params.tokens) * params.query_heads * params.head_dim)
    return;
  const uint logical_head = i / params.head_dim;
  const uint token = logical_head / params.query_heads;
  const uint query_head = logical_head - token * params.query_heads;
  const uint feature = i % params.head_dim;
  const ulong gate_index = ulong(token) * params.gate_token_stride +
                           ulong(query_head) * params.gate_head_stride +
                           feature;
  output[i] = bfloat(ungated[i] / (1.0f + exp(-float(gate[gate_index]))));
}
