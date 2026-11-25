#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

struct GdnLayerParams {
  uint key_heads;
  uint value_heads;
  uint key_dim;
  uint value_dim;
  uint conv_width;
  uint conv_kernel;
  uint tokens;
  float epsilon;
};

kernel void omlx_m5_gdn_conv_silu(device const bfloat *qkv [[buffer(0)]],
                                  device const bfloat *conv_weight
                                  [[buffer(1)]],
                                  device bfloat *history [[buffer(2)]],
                                  device bfloat *convolved [[buffer(3)]],
                                  constant GdnLayerParams &params [[buffer(4)]],
                                  uint channel [[thread_position_in_grid]]) {
  if (channel >= params.conv_width) {
    return;
  }
  const uint history_tokens = params.conv_kernel - 1;
  for (uint token = 0; token < params.tokens; ++token) {
    float sum = 0.0f;
    for (uint tap = 0; tap < params.conv_kernel; ++tap) {
      const int source_token = int(token) + int(tap) - int(history_tokens);
      const float input =
          source_token < 0
              ? float(history[ulong(uint(source_token + int(history_tokens))) *
                                  params.conv_width +
                              channel])
              : float(qkv[ulong(uint(source_token)) * params.conv_width +
                          channel]);
      sum = fma(
          input,
          float(conv_weight[ulong(channel) * params.conv_kernel + tap]), sum);
    }
    convolved[ulong(token) * params.conv_width + channel] =
        bfloat(sum / (1.0f + exp(-sum)));
  }
  if (params.tokens == 0) {
    return;
  }
  for (uint history_token = 0; history_token < history_tokens;
       ++history_token) {
    const int source_token =
        int(params.tokens) - int(history_tokens) + int(history_token);
    history[ulong(history_token) * params.conv_width + channel] =
        source_token < 0
            ? history[ulong(uint(source_token + int(history_tokens))) *
                          params.conv_width +
                      channel]
            : qkv[ulong(uint(source_token)) * params.conv_width + channel];
  }
}

// 2D parallel depthwise causal conv + SiLU over (channel, token) for T >= 4.
// `history` is strictly read-only here so zero inter-threadgroup WAR aliasing
// can occur; `omlx_m5_gdn_commit_history` updates `history` after the barrier.
kernel void omlx_m5_gdn_conv_silu_2d(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device const bfloat *history [[buffer(2)]],
    device bfloat *convolved [[buffer(3)]],
    constant GdnLayerParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint channel = gid.x;
  const uint token = gid.y;
  if (channel >= params.conv_width || token >= params.tokens) {
    return;
  }
  const uint history_tokens = params.conv_kernel - 1;
  float sum = 0.0f;
  for (uint tap = 0; tap < params.conv_kernel; ++tap) {
    const int source_token = int(token) + int(tap) - int(history_tokens);
    const float input =
        source_token < 0
            ? float(history[ulong(uint(source_token + int(history_tokens))) *
                                params.conv_width +
                            channel])
            : float(
                  qkv[ulong(uint(source_token)) * params.conv_width + channel]);
    sum = fma(input,
              float(conv_weight[ulong(channel) * params.conv_kernel + tap]),
              sum);
  }
  convolved[ulong(token) * params.conv_width + channel] =
      bfloat(sum / (1.0f + exp(-sum)));
}

// Coalesced bfloat4 x 8-token register-sliding-window causal conv + SiLU (0 B TG SRAM,
// 0-ULP bit-identical to `omlx_m5_gdn_conv_silu_2d` & `omlx_m5_gdn_conv_silu`).
kernel void omlx_m5_gdn_conv_silu_vec4_t8(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device const bfloat *history [[buffer(2)]],
    device bfloat *convolved [[buffer(3)]],
    constant GdnLayerParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint ch0 = gid.x * 4u;
  const uint t_base = gid.y * 8u;
  if (ch0 >= params.conv_width || t_base >= params.tokens) {
    return;
  }
  const uint count = min(8u, params.tokens - t_base);
  if (params.conv_kernel == 4u && (params.conv_width & 3u) == 0u) {
    device const bfloat4 *wp = reinterpret_cast<device const bfloat4 *>(
        conv_weight + ulong(ch0) * 4ul);
    const float4 c0 = float4(wp[0]);
    const float4 c1 = float4(wp[1]);
    const float4 c2 = float4(wp[2]);
    const float4 c3 = float4(wp[3]);
    const float4 w0(c0.x, c1.x, c2.x, c3.x);
    const float4 w1(c0.y, c1.y, c2.y, c3.y);
    const float4 w2(c0.z, c1.z, c2.z, c3.z);
    const float4 w3(c0.w, c1.w, c2.w, c3.w);

    auto load_row = [&](int src_tok) -> float4 {
      device const bfloat *p =
          src_tok < 0
              ? (history + ulong(uint(src_tok + 3)) * ulong(params.conv_width) +
                 ulong(ch0))
              : (qkv + ulong(uint(src_tok)) * ulong(params.conv_width) +
                 ulong(ch0));
      return float4(*reinterpret_cast<device const bfloat4 *>(p));
    };

    float4 xm3 = load_row(int(t_base) - 3);
    float4 xm2 = load_row(int(t_base) - 2);
    float4 xm1 = load_row(int(t_base) - 1);

    for (uint s = 0u; s < count; ++s) {
      const uint tok = t_base + s;
      const float4 xcur = float4(*reinterpret_cast<device const bfloat4 *>(
          qkv + ulong(tok) * ulong(params.conv_width) + ulong(ch0)));
      float4 sum = fma(xm3, w0, float4(0.0f));
      sum = fma(xm2, w1, sum);
      sum = fma(xm1, w2, sum);
      sum = fma(xcur, w3, sum);
      *reinterpret_cast<device bfloat4 *>(
          convolved + ulong(tok) * ulong(params.conv_width) + ulong(ch0)) =
          bfloat4(sum / (1.0f + exp(-sum)));
      xm3 = xm2;
      xm2 = xm1;
      xm1 = xcur;
    }
    return;
  }

  const uint history_tokens = params.conv_kernel - 1u;
  for (uint s = 0u; s < count; ++s) {
    const uint tok = t_base + s;
    for (uint j = 0u; j < 4u && (ch0 + j) < params.conv_width; ++j) {
      const uint ch = ch0 + j;
      float sum = 0.0f;
      for (uint tap = 0u; tap < params.conv_kernel; ++tap) {
        const int src_tok = int(tok) + int(tap) - int(history_tokens);
        const float inp =
            src_tok < 0
                ? float(history[ulong(uint(src_tok + int(history_tokens))) *
                                    params.conv_width +
                                ch])
                : float(qkv[ulong(uint(src_tok)) * params.conv_width + ch]);
        sum = fma(inp,
                  float(conv_weight[ulong(ch) * params.conv_kernel + tap]),
                  sum);
      }
      convolved[ulong(tok) * params.conv_width + ch] =
          bfloat(sum / (1.0f + exp(-sum)));
    }
  }
}

kernel void omlx_m5_gdn_recurrent(device const bfloat *convolved [[buffer(0)]],
                                  device const bfloat *a [[buffer(1)]],
                                  device const bfloat *b [[buffer(2)]],
                                  device const bfloat *a_log [[buffer(3)]],
                                  device const bfloat *dt_bias [[buffer(4)]],
                                  device float *state [[buffer(5)]],
                                  device bfloat *recurrent [[buffer(6)]],
                                  constant GdnLayerParams &params [[buffer(7)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint3 tg_position
                                  [[threadgroup_position_in_grid]]) {
  const uint row_index = tg_position.y;
  const uint value_width = params.value_heads * params.value_dim;
  if (row_index >= value_width || params.tokens == 0)
    return;
  const uint value_head = row_index / params.value_dim;
  const uint value_index = row_index - value_head * params.value_dim;
  const uint repeat = params.value_heads / params.key_heads;
  const uint key_head = value_head / repeat;
  const uint key_width = params.key_heads * params.key_dim;
  device float *state_row = state + ulong(row_index) * params.key_dim;
  const float inverse_scale = rsqrt(float(params.key_dim));
  const float dt_bias_val = lane == 0 ? float(dt_bias[value_head]) : 0.0f;
  const float neg_exp_a = lane == 0 ? -exp(float(a_log[value_head])) : 0.0f;
  const ulong q_head_offset = ulong(key_head) * params.key_dim;
  const ulong k_head_offset = ulong(key_width) + q_head_offset;
  const ulong v_elem_offset =
      ulong(2 * key_width) + ulong(value_head * params.value_dim + value_index);

  if (params.key_dim == 128) {
    float s0 = state_row[lane];
    float s1 = state_row[lane + 32];
    float s2 = state_row[lane + 64];
    float s3 = state_row[lane + 96];

    for (uint token = 0; token < params.tokens; ++token) {
      device const bfloat *token_row =
          convolved + ulong(token) * params.conv_width;
      device const bfloat *query = token_row + q_head_offset;
      device const bfloat *key = token_row + k_head_offset;

      const float q0 = float(query[lane]);
      const float q1 = float(query[lane + 32]);
      const float q2 = float(query[lane + 64]);
      const float q3 = float(query[lane + 96]);
      const float k0 = float(key[lane]);
      const float k1 = float(key[lane + 32]);
      const float k2 = float(key[lane + 64]);
      const float k3 = float(key[lane + 96]);

      float query_square = fma(q0, q0, 0.0f);
      query_square = fma(q1, q1, query_square);
      query_square = fma(q2, q2, query_square);
      query_square = fma(q3, q3, query_square);

      float key_square = fma(k0, k0, 0.0f);
      key_square = fma(k1, k1, key_square);
      key_square = fma(k2, k2, key_square);
      key_square = fma(k3, k3, key_square);

      const float query_inverse =
          rsqrt(simd_sum(query_square) / float(params.key_dim) + 1.0e-6f);
      const float key_inverse =
          rsqrt(simd_sum(key_square) / float(params.key_dim) + 1.0e-6f);

      float decay = 0.0f;
      float beta = 0.0f;
      float v_val = 0.0f;
      if (lane == 0) {
        const ulong h_idx = ulong(token) * params.value_heads + value_head;
        const float activation = float(a[h_idx]) + dt_bias_val;
        const float softplus =
            activation > 20.0f ? activation : log(1.0f + exp(activation));
        decay = exp(neg_exp_a * softplus);
        beta = 1.0f / (1.0f + exp(-float(b[h_idx])));
        v_val = float(token_row[v_elem_offset]);
      }
      decay = simd_broadcast_first(decay);
      beta = simd_broadcast_first(beta);
      v_val = simd_broadcast_first(v_val);

      const float nk0 = k0 * key_inverse * inverse_scale;
      const float nk1 = k1 * key_inverse * inverse_scale;
      const float nk2 = k2 * key_inverse * inverse_scale;
      const float nk3 = k3 * key_inverse * inverse_scale;

      s0 *= decay;
      float memory = fma(s0, nk0, 0.0f);
      s1 *= decay;
      memory = fma(s1, nk1, memory);
      s2 *= decay;
      memory = fma(s2, nk2, memory);
      s3 *= decay;
      memory = fma(s3, nk3, memory);
      memory = simd_sum(memory);

      const float delta = (v_val - memory) * beta;

      s0 = fma(nk0, delta, s0);
      float projected =
          fma(s0, q0 * query_inverse * inverse_scale * inverse_scale, 0.0f);
      s1 = fma(nk1, delta, s1);
      projected = fma(s1, q1 * query_inverse * inverse_scale * inverse_scale,
                      projected);
      s2 = fma(nk2, delta, s2);
      projected = fma(s2, q2 * query_inverse * inverse_scale * inverse_scale,
                      projected);
      s3 = fma(nk3, delta, s3);
      projected = fma(s3, q3 * query_inverse * inverse_scale * inverse_scale,
                      projected);
      projected = simd_sum(projected);
      if (lane == 0) {
        recurrent[ulong(token) * value_width + row_index] = bfloat(projected);
      }
    }

    state_row[lane] = s0;
    state_row[lane + 32] = s1;
    state_row[lane + 64] = s2;
    state_row[lane + 96] = s3;
    return;
  }

  float s_reg[8];
#pragma clang loop unroll(full)
  for (uint idx = 0; idx < 8; ++idx) {
    const uint feature = lane + idx * 32;
    s_reg[idx] = feature < params.key_dim ? state_row[feature] : 0.0f;
  }

  for (uint token = 0; token < params.tokens; ++token) {
    device const bfloat *token_row =
        convolved + ulong(token) * params.conv_width;
    device const bfloat *query = token_row + q_head_offset;
    device const bfloat *key = token_row + k_head_offset;

    float q_reg[8];
    float k_reg[8];
    float query_square = 0.0f;
    float key_square = 0.0f;
#pragma clang loop unroll(full)
    for (uint idx = 0; idx < 8; ++idx) {
      const uint feature = lane + idx * 32;
      if (feature < params.key_dim) {
        const float q = float(query[feature]);
        const float k = float(key[feature]);
        q_reg[idx] = q;
        k_reg[idx] = k;
        query_square = fma(q, q, query_square);
        key_square = fma(k, k, key_square);
      } else {
        q_reg[idx] = 0.0f;
        k_reg[idx] = 0.0f;
      }
    }
    const float query_inverse =
        rsqrt(simd_sum(query_square) / float(params.key_dim) + 1.0e-6f);
    const float key_inverse =
        rsqrt(simd_sum(key_square) / float(params.key_dim) + 1.0e-6f);

    float decay = 0.0f;
    float beta = 0.0f;
    float v_val = 0.0f;
    if (lane == 0) {
      const ulong h_idx = ulong(token) * params.value_heads + value_head;
      const float activation = float(a[h_idx]) + dt_bias_val;
      const float softplus =
          activation > 20.0f ? activation : log(1.0f + exp(activation));
      decay = exp(neg_exp_a * softplus);
      beta = 1.0f / (1.0f + exp(-float(b[h_idx])));
      v_val = float(token_row[v_elem_offset]);
    }
    decay = simd_broadcast_first(decay);
    beta = simd_broadcast_first(beta);
    v_val = simd_broadcast_first(v_val);

    float memory = 0.0f;
#pragma clang loop unroll(full)
    for (uint idx = 0; idx < 8; ++idx) {
      const uint feature = lane + idx * 32;
      if (feature < params.key_dim) {
        s_reg[idx] *= decay;
        const float normalized_key = k_reg[idx] * key_inverse * inverse_scale;
        memory = fma(s_reg[idx], normalized_key, memory);
      }
    }
    memory = simd_sum(memory);
    const float delta = (v_val - memory) * beta;
    float projected = 0.0f;
#pragma clang loop unroll(full)
    for (uint idx = 0; idx < 8; ++idx) {
      const uint feature = lane + idx * 32;
      if (feature < params.key_dim) {
        const float normalized_key = k_reg[idx] * key_inverse * inverse_scale;
        s_reg[idx] = fma(normalized_key, delta, s_reg[idx]);
        const float normalized_query =
            q_reg[idx] * query_inverse * inverse_scale * inverse_scale;
        projected = fma(s_reg[idx], normalized_query, projected);
      }
    }
    projected = simd_sum(projected);
    if (lane == 0) {
      recurrent[ulong(token) * value_width + row_index] = bfloat(projected);
    }
  }

#pragma clang loop unroll(full)
  for (uint idx = 0; idx < 8; ++idx) {
    const uint feature = lane + idx * 32;
    if (feature < params.key_dim) {
      state_row[feature] = s_reg[idx];
    }
  }
}

// Exact-recurrence TB=8 threadgroup-staged GDN prefill (`key_dim == 128`,
// `value_dim % 16 == 0`). 4 SIMDgroups (128 threads/TG) own 16 consecutive
// `value_index` rows (`4` rows/SG), staging 8 normalized Q/K vectors, V slices,
// and gates in threadgroup SRAM while executing the exact 32-lane `simd_sum`
// FP32 recurrence (`0 ULP` bit-identical to `omlx_m5_gdn_recurrent`).
kernel void omlx_m5_gdn_recurrent_tb8(
    device const bfloat *convolved [[buffer(0)]],
    device const bfloat *a [[buffer(1)]],
    device const bfloat *b [[buffer(2)]],
    device const bfloat *a_log [[buffer(3)]],
    device const bfloat *dt_bias [[buffer(4)]],
    device float *state [[buffer(5)]],
    device bfloat *recurrent [[buffer(6)]],
    constant GdnLayerParams &params [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint row_block_base = tg_position.y * 16u;
  const uint value_width = params.value_heads * params.value_dim;
  if (row_block_base >= value_width || params.tokens == 0) {
    return;
  }
  const uint tid = sg * 32u + lane;
  const uint value_head = row_block_base / params.value_dim;
  const uint dv0 = row_block_base - value_head * params.value_dim;
  const uint repeat = params.value_heads / params.key_heads;
  const uint key_head = value_head / repeat;
  const uint key_width = params.key_heads * 128u;
  const float inverse_scale = rsqrt(128.0f);
  const float dt_bias_val = float(dt_bias[value_head]);
  const float neg_exp_a = -exp(float(a_log[value_head]));
  const ulong q_head_offset = ulong(key_head) * 128u;
  const ulong k_head_offset = ulong(key_width) + q_head_offset;
  const ulong v_block_offset =
      ulong(2u * key_width) + ulong(value_head * params.value_dim + dv0);

  threadgroup float tg_nk[8][128];
  threadgroup float tg_nq[8][128];
  threadgroup float tg_v[8][16];
  threadgroup float tg_decay[8];
  threadgroup float tg_beta[8];

  const uint row_sg = row_block_base + sg * 4u;
  device float *st0_ptr = state + ulong(row_sg + 0u) * 128u;
  device float *st1_ptr = state + ulong(row_sg + 1u) * 128u;
  device float *st2_ptr = state + ulong(row_sg + 2u) * 128u;
  device float *st3_ptr = state + ulong(row_sg + 3u) * 128u;

  float r0_s0 = st0_ptr[lane], r0_s1 = st0_ptr[lane + 32u],
        r0_s2 = st0_ptr[lane + 64u], r0_s3 = st0_ptr[lane + 96u];
  float r1_s0 = st1_ptr[lane], r1_s1 = st1_ptr[lane + 32u],
        r1_s2 = st1_ptr[lane + 64u], r1_s3 = st1_ptr[lane + 96u];
  float r2_s0 = st2_ptr[lane], r2_s1 = st2_ptr[lane + 32u],
        r2_s2 = st2_ptr[lane + 64u], r2_s3 = st2_ptr[lane + 96u];
  float r3_s0 = st3_ptr[lane], r3_s1 = st3_ptr[lane + 32u],
        r3_s2 = st3_ptr[lane + 64u], r3_s3 = st3_ptr[lane + 96u];

  for (uint t0 = 0; t0 < params.tokens; t0 += 8u) {
    const uint tt = min(8u, params.tokens - t0);

    // Cooperatively normalize Q and K for up to 8 tokens (1 SIMDgroup per token).
    for (uint sub = sg; sub < tt; sub += 4u) {
      device const bfloat *token_row =
          convolved + ulong(t0 + sub) * params.conv_width;
      device const bfloat *query = token_row + q_head_offset;
      device const bfloat *key = token_row + k_head_offset;

      const float q0 = float(query[lane]);
      const float q1 = float(query[lane + 32u]);
      const float q2 = float(query[lane + 64u]);
      const float q3 = float(query[lane + 96u]);
      const float k0 = float(key[lane]);
      const float k1 = float(key[lane + 32u]);
      const float k2 = float(key[lane + 64u]);
      const float k3 = float(key[lane + 96u]);

      float query_square = fma(q0, q0, 0.0f);
      query_square = fma(q1, q1, query_square);
      query_square = fma(q2, q2, query_square);
      query_square = fma(q3, q3, query_square);

      float key_square = fma(k0, k0, 0.0f);
      key_square = fma(k1, k1, key_square);
      key_square = fma(k2, k2, key_square);
      key_square = fma(k3, k3, key_square);

      const float query_inverse =
          rsqrt(simd_sum(query_square) / 128.0f + 1.0e-6f);
      const float key_inverse = rsqrt(simd_sum(key_square) / 128.0f + 1.0e-6f);

      tg_nk[sub][lane] = k0 * key_inverse * inverse_scale;
      tg_nk[sub][lane + 32u] = k1 * key_inverse * inverse_scale;
      tg_nk[sub][lane + 64u] = k2 * key_inverse * inverse_scale;
      tg_nk[sub][lane + 96u] = k3 * key_inverse * inverse_scale;

      tg_nq[sub][lane] = q0 * query_inverse * inverse_scale * inverse_scale;
      tg_nq[sub][lane + 32u] =
          q1 * query_inverse * inverse_scale * inverse_scale;
      tg_nq[sub][lane + 64u] =
          q2 * query_inverse * inverse_scale * inverse_scale;
      tg_nq[sub][lane + 96u] =
          q3 * query_inverse * inverse_scale * inverse_scale;
    }

    if (tid < tt) {
      const ulong h_idx = ulong(t0 + tid) * params.value_heads + value_head;
      const float activation = float(a[h_idx]) + dt_bias_val;
      const float softplus =
          activation > 20.0f ? activation : log(1.0f + exp(activation));
      tg_decay[tid] = exp(neg_exp_a * softplus);
      tg_beta[tid] = 1.0f / (1.0f + exp(-float(b[h_idx])));
    }

    for (uint idx = tid; idx < tt * 16u; idx += 128u) {
      const uint r = idx >> 4u;
      const uint c = idx & 15u;
      tg_v[r][c] = float(
          convolved[ulong(t0 + r) * params.conv_width + v_block_offset + c]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = 0; r < tt; ++r) {
      const float decay = tg_decay[r];
      const float beta = tg_beta[r];
      const float nk0 = tg_nk[r][lane];
      const float nk1 = tg_nk[r][lane + 32u];
      const float nk2 = tg_nk[r][lane + 64u];
      const float nk3 = tg_nk[r][lane + 96u];
      const float nq0 = tg_nq[r][lane];
      const float nq1 = tg_nq[r][lane + 32u];
      const float nq2 = tg_nq[r][lane + 64u];
      const float nq3 = tg_nq[r][lane + 96u];
      const ulong out_base = ulong(t0 + r) * value_width + row_sg;

      // Row 0
      {
        const float v_val = tg_v[r][sg * 4u + 0u];
        r0_s0 *= decay;
        float mem = fma(r0_s0, nk0, 0.0f);
        r0_s1 *= decay;
        mem = fma(r0_s1, nk1, mem);
        r0_s2 *= decay;
        mem = fma(r0_s2, nk2, mem);
        r0_s3 *= decay;
        mem = fma(r0_s3, nk3, mem);
        const float delta = (v_val - simd_sum(mem)) * beta;
        r0_s0 = fma(nk0, delta, r0_s0);
        float proj = fma(r0_s0, nq0, 0.0f);
        r0_s1 = fma(nk1, delta, r0_s1);
        proj = fma(r0_s1, nq1, proj);
        r0_s2 = fma(nk2, delta, r0_s2);
        proj = fma(r0_s2, nq2, proj);
        r0_s3 = fma(nk3, delta, r0_s3);
        proj = fma(r0_s3, nq3, proj);
        proj = simd_sum(proj);
        if (lane == 0)
          recurrent[out_base + 0u] = bfloat(proj);
      }
      // Row 1
      {
        const float v_val = tg_v[r][sg * 4u + 1u];
        r1_s0 *= decay;
        float mem = fma(r1_s0, nk0, 0.0f);
        r1_s1 *= decay;
        mem = fma(r1_s1, nk1, mem);
        r1_s2 *= decay;
        mem = fma(r1_s2, nk2, mem);
        r1_s3 *= decay;
        mem = fma(r1_s3, nk3, mem);
        const float delta = (v_val - simd_sum(mem)) * beta;
        r1_s0 = fma(nk0, delta, r1_s0);
        float proj = fma(r1_s0, nq0, 0.0f);
        r1_s1 = fma(nk1, delta, r1_s1);
        proj = fma(r1_s1, nq1, proj);
        r1_s2 = fma(nk2, delta, r1_s2);
        proj = fma(r1_s2, nq2, proj);
        r1_s3 = fma(nk3, delta, r1_s3);
        proj = fma(r1_s3, nq3, proj);
        proj = simd_sum(proj);
        if (lane == 0)
          recurrent[out_base + 1u] = bfloat(proj);
      }
      // Row 2
      {
        const float v_val = tg_v[r][sg * 4u + 2u];
        r2_s0 *= decay;
        float mem = fma(r2_s0, nk0, 0.0f);
        r2_s1 *= decay;
        mem = fma(r2_s1, nk1, mem);
        r2_s2 *= decay;
        mem = fma(r2_s2, nk2, mem);
        r2_s3 *= decay;
        mem = fma(r2_s3, nk3, mem);
        const float delta = (v_val - simd_sum(mem)) * beta;
        r2_s0 = fma(nk0, delta, r2_s0);
        float proj = fma(r2_s0, nq0, 0.0f);
        r2_s1 = fma(nk1, delta, r2_s1);
        proj = fma(r2_s1, nq1, proj);
        r2_s2 = fma(nk2, delta, r2_s2);
        proj = fma(r2_s2, nq2, proj);
        r2_s3 = fma(nk3, delta, r2_s3);
        proj = fma(r2_s3, nq3, proj);
        proj = simd_sum(proj);
        if (lane == 0)
          recurrent[out_base + 2u] = bfloat(proj);
      }
      // Row 3
      {
        const float v_val = tg_v[r][sg * 4u + 3u];
        r3_s0 *= decay;
        float mem = fma(r3_s0, nk0, 0.0f);
        r3_s1 *= decay;
        mem = fma(r3_s1, nk1, mem);
        r3_s2 *= decay;
        mem = fma(r3_s2, nk2, mem);
        r3_s3 *= decay;
        mem = fma(r3_s3, nk3, mem);
        const float delta = (v_val - simd_sum(mem)) * beta;
        r3_s0 = fma(nk0, delta, r3_s0);
        float proj = fma(r3_s0, nq0, 0.0f);
        r3_s1 = fma(nk1, delta, r3_s1);
        proj = fma(r3_s1, nq1, proj);
        r3_s2 = fma(nk2, delta, r3_s2);
        proj = fma(r3_s2, nq2, proj);
        r3_s3 = fma(nk3, delta, r3_s3);
        proj = fma(r3_s3, nq3, proj);
        proj = simd_sum(proj);
        if (lane == 0)
          recurrent[out_base + 3u] = bfloat(proj);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  st0_ptr[lane] = r0_s0;
  st0_ptr[lane + 32u] = r0_s1;
  st0_ptr[lane + 64u] = r0_s2;
  st0_ptr[lane + 96u] = r0_s3;

  st1_ptr[lane] = r1_s0;
  st1_ptr[lane + 32u] = r1_s1;
  st1_ptr[lane + 64u] = r1_s2;
  st1_ptr[lane + 96u] = r1_s3;

  st2_ptr[lane] = r2_s0;
  st2_ptr[lane + 32u] = r2_s1;
  st2_ptr[lane + 64u] = r2_s2;
  st2_ptr[lane + 96u] = r2_s3;

  st3_ptr[lane] = r3_s0;
  st3_ptr[lane + 32u] = r3_s1;
  st3_ptr[lane + 64u] = r3_s2;
  st3_ptr[lane + 96u] = r3_s3;
}

// P07 diagnostic candidates: keep the row-wise FP32 recurrence and token order.
// Only hoist operands whose ownership is shared by many value rows. These
// kernels are not selected by QwenForward without a full-slice qualification.
kernel void omlx_m5_gdn_prepare_qk_inverse(
    device const bfloat *convolved [[buffer(0)]],
    device float *inverse [[buffer(1)]],
    constant GdnLayerParams &p [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  const uint token = group.y / p.key_heads;
  if (token >= p.tokens) return;
  const uint head = group.y % p.key_heads;
  const uint key_width = p.key_heads * p.key_dim;
  device const bfloat *q =
      convolved + ulong(token) * p.conv_width + ulong(head) * p.key_dim;
  device const bfloat *k = q + key_width;
  float q2 = 0.0f, k2 = 0.0f;
  for (uint d = lane; d < p.key_dim; d += 32) {
    float qv = float(q[d]), kv = float(k[d]);
    q2 = fma(qv, qv, q2);
    k2 = fma(kv, kv, k2);
  }
  const float qi = rsqrt(simd_sum(q2) / float(p.key_dim) + 1.0e-6f);
  const float ki = rsqrt(simd_sum(k2) / float(p.key_dim) + 1.0e-6f);
  if (lane == 0) {
    const ulong offset = (ulong(token) * p.key_heads + head) * 2;
    inverse[offset] = qi;
    inverse[offset + 1] = ki;
  }
}

kernel void omlx_m5_gdn_prepare_gate(
    device const bfloat *a [[buffer(0)]],
    device const bfloat *b [[buffer(1)]],
    device const bfloat *a_log [[buffer(2)]],
    device const bfloat *dt_bias [[buffer(3)]],
    device float *coeff [[buffer(4)]],
    constant GdnLayerParams &p [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
  if (ulong(i) >= ulong(p.tokens) * p.value_heads) return;
  const uint head = i % p.value_heads;
  const float activation = float(a[i]) + float(dt_bias[head]);
  const float softplus = activation > 20.0f
      ? activation : log(1.0f + exp(activation));
  coeff[ulong(i) * 2] = exp(-exp(float(a_log[head])) * softplus);
  coeff[ulong(i) * 2 + 1] = 1.0f / (1.0f + exp(-float(b[i])));
}

template <bool ShareQK, bool ShareGate>
inline void gdn_recurrent_reuse(
    device const bfloat *convolved, device const bfloat *a,
    device const bfloat *b, device const bfloat *a_log,
    device const bfloat *dt_bias, device float *state,
    device bfloat *recurrent, device const float *qk_inverse,
    device const float *gate_coeff, constant GdnLayerParams &p,
    uint lane, uint row_index) {
  const uint value_width = p.value_heads * p.value_dim;
  if (row_index >= value_width || p.tokens == 0)
    return;
  const uint value_head = row_index / p.value_dim;
  const uint value_index = row_index - value_head * p.value_dim;
  const uint key_head = value_head / (p.value_heads / p.key_heads);
  const uint key_width = p.key_heads * p.key_dim;
  device float *state_row = state + ulong(row_index) * p.key_dim;
  const float inverse_scale = rsqrt(float(p.key_dim));
  const float dt_bias_val =
      (!ShareGate && lane == 0) ? float(dt_bias[value_head]) : 0.0f;
  const float neg_exp_a =
      (!ShareGate && lane == 0) ? -exp(float(a_log[value_head])) : 0.0f;
  const ulong q_head_offset = ulong(key_head) * p.key_dim;
  const ulong k_head_offset = ulong(key_width) + q_head_offset;
  const ulong v_elem_offset =
      ulong(2 * key_width) + ulong(value_head * p.value_dim + value_index);

  float s_reg[8];
#pragma clang loop unroll(full)
  for (uint idx = 0; idx < 8; ++idx) {
    const uint d = lane + idx * 32;
    s_reg[idx] = d < p.key_dim ? state_row[d] : 0.0f;
  }

  for (uint token = 0; token < p.tokens; ++token) {
    device const bfloat *token_row = convolved + ulong(token) * p.conv_width;
    device const bfloat *query = token_row + q_head_offset;
    device const bfloat *key = token_row + k_head_offset;
    float q_reg[8];
    float k_reg[8];
    float query_square = 0.0f, key_square = 0.0f;
#pragma clang loop unroll(full)
    for (uint idx = 0; idx < 8; ++idx) {
      const uint d = lane + idx * 32;
      if (d < p.key_dim) {
        const float q = float(query[d]), k = float(key[d]);
        q_reg[idx] = q;
        k_reg[idx] = k;
        if constexpr (!ShareQK) {
          query_square = fma(q, q, query_square);
          key_square = fma(k, k, key_square);
        }
      } else {
        q_reg[idx] = 0.0f;
        k_reg[idx] = 0.0f;
      }
    }
    float query_inverse, key_inverse;
    if constexpr (ShareQK) {
      const ulong i = (ulong(token) * p.key_heads + key_head) * 2;
      query_inverse = qk_inverse[i];
      key_inverse = qk_inverse[i + 1];
    } else {
      query_inverse = rsqrt(simd_sum(query_square) / float(p.key_dim) + 1.0e-6f);
      key_inverse = rsqrt(simd_sum(key_square) / float(p.key_dim) + 1.0e-6f);
    }
    float decay = 0.0f, beta = 0.0f, v_val = 0.0f;
    if (lane == 0) {
      if constexpr (ShareGate) {
        const ulong i = (ulong(token) * p.value_heads + value_head) * 2;
        decay = gate_coeff[i];
        beta = gate_coeff[i + 1];
      } else {
        const ulong h_idx = ulong(token) * p.value_heads + value_head;
        const float activation = float(a[h_idx]) + dt_bias_val;
        const float softplus = activation > 20.0f
            ? activation : log(1.0f + exp(activation));
        decay = exp(neg_exp_a * softplus);
        beta = 1.0f / (1.0f + exp(-float(b[h_idx])));
      }
      v_val = float(token_row[v_elem_offset]);
    }
    decay = simd_broadcast_first(decay);
    beta = simd_broadcast_first(beta);
    v_val = simd_broadcast_first(v_val);

    float memory = 0.0f;
#pragma clang loop unroll(full)
    for (uint idx = 0; idx < 8; ++idx) {
      const uint d = lane + idx * 32;
      if (d < p.key_dim) {
        s_reg[idx] *= decay;
        const float normalized_key = k_reg[idx] * key_inverse * inverse_scale;
        memory = fma(s_reg[idx], normalized_key, memory);
      }
    }
    memory = simd_sum(memory);
    const float delta = (v_val - memory) * beta;
    float projected = 0.0f;
#pragma clang loop unroll(full)
    for (uint idx = 0; idx < 8; ++idx) {
      const uint d = lane + idx * 32;
      if (d < p.key_dim) {
        const float normalized_key = k_reg[idx] * key_inverse * inverse_scale;
        s_reg[idx] = fma(normalized_key, delta, s_reg[idx]);
        const float normalized_query =
            q_reg[idx] * query_inverse * inverse_scale * inverse_scale;
        projected = fma(s_reg[idx], normalized_query, projected);
      }
    }
    projected = simd_sum(projected);
    if (lane == 0)
      recurrent[ulong(token) * value_width + row_index] = bfloat(projected);
  }

#pragma clang loop unroll(full)
  for (uint idx = 0; idx < 8; ++idx) {
    const uint d = lane + idx * 32;
    if (d < p.key_dim) {
      state_row[d] = s_reg[idx];
    }
  }
}

#define GDN_REUSE_KERNEL(NAME, QK, GATE)                                      \
kernel void NAME(                                                               \
    device const bfloat *convolved [[buffer(0)]],                                \
    device const bfloat *a [[buffer(1)]], device const bfloat *b [[buffer(2)]],  \
    device const bfloat *a_log [[buffer(3)]],                                     \
    device const bfloat *dt_bias [[buffer(4)]],                                   \
    device float *state [[buffer(5)]], device bfloat *recurrent [[buffer(6)]],   \
    device const float *qk_inverse [[buffer(7)]],                                 \
    device const float *gate_coeff [[buffer(8)]],                                 \
    constant GdnLayerParams &p [[buffer(9)]],                                     \
    uint lane [[thread_index_in_simdgroup]],                                     \
    uint3 group [[threadgroup_position_in_grid]]) {                              \
  gdn_recurrent_reuse<QK, GATE>(convolved, a, b, a_log, dt_bias, state,          \
                                recurrent, qk_inverse, gate_coeff, p, lane,     \
                                group.y);                                        \
}
GDN_REUSE_KERNEL(omlx_m5_gdn_recurrent_share_qk, true, false)
GDN_REUSE_KERNEL(omlx_m5_gdn_recurrent_share_gate, false, true)
GDN_REUSE_KERNEL(omlx_m5_gdn_recurrent_share_both, true, true)
#undef GDN_REUSE_KERNEL

kernel void omlx_m5_gdn_norm_gate(device const bfloat *recurrent [[buffer(0)]],
                                  device const bfloat *z [[buffer(1)]],
                                  device const bfloat *norm_weight
                                  [[buffer(2)]],
                                  device bfloat *output [[buffer(3)]],
                                  constant GdnLayerParams &params [[buffer(4)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint3 tg_position
                                  [[threadgroup_position_in_grid]]) {
  const uint logical_head = tg_position.y;
  const uint token = logical_head / params.value_heads;
  if (token >= params.tokens)
    return;
  const uint value_head = logical_head - token * params.value_heads;
  const ulong value_width = ulong(params.value_heads) * params.value_dim;
  device const bfloat *row =
      recurrent + ulong(token) * value_width + ulong(value_head) * params.value_dim;
  device const bfloat *gate =
      z + ulong(token) * value_width + ulong(value_head) * params.value_dim;
  device bfloat *destination =
      output + ulong(token) * value_width + ulong(value_head) * params.value_dim;
  float square_sum = 0.0f;
  for (uint feature = lane; feature < params.value_dim; feature += 32) {
    const float value = float(row[feature]);
    square_sum = fma(value, value, square_sum);
  }
  const float inverse =
      rsqrt(simd_sum(square_sum) / float(params.value_dim) + params.epsilon);
  for (uint feature = lane; feature < params.value_dim; feature += 32) {
    const float gate_value = float(gate[feature]);
    const float normalized =
        float(row[feature]) * inverse * float(norm_weight[feature]);
    destination[feature] =
        bfloat(normalized * gate_value / (1.0f + exp(-gate_value)));
  }
}

// Fused GDN RMSNorm + SiLU gate -> FP16 output (for M5 MXFP4 NAX) with tail-row zero padding.
// Grid height = mp * value_heads (mp multiple of 8 >= params.tokens).
kernel void omlx_m5_gdn_norm_gate_f16(device const bfloat *recurrent [[buffer(0)]],
                                      device const bfloat *z [[buffer(1)]],
                                      device const bfloat *norm_weight [[buffer(2)]],
                                      device half *output_f16 [[buffer(3)]],
                                      constant GdnLayerParams &params [[buffer(4)]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint logical_head = tg_position.y;
  const uint token = logical_head / params.value_heads;
  const uint value_head = logical_head - token * params.value_heads;
  const ulong value_width = ulong(params.value_heads) * params.value_dim;
  device half *destination =
      output_f16 + ulong(token) * value_width + ulong(value_head) * params.value_dim;
  if (token >= params.tokens) {
    for (uint feature = lane; feature < params.value_dim; feature += 32) {
      destination[feature] = half(0.0h);
    }
    return;
  }
  device const bfloat *row =
      recurrent + ulong(token) * value_width + ulong(value_head) * params.value_dim;
  device const bfloat *gate =
      z + ulong(token) * value_width + ulong(value_head) * params.value_dim;
  float square_sum = 0.0f;
  for (uint feature = lane; feature < params.value_dim; feature += 32) {
    const float value = float(row[feature]);
    square_sum = fma(value, value, square_sum);
  }
  const float inverse =
      rsqrt(simd_sum(square_sum) / float(params.value_dim) + params.epsilon);
  for (uint feature = lane; feature < params.value_dim; feature += 32) {
    const float gate_value = float(gate[feature]);
    const float normalized =
        float(row[feature]) * inverse * float(norm_weight[feature]);
    const float gated =
        float(bfloat(normalized * gate_value / (1.0f + exp(-gate_value))));
    destination[feature] = half(clamp(gated, -65504.0f, 65504.0f));
  }
}

// Speculative verify GDN recurrence (`state` is strictly read-only `const`!
// Dead `.scratch` writeback eliminated, saving 151 MiB/slot RAM & 453 MiB/round DRAM).
kernel void omlx_m5_gdn_recurrent_record(
    device const bfloat *convolved [[buffer(0)]],
    device const bfloat *a [[buffer(1)]], device const bfloat *b [[buffer(2)]],
    device const bfloat *a_log [[buffer(3)]],
    device const bfloat *dt_bias [[buffer(4)]],
    device const float *state [[buffer(5)]],
    device bfloat *recurrent [[buffer(6)]],
    device float *record_keys [[buffer(7)]],
    device float *record_decays [[buffer(8)]],
    device float *record_deltas [[buffer(9)]],
    constant GdnLayerParams &params [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint row_index = tg_position.y;
  const uint value_width = params.value_heads * params.value_dim;
  if (row_index >= value_width || params.tokens == 0)
    return;
  const uint value_head = row_index / params.value_dim;
  const uint value_index = row_index - value_head * params.value_dim;
  const uint repeat = params.value_heads / params.key_heads;
  const uint key_head = value_head / repeat;
  const uint key_width = params.key_heads * params.key_dim;
  device const float *state_row = state + ulong(row_index) * params.key_dim;
  const float inverse_scale = rsqrt(float(params.key_dim));
  const float dt_bias_val = lane == 0 ? float(dt_bias[value_head]) : 0.0f;
  const float neg_exp_a = lane == 0 ? -exp(float(a_log[value_head])) : 0.0f;
  const ulong q_head_offset = ulong(key_head) * params.key_dim;
  const ulong k_head_offset = ulong(key_width) + q_head_offset;
  const ulong v_elem_offset =
      ulong(2 * key_width) + ulong(value_head * params.value_dim + value_index);

  if (params.key_dim == 128) {
    float s0 = state_row[lane];
    float s1 = state_row[lane + 32];
    float s2 = state_row[lane + 64];
    float s3 = state_row[lane + 96];

    for (uint token = 0; token < params.tokens; ++token) {
      device const bfloat *token_row =
          convolved + ulong(token) * params.conv_width;
      device const bfloat *query = token_row + q_head_offset;
      device const bfloat *key = token_row + k_head_offset;

      const float q0 = float(query[lane]);
      const float q1 = float(query[lane + 32]);
      const float q2 = float(query[lane + 64]);
      const float q3 = float(query[lane + 96]);
      const float k0 = float(key[lane]);
      const float k1 = float(key[lane + 32]);
      const float k2 = float(key[lane + 64]);
      const float k3 = float(key[lane + 96]);

      float query_square = fma(q0, q0, 0.0f);
      query_square = fma(q1, q1, query_square);
      query_square = fma(q2, q2, query_square);
      query_square = fma(q3, q3, query_square);

      float key_square = fma(k0, k0, 0.0f);
      key_square = fma(k1, k1, key_square);
      key_square = fma(k2, k2, key_square);
      key_square = fma(k3, k3, key_square);

      const float query_inverse =
          rsqrt(simd_sum(query_square) / float(params.key_dim) + 1.0e-6f);
      const float key_inverse =
          rsqrt(simd_sum(key_square) / float(params.key_dim) + 1.0e-6f);

      float decay = 0.0f;
      float beta = 0.0f;
      float v_val = 0.0f;
      if (lane == 0) {
        const ulong h_idx = ulong(token) * params.value_heads + value_head;
        const float activation = float(a[h_idx]) + dt_bias_val;
        const float softplus =
            activation > 20.0f ? activation : log(1.0f + exp(activation));
        decay = exp(neg_exp_a * softplus);
        beta = 1.0f / (1.0f + exp(-float(b[h_idx])));
        v_val = float(token_row[v_elem_offset]);
      }
      decay = simd_broadcast_first(decay);
      beta = simd_broadcast_first(beta);
      v_val = simd_broadcast_first(v_val);

      const float nk0 = k0 * key_inverse * inverse_scale;
      const float nk1 = k1 * key_inverse * inverse_scale;
      const float nk2 = k2 * key_inverse * inverse_scale;
      const float nk3 = k3 * key_inverse * inverse_scale;

      s0 *= decay;
      float memory = fma(s0, nk0, 0.0f);
      s1 *= decay;
      memory = fma(s1, nk1, memory);
      s2 *= decay;
      memory = fma(s2, nk2, memory);
      s3 *= decay;
      memory = fma(s3, nk3, memory);
      memory = simd_sum(memory);

      const float delta = (v_val - memory) * beta;
      if (lane == 0) {
        record_deltas[ulong(token) * value_width + row_index] = delta;
        if (value_index == 0)
          record_decays[ulong(token) * params.value_heads + value_head] = decay;
      }
      if (value_index == 0 && value_head % repeat == 0) {
        const ulong k_rec_base = ulong(token) * key_width + q_head_offset;
        record_keys[k_rec_base + lane] = nk0;
        record_keys[k_rec_base + lane + 32] = nk1;
        record_keys[k_rec_base + lane + 64] = nk2;
        record_keys[k_rec_base + lane + 96] = nk3;
      }

      s0 = fma(nk0, delta, s0);
      float projected =
          fma(s0, q0 * query_inverse * inverse_scale * inverse_scale, 0.0f);
      s1 = fma(nk1, delta, s1);
      projected = fma(s1, q1 * query_inverse * inverse_scale * inverse_scale,
                      projected);
      s2 = fma(nk2, delta, s2);
      projected = fma(s2, q2 * query_inverse * inverse_scale * inverse_scale,
                      projected);
      s3 = fma(nk3, delta, s3);
      projected = fma(s3, q3 * query_inverse * inverse_scale * inverse_scale,
                      projected);
      projected = simd_sum(projected);
      if (lane == 0) {
        recurrent[ulong(token) * value_width + row_index] = bfloat(projected);
      }
    }
    return;
  }

  float s_reg[8];
#pragma clang loop unroll(full)
  for (uint idx = 0; idx < 8; ++idx) {
    const uint feature = lane + idx * 32;
    s_reg[idx] = feature < params.key_dim ? state_row[feature] : 0.0f;
  }

  for (uint token = 0; token < params.tokens; ++token) {
    device const bfloat *token_row =
        convolved + ulong(token) * params.conv_width;
    device const bfloat *query = token_row + q_head_offset;
    device const bfloat *key = token_row + k_head_offset;

    float q_reg[8];
    float k_reg[8];
    float query_square = 0.0f;
    float key_square = 0.0f;
#pragma clang loop unroll(full)
    for (uint idx = 0; idx < 8; ++idx) {
      const uint feature = lane + idx * 32;
      if (feature < params.key_dim) {
        const float q = float(query[feature]);
        const float k = float(key[feature]);
        q_reg[idx] = q;
        k_reg[idx] = k;
        query_square = fma(q, q, query_square);
        key_square = fma(k, k, key_square);
      } else {
        q_reg[idx] = 0.0f;
        k_reg[idx] = 0.0f;
      }
    }
    const float query_inverse =
        rsqrt(simd_sum(query_square) / float(params.key_dim) + 1.0e-6f);
    const float key_inverse =
        rsqrt(simd_sum(key_square) / float(params.key_dim) + 1.0e-6f);

    float decay = 0.0f;
    float beta = 0.0f;
    float v_val = 0.0f;
    if (lane == 0) {
      const ulong h_idx = ulong(token) * params.value_heads + value_head;
      const float activation = float(a[h_idx]) + dt_bias_val;
      const float softplus =
          activation > 20.0f ? activation : log(1.0f + exp(activation));
      decay = exp(neg_exp_a * softplus);
      beta = 1.0f / (1.0f + exp(-float(b[h_idx])));
      v_val = float(token_row[v_elem_offset]);
    }
    decay = simd_broadcast_first(decay);
    beta = simd_broadcast_first(beta);
    v_val = simd_broadcast_first(v_val);

    float memory = 0.0f;
#pragma clang loop unroll(full)
    for (uint idx = 0; idx < 8; ++idx) {
      const uint feature = lane + idx * 32;
      if (feature < params.key_dim) {
        s_reg[idx] *= decay;
        const float normalized_key = k_reg[idx] * key_inverse * inverse_scale;
        memory = fma(s_reg[idx], normalized_key, memory);
      }
    }
    memory = simd_sum(memory);
    const float delta = (v_val - memory) * beta;
    if (lane == 0) {
      record_deltas[ulong(token) * value_width + row_index] = delta;
      if (value_index == 0)
        record_decays[ulong(token) * params.value_heads + value_head] = decay;
    }
    if (value_index == 0 && value_head % repeat == 0) {
#pragma clang loop unroll(full)
      for (uint idx = 0; idx < 8; ++idx) {
        const uint feature = lane + idx * 32;
        if (feature < params.key_dim)
          record_keys[ulong(token) * key_width + q_head_offset + feature] =
              k_reg[idx] * key_inverse * inverse_scale;
      }
    }
    float projected = 0.0f;
#pragma clang loop unroll(full)
    for (uint idx = 0; idx < 8; ++idx) {
      const uint feature = lane + idx * 32;
      if (feature < params.key_dim) {
        const float normalized_key = k_reg[idx] * key_inverse * inverse_scale;
        s_reg[idx] = fma(normalized_key, delta, s_reg[idx]);
        const float normalized_query =
            q_reg[idx] * query_inverse * inverse_scale * inverse_scale;
        projected = fma(s_reg[idx], normalized_query, projected);
      }
    }
    projected = simd_sum(projected);
    if (lane == 0) {
      recurrent[ulong(token) * value_width + row_index] = bfloat(projected);
    }
  }
}

// 4-value-row-per-SIMDgroup fast path for `omlx_m5_gdn_recurrent_record`
// (`key_dim == 128`, `value_dim % 4 == 0`, grid.height = gv / 4). Amortizes
// Q/K loads, `simd_sum` L2-norm reductions, and `exp/log` gates 4x across 4
// consecutive `value_index` rows with 0 ULP difference and zero state writeback.
kernel void omlx_m5_gdn_recurrent_record_4row(
    device const bfloat *convolved [[buffer(0)]],
    device const bfloat *a [[buffer(1)]], device const bfloat *b [[buffer(2)]],
    device const bfloat *a_log [[buffer(3)]],
    device const bfloat *dt_bias [[buffer(4)]],
    device const float *state [[buffer(5)]],
    device bfloat *recurrent [[buffer(6)]],
    device float *record_keys [[buffer(7)]],
    device float *record_decays [[buffer(8)]],
    device float *record_deltas [[buffer(9)]],
    constant GdnLayerParams &params [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint row_base = tg_position.y * 4u;
  const uint value_width = params.value_heads * params.value_dim;
  if (row_base >= value_width || params.tokens == 0)
    return;
  const uint value_head = row_base / params.value_dim;
  const uint value_index0 = row_base - value_head * params.value_dim;
  const uint repeat = params.value_heads / params.key_heads;
  const uint key_head = value_head / repeat;
  const uint key_width = params.key_heads * 128u;
  const float inverse_scale = rsqrt(128.0f);
  const float dt_bias_val = lane == 0 ? float(dt_bias[value_head]) : 0.0f;
  const float neg_exp_a = lane == 0 ? -exp(float(a_log[value_head])) : 0.0f;
  const ulong q_head_offset = ulong(key_head) * 128u;
  const ulong k_head_offset = ulong(key_width) + q_head_offset;
  const ulong v_base_offset =
      ulong(2u * key_width) + ulong(value_head * params.value_dim + value_index0);

  device const float *st0 = state + ulong(row_base + 0u) * 128u;
  device const float *st1 = state + ulong(row_base + 1u) * 128u;
  device const float *st2 = state + ulong(row_base + 2u) * 128u;
  device const float *st3 = state + ulong(row_base + 3u) * 128u;

  float r0_s0 = st0[lane], r0_s1 = st0[lane + 32u], r0_s2 = st0[lane + 64u],
        r0_s3 = st0[lane + 96u];
  float r1_s0 = st1[lane], r1_s1 = st1[lane + 32u], r1_s2 = st1[lane + 64u],
        r1_s3 = st1[lane + 96u];
  float r2_s0 = st2[lane], r2_s1 = st2[lane + 32u], r2_s2 = st2[lane + 64u],
        r2_s3 = st2[lane + 96u];
  float r3_s0 = st3[lane], r3_s1 = st3[lane + 32u], r3_s2 = st3[lane + 64u],
        r3_s3 = st3[lane + 96u];

  for (uint token = 0; token < params.tokens; ++token) {
    device const bfloat *token_row =
        convolved + ulong(token) * params.conv_width;
    device const bfloat *query = token_row + q_head_offset;
    device const bfloat *key = token_row + k_head_offset;

    const float q0 = float(query[lane]);
    const float q1 = float(query[lane + 32u]);
    const float q2 = float(query[lane + 64u]);
    const float q3 = float(query[lane + 96u]);
    const float k0 = float(key[lane]);
    const float k1 = float(key[lane + 32u]);
    const float k2 = float(key[lane + 64u]);
    const float k3 = float(key[lane + 96u]);

    float query_square = fma(q0, q0, 0.0f);
    query_square = fma(q1, q1, query_square);
    query_square = fma(q2, q2, query_square);
    query_square = fma(q3, q3, query_square);

    float key_square = fma(k0, k0, 0.0f);
    key_square = fma(k1, k1, key_square);
    key_square = fma(k2, k2, key_square);
    key_square = fma(k3, k3, key_square);

    const float query_inverse =
        rsqrt(simd_sum(query_square) / 128.0f + 1.0e-6f);
    const float key_inverse = rsqrt(simd_sum(key_square) / 128.0f + 1.0e-6f);

    float decay = 0.0f;
    float beta = 0.0f;
    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    if (lane == 0) {
      const ulong h_idx = ulong(token) * params.value_heads + value_head;
      const float activation = float(a[h_idx]) + dt_bias_val;
      const float softplus =
          activation > 20.0f ? activation : log(1.0f + exp(activation));
      decay = exp(neg_exp_a * softplus);
      beta = 1.0f / (1.0f + exp(-float(b[h_idx])));
      const bfloat4 v4 =
          *reinterpret_cast<device const bfloat4 *>(token_row + v_base_offset);
      v0 = float(v4.x);
      v1 = float(v4.y);
      v2 = float(v4.z);
      v3 = float(v4.w);
    }
    decay = simd_broadcast_first(decay);
    beta = simd_broadcast_first(beta);
    v0 = simd_broadcast_first(v0);
    v1 = simd_broadcast_first(v1);
    v2 = simd_broadcast_first(v2);
    v3 = simd_broadcast_first(v3);

    const float nk0 = k0 * key_inverse * inverse_scale;
    const float nk1 = k1 * key_inverse * inverse_scale;
    const float nk2 = k2 * key_inverse * inverse_scale;
    const float nk3 = k3 * key_inverse * inverse_scale;
    const float nq0 = q0 * query_inverse * inverse_scale * inverse_scale;
    const float nq1 = q1 * query_inverse * inverse_scale * inverse_scale;
    const float nq2 = q2 * query_inverse * inverse_scale * inverse_scale;
    const float nq3 = q3 * query_inverse * inverse_scale * inverse_scale;

    if (value_index0 == 0u) {
      if (lane == 0) {
        record_decays[ulong(token) * params.value_heads + value_head] = decay;
      }
      if ((value_head % repeat) == 0u) {
        const ulong k_rec_base = ulong(token) * key_width + q_head_offset;
        record_keys[k_rec_base + lane] = nk0;
        record_keys[k_rec_base + lane + 32u] = nk1;
        record_keys[k_rec_base + lane + 64u] = nk2;
        record_keys[k_rec_base + lane + 96u] = nk3;
      }
    }

    const ulong out_base = ulong(token) * value_width + row_base;

    // Row 0
    {
      r0_s0 *= decay;
      float mem = fma(r0_s0, nk0, 0.0f);
      r0_s1 *= decay;
      mem = fma(r0_s1, nk1, mem);
      r0_s2 *= decay;
      mem = fma(r0_s2, nk2, mem);
      r0_s3 *= decay;
      mem = fma(r0_s3, nk3, mem);
      const float delta = (v0 - simd_sum(mem)) * beta;
      r0_s0 = fma(nk0, delta, r0_s0);
      float proj = fma(r0_s0, nq0, 0.0f);
      r0_s1 = fma(nk1, delta, r0_s1);
      proj = fma(r0_s1, nq1, proj);
      r0_s2 = fma(nk2, delta, r0_s2);
      proj = fma(r0_s2, nq2, proj);
      r0_s3 = fma(nk3, delta, r0_s3);
      proj = fma(r0_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0) {
        record_deltas[out_base + 0u] = delta;
        recurrent[out_base + 0u] = bfloat(proj);
      }
    }
    // Row 1
    {
      r1_s0 *= decay;
      float mem = fma(r1_s0, nk0, 0.0f);
      r1_s1 *= decay;
      mem = fma(r1_s1, nk1, mem);
      r1_s2 *= decay;
      mem = fma(r1_s2, nk2, mem);
      r1_s3 *= decay;
      mem = fma(r1_s3, nk3, mem);
      const float delta = (v1 - simd_sum(mem)) * beta;
      r1_s0 = fma(nk0, delta, r1_s0);
      float proj = fma(r1_s0, nq0, 0.0f);
      r1_s1 = fma(nk1, delta, r1_s1);
      proj = fma(r1_s1, nq1, proj);
      r1_s2 = fma(nk2, delta, r1_s2);
      proj = fma(r1_s2, nq2, proj);
      r1_s3 = fma(nk3, delta, r1_s3);
      proj = fma(r1_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0) {
        record_deltas[out_base + 1u] = delta;
        recurrent[out_base + 1u] = bfloat(proj);
      }
    }
    // Row 2
    {
      r2_s0 *= decay;
      float mem = fma(r2_s0, nk0, 0.0f);
      r2_s1 *= decay;
      mem = fma(r2_s1, nk1, mem);
      r2_s2 *= decay;
      mem = fma(r2_s2, nk2, mem);
      r2_s3 *= decay;
      mem = fma(r2_s3, nk3, mem);
      const float delta = (v2 - simd_sum(mem)) * beta;
      r2_s0 = fma(nk0, delta, r2_s0);
      float proj = fma(r2_s0, nq0, 0.0f);
      r2_s1 = fma(nk1, delta, r2_s1);
      proj = fma(r2_s1, nq1, proj);
      r2_s2 = fma(nk2, delta, r2_s2);
      proj = fma(r2_s2, nq2, proj);
      r2_s3 = fma(nk3, delta, r2_s3);
      proj = fma(r2_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0) {
        record_deltas[out_base + 2u] = delta;
        recurrent[out_base + 2u] = bfloat(proj);
      }
    }
    // Row 3
    {
      r3_s0 *= decay;
      float mem = fma(r3_s0, nk0, 0.0f);
      r3_s1 *= decay;
      mem = fma(r3_s1, nk1, mem);
      r3_s2 *= decay;
      mem = fma(r3_s2, nk2, mem);
      r3_s3 *= decay;
      mem = fma(r3_s3, nk3, mem);
      const float delta = (v3 - simd_sum(mem)) * beta;
      r3_s0 = fma(nk0, delta, r3_s0);
      float proj = fma(r3_s0, nq0, 0.0f);
      r3_s1 = fma(nk1, delta, r3_s1);
      proj = fma(r3_s1, nq1, proj);
      r3_s2 = fma(nk2, delta, r3_s2);
      proj = fma(r3_s2, nq2, proj);
      r3_s3 = fma(nk3, delta, r3_s3);
      proj = fma(r3_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0) {
        record_deltas[out_base + 3u] = delta;
        recurrent[out_base + 3u] = bfloat(proj);
      }
    }
  }
}

// Apply the accepted updates to the untouched committed state. No projections,
// normalization, attention or model weights are evaluated during this replay.
kernel void omlx_m5_gdn_replay_prefix(device float *state [[buffer(0)]],
                                      device const float *keys [[buffer(1)]],
                                      device const float *decays [[buffer(2)]],
                                      device const float *deltas [[buffer(3)]],
                                      constant GdnLayerParams &p [[buffer(4)]],
                                      uint i [[thread_position_in_grid]]) {
  uint width = p.value_heads * p.value_dim;
  if (ulong(i) >= ulong(width) * p.key_dim)
    return;
  uint row = i / p.key_dim, feature = i % p.key_dim;
  uint head = row / p.value_dim,
       key_head = head / (p.value_heads / p.key_heads);
  float value = state[i];
  for (uint t = 0; t < p.tokens; ++t) {
    value *= decays[ulong(t) * p.value_heads + head];
    value = fma(keys[ulong(t) * p.key_heads * p.key_dim +
                     ulong(key_head) * p.key_dim + feature],
                deltas[ulong(t) * width + row], value);
  }
  state[i] = value;
}

kernel void omlx_m5_gdn_commit_history(device bfloat *history [[buffer(0)]],
                                       device const bfloat *qkv [[buffer(1)]],
                                       constant GdnLayerParams &p [[buffer(2)]],
                                       uint channel
                                       [[thread_position_in_grid]]) {
  if (channel >= p.conv_width || p.tokens == 0)
    return;
  uint rows = p.conv_kernel - 1;
  for (uint row = 0; row < rows; ++row) {
    int source = int(p.tokens) - int(rows) + int(row);
    history[ulong(row) * p.conv_width + channel] =
        source < 0
            ? history[ulong(uint(source + int(rows))) * p.conv_width + channel]
            : qkv[ulong(uint(source)) * p.conv_width + channel];
  }
}


// Coalesced bfloat4 T=1 causal conv + SiLU + in-place history shift (0-ULP identical to omlx_m5_gdn_conv_silu).
kernel void omlx_m5_gdn_conv_silu_vec4_t1(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device bfloat *history [[buffer(2)]],
    device bfloat *convolved [[buffer(3)]],
    constant GdnLayerParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const uint ch0 = gid * 4u;
  if (ch0 >= params.conv_width || params.tokens == 0u) {
    return;
  }
  if (params.tokens == 1u && params.conv_kernel == 4u && (params.conv_width & 3u) == 0u) {
    device const bfloat4 *wp =
        reinterpret_cast<device const bfloat4 *>(conv_weight + ulong(ch0) * 4ul);
    const float4 c0 = float4(wp[0]);
    const float4 c1 = float4(wp[1]);
    const float4 c2 = float4(wp[2]);
    const float4 c3 = float4(wp[3]);
    const float4 w0(c0.x, c1.x, c2.x, c3.x);
    const float4 w1(c0.y, c1.y, c2.y, c3.y);
    const float4 w2(c0.z, c1.z, c2.z, c3.z);
    const float4 w3(c0.w, c1.w, c2.w, c3.w);

    device bfloat4 *h0_ptr =
        reinterpret_cast<device bfloat4 *>(history + 0ul * params.conv_width + ch0);
    device bfloat4 *h1_ptr =
        reinterpret_cast<device bfloat4 *>(history + 1ul * params.conv_width + ch0);
    device bfloat4 *h2_ptr =
        reinterpret_cast<device bfloat4 *>(history + 2ul * params.conv_width + ch0);
    const bfloat4 h0_bf = *h0_ptr;
    const bfloat4 h1_bf = *h1_ptr;
    const bfloat4 h2_bf = *h2_ptr;
    const bfloat4 x0_bf =
        *reinterpret_cast<device const bfloat4 *>(qkv + ch0);

    float4 sum = fma(float4(h0_bf), w0, float4(0.0f));
    sum = fma(float4(h1_bf), w1, sum);
    sum = fma(float4(h2_bf), w2, sum);
    sum = fma(float4(x0_bf), w3, sum);
    *reinterpret_cast<device bfloat4 *>(convolved + ch0) =
        bfloat4(sum / (1.0f + exp(-sum)));

    *h0_ptr = h1_bf;
    *h1_ptr = h2_bf;
    *h2_ptr = x0_bf;
    return;
  }

  const uint history_tokens = params.conv_kernel - 1u;
  for (uint j = 0u; j < 4u && (ch0 + j) < params.conv_width; ++j) {
    const uint channel = ch0 + j;
    for (uint token = 0u; token < params.tokens; ++token) {
      float sum = 0.0f;
      for (uint tap = 0u; tap < params.conv_kernel; ++tap) {
        const int source_token = int(token) + int(tap) - int(history_tokens);
        const float input =
            source_token < 0
                ? float(history[ulong(uint(source_token + int(history_tokens))) *
                                    params.conv_width +
                                channel])
                : float(qkv[ulong(uint(source_token)) * params.conv_width +
                            channel]);
        sum = fma(input,
                  float(conv_weight[ulong(channel) * params.conv_kernel + tap]),
                  sum);
      }
      convolved[ulong(token) * params.conv_width + channel] =
          bfloat(sum / (1.0f + exp(-sum)));
    }
    for (uint history_token = 0u; history_token < history_tokens;
         ++history_token) {
      const int source_token =
          int(params.tokens) - int(history_tokens) + int(history_token);
      history[ulong(history_token) * params.conv_width + channel] =
          source_token < 0
              ? history[ulong(uint(source_token + int(history_tokens))) *
                            params.conv_width +
                        channel]
              : qkv[ulong(uint(source_token)) * params.conv_width + channel];
    }
  }
}

// 0-ULP bit-identical T=1 fused 1-step GDN recurrence + RMSNorm_128 * SiLU(z) gate.
// Grid: (256, value_heads, 1), Threads: (256, 1, 1) -> 8 SGs/TG (16 value rows/SG).
template <bool OutF16>
inline void gdn_t1_recurrent_norm_gate_impl(
    device const bfloat *convolved,
    device const bfloat *a,
    device const bfloat *b,
    device const bfloat *a_log,
    device const bfloat *dt_bias,
    device float *state,
    device const bfloat *z,
    device const bfloat *norm_weight,
    device bfloat *out_bf16,
    device half *out_f16,
    constant GdnLayerParams &params,
    threadgroup float *tg_rec,
    uint lane,
    uint sg,
    uint3 tg_position) {
  const uint value_head = tg_position.y;
  if (value_head >= params.value_heads) {
    return;
  }
  const ulong value_width = ulong(params.value_heads) * 128ul;
  if (params.tokens == 0u) {
    if constexpr (OutF16) {
      device half *dst_row =
          out_f16 + ulong(sg) * value_width + ulong(value_head) * 128ul;
      for (uint feature = lane; feature < 128u; feature += 32u) {
        dst_row[feature] = half(0.0h);
      }
    }
    return;
  }

  const uint repeat = params.value_heads / params.key_heads;
  const uint key_head = value_head / repeat;
  const uint key_width = params.key_heads * 128u;
  const float inverse_scale = rsqrt(128.0f);
  const float dt_bias_val = (lane == 0u) ? float(dt_bias[value_head]) : 0.0f;
  const float neg_exp_a = (lane == 0u) ? -exp(float(a_log[value_head])) : 0.0f;
  const ulong q_head_offset = ulong(key_head) * 128ul;
  const ulong k_head_offset = ulong(key_width) + q_head_offset;
  const ulong v_head_offset =
      ulong(2u * key_width) + ulong(value_head) * 128ul;

  device const bfloat *query = convolved + q_head_offset;
  device const bfloat *key = convolved + k_head_offset;
  const float q0 = float(query[lane]);
  const float q1 = float(query[lane + 32u]);
  const float q2 = float(query[lane + 64u]);
  const float q3 = float(query[lane + 96u]);
  const float k0 = float(key[lane]);
  const float k1 = float(key[lane + 32u]);
  const float k2 = float(key[lane + 64u]);
  const float k3 = float(key[lane + 96u]);

  float query_square = fma(q0, q0, 0.0f);
  query_square = fma(q1, q1, query_square);
  query_square = fma(q2, q2, query_square);
  query_square = fma(q3, q3, query_square);

  float key_square = fma(k0, k0, 0.0f);
  key_square = fma(k1, k1, key_square);
  key_square = fma(k2, k2, key_square);
  key_square = fma(k3, k3, key_square);

  const float query_inverse = rsqrt(simd_sum(query_square) / 128.0f + 1.0e-6f);
  const float key_inverse = rsqrt(simd_sum(key_square) / 128.0f + 1.0e-6f);

  float decay = 0.0f;
  float beta = 0.0f;
  if (lane == 0u) {
    const float activation = float(a[value_head]) + dt_bias_val;
    const float softplus =
        activation > 20.0f ? activation : log(1.0f + exp(activation));
    decay = exp(neg_exp_a * softplus);
    beta = 1.0f / (1.0f + exp(-float(b[value_head])));
  }
  decay = simd_broadcast_first(decay);
  beta = simd_broadcast_first(beta);

  const float nk0 = k0 * key_inverse * inverse_scale;
  const float nk1 = k1 * key_inverse * inverse_scale;
  const float nk2 = k2 * key_inverse * inverse_scale;
  const float nk3 = k3 * key_inverse * inverse_scale;
  const float nq0 = q0 * query_inverse * inverse_scale * inverse_scale;
  const float nq1 = q1 * query_inverse * inverse_scale * inverse_scale;
  const float nq2 = q2 * query_inverse * inverse_scale * inverse_scale;
  const float nq3 = q3 * query_inverse * inverse_scale * inverse_scale;

  for (uint base4 = 0u; base4 < 16u; base4 += 4u) {
    const uint r_local = sg * 16u + base4;
    const uint row_base = value_head * 128u + r_local;

    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    if (lane == 0u) {
      const bfloat4 v4 = *reinterpret_cast<device const bfloat4 *>(
          convolved + v_head_offset + r_local);
      v0 = float(v4.x);
      v1 = float(v4.y);
      v2 = float(v4.z);
      v3 = float(v4.w);
    }
    v0 = simd_broadcast_first(v0);
    v1 = simd_broadcast_first(v1);
    v2 = simd_broadcast_first(v2);
    v3 = simd_broadcast_first(v3);

    device float *st0 = state + ulong(row_base + 0u) * 128ul;
    device float *st1 = state + ulong(row_base + 1u) * 128ul;
    device float *st2 = state + ulong(row_base + 2u) * 128ul;
    device float *st3 = state + ulong(row_base + 3u) * 128ul;

    float r0_s0 = st0[lane], r0_s1 = st0[lane + 32u], r0_s2 = st0[lane + 64u],
          r0_s3 = st0[lane + 96u];
    float r1_s0 = st1[lane], r1_s1 = st1[lane + 32u], r1_s2 = st1[lane + 64u],
          r1_s3 = st1[lane + 96u];
    float r2_s0 = st2[lane], r2_s1 = st2[lane + 32u], r2_s2 = st2[lane + 64u],
          r2_s3 = st2[lane + 96u];
    float r3_s0 = st3[lane], r3_s1 = st3[lane + 32u], r3_s2 = st3[lane + 64u],
          r3_s3 = st3[lane + 96u];

    {
      r0_s0 *= decay;
      float mem = fma(r0_s0, nk0, 0.0f);
      r0_s1 *= decay;
      mem = fma(r0_s1, nk1, mem);
      r0_s2 *= decay;
      mem = fma(r0_s2, nk2, mem);
      r0_s3 *= decay;
      mem = fma(r0_s3, nk3, mem);
      const float delta = (v0 - simd_sum(mem)) * beta;
      r0_s0 = fma(nk0, delta, r0_s0);
      float proj = fma(r0_s0, nq0, 0.0f);
      r0_s1 = fma(nk1, delta, r0_s1);
      proj = fma(r0_s1, nq1, proj);
      r0_s2 = fma(nk2, delta, r0_s2);
      proj = fma(r0_s2, nq2, proj);
      r0_s3 = fma(nk3, delta, r0_s3);
      proj = fma(r0_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) {
        tg_rec[r_local + 0u] = float(bfloat(proj));
      }
    }
    {
      r1_s0 *= decay;
      float mem = fma(r1_s0, nk0, 0.0f);
      r1_s1 *= decay;
      mem = fma(r1_s1, nk1, mem);
      r1_s2 *= decay;
      mem = fma(r1_s2, nk2, mem);
      r1_s3 *= decay;
      mem = fma(r1_s3, nk3, mem);
      const float delta = (v1 - simd_sum(mem)) * beta;
      r1_s0 = fma(nk0, delta, r1_s0);
      float proj = fma(r1_s0, nq0, 0.0f);
      r1_s1 = fma(nk1, delta, r1_s1);
      proj = fma(r1_s1, nq1, proj);
      r1_s2 = fma(nk2, delta, r1_s2);
      proj = fma(r1_s2, nq2, proj);
      r1_s3 = fma(nk3, delta, r1_s3);
      proj = fma(r1_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) {
        tg_rec[r_local + 1u] = float(bfloat(proj));
      }
    }
    {
      r2_s0 *= decay;
      float mem = fma(r2_s0, nk0, 0.0f);
      r2_s1 *= decay;
      mem = fma(r2_s1, nk1, mem);
      r2_s2 *= decay;
      mem = fma(r2_s2, nk2, mem);
      r2_s3 *= decay;
      mem = fma(r2_s3, nk3, mem);
      const float delta = (v2 - simd_sum(mem)) * beta;
      r2_s0 = fma(nk0, delta, r2_s0);
      float proj = fma(r2_s0, nq0, 0.0f);
      r2_s1 = fma(nk1, delta, r2_s1);
      proj = fma(r2_s1, nq1, proj);
      r2_s2 = fma(nk2, delta, r2_s2);
      proj = fma(r2_s2, nq2, proj);
      r2_s3 = fma(nk3, delta, r2_s3);
      proj = fma(r2_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) {
        tg_rec[r_local + 2u] = float(bfloat(proj));
      }
    }
    {
      r3_s0 *= decay;
      float mem = fma(r3_s0, nk0, 0.0f);
      r3_s1 *= decay;
      mem = fma(r3_s1, nk1, mem);
      r3_s2 *= decay;
      mem = fma(r3_s2, nk2, mem);
      r3_s3 *= decay;
      mem = fma(r3_s3, nk3, mem);
      const float delta = (v3 - simd_sum(mem)) * beta;
      r3_s0 = fma(nk0, delta, r3_s0);
      float proj = fma(r3_s0, nq0, 0.0f);
      r3_s1 = fma(nk1, delta, r3_s1);
      proj = fma(r3_s1, nq1, proj);
      r3_s2 = fma(nk2, delta, r3_s2);
      proj = fma(r3_s2, nq2, proj);
      r3_s3 = fma(nk3, delta, r3_s3);
      proj = fma(r3_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) {
        tg_rec[r_local + 3u] = float(bfloat(proj));
      }
    }

    st0[lane] = r0_s0;
    st0[lane + 32u] = r0_s1;
    st0[lane + 64u] = r0_s2;
    st0[lane + 96u] = r0_s3;

    st1[lane] = r1_s0;
    st1[lane + 32u] = r1_s1;
    st1[lane + 64u] = r1_s2;
    st1[lane + 96u] = r1_s3;

    st2[lane] = r2_s0;
    st2[lane + 32u] = r2_s1;
    st2[lane + 64u] = r2_s2;
    st2[lane + 96u] = r2_s3;

    st3[lane] = r3_s0;
    st3[lane + 32u] = r3_s1;
    st3[lane + 64u] = r3_s2;
    st3[lane + 96u] = r3_s3;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u) {
    float square_sum = 0.0f;
    for (uint feature = lane; feature < 128u; feature += 32u) {
      const float value = tg_rec[feature];
      square_sum = fma(value, value, square_sum);
    }
    const float inverse =
        rsqrt(simd_sum(square_sum) / 128.0f + params.epsilon);
    device const bfloat *gate = z + ulong(value_head) * 128ul;
    for (uint feature = lane; feature < 128u; feature += 32u) {
      const float gate_value = float(gate[feature]);
      const float normalized =
          tg_rec[feature] * inverse * float(norm_weight[feature]);
      if constexpr (OutF16) {
        const float gated =
            float(bfloat(normalized * gate_value / (1.0f + exp(-gate_value))));
        out_f16[ulong(value_head) * 128ul + feature] =
            half(clamp(gated, -65504.0f, 65504.0f));
      } else {
        out_bf16[ulong(value_head) * 128ul + feature] =
            bfloat(normalized * gate_value / (1.0f + exp(-gate_value)));
      }
    }
  } else if constexpr (OutF16) {
    device half *dst_pad =
        out_f16 + ulong(sg) * value_width + ulong(value_head) * 128ul;
    for (uint feature = lane; feature < 128u; feature += 32u) {
      dst_pad[feature] = half(0.0h);
    }
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_gdn_t1_recurrent_norm_gate(
    device const bfloat *convolved [[buffer(0)]],
    device const bfloat *a [[buffer(1)]],
    device const bfloat *b [[buffer(2)]],
    device const bfloat *a_log [[buffer(3)]],
    device const bfloat *dt_bias [[buffer(4)]],
    device float *state [[buffer(5)]],
    device const bfloat *z [[buffer(6)]],
    device const bfloat *norm_weight [[buffer(7)]],
    device bfloat *output [[buffer(8)]],
    constant GdnLayerParams &params [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  threadgroup float tg_rec[128];
  gdn_t1_recurrent_norm_gate_impl<false>(
      convolved, a, b, a_log, dt_bias, state, z, norm_weight, output, nullptr,
      params, tg_rec, lane, sg, tg_position);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_gdn_t1_recurrent_norm_gate_f16(
    device const bfloat *convolved [[buffer(0)]],
    device const bfloat *a [[buffer(1)]],
    device const bfloat *b [[buffer(2)]],
    device const bfloat *a_log [[buffer(3)]],
    device const bfloat *dt_bias [[buffer(4)]],
    device float *state [[buffer(5)]],
    device const bfloat *z [[buffer(6)]],
    device const bfloat *norm_weight [[buffer(7)]],
    device half *output_f16 [[buffer(8)]],
    constant GdnLayerParams &params [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  threadgroup float tg_rec[128];
  gdn_t1_recurrent_norm_gate_impl<true>(
      convolved, a, b, a_log, dt_bias, state, z, norm_weight, nullptr, output_f16,
      params, tg_rec, lane, sg, tg_position);
}

inline void gdn_conv_silu_vec4_128ch_to_tg(
    device const bfloat *qkv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    uint conv_width,
    ulong ch_base,
    uint lane,
    threadgroup float *tg_dst) {
  const ulong ch0 = ch_base + ulong(lane) * 4ul;
  device const bfloat4 *wp =
      reinterpret_cast<device const bfloat4 *>(conv_weight + ch0 * 4ul);
  const float4 c0 = float4(wp[0]);
  const float4 c1 = float4(wp[1]);
  const float4 c2 = float4(wp[2]);
  const float4 c3 = float4(wp[3]);
  const float4 w0(c0.x, c1.x, c2.x, c3.x);
  const float4 w1(c0.y, c1.y, c2.y, c3.y);
  const float4 w2(c0.z, c1.z, c2.z, c3.z);
  const float4 w3(c0.w, c1.w, c2.w, c3.w);

  const bfloat4 h0_bf = *reinterpret_cast<device const bfloat4 *>(
      history + 0ul * conv_width + ch0);
  const bfloat4 h1_bf = *reinterpret_cast<device const bfloat4 *>(
      history + 1ul * conv_width + ch0);
  const bfloat4 h2_bf = *reinterpret_cast<device const bfloat4 *>(
      history + 2ul * conv_width + ch0);
  const bfloat4 x0_bf =
      *reinterpret_cast<device const bfloat4 *>(qkv + ch0);

  float4 sum = fma(float4(h0_bf), w0, float4(0.0f));
  sum = fma(float4(h1_bf), w1, sum);
  sum = fma(float4(h2_bf), w2, sum);
  sum = fma(float4(x0_bf), w3, sum);
  const float4 out_f = float4(bfloat4(sum / (1.0f + exp(-sum))));
  const uint base = lane * 4u;
  tg_dst[base + 0u] = out_f.x;
  tg_dst[base + 1u] = out_f.y;
  tg_dst[base + 2u] = out_f.z;
  tg_dst[base + 3u] = out_f.w;
}

template <bool OutF16>
inline void gdn_t1_conv_recurrent_norm_gate_impl(
    device const bfloat *qkv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    device const bfloat *a,
    device const bfloat *b,
    device const bfloat *a_log,
    device const bfloat *dt_bias,
    device float *state,
    device const bfloat *z,
    device const bfloat *norm_weight,
    device bfloat *out_bf16,
    device half *out_f16,
    constant GdnLayerParams &params,
    threadgroup float *tg_q,
    threadgroup float *tg_k,
    threadgroup float *tg_v,
    threadgroup float *tg_rec,
    uint lane,
    uint sg,
    uint3 tg_position) {
  const uint value_head = tg_position.y;
  if (value_head >= params.value_heads) {
    return;
  }
  const ulong value_width = ulong(params.value_heads) * 128ul;
  if (params.tokens == 0u) {
    if constexpr (OutF16) {
      device half *dst_row =
          out_f16 + ulong(sg) * value_width + ulong(value_head) * 128ul;
      for (uint feature = lane; feature < 128u; feature += 32u) {
        dst_row[feature] = half(0.0h);
      }
    }
    return;
  }

  const uint repeat = params.value_heads / params.key_heads;
  const uint key_head = value_head / repeat;
  const uint key_width = params.key_heads * 128u;
  const ulong q_head_offset = ulong(key_head) * 128ul;
  const ulong k_head_offset = ulong(key_width) + q_head_offset;
  const ulong v_head_offset =
      ulong(2u * key_width) + ulong(value_head) * 128ul;

  if (sg == 0u) {
    gdn_conv_silu_vec4_128ch_to_tg(qkv, conv_weight, history, params.conv_width,
                                   q_head_offset, lane, tg_q);
  } else if (sg == 1u) {
    gdn_conv_silu_vec4_128ch_to_tg(qkv, conv_weight, history, params.conv_width,
                                   k_head_offset, lane, tg_k);
  } else if (sg == 2u) {
    gdn_conv_silu_vec4_128ch_to_tg(qkv, conv_weight, history, params.conv_width,
                                   v_head_offset, lane, tg_v);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float inverse_scale = rsqrt(128.0f);
  const float dt_bias_val = (lane == 0u) ? float(dt_bias[value_head]) : 0.0f;
  const float neg_exp_a = (lane == 0u) ? -exp(float(a_log[value_head])) : 0.0f;

  const float q0 = tg_q[lane];
  const float q1 = tg_q[lane + 32u];
  const float q2 = tg_q[lane + 64u];
  const float q3 = tg_q[lane + 96u];
  const float k0 = tg_k[lane];
  const float k1 = tg_k[lane + 32u];
  const float k2 = tg_k[lane + 64u];
  const float k3 = tg_k[lane + 96u];

  float query_square = fma(q0, q0, 0.0f);
  query_square = fma(q1, q1, query_square);
  query_square = fma(q2, q2, query_square);
  query_square = fma(q3, q3, query_square);

  float key_square = fma(k0, k0, 0.0f);
  key_square = fma(k1, k1, key_square);
  key_square = fma(k2, k2, key_square);
  key_square = fma(k3, k3, key_square);

  const float query_inverse = rsqrt(simd_sum(query_square) / 128.0f + 1.0e-6f);
  const float key_inverse = rsqrt(simd_sum(key_square) / 128.0f + 1.0e-6f);

  float decay = 0.0f;
  float beta = 0.0f;
  if (lane == 0u) {
    const float activation = float(a[value_head]) + dt_bias_val;
    const float softplus =
        activation > 20.0f ? activation : log(1.0f + exp(activation));
    decay = exp(neg_exp_a * softplus);
    beta = 1.0f / (1.0f + exp(-float(b[value_head])));
  }
  decay = simd_broadcast_first(decay);
  beta = simd_broadcast_first(beta);

  const float nk0 = k0 * key_inverse * inverse_scale;
  const float nk1 = k1 * key_inverse * inverse_scale;
  const float nk2 = k2 * key_inverse * inverse_scale;
  const float nk3 = k3 * key_inverse * inverse_scale;
  const float nq0 = q0 * query_inverse * inverse_scale * inverse_scale;
  const float nq1 = q1 * query_inverse * inverse_scale * inverse_scale;
  const float nq2 = q2 * query_inverse * inverse_scale * inverse_scale;
  const float nq3 = q3 * query_inverse * inverse_scale * inverse_scale;

  for (uint base4 = 0u; base4 < 16u; base4 += 4u) {
    const uint r_local = sg * 16u + base4;
    const uint row_base = value_head * 128u + r_local;

    const float v0 = tg_v[r_local + 0u];
    const float v1 = tg_v[r_local + 1u];
    const float v2 = tg_v[r_local + 2u];
    const float v3 = tg_v[r_local + 3u];

    device float *st0 = state + ulong(row_base + 0u) * 128ul;
    device float *st1 = state + ulong(row_base + 1u) * 128ul;
    device float *st2 = state + ulong(row_base + 2u) * 128ul;
    device float *st3 = state + ulong(row_base + 3u) * 128ul;

    float r0_s0 = st0[lane], r0_s1 = st0[lane + 32u], r0_s2 = st0[lane + 64u],
          r0_s3 = st0[lane + 96u];
    float r1_s0 = st1[lane], r1_s1 = st1[lane + 32u], r1_s2 = st1[lane + 64u],
          r1_s3 = st1[lane + 96u];
    float r2_s0 = st2[lane], r2_s1 = st2[lane + 32u], r2_s2 = st2[lane + 64u],
          r2_s3 = st2[lane + 96u];
    float r3_s0 = st3[lane], r3_s1 = st3[lane + 32u], r3_s2 = st3[lane + 64u],
          r3_s3 = st3[lane + 96u];

    {
      r0_s0 *= decay;
      float mem = fma(r0_s0, nk0, 0.0f);
      r0_s1 *= decay;
      mem = fma(r0_s1, nk1, mem);
      r0_s2 *= decay;
      mem = fma(r0_s2, nk2, mem);
      r0_s3 *= decay;
      mem = fma(r0_s3, nk3, mem);
      const float delta = (v0 - simd_sum(mem)) * beta;
      r0_s0 = fma(nk0, delta, r0_s0);
      float proj = fma(r0_s0, nq0, 0.0f);
      r0_s1 = fma(nk1, delta, r0_s1);
      proj = fma(r0_s1, nq1, proj);
      r0_s2 = fma(nk2, delta, r0_s2);
      proj = fma(r0_s2, nq2, proj);
      r0_s3 = fma(nk3, delta, r0_s3);
      proj = fma(r0_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) {
        tg_rec[r_local + 0u] = float(bfloat(proj));
      }
    }
    {
      r1_s0 *= decay;
      float mem = fma(r1_s0, nk0, 0.0f);
      r1_s1 *= decay;
      mem = fma(r1_s1, nk1, mem);
      r1_s2 *= decay;
      mem = fma(r1_s2, nk2, mem);
      r1_s3 *= decay;
      mem = fma(r1_s3, nk3, mem);
      const float delta = (v1 - simd_sum(mem)) * beta;
      r1_s0 = fma(nk0, delta, r1_s0);
      float proj = fma(r1_s0, nq0, 0.0f);
      r1_s1 = fma(nk1, delta, r1_s1);
      proj = fma(r1_s1, nq1, proj);
      r1_s2 = fma(nk2, delta, r1_s2);
      proj = fma(r1_s2, nq2, proj);
      r1_s3 = fma(nk3, delta, r1_s3);
      proj = fma(r1_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) {
        tg_rec[r_local + 1u] = float(bfloat(proj));
      }
    }
    {
      r2_s0 *= decay;
      float mem = fma(r2_s0, nk0, 0.0f);
      r2_s1 *= decay;
      mem = fma(r2_s1, nk1, mem);
      r2_s2 *= decay;
      mem = fma(r2_s2, nk2, mem);
      r2_s3 *= decay;
      mem = fma(r2_s3, nk3, mem);
      const float delta = (v2 - simd_sum(mem)) * beta;
      r2_s0 = fma(nk0, delta, r2_s0);
      float proj = fma(r2_s0, nq0, 0.0f);
      r2_s1 = fma(nk1, delta, r2_s1);
      proj = fma(r2_s1, nq1, proj);
      r2_s2 = fma(nk2, delta, r2_s2);
      proj = fma(r2_s2, nq2, proj);
      r2_s3 = fma(nk3, delta, r2_s3);
      proj = fma(r2_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) {
        tg_rec[r_local + 2u] = float(bfloat(proj));
      }
    }
    {
      r3_s0 *= decay;
      float mem = fma(r3_s0, nk0, 0.0f);
      r3_s1 *= decay;
      mem = fma(r3_s1, nk1, mem);
      r3_s2 *= decay;
      mem = fma(r3_s2, nk2, mem);
      r3_s3 *= decay;
      mem = fma(r3_s3, nk3, mem);
      const float delta = (v3 - simd_sum(mem)) * beta;
      r3_s0 = fma(nk0, delta, r3_s0);
      float proj = fma(r3_s0, nq0, 0.0f);
      r3_s1 = fma(nk1, delta, r3_s1);
      proj = fma(r3_s1, nq1, proj);
      r3_s2 = fma(nk2, delta, r3_s2);
      proj = fma(r3_s2, nq2, proj);
      r3_s3 = fma(nk3, delta, r3_s3);
      proj = fma(r3_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) {
        tg_rec[r_local + 3u] = float(bfloat(proj));
      }
    }

    st0[lane] = r0_s0;
    st0[lane + 32u] = r0_s1;
    st0[lane + 64u] = r0_s2;
    st0[lane + 96u] = r0_s3;

    st1[lane] = r1_s0;
    st1[lane + 32u] = r1_s1;
    st1[lane + 64u] = r1_s2;
    st1[lane + 96u] = r1_s3;

    st2[lane] = r2_s0;
    st2[lane + 32u] = r2_s1;
    st2[lane + 64u] = r2_s2;
    st2[lane + 96u] = r2_s3;

    st3[lane] = r3_s0;
    st3[lane + 32u] = r3_s1;
    st3[lane + 64u] = r3_s2;
    st3[lane + 96u] = r3_s3;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u) {
    float square_sum = 0.0f;
    for (uint feature = lane; feature < 128u; feature += 32u) {
      const float value = tg_rec[feature];
      square_sum = fma(value, value, square_sum);
    }
    const float inverse =
        rsqrt(simd_sum(square_sum) / 128.0f + params.epsilon);
    device const bfloat *gate = z + ulong(value_head) * 128ul;
    for (uint feature = lane; feature < 128u; feature += 32u) {
      const float gate_value = float(gate[feature]);
      const float normalized =
          tg_rec[feature] * inverse * float(norm_weight[feature]);
      if constexpr (OutF16) {
        const float gated =
            float(bfloat(normalized * gate_value / (1.0f + exp(-gate_value))));
        out_f16[ulong(value_head) * 128ul + feature] =
            half(clamp(gated, -65504.0f, 65504.0f));
      } else {
        out_bf16[ulong(value_head) * 128ul + feature] =
            bfloat(normalized * gate_value / (1.0f + exp(-gate_value)));
      }
    }
  } else if constexpr (OutF16) {
    device half *dst_pad =
        out_f16 + ulong(sg) * value_width + ulong(value_head) * 128ul;
    for (uint feature = lane; feature < 128u; feature += 32u) {
      dst_pad[feature] = half(0.0h);
    }
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_gdn_t1_conv_recurrent_norm_gate(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device const bfloat *history [[buffer(2)]],
    device const bfloat *a [[buffer(3)]],
    device const bfloat *b [[buffer(4)]],
    device const bfloat *a_log [[buffer(5)]],
    device const bfloat *dt_bias [[buffer(6)]],
    device float *state [[buffer(7)]],
    device const bfloat *z [[buffer(8)]],
    device const bfloat *norm_weight [[buffer(9)]],
    device bfloat *output [[buffer(10)]],
    constant GdnLayerParams &params [[buffer(11)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  threadgroup float tg_q[128];
  threadgroup float tg_k[128];
  threadgroup float tg_v[128];
  threadgroup float tg_rec[128];
  gdn_t1_conv_recurrent_norm_gate_impl<false>(
      qkv, conv_weight, history, a, b, a_log, dt_bias, state, z, norm_weight,
      output, nullptr, params, tg_q, tg_k, tg_v, tg_rec, lane, sg, tg_position);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_gdn_t1_conv_recurrent_norm_gate_f16(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device const bfloat *history [[buffer(2)]],
    device const bfloat *a [[buffer(3)]],
    device const bfloat *b [[buffer(4)]],
    device const bfloat *a_log [[buffer(5)]],
    device const bfloat *dt_bias [[buffer(6)]],
    device float *state [[buffer(7)]],
    device const bfloat *z [[buffer(8)]],
    device const bfloat *norm_weight [[buffer(9)]],
    device half *output_f16 [[buffer(10)]],
    constant GdnLayerParams &params [[buffer(11)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  threadgroup float tg_q[128];
  threadgroup float tg_k[128];
  threadgroup float tg_v[128];
  threadgroup float tg_rec[128];
  gdn_t1_conv_recurrent_norm_gate_impl<true>(
      qkv, conv_weight, history, a, b, a_log, dt_bias, state, z, norm_weight,
      nullptr, output_f16, params, tg_q, tg_k, tg_v, tg_rec, lane, sg,
      tg_position);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_gdn_commit_history_vec4_t1(
    device bfloat *history [[buffer(0)]],
    device const bfloat *qkv [[buffer(1)]],
    constant GdnLayerParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const uint ch0 = gid * 4u;
  if (ch0 >= params.conv_width || params.tokens == 0u) {
    return;
  }
  device bfloat4 *h0_ptr =
      reinterpret_cast<device bfloat4 *>(history + 0ul * params.conv_width + ch0);
  device bfloat4 *h1_ptr =
      reinterpret_cast<device bfloat4 *>(history + 1ul * params.conv_width + ch0);
  device bfloat4 *h2_ptr =
      reinterpret_cast<device bfloat4 *>(history + 2ul * params.conv_width + ch0);
  const bfloat4 h1_bf = *h1_ptr;
  const bfloat4 h2_bf = *h2_ptr;
  const bfloat4 x0_bf = *reinterpret_cast<device const bfloat4 *>(qkv + ch0);
  *h0_ptr = h1_bf;
  *h1_ptr = h2_bf;
  *h2_ptr = x0_bf;
}

// Speculative verify fused causal conv + SiLU + replay_qkv save:
// Reads `state1` directly as `device const bfloat *history` (eliminating 48
// `state1 -> history` copy dispatches AND 48 dead `commit_history` dispatches)
// and writes `replay_qkv` in-flight (eliminating 48 `copy_rows` dispatches).
// 100% 0-ULP bit-identical to `omlx_m5_gdn_conv_silu_vec4_t8` & `omlx_m5_gdn_conv_silu`.
kernel void omlx_m5_gdn_conv_silu_vec4_t8_verify_save(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device const bfloat *history [[buffer(2)]],
    device bfloat *convolved [[buffer(3)]],
    device bfloat *replay_qkv [[buffer(4)]],
    constant GdnLayerParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint ch0 = gid.x * 4u;
  const uint t_base = gid.y * 8u;
  if (ch0 >= params.conv_width || t_base >= params.tokens) {
    return;
  }
  const uint count = min(8u, params.tokens - t_base);
  if (params.conv_kernel == 4u && (params.conv_width & 3u) == 0u) {
    device const bfloat4 *wp = reinterpret_cast<device const bfloat4 *>(
        conv_weight + ulong(ch0) * 4ul);
    const float4 c0 = float4(wp[0]);
    const float4 c1 = float4(wp[1]);
    const float4 c2 = float4(wp[2]);
    const float4 c3 = float4(wp[3]);
    const float4 w0(c0.x, c1.x, c2.x, c3.x);
    const float4 w1(c0.y, c1.y, c2.y, c3.y);
    const float4 w2(c0.z, c1.z, c2.z, c3.z);
    const float4 w3(c0.w, c1.w, c2.w, c3.w);

    auto load_row = [&](int src_tok) -> float4 {
      device const bfloat *p =
          src_tok < 0
              ? (history + ulong(uint(src_tok + 3)) * ulong(params.conv_width) +
                 ulong(ch0))
              : (qkv + ulong(uint(src_tok)) * ulong(params.conv_width) +
                 ulong(ch0));
      return float4(*reinterpret_cast<device const bfloat4 *>(p));
    };

    float4 xm3 = load_row(int(t_base) - 3);
    float4 xm2 = load_row(int(t_base) - 2);
    float4 xm1 = load_row(int(t_base) - 1);

    for (uint s = 0u; s < count; ++s) {
      const uint tok = t_base + s;
      const ulong off = ulong(tok) * ulong(params.conv_width) + ulong(ch0);
      const bfloat4 xcur_bf =
          *reinterpret_cast<device const bfloat4 *>(qkv + off);
      *reinterpret_cast<device bfloat4 *>(replay_qkv + off) = xcur_bf;
      const float4 xcur = float4(xcur_bf);
      float4 sum = fma(xm3, w0, float4(0.0f));
      sum = fma(xm2, w1, sum);
      sum = fma(xm1, w2, sum);
      sum = fma(xcur, w3, sum);
      *reinterpret_cast<device bfloat4 *>(convolved + off) =
          bfloat4(sum / (1.0f + exp(-sum)));
      xm3 = xm2;
      xm2 = xm1;
      xm1 = xcur;
    }
    return;
  }

  const uint history_tokens = params.conv_kernel - 1u;
  for (uint s = 0u; s < count; ++s) {
    const uint tok = t_base + s;
    for (uint j = 0u; j < 4u && (ch0 + j) < params.conv_width; ++j) {
      const uint ch = ch0 + j;
      const ulong off = ulong(tok) * params.conv_width + ch;
      replay_qkv[off] = qkv[off];
      float sum = 0.0f;
      for (uint tap = 0u; tap < params.conv_kernel; ++tap) {
        const int src_tok = int(tok) + int(tap) - int(history_tokens);
        const float inp =
            src_tok < 0
                ? float(history[ulong(uint(src_tok + int(history_tokens))) *
                                    params.conv_width +
                                ch])
                : float(qkv[ulong(uint(src_tok)) * params.conv_width + ch]);
        sum = fma(inp,
                  float(conv_weight[ulong(ch) * params.conv_kernel + tap]),
                  sum);
      }
      convolved[off] = bfloat(sum / (1.0f + exp(-sum)));
    }
  }
}

// Speculative commit_prefix fused kernel (96 -> 48 dispatches across 48 GDN layers,
// 0 inter-layer barriers): applies accepted prefix updates to `state0` (coalesced
// float4 per lane) and shifts `state1` conv history from `replay_qkv` in 1 launch!
[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_replay_prefix_commit_history(
    device float *state [[buffer(0)]],
    device const float *keys [[buffer(1)]],
    device const float *decays [[buffer(2)]],
    device const float *deltas [[buffer(3)]],
    device bfloat *history [[buffer(4)]],
    device const bfloat *qkv [[buffer(5)]],
    constant GdnLayerParams &p [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if (p.tokens == 0u) {
    return;
  }
  const uint width = p.value_heads * p.value_dim;
  const uint repeat = p.value_heads / p.key_heads;
  if ((p.key_dim & 3u) == 0u) {
    const uint kd4 = p.key_dim >> 2u;
    const uint state_vec4s = width * kd4;
    const uint state_tgs = (state_vec4s + 127u) >> 7u;
    if (tg.x < state_tgs) {
      const uint idx4 = tg.x * 128u + tid;
      if (idx4 >= state_vec4s) {
        return;
      }
      const uint row = idx4 / kd4;
      const uint feat4 = idx4 - row * kd4;
      const uint head = row / p.value_dim;
      const uint key_head = head / repeat;
      device float4 *state4 = reinterpret_cast<device float4 *>(state);
      device const float4 *keys4 =
          reinterpret_cast<device const float4 *>(keys);
      float4 val = state4[idx4];
      for (uint t = 0u; t < p.tokens; ++t) {
        const float dec = decays[ulong(t) * p.value_heads + head];
        const float del = deltas[ulong(t) * width + row];
        const float4 k4 =
            keys4[(ulong(t) * p.key_heads + key_head) * kd4 + feat4];
        val *= dec;
        val.x = fma(k4.x, del, val.x);
        val.y = fma(k4.y, del, val.y);
        val.z = fma(k4.z, del, val.z);
        val.w = fma(k4.w, del, val.w);
      }
      state4[idx4] = val;
      return;
    }
    const uint h_tid = (tg.x - state_tgs) * 128u + tid;
    const uint rows = p.conv_kernel - 1u;
    if ((p.conv_width & 3u) == 0u) {
      const uint ch0 = h_tid * 4u;
      if (ch0 >= p.conv_width) {
        return;
      }
      for (uint r = 0u; r < rows; ++r) {
        const int src = int(p.tokens) - int(rows) + int(r);
        const bfloat4 v =
            src < 0
                ? *reinterpret_cast<device const bfloat4 *>(
                      history + ulong(uint(src + int(rows))) * p.conv_width +
                      ch0)
                : *reinterpret_cast<device const bfloat4 *>(
                      qkv + ulong(uint(src)) * p.conv_width + ch0);
        *reinterpret_cast<device bfloat4 *>(history + ulong(r) * p.conv_width +
                                            ch0) = v;
      }
    } else {
      if (h_tid >= p.conv_width) {
        return;
      }
      for (uint r = 0u; r < rows; ++r) {
        const int src = int(p.tokens) - int(rows) + int(r);
        history[ulong(r) * p.conv_width + h_tid] =
            src < 0 ? history[ulong(uint(src + int(rows))) * p.conv_width + h_tid]
                    : qkv[ulong(uint(src)) * p.conv_width + h_tid];
      }
    }
    return;
  }

  const ulong state_elems = ulong(width) * p.key_dim;
  const uint state_tgs = uint((state_elems + 127ul) >> 7ul);
  if (tg.x < state_tgs) {
    const ulong i = ulong(tg.x) * 128ul + tid;
    if (i >= state_elems) {
      return;
    }
    const uint row = uint(i / p.key_dim);
    const uint feature = uint(i % p.key_dim);
    const uint head = row / p.value_dim;
    const uint key_head = head / repeat;
    float value = state[i];
    for (uint t = 0u; t < p.tokens; ++t) {
      value *= decays[ulong(t) * p.value_heads + head];
      value = fma(keys[ulong(t) * p.key_heads * p.key_dim +
                       ulong(key_head) * p.key_dim + feature],
                  deltas[ulong(t) * width + row], value);
    }
    state[i] = value;
    return;
  }
  const uint channel = (tg.x - state_tgs) * 128u + tid;
  if (channel >= p.conv_width) {
    return;
  }
  const uint rows = p.conv_kernel - 1u;
  for (uint r = 0u; r < rows; ++r) {
    const int src = int(p.tokens) - int(rows) + int(r);
    history[ulong(r) * p.conv_width + channel] =
        src < 0 ? history[ulong(uint(src + int(rows))) * p.conv_width + channel]
                : qkv[ulong(uint(src)) * p.conv_width + channel];
  }
}

// ============================================================================
// Dim D (v0.8.0-m5rt.14): Multi-Slot (grid.z = B, B <= 8) Strided GDN Kernels
// Eliminates per-slot for(b=0..B-1) loops AND emit_split2 copies on all GDN
// layers for both Batched T=1 AR and Batched Speculative Verify (0-ULP exact).
// ============================================================================

struct GdnBatchedT1Params {
  uint key_heads;
  uint value_heads;
  uint key_dim;
  uint value_dim;
  uint conv_width;
  uint conv_kernel;
  uint batch_size;
  float epsilon;
  uint qkv_row_stride;
  uint z_row_stride;
  uint a_row_stride;
  uint b_row_stride;
  uint mp;
};

struct GdnBatchedVerifyParams {
  uint key_heads;
  uint value_heads;
  uint key_dim;
  uint value_dim;
  uint conv_width;
  uint conv_kernel;
  uint batch_size;
  uint total_tokens;
  float epsilon;
  uint qkv_row_stride;
  uint z_row_stride;
  uint a_row_stride;
  uint b_row_stride;
  uint slot_base;
  uint mp;
  uint reserved0;
  uint row_offsets[8];
  uint widths[8];
};

inline device const bfloat *gdn_select_const_bf16_8(
    uint slot,
    device const bfloat *p0, device const bfloat *p1,
    device const bfloat *p2, device const bfloat *p3,
    device const bfloat *p4, device const bfloat *p5,
    device const bfloat *p6, device const bfloat *p7) {
  switch (slot) {
  case 0u: return p0;
  case 1u: return p1;
  case 2u: return p2;
  case 3u: return p3;
  case 4u: return p4;
  case 5u: return p5;
  case 6u: return p6;
  default: return p7;
  }
}

inline device bfloat *gdn_select_bf16_8(
    uint slot,
    device bfloat *p0, device bfloat *p1,
    device bfloat *p2, device bfloat *p3,
    device bfloat *p4, device bfloat *p5,
    device bfloat *p6, device bfloat *p7) {
  switch (slot) {
  case 0u: return p0;
  case 1u: return p1;
  case 2u: return p2;
  case 3u: return p3;
  case 4u: return p4;
  case 5u: return p5;
  case 6u: return p6;
  default: return p7;
  }
}

inline device float *gdn_select_f32_8(
    uint slot,
    device float *p0, device float *p1,
    device float *p2, device float *p3,
    device float *p4, device float *p5,
    device float *p6, device float *p7) {
  switch (slot) {
  case 0u: return p0;
  case 1u: return p1;
  case 2u: return p2;
  case 3u: return p3;
  case 4u: return p4;
  case 5u: return p5;
  case 6u: return p6;
  default: return p7;
  }
}

inline device const float *gdn_select_const_f32_4(
    uint slot,
    device const float *p0, device const float *p1,
    device const float *p2, device const float *p3) {
  switch (slot) {
  case 0u: return p0;
  case 1u: return p1;
  case 2u: return p2;
  default: return p3;
  }
}

inline device float *gdn_select_f32_4(
    uint slot,
    device float *p0, device float *p1,
    device float *p2, device float *p3) {
  switch (slot) {
  case 0u: return p0;
  case 1u: return p1;
  case 2u: return p2;
  default: return p3;
  }
}

inline float4 gdn_load_strided_conv_row4(
    int src_tok,
    device const bfloat *history,
    device const bfloat *qkv_slot,
    uint conv_width,
    uint qkv_row_stride,
    uint ch0) {
  device const bfloat *p =
      src_tok < 0
          ? (history + ulong(uint(src_tok + 3)) * ulong(conv_width) + ulong(ch0))
          : (qkv_slot + ulong(uint(src_tok)) * ulong(qkv_row_stride) + ulong(ch0));
  return float4(*reinterpret_cast<device const bfloat4 *>(p));
}

template <bool OutF16>
inline void gdn_t1_batched_conv_recurrent_norm_gate_impl(
    device const bfloat *qkv,
    device const bfloat *conv_weight,
    device const bfloat *a,
    device const bfloat *b,
    device const bfloat *a_log,
    device const bfloat *dt_bias,
    device const bfloat *z,
    device const bfloat *norm_weight,
    device bfloat *out_bf16,
    device half *out_f16,
    device const bfloat *history,
    device float *state,
    constant GdnBatchedT1Params &params,
    threadgroup float *tg_q,
    threadgroup float *tg_k,
    threadgroup float *tg_v,
    threadgroup float *tg_rec,
    uint lane,
    uint sg,
    uint value_head,
    uint slot) {
  const ulong value_width = ulong(params.value_heads) * 128ul;
  const uint repeat = params.value_heads / params.key_heads;
  const uint key_head = value_head / repeat;
  const uint key_width = params.key_heads * 128u;
  const ulong q_head_offset = ulong(key_head) * 128ul;
  const ulong k_head_offset = ulong(key_width) + q_head_offset;
  const ulong v_head_offset = ulong(2u * key_width) + ulong(value_head) * 128ul;

  device const bfloat *qkv_slot = qkv + ulong(slot) * params.qkv_row_stride;
  if (sg == 0u) {
    gdn_conv_silu_vec4_128ch_to_tg(qkv_slot, conv_weight, history,
                                   params.conv_width, q_head_offset, lane, tg_q);
  } else if (sg == 1u) {
    gdn_conv_silu_vec4_128ch_to_tg(qkv_slot, conv_weight, history,
                                   params.conv_width, k_head_offset, lane, tg_k);
  } else if (sg == 2u) {
    gdn_conv_silu_vec4_128ch_to_tg(qkv_slot, conv_weight, history,
                                   params.conv_width, v_head_offset, lane, tg_v);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float inverse_scale = rsqrt(128.0f);
  const float dt_bias_val = (lane == 0u) ? float(dt_bias[value_head]) : 0.0f;
  const float neg_exp_a = (lane == 0u) ? -exp(float(a_log[value_head])) : 0.0f;

  const float q0 = tg_q[lane];
  const float q1 = tg_q[lane + 32u];
  const float q2 = tg_q[lane + 64u];
  const float q3 = tg_q[lane + 96u];
  const float k0 = tg_k[lane];
  const float k1 = tg_k[lane + 32u];
  const float k2 = tg_k[lane + 64u];
  const float k3 = tg_k[lane + 96u];

  float query_square = fma(q0, q0, 0.0f);
  query_square = fma(q1, q1, query_square);
  query_square = fma(q2, q2, query_square);
  query_square = fma(q3, q3, query_square);

  float key_square = fma(k0, k0, 0.0f);
  key_square = fma(k1, k1, key_square);
  key_square = fma(k2, k2, key_square);
  key_square = fma(k3, k3, key_square);

  const float query_inverse = rsqrt(simd_sum(query_square) / 128.0f + 1.0e-6f);
  const float key_inverse = rsqrt(simd_sum(key_square) / 128.0f + 1.0e-6f);

  float decay = 0.0f;
  float beta = 0.0f;
  if (lane == 0u) {
    const float activation =
        float(a[ulong(slot) * params.a_row_stride + value_head]) + dt_bias_val;
    const float softplus =
        activation > 20.0f ? activation : log(1.0f + exp(activation));
    decay = exp(neg_exp_a * softplus);
    beta = 1.0f / (1.0f + exp(-float(b[ulong(slot) * params.b_row_stride + value_head])));
  }
  decay = simd_broadcast_first(decay);
  beta = simd_broadcast_first(beta);

  const float nk0 = k0 * key_inverse * inverse_scale;
  const float nk1 = k1 * key_inverse * inverse_scale;
  const float nk2 = k2 * key_inverse * inverse_scale;
  const float nk3 = k3 * key_inverse * inverse_scale;
  const float nq0 = q0 * query_inverse * inverse_scale * inverse_scale;
  const float nq1 = q1 * query_inverse * inverse_scale * inverse_scale;
  const float nq2 = q2 * query_inverse * inverse_scale * inverse_scale;
  const float nq3 = q3 * query_inverse * inverse_scale * inverse_scale;

  for (uint base4 = 0u; base4 < 16u; base4 += 4u) {
    const uint r_local = sg * 16u + base4;
    const uint row_base = value_head * 128u + r_local;

    const float v0 = tg_v[r_local + 0u];
    const float v1 = tg_v[r_local + 1u];
    const float v2 = tg_v[r_local + 2u];
    const float v3 = tg_v[r_local + 3u];

    device float *st0 = state + ulong(row_base + 0u) * 128ul;
    device float *st1 = state + ulong(row_base + 1u) * 128ul;
    device float *st2 = state + ulong(row_base + 2u) * 128ul;
    device float *st3 = state + ulong(row_base + 3u) * 128ul;

    float r0_s0 = st0[lane], r0_s1 = st0[lane + 32u], r0_s2 = st0[lane + 64u],
          r0_s3 = st0[lane + 96u];
    float r1_s0 = st1[lane], r1_s1 = st1[lane + 32u], r1_s2 = st1[lane + 64u],
          r1_s3 = st1[lane + 96u];
    float r2_s0 = st2[lane], r2_s1 = st2[lane + 32u], r2_s2 = st2[lane + 64u],
          r2_s3 = st2[lane + 96u];
    float r3_s0 = st3[lane], r3_s1 = st3[lane + 32u], r3_s2 = st3[lane + 64u],
          r3_s3 = st3[lane + 96u];

    {
      r0_s0 *= decay;
      float mem = fma(r0_s0, nk0, 0.0f);
      r0_s1 *= decay;
      mem = fma(r0_s1, nk1, mem);
      r0_s2 *= decay;
      mem = fma(r0_s2, nk2, mem);
      r0_s3 *= decay;
      mem = fma(r0_s3, nk3, mem);
      const float delta = (v0 - simd_sum(mem)) * beta;
      r0_s0 = fma(nk0, delta, r0_s0);
      float proj = fma(r0_s0, nq0, 0.0f);
      r0_s1 = fma(nk1, delta, r0_s1);
      proj = fma(r0_s1, nq1, proj);
      r0_s2 = fma(nk2, delta, r0_s2);
      proj = fma(r0_s2, nq2, proj);
      r0_s3 = fma(nk3, delta, r0_s3);
      proj = fma(r0_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) tg_rec[r_local + 0u] = float(bfloat(proj));
    }
    {
      r1_s0 *= decay;
      float mem = fma(r1_s0, nk0, 0.0f);
      r1_s1 *= decay;
      mem = fma(r1_s1, nk1, mem);
      r1_s2 *= decay;
      mem = fma(r1_s2, nk2, mem);
      r1_s3 *= decay;
      mem = fma(r1_s3, nk3, mem);
      const float delta = (v1 - simd_sum(mem)) * beta;
      r1_s0 = fma(nk0, delta, r1_s0);
      float proj = fma(r1_s0, nq0, 0.0f);
      r1_s1 = fma(nk1, delta, r1_s1);
      proj = fma(r1_s1, nq1, proj);
      r1_s2 = fma(nk2, delta, r1_s2);
      proj = fma(r1_s2, nq2, proj);
      r1_s3 = fma(nk3, delta, r1_s3);
      proj = fma(r1_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) tg_rec[r_local + 1u] = float(bfloat(proj));
    }
    {
      r2_s0 *= decay;
      float mem = fma(r2_s0, nk0, 0.0f);
      r2_s1 *= decay;
      mem = fma(r2_s1, nk1, mem);
      r2_s2 *= decay;
      mem = fma(r2_s2, nk2, mem);
      r2_s3 *= decay;
      mem = fma(r2_s3, nk3, mem);
      const float delta = (v2 - simd_sum(mem)) * beta;
      r2_s0 = fma(nk0, delta, r2_s0);
      float proj = fma(r2_s0, nq0, 0.0f);
      r2_s1 = fma(nk1, delta, r2_s1);
      proj = fma(r2_s1, nq1, proj);
      r2_s2 = fma(nk2, delta, r2_s2);
      proj = fma(r2_s2, nq2, proj);
      r2_s3 = fma(nk3, delta, r2_s3);
      proj = fma(r2_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) tg_rec[r_local + 2u] = float(bfloat(proj));
    }
    {
      r3_s0 *= decay;
      float mem = fma(r3_s0, nk0, 0.0f);
      r3_s1 *= decay;
      mem = fma(r3_s1, nk1, mem);
      r3_s2 *= decay;
      mem = fma(r3_s2, nk2, mem);
      r3_s3 *= decay;
      mem = fma(r3_s3, nk3, mem);
      const float delta = (v3 - simd_sum(mem)) * beta;
      r3_s0 = fma(nk0, delta, r3_s0);
      float proj = fma(r3_s0, nq0, 0.0f);
      r3_s1 = fma(nk1, delta, r3_s1);
      proj = fma(r3_s1, nq1, proj);
      r3_s2 = fma(nk2, delta, r3_s2);
      proj = fma(r3_s2, nq2, proj);
      r3_s3 = fma(nk3, delta, r3_s3);
      proj = fma(r3_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) tg_rec[r_local + 3u] = float(bfloat(proj));
    }

    st0[lane] = r0_s0; st0[lane + 32u] = r0_s1; st0[lane + 64u] = r0_s2; st0[lane + 96u] = r0_s3;
    st1[lane] = r1_s0; st1[lane + 32u] = r1_s1; st1[lane + 64u] = r1_s2; st1[lane + 96u] = r1_s3;
    st2[lane] = r2_s0; st2[lane + 32u] = r2_s1; st2[lane + 64u] = r2_s2; st2[lane + 96u] = r2_s3;
    st3[lane] = r3_s0; st3[lane + 32u] = r3_s1; st3[lane + 64u] = r3_s2; st3[lane + 96u] = r3_s3;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u) {
    float square_sum = 0.0f;
    for (uint feature = lane; feature < 128u; feature += 32u) {
      const float value = tg_rec[feature];
      square_sum = fma(value, value, square_sum);
    }
    const float inverse = rsqrt(simd_sum(square_sum) / 128.0f + params.epsilon);
    device const bfloat *gate =
        z + ulong(slot) * params.z_row_stride + ulong(value_head) * 128ul;
    const ulong dst_base = ulong(slot) * value_width + ulong(value_head) * 128ul;
    for (uint feature = lane; feature < 128u; feature += 32u) {
      const float gate_value = float(gate[feature]);
      const float normalized =
          tg_rec[feature] * inverse * float(norm_weight[feature]);
      if constexpr (OutF16) {
        const float gated =
            float(bfloat(normalized * gate_value / (1.0f + exp(-gate_value))));
        out_f16[dst_base + feature] = half(clamp(gated, -65504.0f, 65504.0f));
      } else {
        out_bf16[dst_base + feature] =
            bfloat(normalized * gate_value / (1.0f + exp(-gate_value)));
      }
    }
  }
  if constexpr (OutF16) {
    if (slot == 0u) {
      for (uint pad_row = params.batch_size + sg; pad_row < params.mp; pad_row += 8u) {
        device half *dst_pad =
            out_f16 + ulong(pad_row) * value_width + ulong(value_head) * 128ul;
        for (uint feature = lane; feature < 128u; feature += 32u) {
          dst_pad[feature] = half(0.0h);
        }
      }
    }
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_gdn_t1_batched_conv_recurrent_norm_gate(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device const bfloat *a [[buffer(2)]],
    device const bfloat *b [[buffer(3)]],
    device const bfloat *a_log [[buffer(4)]],
    device const bfloat *dt_bias [[buffer(5)]],
    device const bfloat *z [[buffer(6)]],
    device const bfloat *norm_weight [[buffer(7)]],
    device bfloat *output [[buffer(8)]],
    device const bfloat *h0 [[buffer(9)]],
    device const bfloat *h1 [[buffer(10)]],
    device const bfloat *h2 [[buffer(11)]],
    device const bfloat *h3 [[buffer(12)]],
    device const bfloat *h4 [[buffer(13)]],
    device const bfloat *h5 [[buffer(14)]],
    device const bfloat *h6 [[buffer(15)]],
    device const bfloat *h7 [[buffer(16)]],
    device float *st0 [[buffer(17)]],
    device float *st1 [[buffer(18)]],
    device float *st2 [[buffer(19)]],
    device float *st3 [[buffer(20)]],
    device float *st4 [[buffer(21)]],
    device float *st5 [[buffer(22)]],
    device float *st6 [[buffer(23)]],
    device float *st7 [[buffer(24)]],
    constant GdnBatchedT1Params &params [[buffer(25)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint value_head = tg_position.y;
  const uint slot = tg_position.z;
  if (value_head >= params.value_heads || slot >= params.batch_size) {
    return;
  }
  threadgroup float tg_q[128];
  threadgroup float tg_k[128];
  threadgroup float tg_v[128];
  threadgroup float tg_rec[128];
  device const bfloat *hist =
      gdn_select_const_bf16_8(slot, h0, h1, h2, h3, h4, h5, h6, h7);
  device float *st =
      gdn_select_f32_8(slot, st0, st1, st2, st3, st4, st5, st6, st7);
  gdn_t1_batched_conv_recurrent_norm_gate_impl<false>(
      qkv, conv_weight, a, b, a_log, dt_bias, z, norm_weight, output, nullptr,
      hist, st, params, tg_q, tg_k, tg_v, tg_rec, lane, sg, value_head, slot);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_gdn_t1_batched_conv_recurrent_norm_gate_f16(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device const bfloat *a [[buffer(2)]],
    device const bfloat *b [[buffer(3)]],
    device const bfloat *a_log [[buffer(4)]],
    device const bfloat *dt_bias [[buffer(5)]],
    device const bfloat *z [[buffer(6)]],
    device const bfloat *norm_weight [[buffer(7)]],
    device half *output_f16 [[buffer(8)]],
    device const bfloat *h0 [[buffer(9)]],
    device const bfloat *h1 [[buffer(10)]],
    device const bfloat *h2 [[buffer(11)]],
    device const bfloat *h3 [[buffer(12)]],
    device const bfloat *h4 [[buffer(13)]],
    device const bfloat *h5 [[buffer(14)]],
    device const bfloat *h6 [[buffer(15)]],
    device const bfloat *h7 [[buffer(16)]],
    device float *st0 [[buffer(17)]],
    device float *st1 [[buffer(18)]],
    device float *st2 [[buffer(19)]],
    device float *st3 [[buffer(20)]],
    device float *st4 [[buffer(21)]],
    device float *st5 [[buffer(22)]],
    device float *st6 [[buffer(23)]],
    device float *st7 [[buffer(24)]],
    constant GdnBatchedT1Params &params [[buffer(25)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint value_head = tg_position.y;
  const uint slot = tg_position.z;
  if (value_head >= params.value_heads || slot >= params.batch_size) {
    return;
  }
  threadgroup float tg_q[128];
  threadgroup float tg_k[128];
  threadgroup float tg_v[128];
  threadgroup float tg_rec[128];
  device const bfloat *hist =
      gdn_select_const_bf16_8(slot, h0, h1, h2, h3, h4, h5, h6, h7);
  device float *st =
      gdn_select_f32_8(slot, st0, st1, st2, st3, st4, st5, st6, st7);
  gdn_t1_batched_conv_recurrent_norm_gate_impl<true>(
      qkv, conv_weight, a, b, a_log, dt_bias, z, norm_weight, nullptr,
      output_f16, hist, st, params, tg_q, tg_k, tg_v, tg_rec, lane, sg,
      value_head, slot);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_gdn_commit_history_batched_vec4_t1(
    device const bfloat *qkv [[buffer(0)]],
    device bfloat *h0 [[buffer(1)]],
    device bfloat *h1 [[buffer(2)]],
    device bfloat *h2 [[buffer(3)]],
    device bfloat *h3 [[buffer(4)]],
    device bfloat *h4 [[buffer(5)]],
    device bfloat *h5 [[buffer(6)]],
    device bfloat *h6 [[buffer(7)]],
    device bfloat *h7 [[buffer(8)]],
    constant GdnBatchedT1Params &params [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint ch0 = gid.x * 4u;
  const uint slot = gid.y;
  if (ch0 >= params.conv_width || slot >= params.batch_size) {
    return;
  }
  device bfloat *history =
      gdn_select_bf16_8(slot, h0, h1, h2, h3, h4, h5, h6, h7);
  device bfloat4 *r0 =
      reinterpret_cast<device bfloat4 *>(history + 0ul * params.conv_width + ch0);
  device bfloat4 *r1 =
      reinterpret_cast<device bfloat4 *>(history + 1ul * params.conv_width + ch0);
  device bfloat4 *r2 =
      reinterpret_cast<device bfloat4 *>(history + 2ul * params.conv_width + ch0);
  const bfloat4 h1_bf = *r1;
  const bfloat4 h2_bf = *r2;
  const bfloat4 x0_bf = *reinterpret_cast<device const bfloat4 *>(
      qkv + ulong(slot) * params.qkv_row_stride + ch0);
  *r0 = h1_bf;
  *r1 = h2_bf;
  *r2 = x0_bf;
}

kernel void omlx_m5_gdn_conv_silu_vec4_t8_verify_save_batched(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device bfloat *convolved [[buffer(2)]],
    device const bfloat *h0 [[buffer(3)]],
    device const bfloat *h1 [[buffer(4)]],
    device const bfloat *h2 [[buffer(5)]],
    device const bfloat *h3 [[buffer(6)]],
    device const bfloat *h4 [[buffer(7)]],
    device const bfloat *h5 [[buffer(8)]],
    device const bfloat *h6 [[buffer(9)]],
    device const bfloat *h7 [[buffer(10)]],
    device bfloat *rqk0 [[buffer(11)]],
    device bfloat *rqk1 [[buffer(12)]],
    device bfloat *rqk2 [[buffer(13)]],
    device bfloat *rqk3 [[buffer(14)]],
    device bfloat *rqk4 [[buffer(15)]],
    device bfloat *rqk5 [[buffer(16)]],
    device bfloat *rqk6 [[buffer(17)]],
    device bfloat *rqk7 [[buffer(18)]],
    constant GdnBatchedVerifyParams &params [[buffer(19)]],
    uint3 gid [[thread_position_in_grid]]) {
  const uint ch0 = gid.x * 4u;
  const uint t_base = gid.y * 8u;
  const uint slot = gid.z;
  if (slot >= params.batch_size) {
    return;
  }
  const uint tokens = params.widths[slot];
  if (ch0 >= params.conv_width || t_base >= tokens) {
    return;
  }
  const uint row_off = params.row_offsets[slot];
  device const bfloat *qkv_slot = qkv + ulong(row_off) * params.qkv_row_stride;
  device bfloat *conv_slot = convolved + ulong(row_off) * params.conv_width;
  device const bfloat *history =
      gdn_select_const_bf16_8(slot, h0, h1, h2, h3, h4, h5, h6, h7);
  device bfloat *replay_qkv =
      gdn_select_bf16_8(slot, rqk0, rqk1, rqk2, rqk3, rqk4, rqk5, rqk6, rqk7);
  const uint count = min(8u, tokens - t_base);

  if (params.conv_kernel == 4u && (params.conv_width & 3u) == 0u) {
    device const bfloat4 *wp = reinterpret_cast<device const bfloat4 *>(
        conv_weight + ulong(ch0) * 4ul);
    const float4 c0 = float4(wp[0]);
    const float4 c1 = float4(wp[1]);
    const float4 c2 = float4(wp[2]);
    const float4 c3 = float4(wp[3]);
    const float4 w0(c0.x, c1.x, c2.x, c3.x);
    const float4 w1(c0.y, c1.y, c2.y, c3.y);
    const float4 w2(c0.z, c1.z, c2.z, c3.z);
    const float4 w3(c0.w, c1.w, c2.w, c3.w);

    float4 xm3 = gdn_load_strided_conv_row4(int(t_base) - 3, history, qkv_slot,
                                            params.conv_width,
                                            params.qkv_row_stride, ch0);
    float4 xm2 = gdn_load_strided_conv_row4(int(t_base) - 2, history, qkv_slot,
                                            params.conv_width,
                                            params.qkv_row_stride, ch0);
    float4 xm1 = gdn_load_strided_conv_row4(int(t_base) - 1, history, qkv_slot,
                                            params.conv_width,
                                            params.qkv_row_stride, ch0);

    for (uint s = 0u; s < count; ++s) {
      const uint tok = t_base + s;
      const ulong qkv_off = ulong(tok) * ulong(params.qkv_row_stride) + ulong(ch0);
      const ulong dst_off = ulong(tok) * ulong(params.conv_width) + ulong(ch0);
      const bfloat4 xcur_bf =
          *reinterpret_cast<device const bfloat4 *>(qkv_slot + qkv_off);
      *reinterpret_cast<device bfloat4 *>(replay_qkv + dst_off) = xcur_bf;
      const float4 xcur = float4(xcur_bf);
      float4 sum = fma(xm3, w0, float4(0.0f));
      sum = fma(xm2, w1, sum);
      sum = fma(xm1, w2, sum);
      sum = fma(xcur, w3, sum);
      *reinterpret_cast<device bfloat4 *>(conv_slot + dst_off) =
          bfloat4(sum / (1.0f + exp(-sum)));
      xm3 = xm2;
      xm2 = xm1;
      xm1 = xcur;
    }
    return;
  }

  const uint history_tokens = params.conv_kernel - 1u;
  for (uint s = 0u; s < count; ++s) {
    const uint tok = t_base + s;
    for (uint j = 0u; j < 4u && (ch0 + j) < params.conv_width; ++j) {
      const uint ch = ch0 + j;
      const ulong dst_off = ulong(tok) * params.conv_width + ch;
      replay_qkv[dst_off] = qkv_slot[ulong(tok) * params.qkv_row_stride + ch];
      float sum = 0.0f;
      for (uint tap = 0u; tap < params.conv_kernel; ++tap) {
        const int src_tok = int(tok) + int(tap) - int(history_tokens);
        const float inp =
            src_tok < 0
                ? float(history[ulong(uint(src_tok + int(history_tokens))) *
                                    params.conv_width +
                                ch])
                : float(qkv_slot[ulong(uint(src_tok)) * params.qkv_row_stride + ch]);
        sum = fma(inp,
                  float(conv_weight[ulong(ch) * params.conv_kernel + tap]),
                  sum);
      }
      conv_slot[dst_off] = bfloat(sum / (1.0f + exp(-sum)));
    }
  }
}

kernel void omlx_m5_gdn_recurrent_record_4row_batched4(
    device const bfloat *convolved [[buffer(0)]],
    device const bfloat *a [[buffer(1)]],
    device const bfloat *b [[buffer(2)]],
    device const bfloat *a_log [[buffer(3)]],
    device const bfloat *dt_bias [[buffer(4)]],
    device bfloat *recurrent [[buffer(5)]],
    device const float *st0_buf [[buffer(6)]],
    device float *rk0_buf [[buffer(7)]],
    device float *rd0_buf [[buffer(8)]],
    device float *rdel0_buf [[buffer(9)]],
    device const float *st1_buf [[buffer(10)]],
    device float *rk1_buf [[buffer(11)]],
    device float *rd1_buf [[buffer(12)]],
    device float *rdel1_buf [[buffer(13)]],
    device const float *st2_buf [[buffer(14)]],
    device float *rk2_buf [[buffer(15)]],
    device float *rd2_buf [[buffer(16)]],
    device float *rdel2_buf [[buffer(17)]],
    device const float *st3_buf [[buffer(18)]],
    device float *rk3_buf [[buffer(19)]],
    device float *rd3_buf [[buffer(20)]],
    device float *rdel3_buf [[buffer(21)]],
    constant GdnBatchedVerifyParams &params [[buffer(22)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint row_base = tg_position.y * 4u;
  const uint local_slot = tg_position.z;
  const uint slot = params.slot_base + local_slot;
  const uint value_width = params.value_heads * params.value_dim;
  if (row_base >= value_width || local_slot >= 4u || slot >= params.batch_size) {
    return;
  }
  const uint tokens = params.widths[slot];
  if (tokens == 0u) {
    return;
  }
  const uint row_off = params.row_offsets[slot];
  device const float *state =
      gdn_select_const_f32_4(local_slot, st0_buf, st1_buf, st2_buf, st3_buf);
  device float *record_keys =
      gdn_select_f32_4(local_slot, rk0_buf, rk1_buf, rk2_buf, rk3_buf);
  device float *record_decays =
      gdn_select_f32_4(local_slot, rd0_buf, rd1_buf, rd2_buf, rd3_buf);
  device float *record_deltas =
      gdn_select_f32_4(local_slot, rdel0_buf, rdel1_buf, rdel2_buf, rdel3_buf);

  const uint value_head = row_base / params.value_dim;
  const uint value_index0 = row_base - value_head * params.value_dim;
  const uint repeat = params.value_heads / params.key_heads;
  const uint key_head = value_head / repeat;
  const uint key_width = params.key_heads * 128u;
  const float inverse_scale = rsqrt(128.0f);
  const float dt_bias_val = lane == 0u ? float(dt_bias[value_head]) : 0.0f;
  const float neg_exp_a = lane == 0u ? -exp(float(a_log[value_head])) : 0.0f;
  const ulong q_head_offset = ulong(key_head) * 128ul;
  const ulong k_head_offset = ulong(key_width) + q_head_offset;
  const ulong v_base_offset =
      ulong(2u * key_width) + ulong(value_head * params.value_dim + value_index0);

  device const float *st0 = state + ulong(row_base + 0u) * 128ul;
  device const float *st1 = state + ulong(row_base + 1u) * 128ul;
  device const float *st2 = state + ulong(row_base + 2u) * 128ul;
  device const float *st3 = state + ulong(row_base + 3u) * 128ul;

  float r0_s0 = st0[lane], r0_s1 = st0[lane + 32u], r0_s2 = st0[lane + 64u],
        r0_s3 = st0[lane + 96u];
  float r1_s0 = st1[lane], r1_s1 = st1[lane + 32u], r1_s2 = st1[lane + 64u],
        r1_s3 = st1[lane + 96u];
  float r2_s0 = st2[lane], r2_s1 = st2[lane + 32u], r2_s2 = st2[lane + 64u],
        r2_s3 = st2[lane + 96u];
  float r3_s0 = st3[lane], r3_s1 = st3[lane + 32u], r3_s2 = st3[lane + 64u],
        r3_s3 = st3[lane + 96u];

  for (uint token = 0u; token < tokens; ++token) {
    const uint global_tok = row_off + token;
    device const bfloat *token_row =
        convolved + ulong(global_tok) * params.conv_width;
    device const bfloat *query = token_row + q_head_offset;
    device const bfloat *key = token_row + k_head_offset;

    const float q0 = float(query[lane]);
    const float q1 = float(query[lane + 32u]);
    const float q2 = float(query[lane + 64u]);
    const float q3 = float(query[lane + 96u]);
    const float k0 = float(key[lane]);
    const float k1 = float(key[lane + 32u]);
    const float k2 = float(key[lane + 64u]);
    const float k3 = float(key[lane + 96u]);

    float query_square = fma(q0, q0, 0.0f);
    query_square = fma(q1, q1, query_square);
    query_square = fma(q2, q2, query_square);
    query_square = fma(q3, q3, query_square);

    float key_square = fma(k0, k0, 0.0f);
    key_square = fma(k1, k1, key_square);
    key_square = fma(k2, k2, key_square);
    key_square = fma(k3, k3, key_square);

    const float query_inverse = rsqrt(simd_sum(query_square) / 128.0f + 1.0e-6f);
    const float key_inverse = rsqrt(simd_sum(key_square) / 128.0f + 1.0e-6f);

    float decay = 0.0f;
    float beta = 0.0f;
    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    if (lane == 0u) {
      const float activation =
          float(a[ulong(global_tok) * params.a_row_stride + value_head]) +
          dt_bias_val;
      const float softplus =
          activation > 20.0f ? activation : log(1.0f + exp(activation));
      decay = exp(neg_exp_a * softplus);
      beta = 1.0f /
             (1.0f + exp(-float(b[ulong(global_tok) * params.b_row_stride + value_head])));
      const bfloat4 v4 =
          *reinterpret_cast<device const bfloat4 *>(token_row + v_base_offset);
      v0 = float(v4.x); v1 = float(v4.y); v2 = float(v4.z); v3 = float(v4.w);
    }
    decay = simd_broadcast_first(decay);
    beta = simd_broadcast_first(beta);
    v0 = simd_broadcast_first(v0);
    v1 = simd_broadcast_first(v1);
    v2 = simd_broadcast_first(v2);
    v3 = simd_broadcast_first(v3);

    const float nk0 = k0 * key_inverse * inverse_scale;
    const float nk1 = k1 * key_inverse * inverse_scale;
    const float nk2 = k2 * key_inverse * inverse_scale;
    const float nk3 = k3 * key_inverse * inverse_scale;
    const float nq0 = q0 * query_inverse * inverse_scale * inverse_scale;
    const float nq1 = q1 * query_inverse * inverse_scale * inverse_scale;
    const float nq2 = q2 * query_inverse * inverse_scale * inverse_scale;
    const float nq3 = q3 * query_inverse * inverse_scale * inverse_scale;

    if (value_index0 == 0u) {
      if (lane == 0u) {
        record_decays[ulong(token) * params.value_heads + value_head] = decay;
      }
      if ((value_head % repeat) == 0u) {
        const ulong k_rec_base = ulong(token) * key_width + q_head_offset;
        record_keys[k_rec_base + lane] = nk0;
        record_keys[k_rec_base + lane + 32u] = nk1;
        record_keys[k_rec_base + lane + 64u] = nk2;
        record_keys[k_rec_base + lane + 96u] = nk3;
      }
    }

    const ulong rec_out_base = ulong(global_tok) * value_width + row_base;
    const ulong del_out_base = ulong(token) * value_width + row_base;

    {
      r0_s0 *= decay;
      float mem = fma(r0_s0, nk0, 0.0f);
      r0_s1 *= decay;
      mem = fma(r0_s1, nk1, mem);
      r0_s2 *= decay;
      mem = fma(r0_s2, nk2, mem);
      r0_s3 *= decay;
      mem = fma(r0_s3, nk3, mem);
      const float delta = (v0 - simd_sum(mem)) * beta;
      r0_s0 = fma(nk0, delta, r0_s0);
      float proj = fma(r0_s0, nq0, 0.0f);
      r0_s1 = fma(nk1, delta, r0_s1);
      proj = fma(r0_s1, nq1, proj);
      r0_s2 = fma(nk2, delta, r0_s2);
      proj = fma(r0_s2, nq2, proj);
      r0_s3 = fma(nk3, delta, r0_s3);
      proj = fma(r0_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) {
        record_deltas[del_out_base + 0u] = delta;
        recurrent[rec_out_base + 0u] = bfloat(proj);
      }
    }
    {
      r1_s0 *= decay;
      float mem = fma(r1_s0, nk0, 0.0f);
      r1_s1 *= decay;
      mem = fma(r1_s1, nk1, mem);
      r1_s2 *= decay;
      mem = fma(r1_s2, nk2, mem);
      r1_s3 *= decay;
      mem = fma(r1_s3, nk3, mem);
      const float delta = (v1 - simd_sum(mem)) * beta;
      r1_s0 = fma(nk0, delta, r1_s0);
      float proj = fma(r1_s0, nq0, 0.0f);
      r1_s1 = fma(nk1, delta, r1_s1);
      proj = fma(r1_s1, nq1, proj);
      r1_s2 = fma(nk2, delta, r1_s2);
      proj = fma(r1_s2, nq2, proj);
      r1_s3 = fma(nk3, delta, r1_s3);
      proj = fma(r1_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) {
        record_deltas[del_out_base + 1u] = delta;
        recurrent[rec_out_base + 1u] = bfloat(proj);
      }
    }
    {
      r2_s0 *= decay;
      float mem = fma(r2_s0, nk0, 0.0f);
      r2_s1 *= decay;
      mem = fma(r2_s1, nk1, mem);
      r2_s2 *= decay;
      mem = fma(r2_s2, nk2, mem);
      r2_s3 *= decay;
      mem = fma(r2_s3, nk3, mem);
      const float delta = (v2 - simd_sum(mem)) * beta;
      r2_s0 = fma(nk0, delta, r2_s0);
      float proj = fma(r2_s0, nq0, 0.0f);
      r2_s1 = fma(nk1, delta, r2_s1);
      proj = fma(r2_s1, nq1, proj);
      r2_s2 = fma(nk2, delta, r2_s2);
      proj = fma(r2_s2, nq2, proj);
      r2_s3 = fma(nk3, delta, r2_s3);
      proj = fma(r2_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) {
        record_deltas[del_out_base + 2u] = delta;
        recurrent[rec_out_base + 2u] = bfloat(proj);
      }
    }
    {
      r3_s0 *= decay;
      float mem = fma(r3_s0, nk0, 0.0f);
      r3_s1 *= decay;
      mem = fma(r3_s1, nk1, mem);
      r3_s2 *= decay;
      mem = fma(r3_s2, nk2, mem);
      r3_s3 *= decay;
      mem = fma(r3_s3, nk3, mem);
      const float delta = (v3 - simd_sum(mem)) * beta;
      r3_s0 = fma(nk0, delta, r3_s0);
      float proj = fma(r3_s0, nq0, 0.0f);
      r3_s1 = fma(nk1, delta, r3_s1);
      proj = fma(r3_s1, nq1, proj);
      r3_s2 = fma(nk2, delta, r3_s2);
      proj = fma(r3_s2, nq2, proj);
      r3_s3 = fma(nk3, delta, r3_s3);
      proj = fma(r3_s3, nq3, proj);
      proj = simd_sum(proj);
      if (lane == 0u) {
        record_deltas[del_out_base + 3u] = delta;
        recurrent[rec_out_base + 3u] = bfloat(proj);
      }
    }
  }
}

kernel void omlx_m5_gdn_norm_gate_strided(
    device const bfloat *recurrent [[buffer(0)]],
    device const bfloat *z [[buffer(1)]],
    device const bfloat *norm_weight [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant GdnBatchedVerifyParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint logical_head = tg_position.y;
  const uint token = logical_head / params.value_heads;
  if (token >= params.total_tokens) {
    return;
  }
  const uint value_head = logical_head - token * params.value_heads;
  const ulong value_width = ulong(params.value_heads) * params.value_dim;
  device const bfloat *row =
      recurrent + ulong(token) * value_width + ulong(value_head) * params.value_dim;
  device const bfloat *gate =
      z + ulong(token) * params.z_row_stride + ulong(value_head) * params.value_dim;
  device bfloat *destination =
      output + ulong(token) * value_width + ulong(value_head) * params.value_dim;
  float square_sum = 0.0f;
  for (uint feature = lane; feature < params.value_dim; feature += 32u) {
    const float value = float(row[feature]);
    square_sum = fma(value, value, square_sum);
  }
  const float inverse =
      rsqrt(simd_sum(square_sum) / float(params.value_dim) + params.epsilon);
  for (uint feature = lane; feature < params.value_dim; feature += 32u) {
    const float gate_value = float(gate[feature]);
    const float normalized =
        float(row[feature]) * inverse * float(norm_weight[feature]);
    destination[feature] =
        bfloat(normalized * gate_value / (1.0f + exp(-gate_value)));
  }
}

kernel void omlx_m5_gdn_norm_gate_f16_strided(
    device const bfloat *recurrent [[buffer(0)]],
    device const bfloat *z [[buffer(1)]],
    device const bfloat *norm_weight [[buffer(2)]],
    device half *output_f16 [[buffer(3)]],
    constant GdnBatchedVerifyParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint logical_head = tg_position.y;
  const uint token = logical_head / params.value_heads;
  if (token >= params.mp) {
    return;
  }
  const uint value_head = logical_head - token * params.value_heads;
  const ulong value_width = ulong(params.value_heads) * params.value_dim;
  device half *destination =
      output_f16 + ulong(token) * value_width + ulong(value_head) * params.value_dim;
  if (token >= params.total_tokens) {
    for (uint feature = lane; feature < params.value_dim; feature += 32u) {
      destination[feature] = half(0.0h);
    }
    return;
  }
  device const bfloat *row =
      recurrent + ulong(token) * value_width + ulong(value_head) * params.value_dim;
  device const bfloat *gate =
      z + ulong(token) * params.z_row_stride + ulong(value_head) * params.value_dim;
  float square_sum = 0.0f;
  for (uint feature = lane; feature < params.value_dim; feature += 32u) {
    const float value = float(row[feature]);
    square_sum = fma(value, value, square_sum);
  }
  const float inverse =
      rsqrt(simd_sum(square_sum) / float(params.value_dim) + params.epsilon);
  for (uint feature = lane; feature < params.value_dim; feature += 32u) {
    const float gate_value = float(gate[feature]);
    const float normalized =
        float(row[feature]) * inverse * float(norm_weight[feature]);
    const float gated =
        float(bfloat(normalized * gate_value / (1.0f + exp(-gate_value))));
    destination[feature] = half(clamp(gated, -65504.0f, 65504.0f));
  }
}
