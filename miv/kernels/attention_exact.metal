// P06-TF + W2-B P0-5: fixed per-row sequential arithmetic and explicit logical
// positions, with GQA + multi-row (BQ=1..8) cooperative threadgroup KV tile
// sharing when launched with threads_per_threadgroup.y > 1 (0 ULP identical).
#include "attention_postprocess.metal"
#include "attention_split_kv.metal"

inline float exact_dot_hd256(float4 q0, float4 q1, device const bfloat *k_row,
                             uint lane) {
  device const bfloat4 *kv =
      reinterpret_cast<device const bfloat4 *>(k_row + lane * 8u);
  const bfloat4 k0 = kv[0];
  const bfloat4 k1 = kv[1];
  float dot = fma(q0.x, float(k0.x), 0.0f);
  dot = fma(q0.y, float(k0.y), dot);
  dot = fma(q0.z, float(k0.z), dot);
  dot = fma(q0.w, float(k0.w), dot);
  dot = fma(q1.x, float(k1.x), dot);
  dot = fma(q1.y, float(k1.y), dot);
  dot = fma(q1.z, float(k1.z), dot);
  dot = fma(q1.w, float(k1.w), dot);
  return simd_sum(dot);
}

inline void exact_v_accum_hd256(float prob, device const bfloat *v_row,
                                uint lane, thread float4 &acc0,
                                thread float4 &acc1) {
  device const bfloat4 *vv =
      reinterpret_cast<device const bfloat4 *>(v_row + lane * 8u);
  const bfloat4 v0 = vv[0];
  const bfloat4 v1 = vv[1];
  acc0 = fma(float4(prob), float4(v0), acc0);
  acc1 = fma(float4(prob), float4(v1), acc1);
}

inline float exact_dot_hd256_tg(float4 q0, float4 q1,
                                threadgroup const bfloat4 *kv_base, uint lane) {
  const bfloat4 k0 = kv_base[lane * 2u + 0u];
  const bfloat4 k1 = kv_base[lane * 2u + 1u];
  float dot = fma(q0.x, float(k0.x), 0.0f);
  dot = fma(q0.y, float(k0.y), dot);
  dot = fma(q0.z, float(k0.z), dot);
  dot = fma(q0.w, float(k0.w), dot);
  dot = fma(q1.x, float(k1.x), dot);
  dot = fma(q1.y, float(k1.y), dot);
  dot = fma(q1.z, float(k1.z), dot);
  dot = fma(q1.w, float(k1.w), dot);
  return simd_sum(dot);
}

inline void exact_v_accum_hd256_tg(float prob,
                                   threadgroup const bfloat4 *vv_base,
                                   uint lane, thread float4 &acc0,
                                   thread float4 &acc1) {
  const bfloat4 v0 = vv_base[lane * 2u + 0u];
  const bfloat4 v1 = vv_base[lane * 2u + 1u];
  acc0 = fma(float4(prob), float4(v0), acc0);
  acc1 = fma(float4(prob), float4(v1), acc1);
}

constexpr constant uint kExactKvTileRows = 8u;

kernel void omlx_m5_paged_attention_exact_max(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *key [[buffer(1)]],
    device float *partial_max [[buffer(2)]],
    device const uint *table [[buffer(3)]],
    device const uint *positions [[buffer(4)]],
    constant PagedSplitAttentionParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint repeat = p.query_heads / p.key_value_heads;
  if (tptg.y > 1u && p.head_dim == 256u && repeat > 0u &&
      (tptg.y % repeat) == 0u && tptg.y <= 32u) {
    threadgroup bfloat4 tg_k[kExactKvTileRows * 64u];
    threadgroup uint tg_row_last[32];

    const uint split = tg.x;
    const uint bq = tptg.y / repeat;
    const uint kvh = tg.y % p.key_value_heads;
    const uint t_base = (tg.y / p.key_value_heads) * bq;
    const uint sg_y = tid_in_tg.y;
    const uint num_sgs = tptg.y;
    const uint flat_tid = sg_y * 32u + lane;
    const uint total_threads = num_sgs * 32u;

    const uint row = t_base + (sg_y / repeat);
    const uint qh = kvh * repeat + (sg_y % repeat);
    const bool active = (split < p.split_count) && (row < p.tokens);

    const uint first = split * p.split_tokens;
    const uint last =
        active ? min(first + p.split_tokens, positions[row] + 1u) : first;
    if (lane == 0u) {
      tg_row_last[sg_y] = (first < last) ? last : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint max_last = 0u;
    for (uint s = 0; s < num_sgs; ++s) {
      max_last = max(max_last, tg_row_last[s]);
    }

    const ulong width = ulong(p.key_value_heads) * 256u;
    const float scale = rsqrt(256.0f);
    float maximum = -INFINITY;
    float4 q0 = float4(0.0f), q1 = float4(0.0f);
    if (active && first < last) {
      device const bfloat *q =
          query + ulong(row) * p.query_token_stride + ulong(qh) * p.query_head_stride;
      device const bfloat4 *qv =
          reinterpret_cast<device const bfloat4 *>(q + lane * 8u);
      q0 = float4(qv[0]);
      q1 = float4(qv[1]);
    }

    if (first < max_last) {
      const uint first_page = first / p.page_tokens;
      const uint last_page = (max_last - 1u) / p.page_tokens;
      for (uint page = first_page; page <= last_page; ++page) {
        const ulong page_base = ulong(table[page]) * p.page_tokens;
        const uint p_start = page * p.page_tokens;
        const uint page_begin = max(first, p_start) - p_start;
        const uint page_end_max =
            min(max_last, p_start + p.page_tokens) - p_start;
        const uint row_end =
            (active && last > p_start)
                ? (min(last, p_start + p.page_tokens) - p_start)
                : 0u;
        device const bfloat *k_page =
            key + page_base * width + ulong(kvh) * 256u;

        for (uint tile_begin = page_begin; tile_begin < page_end_max;
             tile_begin += kExactKvTileRows) {
          const uint tile_len =
              min(kExactKvTileRows, page_end_max - tile_begin);
          const uint total_vecs = tile_len * 64u;
          for (uint idx = flat_tid; idx < total_vecs; idx += total_threads) {
            const uint w_off = idx >> 6u;
            const uint v_off = idx & 63u;
            device const bfloat4 *src = reinterpret_cast<device const bfloat4 *>(
                k_page + ulong(tile_begin + w_off) * width);
            tg_k[idx] = src[v_off];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (active && row_end > tile_begin) {
            const uint local_valid = min(tile_len, row_end - tile_begin);
            uint w = 0u;
            for (; w + 3u < local_valid; w += 4u) {
              const float s0 =
                  exact_dot_hd256_tg(q0, q1, tg_k + (w + 0u) * 64u, lane) *
                  scale;
              const float s1 =
                  exact_dot_hd256_tg(q0, q1, tg_k + (w + 1u) * 64u, lane) *
                  scale;
              const float s2 =
                  exact_dot_hd256_tg(q0, q1, tg_k + (w + 2u) * 64u, lane) *
                  scale;
              const float s3 =
                  exact_dot_hd256_tg(q0, q1, tg_k + (w + 3u) * 64u, lane) *
                  scale;
              maximum = max(maximum, s0);
              maximum = max(maximum, s1);
              maximum = max(maximum, s2);
              maximum = max(maximum, s3);
            }
            for (; w < local_valid; ++w) {
              const float s0 =
                  exact_dot_hd256_tg(q0, q1, tg_k + w * 64u, lane) * scale;
              maximum = max(maximum, s0);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    }

    if (active && lane == 0u) {
      const ulong logical_head = ulong(row) * p.query_heads + qh;
      partial_max[logical_head * p.split_count + split] = maximum;
    }
    return;
  }

  const uint split = tg.x, row = tg.y / p.query_heads, qh = tg.y % p.query_heads;
  if (split >= p.split_count || row >= p.tokens)
    return;
  const uint kvh = qh / repeat;
  const ulong width = ulong(p.key_value_heads) * p.head_dim;
  const uint first = split * p.split_tokens,
             last = min(first + p.split_tokens, positions[row] + 1);
  const device bfloat *q =
      query + ulong(row) * p.query_token_stride + ulong(qh) * p.query_head_stride;
  const float scale = rsqrt(float(p.head_dim));
  float maximum = -INFINITY;
  if (first < last) {
    if (p.head_dim == 256) {
      device const bfloat4 *qv =
          reinterpret_cast<device const bfloat4 *>(q + lane * 8u);
      const float4 q0 = float4(qv[0]);
      const float4 q1 = float4(qv[1]);
      const uint first_page = first / p.page_tokens;
      const uint last_page = (last - 1u) / p.page_tokens;
      for (uint page = first_page; page <= last_page; ++page) {
        const ulong page_base = ulong(table[page]) * p.page_tokens;
        const uint p_start = page * p.page_tokens;
        const uint begin = max(first, p_start) - p_start;
        const uint end = min(last, p_start + p.page_tokens) - p_start;
        device const bfloat *k_page =
            key + page_base * width + ulong(kvh) * 256u;
        uint within = begin;
        for (; within + 3u < end; within += 4u) {
          const float s0 =
              exact_dot_hd256(q0, q1, k_page + ulong(within) * width, lane) *
              scale;
          const float s1 =
              exact_dot_hd256(q0, q1, k_page + ulong(within + 1u) * width,
                              lane) *
              scale;
          const float s2 =
              exact_dot_hd256(q0, q1, k_page + ulong(within + 2u) * width,
                              lane) *
              scale;
          const float s3 =
              exact_dot_hd256(q0, q1, k_page + ulong(within + 3u) * width,
                              lane) *
              scale;
          maximum = max(maximum, s0);
          maximum = max(maximum, s1);
          maximum = max(maximum, s2);
          maximum = max(maximum, s3);
        }
        for (; within < end; ++within) {
          const float s0 =
              exact_dot_hd256(q0, q1, k_page + ulong(within) * width, lane) *
              scale;
          maximum = max(maximum, s0);
        }
      }
    } else {
      float q_reg[8];
      const uint values_per_lane = (p.head_dim + 31u) / 32u;
      for (uint idx = 0; idx < values_per_lane; ++idx) {
        const uint d = lane + idx * 32u;
        q_reg[idx] = d < p.head_dim ? float(q[d]) : 0.0f;
      }
      for (uint at = first; at < last; ++at) {
        const ulong physical =
            ulong(table[at / p.page_tokens]) * p.page_tokens +
            at % p.page_tokens;
        const device bfloat *k = key + physical * width + ulong(kvh) * p.head_dim;
        float dot = 0;
        for (uint idx = 0; idx < values_per_lane; ++idx) {
          const uint d = lane + idx * 32u;
          if (d < p.head_dim)
            dot = fma(q_reg[idx], float(k[d]), dot);
        }
        maximum = max(maximum, simd_sum(dot) * scale);
      }
    }
  }
  if (lane == 0)
    partial_max[ulong(tg.y) * p.split_count + split] = maximum;
}

// Unfused reference retains FP32 normalized values, matching the fused
// storage boundary instead of inserting an extra BF16 rounding.
kernel void omlx_m5_qk_norm_fp32_reference(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *key [[buffer(1)]],
    device const bfloat *qn [[buffer(2)]],
    device const bfloat *kn [[buffer(3)]],
    device float *qo [[buffer(4)]],
    device float *ko [[buffer(5)]],
    constant AttentionPostprocessParams &p [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y / (p.query_heads + p.key_value_heads),
             local = tg.y % (p.query_heads + p.key_value_heads);
  if (row >= p.tokens)
    return;
  const bool isq = local < p.query_heads;
  const uint h = isq ? local : local - p.query_heads;
  const ulong stride = isq ? p.query_token_stride : p.key_token_stride,
              hs = isq ? p.query_head_stride : p.key_head_stride;
  const device bfloat *x =
      (isq ? query : key) + ulong(row) * stride + ulong(h) * hs;
  const device bfloat *w = isq ? qn : kn;
  device float *out = (isq ? qo : ko) + ulong(row) * stride + ulong(h) * hs;
  float sum = 0;
  for (uint d = lane; d < p.head_dim; d += 32)
    sum = fma(float(x[d]), float(x[d]), sum);
  const float inv = rsqrt(simd_sum(sum) / float(p.head_dim) + p.epsilon);
  for (uint d = lane; d < p.head_dim; d += 32)
    out[d] = float(x[d]) * inv * (float(w[d]) + p.weight_offset);
}
kernel void omlx_m5_qk_rope_fp32_reference(
    device const float *qi [[buffer(0)]],
    device const float *ki [[buffer(1)]],
    device bfloat *query [[buffer(2)]],
    device bfloat *key [[buffer(3)]],
    device const uint *positions [[buffer(4)]],
    constant AttentionPostprocessParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y / (p.query_heads + p.key_value_heads),
             local = tg.y % (p.query_heads + p.key_value_heads);
  if (row >= p.tokens)
    return;
  const bool isq = local < p.query_heads;
  const uint h = isq ? local : local - p.query_heads;
  const ulong stride = isq ? p.query_token_stride : p.key_token_stride,
              hs = isq ? p.query_head_stride : p.key_head_stride;
  const device float *in =
      (isq ? qi : ki) + ulong(row) * stride + ulong(h) * hs;
  device bfloat *out =
      (isq ? query : key) + ulong(row) * stride + ulong(h) * hs;
  const uint half_rotary = p.rotary_dim / 2;
  const float position = float(positions[row]);
  for (uint pair = lane; pair < half_rotary; pair += 32) {
    const float first = in[pair], second = in[pair + half_rotary];
    const float angle = position / pow(p.theta, float(2 * pair) / float(p.rotary_dim));
    const float cs = cos(angle), sn = sin(angle);
    out[pair] = bfloat(fma(-second, sn, first * cs));
    out[pair + half_rotary] = bfloat(fma(first, sn, second * cs));
  }
  for (uint d = p.rotary_dim + lane; d < p.head_dim; d += 32)
    out[d] = bfloat(in[d]);
}

kernel void omlx_m5_attention_qk_norm_rope_exact(
    device bfloat *query [[buffer(0)]], device bfloat *key [[buffer(1)]],
    device const bfloat *query_norm [[buffer(2)]],
    device const bfloat *key_norm [[buffer(3)]],
    device const uint *positions [[buffer(4)]],
    constant AttentionPostprocessParams &params [[buffer(5)]],
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

  const bool has_qk_norm = (params.epsilon >= 0.0f);
  float inverse_rms = 1.0f;
  if (has_qk_norm) {
    float square_sum = 0.0f;
    for (uint column = lane; column < params.head_dim; column += 32) {
      const float value = float(row[column]);
      square_sum = fma(value, value, square_sum);
    }
    inverse_rms =
        rsqrt(simd_sum(square_sum) / float(params.head_dim) + params.epsilon);
  }
  const uint half_rotary = params.rotary_dim / 2;
  const float position = float(positions[token]);
  for (uint pair = lane; pair < half_rotary; pair += 32) {
    const uint first_index = pair;
    const uint second_index = pair + half_rotary;
    const float first =
        has_qk_norm ? (float(row[first_index]) * inverse_rms *
                       (float(norm[first_index]) + params.weight_offset))
                    : float(row[first_index]);
    const float second =
        has_qk_norm ? (float(row[second_index]) * inverse_rms *
                       (float(norm[second_index]) + params.weight_offset))
                    : float(row[second_index]);
    const float exponent = float(2 * pair) / float(params.rotary_dim);
    const float angle = position / pow(params.theta, exponent);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    row[first_index] = bfloat(fma(-second, sine, first * cosine));
    row[second_index] = bfloat(fma(first, sine, second * cosine));
  }
  if (has_qk_norm) {
    for (uint column = params.rotary_dim + lane; column < params.head_dim;
         column += 32) {
      row[column] = bfloat(float(row[column]) * inverse_rms *
                           (float(norm[column]) + params.weight_offset));
    }
  }
}

// Fused 0-ULP QK-Norm + RoPE + Paged-KV-Append (page_tokens == 64), eliminating
// the separate omlx_m5_paged_kv_append dispatch + full-chip barrier on every full-attn layer!
kernel void omlx_m5_attention_qk_norm_rope_kv_append_exact(
    device bfloat *query [[buffer(0)]],
    device bfloat *key [[buffer(1)]],
    device const bfloat *value [[buffer(2)]],
    device const bfloat *query_norm [[buffer(3)]],
    device const bfloat *key_norm [[buffer(4)]],
    device bfloat *cache_key [[buffer(5)]],
    device bfloat *cache_value [[buffer(6)]],
    device const uint *page_table [[buffer(7)]],
    device const uint *positions [[buffer(8)]],
    constant AttentionPostprocessParams &params [[buffer(9)]],
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

  const bool has_qk_norm = (params.epsilon >= 0.0f);
  float inverse_rms = 1.0f;
  if (has_qk_norm) {
    float square_sum = 0.0f;
    for (uint column = lane; column < params.head_dim; column += 32) {
      const float val = float(row[column]);
      square_sum = fma(val, val, square_sum);
    }
    inverse_rms =
        rsqrt(simd_sum(square_sum) / float(params.head_dim) + params.epsilon);
  }
  const uint half_rotary = params.rotary_dim / 2;
  const uint logical_pos = positions[token];
  const float position = float(logical_pos);
  if (is_query) {
    for (uint pair = lane; pair < half_rotary; pair += 32) {
      const uint first_index = pair;
      const uint second_index = pair + half_rotary;
      const float first =
          has_qk_norm ? (float(row[first_index]) * inverse_rms *
                         (float(norm[first_index]) + params.weight_offset))
                      : float(row[first_index]);
      const float second =
          has_qk_norm ? (float(row[second_index]) * inverse_rms *
                         (float(norm[second_index]) + params.weight_offset))
                      : float(row[second_index]);
      const float exponent = float(2 * pair) / float(params.rotary_dim);
      const float angle = position / pow(params.theta, exponent);
      const float cosine = cos(angle);
      const float sine = sin(angle);
      row[first_index] = bfloat(fma(-second, sine, first * cosine));
      row[second_index] = bfloat(fma(first, sine, second * cosine));
    }
    if (has_qk_norm) {
      for (uint column = params.rotary_dim + lane; column < params.head_dim;
           column += 32) {
        row[column] = bfloat(float(row[column]) * inverse_rms *
                             (float(norm[column]) + params.weight_offset));
      }
    }
  } else {
    const uint page = logical_pos >> 6u;
    const uint within = logical_pos & 63u;
    const ulong paged_pos = (ulong(page_table[page]) << 6ul) + ulong(within);
    const ulong kv_width = ulong(params.key_value_heads) * params.head_dim;
    const ulong dst_base = paged_pos * kv_width + ulong(head) * params.head_dim;
    device bfloat *ck_head = cache_key + dst_base;
    device bfloat *cv_head = cache_value + dst_base;
    device const bfloat *v_head =
        value + ulong(token) * kv_width + ulong(head) * params.head_dim;
    for (uint pair = lane; pair < half_rotary; pair += 32) {
      const uint first_index = pair;
      const uint second_index = pair + half_rotary;
      const float first =
          has_qk_norm ? (float(row[first_index]) * inverse_rms *
                         (float(norm[first_index]) + params.weight_offset))
                      : float(row[first_index]);
      const float second =
          has_qk_norm ? (float(row[second_index]) * inverse_rms *
                         (float(norm[second_index]) + params.weight_offset))
                      : float(row[second_index]);
      const float exponent = float(2 * pair) / float(params.rotary_dim);
      const float angle = position / pow(params.theta, exponent);
      const float cosine = cos(angle);
      const float sine = sin(angle);
      const bfloat k0 = bfloat(fma(-second, sine, first * cosine));
      const bfloat k1 = bfloat(fma(first, sine, second * cosine));
      row[first_index] = k0;
      row[second_index] = k1;
      ck_head[first_index] = k0;
      ck_head[second_index] = k1;
      cv_head[first_index] = v_head[first_index];
      cv_head[second_index] = v_head[second_index];
    }
    for (uint column = params.rotary_dim + lane; column < params.head_dim;
         column += 32) {
      const bfloat kv_normed =
          has_qk_norm ? bfloat(float(row[column]) * inverse_rms *
                               (float(norm[column]) + params.weight_offset))
                      : row[column];
      row[column] = kv_normed;
      ck_head[column] = kv_normed;
      cv_head[column] = v_head[column];
    }
  }
}

kernel void omlx_m5_paged_causal_attention_exact(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const bfloat *cache_key [[buffer(2)]],
    device const bfloat *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *page_table [[buffer(5)]],
    device const uint *positions [[buffer(6)]],
    constant PagedAttentionParams &params [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint repeat = params.query_heads / params.key_value_heads;
  if (tptg.y > 1u && params.head_dim == 256u && repeat > 0u &&
      (tptg.y % repeat) == 0u && tptg.y <= 32u) {
    threadgroup bfloat4 tg_k[kExactKvTileRows * 64u];
    threadgroup bfloat4 tg_v[kExactKvTileRows * 64u];
    threadgroup uint tg_ctx[32];

    const uint bq = tptg.y / repeat;
    const uint kv_head = tg_position.y % params.key_value_heads;
    const uint t_base = (tg_position.y / params.key_value_heads) * bq;
    const uint sg_y = tid_in_tg.y;
    const uint num_sgs = tptg.y;
    const uint flat_tid = sg_y * 32u + lane;
    const uint total_threads = num_sgs * 32u;

    const uint token = t_base + (sg_y / repeat);
    const uint query_head = kv_head * repeat + (sg_y % repeat);
    const bool active = token < params.tokens;
    const uint context_tokens = active ? (positions[token] + 1u) : 0u;

    if (lane == 0u) {
      tg_ctx[sg_y] = context_tokens;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint max_context_tokens = 0u;
    for (uint s = 0; s < num_sgs; ++s) {
      max_context_tokens = max(max_context_tokens, tg_ctx[s]);
    }

    const ulong kv_width = ulong(params.key_value_heads) * 256u;
    const float scale = rsqrt(256.0f);
    float4 q0 = float4(0.0f), q1 = float4(0.0f);
    if (active) {
      device const bfloat *query_row =
          query + ulong(token) * params.query_token_stride +
          ulong(query_head) * params.query_head_stride;
      device const bfloat4 *qv =
          reinterpret_cast<device const bfloat4 *>(query_row + lane * 8u);
      q0 = float4(qv[0]);
      q1 = float4(qv[1]);
    }

    const uint max_pages =
        (max_context_tokens + params.page_tokens - 1u) / params.page_tokens;
    float maximum = -INFINITY;
    for (uint page = 0; page < max_pages; ++page) {
      const ulong page_base = ulong(page_table[page]) * params.page_tokens;
      const uint p_start = page * params.page_tokens;
      const uint valid_max =
          min(params.page_tokens, max_context_tokens - p_start);
      const uint row_valid =
          (active && context_tokens > p_start)
              ? min(params.page_tokens, context_tokens - p_start)
              : 0u;
      device const bfloat *k_page =
          cache_key + page_base * kv_width + ulong(kv_head) * 256u;

      for (uint tile_begin = 0; tile_begin < valid_max;
           tile_begin += kExactKvTileRows) {
        const uint tile_len = min(kExactKvTileRows, valid_max - tile_begin);
        const uint total_vecs = tile_len * 64u;
        for (uint idx = flat_tid; idx < total_vecs; idx += total_threads) {
          const uint w_off = idx >> 6u;
          const uint v_off = idx & 63u;
          device const bfloat4 *src = reinterpret_cast<device const bfloat4 *>(
              k_page + ulong(tile_begin + w_off) * kv_width);
          tg_k[idx] = src[v_off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active && row_valid > tile_begin) {
          const uint local_valid = min(tile_len, row_valid - tile_begin);
          uint w = 0u;
          for (; w + 3u < local_valid; w += 4u) {
            const float s0 =
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 0u) * 64u, lane) * scale;
            const float s1 =
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 1u) * 64u, lane) * scale;
            const float s2 =
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 2u) * 64u, lane) * scale;
            const float s3 =
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 3u) * 64u, lane) * scale;
            maximum = max(maximum, s0);
            maximum = max(maximum, s1);
            maximum = max(maximum, s2);
            maximum = max(maximum, s3);
          }
          for (; w < local_valid; ++w) {
            maximum = max(
                maximum,
                exact_dot_hd256_tg(q0, q1, tg_k + w * 64u, lane) * scale);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    float denominator = 0.0f;
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
    for (uint page = 0; page < max_pages; ++page) {
      const ulong page_base = ulong(page_table[page]) * params.page_tokens;
      const uint p_start = page * params.page_tokens;
      const uint valid_max =
          min(params.page_tokens, max_context_tokens - p_start);
      const uint row_valid =
          (active && context_tokens > p_start)
              ? min(params.page_tokens, context_tokens - p_start)
              : 0u;
      device const bfloat *k_page =
          cache_key + page_base * kv_width + ulong(kv_head) * 256u;
      device const bfloat *v_page =
          cache_value + page_base * kv_width + ulong(kv_head) * 256u;

      for (uint tile_begin = 0; tile_begin < valid_max;
           tile_begin += kExactKvTileRows) {
        const uint tile_len = min(kExactKvTileRows, valid_max - tile_begin);
        const uint total_vecs = tile_len * 64u;
        for (uint idx = flat_tid; idx < total_vecs; idx += total_threads) {
          const uint w_off = idx >> 6u;
          const uint v_off = idx & 63u;
          const ulong row_off = ulong(tile_begin + w_off) * kv_width;
          tg_k[idx] =
              reinterpret_cast<device const bfloat4 *>(k_page + row_off)[v_off];
          tg_v[idx] =
              reinterpret_cast<device const bfloat4 *>(v_page + row_off)[v_off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active && row_valid > tile_begin) {
          const uint local_valid = min(tile_len, row_valid - tile_begin);
          uint w = 0u;
          for (; w + 3u < local_valid; w += 4u) {
            const float p0 = exp(
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 0u) * 64u, lane) *
                    scale -
                maximum);
            const float p1 = exp(
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 1u) * 64u, lane) *
                    scale -
                maximum);
            const float p2 = exp(
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 2u) * 64u, lane) *
                    scale -
                maximum);
            const float p3 = exp(
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 3u) * 64u, lane) *
                    scale -
                maximum);
            denominator += p0;
            exact_v_accum_hd256_tg(p0, tg_v + (w + 0u) * 64u, lane, acc0, acc1);
            denominator += p1;
            exact_v_accum_hd256_tg(p1, tg_v + (w + 1u) * 64u, lane, acc0, acc1);
            denominator += p2;
            exact_v_accum_hd256_tg(p2, tg_v + (w + 2u) * 64u, lane, acc0, acc1);
            denominator += p3;
            exact_v_accum_hd256_tg(p3, tg_v + (w + 3u) * 64u, lane, acc0, acc1);
          }
          for (; w < local_valid; ++w) {
            const float p0 = exp(
                exact_dot_hd256_tg(q0, q1, tg_k + w * 64u, lane) * scale -
                maximum);
            denominator += p0;
            exact_v_accum_hd256_tg(p0, tg_v + w * 64u, lane, acc0, acc1);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    if (active) {
      const float4 norm0 = acc0 / denominator;
      const float4 norm1 = acc1 / denominator;
      float4 o0 = norm0;
      float4 o1 = norm1;
      if (params.gate_head_stride != 0u) {
        device const bfloat *gate_row =
            gate + ulong(token) * params.gate_token_stride +
            ulong(query_head) * params.gate_head_stride;
        device const bfloat4 *gv =
            reinterpret_cast<device const bfloat4 *>(gate_row + lane * 8u);
        o0 /= (1.0f + exp(-float4(gv[0])));
        o1 /= (1.0f + exp(-float4(gv[1])));
      }
      device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(
          output + (ulong(token) * params.query_heads + query_head) * 256u +
          lane * 8u);
      ov[0] = bfloat4(o0);
      ov[1] = bfloat4(o1);
    }
    return;
  }

  const uint logical_head = tg_position.y;
  const uint token = logical_head / params.query_heads;
  if (token >= params.tokens)
    return;
  const uint query_head = logical_head - token * params.query_heads;
  const uint kv_head = query_head / repeat;
  device const bfloat *query_row = query +
                                   ulong(token) * params.query_token_stride +
                                   ulong(query_head) * params.query_head_stride;
  device const bfloat *gate_row = gate +
                                  ulong(token) * params.gate_token_stride +
                                  ulong(query_head) * params.gate_head_stride;
  const ulong kv_width = ulong(params.key_value_heads) * params.head_dim;
  const uint context_tokens = positions[token] + 1;
  const float scale = rsqrt(float(params.head_dim));

  const uint logical_pages =
      (context_tokens + params.page_tokens - 1) / params.page_tokens;
  if (params.head_dim == 256) {
    device const bfloat4 *qv =
        reinterpret_cast<device const bfloat4 *>(query_row + lane * 8u);
    const float4 q0 = float4(qv[0]);
    const float4 q1 = float4(qv[1]);
    float maximum = -INFINITY;
    for (uint page = 0; page < logical_pages; ++page) {
      const ulong page_base = ulong(page_table[page]) * params.page_tokens;
      const uint valid =
          min(params.page_tokens, context_tokens - page * params.page_tokens);
      device const bfloat *k_page =
          cache_key + page_base * kv_width + ulong(kv_head) * 256u;
      uint within = 0;
      for (; within + 3u < valid; within += 4u) {
        const float s0 =
            exact_dot_hd256(q0, q1, k_page + ulong(within) * kv_width, lane) *
            scale;
        const float s1 =
            exact_dot_hd256(q0, q1, k_page + ulong(within + 1u) * kv_width,
                            lane) *
            scale;
        const float s2 =
            exact_dot_hd256(q0, q1, k_page + ulong(within + 2u) * kv_width,
                            lane) *
            scale;
        const float s3 =
            exact_dot_hd256(q0, q1, k_page + ulong(within + 3u) * kv_width,
                            lane) *
            scale;
        maximum = max(maximum, s0);
        maximum = max(maximum, s1);
        maximum = max(maximum, s2);
        maximum = max(maximum, s3);
      }
      for (; within < valid; ++within) {
        maximum = max(
            maximum,
            exact_dot_hd256(q0, q1, k_page + ulong(within) * kv_width, lane) *
                scale);
      }
    }
    float denominator = 0.0f;
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
    for (uint page = 0; page < logical_pages; ++page) {
      const ulong page_base = ulong(page_table[page]) * params.page_tokens;
      const uint valid =
          min(params.page_tokens, context_tokens - page * params.page_tokens);
      device const bfloat *k_page =
          cache_key + page_base * kv_width + ulong(kv_head) * 256u;
      device const bfloat *v_page =
          cache_value + page_base * kv_width + ulong(kv_head) * 256u;
      uint within = 0;
      for (; within + 3u < valid; within += 4u) {
        const float p0 = exp(
            exact_dot_hd256(q0, q1, k_page + ulong(within) * kv_width, lane) *
                scale -
            maximum);
        const float p1 = exp(
            exact_dot_hd256(q0, q1, k_page + ulong(within + 1u) * kv_width,
                            lane) *
                scale -
            maximum);
        const float p2 = exp(
            exact_dot_hd256(q0, q1, k_page + ulong(within + 2u) * kv_width,
                            lane) *
                scale -
            maximum);
        const float p3 = exp(
            exact_dot_hd256(q0, q1, k_page + ulong(within + 3u) * kv_width,
                            lane) *
                scale -
            maximum);
        denominator += p0;
        exact_v_accum_hd256(p0, v_page + ulong(within) * kv_width, lane, acc0,
                            acc1);
        denominator += p1;
        exact_v_accum_hd256(p1, v_page + ulong(within + 1u) * kv_width, lane,
                            acc0, acc1);
        denominator += p2;
        exact_v_accum_hd256(p2, v_page + ulong(within + 2u) * kv_width, lane,
                            acc0, acc1);
        denominator += p3;
        exact_v_accum_hd256(p3, v_page + ulong(within + 3u) * kv_width, lane,
                            acc0, acc1);
      }
      for (; within < valid; ++within) {
        const float p0 = exp(
            exact_dot_hd256(q0, q1, k_page + ulong(within) * kv_width, lane) *
                scale -
            maximum);
        denominator += p0;
        exact_v_accum_hd256(p0, v_page + ulong(within) * kv_width, lane, acc0,
                            acc1);
      }
    }
    float4 o0 = acc0 / denominator;
    float4 o1 = acc1 / denominator;
    if (params.gate_head_stride != 0u) {
      device const bfloat4 *gv =
          reinterpret_cast<device const bfloat4 *>(gate_row + lane * 8u);
      o0 /= (1.0f + exp(-float4(gv[0])));
      o1 /= (1.0f + exp(-float4(gv[1])));
    }
    device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(
        output + (ulong(token) * params.query_heads + query_head) * 256u +
        lane * 8u);
    ov[0] = bfloat4(o0);
    ov[1] = bfloat4(o1);
    return;
  }

  float maximum = -INFINITY;
  for (uint page = 0; page < logical_pages; ++page) {
    const ulong page_base = ulong(page_table[page]) * params.page_tokens;
    const uint valid =
        min(params.page_tokens, context_tokens - page * params.page_tokens);
    for (uint within = 0; within < valid; ++within) {
      const ulong position = page_base + within;
      device const bfloat *key_row =
          cache_key + position * kv_width + ulong(kv_head) * params.head_dim;
      float partial = 0.0f;
      for (uint feature = lane; feature < params.head_dim; feature += 32) {
        partial =
            fma(float(query_row[feature]), float(key_row[feature]), partial);
      }
      maximum = max(maximum, simd_sum(partial) * scale);
    }
  }

  float denominator = 0.0f;
  float accumulators[8];
  const uint values_per_lane = (params.head_dim + 31) / 32;
  for (uint index = 0; index < values_per_lane; ++index) {
    accumulators[index] = 0.0f;
  }
  for (uint page = 0; page < logical_pages; ++page) {
    const ulong page_base = ulong(page_table[page]) * params.page_tokens;
    const uint valid =
        min(params.page_tokens, context_tokens - page * params.page_tokens);
    for (uint within = 0; within < valid; ++within) {
      const ulong position = page_base + within;
      device const bfloat *key_row =
          cache_key + position * kv_width + ulong(kv_head) * params.head_dim;
      device const bfloat *value_row =
          cache_value + position * kv_width + ulong(kv_head) * params.head_dim;
      float partial = 0.0f;
      for (uint feature = lane; feature < params.head_dim; feature += 32) {
        partial =
            fma(float(query_row[feature]), float(key_row[feature]), partial);
      }
      const float probability = exp(simd_sum(partial) * scale - maximum);
      denominator += probability;
      for (uint index = 0; index < values_per_lane; ++index) {
        const uint feature = lane + index * 32;
        if (feature < params.head_dim) {
          accumulators[index] =
              fma(probability, float(value_row[feature]), accumulators[index]);
        }
      }
    }
  }
  for (uint index = 0; index < values_per_lane; ++index) {
    const uint feature = lane + index * 32;
    if (feature < params.head_dim) {
      const float norm_o = accumulators[index] / denominator;
      const float out_val =
          (params.gate_head_stride != 0u)
              ? (norm_o / (1.0f + exp(-float(gate_row[feature]))))
              : norm_o;
      output[(ulong(token) * params.query_heads + query_head) *
                 params.head_dim +
             feature] = bfloat(out_val);
    }
  }
}

kernel void omlx_m5_paged_attention_exact_merge(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const bfloat *cache_key [[buffer(2)]],
    device const bfloat *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *page_table [[buffer(5)]],
    device const float *partial_max [[buffer(6)]],
    device const uint *positions [[buffer(7)]],
    constant PagedSplitAttentionParams &params [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint repeat = params.query_heads / params.key_value_heads;
  if (tptg.y > 1u && params.head_dim == 256u && repeat > 0u &&
      (tptg.y % repeat) == 0u && tptg.y <= 32u) {
    threadgroup bfloat4 tg_k[kExactKvTileRows * 64u];
    threadgroup bfloat4 tg_v[kExactKvTileRows * 64u];
    threadgroup uint tg_ctx[32];

    const uint bq = tptg.y / repeat;
    const uint kv_head = tg_position.y % params.key_value_heads;
    const uint t_base = (tg_position.y / params.key_value_heads) * bq;
    const uint sg_y = tid_in_tg.y;
    const uint num_sgs = tptg.y;
    const uint flat_tid = sg_y * 32u + lane;
    const uint total_threads = num_sgs * 32u;

    const uint token = t_base + (sg_y / repeat);
    const uint query_head = kv_head * repeat + (sg_y % repeat);
    const bool active = token < params.tokens;
    const uint context_tokens = active ? (positions[token] + 1u) : 0u;

    if (lane == 0u) {
      tg_ctx[sg_y] = context_tokens;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint max_context_tokens = 0u;
    for (uint s = 0; s < num_sgs; ++s) {
      max_context_tokens = max(max_context_tokens, tg_ctx[s]);
    }

    const ulong kv_width = ulong(params.key_value_heads) * 256u;
    const float scale = rsqrt(256.0f);
    float maximum = -INFINITY;
    float4 q0 = float4(0.0f), q1 = float4(0.0f);
    if (active) {
      const ulong canonical_head = ulong(token) * params.query_heads + query_head;
      for (uint split = 0; split < params.split_count; ++split) {
        maximum =
            max(maximum, partial_max[canonical_head * params.split_count + split]);
      }
      device const bfloat *query_row =
          query + ulong(token) * params.query_token_stride +
          ulong(query_head) * params.query_head_stride;
      device const bfloat4 *qv =
          reinterpret_cast<device const bfloat4 *>(query_row + lane * 8u);
      q0 = float4(qv[0]);
      q1 = float4(qv[1]);
    }

    float denominator = 0.0f;
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
    const uint max_pages =
        (max_context_tokens + params.page_tokens - 1u) / params.page_tokens;
    for (uint page = 0; page < max_pages; ++page) {
      const ulong page_base = ulong(page_table[page]) * params.page_tokens;
      const uint p_start = page * params.page_tokens;
      const uint valid_max =
          min(params.page_tokens, max_context_tokens - p_start);
      const uint row_valid =
          (active && context_tokens > p_start)
              ? min(params.page_tokens, context_tokens - p_start)
              : 0u;
      device const bfloat *k_page =
          cache_key + page_base * kv_width + ulong(kv_head) * 256u;
      device const bfloat *v_page =
          cache_value + page_base * kv_width + ulong(kv_head) * 256u;

      for (uint tile_begin = 0; tile_begin < valid_max;
           tile_begin += kExactKvTileRows) {
        const uint tile_len = min(kExactKvTileRows, valid_max - tile_begin);
        const uint total_vecs = tile_len * 64u;
        for (uint idx = flat_tid; idx < total_vecs; idx += total_threads) {
          const uint w_off = idx >> 6u;
          const uint v_off = idx & 63u;
          const ulong row_off = ulong(tile_begin + w_off) * kv_width;
          tg_k[idx] =
              reinterpret_cast<device const bfloat4 *>(k_page + row_off)[v_off];
          tg_v[idx] =
              reinterpret_cast<device const bfloat4 *>(v_page + row_off)[v_off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active && row_valid > tile_begin) {
          const uint local_valid = min(tile_len, row_valid - tile_begin);
          uint w = 0u;
          for (; w + 3u < local_valid; w += 4u) {
            const float p0 = exp(
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 0u) * 64u, lane) *
                    scale -
                maximum);
            const float p1 = exp(
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 1u) * 64u, lane) *
                    scale -
                maximum);
            const float p2 = exp(
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 2u) * 64u, lane) *
                    scale -
                maximum);
            const float p3 = exp(
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 3u) * 64u, lane) *
                    scale -
                maximum);
            denominator += p0;
            exact_v_accum_hd256_tg(p0, tg_v + (w + 0u) * 64u, lane, acc0, acc1);
            denominator += p1;
            exact_v_accum_hd256_tg(p1, tg_v + (w + 1u) * 64u, lane, acc0, acc1);
            denominator += p2;
            exact_v_accum_hd256_tg(p2, tg_v + (w + 2u) * 64u, lane, acc0, acc1);
            denominator += p3;
            exact_v_accum_hd256_tg(p3, tg_v + (w + 3u) * 64u, lane, acc0, acc1);
          }
          for (; w < local_valid; ++w) {
            const float p0 = exp(
                exact_dot_hd256_tg(q0, q1, tg_k + w * 64u, lane) * scale -
                maximum);
            denominator += p0;
            exact_v_accum_hd256_tg(p0, tg_v + w * 64u, lane, acc0, acc1);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    if (active) {
      const float4 norm0 = acc0 / denominator;
      const float4 norm1 = acc1 / denominator;
      float4 o0 = norm0;
      float4 o1 = norm1;
      if (params.gate_head_stride != 0u) {
        device const bfloat *gate_row =
            gate + ulong(token) * params.gate_token_stride +
            ulong(query_head) * params.gate_head_stride;
        device const bfloat4 *gv =
            reinterpret_cast<device const bfloat4 *>(gate_row + lane * 8u);
        o0 /= (1.0f + exp(-float4(gv[0])));
        o1 /= (1.0f + exp(-float4(gv[1])));
      }
      device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(
          output + (ulong(token) * params.query_heads + query_head) * 256u +
          lane * 8u);
      ov[0] = bfloat4(o0);
      ov[1] = bfloat4(o1);
    }
    return;
  }

  const uint logical_head = tg_position.y;
  const uint token = logical_head / params.query_heads;
  if (token >= params.tokens)
    return;
  const uint query_head = logical_head - token * params.query_heads;
  const uint kv_head = query_head / repeat;
  device const bfloat *query_row = query +
                                   ulong(token) * params.query_token_stride +
                                   ulong(query_head) * params.query_head_stride;
  device const bfloat *gate_row = gate +
                                  ulong(token) * params.gate_token_stride +
                                  ulong(query_head) * params.gate_head_stride;
  const ulong kv_width = ulong(params.key_value_heads) * params.head_dim;
  const uint context_tokens = positions[token] + 1;
  const float scale = rsqrt(float(params.head_dim));

  float maximum = -INFINITY;
  for (uint split = 0; split < params.split_count; ++split)
    maximum =
        max(maximum, partial_max[ulong(logical_head) * params.split_count + split]);
  const uint logical_pages =
      (context_tokens + params.page_tokens - 1) / params.page_tokens;
  if (params.head_dim == 256) {
    device const bfloat4 *qv =
        reinterpret_cast<device const bfloat4 *>(query_row + lane * 8u);
    const float4 q0 = float4(qv[0]);
    const float4 q1 = float4(qv[1]);
    float denominator = 0.0f;
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
    for (uint page = 0; page < logical_pages; ++page) {
      const ulong page_base = ulong(page_table[page]) * params.page_tokens;
      const uint valid =
          min(params.page_tokens, context_tokens - page * params.page_tokens);
      device const bfloat *k_page =
          cache_key + page_base * kv_width + ulong(kv_head) * 256u;
      device const bfloat *v_page =
          cache_value + page_base * kv_width + ulong(kv_head) * 256u;
      uint within = 0;
      for (; within + 3u < valid; within += 4u) {
        const float p0 = exp(
            exact_dot_hd256(q0, q1, k_page + ulong(within) * kv_width, lane) *
                scale -
            maximum);
        const float p1 = exp(
            exact_dot_hd256(q0, q1, k_page + ulong(within + 1u) * kv_width,
                            lane) *
                scale -
            maximum);
        const float p2 = exp(
            exact_dot_hd256(q0, q1, k_page + ulong(within + 2u) * kv_width,
                            lane) *
                scale -
            maximum);
        const float p3 = exp(
            exact_dot_hd256(q0, q1, k_page + ulong(within + 3u) * kv_width,
                            lane) *
                scale -
            maximum);
        denominator += p0;
        exact_v_accum_hd256(p0, v_page + ulong(within) * kv_width, lane, acc0,
                            acc1);
        denominator += p1;
        exact_v_accum_hd256(p1, v_page + ulong(within + 1u) * kv_width, lane,
                            acc0, acc1);
        denominator += p2;
        exact_v_accum_hd256(p2, v_page + ulong(within + 2u) * kv_width, lane,
                            acc0, acc1);
        denominator += p3;
        exact_v_accum_hd256(p3, v_page + ulong(within + 3u) * kv_width, lane,
                            acc0, acc1);
      }
      for (; within < valid; ++within) {
        const float p0 = exp(
            exact_dot_hd256(q0, q1, k_page + ulong(within) * kv_width, lane) *
                scale -
            maximum);
        denominator += p0;
        exact_v_accum_hd256(p0, v_page + ulong(within) * kv_width, lane, acc0,
                            acc1);
      }
    }
    float4 o0 = acc0 / denominator;
    float4 o1 = acc1 / denominator;
    if (params.gate_head_stride != 0u) {
      device const bfloat4 *gv =
          reinterpret_cast<device const bfloat4 *>(gate_row + lane * 8u);
      const float4 g0 = float4(gv[0]);
      const float4 g1 = float4(gv[1]);
      o0 /= (1.0f + exp(-g0));
      o1 /= (1.0f + exp(-g1));
    }
    device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(
        output + (ulong(token) * params.query_heads + query_head) * 256u +
        lane * 8u);
    ov[0] = bfloat4(o0);
    ov[1] = bfloat4(o1);
    return;
  }
  float denominator = 0.0f;
  float accumulators[8];
  const uint values_per_lane = (params.head_dim + 31) / 32;
  for (uint index = 0; index < values_per_lane; ++index) {
    accumulators[index] = 0.0f;
  }
  for (uint page = 0; page < logical_pages; ++page) {
    const ulong page_base = ulong(page_table[page]) * params.page_tokens;
    const uint valid =
        min(params.page_tokens, context_tokens - page * params.page_tokens);
    for (uint within = 0; within < valid; ++within) {
      const ulong position = page_base + within;
      device const bfloat *key_row =
          cache_key + position * kv_width + ulong(kv_head) * params.head_dim;
      device const bfloat *value_row =
          cache_value + position * kv_width + ulong(kv_head) * params.head_dim;
      float partial = 0.0f;
      for (uint feature = lane; feature < params.head_dim; feature += 32) {
        partial =
            fma(float(query_row[feature]), float(key_row[feature]), partial);
      }
      const float probability = exp(simd_sum(partial) * scale - maximum);
      denominator += probability;
      for (uint index = 0; index < values_per_lane; ++index) {
        const uint feature = lane + index * 32;
        if (feature < params.head_dim) {
          accumulators[index] =
              fma(probability, float(value_row[feature]), accumulators[index]);
        }
      }
    }
  }
  for (uint index = 0; index < values_per_lane; ++index) {
    const uint feature = lane + index * 32;
    if (feature < params.head_dim) {
      float out_val = accumulators[index] / denominator;
      if (params.gate_head_stride != 0u) {
        const float gate_value = float(gate_row[feature]);
        out_val /= (1.0f + exp(-gate_value));
      }
      output[(ulong(token) * params.query_heads + query_head) *
                 params.head_dim +
             feature] = bfloat(out_val);
    }
  }
}

// ============================================================================
// Step 1 (M1): 1-Pass Online Softmax BF16 Paged Decode/Verify Split-KV & Single
// Guarantees 0 ULP bit-identity across AR (T=1), DFlash2 verify (T=2..8),
// Batch (run_packed_batch), GQA TG tile-sharing (tptg.y > 1), and Split-KV vs
// Single-Pass via canonical 512-token (8-page) super-chunk boundaries and
// deterministic sequential partition reduction (p = 0 .. num_parts - 1).
// TG SRAM: 8 * 64 * 8B (tg_k: 4096B) + 8 * 64 * 8B (tg_v: 4096B) + 32 * 4B
// (tg_row_last: 128B) = 8,320 B (8.125 KiB <= 32 KiB, 3 TGs/core on Apple10).
// ============================================================================

constexpr constant uint kExact1PassKvTileRows = 8u;
constexpr constant uint kMaxExact1PassPartitions = 256u;

inline uint exact_1pass_chunk_tokens(uint max_context) {
  const uint total_max_sbs = (max_context + 511u) / 512u;
  return (total_max_sbs > kMaxExact1PassPartitions)
             ? (((max_context + kMaxExact1PassPartitions * 512u - 1u) /
                 (kMaxExact1PassPartitions * 512u)) *
                512u)
             : 512u;
}

__attribute__((always_inline)) inline void exact_online_update4_hd256_tg(
    float s0, float s1, float s2, float s3,
    threadgroup const bfloat4 *vv_base, uint lane,
    thread float &m_blk, thread float &l_blk,
    thread float4 &acc0, thread float4 &acc1) {
  const float qmax = max(max(s0, s1), max(s2, s3));
  if (qmax <= -INFINITY) {
    return;
  }
  if (qmax > m_blk) {
    const float rescale =
        (m_blk == -INFINITY || qmax == INFINITY) ? 0.0f : exp(m_blk - qmax);
    l_blk *= rescale;
    acc0 *= rescale;
    acc1 *= rescale;
    m_blk = qmax;
  }
  const float p0 = (s0 > -INFINITY) ? exp(s0 - m_blk) : 0.0f;
  const float p1 = (s1 > -INFINITY) ? exp(s1 - m_blk) : 0.0f;
  const float p2 = (s2 > -INFINITY) ? exp(s2 - m_blk) : 0.0f;
  const float p3 = (s3 > -INFINITY) ? exp(s3 - m_blk) : 0.0f;
  l_blk += p0;
  exact_v_accum_hd256_tg(p0, vv_base + 0u * 64u, lane, acc0, acc1);
  l_blk += p1;
  exact_v_accum_hd256_tg(p1, vv_base + 1u * 64u, lane, acc0, acc1);
  l_blk += p2;
  exact_v_accum_hd256_tg(p2, vv_base + 2u * 64u, lane, acc0, acc1);
  l_blk += p3;
  exact_v_accum_hd256_tg(p3, vv_base + 3u * 64u, lane, acc0, acc1);
}

__attribute__((always_inline)) inline void exact_online_update1_hd256_tg(
    float s0, threadgroup const bfloat4 *vv_base, uint lane,
    thread float &m_blk, thread float &l_blk,
    thread float4 &acc0, thread float4 &acc1) {
  if (s0 <= -INFINITY) {
    return;
  }
  if (s0 > m_blk) {
    const float rescale =
        (m_blk == -INFINITY || s0 == INFINITY) ? 0.0f : exp(m_blk - s0);
    l_blk *= rescale;
    acc0 *= rescale;
    acc1 *= rescale;
    m_blk = s0;
  }
  const float p0 = exp(s0 - m_blk);
  l_blk += p0;
  exact_v_accum_hd256_tg(p0, vv_base, lane, acc0, acc1);
}

__attribute__((always_inline)) inline void exact_online_update4_hd256_dev(
    float s0, float s1, float s2, float s3,
    device const bfloat *v_base, ulong kv_width, uint lane,
    thread float &m_blk, thread float &l_blk,
    thread float4 &acc0, thread float4 &acc1) {
  const float qmax = max(max(s0, s1), max(s2, s3));
  if (qmax <= -INFINITY) {
    return;
  }
  if (qmax > m_blk) {
    const float rescale =
        (m_blk == -INFINITY || qmax == INFINITY) ? 0.0f : exp(m_blk - qmax);
    l_blk *= rescale;
    acc0 *= rescale;
    acc1 *= rescale;
    m_blk = qmax;
  }
  const float p0 = (s0 > -INFINITY) ? exp(s0 - m_blk) : 0.0f;
  const float p1 = (s1 > -INFINITY) ? exp(s1 - m_blk) : 0.0f;
  const float p2 = (s2 > -INFINITY) ? exp(s2 - m_blk) : 0.0f;
  const float p3 = (s3 > -INFINITY) ? exp(s3 - m_blk) : 0.0f;
  l_blk += p0;
  exact_v_accum_hd256(p0, v_base + 0ul * kv_width, lane, acc0, acc1);
  l_blk += p1;
  exact_v_accum_hd256(p1, v_base + 1ul * kv_width, lane, acc0, acc1);
  l_blk += p2;
  exact_v_accum_hd256(p2, v_base + 2ul * kv_width, lane, acc0, acc1);
  l_blk += p3;
  exact_v_accum_hd256(p3, v_base + 3ul * kv_width, lane, acc0, acc1);
}

__attribute__((always_inline)) inline void exact_online_update1_hd256_dev(
    float s0, device const bfloat *v_row, uint lane,
    thread float &m_blk, thread float &l_blk,
    thread float4 &acc0, thread float4 &acc1) {
  if (s0 <= -INFINITY) {
    return;
  }
  if (s0 > m_blk) {
    const float rescale =
        (m_blk == -INFINITY || s0 == INFINITY) ? 0.0f : exp(m_blk - s0);
    l_blk *= rescale;
    acc0 *= rescale;
    acc1 *= rescale;
    m_blk = s0;
  }
  const float p0 = exp(s0 - m_blk);
  l_blk += p0;
  exact_v_accum_hd256(p0, v_row, lane, acc0, acc1);
}

kernel void omlx_m5_paged_attention_exact_1pass_partial(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *cache_key [[buffer(1)]],
    device const bfloat *cache_value [[buffer(2)]],
    device float *partial_max [[buffer(3)]],
    device float *partial_sum [[buffer(4)]],
    device float *partial_output [[buffer(5)]],
    device const uint *page_table [[buffer(6)]],
    device const uint *positions [[buffer(7)]],
    constant PagedSplitAttentionParams &p [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint repeat = p.query_heads / p.key_value_heads;
  if (tptg.y > 1u && p.head_dim == 256u && repeat > 0u &&
      (tptg.y % repeat) == 0u && tptg.y <= 32u) {
    threadgroup bfloat4 tg_k[kExact1PassKvTileRows * 64u];
    threadgroup bfloat4 tg_v[kExact1PassKvTileRows * 64u];
    threadgroup uint tg_row_last[32];

    const uint split = tg.x;
    const uint bq = tptg.y / repeat;
    const uint kvh = tg.y % p.key_value_heads;
    const uint t_base = (tg.y / p.key_value_heads) * bq;
    const uint sg_y = tid_in_tg.y;
    const uint num_sgs = tptg.y;
    const uint flat_tid = sg_y * 32u + lane;
    const uint total_threads = num_sgs * 32u;

    const uint row = t_base + (sg_y / repeat);
    const uint qh = kvh * repeat + (sg_y % repeat);
    const bool active = (split < p.split_count) && (row < p.tokens);

    const uint first = split * p.split_tokens;
    const uint last =
        active ? min(first + p.split_tokens, positions[row] + 1u) : first;
    if (lane == 0u) {
      tg_row_last[sg_y] = (first < last) ? last : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint max_last = 0u;
    for (uint s = 0; s < num_sgs; ++s) {
      max_last = max(max_last, tg_row_last[s]);
    }

    if (first >= max_last) {
      if (active && lane == 0u) {
        const ulong record =
            (ulong(row) * p.query_heads + qh) * p.split_count + split;
        partial_max[record] = -INFINITY;
        partial_sum[record] = 0.0f;
      }
      return;
    }

    const ulong width = ulong(p.key_value_heads) * 256ul;
    const float scale = rsqrt(256.0f);
    float m_blk = -INFINITY;
    float l_blk = 0.0f;
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
    float4 q0 = float4(0.0f), q1 = float4(0.0f);
    if (active && first < last) {
      device const bfloat *q =
          query + ulong(row) * p.query_token_stride +
          ulong(qh) * p.query_head_stride;
      device const bfloat4 *qv =
          reinterpret_cast<device const bfloat4 *>(q + lane * 8u);
      q0 = float4(qv[0]);
      q1 = float4(qv[1]);
    }

    const uint first_page = first / p.page_tokens;
    const uint last_page = (max_last - 1u) / p.page_tokens;
    for (uint page = first_page; page <= last_page; ++page) {
      const ulong page_base = ulong(page_table[page]) * p.page_tokens;
      const uint p_start = page * p.page_tokens;
      const uint page_begin = max(first, p_start) - p_start;
      const uint page_end_max =
          min(max_last, p_start + p.page_tokens) - p_start;
      const uint row_end =
          (active && last > first && last > p_start)
              ? (min(last, p_start + p.page_tokens) - p_start)
              : 0u;
      device const bfloat *k_page =
          cache_key + page_base * width + ulong(kvh) * 256ul;
      device const bfloat *v_page =
          cache_value + page_base * width + ulong(kvh) * 256ul;

      for (uint tile_begin = page_begin; tile_begin < page_end_max;
           tile_begin += kExact1PassKvTileRows) {
        const uint tile_len =
            min(kExact1PassKvTileRows, page_end_max - tile_begin);
        const uint total_vecs = tile_len * 64u;
        for (uint idx = flat_tid; idx < total_vecs; idx += total_threads) {
          const uint w_off = idx >> 6u;
          const uint v_off = idx & 63u;
          const ulong row_off = ulong(tile_begin + w_off) * width;
          tg_k[idx] =
              reinterpret_cast<device const bfloat4 *>(k_page + row_off)[v_off];
          tg_v[idx] =
              reinterpret_cast<device const bfloat4 *>(v_page + row_off)[v_off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active && last > first && row_end > tile_begin) {
          const uint local_valid = min(tile_len, row_end - tile_begin);
          uint w = 0u;
          for (; w + 3u < local_valid; w += 4u) {
            const float s0 =
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 0u) * 64u, lane) * scale;
            const float s1 =
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 1u) * 64u, lane) * scale;
            const float s2 =
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 2u) * 64u, lane) * scale;
            const float s3 =
                exact_dot_hd256_tg(q0, q1, tg_k + (w + 3u) * 64u, lane) * scale;
            exact_online_update4_hd256_tg(s0, s1, s2, s3, tg_v + w * 64u, lane,
                                          m_blk, l_blk, acc0, acc1);
          }
          for (; w < local_valid; ++w) {
            const float s0 =
                exact_dot_hd256_tg(q0, q1, tg_k + w * 64u, lane) * scale;
            exact_online_update1_hd256_tg(s0, tg_v + w * 64u, lane, m_blk,
                                          l_blk, acc0, acc1);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    if (active) {
      const ulong record =
          (ulong(row) * p.query_heads + qh) * p.split_count + split;
      if (lane == 0u) {
        partial_max[record] = m_blk;
        partial_sum[record] = l_blk;
      }
      if (m_blk > -INFINITY) {
        device float4 *po4 = reinterpret_cast<device float4 *>(
            partial_output + record * 256ul + lane * 8u);
        po4[0] = acc0;
        po4[1] = acc1;
      }
    }
    return;
  }

  const uint split = tg.x, row = tg.y / p.query_heads, qh = tg.y % p.query_heads;
  if (split >= p.split_count || row >= p.tokens) {
    return;
  }
  const uint kvh = qh / repeat;
  const ulong width = ulong(p.key_value_heads) * p.head_dim;
  const uint first = split * p.split_tokens;
  const uint last = min(first + p.split_tokens, positions[row] + 1u);
  const ulong record = (ulong(row) * p.query_heads + qh) * p.split_count + split;
  if (first >= last) {
    if (lane == 0u) {
      partial_max[record] = -INFINITY;
      partial_sum[record] = 0.0f;
    }
    return;
  }

  device const bfloat *q =
      query + ulong(row) * p.query_token_stride + ulong(qh) * p.query_head_stride;
  const float scale = rsqrt(float(p.head_dim));
  const uint first_page = first / p.page_tokens;
  const uint last_page = (last - 1u) / p.page_tokens;

  if (p.head_dim == 256u) {
    device const bfloat4 *qv =
        reinterpret_cast<device const bfloat4 *>(q + lane * 8u);
    const float4 q0 = float4(qv[0]);
    const float4 q1 = float4(qv[1]);
    float m_blk = -INFINITY;
    float l_blk = 0.0f;
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f);

    for (uint page = first_page; page <= last_page; ++page) {
      const ulong page_base = ulong(page_table[page]) * p.page_tokens;
      const uint p_start = page * p.page_tokens;
      const uint begin = max(first, p_start) - p_start;
      const uint end = min(last, p_start + p.page_tokens) - p_start;
      device const bfloat *k_page =
          cache_key + page_base * width + ulong(kvh) * 256ul;
      device const bfloat *v_page =
          cache_value + page_base * width + ulong(kvh) * 256ul;
      uint within = begin;
      for (; within + 3u < end; within += 4u) {
        const float s0 =
            exact_dot_hd256(q0, q1, k_page + ulong(within + 0u) * width, lane) *
            scale;
        const float s1 =
            exact_dot_hd256(q0, q1, k_page + ulong(within + 1u) * width, lane) *
            scale;
        const float s2 =
            exact_dot_hd256(q0, q1, k_page + ulong(within + 2u) * width, lane) *
            scale;
        const float s3 =
            exact_dot_hd256(q0, q1, k_page + ulong(within + 3u) * width, lane) *
            scale;
        exact_online_update4_hd256_dev(s0, s1, s2, s3,
                                       v_page + ulong(within) * width, width,
                                       lane, m_blk, l_blk, acc0, acc1);
      }
      for (; within < end; ++within) {
        const float s0 =
            exact_dot_hd256(q0, q1, k_page + ulong(within) * width, lane) *
            scale;
        exact_online_update1_hd256_dev(s0, v_page + ulong(within) * width, lane,
                                       m_blk, l_blk, acc0, acc1);
      }
    }

    if (lane == 0u) {
      partial_max[record] = m_blk;
      partial_sum[record] = l_blk;
    }
    if (m_blk > -INFINITY) {
      device float4 *po4 = reinterpret_cast<device float4 *>(
          partial_output + record * 256ul + lane * 8u);
      po4[0] = acc0;
      po4[1] = acc1;
    }
    return;
  }

  float q_reg[8];
  float acc_blk[8];
  const uint values_per_lane = (p.head_dim + 31u) / 32u;
  for (uint idx = 0; idx < 8u; ++idx) {
    const uint d = lane + idx * 32u;
    q_reg[idx] = d < p.head_dim ? float(q[d]) : 0.0f;
    acc_blk[idx] = 0.0f;
  }
  float m_blk = -INFINITY;
  float l_blk = 0.0f;
  for (uint page = first_page; page <= last_page; ++page) {
    const ulong page_base = ulong(page_table[page]) * p.page_tokens;
    const uint p_start = page * p.page_tokens;
    const uint begin = max(first, p_start) - p_start;
    const uint end = min(last, p_start + p.page_tokens) - p_start;
    for (uint within = begin; within < end; ++within) {
      const ulong physical = page_base + within;
      device const bfloat *k =
          cache_key + physical * width + ulong(kvh) * p.head_dim;
      device const bfloat *v =
          cache_value + physical * width + ulong(kvh) * p.head_dim;
      float dot = 0.0f;
      for (uint idx = 0; idx < values_per_lane; ++idx) {
        const uint d = lane + idx * 32u;
        if (d < p.head_dim) {
          dot = fma(q_reg[idx], float(k[d]), dot);
        }
      }
      const float s0 = simd_sum(dot) * scale;
      if (s0 > -INFINITY) {
        if (s0 > m_blk) {
          const float rescale =
              (m_blk == -INFINITY || s0 == INFINITY) ? 0.0f : exp(m_blk - s0);
          l_blk *= rescale;
          for (uint idx = 0; idx < values_per_lane; ++idx) {
            acc_blk[idx] *= rescale;
          }
          m_blk = s0;
        }
        const float p0 = exp(s0 - m_blk);
        l_blk += p0;
        for (uint idx = 0; idx < values_per_lane; ++idx) {
          const uint d = lane + idx * 32u;
          if (d < p.head_dim) {
            acc_blk[idx] = fma(p0, float(v[d]), acc_blk[idx]);
          }
        }
      }
    }
  }
  if (lane == 0u) {
    partial_max[record] = m_blk;
    partial_sum[record] = l_blk;
  }
  if (m_blk > -INFINITY) {
    for (uint idx = 0; idx < values_per_lane; ++idx) {
      const uint d = lane + idx * 32u;
      if (d < p.head_dim) {
        partial_output[record * p.head_dim + d] = acc_blk[idx];
      }
    }
  }
}

kernel void omlx_m5_paged_attention_exact_1pass_merge(
    device const bfloat *gate [[buffer(0)]],
    device const float *partial_max [[buffer(1)]],
    device const float *partial_sum [[buffer(2)]],
    device const float *partial_output [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant PagedSplitAttentionParams &params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint logical_head = tg.y;
  const uint token = logical_head / params.query_heads;
  if (token >= params.tokens) {
    return;
  }
  const uint query_head = logical_head - token * params.query_heads;
  const ulong base_record = ulong(logical_head) * params.split_count;
  device const bfloat *gate_row = gate +
                                  ulong(token) * params.gate_token_stride +
                                  ulong(query_head) * params.gate_head_stride;

  if (params.head_dim == 256u) {
    float4 acc_run0 = float4(0.0f), acc_run1 = float4(0.0f);
    float m_run = -INFINITY;
    float l_run = 0.0f;
    for (uint s = 0; s < params.split_count; ++s) {
      const ulong rec = base_record + s;
      const float m_s = partial_max[rec];
      if (m_s > -INFINITY) {
        const float new_m = max(m_run, m_s);
        const float w_old = (m_run == -INFINITY) ? 0.0f : exp(m_run - new_m);
        const float w_new = exp(m_s - new_m);
        m_run = new_m;
        l_run = fma(l_run, w_old, partial_sum[rec] * w_new);
        device const float4 *po4 = reinterpret_cast<device const float4 *>(
            partial_output + rec * 256ul + lane * 8u);
        acc_run0 = fma(acc_run0, float4(w_old), po4[0] * float4(w_new));
        acc_run1 = fma(acc_run1, float4(w_old), po4[1] * float4(w_new));
      }
    }
    const float4 norm0 = (l_run > 0.0f) ? (acc_run0 / l_run) : float4(0.0f);
    const float4 norm1 = (l_run > 0.0f) ? (acc_run1 / l_run) : float4(0.0f);
    float4 o0 = norm0;
    float4 o1 = norm1;
    if (params.gate_head_stride != 0u) {
      device const bfloat4 *gv =
          reinterpret_cast<device const bfloat4 *>(gate_row + lane * 8u);
      o0 /= (1.0f + exp(-float4(gv[0])));
      o1 /= (1.0f + exp(-float4(gv[1])));
    }
    device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(
        output + ulong(logical_head) * 256ul + lane * 8u);
    ov[0] = bfloat4(o0);
    ov[1] = bfloat4(o1);
    return;
  }

  const uint values_per_lane = (params.head_dim + 31u) / 32u;
  float acc_run[8];
  for (uint idx = 0; idx < 8u; ++idx) {
    acc_run[idx] = 0.0f;
  }
  float m_run = -INFINITY;
  float l_run = 0.0f;
  for (uint s = 0; s < params.split_count; ++s) {
    const ulong rec = base_record + s;
    const float m_s = partial_max[rec];
    if (m_s > -INFINITY) {
      const float new_m = max(m_run, m_s);
      const float w_old = (m_run == -INFINITY) ? 0.0f : exp(m_run - new_m);
      const float w_new = exp(m_s - new_m);
      m_run = new_m;
      l_run = fma(l_run, w_old, partial_sum[rec] * w_new);
      device const float *part_o = partial_output + rec * params.head_dim;
      for (uint idx = 0; idx < values_per_lane; ++idx) {
        const uint d = lane + idx * 32u;
        if (d < params.head_dim) {
          acc_run[idx] = fma(acc_run[idx], w_old, part_o[d] * w_new);
        }
      }
    }
  }
  device bfloat *out_row = output + ulong(logical_head) * params.head_dim;
  for (uint idx = 0; idx < values_per_lane; ++idx) {
    const uint d = lane + idx * 32u;
    if (d < params.head_dim) {
      const float norm_o = (l_run > 0.0f) ? (acc_run[idx] / l_run) : 0.0f;
      const float out_val =
          (params.gate_head_stride != 0u)
              ? (norm_o / (1.0f + exp(-float(gate_row[d]))))
              : norm_o;
      out_row[d] = bfloat(out_val);
    }
  }
}

kernel void omlx_m5_paged_causal_attention_exact_1pass(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const bfloat *cache_key [[buffer(2)]],
    device const bfloat *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *page_table [[buffer(5)]],
    device const uint *positions [[buffer(6)]],
    constant PagedAttentionParams &params [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint repeat = params.query_heads / params.key_value_heads;
  const uint chunk_tokens = exact_1pass_chunk_tokens(params.cache_capacity);

  if (tptg.y > 1u && params.head_dim == 256u && repeat > 0u &&
      (tptg.y % repeat) == 0u && tptg.y <= 32u) {
    threadgroup bfloat4 tg_k[kExact1PassKvTileRows * 64u];
    threadgroup bfloat4 tg_v[kExact1PassKvTileRows * 64u];
    threadgroup uint tg_ctx[32];

    const uint bq = tptg.y / repeat;
    const uint kv_head = tg_position.y % params.key_value_heads;
    const uint t_base = (tg_position.y / params.key_value_heads) * bq;
    const uint sg_y = tid_in_tg.y;
    const uint num_sgs = tptg.y;
    const uint flat_tid = sg_y * 32u + lane;
    const uint total_threads = num_sgs * 32u;

    const uint token = t_base + (sg_y / repeat);
    const uint query_head = kv_head * repeat + (sg_y % repeat);
    const bool active = token < params.tokens;
    const uint context_tokens = active ? (positions[token] + 1u) : 0u;

    if (lane == 0u) {
      tg_ctx[sg_y] = context_tokens;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint max_context_tokens = 0u;
    for (uint s = 0; s < num_sgs; ++s) {
      max_context_tokens = max(max_context_tokens, tg_ctx[s]);
    }

    const ulong kv_width = ulong(params.key_value_heads) * 256ul;
    const float scale = rsqrt(256.0f);
    float4 q0 = float4(0.0f), q1 = float4(0.0f);
    if (active) {
      device const bfloat *query_row =
          query + ulong(token) * params.query_token_stride +
          ulong(query_head) * params.query_head_stride;
      device const bfloat4 *qv =
          reinterpret_cast<device const bfloat4 *>(query_row + lane * 8u);
      q0 = float4(qv[0]);
      q1 = float4(qv[1]);
    }

    float m_run = -INFINITY;
    float l_run = 0.0f;
    float4 acc_run0 = float4(0.0f), acc_run1 = float4(0.0f);

    for (uint first = 0u; first < max_context_tokens; first += chunk_tokens) {
      const uint max_last = min(first + chunk_tokens, max_context_tokens);
      const uint row_last =
          (active && context_tokens > first)
              ? min(first + chunk_tokens, context_tokens)
              : first;
      float m_blk = -INFINITY;
      float l_blk = 0.0f;
      float4 acc0 = float4(0.0f), acc1 = float4(0.0f);

      const uint first_page = first / params.page_tokens;
      const uint last_page = (max_last - 1u) / params.page_tokens;
      for (uint page = first_page; page <= last_page; ++page) {
        const ulong page_base = ulong(page_table[page]) * params.page_tokens;
        const uint p_start = page * params.page_tokens;
        const uint page_begin = max(first, p_start) - p_start;
        const uint page_end_max =
            min(max_last, p_start + params.page_tokens) - p_start;
        const uint row_end =
            (active && row_last > first && row_last > p_start)
                ? (min(row_last, p_start + params.page_tokens) - p_start)
                : 0u;
        device const bfloat *k_page =
            cache_key + page_base * kv_width + ulong(kv_head) * 256ul;
        device const bfloat *v_page =
            cache_value + page_base * kv_width + ulong(kv_head) * 256ul;

        for (uint tile_begin = page_begin; tile_begin < page_end_max;
             tile_begin += kExact1PassKvTileRows) {
          const uint tile_len =
              min(kExact1PassKvTileRows, page_end_max - tile_begin);
          const uint total_vecs = tile_len * 64u;
          for (uint idx = flat_tid; idx < total_vecs; idx += total_threads) {
            const uint w_off = idx >> 6u;
            const uint v_off = idx & 63u;
            const ulong row_off = ulong(tile_begin + w_off) * kv_width;
            tg_k[idx] = reinterpret_cast<device const bfloat4 *>(
                k_page + row_off)[v_off];
            tg_v[idx] = reinterpret_cast<device const bfloat4 *>(
                v_page + row_off)[v_off];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (active && row_last > first && row_end > tile_begin) {
            const uint local_valid = min(tile_len, row_end - tile_begin);
            uint w = 0u;
            for (; w + 3u < local_valid; w += 4u) {
              const float s0 =
                  exact_dot_hd256_tg(q0, q1, tg_k + (w + 0u) * 64u, lane) *
                  scale;
              const float s1 =
                  exact_dot_hd256_tg(q0, q1, tg_k + (w + 1u) * 64u, lane) *
                  scale;
              const float s2 =
                  exact_dot_hd256_tg(q0, q1, tg_k + (w + 2u) * 64u, lane) *
                  scale;
              const float s3 =
                  exact_dot_hd256_tg(q0, q1, tg_k + (w + 3u) * 64u, lane) *
                  scale;
              exact_online_update4_hd256_tg(s0, s1, s2, s3, tg_v + w * 64u,
                                            lane, m_blk, l_blk, acc0, acc1);
            }
            for (; w < local_valid; ++w) {
              const float s0 =
                  exact_dot_hd256_tg(q0, q1, tg_k + w * 64u, lane) * scale;
              exact_online_update1_hd256_tg(s0, tg_v + w * 64u, lane, m_blk,
                                            l_blk, acc0, acc1);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }

      if (m_blk > -INFINITY) {
        const float new_m = max(m_run, m_blk);
        const float w_old = (m_run == -INFINITY) ? 0.0f : exp(m_run - new_m);
        const float w_new = exp(m_blk - new_m);
        m_run = new_m;
        l_run = fma(l_run, w_old, l_blk * w_new);
        acc_run0 = fma(acc_run0, float4(w_old), acc0 * float4(w_new));
        acc_run1 = fma(acc_run1, float4(w_old), acc1 * float4(w_new));
      }
    }

    if (active) {
      const float4 norm0 = (l_run > 0.0f) ? (acc_run0 / l_run) : float4(0.0f);
      const float4 norm1 = (l_run > 0.0f) ? (acc_run1 / l_run) : float4(0.0f);
      float4 o0 = norm0;
      float4 o1 = norm1;
      if (params.gate_head_stride != 0u) {
        device const bfloat *gate_row =
            gate + ulong(token) * params.gate_token_stride +
            ulong(query_head) * params.gate_head_stride;
        device const bfloat4 *gv =
            reinterpret_cast<device const bfloat4 *>(gate_row + lane * 8u);
        o0 /= (1.0f + exp(-float4(gv[0])));
        o1 /= (1.0f + exp(-float4(gv[1])));
      }
      device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(
          output + (ulong(token) * params.query_heads + query_head) * 256ul +
          lane * 8u);
      ov[0] = bfloat4(o0);
      ov[1] = bfloat4(o1);
    }
    return;
  }

  const uint logical_head = tg_position.y;
  const uint token = logical_head / params.query_heads;
  if (token >= params.tokens) {
    return;
  }
  const uint query_head = logical_head - token * params.query_heads;
  const uint kv_head = query_head / repeat;
  device const bfloat *query_row = query +
                                   ulong(token) * params.query_token_stride +
                                   ulong(query_head) * params.query_head_stride;
  device const bfloat *gate_row = gate +
                                  ulong(token) * params.gate_token_stride +
                                  ulong(query_head) * params.gate_head_stride;
  const ulong kv_width = ulong(params.key_value_heads) * params.head_dim;
  const uint context_tokens = positions[token] + 1u;
  const float scale = rsqrt(float(params.head_dim));

  if (params.head_dim == 256u) {
    device const bfloat4 *qv =
        reinterpret_cast<device const bfloat4 *>(query_row + lane * 8u);
    const float4 q0 = float4(qv[0]);
    const float4 q1 = float4(qv[1]);
    float m_run = -INFINITY;
    float l_run = 0.0f;
    float4 acc_run0 = float4(0.0f), acc_run1 = float4(0.0f);

    for (uint first = 0u; first < context_tokens; first += chunk_tokens) {
      const uint last = min(first + chunk_tokens, context_tokens);
      float m_blk = -INFINITY;
      float l_blk = 0.0f;
      float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
      const uint first_page = first / params.page_tokens;
      const uint last_page = (last - 1u) / params.page_tokens;
      for (uint page = first_page; page <= last_page; ++page) {
        const ulong page_base = ulong(page_table[page]) * params.page_tokens;
        const uint p_start = page * params.page_tokens;
        const uint begin = max(first, p_start) - p_start;
        const uint end = min(last, p_start + params.page_tokens) - p_start;
        device const bfloat *k_page =
            cache_key + page_base * kv_width + ulong(kv_head) * 256ul;
        device const bfloat *v_page =
            cache_value + page_base * kv_width + ulong(kv_head) * 256ul;
        uint within = begin;
        for (; within + 3u < end; within += 4u) {
          const float s0 =
              exact_dot_hd256(q0, q1, k_page + ulong(within + 0u) * kv_width,
                              lane) *
              scale;
          const float s1 =
              exact_dot_hd256(q0, q1, k_page + ulong(within + 1u) * kv_width,
                              lane) *
              scale;
          const float s2 =
              exact_dot_hd256(q0, q1, k_page + ulong(within + 2u) * kv_width,
                              lane) *
              scale;
          const float s3 =
              exact_dot_hd256(q0, q1, k_page + ulong(within + 3u) * kv_width,
                              lane) *
              scale;
          exact_online_update4_hd256_dev(s0, s1, s2, s3,
                                         v_page + ulong(within) * kv_width,
                                         kv_width, lane, m_blk, l_blk, acc0,
                                         acc1);
        }
        for (; within < end; ++within) {
          const float s0 =
              exact_dot_hd256(q0, q1, k_page + ulong(within) * kv_width, lane) *
              scale;
          exact_online_update1_hd256_dev(s0, v_page + ulong(within) * kv_width,
                                         lane, m_blk, l_blk, acc0, acc1);
        }
      }
      if (m_blk > -INFINITY) {
        const float new_m = max(m_run, m_blk);
        const float w_old = (m_run == -INFINITY) ? 0.0f : exp(m_run - new_m);
        const float w_new = exp(m_blk - new_m);
        m_run = new_m;
        l_run = fma(l_run, w_old, l_blk * w_new);
        acc_run0 = fma(acc_run0, float4(w_old), acc0 * float4(w_new));
        acc_run1 = fma(acc_run1, float4(w_old), acc1 * float4(w_new));
      }
    }

    const float4 norm0 = (l_run > 0.0f) ? (acc_run0 / l_run) : float4(0.0f);
    const float4 norm1 = (l_run > 0.0f) ? (acc_run1 / l_run) : float4(0.0f);
    float4 o0 = norm0;
    float4 o1 = norm1;
    if (params.gate_head_stride != 0u) {
      device const bfloat4 *gv =
          reinterpret_cast<device const bfloat4 *>(gate_row + lane * 8u);
      o0 /= (1.0f + exp(-float4(gv[0])));
      o1 /= (1.0f + exp(-float4(gv[1])));
    }
    device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(
        output + ulong(logical_head) * 256ul + lane * 8u);
    ov[0] = bfloat4(o0);
    ov[1] = bfloat4(o1);
    return;
  }

  float q_reg[8];
  float acc_run[8];
  const uint values_per_lane = (params.head_dim + 31u) / 32u;
  for (uint idx = 0; idx < 8u; ++idx) {
    const uint d = lane + idx * 32u;
    q_reg[idx] = d < params.head_dim ? float(query_row[d]) : 0.0f;
    acc_run[idx] = 0.0f;
  }
  float m_run = -INFINITY;
  float l_run = 0.0f;

  for (uint first = 0u; first < context_tokens; first += chunk_tokens) {
    const uint last = min(first + chunk_tokens, context_tokens);
    float m_blk = -INFINITY;
    float l_blk = 0.0f;
    float acc_blk[8];
    for (uint idx = 0; idx < 8u; ++idx) {
      acc_blk[idx] = 0.0f;
    }
    const uint first_page = first / params.page_tokens;
    const uint last_page = (last - 1u) / params.page_tokens;
    for (uint page = first_page; page <= last_page; ++page) {
      const ulong page_base = ulong(page_table[page]) * params.page_tokens;
      const uint p_start = page * params.page_tokens;
      const uint begin = max(first, p_start) - p_start;
      const uint end = min(last, p_start + params.page_tokens) - p_start;
      for (uint within = begin; within < end; ++within) {
        const ulong position = page_base + within;
        device const bfloat *key_row =
            cache_key + position * kv_width + ulong(kv_head) * params.head_dim;
        device const bfloat *value_row =
            cache_value + position * kv_width + ulong(kv_head) * params.head_dim;
        float partial = 0.0f;
        for (uint idx = 0; idx < values_per_lane; ++idx) {
          const uint feature = lane + idx * 32u;
          if (feature < params.head_dim) {
            partial = fma(q_reg[idx], float(key_row[feature]), partial);
          }
        }
        const float s0 = simd_sum(partial) * scale;
        if (s0 > -INFINITY) {
          if (s0 > m_blk) {
            const float rescale =
                (m_blk == -INFINITY || s0 == INFINITY) ? 0.0f : exp(m_blk - s0);
            l_blk *= rescale;
            for (uint idx = 0; idx < values_per_lane; ++idx) {
              acc_blk[idx] *= rescale;
            }
            m_blk = s0;
          }
          const float p0 = exp(s0 - m_blk);
          l_blk += p0;
          for (uint idx = 0; idx < values_per_lane; ++idx) {
            const uint feature = lane + idx * 32u;
            if (feature < params.head_dim) {
              acc_blk[idx] = fma(p0, float(value_row[feature]), acc_blk[idx]);
            }
          }
        }
      }
    }
    if (m_blk > -INFINITY) {
      const float new_m = max(m_run, m_blk);
      const float w_old = (m_run == -INFINITY) ? 0.0f : exp(m_run - new_m);
      const float w_new = exp(m_blk - new_m);
      m_run = new_m;
      l_run = fma(l_run, w_old, l_blk * w_new);
      for (uint idx = 0; idx < values_per_lane; ++idx) {
        acc_run[idx] = fma(acc_run[idx], w_old, acc_blk[idx] * w_new);
      }
    }
  }

  for (uint idx = 0; idx < values_per_lane; ++idx) {
    const uint feature = lane + idx * 32u;
    if (feature < params.head_dim) {
      const float norm_o = (l_run > 0.0f) ? (acc_run[idx] / l_run) : 0.0f;
      const float out_val =
          (params.gate_head_stride != 0u)
              ? (norm_o / (1.0f + exp(-float(gate_row[feature]))))
              : norm_o;
      output[ulong(logical_head) * params.head_dim + feature] =
          bfloat(out_val);
    }
  }
}

// ============================================================================
// Dim D (v0.8.0-m5rt.14): Multi-Slot (grid.z = B, B <= 8) Strided Exact
// Attention Kernels (Shared KV Arena, Per-Slot Page Tables, Zero emit_split3).
// 100% 0-ULP bit-identical to single-slot exact 1-pass and strict 2-pass!
// ============================================================================

struct AttentionBatchedPostprocessParams {
  uint query_heads;
  uint key_value_heads;
  uint head_dim;
  uint rotary_dim;
  uint total_tokens;
  uint query_token_stride;
  uint query_head_stride;
  uint key_token_stride;
  uint key_head_stride;
  uint value_token_stride;
  uint batch_size;
  float epsilon;
  float theta;
  float weight_offset;
  uint row_offsets[8];
  uint widths[8];
};

struct PagedAttentionBatchedParams {
  uint query_heads;
  uint key_value_heads;
  uint head_dim;
  uint batch_size;
  uint page_tokens;
  uint physical_pages;
  uint query_token_stride;
  uint query_head_stride;
  uint gate_token_stride;
  uint gate_head_stride;
  uint row_offsets[8];
  uint widths[8];
  uint slot_cache_capacity[8];
};

inline device const uint *attn_select_pt8(
    uint slot,
    device const uint *pt0, device const uint *pt1,
    device const uint *pt2, device const uint *pt3,
    device const uint *pt4, device const uint *pt5,
    device const uint *pt6, device const uint *pt7) {
  switch (slot) {
  case 0u: return pt0;
  case 1u: return pt1;
  case 2u: return pt2;
  case 3u: return pt3;
  case 4u: return pt4;
  case 5u: return pt5;
  case 6u: return pt6;
  default: return pt7;
  }
}

inline void exact_load_q_gen8(device const bfloat *row, uint lane, uint head_dim,
                              thread float4 &q_lo, thread float4 &q_hi) {
  q_lo = float4(
      (lane + 0u < head_dim) ? float(row[lane + 0u]) : 0.0f,
      (lane + 32u < head_dim) ? float(row[lane + 32u]) : 0.0f,
      (lane + 64u < head_dim) ? float(row[lane + 64u]) : 0.0f,
      (lane + 96u < head_dim) ? float(row[lane + 96u]) : 0.0f);
  q_hi = float4(
      (lane + 128u < head_dim) ? float(row[lane + 128u]) : 0.0f,
      (lane + 160u < head_dim) ? float(row[lane + 160u]) : 0.0f,
      (lane + 192u < head_dim) ? float(row[lane + 192u]) : 0.0f,
      (lane + 224u < head_dim) ? float(row[lane + 224u]) : 0.0f);
}

inline float exact_dot_gen8(float4 q_lo, float4 q_hi, device const bfloat *k_row,
                            uint lane, uint head_dim) {
  float dot = 0.0f;
  if (lane + 0u < head_dim) dot = fma(q_lo.x, float(k_row[lane + 0u]), dot);
  if (lane + 32u < head_dim) dot = fma(q_lo.y, float(k_row[lane + 32u]), dot);
  if (lane + 64u < head_dim) dot = fma(q_lo.z, float(k_row[lane + 64u]), dot);
  if (lane + 96u < head_dim) dot = fma(q_lo.w, float(k_row[lane + 96u]), dot);
  if (lane + 128u < head_dim) dot = fma(q_hi.x, float(k_row[lane + 128u]), dot);
  if (lane + 160u < head_dim) dot = fma(q_hi.y, float(k_row[lane + 160u]), dot);
  if (lane + 192u < head_dim) dot = fma(q_hi.z, float(k_row[lane + 192u]), dot);
  if (lane + 224u < head_dim) dot = fma(q_hi.w, float(k_row[lane + 224u]), dot);
  return simd_sum(dot);
}

inline void exact_v_accum_gen8(float prob, device const bfloat *v_row,
                               uint lane, uint head_dim,
                               thread float4 &a_lo, thread float4 &a_hi) {
  if (lane + 0u < head_dim) a_lo.x = fma(prob, float(v_row[lane + 0u]), a_lo.x);
  if (lane + 32u < head_dim) a_lo.y = fma(prob, float(v_row[lane + 32u]), a_lo.y);
  if (lane + 64u < head_dim) a_lo.z = fma(prob, float(v_row[lane + 64u]), a_lo.z);
  if (lane + 96u < head_dim) a_lo.w = fma(prob, float(v_row[lane + 96u]), a_lo.w);
  if (lane + 128u < head_dim) a_hi.x = fma(prob, float(v_row[lane + 128u]), a_hi.x);
  if (lane + 160u < head_dim) a_hi.y = fma(prob, float(v_row[lane + 160u]), a_hi.y);
  if (lane + 192u < head_dim) a_hi.z = fma(prob, float(v_row[lane + 192u]), a_hi.z);
  if (lane + 224u < head_dim) a_hi.w = fma(prob, float(v_row[lane + 224u]), a_hi.w);
}

inline void exact_store_gated_gen8(float4 a_lo, float4 a_hi, float denom,
                                   device const bfloat *gate_row,
                                   device bfloat *out_row, uint lane,
                                   uint head_dim, bool has_gate) {
  const bool valid = (denom > 0.0f);
  const float vals[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                         a_hi.x, a_hi.y, a_hi.z, a_hi.w};
#pragma clang loop unroll(full)
  for (uint idx = 0u; idx < 8u; ++idx) {
    const uint d = lane + idx * 32u;
    if (d < head_dim) {
      const float o = valid ? (vals[idx] / denom) : 0.0f;
      out_row[d] = bfloat(has_gate ? (o / (1.0f + exp(-float(gate_row[d]))))
                                   : o);
    }
  }
}

kernel void omlx_m5_attention_qk_norm_rope_kv_append_batched_exact(
    device bfloat *query [[buffer(0)]],
    device bfloat *key [[buffer(1)]],
    device const bfloat *value [[buffer(2)]],
    device const bfloat *query_norm [[buffer(3)]],
    device const bfloat *key_norm [[buffer(4)]],
    device bfloat *cache_key [[buffer(5)]],
    device bfloat *cache_value [[buffer(6)]],
    device const uint *positions [[buffer(7)]],
    device const uint *pt0 [[buffer(8)]],
    device const uint *pt1 [[buffer(9)]],
    device const uint *pt2 [[buffer(10)]],
    device const uint *pt3 [[buffer(11)]],
    device const uint *pt4 [[buffer(12)]],
    device const uint *pt5 [[buffer(13)]],
    device const uint *pt6 [[buffer(14)]],
    device const uint *pt7 [[buffer(15)]],
    constant AttentionBatchedPostprocessParams &params [[buffer(16)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint heads_per_token = params.query_heads + params.key_value_heads;
  const uint logical_head = tg_position.y;
  const uint token = logical_head / heads_per_token;
  if (token >= params.total_tokens) {
    return;
  }
  const uint local_head = logical_head - token * heads_per_token;
  const bool is_query = local_head < params.query_heads;
  const uint head = is_query ? local_head : local_head - params.query_heads;
  device bfloat *row =
      is_query ? query + ulong(token) * params.query_token_stride +
                     ulong(head) * params.query_head_stride
               : key + ulong(token) * params.key_token_stride +
                     ulong(head) * params.key_head_stride;
  device const bfloat *norm = is_query ? query_norm : key_norm;

  const bool has_qk_norm = (params.epsilon >= 0.0f);
  float inverse_rms = 1.0f;
  if (has_qk_norm) {
    float square_sum = 0.0f;
    for (uint column = lane; column < params.head_dim; column += 32u) {
      const float val = float(row[column]);
      square_sum = fma(val, val, square_sum);
    }
    inverse_rms =
        rsqrt(simd_sum(square_sum) / float(params.head_dim) + params.epsilon);
  }
  const uint half_rotary = params.rotary_dim / 2u;
  const uint logical_pos = positions[token];
  const float position = float(logical_pos);

  if (is_query) {
    for (uint pair = lane; pair < half_rotary; pair += 32u) {
      const uint first_index = pair;
      const uint second_index = pair + half_rotary;
      const float first =
          has_qk_norm ? (float(row[first_index]) * inverse_rms *
                         (float(norm[first_index]) + params.weight_offset))
                      : float(row[first_index]);
      const float second =
          has_qk_norm ? (float(row[second_index]) * inverse_rms *
                         (float(norm[second_index]) + params.weight_offset))
                      : float(row[second_index]);
      const float exponent = float(2u * pair) / float(params.rotary_dim);
      const float angle = position / pow(params.theta, exponent);
      const float cosine = cos(angle);
      const float sine = sin(angle);
      row[first_index] = bfloat(fma(-second, sine, first * cosine));
      row[second_index] = bfloat(fma(first, sine, second * cosine));
    }
    if (has_qk_norm) {
      for (uint column = params.rotary_dim + lane; column < params.head_dim;
           column += 32u) {
        row[column] = bfloat(float(row[column]) * inverse_rms *
                             (float(norm[column]) + params.weight_offset));
      }
    }
  } else {
    uint slot = 0u;
    for (uint s = 0u; s < params.batch_size; ++s) {
      if (token >= params.row_offsets[s] &&
          token < params.row_offsets[s] + params.widths[s]) {
        slot = s;
        break;
      }
    }
    device const uint *page_table =
        attn_select_pt8(slot, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7);
    const uint page = logical_pos >> 6u;
    const uint within = logical_pos & 63u;
    const ulong paged_pos = (ulong(page_table[page]) << 6ul) + ulong(within);
    const ulong kv_width = ulong(params.key_value_heads) * params.head_dim;
    const ulong dst_base = paged_pos * kv_width + ulong(head) * params.head_dim;
    device bfloat *ck_head = cache_key + dst_base;
    device bfloat *cv_head = cache_value + dst_base;
    device const bfloat *v_head = value + ulong(token) * params.value_token_stride +
                                  ulong(head) * params.head_dim;
    for (uint pair = lane; pair < half_rotary; pair += 32u) {
      const uint first_index = pair;
      const uint second_index = pair + half_rotary;
      const float first =
          has_qk_norm ? (float(row[first_index]) * inverse_rms *
                         (float(norm[first_index]) + params.weight_offset))
                      : float(row[first_index]);
      const float second =
          has_qk_norm ? (float(row[second_index]) * inverse_rms *
                         (float(norm[second_index]) + params.weight_offset))
                      : float(row[second_index]);
      const float exponent = float(2u * pair) / float(params.rotary_dim);
      const float angle = position / pow(params.theta, exponent);
      const float cosine = cos(angle);
      const float sine = sin(angle);
      const bfloat k0 = bfloat(fma(-second, sine, first * cosine));
      const bfloat k1 = bfloat(fma(first, sine, second * cosine));
      row[first_index] = k0;
      row[second_index] = k1;
      ck_head[first_index] = k0;
      ck_head[second_index] = k1;
      cv_head[first_index] = v_head[first_index];
      cv_head[second_index] = v_head[second_index];
    }
    for (uint column = params.rotary_dim + lane; column < params.head_dim;
         column += 32u) {
      const bfloat kv_normed =
          has_qk_norm ? bfloat(float(row[column]) * inverse_rms *
                               (float(norm[column]) + params.weight_offset))
                      : row[column];
      row[column] = kv_normed;
      ck_head[column] = kv_normed;
      cv_head[column] = v_head[column];
    }
  }
}

kernel void omlx_m5_paged_causal_attention_batched_exact_1pass(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const bfloat *cache_key [[buffer(2)]],
    device const bfloat *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *positions [[buffer(5)]],
    device const uint *pt0 [[buffer(6)]],
    device const uint *pt1 [[buffer(7)]],
    device const uint *pt2 [[buffer(8)]],
    device const uint *pt3 [[buffer(9)]],
    device const uint *pt4 [[buffer(10)]],
    device const uint *pt5 [[buffer(11)]],
    device const uint *pt6 [[buffer(12)]],
    device const uint *pt7 [[buffer(13)]],
    constant PagedAttentionBatchedParams &params [[buffer(14)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint slot = tg_position.z;
  if (slot >= params.batch_size) {
    return;
  }
  const uint r_off = params.row_offsets[slot];
  const uint slot_tokens = params.widths[slot];
  device const uint *page_table =
      attn_select_pt8(slot, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7);
  const uint repeat = params.query_heads / params.key_value_heads;
  const uint chunk_tokens =
      exact_1pass_chunk_tokens(params.slot_cache_capacity[slot]);

  if (tptg.y > 1u && params.head_dim == 256u && repeat > 0u &&
      (tptg.y % repeat) == 0u && tptg.y <= 32u) {
    threadgroup bfloat4 tg_k[kExact1PassKvTileRows * 64u];
    threadgroup bfloat4 tg_v[kExact1PassKvTileRows * 64u];
    threadgroup uint tg_ctx[32];

    const uint bq = tptg.y / repeat;
    const uint kv_head = tg_position.y % params.key_value_heads;
    const uint t_base = (tg_position.y / params.key_value_heads) * bq;
    const uint sg_y = tid_in_tg.y;
    const uint num_sgs = tptg.y;
    const uint flat_tid = sg_y * 32u + lane;
    const uint total_threads = num_sgs * 32u;

    const uint local_tok = t_base + (sg_y / repeat);
    const uint query_head = kv_head * repeat + (sg_y % repeat);
    const bool active = local_tok < slot_tokens;
    const uint global_tok = r_off + local_tok;
    const uint context_tokens = active ? (positions[global_tok] + 1u) : 0u;

    if (lane == 0u) {
      tg_ctx[sg_y] = context_tokens;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint max_context_tokens = 0u;
    for (uint s = 0u; s < num_sgs; ++s) {
      max_context_tokens = max(max_context_tokens, tg_ctx[s]);
    }

    const ulong kv_width = ulong(params.key_value_heads) * 256ul;
    const float scale = rsqrt(256.0f);
    float4 q0 = float4(0.0f), q1 = float4(0.0f);
    if (active) {
      device const bfloat *query_row =
          query + ulong(global_tok) * params.query_token_stride +
          ulong(query_head) * params.query_head_stride;
      device const bfloat4 *qv =
          reinterpret_cast<device const bfloat4 *>(query_row + lane * 8u);
      q0 = float4(qv[0]);
      q1 = float4(qv[1]);
    }

    float m_run = -INFINITY;
    float l_run = 0.0f;
    float4 acc_run0 = float4(0.0f), acc_run1 = float4(0.0f);

    for (uint first = 0u; first < max_context_tokens; first += chunk_tokens) {
      const uint max_last = min(first + chunk_tokens, max_context_tokens);
      const uint row_last =
          (active && context_tokens > first)
              ? min(first + chunk_tokens, context_tokens)
              : first;
      float m_blk = -INFINITY;
      float l_blk = 0.0f;
      float4 acc0 = float4(0.0f), acc1 = float4(0.0f);

      const uint first_page = first / params.page_tokens;
      const uint last_page = (max_last - 1u) / params.page_tokens;
      for (uint page = first_page; page <= last_page; ++page) {
        const ulong page_base = ulong(page_table[page]) * params.page_tokens;
        const uint p_start = page * params.page_tokens;
        const uint page_begin = max(first, p_start) - p_start;
        const uint page_end_max =
            min(max_last, p_start + params.page_tokens) - p_start;
        const uint row_end =
            (active && row_last > first && row_last > p_start)
                ? (min(row_last, p_start + params.page_tokens) - p_start)
                : 0u;
        device const bfloat *k_page =
            cache_key + page_base * kv_width + ulong(kv_head) * 256ul;
        device const bfloat *v_page =
            cache_value + page_base * kv_width + ulong(kv_head) * 256ul;

        for (uint tile_begin = page_begin; tile_begin < page_end_max;
             tile_begin += kExact1PassKvTileRows) {
          const uint tile_len =
              min(kExact1PassKvTileRows, page_end_max - tile_begin);
          const uint total_vecs = tile_len * 64u;
          for (uint idx = flat_tid; idx < total_vecs; idx += total_threads) {
            const uint w_off = idx >> 6u;
            const uint v_off = idx & 63u;
            const ulong row_off = ulong(tile_begin + w_off) * kv_width;
            tg_k[idx] = reinterpret_cast<device const bfloat4 *>(
                k_page + row_off)[v_off];
            tg_v[idx] = reinterpret_cast<device const bfloat4 *>(
                v_page + row_off)[v_off];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (active && row_last > first && row_end > tile_begin) {
            const uint local_valid = min(tile_len, row_end - tile_begin);
            uint w = 0u;
            for (; w + 3u < local_valid; w += 4u) {
              const float s0 =
                  exact_dot_hd256_tg(q0, q1, tg_k + (w + 0u) * 64u, lane) * scale;
              const float s1 =
                  exact_dot_hd256_tg(q0, q1, tg_k + (w + 1u) * 64u, lane) * scale;
              const float s2 =
                  exact_dot_hd256_tg(q0, q1, tg_k + (w + 2u) * 64u, lane) * scale;
              const float s3 =
                  exact_dot_hd256_tg(q0, q1, tg_k + (w + 3u) * 64u, lane) * scale;
              exact_online_update4_hd256_tg(s0, s1, s2, s3, tg_v + w * 64u,
                                            lane, m_blk, l_blk, acc0, acc1);
            }
            for (; w < local_valid; ++w) {
              const float s0 =
                  exact_dot_hd256_tg(q0, q1, tg_k + w * 64u, lane) * scale;
              exact_online_update1_hd256_tg(s0, tg_v + w * 64u, lane, m_blk,
                                            l_blk, acc0, acc1);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }

      if (m_blk > -INFINITY) {
        const float new_m = max(m_run, m_blk);
        const float w_old = (m_run == -INFINITY) ? 0.0f : exp(m_run - new_m);
        const float w_new = exp(m_blk - new_m);
        m_run = new_m;
        l_run = fma(l_run, w_old, l_blk * w_new);
        acc_run0 = fma(acc_run0, float4(w_old), acc0 * float4(w_new));
        acc_run1 = fma(acc_run1, float4(w_old), acc1 * float4(w_new));
      }
    }

    if (active) {
      const float4 norm0 = (l_run > 0.0f) ? (acc_run0 / l_run) : float4(0.0f);
      const float4 norm1 = (l_run > 0.0f) ? (acc_run1 / l_run) : float4(0.0f);
      float4 o0 = norm0;
      float4 o1 = norm1;
      if (params.gate_head_stride != 0u) {
        device const bfloat *gate_row =
            gate + ulong(global_tok) * params.gate_token_stride +
            ulong(query_head) * params.gate_head_stride;
        device const bfloat4 *gv =
            reinterpret_cast<device const bfloat4 *>(gate_row + lane * 8u);
        o0 /= (1.0f + exp(-float4(gv[0])));
        o1 /= (1.0f + exp(-float4(gv[1])));
      }
      device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(
          output + (ulong(global_tok) * params.query_heads + query_head) * 256ul +
          lane * 8u);
      ov[0] = bfloat4(o0);
      ov[1] = bfloat4(o1);
    }
    return;
  }

  const uint logical_head = tg_position.y;
  const uint local_tok = logical_head / params.query_heads;
  if (local_tok >= slot_tokens) {
    return;
  }
  const uint global_tok = r_off + local_tok;
  const uint query_head = logical_head - local_tok * params.query_heads;
  const uint kv_head = query_head / repeat;
  device const bfloat *query_row = query +
                                   ulong(global_tok) * params.query_token_stride +
                                   ulong(query_head) * params.query_head_stride;
  device const bfloat *gate_row = gate +
                                  ulong(global_tok) * params.gate_token_stride +
                                  ulong(query_head) * params.gate_head_stride;
  const ulong kv_width = ulong(params.key_value_heads) * params.head_dim;
  const uint context_tokens = positions[global_tok] + 1u;
  const float scale = rsqrt(float(params.head_dim));

  if (params.head_dim == 256u) {
    device const bfloat4 *qv =
        reinterpret_cast<device const bfloat4 *>(query_row + lane * 8u);
    const float4 q0 = float4(qv[0]);
    const float4 q1 = float4(qv[1]);
    float m_run = -INFINITY;
    float l_run = 0.0f;
    float4 acc_run0 = float4(0.0f), acc_run1 = float4(0.0f);

    for (uint first = 0u; first < context_tokens; first += chunk_tokens) {
      const uint last = min(first + chunk_tokens, context_tokens);
      float m_blk = -INFINITY;
      float l_blk = 0.0f;
      float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
      const uint first_page = first / params.page_tokens;
      const uint last_page = (last - 1u) / params.page_tokens;
      for (uint page = first_page; page <= last_page; ++page) {
        const ulong page_base = ulong(page_table[page]) * params.page_tokens;
        const uint p_start = page * params.page_tokens;
        const uint begin = max(first, p_start) - p_start;
        const uint end = min(last, p_start + params.page_tokens) - p_start;
        device const bfloat *k_page =
            cache_key + page_base * kv_width + ulong(kv_head) * 256ul;
        device const bfloat *v_page =
            cache_value + page_base * kv_width + ulong(kv_head) * 256ul;
        uint within = begin;
        for (; within + 3u < end; within += 4u) {
          const float s0 =
              exact_dot_hd256(q0, q1, k_page + ulong(within + 0u) * kv_width, lane) * scale;
          const float s1 =
              exact_dot_hd256(q0, q1, k_page + ulong(within + 1u) * kv_width, lane) * scale;
          const float s2 =
              exact_dot_hd256(q0, q1, k_page + ulong(within + 2u) * kv_width, lane) * scale;
          const float s3 =
              exact_dot_hd256(q0, q1, k_page + ulong(within + 3u) * kv_width, lane) * scale;
          exact_online_update4_hd256_dev(s0, s1, s2, s3,
                                         v_page + ulong(within) * kv_width,
                                         kv_width, lane, m_blk, l_blk, acc0, acc1);
        }
        for (; within < end; ++within) {
          const float s0 =
              exact_dot_hd256(q0, q1, k_page + ulong(within) * kv_width, lane) * scale;
          exact_online_update1_hd256_dev(s0, v_page + ulong(within) * kv_width,
                                         lane, m_blk, l_blk, acc0, acc1);
        }
      }
      if (m_blk > -INFINITY) {
        const float new_m = max(m_run, m_blk);
        const float w_old = (m_run == -INFINITY) ? 0.0f : exp(m_run - new_m);
        const float w_new = exp(m_blk - new_m);
        m_run = new_m;
        l_run = fma(l_run, w_old, l_blk * w_new);
        acc_run0 = fma(acc_run0, float4(w_old), acc0 * float4(w_new));
        acc_run1 = fma(acc_run1, float4(w_old), acc1 * float4(w_new));
      }
    }

    const float4 norm0 = (l_run > 0.0f) ? (acc_run0 / l_run) : float4(0.0f);
    const float4 norm1 = (l_run > 0.0f) ? (acc_run1 / l_run) : float4(0.0f);
    float4 o0 = norm0;
    float4 o1 = norm1;
    if (params.gate_head_stride != 0u) {
      device const bfloat4 *gv =
          reinterpret_cast<device const bfloat4 *>(gate_row + lane * 8u);
      o0 /= (1.0f + exp(-float4(gv[0])));
      o1 /= (1.0f + exp(-float4(gv[1])));
    }
    device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(
        output + (ulong(global_tok) * params.query_heads + query_head) * 256ul +
        lane * 8u);
    ov[0] = bfloat4(o0);
    ov[1] = bfloat4(o1);
    return;
  }

  float4 q_lo, q_hi;
  exact_load_q_gen8(query_row, lane, params.head_dim, q_lo, q_hi);
  float4 acc_run_lo = float4(0.0f), acc_run_hi = float4(0.0f);
  float m_run = -INFINITY;
  float l_run = 0.0f;

  for (uint first = 0u; first < context_tokens; first += chunk_tokens) {
    const uint last = min(first + chunk_tokens, context_tokens);
    float m_blk = -INFINITY;
    float l_blk = 0.0f;
    float4 acc_blk_lo = float4(0.0f), acc_blk_hi = float4(0.0f);
    const uint first_page = first / params.page_tokens;
    const uint last_page = (last - 1u) / params.page_tokens;
    for (uint page = first_page; page <= last_page; ++page) {
      const ulong page_base = ulong(page_table[page]) * params.page_tokens;
      const uint p_start = page * params.page_tokens;
      const uint begin = max(first, p_start) - p_start;
      const uint end = min(last, p_start + params.page_tokens) - p_start;
      for (uint within = begin; within < end; ++within) {
        const ulong position = page_base + within;
        device const bfloat *key_row =
            cache_key + position * kv_width + ulong(kv_head) * params.head_dim;
        device const bfloat *value_row =
            cache_value + position * kv_width + ulong(kv_head) * params.head_dim;
        const float s0 = exact_dot_gen8(q_lo, q_hi, key_row, lane, params.head_dim) * scale;
        if (s0 > -INFINITY) {
          if (s0 > m_blk) {
            const float rescale =
                (m_blk == -INFINITY || s0 == INFINITY) ? 0.0f : exp(m_blk - s0);
            l_blk *= rescale;
            acc_blk_lo *= rescale;
            acc_blk_hi *= rescale;
            m_blk = s0;
          }
          const float p0 = exp(s0 - m_blk);
          l_blk += p0;
          exact_v_accum_gen8(p0, value_row, lane, params.head_dim, acc_blk_lo, acc_blk_hi);
        }
      }
    }
    if (m_blk > -INFINITY) {
      const float new_m = max(m_run, m_blk);
      const float w_old = (m_run == -INFINITY) ? 0.0f : exp(m_run - new_m);
      const float w_new = exp(m_blk - new_m);
      m_run = new_m;
      l_run = fma(l_run, w_old, l_blk * w_new);
      acc_run_lo = fma(acc_run_lo, float4(w_old), acc_blk_lo * float4(w_new));
      acc_run_hi = fma(acc_run_hi, float4(w_old), acc_blk_hi * float4(w_new));
    }
  }

  device bfloat *out_row =
      output + (ulong(global_tok) * params.query_heads + query_head) * params.head_dim;
  exact_store_gated_gen8(acc_run_lo, acc_run_hi, l_run, gate_row, out_row, lane,
                         params.head_dim, params.gate_head_stride != 0u);
}

kernel void omlx_m5_paged_causal_attention_batched_exact(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const bfloat *cache_key [[buffer(2)]],
    device const bfloat *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *positions [[buffer(5)]],
    device const uint *pt0 [[buffer(6)]],
    device const uint *pt1 [[buffer(7)]],
    device const uint *pt2 [[buffer(8)]],
    device const uint *pt3 [[buffer(9)]],
    device const uint *pt4 [[buffer(10)]],
    device const uint *pt5 [[buffer(11)]],
    device const uint *pt6 [[buffer(12)]],
    device const uint *pt7 [[buffer(13)]],
    constant PagedAttentionBatchedParams &params [[buffer(14)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint slot = tg_position.z;
  if (slot >= params.batch_size) {
    return;
  }
  const uint r_off = params.row_offsets[slot];
  const uint slot_tokens = params.widths[slot];
  device const uint *page_table =
      attn_select_pt8(slot, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7);
  const uint repeat = params.query_heads / params.key_value_heads;

  if (tptg.y > 1u && params.head_dim == 256u && repeat > 0u &&
      (tptg.y % repeat) == 0u && tptg.y <= 32u) {
    threadgroup bfloat4 tg_k[kExactKvTileRows * 64u];
    threadgroup bfloat4 tg_v[kExactKvTileRows * 64u];
    threadgroup uint tg_ctx[32];

    const uint bq = tptg.y / repeat;
    const uint kv_head = tg_position.y % params.key_value_heads;
    const uint t_base = (tg_position.y / params.key_value_heads) * bq;
    const uint sg_y = tid_in_tg.y;
    const uint num_sgs = tptg.y;
    const uint flat_tid = sg_y * 32u + lane;
    const uint total_threads = num_sgs * 32u;

    const uint local_tok = t_base + (sg_y / repeat);
    const uint query_head = kv_head * repeat + (sg_y % repeat);
    const bool active = local_tok < slot_tokens;
    const uint global_tok = r_off + local_tok;
    const uint context_tokens = active ? (positions[global_tok] + 1u) : 0u;

    if (lane == 0u) {
      tg_ctx[sg_y] = context_tokens;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint max_context_tokens = 0u;
    for (uint s = 0u; s < num_sgs; ++s) {
      max_context_tokens = max(max_context_tokens, tg_ctx[s]);
    }

    const ulong kv_width = ulong(params.key_value_heads) * 256ul;
    const float scale = rsqrt(256.0f);
    float4 q0 = float4(0.0f), q1 = float4(0.0f);
    if (active) {
      device const bfloat *query_row =
          query + ulong(global_tok) * params.query_token_stride +
          ulong(query_head) * params.query_head_stride;
      device const bfloat4 *qv =
          reinterpret_cast<device const bfloat4 *>(query_row + lane * 8u);
      q0 = float4(qv[0]);
      q1 = float4(qv[1]);
    }

    const uint max_pages =
        (max_context_tokens + params.page_tokens - 1u) / params.page_tokens;
    float maximum = -INFINITY;
    for (uint page = 0u; page < max_pages; ++page) {
      const ulong page_base = ulong(page_table[page]) * params.page_tokens;
      const uint p_start = page * params.page_tokens;
      const uint valid_max =
          min(params.page_tokens, max_context_tokens - p_start);
      const uint row_valid =
          (active && context_tokens > p_start)
              ? min(params.page_tokens, context_tokens - p_start)
              : 0u;
      device const bfloat *k_page =
          cache_key + page_base * kv_width + ulong(kv_head) * 256ul;

      for (uint tile_begin = 0u; tile_begin < valid_max;
           tile_begin += kExactKvTileRows) {
        const uint tile_len = min(kExactKvTileRows, valid_max - tile_begin);
        const uint total_vecs = tile_len * 64u;
        for (uint idx = flat_tid; idx < total_vecs; idx += total_threads) {
          const uint w_off = idx >> 6u;
          const uint v_off = idx & 63u;
          device const bfloat4 *src = reinterpret_cast<device const bfloat4 *>(
              k_page + ulong(tile_begin + w_off) * kv_width);
          tg_k[idx] = src[v_off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active && row_valid > tile_begin) {
          const uint local_valid = min(tile_len, row_valid - tile_begin);
          uint w = 0u;
          for (; w + 3u < local_valid; w += 4u) {
            maximum = max(maximum, exact_dot_hd256_tg(q0, q1, tg_k + (w + 0u) * 64u, lane) * scale);
            maximum = max(maximum, exact_dot_hd256_tg(q0, q1, tg_k + (w + 1u) * 64u, lane) * scale);
            maximum = max(maximum, exact_dot_hd256_tg(q0, q1, tg_k + (w + 2u) * 64u, lane) * scale);
            maximum = max(maximum, exact_dot_hd256_tg(q0, q1, tg_k + (w + 3u) * 64u, lane) * scale);
          }
          for (; w < local_valid; ++w) {
            maximum = max(maximum, exact_dot_hd256_tg(q0, q1, tg_k + w * 64u, lane) * scale);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    float denominator = 0.0f;
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
    for (uint page = 0u; page < max_pages; ++page) {
      const ulong page_base = ulong(page_table[page]) * params.page_tokens;
      const uint p_start = page * params.page_tokens;
      const uint valid_max =
          min(params.page_tokens, max_context_tokens - p_start);
      const uint row_valid =
          (active && context_tokens > p_start)
              ? min(params.page_tokens, context_tokens - p_start)
              : 0u;
      device const bfloat *k_page =
          cache_key + page_base * kv_width + ulong(kv_head) * 256ul;
      device const bfloat *v_page =
          cache_value + page_base * kv_width + ulong(kv_head) * 256ul;

      for (uint tile_begin = 0u; tile_begin < valid_max;
           tile_begin += kExactKvTileRows) {
        const uint tile_len = min(kExactKvTileRows, valid_max - tile_begin);
        const uint total_vecs = tile_len * 64u;
        for (uint idx = flat_tid; idx < total_vecs; idx += total_threads) {
          const uint w_off = idx >> 6u;
          const uint v_off = idx & 63u;
          const ulong row_off = ulong(tile_begin + w_off) * kv_width;
          tg_k[idx] =
              reinterpret_cast<device const bfloat4 *>(k_page + row_off)[v_off];
          tg_v[idx] =
              reinterpret_cast<device const bfloat4 *>(v_page + row_off)[v_off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active && row_valid > tile_begin) {
          const uint local_valid = min(tile_len, row_valid - tile_begin);
          uint w = 0u;
          for (; w + 3u < local_valid; w += 4u) {
            const float p0 = exp(exact_dot_hd256_tg(q0, q1, tg_k + (w + 0u) * 64u, lane) * scale - maximum);
            const float p1 = exp(exact_dot_hd256_tg(q0, q1, tg_k + (w + 1u) * 64u, lane) * scale - maximum);
            const float p2 = exp(exact_dot_hd256_tg(q0, q1, tg_k + (w + 2u) * 64u, lane) * scale - maximum);
            const float p3 = exp(exact_dot_hd256_tg(q0, q1, tg_k + (w + 3u) * 64u, lane) * scale - maximum);
            denominator += p0; exact_v_accum_hd256_tg(p0, tg_v + (w + 0u) * 64u, lane, acc0, acc1);
            denominator += p1; exact_v_accum_hd256_tg(p1, tg_v + (w + 1u) * 64u, lane, acc0, acc1);
            denominator += p2; exact_v_accum_hd256_tg(p2, tg_v + (w + 2u) * 64u, lane, acc0, acc1);
            denominator += p3; exact_v_accum_hd256_tg(p3, tg_v + (w + 3u) * 64u, lane, acc0, acc1);
          }
          for (; w < local_valid; ++w) {
            const float p0 = exp(exact_dot_hd256_tg(q0, q1, tg_k + w * 64u, lane) * scale - maximum);
            denominator += p0;
            exact_v_accum_hd256_tg(p0, tg_v + w * 64u, lane, acc0, acc1);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    if (active) {
      float4 o0 = acc0 / denominator;
      float4 o1 = acc1 / denominator;
      if (params.gate_head_stride != 0u) {
        device const bfloat *gate_row =
            gate + ulong(global_tok) * params.gate_token_stride +
            ulong(query_head) * params.gate_head_stride;
        device const bfloat4 *gv =
            reinterpret_cast<device const bfloat4 *>(gate_row + lane * 8u);
        o0 /= (1.0f + exp(-float4(gv[0])));
        o1 /= (1.0f + exp(-float4(gv[1])));
      }
      device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(
          output + (ulong(global_tok) * params.query_heads + query_head) * 256ul +
          lane * 8u);
      ov[0] = bfloat4(o0);
      ov[1] = bfloat4(o1);
    }
    return;
  }

  const uint logical_head = tg_position.y;
  const uint local_tok = logical_head / params.query_heads;
  if (local_tok >= slot_tokens) {
    return;
  }
  const uint global_tok = r_off + local_tok;
  const uint query_head = logical_head - local_tok * params.query_heads;
  const uint kv_head = query_head / repeat;
  device const bfloat *query_row = query +
                                   ulong(global_tok) * params.query_token_stride +
                                   ulong(query_head) * params.query_head_stride;
  device const bfloat *gate_row = gate +
                                  ulong(global_tok) * params.gate_token_stride +
                                  ulong(query_head) * params.gate_head_stride;
  const ulong kv_width = ulong(params.key_value_heads) * params.head_dim;
  const uint context_tokens = positions[global_tok] + 1u;
  const float scale = rsqrt(float(params.head_dim));
  const uint logical_pages =
      (context_tokens + params.page_tokens - 1u) / params.page_tokens;

  if (params.head_dim == 256u) {
    device const bfloat4 *qv =
        reinterpret_cast<device const bfloat4 *>(query_row + lane * 8u);
    const float4 q0 = float4(qv[0]);
    const float4 q1 = float4(qv[1]);
    float maximum = -INFINITY;
    for (uint page = 0u; page < logical_pages; ++page) {
      const ulong page_base = ulong(page_table[page]) * params.page_tokens;
      const uint valid =
          min(params.page_tokens, context_tokens - page * params.page_tokens);
      device const bfloat *k_page =
          cache_key + page_base * kv_width + ulong(kv_head) * 256ul;
      uint within = 0u;
      for (; within + 3u < valid; within += 4u) {
        maximum = max(maximum, exact_dot_hd256(q0, q1, k_page + ulong(within + 0u) * kv_width, lane) * scale);
        maximum = max(maximum, exact_dot_hd256(q0, q1, k_page + ulong(within + 1u) * kv_width, lane) * scale);
        maximum = max(maximum, exact_dot_hd256(q0, q1, k_page + ulong(within + 2u) * kv_width, lane) * scale);
        maximum = max(maximum, exact_dot_hd256(q0, q1, k_page + ulong(within + 3u) * kv_width, lane) * scale);
      }
      for (; within < valid; ++within) {
        maximum = max(maximum, exact_dot_hd256(q0, q1, k_page + ulong(within) * kv_width, lane) * scale);
      }
    }
    float denominator = 0.0f;
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
    for (uint page = 0u; page < logical_pages; ++page) {
      const ulong page_base = ulong(page_table[page]) * params.page_tokens;
      const uint valid =
          min(params.page_tokens, context_tokens - page * params.page_tokens);
      device const bfloat *k_page =
          cache_key + page_base * kv_width + ulong(kv_head) * 256ul;
      device const bfloat *v_page =
          cache_value + page_base * kv_width + ulong(kv_head) * 256ul;
      uint within = 0u;
      for (; within + 3u < valid; within += 4u) {
        const float p0 = exp(exact_dot_hd256(q0, q1, k_page + ulong(within + 0u) * kv_width, lane) * scale - maximum);
        const float p1 = exp(exact_dot_hd256(q0, q1, k_page + ulong(within + 1u) * kv_width, lane) * scale - maximum);
        const float p2 = exp(exact_dot_hd256(q0, q1, k_page + ulong(within + 2u) * kv_width, lane) * scale - maximum);
        const float p3 = exp(exact_dot_hd256(q0, q1, k_page + ulong(within + 3u) * kv_width, lane) * scale - maximum);
        denominator += p0; exact_v_accum_hd256(p0, v_page + ulong(within + 0u) * kv_width, lane, acc0, acc1);
        denominator += p1; exact_v_accum_hd256(p1, v_page + ulong(within + 1u) * kv_width, lane, acc0, acc1);
        denominator += p2; exact_v_accum_hd256(p2, v_page + ulong(within + 2u) * kv_width, lane, acc0, acc1);
        denominator += p3; exact_v_accum_hd256(p3, v_page + ulong(within + 3u) * kv_width, lane, acc0, acc1);
      }
      for (; within < valid; ++within) {
        const float p0 = exp(exact_dot_hd256(q0, q1, k_page + ulong(within) * kv_width, lane) * scale - maximum);
        denominator += p0;
        exact_v_accum_hd256(p0, v_page + ulong(within) * kv_width, lane, acc0, acc1);
      }
    }
    float4 o0 = acc0 / denominator;
    float4 o1 = acc1 / denominator;
    if (params.gate_head_stride != 0u) {
      device const bfloat4 *gv =
          reinterpret_cast<device const bfloat4 *>(gate_row + lane * 8u);
      o0 /= (1.0f + exp(-float4(gv[0])));
      o1 /= (1.0f + exp(-float4(gv[1])));
    }
    device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(
        output + (ulong(global_tok) * params.query_heads + query_head) * 256ul +
        lane * 8u);
    ov[0] = bfloat4(o0);
    ov[1] = bfloat4(o1);
    return;
  }

  float4 q_lo, q_hi;
  exact_load_q_gen8(query_row, lane, params.head_dim, q_lo, q_hi);
  float maximum = -INFINITY;
  for (uint page = 0u; page < logical_pages; ++page) {
    const ulong page_base = ulong(page_table[page]) * params.page_tokens;
    const uint valid =
        min(params.page_tokens, context_tokens - page * params.page_tokens);
    for (uint within = 0u; within < valid; ++within) {
      const ulong position = page_base + within;
      device const bfloat *key_row =
          cache_key + position * kv_width + ulong(kv_head) * params.head_dim;
      maximum = max(maximum, exact_dot_gen8(q_lo, q_hi, key_row, lane, params.head_dim) * scale);
    }
  }
  float denominator = 0.0f;
  float4 acc_lo = float4(0.0f), acc_hi = float4(0.0f);
  for (uint page = 0u; page < logical_pages; ++page) {
    const ulong page_base = ulong(page_table[page]) * params.page_tokens;
    const uint valid =
        min(params.page_tokens, context_tokens - page * params.page_tokens);
    for (uint within = 0u; within < valid; ++within) {
      const ulong position = page_base + within;
      device const bfloat *key_row =
          cache_key + position * kv_width + ulong(kv_head) * params.head_dim;
      device const bfloat *value_row =
          cache_value + position * kv_width + ulong(kv_head) * params.head_dim;
      const float prob = exp(exact_dot_gen8(q_lo, q_hi, key_row, lane, params.head_dim) * scale - maximum);
      denominator += prob;
      exact_v_accum_gen8(prob, value_row, lane, params.head_dim, acc_lo, acc_hi);
    }
  }
  device bfloat *out_row =
      output + (ulong(global_tok) * params.query_heads + query_head) * params.head_dim;
  exact_store_gated_gen8(acc_lo, acc_hi, denominator, gate_row, out_row, lane,
                         params.head_dim, params.gate_head_stride != 0u);
}
