#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
#include "position_sampling_shared.h"

struct Conv {
  uint tokens, hidden, group_size, output_half;
};
kernel void omlx_m5_draft_conv(device const bfloat *x [[buffer(0)]],
                               device const bfloat *dynamic [[buffer(1)]],
                               device const bfloat *base [[buffer(2)]],
                               device bfloat *out [[buffer(3)]],
                               constant Conv &p [[buffer(4)]],
                               uint i [[thread_position_in_grid]]) {
  if (i >= p.tokens * p.hidden)
    return;
  uint t = i / p.hidden, f = i % p.hidden, groups = p.hidden / p.group_size;
  uint d = t * 4 * groups + p.output_half * 2 * groups + f / p.group_size;
  uint b = p.output_half * 2 * p.hidden + f;
  // Match the two public BF16 additions/multiplications of dynamic grouped
  // conv.
  bfloat w0 = bfloat(float(base[b]) + float(dynamic[d]));
  bfloat value = bfloat(float(w0) * float(x[i]));
  if (t) {
    bfloat w1 = bfloat(float(base[b + p.hidden]) + float(dynamic[d + groups]));
    value = bfloat(float(value) +
                   float(bfloat(float(w1) * float(x[i - p.hidden]))));
  }
  out[i] = value;
}

// Vectorized 2D bfloat4 grouped conv (e.g. 80 TGs across (10 chunks, 8 rows) on
// hidden=5120 so all 64 M5 Ultra GPU cores participate with zero integer div).
[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_draft_conv_vec4(device const bfloat *x [[buffer(0)]],
                                    device const bfloat *dynamic [[buffer(1)]],
                                    device const bfloat *base [[buffer(2)]],
                                    device bfloat *out [[buffer(3)]],
                                    constant Conv &p [[buffer(4)]],
                                    uint2 gid [[thread_position_in_grid]]) {
  const uint vec_cols = p.hidden >> 2u;
  const uint v = gid.x;
  const uint t = gid.y;
  if (t >= p.tokens || v >= vec_cols)
    return;
  const uint groups = p.hidden / p.group_size;
  const uint g =
      (p.group_size == 16u) ? (v >> 2u) : ((v << 2u) / p.group_size);
  const uint d = t * 4u * groups + p.output_half * 2u * groups + g;
  device const bfloat4 *x4 = reinterpret_cast<device const bfloat4 *>(x);
  device const bfloat4 *base4 = reinterpret_cast<device const bfloat4 *>(base);
  device bfloat4 *out4 = reinterpret_cast<device bfloat4 *>(out);
  const uint b_vec = (p.output_half * 2u * p.hidden) >> 2u;
  const float4 w0 = float4(bfloat4(float4(base4[b_vec + v]) + float(dynamic[d])));
  bfloat4 value = bfloat4(w0 * float4(x4[ulong(t) * vec_cols + v]));
  if (t > 0u) {
    const float4 w1 = float4(bfloat4(float4(base4[b_vec + vec_cols + v]) +
                                     float(dynamic[d + groups])));
    const float4 prev =
        float4(bfloat4(w1 * float4(x4[ulong(t - 1u) * vec_cols + v])));
    value = bfloat4(float4(value) + prev);
  }
  out4[ulong(t) * vec_cols + v] = value;
}

struct DraftRmsNormParams {
  uint hidden;
  uint tokens;
  float epsilon;
  float weight_offset;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_draft_rmsnorm(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant DraftRmsNormParams &p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint kThreads = 256u;
  const uint token = tg.y;
  if (token >= p.tokens)
    return;
  threadgroup float simd_sums[8];
  const ulong row_base = ulong(token) * p.hidden;
  device const bfloat *x_row = input + row_base;
  device bfloat *o_row = output + row_base;

  if (p.hidden == 5120u) {
    device const bfloat4 *xv = reinterpret_cast<device const bfloat4 *>(x_row);
    device const bfloat4 *wv = reinterpret_cast<device const bfloat4 *>(weight);
    device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(o_row);
    const bfloat4 rx0 = xv[tid + 0u * kThreads];
    const bfloat4 rx1 = xv[tid + 1u * kThreads];
    const bfloat4 rx2 = xv[tid + 2u * kThreads];
    const bfloat4 rx3 = xv[tid + 3u * kThreads];
    const bfloat4 rx4 = xv[tid + 4u * kThreads];
    float square_sum = 0.0f;
    const auto acc_sq = [&](bfloat4 x4) {
      const float4 xf = float4(x4);
      square_sum = fma(xf.x, xf.x, square_sum);
      square_sum = fma(xf.y, xf.y, square_sum);
      square_sum = fma(xf.z, xf.z, square_sum);
      square_sum = fma(xf.w, xf.w, square_sum);
    };
    acc_sq(rx0);
    acc_sq(rx1);
    acc_sq(rx2);
    acc_sq(rx3);
    acc_sq(rx4);
    square_sum = simd_sum(square_sum);
    if (lane == 0u)
      simd_sums[sg] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        (((simd_sums[0] + simd_sums[1]) + (simd_sums[2] + simd_sums[3])) +
         ((simd_sums[4] + simd_sums[5]) + (simd_sums[6] + simd_sums[7])));
    const float inv = rsqrt(total * (1.0f / 5120.0f) + p.epsilon);
    const float4 off4 = float4(p.weight_offset);
    ov[tid + 0u * kThreads] =
        bfloat4(float4(rx0) * inv * (float4(wv[tid + 0u * kThreads]) + off4));
    ov[tid + 1u * kThreads] =
        bfloat4(float4(rx1) * inv * (float4(wv[tid + 1u * kThreads]) + off4));
    ov[tid + 2u * kThreads] =
        bfloat4(float4(rx2) * inv * (float4(wv[tid + 2u * kThreads]) + off4));
    ov[tid + 3u * kThreads] =
        bfloat4(float4(rx3) * inv * (float4(wv[tid + 3u * kThreads]) + off4));
    ov[tid + 4u * kThreads] =
        bfloat4(float4(rx4) * inv * (float4(wv[tid + 4u * kThreads]) + off4));
    return;
  }

  float square_sum = 0.0f;
  for (uint f = tid; f < p.hidden; f += kThreads) {
    const float val = float(x_row[f]);
    square_sum = fma(val, val, square_sum);
  }
  square_sum = simd_sum(square_sum);
  if (lane == 0u)
    simd_sums[sg] = square_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total =
      (((simd_sums[0] + simd_sums[1]) + (simd_sums[2] + simd_sums[3])) +
       ((simd_sums[4] + simd_sums[5]) + (simd_sums[6] + simd_sums[7])));
  const float inv = rsqrt(total / float(p.hidden) + p.epsilon);
  for (uint f = tid; f < p.hidden; f += kThreads) {
    o_row[f] =
        bfloat(float(x_row[f]) * inv * (float(weight[f]) + p.weight_offset));
  }
}

struct DraftConvAddNormParams {
  uint tokens;
  uint hidden;
  uint group_size;
  float epsilon;
};

__attribute__((always_inline)) inline bfloat4 draft_conv_add_norm_step(
    uint v, uint t, uint dyn_base0, uint dyn_base1,
    device const bfloat *dynamic, device const bfloat4 *b0v,
    device const bfloat4 *b1v, device const bfloat4 *pv_cur,
    device const bfloat4 *pv_prev, device bfloat4 *hv,
    thread float &square_sum) {
  const uint g = v >> 2u;
  const float4 w0 =
      float4(bfloat4(float4(b0v[v]) + float(dynamic[dyn_base0 + g])));
  bfloat4 delta4 = bfloat4(w0 * float4(pv_cur[v]));
  if (t > 0u) {
    const float4 w1 =
        float4(bfloat4(float4(b1v[v]) + float(dynamic[dyn_base1 + g])));
    const float4 prev4 = float4(bfloat4(w1 * float4(pv_prev[v])));
    delta4 = bfloat4(float4(delta4) + prev4);
  }
  const bfloat4 s4 = bfloat4(float4(hv[v]) + float4(delta4));
  hv[v] = s4;
  const float4 sf = float4(s4);
  square_sum = fma(sf.x, sf.x, square_sum);
  square_sum = fma(sf.y, sf.y, square_sum);
  square_sum = fma(sf.z, sf.z, square_sum);
  square_sum = fma(sf.w, sf.w, square_sum);
  return s4;
}

// Fused draft_conv (output_half=1) + residual add + RMSNorm across 8 unrolled
// warps/TG (256 threads/TG, zero stack spill, eliminates delta global traffic
// and 10 dispatches + 10 barriers across 5 layers).
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_draft_conv_add_rmsnorm(
    device const bfloat *projected [[buffer(0)]],
    device const bfloat *dynamic [[buffer(1)]],
    device const bfloat *base [[buffer(2)]],
    device bfloat *hidden_inout [[buffer(3)]],
    device const bfloat *norm_weight [[buffer(4)]],
    device bfloat *norm_out [[buffer(5)]],
    constant DraftConvAddNormParams &p [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint kThreads = 256u;
  const uint t = tg.y;
  if (t >= p.tokens)
    return;
  threadgroup float simd_sums[8];
  const ulong row_base = ulong(t) * p.hidden;

  if (p.hidden == 5120u && p.group_size == 16u) {
    constexpr uint kGroups = 320u;
    device const bfloat4 *pv_cur =
        reinterpret_cast<device const bfloat4 *>(projected + row_base);
    device const bfloat4 *pv_prev =
        t > 0u ? reinterpret_cast<device const bfloat4 *>(
                     projected + row_base - 5120ul)
               : nullptr;
    device const bfloat4 *b0v =
        reinterpret_cast<device const bfloat4 *>(base + 2ul * 5120ul);
    device const bfloat4 *b1v =
        reinterpret_cast<device const bfloat4 *>(base + 3ul * 5120ul);
    device bfloat4 *hv =
        reinterpret_cast<device bfloat4 *>(hidden_inout + row_base);
    device const bfloat4 *wv =
        reinterpret_cast<device const bfloat4 *>(norm_weight);
    device bfloat4 *ov =
        reinterpret_cast<device bfloat4 *>(norm_out + row_base);
    const uint dyn_base0 = t * (4u * kGroups) + 2u * kGroups;
    const uint dyn_base1 = dyn_base0 + kGroups;

    float square_sum = 0.0f;
    const bfloat4 rs0 = draft_conv_add_norm_step(
        tid + 0u * kThreads, t, dyn_base0, dyn_base1, dynamic, b0v, b1v,
        pv_cur, pv_prev, hv, square_sum);
    const bfloat4 rs1 = draft_conv_add_norm_step(
        tid + 1u * kThreads, t, dyn_base0, dyn_base1, dynamic, b0v, b1v,
        pv_cur, pv_prev, hv, square_sum);
    const bfloat4 rs2 = draft_conv_add_norm_step(
        tid + 2u * kThreads, t, dyn_base0, dyn_base1, dynamic, b0v, b1v,
        pv_cur, pv_prev, hv, square_sum);
    const bfloat4 rs3 = draft_conv_add_norm_step(
        tid + 3u * kThreads, t, dyn_base0, dyn_base1, dynamic, b0v, b1v,
        pv_cur, pv_prev, hv, square_sum);
    const bfloat4 rs4 = draft_conv_add_norm_step(
        tid + 4u * kThreads, t, dyn_base0, dyn_base1, dynamic, b0v, b1v,
        pv_cur, pv_prev, hv, square_sum);
    square_sum = simd_sum(square_sum);
    if (lane == 0u)
      simd_sums[sg] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        (((simd_sums[0] + simd_sums[1]) + (simd_sums[2] + simd_sums[3])) +
         ((simd_sums[4] + simd_sums[5]) + (simd_sums[6] + simd_sums[7])));
    const float inv = rsqrt(total * (1.0f / 5120.0f) + p.epsilon);
    ov[tid + 0u * kThreads] =
        bfloat4(float4(rs0) * inv * float4(wv[tid + 0u * kThreads]));
    ov[tid + 1u * kThreads] =
        bfloat4(float4(rs1) * inv * float4(wv[tid + 1u * kThreads]));
    ov[tid + 2u * kThreads] =
        bfloat4(float4(rs2) * inv * float4(wv[tid + 2u * kThreads]));
    ov[tid + 3u * kThreads] =
        bfloat4(float4(rs3) * inv * float4(wv[tid + 3u * kThreads]));
    ov[tid + 4u * kThreads] =
        bfloat4(float4(rs4) * inv * float4(wv[tid + 4u * kThreads]));
    return;
  }

  const uint groups = p.hidden / p.group_size;
  const uint dyn_base = t * 4u * groups + 2u * groups;
  const uint b_off = 2u * p.hidden;
  device bfloat *h_row = hidden_inout + row_base;
  device bfloat *o_row = norm_out + row_base;
  float square_sum = 0.0f;
  for (uint f = tid; f < p.hidden; f += kThreads) {
    const uint g = f / p.group_size;
    const bfloat w0 = bfloat(float(base[b_off + f]) + float(dynamic[dyn_base + g]));
    bfloat delta = bfloat(float(w0) * float(projected[row_base + f]));
    if (t > 0u) {
      const bfloat w1 = bfloat(float(base[b_off + p.hidden + f]) +
                               float(dynamic[dyn_base + groups + g]));
      delta = bfloat(
          float(delta) +
          float(bfloat(float(w1) * float(projected[row_base - p.hidden + f]))));
    }
    const bfloat s = bfloat(float(h_row[f]) + float(delta));
    h_row[f] = s;
    const float sf = float(s);
    square_sum = fma(sf, sf, square_sum);
  }
  square_sum = simd_sum(square_sum);
  if (lane == 0u)
    simd_sums[sg] = square_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total =
      (((simd_sums[0] + simd_sums[1]) + (simd_sums[2] + simd_sums[3])) +
       ((simd_sums[4] + simd_sums[5]) + (simd_sums[6] + simd_sums[7])));
  const float inv = rsqrt(total / float(p.hidden) + p.epsilon);
  for (uint f = tid; f < p.hidden; f += kThreads) {
    o_row[f] = bfloat(float(h_row[f]) * inv * float(norm_weight[f]));
  }
}

struct PackQkvParams {
  uint q_rows, kv_rows, cols;
};

kernel void omlx_m5_draft_pack_qkv_bf16(
    device const bfloat *wq [[buffer(0)]],
    device const bfloat *wk [[buffer(1)]],
    device const bfloat *wv [[buffer(2)]],
    device bfloat *dst [[buffer(3)]],
    constant PackQkvParams &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint total_rows = p.q_rows + 2u * p.kv_rows;
  const uint row = gid.y;
  if (row >= total_rows)
    return;
  if ((p.cols & 7u) == 0u) {
    const uint vec_cols = p.cols >> 3u;
    const uint v = gid.x;
    if (v >= vec_cols)
      return;
    device const uint4 *src4 = nullptr;
    if (row < p.q_rows) {
      src4 = reinterpret_cast<device const uint4 *>(wq + ulong(row) * p.cols);
    } else if (row < p.q_rows + p.kv_rows) {
      src4 = reinterpret_cast<device const uint4 *>(
          wk + ulong(row - p.q_rows) * p.cols);
    } else {
      src4 = reinterpret_cast<device const uint4 *>(
          wv + ulong(row - p.q_rows - p.kv_rows) * p.cols);
    }
    device uint4 *dst4 =
        reinterpret_cast<device uint4 *>(dst + ulong(row) * p.cols);
    dst4[v] = src4[v];
    return;
  }
  const uint col = gid.x;
  if (col >= p.cols)
    return;
  if (row < p.q_rows) {
    dst[ulong(row) * p.cols + col] = wq[ulong(row) * p.cols + col];
  } else if (row < p.q_rows + p.kv_rows) {
    dst[ulong(row) * p.cols + col] = wk[ulong(row - p.q_rows) * p.cols + col];
  } else {
    dst[ulong(row) * p.cols + col] =
        wv[ulong(row - p.q_rows - p.kv_rows) * p.cols + col];
  }
}

struct Shape {
  uint tokens, width;
};
kernel void omlx_m5_draft_residual(device bfloat *hidden [[buffer(0)]],
                                   device const bfloat *value [[buffer(1)]],
                                   constant Shape &p [[buffer(2)]],
                                   uint i [[thread_position_in_grid]]) {
  if (i < p.tokens * p.width)
    hidden[i] = bfloat(float(hidden[i]) + float(value[i]));
}

struct Rope {
  uint tokens, heads, dim, position;
  float epsilon, theta;
};
kernel void omlx_m5_draft_rope(device bfloat *x [[buffer(0)]],
                               device const bfloat *weight [[buffer(1)]],
                               constant Rope &p [[buffer(2)]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint3 group [[threadgroup_position_in_grid]]) {
  uint t = group.y / p.heads;
  if (t >= p.tokens)
    return;
  ulong base = ulong(group.y) * p.dim;
  uint half_dim = p.dim / 2;
  float square = 0;
  for (uint f = lane; f < p.dim; f += 32)
    square = fma(float(x[base + f]), float(x[base + f]), square);
  float inverse = rsqrt(simd_sum(square) / float(p.dim) + p.epsilon);
  // All source reads complete before overwriting either half_dim of a pair.
  for (uint pair = lane; pair < half_dim; pair += 32) {
    float left =
        float(bfloat(float(x[base + pair]) * inverse * float(weight[pair])));
    float right = float(bfloat(float(x[base + half_dim + pair]) * inverse *
                               float(weight[half_dim + pair])));
    float angle =
        float(p.position + t) * pow(p.theta, -float(pair) / float(half_dim));
    x[base + pair] = bfloat(left * cos(angle) - right * sin(angle));
    x[base + half_dim + pair] = bfloat(left * sin(angle) + right * cos(angle));
  }
}

struct QkRopeStridedParams {
  uint tokens;
  uint q_heads;
  uint kv_heads;
  uint dim;
  uint qkv_stride;
  uint position;
  float epsilon;
  float theta;
};

// Fused Q + K RMSNorm + RoPE in-place over interleaved [tokens, q + 2*kv] QKV
// output (80 TGs of 4 warps = 320 warps across all 64 M5 Ultra GPU cores).
[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_draft_qk_rope_strided(
    device bfloat *qkv [[buffer(0)]],
    device const bfloat *q_weight [[buffer(1)]],
    device const bfloat *k_weight [[buffer(2)]],
    constant QkRopeStridedParams &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint total_heads = p.q_heads + p.kv_heads;
  const uint item = tg.y * 4u + sg;
  const uint t = item / total_heads;
  if (t >= p.tokens)
    return;
  const uint h_idx = item % total_heads;
  const bool is_q = h_idx < p.q_heads;
  const ulong base =
      ulong(t) * p.qkv_stride +
      (is_q ? ulong(h_idx) * p.dim
            : ulong(p.q_heads) * p.dim + ulong(h_idx - p.q_heads) * p.dim);
  device const bfloat *weight = is_q ? q_weight : k_weight;
  const uint half_dim = p.dim >> 1u;

  float square = 0.0f;
  for (uint f = lane; f < p.dim; f += 32u)
    square = fma(float(qkv[base + f]), float(qkv[base + f]), square);
  const float inverse = rsqrt(simd_sum(square) / float(p.dim) + p.epsilon);
  const float pos_f = float(p.position + t);
  for (uint pair = lane; pair < half_dim; pair += 32u) {
    const float left =
        float(bfloat(float(qkv[base + pair]) * inverse * float(weight[pair])));
    const float right = float(bfloat(float(qkv[base + half_dim + pair]) *
                                     inverse * float(weight[half_dim + pair])));
    const float angle = pos_f * pow(p.theta, -float(pair) / float(half_dim));
    const float c = cos(angle);
    const float s = sin(angle);
    qkv[base + pair] = bfloat(left * c - right * s);
    qkv[base + half_dim + pair] = bfloat(left * s + right * c);
  }
}

struct ContextRopeAppendParams {
  uint tokens;
  uint kv_heads;
  uint dim;
  uint position;
  uint capacity;
  float epsilon;
  float theta;
  uint pad;
};

// Fused context K RMSNorm + RoPE + KV ring-cache append directly from
// [tokens, 2*kv] fused KV projection output.
[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_draft_context_rope_cache_append(
    device const bfloat *context_kv [[buffer(0)]],
    device const bfloat *k_weight [[buffer(1)]],
    device bfloat *cache_keys [[buffer(2)]],
    device bfloat *cache_values [[buffer(3)]],
    constant ContextRopeAppendParams &p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  if (p.capacity == 0u)
    return;
  const uint t = group.y / p.kv_heads;
  if (t >= p.tokens)
    return;
  const uint head = group.y % p.kv_heads;
  const ulong kv_width = ulong(p.kv_heads) * p.dim;
  const ulong src_k = ulong(t) * (2ul * kv_width) + ulong(head) * p.dim;
  const ulong src_v = src_k + kv_width;
  const uint dst_row = (p.position + t) % p.capacity;
  const ulong dst_off = ulong(dst_row) * kv_width + ulong(head) * p.dim;
  const uint half_dim = p.dim >> 1u;

  float square = 0.0f;
  for (uint f = lane; f < p.dim; f += 32u) {
    const float kv_f = float(context_kv[src_k + f]);
    square = fma(kv_f, kv_f, square);
  }
  const float inverse = rsqrt(simd_sum(square) / float(p.dim) + p.epsilon);
  const float pos_f = float(p.position + t);
  for (uint pair = lane; pair < half_dim; pair += 32u) {
    const float left = float(bfloat(float(context_kv[src_k + pair]) * inverse *
                                    float(k_weight[pair])));
    const float right =
        float(bfloat(float(context_kv[src_k + half_dim + pair]) * inverse *
                     float(k_weight[half_dim + pair])));
    const float angle = pos_f * pow(p.theta, -float(pair) / float(half_dim));
    const float c = cos(angle);
    const float s = sin(angle);
    cache_keys[dst_off + pair] = bfloat(left * c - right * s);
    cache_keys[dst_off + half_dim + pair] = bfloat(left * s + right * c);
  }
  for (uint f = lane; f < p.dim; f += 32u) {
    cache_values[dst_off + f] = context_kv[src_v + f];
  }
}

struct Attention {
  uint tokens, context, query_heads, kv_heads, dim, window, capacity;
  uint q_stride, nk_stride;
};

inline void draft_online_softmax_weights(float score, thread float &maximum,
                                         thread float &old_scale,
                                         thread float &weight) {
  if (score > maximum) {
    old_scale = (maximum == -INFINITY || score == INFINITY)
                    ? 0.0f
                    : exp(maximum - score);
    weight = 1.0f;
    maximum = score;
  } else {
    old_scale = (maximum == -INFINITY) ? 0.0f : 1.0f;
    weight = (maximum == -INFINITY)
                 ? 0.0f
                 : (maximum == INFINITY ? (score == INFINITY ? 1.0f : 0.0f)
                                        : exp(score - maximum));
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_draft_attention(device const bfloat *q [[buffer(0)]],
                                    device const bfloat *nk [[buffer(1)]],
                                    device const bfloat *nv [[buffer(2)]],
                                    device const bfloat *ck [[buffer(3)]],
                                    device const bfloat *cv [[buffer(4)]],
                                    device bfloat *output [[buffer(5)]],
                                    constant Attention &p [[buffer(6)]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg_id [[simdgroup_index_in_threadgroup]],
                                    uint num_sgs [[simdgroups_per_threadgroup]],
                                    uint3 group
                                    [[threadgroup_position_in_grid]]) {
  threadgroup float tg_max[8];
  threadgroup float tg_den[8];
  threadgroup float4 tg_acc0[8 * 32];
  threadgroup float4 tg_acc1[8 * 32];

  uint t = group.y / p.query_heads, head = group.y % p.query_heads;
  if (t >= p.tokens)
    return;
  uint kv = head / (p.query_heads / p.kv_heads);
  ulong qw = ulong(p.query_heads) * p.dim, kw = ulong(p.kv_heads) * p.dim;
  const ulong q_row_stride = p.q_stride ? ulong(p.q_stride) : qw;
  const ulong nk_row_stride = p.nk_stride ? ulong(p.nk_stride) : kw;
  const float scale = rsqrt(float(p.dim));
  const uint first = min(
      p.context, p.context + t >= p.window ? p.context + t - p.window + 1 : 0);
  const uint end_idx = p.context + p.tokens;
  const uint total_keys = (end_idx > first) ? (end_idx - first) : 0u;
  const uint active_sgs =
      (num_sgs > 1u && total_keys > 64u && (p.dim == 128u || p.dim == 256u))
          ? min(num_sgs, 8u)
          : 1u;
  const ulong q_src_offset = ulong(t) * q_row_stride + ulong(head) * p.dim;
  const ulong out_offset = ulong(t) * qw + ulong(head) * p.dim;

  if (p.dim == 256u) {
    if (active_sgs == 1u && sg_id > 0u)
      return;
    const uint chunk = (total_keys + active_sgs - 1u) / active_sgs;
    const uint c_beg =
        (sg_id < active_sgs) ? min(end_idx, first + sg_id * chunk) : end_idx;
    const uint c_end =
        (sg_id < active_sgs) ? min(end_idx, c_beg + chunk) : end_idx;
    device const bfloat4 *qv =
        reinterpret_cast<device const bfloat4 *>(q + q_src_offset + lane * 8u);
    const float4 q0 = float4(qv[0]);
    const float4 q1 = float4(qv[1]);
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint index = c_beg; index < c_end; ++index) {
      const bool noise = index >= p.context;
      const uint row =
          noise ? index - p.context : (p.capacity ? index % p.capacity : 0u);
      const ulong k_stride = noise ? nk_row_stride : kw;
      device const bfloat4 *key4 = reinterpret_cast<device const bfloat4 *>(
          (noise ? nk : ck) + ulong(row) * k_stride + ulong(kv) * 256u +
          lane * 8u);
      const bfloat4 k0 = key4[0];
      const bfloat4 k1 = key4[1];
      float score = fma(q0.x, float(k0.x), 0.0f);
      score = fma(q0.y, float(k0.y), score);
      score = fma(q0.z, float(k0.z), score);
      score = fma(q0.w, float(k0.w), score);
      score = fma(q1.x, float(k1.x), score);
      score = fma(q1.y, float(k1.y), score);
      score = fma(q1.z, float(k1.z), score);
      score = fma(q1.w, float(k1.w), score);
      score = simd_sum(score) * scale;
      if (!isnan(score)) {
        float old_scale = 0.0f, weight = 0.0f;
        draft_online_softmax_weights(score, maximum, old_scale, weight);
        denominator = denominator * old_scale + weight;
        device const bfloat4 *val4 = reinterpret_cast<device const bfloat4 *>(
            (noise ? nv : cv) + ulong(row) * k_stride + ulong(kv) * 256u +
            lane * 8u);
        acc0 = acc0 * old_scale + weight * float4(val4[0]);
        acc1 = acc1 * old_scale + weight * float4(val4[1]);
      }
    }
    if (active_sgs > 1u) {
      if (sg_id < active_sgs) {
        if (lane == 0u) {
          tg_max[sg_id] = maximum;
          tg_den[sg_id] = denominator;
        }
        tg_acc0[sg_id * 32u + lane] = acc0;
        tg_acc1[sg_id * 32u + lane] = acc1;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg_id > 0u)
        return;
      for (uint s = 1u; s < active_sgs; ++s) {
        const float d_s = tg_den[s];
        if (d_s <= 0.0f)
          continue;
        const float m_s = tg_max[s];
        const float4 a0_s = tg_acc0[s * 32u + lane];
        const float4 a1_s = tg_acc1[s * 32u + lane];
        if (denominator <= 0.0f) {
          maximum = m_s;
          denominator = d_s;
          acc0 = a0_s;
          acc1 = a1_s;
        } else {
          const float m_new = max(maximum, m_s);
          const float sc_cur =
              (maximum == m_new) ? 1.0f : exp(maximum - m_new);
          const float sc_s = (m_s == m_new) ? 1.0f : exp(m_s - m_new);
          denominator = denominator * sc_cur + d_s * sc_s;
          acc0 = acc0 * sc_cur + a0_s * sc_s;
          acc1 = acc1 * sc_cur + a1_s * sc_s;
          maximum = m_new;
        }
      }
    }
    device bfloat4 *ov =
        reinterpret_cast<device bfloat4 *>(output + out_offset + lane * 8u);
    if (denominator > 0.0f) {
      const float inv_d = 1.0f / denominator;
      ov[0] = bfloat4(acc0 * inv_d);
      ov[1] = bfloat4(acc1 * inv_d);
    } else {
      ov[0] = bfloat4(bfloat(0.0f));
      ov[1] = bfloat4(bfloat(0.0f));
    }
    return;
  }

  if (p.dim == 128u) {
    if (active_sgs == 1u && sg_id > 0u)
      return;
    const uint chunk = (total_keys + active_sgs - 1u) / active_sgs;
    const uint c_beg =
        (sg_id < active_sgs) ? min(end_idx, first + sg_id * chunk) : end_idx;
    const uint c_end =
        (sg_id < active_sgs) ? min(end_idx, c_beg + chunk) : end_idx;
    device const bfloat4 *qv =
        reinterpret_cast<device const bfloat4 *>(q + q_src_offset + lane * 4u);
    const float4 q0 = float4(qv[0]);
    float4 acc0 = float4(0.0f);
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint index = c_beg; index < c_end; ++index) {
      const bool noise = index >= p.context;
      const uint row =
          noise ? index - p.context : (p.capacity ? index % p.capacity : 0u);
      const ulong k_stride = noise ? nk_row_stride : kw;
      device const bfloat4 *key4 = reinterpret_cast<device const bfloat4 *>(
          (noise ? nk : ck) + ulong(row) * k_stride + ulong(kv) * 128u +
          lane * 4u);
      const bfloat4 k0 = key4[0];
      float score = fma(q0.x, float(k0.x), 0.0f);
      score = fma(q0.y, float(k0.y), score);
      score = fma(q0.z, float(k0.z), score);
      score = fma(q0.w, float(k0.w), score);
      score = simd_sum(score) * scale;
      if (!isnan(score)) {
        float old_scale = 0.0f, weight = 0.0f;
        draft_online_softmax_weights(score, maximum, old_scale, weight);
        denominator = denominator * old_scale + weight;
        device const bfloat4 *val4 = reinterpret_cast<device const bfloat4 *>(
            (noise ? nv : cv) + ulong(row) * k_stride + ulong(kv) * 128u +
            lane * 4u);
        acc0 = acc0 * old_scale + weight * float4(val4[0]);
      }
    }
    if (active_sgs > 1u) {
      if (sg_id < active_sgs) {
        if (lane == 0u) {
          tg_max[sg_id] = maximum;
          tg_den[sg_id] = denominator;
        }
        tg_acc0[sg_id * 32u + lane] = acc0;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg_id > 0u)
        return;
      for (uint s = 1u; s < active_sgs; ++s) {
        const float d_s = tg_den[s];
        if (d_s <= 0.0f)
          continue;
        const float m_s = tg_max[s];
        const float4 a0_s = tg_acc0[s * 32u + lane];
        if (denominator <= 0.0f) {
          maximum = m_s;
          denominator = d_s;
          acc0 = a0_s;
        } else {
          const float m_new = max(maximum, m_s);
          const float sc_cur =
              (maximum == m_new) ? 1.0f : exp(maximum - m_new);
          const float sc_s = (m_s == m_new) ? 1.0f : exp(m_s - m_new);
          denominator = denominator * sc_cur + d_s * sc_s;
          acc0 = acc0 * sc_cur + a0_s * sc_s;
          maximum = m_new;
        }
      }
    }
    device bfloat4 *ov =
        reinterpret_cast<device bfloat4 *>(output + out_offset + lane * 4u);
    if (denominator > 0.0f) {
      ov[0] = bfloat4(acc0 / denominator);
    } else {
      ov[0] = bfloat4(bfloat(0.0f));
    }
    return;
  }

  if (sg_id > 0u)
    return;

  const auto load_q = [&](uint idx) -> float {
    const uint f = lane + idx * 32u;
    return f < p.dim ? float(q[q_src_offset + f]) : 0.0f;
  };
  const float4 qr0 = float4(load_q(0u), load_q(1u), load_q(2u), load_q(3u));
  const float4 qr1 = float4(load_q(4u), load_q(5u), load_q(6u), load_q(7u));
  float4 ac0 = float4(0.0f);
  float4 ac1 = float4(0.0f);
  float maximum = -INFINITY, denominator = 0.0f;
  for (uint index = first; index < p.context + p.tokens; ++index) {
    bool noise = index >= p.context;
    uint row =
        noise ? index - p.context : (p.capacity ? index % p.capacity : 0u);
    const ulong k_stride = noise ? nk_row_stride : kw;
    device const bfloat *key =
        (noise ? nk : ck) + ulong(row) * k_stride + ulong(kv) * p.dim;
    device const bfloat *value =
        (noise ? nv : cv) + ulong(row) * k_stride + ulong(kv) * p.dim;
    float score = 0.0f;
    if (lane + 0u * 32u < p.dim) score = fma(qr0.x, float(key[lane + 0u * 32u]), score);
    if (lane + 1u * 32u < p.dim) score = fma(qr0.y, float(key[lane + 1u * 32u]), score);
    if (lane + 2u * 32u < p.dim) score = fma(qr0.z, float(key[lane + 2u * 32u]), score);
    if (lane + 3u * 32u < p.dim) score = fma(qr0.w, float(key[lane + 3u * 32u]), score);
    if (lane + 4u * 32u < p.dim) score = fma(qr1.x, float(key[lane + 4u * 32u]), score);
    if (lane + 5u * 32u < p.dim) score = fma(qr1.y, float(key[lane + 5u * 32u]), score);
    if (lane + 6u * 32u < p.dim) score = fma(qr1.z, float(key[lane + 6u * 32u]), score);
    if (lane + 7u * 32u < p.dim) score = fma(qr1.w, float(key[lane + 7u * 32u]), score);
    score = simd_sum(score) * scale;
    if (!isnan(score)) {
      float old = 0.0f, weight = 0.0f;
      draft_online_softmax_weights(score, maximum, old, weight);
      denominator = denominator * old + weight;
      const auto lv = [&](uint idx) -> float {
        const uint f = lane + idx * 32u;
        return f < p.dim ? float(value[f]) : 0.0f;
      };
      ac0 = ac0 * old + weight * float4(lv(0u), lv(1u), lv(2u), lv(3u));
      ac1 = ac1 * old + weight * float4(lv(4u), lv(5u), lv(6u), lv(7u));
    }
  }
  const auto store_o = [&](uint idx, float val) {
    const uint f = lane + idx * 32u;
    if (f < p.dim)
      output[out_offset + f] =
          denominator > 0.0f ? bfloat(val / denominator) : bfloat(0.0f);
  };
  store_o(0u, ac0.x);
  store_o(1u, ac0.y);
  store_o(2u, ac0.z);
  store_o(3u, ac0.w);
  store_o(4u, ac1.x);
  store_o(5u, ac1.y);
  store_o(6u, ac1.z);
  store_o(7u, ac1.w);
}

// Target layer features remain on the GPU. Only newly committed rows are
// copied into the temporary FC input, then projected into persistent caches.
// When ring_rows > 0, target feature_history is a cyclic ring buffer indexed
// modulo ring_rows while p.position remains the true logical token position.
struct ContextGather {
  uint tokens, width, position, ring_rows;
};
kernel void omlx_m5_draft_context_gather(device const bfloat *source
                                         [[buffer(0)]],
                                         device bfloat *out [[buffer(1)]],
                                         constant ContextGather &p
                                         [[buffer(2)]],
                                         uint i [[thread_position_in_grid]]) {
  if (ulong(i) >= ulong(p.tokens) * p.width)
    return;
  const uint t = i / p.width;
  const uint col = i % p.width;
  const uint src_row =
      p.ring_rows ? ((p.position + t) % p.ring_rows) : (p.position + t);
  out[i] = source[ulong(src_row) * p.width + col];
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_draft_context_gather_vec8(
    device const bfloat *source [[buffer(0)]],
    device bfloat *out [[buffer(1)]],
    constant ContextGather &p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint vec_width = p.width >> 3u;
  const uint col8 = gid.x;
  const uint t = gid.y;
  if (t >= p.tokens || col8 >= vec_width)
    return;
  const uint src_row =
      p.ring_rows ? ((p.position + t) % p.ring_rows) : (p.position + t);
  device const uint4 *src4 =
      reinterpret_cast<device const uint4 *>(source + ulong(src_row) * p.width);
  device uint4 *dst4 =
      reinterpret_cast<device uint4 *>(out + ulong(t) * p.width);
  dst4[col8] = src4[col8];
}

struct ContextRows {
  uint tokens, width, position, capacity;
};
kernel void omlx_m5_draft_cache_append(device const bfloat *keys [[buffer(0)]],
                                       device const bfloat *values
                                       [[buffer(1)]],
                                       device bfloat *cache_keys [[buffer(2)]],
                                       device bfloat *cache_values
                                       [[buffer(3)]],
                                       constant ContextRows &p [[buffer(4)]],
                                       uint i [[thread_position_in_grid]]) {
  if (p.capacity == 0 || ulong(i) >= ulong(p.tokens) * p.width)
    return;
  uint row = (p.position + i / p.width) % p.capacity;
  ulong destination = ulong(row) * p.width + i % p.width;
  cache_keys[destination] = keys[i];
  cache_values[destination] = values[i];
}

// P03: the target shared head writes BF16 logits. Keep the full vocabulary on
// the GPU; only the selected K candidates leave the device.
struct SelectorParams {
  uint vocab, hidden, rank, top_k, blocks, groups, width, anchor, position;
  float temperature;
  float uniforms[7];
};
__attribute__((always_inline)) inline bool selector_better(float score, uint id,
                                                           float other,
                                                           uint other_id) {
  return score > other || (score == other && id < other_id);
}

__attribute__((always_inline)) inline void
selector_insert_top4(float v, uint id, thread float &v0, thread uint &i0,
                     thread float &v1, thread uint &i1, thread float &v2,
                     thread uint &i2, thread float &v3, thread uint &i3) {
  if (selector_better(v, id, v3, i3)) {
    if (selector_better(v, id, v2, i2)) {
      v3 = v2;
      i3 = i2;
      if (selector_better(v, id, v1, i1)) {
        v2 = v1;
        i2 = i1;
        if (selector_better(v, id, v0, i0)) {
          v1 = v0;
          i1 = i0;
          v0 = v;
          i0 = id;
        } else {
          v1 = v;
          i1 = id;
        }
      } else {
        v2 = v;
        i2 = id;
      }
    } else {
      v3 = v;
      i3 = id;
    }
  }
}

__attribute__((always_inline)) inline void selector_stage1_scan_thread(
    device const bfloat *row_logits, uint vocab, uint chunk, uint tid,
    bool has_upper_bound, float bound_v, uint bound_i, thread float &v0,
    thread uint &i0, thread float &v1, thread uint &i1, thread float &v2,
    thread uint &i2, thread float &v3, thread uint &i3) {
  v0 = -INFINITY;
  v1 = -INFINITY;
  v2 = -INFINITY;
  v3 = -INFINITY;
  i0 = 0xffffffffu;
  i1 = 0xffffffffu;
  i2 = 0xffffffffu;
  i3 = 0xffffffffu;
  const uint vocab4 = vocab >> 2u;
  const bool aligned4 = (reinterpret_cast<ulong>(row_logits) & 7ul) == 0ul;
  const uint start_vec = chunk * 256u + tid;
  if (aligned4) {
    device const bfloat4 *row_logits4 =
        reinterpret_cast<device const bfloat4 *>(row_logits);
    for (uint vec = start_vec; vec < vocab4; vec += 64u * 256u) {
      const float4 fv = float4(row_logits4[vec]);
      const uint base = vec << 2u;
      if (!has_upper_bound || selector_better(bound_v, bound_i, fv.x, base + 0u))
        selector_insert_top4(fv.x, base + 0u, v0, i0, v1, i1, v2, i2, v3, i3);
      if (!has_upper_bound || selector_better(bound_v, bound_i, fv.y, base + 1u))
        selector_insert_top4(fv.y, base + 1u, v0, i0, v1, i1, v2, i2, v3, i3);
      if (!has_upper_bound || selector_better(bound_v, bound_i, fv.z, base + 2u))
        selector_insert_top4(fv.z, base + 2u, v0, i0, v1, i1, v2, i2, v3, i3);
      if (!has_upper_bound || selector_better(bound_v, bound_i, fv.w, base + 3u))
        selector_insert_top4(fv.w, base + 3u, v0, i0, v1, i1, v2, i2, v3, i3);
    }
  } else {
    for (uint vec = start_vec; vec < vocab4; vec += 64u * 256u) {
      const uint base = vec << 2u;
      const float vx = float(row_logits[base + 0u]);
      const float vy = float(row_logits[base + 1u]);
      const float vz = float(row_logits[base + 2u]);
      const float vw = float(row_logits[base + 3u]);
      if (!has_upper_bound || selector_better(bound_v, bound_i, vx, base + 0u))
        selector_insert_top4(vx, base + 0u, v0, i0, v1, i1, v2, i2, v3, i3);
      if (!has_upper_bound || selector_better(bound_v, bound_i, vy, base + 1u))
        selector_insert_top4(vy, base + 1u, v0, i0, v1, i1, v2, i2, v3, i3);
      if (!has_upper_bound || selector_better(bound_v, bound_i, vz, base + 2u))
        selector_insert_top4(vz, base + 2u, v0, i0, v1, i1, v2, i2, v3, i3);
      if (!has_upper_bound || selector_better(bound_v, bound_i, vw, base + 3u))
        selector_insert_top4(vw, base + 3u, v0, i0, v1, i1, v2, i2, v3, i3);
    }
  }
  if ((vocab & 3u) != 0u && (vocab4 & (64u * 256u - 1u)) == start_vec) {
    for (uint t = vocab4 << 2u; t < vocab; ++t) {
      const float v = float(row_logits[t]);
      if (!has_upper_bound || selector_better(bound_v, bound_i, v, t))
        selector_insert_top4(v, t, v0, i0, v1, i1, v2, i2, v3, i3);
    }
  }
}

// Coalesced Zero-Spill Stage 1 Top-K (K <= 32, tuned for K=16 on M5 Ultra):
// Grid: {.width = 64 * 256, .height = 7}, Threads: {.width = 256}.
// 448 TGs across all 64 M5 Ultra GPU cores; 100% SIMD-coalesced bfloat4 SLC
// reads, pure scalar top-4 register cache + double-buffered 1-barrier warp
// reduction in 128B TG SRAM (0 bytes stack spill, 0-ULP selector_better).
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_selector_topk16_stage1(
    device const bfloat *logits [[buffer(0)]],
    device uint *partial_ids [[buffer(1)]],
    device float *partial_scores [[buffer(2)]],
    constant SelectorParams &p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint chunk = tg.x;
  const uint row = tg.y + 1u;
  if (chunk >= 64u || row >= p.width || p.top_k == 0u)
    return;
  const uint top_k = min(p.top_k, 32u);
  device const bfloat *row_logits = logits + ulong(row) * p.vocab;

  float v0, v1, v2, v3;
  uint i0, i1, i2, i3;
  selector_stage1_scan_thread(row_logits, p.vocab, chunk, tid, false, 0.0f, 0u,
                              v0, i0, v1, i1, v2, i2, v3, i3);

  threadgroup float sg_v[2][8];
  threadgroup uint sg_i[2][8];
  const ulong row_base = ulong(row) * p.top_k;
  uint popped = 0u;

  for (uint step = 0u; step < top_k; ++step) {
    const uint buf = step & 1u;
    float rv = v0;
    uint ri = i0;
#pragma clang loop unroll(full)
    for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
      const float ov = simd_shuffle_down(rv, offset);
      const uint oi = simd_shuffle_down(ri, offset);
      if (selector_better(ov, oi, rv, ri)) {
        rv = ov;
        ri = oi;
      }
    }
    if (lane == 0u) {
      sg_v[buf][sg] = rv;
      sg_i[buf][sg] = ri;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float red_v = (lane < 8u) ? sg_v[buf][lane] : -INFINITY;
    uint red_i = (lane < 8u) ? sg_i[buf][lane] : 0xffffffffu;
#pragma clang loop unroll(full)
    for (ushort offset = 4u; offset > 0u; offset >>= 1u) {
      const float ov = simd_shuffle_down(red_v, offset);
      const uint oi = simd_shuffle_down(red_i, offset);
      if (selector_better(ov, oi, red_v, red_i)) {
        red_v = ov;
        red_i = oi;
      }
    }
    const float win_v = simd_broadcast(red_v, 0u);
    const uint win_i = simd_broadcast(red_i, 0u);

    if (tid == 0u) {
      const ulong out_idx = (row_base + step) * 64ul + chunk;
      partial_ids[out_idx] = win_i;
      partial_scores[out_idx] = win_v;
    }
    if (win_i == 0xffffffffu) {
      if (tid == 0u) {
        for (uint s = step + 1u; s < top_k; ++s) {
          const ulong out_idx = (row_base + s) * 64ul + chunk;
          partial_ids[out_idx] = 0xffffffffu;
          partial_scores[out_idx] = -INFINITY;
        }
      }
      break;
    }
    if (((win_i >> 2u) & 255u) == tid) {
      ++popped;
      if (popped < 4u) {
        v0 = v1;
        i0 = i1;
        v1 = v2;
        i1 = i2;
        v2 = v3;
        i2 = i3;
        v3 = -INFINITY;
        i3 = 0xffffffffu;
      } else if (step + 1u < top_k) {
        selector_stage1_scan_thread(row_logits, p.vocab, chunk, tid, true,
                                    win_v, win_i, v0, i0, v1, i1, v2, i2, v3,
                                    i3);
        popped = 0u;
      }
    }
  }
}

// Coalesced Zero-Spill Stage 2 Top-K Merge:
// Grid: {.width = 64, .height = 7}, Threads: {.width = 64}.
// 64 threads (2 warps) merge 64 pre-sorted chunk streams in 16 register/SRAM
// steps with 0 bytes stack spill and exact 0-ULP selector_better tie-breaking.
[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_selector_topk16_stage2(
    device const uint *partial_ids [[buffer(0)]],
    device const float *partial_scores [[buffer(1)]],
    device uint *ids_out [[buffer(2)]],
    device float *scores_out [[buffer(3)]],
    constant SelectorParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y + 1u;
  if (row >= p.width || p.top_k == 0u)
    return;
  const uint top_k = min(p.top_k, 32u);
  threadgroup float sg_v[2][2];
  threadgroup uint sg_i[2][2];

  uint cursor = 0u;
  const ulong row_base = ulong(row) * p.top_k;
  uint cur_i = partial_ids[row_base * 64ul + tid];
  float cur_v = partial_scores[row_base * 64ul + tid];

  for (uint step = 0u; step < top_k; ++step) {
    const uint buf = step & 1u;
    float rv = cur_v;
    uint ri = cur_i;
#pragma clang loop unroll(full)
    for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
      const float ov = simd_shuffle_down(rv, offset);
      const uint oi = simd_shuffle_down(ri, offset);
      if (selector_better(ov, oi, rv, ri)) {
        rv = ov;
        ri = oi;
      }
    }
    if (lane == 0u) {
      sg_v[buf][sg] = rv;
      sg_i[buf][sg] = ri;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float win_v = sg_v[buf][0];
    uint win_i = sg_i[buf][0];
    if (selector_better(sg_v[buf][1], sg_i[buf][1], win_v, win_i)) {
      win_v = sg_v[buf][1];
      win_i = sg_i[buf][1];
    }

    if (tid == 0u) {
      ids_out[row_base + step] = win_i;
      scores_out[row_base + step] = win_v;
    }
    if (win_i == 0xffffffffu) {
      if (tid == 0u) {
        for (uint s = step + 1u; s < top_k; ++s) {
          ids_out[row_base + s] = 0xffffffffu;
          scores_out[row_base + s] = -INFINITY;
        }
      }
      break;
    }
    if (cur_i == win_i) {
      ++cursor;
      if (cursor < top_k) {
        const ulong idx = (row_base + cursor) * 64ul + tid;
        cur_i = partial_ids[idx];
        cur_v = partial_scores[idx];
      } else {
        cur_i = 0xffffffffu;
        cur_v = -INFINITY;
      }
    }
  }
}

kernel void
omlx_m5_selector_topk_chunks(device const bfloat *logits [[buffer(0)]],
                             device uint *partial_ids [[buffer(1)]],
                             device float *partial_scores [[buffer(2)]],
                             constant SelectorParams &p [[buffer(3)]],
                             uint2 index [[thread_position_in_grid]]) {
  uint block = index.x, row = index.y;
  if (block >= p.blocks || row == 0 || row >= p.width)
    return;
  float scores[32];
  uint ids[32];
  for (uint i = 0; i < p.top_k; ++i) {
    scores[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  for (uint token = block * 256; token < min((block + 1) * 256, p.vocab);
       ++token) {
    float value = float(logits[ulong(row) * p.vocab + token]);
    if (!selector_better(value, token, scores[p.top_k - 1], ids[p.top_k - 1]))
      continue;
    uint position = p.top_k - 1;
    while (position && selector_better(value, token, scores[position - 1],
                                       ids[position - 1])) {
      scores[position] = scores[position - 1];
      ids[position] = ids[position - 1];
      --position;
    }
    scores[position] = value;
    ids[position] = token;
  }
  ulong base = (ulong(row) * p.blocks + block) * p.top_k;
  for (uint i = 0; i < p.top_k; ++i) {
    partial_ids[base + i] = ids[i];
    partial_scores[base + i] = scores[i];
  }
}
kernel void
omlx_m5_selector_topk_reduce32(device const uint *partial_ids [[buffer(0)]],
                               device const float *partial_scores [[buffer(1)]],
                               device uint *group_ids [[buffer(2)]],
                               device float *group_scores [[buffer(3)]],
                               constant SelectorParams &p [[buffer(4)]],
                               uint2 index [[thread_position_in_grid]]) {
  uint group = index.x, row = index.y;
  if (group >= p.groups || row == 0 || row >= p.width)
    return;
  float scores[32];
  uint ids[32];
  for (uint i = 0; i < p.top_k; ++i) {
    scores[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  for (uint block = group * 32; block < min((group + 1) * 32, p.blocks);
       ++block) {
    ulong base = (ulong(row) * p.blocks + block) * p.top_k;
    for (uint candidate = 0; candidate < p.top_k; ++candidate) {
      uint token = partial_ids[base + candidate];
      float value = partial_scores[base + candidate];
      if (token == 0xffffffffu ||
          !selector_better(value, token, scores[p.top_k - 1], ids[p.top_k - 1]))
        continue;
      uint position = p.top_k - 1;
      while (position && selector_better(value, token, scores[position - 1],
                                         ids[position - 1])) {
        scores[position] = scores[position - 1];
        ids[position] = ids[position - 1];
        --position;
      }
      scores[position] = value;
      ids[position] = token;
    }
  }
  ulong base = (ulong(row) * p.groups + group) * p.top_k;
  for (uint i = 0; i < p.top_k; ++i) {
    group_ids[base + i] = ids[i];
    group_scores[base + i] = scores[i];
  }
}
kernel void
omlx_m5_selector_topk_merge(device const uint *partial_ids [[buffer(0)]],
                            device const float *partial_scores [[buffer(1)]],
                            device uint *ids_out [[buffer(2)]],
                            device float *scores_out [[buffer(3)]],
                            constant SelectorParams &p [[buffer(4)]],
                            uint row [[thread_position_in_grid]]) {
  if (row == 0 || row >= p.width)
    return;
  float scores[32];
  uint ids[32];
  for (uint i = 0; i < p.top_k; ++i) {
    scores[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  for (uint block = 0; block < p.groups; ++block) {
    ulong base = (ulong(row) * p.groups + block) * p.top_k;
    for (uint candidate = 0; candidate < p.top_k; ++candidate) {
      uint token = partial_ids[base + candidate];
      float value = partial_scores[base + candidate];
      if (token == 0xffffffffu ||
          !selector_better(value, token, scores[p.top_k - 1], ids[p.top_k - 1]))
        continue;
      uint position = p.top_k - 1;
      while (position && selector_better(value, token, scores[position - 1],
                                         ids[position - 1])) {
        scores[position] = scores[position - 1];
        ids[position] = ids[position - 1];
        --position;
      }
      scores[position] = value;
      ids[position] = token;
    }
  }
  ulong base = ulong(row) * p.top_k;
  for (uint i = 0; i < p.top_k; ++i) {
    ids_out[base + i] = ids[i];
    scores_out[base + i] = scores[i];
  }
}

// Supports both legacy 1D scalar launch ({.width = 8 * rank}, height == 1) and
// 2D SIMDgroup-coalesced launch ({.width = 32, .height = rank}, tptg = {32, 1}).
kernel void
omlx_m5_selector_project_hidden(device const bfloat *hidden [[buffer(0)]],
                                device const bfloat *weights [[buffer(1)]],
                                device float *projected [[buffer(2)]],
                                constant SelectorParams &p [[buffer(3)]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint3 gid [[thread_position_in_grid]],
                                uint3 grid_size [[threads_per_grid]]) {
  if (grid_size.y > 1u) {
    const uint rank = gid.y;
    if (rank >= p.rank || p.width <= 1u)
      return;
    device const bfloat *w_row = weights + ulong(rank) * p.hidden;
    if (p.hidden >= 256u && (p.hidden & 3u) == 0u) {
      const uint vecs = p.hidden >> 2u;
      device const bfloat4 *w4 = reinterpret_cast<device const bfloat4 *>(w_row);
      const uint active_rows = min(p.width - 1u, 7u);
      float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f, p4 = 0.0f, p5 = 0.0f,
            p6 = 0.0f;
      const auto dot_row = [&](uint r, float4 wv, uint v, float acc) -> float {
        if (r >= active_rows)
          return acc;
        device const bfloat4 *h4 = reinterpret_cast<device const bfloat4 *>(
            hidden + ulong(r + 1u) * p.hidden);
        const float4 hv = float4(h4[v]);
        float dot = fma(wv.x, hv.x, acc);
        dot = fma(wv.y, hv.y, dot);
        dot = fma(wv.z, hv.z, dot);
        return fma(wv.w, hv.w, dot);
      };
      for (uint v = lane; v < vecs; v += 32u) {
        const float4 wv = float4(w4[v]);
        p0 = dot_row(0u, wv, v, p0);
        p1 = dot_row(1u, wv, v, p1);
        p2 = dot_row(2u, wv, v, p2);
        p3 = dot_row(3u, wv, v, p3);
        p4 = dot_row(4u, wv, v, p4);
        p5 = dot_row(5u, wv, v, p5);
        p6 = dot_row(6u, wv, v, p6);
      }
      const auto write_row = [&](uint r, float part) {
        if (r < active_rows) {
          const float total = simd_sum(part);
          if (lane == 0u)
            projected[ulong(r + 1u) * p.rank + rank] = total;
        }
      };
      write_row(0u, p0);
      write_row(1u, p1);
      write_row(2u, p2);
      write_row(3u, p3);
      write_row(4u, p4);
      write_row(5u, p5);
      write_row(6u, p6);
      return;
    }
    // Exact sequential accumulation with coalesced 32-lane loads for small fixtures.
    const uint active_rows = min(p.width - 1u, 7u);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f,
          a6 = 0.0f;
    const auto seq_row = [&](uint r, float wv, uint f, uint valid_lanes,
                             float acc) -> float {
      if (r >= active_rows)
        return acc;
      const float hv =
          f < p.hidden ? float(hidden[ulong(r + 1u) * p.hidden + f]) : 0.0f;
      for (uint k = 0; k < valid_lanes; ++k) {
        acc = fma(simd_broadcast(wv, ushort(k)),
                  simd_broadcast(hv, ushort(k)), acc);
      }
      return acc;
    };
    for (uint base_f = 0; base_f < p.hidden; base_f += 32u) {
      const uint f = base_f + lane;
      const float wv = f < p.hidden ? float(w_row[f]) : 0.0f;
      const uint valid_lanes = min(32u, p.hidden - base_f);
      a0 = seq_row(0u, wv, f, valid_lanes, a0);
      a1 = seq_row(1u, wv, f, valid_lanes, a1);
      a2 = seq_row(2u, wv, f, valid_lanes, a2);
      a3 = seq_row(3u, wv, f, valid_lanes, a3);
      a4 = seq_row(4u, wv, f, valid_lanes, a4);
      a5 = seq_row(5u, wv, f, valid_lanes, a5);
      a6 = seq_row(6u, wv, f, valid_lanes, a6);
    }
    if (lane == 0u) {
      if (0u < active_rows) projected[1ul * p.rank + rank] = a0;
      if (1u < active_rows) projected[2ul * p.rank + rank] = a1;
      if (2u < active_rows) projected[3ul * p.rank + rank] = a2;
      if (3u < active_rows) projected[4ul * p.rank + rank] = a3;
      if (4u < active_rows) projected[5ul * p.rank + rank] = a4;
      if (5u < active_rows) projected[6ul * p.rank + rank] = a5;
      if (6u < active_rows) projected[7ul * p.rank + rank] = a6;
    }
    return;
  }

  uint row = gid.x / p.rank, rank = gid.x % p.rank;
  if (row == 0 || row >= p.width)
    return;
  float sum = 0.0f;
  for (uint f = 0; f < p.hidden; ++f)
    sum = fma(float(weights[ulong(rank) * p.hidden + f]),
              float(hidden[ulong(row) * p.hidden + f]), sum);
  projected[gid.x] = sum;
}

__attribute__((always_inline)) inline void selector_compute_candidate_energies(
    device const uint *ids, device const float *unary,
    device const float *projected, device const bfloat *predecessor,
    device const bfloat *successor, constant SelectorParams &p, uint row,
    uint previous, bool cache_pp, threadgroup float *tg_pred_proj,
    threadgroup float *tg_energies, uint lane, uint sg, uint tid,
    uint threads_in_tg) {
  const bool valid_prev = (previous < p.vocab);
  if (cache_pp) {
    for (uint r = tid; r < p.rank; r += threads_in_tg) {
      tg_pred_proj[r] = valid_prev
                            ? (float(predecessor[ulong(previous) * p.rank + r]) *
                               projected[ulong(row) * p.rank + r])
                            : 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (threads_in_tg >= 1024u) {
    // 32 SIMDgroups: SIMDgroup `sg` owns candidate `sg` with coalesced loads.
    if (sg < p.top_k) {
      const uint token = ids[ulong(row) * p.top_k + sg];
      if (!valid_prev || token >= p.vocab) {
        if (lane == 0u) {
          tg_energies[sg] = -INFINITY;
        }
      } else {
        device const bfloat *succ = successor + ulong(token) * p.rank;
        if (p.rank >= 128u && (p.rank & 3u) == 0u) {
          float part = 0.0f;
          const uint vecs = p.rank >> 2u;
          device const bfloat4 *succ4 =
              reinterpret_cast<device const bfloat4 *>(succ);
          if (cache_pp) {
            for (uint v = lane; v < vecs; v += 32u) {
              const bfloat4 sv = succ4[v];
              const uint b = v << 2u;
              part = fma(tg_pred_proj[b + 0u], float(sv.x), part);
              part = fma(tg_pred_proj[b + 1u], float(sv.y), part);
              part = fma(tg_pred_proj[b + 2u], float(sv.z), part);
              part = fma(tg_pred_proj[b + 3u], float(sv.w), part);
            }
          } else {
            const ulong pred_base = ulong(previous) * p.rank;
            const ulong proj_base = ulong(row) * p.rank;
            for (uint v = lane; v < vecs; v += 32u) {
              const bfloat4 sv = succ4[v];
              const uint b = v << 2u;
              part = fma(float(predecessor[pred_base + b + 0u]) *
                             projected[proj_base + b + 0u],
                         float(sv.x), part);
              part = fma(float(predecessor[pred_base + b + 1u]) *
                             projected[proj_base + b + 1u],
                         float(sv.y), part);
              part = fma(float(predecessor[pred_base + b + 2u]) *
                             projected[proj_base + b + 2u],
                         float(sv.z), part);
              part = fma(float(predecessor[pred_base + b + 3u]) *
                             projected[proj_base + b + 3u],
                         float(sv.w), part);
            }
          }
          const float interaction = simd_sum(part);
          if (lane == 0u) {
            tg_energies[sg] = unary[ulong(row) * p.top_k + sg] + interaction;
          }
        } else {
          float interaction = 0.0f;
          for (uint base_r = 0; base_r < p.rank; base_r += 32u) {
            const uint r = base_r + lane;
            const float sv = r < p.rank ? float(succ[r]) : 0.0f;
            const float pp =
                r < p.rank
                    ? (cache_pp
                           ? tg_pred_proj[r]
                           : float(predecessor[ulong(previous) * p.rank + r]) *
                                 projected[ulong(row) * p.rank + r])
                    : 0.0f;
            const uint valid_lanes = min(32u, p.rank - base_r);
            for (uint k = 0; k < valid_lanes; ++k) {
              interaction = fma(simd_broadcast(pp, ushort(k)),
                                simd_broadcast(sv, ushort(k)), interaction);
            }
          }
          if (lane == 0u) {
            tg_energies[sg] = unary[ulong(row) * p.top_k + sg] + interaction;
          }
        }
      }
    }
  } else {
    if (tid < p.top_k) {
      const uint token = ids[ulong(row) * p.top_k + tid];
      if (!valid_prev || token >= p.vocab) {
        tg_energies[tid] = -INFINITY;
      } else {
        float interaction = 0.0f;
        device const bfloat *succ = successor + ulong(token) * p.rank;
        if (cache_pp) {
          for (uint r = 0; r < p.rank; ++r) {
            interaction = fma(tg_pred_proj[r], float(succ[r]), interaction);
          }
        } else {
          for (uint r = 0; r < p.rank; ++r) {
            interaction = fma(float(predecessor[ulong(previous) * p.rank + r]) *
                                  projected[ulong(row) * p.rank + r],
                              float(succ[r]), interaction);
          }
        }
        tg_energies[tid] = unary[ulong(row) * p.top_k + tid] + interaction;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void omlx_m5_selector_finish(device const uint *ids [[buffer(0)]],
                                    device const float *unary [[buffer(1)]],
                                    device const float *projected [[buffer(2)]],
                                    device const bfloat *predecessor
                                    [[buffer(3)]],
                                    device const bfloat *successor
                                    [[buffer(4)]],
                                    device uint *selected [[buffer(5)]],
                                    device float *probabilities [[buffer(6)]],
                                    device uint *fallback [[buffer(7)]],
                                    constant SelectorParams &p [[buffer(8)]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint3 tptg [[threads_per_threadgroup]]) {
  if (p.vocab == 0u || p.top_k == 0u) {
    if (tid == 0u) {
      fallback[0] = 1u;
    }
    return;
  }
  threadgroup float tg_energies[32];
  threadgroup float tg_values[32];
  threadgroup float tg_pred_proj[512];
  threadgroup uint tg_previous;
  if (tid == 0u) {
    fallback[0] = (p.anchor >= p.vocab) ? 1u : 0u;
    tg_previous = p.anchor;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint threads_in_tg = tptg.x * tptg.y * tptg.z;
  const bool cache_pp = (p.rank <= 512u);
  for (uint row = 1; row < p.width; ++row) {
    const uint previous = tg_previous;
    selector_compute_candidate_energies(
        ids, unary, projected, predecessor, successor, p, row, previous,
        cache_pp, tg_pred_proj, tg_energies, lane, sg, tid, threads_in_tg);
    if (tid == 0u) {
      const ulong row_off = ulong(row) * p.top_k;
      if (previous >= p.vocab) {
        fallback[0] = 1u;
      }
      float maximum = -INFINITY;
      for (uint candidate = 0; candidate < p.top_k; ++candidate) {
        if (ids[row_off + candidate] < p.vocab) {
          maximum = max(maximum, tg_energies[candidate]);
        }
      }
      uint chosen = 0;
      if (p.temperature <= 0.0f) {
        for (uint candidate = 1; candidate < p.top_k; ++candidate) {
          const uint id_c = ids[row_off + candidate];
          const uint id_b = ids[row_off + chosen];
          if (id_c < p.vocab &&
              (id_b >= p.vocab || tg_energies[candidate] > tg_energies[chosen]))
            chosen = candidate;
        }
        for (uint candidate = 0; candidate < p.top_k; ++candidate)
          probabilities[row_off + candidate] =
              candidate == chosen ? 1.0f : 0.0f;
      } else {
        float total = 0.0f;
        for (uint candidate = 0; candidate < p.top_k; ++candidate) {
          const bool valid_c =
              ids[row_off + candidate] < p.vocab && isfinite(maximum);
          tg_values[candidate] =
              valid_c ? exp((tg_energies[candidate] - maximum) / p.temperature)
                      : 0.0f;
          total += tg_values[candidate];
        }
        float cumulative = 0.0f, draw = p.uniforms[row - 1];
        uint picked = p.top_k;
        uint last_valid = 0;
        bool has_valid = false;
        for (uint candidate = 0; candidate < p.top_k; ++candidate) {
          const bool valid_c = ids[row_off + candidate] < p.vocab;
          float probability = (valid_c && total > 0.0f && isfinite(total))
                                  ? (tg_values[candidate] / total)
                                  : 0.0f;
          probabilities[row_off + candidate] = probability;
          cumulative += probability;
          if (valid_c &&
              (fabs(draw - cumulative) < 1e-4f || !isfinite(probability)))
            fallback[0] = 1;
          if (valid_c) {
            last_valid = candidate;
            has_valid = true;
            if (picked == p.top_k && draw < cumulative)
              picked = candidate;
          }
        }
        if (!has_valid || !(total > 0.0f) || !isfinite(total))
          fallback[0] = 1;
        chosen = (picked < p.top_k) ? picked : last_valid;
      }
      uint next_token = ids[row_off + chosen];
      if (next_token >= p.vocab || !isfinite(tg_energies[chosen])) {
        fallback[0] = 1u;
        if (next_token >= p.vocab) {
          next_token = 0u;
        }
      }
      tg_previous = next_token;
      selected[row] = next_token;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

struct SelectorPositionKey {
  ulong seed, position, purpose;
};
kernel void omlx_m5_selector_finish_position(
    device const uint *ids [[buffer(0)]],
    device const float *unary [[buffer(1)]],
    device const float *projected [[buffer(2)]],
    device const bfloat *predecessor [[buffer(3)]],
    device const bfloat *successor [[buffer(4)]],
    device uint *selected [[buffer(5)]],
    device float *probabilities [[buffer(6)]],
    device uint *fallback [[buffer(7)]],
    constant SelectorParams &p [[buffer(8)]],
    constant SelectorPositionKey &key [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]) {
  if (p.vocab == 0u || p.top_k == 0u) {
    if (tid == 0u) {
      fallback[0] = 1u;
    }
    return;
  }
  threadgroup float tg_energies[32];
  threadgroup float tg_values[32];
  threadgroup float tg_pred_proj[512];
  threadgroup uint tg_previous;
  if (tid == 0u) {
    fallback[0] = (p.anchor >= p.vocab) ? 1u : 0u;
    tg_previous = p.anchor;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint threads_in_tg = tptg.x * tptg.y * tptg.z;
  const bool cache_pp = (p.rank <= 512u);
  for (uint row = 1; row < p.width; ++row) {
    const uint previous = tg_previous;
    selector_compute_candidate_energies(
        ids, unary, projected, predecessor, successor, p, row, previous,
        cache_pp, tg_pred_proj, tg_energies, lane, sg, tid, threads_in_tg);
    if (tid == 0u) {
      const ulong row_off = ulong(row) * p.top_k;
      if (previous >= p.vocab) {
        fallback[0] = 1u;
      }
      float maximum = -INFINITY;
      for (uint i = 0; i < p.top_k; ++i) {
        const uint id_i = ids[row_off + i];
        if (id_i >= p.vocab || !isfinite(tg_energies[i]))
          fallback[0] = 1;
        if (id_i < p.vocab)
          maximum = max(maximum, tg_energies[i]);
      }
      uint chosen = 0;
      if (p.temperature <= 0.0f) {
        for (uint i = 1; i < p.top_k; ++i) {
          const uint id_i = ids[row_off + i];
          const uint id_b = ids[row_off + chosen];
          if (id_i < p.vocab &&
              (id_b >= p.vocab || tg_energies[i] > tg_energies[chosen] ||
               (tg_energies[i] == tg_energies[chosen] && id_i < id_b)))
            chosen = i;
        }
        for (uint i = 0; i < p.top_k; ++i)
          probabilities[row_off + i] = i == chosen ? 1.f : 0.f;
      } else {
        float total = 0.0f;
        for (uint i = 0; i < p.top_k; ++i) {
          const bool valid_i = ids[row_off + i] < p.vocab && isfinite(maximum);
          tg_values[i] =
              valid_i ? exp((tg_energies[i] - maximum) / p.temperature) : 0.0f;
          total += tg_values[i];
        }
        float best = -INFINITY, runner = -INFINITY;
        bool has_valid = false;
        for (uint i = 0; i < p.top_k; ++i) {
          const uint id_i = ids[row_off + i];
          probabilities[row_off + i] =
              (id_i < p.vocab && total > 0.0f && isfinite(total))
                  ? (tg_values[i] / total)
                  : 0.0f;
          if (id_i >= p.vocab || !isfinite(maximum))
            continue;
          const float score =
              (tg_energies[i] - maximum) / p.temperature +
              position_gumbel(key.seed, key.position + row - 1, id_i,
                              key.purpose);
          if (!has_valid || score > best ||
              (score == best && id_i < ids[row_off + chosen])) {
            runner = best;
            best = score;
            chosen = i;
            has_valid = true;
          } else {
            runner = max(runner, score);
          }
        }
        if (!has_valid || !(total > 0.0f) || !isfinite(total) ||
            best - runner < 2e-4f)
          fallback[0] = 1;
      }
      uint next_token = ids[row_off + chosen];
      if (next_token >= p.vocab || !isfinite(tg_energies[chosen])) {
        fallback[0] = 1u;
        if (next_token >= p.vocab) {
          next_token = 0u;
        }
      }
      tg_previous = next_token;
      selected[row] = next_token;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ============================================================================
// Batched multi-slot [8*B, H] DFlash2 packed drafter kernels (B = 2..8).
// Each slot b occupies fixed rows [8*b .. 8*b + 7] for backbone/head/selector
// while context rows are packed contiguously at [dst_row_offset .. +tokens].
// Zero cross-slot conv/RoPE/KV/selector leakage; 0-ULP identical to B=1.
// ============================================================================

struct ContextGatherOffset {
  uint tokens, width, position, ring_rows;
  uint dst_row_offset, pad0, pad1, pad2;
};

kernel void omlx_m5_draft_context_gather_offset(
    device const bfloat *source [[buffer(0)]],
    device bfloat *out [[buffer(1)]],
    constant ContextGatherOffset &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (ulong(i) >= ulong(p.tokens) * p.width)
    return;
  const uint t = i / p.width;
  const uint col = i % p.width;
  const uint src_row =
      p.ring_rows ? ((p.position + t) % p.ring_rows) : (p.position + t);
  out[ulong(p.dst_row_offset + t) * p.width + col] =
      source[ulong(src_row) * p.width + col];
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_draft_context_gather_vec8_offset(
    device const bfloat *source [[buffer(0)]],
    device bfloat *out [[buffer(1)]],
    constant ContextGatherOffset &p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint vec_width = p.width >> 3u;
  const uint col8 = gid.x;
  const uint t = gid.y;
  if (t >= p.tokens || col8 >= vec_width)
    return;
  const uint src_row =
      p.ring_rows ? ((p.position + t) % p.ring_rows) : (p.position + t);
  device const uint4 *src4 =
      reinterpret_cast<device const uint4 *>(source + ulong(src_row) * p.width);
  device uint4 *dst4 = reinterpret_cast<device uint4 *>(
      out + ulong(p.dst_row_offset + t) * p.width);
  dst4[col8] = src4[col8];
}

struct ContextRopeAppendOffsetParams {
  uint tokens;
  uint kv_heads;
  uint dim;
  uint position;
  uint capacity;
  float epsilon;
  float theta;
  uint src_row_offset;
};

[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_draft_context_rope_cache_append_offset(
    device const bfloat *context_kv [[buffer(0)]],
    device const bfloat *k_weight [[buffer(1)]],
    device bfloat *cache_keys [[buffer(2)]],
    device bfloat *cache_values [[buffer(3)]],
    constant ContextRopeAppendOffsetParams &p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  if (p.capacity == 0u)
    return;
  const uint t = group.y / p.kv_heads;
  if (t >= p.tokens)
    return;
  const uint head = group.y % p.kv_heads;
  const ulong kv_width = ulong(p.kv_heads) * p.dim;
  const ulong src_k =
      ulong(p.src_row_offset + t) * (2ul * kv_width) + ulong(head) * p.dim;
  const ulong src_v = src_k + kv_width;
  const uint dst_row = (p.position + t) % p.capacity;
  const ulong dst_off = ulong(dst_row) * kv_width + ulong(head) * p.dim;
  const uint half_dim = p.dim >> 1u;

  float square = 0.0f;
  for (uint f = lane; f < p.dim; f += 32u) {
    const float kv_f = float(context_kv[src_k + f]);
    square = fma(kv_f, kv_f, square);
  }
  const float inverse = rsqrt(simd_sum(square) / float(p.dim) + p.epsilon);
  const float pos_f = float(p.position + t);
  for (uint pair = lane; pair < half_dim; pair += 32u) {
    const float left = float(bfloat(float(context_kv[src_k + pair]) * inverse *
                                    float(k_weight[pair])));
    const float right =
        float(bfloat(float(context_kv[src_k + half_dim + pair]) * inverse *
                     float(k_weight[half_dim + pair])));
    const float angle = pos_f * pow(p.theta, -float(pair) / float(half_dim));
    const float c = cos(angle);
    const float s = sin(angle);
    cache_keys[dst_off + pair] = bfloat(left * c - right * s);
    cache_keys[dst_off + half_dim + pair] = bfloat(left * s + right * c);
  }
  for (uint f = lane; f < p.dim; f += 32u) {
    cache_values[dst_off + f] = context_kv[src_v + f];
  }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_draft_context_rope_cache_append_split_offset(
    device const bfloat *context_k [[buffer(0)]],
    device const bfloat *context_v [[buffer(1)]],
    device const bfloat *k_weight [[buffer(2)]],
    device bfloat *cache_keys [[buffer(3)]],
    device bfloat *cache_values [[buffer(4)]],
    constant ContextRopeAppendOffsetParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  if (p.capacity == 0u)
    return;
  const uint t = group.y / p.kv_heads;
  if (t >= p.tokens)
    return;
  const uint head = group.y % p.kv_heads;
  const ulong kv_width = ulong(p.kv_heads) * p.dim;
  const ulong src_off =
      ulong(p.src_row_offset + t) * kv_width + ulong(head) * p.dim;
  const uint dst_row = (p.position + t) % p.capacity;
  const ulong dst_off = ulong(dst_row) * kv_width + ulong(head) * p.dim;
  const uint half_dim = p.dim >> 1u;

  float square = 0.0f;
  for (uint f = lane; f < p.dim; f += 32u) {
    const float kv_f = float(context_k[src_off + f]);
    square = fma(kv_f, kv_f, square);
  }
  const float inverse = rsqrt(simd_sum(square) / float(p.dim) + p.epsilon);
  const float pos_f = float(p.position + t);
  for (uint pair = lane; pair < half_dim; pair += 32u) {
    const float left = float(bfloat(float(context_k[src_off + pair]) * inverse *
                                    float(k_weight[pair])));
    const float right =
        float(bfloat(float(context_k[src_off + half_dim + pair]) * inverse *
                     float(k_weight[half_dim + pair])));
    const float angle = pos_f * pow(p.theta, -float(pair) / float(half_dim));
    const float c = cos(angle);
    const float s = sin(angle);
    cache_keys[dst_off + pair] = bfloat(left * c - right * s);
    cache_keys[dst_off + half_dim + pair] = bfloat(left * s + right * c);
  }
  for (uint f = lane; f < p.dim; f += 32u) {
    cache_values[dst_off + f] = context_v[src_off + f];
  }
}

struct BatchedConvParams {
  uint batch_size, hidden, group_size, output_half;
  uint slot_widths[8];
};

kernel void omlx_m5_draft_conv_batched(device const bfloat *x [[buffer(0)]],
                                       device const bfloat *dynamic
                                       [[buffer(1)]],
                                       device const bfloat *base [[buffer(2)]],
                                       device bfloat *out [[buffer(3)]],
                                       constant BatchedConvParams &p
                                       [[buffer(4)]],
                                       uint2 gid [[thread_position_in_grid]]) {
  const uint row = gid.y;
  const uint f = gid.x;
  const uint slot = row >> 3u;
  const uint local_t = row & 7u;
  if (slot >= p.batch_size || local_t >= p.slot_widths[slot] || f >= p.hidden)
    return;
  const uint groups = p.hidden / p.group_size;
  const uint d = row * 4u * groups + p.output_half * 2u * groups +
                 f / p.group_size;
  const uint b_idx = p.output_half * 2u * p.hidden + f;
  const ulong pos = ulong(row) * p.hidden + f;
  const bfloat w0 = bfloat(float(base[b_idx]) + float(dynamic[d]));
  bfloat value = bfloat(float(w0) * float(x[pos]));
  if (local_t > 0u) {
    const bfloat w1 =
        bfloat(float(base[b_idx + p.hidden]) + float(dynamic[d + groups]));
    value = bfloat(float(value) +
                   float(bfloat(float(w1) * float(x[pos - p.hidden]))));
  }
  out[pos] = value;
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_draft_conv_vec4_batched(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *dynamic [[buffer(1)]],
    device const bfloat *base [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant BatchedConvParams &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint vec_cols = p.hidden >> 2u;
  const uint v = gid.x;
  const uint row = gid.y;
  const uint slot = row >> 3u;
  const uint local_t = row & 7u;
  if (slot >= p.batch_size || local_t >= p.slot_widths[slot] || v >= vec_cols)
    return;
  const uint groups = p.hidden / p.group_size;
  const uint g =
      (p.group_size == 16u) ? (v >> 2u) : ((v << 2u) / p.group_size);
  const uint d = row * 4u * groups + p.output_half * 2u * groups + g;
  device const bfloat4 *x4 = reinterpret_cast<device const bfloat4 *>(x);
  device const bfloat4 *base4 = reinterpret_cast<device const bfloat4 *>(base);
  device bfloat4 *out4 = reinterpret_cast<device bfloat4 *>(out);
  const uint b_vec = (p.output_half * 2u * p.hidden) >> 2u;
  const float4 w0 =
      float4(bfloat4(float4(base4[b_vec + v]) + float(dynamic[d])));
  bfloat4 value = bfloat4(w0 * float4(x4[ulong(row) * vec_cols + v]));
  if (local_t > 0u) {
    const float4 w1 = float4(bfloat4(float4(base4[b_vec + vec_cols + v]) +
                                     float(dynamic[d + groups])));
    const float4 prev =
        float4(bfloat4(w1 * float4(x4[ulong(row - 1u) * vec_cols + v])));
    value = bfloat4(float4(value) + prev);
  }
  out4[ulong(row) * vec_cols + v] = value;
}

struct BatchedConvAddNormParams {
  uint batch_size;
  uint hidden;
  uint group_size;
  float epsilon;
  uint slot_widths[8];
};

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_draft_conv_add_rmsnorm_batched(
    device const bfloat *projected [[buffer(0)]],
    device const bfloat *dynamic [[buffer(1)]],
    device const bfloat *base [[buffer(2)]],
    device bfloat *hidden_inout [[buffer(3)]],
    device const bfloat *norm_weight [[buffer(4)]],
    device bfloat *norm_out [[buffer(5)]],
    constant BatchedConvAddNormParams &p [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint kThreads = 256u;
  const uint row = tg.y;
  const uint slot = row >> 3u;
  const uint local_t = row & 7u;
  if (slot >= p.batch_size || local_t >= p.slot_widths[slot])
    return;
  threadgroup float simd_sums[8];
  const ulong row_base = ulong(row) * p.hidden;

  if (p.hidden == 5120u && p.group_size == 16u) {
    constexpr uint kGroups = 320u;
    device const bfloat4 *pv_cur =
        reinterpret_cast<device const bfloat4 *>(projected + row_base);
    device const bfloat4 *pv_prev =
        local_t > 0u ? reinterpret_cast<device const bfloat4 *>(
                           projected + row_base - 5120ul)
                     : nullptr;
    device const bfloat4 *b0v =
        reinterpret_cast<device const bfloat4 *>(base + 2ul * 5120ul);
    device const bfloat4 *b1v =
        reinterpret_cast<device const bfloat4 *>(base + 3ul * 5120ul);
    device bfloat4 *hv =
        reinterpret_cast<device bfloat4 *>(hidden_inout + row_base);
    device const bfloat4 *wv =
        reinterpret_cast<device const bfloat4 *>(norm_weight);
    device bfloat4 *ov =
        reinterpret_cast<device bfloat4 *>(norm_out + row_base);
    const uint dyn_base0 = row * (4u * kGroups) + 2u * kGroups;
    const uint dyn_base1 = dyn_base0 + kGroups;

    float square_sum = 0.0f;
    const bfloat4 rs0 = draft_conv_add_norm_step(
        tid + 0u * kThreads, local_t, dyn_base0, dyn_base1, dynamic, b0v, b1v,
        pv_cur, pv_prev, hv, square_sum);
    const bfloat4 rs1 = draft_conv_add_norm_step(
        tid + 1u * kThreads, local_t, dyn_base0, dyn_base1, dynamic, b0v, b1v,
        pv_cur, pv_prev, hv, square_sum);
    const bfloat4 rs2 = draft_conv_add_norm_step(
        tid + 2u * kThreads, local_t, dyn_base0, dyn_base1, dynamic, b0v, b1v,
        pv_cur, pv_prev, hv, square_sum);
    const bfloat4 rs3 = draft_conv_add_norm_step(
        tid + 3u * kThreads, local_t, dyn_base0, dyn_base1, dynamic, b0v, b1v,
        pv_cur, pv_prev, hv, square_sum);
    const bfloat4 rs4 = draft_conv_add_norm_step(
        tid + 4u * kThreads, local_t, dyn_base0, dyn_base1, dynamic, b0v, b1v,
        pv_cur, pv_prev, hv, square_sum);
    square_sum = simd_sum(square_sum);
    if (lane == 0u)
      simd_sums[sg] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        (((simd_sums[0] + simd_sums[1]) + (simd_sums[2] + simd_sums[3])) +
         ((simd_sums[4] + simd_sums[5]) + (simd_sums[6] + simd_sums[7])));
    const float inv = rsqrt(total * (1.0f / 5120.0f) + p.epsilon);
    ov[tid + 0u * kThreads] =
        bfloat4(float4(rs0) * inv * float4(wv[tid + 0u * kThreads]));
    ov[tid + 1u * kThreads] =
        bfloat4(float4(rs1) * inv * float4(wv[tid + 1u * kThreads]));
    ov[tid + 2u * kThreads] =
        bfloat4(float4(rs2) * inv * float4(wv[tid + 2u * kThreads]));
    ov[tid + 3u * kThreads] =
        bfloat4(float4(rs3) * inv * float4(wv[tid + 3u * kThreads]));
    ov[tid + 4u * kThreads] =
        bfloat4(float4(rs4) * inv * float4(wv[tid + 4u * kThreads]));
    return;
  }

  const uint groups = p.hidden / p.group_size;
  const uint dyn_base = row * 4u * groups + 2u * groups;
  const uint b_off = 2u * p.hidden;
  device bfloat *h_row = hidden_inout + row_base;
  device bfloat *o_row = norm_out + row_base;
  float square_sum = 0.0f;
  for (uint f = tid; f < p.hidden; f += kThreads) {
    const uint g = f / p.group_size;
    const bfloat w0 =
        bfloat(float(base[b_off + f]) + float(dynamic[dyn_base + g]));
    bfloat delta = bfloat(float(w0) * float(projected[row_base + f]));
    if (local_t > 0u) {
      const bfloat w1 = bfloat(float(base[b_off + p.hidden + f]) +
                               float(dynamic[dyn_base + groups + g]));
      delta = bfloat(
          float(delta) +
          float(bfloat(float(w1) * float(projected[row_base - p.hidden + f]))));
    }
    const bfloat s = bfloat(float(h_row[f]) + float(delta));
    h_row[f] = s;
    const float sf = float(s);
    square_sum = fma(sf, sf, square_sum);
  }
  square_sum = simd_sum(square_sum);
  if (lane == 0u)
    simd_sums[sg] = square_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total =
      (((simd_sums[0] + simd_sums[1]) + (simd_sums[2] + simd_sums[3])) +
       ((simd_sums[4] + simd_sums[5]) + (simd_sums[6] + simd_sums[7])));
  const float inv = rsqrt(total / float(p.hidden) + p.epsilon);
  for (uint f = tid; f < p.hidden; f += kThreads) {
    o_row[f] = bfloat(float(h_row[f]) * inv * float(norm_weight[f]));
  }
}

struct BatchedQkRopeStridedParams {
  uint batch_size;
  uint q_heads;
  uint kv_heads;
  uint dim;
  uint qkv_stride;
  float epsilon;
  float theta;
  uint pad;
  uint slot_widths[8];
  uint slot_positions[8];
};

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_draft_qk_rope_strided_batched(
    device bfloat *qkv [[buffer(0)]],
    device const bfloat *q_weight [[buffer(1)]],
    device const bfloat *k_weight [[buffer(2)]],
    constant BatchedQkRopeStridedParams &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint total_heads = p.q_heads + p.kv_heads;
  const uint item = tg.y * 4u + sg;
  const uint row = item / total_heads;
  const uint slot = row >> 3u;
  const uint local_t = row & 7u;
  if (slot >= p.batch_size || local_t >= p.slot_widths[slot])
    return;
  const uint h_idx = item - row * total_heads;
  const bool is_q = h_idx < p.q_heads;
  const ulong base =
      ulong(row) * p.qkv_stride +
      (is_q ? ulong(h_idx) * p.dim
            : ulong(p.q_heads) * p.dim + ulong(h_idx - p.q_heads) * p.dim);
  device const bfloat *weight = is_q ? q_weight : k_weight;
  const uint half_dim = p.dim >> 1u;

  float square = 0.0f;
  for (uint f = lane; f < p.dim; f += 32u)
    square = fma(float(qkv[base + f]), float(qkv[base + f]), square);
  const float inverse = rsqrt(simd_sum(square) / float(p.dim) + p.epsilon);
  const float pos_f = float(p.slot_positions[slot] + local_t);
  for (uint pair = lane; pair < half_dim; pair += 32u) {
    const float left =
        float(bfloat(float(qkv[base + pair]) * inverse * float(weight[pair])));
    const float right = float(bfloat(float(qkv[base + half_dim + pair]) *
                                     inverse * float(weight[half_dim + pair])));
    const float angle = pos_f * pow(p.theta, -float(pair) / float(half_dim));
    const float c = cos(angle);
    const float s = sin(angle);
    qkv[base + pair] = bfloat(left * c - right * s);
    qkv[base + half_dim + pair] = bfloat(left * s + right * c);
  }
}

struct BatchedRopeParams {
  uint batch_size, heads, dim, pad;
  float epsilon, theta;
  uint slot_widths[8];
  uint slot_positions[8];
};

[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_draft_rope_batched(
    device bfloat *x [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    constant BatchedRopeParams &p [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  const uint row = group.y / p.heads;
  const uint slot = row >> 3u;
  const uint local_t = row & 7u;
  if (slot >= p.batch_size || local_t >= p.slot_widths[slot])
    return;
  const ulong base = ulong(group.y) * p.dim;
  const uint half_dim = p.dim >> 1u;
  float square = 0.0f;
  for (uint f = lane; f < p.dim; f += 32u)
    square = fma(float(x[base + f]), float(x[base + f]), square);
  const float inverse = rsqrt(simd_sum(square) / float(p.dim) + p.epsilon);
  const float pos_f = float(p.slot_positions[slot] + local_t);
  for (uint pair = lane; pair < half_dim; pair += 32u) {
    const float left =
        float(bfloat(float(x[base + pair]) * inverse * float(weight[pair])));
    const float right = float(bfloat(float(x[base + half_dim + pair]) *
                                     inverse * float(weight[half_dim + pair])));
    const float angle = pos_f * pow(p.theta, -float(pair) / float(half_dim));
    x[base + pair] = bfloat(left * cos(angle) - right * sin(angle));
    x[base + half_dim + pair] = bfloat(left * sin(angle) + right * cos(angle));
  }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_selector_project_hidden_batched(
    device const bfloat *hidden [[buffer(0)]],
    device const bfloat *weights [[buffer(1)]],
    device float *projected [[buffer(2)]],
    constant SelectorParams *ps [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 gid [[thread_position_in_grid]]) {
  const uint slot = gid.z;
  const uint rank = gid.y;
  const SelectorParams p = ps[slot];
  if (rank >= p.rank || p.width <= 1u)
    return;
  device const bfloat *slot_hidden = hidden + ulong(slot * 8u) * p.hidden;
  device float *slot_projected = projected + ulong(slot * 8u) * p.rank;
  device const bfloat *w_row = weights + ulong(rank) * p.hidden;
  const uint active_rows = min(p.width - 1u, 7u);
  if (p.hidden >= 256u && (p.hidden & 3u) == 0u) {
    const uint vecs = p.hidden >> 2u;
    device const bfloat4 *w4 = reinterpret_cast<device const bfloat4 *>(w_row);
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f, p4 = 0.0f, p5 = 0.0f,
          p6 = 0.0f;
    const auto dot_row = [&](uint r, float4 wv, uint v, float acc) -> float {
      if (r >= active_rows)
        return acc;
      device const bfloat4 *h4 = reinterpret_cast<device const bfloat4 *>(
          slot_hidden + ulong(r + 1u) * p.hidden);
      const float4 hv = float4(h4[v]);
      float dot = fma(wv.x, hv.x, acc);
      dot = fma(wv.y, hv.y, dot);
      dot = fma(wv.z, hv.z, dot);
      return fma(wv.w, hv.w, dot);
    };
    for (uint v = lane; v < vecs; v += 32u) {
      const float4 wv = float4(w4[v]);
      p0 = dot_row(0u, wv, v, p0);
      p1 = dot_row(1u, wv, v, p1);
      p2 = dot_row(2u, wv, v, p2);
      p3 = dot_row(3u, wv, v, p3);
      p4 = dot_row(4u, wv, v, p4);
      p5 = dot_row(5u, wv, v, p5);
      p6 = dot_row(6u, wv, v, p6);
    }
    const auto write_row = [&](uint r, float part) {
      if (r < active_rows) {
        const float total = simd_sum(part);
        if (lane == 0u)
          slot_projected[ulong(r + 1u) * p.rank + rank] = total;
      }
    };
    write_row(0u, p0);
    write_row(1u, p1);
    write_row(2u, p2);
    write_row(3u, p3);
    write_row(4u, p4);
    write_row(5u, p5);
    write_row(6u, p6);
    return;
  }
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f,
        a6 = 0.0f;
  const auto seq_row = [&](uint r, float wv, uint f, uint valid_lanes,
                           float acc) -> float {
    if (r >= active_rows)
      return acc;
    const float hv =
        f < p.hidden ? float(slot_hidden[ulong(r + 1u) * p.hidden + f]) : 0.0f;
    for (uint k = 0; k < valid_lanes; ++k) {
      acc = fma(simd_broadcast(wv, ushort(k)),
                simd_broadcast(hv, ushort(k)), acc);
    }
    return acc;
  };
  for (uint base_f = 0; base_f < p.hidden; base_f += 32u) {
    const uint f = base_f + lane;
    const float wv = f < p.hidden ? float(w_row[f]) : 0.0f;
    const uint valid_lanes = min(32u, p.hidden - base_f);
    a0 = seq_row(0u, wv, f, valid_lanes, a0);
    a1 = seq_row(1u, wv, f, valid_lanes, a1);
    a2 = seq_row(2u, wv, f, valid_lanes, a2);
    a3 = seq_row(3u, wv, f, valid_lanes, a3);
    a4 = seq_row(4u, wv, f, valid_lanes, a4);
    a5 = seq_row(5u, wv, f, valid_lanes, a5);
    a6 = seq_row(6u, wv, f, valid_lanes, a6);
  }
  if (lane == 0u) {
    if (0u < active_rows) slot_projected[1ul * p.rank + rank] = a0;
    if (1u < active_rows) slot_projected[2ul * p.rank + rank] = a1;
    if (2u < active_rows) slot_projected[3ul * p.rank + rank] = a2;
    if (3u < active_rows) slot_projected[4ul * p.rank + rank] = a3;
    if (4u < active_rows) slot_projected[5ul * p.rank + rank] = a4;
    if (5u < active_rows) slot_projected[6ul * p.rank + rank] = a5;
    if (6u < active_rows) slot_projected[7ul * p.rank + rank] = a6;
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_selector_topk16_stage1_batched(
    device const bfloat *logits [[buffer(0)]],
    device uint *partial_ids [[buffer(1)]],
    device float *partial_scores [[buffer(2)]],
    constant SelectorParams *ps [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint chunk = tg.x;
  const uint slot = tg.y / 7u;
  const uint local_row = (tg.y - slot * 7u) + 1u;
  const SelectorParams p = ps[slot];
  if (chunk >= 64u || local_row >= p.width || p.top_k == 0u)
    return;
  const uint top_k = min(p.top_k, 32u);
  const uint global_row = slot * 8u + local_row;
  device const bfloat *row_logits = logits + ulong(global_row) * p.vocab;

  float v0, v1, v2, v3;
  uint i0, i1, i2, i3;
  selector_stage1_scan_thread(row_logits, p.vocab, chunk, tid, false, 0.0f, 0u,
                              v0, i0, v1, i1, v2, i2, v3, i3);

  threadgroup float sg_v[2][8];
  threadgroup uint sg_i[2][8];
  const ulong row_base = ulong(global_row) * p.top_k;
  uint popped = 0u;

  for (uint step = 0u; step < top_k; ++step) {
    const uint buf = step & 1u;
    float rv = v0;
    uint ri = i0;
#pragma clang loop unroll(full)
    for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
      const float ov = simd_shuffle_down(rv, offset);
      const uint oi = simd_shuffle_down(ri, offset);
      if (selector_better(ov, oi, rv, ri)) {
        rv = ov;
        ri = oi;
      }
    }
    if (lane == 0u) {
      sg_v[buf][sg] = rv;
      sg_i[buf][sg] = ri;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float red_v = (lane < 8u) ? sg_v[buf][lane] : -INFINITY;
    uint red_i = (lane < 8u) ? sg_i[buf][lane] : 0xffffffffu;
#pragma clang loop unroll(full)
    for (ushort offset = 4u; offset > 0u; offset >>= 1u) {
      const float ov = simd_shuffle_down(red_v, offset);
      const uint oi = simd_shuffle_down(red_i, offset);
      if (selector_better(ov, oi, red_v, red_i)) {
        red_v = ov;
        red_i = oi;
      }
    }
    const float win_v = simd_broadcast(red_v, 0u);
    const uint win_i = simd_broadcast(red_i, 0u);

    if (tid == 0u) {
      const ulong out_idx = (row_base + step) * 64ul + chunk;
      partial_ids[out_idx] = win_i;
      partial_scores[out_idx] = win_v;
    }
    if (win_i == 0xffffffffu) {
      if (tid == 0u) {
        for (uint s = step + 1u; s < top_k; ++s) {
          const ulong out_idx = (row_base + s) * 64ul + chunk;
          partial_ids[out_idx] = 0xffffffffu;
          partial_scores[out_idx] = -INFINITY;
        }
      }
      break;
    }
    if (((win_i >> 2u) & 255u) == tid) {
      ++popped;
      if (popped < 4u) {
        v0 = v1;
        i0 = i1;
        v1 = v2;
        i1 = i2;
        v2 = v3;
        i2 = i3;
        v3 = -INFINITY;
        i3 = 0xffffffffu;
      } else if (step + 1u < top_k) {
        selector_stage1_scan_thread(row_logits, p.vocab, chunk, tid, true,
                                    win_v, win_i, v0, i0, v1, i1, v2, i2, v3,
                                    i3);
        popped = 0u;
      }
    }
  }
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_selector_topk16_stage2_batched(
    device const uint *partial_ids [[buffer(0)]],
    device const float *partial_scores [[buffer(1)]],
    device uint *ids_out [[buffer(2)]],
    device float *scores_out [[buffer(3)]],
    constant SelectorParams *ps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint slot = tg.y / 7u;
  const uint local_row = (tg.y - slot * 7u) + 1u;
  const SelectorParams p = ps[slot];
  if (local_row >= p.width || p.top_k == 0u)
    return;
  const uint top_k = min(p.top_k, 32u);
  threadgroup float sg_v[2][2];
  threadgroup uint sg_i[2][2];

  uint cursor = 0u;
  const uint global_row = slot * 8u + local_row;
  const ulong row_base = ulong(global_row) * p.top_k;
  uint cur_i = partial_ids[row_base * 64ul + tid];
  float cur_v = partial_scores[row_base * 64ul + tid];

  for (uint step = 0u; step < top_k; ++step) {
    const uint buf = step & 1u;
    float rv = cur_v;
    uint ri = cur_i;
#pragma clang loop unroll(full)
    for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
      const float ov = simd_shuffle_down(rv, offset);
      const uint oi = simd_shuffle_down(ri, offset);
      if (selector_better(ov, oi, rv, ri)) {
        rv = ov;
        ri = oi;
      }
    }
    if (lane == 0u) {
      sg_v[buf][sg] = rv;
      sg_i[buf][sg] = ri;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float win_v = sg_v[buf][0];
    uint win_i = sg_i[buf][0];
    if (selector_better(sg_v[buf][1], sg_i[buf][1], win_v, win_i)) {
      win_v = sg_v[buf][1];
      win_i = sg_i[buf][1];
    }

    if (tid == 0u) {
      ids_out[row_base + step] = win_i;
      scores_out[row_base + step] = win_v;
    }
    if (win_i == 0xffffffffu) {
      if (tid == 0u) {
        for (uint s = step + 1u; s < top_k; ++s) {
          ids_out[row_base + s] = 0xffffffffu;
          scores_out[row_base + s] = -INFINITY;
        }
      }
      break;
    }
    if (cur_i == win_i) {
      ++cursor;
      if (cursor < top_k) {
        const ulong idx = (row_base + cursor) * 64ul + tid;
        cur_i = partial_ids[idx];
        cur_v = partial_scores[idx];
      } else {
        cur_i = 0xffffffffu;
        cur_v = -INFINITY;
      }
    }
  }
}

kernel void omlx_m5_selector_finish_batched(
    device const uint *ids [[buffer(0)]],
    device const float *unary [[buffer(1)]],
    device const float *projected [[buffer(2)]],
    device const bfloat *predecessor [[buffer(3)]],
    device const bfloat *successor [[buffer(4)]],
    device uint *selected [[buffer(5)]],
    device float *probabilities [[buffer(6)]],
    device uint *fallback [[buffer(7)]],
    constant SelectorParams *ps [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint slot = tg.y;
  constant SelectorParams &p = ps[slot];
  if (p.vocab == 0u || p.top_k == 0u) {
    if (tid == 0u) {
      fallback[slot] = 1u;
    }
    return;
  }
  device const uint *s_ids = ids + ulong(slot * 8u) * p.top_k;
  device const float *s_unary = unary + ulong(slot * 8u) * p.top_k;
  device const float *s_projected = projected + ulong(slot * 8u) * p.rank;
  device uint *s_selected = selected + ulong(slot * 8u);
  device float *s_probs = probabilities + ulong(slot * 8u) * p.top_k;
  threadgroup float tg_energies[32];
  threadgroup float tg_values[32];
  threadgroup float tg_pred_proj[512];
  threadgroup uint tg_previous;
  if (tid == 0u) {
    fallback[slot] = (p.anchor >= p.vocab) ? 1u : 0u;
    tg_previous = p.anchor;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint threads_in_tg = tptg.x * tptg.y * tptg.z;
  const bool cache_pp = (p.rank <= 512u);
  for (uint row = 1u; row < p.width; ++row) {
    const uint previous = tg_previous;
    selector_compute_candidate_energies(
        s_ids, s_unary, s_projected, predecessor, successor, p, row, previous,
        cache_pp, tg_pred_proj, tg_energies, lane, sg, tid, threads_in_tg);
    if (tid == 0u) {
      const ulong row_off = ulong(row) * p.top_k;
      if (previous >= p.vocab) {
        fallback[slot] = 1u;
      }
      float maximum = -INFINITY;
      for (uint candidate = 0u; candidate < p.top_k; ++candidate) {
        if (s_ids[row_off + candidate] < p.vocab) {
          maximum = max(maximum, tg_energies[candidate]);
        }
      }
      uint chosen = 0u;
      if (p.temperature <= 0.0f) {
        for (uint candidate = 1u; candidate < p.top_k; ++candidate) {
          const uint id_c = s_ids[row_off + candidate];
          const uint id_b = s_ids[row_off + chosen];
          if (id_c < p.vocab &&
              (id_b >= p.vocab || tg_energies[candidate] > tg_energies[chosen]))
            chosen = candidate;
        }
        for (uint candidate = 0u; candidate < p.top_k; ++candidate)
          s_probs[row_off + candidate] =
              candidate == chosen ? 1.0f : 0.0f;
      } else {
        float total = 0.0f;
        for (uint candidate = 0u; candidate < p.top_k; ++candidate) {
          const bool valid_c =
              s_ids[row_off + candidate] < p.vocab && isfinite(maximum);
          tg_values[candidate] =
              valid_c ? exp((tg_energies[candidate] - maximum) / p.temperature)
                      : 0.0f;
          total += tg_values[candidate];
        }
        float cumulative = 0.0f, draw = p.uniforms[row - 1u];
        uint picked = p.top_k;
        uint last_valid = 0u;
        bool has_valid = false;
        for (uint candidate = 0u; candidate < p.top_k; ++candidate) {
          const bool valid_c = s_ids[row_off + candidate] < p.vocab;
          const float probability =
              (valid_c && total > 0.0f && isfinite(total))
                  ? (tg_values[candidate] / total)
                  : 0.0f;
          s_probs[row_off + candidate] = probability;
          cumulative += probability;
          if (valid_c &&
              (fabs(draw - cumulative) < 1e-4f || !isfinite(probability)))
            fallback[slot] = 1u;
          if (valid_c) {
            last_valid = candidate;
            has_valid = true;
            if (picked == p.top_k && draw < cumulative)
              picked = candidate;
          }
        }
        if (!has_valid || !(total > 0.0f) || !isfinite(total))
          fallback[slot] = 1u;
        chosen = (picked < p.top_k) ? picked : last_valid;
      }
      uint next_token = s_ids[row_off + chosen];
      if (next_token >= p.vocab || !isfinite(tg_energies[chosen])) {
        fallback[slot] = 1u;
        if (next_token >= p.vocab) {
          next_token = 0u;
        }
      }
      tg_previous = next_token;
      s_selected[row] = next_token;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void omlx_m5_selector_finish_position_batched(
    device const uint *ids [[buffer(0)]],
    device const float *unary [[buffer(1)]],
    device const float *projected [[buffer(2)]],
    device const bfloat *predecessor [[buffer(3)]],
    device const bfloat *successor [[buffer(4)]],
    device uint *selected [[buffer(5)]],
    device float *probabilities [[buffer(6)]],
    device uint *fallback [[buffer(7)]],
    constant SelectorParams *ps [[buffer(8)]],
    constant SelectorPositionKey *keys [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint slot = tg.y;
  constant SelectorParams &p = ps[slot];
  if (p.vocab == 0u || p.top_k == 0u) {
    if (tid == 0u) {
      fallback[slot] = 1u;
    }
    return;
  }
  constant SelectorPositionKey &key = keys[slot];
  device const uint *s_ids = ids + ulong(slot * 8u) * p.top_k;
  device const float *s_unary = unary + ulong(slot * 8u) * p.top_k;
  device const float *s_projected = projected + ulong(slot * 8u) * p.rank;
  device uint *s_selected = selected + ulong(slot * 8u);
  device float *s_probs = probabilities + ulong(slot * 8u) * p.top_k;
  threadgroup float tg_energies[32];
  threadgroup float tg_values[32];
  threadgroup float tg_pred_proj[512];
  threadgroup uint tg_previous;
  if (tid == 0u) {
    fallback[slot] = (p.anchor >= p.vocab) ? 1u : 0u;
    tg_previous = p.anchor;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint threads_in_tg = tptg.x * tptg.y * tptg.z;
  const bool cache_pp = (p.rank <= 512u);
  for (uint row = 1u; row < p.width; ++row) {
    const uint previous = tg_previous;
    selector_compute_candidate_energies(
        s_ids, s_unary, s_projected, predecessor, successor, p, row, previous,
        cache_pp, tg_pred_proj, tg_energies, lane, sg, tid, threads_in_tg);
    if (tid == 0u) {
      const ulong row_off = ulong(row) * p.top_k;
      if (previous >= p.vocab) {
        fallback[slot] = 1u;
      }
      float maximum = -INFINITY;
      for (uint i = 0u; i < p.top_k; ++i) {
        const uint id_i = s_ids[row_off + i];
        if (id_i >= p.vocab || !isfinite(tg_energies[i]))
          fallback[slot] = 1u;
        if (id_i < p.vocab)
          maximum = max(maximum, tg_energies[i]);
      }
      uint chosen = 0u;
      if (p.temperature <= 0.0f) {
        for (uint i = 1u; i < p.top_k; ++i) {
          const uint id_i = s_ids[row_off + i];
          const uint id_b = s_ids[row_off + chosen];
          if (id_i < p.vocab &&
              (id_b >= p.vocab || tg_energies[i] > tg_energies[chosen] ||
               (tg_energies[i] == tg_energies[chosen] && id_i < id_b)))
            chosen = i;
        }
        for (uint i = 0u; i < p.top_k; ++i)
          s_probs[row_off + i] = i == chosen ? 1.0f : 0.0f;
      } else {
        float total = 0.0f;
        for (uint i = 0u; i < p.top_k; ++i) {
          const bool valid_i = s_ids[row_off + i] < p.vocab && isfinite(maximum);
          tg_values[i] =
              valid_i ? exp((tg_energies[i] - maximum) / p.temperature) : 0.0f;
          total += tg_values[i];
        }
        float best = -INFINITY, runner = -INFINITY;
        bool has_valid = false;
        for (uint i = 0u; i < p.top_k; ++i) {
          const uint id_i = s_ids[row_off + i];
          s_probs[row_off + i] =
              (id_i < p.vocab && total > 0.0f && isfinite(total))
                  ? (tg_values[i] / total)
                  : 0.0f;
          if (id_i >= p.vocab || !isfinite(maximum))
            continue;
          const float score =
              (tg_energies[i] - maximum) / p.temperature +
              position_gumbel(key.seed, key.position + row - 1u, id_i,
                              key.purpose);
          if (!has_valid || score > best ||
              (score == best && id_i < s_ids[row_off + chosen])) {
            runner = best;
            best = score;
            chosen = i;
            has_valid = true;
          } else {
            runner = max(runner, score);
          }
        }
        if (!has_valid || !(total > 0.0f) || !isfinite(total) ||
            best - runner < 2e-4f)
          fallback[slot] = 1u;
      }
      uint next_token = s_ids[row_off + chosen];
      if (next_token >= p.vocab || !isfinite(tg_energies[chosen])) {
        fallback[slot] = 1u;
        if (next_token >= p.vocab) {
          next_token = 0u;
        }
      }
      tg_previous = next_token;
      s_selected[row] = next_token;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
