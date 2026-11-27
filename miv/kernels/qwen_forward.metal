#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

struct ForwardRowParams {
  uint rows;
  uint width;
};
struct ForwardLinearParams {
  uint n;
  uint k;
  uint padded_k;
  uint tokens;
};
struct FeatureParams {
  uint tokens;
  uint hidden;
  uint taps;
  uint tap;
};
struct FeatureRingParams {
  uint tokens;
  uint hidden;
  uint taps;
  uint tap;
  uint start_position;
  uint ring_rows;
};

inline float qf_decode_e2m1(uchar code) {
  // Pure 32-bit ALU E2M1 decode (avoids Metal 4 shader-validation constant-LUT trap).
  const uint mag = uint(code) & 7u;
  const float m = float((0xC8643210u >> (mag * 4u)) & 0xFu) * 0.5f;
  return (code & 0x8u) ? -m : m;
}

inline float qf_decode_e8m0(uchar code) {
  if (code == 0xFFu)
    return NAN;
  if (code == 0u)
    return ldexp(1.0f, -127);
  return ldexp(1.0f, int(code) - 127);
}

inline float qf_bf16_bits_to_float(ushort bits) {
  return as_type<float>(uint(bits) << 16u);
}

kernel void omlx_m5_forward_capture_routes(device const uint *ids [[buffer(0)]],
                                           device uint *trace [[buffer(1)]],
                                           constant FeatureParams &p
                                           [[buffer(2)]],
                                           uint i [[thread_position_in_grid]]) {
  if (ulong(i) < ulong(p.tokens) * p.hidden)
    trace[((ulong(i / p.hidden) * p.taps + p.tap) * p.hidden) + i % p.hidden] =
        ids[i];
}

// Normalized top-k softmax can select on logits: softmax is monotone and
// its global denominator cancels when the selected scores are renormalized.
// Avoid rounding all 256 probabilities to BF16 before the discrete selection.
kernel void omlx_m5_forward_router_top8_logits(
    device const bfloat *logits [[buffer(0)]],
    device uint *indices [[buffer(1)]], device bfloat *scores [[buffer(2)]],
    constant ForwardRowParams &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  if (group.y >= p.rows)
    return;
  device const ushort4 *lv = reinterpret_cast<device const ushort4 *>(
      logits + ulong(group.y) * 256ul + lane * 8ul);
  const ushort4 b0 = lv[0];
  const ushort4 b1 = lv[1];
  const ushort raw[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
  bool local_nan = false;
#pragma clang loop unroll(full)
  for (uint i = 0; i < 8; ++i) {
    local_nan |= ((raw[i] & 0x7FFFu) > 0x7F80u);
  }
  float selected[8];
  uint ids[8];
  if (!simd_any(local_nan)) {
    uint packed[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
      const uint bf = ((raw[i] & 0x7FFFu) == 0u) ? 0u : uint(raw[i]);
      const uint key16 =
          (bf & 0x8000u) ? ((~bf) & 0xFFFFu) : (bf | 0x8000u);
      packed[i] = (key16 << 16u) | 0x0100u | (lane * 8u + i);
    }
#pragma clang loop unroll(full)
    for (uint k = 0; k < 8; ++k) {
      uint best_p = max(max(max(packed[0], packed[1]), max(packed[2], packed[3])),
                        max(max(packed[4], packed[5]), max(packed[6], packed[7])));
      const uint win_p = simd_max(best_p);
      const uint winner = win_p & 0xFFu;
      const uint win_key16 = win_p >> 16u;
      const uint orig_bf = (win_key16 & 0x8000u) ? (win_key16 & 0x7FFFu)
                                                 : ((~win_key16) & 0xFFFFu);
      selected[k] = as_type<float>(orig_bf << 16u);
      ids[k] = winner;
      if ((winner >> 3u) == lane) {
        packed[winner & 7u] = 0u;
      }
    }
  } else {
    float values[8];
    bool taken[8];
    for (uint i = 0; i < 8; ++i) {
      values[i] = as_type<float>(uint(raw[i]) << 16u);
      taken[i] = false;
    }
    for (uint k = 0; k < 8; ++k) {
      int tier = -1;
      float best = -INFINITY;
      uint best_local = 0;
      for (uint i = 0; i < 8; ++i) {
        if (taken[i])
          continue;
        const int candidate_tier = isnan(values[i]) ? 0 : 1;
        const float candidate_val = candidate_tier == 1 ? values[i] : -INFINITY;
        if (candidate_tier > tier ||
            (candidate_tier == tier && candidate_val >= best)) {
          tier = candidate_tier;
          best = candidate_val;
          best_local = i;
        }
      }
      const int global_tier = simd_max(tier);
      if (global_tier < 0) {
        selected[k] = -INFINITY;
        ids[k] = 0u;
        continue;
      }
      const bool in_tier = (tier == global_tier);
      const float maximum = simd_max(in_tier ? best : -INFINITY);
      const bool is_best = in_tier && (best == maximum);
      const uint id = lane * 8 + best_local;
      const uint winner = simd_max(is_best ? id : 0u);
      if (is_best && id == winner)
        taken[best_local] = true;
      selected[k] = global_tier == 1 ? maximum : -INFINITY;
      ids[k] = winner;
    }
  }
  if (lane == 0) {
    float sum = 0.0f;
    float probabilities[8];
    const float max_logit = selected[0];
    for (uint k = 0; k < 8; ++k) {
      if (max_logit == -INFINITY || isnan(max_logit) || isnan(selected[k])) {
        probabilities[k] = 0.0f;
      } else if (max_logit == INFINITY) {
        probabilities[k] = selected[k] == INFINITY ? 1.0f : 0.0f;
      } else {
        probabilities[k] =
            selected[k] == -INFINITY ? 0.0f : exp(selected[k] - max_logit);
      }
      sum += probabilities[k];
    }
    for (uint k = 0; k < 8; ++k) {
      indices[ulong(group.y) * 8 + k] = ids[k];
      scores[ulong(group.y) * 8 + k] =
          sum > 0.0f && isfinite(sum) ? bfloat(probabilities[k] / sum)
                                      : bfloat(0.0f);
    }
  }
}

kernel void omlx_m5_forward_capture(device const bfloat *hidden [[buffer(0)]],
                                    device bfloat *features [[buffer(1)]],
                                    constant FeatureParams &p [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
  if (ulong(i) < ulong(p.tokens) * p.hidden)
    features[((ulong(i / p.hidden) * p.taps + p.tap) * p.hidden) +
             i % p.hidden] = hidden[i];
}

// Cyclic ring-buffer feature history writer: writes token r to
// (start_position + r) % ring_rows.
struct CopyRingRowsParams {
  uint tokens;
  uint width;
  uint start_position;
  uint ring_rows;
};

kernel void omlx_m5_forward_copy_rows_ring(
    device const bfloat *source [[buffer(0)]],
    device bfloat *destination [[buffer(1)]],
    constant CopyRingRowsParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  const ulong total = ulong(p.tokens) * p.width;
  if (ulong(i) >= total || p.ring_rows == 0u)
    return;
  const uint row = i / p.width;
  const uint col = i % p.width;
  const uint dst_row = (p.start_position + row) % p.ring_rows;
  destination[ulong(dst_row) * p.width + col] = source[i];
}

// Split a concatenated [tokens, n0 + n1] matrix into [tokens, n0] and [tokens, n1].
struct Split2ColsParams {
  uint tokens;
  uint n0;
  uint n1;
};

kernel void omlx_m5_forward_split2_cols(
    device const bfloat *src [[buffer(0)]],
    device bfloat *dst0 [[buffer(1)]],
    device bfloat *dst1 [[buffer(2)]],
    constant Split2ColsParams &p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  const uint stride = p.n0 + p.n1;
  const ulong total = ulong(p.tokens) * stride;
  if (ulong(i) >= total)
    return;
  const uint row = i / stride;
  const uint col = i % stride;
  if (col < p.n0)
    dst0[ulong(row) * p.n0 + col] = src[i];
  else
    dst1[ulong(row) * p.n1 + (col - p.n0)] = src[i];
}

struct Split3ColsParams {
  uint tokens;
  uint n0;
  uint n1;
  uint n2;
};

kernel void omlx_m5_forward_split3_cols(
    device const bfloat *src [[buffer(0)]],
    device bfloat *dst0 [[buffer(1)]],
    device bfloat *dst1 [[buffer(2)]],
    device bfloat *dst2 [[buffer(3)]],
    constant Split3ColsParams &p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  const uint stride = p.n0 + p.n1 + p.n2;
  const ulong total = ulong(p.tokens) * stride;
  if (ulong(i) >= total)
    return;
  const uint row = i / stride;
  const uint col = i % stride;
  if (col < p.n0)
    dst0[ulong(row) * p.n0 + col] = src[i];
  else if (col < p.n0 + p.n1)
    dst1[ulong(row) * p.n1 + (col - p.n0)] = src[i];
  else
    dst2[ulong(row) * p.n2 + (col - p.n0 - p.n1)] = src[i];
}

kernel void omlx_m5_forward_add(device bfloat *residual [[buffer(0)]],
                                device const bfloat *delta [[buffer(1)]],
                                constant ForwardRowParams &p [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
  if (ulong(i) < ulong(p.rows) * p.width)
    residual[i] = bfloat(float(residual[i]) + float(delta[i]));
}

kernel void omlx_m5_forward_last_row(device const bfloat *rows [[buffer(0)]],
                                     device bfloat *last [[buffer(1)]],
                                     constant ForwardRowParams &p [[buffer(2)]],
                                     uint i [[thread_position_in_grid]]) {
  if (p.rows > 0 && i < p.width)
    last[i] = rows[ulong(p.rows - 1) * p.width + i];
}

// Direct BF16 parameters (e.g. protected projections) retain their stored
// dtype.
kernel void
omlx_m5_forward_bf16_linear(device const bfloat *x [[buffer(0)]],
                            device const bfloat *w [[buffer(1)]],
                            device bfloat *y [[buffer(2)]],
                            constant ForwardLinearParams &p [[buffer(3)]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint3 group [[threadgroup_position_in_grid]]) {
  uint row = group.x * 4 + sg;
  uint token = group.y;
  if (row >= p.n || token >= p.tokens)
    return;
  const ulong w_stride = ulong(p.padded_k ? p.padded_k : p.k);
  float sum = 0.0f;
  for (uint k = lane; k < p.k; k += 32)
    sum = fma(float(x[ulong(token) * p.k + k]),
              float(w[ulong(row) * w_stride + k]), sum);
  sum = simd_sum(sum);
  if (lane == 0)
    y[ulong(token) * p.n + row] = bfloat(sum);
}

// Portable row-exact SIMD32 kernels for 3-bit oQ affine (G64 & G128).
template <uint GROUP_SIZE>
inline void forward_q3_linear_impl(
    device const bfloat *x,
    device const uchar *low_2bit,
    device const uchar *high_bit,
    device const ushort *scales,
    device const ushort *biases,
    device bfloat *y,
    constant ForwardLinearParams &p,
    uint lane,
    uint sg,
    uint3 group) {
  const uint row = group.x * 4u + sg;
  const uint token = group.y;
  if (row >= p.n || token >= p.tokens)
    return;
  const ulong low_stride = ulong(p.padded_k / 4u);
  const ulong high_stride = ulong(p.padded_k / 8u);
  const ulong sb_stride = ulong(p.padded_k / GROUP_SIZE);
  const uint num_groups = p.k / GROUP_SIZE;

  float total = 0.0f;
  for (uint g = 0; g < num_groups; ++g) {
    float g_dot = 0.0f;
    float g_xsum = 0.0f;
    for (uint off = lane; off < GROUP_SIZE; off += 32u) {
      const uint col = g * GROUP_SIZE + off;
      const float xv = float(x[ulong(token) * p.k + col]);
      const uchar lb = low_2bit[ulong(row) * low_stride + col / 4u];
      const uchar hb = high_bit[ulong(row) * high_stride + col / 8u];
      const uint q = ((uint(lb) >> ((col & 3u) * 2u)) & 3u) |
                     (((uint(hb) >> (col & 7u)) & 1u) << 2u);
      g_dot = fma(xv, float(q), g_dot);
      g_xsum += xv;
    }
    g_dot = simd_sum(g_dot);
    g_xsum = simd_sum(g_xsum);
    if (lane == 0u) {
      const float sc = qf_bf16_bits_to_float(scales[ulong(row) * sb_stride + g]);
      const float bi = qf_bf16_bits_to_float(biases[ulong(row) * sb_stride + g]);
      total = fma(sc, g_dot, fma(bi, g_xsum, total));
    }
  }
  if (lane == 0u) {
    y[ulong(token) * p.n + row] = bfloat(total);
  }
}

kernel void omlx_m5_forward_q3_g64_linear(
    device const bfloat *x [[buffer(0)]],
    device const uchar *low_2bit [[buffer(1)]],
    device const uchar *high_bit [[buffer(2)]],
    device const ushort *scales [[buffer(3)]],
    device const ushort *biases [[buffer(4)]],
    device bfloat *y [[buffer(5)]],
    constant ForwardLinearParams &p [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  forward_q3_linear_impl<64u>(x, low_2bit, high_bit, scales, biases, y, p, lane,
                              sg, group);
}

kernel void omlx_m5_forward_q3_g128_linear(
    device const bfloat *x [[buffer(0)]],
    device const uchar *low_2bit [[buffer(1)]],
    device const uchar *high_bit [[buffer(2)]],
    device const ushort *scales [[buffer(3)]],
    device const ushort *biases [[buffer(4)]],
    device bfloat *y [[buffer(5)]],
    constant ForwardLinearParams &p [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  forward_q3_linear_impl<128u>(x, low_2bit, high_bit, scales, biases, y, p,
                               lane, sg, group);
}

struct ForwardEmbedParams {
  uint vocab;
  uint reserved0;
  uint rows;
  uint hidden;
  uint padded_k;
  uint tokens;
  uint reserved1;
  uint reserved2;
};

template <uint GROUP_SIZE>
inline void forward_embedding_q3_impl(
    device const uint *token_ids,
    device const uchar *low_2bit,
    device const uchar *high_bit,
    device const ushort *scales,
    device const ushort *biases,
    device bfloat *hidden,
    constant ForwardEmbedParams &p,
    uint gid) {
  const ulong total = ulong(p.tokens) * p.hidden;
  if (ulong(gid) >= total)
    return;
  const uint t = gid / p.hidden;
  const uint col = gid % p.hidden;
  const uint tok = token_ids[t];
  if (tok >= p.vocab) {
    hidden[gid] = bfloat(0.0f);
    return;
  }
  const ulong low_stride = ulong(p.padded_k / 4u);
  const ulong high_stride = ulong(p.padded_k / 8u);
  const ulong sb_stride = ulong(p.padded_k / GROUP_SIZE);
  const uchar lb = low_2bit[ulong(tok) * low_stride + col / 4u];
  const uchar hb = high_bit[ulong(tok) * high_stride + col / 8u];
  const uint q = ((uint(lb) >> ((col & 3u) * 2u)) & 3u) |
                 (((uint(hb) >> (col & 7u)) & 1u) << 2u);
  const uint g = col / GROUP_SIZE;
  const float sc = qf_bf16_bits_to_float(scales[ulong(tok) * sb_stride + g]);
  const float bi = qf_bf16_bits_to_float(biases[ulong(tok) * sb_stride + g]);
  hidden[gid] = bfloat(fma(sc, float(q), bi));
}

kernel void omlx_m5_forward_embedding_q3_g64(
    device const uint *token_ids [[buffer(0)]],
    device const uchar *low_2bit [[buffer(1)]],
    device const uchar *high_bit [[buffer(2)]],
    device const ushort *scales [[buffer(3)]],
    device const ushort *biases [[buffer(4)]],
    device bfloat *hidden [[buffer(5)]],
    constant ForwardEmbedParams &p [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  forward_embedding_q3_impl<64u>(token_ids, low_2bit, high_bit, scales, biases,
                                 hidden, p, gid);
}

kernel void omlx_m5_forward_embedding_q3_g128(
    device const uint *token_ids [[buffer(0)]],
    device const uchar *low_2bit [[buffer(1)]],
    device const uchar *high_bit [[buffer(2)]],
    device const ushort *scales [[buffer(3)]],
    device const ushort *biases [[buffer(4)]],
    device bfloat *hidden [[buffer(5)]],
    constant ForwardEmbedParams &p [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  forward_embedding_q3_impl<128u>(token_ids, low_2bit, high_bit, scales, biases,
                                  hidden, p, gid);
}

#include "fwht64.h"

// Bug #1 portable SIMD32 fallback kernels directly reading pre-tiled NAX planes
// (mx_nax_tiled_bk128_v1 and oq_nax_{tiled,lane}_g{64,128}_v{1,2}) on non-NAX dev hosts (M4)
// or T-domain / SharedHeadProjection fallbacks.
kernel void omlx_m5_forward_pretiled_mxfp4_linear(
    device const bfloat *x [[buffer(0)]],
    device const uchar *nax_data [[buffer(1)]],
    device const uchar *nax_scale [[buffer(2)]],
    device bfloat *y [[buffer(3)]],
    constant ForwardLinearParams &p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  const uint row = group.x * 4u + sg;
  const uint token = group.y;
  if (row >= p.n || token >= p.tokens)
    return;
  const bool fwht64 = ((p.padded_k >> 31u) != 0u);
  const uint bn = (p.n >= 65536u) ? 32u : 16u;
  const uint k_blocks = p.k / 128u;
  const uint rn = row % bn;
  const uint tile_row = row / bn;

  float total = 0.0f;
  for (uint blk = 0; blk < k_blocks; ++blk) {
    const ulong tile_idx = ulong(tile_row) * k_blocks + blk;
    device const uchar *d_row =
        nax_data + tile_idx * (ulong(bn) * 64ul) + ulong(rn) * 64ul;
    device const uchar *s_row =
        nax_scale + tile_idx * (ulong(bn) * 4ul) + ulong(rn) * 4ul;
    if (!fwht64) {
      for (uint sub = 0; sub < 4u; ++sub) {
        const uint col = blk * 128u + sub * 32u + lane;
        const uchar packed = d_row[sub * 16u + (lane >> 1u)];
        const uchar code = (packed >> ((lane & 1u) * 4u)) & 0xFu;
        const float wv = qf_decode_e2m1(code) * qf_decode_e8m0(s_row[sub]);
        const float xv = float(x[ulong(token) * p.k + col]);
        total = fma(xv, wv, total);
      }
    } else {
      for (uint h64 = 0u; h64 < 2u; ++h64) {
        const uint base_k = blk * 128u + h64 * 64u;
        float v0 = float(x[ulong(token) * p.k + base_k + lane * 2u + 0u]);
        float v1 = float(x[ulong(token) * p.k + base_k + lane * 2u + 1u]);
        omlx_m5_fwht64_sg32(v0, v1, lane);
        const uint sub = h64 * 2u + (lane >> 4u);
        const float sc = qf_decode_e8m0(s_row[sub]);
        const uchar packed = d_row[h64 * 32u + lane];
        const float w0 = qf_decode_e2m1(packed & 0xFu) * sc;
        const float w1 = qf_decode_e2m1((packed >> 4u) & 0xFu) * sc;
        total = fma(v0, w0, fma(v1, w1, total));
      }
    }
  }
  total = simd_sum(total);
  if (lane == 0u) {
    y[ulong(token) * p.n + row] = bfloat(total);
  }
}

struct PretiledAffineParams {
  uint n;
  uint k;
  uint bits;
  uint tokens;
  uint group_size;
};

kernel void omlx_m5_forward_pretiled_affine_linear(
    device const bfloat *x [[buffer(0)]],
    device const uchar *nax_codes [[buffer(1)]],
    device const uchar *nax_high [[buffer(2)]],
    device const uint *nax_sb [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant PretiledAffineParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  const uint row = group.x * 4u + sg;
  const uint token = group.y;
  if (row >= p.n || token >= p.tokens)
    return;
  const uint bits = p.bits & 0xFFu;
  const bool is_lane_v2 = ((p.bits & (1u << 30u)) != 0u);
  const bool fwht64 = ((p.bits & (1u << 31u)) != 0u);
  const uint bn = (p.n >= 65536u) ? 32u : 16u;
  const uint padded_n = ((p.n + bn - 1u) / bn) * bn;
  const uint groups = p.k / 64u;
  const uint rn = row % bn;
  const uint tile_row = row / bn;

  float total = 0.0f;
  for (uint g = 0; g < groups; ++g) {
    const ulong tile_idx = ulong(tile_row) * groups + g;
    float v0 = float(x[ulong(token) * p.k + g * 64u + lane * 2u + 0u]);
    float v1 = float(x[ulong(token) * p.k + g * 64u + lane * 2u + 1u]);
    if (fwht64) {
      omlx_m5_fwht64_sg32(v0, v1, lane);
    }
    float g_dot = 0.0f;
    float g_xsum = v0 + v1;
    #pragma clang loop unroll(full)
    for (uint e = 0u; e < 2u; ++e) {
      const uint off = lane * 2u + e;
      const uint slot = is_lane_v2 ? omlx_m5_pi64(off) : off;
      const float xv = (e == 0u) ? v0 : v1;
      uint q = 0u;
      if (bits == 3u) {
        const uchar lb =
            nax_codes[tile_idx * (ulong(bn) * 16ul) + ulong(rn) * 16ul + slot / 4u];
        const uchar hb =
            nax_high[tile_idx * (ulong(bn) * 8ul) + ulong(rn) * 8ul + slot / 8u];
        q = ((uint(lb) >> ((slot & 3u) * 2u)) & 3u) |
            (((uint(hb) >> (slot & 7u)) & 1u) << 2u);
      } else if (bits == 8u) {
        q = uint(
            nax_codes[tile_idx * (ulong(bn) * 64ul) + ulong(rn) * 64ul + slot]);
      } else {
        const uchar cb =
            nax_codes[tile_idx * (ulong(bn) * 32ul) + ulong(rn) * 32ul + slot / 2u];
        q = (uint(cb) >> ((slot & 1u) * 4u)) & 0xFu;
        if (bits == 5u || bits == 6u) {
          const uchar hb =
              nax_high[tile_idx * (ulong(bn) * 16ul) + ulong(rn) * 16ul + slot / 4u];
          const uint h2 = (uint(hb) >> ((slot & 3u) * 2u)) & 3u;
          q |= (bits == 5u ? (h2 & 1u) : h2) << 4u;
        }
      }
      g_dot = fma(xv, float(q), g_dot);
    }
    g_dot = simd_sum(g_dot);
    g_xsum = simd_sum(g_xsum);
    if (lane == 0u) {
      const uint sb_g = (p.group_size == 128u) ? (g >> 1u) : g;
      const uint packed_sb = nax_sb[ulong(sb_g) * padded_n + row];
      const float sc = qf_bf16_bits_to_float(ushort(packed_sb & 0xFFFFu));
      const float bi = qf_bf16_bits_to_float(ushort(packed_sb >> 16u));
      total = fma(sc, g_dot, fma(bi, g_xsum, total));
    }
  }
  if (lane == 0u) {
    y[ulong(token) * p.n + row] = bfloat(total);
  }
}

struct Fwht64Bf16Params {
  uint k;
  uint reserved0;
  uint reserved1;
  uint rows;
};

// Universal out-of-place or in-place BF16 FWHT-64 rotation (H_64 = (1/8)*H_tilde_64).
// Grid: width = (k / 64) * 32, height = rows * row_mul, threads = 32. 0 B Threadgroup SRAM.
[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_fwht64_bf16(
    device const bfloat *src [[buffer(0)]],
    device bfloat *dst [[buffer(1)]],
    constant Fwht64Bf16Params &p [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g = tg.x;
  const uint row = tg.y;
  const uint groups = p.k / 64u;
  const uint row_mul = (p.reserved0 != 0u) ? p.reserved0 : 1u;
  const uint total_rows = p.rows * row_mul;
  if (g >= groups || row >= total_rows) {
    return;
  }
  const ulong off = ulong(row) * p.k + ulong(g) * 64ul + ulong(lane) * 2ul;
  const bfloat2 in2 = *reinterpret_cast<device const bfloat2 *>(src + off);
  float v0 = float(in2.x);
  float v1 = float(in2.y);
  omlx_m5_fwht64_sg32(v0, v1, lane);
  *reinterpret_cast<device bfloat2 *>(dst + off) = bfloat2(bfloat(v0), bfloat(v1));
}

kernel void omlx_m5_forward_copy_bytes(device const uchar *source [[buffer(0)]],
                                       device uchar *destination [[buffer(1)]],
                                       constant uint &count [[buffer(2)]],
                                       uint i [[thread_position_in_grid]]) {
  const ulong base = ulong(i) * 16ul;
  if (base >= ulong(count))
    return;
  if (base + 16ul <= ulong(count) &&
      ((reinterpret_cast<ulong>(source + base) |
        reinterpret_cast<ulong>(destination + base)) &
       15ul) == 0ul) {
    *reinterpret_cast<device uint4 *>(destination + base) =
        *reinterpret_cast<device const uint4 *>(source + base);
  } else {
    const ulong limit = min(base + 16ul, ulong(count));
    for (ulong pos = base; pos < limit; ++pos)
      destination[pos] = source[pos];
  }
}
struct CopyRows {
  uint tokens, width, position;
};
kernel void omlx_m5_forward_copy_rows(device const bfloat *source [[buffer(0)]],
                                      device bfloat *destination [[buffer(1)]],
                                      constant CopyRows &p [[buffer(2)]],
                                      uint i [[thread_position_in_grid]]) {
  if (ulong(i) < ulong(p.tokens) * p.width)
    destination[ulong(p.position) * p.width + i] = source[i];
}
kernel void omlx_m5_forward_slice_rows(device const bfloat *source [[buffer(0)]],
                                       device bfloat *destination [[buffer(1)]],
                                       constant CopyRows &p [[buffer(2)]],
                                       uint i [[thread_position_in_grid]]) {
  if (ulong(i) < ulong(p.tokens) * p.width)
    destination[i] = source[ulong(p.position) * p.width + i];
}

struct QfKvCopyParams {
  uint source_page;
  uint destination_page;
  uint words16_per_page;
};

kernel void omlx_m5_forward_kv_page_copy_words16(
    device ushort *cache_key [[buffer(0)]],
    device ushort *cache_value [[buffer(1)]],
    constant QfKvCopyParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.words16_per_page)
    return;
  const ulong src = ulong(p.source_page) * p.words16_per_page + i;
  const ulong dst = ulong(p.destination_page) * p.words16_per_page + i;
  cache_key[dst] = cache_key[src];
  cache_value[dst] = cache_value[src];
}

struct QfPagedAttentionParams {
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

struct QfAffine4VecMeta {
  half scale[4];
  half zero[4];
};

inline void qf_quantize_affine4_vec(device const bfloat *src_vec,
                                    device uchar *codes_head,
                                    device QfAffine4VecMeta *meta_head,
                                    uint page_tokens, uint within,
                                    uint head_dim, uint lane) {
  const uint gs = min(head_dim, 128u);
  const uint groups = head_dim / gs;
  const uint vec4_count = gs / 4u;
  QfAffine4VecMeta meta_local;
  for (uint i = 0; i < 4u; ++i) {
    meta_local.scale[i] = half(0.0h);
    meta_local.zero[i] = half(0.0h);
  }
  constexpr float kHalfMax = 65504.0f;
  for (uint g = 0; g < groups; ++g) {
    float4 v = float4(0.0f);
    float lane_lo = INFINITY;
    float lane_hi = -INFINITY;
    if (lane < vec4_count) {
      device const bfloat4 *src4 =
          reinterpret_cast<device const bfloat4 *>(src_vec + g * gs);
      v = clamp(float4(src4[lane]), float4(-kHalfMax), float4(kHalfMax));
      lane_lo = min(min(v.x, v.y), min(v.z, v.w));
      lane_hi = max(max(v.x, v.y), max(v.z, v.w));
    }
    const float gmin = clamp(simd_min(lane_lo), -kHalfMax, kHalfMax);
    const float gmax = clamp(simd_max(lane_hi), -kHalfMax, kHalfMax);
    const half z_h = half(gmin);
    const float z_f = float(z_h);
    const float span = max(0.0f, gmax - z_f);
    const half s_h = half(min(span * (1.0f / 15.0f), kHalfMax));
    const float s_f = float(s_h);
    const float inv_s = (s_f > 0.0f) ? (1.0f / s_f) : 0.0f;
    if (lane < vec4_count) {
      const uint q0 = uint(clamp(rint((v.x - z_f) * inv_s), 0.0f, 15.0f));
      const uint q1 = uint(clamp(rint((v.y - z_f) * inv_s), 0.0f, 15.0f));
      const uint q2 = uint(clamp(rint((v.z - z_f) * inv_s), 0.0f, 15.0f));
      const uint q3 = uint(clamp(rint((v.w - z_f) * inv_s), 0.0f, 15.0f));
      const ushort packed = ushort(q0 | (q1 << 4u) | (q2 << 8u) | (q3 << 12u));
      device ushort *dst16 = reinterpret_cast<device ushort *>(
          codes_head + (ulong(g) * page_tokens + within) * ulong(gs / 2u));
      dst16[lane] = packed;
    }
    meta_local.scale[g] = s_h;
    meta_local.zero[g] = z_h;
  }
  if (lane == 0u) {
    meta_head[within] = meta_local;
  }
}

kernel void omlx_m5_forward_paged_kv_append_affine4(
    device const bfloat *key [[buffer(0)]],
    device const bfloat *value [[buffer(1)]],
    device uchar *cache_key [[buffer(2)]],
    device uchar *cache_value [[buffer(3)]],
    device const uint *page_table [[buffer(4)]],
    constant QfPagedAttentionParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  const uint row_head = group.y;
  const uint token = row_head / p.key_value_heads;
  if (token >= p.tokens)
    return;
  const uint kvh = row_head - token * p.key_value_heads;
  const uint logical = p.cache_start + token;
  const uint lpage = logical / p.page_tokens;
  const uint within = logical - lpage * p.page_tokens;
  const uint phys = page_table[lpage];
  const ulong vec_bytes = ulong(p.head_dim / 2u) + 16ul;
  const ulong head_bytes = ulong(p.page_tokens) * vec_bytes + 32ul;
  const ulong page_bytes = ulong(p.key_value_heads) * head_bytes;
  const ulong code_bytes = ulong(p.page_tokens) * ulong(p.head_dim / 2u);

  device uchar *kc = cache_key + ulong(phys) * page_bytes + ulong(kvh) * head_bytes;
  device uchar *vc = cache_value + ulong(phys) * page_bytes + ulong(kvh) * head_bytes;
  device QfAffine4VecMeta *km = reinterpret_cast<device QfAffine4VecMeta *>(kc + code_bytes);
  device QfAffine4VecMeta *vm = reinterpret_cast<device QfAffine4VecMeta *>(vc + code_bytes);

  qf_quantize_affine4_vec(
      key + ulong(token) * p.key_token_stride + ulong(kvh) * p.head_dim,
      kc, km, p.page_tokens, within, p.head_dim, lane);
  qf_quantize_affine4_vec(
      value + ulong(token) * p.value_token_stride + ulong(kvh) * p.head_dim,
      vc, vm, p.page_tokens, within, p.head_dim, lane);
}

kernel void omlx_m5_forward_paged_causal_attention_affine4_exact(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const uchar *cache_key [[buffer(2)]],
    device const uchar *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *page_table [[buffer(5)]],
    constant QfPagedAttentionParams &p [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  const uint logical_head = group.y;
  const uint token = logical_head / p.query_heads;
  if (token >= p.tokens)
    return;
  const uint qh = logical_head - token * p.query_heads;
  const uint repeat = p.query_heads / p.key_value_heads;
  const uint kvh = qh / repeat;
  device const bfloat *q_row = query + ulong(token) * p.query_token_stride +
                               ulong(qh) * p.query_head_stride;
  device const bfloat *g_row = gate + ulong(token) * p.gate_token_stride +
                               ulong(qh) * p.gate_head_stride;

  const uint gs = min(p.head_dim, 128u);
  const uint context_tokens = p.cache_start + token + 1u;
  const uint logical_pages = (context_tokens + p.page_tokens - 1u) / p.page_tokens;
  const float scale = rsqrt(float(p.head_dim));
  const ulong vec_bytes = ulong(p.head_dim / 2u) + 16ul;
  const ulong head_bytes = ulong(p.page_tokens) * vec_bytes + 32ul;
  const ulong page_bytes = ulong(p.key_value_heads) * head_bytes;
  const ulong code_bytes = ulong(p.page_tokens) * ulong(p.head_dim / 2u);

  float q_reg[8];
  for (uint idx = 0; idx < 8u; ++idx) {
    const uint d = lane + idx * 32u;
    q_reg[idx] = (d < p.head_dim) ? float(q_row[d]) : 0.0f;
  }

  // Pass 1: exact global max across [0 .. context_tokens)
  float max_score = -INFINITY;
  for (uint pg = 0; pg < logical_pages; ++pg) {
    const uint phys = page_table[pg];
    const uint valid = min(p.page_tokens, context_tokens - pg * p.page_tokens);
    device const uchar *kc = cache_key + ulong(phys) * page_bytes + ulong(kvh) * head_bytes;
    device const QfAffine4VecMeta *km =
        reinterpret_cast<device const QfAffine4VecMeta *>(kc + code_bytes);
    for (uint w = 0; w < valid; ++w) {
      const QfAffine4VecMeta m = km[w];
      float dot = 0.0f;
      for (uint idx = 0; idx < 8u; ++idx) {
        const uint d = lane + idx * 32u;
        if (d < p.head_dim) {
          const uint g = d / gs;
          const uint off = d - g * gs;
          const uchar b = kc[(ulong(g) * p.page_tokens + w) * ulong(gs / 2u) + (off >> 1u)];
          const uint nib = (uint(b) >> ((off & 1u) * 4u)) & 0xFu;
          const float kv = fma(float(m.scale[g]), float(nib), float(m.zero[g]));
          dot = fma(q_reg[idx], kv, dot);
        }
      }
      const float s = simd_sum(dot) * scale;
      max_score = max(max_score, s);
    }
  }

  // Pass 2: ordered accumulation using exact global max -> 0 ULP row-exact across T=1..1024!
  float denom = 0.0f;
  float acc[8];
  for (uint idx = 0; idx < 8u; ++idx) {
    acc[idx] = 0.0f;
  }
  for (uint pg = 0; pg < logical_pages; ++pg) {
    const uint phys = page_table[pg];
    const uint valid = min(p.page_tokens, context_tokens - pg * p.page_tokens);
    device const uchar *kc = cache_key + ulong(phys) * page_bytes + ulong(kvh) * head_bytes;
    device const uchar *vc = cache_value + ulong(phys) * page_bytes + ulong(kvh) * head_bytes;
    device const QfAffine4VecMeta *km =
        reinterpret_cast<device const QfAffine4VecMeta *>(kc + code_bytes);
    device const QfAffine4VecMeta *vm =
        reinterpret_cast<device const QfAffine4VecMeta *>(vc + code_bytes);
    for (uint w = 0; w < valid; ++w) {
      const QfAffine4VecMeta mk = km[w];
      const QfAffine4VecMeta mv = vm[w];
      float dot = 0.0f;
      for (uint idx = 0; idx < 8u; ++idx) {
        const uint d = lane + idx * 32u;
        if (d < p.head_dim) {
          const uint g = d / gs;
          const uint off = d - g * gs;
          const uchar b = kc[(ulong(g) * p.page_tokens + w) * ulong(gs / 2u) + (off >> 1u)];
          const uint nib = (uint(b) >> ((off & 1u) * 4u)) & 0xFu;
          const float kv = fma(float(mk.scale[g]), float(nib), float(mk.zero[g]));
          dot = fma(q_reg[idx], kv, dot);
        }
      }
      const float s = simd_sum(dot) * scale;
      const float prob = (s == max_score) ? 1.0f : exp(s - max_score);
      denom += prob;
      for (uint idx = 0; idx < 8u; ++idx) {
        const uint d = lane + idx * 32u;
        if (d < p.head_dim) {
          const uint g = d / gs;
          const uint off = d - g * gs;
          const uchar b = vc[(ulong(g) * p.page_tokens + w) * ulong(gs / 2u) + (off >> 1u)];
          const uint nib = (uint(b) >> ((off & 1u) * 4u)) & 0xFu;
          const float vv = fma(float(mv.scale[g]), float(nib), float(mv.zero[g]));
          acc[idx] = fma(prob, vv, acc[idx]);
        }
      }
    }
  }

  for (uint idx = 0; idx < 8u; ++idx) {
    const uint d = lane + idx * 32u;
    if (d < p.head_dim) {
      const float gv = float(g_row[d]);
      const float gated =
          (denom > 0.0f ? (acc[idx] / denom) : 0.0f) / (1.0f + exp(-gv));
      output[(ulong(token) * p.query_heads + qh) * p.head_dim + d] =
          bfloat(gated);
    }
  }
}

