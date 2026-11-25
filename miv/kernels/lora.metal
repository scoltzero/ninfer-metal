#include "fp4_codec.h"
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

inline float lora_decode_e8m0(uchar code) {
  if (code == 0xFFu) {
    return NAN;
  }
  if (code == 0u) {
    return ldexp(1.0f, -127);
  }
  return ldexp(1.0f, int(code) - 127);
}

inline float lora_bf16_bits_to_float(ushort bits) {
  return as_type<float>(uint(bits) << 16u);
}

struct LoraMergeParams {
  uint n;
  uint k;
  uint src_padded_k;
  uint padded_n;
  uint bn;        // 16 or 32
  uint base_kind; // 0 = direct_bf16, 1 = mxfp4_untiled, 2 = mxfp4_tiled, 3 = affine_untiled, 4 = affine_tiled
  uint base_bits; // 3, 4, 5, 6, 8
  uint base_gs;   // 64 or 128 (32 for mxfp4)
  uint rank;
  float scale;
};

struct LoraDownParams {
  uint r_out;
  uint k;
  uint mp;
  uint rows; // word 3 (dynamic token patch index)
};

struct LoraUpAccumParams {
  uint n;
  uint r_valid;
  uint r_stride_b;
  uint rows; // word 3 (dynamic token patch index)
  uint tmp_stride;
  uint tmp_col_offset;
  uint y_stride;
  uint y_col_offset;
  float scale;
};

inline float lora_unpack_base_elem(
    device const uchar *p0,
    device const uchar *p1,
    device const uchar *p2,
    device const uchar *p3,
    constant LoraMergeParams &p,
    uint row,
    uint col) {
  if (p.base_kind == 0u) {
    device const bfloat *bf = reinterpret_cast<device const bfloat *>(p0);
    return float(bf[ulong(row) * p.k + col]);
  }
  if (p.base_kind == 1u) {
    // Untiled mxfp4_e2m1_g32_e8m0_v1
    const ulong d_row = ulong(p.src_padded_k / 2u);
    const ulong s_row = ulong(p.src_padded_k / 32u);
    const uchar packed = p0[ulong(row) * d_row + (col >> 1u)];
    const uint code = (uint(packed) >> ((col & 1u) * 4u)) & 0xFu;
    const uchar sc = p2[ulong(row) * s_row + (col >> 5u)];
    return omlx_m5_decode_fp4(code) * lora_decode_e8m0(sc);
  }
  if (p.base_kind == 2u) {
    // Tiled mx_nax_tiled_bk128_v1
    const uint k_blocks = p.k / 128u;
    const uint blk = col / 128u;
    const uint col_in_blk = col % 128u;
    const uint rn = row % p.bn;
    const ulong tile_idx = ulong(row / p.bn) * k_blocks + blk;
    const uchar packed =
        p0[tile_idx * (ulong(p.bn) * 64ul) + ulong(rn) * 64ul + (col_in_blk >> 1u)];
    const uint code = (uint(packed) >> ((col_in_blk & 1u) * 4u)) & 0xFu;
    const uchar sc =
        p2[tile_idx * (ulong(p.bn) * 4ul) + ulong(rn) * 4ul + (col_in_blk >> 5u)];
    return omlx_m5_decode_fp4(code) * lora_decode_e8m0(sc);
  }
  if (p.base_kind == 3u) {
    // Untiled oq_affine_q{3,4,5,6,8}_g{64,128}_bf16_v1
    uint code = 0u;
    if (p.base_bits == 8u) {
      code = uint(p0[ulong(row) * p.src_padded_k + col]);
    } else if (p.base_bits == 3u) {
      const uchar lb = p0[ulong(row) * (p.src_padded_k / 4u) + col / 4u];
      const uchar hb = p1[ulong(row) * (p.src_padded_k / 8u) + col / 8u];
      code = ((uint(lb) >> ((col & 3u) * 2u)) & 0x3u) |
             (((uint(hb) >> (col & 7u)) & 1u) << 2u);
    } else {
      const uchar bb = p0[ulong(row) * (p.src_padded_k / 2u) + col / 2u];
      code = (uint(bb) >> ((col & 1u) * 4u)) & 0xFu;
      if (p.base_bits == 5u) {
        const uchar hb = p1[ulong(row) * (p.src_padded_k / 8u) + col / 8u];
        code |= ((uint(hb) >> (col & 7u)) & 1u) << 4u;
      } else if (p.base_bits == 6u) {
        const uchar hb = p1[ulong(row) * (p.src_padded_k / 4u) + col / 4u];
        code |= ((uint(hb) >> ((col & 3u) * 2u)) & 3u) << 4u;
      }
    }
    const uint gs = (p.base_gs == 128u) ? 128u : 64u;
    const ulong grp_row = ulong(p.src_padded_k / gs);
    const ulong g_idx = ulong(row) * grp_row + ulong(col / gs);
    device const ushort *scales = reinterpret_cast<device const ushort *>(p2);
    device const ushort *biases = reinterpret_cast<device const ushort *>(p3);
    const float sc = lora_bf16_bits_to_float(scales[g_idx]);
    const float bi = lora_bf16_bits_to_float(biases[g_idx]);
    return fma(sc, float(code), bi);
  }
  // Tiled oq_nax_tiled_g64_v1 / g128_v1
  const uint groups = p.k / 64u;
  const uint g = col / 64u;
  const uint off = col % 64u;
  const uint rn = row % p.bn;
  const ulong tile_idx = ulong(row / p.bn) * groups + g;
  uint code = 0u;
  if (p.base_bits == 3u) {
    const uchar lb =
        p0[tile_idx * (ulong(p.bn) * 16ul) + ulong(rn) * 16ul + off / 4u];
    const uchar hb =
        p1[tile_idx * (ulong(p.bn) * 8ul) + ulong(rn) * 8ul + off / 8u];
    code = ((uint(lb) >> ((off & 3u) * 2u)) & 3u) |
           (((uint(hb) >> (off & 7u)) & 1u) << 2u);
  } else if (p.base_bits == 8u) {
    code = uint(p0[tile_idx * (ulong(p.bn) * 64ul) + ulong(rn) * 64ul + off]);
  } else {
    const uchar cb =
        p0[tile_idx * (ulong(p.bn) * 32ul) + ulong(rn) * 32ul + off / 2u];
    code = (uint(cb) >> ((off & 1u) * 4u)) & 0xFu;
    if (p.base_bits == 5u || p.base_bits == 6u) {
      const uchar hb =
          p1[tile_idx * (ulong(p.bn) * 16ul) + ulong(rn) * 16ul + off / 4u];
      const uint h2 = (uint(hb) >> ((off & 3u) * 2u)) & 3u;
      code |= (p.base_bits == 5u ? (h2 & 1u) : h2) << 4u;
    }
  }
  device const uint *nax_sb = reinterpret_cast<device const uint *>(p2);
  const uint packed_sb = nax_sb[ulong(g) * p.padded_n + row];
  const float sc = lora_bf16_bits_to_float(ushort(packed_sb & 0xFFFFu));
  const float bi = lora_bf16_bits_to_float(ushort(packed_sb >> 16u));
  return fma(sc, float(code), bi);
}

// Mode A ("merge"): Load-time UMA GPU fusion W_eff = W_base + scale * (B @ A)
// -> direct quantization & NAX pre-tiling into oq_nax_tiled_g64_v1 q8 + sb_tiled.
[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_lora_merge_to_q8_g64(
    device const uchar *src_p0 [[buffer(0)]],
    device const uchar *src_p1 [[buffer(1)]],
    device const uchar *src_p2 [[buffer(2)]],
    device const uchar *src_p3 [[buffer(3)]],
    device const float *lora_a [[buffer(4)]],
    device const float *lora_b [[buffer(5)]],
    device uchar *dst_nax_codes [[buffer(6)]],
    device uint *dst_nax_sb [[buffer(7)]],
    device uint *dst_sb_tiled [[buffer(8)]],
    constant LoraMergeParams &p [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g = tg.x;
  const uint row = tg.y;
  const uint groups = p.k / 64u;
  if (g >= groups || row >= p.padded_n) {
    return;
  }
  const uint tile_n = row / p.bn;
  const uint rn = row % p.bn;
  const ulong tile_idx = ulong(tile_n) * groups + g;
  const uint off0 = lane * 2u;
  const uint off1 = off0 + 1u;

  if (row >= p.n) {
    dst_nax_codes[tile_idx * (ulong(p.bn) * 64ul) + ulong(rn) * 64ul + off0] = 0u;
    dst_nax_codes[tile_idx * (ulong(p.bn) * 64ul) + ulong(rn) * 64ul + off1] = 0u;
    if (lane == 0u) {
      dst_nax_sb[ulong(g) * p.padded_n + row] = 0u;
      dst_sb_tiled[(ulong(tile_n) * groups + g) * p.bn + rn] = 0u;
    }
    return;
  }

  const uint col0 = g * 64u + off0;
  const uint col1 = g * 64u + off1;
  float w0 = lora_unpack_base_elem(src_p0, src_p1, src_p2, src_p3, p, row, col0);
  float w1 = lora_unpack_base_elem(src_p0, src_p1, src_p2, src_p3, p, row, col1);

  float d0 = 0.0f;
  float d1 = 0.0f;
  device const float *b_row = lora_b + ulong(row) * p.rank;
  for (uint ri = 0u; ri < p.rank; ++ri) {
    const float bv = b_row[ri];
    device const float *a_row = lora_a + ulong(ri) * p.k;
    d0 = fma(bv, a_row[col0], d0);
    d1 = fma(bv, a_row[col1], d1);
  }
  w0 = fma(p.scale, d0, w0);
  w1 = fma(p.scale, d1, w1);

  const float vmin = simd_min(min(w0, w1));
  const float vmax = simd_max(max(w0, w1));
  const bfloat bias_bf = bfloat(vmin);
  const float bias_f = float(bias_bf);
  const float span = max(0.0f, vmax - bias_f);
  const bfloat scale_bf =
      (span > 0.0f) ? bfloat(span * (1.0f / 255.0f)) : bfloat(0.0f);
  const float scale_f = float(scale_bf);
  const float inv_scale = (scale_f > 0.0f) ? (1.0f / scale_f) : 0.0f;
  const uint q0 = uint(clamp(rint((w0 - bias_f) * inv_scale), 0.0f, 255.0f));
  const uint q1 = uint(clamp(rint((w1 - bias_f) * inv_scale), 0.0f, 255.0f));

  dst_nax_codes[tile_idx * (ulong(p.bn) * 64ul) + ulong(rn) * 64ul + off0] =
      uchar(q0);
  dst_nax_codes[tile_idx * (ulong(p.bn) * 64ul) + ulong(rn) * 64ul + off1] =
      uchar(q1);
  if (lane == 0u) {
    const uint packed_sb = uint(as_type<ushort>(scale_bf)) |
                           (uint(as_type<ushort>(bias_bf)) << 16u);
    dst_nax_sb[ulong(g) * p.padded_n + row] = packed_sb;
    dst_sb_tiled[(ulong(tile_n) * groups + g) * p.bn + rn] = packed_sb;
  }
}

// Mode A ("merge") fallback for K < 512 or non-NAX devices:
// writes untiled oq_affine_q8_g64_bf16_v1 (oq_rowsplit_k128_v1: code, scale, bias).
[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_lora_merge_to_q8_rowsplit_g64(
    device const uchar *src_p0 [[buffer(0)]],
    device const uchar *src_p1 [[buffer(1)]],
    device const uchar *src_p2 [[buffer(2)]],
    device const uchar *src_p3 [[buffer(3)]],
    device const float *lora_a [[buffer(4)]],
    device const float *lora_b [[buffer(5)]],
    device uchar *dst_code [[buffer(6)]],
    device ushort *dst_scale [[buffer(7)]],
    device ushort *dst_bias [[buffer(8)]],
    constant LoraMergeParams &p [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g = tg.x;
  const uint row = tg.y;
  const uint dst_padded_k = ((p.k + 127u) / 128u) * 128u;
  const uint padded_groups = dst_padded_k / 64u;
  if (g >= padded_groups || row >= p.n) {
    return;
  }
  const uint off0 = lane * 2u;
  const uint off1 = off0 + 1u;
  const uint col0 = g * 64u + off0;
  const uint col1 = g * 64u + off1;
  if (col0 >= p.k) {
    dst_code[ulong(row) * dst_padded_k + col0] = 0u;
    dst_code[ulong(row) * dst_padded_k + col1] = 0u;
    if (lane == 0u) {
      dst_scale[ulong(row) * padded_groups + g] = 0u;
      dst_bias[ulong(row) * padded_groups + g] = 0u;
    }
    return;
  }

  float w0 = lora_unpack_base_elem(src_p0, src_p1, src_p2, src_p3, p, row, col0);
  float w1 = lora_unpack_base_elem(src_p0, src_p1, src_p2, src_p3, p, row, col1);

  float d0 = 0.0f;
  float d1 = 0.0f;
  device const float *b_row = lora_b + ulong(row) * p.rank;
  for (uint ri = 0u; ri < p.rank; ++ri) {
    const float bv = b_row[ri];
    device const float *a_row = lora_a + ulong(ri) * p.k;
    d0 = fma(bv, a_row[col0], d0);
    d1 = fma(bv, a_row[col1], d1);
  }
  w0 = fma(p.scale, d0, w0);
  w1 = fma(p.scale, d1, w1);

  const float vmin = simd_min(min(w0, w1));
  const float vmax = simd_max(max(w0, w1));
  const bfloat bias_bf = bfloat(vmin);
  const float bias_f = float(bias_bf);
  const float span = max(0.0f, vmax - bias_f);
  const bfloat scale_bf =
      (span > 0.0f) ? bfloat(span * (1.0f / 255.0f)) : bfloat(0.0f);
  const float scale_f = float(scale_bf);
  const float inv_scale = (scale_f > 0.0f) ? (1.0f / scale_f) : 0.0f;
  const uint q0 = uint(clamp(rint((w0 - bias_f) * inv_scale), 0.0f, 255.0f));
  const uint q1 = uint(clamp(rint((w1 - bias_f) * inv_scale), 0.0f, 255.0f));

  dst_code[ulong(row) * dst_padded_k + col0] = uchar(q0);
  dst_code[ulong(row) * dst_padded_k + col1] = uchar(q1);
  if (lane == 0u) {
    dst_scale[ulong(row) * padded_groups + g] = as_type<ushort>(scale_bf);
    dst_bias[ulong(row) * padded_groups + g] = as_type<ushort>(bias_bf);
  }
}

// Mode B ("bypass") Down-Projection for T < 16: tmp [rows, r_out] = x [rows, k] @ A^T [k, r_out]
[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_lora_down_gemv_bf16(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *a [[buffer(1)]],
    device bfloat *tmp [[buffer(2)]],
    constant LoraDownParams &p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint r_idx = tg.x;
  const uint row = tg.y;
  if (r_idx >= p.r_out || row >= p.rows) {
    return;
  }
  device const bfloat *x_row = x + ulong(row) * p.k;
  device const bfloat *a_row = a + ulong(r_idx) * p.k;
  float acc = 0.0f;
  if ((p.k & 3u) == 0u) {
    device const bfloat4 *xv = reinterpret_cast<device const bfloat4 *>(x_row);
    device const bfloat4 *av = reinterpret_cast<device const bfloat4 *>(a_row);
    const uint vecs = p.k >> 2u;
    for (uint v = tid; v < vecs; v += 128u) {
      const float4 xf = float4(xv[v]);
      const float4 af = float4(av[v]);
      acc = fma(xf.x, af.x, acc);
      acc = fma(xf.y, af.y, acc);
      acc = fma(xf.z, af.z, acc);
      acc = fma(xf.w, af.w, acc);
    }
  } else {
    for (uint c = tid; c < p.k; c += 128u) {
      acc = fma(float(x_row[c]), float(a_row[c]), acc);
    }
  }
  acc = simd_sum(acc);
  threadgroup float sg_sums[4];
  if (lane == 0u) {
    sg_sums[sg] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    const float total =
        ((sg_sums[0] + sg_sums[1]) + sg_sums[2]) + sg_sums[3];
    tmp[ulong(row) * p.r_out + r_idx] = bfloat(total);
  }
}

// Mode B ("bypass") Portable Down-Projection GEMM fallback for T >= 16 (non-NAX / K < 256)
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_lora_down_gemm_bf16(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *a [[buffer(1)]],
    device bfloat *tmp [[buffer(2)]],
    constant LoraDownParams &p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint BM = 16u;
  constexpr uint BN = 16u;
  constexpr uint BK = 32u;
  const uint lm = tid >> 4u;  // 0..15
  const uint ln = tid & 15u;  // 0..15
  const uint row = tg.y * BM + lm;
  const uint col = tg.x * BN + ln;

  threadgroup float xs[BM][BK + 1u];
  threadgroup float as_tile[BN][BK + 1u];

  float acc = 0.0f;
  for (uint k0 = 0u; k0 < p.k; k0 += BK) {
    for (uint sub = 0u; sub < 2u; ++sub) {
      const uint kk = ln * 2u + sub;
      const uint gk = k0 + kk;
      xs[lm][kk] =
          (row < p.rows && gk < p.k) ? float(x[ulong(row) * p.k + gk]) : 0.0f;
      const uint a_col = tg.x * BN + lm;
      as_tile[lm][kk] = (a_col < p.r_out && gk < p.k)
                            ? float(a[ulong(a_col) * p.k + gk])
                            : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0u; kk < BK; ++kk) {
      acc = fma(xs[lm][kk], as_tile[ln][kk], acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < p.rows && col < p.r_out) {
    tmp[ulong(row) * p.r_out + col] = bfloat(acc);
  }
}

// Mode B ("bypass") Up-Projection + In-Place Accumulate for T < 16 (and T == 1 fast path):
// y[row * y_stride + y_col_offset + col] += scale * sum_{i=0..r_valid-1} tmp[row, tmp_col_offset + i] * B[col, i]
[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_lora_up_accum_small_t_bf16(
    device const bfloat *tmp [[buffer(0)]],
    device const bfloat *b [[buffer(1)]],
    device bfloat *y [[buffer(2)]],
    constant LoraUpAccumParams &p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if (p.rows == 0u) {
    return;
  }
  const uint col = tg.x * 128u + tid;
  const uint active_rows = min(p.rows, 16u);
  threadgroup float tg_tmp[16u][64u];

  if (p.rows == 1u) {
    float acc0 = 0.0f;
    for (uint r0 = 0u; r0 < p.r_valid; r0 += 64u) {
      const uint chunk = min(64u, p.r_valid - r0);
      if (tid < chunk) {
        tg_tmp[0][tid] = float(tmp[p.tmp_col_offset + r0 + tid]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (col < p.n) {
        device const bfloat *b_row = b + ulong(col) * p.r_stride_b + r0;
        for (uint i = 0u; i < chunk; ++i) {
          acc0 = fma(tg_tmp[0][i], float(b_row[i]), acc0);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (col < p.n) {
      const ulong dst = ulong(p.y_col_offset) + col;
      y[dst] = bfloat(fma(p.scale, acc0, float(y[dst])));
    }
    return;
  }

  float acc[16];
  for (uint r = 0u; r < 16u; ++r) {
    acc[r] = 0.0f;
  }
  for (uint r0 = 0u; r0 < p.r_valid; r0 += 64u) {
    const uint chunk = min(64u, p.r_valid - r0);
    const uint total_items = active_rows * chunk;
    for (uint idx = tid; idx < total_items; idx += 128u) {
      const uint tr = idx / chunk;
      const uint tc = idx % chunk;
      tg_tmp[tr][tc] = float(
          tmp[ulong(tr) * p.tmp_stride + p.tmp_col_offset + r0 + tc]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (col < p.n) {
      device const bfloat *b_row = b + ulong(col) * p.r_stride_b + r0;
      for (uint i = 0u; i < chunk; ++i) {
        const float bv = float(b_row[i]);
        for (uint tr = 0u; tr < active_rows; ++tr) {
          acc[tr] = fma(tg_tmp[tr][i], bv, acc[tr]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (col < p.n) {
    for (uint tr = 0u; tr < active_rows; ++tr) {
      const ulong dst = ulong(tr) * p.y_stride + p.y_col_offset + col;
      y[dst] = bfloat(fma(p.scale, acc[tr], float(y[dst])));
    }
  }
}

// Mode B ("bypass") Portable Tiled Up-Projection + In-Place Accumulate for T >= 16
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_lora_up_accum_tiled_bf16(
    device const bfloat *tmp [[buffer(0)]],
    device const bfloat *b [[buffer(1)]],
    device bfloat *y [[buffer(2)]],
    constant LoraUpAccumParams &p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint BM = 16u;
  constexpr uint BN = 16u;
  constexpr uint BK = 16u;
  const uint lm = tid >> 4u;
  const uint ln = tid & 15u;
  const uint row = tg.y * BM + lm;
  const uint col = tg.x * BN + ln;
  if (tg.y * BM >= p.rows || tg.x * BN >= p.n) {
    return;
  }

  threadgroup float ts[BM][BK + 1u];
  threadgroup float bs[BN][BK + 1u];

  float acc = 0.0f;
  for (uint k0 = 0u; k0 < p.r_valid; k0 += BK) {
    const uint gk = k0 + ln;
    ts[lm][ln] =
        (row < p.rows && gk < p.r_valid)
            ? float(tmp[ulong(row) * p.tmp_stride + p.tmp_col_offset + gk])
            : 0.0f;
    const uint b_col = tg.x * BN + lm;
    bs[lm][ln] =
        (b_col < p.n && gk < p.r_valid)
            ? float(b[ulong(b_col) * p.r_stride_b + gk])
            : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0u; kk < BK; ++kk) {
      acc = fma(ts[lm][kk], bs[ln][kk], acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < p.rows && col < p.n) {
    const ulong dst = ulong(row) * p.y_stride + p.y_col_offset + col;
    y[dst] = bfloat(fma(p.scale, acc, float(y[dst])));
  }
}
