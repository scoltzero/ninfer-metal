#include "common_math.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct AffineNaxRepackParams {
  uint n;
  uint k;
  uint src_padded_k;
  uint padded_n;
  uint bits;       // 3, 4, 5, 6, or 8
  uint group_size; // 64 or 128
  uint dst_row_offset;
};

struct AffineNaxSbCopyParams {
  uint groups;
  uint src_padded_n;
  uint src_n;
  uint dst_padded_n;
  uint dst_row_offset;
};

struct AffineNaxXsumParams {
  uint k;
  uint mp;
  uint reserved;
  uint rows;
};

struct AffineRmsNormXsumParams {
  uint k;
  uint rows;
  uint mp;
  float epsilon;
  float weight_offset;
};

struct AffineNaxParams {
  uint n;
  uint k;
  uint mp;
  uint rows;
  uint padded_n;
  uint sk;
};

struct AffineNaxSliceParams {
  uint n;          // slice column count (N or N - row_offset)
  uint k;
  uint mp;
  uint rows;       // word 3 (dynamic token patch)
  uint padded_n;   // full weight padded_n for sb stride
  uint sk;         // 1
  uint row_offset; // % 64 == 0 (starting output channel in weight & y)
  uint out_stride; // output row stride in elements (0 => row_offset + n)
};

// Fast GPU xsum for 64-element sub-groups:
// xs[g * mp + row] = sum_{i=0..63} float(x[row * k + g * 64 + i]).
// Shared across both g64 and g128 affine NAX weights!
kernel void omlx_m5_affine_nax_xsum_g64(
    device const bfloat *x [[buffer(0)]],
    device float *xs [[buffer(1)]],
    constant AffineNaxXsumParams &p [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g = tg.x;
  const uint row = tg.y;
  const uint groups = p.k / 64u;
  if (g >= groups || row >= p.mp) {
    return;
  }
  if (row >= p.rows) {
    if (lane == 0u) {
      xs[ulong(g) * p.mp + row] = 0.0f;
    }
    return;
  }
  device const bfloat2 *xv = reinterpret_cast<device const bfloat2 *>(
      x + ulong(row) * p.k + ulong(g) * 64u);
  const float2 pair = float2(xv[lane]);
  const float sum = simd_sum(pair.x + pair.y);
  if (lane == 0u) {
    xs[ulong(g) * p.mp + row] = sum;
  }
}

// 0-ULP fused RMSNorm -> BF16 + Affine group-64 xsum (eliminates standalone xsum_g64
// dispatch + barrier while matching omlx_m5_rmsnorm_a16 + omlx_m5_affine_nax_xsum_g64 bit-for-bit).
[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_affine_rmsnorm_xsum_g64(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device bfloat *output_a16 [[buffer(2)]],
    device float *xs [[buffer(3)]],
    constant AffineRmsNormXsumParams &p [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  constexpr uint threads = 128u;
  constexpr uint simdgroups = 4u;
  const uint row = tg_position.y;
  if (row >= p.mp) {
    return;
  }
  const uint groups = p.k / 64u;
  if (row >= p.rows) {
    for (uint g = simdgroup; g < groups; g += simdgroups) {
      if (lane == 0u) {
        xs[ulong(g) * p.mp + row] = 0.0f;
      }
    }
    return;
  }
  const ulong row_base = ulong(row) * p.k;
  device const bfloat *in_row = input + row_base;
  device bfloat *out_row = output_a16 + row_base;

  float square_sum = 0.0f;
  for (uint column = thread_index; column < p.k; column += threads) {
    const float value = float(in_row[column]);
    square_sum = fma(value, value, square_sum);
  }
  square_sum = simd_sum(square_sum);
  threadgroup float simd_sums[4];
  if (lane == 0u) {
    simd_sums[simdgroup] = square_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint index = 0u; index < simdgroups; ++index) {
    total += simd_sums[index];
  }
  const float inv = rsqrt(total / float(p.k) + p.epsilon);
  const float2 off2 = float2(p.weight_offset);
  for (uint g = simdgroup; g < groups; g += simdgroups) {
    const ulong col_off = ulong(g) * 64ul;
    const bfloat2 x2 = reinterpret_cast<device const bfloat2 *>(in_row + col_off)[lane];
    const bfloat2 w2 = reinterpret_cast<device const bfloat2 *>(weight + col_off)[lane];
    const bfloat2 o2 = bfloat2(float2(x2) * inv * (float2(w2) + off2));
    reinterpret_cast<device bfloat2 *>(out_row + col_off)[lane] = o2;
    const float2 pair = float2(o2);
    const float sum = simd_sum(pair.x + pair.y);
    if (lane == 0u) {
      xs[ulong(g) * p.mp + row] = sum;
    }
  }
}

// 0-ULP fused residual Add + RMSNorm -> BF16 + Affine group-64 xsum (matches
// omlx_m5_add_rmsnorm_a16 + omlx_m5_affine_nax_xsum_g64 bit-for-bit).
[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_affine_add_rmsnorm_xsum_g64(
    device bfloat *hidden [[buffer(0)]],
    device const bfloat *delta [[buffer(1)]],
    device const bfloat *weight [[buffer(2)]],
    device bfloat *output_a16 [[buffer(3)]],
    device float *xs [[buffer(4)]],
    constant AffineRmsNormXsumParams &p [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  constexpr uint threads = 128u;
  constexpr uint simdgroups = 4u;
  const uint row = tg_position.y;
  if (row >= p.mp) {
    return;
  }
  const uint groups = p.k / 64u;
  if (row >= p.rows) {
    for (uint g = simdgroup; g < groups; g += simdgroups) {
      if (lane == 0u) {
        xs[ulong(g) * p.mp + row] = 0.0f;
      }
    }
    return;
  }
  const ulong row_base = ulong(row) * p.k;
  device bfloat *h_row = hidden + row_base;
  device const bfloat *d_row = delta + row_base;
  device bfloat *o_row = output_a16 + row_base;

  threadgroup float simd_sums[4];
  float square_sum = 0.0f;
  float inv = 1.0f;
  if (p.k == 5120u) {
    device bfloat4 *hv = reinterpret_cast<device bfloat4 *>(h_row);
    device const bfloat4 *dv = reinterpret_cast<device const bfloat4 *>(d_row);
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 10u; ++i) {
      const uint v = thread_index + i * threads;
      const bfloat4 s4 = bfloat4(float4(hv[v]) + float4(dv[v]));
      hv[v] = s4;
      const float4 sf = float4(s4);
      square_sum = fma(sf.x, sf.x, square_sum);
      square_sum = fma(sf.y, sf.y, square_sum);
      square_sum = fma(sf.z, sf.z, square_sum);
      square_sum = fma(sf.w, sf.w, square_sum);
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0u) simd_sums[simdgroup] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    inv = rsqrt(total * (1.0f / 5120.0f) + p.epsilon);
  } else if (p.k == 2048u) {
    device bfloat4 *hv = reinterpret_cast<device bfloat4 *>(h_row);
    device const bfloat4 *dv = reinterpret_cast<device const bfloat4 *>(d_row);
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
      const uint v = thread_index + i * threads;
      const bfloat4 s4 = bfloat4(float4(hv[v]) + float4(dv[v]));
      hv[v] = s4;
      const float4 sf = float4(s4);
      square_sum = fma(sf.x, sf.x, square_sum);
      square_sum = fma(sf.y, sf.y, square_sum);
      square_sum = fma(sf.z, sf.z, square_sum);
      square_sum = fma(sf.w, sf.w, square_sum);
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0u) simd_sums[simdgroup] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    inv = rsqrt(total * (1.0f / 2048.0f) + p.epsilon);
  } else if ((p.k & 511u) == 0u) {
    device bfloat4 *hv = reinterpret_cast<device bfloat4 *>(h_row);
    device const bfloat4 *dv = reinterpret_cast<device const bfloat4 *>(d_row);
    const uint vecs = p.k >> 2u;
    for (uint v = thread_index; v < vecs; v += threads) {
      const bfloat4 s4 = bfloat4(float4(hv[v]) + float4(dv[v]));
      hv[v] = s4;
      const float4 sf = float4(s4);
      square_sum = fma(sf.x, sf.x, square_sum);
      square_sum = fma(sf.y, sf.y, square_sum);
      square_sum = fma(sf.z, sf.z, square_sum);
      square_sum = fma(sf.w, sf.w, square_sum);
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0u) simd_sums[simdgroup] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    inv = rsqrt(total / float(p.k) + p.epsilon);
  } else {
    for (uint column = thread_index; column < p.k; column += threads) {
      const bfloat sum = bfloat(float(h_row[column]) + float(d_row[column]));
      h_row[column] = sum;
      const float value = float(sum);
      square_sum = fma(value, value, square_sum);
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0u) simd_sums[simdgroup] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    inv = rsqrt(total / float(p.k) + p.epsilon);
  }

  // Note: Pass 2 reads h_row updated by other threads in this TG; ensure device visibility.
  threadgroup_barrier(mem_flags::mem_device);

  const float2 off2 = float2(p.weight_offset);
  for (uint g = simdgroup; g < groups; g += simdgroups) {
    const ulong col_off = ulong(g) * 64ul;
    const bfloat2 h2 = reinterpret_cast<device const bfloat2 *>(h_row + col_off)[lane];
    const bfloat2 w2 = reinterpret_cast<device const bfloat2 *>(weight + col_off)[lane];
    const bfloat2 o2 = bfloat2(float2(h2) * inv * (float2(w2) + off2));
    reinterpret_cast<device bfloat2 *>(o_row + col_off)[lane] = o2;
    const float2 pair = float2(o2);
    const float sum = simd_sum(pair.x + pair.y);
    if (lane == 0u) {
      xs[ulong(g) * p.mp + row] = sum;
    }
  }
}

// GPU repack: row-major oq_affine_q{3,4,5,6,8}_g{64,128} -> tiled [padded_n/BN, k/64, BN, 64]
// - bits == 3: src_base = low_2bit (k/4 B), src_high = high_bit (k/8 B)
//              -> dst_codes: BN*16 B/tile (uint2b), dst_high2: BN*8 B/tile (1-bit)
// - bits == 4,5,6: src_base = base_code (k/2 B) -> dst_codes: BN*32 B/tile (uint4b)
//                  dst_high2: BN*16 B/tile (uint2b, written for bits==5 or 6)
// - bits == 8: src_base = code (k B) -> dst_codes: BN*64 B/tile (uint8_t)
// - dst_sb: [k/64, padded_n] packed uint (low16 = scale bf16, high16 = bias bf16;
//           repeated across both 64-weight halves when group_size == 128).
template <uint BN>
inline void repack_affine_g64_impl(
    device const uchar *src_base,
    device const uchar *src_high,
    device const ushort *src_scale,
    device const ushort *src_bias,
    device uchar *dst_codes,
    device uchar *dst_high2,
    device uint *dst_sb,
    constant AffineNaxRepackParams &p,
    uint2 gid) {
  const uint g = gid.x;
  const uint local_row = gid.y;
  const uint groups = p.k / 64u;
  const uint slice_rows = min(p.padded_n - p.dst_row_offset,
                              ((p.n + BN - 1u) / BN) * BN);
  if (g >= groups || local_row >= slice_rows) {
    return;
  }
  const uint dst_row = p.dst_row_offset + local_row;
  const ulong tile_idx = ulong(dst_row / BN) * ulong(groups) + ulong(g);
  const uint rn = dst_row % BN;
  const uint gs = (p.group_size == 128u) ? 128u : 64u;

  if (local_row < p.n) {
    const uint sb_g = (gs == 128u) ? (g >> 1u) : g;
    const ulong sb_src = ulong(local_row) * ulong(p.src_padded_k / gs) + ulong(sb_g);
    const uint packed_sb = uint(src_scale[sb_src]) | (uint(src_bias[sb_src]) << 16u);
    dst_sb[ulong(g) * p.padded_n + dst_row] = packed_sb;

    if (p.bits == 3u) {
      device uint4 *dc = reinterpret_cast<device uint4 *>(
          dst_codes + tile_idx * (ulong(BN) * 16ul) + ulong(rn) * 16ul);
      device const uint4 *sc = reinterpret_cast<device const uint4 *>(
          src_base + ulong(local_row) * ulong(p.src_padded_k / 4u) + ulong(g) * 16ul);
      dc[0] = sc[0];
      device uint2 *dh = reinterpret_cast<device uint2 *>(
          dst_high2 + tile_idx * (ulong(BN) * 8ul) + ulong(rn) * 8ul);
      device const uint2 *sh = reinterpret_cast<device const uint2 *>(
          src_high + ulong(local_row) * ulong(p.src_padded_k / 8u) + ulong(g) * 8ul);
      dh[0] = sh[0];
    } else if (p.bits == 8u) {
      device uint4 *dc = reinterpret_cast<device uint4 *>(
          dst_codes + tile_idx * (ulong(BN) * 64ul) + ulong(rn) * 64ul);
      device const uint4 *sc = reinterpret_cast<device const uint4 *>(
          src_base + ulong(local_row) * ulong(p.src_padded_k) + ulong(g) * 64ul);
      dc[0] = sc[0];
      dc[1] = sc[1];
      dc[2] = sc[2];
      dc[3] = sc[3];
    } else {
      device uint4 *dc = reinterpret_cast<device uint4 *>(
          dst_codes + tile_idx * (ulong(BN) * 32ul) + ulong(rn) * 32ul);
      device const uint4 *sc = reinterpret_cast<device const uint4 *>(
          src_base + ulong(local_row) * ulong(p.src_padded_k / 2u) + ulong(g) * 32ul);
      dc[0] = sc[0];
      dc[1] = sc[1];
      if (p.bits == 5u) {
        device const uchar *sh =
            src_high + ulong(local_row) * ulong(p.src_padded_k / 8u) + ulong(g) * 8ul;
        device uchar *dh =
            dst_high2 + tile_idx * (ulong(BN) * 16ul) + ulong(rn) * 16ul;
        for (uint i = 0; i < 8u; ++i) {
          const uint b = uint(sh[i]);
          dh[i * 2u] = expand_4x1bit_to_4x2bit(b & 0x0fu);
          dh[i * 2u + 1u] = expand_4x1bit_to_4x2bit(b >> 4u);
        }
      } else if (p.bits == 6u) {
        device const uint4 *sh = reinterpret_cast<device const uint4 *>(
            src_high + ulong(local_row) * ulong(p.src_padded_k / 4u) + ulong(g) * 16ul);
        device uint4 *dh = reinterpret_cast<device uint4 *>(
            dst_high2 + tile_idx * (ulong(BN) * 16ul) + ulong(rn) * 16ul);
        dh[0] = sh[0];
      }
    }
  } else {
    dst_sb[ulong(g) * p.padded_n + dst_row] = 0u;
    if (p.bits == 3u) {
      device uint4 *dc = reinterpret_cast<device uint4 *>(
          dst_codes + tile_idx * (ulong(BN) * 16ul) + ulong(rn) * 16ul);
      dc[0] = uint4(0u);
      device uint2 *dh = reinterpret_cast<device uint2 *>(
          dst_high2 + tile_idx * (ulong(BN) * 8ul) + ulong(rn) * 8ul);
      dh[0] = uint2(0u);
    } else if (p.bits == 8u) {
      device uint4 *dc = reinterpret_cast<device uint4 *>(
          dst_codes + tile_idx * (ulong(BN) * 64ul) + ulong(rn) * 64ul);
      dc[0] = uint4(0u);
      dc[1] = uint4(0u);
      dc[2] = uint4(0u);
      dc[3] = uint4(0u);
    } else {
      device uint4 *dc = reinterpret_cast<device uint4 *>(
          dst_codes + tile_idx * (ulong(BN) * 32ul) + ulong(rn) * 32ul);
      dc[0] = uint4(0u);
      dc[1] = uint4(0u);
      if (p.bits == 5u || p.bits == 6u) {
        device uint4 *dh = reinterpret_cast<device uint4 *>(
            dst_high2 + tile_idx * (ulong(BN) * 16ul) + ulong(rn) * 16ul);
        dh[0] = uint4(0u);
      }
    }
  }
}

kernel void omlx_m5_repack_affine_g64_bn16(
    device const uchar *src_base [[buffer(0)]],
    device const uchar *src_high [[buffer(1)]],
    device const ushort *src_scale [[buffer(2)]],
    device const ushort *src_bias [[buffer(3)]],
    device uchar *dst_codes [[buffer(4)]],
    device uchar *dst_high2 [[buffer(5)]],
    device uint *dst_sb [[buffer(6)]],
    constant AffineNaxRepackParams &p [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  repack_affine_g64_impl<16u>(src_base, src_high, src_scale, src_bias,
                              dst_codes, dst_high2, dst_sb, p, gid);
}

kernel void omlx_m5_repack_affine_g64_bn32(
    device const uchar *src_base [[buffer(0)]],
    device const uchar *src_high [[buffer(1)]],
    device const ushort *src_scale [[buffer(2)]],
    device const ushort *src_bias [[buffer(3)]],
    device uchar *dst_codes [[buffer(4)]],
    device uchar *dst_high2 [[buffer(5)]],
    device uint *dst_sb [[buffer(6)]],
    constant AffineNaxRepackParams &p [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  repack_affine_g64_impl<32u>(src_base, src_high, src_scale, src_bias,
                              dst_codes, dst_high2, dst_sb, p, gid);
}

kernel void omlx_m5_copy_affine_u32_words(
    device const uint *src [[buffer(0)]],
    device uint *dst [[buffer(1)]],
    constant uint &word_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < word_count) {
    dst[gid] = src[gid];
  }
}

kernel void omlx_m5_copy_affine_sb_slice(
    device const uint *src_sb [[buffer(0)]],
    device uint *dst_sb [[buffer(1)]],
    constant AffineNaxSbCopyParams &p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint g = gid.x;
  const uint local_r = gid.y;
  const uint slice_rows = min(p.dst_padded_n - p.dst_row_offset, p.src_padded_n);
  if (g >= p.groups || local_r >= slice_rows) {
    return;
  }
  const uint val = (local_r < p.src_n) ? src_sb[ulong(g) * p.src_padded_n + local_r] : 0u;
  dst_sb[ulong(g) * p.dst_padded_n + p.dst_row_offset + local_r] = val;
}

// Mode: 0 = q4 (uint4b), 1 = q56 (uint4b + uint2b), 2 = q3 (uint2b + 1-bit high1), 3 = q8 (uint8_t)
template <uint BN, uint SK, uint Mode, uint RF = 1u, uint NSG = 1u>
inline void affine_nax_linear_impl(
    device const bfloat *x,
    device const uchar *codes,
    device const uchar *high_plane,
    device const uint *sb,
    device const float *xs,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup float *parts,
    threadgroup uchar *tg_h2_all,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM = 8u;
  constexpr uint GS = 64u;
  constexpr uint kCap = (16u * BN) / 32u;
  const uint rb = tg.y * (BM * RF);
  const uint tg_n0 = tg.x * (BN * NSG);
  if (rb >= p.rows || tg_n0 >= p.n) {
    return;
  }

  const uint sg_n = (NSG > 1u) ? (sg / SK) : 0u;
  const uint sg_k = (NSG > 1u) ? (sg % SK) : sg;
  const uint tile_n = tg.x * NSG + sg_n;
  const uint n0 = tile_n * BN;
  const bool col_active = (sg_n < NSG) && (n0 < p.n);

  const uint groups = p.k / GS;
  const uint g0 = (sg_k * groups) / SK;
  const uint g1 = ((sg_k + 1u) * groups) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;

  float acc[RF][kCap];
  for (uint r = 0u; r < RF; ++r) {
    for (uint i = 0u; i < kCap; ++i) {
      acc[r][i] = 0.0f;
    }
  }

  const ulong tile_base = ulong(tile_n) * groups;
  threadgroup uchar *my_tg_h2 =
      (Mode == 2u && tg_h2_all != nullptr) ? (tg_h2_all + sg * (BN * 16u)) : nullptr;

  if (col_active) {
    for (uint g = g0; g < g1; ++g) {
      const ulong idx = tile_base + g;
      device const uint *sb_g =
          (p.padded_n == 0u) ? (sb + idx * BN) : (sb + ulong(g) * p.padded_n + n0);
      float inline_xsum0 = 0.0f;
      if (p.rows == 1u) {
        device const bfloat2 *xv0 =
            reinterpret_cast<device const bfloat2 *>(x + ulong(g) * 64ul);
        const float2 pair0 = float2(xv0[lane]);
        inline_xsum0 = simd_sum(pair0.x + pair0.y);
      }

      if constexpr (Mode == 2u) {
        device const uchar *sh1 = high_plane + idx * (ulong(BN) * 8ul);
        for (uint j = lane; j < BN * 8u; j += 32u) {
          const uint b = uint(sh1[j]);
          my_tg_h2[j * 2u] = expand_4x1bit_to_4x2bit(b & 0x0fu);
          my_tg_h2[j * 2u + 1u] = expand_4x1bit_to_4x2bit(b >> 4u);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }

      for (uint r = 0u; r < RF; ++r) {
        const uint rb_r = rb + r * BM;
        if (rb_r >= p.rows) {
          break;
        }
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a_parent_r(
            (device bfloat *)(x + ulong(rb_r) * p.k),
            dextents<int32_t, 2>(p.k, p.rows - rb_r));
        auto a = a_parent_r.slice(g * GS, 0);

        if constexpr (Mode == 0u) {
          tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b4(
              (device uchar *)(codes + idx * (ulong(BN) * 32ul)),
              dextents<int32_t, 2>(GS, BN));
          auto prod4 = op.template get_destination_cooperative_tensor<decltype(a), decltype(b4), float>();
          op.run(a, b4, prod4);
          const uint elems = prod4.get_capacity();
          for (uint i = 0u; i < elems; ++i) {
            if (prod4.is_valid_element(i)) {
              const auto coord = prod4.get_multidimensional_index(i);
              const uint col_local = uint(coord[0]);
              const uint row = rb_r + uint(coord[1]);
              const vec<bfloat, 2> pair = as_type<vec<bfloat, 2>>(sb_g[col_local]);
              const float xsum = (p.rows == 1u)
                                     ? ((row == 0u) ? inline_xsum0 : 0.0f)
                                     : (row < p.mp ? xs[ulong(g) * p.mp + row] : 0.0f);
              acc[r][i] = fma(float(pair[0]), prod4[i], fma(float(pair[1]), xsum, acc[r][i]));
            }
          }
        } else if constexpr (Mode == 1u) {
          tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b4(
              (device uchar *)(codes + idx * (ulong(BN) * 32ul)),
              dextents<int32_t, 2>(GS, BN));
          tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> b2(
              (device uchar *)(high_plane + idx * (ulong(BN) * 16ul)),
              dextents<int32_t, 2>(GS, BN));
          auto prod4 = op.template get_destination_cooperative_tensor<decltype(a), decltype(b4), float>();
          auto prod2 = op.template get_destination_cooperative_tensor<decltype(a), decltype(b2), float>();
          op.run(a, b4, prod4);
          op.run(a, b2, prod2);
          const uint elems = prod4.get_capacity();
          for (uint i = 0u; i < elems; ++i) {
            if (prod4.is_valid_element(i)) {
              const auto coord = prod4.get_multidimensional_index(i);
              const uint col_local = uint(coord[0]);
              const uint row = rb_r + uint(coord[1]);
              const vec<bfloat, 2> pair = as_type<vec<bfloat, 2>>(sb_g[col_local]);
              const float dot = fma(16.0f, prod2[i], prod4[i]);
              const float xsum = (p.rows == 1u)
                                     ? ((row == 0u) ? inline_xsum0 : 0.0f)
                                     : (row < p.mp ? xs[ulong(g) * p.mp + row] : 0.0f);
              acc[r][i] = fma(float(pair[0]), dot, fma(float(pair[1]), xsum, acc[r][i]));
            }
          }
        } else if constexpr (Mode == 2u) {
          tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> blo(
              (device uchar *)(codes + idx * (ulong(BN) * 16ul)),
              dextents<int32_t, 2>(GS, BN));
          tensor<threadgroup uint2b_format, dextents<int32_t, 2>, tensor_inline> bhi(
              my_tg_h2, dextents<int32_t, 2>(GS, BN));
          auto prod_lo = op.template get_destination_cooperative_tensor<decltype(a), decltype(blo), float>();
          auto prod_hi = op.template get_destination_cooperative_tensor<decltype(a), decltype(bhi), float>();
          op.run(a, blo, prod_lo);
          op.run(a, bhi, prod_hi);
          const uint elems = prod_lo.get_capacity();
          for (uint i = 0u; i < elems; ++i) {
            if (prod_lo.is_valid_element(i)) {
              const auto coord = prod_lo.get_multidimensional_index(i);
              const uint col_local = uint(coord[0]);
              const uint row = rb_r + uint(coord[1]);
              const vec<bfloat, 2> pair = as_type<vec<bfloat, 2>>(sb_g[col_local]);
              const float dot = fma(4.0f, prod_hi[i], prod_lo[i]);
              const float xsum = (p.rows == 1u)
                                     ? ((row == 0u) ? inline_xsum0 : 0.0f)
                                     : (row < p.mp ? xs[ulong(g) * p.mp + row] : 0.0f);
              acc[r][i] = fma(float(pair[0]), dot, fma(float(pair[1]), xsum, acc[r][i]));
            }
          }
        } else {
          tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> b8(
              (device uchar *)(codes + idx * (ulong(BN) * 64ul)),
              dextents<int32_t, 2>(GS, BN));
          auto prod8 = op.template get_destination_cooperative_tensor<decltype(a), decltype(b8), float>();
          op.run(a, b8, prod8);
          const uint elems = prod8.get_capacity();
          for (uint i = 0u; i < elems; ++i) {
            if (prod8.is_valid_element(i)) {
              const auto coord = prod8.get_multidimensional_index(i);
              const uint col_local = uint(coord[0]);
              const uint row = rb_r + uint(coord[1]);
              const vec<bfloat, 2> pair = as_type<vec<bfloat, 2>>(sb_g[col_local]);
              const float xsum = (p.rows == 1u)
                                     ? ((row == 0u) ? inline_xsum0 : 0.0f)
                                     : (row < p.mp ? xs[ulong(g) * p.mp + row] : 0.0f);
              acc[r][i] = fma(float(pair[0]), prod8[i], fma(float(pair[1]), xsum, acc[r][i]));
            }
          }
        }
      }
      if constexpr (Mode == 2u) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }

  constexpr uint kPartStride = (SK > 1u ? (SK - 1u) : 1u) * kCap * 32u;
  threadgroup float *my_parts = parts + ((NSG > 1u && sg_n < NSG) ? (sg_n * kPartStride) : 0u);

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a_lay(
      (device bfloat *)(x + ulong(rb) * p.k),
      dextents<int32_t, 2>(p.k, p.rows - rb));
  auto a0_lay = a_lay.slice(0, 0);
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0_lay(
      (device uchar *)codes, dextents<int32_t, 2>(GS, BN));
  auto layout = op.template get_destination_cooperative_tensor<decltype(a0_lay), decltype(b0_lay), float>();
  const uint elems = layout.get_capacity();

  for (uint r = 0u; r < RF; ++r) {
    const uint rb_r = rb + r * BM;
    if (rb_r >= p.rows) {
      break;
    }
    if constexpr (SK > 1u) {
      if (r > 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (sg_k > 0u && col_active) {
        for (uint i = 0u; i < kCap; ++i) {
          my_parts[((sg_k - 1u) * kCap + i) * 32u + lane] = acc[r][i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg_k == 0u && col_active) {
        for (uint s = 1u; s < SK; ++s) {
          for (uint i = 0u; i < kCap; ++i) {
            acc[r][i] += my_parts[((s - 1u) * kCap + i) * 32u + lane];
          }
        }
      }
    }
    if (sg_k == 0u && col_active) {
      for (uint i = 0u; i < elems; ++i) {
        if (layout.is_valid_element(i)) {
          const auto coord = layout.get_multidimensional_index(i);
          const uint col = n0 + uint(coord[0]);
          const uint row = rb_r + uint(coord[1]);
          if (row < p.rows && col < p.n) {
            y[ulong(row) * p.n + col] = bfloat(acc[r][i]);
          }
        }
      }
    }
  }
}

template <uint SK, uint Mode, uint RF = 1u>
inline void affine_nax_swiglu_n16_impl(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const uint *g_sb,
    device const uchar *u_codes,
    device const uchar *u_high,
    device const uint *u_sb,
    device const float *xs,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup float *parts,
    threadgroup uchar *tg_h2_all,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM = 8u;
  constexpr uint BN = 16u;
  constexpr uint GS = 64u;
  constexpr uint kCap = (16u * BN) / 32u;
  const uint rb = tg.y * (BM * RF);
  const uint n0 = tg.x * BN;
  if (rb >= p.rows || n0 >= p.n) {
    return;
  }

  const uint groups = p.k / GS;
  const uint g0 = (sg * groups) / SK;
  const uint g1 = ((sg + 1u) * groups) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;

  float acc_g[RF][kCap];
  float acc_u[RF][kCap];
  for (uint r = 0u; r < RF; ++r) {
    for (uint i = 0u; i < kCap; ++i) {
      acc_g[r][i] = 0.0f;
      acc_u[r][i] = 0.0f;
    }
  }

  const ulong tile_base = ulong(tg.x) * groups;
  threadgroup uchar *my_tg_gh2 =
      (Mode == 2u && tg_h2_all != nullptr) ? (tg_h2_all + sg * (2u * BN * 16u)) : nullptr;
  threadgroup uchar *my_tg_uh2 =
      (Mode == 2u && tg_h2_all != nullptr) ? (my_tg_gh2 + BN * 16u) : nullptr;

  for (uint g = g0; g < g1; ++g) {
    const ulong idx = tile_base + g;
    device const uint *gsb_g =
        (p.padded_n == 0u) ? (g_sb + idx * BN) : (g_sb + ulong(g) * p.padded_n + n0);
    device const uint *usb_g =
        (p.padded_n == 0u) ? (u_sb + idx * BN) : (u_sb + ulong(g) * p.padded_n + n0);
    float inline_xsum0 = 0.0f;
    if (p.rows == 1u) {
      device const bfloat2 *xv0 =
          reinterpret_cast<device const bfloat2 *>(x + ulong(g) * 64ul);
      const float2 pair0 = float2(xv0[lane]);
      inline_xsum0 = simd_sum(pair0.x + pair0.y);
    }

    if constexpr (Mode == 2u) {
      device const uchar *gsh1 = g_high + idx * (ulong(BN) * 8ul);
      device const uchar *ush1 = u_high + idx * (ulong(BN) * 8ul);
      for (uint j = lane; j < BN * 8u; j += 32u) {
        const uint gb = uint(gsh1[j]);
        const uint ub = uint(ush1[j]);
        my_tg_gh2[j * 2u] = expand_4x1bit_to_4x2bit(gb & 0x0fu);
        my_tg_gh2[j * 2u + 1u] = expand_4x1bit_to_4x2bit(gb >> 4u);
        my_tg_uh2[j * 2u] = expand_4x1bit_to_4x2bit(ub & 0x0fu);
        my_tg_uh2[j * 2u + 1u] = expand_4x1bit_to_4x2bit(ub >> 4u);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint r = 0u; r < RF; ++r) {
      const uint rb_r = rb + r * BM;
      if (rb_r >= p.rows) {
        break;
      }
      tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a_parent_r(
          (device bfloat *)(x + ulong(rb_r) * p.k),
          dextents<int32_t, 2>(p.k, p.rows - rb_r));
      auto a = a_parent_r.slice(g * GS, 0);

      if constexpr (Mode == 0u) {
        tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bg(
            (device uchar *)(g_codes + idx * (ulong(BN) * 32ul)),
            dextents<int32_t, 2>(GS, BN));
        tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bu(
            (device uchar *)(u_codes + idx * (ulong(BN) * 32ul)),
            dextents<int32_t, 2>(GS, BN));
        auto pg = op.template get_destination_cooperative_tensor<decltype(a), decltype(bg), float>();
        auto pu = op.template get_destination_cooperative_tensor<decltype(a), decltype(bu), float>();
        op.run(a, bg, pg);
        op.run(a, bu, pu);
        const uint elems = pg.get_capacity();
        for (uint i = 0u; i < elems; ++i) {
          if (pg.is_valid_element(i)) {
            const auto coord = pg.get_multidimensional_index(i);
            const uint col_local = uint(coord[0]);
            const uint row = rb_r + uint(coord[1]);
            const vec<bfloat, 2> gp = as_type<vec<bfloat, 2>>(gsb_g[col_local]);
            const vec<bfloat, 2> up = as_type<vec<bfloat, 2>>(usb_g[col_local]);
            const float xsum = (p.rows == 1u)
                                   ? ((row == 0u) ? inline_xsum0 : 0.0f)
                                   : (row < p.mp ? xs[ulong(g) * p.mp + row] : 0.0f);
            acc_g[r][i] = fma(float(gp[0]), pg[i], fma(float(gp[1]), xsum, acc_g[r][i]));
            acc_u[r][i] = fma(float(up[0]), pu[i], fma(float(up[1]), xsum, acc_u[r][i]));
          }
        }
      } else if constexpr (Mode == 1u) {
        tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bg4(
            (device uchar *)(g_codes + idx * (ulong(BN) * 32ul)),
            dextents<int32_t, 2>(GS, BN));
        tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> bg2(
            (device uchar *)(g_high + idx * (ulong(BN) * 16ul)),
            dextents<int32_t, 2>(GS, BN));
        tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bu4(
            (device uchar *)(u_codes + idx * (ulong(BN) * 32ul)),
            dextents<int32_t, 2>(GS, BN));
        tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> bu2(
            (device uchar *)(u_high + idx * (ulong(BN) * 16ul)),
            dextents<int32_t, 2>(GS, BN));
        auto pg4 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bg4), float>();
        auto pg2 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bg2), float>();
        auto pu4 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bu4), float>();
        auto pu2 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bu2), float>();
        op.run(a, bg4, pg4);
        op.run(a, bg2, pg2);
        op.run(a, bu4, pu4);
        op.run(a, bu2, pu2);
        const uint elems = pg4.get_capacity();
        for (uint i = 0u; i < elems; ++i) {
          if (pg4.is_valid_element(i)) {
            const auto coord = pg4.get_multidimensional_index(i);
            const uint col_local = uint(coord[0]);
            const uint row = rb_r + uint(coord[1]);
            const vec<bfloat, 2> gp = as_type<vec<bfloat, 2>>(gsb_g[col_local]);
            const vec<bfloat, 2> up = as_type<vec<bfloat, 2>>(usb_g[col_local]);
            const float xsum = (p.rows == 1u)
                                   ? ((row == 0u) ? inline_xsum0 : 0.0f)
                                   : (row < p.mp ? xs[ulong(g) * p.mp + row] : 0.0f);
            const float dg = fma(16.0f, pg2[i], pg4[i]);
            const float du = fma(16.0f, pu2[i], pu4[i]);
            acc_g[r][i] = fma(float(gp[0]), dg, fma(float(gp[1]), xsum, acc_g[r][i]));
            acc_u[r][i] = fma(float(up[0]), du, fma(float(up[1]), xsum, acc_u[r][i]));
          }
        }
      } else if constexpr (Mode == 2u) {
        tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> bglo(
            (device uchar *)(g_codes + idx * (ulong(BN) * 16ul)),
            dextents<int32_t, 2>(GS, BN));
        tensor<threadgroup uint2b_format, dextents<int32_t, 2>, tensor_inline> bghi(
            my_tg_gh2, dextents<int32_t, 2>(GS, BN));
        tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> bulo(
            (device uchar *)(u_codes + idx * (ulong(BN) * 16ul)),
            dextents<int32_t, 2>(GS, BN));
        tensor<threadgroup uint2b_format, dextents<int32_t, 2>, tensor_inline> buhi(
            my_tg_uh2, dextents<int32_t, 2>(GS, BN));
        auto pglo = op.template get_destination_cooperative_tensor<decltype(a), decltype(bglo), float>();
        auto pghi = op.template get_destination_cooperative_tensor<decltype(a), decltype(bghi), float>();
        auto pulo = op.template get_destination_cooperative_tensor<decltype(a), decltype(bulo), float>();
        auto puhi = op.template get_destination_cooperative_tensor<decltype(a), decltype(buhi), float>();
        op.run(a, bglo, pglo);
        op.run(a, bghi, pghi);
        op.run(a, bulo, pulo);
        op.run(a, buhi, puhi);
        const uint elems = pglo.get_capacity();
        for (uint i = 0u; i < elems; ++i) {
          if (pglo.is_valid_element(i)) {
            const auto coord = pglo.get_multidimensional_index(i);
            const uint col_local = uint(coord[0]);
            const uint row = rb_r + uint(coord[1]);
            const vec<bfloat, 2> gp = as_type<vec<bfloat, 2>>(gsb_g[col_local]);
            const vec<bfloat, 2> up = as_type<vec<bfloat, 2>>(usb_g[col_local]);
            const float xsum = (p.rows == 1u)
                                   ? ((row == 0u) ? inline_xsum0 : 0.0f)
                                   : (row < p.mp ? xs[ulong(g) * p.mp + row] : 0.0f);
            const float dg = fma(4.0f, pghi[i], pglo[i]);
            const float du = fma(4.0f, puhi[i], pulo[i]);
            acc_g[r][i] = fma(float(gp[0]), dg, fma(float(gp[1]), xsum, acc_g[r][i]));
            acc_u[r][i] = fma(float(up[0]), du, fma(float(up[1]), xsum, acc_u[r][i]));
          }
        }
      } else {
        tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> bg8(
            (device uchar *)(g_codes + idx * (ulong(BN) * 64ul)),
            dextents<int32_t, 2>(GS, BN));
        tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> bu8(
            (device uchar *)(u_codes + idx * (ulong(BN) * 64ul)),
            dextents<int32_t, 2>(GS, BN));
        auto pg8 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bg8), float>();
        auto pu8 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bu8), float>();
        op.run(a, bg8, pg8);
        op.run(a, bu8, pu8);
        const uint elems = pg8.get_capacity();
        for (uint i = 0u; i < elems; ++i) {
          if (pg8.is_valid_element(i)) {
            const auto coord = pg8.get_multidimensional_index(i);
            const uint col_local = uint(coord[0]);
            const uint row = rb_r + uint(coord[1]);
            const vec<bfloat, 2> gp = as_type<vec<bfloat, 2>>(gsb_g[col_local]);
            const vec<bfloat, 2> up = as_type<vec<bfloat, 2>>(usb_g[col_local]);
            const float xsum = (p.rows == 1u)
                                   ? ((row == 0u) ? inline_xsum0 : 0.0f)
                                   : (row < p.mp ? xs[ulong(g) * p.mp + row] : 0.0f);
            acc_g[r][i] = fma(float(gp[0]), pg8[i], fma(float(gp[1]), xsum, acc_g[r][i]));
            acc_u[r][i] = fma(float(up[0]), pu8[i], fma(float(up[1]), xsum, acc_u[r][i]));
          }
        }
      }
    }
    if constexpr (Mode == 2u) {
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a_lay(
      (device bfloat *)(x + ulong(rb) * p.k),
      dextents<int32_t, 2>(p.k, p.rows - rb));
  auto a0_lay = a_lay.slice(0, 0);
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0_lay(
      (device uchar *)g_codes, dextents<int32_t, 2>(GS, BN));
  auto layout = op.template get_destination_cooperative_tensor<decltype(a0_lay), decltype(b0_lay), float>();
  const uint elems = layout.get_capacity();

  for (uint r = 0u; r < RF; ++r) {
    const uint rb_r = rb + r * BM;
    if (rb_r >= p.rows) {
      break;
    }
    if constexpr (SK > 1u) {
      if (r > 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (sg > 0u) {
        for (uint i = 0u; i < kCap; ++i) {
          parts[((sg - 1u) * (2u * kCap) + i) * 32u + lane] = acc_g[r][i];
          parts[((sg - 1u) * (2u * kCap) + kCap + i) * 32u + lane] = acc_u[r][i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0u) {
        for (uint s = 1u; s < SK; ++s) {
          for (uint i = 0u; i < kCap; ++i) {
            acc_g[r][i] += parts[((s - 1u) * (2u * kCap) + i) * 32u + lane];
            acc_u[r][i] += parts[((s - 1u) * (2u * kCap) + kCap + i) * 32u + lane];
          }
        }
      }
    }
    if (sg == 0u) {
      for (uint i = 0u; i < elems; ++i) {
        if (layout.is_valid_element(i)) {
          const auto coord = layout.get_multidimensional_index(i);
          const uint col = n0 + uint(coord[0]);
          const uint row = rb_r + uint(coord[1]);
          if (row < p.rows && col < p.n) {
            const float gv = float(bfloat(acc_g[r][i]));
            const float uv = float(bfloat(acc_u[r][i]));
            y[ulong(row) * p.n + col] = bfloat(nax_silu_f32(gv) * uv);
          }
        }
      }
    }
  }
}

// Kernel entry-point wrappers (5-arg signature for q4/q8, 6-arg signature for q3/q56)
#define OMLX_M5_AFFINE_NO_HIGH_KERNEL(NAME, BN, SK, MODE, RF, NSG)             \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *codes [[buffer(1)]],                                 \
      device const uint *sb [[buffer(2)]],                                     \
      device const float *xs [[buffer(3)]],                                    \
      device bfloat *y [[buffer(4)]],                                          \
      constant AffineNaxParams &p [[buffer(5)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[NSG * (SK > 1u ? SK - 1u : 1u) * ((16u * BN) / 32u) * 32u]; \
    affine_nax_linear_impl<BN, SK, MODE, RF, NSG>(x, codes, nullptr, sb, xs,   \
                                                  y, p, parts, nullptr, lane,  \
                                                  sg, tg);                     \
  }

#define OMLX_M5_AFFINE_Q56_KERNEL(NAME, BN, SK, RF, NSG)                       \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *codes [[buffer(1)]],                                 \
      device const uchar *high2 [[buffer(2)]],                                 \
      device const uint *sb [[buffer(3)]],                                     \
      device const float *xs [[buffer(4)]],                                    \
      device bfloat *y [[buffer(5)]],                                          \
      constant AffineNaxParams &p [[buffer(6)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[NSG * (SK > 1u ? SK - 1u : 1u) * ((16u * BN) / 32u) * 32u]; \
    affine_nax_linear_impl<BN, SK, 1u, RF, NSG>(x, codes, high2, sb, xs, y, p, \
                                                parts, nullptr, lane, sg, tg); \
  }

#define OMLX_M5_AFFINE_Q3_KERNEL(NAME, BN, SK, RF, NSG)                        \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *codes [[buffer(1)]],                                 \
      device const uchar *high1 [[buffer(2)]],                                 \
      device const uint *sb [[buffer(3)]],                                     \
      device const float *xs [[buffer(4)]],                                    \
      device bfloat *y [[buffer(5)]],                                          \
      constant AffineNaxParams &p [[buffer(6)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[NSG * (SK > 1u ? SK - 1u : 1u) * ((16u * BN) / 32u) * 32u]; \
    threadgroup uchar tg_h2[NSG * SK * BN * 16u];                              \
    affine_nax_linear_impl<BN, SK, 2u, RF, NSG>(x, codes, high1, sb, xs, y, p, \
                                                parts, tg_h2, lane, sg, tg);   \
  }

#define OMLX_M5_AFFINE_SWIGLU_NO_HIGH_KERNEL(NAME, SK, MODE, RF)               \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *g_codes [[buffer(1)]],                               \
      device const uint *g_sb [[buffer(2)]],                                   \
      device const uchar *u_codes [[buffer(3)]],                               \
      device const uint *u_sb [[buffer(4)]],                                   \
      device const float *xs [[buffer(5)]],                                    \
      device bfloat *y [[buffer(6)]],                                          \
      constant AffineNaxParams &p [[buffer(7)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 16u * 32u];             \
    affine_nax_swiglu_n16_impl<SK, MODE, RF>(x, g_codes, nullptr, g_sb,        \
                                             u_codes, nullptr, u_sb, xs, y, p, \
                                             parts, nullptr, lane, sg, tg);    \
  }

#define OMLX_M5_AFFINE_SWIGLU_Q56_KERNEL(NAME, SK, RF)                         \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *g_codes [[buffer(1)]],                               \
      device const uchar *g_high [[buffer(2)]],                                \
      device const uint *g_sb [[buffer(3)]],                                   \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_high [[buffer(5)]],                                \
      device const uint *u_sb [[buffer(6)]],                                   \
      device const float *xs [[buffer(7)]],                                    \
      device bfloat *y [[buffer(8)]],                                          \
      constant AffineNaxParams &p [[buffer(9)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 16u * 32u];             \
    affine_nax_swiglu_n16_impl<SK, 1u, RF>(x, g_codes, g_high, g_sb, u_codes,  \
                                           u_high, u_sb, xs, y, p, parts,      \
                                           nullptr, lane, sg, tg);             \
  }

#define OMLX_M5_AFFINE_SWIGLU_Q3_KERNEL(NAME, SK, RF)                          \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *g_codes [[buffer(1)]],                               \
      device const uchar *g_high [[buffer(2)]],                                \
      device const uint *g_sb [[buffer(3)]],                                   \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_high [[buffer(5)]],                                \
      device const uint *u_sb [[buffer(6)]],                                   \
      device const float *xs [[buffer(7)]],                                    \
      device bfloat *y [[buffer(8)]],                                          \
      constant AffineNaxParams &p [[buffer(9)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 16u * 32u];             \
    threadgroup uchar tg_h2[SK * 2u * 16u * 16u];                              \
    affine_nax_swiglu_n16_impl<SK, 2u, RF>(x, g_codes, g_high, g_sb, u_codes,  \
                                           u_high, u_sb, xs, y, p, parts,      \
                                           tg_h2, lane, sg, tg);               \
  }

#define OMLX_M5_AFFINE_NAX_N16_BASE(SK)                                        \
  OMLX_M5_AFFINE_NO_HIGH_KERNEL(omlx_m5_affine_nax_q4_n16_sk##SK, 16u, SK, 0u, 1u, 1u) \
  OMLX_M5_AFFINE_NO_HIGH_KERNEL(omlx_m5_affine_nax_q4_n16_sk##SK##_rf1_nsg2, 16u, SK, 0u, 1u, 2u) \
  OMLX_M5_AFFINE_Q56_KERNEL(omlx_m5_affine_nax_q56_n16_sk##SK, 16u, SK, 1u, 1u) \
  OMLX_M5_AFFINE_Q56_KERNEL(omlx_m5_affine_nax_q56_n16_sk##SK##_rf1_nsg2, 16u, SK, 1u, 2u) \
  OMLX_M5_AFFINE_Q3_KERNEL(omlx_m5_affine_nax_q3_n16_sk##SK, 16u, SK, 1u, 1u)  \
  OMLX_M5_AFFINE_Q3_KERNEL(omlx_m5_affine_nax_q3_n16_sk##SK##_rf1_nsg2, 16u, SK, 1u, 2u) \
  OMLX_M5_AFFINE_NO_HIGH_KERNEL(omlx_m5_affine_nax_q8_n16_sk##SK, 16u, SK, 3u, 1u, 1u) \
  OMLX_M5_AFFINE_NO_HIGH_KERNEL(omlx_m5_affine_nax_q8_n16_sk##SK##_rf1_nsg2, 16u, SK, 3u, 1u, 2u) \
  OMLX_M5_AFFINE_SWIGLU_NO_HIGH_KERNEL(omlx_m5_affine_nax_q4_swiglu_n16_sk##SK, SK, 0u, 1u) \
  OMLX_M5_AFFINE_SWIGLU_Q56_KERNEL(omlx_m5_affine_nax_q56_swiglu_n16_sk##SK, SK, 1u) \
  OMLX_M5_AFFINE_SWIGLU_Q3_KERNEL(omlx_m5_affine_nax_q3_swiglu_n16_sk##SK, SK, 1u) \
  OMLX_M5_AFFINE_SWIGLU_NO_HIGH_KERNEL(omlx_m5_affine_nax_q8_swiglu_n16_sk##SK, SK, 3u, 1u)

#define OMLX_M5_AFFINE_NAX_N16_RF(SK, RF)                                      \
  OMLX_M5_AFFINE_NO_HIGH_KERNEL(omlx_m5_affine_nax_q4_n16_sk##SK##_rf##RF##_nsg1, 16u, SK, 0u, RF, 1u) \
  OMLX_M5_AFFINE_NO_HIGH_KERNEL(omlx_m5_affine_nax_q4_n16_sk##SK##_rf##RF##_nsg2, 16u, SK, 0u, RF, 2u) \
  OMLX_M5_AFFINE_Q56_KERNEL(omlx_m5_affine_nax_q56_n16_sk##SK##_rf##RF##_nsg1, 16u, SK, RF, 1u) \
  OMLX_M5_AFFINE_Q56_KERNEL(omlx_m5_affine_nax_q56_n16_sk##SK##_rf##RF##_nsg2, 16u, SK, RF, 2u) \
  OMLX_M5_AFFINE_Q3_KERNEL(omlx_m5_affine_nax_q3_n16_sk##SK##_rf##RF##_nsg1, 16u, SK, RF, 1u) \
  OMLX_M5_AFFINE_Q3_KERNEL(omlx_m5_affine_nax_q3_n16_sk##SK##_rf##RF##_nsg2, 16u, SK, RF, 2u) \
  OMLX_M5_AFFINE_NO_HIGH_KERNEL(omlx_m5_affine_nax_q8_n16_sk##SK##_rf##RF##_nsg1, 16u, SK, 3u, RF, 1u) \
  OMLX_M5_AFFINE_NO_HIGH_KERNEL(omlx_m5_affine_nax_q8_n16_sk##SK##_rf##RF##_nsg2, 16u, SK, 3u, RF, 2u) \
  OMLX_M5_AFFINE_SWIGLU_NO_HIGH_KERNEL(omlx_m5_affine_nax_q4_swiglu_n16_sk##SK##_rf##RF, SK, 0u, RF) \
  OMLX_M5_AFFINE_SWIGLU_Q56_KERNEL(omlx_m5_affine_nax_q56_swiglu_n16_sk##SK##_rf##RF, SK, RF) \
  OMLX_M5_AFFINE_SWIGLU_Q3_KERNEL(omlx_m5_affine_nax_q3_swiglu_n16_sk##SK##_rf##RF, SK, RF) \
  OMLX_M5_AFFINE_SWIGLU_NO_HIGH_KERNEL(omlx_m5_affine_nax_q8_swiglu_n16_sk##SK##_rf##RF, SK, 3u, RF)

#define OMLX_M5_AFFINE_NAX_N16_ALL(SK)                                         \
  OMLX_M5_AFFINE_NAX_N16_BASE(SK)                                              \
  OMLX_M5_AFFINE_NAX_N16_RF(SK, 2)                                             \
  OMLX_M5_AFFINE_NAX_N16_RF(SK, 3)                                             \
  OMLX_M5_AFFINE_NAX_N16_RF(SK, 4)                                             \
  OMLX_M5_AFFINE_NAX_N16_RF(SK, 6)                                             \
  OMLX_M5_AFFINE_NAX_N16_RF(SK, 8)

#define OMLX_M5_AFFINE_NAX_N32_BASE(SK)                                        \
  OMLX_M5_AFFINE_NO_HIGH_KERNEL(omlx_m5_affine_nax_q4_n32_sk##SK, 32u, SK, 0u, 1u, 1u) \
  OMLX_M5_AFFINE_Q56_KERNEL(omlx_m5_affine_nax_q56_n32_sk##SK, 32u, SK, 1u, 1u) \
  OMLX_M5_AFFINE_Q3_KERNEL(omlx_m5_affine_nax_q3_n32_sk##SK, 32u, SK, 1u, 1u)  \
  OMLX_M5_AFFINE_NO_HIGH_KERNEL(omlx_m5_affine_nax_q8_n32_sk##SK, 32u, SK, 3u, 1u, 1u)

#define OMLX_M5_AFFINE_NAX_N32_RF(SK, RF)                                      \
  OMLX_M5_AFFINE_NO_HIGH_KERNEL(omlx_m5_affine_nax_q4_n32_sk##SK##_rf##RF, 32u, SK, 0u, RF, 1u) \
  OMLX_M5_AFFINE_Q56_KERNEL(omlx_m5_affine_nax_q56_n32_sk##SK##_rf##RF, 32u, SK, RF, 1u) \
  OMLX_M5_AFFINE_Q3_KERNEL(omlx_m5_affine_nax_q3_n32_sk##SK##_rf##RF, 32u, SK, RF, 1u) \
  OMLX_M5_AFFINE_NO_HIGH_KERNEL(omlx_m5_affine_nax_q8_n32_sk##SK##_rf##RF, 32u, SK, 3u, RF, 1u)

#define OMLX_M5_AFFINE_NAX_N32_ALL(SK)                                         \
  OMLX_M5_AFFINE_NAX_N32_BASE(SK)                                              \
  OMLX_M5_AFFINE_NAX_N32_RF(SK, 2)                                             \
  OMLX_M5_AFFINE_NAX_N32_RF(SK, 3)                                             \
  OMLX_M5_AFFINE_NAX_N32_RF(SK, 4)                                             \
  OMLX_M5_AFFINE_NAX_N32_RF(SK, 6)                                             \
  OMLX_M5_AFFINE_NAX_N32_RF(SK, 8)

OMLX_M5_AFFINE_NAX_N16_ALL(1)
OMLX_M5_AFFINE_NAX_N16_ALL(2)
OMLX_M5_AFFINE_NAX_N16_ALL(4)
OMLX_M5_AFFINE_NAX_N16_ALL(8)
OMLX_M5_AFFINE_NAX_N32_ALL(1)
OMLX_M5_AFFINE_NAX_N32_ALL(2)
OMLX_M5_AFFINE_NAX_N32_ALL(4)
OMLX_M5_AFFINE_NAX_N32_ALL(8)

// ============================================================================
// v0.8.0-m5rt.14 0-ULP Hardware NAX BM=16 & BM=32, BN=16/32, SK=1..8
// Proven on M5 Ultra cold-DRAM micro-probe: 1.62x at M=16 (BM=16 SK=8),
// 1.65x at M=32 (BM=32 SK=8), strict 0-ULP bit-identical to BM=8 SK=8!
// ============================================================================

template <uint BM, uint BN, uint SK, uint Mode>
inline void affine_nax_linear_bm_impl(
    device const bfloat *x,
    device const uchar *codes,
    device const uchar *high_plane,
    device const uint *sb,
    device const float *xs,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup float *parts,
    threadgroup uchar *tg_h2_all,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint GS = 64u;
  constexpr uint kCap = (BM * BN) / 32u;
  const uint rb = tg.y * BM;
  const uint tile_n = tg.x;
  const uint n0 = tile_n * BN;
  if (rb >= p.rows || n0 >= p.n) {
    return;
  }

  const uint groups = p.k / GS;
  const uint g0 = (sg * groups) / SK;
  const uint g1 = ((sg + 1u) * groups) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;

  float acc[kCap];
#pragma clang loop unroll(full)
  for (uint i = 0u; i < kCap; ++i) {
    acc[i] = 0.0f;
  }

  const ulong tile_base = ulong(tile_n) * groups;
  threadgroup uchar *my_tg_h2 =
      (Mode == 2u && tg_h2_all != nullptr) ? (tg_h2_all + sg * (BN * 16u)) : nullptr;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a_parent(
      (device bfloat *)(x + ulong(rb) * p.k),
      dextents<int32_t, 2>(p.k, p.rows - rb));

  for (uint g = g0; g < g1; ++g) {
    const ulong idx = tile_base + g;
    device const uint *sb_g =
        (p.padded_n == 0u) ? (sb + idx * BN) : (sb + ulong(g) * p.padded_n + n0);
    if constexpr (Mode == 2u) {
      device const uchar *sh1 = high_plane + idx * (ulong(BN) * 8ul);
      for (uint j = lane; j < BN * 8u; j += 32u) {
        const uint b = uint(sh1[j]);
        my_tg_h2[j * 2u] = expand_4x1bit_to_4x2bit(b & 0x0fu);
        my_tg_h2[j * 2u + 1u] = expand_4x1bit_to_4x2bit(b >> 4u);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto a = a_parent.slice(g * GS, 0);
    if constexpr (Mode == 0u) {
      tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b4(
          (device uchar *)(codes + idx * (ulong(BN) * 32ul)),
          dextents<int32_t, 2>(GS, BN));
      auto prod4 = op.template get_destination_cooperative_tensor<decltype(a), decltype(b4), float>();
      op.run(a, b4, prod4);
      const uint elems = prod4.get_capacity();
#pragma clang loop unroll(full)
      for (uint i = 0u; i < elems; ++i) {
        if (prod4.is_valid_element(i)) {
          const auto coord = prod4.get_multidimensional_index(i);
          const uint col_local = uint(coord[0]);
          const uint row = rb + uint(coord[1]);
          const vec<bfloat, 2> pair = as_type<vec<bfloat, 2>>(sb_g[col_local]);
          const float xsum = (row < p.mp) ? xs[ulong(g) * p.mp + row] : 0.0f;
          acc[i] = fma(float(pair[0]), prod4[i], fma(float(pair[1]), xsum, acc[i]));
        }
      }
    } else if constexpr (Mode == 1u) {
      tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b4(
          (device uchar *)(codes + idx * (ulong(BN) * 32ul)),
          dextents<int32_t, 2>(GS, BN));
      tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> b2(
          (device uchar *)(high_plane + idx * (ulong(BN) * 16ul)),
          dextents<int32_t, 2>(GS, BN));
      auto prod4 = op.template get_destination_cooperative_tensor<decltype(a), decltype(b4), float>();
      auto prod2 = op.template get_destination_cooperative_tensor<decltype(a), decltype(b2), float>();
      op.run(a, b4, prod4);
      op.run(a, b2, prod2);
      const uint elems = prod4.get_capacity();
#pragma clang loop unroll(full)
      for (uint i = 0u; i < elems; ++i) {
        if (prod4.is_valid_element(i)) {
          const auto coord = prod4.get_multidimensional_index(i);
          const uint col_local = uint(coord[0]);
          const uint row = rb + uint(coord[1]);
          const vec<bfloat, 2> pair = as_type<vec<bfloat, 2>>(sb_g[col_local]);
          const float dot = fma(16.0f, prod2[i], prod4[i]);
          const float xsum = (row < p.mp) ? xs[ulong(g) * p.mp + row] : 0.0f;
          acc[i] = fma(float(pair[0]), dot, fma(float(pair[1]), xsum, acc[i]));
        }
      }
    } else if constexpr (Mode == 2u) {
      tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> blo(
          (device uchar *)(codes + idx * (ulong(BN) * 16ul)),
          dextents<int32_t, 2>(GS, BN));
      tensor<threadgroup uint2b_format, dextents<int32_t, 2>, tensor_inline> bhi(
          my_tg_h2, dextents<int32_t, 2>(GS, BN));
      auto prod_lo = op.template get_destination_cooperative_tensor<decltype(a), decltype(blo), float>();
      auto prod_hi = op.template get_destination_cooperative_tensor<decltype(a), decltype(bhi), float>();
      op.run(a, blo, prod_lo);
      op.run(a, bhi, prod_hi);
      const uint elems = prod_lo.get_capacity();
#pragma clang loop unroll(full)
      for (uint i = 0u; i < elems; ++i) {
        if (prod_lo.is_valid_element(i)) {
          const auto coord = prod_lo.get_multidimensional_index(i);
          const uint col_local = uint(coord[0]);
          const uint row = rb + uint(coord[1]);
          const vec<bfloat, 2> pair = as_type<vec<bfloat, 2>>(sb_g[col_local]);
          const float dot = fma(4.0f, prod_hi[i], prod_lo[i]);
          const float xsum = (row < p.mp) ? xs[ulong(g) * p.mp + row] : 0.0f;
          acc[i] = fma(float(pair[0]), dot, fma(float(pair[1]), xsum, acc[i]));
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> b8(
          (device uchar *)(codes + idx * (ulong(BN) * 64ul)),
          dextents<int32_t, 2>(GS, BN));
      auto prod8 = op.template get_destination_cooperative_tensor<decltype(a), decltype(b8), float>();
      op.run(a, b8, prod8);
      const uint elems = prod8.get_capacity();
#pragma clang loop unroll(full)
      for (uint i = 0u; i < elems; ++i) {
        if (prod8.is_valid_element(i)) {
          const auto coord = prod8.get_multidimensional_index(i);
          const uint col_local = uint(coord[0]);
          const uint row = rb + uint(coord[1]);
          const vec<bfloat, 2> pair = as_type<vec<bfloat, 2>>(sb_g[col_local]);
          const float xsum = (row < p.mp) ? xs[ulong(g) * p.mp + row] : 0.0f;
          acc[i] = fma(float(pair[0]), prod8[i], fma(float(pair[1]), xsum, acc[i]));
        }
      }
    }
  }

  auto a0_lay = a_parent.slice(0, 0);
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0_lay(
      (device uchar *)codes, dextents<int32_t, 2>(GS, BN));
  auto layout = op.template get_destination_cooperative_tensor<decltype(a0_lay), decltype(b0_lay), float>();
  const uint elems = layout.get_capacity();

  if constexpr (SK > 1u) {
    if constexpr (Mode == 2u) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg > 0u) {
#pragma clang loop unroll(full)
      for (uint i = 0u; i < kCap; ++i) {
        parts[((sg - 1u) * kCap + i) * 32u + lane] = acc[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      for (uint s = 1u; s < SK; ++s) {
#pragma clang loop unroll(full)
        for (uint i = 0u; i < kCap; ++i) {
          acc[i] += parts[((s - 1u) * kCap + i) * 32u + lane];
        }
      }
    }
  }
  if (sg == 0u) {
#pragma clang loop unroll(full)
    for (uint i = 0u; i < elems; ++i) {
      if (layout.is_valid_element(i)) {
        const auto coord = layout.get_multidimensional_index(i);
        const uint col = n0 + uint(coord[0]);
        const uint row = rb + uint(coord[1]);
        if (row < p.rows && col < p.n) {
          y[ulong(row) * p.n + col] = bfloat(acc[i]);
        }
      }
    }
  }
}

template <uint BM, uint SK, uint Mode>
inline void affine_nax_swiglu_bm_n16_impl(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const uint *g_sb,
    device const uchar *u_codes,
    device const uchar *u_high,
    device const uint *u_sb,
    device const float *xs,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup float *parts,
    threadgroup uchar *tg_h2_all,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BN = 16u;
  constexpr uint GS = 64u;
  constexpr uint kCap = (BM * BN) / 32u;
  const uint rb = tg.y * BM;
  const uint n0 = tg.x * BN;
  if (rb >= p.rows || n0 >= p.n) {
    return;
  }

  const uint groups = p.k / GS;
  const uint g0 = (sg * groups) / SK;
  const uint g1 = ((sg + 1u) * groups) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;

  float acc_g[kCap];
  float acc_u[kCap];
#pragma clang loop unroll(full)
  for (uint i = 0u; i < kCap; ++i) {
    acc_g[i] = 0.0f;
    acc_u[i] = 0.0f;
  }

  const ulong tile_base = ulong(tg.x) * groups;
  threadgroup uchar *my_tg_gh2 =
      (Mode == 2u && tg_h2_all != nullptr) ? (tg_h2_all + sg * (2u * BN * 16u)) : nullptr;
  threadgroup uchar *my_tg_uh2 =
      (Mode == 2u && tg_h2_all != nullptr) ? (my_tg_gh2 + BN * 16u) : nullptr;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a_parent(
      (device bfloat *)(x + ulong(rb) * p.k),
      dextents<int32_t, 2>(p.k, p.rows - rb));

  for (uint g = g0; g < g1; ++g) {
    const ulong idx = tile_base + g;
    device const uint *gsb_g =
        (p.padded_n == 0u) ? (g_sb + idx * BN) : (g_sb + ulong(g) * p.padded_n + n0);
    device const uint *usb_g =
        (p.padded_n == 0u) ? (u_sb + idx * BN) : (u_sb + ulong(g) * p.padded_n + n0);
    if constexpr (Mode == 2u) {
      device const uchar *gsh1 = g_high + idx * (ulong(BN) * 8ul);
      device const uchar *ush1 = u_high + idx * (ulong(BN) * 8ul);
      for (uint j = lane; j < BN * 8u; j += 32u) {
        const uint gb = uint(gsh1[j]);
        const uint ub = uint(ush1[j]);
        my_tg_gh2[j * 2u] = expand_4x1bit_to_4x2bit(gb & 0x0fu);
        my_tg_gh2[j * 2u + 1u] = expand_4x1bit_to_4x2bit(gb >> 4u);
        my_tg_uh2[j * 2u] = expand_4x1bit_to_4x2bit(ub & 0x0fu);
        my_tg_uh2[j * 2u + 1u] = expand_4x1bit_to_4x2bit(ub >> 4u);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto a = a_parent.slice(g * GS, 0);
    if constexpr (Mode == 0u) {
      tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bg(
          (device uchar *)(g_codes + idx * (ulong(BN) * 32ul)),
          dextents<int32_t, 2>(GS, BN));
      tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bu(
          (device uchar *)(u_codes + idx * (ulong(BN) * 32ul)),
          dextents<int32_t, 2>(GS, BN));
      auto pg = op.template get_destination_cooperative_tensor<decltype(a), decltype(bg), float>();
      auto pu = op.template get_destination_cooperative_tensor<decltype(a), decltype(bu), float>();
      op.run(a, bg, pg);
      op.run(a, bu, pu);
      const uint elems = pg.get_capacity();
#pragma clang loop unroll(full)
      for (uint i = 0u; i < elems; ++i) {
        if (pg.is_valid_element(i)) {
          const auto coord = pg.get_multidimensional_index(i);
          const uint col_local = uint(coord[0]);
          const uint row = rb + uint(coord[1]);
          const vec<bfloat, 2> gp = as_type<vec<bfloat, 2>>(gsb_g[col_local]);
          const vec<bfloat, 2> up = as_type<vec<bfloat, 2>>(usb_g[col_local]);
          const float xsum = (row < p.mp) ? xs[ulong(g) * p.mp + row] : 0.0f;
          acc_g[i] = fma(float(gp[0]), pg[i], fma(float(gp[1]), xsum, acc_g[i]));
          acc_u[i] = fma(float(up[0]), pu[i], fma(float(up[1]), xsum, acc_u[i]));
        }
      }
    } else if constexpr (Mode == 1u) {
      tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bg4(
          (device uchar *)(g_codes + idx * (ulong(BN) * 32ul)),
          dextents<int32_t, 2>(GS, BN));
      tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> bg2(
          (device uchar *)(g_high + idx * (ulong(BN) * 16ul)),
          dextents<int32_t, 2>(GS, BN));
      tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bu4(
          (device uchar *)(u_codes + idx * (ulong(BN) * 32ul)),
          dextents<int32_t, 2>(GS, BN));
      tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> bu2(
          (device uchar *)(u_high + idx * (ulong(BN) * 16ul)),
          dextents<int32_t, 2>(GS, BN));
      auto pg4 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bg4), float>();
      auto pg2 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bg2), float>();
      auto pu4 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bu4), float>();
      auto pu2 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bu2), float>();
      op.run(a, bg4, pg4);
      op.run(a, bg2, pg2);
      op.run(a, bu4, pu4);
      op.run(a, bu2, pu2);
      const uint elems = pg4.get_capacity();
#pragma clang loop unroll(full)
      for (uint i = 0u; i < elems; ++i) {
        if (pg4.is_valid_element(i)) {
          const auto coord = pg4.get_multidimensional_index(i);
          const uint col_local = uint(coord[0]);
          const uint row = rb + uint(coord[1]);
          const vec<bfloat, 2> gp = as_type<vec<bfloat, 2>>(gsb_g[col_local]);
          const vec<bfloat, 2> up = as_type<vec<bfloat, 2>>(usb_g[col_local]);
          const float xsum = (row < p.mp) ? xs[ulong(g) * p.mp + row] : 0.0f;
          const float dg = fma(16.0f, pg2[i], pg4[i]);
          const float du = fma(16.0f, pu2[i], pu4[i]);
          acc_g[i] = fma(float(gp[0]), dg, fma(float(gp[1]), xsum, acc_g[i]));
          acc_u[i] = fma(float(up[0]), du, fma(float(up[1]), xsum, acc_u[i]));
        }
      }
    } else if constexpr (Mode == 2u) {
      tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> bglo(
          (device uchar *)(g_codes + idx * (ulong(BN) * 16ul)),
          dextents<int32_t, 2>(GS, BN));
      tensor<threadgroup uint2b_format, dextents<int32_t, 2>, tensor_inline> bghi(
          my_tg_gh2, dextents<int32_t, 2>(GS, BN));
      tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> bulo(
          (device uchar *)(u_codes + idx * (ulong(BN) * 16ul)),
          dextents<int32_t, 2>(GS, BN));
      tensor<threadgroup uint2b_format, dextents<int32_t, 2>, tensor_inline> buhi(
          my_tg_uh2, dextents<int32_t, 2>(GS, BN));
      auto pglo = op.template get_destination_cooperative_tensor<decltype(a), decltype(bglo), float>();
      auto pghi = op.template get_destination_cooperative_tensor<decltype(a), decltype(bghi), float>();
      auto pulo = op.template get_destination_cooperative_tensor<decltype(a), decltype(bulo), float>();
      auto puhi = op.template get_destination_cooperative_tensor<decltype(a), decltype(buhi), float>();
      op.run(a, bglo, pglo);
      op.run(a, bghi, pghi);
      op.run(a, bulo, pulo);
      op.run(a, buhi, puhi);
      const uint elems = pglo.get_capacity();
#pragma clang loop unroll(full)
      for (uint i = 0u; i < elems; ++i) {
        if (pglo.is_valid_element(i)) {
          const auto coord = pglo.get_multidimensional_index(i);
          const uint col_local = uint(coord[0]);
          const uint row = rb + uint(coord[1]);
          const vec<bfloat, 2> gp = as_type<vec<bfloat, 2>>(gsb_g[col_local]);
          const vec<bfloat, 2> up = as_type<vec<bfloat, 2>>(usb_g[col_local]);
          const float xsum = (row < p.mp) ? xs[ulong(g) * p.mp + row] : 0.0f;
          const float dg = fma(4.0f, pghi[i], pglo[i]);
          const float du = fma(4.0f, puhi[i], pulo[i]);
          acc_g[i] = fma(float(gp[0]), dg, fma(float(gp[1]), xsum, acc_g[i]));
          acc_u[i] = fma(float(up[0]), du, fma(float(up[1]), xsum, acc_u[i]));
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> bg8(
          (device uchar *)(g_codes + idx * (ulong(BN) * 64ul)),
          dextents<int32_t, 2>(GS, BN));
      tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> bu8(
          (device uchar *)(u_codes + idx * (ulong(BN) * 64ul)),
          dextents<int32_t, 2>(GS, BN));
      auto pg8 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bg8), float>();
      auto pu8 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bu8), float>();
      op.run(a, bg8, pg8);
      op.run(a, bu8, pu8);
      const uint elems = pg8.get_capacity();
#pragma clang loop unroll(full)
      for (uint i = 0u; i < elems; ++i) {
        if (pg8.is_valid_element(i)) {
          const auto coord = pg8.get_multidimensional_index(i);
          const uint col_local = uint(coord[0]);
          const uint row = rb + uint(coord[1]);
          const vec<bfloat, 2> gp = as_type<vec<bfloat, 2>>(gsb_g[col_local]);
          const vec<bfloat, 2> up = as_type<vec<bfloat, 2>>(usb_g[col_local]);
          const float xsum = (row < p.mp) ? xs[ulong(g) * p.mp + row] : 0.0f;
          acc_g[i] = fma(float(gp[0]), pg8[i], fma(float(gp[1]), xsum, acc_g[i]));
          acc_u[i] = fma(float(up[0]), pu8[i], fma(float(up[1]), xsum, acc_u[i]));
        }
      }
    }
  }

  auto a0_lay = a_parent.slice(0, 0);
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0_lay(
      (device uchar *)g_codes, dextents<int32_t, 2>(GS, BN));
  auto layout = op.template get_destination_cooperative_tensor<decltype(a0_lay), decltype(b0_lay), float>();
  const uint elems = layout.get_capacity();

  if constexpr (SK > 1u) {
    if constexpr (Mode == 2u) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg > 0u) {
#pragma clang loop unroll(full)
      for (uint i = 0u; i < kCap; ++i) {
        parts[((sg - 1u) * (2u * kCap) + i) * 32u + lane] = acc_g[i];
        parts[((sg - 1u) * (2u * kCap) + kCap + i) * 32u + lane] = acc_u[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      for (uint s = 1u; s < SK; ++s) {
#pragma clang loop unroll(full)
        for (uint i = 0u; i < kCap; ++i) {
          acc_g[i] += parts[((s - 1u) * (2u * kCap) + i) * 32u + lane];
          acc_u[i] += parts[((s - 1u) * (2u * kCap) + kCap + i) * 32u + lane];
        }
      }
    }
  }
  if (sg == 0u) {
#pragma clang loop unroll(full)
    for (uint i = 0u; i < elems; ++i) {
      if (layout.is_valid_element(i)) {
        const auto coord = layout.get_multidimensional_index(i);
        const uint col = n0 + uint(coord[0]);
        const uint row = rb + uint(coord[1]);
        if (row < p.rows && col < p.n) {
          const float gv = float(bfloat(acc_g[i]));
          const float uv = float(bfloat(acc_u[i]));
          y[ulong(row) * p.n + col] = bfloat(nax_silu_f32(gv) * uv);
        }
      }
    }
  }
}

#define OMLX_M5_AFFINE_BM_NO_HIGH_KERNEL(NAME, BM, BN, SK, MODE)               \
  [[max_total_threads_per_threadgroup((SK) * 32u)]]                            \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *codes [[buffer(1)]],                                 \
      device const uint *sb [[buffer(2)]],                                     \
      device const float *xs [[buffer(3)]],                                    \
      device bfloat *y [[buffer(4)]],                                          \
      constant AffineNaxParams &p [[buffer(5)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * ((BM * BN) / 32u) * 32u]; \
    affine_nax_linear_bm_impl<BM, BN, SK, MODE>(                               \
        x, codes, nullptr, sb, xs, y, p, parts, nullptr, lane, sg, tg);        \
  }

#define OMLX_M5_AFFINE_BM_Q56_KERNEL(NAME, BM, BN, SK)                         \
  [[max_total_threads_per_threadgroup((SK) * 32u)]]                            \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *codes [[buffer(1)]],                                 \
      device const uchar *high2 [[buffer(2)]],                                 \
      device const uint *sb [[buffer(3)]],                                     \
      device const float *xs [[buffer(4)]],                                    \
      device bfloat *y [[buffer(5)]],                                          \
      constant AffineNaxParams &p [[buffer(6)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * ((BM * BN) / 32u) * 32u]; \
    affine_nax_linear_bm_impl<BM, BN, SK, 1u>(                                 \
        x, codes, high2, sb, xs, y, p, parts, nullptr, lane, sg, tg);          \
  }

#define OMLX_M5_AFFINE_BM_Q3_KERNEL(NAME, BM, BN, SK)                          \
  [[max_total_threads_per_threadgroup((SK) * 32u)]]                            \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *codes [[buffer(1)]],                                 \
      device const uchar *high1 [[buffer(2)]],                                 \
      device const uint *sb [[buffer(3)]],                                     \
      device const float *xs [[buffer(4)]],                                    \
      device bfloat *y [[buffer(5)]],                                          \
      constant AffineNaxParams &p [[buffer(6)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    constexpr uint kPartsFloats = (SK > 1u ? SK - 1u : 1u) * ((BM * BN) / 32u) * 32u; \
    constexpr uint kH2Floats = ((SK * BN * 16u) + 3u) / 4u;                    \
    threadgroup float scratch[kPartsFloats > kH2Floats ? kPartsFloats : kH2Floats]; \
    affine_nax_linear_bm_impl<BM, BN, SK, 2u>(                                 \
        x, codes, high1, sb, xs, y, p, scratch,                                \
        reinterpret_cast<threadgroup uchar *>(scratch), lane, sg, tg);         \
  }

#define OMLX_M5_AFFINE_BM_SWIGLU_NO_HIGH_KERNEL(NAME, BM, SK, MODE)            \
  [[max_total_threads_per_threadgroup((SK) * 32u)]]                            \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *g_codes [[buffer(1)]],                               \
      device const uint *g_sb [[buffer(2)]],                                   \
      device const uchar *u_codes [[buffer(3)]],                               \
      device const uint *u_sb [[buffer(4)]],                                   \
      device const float *xs [[buffer(5)]],                                    \
      device bfloat *y [[buffer(6)]],                                          \
      constant AffineNaxParams &p [[buffer(7)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 2u * ((BM * 16u) / 32u) * 32u]; \
    affine_nax_swiglu_bm_n16_impl<BM, SK, MODE>(                               \
        x, g_codes, nullptr, g_sb, u_codes, nullptr, u_sb, xs, y, p, parts,    \
        nullptr, lane, sg, tg);                                                \
  }

#define OMLX_M5_AFFINE_BM_SWIGLU_Q56_KERNEL(NAME, BM, SK)                      \
  [[max_total_threads_per_threadgroup((SK) * 32u)]]                            \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *g_codes [[buffer(1)]],                               \
      device const uchar *g_high [[buffer(2)]],                                \
      device const uint *g_sb [[buffer(3)]],                                   \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_high [[buffer(5)]],                                \
      device const uint *u_sb [[buffer(6)]],                                   \
      device const float *xs [[buffer(7)]],                                    \
      device bfloat *y [[buffer(8)]],                                          \
      constant AffineNaxParams &p [[buffer(9)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 2u * ((BM * 16u) / 32u) * 32u]; \
    affine_nax_swiglu_bm_n16_impl<BM, SK, 1u>(                                 \
        x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, xs, y, p, parts,      \
        nullptr, lane, sg, tg);                                                \
  }

#define OMLX_M5_AFFINE_BM_SWIGLU_Q3_KERNEL(NAME, BM, SK)                       \
  [[max_total_threads_per_threadgroup((SK) * 32u)]]                            \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *g_codes [[buffer(1)]],                               \
      device const uchar *g_high [[buffer(2)]],                                \
      device const uint *g_sb [[buffer(3)]],                                   \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_high [[buffer(5)]],                                \
      device const uint *u_sb [[buffer(6)]],                                   \
      device const float *xs [[buffer(7)]],                                    \
      device bfloat *y [[buffer(8)]],                                          \
      constant AffineNaxParams &p [[buffer(9)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    constexpr uint kPartsFloats = (SK > 1u ? SK - 1u : 1u) * 2u * ((BM * 16u) / 32u) * 32u; \
    constexpr uint kH2Floats = ((SK * 2u * 16u * 16u) + 3u) / 4u;              \
    threadgroup float scratch[kPartsFloats > kH2Floats ? kPartsFloats : kH2Floats]; \
    affine_nax_swiglu_bm_n16_impl<BM, SK, 2u>(                                 \
        x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, xs, y, p, scratch,    \
        reinterpret_cast<threadgroup uchar *>(scratch), lane, sg, tg);         \
  }

#define OMLX_M5_AFFINE_NAX_BM_N16(BM, SK)                                      \
  OMLX_M5_AFFINE_BM_NO_HIGH_KERNEL(omlx_m5_affine_nax_q4_bm##BM##_n16_sk##SK, BM##u, 16u, SK, 0u) \
  OMLX_M5_AFFINE_BM_Q56_KERNEL(omlx_m5_affine_nax_q56_bm##BM##_n16_sk##SK, BM##u, 16u, SK) \
  OMLX_M5_AFFINE_BM_Q3_KERNEL(omlx_m5_affine_nax_q3_bm##BM##_n16_sk##SK, BM##u, 16u, SK) \
  OMLX_M5_AFFINE_BM_NO_HIGH_KERNEL(omlx_m5_affine_nax_q8_bm##BM##_n16_sk##SK, BM##u, 16u, SK, 3u) \
  OMLX_M5_AFFINE_BM_SWIGLU_NO_HIGH_KERNEL(omlx_m5_affine_nax_q4_swiglu_bm##BM##_n16_sk##SK, BM##u, SK, 0u) \
  OMLX_M5_AFFINE_BM_SWIGLU_Q56_KERNEL(omlx_m5_affine_nax_q56_swiglu_bm##BM##_n16_sk##SK, BM##u, SK) \
  OMLX_M5_AFFINE_BM_SWIGLU_Q3_KERNEL(omlx_m5_affine_nax_q3_swiglu_bm##BM##_n16_sk##SK, BM##u, SK) \
  OMLX_M5_AFFINE_BM_SWIGLU_NO_HIGH_KERNEL(omlx_m5_affine_nax_q8_swiglu_bm##BM##_n16_sk##SK, BM##u, SK, 3u)

#define OMLX_M5_AFFINE_NAX_BM_N32(BM, SK)                                      \
  OMLX_M5_AFFINE_BM_NO_HIGH_KERNEL(omlx_m5_affine_nax_q4_bm##BM##_n32_sk##SK, BM##u, 32u, SK, 0u) \
  OMLX_M5_AFFINE_BM_Q56_KERNEL(omlx_m5_affine_nax_q56_bm##BM##_n32_sk##SK, BM##u, 32u, SK) \
  OMLX_M5_AFFINE_BM_Q3_KERNEL(omlx_m5_affine_nax_q3_bm##BM##_n32_sk##SK, BM##u, 32u, SK) \
  OMLX_M5_AFFINE_BM_NO_HIGH_KERNEL(omlx_m5_affine_nax_q8_bm##BM##_n32_sk##SK, BM##u, 32u, SK, 3u)

OMLX_M5_AFFINE_NAX_BM_N16(16, 1)
OMLX_M5_AFFINE_NAX_BM_N16(16, 2)
OMLX_M5_AFFINE_NAX_BM_N16(16, 4)
OMLX_M5_AFFINE_NAX_BM_N16(16, 8)
OMLX_M5_AFFINE_NAX_BM_N16(32, 1)
OMLX_M5_AFFINE_NAX_BM_N16(32, 2)
OMLX_M5_AFFINE_NAX_BM_N16(32, 4)
OMLX_M5_AFFINE_NAX_BM_N16(32, 8)
OMLX_M5_AFFINE_NAX_BM_N32(16, 1)
OMLX_M5_AFFINE_NAX_BM_N32(16, 2)
OMLX_M5_AFFINE_NAX_BM_N32(32, 1)
OMLX_M5_AFFINE_NAX_BM_N32(32, 2)

// ============================================================================
// Step 2 (M2) + Tier-1 (T1-A): Cooperative TG-SRAM Staged + SRAM-Dequant Wide SK=1
// Affine NAX (As[64][64] + Ws[64][64] = 16 KiB Linear, 24 KiB SwiGLU <= 32 KiB)
// - Zero omlx_m5_affine_nax_xsum_g64 launch needed on Wide SK=1
// - 1x bfloat x bfloat NAX matmul2d across all K (eliminates 2x q3/q5/q6 ops &
//   all per-group scalar FP32 FMA stalls)
// - L2 GROUP_M=8 swizzling for 8x SLC/L2 weight reuse
// ============================================================================

inline uint2 affine_swizzle_tg_group_m8(uint2 tg_xy, uint num_pid_n, uint num_pid_m) {
  constexpr uint GROUP_M = 8u;
  if (num_pid_n == 0u || num_pid_m == 0u) {
    return tg_xy;
  }
  const uint pid = tg_xy.y * num_pid_n + tg_xy.x;
  const uint num_pid_in_group = GROUP_M * num_pid_n;
  const uint group_id = pid / num_pid_in_group;
  const uint first_pid_m = group_id * GROUP_M;
  const uint group_size_m = min(num_pid_m - first_pid_m, GROUP_M);
  const uint pid_m = first_pid_m + (pid % group_size_m);
  const uint pid_n = (pid % num_pid_in_group) / group_size_m;
  return uint2(pid_n, pid_m);
}

inline void affine_mma_16x32x16_bf16(
    thread vec<float, 8> &Cn0,
    thread vec<float, 8> &Cn1,
    thread const vec<bfloat, 8> &A,
    thread const vec<bfloat, 8> &Bn0,
    thread const vec<bfloat, 8> &Bn1) {
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b =
      gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>,
      remove_addrspace_t<decltype(ct_b)>,
      float>();
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 8u; ++i) {
    ct_a[i] = A[i];
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 8u; ++i) {
    ct_b[i] = Bn0[i];
    ct_b[8u + i] = Bn1[i];
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 8u; ++i) {
    ct_c[i] = Cn0[i];
    ct_c[8u + i] = Cn1[i];
  }
  gemm_op.run(ct_a, ct_b, ct_c);
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 8u; ++i) {
    Cn0[i] = ct_c[i];
    Cn1[i] = ct_c[8u + i];
  }
}

inline bool affine_has_apple10_nax_layout() {
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  const auto c0 = ct_a.get_multidimensional_index(ushort(0u));
  const auto c2 = ct_a.get_multidimensional_index(ushort(2u));
  return (int(c2[0]) - int(c0[0]) == 2) && (int(c2[1]) == int(c0[1]));
}

template <uint Mode>
struct AffineRaw32 {
  uint sb_u;
  uint4 r0;
  uint4 r1;
};

template <uint Mode, uint BN_STORED>
inline AffineRaw32<Mode> affine_fetch_raw_32_fast(
    device const uchar *wc_row,
    device const uchar *wh_row,
    device const uint *sb_col_base,
    uint padded_n,
    uint g,
    uint half_k) {
  AffineRaw32<Mode> out;
  out.sb_u = sb_col_base[ulong(g) * ulong(padded_n)];
  out.r1 = uint4(0u);
  if constexpr (Mode == 0u) {
    out.r0 = reinterpret_cast<device const uint4 *>(
        wc_row + ulong(g) * (ulong(BN_STORED) * 32ul))[half_k];
  } else if constexpr (Mode == 1u) {
    out.r0 = reinterpret_cast<device const uint4 *>(
        wc_row + ulong(g) * (ulong(BN_STORED) * 32ul))[half_k];
    const uint2 hi2 = reinterpret_cast<device const uint2 *>(
        wh_row + ulong(g) * (ulong(BN_STORED) * 16ul))[half_k];
    out.r1 = uint4(hi2.x, hi2.y, 0u, 0u);
  } else if constexpr (Mode == 2u) {
    const uint2 lo2 = reinterpret_cast<device const uint2 *>(
        wc_row + ulong(g) * (ulong(BN_STORED) * 16ul))[half_k];
    const uint hi1 = reinterpret_cast<device const uint *>(
        wh_row + ulong(g) * (ulong(BN_STORED) * 8ul))[half_k];
    out.r0 = uint4(lo2.x, lo2.y, hi1, 0u);
  } else {
    device const uint4 *p4 = reinterpret_cast<device const uint4 *>(
        wc_row + ulong(g) * (ulong(BN_STORED) * 64ul)) + half_k * 2u;
    out.r0 = p4[0];
    out.r1 = p4[1];
  }
  return out;
}

template <uint Mode, uint BN_STORED>
inline AffineRaw32<Mode> affine_fetch_raw_32_checked(
    device const uchar *codes,
    device const uchar *high_plane,
    device const uint *sb,
    uint col,
    uint n_limit,
    uint padded_n,
    uint groups,
    uint g,
    uint half_k) {
  AffineRaw32<Mode> out = {0u, uint4(0u), uint4(0u)};
  if (col >= n_limit) {
    return out;
  }
  const uint tile_n = col / BN_STORED;
  const uint rn = col % BN_STORED;
  const ulong tile_base = ulong(tile_n) * ulong(groups);
  out.sb_u = sb[ulong(g) * ulong(padded_n) + col];
  if constexpr (Mode == 0u) {
    out.r0 = reinterpret_cast<device const uint4 *>(
        codes + (tile_base + ulong(g)) * (ulong(BN_STORED) * 32ul) + ulong(rn) * 32ul)[half_k];
  } else if constexpr (Mode == 1u) {
    out.r0 = reinterpret_cast<device const uint4 *>(
        codes + (tile_base + ulong(g)) * (ulong(BN_STORED) * 32ul) + ulong(rn) * 32ul)[half_k];
    const uint2 hi2 = reinterpret_cast<device const uint2 *>(
        high_plane + (tile_base + ulong(g)) * (ulong(BN_STORED) * 16ul) + ulong(rn) * 16ul)[half_k];
    out.r1 = uint4(hi2.x, hi2.y, 0u, 0u);
  } else if constexpr (Mode == 2u) {
    const uint2 lo2 = reinterpret_cast<device const uint2 *>(
        codes + (tile_base + ulong(g)) * (ulong(BN_STORED) * 16ul) + ulong(rn) * 16ul)[half_k];
    const uint hi1 = reinterpret_cast<device const uint *>(
        high_plane + (tile_base + ulong(g)) * (ulong(BN_STORED) * 8ul) + ulong(rn) * 8ul)[half_k];
    out.r0 = uint4(lo2.x, lo2.y, hi1, 0u);
  } else {
    device const uint4 *p4 = reinterpret_cast<device const uint4 *>(
        codes + (tile_base + ulong(g)) * (ulong(BN_STORED) * 64ul) + ulong(rn) * 64ul) + half_k * 2u;
    out.r0 = p4[0];
    out.r1 = p4[1];
  }
  return out;
}

template <uint Mode>
inline void affine_unpack_raw_32(
    const AffineRaw32<Mode> raw,
    threadgroup vec<bfloat, 8> *w_dst8) {
  const vec<bfloat, 2> pair = as_type<vec<bfloat, 2>>(raw.sb_u);
  const float sc = float(pair[0]);
  const float bi = float(pair[1]);
  if constexpr (Mode == 0u) {
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      const uint wd = raw.r0[w];
      vec<bfloat, 8> out;
      #pragma clang loop unroll(full)
      for (uint j = 0u; j < 8u; ++j) {
        out[j] = bfloat(fma(sc, float((wd >> (j * 4u)) & 0x0fu), bi));
      }
      w_dst8[w] = out;
    }
  } else if constexpr (Mode == 1u) {
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      const uint wd = raw.r0[w];
      const uint h16 = (w < 2u) ? (raw.r1.x >> (w * 16u)) : (raw.r1.y >> ((w - 2u) * 16u));
      vec<bfloat, 8> out;
      #pragma clang loop unroll(full)
      for (uint j = 0u; j < 8u; ++j) {
        out[j] = bfloat(fma(sc, float(((wd >> (j * 4u)) & 0x0fu) | (((h16 >> (j * 2u)) & 3u) << 4u)), bi));
      }
      w_dst8[w] = out;
    }
  } else if constexpr (Mode == 2u) {
    #pragma clang loop unroll(full)
    for (uint q16 = 0u; q16 < 2u; ++q16) {
      const uint lword = (q16 == 0u) ? raw.r0.x : raw.r0.y;
      const uint h16 = (raw.r0.z >> (q16 * 16u)) & 0xffffu;
      #pragma clang loop unroll(full)
      for (uint half8 = 0u; half8 < 2u; ++half8) {
        const uint l16 = lword >> (half8 * 16u);
        const uint h8 = h16 >> (half8 * 8u);
        vec<bfloat, 8> out;
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j) {
          out[j] = bfloat(fma(sc, float(((l16 >> (j * 2u)) & 3u) | (((h8 >> j) & 1u) << 2u)), bi));
        }
        w_dst8[q16 * 2u + half8] = out;
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (uint q16 = 0u; q16 < 2u; ++q16) {
      const uint4 pk = (q16 == 0u) ? raw.r0 : raw.r1;
      #pragma clang loop unroll(full)
      for (uint half8 = 0u; half8 < 2u; ++half8) {
        const uint w0 = pk[half8 * 2u + 0u];
        const uint w1 = pk[half8 * 2u + 1u];
        w_dst8[q16 * 2u + half8] = vec<bfloat, 8>(
            bfloat(fma(sc, float((w0 >>  0u) & 0xffu), bi)),
            bfloat(fma(sc, float((w0 >>  8u) & 0xffu), bi)),
            bfloat(fma(sc, float((w0 >> 16u) & 0xffu), bi)),
            bfloat(fma(sc, float((w0 >> 24u) & 0xffu), bi)),
            bfloat(fma(sc, float((w1 >>  0u) & 0xffu), bi)),
            bfloat(fma(sc, float((w1 >>  8u) & 0xffu), bi)),
            bfloat(fma(sc, float((w1 >> 16u) & 0xffu), bi)),
            bfloat(fma(sc, float((w1 >> 24u) & 0xffu), bi)));
      }
    }
  }
}

template <uint Mode, uint BN_STORED, uint WM, uint WN>
inline void affine_nax_wide_sk1_linear_apple9(
    device const bfloat *x,
    device const uchar *codes,
    device const uchar *high_plane,
    device const uint *sb,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup bfloat *Ws,
    uint tid,
    uint sg_m,
    uint sg_n,
    uint rb_tg,
    uint n0_tg) {
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint SM = 32u;
  constexpr uint SN = 32u;
  constexpr uint BN_TG = WN * SN;
  constexpr uint TPTG = WM * WN * 32u;
  constexpr uint kFetches = (BN_TG * 2u) / TPTG;

  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * SN;
  const bool sg_active = (sg_m < WM) && (sg_n < WN) && (rb_sg < p.rows) && (n0_sg < p.n);
  const uint groups = p.k / kBK;

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_c0 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
  auto ct_c1 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 16u; ++i) {
    ct_c0[i] = 0.0f;
    ct_c1[i] = 0.0f;
  }

  for (uint g = 0u; g < groups; ++g) {
    const uint k_off = g * kBK;
    AffineRaw32<Mode> raw_f[kFetches];
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      raw_f[f] = affine_fetch_raw_32_checked<Mode, BN_STORED>(
          codes, high_plane, sb, n0_tg + (idx32 >> 1u), p.n, p.padded_n,
          groups, g, idx32 & 1u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      threadgroup vec<bfloat, 8> *w_dst8 = reinterpret_cast<threadgroup vec<bfloat, 8> *>(
          Ws + (idx32 >> 1u) * kBKP + (idx32 & 1u) * 32u);
      affine_unpack_raw_32<Mode>(raw_f[f], w_dst8);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_active) {
      #pragma clang loop unroll(full)
      for (uint kk = 0u; kk < kBK; kk += 16u) {
        const threadgroup bfloat *ws_sub = Ws + (sg_n * SN) * kBKP + kk;
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          const auto idx_b = ct_b.get_multidimensional_index(i);
          ct_b[i] = ws_sub[uint(idx_b[1]) * kBKP + uint(idx_b[0])];
        }
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 8u; ++i) {
          const auto idx_a = ct_a.get_multidimensional_index(i);
          const uint r0 = rb_sg + uint(idx_a[1]);
          ct_a[i] = (r0 < p.rows) ? x[ulong(r0) * p.k + k_off + kk + uint(idx_a[0])] : bfloat(0.0f);
        }
        gemm_op.run(ct_a, ct_b, ct_c0);
        if (rb_sg + 16u < p.rows) {
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 8u; ++i) {
            const auto idx_a = ct_a.get_multidimensional_index(i);
            const uint r1 = rb_sg + 16u + uint(idx_a[1]);
            ct_a[i] = (r1 < p.rows) ? x[ulong(r1) * p.k + k_off + kk + uint(idx_a[0])] : bfloat(0.0f);
          }
          gemm_op.run(ct_a, ct_b, ct_c1);
        }
      }
    }
  }

  if (sg_active) {
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      const auto idx = ct_c0.get_multidimensional_index(i);
      const uint col = n0_sg + uint(idx[0]);
      const uint r0 = rb_sg + uint(idx[1]);
      if (r0 < p.rows && col < p.n) {
        y[ulong(r0) * p.n + col] = bfloat(ct_c0[i]);
      }
      const uint r1 = r0 + 16u;
      if (r1 < p.rows && col < p.n) {
        y[ulong(r1) * p.n + col] = bfloat(ct_c1[i]);
      }
    }
  }
}

template <uint Mode, uint WM, uint WN>
inline void affine_nax_wide_sk1_swiglu_apple9(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const uint *g_sb,
    device const uchar *u_codes,
    device const uchar *u_high,
    device const uint *u_sb,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup bfloat *Ws,
    uint tid,
    uint sg_m,
    uint sg_n,
    uint rb_tg,
    uint n0_tg) {
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint SM = 32u;
  constexpr uint OUT_SN = 16u;
  constexpr uint WS_ROWS = WN * 32u;
  constexpr uint TPTG = WM * WN * 32u;
  constexpr uint kFetches = (WS_ROWS * 2u) / TPTG;

  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * OUT_SN;
  const bool sg_active = (sg_m < WM) && (sg_n < WN) && (rb_sg < p.rows) && (n0_sg < p.n);
  const uint groups = p.k / kBK;

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_c0 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
  auto ct_c1 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 16u; ++i) {
    ct_c0[i] = 0.0f;
    ct_c1[i] = 0.0f;
  }

  for (uint g = 0u; g < groups; ++g) {
    const uint k_off = g * kBK;
    AffineRaw32<Mode> raw_f[kFetches];
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      const uint w_row = idx32 >> 1u;
      const bool is_up = ((w_row >> 4u) & 1u) != 0u;
      const uint out_col = n0_tg + ((w_row >> 5u) * 16u) + (w_row & 15u);
      raw_f[f] = affine_fetch_raw_32_checked<Mode, 16u>(
          is_up ? u_codes : g_codes, is_up ? u_high : g_high,
          is_up ? u_sb : g_sb, out_col, p.n, p.padded_n, groups, g, idx32 & 1u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      threadgroup vec<bfloat, 8> *w_dst8 = reinterpret_cast<threadgroup vec<bfloat, 8> *>(
          Ws + (idx32 >> 1u) * kBKP + (idx32 & 1u) * 32u);
      affine_unpack_raw_32<Mode>(raw_f[f], w_dst8);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_active) {
      #pragma clang loop unroll(full)
      for (uint kk = 0u; kk < kBK; kk += 16u) {
        const threadgroup bfloat *ws_sub = Ws + (sg_n * 32u) * kBKP + kk;
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          const auto idx_b = ct_b.get_multidimensional_index(i);
          ct_b[i] = ws_sub[uint(idx_b[1]) * kBKP + uint(idx_b[0])];
        }
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 8u; ++i) {
          const auto idx_a = ct_a.get_multidimensional_index(i);
          const uint r0 = rb_sg + uint(idx_a[1]);
          ct_a[i] = (r0 < p.rows) ? x[ulong(r0) * p.k + k_off + kk + uint(idx_a[0])] : bfloat(0.0f);
        }
        gemm_op.run(ct_a, ct_b, ct_c0);
        if (rb_sg + 16u < p.rows) {
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 8u; ++i) {
            const auto idx_a = ct_a.get_multidimensional_index(i);
            const uint r1 = rb_sg + 16u + uint(idx_a[1]);
            ct_a[i] = (r1 < p.rows) ? x[ulong(r1) * p.k + k_off + kk + uint(idx_a[0])] : bfloat(0.0f);
          }
          gemm_op.run(ct_a, ct_b, ct_c1);
        }
      }
    }
  }

  if (sg_active) {
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm) {
      if (rb_sg + mm * 16u < p.rows) {
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          const auto idx_g = ct_c0.get_multidimensional_index(i);
          const uint c_g = uint(idx_g[0]);
          const uint r_g = uint(idx_g[1]);
          if (c_g < 16u) {
            const uint col = n0_sg + c_g;
            const uint row = rb_sg + mm * 16u + r_g;
            if (row < p.rows && col < p.n) {
              float gv = (mm == 0u) ? ct_c0[i] : ct_c1[i];
              float uv = 0.0f;
              #pragma clang loop unroll(full)
              for (ushort j = 0u; j < 16u; ++j) {
                const auto idx_u = ct_c0.get_multidimensional_index(j);
                if (uint(idx_u[0]) == c_g + 16u && uint(idx_u[1]) == r_g) {
                  uv = (mm == 0u) ? ct_c0[j] : ct_c1[j];
                }
              }
              gv = float(bfloat(gv));
              uv = float(bfloat(uv));
              y[ulong(row) * p.n + col] = bfloat(nax_silu_f32(gv) * uv);
            }
          }
        }
      }
    }
  }
}

template <uint Mode, uint GroupSize, uint BN_STORED, uint WM, uint WN>
inline void affine_nax_wide_sk1_linear_impl(
    device const bfloat *x,
    device const uchar *codes,
    device const uchar *high_plane,
    device const uint *sb,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup bfloat *Ws,
    uint lane,
    uint sg,
    uint3 tg) {
  (void)GroupSize;
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint SM = 32u;
  constexpr uint SN = 32u;
  constexpr uint BM_TG = WM * SM;
  constexpr uint BN_TG = WN * SN;
  constexpr uint TPTG = WM * WN * 32u;
  constexpr uint kFetches = (BN_TG * 2u) / TPTG;

  const uint num_pid_n = (p.n + BN_TG - 1u) / BN_TG;
  const uint num_pid_m = (p.mp + BM_TG - 1u) / BM_TG;
  const uint2 pid_nm = affine_swizzle_tg_group_m8(tg.xy, num_pid_n, num_pid_m);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg = pid_nm.x * BN_TG;
  if (rb_tg >= p.rows || n0_tg >= p.n) {
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint sg_m = (WM > 1u) ? (sg / WN) : 0u;
  const uint sg_n = (WN > 1u) ? (sg % WN) : sg;
  if (!affine_has_apple10_nax_layout()) {
    affine_nax_wide_sk1_linear_apple9<Mode, BN_STORED, WM, WN>(
        x, codes, high_plane, sb, y, p, Ws, tid, sg_m, sg_n, rb_tg, n0_tg);
    return;
  }

  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * SN;
  const uint groups = p.k / kBK;
  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

  vec<float, 8> D[2][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < 2u; ++mm) {
    D[mm][0] = vec<float, 8>(0.0f);
    D[mm][1] = vec<float, 8>(0.0f);
  }

  constexpr ulong kCodeBytes = (Mode == 2u) ? 16ul : (Mode == 3u) ? 64ul : 32ul;
  constexpr ulong kHighBytes = (Mode == 1u) ? 16ul : (Mode == 2u) ? 8ul : 0ul;
  const bool full_tile = (rb_tg + BM_TG <= p.rows) && (n0_tg + BN_TG <= p.n);
  if (full_tile) {
    device const uchar *wc_row[kFetches];
    device const uchar *wh_row[kFetches];
    device const uint *sb_col[kFetches];
    uint half_k_f[kFetches];
    threadgroup vec<bfloat, 8> *w_dst8[kFetches];
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      const uint w_row = idx32 >> 1u;
      const uint half_k = idx32 & 1u;
      const uint col = n0_tg + w_row;
      const uint tile_n = col / BN_STORED;
      const uint rn = col % BN_STORED;
      const ulong tile_base = ulong(tile_n) * ulong(groups);
      wc_row[f] = codes + tile_base * (ulong(BN_STORED) * kCodeBytes) + ulong(rn) * kCodeBytes;
      wh_row[f] = (kHighBytes > 0ul)
          ? (high_plane + tile_base * (ulong(BN_STORED) * kHighBytes) + ulong(rn) * kHighBytes)
          : nullptr;
      sb_col[f] = sb + col;
      half_k_f[f] = half_k;
      w_dst8[f] = reinterpret_cast<threadgroup vec<bfloat, 8> *>(Ws + w_row * kBKP + half_k * 32u);
    }
    device const bfloat *x_sg = x + ulong(rb_sg + fm) * p.k + fn;
    const threadgroup bfloat *ws_sg = Ws + (sg_n * 32u + fm) * kBKP + fn;

    for (uint g = 0u; g < groups; ++g) {
      const uint k0 = g * kBK;
      AffineRaw32<Mode> raw_f[kFetches];
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        raw_f[f] = affine_fetch_raw_32_fast<Mode, BN_STORED>(
            wc_row[f], wh_row[f], sb_col[f], p.padded_n, g, half_k_f[f]);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        affine_unpack_raw_32<Mode>(raw_f[f], w_dst8[f]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      #pragma clang loop unroll(full)
      for (uint kk = 0u; kk < kBK; kk += 32u) {
        volatile int cbar;
        vec<bfloat, 8> a00, a01, a10, a11, b00, b01, b10, b11;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 2u; ++r) {
          const bfloat4 va0 = *reinterpret_cast<device const bfloat4 *>(x_sg + ulong(r * 8u) * p.k + k0 + kk + 0u);
          const bfloat4 va1 = *reinterpret_cast<device const bfloat4 *>(x_sg + ulong(r * 8u) * p.k + k0 + kk + 16u);
          const bfloat4 va2 = *reinterpret_cast<device const bfloat4 *>(x_sg + ulong(16u + r * 8u) * p.k + k0 + kk + 0u);
          const bfloat4 va3 = *reinterpret_cast<device const bfloat4 *>(x_sg + ulong(16u + r * 8u) * p.k + k0 + kk + 16u);
          const bfloat4 vb0 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 0u);
          const bfloat4 vb1 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 16u);
          const bfloat4 vb2 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 0u);
          const bfloat4 vb3 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 16u);
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            a00[r * 4u + c] = va0[c]; a01[r * 4u + c] = va1[c];
            a10[r * 4u + c] = va2[c]; a11[r * 4u + c] = va3[c];
            b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
            b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
          }
        }
        affine_mma_16x32x16_bf16(D[0][0], D[0][1], a00, b00, b10);
        affine_mma_16x32x16_bf16(D[0][0], D[0][1], a01, b01, b11);
        affine_mma_16x32x16_bf16(D[1][0], D[1][1], a10, b00, b10);
        affine_mma_16x32x16_bf16(D[1][0], D[1][1], a11, b01, b11);
        (void)cbar;
      }
    }
  } else {
    const bool sg_active = (sg_m < WM) && (sg_n < WN) && (rb_sg < p.rows) && (n0_sg < p.n);
    const threadgroup bfloat *ws_sg = Ws + (sg_n * SN + fm) * kBKP + fn;
    for (uint g = 0u; g < groups; ++g) {
      const uint k_off = g * kBK;
      AffineRaw32<Mode> raw_f[kFetches];
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint idx32 = tid + f * TPTG;
        raw_f[f] = affine_fetch_raw_32_checked<Mode, BN_STORED>(
            codes, high_plane, sb, n0_tg + (idx32 >> 1u), p.n, p.padded_n,
            groups, g, idx32 & 1u);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint idx32 = tid + f * TPTG;
        threadgroup vec<bfloat, 8> *w_dst8 = reinterpret_cast<threadgroup vec<bfloat, 8> *>(
            Ws + (idx32 >> 1u) * kBKP + (idx32 & 1u) * 32u);
        affine_unpack_raw_32<Mode>(raw_f[f], w_dst8);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_active) {
        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < kBK; kk += 32u) {
          volatile int cbar;
          vec<bfloat, 8> b00, b01, b10, b11;
          #pragma clang loop unroll(full)
          for (uint r = 0u; r < 2u; ++r) {
            const bfloat4 vb0 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 0u);
            const bfloat4 vb1 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 16u);
            const bfloat4 vb2 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 0u);
            const bfloat4 vb3 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 16u);
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) {
              b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
              b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
            }
          }
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            if (rb_sg + mm * 16u < p.rows) {
              vec<bfloat, 8> a0, a1;
              #pragma clang loop unroll(full)
              for (uint r = 0u; r < 2u; ++r) {
                const uint row = rb_sg + mm * 16u + fm + r * 8u;
                const bfloat4 va0 = (row < p.rows)
                    ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p.k + k_off + kk + fn)
                    : bfloat4(bfloat(0.0f));
                const bfloat4 va1 = (row < p.rows)
                    ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p.k + k_off + kk + 16u + fn)
                    : bfloat4(bfloat(0.0f));
                #pragma clang loop unroll(full)
                for (uint c = 0u; c < 4u; ++c) {
                  a0[r * 4u + c] = va0[c];
                  a1[r * 4u + c] = va1[c];
                }
              }
              affine_mma_16x32x16_bf16(D[mm][0], D[mm][1], a0, b00, b10);
              affine_mma_16x32x16_bf16(D[mm][0], D[mm][1], a1, b01, b11);
            }
          }
          (void)cbar;
        }
      }
    }
  }

  const bool sg_active = (sg_m < WM) && (sg_n < WN) && (rb_sg < p.rows) && (n0_sg < p.n);
  if (sg_active) {
    const bool aligned_n = ((p.n & 3u) == 0u);
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm) {
      if (rb_sg + mm * 16u < p.rows) {
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint row = rb_sg + mm * 16u + fm + h * 8u;
          if (row < p.rows) {
            const uint e = h * 4u;
            #pragma clang loop unroll(full)
            for (uint nn = 0u; nn < 2u; ++nn) {
              const uint col = n0_sg + nn * 16u + fn;
              if (col < p.n) {
                const bfloat4 out_v = bfloat4(
                    bfloat(D[mm][nn][e + 0u]),
                    bfloat(D[mm][nn][e + 1u]),
                    bfloat(D[mm][nn][e + 2u]),
                    bfloat(D[mm][nn][e + 3u]));
                if (aligned_n && col + 4u <= p.n) {
                  *reinterpret_cast<device bfloat4 *>(y + ulong(row) * p.n + col) = out_v;
                } else {
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    if (col + j < p.n) {
                      y[ulong(row) * p.n + col + j] = out_v[j];
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template <uint Mode, uint GroupSize, uint WM, uint WN>
inline void affine_nax_wide_sk1_swiglu_impl(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const uint *g_sb,
    device const uchar *u_codes,
    device const uchar *u_high,
    device const uint *u_sb,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup bfloat *Ws,
    uint lane,
    uint sg,
    uint3 tg) {
  (void)GroupSize;
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint SM = 32u;
  constexpr uint OUT_SN = 16u;
  constexpr uint BM_TG = WM * SM;
  constexpr uint OUT_N_TG = WN * OUT_SN;
  constexpr uint WS_ROWS = WN * 32u;
  constexpr uint TPTG = WM * WN * 32u;
  constexpr uint kFetches = (WS_ROWS * 2u) / TPTG;

  const uint num_pid_n = (p.n + OUT_N_TG - 1u) / OUT_N_TG;
  const uint num_pid_m = (p.mp + BM_TG - 1u) / BM_TG;
  const uint2 pid_nm = affine_swizzle_tg_group_m8(tg.xy, num_pid_n, num_pid_m);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg = pid_nm.x * OUT_N_TG;
  if (rb_tg >= p.rows || n0_tg >= p.n) {
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint sg_m = (WM > 1u) ? (sg / WN) : 0u;
  const uint sg_n = (WN > 1u) ? (sg % WN) : sg;
  if (!affine_has_apple10_nax_layout()) {
    affine_nax_wide_sk1_swiglu_apple9<Mode, WM, WN>(
        x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p, Ws, tid,
        sg_m, sg_n, rb_tg, n0_tg);
    return;
  }

  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * OUT_SN;
  const uint groups = p.k / kBK;
  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

  vec<float, 8> D[2][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < 2u; ++mm) {
    D[mm][0] = vec<float, 8>(0.0f);
    D[mm][1] = vec<float, 8>(0.0f);
  }

  constexpr ulong kCodeBytes = (Mode == 2u) ? 16ul : (Mode == 3u) ? 64ul : 32ul;
  constexpr ulong kHighBytes = (Mode == 1u) ? 16ul : (Mode == 2u) ? 8ul : 0ul;
  const bool full_tile = (rb_tg + BM_TG <= p.rows) && (n0_tg + OUT_N_TG <= p.n);
  if (full_tile) {
    device const uchar *wc_row[kFetches];
    device const uchar *wh_row[kFetches];
    device const uint *sb_col[kFetches];
    uint half_k_f[kFetches];
    threadgroup vec<bfloat, 8> *w_dst8[kFetches];
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      const uint w_row = idx32 >> 1u;
      const uint half_k = idx32 & 1u;
      const bool is_up = ((w_row >> 4u) & 1u) != 0u;
      const uint out_col = n0_tg + ((w_row >> 5u) * 16u) + (w_row & 15u);
      const uint tile_n = out_col >> 4u;
      const uint rn = out_col & 15u;
      const ulong tile_base = ulong(tile_n) * ulong(groups);
      device const uchar *sel_c = is_up ? u_codes : g_codes;
      device const uchar *sel_h = is_up ? u_high : g_high;
      wc_row[f] = sel_c + tile_base * (16ul * kCodeBytes) + ulong(rn) * kCodeBytes;
      wh_row[f] = (kHighBytes > 0ul)
          ? (sel_h + tile_base * (16ul * kHighBytes) + ulong(rn) * kHighBytes)
          : nullptr;
      sb_col[f] = (is_up ? u_sb : g_sb) + out_col;
      half_k_f[f] = half_k;
      w_dst8[f] = reinterpret_cast<threadgroup vec<bfloat, 8> *>(Ws + w_row * kBKP + half_k * 32u);
    }
    device const bfloat *x_sg = x + ulong(rb_sg + fm) * p.k + fn;
    const threadgroup bfloat *ws_sg = Ws + (sg_n * 32u + fm) * kBKP + fn;

    for (uint g = 0u; g < groups; ++g) {
      const uint k0 = g * kBK;
      AffineRaw32<Mode> raw_f[kFetches];
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        raw_f[f] = affine_fetch_raw_32_fast<Mode, 16u>(
            wc_row[f], wh_row[f], sb_col[f], p.padded_n, g, half_k_f[f]);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        affine_unpack_raw_32<Mode>(raw_f[f], w_dst8[f]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      #pragma clang loop unroll(full)
      for (uint kk = 0u; kk < kBK; kk += 32u) {
        volatile int cbar;
        vec<bfloat, 8> a00, a01, a10, a11, b00, b01, b10, b11;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 2u; ++r) {
          const bfloat4 va0 = *reinterpret_cast<device const bfloat4 *>(x_sg + ulong(r * 8u) * p.k + k0 + kk + 0u);
          const bfloat4 va1 = *reinterpret_cast<device const bfloat4 *>(x_sg + ulong(r * 8u) * p.k + k0 + kk + 16u);
          const bfloat4 va2 = *reinterpret_cast<device const bfloat4 *>(x_sg + ulong(16u + r * 8u) * p.k + k0 + kk + 0u);
          const bfloat4 va3 = *reinterpret_cast<device const bfloat4 *>(x_sg + ulong(16u + r * 8u) * p.k + k0 + kk + 16u);
          const bfloat4 vb0 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 0u);
          const bfloat4 vb1 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 16u);
          const bfloat4 vb2 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 0u);
          const bfloat4 vb3 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 16u);
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            a00[r * 4u + c] = va0[c]; a01[r * 4u + c] = va1[c];
            a10[r * 4u + c] = va2[c]; a11[r * 4u + c] = va3[c];
            b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
            b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
          }
        }
        affine_mma_16x32x16_bf16(D[0][0], D[0][1], a00, b00, b10);
        affine_mma_16x32x16_bf16(D[0][0], D[0][1], a01, b01, b11);
        affine_mma_16x32x16_bf16(D[1][0], D[1][1], a10, b00, b10);
        affine_mma_16x32x16_bf16(D[1][0], D[1][1], a11, b01, b11);
        (void)cbar;
      }
    }
  } else {
    const bool sg_active = (sg_m < WM) && (sg_n < WN) && (rb_sg < p.rows) && (n0_sg < p.n);
    const threadgroup bfloat *ws_sg = Ws + (sg_n * 32u + fm) * kBKP + fn;
    for (uint g = 0u; g < groups; ++g) {
      const uint k_off = g * kBK;
      AffineRaw32<Mode> raw_f[kFetches];
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint idx32 = tid + f * TPTG;
        const uint w_row = idx32 >> 1u;
        const bool is_up = ((w_row >> 4u) & 1u) != 0u;
        const uint out_col = n0_tg + ((w_row >> 5u) * 16u) + (w_row & 15u);
        raw_f[f] = affine_fetch_raw_32_checked<Mode, 16u>(
            is_up ? u_codes : g_codes, is_up ? u_high : g_high,
            is_up ? u_sb : g_sb, out_col, p.n, p.padded_n, groups, g, idx32 & 1u);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint idx32 = tid + f * TPTG;
        threadgroup vec<bfloat, 8> *w_dst8 = reinterpret_cast<threadgroup vec<bfloat, 8> *>(
            Ws + (idx32 >> 1u) * kBKP + (idx32 & 1u) * 32u);
        affine_unpack_raw_32<Mode>(raw_f[f], w_dst8);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_active) {
        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < kBK; kk += 32u) {
          volatile int cbar;
          vec<bfloat, 8> b00, b01, b10, b11;
          #pragma clang loop unroll(full)
          for (uint r = 0u; r < 2u; ++r) {
            const bfloat4 vb0 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 0u);
            const bfloat4 vb1 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 16u);
            const bfloat4 vb2 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 0u);
            const bfloat4 vb3 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 16u);
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) {
              b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
              b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
            }
          }
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            if (rb_sg + mm * 16u < p.rows) {
              vec<bfloat, 8> a0, a1;
              #pragma clang loop unroll(full)
              for (uint r = 0u; r < 2u; ++r) {
                const uint row = rb_sg + mm * 16u + fm + r * 8u;
                const bfloat4 va0 = (row < p.rows)
                    ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p.k + k_off + kk + fn)
                    : bfloat4(bfloat(0.0f));
                const bfloat4 va1 = (row < p.rows)
                    ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p.k + k_off + kk + 16u + fn)
                    : bfloat4(bfloat(0.0f));
                #pragma clang loop unroll(full)
                for (uint c = 0u; c < 4u; ++c) {
                  a0[r * 4u + c] = va0[c];
                  a1[r * 4u + c] = va1[c];
                }
              }
              affine_mma_16x32x16_bf16(D[mm][0], D[mm][1], a0, b00, b10);
              affine_mma_16x32x16_bf16(D[mm][0], D[mm][1], a1, b01, b11);
            }
          }
          (void)cbar;
        }
      }
    }
  }

  const bool sg_active = (sg_m < WM) && (sg_n < WN) && (rb_sg < p.rows) && (n0_sg < p.n);
  if (sg_active) {
    const bool aligned_n = ((p.n & 3u) == 0u);
    const uint col = n0_sg + fn;
    if (col < p.n) {
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        if (rb_sg + mm * 16u < p.rows) {
          #pragma clang loop unroll(full)
          for (uint h = 0u; h < 2u; ++h) {
            const uint row = rb_sg + mm * 16u + fm + h * 8u;
            if (row < p.rows) {
              const uint e = h * 4u;
              bfloat4 out_b;
              #pragma clang loop unroll(full)
              for (uint j = 0u; j < 4u; ++j) {
                const float gv = float(bfloat(D[mm][0][e + j]));
                const float uv = float(bfloat(D[mm][1][e + j]));
                out_b[j] = bfloat(nax_silu_f32(gv) * uv);
              }
              if (aligned_n && col + 4u <= p.n) {
                *reinterpret_cast<device bfloat4 *>(y + ulong(row) * p.n + col) = out_b;
              } else {
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  if (col + j < p.n) {
                    y[ulong(row) * p.n + col + j] = out_b[j];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

#define OMLX_M5_AFFINE_WIDE_NO_HIGH_LINEAR(NAME, MODE, GS, BNS, WM, WN)        \
  [[max_total_threads_per_threadgroup((WM) * (WN) * 32u)]]                     \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *codes [[buffer(1)]],                                 \
      device const uint *sb [[buffer(2)]],                                     \
      device bfloat *y [[buffer(3)]],                                          \
      constant AffineNaxParams &p [[buffer(4)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[(WN) * 32u * 72u];                       \
    affine_nax_wide_sk1_linear_impl<MODE, GS, BNS, WM, WN>(                    \
        x, codes, nullptr, sb, y, p, Ws, lane, sg, tg);                        \
  }

#define OMLX_M5_AFFINE_WIDE_Q56_LINEAR(NAME, GS, BNS, WM, WN)                  \
  [[max_total_threads_per_threadgroup((WM) * (WN) * 32u)]]                     \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *codes [[buffer(1)]],                                 \
      device const uchar *high2 [[buffer(2)]],                                 \
      device const uint *sb [[buffer(3)]],                                     \
      device bfloat *y [[buffer(4)]],                                          \
      constant AffineNaxParams &p [[buffer(5)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[(WN) * 32u * 72u];                       \
    affine_nax_wide_sk1_linear_impl<1u, GS, BNS, WM, WN>(                      \
        x, codes, high2, sb, y, p, Ws, lane, sg, tg);                          \
  }

#define OMLX_M5_AFFINE_WIDE_Q3_LINEAR(NAME, GS, BNS, WM, WN)                   \
  [[max_total_threads_per_threadgroup((WM) * (WN) * 32u)]]                     \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *codes [[buffer(1)]],                                 \
      device const uchar *high1 [[buffer(2)]],                                 \
      device const uint *sb [[buffer(3)]],                                     \
      device bfloat *y [[buffer(4)]],                                          \
      constant AffineNaxParams &p [[buffer(5)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[(WN) * 32u * 72u];                       \
    affine_nax_wide_sk1_linear_impl<2u, GS, BNS, WM, WN>(                      \
        x, codes, high1, sb, y, p, Ws, lane, sg, tg);                          \
  }

#define OMLX_M5_AFFINE_WIDE_NO_HIGH_SWIGLU(NAME, MODE, GS, WM, WN)             \
  [[max_total_threads_per_threadgroup((WM) * (WN) * 32u)]]                     \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *g_codes [[buffer(1)]],                               \
      device const uint *g_sb [[buffer(2)]],                                   \
      device const uchar *u_codes [[buffer(3)]],                               \
      device const uint *u_sb [[buffer(4)]],                                   \
      device bfloat *y [[buffer(5)]],                                          \
      constant AffineNaxParams &p [[buffer(6)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[(WN) * 32u * 72u];                       \
    affine_nax_wide_sk1_swiglu_impl<MODE, GS, WM, WN>(                         \
        x, g_codes, nullptr, g_sb, u_codes, nullptr, u_sb, y, p, Ws,           \
        lane, sg, tg);                                                         \
  }

#define OMLX_M5_AFFINE_WIDE_Q56_SWIGLU(NAME, GS, WM, WN)                       \
  [[max_total_threads_per_threadgroup((WM) * (WN) * 32u)]]                     \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *g_codes [[buffer(1)]],                               \
      device const uchar *g_high [[buffer(2)]],                                \
      device const uint *g_sb [[buffer(3)]],                                   \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_high [[buffer(5)]],                                \
      device const uint *u_sb [[buffer(6)]],                                   \
      device bfloat *y [[buffer(7)]],                                          \
      constant AffineNaxParams &p [[buffer(8)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[(WN) * 32u * 72u];                       \
    affine_nax_wide_sk1_swiglu_impl<1u, GS, WM, WN>(                           \
        x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p, Ws,             \
        lane, sg, tg);                                                         \
  }

#define OMLX_M5_AFFINE_WIDE_Q3_SWIGLU(NAME, GS, WM, WN)                        \
  [[max_total_threads_per_threadgroup((WM) * (WN) * 32u)]]                     \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *g_codes [[buffer(1)]],                               \
      device const uchar *g_high [[buffer(2)]],                                \
      device const uint *g_sb [[buffer(3)]],                                   \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_high [[buffer(5)]],                                \
      device const uint *u_sb [[buffer(6)]],                                   \
      device bfloat *y [[buffer(7)]],                                          \
      constant AffineNaxParams &p [[buffer(8)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[(WN) * 32u * 72u];                       \
    affine_nax_wide_sk1_swiglu_impl<2u, GS, WM, WN>(                           \
        x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p, Ws,             \
        lane, sg, tg);                                                         \
  }

#define OMLX_M5_AFFINE_WIDE_INSTANTIATE_GS(GS)                                 \
  OMLX_M5_AFFINE_WIDE_NO_HIGH_LINEAR(omlx_m5_affine_nax_q4_g##GS##_bm64_bn64_bns16_sk1, 0u, GS##u, 16u, 2u, 2u) \
  OMLX_M5_AFFINE_WIDE_NO_HIGH_LINEAR(omlx_m5_affine_nax_q4_g##GS##_bm64_bn64_bns32_sk1, 0u, GS##u, 32u, 2u, 2u) \
  OMLX_M5_AFFINE_WIDE_NO_HIGH_LINEAR(omlx_m5_affine_nax_q4_g##GS##_bm32_bn32_bns16_sk1, 0u, GS##u, 16u, 1u, 1u) \
  OMLX_M5_AFFINE_WIDE_NO_HIGH_LINEAR(omlx_m5_affine_nax_q4_g##GS##_bm32_bn32_bns32_sk1, 0u, GS##u, 32u, 1u, 1u) \
  OMLX_M5_AFFINE_WIDE_Q56_LINEAR(omlx_m5_affine_nax_q56_g##GS##_bm64_bn64_bns16_sk1, GS##u, 16u, 2u, 2u) \
  OMLX_M5_AFFINE_WIDE_Q56_LINEAR(omlx_m5_affine_nax_q56_g##GS##_bm64_bn64_bns32_sk1, GS##u, 32u, 2u, 2u) \
  OMLX_M5_AFFINE_WIDE_Q56_LINEAR(omlx_m5_affine_nax_q56_g##GS##_bm32_bn32_bns16_sk1, GS##u, 16u, 1u, 1u) \
  OMLX_M5_AFFINE_WIDE_Q56_LINEAR(omlx_m5_affine_nax_q56_g##GS##_bm32_bn32_bns32_sk1, GS##u, 32u, 1u, 1u) \
  OMLX_M5_AFFINE_WIDE_Q3_LINEAR(omlx_m5_affine_nax_q3_g##GS##_bm64_bn64_bns16_sk1, GS##u, 16u, 2u, 2u) \
  OMLX_M5_AFFINE_WIDE_Q3_LINEAR(omlx_m5_affine_nax_q3_g##GS##_bm64_bn64_bns32_sk1, GS##u, 32u, 2u, 2u) \
  OMLX_M5_AFFINE_WIDE_Q3_LINEAR(omlx_m5_affine_nax_q3_g##GS##_bm32_bn32_bns16_sk1, GS##u, 16u, 1u, 1u) \
  OMLX_M5_AFFINE_WIDE_Q3_LINEAR(omlx_m5_affine_nax_q3_g##GS##_bm32_bn32_bns32_sk1, GS##u, 32u, 1u, 1u) \
  OMLX_M5_AFFINE_WIDE_NO_HIGH_LINEAR(omlx_m5_affine_nax_q8_g##GS##_bm64_bn64_bns16_sk1, 3u, GS##u, 16u, 2u, 2u) \
  OMLX_M5_AFFINE_WIDE_NO_HIGH_LINEAR(omlx_m5_affine_nax_q8_g##GS##_bm64_bn64_bns32_sk1, 3u, GS##u, 32u, 2u, 2u) \
  OMLX_M5_AFFINE_WIDE_NO_HIGH_LINEAR(omlx_m5_affine_nax_q8_g##GS##_bm32_bn32_bns16_sk1, 3u, GS##u, 16u, 1u, 1u) \
  OMLX_M5_AFFINE_WIDE_NO_HIGH_LINEAR(omlx_m5_affine_nax_q8_g##GS##_bm32_bn32_bns32_sk1, 3u, GS##u, 32u, 1u, 1u) \
  OMLX_M5_AFFINE_WIDE_NO_HIGH_SWIGLU(omlx_m5_affine_nax_q4_g##GS##_swiglu_bm64_bn64_sk1, 0u, GS##u, 2u, 2u) \
  OMLX_M5_AFFINE_WIDE_NO_HIGH_SWIGLU(omlx_m5_affine_nax_q4_g##GS##_swiglu_bm32_bn32_sk1, 0u, GS##u, 1u, 1u) \
  OMLX_M5_AFFINE_WIDE_Q56_SWIGLU(omlx_m5_affine_nax_q56_g##GS##_swiglu_bm64_bn64_sk1, GS##u, 2u, 2u) \
  OMLX_M5_AFFINE_WIDE_Q56_SWIGLU(omlx_m5_affine_nax_q56_g##GS##_swiglu_bm32_bn32_sk1, GS##u, 1u, 1u) \
  OMLX_M5_AFFINE_WIDE_Q3_SWIGLU(omlx_m5_affine_nax_q3_g##GS##_swiglu_bm64_bn64_sk1, GS##u, 2u, 2u) \
  OMLX_M5_AFFINE_WIDE_Q3_SWIGLU(omlx_m5_affine_nax_q3_g##GS##_swiglu_bm32_bn32_sk1, GS##u, 1u, 1u) \
  OMLX_M5_AFFINE_WIDE_NO_HIGH_SWIGLU(omlx_m5_affine_nax_q8_g##GS##_swiglu_bm64_bn64_sk1, 3u, GS##u, 2u, 2u) \
  OMLX_M5_AFFINE_WIDE_NO_HIGH_SWIGLU(omlx_m5_affine_nax_q8_g##GS##_swiglu_bm32_bn32_sk1, 3u, GS##u, 1u, 1u)

OMLX_M5_AFFINE_WIDE_INSTANTIATE_GS(64)
OMLX_M5_AFFINE_WIDE_INSTANTIATE_GS(128)

// ============================================================================
// Lossless Base 1.2: Large-Tile (BM=64, BN_out=64, BK=64) Cooperative As-Staged
// Gate + Up + SwiGLU Fusion (8 SGs = 256 threads, WM=2, WN=4, 27.0 KiB SRAM)
// + Dual-ANE Output-Channel-Split (`row_offset` % 64 == 0, `out_stride`)
// ============================================================================

template <uint ModeG, uint ModeU>
inline void affine_nax_wide_sk1_swiglu_out64_apple9(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const uint *g_sb,
    device const uchar *u_codes,
    device const uchar *u_high,
    device const uint *u_sb,
    device bfloat *y,
    constant AffineNaxSliceParams &p,
    threadgroup bfloat *As,
    threadgroup bfloat *Ws,
    uint tid,
    uint sg_m,
    uint sg_n,
    uint rb_tg,
    uint n0_tg) {
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint SM = 32u;
  constexpr uint OUT_SN = 16u;

  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * OUT_SN;
  const bool sg_active = (sg_m < 2u) && (sg_n < 4u) && (rb_sg < p.rows) && (n0_sg < p.n);
  const uint groups = p.k / kBK;

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_c0 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
  auto ct_c1 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 16u; ++i) {
    ct_c0[i] = 0.0f;
    ct_c1[i] = 0.0f;
  }

  const uint w_row = tid >> 1u;
  const uint half_k = tid & 1u;
  const bool is_up = ((w_row >> 4u) & 1u) != 0u;
  const uint local_col = n0_tg + ((w_row >> 5u) * 16u) + (w_row & 15u);
  const uint global_col = p.row_offset + local_col;
  const uint col_limit = p.row_offset + p.n;
  threadgroup vec<bfloat, 8> *w_dst8 =
      reinterpret_cast<threadgroup vec<bfloat, 8> *>(Ws + w_row * kBKP + half_k * 32u);

  const uint a_col4 = (tid & 15u) * 4u;
  const uint a_r0 = tid >> 4u;

  for (uint g = 0u; g < groups; ++g) {
    const uint k_off = g * kBK;
    bfloat4 xa[4];
    #pragma clang loop unroll(full)
    for (uint s = 0u; s < 4u; ++s) {
      const uint r = a_r0 + s * 16u;
      const uint row = rb_tg + r;
      xa[s] = (row < p.rows)
          ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p.k + k_off + a_col4)
          : bfloat4(bfloat(0.0f));
    }
    AffineRaw32<ModeG> raw_g;
    AffineRaw32<ModeU> raw_u;
    if (!is_up) {
      raw_g = affine_fetch_raw_32_checked<ModeG, 16u>(
          g_codes, g_high, g_sb, global_col, col_limit, p.padded_n, groups, g, half_k);
    } else {
      raw_u = affine_fetch_raw_32_checked<ModeU, 16u>(
          u_codes, u_high, u_sb, global_col, col_limit, p.padded_n, groups, g, half_k);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint s = 0u; s < 4u; ++s) {
      const uint r = a_r0 + s * 16u;
      *reinterpret_cast<threadgroup bfloat4 *>(As + r * kBKP + a_col4) = xa[s];
    }
    if (!is_up) {
      affine_unpack_raw_32<ModeG>(raw_g, w_dst8);
    } else {
      affine_unpack_raw_32<ModeU>(raw_u, w_dst8);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_active) {
      #pragma clang loop unroll(full)
      for (uint kk = 0u; kk < kBK; kk += 16u) {
        const threadgroup bfloat *as_sub = As + (sg_m * SM) * kBKP + kk;
        const threadgroup bfloat *ws_sub = Ws + (sg_n * 32u) * kBKP + kk;
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          const auto idx_b = ct_b.get_multidimensional_index(i);
          ct_b[i] = ws_sub[uint(idx_b[1]) * kBKP + uint(idx_b[0])];
        }
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 8u; ++i) {
          const auto idx_a = ct_a.get_multidimensional_index(i);
          ct_a[i] = as_sub[uint(idx_a[1]) * kBKP + uint(idx_a[0])];
        }
        gemm_op.run(ct_a, ct_b, ct_c0);
        if (rb_sg + 16u < p.rows) {
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 8u; ++i) {
            const auto idx_a = ct_a.get_multidimensional_index(i);
            ct_a[i] = as_sub[(16u + uint(idx_a[1])) * kBKP + uint(idx_a[0])];
          }
          gemm_op.run(ct_a, ct_b, ct_c1);
        }
      }
    }
  }

  if (sg_active) {
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm) {
      if (rb_sg + mm * 16u < p.rows) {
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          const auto idx_g = ct_c0.get_multidimensional_index(i);
          const uint c_g = uint(idx_g[0]);
          const uint r_g = uint(idx_g[1]);
          if (c_g < 16u) {
            const uint lcol = n0_sg + c_g;
            const uint row = rb_sg + mm * 16u + r_g;
            if (row < p.rows && lcol < p.n) {
              float gv = (mm == 0u) ? ct_c0[i] : ct_c1[i];
              float uv = 0.0f;
              #pragma clang loop unroll(full)
              for (ushort j = 0u; j < 16u; ++j) {
                const auto idx_u = ct_c0.get_multidimensional_index(j);
                if (uint(idx_u[0]) == c_g + 16u && uint(idx_u[1]) == r_g) {
                  uv = (mm == 0u) ? ct_c0[j] : ct_c1[j];
                }
              }
              gv = float(bfloat(gv));
              uv = float(bfloat(uv));
              y[ulong(row) * ulong(out_stride) + ulong(p.row_offset + lcol)] =
                  bfloat(nax_silu_f32(gv) * uv);
            }
          }
        }
      }
    }
  }
}

template <uint ModeG, uint ModeU, uint GroupSize>
inline void affine_nax_wide_sk1_swiglu_out64_impl(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const uint *g_sb,
    device const uchar *u_codes,
    device const uchar *u_high,
    device const uint *u_sb,
    device bfloat *y,
    constant AffineNaxSliceParams &p,
    threadgroup bfloat *As,
    threadgroup bfloat *Ws,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 grid_tgs) {
  (void)GroupSize;
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint SM = 32u;
  constexpr uint OUT_SN = 16u;
  constexpr uint BM_TG = 64u;
  constexpr uint OUT_N_TG = 64u;

  const uint2 pid_nm = affine_swizzle_tg_group_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg = pid_nm.x * OUT_N_TG;
  if (rb_tg >= p.rows || n0_tg >= p.n) {
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint sg_m = sg >> 2u;
  const uint sg_n = sg & 3u;
  if (!affine_has_apple10_nax_layout()) {
    affine_nax_wide_sk1_swiglu_out64_apple9<ModeG, ModeU>(
        x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p, As, Ws, tid,
        sg_m, sg_n, rb_tg, n0_tg);
    return;
  }

  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * OUT_SN;
  const uint groups = p.k / kBK;
  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

  vec<float, 8> D[2][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < 2u; ++mm) {
    D[mm][0] = vec<float, 8>(0.0f);
    D[mm][1] = vec<float, 8>(0.0f);
  }

  constexpr ulong kCodeBytesG = (ModeG == 2u) ? 16ul : (ModeG == 3u) ? 64ul : 32ul;
  constexpr ulong kHighBytesG = (ModeG == 1u) ? 16ul : (ModeG == 2u) ? 8ul : 0ul;
  constexpr ulong kCodeBytesU = (ModeU == 2u) ? 16ul : (ModeU == 3u) ? 64ul : 32ul;
  constexpr ulong kHighBytesU = (ModeU == 1u) ? 16ul : (ModeU == 2u) ? 8ul : 0ul;

  const uint w_row = tid >> 1u;
  const uint half_k = tid & 1u;
  const bool is_up = ((w_row >> 4u) & 1u) != 0u;
  const uint local_col = n0_tg + ((w_row >> 5u) * 16u) + (w_row & 15u);
  const uint global_col = p.row_offset + local_col;
  const uint col_limit = p.row_offset + p.n;
  threadgroup vec<bfloat, 8> *w_dst8 =
      reinterpret_cast<threadgroup vec<bfloat, 8> *>(Ws + w_row * kBKP + half_k * 32u);

  const uint a_col4 = (tid & 15u) * 4u;
  const uint a_r0 = tid >> 4u;

  const threadgroup bfloat *as_sg = As + (sg_m * SM + fm) * kBKP + fn;
  const threadgroup bfloat *ws_sg = Ws + (sg_n * 32u + fm) * kBKP + fn;

  const bool full_tile = (rb_tg + BM_TG <= p.rows) && (n0_tg + OUT_N_TG <= p.n);
  if (full_tile) {
    const uint tile_n = global_col >> 4u;
    const uint rn = global_col & 15u;
    const ulong tile_base = ulong(tile_n) * ulong(groups);
    device const uchar *wc_g = g_codes + tile_base * (16ul * kCodeBytesG) + ulong(rn) * kCodeBytesG;
    device const uchar *wh_g = (kHighBytesG > 0ul)
        ? (g_high + tile_base * (16ul * kHighBytesG) + ulong(rn) * kHighBytesG)
        : nullptr;
    device const uchar *wc_u = u_codes + tile_base * (16ul * kCodeBytesU) + ulong(rn) * kCodeBytesU;
    device const uchar *wh_u = (kHighBytesU > 0ul)
        ? (u_high + tile_base * (16ul * kHighBytesU) + ulong(rn) * kHighBytesU)
        : nullptr;
    device const uint *sb_g = g_sb + global_col;
    device const uint *sb_u = u_sb + global_col;

    for (uint g = 0u; g < groups; ++g) {
      const uint k0 = g * kBK;
      bfloat4 xa[4];
      #pragma clang loop unroll(full)
      for (uint s = 0u; s < 4u; ++s) {
        const uint row = rb_tg + a_r0 + s * 16u;
        xa[s] = *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p.k + k0 + a_col4);
      }
      AffineRaw32<ModeG> raw_g;
      AffineRaw32<ModeU> raw_u;
      if (!is_up) {
        raw_g = affine_fetch_raw_32_fast<ModeG, 16u>(wc_g, wh_g, sb_g, p.padded_n, g, half_k);
      } else {
        raw_u = affine_fetch_raw_32_fast<ModeU, 16u>(wc_u, wh_u, sb_u, p.padded_n, g, half_k);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint s = 0u; s < 4u; ++s) {
        *reinterpret_cast<threadgroup bfloat4 *>(As + (a_r0 + s * 16u) * kBKP + a_col4) = xa[s];
      }
      if (!is_up) {
        affine_unpack_raw_32<ModeG>(raw_g, w_dst8);
      } else {
        affine_unpack_raw_32<ModeU>(raw_u, w_dst8);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      #pragma clang loop unroll(full)
      for (uint kk = 0u; kk < kBK; kk += 32u) {
        volatile int cbar;
        vec<bfloat, 8> a00, a01, a10, a11, b00, b01, b10, b11;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 2u; ++r) {
          const bfloat4 va0 = *reinterpret_cast<const threadgroup bfloat4 *>(as_sg + r * 8u * kBKP + kk + 0u);
          const bfloat4 va1 = *reinterpret_cast<const threadgroup bfloat4 *>(as_sg + r * 8u * kBKP + kk + 16u);
          const bfloat4 va2 = *reinterpret_cast<const threadgroup bfloat4 *>(as_sg + (16u + r * 8u) * kBKP + kk + 0u);
          const bfloat4 va3 = *reinterpret_cast<const threadgroup bfloat4 *>(as_sg + (16u + r * 8u) * kBKP + kk + 16u);
          const bfloat4 vb0 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 0u);
          const bfloat4 vb1 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 16u);
          const bfloat4 vb2 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 0u);
          const bfloat4 vb3 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 16u);
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            a00[r * 4u + c] = va0[c]; a01[r * 4u + c] = va1[c];
            a10[r * 4u + c] = va2[c]; a11[r * 4u + c] = va3[c];
            b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
            b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
          }
        }
        affine_mma_16x32x16_bf16(D[0][0], D[0][1], a00, b00, b10);
        affine_mma_16x32x16_bf16(D[0][0], D[0][1], a01, b01, b11);
        affine_mma_16x32x16_bf16(D[1][0], D[1][1], a10, b00, b10);
        affine_mma_16x32x16_bf16(D[1][0], D[1][1], a11, b01, b11);
        (void)cbar;
      }
    }
  } else {
    const bool sg_active = (sg_m < 2u) && (sg_n < 4u) && (rb_sg < p.rows) && (n0_sg < p.n);
    for (uint g = 0u; g < groups; ++g) {
      const uint k0 = g * kBK;
      bfloat4 xa[4];
      #pragma clang loop unroll(full)
      for (uint s = 0u; s < 4u; ++s) {
        const uint row = rb_tg + a_r0 + s * 16u;
        xa[s] = (row < p.rows)
            ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p.k + k0 + a_col4)
            : bfloat4(bfloat(0.0f));
      }
      AffineRaw32<ModeG> raw_g;
      AffineRaw32<ModeU> raw_u;
      if (!is_up) {
        raw_g = affine_fetch_raw_32_checked<ModeG, 16u>(
            g_codes, g_high, g_sb, global_col, col_limit, p.padded_n, groups, g, half_k);
      } else {
        raw_u = affine_fetch_raw_32_checked<ModeU, 16u>(
            u_codes, u_high, u_sb, global_col, col_limit, p.padded_n, groups, g, half_k);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint s = 0u; s < 4u; ++s) {
        *reinterpret_cast<threadgroup bfloat4 *>(As + (a_r0 + s * 16u) * kBKP + a_col4) = xa[s];
      }
      if (!is_up) {
        affine_unpack_raw_32<ModeG>(raw_g, w_dst8);
      } else {
        affine_unpack_raw_32<ModeU>(raw_u, w_dst8);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_active) {
        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < kBK; kk += 32u) {
          volatile int cbar;
          vec<bfloat, 8> b00, b01, b10, b11;
          #pragma clang loop unroll(full)
          for (uint r = 0u; r < 2u; ++r) {
            const bfloat4 vb0 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 0u);
            const bfloat4 vb1 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 16u);
            const bfloat4 vb2 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 0u);
            const bfloat4 vb3 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 16u);
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) {
              b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
              b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
            }
          }
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            if (rb_sg + mm * 16u < p.rows) {
              vec<bfloat, 8> a0, a1;
              #pragma clang loop unroll(full)
              for (uint r = 0u; r < 2u; ++r) {
                const bfloat4 va0 = *reinterpret_cast<const threadgroup bfloat4 *>(
                    as_sg + (mm * 16u + r * 8u) * kBKP + kk + 0u);
                const bfloat4 va1 = *reinterpret_cast<const threadgroup bfloat4 *>(
                    as_sg + (mm * 16u + r * 8u) * kBKP + kk + 16u);
                #pragma clang loop unroll(full)
                for (uint c = 0u; c < 4u; ++c) {
                  a0[r * 4u + c] = va0[c];
                  a1[r * 4u + c] = va1[c];
                }
              }
              affine_mma_16x32x16_bf16(D[mm][0], D[mm][1], a0, b00, b10);
              affine_mma_16x32x16_bf16(D[mm][0], D[mm][1], a1, b01, b11);
            }
          }
          (void)cbar;
        }
      }
    }
  }

  const bool sg_active = (sg_m < 2u) && (sg_n < 4u) && (rb_sg < p.rows) && (n0_sg < p.n);
  if (sg_active) {
    const bool aligned_out = ((out_stride & 3u) == 0u) && ((p.row_offset & 3u) == 0u);
    const uint lcol = n0_sg + fn;
    const uint gcol = p.row_offset + lcol;
    if (lcol < p.n) {
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        if (rb_sg + mm * 16u < p.rows) {
          #pragma clang loop unroll(full)
          for (uint h = 0u; h < 2u; ++h) {
            const uint row = rb_sg + mm * 16u + fm + h * 8u;
            if (row < p.rows) {
              const uint e = h * 4u;
              bfloat4 out_b;
              #pragma clang loop unroll(full)
              for (uint j = 0u; j < 4u; ++j) {
                const float gv = float(bfloat(D[mm][0][e + j]));
                const float uv = float(bfloat(D[mm][1][e + j]));
                out_b[j] = bfloat(nax_silu_f32(gv) * uv);
              }
              if (aligned_out && lcol + 4u <= p.n) {
                *reinterpret_cast<device bfloat4 *>(y + ulong(row) * ulong(out_stride) + ulong(gcol)) = out_b;
              } else {
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  if (lcol + j < p.n) {
                    y[ulong(row) * ulong(out_stride) + ulong(gcol + j)] = out_b[j];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template <uint Mode, uint GroupSize, uint BN_STORED>
inline void affine_nax_wide_sk1_linear_slice_impl(
    device const bfloat *x,
    device const uchar *codes,
    device const uchar *high_plane,
    device const uint *sb,
    device bfloat *y,
    constant AffineNaxSliceParams &p,
    threadgroup bfloat *Ws,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 grid_tgs) {
  (void)GroupSize;
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint SM = 32u;
  constexpr uint SN = 32u;
  constexpr uint BM_TG = 64u;
  constexpr uint BN_TG = 64u;

  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  const uint2 pid_nm = affine_swizzle_tg_group_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg = pid_nm.x * BN_TG;
  if (rb_tg >= p.rows || n0_tg >= p.n) {
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * SN;
  const uint groups = p.k / kBK;
  const uint col_limit = p.row_offset + p.n;
  const bool sg_active = (sg_m < 2u) && (sg_n < 2u) && (rb_sg < p.rows) && (n0_sg < p.n);

  if (!affine_has_apple10_nax_layout()) {
    constexpr auto desc = matmul2d_descriptor(
        16, 32, 16, false, true, /*relaxed_precision=*/true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> gemm_op;
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c0 = gemm_op.template get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
    auto ct_c1 = gemm_op.template get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      ct_c0[i] = 0.0f;
      ct_c1[i] = 0.0f;
    }
    const uint w_row = tid >> 1u;
    const uint half_k = tid & 1u;
    const uint gcol_w = p.row_offset + n0_tg + w_row;
    threadgroup vec<bfloat, 8> *w_dst8 =
        reinterpret_cast<threadgroup vec<bfloat, 8> *>(Ws + w_row * kBKP + half_k * 32u);
    for (uint g = 0u; g < groups; ++g) {
      const uint k_off = g * kBK;
      const AffineRaw32<Mode> raw = affine_fetch_raw_32_checked<Mode, BN_STORED>(
          codes, high_plane, sb, gcol_w, col_limit, p.padded_n, groups, g, half_k);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      affine_unpack_raw_32<Mode>(raw, w_dst8);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg_active) {
        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < kBK; kk += 16u) {
          const threadgroup bfloat *ws_sub = Ws + (sg_n * SN) * kBKP + kk;
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            const auto idx_b = ct_b.get_multidimensional_index(i);
            ct_b[i] = ws_sub[uint(idx_b[1]) * kBKP + uint(idx_b[0])];
          }
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 8u; ++i) {
            const auto idx_a = ct_a.get_multidimensional_index(i);
            const uint r0 = rb_sg + uint(idx_a[1]);
            ct_a[i] = (r0 < p.rows) ? x[ulong(r0) * p.k + k_off + kk + uint(idx_a[0])] : bfloat(0.0f);
          }
          gemm_op.run(ct_a, ct_b, ct_c0);
          if (rb_sg + 16u < p.rows) {
            #pragma clang loop unroll(full)
            for (ushort i = 0u; i < 8u; ++i) {
              const auto idx_a = ct_a.get_multidimensional_index(i);
              const uint r1 = rb_sg + 16u + uint(idx_a[1]);
              ct_a[i] = (r1 < p.rows) ? x[ulong(r1) * p.k + k_off + kk + uint(idx_a[0])] : bfloat(0.0f);
            }
            gemm_op.run(ct_a, ct_b, ct_c1);
          }
        }
      }
    }
    if (sg_active) {
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) {
        const auto idx = ct_c0.get_multidimensional_index(i);
        const uint lcol = n0_sg + uint(idx[0]);
        const uint r0 = rb_sg + uint(idx[1]);
        if (r0 < p.rows && lcol < p.n) {
          y[ulong(r0) * ulong(out_stride) + ulong(p.row_offset + lcol)] = bfloat(ct_c0[i]);
        }
        const uint r1 = r0 + 16u;
        if (r1 < p.rows && lcol < p.n) {
          y[ulong(r1) * ulong(out_stride) + ulong(p.row_offset + lcol)] = bfloat(ct_c1[i]);
        }
      }
    }
    return;
  }

  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
  vec<float, 8> D[2][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < 2u; ++mm) {
    D[mm][0] = vec<float, 8>(0.0f);
    D[mm][1] = vec<float, 8>(0.0f);
  }
  const uint w_row = tid >> 1u;
  const uint half_k = tid & 1u;
  const uint gcol_w = p.row_offset + n0_tg + w_row;
  threadgroup vec<bfloat, 8> *w_dst8 =
      reinterpret_cast<threadgroup vec<bfloat, 8> *>(Ws + w_row * kBKP + half_k * 32u);
  const threadgroup bfloat *ws_sg = Ws + (sg_n * SN + fm) * kBKP + fn;

  for (uint g = 0u; g < groups; ++g) {
    const uint k_off = g * kBK;
    const AffineRaw32<Mode> raw = affine_fetch_raw_32_checked<Mode, BN_STORED>(
        codes, high_plane, sb, gcol_w, col_limit, p.padded_n, groups, g, half_k);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    affine_unpack_raw_32<Mode>(raw, w_dst8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_active) {
      #pragma clang loop unroll(full)
      for (uint kk = 0u; kk < kBK; kk += 32u) {
        volatile int cbar;
        vec<bfloat, 8> b00, b01, b10, b11;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 2u; ++r) {
          const bfloat4 vb0 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 0u);
          const bfloat4 vb1 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 16u);
          const bfloat4 vb2 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 0u);
          const bfloat4 vb3 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 16u);
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
            b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
          }
        }
        #pragma clang loop unroll(full)
        for (uint mm = 0u; mm < 2u; ++mm) {
          if (rb_sg + mm * 16u < p.rows) {
            vec<bfloat, 8> a0, a1;
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 2u; ++r) {
              const uint row = rb_sg + mm * 16u + fm + r * 8u;
              const bfloat4 va0 = (row < p.rows)
                  ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p.k + k_off + kk + fn)
                  : bfloat4(bfloat(0.0f));
              const bfloat4 va1 = (row < p.rows)
                  ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p.k + k_off + kk + 16u + fn)
                  : bfloat4(bfloat(0.0f));
              #pragma clang loop unroll(full)
              for (uint c = 0u; c < 4u; ++c) {
                a0[r * 4u + c] = va0[c];
                a1[r * 4u + c] = va1[c];
              }
            }
            affine_mma_16x32x16_bf16(D[mm][0], D[mm][1], a0, b00, b10);
            affine_mma_16x32x16_bf16(D[mm][0], D[mm][1], a1, b01, b11);
          }
        }
        (void)cbar;
      }
    }
  }

  if (sg_active) {
    const bool aligned_out = ((out_stride & 3u) == 0u) && ((p.row_offset & 3u) == 0u);
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm) {
      if (rb_sg + mm * 16u < p.rows) {
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint row = rb_sg + mm * 16u + fm + h * 8u;
          if (row < p.rows) {
            const uint e = h * 4u;
            #pragma clang loop unroll(full)
            for (uint nn = 0u; nn < 2u; ++nn) {
              const uint lcol = n0_sg + nn * 16u + fn;
              const uint gcol = p.row_offset + lcol;
              if (lcol < p.n) {
                const bfloat4 out_v = bfloat4(
                    bfloat(D[mm][nn][e + 0u]), bfloat(D[mm][nn][e + 1u]),
                    bfloat(D[mm][nn][e + 2u]), bfloat(D[mm][nn][e + 3u]));
                if (aligned_out && lcol + 4u <= p.n) {
                  *reinterpret_cast<device bfloat4 *>(y + ulong(row) * ulong(out_stride) + ulong(gcol)) = out_v;
                } else {
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    if (lcol + j < p.n) {
                      y[ulong(row) * ulong(out_stride) + ulong(gcol + j)] = out_v[j];
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

#define OMLX_M5_AFFINE_OUT64_NO_HIGH_SWIGLU(NAME, MODE, GS)                    \
  [[max_total_threads_per_threadgroup(256)]]                                   \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *g_codes [[buffer(1)]],                               \
      device const uint *g_sb [[buffer(2)]],                                   \
      device const uchar *u_codes [[buffer(3)]],                               \
      device const uint *u_sb [[buffer(4)]],                                   \
      device bfloat *y [[buffer(5)]],                                          \
      constant AffineNaxSliceParams &p [[buffer(6)]],                          \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 grid_tgs [[threadgroups_per_grid]]) {                              \
    alignas(16) threadgroup bfloat As[64u * 72u];                              \
    alignas(16) threadgroup bfloat Ws[128u * 72u];                             \
    affine_nax_wide_sk1_swiglu_out64_impl<MODE, MODE, GS>(                     \
        x, g_codes, nullptr, g_sb, u_codes, nullptr, u_sb, y, p, As, Ws,       \
        lane, sg, tg, grid_tgs);                                               \
  }

#define OMLX_M5_AFFINE_OUT64_WITH_HIGH_SWIGLU(NAME, MODEG, MODEU, GS)          \
  [[max_total_threads_per_threadgroup(256)]]                                   \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *g_codes [[buffer(1)]],                               \
      device const uchar *g_high [[buffer(2)]],                                \
      device const uint *g_sb [[buffer(3)]],                                   \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_high [[buffer(5)]],                                \
      device const uint *u_sb [[buffer(6)]],                                   \
      device bfloat *y [[buffer(7)]],                                          \
      constant AffineNaxSliceParams &p [[buffer(8)]],                          \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 grid_tgs [[threadgroups_per_grid]]) {                              \
    alignas(16) threadgroup bfloat As[64u * 72u];                              \
    alignas(16) threadgroup bfloat Ws[128u * 72u];                             \
    affine_nax_wide_sk1_swiglu_out64_impl<MODEG, MODEU, GS>(                   \
        x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p, As, Ws,         \
        lane, sg, tg, grid_tgs);                                               \
  }

#define OMLX_M5_AFFINE_SLICE_NO_HIGH_LINEAR(NAME, MODE, GS, BNS)               \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *codes [[buffer(1)]],                                 \
      device const uint *sb [[buffer(2)]],                                     \
      device bfloat *y [[buffer(3)]],                                          \
      constant AffineNaxSliceParams &p [[buffer(4)]],                          \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 grid_tgs [[threadgroups_per_grid]]) {                              \
    alignas(16) threadgroup bfloat Ws[64u * 72u];                              \
    affine_nax_wide_sk1_linear_slice_impl<MODE, GS, BNS>(                      \
        x, codes, nullptr, sb, y, p, Ws, lane, sg, tg, grid_tgs);              \
  }

#define OMLX_M5_AFFINE_SLICE_WITH_HIGH_LINEAR(NAME, MODE, GS, BNS)             \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *codes [[buffer(1)]],                                 \
      device const uchar *high_p [[buffer(2)]],                                \
      device const uint *sb [[buffer(3)]],                                     \
      device bfloat *y [[buffer(4)]],                                          \
      constant AffineNaxSliceParams &p [[buffer(5)]],                          \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 grid_tgs [[threadgroups_per_grid]]) {                              \
    alignas(16) threadgroup bfloat Ws[64u * 72u];                              \
    affine_nax_wide_sk1_linear_slice_impl<MODE, GS, BNS>(                      \
        x, codes, high_p, sb, y, p, Ws, lane, sg, tg, grid_tgs);               \
  }

#define OMLX_M5_AFFINE_OUT64_INSTANTIATE_GS(GS)                                \
  OMLX_M5_AFFINE_OUT64_NO_HIGH_SWIGLU(omlx_m5_affine_nax_q4_g##GS##_swiglu_bm64_bn64_out64_sk1, 0u, GS##u) \
  OMLX_M5_AFFINE_OUT64_WITH_HIGH_SWIGLU(omlx_m5_affine_nax_q56_g##GS##_swiglu_bm64_bn64_out64_sk1, 1u, 1u, GS##u) \
  OMLX_M5_AFFINE_OUT64_WITH_HIGH_SWIGLU(omlx_m5_affine_nax_q3_g##GS##_swiglu_bm64_bn64_out64_sk1, 2u, 2u, GS##u) \
  OMLX_M5_AFFINE_OUT64_NO_HIGH_SWIGLU(omlx_m5_affine_nax_q8_g##GS##_swiglu_bm64_bn64_out64_sk1, 3u, GS##u) \
  OMLX_M5_AFFINE_OUT64_WITH_HIGH_SWIGLU(omlx_m5_affine_nax_q4_q56_g##GS##_swiglu_bm64_bn64_out64_sk1, 0u, 1u, GS##u) \
  OMLX_M5_AFFINE_OUT64_WITH_HIGH_SWIGLU(omlx_m5_affine_nax_q56_q4_g##GS##_swiglu_bm64_bn64_out64_sk1, 1u, 0u, GS##u) \
  OMLX_M5_AFFINE_SLICE_NO_HIGH_LINEAR(omlx_m5_affine_nax_q4_g##GS##_bm64_bn64_bns16_slice_sk1, 0u, GS##u, 16u) \
  OMLX_M5_AFFINE_SLICE_NO_HIGH_LINEAR(omlx_m5_affine_nax_q4_g##GS##_bm64_bn64_bns32_slice_sk1, 0u, GS##u, 32u) \
  OMLX_M5_AFFINE_SLICE_WITH_HIGH_LINEAR(omlx_m5_affine_nax_q56_g##GS##_bm64_bn64_bns16_slice_sk1, 1u, GS##u, 16u) \
  OMLX_M5_AFFINE_SLICE_WITH_HIGH_LINEAR(omlx_m5_affine_nax_q56_g##GS##_bm64_bn64_bns32_slice_sk1, 1u, GS##u, 32u) \
  OMLX_M5_AFFINE_SLICE_WITH_HIGH_LINEAR(omlx_m5_affine_nax_q3_g##GS##_bm64_bn64_bns16_slice_sk1, 2u, GS##u, 16u) \
  OMLX_M5_AFFINE_SLICE_WITH_HIGH_LINEAR(omlx_m5_affine_nax_q3_g##GS##_bm64_bn64_bns32_slice_sk1, 2u, GS##u, 32u) \
  OMLX_M5_AFFINE_SLICE_NO_HIGH_LINEAR(omlx_m5_affine_nax_q8_g##GS##_bm64_bn64_bns16_slice_sk1, 3u, GS##u, 16u) \
  OMLX_M5_AFFINE_SLICE_NO_HIGH_LINEAR(omlx_m5_affine_nax_q8_g##GS##_bm64_bn64_bns32_slice_sk1, 3u, GS##u, 32u)

OMLX_M5_AFFINE_OUT64_INSTANTIATE_GS(64)
OMLX_M5_AFFINE_OUT64_INSTANTIATE_GS(128)

// ============================================================================
// v0.8.0-m5rt.13 Narrow/Verify Q5-Only BM=16, BN=32, BK=64 Zero-As bfloat
// 16x32x16 NAX + 1-bit q5_high1 + sb_tiled + Zero-uitofp XOR-Ws:
// - Proven on live M5 Ultra cold-DRAM micro-probe: 840.3 GB/s (+24.7%) on
//   SwiGLU [8, 5120, 17408] and 667.4 GB/s (+15.9%) on Down [8, 17408, 5120].
// - Reads 1-bit q5_high1 (-50% high-plane DRAM) + contiguous sb_tiled with
//   0 xsum dispatches and strict 0-ULP row-independence across T = 1..48.
// ============================================================================

inline void affine_unpack_q5_h1_32_xor(
    uint4 raw_c,
    uint raw_h1,
    uint sb_u,
    uint w_row,
    uint half_k,
    threadgroup bfloat4 *ws_sg4) {
  const float sc = as_type<float>((sb_u & 0xFFFFu) << 16u);
  const float bi_adj = as_type<float>(sb_u & 0xFFFF0000u) - 1024.0f * sc;
  const uint xor_mask = (w_row & 3u) << 2u;
  const uint base_c4 = half_k * 8u;
  #pragma clang loop unroll(full)
  for (uint w = 0u; w < 4u; ++w) {
    const uint v = raw_c[w];
    const uint hb = (raw_h1 >> (w * 8u)) & 0xFFu;
    bfloat4 lo4, hi4;
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
      const uint q0 = ((v >> (j * 4u)) & 0x0fu) | (((hb >> j) & 1u) << 4u);
      const uint q1 = ((v >> (16u + j * 4u)) & 0x0fu) | (((hb >> (4u + j)) & 1u) << 4u);
      lo4[j] = bfloat(fma(sc, float(as_type<half>(ushort(0x6400u | q0))), bi_adj));
      hi4[j] = bfloat(fma(sc, float(as_type<half>(ushort(0x6400u | q1))), bi_adj));
    }
    ws_sg4[w_row * 16u + ((base_c4 + w * 2u + 0u) ^ xor_mask)] = lo4;
    ws_sg4[w_row * 16u + ((base_c4 + w * 2u + 1u) ^ xor_mask)] = hi4;
  }
}

inline bfloat affine_ws_read_xor(const threadgroup bfloat4 *ws_sg4, uint w_row, uint k_col) {
  const uint c4 = (k_col >> 2u) ^ ((w_row & 3u) << 2u);
  return ws_sg4[w_row * 16u + c4][k_col & 3u];
}

template <uint BN_STORED, uint SK>
inline void affine_nax_q5_h1_bm16_linear_apple9(
    device const bfloat *x,
    device const uchar *codes,
    device const uchar *high1,
    device const uint *sb_tiled,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup bfloat4 *ws_all4,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint kBK = 64u;
  const uint rb = tg.y * 16u;
  const uint n0_sg = tg.x * 32u;
  if (rb >= p.rows || n0_sg >= p.n) return;

  const uint groups = p.k / kBK;
  const uint g0 = (sg * groups) / SK;
  const uint g1 = ((sg + 1u) * groups) / SK;
  threadgroup bfloat4 *ws_sg4 = ws_all4 + sg * 512u;

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 16u; ++i) ct_c[i] = 0.0f;

  for (uint g = g0; g < g1; ++g) {
    const uint k0 = g * kBK;
    uint4 raw_c[2];
    uint raw_h[2], sb_u[2];
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      const uint idx32 = lane + f * 32u;
      const uint wr = idx32 >> 1u;
      const uint hk = idx32 & 1u;
      const uint col = n0_sg + wr;
      if (col < p.n) {
        const uint tile_n = col / BN_STORED;
        const uint rn = col % BN_STORED;
        const ulong idx = ulong(tile_n) * ulong(groups) + ulong(g);
        raw_c[f] = reinterpret_cast<device const uint4 *>(
            codes + idx * (ulong(BN_STORED) * 32ul) + ulong(rn) * 32ul)[hk];
        raw_h[f] = reinterpret_cast<device const uint *>(
            high1 + idx * (ulong(BN_STORED) * 8ul) + ulong(rn) * 8ul)[hk];
        sb_u[f] = sb_tiled[idx * ulong(BN_STORED) + rn];
      } else {
        raw_c[f] = uint4(0u); raw_h[f] = 0u; sb_u[f] = 0u;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      const uint idx32 = lane + f * 32u;
      affine_unpack_q5_h1_32_xor(raw_c[f], raw_h[f], sb_u[f], idx32 >> 1u, idx32 & 1u, ws_sg4);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (uint kk = 0u; kk < kBK; kk += 16u) {
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) {
        const auto ib = ct_b.get_multidimensional_index(i);
        ct_b[i] = affine_ws_read_xor(ws_sg4, uint(ib[1]), kk + uint(ib[0]));
      }
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 8u; ++i) {
        const auto ia = ct_a.get_multidimensional_index(i);
        const uint r0 = rb + uint(ia[1]);
        ct_a[i] = (r0 < p.rows) ? x[ulong(r0) * p.k + k0 + kk + uint(ia[0])] : bfloat(0.0f);
      }
      gemm_op.run(ct_a, ct_b, ct_c);
    }
  }

  if constexpr (SK > 1u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *parts = reinterpret_cast<threadgroup float *>(ws_all4);
    if (sg > 0u) {
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) parts[((sg - 1u) * 16u + i) * 32u + lane] = ct_c[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      for (uint s = 1u; s < SK; ++s) {
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) ct_c[i] += parts[((s - 1u) * 16u + i) * 32u + lane];
      }
    }
  }
  if (sg == 0u) {
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      const auto ic = ct_c.get_multidimensional_index(i);
      const uint col = n0_sg + uint(ic[0]);
      const uint row = rb + uint(ic[1]);
      if (row < p.rows && col < p.n) y[ulong(row) * p.n + col] = bfloat(ct_c[i]);
    }
  }
}

template <uint BN_STORED, uint SK>
inline void affine_nax_q5_h1_bm16_linear_impl(
    device const bfloat *x,
    device const uchar *codes,
    device const uchar *high1,
    device const uint *sb_tiled,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup bfloat4 *ws_all4,
    uint lane,
    uint sg,
    uint3 tg) {
  if (!affine_has_apple10_nax_layout()) {
    affine_nax_q5_h1_bm16_linear_apple9<BN_STORED, SK>(
        x, codes, high1, sb_tiled, y, p, ws_all4, lane, sg, tg);
    return;
  }
  constexpr uint kBK = 64u;
  const uint rb = tg.y * 16u;
  const uint n0_sg = tg.x * 32u;
  if (rb >= p.rows || n0_sg >= p.n) return;

  const uint groups = p.k / kBK;
  const uint g0 = (sg * groups) / SK;
  const uint g1 = ((sg + 1u) * groups) / SK;

  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
  const uint c4_lo = fn >> 2u;

  vec<float, 8> D0(0.0f), D1(0.0f);
  threadgroup bfloat4 *ws_sg4 = ws_all4 + sg * 512u;

  device const uchar *wc_ptr[2];
  device const uchar *wh_ptr[2];
  device const uint *wsb_ptr[2];
  uint w_row_f[2], half_k_f[2];
  #pragma clang loop unroll(full)
  for (uint f = 0u; f < 2u; ++f) {
    const uint idx32 = lane + f * 32u;
    const uint wr = idx32 >> 1u;
    const uint hk = idx32 & 1u;
    const uint col = n0_sg + wr;
    const uint tile_n = col / BN_STORED;
    const uint rn = col % BN_STORED;
    const ulong tbase = ulong(tile_n) * ulong(groups);
    wc_ptr[f] = (col < p.n) ? (codes + tbase * (ulong(BN_STORED) * 32ul) + ulong(rn) * 32ul) : nullptr;
    wh_ptr[f] = (col < p.n) ? (high1 + tbase * (ulong(BN_STORED) * 8ul) + ulong(rn) * 8ul) : nullptr;
    wsb_ptr[f] = (col < p.n) ? (sb_tiled + tbase * ulong(BN_STORED) + rn) : nullptr;
    w_row_f[f] = wr;
    half_k_f[f] = hk;
  }

  const uint row0 = rb + fm;
  const uint row1 = rb + fm + 8u;
  const bool has_r0 = (row0 < p.rows);
  const bool has_r1 = (row1 < p.rows);
  device const bfloat *x_r0 = x + ulong(row0) * p.k + fn;
  device const bfloat *x_r1 = x + ulong(row1) * p.k + fn;

  for (uint g = g0; g < g1; ++g) {
    const uint k0 = g * kBK;
    uint4 raw_c[2];
    uint raw_h[2], sb_u[2];
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      if (wc_ptr[f] != nullptr) {
        raw_c[f] = reinterpret_cast<device const uint4 *>(wc_ptr[f] + ulong(g) * (ulong(BN_STORED) * 32ul))[half_k_f[f]];
        raw_h[f] = reinterpret_cast<device const uint *>(wh_ptr[f] + ulong(g) * (ulong(BN_STORED) * 8ul))[half_k_f[f]];
        sb_u[f] = wsb_ptr[f][ulong(g) * ulong(BN_STORED)];
      } else {
        raw_c[f] = uint4(0u); raw_h[f] = 0u; sb_u[f] = 0u;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      affine_unpack_q5_h1_32_xor(raw_c[f], raw_h[f], sb_u[f], w_row_f[f], half_k_f[f], ws_sg4);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint xor_m = (fm & 3u) << 2u;
    #pragma clang loop unroll(full)
    for (uint kk4 = 0u; kk4 < 16u; kk4 += 8u) {
      const uint k_cur = k0 + kk4 * 4u;
      vec<bfloat, 8> a0, a1, b00, b01, b10, b11;
      const bfloat4 va00 = has_r0 ? *reinterpret_cast<device const bfloat4 *>(x_r0 + k_cur + 0u) : bfloat4(0.0bf);
      const bfloat4 va01 = has_r0 ? *reinterpret_cast<device const bfloat4 *>(x_r0 + k_cur + 16u) : bfloat4(0.0bf);
      const bfloat4 va10 = has_r1 ? *reinterpret_cast<device const bfloat4 *>(x_r1 + k_cur + 0u) : bfloat4(0.0bf);
      const bfloat4 va11 = has_r1 ? *reinterpret_cast<device const bfloat4 *>(x_r1 + k_cur + 16u) : bfloat4(0.0bf);
      #pragma clang loop unroll(full)
      for (uint c = 0u; c < 4u; ++c) {
        a0[c] = va00[c]; a0[4u + c] = va10[c];
        a1[c] = va01[c]; a1[4u + c] = va11[c];
      }
      #pragma clang loop unroll(full)
      for (uint r = 0u; r < 2u; ++r) {
        const uint r0 = fm + r * 8u;
        const uint r1 = 16u + fm + r * 8u;
        const bfloat4 vb0 = ws_sg4[r0 * 16u + ((kk4 + 0u + c4_lo) ^ xor_m)];
        const bfloat4 vb1 = ws_sg4[r0 * 16u + ((kk4 + 4u + c4_lo) ^ xor_m)];
        const bfloat4 vb2 = ws_sg4[r1 * 16u + ((kk4 + 0u + c4_lo) ^ xor_m)];
        const bfloat4 vb3 = ws_sg4[r1 * 16u + ((kk4 + 4u + c4_lo) ^ xor_m)];
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) {
          b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
          b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
        }
      }
      affine_mma_16x32x16_bf16(D0, D1, a0, b00, b10);
      affine_mma_16x32x16_bf16(D0, D1, a1, b01, b11);
    }
  }

  if constexpr (SK > 1u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *parts = reinterpret_cast<threadgroup float *>(ws_all4);
    if (sg > 0u) {
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 8u; ++i) {
        parts[((sg - 1u) * 16u + i) * 32u + lane] = D0[i];
        parts[((sg - 1u) * 16u + 8u + i) * 32u + lane] = D1[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      for (uint s = 1u; s < SK; ++s) {
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 8u; ++i) {
          D0[i] += parts[((s - 1u) * 16u + i) * 32u + lane];
          D1[i] += parts[((s - 1u) * 16u + 8u + i) * 32u + lane];
        }
      }
    }
  }

  if (sg == 0u) {
    const bool aligned_n = ((p.n & 3u) == 0u);
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint row = rb + fm + h * 8u;
      if (row < p.rows) {
        const uint e = h * 4u;
        const uint col0 = n0_sg + fn;
        if (col0 < p.n) {
          const bfloat4 out0 = bfloat4(bfloat(D0[e+0]), bfloat(D0[e+1]), bfloat(D0[e+2]), bfloat(D0[e+3]));
          if (aligned_n && col0 + 4u <= p.n) {
            *reinterpret_cast<device bfloat4 *>(y + ulong(row) * p.n + col0) = out0;
          } else {
            for (uint j = 0u; j < 4u; ++j) {
              if (col0 + j < p.n) y[ulong(row) * p.n + col0 + j] = out0[j];
            }
          }
        }
        const uint col1 = n0_sg + 16u + fn;
        if (col1 < p.n) {
          const bfloat4 out1 = bfloat4(bfloat(D1[e+0]), bfloat(D1[e+1]), bfloat(D1[e+2]), bfloat(D1[e+3]));
          if (aligned_n && col1 + 4u <= p.n) {
            *reinterpret_cast<device bfloat4 *>(y + ulong(row) * p.n + col1) = out1;
          } else {
            for (uint j = 0u; j < 4u; ++j) {
              if (col1 + j < p.n) y[ulong(row) * p.n + col1 + j] = out1[j];
            }
          }
        }
      }
    }
  }
}

template <uint SK>
inline void affine_nax_q5_h1_bm16_swiglu_apple9(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_high1,
    device const uint *g_sb_tiled,
    device const uchar *u_codes,
    device const uchar *u_high1,
    device const uint *u_sb_tiled,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup bfloat4 *ws_all4,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint kBK = 64u;
  const uint rb = tg.y * 16u;
  const uint tile_n = tg.x;
  const uint n0_sg = tile_n * 16u;
  if (rb >= p.rows || n0_sg >= p.n) return;

  const uint groups = p.k / kBK;
  const uint g0 = (sg * groups) / SK;
  const uint g1 = ((sg + 1u) * groups) / SK;
  threadgroup bfloat4 *ws_sg4 = ws_all4 + sg * 512u;

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 16u; ++i) ct_c[i] = 0.0f;

  for (uint g = g0; g < g1; ++g) {
    const uint k0 = g * kBK;
    const ulong idx = ulong(tile_n) * ulong(groups) + ulong(g);
    uint4 raw_c[2];
    uint raw_h[2], sb_u[2];
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      const uint idx32 = lane + f * 32u;
      const uint wr = idx32 >> 1u;
      const uint hk = idx32 & 1u;
      const bool is_up = (wr >= 16u);
      const uint rn = wr & 15u;
      if (n0_sg + rn < p.n) {
        raw_c[f] = reinterpret_cast<device const uint4 *>(
            (is_up ? u_codes : g_codes) + idx * 512ul + ulong(rn) * 32ul)[hk];
        raw_h[f] = reinterpret_cast<device const uint *>(
            (is_up ? u_high1 : g_high1) + idx * 128ul + ulong(rn) * 8ul)[hk];
        sb_u[f] = (is_up ? u_sb_tiled : g_sb_tiled)[idx * 16ul + rn];
      } else {
        raw_c[f] = uint4(0u); raw_h[f] = 0u; sb_u[f] = 0u;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      const uint idx32 = lane + f * 32u;
      affine_unpack_q5_h1_32_xor(raw_c[f], raw_h[f], sb_u[f], idx32 >> 1u, idx32 & 1u, ws_sg4);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (uint kk = 0u; kk < kBK; kk += 16u) {
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) {
        const auto ib = ct_b.get_multidimensional_index(i);
        ct_b[i] = affine_ws_read_xor(ws_sg4, uint(ib[1]), kk + uint(ib[0]));
      }
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 8u; ++i) {
        const auto ia = ct_a.get_multidimensional_index(i);
        const uint r0 = rb + uint(ia[1]);
        ct_a[i] = (r0 < p.rows) ? x[ulong(r0) * p.k + k0 + kk + uint(ia[0])] : bfloat(0.0f);
      }
      gemm_op.run(ct_a, ct_b, ct_c);
    }
  }

  if constexpr (SK > 1u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *parts = reinterpret_cast<threadgroup float *>(ws_all4);
    if (sg > 0u) {
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) parts[((sg - 1u) * 16u + i) * 32u + lane] = ct_c[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      for (uint s = 1u; s < SK; ++s) {
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) ct_c[i] += parts[((s - 1u) * 16u + i) * 32u + lane];
      }
    }
  }
  if (sg == 0u) {
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      const auto ig = ct_c.get_multidimensional_index(i);
      const uint cg = uint(ig[0]);
      const uint rg = uint(ig[1]);
      if (cg < 16u) {
        const uint col = n0_sg + cg;
        const uint row = rb + rg;
        if (row < p.rows && col < p.n) {
          float gv = ct_c[i];
          float uv = 0.0f;
          #pragma clang loop unroll(full)
          for (ushort j = 0u; j < 16u; ++j) {
            const auto iu = ct_c.get_multidimensional_index(j);
            if (uint(iu[0]) == cg + 16u && uint(iu[1]) == rg) uv = ct_c[j];
          }
          gv = float(bfloat(gv));
          uv = float(bfloat(uv));
          y[ulong(row) * p.n + col] = bfloat(nax_silu_f32(gv) * uv);
        }
      }
    }
  }
}

template <uint SK>
inline void affine_nax_q5_h1_bm16_swiglu_impl(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_high1,
    device const uint *g_sb_tiled,
    device const uchar *u_codes,
    device const uchar *u_high1,
    device const uint *u_sb_tiled,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup bfloat4 *ws_all4,
    uint lane,
    uint sg,
    uint3 tg) {
  if (!affine_has_apple10_nax_layout()) {
    affine_nax_q5_h1_bm16_swiglu_apple9<SK>(
        x, g_codes, g_high1, g_sb_tiled, u_codes, u_high1, u_sb_tiled, y, p, ws_all4, lane, sg, tg);
    return;
  }
  constexpr uint kBK = 64u;
  const uint rb = tg.y * 16u;
  const uint tile_n = tg.x;
  const uint n0_sg = tile_n * 16u;
  if (rb >= p.rows || n0_sg >= p.n) return;

  const uint groups = p.k / kBK;
  const uint g0 = (sg * groups) / SK;
  const uint g1 = ((sg + 1u) * groups) / SK;

  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
  const uint c4_lo = fn >> 2u;

  vec<float, 8> Dg(0.0f), Du(0.0f);
  threadgroup bfloat4 *ws_sg4 = ws_all4 + sg * 512u;

  device const uchar *wc_ptr[2];
  device const uchar *wh_ptr[2];
  device const uint *wsb_ptr[2];
  uint w_row_f[2], half_k_f[2];
  const ulong tbase = ulong(tile_n) * ulong(groups);
  #pragma clang loop unroll(full)
  for (uint f = 0u; f < 2u; ++f) {
    const uint idx32 = lane + f * 32u;
    const uint wr = idx32 >> 1u;
    const uint hk = idx32 & 1u;
    const bool is_up = (wr >= 16u);
    const uint rn = wr & 15u;
    const uint col = n0_sg + rn;
    wc_ptr[f] = (col < p.n) ? ((is_up ? u_codes : g_codes) + tbase * 512ul + ulong(rn) * 32ul) : nullptr;
    wh_ptr[f] = (col < p.n) ? ((is_up ? u_high1 : g_high1) + tbase * 128ul + ulong(rn) * 8ul) : nullptr;
    wsb_ptr[f] = (col < p.n) ? ((is_up ? u_sb_tiled : g_sb_tiled) + tbase * 16ul + rn) : nullptr;
    w_row_f[f] = wr;
    half_k_f[f] = hk;
  }

  const uint row0 = rb + fm;
  const uint row1 = rb + fm + 8u;
  const bool has_r0 = (row0 < p.rows);
  const bool has_r1 = (row1 < p.rows);
  device const bfloat *x_r0 = x + ulong(row0) * p.k + fn;
  device const bfloat *x_r1 = x + ulong(row1) * p.k + fn;

  for (uint g = g0; g < g1; ++g) {
    const uint k0 = g * kBK;
    uint4 raw_c[2];
    uint raw_h[2], sb_u[2];
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      if (wc_ptr[f] != nullptr) {
        raw_c[f] = reinterpret_cast<device const uint4 *>(wc_ptr[f] + ulong(g) * 512ul)[half_k_f[f]];
        raw_h[f] = reinterpret_cast<device const uint *>(wh_ptr[f] + ulong(g) * 128ul)[half_k_f[f]];
        sb_u[f] = wsb_ptr[f][ulong(g) * 16ul];
      } else {
        raw_c[f] = uint4(0u); raw_h[f] = 0u; sb_u[f] = 0u;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      affine_unpack_q5_h1_32_xor(raw_c[f], raw_h[f], sb_u[f], w_row_f[f], half_k_f[f], ws_sg4);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint xor_m = (fm & 3u) << 2u;
    #pragma clang loop unroll(full)
    for (uint kk4 = 0u; kk4 < 16u; kk4 += 8u) {
      const uint k_cur = k0 + kk4 * 4u;
      vec<bfloat, 8> a0, a1, b00, b01, b10, b11;
      const bfloat4 va00 = has_r0 ? *reinterpret_cast<device const bfloat4 *>(x_r0 + k_cur + 0u) : bfloat4(0.0bf);
      const bfloat4 va01 = has_r0 ? *reinterpret_cast<device const bfloat4 *>(x_r0 + k_cur + 16u) : bfloat4(0.0bf);
      const bfloat4 va10 = has_r1 ? *reinterpret_cast<device const bfloat4 *>(x_r1 + k_cur + 0u) : bfloat4(0.0bf);
      const bfloat4 va11 = has_r1 ? *reinterpret_cast<device const bfloat4 *>(x_r1 + k_cur + 16u) : bfloat4(0.0bf);
      #pragma clang loop unroll(full)
      for (uint c = 0u; c < 4u; ++c) {
        a0[c] = va00[c]; a0[4u + c] = va10[c];
        a1[c] = va01[c]; a1[4u + c] = va11[c];
      }
      #pragma clang loop unroll(full)
      for (uint r = 0u; r < 2u; ++r) {
        const uint r0 = fm + r * 8u;
        const uint r1 = 16u + fm + r * 8u;
        const bfloat4 vb0 = ws_sg4[r0 * 16u + ((kk4 + 0u + c4_lo) ^ xor_m)];
        const bfloat4 vb1 = ws_sg4[r0 * 16u + ((kk4 + 4u + c4_lo) ^ xor_m)];
        const bfloat4 vb2 = ws_sg4[r1 * 16u + ((kk4 + 0u + c4_lo) ^ xor_m)];
        const bfloat4 vb3 = ws_sg4[r1 * 16u + ((kk4 + 4u + c4_lo) ^ xor_m)];
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) {
          b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
          b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
        }
      }
      affine_mma_16x32x16_bf16(Dg, Du, a0, b00, b10);
      affine_mma_16x32x16_bf16(Dg, Du, a1, b01, b11);
    }
  }

  if constexpr (SK > 1u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *parts = reinterpret_cast<threadgroup float *>(ws_all4);
    if (sg > 0u) {
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 8u; ++i) {
        parts[((sg - 1u) * 16u + i) * 32u + lane] = Dg[i];
        parts[((sg - 1u) * 16u + 8u + i) * 32u + lane] = Du[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      for (uint s = 1u; s < SK; ++s) {
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 8u; ++i) {
          Dg[i] += parts[((s - 1u) * 16u + i) * 32u + lane];
          Du[i] += parts[((s - 1u) * 16u + 8u + i) * 32u + lane];
        }
      }
    }
  }

  if (sg == 0u) {
    const bool aligned_n = ((p.n & 3u) == 0u);
    const uint col = n0_sg + fn;
    if (col < p.n) {
      #pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const uint row = rb + fm + h * 8u;
        if (row < p.rows) {
          const uint e = h * 4u;
          bfloat4 out_b;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            const float gv = float(bfloat(Dg[e + j]));
            const float uv = float(bfloat(Du[e + j]));
            out_b[j] = bfloat(nax_silu_f32(gv) * uv);
          }
          if (aligned_n && col + 4u <= p.n) {
            *reinterpret_cast<device bfloat4 *>(y + ulong(row) * p.n + col) = out_b;
          } else {
            for (uint j = 0u; j < 4u; ++j) {
              if (col + j < p.n) y[ulong(row) * p.n + col + j] = out_b[j];
            }
          }
        }
      }
    }
  }
}

#define OMLX_M5_AFFINE_Q5_H1_BM16_LINEAR(NAME, BNS, SK)                         \
  [[max_total_threads_per_threadgroup((SK) * 32u)]]                             \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *codes [[buffer(1)]],                                  \
      device const uchar *high1 [[buffer(2)]],                                  \
      device const uint *sb_tiled [[buffer(3)]],                                \
      device bfloat *y [[buffer(4)]],                                           \
      constant AffineNaxParams &p [[buffer(5)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    alignas(16) threadgroup bfloat4 ws[(SK) * 512u];                            \
    affine_nax_q5_h1_bm16_linear_impl<BNS, SK>(                                 \
        x, codes, high1, sb_tiled, y, p, ws, lane, sg, tg);                     \
  }

#define OMLX_M5_AFFINE_Q5_H1_BM16_SWIGLU(NAME, SK)                              \
  [[max_total_threads_per_threadgroup((SK) * 32u)]]                             \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *g_codes [[buffer(1)]],                                \
      device const uchar *g_high1 [[buffer(2)]],                                \
      device const uint *g_sb_tiled [[buffer(3)]],                              \
      device const uchar *u_codes [[buffer(4)]],                                \
      device const uchar *u_high1 [[buffer(5)]],                                \
      device const uint *u_sb_tiled [[buffer(6)]],                              \
      device bfloat *y [[buffer(7)]],                                           \
      constant AffineNaxParams &p [[buffer(8)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    alignas(16) threadgroup bfloat4 ws[(SK) * 512u];                            \
    affine_nax_q5_h1_bm16_swiglu_impl<SK>(                                      \
        x, g_codes, g_high1, g_sb_tiled, u_codes, u_high1, u_sb_tiled, y, p,    \
        ws, lane, sg, tg);                                                      \
  }

OMLX_M5_AFFINE_Q5_H1_BM16_LINEAR(omlx_m5_affine_nax_q5_h1_bm16_bns16_sk2, 16u, 2u)
OMLX_M5_AFFINE_Q5_H1_BM16_LINEAR(omlx_m5_affine_nax_q5_h1_bm16_bns16_sk4, 16u, 4u)
OMLX_M5_AFFINE_Q5_H1_BM16_LINEAR(omlx_m5_affine_nax_q5_h1_bm16_bns16_sk8, 16u, 8u)
OMLX_M5_AFFINE_Q5_H1_BM16_LINEAR(omlx_m5_affine_nax_q5_h1_bm16_bns32_sk2, 32u, 2u)
OMLX_M5_AFFINE_Q5_H1_BM16_LINEAR(omlx_m5_affine_nax_q5_h1_bm16_bns32_sk4, 32u, 4u)
OMLX_M5_AFFINE_Q5_H1_BM16_SWIGLU(omlx_m5_affine_nax_q5_h1_swiglu_bm16_sk2, 2u)
OMLX_M5_AFFINE_Q5_H1_BM16_SWIGLU(omlx_m5_affine_nax_q5_h1_swiglu_bm16_sk4, 4u)

// ============================================================================
// v0.8.0-m5rt.14 Q5-Only BM=32, BN=32, BK=64 Sibling (0-ULP vs Q5 BM=16!)
// Halves Q5 weight DRAM reads for M=17..32 (B=4 DFlash2/MTP verify), reusing
// the exact same ws_sg4 unpack + 16x32x16 NAX MMA across two 16-row slices and
// reducing both halves inside ws_all4 ((SK-1)*4096 B < SK*4096 B) in 1 pass!
// ============================================================================

template <uint BN_STORED, uint SK>
inline void affine_nax_q5_h1_bm32_linear_apple9(
    device const bfloat *x,
    device const uchar *codes,
    device const uchar *high1,
    device const uint *sb_tiled,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup bfloat4 *ws_all4,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint kBK = 64u;
  const uint rb = tg.y * 32u;
  const uint n0_sg = tg.x * 32u;
  if (rb >= p.rows || n0_sg >= p.n) return;

  const uint groups = p.k / kBK;
  const uint g0 = (sg * groups) / SK;
  const uint g1 = ((sg + 1u) * groups) / SK;
  threadgroup bfloat4 *ws_sg4 = ws_all4 + sg * 512u;

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_c0 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
  auto ct_c1 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
#pragma clang loop unroll(full)
  for (ushort i = 0u; i < 16u; ++i) {
    ct_c0[i] = 0.0f;
    ct_c1[i] = 0.0f;
  }

  for (uint g = g0; g < g1; ++g) {
    const uint k0 = g * kBK;
    uint4 raw_c[2];
    uint raw_h[2], sb_u[2];
#pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      const uint idx32 = lane + f * 32u;
      const uint wr = idx32 >> 1u;
      const uint hk = idx32 & 1u;
      const uint col = n0_sg + wr;
      if (col < p.n) {
        const uint tile_n = col / BN_STORED;
        const uint rn = col % BN_STORED;
        const ulong idx = ulong(tile_n) * ulong(groups) + ulong(g);
        raw_c[f] = reinterpret_cast<device const uint4 *>(
            codes + idx * (ulong(BN_STORED) * 32ul) + ulong(rn) * 32ul)[hk];
        raw_h[f] = reinterpret_cast<device const uint *>(
            high1 + idx * (ulong(BN_STORED) * 8ul) + ulong(rn) * 8ul)[hk];
        sb_u[f] = sb_tiled[idx * ulong(BN_STORED) + rn];
      } else {
        raw_c[f] = uint4(0u); raw_h[f] = 0u; sb_u[f] = 0u;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      const uint idx32 = lane + f * 32u;
      affine_unpack_q5_h1_32_xor(raw_c[f], raw_h[f], sb_u[f], idx32 >> 1u, idx32 & 1u, ws_sg4);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const bool has_hi = (rb + 16u < p.rows);
#pragma clang loop unroll(full)
    for (uint kk = 0u; kk < kBK; kk += 16u) {
#pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) {
        const auto ib = ct_b.get_multidimensional_index(i);
        ct_b[i] = affine_ws_read_xor(ws_sg4, uint(ib[1]), kk + uint(ib[0]));
      }
#pragma clang loop unroll(full)
      for (ushort i = 0u; i < 8u; ++i) {
        const auto ia = ct_a.get_multidimensional_index(i);
        const uint r0 = rb + uint(ia[1]);
        ct_a[i] = (r0 < p.rows) ? x[ulong(r0) * p.k + k0 + kk + uint(ia[0])] : bfloat(0.0f);
      }
      gemm_op.run(ct_a, ct_b, ct_c0);
      if (has_hi) {
#pragma clang loop unroll(full)
        for (ushort i = 0u; i < 8u; ++i) {
          const auto ia = ct_a.get_multidimensional_index(i);
          const uint r1 = rb + 16u + uint(ia[1]);
          ct_a[i] = (r1 < p.rows) ? x[ulong(r1) * p.k + k0 + kk + uint(ia[0])] : bfloat(0.0f);
        }
        gemm_op.run(ct_a, ct_b, ct_c1);
      }
    }
  }

  if constexpr (SK > 1u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *parts = reinterpret_cast<threadgroup float *>(ws_all4);
    if (sg > 0u) {
#pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) {
        parts[((sg - 1u) * 32u + i) * 32u + lane] = ct_c0[i];
        parts[((sg - 1u) * 32u + 16u + i) * 32u + lane] = ct_c1[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      for (uint s = 1u; s < SK; ++s) {
#pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          ct_c0[i] += parts[((s - 1u) * 32u + i) * 32u + lane];
          ct_c1[i] += parts[((s - 1u) * 32u + 16u + i) * 32u + lane];
        }
      }
    }
  }
  if (sg == 0u) {
#pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      const auto ic = ct_c0.get_multidimensional_index(i);
      const uint col = n0_sg + uint(ic[0]);
      const uint r0 = rb + uint(ic[1]);
      if (r0 < p.rows && col < p.n) y[ulong(r0) * p.n + col] = bfloat(ct_c0[i]);
      const uint r1 = r0 + 16u;
      if (r1 < p.rows && col < p.n) y[ulong(r1) * p.n + col] = bfloat(ct_c1[i]);
    }
  }
}

template <uint BN_STORED, uint SK>
inline void affine_nax_q5_h1_bm32_linear_impl(
    device const bfloat *x,
    device const uchar *codes,
    device const uchar *high1,
    device const uint *sb_tiled,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup bfloat4 *ws_all4,
    uint lane,
    uint sg,
    uint3 tg) {
  if (!affine_has_apple10_nax_layout()) {
    affine_nax_q5_h1_bm32_linear_apple9<BN_STORED, SK>(
        x, codes, high1, sb_tiled, y, p, ws_all4, lane, sg, tg);
    return;
  }
  constexpr uint kBK = 64u;
  const uint rb = tg.y * 32u;
  const uint n0_sg = tg.x * 32u;
  if (rb >= p.rows || n0_sg >= p.n) return;

  const uint groups = p.k / kBK;
  const uint g0 = (sg * groups) / SK;
  const uint g1 = ((sg + 1u) * groups) / SK;

  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
  const uint c4_lo = fn >> 2u;

  vec<float, 8> D00(0.0f), D01(0.0f), D10(0.0f), D11(0.0f);
  threadgroup bfloat4 *ws_sg4 = ws_all4 + sg * 512u;

  device const uchar *wc_ptr[2];
  device const uchar *wh_ptr[2];
  device const uint *wsb_ptr[2];
  uint w_row_f[2], half_k_f[2];
#pragma clang loop unroll(full)
  for (uint f = 0u; f < 2u; ++f) {
    const uint idx32 = lane + f * 32u;
    const uint wr = idx32 >> 1u;
    const uint hk = idx32 & 1u;
    const uint col = n0_sg + wr;
    const uint tile_n = col / BN_STORED;
    const uint rn = col % BN_STORED;
    const ulong tbase = ulong(tile_n) * ulong(groups);
    wc_ptr[f] = (col < p.n) ? (codes + tbase * (ulong(BN_STORED) * 32ul) + ulong(rn) * 32ul) : nullptr;
    wh_ptr[f] = (col < p.n) ? (high1 + tbase * (ulong(BN_STORED) * 8ul) + ulong(rn) * 8ul) : nullptr;
    wsb_ptr[f] = (col < p.n) ? (sb_tiled + tbase * ulong(BN_STORED) + rn) : nullptr;
    w_row_f[f] = wr;
    half_k_f[f] = hk;
  }

  const uint row0 = rb + fm;
  const uint row1 = rb + fm + 8u;
  const uint row2 = rb + fm + 16u;
  const uint row3 = rb + fm + 24u;
  const bool has_r0 = (row0 < p.rows);
  const bool has_r1 = (row1 < p.rows);
  const bool has_r2 = (row2 < p.rows);
  const bool has_r3 = (row3 < p.rows);
  const bool has_hi = (rb + 16u < p.rows);
  device const bfloat *x_r0 = x + ulong(row0) * p.k + fn;
  device const bfloat *x_r1 = x + ulong(row1) * p.k + fn;
  device const bfloat *x_r2 = x + ulong(row2) * p.k + fn;
  device const bfloat *x_r3 = x + ulong(row3) * p.k + fn;

  for (uint g = g0; g < g1; ++g) {
    const uint k0 = g * kBK;
    uint4 raw_c[2];
    uint raw_h[2], sb_u[2];
#pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      if (wc_ptr[f] != nullptr) {
        raw_c[f] = reinterpret_cast<device const uint4 *>(wc_ptr[f] + ulong(g) * (ulong(BN_STORED) * 32ul))[half_k_f[f]];
        raw_h[f] = reinterpret_cast<device const uint *>(wh_ptr[f] + ulong(g) * (ulong(BN_STORED) * 8ul))[half_k_f[f]];
        sb_u[f] = wsb_ptr[f][ulong(g) * ulong(BN_STORED)];
      } else {
        raw_c[f] = uint4(0u); raw_h[f] = 0u; sb_u[f] = 0u;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      affine_unpack_q5_h1_32_xor(raw_c[f], raw_h[f], sb_u[f], w_row_f[f], half_k_f[f], ws_sg4);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint xor_m = (fm & 3u) << 2u;
#pragma clang loop unroll(full)
    for (uint kk4 = 0u; kk4 < 16u; kk4 += 8u) {
      const uint k_cur = k0 + kk4 * 4u;
      vec<bfloat, 8> b00, b01, b10, b11;
#pragma clang loop unroll(full)
      for (uint r = 0u; r < 2u; ++r) {
        const uint r0 = fm + r * 8u;
        const uint r1 = 16u + fm + r * 8u;
        const bfloat4 vb0 = ws_sg4[r0 * 16u + ((kk4 + 0u + c4_lo) ^ xor_m)];
        const bfloat4 vb1 = ws_sg4[r0 * 16u + ((kk4 + 4u + c4_lo) ^ xor_m)];
        const bfloat4 vb2 = ws_sg4[r1 * 16u + ((kk4 + 0u + c4_lo) ^ xor_m)];
        const bfloat4 vb3 = ws_sg4[r1 * 16u + ((kk4 + 4u + c4_lo) ^ xor_m)];
#pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) {
          b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
          b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
        }
      }
      {
        vec<bfloat, 8> a0, a1;
        const bfloat4 va00 = has_r0 ? *reinterpret_cast<device const bfloat4 *>(x_r0 + k_cur + 0u) : bfloat4(0.0bf);
        const bfloat4 va01 = has_r0 ? *reinterpret_cast<device const bfloat4 *>(x_r0 + k_cur + 16u) : bfloat4(0.0bf);
        const bfloat4 va10 = has_r1 ? *reinterpret_cast<device const bfloat4 *>(x_r1 + k_cur + 0u) : bfloat4(0.0bf);
        const bfloat4 va11 = has_r1 ? *reinterpret_cast<device const bfloat4 *>(x_r1 + k_cur + 16u) : bfloat4(0.0bf);
#pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) {
          a0[c] = va00[c]; a0[4u + c] = va10[c];
          a1[c] = va01[c]; a1[4u + c] = va11[c];
        }
        affine_mma_16x32x16_bf16(D00, D01, a0, b00, b10);
        affine_mma_16x32x16_bf16(D00, D01, a1, b01, b11);
      }
      if (has_hi) {
        vec<bfloat, 8> a2, a3;
        const bfloat4 va20 = has_r2 ? *reinterpret_cast<device const bfloat4 *>(x_r2 + k_cur + 0u) : bfloat4(0.0bf);
        const bfloat4 va21 = has_r2 ? *reinterpret_cast<device const bfloat4 *>(x_r2 + k_cur + 16u) : bfloat4(0.0bf);
        const bfloat4 va30 = has_r3 ? *reinterpret_cast<device const bfloat4 *>(x_r3 + k_cur + 0u) : bfloat4(0.0bf);
        const bfloat4 va31 = has_r3 ? *reinterpret_cast<device const bfloat4 *>(x_r3 + k_cur + 16u) : bfloat4(0.0bf);
#pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) {
          a2[c] = va20[c]; a2[4u + c] = va30[c];
          a3[c] = va21[c]; a3[4u + c] = va31[c];
        }
        affine_mma_16x32x16_bf16(D10, D11, a2, b00, b10);
        affine_mma_16x32x16_bf16(D10, D11, a3, b01, b11);
      }
    }
  }

  if constexpr (SK > 1u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *parts = reinterpret_cast<threadgroup float *>(ws_all4);
    if (sg > 0u) {
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 8u; ++i) {
        parts[((sg - 1u) * 32u + i) * 32u + lane] = D00[i];
        parts[((sg - 1u) * 32u + 8u + i) * 32u + lane] = D01[i];
        parts[((sg - 1u) * 32u + 16u + i) * 32u + lane] = D10[i];
        parts[((sg - 1u) * 32u + 24u + i) * 32u + lane] = D11[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      for (uint s = 1u; s < SK; ++s) {
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 8u; ++i) {
          D00[i] += parts[((s - 1u) * 32u + i) * 32u + lane];
          D01[i] += parts[((s - 1u) * 32u + 8u + i) * 32u + lane];
          D10[i] += parts[((s - 1u) * 32u + 16u + i) * 32u + lane];
          D11[i] += parts[((s - 1u) * 32u + 24u + i) * 32u + lane];
        }
      }
    }
  }

  if (sg == 0u) {
    const bool aligned_n = ((p.n & 3u) == 0u);
    const uint col0 = n0_sg + fn;
    const uint col1 = n0_sg + 16u + fn;
#pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint row = rb + fm + h * 8u;
      if (row < p.rows) {
        const uint e = h * 4u;
        if (col0 < p.n) {
          const bfloat4 out0 = bfloat4(bfloat(D00[e+0]), bfloat(D00[e+1]), bfloat(D00[e+2]), bfloat(D00[e+3]));
          if (aligned_n && col0 + 4u <= p.n) {
            *reinterpret_cast<device bfloat4 *>(y + ulong(row) * p.n + col0) = out0;
          } else {
            for (uint j = 0u; j < 4u; ++j) {
              if (col0 + j < p.n) y[ulong(row) * p.n + col0 + j] = out0[j];
            }
          }
        }
        if (col1 < p.n) {
          const bfloat4 out1 = bfloat4(bfloat(D01[e+0]), bfloat(D01[e+1]), bfloat(D01[e+2]), bfloat(D01[e+3]));
          if (aligned_n && col1 + 4u <= p.n) {
            *reinterpret_cast<device bfloat4 *>(y + ulong(row) * p.n + col1) = out1;
          } else {
            for (uint j = 0u; j < 4u; ++j) {
              if (col1 + j < p.n) y[ulong(row) * p.n + col1 + j] = out1[j];
            }
          }
        }
      }
    }
    if (has_hi) {
#pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const uint row = rb + 16u + fm + h * 8u;
        if (row < p.rows) {
          const uint e = h * 4u;
          if (col0 < p.n) {
            const bfloat4 out0 = bfloat4(bfloat(D10[e+0]), bfloat(D10[e+1]), bfloat(D10[e+2]), bfloat(D10[e+3]));
            if (aligned_n && col0 + 4u <= p.n) {
              *reinterpret_cast<device bfloat4 *>(y + ulong(row) * p.n + col0) = out0;
            } else {
              for (uint j = 0u; j < 4u; ++j) {
                if (col0 + j < p.n) y[ulong(row) * p.n + col0 + j] = out0[j];
              }
            }
          }
          if (col1 < p.n) {
            const bfloat4 out1 = bfloat4(bfloat(D11[e+0]), bfloat(D11[e+1]), bfloat(D11[e+2]), bfloat(D11[e+3]));
            if (aligned_n && col1 + 4u <= p.n) {
              *reinterpret_cast<device bfloat4 *>(y + ulong(row) * p.n + col1) = out1;
            } else {
              for (uint j = 0u; j < 4u; ++j) {
                if (col1 + j < p.n) y[ulong(row) * p.n + col1 + j] = out1[j];
              }
            }
          }
        }
      }
    }
  }
}

template <uint SK>
inline void affine_nax_q5_h1_bm32_swiglu_apple9(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_high1,
    device const uint *g_sb_tiled,
    device const uchar *u_codes,
    device const uchar *u_high1,
    device const uint *u_sb_tiled,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup bfloat4 *ws_all4,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint kBK = 64u;
  const uint rb = tg.y * 32u;
  const uint tile_n = tg.x;
  const uint n0_sg = tile_n * 16u;
  if (rb >= p.rows || n0_sg >= p.n) return;

  const uint groups = p.k / kBK;
  const uint g0 = (sg * groups) / SK;
  const uint g1 = ((sg + 1u) * groups) / SK;
  threadgroup bfloat4 *ws_sg4 = ws_all4 + sg * 512u;

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_c0 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
  auto ct_c1 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
#pragma clang loop unroll(full)
  for (ushort i = 0u; i < 16u; ++i) {
    ct_c0[i] = 0.0f;
    ct_c1[i] = 0.0f;
  }

  const bool has_hi = (rb + 16u < p.rows);
  for (uint g = g0; g < g1; ++g) {
    const uint k0 = g * kBK;
    const ulong idx = ulong(tile_n) * ulong(groups) + ulong(g);
    uint4 raw_c[2];
    uint raw_h[2], sb_u[2];
#pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      const uint idx32 = lane + f * 32u;
      const uint wr = idx32 >> 1u;
      const uint hk = idx32 & 1u;
      const bool is_up = (wr >= 16u);
      const uint rn = wr & 15u;
      if (n0_sg + rn < p.n) {
        raw_c[f] = reinterpret_cast<device const uint4 *>(
            (is_up ? u_codes : g_codes) + idx * 512ul + ulong(rn) * 32ul)[hk];
        raw_h[f] = reinterpret_cast<device const uint *>(
            (is_up ? u_high1 : g_high1) + idx * 128ul + ulong(rn) * 8ul)[hk];
        sb_u[f] = (is_up ? u_sb_tiled : g_sb_tiled)[idx * 16ul + rn];
      } else {
        raw_c[f] = uint4(0u); raw_h[f] = 0u; sb_u[f] = 0u;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      const uint idx32 = lane + f * 32u;
      affine_unpack_q5_h1_32_xor(raw_c[f], raw_h[f], sb_u[f], idx32 >> 1u, idx32 & 1u, ws_sg4);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
    for (uint kk = 0u; kk < kBK; kk += 16u) {
#pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) {
        const auto ib = ct_b.get_multidimensional_index(i);
        ct_b[i] = affine_ws_read_xor(ws_sg4, uint(ib[1]), kk + uint(ib[0]));
      }
#pragma clang loop unroll(full)
      for (ushort i = 0u; i < 8u; ++i) {
        const auto ia = ct_a.get_multidimensional_index(i);
        const uint r0 = rb + uint(ia[1]);
        ct_a[i] = (r0 < p.rows) ? x[ulong(r0) * p.k + k0 + kk + uint(ia[0])] : bfloat(0.0f);
      }
      gemm_op.run(ct_a, ct_b, ct_c0);
      if (has_hi) {
#pragma clang loop unroll(full)
        for (ushort i = 0u; i < 8u; ++i) {
          const auto ia = ct_a.get_multidimensional_index(i);
          const uint r1 = rb + 16u + uint(ia[1]);
          ct_a[i] = (r1 < p.rows) ? x[ulong(r1) * p.k + k0 + kk + uint(ia[0])] : bfloat(0.0f);
        }
        gemm_op.run(ct_a, ct_b, ct_c1);
      }
    }
  }

  if constexpr (SK > 1u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *parts = reinterpret_cast<threadgroup float *>(ws_all4);
    if (sg > 0u) {
#pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) {
        parts[((sg - 1u) * 32u + i) * 32u + lane] = ct_c0[i];
        parts[((sg - 1u) * 32u + 16u + i) * 32u + lane] = ct_c1[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      for (uint s = 1u; s < SK; ++s) {
#pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          ct_c0[i] += parts[((s - 1u) * 32u + i) * 32u + lane];
          ct_c1[i] += parts[((s - 1u) * 32u + 16u + i) * 32u + lane];
        }
      }
    }
  }
  if (sg == 0u) {
#pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm) {
      if (mm == 1u && !has_hi) break;
#pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) {
        const auto ig = ct_c0.get_multidimensional_index(i);
        const uint cg = uint(ig[0]);
        const uint rg = uint(ig[1]);
        if (cg < 16u) {
          const uint col = n0_sg + cg;
          const uint row = rb + mm * 16u + rg;
          if (row < p.rows && col < p.n) {
            float gv = (mm == 0u) ? ct_c0[i] : ct_c1[i];
            float uv = 0.0f;
#pragma clang loop unroll(full)
            for (ushort j = 0u; j < 16u; ++j) {
              const auto iu = ct_c0.get_multidimensional_index(j);
              if (uint(iu[0]) == cg + 16u && uint(iu[1]) == rg) {
                uv = (mm == 0u) ? ct_c0[j] : ct_c1[j];
              }
            }
            gv = float(bfloat(gv));
            uv = float(bfloat(uv));
            y[ulong(row) * p.n + col] = bfloat(nax_silu_f32(gv) * uv);
          }
        }
      }
    }
  }
}

template <uint SK>
inline void affine_nax_q5_h1_bm32_swiglu_impl(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_high1,
    device const uint *g_sb_tiled,
    device const uchar *u_codes,
    device const uchar *u_high1,
    device const uint *u_sb_tiled,
    device bfloat *y,
    constant AffineNaxParams &p,
    threadgroup bfloat4 *ws_all4,
    uint lane,
    uint sg,
    uint3 tg) {
  if (!affine_has_apple10_nax_layout()) {
    affine_nax_q5_h1_bm32_swiglu_apple9<SK>(
        x, g_codes, g_high1, g_sb_tiled, u_codes, u_high1, u_sb_tiled, y, p, ws_all4, lane, sg, tg);
    return;
  }
  constexpr uint kBK = 64u;
  const uint rb = tg.y * 32u;
  const uint tile_n = tg.x;
  const uint n0_sg = tile_n * 16u;
  if (rb >= p.rows || n0_sg >= p.n) return;

  const uint groups = p.k / kBK;
  const uint g0 = (sg * groups) / SK;
  const uint g1 = ((sg + 1u) * groups) / SK;

  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
  const uint c4_lo = fn >> 2u;

  vec<float, 8> Dg0(0.0f), Du0(0.0f), Dg1(0.0f), Du1(0.0f);
  threadgroup bfloat4 *ws_sg4 = ws_all4 + sg * 512u;

  device const uchar *wc_ptr[2];
  device const uchar *wh_ptr[2];
  device const uint *wsb_ptr[2];
  uint w_row_f[2], half_k_f[2];
  const ulong tbase = ulong(tile_n) * ulong(groups);
#pragma clang loop unroll(full)
  for (uint f = 0u; f < 2u; ++f) {
    const uint idx32 = lane + f * 32u;
    const uint wr = idx32 >> 1u;
    const uint hk = idx32 & 1u;
    const bool is_up = (wr >= 16u);
    const uint rn = wr & 15u;
    const uint col = n0_sg + rn;
    wc_ptr[f] = (col < p.n) ? ((is_up ? u_codes : g_codes) + tbase * 512ul + ulong(rn) * 32ul) : nullptr;
    wh_ptr[f] = (col < p.n) ? ((is_up ? u_high1 : g_high1) + tbase * 128ul + ulong(rn) * 8ul) : nullptr;
    wsb_ptr[f] = (col < p.n) ? ((is_up ? u_sb_tiled : g_sb_tiled) + tbase * 16ul + rn) : nullptr;
    w_row_f[f] = wr;
    half_k_f[f] = hk;
  }

  const uint row0 = rb + fm;
  const uint row1 = rb + fm + 8u;
  const uint row2 = rb + fm + 16u;
  const uint row3 = rb + fm + 24u;
  const bool has_r0 = (row0 < p.rows);
  const bool has_r1 = (row1 < p.rows);
  const bool has_r2 = (row2 < p.rows);
  const bool has_r3 = (row3 < p.rows);
  const bool has_hi = (rb + 16u < p.rows);
  device const bfloat *x_r0 = x + ulong(row0) * p.k + fn;
  device const bfloat *x_r1 = x + ulong(row1) * p.k + fn;
  device const bfloat *x_r2 = x + ulong(row2) * p.k + fn;
  device const bfloat *x_r3 = x + ulong(row3) * p.k + fn;

  for (uint g = g0; g < g1; ++g) {
    const uint k0 = g * kBK;
    uint4 raw_c[2];
    uint raw_h[2], sb_u[2];
#pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      if (wc_ptr[f] != nullptr) {
        raw_c[f] = reinterpret_cast<device const uint4 *>(wc_ptr[f] + ulong(g) * 512ul)[half_k_f[f]];
        raw_h[f] = reinterpret_cast<device const uint *>(wh_ptr[f] + ulong(g) * 128ul)[half_k_f[f]];
        sb_u[f] = wsb_ptr[f][ulong(g) * 16ul];
      } else {
        raw_c[f] = uint4(0u); raw_h[f] = 0u; sb_u[f] = 0u;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint f = 0u; f < 2u; ++f) {
      affine_unpack_q5_h1_32_xor(raw_c[f], raw_h[f], sb_u[f], w_row_f[f], half_k_f[f], ws_sg4);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint xor_m = (fm & 3u) << 2u;
#pragma clang loop unroll(full)
    for (uint kk4 = 0u; kk4 < 16u; kk4 += 8u) {
      const uint k_cur = k0 + kk4 * 4u;
      vec<bfloat, 8> b00, b01, b10, b11;
#pragma clang loop unroll(full)
      for (uint r = 0u; r < 2u; ++r) {
        const uint r0 = fm + r * 8u;
        const uint r1 = 16u + fm + r * 8u;
        const bfloat4 vb0 = ws_sg4[r0 * 16u + ((kk4 + 0u + c4_lo) ^ xor_m)];
        const bfloat4 vb1 = ws_sg4[r0 * 16u + ((kk4 + 4u + c4_lo) ^ xor_m)];
        const bfloat4 vb2 = ws_sg4[r1 * 16u + ((kk4 + 0u + c4_lo) ^ xor_m)];
        const bfloat4 vb3 = ws_sg4[r1 * 16u + ((kk4 + 4u + c4_lo) ^ xor_m)];
#pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) {
          b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
          b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
        }
      }
      {
        vec<bfloat, 8> a0, a1;
        const bfloat4 va00 = has_r0 ? *reinterpret_cast<device const bfloat4 *>(x_r0 + k_cur + 0u) : bfloat4(0.0bf);
        const bfloat4 va01 = has_r0 ? *reinterpret_cast<device const bfloat4 *>(x_r0 + k_cur + 16u) : bfloat4(0.0bf);
        const bfloat4 va10 = has_r1 ? *reinterpret_cast<device const bfloat4 *>(x_r1 + k_cur + 0u) : bfloat4(0.0bf);
        const bfloat4 va11 = has_r1 ? *reinterpret_cast<device const bfloat4 *>(x_r1 + k_cur + 16u) : bfloat4(0.0bf);
#pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) {
          a0[c] = va00[c]; a0[4u + c] = va10[c];
          a1[c] = va01[c]; a1[4u + c] = va11[c];
        }
        affine_mma_16x32x16_bf16(Dg0, Du0, a0, b00, b10);
        affine_mma_16x32x16_bf16(Dg0, Du0, a1, b01, b11);
      }
      if (has_hi) {
        vec<bfloat, 8> a2, a3;
        const bfloat4 va20 = has_r2 ? *reinterpret_cast<device const bfloat4 *>(x_r2 + k_cur + 0u) : bfloat4(0.0bf);
        const bfloat4 va21 = has_r2 ? *reinterpret_cast<device const bfloat4 *>(x_r2 + k_cur + 16u) : bfloat4(0.0bf);
        const bfloat4 va30 = has_r3 ? *reinterpret_cast<device const bfloat4 *>(x_r3 + k_cur + 0u) : bfloat4(0.0bf);
        const bfloat4 va31 = has_r3 ? *reinterpret_cast<device const bfloat4 *>(x_r3 + k_cur + 16u) : bfloat4(0.0bf);
#pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) {
          a2[c] = va20[c]; a2[4u + c] = va30[c];
          a3[c] = va21[c]; a3[4u + c] = va31[c];
        }
        affine_mma_16x32x16_bf16(Dg1, Du1, a2, b00, b10);
        affine_mma_16x32x16_bf16(Dg1, Du1, a3, b01, b11);
      }
    }
  }

  if constexpr (SK > 1u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *parts = reinterpret_cast<threadgroup float *>(ws_all4);
    if (sg > 0u) {
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 8u; ++i) {
        parts[((sg - 1u) * 32u + i) * 32u + lane] = Dg0[i];
        parts[((sg - 1u) * 32u + 8u + i) * 32u + lane] = Du0[i];
        parts[((sg - 1u) * 32u + 16u + i) * 32u + lane] = Dg1[i];
        parts[((sg - 1u) * 32u + 24u + i) * 32u + lane] = Du1[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      for (uint s = 1u; s < SK; ++s) {
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 8u; ++i) {
          Dg0[i] += parts[((s - 1u) * 32u + i) * 32u + lane];
          Du0[i] += parts[((s - 1u) * 32u + 8u + i) * 32u + lane];
          Dg1[i] += parts[((s - 1u) * 32u + 16u + i) * 32u + lane];
          Du1[i] += parts[((s - 1u) * 32u + 24u + i) * 32u + lane];
        }
      }
    }
  }

  if (sg == 0u) {
    const bool aligned_n = ((p.n & 3u) == 0u);
    const uint col = n0_sg + fn;
    if (col < p.n) {
#pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const uint row = rb + fm + h * 8u;
        if (row < p.rows) {
          const uint e = h * 4u;
          bfloat4 out_b;
#pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            const float gv = float(bfloat(Dg0[e + j]));
            const float uv = float(bfloat(Du0[e + j]));
            out_b[j] = bfloat(nax_silu_f32(gv) * uv);
          }
          if (aligned_n && col + 4u <= p.n) {
            *reinterpret_cast<device bfloat4 *>(y + ulong(row) * p.n + col) = out_b;
          } else {
            for (uint j = 0u; j < 4u; ++j) {
              if (col + j < p.n) y[ulong(row) * p.n + col + j] = out_b[j];
            }
          }
        }
      }
      if (has_hi) {
#pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint row = rb + 16u + fm + h * 8u;
          if (row < p.rows) {
            const uint e = h * 4u;
            bfloat4 out_b;
#pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) {
              const float gv = float(bfloat(Dg1[e + j]));
              const float uv = float(bfloat(Du1[e + j]));
              out_b[j] = bfloat(nax_silu_f32(gv) * uv);
            }
            if (aligned_n && col + 4u <= p.n) {
              *reinterpret_cast<device bfloat4 *>(y + ulong(row) * p.n + col) = out_b;
            } else {
              for (uint j = 0u; j < 4u; ++j) {
                if (col + j < p.n) y[ulong(row) * p.n + col + j] = out_b[j];
              }
            }
          }
        }
      }
    }
  }
}

#define OMLX_M5_AFFINE_Q5_H1_BM32_LINEAR(NAME, BNS, SK)                         \
  [[max_total_threads_per_threadgroup((SK) * 32u)]]                             \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *codes [[buffer(1)]],                                  \
      device const uchar *high1 [[buffer(2)]],                                  \
      device const uint *sb_tiled [[buffer(3)]],                                \
      device bfloat *y [[buffer(4)]],                                           \
      constant AffineNaxParams &p [[buffer(5)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    alignas(16) threadgroup bfloat4 ws[(SK) * 512u];                            \
    affine_nax_q5_h1_bm32_linear_impl<BNS, SK>(                                 \
        x, codes, high1, sb_tiled, y, p, ws, lane, sg, tg);                     \
  }

#define OMLX_M5_AFFINE_Q5_H1_BM32_SWIGLU(NAME, SK)                              \
  [[max_total_threads_per_threadgroup((SK) * 32u)]]                             \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *g_codes [[buffer(1)]],                                \
      device const uchar *g_high1 [[buffer(2)]],                                \
      device const uint *g_sb_tiled [[buffer(3)]],                              \
      device const uchar *u_codes [[buffer(4)]],                                \
      device const uchar *u_high1 [[buffer(5)]],                                \
      device const uint *u_sb_tiled [[buffer(6)]],                              \
      device bfloat *y [[buffer(7)]],                                           \
      constant AffineNaxParams &p [[buffer(8)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    alignas(16) threadgroup bfloat4 ws[(SK) * 512u];                            \
    affine_nax_q5_h1_bm32_swiglu_impl<SK>(                                      \
        x, g_codes, g_high1, g_sb_tiled, u_codes, u_high1, u_sb_tiled, y, p,    \
        ws, lane, sg, tg);                                                      \
  }

OMLX_M5_AFFINE_Q5_H1_BM32_LINEAR(omlx_m5_affine_nax_q5_h1_bm32_bns16_sk2, 16u, 2u)
OMLX_M5_AFFINE_Q5_H1_BM32_LINEAR(omlx_m5_affine_nax_q5_h1_bm32_bns16_sk4, 16u, 4u)
OMLX_M5_AFFINE_Q5_H1_BM32_LINEAR(omlx_m5_affine_nax_q5_h1_bm32_bns16_sk8, 16u, 8u)
OMLX_M5_AFFINE_Q5_H1_BM32_LINEAR(omlx_m5_affine_nax_q5_h1_bm32_bns32_sk2, 32u, 2u)
OMLX_M5_AFFINE_Q5_H1_BM32_LINEAR(omlx_m5_affine_nax_q5_h1_bm32_bns32_sk4, 32u, 4u)
OMLX_M5_AFFINE_Q5_H1_BM32_SWIGLU(omlx_m5_affine_nax_q5_h1_swiglu_bm32_sk2, 2u)
OMLX_M5_AFFINE_Q5_H1_BM32_SWIGLU(omlx_m5_affine_nax_q5_h1_swiglu_bm32_sk4, 4u)
