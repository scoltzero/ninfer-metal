#include "common_math.h"
#include "fwht64.h"
#include <metal_stdlib>

using namespace metal;

struct AnePackParams {
  uint T;
  uint K;
  uint src_stride;
  uint reserved;
};

struct AneUnpackParams {
  uint T;
  uint ane0_n;
  uint ane1_n;
  uint out_stride;
  uint dst_col_offset;
  uint reserved;
};

struct AneUnpackQuantParams {
  uint T;
  uint mp;
  uint ane0_n;
  uint ane1_n;
  uint out_stride;
  uint dst_col_offset;
  uint reserved0;
  uint reserved1;
};

// Coalesced 32x32 threadgroup-tiled transpose + BF16 -> FP16 pack:
//   src:  [T, src_stride] BF16 (reads columns [0, K))
//   dst0: [1, K, 1, T] FP16 planar IOSurface
//   dst1: optional second die [1, K, 1, T] FP16 planar IOSurface (DualMode=true)
// Dispatch with threads_per_threadgroup = (8, 32, 1),
// threadgroups = (ceil(K / 32), ceil(T / 32), 1).
template <bool DualMode>
inline void ane_pack_bf16_to_fp16_planar_impl(
    device const bfloat *src,
    device half *dst0,
    device half *dst1,
    constant AnePackParams &p,
    threadgroup half tile[32][33],
    uint2 lid,
    uint2 tg) {
  const uint tx = lid.x; // 0..7 (each handles 4 contiguous elements)
  const uint ty = lid.y; // 0..31

  const uint t = tg.y * 32u + ty;
  const uint k0 = tg.x * 32u + tx * 4u;
  const uint stride = p.src_stride != 0u ? p.src_stride : p.K;

  if (t < p.T && k0 + 3u < p.K && ((stride & 3u) == 0u)) {
    const bfloat4 bv = *reinterpret_cast<device const bfloat4 *>(
        src + ulong(t) * stride + k0);
    const half4 hv =
        half4(clamp(float4(bv), float4(-65504.0f), float4(65504.0f)));
    tile[ty][tx * 4u + 0u] = hv.x;
    tile[ty][tx * 4u + 1u] = hv.y;
    tile[ty][tx * 4u + 2u] = hv.z;
    tile[ty][tx * 4u + 3u] = hv.w;
  } else {
    for (uint i = 0u; i < 4u; ++i) {
      half h = half(0.0h);
      if (t < p.T && (k0 + i) < p.K) {
        const float v = float(src[ulong(t) * stride + (k0 + i)]);
        h = half(clamp(v, -65504.0f, 65504.0f));
      }
      tile[ty][tx * 4u + i] = h;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint k = tg.x * 32u + ty;
  const uint t0 = tg.y * 32u + tx * 4u;
  if (k >= p.K) {
    return;
  }

  const half4 out_v = half4(tile[tx * 4u + 0u][ty], tile[tx * 4u + 1u][ty],
                            tile[tx * 4u + 2u][ty], tile[tx * 4u + 3u][ty]);
  if (t0 + 3u < p.T && ((p.T & 3u) == 0u)) {
    *reinterpret_cast<device half4 *>(dst0 + ulong(k) * p.T + t0) = out_v;
    if (DualMode) {
      *reinterpret_cast<device half4 *>(dst1 + ulong(k) * p.T + t0) = out_v;
    }
  } else {
    for (uint i = 0u; i < 4u; ++i) {
      if (t0 + i < p.T) {
        dst0[ulong(k) * p.T + (t0 + i)] = out_v[i];
        if (DualMode) {
          dst1[ulong(k) * p.T + (t0 + i)] = out_v[i];
        }
      }
    }
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_ane_pack_bf16_to_fp16_planar_single(
    device const bfloat *src [[buffer(0)]],
    device half *dst0 [[buffer(1)]],
    constant AnePackParams &p [[buffer(2)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup half tile[32][33];
  ane_pack_bf16_to_fp16_planar_impl<false>(src, dst0, dst0, p, tile, lid, tg);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_ane_pack_bf16_to_fp16_planar_dual(
    device const bfloat *src [[buffer(0)]],
    device half *dst0 [[buffer(1)]],
    device half *dst1 [[buffer(2)]],
    constant AnePackParams &p [[buffer(3)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup half tile[32][33];
  ane_pack_bf16_to_fp16_planar_impl<true>(src, dst0, dst1, p, tile, lid, tg);
}

// Coalesced 32x32 threadgroup-tiled transpose + FP16 -> BF16 unpack:
//   src0: [1, ane0_n, 1, T] FP16 planar IOSurface (ane0: channels [0, ane0_n))
//   src1: [1, ane1_n, 1, T] FP16 planar IOSurface (ane1: channels [ane0_n, ane0_n + ane1_n))
//   dst:  [T, out_stride] BF16 (writes columns [dst_col_offset, dst_col_offset + ane0_n + ane1_n))
// Dispatch with threads_per_threadgroup = (8, 32, 1),
// threadgroups = (ceil((ane0_n + ane1_n) / 32), ceil(T / 32), 1).
template <bool DualMode>
inline void ane_unpack_fp16_planar_to_bf16_impl(
    device const half *src0,
    device const half *src1,
    device bfloat *dst,
    constant AneUnpackParams &p,
    threadgroup half tile[32][33],
    uint2 lid,
    uint2 tg) {
  const uint tx = lid.x; // 0..7
  const uint ty = lid.y; // 0..31

  const uint total_n = DualMode ? (p.ane0_n + p.ane1_n) : p.ane0_n;
  const uint n = tg.x * 32u + ty;
  const uint t0 = tg.y * 32u + tx * 4u;

  if (n < total_n) {
    const bool use_die1 = DualMode && (n >= p.ane0_n);
    device const half *src = use_die1 ? src1 : src0;
    const uint local_n = use_die1 ? (n - p.ane0_n) : n;
    if (t0 + 3u < p.T && ((p.T & 3u) == 0u)) {
      const half4 hv = *reinterpret_cast<device const half4 *>(
          src + ulong(local_n) * p.T + t0);
      tile[ty][tx * 4u + 0u] = hv.x;
      tile[ty][tx * 4u + 1u] = hv.y;
      tile[ty][tx * 4u + 2u] = hv.z;
      tile[ty][tx * 4u + 3u] = hv.w;
    } else {
      for (uint i = 0u; i < 4u; ++i) {
        tile[ty][tx * 4u + i] =
            (t0 + i < p.T) ? src[ulong(local_n) * p.T + (t0 + i)] : half(0.0h);
      }
    }
  } else {
    tile[ty][tx * 4u + 0u] = half(0.0h);
    tile[ty][tx * 4u + 1u] = half(0.0h);
    tile[ty][tx * 4u + 2u] = half(0.0h);
    tile[ty][tx * 4u + 3u] = half(0.0h);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint t = tg.y * 32u + ty;
  const uint n0 = tg.x * 32u + tx * 4u;
  if (t >= p.T || n0 >= total_n) {
    return;
  }

  const float4 fv =
      float4(float(tile[tx * 4u + 0u][ty]), float(tile[tx * 4u + 1u][ty]),
             float(tile[tx * 4u + 2u][ty]), float(tile[tx * 4u + 3u][ty]));
  const bfloat4 bv = bfloat4(fv);
  const ulong row_base = ulong(t) * p.out_stride + p.dst_col_offset;

  if (n0 + 3u < total_n && (((p.out_stride | p.dst_col_offset) & 3u) == 0u)) {
    *reinterpret_cast<device bfloat4 *>(dst + row_base + n0) = bv;
  } else {
    for (uint i = 0u; i < 4u; ++i) {
      if (n0 + i < total_n) {
        dst[row_base + (n0 + i)] = bv[i];
      }
    }
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_ane_unpack_fp16_planar_to_bf16_single(
    device const half *src0 [[buffer(0)]],
    device bfloat *dst [[buffer(1)]],
    constant AneUnpackParams &p [[buffer(2)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup half tile[32][33];
  ane_unpack_fp16_planar_to_bf16_impl<false>(src0, src0, dst, p, tile, lid, tg);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_ane_unpack_fp16_planar_to_bf16_dual(
    device const half *src0 [[buffer(0)]],
    device const half *src1 [[buffer(1)]],
    device bfloat *dst [[buffer(2)]],
    constant AneUnpackParams &p [[buffer(3)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup half tile[32][33];
  ane_unpack_fp16_planar_to_bf16_impl<true>(src0, src1, dst, p, tile, lid, tg);
}

// ============================================================================
// WP 3C Mode B Fused Unpack + FWHT-64 + GS64 INT8 (W4A8)
//   src0:   [1, ane0_n, 1, T] FP16 planar IOSurface (ane0: channels [0, ane0_n))
//   src1:   [1, ane1_n, 1, T] FP16 planar IOSurface (ane1: channels [ane0_n, total_n))
//   qa_out: [mp, out_stride] int8_t (writes cols [dst_col_offset, dst_col_offset + total_n))
//   sa_out: [out_stride / 64, mp] float (group-major: sa_out[g_col * mp + row])
//   ra_out: [out_stride / 64, mp] float (group-major: ra_out[g_col * mp + row], unscaled sum)
// Static TG SRAM: 32 * 65 * 2 = 4,160 B (4.06 KiB <= 4.2 KiB), 128 threads (4 SGs), 0 B spills.
// Dispatch: threadgroups = (total_n / 64, (mp + 31) / 32, 1), threads = (128, 1, 1).
// ============================================================================
template <bool DualMode>
inline void ane_unpack_fp16_planar_to_fwht64_a8_impl(
    device const half *src0,
    device const half *src1,
    device int8_t *qa_out,
    device float *sa_out,
    device float *ra_out,
    constant AneUnpackQuantParams &p,
    threadgroup half tile[32][65],
    uint tid,
    uint2 tg) {
  const uint total_n = DualMode ? (p.ane0_n + p.ane1_n) : p.ane0_n;
  const uint c_base = tg.x * 64u;
  const uint t_base = tg.y * 32u;
  if (c_base >= total_n || t_base >= p.mp) {
    return;
  }

  if (t_base < p.T) {
    const uint tx = tid & 7u;   // 0..7 (4 contiguous tokens per thread)
    const uint cy0 = tid >> 3u; // 0..15 (16 channels per step x 4 steps = 64 ch)
    const uint t0 = t_base + tx * 4u;
    const bool aligned_t = (t0 + 3u < p.T) && ((p.T & 3u) == 0u);
    for (uint step = 0u; step < 4u; ++step) {
      const uint c_local = step * 16u + cy0;
      const uint n = c_base + c_local;
      if (n < total_n) {
        const bool use_die1 = DualMode && (n >= p.ane0_n);
        device const half *src = use_die1 ? src1 : src0;
        const uint local_n = use_die1 ? (n - p.ane0_n) : n;
        if (aligned_t) {
          const half4 hv = *reinterpret_cast<device const half4 *>(
              src + ulong(local_n) * ulong(p.T) + t0);
          tile[tx * 4u + 0u][c_local] = hv.x;
          tile[tx * 4u + 1u][c_local] = hv.y;
          tile[tx * 4u + 2u][c_local] = hv.z;
          tile[tx * 4u + 3u][c_local] = hv.w;
        } else {
          for (uint i = 0u; i < 4u; ++i) {
            tile[tx * 4u + i][c_local] =
                (t0 + i < p.T)
                    ? src[ulong(local_n) * ulong(p.T) + (t0 + i)]
                    : half(0.0h);
          }
        }
      } else {
        tile[tx * 4u + 0u][c_local] = half(0.0h);
        tile[tx * 4u + 1u][c_local] = half(0.0h);
        tile[tx * 4u + 2u][c_local] = half(0.0h);
        tile[tx * 4u + 3u][c_local] = half(0.0h);
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg = tid >> 5u;    // 0..3 (8 tokens per SG)
  const uint lane = tid & 31u;  // 0..31 (2 channels per lane -> 64 channels)
  const uint g_col = (p.dst_col_offset + c_base) / 64u;
  const uint base_t = sg * 8u;

  #pragma clang loop unroll(disable)
  for (uint rr = 0u; rr < 8u; ++rr) {
    const uint t_local = base_t + rr;
    const uint row = t_base + t_local;
    if (row >= p.mp) {
      break;
    }
    const ulong q_off = ulong(row) * ulong(p.out_stride) + ulong(g_col) * 64ul;
    const ulong s_off = ulong(g_col) * ulong(p.mp) + ulong(row);
    if (row < p.T) {
      float v0 = float(tile[t_local][lane * 2u + 0u]);
      float v1 = float(tile[t_local][lane * 2u + 1u]);
      omlx_m5_fwht64_sg32(v0, v1, lane);
      omlx_m5_quantize_pair_g64_f32(
          v0, v1, lane, qa_out + q_off, sa_out + s_off, ra_out + s_off);
    } else {
      qa_out[q_off + lane * 2u + 0u] = int8_t(0);
      qa_out[q_off + lane * 2u + 1u] = int8_t(0);
      if (lane == 0u) {
        sa_out[s_off] = 0.0f;
        ra_out[s_off] = 0.0f;
      }
    }
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_ane_unpack_fp16_planar_to_fwht64_a8_single(
    device const half *src0 [[buffer(0)]],
    device int8_t *qa_out [[buffer(1)]],
    device float *sa_out [[buffer(2)]],
    device float *ra_out [[buffer(3)]],
    constant AneUnpackQuantParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup half tile[32][65];
  static_assert(sizeof(tile) == 4160ul,
                "ane_unpack_fwht64_a8 tile must be 4,160 B <= 4.2 KiB");
  ane_unpack_fp16_planar_to_fwht64_a8_impl<false>(
      src0, src0, qa_out, sa_out, ra_out, p, tile, tid, tg);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_ane_unpack_fp16_planar_to_fwht64_a8_dual(
    device const half *src0 [[buffer(0)]],
    device const half *src1 [[buffer(1)]],
    device int8_t *qa_out [[buffer(2)]],
    device float *sa_out [[buffer(3)]],
    device float *ra_out [[buffer(4)]],
    constant AneUnpackQuantParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup half tile[32][65];
  static_assert(sizeof(tile) == 4160ul,
                "ane_unpack_fwht64_a8 tile must be 4,160 B <= 4.2 KiB");
  ane_unpack_fp16_planar_to_fwht64_a8_impl<true>(
      src0, src1, qa_out, sa_out, ra_out, p, tile, tid, tg);
}

// ============================================================================
// WP 3C Mode B Fused Unpack + FWHT-64 + MXFP4 A4 bm16_bk128 (W4A4)
//   Bit-identical output layout & quantization to
//   `omlx_m5_bf16_fwht64_to_mxfp4_a4_bm16_bk128`:
//     tile_idx = (row / 16) * (out_stride / 128) + blk128
//     dst_codes:  [mp/16, out_stride/128, 16, 64 B] (1,024 B/tile)
//     dst_scales: [mp/16, out_stride/128, 16,  4 B] (   64 B/tile)
// Static TG SRAM: 16 * 132 * 2 = 4,224 B (4.125 KiB <= 4.3 KiB), 128 threads, 0 B spills.
// Dispatch: threadgroups = (total_n / 128, (mp + 15) / 16, 1), threads = (128, 1, 1).
// ============================================================================
template <bool DualMode>
inline void ane_unpack_fp16_planar_to_fwht64_a4_impl(
    device const half *src0,
    device const half *src1,
    device uchar *dst_codes,
    device uchar *dst_scales,
    constant AneUnpackQuantParams &p,
    threadgroup half tile[16][132],
    uint tid,
    uint2 tg) {
  const uint total_n = DualMode ? (p.ane0_n + p.ane1_n) : p.ane0_n;
  const uint c_base = tg.x * 128u;
  const uint t_base = tg.y * 16u;
  if (c_base >= total_n || t_base >= p.mp) {
    return;
  }

  if (t_base < p.T) {
    const uint tx = tid & 3u;   // 0..3 (4 contiguous tokens x 4 = 16 tokens)
    const uint cy0 = tid >> 2u; // 0..31 (32 channels per step x 4 steps = 128 ch)
    const uint t0 = t_base + tx * 4u;
    const bool aligned_t = (t0 + 3u < p.T) && ((p.T & 3u) == 0u);
    for (uint step = 0u; step < 4u; ++step) {
      const uint c_local = step * 32u + cy0;
      const uint n = c_base + c_local;
      if (n < total_n) {
        const bool use_die1 = DualMode && (n >= p.ane0_n);
        device const half *src = use_die1 ? src1 : src0;
        const uint local_n = use_die1 ? (n - p.ane0_n) : n;
        if (aligned_t) {
          const half4 hv = *reinterpret_cast<device const half4 *>(
              src + ulong(local_n) * ulong(p.T) + t0);
          tile[tx * 4u + 0u][c_local] = hv.x;
          tile[tx * 4u + 1u][c_local] = hv.y;
          tile[tx * 4u + 2u][c_local] = hv.z;
          tile[tx * 4u + 3u][c_local] = hv.w;
        } else {
          for (uint i = 0u; i < 4u; ++i) {
            tile[tx * 4u + i][c_local] =
                (t0 + i < p.T)
                    ? src[ulong(local_n) * ulong(p.T) + (t0 + i)]
                    : half(0.0h);
          }
        }
      } else {
        tile[tx * 4u + 0u][c_local] = half(0.0h);
        tile[tx * 4u + 1u][c_local] = half(0.0h);
        tile[tx * 4u + 2u][c_local] = half(0.0h);
        tile[tx * 4u + 3u][c_local] = half(0.0h);
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg = tid >> 5u;   // 0..3 (4 tokens per SG)
  const uint lane = tid & 31u; // 0..31 (2 elements per lane across each half64)
  const uint k_blocks = p.out_stride / 128u;
  const uint blk128 = (p.dst_col_offset + c_base) / 128u;
  const ulong tile_idx = ulong(tg.y) * ulong(k_blocks) + ulong(blk128);
  const uint base_rm = sg * 4u;

  #pragma clang loop unroll(disable)
  for (uint rr = 0u; rr < 4u; ++rr) {
    const uint rm = base_rm + rr;
    const uint row = t_base + rm;
    if (row >= p.mp) {
      break;
    }
    #pragma clang loop unroll(disable)
    for (uint half64 = 0u; half64 < 2u; ++half64) {
      const uint sub32_idx = half64 * 2u + (lane >> 4u);
      device uchar *c_out =
          dst_codes + tile_idx * 1024ul + ulong(rm) * 64ul +
          ulong(sub32_idx) * 16ul;
      device uchar *s_out =
          dst_scales + tile_idx * 64ul + ulong(rm) * 4ul + ulong(sub32_idx);
      if (row >= p.T) {
        c_out[lane & 15u] = 0u;
        if ((lane & 15u) == 0u) {
          *s_out = 0x7fu;
        }
        continue;
      }
      const uint col = half64 * 64u + lane * 2u;
      float v0 = float(tile[rm][col + 0u]);
      float v1 = float(tile[rm][col + 1u]);
      omlx_m5_fwht64_sg32(v0, v1, lane);

      // 16-lane butterfly amax within each 32-element sub-group (lanes 0..15 / 16..31).
      float m = max(fabs(v0), fabs(v1));
      m = max(m, simd_shuffle_xor(m, 1u));
      m = max(m, simd_shuffle_xor(m, 2u));
      m = max(m, simd_shuffle_xor(m, 4u));
      m = max(m, simd_shuffle_xor(m, 8u));

      const uchar sc = nax_encode_e8m0_round_up(m / 6.0f);
      const float scale = nax_decode_e8m0(sc);
      const float inv_scale = scale > 0.0f ? (1.0f / scale) : 0.0f;
      const uchar c0 = nax_encode_e2m1(v0 * inv_scale);
      const uchar c1 = nax_encode_e2m1(v1 * inv_scale);
      c_out[lane & 15u] = uchar(c0 | (c1 << 4u));
      if ((lane & 15u) == 0u) {
        *s_out = sc;
      }
    }
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_ane_unpack_fp16_planar_to_fwht64_a4_single(
    device const half *src0 [[buffer(0)]],
    device uchar *dst_codes [[buffer(1)]],
    device uchar *dst_scales [[buffer(2)]],
    constant AneUnpackQuantParams &p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup half tile[16][132];
  static_assert(sizeof(tile) == 4224ul,
                "ane_unpack_fwht64_a4 tile must be 4,224 B <= 4.3 KiB");
  ane_unpack_fp16_planar_to_fwht64_a4_impl<false>(
      src0, src0, dst_codes, dst_scales, p, tile, tid, tg);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_ane_unpack_fp16_planar_to_fwht64_a4_dual(
    device const half *src0 [[buffer(0)]],
    device const half *src1 [[buffer(1)]],
    device uchar *dst_codes [[buffer(2)]],
    device uchar *dst_scales [[buffer(3)]],
    constant AneUnpackQuantParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup half tile[16][132];
  static_assert(sizeof(tile) == 4224ul,
                "ane_unpack_fwht64_a4 tile must be 4,224 B <= 4.3 KiB");
  ane_unpack_fp16_planar_to_fwht64_a4_impl<true>(
      src0, src1, dst_codes, dst_scales, p, tile, tid, tg);
}
