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
