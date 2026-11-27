#include "common_math.h"
#include "fwht64.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct Mxfp4RepackParams {
  uint n;
  uint k;
  uint src_padded_k;
  uint padded_n;
};

struct Mxfp4RepackSliceParams {
  uint n;
  uint k;
  uint src_padded_k;
  uint dst_row_offset;
  uint dst_padded_n;
};

struct Mxfp4CastParams {
  uint k;
  uint mp;
  uint reserved;
  uint rows;
};

struct Mxfp4NaxParams {
  uint n;
  uint k;
  uint mp;
  uint rows;
};

struct Mxfp4NaxSliceParams {
  uint n;          // slice column count (N or N - row_offset)
  uint k;
  uint mp;
  uint rows;       // word 3 (dynamic token patch)
  uint row_offset; // % 64 == 0 (starting output channel in weight & y)
  uint out_stride; // output row stride in elements (0 => row_offset + n)
};

struct Mxfp4NaxOut64A8Params {
  uint n;          // slice column count (% 64 == 0)
  uint k;
  uint mp;         // a8_mp for sa_out/ra_out stride
  uint rows;       // word 3 (dynamic token patch)
  uint row_offset; // % 64 == 0
  uint out_stride; // full F stride
  uint flags;      // bit 0: also_write_bf16
};

// GPU weight repack: row-major [n, src_padded_k] MXFP4 -> tiled [padded_n/BN, k/128, BN, 128]
template <uint BN>
inline void repack_mxfp4_bk128_impl(
    device const uchar *src_codes,
    device const uchar *src_scales,
    device uchar *dst_codes,
    device uchar *dst_scales,
    constant Mxfp4RepackParams &p,
    uint2 gid) {
  const uint blk = gid.x;
  const uint row = gid.y;
  const uint k_blocks = p.k / 128u;
  if (blk >= k_blocks || row >= p.padded_n) {
    return;
  }
  const ulong tile_idx = ulong(row / BN) * ulong(k_blocks) + ulong(blk);
  const uint rn = row % BN;
  constexpr ulong tile_code_bytes = ulong(BN) * 64ul;
  constexpr ulong tile_scale_bytes = ulong(BN) * 4ul;
  device uint4 *dst_c = reinterpret_cast<device uint4 *>(
      dst_codes + tile_idx * tile_code_bytes + ulong(rn) * 64ul);
  device uint *dst_s = reinterpret_cast<device uint *>(
      dst_scales + tile_idx * tile_scale_bytes + ulong(rn) * 4ul);

  if (row < p.n) {
    device const uint4 *src_c = reinterpret_cast<device const uint4 *>(
        src_codes + ulong(row) * ulong(p.src_padded_k / 2u) + ulong(blk) * 64ul);
    device const uint *src_s = reinterpret_cast<device const uint *>(
        src_scales + ulong(row) * ulong(p.src_padded_k / 32u) + ulong(blk) * 4ul);
    dst_c[0] = src_c[0];
    dst_c[1] = src_c[1];
    dst_c[2] = src_c[2];
    dst_c[3] = src_c[3];
    dst_s[0] = src_s[0];
  } else {
    dst_c[0] = uint4(0u);
    dst_c[1] = uint4(0u);
    dst_c[2] = uint4(0u);
    dst_c[3] = uint4(0u);
    dst_s[0] = 0x7f7f7f7fu;
  }
}

kernel void omlx_m5_repack_mxfp4_bn16_bk128(
    device const uchar *src_codes [[buffer(0)]],
    device const uchar *src_scales [[buffer(1)]],
    device uchar *dst_codes [[buffer(2)]],
    device uchar *dst_scales [[buffer(3)]],
    constant Mxfp4RepackParams &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  repack_mxfp4_bk128_impl<16u>(src_codes, src_scales, dst_codes, dst_scales, p, gid);
}

kernel void omlx_m5_repack_mxfp4_bn32_bk128(
    device const uchar *src_codes [[buffer(0)]],
    device const uchar *src_scales [[buffer(1)]],
    device uchar *dst_codes [[buffer(2)]],
    device uchar *dst_scales [[buffer(3)]],
    constant Mxfp4RepackParams &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  repack_mxfp4_bk128_impl<32u>(src_codes, src_scales, dst_codes, dst_scales, p, gid);
}

template <uint BN>
inline void repack_mxfp4_slice_bk128_impl(
    device const uchar *src_codes,
    device const uchar *src_scales,
    device uchar *dst_codes,
    device uchar *dst_scales,
    constant Mxfp4RepackSliceParams &p,
    uint2 gid) {
  const uint blk = gid.x;
  const uint local_row = gid.y;
  const uint k_blocks = p.k / 128u;
  const uint slice_rows = min(p.dst_padded_n - p.dst_row_offset,
                              ((p.n + BN - 1u) / BN) * BN);
  if (blk >= k_blocks || local_row >= slice_rows) {
    return;
  }
  const uint dst_row = p.dst_row_offset + local_row;
  const ulong tile_idx = ulong(dst_row / BN) * ulong(k_blocks) + ulong(blk);
  const uint rn = dst_row % BN;
  constexpr ulong tile_code_bytes = ulong(BN) * 64ul;
  constexpr ulong tile_scale_bytes = ulong(BN) * 4ul;
  device uint4 *dst_c = reinterpret_cast<device uint4 *>(
      dst_codes + tile_idx * tile_code_bytes + ulong(rn) * 64ul);
  device uint *dst_s = reinterpret_cast<device uint *>(
      dst_scales + tile_idx * tile_scale_bytes + ulong(rn) * 4ul);

  if (local_row < p.n) {
    device const uint4 *src_c = reinterpret_cast<device const uint4 *>(
        src_codes + ulong(local_row) * ulong(p.src_padded_k / 2u) + ulong(blk) * 64ul);
    device const uint *src_s = reinterpret_cast<device const uint *>(
        src_scales + ulong(local_row) * ulong(p.src_padded_k / 32u) + ulong(blk) * 4ul);
    dst_c[0] = src_c[0];
    dst_c[1] = src_c[1];
    dst_c[2] = src_c[2];
    dst_c[3] = src_c[3];
    dst_s[0] = src_s[0];
  } else {
    dst_c[0] = uint4(0u);
    dst_c[1] = uint4(0u);
    dst_c[2] = uint4(0u);
    dst_c[3] = uint4(0u);
    dst_s[0] = 0x7f7f7f7fu;
  }
}

kernel void omlx_m5_repack_mxfp4_slice_bn16_bk128(
    device const uchar *src_codes [[buffer(0)]],
    device const uchar *src_scales [[buffer(1)]],
    device uchar *dst_codes [[buffer(2)]],
    device uchar *dst_scales [[buffer(3)]],
    constant Mxfp4RepackSliceParams &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  repack_mxfp4_slice_bk128_impl<16u>(src_codes, src_scales, dst_codes, dst_scales, p, gid);
}

kernel void omlx_m5_copy_u32_words(
    device const uint *src [[buffer(0)]],
    device uint *dst [[buffer(1)]],
    constant uint &word_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < word_count) {
    dst[gid] = src[gid];
  }
}

// Overflow-proof, strictly row-independent BF16 -> FP16 cast with exact power-of-2 row scale.
// Also zero-pads tail rows [rows, mp) so BM=8 NAX tiles are always finite and in-bounds.
kernel void omlx_m5_bf16_to_f16_rowscale(
    device const bfloat *src [[buffer(0)]],
    device half *dst [[buffer(1)]],
    device float *row_scale [[buffer(2)]],
    constant Mxfp4CastParams &p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  if (row >= p.mp) {
    return;
  }
  const uint k4 = p.k / 4u;
  device half4 *dst_row = reinterpret_cast<device half4 *>(dst + ulong(row) * p.k);

  if (row >= p.rows) {
    for (uint c = tid; c < k4; c += 256u) {
      dst_row[c] = half4(0.0h);
    }
    if (tid == 0) {
      row_scale[row] = 1.0f;
    }
    return;
  }

  device const bfloat4 *src_row =
      reinterpret_cast<device const bfloat4 *>(src + ulong(row) * p.k);

  float local_max = 0.0f;
  for (uint c = tid; c < k4; c += 256u) {
    const float4 v = fabs(float4(src_row[c]));
    local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));
  }
  local_max = simd_max(local_max);

  threadgroup float smax[8];
  threadgroup float inv_scale_tg;
  if (tid == 0) {
    inv_scale_tg = 1.0f;
  }
  if (lane == 0) {
    smax[sg] = local_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    float gmax = smax[0];
    for (uint i = 1; i < 8; ++i) {
      gmax = max(gmax, smax[i]);
    }
    float rs = 1.0f;
    float inv_rs = 1.0f;
    nax_compute_row_scale_f16(gmax, rs, inv_rs);
    row_scale[row] = rs;
    inv_scale_tg = inv_rs;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float inv = inv_scale_tg;
  if (inv == 1.0f) {
    for (uint c = tid; c < k4; c += 256u) {
      dst_row[c] = half4(float4(src_row[c]));
    }
  } else {
    for (uint c = tid; c < k4; c += 256u) {
      dst_row[c] = half4(float4(src_row[c]) * inv);
    }
  }
}

// Fused FWHT-64 + BF16 -> FP16 cast with exact power-of-2 row scale (36 B TG SRAM).
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_bf16_fwht64_to_f16_rowscale(
    device const bfloat *src [[buffer(0)]],
    device half *dst [[buffer(1)]],
    device float *row_scale [[buffer(2)]],
    constant Mxfp4CastParams &p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  if (row >= p.mp) {
    return;
  }
  const uint k4 = p.k / 4u;
  device half4 *dst_row4 = reinterpret_cast<device half4 *>(dst + ulong(row) * p.k);

  if (row >= p.rows) {
    for (uint c = tid; c < k4; c += 256u) {
      dst_row4[c] = half4(0.0h);
    }
    if (tid == 0u) {
      row_scale[row] = 1.0f;
    }
    return;
  }

  const uint groups = p.k / 64u;
  device const bfloat *src_row = src + ulong(row) * p.k;
  device half *dst_row = dst + ulong(row) * p.k;

  float local_max = 0.0f;
  for (uint g = sg; g < groups; g += 8u) {
    const bfloat2 x2 = reinterpret_cast<device const bfloat2 *>(src_row + ulong(g) * 64ul)[lane];
    float v0 = float(x2.x);
    float v1 = float(x2.y);
    omlx_m5_fwht64_sg32(v0, v1, lane);
    local_max = max(local_max, max(fabs(v0), fabs(v1)));
  }
  local_max = simd_max(local_max);

  threadgroup float smax[9];
  if (lane == 0u) {
    smax[sg] = local_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    float gmax = smax[0];
    for (uint i = 1u; i < 8u; ++i) {
      gmax = max(gmax, smax[i]);
    }
    float rs = 1.0f;
    float inv_rs = 1.0f;
    nax_compute_row_scale_f16(gmax, rs, inv_rs);
    row_scale[row] = rs;
    smax[8] = inv_rs;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float inv = smax[8];
  for (uint g = sg; g < groups; g += 8u) {
    const bfloat2 x2 = reinterpret_cast<device const bfloat2 *>(src_row + ulong(g) * 64ul)[lane];
    float v0 = float(x2.x);
    float v1 = float(x2.y);
    omlx_m5_fwht64_sg32(v0, v1, lane);
    reinterpret_cast<device half2 *>(dst_row + ulong(g) * 64ul)[lane] =
        half2(half(v0 * inv), half(v1 * inv));
  }
}

using Mxfp4ScaleTag =
    tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>;
using Mxfp4WeightTensor =
    tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline, Mxfp4ScaleTag>;

template <uint SK, uint kCap, typename CoopT>
inline void mxfp4_reduce_store_linear_slice(
    thread CoopT &prod,
    uint rb_slice,
    uint n0,
    bool col_active,
    uint sg_k,
    uint lane,
    threadgroup float *my_parts,
    device const float *row_scale,
    device bfloat *y,
    constant Mxfp4NaxParams &p,
    bool need_pre_barrier) {
  const uint elems = prod.get_capacity();
  if constexpr (SK > 1u) {
    if (need_pre_barrier) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg_k > 0u && col_active) {
      for (uint i = 0; i < elems; ++i) {
        my_parts[((sg_k - 1u) * kCap + i) * 32u + lane] = prod[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_k == 0u && col_active) {
      for (uint s = 1u; s < SK; ++s) {
        for (uint i = 0; i < elems; ++i) {
          prod[i] += my_parts[((s - 1u) * kCap + i) * 32u + lane];
        }
      }
    }
  }
  if (sg_k == 0u && col_active) {
    for (uint i = 0; i < elems; ++i) {
      if (prod.is_valid_element(i)) {
        const auto idx = prod.get_multidimensional_index(i);
        const uint col = n0 + uint(idx[0]);
        const uint row = rb_slice + uint(idx[1]);
        if (row < p.rows && col < p.n) {
          y[ulong(row) * p.n + col] = bfloat(prod[i] * row_scale[row]);
        }
      }
    }
  }
}

template <uint BN, uint SK, uint RF = 1u, uint NSG = 1u>
inline void mxfp4_nax_linear_impl(
    device const half *x,
    device const float *row_scale,
    device const uchar *codes,
    device const uchar *scales,
    device bfloat *y,
    constant Mxfp4NaxParams &p,
    threadgroup float *parts,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM = 8u;
  constexpr uint BK = 128u;
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

  const uint blocks = p.k / BK;
  const uint b0 = (sg_k * blocks) / SK;
  const uint b1 = ((sg_k + 1u) * blocks) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  tensor<device half, dextents<int32_t, 2>, tensor_inline> a_full(
      (device half *)x, dextents<int32_t, 2>(p.k, p.mp));

  constexpr ulong tile_code_bytes = ulong(BN) * (BK / 2u);
  constexpr ulong tile_scale_bytes = ulong(BN) * (BK / 32u);
  const ulong tile_base_blk = ulong(tile_n) * blocks;

  auto a_init = a_full.slice(0, rb);
  Mxfp4WeightTensor b_init((device uchar *)codes, dextents<int32_t, 2>(BK, BN),
                           Mxfp4ScaleTag((device uchar *)scales));
  auto prod0 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto prod1 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto prod2 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto prod3 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto prod4 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto prod5 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto prod6 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto prod7 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  const uint elems = prod0.get_capacity();
  for (uint i = 0; i < elems; ++i) {
    prod0[i] = 0.0f;
    if constexpr (RF > 1u) prod1[i] = 0.0f;
    if constexpr (RF > 2u) prod2[i] = 0.0f;
    if constexpr (RF > 3u) prod3[i] = 0.0f;
    if constexpr (RF > 4u) prod4[i] = 0.0f;
    if constexpr (RF > 5u) prod5[i] = 0.0f;
    if constexpr (RF > 6u) prod6[i] = 0.0f;
    if constexpr (RF > 7u) prod7[i] = 0.0f;
  }

  if (col_active) {
    for (uint blk = b0; blk < b1; ++blk) {
      const uint k = blk * BK;
      const ulong idx = tile_base_blk + blk;
      Mxfp4WeightTensor b((device uchar *)(codes + idx * tile_code_bytes),
                          dextents<int32_t, 2>(BK, BN),
                          Mxfp4ScaleTag((device uchar *)(scales + idx * tile_scale_bytes)));
      auto a0 = a_full.slice(k, rb);
      op.run(a0, b, prod0);
      if constexpr (RF > 1u) { if (rb + 8u < p.rows) { auto a1 = a_full.slice(k, rb + 8u); op.run(a1, b, prod1); } }
      if constexpr (RF > 2u) { if (rb + 16u < p.rows) { auto a2 = a_full.slice(k, rb + 16u); op.run(a2, b, prod2); } }
      if constexpr (RF > 3u) { if (rb + 24u < p.rows) { auto a3 = a_full.slice(k, rb + 24u); op.run(a3, b, prod3); } }
      if constexpr (RF > 4u) { if (rb + 32u < p.rows) { auto a4 = a_full.slice(k, rb + 32u); op.run(a4, b, prod4); } }
      if constexpr (RF > 5u) { if (rb + 40u < p.rows) { auto a5 = a_full.slice(k, rb + 40u); op.run(a5, b, prod5); } }
      if constexpr (RF > 6u) { if (rb + 48u < p.rows) { auto a6 = a_full.slice(k, rb + 48u); op.run(a6, b, prod6); } }
      if constexpr (RF > 7u) { if (rb + 56u < p.rows) { auto a7 = a_full.slice(k, rb + 56u); op.run(a7, b, prod7); } }
    }
  }

  constexpr uint kPartStride = (SK > 1u ? (SK - 1u) : 1u) * kCap * 32u;
  threadgroup float *my_parts = parts + ((NSG > 1u && sg_n < NSG) ? (sg_n * kPartStride) : 0u);

  mxfp4_reduce_store_linear_slice<SK, kCap>(
      prod0, rb, n0, col_active, sg_k, lane, my_parts, row_scale, y, p, false);
  if constexpr (RF > 1u) {
    if (rb + 8u < p.rows) {
      mxfp4_reduce_store_linear_slice<SK, kCap>(
          prod1, rb + 8u, n0, col_active, sg_k, lane, my_parts, row_scale, y, p, true);
    }
  }
  if constexpr (RF > 2u) {
    if (rb + 16u < p.rows) {
      mxfp4_reduce_store_linear_slice<SK, kCap>(
          prod2, rb + 16u, n0, col_active, sg_k, lane, my_parts, row_scale, y, p, true);
    }
  }
  if constexpr (RF > 3u) {
    if (rb + 24u < p.rows) {
      mxfp4_reduce_store_linear_slice<SK, kCap>(
          prod3, rb + 24u, n0, col_active, sg_k, lane, my_parts, row_scale, y, p, true);
    }
  }
  if constexpr (RF > 4u) {
    if (rb + 32u < p.rows) {
      mxfp4_reduce_store_linear_slice<SK, kCap>(
          prod4, rb + 32u, n0, col_active, sg_k, lane, my_parts, row_scale, y, p, true);
    }
  }
  if constexpr (RF > 5u) {
    if (rb + 40u < p.rows) {
      mxfp4_reduce_store_linear_slice<SK, kCap>(
          prod5, rb + 40u, n0, col_active, sg_k, lane, my_parts, row_scale, y, p, true);
    }
  }
  if constexpr (RF > 6u) {
    if (rb + 48u < p.rows) {
      mxfp4_reduce_store_linear_slice<SK, kCap>(
          prod6, rb + 48u, n0, col_active, sg_k, lane, my_parts, row_scale, y, p, true);
    }
  }
  if constexpr (RF > 7u) {
    if (rb + 56u < p.rows) {
      mxfp4_reduce_store_linear_slice<SK, kCap>(
          prod7, rb + 56u, n0, col_active, sg_k, lane, my_parts, row_scale, y, p, true);
    }
  }
}

template <uint SK, uint kCap, bool OutF16, typename CoopT>
inline void mxfp4_reduce_store_swiglu_slice(
    thread CoopT &pg,
    thread CoopT &pu,
    uint rb_slice,
    uint n0,
    bool col_active,
    uint sg_k,
    uint lane,
    threadgroup float *my_parts,
    device const float *row_scale,
    device bfloat *y_bf16,
    device half *y_f16,
    constant Mxfp4NaxParams &p,
    bool need_pre_barrier) {
  const uint elems = pg.get_capacity();
  if constexpr (SK > 1u) {
    if (need_pre_barrier) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg_k > 0u && col_active) {
      for (uint i = 0; i < elems; ++i) {
        my_parts[((sg_k - 1u) * (2u * kCap) + i) * 32u + lane] = pg[i];
        my_parts[((sg_k - 1u) * (2u * kCap) + kCap + i) * 32u + lane] = pu[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_k == 0u && col_active) {
      for (uint s = 1u; s < SK; ++s) {
        for (uint i = 0; i < elems; ++i) {
          pg[i] += my_parts[((s - 1u) * (2u * kCap) + i) * 32u + lane];
          pu[i] += my_parts[((s - 1u) * (2u * kCap) + kCap + i) * 32u + lane];
        }
      }
    }
  }
  if (sg_k == 0u && col_active) {
    for (uint i = 0; i < elems; ++i) {
      if (pg.is_valid_element(i)) {
        const auto idx = pg.get_multidimensional_index(i);
        const uint col = n0 + uint(idx[0]);
        const uint row = rb_slice + uint(idx[1]);
        if constexpr (OutF16) {
          if (row < p.mp && col < p.n) {
            if (row < p.rows) {
              const float rs = row_scale[row];
              const float gv = float(bfloat(pg[i] * rs));
              const float uv = float(bfloat(pu[i] * rs));
              const float ov = float(bfloat(nax_silu_f32(gv) * uv));
              y_f16[ulong(row) * p.n + col] = half(clamp(ov, -65504.0f, 65504.0f));
            } else {
              y_f16[ulong(row) * p.n + col] = half(0.0h);
            }
          }
        } else {
          if (row < p.rows && col < p.n) {
            const float rs = row_scale[row];
            const float gv = float(bfloat(pg[i] * rs));
            const float uv = float(bfloat(pu[i] * rs));
            y_bf16[ulong(row) * p.n + col] = bfloat(nax_silu_f32(gv) * uv);
          }
        }
      }
    }
  }
}

template <uint BN, uint SK, bool OutF16, uint RF = 1u, uint NSG = 1u>
inline void mxfp4_nax_swiglu_impl(
    device const half *x,
    device const float *row_scale,
    device const uchar *g_codes,
    device const uchar *g_scales,
    device const uchar *u_codes,
    device const uchar *u_scales,
    device bfloat *y_bf16,
    device half *y_f16,
    constant Mxfp4NaxParams &p,
    threadgroup float *parts,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM = 8u;
  constexpr uint BK = 128u;
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

  const uint blocks = p.k / BK;
  const uint b0 = (sg_k * blocks) / SK;
  const uint b1 = ((sg_k + 1u) * blocks) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  tensor<device half, dextents<int32_t, 2>, tensor_inline> a_full(
      (device half *)x, dextents<int32_t, 2>(p.k, p.mp));

  constexpr ulong tile_code_bytes = ulong(BN) * (BK / 2u);
  constexpr ulong tile_scale_bytes = ulong(BN) * (BK / 32u);
  const ulong tile_base_blk = ulong(tile_n) * blocks;

  auto a_init = a_full.slice(0, rb);
  Mxfp4WeightTensor b_init((device uchar *)g_codes, dextents<int32_t, 2>(BK, BN),
                           Mxfp4ScaleTag((device uchar *)g_scales));
  auto pg0 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu0 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pg1 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu1 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pg2 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu2 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pg3 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu3 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pg4 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu4 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pg5 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu5 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pg6 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu6 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pg7 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu7 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  const uint elems = pg0.get_capacity();
  for (uint i = 0; i < elems; ++i) {
    pg0[i] = 0.0f; pu0[i] = 0.0f;
    if constexpr (RF > 1u) { pg1[i] = 0.0f; pu1[i] = 0.0f; }
    if constexpr (RF > 2u) { pg2[i] = 0.0f; pu2[i] = 0.0f; }
    if constexpr (RF > 3u) { pg3[i] = 0.0f; pu3[i] = 0.0f; }
    if constexpr (RF > 4u) { pg4[i] = 0.0f; pu4[i] = 0.0f; }
    if constexpr (RF > 5u) { pg5[i] = 0.0f; pu5[i] = 0.0f; }
    if constexpr (RF > 6u) { pg6[i] = 0.0f; pu6[i] = 0.0f; }
    if constexpr (RF > 7u) { pg7[i] = 0.0f; pu7[i] = 0.0f; }
  }

  if (col_active) {
    for (uint blk = b0; blk < b1; ++blk) {
      const uint k = blk * BK;
      const ulong idx = tile_base_blk + blk;
      Mxfp4WeightTensor bg((device uchar *)(g_codes + idx * tile_code_bytes),
                           dextents<int32_t, 2>(BK, BN),
                           Mxfp4ScaleTag((device uchar *)(g_scales + idx * tile_scale_bytes)));
      Mxfp4WeightTensor bu((device uchar *)(u_codes + idx * tile_code_bytes),
                           dextents<int32_t, 2>(BK, BN),
                           Mxfp4ScaleTag((device uchar *)(u_scales + idx * tile_scale_bytes)));
      auto a0 = a_full.slice(k, rb);
      op.run(a0, bg, pg0);
      op.run(a0, bu, pu0);
      if constexpr (RF > 1u) { if (rb + 8u < p.rows) { auto a1 = a_full.slice(k, rb + 8u); op.run(a1, bg, pg1); op.run(a1, bu, pu1); } }
      if constexpr (RF > 2u) { if (rb + 16u < p.rows) { auto a2 = a_full.slice(k, rb + 16u); op.run(a2, bg, pg2); op.run(a2, bu, pu2); } }
      if constexpr (RF > 3u) { if (rb + 24u < p.rows) { auto a3 = a_full.slice(k, rb + 24u); op.run(a3, bg, pg3); op.run(a3, bu, pu3); } }
      if constexpr (RF > 4u) { if (rb + 32u < p.rows) { auto a4 = a_full.slice(k, rb + 32u); op.run(a4, bg, pg4); op.run(a4, bu, pu4); } }
      if constexpr (RF > 5u) { if (rb + 40u < p.rows) { auto a5 = a_full.slice(k, rb + 40u); op.run(a5, bg, pg5); op.run(a5, bu, pu5); } }
      if constexpr (RF > 6u) { if (rb + 48u < p.rows) { auto a6 = a_full.slice(k, rb + 48u); op.run(a6, bg, pg6); op.run(a6, bu, pu6); } }
      if constexpr (RF > 7u) { if (rb + 56u < p.rows) { auto a7 = a_full.slice(k, rb + 56u); op.run(a7, bg, pg7); op.run(a7, bu, pu7); } }
    }
  }

  constexpr uint kPartStride = (SK > 1u ? (SK - 1u) : 1u) * 2u * kCap * 32u;
  threadgroup float *my_parts = parts + ((NSG > 1u && sg_n < NSG) ? (sg_n * kPartStride) : 0u);

  mxfp4_reduce_store_swiglu_slice<SK, kCap, OutF16>(
      pg0, pu0, rb, n0, col_active, sg_k, lane, my_parts, row_scale, y_bf16, y_f16, p, false);
  if constexpr (RF > 1u) {
    if (rb + 8u < p.rows) {
      mxfp4_reduce_store_swiglu_slice<SK, kCap, OutF16>(
          pg1, pu1, rb + 8u, n0, col_active, sg_k, lane, my_parts, row_scale, y_bf16, y_f16, p, true);
    }
  }
  if constexpr (RF > 2u) {
    if (rb + 16u < p.rows) {
      mxfp4_reduce_store_swiglu_slice<SK, kCap, OutF16>(
          pg2, pu2, rb + 16u, n0, col_active, sg_k, lane, my_parts, row_scale, y_bf16, y_f16, p, true);
    }
  }
  if constexpr (RF > 3u) {
    if (rb + 24u < p.rows) {
      mxfp4_reduce_store_swiglu_slice<SK, kCap, OutF16>(
          pg3, pu3, rb + 24u, n0, col_active, sg_k, lane, my_parts, row_scale, y_bf16, y_f16, p, true);
    }
  }
  if constexpr (RF > 4u) {
    if (rb + 32u < p.rows) {
      mxfp4_reduce_store_swiglu_slice<SK, kCap, OutF16>(
          pg4, pu4, rb + 32u, n0, col_active, sg_k, lane, my_parts, row_scale, y_bf16, y_f16, p, true);
    }
  }
  if constexpr (RF > 5u) {
    if (rb + 40u < p.rows) {
      mxfp4_reduce_store_swiglu_slice<SK, kCap, OutF16>(
          pg5, pu5, rb + 40u, n0, col_active, sg_k, lane, my_parts, row_scale, y_bf16, y_f16, p, true);
    }
  }
  if constexpr (RF > 6u) {
    if (rb + 48u < p.rows) {
      mxfp4_reduce_store_swiglu_slice<SK, kCap, OutF16>(
          pg6, pu6, rb + 48u, n0, col_active, sg_k, lane, my_parts, row_scale, y_bf16, y_f16, p, true);
    }
  }
  if constexpr (RF > 7u) {
    if (rb + 56u < p.rows) {
      mxfp4_reduce_store_swiglu_slice<SK, kCap, OutF16>(
          pg7, pu7, rb + 56u, n0, col_active, sg_k, lane, my_parts, row_scale, y_bf16, y_f16, p, true);
    }
  }
}

#define OMLX_M5_MXFP4_NAX_N16(SK)                                              \
  kernel void omlx_m5_mxfp4_nax_n16_sk##SK(                                    \
      device const half *x [[buffer(0)]],                                      \
      device const float *row_scale [[buffer(1)]],                             \
      device const uchar *codes [[buffer(2)]],                                 \
      device const uchar *scales [[buffer(3)]],                                \
      device bfloat *y [[buffer(4)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(5)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 8u * 32u];              \
    mxfp4_nax_linear_impl<16u, SK, 1u, 1u>(x, row_scale, codes, scales, y, p,  \
                                           parts, lane, sg, tg);               \
  }                                                                            \
  kernel void omlx_m5_mxfp4_nax_n16_sk##SK##_rf1_nsg2(                         \
      device const half *x [[buffer(0)]],                                      \
      device const float *row_scale [[buffer(1)]],                             \
      device const uchar *codes [[buffer(2)]],                                 \
      device const uchar *scales [[buffer(3)]],                                \
      device bfloat *y [[buffer(4)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(5)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[2u * (SK > 1u ? SK - 1u : 1u) * 8u * 32u];         \
    mxfp4_nax_linear_impl<16u, SK, 1u, 2u>(x, row_scale, codes, scales, y, p,  \
                                           parts, lane, sg, tg);               \
  }                                                                            \
  kernel void omlx_m5_mxfp4_nax_swiglu_n16_sk##SK(                             \
      device const half *x [[buffer(0)]],                                      \
      device const float *row_scale [[buffer(1)]],                             \
      device const uchar *g_codes [[buffer(2)]],                               \
      device const uchar *g_scales [[buffer(3)]],                              \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_scales [[buffer(5)]],                              \
      device bfloat *y [[buffer(6)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(7)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 16u * 32u];             \
    mxfp4_nax_swiglu_impl<16u, SK, false, 1u, 1u>(x, row_scale, g_codes,       \
                                                  g_scales, u_codes, u_scales, \
                                                  y, nullptr, p, parts, lane,  \
                                                  sg, tg);                     \
  }                                                                            \
  kernel void omlx_m5_mxfp4_nax_swiglu_f16_n16_sk##SK(                         \
      device const half *x [[buffer(0)]],                                      \
      device const float *row_scale [[buffer(1)]],                             \
      device const uchar *g_codes [[buffer(2)]],                               \
      device const uchar *g_scales [[buffer(3)]],                              \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_scales [[buffer(5)]],                              \
      device half *y_f16 [[buffer(6)]],                                        \
      constant Mxfp4NaxParams &p [[buffer(7)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 16u * 32u];             \
    mxfp4_nax_swiglu_impl<16u, SK, true, 1u, 1u>(x, row_scale, g_codes,        \
                                                 g_scales, u_codes, u_scales,  \
                                                 nullptr, y_f16, p, parts,     \
                                                 lane, sg, tg);                \
  }

#define OMLX_M5_MXFP4_NAX_N16_RF(SK, RF)                                       \
  kernel void omlx_m5_mxfp4_nax_n16_sk##SK##_rf##RF##_nsg1(                    \
      device const half *x [[buffer(0)]],                                      \
      device const float *row_scale [[buffer(1)]],                             \
      device const uchar *codes [[buffer(2)]],                                 \
      device const uchar *scales [[buffer(3)]],                                \
      device bfloat *y [[buffer(4)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(5)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 8u * 32u];              \
    mxfp4_nax_linear_impl<16u, SK, RF, 1u>(x, row_scale, codes, scales, y, p,  \
                                           parts, lane, sg, tg);               \
  }                                                                            \
  kernel void omlx_m5_mxfp4_nax_n16_sk##SK##_rf##RF##_nsg2(                    \
      device const half *x [[buffer(0)]],                                      \
      device const float *row_scale [[buffer(1)]],                             \
      device const uchar *codes [[buffer(2)]],                                 \
      device const uchar *scales [[buffer(3)]],                                \
      device bfloat *y [[buffer(4)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(5)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[2u * (SK > 1u ? SK - 1u : 1u) * 8u * 32u];         \
    mxfp4_nax_linear_impl<16u, SK, RF, 2u>(x, row_scale, codes, scales, y, p,  \
                                           parts, lane, sg, tg);               \
  }                                                                            \
  kernel void omlx_m5_mxfp4_nax_swiglu_n16_sk##SK##_rf##RF(                    \
      device const half *x [[buffer(0)]],                                      \
      device const float *row_scale [[buffer(1)]],                             \
      device const uchar *g_codes [[buffer(2)]],                               \
      device const uchar *g_scales [[buffer(3)]],                              \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_scales [[buffer(5)]],                              \
      device bfloat *y [[buffer(6)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(7)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 16u * 32u];             \
    mxfp4_nax_swiglu_impl<16u, SK, false, RF, 1u>(x, row_scale, g_codes,       \
                                                  g_scales, u_codes, u_scales, \
                                                  y, nullptr, p, parts, lane,  \
                                                  sg, tg);                     \
  }                                                                            \
  kernel void omlx_m5_mxfp4_nax_swiglu_f16_n16_sk##SK##_rf##RF(                \
      device const half *x [[buffer(0)]],                                      \
      device const float *row_scale [[buffer(1)]],                             \
      device const uchar *g_codes [[buffer(2)]],                               \
      device const uchar *g_scales [[buffer(3)]],                              \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_scales [[buffer(5)]],                              \
      device half *y_f16 [[buffer(6)]],                                        \
      constant Mxfp4NaxParams &p [[buffer(7)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 16u * 32u];             \
    mxfp4_nax_swiglu_impl<16u, SK, true, RF, 1u>(x, row_scale, g_codes,        \
                                                 g_scales, u_codes, u_scales,  \
                                                 nullptr, y_f16, p, parts,     \
                                                 lane, sg, tg);                \
  }

#define OMLX_M5_MXFP4_NAX_N16_ALL_RF(SK)                                       \
  OMLX_M5_MXFP4_NAX_N16(SK)                                                    \
  OMLX_M5_MXFP4_NAX_N16_RF(SK, 2)                                              \
  OMLX_M5_MXFP4_NAX_N16_RF(SK, 3)                                              \
  OMLX_M5_MXFP4_NAX_N16_RF(SK, 4)                                              \
  OMLX_M5_MXFP4_NAX_N16_RF(SK, 6)                                              \
  OMLX_M5_MXFP4_NAX_N16_RF(SK, 8)

#define OMLX_M5_MXFP4_NAX_N32(SK)                                              \
  kernel void omlx_m5_mxfp4_nax_n32_sk##SK(                                    \
      device const half *x [[buffer(0)]],                                      \
      device const float *row_scale [[buffer(1)]],                             \
      device const uchar *codes [[buffer(2)]],                                 \
      device const uchar *scales [[buffer(3)]],                                \
      device bfloat *y [[buffer(4)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(5)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 16u * 32u];             \
    mxfp4_nax_linear_impl<32u, SK, 1u, 1u>(x, row_scale, codes, scales, y, p,  \
                                           parts, lane, sg, tg);               \
  }

#define OMLX_M5_MXFP4_NAX_N32_RF(SK, RF)                                       \
  kernel void omlx_m5_mxfp4_nax_n32_sk##SK##_rf##RF(                           \
      device const half *x [[buffer(0)]],                                      \
      device const float *row_scale [[buffer(1)]],                             \
      device const uchar *codes [[buffer(2)]],                                 \
      device const uchar *scales [[buffer(3)]],                                \
      device bfloat *y [[buffer(4)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(5)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 16u * 32u];             \
    mxfp4_nax_linear_impl<32u, SK, RF, 1u>(x, row_scale, codes, scales, y, p,  \
                                           parts, lane, sg, tg);               \
  }

#define OMLX_M5_MXFP4_NAX_N32_ALL_RF(SK)                                       \
  OMLX_M5_MXFP4_NAX_N32(SK)                                                    \
  OMLX_M5_MXFP4_NAX_N32_RF(SK, 2)                                              \
  OMLX_M5_MXFP4_NAX_N32_RF(SK, 3)                                              \
  OMLX_M5_MXFP4_NAX_N32_RF(SK, 4)                                              \
  OMLX_M5_MXFP4_NAX_N32_RF(SK, 6)                                              \
  OMLX_M5_MXFP4_NAX_N32_RF(SK, 8)

OMLX_M5_MXFP4_NAX_N16_ALL_RF(1)
OMLX_M5_MXFP4_NAX_N16_ALL_RF(2)
OMLX_M5_MXFP4_NAX_N16_ALL_RF(4)
OMLX_M5_MXFP4_NAX_N16_ALL_RF(8)
OMLX_M5_MXFP4_NAX_N32_ALL_RF(1)
OMLX_M5_MXFP4_NAX_N32_ALL_RF(2)
OMLX_M5_MXFP4_NAX_N32_ALL_RF(4)
OMLX_M5_MXFP4_NAX_N32_ALL_RF(8)

// ============================================================================
// v0.8.0-m5rt.14 0-ULP Hardware NAX BM=16 & BM=32, BN=16/32, SK=1..8
// Proven on M5 Ultra cold-DRAM micro-probe: 1.69x-1.74x at M=16 (BM=16 SK=8),
// 2.12x-2.16x at M=32 (BM=32 SK=8, SwiGLU 28 KiB <= 32 KiB TG SRAM), strict
// 0-ULP bit-identical to BM=8 SK=8 across Linear, Concat, SwiGLU, and lm_head!
// ============================================================================

template <uint BM, uint BN, uint SK>
inline void mxfp4_nax_linear_bm_impl(
    device const half *x,
    device const float *row_scale,
    device const uchar *codes,
    device const uchar *scales,
    device bfloat *y,
    constant Mxfp4NaxParams &p,
    threadgroup float *parts,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BK = 128u;
  constexpr uint kCap = (BM * BN) / 32u;
  const uint rb = tg.y * BM;
  const uint tile_n = tg.x;
  const uint n0 = tile_n * BN;
  if (rb >= p.rows || n0 >= p.n) {
    return;
  }

  const uint blocks = p.k / BK;
  const uint b0 = (sg * blocks) / SK;
  const uint b1 = ((sg + 1u) * blocks) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  tensor<device half, dextents<int32_t, 2>, tensor_inline> a_full(
      (device half *)x, dextents<int32_t, 2>(p.k, p.mp));

  constexpr ulong tile_code_bytes = ulong(BN) * (BK / 2u);
  constexpr ulong tile_scale_bytes = ulong(BN) * (BK / 32u);
  const ulong tile_base_blk = ulong(tile_n) * blocks;

  auto a_init = a_full.slice(0, rb);
  Mxfp4WeightTensor b_init((device uchar *)codes, dextents<int32_t, 2>(BK, BN),
                           Mxfp4ScaleTag((device uchar *)scales));
  auto prod = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  const uint elems = prod.get_capacity();
  for (uint i = 0u; i < elems; ++i) {
    prod[i] = 0.0f;
  }

  for (uint blk = b0; blk < b1; ++blk) {
    const uint k = blk * BK;
    const ulong idx = tile_base_blk + blk;
    Mxfp4WeightTensor b((device uchar *)(codes + idx * tile_code_bytes),
                        dextents<int32_t, 2>(BK, BN),
                        Mxfp4ScaleTag((device uchar *)(scales + idx * tile_scale_bytes)));
    auto a = a_full.slice(k, rb);
    op.run(a, b, prod);
  }

  mxfp4_reduce_store_linear_slice<SK, kCap>(
      prod, rb, n0, true, sg, lane, parts, row_scale, y, p, false);
}

template <uint BM, uint BN, uint SK, bool OutF16>
inline void mxfp4_nax_swiglu_bm_impl(
    device const half *x,
    device const float *row_scale,
    device const uchar *g_codes,
    device const uchar *g_scales,
    device const uchar *u_codes,
    device const uchar *u_scales,
    device bfloat *y_bf16,
    device half *y_f16,
    constant Mxfp4NaxParams &p,
    threadgroup float *parts,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BK = 128u;
  constexpr uint kCap = (BM * BN) / 32u;
  const uint rb = tg.y * BM;
  const uint tile_n = tg.x;
  const uint n0 = tile_n * BN;
  if (rb >= p.rows || n0 >= p.n) {
    return;
  }

  const uint blocks = p.k / BK;
  const uint b0 = (sg * blocks) / SK;
  const uint b1 = ((sg + 1u) * blocks) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  tensor<device half, dextents<int32_t, 2>, tensor_inline> a_full(
      (device half *)x, dextents<int32_t, 2>(p.k, p.mp));

  constexpr ulong tile_code_bytes = ulong(BN) * (BK / 2u);
  constexpr ulong tile_scale_bytes = ulong(BN) * (BK / 32u);
  const ulong tile_base_blk = ulong(tile_n) * blocks;

  auto a_init = a_full.slice(0, rb);
  Mxfp4WeightTensor b_init((device uchar *)g_codes, dextents<int32_t, 2>(BK, BN),
                           Mxfp4ScaleTag((device uchar *)g_scales));
  auto pg = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  const uint elems = pg.get_capacity();
  for (uint i = 0u; i < elems; ++i) {
    pg[i] = 0.0f;
    pu[i] = 0.0f;
  }

  for (uint blk = b0; blk < b1; ++blk) {
    const uint k = blk * BK;
    const ulong idx = tile_base_blk + blk;
    Mxfp4WeightTensor bg((device uchar *)(g_codes + idx * tile_code_bytes),
                         dextents<int32_t, 2>(BK, BN),
                         Mxfp4ScaleTag((device uchar *)(g_scales + idx * tile_scale_bytes)));
    Mxfp4WeightTensor bu((device uchar *)(u_codes + idx * tile_code_bytes),
                         dextents<int32_t, 2>(BK, BN),
                         Mxfp4ScaleTag((device uchar *)(u_scales + idx * tile_scale_bytes)));
    auto a = a_full.slice(k, rb);
    op.run(a, bg, pg);
    op.run(a, bu, pu);
  }

  mxfp4_reduce_store_swiglu_slice<SK, kCap, OutF16>(
      pg, pu, rb, n0, true, sg, lane, parts, row_scale, y_bf16, y_f16, p, false);
}

#define OMLX_M5_MXFP4_NAX_BM_N16_LINEAR(BM, SK)                                \
  [[max_total_threads_per_threadgroup((SK) * 32u)]]                            \
  kernel void omlx_m5_mxfp4_nax_bm##BM##_n16_sk##SK(                           \
      device const half *x [[buffer(0)]],                                      \
      device const float *row_scale [[buffer(1)]],                             \
      device const uchar *codes [[buffer(2)]],                                 \
      device const uchar *scales [[buffer(3)]],                                \
      device bfloat *y [[buffer(4)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(5)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * ((BM * 16u) / 32u) * 32u]; \
    static_assert(sizeof(parts) <= kApple10MaxPrefillTgSramBytes,              \
                  "mxfp4 bm_n16_linear parts exceeds 16.0 KiB");               \
    mxfp4_nax_linear_bm_impl<BM##u, 16u, SK>(                                  \
        x, row_scale, codes, scales, y, p, parts, lane, sg, tg);               \
  }

#define OMLX_M5_MXFP4_NAX_BM_N16_SWIGLU(BM, SK)                                \
  [[max_total_threads_per_threadgroup((SK) * 32u)]]                            \
  kernel void omlx_m5_mxfp4_nax_swiglu_bm##BM##_n16_sk##SK(                    \
      device const half *x [[buffer(0)]],                                      \
      device const float *row_scale [[buffer(1)]],                             \
      device const uchar *g_codes [[buffer(2)]],                               \
      device const uchar *g_scales [[buffer(3)]],                              \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_scales [[buffer(5)]],                              \
      device bfloat *y [[buffer(6)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(7)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 2u * ((BM * 16u) / 32u) * 32u]; \
    static_assert(sizeof(parts) <= kApple10MaxPrefillTgSramBytes,              \
                  "mxfp4 bm_n16_swiglu parts exceeds 16.0 KiB");               \
    mxfp4_nax_swiglu_bm_impl<BM##u, 16u, SK, false>(                           \
        x, row_scale, g_codes, g_scales, u_codes, u_scales, y, nullptr, p,     \
        parts, lane, sg, tg);                                                  \
  }                                                                            \
  [[max_total_threads_per_threadgroup((SK) * 32u)]]                            \
  kernel void omlx_m5_mxfp4_nax_swiglu_f16_bm##BM##_n16_sk##SK(                \
      device const half *x [[buffer(0)]],                                      \
      device const float *row_scale [[buffer(1)]],                             \
      device const uchar *g_codes [[buffer(2)]],                               \
      device const uchar *g_scales [[buffer(3)]],                              \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_scales [[buffer(5)]],                              \
      device half *y_f16 [[buffer(6)]],                                        \
      constant Mxfp4NaxParams &p [[buffer(7)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 2u * ((BM * 16u) / 32u) * 32u]; \
    static_assert(sizeof(parts) <= kApple10MaxPrefillTgSramBytes,              \
                  "mxfp4 bm_n16_swiglu_f16 parts exceeds 16.0 KiB");           \
    mxfp4_nax_swiglu_bm_impl<BM##u, 16u, SK, true>(                            \
        x, row_scale, g_codes, g_scales, u_codes, u_scales, nullptr, y_f16, p, \
        parts, lane, sg, tg);                                                  \
  }

#define OMLX_M5_MXFP4_NAX_BM_N16(BM, SK)                                       \
  OMLX_M5_MXFP4_NAX_BM_N16_LINEAR(BM, SK)                                      \
  OMLX_M5_MXFP4_NAX_BM_N16_SWIGLU(BM, SK)

#define OMLX_M5_MXFP4_NAX_BM_N32(BM, SK)                                       \
  [[max_total_threads_per_threadgroup((SK) * 32u)]]                            \
  kernel void omlx_m5_mxfp4_nax_bm##BM##_n32_sk##SK(                           \
      device const half *x [[buffer(0)]],                                      \
      device const float *row_scale [[buffer(1)]],                             \
      device const uchar *codes [[buffer(2)]],                                 \
      device const uchar *scales [[buffer(3)]],                                \
      device bfloat *y [[buffer(4)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(5)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * ((BM * 32u) / 32u) * 32u]; \
    static_assert(sizeof(parts) <= kApple10MaxPrefillTgSramBytes,              \
                  "mxfp4 bm_n32_linear parts exceeds 16.0 KiB");               \
    mxfp4_nax_linear_bm_impl<BM##u, 32u, SK>(                                  \
        x, row_scale, codes, scales, y, p, parts, lane, sg, tg);               \
  }

OMLX_M5_MXFP4_NAX_BM_N16(16, 1)
OMLX_M5_MXFP4_NAX_BM_N16(16, 2)
OMLX_M5_MXFP4_NAX_BM_N16(16, 4)
OMLX_M5_MXFP4_NAX_BM_N16(16, 8)
OMLX_M5_MXFP4_NAX_BM_N16(32, 1)
OMLX_M5_MXFP4_NAX_BM_N16(32, 2)
OMLX_M5_MXFP4_NAX_BM_N16(32, 4)
// WP3.1: bm32_n16_sk8 SwiGLU (28 KiB) removed; Linear (14 KiB) retained.
OMLX_M5_MXFP4_NAX_BM_N16_LINEAR(32, 8)
OMLX_M5_MXFP4_NAX_BM_N32(16, 1)
OMLX_M5_MXFP4_NAX_BM_N32(16, 2)
OMLX_M5_MXFP4_NAX_BM_N32(32, 1)
OMLX_M5_MXFP4_NAX_BM_N32(32, 2)

// ============================================================================
// Step 2 (M2) + Tier-1 (T1-A): Cooperative TG-SRAM Staged + SRAM-Dequant Wide SK=1
// MXFP4 W4A16 (Lossless half As[64*72] + bfloat Ws[64*72], Register-Fragment
// relaxed_precision=true 32x32x32 NAX MMA, L2 GROUP_M=8 Swizzling)
// ============================================================================

inline uint2 mxfp4_swizzle_tg_group_m8(uint2 tg_xy, uint num_pid_n, uint num_pid_m) {
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

template <typename T>
inline void mxfp4_mma_16x32x16_fp(
    thread vec<float, 8> &Cn0,
    thread vec<float, 8> &Cn1,
    thread const vec<T, 8> &A,
    thread const vec<T, 8> &Bn0,
    thread const vec<T, 8> &Bn1) {
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<T, T, float>();
  auto ct_b =
      gemm_op.template get_right_input_cooperative_tensor<T, T, float>();
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

template <typename T>
inline bool mxfp4_has_apple10_nax_layout_t() {
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<T, T, float>();
  const auto c0 = ct_a.get_multidimensional_index(ushort(0u));
  const auto c2 = ct_a.get_multidimensional_index(ushort(2u));
  return (int(c2[0]) - int(c0[0]) == 2) && (int(c2[1]) == int(c0[1]));
}

template <typename T>
inline void mxfp4_unpack_raw_32(
    uint4 raw,
    float sc,
    threadgroup vec<T, 8> *w_dst8) {
  #pragma clang loop unroll(full)
  for (uint w = 0u; w < 4u; ++w) {
    const uint v = raw[w];
    vec<T, 8> out;
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 8u; ++j) {
      out[j] = T(sc * fp4_nibble_to_float((v >> (j * 4u)) & 0x0fu));
    }
    w_dst8[w] = out;
  }
}

template <uint BN_STORED>
inline void mxfp4_fetch_raw_32(
    device const uchar *codes,
    device const uchar *scales,
    uint col,
    uint n_limit,
    uint blocks128,
    uint blk128,
    uint sub32,
    thread uint4 &raw_out,
    thread float &sc_out) {
  if (col < n_limit) {
    const uint tile_n = col / BN_STORED;
    const uint rn = col % BN_STORED;
    const ulong tile_idx = ulong(tile_n) * ulong(blocks128) + ulong(blk128);
    raw_out = reinterpret_cast<device const uint4 *>(
        codes + tile_idx * (ulong(BN_STORED) * 64ul) + ulong(rn) * 64ul)[sub32];
    sc_out = e8m0_to_float(
        scales[tile_idx * (ulong(BN_STORED) * 4ul) + ulong(rn) * 4ul + sub32]);
  } else {
    raw_out = uint4(0u);
    sc_out = 0.0f;
  }
}

template <typename InT, bool HasRowScale, uint BN_STORED, uint WM, uint WN>
inline void mxfp4_nax_wide_sk1_linear_apple9(
    device const InT *x,
    device const float *row_scale,
    device const uchar *codes,
    device const uchar *scales,
    device bfloat *y,
    constant Mxfp4NaxParams &p,
    threadgroup InT *Ws,
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

  const bool unit_scale = !HasRowScale || ((p.mp >> 31u) != 0u);
  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * SN;
  const bool sg_active = (sg_m < WM) && (sg_n < WN) && (rb_sg < p.rows) && (n0_sg < p.n);
  const uint blocks128 = p.k / 128u;
  const uint k_blocks = p.k / kBK;

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a = gemm_op.template get_left_input_cooperative_tensor<InT, InT, float>();
  auto ct_b = gemm_op.template get_right_input_cooperative_tensor<InT, InT, float>();
  auto ct_c0 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
  auto ct_c1 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 16u; ++i) {
    ct_c0[i] = 0.0f;
    ct_c1[i] = 0.0f;
  }

  for (uint kb = 0u; kb < k_blocks; ++kb) {
    const uint k_off = kb * kBK;
    const uint blk128 = kb >> 1u;
    const uint half64 = kb & 1u;
    uint4 raw_f[kFetches];
    float sc_f[kFetches];
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      mxfp4_fetch_raw_32<BN_STORED>(
          codes, scales, n0_tg + (idx32 >> 1u), p.n, blocks128, blk128,
          half64 * 2u + (idx32 & 1u), raw_f[f], sc_f[f]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      threadgroup vec<InT, 8> *w_dst8 = reinterpret_cast<threadgroup vec<InT, 8> *>(
          Ws + (idx32 >> 1u) * kBKP + (idx32 & 1u) * 32u);
      mxfp4_unpack_raw_32<InT>(raw_f[f], sc_f[f], w_dst8);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_active) {
      #pragma clang loop unroll(full)
      for (uint kk = 0u; kk < kBK; kk += 16u) {
        const threadgroup InT *ws_sub = Ws + (sg_n * SN) * kBKP + kk;
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          const auto idx_b = ct_b.get_multidimensional_index(i);
          ct_b[i] = ws_sub[uint(idx_b[1]) * kBKP + uint(idx_b[0])];
        }
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 8u; ++i) {
          const auto idx_a = ct_a.get_multidimensional_index(i);
          const uint r0 = rb_sg + uint(idx_a[1]);
          ct_a[i] = (r0 < p.rows) ? x[ulong(r0) * p.k + k_off + kk + uint(idx_a[0])] : InT(0.0f);
        }
        gemm_op.run(ct_a, ct_b, ct_c0);
        if (rb_sg + 16u < p.rows) {
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 8u; ++i) {
            const auto idx_a = ct_a.get_multidimensional_index(i);
            const uint r1 = rb_sg + 16u + uint(idx_a[1]);
            ct_a[i] = (r1 < p.rows) ? x[ulong(r1) * p.k + k_off + kk + uint(idx_a[0])] : InT(0.0f);
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
        if constexpr (HasRowScale) {
          const float rs0 = unit_scale ? 1.0f : row_scale[r0];
          y[ulong(r0) * p.n + col] = bfloat(ct_c0[i] * rs0);
        } else {
          y[ulong(r0) * p.n + col] = bfloat(ct_c0[i]);
        }
      }
      const uint r1 = r0 + 16u;
      if (r1 < p.rows && col < p.n) {
        if constexpr (HasRowScale) {
          const float rs1 = unit_scale ? 1.0f : row_scale[r1];
          y[ulong(r1) * p.n + col] = bfloat(ct_c1[i] * rs1);
        } else {
          y[ulong(r1) * p.n + col] = bfloat(ct_c1[i]);
        }
      }
    }
  }
}

template <typename InT, bool HasRowScale, bool OutF16, uint WM, uint WN>
inline void mxfp4_nax_wide_sk1_swiglu_apple9(
    device const InT *x,
    device const float *row_scale,
    device const uchar *g_codes,
    device const uchar *g_scales,
    device const uchar *u_codes,
    device const uchar *u_scales,
    device bfloat *y_bf16,
    device half *y_f16,
    constant Mxfp4NaxParams &p,
    threadgroup InT *Ws,
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

  const uint mp_eff = p.mp & 0x7FFFFFFFu;
  const bool unit_scale = !HasRowScale || ((p.mp >> 31u) != 0u);
  const uint store_row_limit = OutF16 ? mp_eff : p.rows;
  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * OUT_SN;
  const bool sg_valid = (sg_m < WM) && (sg_n < WN) && (n0_sg < p.n);
  const bool sg_compute = sg_valid && (rb_sg < p.rows);
  const bool sg_store = sg_valid && (rb_sg < store_row_limit);
  const uint blocks128 = p.k / 128u;
  const uint k_blocks = p.k / kBK;

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a = gemm_op.template get_left_input_cooperative_tensor<InT, InT, float>();
  auto ct_b = gemm_op.template get_right_input_cooperative_tensor<InT, InT, float>();
  auto ct_c0 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
  auto ct_c1 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 16u; ++i) {
    ct_c0[i] = 0.0f;
    ct_c1[i] = 0.0f;
  }

  if (rb_tg < p.rows) {
    for (uint kb = 0u; kb < k_blocks; ++kb) {
      const uint k_off = kb * kBK;
      const uint blk128 = kb >> 1u;
      const uint half64 = kb & 1u;
      uint4 raw_f[kFetches];
      float sc_f[kFetches];
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint idx32 = tid + f * TPTG;
        const uint w_row = idx32 >> 1u;
        const bool is_up = ((w_row >> 4u) & 1u) != 0u;
        const uint out_col = n0_tg + ((w_row >> 5u) * 16u) + (w_row & 15u);
        mxfp4_fetch_raw_32<16u>(
            is_up ? u_codes : g_codes, is_up ? u_scales : g_scales,
            out_col, p.n, blocks128, blk128, half64 * 2u + (idx32 & 1u),
            raw_f[f], sc_f[f]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint idx32 = tid + f * TPTG;
        threadgroup vec<InT, 8> *w_dst8 = reinterpret_cast<threadgroup vec<InT, 8> *>(
            Ws + (idx32 >> 1u) * kBKP + (idx32 & 1u) * 32u);
        mxfp4_unpack_raw_32<InT>(raw_f[f], sc_f[f], w_dst8);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_compute) {
        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < kBK; kk += 16u) {
          const threadgroup InT *ws_sub = Ws + (sg_n * 32u) * kBKP + kk;
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            const auto idx_b = ct_b.get_multidimensional_index(i);
            ct_b[i] = ws_sub[uint(idx_b[1]) * kBKP + uint(idx_b[0])];
          }
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 8u; ++i) {
            const auto idx_a = ct_a.get_multidimensional_index(i);
            const uint r0 = rb_sg + uint(idx_a[1]);
            ct_a[i] = (r0 < p.rows) ? x[ulong(r0) * p.k + k_off + kk + uint(idx_a[0])] : InT(0.0f);
          }
          gemm_op.run(ct_a, ct_b, ct_c0);
          if (rb_sg + 16u < p.rows) {
            #pragma clang loop unroll(full)
            for (ushort i = 0u; i < 8u; ++i) {
              const auto idx_a = ct_a.get_multidimensional_index(i);
              const uint r1 = rb_sg + 16u + uint(idx_a[1]);
              ct_a[i] = (r1 < p.rows) ? x[ulong(r1) * p.k + k_off + kk + uint(idx_a[0])] : InT(0.0f);
            }
            gemm_op.run(ct_a, ct_b, ct_c1);
          }
        }
      }
    }
  }

  if (sg_store) {
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm) {
      if (rb_sg + mm * 16u < store_row_limit) {
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          const auto idx_g = ct_c0.get_multidimensional_index(i);
          const uint c_g = uint(idx_g[0]);
          const uint r_g = uint(idx_g[1]);
          if (c_g < 16u) {
            const uint col = n0_sg + c_g;
            const uint row = rb_sg + mm * 16u + r_g;
            if (col < p.n) {
              if (row < p.rows) {
                float gv = (mm == 0u) ? ct_c0[i] : ct_c1[i];
                float uv = 0.0f;
                #pragma clang loop unroll(full)
                for (ushort j = 0u; j < 16u; ++j) {
                  const auto idx_u = ct_c0.get_multidimensional_index(j);
                  if (uint(idx_u[0]) == c_g + 16u && uint(idx_u[1]) == r_g) {
                    uv = (mm == 0u) ? ct_c0[j] : ct_c1[j];
                  }
                }
                if constexpr (HasRowScale) {
                  if (!unit_scale) {
                    const float rs = row_scale[row];
                    gv *= rs;
                    uv *= rs;
                  }
                }
                gv = float(bfloat(gv));
                uv = float(bfloat(uv));
                if constexpr (OutF16) {
                  const float ov = float(bfloat(nax_silu_f32(gv) * uv));
                  y_f16[ulong(row) * p.n + col] = half(clamp(ov, -65504.0f, 65504.0f));
                } else {
                  y_bf16[ulong(row) * p.n + col] = bfloat(nax_silu_f32(gv) * uv);
                }
              } else if constexpr (OutF16) {
                if (row < mp_eff) {
                  y_f16[ulong(row) * p.n + col] = half(0.0h);
                }
              }
            }
          }
        }
      }
    }
  }
}

template <typename InT, bool HasRowScale, uint BN_STORED, uint WM, uint WN>
inline void mxfp4_nax_wide_sk1_linear_impl(
    device const InT *x,
    device const float *row_scale,
    device const uchar *codes,
    device const uchar *scales,
    device bfloat *y,
    constant Mxfp4NaxParams &p,
    threadgroup InT *Ws,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint SM = 32u;
  constexpr uint SN = 32u;
  constexpr uint BM_TG = WM * SM;
  constexpr uint BN_TG = WN * SN;
  constexpr uint TPTG = WM * WN * 32u;
  constexpr uint kFetches = (BN_TG * 2u) / TPTG;
  using Vec4T = vec<InT, 4>;

  const uint mp_eff = p.mp & 0x7FFFFFFFu;
  const bool unit_scale = !HasRowScale || ((p.mp >> 31u) != 0u);
  const uint num_pid_n = (p.n + BN_TG - 1u) / BN_TG;
  const uint num_pid_m = (mp_eff + BM_TG - 1u) / BM_TG;
  const uint2 pid_nm = mxfp4_swizzle_tg_group_m8(tg.xy, num_pid_n, num_pid_m);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg = pid_nm.x * BN_TG;
  if (rb_tg >= p.rows || n0_tg >= p.n) {
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint sg_m = (WM > 1u) ? (sg / WN) : 0u;
  const uint sg_n = (WN > 1u) ? (sg % WN) : sg;
  if (!mxfp4_has_apple10_nax_layout_t<InT>()) {
    mxfp4_nax_wide_sk1_linear_apple9<InT, HasRowScale, BN_STORED, WM, WN>(
        x, row_scale, codes, scales, y, p, Ws, tid, sg_m, sg_n, rb_tg, n0_tg);
    return;
  }

  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * SN;
  const uint blocks128 = p.k / 128u;
  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

  vec<float, 8> D[2][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < 2u; ++mm) {
    D[mm][0] = vec<float, 8>(0.0f);
    D[mm][1] = vec<float, 8>(0.0f);
  }

  const bool full_tile = (rb_tg + BM_TG <= p.rows) && (n0_tg + BN_TG <= p.n);
  if (full_tile) {
    device const uchar *wc_row[kFetches];
    device const uchar *ws_row[kFetches];
    uint col32_f[kFetches];
    threadgroup vec<InT, 8> *w_dst8[kFetches];
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      const uint w_row = idx32 >> 1u;
      const uint col32 = idx32 & 1u;
      const uint tile_n = (n0_tg + w_row) / BN_STORED;
      const uint rn = (n0_tg + w_row) % BN_STORED;
      wc_row[f] = codes + ulong(tile_n) * ulong(blocks128) * (ulong(BN_STORED) * 64ul) + ulong(rn) * 64ul;
      ws_row[f] = scales + ulong(tile_n) * ulong(blocks128) * (ulong(BN_STORED) * 4ul) + ulong(rn) * 4ul;
      col32_f[f] = col32;
      w_dst8[f] = reinterpret_cast<threadgroup vec<InT, 8> *>(Ws + w_row * kBKP + col32 * 32u);
    }
    device const InT *x_sg = x + ulong(rb_sg + fm) * p.k + fn;
    const threadgroup InT *ws_sg = Ws + (sg_n * 32u + fm) * kBKP + fn;

    for (uint k0 = 0u; k0 < p.k; k0 += kBK) {
      const uint blk128 = k0 >> 7u;
      const uint sub_base = (k0 >> 5u) & 3u;
      uint4 raw_f[kFetches];
      float sc_f[kFetches];
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint sub32 = sub_base + col32_f[f];
        raw_f[f] = reinterpret_cast<device const uint4 *>(
            wc_row[f] + ulong(blk128) * (ulong(BN_STORED) * 64ul))[sub32];
        sc_f[f] = e8m0_to_float(
            (ws_row[f] + ulong(blk128) * (ulong(BN_STORED) * 4ul))[sub32]);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        mxfp4_unpack_raw_32<InT>(raw_f[f], sc_f[f], w_dst8[f]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      #pragma clang loop unroll(disable)
      for (uint kk = 0u; kk < kBK; kk += 32u) {
        vec<InT, 8> a00, a01, a10, a11, b00, b01, b10, b11;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 2u; ++r) {
          const Vec4T va0 = *reinterpret_cast<device const Vec4T *>(x_sg + ulong(r * 8u) * p.k + k0 + kk + 0u);
          const Vec4T va1 = *reinterpret_cast<device const Vec4T *>(x_sg + ulong(r * 8u) * p.k + k0 + kk + 16u);
          const Vec4T va2 = *reinterpret_cast<device const Vec4T *>(x_sg + ulong(16u + r * 8u) * p.k + k0 + kk + 0u);
          const Vec4T va3 = *reinterpret_cast<device const Vec4T *>(x_sg + ulong(16u + r * 8u) * p.k + k0 + kk + 16u);
          const Vec4T vb0 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + r * 8u * kBKP + kk + 0u);
          const Vec4T vb1 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + r * 8u * kBKP + kk + 16u);
          const Vec4T vb2 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + (16u + r * 8u) * kBKP + kk + 0u);
          const Vec4T vb3 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + (16u + r * 8u) * kBKP + kk + 16u);
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            a00[r * 4u + c] = va0[c]; a01[r * 4u + c] = va1[c];
            a10[r * 4u + c] = va2[c]; a11[r * 4u + c] = va3[c];
            b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
            b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
          }
        }
        mxfp4_mma_16x32x16_fp<InT>(D[0][0], D[0][1], a00, b00, b10);
        mxfp4_mma_16x32x16_fp<InT>(D[0][0], D[0][1], a01, b01, b11);
        mxfp4_mma_16x32x16_fp<InT>(D[1][0], D[1][1], a10, b00, b10);
        mxfp4_mma_16x32x16_fp<InT>(D[1][0], D[1][1], a11, b01, b11);
      }
    }
  } else {
    const bool sg_active = (sg_m < WM) && (sg_n < WN) && (rb_sg < p.rows) && (n0_sg < p.n);
    const uint k_blocks = p.k / kBK;
    const threadgroup InT *ws_sg = Ws + (sg_n * SN + fm) * kBKP + fn;
    for (uint kb = 0u; kb < k_blocks; ++kb) {
      const uint k_off = kb * kBK;
      const uint blk128 = kb >> 1u;
      const uint half64 = kb & 1u;
      uint4 raw_f[kFetches];
      float sc_f[kFetches];
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint idx32 = tid + f * TPTG;
        mxfp4_fetch_raw_32<BN_STORED>(
            codes, scales, n0_tg + (idx32 >> 1u), p.n, blocks128, blk128,
            half64 * 2u + (idx32 & 1u), raw_f[f], sc_f[f]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint idx32 = tid + f * TPTG;
        threadgroup vec<InT, 8> *w_dst8 = reinterpret_cast<threadgroup vec<InT, 8> *>(
            Ws + (idx32 >> 1u) * kBKP + (idx32 & 1u) * 32u);
        mxfp4_unpack_raw_32<InT>(raw_f[f], sc_f[f], w_dst8);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_active) {
        #pragma clang loop unroll(disable)
        for (uint kk = 0u; kk < kBK; kk += 32u) {
          vec<InT, 8> b00, b01, b10, b11;
          #pragma clang loop unroll(full)
          for (uint r = 0u; r < 2u; ++r) {
            const Vec4T vb0 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + r * 8u * kBKP + kk + 0u);
            const Vec4T vb1 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + r * 8u * kBKP + kk + 16u);
            const Vec4T vb2 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + (16u + r * 8u) * kBKP + kk + 0u);
            const Vec4T vb3 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + (16u + r * 8u) * kBKP + kk + 16u);
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) {
              b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
              b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
            }
          }
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            if (rb_sg + mm * 16u < p.rows) {
              vec<InT, 8> a0, a1;
              #pragma clang loop unroll(full)
              for (uint r = 0u; r < 2u; ++r) {
                const uint row = rb_sg + mm * 16u + fm + r * 8u;
                const Vec4T va0 = (row < p.rows)
                    ? *reinterpret_cast<device const Vec4T *>(x + ulong(row) * p.k + k_off + kk + fn)
                    : Vec4T(InT(0.0f));
                const Vec4T va1 = (row < p.rows)
                    ? *reinterpret_cast<device const Vec4T *>(x + ulong(row) * p.k + k_off + kk + 16u + fn)
                    : Vec4T(InT(0.0f));
                #pragma clang loop unroll(full)
                for (uint c = 0u; c < 4u; ++c) {
                  a0[r * 4u + c] = va0[c];
                  a1[r * 4u + c] = va1[c];
                }
              }
              mxfp4_mma_16x32x16_fp<InT>(D[mm][0], D[mm][1], a0, b00, b10);
              mxfp4_mma_16x32x16_fp<InT>(D[mm][0], D[mm][1], a1, b01, b11);
            }
          }
          }
      }
    }
  }

  const bool sg_active = (sg_m < WM) && (sg_n < WN) && (rb_sg < p.rows) && (n0_sg < p.n);
  if (sg_active) {
    const bool aligned_n = ((p.n & 3u) == 0u);
    if (!HasRowScale || unit_scale) {
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
    } else if constexpr (HasRowScale) {
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        if (rb_sg + mm * 16u < p.rows) {
          #pragma clang loop unroll(full)
          for (uint h = 0u; h < 2u; ++h) {
            const uint row = rb_sg + mm * 16u + fm + h * 8u;
            if (row < p.rows) {
              const float rs = row_scale[row];
              const uint e = h * 4u;
              #pragma clang loop unroll(full)
              for (uint nn = 0u; nn < 2u; ++nn) {
                const uint col = n0_sg + nn * 16u + fn;
                if (col < p.n) {
                  const bfloat4 out_v = bfloat4(
                      bfloat(D[mm][nn][e + 0u] * rs),
                      bfloat(D[mm][nn][e + 1u] * rs),
                      bfloat(D[mm][nn][e + 2u] * rs),
                      bfloat(D[mm][nn][e + 3u] * rs));
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
}

template <typename InT, bool HasRowScale, bool OutF16, uint WM, uint WN>
inline void mxfp4_nax_wide_sk1_swiglu_impl(
    device const InT *x,
    device const float *row_scale,
    device const uchar *g_codes,
    device const uchar *g_scales,
    device const uchar *u_codes,
    device const uchar *u_scales,
    device bfloat *y_bf16,
    device half *y_f16,
    constant Mxfp4NaxParams &p,
    threadgroup InT *Ws,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint SM = 32u;
  constexpr uint OUT_SN = 16u;
  constexpr uint BM_TG = WM * SM;
  constexpr uint OUT_N_TG = WN * OUT_SN;
  constexpr uint WS_ROWS = WN * 32u;
  constexpr uint TPTG = WM * WN * 32u;
  constexpr uint kFetches = (WS_ROWS * 2u) / TPTG;
  using Vec4T = vec<InT, 4>;

  const uint mp_eff = p.mp & 0x7FFFFFFFu;
  const bool unit_scale = !HasRowScale || ((p.mp >> 31u) != 0u);
  const uint store_row_limit = OutF16 ? mp_eff : p.rows;
  const uint num_pid_n = (p.n + OUT_N_TG - 1u) / OUT_N_TG;
  const uint num_pid_m = (mp_eff + BM_TG - 1u) / BM_TG;
  const uint2 pid_nm = mxfp4_swizzle_tg_group_m8(tg.xy, num_pid_n, num_pid_m);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg = pid_nm.x * OUT_N_TG;
  if (rb_tg >= store_row_limit || n0_tg >= p.n) {
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint sg_m = (WM > 1u) ? (sg / WN) : 0u;
  const uint sg_n = (WN > 1u) ? (sg % WN) : sg;
  if (!mxfp4_has_apple10_nax_layout_t<InT>()) {
    mxfp4_nax_wide_sk1_swiglu_apple9<InT, HasRowScale, OutF16, WM, WN>(
        x, row_scale, g_codes, g_scales, u_codes, u_scales, y_bf16, y_f16, p,
        Ws, tid, sg_m, sg_n, rb_tg, n0_tg);
    return;
  }

  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * OUT_SN;
  const bool sg_valid = (sg_m < WM) && (sg_n < WN) && (n0_sg < p.n);
  const bool sg_compute = sg_valid && (rb_sg < p.rows);
  const bool sg_store = sg_valid && (rb_sg < store_row_limit);
  const uint blocks128 = p.k / 128u;
  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

  vec<float, 8> D[2][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < 2u; ++mm) {
    D[mm][0] = vec<float, 8>(0.0f);
    D[mm][1] = vec<float, 8>(0.0f);
  }

  const bool full_tile = (rb_tg + BM_TG <= p.rows) && (n0_tg + OUT_N_TG <= p.n);
  if (full_tile) {
    device const uchar *wc_row[kFetches];
    device const uchar *ws_row[kFetches];
    uint col32_f[kFetches];
    threadgroup vec<InT, 8> *w_dst8[kFetches];
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      const uint w_row = idx32 >> 1u;
      const uint col32 = idx32 & 1u;
      const bool is_up = ((w_row >> 4u) & 1u) != 0u;
      const uint out_col = n0_tg + ((w_row >> 5u) * 16u) + (w_row & 15u);
      const uint tile_n = out_col >> 4u;
      const uint rn = out_col & 15u;
      wc_row[f] = (is_up ? u_codes : g_codes) + ulong(tile_n) * ulong(blocks128) * 1024ul + ulong(rn) * 64ul;
      ws_row[f] = (is_up ? u_scales : g_scales) + ulong(tile_n) * ulong(blocks128) * 64ul + ulong(rn) * 4ul;
      col32_f[f] = col32;
      w_dst8[f] = reinterpret_cast<threadgroup vec<InT, 8> *>(Ws + w_row * kBKP + col32 * 32u);
    }
    device const InT *x_sg = x + ulong(rb_sg + fm) * p.k + fn;
    const threadgroup InT *ws_sg = Ws + (sg_n * 32u + fm) * kBKP + fn;

    for (uint k0 = 0u; k0 < p.k; k0 += kBK) {
      const uint blk128 = k0 >> 7u;
      const uint sub_base = (k0 >> 5u) & 3u;
      uint4 raw_f[kFetches];
      float sc_f[kFetches];
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint sub32 = sub_base + col32_f[f];
        raw_f[f] = reinterpret_cast<device const uint4 *>(
            wc_row[f] + ulong(blk128) * 1024ul)[sub32];
        sc_f[f] = e8m0_to_float((ws_row[f] + ulong(blk128) * 64ul)[sub32]);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        mxfp4_unpack_raw_32<InT>(raw_f[f], sc_f[f], w_dst8[f]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      #pragma clang loop unroll(disable)
      for (uint kk = 0u; kk < kBK; kk += 32u) {
        vec<InT, 8> a00, a01, a10, a11, b00, b01, b10, b11;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 2u; ++r) {
          const Vec4T va0 = *reinterpret_cast<device const Vec4T *>(x_sg + ulong(r * 8u) * p.k + k0 + kk + 0u);
          const Vec4T va1 = *reinterpret_cast<device const Vec4T *>(x_sg + ulong(r * 8u) * p.k + k0 + kk + 16u);
          const Vec4T va2 = *reinterpret_cast<device const Vec4T *>(x_sg + ulong(16u + r * 8u) * p.k + k0 + kk + 0u);
          const Vec4T va3 = *reinterpret_cast<device const Vec4T *>(x_sg + ulong(16u + r * 8u) * p.k + k0 + kk + 16u);
          const Vec4T vb0 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + r * 8u * kBKP + kk + 0u);
          const Vec4T vb1 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + r * 8u * kBKP + kk + 16u);
          const Vec4T vb2 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + (16u + r * 8u) * kBKP + kk + 0u);
          const Vec4T vb3 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + (16u + r * 8u) * kBKP + kk + 16u);
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            a00[r * 4u + c] = va0[c]; a01[r * 4u + c] = va1[c];
            a10[r * 4u + c] = va2[c]; a11[r * 4u + c] = va3[c];
            b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
            b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
          }
        }
        mxfp4_mma_16x32x16_fp<InT>(D[0][0], D[0][1], a00, b00, b10);
        mxfp4_mma_16x32x16_fp<InT>(D[0][0], D[0][1], a01, b01, b11);
        mxfp4_mma_16x32x16_fp<InT>(D[1][0], D[1][1], a10, b00, b10);
        mxfp4_mma_16x32x16_fp<InT>(D[1][0], D[1][1], a11, b01, b11);
      }
    }
  } else if (rb_tg < p.rows) {
    const uint k_blocks = p.k / kBK;
    const threadgroup InT *ws_sg = Ws + (sg_n * 32u + fm) * kBKP + fn;
    for (uint kb = 0u; kb < k_blocks; ++kb) {
      const uint k_off = kb * kBK;
      const uint blk128 = kb >> 1u;
      const uint half64 = kb & 1u;
      uint4 raw_f[kFetches];
      float sc_f[kFetches];
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint idx32 = tid + f * TPTG;
        const uint w_row = idx32 >> 1u;
        const bool is_up = ((w_row >> 4u) & 1u) != 0u;
        const uint out_col = n0_tg + ((w_row >> 5u) * 16u) + (w_row & 15u);
        mxfp4_fetch_raw_32<16u>(
            is_up ? u_codes : g_codes, is_up ? u_scales : g_scales,
            out_col, p.n, blocks128, blk128, half64 * 2u + (idx32 & 1u),
            raw_f[f], sc_f[f]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint idx32 = tid + f * TPTG;
        threadgroup vec<InT, 8> *w_dst8 = reinterpret_cast<threadgroup vec<InT, 8> *>(
            Ws + (idx32 >> 1u) * kBKP + (idx32 & 1u) * 32u);
        mxfp4_unpack_raw_32<InT>(raw_f[f], sc_f[f], w_dst8);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_compute) {
        #pragma clang loop unroll(disable)
        for (uint kk = 0u; kk < kBK; kk += 32u) {
          vec<InT, 8> b00, b01, b10, b11;
          #pragma clang loop unroll(full)
          for (uint r = 0u; r < 2u; ++r) {
            const Vec4T vb0 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + r * 8u * kBKP + kk + 0u);
            const Vec4T vb1 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + r * 8u * kBKP + kk + 16u);
            const Vec4T vb2 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + (16u + r * 8u) * kBKP + kk + 0u);
            const Vec4T vb3 = *reinterpret_cast<const threadgroup Vec4T *>(ws_sg + (16u + r * 8u) * kBKP + kk + 16u);
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) {
              b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
              b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
            }
          }
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            if (rb_sg + mm * 16u < p.rows) {
              vec<InT, 8> a0, a1;
              #pragma clang loop unroll(full)
              for (uint r = 0u; r < 2u; ++r) {
                const uint row = rb_sg + mm * 16u + fm + r * 8u;
                const Vec4T va0 = (row < p.rows)
                    ? *reinterpret_cast<device const Vec4T *>(x + ulong(row) * p.k + k_off + kk + fn)
                    : Vec4T(InT(0.0f));
                const Vec4T va1 = (row < p.rows)
                    ? *reinterpret_cast<device const Vec4T *>(x + ulong(row) * p.k + k_off + kk + 16u + fn)
                    : Vec4T(InT(0.0f));
                #pragma clang loop unroll(full)
                for (uint c = 0u; c < 4u; ++c) {
                  a0[r * 4u + c] = va0[c];
                  a1[r * 4u + c] = va1[c];
                }
              }
              mxfp4_mma_16x32x16_fp<InT>(D[mm][0], D[mm][1], a0, b00, b10);
              mxfp4_mma_16x32x16_fp<InT>(D[mm][0], D[mm][1], a1, b01, b11);
            }
          }
          }
      }
    }
  }

  if (sg_store) {
    const bool aligned_n = ((p.n & 3u) == 0u);
    const uint col = n0_sg + fn;
    if (col < p.n) {
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        if (rb_sg + mm * 16u < store_row_limit) {
          #pragma clang loop unroll(full)
          for (uint h = 0u; h < 2u; ++h) {
            const uint row = rb_sg + mm * 16u + fm + h * 8u;
            const uint e = h * 4u;
            if constexpr (OutF16) {
              if (row < p.rows) {
                const float rs = (!HasRowScale || unit_scale) ? 1.0f : row_scale[row];
                half4 out_h;
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  const float gv = (!HasRowScale || unit_scale) ? float(bfloat(D[mm][0][e + j])) : float(bfloat(D[mm][0][e + j] * rs));
                  const float uv = (!HasRowScale || unit_scale) ? float(bfloat(D[mm][1][e + j])) : float(bfloat(D[mm][1][e + j] * rs));
                  const float ov = float(bfloat(nax_silu_f32(gv) * uv));
                  out_h[j] = half(clamp(ov, -65504.0f, 65504.0f));
                }
                if (aligned_n && col + 4u <= p.n) {
                  *reinterpret_cast<device half4 *>(y_f16 + ulong(row) * p.n + col) = out_h;
                } else {
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    if (col + j < p.n) {
                      y_f16[ulong(row) * p.n + col + j] = out_h[j];
                    }
                  }
                }
              } else if (row < mp_eff) {
                if (aligned_n && col + 4u <= p.n) {
                  *reinterpret_cast<device half4 *>(y_f16 + ulong(row) * p.n + col) = half4(0.0h);
                } else {
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    if (col + j < p.n) {
                      y_f16[ulong(row) * p.n + col + j] = half(0.0h);
                    }
                  }
                }
              }
            } else {
              if (row < p.rows) {
                const float rs = (!HasRowScale || unit_scale) ? 1.0f : row_scale[row];
                bfloat4 out_b;
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  const float gv = (!HasRowScale || unit_scale) ? float(bfloat(D[mm][0][e + j])) : float(bfloat(D[mm][0][e + j] * rs));
                  const float uv = (!HasRowScale || unit_scale) ? float(bfloat(D[mm][1][e + j])) : float(bfloat(D[mm][1][e + j] * rs));
                  out_b[j] = bfloat(nax_silu_f32(gv) * uv);
                }
                if (aligned_n && col + 4u <= p.n) {
                  *reinterpret_cast<device bfloat4 *>(y_bf16 + ulong(row) * p.n + col) = out_b;
                } else {
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    if (col + j < p.n) {
                      y_bf16[ulong(row) * p.n + col + j] = out_b[j];
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

#define OMLX_M5_MXFP4_NAX_WIDE_SK1_LINEAR(NAME, BNS, WM, WN)                   \
  [[max_total_threads_per_threadgroup((WM) * (WN) * 32u)]]                     \
  kernel void NAME(                                                            \
      device const half *x [[buffer(0)]],                                      \
      device const float *row_scale [[buffer(1)]],                             \
      device const uchar *codes [[buffer(2)]],                                 \
      device const uchar *scales [[buffer(3)]],                                \
      device bfloat *y [[buffer(4)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(5)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup half Ws[(WN) * 32u * 72u];                         \
    mxfp4_nax_wide_sk1_linear_impl<half, true, BNS, WM, WN>(                   \
        x, row_scale, codes, scales, y, p, Ws, lane, sg, tg);                  \
  }

#define OMLX_M5_MXFP4_NAX_BF16_WIDE_SK1_LINEAR(NAME, BNS, WM, WN)              \
  [[max_total_threads_per_threadgroup((WM) * (WN) * 32u)]]                     \
  kernel void NAME(                                                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *codes [[buffer(1)]],                                 \
      device const uchar *scales [[buffer(2)]],                                \
      device bfloat *y [[buffer(3)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(4)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[(WN) * 32u * 72u];                       \
    mxfp4_nax_wide_sk1_linear_impl<bfloat, false, BNS, WM, WN>(                \
        x, nullptr, codes, scales, y, p, Ws, lane, sg, tg);                    \
  }

OMLX_M5_MXFP4_NAX_WIDE_SK1_LINEAR(omlx_m5_mxfp4_nax_bm64_bn64_bns16_sk1, 16u, 2u, 2u)
OMLX_M5_MXFP4_NAX_WIDE_SK1_LINEAR(omlx_m5_mxfp4_nax_bm64_bn64_bns32_sk1, 32u, 2u, 2u)
OMLX_M5_MXFP4_NAX_WIDE_SK1_LINEAR(omlx_m5_mxfp4_nax_bm32_bn32_bns16_sk1, 16u, 1u, 1u)
OMLX_M5_MXFP4_NAX_WIDE_SK1_LINEAR(omlx_m5_mxfp4_nax_bm32_bn32_bns32_sk1, 32u, 1u, 1u)

OMLX_M5_MXFP4_NAX_BF16_WIDE_SK1_LINEAR(omlx_m5_mxfp4_nax_bf16_bm64_bn64_bns16_sk1, 16u, 2u, 2u)
OMLX_M5_MXFP4_NAX_BF16_WIDE_SK1_LINEAR(omlx_m5_mxfp4_nax_bf16_bm64_bn64_bns32_sk1, 32u, 2u, 2u)
OMLX_M5_MXFP4_NAX_BF16_WIDE_SK1_LINEAR(omlx_m5_mxfp4_nax_bf16_bm32_bn32_bns16_sk1, 16u, 1u, 1u)
OMLX_M5_MXFP4_NAX_BF16_WIDE_SK1_LINEAR(omlx_m5_mxfp4_nax_bf16_bm32_bn32_bns32_sk1, 32u, 1u, 1u)

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_nax_swiglu_bm64_bn64_sk1(
    device const half *x [[buffer(0)]],
    device const float *row_scale [[buffer(1)]],
    device const uchar *g_codes [[buffer(2)]],
    device const uchar *g_scales [[buffer(3)]],
    device const uchar *u_codes [[buffer(4)]],
    device const uchar *u_scales [[buffer(5)]],
    device bfloat *y [[buffer(6)]],
    constant Mxfp4NaxParams &p [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup half Ws[64u * 72u];
  mxfp4_nax_wide_sk1_swiglu_impl<half, true, false, 2u, 2u>(
      x, row_scale, g_codes, g_scales, u_codes, u_scales, y, nullptr, p, Ws, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_nax_swiglu_f16_bm64_bn64_sk1(
    device const half *x [[buffer(0)]],
    device const float *row_scale [[buffer(1)]],
    device const uchar *g_codes [[buffer(2)]],
    device const uchar *g_scales [[buffer(3)]],
    device const uchar *u_codes [[buffer(4)]],
    device const uchar *u_scales [[buffer(5)]],
    device half *y_f16 [[buffer(6)]],
    constant Mxfp4NaxParams &p [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup half Ws[64u * 72u];
  mxfp4_nax_wide_sk1_swiglu_impl<half, true, true, 2u, 2u>(
      x, row_scale, g_codes, g_scales, u_codes, u_scales, nullptr, y_f16, p, Ws, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_mxfp4_nax_swiglu_bm32_bn32_sk1(
    device const half *x [[buffer(0)]],
    device const float *row_scale [[buffer(1)]],
    device const uchar *g_codes [[buffer(2)]],
    device const uchar *g_scales [[buffer(3)]],
    device const uchar *u_codes [[buffer(4)]],
    device const uchar *u_scales [[buffer(5)]],
    device bfloat *y [[buffer(6)]],
    constant Mxfp4NaxParams &p [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup half Ws[32u * 72u];
  mxfp4_nax_wide_sk1_swiglu_impl<half, true, false, 1u, 1u>(
      x, row_scale, g_codes, g_scales, u_codes, u_scales, y, nullptr, p, Ws, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_mxfp4_nax_swiglu_f16_bm32_bn32_sk1(
    device const half *x [[buffer(0)]],
    device const float *row_scale [[buffer(1)]],
    device const uchar *g_codes [[buffer(2)]],
    device const uchar *g_scales [[buffer(3)]],
    device const uchar *u_codes [[buffer(4)]],
    device const uchar *u_scales [[buffer(5)]],
    device half *y_f16 [[buffer(6)]],
    constant Mxfp4NaxParams &p [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup half Ws[32u * 72u];
  mxfp4_nax_wide_sk1_swiglu_impl<half, true, true, 1u, 1u>(
      x, row_scale, g_codes, g_scales, u_codes, u_scales, nullptr, y_f16, p, Ws, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_nax_swiglu_bf16_bm64_bn64_sk1(
    device const bfloat *x [[buffer(0)]],
    device const uchar *g_codes [[buffer(1)]],
    device const uchar *g_scales [[buffer(2)]],
    device const uchar *u_codes [[buffer(3)]],
    device const uchar *u_scales [[buffer(4)]],
    device bfloat *y [[buffer(5)]],
    constant Mxfp4NaxParams &p [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup bfloat Ws[64u * 72u];
  mxfp4_nax_wide_sk1_swiglu_impl<bfloat, false, false, 2u, 2u>(
      x, nullptr, g_codes, g_scales, u_codes, u_scales, y, nullptr, p, Ws, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_mxfp4_nax_swiglu_bf16_bm32_bn32_sk1(
    device const bfloat *x [[buffer(0)]],
    device const uchar *g_codes [[buffer(1)]],
    device const uchar *g_scales [[buffer(2)]],
    device const uchar *u_codes [[buffer(3)]],
    device const uchar *u_scales [[buffer(4)]],
    device bfloat *y [[buffer(5)]],
    constant Mxfp4NaxParams &p [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup bfloat Ws[32u * 72u];
  mxfp4_nax_wide_sk1_swiglu_impl<bfloat, false, false, 1u, 1u>(
      x, nullptr, g_codes, g_scales, u_codes, u_scales, y, nullptr, p, Ws, lane, sg, tg);
}

// ============================================================================
// WP3.1: Compact Ws-Staged 2-Pass (BM=64, BN_out=64, BK=64) Direct-L1-X
// MXFP4 Gate + Up + SwiGLU Fusion (9,216 B = 9.00 KiB <= 10.0 KiB, 0 spills!)
// - Mode A (BF16): 4 SGs = 128 thr (WM=2, WN=2), 2-pass x BN_out=32, pipelined g+1
// - Mode B (Fused FWHT-64 + GS64 INT8): 8 SGs = 256 thr (WM=4, WN=2), 1-phase SwiBuf[64*68]
// + Dual-ANE Output-Channel-Split (`row_offset` % 64 == 0, `out_stride`)
// ============================================================================

struct Mxfp4Out64TgStorage {
  union {
    alignas(16) bfloat Ws[64u * 72u];      // 9,216 B (9.00 KiB)
    alignas(16) bfloat SwiBuf[64u * 68u];  // 8,704 B (Mode B 1-phase epilogue)
  };
};
static_assert(sizeof(Mxfp4Out64TgStorage) == 9216ul &&
                  sizeof(Mxfp4Out64TgStorage) <= kApple10HighOccTgSramBytes,
              "Mxfp4Out64TgStorage must stay <= 10.0 KiB");

inline void mxfp4_out64_fetch_raw_32_checked(
    device const uchar *codes,
    device const uchar *scales,
    uint gcol,
    uint col_limit,
    uint blocks128,
    uint blk128,
    uint sub32,
    thread uint4 &raw,
    thread float &sc) {
  if (gcol < col_limit) {
    const uint tile_n = gcol >> 4u;
    const uint rn = gcol & 15u;
    const ulong blk_idx = ulong(tile_n) * ulong(blocks128) + ulong(blk128);
    raw = reinterpret_cast<device const uint4 *>(
        codes + blk_idx * 1024ul + ulong(rn) * 64ul)[sub32];
    sc = e8m0_to_float(scales[blk_idx * 64ul + ulong(rn) * 4ul + ulong(sub32)]);
  } else {
    raw = uint4(0u);
    sc = 0.0f;
  }
}

template <bool FUSED_FWHT64_A8>
inline void mxfp4_nax_wide_sk1_swiglu_bf16_out64_apple9(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_scales,
    device const uchar *u_codes,
    device const uchar *u_scales,
    device bfloat *y_bf16,
    device int8_t *qa_out,
    device float *sa_out,
    device float *ra_out,
    uint p_n,
    uint p_k,
    uint p_mp,
    uint p_rows,
    uint p_row_offset,
    uint p_out_stride,
    uint p_flags,
    threadgroup Mxfp4Out64TgStorage &tg_mem,
    uint tid,
    uint lane,
    uint sg,
    uint rb_tg,
    uint n0_tg_base) {
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint kSwiStride = 68u;
  constexpr uint WM = FUSED_FWHT64_A8 ? 4u : 2u;
  constexpr uint SM = FUSED_FWHT64_A8 ? 16u : 32u;
  constexpr uint OUT_SN = 16u;
  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const uint rb_sg = rb_tg + sg_m * SM;
  const uint out_stride = (p_out_stride == 0u) ? (p_row_offset + p_n) : p_out_stride;
  const uint blocks128 = p_k / 128u;
  const uint k_blocks = p_k / kBK;
  const uint col_limit = p_row_offset + p_n;

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;

  const bool do_w_load = (tid < 128u);
  const uint w_row = do_w_load ? (tid >> 1u) : 0u;
  const uint col32 = tid & 1u;
  const bool is_up = ((w_row >> 4u) & 1u) != 0u;
  const uint w_col_in32 = ((w_row >> 5u) * 16u) + (w_row & 15u);
  threadgroup vec<bfloat, 8> *w_dst8 =
      reinterpret_cast<threadgroup vec<bfloat, 8> *>(tg_mem.Ws + w_row * kBKP + col32 * 32u);

  vec<bfloat, 8> swi_save_lo = vec<bfloat, 8>(bfloat(0.0f));
  vec<bfloat, 8> swi_save_hi = vec<bfloat, 8>(bfloat(0.0f));

  constexpr uint kPasses = FUSED_FWHT64_A8 ? 2u : 1u;
  #pragma clang loop unroll(full)
  for (uint hn = 0u; hn < kPasses; ++hn) {
    const uint n0_tg = n0_tg_base + hn * 32u;
    const uint n0_sg = n0_tg + sg_n * OUT_SN;
    const bool sg_active = (sg_m < WM) && (rb_sg < p_rows) && (n0_sg < p_n);
    const uint gcol_w = p_row_offset + n0_tg + w_col_in32;

    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c0 = gemm_op.template get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
    auto ct_c1 = gemm_op.template get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) { ct_c0[i] = 0.0f; ct_c1[i] = 0.0f; }

    for (uint kb = 0u; kb < k_blocks; ++kb) {
      const uint k_off = kb * kBK;
      const uint blk128 = kb >> 1u;
      const uint sub32 = (kb & 1u) * 2u + col32;
      uint4 raw = uint4(0u);
      float sc = 0.0f;
      if (do_w_load) {
        mxfp4_out64_fetch_raw_32_checked(
            is_up ? u_codes : g_codes, is_up ? u_scales : g_scales,
            gcol_w, col_limit, blocks128, blk128, sub32, raw, sc);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (do_w_load) {
        mxfp4_unpack_raw_32<bfloat>(raw, sc, w_dst8);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_active) {
        #pragma clang loop unroll(disable)
        for (uint kk = 0u; kk < kBK; kk += 16u) {
          const threadgroup bfloat *ws_sub = tg_mem.Ws + (sg_n * 32u) * kBKP + kk;
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            const auto ib = ct_b.get_multidimensional_index(i);
            ct_b[i] = ws_sub[uint(ib[1]) * kBKP + uint(ib[0])];
          }
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 8u; ++i) {
            const auto ia = ct_a.get_multidimensional_index(i);
            const uint r0 = rb_sg + uint(ia[1]);
            ct_a[i] = (r0 < p_rows) ? x[ulong(r0) * p_k + k_off + kk + uint(ia[0])] : bfloat(0.0f);
          }
          gemm_op.run(ct_a, ct_b, ct_c0);
          if constexpr (!FUSED_FWHT64_A8) {
            if (rb_sg + 16u < p_rows) {
              #pragma clang loop unroll(full)
              for (ushort i = 0u; i < 8u; ++i) {
                const auto ia = ct_a.get_multidimensional_index(i);
                const uint r1 = rb_sg + 16u + uint(ia[1]);
                ct_a[i] = (r1 < p_rows) ? x[ulong(r1) * p_k + k_off + kk + uint(ia[0])] : bfloat(0.0f);
              }
              gemm_op.run(ct_a, ct_b, ct_c1);
            }
          }
        }
      }
    }

    if constexpr (!FUSED_FWHT64_A8) {
      if (sg_active) {
        #pragma clang loop unroll(full)
        for (uint mm = 0u; mm < 2u; ++mm) {
          if (rb_sg + mm * 16u < p_rows) {
            #pragma clang loop unroll(full)
            for (ushort i = 0u; i < 16u; ++i) {
              const auto idx_g = ct_c0.get_multidimensional_index(i);
              const uint c_g = uint(idx_g[0]);
              const uint r_g = uint(idx_g[1]);
              if (c_g < 16u) {
                const uint lcol = n0_sg + c_g;
                const uint row = rb_sg + mm * 16u + r_g;
                if (row < p_rows && lcol < p_n) {
                  float gv = (mm == 0u) ? ct_c0[i] : ct_c1[i];
                  float uv = 0.0f;
                  #pragma clang loop unroll(full)
                  for (ushort j = 0u; j < 16u; ++j) {
                    const auto idx_u = ct_c0.get_multidimensional_index(j);
                    if (uint(idx_u[0]) == c_g + 16u && uint(idx_u[1]) == r_g) {
                      uv = (mm == 0u) ? ct_c0[j] : ct_c1[j];
                    }
                  }
                  y_bf16[ulong(row) * ulong(out_stride) + ulong(p_row_offset + lcol)] =
                      bfloat(nax_silu_f32(float(bfloat(gv))) * float(bfloat(uv)));
                }
              }
            }
          }
        }
      }
    } else {
      if (hn == 0u) {
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          const auto idx_g = ct_c0.get_multidimensional_index(i);
          const uint c_g = uint(idx_g[0]);
          const uint r_g = uint(idx_g[1]);
          if (c_g < 16u) {
            const uint lcol = n0_sg + c_g;
            const uint row = rb_sg + r_g;
            if (sg_active && row < p_rows && lcol < p_n) {
              float uv = 0.0f;
              #pragma clang loop unroll(full)
              for (ushort j = 0u; j < 16u; ++j) {
                const auto idx_u = ct_c0.get_multidimensional_index(j);
                if (uint(idx_u[0]) == c_g + 16u && uint(idx_u[1]) == r_g) uv = ct_c0[j];
              }
              const bfloat sv =
                  bfloat(nax_silu_f32(float(bfloat(ct_c0[i]))) * float(bfloat(uv)));
              if (i < 8u) {
                swi_save_lo[i] = sv;
              } else {
                swi_save_hi[i - 8u] = sv;
              }
            } else {
              if (i < 8u) {
                swi_save_lo[i] = bfloat(0.0f);
              } else {
                swi_save_hi[i - 8u] = bfloat(0.0f);
              }
            }
          }
        }
      } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          const auto idx_g = ct_c0.get_multidimensional_index(i);
          const uint c_g = uint(idx_g[0]);
          const uint r_g = uint(idx_g[1]);
          if (c_g < 16u) {
            const uint r_in_tg = sg_m * SM + r_g;
            const uint c0_in_tg = sg_n * OUT_SN + c_g;
            const uint lcol1 = n0_sg + c_g;
            const uint row = rb_tg + r_in_tg;
            bfloat v1 = bfloat(0.0f);
            if (sg_active && row < p_rows && lcol1 < p_n) {
              float uv = 0.0f;
              #pragma clang loop unroll(full)
              for (ushort j = 0u; j < 16u; ++j) {
                const auto idx_u = ct_c0.get_multidimensional_index(j);
                if (uint(idx_u[0]) == c_g + 16u && uint(idx_u[1]) == r_g) uv = ct_c0[j];
              }
              v1 = bfloat(nax_silu_f32(float(bfloat(ct_c0[i]))) * float(bfloat(uv)));
            }
            const bfloat v0 = (i < 8u) ? swi_save_lo[i] : swi_save_hi[i - 8u];
            tg_mem.SwiBuf[r_in_tg * kSwiStride + c0_in_tg] = v0;
            tg_mem.SwiBuf[r_in_tg * kSwiStride + 32u + c0_in_tg] = v1;
          }
        }
      }
    }
  }

  if constexpr (FUSED_FWHT64_A8) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint g_col = (p_row_offset + n0_tg_base) >> 6u;
    const bool write_bf16 = ((p_flags & 1u) != 0u);
    for (uint rr = 0u; rr < 8u; ++rr) {
      const uint r = sg * 8u + rr;
      const uint row = rb_tg + r;
      if (row >= p_mp) continue;
      const ulong out_off = ulong(row) * ulong(out_stride) + ulong(p_row_offset + n0_tg_base);
      if (row >= p_rows) {
        reinterpret_cast<device char2 *>(qa_out + out_off)[lane] = char2(0);
        if (lane == 0u) {
          sa_out[ulong(g_col) * ulong(p_mp) + row] = 0.0f;
          ra_out[ulong(g_col) * ulong(p_mp) + row] = 0.0f;
        }
        continue;
      }
      const bfloat2 pair = reinterpret_cast<const threadgroup bfloat2 *>(tg_mem.SwiBuf + r * kSwiStride)[lane];
      if (write_bf16) {
        if (n0_tg_base + lane * 2u + 1u < p_n) {
          reinterpret_cast<device bfloat2 *>(y_bf16 + out_off)[lane] = pair;
        } else if (n0_tg_base + lane * 2u < p_n) {
          y_bf16[out_off + lane * 2u] = pair.x;
        }
      }
      float v0 = float(pair.x);
      float v1 = float(pair.y);
      omlx_m5_fwht64_sg32(v0, v1, lane);
      omlx_m5_quantize_pair_g64_f32(
          v0, v1, lane, qa_out + out_off,
          sa_out + ulong(g_col) * ulong(p_mp) + row,
          ra_out + ulong(g_col) * ulong(p_mp) + row);
    }
  } else {
    (void)qa_out; (void)sa_out; (void)ra_out; (void)p_mp; (void)p_flags; (void)lane;
  }
}

template <bool FUSED_FWHT64_A8>
inline void mxfp4_nax_swiglu_bm64_bn64_out64_sk1_impl(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_scales,
    device const uchar *u_codes,
    device const uchar *u_scales,
    device bfloat *y_bf16,
    device int8_t *qa_out,
    device float *sa_out,
    device float *ra_out,
    uint p_n,
    uint p_k,
    uint p_mp,
    uint p_rows,
    uint p_row_offset,
    uint p_out_stride,
    uint p_flags,
    threadgroup Mxfp4Out64TgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 grid_tgs) {
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint kSwiStride = 68u;
  constexpr uint BM_TG = 64u;
  constexpr uint OUT_N_TG = FUSED_FWHT64_A8 ? 64u : 32u;
  constexpr uint SM = FUSED_FWHT64_A8 ? 16u : 32u;
  constexpr uint OUT_SN = 16u;

  const uint2 pid_nm = mxfp4_swizzle_tg_group_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg_base = pid_nm.x * OUT_N_TG;
  const uint row_bound = FUSED_FWHT64_A8 ? p_mp : p_rows;
  if (rb_tg >= row_bound || n0_tg_base >= p_n) {
    return;
  }

  const uint out_stride = (p_out_stride == 0u) ? (p_row_offset + p_n) : p_out_stride;
  if constexpr (FUSED_FWHT64_A8) {
    if (rb_tg >= p_rows) {
      const uint g_col = (p_row_offset + n0_tg_base) >> 6u;
      for (uint r = sg; r < 64u; r += 8u) {
        const uint row = rb_tg + r;
        if (row < p_mp) {
          const ulong out_off = ulong(row) * ulong(out_stride) + ulong(p_row_offset + n0_tg_base);
          reinterpret_cast<device char2 *>(qa_out + out_off)[lane] = char2(0);
          if (lane == 0u) {
            sa_out[ulong(g_col) * ulong(p_mp) + row] = 0.0f;
            ra_out[ulong(g_col) * ulong(p_mp) + row] = 0.0f;
          }
        }
      }
      return;
    }
  }

  const uint tid = sg * 32u + lane;
  if (!mxfp4_has_apple10_nax_layout_t<bfloat>()) {
    mxfp4_nax_wide_sk1_swiglu_bf16_out64_apple9<FUSED_FWHT64_A8>(
        x, g_codes, g_scales, u_codes, u_scales, y_bf16, qa_out, sa_out, ra_out,
        p_n, p_k, p_mp, p_rows, p_row_offset, p_out_stride, p_flags,
        tg_mem, tid, lane, sg, rb_tg, n0_tg_base);
    return;
  }

  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const uint rb_sg = rb_tg + sg_m * SM;
  const uint blocks128 = p_k / 128u;
  const uint k_blocks = p_k / kBK;
  const uint col_limit = p_row_offset + p_n;
  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

  const bool do_w_load = (tid < 128u);
  const uint w_row = tid >> 1u;
  const uint col32 = tid & 1u;
  const bool is_up = ((w_row >> 4u) & 1u) != 0u;
  const uint w_col_in32 = ((w_row >> 5u) * 16u) + (w_row & 15u);
  threadgroup vec<bfloat, 8> *w_dst8 =
      reinterpret_cast<threadgroup vec<bfloat, 8> *>(tg_mem.Ws + w_row * kBKP + col32 * 32u);
  const threadgroup bfloat *ws_sg = tg_mem.Ws + (sg_n * 32u + fm) * kBKP + fn;

  const bool full_tile = (rb_tg + BM_TG <= p_rows) && (n0_tg_base + OUT_N_TG <= p_n);
  const bool aligned_out = ((out_stride & 3u) == 0u) && ((p_row_offset & 3u) == 0u);

  device const uchar *wc_base = nullptr;
  device const uchar *ws_base = nullptr;
  if (full_tile && do_w_load) {
    const uint gcol0 = p_row_offset + n0_tg_base + w_col_in32;
    const uint tile_n = gcol0 >> 4u;
    const uint rn = gcol0 & 15u;
    wc_base = (is_up ? u_codes : g_codes) + ulong(tile_n) * ulong(blocks128) * 1024ul + ulong(rn) * 64ul;
    ws_base = (is_up ? u_scales : g_scales) + ulong(tile_n) * ulong(blocks128) * 64ul + ulong(rn) * 4ul;
  }

  if constexpr (!FUSED_FWHT64_A8) {
    (void)qa_out; (void)sa_out; (void)ra_out; (void)p_mp; (void)p_flags;
    device const bfloat *x_sg = x + ulong(rb_sg + fm) * p_k + fn;
    const uint n0_tg = n0_tg_base;
    const uint n0_sg = n0_tg + sg_n * OUT_SN;
    const bool sg_active = (rb_sg < p_rows) && (n0_sg < p_n);
    const uint gcol_w = p_row_offset + n0_tg + w_col_in32;

    vec<float, 8> D00 = vec<float, 8>(0.0f), D01 = vec<float, 8>(0.0f);
    vec<float, 8> D10 = vec<float, 8>(0.0f), D11 = vec<float, 8>(0.0f);

    if (full_tile) {
      uint4 raw = reinterpret_cast<device const uint4 *>(wc_base)[col32];
      float sc = e8m0_to_float(ws_base[col32]);
      for (uint kb = 0u; kb < k_blocks; ++kb) {
        const uint k0 = kb * kBK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mxfp4_unpack_raw_32<bfloat>(raw, sc, w_dst8);
        if (kb + 1u < k_blocks) {
          const uint nkb = kb + 1u;
          const uint nblk = nkb >> 1u;
          const uint nsub = (nkb & 1u) * 2u + col32;
          raw = reinterpret_cast<device const uint4 *>(wc_base + ulong(nblk) * 1024ul)[nsub];
          sc = e8m0_to_float((ws_base + ulong(nblk) * 64ul)[nsub]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(disable)
        for (uint kk = 0u; kk < kBK; kk += 32u) {
          vec<bfloat, 8> a00, a01, a10, a11, b00, b01, b10, b11;
          #pragma clang loop unroll(full)
          for (uint r = 0u; r < 2u; ++r) {
            const bfloat4 va0 = *reinterpret_cast<device const bfloat4 *>(x_sg + ulong(r * 8u) * p_k + k0 + kk + 0u);
            const bfloat4 va1 = *reinterpret_cast<device const bfloat4 *>(x_sg + ulong(r * 8u) * p_k + k0 + kk + 16u);
            const bfloat4 va2 = *reinterpret_cast<device const bfloat4 *>(x_sg + ulong(16u + r * 8u) * p_k + k0 + kk + 0u);
            const bfloat4 va3 = *reinterpret_cast<device const bfloat4 *>(x_sg + ulong(16u + r * 8u) * p_k + k0 + kk + 16u);
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
          mxfp4_mma_16x32x16_fp<bfloat>(D00, D01, a00, b00, b10);
          mxfp4_mma_16x32x16_fp<bfloat>(D00, D01, a01, b01, b11);
          mxfp4_mma_16x32x16_fp<bfloat>(D10, D11, a10, b00, b10);
          mxfp4_mma_16x32x16_fp<bfloat>(D10, D11, a11, b01, b11);
        }
      }
    } else {
      for (uint kb = 0u; kb < k_blocks; ++kb) {
        const uint k0 = kb * kBK;
        const uint blk128 = kb >> 1u;
        const uint sub32 = (kb & 1u) * 2u + col32;
        uint4 raw;
        float sc;
        mxfp4_out64_fetch_raw_32_checked(
            is_up ? u_codes : g_codes, is_up ? u_scales : g_scales,
            gcol_w, col_limit, blocks128, blk128, sub32, raw, sc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mxfp4_unpack_raw_32<bfloat>(raw, sc, w_dst8);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
          #pragma clang loop unroll(disable)
          for (uint kk = 0u; kk < kBK; kk += 32u) {
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
            {
              vec<bfloat, 8> a0, a1;
              #pragma clang loop unroll(full)
              for (uint r = 0u; r < 2u; ++r) {
                const uint row = rb_sg + fm + r * 8u;
                const bfloat4 va0 = (row < p_rows)
                    ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p_k + k0 + kk + fn)
                    : bfloat4(bfloat(0.0f));
                const bfloat4 va1 = (row < p_rows)
                    ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p_k + k0 + kk + 16u + fn)
                    : bfloat4(bfloat(0.0f));
                #pragma clang loop unroll(full)
                for (uint c = 0u; c < 4u; ++c) { a0[r * 4u + c] = va0[c]; a1[r * 4u + c] = va1[c]; }
              }
              mxfp4_mma_16x32x16_fp<bfloat>(D00, D01, a0, b00, b10);
              mxfp4_mma_16x32x16_fp<bfloat>(D00, D01, a1, b01, b11);
            }
            if (rb_sg + 16u < p_rows) {
              vec<bfloat, 8> a0, a1;
              #pragma clang loop unroll(full)
              for (uint r = 0u; r < 2u; ++r) {
                const uint row = rb_sg + 16u + fm + r * 8u;
                const bfloat4 va0 = (row < p_rows)
                    ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p_k + k0 + kk + fn)
                    : bfloat4(bfloat(0.0f));
                const bfloat4 va1 = (row < p_rows)
                    ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p_k + k0 + kk + 16u + fn)
                    : bfloat4(bfloat(0.0f));
                #pragma clang loop unroll(full)
                for (uint c = 0u; c < 4u; ++c) { a0[r * 4u + c] = va0[c]; a1[r * 4u + c] = va1[c]; }
              }
              mxfp4_mma_16x32x16_fp<bfloat>(D10, D11, a0, b00, b10);
              mxfp4_mma_16x32x16_fp<bfloat>(D10, D11, a1, b01, b11);
            }
          }
        }
      }
    }

    if (sg_active) {
      const uint lcol = n0_sg + fn;
      const uint gcol = p_row_offset + lcol;
      if (lcol < p_n) {
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint row0 = rb_sg + fm + h * 8u;
          const uint row1 = row0 + 16u;
          const uint e = h * 4u;
          if (row0 < p_rows) {
            bfloat4 ob;
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) {
              ob[j] = bfloat(nax_silu_f32(float(bfloat(D00[e + j]))) * float(bfloat(D01[e + j])));
            }
            if (aligned_out && lcol + 4u <= p_n) {
              *reinterpret_cast<device bfloat4 *>(y_bf16 + ulong(row0) * ulong(out_stride) + ulong(gcol)) = ob;
            } else {
              #pragma clang loop unroll(full)
              for (uint j = 0u; j < 4u; ++j)
                if (lcol + j < p_n) y_bf16[ulong(row0) * ulong(out_stride) + ulong(gcol + j)] = ob[j];
            }
          }
          if (row1 < p_rows) {
            bfloat4 ob;
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) {
              ob[j] = bfloat(nax_silu_f32(float(bfloat(D10[e + j]))) * float(bfloat(D11[e + j])));
            }
            if (aligned_out && lcol + 4u <= p_n) {
              *reinterpret_cast<device bfloat4 *>(y_bf16 + ulong(row1) * ulong(out_stride) + ulong(gcol)) = ob;
            } else {
              #pragma clang loop unroll(full)
              for (uint j = 0u; j < 4u; ++j)
                if (lcol + j < p_n) y_bf16[ulong(row1) * ulong(out_stride) + ulong(gcol + j)] = ob[j];
            }
          }
        }
      }
    }
    return;
  }

  // Mode B (256 thr, WM=4, WN=2 -> SM=16, only 16 acc regs + 4 swi0 regs -> 0 B spills, 1-phase SwiBuf!)
  bfloat4 swi_save[2];
  #pragma clang loop unroll(disable)
  for (uint hn = 0u; hn < 2u; ++hn) {
    const uint n0_tg = n0_tg_base + hn * 32u;
    const uint n0_sg = n0_tg + sg_n * OUT_SN;
    const bool sg_active = (rb_sg < p_rows) && (n0_sg < p_n);
    const uint gcol_w = p_row_offset + n0_tg + w_col_in32;
    device const uchar *wc_row = wc_base + ulong(hn * 2u) * ulong(blocks128) * 1024ul;
    device const uchar *ws_row = ws_base + ulong(hn * 2u) * ulong(blocks128) * 64ul;

    vec<float, 8> D0 = vec<float, 8>(0.0f), D1 = vec<float, 8>(0.0f);

    for (uint kb = 0u; kb < k_blocks; ++kb) {
      const uint k0 = kb * kBK;
      const uint blk128 = kb >> 1u;
      const uint sub32 = (kb & 1u) * 2u + col32;
      uint4 raw = uint4(0u);
      float sc = 0.0f;
      if (do_w_load) {
        if (full_tile) {
          raw = reinterpret_cast<device const uint4 *>(wc_row + ulong(blk128) * 1024ul)[sub32];
          sc = e8m0_to_float((ws_row + ulong(blk128) * 64ul)[sub32]);
        } else {
          mxfp4_out64_fetch_raw_32_checked(
              is_up ? u_codes : g_codes, is_up ? u_scales : g_scales,
              gcol_w, col_limit, blocks128, blk128, sub32, raw, sc);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (do_w_load) {
        mxfp4_unpack_raw_32<bfloat>(raw, sc, w_dst8);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_active) {
        #pragma clang loop unroll(disable)
        for (uint kk = 0u; kk < kBK; kk += 32u) {
          vec<bfloat, 8> a0, a1, b00, b01, b10, b11;
          #pragma clang loop unroll(full)
          for (uint r = 0u; r < 2u; ++r) {
            const uint row = rb_sg + fm + r * 8u;
            const bfloat4 va0 = (full_tile || row < p_rows)
                ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p_k + k0 + kk + fn)
                : bfloat4(bfloat(0.0f));
            const bfloat4 va1 = (full_tile || row < p_rows)
                ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p_k + k0 + kk + 16u + fn)
                : bfloat4(bfloat(0.0f));
            const bfloat4 vb0 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 0u);
            const bfloat4 vb1 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + r * 8u * kBKP + kk + 16u);
            const bfloat4 vb2 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 0u);
            const bfloat4 vb3 = *reinterpret_cast<const threadgroup bfloat4 *>(ws_sg + (16u + r * 8u) * kBKP + kk + 16u);
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) {
              a0[r * 4u + c] = va0[c]; a1[r * 4u + c] = va1[c];
              b00[r * 4u + c] = vb0[c]; b01[r * 4u + c] = vb1[c];
              b10[r * 4u + c] = vb2[c]; b11[r * 4u + c] = vb3[c];
            }
          }
          mxfp4_mma_16x32x16_fp<bfloat>(D0, D1, a0, b00, b10);
          mxfp4_mma_16x32x16_fp<bfloat>(D0, D1, a1, b01, b11);
        }
      }
    }

    const uint lcol = n0_sg + fn;
    if (hn == 0u) {
      #pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const uint row = rb_sg + fm + h * 8u;
        bfloat4 out_b = bfloat4(bfloat(0.0f));
        if (row < p_rows && lcol < p_n) {
          const uint e = h * 4u;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            if (lcol + j < p_n) {
              out_b[j] = bfloat(nax_silu_f32(float(bfloat(D0[e + j]))) * float(bfloat(D1[e + j])));
            }
          }
        }
        swi_save[h] = out_b;
      }
    } else {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const uint r_in_tg = sg_m * SM + fm + h * 8u;
        const uint row = rb_tg + r_in_tg;
        bfloat4 out_b = bfloat4(bfloat(0.0f));
        if (row < p_rows && lcol < p_n) {
          const uint e = h * 4u;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            if (lcol + j < p_n) {
              out_b[j] = bfloat(nax_silu_f32(float(bfloat(D0[e + j]))) * float(bfloat(D1[e + j])));
            }
          }
        }
        *reinterpret_cast<threadgroup bfloat4 *>(tg_mem.SwiBuf + r_in_tg * kSwiStride + sg_n * 16u + fn) = swi_save[h];
        *reinterpret_cast<threadgroup bfloat4 *>(tg_mem.SwiBuf + r_in_tg * kSwiStride + 32u + sg_n * 16u + fn) = out_b;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint g_col = (p_row_offset + n0_tg_base) >> 6u;
  const bool write_bf16 = ((p_flags & 1u) != 0u);
  #pragma clang loop unroll(disable)
  for (uint rr = 0u; rr < 8u; ++rr) {
    const uint r = sg * 8u + rr;
    const uint row = rb_tg + r;
    if (row >= p_mp) continue;
    const ulong out_off = ulong(row) * ulong(out_stride) + ulong(p_row_offset + n0_tg_base);
    if (row >= p_rows) {
      reinterpret_cast<device char2 *>(qa_out + out_off)[lane] = char2(0);
      if (lane == 0u) {
        sa_out[ulong(g_col) * ulong(p_mp) + row] = 0.0f;
        ra_out[ulong(g_col) * ulong(p_mp) + row] = 0.0f;
      }
      continue;
    }
    const bfloat2 pair = reinterpret_cast<const threadgroup bfloat2 *>(tg_mem.SwiBuf + r * kSwiStride)[lane];
    if (write_bf16) {
      if (n0_tg_base + lane * 2u + 1u < p_n) {
        reinterpret_cast<device bfloat2 *>(y_bf16 + out_off)[lane] = pair;
      } else if (n0_tg_base + lane * 2u < p_n) {
        y_bf16[out_off + lane * 2u] = pair.x;
      }
    }
    float v0 = float(pair.x);
    float v1 = float(pair.y);
    omlx_m5_fwht64_sg32(v0, v1, lane);
    omlx_m5_quantize_pair_g64_f32(
        v0, v1, lane, qa_out + out_off,
        sa_out + ulong(g_col) * ulong(p_mp) + row,
        ra_out + ulong(g_col) * ulong(p_mp) + row);
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_nax_swiglu_bf16_bm64_bn64_out64_sk1(
    device const bfloat *x [[buffer(0)]],
    device const uchar *g_codes [[buffer(1)]],
    device const uchar *g_scales [[buffer(2)]],
    device const uchar *u_codes [[buffer(3)]],
    device const uchar *u_scales [[buffer(4)]],
    device bfloat *y_bf16 [[buffer(5)]],
    constant Mxfp4NaxSliceParams &p [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup Mxfp4Out64TgStorage tg_mem;
  mxfp4_nax_swiglu_bm64_bn64_out64_sk1_impl<false>(
      x, g_codes, g_scales, u_codes, u_scales, y_bf16, nullptr, nullptr, nullptr,
      p.n, p.k, p.rows, p.rows, p.row_offset, p.out_stride, 0u,
      tg_mem, lane, sg, tg, grid_tgs);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_mxfp4_nax_swiglu_bf16_bm64_bn64_out64_sk1_fwht64_a8(
    device const bfloat *x [[buffer(0)]],
    device const uchar *g_codes [[buffer(1)]],
    device const uchar *g_scales [[buffer(2)]],
    device const uchar *u_codes [[buffer(3)]],
    device const uchar *u_scales [[buffer(4)]],
    device int8_t *qa_out [[buffer(5)]],
    device float *sa_out [[buffer(6)]],
    device float *ra_out [[buffer(7)]],
    device bfloat *y_bf16 [[buffer(8)]],
    constant Mxfp4NaxOut64A8Params &p [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup Mxfp4Out64TgStorage tg_mem;
  mxfp4_nax_swiglu_bm64_bn64_out64_sk1_impl<true>(
      x, g_codes, g_scales, u_codes, u_scales, y_bf16, qa_out, sa_out, ra_out,
      p.n, p.k, p.mp, p.rows, p.row_offset, p.out_stride, p.flags,
      tg_mem, lane, sg, tg, grid_tgs);
}

template <uint BN_STORED>
inline void mxfp4_nax_bf16_wide_sk1_linear_slice_impl(
    device const bfloat *x,
    device const uchar *codes,
    device const uchar *scales,
    device bfloat *y,
    constant Mxfp4NaxSliceParams &p,
    threadgroup bfloat *Ws,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 grid_tgs) {
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint SM = 32u;
  constexpr uint SN = 32u;
  constexpr uint BM_TG = 64u;
  constexpr uint BN_TG = 64u;

  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  const uint2 pid_nm = mxfp4_swizzle_tg_group_m8(tg.xy, grid_tgs.x, grid_tgs.y);
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
  const bool sg_active = (sg_m < 2u) && (sg_n < 2u) && (rb_sg < p.rows) && (n0_sg < p.n);
  const uint blocks128 = p.k / 128u;
  const uint k_blocks = p.k / kBK;
  const uint col_limit = p.row_offset + p.n;
  const uint w_row = tid >> 1u;
  const uint col32 = tid & 1u;
  const uint gcol_w = p.row_offset + n0_tg + w_row;
  threadgroup vec<bfloat, 8> *w_dst8 =
      reinterpret_cast<threadgroup vec<bfloat, 8> *>(Ws + w_row * kBKP + col32 * 32u);

  if (!mxfp4_has_apple10_nax_layout_t<bfloat>()) {
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
    for (uint kb = 0u; kb < k_blocks; ++kb) {
      const uint k_off = kb * kBK;
      uint4 raw;
      float sc;
      mxfp4_fetch_raw_32<BN_STORED>(
          codes, scales, gcol_w, col_limit, blocks128, kb >> 1u,
          (kb & 1u) * 2u + col32, raw, sc);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mxfp4_unpack_raw_32<bfloat>(raw, sc, w_dst8);
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
  const threadgroup bfloat *ws_sg = Ws + (sg_n * SN + fm) * kBKP + fn;
  for (uint kb = 0u; kb < k_blocks; ++kb) {
    const uint k_off = kb * kBK;
    uint4 raw;
    float sc;
    mxfp4_fetch_raw_32<BN_STORED>(
        codes, scales, gcol_w, col_limit, blocks128, kb >> 1u,
        (kb & 1u) * 2u + col32, raw, sc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mxfp4_unpack_raw_32<bfloat>(raw, sc, w_dst8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_active) {
      #pragma clang loop unroll(disable)
      for (uint kk = 0u; kk < kBK; kk += 32u) {
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
            mxfp4_mma_16x32x16_fp<bfloat>(D[mm][0], D[mm][1], a0, b00, b10);
            mxfp4_mma_16x32x16_fp<bfloat>(D[mm][0], D[mm][1], a1, b01, b11);
          }
        }
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

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_nax_bf16_bm64_bn64_bns16_slice_sk1(
    device const bfloat *x [[buffer(0)]],
    device const uchar *codes [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device bfloat *y [[buffer(3)]],
    constant Mxfp4NaxSliceParams &p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  alignas(16) threadgroup bfloat Ws[64u * 72u];
  mxfp4_nax_bf16_wide_sk1_linear_slice_impl<16u>(x, codes, scales, y, p, Ws, lane, sg, tg, grid_tgs);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_nax_bf16_bm64_bn64_bns32_slice_sk1(
    device const bfloat *x [[buffer(0)]],
    device const uchar *codes [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device bfloat *y [[buffer(3)]],
    constant Mxfp4NaxSliceParams &p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  alignas(16) threadgroup bfloat Ws[64u * 72u];
  mxfp4_nax_bf16_wide_sk1_linear_slice_impl<32u>(x, codes, scales, y, p, Ws, lane, sg, tg, grid_tgs);
}

// ============================================================================
// Native Apple10 M5 NAX W4A4 (FP4 E2M1+UE8M0 activation x FP4 E2M1+UE8M0 weight)
// For opt-in large-T prefill / high-throughput routes (BM=16 * RF, BN=16/32, BK=128).
// ============================================================================

kernel void omlx_m5_bf16_to_mxfp4_a4_bm16_bk128(
    device const bfloat *src [[buffer(0)]],
    device uchar *dst_codes [[buffer(1)]],
    device uchar *dst_scales [[buffer(2)]],
    constant Mxfp4CastParams &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g = tg.x;
  const uint row = tg.y;
  const uint groups = p.k / 32u;
  if (g >= groups || row >= p.mp) {
    return;
  }
  const uint k_blocks = p.k / 128u;
  const ulong tile_idx = ulong(row / 16u) * ulong(k_blocks) + ulong(g / 4u);
  const uint rm = row % 16u;
  const uint sub_g = g % 4u;
  device uchar *c_out =
      dst_codes + tile_idx * 1024ul + ulong(rm) * 64ul + ulong(sub_g) * 16ul;
  device uchar *s_out =
      dst_scales + tile_idx * 64ul + ulong(rm) * 4ul + ulong(sub_g);

  if (row >= p.rows) {
    if ((lane & 1u) == 0u) {
      c_out[lane >> 1u] = 0u;
    }
    if (lane == 0u) {
      *s_out = 0x7fu;
    }
    return;
  }

  const float v = float(src[ulong(row) * p.k + g * 32u + lane]);
  const float maximum = simd_max(fabs(v));
  const uchar sc = nax_encode_e8m0_round_up(maximum / 6.0f);
  const float scale = nax_decode_e8m0(sc);
  const float inv_scale = scale > 0.0f ? (1.0f / scale) : 0.0f;
  const uchar c_lo = nax_encode_e2m1(v * inv_scale);
  const uchar c_hi = simd_shuffle_down(c_lo, 1u);
  if ((lane & 1u) == 0u) {
    c_out[lane >> 1u] = uchar(c_lo | (c_hi << 4u));
  }
  if (lane == 0u) {
    *s_out = sc;
  }
}

// Fused FWHT-64 + BF16 -> MXFP4 E2M1 + UE8M0 quantization (1 SG = 64 elements = 2 x 32-elem sub-groups, 0 B SRAM).
[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_bf16_fwht64_to_mxfp4_a4_bm16_bk128(
    device const bfloat *src [[buffer(0)]],
    device uchar *dst_codes [[buffer(1)]],
    device uchar *dst_scales [[buffer(2)]],
    constant Mxfp4CastParams &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g64 = tg.x;
  const uint row = tg.y;
  const uint groups64 = p.k / 64u;
  if (g64 >= groups64 || row >= p.mp) {
    return;
  }
  const uint k_blocks = p.k / 128u;
  const ulong tile_idx = ulong(row / 16u) * ulong(k_blocks) + ulong(g64 >> 1u);
  const uint rm = row % 16u;
  const uint half64 = g64 & 1u;
  const uint sub32_idx = half64 * 2u + (lane >> 4u);
  device uchar *c_out =
      dst_codes + tile_idx * 1024ul + ulong(rm) * 64ul + ulong(sub32_idx) * 16ul;
  device uchar *s_out =
      dst_scales + tile_idx * 64ul + ulong(rm) * 4ul + ulong(sub32_idx);

  if (row >= p.rows) {
    c_out[lane & 15u] = 0u;
    if ((lane & 15u) == 0u) {
      *s_out = 0x7fu;
    }
    return;
  }

  const bfloat2 x2 = reinterpret_cast<device const bfloat2 *>(
      src + ulong(row) * p.k + ulong(g64) * 64ul)[lane];
  float v0 = float(x2.x);
  float v1 = float(x2.y);
  omlx_m5_fwht64_sg32(v0, v1, lane);

  // 16-lane butterfly amax within each 32-element half (lanes 0..15 and lanes 16..31).
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

template <uint SK, uint kCap, typename CoopT>
inline void mxfp4_w4a4_reduce_store_linear_slice(
    thread CoopT &prod,
    uint rb_slice,
    uint n0,
    uint sg,
    uint lane,
    threadgroup float *parts,
    device bfloat *y,
    constant Mxfp4NaxParams &p,
    bool need_pre_barrier) {
  const uint elems = prod.get_capacity();
  if constexpr (SK > 1u) {
    if (need_pre_barrier) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg > 0u) {
      for (uint i = 0; i < elems; ++i) {
        parts[((sg - 1u) * kCap + i) * 32u + lane] = prod[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      for (uint s = 1u; s < SK; ++s) {
        for (uint i = 0; i < elems; ++i) {
          prod[i] += parts[((s - 1u) * kCap + i) * 32u + lane];
        }
      }
    }
  }
  if (sg == 0u) {
    for (uint i = 0; i < elems; ++i) {
      if (prod.is_valid_element(i)) {
        const auto idx = prod.get_multidimensional_index(i);
        const uint col = n0 + uint(idx[0]);
        const uint row = rb_slice + uint(idx[1]);
        if (row < p.rows && col < p.n) {
          y[ulong(row) * p.n + col] = bfloat(prod[i]);
        }
      }
    }
  }
}

template <uint BN, uint SK, uint RF = 1u>
inline void mxfp4_w4a4_nax_linear_impl(
    device const uchar *a_codes,
    device const uchar *a_scales,
    device const uchar *b_codes,
    device const uchar *b_scales,
    device bfloat *y,
    constant Mxfp4NaxParams &p,
    threadgroup float *parts,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM = 16u;
  constexpr uint BK = 128u;
  constexpr uint kCap = (16u * BN) / 32u;
  const uint rb = tg.y * (BM * RF);
  const uint n0 = tg.x * BN;
  if (rb >= p.rows || n0 >= p.n) {
    return;
  }

  const uint blocks = p.k / BK;
  const uint b0 = (sg * blocks) / SK;
  const uint b1 = ((sg + 1u) * blocks) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  constexpr ulong b_tile_code_bytes = ulong(BN) * (BK / 2u);
  constexpr ulong b_tile_scale_bytes = ulong(BN) * (BK / 32u);
  const ulong a_tile_base0 = ulong(tg.y * RF) * blocks;
  const ulong b_tile_base = ulong(tg.x) * blocks;

  Mxfp4WeightTensor a_init((device uchar *)a_codes, dextents<int32_t, 2>(BK, BM),
                           Mxfp4ScaleTag((device uchar *)a_scales));
  Mxfp4WeightTensor b_init((device uchar *)b_codes, dextents<int32_t, 2>(BK, BN),
                           Mxfp4ScaleTag((device uchar *)b_scales));
  auto prod0 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto prod1 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto prod2 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto prod3 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  const uint elems = prod0.get_capacity();
  for (uint i = 0; i < elems; ++i) {
    prod0[i] = 0.0f;
    if constexpr (RF > 1u) prod1[i] = 0.0f;
    if constexpr (RF > 2u) prod2[i] = 0.0f;
    if constexpr (RF > 3u) prod3[i] = 0.0f;
  }

  for (uint blk = b0; blk < b1; ++blk) {
    const ulong b_idx = b_tile_base + blk;
    Mxfp4WeightTensor b((device uchar *)(b_codes + b_idx * b_tile_code_bytes),
                        dextents<int32_t, 2>(BK, BN),
                        Mxfp4ScaleTag((device uchar *)(b_scales + b_idx * b_tile_scale_bytes)));
    {
      const ulong a_idx = a_tile_base0 + blk;
      Mxfp4WeightTensor a0((device uchar *)(a_codes + a_idx * 1024ul),
                           dextents<int32_t, 2>(BK, BM),
                           Mxfp4ScaleTag((device uchar *)(a_scales + a_idx * 64ul)));
      op.run(a0, b, prod0);
    }
    if constexpr (RF > 1u) {
      if (rb + 16u < p.rows) {
        const ulong a_idx = a_tile_base0 + blocks + blk;
        Mxfp4WeightTensor a1((device uchar *)(a_codes + a_idx * 1024ul),
                             dextents<int32_t, 2>(BK, BM),
                             Mxfp4ScaleTag((device uchar *)(a_scales + a_idx * 64ul)));
        op.run(a1, b, prod1);
      }
    }
    if constexpr (RF > 2u) {
      if (rb + 32u < p.rows) {
        const ulong a_idx = a_tile_base0 + 2ul * blocks + blk;
        Mxfp4WeightTensor a2((device uchar *)(a_codes + a_idx * 1024ul),
                             dextents<int32_t, 2>(BK, BM),
                             Mxfp4ScaleTag((device uchar *)(a_scales + a_idx * 64ul)));
        op.run(a2, b, prod2);
      }
    }
    if constexpr (RF > 3u) {
      if (rb + 48u < p.rows) {
        const ulong a_idx = a_tile_base0 + 3ul * blocks + blk;
        Mxfp4WeightTensor a3((device uchar *)(a_codes + a_idx * 1024ul),
                             dextents<int32_t, 2>(BK, BM),
                             Mxfp4ScaleTag((device uchar *)(a_scales + a_idx * 64ul)));
        op.run(a3, b, prod3);
      }
    }
  }

  mxfp4_w4a4_reduce_store_linear_slice<SK, kCap>(prod0, rb, n0, sg, lane, parts, y, p, false);
  if constexpr (RF > 1u) {
    if (rb + 16u < p.rows) {
      mxfp4_w4a4_reduce_store_linear_slice<SK, kCap>(prod1, rb + 16u, n0, sg, lane, parts, y, p, true);
    }
  }
  if constexpr (RF > 2u) {
    if (rb + 32u < p.rows) {
      mxfp4_w4a4_reduce_store_linear_slice<SK, kCap>(prod2, rb + 32u, n0, sg, lane, parts, y, p, true);
    }
  }
  if constexpr (RF > 3u) {
    if (rb + 48u < p.rows) {
      mxfp4_w4a4_reduce_store_linear_slice<SK, kCap>(prod3, rb + 48u, n0, sg, lane, parts, y, p, true);
    }
  }
}

template <uint SK, uint kCap, typename CoopT>
inline void mxfp4_w4a4_reduce_store_swiglu_slice(
    thread CoopT &pg,
    thread CoopT &pu,
    uint rb_slice,
    uint n0,
    uint sg,
    uint lane,
    threadgroup float *parts,
    device bfloat *y,
    constant Mxfp4NaxParams &p,
    bool need_pre_barrier) {
  const uint elems = pg.get_capacity();
  if constexpr (SK > 1u) {
    if (need_pre_barrier) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg > 0u) {
      for (uint i = 0; i < elems; ++i) {
        parts[((sg - 1u) * (2u * kCap) + i) * 32u + lane] = pg[i];
        parts[((sg - 1u) * (2u * kCap) + kCap + i) * 32u + lane] = pu[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      for (uint s = 1u; s < SK; ++s) {
        for (uint i = 0; i < elems; ++i) {
          pg[i] += parts[((s - 1u) * (2u * kCap) + i) * 32u + lane];
          pu[i] += parts[((s - 1u) * (2u * kCap) + kCap + i) * 32u + lane];
        }
      }
    }
  }
  if (sg == 0u) {
    for (uint i = 0; i < elems; ++i) {
      if (pg.is_valid_element(i)) {
        const auto idx = pg.get_multidimensional_index(i);
        const uint col = n0 + uint(idx[0]);
        const uint row = rb_slice + uint(idx[1]);
        if (row < p.rows && col < p.n) {
          const float gv = float(bfloat(pg[i]));
          const float uv = float(bfloat(pu[i]));
          y[ulong(row) * p.n + col] = bfloat(nax_silu_f32(gv) * uv);
        }
      }
    }
  }
}

template <uint BN, uint SK, uint RF = 1u>
inline void mxfp4_w4a4_nax_swiglu_impl(
    device const uchar *a_codes,
    device const uchar *a_scales,
    device const uchar *g_codes,
    device const uchar *g_scales,
    device const uchar *u_codes,
    device const uchar *u_scales,
    device bfloat *y,
    constant Mxfp4NaxParams &p,
    threadgroup float *parts,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM = 16u;
  constexpr uint BK = 128u;
  constexpr uint kCap = (16u * BN) / 32u;
  const uint rb = tg.y * (BM * RF);
  const uint n0 = tg.x * BN;
  if (rb >= p.rows || n0 >= p.n) {
    return;
  }

  const uint blocks = p.k / BK;
  const uint b0 = (sg * blocks) / SK;
  const uint b1 = ((sg + 1u) * blocks) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  constexpr ulong b_tile_code_bytes = ulong(BN) * (BK / 2u);
  constexpr ulong b_tile_scale_bytes = ulong(BN) * (BK / 32u);
  const ulong a_tile_base0 = ulong(tg.y * RF) * blocks;
  const ulong b_tile_base = ulong(tg.x) * blocks;

  Mxfp4WeightTensor a_init((device uchar *)a_codes, dextents<int32_t, 2>(BK, BM),
                           Mxfp4ScaleTag((device uchar *)a_scales));
  Mxfp4WeightTensor b_init((device uchar *)g_codes, dextents<int32_t, 2>(BK, BN),
                           Mxfp4ScaleTag((device uchar *)g_scales));
  auto pg0 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu0 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pg1 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu1 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pg2 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu2 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pg3 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu3 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  const uint elems = pg0.get_capacity();
  for (uint i = 0; i < elems; ++i) {
    pg0[i] = 0.0f; pu0[i] = 0.0f;
    if constexpr (RF > 1u) { pg1[i] = 0.0f; pu1[i] = 0.0f; }
    if constexpr (RF > 2u) { pg2[i] = 0.0f; pu2[i] = 0.0f; }
    if constexpr (RF > 3u) { pg3[i] = 0.0f; pu3[i] = 0.0f; }
  }

  for (uint blk = b0; blk < b1; ++blk) {
    const ulong b_idx = b_tile_base + blk;
    Mxfp4WeightTensor bg((device uchar *)(g_codes + b_idx * b_tile_code_bytes),
                         dextents<int32_t, 2>(BK, BN),
                         Mxfp4ScaleTag((device uchar *)(g_scales + b_idx * b_tile_scale_bytes)));
    Mxfp4WeightTensor bu((device uchar *)(u_codes + b_idx * b_tile_code_bytes),
                         dextents<int32_t, 2>(BK, BN),
                         Mxfp4ScaleTag((device uchar *)(u_scales + b_idx * b_tile_scale_bytes)));
    {
      const ulong a_idx = a_tile_base0 + blk;
      Mxfp4WeightTensor a0((device uchar *)(a_codes + a_idx * 1024ul),
                           dextents<int32_t, 2>(BK, BM),
                           Mxfp4ScaleTag((device uchar *)(a_scales + a_idx * 64ul)));
      op.run(a0, bg, pg0);
      op.run(a0, bu, pu0);
    }
    if constexpr (RF > 1u) {
      if (rb + 16u < p.rows) {
        const ulong a_idx = a_tile_base0 + blocks + blk;
        Mxfp4WeightTensor a1((device uchar *)(a_codes + a_idx * 1024ul),
                             dextents<int32_t, 2>(BK, BM),
                             Mxfp4ScaleTag((device uchar *)(a_scales + a_idx * 64ul)));
        op.run(a1, bg, pg1);
        op.run(a1, bu, pu1);
      }
    }
    if constexpr (RF > 2u) {
      if (rb + 32u < p.rows) {
        const ulong a_idx = a_tile_base0 + 2ul * blocks + blk;
        Mxfp4WeightTensor a2((device uchar *)(a_codes + a_idx * 1024ul),
                             dextents<int32_t, 2>(BK, BM),
                             Mxfp4ScaleTag((device uchar *)(a_scales + a_idx * 64ul)));
        op.run(a2, bg, pg2);
        op.run(a2, bu, pu2);
      }
    }
    if constexpr (RF > 3u) {
      if (rb + 48u < p.rows) {
        const ulong a_idx = a_tile_base0 + 3ul * blocks + blk;
        Mxfp4WeightTensor a3((device uchar *)(a_codes + a_idx * 1024ul),
                             dextents<int32_t, 2>(BK, BM),
                             Mxfp4ScaleTag((device uchar *)(a_scales + a_idx * 64ul)));
        op.run(a3, bg, pg3);
        op.run(a3, bu, pu3);
      }
    }
  }

  mxfp4_w4a4_reduce_store_swiglu_slice<SK, kCap>(pg0, pu0, rb, n0, sg, lane, parts, y, p, false);
  if constexpr (RF > 1u) {
    if (rb + 16u < p.rows) {
      mxfp4_w4a4_reduce_store_swiglu_slice<SK, kCap>(pg1, pu1, rb + 16u, n0, sg, lane, parts, y, p, true);
    }
  }
  if constexpr (RF > 2u) {
    if (rb + 32u < p.rows) {
      mxfp4_w4a4_reduce_store_swiglu_slice<SK, kCap>(pg2, pu2, rb + 32u, n0, sg, lane, parts, y, p, true);
    }
  }
  if constexpr (RF > 3u) {
    if (rb + 48u < p.rows) {
      mxfp4_w4a4_reduce_store_swiglu_slice<SK, kCap>(pg3, pu3, rb + 48u, n0, sg, lane, parts, y, p, true);
    }
  }
}

#define OMLX_M5_MXFP4_W4A4_NAX(BN, SK)                                         \
  kernel void omlx_m5_mxfp4_w4a4_nax_n##BN##_sk##SK(                           \
      device const uchar *a_codes [[buffer(0)]],                               \
      device const uchar *a_scales [[buffer(1)]],                              \
      device const uchar *b_codes [[buffer(2)]],                               \
      device const uchar *b_scales [[buffer(3)]],                              \
      device bfloat *y [[buffer(4)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(5)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * ((16u * BN) / 32u) * 32u]; \
    mxfp4_w4a4_nax_linear_impl<BN, SK, 1u>(a_codes, a_scales, b_codes,         \
                                           b_scales, y, p, parts, lane, sg,    \
                                           tg);                                \
  }                                                                            \
  kernel void omlx_m5_mxfp4_w4a4_nax_n##BN##_sk##SK##_rf2(                     \
      device const uchar *a_codes [[buffer(0)]],                               \
      device const uchar *a_scales [[buffer(1)]],                              \
      device const uchar *b_codes [[buffer(2)]],                               \
      device const uchar *b_scales [[buffer(3)]],                              \
      device bfloat *y [[buffer(4)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(5)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * ((16u * BN) / 32u) * 32u]; \
    mxfp4_w4a4_nax_linear_impl<BN, SK, 2u>(a_codes, a_scales, b_codes,         \
                                           b_scales, y, p, parts, lane, sg,    \
                                           tg);                                \
  }                                                                            \
  kernel void omlx_m5_mxfp4_w4a4_nax_n##BN##_sk##SK##_rf4(                     \
      device const uchar *a_codes [[buffer(0)]],                               \
      device const uchar *a_scales [[buffer(1)]],                              \
      device const uchar *b_codes [[buffer(2)]],                               \
      device const uchar *b_scales [[buffer(3)]],                              \
      device bfloat *y [[buffer(4)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(5)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * ((16u * BN) / 32u) * 32u]; \
    mxfp4_w4a4_nax_linear_impl<BN, SK, 4u>(a_codes, a_scales, b_codes,         \
                                           b_scales, y, p, parts, lane, sg,    \
                                           tg);                                \
  }

#define OMLX_M5_MXFP4_W4A4_NAX_SWIGLU(SK)                                      \
  kernel void omlx_m5_mxfp4_w4a4_nax_swiglu_n16_sk##SK(                        \
      device const uchar *a_codes [[buffer(0)]],                               \
      device const uchar *a_scales [[buffer(1)]],                              \
      device const uchar *g_codes [[buffer(2)]],                               \
      device const uchar *g_scales [[buffer(3)]],                              \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_scales [[buffer(5)]],                              \
      device bfloat *y [[buffer(6)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(7)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 16u * 32u];             \
    mxfp4_w4a4_nax_swiglu_impl<16u, SK, 1u>(a_codes, a_scales, g_codes,        \
                                            g_scales, u_codes, u_scales, y, p, \
                                            parts, lane, sg, tg);              \
  }                                                                            \
  kernel void omlx_m5_mxfp4_w4a4_nax_swiglu_n16_sk##SK##_rf2(                  \
      device const uchar *a_codes [[buffer(0)]],                               \
      device const uchar *a_scales [[buffer(1)]],                              \
      device const uchar *g_codes [[buffer(2)]],                               \
      device const uchar *g_scales [[buffer(3)]],                              \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_scales [[buffer(5)]],                              \
      device bfloat *y [[buffer(6)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(7)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 16u * 32u];             \
    mxfp4_w4a4_nax_swiglu_impl<16u, SK, 2u>(a_codes, a_scales, g_codes,        \
                                            g_scales, u_codes, u_scales, y, p, \
                                            parts, lane, sg, tg);              \
  }                                                                            \
  kernel void omlx_m5_mxfp4_w4a4_nax_swiglu_n16_sk##SK##_rf4(                  \
      device const uchar *a_codes [[buffer(0)]],                               \
      device const uchar *a_scales [[buffer(1)]],                              \
      device const uchar *g_codes [[buffer(2)]],                               \
      device const uchar *g_scales [[buffer(3)]],                              \
      device const uchar *u_codes [[buffer(4)]],                               \
      device const uchar *u_scales [[buffer(5)]],                              \
      device bfloat *y [[buffer(6)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(7)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? SK - 1u : 1u) * 16u * 32u];             \
    mxfp4_w4a4_nax_swiglu_impl<16u, SK, 4u>(a_codes, a_scales, g_codes,        \
                                            g_scales, u_codes, u_scales, y, p, \
                                            parts, lane, sg, tg);              \
  }

OMLX_M5_MXFP4_W4A4_NAX(16, 1)
OMLX_M5_MXFP4_W4A4_NAX(16, 2)
OMLX_M5_MXFP4_W4A4_NAX(16, 4)
OMLX_M5_MXFP4_W4A4_NAX(16, 8)
OMLX_M5_MXFP4_W4A4_NAX(32, 1)
OMLX_M5_MXFP4_W4A4_NAX(32, 2)
OMLX_M5_MXFP4_W4A4_NAX(32, 4)
OMLX_M5_MXFP4_W4A4_NAX(32, 8)
OMLX_M5_MXFP4_W4A4_NAX_SWIGLU(2)
OMLX_M5_MXFP4_W4A4_NAX_SWIGLU(4)
OMLX_M5_MXFP4_W4A4_NAX_SWIGLU(8)

// ============================================================================
// Step 2 (M2): Zero-Copy Cooperative Wide SK=1 Prefill MXFP4 W4A4 (BM_OP=16, 0B TG SRAM)
// ============================================================================

template <typename CoopT>
inline void mxfp4_w4a4_wide_store_linear_slice(
    thread CoopT &prod,
    uint rb_slice,
    uint n0,
    device bfloat *y,
    constant Mxfp4NaxParams &p) {
  const uint elems = prod.get_capacity();
  for (uint i = 0u; i < elems; ++i) {
    if (prod.is_valid_element(i)) {
      const auto idx = prod.get_multidimensional_index(i);
      const uint col = n0 + uint(idx[0]);
      const uint row = rb_slice + uint(idx[1]);
      if (row < p.rows && col < p.n) {
        y[ulong(row) * p.n + col] = bfloat(prod[i]);
      }
    }
  }
}

template <typename CoopT>
inline void mxfp4_w4a4_wide_store_swiglu_slice(
    thread CoopT &pg,
    thread CoopT &pu,
    uint rb_slice,
    uint n0,
    device bfloat *y,
    constant Mxfp4NaxParams &p) {
  const uint elems = pg.get_capacity();
  for (uint i = 0u; i < elems; ++i) {
    if (pg.is_valid_element(i)) {
      const auto idx = pg.get_multidimensional_index(i);
      const uint col = n0 + uint(idx[0]);
      const uint row = rb_slice + uint(idx[1]);
      if (row < p.rows && col < p.n) {
        const float gv = float(bfloat(pg[i]));
        const float uv = float(bfloat(pu[i]));
        y[ulong(row) * p.n + col] = bfloat(nax_silu_f32(gv) * uv);
      }
    }
  }
}

template <uint BN_STORED, uint WM, uint WN>
inline void mxfp4_w4a4_nax_wide_sk1_linear_impl(
    device const uchar *a_codes,
    device const uchar *a_scales,
    device const uchar *b_codes,
    device const uchar *b_scales,
    device bfloat *y,
    constant Mxfp4NaxParams &p,
    uint sg,
    uint3 tg) {
  constexpr uint BM_OP = 16u;
  constexpr uint BK = 128u;
  constexpr uint SM = 32u;
  constexpr uint SN = 32u;
  constexpr uint BM_TG = WM * SM;
  constexpr uint BN_TG = WN * SN;

  const uint num_pid_n = (p.n + BN_TG - 1u) / BN_TG;
  const uint num_pid_m = (p.mp + BM_TG - 1u) / BM_TG;
  const uint2 pid_nm = mxfp4_swizzle_tg_group_m8(tg.xy, num_pid_n, num_pid_m);
  const uint sg_m = (WM > 1u) ? (sg / WN) : 0u;
  const uint sg_n = (WN > 1u) ? (sg % WN) : sg;
  if (sg_m >= WM || sg_n >= WN) {
    return;
  }
  const uint rb_sg = pid_nm.y * BM_TG + sg_m * SM;
  const uint n0_sg = pid_nm.x * BN_TG + sg_n * SN;
  if (rb_sg >= p.rows || n0_sg >= p.n) {
    return;
  }
  const bool row1_active = (rb_sg + BM_OP < p.rows);
  const uint blocks = p.k / BK;
  const ulong a_base0 = ulong(rb_sg / 16u) * blocks;
  const ulong a_base1 = a_base0 + blocks;

  constexpr auto desc = matmul2d_descriptor(
      int(BM_OP), int(BN_STORED), int(BK), false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  Mxfp4WeightTensor a_init((device uchar *)a_codes,
                           dextents<int32_t, 2>(BK, BM_OP),
                           Mxfp4ScaleTag((device uchar *)a_scales));
  Mxfp4WeightTensor b_init((device uchar *)b_codes,
                           dextents<int32_t, 2>(BK, BN_STORED),
                           Mxfp4ScaleTag((device uchar *)b_scales));

  if constexpr (BN_STORED == 16u) {
    const uint tile_n0 = n0_sg / 16u;
    const bool col1_active = (n0_sg + 16u < p.n);
    const ulong b_base0 = ulong(tile_n0) * blocks;
    const ulong b_base1 = b_base0 + blocks;

    auto c00 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
    auto c01 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
    auto c10 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
    auto c11 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
    const uint elems = c00.get_capacity();
    for (uint i = 0u; i < elems; ++i) {
      c00[i] = 0.0f; c01[i] = 0.0f;
      c10[i] = 0.0f; c11[i] = 0.0f;
    }

    for (uint blk = 0u; blk < blocks; ++blk) {
      const ulong ai0 = a_base0 + blk;
      const ulong bi0 = b_base0 + blk;
      Mxfp4WeightTensor a0((device uchar *)(a_codes + ai0 * 1024ul),
                           dextents<int32_t, 2>(BK, BM_OP),
                           Mxfp4ScaleTag((device uchar *)(a_scales + ai0 * 64ul)));
      Mxfp4WeightTensor b0((device uchar *)(b_codes + bi0 * 1024ul),
                           dextents<int32_t, 2>(BK, 16),
                           Mxfp4ScaleTag((device uchar *)(b_scales + bi0 * 64ul)));
      op.run(a0, b0, c00);
      if (row1_active) {
        const ulong ai1 = a_base1 + blk;
        Mxfp4WeightTensor a1((device uchar *)(a_codes + ai1 * 1024ul),
                             dextents<int32_t, 2>(BK, BM_OP),
                             Mxfp4ScaleTag((device uchar *)(a_scales + ai1 * 64ul)));
        op.run(a1, b0, c10);
      }
      if (col1_active) {
        const ulong bi1 = b_base1 + blk;
        Mxfp4WeightTensor b1((device uchar *)(b_codes + bi1 * 1024ul),
                             dextents<int32_t, 2>(BK, 16),
                             Mxfp4ScaleTag((device uchar *)(b_scales + bi1 * 64ul)));
        op.run(a0, b1, c01);
        if (row1_active) {
          const ulong ai1 = a_base1 + blk;
          Mxfp4WeightTensor a1((device uchar *)(a_codes + ai1 * 1024ul),
                               dextents<int32_t, 2>(BK, BM_OP),
                               Mxfp4ScaleTag((device uchar *)(a_scales + ai1 * 64ul)));
          op.run(a1, b1, c11);
        }
      }
    }

    mxfp4_w4a4_wide_store_linear_slice(c00, rb_sg, n0_sg, y, p);
    if (col1_active) {
      mxfp4_w4a4_wide_store_linear_slice(c01, rb_sg, n0_sg + 16u, y, p);
    }
    if (row1_active) {
      mxfp4_w4a4_wide_store_linear_slice(c10, rb_sg + BM_OP, n0_sg, y, p);
      if (col1_active) {
        mxfp4_w4a4_wide_store_linear_slice(c11, rb_sg + BM_OP, n0_sg + 16u, y, p);
      }
    }
  } else {
    const uint tile_n = n0_sg / 32u;
    const ulong b_base = ulong(tile_n) * blocks;
    auto c0 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
    auto c1 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
    const uint elems = c0.get_capacity();
    for (uint i = 0u; i < elems; ++i) {
      c0[i] = 0.0f;
      c1[i] = 0.0f;
    }

    for (uint blk = 0u; blk < blocks; ++blk) {
      const ulong ai0 = a_base0 + blk;
      const ulong bi = b_base + blk;
      Mxfp4WeightTensor a0((device uchar *)(a_codes + ai0 * 1024ul),
                           dextents<int32_t, 2>(BK, BM_OP),
                           Mxfp4ScaleTag((device uchar *)(a_scales + ai0 * 64ul)));
      Mxfp4WeightTensor b((device uchar *)(b_codes + bi * 2048ul),
                          dextents<int32_t, 2>(BK, 32),
                          Mxfp4ScaleTag((device uchar *)(b_scales + bi * 128ul)));
      op.run(a0, b, c0);
      if (row1_active) {
        const ulong ai1 = a_base1 + blk;
        Mxfp4WeightTensor a1((device uchar *)(a_codes + ai1 * 1024ul),
                             dextents<int32_t, 2>(BK, BM_OP),
                             Mxfp4ScaleTag((device uchar *)(a_scales + ai1 * 64ul)));
        op.run(a1, b, c1);
      }
    }

    mxfp4_w4a4_wide_store_linear_slice(c0, rb_sg, n0_sg, y, p);
    if (row1_active) {
      mxfp4_w4a4_wide_store_linear_slice(c1, rb_sg + BM_OP, n0_sg, y, p);
    }
  }
}

template <uint WM, uint WN>
inline void mxfp4_w4a4_nax_wide_sk1_swiglu_impl(
    device const uchar *a_codes,
    device const uchar *a_scales,
    device const uchar *g_codes,
    device const uchar *g_scales,
    device const uchar *u_codes,
    device const uchar *u_scales,
    device bfloat *y,
    constant Mxfp4NaxParams &p,
    uint sg,
    uint3 tg) {
  constexpr uint BM_OP = 16u;
  constexpr uint BK = 128u;
  constexpr uint SM = 32u;
  constexpr uint SN = 32u;
  constexpr uint BM_TG = WM * SM;
  constexpr uint BN_TG = WN * SN;

  const uint num_pid_n = (p.n + BN_TG - 1u) / BN_TG;
  const uint num_pid_m = (p.mp + BM_TG - 1u) / BM_TG;
  const uint2 pid_nm = mxfp4_swizzle_tg_group_m8(tg.xy, num_pid_n, num_pid_m);
  const uint sg_m = (WM > 1u) ? (sg / WN) : 0u;
  const uint sg_n = (WN > 1u) ? (sg % WN) : sg;
  if (sg_m >= WM || sg_n >= WN) {
    return;
  }
  const uint rb_sg = pid_nm.y * BM_TG + sg_m * SM;
  const uint n0_sg = pid_nm.x * BN_TG + sg_n * SN;
  if (rb_sg >= p.rows || n0_sg >= p.n) {
    return;
  }
  const bool row1_active = (rb_sg + BM_OP < p.rows);
  const bool col1_active = (n0_sg + 16u < p.n);
  const uint blocks = p.k / BK;
  const ulong a_base0 = ulong(rb_sg / 16u) * blocks;
  const ulong a_base1 = a_base0 + blocks;
  const ulong b_base0 = ulong(n0_sg / 16u) * blocks;
  const ulong b_base1 = b_base0 + blocks;

  constexpr auto desc = matmul2d_descriptor(
      int(BM_OP), 16, int(BK), false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  Mxfp4WeightTensor a_init((device uchar *)a_codes,
                           dextents<int32_t, 2>(BK, BM_OP),
                           Mxfp4ScaleTag((device uchar *)a_scales));
  Mxfp4WeightTensor b_init((device uchar *)g_codes,
                           dextents<int32_t, 2>(BK, 16),
                           Mxfp4ScaleTag((device uchar *)g_scales));

  auto pg00 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu00 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pg01 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu01 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pg10 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu10 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pg11 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  auto pu11 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(b_init), float>();
  const uint elems = pg00.get_capacity();
  for (uint i = 0u; i < elems; ++i) {
    pg00[i] = 0.0f; pu00[i] = 0.0f;
    pg01[i] = 0.0f; pu01[i] = 0.0f;
    pg10[i] = 0.0f; pu10[i] = 0.0f;
    pg11[i] = 0.0f; pu11[i] = 0.0f;
  }

  for (uint blk = 0u; blk < blocks; ++blk) {
    const ulong ai0 = a_base0 + blk;
    const ulong bi0 = b_base0 + blk;
    Mxfp4WeightTensor a0((device uchar *)(a_codes + ai0 * 1024ul),
                         dextents<int32_t, 2>(BK, BM_OP),
                         Mxfp4ScaleTag((device uchar *)(a_scales + ai0 * 64ul)));
    Mxfp4WeightTensor bg0((device uchar *)(g_codes + bi0 * 1024ul),
                          dextents<int32_t, 2>(BK, 16),
                          Mxfp4ScaleTag((device uchar *)(g_scales + bi0 * 64ul)));
    Mxfp4WeightTensor bu0((device uchar *)(u_codes + bi0 * 1024ul),
                          dextents<int32_t, 2>(BK, 16),
                          Mxfp4ScaleTag((device uchar *)(u_scales + bi0 * 64ul)));
    op.run(a0, bg0, pg00);
    op.run(a0, bu0, pu00);
    if (row1_active) {
      const ulong ai1 = a_base1 + blk;
      Mxfp4WeightTensor a1((device uchar *)(a_codes + ai1 * 1024ul),
                           dextents<int32_t, 2>(BK, BM_OP),
                           Mxfp4ScaleTag((device uchar *)(a_scales + ai1 * 64ul)));
      op.run(a1, bg0, pg10);
      op.run(a1, bu0, pu10);
    }
    if (col1_active) {
      const ulong bi1 = b_base1 + blk;
      Mxfp4WeightTensor bg1((device uchar *)(g_codes + bi1 * 1024ul),
                            dextents<int32_t, 2>(BK, 16),
                            Mxfp4ScaleTag((device uchar *)(g_scales + bi1 * 64ul)));
      Mxfp4WeightTensor bu1((device uchar *)(u_codes + bi1 * 1024ul),
                            dextents<int32_t, 2>(BK, 16),
                            Mxfp4ScaleTag((device uchar *)(u_scales + bi1 * 64ul)));
      op.run(a0, bg1, pg01);
      op.run(a0, bu1, pu01);
      if (row1_active) {
        const ulong ai1 = a_base1 + blk;
        Mxfp4WeightTensor a1((device uchar *)(a_codes + ai1 * 1024ul),
                             dextents<int32_t, 2>(BK, BM_OP),
                             Mxfp4ScaleTag((device uchar *)(a_scales + ai1 * 64ul)));
        op.run(a1, bg1, pg11);
        op.run(a1, bu1, pu11);
      }
    }
  }

  mxfp4_w4a4_wide_store_swiglu_slice(pg00, pu00, rb_sg, n0_sg, y, p);
  if (col1_active) {
    mxfp4_w4a4_wide_store_swiglu_slice(pg01, pu01, rb_sg, n0_sg + 16u, y, p);
  }
  if (row1_active) {
    mxfp4_w4a4_wide_store_swiglu_slice(pg10, pu10, rb_sg + BM_OP, n0_sg, y, p);
    if (col1_active) {
      mxfp4_w4a4_wide_store_swiglu_slice(pg11, pu11, rb_sg + BM_OP, n0_sg + 16u, y, p);
    }
  }
}

#define OMLX_M5_MXFP4_W4A4_WIDE_SK1_LINEAR(NAME, BNS, WM, WN)                  \
  [[max_total_threads_per_threadgroup((WM) * (WN) * 32u)]]                     \
  kernel void NAME(                                                            \
      device const uchar *a_codes [[buffer(0)]],                               \
      device const uchar *a_scales [[buffer(1)]],                              \
      device const uchar *b_codes [[buffer(2)]],                               \
      device const uchar *b_scales [[buffer(3)]],                              \
      device bfloat *y [[buffer(4)]],                                          \
      constant Mxfp4NaxParams &p [[buffer(5)]],                                \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    mxfp4_w4a4_nax_wide_sk1_linear_impl<BNS, WM, WN>(                          \
        a_codes, a_scales, b_codes, b_scales, y, p, sg, tg);                   \
  }

OMLX_M5_MXFP4_W4A4_WIDE_SK1_LINEAR(omlx_m5_mxfp4_w4a4_nax_bm64_bn64_bns16_sk1, 16u, 2u, 2u)
OMLX_M5_MXFP4_W4A4_WIDE_SK1_LINEAR(omlx_m5_mxfp4_w4a4_nax_bm64_bn64_bns32_sk1, 32u, 2u, 2u)
OMLX_M5_MXFP4_W4A4_WIDE_SK1_LINEAR(omlx_m5_mxfp4_w4a4_nax_bm32_bn32_bns16_sk1, 16u, 1u, 1u)
OMLX_M5_MXFP4_W4A4_WIDE_SK1_LINEAR(omlx_m5_mxfp4_w4a4_nax_bm32_bn32_bns32_sk1, 32u, 1u, 1u)

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_w4a4_nax_swiglu_bm64_bn64_sk1(
    device const uchar *a_codes [[buffer(0)]],
    device const uchar *a_scales [[buffer(1)]],
    device const uchar *g_codes [[buffer(2)]],
    device const uchar *g_scales [[buffer(3)]],
    device const uchar *u_codes [[buffer(4)]],
    device const uchar *u_scales [[buffer(5)]],
    device bfloat *y [[buffer(6)]],
    constant Mxfp4NaxParams &p [[buffer(7)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  mxfp4_w4a4_nax_wide_sk1_swiglu_impl<2u, 2u>(
      a_codes, a_scales, g_codes, g_scales, u_codes, u_scales, y, p, sg, tg);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_mxfp4_w4a4_nax_swiglu_bm32_bn32_sk1(
    device const uchar *a_codes [[buffer(0)]],
    device const uchar *a_scales [[buffer(1)]],
    device const uchar *g_codes [[buffer(2)]],
    device const uchar *g_scales [[buffer(3)]],
    device const uchar *u_codes [[buffer(4)]],
    device const uchar *u_scales [[buffer(5)]],
    device bfloat *y [[buffer(6)]],
    constant Mxfp4NaxParams &p [[buffer(7)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  mxfp4_w4a4_nax_wide_sk1_swiglu_impl<1u, 1u>(
      a_codes, a_scales, g_codes, g_scales, u_codes, u_scales, y, p, sg, tg);
}
