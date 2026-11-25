#include "common_math.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct Bf16NaxParams {
  uint n;
  uint k;
  uint mp;
  uint rows;
};

template <uint SK, uint kCap, typename CoopT>
inline void bf16_reduce_store_linear_slice(
    thread CoopT &prod,
    uint rb_slice,
    uint n0,
    bool col_active,
    uint sg_k,
    uint lane,
    threadgroup float *my_parts,
    device bfloat *y,
    constant Bf16NaxParams &p,
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
          y[ulong(row) * p.n + col] = bfloat(prod[i]);
        }
      }
    }
  }
}

template <uint BN, uint BK, uint SK, uint RF = 1u, uint NSG = 1u>
inline void bf16_nax_linear_impl(
    device const bfloat *x,
    device const bfloat *w,
    device bfloat *y,
    constant Bf16NaxParams &p,
    threadgroup float *parts,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM = 8u;
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

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a_parent0(
      (device bfloat *)(x + ulong(rb) * p.k),
      dextents<int32_t, 2>(p.k, p.rows - rb));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> b_dummy(
      (device bfloat *)w, dextents<int32_t, 2>(p.k, min(BN, p.n)));

  auto a_init = a_parent0.slice(0, 0);
  auto b_init = b_dummy.slice(0, 0);
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
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> b_parent(
        (device bfloat *)(w + ulong(n0) * p.k),
        dextents<int32_t, 2>(p.k, p.n - n0));
    for (uint blk = b0; blk < b1; ++blk) {
      const uint k_off = blk * BK;
      auto b = b_parent.slice(k_off, 0);
      auto a0 = a_parent0.slice(k_off, 0);
      op.run(a0, b, prod0);
      if constexpr (RF > 1u) {
        if (rb + 8u < p.rows) {
          tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> ap((device bfloat *)(x + ulong(rb + 8u) * p.k), dextents<int32_t, 2>(p.k, p.rows - (rb + 8u)));
          auto ar = ap.slice(k_off, 0);
          op.run(ar, b, prod1);
        }
      }
      if constexpr (RF > 2u) {
        if (rb + 16u < p.rows) {
          tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> ap((device bfloat *)(x + ulong(rb + 16u) * p.k), dextents<int32_t, 2>(p.k, p.rows - (rb + 16u)));
          auto ar = ap.slice(k_off, 0);
          op.run(ar, b, prod2);
        }
      }
      if constexpr (RF > 3u) {
        if (rb + 24u < p.rows) {
          tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> ap((device bfloat *)(x + ulong(rb + 24u) * p.k), dextents<int32_t, 2>(p.k, p.rows - (rb + 24u)));
          auto ar = ap.slice(k_off, 0);
          op.run(ar, b, prod3);
        }
      }
      if constexpr (RF > 4u) {
        if (rb + 32u < p.rows) {
          tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> ap((device bfloat *)(x + ulong(rb + 32u) * p.k), dextents<int32_t, 2>(p.k, p.rows - (rb + 32u)));
          auto ar = ap.slice(k_off, 0);
          op.run(ar, b, prod4);
        }
      }
      if constexpr (RF > 5u) {
        if (rb + 40u < p.rows) {
          tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> ap((device bfloat *)(x + ulong(rb + 40u) * p.k), dextents<int32_t, 2>(p.k, p.rows - (rb + 40u)));
          auto ar = ap.slice(k_off, 0);
          op.run(ar, b, prod5);
        }
      }
      if constexpr (RF > 6u) {
        if (rb + 48u < p.rows) {
          tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> ap((device bfloat *)(x + ulong(rb + 48u) * p.k), dextents<int32_t, 2>(p.k, p.rows - (rb + 48u)));
          auto ar = ap.slice(k_off, 0);
          op.run(ar, b, prod6);
        }
      }
      if constexpr (RF > 7u) {
        if (rb + 56u < p.rows) {
          tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> ap((device bfloat *)(x + ulong(rb + 56u) * p.k), dextents<int32_t, 2>(p.k, p.rows - (rb + 56u)));
          auto ar = ap.slice(k_off, 0);
          op.run(ar, b, prod7);
        }
      }
    }
  }

  constexpr uint kPartStride = (SK > 1u ? (SK - 1u) : 1u) * kCap * 32u;
  threadgroup float *my_parts = parts + ((NSG > 1u && sg_n < NSG) ? (sg_n * kPartStride) : 0u);

  bf16_reduce_store_linear_slice<SK, kCap>(prod0, rb, n0, col_active, sg_k, lane, my_parts, y, p, false);
  if constexpr (RF > 1u) { if (rb + 8u < p.rows) bf16_reduce_store_linear_slice<SK, kCap>(prod1, rb + 8u, n0, col_active, sg_k, lane, my_parts, y, p, true); }
  if constexpr (RF > 2u) { if (rb + 16u < p.rows) bf16_reduce_store_linear_slice<SK, kCap>(prod2, rb + 16u, n0, col_active, sg_k, lane, my_parts, y, p, true); }
  if constexpr (RF > 3u) { if (rb + 24u < p.rows) bf16_reduce_store_linear_slice<SK, kCap>(prod3, rb + 24u, n0, col_active, sg_k, lane, my_parts, y, p, true); }
  if constexpr (RF > 4u) { if (rb + 32u < p.rows) bf16_reduce_store_linear_slice<SK, kCap>(prod4, rb + 32u, n0, col_active, sg_k, lane, my_parts, y, p, true); }
  if constexpr (RF > 5u) { if (rb + 40u < p.rows) bf16_reduce_store_linear_slice<SK, kCap>(prod5, rb + 40u, n0, col_active, sg_k, lane, my_parts, y, p, true); }
  if constexpr (RF > 6u) { if (rb + 48u < p.rows) bf16_reduce_store_linear_slice<SK, kCap>(prod6, rb + 48u, n0, col_active, sg_k, lane, my_parts, y, p, true); }
  if constexpr (RF > 7u) { if (rb + 56u < p.rows) bf16_reduce_store_linear_slice<SK, kCap>(prod7, rb + 56u, n0, col_active, sg_k, lane, my_parts, y, p, true); }
}

template <uint SK, uint kCap, typename CoopT>
inline void bf16_reduce_store_swiglu_slice(
    thread CoopT &pg,
    thread CoopT &pu,
    uint rb_slice,
    uint n0,
    uint sg,
    uint lane,
    threadgroup float *parts,
    device bfloat *y,
    constant Bf16NaxParams &p,
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

template <uint BN, uint BK, uint SK, uint RF = 1u>
inline void bf16_nax_swiglu_impl(
    device const bfloat *x,
    device const bfloat *wg,
    device const bfloat *wu,
    device bfloat *y,
    constant Bf16NaxParams &p,
    threadgroup float *parts,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM = 8u;
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

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a_parent0(
      (device bfloat *)(x + ulong(rb) * p.k),
      dextents<int32_t, 2>(p.k, p.rows - rb));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> bg_parent(
      (device bfloat *)(wg + ulong(n0) * p.k),
      dextents<int32_t, 2>(p.k, p.n - n0));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> bu_parent(
      (device bfloat *)(wu + ulong(n0) * p.k),
      dextents<int32_t, 2>(p.k, p.n - n0));

  auto a_init = a_parent0.slice(0, 0);
  auto bg_init = bg_parent.slice(0, 0);
  auto pg0 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pu0 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pg1 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pu1 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pg2 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pu2 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pg3 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pu3 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pg4 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pu4 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pg5 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pu5 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pg6 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pu6 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pg7 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pu7 = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
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

  for (uint blk = b0; blk < b1; ++blk) {
    const uint k_off = blk * BK;
    auto bg = bg_parent.slice(k_off, 0);
    auto bu = bu_parent.slice(k_off, 0);
    auto a0 = a_parent0.slice(k_off, 0);
    op.run(a0, bg, pg0);
    op.run(a0, bu, pu0);
    if constexpr (RF > 1u) {
      if (rb + 8u < p.rows) {
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> ap((device bfloat *)(x + ulong(rb + 8u) * p.k), dextents<int32_t, 2>(p.k, p.rows - (rb + 8u)));
        auto ar = ap.slice(k_off, 0); op.run(ar, bg, pg1); op.run(ar, bu, pu1);
      }
    }
    if constexpr (RF > 2u) {
      if (rb + 16u < p.rows) {
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> ap((device bfloat *)(x + ulong(rb + 16u) * p.k), dextents<int32_t, 2>(p.k, p.rows - (rb + 16u)));
        auto ar = ap.slice(k_off, 0); op.run(ar, bg, pg2); op.run(ar, bu, pu2);
      }
    }
    if constexpr (RF > 3u) {
      if (rb + 24u < p.rows) {
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> ap((device bfloat *)(x + ulong(rb + 24u) * p.k), dextents<int32_t, 2>(p.k, p.rows - (rb + 24u)));
        auto ar = ap.slice(k_off, 0); op.run(ar, bg, pg3); op.run(ar, bu, pu3);
      }
    }
    if constexpr (RF > 4u) {
      if (rb + 32u < p.rows) {
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> ap((device bfloat *)(x + ulong(rb + 32u) * p.k), dextents<int32_t, 2>(p.k, p.rows - (rb + 32u)));
        auto ar = ap.slice(k_off, 0); op.run(ar, bg, pg4); op.run(ar, bu, pu4);
      }
    }
    if constexpr (RF > 5u) {
      if (rb + 40u < p.rows) {
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> ap((device bfloat *)(x + ulong(rb + 40u) * p.k), dextents<int32_t, 2>(p.k, p.rows - (rb + 40u)));
        auto ar = ap.slice(k_off, 0); op.run(ar, bg, pg5); op.run(ar, bu, pu5);
      }
    }
    if constexpr (RF > 6u) {
      if (rb + 48u < p.rows) {
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> ap((device bfloat *)(x + ulong(rb + 48u) * p.k), dextents<int32_t, 2>(p.k, p.rows - (rb + 48u)));
        auto ar = ap.slice(k_off, 0); op.run(ar, bg, pg6); op.run(ar, bu, pu6);
      }
    }
    if constexpr (RF > 7u) {
      if (rb + 56u < p.rows) {
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> ap((device bfloat *)(x + ulong(rb + 56u) * p.k), dextents<int32_t, 2>(p.k, p.rows - (rb + 56u)));
        auto ar = ap.slice(k_off, 0); op.run(ar, bg, pg7); op.run(ar, bu, pu7);
      }
    }
  }

  bf16_reduce_store_swiglu_slice<SK, kCap>(pg0, pu0, rb, n0, sg, lane, parts, y, p, false);
  if constexpr (RF > 1u) { if (rb + 8u < p.rows) bf16_reduce_store_swiglu_slice<SK, kCap>(pg1, pu1, rb + 8u, n0, sg, lane, parts, y, p, true); }
  if constexpr (RF > 2u) { if (rb + 16u < p.rows) bf16_reduce_store_swiglu_slice<SK, kCap>(pg2, pu2, rb + 16u, n0, sg, lane, parts, y, p, true); }
  if constexpr (RF > 3u) { if (rb + 24u < p.rows) bf16_reduce_store_swiglu_slice<SK, kCap>(pg3, pu3, rb + 24u, n0, sg, lane, parts, y, p, true); }
  if constexpr (RF > 4u) { if (rb + 32u < p.rows) bf16_reduce_store_swiglu_slice<SK, kCap>(pg4, pu4, rb + 32u, n0, sg, lane, parts, y, p, true); }
  if constexpr (RF > 5u) { if (rb + 40u < p.rows) bf16_reduce_store_swiglu_slice<SK, kCap>(pg5, pu5, rb + 40u, n0, sg, lane, parts, y, p, true); }
  if constexpr (RF > 6u) { if (rb + 48u < p.rows) bf16_reduce_store_swiglu_slice<SK, kCap>(pg6, pu6, rb + 48u, n0, sg, lane, parts, y, p, true); }
  if constexpr (RF > 7u) { if (rb + 56u < p.rows) bf16_reduce_store_swiglu_slice<SK, kCap>(pg7, pu7, rb + 56u, n0, sg, lane, parts, y, p, true); }
}

#define OMLX_M5_BF16_NAX_LINEAR_N16_RF(BK, SK, RF)                             \
  kernel void omlx_m5_bf16_nax_bk##BK##_n16_sk##SK##_rf##RF##_nsg1(            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const bfloat *w [[buffer(1)]],                                    \
      device bfloat *y [[buffer(2)]],                                          \
      constant Bf16NaxParams &p [[buffer(3)]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? (SK - 1u) : 1u) * 8u * 32u];            \
    bf16_nax_linear_impl<16u, BK, SK, RF, 1u>(x, w, y, p, parts, lane, sg, tg);\
  }                                                                            \
  kernel void omlx_m5_bf16_nax_bk##BK##_n16_sk##SK##_rf##RF##_nsg2(            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const bfloat *w [[buffer(1)]],                                    \
      device bfloat *y [[buffer(2)]],                                          \
      constant Bf16NaxParams &p [[buffer(3)]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[2u * (SK > 1u ? (SK - 1u) : 1u) * 8u * 32u];       \
    bf16_nax_linear_impl<16u, BK, SK, RF, 2u>(x, w, y, p, parts, lane, sg, tg);\
  }

#define OMLX_M5_BF16_NAX_LINEAR_N16_ALL(BK, SK)                                \
  kernel void omlx_m5_bf16_nax_bk##BK##_n16_sk##SK(                            \
      device const bfloat *x [[buffer(0)]],                                    \
      device const bfloat *w [[buffer(1)]],                                    \
      device bfloat *y [[buffer(2)]],                                          \
      constant Bf16NaxParams &p [[buffer(3)]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? (SK - 1u) : 1u) * 8u * 32u];            \
    bf16_nax_linear_impl<16u, BK, SK, 1u, 1u>(x, w, y, p, parts, lane, sg, tg);\
  }                                                                            \
  kernel void omlx_m5_bf16_nax_bk##BK##_n16_sk##SK##_rf1_nsg2(                 \
      device const bfloat *x [[buffer(0)]],                                    \
      device const bfloat *w [[buffer(1)]],                                    \
      device bfloat *y [[buffer(2)]],                                          \
      constant Bf16NaxParams &p [[buffer(3)]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[2u * (SK > 1u ? (SK - 1u) : 1u) * 8u * 32u];       \
    bf16_nax_linear_impl<16u, BK, SK, 1u, 2u>(x, w, y, p, parts, lane, sg, tg);\
  }                                                                            \
  OMLX_M5_BF16_NAX_LINEAR_N16_RF(BK, SK, 2)                                    \
  OMLX_M5_BF16_NAX_LINEAR_N16_RF(BK, SK, 3)                                    \
  OMLX_M5_BF16_NAX_LINEAR_N16_RF(BK, SK, 4)                                    \
  OMLX_M5_BF16_NAX_LINEAR_N16_RF(BK, SK, 6)                                    \
  OMLX_M5_BF16_NAX_LINEAR_N16_RF(BK, SK, 8)

#define OMLX_M5_BF16_NAX_LINEAR_WIDE_RF(BN, BK, SK, RF)                        \
  kernel void omlx_m5_bf16_nax_bk##BK##_n##BN##_sk##SK##_rf##RF(               \
      device const bfloat *x [[buffer(0)]],                                    \
      device const bfloat *w [[buffer(1)]],                                    \
      device bfloat *y [[buffer(2)]],                                          \
      constant Bf16NaxParams &p [[buffer(3)]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? (SK - 1u) : 1u) * ((16u * BN) / 32u) * 32u]; \
    bf16_nax_linear_impl<BN, BK, SK, RF, 1u>(x, w, y, p, parts, lane, sg, tg); \
  }

#define OMLX_M5_BF16_NAX_LINEAR_WIDE_ALL(BN, BK, SK)                           \
  kernel void omlx_m5_bf16_nax_bk##BK##_n##BN##_sk##SK(                        \
      device const bfloat *x [[buffer(0)]],                                    \
      device const bfloat *w [[buffer(1)]],                                    \
      device bfloat *y [[buffer(2)]],                                          \
      constant Bf16NaxParams &p [[buffer(3)]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? (SK - 1u) : 1u) * ((16u * BN) / 32u) * 32u]; \
    bf16_nax_linear_impl<BN, BK, SK, 1u, 1u>(x, w, y, p, parts, lane, sg, tg); \
  }                                                                            \
  OMLX_M5_BF16_NAX_LINEAR_WIDE_RF(BN, BK, SK, 2)                               \
  OMLX_M5_BF16_NAX_LINEAR_WIDE_RF(BN, BK, SK, 3)                               \
  OMLX_M5_BF16_NAX_LINEAR_WIDE_RF(BN, BK, SK, 4)                               \
  OMLX_M5_BF16_NAX_LINEAR_WIDE_RF(BN, BK, SK, 6)                               \
  OMLX_M5_BF16_NAX_LINEAR_WIDE_RF(BN, BK, SK, 8)

#define OMLX_M5_BF16_NAX_SWIGLU_RF(BN, BK, SK, RF)                             \
  kernel void omlx_m5_bf16_nax_swiglu_bk##BK##_n##BN##_sk##SK##_rf##RF(        \
      device const bfloat *x [[buffer(0)]],                                    \
      device const bfloat *wg [[buffer(1)]],                                   \
      device const bfloat *wu [[buffer(2)]],                                   \
      device bfloat *y [[buffer(3)]],                                          \
      constant Bf16NaxParams &p [[buffer(4)]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? (SK - 1u) : 1u) * 2u * ((16u * BN) / 32u) * 32u]; \
    bf16_nax_swiglu_impl<BN, BK, SK, RF>(x, wg, wu, y, p, parts, lane, sg, tg);\
  }

#define OMLX_M5_BF16_NAX_SWIGLU_ALL(BN, BK, SK)                                \
  kernel void omlx_m5_bf16_nax_swiglu_bk##BK##_n##BN##_sk##SK(                 \
      device const bfloat *x [[buffer(0)]],                                    \
      device const bfloat *wg [[buffer(1)]],                                   \
      device const bfloat *wu [[buffer(2)]],                                   \
      device bfloat *y [[buffer(3)]],                                          \
      constant Bf16NaxParams &p [[buffer(4)]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[(SK > 1u ? (SK - 1u) : 1u) * 2u * ((16u * BN) / 32u) * 32u]; \
    bf16_nax_swiglu_impl<BN, BK, SK, 1u>(x, wg, wu, y, p, parts, lane, sg, tg);\
  }                                                                            \
  OMLX_M5_BF16_NAX_SWIGLU_RF(BN, BK, SK, 2)                                    \
  OMLX_M5_BF16_NAX_SWIGLU_RF(BN, BK, SK, 3)                                    \
  OMLX_M5_BF16_NAX_SWIGLU_RF(BN, BK, SK, 4)                                    \
  OMLX_M5_BF16_NAX_SWIGLU_RF(BN, BK, SK, 6)                                    \
  OMLX_M5_BF16_NAX_SWIGLU_RF(BN, BK, SK, 8)

OMLX_M5_BF16_NAX_LINEAR_N16_ALL(128, 1)
OMLX_M5_BF16_NAX_LINEAR_N16_ALL(128, 2)
OMLX_M5_BF16_NAX_LINEAR_N16_ALL(128, 4)
OMLX_M5_BF16_NAX_LINEAR_N16_ALL(128, 8)
OMLX_M5_BF16_NAX_LINEAR_WIDE_ALL(32, 128, 1)
OMLX_M5_BF16_NAX_LINEAR_WIDE_ALL(32, 128, 2)
OMLX_M5_BF16_NAX_LINEAR_WIDE_ALL(32, 128, 4)
OMLX_M5_BF16_NAX_LINEAR_WIDE_ALL(32, 128, 8)
OMLX_M5_BF16_NAX_LINEAR_WIDE_ALL(64, 128, 1)
OMLX_M5_BF16_NAX_LINEAR_WIDE_ALL(64, 128, 2)
OMLX_M5_BF16_NAX_LINEAR_WIDE_ALL(64, 128, 4)

OMLX_M5_BF16_NAX_LINEAR_N16_ALL(64, 1)
OMLX_M5_BF16_NAX_LINEAR_N16_ALL(64, 2)
OMLX_M5_BF16_NAX_LINEAR_N16_ALL(64, 4)
OMLX_M5_BF16_NAX_LINEAR_N16_ALL(64, 8)
OMLX_M5_BF16_NAX_LINEAR_WIDE_ALL(32, 64, 2)
OMLX_M5_BF16_NAX_LINEAR_WIDE_ALL(32, 64, 4)

OMLX_M5_BF16_NAX_SWIGLU_ALL(16, 128, 2)
OMLX_M5_BF16_NAX_SWIGLU_ALL(16, 128, 4)
OMLX_M5_BF16_NAX_SWIGLU_ALL(16, 128, 8)
OMLX_M5_BF16_NAX_SWIGLU_ALL(32, 128, 2)
OMLX_M5_BF16_NAX_SWIGLU_ALL(32, 128, 4)
OMLX_M5_BF16_NAX_SWIGLU_ALL(16, 64, 2)
OMLX_M5_BF16_NAX_SWIGLU_ALL(16, 64, 4)
OMLX_M5_BF16_NAX_SWIGLU_ALL(16, 64, 8)

// ============================================================================
// Step 2 (M2) + Tier-1 (T1-A): Cooperative TG-SRAM Staged Wide SK=1 Prefill
// BF16 NAX (As[64*72] + Ws[64*72], Register-Fragment relaxed_precision=true
// 32x32x32 NAX MMA, L2 GROUP_M=8 Swizzling)
// ============================================================================

inline uint2 bf16_swizzle_tg_group_m8(uint2 tg_xy, uint num_pid_n, uint num_pid_m) {
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

inline void bf16_mma_16x32x16(
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

inline bool bf16_has_apple10_nax_layout() {
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

template <uint WM, uint WN>
inline void bf16_nax_wide_sk1_linear_apple9(
    device const bfloat *x,
    device const bfloat *w,
    device bfloat *y,
    constant Bf16NaxParams &p,
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
    uint4 raw_f[kFetches][4];
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      const uint col = n0_tg + (idx32 >> 1u);
      const uint col32 = (idx32 & 1u) * 32u;
      if (col < p.n) {
        device const uint4 *src = reinterpret_cast<device const uint4 *>(w + ulong(col) * p.k + k_off + col32);
        #pragma clang loop unroll(full)
        for (uint u = 0u; u < 4u; ++u) raw_f[f][u] = src[u];
      } else {
        #pragma clang loop unroll(full)
        for (uint u = 0u; u < 4u; ++u) raw_f[f][u] = uint4(0u);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      threadgroup uint4 *dst = reinterpret_cast<threadgroup uint4 *>(
          Ws + (idx32 >> 1u) * kBKP + (idx32 & 1u) * 32u);
      #pragma clang loop unroll(full)
      for (uint u = 0u; u < 4u; ++u) dst[u] = raw_f[f][u];
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

template <uint WM, uint WN>
inline void bf16_nax_wide_sk1_swiglu_apple9(
    device const bfloat *x,
    device const bfloat *wg,
    device const bfloat *wu,
    device bfloat *y,
    constant Bf16NaxParams &p,
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
    uint4 raw_f[kFetches][4];
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      const uint w_row = idx32 >> 1u;
      const uint col32 = (idx32 & 1u) * 32u;
      const bool is_up = ((w_row >> 4u) & 1u) != 0u;
      const uint out_col = n0_tg + ((w_row >> 5u) * 16u) + (w_row & 15u);
      device const bfloat *sel_w = is_up ? wu : wg;
      if (out_col < p.n) {
        device const uint4 *src = reinterpret_cast<device const uint4 *>(sel_w + ulong(out_col) * p.k + k_off + col32);
        #pragma clang loop unroll(full)
        for (uint u = 0u; u < 4u; ++u) raw_f[f][u] = src[u];
      } else {
        #pragma clang loop unroll(full)
        for (uint u = 0u; u < 4u; ++u) raw_f[f][u] = uint4(0u);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      threadgroup uint4 *dst = reinterpret_cast<threadgroup uint4 *>(
          Ws + (idx32 >> 1u) * kBKP + (idx32 & 1u) * 32u);
      #pragma clang loop unroll(full)
      for (uint u = 0u; u < 4u; ++u) dst[u] = raw_f[f][u];
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

template <uint BK, uint WM, uint WN>
inline void bf16_nax_wide_sk1_linear_impl(
    device const bfloat *x,
    device const bfloat *w,
    device bfloat *y,
    constant Bf16NaxParams &p,
    threadgroup bfloat *Ws,
    uint lane,
    uint sg,
    uint3 tg) {
  (void)BK;
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
  const uint2 pid_nm = bf16_swizzle_tg_group_m8(tg.xy, num_pid_n, num_pid_m);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg = pid_nm.x * BN_TG;
  if (rb_tg >= p.rows || n0_tg >= p.n) {
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint sg_m = (WM > 1u) ? (sg / WN) : 0u;
  const uint sg_n = (WN > 1u) ? (sg % WN) : sg;
  if (!bf16_has_apple10_nax_layout()) {
    bf16_nax_wide_sk1_linear_apple9<WM, WN>(x, w, y, p, Ws, tid, sg_m, sg_n, rb_tg, n0_tg);
    return;
  }

  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * SN;
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
    device const bfloat *w_row_ptr[kFetches];
    threadgroup uint4 *w_dst4[kFetches];
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      const uint w_row = idx32 >> 1u;
      const uint col32 = (idx32 & 1u) * 32u;
      w_row_ptr[f] = w + ulong(n0_tg + w_row) * p.k + col32;
      w_dst4[f] = reinterpret_cast<threadgroup uint4 *>(Ws + w_row * kBKP + col32);
    }
    device const bfloat *x_sg = x + ulong(rb_sg + fm) * p.k + fn;
    const threadgroup bfloat *ws_sg = Ws + (sg_n * 32u + fm) * kBKP + fn;

    for (uint k0 = 0u; k0 < p.k; k0 += kBK) {
      uint4 raw_f[kFetches][4];
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        device const uint4 *src = reinterpret_cast<device const uint4 *>(w_row_ptr[f] + k0);
        #pragma clang loop unroll(full)
        for (uint u = 0u; u < 4u; ++u) raw_f[f][u] = src[u];
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        #pragma clang loop unroll(full)
        for (uint u = 0u; u < 4u; ++u) w_dst4[f][u] = raw_f[f][u];
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
        bf16_mma_16x32x16(D[0][0], D[0][1], a00, b00, b10);
        bf16_mma_16x32x16(D[0][0], D[0][1], a01, b01, b11);
        bf16_mma_16x32x16(D[1][0], D[1][1], a10, b00, b10);
        bf16_mma_16x32x16(D[1][0], D[1][1], a11, b01, b11);
        (void)cbar;
      }
    }
  } else {
    const bool sg_active = (sg_m < WM) && (sg_n < WN) && (rb_sg < p.rows) && (n0_sg < p.n);
    const threadgroup bfloat *ws_sg = Ws + (sg_n * SN + fm) * kBKP + fn;
    for (uint k0 = 0u; k0 < p.k; k0 += kBK) {
      uint4 raw_f[kFetches][4];
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint idx32 = tid + f * TPTG;
        const uint col = n0_tg + (idx32 >> 1u);
        const uint col32 = (idx32 & 1u) * 32u;
        if (col < p.n) {
          device const uint4 *src = reinterpret_cast<device const uint4 *>(w + ulong(col) * p.k + k0 + col32);
          #pragma clang loop unroll(full)
          for (uint u = 0u; u < 4u; ++u) raw_f[f][u] = src[u];
        } else {
          #pragma clang loop unroll(full)
          for (uint u = 0u; u < 4u; ++u) raw_f[f][u] = uint4(0u);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint idx32 = tid + f * TPTG;
        threadgroup uint4 *dst = reinterpret_cast<threadgroup uint4 *>(
            Ws + (idx32 >> 1u) * kBKP + (idx32 & 1u) * 32u);
        #pragma clang loop unroll(full)
        for (uint u = 0u; u < 4u; ++u) dst[u] = raw_f[f][u];
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
                    ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p.k + k0 + kk + fn)
                    : bfloat4(bfloat(0.0f));
                const bfloat4 va1 = (row < p.rows)
                    ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p.k + k0 + kk + 16u + fn)
                    : bfloat4(bfloat(0.0f));
                #pragma clang loop unroll(full)
                for (uint c = 0u; c < 4u; ++c) {
                  a0[r * 4u + c] = va0[c];
                  a1[r * 4u + c] = va1[c];
                }
              }
              bf16_mma_16x32x16(D[mm][0], D[mm][1], a0, b00, b10);
              bf16_mma_16x32x16(D[mm][0], D[mm][1], a1, b01, b11);
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

template <uint BK, uint WM, uint WN>
inline void bf16_nax_wide_sk1_swiglu_impl(
    device const bfloat *x,
    device const bfloat *wg,
    device const bfloat *wu,
    device bfloat *y,
    constant Bf16NaxParams &p,
    threadgroup bfloat *Ws,
    uint lane,
    uint sg,
    uint3 tg) {
  (void)BK;
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
  const uint2 pid_nm = bf16_swizzle_tg_group_m8(tg.xy, num_pid_n, num_pid_m);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg = pid_nm.x * OUT_N_TG;
  if (rb_tg >= p.rows || n0_tg >= p.n) {
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint sg_m = (WM > 1u) ? (sg / WN) : 0u;
  const uint sg_n = (WN > 1u) ? (sg % WN) : sg;
  if (!bf16_has_apple10_nax_layout()) {
    bf16_nax_wide_sk1_swiglu_apple9<WM, WN>(x, wg, wu, y, p, Ws, tid, sg_m, sg_n, rb_tg, n0_tg);
    return;
  }

  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * OUT_SN;
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
    device const bfloat *w_row_ptr[kFetches];
    threadgroup uint4 *w_dst4[kFetches];
    #pragma clang loop unroll(full)
    for (uint f = 0u; f < kFetches; ++f) {
      const uint idx32 = tid + f * TPTG;
      const uint w_row = idx32 >> 1u;
      const uint col32 = (idx32 & 1u) * 32u;
      const bool is_up = ((w_row >> 4u) & 1u) != 0u;
      const uint out_col = n0_tg + ((w_row >> 5u) * 16u) + (w_row & 15u);
      w_row_ptr[f] = (is_up ? wu : wg) + ulong(out_col) * p.k + col32;
      w_dst4[f] = reinterpret_cast<threadgroup uint4 *>(Ws + w_row * kBKP + col32);
    }
    device const bfloat *x_sg = x + ulong(rb_sg + fm) * p.k + fn;
    const threadgroup bfloat *ws_sg = Ws + (sg_n * 32u + fm) * kBKP + fn;

    for (uint k0 = 0u; k0 < p.k; k0 += kBK) {
      uint4 raw_f[kFetches][4];
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        device const uint4 *src = reinterpret_cast<device const uint4 *>(w_row_ptr[f] + k0);
        #pragma clang loop unroll(full)
        for (uint u = 0u; u < 4u; ++u) raw_f[f][u] = src[u];
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        #pragma clang loop unroll(full)
        for (uint u = 0u; u < 4u; ++u) w_dst4[f][u] = raw_f[f][u];
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
        bf16_mma_16x32x16(D[0][0], D[0][1], a00, b00, b10);
        bf16_mma_16x32x16(D[0][0], D[0][1], a01, b01, b11);
        bf16_mma_16x32x16(D[1][0], D[1][1], a10, b00, b10);
        bf16_mma_16x32x16(D[1][0], D[1][1], a11, b01, b11);
        (void)cbar;
      }
    }
  } else {
    const bool sg_active = (sg_m < WM) && (sg_n < WN) && (rb_sg < p.rows) && (n0_sg < p.n);
    const threadgroup bfloat *ws_sg = Ws + (sg_n * 32u + fm) * kBKP + fn;
    for (uint k0 = 0u; k0 < p.k; k0 += kBK) {
      uint4 raw_f[kFetches][4];
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint idx32 = tid + f * TPTG;
        const uint w_row = idx32 >> 1u;
        const uint col32 = (idx32 & 1u) * 32u;
        const bool is_up = ((w_row >> 4u) & 1u) != 0u;
        const uint out_col = n0_tg + ((w_row >> 5u) * 16u) + (w_row & 15u);
        device const bfloat *sel_w = is_up ? wu : wg;
        if (out_col < p.n) {
          device const uint4 *src = reinterpret_cast<device const uint4 *>(sel_w + ulong(out_col) * p.k + k0 + col32);
          #pragma clang loop unroll(full)
          for (uint u = 0u; u < 4u; ++u) raw_f[f][u] = src[u];
        } else {
          #pragma clang loop unroll(full)
          for (uint u = 0u; u < 4u; ++u) raw_f[f][u] = uint4(0u);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint f = 0u; f < kFetches; ++f) {
        const uint idx32 = tid + f * TPTG;
        threadgroup uint4 *dst = reinterpret_cast<threadgroup uint4 *>(
            Ws + (idx32 >> 1u) * kBKP + (idx32 & 1u) * 32u);
        #pragma clang loop unroll(full)
        for (uint u = 0u; u < 4u; ++u) dst[u] = raw_f[f][u];
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
                    ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p.k + k0 + kk + fn)
                    : bfloat4(bfloat(0.0f));
                const bfloat4 va1 = (row < p.rows)
                    ? *reinterpret_cast<device const bfloat4 *>(x + ulong(row) * p.k + k0 + kk + 16u + fn)
                    : bfloat4(bfloat(0.0f));
                #pragma clang loop unroll(full)
                for (uint c = 0u; c < 4u; ++c) {
                  a0[r * 4u + c] = va0[c];
                  a1[r * 4u + c] = va1[c];
                }
              }
              bf16_mma_16x32x16(D[mm][0], D[mm][1], a0, b00, b10);
              bf16_mma_16x32x16(D[mm][0], D[mm][1], a1, b01, b11);
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

#define OMLX_M5_BF16_NAX_WIDE_SK1_KERNELS(BK)                                  \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_bf16_nax_bk##BK##_bm64_bn64_sk1(                         \
      device const bfloat *x [[buffer(0)]],                                    \
      device const bfloat *w [[buffer(1)]],                                    \
      device bfloat *y [[buffer(2)]],                                          \
      constant Bf16NaxParams &p [[buffer(3)]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[64u * 72u];                              \
    bf16_nax_wide_sk1_linear_impl<BK, 2u, 2u>(x, w, y, p, Ws, lane, sg, tg);   \
  }                                                                            \
  [[max_total_threads_per_threadgroup(32)]]                                    \
  kernel void omlx_m5_bf16_nax_bk##BK##_bm32_bn32_sk1(                         \
      device const bfloat *x [[buffer(0)]],                                    \
      device const bfloat *w [[buffer(1)]],                                    \
      device bfloat *y [[buffer(2)]],                                          \
      constant Bf16NaxParams &p [[buffer(3)]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[32u * 72u];                              \
    bf16_nax_wide_sk1_linear_impl<BK, 1u, 1u>(x, w, y, p, Ws, lane, sg, tg);   \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_bf16_nax_swiglu_bk##BK##_bm64_bn64_sk1(                  \
      device const bfloat *x [[buffer(0)]],                                    \
      device const bfloat *wg [[buffer(1)]],                                   \
      device const bfloat *wu [[buffer(2)]],                                   \
      device bfloat *y [[buffer(3)]],                                          \
      constant Bf16NaxParams &p [[buffer(4)]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[64u * 72u];                              \
    bf16_nax_wide_sk1_swiglu_impl<BK, 2u, 2u>(x, wg, wu, y, p, Ws, lane, sg, tg);\
  }                                                                            \
  [[max_total_threads_per_threadgroup(32)]]                                    \
  kernel void omlx_m5_bf16_nax_swiglu_bk##BK##_bm32_bn32_sk1(                  \
      device const bfloat *x [[buffer(0)]],                                    \
      device const bfloat *wg [[buffer(1)]],                                   \
      device const bfloat *wu [[buffer(2)]],                                   \
      device bfloat *y [[buffer(3)]],                                          \
      constant Bf16NaxParams &p [[buffer(4)]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[32u * 72u];                              \
    bf16_nax_wide_sk1_swiglu_impl<BK, 1u, 1u>(x, wg, wu, y, p, Ws, lane, sg, tg);\
  }

OMLX_M5_BF16_NAX_WIDE_SK1_KERNELS(128)
OMLX_M5_BF16_NAX_WIDE_SK1_KERNELS(64)

// ===========================================================================
// Dim D 0-ULP Hardware BM=16 & BM=32 BF16 NAX (DFlash2 packed B=2..8 drafter)
// ===========================================================================

template <uint BM, uint BN, uint BK, uint SK>
inline void bf16_nax_linear_bm_impl(
    device const bfloat *x,
    device const bfloat *w,
    device bfloat *y,
    constant Bf16NaxParams &p,
    threadgroup float *parts,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint kCap = (BM * BN) / 32u;
  const uint rb = tg.y * BM;
  const uint n0 = tg.x * BN;
  if (rb >= p.rows || n0 >= p.n) {
    return;
  }

  const uint blocks = p.k / BK;
  const uint b0 = (sg * blocks) / SK;
  const uint b1 = ((sg + 1u) * blocks) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a_parent(
      (device bfloat *)(x + ulong(rb) * p.k),
      dextents<int32_t, 2>(p.k, p.rows - rb));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> b_parent(
      (device bfloat *)(w + ulong(n0) * p.k),
      dextents<int32_t, 2>(p.k, p.n - n0));

  auto a_init = a_parent.slice(0, 0);
  auto b_init = b_parent.slice(0, 0);
  auto prod = op.template get_destination_cooperative_tensor<
      decltype(a_init), decltype(b_init), float>();
  const uint elems = prod.get_capacity();
  for (uint i = 0u; i < elems; ++i) {
    prod[i] = 0.0f;
  }

  for (uint blk = b0; blk < b1; ++blk) {
    const uint k_off = blk * BK;
    auto a = a_parent.slice(k_off, 0);
    auto b = b_parent.slice(k_off, 0);
    op.run(a, b, prod);
  }

  bf16_reduce_store_linear_slice<SK, kCap>(
      prod, rb, n0, true, sg, lane, parts, y, p, false);
}

template <uint BM, uint BN, uint BK, uint SK>
inline void bf16_nax_swiglu_bm_impl(
    device const bfloat *x,
    device const bfloat *wg,
    device const bfloat *wu,
    device bfloat *y,
    constant Bf16NaxParams &p,
    threadgroup float *parts,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint kCap = (BM * BN) / 32u;
  const uint rb = tg.y * BM;
  const uint n0 = tg.x * BN;
  if (rb >= p.rows || n0 >= p.n) {
    return;
  }

  const uint blocks = p.k / BK;
  const uint b0 = (sg * blocks) / SK;
  const uint b1 = ((sg + 1u) * blocks) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a_parent(
      (device bfloat *)(x + ulong(rb) * p.k),
      dextents<int32_t, 2>(p.k, p.rows - rb));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> bg_parent(
      (device bfloat *)(wg + ulong(n0) * p.k),
      dextents<int32_t, 2>(p.k, p.n - n0));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> bu_parent(
      (device bfloat *)(wu + ulong(n0) * p.k),
      dextents<int32_t, 2>(p.k, p.n - n0));

  auto a_init = a_parent.slice(0, 0);
  auto bg_init = bg_parent.slice(0, 0);
  auto pg = op.template get_destination_cooperative_tensor<
      decltype(a_init), decltype(bg_init), float>();
  auto pu = op.template get_destination_cooperative_tensor<
      decltype(a_init), decltype(bg_init), float>();
  const uint elems = pg.get_capacity();
  for (uint i = 0u; i < elems; ++i) {
    pg[i] = 0.0f;
    pu[i] = 0.0f;
  }

  for (uint blk = b0; blk < b1; ++blk) {
    const uint k_off = blk * BK;
    auto a = a_parent.slice(k_off, 0);
    auto bg = bg_parent.slice(k_off, 0);
    auto bu = bu_parent.slice(k_off, 0);
    op.run(a, bg, pg);
    op.run(a, bu, pu);
  }

  bf16_reduce_store_swiglu_slice<SK, kCap>(
      pg, pu, rb, n0, sg, lane, parts, y, p, false);
}

#define OMLX_M5_BF16_NAX_LINEAR_BM(BK, BM, BN, SK)                             \
  [[max_total_threads_per_threadgroup(32 * (SK))]]                             \
  kernel void omlx_m5_bf16_nax_bk##BK##_bm##BM##_n##BN##_sk##SK(               \
      device const bfloat *x [[buffer(0)]],                                    \
      device const bfloat *w [[buffer(1)]],                                    \
      device bfloat *y [[buffer(2)]],                                          \
      constant Bf16NaxParams &p [[buffer(3)]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[((SK) > 1u ? ((SK) - 1u) : 1u) *                   \
                            ((BM) * (BN))];                                    \
    bf16_nax_linear_bm_impl<BM, BN, BK, SK>(x, w, y, p, parts, lane, sg, tg);  \
  }

#define OMLX_M5_BF16_NAX_SWIGLU_BM(BK, BM, BN, SK)                             \
  [[max_total_threads_per_threadgroup(32 * (SK))]]                             \
  kernel void omlx_m5_bf16_nax_swiglu_bk##BK##_bm##BM##_n##BN##_sk##SK(        \
      device const bfloat *x [[buffer(0)]],                                    \
      device const bfloat *wg [[buffer(1)]],                                   \
      device const bfloat *wu [[buffer(2)]],                                   \
      device bfloat *y [[buffer(3)]],                                          \
      constant Bf16NaxParams &p [[buffer(4)]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[((SK) > 1u ? ((SK) - 1u) : 1u) *                   \
                            (2u * (BM) * (BN))];                               \
    bf16_nax_swiglu_bm_impl<BM, BN, BK, SK>(x, wg, wu, y, p, parts, lane, sg,  \
                                            tg);                               \
  }

#define OMLX_M5_BF16_NAX_BM_SUITE(BK)                                          \
  OMLX_M5_BF16_NAX_LINEAR_BM(BK, 16, 16, 2)                                    \
  OMLX_M5_BF16_NAX_LINEAR_BM(BK, 16, 16, 4)                                    \
  OMLX_M5_BF16_NAX_LINEAR_BM(BK, 16, 16, 8)                                    \
  OMLX_M5_BF16_NAX_LINEAR_BM(BK, 32, 16, 2)                                    \
  OMLX_M5_BF16_NAX_LINEAR_BM(BK, 32, 16, 4)                                    \
  OMLX_M5_BF16_NAX_LINEAR_BM(BK, 32, 16, 8)                                    \
  OMLX_M5_BF16_NAX_LINEAR_BM(BK, 16, 32, 2)                                    \
  OMLX_M5_BF16_NAX_LINEAR_BM(BK, 32, 32, 2)                                    \
  OMLX_M5_BF16_NAX_SWIGLU_BM(BK, 16, 16, 2)                                    \
  OMLX_M5_BF16_NAX_SWIGLU_BM(BK, 16, 16, 4)                                    \
  OMLX_M5_BF16_NAX_SWIGLU_BM(BK, 16, 16, 8)                                    \
  OMLX_M5_BF16_NAX_SWIGLU_BM(BK, 32, 16, 2)                                    \
  OMLX_M5_BF16_NAX_SWIGLU_BM(BK, 32, 16, 4)                                    \
  OMLX_M5_BF16_NAX_SWIGLU_BM(BK, 32, 16, 8)

OMLX_M5_BF16_NAX_BM_SUITE(128)
OMLX_M5_BF16_NAX_BM_SUITE(64)

struct Bf16LoraUpAccumParams {
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

// Metal 4 NAX cooperative matmul2d up-projection + in-place BF16 scaled accumulate
// for Mode B ("bypass") Prefill / Verify (T >= 16).
[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_lora_up_accum_bf16_nax(
    device const bfloat *tmp [[buffer(0)]],
    device const bfloat *b [[buffer(1)]],
    device bfloat *y [[buffer(2)]],
    constant Bf16LoraUpAccumParams &p [[buffer(3)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint BM = 16u;
  constexpr uint BN = 16u;
  constexpr uint BK = 16u;
  constexpr uint NSG = 4u;
  const uint rb = tg.y * BM;
  const uint n0 = (tg.x * NSG + sg) * BN;
  if (rb >= p.rows || n0 >= p.n || p.r_stride_b == 0u) {
    return;
  }

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a_parent(
      (device bfloat *)(tmp + ulong(rb) * p.tmp_stride),
      dextents<int32_t, 2>(p.tmp_stride, p.rows - rb));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> b_parent(
      (device bfloat *)(b + ulong(n0) * p.r_stride_b),
      dextents<int32_t, 2>(p.r_stride_b, p.n - n0));

  auto a_init = a_parent.slice(p.tmp_col_offset, 0);
  auto b_init = b_parent.slice(0, 0);
  auto prod = op.template get_destination_cooperative_tensor<
      decltype(a_init), decltype(b_init), float>();
  const uint elems = prod.get_capacity();
  for (uint i = 0u; i < elems; ++i) {
    prod[i] = 0.0f;
  }

  const uint blocks = p.r_stride_b / BK;
  for (uint blk = 0u; blk < blocks; ++blk) {
    const uint k_off = blk * BK;
    auto a_tile = a_parent.slice(p.tmp_col_offset + k_off, 0);
    auto b_tile = b_parent.slice(k_off, 0);
    op.run(a_tile, b_tile, prod);
  }

  for (uint i = 0u; i < elems; ++i) {
    if (prod.is_valid_element(i)) {
      const auto idx = prod.get_multidimensional_index(i);
      const uint col = n0 + uint(idx[0]);
      const uint row = rb + uint(idx[1]);
      if (row < p.rows && col < p.n) {
        const ulong dst = ulong(row) * p.y_stride + p.y_col_offset + col;
        y[dst] = bfloat(fma(p.scale, prod[i], float(y[dst])));
      }
    }
  }
}
