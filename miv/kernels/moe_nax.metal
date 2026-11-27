#include "common_math.h"
#include "fwht64.h"
#include <metal_atomic>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct MoeNaxBucketParams {
  uint tokens;
  uint top_k;
  uint h;
  uint max_tiles;
};

struct MoeNaxGemmParams {
  uint n;
  uint k;
  uint padded_k;
  uint max_tiles;
};

using MoeMxfp4ScaleTag =
    tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>;
using MoeMxfp4WeightTensor =
    tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline, MoeMxfp4ScaleTag>;

// Helper: Exact 0-ULP top-8 logit selection per token inside a 256-thread (8-SG) TG.
// Matches omlx_m5_forward_router_top8_logits bit-for-bit.
inline void moe_nax_router_top8_in_tg(
    device const bfloat *logits,
    device uint *indices,
    device bfloat *scores,
    uint tokens,
    uint lane,
    uint sg) {
  for (uint tok = sg; tok < tokens; tok += 8u) {
    float values[8];
    uint taken_mask = 0u;
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
      values[i] = float(logits[ulong(tok) * 256ul + lane * 8u + i]);
    }
    float selected[8];
    uint ids[8];
#pragma clang loop unroll(full)
    for (uint k = 0u; k < 8u; ++k) {
      int tier = -1;
      float best = -INFINITY;
      uint best_local = 0u;
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 8u; ++i) {
        if ((taken_mask & (1u << i)) != 0u) {
          continue;
        }
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
      const uint id = lane * 8u + best_local;
      const uint winner = simd_max(is_best ? id : 0u);
      if (is_best && id == winner) {
        taken_mask |= (1u << best_local);
      }
      selected[k] = global_tier == 1 ? maximum : -INFINITY;
      ids[k] = winner;
    }
    if (lane == 0u) {
      float sum = 0.0f;
      float probabilities[8];
      const float max_logit = selected[0];
      for (uint k = 0u; k < 8u; ++k) {
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
      for (uint k = 0u; k < 8u; ++k) {
        indices[ulong(tok) * 8ul + k] = ids[k];
        scores[ulong(tok) * 8ul + k] =
            sum > 0.0f && isfinite(sum) ? bfloat(probabilities[k] / sum)
                                        : bfloat(0.0f);
      }
    }
  }
}

inline void moe_nax_bucket_schedule_bm8_impl(
    device const uint *route_ids,
    device uint *tile_expert,
    device uint *tile_valid_rows,
    device uint *tile_assign,
    device uint *total_tiles_out,
    constant MoeNaxBucketParams &p,
    threadgroup ushort (&chunk_hist)[256][16],
    threadgroup uint (&tile_starts)[256],
    threadgroup uint (&sg_sums)[8],
    uint tid,
    uint lane,
    uint sg) {
  for (uint c = 0u; c < 16u; ++c) {
    chunk_hist[tid][c] = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint total_assign = p.tokens * p.top_k;
  const uint chunk_len = (total_assign + 15u) / 16u;

  if (tid < 16u && chunk_len > 0u) {
    const uint a0 = min(tid * chunk_len, total_assign);
    const uint a1 = min(a0 + chunk_len, total_assign);
    for (uint a = a0; a < a1; ++a) {
      const uint exp_id = route_ids[a];
      if (exp_id < 256u) {
        chunk_hist[exp_id][tid] = ushort(uint(chunk_hist[exp_id][tid]) + 1u);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint my_count = 0u;
  for (uint c = 0u; c < 16u; ++c) {
    const uint cnt = uint(chunk_hist[tid][c]);
    chunk_hist[tid][c] = ushort(my_count);
    my_count += cnt;
  }
  const uint my_tiles = (my_count + 7u) / 8u;

  const uint prefix_in_sg = simd_prefix_exclusive_sum(my_tiles);
  const uint sg_total = simd_sum(my_tiles);
  if (lane == 0u) {
    sg_sums[sg] = sg_total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint sg_base = 0u;
  for (uint s = 0u; s < sg; ++s) {
    sg_base += sg_sums[s];
  }
  const uint start_tile = sg_base + prefix_in_sg;
  tile_starts[tid] = start_tile;
  if (tid == 255u) {
    const uint capped = min(start_tile + my_tiles, p.max_tiles);
    total_tiles_out[0] = capped;
    total_tiles_out[1] = 1u;
    total_tiles_out[2] = max(capped, 1u);
    total_tiles_out[3] = 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint t = 0u; t < my_tiles; ++t) {
    const uint global_t = start_tile + t;
    if (global_t < p.max_tiles) {
      tile_expert[global_t] = tid;
      const uint rem = my_count - t * 8u;
      const uint vr = min(8u, rem);
      tile_valid_rows[global_t] = vr;
      for (uint r = vr; r < 8u; ++r) {
        tile_assign[global_t * 8u + r] = 0xFFFFFFFFu;
      }
    }
  }

  if (tid < 16u && chunk_len > 0u) {
    const uint a0 = min(tid * chunk_len, total_assign);
    const uint a1 = min(a0 + chunk_len, total_assign);
    for (uint a = a0; a < a1; ++a) {
      const uint exp_id = route_ids[a];
      if (exp_id < 256u) {
        const uint cursor = uint(chunk_hist[exp_id][tid]);
        chunk_hist[exp_id][tid] = ushort(cursor + 1u);
        const uint global_t = tile_starts[exp_id] + (cursor / 8u);
        const uint rm = cursor % 8u;
        if (global_t < p.max_tiles) {
          tile_assign[global_t * 8u + rm] = a;
        }
      }
    }
  }
}

template <uint WBM>
inline void moe_nax_bucket_schedule_wide_impl(
    device const uint *route_ids,
    device uint *tile_expert,
    device uint *tile_valid_rows,
    device uint *tile_assign,
    device uint *total_tiles_out,
    constant MoeNaxBucketParams &p,
    threadgroup ushort (&chunk_hist)[256][16],
    threadgroup uint (&tile_starts_8)[256],
    threadgroup uint (&tile_starts_w)[256],
    threadgroup uint (&sg_sums_8)[8],
    threadgroup uint (&sg_sums_w)[8],
    uint tid,
    uint lane,
    uint sg) {
  static_assert(WBM == 32u || WBM == 64u, "WBM must be 32 or 64");
  constexpr uint kEightPerW = WBM / 8u;
  for (uint c = 0u; c < 16u; ++c) {
    chunk_hist[tid][c] = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint total_assign = p.tokens * p.top_k;
  const uint chunk_len = (total_assign + 15u) / 16u;

  if (tid < 16u && chunk_len > 0u) {
    const uint a0 = min(tid * chunk_len, total_assign);
    const uint a1 = min(a0 + chunk_len, total_assign);
    for (uint a = a0; a < a1; ++a) {
      const uint exp_id = route_ids[a];
      if (exp_id < 256u) {
        chunk_hist[exp_id][tid] = ushort(uint(chunk_hist[exp_id][tid]) + 1u);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint my_count = 0u;
  for (uint c = 0u; c < 16u; ++c) {
    const uint cnt = uint(chunk_hist[tid][c]);
    chunk_hist[tid][c] = ushort(my_count);
    my_count += cnt;
  }
  const uint my_8tiles = (my_count + 7u) / 8u;
  const uint my_wtiles = (my_count + (WBM - 1u)) / WBM;

  const uint prefix_8 = simd_prefix_exclusive_sum(my_8tiles);
  const uint sg_total_8 = simd_sum(my_8tiles);
  const uint prefix_w = simd_prefix_exclusive_sum(my_wtiles);
  const uint sg_total_w = simd_sum(my_wtiles);
  if (lane == 0u) {
    sg_sums_8[sg] = sg_total_8;
    sg_sums_w[sg] = sg_total_w;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint sg_base_8 = 0u;
  uint sg_base_w = 0u;
  for (uint s = 0u; s < sg; ++s) {
    sg_base_8 += sg_sums_8[s];
    sg_base_w += sg_sums_w[s];
  }
  const uint start_8tile = sg_base_8 + prefix_8;
  const uint start_wtile = sg_base_w + prefix_w;
  tile_starts_8[tid] = start_8tile;
  tile_starts_w[tid] = start_wtile;
  if (tid == 255u) {
    const uint capped_8 = min(start_8tile + my_8tiles, p.max_tiles);
    const uint capped_w = min(start_wtile + my_wtiles, p.max_tiles);
    total_tiles_out[0] = capped_8;
    total_tiles_out[1] = capped_w;
    total_tiles_out[2] = max(capped_8, 1u);
    total_tiles_out[3] = max(capped_w, 1u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint t8 = 0u; t8 < my_8tiles; ++t8) {
    const uint global_t8 = start_8tile + t8;
    if (global_t8 < p.max_tiles) {
      const uint rem8 = my_count - t8 * 8u;
      const uint vr8 = min(8u, rem8);
      for (uint r = vr8; r < 8u; ++r) {
        tile_assign[global_t8 * 8u + r] = 0xFFFFFFFFu;
      }
    }
  }

  for (uint tw = 0u; tw < my_wtiles; ++tw) {
    const uint global_tw = start_wtile + tw;
    if (global_tw < p.max_tiles) {
      const uint first_8tile = start_8tile + tw * kEightPerW;
      const uint rem_w = my_count - tw * WBM;
      const uint vr_w = (first_8tile < p.max_tiles)
                            ? min(min(WBM, rem_w), (p.max_tiles - first_8tile) * 8u)
                            : 0u;
      tile_expert[global_tw] = (first_8tile << 16u) | tid;
      tile_valid_rows[global_tw] = vr_w;
    }
  }

  if (tid < 16u && chunk_len > 0u) {
    const uint a0 = min(tid * chunk_len, total_assign);
    const uint a1 = min(a0 + chunk_len, total_assign);
    for (uint a = a0; a < a1; ++a) {
      const uint exp_id = route_ids[a];
      if (exp_id < 256u) {
        const uint cursor = uint(chunk_hist[exp_id][tid]);
        chunk_hist[exp_id][tid] = ushort(cursor + 1u);
        const uint global_t8 = tile_starts_8[exp_id] + (cursor / 8u);
        const uint rm = cursor % 8u;
        if (global_t8 < p.max_tiles) {
          tile_assign[global_t8 * 8u + rm] = a;
        }
      }
    }
  }
}

// Step 1: Deterministic parallel 16-chunk bucketing of (tokens * top_k) expert
// assignments into BM=8 tiles per expert (up to 256 experts).
// Avoids 256x redundant full-array scans while preserving exact ascending
// assignment order within every expert tile.
kernel void omlx_m5_moe_nax_bucket_schedule(
    device const uint *route_ids [[buffer(0)]],
    device uint *tile_expert [[buffer(1)]],
    device uint *tile_valid_rows [[buffer(2)]],
    device uint *tile_assign [[buffer(3)]],
    device uint *total_tiles_out [[buffer(4)]],
    constant MoeNaxBucketParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup ushort chunk_hist[256][16];
  threadgroup uint tile_starts[256];
  threadgroup uint sg_sums[8];
  moe_nax_bucket_schedule_bm8_impl(
      route_ids, tile_expert, tile_valid_rows, tile_assign, total_tiles_out, p,
      chunk_hist, tile_starts, sg_sums, tid, lane, sg);
}

kernel void omlx_m5_moe_nax_router_top8_bucket_schedule(
    device const bfloat *router_logits [[buffer(0)]],
    device uint *route_ids [[buffer(1)]],
    device bfloat *route_scores [[buffer(2)]],
    device uint *tile_expert [[buffer(3)]],
    device uint *tile_valid_rows [[buffer(4)]],
    device uint *tile_assign [[buffer(5)]],
    device uint *total_tiles_out [[buffer(6)]],
    constant MoeNaxBucketParams &p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup ushort chunk_hist[256][16];
  threadgroup uint tile_starts[256];
  threadgroup uint sg_sums[8];
  moe_nax_router_top8_in_tg(router_logits, route_ids, route_scores, p.tokens, lane, sg);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  moe_nax_bucket_schedule_bm8_impl(
      route_ids, tile_expert, tile_valid_rows, tile_assign, total_tiles_out, p,
      chunk_hist, tile_starts, sg_sums, tid, lane, sg);
}

kernel void omlx_m5_moe_nax_bucket_schedule_bm32(
    device const uint *route_ids [[buffer(0)]],
    device uint *tile_expert [[buffer(1)]],
    device uint *tile_valid_rows [[buffer(2)]],
    device uint *tile_assign [[buffer(3)]],
    device uint *total_tiles_out [[buffer(4)]],
    constant MoeNaxBucketParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup ushort chunk_hist[256][16];
  threadgroup uint tile_starts_8[256];
  threadgroup uint tile_starts_w[256];
  threadgroup uint sg_sums_8[8];
  threadgroup uint sg_sums_w[8];
  moe_nax_bucket_schedule_wide_impl<32u>(
      route_ids, tile_expert, tile_valid_rows, tile_assign, total_tiles_out, p,
      chunk_hist, tile_starts_8, tile_starts_w, sg_sums_8, sg_sums_w, tid, lane, sg);
}

kernel void omlx_m5_moe_nax_router_top8_bucket_schedule_bm32(
    device const bfloat *router_logits [[buffer(0)]],
    device uint *route_ids [[buffer(1)]],
    device bfloat *route_scores [[buffer(2)]],
    device uint *tile_expert [[buffer(3)]],
    device uint *tile_valid_rows [[buffer(4)]],
    device uint *tile_assign [[buffer(5)]],
    device uint *total_tiles_out [[buffer(6)]],
    constant MoeNaxBucketParams &p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup ushort chunk_hist[256][16];
  threadgroup uint tile_starts_8[256];
  threadgroup uint tile_starts_w[256];
  threadgroup uint sg_sums_8[8];
  threadgroup uint sg_sums_w[8];
  moe_nax_router_top8_in_tg(router_logits, route_ids, route_scores, p.tokens, lane, sg);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  moe_nax_bucket_schedule_wide_impl<32u>(
      route_ids, tile_expert, tile_valid_rows, tile_assign, total_tiles_out, p,
      chunk_hist, tile_starts_8, tile_starts_w, sg_sums_8, sg_sums_w, tid, lane, sg);
}

kernel void omlx_m5_moe_nax_bucket_schedule_bm64(
    device const uint *route_ids [[buffer(0)]],
    device uint *tile_expert [[buffer(1)]],
    device uint *tile_valid_rows [[buffer(2)]],
    device uint *tile_assign [[buffer(3)]],
    device uint *total_tiles_out [[buffer(4)]],
    constant MoeNaxBucketParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup ushort chunk_hist[256][16];
  threadgroup uint tile_starts_8[256];
  threadgroup uint tile_starts_w[256];
  threadgroup uint sg_sums_8[8];
  threadgroup uint sg_sums_w[8];
  moe_nax_bucket_schedule_wide_impl<64u>(
      route_ids, tile_expert, tile_valid_rows, tile_assign, total_tiles_out, p,
      chunk_hist, tile_starts_8, tile_starts_w, sg_sums_8, sg_sums_w, tid, lane, sg);
}

kernel void omlx_m5_moe_nax_router_top8_bucket_schedule_bm64(
    device const bfloat *router_logits [[buffer(0)]],
    device uint *route_ids [[buffer(1)]],
    device bfloat *route_scores [[buffer(2)]],
    device uint *tile_expert [[buffer(3)]],
    device uint *tile_valid_rows [[buffer(4)]],
    device uint *tile_assign [[buffer(5)]],
    device uint *total_tiles_out [[buffer(6)]],
    constant MoeNaxBucketParams &p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup ushort chunk_hist[256][16];
  threadgroup uint tile_starts_8[256];
  threadgroup uint tile_starts_w[256];
  threadgroup uint sg_sums_8[8];
  threadgroup uint sg_sums_w[8];
  moe_nax_router_top8_in_tg(router_logits, route_ids, route_scores, p.tokens, lane, sg);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  moe_nax_bucket_schedule_wide_impl<64u>(
      route_ids, tile_expert, tile_valid_rows, tile_assign, total_tiles_out, p,
      chunk_hist, tile_starts_8, tile_starts_w, sg_sums_8, sg_sums_w, tid, lane, sg);
}

// Step 2 (MXFP4): Gather & cast BF16 norm rows into tiled FP16 [max_tiles * 8, h] with exact power-of-2 row scale.
kernel void omlx_m5_moe_nax_gather_f16(
    device const bfloat *norm [[buffer(0)]],
    device const uint *tile_assign [[buffer(1)]],
    device const uint *total_tiles [[buffer(2)]],
    device half *x_f16 [[buffer(3)]],
    device float *row_scale [[buffer(4)]],
    constant MoeNaxBucketParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint out_row = tg.y;
  const uint active_rows = total_tiles[0] * 8u;
  if (out_row >= active_rows || out_row >= p.max_tiles * 8u) {
    return;
  }
  const uint assign = tile_assign[out_row];
  const uint h4 = p.h / 4u;
  device half4 *dst_row = reinterpret_cast<device half4 *>(x_f16 + ulong(out_row) * p.h);

  if (assign == 0xFFFFFFFFu) {
    for (uint c = tid; c < h4; c += 128u) {
      dst_row[c] = half4(0.0h);
    }
    if (tid == 0u) {
      row_scale[out_row] = 1.0f;
    }
    return;
  }

  const uint token = assign / p.top_k;
  device const bfloat4 *src_row =
      reinterpret_cast<device const bfloat4 *>(norm + ulong(token) * p.h);

  float local_max = 0.0f;
  for (uint c = tid; c < h4; c += 128u) {
    const float4 v = fabs(float4(src_row[c]));
    local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));
  }
  local_max = simd_max(local_max);

  threadgroup float smax[4];
  threadgroup float inv_scale_tg;
  if (tid == 0u) {
    inv_scale_tg = 1.0f;
  }
  if (lane == 0u) {
    smax[sg] = local_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    const float gmax = max(max(smax[0], smax[1]), max(smax[2], smax[3]));
    float rs = 1.0f;
    float inv_rs = 1.0f;
    nax_compute_row_scale_f16(gmax, rs, inv_rs);
    row_scale[out_row] = rs;
    inv_scale_tg = inv_rs;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float inv = inv_scale_tg;
  for (uint c = tid; c < h4; c += 128u) {
    dst_row[c] = half4(float4(src_row[c]) * inv);
  }
}

// Step 3 (MXFP4): Native M5 NAX Fused Gate + Up + SwiGLU across expert tiles (zero weight repack!)
template <uint NSG>
inline void moe_mxfp4_nax_swiglu_n16_sk4_impl(
    device const half *x_f16,
    device const float *row_scale,
    device const uint *tile_expert,
    device const uint *tile_valid_rows,
    device const uint *total_tiles,
    device const uchar *g_codes,
    device const uchar *g_scales,
    device const uchar *u_codes,
    device const uchar *u_scales,
    device half *mid_f16,
    constant MoeNaxGemmParams &p,
    threadgroup float *parts,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM = 8u;
  constexpr uint BN = 16u;
  constexpr uint BK = 128u;
  constexpr uint SK = 4u;
  constexpr uint kCap = (16u * BN) / 32u;

  const uint m_tile = tg.y;
  const uint tg_n0 = tg.x * (BN * NSG);
  if (m_tile >= total_tiles[0] || tg_n0 >= p.n) {
    return;
  }
  const uint sg_n = (NSG > 1u) ? (sg / SK) : 0u;
  const uint sg_k = (NSG > 1u) ? (sg % SK) : sg;
  const uint n0 = (tg.x * NSG + sg_n) * BN;
  const bool col_active = (sg_n < NSG) && (n0 < p.n);

  const uint expert = tile_expert[m_tile];
  const uint valid_rows = tile_valid_rows[m_tile];
  const uint rb = m_tile * BM;

  const uint blocks = p.k / BK;
  const uint b0 = (sg_k * blocks) / SK;
  const uint b1 = ((sg_k + 1u) * blocks) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  tensor<device half, dextents<int32_t, 2>, tensor_inline> a_full(
      (device half *)(x_f16 + ulong(rb) * p.k), dextents<int32_t, 2>(p.k, BM));

  const ulong exp_c_off = ulong(expert) * p.n * (p.padded_k / 2u);
  const ulong exp_s_off = ulong(expert) * p.n * (p.padded_k / 32u);
  MoeMxfp4WeightTensor bg_full(
      (device uchar *)(g_codes + exp_c_off),
      dextents<int32_t, 2>(p.padded_k, p.n),
      MoeMxfp4ScaleTag((device uchar *)(g_scales + exp_s_off)));
  MoeMxfp4WeightTensor bu_full(
      (device uchar *)(u_codes + exp_c_off),
      dextents<int32_t, 2>(p.padded_k, p.n),
      MoeMxfp4ScaleTag((device uchar *)(u_scales + exp_s_off)));

  auto a_init = a_full.slice(0, 0);
  auto bg_init = bg_full.slice(0, 0);
  auto pg = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  auto pu = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bg_init), float>();
  const uint elems = pg.get_capacity();
  for (uint i = 0; i < elems; ++i) {
    pg[i] = 0.0f;
    pu[i] = 0.0f;
  }

  if (col_active) {
    for (uint blk = b0; blk < b1; ++blk) {
      const uint k_off = blk * BK;
      auto a = a_full.slice(k_off, 0);
      auto bg = bg_full.slice(k_off, n0);
      auto bu = bu_full.slice(k_off, n0);
      op.run(a, bg, pg);
      op.run(a, bu, pu);
    }
  }

  constexpr uint kPartStride = (SK - 1u) * 2u * kCap * 32u;
  threadgroup float *my_parts = parts + ((NSG > 1u && sg_n < NSG) ? (sg_n * kPartStride) : 0u);
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
    for (uint i = 0; i < elems; ++i) {
      if (pg.is_valid_element(i)) {
        const auto idx = pg.get_multidimensional_index(i);
        const uint col = n0 + uint(idx[0]);
        const uint rm = uint(idx[1]);
        if (col < p.n) {
          if (rm < valid_rows) {
            const float rs = row_scale[rb + rm];
            const float gv = float(bfloat(pg[i] * rs));
            const float uv = float(bfloat(pu[i] * rs));
            const float ov = float(bfloat(nax_silu_f32(gv) * uv));
            mid_f16[ulong(rb + rm) * p.n + col] = half(clamp(ov, -65504.0f, 65504.0f));
          } else {
            mid_f16[ulong(rb + rm) * p.n + col] = half(0.0h);
          }
        }
      }
    }
  }
}

kernel void omlx_m5_moe_mxfp4_nax_swiglu_n16_sk4(
    device const half *x_f16 [[buffer(0)]],
    device const float *row_scale [[buffer(1)]],
    device const uint *tile_expert [[buffer(2)]],
    device const uint *tile_valid_rows [[buffer(3)]],
    device const uint *total_tiles [[buffer(4)]],
    device const uchar *g_codes [[buffer(5)]],
    device const uchar *g_scales [[buffer(6)]],
    device const uchar *u_codes [[buffer(7)]],
    device const uchar *u_scales [[buffer(8)]],
    device half *mid_f16 [[buffer(9)]],
    constant MoeNaxGemmParams &p [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float parts[3u * 16u * 32u];
  moe_mxfp4_nax_swiglu_n16_sk4_impl<1u>(
      x_f16, row_scale, tile_expert, tile_valid_rows, total_tiles, g_codes,
      g_scales, u_codes, u_scales, mid_f16, p, parts, lane, sg, tg);
}

kernel void omlx_m5_moe_mxfp4_nax_swiglu_n16_sk4_nsg2(
    device const half *x_f16 [[buffer(0)]],
    device const float *row_scale [[buffer(1)]],
    device const uint *tile_expert [[buffer(2)]],
    device const uint *tile_valid_rows [[buffer(3)]],
    device const uint *total_tiles [[buffer(4)]],
    device const uchar *g_codes [[buffer(5)]],
    device const uchar *g_scales [[buffer(6)]],
    device const uchar *u_codes [[buffer(7)]],
    device const uchar *u_scales [[buffer(8)]],
    device half *mid_f16 [[buffer(9)]],
    constant MoeNaxGemmParams &p [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float parts[2u * 3u * 16u * 32u];
  moe_mxfp4_nax_swiglu_n16_sk4_impl<2u>(
      x_f16, row_scale, tile_expert, tile_valid_rows, total_tiles, g_codes,
      g_scales, u_codes, u_scales, mid_f16, p, parts, lane, sg, tg);
}

// Step 4 (MXFP4): Native M5 NAX Down Projection + Scatter weighted result to expert_output[assignment, col]
template <uint NSG>
inline void moe_mxfp4_nax_down_scatter_n16_sk2_impl(
    device const half *mid_f16,
    device const uint *tile_expert,
    device const uint *tile_valid_rows,
    device const uint *tile_assign,
    device const uint *total_tiles,
    device const bfloat *route_scores,
    device const uchar *d_codes,
    device const uchar *d_scales,
    device bfloat *expert_output,
    constant MoeNaxGemmParams &p,
    threadgroup float *parts,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM = 8u;
  constexpr uint BN = 16u;
  constexpr uint BK = 128u;
  constexpr uint SK = 2u;
  constexpr uint kCap = (16u * BN) / 32u;

  const uint m_tile = tg.y;
  const uint tg_n0 = tg.x * (BN * NSG);
  if (m_tile >= total_tiles[0] || tg_n0 >= p.n) {
    return;
  }
  const uint sg_n = (NSG > 1u) ? (sg / SK) : 0u;
  const uint sg_k = (NSG > 1u) ? (sg % SK) : sg;
  const uint n0 = (tg.x * NSG + sg_n) * BN;
  const bool col_active = (sg_n < NSG) && (n0 < p.n);

  const uint expert = tile_expert[m_tile];
  const uint valid_rows = tile_valid_rows[m_tile];
  const uint rb = m_tile * BM;

  const uint blocks = p.k / BK;
  const uint b0 = (sg_k * blocks) / SK;
  const uint b1 = ((sg_k + 1u) * blocks) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  tensor<device half, dextents<int32_t, 2>, tensor_inline> a_full(
      (device half *)(mid_f16 + ulong(rb) * p.k), dextents<int32_t, 2>(p.k, BM));

  const ulong exp_c_off = ulong(expert) * p.n * (p.padded_k / 2u);
  const ulong exp_s_off = ulong(expert) * p.n * (p.padded_k / 32u);
  MoeMxfp4WeightTensor bd_full(
      (device uchar *)(d_codes + exp_c_off),
      dextents<int32_t, 2>(p.padded_k, p.n),
      MoeMxfp4ScaleTag((device uchar *)(d_scales + exp_s_off)));

  auto a_init = a_full.slice(0, 0);
  auto bd_init = bd_full.slice(0, 0);
  auto pd = op.template get_destination_cooperative_tensor<decltype(a_init), decltype(bd_init), float>();
  const uint elems = pd.get_capacity();
  for (uint i = 0; i < elems; ++i) {
    pd[i] = 0.0f;
  }

  if (col_active) {
    for (uint blk = b0; blk < b1; ++blk) {
      const uint k_off = blk * BK;
      auto a = a_full.slice(k_off, 0);
      auto bd = bd_full.slice(k_off, n0);
      op.run(a, bd, pd);
    }
  }

  constexpr uint kPartStride = (SK - 1u) * kCap * 32u;
  threadgroup float *my_parts = parts + ((NSG > 1u && sg_n < NSG) ? (sg_n * kPartStride) : 0u);
  if (sg_k > 0u && col_active) {
    for (uint i = 0; i < elems; ++i) {
      my_parts[((sg_k - 1u) * kCap + i) * 32u + lane] = pd[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg_k == 0u && col_active) {
    for (uint s = 1u; s < SK; ++s) {
      for (uint i = 0; i < elems; ++i) {
        pd[i] += my_parts[((s - 1u) * kCap + i) * 32u + lane];
      }
    }
    for (uint i = 0; i < elems; ++i) {
      if (pd.is_valid_element(i)) {
        const auto idx = pd.get_multidimensional_index(i);
        const uint col = n0 + uint(idx[0]);
        const uint rm = uint(idx[1]);
        if (rm < valid_rows && col < p.n) {
          const uint assign = tile_assign[rb + rm];
          const float score = float(route_scores[assign]);
          expert_output[ulong(assign) * p.n + col] = bfloat(pd[i] * score);
        }
      }
    }
  }
}

kernel void omlx_m5_moe_mxfp4_nax_down_scatter_n16_sk2(
    device const half *mid_f16 [[buffer(0)]],
    device const uint *tile_expert [[buffer(1)]],
    device const uint *tile_valid_rows [[buffer(2)]],
    device const uint *tile_assign [[buffer(3)]],
    device const uint *total_tiles [[buffer(4)]],
    device const bfloat *route_scores [[buffer(5)]],
    device const uchar *d_codes [[buffer(6)]],
    device const uchar *d_scales [[buffer(7)]],
    device bfloat *expert_output [[buffer(8)]],
    constant MoeNaxGemmParams &p [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float parts[1u * 8u * 32u];
  moe_mxfp4_nax_down_scatter_n16_sk2_impl<1u>(
      mid_f16, tile_expert, tile_valid_rows, tile_assign, total_tiles,
      route_scores, d_codes, d_scales, expert_output, p, parts, lane, sg, tg);
}

kernel void omlx_m5_moe_mxfp4_nax_down_scatter_n16_sk2_nsg2(
    device const half *mid_f16 [[buffer(0)]],
    device const uint *tile_expert [[buffer(1)]],
    device const uint *tile_valid_rows [[buffer(2)]],
    device const uint *tile_assign [[buffer(3)]],
    device const uint *total_tiles [[buffer(4)]],
    device const bfloat *route_scores [[buffer(5)]],
    device const uchar *d_codes [[buffer(6)]],
    device const uchar *d_scales [[buffer(7)]],
    device bfloat *expert_output [[buffer(8)]],
    constant MoeNaxGemmParams &p [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float parts[2u * 1u * 8u * 32u];
  moe_mxfp4_nax_down_scatter_n16_sk2_impl<2u>(
      mid_f16, tile_expert, tile_valid_rows, tile_assign, total_tiles,
      route_scores, d_codes, d_scales, expert_output, p, parts, lane, sg, tg);
}

// ============================================================================
// Wide Zero-Gather MoE NAX Helpers: Register-Fragment 16x32x16 relaxed_precision
// MMA + Bank-Conflict-Padded (BKP = 72) Cooperative As & Ws Threadgroup Staging
// ============================================================================

inline void moe_nax_mma_16x32x16(
    thread vec<float, 8> &Cn0,
    thread vec<float, 8> &Cn1,
    thread const vec<bfloat, 8> &A,
    thread const vec<bfloat, 8> &Bn0,
    thread const vec<bfloat, 8> &Bn1) {
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b =
      gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
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

inline bool moe_has_apple10_nax_layout() {
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  const auto c0 = ct_a.get_multidimensional_index(ushort(0u));
  const auto c2 = ct_a.get_multidimensional_index(ushort(2u));
  return (int(c2[0]) - int(c0[0]) == 2) && (int(c2[1]) == int(c0[1]));
}

template <uint BKP>
inline vec<bfloat, 8> moe_nax_load_frag_tg(
    const threadgroup bfloat *base,
    uint fm,
    uint fn) {
  const bfloat4 v0 =
      *reinterpret_cast<const threadgroup bfloat4 *>(base + fm * BKP + fn);
  const bfloat4 v1 =
      *reinterpret_cast<const threadgroup bfloat4 *>(base + (fm + 8u) * BKP + fn);
  return vec<bfloat, 8>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
}

inline void moe_nax_fetch_mxfp4_32(
    device const uchar *codes,
    device const uchar *scales,
    ulong expert_row,
    uint padded_k,
    uint k_col,
    bool row_live,
    thread uint4 &raw_out,
    thread float &sc_out) {
  if (row_live) {
    raw_out = *reinterpret_cast<device const uint4 *>(
        codes + expert_row * ulong(padded_k / 2u) + (k_col / 2u));
    sc_out = e8m0_to_float(scales[expert_row * ulong(padded_k / 32u) + (k_col / 32u)]);
  } else {
    raw_out = uint4(0u);
    sc_out = 0.0f;
  }
}

inline void moe_nax_unpack_mxfp4_32(
    uint4 raw,
    float sc,
    threadgroup vec<bfloat, 8> *w_dst8) {
  #pragma clang loop unroll(full)
  for (uint w = 0u; w < 4u; ++w) {
    const uint wd = raw[w];
    vec<bfloat, 8> out;
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 8u; ++j) {
      out[j] = bfloat(sc * fp4_nibble_to_float((wd >> (j * 4u)) & 15u));
    }
    w_dst8[w] = out;
  }
}

template <uint WBM>
inline void moe_mxfp4_nax_swiglu_wide_sk1_impl(
    device const bfloat *norm,
    device const uint *tile_assign,
    device const uint *tile_expert,
    device const uint *tile_valid_rows,
    device const uint *total_tiles,
    device const uchar *g_codes,
    device const uchar *g_scales,
    device const uchar *u_codes,
    device const uchar *u_scales,
    device bfloat *mid_bf16,
    constant MoeNaxGemmParams &p,
    threadgroup bfloat *Ws,
    threadgroup uint *tg_tok,
    uint tid,
    uint lane,
    uint sg,
    uint3 tg) {
  static_assert(WBM == 32u || WBM == 64u, "WBM must be 32 or 64");
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint kOutColsPerTg = 32u;
  constexpr uint kSM = WBM / 2u;
  constexpr uint kTM = kSM / 16u;

  const uint n_cols = (p.n + kOutColsPerTg - 1u) / kOutColsPerTg;
  const bool use_gx32 = (p.max_tiles >> 31u) != 0u;
  const uint m_tile = use_gx32 ? ((tg.y / n_cols) * 32u + tg.x) : tg.y;
  const uint col_tile = use_gx32 ? (tg.y % n_cols) : tg.x;
  const uint tg_n0 = col_tile * kOutColsPerTg;
  if (m_tile >= total_tiles[1] || col_tile >= n_cols || tg_n0 >= p.n) {
    return;
  }
  const uint valid_rows = tile_valid_rows[m_tile];
  if (valid_rows == 0u) {
    return;
  }
  const uint packed = tile_expert[m_tile];
  const uint expert = packed & 0xFFFFu;
  const uint rb = (packed >> 16u) * 8u;

  if (tid < WBM) {
    uint tok = 0xFFFFFFFFu;
    if (tid < valid_rows) {
      const uint assign = tile_assign[rb + tid];
      if (assign != 0xFFFFFFFFu) {
        tok = assign >> 3u;
      }
    }
    tg_tok[tid] = tok;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const bool compact32 = (WBM == 64u) && (valid_rows <= 32u);
  const uint m_off = sg_m * (compact32 ? 16u : kSM);
  const bool sg_active = (m_off < valid_rows);

  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

  ulong a_off[kTM][2];
  bool a_ok[kTM][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < kTM; ++mm) {
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint rm = m_off + mm * 16u + fm + h * 8u;
      const uint tok = ((mm == 0u || !compact32) && rm < valid_rows) ? tg_tok[rm] : 0xFFFFFFFFu;
      a_ok[mm][h] = (tok != 0xFFFFFFFFu);
      a_off[mm][h] = a_ok[mm][h] ? (ulong(tok) * p.k + fn) : 0ul;
    }
  }

  vec<float, 8> D[kTM][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < kTM; ++mm) {
    D[mm][0] = vec<float, 8>(0.0f);
    D[mm][1] = vec<float, 8>(0.0f);
  }

  const uint w_tile_row = tid >> 1u;
  const uint w_col_sub = (tid & 1u) * 32u;
  const bool is_up = ((w_tile_row >> 4u) & 1u) != 0u;
  const uint out_col = tg_n0 + ((w_tile_row >> 5u) * 16u) + (w_tile_row & 15u);
  const bool w_row_live = (out_col < p.n);
  const ulong expert_row = ulong(expert) * p.n + out_col;
  device const uchar *sel_codes = is_up ? u_codes : g_codes;
  device const uchar *sel_scales = is_up ? u_scales : g_scales;
  threadgroup vec<bfloat, 8> *w_dst8 =
      reinterpret_cast<threadgroup vec<bfloat, 8> *>(Ws + w_tile_row * kBKP + w_col_sub);

  if (!moe_has_apple10_nax_layout()) {
    constexpr auto desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> gemm_op;
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c0 = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, float>();
    auto ct_c1 = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      ct_c0[i] = 0.0f;
      ct_c1[i] = 0.0f;
    }
    for (uint k0 = 0u; k0 < p.k; k0 += kBK) {
      uint4 raw;
      float sc;
      moe_nax_fetch_mxfp4_32(
          sel_codes, sel_scales, expert_row, p.padded_k, k0 + w_col_sub, w_row_live, raw, sc);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      moe_nax_unpack_mxfp4_32(raw, sc, w_dst8);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg_active) {
        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < kBK; kk += 16u) {
          const threadgroup bfloat *ws_sub = Ws + (sg_n * 32u) * kBKP + kk;
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            const auto ib = ct_b.get_multidimensional_index(i);
            ct_b[i] = ws_sub[uint(ib[1]) * kBKP + uint(ib[0])];
          }
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 8u; ++i) {
            const auto ia = ct_a.get_multidimensional_index(i);
            const uint rm = m_off + uint(ia[1]);
            const uint tok = (rm < valid_rows) ? tg_tok[rm] : 0xFFFFFFFFu;
            ct_a[i] = (tok != 0xFFFFFFFFu)
                ? norm[ulong(tok) * p.k + k0 + kk + uint(ia[0])]
                : bfloat(0.0f);
          }
          gemm_op.run(ct_a, ct_b, ct_c0);
          if (kTM > 1u && !compact32 && m_off + 16u < valid_rows) {
            #pragma clang loop unroll(full)
            for (ushort i = 0u; i < 8u; ++i) {
              const auto ia = ct_a.get_multidimensional_index(i);
              const uint rm = m_off + 16u + uint(ia[1]);
              const uint tok = (rm < valid_rows) ? tg_tok[rm] : 0xFFFFFFFFu;
              ct_a[i] = (tok != 0xFFFFFFFFu)
                  ? norm[ulong(tok) * p.k + k0 + kk + uint(ia[0])]
                  : bfloat(0.0f);
            }
            gemm_op.run(ct_a, ct_b, ct_c1);
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_active) {
      threadgroup bfloat *sg_buf = Ws + sg * 512u;
      const uint col_base = tg_n0 + sg_n * 16u;
      const uint rb_8ceil = (valid_rows + 7u) & ~7u;
      const uint r_out_base = (lane >> 2u) * 2u;
      const uint c_out4 = (lane & 3u) * 4u;
      const uint col = col_base + c_out4;
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < kTM; ++mm) {
        if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            if (ct_c0.is_valid_element(i)) {
              const auto idx = ct_c0.get_multidimensional_index(i);
              sg_buf[uint(idx[1]) * 32u + uint(idx[0])] =
                  bfloat((mm == 0u) ? ct_c0[i] : ct_c1[i]);
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          if (col < p.n) {
            #pragma clang loop unroll(full)
            for (uint dr = 0u; dr < 2u; ++dr) {
              const uint r = r_out_base + dr;
              const uint rm = m_off + mm * 16u + r;
              if (rm < valid_rows) {
                const bfloat4 g4 = *reinterpret_cast<const threadgroup bfloat4 *>(sg_buf + r * 32u + c_out4);
                const bfloat4 u4 = *reinterpret_cast<const threadgroup bfloat4 *>(sg_buf + r * 32u + 16u + c_out4);
                bfloat4 out_b;
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  out_b[j] = bfloat(nax_silu_f32(float(g4[j])) * float(u4[j]));
                }
                *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb + rm) * p.n + col) = out_b;
              } else if (rm < rb_8ceil) {
                *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb + rm) * p.n + col) = bfloat4(bfloat(0.0f));
              }
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    }
    return;
  }

  const threadgroup bfloat *ws_sg = Ws + (sg_n * 32u + fm) * kBKP + fn;

  for (uint k0 = 0u; k0 < p.k; k0 += kBK) {
    uint4 raw;
    float sc;
    moe_nax_fetch_mxfp4_32(
        sel_codes, sel_scales, expert_row, p.padded_k, k0 + w_col_sub, w_row_live, raw, sc);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    moe_nax_unpack_mxfp4_32(raw, sc, w_dst8);
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
        for (uint mm = 0u; mm < kTM; ++mm) {
          if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
            vec<bfloat, 8> a0, a1;
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 2u; ++r) {
              const bfloat4 va0 = a_ok[mm][r]
                  ? *reinterpret_cast<device const bfloat4 *>(norm + a_off[mm][r] + k0 + kk + 0u)
                  : bfloat4(bfloat(0.0f));
              const bfloat4 va1 = a_ok[mm][r]
                  ? *reinterpret_cast<device const bfloat4 *>(norm + a_off[mm][r] + k0 + kk + 16u)
                  : bfloat4(bfloat(0.0f));
              #pragma clang loop unroll(full)
              for (uint c = 0u; c < 4u; ++c) {
                a0[r * 4u + c] = va0[c];
                a1[r * 4u + c] = va1[c];
              }
            }
            moe_nax_mma_16x32x16(D[mm][0], D[mm][1], a0, b00, b10);
            moe_nax_mma_16x32x16(D[mm][0], D[mm][1], a1, b01, b11);
          }
        }
      }
    }
  }

  if (sg_active) {
    const uint col_base = tg_n0 + sg_n * 16u;
    const uint rb_8ceil = (valid_rows + 7u) & ~7u;
    if (col_base + fn < p.n) {
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < kTM; ++mm) {
        if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
          #pragma clang loop unroll(full)
          for (uint h = 0u; h < 2u; ++h) {
            const uint rm = m_off + mm * 16u + fm + h * 8u;
            if (rm < valid_rows) {
              bfloat4 out_v;
              #pragma clang loop unroll(full)
              for (uint j = 0u; j < 4u; ++j) {
                const uint e = h * 4u + j;
                const float gv = float(bfloat(D[mm][0][e]));
                const float uv = float(bfloat(D[mm][1][e]));
                out_v[j] = bfloat(nax_silu_f32(gv) * uv);
              }
              *reinterpret_cast<device bfloat4 *>(
                  mid_bf16 + ulong(rb + rm) * p.n + col_base + fn) = out_v;
            } else if (rm < rb_8ceil) {
              *reinterpret_cast<device bfloat4 *>(
                  mid_bf16 + ulong(rb + rm) * p.n + col_base + fn) = bfloat4(bfloat(0.0f));
            }
          }
        }
      }
    }
  }
}

template <uint WBM>
inline void moe_mxfp4_nax_down_scatter_wide_sk1_impl(
    device const bfloat *mid_bf16,
    device const uint *tile_expert,
    device const uint *tile_valid_rows,
    device const uint *tile_assign,
    device const uint *total_tiles,
    device const bfloat *route_scores,
    device const uchar *d_codes,
    device const uchar *d_scales,
    device bfloat *expert_output,
    constant MoeNaxGemmParams &p,
    threadgroup bfloat *Ws,
    uint tid,
    uint lane,
    uint sg,
    uint3 tg) {
  static_assert(WBM == 32u || WBM == 64u, "WBM must be 32 or 64");
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint kOutColsPerTg = 64u;
  constexpr uint kSM = WBM / 2u;
  constexpr uint kTM = kSM / 16u;

  const uint n_cols = (p.n + kOutColsPerTg - 1u) / kOutColsPerTg;
  const bool use_gx32 = (p.max_tiles >> 31u) != 0u;
  const uint m_tile = use_gx32 ? ((tg.y / n_cols) * 32u + tg.x) : tg.y;
  const uint col_tile = use_gx32 ? (tg.y % n_cols) : tg.x;
  const uint tg_n0 = col_tile * kOutColsPerTg;
  if (m_tile >= total_tiles[1] || col_tile >= n_cols || tg_n0 >= p.n) {
    return;
  }
  const uint valid_rows = tile_valid_rows[m_tile];
  if (valid_rows == 0u) {
    return;
  }
  const uint packed = tile_expert[m_tile];
  const uint expert = packed & 0xFFFFu;
  const uint rb = (packed >> 16u) * 8u;

  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const bool compact32 = (WBM == 64u) && (valid_rows <= 32u);
  const uint m_off = sg_m * (compact32 ? 16u : kSM);
  const bool sg_active = (m_off < valid_rows);

  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

  ulong a_off[kTM][2];
  bool a_ok[kTM][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < kTM; ++mm) {
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint rm = m_off + mm * 16u + fm + h * 8u;
      a_ok[mm][h] = (mm == 0u || !compact32) && (rm < valid_rows);
      a_off[mm][h] = a_ok[mm][h] ? (ulong(rb + rm) * p.k + fn) : 0ul;
    }
  }

  vec<float, 8> D[kTM][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < kTM; ++mm) {
    D[mm][0] = vec<float, 8>(0.0f);
    D[mm][1] = vec<float, 8>(0.0f);
  }

  const uint w_tile_row = tid >> 1u;
  const uint w_col_sub = (tid & 1u) * 32u;
  const uint out_col = tg_n0 + w_tile_row;
  const bool w_row_live = (out_col < p.n);
  const ulong expert_row = ulong(expert) * p.n + out_col;
  threadgroup vec<bfloat, 8> *w_dst8 =
      reinterpret_cast<threadgroup vec<bfloat, 8> *>(Ws + w_tile_row * kBKP + w_col_sub);

  if (!moe_has_apple10_nax_layout()) {
    constexpr auto desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> gemm_op;
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c0 = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, float>();
    auto ct_c1 = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      ct_c0[i] = 0.0f;
      ct_c1[i] = 0.0f;
    }
    for (uint k0 = 0u; k0 < p.k; k0 += kBK) {
      uint4 raw;
      float sc;
      moe_nax_fetch_mxfp4_32(
          d_codes, d_scales, expert_row, p.padded_k, k0 + w_col_sub, w_row_live, raw, sc);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      moe_nax_unpack_mxfp4_32(raw, sc, w_dst8);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg_active) {
        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < kBK; kk += 16u) {
          const threadgroup bfloat *ws_sub = Ws + (sg_n * 32u) * kBKP + kk;
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            const auto ib = ct_b.get_multidimensional_index(i);
            ct_b[i] = ws_sub[uint(ib[1]) * kBKP + uint(ib[0])];
          }
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 8u; ++i) {
            const auto ia = ct_a.get_multidimensional_index(i);
            const uint rm = m_off + uint(ia[1]);
            ct_a[i] = (rm < valid_rows)
                ? mid_bf16[ulong(rb + rm) * p.k + k0 + kk + uint(ia[0])]
                : bfloat(0.0f);
          }
          gemm_op.run(ct_a, ct_b, ct_c0);
          if (kTM > 1u && !compact32 && m_off + 16u < valid_rows) {
            #pragma clang loop unroll(full)
            for (ushort i = 0u; i < 8u; ++i) {
              const auto ia = ct_a.get_multidimensional_index(i);
              const uint rm = m_off + 16u + uint(ia[1]);
              ct_a[i] = (rm < valid_rows)
                  ? mid_bf16[ulong(rb + rm) * p.k + k0 + kk + uint(ia[0])]
                  : bfloat(0.0f);
            }
            gemm_op.run(ct_a, ct_b, ct_c1);
          }
        }
      }
    }
    if (sg_active) {
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < kTM; ++mm) {
        if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            if (ct_c0.is_valid_element(i)) {
              const auto idx = ct_c0.get_multidimensional_index(i);
              const uint col = tg_n0 + sg_n * 32u + uint(idx[0]);
              const uint rm = m_off + mm * 16u + uint(idx[1]);
              if (rm < valid_rows && col < p.n) {
                const uint assign = tile_assign[rb + rm];
                const float score = float(route_scores[assign]);
                const float val = (mm == 0u) ? ct_c0[i] : ct_c1[i];
                expert_output[ulong(assign) * p.n + col] = bfloat(val * score);
              }
            }
          }
        }
      }
    }
    return;
  }

  const threadgroup bfloat *ws_sg = Ws + (sg_n * 32u + fm) * kBKP + fn;

  for (uint k0 = 0u; k0 < p.k; k0 += kBK) {
    uint4 raw;
    float sc;
    moe_nax_fetch_mxfp4_32(
        d_codes, d_scales, expert_row, p.padded_k, k0 + w_col_sub, w_row_live, raw, sc);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    moe_nax_unpack_mxfp4_32(raw, sc, w_dst8);
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
        for (uint mm = 0u; mm < kTM; ++mm) {
          if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
            vec<bfloat, 8> a0, a1;
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 2u; ++r) {
              const bfloat4 va0 = a_ok[mm][r]
                  ? *reinterpret_cast<device const bfloat4 *>(mid_bf16 + a_off[mm][r] + k0 + kk + 0u)
                  : bfloat4(bfloat(0.0f));
              const bfloat4 va1 = a_ok[mm][r]
                  ? *reinterpret_cast<device const bfloat4 *>(mid_bf16 + a_off[mm][r] + k0 + kk + 16u)
                  : bfloat4(bfloat(0.0f));
              #pragma clang loop unroll(full)
              for (uint c = 0u; c < 4u; ++c) {
                a0[r * 4u + c] = va0[c];
                a1[r * 4u + c] = va1[c];
              }
            }
            moe_nax_mma_16x32x16(D[mm][0], D[mm][1], a0, b00, b10);
            moe_nax_mma_16x32x16(D[mm][0], D[mm][1], a1, b01, b11);
          }
        }
      }
    }
  }

  if (sg_active) {
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < kTM; ++mm) {
      if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint rm = m_off + mm * 16u + fm + h * 8u;
          if (rm < valid_rows) {
            const uint assign = tile_assign[rb + rm];
            const float score = float(route_scores[assign]);
            #pragma clang loop unroll(full)
            for (uint nn = 0u; nn < 2u; ++nn) {
              const uint col = tg_n0 + sg_n * 32u + nn * 16u + fn;
              if (col < p.n) {
                const uint e = h * 4u;
                const bfloat4 out_v = bfloat4(
                    bfloat(D[mm][nn][e + 0u] * score),
                    bfloat(D[mm][nn][e + 1u] * score),
                    bfloat(D[mm][nn][e + 2u] * score),
                    bfloat(D[mm][nn][e + 3u] * score));
                *reinterpret_cast<device bfloat4 *>(
                    expert_output + ulong(assign) * p.n + col) = out_v;
              }
            }
          }
        }
      }
    }
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_moe_mxfp4_nax_swiglu_bm32_bn64_sk1(
    device const bfloat *norm [[buffer(0)]],
    device const uint *tile_assign [[buffer(1)]],
    device const uint *tile_expert [[buffer(2)]],
    device const uint *tile_valid_rows [[buffer(3)]],
    device const uint *total_tiles [[buffer(4)]],
    device const uchar *g_codes [[buffer(5)]],
    device const uchar *g_scales [[buffer(6)]],
    device const uchar *u_codes [[buffer(7)]],
    device const uchar *u_scales [[buffer(8)]],
    device bfloat *mid_bf16 [[buffer(9)]],
    constant MoeNaxGemmParams &p [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup bfloat Ws[64u * 72u];
  threadgroup uint tg_tok[32u];
  moe_mxfp4_nax_swiglu_wide_sk1_impl<32u>(
      norm, tile_assign, tile_expert, tile_valid_rows, total_tiles,
      g_codes, g_scales, u_codes, u_scales, mid_bf16, p, Ws, tg_tok,
      tid, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_moe_mxfp4_nax_down_scatter_bm32_bn64_sk1(
    device const bfloat *mid_bf16 [[buffer(0)]],
    device const uint *tile_expert [[buffer(1)]],
    device const uint *tile_valid_rows [[buffer(2)]],
    device const uint *tile_assign [[buffer(3)]],
    device const uint *total_tiles [[buffer(4)]],
    device const bfloat *route_scores [[buffer(5)]],
    device const uchar *d_codes [[buffer(6)]],
    device const uchar *d_scales [[buffer(7)]],
    device bfloat *expert_output [[buffer(8)]],
    constant MoeNaxGemmParams &p [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup bfloat Ws[64u * 72u];
  moe_mxfp4_nax_down_scatter_wide_sk1_impl<32u>(
      mid_bf16, tile_expert, tile_valid_rows, tile_assign, total_tiles,
      route_scores, d_codes, d_scales, expert_output, p, Ws,
      tid, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_moe_mxfp4_nax_swiglu_bm64_bn64_sk1(
    device const bfloat *norm [[buffer(0)]],
    device const uint *tile_assign [[buffer(1)]],
    device const uint *tile_expert [[buffer(2)]],
    device const uint *tile_valid_rows [[buffer(3)]],
    device const uint *total_tiles [[buffer(4)]],
    device const uchar *g_codes [[buffer(5)]],
    device const uchar *g_scales [[buffer(6)]],
    device const uchar *u_codes [[buffer(7)]],
    device const uchar *u_scales [[buffer(8)]],
    device bfloat *mid_bf16 [[buffer(9)]],
    constant MoeNaxGemmParams &p [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup bfloat Ws[64u * 72u];
  threadgroup uint tg_tok[64u];
  moe_mxfp4_nax_swiglu_wide_sk1_impl<64u>(
      norm, tile_assign, tile_expert, tile_valid_rows, total_tiles,
      g_codes, g_scales, u_codes, u_scales, mid_bf16, p, Ws, tg_tok,
      tid, lane, sg, tg);
}

// WP3.1: Compact Ws-Staged 2-Pass (BM=64, BN_out=64, BK=64) Direct-L1-X MoE SwiGLU
// 4 SGs = 128 threads (WM=2, WN=2), SRAM = 9,216 B Ws + 256 B tg_tok = 9,472 B (9.25 KiB <= 10.0 KiB!), 0 spills!
struct MoeOut64TgStorage {
  alignas(16) bfloat Ws[64u * 72u];
  uint tg_tok[64u];
};
static_assert(sizeof(MoeOut64TgStorage) == 9472ul &&
                  sizeof(MoeOut64TgStorage) <= kApple10HighOccTgSramBytes,
              "MoeOut64TgStorage must stay <= 10.0 KiB");

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_moe_mxfp4_nax_swiglu_bm64_out64_sk1(
    device const bfloat *norm [[buffer(0)]],
    device const uint *tile_assign [[buffer(1)]],
    device const uint *tile_expert [[buffer(2)]],
    device const uint *tile_valid_rows [[buffer(3)]],
    device const uint *total_tiles [[buffer(4)]],
    device const uchar *g_codes [[buffer(5)]],
    device const uchar *g_scales [[buffer(6)]],
    device const uchar *u_codes [[buffer(7)]],
    device const uchar *u_scales [[buffer(8)]],
    device bfloat *mid_bf16 [[buffer(9)]],
    constant MoeNaxGemmParams &p [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint kOutColsPerTg = 32u;

  threadgroup MoeOut64TgStorage tg_mem;

  const uint n_cols = (p.n + kOutColsPerTg - 1u) / kOutColsPerTg;
  const bool use_gx32 = (p.max_tiles >> 31u) != 0u;
  const uint m_tile = use_gx32 ? ((tg.y / n_cols) * 32u + tg.x) : tg.y;
  const uint col_tile = use_gx32 ? (tg.y % n_cols) : tg.x;
  const uint tg_n0 = col_tile * kOutColsPerTg;
  if (m_tile >= total_tiles[1] || col_tile >= n_cols || tg_n0 >= p.n) {
    return;
  }
  const uint valid_rows = tile_valid_rows[m_tile];
  if (valid_rows == 0u) {
    return;
  }
  const uint packed = tile_expert[m_tile];
  const uint expert = packed & 0xFFFFu;
  const uint rb = (packed >> 16u) * 8u;

  if (tid < 64u) {
    uint tok = 0xFFFFFFFFu;
    if (tid < valid_rows) {
      const uint assign = tile_assign[rb + tid];
      if (assign != 0xFFFFFFFFu) {
        tok = assign >> 3u;
      }
    }
    tg_mem.tg_tok[tid] = tok;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const bool compact32 = (valid_rows <= 32u);
  const uint m_off = sg_m * (compact32 ? 16u : 32u);
  const uint col_base = tg_n0 + sg_n * 16u;
  const bool sg_active = (m_off < valid_rows) && (col_base < p.n);

  const uint w_tile_row = tid >> 1u;
  const uint w_col_sub = (tid & 1u) * 32u;
  const bool is_up = ((w_tile_row >> 4u) & 1u) != 0u;
  const uint out_col = tg_n0 + ((w_tile_row >> 5u) * 16u) + (w_tile_row & 15u);
  const bool w_live = (out_col < p.n);
  device const uchar *w_codes = is_up ? u_codes : g_codes;
  device const uchar *w_scales = is_up ? u_scales : g_scales;
  threadgroup vec<bfloat, 8> *w_dst8 =
      reinterpret_cast<threadgroup vec<bfloat, 8> *>(tg_mem.Ws + w_tile_row * kBKP + w_col_sub);
  const ulong expert_row = ulong(expert) * p.n + out_col;
  const ulong c_stride = ulong(p.padded_k >> 1u);
  const ulong s_stride = ulong(p.padded_k >> 5u);
  const uint k_blocks = p.k / kBK;
  const uint rb_8ceil = (valid_rows + 7u) & ~7u;

  if (!moe_has_apple10_nax_layout()) {
    constexpr auto desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> gemm_op;
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c0 = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, float>();
    auto ct_c1 = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) { ct_c0[i] = 0.0f; ct_c1[i] = 0.0f; }

    for (uint kb = 0u; kb < k_blocks; ++kb) {
      const uint k0 = kb * kBK;
      const uint sub32 = (k0 + w_col_sub) >> 5u;
      const uint4 raw = w_live
          ? *reinterpret_cast<device const uint4 *>(w_codes + expert_row * c_stride + ulong(sub32) * 16ul)
          : uint4(0u);
      const float sc = w_live ? e8m0_to_float(w_scales[expert_row * s_stride + ulong(sub32)]) : 0.0f;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      moe_nax_unpack_mxfp4_32(raw, sc, w_dst8);
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
            const uint tok = tg_mem.tg_tok[m_off + uint(ia[1])];
            ct_a[i] = (tok != 0xFFFFFFFFu) ? norm[ulong(tok) * p.k + k0 + kk + uint(ia[0])] : bfloat(0.0f);
          }
          gemm_op.run(ct_a, ct_b, ct_c0);
          if (!compact32 && m_off + 16u < valid_rows) {
            #pragma clang loop unroll(full)
            for (ushort i = 0u; i < 8u; ++i) {
              const auto ia = ct_a.get_multidimensional_index(i);
              const uint tok = tg_mem.tg_tok[m_off + 16u + uint(ia[1])];
              ct_a[i] = (tok != 0xFFFFFFFFu) ? norm[ulong(tok) * p.k + k0 + kk + uint(ia[0])] : bfloat(0.0f);
            }
            gemm_op.run(ct_a, ct_b, ct_c1);
          }
        }
      }
    }

    if (sg_active) {
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            const auto idx_g = ct_c0.get_multidimensional_index(i);
            const uint c_g = uint(idx_g[0]);
            const uint r_g = uint(idx_g[1]);
            if (c_g < 16u) {
              const uint col = col_base + c_g;
              const uint rm = m_off + mm * 16u + r_g;
              if (col < p.n && rm < valid_rows) {
                float gv = (mm == 0u) ? ct_c0[i] : ct_c1[i];
                float uv = 0.0f;
                #pragma clang loop unroll(full)
                for (ushort j = 0u; j < 16u; ++j) {
                  const auto idx_u = ct_c0.get_multidimensional_index(j);
                  if (uint(idx_u[0]) == c_g + 16u && uint(idx_u[1]) == r_g) {
                    uv = (mm == 0u) ? ct_c0[j] : ct_c1[j];
                  }
                }
                mid_bf16[ulong(rb + rm) * p.n + col] =
                    bfloat(nax_silu_f32(float(bfloat(gv))) * float(bfloat(uv)));
              } else if (col < p.n && rm < rb_8ceil) {
                mid_bf16[ulong(rb + rm) * p.n + col] = bfloat(0.0f);
              }
            }
          }
        }
      }
    }
    return;
  }

  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
  const threadgroup bfloat *ws_sg = tg_mem.Ws + (sg_n * 32u + fm) * kBKP + fn;

  ulong a_off[2][2];
  bool a_ok[2][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < 2u; ++mm) {
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint rm = m_off + mm * 16u + fm + h * 8u;
      const uint tok = (mm == 0u || !compact32) ? tg_mem.tg_tok[rm] : 0xFFFFFFFFu;
      a_ok[mm][h] = (tok != 0xFFFFFFFFu);
      a_off[mm][h] = a_ok[mm][h] ? (ulong(tok) * p.k + fn) : 0ul;
    }
  }

  vec<float, 8> D00 = vec<float, 8>(0.0f), D01 = vec<float, 8>(0.0f);
  vec<float, 8> D10 = vec<float, 8>(0.0f), D11 = vec<float, 8>(0.0f);

  uint4 raw = w_live
      ? *reinterpret_cast<device const uint4 *>(w_codes + expert_row * c_stride + ulong(w_col_sub >> 1u))
      : uint4(0u);
  float sc = w_live
      ? e8m0_to_float(w_scales[expert_row * s_stride + ulong(w_col_sub >> 5u)])
      : 0.0f;

  for (uint kb = 0u; kb < k_blocks; ++kb) {
    const uint k0 = kb * kBK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    moe_nax_unpack_mxfp4_32(raw, sc, w_dst8);
    if (kb + 1u < k_blocks) {
      const uint nsub = ((kb + 1u) * kBK + w_col_sub) >> 5u;
      raw = w_live
          ? *reinterpret_cast<device const uint4 *>(w_codes + expert_row * c_stride + ulong(nsub) * 16ul)
          : uint4(0u);
      sc = w_live
          ? e8m0_to_float(w_scales[expert_row * s_stride + ulong(nsub)])
          : 0.0f;
    }
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
            const bfloat4 va0 = a_ok[0][r]
                ? *reinterpret_cast<device const bfloat4 *>(norm + a_off[0][r] + k0 + kk + 0u)
                : bfloat4(bfloat(0.0f));
            const bfloat4 va1 = a_ok[0][r]
                ? *reinterpret_cast<device const bfloat4 *>(norm + a_off[0][r] + k0 + kk + 16u)
                : bfloat4(bfloat(0.0f));
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) { a0[r * 4u + c] = va0[c]; a1[r * 4u + c] = va1[c]; }
          }
          moe_nax_mma_16x32x16(D00, D01, a0, b00, b10);
          moe_nax_mma_16x32x16(D00, D01, a1, b01, b11);
        }
        if (!compact32 && (m_off + 16u < valid_rows)) {
          vec<bfloat, 8> a0, a1;
          #pragma clang loop unroll(full)
          for (uint r = 0u; r < 2u; ++r) {
            const bfloat4 va0 = a_ok[1][r]
                ? *reinterpret_cast<device const bfloat4 *>(norm + a_off[1][r] + k0 + kk + 0u)
                : bfloat4(bfloat(0.0f));
            const bfloat4 va1 = a_ok[1][r]
                ? *reinterpret_cast<device const bfloat4 *>(norm + a_off[1][r] + k0 + kk + 16u)
                : bfloat4(bfloat(0.0f));
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) { a0[r * 4u + c] = va0[c]; a1[r * 4u + c] = va1[c]; }
          }
          moe_nax_mma_16x32x16(D10, D11, a0, b00, b10);
          moe_nax_mma_16x32x16(D10, D11, a1, b01, b11);
        }
      }
    }
  }

  if (sg_active && col_base + fn < p.n) {
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint e = h * 4u;
      const uint rm0 = m_off + fm + h * 8u;
      if (rm0 < valid_rows) {
        bfloat4 ov;
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j)
          ov[j] = bfloat(nax_silu_f32(float(bfloat(D00[e + j]))) * float(bfloat(D01[e + j])));
        *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb + rm0) * p.n + col_base + fn) = ov;
      } else if (rm0 < rb_8ceil) {
        *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb + rm0) * p.n + col_base + fn) = bfloat4(bfloat(0.0f));
      }
      if (!compact32 && (m_off + 16u < valid_rows)) {
        const uint rm1 = m_off + 16u + fm + h * 8u;
        if (rm1 < valid_rows) {
          bfloat4 ov;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j)
            ov[j] = bfloat(nax_silu_f32(float(bfloat(D10[e + j]))) * float(bfloat(D11[e + j])));
          *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb + rm1) * p.n + col_base + fn) = ov;
        } else if (rm1 < rb_8ceil) {
          *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb + rm1) * p.n + col_base + fn) = bfloat4(bfloat(0.0f));
        }
      }
    }
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_moe_mxfp4_nax_down_scatter_bm64_bn64_sk1(
    device const bfloat *mid_bf16 [[buffer(0)]],
    device const uint *tile_expert [[buffer(1)]],
    device const uint *tile_valid_rows [[buffer(2)]],
    device const uint *tile_assign [[buffer(3)]],
    device const uint *total_tiles [[buffer(4)]],
    device const bfloat *route_scores [[buffer(5)]],
    device const uchar *d_codes [[buffer(6)]],
    device const uchar *d_scales [[buffer(7)]],
    device bfloat *expert_output [[buffer(8)]],
    constant MoeNaxGemmParams &p [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup bfloat Ws[64u * 72u];
  moe_mxfp4_nax_down_scatter_wide_sk1_impl<64u>(
      mid_bf16, tile_expert, tile_valid_rows, tile_assign, total_tiles,
      route_scores, d_codes, d_scales, expert_output, p, Ws,
      tid, lane, sg, tg);
}

// ============================================================================
// Native Zero-Repack oq_affine (q3/q4/q5/q6/q8 x g64/g128) MoE NAX Kernels
// ============================================================================

// Gather BF16 norm rows into tiled BF16 [max_tiles * 8, h] and compute per-64 xsum.
// Grid: width = (h / 64) * 32, height = max_tiles * 8, threads = 32.
kernel void omlx_m5_moe_nax_gather_bf16_xsum_g64(
    device const bfloat *norm [[buffer(0)]],
    device const uint *tile_assign [[buffer(1)]],
    device const uint *total_tiles [[buffer(2)]],
    device bfloat *x_bf16 [[buffer(3)]],
    device float *xsum_out [[buffer(4)]],
    constant MoeNaxBucketParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g = tg.x;
  const uint out_row = tg.y;
  const uint groups = p.h / 64u;
  const uint active_rows = total_tiles[0] * 8u;
  if (g >= groups || out_row >= active_rows || out_row >= p.max_tiles * 8u) {
    return;
  }
  const uint assign = tile_assign[out_row];
  device bfloat2 *dst_v =
      reinterpret_cast<device bfloat2 *>(x_bf16 + ulong(out_row) * p.h + ulong(g) * 64u);
  if (assign == 0xFFFFFFFFu) {
    dst_v[lane] = bfloat2(bfloat(0.0f));
    if (lane == 0u) {
      xsum_out[ulong(out_row) * groups + g] = 0.0f;
    }
    return;
  }
  const uint token = assign / p.top_k;
  device const bfloat2 *src_v =
      reinterpret_cast<device const bfloat2 *>(norm + ulong(token) * p.h + ulong(g) * 64u);
  const bfloat2 pair = src_v[lane];
  dst_v[lane] = pair;
  const float2 fpair = float2(pair);
  const float sum = simd_sum(fpair.x + fpair.y);
  if (lane == 0u) {
    xsum_out[ulong(out_row) * groups + g] = sum;
  }
}

// Compute per-64 xsum over mid_bf16 [active_rows, f] before affine down_scatter.
// Grid: width = (f / 64) * 32, height = max_tiles * 8, threads = 32.
kernel void omlx_m5_moe_nax_mid_xsum_g64(
    device const bfloat *mid_bf16 [[buffer(0)]],
    device const uint *total_tiles [[buffer(1)]],
    device float *xsum_out [[buffer(2)]],
    constant MoeNaxBucketParams &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g = tg.x;
  const uint row = tg.y;
  const uint groups = p.h / 64u;
  const uint active_rows = total_tiles[0] * 8u;
  if (g >= groups || row >= active_rows || row >= p.max_tiles * 8u) {
    return;
  }
  device const bfloat2 *src_v =
      reinterpret_cast<device const bfloat2 *>(mid_bf16 + ulong(row) * p.h + ulong(g) * 64u);
  const float2 fpair = float2(src_v[lane]);
  const float sum = simd_sum(fpair.x + fpair.y);
  if (lane == 0u) {
    xsum_out[ulong(row) * groups + g] = sum;
  }
}

// WP1 Part B: In-place FWHT-64 over MoE BF16 intermediate buffer [active_rows, f].
// Grid: width = (f / 64) * 32, height = max_tiles * 8, threads = 32, 0 B SRAM.
[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_moe_nax_mid_fwht64_bf16(
    device bfloat *mid_bf16 [[buffer(0)]],
    device const uint *total_tiles [[buffer(1)]],
    constant MoeNaxBucketParams &p [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g = tg.x;
  const uint row = tg.y;
  const uint groups = p.h / 64u;
  const uint active_rows = total_tiles[0] * 8u;
  if (g >= groups || row >= active_rows || row >= p.max_tiles * 8u) {
    return;
  }
  device bfloat2 *row_v =
      reinterpret_cast<device bfloat2 *>(mid_bf16 + ulong(row) * p.h + ulong(g) * 64u);
  const float2 fpair = float2(row_v[lane]);
  float v0 = fpair.x;
  float v1 = fpair.y;
  omlx_m5_fwht64_sg32(v0, v1, lane);
  row_v[lane] = bfloat2(bfloat(v0), bfloat(v1));
}

// WP1 Part B: In-place FWHT-64 over MoE FP16 intermediate buffer [active_rows, f] (narrow MXFP4 path).
// Grid: width = (f / 64) * 32, height = max_tiles * 8, threads = 32, 0 B SRAM.
[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_moe_nax_mid_fwht64_f16(
    device half *mid_f16 [[buffer(0)]],
    device const uint *total_tiles [[buffer(1)]],
    constant MoeNaxBucketParams &p [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g = tg.x;
  const uint row = tg.y;
  const uint groups = p.h / 64u;
  const uint active_rows = total_tiles[0] * 8u;
  if (g >= groups || row >= active_rows || row >= p.max_tiles * 8u) {
    return;
  }
  device half2 *row_v =
      reinterpret_cast<device half2 *>(mid_f16 + ulong(row) * p.h + ulong(g) * 64u);
  const float2 fpair = float2(row_v[lane]);
  float v0 = fpair.x;
  float v1 = fpair.y;
  omlx_m5_fwht64_sg32(v0, v1, lane);
  row_v[lane] = half2(half(v0), half(v1));
}

template <uint Bits, typename OpT, typename SliceAT>
inline void moe_affine_nax_dual_step_bn16(
    thread OpT &op,
    thread SliceAT &a,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const uchar *u_codes,
    device const uchar *u_high,
    threadgroup uchar *my_tg_gh2,
    threadgroup uchar *my_tg_uh2,
    uint expert,
    uint n0,
    uint n_max,
    uint padded_k,
    uint g,
    uint lane,
    thread float (&dg_out)[8],
    thread float (&du_out)[8]) {
  constexpr uint BN = 16u;
  constexpr uint GS = 64u;
  if constexpr (Bits == 3u || Bits == 5u) {
    const ulong h_row_bytes = padded_k / 8u;
    const ulong exp_h_base = (ulong(expert) * n_max + ulong(n0)) * h_row_bytes + ulong(g) * 8ul;
    for (uint j = lane; j < BN * 8u; j += 32u) {
      const uint rn = j >> 3u;
      const uint bi = j & 7u;
      const uint gb = (n0 + rn < n_max) ? uint(g_high[exp_h_base + ulong(rn) * h_row_bytes + bi]) : 0u;
      const uint ub = (n0 + rn < n_max) ? uint(u_high[exp_h_base + ulong(rn) * h_row_bytes + bi]) : 0u;
      my_tg_gh2[rn * 16u + bi * 2u] = expand_4x1bit_to_4x2bit(gb & 0x0fu);
      my_tg_gh2[rn * 16u + bi * 2u + 1u] = expand_4x1bit_to_4x2bit(gb >> 4u);
      my_tg_uh2[rn * 16u + bi * 2u] = expand_4x1bit_to_4x2bit(ub & 0x0fu);
      my_tg_uh2[rn * 16u + bi * 2u + 1u] = expand_4x1bit_to_4x2bit(ub >> 4u);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  if constexpr (Bits == 4u) {
    const ulong exp_c_off = ulong(expert) * n_max * (padded_k / 2u);
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bg_full(
        (device uchar *)(g_codes + exp_c_off), dextents<int32_t, 2>(padded_k, n_max));
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bu_full(
        (device uchar *)(u_codes + exp_c_off), dextents<int32_t, 2>(padded_k, n_max));
    auto bg = bg_full.slice(g * GS, n0);
    auto bu = bu_full.slice(g * GS, n0);
    auto pg = op.template get_destination_cooperative_tensor<decltype(a), decltype(bg), float>();
    auto pu = op.template get_destination_cooperative_tensor<decltype(a), decltype(bu), float>();
    op.run(a, bg, pg);
    op.run(a, bu, pu);
    const uint elems = pg.get_capacity();
    for (uint i = 0u; i < elems; ++i) {
      dg_out[i] = pg[i];
      du_out[i] = pu[i];
    }
  } else if constexpr (Bits == 6u) {
    const ulong exp_c_off = ulong(expert) * n_max * (padded_k / 2u);
    const ulong exp_h_off = ulong(expert) * n_max * (padded_k / 4u);
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bg4_full(
        (device uchar *)(g_codes + exp_c_off), dextents<int32_t, 2>(padded_k, n_max));
    tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> bg2_full(
        (device uchar *)(g_high + exp_h_off), dextents<int32_t, 2>(padded_k, n_max));
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bu4_full(
        (device uchar *)(u_codes + exp_c_off), dextents<int32_t, 2>(padded_k, n_max));
    tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> bu2_full(
        (device uchar *)(u_high + exp_h_off), dextents<int32_t, 2>(padded_k, n_max));
    auto bg4 = bg4_full.slice(g * GS, n0);
    auto bg2 = bg2_full.slice(g * GS, n0);
    auto bu4 = bu4_full.slice(g * GS, n0);
    auto bu2 = bu2_full.slice(g * GS, n0);
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
      dg_out[i] = fma(16.0f, pg2[i], pg4[i]);
      du_out[i] = fma(16.0f, pu2[i], pu4[i]);
    }
  } else if constexpr (Bits == 5u) {
    const ulong exp_c_off = ulong(expert) * n_max * (padded_k / 2u);
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bg4_full(
        (device uchar *)(g_codes + exp_c_off), dextents<int32_t, 2>(padded_k, n_max));
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bu4_full(
        (device uchar *)(u_codes + exp_c_off), dextents<int32_t, 2>(padded_k, n_max));
    tensor<threadgroup uint2b_format, dextents<int32_t, 2>, tensor_inline> bg2(
        my_tg_gh2, dextents<int32_t, 2>(GS, BN));
    tensor<threadgroup uint2b_format, dextents<int32_t, 2>, tensor_inline> bu2(
        my_tg_uh2, dextents<int32_t, 2>(GS, BN));
    auto bg4 = bg4_full.slice(g * GS, n0);
    auto bu4 = bu4_full.slice(g * GS, n0);
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
      dg_out[i] = fma(16.0f, pg2[i], pg4[i]);
      du_out[i] = fma(16.0f, pu2[i], pu4[i]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  } else if constexpr (Bits == 3u) {
    const ulong exp_c_off = ulong(expert) * n_max * (padded_k / 4u);
    tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> bglo_full(
        (device uchar *)(g_codes + exp_c_off), dextents<int32_t, 2>(padded_k, n_max));
    tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> bulo_full(
        (device uchar *)(u_codes + exp_c_off), dextents<int32_t, 2>(padded_k, n_max));
    tensor<threadgroup uint2b_format, dextents<int32_t, 2>, tensor_inline> bghi(
        my_tg_gh2, dextents<int32_t, 2>(GS, BN));
    tensor<threadgroup uint2b_format, dextents<int32_t, 2>, tensor_inline> buhi(
        my_tg_uh2, dextents<int32_t, 2>(GS, BN));
    auto bglo = bglo_full.slice(g * GS, n0);
    auto bulo = bulo_full.slice(g * GS, n0);
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
      dg_out[i] = fma(4.0f, pghi[i], pglo[i]);
      du_out[i] = fma(4.0f, puhi[i], pulo[i]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  } else {
    const ulong exp_c_off = ulong(expert) * n_max * padded_k;
    tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> bg8_full(
        (device uchar *)(g_codes + exp_c_off), dextents<int32_t, 2>(padded_k, n_max));
    tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> bu8_full(
        (device uchar *)(u_codes + exp_c_off), dextents<int32_t, 2>(padded_k, n_max));
    auto bg8 = bg8_full.slice(g * GS, n0);
    auto bu8 = bu8_full.slice(g * GS, n0);
    auto pg8 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bg8), float>();
    auto pu8 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bu8), float>();
    op.run(a, bg8, pg8);
    op.run(a, bu8, pu8);
    const uint elems = pg8.get_capacity();
    for (uint i = 0u; i < elems; ++i) {
      dg_out[i] = pg8[i];
      du_out[i] = pu8[i];
    }
  }
}

template <uint Bits, typename OpT, typename SliceAT>
inline void moe_affine_nax_single_step_bn16(
    thread OpT &op,
    thread SliceAT &a,
    device const uchar *d_codes,
    device const uchar *d_high,
    threadgroup uchar *my_tg_h2,
    uint expert,
    uint n0,
    uint n_max,
    uint padded_k,
    uint g,
    uint lane,
    thread float (&dd_out)[8]) {
  constexpr uint BN = 16u;
  constexpr uint GS = 64u;
  if constexpr (Bits == 3u || Bits == 5u) {
    const ulong h_row_bytes = padded_k / 8u;
    const ulong exp_h_base = (ulong(expert) * n_max + ulong(n0)) * h_row_bytes + ulong(g) * 8ul;
    for (uint j = lane; j < BN * 8u; j += 32u) {
      const uint rn = j >> 3u;
      const uint bi = j & 7u;
      const uint db = (n0 + rn < n_max) ? uint(d_high[exp_h_base + ulong(rn) * h_row_bytes + bi]) : 0u;
      my_tg_h2[rn * 16u + bi * 2u] = expand_4x1bit_to_4x2bit(db & 0x0fu);
      my_tg_h2[rn * 16u + bi * 2u + 1u] = expand_4x1bit_to_4x2bit(db >> 4u);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  if constexpr (Bits == 4u) {
    const ulong exp_c_off = ulong(expert) * n_max * (padded_k / 2u);
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bd_full(
        (device uchar *)(d_codes + exp_c_off), dextents<int32_t, 2>(padded_k, n_max));
    auto bd = bd_full.slice(g * GS, n0);
    auto pd = op.template get_destination_cooperative_tensor<decltype(a), decltype(bd), float>();
    op.run(a, bd, pd);
    const uint elems = pd.get_capacity();
    for (uint i = 0u; i < elems; ++i) {
      dd_out[i] = pd[i];
    }
  } else if constexpr (Bits == 6u) {
    const ulong exp_c_off = ulong(expert) * n_max * (padded_k / 2u);
    const ulong exp_h_off = ulong(expert) * n_max * (padded_k / 4u);
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bd4_full(
        (device uchar *)(d_codes + exp_c_off), dextents<int32_t, 2>(padded_k, n_max));
    tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> bd2_full(
        (device uchar *)(d_high + exp_h_off), dextents<int32_t, 2>(padded_k, n_max));
    auto bd4 = bd4_full.slice(g * GS, n0);
    auto bd2 = bd2_full.slice(g * GS, n0);
    auto pd4 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bd4), float>();
    auto pd2 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bd2), float>();
    op.run(a, bd4, pd4);
    op.run(a, bd2, pd2);
    const uint elems = pd4.get_capacity();
    for (uint i = 0u; i < elems; ++i) {
      dd_out[i] = fma(16.0f, pd2[i], pd4[i]);
    }
  } else if constexpr (Bits == 5u) {
    const ulong exp_c_off = ulong(expert) * n_max * (padded_k / 2u);
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> bd4_full(
        (device uchar *)(d_codes + exp_c_off), dextents<int32_t, 2>(padded_k, n_max));
    tensor<threadgroup uint2b_format, dextents<int32_t, 2>, tensor_inline> bd2(
        my_tg_h2, dextents<int32_t, 2>(GS, BN));
    auto bd4 = bd4_full.slice(g * GS, n0);
    auto pd4 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bd4), float>();
    auto pd2 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bd2), float>();
    op.run(a, bd4, pd4);
    op.run(a, bd2, pd2);
    const uint elems = pd4.get_capacity();
    for (uint i = 0u; i < elems; ++i) {
      dd_out[i] = fma(16.0f, pd2[i], pd4[i]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  } else if constexpr (Bits == 3u) {
    const ulong exp_c_off = ulong(expert) * n_max * (padded_k / 4u);
    tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> bdlo_full(
        (device uchar *)(d_codes + exp_c_off), dextents<int32_t, 2>(padded_k, n_max));
    tensor<threadgroup uint2b_format, dextents<int32_t, 2>, tensor_inline> bdhi(
        my_tg_h2, dextents<int32_t, 2>(GS, BN));
    auto bdlo = bdlo_full.slice(g * GS, n0);
    auto pdlo = op.template get_destination_cooperative_tensor<decltype(a), decltype(bdlo), float>();
    auto pdhi = op.template get_destination_cooperative_tensor<decltype(a), decltype(bdhi), float>();
    op.run(a, bdlo, pdlo);
    op.run(a, bdhi, pdhi);
    const uint elems = pdlo.get_capacity();
    for (uint i = 0u; i < elems; ++i) {
      dd_out[i] = fma(4.0f, pdhi[i], pdlo[i]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  } else {
    const ulong exp_c_off = ulong(expert) * n_max * padded_k;
    tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> bd8_full(
        (device uchar *)(d_codes + exp_c_off), dextents<int32_t, 2>(padded_k, n_max));
    auto bd8 = bd8_full.slice(g * GS, n0);
    auto pd8 = op.template get_destination_cooperative_tensor<decltype(a), decltype(bd8), float>();
    op.run(a, bd8, pd8);
    const uint elems = pd8.get_capacity();
    for (uint i = 0u; i < elems; ++i) {
      dd_out[i] = pd8[i];
    }
  }
}

// Bits: 3, 4, 5, 6, 8; GroupSize: 64 or 128; NSG: 1 or 2
template <uint Bits, uint GroupSize, uint NSG = 1u>
inline void moe_affine_nax_swiglu_n16_sk4_impl(
    device const bfloat *x_bf16,
    device const float *xsum,
    device const uint *tile_expert,
    device const uint *tile_valid_rows,
    device const uint *total_tiles,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const bfloat *g_scales,
    device const bfloat *g_biases,
    device const uchar *u_codes,
    device const uchar *u_high,
    device const bfloat *u_scales,
    device const bfloat *u_biases,
    device bfloat *mid_bf16,
    constant MoeNaxGemmParams &p,
    threadgroup float *parts,
    threadgroup uchar *tg_h2_all,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM = 8u;
  constexpr uint BN = 16u;
  constexpr uint GS = 64u;
  constexpr uint SK = 4u;
  constexpr uint kCap = (16u * BN) / 32u;

  const uint m_tile = tg.y;
  const uint tg_n0 = tg.x * (BN * NSG);
  if (m_tile >= total_tiles[0] || tg_n0 >= p.n) {
    return;
  }
  const uint sg_n = (NSG > 1u) ? (sg / SK) : 0u;
  const uint sg_k = (NSG > 1u) ? (sg % SK) : sg;
  const uint n0 = (tg.x * NSG + sg_n) * BN;
  const bool col_active = (sg_n < NSG) && (n0 < p.n);

  const uint expert = tile_expert[m_tile];
  const uint valid_rows = tile_valid_rows[m_tile];
  const uint rb = m_tile * BM;

  const uint groups = p.k / GS;
  const uint g0 = (sg_k * groups) / SK;
  const uint g1 = ((sg_k + 1u) * groups) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a_full(
      (device bfloat *)(x_bf16 + ulong(rb) * p.k), dextents<int32_t, 2>(p.k, BM));

  auto a0_lay = a_full.slice(0, 0);
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0_lay(
      (device uchar *)g_codes, dextents<int32_t, 2>(GS, BN));
  auto layout = op.template get_destination_cooperative_tensor<decltype(a0_lay), decltype(b0_lay), float>();
  const uint elems = layout.get_capacity();
  uchar col_off[kCap];
  uchar row_off[kCap];
  uchar valid_elem[kCap];
  float acc_g[kCap];
  float acc_u[kCap];
  for (uint i = 0u; i < kCap; ++i) {
    acc_g[i] = 0.0f;
    acc_u[i] = 0.0f;
    if (i < elems && layout.is_valid_element(i)) {
      const auto coord = layout.get_multidimensional_index(i);
      col_off[i] = uchar(coord[0]);
      row_off[i] = uchar(coord[1]);
      valid_elem[i] = (n0 + uint(coord[0]) < p.n) ? 1u : 0u;
    } else {
      col_off[i] = 0u;
      row_off[i] = 0u;
      valid_elem[i] = 0u;
    }
  }

  const ulong sb_row_stride = p.padded_k / GroupSize;
  const ulong exp_sb_off = ulong(expert) * p.n * sb_row_stride;
  threadgroup uchar *my_tg_gh2 =
      ((Bits == 3u || Bits == 5u) && tg_h2_all != nullptr && sg < (SK * NSG))
          ? (tg_h2_all + sg * (2u * BN * 16u))
          : nullptr;
  threadgroup uchar *my_tg_uh2 =
      ((Bits == 3u || Bits == 5u) && my_tg_gh2 != nullptr)
          ? (my_tg_gh2 + BN * 16u)
          : nullptr;

  if (col_active) {
    for (uint g = g0; g < g1; ++g) {
      auto a = a_full.slice(g * GS, 0);
      const uint sb_g = (GroupSize == 128u) ? (g >> 1u) : g;
      float dg[kCap];
      float du[kCap];
      moe_affine_nax_dual_step_bn16<Bits>(
          op, a, g_codes, g_high, u_codes, u_high, my_tg_gh2, my_tg_uh2,
          expert, n0, p.n, p.padded_k, g, lane, dg, du);
      for (uint i = 0u; i < elems; ++i) {
        if (valid_elem[i] != 0u) {
          const uint col = n0 + uint(col_off[i]);
          const uint rm = uint(row_off[i]);
          const ulong sb_idx = exp_sb_off + ulong(col) * sb_row_stride + sb_g;
          const float xs = xsum[ulong(rb + rm) * groups + g];
          acc_g[i] = fma(float(g_scales[sb_idx]), dg[i], fma(float(g_biases[sb_idx]), xs, acc_g[i]));
          acc_u[i] = fma(float(u_scales[sb_idx]), du[i], fma(float(u_biases[sb_idx]), xs, acc_u[i]));
        }
      }
    }
  }

  constexpr uint kPartStride = (SK - 1u) * 2u * kCap * 32u;
  threadgroup float *my_parts = parts + ((NSG > 1u && sg_n < NSG) ? (sg_n * kPartStride) : 0u);
  if (sg_k > 0u && col_active) {
    for (uint i = 0u; i < kCap; ++i) {
      my_parts[((sg_k - 1u) * (2u * kCap) + i) * 32u + lane] = acc_g[i];
      my_parts[((sg_k - 1u) * (2u * kCap) + kCap + i) * 32u + lane] = acc_u[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg_k == 0u && col_active) {
    for (uint s = 1u; s < SK; ++s) {
      for (uint i = 0u; i < kCap; ++i) {
        acc_g[i] += my_parts[((s - 1u) * (2u * kCap) + i) * 32u + lane];
        acc_u[i] += my_parts[((s - 1u) * (2u * kCap) + kCap + i) * 32u + lane];
      }
    }
    for (uint i = 0u; i < elems; ++i) {
      if (valid_elem[i] != 0u) {
        const uint col = n0 + uint(col_off[i]);
        const uint rm = uint(row_off[i]);
        if (rm < valid_rows) {
          const float gv = float(bfloat(acc_g[i]));
          const float uv = float(bfloat(acc_u[i]));
          mid_bf16[ulong(rb + rm) * p.n + col] = bfloat(nax_silu_f32(gv) * uv);
        } else {
          mid_bf16[ulong(rb + rm) * p.n + col] = bfloat(0.0f);
        }
      }
    }
  }
}

template <uint Bits, uint GroupSize, uint NSG = 1u>
inline void moe_affine_nax_down_scatter_n16_sk2_impl(
    device const bfloat *mid_bf16,
    device const float *mid_xsum,
    device const uint *tile_expert,
    device const uint *tile_valid_rows,
    device const uint *tile_assign,
    device const uint *total_tiles,
    device const bfloat *route_scores,
    device const uchar *d_codes,
    device const uchar *d_high,
    device const bfloat *d_scales,
    device const bfloat *d_biases,
    device bfloat *expert_output,
    constant MoeNaxGemmParams &p,
    threadgroup float *parts,
    threadgroup uchar *tg_h2_all,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM = 8u;
  constexpr uint BN = 16u;
  constexpr uint GS = 64u;
  constexpr uint SK = 2u;
  constexpr uint kCap = (16u * BN) / 32u;

  const uint m_tile = tg.y;
  const uint tg_n0 = tg.x * (BN * NSG);
  if (m_tile >= total_tiles[0] || tg_n0 >= p.n) {
    return;
  }
  const uint sg_n = (NSG > 1u) ? (sg / SK) : 0u;
  const uint sg_k = (NSG > 1u) ? (sg % SK) : sg;
  const uint n0 = (tg.x * NSG + sg_n) * BN;
  const bool col_active = (sg_n < NSG) && (n0 < p.n);

  const uint expert = tile_expert[m_tile];
  const uint valid_rows = tile_valid_rows[m_tile];
  const uint rb = m_tile * BM;

  const uint groups = p.k / GS;
  const uint g0 = (sg_k * groups) / SK;
  const uint g1 = ((sg_k + 1u) * groups) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a_full(
      (device bfloat *)(mid_bf16 + ulong(rb) * p.k), dextents<int32_t, 2>(p.k, BM));

  auto a0_lay = a_full.slice(0, 0);
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0_lay(
      (device uchar *)d_codes, dextents<int32_t, 2>(GS, BN));
  auto layout = op.template get_destination_cooperative_tensor<decltype(a0_lay), decltype(b0_lay), float>();
  const uint elems = layout.get_capacity();
  uchar col_off[kCap];
  uchar row_off[kCap];
  uchar valid_elem[kCap];
  float acc[kCap];
  for (uint i = 0u; i < kCap; ++i) {
    acc[i] = 0.0f;
    if (i < elems && layout.is_valid_element(i)) {
      const auto coord = layout.get_multidimensional_index(i);
      col_off[i] = uchar(coord[0]);
      row_off[i] = uchar(coord[1]);
      valid_elem[i] = (n0 + uint(coord[0]) < p.n) ? 1u : 0u;
    } else {
      col_off[i] = 0u;
      row_off[i] = 0u;
      valid_elem[i] = 0u;
    }
  }

  const ulong sb_row_stride = p.padded_k / GroupSize;
  const ulong exp_sb_off = ulong(expert) * p.n * sb_row_stride;
  threadgroup uchar *my_tg_h2 =
      ((Bits == 3u || Bits == 5u) && tg_h2_all != nullptr && sg < (SK * NSG))
          ? (tg_h2_all + sg * (BN * 16u))
          : nullptr;

  if (col_active) {
    for (uint g = g0; g < g1; ++g) {
      auto a = a_full.slice(g * GS, 0);
      const uint sb_g = (GroupSize == 128u) ? (g >> 1u) : g;
      float dd[kCap];
      moe_affine_nax_single_step_bn16<Bits>(
          op, a, d_codes, d_high, my_tg_h2, expert, n0, p.n, p.padded_k, g, lane, dd);
      for (uint i = 0u; i < elems; ++i) {
        if (valid_elem[i] != 0u) {
          const uint col = n0 + uint(col_off[i]);
          const uint rm = uint(row_off[i]);
          const ulong sb_idx = exp_sb_off + ulong(col) * sb_row_stride + sb_g;
          const float xs = mid_xsum[ulong(rb + rm) * groups + g];
          acc[i] = fma(float(d_scales[sb_idx]), dd[i], fma(float(d_biases[sb_idx]), xs, acc[i]));
        }
      }
    }
  }

  constexpr uint kPartStride = (SK - 1u) * kCap * 32u;
  threadgroup float *my_parts = parts + ((NSG > 1u && sg_n < NSG) ? (sg_n * kPartStride) : 0u);
  if (sg_k > 0u && col_active) {
    for (uint i = 0u; i < kCap; ++i) {
      my_parts[((sg_k - 1u) * kCap + i) * 32u + lane] = acc[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg_k == 0u && col_active) {
    for (uint s = 1u; s < SK; ++s) {
      for (uint i = 0u; i < kCap; ++i) {
        acc[i] += my_parts[((s - 1u) * kCap + i) * 32u + lane];
      }
    }
    for (uint i = 0u; i < elems; ++i) {
      if (valid_elem[i] != 0u) {
        const uint col = n0 + uint(col_off[i]);
        const uint rm = uint(row_off[i]);
        if (rm < valid_rows) {
          const uint assign = tile_assign[rb + rm];
          const float score = float(route_scores[assign]);
          expert_output[ulong(assign) * p.n + col] = bfloat(acc[i] * score);
        }
      }
    }
  }
}

// Wide Zero-Gather Affine MoE Prefill: BM=32 / BM=64, BN=64, SK=1 (WM=2, WN=2, 128 threads/TG)
struct MoeAffineRaw32 {
  float sc;
  float bi;
  uint4 r0;
  uint4 r1;
};

template <uint Bits, uint GroupSize>
inline MoeAffineRaw32 moe_nax_fetch_affine_32(
    device const uchar *codes,
    device const uchar *high,
    device const bfloat *scales,
    device const bfloat *biases,
    ulong expert_row,
    uint padded_k,
    uint k_col,
    bool row_live) {
  MoeAffineRaw32 out = {0.0f, 0.0f, uint4(0u), uint4(0u)};
  if (!row_live) {
    return out;
  }
  const ulong sb_idx =
      expert_row * ulong(padded_k / GroupSize) + ulong(k_col / GroupSize);
  out.sc = float(scales[sb_idx]);
  out.bi = float(biases[sb_idx]);
  if constexpr (Bits == 4u) {
    out.r0 = *reinterpret_cast<device const uint4 *>(
        codes + expert_row * ulong(padded_k / 2u) + ulong(k_col / 2u));
  } else if constexpr (Bits == 8u) {
    device const uint4 *p4 = reinterpret_cast<device const uint4 *>(
        codes + expert_row * ulong(padded_k) + ulong(k_col));
    out.r0 = p4[0];
    out.r1 = p4[1];
  } else if constexpr (Bits == 6u) {
    out.r0 = *reinterpret_cast<device const uint4 *>(
        codes + expert_row * ulong(padded_k / 2u) + ulong(k_col / 2u));
    const uint2 hi2 = *reinterpret_cast<device const uint2 *>(
        high + expert_row * ulong(padded_k / 4u) + ulong(k_col / 4u));
    out.r1 = uint4(hi2.x, hi2.y, 0u, 0u);
  } else if constexpr (Bits == 5u) {
    out.r0 = *reinterpret_cast<device const uint4 *>(
        codes + expert_row * ulong(padded_k / 2u) + ulong(k_col / 2u));
    const uint hi1 = *reinterpret_cast<device const uint *>(
        high + expert_row * ulong(padded_k / 8u) + ulong(k_col / 8u));
    out.r1 = uint4(hi1, 0u, 0u, 0u);
  } else {
    const uint2 lo2 = *reinterpret_cast<device const uint2 *>(
        codes + expert_row * ulong(padded_k / 4u) + ulong(k_col / 4u));
    const uint hi1 = *reinterpret_cast<device const uint *>(
        high + expert_row * ulong(padded_k / 8u) + ulong(k_col / 8u));
    out.r0 = uint4(lo2.x, lo2.y, hi1, 0u);
  }
  return out;
}

template <uint Bits>
inline void moe_nax_unpack_affine_32(
    const MoeAffineRaw32 raw,
    threadgroup vec<bfloat, 8> *w_dst8) {
  const float sc = raw.sc;
  const float bi = raw.bi;
  if constexpr (Bits == 4u) {
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      const uint wd = raw.r0[w];
      vec<bfloat, 8> out;
      #pragma clang loop unroll(full)
      for (uint j = 0u; j < 8u; ++j) {
        out[j] = bfloat(fma(sc, float((wd >> (j * 4u)) & 15u), bi));
      }
      w_dst8[w] = out;
    }
  } else if constexpr (Bits == 8u) {
    #pragma clang loop unroll(full)
    for (uint q16 = 0u; q16 < 2u; ++q16) {
      const uint4 pk = (q16 == 0u) ? raw.r0 : raw.r1;
      #pragma clang loop unroll(full)
      for (uint h8 = 0u; h8 < 2u; ++h8) {
        const uint w0 = pk[h8 * 2u + 0u];
        const uint w1 = pk[h8 * 2u + 1u];
        w_dst8[q16 * 2u + h8] = vec<bfloat, 8>(
            bfloat(fma(sc, float((w0 >>  0u) & 255u), bi)),
            bfloat(fma(sc, float((w0 >>  8u) & 255u), bi)),
            bfloat(fma(sc, float((w0 >> 16u) & 255u), bi)),
            bfloat(fma(sc, float((w0 >> 24u) & 255u), bi)),
            bfloat(fma(sc, float((w1 >>  0u) & 255u), bi)),
            bfloat(fma(sc, float((w1 >>  8u) & 255u), bi)),
            bfloat(fma(sc, float((w1 >> 16u) & 255u), bi)),
            bfloat(fma(sc, float((w1 >> 24u) & 255u), bi)));
      }
    }
  } else if constexpr (Bits == 6u) {
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      const uint wd = raw.r0[w];
      const uint hd = (w < 2u) ? (raw.r1.x >> (w * 16u)) : (raw.r1.y >> ((w - 2u) * 16u));
      vec<bfloat, 8> out;
      #pragma clang loop unroll(full)
      for (uint j = 0u; j < 8u; ++j) {
        out[j] = bfloat(fma(sc, float(((wd >> (j * 4u)) & 15u) | (((hd >> (j * 2u)) & 3u) << 4u)), bi));
      }
      w_dst8[w] = out;
    }
  } else if constexpr (Bits == 5u) {
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      const uint wd = raw.r0[w];
      const uint hb = (raw.r1.x >> (w * 8u)) & 255u;
      vec<bfloat, 8> out;
      #pragma clang loop unroll(full)
      for (uint j = 0u; j < 8u; ++j) {
        out[j] = bfloat(fma(sc, float(((wd >> (j * 4u)) & 15u) | (((hb >> j) & 1u) << 4u)), bi));
      }
      w_dst8[w] = out;
    }
  } else {
    #pragma clang loop unroll(full)
    for (uint q16 = 0u; q16 < 2u; ++q16) {
      const uint lword = (q16 == 0u) ? raw.r0.x : raw.r0.y;
      const uint h16 = (raw.r0.z >> (q16 * 16u)) & 0xffffu;
      #pragma clang loop unroll(full)
      for (uint h8 = 0u; h8 < 2u; ++h8) {
        const uint ld = lword >> (h8 * 16u);
        const uint hb = h16 >> (h8 * 8u);
        vec<bfloat, 8> out;
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j) {
          out[j] = bfloat(fma(sc, float(((ld >> (j * 2u)) & 3u) | (((hb >> j) & 1u) << 2u)), bi));
        }
        w_dst8[q16 * 2u + h8] = out;
      }
    }
  }
}

template <uint Bits, uint GroupSize, uint WBM>
inline void moe_affine_nax_swiglu_wide_sk1_impl(
    device const bfloat *norm,
    device const uint *tile_assign,
    device const uint *tile_expert,
    device const uint *tile_valid_rows,
    device const uint *total_tiles,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const bfloat *g_scales,
    device const bfloat *g_biases,
    device const uchar *u_codes,
    device const uchar *u_high,
    device const bfloat *u_scales,
    device const bfloat *u_biases,
    device bfloat *mid_bf16,
    constant MoeNaxGemmParams &p,
    threadgroup bfloat *Ws,
    threadgroup uint *tg_tok,
    uint tid,
    uint lane,
    uint sg,
    uint3 tg) {
  static_assert(WBM == 32u || WBM == 64u, "WBM must be 32 or 64");
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint kOutColsPerTg = 32u;
  constexpr uint kSM = WBM / 2u;
  constexpr uint kTM = kSM / 16u;

  const uint n_cols = (p.n + kOutColsPerTg - 1u) / kOutColsPerTg;
  const bool use_gx32 = (p.max_tiles >> 31u) != 0u;
  const uint m_tile = use_gx32 ? ((tg.y / n_cols) * 32u + tg.x) : tg.y;
  const uint col_tile = use_gx32 ? (tg.y % n_cols) : tg.x;
  const uint tg_n0 = col_tile * kOutColsPerTg;
  if (m_tile >= total_tiles[1] || col_tile >= n_cols || tg_n0 >= p.n) {
    return;
  }
  const uint valid_rows = tile_valid_rows[m_tile];
  if (valid_rows == 0u) {
    return;
  }
  const uint packed = tile_expert[m_tile];
  const uint expert = packed & 0xFFFFu;
  const uint rb = (packed >> 16u) * 8u;

  if (tid < WBM) {
    uint tok = 0xFFFFFFFFu;
    if (tid < valid_rows) {
      const uint assign = tile_assign[rb + tid];
      if (assign != 0xFFFFFFFFu) {
        tok = assign >> 3u;
      }
    }
    tg_tok[tid] = tok;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const bool compact32 = (WBM == 64u) && (valid_rows <= 32u);
  const uint m_off = sg_m * (compact32 ? 16u : kSM);
  const bool sg_active = (m_off < valid_rows);

  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

  ulong a_off[kTM][2];
  bool a_ok[kTM][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < kTM; ++mm) {
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint rm = m_off + mm * 16u + fm + h * 8u;
      const uint tok = ((mm == 0u || !compact32) && rm < valid_rows) ? tg_tok[rm] : 0xFFFFFFFFu;
      a_ok[mm][h] = (tok != 0xFFFFFFFFu);
      a_off[mm][h] = a_ok[mm][h] ? (ulong(tok) * p.k + fn) : 0ul;
    }
  }

  vec<float, 8> D[kTM][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < kTM; ++mm) {
    D[mm][0] = vec<float, 8>(0.0f);
    D[mm][1] = vec<float, 8>(0.0f);
  }

  const uint w_tile_row = tid >> 1u;
  const uint w_col_sub = (tid & 1u) * 32u;
  const bool is_up = ((w_tile_row >> 4u) & 1u) != 0u;
  const uint out_col = tg_n0 + ((w_tile_row >> 5u) * 16u) + (w_tile_row & 15u);
  const bool w_row_live = (out_col < p.n);
  const ulong expert_row = ulong(expert) * p.n + out_col;
  device const uchar *sel_codes = is_up ? u_codes : g_codes;
  device const uchar *sel_high = is_up ? u_high : g_high;
  device const bfloat *sel_scales = is_up ? u_scales : g_scales;
  device const bfloat *sel_biases = is_up ? u_biases : g_biases;
  threadgroup vec<bfloat, 8> *w_dst8 =
      reinterpret_cast<threadgroup vec<bfloat, 8> *>(Ws + w_tile_row * kBKP + w_col_sub);

  if (!moe_has_apple10_nax_layout()) {
    constexpr auto desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> gemm_op;
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c0 = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, float>();
    auto ct_c1 = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      ct_c0[i] = 0.0f;
      ct_c1[i] = 0.0f;
    }
    for (uint k0 = 0u; k0 < p.k; k0 += kBK) {
      const MoeAffineRaw32 raw = moe_nax_fetch_affine_32<Bits, GroupSize>(
          sel_codes, sel_high, sel_scales, sel_biases, expert_row, p.padded_k,
          k0 + w_col_sub, w_row_live);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      moe_nax_unpack_affine_32<Bits>(raw, w_dst8);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg_active) {
        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < kBK; kk += 16u) {
          const threadgroup bfloat *ws_sub = Ws + (sg_n * 32u) * kBKP + kk;
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            const auto ib = ct_b.get_multidimensional_index(i);
            ct_b[i] = ws_sub[uint(ib[1]) * kBKP + uint(ib[0])];
          }
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 8u; ++i) {
            const auto ia = ct_a.get_multidimensional_index(i);
            const uint rm = m_off + uint(ia[1]);
            const uint tok = (rm < valid_rows) ? tg_tok[rm] : 0xFFFFFFFFu;
            ct_a[i] = (tok != 0xFFFFFFFFu)
                ? norm[ulong(tok) * p.k + k0 + kk + uint(ia[0])]
                : bfloat(0.0f);
          }
          gemm_op.run(ct_a, ct_b, ct_c0);
          if (kTM > 1u && !compact32 && m_off + 16u < valid_rows) {
            #pragma clang loop unroll(full)
            for (ushort i = 0u; i < 8u; ++i) {
              const auto ia = ct_a.get_multidimensional_index(i);
              const uint rm = m_off + 16u + uint(ia[1]);
              const uint tok = (rm < valid_rows) ? tg_tok[rm] : 0xFFFFFFFFu;
              ct_a[i] = (tok != 0xFFFFFFFFu)
                  ? norm[ulong(tok) * p.k + k0 + kk + uint(ia[0])]
                  : bfloat(0.0f);
            }
            gemm_op.run(ct_a, ct_b, ct_c1);
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_active) {
      threadgroup bfloat *sg_buf = Ws + sg * 512u;
      const uint col_base = tg_n0 + sg_n * 16u;
      const uint rb_8ceil = (valid_rows + 7u) & ~7u;
      const uint r_out_base = (lane >> 2u) * 2u;
      const uint c_out4 = (lane & 3u) * 4u;
      const uint col = col_base + c_out4;
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < kTM; ++mm) {
        if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            if (ct_c0.is_valid_element(i)) {
              const auto idx = ct_c0.get_multidimensional_index(i);
              sg_buf[uint(idx[1]) * 32u + uint(idx[0])] =
                  bfloat((mm == 0u) ? ct_c0[i] : ct_c1[i]);
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          if (col < p.n) {
            #pragma clang loop unroll(full)
            for (uint dr = 0u; dr < 2u; ++dr) {
              const uint r = r_out_base + dr;
              const uint rm = m_off + mm * 16u + r;
              if (rm < valid_rows) {
                const bfloat4 g4 = *reinterpret_cast<const threadgroup bfloat4 *>(sg_buf + r * 32u + c_out4);
                const bfloat4 u4 = *reinterpret_cast<const threadgroup bfloat4 *>(sg_buf + r * 32u + 16u + c_out4);
                bfloat4 out_b;
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  out_b[j] = bfloat(nax_silu_f32(float(g4[j])) * float(u4[j]));
                }
                *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb + rm) * p.n + col) = out_b;
              } else if (rm < rb_8ceil) {
                *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb + rm) * p.n + col) = bfloat4(bfloat(0.0f));
              }
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    }
    return;
  }

  const threadgroup bfloat *ws_sg = Ws + (sg_n * 32u + fm) * kBKP + fn;

  for (uint k0 = 0u; k0 < p.k; k0 += kBK) {
    const MoeAffineRaw32 raw = moe_nax_fetch_affine_32<Bits, GroupSize>(
        sel_codes, sel_high, sel_scales, sel_biases, expert_row, p.padded_k,
        k0 + w_col_sub, w_row_live);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    moe_nax_unpack_affine_32<Bits>(raw, w_dst8);
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
        for (uint mm = 0u; mm < kTM; ++mm) {
          if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
            vec<bfloat, 8> a0, a1;
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 2u; ++r) {
              const bfloat4 va0 = a_ok[mm][r]
                  ? *reinterpret_cast<device const bfloat4 *>(norm + a_off[mm][r] + k0 + kk + 0u)
                  : bfloat4(bfloat(0.0f));
              const bfloat4 va1 = a_ok[mm][r]
                  ? *reinterpret_cast<device const bfloat4 *>(norm + a_off[mm][r] + k0 + kk + 16u)
                  : bfloat4(bfloat(0.0f));
              #pragma clang loop unroll(full)
              for (uint c = 0u; c < 4u; ++c) {
                a0[r * 4u + c] = va0[c];
                a1[r * 4u + c] = va1[c];
              }
            }
            moe_nax_mma_16x32x16(D[mm][0], D[mm][1], a0, b00, b10);
            moe_nax_mma_16x32x16(D[mm][0], D[mm][1], a1, b01, b11);
          }
        }
      }
    }
  }

  if (sg_active) {
    const uint col_base = tg_n0 + sg_n * 16u;
    const uint rb_8ceil = (valid_rows + 7u) & ~7u;
    if (col_base + fn < p.n) {
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < kTM; ++mm) {
        if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
          #pragma clang loop unroll(full)
          for (uint h = 0u; h < 2u; ++h) {
            const uint rm = m_off + mm * 16u + fm + h * 8u;
            if (rm < valid_rows) {
              bfloat4 out_v;
              #pragma clang loop unroll(full)
              for (uint j = 0u; j < 4u; ++j) {
                const uint e = h * 4u + j;
                const float gv = float(bfloat(D[mm][0][e]));
                const float uv = float(bfloat(D[mm][1][e]));
                out_v[j] = bfloat(nax_silu_f32(gv) * uv);
              }
              *reinterpret_cast<device bfloat4 *>(
                  mid_bf16 + ulong(rb + rm) * p.n + col_base + fn) = out_v;
            } else if (rm < rb_8ceil) {
              *reinterpret_cast<device bfloat4 *>(
                  mid_bf16 + ulong(rb + rm) * p.n + col_base + fn) = bfloat4(bfloat(0.0f));
            }
          }
        }
      }
    }
  }
}

template <uint Bits, uint GroupSize, uint WBM>
inline void moe_affine_nax_down_scatter_wide_sk1_impl(
    device const bfloat *mid_bf16,
    device const uint *tile_expert,
    device const uint *tile_valid_rows,
    device const uint *tile_assign,
    device const uint *total_tiles,
    device const bfloat *route_scores,
    device const uchar *d_codes,
    device const uchar *d_high,
    device const bfloat *d_scales,
    device const bfloat *d_biases,
    device bfloat *expert_output,
    constant MoeNaxGemmParams &p,
    threadgroup bfloat *Ws,
    uint tid,
    uint lane,
    uint sg,
    uint3 tg) {
  static_assert(WBM == 32u || WBM == 64u, "WBM must be 32 or 64");
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint kOutColsPerTg = 64u;
  constexpr uint kSM = WBM / 2u;
  constexpr uint kTM = kSM / 16u;

  const uint n_cols = (p.n + kOutColsPerTg - 1u) / kOutColsPerTg;
  const bool use_gx32 = (p.max_tiles >> 31u) != 0u;
  const uint m_tile = use_gx32 ? ((tg.y / n_cols) * 32u + tg.x) : tg.y;
  const uint col_tile = use_gx32 ? (tg.y % n_cols) : tg.x;
  const uint tg_n0 = col_tile * kOutColsPerTg;
  if (m_tile >= total_tiles[1] || col_tile >= n_cols || tg_n0 >= p.n) {
    return;
  }
  const uint valid_rows = tile_valid_rows[m_tile];
  if (valid_rows == 0u) {
    return;
  }
  const uint packed = tile_expert[m_tile];
  const uint expert = packed & 0xFFFFu;
  const uint rb = (packed >> 16u) * 8u;

  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const bool compact32 = (WBM == 64u) && (valid_rows <= 32u);
  const uint m_off = sg_m * (compact32 ? 16u : kSM);
  const bool sg_active = (m_off < valid_rows);

  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

  ulong a_off[kTM][2];
  bool a_ok[kTM][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < kTM; ++mm) {
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint rm = m_off + mm * 16u + fm + h * 8u;
      a_ok[mm][h] = (mm == 0u || !compact32) && (rm < valid_rows);
      a_off[mm][h] = a_ok[mm][h] ? (ulong(rb + rm) * p.k + fn) : 0ul;
    }
  }

  vec<float, 8> D[kTM][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < kTM; ++mm) {
    D[mm][0] = vec<float, 8>(0.0f);
    D[mm][1] = vec<float, 8>(0.0f);
  }

  const uint w_tile_row = tid >> 1u;
  const uint w_col_sub = (tid & 1u) * 32u;
  const uint out_col = tg_n0 + w_tile_row;
  const bool w_row_live = (out_col < p.n);
  const ulong expert_row = ulong(expert) * p.n + out_col;
  threadgroup vec<bfloat, 8> *w_dst8 =
      reinterpret_cast<threadgroup vec<bfloat, 8> *>(Ws + w_tile_row * kBKP + w_col_sub);

  if (!moe_has_apple10_nax_layout()) {
    constexpr auto desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> gemm_op;
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c0 = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, float>();
    auto ct_c1 = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      ct_c0[i] = 0.0f;
      ct_c1[i] = 0.0f;
    }
    for (uint k0 = 0u; k0 < p.k; k0 += kBK) {
      const MoeAffineRaw32 raw = moe_nax_fetch_affine_32<Bits, GroupSize>(
          d_codes, d_high, d_scales, d_biases, expert_row, p.padded_k,
          k0 + w_col_sub, w_row_live);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      moe_nax_unpack_affine_32<Bits>(raw, w_dst8);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg_active) {
        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < kBK; kk += 16u) {
          const threadgroup bfloat *ws_sub = Ws + (sg_n * 32u) * kBKP + kk;
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            const auto ib = ct_b.get_multidimensional_index(i);
            ct_b[i] = ws_sub[uint(ib[1]) * kBKP + uint(ib[0])];
          }
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 8u; ++i) {
            const auto ia = ct_a.get_multidimensional_index(i);
            const uint rm = m_off + uint(ia[1]);
            ct_a[i] = (rm < valid_rows)
                ? mid_bf16[ulong(rb + rm) * p.k + k0 + kk + uint(ia[0])]
                : bfloat(0.0f);
          }
          gemm_op.run(ct_a, ct_b, ct_c0);
          if (kTM > 1u && !compact32 && m_off + 16u < valid_rows) {
            #pragma clang loop unroll(full)
            for (ushort i = 0u; i < 8u; ++i) {
              const auto ia = ct_a.get_multidimensional_index(i);
              const uint rm = m_off + 16u + uint(ia[1]);
              ct_a[i] = (rm < valid_rows)
                  ? mid_bf16[ulong(rb + rm) * p.k + k0 + kk + uint(ia[0])]
                  : bfloat(0.0f);
            }
            gemm_op.run(ct_a, ct_b, ct_c1);
          }
        }
      }
    }
    if (sg_active) {
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < kTM; ++mm) {
        if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            if (ct_c0.is_valid_element(i)) {
              const auto idx = ct_c0.get_multidimensional_index(i);
              const uint col = tg_n0 + sg_n * 32u + uint(idx[0]);
              const uint rm = m_off + mm * 16u + uint(idx[1]);
              if (rm < valid_rows && col < p.n) {
                const uint assign = tile_assign[rb + rm];
                const float score = float(route_scores[assign]);
                const float val = (mm == 0u) ? ct_c0[i] : ct_c1[i];
                expert_output[ulong(assign) * p.n + col] = bfloat(val * score);
              }
            }
          }
        }
      }
    }
    return;
  }

  const threadgroup bfloat *ws_sg = Ws + (sg_n * 32u + fm) * kBKP + fn;

  for (uint k0 = 0u; k0 < p.k; k0 += kBK) {
    const MoeAffineRaw32 raw = moe_nax_fetch_affine_32<Bits, GroupSize>(
        d_codes, d_high, d_scales, d_biases, expert_row, p.padded_k,
        k0 + w_col_sub, w_row_live);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    moe_nax_unpack_affine_32<Bits>(raw, w_dst8);
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
        for (uint mm = 0u; mm < kTM; ++mm) {
          if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
            vec<bfloat, 8> a0, a1;
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 2u; ++r) {
              const bfloat4 va0 = a_ok[mm][r]
                  ? *reinterpret_cast<device const bfloat4 *>(mid_bf16 + a_off[mm][r] + k0 + kk + 0u)
                  : bfloat4(bfloat(0.0f));
              const bfloat4 va1 = a_ok[mm][r]
                  ? *reinterpret_cast<device const bfloat4 *>(mid_bf16 + a_off[mm][r] + k0 + kk + 16u)
                  : bfloat4(bfloat(0.0f));
              #pragma clang loop unroll(full)
              for (uint c = 0u; c < 4u; ++c) {
                a0[r * 4u + c] = va0[c];
                a1[r * 4u + c] = va1[c];
              }
            }
            moe_nax_mma_16x32x16(D[mm][0], D[mm][1], a0, b00, b10);
            moe_nax_mma_16x32x16(D[mm][0], D[mm][1], a1, b01, b11);
          }
        }
      }
    }
  }

  if (sg_active) {
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < kTM; ++mm) {
      if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint rm = m_off + mm * 16u + fm + h * 8u;
          if (rm < valid_rows) {
            const uint assign = tile_assign[rb + rm];
            const float score = float(route_scores[assign]);
            #pragma clang loop unroll(full)
            for (uint nn = 0u; nn < 2u; ++nn) {
              const uint col = tg_n0 + sg_n * 32u + nn * 16u + fn;
              if (col < p.n) {
                const uint e = h * 4u;
                const bfloat4 out_v = bfloat4(
                    bfloat(D[mm][nn][e + 0u] * score),
                    bfloat(D[mm][nn][e + 1u] * score),
                    bfloat(D[mm][nn][e + 2u] * score),
                    bfloat(D[mm][nn][e + 3u] * score));
                *reinterpret_cast<device bfloat4 *>(
                    expert_output + ulong(assign) * p.n + col) = out_v;
              }
            }
          }
        }
      }
    }
  }
}

template <uint Bits, uint GroupSize>
inline bfloat moe_nax_decode_affine_elem(
    device const uchar *codes,
    device const uchar *high,
    device const bfloat *scales,
    device const bfloat *biases,
    ulong expert_row,
    uint padded_k,
    uint k_elem,
    bool row_live) {
  if (!row_live) {
    return bfloat(0.0f);
  }
  const ulong sb_idx =
      expert_row * ulong(padded_k / GroupSize) + ulong(k_elem / GroupSize);
  const float sc = float(scales[sb_idx]);
  const float bi = float(biases[sb_idx]);
  uint q = 0u;
  if constexpr (Bits == 4u) {
    const uchar b = codes[expert_row * ulong(padded_k >> 1u) + ulong(k_elem >> 1u)];
    q = (uint(b) >> ((k_elem & 1u) * 4u)) & 15u;
  } else if constexpr (Bits == 8u) {
    q = uint(codes[expert_row * ulong(padded_k) + ulong(k_elem)]);
  } else if constexpr (Bits == 6u) {
    const uchar b = codes[expert_row * ulong(padded_k >> 1u) + ulong(k_elem >> 1u)];
    const uchar h = high[expert_row * ulong(padded_k >> 2u) + ulong(k_elem >> 2u)];
    q = ((uint(b) >> ((k_elem & 1u) * 4u)) & 15u) |
        (((uint(h) >> ((k_elem & 3u) * 2u)) & 3u) << 4u);
  } else if constexpr (Bits == 5u) {
    const uchar b = codes[expert_row * ulong(padded_k >> 1u) + ulong(k_elem >> 1u)];
    const uchar h = high[expert_row * ulong(padded_k >> 3u) + ulong(k_elem >> 3u)];
    q = ((uint(b) >> ((k_elem & 1u) * 4u)) & 15u) |
        (((uint(h) >> (k_elem & 7u)) & 1u) << 4u);
  } else {
    const uchar b = codes[expert_row * ulong(padded_k >> 2u) + ulong(k_elem >> 2u)];
    const uchar h = high[expert_row * ulong(padded_k >> 3u) + ulong(k_elem >> 3u)];
    q = ((uint(b) >> ((k_elem & 3u) * 2u)) & 3u) |
        (((uint(h) >> (k_elem & 7u)) & 1u) << 2u);
  }
  return bfloat(fma(sc, float(q), bi));
}

template <uint Bits>
inline bfloat4 moe_nax_unpack_affine_q4_linear(
    device const uchar *codes,
    device const uchar *high,
    ulong expert_row,
    uint padded_k,
    uint k_elem,
    bool row_live,
    float sc,
    float bi) {
  if (!row_live) {
    return bfloat4(bfloat(0.0f));
  }
  bfloat4 out;
  if constexpr (Bits == 4u) {
    const ushort pk = *reinterpret_cast<device const ushort *>(
        codes + expert_row * ulong(padded_k >> 1u) + ulong(k_elem >> 1u));
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
      out[j] = bfloat(fma(sc, float((uint(pk) >> (j * 4u)) & 15u), bi));
    }
  } else if constexpr (Bits == 8u) {
    const uchar4 pk = *reinterpret_cast<device const uchar4 *>(
        codes + expert_row * ulong(padded_k) + ulong(k_elem));
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
      out[j] = bfloat(fma(sc, float(pk[j]), bi));
    }
  } else if constexpr (Bits == 6u) {
    const ushort lo = *reinterpret_cast<device const ushort *>(
        codes + expert_row * ulong(padded_k >> 1u) + ulong(k_elem >> 1u));
    const uchar hi = high[expert_row * ulong(padded_k >> 2u) + ulong(k_elem >> 2u)];
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
      const uint q = ((uint(lo) >> (j * 4u)) & 15u) |
                     (((uint(hi) >> (j * 2u)) & 3u) << 4u);
      out[j] = bfloat(fma(sc, float(q), bi));
    }
  } else if constexpr (Bits == 5u) {
    const ushort lo = *reinterpret_cast<device const ushort *>(
        codes + expert_row * ulong(padded_k >> 1u) + ulong(k_elem >> 1u));
    const uchar hb = high[expert_row * ulong(padded_k >> 3u) + ulong(k_elem >> 3u)];
    const uint h4 = (uint(hb) >> (k_elem & 4u)) & 15u;
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
      const uint q = ((uint(lo) >> (j * 4u)) & 15u) | (((h4 >> j) & 1u) << 4u);
      out[j] = bfloat(fma(sc, float(q), bi));
    }
  } else {
    const uchar lo = codes[expert_row * ulong(padded_k >> 2u) + ulong(k_elem >> 2u)];
    const uchar hb = high[expert_row * ulong(padded_k >> 3u) + ulong(k_elem >> 3u)];
    const uint h4 = (uint(hb) >> (k_elem & 4u)) & 15u;
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
      const uint q = ((uint(lo) >> (j * 2u)) & 3u) | (((h4 >> j) & 1u) << 2u);
      out[j] = bfloat(fma(sc, float(q), bi));
    }
  }
  return out;
}

// WP3.1: Compact Ws-Staged 2-Pass (BM=64, BN_out=64, BK=64) Direct-L1-X MoE Affine SwiGLU
// 4 SGs = 128 threads (WM=2, WN=2), SRAM = 9,216 B Ws + 256 B tg_tok = 9,472 B (9.25 KiB <= 10.0 KiB!), 0 spills!
template <uint Bits, uint GroupSize>
inline void moe_affine_nax_swiglu_bm64_out64_impl(
    device const bfloat *norm,
    device const uint *tile_assign,
    device const uint *tile_expert,
    device const uint *tile_valid_rows,
    device const uint *total_tiles,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const bfloat *g_scales,
    device const bfloat *g_biases,
    device const uchar *u_codes,
    device const uchar *u_high,
    device const bfloat *u_scales,
    device const bfloat *u_biases,
    device bfloat *mid_bf16,
    constant MoeNaxGemmParams &p,
    threadgroup MoeOut64TgStorage &tg_mem,
    uint tid,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint kBK = 64u;
  constexpr uint kBKP = 72u;
  constexpr uint kOutColsPerTg = 32u;

  const uint n_cols = (p.n + kOutColsPerTg - 1u) / kOutColsPerTg;
  const bool use_gx32 = (p.max_tiles >> 31u) != 0u;
  const uint m_tile = use_gx32 ? ((tg.y / n_cols) * 32u + tg.x) : tg.y;
  const uint col_tile = use_gx32 ? (tg.y % n_cols) : tg.x;
  const uint tg_n0 = col_tile * kOutColsPerTg;
  if (m_tile >= total_tiles[1] || col_tile >= n_cols || tg_n0 >= p.n) {
    return;
  }
  const uint valid_rows = tile_valid_rows[m_tile];
  if (valid_rows == 0u) {
    return;
  }
  const uint packed = tile_expert[m_tile];
  const uint expert = packed & 0xFFFFu;
  const uint rb = (packed >> 16u) * 8u;

  if (tid < 64u) {
    uint tok = 0xFFFFFFFFu;
    if (tid < valid_rows) {
      const uint assign = tile_assign[rb + tid];
      if (assign != 0xFFFFFFFFu) {
        tok = assign >> 3u;
      }
    }
    tg_mem.tg_tok[tid] = tok;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const bool compact32 = (valid_rows <= 32u);
  const uint m_off = sg_m * (compact32 ? 16u : 32u);
  const uint col_base = tg_n0 + sg_n * 16u;
  const bool sg_active = (m_off < valid_rows) && (col_base < p.n);

  const uint w_tile_row = tid >> 1u;
  const uint w_col_sub = (tid & 1u) * 32u;
  const bool is_up = ((w_tile_row >> 4u) & 1u) != 0u;
  const uint out_col = tg_n0 + ((w_tile_row >> 5u) * 16u) + (w_tile_row & 15u);
  const bool w_live = (out_col < p.n);
  device const uchar *w_codes = is_up ? u_codes : g_codes;
  device const uchar *w_high = is_up ? u_high : g_high;
  device const bfloat *w_scales = is_up ? u_scales : g_scales;
  device const bfloat *w_biases = is_up ? u_biases : g_biases;
  threadgroup vec<bfloat, 8> *w_dst8 =
      reinterpret_cast<threadgroup vec<bfloat, 8> *>(tg_mem.Ws + w_tile_row * kBKP + w_col_sub);
  const ulong expert_row = ulong(expert) * p.n + out_col;
  const uint k_blocks = p.k / kBK;
  const uint rb_8ceil = (valid_rows + 7u) & ~7u;

  if (!moe_has_apple10_nax_layout()) {
    constexpr auto desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> gemm_op;
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c0 = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, float>();
    auto ct_c1 = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) { ct_c0[i] = 0.0f; ct_c1[i] = 0.0f; }

    for (uint kb = 0u; kb < k_blocks; ++kb) {
      const uint k0 = kb * kBK;
      const MoeAffineRaw32 raw = moe_nax_fetch_affine_32<Bits, GroupSize>(
          w_codes, w_high, w_scales, w_biases, expert_row, p.padded_k, k0 + w_col_sub, w_live);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      moe_nax_unpack_affine_32<Bits>(raw, w_dst8);
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
            const uint tok = tg_mem.tg_tok[m_off + uint(ia[1])];
            ct_a[i] = (tok != 0xFFFFFFFFu) ? norm[ulong(tok) * p.k + k0 + kk + uint(ia[0])] : bfloat(0.0f);
          }
          gemm_op.run(ct_a, ct_b, ct_c0);
          if (!compact32 && m_off + 16u < valid_rows) {
            #pragma clang loop unroll(full)
            for (ushort i = 0u; i < 8u; ++i) {
              const auto ia = ct_a.get_multidimensional_index(i);
              const uint tok = tg_mem.tg_tok[m_off + 16u + uint(ia[1])];
              ct_a[i] = (tok != 0xFFFFFFFFu) ? norm[ulong(tok) * p.k + k0 + kk + uint(ia[0])] : bfloat(0.0f);
            }
            gemm_op.run(ct_a, ct_b, ct_c1);
          }
        }
      }
    }

    if (sg_active) {
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            const auto idx_g = ct_c0.get_multidimensional_index(i);
            const uint c_g = uint(idx_g[0]);
            const uint r_g = uint(idx_g[1]);
            if (c_g < 16u) {
              const uint col = col_base + c_g;
              const uint rm = m_off + mm * 16u + r_g;
              if (col < p.n && rm < valid_rows) {
                float gv = (mm == 0u) ? ct_c0[i] : ct_c1[i];
                float uv = 0.0f;
                #pragma clang loop unroll(full)
                for (ushort j = 0u; j < 16u; ++j) {
                  const auto idx_u = ct_c0.get_multidimensional_index(j);
                  if (uint(idx_u[0]) == c_g + 16u && uint(idx_u[1]) == r_g) {
                    uv = (mm == 0u) ? ct_c0[j] : ct_c1[j];
                  }
                }
                mid_bf16[ulong(rb + rm) * p.n + col] =
                    bfloat(nax_silu_f32(float(bfloat(gv))) * float(bfloat(uv)));
              } else if (col < p.n && rm < rb_8ceil) {
                mid_bf16[ulong(rb + rm) * p.n + col] = bfloat(0.0f);
              }
            }
          }
        }
      }
    }
    return;
  }

  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
  const threadgroup bfloat *ws_sg = tg_mem.Ws + (sg_n * 32u + fm) * kBKP + fn;

  uint a_tok[2][2];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < 2u; ++mm) {
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint rm = m_off + mm * 16u + fm + h * 8u;
      a_tok[mm][h] = (mm == 0u || !compact32) ? tg_mem.tg_tok[rm] : 0xFFFFFFFFu;
    }
  }

  vec<float, 8> D00 = vec<float, 8>(0.0f), D01 = vec<float, 8>(0.0f);
  vec<float, 8> D10 = vec<float, 8>(0.0f), D11 = vec<float, 8>(0.0f);

  MoeAffineRaw32 raw = moe_nax_fetch_affine_32<Bits, GroupSize>(
      w_codes, w_high, w_scales, w_biases, expert_row, p.padded_k, w_col_sub, w_live);

  for (uint kb = 0u; kb < k_blocks; ++kb) {
    const uint k0 = kb * kBK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    moe_nax_unpack_affine_32<Bits>(raw, w_dst8);
    if (kb + 1u < k_blocks) {
      raw = moe_nax_fetch_affine_32<Bits, GroupSize>(
          w_codes, w_high, w_scales, w_biases, expert_row, p.padded_k,
          (kb + 1u) * kBK + w_col_sub, w_live);
    }
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
            const uint tok = a_tok[0][r];
            const bfloat4 va0 = (tok != 0xFFFFFFFFu)
                ? *reinterpret_cast<device const bfloat4 *>(norm + ulong(tok) * p.k + k0 + kk + fn)
                : bfloat4(bfloat(0.0f));
            const bfloat4 va1 = (tok != 0xFFFFFFFFu)
                ? *reinterpret_cast<device const bfloat4 *>(norm + ulong(tok) * p.k + k0 + kk + 16u + fn)
                : bfloat4(bfloat(0.0f));
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) { a0[r * 4u + c] = va0[c]; a1[r * 4u + c] = va1[c]; }
          }
          moe_nax_mma_16x32x16(D00, D01, a0, b00, b10);
          moe_nax_mma_16x32x16(D00, D01, a1, b01, b11);
        }
        if (!compact32 && (m_off + 16u < valid_rows)) {
          vec<bfloat, 8> a0, a1;
          #pragma clang loop unroll(full)
          for (uint r = 0u; r < 2u; ++r) {
            const uint tok = a_tok[1][r];
            const bfloat4 va0 = (tok != 0xFFFFFFFFu)
                ? *reinterpret_cast<device const bfloat4 *>(norm + ulong(tok) * p.k + k0 + kk + fn)
                : bfloat4(bfloat(0.0f));
            const bfloat4 va1 = (tok != 0xFFFFFFFFu)
                ? *reinterpret_cast<device const bfloat4 *>(norm + ulong(tok) * p.k + k0 + kk + 16u + fn)
                : bfloat4(bfloat(0.0f));
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) { a0[r * 4u + c] = va0[c]; a1[r * 4u + c] = va1[c]; }
          }
          moe_nax_mma_16x32x16(D10, D11, a0, b00, b10);
          moe_nax_mma_16x32x16(D10, D11, a1, b01, b11);
        }
      }
    }
  }

  if (sg_active && col_base + fn < p.n) {
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint e = h * 4u;
      const uint rm0 = m_off + fm + h * 8u;
      if (rm0 < valid_rows) {
        bfloat4 ov;
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j)
          ov[j] = bfloat(nax_silu_f32(float(bfloat(D00[e + j]))) * float(bfloat(D01[e + j])));
        *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb + rm0) * p.n + col_base + fn) = ov;
      } else if (rm0 < rb_8ceil) {
        *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb + rm0) * p.n + col_base + fn) = bfloat4(bfloat(0.0f));
      }
      if (!compact32 && (m_off + 16u < valid_rows)) {
        const uint rm1 = m_off + 16u + fm + h * 8u;
        if (rm1 < valid_rows) {
          bfloat4 ov;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j)
            ov[j] = bfloat(nax_silu_f32(float(bfloat(D10[e + j]))) * float(bfloat(D11[e + j])));
          *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb + rm1) * p.n + col_base + fn) = ov;
        } else if (rm1 < rb_8ceil) {
          *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb + rm1) * p.n + col_base + fn) = bfloat4(bfloat(0.0f));
        }
      }
    }
  }
}

#define OMLX_M5_MOE_AFFINE_NO_HIGH(BITS, GS)                                   \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_moe_affine_nax_swiglu_q##BITS##_g##GS##_bm64_out64_sk1(  \
      device const bfloat *norm [[buffer(0)]],                                 \
      device const uint *tile_assign [[buffer(1)]],                            \
      device const uint *tile_expert [[buffer(2)]],                            \
      device const uint *tile_valid_rows [[buffer(3)]],                        \
      device const uint *total_tiles [[buffer(4)]],                            \
      device const uchar *g_codes [[buffer(5)]],                               \
      device const bfloat *g_scales [[buffer(6)]],                             \
      device const bfloat *g_biases [[buffer(7)]],                             \
      device const uchar *u_codes [[buffer(8)]],                               \
      device const bfloat *u_scales [[buffer(9)]],                             \
      device const bfloat *u_biases [[buffer(10)]],                            \
      device bfloat *mid_bf16 [[buffer(11)]],                                  \
      constant MoeNaxGemmParams &p [[buffer(12)]],                             \
      uint tid [[thread_index_in_threadgroup]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup MoeOut64TgStorage tg_mem;                                      \
    moe_affine_nax_swiglu_bm64_out64_impl<BITS, GS>(                           \
        norm, tile_assign, tile_expert, tile_valid_rows, total_tiles, g_codes, \
        nullptr, g_scales, g_biases, u_codes, nullptr, u_scales, u_biases,     \
        mid_bf16, p, tg_mem, tid, lane, sg, tg);                               \
  }                                                                            \
  kernel void omlx_m5_moe_affine_nax_swiglu_q##BITS##_g##GS##_n16_sk4(         \
      device const bfloat *x_bf16 [[buffer(0)]],                               \
      device const float *xsum [[buffer(1)]],                                  \
      device const uint *tile_expert [[buffer(2)]],                            \
      device const uint *tile_valid_rows [[buffer(3)]],                        \
      device const uint *total_tiles [[buffer(4)]],                            \
      device const uchar *g_codes [[buffer(5)]],                               \
      device const bfloat *g_scales [[buffer(6)]],                             \
      device const bfloat *g_biases [[buffer(7)]],                             \
      device const uchar *u_codes [[buffer(8)]],                               \
      device const bfloat *u_scales [[buffer(9)]],                             \
      device const bfloat *u_biases [[buffer(10)]],                            \
      device bfloat *mid_bf16 [[buffer(11)]],                                  \
      constant MoeNaxGemmParams &p [[buffer(12)]],                             \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[3u * 16u * 32u];                                   \
    moe_affine_nax_swiglu_n16_sk4_impl<BITS, GS, 1u>(                          \
        x_bf16, xsum, tile_expert, tile_valid_rows, total_tiles, g_codes,      \
        nullptr, g_scales, g_biases, u_codes, nullptr, u_scales, u_biases,     \
        mid_bf16, p, parts, nullptr, lane, sg, tg);                            \
  }                                                                            \
  kernel void omlx_m5_moe_affine_nax_swiglu_q##BITS##_g##GS##_n16_sk4_nsg2(    \
      device const bfloat *x_bf16 [[buffer(0)]],                               \
      device const float *xsum [[buffer(1)]],                                  \
      device const uint *tile_expert [[buffer(2)]],                            \
      device const uint *tile_valid_rows [[buffer(3)]],                        \
      device const uint *total_tiles [[buffer(4)]],                            \
      device const uchar *g_codes [[buffer(5)]],                               \
      device const bfloat *g_scales [[buffer(6)]],                             \
      device const bfloat *g_biases [[buffer(7)]],                             \
      device const uchar *u_codes [[buffer(8)]],                               \
      device const bfloat *u_scales [[buffer(9)]],                             \
      device const bfloat *u_biases [[buffer(10)]],                            \
      device bfloat *mid_bf16 [[buffer(11)]],                                  \
      constant MoeNaxGemmParams &p [[buffer(12)]],                             \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[2u * 3u * 16u * 32u];                              \
    moe_affine_nax_swiglu_n16_sk4_impl<BITS, GS, 2u>(                          \
        x_bf16, xsum, tile_expert, tile_valid_rows, total_tiles, g_codes,      \
        nullptr, g_scales, g_biases, u_codes, nullptr, u_scales, u_biases,     \
        mid_bf16, p, parts, nullptr, lane, sg, tg);                            \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_moe_affine_nax_swiglu_q##BITS##_g##GS##_bm32_bn64_sk1(   \
      device const bfloat *norm [[buffer(0)]],                                 \
      device const uint *tile_assign [[buffer(1)]],                            \
      device const uint *tile_expert [[buffer(2)]],                            \
      device const uint *tile_valid_rows [[buffer(3)]],                        \
      device const uint *total_tiles [[buffer(4)]],                            \
      device const uchar *g_codes [[buffer(5)]],                               \
      device const bfloat *g_scales [[buffer(6)]],                             \
      device const bfloat *g_biases [[buffer(7)]],                             \
      device const uchar *u_codes [[buffer(8)]],                               \
      device const bfloat *u_scales [[buffer(9)]],                             \
      device const bfloat *u_biases [[buffer(10)]],                            \
      device bfloat *mid_bf16 [[buffer(11)]],                                  \
      constant MoeNaxGemmParams &p [[buffer(12)]],                             \
      uint tid [[thread_index_in_threadgroup]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[64u * 72u];                              \
    threadgroup uint tg_tok[32u];                                              \
    moe_affine_nax_swiglu_wide_sk1_impl<BITS, GS, 32u>(                        \
        norm, tile_assign, tile_expert, tile_valid_rows, total_tiles, g_codes, \
        nullptr, g_scales, g_biases, u_codes, nullptr, u_scales, u_biases,     \
        mid_bf16, p, Ws, tg_tok, tid, lane, sg, tg);                           \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_moe_affine_nax_swiglu_q##BITS##_g##GS##_bm64_bn64_sk1(   \
      device const bfloat *norm [[buffer(0)]],                                 \
      device const uint *tile_assign [[buffer(1)]],                            \
      device const uint *tile_expert [[buffer(2)]],                            \
      device const uint *tile_valid_rows [[buffer(3)]],                        \
      device const uint *total_tiles [[buffer(4)]],                            \
      device const uchar *g_codes [[buffer(5)]],                               \
      device const bfloat *g_scales [[buffer(6)]],                             \
      device const bfloat *g_biases [[buffer(7)]],                             \
      device const uchar *u_codes [[buffer(8)]],                               \
      device const bfloat *u_scales [[buffer(9)]],                             \
      device const bfloat *u_biases [[buffer(10)]],                            \
      device bfloat *mid_bf16 [[buffer(11)]],                                  \
      constant MoeNaxGemmParams &p [[buffer(12)]],                             \
      uint tid [[thread_index_in_threadgroup]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[64u * 72u];                              \
    threadgroup uint tg_tok[64u];                                              \
    moe_affine_nax_swiglu_wide_sk1_impl<BITS, GS, 64u>(                        \
        norm, tile_assign, tile_expert, tile_valid_rows, total_tiles, g_codes, \
        nullptr, g_scales, g_biases, u_codes, nullptr, u_scales, u_biases,     \
        mid_bf16, p, Ws, tg_tok, tid, lane, sg, tg);                           \
  }                                                                            \
  kernel void omlx_m5_moe_affine_nax_down_scatter_q##BITS##_g##GS##_n16_sk2(   \
      device const bfloat *mid_bf16 [[buffer(0)]],                             \
      device const float *mid_xsum [[buffer(1)]],                              \
      device const uint *tile_expert [[buffer(2)]],                            \
      device const uint *tile_valid_rows [[buffer(3)]],                        \
      device const uint *tile_assign [[buffer(4)]],                            \
      device const uint *total_tiles [[buffer(5)]],                            \
      device const bfloat *route_scores [[buffer(6)]],                         \
      device const uchar *d_codes [[buffer(7)]],                               \
      device const bfloat *d_scales [[buffer(8)]],                             \
      device const bfloat *d_biases [[buffer(9)]],                             \
      device bfloat *expert_output [[buffer(10)]],                             \
      constant MoeNaxGemmParams &p [[buffer(11)]],                             \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[1u * 8u * 32u];                                    \
    moe_affine_nax_down_scatter_n16_sk2_impl<BITS, GS, 1u>(                    \
        mid_bf16, mid_xsum, tile_expert, tile_valid_rows, tile_assign,         \
        total_tiles, route_scores, d_codes, nullptr, d_scales, d_biases,       \
        expert_output, p, parts, nullptr, lane, sg, tg);                       \
  }                                                                            \
  kernel void omlx_m5_moe_affine_nax_down_scatter_q##BITS##_g##GS##_n16_sk2_nsg2(\
      device const bfloat *mid_bf16 [[buffer(0)]],                             \
      device const float *mid_xsum [[buffer(1)]],                              \
      device const uint *tile_expert [[buffer(2)]],                            \
      device const uint *tile_valid_rows [[buffer(3)]],                        \
      device const uint *tile_assign [[buffer(4)]],                            \
      device const uint *total_tiles [[buffer(5)]],                            \
      device const bfloat *route_scores [[buffer(6)]],                         \
      device const uchar *d_codes [[buffer(7)]],                               \
      device const bfloat *d_scales [[buffer(8)]],                             \
      device const bfloat *d_biases [[buffer(9)]],                             \
      device bfloat *expert_output [[buffer(10)]],                             \
      constant MoeNaxGemmParams &p [[buffer(11)]],                             \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[2u * 1u * 8u * 32u];                               \
    moe_affine_nax_down_scatter_n16_sk2_impl<BITS, GS, 2u>(                    \
        mid_bf16, mid_xsum, tile_expert, tile_valid_rows, tile_assign,         \
        total_tiles, route_scores, d_codes, nullptr, d_scales, d_biases,       \
        expert_output, p, parts, nullptr, lane, sg, tg);                       \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_moe_affine_nax_down_scatter_q##BITS##_g##GS##_bm32_bn64_sk1(\
      device const bfloat *mid_bf16 [[buffer(0)]],                             \
      device const uint *tile_expert [[buffer(1)]],                            \
      device const uint *tile_valid_rows [[buffer(2)]],                        \
      device const uint *tile_assign [[buffer(3)]],                            \
      device const uint *total_tiles [[buffer(4)]],                            \
      device const bfloat *route_scores [[buffer(5)]],                         \
      device const uchar *d_codes [[buffer(6)]],                               \
      device const bfloat *d_scales [[buffer(7)]],                             \
      device const bfloat *d_biases [[buffer(8)]],                             \
      device bfloat *expert_output [[buffer(9)]],                              \
      constant MoeNaxGemmParams &p [[buffer(10)]],                             \
      uint tid [[thread_index_in_threadgroup]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[64u * 72u];                              \
    moe_affine_nax_down_scatter_wide_sk1_impl<BITS, GS, 32u>(                  \
        mid_bf16, tile_expert, tile_valid_rows, tile_assign, total_tiles,      \
        route_scores, d_codes, nullptr, d_scales, d_biases, expert_output, p,  \
        Ws, tid, lane, sg, tg);                                                \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_moe_affine_nax_down_scatter_q##BITS##_g##GS##_bm64_bn64_sk1(\
      device const bfloat *mid_bf16 [[buffer(0)]],                             \
      device const uint *tile_expert [[buffer(1)]],                            \
      device const uint *tile_valid_rows [[buffer(2)]],                        \
      device const uint *tile_assign [[buffer(3)]],                            \
      device const uint *total_tiles [[buffer(4)]],                            \
      device const bfloat *route_scores [[buffer(5)]],                         \
      device const uchar *d_codes [[buffer(6)]],                               \
      device const bfloat *d_scales [[buffer(7)]],                             \
      device const bfloat *d_biases [[buffer(8)]],                             \
      device bfloat *expert_output [[buffer(9)]],                              \
      constant MoeNaxGemmParams &p [[buffer(10)]],                             \
      uint tid [[thread_index_in_threadgroup]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[64u * 72u];                              \
    moe_affine_nax_down_scatter_wide_sk1_impl<BITS, GS, 64u>(                  \
        mid_bf16, tile_expert, tile_valid_rows, tile_assign, total_tiles,      \
        route_scores, d_codes, nullptr, d_scales, d_biases, expert_output, p,  \
        Ws, tid, lane, sg, tg);                                                \
  }

#define OMLX_M5_MOE_AFFINE_WITH_HIGH(BITS, GS)                                 \
  kernel void omlx_m5_moe_affine_nax_swiglu_q##BITS##_g##GS##_n16_sk4(         \
      device const bfloat *x_bf16 [[buffer(0)]],                               \
      device const float *xsum [[buffer(1)]],                                  \
      device const uint *tile_expert [[buffer(2)]],                            \
      device const uint *tile_valid_rows [[buffer(3)]],                        \
      device const uint *total_tiles [[buffer(4)]],                            \
      device const uchar *g_codes [[buffer(5)]],                               \
      device const uchar *g_high [[buffer(6)]],                                \
      device const bfloat *g_scales [[buffer(7)]],                             \
      device const bfloat *g_biases [[buffer(8)]],                             \
      device const uchar *u_codes [[buffer(9)]],                               \
      device const uchar *u_high [[buffer(10)]],                               \
      device const bfloat *u_scales [[buffer(11)]],                            \
      device const bfloat *u_biases [[buffer(12)]],                            \
      device bfloat *mid_bf16 [[buffer(13)]],                                  \
      constant MoeNaxGemmParams &p [[buffer(14)]],                             \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[3u * 16u * 32u];                                   \
    threadgroup uchar tg_h2[(BITS == 3u || BITS == 5u) ? (4u * 2u * 256u) : 1u]; \
    moe_affine_nax_swiglu_n16_sk4_impl<BITS, GS, 1u>(                          \
        x_bf16, xsum, tile_expert, tile_valid_rows, total_tiles, g_codes,      \
        g_high, g_scales, g_biases, u_codes, u_high, u_scales, u_biases,       \
        mid_bf16, p, parts, tg_h2, lane, sg, tg);                              \
  }                                                                            \
  kernel void omlx_m5_moe_affine_nax_swiglu_q##BITS##_g##GS##_n16_sk4_nsg2(    \
      device const bfloat *x_bf16 [[buffer(0)]],                               \
      device const float *xsum [[buffer(1)]],                                  \
      device const uint *tile_expert [[buffer(2)]],                            \
      device const uint *tile_valid_rows [[buffer(3)]],                        \
      device const uint *total_tiles [[buffer(4)]],                            \
      device const uchar *g_codes [[buffer(5)]],                               \
      device const uchar *g_high [[buffer(6)]],                                \
      device const bfloat *g_scales [[buffer(7)]],                             \
      device const bfloat *g_biases [[buffer(8)]],                             \
      device const uchar *u_codes [[buffer(9)]],                               \
      device const uchar *u_high [[buffer(10)]],                               \
      device const bfloat *u_scales [[buffer(11)]],                            \
      device const bfloat *u_biases [[buffer(12)]],                            \
      device bfloat *mid_bf16 [[buffer(13)]],                                  \
      constant MoeNaxGemmParams &p [[buffer(14)]],                             \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[2u * 3u * 16u * 32u];                              \
    threadgroup uchar tg_h2[(BITS == 3u || BITS == 5u) ? (8u * 2u * 256u) : 1u]; \
    moe_affine_nax_swiglu_n16_sk4_impl<BITS, GS, 2u>(                          \
        x_bf16, xsum, tile_expert, tile_valid_rows, total_tiles, g_codes,      \
        g_high, g_scales, g_biases, u_codes, u_high, u_scales, u_biases,       \
        mid_bf16, p, parts, tg_h2, lane, sg, tg);                              \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_moe_affine_nax_swiglu_q##BITS##_g##GS##_bm32_bn64_sk1(   \
      device const bfloat *norm [[buffer(0)]],                                 \
      device const uint *tile_assign [[buffer(1)]],                            \
      device const uint *tile_expert [[buffer(2)]],                            \
      device const uint *tile_valid_rows [[buffer(3)]],                        \
      device const uint *total_tiles [[buffer(4)]],                            \
      device const uchar *g_codes [[buffer(5)]],                               \
      device const uchar *g_high [[buffer(6)]],                                \
      device const bfloat *g_scales [[buffer(7)]],                             \
      device const bfloat *g_biases [[buffer(8)]],                             \
      device const uchar *u_codes [[buffer(9)]],                               \
      device const uchar *u_high [[buffer(10)]],                               \
      device const bfloat *u_scales [[buffer(11)]],                            \
      device const bfloat *u_biases [[buffer(12)]],                            \
      device bfloat *mid_bf16 [[buffer(13)]],                                  \
      constant MoeNaxGemmParams &p [[buffer(14)]],                             \
      uint tid [[thread_index_in_threadgroup]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[64u * 72u];                              \
    threadgroup uint tg_tok[32u];                                              \
    moe_affine_nax_swiglu_wide_sk1_impl<BITS, GS, 32u>(                        \
        norm, tile_assign, tile_expert, tile_valid_rows, total_tiles, g_codes, \
        g_high, g_scales, g_biases, u_codes, u_high, u_scales, u_biases,       \
        mid_bf16, p, Ws, tg_tok, tid, lane, sg, tg);                           \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_moe_affine_nax_swiglu_q##BITS##_g##GS##_bm64_out64_sk1(  \
      device const bfloat *norm [[buffer(0)]],                                 \
      device const uint *tile_assign [[buffer(1)]],                            \
      device const uint *tile_expert [[buffer(2)]],                            \
      device const uint *tile_valid_rows [[buffer(3)]],                        \
      device const uint *total_tiles [[buffer(4)]],                            \
      device const uchar *g_codes [[buffer(5)]],                               \
      device const uchar *g_high [[buffer(6)]],                                \
      device const bfloat *g_scales [[buffer(7)]],                             \
      device const bfloat *g_biases [[buffer(8)]],                             \
      device const uchar *u_codes [[buffer(9)]],                               \
      device const uchar *u_high [[buffer(10)]],                               \
      device const bfloat *u_scales [[buffer(11)]],                            \
      device const bfloat *u_biases [[buffer(12)]],                            \
      device bfloat *mid_bf16 [[buffer(13)]],                                  \
      constant MoeNaxGemmParams &p [[buffer(14)]],                             \
      uint tid [[thread_index_in_threadgroup]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup MoeOut64TgStorage tg_mem;                                      \
    moe_affine_nax_swiglu_bm64_out64_impl<BITS, GS>(                           \
        norm, tile_assign, tile_expert, tile_valid_rows, total_tiles, g_codes, \
        g_high, g_scales, g_biases, u_codes, u_high, u_scales, u_biases,       \
        mid_bf16, p, tg_mem, tid, lane, sg, tg);                               \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_moe_affine_nax_swiglu_q##BITS##_g##GS##_bm64_bn64_sk1(   \
      device const bfloat *norm [[buffer(0)]],                                 \
      device const uint *tile_assign [[buffer(1)]],                            \
      device const uint *tile_expert [[buffer(2)]],                            \
      device const uint *tile_valid_rows [[buffer(3)]],                        \
      device const uint *total_tiles [[buffer(4)]],                            \
      device const uchar *g_codes [[buffer(5)]],                               \
      device const uchar *g_high [[buffer(6)]],                                \
      device const bfloat *g_scales [[buffer(7)]],                             \
      device const bfloat *g_biases [[buffer(8)]],                             \
      device const uchar *u_codes [[buffer(9)]],                               \
      device const uchar *u_high [[buffer(10)]],                               \
      device const bfloat *u_scales [[buffer(11)]],                            \
      device const bfloat *u_biases [[buffer(12)]],                            \
      device bfloat *mid_bf16 [[buffer(13)]],                                  \
      constant MoeNaxGemmParams &p [[buffer(14)]],                             \
      uint tid [[thread_index_in_threadgroup]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[64u * 72u];                              \
    threadgroup uint tg_tok[64u];                                              \
    moe_affine_nax_swiglu_wide_sk1_impl<BITS, GS, 64u>(                        \
        norm, tile_assign, tile_expert, tile_valid_rows, total_tiles, g_codes, \
        g_high, g_scales, g_biases, u_codes, u_high, u_scales, u_biases,       \
        mid_bf16, p, Ws, tg_tok, tid, lane, sg, tg);                           \
  }                                                                            \
  kernel void omlx_m5_moe_affine_nax_down_scatter_q##BITS##_g##GS##_n16_sk2(   \
      device const bfloat *mid_bf16 [[buffer(0)]],                             \
      device const float *mid_xsum [[buffer(1)]],                              \
      device const uint *tile_expert [[buffer(2)]],                            \
      device const uint *tile_valid_rows [[buffer(3)]],                        \
      device const uint *tile_assign [[buffer(4)]],                            \
      device const uint *total_tiles [[buffer(5)]],                            \
      device const bfloat *route_scores [[buffer(6)]],                         \
      device const uchar *d_codes [[buffer(7)]],                               \
      device const uchar *d_high [[buffer(8)]],                                \
      device const bfloat *d_scales [[buffer(9)]],                             \
      device const bfloat *d_biases [[buffer(10)]],                            \
      device bfloat *expert_output [[buffer(11)]],                             \
      constant MoeNaxGemmParams &p [[buffer(12)]],                             \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[1u * 8u * 32u];                                    \
    threadgroup uchar tg_h2[(BITS == 3u || BITS == 5u) ? (2u * 256u) : 1u];    \
    moe_affine_nax_down_scatter_n16_sk2_impl<BITS, GS, 1u>(                    \
        mid_bf16, mid_xsum, tile_expert, tile_valid_rows, tile_assign,         \
        total_tiles, route_scores, d_codes, d_high, d_scales, d_biases,        \
        expert_output, p, parts, tg_h2, lane, sg, tg);                         \
  }                                                                            \
  kernel void omlx_m5_moe_affine_nax_down_scatter_q##BITS##_g##GS##_n16_sk2_nsg2(\
      device const bfloat *mid_bf16 [[buffer(0)]],                             \
      device const float *mid_xsum [[buffer(1)]],                              \
      device const uint *tile_expert [[buffer(2)]],                            \
      device const uint *tile_valid_rows [[buffer(3)]],                        \
      device const uint *tile_assign [[buffer(4)]],                            \
      device const uint *total_tiles [[buffer(5)]],                            \
      device const bfloat *route_scores [[buffer(6)]],                         \
      device const uchar *d_codes [[buffer(7)]],                               \
      device const uchar *d_high [[buffer(8)]],                                \
      device const bfloat *d_scales [[buffer(9)]],                             \
      device const bfloat *d_biases [[buffer(10)]],                            \
      device bfloat *expert_output [[buffer(11)]],                             \
      constant MoeNaxGemmParams &p [[buffer(12)]],                             \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup float parts[2u * 1u * 8u * 32u];                               \
    threadgroup uchar tg_h2[(BITS == 3u || BITS == 5u) ? (4u * 256u) : 1u];    \
    moe_affine_nax_down_scatter_n16_sk2_impl<BITS, GS, 2u>(                    \
        mid_bf16, mid_xsum, tile_expert, tile_valid_rows, tile_assign,         \
        total_tiles, route_scores, d_codes, d_high, d_scales, d_biases,        \
        expert_output, p, parts, tg_h2, lane, sg, tg);                         \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_moe_affine_nax_down_scatter_q##BITS##_g##GS##_bm32_bn64_sk1(\
      device const bfloat *mid_bf16 [[buffer(0)]],                             \
      device const uint *tile_expert [[buffer(1)]],                            \
      device const uint *tile_valid_rows [[buffer(2)]],                        \
      device const uint *tile_assign [[buffer(3)]],                            \
      device const uint *total_tiles [[buffer(4)]],                            \
      device const bfloat *route_scores [[buffer(5)]],                         \
      device const uchar *d_codes [[buffer(6)]],                               \
      device const uchar *d_high [[buffer(7)]],                                \
      device const bfloat *d_scales [[buffer(8)]],                             \
      device const bfloat *d_biases [[buffer(9)]],                             \
      device bfloat *expert_output [[buffer(10)]],                             \
      constant MoeNaxGemmParams &p [[buffer(11)]],                             \
      uint tid [[thread_index_in_threadgroup]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[64u * 72u];                              \
    moe_affine_nax_down_scatter_wide_sk1_impl<BITS, GS, 32u>(                  \
        mid_bf16, tile_expert, tile_valid_rows, tile_assign, total_tiles,      \
        route_scores, d_codes, d_high, d_scales, d_biases, expert_output, p,   \
        Ws, tid, lane, sg, tg);                                                \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_moe_affine_nax_down_scatter_q##BITS##_g##GS##_bm64_bn64_sk1(\
      device const bfloat *mid_bf16 [[buffer(0)]],                             \
      device const uint *tile_expert [[buffer(1)]],                            \
      device const uint *tile_valid_rows [[buffer(2)]],                        \
      device const uint *tile_assign [[buffer(3)]],                            \
      device const uint *total_tiles [[buffer(4)]],                            \
      device const bfloat *route_scores [[buffer(5)]],                         \
      device const uchar *d_codes [[buffer(6)]],                               \
      device const uchar *d_high [[buffer(7)]],                                \
      device const bfloat *d_scales [[buffer(8)]],                             \
      device const bfloat *d_biases [[buffer(9)]],                             \
      device bfloat *expert_output [[buffer(10)]],                             \
      constant MoeNaxGemmParams &p [[buffer(11)]],                             \
      uint tid [[thread_index_in_threadgroup]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup bfloat Ws[64u * 72u];                              \
    moe_affine_nax_down_scatter_wide_sk1_impl<BITS, GS, 64u>(                  \
        mid_bf16, tile_expert, tile_valid_rows, tile_assign, total_tiles,      \
        route_scores, d_codes, d_high, d_scales, d_biases, expert_output, p,   \
        Ws, tid, lane, sg, tg);                                                \
  }

OMLX_M5_MOE_AFFINE_WITH_HIGH(3, 64)
OMLX_M5_MOE_AFFINE_WITH_HIGH(3, 128)
OMLX_M5_MOE_AFFINE_NO_HIGH(4, 64)
OMLX_M5_MOE_AFFINE_NO_HIGH(4, 128)
OMLX_M5_MOE_AFFINE_WITH_HIGH(5, 64)
OMLX_M5_MOE_AFFINE_WITH_HIGH(5, 128)
OMLX_M5_MOE_AFFINE_WITH_HIGH(6, 64)
OMLX_M5_MOE_AFFINE_WITH_HIGH(6, 128)
OMLX_M5_MOE_AFFINE_NO_HIGH(8, 64)
OMLX_M5_MOE_AFFINE_NO_HIGH(8, 128)


// ===========================================================================
// Dim B T=1 Pure-AR Fused 9-Expert (Top-8 Routed + 1 Shared) MoE Kernels
// ===========================================================================

struct Moe9RouterParams {
  uint h;                  // hidden size (multiple of 64, e.g. 2048)
  uint router_padded_k;    // row-major q8 router padded_k
  uint router_group_size;  // 64 or 128 for q8 router
  uint router_mode;        // 0 = direct_bf16 [256, h], 1 = oq_nax_tiled q8 (BN=16, GS=64), 2 = row-major q8
  uint sgate_padded_k;     // row-major q8 shared_gate padded_k
  uint sgate_group_size;   // 64 or 128 for q8 shared_gate
  uint sgate_padded_n;     // padded_n for oq_nax_tiled q8 shared_gate (typically 16)
  uint sgate_mode;         // 0 = skip, 1 = direct_bf16 [1, h], 2 = oq_nax_tiled q8, 3 = row-major q8
};

struct Moe9GemmParams {
  uint f;               // intermediate size (e.g. 512, multiple of 16)
  uint h;               // hidden size (e.g. 2048, multiple of 128)
  uint routed_padded_k; // gw.padded_k (Up) or dw.padded_k (Down)
  uint shared_padded_k; // shared_gw.padded_k (Up) or shared_dw.padded_k (Down) when shared_tiled == 0
  uint shared_tiled;    // 1 = mx_nax_tiled_bk128_v1 (BN=16), 0 = mx_rowsplit_k128_v1
  uint num_experts;     // 256
  uint top_k;           // 8
  uint flags;           // Down flags: bit 0 = compute inline BF16 shared_gate dot, bit 1 = add into norm_or_residual
};

// Subnormal-safe branchless FP4 E2M1 -> FP16 decoder (preserves +/-0.5 on Apple GPU).
inline half moe9_fp4_to_h16(uint nib) {
  const float f = float(as_type<half>(ushort((nib & 7u) << 9u))) * 16384.0f;
  return half((nib & 8u) != 0u ? -f : f);
}

// Single-TG (256 threads = 8 SGs) T=1 Fused Router QMV + Softmax + Top-8 + Shared Gate:
// Replaces up to 5 dispatches (xsum + router linear + top8 + shared_gate linear)
// with 1 single TG (~4 us on M5 Ultra, 1,056 B static TG SRAM).
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_moe9_router_top8_shared_gate_t1(
    device const bfloat *norm [[buffer(0)]],
    device const uchar *r_codes [[buffer(1)]],
    device const uchar *r_aux0 [[buffer(2)]],
    device const uchar *r_aux1 [[buffer(3)]],
    device const uchar *sg_codes [[buffer(4)]],
    device const uchar *sg_aux0 [[buffer(5)]],
    device const uchar *sg_aux1 [[buffer(6)]],
    device bfloat *router_logits_out [[buffer(7)]],
    device uint *route_ids_out [[buffer(8)]],
    device bfloat *route_scores_out [[buffer(9)]],
    device bfloat *shared_gate_out [[buffer(10)]],
    constant Moe9RouterParams &p [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float tg_logits[256];
  threadgroup float tg_sgate[8];

  const uint groups64 = p.h >> 6u;
  const uint r_sub = lane >> 3u; // 0..3 (4 rows per SG step)
  const uint k_l = lane & 7u;    // 0..7 (8 elements per lane -> 64 elements per group)

  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (uint g64 = 0u; g64 < groups64; ++g64) {
    device const bfloat4 *x4 =
        reinterpret_cast<device const bfloat4 *>(norm + g64 * 64u + k_l * 8u);
    const float4 xf0 = float4(x4[0]);
    const float4 xf1 = float4(x4[1]);

    float xsum64 = (xf0.x + xf0.y + xf0.z + xf0.w) +
                   (xf1.x + xf1.y + xf1.z + xf1.w);
    xsum64 += simd_shuffle_xor(xsum64, 4u);
    xsum64 += simd_shuffle_xor(xsum64, 2u);
    xsum64 += simd_shuffle_xor(xsum64, 1u);

    if (p.router_mode == 1u) {
      // oq_nax_tiled q8 (BN = 16, GS = 64): 32 lanes read 256 B contiguous (2 cache lines)!
      device const uint *sb_all = reinterpret_cast<device const uint *>(r_aux0);
      #pragma clang loop unroll(full)
      for (uint ps = 0u; ps < 8u; ++ps) {
        const uint row = sg * 32u + ps * 4u + r_sub;
        const uint tile_n = row >> 4u;
        const uint rn = row & 15u;
        const ulong idx = ulong(tile_n) * groups64 + g64;
        const uchar4 qw0 = *reinterpret_cast<device const uchar4 *>(
            r_codes + idx * 1024ul + rn * 64ul + k_l * 8ul + 0ul);
        const uchar4 qw1 = *reinterpret_cast<device const uchar4 *>(
            r_codes + idx * 1024ul + rn * 64ul + k_l * 8ul + 4ul);
        const vec<bfloat, 2> sbp =
            as_type<vec<bfloat, 2>>(sb_all[ulong(g64) * 256ul + row]);
        const float dot8 =
            dot(xf0, float4(qw0)) + dot(xf1, float4(qw1));
        acc[ps] = fma(float(sbp[0]), dot8, acc[ps]);
        if (k_l == 0u) {
          acc[ps] = fma(float(sbp[1]), xsum64, acc[ps]);
        }
      }
    } else if (p.router_mode == 2u) {
      // Row-major oq_affine_q8_g{64,128}
      const uint gs_r = ((p.router_group_size & 0x7FFFFFFFu) == 128u) ? 128u : 64u;
      const uint sb_g = (gs_r == 128u) ? (g64 >> 1u) : g64;
      const uint groups_per_row = p.router_padded_k / gs_r;
      device const bfloat *scales = reinterpret_cast<device const bfloat *>(r_aux0);
      device const bfloat *biases = reinterpret_cast<device const bfloat *>(r_aux1);
      #pragma clang loop unroll(full)
      for (uint ps = 0u; ps < 8u; ++ps) {
        const uint row = sg * 32u + ps * 4u + r_sub;
        device const uchar *rc =
            r_codes + ulong(row) * p.router_padded_k + g64 * 64u + k_l * 8u;
        const uchar4 qw0 = *reinterpret_cast<device const uchar4 *>(rc + 0u);
        const uchar4 qw1 = *reinterpret_cast<device const uchar4 *>(rc + 4u);
        const ulong sb_idx = ulong(row) * groups_per_row + sb_g;
        const float sc = float(scales[sb_idx]);
        const float bi = float(biases[sb_idx]);
        const float dot8 =
            dot(xf0, float4(qw0)) + dot(xf1, float4(qw1));
        acc[ps] = fma(sc, dot8, acc[ps]);
        if (k_l == 0u) {
          acc[ps] = fma(bi, xsum64, acc[ps]);
        }
      }
    } else {
      // direct_bf16_v1 [256, h]
      #pragma clang loop unroll(full)
      for (uint ps = 0u; ps < 8u; ++ps) {
        const uint row = sg * 32u + ps * 4u + r_sub;
        device const bfloat4 *w4 = reinterpret_cast<device const bfloat4 *>(
            r_codes + (ulong(row) * p.h + g64 * 64u + k_l * 8u) * 2ul);
        acc[ps] += dot(xf0, float4(w4[0])) + dot(xf1, float4(w4[1]));
      }
    }
  }

  #pragma clang loop unroll(full)
  for (uint ps = 0u; ps < 8u; ++ps) {
    float s = acc[ps];
    s += simd_shuffle_xor(s, 4u);
    s += simd_shuffle_xor(s, 2u);
    s += simd_shuffle_xor(s, 1u);
    if (k_l == 0u) {
      const uint row = sg * 32u + ps * 4u + r_sub;
      const bfloat bf = bfloat(s);
      router_logits_out[row] = bf;
      tg_logits[row] = float(bf);
    }
  }

  // Compute optional shared_gate [1, h] dot across all 256 threads in parallel
  if (p.sgate_mode != 0u) {
    float sg_acc = 0.0f;
    if (p.sgate_mode == 1u) {
      device const bfloat *sgw = reinterpret_cast<device const bfloat *>(sg_codes);
      for (uint col = tid; col < p.h; col += 256u) {
        sg_acc = fma(float(norm[col]), float(sgw[col]), sg_acc);
      }
    } else if (p.sgate_mode == 2u) {
      // oq_nax_tiled q8: row 0 of tile 0 -> tile_idx = g64, rn = 0 (64 B/row, BN=16 -> 1024 B/tile)
      device const uint *sg_sb = reinterpret_cast<device const uint *>(sg_aux0);
      const uint pad_n = max(p.sgate_padded_n, 16u);
      for (uint col = tid; col < p.h; col += 256u) {
        const uint g64 = col >> 6u;
        const float xv = float(norm[col]);
        const float qv = float(sg_codes[ulong(g64) * 1024ul + (col & 63u)]);
        const vec<bfloat, 2> sbp =
            as_type<vec<bfloat, 2>>(sg_sb[ulong(g64) * pad_n]);
        sg_acc = fma(xv, fma(float(sbp[0]), qv, float(sbp[1])), sg_acc);
      }
    } else if (p.sgate_mode == 3u) {
      const uint gs_s = ((p.sgate_group_size & 0x7FFFFFFFu) == 128u) ? 128u : 64u;
      device const bfloat *scales = reinterpret_cast<device const bfloat *>(sg_aux0);
      device const bfloat *biases = reinterpret_cast<device const bfloat *>(sg_aux1);
      for (uint col = tid; col < p.h; col += 256u) {
        const uint g = col / gs_s;
        const float xv = float(norm[col]);
        const float qv = float(sg_codes[col]);
        sg_acc = fma(xv, fma(float(scales[g]), qv, float(biases[g])), sg_acc);
      }
    }
    sg_acc = simd_sum(sg_acc);
    if (lane == 0u) {
      tg_sgate[sg] = sg_acc;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (p.sgate_mode != 0u && tid == 0u) {
    float gtot = 0.0f;
    for (uint s = 0u; s < 8u; ++s) {
      gtot += tg_sgate[s];
    }
    shared_gate_out[0] = bfloat(gtot);
  }

  // Exact 0-ULP top-8 selection + softmax on SG 0 (matches moe_nax_router_top8_in_tg bit-for-bit)
  if (sg == 0u) {
    float values[8];
    uint taken_mask = 0u;
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
      values[i] = tg_logits[lane * 8u + i];
    }
    float selected[8];
    uint ids[8];
#pragma clang loop unroll(full)
    for (uint k = 0u; k < 8u; ++k) {
      int tier = -1;
      float best = -INFINITY;
      uint best_local = 0u;
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 8u; ++i) {
        if ((taken_mask & (1u << i)) != 0u) continue;
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
      const uint id = lane * 8u + best_local;
      const uint winner = simd_max(is_best ? id : 0u);
      if (is_best && id == winner) {
        taken_mask |= (1u << best_local);
      }
      selected[k] = global_tier == 1 ? maximum : -INFINITY;
      ids[k] = winner;
    }
    if (lane == 0u) {
      float sum = 0.0f;
      float probabilities[8];
      const float max_logit = selected[0];
      for (uint k = 0u; k < 8u; ++k) {
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
      for (uint k = 0u; k < 8u; ++k) {
        route_ids_out[k] = ids[k];
        route_scores_out[k] =
            sum > 0.0f && isfinite(sum) ? bfloat(probabilities[k] / sum)
                                        : bfloat(0.0f);
      }
    }
  }
}

// 65-TG (tg.x = 0..63 compute router_logits_out[0..255] across all 64 GPU cores;
// tg.x = 64 computes shared_gate_out[0] concurrently with ZERO single-TG tail serialization!).
[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_moe9_router_qmv_64tg_t1(
    device const bfloat *norm [[buffer(0)]],
    device const uchar *r_codes [[buffer(1)]],
    device const uchar *r_aux0 [[buffer(2)]],
    device const uchar *r_aux1 [[buffer(3)]],
    device const uchar *sg_codes [[buffer(4)]],
    device const uchar *sg_aux0 [[buffer(5)]],
    device const uchar *sg_aux1 [[buffer(6)]],
    device bfloat *router_logits_out [[buffer(7)]],
    device bfloat *shared_gate_out [[buffer(8)]],
    constant Moe9RouterParams &p [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint groups64 = p.h >> 6u;
  if (tg.x < 64u) {
    const uint row_sg = (tg.x * 2u + sg) * 2u; // 0..254
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    if (p.router_mode == 1u) {
      device const uint *sb = reinterpret_cast<device const uint *>(r_aux0);
      const uint tile_n = row_sg >> 4u;
      const uint rn0 = row_sg & 15u;
      const ulong tile_base = ulong(tile_n) * groups64;
      const bool r_lane_v2 = (p.router_group_size >> 31u) != 0u;
      const uint g_sub = lane >> 3u; // 0..3 (4 groups of 64 = 256 K-elems/step!)
      const uint k_l = lane & 7u;    // 0..7 (8 bytes = uint2 per lane!)
      const uint c4_0 = r_lane_v2 ? omlx_m5_pi16(k_l * 2u) : (k_l * 2u);
      const uint c4_1 = r_lane_v2 ? omlx_m5_pi16(k_l * 2u + 1u) : (k_l * 2u + 1u);
      for (uint g = 0u; g < groups64; g += 4u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          device const bfloat4 *xb4 = reinterpret_cast<device const bfloat4 *>(
              norm + cur_g * 64u + k_l * 8u);
          const float4 xf0 = float4(xb4[0]);
          const float4 xf1 = float4(xb4[1]);
          const float lsum = ((xf0.x + xf0.y) + (xf0.z + xf0.w)) +
                             ((xf1.x + xf1.y) + (xf1.z + xf1.w));
          const ulong idx = tile_base + cur_g;
          device const uchar *c_tile = r_codes + idx * 1024ul;
          device const uint *r0_u32 = reinterpret_cast<device const uint *>(
              c_tile + ulong(rn0 + 0u) * 64ul);
          device const uint *r1_u32 = reinterpret_cast<device const uint *>(
              c_tile + ulong(rn0 + 1u) * 64ul);
          const uint2 cw0 = uint2(r0_u32[c4_0], r0_u32[c4_1]);
          const uint2 cw1 = uint2(r1_u32[c4_0], r1_u32[c4_1]);
          const uint sbp0 = sb[ulong(cur_g) * 256ul + row_sg + 0u];
          const uint sbp1 = sb[ulong(cur_g) * 256ul + row_sg + 1u];
          const float sc0 = as_type<float>((sbp0 & 0xFFFFu) << 16u);
          const float bi0 = as_type<float>(sbp0 & 0xFFFF0000u) - 1024.0f * sc0;
          const float sc1 = as_type<float>((sbp1 & 0xFFFFu) << 16u);
          const float bi1 = as_type<float>(sbp1 & 0xFFFF0000u) - 1024.0f * sc1;
          float d0 = fma(xf0.x, float(as_type<half>(ushort(0x6400u | (cw0.x & 0xFFu)))), 0.0f);
          d0 = fma(xf0.y, float(as_type<half>(ushort(0x6400u | ((cw0.x >> 8u) & 0xFFu)))), d0);
          d0 = fma(xf0.z, float(as_type<half>(ushort(0x6400u | ((cw0.x >> 16u) & 0xFFu)))), d0);
          d0 = fma(xf0.w, float(as_type<half>(ushort(0x6400u | (cw0.x >> 24u)))), d0);
          d0 = fma(xf1.x, float(as_type<half>(ushort(0x6400u | (cw0.y & 0xFFu)))), d0);
          d0 = fma(xf1.y, float(as_type<half>(ushort(0x6400u | ((cw0.y >> 8u) & 0xFFu)))), d0);
          d0 = fma(xf1.z, float(as_type<half>(ushort(0x6400u | ((cw0.y >> 16u) & 0xFFu)))), d0);
          d0 = fma(xf1.w, float(as_type<half>(ushort(0x6400u | (cw0.y >> 24u)))), d0);
          float d1 = fma(xf0.x, float(as_type<half>(ushort(0x6400u | (cw1.x & 0xFFu)))), 0.0f);
          d1 = fma(xf0.y, float(as_type<half>(ushort(0x6400u | ((cw1.x >> 8u) & 0xFFu)))), d1);
          d1 = fma(xf0.z, float(as_type<half>(ushort(0x6400u | ((cw1.x >> 16u) & 0xFFu)))), d1);
          d1 = fma(xf0.w, float(as_type<half>(ushort(0x6400u | (cw1.x >> 24u)))), d1);
          d1 = fma(xf1.x, float(as_type<half>(ushort(0x6400u | (cw1.y & 0xFFu)))), d1);
          d1 = fma(xf1.y, float(as_type<half>(ushort(0x6400u | ((cw1.y >> 8u) & 0xFFu)))), d1);
          d1 = fma(xf1.z, float(as_type<half>(ushort(0x6400u | ((cw1.y >> 16u) & 0xFFu)))), d1);
          d1 = fma(xf1.w, float(as_type<half>(ushort(0x6400u | (cw1.y >> 24u)))), d1);
          acc0 = fma(sc0, d0, fma(bi0, lsum, acc0));
          acc1 = fma(sc1, d1, fma(bi1, lsum, acc1));
        }
      }
    } else if (p.router_mode == 0u) {
      device const bfloat *rw = reinterpret_cast<device const bfloat *>(r_codes);
      const uint g_sub = lane >> 3u;
      const uint k_l = lane & 7u;
      for (uint g = 0u; g < groups64; g += 4u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const uint col = cur_g * 64u + k_l * 8u;
          device const bfloat4 *xb4 = reinterpret_cast<device const bfloat4 *>(norm + col);
          device const bfloat4 *w0b = reinterpret_cast<device const bfloat4 *>(rw + ulong(row_sg + 0u) * p.h + col);
          device const bfloat4 *w1b = reinterpret_cast<device const bfloat4 *>(rw + ulong(row_sg + 1u) * p.h + col);
          const float4 xf0 = float4(xb4[0]), xf1 = float4(xb4[1]);
          const float4 w00 = float4(w0b[0]), w01 = float4(w0b[1]);
          const float4 w10 = float4(w1b[0]), w11 = float4(w1b[1]);
          acc0 = fma(xf0.x, w00.x, fma(xf0.y, w00.y, fma(xf0.z, w00.z, fma(xf0.w, w00.w, acc0))));
          acc0 = fma(xf1.x, w01.x, fma(xf1.y, w01.y, fma(xf1.z, w01.z, fma(xf1.w, w01.w, acc0))));
          acc1 = fma(xf0.x, w10.x, fma(xf0.y, w10.y, fma(xf0.z, w10.z, fma(xf0.w, w10.w, acc1))));
          acc1 = fma(xf1.x, w11.x, fma(xf1.y, w11.y, fma(xf1.z, w11.z, fma(xf1.w, w11.w, acc1))));
        }
      }
    } else {
      const uint gs = ((p.router_group_size & 0x7FFFFFFFu) == 128u) ? 128u : 64u;
      const uint sb_stride = p.router_padded_k / gs;
      device const bfloat *scales = reinterpret_cast<device const bfloat *>(r_aux0);
      device const bfloat *biases = reinterpret_cast<device const bfloat *>(r_aux1);
      const uint g_sub = lane >> 4u;
      const uint k_l = lane & 15u;
      for (uint g = 0u; g < groups64; g += 2u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const float4 xf = float4(*reinterpret_cast<device const bfloat4 *>(
              norm + cur_g * 64u + k_l * 4u));
          const float lsum = (xf.x + xf.y) + (xf.z + xf.w);
          const uint g_src = (gs == 128u) ? (cur_g >> 1u) : cur_g;
          const uchar4 cb0 = reinterpret_cast<device const uchar4 *>(
              r_codes + ulong(row_sg + 0u) * p.router_padded_k + cur_g * 64u)[k_l];
          const uchar4 cb1 = reinterpret_cast<device const uchar4 *>(
              r_codes + ulong(row_sg + 1u) * p.router_padded_k + cur_g * 64u)[k_l];
          const float sc0 = float(scales[ulong(row_sg + 0u) * sb_stride + g_src]);
          const float bi0 = float(biases[ulong(row_sg + 0u) * sb_stride + g_src]) - 1024.0f * sc0;
          const float sc1 = float(scales[ulong(row_sg + 1u) * sb_stride + g_src]);
          const float bi1 = float(biases[ulong(row_sg + 1u) * sb_stride + g_src]) - 1024.0f * sc1;
          float d0 = fma(xf.x, float(as_type<half>(ushort(0x6400u | cb0.x))), 0.0f);
          d0 = fma(xf.y, float(as_type<half>(ushort(0x6400u | cb0.y))), d0);
          d0 = fma(xf.z, float(as_type<half>(ushort(0x6400u | cb0.z))), d0);
          d0 = fma(xf.w, float(as_type<half>(ushort(0x6400u | cb0.w))), d0);
          float d1 = fma(xf.x, float(as_type<half>(ushort(0x6400u | cb1.x))), 0.0f);
          d1 = fma(xf.y, float(as_type<half>(ushort(0x6400u | cb1.y))), d1);
          d1 = fma(xf.z, float(as_type<half>(ushort(0x6400u | cb1.z))), d1);
          d1 = fma(xf.w, float(as_type<half>(ushort(0x6400u | cb1.w))), d1);
          acc0 = fma(sc0, d0, fma(bi0, lsum, acc0));
          acc1 = fma(sc1, d1, fma(bi1, lsum, acc1));
        }
      }
    }
    const float v0 = simd_sum(acc0);
    const float v1 = simd_sum(acc1);
    if (lane == 0u) {
      router_logits_out[row_sg + 0u] = bfloat(v0);
      router_logits_out[row_sg + 1u] = bfloat(v1);
    }
    return;
  }

  // Dedicated tg.x == 64: computes shared_gate_out[0] concurrently with router TGs!
  if (p.sgate_mode != 0u) {
    threadgroup float tg_sg[2];
    float sg_acc = 0.0f;
    const uint g_sub = sg * 2u + (lane >> 4u); // 0..3 (4 groups of 64 = 256 K-elems/iter!)
    const uint k_l = lane & 15u;               // 4 elements/lane (bfloat4 / uchar4)
    if (p.sgate_mode == 1u) {
      device const bfloat *sgw = reinterpret_cast<device const bfloat *>(sg_codes);
      for (uint g = 0u; g < groups64; g += 4u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const uint col = cur_g * 64u + k_l * 4u;
          const float4 xf = float4(*reinterpret_cast<device const bfloat4 *>(norm + col));
          const float4 wf = float4(*reinterpret_cast<device const bfloat4 *>(sgw + col));
          sg_acc = fma(xf.x, wf.x, fma(xf.y, wf.y, fma(xf.z, wf.z, fma(xf.w, wf.w, sg_acc))));
        }
      }
    } else if (p.sgate_mode == 2u) {
      device const uint *sg_sb = reinterpret_cast<device const uint *>(sg_aux0);
      const uint pad_n = max(p.sgate_padded_n, 16u);
      for (uint g = 0u; g < groups64; g += 4u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const float4 xf = float4(*reinterpret_cast<device const bfloat4 *>(
              norm + cur_g * 64u + k_l * 4u));
          const float lsum = (xf.x + xf.y) + (xf.z + xf.w);
          const uint sg_c4 = ((p.sgate_group_size >> 31u) != 0u) ? omlx_m5_pi16(k_l) : k_l;
          const uchar4 cb = reinterpret_cast<device const uchar4 *>(
              sg_codes + ulong(cur_g) * 1024ul)[sg_c4];
          const uint sbp = sg_sb[ulong(cur_g) * pad_n];
          const float sc = as_type<float>((sbp & 0xFFFFu) << 16u);
          const float bi = as_type<float>(sbp & 0xFFFF0000u) - 1024.0f * sc;
          float d = fma(xf.x, float(as_type<half>(ushort(0x6400u | cb.x))), 0.0f);
          d = fma(xf.y, float(as_type<half>(ushort(0x6400u | cb.y))), d);
          d = fma(xf.z, float(as_type<half>(ushort(0x6400u | cb.z))), d);
          d = fma(xf.w, float(as_type<half>(ushort(0x6400u | cb.w))), d);
          sg_acc = fma(sc, d, fma(bi, lsum, sg_acc));
        }
      }
    } else if (p.sgate_mode == 3u) {
      const uint gs_s = ((p.sgate_group_size & 0x7FFFFFFFu) == 128u) ? 128u : 64u;
      device const bfloat *scales = reinterpret_cast<device const bfloat *>(sg_aux0);
      device const bfloat *biases = reinterpret_cast<device const bfloat *>(sg_aux1);
      for (uint g = 0u; g < groups64; g += 4u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const uint col = cur_g * 64u + k_l * 4u;
          const float4 xf = float4(*reinterpret_cast<device const bfloat4 *>(norm + col));
          const float lsum = (xf.x + xf.y) + (xf.z + xf.w);
          const uchar4 cb = reinterpret_cast<device const uchar4 *>(sg_codes + col)[0];
          const uint g_src = (gs_s == 128u) ? (cur_g >> 1u) : cur_g;
          const float sc = float(scales[g_src]);
          const float bi = float(biases[g_src]) - 1024.0f * sc;
          float d = fma(xf.x, float(as_type<half>(ushort(0x6400u | cb.x))), 0.0f);
          d = fma(xf.y, float(as_type<half>(ushort(0x6400u | cb.y))), d);
          d = fma(xf.z, float(as_type<half>(ushort(0x6400u | cb.z))), d);
          d = fma(xf.w, float(as_type<half>(ushort(0x6400u | cb.w))), d);
          sg_acc = fma(sc, d, fma(bi, lsum, sg_acc));
        }
      }
    }
    sg_acc = simd_sum(sg_acc);
    if (lane == 0u) {
      tg_sg[sg] = sg_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      shared_gate_out[0] = bfloat(tg_sg[0] + tg_sg[1]);
    }
  }
}

// Specialized zero-branch Q8 pre-tiled 65-TG router + shared_gate for 35B-A3B (r_mode==1, sgate_mode==2)!
[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_moe9_router_q8_pretiled_65tg_t1(
    device const bfloat *norm [[buffer(0)]],
    device const uchar *r_codes [[buffer(1)]],
    device const uchar *r_aux0 [[buffer(2)]],
    device const uchar *r_aux1 [[buffer(3)]],
    device const uchar *sg_codes [[buffer(4)]],
    device const uchar *sg_aux0 [[buffer(5)]],
    device const uchar *sg_aux1 [[buffer(6)]],
    device bfloat *router_logits_out [[buffer(7)]],
    device bfloat *shared_gate_out [[buffer(8)]],
    constant Moe9RouterParams &p [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  (void)r_aux1;
  (void)sg_aux1;
  const uint groups64 = p.h >> 6u;
  if (tg.x < 64u) {
    const uint row_sg = (tg.x * 2u + sg) * 2u;
    device const uint *sb = reinterpret_cast<device const uint *>(r_aux0);
    const uint tile_n = row_sg >> 4u;
    const uint rn0 = row_sg & 15u;
    const ulong tile_base = ulong(tile_n) * groups64;
    const uint g_sub = lane >> 3u;
    const uint k_l = lane & 7u;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint g = 0u; g < groups64; g += 4u) {
      const uint cur_g = g + g_sub;
      if (cur_g < groups64) {
        device const bfloat4 *xb4 = reinterpret_cast<device const bfloat4 *>(
            norm + cur_g * 64u + k_l * 8u);
        const float4 xf0 = float4(xb4[0]);
        const float4 xf1 = float4(xb4[1]);
        const float lsum = ((xf0.x + xf0.y) + (xf0.z + xf0.w)) +
                           ((xf1.x + xf1.y) + (xf1.z + xf1.w));
        const ulong idx = tile_base + cur_g;
        device const uchar *c_tile = r_codes + idx * 1024ul;
        const bool r_lane_v2 = (p.router_group_size >> 31u) != 0u;
        const uint c4_0 = r_lane_v2 ? omlx_m5_pi16(k_l * 2u) : (k_l * 2u);
        const uint c4_1 = r_lane_v2 ? omlx_m5_pi16(k_l * 2u + 1u) : (k_l * 2u + 1u);
        device const uint *r0_u32 = reinterpret_cast<device const uint *>(
            c_tile + ulong(rn0 + 0u) * 64ul);
        device const uint *r1_u32 = reinterpret_cast<device const uint *>(
            c_tile + ulong(rn0 + 1u) * 64ul);
        const uint2 cw0 = uint2(r0_u32[c4_0], r0_u32[c4_1]);
        const uint2 cw1 = uint2(r1_u32[c4_0], r1_u32[c4_1]);
        const uint sbp0 = sb[ulong(cur_g) * 256ul + row_sg + 0u];
        const uint sbp1 = sb[ulong(cur_g) * 256ul + row_sg + 1u];
        const float sc0 = as_type<float>((sbp0 & 0xFFFFu) << 16u);
        const float bi0 = as_type<float>(sbp0 & 0xFFFF0000u) - 1024.0f * sc0;
        const float sc1 = as_type<float>((sbp1 & 0xFFFFu) << 16u);
        const float bi1 = as_type<float>(sbp1 & 0xFFFF0000u) - 1024.0f * sc1;
        float d0 = fma(xf0.x, float(as_type<half>(ushort(0x6400u | (cw0.x & 0xFFu)))), 0.0f);
        d0 = fma(xf0.y, float(as_type<half>(ushort(0x6400u | ((cw0.x >> 8u) & 0xFFu)))), d0);
        d0 = fma(xf0.z, float(as_type<half>(ushort(0x6400u | ((cw0.x >> 16u) & 0xFFu)))), d0);
        d0 = fma(xf0.w, float(as_type<half>(ushort(0x6400u | (cw0.x >> 24u)))), d0);
        d0 = fma(xf1.x, float(as_type<half>(ushort(0x6400u | (cw0.y & 0xFFu)))), d0);
        d0 = fma(xf1.y, float(as_type<half>(ushort(0x6400u | ((cw0.y >> 8u) & 0xFFu)))), d0);
        d0 = fma(xf1.z, float(as_type<half>(ushort(0x6400u | ((cw0.y >> 16u) & 0xFFu)))), d0);
        d0 = fma(xf1.w, float(as_type<half>(ushort(0x6400u | (cw0.y >> 24u)))), d0);
        float d1 = fma(xf0.x, float(as_type<half>(ushort(0x6400u | (cw1.x & 0xFFu)))), 0.0f);
        d1 = fma(xf0.y, float(as_type<half>(ushort(0x6400u | ((cw1.x >> 8u) & 0xFFu)))), d1);
        d1 = fma(xf0.z, float(as_type<half>(ushort(0x6400u | ((cw1.x >> 16u) & 0xFFu)))), d1);
        d1 = fma(xf0.w, float(as_type<half>(ushort(0x6400u | (cw1.x >> 24u)))), d1);
        d1 = fma(xf1.x, float(as_type<half>(ushort(0x6400u | (cw1.y & 0xFFu)))), d1);
        d1 = fma(xf1.y, float(as_type<half>(ushort(0x6400u | ((cw1.y >> 8u) & 0xFFu)))), d1);
        d1 = fma(xf1.z, float(as_type<half>(ushort(0x6400u | ((cw1.y >> 16u) & 0xFFu)))), d1);
        d1 = fma(xf1.w, float(as_type<half>(ushort(0x6400u | (cw1.y >> 24u)))), d1);
        acc0 = fma(sc0, d0, fma(bi0, lsum, acc0));
        acc1 = fma(sc1, d1, fma(bi1, lsum, acc1));
      }
    }
    const float v0 = simd_sum(acc0);
    const float v1 = simd_sum(acc1);
    if (lane == 0u) {
      router_logits_out[row_sg + 0u] = bfloat(v0);
      router_logits_out[row_sg + 1u] = bfloat(v1);
    }
    return;
  }
  if (p.sgate_mode == 2u) {
    threadgroup float tg_sg[2];
    device const uint *sg_sb = reinterpret_cast<device const uint *>(sg_aux0);
    const uint pad_n = max(p.sgate_padded_n, 16u);
    const uint g_sub = sg * 2u + (lane >> 4u);
    const uint k_l = lane & 15u;
    float sg_acc = 0.0f;
    for (uint g = 0u; g < groups64; g += 4u) {
      const uint cur_g = g + g_sub;
      if (cur_g < groups64) {
        const float4 xf = float4(*reinterpret_cast<device const bfloat4 *>(
            norm + cur_g * 64u + k_l * 4u));
        const float lsum = (xf.x + xf.y) + (xf.z + xf.w);
        const uint sg_c4 = ((p.sgate_group_size >> 31u) != 0u) ? omlx_m5_pi16(k_l) : k_l;
        const uchar4 cb = reinterpret_cast<device const uchar4 *>(
            sg_codes + ulong(cur_g) * 1024ul)[sg_c4];
        const uint sbp = sg_sb[ulong(cur_g) * pad_n];
        const float sc = as_type<float>((sbp & 0xFFFFu) << 16u);
        const float bi = as_type<float>(sbp & 0xFFFF0000u) - 1024.0f * sc;
        float d = fma(xf.x, float(as_type<half>(ushort(0x6400u | cb.x))), 0.0f);
        d = fma(xf.y, float(as_type<half>(ushort(0x6400u | cb.y))), d);
        d = fma(xf.z, float(as_type<half>(ushort(0x6400u | cb.z))), d);
        d = fma(xf.w, float(as_type<half>(ushort(0x6400u | cb.w))), d);
        sg_acc = fma(sc, d, fma(bi, lsum, sg_acc));
      }
    }
    sg_acc = simd_sum(sg_acc);
    if (lane == 0u) {
      tg_sg[sg] = sg_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      shared_gate_out[0] = bfloat(tg_sg[0] + tg_sg[1]);
    }
  }
}

// Grid: ((f + 3) / 4, 9, 1), Threads: (64, 1, 1) (2 SGs x 2 rows/SG = 4 rows/TG -> 1,152 TGs = 36 SGs/core on M5 Ultra, 0 B TG SRAM).
// tg.y = 0..7 reads row-major routed experts [256, f, h] (128 B full cache-line reads across 32 lanes!);
// tg.y = 8 reads shared_expert [f, h] (either pre-tiled mx_nax_tiled_bk128_v1 or row-major).
[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_moe9_mxfp4_kl16_swiglu_t1(
    device const bfloat *norm [[buffer(0)]],
    device const uint *route_ids [[buffer(1)]],
    device const uchar *routed_gc [[buffer(2)]],
    device const uchar *routed_gs [[buffer(3)]],
    device const uchar *routed_uc [[buffer(4)]],
    device const uchar *routed_us [[buffer(5)]],
    device const uchar *shared_gc [[buffer(6)]],
    device const uchar *shared_gs [[buffer(7)]],
    device const uchar *shared_uc [[buffer(8)]],
    device const uchar *shared_us [[buffer(9)]],
    device bfloat *mid_out [[buffer(10)]],
    constant Moe9GemmParams &p [[buffer(11)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint slot = tg.y;
  const uint row_sg = tg.x * 4u + sg * 2u;
  if (slot > p.top_k || row_sg >= p.f) {
    return;
  }

  uint exp_id = 0u;
  if (slot < p.top_k) {
    exp_id = route_ids[slot];
    if (exp_id >= p.num_experts) {
      if (lane < 2u && row_sg + lane < p.f) {
        mid_out[ulong(slot) * p.f + row_sg + lane] = bfloat(0.0f);
      }
      return;
    }
  }

  const uint b_sub = lane >> 4u; // 0..1 (2 blocks of 128 = 256 K-elements per step)
  const uint k_l = lane & 15u;   // 0..15 (8 nibbles = 4 B per lane -> 128 B across 32 lanes)
  const uint sub32 = k_l >> 2u;  // 0..3
  const uint blocks = p.h >> 7u;

  float acc_g[2] = {0.0f, 0.0f};
  float acc_u[2] = {0.0f, 0.0f};

  const ulong r_c_stride = ulong(p.routed_padded_k >> 1u);
  const ulong r_s_stride = ulong(p.routed_padded_k >> 5u);
  const ulong s_c_stride = ulong(p.shared_padded_k >> 1u);
  const ulong s_s_stride = ulong(p.shared_padded_k >> 5u);

  for (uint blk = 0u; blk < blocks; blk += 2u) {
    const uint cur_blk = blk + b_sub;
    if (cur_blk < blocks) {
      device const bfloat4 *xb =
          reinterpret_cast<device const bfloat4 *>(norm + cur_blk * 128u + k_l * 8u);
      const half4 h0 = half4(float4(xb[0]));
      const half4 h1 = half4(float4(xb[1]));

      if (slot < p.top_k) {
        const ulong exp_row0 = ulong(exp_id) * p.f + row_sg;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 2u; ++r) {
          if (row_sg + r < p.f) {
            const ulong row_idx = exp_row0 + r;
            const uint gw = reinterpret_cast<device const uint *>(
                routed_gc + row_idx * r_c_stride + cur_blk * 64ul)[k_l];
            const uint uw = reinterpret_cast<device const uint *>(
                routed_uc + row_idx * r_c_stride + cur_blk * 64ul)[k_l];
            const float gsc = e8m0_to_float(
                (routed_gs + row_idx * r_s_stride + cur_blk * 4ul)[sub32]);
            const float usc = e8m0_to_float(
                (routed_us + row_idx * r_s_stride + cur_blk * 4ul)[sub32]);
            const half dg =
                h0.x * moe9_fp4_to_h16(gw) + h0.y * moe9_fp4_to_h16(gw >> 4u) +
                h0.z * moe9_fp4_to_h16(gw >> 8u) + h0.w * moe9_fp4_to_h16(gw >> 12u) +
                h1.x * moe9_fp4_to_h16(gw >> 16u) + h1.y * moe9_fp4_to_h16(gw >> 20u) +
                h1.z * moe9_fp4_to_h16(gw >> 24u) + h1.w * moe9_fp4_to_h16(gw >> 28u);
            const half du =
                h0.x * moe9_fp4_to_h16(uw) + h0.y * moe9_fp4_to_h16(uw >> 4u) +
                h0.z * moe9_fp4_to_h16(uw >> 8u) + h0.w * moe9_fp4_to_h16(uw >> 12u) +
                h1.x * moe9_fp4_to_h16(uw >> 16u) + h1.y * moe9_fp4_to_h16(uw >> 20u) +
                h1.z * moe9_fp4_to_h16(uw >> 24u) + h1.w * moe9_fp4_to_h16(uw >> 28u);
            acc_g[r] = fma(float(dg), gsc, acc_g[r]);
            acc_u[r] = fma(float(du), usc, acc_u[r]);
          }
        }
      } else if (p.shared_tiled != 0u) {
        const uint tile_n = row_sg >> 4u;
        const uint rn0 = row_sg & 15u;
        const ulong idx = ulong(tile_n) * blocks + cur_blk;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 2u; ++r) {
          if (row_sg + r < p.f) {
            const uint rn = rn0 + r;
            const uint gw = reinterpret_cast<device const uint *>(
                shared_gc + idx * 1024ul + rn * 64ul)[k_l];
            const uint uw = reinterpret_cast<device const uint *>(
                shared_uc + idx * 1024ul + rn * 64ul)[k_l];
            const float gsc = e8m0_to_float(
                (shared_gs + idx * 64ul + rn * 4ul)[sub32]);
            const float usc = e8m0_to_float(
                (shared_us + idx * 64ul + rn * 4ul)[sub32]);
            const half dg =
                h0.x * moe9_fp4_to_h16(gw) + h0.y * moe9_fp4_to_h16(gw >> 4u) +
                h0.z * moe9_fp4_to_h16(gw >> 8u) + h0.w * moe9_fp4_to_h16(gw >> 12u) +
                h1.x * moe9_fp4_to_h16(gw >> 16u) + h1.y * moe9_fp4_to_h16(gw >> 20u) +
                h1.z * moe9_fp4_to_h16(gw >> 24u) + h1.w * moe9_fp4_to_h16(gw >> 28u);
            const half du =
                h0.x * moe9_fp4_to_h16(uw) + h0.y * moe9_fp4_to_h16(uw >> 4u) +
                h0.z * moe9_fp4_to_h16(uw >> 8u) + h0.w * moe9_fp4_to_h16(uw >> 12u) +
                h1.x * moe9_fp4_to_h16(uw >> 16u) + h1.y * moe9_fp4_to_h16(uw >> 20u) +
                h1.z * moe9_fp4_to_h16(uw >> 24u) + h1.w * moe9_fp4_to_h16(uw >> 28u);
            acc_g[r] = fma(float(dg), gsc, acc_g[r]);
            acc_u[r] = fma(float(du), usc, acc_u[r]);
          }
        }
      } else {
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 2u; ++r) {
          const uint row = row_sg + r;
          if (row < p.f) {
            const uint gw = reinterpret_cast<device const uint *>(
                shared_gc + ulong(row) * s_c_stride + cur_blk * 64ul)[k_l];
            const uint uw = reinterpret_cast<device const uint *>(
                shared_uc + ulong(row) * s_c_stride + cur_blk * 64ul)[k_l];
            const float gsc = e8m0_to_float(
                (shared_gs + ulong(row) * s_s_stride + cur_blk * 4ul)[sub32]);
            const float usc = e8m0_to_float(
                (shared_us + ulong(row) * s_s_stride + cur_blk * 4ul)[sub32]);
            const half dg =
                h0.x * moe9_fp4_to_h16(gw) + h0.y * moe9_fp4_to_h16(gw >> 4u) +
                h0.z * moe9_fp4_to_h16(gw >> 8u) + h0.w * moe9_fp4_to_h16(gw >> 12u) +
                h1.x * moe9_fp4_to_h16(gw >> 16u) + h1.y * moe9_fp4_to_h16(gw >> 20u) +
                h1.z * moe9_fp4_to_h16(gw >> 24u) + h1.w * moe9_fp4_to_h16(gw >> 28u);
            const half du =
                h0.x * moe9_fp4_to_h16(uw) + h0.y * moe9_fp4_to_h16(uw >> 4u) +
                h0.z * moe9_fp4_to_h16(uw >> 8u) + h0.w * moe9_fp4_to_h16(uw >> 12u) +
                h1.x * moe9_fp4_to_h16(uw >> 16u) + h1.y * moe9_fp4_to_h16(uw >> 20u) +
                h1.z * moe9_fp4_to_h16(uw >> 24u) + h1.w * moe9_fp4_to_h16(uw >> 28u);
            acc_g[r] = fma(float(dg), gsc, acc_g[r]);
            acc_u[r] = fma(float(du), usc, acc_u[r]);
          }
        }
      }
    }
  }

  #pragma clang loop unroll(full)
  for (uint r = 0u; r < 2u; ++r) {
    const float gsum = simd_sum(acc_g[r]);
    const float usum = simd_sum(acc_u[r]);
    if (lane == 0u && row_sg + r < p.f) {
      const float gv = float(bfloat(gsum));
      const float uv = float(bfloat(usum));
      mid_out[ulong(slot) * p.f + row_sg + r] = bfloat(nax_silu_f32(gv) * uv);
    }
  }
}

// Grid: ((h + 7) / 8, 1, 1), Threads: (96, 1, 1) (3 SGs x 3 slots/SG = 9 experts in parallel, 140 B TG SRAM).
// Writes combined delta_out[0..h-1] directly for zero-extra-dispatch consumption by add_normalize.
[[max_total_threads_per_threadgroup(96)]]
kernel void omlx_m5_moe9_mxfp4_kl16_down_sk3_r8_t1(
    device const bfloat *mid_in [[buffer(0)]],
    device const uint *route_ids [[buffer(1)]],
    device const bfloat *route_scores [[buffer(2)]],
    device const bfloat *shared_gate_buf [[buffer(3)]],
    device const bfloat *norm_or_residual [[buffer(4)]],
    device const uchar *routed_dc [[buffer(5)]],
    device const uchar *routed_ds [[buffer(6)]],
    device const uchar *shared_dc [[buffer(7)]],
    device const uchar *shared_ds [[buffer(8)]],
    device bfloat *delta_out [[buffer(9)]],
    constant Moe9GemmParams &p [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row_base = tg.x * 8u;
  if (row_base >= p.h || sg >= 3u) {
    return;
  }

  threadgroup float sg_gate_part[3];
  threadgroup float tg_red[3][8];
  threadgroup float tg_shared[8];

  float gate_logit = 0.0f;
  if ((p.flags & 1u) != 0u) {
    const uint tid = sg * 32u + lane;
    const uint h4 = p.h >> 2u;
    device const bfloat4 *gw4 = reinterpret_cast<device const bfloat4 *>(shared_gate_buf);
    device const bfloat4 *xn4 = reinterpret_cast<device const bfloat4 *>(norm_or_residual);
    float g_dot = 0.0f;
    for (uint i = tid; i < h4; i += 96u) {
      g_dot += dot(float4(gw4[i]), float4(xn4[i]));
    }
    g_dot = simd_sum(g_dot);
    if (lane == 0u) sg_gate_part[sg] = g_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gate_logit = float(bfloat(sg_gate_part[0] + sg_gate_part[1] + sg_gate_part[2]));
  } else {
    gate_logit = float(shared_gate_buf[0]);
  }

  const uint b_sub = lane >> 4u;
  const uint k_l = lane & 15u;
  const uint sub32 = k_l >> 2u;
  const uint blocks = p.f >> 7u;

  const ulong r_c_stride = ulong(p.routed_padded_k >> 1u);
  const ulong r_s_stride = ulong(p.routed_padded_k >> 5u);
  const ulong s_c_stride = ulong(p.shared_padded_k >> 1u);
  const ulong s_s_stride = ulong(p.shared_padded_k >> 5u);

  float partial_routed[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float shared_vals[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  const uint s0 = sg * 3u;
  for (uint slot = s0; slot < s0 + 3u; ++slot) {
    if (slot > p.top_k) continue;

    uint exp_id = 0u;
    float w_score = 1.0f;
    if (slot < p.top_k) {
      exp_id = route_ids[slot];
      w_score = float(route_scores[slot]);
      if (exp_id >= p.num_experts || w_score == 0.0f) {
        continue;
      }
    }

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    device const bfloat *xm = mid_in + ulong(slot) * p.f;

    const bool do_fwht =
        (slot < p.top_k) ? ((p.flags & 4u) != 0u) : ((p.flags & 8u) != 0u);
    for (uint blk = 0u; blk < blocks; blk += 2u) {
      const uint cur_blk = blk + b_sub;
      float4 xf0 = float4(0.0f);
      float4 xf1 = float4(0.0f);
      if (cur_blk < blocks) {
        device const bfloat4 *xb =
            reinterpret_cast<device const bfloat4 *>(xm + cur_blk * 128u + k_l * 8u);
        xf0 = float4(xb[0]);
        xf1 = float4(xb[1]);
      }
      if (do_fwht) {
        omlx_m5_fwht64_sub8(xf0, xf1, lane);
      }
      if (cur_blk < blocks) {
        const half4 h0 = half4(xf0);
        const half4 h1 = half4(xf1);

        if (slot < p.top_k) {
          const ulong exp_row0 = ulong(exp_id) * p.h + row_base;
          #pragma clang loop unroll(full)
          for (uint r = 0u; r < 8u; ++r) {
            if (row_base + r < p.h) {
              const ulong row_idx = exp_row0 + r;
              const uint w = reinterpret_cast<device const uint *>(
                  routed_dc + row_idx * r_c_stride + cur_blk * 64ul)[k_l];
              const float sc = e8m0_to_float(
                  (routed_ds + row_idx * r_s_stride + cur_blk * 4ul)[sub32]);
              const half d =
                  h0.x * moe9_fp4_to_h16(w) + h0.y * moe9_fp4_to_h16(w >> 4u) +
                  h0.z * moe9_fp4_to_h16(w >> 8u) + h0.w * moe9_fp4_to_h16(w >> 12u) +
                  h1.x * moe9_fp4_to_h16(w >> 16u) + h1.y * moe9_fp4_to_h16(w >> 20u) +
                  h1.z * moe9_fp4_to_h16(w >> 24u) + h1.w * moe9_fp4_to_h16(w >> 28u);
              acc[r] = fma(float(d), sc, acc[r]);
            }
          }
        } else if (p.shared_tiled != 0u) {
          const uint tile_n = row_base >> 4u;
          const uint rn0 = row_base & 15u;
          const ulong idx = ulong(tile_n) * blocks + cur_blk;
          #pragma clang loop unroll(full)
          for (uint r = 0u; r < 8u; ++r) {
            if (row_base + r < p.h) {
              const uint rn = rn0 + r;
              const uint w = reinterpret_cast<device const uint *>(
                  shared_dc + idx * 1024ul + rn * 64ul)[k_l];
              const float sc = e8m0_to_float(
                  (shared_ds + idx * 64ul + rn * 4ul)[sub32]);
              const half d =
                  h0.x * moe9_fp4_to_h16(w) + h0.y * moe9_fp4_to_h16(w >> 4u) +
                  h0.z * moe9_fp4_to_h16(w >> 8u) + h0.w * moe9_fp4_to_h16(w >> 12u) +
                  h1.x * moe9_fp4_to_h16(w >> 16u) + h1.y * moe9_fp4_to_h16(w >> 20u) +
                  h1.z * moe9_fp4_to_h16(w >> 24u) + h1.w * moe9_fp4_to_h16(w >> 28u);
              acc[r] = fma(float(d), sc, acc[r]);
            }
          }
        } else {
          #pragma clang loop unroll(full)
          for (uint r = 0u; r < 8u; ++r) {
            const uint row = row_base + r;
            if (row < p.h) {
              const uint w = reinterpret_cast<device const uint *>(
                  shared_dc + ulong(row) * s_c_stride + cur_blk * 64ul)[k_l];
              const float sc = e8m0_to_float(
                  (shared_ds + ulong(row) * s_s_stride + cur_blk * 4ul)[sub32]);
              const half d =
                  h0.x * moe9_fp4_to_h16(w) + h0.y * moe9_fp4_to_h16(w >> 4u) +
                  h0.z * moe9_fp4_to_h16(w >> 8u) + h0.w * moe9_fp4_to_h16(w >> 12u) +
                  h1.x * moe9_fp4_to_h16(w >> 16u) + h1.y * moe9_fp4_to_h16(w >> 20u) +
                  h1.z * moe9_fp4_to_h16(w >> 24u) + h1.w * moe9_fp4_to_h16(w >> 28u);
              acc[r] = fma(float(d), sc, acc[r]);
            }
          }
        }
      }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; ++r) {
      const float s = simd_sum(acc[r]);
      if (lane == 0u) {
        if (slot < p.top_k) {
          partial_routed[r] += float(bfloat(s * w_score));
        } else {
          shared_vals[r] = float(bfloat(s));
        }
      }
    }
  }

  if (lane == 0u) {
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; ++r) {
      tg_red[sg][r] = partial_routed[r];
      if (sg == 2u) {
        tg_shared[r] = shared_vals[r];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u && lane < 8u && row_base + lane < p.h) {
    const uint col = row_base + lane;
    const bfloat routed_bf16 =
        bfloat(tg_red[0][lane] + tg_red[1][lane] + tg_red[2][lane]);
    const float shared_scale = 1.0f / (1.0f + exp(-gate_logit));
    const float delta_f =
        float(routed_bf16) + shared_scale * tg_shared[lane];
    if ((p.flags & 2u) != 0u) {
      delta_out[col] = bfloat(float(norm_or_residual[col]) + delta_f);
    } else {
      delta_out[col] = bfloat(delta_f);
    }
  }
}

// ===========================================================================
// v0.8.0-m5rt.11 Radical Bespoke T=1 MoE 1-Dispatch Router+Top8+SharedGate+Norm
// & Fused Down+Add+RmsSq Epilogue (Eliminates 80 dispatches + 80 barriers on 35B!)
// ===========================================================================

struct Moe9RouterFusedTop8Params {
  uint h;
  uint router_padded_k;
  uint router_group_size;
  uint router_mode;
  uint sgate_padded_k;
  uint sgate_group_size;
  uint sgate_padded_n;
  uint sgate_mode;
  uint sq_slot;
  uint moe_slot;
  float inv_h;
  float eps;
  float w_off;
};

inline void moe9_top8_from_atomic_u32(
    device atomic_uint *logits_u32,
    device uint *indices,
    device bfloat *scores,
    uint lane) {
  const uint u0 = atomic_load_explicit(&logits_u32[lane * 4u + 0u], memory_order_relaxed);
  const uint u1 = atomic_load_explicit(&logits_u32[lane * 4u + 1u], memory_order_relaxed);
  const uint u2 = atomic_load_explicit(&logits_u32[lane * 4u + 2u], memory_order_relaxed);
  const uint u3 = atomic_load_explicit(&logits_u32[lane * 4u + 3u], memory_order_relaxed);
  const ushort raw[8] = {
      ushort(u0 & 0xFFFFu), ushort(u0 >> 16u),
      ushort(u1 & 0xFFFFu), ushort(u1 >> 16u),
      ushort(u2 & 0xFFFFu), ushort(u2 >> 16u),
      ushort(u3 & 0xFFFFu), ushort(u3 >> 16u),
  };
  bool local_nan = false;
#pragma clang loop unroll(full)
  for (uint i = 0u; i < 8u; ++i) {
    local_nan |= ((raw[i] & 0x7FFFu) > 0x7F80u);
  }
  float selected[8];
  uint ids[8];
  if (!simd_any(local_nan)) {
    uint packed[8];
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
      const uint bf = ((raw[i] & 0x7FFFu) == 0u) ? 0u : uint(raw[i]);
      const uint key16 =
          (bf & 0x8000u) ? ((~bf) & 0xFFFFu) : (bf | 0x8000u);
      packed[i] = (key16 << 16u) | 0x0100u | (lane * 8u + i);
    }
#pragma clang loop unroll(full)
    for (uint k = 0u; k < 8u; ++k) {
      const uint best_p =
          max(max(max(packed[0], packed[1]), max(packed[2], packed[3])),
              max(max(packed[4], packed[5]), max(packed[6], packed[7])));
      const uint win_p = simd_max(best_p);
      const uint winner = win_p & 0xFFu;
      const uint win_key16 = win_p >> 16u;
      const uint orig_bf = (win_key16 & 0x8000u) ? (win_key16 & 0x7FFFu)
                                                 : ((~win_key16) & 0xFFFFu);
      selected[k] = as_type<float>(orig_bf << 16u);
      ids[k] = winner;
      if ((winner >> 3u) == lane) {
        const uint w_slot = winner & 7u;
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 8u; ++i) {
          if (i == w_slot) {
            packed[i] = 0u;
          }
        }
      }
    }
  } else {
    float values[8];
    uint taken_mask = 0u;
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
      values[i] = as_type<float>(uint(raw[i]) << 16u);
    }
#pragma clang loop unroll(full)
    for (uint k = 0u; k < 8u; ++k) {
      int tier = -1;
      float best = -INFINITY;
      uint best_local = 0u;
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 8u; ++i) {
        if ((taken_mask & (1u << i)) != 0u) continue;
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
      const uint id = lane * 8u + best_local;
      const uint winner = simd_max(is_best ? id : 0u);
      if (is_best && id == winner) {
        taken_mask |= (1u << best_local);
      }
      selected[k] = global_tier == 1 ? maximum : -INFINITY;
      ids[k] = winner;
    }
  }
  if (lane == 0u) {
    float sum = 0.0f;
    float probabilities[8];
    const float max_logit = selected[0];
    for (uint k = 0u; k < 8u; ++k) {
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
    for (uint k = 0u; k < 8u; ++k) {
      indices[k] = ids[k];
      scores[k] = sum > 0.0f && isfinite(sum) ? bfloat(probabilities[k] / sum)
                                              : bfloat(0.0f);
    }
  }
}

template <bool PROLOGUE_NORM, bool FAST_Q8_PRETILED>
inline void moe9_router_fused_top8_impl(
    device const bfloat *x_or_hidden,
    device const bfloat *rms_w,
    device const float *rms_sq_buf,
    device bfloat *norm_out,
    device const uchar *r_codes,
    device const uchar *r_aux0,
    device const uchar *r_aux1,
    device const uchar *sg_codes,
    device const uchar *sg_aux0,
    device const uchar *sg_aux1,
    device atomic_uint *router_logits_u32,
    device uint *route_ids_out,
    device bfloat *route_scores_out,
    device bfloat *shared_gate_out,
    device atomic_uint *moe_tg_done_buf,
    threadgroup float (&tg_sg)[2],
    constant Moe9RouterFusedTop8Params &p,
    uint tid,
    uint lane,
    uint sg,
    uint3 tg) {
  float inv_rms = 1.0f;
  float4 off4 = float4(0.0f);
  if constexpr (PROLOGUE_NORM) {
    inv_rms = rsqrt(rms_sq_buf[p.sq_slot] * p.inv_h + p.eps);
    off4 = float4(p.w_off);
    const uint norm_col4 = tg.x * 64u + tid;
    if (norm_col4 < (p.h >> 2u)) {
      device const bfloat4 *hv4 = reinterpret_cast<device const bfloat4 *>(x_or_hidden);
      device const bfloat4 *wv4 = reinterpret_cast<device const bfloat4 *>(rms_w);
      device bfloat4 *nv4 = reinterpret_cast<device bfloat4 *>(norm_out);
      nv4[norm_col4] =
          bfloat4(float4(hv4[norm_col4]) * inv_rms * (float4(wv4[norm_col4]) + off4));
    }
  } else {
    (void)rms_w;
    (void)rms_sq_buf;
    (void)norm_out;
  }

  const uint groups64 = p.h >> 6u;
  if (tg.x < 64u) {
    const uint row_sg = (tg.x * 2u + sg) * 2u;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    if (FAST_Q8_PRETILED || p.router_mode == 1u) {
      device const uint *sb = reinterpret_cast<device const uint *>(r_aux0);
      const uint tile_n = row_sg >> 4u;
      const uint rn0 = row_sg & 15u;
      const ulong tile_base = ulong(tile_n) * groups64;
      const uint g_sub = lane >> 3u;
      const uint k_l = lane & 7u;
      for (uint g = 0u; g < groups64; g += 4u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const uint col = cur_g * 64u + k_l * 8u;
          device const bfloat4 *xb4 = reinterpret_cast<device const bfloat4 *>(x_or_hidden + col);
          float4 xf0, xf1;
          if constexpr (PROLOGUE_NORM) {
            device const bfloat4 *wb4 = reinterpret_cast<device const bfloat4 *>(rms_w + col);
            xf0 = float4(bfloat4(float4(xb4[0]) * inv_rms * (float4(wb4[0]) + off4)));
            xf1 = float4(bfloat4(float4(xb4[1]) * inv_rms * (float4(wb4[1]) + off4)));
          } else {
            xf0 = float4(xb4[0]);
            xf1 = float4(xb4[1]);
          }
          const float lsum = ((xf0.x + xf0.y) + (xf0.z + xf0.w)) +
                             ((xf1.x + xf1.y) + (xf1.z + xf1.w));
          const ulong idx = tile_base + cur_g;
          device const uchar *c_tile = r_codes + idx * 1024ul;
          const bool r_lane_v2 = (p.router_group_size >> 31u) != 0u;
          const uint c4_0 = r_lane_v2 ? omlx_m5_pi16(k_l * 2u) : (k_l * 2u);
          const uint c4_1 = r_lane_v2 ? omlx_m5_pi16(k_l * 2u + 1u) : (k_l * 2u + 1u);
          device const uint *r0_u32 = reinterpret_cast<device const uint *>(
              c_tile + ulong(rn0 + 0u) * 64ul);
          device const uint *r1_u32 = reinterpret_cast<device const uint *>(
              c_tile + ulong(rn0 + 1u) * 64ul);
          const uint2 cw0 = uint2(r0_u32[c4_0], r0_u32[c4_1]);
          const uint2 cw1 = uint2(r1_u32[c4_0], r1_u32[c4_1]);
          const uint sbp0 = sb[ulong(cur_g) * 256ul + row_sg + 0u];
          const uint sbp1 = sb[ulong(cur_g) * 256ul + row_sg + 1u];
          const float sc0 = as_type<float>((sbp0 & 0xFFFFu) << 16u);
          const float bi0 = as_type<float>(sbp0 & 0xFFFF0000u) - 1024.0f * sc0;
          const float sc1 = as_type<float>((sbp1 & 0xFFFFu) << 16u);
          const float bi1 = as_type<float>(sbp1 & 0xFFFF0000u) - 1024.0f * sc1;
          float d0 = fma(xf0.x, float(as_type<half>(ushort(0x6400u | (cw0.x & 0xFFu)))), 0.0f);
          d0 = fma(xf0.y, float(as_type<half>(ushort(0x6400u | ((cw0.x >> 8u) & 0xFFu)))), d0);
          d0 = fma(xf0.z, float(as_type<half>(ushort(0x6400u | ((cw0.x >> 16u) & 0xFFu)))), d0);
          d0 = fma(xf0.w, float(as_type<half>(ushort(0x6400u | (cw0.x >> 24u)))), d0);
          d0 = fma(xf1.x, float(as_type<half>(ushort(0x6400u | (cw0.y & 0xFFu)))), d0);
          d0 = fma(xf1.y, float(as_type<half>(ushort(0x6400u | ((cw0.y >> 8u) & 0xFFu)))), d0);
          d0 = fma(xf1.z, float(as_type<half>(ushort(0x6400u | ((cw0.y >> 16u) & 0xFFu)))), d0);
          d0 = fma(xf1.w, float(as_type<half>(ushort(0x6400u | (cw0.y >> 24u)))), d0);
          float d1 = fma(xf0.x, float(as_type<half>(ushort(0x6400u | (cw1.x & 0xFFu)))), 0.0f);
          d1 = fma(xf0.y, float(as_type<half>(ushort(0x6400u | ((cw1.x >> 8u) & 0xFFu)))), d1);
          d1 = fma(xf0.z, float(as_type<half>(ushort(0x6400u | ((cw1.x >> 16u) & 0xFFu)))), d1);
          d1 = fma(xf0.w, float(as_type<half>(ushort(0x6400u | (cw1.x >> 24u)))), d1);
          d1 = fma(xf1.x, float(as_type<half>(ushort(0x6400u | (cw1.y & 0xFFu)))), d1);
          d1 = fma(xf1.y, float(as_type<half>(ushort(0x6400u | ((cw1.y >> 8u) & 0xFFu)))), d1);
          d1 = fma(xf1.z, float(as_type<half>(ushort(0x6400u | ((cw1.y >> 16u) & 0xFFu)))), d1);
          d1 = fma(xf1.w, float(as_type<half>(ushort(0x6400u | (cw1.y >> 24u)))), d1);
          acc0 = fma(sc0, d0, fma(bi0, lsum, acc0));
          acc1 = fma(sc1, d1, fma(bi1, lsum, acc1));
        }
      }
    } else if (!FAST_Q8_PRETILED && p.router_mode == 0u) {
      device const bfloat *rw = reinterpret_cast<device const bfloat *>(r_codes);
      const uint g_sub = lane >> 3u;
      const uint k_l = lane & 7u;
      for (uint g = 0u; g < groups64; g += 4u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const uint col = cur_g * 64u + k_l * 8u;
          device const bfloat4 *xb4 = reinterpret_cast<device const bfloat4 *>(x_or_hidden + col);
          float4 xf0, xf1;
          if constexpr (PROLOGUE_NORM) {
            device const bfloat4 *wb4 = reinterpret_cast<device const bfloat4 *>(rms_w + col);
            xf0 = float4(bfloat4(float4(xb4[0]) * inv_rms * (float4(wb4[0]) + off4)));
            xf1 = float4(bfloat4(float4(xb4[1]) * inv_rms * (float4(wb4[1]) + off4)));
          } else {
            xf0 = float4(xb4[0]);
            xf1 = float4(xb4[1]);
          }
          device const bfloat4 *w0b = reinterpret_cast<device const bfloat4 *>(rw + ulong(row_sg + 0u) * p.h + col);
          device const bfloat4 *w1b = reinterpret_cast<device const bfloat4 *>(rw + ulong(row_sg + 1u) * p.h + col);
          const float4 w00 = float4(w0b[0]), w01 = float4(w0b[1]);
          const float4 w10 = float4(w1b[0]), w11 = float4(w1b[1]);
          acc0 = fma(xf0.x, w00.x, fma(xf0.y, w00.y, fma(xf0.z, w00.z, fma(xf0.w, w00.w, acc0))));
          acc0 = fma(xf1.x, w01.x, fma(xf1.y, w01.y, fma(xf1.z, w01.z, fma(xf1.w, w01.w, acc0))));
          acc1 = fma(xf0.x, w10.x, fma(xf0.y, w10.y, fma(xf0.z, w10.z, fma(xf0.w, w10.w, acc1))));
          acc1 = fma(xf1.x, w11.x, fma(xf1.y, w11.y, fma(xf1.z, w11.z, fma(xf1.w, w11.w, acc1))));
        }
      }
    } else if (!FAST_Q8_PRETILED) {
      const uint gs = ((p.router_group_size & 0x7FFFFFFFu) == 128u) ? 128u : 64u;
      const uint sb_stride = p.router_padded_k / gs;
      device const bfloat *scales = reinterpret_cast<device const bfloat *>(r_aux0);
      device const bfloat *biases = reinterpret_cast<device const bfloat *>(r_aux1);
      const uint g_sub = lane >> 4u;
      const uint k_l = lane & 15u;
      for (uint g = 0u; g < groups64; g += 2u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const uint col = cur_g * 64u + k_l * 4u;
          const bfloat4 x_in = *reinterpret_cast<device const bfloat4 *>(x_or_hidden + col);
          float4 xf;
          if constexpr (PROLOGUE_NORM) {
            const bfloat4 w_in = *reinterpret_cast<device const bfloat4 *>(rms_w + col);
            xf = float4(bfloat4(float4(x_in) * inv_rms * (float4(w_in) + off4)));
          } else {
            xf = float4(x_in);
          }
          const float lsum = (xf.x + xf.y) + (xf.z + xf.w);
          const uint g_src = (gs == 128u) ? (cur_g >> 1u) : cur_g;
          const uchar4 cb0 = reinterpret_cast<device const uchar4 *>(
              r_codes + ulong(row_sg + 0u) * p.router_padded_k + cur_g * 64u)[k_l];
          const uchar4 cb1 = reinterpret_cast<device const uchar4 *>(
              r_codes + ulong(row_sg + 1u) * p.router_padded_k + cur_g * 64u)[k_l];
          const float sc0 = float(scales[ulong(row_sg + 0u) * sb_stride + g_src]);
          const float bi0 = float(biases[ulong(row_sg + 0u) * sb_stride + g_src]) - 1024.0f * sc0;
          const float sc1 = float(scales[ulong(row_sg + 1u) * sb_stride + g_src]);
          const float bi1 = float(biases[ulong(row_sg + 1u) * sb_stride + g_src]) - 1024.0f * sc1;
          float d0 = fma(xf.x, float(as_type<half>(ushort(0x6400u | cb0.x))), 0.0f);
          d0 = fma(xf.y, float(as_type<half>(ushort(0x6400u | cb0.y))), d0);
          d0 = fma(xf.z, float(as_type<half>(ushort(0x6400u | cb0.z))), d0);
          d0 = fma(xf.w, float(as_type<half>(ushort(0x6400u | cb0.w))), d0);
          float d1 = fma(xf.x, float(as_type<half>(ushort(0x6400u | cb1.x))), 0.0f);
          d1 = fma(xf.y, float(as_type<half>(ushort(0x6400u | cb1.y))), d1);
          d1 = fma(xf.z, float(as_type<half>(ushort(0x6400u | cb1.z))), d1);
          d1 = fma(xf.w, float(as_type<half>(ushort(0x6400u | cb1.w))), d1);
          acc0 = fma(sc0, d0, fma(bi0, lsum, acc0));
          acc1 = fma(sc1, d1, fma(bi1, lsum, acc1));
        }
      }
    }

    const float v0 = simd_sum(acc0);
    const float v1 = simd_sum(acc1);
    if (lane == 0u) {
      const uint lo16 = uint(as_type<ushort>(bfloat(v0)));
      const uint hi16 = uint(as_type<ushort>(bfloat(v1)));
      atomic_store_explicit(&router_logits_u32[tg.x * 2u + sg],
                            lo16 | (hi16 << 16u), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);
    uint is_last = 0u;
    if (sg == 0u && lane == 0u) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                          thread_scope_device);
      if (atomic_fetch_add_explicit(&moe_tg_done_buf[p.moe_slot], 1u,
                                    memory_order_acq_rel,
                                    mem_flags::mem_device) == 63u) {
        is_last = 1u;
      }
    }
    if (sg == 0u) {
      is_last = simd_broadcast_first(is_last);
      if (is_last != 0u) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                            thread_scope_device);
        if (lane == 0u) {
          atomic_store_explicit(&moe_tg_done_buf[p.moe_slot], 0u,
                                memory_order_relaxed);
        }
        moe9_top8_from_atomic_u32(router_logits_u32, route_ids_out,
                                  route_scores_out, lane);
      }
    }
    return;
  }

  // Concurrent tg.x == 64: computes shared_gate_out[0]
  if (p.sgate_mode != 0u) {
    float sg_acc = 0.0f;
    const uint g_sub = sg * 2u + (lane >> 4u);
    const uint k_l = lane & 15u;
    if (FAST_Q8_PRETILED || p.sgate_mode == 2u) {
      device const uint *sg_sb = reinterpret_cast<device const uint *>(sg_aux0);
      const uint pad_n = max(p.sgate_padded_n, 16u);
      for (uint g = 0u; g < groups64; g += 4u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const uint col = cur_g * 64u + k_l * 4u;
          const bfloat4 x_in = *reinterpret_cast<device const bfloat4 *>(x_or_hidden + col);
          float4 xf;
          if constexpr (PROLOGUE_NORM) {
            const bfloat4 w_in = *reinterpret_cast<device const bfloat4 *>(rms_w + col);
            xf = float4(bfloat4(float4(x_in) * inv_rms * (float4(w_in) + off4)));
          } else {
            xf = float4(x_in);
          }
          const float lsum = (xf.x + xf.y) + (xf.z + xf.w);
          const uint sg_c4 = ((p.sgate_group_size >> 31u) != 0u) ? omlx_m5_pi16(k_l) : k_l;
          const uchar4 cb = reinterpret_cast<device const uchar4 *>(
              sg_codes + ulong(cur_g) * 1024ul)[sg_c4];
          const uint sbp = sg_sb[ulong(cur_g) * pad_n];
          const float sc = as_type<float>((sbp & 0xFFFFu) << 16u);
          const float bi = as_type<float>(sbp & 0xFFFF0000u) - 1024.0f * sc;
          float d = fma(xf.x, float(as_type<half>(ushort(0x6400u | cb.x))), 0.0f);
          d = fma(xf.y, float(as_type<half>(ushort(0x6400u | cb.y))), d);
          d = fma(xf.z, float(as_type<half>(ushort(0x6400u | cb.z))), d);
          d = fma(xf.w, float(as_type<half>(ushort(0x6400u | cb.w))), d);
          sg_acc = fma(sc, d, fma(bi, lsum, sg_acc));
        }
      }
    } else if (!FAST_Q8_PRETILED && p.sgate_mode == 1u) {
      device const bfloat *sgw = reinterpret_cast<device const bfloat *>(sg_codes);
      for (uint g = 0u; g < groups64; g += 4u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const uint col = cur_g * 64u + k_l * 4u;
          const bfloat4 x_in = *reinterpret_cast<device const bfloat4 *>(x_or_hidden + col);
          float4 xf;
          if constexpr (PROLOGUE_NORM) {
            const bfloat4 w_in = *reinterpret_cast<device const bfloat4 *>(rms_w + col);
            xf = float4(bfloat4(float4(x_in) * inv_rms * (float4(w_in) + off4)));
          } else {
            xf = float4(x_in);
          }
          const float4 wf = float4(*reinterpret_cast<device const bfloat4 *>(sgw + col));
          sg_acc = fma(xf.x, wf.x, fma(xf.y, wf.y, fma(xf.z, wf.z, fma(xf.w, wf.w, sg_acc))));
        }
      }
    } else if (!FAST_Q8_PRETILED && p.sgate_mode == 3u) {
      const uint gs_s = ((p.sgate_group_size & 0x7FFFFFFFu) == 128u) ? 128u : 64u;
      device const bfloat *scales = reinterpret_cast<device const bfloat *>(sg_aux0);
      device const bfloat *biases = reinterpret_cast<device const bfloat *>(sg_aux1);
      for (uint g = 0u; g < groups64; g += 4u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const uint col = cur_g * 64u + k_l * 4u;
          const bfloat4 x_in = *reinterpret_cast<device const bfloat4 *>(x_or_hidden + col);
          float4 xf;
          if constexpr (PROLOGUE_NORM) {
            const bfloat4 w_in = *reinterpret_cast<device const bfloat4 *>(rms_w + col);
            xf = float4(bfloat4(float4(x_in) * inv_rms * (float4(w_in) + off4)));
          } else {
            xf = float4(x_in);
          }
          const float lsum = (xf.x + xf.y) + (xf.z + xf.w);
          const uchar4 cb = reinterpret_cast<device const uchar4 *>(sg_codes + col)[0];
          const uint g_src = (gs_s == 128u) ? (cur_g >> 1u) : cur_g;
          const float sc = float(scales[g_src]);
          const float bi = float(biases[g_src]) - 1024.0f * sc;
          float d = fma(xf.x, float(as_type<half>(ushort(0x6400u | cb.x))), 0.0f);
          d = fma(xf.y, float(as_type<half>(ushort(0x6400u | cb.y))), d);
          d = fma(xf.z, float(as_type<half>(ushort(0x6400u | cb.z))), d);
          d = fma(xf.w, float(as_type<half>(ushort(0x6400u | cb.w))), d);
          sg_acc = fma(sc, d, fma(bi, lsum, sg_acc));
        }
      }
    }
    sg_acc = simd_sum(sg_acc);
    if (lane == 0u) {
      tg_sg[sg] = sg_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      shared_gate_out[0] = bfloat(tg_sg[0] + tg_sg[1]);
    }
  }
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_moe9_router_q8_pretiled_fused_top8_t1(
    device const bfloat *norm [[buffer(0)]],
    device const uchar *r_codes [[buffer(1)]],
    device const uchar *r_aux0 [[buffer(2)]],
    device const uchar *r_aux1 [[buffer(3)]],
    device const uchar *sg_codes [[buffer(4)]],
    device const uchar *sg_aux0 [[buffer(5)]],
    device const uchar *sg_aux1 [[buffer(6)]],
    device atomic_uint *router_logits_u32 [[buffer(7)]],
    device uint *route_ids_out [[buffer(8)]],
    device bfloat *route_scores_out [[buffer(9)]],
    device bfloat *shared_gate_out [[buffer(10)]],
    device atomic_uint *moe_tg_done_buf [[buffer(11)]],
    constant Moe9RouterFusedTop8Params &p [[buffer(12)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  (void)r_aux1;
  (void)sg_aux1;
  threadgroup float tg_sg[2];
  moe9_router_fused_top8_impl<false, true>(
      norm, nullptr, nullptr, nullptr, r_codes, r_aux0, r_aux1, sg_codes,
      sg_aux0, sg_aux1, router_logits_u32, route_ids_out, route_scores_out,
      shared_gate_out, moe_tg_done_buf, tg_sg, p, tid, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_moe9_router_qmv_fused_top8_t1(
    device const bfloat *norm [[buffer(0)]],
    device const uchar *r_codes [[buffer(1)]],
    device const uchar *r_aux0 [[buffer(2)]],
    device const uchar *r_aux1 [[buffer(3)]],
    device const uchar *sg_codes [[buffer(4)]],
    device const uchar *sg_aux0 [[buffer(5)]],
    device const uchar *sg_aux1 [[buffer(6)]],
    device atomic_uint *router_logits_u32 [[buffer(7)]],
    device uint *route_ids_out [[buffer(8)]],
    device bfloat *route_scores_out [[buffer(9)]],
    device bfloat *shared_gate_out [[buffer(10)]],
    device atomic_uint *moe_tg_done_buf [[buffer(11)]],
    constant Moe9RouterFusedTop8Params &p [[buffer(12)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float tg_sg[2];
  moe9_router_fused_top8_impl<false, false>(
      norm, nullptr, nullptr, nullptr, r_codes, r_aux0, r_aux1, sg_codes,
      sg_aux0, sg_aux1, router_logits_u32, route_ids_out, route_scores_out,
      shared_gate_out, moe_tg_done_buf, tg_sg, p, tid, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_moe9_router_q8_pretiled_norm_fused_top8_t1(
    device const bfloat *hidden [[buffer(0)]],
    device const bfloat *rms_w [[buffer(1)]],
    device const float *rms_sq_buf [[buffer(2)]],
    device bfloat *norm_out [[buffer(3)]],
    device const uchar *r_codes [[buffer(4)]],
    device const uchar *r_aux0 [[buffer(5)]],
    device const uchar *r_aux1 [[buffer(6)]],
    device const uchar *sg_codes [[buffer(7)]],
    device const uchar *sg_aux0 [[buffer(8)]],
    device const uchar *sg_aux1 [[buffer(9)]],
    device atomic_uint *router_logits_u32 [[buffer(10)]],
    device uint *route_ids_out [[buffer(11)]],
    device bfloat *route_scores_out [[buffer(12)]],
    device bfloat *shared_gate_out [[buffer(13)]],
    device atomic_uint *moe_tg_done_buf [[buffer(14)]],
    constant Moe9RouterFusedTop8Params &p [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  (void)r_aux1;
  (void)sg_aux1;
  threadgroup float tg_sg[2];
  moe9_router_fused_top8_impl<true, true>(
      hidden, rms_w, rms_sq_buf, norm_out, r_codes, r_aux0, r_aux1, sg_codes,
      sg_aux0, sg_aux1, router_logits_u32, route_ids_out, route_scores_out,
      shared_gate_out, moe_tg_done_buf, tg_sg, p, tid, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_moe9_router_qmv_norm_fused_top8_t1(
    device const bfloat *hidden [[buffer(0)]],
    device const bfloat *rms_w [[buffer(1)]],
    device const float *rms_sq_buf [[buffer(2)]],
    device bfloat *norm_out [[buffer(3)]],
    device const uchar *r_codes [[buffer(4)]],
    device const uchar *r_aux0 [[buffer(5)]],
    device const uchar *r_aux1 [[buffer(6)]],
    device const uchar *sg_codes [[buffer(7)]],
    device const uchar *sg_aux0 [[buffer(8)]],
    device const uchar *sg_aux1 [[buffer(9)]],
    device atomic_uint *router_logits_u32 [[buffer(10)]],
    device uint *route_ids_out [[buffer(11)]],
    device bfloat *route_scores_out [[buffer(12)]],
    device bfloat *shared_gate_out [[buffer(13)]],
    device atomic_uint *moe_tg_done_buf [[buffer(14)]],
    constant Moe9RouterFusedTop8Params &p [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float tg_sg[2];
  moe9_router_fused_top8_impl<true, false>(
      hidden, rms_w, rms_sq_buf, norm_out, r_codes, r_aux0, r_aux1, sg_codes,
      sg_aux0, sg_aux1, router_logits_u32, route_ids_out, route_scores_out,
      shared_gate_out, moe_tg_done_buf, tg_sg, p, tid, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(96)]]
kernel void omlx_m5_moe9_mxfp4_kl16_down_sk3_r8_add_sq_t1(
    device const bfloat *mid_in [[buffer(0)]],
    device const uint *route_ids [[buffer(1)]],
    device const bfloat *route_scores [[buffer(2)]],
    device const bfloat *shared_gate_buf [[buffer(3)]],
    device const bfloat *norm_in [[buffer(4)]],
    device const uchar *routed_dc [[buffer(5)]],
    device const uchar *routed_ds [[buffer(6)]],
    device const uchar *shared_dc [[buffer(7)]],
    device const uchar *shared_ds [[buffer(8)]],
    device bfloat *hidden_inout [[buffer(9)]],
    device atomic_float *rms_sq_buf [[buffer(10)]],
    constant Moe9GemmParams &p [[buffer(11)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row_base = tg.x * 8u;
  if (row_base >= p.h || sg >= 3u) {
    return;
  }

  threadgroup float sg_gate_part[3];
  threadgroup float tg_red[3][8];
  threadgroup float tg_shared[8];

  float gate_logit = 0.0f;
  if ((p.flags & 1u) != 0u) {
    const uint tid = sg * 32u + lane;
    const uint h4 = p.h >> 2u;
    device const bfloat4 *gw4 = reinterpret_cast<device const bfloat4 *>(shared_gate_buf);
    device const bfloat4 *xn4 = reinterpret_cast<device const bfloat4 *>(norm_in);
    float g_dot = 0.0f;
    for (uint i = tid; i < h4; i += 96u) {
      g_dot += dot(float4(gw4[i]), float4(xn4[i]));
    }
    g_dot = simd_sum(g_dot);
    if (lane == 0u) sg_gate_part[sg] = g_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gate_logit = float(bfloat(sg_gate_part[0] + sg_gate_part[1] + sg_gate_part[2]));
  } else {
    gate_logit = float(shared_gate_buf[0]);
  }

  const uint b_sub = lane >> 4u;
  const uint k_l = lane & 15u;
  const uint sub32 = k_l >> 2u;
  const uint blocks = p.f >> 7u;

  const ulong r_c_stride = ulong(p.routed_padded_k >> 1u);
  const ulong r_s_stride = ulong(p.routed_padded_k >> 5u);
  const ulong s_c_stride = ulong(p.shared_padded_k >> 1u);
  const ulong s_s_stride = ulong(p.shared_padded_k >> 5u);

  float partial_routed[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float shared_vals[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  const uint s0 = sg * 3u;
  for (uint slot = s0; slot < s0 + 3u; ++slot) {
    if (slot > p.top_k) continue;

    uint exp_id = 0u;
    float w_score = 1.0f;
    if (slot < p.top_k) {
      exp_id = route_ids[slot];
      w_score = float(route_scores[slot]);
      if (exp_id >= p.num_experts || w_score == 0.0f) {
        continue;
      }
    }

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    device const bfloat *xm = mid_in + ulong(slot) * p.f;

    const bool do_fwht =
        (slot < p.top_k) ? ((p.flags & 4u) != 0u) : ((p.flags & 8u) != 0u);
    for (uint blk = 0u; blk < blocks; blk += 2u) {
      const uint cur_blk = blk + b_sub;
      float4 xf0 = float4(0.0f);
      float4 xf1 = float4(0.0f);
      if (cur_blk < blocks) {
        device const bfloat4 *xb =
            reinterpret_cast<device const bfloat4 *>(xm + cur_blk * 128u + k_l * 8u);
        xf0 = float4(xb[0]);
        xf1 = float4(xb[1]);
      }
      if (do_fwht) {
        omlx_m5_fwht64_sub8(xf0, xf1, lane);
      }
      if (cur_blk < blocks) {
        const half4 h0 = half4(xf0);
        const half4 h1 = half4(xf1);

        if (slot < p.top_k) {
          const ulong exp_row0 = ulong(exp_id) * p.h + row_base;
#pragma clang loop unroll(full)
          for (uint r = 0u; r < 8u; ++r) {
            if (row_base + r < p.h) {
              const ulong row_idx = exp_row0 + r;
              const uint w = reinterpret_cast<device const uint *>(
                  routed_dc + row_idx * r_c_stride + cur_blk * 64ul)[k_l];
              const float sc = e8m0_to_float(
                  (routed_ds + row_idx * r_s_stride + cur_blk * 4ul)[sub32]);
              const half d =
                  h0.x * moe9_fp4_to_h16(w) + h0.y * moe9_fp4_to_h16(w >> 4u) +
                  h0.z * moe9_fp4_to_h16(w >> 8u) + h0.w * moe9_fp4_to_h16(w >> 12u) +
                  h1.x * moe9_fp4_to_h16(w >> 16u) + h1.y * moe9_fp4_to_h16(w >> 20u) +
                  h1.z * moe9_fp4_to_h16(w >> 24u) + h1.w * moe9_fp4_to_h16(w >> 28u);
              acc[r] = fma(float(d), sc, acc[r]);
            }
          }
        } else if (p.shared_tiled != 0u) {
          const uint tile_n = row_base >> 4u;
          const uint rn0 = row_base & 15u;
          const ulong idx = ulong(tile_n) * blocks + cur_blk;
#pragma clang loop unroll(full)
          for (uint r = 0u; r < 8u; ++r) {
            if (row_base + r < p.h) {
              const uint rn = rn0 + r;
              const uint w = reinterpret_cast<device const uint *>(
                  shared_dc + idx * 1024ul + rn * 64ul)[k_l];
              const float sc = e8m0_to_float(
                  (shared_ds + idx * 64ul + rn * 4ul)[sub32]);
              const half d =
                  h0.x * moe9_fp4_to_h16(w) + h0.y * moe9_fp4_to_h16(w >> 4u) +
                  h0.z * moe9_fp4_to_h16(w >> 8u) + h0.w * moe9_fp4_to_h16(w >> 12u) +
                  h1.x * moe9_fp4_to_h16(w >> 16u) + h1.y * moe9_fp4_to_h16(w >> 20u) +
                  h1.z * moe9_fp4_to_h16(w >> 24u) + h1.w * moe9_fp4_to_h16(w >> 28u);
              acc[r] = fma(float(d), sc, acc[r]);
            }
          }
        } else {
#pragma clang loop unroll(full)
          for (uint r = 0u; r < 8u; ++r) {
            const uint row = row_base + r;
            if (row < p.h) {
              const uint w = reinterpret_cast<device const uint *>(
                  shared_dc + ulong(row) * s_c_stride + cur_blk * 64ul)[k_l];
              const float sc = e8m0_to_float(
                  (shared_ds + ulong(row) * s_s_stride + cur_blk * 4ul)[sub32]);
              const half d =
                  h0.x * moe9_fp4_to_h16(w) + h0.y * moe9_fp4_to_h16(w >> 4u) +
                  h0.z * moe9_fp4_to_h16(w >> 8u) + h0.w * moe9_fp4_to_h16(w >> 12u) +
                  h1.x * moe9_fp4_to_h16(w >> 16u) + h1.y * moe9_fp4_to_h16(w >> 20u) +
                  h1.z * moe9_fp4_to_h16(w >> 24u) + h1.w * moe9_fp4_to_h16(w >> 28u);
              acc[r] = fma(float(d), sc, acc[r]);
            }
          }
        }
      }
    }

#pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; ++r) {
      const float s = simd_sum(acc[r]);
      if (lane == 0u) {
        if (slot < p.top_k) {
          partial_routed[r] += float(bfloat(s * w_score));
        } else {
          shared_vals[r] = float(bfloat(s));
        }
      }
    }
  }

  if (lane == 0u) {
#pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; ++r) {
      tg_red[sg][r] = partial_routed[r];
      if (sg == 2u) {
        tg_shared[r] = shared_vals[r];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u) {
    float sq_lane = 0.0f;
    if (lane < 8u && row_base + lane < p.h) {
      const uint col = row_base + lane;
      const bfloat routed_bf16 =
          bfloat(tg_red[0][lane] + tg_red[1][lane] + tg_red[2][lane]);
      const float shared_scale = 1.0f / (1.0f + exp(-gate_logit));
      const bfloat delta_bf =
          bfloat(float(routed_bf16) + shared_scale * tg_shared[lane]);
      const bfloat h_new = bfloat(float(hidden_inout[col]) + float(delta_bf));
      hidden_inout[col] = h_new;
      const float hf = float(h_new);
      sq_lane = hf * hf;
    }
    const float tg_sq = simd_sum(sq_lane);
    if (lane == 0u) {
      atomic_fetch_add_explicit(&rms_sq_buf[p.flags >> 8u], tg_sq,
                                memory_order_relaxed);
    }
  }
}

// ===========================================================================
// v0.8.0-m5rt.14 Zero-Spill Batched 9-Expert MoE (B in [2, 8], 3 dispatches/L)
//   Dispatch 1: omlx_m5_moe9_router_{q8_pretiled,qmv}[_norm]_fused_top8_bt
//               grid=(65*64, B, 1), tg=(64, 1, 1), self-resetting moe_tg_done_buf
//   Dispatch 2: omlx_m5_moe9_mxfp4_kl16_swiglu_bt
//               grid=((F+3)/4*64, 9*B, 1), tg=(64, 1, 1), 4 KiB tg_x cooperative load
//   Dispatch 3: omlx_m5_moe9_mxfp4_kl16_down_sk3_r8[_add_sq]_bt
//               grid=((H+7)/8*96, B, 1), tg=(96, 1, 1), scalar registers (0 alloca)
// ===========================================================================

struct Moe9RouterFusedTop8BtParams {
  uint h;
  uint router_padded_k;
  uint router_group_size;
  uint router_mode;
  uint sgate_padded_k;
  uint sgate_group_size;
  uint sgate_padded_n;
  uint sgate_mode;
  uint sq_slot;
  uint moe_slot;
  float inv_h;
  float eps;
  float w_off;
  uint rows;
};

struct Moe9GemmBtParams {
  uint f;               // intermediate size (e.g. 512, multiple of 16)
  uint h;               // hidden size (e.g. 2048, multiple of 128)
  uint routed_padded_k; // gw.padded_k (Up) or dw.padded_k (Down)
  uint shared_padded_k; // shared_gw.padded_k (Up) or shared_dw.padded_k (Down) when shared_tiled == 0
  uint shared_tiled;    // 1 = mx_nax_tiled_bk128_v1 (BN=16), 0 = mx_rowsplit_k128_v1
  uint num_experts;     // 256
  uint top_k;           // 8
  uint flags;           // Down flags: bit 0 = inline BF16 shared_gate dot, bit 1 = add residual; add_sq: bits 8..31 = sq_slot
  uint rows;            // B in [1, 8]
};

inline half moe9_dot8_fp4_h16(half4 h0, half4 h1, uint w) {
  return h0.x * moe9_fp4_to_h16(w) + h0.y * moe9_fp4_to_h16(w >> 4u) +
         h0.z * moe9_fp4_to_h16(w >> 8u) + h0.w * moe9_fp4_to_h16(w >> 12u) +
         h1.x * moe9_fp4_to_h16(w >> 16u) + h1.y * moe9_fp4_to_h16(w >> 20u) +
         h1.z * moe9_fp4_to_h16(w >> 24u) + h1.w * moe9_fp4_to_h16(w >> 28u);
}

inline float moe9_mxfp4_row_dot_acc(
    device const uchar *codes,
    device const uchar *scales,
    ulong c_off,
    ulong s_off,
    uint k_l,
    uint sub32,
    half4 h0,
    half4 h1,
    float acc) {
  const uint w = reinterpret_cast<device const uint *>(codes + c_off)[k_l];
  const float sc = e8m0_to_float((scales + s_off)[sub32]);
  const half d = moe9_dot8_fp4_h16(h0, h1, w);
  return fma(float(d), sc, acc);
}

inline uint moe9_bf16_sort_key(ushort raw, uint expert_id) {
  const uint bf = ((raw & 0x7FFFu) == 0u) ? 0u : uint(raw);
  const uint key16 = (bf & 0x8000u) ? ((~bf) & 0xFFFFu) : (bf | 0x8000u);
  return (key16 << 16u) | 0x0100u | expert_id;
}

__attribute__((always_inline))
inline void moe9_fast_pick_one(
    thread uint &p0, thread uint &p1, thread uint &p2, thread uint &p3,
    thread uint &p4, thread uint &p5, thread uint &p6, thread uint &p7,
    uint base,
    thread float &out_sel,
    thread uint &out_id) {
  const uint best_p = max(max(max(p0, p1), max(p2, p3)),
                          max(max(p4, p5), max(p6, p7)));
  const uint win_p = simd_max(best_p);
  const uint winner = win_p & 0xFFu;
  const uint win_key16 = win_p >> 16u;
  const uint orig_bf = (win_key16 & 0x8000u) ? (win_key16 & 0x7FFFu)
                                             : ((~win_key16) & 0xFFFFu);
  out_sel = as_type<float>(orig_bf << 16u);
  out_id = winner;
  if (winner == base + 0u) p0 = 0u;
  if (winner == base + 1u) p1 = 0u;
  if (winner == base + 2u) p2 = 0u;
  if (winner == base + 3u) p3 = 0u;
  if (winner == base + 4u) p4 = 0u;
  if (winner == base + 5u) p5 = 0u;
  if (winner == base + 6u) p6 = 0u;
  if (winner == base + 7u) p7 = 0u;
}

__attribute__((always_inline))
inline void moe9_nan_cand(
    float val,
    uint idx,
    uint taken_mask,
    thread int &tier,
    thread float &best,
    thread uint &best_local) {
  if ((taken_mask & (1u << idx)) != 0u) {
    return;
  }
  const int ct = isnan(val) ? 0 : 1;
  const float cv = (ct == 1) ? val : -INFINITY;
  if (ct > tier || (ct == tier && cv >= best)) {
    tier = ct;
    best = cv;
    best_local = idx;
  }
}

__attribute__((always_inline))
inline void moe9_nan_pick_one(
    float v0, float v1, float v2, float v3,
    float v4, float v5, float v6, float v7,
    thread uint &taken_mask,
    uint lane,
    thread float &out_sel,
    thread uint &out_id) {
  int tier = -1;
  float best = -INFINITY;
  uint best_local = 0u;
  moe9_nan_cand(v0, 0u, taken_mask, tier, best, best_local);
  moe9_nan_cand(v1, 1u, taken_mask, tier, best, best_local);
  moe9_nan_cand(v2, 2u, taken_mask, tier, best, best_local);
  moe9_nan_cand(v3, 3u, taken_mask, tier, best, best_local);
  moe9_nan_cand(v4, 4u, taken_mask, tier, best, best_local);
  moe9_nan_cand(v5, 5u, taken_mask, tier, best, best_local);
  moe9_nan_cand(v6, 6u, taken_mask, tier, best, best_local);
  moe9_nan_cand(v7, 7u, taken_mask, tier, best, best_local);
  const int global_tier = simd_max(tier);
  if (global_tier < 0) {
    out_sel = -INFINITY;
    out_id = 0u;
    return;
  }
  const bool in_tier = (tier == global_tier);
  const float maximum = simd_max(in_tier ? best : -INFINITY);
  const bool is_best = in_tier && (best == maximum);
  const uint id = lane * 8u + best_local;
  const uint winner = simd_max(is_best ? id : 0u);
  if (is_best && id == winner) {
    taken_mask |= (1u << best_local);
  }
  out_sel = (global_tier == 1) ? maximum : -INFINITY;
  out_id = winner;
}

inline float moe9_softmax_prob(float val, float max_logit) {
  if (max_logit == -INFINITY || isnan(max_logit) || isnan(val)) {
    return 0.0f;
  }
  if (max_logit == INFINITY) {
    return val == INFINITY ? 1.0f : 0.0f;
  }
  return val == -INFINITY ? 0.0f : exp(val - max_logit);
}

// Zero-alloca warp-register top-8 selection + softmax matching moe9_top8_from_atomic_u32 bit-for-bit.
inline void moe9_top8_from_atomic_u32_nospill(
    device atomic_uint *logits_u32,
    device uint *indices,
    device bfloat *scores,
    uint lane) {
  const uint u0 = atomic_load_explicit(&logits_u32[lane * 4u + 0u], memory_order_relaxed);
  const uint u1 = atomic_load_explicit(&logits_u32[lane * 4u + 1u], memory_order_relaxed);
  const uint u2 = atomic_load_explicit(&logits_u32[lane * 4u + 2u], memory_order_relaxed);
  const uint u3 = atomic_load_explicit(&logits_u32[lane * 4u + 3u], memory_order_relaxed);
  const ushort r0 = ushort(u0 & 0xFFFFu), r1 = ushort(u0 >> 16u);
  const ushort r2 = ushort(u1 & 0xFFFFu), r3 = ushort(u1 >> 16u);
  const ushort r4 = ushort(u2 & 0xFFFFu), r5 = ushort(u2 >> 16u);
  const ushort r6 = ushort(u3 & 0xFFFFu), r7 = ushort(u3 >> 16u);

  const bool local_nan =
      ((r0 & 0x7FFFu) > 0x7F80u) || ((r1 & 0x7FFFu) > 0x7F80u) ||
      ((r2 & 0x7FFFu) > 0x7F80u) || ((r3 & 0x7FFFu) > 0x7F80u) ||
      ((r4 & 0x7FFFu) > 0x7F80u) || ((r5 & 0x7FFFu) > 0x7F80u) ||
      ((r6 & 0x7FFFu) > 0x7F80u) || ((r7 & 0x7FFFu) > 0x7F80u);

  float sel0 = 0.0f, sel1 = 0.0f, sel2 = 0.0f, sel3 = 0.0f;
  float sel4 = 0.0f, sel5 = 0.0f, sel6 = 0.0f, sel7 = 0.0f;
  uint id0 = 0u, id1 = 0u, id2 = 0u, id3 = 0u;
  uint id4 = 0u, id5 = 0u, id6 = 0u, id7 = 0u;

  if (!simd_any(local_nan)) {
    const uint base = lane * 8u;
    uint p0 = moe9_bf16_sort_key(r0, base + 0u);
    uint p1 = moe9_bf16_sort_key(r1, base + 1u);
    uint p2 = moe9_bf16_sort_key(r2, base + 2u);
    uint p3 = moe9_bf16_sort_key(r3, base + 3u);
    uint p4 = moe9_bf16_sort_key(r4, base + 4u);
    uint p5 = moe9_bf16_sort_key(r5, base + 5u);
    uint p6 = moe9_bf16_sort_key(r6, base + 6u);
    uint p7 = moe9_bf16_sort_key(r7, base + 7u);
    moe9_fast_pick_one(p0, p1, p2, p3, p4, p5, p6, p7, base, sel0, id0);
    moe9_fast_pick_one(p0, p1, p2, p3, p4, p5, p6, p7, base, sel1, id1);
    moe9_fast_pick_one(p0, p1, p2, p3, p4, p5, p6, p7, base, sel2, id2);
    moe9_fast_pick_one(p0, p1, p2, p3, p4, p5, p6, p7, base, sel3, id3);
    moe9_fast_pick_one(p0, p1, p2, p3, p4, p5, p6, p7, base, sel4, id4);
    moe9_fast_pick_one(p0, p1, p2, p3, p4, p5, p6, p7, base, sel5, id5);
    moe9_fast_pick_one(p0, p1, p2, p3, p4, p5, p6, p7, base, sel6, id6);
    moe9_fast_pick_one(p0, p1, p2, p3, p4, p5, p6, p7, base, sel7, id7);
  } else {
    const float v0 = as_type<float>(uint(r0) << 16u);
    const float v1 = as_type<float>(uint(r1) << 16u);
    const float v2 = as_type<float>(uint(r2) << 16u);
    const float v3 = as_type<float>(uint(r3) << 16u);
    const float v4 = as_type<float>(uint(r4) << 16u);
    const float v5 = as_type<float>(uint(r5) << 16u);
    const float v6 = as_type<float>(uint(r6) << 16u);
    const float v7 = as_type<float>(uint(r7) << 16u);
    uint taken_mask = 0u;
    moe9_nan_pick_one(v0, v1, v2, v3, v4, v5, v6, v7, taken_mask, lane, sel0, id0);
    moe9_nan_pick_one(v0, v1, v2, v3, v4, v5, v6, v7, taken_mask, lane, sel1, id1);
    moe9_nan_pick_one(v0, v1, v2, v3, v4, v5, v6, v7, taken_mask, lane, sel2, id2);
    moe9_nan_pick_one(v0, v1, v2, v3, v4, v5, v6, v7, taken_mask, lane, sel3, id3);
    moe9_nan_pick_one(v0, v1, v2, v3, v4, v5, v6, v7, taken_mask, lane, sel4, id4);
    moe9_nan_pick_one(v0, v1, v2, v3, v4, v5, v6, v7, taken_mask, lane, sel5, id5);
    moe9_nan_pick_one(v0, v1, v2, v3, v4, v5, v6, v7, taken_mask, lane, sel6, id6);
    moe9_nan_pick_one(v0, v1, v2, v3, v4, v5, v6, v7, taken_mask, lane, sel7, id7);
  }

  if (lane == 0u) {
    const float max_logit = sel0;
    const float q0 = moe9_softmax_prob(sel0, max_logit);
    const float q1 = moe9_softmax_prob(sel1, max_logit);
    const float q2 = moe9_softmax_prob(sel2, max_logit);
    const float q3 = moe9_softmax_prob(sel3, max_logit);
    const float q4 = moe9_softmax_prob(sel4, max_logit);
    const float q5 = moe9_softmax_prob(sel5, max_logit);
    const float q6 = moe9_softmax_prob(sel6, max_logit);
    const float q7 = moe9_softmax_prob(sel7, max_logit);
    const float sum = ((((((q0 + q1) + q2) + q3) + q4) + q5) + q6) + q7;
    const bool ok = (sum > 0.0f && isfinite(sum));
    indices[0] = id0; scores[0] = ok ? bfloat(q0 / sum) : bfloat(0.0f);
    indices[1] = id1; scores[1] = ok ? bfloat(q1 / sum) : bfloat(0.0f);
    indices[2] = id2; scores[2] = ok ? bfloat(q2 / sum) : bfloat(0.0f);
    indices[3] = id3; scores[3] = ok ? bfloat(q3 / sum) : bfloat(0.0f);
    indices[4] = id4; scores[4] = ok ? bfloat(q4 / sum) : bfloat(0.0f);
    indices[5] = id5; scores[5] = ok ? bfloat(q5 / sum) : bfloat(0.0f);
    indices[6] = id6; scores[6] = ok ? bfloat(q6 / sum) : bfloat(0.0f);
    indices[7] = id7; scores[7] = ok ? bfloat(q7 / sum) : bfloat(0.0f);
  }
}

template <bool PROLOGUE_NORM, bool FAST_Q8_PRETILED>
inline void moe9_router_fused_top8_bt_impl(
    device const bfloat *x_or_hidden,
    device const bfloat *rms_w,
    device const float *rms_sq_buf,
    device bfloat *norm_out,
    device const uchar *r_codes,
    device const uchar *r_aux0,
    device const uchar *r_aux1,
    device const uchar *sg_codes,
    device const uchar *sg_aux0,
    device const uchar *sg_aux1,
    device atomic_uint *router_logits_u32,
    device uint *route_ids_out,
    device bfloat *route_scores_out,
    device bfloat *shared_gate_out,
    device atomic_uint *moe_tg_done_buf,
    threadgroup float (&tg_sg)[2],
    constant Moe9RouterFusedTop8BtParams &p,
    uint tid,
    uint lane,
    uint sg,
    uint3 tg) {
  const uint b = tg.y;
  if (b >= p.rows) {
    return;
  }
  if constexpr (FAST_Q8_PRETILED) {
    (void)r_aux1;
    (void)sg_aux1;
  }

  device const bfloat *x_row = x_or_hidden + ulong(b) * p.h;
  float inv_rms = 1.0f;
  float4 off4 = float4(0.0f);
  if constexpr (PROLOGUE_NORM) {
    inv_rms = rsqrt(rms_sq_buf[p.sq_slot * 8u + b] * p.inv_h + p.eps);
    off4 = float4(p.w_off);
    const uint norm_col4 = tg.x * 64u + tid;
    if (norm_col4 < (p.h >> 2u)) {
      device const bfloat4 *hv4 = reinterpret_cast<device const bfloat4 *>(x_row);
      device const bfloat4 *wv4 = reinterpret_cast<device const bfloat4 *>(rms_w);
      device bfloat4 *nv4 =
          reinterpret_cast<device bfloat4 *>(norm_out + ulong(b) * p.h);
      nv4[norm_col4] =
          bfloat4(float4(hv4[norm_col4]) * inv_rms * (float4(wv4[norm_col4]) + off4));
    }
  } else {
    (void)rms_w;
    (void)rms_sq_buf;
    (void)norm_out;
  }

  const uint groups64 = p.h >> 6u;
  if (tg.x < 64u) {
    const uint row_sg = (tg.x * 2u + sg) * 2u;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    if (FAST_Q8_PRETILED || p.router_mode == 1u) {
      device const uint *sb = reinterpret_cast<device const uint *>(r_aux0);
      const uint tile_n = row_sg >> 4u;
      const uint rn0 = row_sg & 15u;
      const ulong tile_base = ulong(tile_n) * groups64;
      const uint g_sub = lane >> 3u;
      const uint k_l = lane & 7u;
      for (uint g = 0u; g < groups64; g += 4u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const uint col = cur_g * 64u + k_l * 8u;
          device const bfloat4 *xb4 =
              reinterpret_cast<device const bfloat4 *>(x_row + col);
          float4 xf0, xf1;
          if constexpr (PROLOGUE_NORM) {
            device const bfloat4 *wb4 =
                reinterpret_cast<device const bfloat4 *>(rms_w + col);
            xf0 = float4(bfloat4(float4(xb4[0]) * inv_rms * (float4(wb4[0]) + off4)));
            xf1 = float4(bfloat4(float4(xb4[1]) * inv_rms * (float4(wb4[1]) + off4)));
          } else {
            xf0 = float4(xb4[0]);
            xf1 = float4(xb4[1]);
          }
          const float lsum = ((xf0.x + xf0.y) + (xf0.z + xf0.w)) +
                             ((xf1.x + xf1.y) + (xf1.z + xf1.w));
          const ulong idx = tile_base + cur_g;
          device const uchar *c_tile = r_codes + idx * 1024ul;
          const bool r_lane_v2 = (p.router_group_size >> 31u) != 0u;
          const uint c4_0 = r_lane_v2 ? omlx_m5_pi16(k_l * 2u) : (k_l * 2u);
          const uint c4_1 = r_lane_v2 ? omlx_m5_pi16(k_l * 2u + 1u) : (k_l * 2u + 1u);
          device const uint *r0_u32 = reinterpret_cast<device const uint *>(
              c_tile + ulong(rn0 + 0u) * 64ul);
          device const uint *r1_u32 = reinterpret_cast<device const uint *>(
              c_tile + ulong(rn0 + 1u) * 64ul);
          const uint2 cw0 = uint2(r0_u32[c4_0], r0_u32[c4_1]);
          const uint2 cw1 = uint2(r1_u32[c4_0], r1_u32[c4_1]);
          const uint sbp0 = sb[ulong(cur_g) * 256ul + row_sg + 0u];
          const uint sbp1 = sb[ulong(cur_g) * 256ul + row_sg + 1u];
          const float sc0 = as_type<float>((sbp0 & 0xFFFFu) << 16u);
          const float bi0 = as_type<float>(sbp0 & 0xFFFF0000u) - 1024.0f * sc0;
          const float sc1 = as_type<float>((sbp1 & 0xFFFFu) << 16u);
          const float bi1 = as_type<float>(sbp1 & 0xFFFF0000u) - 1024.0f * sc1;
          float d0 = fma(xf0.x, float(as_type<half>(ushort(0x6400u | (cw0.x & 0xFFu)))), 0.0f);
          d0 = fma(xf0.y, float(as_type<half>(ushort(0x6400u | ((cw0.x >> 8u) & 0xFFu)))), d0);
          d0 = fma(xf0.z, float(as_type<half>(ushort(0x6400u | ((cw0.x >> 16u) & 0xFFu)))), d0);
          d0 = fma(xf0.w, float(as_type<half>(ushort(0x6400u | (cw0.x >> 24u)))), d0);
          d0 = fma(xf1.x, float(as_type<half>(ushort(0x6400u | (cw0.y & 0xFFu)))), d0);
          d0 = fma(xf1.y, float(as_type<half>(ushort(0x6400u | ((cw0.y >> 8u) & 0xFFu)))), d0);
          d0 = fma(xf1.z, float(as_type<half>(ushort(0x6400u | ((cw0.y >> 16u) & 0xFFu)))), d0);
          d0 = fma(xf1.w, float(as_type<half>(ushort(0x6400u | (cw0.y >> 24u)))), d0);
          float d1 = fma(xf0.x, float(as_type<half>(ushort(0x6400u | (cw1.x & 0xFFu)))), 0.0f);
          d1 = fma(xf0.y, float(as_type<half>(ushort(0x6400u | ((cw1.x >> 8u) & 0xFFu)))), d1);
          d1 = fma(xf0.z, float(as_type<half>(ushort(0x6400u | ((cw1.x >> 16u) & 0xFFu)))), d1);
          d1 = fma(xf0.w, float(as_type<half>(ushort(0x6400u | (cw1.x >> 24u)))), d1);
          d1 = fma(xf1.x, float(as_type<half>(ushort(0x6400u | (cw1.y & 0xFFu)))), d1);
          d1 = fma(xf1.y, float(as_type<half>(ushort(0x6400u | ((cw1.y >> 8u) & 0xFFu)))), d1);
          d1 = fma(xf1.z, float(as_type<half>(ushort(0x6400u | ((cw1.y >> 16u) & 0xFFu)))), d1);
          d1 = fma(xf1.w, float(as_type<half>(ushort(0x6400u | (cw1.y >> 24u)))), d1);
          acc0 = fma(sc0, d0, fma(bi0, lsum, acc0));
          acc1 = fma(sc1, d1, fma(bi1, lsum, acc1));
        }
      }
    } else if (!FAST_Q8_PRETILED && p.router_mode == 0u) {
      device const bfloat *rw = reinterpret_cast<device const bfloat *>(r_codes);
      const uint g_sub = lane >> 3u;
      const uint k_l = lane & 7u;
      for (uint g = 0u; g < groups64; g += 4u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const uint col = cur_g * 64u + k_l * 8u;
          device const bfloat4 *xb4 =
              reinterpret_cast<device const bfloat4 *>(x_row + col);
          float4 xf0, xf1;
          if constexpr (PROLOGUE_NORM) {
            device const bfloat4 *wb4 =
                reinterpret_cast<device const bfloat4 *>(rms_w + col);
            xf0 = float4(bfloat4(float4(xb4[0]) * inv_rms * (float4(wb4[0]) + off4)));
            xf1 = float4(bfloat4(float4(xb4[1]) * inv_rms * (float4(wb4[1]) + off4)));
          } else {
            xf0 = float4(xb4[0]);
            xf1 = float4(xb4[1]);
          }
          device const bfloat4 *w0b = reinterpret_cast<device const bfloat4 *>(
              rw + ulong(row_sg + 0u) * p.h + col);
          device const bfloat4 *w1b = reinterpret_cast<device const bfloat4 *>(
              rw + ulong(row_sg + 1u) * p.h + col);
          const float4 w00 = float4(w0b[0]), w01 = float4(w0b[1]);
          const float4 w10 = float4(w1b[0]), w11 = float4(w1b[1]);
          acc0 = fma(xf0.x, w00.x, fma(xf0.y, w00.y, fma(xf0.z, w00.z, fma(xf0.w, w00.w, acc0))));
          acc0 = fma(xf1.x, w01.x, fma(xf1.y, w01.y, fma(xf1.z, w01.z, fma(xf1.w, w01.w, acc0))));
          acc1 = fma(xf0.x, w10.x, fma(xf0.y, w10.y, fma(xf0.z, w10.z, fma(xf0.w, w10.w, acc1))));
          acc1 = fma(xf1.x, w11.x, fma(xf1.y, w11.y, fma(xf1.z, w11.z, fma(xf1.w, w11.w, acc1))));
        }
      }
    } else if (!FAST_Q8_PRETILED) {
      const uint gs = ((p.router_group_size & 0x7FFFFFFFu) == 128u) ? 128u : 64u;
      const uint sb_stride = p.router_padded_k / gs;
      device const bfloat *scales = reinterpret_cast<device const bfloat *>(r_aux0);
      device const bfloat *biases = reinterpret_cast<device const bfloat *>(r_aux1);
      const uint g_sub = lane >> 4u;
      const uint k_l = lane & 15u;
      for (uint g = 0u; g < groups64; g += 2u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const uint col = cur_g * 64u + k_l * 4u;
          const bfloat4 x_in = *reinterpret_cast<device const bfloat4 *>(x_row + col);
          float4 xf;
          if constexpr (PROLOGUE_NORM) {
            const bfloat4 w_in = *reinterpret_cast<device const bfloat4 *>(rms_w + col);
            xf = float4(bfloat4(float4(x_in) * inv_rms * (float4(w_in) + off4)));
          } else {
            xf = float4(x_in);
          }
          const float lsum = (xf.x + xf.y) + (xf.z + xf.w);
          const uint g_src = (gs == 128u) ? (cur_g >> 1u) : cur_g;
          const uchar4 cb0 = reinterpret_cast<device const uchar4 *>(
              r_codes + ulong(row_sg + 0u) * p.router_padded_k + cur_g * 64u)[k_l];
          const uchar4 cb1 = reinterpret_cast<device const uchar4 *>(
              r_codes + ulong(row_sg + 1u) * p.router_padded_k + cur_g * 64u)[k_l];
          const float sc0 = float(scales[ulong(row_sg + 0u) * sb_stride + g_src]);
          const float bi0 = float(biases[ulong(row_sg + 0u) * sb_stride + g_src]) - 1024.0f * sc0;
          const float sc1 = float(scales[ulong(row_sg + 1u) * sb_stride + g_src]);
          const float bi1 = float(biases[ulong(row_sg + 1u) * sb_stride + g_src]) - 1024.0f * sc1;
          float d0 = fma(xf.x, float(as_type<half>(ushort(0x6400u | cb0.x))), 0.0f);
          d0 = fma(xf.y, float(as_type<half>(ushort(0x6400u | cb0.y))), d0);
          d0 = fma(xf.z, float(as_type<half>(ushort(0x6400u | cb0.z))), d0);
          d0 = fma(xf.w, float(as_type<half>(ushort(0x6400u | cb0.w))), d0);
          float d1 = fma(xf.x, float(as_type<half>(ushort(0x6400u | cb1.x))), 0.0f);
          d1 = fma(xf.y, float(as_type<half>(ushort(0x6400u | cb1.y))), d1);
          d1 = fma(xf.z, float(as_type<half>(ushort(0x6400u | cb1.z))), d1);
          d1 = fma(xf.w, float(as_type<half>(ushort(0x6400u | cb1.w))), d1);
          acc0 = fma(sc0, d0, fma(bi0, lsum, acc0));
          acc1 = fma(sc1, d1, fma(bi1, lsum, acc1));
        }
      }
    }

    device atomic_uint *row_logits_u32 = router_logits_u32 + ulong(b) * 128ul;
    const float v0 = simd_sum(acc0);
    const float v1 = simd_sum(acc1);
    if (lane == 0u) {
      const uint lo16 = uint(as_type<ushort>(bfloat(v0)));
      const uint hi16 = uint(as_type<ushort>(bfloat(v1)));
      atomic_store_explicit(&row_logits_u32[tg.x * 2u + sg],
                            lo16 | (hi16 << 16u), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);
    uint is_last = 0u;
    const uint done_idx = p.moe_slot * 8u + b;
    if (sg == 0u && lane == 0u) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                          thread_scope_device);
      if (atomic_fetch_add_explicit(&moe_tg_done_buf[done_idx], 1u,
                                    memory_order_acq_rel,
                                    mem_flags::mem_device) == 63u) {
        is_last = 1u;
      }
    }
    if (sg == 0u) {
      is_last = simd_broadcast_first(is_last);
      if (is_last != 0u) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                            thread_scope_device);
        if (lane == 0u) {
          atomic_store_explicit(&moe_tg_done_buf[done_idx], 0u,
                                memory_order_relaxed);
        }
        moe9_top8_from_atomic_u32_nospill(row_logits_u32,
                                          route_ids_out + ulong(b) * 8ul,
                                          route_scores_out + ulong(b) * 8ul,
                                          lane);
      }
    }
    return;
  }

  // Concurrent tg.x == 64: computes shared_gate_out[b] for row b
  if (p.sgate_mode != 0u) {
    float sg_acc = 0.0f;
    const uint g_sub = sg * 2u + (lane >> 4u);
    const uint k_l = lane & 15u;
    if (FAST_Q8_PRETILED || p.sgate_mode == 2u) {
      device const uint *sg_sb = reinterpret_cast<device const uint *>(sg_aux0);
      const uint pad_n = max(p.sgate_padded_n, 16u);
      for (uint g = 0u; g < groups64; g += 4u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const uint col = cur_g * 64u + k_l * 4u;
          const bfloat4 x_in = *reinterpret_cast<device const bfloat4 *>(x_row + col);
          float4 xf;
          if constexpr (PROLOGUE_NORM) {
            const bfloat4 w_in = *reinterpret_cast<device const bfloat4 *>(rms_w + col);
            xf = float4(bfloat4(float4(x_in) * inv_rms * (float4(w_in) + off4)));
          } else {
            xf = float4(x_in);
          }
          const float lsum = (xf.x + xf.y) + (xf.z + xf.w);
          const uint sg_c4 = ((p.sgate_group_size >> 31u) != 0u) ? omlx_m5_pi16(k_l) : k_l;
          const uchar4 cb = reinterpret_cast<device const uchar4 *>(
              sg_codes + ulong(cur_g) * 1024ul)[sg_c4];
          const uint sbp = sg_sb[ulong(cur_g) * pad_n];
          const float sc = as_type<float>((sbp & 0xFFFFu) << 16u);
          const float bi = as_type<float>(sbp & 0xFFFF0000u) - 1024.0f * sc;
          float d = fma(xf.x, float(as_type<half>(ushort(0x6400u | cb.x))), 0.0f);
          d = fma(xf.y, float(as_type<half>(ushort(0x6400u | cb.y))), d);
          d = fma(xf.z, float(as_type<half>(ushort(0x6400u | cb.z))), d);
          d = fma(xf.w, float(as_type<half>(ushort(0x6400u | cb.w))), d);
          sg_acc = fma(sc, d, fma(bi, lsum, sg_acc));
        }
      }
    } else if (!FAST_Q8_PRETILED && p.sgate_mode == 1u) {
      device const bfloat *sgw = reinterpret_cast<device const bfloat *>(sg_codes);
      for (uint g = 0u; g < groups64; g += 4u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const uint col = cur_g * 64u + k_l * 4u;
          const bfloat4 x_in = *reinterpret_cast<device const bfloat4 *>(x_row + col);
          float4 xf;
          if constexpr (PROLOGUE_NORM) {
            const bfloat4 w_in = *reinterpret_cast<device const bfloat4 *>(rms_w + col);
            xf = float4(bfloat4(float4(x_in) * inv_rms * (float4(w_in) + off4)));
          } else {
            xf = float4(x_in);
          }
          const float4 wf = float4(*reinterpret_cast<device const bfloat4 *>(sgw + col));
          sg_acc = fma(xf.x, wf.x, fma(xf.y, wf.y, fma(xf.z, wf.z, fma(xf.w, wf.w, sg_acc))));
        }
      }
    } else if (!FAST_Q8_PRETILED && p.sgate_mode == 3u) {
      const uint gs_s = ((p.sgate_group_size & 0x7FFFFFFFu) == 128u) ? 128u : 64u;
      device const bfloat *scales = reinterpret_cast<device const bfloat *>(sg_aux0);
      device const bfloat *biases = reinterpret_cast<device const bfloat *>(sg_aux1);
      for (uint g = 0u; g < groups64; g += 4u) {
        const uint cur_g = g + g_sub;
        if (cur_g < groups64) {
          const uint col = cur_g * 64u + k_l * 4u;
          const bfloat4 x_in = *reinterpret_cast<device const bfloat4 *>(x_row + col);
          float4 xf;
          if constexpr (PROLOGUE_NORM) {
            const bfloat4 w_in = *reinterpret_cast<device const bfloat4 *>(rms_w + col);
            xf = float4(bfloat4(float4(x_in) * inv_rms * (float4(w_in) + off4)));
          } else {
            xf = float4(x_in);
          }
          const float lsum = (xf.x + xf.y) + (xf.z + xf.w);
          const uchar4 cb = reinterpret_cast<device const uchar4 *>(sg_codes + col)[0];
          const uint g_src = (gs_s == 128u) ? (cur_g >> 1u) : cur_g;
          const float sc = float(scales[g_src]);
          const float bi = float(biases[g_src]) - 1024.0f * sc;
          float d = fma(xf.x, float(as_type<half>(ushort(0x6400u | cb.x))), 0.0f);
          d = fma(xf.y, float(as_type<half>(ushort(0x6400u | cb.y))), d);
          d = fma(xf.z, float(as_type<half>(ushort(0x6400u | cb.z))), d);
          d = fma(xf.w, float(as_type<half>(ushort(0x6400u | cb.w))), d);
          sg_acc = fma(sc, d, fma(bi, lsum, sg_acc));
        }
      }
    }
    sg_acc = simd_sum(sg_acc);
    if (lane == 0u) {
      tg_sg[sg] = sg_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      shared_gate_out[b] = bfloat(tg_sg[0] + tg_sg[1]);
    }
  }
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_moe9_router_q8_pretiled_fused_top8_bt(
    device const bfloat *norm [[buffer(0)]],
    device const uchar *r_codes [[buffer(1)]],
    device const uchar *r_aux0 [[buffer(2)]],
    device const uchar *r_aux1 [[buffer(3)]],
    device const uchar *sg_codes [[buffer(4)]],
    device const uchar *sg_aux0 [[buffer(5)]],
    device const uchar *sg_aux1 [[buffer(6)]],
    device atomic_uint *router_logits_u32 [[buffer(7)]],
    device uint *route_ids_out [[buffer(8)]],
    device bfloat *route_scores_out [[buffer(9)]],
    device bfloat *shared_gate_out [[buffer(10)]],
    device atomic_uint *moe_tg_done_buf [[buffer(11)]],
    constant Moe9RouterFusedTop8BtParams &p [[buffer(12)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  (void)r_aux1;
  (void)sg_aux1;
  threadgroup float tg_sg[2];
  moe9_router_fused_top8_bt_impl<false, true>(
      norm, nullptr, nullptr, nullptr, r_codes, r_aux0, r_aux1, sg_codes,
      sg_aux0, sg_aux1, router_logits_u32, route_ids_out, route_scores_out,
      shared_gate_out, moe_tg_done_buf, tg_sg, p, tid, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_moe9_router_qmv_fused_top8_bt(
    device const bfloat *norm [[buffer(0)]],
    device const uchar *r_codes [[buffer(1)]],
    device const uchar *r_aux0 [[buffer(2)]],
    device const uchar *r_aux1 [[buffer(3)]],
    device const uchar *sg_codes [[buffer(4)]],
    device const uchar *sg_aux0 [[buffer(5)]],
    device const uchar *sg_aux1 [[buffer(6)]],
    device atomic_uint *router_logits_u32 [[buffer(7)]],
    device uint *route_ids_out [[buffer(8)]],
    device bfloat *route_scores_out [[buffer(9)]],
    device bfloat *shared_gate_out [[buffer(10)]],
    device atomic_uint *moe_tg_done_buf [[buffer(11)]],
    constant Moe9RouterFusedTop8BtParams &p [[buffer(12)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float tg_sg[2];
  moe9_router_fused_top8_bt_impl<false, false>(
      norm, nullptr, nullptr, nullptr, r_codes, r_aux0, r_aux1, sg_codes,
      sg_aux0, sg_aux1, router_logits_u32, route_ids_out, route_scores_out,
      shared_gate_out, moe_tg_done_buf, tg_sg, p, tid, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_moe9_router_q8_pretiled_norm_fused_top8_bt(
    device const bfloat *hidden [[buffer(0)]],
    device const bfloat *rms_w [[buffer(1)]],
    device const float *rms_sq_buf [[buffer(2)]],
    device bfloat *norm_out [[buffer(3)]],
    device const uchar *r_codes [[buffer(4)]],
    device const uchar *r_aux0 [[buffer(5)]],
    device const uchar *r_aux1 [[buffer(6)]],
    device const uchar *sg_codes [[buffer(7)]],
    device const uchar *sg_aux0 [[buffer(8)]],
    device const uchar *sg_aux1 [[buffer(9)]],
    device atomic_uint *router_logits_u32 [[buffer(10)]],
    device uint *route_ids_out [[buffer(11)]],
    device bfloat *route_scores_out [[buffer(12)]],
    device bfloat *shared_gate_out [[buffer(13)]],
    device atomic_uint *moe_tg_done_buf [[buffer(14)]],
    constant Moe9RouterFusedTop8BtParams &p [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  (void)r_aux1;
  (void)sg_aux1;
  threadgroup float tg_sg[2];
  moe9_router_fused_top8_bt_impl<true, true>(
      hidden, rms_w, rms_sq_buf, norm_out, r_codes, r_aux0, r_aux1, sg_codes,
      sg_aux0, sg_aux1, router_logits_u32, route_ids_out, route_scores_out,
      shared_gate_out, moe_tg_done_buf, tg_sg, p, tid, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_moe9_router_qmv_norm_fused_top8_bt(
    device const bfloat *hidden [[buffer(0)]],
    device const bfloat *rms_w [[buffer(1)]],
    device const float *rms_sq_buf [[buffer(2)]],
    device bfloat *norm_out [[buffer(3)]],
    device const uchar *r_codes [[buffer(4)]],
    device const uchar *r_aux0 [[buffer(5)]],
    device const uchar *r_aux1 [[buffer(6)]],
    device const uchar *sg_codes [[buffer(7)]],
    device const uchar *sg_aux0 [[buffer(8)]],
    device const uchar *sg_aux1 [[buffer(9)]],
    device atomic_uint *router_logits_u32 [[buffer(10)]],
    device uint *route_ids_out [[buffer(11)]],
    device bfloat *route_scores_out [[buffer(12)]],
    device bfloat *shared_gate_out [[buffer(13)]],
    device atomic_uint *moe_tg_done_buf [[buffer(14)]],
    constant Moe9RouterFusedTop8BtParams &p [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float tg_sg[2];
  moe9_router_fused_top8_bt_impl<true, false>(
      hidden, rms_w, rms_sq_buf, norm_out, r_codes, r_aux0, r_aux1, sg_codes,
      sg_aux0, sg_aux1, router_logits_u32, route_ids_out, route_scores_out,
      shared_gate_out, moe_tg_done_buf, tg_sg, p, tid, lane, sg, tg);
}

// Dispatch 2: Zero-Spill Batched MXFP4 SwiGLU (grid = ((F+3)/4, 9*B, 1), 64 threads, 4 KiB tg_x).
// Scalar accumulators (0 alloca); supports H <= 2048 in 1 cooperative load and H > 2048 via 2048-elem chunks.
[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_moe9_mxfp4_kl16_swiglu_bt(
    device const bfloat *norm [[buffer(0)]],
    device const uint *route_ids [[buffer(1)]],
    device const uchar *routed_gc [[buffer(2)]],
    device const uchar *routed_gs [[buffer(3)]],
    device const uchar *routed_uc [[buffer(4)]],
    device const uchar *routed_us [[buffer(5)]],
    device const uchar *shared_gc [[buffer(6)]],
    device const uchar *shared_gs [[buffer(7)]],
    device const uchar *shared_uc [[buffer(8)]],
    device const uchar *shared_us [[buffer(9)]],
    device bfloat *mid_out [[buffer(10)]],
    constant Moe9GemmBtParams &p [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint b = tg.y / 9u;
  const uint slot = tg.y - b * 9u;
  const uint tg_row0 = tg.x * 4u;
  if (b >= p.rows || slot > p.top_k || tg_row0 >= p.f) {
    return;
  }

  uint exp_id = 0u;
  if (slot < p.top_k) {
    exp_id = route_ids[ulong(b) * p.top_k + slot];
    if (exp_id >= p.num_experts) {
      if (tid < 4u && tg_row0 + tid < p.f) {
        mid_out[ulong(tg.y) * p.f + tg_row0 + tid] = bfloat(0.0f);
      }
      return;
    }
  }

  threadgroup bfloat4 tg_x4[512]; // 2048 bfloats = 4 KiB cooperative load

  const uint row_sg = tg_row0 + sg * 2u;
  const bool valid0 = (row_sg + 0u < p.f);
  const bool valid1 = (row_sg + 1u < p.f);
  const uint b_sub = lane >> 4u; // 0..1 (2 blocks of 128 = 256 K-elements per step)
  const uint k_l = lane & 15u;   // 0..15 (8 nibbles = 4 B per lane -> 128 B across 32 lanes)
  const uint sub32 = k_l >> 2u;  // 0..3
  const uint blocks = p.h >> 7u;

  float ag0 = 0.0f, ag1 = 0.0f;
  float au0 = 0.0f, au1 = 0.0f;

  const ulong r_c_stride = ulong(p.routed_padded_k >> 1u);
  const ulong r_s_stride = ulong(p.routed_padded_k >> 5u);
  const ulong s_c_stride = ulong(p.shared_padded_k >> 1u);
  const ulong s_s_stride = ulong(p.shared_padded_k >> 5u);
  device const bfloat *x_row = norm + ulong(b) * p.h;

  for (uint blk_base = 0u; blk_base < blocks; blk_base += 16u) {
    const uint chunk_blks = min(16u, blocks - blk_base);
    const uint chunk_vecs = chunk_blks * 32u;
    device const bfloat4 *x_src4 =
        reinterpret_cast<device const bfloat4 *>(x_row + blk_base * 128u);
    for (uint i = tid; i < chunk_vecs; i += 64u) {
      tg_x4[i] = x_src4[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (slot < p.top_k) {
      const ulong row0 = ulong(exp_id) * p.f + row_sg;
      const ulong row1 = row0 + 1ul;
      for (uint blk = 0u; blk < chunk_blks; blk += 2u) {
        const uint cur_local = blk + b_sub;
        if (cur_local < chunk_blks) {
          const uint cur_blk = blk_base + cur_local;
          const uint x_idx = cur_local * 32u + k_l * 2u;
          const half4 h0 = half4(float4(tg_x4[x_idx + 0u]));
          const half4 h1 = half4(float4(tg_x4[x_idx + 1u]));
          const ulong cb = cur_blk * 64ul;
          const ulong sb = cur_blk * 4ul;
          if (valid0) {
            ag0 = moe9_mxfp4_row_dot_acc(routed_gc, routed_gs, row0 * r_c_stride + cb,
                                         row0 * r_s_stride + sb, k_l, sub32, h0, h1, ag0);
            au0 = moe9_mxfp4_row_dot_acc(routed_uc, routed_us, row0 * r_c_stride + cb,
                                         row0 * r_s_stride + sb, k_l, sub32, h0, h1, au0);
          }
          if (valid1) {
            ag1 = moe9_mxfp4_row_dot_acc(routed_gc, routed_gs, row1 * r_c_stride + cb,
                                         row1 * r_s_stride + sb, k_l, sub32, h0, h1, ag1);
            au1 = moe9_mxfp4_row_dot_acc(routed_uc, routed_us, row1 * r_c_stride + cb,
                                         row1 * r_s_stride + sb, k_l, sub32, h0, h1, au1);
          }
        }
      }
    } else if (p.shared_tiled != 0u) {
      const uint tile_n = row_sg >> 4u;
      const uint rn0 = row_sg & 15u;
      const uint rn1 = rn0 + 1u;
      for (uint blk = 0u; blk < chunk_blks; blk += 2u) {
        const uint cur_local = blk + b_sub;
        if (cur_local < chunk_blks) {
          const uint cur_blk = blk_base + cur_local;
          const uint x_idx = cur_local * 32u + k_l * 2u;
          const half4 h0 = half4(float4(tg_x4[x_idx + 0u]));
          const half4 h1 = half4(float4(tg_x4[x_idx + 1u]));
          const ulong idx = ulong(tile_n) * blocks + cur_blk;
          const ulong c_base = idx * 1024ul;
          const ulong s_base = idx * 64ul;
          if (valid0) {
            ag0 = moe9_mxfp4_row_dot_acc(shared_gc, shared_gs, c_base + rn0 * 64ul,
                                         s_base + rn0 * 4ul, k_l, sub32, h0, h1, ag0);
            au0 = moe9_mxfp4_row_dot_acc(shared_uc, shared_us, c_base + rn0 * 64ul,
                                         s_base + rn0 * 4ul, k_l, sub32, h0, h1, au0);
          }
          if (valid1) {
            ag1 = moe9_mxfp4_row_dot_acc(shared_gc, shared_gs, c_base + rn1 * 64ul,
                                         s_base + rn1 * 4ul, k_l, sub32, h0, h1, ag1);
            au1 = moe9_mxfp4_row_dot_acc(shared_uc, shared_us, c_base + rn1 * 64ul,
                                         s_base + rn1 * 4ul, k_l, sub32, h0, h1, au1);
          }
        }
      }
    } else {
      const ulong row0 = ulong(row_sg);
      const ulong row1 = row0 + 1ul;
      for (uint blk = 0u; blk < chunk_blks; blk += 2u) {
        const uint cur_local = blk + b_sub;
        if (cur_local < chunk_blks) {
          const uint cur_blk = blk_base + cur_local;
          const uint x_idx = cur_local * 32u + k_l * 2u;
          const half4 h0 = half4(float4(tg_x4[x_idx + 0u]));
          const half4 h1 = half4(float4(tg_x4[x_idx + 1u]));
          const ulong cb = cur_blk * 64ul;
          const ulong sb = cur_blk * 4ul;
          if (valid0) {
            ag0 = moe9_mxfp4_row_dot_acc(shared_gc, shared_gs, row0 * s_c_stride + cb,
                                         row0 * s_s_stride + sb, k_l, sub32, h0, h1, ag0);
            au0 = moe9_mxfp4_row_dot_acc(shared_uc, shared_us, row0 * s_c_stride + cb,
                                         row0 * s_s_stride + sb, k_l, sub32, h0, h1, au0);
          }
          if (valid1) {
            ag1 = moe9_mxfp4_row_dot_acc(shared_gc, shared_gs, row1 * s_c_stride + cb,
                                         row1 * s_s_stride + sb, k_l, sub32, h0, h1, ag1);
            au1 = moe9_mxfp4_row_dot_acc(shared_uc, shared_us, row1 * s_c_stride + cb,
                                         row1 * s_s_stride + sb, k_l, sub32, h0, h1, au1);
          }
        }
      }
    }

    if (blk_base + 16u < blocks) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  const float gsum0 = simd_sum(ag0);
  const float usum0 = simd_sum(au0);
  const float gsum1 = simd_sum(ag1);
  const float usum1 = simd_sum(au1);
  if (lane == 0u) {
    const ulong out_base = ulong(tg.y) * p.f;
    if (valid0) {
      const float gv = float(bfloat(gsum0));
      const float uv = float(bfloat(usum0));
      mid_out[out_base + row_sg + 0u] = bfloat(nax_silu_f32(gv) * uv);
    }
    if (valid1) {
      const float gv = float(bfloat(gsum1));
      const float uv = float(bfloat(usum1));
      mid_out[out_base + row_sg + 1u] = bfloat(nax_silu_f32(gv) * uv);
    }
  }
}

// Dispatch 3: Zero-Spill Batched MXFP4 Down + Combine (grid = ((H+7)/8, B, 1), 96 threads = 3 SGs x 3 slots).
template <bool ADD_SQ>
inline void moe9_mxfp4_kl16_down_sk3_r8_bt_impl(
    device const bfloat *mid_in,
    device const uint *route_ids,
    device const bfloat *route_scores,
    device const bfloat *shared_gate_buf,
    device const bfloat *norm_or_residual,
    device const uchar *routed_dc,
    device const uchar *routed_ds,
    device const uchar *shared_dc,
    device const uchar *shared_ds,
    device bfloat *out_buf,
    device atomic_float *rms_sq_buf,
    threadgroup float (&sg_gate_part)[3],
    threadgroup float (&tg_red)[3][8],
    threadgroup float (&tg_shared)[8],
    constant Moe9GemmBtParams &p,
    uint lane,
    uint sg,
    uint3 tg) {
  const uint b = tg.y;
  const uint row_base = tg.x * 8u;
  if (b >= p.rows || row_base >= p.h || sg >= 3u) {
    return;
  }

  if (sg == 2u && lane < 8u) {
    tg_shared[lane] = 0.0f;
  }

  float gate_logit = 0.0f;
  if ((p.flags & 1u) != 0u) {
    const uint tid = sg * 32u + lane;
    const uint h4 = p.h >> 2u;
    device const bfloat4 *gw4 =
        reinterpret_cast<device const bfloat4 *>(shared_gate_buf);
    device const bfloat4 *xn4 = reinterpret_cast<device const bfloat4 *>(
        norm_or_residual + ulong(b) * p.h);
    float g_dot = 0.0f;
    for (uint i = tid; i < h4; i += 96u) {
      g_dot += dot(float4(gw4[i]), float4(xn4[i]));
    }
    g_dot = simd_sum(g_dot);
    if (lane == 0u) {
      sg_gate_part[sg] = g_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gate_logit = float(bfloat(sg_gate_part[0] + sg_gate_part[1] + sg_gate_part[2]));
  } else {
    gate_logit = float(shared_gate_buf[b]);
  }

  const uint b_sub = lane >> 4u;
  const uint k_l = lane & 15u;
  const uint sub32 = k_l >> 2u;
  const uint blocks = p.f >> 7u;
  const bool full8 = (row_base + 7u < p.h);

  const ulong r_c_stride = ulong(p.routed_padded_k >> 1u);
  const ulong r_s_stride = ulong(p.routed_padded_k >> 5u);
  const ulong s_c_stride = ulong(p.shared_padded_k >> 1u);
  const ulong s_s_stride = ulong(p.shared_padded_k >> 5u);

  float pr0 = 0.0f, pr1 = 0.0f, pr2 = 0.0f, pr3 = 0.0f;
  float pr4 = 0.0f, pr5 = 0.0f, pr6 = 0.0f, pr7 = 0.0f;

  const uint s0 = sg * 3u;
  for (uint slot = s0; slot < s0 + 3u; ++slot) {
    if (slot > p.top_k) {
      continue;
    }

    uint exp_id = 0u;
    float w_score = 1.0f;
    if (slot < p.top_k) {
      const ulong r_idx = ulong(b) * p.top_k + slot;
      exp_id = route_ids[r_idx];
      w_score = float(route_scores[r_idx]);
      if (exp_id >= p.num_experts || w_score == 0.0f) {
        continue;
      }
    }

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    device const bfloat *xm = mid_in + ulong(b * 9u + slot) * p.f;

    if (slot < p.top_k) {
      const ulong exp_row0 = ulong(exp_id) * p.h + row_base;
      const bool do_fwht =
          (slot < p.top_k) ? ((p.flags & 4u) != 0u) : ((p.flags & 8u) != 0u);
      for (uint blk = 0u; blk < blocks; blk += 2u) {
        const uint cur_blk = blk + b_sub;
        float4 xf0 = float4(0.0f);
        float4 xf1 = float4(0.0f);
        if (cur_blk < blocks) {
          device const bfloat4 *xb =
              reinterpret_cast<device const bfloat4 *>(xm + cur_blk * 128u + k_l * 8u);
          xf0 = float4(xb[0]);
          xf1 = float4(xb[1]);
        }
        if (do_fwht) {
          omlx_m5_fwht64_sub8(xf0, xf1, lane);
        }
        if (cur_blk < blocks) {
          const half4 h0 = half4(xf0);
          const half4 h1 = half4(xf1);
          const ulong cb = cur_blk * 64ul;
          const ulong sb = cur_blk * 4ul;
          if (full8) {
            a0 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 0ul) * r_c_stride + cb, (exp_row0 + 0ul) * r_s_stride + sb, k_l, sub32, h0, h1, a0);
            a1 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 1ul) * r_c_stride + cb, (exp_row0 + 1ul) * r_s_stride + sb, k_l, sub32, h0, h1, a1);
            a2 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 2ul) * r_c_stride + cb, (exp_row0 + 2ul) * r_s_stride + sb, k_l, sub32, h0, h1, a2);
            a3 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 3ul) * r_c_stride + cb, (exp_row0 + 3ul) * r_s_stride + sb, k_l, sub32, h0, h1, a3);
            a4 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 4ul) * r_c_stride + cb, (exp_row0 + 4ul) * r_s_stride + sb, k_l, sub32, h0, h1, a4);
            a5 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 5ul) * r_c_stride + cb, (exp_row0 + 5ul) * r_s_stride + sb, k_l, sub32, h0, h1, a5);
            a6 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 6ul) * r_c_stride + cb, (exp_row0 + 6ul) * r_s_stride + sb, k_l, sub32, h0, h1, a6);
            a7 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 7ul) * r_c_stride + cb, (exp_row0 + 7ul) * r_s_stride + sb, k_l, sub32, h0, h1, a7);
          } else {
            if (row_base + 0u < p.h) a0 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 0ul) * r_c_stride + cb, (exp_row0 + 0ul) * r_s_stride + sb, k_l, sub32, h0, h1, a0);
            if (row_base + 1u < p.h) a1 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 1ul) * r_c_stride + cb, (exp_row0 + 1ul) * r_s_stride + sb, k_l, sub32, h0, h1, a1);
            if (row_base + 2u < p.h) a2 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 2ul) * r_c_stride + cb, (exp_row0 + 2ul) * r_s_stride + sb, k_l, sub32, h0, h1, a2);
            if (row_base + 3u < p.h) a3 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 3ul) * r_c_stride + cb, (exp_row0 + 3ul) * r_s_stride + sb, k_l, sub32, h0, h1, a3);
            if (row_base + 4u < p.h) a4 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 4ul) * r_c_stride + cb, (exp_row0 + 4ul) * r_s_stride + sb, k_l, sub32, h0, h1, a4);
            if (row_base + 5u < p.h) a5 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 5ul) * r_c_stride + cb, (exp_row0 + 5ul) * r_s_stride + sb, k_l, sub32, h0, h1, a5);
            if (row_base + 6u < p.h) a6 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 6ul) * r_c_stride + cb, (exp_row0 + 6ul) * r_s_stride + sb, k_l, sub32, h0, h1, a6);
            if (row_base + 7u < p.h) a7 = moe9_mxfp4_row_dot_acc(routed_dc, routed_ds, (exp_row0 + 7ul) * r_c_stride + cb, (exp_row0 + 7ul) * r_s_stride + sb, k_l, sub32, h0, h1, a7);
          }
        }
      }
    } else if (p.shared_tiled != 0u) {
      const uint tile_n = row_base >> 4u;
      const uint rn0 = row_base & 15u;
      const bool do_fwht =
          (slot < p.top_k) ? ((p.flags & 4u) != 0u) : ((p.flags & 8u) != 0u);
      for (uint blk = 0u; blk < blocks; blk += 2u) {
        const uint cur_blk = blk + b_sub;
        float4 xf0 = float4(0.0f);
        float4 xf1 = float4(0.0f);
        if (cur_blk < blocks) {
          device const bfloat4 *xb =
              reinterpret_cast<device const bfloat4 *>(xm + cur_blk * 128u + k_l * 8u);
          xf0 = float4(xb[0]);
          xf1 = float4(xb[1]);
        }
        if (do_fwht) {
          omlx_m5_fwht64_sub8(xf0, xf1, lane);
        }
        if (cur_blk < blocks) {
          const half4 h0 = half4(xf0);
          const half4 h1 = half4(xf1);
          const ulong idx = ulong(tile_n) * blocks + cur_blk;
          const ulong c_base = idx * 1024ul;
          const ulong s_base = idx * 64ul;
          if (full8) {
            a0 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 0u) * 64ul, s_base + (rn0 + 0u) * 4ul, k_l, sub32, h0, h1, a0);
            a1 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 1u) * 64ul, s_base + (rn0 + 1u) * 4ul, k_l, sub32, h0, h1, a1);
            a2 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 2u) * 64ul, s_base + (rn0 + 2u) * 4ul, k_l, sub32, h0, h1, a2);
            a3 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 3u) * 64ul, s_base + (rn0 + 3u) * 4ul, k_l, sub32, h0, h1, a3);
            a4 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 4u) * 64ul, s_base + (rn0 + 4u) * 4ul, k_l, sub32, h0, h1, a4);
            a5 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 5u) * 64ul, s_base + (rn0 + 5u) * 4ul, k_l, sub32, h0, h1, a5);
            a6 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 6u) * 64ul, s_base + (rn0 + 6u) * 4ul, k_l, sub32, h0, h1, a6);
            a7 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 7u) * 64ul, s_base + (rn0 + 7u) * 4ul, k_l, sub32, h0, h1, a7);
          } else {
            if (row_base + 0u < p.h) a0 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 0u) * 64ul, s_base + (rn0 + 0u) * 4ul, k_l, sub32, h0, h1, a0);
            if (row_base + 1u < p.h) a1 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 1u) * 64ul, s_base + (rn0 + 1u) * 4ul, k_l, sub32, h0, h1, a1);
            if (row_base + 2u < p.h) a2 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 2u) * 64ul, s_base + (rn0 + 2u) * 4ul, k_l, sub32, h0, h1, a2);
            if (row_base + 3u < p.h) a3 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 3u) * 64ul, s_base + (rn0 + 3u) * 4ul, k_l, sub32, h0, h1, a3);
            if (row_base + 4u < p.h) a4 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 4u) * 64ul, s_base + (rn0 + 4u) * 4ul, k_l, sub32, h0, h1, a4);
            if (row_base + 5u < p.h) a5 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 5u) * 64ul, s_base + (rn0 + 5u) * 4ul, k_l, sub32, h0, h1, a5);
            if (row_base + 6u < p.h) a6 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 6u) * 64ul, s_base + (rn0 + 6u) * 4ul, k_l, sub32, h0, h1, a6);
            if (row_base + 7u < p.h) a7 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, c_base + (rn0 + 7u) * 64ul, s_base + (rn0 + 7u) * 4ul, k_l, sub32, h0, h1, a7);
          }
        }
      }
    } else {
      const ulong row0 = ulong(row_base);
      const bool do_fwht =
          (slot < p.top_k) ? ((p.flags & 4u) != 0u) : ((p.flags & 8u) != 0u);
      for (uint blk = 0u; blk < blocks; blk += 2u) {
        const uint cur_blk = blk + b_sub;
        float4 xf0 = float4(0.0f);
        float4 xf1 = float4(0.0f);
        if (cur_blk < blocks) {
          device const bfloat4 *xb =
              reinterpret_cast<device const bfloat4 *>(xm + cur_blk * 128u + k_l * 8u);
          xf0 = float4(xb[0]);
          xf1 = float4(xb[1]);
        }
        if (do_fwht) {
          omlx_m5_fwht64_sub8(xf0, xf1, lane);
        }
        if (cur_blk < blocks) {
          const half4 h0 = half4(xf0);
          const half4 h1 = half4(xf1);
          const ulong cb = cur_blk * 64ul;
          const ulong sb = cur_blk * 4ul;
          if (full8) {
            a0 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 0ul) * s_c_stride + cb, (row0 + 0ul) * s_s_stride + sb, k_l, sub32, h0, h1, a0);
            a1 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 1ul) * s_c_stride + cb, (row0 + 1ul) * s_s_stride + sb, k_l, sub32, h0, h1, a1);
            a2 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 2ul) * s_c_stride + cb, (row0 + 2ul) * s_s_stride + sb, k_l, sub32, h0, h1, a2);
            a3 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 3ul) * s_c_stride + cb, (row0 + 3ul) * s_s_stride + sb, k_l, sub32, h0, h1, a3);
            a4 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 4ul) * s_c_stride + cb, (row0 + 4ul) * s_s_stride + sb, k_l, sub32, h0, h1, a4);
            a5 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 5ul) * s_c_stride + cb, (row0 + 5ul) * s_s_stride + sb, k_l, sub32, h0, h1, a5);
            a6 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 6ul) * s_c_stride + cb, (row0 + 6ul) * s_s_stride + sb, k_l, sub32, h0, h1, a6);
            a7 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 7ul) * s_c_stride + cb, (row0 + 7ul) * s_s_stride + sb, k_l, sub32, h0, h1, a7);
          } else {
            if (row_base + 0u < p.h) a0 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 0ul) * s_c_stride + cb, (row0 + 0ul) * s_s_stride + sb, k_l, sub32, h0, h1, a0);
            if (row_base + 1u < p.h) a1 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 1ul) * s_c_stride + cb, (row0 + 1ul) * s_s_stride + sb, k_l, sub32, h0, h1, a1);
            if (row_base + 2u < p.h) a2 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 2ul) * s_c_stride + cb, (row0 + 2ul) * s_s_stride + sb, k_l, sub32, h0, h1, a2);
            if (row_base + 3u < p.h) a3 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 3ul) * s_c_stride + cb, (row0 + 3ul) * s_s_stride + sb, k_l, sub32, h0, h1, a3);
            if (row_base + 4u < p.h) a4 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 4ul) * s_c_stride + cb, (row0 + 4ul) * s_s_stride + sb, k_l, sub32, h0, h1, a4);
            if (row_base + 5u < p.h) a5 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 5ul) * s_c_stride + cb, (row0 + 5ul) * s_s_stride + sb, k_l, sub32, h0, h1, a5);
            if (row_base + 6u < p.h) a6 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 6ul) * s_c_stride + cb, (row0 + 6ul) * s_s_stride + sb, k_l, sub32, h0, h1, a6);
            if (row_base + 7u < p.h) a7 = moe9_mxfp4_row_dot_acc(shared_dc, shared_ds, (row0 + 7ul) * s_c_stride + cb, (row0 + 7ul) * s_s_stride + sb, k_l, sub32, h0, h1, a7);
          }
        }
      }
    }

    const float v0 = simd_sum(a0);
    const float v1 = simd_sum(a1);
    const float v2 = simd_sum(a2);
    const float v3 = simd_sum(a3);
    const float v4 = simd_sum(a4);
    const float v5 = simd_sum(a5);
    const float v6 = simd_sum(a6);
    const float v7 = simd_sum(a7);
    if (lane == 0u) {
      if (slot < p.top_k) {
        pr0 += float(bfloat(v0 * w_score));
        pr1 += float(bfloat(v1 * w_score));
        pr2 += float(bfloat(v2 * w_score));
        pr3 += float(bfloat(v3 * w_score));
        pr4 += float(bfloat(v4 * w_score));
        pr5 += float(bfloat(v5 * w_score));
        pr6 += float(bfloat(v6 * w_score));
        pr7 += float(bfloat(v7 * w_score));
      } else {
        tg_shared[0] = float(bfloat(v0));
        tg_shared[1] = float(bfloat(v1));
        tg_shared[2] = float(bfloat(v2));
        tg_shared[3] = float(bfloat(v3));
        tg_shared[4] = float(bfloat(v4));
        tg_shared[5] = float(bfloat(v5));
        tg_shared[6] = float(bfloat(v6));
        tg_shared[7] = float(bfloat(v7));
      }
    }
  }

  if (lane == 0u) {
    tg_red[sg][0] = pr0;
    tg_red[sg][1] = pr1;
    tg_red[sg][2] = pr2;
    tg_red[sg][3] = pr3;
    tg_red[sg][4] = pr4;
    tg_red[sg][5] = pr5;
    tg_red[sg][6] = pr6;
    tg_red[sg][7] = pr7;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if constexpr (ADD_SQ) {
    if (sg == 0u) {
      float sq_lane = 0.0f;
      if (lane < 8u && row_base + lane < p.h) {
        const uint col = row_base + lane;
        const bfloat routed_bf16 =
            bfloat(tg_red[0][lane] + tg_red[1][lane] + tg_red[2][lane]);
        const float shared_scale = 1.0f / (1.0f + exp(-gate_logit));
        const bfloat delta_bf =
            bfloat(float(routed_bf16) + shared_scale * tg_shared[lane]);
        const ulong row_off = ulong(b) * p.h + col;
        const bfloat h_new = bfloat(float(out_buf[row_off]) + float(delta_bf));
        out_buf[row_off] = h_new;
        const float hf = float(h_new);
        sq_lane = hf * hf;
      }
      const float tg_sq = simd_sum(sq_lane);
      if (lane == 0u) {
        atomic_fetch_add_explicit(&rms_sq_buf[(p.flags >> 8u) * 8u + b], tg_sq,
                                  memory_order_relaxed);
      }
    }
  } else {
    (void)rms_sq_buf;
    if (sg == 0u && lane < 8u && row_base + lane < p.h) {
      const uint col = row_base + lane;
      const bfloat routed_bf16 =
          bfloat(tg_red[0][lane] + tg_red[1][lane] + tg_red[2][lane]);
      const float shared_scale = 1.0f / (1.0f + exp(-gate_logit));
      const float delta_f =
          float(routed_bf16) + shared_scale * tg_shared[lane];
      const ulong row_off = ulong(b) * p.h + col;
      if ((p.flags & 2u) != 0u) {
        out_buf[row_off] = bfloat(float(norm_or_residual[row_off]) + delta_f);
      } else {
        out_buf[row_off] = bfloat(delta_f);
      }
    }
  }
}

[[max_total_threads_per_threadgroup(96)]]
kernel void omlx_m5_moe9_mxfp4_kl16_down_sk3_r8_bt(
    device const bfloat *mid_in [[buffer(0)]],
    device const uint *route_ids [[buffer(1)]],
    device const bfloat *route_scores [[buffer(2)]],
    device const bfloat *shared_gate_buf [[buffer(3)]],
    device const bfloat *norm_or_residual [[buffer(4)]],
    device const uchar *routed_dc [[buffer(5)]],
    device const uchar *routed_ds [[buffer(6)]],
    device const uchar *shared_dc [[buffer(7)]],
    device const uchar *shared_ds [[buffer(8)]],
    device bfloat *delta_out [[buffer(9)]],
    constant Moe9GemmBtParams &p [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float sg_gate_part[3];
  threadgroup float tg_red[3][8];
  threadgroup float tg_shared[8];
  moe9_mxfp4_kl16_down_sk3_r8_bt_impl<false>(
      mid_in, route_ids, route_scores, shared_gate_buf, norm_or_residual,
      routed_dc, routed_ds, shared_dc, shared_ds, delta_out, nullptr,
      sg_gate_part, tg_red, tg_shared, p, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(96)]]
kernel void omlx_m5_moe9_mxfp4_kl16_down_sk3_r8_add_sq_bt(
    device const bfloat *mid_in [[buffer(0)]],
    device const uint *route_ids [[buffer(1)]],
    device const bfloat *route_scores [[buffer(2)]],
    device const bfloat *shared_gate_buf [[buffer(3)]],
    device const bfloat *norm_in [[buffer(4)]],
    device const uchar *routed_dc [[buffer(5)]],
    device const uchar *routed_ds [[buffer(6)]],
    device const uchar *shared_dc [[buffer(7)]],
    device const uchar *shared_ds [[buffer(8)]],
    device bfloat *hidden_inout [[buffer(9)]],
    device atomic_float *rms_sq_buf [[buffer(10)]],
    constant Moe9GemmBtParams &p [[buffer(11)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float sg_gate_part[3];
  threadgroup float tg_red[3][8];
  threadgroup float tg_shared[8];
  moe9_mxfp4_kl16_down_sk3_r8_bt_impl<true>(
      mid_in, route_ids, route_scores, shared_gate_buf, norm_in,
      routed_dc, routed_ds, shared_dc, shared_ds, hidden_inout, rms_sq_buf,
      sg_gate_part, tg_red, tg_shared, p, lane, sg, tg);
}
