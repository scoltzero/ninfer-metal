#include "fp4_codec.h"
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

struct MoeParams {
  uint experts;
  uint top_k;
  uint output_features;
  uint input_features;
  uint padded_input_features;
  uint tokens;
  uint apply_scores;
  uint assignment_input;
};

inline float moe_decode_e2m1(uint code) { return omlx_m5_decode_fp4(code); }

inline float moe_decode_e8m0(uchar code) {
  const uint bits = code == 0 ? 0x00400000u : uint(code) << 23;
  return as_type<float>(bits);
}

template <uint Bits>
inline uint moe_affine_code(device const uchar *base, device const uchar *high,
                            uint column) {
  if constexpr (Bits == 8) {
    return uint(base[column]);
  } else {
    const uint low = (uint(base[column / 2]) >> ((column & 1u) * 4u)) & 0x0fu;
    if constexpr (Bits == 5) {
      return low | (((uint(high[column / 8]) >> (column & 7u)) & 1u) << 4u);
    } else if constexpr (Bits == 6) {
      return low |
             (((uint(high[column / 4]) >> ((column & 3u) * 2u)) & 3u) << 4u);
    } else {
      return low;
    }
  }
}

kernel void omlx_m5_moe_router_top8(device const bfloat *probabilities
                                    [[buffer(0)]],
                                    device uint *indices [[buffer(1)]],
                                    device bfloat *scores [[buffer(2)]],
                                    constant MoeParams &params [[buffer(3)]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint3 tg_position
                                    [[threadgroup_position_in_grid]]) {
  constexpr uint top_k = 8;
  const uint token = tg_position.y;
  if (token >= params.tokens)
    return;
  const uint per_lane = min(params.experts / 32u, 8u);
  device const bfloat *row = probabilities + ulong(token) * params.experts;
  float values[8];
  bool taken[8];
  for (uint index = 0; index < per_lane; ++index) {
    values[index] = float(row[lane * per_lane + index]);
    taken[index] = false;
  }
  float selected_values[top_k];
  uint selected_indices[top_k];
  for (uint selected = 0; selected < top_k; ++selected) {
    int tier = -1;
    float best = -INFINITY;
    uint best_local = 0;
    for (uint index = 0; index < per_lane; ++index) {
      if (taken[index])
        continue;
      const int candidate_tier = isnan(values[index]) ? 0 : 1;
      const float candidate_val =
          candidate_tier == 1 ? values[index] : -INFINITY;
      if (candidate_tier > tier ||
          (candidate_tier == tier && candidate_val >= best)) {
        tier = candidate_tier;
        best = candidate_val;
        best_local = index;
      }
    }
    const int global_tier = simd_max(tier);
    if (global_tier < 0) {
      selected_values[selected] = 0.0f;
      selected_indices[selected] = 0u;
      continue;
    }
    const bool in_tier = (tier == global_tier);
    const float global_best = simd_max(in_tier ? best : -INFINITY);
    const bool is_best = in_tier && (best == global_best);
    const uint best_index = lane * per_lane + best_local;
    const uint global_index = simd_max(is_best ? best_index : 0u);
    if (is_best && best_index == global_index) {
      taken[best_local] = true;
    }
    selected_values[selected] = global_tier == 1 ? global_best : 0.0f;
    selected_indices[selected] = global_index;
  }
  if (lane == 0) {
    float total = 0.0f;
    for (uint selected = 0; selected < top_k; ++selected) {
      total += selected_values[selected];
    }
    const bool valid_total = isfinite(total) && total > 0.0f;
    const float inverse = valid_total ? 1.0f / total : 0.0f;
    for (uint selected = 0; selected < top_k; ++selected) {
      indices[ulong(token) * top_k + selected] = selected_indices[selected];
      scores[ulong(token) * top_k + selected] =
          valid_total && isfinite(selected_values[selected])
              ? bfloat(selected_values[selected] * inverse)
              : bfloat(0.0f);
    }
  }
}

template <uint GroupSize>
inline void moe_group_sum(device const bfloat *input, device float *group_sums,
                          constant MoeParams &params, uint lane,
                          uint input_row_group) {
  const uint groups = params.input_features / GroupSize;
  const uint input_row = input_row_group / groups;
  if (input_row >=
      params.tokens * (params.assignment_input != 0 ? params.top_k : 1))
    return;
  const uint group = input_row_group - input_row * groups;
  float partial = 0.0f;
  const uint begin = group * GroupSize;
  for (uint column = begin + lane; column < begin + GroupSize; column += 32) {
    partial += float(input[ulong(input_row) * params.input_features + column]);
  }
  partial = simd_sum(partial);
  if (lane == 0) {
    group_sums[input_row_group] = partial;
  }
}

template <uint Bits, uint GroupSize>
inline float moe_affine_dot_row(
    device const bfloat *in_row, device const uchar *base,
    device const uchar *high, device const bfloat *row_scales,
    device const bfloat *row_biases, device const float *row_gsums,
    uint input_features, uint lane) {
  const uint logical_groups = input_features / GroupSize;
  float dot = 0.0f;
  float bias_acc = 0.0f;
  for (uint group = 0; group < logical_groups; ++group) {
    const float s = float(row_scales[group]);
    if constexpr (GroupSize == 64) {
      float w0, w1;
      if constexpr (Bits == 8) {
        const uchar2 b =
            reinterpret_cast<device const uchar2 *>(base + ulong(group) * 64u)[lane];
        w0 = s * float(b.x);
        w1 = s * float(b.y);
      } else {
        const uint b = uint(base[ulong(group) * 32u + lane]);
        uint q0 = b & 0x0fu;
        uint q1 = b >> 4u;
        if constexpr (Bits == 5) {
          const uint h = uint(high[ulong(group) * 8u + (lane >> 2u)]) >>
                         ((lane & 3u) * 2u);
          q0 |= (h & 1u) << 4u;
          q1 |= ((h >> 1u) & 1u) << 4u;
        } else if constexpr (Bits == 6) {
          const uint h = uint(high[ulong(group) * 16u + (lane >> 1u)]) >>
                         ((lane & 1u) * 4u);
          q0 |= (h & 3u) << 4u;
          q1 |= ((h >> 2u) & 3u) << 4u;
        }
        w0 = s * float(q0);
        w1 = s * float(q1);
      }
      const bfloat2 xv = reinterpret_cast<device const bfloat2 *>(
          in_row + group * 64u + lane * 2u)[0];
      dot = fma(float(xv.x), w0, dot);
      dot = fma(float(xv.y), w1, dot);
    } else {
      float w0, w1, w2, w3;
      if constexpr (Bits == 8) {
        const uchar4 b =
            reinterpret_cast<device const uchar4 *>(base + ulong(group) * 128u)[lane];
        w0 = s * float(b.x);
        w1 = s * float(b.y);
        w2 = s * float(b.z);
        w3 = s * float(b.w);
      } else {
        const ushort b2 = reinterpret_cast<device const ushort *>(
            base + ulong(group) * 64u)[lane];
        uint q0 = uint(b2) & 0x0fu;
        uint q1 = (uint(b2) >> 4u) & 0x0fu;
        uint q2 = (uint(b2) >> 8u) & 0x0fu;
        uint q3 = (uint(b2) >> 12u) & 0x0fu;
        if constexpr (Bits == 5) {
          const uint h = uint(high[ulong(group) * 16u + (lane >> 1u)]) >>
                         ((lane & 1u) * 4u);
          q0 |= (h & 1u) << 4u;
          q1 |= ((h >> 1u) & 1u) << 4u;
          q2 |= ((h >> 2u) & 1u) << 4u;
          q3 |= ((h >> 3u) & 1u) << 4u;
        } else if constexpr (Bits == 6) {
          const uint h = uint(high[ulong(group) * 32u + lane]);
          q0 |= (h & 3u) << 4u;
          q1 |= ((h >> 2u) & 3u) << 4u;
          q2 |= ((h >> 4u) & 3u) << 4u;
          q3 |= ((h >> 6u) & 3u) << 4u;
        }
        w0 = s * float(q0);
        w1 = s * float(q1);
        w2 = s * float(q2);
        w3 = s * float(q3);
      }
      const bfloat4 xv = reinterpret_cast<device const bfloat4 *>(
          in_row + group * 128u + lane * 4u)[0];
      dot = fma(float(xv.x), w0, dot);
      dot = fma(float(xv.y), w1, dot);
      dot = fma(float(xv.z), w2, dot);
      dot = fma(float(xv.w), w3, dot);
    }
    if (lane == 0) {
      bias_acc = fma(float(row_biases[group]), row_gsums[group], bias_acc);
    }
  }
  return simd_sum(dot) + bias_acc;
}

template <uint Bits, uint GroupSize, uint RowsPerThreadgroup>
inline void moe_affine_assignment(
    device const bfloat *input, device const uint *indices,
    device const bfloat *route_scores, device const uchar *base_codes,
    device const uchar *high_codes, device const bfloat *scales,
    device const bfloat *biases, device const float *group_sums,
    device bfloat *output, constant MoeParams &params, uint lane,
    uint simdgroup, uint threadgroup_index) {
  const uint linear = threadgroup_index * RowsPerThreadgroup + simdgroup;
  const uint assignments = params.tokens * params.top_k;
  const ulong total_rows = ulong(assignments) * params.output_features;
  if (ulong(linear) >= total_rows) {
    return;
  }
  const uint assignment = linear / params.output_features;
  const uint row = linear - assignment * params.output_features;
  const uint token = assignment / params.top_k;
  const uint input_row = params.assignment_input != 0 ? assignment : token;
  const uint expert = indices[assignment];
  if (expert >= params.experts) {
    if (lane == 0) {
      output[linear] = bfloat(0.0f);
    }
    return;
  }
  const ulong flattened_row = ulong(expert) * params.output_features + row;
  const ulong base_row_bytes = Bits == 8 ? params.padded_input_features
                                         : params.padded_input_features / 2;
  const ulong high_row_bytes =
      Bits == 5 ? params.padded_input_features / 8
                : (Bits == 6 ? params.padded_input_features / 4 : 0);
  const ulong groups_per_row = params.padded_input_features / GroupSize;
  const uint logical_groups = params.input_features / GroupSize;
  float result = moe_affine_dot_row<Bits, GroupSize>(
      input + ulong(input_row) * params.input_features,
      base_codes + flattened_row * base_row_bytes,
      high_codes + flattened_row * high_row_bytes,
      scales + flattened_row * groups_per_row,
      biases + flattened_row * groups_per_row,
      group_sums + ulong(input_row) * logical_groups, params.input_features,
      lane);
  if (lane == 0) {
    if (params.apply_scores != 0) {
      result *= float(route_scores[assignment]);
    }
    output[linear] = bfloat(result);
  }
}

inline float moe_mxfp4_dot_row(device const bfloat *in_row,
                               device const uchar *codes,
                               device const uchar *scales,
                               uint input_features, uint lane) {
  constexpr uint group_size = 32;
  const uint logical_groups = input_features / group_size;
  float acc = 0.0f;
  const uint sg_off = lane >> 2u;
  const uint sl = lane & 3u;
  const uint col_lane_off = sg_off * 32u + sl * 8u;
  uint group = 0;
  for (; group + 7u < logical_groups; group += 8u) {
    const uint packed32 =
        reinterpret_cast<device const uint *>(codes + ulong(group) * 16u)[lane];
    const float s = moe_decode_e8m0(scales[group + sg_off]);
    const float w0 = moe_decode_e2m1(packed32 & 0x0fu) * s;
    const float w1 = moe_decode_e2m1((packed32 >> 4u) & 0x0fu) * s;
    const float w2 = moe_decode_e2m1((packed32 >> 8u) & 0x0fu) * s;
    const float w3 = moe_decode_e2m1((packed32 >> 12u) & 0x0fu) * s;
    const float w4 = moe_decode_e2m1((packed32 >> 16u) & 0x0fu) * s;
    const float w5 = moe_decode_e2m1((packed32 >> 20u) & 0x0fu) * s;
    const float w6 = moe_decode_e2m1((packed32 >> 24u) & 0x0fu) * s;
    const float w7 = moe_decode_e2m1((packed32 >> 28u) & 0x0fu) * s;
    device const bfloat4 *xv = reinterpret_cast<device const bfloat4 *>(
        in_row + group * group_size + col_lane_off);
    const bfloat4 a0 = xv[0];
    const bfloat4 a1 = xv[1];
    acc = fma(float(a0.x), w0, acc);
    acc = fma(float(a0.y), w1, acc);
    acc = fma(float(a0.z), w2, acc);
    acc = fma(float(a0.w), w3, acc);
    acc = fma(float(a1.x), w4, acc);
    acc = fma(float(a1.y), w5, acc);
    acc = fma(float(a1.z), w6, acc);
    acc = fma(float(a1.w), w7, acc);
  }
  const uint shift = (lane & 1u) * 4u;
  const uint byte_in_group = lane >> 1u;
  for (; group < logical_groups; ++group) {
    const uint column = group * group_size + lane;
    const uchar packed = codes[ulong(group) * 16u + byte_in_group];
    const uint code = (uint(packed) >> shift) & 0x0fu;
    const float w = moe_decode_e2m1(code) * moe_decode_e8m0(scales[group]);
    acc = fma(float(in_row[column]), w, acc);
  }
  return simd_sum(acc);
}

template <uint RowsPerThreadgroup>
inline void moe_mxfp4_w4a16_assignment(device const bfloat *input,
                                       device const uint *indices,
                                       device const bfloat *route_scores,
                                       device const uchar *weight_codes,
                                       device const uchar *weight_scales,
                                       device bfloat *output,
                                       constant MoeParams &params, uint lane,
                                       uint simdgroup, uint threadgroup_index) {
  constexpr uint group_size = 32;
  const uint linear = threadgroup_index * RowsPerThreadgroup + simdgroup;
  const uint assignments = params.tokens * params.top_k;
  const ulong total_rows = ulong(assignments) * params.output_features;
  if (ulong(linear) >= total_rows) {
    return;
  }
  const uint assignment = linear / params.output_features;
  const uint row = linear - assignment * params.output_features;
  const uint token = assignment / params.top_k;
  const uint input_row = params.assignment_input != 0 ? assignment : token;
  const uint expert = indices[assignment];
  if (expert >= params.experts) {
    if (lane == 0) {
      output[linear] = bfloat(0.0f);
    }
    return;
  }
  const ulong flattened_row = ulong(expert) * params.output_features + row;
  const ulong code_row_bytes = params.padded_input_features / 2;
  const ulong groups_per_row = params.padded_input_features / group_size;
  float result = moe_mxfp4_dot_row(
      input + ulong(input_row) * params.input_features,
      weight_codes + flattened_row * code_row_bytes,
      weight_scales + flattened_row * groups_per_row, params.input_features,
      lane);
  if (lane == 0) {
    if (params.apply_scores != 0) {
      result *= float(route_scores[assignment]);
    }
    output[linear] = bfloat(result);
  }
}

// Fused MoE gate + up + SwiGLU for MXFP4 (4 output rows/SG = 16 rows/TG,
// reuses x in registers 8x across 4 gate + 4 up rows, 0-ULP vs _fallback!)
inline float moe_simd_reduce4_at_lane(float q0, float q1, float q2, float q3,
                                      uint lane) {
  const float p0 = q0 + simd_shuffle_xor(q0, 1u);
  const float p1 = q1 + simd_shuffle_xor(q1, 1u);
  float ab = ((lane & 1u) != 0u) ? p1 : p0;
  ab += simd_shuffle_xor(ab, 2u);
  const float p2 = q2 + simd_shuffle_xor(q2, 1u);
  const float p3 = q3 + simd_shuffle_xor(q3, 1u);
  float cd = ((lane & 1u) != 0u) ? p3 : p2;
  cd += simd_shuffle_xor(cd, 2u);
  float v = ((lane & 2u) != 0u) ? cd : ab;
  v += simd_shuffle_xor(v, 4u);
  v += simd_shuffle_xor(v, 8u);
  v += simd_shuffle_xor(v, 16u);
  return v;
}

kernel void omlx_m5_mxfp4_moe_w4a16_swiglu_grouped(
    device const bfloat *input [[buffer(0)]],
    device const uint *indices [[buffer(1)]],
    device const uchar *g_codes [[buffer(2)]],
    device const uchar *g_scales [[buffer(3)]],
    device const uchar *u_codes [[buffer(4)]],
    device const uchar *u_scales [[buffer(5)]],
    device bfloat *output [[buffer(6)]],
    constant MoeParams &params [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  constexpr uint group_size = 32;
  const uint tiles_per_assign = (params.output_features + 15u) / 16u;
  const uint assignments = params.tokens * params.top_k;
  if (tiles_per_assign == 0u || ulong(tg_position.x) >= ulong(assignments) * tiles_per_assign) {
    return;
  }
  const uint assignment = tg_position.x / tiles_per_assign;
  const uint row_base = (tg_position.x - assignment * tiles_per_assign) * 16u + simdgroup * 4u;
  if (row_base >= params.output_features) {
    return;
  }
  const uint valid_r = min(4u, params.output_features - row_base);
  const uint token = assignment / params.top_k;
  const uint expert = indices[assignment];
  const ulong out_base = ulong(assignment) * params.output_features + row_base;
  if (expert >= params.experts) {
    if (lane < valid_r) {
      output[out_base + lane] = bfloat(0.0f);
    }
    return;
  }
  const ulong flattened_base = ulong(expert) * params.output_features + row_base;
  const ulong code_row_bytes = params.padded_input_features / 2;
  const ulong groups_per_row = params.padded_input_features / group_size;
  const uint logical_groups = params.input_features / group_size;
  device const bfloat *in_row = input + ulong(token) * params.input_features;

  float g_acc[4] = {0.f, 0.f, 0.f, 0.f};
  float u_acc[4] = {0.f, 0.f, 0.f, 0.f};
  const uint sg_off = lane >> 2u;
  const uint sl = lane & 3u;
  const uint col_lane_off = sg_off * 32u + sl * 8u;
  uint group = 0;
  for (; group + 7u < logical_groups; group += 8u) {
    device const bfloat4 *xv = reinterpret_cast<device const bfloat4 *>(
        in_row + group * group_size + col_lane_off);
    const bfloat4 a0 = xv[0];
    const bfloat4 a1 = xv[1];
    const float f0 = float(a0.x), f1 = float(a0.y), f2 = float(a0.z), f3 = float(a0.w);
    const float f4 = float(a1.x), f5 = float(a1.y), f6 = float(a1.z), f7 = float(a1.w);

    for (uint r = 0; r < valid_r; ++r) {
      device const uchar *gc = g_codes + (flattened_base + r) * code_row_bytes;
      device const uchar *gs = g_scales + (flattened_base + r) * groups_per_row;
      device const uchar *uc = u_codes + (flattened_base + r) * code_row_bytes;
      device const uchar *us = u_scales + (flattened_base + r) * groups_per_row;
      const uint gp = reinterpret_cast<device const uint *>(gc + ulong(group) * 16u)[lane];
      const float gsc = moe_decode_e8m0(gs[group + sg_off]);
      g_acc[r] = fma(f0, moe_decode_e2m1(gp & 0x0fu) * gsc, g_acc[r]);
      g_acc[r] = fma(f1, moe_decode_e2m1((gp >> 4u) & 0x0fu) * gsc, g_acc[r]);
      g_acc[r] = fma(f2, moe_decode_e2m1((gp >> 8u) & 0x0fu) * gsc, g_acc[r]);
      g_acc[r] = fma(f3, moe_decode_e2m1((gp >> 12u) & 0x0fu) * gsc, g_acc[r]);
      g_acc[r] = fma(f4, moe_decode_e2m1((gp >> 16u) & 0x0fu) * gsc, g_acc[r]);
      g_acc[r] = fma(f5, moe_decode_e2m1((gp >> 20u) & 0x0fu) * gsc, g_acc[r]);
      g_acc[r] = fma(f6, moe_decode_e2m1((gp >> 24u) & 0x0fu) * gsc, g_acc[r]);
      g_acc[r] = fma(f7, moe_decode_e2m1((gp >> 28u) & 0x0fu) * gsc, g_acc[r]);

      const uint up = reinterpret_cast<device const uint *>(uc + ulong(group) * 16u)[lane];
      const float usc = moe_decode_e8m0(us[group + sg_off]);
      u_acc[r] = fma(f0, moe_decode_e2m1(up & 0x0fu) * usc, u_acc[r]);
      u_acc[r] = fma(f1, moe_decode_e2m1((up >> 4u) & 0x0fu) * usc, u_acc[r]);
      u_acc[r] = fma(f2, moe_decode_e2m1((up >> 8u) & 0x0fu) * usc, u_acc[r]);
      u_acc[r] = fma(f3, moe_decode_e2m1((up >> 12u) & 0x0fu) * usc, u_acc[r]);
      u_acc[r] = fma(f4, moe_decode_e2m1((up >> 16u) & 0x0fu) * usc, u_acc[r]);
      u_acc[r] = fma(f5, moe_decode_e2m1((up >> 20u) & 0x0fu) * usc, u_acc[r]);
      u_acc[r] = fma(f6, moe_decode_e2m1((up >> 24u) & 0x0fu) * usc, u_acc[r]);
      u_acc[r] = fma(f7, moe_decode_e2m1((up >> 28u) & 0x0fu) * usc, u_acc[r]);
    }
  }
  const uint shift = (lane & 1u) * 4u;
  const uint byte_in_group = lane >> 1u;
  for (; group < logical_groups; ++group) {
    const uint column = group * group_size + lane;
    const float xv = float(in_row[column]);
    for (uint r = 0; r < valid_r; ++r) {
      device const uchar *gc = g_codes + (flattened_base + r) * code_row_bytes;
      device const uchar *gs = g_scales + (flattened_base + r) * groups_per_row;
      device const uchar *uc = u_codes + (flattened_base + r) * code_row_bytes;
      device const uchar *us = u_scales + (flattened_base + r) * groups_per_row;
      const uint gcode = (uint(gc[ulong(group) * 16u + byte_in_group]) >> shift) & 0x0fu;
      const uint ucode = (uint(uc[ulong(group) * 16u + byte_in_group]) >> shift) & 0x0fu;
      g_acc[r] = fma(xv, moe_decode_e2m1(gcode) * moe_decode_e8m0(gs[group]), g_acc[r]);
      u_acc[r] = fma(xv, moe_decode_e2m1(ucode) * moe_decode_e8m0(us[group]), u_acc[r]);
    }
  }
  const float rg = moe_simd_reduce4_at_lane(g_acc[0], g_acc[1], g_acc[2], g_acc[3], lane);
  const float ru = moe_simd_reduce4_at_lane(u_acc[0], u_acc[1], u_acc[2], u_acc[3], lane);
  if (lane < valid_r) {
    const float gv = float(bfloat(rg));
    const float uv = float(bfloat(ru));
    output[out_base + lane] = bfloat((gv / (1.0f + exp(-gv))) * uv);
  }
}

// Fused MoE down_proj + topk weighted reduce for MXFP4 (8 output rows per TG):
// 8 simdgroups (256 threads) per (token, 8 out_rows) -> 1 simdgroup per active expert.
// Reuses activation in registers across 8 output rows and folds 8 reductions via butterfly!
inline float moe_simd_reduce8_at_lane(float q0, float q1, float q2, float q3,
                                      float q4, float q5, float q6, float q7,
                                      uint lane) {
  const float p0 = q0 + simd_shuffle_xor(q0, 1u);
  const float p1 = q1 + simd_shuffle_xor(q1, 1u);
  float ab = ((lane & 1u) != 0u) ? p1 : p0;
  ab += simd_shuffle_xor(ab, 2u);
  const float p2 = q2 + simd_shuffle_xor(q2, 1u);
  const float p3 = q3 + simd_shuffle_xor(q3, 1u);
  float cd = ((lane & 1u) != 0u) ? p3 : p2;
  cd += simd_shuffle_xor(cd, 2u);
  float abcd = ((lane & 2u) != 0u) ? cd : ab;
  abcd += simd_shuffle_xor(abcd, 4u);

  const float p4 = q4 + simd_shuffle_xor(q4, 1u);
  const float p5 = q5 + simd_shuffle_xor(q5, 1u);
  float ef = ((lane & 1u) != 0u) ? p5 : p4;
  ef += simd_shuffle_xor(ef, 2u);
  const float p6 = q6 + simd_shuffle_xor(q6, 1u);
  const float p7 = q7 + simd_shuffle_xor(q7, 1u);
  float gh = ((lane & 1u) != 0u) ? p7 : p6;
  gh += simd_shuffle_xor(gh, 2u);
  float efgh = ((lane & 2u) != 0u) ? gh : ef;
  efgh += simd_shuffle_xor(efgh, 4u);

  float v = ((lane & 4u) != 0u) ? efgh : abcd;
  v += simd_shuffle_xor(v, 8u);
  v += simd_shuffle_xor(v, 16u);
  return v;
}

kernel void omlx_m5_mxfp4_moe_w4a16_down_reduce(
    device const bfloat *input [[buffer(0)]],
    device const uint *indices [[buffer(1)]],
    device const bfloat *route_scores [[buffer(2)]],
    device const uchar *weight_codes [[buffer(3)]],
    device const uchar *weight_scales [[buffer(4)]],
    device bfloat *output [[buffer(5)]],
    constant MoeParams &params [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  constexpr uint group_size = 32;
  const uint tiles_per_tok = (params.output_features + 7u) / 8u;
  const uint tile_idx = tg_position.x;
  if (ulong(tile_idx) >= ulong(params.tokens) * tiles_per_tok) {
    return;
  }
  const uint token = tile_idx / tiles_per_tok;
  const uint row_base = (tile_idx - token * tiles_per_tok) * 8u;
  const uint valid_rows = min(8u, params.output_features - row_base);
  const uint k_slot = simdgroup;
  threadgroup float partials[64]; // [8 experts][8 rows]

  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float score = 0.0f;
  if (k_slot < params.top_k) {
    const uint assignment = token * params.top_k + k_slot;
    const uint expert = indices[assignment];
    if (expert < params.experts) {
      score = float(route_scores[assignment]);
      const ulong code_row_bytes = params.padded_input_features / 2;
      const ulong groups_per_row = params.padded_input_features / group_size;
      const uint logical_groups = params.input_features / group_size;
      device const bfloat *in_row = input + ulong(assignment) * params.input_features;
      const ulong exp_base = ulong(expert) * params.output_features + row_base;

      const uint sg_off = lane >> 2u;
      const uint sl = lane & 3u;
      const uint col_lane_off = sg_off * 32u + sl * 8u;
      uint group = 0;
      for (; group + 7u < logical_groups; group += 8u) {
        device const bfloat4 *xv = reinterpret_cast<device const bfloat4 *>(
            in_row + group * group_size + col_lane_off);
        const bfloat4 a0 = xv[0];
        const bfloat4 a1 = xv[1];
        const float f0 = float(a0.x), f1 = float(a0.y), f2 = float(a0.z), f3 = float(a0.w);
        const float f4 = float(a1.x), f5 = float(a1.y), f6 = float(a1.z), f7 = float(a1.w);
        for (uint r = 0; r < valid_rows; ++r) {
          device const uchar *rc = weight_codes + (exp_base + r) * code_row_bytes;
          device const uchar *rs = weight_scales + (exp_base + r) * groups_per_row;
          const uint p32 = reinterpret_cast<device const uint *>(rc + ulong(group) * 16u)[lane];
          const float s = moe_decode_e8m0(rs[group + sg_off]);
          acc[r] = fma(f0, moe_decode_e2m1(p32 & 0x0fu) * s, acc[r]);
          acc[r] = fma(f1, moe_decode_e2m1((p32 >> 4u) & 0x0fu) * s, acc[r]);
          acc[r] = fma(f2, moe_decode_e2m1((p32 >> 8u) & 0x0fu) * s, acc[r]);
          acc[r] = fma(f3, moe_decode_e2m1((p32 >> 12u) & 0x0fu) * s, acc[r]);
          acc[r] = fma(f4, moe_decode_e2m1((p32 >> 16u) & 0x0fu) * s, acc[r]);
          acc[r] = fma(f5, moe_decode_e2m1((p32 >> 20u) & 0x0fu) * s, acc[r]);
          acc[r] = fma(f6, moe_decode_e2m1((p32 >> 24u) & 0x0fu) * s, acc[r]);
          acc[r] = fma(f7, moe_decode_e2m1((p32 >> 28u) & 0x0fu) * s, acc[r]);
        }
      }
      const uint shift = (lane & 1u) * 4u;
      const uint byte_in_group = lane >> 1u;
      for (; group < logical_groups; ++group) {
        const uint column = group * group_size + lane;
        const float xv = float(in_row[column]);
        for (uint r = 0; r < valid_rows; ++r) {
          device const uchar *rc = weight_codes + (exp_base + r) * code_row_bytes;
          device const uchar *rs = weight_scales + (exp_base + r) * groups_per_row;
          const uint code = (uint(rc[ulong(group) * 16u + byte_in_group]) >> shift) & 0x0fu;
          acc[r] = fma(xv, moe_decode_e2m1(code) * moe_decode_e8m0(rs[group]), acc[r]);
        }
      }
    }
  }
  const float red = moe_simd_reduce8_at_lane(
      acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7], lane);
  if (lane < 8u && k_slot < 8u) {
    partials[k_slot * 8u + lane] =
        (k_slot < params.top_k && lane < valid_rows)
            ? float(bfloat(red * score))
            : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup == 0u && lane < valid_rows) {
    float total = 0.0f;
    for (uint k = 0; k < params.top_k; ++k) {
      total += partials[k * 8u + lane];
    }
    output[ulong(token) * params.output_features + row_base + lane] = bfloat(total);
  }
}

template <uint Bits, uint GroupSize>
inline void moe_affine_swiglu_grouped_impl(
    device const bfloat *input, device const uint *indices,
    device const uchar *g_base, device const uchar *g_high,
    device const bfloat *g_scales, device const bfloat *g_biases,
    device const uchar *u_base, device const uchar *u_high,
    device const bfloat *u_scales, device const bfloat *u_biases,
    device const float *group_sums, device bfloat *output,
    constant MoeParams &params, uint lane, uint simdgroup, uint tg_x) {
  const uint tiles_per_assign = (params.output_features + 15u) / 16u;
  const uint assignments = params.tokens * params.top_k;
  if (tiles_per_assign == 0u || ulong(tg_x) >= ulong(assignments) * tiles_per_assign) return;
  const uint assignment = tg_x / tiles_per_assign;
  const uint row_base = (tg_x - assignment * tiles_per_assign) * 16u + simdgroup * 4u;
  if (row_base >= params.output_features) return;
  const uint valid_r = min(4u, params.output_features - row_base);
  const uint token = assignment / params.top_k;
  const uint expert = indices[assignment];
  const ulong out_base = ulong(assignment) * params.output_features + row_base;
  if (expert >= params.experts) {
    if (lane < valid_r) output[out_base + lane] = bfloat(0.0f);
    return;
  }
  const ulong flattened_base = ulong(expert) * params.output_features + row_base;
  const ulong base_row_bytes = Bits == 8 ? params.padded_input_features
                                         : params.padded_input_features / 2;
  const ulong high_row_bytes =
      Bits == 5 ? params.padded_input_features / 8
                : (Bits == 6 ? params.padded_input_features / 4 : 0);
  const ulong groups_per_row = params.padded_input_features / GroupSize;
  const uint logical_groups = params.input_features / GroupSize;
  device const bfloat *in_row = input + ulong(token) * params.input_features;
  device const float *gsum_row = group_sums + ulong(token) * logical_groups;

  float g_dot[4] = {0.f, 0.f, 0.f, 0.f};
  float u_dot[4] = {0.f, 0.f, 0.f, 0.f};
  float g_bias = 0.f, u_bias = 0.f;

  for (uint group = 0; group < logical_groups; ++group) {
    if constexpr (GroupSize == 64) {
      const float2 xv = float2(reinterpret_cast<device const bfloat2 *>(
          in_row + group * 64u + lane * 2u)[0]);
      for (uint r = 0; r < valid_r; ++r) {
        const ulong frow = flattened_base + r;
        const float gs = float(g_scales[frow * groups_per_row + group]);
        const float us = float(u_scales[frow * groups_per_row + group]);
        if constexpr (Bits == 8) {
          const uchar2 gb = reinterpret_cast<device const uchar2 *>(
              g_base + frow * base_row_bytes + ulong(group) * 64u)[lane];
          const uchar2 ub = reinterpret_cast<device const uchar2 *>(
              u_base + frow * base_row_bytes + ulong(group) * 64u)[lane];
          g_dot[r] = fma(xv.x, gs * float(gb.x), g_dot[r]);
          g_dot[r] = fma(xv.y, gs * float(gb.y), g_dot[r]);
          u_dot[r] = fma(xv.x, us * float(ub.x), u_dot[r]);
          u_dot[r] = fma(xv.y, us * float(ub.y), u_dot[r]);
        } else {
          const uint gb = uint(g_base[frow * base_row_bytes + ulong(group) * 32u + lane]);
          const uint ub = uint(u_base[frow * base_row_bytes + ulong(group) * 32u + lane]);
          uint gq0 = gb & 0x0fu, gq1 = gb >> 4u;
          uint uq0 = ub & 0x0fu, uq1 = ub >> 4u;
          if constexpr (Bits == 5) {
            const uint gh = uint(g_high[frow * high_row_bytes + ulong(group) * 8u + (lane >> 2u)]) >> ((lane & 3u) * 2u);
            const uint uh = uint(u_high[frow * high_row_bytes + ulong(group) * 8u + (lane >> 2u)]) >> ((lane & 3u) * 2u);
            gq0 |= (gh & 1u) << 4u; gq1 |= ((gh >> 1u) & 1u) << 4u;
            uq0 |= (uh & 1u) << 4u; uq1 |= ((uh >> 1u) & 1u) << 4u;
          } else if constexpr (Bits == 6) {
            const uint gh = uint(g_high[frow * high_row_bytes + ulong(group) * 16u + (lane >> 1u)]) >> ((lane & 1u) * 4u);
            const uint uh = uint(u_high[frow * high_row_bytes + ulong(group) * 16u + (lane >> 1u)]) >> ((lane & 1u) * 4u);
            gq0 |= (gh & 3u) << 4u; gq1 |= ((gh >> 2u) & 3u) << 4u;
            uq0 |= (uh & 3u) << 4u; uq1 |= ((uh >> 2u) & 3u) << 4u;
          }
          g_dot[r] = fma(xv.x, gs * float(gq0), g_dot[r]);
          g_dot[r] = fma(xv.y, gs * float(gq1), g_dot[r]);
          u_dot[r] = fma(xv.x, us * float(uq0), u_dot[r]);
          u_dot[r] = fma(xv.y, us * float(uq1), u_dot[r]);
        }
      }
    } else {
      const float4 xv = float4(reinterpret_cast<device const bfloat4 *>(
          in_row + group * 128u + lane * 4u)[0]);
      for (uint r = 0; r < valid_r; ++r) {
        const ulong frow = flattened_base + r;
        const float gs = float(g_scales[frow * groups_per_row + group]);
        const float us = float(u_scales[frow * groups_per_row + group]);
        if constexpr (Bits == 8) {
          const uchar4 gb = reinterpret_cast<device const uchar4 *>(
              g_base + frow * base_row_bytes + ulong(group) * 128u)[lane];
          const uchar4 ub = reinterpret_cast<device const uchar4 *>(
              u_base + frow * base_row_bytes + ulong(group) * 128u)[lane];
          g_dot[r] = fma(xv.x, gs * float(gb.x), g_dot[r]);
          g_dot[r] = fma(xv.y, gs * float(gb.y), g_dot[r]);
          g_dot[r] = fma(xv.z, gs * float(gb.z), g_dot[r]);
          g_dot[r] = fma(xv.w, gs * float(gb.w), g_dot[r]);
          u_dot[r] = fma(xv.x, us * float(ub.x), u_dot[r]);
          u_dot[r] = fma(xv.y, us * float(ub.y), u_dot[r]);
          u_dot[r] = fma(xv.z, us * float(ub.z), u_dot[r]);
          u_dot[r] = fma(xv.w, us * float(ub.w), u_dot[r]);
        } else {
          const ushort gb = reinterpret_cast<device const ushort *>(
              g_base + frow * base_row_bytes + ulong(group) * 64u)[lane];
          const ushort ub = reinterpret_cast<device const ushort *>(
              u_base + frow * base_row_bytes + ulong(group) * 64u)[lane];
          uint gq0 = uint(gb) & 0x0fu, gq1 = (uint(gb) >> 4u) & 0x0fu, gq2 = (uint(gb) >> 8u) & 0x0fu, gq3 = (uint(gb) >> 12u) & 0x0fu;
          uint uq0 = uint(ub) & 0x0fu, uq1 = (uint(ub) >> 4u) & 0x0fu, uq2 = (uint(ub) >> 8u) & 0x0fu, uq3 = (uint(ub) >> 12u) & 0x0fu;
          if constexpr (Bits == 5) {
            const uint gh = uint(g_high[frow * high_row_bytes + ulong(group) * 16u + (lane >> 1u)]) >> ((lane & 1u) * 4u);
            const uint uh = uint(u_high[frow * high_row_bytes + ulong(group) * 16u + (lane >> 1u)]) >> ((lane & 1u) * 4u);
            gq0 |= (gh & 1u) << 4u; gq1 |= ((gh >> 1u) & 1u) << 4u; gq2 |= ((gh >> 2u) & 1u) << 4u; gq3 |= ((gh >> 3u) & 1u) << 4u;
            uq0 |= (uh & 1u) << 4u; uq1 |= ((uh >> 1u) & 1u) << 4u; uq2 |= ((uh >> 2u) & 1u) << 4u; uq3 |= ((uh >> 3u) & 1u) << 4u;
          } else if constexpr (Bits == 6) {
            const uint gh = uint(g_high[frow * high_row_bytes + ulong(group) * 32u + lane]);
            const uint uh = uint(u_high[frow * high_row_bytes + ulong(group) * 32u + lane]);
            gq0 |= (gh & 3u) << 4u; gq1 |= ((gh >> 2u) & 3u) << 4u; gq2 |= ((gh >> 4u) & 3u) << 4u; gq3 |= ((gh >> 6u) & 3u) << 4u;
            uq0 |= (uh & 3u) << 4u; uq1 |= ((uh >> 2u) & 3u) << 4u; uq2 |= ((uh >> 4u) & 3u) << 4u; uq3 |= ((uh >> 6u) & 3u) << 4u;
          }
          g_dot[r] = fma(xv.x, gs * float(gq0), g_dot[r]);
          g_dot[r] = fma(xv.y, gs * float(gq1), g_dot[r]);
          g_dot[r] = fma(xv.z, gs * float(gq2), g_dot[r]);
          g_dot[r] = fma(xv.w, gs * float(gq3), g_dot[r]);
          u_dot[r] = fma(xv.x, us * float(uq0), u_dot[r]);
          u_dot[r] = fma(xv.y, us * float(uq1), u_dot[r]);
          u_dot[r] = fma(xv.z, us * float(uq2), u_dot[r]);
          u_dot[r] = fma(xv.w, us * float(uq3), u_dot[r]);
        }
      }
    }
    if (lane < valid_r) {
      const float ls = gsum_row[group];
      g_bias = fma(float(g_biases[(flattened_base + lane) * groups_per_row + group]), ls, g_bias);
      u_bias = fma(float(u_biases[(flattened_base + lane) * groups_per_row + group]), ls, u_bias);
    }
  }
  const float rg = moe_simd_reduce4_at_lane(g_dot[0], g_dot[1], g_dot[2], g_dot[3], lane);
  const float ru = moe_simd_reduce4_at_lane(u_dot[0], u_dot[1], u_dot[2], u_dot[3], lane);
  if (lane < valid_r) {
    const float gv = float(bfloat(rg + g_bias));
    const float uv = float(bfloat(ru + u_bias));
    const float silu = gv / (1.0f + exp(-gv));
    output[out_base + lane] = bfloat(silu * uv);
  }
}

template <uint Bits, uint GroupSize>
inline void moe_affine_down_reduce_impl(
    device const bfloat *input, device const uint *indices,
    device const bfloat *route_scores, device const uchar *base_codes,
    device const uchar *high_codes, device const bfloat *scales,
    device const bfloat *biases, device const float *group_sums,
    device bfloat *output, constant MoeParams &params,
    threadgroup float *partials, uint lane,
    uint simdgroup, uint tg_x) {
  const uint out_idx = tg_x;
  const ulong total_out = ulong(params.tokens) * params.output_features;
  if (ulong(out_idx) >= total_out) return;
  const uint token = out_idx / params.output_features;
  const uint row = out_idx - token * params.output_features;
  const uint k_slot = simdgroup;
  float val = 0.0f;
  if (k_slot < params.top_k) {
    const uint assignment = token * params.top_k + k_slot;
    const uint expert = indices[assignment];
    if (expert < params.experts) {
      const ulong flattened_row = ulong(expert) * params.output_features + row;
      const ulong base_row_bytes = Bits == 8 ? params.padded_input_features
                                             : params.padded_input_features / 2;
      const ulong high_row_bytes =
          Bits == 5 ? params.padded_input_features / 8
                    : (Bits == 6 ? params.padded_input_features / 4 : 0);
      const ulong groups_per_row = params.padded_input_features / GroupSize;
      const uint logical_groups = params.input_features / GroupSize;
      const float dot = moe_affine_dot_row<Bits, GroupSize>(
          input + ulong(assignment) * params.input_features,
          base_codes + flattened_row * base_row_bytes,
          high_codes + flattened_row * high_row_bytes,
          scales + flattened_row * groups_per_row,
          biases + flattened_row * groups_per_row,
          group_sums + ulong(assignment) * logical_groups,
          params.input_features, lane);
      if (lane == 0) {
        val = float(bfloat(dot * float(route_scores[assignment])));
      }
    }
  }
  if (lane == 0 && k_slot < 8u) partials[k_slot] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup == 0 && lane == 0) {
    float total = 0.0f;
    for (uint k = 0; k < params.top_k; ++k) total += partials[k];
    output[out_idx] = bfloat(total);
  }
}

template <uint RowsPerThreadgroup>
inline void moe_mxfp4_w4a4_assignment(
    device const uchar *activation_codes, device const uchar *activation_scales,
    device const uint *indices, device const bfloat *route_scores,
    device const uchar *weight_codes, device const uchar *weight_scales,
    device bfloat *output, constant MoeParams &params, uint lane,
    uint simdgroup, uint threadgroup_index) {
  constexpr uint group_size = 32;
  const uint linear = threadgroup_index * RowsPerThreadgroup + simdgroup;
  const uint assignments = params.tokens * params.top_k;
  const ulong total_rows = ulong(assignments) * params.output_features;
  if (ulong(linear) >= total_rows) {
    return;
  }
  const uint assignment = linear / params.output_features;
  const uint row = linear - assignment * params.output_features;
  const uint token = assignment / params.top_k;
  const uint input_row = params.assignment_input != 0 ? assignment : token;
  const uint expert = indices[assignment];
  if (expert >= params.experts) {
    if (lane == 0) {
      output[linear] = bfloat(0.0f);
    }
    return;
  }
  const ulong flattened_row = ulong(expert) * params.output_features + row;
  const ulong weight_code_row_bytes = params.padded_input_features / 2;
  const ulong weight_groups_per_row = params.padded_input_features / group_size;
  const ulong activation_code_row_bytes = params.input_features / 2;
  const uint logical_groups = params.input_features / group_size;
  device const uchar *codes =
      weight_codes + flattened_row * weight_code_row_bytes;
  device const uchar *scales =
      weight_scales + flattened_row * weight_groups_per_row;
  float result = 0.0f;
  for (uint group = 0; group < logical_groups; ++group) {
    const uint column = group * group_size + lane;
    const uchar weight_packed = codes[column / 2];
    const uint weight_code =
        (uint(weight_packed) >> ((column & 1u) * 4u)) & 0x0fu;
    const uchar activation_packed =
        activation_codes[ulong(input_row) * activation_code_row_bytes +
                         column / 2];
    const uint activation_code =
        (uint(activation_packed) >> ((column & 1u) * 4u)) & 0x0fu;
    float partial =
        moe_decode_e2m1(weight_code) * moe_decode_e2m1(activation_code);
    partial = simd_sum(partial);
    if (lane == 0) {
      const float scale =
          moe_decode_e8m0(scales[group]) *
          moe_decode_e8m0(
              activation_scales[ulong(input_row) * logical_groups + group]);
      result = fma(scale, partial, result);
    }
  }
  if (lane == 0) {
    if (params.apply_scores != 0) {
      result *= float(route_scores[assignment]);
    }
    output[linear] = bfloat(result);
  }
}

kernel void omlx_m5_moe_group_sum_g64(device const bfloat *input [[buffer(0)]],
                                      device float *group_sums [[buffer(1)]],
                                      constant MoeParams &params [[buffer(2)]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint3 tg_position
                                      [[threadgroup_position_in_grid]]) {
  moe_group_sum<64>(input, group_sums, params, lane, tg_position.x);
}

kernel void omlx_m5_moe_group_sum_g128(device const bfloat *input [[buffer(0)]],
                                       device float *group_sums [[buffer(1)]],
                                       constant MoeParams &params [[buffer(2)]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint3 tg_position
                                       [[threadgroup_position_in_grid]]) {
  moe_group_sum<128>(input, group_sums, params, lane, tg_position.x);
}

#define DEFINE_AFFINE_MOE(bits, group_size, suffix, rows_per_tg)               \
  kernel void omlx_m5_oq_q##bits##_g##group_size##_moe_##suffix(               \
      device const bfloat *input [[buffer(0)]],                                \
      device const uint *indices [[buffer(1)]],                                \
      device const bfloat *route_scores [[buffer(2)]],                         \
      device const uchar *base_codes [[buffer(3)]],                            \
      device const uchar *high_codes [[buffer(4)]],                            \
      device const bfloat *scales [[buffer(5)]],                               \
      device const bfloat *biases [[buffer(6)]],                               \
      device const float *group_sums [[buffer(7)]],                            \
      device bfloat *output [[buffer(8)]],                                     \
      constant MoeParams &params [[buffer(9)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simdgroup [[simdgroup_index_in_threadgroup]],                       \
      uint3 tg_position [[threadgroup_position_in_grid]]) {                    \
    moe_affine_assignment<bits, group_size, rows_per_tg>(                      \
        input, indices, route_scores, base_codes, high_codes, scales, biases,  \
        group_sums, output, params, lane, simdgroup, tg_position.x);           \
  }

#define DEFINE_AFFINE_MOE_PAIR(bits, group_size)                               \
  DEFINE_AFFINE_MOE(bits, group_size, grouped, 4)                              \
  DEFINE_AFFINE_MOE(bits, group_size, fallback, 1)                             \
  kernel void omlx_m5_oq_q##bits##_g##group_size##_moe_swiglu_grouped(         \
      device const bfloat *input [[buffer(0)]],                                \
      device const uint *indices [[buffer(1)]],                                \
      device const uchar *g_base [[buffer(2)]],                                \
      device const uchar *g_high [[buffer(3)]],                                \
      device const bfloat *g_scales [[buffer(4)]],                             \
      device const bfloat *g_biases [[buffer(5)]],                             \
      device const uchar *u_base [[buffer(6)]],                                \
      device const uchar *u_high [[buffer(7)]],                                \
      device const bfloat *u_scales [[buffer(8)]],                             \
      device const bfloat *u_biases [[buffer(9)]],                             \
      device const float *group_sums [[buffer(10)]],                           \
      device bfloat *output [[buffer(11)]],                                    \
      constant MoeParams &params [[buffer(12)]],                               \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simdgroup [[simdgroup_index_in_threadgroup]],                       \
      uint3 tg_position [[threadgroup_position_in_grid]]) {                    \
    moe_affine_swiglu_grouped_impl<bits, group_size>(                          \
        input, indices, g_base, g_high, g_scales, g_biases, u_base, u_high,    \
        u_scales, u_biases, group_sums, output, params, lane, simdgroup,       \
        tg_position.x);                                                        \
  }                                                                            \
  kernel void omlx_m5_oq_q##bits##_g##group_size##_moe_down_reduce(            \
      device const bfloat *input [[buffer(0)]],                                \
      device const uint *indices [[buffer(1)]],                                \
      device const bfloat *route_scores [[buffer(2)]],                         \
      device const uchar *base_codes [[buffer(3)]],                            \
      device const uchar *high_codes [[buffer(4)]],                            \
      device const bfloat *scales [[buffer(5)]],                               \
      device const bfloat *biases [[buffer(6)]],                               \
      device const float *group_sums [[buffer(7)]],                            \
      device bfloat *output [[buffer(8)]],                                     \
      constant MoeParams &params [[buffer(9)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simdgroup [[simdgroup_index_in_threadgroup]],                       \
      uint3 tg_position [[threadgroup_position_in_grid]]) {                    \
    threadgroup float partials[8];                                             \
    moe_affine_down_reduce_impl<bits, group_size>(                             \
        input, indices, route_scores, base_codes, high_codes, scales, biases,  \
        group_sums, output, params, partials, lane, simdgroup, tg_position.x); \
  }

DEFINE_AFFINE_MOE_PAIR(4, 64)
DEFINE_AFFINE_MOE_PAIR(4, 128)
DEFINE_AFFINE_MOE_PAIR(5, 64)
DEFINE_AFFINE_MOE_PAIR(5, 128)
DEFINE_AFFINE_MOE_PAIR(6, 64)
DEFINE_AFFINE_MOE_PAIR(6, 128)
DEFINE_AFFINE_MOE_PAIR(8, 64)
DEFINE_AFFINE_MOE_PAIR(8, 128)

#define DEFINE_MX_MOE_A16(suffix, rows_per_tg)                                 \
  kernel void omlx_m5_mxfp4_moe_w4a16_##suffix(                                \
      device const bfloat *input [[buffer(0)]],                                \
      device const uint *indices [[buffer(1)]],                                \
      device const bfloat *route_scores [[buffer(2)]],                         \
      device const uchar *weight_codes [[buffer(3)]],                          \
      device const uchar *weight_scales [[buffer(4)]],                         \
      device bfloat *output [[buffer(5)]],                                     \
      constant MoeParams &params [[buffer(6)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simdgroup [[simdgroup_index_in_threadgroup]],                       \
      uint3 tg_position [[threadgroup_position_in_grid]]) {                    \
    moe_mxfp4_w4a16_assignment<rows_per_tg>(                                   \
        input, indices, route_scores, weight_codes, weight_scales, output,     \
        params, lane, simdgroup, tg_position.x);                               \
  }

#define DEFINE_MX_MOE_A4(suffix, rows_per_tg)                                  \
  kernel void omlx_m5_mxfp4_moe_w4a4_##suffix(                                 \
      device const uchar *activation_codes [[buffer(0)]],                      \
      device const uchar *activation_scales [[buffer(1)]],                     \
      device const uint *indices [[buffer(2)]],                                \
      device const bfloat *route_scores [[buffer(3)]],                         \
      device const uchar *weight_codes [[buffer(4)]],                          \
      device const uchar *weight_scales [[buffer(5)]],                         \
      device bfloat *output [[buffer(6)]],                                     \
      constant MoeParams &params [[buffer(7)]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simdgroup [[simdgroup_index_in_threadgroup]],                       \
      uint3 tg_position [[threadgroup_position_in_grid]]) {                    \
    moe_mxfp4_w4a4_assignment<rows_per_tg>(                                    \
        activation_codes, activation_scales, indices, route_scores,            \
        weight_codes, weight_scales, output, params, lane, simdgroup,          \
        tg_position.x);                                                        \
  }

DEFINE_MX_MOE_A16(grouped, 4)
DEFINE_MX_MOE_A16(fallback, 1)
DEFINE_MX_MOE_A4(grouped, 4)
DEFINE_MX_MOE_A4(fallback, 1)
