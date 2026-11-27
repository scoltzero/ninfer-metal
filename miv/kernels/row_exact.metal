// One compiled arithmetic body at every row count. Reuse the existing T1
// scalar/group-ordered implementation; M controls addressing and bounds only.
// No row-count-dependent reduction, split-K, or activation quantization.
// Multi-row tiling (BM=8) reuses weight/scale/bias loads across up to 8 active
// input rows and folds up to 8 32-lane pairwise SIMD reductions simultaneously
// while preserving 100% bit-for-bit identical per-lane FMA and simd_sum order.
#include "qmv_t1.metal"
#include "fwht64.h"
#include <metal_atomic>

constexpr constant uint kRowExactTileM = 8;

inline float simd_reduce1(float v) {
  v += simd_shuffle_xor(v, 1u);
  v += simd_shuffle_xor(v, 2u);
  v += simd_shuffle_xor(v, 4u);
  v += simd_shuffle_xor(v, 8u);
  v += simd_shuffle_xor(v, 16u);
  return v;
}

inline float simd_reduce2_at_lane(float q0, float q1, uint lane) {
  const float p0 = q0 + simd_shuffle_xor(q0, 1u);
  const float p1 = q1 + simd_shuffle_xor(q1, 1u);
  float v = ((lane & 1u) != 0u) ? p1 : p0;
  v += simd_shuffle_xor(v, 2u);
  v += simd_shuffle_xor(v, 4u);
  v += simd_shuffle_xor(v, 8u);
  v += simd_shuffle_xor(v, 16u);
  return v;
}

inline float simd_reduce4_at_lane(float q0, float q1, float q2, float q3,
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

inline float simd_reduce8_at_lane(float q0, float q1, float q2, float q3,
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

template <uint GS>
inline void exact_group_sum(device const bfloat *x, device float *s,
                            constant QmvT1Params &p, uint lane, uint index) {
  const uint groups = p.input_features / GS;
  const uint token = index / groups;
  if (token >= p.reserved) return;
  activation_group_sum_t1<GS>(x + ulong(token) * p.input_features,
                              s + ulong(token) * groups, lane, index % groups);
}

#define EXACT_SUM(gs)                                                         \
kernel void omlx_m5_row_exact_group_sum_g##gs(                                \
    device const bfloat *x [[buffer(0)]], device float *s [[buffer(1)]],      \
    constant QmvT1Params &p [[buffer(2)]],                                    \
    uint lane [[thread_index_in_simdgroup]],                                  \
    uint3 tg [[threadgroup_position_in_grid]]) {                              \
  exact_group_sum<gs>(x, s, p, lane, tg.x);                                   \
}
EXACT_SUM(64)
EXACT_SUM(128)

template <uint Bits, uint GroupSize, uint MaxM>
inline void exact_affine_multirow(
    device const bfloat *input, device const uchar *base_codes,
    device const uchar *high_codes, device const bfloat *scales,
    device const bfloat *biases, device const float *group_sums,
    device bfloat *output, constant QmvT1Params &params, uint lane,
    uint simdgroup, uint tg_x, uint row_base, uint active_rows) {
  constexpr uint simdgroups_per_threadgroup = 4;
  const uint out_row = tg_x * simdgroups_per_threadgroup + simdgroup;
  if (out_row >= params.output_features) {
    return;
  }
  const ulong base_row_bytes = Bits == 8 ? params.padded_input_features
                                         : params.padded_input_features / 2;
  const ulong high_row_bytes =
      Bits == 5 ? params.padded_input_features / 8
                : (Bits == 6 ? params.padded_input_features / 4 : 0);
  const ulong groups_per_row = params.padded_input_features / GroupSize;
  device const uchar *base = base_codes + ulong(out_row) * base_row_bytes;
  device const uchar *high = high_codes + ulong(out_row) * high_row_bytes;
  device const bfloat *row_scales = scales + ulong(out_row) * groups_per_row;
  device const bfloat *row_biases = biases + ulong(out_row) * groups_per_row;
  const uint logical_groups = params.input_features / GroupSize;

  device const bfloat *in_base = input + ulong(row_base) * params.input_features;
  const uint in_stride = params.input_features;
  device const float *lane_sums =
      lane < active_rows
          ? (group_sums + ulong(row_base + lane) * logical_groups)
          : nullptr;

  float dot[MaxM];
#pragma clang loop unroll(full)
  for (uint m = 0; m < MaxM; ++m) {
    dot[m] = 0.0f;
  }
  float bias_acc = 0.0f;

  for (uint group = 0; group < logical_groups; ++group) {
    const uint begin = group * GroupSize;
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
      const uint col_base = begin + lane * 2u;
      if (active_rows == MaxM) {
#pragma clang loop unroll(full)
        for (uint m = 0; m < MaxM; ++m) {
          const bfloat2 xv = reinterpret_cast<device const bfloat2 *>(
              in_base + ulong(m) * in_stride + col_base)[0];
          dot[m] = fma(float(xv.x), w0, dot[m]);
          dot[m] = fma(float(xv.y), w1, dot[m]);
        }
      } else {
#pragma clang loop unroll(full)
        for (uint m = 0; m < MaxM; ++m) {
          const uint rm = m < active_rows ? m : 0u;
          const bfloat2 xv = reinterpret_cast<device const bfloat2 *>(
              in_base + ulong(rm) * in_stride + col_base)[0];
          dot[m] = fma(float(xv.x), w0, dot[m]);
          dot[m] = fma(float(xv.y), w1, dot[m]);
        }
      }
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
        const uint b = uint(
            reinterpret_cast<device const ushort *>(base + ulong(group) * 64u)[lane]);
        uint q0 = b & 0x0fu;
        uint q1 = (b >> 4u) & 0x0fu;
        uint q2 = (b >> 8u) & 0x0fu;
        uint q3 = (b >> 12u) & 0x0fu;
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
      const uint col_base = begin + lane * 4u;
      if (active_rows == MaxM) {
#pragma clang loop unroll(full)
        for (uint m = 0; m < MaxM; ++m) {
          const bfloat4 xv = reinterpret_cast<device const bfloat4 *>(
              in_base + ulong(m) * in_stride + col_base)[0];
          dot[m] = fma(float(xv.x), w0, dot[m]);
          dot[m] = fma(float(xv.y), w1, dot[m]);
          dot[m] = fma(float(xv.z), w2, dot[m]);
          dot[m] = fma(float(xv.w), w3, dot[m]);
        }
      } else {
#pragma clang loop unroll(full)
        for (uint m = 0; m < MaxM; ++m) {
          const uint rm = m < active_rows ? m : 0u;
          const bfloat4 xv = reinterpret_cast<device const bfloat4 *>(
              in_base + ulong(rm) * in_stride + col_base)[0];
          dot[m] = fma(float(xv.x), w0, dot[m]);
          dot[m] = fma(float(xv.y), w1, dot[m]);
          dot[m] = fma(float(xv.z), w2, dot[m]);
          dot[m] = fma(float(xv.w), w3, dot[m]);
        }
      }
    }
    if (lane < active_rows) {
      const float b = float(row_biases[group]);
      bias_acc = fma(b, lane_sums[group], bias_acc);
    }
  }

  float reduced = 0.0f;
  if constexpr (MaxM == 1) {
    reduced = simd_reduce1(dot[0]);
  } else if constexpr (MaxM == 2) {
    reduced = simd_reduce2_at_lane(dot[0], dot[1], lane);
  } else if constexpr (MaxM == 4) {
    reduced = simd_reduce4_at_lane(dot[0], dot[1], dot[2], dot[3], lane);
  } else {
    reduced = simd_reduce8_at_lane(dot[0], dot[1], dot[2], dot[3],
                                   dot[4], dot[5], dot[6], dot[7], lane);
  }

  if (lane < active_rows) {
    output[ulong(row_base + lane) * params.output_features + out_row] =
        bfloat(reduced + bias_acc);
  }
}

template <uint Bits, uint GroupSize>
inline void dispatch_exact_affine(
    device const bfloat *x, device const uchar *base, device const uchar *high,
    device const bfloat *scale, device const bfloat *bias,
    device const float *sums, device bfloat *y, constant QmvT1Params &p,
    uint lane, uint sg, uint3 tg) {
  if ((tg.y % kRowExactTileM) != 0 || tg.y >= p.reserved) {
    return;
  }
  const uint row_base = tg.y;
  const uint active_rows = min(kRowExactTileM, p.reserved - row_base);
  if (active_rows == 1) {
    exact_affine_multirow<Bits, GroupSize, 1>(x, base, high, scale, bias, sums,
                                              y, p, lane, sg, tg.x, row_base,
                                              1u);
  } else if (active_rows <= 2) {
    exact_affine_multirow<Bits, GroupSize, 2>(x, base, high, scale, bias, sums,
                                              y, p, lane, sg, tg.x, row_base,
                                              active_rows);
  } else if (active_rows <= 4) {
    exact_affine_multirow<Bits, GroupSize, 4>(x, base, high, scale, bias, sums,
                                              y, p, lane, sg, tg.x, row_base,
                                              active_rows);
  } else {
    exact_affine_multirow<Bits, GroupSize, 8>(x, base, high, scale, bias, sums,
                                              y, p, lane, sg, tg.x, row_base,
                                              active_rows);
  }
}

#define EXACT_AFFINE(bits, gs)                                                 \
kernel void omlx_m5_row_exact_q##bits##_g##gs(                                 \
    device const bfloat *x [[buffer(0)]],                                      \
    device const uchar *base [[buffer(1)]],                                    \
    device const uchar *high [[buffer(2)]],                                    \
    device const bfloat *scale [[buffer(3)]],                                  \
    device const bfloat *bias [[buffer(4)]],                                   \
    device const float *sums [[buffer(5)]],                                    \
    device bfloat *y [[buffer(6)]], constant QmvT1Params &p [[buffer(7)]],     \
    uint lane [[thread_index_in_simdgroup]],                                   \
    uint sg [[simdgroup_index_in_threadgroup]],                                \
    uint3 tg [[threadgroup_position_in_grid]]) {                               \
  dispatch_exact_affine<bits, gs>(x, base, high, scale, bias, sums, y, p,      \
                                  lane, sg, tg);                               \
}
EXACT_AFFINE(4, 64)
EXACT_AFFINE(4, 128)
EXACT_AFFINE(5, 64)
EXACT_AFFINE(5, 128)
EXACT_AFFINE(6, 64)
EXACT_AFFINE(6, 128)
EXACT_AFFINE(8, 64)
EXACT_AFFINE(8, 128)

template <uint MaxM>
inline void exact_mxfp4_multirow(device const bfloat *input,
                                 device const uchar *codes,
                                 device const uchar *scales,
                                 device bfloat *output,
                                 constant QmvT1Params &params, uint lane,
                                 uint simdgroup, uint tg_x, uint row_base,
                                 uint active_rows) {
  constexpr uint simdgroups_per_threadgroup = 4;
  constexpr uint group_size = 32;
  const uint out_row = tg_x * simdgroups_per_threadgroup + simdgroup;
  if (out_row >= params.output_features) {
    return;
  }
  const ulong code_row_bytes = params.padded_input_features / 2;
  const ulong groups_per_row = params.padded_input_features / group_size;
  device const uchar *row_codes = codes + ulong(out_row) * code_row_bytes;
  device const uchar *row_scales = scales + ulong(out_row) * groups_per_row;
  const uint logical_groups = params.input_features / group_size;

  device const bfloat *in_base = input + ulong(row_base) * params.input_features;
  const uint in_stride = params.input_features;

  float acc[MaxM];
#pragma clang loop unroll(full)
  for (uint m = 0; m < MaxM; ++m) {
    acc[m] = 0.0f;
  }

  const uint sg_off = lane >> 2u;
  const uint sl = lane & 3u;
  const uint col_lane_off = sg_off * 32u + sl * 8u;
  uint group = 0;
  for (; group + 7u < logical_groups; group += 8u) {
    const uint packed32 =
        reinterpret_cast<device const uint *>(row_codes + group * 16u)[lane];
    const float s = decode_e8m0(row_scales[group + sg_off]);
    const float w0 = decode_e2m1(packed32 & 0x0fu) * s;
    const float w1 = decode_e2m1((packed32 >> 4u) & 0x0fu) * s;
    const float w2 = decode_e2m1((packed32 >> 8u) & 0x0fu) * s;
    const float w3 = decode_e2m1((packed32 >> 12u) & 0x0fu) * s;
    const float w4 = decode_e2m1((packed32 >> 16u) & 0x0fu) * s;
    const float w5 = decode_e2m1((packed32 >> 20u) & 0x0fu) * s;
    const float w6 = decode_e2m1((packed32 >> 24u) & 0x0fu) * s;
    const float w7 = decode_e2m1((packed32 >> 28u) & 0x0fu) * s;
    const uint col_base = group * group_size + col_lane_off;
    if (active_rows == MaxM) {
#pragma clang loop unroll(full)
      for (uint m = 0; m < MaxM; ++m) {
        device const bfloat4 *xv = reinterpret_cast<device const bfloat4 *>(
            in_base + ulong(m) * in_stride + col_base);
        const bfloat4 a0 = xv[0];
        const bfloat4 a1 = xv[1];
        acc[m] = fma(float(a0.x), w0, acc[m]);
        acc[m] = fma(float(a0.y), w1, acc[m]);
        acc[m] = fma(float(a0.z), w2, acc[m]);
        acc[m] = fma(float(a0.w), w3, acc[m]);
        acc[m] = fma(float(a1.x), w4, acc[m]);
        acc[m] = fma(float(a1.y), w5, acc[m]);
        acc[m] = fma(float(a1.z), w6, acc[m]);
        acc[m] = fma(float(a1.w), w7, acc[m]);
      }
    } else {
#pragma clang loop unroll(full)
      for (uint m = 0; m < MaxM; ++m) {
        const uint rm = m < active_rows ? m : 0u;
        device const bfloat4 *xv = reinterpret_cast<device const bfloat4 *>(
            in_base + ulong(rm) * in_stride + col_base);
        const bfloat4 a0 = xv[0];
        const bfloat4 a1 = xv[1];
        acc[m] = fma(float(a0.x), w0, acc[m]);
        acc[m] = fma(float(a0.y), w1, acc[m]);
        acc[m] = fma(float(a0.z), w2, acc[m]);
        acc[m] = fma(float(a0.w), w3, acc[m]);
        acc[m] = fma(float(a1.x), w4, acc[m]);
        acc[m] = fma(float(a1.y), w5, acc[m]);
        acc[m] = fma(float(a1.z), w6, acc[m]);
        acc[m] = fma(float(a1.w), w7, acc[m]);
      }
    }
  }
  const uint shift = (lane & 1u) * 4u;
  const uint byte_in_group = lane >> 1u;
  for (; group < logical_groups; ++group) {
    const uint column = group * group_size + lane;
    const uchar packed = row_codes[group * 16u + byte_in_group];
    const uint code = (uint(packed) >> shift) & 0x0fu;
    const float w_val = decode_e2m1(code) * decode_e8m0(row_scales[group]);
    if (active_rows == MaxM) {
#pragma clang loop unroll(full)
      for (uint m = 0; m < MaxM; ++m) {
        acc[m] = fma(float(in_base[ulong(m) * in_stride + column]), w_val,
                     acc[m]);
      }
    } else {
#pragma clang loop unroll(full)
      for (uint m = 0; m < MaxM; ++m) {
        const uint rm = m < active_rows ? m : 0u;
        acc[m] = fma(float(in_base[ulong(rm) * in_stride + column]), w_val,
                     acc[m]);
      }
    }
  }

  float reduced = 0.0f;
  if constexpr (MaxM == 1) {
    reduced = simd_reduce1(acc[0]);
  } else if constexpr (MaxM == 2) {
    reduced = simd_reduce2_at_lane(acc[0], acc[1], lane);
  } else if constexpr (MaxM == 4) {
    reduced = simd_reduce4_at_lane(acc[0], acc[1], acc[2], acc[3], lane);
  } else {
    reduced = simd_reduce8_at_lane(acc[0], acc[1], acc[2], acc[3],
                                   acc[4], acc[5], acc[6], acc[7], lane);
  }

  if (lane < active_rows) {
    output[ulong(row_base + lane) * params.output_features + out_row] =
        bfloat(reduced);
  }
}

kernel void omlx_m5_row_exact_mxfp4(
    device const bfloat *x [[buffer(0)]],
    device const uchar *code [[buffer(1)]],
    device const uchar *scale [[buffer(2)]], device bfloat *y [[buffer(3)]],
    constant QmvT1Params &p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if ((tg.y % kRowExactTileM) != 0 || tg.y >= p.reserved) {
    return;
  }
  const uint row_base = tg.y;
  const uint active_rows = min(kRowExactTileM, p.reserved - row_base);
  if (active_rows == 1) {
    exact_mxfp4_multirow<1>(x, code, scale, y, p, lane, sg, tg.x, row_base, 1u);
  } else if (active_rows <= 2) {
    exact_mxfp4_multirow<2>(x, code, scale, y, p, lane, sg, tg.x, row_base,
                            active_rows);
  } else if (active_rows <= 4) {
    exact_mxfp4_multirow<4>(x, code, scale, y, p, lane, sg, tg.x, row_base,
                            active_rows);
  } else {
    exact_mxfp4_multirow<8>(x, code, scale, y, p, lane, sg, tg.x, row_base,
                            active_rows);
  }
}

template <uint MaxM>
inline void exact_bf16_multirow(device const bfloat *input,
                                device const bfloat *weights,
                                device bfloat *output,
                                constant QmvT1Params &params, uint lane,
                                uint simdgroup, uint tg_x, uint row_base,
                                uint active_rows) {
  constexpr uint simdgroups_per_threadgroup = 4;
  const uint out_row = tg_x * simdgroups_per_threadgroup + simdgroup;
  if (out_row >= params.output_features) {
    return;
  }
  const ulong weight_stride = ulong(
      params.padded_input_features ? params.padded_input_features
                                   : params.input_features);
  device const bfloat *weight = weights + ulong(out_row) * weight_stride;

  device const bfloat *in_base = input + ulong(row_base) * params.input_features;
  const uint in_stride = params.input_features;
  float result[MaxM];
#pragma clang loop unroll(full)
  for (uint m = 0; m < MaxM; ++m) {
    result[m] = 0.0f;
  }

  uint col = 0;
  if ((params.input_features & 127u) == 0u && (weight_stride & 3ul) == 0ul) {
    for (; col < params.input_features; col += 128u) {
      const uint c = col + lane * 4u;
      const bfloat4 wv =
          reinterpret_cast<device const bfloat4 *>(weight + c)[0];
      const float w0 = float(wv.x), w1 = float(wv.y), w2 = float(wv.z),
                  w3 = float(wv.w);
      if (active_rows == MaxM) {
#pragma clang loop unroll(full)
        for (uint m = 0; m < MaxM; ++m) {
          const bfloat4 xv = reinterpret_cast<device const bfloat4 *>(
              in_base + ulong(m) * in_stride + c)[0];
          result[m] = fma(float(xv.x), w0, result[m]);
          result[m] = fma(float(xv.y), w1, result[m]);
          result[m] = fma(float(xv.z), w2, result[m]);
          result[m] = fma(float(xv.w), w3, result[m]);
        }
      } else {
#pragma clang loop unroll(full)
        for (uint m = 0; m < MaxM; ++m) {
          const uint rm = m < active_rows ? m : 0u;
          const bfloat4 xv = reinterpret_cast<device const bfloat4 *>(
              in_base + ulong(rm) * in_stride + c)[0];
          result[m] = fma(float(xv.x), w0, result[m]);
          result[m] = fma(float(xv.y), w1, result[m]);
          result[m] = fma(float(xv.z), w2, result[m]);
          result[m] = fma(float(xv.w), w3, result[m]);
        }
      }
    }
  } else {
    for (uint column = lane; column < params.input_features; column += 32) {
      const float w_val = float(weight[column]);
#pragma clang loop unroll(full)
      for (uint m = 0; m < MaxM; ++m) {
        const uint rm = m < active_rows ? m : 0u;
        result[m] =
            fma(float(in_base[ulong(rm) * in_stride + column]), w_val, result[m]);
      }
    }
  }

  float reduced = 0.0f;
  if constexpr (MaxM == 1) {
    reduced = simd_reduce1(result[0]);
  } else if constexpr (MaxM == 2) {
    reduced = simd_reduce2_at_lane(result[0], result[1], lane);
  } else if constexpr (MaxM == 4) {
    reduced =
        simd_reduce4_at_lane(result[0], result[1], result[2], result[3], lane);
  } else {
    reduced = simd_reduce8_at_lane(result[0], result[1], result[2], result[3],
                                   result[4], result[5], result[6], result[7],
                                   lane);
  }
  if (lane < active_rows) {
    output[ulong(row_base + lane) * params.output_features + out_row] =
        bfloat(reduced);
  }
}

kernel void omlx_m5_row_exact_bf16(
    device const bfloat *x [[buffer(0)]], device const bfloat *w [[buffer(1)]],
    device bfloat *y [[buffer(2)]], constant QmvT1Params &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if ((tg.y % kRowExactTileM) != 0 || tg.y >= p.reserved) {
    return;
  }
  const uint row_base = tg.y;
  const uint active_rows = min(kRowExactTileM, p.reserved - row_base);
  if (active_rows == 1) {
    exact_bf16_multirow<1>(x, w, y, p, lane, sg, tg.x, row_base, 1u);
  } else if (active_rows <= 2) {
    exact_bf16_multirow<2>(x, w, y, p, lane, sg, tg.x, row_base, active_rows);
  } else if (active_rows <= 4) {
    exact_bf16_multirow<4>(x, w, y, p, lane, sg, tg.x, row_base, active_rows);
  } else {
    exact_bf16_multirow<8>(x, w, y, p, lane, sg, tg.x, row_base, active_rows);
  }
}

// ============================================================================
// NInfer-style fused row-exact gate + up + SwiGLU (linear_swiglu):
// Computes gate_proj(x) and up_proj(x) in a single pass sharing activation
// loads, applies exact BF16-rounded SiLU(gate) * up, and writes product
// directly. Eliminates 128 dispatches + barriers and all gate/up global I/O on
// 64-layer models while staying 100% bit-exact across M=1..128.
// ============================================================================
constexpr constant uint kSwigluTileM = 4;

template <uint MaxM>
inline void exact_mxfp4_swiglu_multirow(
    device const bfloat *input, device const uchar *g_codes,
    device const uchar *g_scales, device const uchar *u_codes,
    device const uchar *u_scales, device bfloat *output,
    constant QmvT1Params &params, uint lane, uint simdgroup, uint tg_x,
    uint row_base, uint active_rows) {
  constexpr uint simdgroups_per_threadgroup = 4;
  constexpr uint group_size = 32;
  const uint out_row = tg_x * simdgroups_per_threadgroup + simdgroup;
  if (out_row >= params.output_features) {
    return;
  }
  const ulong code_row_bytes = params.padded_input_features / 2;
  const ulong groups_per_row = params.padded_input_features / group_size;
  device const uchar *gr_codes = g_codes + ulong(out_row) * code_row_bytes;
  device const uchar *gr_scales = g_scales + ulong(out_row) * groups_per_row;
  device const uchar *ur_codes = u_codes + ulong(out_row) * code_row_bytes;
  device const uchar *ur_scales = u_scales + ulong(out_row) * groups_per_row;
  const uint logical_groups = params.input_features / group_size;

  device const bfloat *in_base = input + ulong(row_base) * params.input_features;
  const uint in_stride = params.input_features;

  float acc_g[MaxM];
  float acc_u[MaxM];
#pragma clang loop unroll(full)
  for (uint m = 0; m < MaxM; ++m) {
    acc_g[m] = 0.0f;
    acc_u[m] = 0.0f;
  }

  const uint sg_off = lane >> 2u;
  const uint sl = lane & 3u;
  const uint col_lane_off = sg_off * 32u + sl * 8u;
  uint group = 0;
  for (; group + 7u < logical_groups; group += 8u) {
    const uint pg =
        reinterpret_cast<device const uint *>(gr_codes + group * 16u)[lane];
    const float sg_s = decode_e8m0(gr_scales[group + sg_off]);
    const float gw0 = decode_e2m1(pg & 0x0fu) * sg_s;
    const float gw1 = decode_e2m1((pg >> 4u) & 0x0fu) * sg_s;
    const float gw2 = decode_e2m1((pg >> 8u) & 0x0fu) * sg_s;
    const float gw3 = decode_e2m1((pg >> 12u) & 0x0fu) * sg_s;
    const float gw4 = decode_e2m1((pg >> 16u) & 0x0fu) * sg_s;
    const float gw5 = decode_e2m1((pg >> 20u) & 0x0fu) * sg_s;
    const float gw6 = decode_e2m1((pg >> 24u) & 0x0fu) * sg_s;
    const float gw7 = decode_e2m1((pg >> 28u) & 0x0fu) * sg_s;

    const uint pu =
        reinterpret_cast<device const uint *>(ur_codes + group * 16u)[lane];
    const float su_s = decode_e8m0(ur_scales[group + sg_off]);
    const float uw0 = decode_e2m1(pu & 0x0fu) * su_s;
    const float uw1 = decode_e2m1((pu >> 4u) & 0x0fu) * su_s;
    const float uw2 = decode_e2m1((pu >> 8u) & 0x0fu) * su_s;
    const float uw3 = decode_e2m1((pu >> 12u) & 0x0fu) * su_s;
    const float uw4 = decode_e2m1((pu >> 16u) & 0x0fu) * su_s;
    const float uw5 = decode_e2m1((pu >> 20u) & 0x0fu) * su_s;
    const float uw6 = decode_e2m1((pu >> 24u) & 0x0fu) * su_s;
    const float uw7 = decode_e2m1((pu >> 28u) & 0x0fu) * su_s;

    const uint col_base = group * group_size + col_lane_off;
    if (active_rows == MaxM) {
#pragma clang loop unroll(full)
      for (uint m = 0; m < MaxM; ++m) {
        device const bfloat4 *xv = reinterpret_cast<device const bfloat4 *>(
            in_base + ulong(m) * in_stride + col_base);
        const float4 a0 = float4(xv[0]);
        const float4 a1 = float4(xv[1]);
        acc_g[m] = fma(a0.x, gw0, acc_g[m]);
        acc_g[m] = fma(a0.y, gw1, acc_g[m]);
        acc_g[m] = fma(a0.z, gw2, acc_g[m]);
        acc_g[m] = fma(a0.w, gw3, acc_g[m]);
        acc_g[m] = fma(a1.x, gw4, acc_g[m]);
        acc_g[m] = fma(a1.y, gw5, acc_g[m]);
        acc_g[m] = fma(a1.z, gw6, acc_g[m]);
        acc_g[m] = fma(a1.w, gw7, acc_g[m]);
        acc_u[m] = fma(a0.x, uw0, acc_u[m]);
        acc_u[m] = fma(a0.y, uw1, acc_u[m]);
        acc_u[m] = fma(a0.z, uw2, acc_u[m]);
        acc_u[m] = fma(a0.w, uw3, acc_u[m]);
        acc_u[m] = fma(a1.x, uw4, acc_u[m]);
        acc_u[m] = fma(a1.y, uw5, acc_u[m]);
        acc_u[m] = fma(a1.z, uw6, acc_u[m]);
        acc_u[m] = fma(a1.w, uw7, acc_u[m]);
      }
    } else {
#pragma clang loop unroll(full)
      for (uint m = 0; m < MaxM; ++m) {
        const uint rm = m < active_rows ? m : 0u;
        device const bfloat4 *xv = reinterpret_cast<device const bfloat4 *>(
            in_base + ulong(rm) * in_stride + col_base);
        const float4 a0 = float4(xv[0]);
        const float4 a1 = float4(xv[1]);
        acc_g[m] = fma(a0.x, gw0, acc_g[m]);
        acc_g[m] = fma(a0.y, gw1, acc_g[m]);
        acc_g[m] = fma(a0.z, gw2, acc_g[m]);
        acc_g[m] = fma(a0.w, gw3, acc_g[m]);
        acc_g[m] = fma(a1.x, gw4, acc_g[m]);
        acc_g[m] = fma(a1.y, gw5, acc_g[m]);
        acc_g[m] = fma(a1.z, gw6, acc_g[m]);
        acc_g[m] = fma(a1.w, gw7, acc_g[m]);
        acc_u[m] = fma(a0.x, uw0, acc_u[m]);
        acc_u[m] = fma(a0.y, uw1, acc_u[m]);
        acc_u[m] = fma(a0.z, uw2, acc_u[m]);
        acc_u[m] = fma(a0.w, uw3, acc_u[m]);
        acc_u[m] = fma(a1.x, uw4, acc_u[m]);
        acc_u[m] = fma(a1.y, uw5, acc_u[m]);
        acc_u[m] = fma(a1.z, uw6, acc_u[m]);
        acc_u[m] = fma(a1.w, uw7, acc_u[m]);
      }
    }
  }
  const uint shift = (lane & 1u) * 4u;
  const uint byte_in_group = lane >> 1u;
  for (; group < logical_groups; ++group) {
    const uint column = group * group_size + lane;
    const uint bg = group * 16u + byte_in_group;
    const float gw = decode_e2m1((uint(gr_codes[bg]) >> shift) & 0x0fu) *
                     decode_e8m0(gr_scales[group]);
    const float uw = decode_e2m1((uint(ur_codes[bg]) >> shift) & 0x0fu) *
                     decode_e8m0(ur_scales[group]);
#pragma clang loop unroll(full)
    for (uint m = 0; m < MaxM; ++m) {
      const uint rm = m < active_rows ? m : 0u;
      const float xval = float(in_base[ulong(rm) * in_stride + column]);
      acc_g[m] = fma(xval, gw, acc_g[m]);
      acc_u[m] = fma(xval, uw, acc_u[m]);
    }
  }

  float rg = 0.0f, ru = 0.0f;
  if constexpr (MaxM == 1) {
    rg = simd_reduce1(acc_g[0]);
    ru = simd_reduce1(acc_u[0]);
  } else if constexpr (MaxM == 2) {
    rg = simd_reduce2_at_lane(acc_g[0], acc_g[1], lane);
    ru = simd_reduce2_at_lane(acc_u[0], acc_u[1], lane);
  } else {
    rg = simd_reduce4_at_lane(acc_g[0], acc_g[1], acc_g[2], acc_g[3], lane);
    ru = simd_reduce4_at_lane(acc_u[0], acc_u[1], acc_u[2], acc_u[3], lane);
  }

  if (lane < active_rows) {
    const float gv = float(bfloat(rg));
    const float uv = float(bfloat(ru));
    output[ulong(row_base + lane) * params.output_features + out_row] =
        bfloat((gv / (1.0f + exp(-gv))) * uv);
  }
}

kernel void omlx_m5_row_exact_mxfp4_swiglu(
    device const bfloat *x [[buffer(0)]],
    device const uchar *g_code [[buffer(1)]],
    device const uchar *g_scale [[buffer(2)]],
    device const uchar *u_code [[buffer(3)]],
    device const uchar *u_scale [[buffer(4)]],
    device bfloat *y [[buffer(5)]],
    constant QmvT1Params &p [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if ((tg.y % kSwigluTileM) != 0 || tg.y >= p.reserved) {
    return;
  }
  const uint row_base = tg.y;
  const uint active_rows = min(kSwigluTileM, p.reserved - row_base);
  if (active_rows == 1) {
    exact_mxfp4_swiglu_multirow<1>(x, g_code, g_scale, u_code, u_scale, y, p,
                                   lane, sg, tg.x, row_base, 1u);
  } else if (active_rows <= 2) {
    exact_mxfp4_swiglu_multirow<2>(x, g_code, g_scale, u_code, u_scale, y, p,
                                   lane, sg, tg.x, row_base, active_rows);
  } else {
    exact_mxfp4_swiglu_multirow<4>(x, g_code, g_scale, u_code, u_scale, y, p,
                                   lane, sg, tg.x, row_base, active_rows);
  }
}

template <uint Bits, uint GroupSize, uint MaxM>
inline void exact_affine_swiglu_multirow(
    device const bfloat *input, device const uchar *g_base,
    device const uchar *g_high, device const bfloat *g_scales,
    device const bfloat *g_biases, device const uchar *u_base,
    device const uchar *u_high, device const bfloat *u_scales,
    device const bfloat *u_biases, device const float *group_sums,
    device bfloat *output, constant QmvT1Params &params, uint lane,
    uint simdgroup, uint tg_x, uint row_base, uint active_rows) {
  constexpr uint simdgroups_per_threadgroup = 4;
  const uint out_row = tg_x * simdgroups_per_threadgroup + simdgroup;
  if (out_row >= params.output_features) {
    return;
  }
  const ulong base_row_bytes = Bits == 8 ? params.padded_input_features
                                         : params.padded_input_features / 2;
  const ulong high_row_bytes =
      Bits == 5 ? params.padded_input_features / 8
                : (Bits == 6 ? params.padded_input_features / 4 : 0);
  const ulong groups_per_row = params.padded_input_features / GroupSize;
  device const uchar *gb = g_base + ulong(out_row) * base_row_bytes;
  device const uchar *gh = g_high + ulong(out_row) * high_row_bytes;
  device const bfloat *gs = g_scales + ulong(out_row) * groups_per_row;
  device const bfloat *gbi = g_biases + ulong(out_row) * groups_per_row;
  device const uchar *ub = u_base + ulong(out_row) * base_row_bytes;
  device const uchar *uh = u_high + ulong(out_row) * high_row_bytes;
  device const bfloat *us = u_scales + ulong(out_row) * groups_per_row;
  device const bfloat *ubi = u_biases + ulong(out_row) * groups_per_row;
  const uint logical_groups = params.input_features / GroupSize;

  device const bfloat *in_base = input + ulong(row_base) * params.input_features;
  const uint in_stride = params.input_features;
  device const float *lane_sums =
      lane < active_rows
          ? (group_sums + ulong(row_base + lane) * logical_groups)
          : nullptr;

  float dot_g[MaxM];
  float dot_u[MaxM];
#pragma clang loop unroll(full)
  for (uint m = 0; m < MaxM; ++m) {
    dot_g[m] = 0.0f;
    dot_u[m] = 0.0f;
  }
  float bias_g = 0.0f, bias_u = 0.0f;

  for (uint group = 0; group < logical_groups; ++group) {
    const uint begin = group * GroupSize;
    const float sg_v = float(gs[group]);
    const float su_v = float(us[group]);
    if constexpr (GroupSize == 64) {
      float gw0, gw1, uw0, uw1;
      if constexpr (Bits == 8) {
        const uchar2 b0 = reinterpret_cast<device const uchar2 *>(gb + ulong(group) * 64u)[lane];
        const uchar2 b1 = reinterpret_cast<device const uchar2 *>(ub + ulong(group) * 64u)[lane];
        gw0 = sg_v * float(b0.x); gw1 = sg_v * float(b0.y);
        uw0 = su_v * float(b1.x); uw1 = su_v * float(b1.y);
      } else {
        const uint b0 = uint(gb[ulong(group) * 32u + lane]);
        const uint b1 = uint(ub[ulong(group) * 32u + lane]);
        uint gq0 = b0 & 0x0fu, gq1 = b0 >> 4u;
        uint uq0 = b1 & 0x0fu, uq1 = b1 >> 4u;
        if constexpr (Bits == 5) {
          const uint h0 = uint(gh[ulong(group) * 8u + (lane >> 2u)]) >> ((lane & 3u) * 2u);
          const uint h1 = uint(uh[ulong(group) * 8u + (lane >> 2u)]) >> ((lane & 3u) * 2u);
          gq0 |= (h0 & 1u) << 4u; gq1 |= ((h0 >> 1u) & 1u) << 4u;
          uq0 |= (h1 & 1u) << 4u; uq1 |= ((h1 >> 1u) & 1u) << 4u;
        } else if constexpr (Bits == 6) {
          const uint h0 = uint(gh[ulong(group) * 16u + (lane >> 1u)]) >> ((lane & 1u) * 4u);
          const uint h1 = uint(uh[ulong(group) * 16u + (lane >> 1u)]) >> ((lane & 1u) * 4u);
          gq0 |= (h0 & 3u) << 4u; gq1 |= ((h0 >> 2u) & 3u) << 4u;
          uq0 |= (h1 & 3u) << 4u; uq1 |= ((h1 >> 2u) & 3u) << 4u;
        }
        gw0 = sg_v * float(gq0); gw1 = sg_v * float(gq1);
        uw0 = su_v * float(uq0); uw1 = su_v * float(uq1);
      }
      const uint col_base = begin + lane * 2u;
#pragma clang loop unroll(full)
      for (uint m = 0; m < MaxM; ++m) {
        const uint rm = (active_rows == MaxM || m < active_rows) ? m : 0u;
        const bfloat2 xv = reinterpret_cast<device const bfloat2 *>(
            in_base + ulong(rm) * in_stride + col_base)[0];
        const float x0 = float(xv.x), x1 = float(xv.y);
        dot_g[m] = fma(x0, gw0, dot_g[m]);
        dot_g[m] = fma(x1, gw1, dot_g[m]);
        dot_u[m] = fma(x0, uw0, dot_u[m]);
        dot_u[m] = fma(x1, uw1, dot_u[m]);
      }
    } else {
      float gw0, gw1, gw2, gw3, uw0, uw1, uw2, uw3;
      if constexpr (Bits == 8) {
        const uchar4 b0 = reinterpret_cast<device const uchar4 *>(gb + ulong(group) * 128u)[lane];
        const uchar4 b1 = reinterpret_cast<device const uchar4 *>(ub + ulong(group) * 128u)[lane];
        gw0 = sg_v * float(b0.x); gw1 = sg_v * float(b0.y); gw2 = sg_v * float(b0.z); gw3 = sg_v * float(b0.w);
        uw0 = su_v * float(b1.x); uw1 = su_v * float(b1.y); uw2 = su_v * float(b1.z); uw3 = su_v * float(b1.w);
      } else {
        const uint b0 = uint(reinterpret_cast<device const ushort *>(gb + ulong(group) * 64u)[lane]);
        const uint b1 = uint(reinterpret_cast<device const ushort *>(ub + ulong(group) * 64u)[lane]);
        uint gq0 = b0 & 15u, gq1 = (b0 >> 4u) & 15u, gq2 = (b0 >> 8u) & 15u, gq3 = (b0 >> 12u) & 15u;
        uint uq0 = b1 & 15u, uq1 = (b1 >> 4u) & 15u, uq2 = (b1 >> 8u) & 15u, uq3 = (b1 >> 12u) & 15u;
        if constexpr (Bits == 5) {
          const uint h0 = uint(gh[ulong(group) * 16u + (lane >> 1u)]) >> ((lane & 1u) * 4u);
          const uint h1 = uint(uh[ulong(group) * 16u + (lane >> 1u)]) >> ((lane & 1u) * 4u);
          gq0 |= (h0 & 1u) << 4u; gq1 |= ((h0 >> 1u) & 1u) << 4u; gq2 |= ((h0 >> 2u) & 1u) << 4u; gq3 |= ((h0 >> 3u) & 1u) << 4u;
          uq0 |= (h1 & 1u) << 4u; uq1 |= ((h1 >> 1u) & 1u) << 4u; uq2 |= ((h1 >> 2u) & 1u) << 4u; uq3 |= ((h1 >> 3u) & 1u) << 4u;
        } else if constexpr (Bits == 6) {
          const uint h0 = uint(gh[ulong(group) * 32u + lane]);
          const uint h1 = uint(uh[ulong(group) * 32u + lane]);
          gq0 |= (h0 & 3u) << 4u; gq1 |= ((h0 >> 2u) & 3u) << 4u; gq2 |= ((h0 >> 4u) & 3u) << 4u; gq3 |= ((h0 >> 6u) & 3u) << 4u;
          uq0 |= (h1 & 3u) << 4u; uq1 |= ((h1 >> 2u) & 3u) << 4u; uq2 |= ((h1 >> 4u) & 3u) << 4u; uq3 |= ((h1 >> 6u) & 3u) << 4u;
        }
        gw0 = sg_v * float(gq0); gw1 = sg_v * float(gq1); gw2 = sg_v * float(gq2); gw3 = sg_v * float(gq3);
        uw0 = su_v * float(uq0); uw1 = su_v * float(uq1); uw2 = su_v * float(uq2); uw3 = su_v * float(uq3);
      }
      const uint col_base = begin + lane * 4u;
#pragma clang loop unroll(full)
      for (uint m = 0; m < MaxM; ++m) {
        const uint rm = (active_rows == MaxM || m < active_rows) ? m : 0u;
        const float4 xv = float4(reinterpret_cast<device const bfloat4 *>(
            in_base + ulong(rm) * in_stride + col_base)[0]);
        dot_g[m] = fma(xv.x, gw0, dot_g[m]); dot_g[m] = fma(xv.y, gw1, dot_g[m]);
        dot_g[m] = fma(xv.z, gw2, dot_g[m]); dot_g[m] = fma(xv.w, gw3, dot_g[m]);
        dot_u[m] = fma(xv.x, uw0, dot_u[m]); dot_u[m] = fma(xv.y, uw1, dot_u[m]);
        dot_u[m] = fma(xv.z, uw2, dot_u[m]); dot_u[m] = fma(xv.w, uw3, dot_u[m]);
      }
    }
    if (lane < active_rows) {
      const float ls = lane_sums[group];
      bias_g = fma(float(gbi[group]), ls, bias_g);
      bias_u = fma(float(ubi[group]), ls, bias_u);
    }
  }

  float rg = 0.0f, ru = 0.0f;
  if constexpr (MaxM == 1) {
    rg = simd_reduce1(dot_g[0]);
    ru = simd_reduce1(dot_u[0]);
  } else if constexpr (MaxM == 2) {
    rg = simd_reduce2_at_lane(dot_g[0], dot_g[1], lane);
    ru = simd_reduce2_at_lane(dot_u[0], dot_u[1], lane);
  } else {
    rg = simd_reduce4_at_lane(dot_g[0], dot_g[1], dot_g[2], dot_g[3], lane);
    ru = simd_reduce4_at_lane(dot_u[0], dot_u[1], dot_u[2], dot_u[3], lane);
  }
  if (lane < active_rows) {
    const float gv = float(bfloat(rg + bias_g));
    const float uv = float(bfloat(ru + bias_u));
    output[ulong(row_base + lane) * params.output_features + out_row] =
        bfloat((gv / (1.0f + exp(-gv))) * uv);
  }
}

#define EXACT_AFFINE_SWIGLU(bits, gs)                                          \
kernel void omlx_m5_row_exact_q##bits##_g##gs##_swiglu(                        \
    device const bfloat *x [[buffer(0)]],                                      \
    device const uchar *g_base [[buffer(1)]],                                  \
    device const uchar *g_high [[buffer(2)]],                                  \
    device const bfloat *g_scale [[buffer(3)]],                                \
    device const bfloat *g_bias [[buffer(4)]],                                 \
    device const uchar *u_base [[buffer(5)]],                                  \
    device const uchar *u_high [[buffer(6)]],                                  \
    device const bfloat *u_scale [[buffer(7)]],                                \
    device const bfloat *u_bias [[buffer(8)]],                                 \
    device const float *sums [[buffer(9)]],                                    \
    device bfloat *y [[buffer(10)]], constant QmvT1Params &p [[buffer(11)]],   \
    uint lane [[thread_index_in_simdgroup]],                                   \
    uint sg [[simdgroup_index_in_threadgroup]],                                \
    uint3 tg [[threadgroup_position_in_grid]]) {                               \
  if ((tg.y % kSwigluTileM) != 0 || tg.y >= p.reserved) {                      \
    return;                                                                    \
  }                                                                            \
  const uint row_base = tg.y;                                                  \
  const uint active_rows = min(kSwigluTileM, p.reserved - row_base);           \
  if (active_rows == 1) {                                                      \
    exact_affine_swiglu_multirow<bits, gs, 1>(                                 \
        x, g_base, g_high, g_scale, g_bias, u_base, u_high, u_scale, u_bias,   \
        sums, y, p, lane, sg, tg.x, row_base, 1u);                             \
  } else if (active_rows <= 2) {                                               \
    exact_affine_swiglu_multirow<bits, gs, 2>(                                 \
        x, g_base, g_high, g_scale, g_bias, u_base, u_high, u_scale, u_bias,   \
        sums, y, p, lane, sg, tg.x, row_base, active_rows);                    \
  } else {                                                                     \
    exact_affine_swiglu_multirow<bits, gs, 4>(                                 \
        x, g_base, g_high, g_scale, g_bias, u_base, u_high, u_scale, u_bias,   \
        sums, y, p, lane, sg, tg.x, row_base, active_rows);                    \
  }                                                                            \
}
EXACT_AFFINE_SWIGLU(4, 64)
EXACT_AFFINE_SWIGLU(4, 128)
EXACT_AFFINE_SWIGLU(5, 64)
EXACT_AFFINE_SWIGLU(5, 128)
EXACT_AFFINE_SWIGLU(6, 64)
EXACT_AFFINE_SWIGLU(6, 128)
EXACT_AFFINE_SWIGLU(8, 64)
EXACT_AFFINE_SWIGLU(8, 128)

// ============================================================================
// Step 4 (M4b): Direct T=1 Coalesced SIMD32 QMV over Pre-Tiled NAX Layouts
// (`mx_nax_tiled_bk128_v1` and `oq_nax_tiled_g64_v1`)
// Active ONLY under OMLX_M5_T_DOMAIN_DISPATCH=1 when speculative is off:
// - 4 simdgroups x 4 rows/SG = 16 output rows/TG (= 1 full BN=16 NAX tile)
// - 64B/32B contiguous warp weight reads, BF16 `x` reused 4x-12x in registers
// - Zero `bf16_to_f16_rowscale` or `group_sum` launch overhead, zero dual RAM!
// ============================================================================

struct PretiledQmvMxfp4Params {
  uint n0;
  uint n1;
  uint n2;
  uint k;
};

struct PretiledQmvConcat4Mxfp4Params {
  uint n0;
  uint n1;
  uint n2;
  uint n3;
  uint k;
};

struct PretiledQmvAffineParams {
  uint n0;
  uint n1;
  uint n2;
  uint k;
  uint bits;
};

struct PretiledQmvConcat4AffineParams {
  uint n0;
  uint n1;
  uint n2;
  uint n3;
  uint k;
  uint bits;
};

// Zero-uitofp IEEE-754 bit-cast FP4 E2M1 -> FP16 decoder (compiled under Metal 27.0).
inline half pretiled_fp4_to_h16(uint nib) {
  const float f = float(as_type<half>(ushort((nib & 7u) << 9u))) * 16384.0f;
  return half((nib & 8u) != 0u ? -f : f);
}

template <uint ROWS_SG, uint BN, bool PROLOGUE_NORM = false, bool EPILOGUE_ADD_SQ = false>
inline void pretiled_qmv_mxfp4_kl32_impl(
    device const bfloat *x,
    device const uchar *nax_data,
    device const uchar *nax_scale,
    device bfloat *y,
    uint n,
    uint k,
    uint tg_local,
    uint sg,
    uint lane,
    device const bfloat *rms_w = nullptr,
    float inv_rms = 1.0f,
    float w_off = 0.0f,
    device atomic_float *rms_sq_buf = nullptr,
    uint sq_slot = 0u) {
  static_assert(BN == 16u || BN == 32u, "BN must be 16 or 32");
  const uint row_sg = (tg_local * 2u + sg) * ROWS_SG;
  if (row_sg >= n) {
    return;
  }
  const bool do_fwht = (k & 0x80000000u) != 0u;
  const uint k_eff = k & 0x3FFFFFFFu;
  const uint k_blocks = k_eff / 128u;
  const uint tile_n = row_sg / BN;
  const uint rn0 = row_sg % BN;
  constexpr ulong tile_code_bytes = ulong(BN) * 64ul;
  constexpr ulong tile_scale_bytes = ulong(BN) * 4ul;

  const uint b_sub = lane >> 4u; // 0..1 (2 blocks of 128 = 256 K-elems/iter)
  const uint k_l = lane & 15u;   // 0..15 (8 nibbles = 1 uint per lane)
  const uint sub32 = k_l >> 2u;  // 0..3

  float acc[ROWS_SG];
#pragma clang loop unroll(full)
  for (uint r = 0u; r < ROWS_SG; ++r) {
    acc[r] = 0.0f;
  }

  const float4 off4 = float4(w_off);
  const ulong tile_base = ulong(tile_n) * k_blocks;
  for (uint blk = 0u; blk < k_blocks; blk += 2u) {
    const uint cur_blk = blk + b_sub;
    float4 f0 = float4(0.0f);
    float4 f1 = float4(0.0f);
    if (cur_blk < k_blocks) {
      const ulong x_off = ulong(cur_blk) * 128ul + ulong(k_l) * 8ul;
      device const bfloat4 *xb = reinterpret_cast<device const bfloat4 *>(x + x_off);
      if constexpr (PROLOGUE_NORM) {
        device const bfloat4 *wb = reinterpret_cast<device const bfloat4 *>(rms_w + x_off);
        f0 = float4(bfloat4(float4(xb[0]) * inv_rms * (float4(wb[0]) + off4)));
        f1 = float4(bfloat4(float4(xb[1]) * inv_rms * (float4(wb[1]) + off4)));
      } else {
        (void)rms_w; (void)inv_rms; (void)off4;
        f0 = float4(xb[0]);
        f1 = float4(xb[1]);
      }
    }
    if (do_fwht) {
      omlx_m5_fwht64_sub8(f0, f1, lane);
    }
    if (cur_blk < k_blocks) {
      const half4 h0 = half4(f0);
      const half4 h1 = half4(f1);
      const ulong idx = tile_base + cur_blk;
      device const uchar *d_tile = nax_data + idx * tile_code_bytes;
      device const uchar *s_tile = nax_scale + idx * tile_scale_bytes;
#pragma clang loop unroll(full)
      for (uint r = 0u; r < ROWS_SG; ++r) {
        const uint rn = rn0 + r;
        const uint w = reinterpret_cast<device const uint *>(
            d_tile + ulong(rn) * 64ul)[k_l];
        const float sc = decode_e8m0(s_tile[ulong(rn) * 4ul + sub32]);
        const half d0 =
            h0.x * pretiled_fp4_to_h16(w) +
            h0.y * pretiled_fp4_to_h16(w >> 4u) +
            h0.z * pretiled_fp4_to_h16(w >> 8u) +
            h0.w * pretiled_fp4_to_h16(w >> 12u);
        const half d1 =
            h1.x * pretiled_fp4_to_h16(w >> 16u) +
            h1.y * pretiled_fp4_to_h16(w >> 20u) +
            h1.z * pretiled_fp4_to_h16(w >> 24u) +
            h1.w * pretiled_fp4_to_h16(w >> 28u);
        acc[r] = fma(float(d0) + float(d1), sc, acc[r]);
      }
    }
  }

  if constexpr (EPILOGUE_ADD_SQ) {
    float sg_sq = 0.0f;
#pragma clang loop unroll(full)
    for (uint r = 0u; r < ROWS_SG; ++r) {
      const float v = simd_sum(acc[r]);
      if (lane == 0u && (row_sg + r) < n) {
        const bfloat h_new = bfloat(float(y[row_sg + r]) + float(bfloat(v)));
        y[row_sg + r] = h_new;
        const float hf = float(h_new);
        sg_sq = fma(hf, hf, sg_sq);
      }
    }
    if (lane == 0u) {
      atomic_fetch_add_explicit(&rms_sq_buf[sq_slot], sg_sq, memory_order_relaxed);
    }
  } else {
    (void)rms_sq_buf; (void)sq_slot;
#pragma clang loop unroll(full)
    for (uint r = 0u; r < ROWS_SG; ++r) {
      const float v = simd_sum(acc[r]);
      if (lane == 0u && (row_sg + r) < n) {
        y[row_sg + r] = bfloat(v);
      }
    }
  }
}

template <uint ROWS_SG>
inline void pretiled_qmv_mxfp4_kl32_dispatch(
    device const bfloat *x,
    device const uchar *nax_data,
    device const uchar *nax_scale,
    device bfloat *y,
    uint n,
    uint k,
    uint tg_local,
    uint sg,
    uint lane) {
  if (n >= 65536u) {
    pretiled_qmv_mxfp4_kl32_impl<ROWS_SG, 32u>(
        x, nax_data, nax_scale, y, n, k, tg_local, sg, lane);
  } else {
    pretiled_qmv_mxfp4_kl32_impl<ROWS_SG, 16u>(
        x, nax_data, nax_scale, y, n, k, tg_local, sg, lane);
  }
}

template <uint ROWS_SG, uint BN, bool PROLOGUE_NORM = false>
inline void pretiled_qmv_mxfp4_swiglu_kl32_impl(
    device const bfloat *x,
    device const uchar *g_data,
    device const uchar *g_scale,
    device const uchar *u_data,
    device const uchar *u_scale,
    device bfloat *y,
    uint n,
    uint k,
    uint tg_local,
    uint sg,
    uint lane,
    device const bfloat *rms_w = nullptr,
    float inv_rms = 1.0f,
    float w_off = 0.0f) {
  static_assert(BN == 16u || BN == 32u, "BN must be 16 or 32");
  const uint row_sg = (tg_local * 2u + sg) * ROWS_SG;
  if (row_sg >= n) {
    return;
  }
  const bool do_fwht = (k & 0x80000000u) != 0u;
  const uint k_eff = k & 0x3FFFFFFFu;
  const uint k_blocks = k_eff / 128u;
  const uint tile_n = row_sg / BN;
  const uint rn0 = row_sg % BN;
  constexpr ulong tile_code_bytes = ulong(BN) * 64ul;
  constexpr ulong tile_scale_bytes = ulong(BN) * 4ul;

  const uint b_sub = lane >> 4u;
  const uint k_l = lane & 15u;
  const uint sub32 = k_l >> 2u;

  float acc_g[ROWS_SG];
  float acc_u[ROWS_SG];
#pragma clang loop unroll(full)
  for (uint r = 0u; r < ROWS_SG; ++r) {
    acc_g[r] = 0.0f;
    acc_u[r] = 0.0f;
  }

  const float4 off4 = float4(w_off);
  const ulong tile_base = ulong(tile_n) * k_blocks;
  for (uint blk = 0u; blk < k_blocks; blk += 2u) {
    const uint cur_blk = blk + b_sub;
    float4 f0 = float4(0.0f);
    float4 f1 = float4(0.0f);
    if (cur_blk < k_blocks) {
      const ulong x_off = ulong(cur_blk) * 128ul + ulong(k_l) * 8ul;
      device const bfloat4 *xb = reinterpret_cast<device const bfloat4 *>(x + x_off);
      if constexpr (PROLOGUE_NORM) {
        device const bfloat4 *wb = reinterpret_cast<device const bfloat4 *>(rms_w + x_off);
        f0 = float4(bfloat4(float4(xb[0]) * inv_rms * (float4(wb[0]) + off4)));
        f1 = float4(bfloat4(float4(xb[1]) * inv_rms * (float4(wb[1]) + off4)));
      } else {
        (void)rms_w; (void)inv_rms; (void)off4;
        f0 = float4(xb[0]);
        f1 = float4(xb[1]);
      }
    }
    if (do_fwht) {
      omlx_m5_fwht64_sub8(f0, f1, lane);
    }
    if (cur_blk < k_blocks) {
      const half4 h0 = half4(f0);
      const half4 h1 = half4(f1);
      const ulong idx = tile_base + cur_blk;
      device const uchar *gd_tile = g_data + idx * tile_code_bytes;
      device const uchar *gs_tile = g_scale + idx * tile_scale_bytes;
      device const uchar *ud_tile = u_data + idx * tile_code_bytes;
      device const uchar *us_tile = u_scale + idx * tile_scale_bytes;
#pragma clang loop unroll(full)
      for (uint r = 0u; r < ROWS_SG; ++r) {
        const uint rn = rn0 + r;
        const uint gw = reinterpret_cast<device const uint *>(
            gd_tile + ulong(rn) * 64ul)[k_l];
        const uint uw = reinterpret_cast<device const uint *>(
            ud_tile + ulong(rn) * 64ul)[k_l];
        const float gsc = decode_e8m0(gs_tile[ulong(rn) * 4ul + sub32]);
        const float usc = decode_e8m0(us_tile[ulong(rn) * 4ul + sub32]);
        const half dg0 =
            h0.x * pretiled_fp4_to_h16(gw) +
            h0.y * pretiled_fp4_to_h16(gw >> 4u) +
            h0.z * pretiled_fp4_to_h16(gw >> 8u) +
            h0.w * pretiled_fp4_to_h16(gw >> 12u);
        const half dg1 =
            h1.x * pretiled_fp4_to_h16(gw >> 16u) +
            h1.y * pretiled_fp4_to_h16(gw >> 20u) +
            h1.z * pretiled_fp4_to_h16(gw >> 24u) +
            h1.w * pretiled_fp4_to_h16(gw >> 28u);
        const half du0 =
            h0.x * pretiled_fp4_to_h16(uw) +
            h0.y * pretiled_fp4_to_h16(uw >> 4u) +
            h0.z * pretiled_fp4_to_h16(uw >> 8u) +
            h0.w * pretiled_fp4_to_h16(uw >> 12u);
        const half du1 =
            h1.x * pretiled_fp4_to_h16(uw >> 16u) +
            h1.y * pretiled_fp4_to_h16(uw >> 20u) +
            h1.z * pretiled_fp4_to_h16(uw >> 24u) +
            h1.w * pretiled_fp4_to_h16(uw >> 28u);

        acc_g[r] = fma(float(dg0) + float(dg1), gsc, acc_g[r]);
        acc_u[r] = fma(float(du0) + float(du1), usc, acc_u[r]);
      }
    }
  }

#pragma clang loop unroll(full)
  for (uint r = 0u; r < ROWS_SG; ++r) {
    const float gsum = simd_sum(acc_g[r]);
    const float usum = simd_sum(acc_u[r]);
    if (lane == 0u && (row_sg + r) < n) {
      const float gv = float(bfloat(gsum));
      const float uv = float(bfloat(usum));
      y[row_sg + r] = bfloat((gv / (1.0f + exp(-gv))) * uv);
    }
  }
}

template <uint ROWS_SG>
inline void pretiled_qmv_mxfp4_swiglu_kl32_dispatch(
    device const bfloat *x,
    device const uchar *g_data,
    device const uchar *g_scale,
    device const uchar *u_data,
    device const uchar *u_scale,
    device bfloat *y,
    uint n,
    uint k,
    uint tg_local,
    uint sg,
    uint lane) {
  if (n >= 65536u) {
    pretiled_qmv_mxfp4_swiglu_kl32_impl<ROWS_SG, 32u>(
        x, g_data, g_scale, u_data, u_scale, y, n, k, tg_local, sg, lane);
  } else {
    pretiled_qmv_mxfp4_swiglu_kl32_impl<ROWS_SG, 16u>(
        x, g_data, g_scale, u_data, u_scale, y, n, k, tg_local, sg, lane);
  }
}

#define OMLX_M5_DEF_MXFP4_T1_SPEC(NAME, ROWS_SG, BN)                            \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *nax_data [[buffer(1)]],                               \
      device const uchar *nax_scale [[buffer(2)]],                              \
      device bfloat *y [[buffer(3)]],                                           \
      constant QmvT1Params &p [[buffer(4)]],                                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    if (p.reserved == 0u) return;                                               \
    pretiled_qmv_mxfp4_kl32_impl<ROWS_SG, BN>(                                  \
        x, nax_data, nax_scale, y, p.output_features, p.input_features, tg.x,   \
        sg, lane);                                                              \
  }

OMLX_M5_DEF_MXFP4_T1_SPEC(omlx_m5_pretiled_qmv_mxfp4_bn16_r4_t1, 4u, 16u)
OMLX_M5_DEF_MXFP4_T1_SPEC(omlx_m5_pretiled_qmv_mxfp4_bn16_r2_t1, 2u, 16u)
OMLX_M5_DEF_MXFP4_T1_SPEC(omlx_m5_pretiled_qmv_mxfp4_bn32_r4_t1, 4u, 32u)

#define OMLX_M5_DEF_MXFP4_SWIGLU_T1_SPEC(NAME, ROWS_SG, BN)                     \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *g_data [[buffer(1)]],                                 \
      device const uchar *g_scale [[buffer(2)]],                                \
      device const uchar *u_data [[buffer(3)]],                                 \
      device const uchar *u_scale [[buffer(4)]],                                \
      device bfloat *y [[buffer(5)]],                                           \
      constant QmvT1Params &p [[buffer(6)]],                                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    if (p.reserved == 0u) return;                                               \
    pretiled_qmv_mxfp4_swiglu_kl32_impl<ROWS_SG, BN>(                           \
        x, g_data, g_scale, u_data, u_scale, y, p.output_features,              \
        p.input_features, tg.x, sg, lane);                                      \
  }

OMLX_M5_DEF_MXFP4_SWIGLU_T1_SPEC(omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r4_t1, 4u, 16u)
OMLX_M5_DEF_MXFP4_SWIGLU_T1_SPEC(omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r2_t1, 2u, 16u)

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_mxfp4_t1(
    device const bfloat *x [[buffer(0)]],
    device const uchar *nax_data [[buffer(1)]],
    device const uchar *nax_scale [[buffer(2)]],
    device bfloat *y [[buffer(3)]],
    constant QmvT1Params &p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if (p.reserved == 0u) {
    return;
  }
  if (p.output_features >= 2048u) {
    pretiled_qmv_mxfp4_kl32_dispatch<4u>(
        x, nax_data, nax_scale, y, p.output_features, p.input_features, tg.x, sg, lane);
  } else {
    pretiled_qmv_mxfp4_kl32_dispatch<2u>(
        x, nax_data, nax_scale, y, p.output_features, p.input_features, tg.x, sg, lane);
  }
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_mxfp4_swiglu_t1(
    device const bfloat *x [[buffer(0)]],
    device const uchar *g_data [[buffer(1)]],
    device const uchar *g_scale [[buffer(2)]],
    device const uchar *u_data [[buffer(3)]],
    device const uchar *u_scale [[buffer(4)]],
    device bfloat *y [[buffer(5)]],
    constant QmvT1Params &p [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if (p.reserved == 0u) {
    return;
  }
  if (p.output_features >= 2048u) {
    pretiled_qmv_mxfp4_swiglu_kl32_dispatch<4u>(
        x, g_data, g_scale, u_data, u_scale, y, p.output_features, p.input_features, tg.x, sg, lane);
  } else {
    pretiled_qmv_mxfp4_swiglu_kl32_dispatch<2u>(
        x, g_data, g_scale, u_data, u_scale, y, p.output_features, p.input_features, tg.x, sg, lane);
  }
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_mxfp4_concat2_t1(
    device const bfloat *x [[buffer(0)]],
    device const uchar *d0 [[buffer(1)]],
    device const uchar *s0 [[buffer(2)]],
    device const uchar *d1 [[buffer(3)]],
    device const uchar *s1 [[buffer(4)]],
    device bfloat *y0 [[buffer(5)]],
    device bfloat *y1 [[buffer(6)]],
    constant PretiledQmvMxfp4Params &p [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if ((p.n0 + p.n1) >= 2048u) {
    const uint tgs0 = (p.n0 + 7u) / 8u;
    const bool s0_sel = (tg.x < tgs0);
    pretiled_qmv_mxfp4_kl32_impl<4u, 16u>(
        x, s0_sel ? d0 : d1, s0_sel ? s0 : s1, s0_sel ? y0 : y1,
        s0_sel ? p.n0 : p.n1, p.k, s0_sel ? tg.x : (tg.x - tgs0), sg, lane);
  } else {
    const uint tgs0 = (p.n0 + 3u) / 4u;
    const bool s0_sel = (tg.x < tgs0);
    pretiled_qmv_mxfp4_kl32_impl<2u, 16u>(
        x, s0_sel ? d0 : d1, s0_sel ? s0 : s1, s0_sel ? y0 : y1,
        s0_sel ? p.n0 : p.n1, p.k, s0_sel ? tg.x : (tg.x - tgs0), sg, lane);
  }
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_mxfp4_concat3_t1(
    device const bfloat *x [[buffer(0)]],
    device const uchar *d0 [[buffer(1)]],
    device const uchar *s0 [[buffer(2)]],
    device const uchar *d1 [[buffer(3)]],
    device const uchar *s1 [[buffer(4)]],
    device const uchar *d2 [[buffer(5)]],
    device const uchar *s2 [[buffer(6)]],
    device bfloat *y0 [[buffer(7)]],
    device bfloat *y1 [[buffer(8)]],
    device bfloat *y2 [[buffer(9)]],
    constant PretiledQmvMxfp4Params &p [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if ((p.n0 + p.n1 + p.n2) >= 2048u) {
    const uint tgs0 = (p.n0 + 7u) / 8u;
    const uint tgs1 = (p.n1 + 7u) / 8u;
    device const uchar *ds = (tg.x < tgs0) ? d0 : ((tg.x < tgs0 + tgs1) ? d1 : d2);
    device const uchar *ss = (tg.x < tgs0) ? s0 : ((tg.x < tgs0 + tgs1) ? s1 : s2);
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2);
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2);
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1));
    pretiled_qmv_mxfp4_kl32_impl<4u, 16u>(x, ds, ss, ys, ns, p.k, tgl, sg, lane);
  } else {
    const uint tgs0 = (p.n0 + 3u) / 4u;
    const uint tgs1 = (p.n1 + 3u) / 4u;
    device const uchar *ds = (tg.x < tgs0) ? d0 : ((tg.x < tgs0 + tgs1) ? d1 : d2);
    device const uchar *ss = (tg.x < tgs0) ? s0 : ((tg.x < tgs0 + tgs1) ? s1 : s2);
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2);
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2);
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1));
    pretiled_qmv_mxfp4_kl32_impl<2u, 16u>(x, ds, ss, ys, ns, p.k, tgl, sg, lane);
  }
}

#define OMLX_M5_DEF_MXFP4_CONCAT3_SPEC(NAME, ROWS_SG)                           \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *d0 [[buffer(1)]],                                     \
      device const uchar *s0 [[buffer(2)]],                                     \
      device const uchar *d1 [[buffer(3)]],                                     \
      device const uchar *s1 [[buffer(4)]],                                     \
      device const uchar *d2 [[buffer(5)]],                                     \
      device const uchar *s2 [[buffer(6)]],                                     \
      device bfloat *y0 [[buffer(7)]],                                          \
      device bfloat *y1 [[buffer(8)]],                                          \
      device bfloat *y2 [[buffer(9)]],                                          \
      constant PretiledQmvMxfp4Params &p [[buffer(10)]],                        \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    constexpr uint RPTG = (ROWS_SG) * 2u;                                       \
    const uint tgs0 = (p.n0 + RPTG - 1u) / RPTG;                                \
    const uint tgs1 = (p.n1 + RPTG - 1u) / RPTG;                                \
    device const uchar *ds = (tg.x < tgs0) ? d0 : ((tg.x < tgs0 + tgs1) ? d1 : d2); \
    device const uchar *ss = (tg.x < tgs0) ? s0 : ((tg.x < tgs0 + tgs1) ? s1 : s2); \
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2); \
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2); \
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1)); \
    pretiled_qmv_mxfp4_kl32_impl<ROWS_SG, 16u>(x, ds, ss, ys, ns, p.k, tgl, sg, lane); \
  }

OMLX_M5_DEF_MXFP4_CONCAT3_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r4_t1, 4u)
OMLX_M5_DEF_MXFP4_CONCAT3_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r2_t1, 2u)

#define OMLX_M5_DEF_MXFP4_CONCAT4_SPEC(NAME, ROWS_SG)                           \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *d0 [[buffer(1)]],                                     \
      device const uchar *s0 [[buffer(2)]],                                     \
      device const uchar *d1 [[buffer(3)]],                                     \
      device const uchar *s1 [[buffer(4)]],                                     \
      device const uchar *d2 [[buffer(5)]],                                     \
      device const uchar *s2 [[buffer(6)]],                                     \
      device const uchar *d3 [[buffer(7)]],                                     \
      device const uchar *s3 [[buffer(8)]],                                     \
      device bfloat *y0 [[buffer(9)]],                                          \
      device bfloat *y1 [[buffer(10)]],                                         \
      device bfloat *y2 [[buffer(11)]],                                         \
      device bfloat *y3 [[buffer(12)]],                                         \
      constant PretiledQmvConcat4Mxfp4Params &p [[buffer(13)]],                 \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    constexpr uint RPTG = (ROWS_SG) * 2u;                                       \
    const uint tgs0 = (p.n0 + RPTG - 1u) / RPTG;                                \
    const uint tgs1 = (p.n1 + RPTG - 1u) / RPTG;                                \
    const uint tgs2 = (p.n2 + RPTG - 1u) / RPTG;                                \
    device const uchar *ds = d3;                                                \
    device const uchar *ss = s3;                                                \
    device bfloat *ys = y3;                                                     \
    uint ns = p.n3;                                                             \
    uint tgl = tg.x - tgs0 - tgs1 - tgs2;                                       \
    if (tg.x < tgs0) {                                                          \
      ds = d0; ss = s0; ys = y0; ns = p.n0; tgl = tg.x;                         \
    } else if (tg.x < tgs0 + tgs1) {                                            \
      ds = d1; ss = s1; ys = y1; ns = p.n1; tgl = tg.x - tgs0;                  \
    } else if (tg.x < tgs0 + tgs1 + tgs2) {                                     \
      ds = d2; ss = s2; ys = y2; ns = p.n2; tgl = tg.x - tgs0 - tgs1;           \
    }                                                                           \
    pretiled_qmv_mxfp4_kl32_impl<ROWS_SG, 16u>(x, ds, ss, ys, ns, p.k, tgl, sg, lane); \
  }

OMLX_M5_DEF_MXFP4_CONCAT4_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat4_r4_t1, 4u)
OMLX_M5_DEF_MXFP4_CONCAT4_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat4_r2_t1, 2u)

// Zero-uitofp IEEE-754 0x6400 bitcast FP32 8-groups/iter (512 K-elems/step, 4 lanes/group,
// uint2 8B / uint4 16B coalesced loads, ZERO simd_shuffle_xor, ZERO overflow!).
template <uint BITS>
inline float pretiled_affine_dot16_f32(
    device const uchar *c_tile,
    device const uchar *h_tile,
    uint rn,
    uint k_l,
    thread const float (&xf)[16]) {
  if constexpr (BITS == 4u) {
    (void)h_tile;
    const uint2 cw =
        reinterpret_cast<device const uint2 *>(c_tile + ulong(rn) * 32ul)[k_l];
    float d = 0.0f;
#pragma clang loop unroll(full)
    for (uint w = 0u; w < 2u; ++w) {
      const uint v = (w == 0u) ? cw.x : cw.y;
#pragma clang loop unroll(full)
      for (uint b = 0u; b < 4u; ++b) {
        const uint byte_v = (v >> (b * 8u)) & 0xFFu;
        d = fma(xf[w * 8u + b * 2u + 0u],
                float(as_type<half>(ushort(0x6400u | (byte_v & 0x0Fu)))), d);
        d = fma(xf[w * 8u + b * 2u + 1u],
                float(as_type<half>(ushort(0x6400u | (byte_v >> 4u)))), d);
      }
    }
    return d;
  } else if constexpr (BITS == 6u) {
    const uint2 cw =
        reinterpret_cast<device const uint2 *>(c_tile + ulong(rn) * 32ul)[k_l];
    const uint hw =
        reinterpret_cast<device const uint *>(h_tile + ulong(rn) * 16ul)[k_l];
    float d = 0.0f;
#pragma clang loop unroll(full)
    for (uint j = 0u; j < 8u; ++j) {
      const uint lo0 = (cw.x >> (j * 4u)) & 0xFu;
      const uint hi0 = (hw >> (j * 2u)) & 3u;
      d = fma(xf[j], float(as_type<half>(ushort(0x6400u | lo0 | (hi0 << 4u)))), d);
    }
#pragma clang loop unroll(full)
    for (uint j = 0u; j < 8u; ++j) {
      const uint lo1 = (cw.y >> (j * 4u)) & 0xFu;
      const uint hi1 = (hw >> (16u + j * 2u)) & 3u;
      d = fma(xf[8u + j],
              float(as_type<half>(ushort(0x6400u | lo1 | (hi1 << 4u)))), d);
    }
    return d;
  } else if constexpr (BITS == 5u) {
    const uint2 cw =
        reinterpret_cast<device const uint2 *>(c_tile + ulong(rn) * 32ul)[k_l];
    const uint hw = uint(
        reinterpret_cast<device const ushort *>(h_tile + ulong(rn) * 8ul)[k_l]);
    float d = 0.0f;
#pragma clang loop unroll(full)
    for (uint j = 0u; j < 8u; ++j) {
      const uint lo0 = (cw.x >> (j * 4u)) & 0xFu;
      const uint hi0 = (hw >> j) & 1u;
      d = fma(xf[j], float(as_type<half>(ushort(0x6400u | lo0 | (hi0 << 4u)))), d);
    }
#pragma clang loop unroll(full)
    for (uint j = 0u; j < 8u; ++j) {
      const uint lo1 = (cw.y >> (j * 4u)) & 0xFu;
      const uint hi1 = (hw >> (8u + j)) & 1u;
      d = fma(xf[8u + j],
              float(as_type<half>(ushort(0x6400u | lo1 | (hi1 << 4u)))), d);
    }
    return d;
  } else if constexpr (BITS == 8u) {
    (void)h_tile;
    const uint4 cb =
        reinterpret_cast<device const uint4 *>(c_tile + ulong(rn) * 64ul)[k_l];
    float d = 0.0f;
#pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      const uint v = cb[w];
      d = fma(xf[w * 4u + 0u],
              float(as_type<half>(ushort(0x6400u | (v & 0xFFu)))), d);
      d = fma(xf[w * 4u + 1u],
              float(as_type<half>(ushort(0x6400u | ((v >> 8u) & 0xFFu)))), d);
      d = fma(xf[w * 4u + 2u],
              float(as_type<half>(ushort(0x6400u | ((v >> 16u) & 0xFFu)))), d);
      d = fma(xf[w * 4u + 3u],
              float(as_type<half>(ushort(0x6400u | (v >> 24u)))), d);
    }
    return d;
  } else {
    const uint lw =
        reinterpret_cast<device const uint *>(c_tile + ulong(rn) * 16ul)[k_l];
    const ushort hw =
        reinterpret_cast<device const ushort *>(h_tile + ulong(rn) * 8ul)[k_l];
    float d = 0.0f;
#pragma clang loop unroll(full)
    for (uint j = 0u; j < 16u; ++j) {
      const uint lo2 = (lw >> (j * 2u)) & 3u;
      const uint hi1 = (uint(hw) >> j) & 1u;
      d = fma(xf[j], float(as_type<half>(ushort(0x6400u | lo2 | (hi1 << 2u)))), d);
    }
    return d;
  }
}

inline float pretiled_affine_dot16_f32_dyn(
    device const uchar *c_tile,
    device const uchar *h_tile,
    uint rn,
    uint bits,
    uint k_l,
    thread const float (&xf)[16]) {
  if (bits == 4u) return pretiled_affine_dot16_f32<4u>(c_tile, h_tile, rn, k_l, xf);
  if (bits == 6u) return pretiled_affine_dot16_f32<6u>(c_tile, h_tile, rn, k_l, xf);
  if (bits == 8u) return pretiled_affine_dot16_f32<8u>(c_tile, h_tile, rn, k_l, xf);
  if (bits == 5u) return pretiled_affine_dot16_f32<5u>(c_tile, h_tile, rn, k_l, xf);
  return pretiled_affine_dot16_f32<3u>(c_tile, h_tile, rn, k_l, xf);
}

template <bool PROLOGUE_NORM>
__attribute__((always_inline)) inline void pretiled_affine_load16_t1(
    device const bfloat *x,
    device const bfloat *rms_w,
    ulong group_base,
    uint k_l,
    uint lane,
    bool valid_g,
    bool is_lane_v2,
    bool do_fwht,
    float inv_rms,
    float4 off4,
    thread float (&xf)[16],
    thread float &local_sum) {
  float4 v[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
  if (valid_g) {
    if (is_lane_v2) {
      device const bfloat4 *xb4 = reinterpret_cast<device const bfloat4 *>(x + group_base);
      if constexpr (PROLOGUE_NORM) {
        device const bfloat4 *wb4 = reinterpret_cast<device const bfloat4 *>(rms_w + group_base);
#pragma clang loop unroll(full)
        for (uint s = 0u; s < 4u; ++s) {
          const uint idx4 = s * 4u + k_l;
          v[s] = float4(bfloat4(float4(xb4[idx4]) * inv_rms * (float4(wb4[idx4]) + off4)));
        }
      } else {
        (void)rms_w; (void)inv_rms; (void)off4;
#pragma clang loop unroll(full)
        for (uint s = 0u; s < 4u; ++s) {
          v[s] = float4(xb4[s * 4u + k_l]);
        }
      }
    } else {
      const ulong x_off = group_base + ulong(k_l) * 16ul;
      device const bfloat4 *xb4 = reinterpret_cast<device const bfloat4 *>(x + x_off);
      if constexpr (PROLOGUE_NORM) {
        device const bfloat4 *wb4 = reinterpret_cast<device const bfloat4 *>(rms_w + x_off);
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 4u; ++i) {
          v[i] = float4(bfloat4(float4(xb4[i]) * inv_rms * (float4(wb4[i]) + off4)));
        }
      } else {
        (void)rms_w; (void)inv_rms; (void)off4;
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 4u; ++i) {
          v[i] = float4(xb4[i]);
        }
      }
    }
  }
  if (do_fwht) {
    if (is_lane_v2) {
      omlx_m5_fwht64_sub4_pi64(v, lane);
    } else {
      omlx_m5_fwht64_sub4_linear(v, lane);
    }
  }
  local_sum = 0.0f;
#pragma clang loop unroll(full)
  for (uint i = 0u; i < 4u; ++i) {
    xf[4u * i + 0u] = v[i].x;
    xf[4u * i + 1u] = v[i].y;
    xf[4u * i + 2u] = v[i].z;
    xf[4u * i + 3u] = v[i].w;
    local_sum += (v[i].x + v[i].y) + (v[i].z + v[i].w);
  }
}

template <uint ROWS_SG, uint BITS, uint BN, bool PROLOGUE_NORM = false, bool EPILOGUE_ADD_SQ = false>
inline void pretiled_qmv_affine_kl32_impl(
    device const bfloat *x,
    device const uchar *nax_codes,
    device const uchar *nax_high,
    device const uint *nax_sb,
    device bfloat *y,
    uint n,
    uint k,
    uint tg_local,
    uint sg,
    uint lane,
    device const bfloat *rms_w = nullptr,
    float inv_rms = 1.0f,
    float w_off = 0.0f,
    device atomic_float *rms_sq_buf = nullptr,
    uint sq_slot = 0u) {
  static_assert(BN == 16u || BN == 32u, "BN must be 16 or 32");
  static_assert(ROWS_SG == 2u || ROWS_SG == 4u, "ROWS_SG must be 2 or 4");
  const uint row_sg = (tg_local * 2u + sg) * ROWS_SG;
  if (row_sg >= n) {
    return;
  }
  const bool is_lane_v2 = (k & 0x40000000u) != 0u;
  const bool do_fwht = (k & 0x80000000u) != 0u;
  const uint k_eff = k & 0x3FFFFFFFu;
  const uint groups = k_eff / 64u;
  const uint tile_n = row_sg / BN;
  const uint rn0 = row_sg % BN;
  constexpr ulong code_tile_bytes =
      ulong(BN) * ((BITS == 3u) ? 16ul : ((BITS == 8u) ? 64ul : 32ul));
  constexpr ulong high_tile_bytes =
      ulong(BN) * ((BITS == 3u || BITS == 5u) ? 8ul : ((BITS == 6u) ? 16ul : 0ul));

  const uint g_sub = lane >> 2u; // 0..7 (8 groups of 64 = 512 K-elems/iter!)
  const uint k_l = lane & 3u;    // 0..3 (16 elems = 8B uint2 per lane!)
  float acc[ROWS_SG];
#pragma clang loop unroll(full)
  for (uint r = 0u; r < ROWS_SG; ++r) {
    acc[r] = 0.0f;
  }

  const float4 off4 = float4(w_off);
  const ulong tile_base = ulong(tile_n) * groups;
  for (uint g = 0u; g < groups; g += 8u) {
    const uint cur_g = g + g_sub;
    float xf[16];
    float local_sum = 0.0f;
    pretiled_affine_load16_t1<PROLOGUE_NORM>(
        x, rms_w, ulong(cur_g) * 64ul, k_l, lane, cur_g < groups,
        is_lane_v2, do_fwht, inv_rms, off4, xf, local_sum);
    if (cur_g < groups) {
      const ulong idx = tile_base + cur_g;
      device const uchar *c_tile = nax_codes + idx * code_tile_bytes;
      device const uchar *h_tile = nax_high + idx * high_tile_bytes;
      device const uint *sb_row = nax_sb + idx * ulong(BN) + rn0;
      if constexpr (ROWS_SG == 4u) {
        const uint4 sb4 = *reinterpret_cast<device const uint4 *>(sb_row);
#pragma clang loop unroll(full)
        for (uint r = 0u; r < 4u; ++r) {
          const uint sb_p = sb4[r];
          const float sc = as_type<float>((sb_p & 0xFFFFu) << 16u);
          const float bi = as_type<float>(sb_p & 0xFFFF0000u) - 1024.0f * sc;
          const float d =
              pretiled_affine_dot16_f32<BITS>(c_tile, h_tile, rn0 + r, k_l, xf);
          acc[r] = fma(sc, d, fma(bi, local_sum, acc[r]));
        }
      } else {
        const uint2 sb2 = *reinterpret_cast<device const uint2 *>(sb_row);
#pragma clang loop unroll(full)
        for (uint r = 0u; r < 2u; ++r) {
          const uint sb_p = sb2[r];
          const float sc = as_type<float>((sb_p & 0xFFFFu) << 16u);
          const float bi = as_type<float>(sb_p & 0xFFFF0000u) - 1024.0f * sc;
          const float d =
              pretiled_affine_dot16_f32<BITS>(c_tile, h_tile, rn0 + r, k_l, xf);
          acc[r] = fma(sc, d, fma(bi, local_sum, acc[r]));
        }
      }
    }
  }
  if constexpr (EPILOGUE_ADD_SQ) {
    float sg_sq = 0.0f;
#pragma clang loop unroll(full)
    for (uint r = 0u; r < ROWS_SG; ++r) {
      const float v = simd_sum(acc[r]);
      if (lane == 0u && (row_sg + r) < n) {
        const bfloat h_new = bfloat(float(y[row_sg + r]) + float(bfloat(v)));
        y[row_sg + r] = h_new;
        const float hf = float(h_new);
        sg_sq = fma(hf, hf, sg_sq);
      }
    }
    if (lane == 0u) {
      atomic_fetch_add_explicit(&rms_sq_buf[sq_slot], sg_sq, memory_order_relaxed);
    }
  } else {
    (void)rms_sq_buf; (void)sq_slot;
#pragma clang loop unroll(full)
    for (uint r = 0u; r < ROWS_SG; ++r) {
      const float v = simd_sum(acc[r]);
      if (lane == 0u && (row_sg + r) < n) {
        y[row_sg + r] = bfloat(v);
      }
    }
  }
}

template <uint ROWS_SG, uint BITS, bool PROLOGUE_NORM = false>
inline void pretiled_qmv_affine_swiglu_kl32_impl(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const uint *g_sb,
    device const uchar *u_codes,
    device const uchar *u_high,
    device const uint *u_sb,
    device bfloat *y,
    uint n,
    uint k,
    uint tg_local,
    uint sg,
    uint lane,
    device const bfloat *rms_w = nullptr,
    float inv_rms = 1.0f,
    float w_off = 0.0f) {
  constexpr uint BN = 16u;
  static_assert(ROWS_SG == 2u || ROWS_SG == 4u, "ROWS_SG must be 2 or 4");
  const uint row_sg = (tg_local * 2u + sg) * ROWS_SG;
  if (row_sg >= n) {
    return;
  }
  const bool is_lane_v2 = (k & 0x40000000u) != 0u;
  const bool do_fwht = (k & 0x80000000u) != 0u;
  const uint k_eff = k & 0x3FFFFFFFu;
  const uint groups = k_eff / 64u;
  const uint tile_n = row_sg / BN;
  const uint rn0 = row_sg % BN;
  constexpr ulong c_tile_bytes =
      ulong(BN) * ((BITS == 3u) ? 16ul : ((BITS == 8u) ? 64ul : 32ul));
  constexpr ulong h_tile_bytes =
      ulong(BN) * ((BITS == 3u || BITS == 5u) ? 8ul : ((BITS == 6u) ? 16ul : 0ul));

  const uint g_sub = lane >> 2u;
  const uint k_l = lane & 3u;
  float acc_g[ROWS_SG], acc_u[ROWS_SG];
#pragma clang loop unroll(full)
  for (uint r = 0u; r < ROWS_SG; ++r) {
    acc_g[r] = 0.0f;
    acc_u[r] = 0.0f;
  }

  const float4 off4 = float4(w_off);
  const ulong tile_base = ulong(tile_n) * groups;
  for (uint g = 0u; g < groups; g += 8u) {
    const uint cur_g = g + g_sub;
    float xf[16];
    float local_sum = 0.0f;
    pretiled_affine_load16_t1<PROLOGUE_NORM>(
        x, rms_w, ulong(cur_g) * 64ul, k_l, lane, cur_g < groups,
        is_lane_v2, do_fwht, inv_rms, off4, xf, local_sum);
    if (cur_g < groups) {
      const ulong idx = tile_base + cur_g;
      device const uchar *gc_t = g_codes + idx * c_tile_bytes;
      device const uchar *gh_t = g_high + idx * h_tile_bytes;
      device const uchar *uc_t = u_codes + idx * c_tile_bytes;
      device const uchar *uh_t = u_high + idx * h_tile_bytes;
      if constexpr (ROWS_SG == 4u) {
        const uint4 gp4 = *reinterpret_cast<device const uint4 *>(g_sb + idx * ulong(BN) + rn0);
        const uint4 up4 = *reinterpret_cast<device const uint4 *>(u_sb + idx * ulong(BN) + rn0);
#pragma clang loop unroll(full)
        for (uint r = 0u; r < 4u; ++r) {
          const uint gp = gp4[r];
          const uint up = up4[r];
          const float gsc = as_type<float>((gp & 0xFFFFu) << 16u);
          const float gbi = as_type<float>(gp & 0xFFFF0000u) - 1024.0f * gsc;
          const float usc = as_type<float>((up & 0xFFFFu) << 16u);
          const float ubi = as_type<float>(up & 0xFFFF0000u) - 1024.0f * usc;
          acc_g[r] = fma(
              gsc, pretiled_affine_dot16_f32<BITS>(gc_t, gh_t, rn0 + r, k_l, xf),
              fma(gbi, local_sum, acc_g[r]));
          acc_u[r] = fma(
              usc, pretiled_affine_dot16_f32<BITS>(uc_t, uh_t, rn0 + r, k_l, xf),
              fma(ubi, local_sum, acc_u[r]));
        }
      } else {
        const uint2 gp2 = *reinterpret_cast<device const uint2 *>(g_sb + idx * ulong(BN) + rn0);
        const uint2 up2 = *reinterpret_cast<device const uint2 *>(u_sb + idx * ulong(BN) + rn0);
#pragma clang loop unroll(full)
        for (uint r = 0u; r < 2u; ++r) {
          const uint gp = gp2[r];
          const uint up = up2[r];
          const float gsc = as_type<float>((gp & 0xFFFFu) << 16u);
          const float gbi = as_type<float>(gp & 0xFFFF0000u) - 1024.0f * gsc;
          const float usc = as_type<float>((up & 0xFFFFu) << 16u);
          const float ubi = as_type<float>(up & 0xFFFF0000u) - 1024.0f * usc;
          acc_g[r] = fma(
              gsc, pretiled_affine_dot16_f32<BITS>(gc_t, gh_t, rn0 + r, k_l, xf),
              fma(gbi, local_sum, acc_g[r]));
          acc_u[r] = fma(
              usc, pretiled_affine_dot16_f32<BITS>(uc_t, uh_t, rn0 + r, k_l, xf),
              fma(ubi, local_sum, acc_u[r]));
        }
      }
    }
  }
#pragma clang loop unroll(full)
  for (uint r = 0u; r < ROWS_SG; ++r) {
    const float gs = simd_sum(acc_g[r]);
    const float us = simd_sum(acc_u[r]);
    if (lane == 0u && (row_sg + r) < n) {
      const float gv = float(bfloat(gs));
      const float uv = float(bfloat(us));
      y[row_sg + r] = bfloat((gv / (1.0f + exp(-gv))) * uv);
    }
  }
}

template <uint ROWS_SG, bool PROLOGUE_NORM = false>
inline void pretiled_qmv_affine_swiglu_kl32_mixed_fallback(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const uint *g_sb,
    device const uchar *u_codes,
    device const uchar *u_high,
    device const uint *u_sb,
    device bfloat *y,
    uint n,
    uint k,
    uint g_bits,
    uint u_bits,
    uint tg_local,
    uint sg,
    uint lane,
    device const bfloat *rms_w = nullptr,
    float inv_rms = 1.0f,
    float w_off = 0.0f) {
  const uint bn = (n >= 65536u) ? 32u : 16u;
  const uint row_sg = (tg_local * 2u + sg) * ROWS_SG;
  if (row_sg >= n) {
    return;
  }
  const bool is_lane_v2 = (k & 0x40000000u) != 0u;
  const bool do_fwht = (k & 0x80000000u) != 0u;
  const uint k_eff = k & 0x3FFFFFFFu;
  const uint groups = k_eff / 64u;
  const uint tile_n = row_sg / bn;
  const uint rn0 = row_sg % bn;
  const ulong gc_tile_bytes =
      ulong(bn) * ((g_bits == 3u) ? 16ul : ((g_bits == 8u) ? 64ul : 32ul));
  const ulong gh_tile_bytes =
      ulong(bn) * ((g_bits == 3u || g_bits == 5u) ? 8ul : ((g_bits == 6u) ? 16ul : 0ul));
  const ulong uc_tile_bytes =
      ulong(bn) * ((u_bits == 3u) ? 16ul : ((u_bits == 8u) ? 64ul : 32ul));
  const ulong uh_tile_bytes =
      ulong(bn) * ((u_bits == 3u || u_bits == 5u) ? 8ul : ((u_bits == 6u) ? 16ul : 0ul));

  const uint g_sub = lane >> 2u;
  const uint k_l = lane & 3u;
  float acc_g[ROWS_SG], acc_u[ROWS_SG];
#pragma clang loop unroll(full)
  for (uint r = 0u; r < ROWS_SG; ++r) {
    acc_g[r] = 0.0f;
    acc_u[r] = 0.0f;
  }

  const float4 off4 = float4(w_off);
  const ulong tile_base = ulong(tile_n) * groups;
  for (uint g = 0u; g < groups; g += 8u) {
    const uint cur_g = g + g_sub;
    float xf[16];
    float local_sum = 0.0f;
    pretiled_affine_load16_t1<PROLOGUE_NORM>(
        x, rms_w, ulong(cur_g) * 64ul, k_l, lane, cur_g < groups,
        is_lane_v2, do_fwht, inv_rms, off4, xf, local_sum);
    if (cur_g < groups) {
      const ulong idx = tile_base + cur_g;
      device const uchar *gc_t = g_codes + idx * gc_tile_bytes;
      device const uchar *gh_t = g_high + idx * gh_tile_bytes;
      device const uchar *uc_t = u_codes + idx * uc_tile_bytes;
      device const uchar *uh_t = u_high + idx * uh_tile_bytes;
      device const uint *gsb_r = g_sb + idx * ulong(bn) + rn0;
      device const uint *usb_r = u_sb + idx * ulong(bn) + rn0;
#pragma clang loop unroll(full)
      for (uint r = 0u; r < ROWS_SG; ++r) {
        const uint gp = gsb_r[r];
        const uint up = usb_r[r];
        const float gsc = as_type<float>((gp & 0xFFFFu) << 16u);
        const float gbi = as_type<float>(gp & 0xFFFF0000u) - 1024.0f * gsc;
        const float usc = as_type<float>((up & 0xFFFFu) << 16u);
        const float ubi = as_type<float>(up & 0xFFFF0000u) - 1024.0f * usc;
        acc_g[r] = fma(
            gsc, pretiled_affine_dot16_f32_dyn(gc_t, gh_t, rn0 + r, g_bits, k_l, xf),
            fma(gbi, local_sum, acc_g[r]));
        acc_u[r] = fma(
            usc, pretiled_affine_dot16_f32_dyn(uc_t, uh_t, rn0 + r, u_bits, k_l, xf),
            fma(ubi, local_sum, acc_u[r]));
      }
    }
  }
#pragma clang loop unroll(full)
  for (uint r = 0u; r < ROWS_SG; ++r) {
    const float gs = simd_sum(acc_g[r]);
    const float us = simd_sum(acc_u[r]);
    if (lane == 0u && (row_sg + r) < n) {
      const float gv = float(bfloat(gs));
      const float uv = float(bfloat(us));
      y[row_sg + r] = bfloat((gv / (1.0f + exp(-gv))) * uv);
    }
  }
}

template <uint ROWS_SG>
inline void pretiled_qmv_affine_kl32_bn16_dispatch(
    device const bfloat *x,
    device const uchar *c,
    device const uchar *h,
    device const uint *sb,
    device bfloat *y,
    uint n,
    uint k,
    uint bits,
    uint tg_l,
    uint sg,
    uint lane) {
  if (bits == 4u) {
    pretiled_qmv_affine_kl32_impl<ROWS_SG, 4u, 16u>(x, c, h, sb, y, n, k, tg_l, sg, lane);
  } else if (bits == 6u) {
    pretiled_qmv_affine_kl32_impl<ROWS_SG, 6u, 16u>(x, c, h, sb, y, n, k, tg_l, sg, lane);
  } else if (bits == 8u) {
    pretiled_qmv_affine_kl32_impl<ROWS_SG, 8u, 16u>(x, c, h, sb, y, n, k, tg_l, sg, lane);
  } else if (bits == 5u) {
    pretiled_qmv_affine_kl32_impl<ROWS_SG, 5u, 16u>(x, c, h, sb, y, n, k, tg_l, sg, lane);
  } else {
    pretiled_qmv_affine_kl32_impl<ROWS_SG, 3u, 16u>(x, c, h, sb, y, n, k, tg_l, sg, lane);
  }
}

template <uint ROWS_SG>
inline void pretiled_qmv_affine_kl32_dispatch(
    device const bfloat *x,
    device const uchar *c,
    device const uchar *h,
    device const uint *sb,
    device bfloat *y,
    uint n,
    uint k,
    uint bits,
    uint tg_l,
    uint sg,
    uint lane) {
  if (n >= 65536u) {
    if (bits == 6u) {
      pretiled_qmv_affine_kl32_impl<ROWS_SG, 6u, 32u>(x, c, h, sb, y, n, k, tg_l, sg, lane);
    } else if (bits == 4u) {
      pretiled_qmv_affine_kl32_impl<ROWS_SG, 4u, 32u>(x, c, h, sb, y, n, k, tg_l, sg, lane);
    } else if (bits == 8u) {
      pretiled_qmv_affine_kl32_impl<ROWS_SG, 8u, 32u>(x, c, h, sb, y, n, k, tg_l, sg, lane);
    } else if (bits == 5u) {
      pretiled_qmv_affine_kl32_impl<ROWS_SG, 5u, 32u>(x, c, h, sb, y, n, k, tg_l, sg, lane);
    } else {
      pretiled_qmv_affine_kl32_impl<ROWS_SG, 3u, 32u>(x, c, h, sb, y, n, k, tg_l, sg, lane);
    }
  } else {
    pretiled_qmv_affine_kl32_bn16_dispatch<ROWS_SG>(x, c, h, sb, y, n, k, bits, tg_l, sg, lane);
  }
}

#define OMLX_M5_DEF_AFFINE_T1_SPEC(NAME, ROWS_SG, BITS, BN)                      \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *c [[buffer(1)]],                                      \
      device const uchar *h [[buffer(2)]],                                      \
      device const uint *sb [[buffer(3)]],                                      \
      device bfloat *y [[buffer(4)]],                                           \
      constant PretiledQmvAffineParams &p [[buffer(5)]],                        \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    if (p.n1 == 0u) return;                                                     \
    pretiled_qmv_affine_kl32_impl<ROWS_SG, BITS, BN>(                           \
        x, c, h, sb, y, p.n0, p.k, tg.x, sg, lane);                             \
  }

OMLX_M5_DEF_AFFINE_T1_SPEC(omlx_m5_pretiled_qmv_affine_q4_bn16_r4_t1, 4u, 4u, 16u)
OMLX_M5_DEF_AFFINE_T1_SPEC(omlx_m5_pretiled_qmv_affine_q4_bn16_r2_t1, 2u, 4u, 16u)
OMLX_M5_DEF_AFFINE_T1_SPEC(omlx_m5_pretiled_qmv_affine_q4_bn32_r4_t1, 4u, 4u, 32u)
OMLX_M5_DEF_AFFINE_T1_SPEC(omlx_m5_pretiled_qmv_affine_q5_bn16_r4_t1, 4u, 5u, 16u)
OMLX_M5_DEF_AFFINE_T1_SPEC(omlx_m5_pretiled_qmv_affine_q5_bn16_r2_t1, 2u, 5u, 16u)
OMLX_M5_DEF_AFFINE_T1_SPEC(omlx_m5_pretiled_qmv_affine_q5_bn32_r4_t1, 4u, 5u, 32u)
OMLX_M5_DEF_AFFINE_T1_SPEC(omlx_m5_pretiled_qmv_affine_q6_bn16_r4_t1, 4u, 6u, 16u)
OMLX_M5_DEF_AFFINE_T1_SPEC(omlx_m5_pretiled_qmv_affine_q6_bn16_r2_t1, 2u, 6u, 16u)
OMLX_M5_DEF_AFFINE_T1_SPEC(omlx_m5_pretiled_qmv_affine_q6_bn32_r4_t1, 4u, 6u, 32u)
OMLX_M5_DEF_AFFINE_T1_SPEC(omlx_m5_pretiled_qmv_affine_q8_bn16_r4_t1, 4u, 8u, 16u)
OMLX_M5_DEF_AFFINE_T1_SPEC(omlx_m5_pretiled_qmv_affine_q8_bn16_r2_t1, 2u, 8u, 16u)
OMLX_M5_DEF_AFFINE_T1_SPEC(omlx_m5_pretiled_qmv_affine_q8_bn32_r4_t1, 4u, 8u, 32u)

#define OMLX_M5_DEF_AFFINE_SWIGLU_T1_SPEC(NAME, ROWS_SG, BITS)                   \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *gc [[buffer(1)]],                                     \
      device const uchar *gh [[buffer(2)]],                                     \
      device const uint *gsb [[buffer(3)]],                                     \
      device const uchar *uc [[buffer(4)]],                                     \
      device const uchar *uh [[buffer(5)]],                                     \
      device const uint *usb [[buffer(6)]],                                     \
      device bfloat *y [[buffer(7)]],                                           \
      constant PretiledQmvAffineParams &p [[buffer(8)]],                        \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    if (p.n1 == 0u) return;                                                     \
    pretiled_qmv_affine_swiglu_kl32_impl<ROWS_SG, BITS>(                        \
        x, gc, gh, gsb, uc, uh, usb, y, p.n0, p.k, tg.x, sg, lane);             \
  }

OMLX_M5_DEF_AFFINE_SWIGLU_T1_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q4_r4_t1, 4u, 4u)
OMLX_M5_DEF_AFFINE_SWIGLU_T1_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q4_r2_t1, 2u, 4u)
OMLX_M5_DEF_AFFINE_SWIGLU_T1_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q5_r4_t1, 4u, 5u)
OMLX_M5_DEF_AFFINE_SWIGLU_T1_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q5_r2_t1, 2u, 5u)
OMLX_M5_DEF_AFFINE_SWIGLU_T1_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q6_r4_t1, 4u, 6u)
OMLX_M5_DEF_AFFINE_SWIGLU_T1_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q6_r2_t1, 2u, 6u)
OMLX_M5_DEF_AFFINE_SWIGLU_T1_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q8_r4_t1, 4u, 8u)
OMLX_M5_DEF_AFFINE_SWIGLU_T1_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q8_r2_t1, 2u, 8u)

template <uint ROWS_SG>
inline void pretiled_qmv_affine_kl32_bn16_q456_dispatch(
    device const bfloat *x,
    device const uchar *c,
    device const uchar *h,
    device const uint *sb,
    device bfloat *y,
    uint n,
    uint k,
    uint bits,
    uint tg_l,
    uint sg,
    uint lane) {
  if (bits == 5u) {
    pretiled_qmv_affine_kl32_impl<ROWS_SG, 5u, 16u>(x, c, h, sb, y, n, k, tg_l, sg, lane);
  } else if (bits == 4u) {
    pretiled_qmv_affine_kl32_impl<ROWS_SG, 4u, 16u>(x, c, h, sb, y, n, k, tg_l, sg, lane);
  } else {
    pretiled_qmv_affine_kl32_impl<ROWS_SG, 6u, 16u>(x, c, h, sb, y, n, k, tg_l, sg, lane);
  }
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_affine_t1(
    device const bfloat *x [[buffer(0)]],
    device const uchar *nax_codes [[buffer(1)]],
    device const uchar *nax_high [[buffer(2)]],
    device const uint *nax_sb [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant PretiledQmvAffineParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if (p.n1 == 0u) {
    return;
  }
  const uint bits = p.bits & 0xFFu;
  if (p.n0 >= 2048u) {
    pretiled_qmv_affine_kl32_dispatch<4u>(
        x, nax_codes, nax_high, nax_sb, y, p.n0, p.k, bits, tg.x, sg, lane);
  } else {
    pretiled_qmv_affine_kl32_dispatch<2u>(
        x, nax_codes, nax_high, nax_sb, y, p.n0, p.k, bits, tg.x, sg, lane);
  }
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_affine_swiglu_t1(
    device const bfloat *x [[buffer(0)]],
    device const uchar *g_codes [[buffer(1)]],
    device const uchar *g_high [[buffer(2)]],
    device const uint *g_sb [[buffer(3)]],
    device const uchar *u_codes [[buffer(4)]],
    device const uchar *u_high [[buffer(5)]],
    device const uint *u_sb [[buffer(6)]],
    device bfloat *y [[buffer(7)]],
    constant PretiledQmvAffineParams &p [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if (p.n1 == 0u) {
    return;
  }
  const uint g_bits = p.bits & 0xFFu;
  const uint u_hi = (p.bits >> 8u) & 0xFFu;
  const uint u_bits = (u_hi != 0u) ? u_hi : g_bits;
  if (p.n0 >= 2048u) {
    if (g_bits == u_bits && p.n0 < 65536u) {
      if (g_bits == 4u) { pretiled_qmv_affine_swiglu_kl32_impl<4u, 4u>(x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p.n0, p.k, tg.x, sg, lane); return; }
      if (g_bits == 6u) { pretiled_qmv_affine_swiglu_kl32_impl<4u, 6u>(x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p.n0, p.k, tg.x, sg, lane); return; }
      if (g_bits == 8u) { pretiled_qmv_affine_swiglu_kl32_impl<4u, 8u>(x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p.n0, p.k, tg.x, sg, lane); return; }
      if (g_bits == 5u) { pretiled_qmv_affine_swiglu_kl32_impl<4u, 5u>(x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p.n0, p.k, tg.x, sg, lane); return; }
      if (g_bits == 3u) { pretiled_qmv_affine_swiglu_kl32_impl<4u, 3u>(x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p.n0, p.k, tg.x, sg, lane); return; }
    }
    pretiled_qmv_affine_swiglu_kl32_mixed_fallback<4u>(
        x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p.n0, p.k, g_bits, u_bits, tg.x, sg, lane);
  } else {
    if (g_bits == u_bits && p.n0 < 65536u) {
      if (g_bits == 4u) { pretiled_qmv_affine_swiglu_kl32_impl<2u, 4u>(x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p.n0, p.k, tg.x, sg, lane); return; }
      if (g_bits == 6u) { pretiled_qmv_affine_swiglu_kl32_impl<2u, 6u>(x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p.n0, p.k, tg.x, sg, lane); return; }
      if (g_bits == 8u) { pretiled_qmv_affine_swiglu_kl32_impl<2u, 8u>(x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p.n0, p.k, tg.x, sg, lane); return; }
      if (g_bits == 5u) { pretiled_qmv_affine_swiglu_kl32_impl<2u, 5u>(x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p.n0, p.k, tg.x, sg, lane); return; }
      if (g_bits == 3u) { pretiled_qmv_affine_swiglu_kl32_impl<2u, 3u>(x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p.n0, p.k, tg.x, sg, lane); return; }
    }
    pretiled_qmv_affine_swiglu_kl32_mixed_fallback<2u>(
        x, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p.n0, p.k, g_bits, u_bits, tg.x, sg, lane);
  }
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_affine_concat2_t1(
    device const bfloat *x [[buffer(0)]],
    device const uchar *c0 [[buffer(1)]],
    device const uchar *h0 [[buffer(2)]],
    device const uint *sb0 [[buffer(3)]],
    device const uchar *c1 [[buffer(4)]],
    device const uchar *h1 [[buffer(5)]],
    device const uint *sb1 [[buffer(6)]],
    device bfloat *y0 [[buffer(7)]],
    device bfloat *y1 [[buffer(8)]],
    constant PretiledQmvAffineParams &p [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint b0 = p.bits & 0xFFu;
  const uint b1_raw = (p.bits >> 8u) & 0xFFu;
  const uint b1 = (b1_raw != 0u) ? b1_raw : b0;
  if ((p.n0 + p.n1) >= 2048u) {
    const uint tgs0 = (p.n0 + 7u) / 8u;
    const bool s0_sel = (tg.x < tgs0);
    pretiled_qmv_affine_kl32_bn16_dispatch<4u>(
        x, s0_sel ? c0 : c1, s0_sel ? h0 : h1, s0_sel ? sb0 : sb1,
        s0_sel ? y0 : y1, s0_sel ? p.n0 : p.n1, p.k, s0_sel ? b0 : b1,
        s0_sel ? tg.x : (tg.x - tgs0), sg, lane);
  } else {
    const uint tgs0 = (p.n0 + 3u) / 4u;
    const bool s0_sel = (tg.x < tgs0);
    pretiled_qmv_affine_kl32_bn16_dispatch<2u>(
        x, s0_sel ? c0 : c1, s0_sel ? h0 : h1, s0_sel ? sb0 : sb1,
        s0_sel ? y0 : y1, s0_sel ? p.n0 : p.n1, p.k, s0_sel ? b0 : b1,
        s0_sel ? tg.x : (tg.x - tgs0), sg, lane);
  }
}

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_affine_concat3_t1(
    device const bfloat *x [[buffer(0)]],
    device const uchar *c0 [[buffer(1)]],
    device const uchar *h0 [[buffer(2)]],
    device const uint *sb0 [[buffer(3)]],
    device const uchar *c1 [[buffer(4)]],
    device const uchar *h1 [[buffer(5)]],
    device const uint *sb1 [[buffer(6)]],
    device const uchar *c2 [[buffer(7)]],
    device const uchar *h2 [[buffer(8)]],
    device const uint *sb2 [[buffer(9)]],
    device bfloat *y0 [[buffer(10)]],
    device bfloat *y1 [[buffer(11)]],
    device bfloat *y2 [[buffer(12)]],
    constant PretiledQmvAffineParams &p [[buffer(13)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint b0 = p.bits & 0xFFu;
  const uint b1_raw = (p.bits >> 8u) & 0xFFu;
  const uint b2_raw = (p.bits >> 16u) & 0xFFu;
  const uint b1 = (b1_raw != 0u) ? b1_raw : b0;
  const uint b2 = (b2_raw != 0u) ? b2_raw : b0;
  if ((p.n0 + p.n1 + p.n2) >= 2048u) {
    const uint tgs0 = (p.n0 + 7u) / 8u;
    const uint tgs1 = (p.n1 + 7u) / 8u;
    device const uchar *cs = (tg.x < tgs0) ? c0 : ((tg.x < tgs0 + tgs1) ? c1 : c2);
    device const uchar *hs = (tg.x < tgs0) ? h0 : ((tg.x < tgs0 + tgs1) ? h1 : h2);
    device const uint *sbs = (tg.x < tgs0) ? sb0 : ((tg.x < tgs0 + tgs1) ? sb1 : sb2);
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2);
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2);
    const uint bs = (tg.x < tgs0) ? b0 : ((tg.x < tgs0 + tgs1) ? b1 : b2);
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1));
    pretiled_qmv_affine_kl32_bn16_dispatch<4u>(x, cs, hs, sbs, ys, ns, p.k, bs, tgl, sg, lane);
  } else {
    const uint tgs0 = (p.n0 + 3u) / 4u;
    const uint tgs1 = (p.n1 + 3u) / 4u;
    device const uchar *cs = (tg.x < tgs0) ? c0 : ((tg.x < tgs0 + tgs1) ? c1 : c2);
    device const uchar *hs = (tg.x < tgs0) ? h0 : ((tg.x < tgs0 + tgs1) ? h1 : h2);
    device const uint *sbs = (tg.x < tgs0) ? sb0 : ((tg.x < tgs0 + tgs1) ? sb1 : sb2);
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2);
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2);
    const uint bs = (tg.x < tgs0) ? b0 : ((tg.x < tgs0 + tgs1) ? b1 : b2);
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1));
    pretiled_qmv_affine_kl32_bn16_dispatch<2u>(x, cs, hs, sbs, ys, ns, p.k, bs, tgl, sg, lane);
  }
}

#define OMLX_M5_DEF_AFFINE_CONCAT3_Q456_SPEC(NAME, ROWS_SG)                      \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *c0 [[buffer(1)]],                                     \
      device const uchar *h0 [[buffer(2)]],                                     \
      device const uint *sb0 [[buffer(3)]],                                     \
      device const uchar *c1 [[buffer(4)]],                                     \
      device const uchar *h1 [[buffer(5)]],                                     \
      device const uint *sb1 [[buffer(6)]],                                     \
      device const uchar *c2 [[buffer(7)]],                                     \
      device const uchar *h2 [[buffer(8)]],                                     \
      device const uint *sb2 [[buffer(9)]],                                     \
      device bfloat *y0 [[buffer(10)]],                                         \
      device bfloat *y1 [[buffer(11)]],                                         \
      device bfloat *y2 [[buffer(12)]],                                         \
      constant PretiledQmvAffineParams &p [[buffer(13)]],                       \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    constexpr uint RPTG = (ROWS_SG) * 2u;                                       \
    const uint b0 = p.bits & 0xFFu;                                             \
    const uint b1_r = (p.bits >> 8u) & 0xFFu;                                   \
    const uint b2_r = (p.bits >> 16u) & 0xFFu;                                  \
    const uint b1 = (b1_r != 0u) ? b1_r : b0;                                   \
    const uint b2 = (b2_r != 0u) ? b2_r : b0;                                   \
    const uint tgs0 = (p.n0 + RPTG - 1u) / RPTG;                                \
    const uint tgs1 = (p.n1 + RPTG - 1u) / RPTG;                                \
    device const uchar *cs = (tg.x < tgs0) ? c0 : ((tg.x < tgs0 + tgs1) ? c1 : c2); \
    device const uchar *hs = (tg.x < tgs0) ? h0 : ((tg.x < tgs0 + tgs1) ? h1 : h2); \
    device const uint *sbs = (tg.x < tgs0) ? sb0 : ((tg.x < tgs0 + tgs1) ? sb1 : sb2); \
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2); \
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2); \
    const uint bs = (tg.x < tgs0) ? b0 : ((tg.x < tgs0 + tgs1) ? b1 : b2);     \
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1)); \
    pretiled_qmv_affine_kl32_bn16_q456_dispatch<ROWS_SG>(x, cs, hs, sbs, ys, ns, p.k, bs, tgl, sg, lane); \
  }

OMLX_M5_DEF_AFFINE_CONCAT3_Q456_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q456_r4_t1, 4u)
OMLX_M5_DEF_AFFINE_CONCAT3_Q456_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q456_r2_t1, 2u)

#define OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_SPEC(NAME, ROWS_SG, BITS)                \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *c0 [[buffer(1)]],                                     \
      device const uchar *h0 [[buffer(2)]],                                     \
      device const uint *sb0 [[buffer(3)]],                                     \
      device const uchar *c1 [[buffer(4)]],                                     \
      device const uchar *h1 [[buffer(5)]],                                     \
      device const uint *sb1 [[buffer(6)]],                                     \
      device const uchar *c2 [[buffer(7)]],                                     \
      device const uchar *h2 [[buffer(8)]],                                     \
      device const uint *sb2 [[buffer(9)]],                                     \
      device bfloat *y0 [[buffer(10)]],                                         \
      device bfloat *y1 [[buffer(11)]],                                         \
      device bfloat *y2 [[buffer(12)]],                                         \
      constant PretiledQmvAffineParams &p [[buffer(13)]],                       \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    constexpr uint RPTG = (ROWS_SG) * 2u;                                       \
    const uint tgs0 = (p.n0 + RPTG - 1u) / RPTG;                                \
    const uint tgs1 = (p.n1 + RPTG - 1u) / RPTG;                                \
    device const uchar *cs = (tg.x < tgs0) ? c0 : ((tg.x < tgs0 + tgs1) ? c1 : c2); \
    device const uchar *hs = (tg.x < tgs0) ? h0 : ((tg.x < tgs0 + tgs1) ? h1 : h2); \
    device const uint *sbs = (tg.x < tgs0) ? sb0 : ((tg.x < tgs0 + tgs1) ? sb1 : sb2); \
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2); \
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2); \
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1)); \
    pretiled_qmv_affine_kl32_impl<ROWS_SG, BITS, 16u>(x, cs, hs, sbs, ys, ns, p.k, tgl, sg, lane); \
  }

OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q4_r4_t1, 4u, 4u)
OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q5_r4_t1, 4u, 5u)
OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q6_r4_t1, 4u, 6u)

#define OMLX_M5_DEF_AFFINE_CONCAT4_SPEC(NAME, ROWS_SG, DISPATCH_FN)              \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *c0 [[buffer(1)]],                                     \
      device const uchar *h0 [[buffer(2)]],                                     \
      device const uint *sb0 [[buffer(3)]],                                     \
      device const uchar *c1 [[buffer(4)]],                                     \
      device const uchar *h1 [[buffer(5)]],                                     \
      device const uint *sb1 [[buffer(6)]],                                     \
      device const uchar *c2 [[buffer(7)]],                                     \
      device const uchar *h2 [[buffer(8)]],                                     \
      device const uint *sb2 [[buffer(9)]],                                     \
      device const uchar *c3 [[buffer(10)]],                                    \
      device const uchar *h3 [[buffer(11)]],                                    \
      device const uint *sb3 [[buffer(12)]],                                    \
      device bfloat *y0 [[buffer(13)]],                                         \
      device bfloat *y1 [[buffer(14)]],                                         \
      device bfloat *y2 [[buffer(15)]],                                         \
      device bfloat *y3 [[buffer(16)]],                                         \
      constant PretiledQmvConcat4AffineParams &p [[buffer(17)]],                \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    constexpr uint RPTG = (ROWS_SG) * 2u;                                       \
    const uint b0 = p.bits & 0xFFu;                                             \
    const uint b1_r = (p.bits >> 8u) & 0xFFu;                                   \
    const uint b2_r = (p.bits >> 16u) & 0xFFu;                                  \
    const uint b3_r = (p.bits >> 24u) & 0xFFu;                                  \
    const uint b1 = (b1_r != 0u) ? b1_r : b0;                                   \
    const uint b2 = (b2_r != 0u) ? b2_r : b0;                                   \
    const uint b3 = (b3_r != 0u) ? b3_r : b0;                                   \
    const uint tgs0 = (p.n0 + RPTG - 1u) / RPTG;                                \
    const uint tgs1 = (p.n1 + RPTG - 1u) / RPTG;                                \
    const uint tgs2 = (p.n2 + RPTG - 1u) / RPTG;                                \
    device const uchar *cs = c3;                                                \
    device const uchar *hs = h3;                                                \
    device const uint *sbs = sb3;                                               \
    device bfloat *ys = y3;                                                     \
    uint ns = p.n3;                                                             \
    uint bs = b3;                                                               \
    uint tgl = tg.x - tgs0 - tgs1 - tgs2;                                       \
    if (tg.x < tgs0) {                                                          \
      cs = c0; hs = h0; sbs = sb0; ys = y0; ns = p.n0; bs = b0; tgl = tg.x;     \
    } else if (tg.x < tgs0 + tgs1) {                                            \
      cs = c1; hs = h1; sbs = sb1; ys = y1; ns = p.n1; bs = b1; tgl = tg.x - tgs0; \
    } else if (tg.x < tgs0 + tgs1 + tgs2) {                                     \
      cs = c2; hs = h2; sbs = sb2; ys = y2; ns = p.n2; bs = b2; tgl = tg.x - tgs0 - tgs1; \
    }                                                                           \
    DISPATCH_FN<ROWS_SG>(x, cs, hs, sbs, ys, ns, p.k, bs, tgl, sg, lane);       \
  }

OMLX_M5_DEF_AFFINE_CONCAT4_SPEC(omlx_m5_pretiled_qmv_affine_concat4_r4_t1, 4u, pretiled_qmv_affine_kl32_bn16_dispatch)
OMLX_M5_DEF_AFFINE_CONCAT4_SPEC(omlx_m5_pretiled_qmv_affine_concat4_r2_t1, 2u, pretiled_qmv_affine_kl32_bn16_dispatch)
OMLX_M5_DEF_AFFINE_CONCAT4_SPEC(omlx_m5_pretiled_qmv_affine_concat4_q456_r4_t1, 4u, pretiled_qmv_affine_kl32_bn16_q456_dispatch)
OMLX_M5_DEF_AFFINE_CONCAT4_SPEC(omlx_m5_pretiled_qmv_affine_concat4_q456_r2_t1, 2u, pretiled_qmv_affine_kl32_bn16_q456_dispatch)

// One-time load-time GPU compaction of Q5 2-bit nax_high2 (16 B/row/tile) -> 1-bit q5_high1 (8 B/row/tile),
// cutting Q5 high-plane DRAM reads in half (-873 MB/tok on Qwen3.8-27B-oQ4e!) at T=1.
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_compact_q5_high2_to_high1(
    device const uint2 *src_high2 [[buffer(0)]],
    device uint *dst_high1 [[buffer(1)]],
    constant uint &num_u32_out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= num_u32_out) {
    return;
  }
  const uint2 s = src_high2[tid];
  uint x0 = s.x & 0x55555555u;
  x0 = (x0 | (x0 >> 1u)) & 0x33333333u;
  x0 = (x0 | (x0 >> 2u)) & 0x0F0F0F0Fu;
  x0 = (x0 | (x0 >> 4u)) & 0x00FF00FFu;
  const uint lo16 = (x0 | (x0 >> 8u)) & 0xFFFFu;

  uint x1 = s.y & 0x55555555u;
  x1 = (x1 | (x1 >> 1u)) & 0x33333333u;
  x1 = (x1 | (x1 >> 2u)) & 0x0F0F0F0Fu;
  x1 = (x1 | (x1 >> 4u)) & 0x00FF00FFu;
  const uint hi16 = (x1 | (x1 >> 8u)) & 0xFFFFu;

  dst_high1[tid] = lo16 | (hi16 << 16u);
}

struct TransposeAffineSbParams {
  uint groups;
  uint padded_n;
  uint bn;
  uint reserved;
};

// One-time load-time transpose of nax_sb [groups, padded_n] -> sb_tiled [tile_n, groups, bn]
// (64 B / 128 B contiguous per tile), eliminating 68 KB - 970 KB scattered TLB/cache thrashing!
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_transpose_affine_sb_to_tiled(
    device const uint *src_sb [[buffer(0)]],
    device uint *dst_sb_tiled [[buffer(1)]],
    constant TransposeAffineSbParams &p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint row = gid.x;
  const uint g = gid.y;
  if (row >= p.padded_n || g >= p.groups) {
    return;
  }
  const uint tile_n = row / p.bn;
  const uint rn = row % p.bn;
  dst_sb_tiled[(ulong(tile_n) * p.groups + g) * p.bn + rn] =
      src_sb[ulong(g) * p.padded_n + row];
}

// ============================================================================
// v0.8.0-m5rt.11 Bespoke Zero-Dispatch RMSNorm / Add-RMSNorm Epilogue & Prologue Kernels
// Eliminates all 128 (27B) / 80 (35B) standalone add_rmsnorm dispatches + full-chip barriers!
// ============================================================================

struct PretiledQmvFusedNormParams {
  uint n0;
  uint n1;
  uint n2;
  uint n3;
  uint k;
  uint bits;
  uint sq_slot;
  float inv_k;
  float eps;
  float w_off;
};

// --- MXFP4 Producers (Epilogue: hidden[row] += y[row], atomic_fetch_add(&rms_sq_buf[sq_slot], sq)) ---
#define OMLX_M5_DEF_MXFP4_ADD_SQ_SPEC(NAME, ROWS_SG, BN)                        \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *nax_data [[buffer(1)]],                               \
      device const uchar *nax_scale [[buffer(2)]],                              \
      device bfloat *hidden_inout [[buffer(3)]],                                \
      device atomic_float *rms_sq_buf [[buffer(4)]],                            \
      constant PretiledQmvFusedNormParams &p [[buffer(5)]],                     \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    pretiled_qmv_mxfp4_kl32_impl<ROWS_SG, BN, false, true>(                     \
        x, nax_data, nax_scale, hidden_inout, p.n0, p.k, tg.x, sg, lane,        \
        nullptr, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);                            \
  }

OMLX_M5_DEF_MXFP4_ADD_SQ_SPEC(omlx_m5_pretiled_qmv_mxfp4_bn16_r4_add_sq_t1, 4u, 16u)
OMLX_M5_DEF_MXFP4_ADD_SQ_SPEC(omlx_m5_pretiled_qmv_mxfp4_bn16_r2_add_sq_t1, 2u, 16u)

// --- Affine Producers (Epilogue: hidden[row] += y[row], atomic_fetch_add(&rms_sq_buf[sq_slot], sq)) ---
#define OMLX_M5_DEF_AFFINE_ADD_SQ_SPEC(NAME, ROWS_SG, BITS, BN)                 \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *c [[buffer(1)]],                                      \
      device const uchar *h [[buffer(2)]],                                      \
      device const uint *sb [[buffer(3)]],                                      \
      device bfloat *hidden_inout [[buffer(4)]],                                \
      device atomic_float *rms_sq_buf [[buffer(5)]],                            \
      constant PretiledQmvFusedNormParams &p [[buffer(6)]],                     \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    pretiled_qmv_affine_kl32_impl<ROWS_SG, BITS, BN, false, true>(              \
        x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane,                   \
        nullptr, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);                            \
  }

OMLX_M5_DEF_AFFINE_ADD_SQ_SPEC(omlx_m5_pretiled_qmv_affine_q4_bn16_r4_add_sq_t1, 4u, 4u, 16u)
OMLX_M5_DEF_AFFINE_ADD_SQ_SPEC(omlx_m5_pretiled_qmv_affine_q5_bn16_r4_add_sq_t1, 4u, 5u, 16u)
OMLX_M5_DEF_AFFINE_ADD_SQ_SPEC(omlx_m5_pretiled_qmv_affine_q6_bn16_r4_add_sq_t1, 4u, 6u, 16u)
OMLX_M5_DEF_AFFINE_ADD_SQ_SPEC(omlx_m5_pretiled_qmv_affine_q8_bn16_r4_add_sq_t1, 4u, 8u, 16u)

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_affine_add_sq_t1(
    device const bfloat *x [[buffer(0)]],
    device const uchar *c [[buffer(1)]],
    device const uchar *h [[buffer(2)]],
    device const uint *sb [[buffer(3)]],
    device bfloat *hidden_inout [[buffer(4)]],
    device atomic_float *rms_sq_buf [[buffer(5)]],
    constant PretiledQmvFusedNormParams &p [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint bits = p.bits & 0xFFu;
  if (p.n0 >= 2048u) {
    if (bits == 4u) pretiled_qmv_affine_kl32_impl<4u, 4u, 16u, false, true>(x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane, nullptr, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);
    else if (bits == 5u) pretiled_qmv_affine_kl32_impl<4u, 5u, 16u, false, true>(x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane, nullptr, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);
    else if (bits == 6u) pretiled_qmv_affine_kl32_impl<4u, 6u, 16u, false, true>(x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane, nullptr, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);
    else if (bits == 8u) pretiled_qmv_affine_kl32_impl<4u, 8u, 16u, false, true>(x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane, nullptr, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);
    else pretiled_qmv_affine_kl32_impl<4u, 3u, 16u, false, true>(x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane, nullptr, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);
  } else {
    if (bits == 4u) pretiled_qmv_affine_kl32_impl<2u, 4u, 16u, false, true>(x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane, nullptr, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);
    else if (bits == 5u) pretiled_qmv_affine_kl32_impl<2u, 5u, 16u, false, true>(x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane, nullptr, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);
    else if (bits == 6u) pretiled_qmv_affine_kl32_impl<2u, 6u, 16u, false, true>(x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane, nullptr, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);
    else if (bits == 8u) pretiled_qmv_affine_kl32_impl<2u, 8u, 16u, false, true>(x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane, nullptr, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);
    else pretiled_qmv_affine_kl32_impl<2u, 3u, 16u, false, true>(x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane, nullptr, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);
  }
}

// --- MXFP4 Consumers (Prologue: x_norm = hidden * rsqrt(rms_sq_buf[sq_slot] * inv_k + eps) * (rms_w + w_off)) ---
#define OMLX_M5_DEF_MXFP4_NORM_T1_SPEC(NAME, ROWS_SG, BN)                       \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *nax_data [[buffer(3)]],                               \
      device const uchar *nax_scale [[buffer(4)]],                              \
      device bfloat *y [[buffer(5)]],                                           \
      constant PretiledQmvFusedNormParams &p [[buffer(6)]],                     \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    const float inv = rsqrt(rms_sq_buf[p.sq_slot] * p.inv_k + p.eps);           \
    pretiled_qmv_mxfp4_kl32_impl<ROWS_SG, BN, true, false>(                     \
        hidden, nax_data, nax_scale, y, p.n0, p.k, tg.x, sg, lane,              \
        rms_w, inv, p.w_off);                                                   \
  }

OMLX_M5_DEF_MXFP4_NORM_T1_SPEC(omlx_m5_pretiled_qmv_mxfp4_bn32_r4_norm_t1, 4u, 32u)
OMLX_M5_DEF_MXFP4_NORM_T1_SPEC(omlx_m5_pretiled_qmv_mxfp4_bn16_r4_norm_t1, 4u, 16u)
OMLX_M5_DEF_MXFP4_NORM_T1_SPEC(omlx_m5_pretiled_qmv_mxfp4_bn16_r2_norm_t1, 2u, 16u)

#define OMLX_M5_DEF_MXFP4_SWIGLU_NORM_T1_SPEC(NAME, ROWS_SG, BN)                \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *gd [[buffer(3)]],                                     \
      device const uchar *gs [[buffer(4)]],                                     \
      device const uchar *ud [[buffer(5)]],                                     \
      device const uchar *us [[buffer(6)]],                                     \
      device bfloat *y [[buffer(7)]],                                           \
      constant PretiledQmvFusedNormParams &p [[buffer(8)]],                     \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    const float inv = rsqrt(rms_sq_buf[p.sq_slot] * p.inv_k + p.eps);           \
    pretiled_qmv_mxfp4_swiglu_kl32_impl<ROWS_SG, BN, true>(                     \
        hidden, gd, gs, ud, us, y, p.n0, p.k, tg.x, sg, lane,                   \
        rms_w, inv, p.w_off);                                                   \
  }

OMLX_M5_DEF_MXFP4_SWIGLU_NORM_T1_SPEC(omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r4_norm_t1, 4u, 16u)
OMLX_M5_DEF_MXFP4_SWIGLU_NORM_T1_SPEC(omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r2_norm_t1, 2u, 16u)

#define OMLX_M5_DEF_MXFP4_CONCAT3_NORM_SPEC(NAME, ROWS_SG)                      \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *d0 [[buffer(3)]],                                     \
      device const uchar *s0 [[buffer(4)]],                                     \
      device const uchar *d1 [[buffer(5)]],                                     \
      device const uchar *s1 [[buffer(6)]],                                     \
      device const uchar *d2 [[buffer(7)]],                                     \
      device const uchar *s2 [[buffer(8)]],                                     \
      device bfloat *y0 [[buffer(9)]],                                          \
      device bfloat *y1 [[buffer(10)]],                                         \
      device bfloat *y2 [[buffer(11)]],                                         \
      constant PretiledQmvFusedNormParams &p [[buffer(12)]],                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    constexpr uint RPTG = (ROWS_SG) * 2u;                                       \
    const uint tgs0 = (p.n0 + RPTG - 1u) / RPTG;                                \
    const uint tgs1 = (p.n1 + RPTG - 1u) / RPTG;                                \
    device const uchar *ds = (tg.x < tgs0) ? d0 : ((tg.x < tgs0 + tgs1) ? d1 : d2); \
    device const uchar *ss = (tg.x < tgs0) ? s0 : ((tg.x < tgs0 + tgs1) ? s1 : s2); \
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2); \
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2); \
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1)); \
    const float inv = rsqrt(rms_sq_buf[p.sq_slot] * p.inv_k + p.eps);           \
    pretiled_qmv_mxfp4_kl32_impl<ROWS_SG, 16u, true, false>(                    \
        hidden, ds, ss, ys, ns, p.k, tgl, sg, lane, rms_w, inv, p.w_off);       \
  }

OMLX_M5_DEF_MXFP4_CONCAT3_NORM_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r4_norm_t1, 4u)
OMLX_M5_DEF_MXFP4_CONCAT3_NORM_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r2_norm_t1, 2u)

#define OMLX_M5_DEF_MXFP4_CONCAT4_NORM_SPEC(NAME, ROWS_SG)                      \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *d0 [[buffer(3)]],                                     \
      device const uchar *s0 [[buffer(4)]],                                     \
      device const uchar *d1 [[buffer(5)]],                                     \
      device const uchar *s1 [[buffer(6)]],                                     \
      device const uchar *d2 [[buffer(7)]],                                     \
      device const uchar *s2 [[buffer(8)]],                                     \
      device const uchar *d3 [[buffer(9)]],                                     \
      device const uchar *s3 [[buffer(10)]],                                    \
      device bfloat *y0 [[buffer(11)]],                                         \
      device bfloat *y1 [[buffer(12)]],                                         \
      device bfloat *y2 [[buffer(13)]],                                         \
      device bfloat *y3 [[buffer(14)]],                                         \
      constant PretiledQmvFusedNormParams &p [[buffer(15)]],                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    constexpr uint RPTG = (ROWS_SG) * 2u;                                       \
    const uint tgs0 = (p.n0 + RPTG - 1u) / RPTG;                                \
    const uint tgs1 = (p.n1 + RPTG - 1u) / RPTG;                                \
    const uint tgs2 = (p.n2 + RPTG - 1u) / RPTG;                                \
    device const uchar *ds = d3;                                                \
    device const uchar *ss = s3;                                                \
    device bfloat *ys = y3;                                                     \
    uint ns = p.n3;                                                             \
    uint tgl = tg.x - tgs0 - tgs1 - tgs2;                                       \
    if (tg.x < tgs0) { ds = d0; ss = s0; ys = y0; ns = p.n0; tgl = tg.x; }      \
    else if (tg.x < tgs0 + tgs1) { ds = d1; ss = s1; ys = y1; ns = p.n1; tgl = tg.x - tgs0; } \
    else if (tg.x < tgs0 + tgs1 + tgs2) { ds = d2; ss = s2; ys = y2; ns = p.n2; tgl = tg.x - tgs0 - tgs1; } \
    const float inv = rsqrt(rms_sq_buf[p.sq_slot] * p.inv_k + p.eps);           \
    pretiled_qmv_mxfp4_kl32_impl<ROWS_SG, 16u, true, false>(                    \
        hidden, ds, ss, ys, ns, p.k, tgl, sg, lane, rms_w, inv, p.w_off);       \
  }

OMLX_M5_DEF_MXFP4_CONCAT4_NORM_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat4_r4_norm_t1, 4u)
OMLX_M5_DEF_MXFP4_CONCAT4_NORM_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat4_r2_norm_t1, 2u)

// --- Affine Consumers (Prologue: x_norm = hidden * rsqrt(rms_sq_buf[sq_slot] * inv_k + eps) * (rms_w + w_off)) ---
template <uint ROWS_SG>
inline void pretiled_qmv_affine_kl32_bn16_norm_dispatch(
    device const bfloat *x, device const uchar *c, device const uchar *h,
    device const uint *sb, device bfloat *y, uint n, uint k, uint bits,
    uint tg_l, uint sg, uint lane, device const bfloat *rms_w, float inv, float w_off) {
  if (bits == 4u) pretiled_qmv_affine_kl32_impl<ROWS_SG, 4u, 16u, true, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv, w_off);
  else if (bits == 5u) pretiled_qmv_affine_kl32_impl<ROWS_SG, 5u, 16u, true, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv, w_off);
  else if (bits == 6u) pretiled_qmv_affine_kl32_impl<ROWS_SG, 6u, 16u, true, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv, w_off);
  else if (bits == 8u) pretiled_qmv_affine_kl32_impl<ROWS_SG, 8u, 16u, true, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv, w_off);
  else pretiled_qmv_affine_kl32_impl<ROWS_SG, 3u, 16u, true, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv, w_off);
}

template <uint ROWS_SG>
inline void pretiled_qmv_affine_kl32_bn16_q456_norm_dispatch(
    device const bfloat *x, device const uchar *c, device const uchar *h,
    device const uint *sb, device bfloat *y, uint n, uint k, uint bits,
    uint tg_l, uint sg, uint lane, device const bfloat *rms_w, float inv, float w_off) {
  if (bits == 5u) pretiled_qmv_affine_kl32_impl<ROWS_SG, 5u, 16u, true, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv, w_off);
  else if (bits == 4u) pretiled_qmv_affine_kl32_impl<ROWS_SG, 4u, 16u, true, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv, w_off);
  else pretiled_qmv_affine_kl32_impl<ROWS_SG, 6u, 16u, true, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv, w_off);
}

#define OMLX_M5_DEF_AFFINE_NORM_T1_SPEC(NAME, ROWS_SG, BITS, BN)                \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *c [[buffer(3)]],                                      \
      device const uchar *h [[buffer(4)]],                                      \
      device const uint *sb [[buffer(5)]],                                      \
      device bfloat *y [[buffer(6)]],                                           \
      constant PretiledQmvFusedNormParams &p [[buffer(7)]],                     \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    const float inv = rsqrt(rms_sq_buf[p.sq_slot] * p.inv_k + p.eps);           \
    pretiled_qmv_affine_kl32_impl<ROWS_SG, BITS, BN, true, false>(              \
        hidden, c, h, sb, y, p.n0, p.k, tg.x, sg, lane, rms_w, inv, p.w_off);   \
  }

OMLX_M5_DEF_AFFINE_NORM_T1_SPEC(omlx_m5_pretiled_qmv_affine_q4_bn32_r4_norm_t1, 4u, 4u, 32u)
OMLX_M5_DEF_AFFINE_NORM_T1_SPEC(omlx_m5_pretiled_qmv_affine_q5_bn32_r4_norm_t1, 4u, 5u, 32u)
OMLX_M5_DEF_AFFINE_NORM_T1_SPEC(omlx_m5_pretiled_qmv_affine_q6_bn32_r4_norm_t1, 4u, 6u, 32u)
OMLX_M5_DEF_AFFINE_NORM_T1_SPEC(omlx_m5_pretiled_qmv_affine_q8_bn32_r4_norm_t1, 4u, 8u, 32u)
OMLX_M5_DEF_AFFINE_NORM_T1_SPEC(omlx_m5_pretiled_qmv_affine_q4_bn16_r4_norm_t1, 4u, 4u, 16u)
OMLX_M5_DEF_AFFINE_NORM_T1_SPEC(omlx_m5_pretiled_qmv_affine_q5_bn16_r4_norm_t1, 4u, 5u, 16u)
OMLX_M5_DEF_AFFINE_NORM_T1_SPEC(omlx_m5_pretiled_qmv_affine_q6_bn16_r4_norm_t1, 4u, 6u, 16u)
OMLX_M5_DEF_AFFINE_NORM_T1_SPEC(omlx_m5_pretiled_qmv_affine_q8_bn16_r4_norm_t1, 4u, 8u, 16u)

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_affine_norm_t1(
    device const bfloat *hidden [[buffer(0)]],
    device const bfloat *rms_w [[buffer(1)]],
    device const float *rms_sq_buf [[buffer(2)]],
    device const uchar *c [[buffer(3)]],
    device const uchar *h [[buffer(4)]],
    device const uint *sb [[buffer(5)]],
    device bfloat *y [[buffer(6)]],
    constant PretiledQmvFusedNormParams &p [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint bits = p.bits & 0xFFu;
  const float inv = rsqrt(rms_sq_buf[p.sq_slot] * p.inv_k + p.eps);
  if (p.n0 >= 65536u) {
    if (bits == 6u) pretiled_qmv_affine_kl32_impl<4u, 6u, 32u, true, false>(hidden, c, h, sb, y, p.n0, p.k, tg.x, sg, lane, rms_w, inv, p.w_off);
    else if (bits == 4u) pretiled_qmv_affine_kl32_impl<4u, 4u, 32u, true, false>(hidden, c, h, sb, y, p.n0, p.k, tg.x, sg, lane, rms_w, inv, p.w_off);
    else if (bits == 5u) pretiled_qmv_affine_kl32_impl<4u, 5u, 32u, true, false>(hidden, c, h, sb, y, p.n0, p.k, tg.x, sg, lane, rms_w, inv, p.w_off);
    else if (bits == 8u) pretiled_qmv_affine_kl32_impl<4u, 8u, 32u, true, false>(hidden, c, h, sb, y, p.n0, p.k, tg.x, sg, lane, rms_w, inv, p.w_off);
    else pretiled_qmv_affine_kl32_impl<4u, 3u, 32u, true, false>(hidden, c, h, sb, y, p.n0, p.k, tg.x, sg, lane, rms_w, inv, p.w_off);
  } else if (p.n0 >= 2048u) {
    pretiled_qmv_affine_kl32_bn16_norm_dispatch<4u>(hidden, c, h, sb, y, p.n0, p.k, bits, tg.x, sg, lane, rms_w, inv, p.w_off);
  } else {
    pretiled_qmv_affine_kl32_bn16_norm_dispatch<2u>(hidden, c, h, sb, y, p.n0, p.k, bits, tg.x, sg, lane, rms_w, inv, p.w_off);
  }
}

#define OMLX_M5_DEF_AFFINE_SWIGLU_NORM_T1_SPEC(NAME, ROWS_SG, BITS)             \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *gc [[buffer(3)]],                                     \
      device const uchar *gh [[buffer(4)]],                                     \
      device const uint *gsb [[buffer(5)]],                                     \
      device const uchar *uc [[buffer(6)]],                                     \
      device const uchar *uh [[buffer(7)]],                                     \
      device const uint *usb [[buffer(8)]],                                     \
      device bfloat *y [[buffer(9)]],                                           \
      constant PretiledQmvFusedNormParams &p [[buffer(10)]],                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    const float inv = rsqrt(rms_sq_buf[p.sq_slot] * p.inv_k + p.eps);           \
    pretiled_qmv_affine_swiglu_kl32_impl<ROWS_SG, BITS, true>(                  \
        hidden, gc, gh, gsb, uc, uh, usb, y, p.n0, p.k, tg.x, sg, lane,         \
        rms_w, inv, p.w_off);                                                   \
  }

OMLX_M5_DEF_AFFINE_SWIGLU_NORM_T1_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q4_r4_norm_t1, 4u, 4u)
OMLX_M5_DEF_AFFINE_SWIGLU_NORM_T1_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q5_r4_norm_t1, 4u, 5u)
OMLX_M5_DEF_AFFINE_SWIGLU_NORM_T1_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q6_r4_norm_t1, 4u, 6u)
OMLX_M5_DEF_AFFINE_SWIGLU_NORM_T1_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q8_r4_norm_t1, 4u, 8u)

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_affine_swiglu_norm_t1(
    device const bfloat *hidden [[buffer(0)]],
    device const bfloat *rms_w [[buffer(1)]],
    device const float *rms_sq_buf [[buffer(2)]],
    device const uchar *gc [[buffer(3)]],
    device const uchar *gh [[buffer(4)]],
    device const uint *gsb [[buffer(5)]],
    device const uchar *uc [[buffer(6)]],
    device const uchar *uh [[buffer(7)]],
    device const uint *usb [[buffer(8)]],
    device bfloat *y [[buffer(9)]],
    constant PretiledQmvFusedNormParams &p [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g_bits = p.bits & 0xFFu;
  const uint u_hi = (p.bits >> 8u) & 0xFFu;
  const uint u_bits = (u_hi != 0u) ? u_hi : g_bits;
  const float inv = rsqrt(rms_sq_buf[p.sq_slot] * p.inv_k + p.eps);
  if (p.n0 >= 2048u) {
    pretiled_qmv_affine_swiglu_kl32_mixed_fallback<4u, true>(
        hidden, gc, gh, gsb, uc, uh, usb, y, p.n0, p.k, g_bits, u_bits, tg.x, sg, lane, rms_w, inv, p.w_off);
  } else {
    pretiled_qmv_affine_swiglu_kl32_mixed_fallback<2u, true>(
        hidden, gc, gh, gsb, uc, uh, usb, y, p.n0, p.k, g_bits, u_bits, tg.x, sg, lane, rms_w, inv, p.w_off);
  }
}

#define OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_NORM_SPEC(NAME, ROWS_SG, BITS)          \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *c0 [[buffer(3)]],                                     \
      device const uchar *h0 [[buffer(4)]],                                     \
      device const uint *sb0 [[buffer(5)]],                                     \
      device const uchar *c1 [[buffer(6)]],                                     \
      device const uchar *h1 [[buffer(7)]],                                     \
      device const uint *sb1 [[buffer(8)]],                                     \
      device const uchar *c2 [[buffer(9)]],                                     \
      device const uchar *h2 [[buffer(10)]],                                    \
      device const uint *sb2 [[buffer(11)]],                                    \
      device bfloat *y0 [[buffer(12)]],                                         \
      device bfloat *y1 [[buffer(13)]],                                         \
      device bfloat *y2 [[buffer(14)]],                                         \
      constant PretiledQmvFusedNormParams &p [[buffer(15)]],                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    constexpr uint RPTG = (ROWS_SG) * 2u;                                       \
    const uint tgs0 = (p.n0 + RPTG - 1u) / RPTG;                                \
    const uint tgs1 = (p.n1 + RPTG - 1u) / RPTG;                                \
    device const uchar *cs = (tg.x < tgs0) ? c0 : ((tg.x < tgs0 + tgs1) ? c1 : c2); \
    device const uchar *hs = (tg.x < tgs0) ? h0 : ((tg.x < tgs0 + tgs1) ? h1 : h2); \
    device const uint *sbs = (tg.x < tgs0) ? sb0 : ((tg.x < tgs0 + tgs1) ? sb1 : sb2); \
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2); \
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2); \
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1)); \
    const float inv = rsqrt(rms_sq_buf[p.sq_slot] * p.inv_k + p.eps);           \
    pretiled_qmv_affine_kl32_impl<ROWS_SG, BITS, 16u, true, false>(             \
        hidden, cs, hs, sbs, ys, ns, p.k, tgl, sg, lane, rms_w, inv, p.w_off);  \
  }

OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_NORM_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q4_r4_norm_t1, 4u, 4u)
OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_NORM_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q5_r4_norm_t1, 4u, 5u)
OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_NORM_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q6_r4_norm_t1, 4u, 6u)

#define OMLX_M5_DEF_AFFINE_CONCAT3_NORM_SPEC(NAME, ROWS_SG, DISPATCH_FN)        \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *c0 [[buffer(3)]],                                     \
      device const uchar *h0 [[buffer(4)]],                                     \
      device const uint *sb0 [[buffer(5)]],                                     \
      device const uchar *c1 [[buffer(6)]],                                     \
      device const uchar *h1 [[buffer(7)]],                                     \
      device const uint *sb1 [[buffer(8)]],                                     \
      device const uchar *c2 [[buffer(9)]],                                     \
      device const uchar *h2 [[buffer(10)]],                                    \
      device const uint *sb2 [[buffer(11)]],                                    \
      device bfloat *y0 [[buffer(12)]],                                         \
      device bfloat *y1 [[buffer(13)]],                                         \
      device bfloat *y2 [[buffer(14)]],                                         \
      constant PretiledQmvFusedNormParams &p [[buffer(15)]],                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    constexpr uint RPTG = (ROWS_SG) * 2u;                                       \
    const uint b0 = p.bits & 0xFFu;                                             \
    const uint b1_r = (p.bits >> 8u) & 0xFFu;                                   \
    const uint b2_r = (p.bits >> 16u) & 0xFFu;                                  \
    const uint b1 = (b1_r != 0u) ? b1_r : b0;                                   \
    const uint b2 = (b2_r != 0u) ? b2_r : b0;                                   \
    const uint tgs0 = (p.n0 + RPTG - 1u) / RPTG;                                \
    const uint tgs1 = (p.n1 + RPTG - 1u) / RPTG;                                \
    device const uchar *cs = (tg.x < tgs0) ? c0 : ((tg.x < tgs0 + tgs1) ? c1 : c2); \
    device const uchar *hs = (tg.x < tgs0) ? h0 : ((tg.x < tgs0 + tgs1) ? h1 : h2); \
    device const uint *sbs = (tg.x < tgs0) ? sb0 : ((tg.x < tgs0 + tgs1) ? sb1 : sb2); \
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2); \
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2); \
    const uint bs = (tg.x < tgs0) ? b0 : ((tg.x < tgs0 + tgs1) ? b1 : b2);     \
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1)); \
    const float inv = rsqrt(rms_sq_buf[p.sq_slot] * p.inv_k + p.eps);           \
    DISPATCH_FN<ROWS_SG>(hidden, cs, hs, sbs, ys, ns, p.k, bs, tgl, sg, lane, rms_w, inv, p.w_off); \
  }

OMLX_M5_DEF_AFFINE_CONCAT3_NORM_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q456_r4_norm_t1, 4u, pretiled_qmv_affine_kl32_bn16_q456_norm_dispatch)
OMLX_M5_DEF_AFFINE_CONCAT3_NORM_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q456_r2_norm_t1, 2u, pretiled_qmv_affine_kl32_bn16_q456_norm_dispatch)
OMLX_M5_DEF_AFFINE_CONCAT3_NORM_SPEC(omlx_m5_pretiled_qmv_affine_concat3_r4_norm_t1, 4u, pretiled_qmv_affine_kl32_bn16_norm_dispatch)
OMLX_M5_DEF_AFFINE_CONCAT3_NORM_SPEC(omlx_m5_pretiled_qmv_affine_concat3_r2_norm_t1, 2u, pretiled_qmv_affine_kl32_bn16_norm_dispatch)

#define OMLX_M5_DEF_AFFINE_CONCAT4_NORM_SPEC(NAME, ROWS_SG, DISPATCH_FN)        \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *c0 [[buffer(3)]],                                     \
      device const uchar *h0 [[buffer(4)]],                                     \
      device const uint *sb0 [[buffer(5)]],                                     \
      device const uchar *c1 [[buffer(6)]],                                     \
      device const uchar *h1 [[buffer(7)]],                                     \
      device const uint *sb1 [[buffer(8)]],                                     \
      device const uchar *c2 [[buffer(9)]],                                     \
      device const uchar *h2 [[buffer(10)]],                                    \
      device const uint *sb2 [[buffer(11)]],                                    \
      device const uchar *c3 [[buffer(12)]],                                    \
      device const uchar *h3 [[buffer(13)]],                                    \
      device const uint *sb3 [[buffer(14)]],                                    \
      device bfloat *y0 [[buffer(15)]],                                         \
      device bfloat *y1 [[buffer(16)]],                                         \
      device bfloat *y2 [[buffer(17)]],                                         \
      device bfloat *y3 [[buffer(18)]],                                         \
      constant PretiledQmvFusedNormParams &p [[buffer(19)]],                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    constexpr uint RPTG = (ROWS_SG) * 2u;                                       \
    const uint b0 = p.bits & 0xFFu;                                             \
    const uint b1_r = (p.bits >> 8u) & 0xFFu;                                   \
    const uint b2_r = (p.bits >> 16u) & 0xFFu;                                  \
    const uint b3_r = (p.bits >> 24u) & 0xFFu;                                  \
    const uint b1 = (b1_r != 0u) ? b1_r : b0;                                   \
    const uint b2 = (b2_r != 0u) ? b2_r : b0;                                   \
    const uint b3 = (b3_r != 0u) ? b3_r : b0;                                   \
    const uint tgs0 = (p.n0 + RPTG - 1u) / RPTG;                                \
    const uint tgs1 = (p.n1 + RPTG - 1u) / RPTG;                                \
    const uint tgs2 = (p.n2 + RPTG - 1u) / RPTG;                                \
    device const uchar *cs = c3;                                                \
    device const uchar *hs = h3;                                                \
    device const uint *sbs = sb3;                                               \
    device bfloat *ys = y3;                                                     \
    uint ns = p.n3;                                                             \
    uint bs = b3;                                                               \
    uint tgl = tg.x - tgs0 - tgs1 - tgs2;                                       \
    if (tg.x < tgs0) { cs = c0; hs = h0; sbs = sb0; ys = y0; ns = p.n0; bs = b0; tgl = tg.x; } \
    else if (tg.x < tgs0 + tgs1) { cs = c1; hs = h1; sbs = sb1; ys = y1; ns = p.n1; bs = b1; tgl = tg.x - tgs0; } \
    else if (tg.x < tgs0 + tgs1 + tgs2) { cs = c2; hs = h2; sbs = sb2; ys = y2; ns = p.n2; bs = b2; tgl = tg.x - tgs0 - tgs1; } \
    const float inv = rsqrt(rms_sq_buf[p.sq_slot] * p.inv_k + p.eps);           \
    DISPATCH_FN<ROWS_SG>(hidden, cs, hs, sbs, ys, ns, p.k, bs, tgl, sg, lane, rms_w, inv, p.w_off); \
  }

OMLX_M5_DEF_AFFINE_CONCAT4_NORM_SPEC(omlx_m5_pretiled_qmv_affine_concat4_r4_norm_t1, 4u, pretiled_qmv_affine_kl32_bn16_norm_dispatch)
OMLX_M5_DEF_AFFINE_CONCAT4_NORM_SPEC(omlx_m5_pretiled_qmv_affine_concat4_r2_norm_t1, 2u, pretiled_qmv_affine_kl32_bn16_norm_dispatch)
OMLX_M5_DEF_AFFINE_CONCAT4_NORM_SPEC(omlx_m5_pretiled_qmv_affine_concat4_q456_r4_norm_t1, 4u, pretiled_qmv_affine_kl32_bn16_q456_norm_dispatch)
OMLX_M5_DEF_AFFINE_CONCAT4_NORM_SPEC(omlx_m5_pretiled_qmv_affine_concat4_q456_r2_norm_t1, 2u, pretiled_qmv_affine_kl32_bn16_q456_norm_dispatch)

// ============================================================================
// v0.8.0-m5rt.14 M=2 ONLY Scalar KL32-BM2 (TDom=1 Batched B=2 Pure AR!)
// Proven on M5 Ultra cold-DRAM micro-probe (#198):
//   - MXFP4 r4: 863.7 GB/s (98.2% DRAM peak, 1.47x vs BM=8 SK=8 NAX!)
//   - Affine Q4/Q5/Q6/Q8 r2 + Q5High1Cache + sb_tiled: 772.3 GB/s (~1.25x!)
// Strictly M=2 ONLY (zero M=3/4/8 bloat; M=4 scalar lost on silicon!).
// ============================================================================

struct RmsNormInitZeroM2Params {
  uint input_features;
  uint num_sq_words;
  float epsilon;
  float weight_offset;
  uint num_moe_words;
};

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_rmsnorm_init_zero_m2(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device bfloat *output_a16 [[buffer(2)]],
    device uint *rms_sq_buf [[buffer(3)]],
    device uint *moe_tg_done_buf [[buffer(4)]],
    constant RmsNormInitZeroM2Params &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  if (row == 0u) {
    for (uint i = tid; i < p.num_sq_words; i += 128u) {
      rms_sq_buf[i] = 0u;
    }
    for (uint i = tid; i < p.num_moe_words; i += 128u) {
      moe_tg_done_buf[i] = 0u;
    }
  }
  if (row >= 2u) return;
  const ulong row_base = ulong(row) * p.input_features;
  device const bfloat *in_row = input + row_base;
  device bfloat *out_row = output_a16 + row_base;

  float sq = 0.0f;
  if ((p.input_features & 3u) == 0u) {
    device const bfloat4 *iv = reinterpret_cast<device const bfloat4 *>(in_row);
    const uint vecs = p.input_features >> 2u;
    for (uint v = tid; v < vecs; v += 128u) {
      const float4 fv = float4(iv[v]);
      sq = fma(fv.x, fv.x, sq);
      sq = fma(fv.y, fv.y, sq);
      sq = fma(fv.z, fv.z, sq);
      sq = fma(fv.w, fv.w, sq);
    }
  } else {
    for (uint c = tid; c < p.input_features; c += 128u) {
      const float fv = float(in_row[c]);
      sq = fma(fv, fv, sq);
    }
  }
  sq = simd_sum(sq);
  threadgroup float sm[4];
  if (lane == 0u) sm[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total = ((sm[0] + sm[1]) + sm[2]) + sm[3];
  const float inv = rsqrt(total / float(p.input_features) + p.epsilon);
  if ((p.input_features & 3u) == 0u) {
    device const bfloat4 *iv = reinterpret_cast<device const bfloat4 *>(in_row);
    device const bfloat4 *wv = reinterpret_cast<device const bfloat4 *>(weight);
    device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(out_row);
    const float4 off4 = float4(p.weight_offset);
    const uint vecs = p.input_features >> 2u;
    for (uint v = tid; v < vecs; v += 128u) {
      ov[v] = bfloat4(float4(iv[v]) * inv * (float4(wv[v]) + off4));
    }
  } else {
    for (uint c = tid; c < p.input_features; c += 128u) {
      out_row[c] = bfloat(float(in_row[c]) * inv * (float(weight[c]) + p.weight_offset));
    }
  }
}

inline void pretiled_mxfp4_m2_row_fma(
    device const uchar *d_tile,
    device const uchar *s_tile,
    uint rn,
    uint k_l,
    uint sub32,
    half4 h0_0,
    half4 h0_1,
    half4 h1_0,
    half4 h1_1,
    thread float &acc0,
    thread float &acc1) {
  const uint w = reinterpret_cast<device const uint *>(
      d_tile + ulong(rn) * 64ul)[k_l];
  const float sc = decode_e8m0(s_tile[ulong(rn) * 4ul + sub32]);
  const half4 wlo(
      pretiled_fp4_to_h16(w),
      pretiled_fp4_to_h16(w >> 4u),
      pretiled_fp4_to_h16(w >> 8u),
      pretiled_fp4_to_h16(w >> 12u));
  const half4 whi(
      pretiled_fp4_to_h16(w >> 16u),
      pretiled_fp4_to_h16(w >> 20u),
      pretiled_fp4_to_h16(w >> 24u),
      pretiled_fp4_to_h16(w >> 28u));
  const half d0_0 = h0_0.x * wlo.x + h0_0.y * wlo.y + h0_0.z * wlo.z + h0_0.w * wlo.w;
  const half d0_1 = h0_1.x * whi.x + h0_1.y * whi.y + h0_1.z * whi.z + h0_1.w * whi.w;
  const half d1_0 = h1_0.x * wlo.x + h1_0.y * wlo.y + h1_0.z * wlo.z + h1_0.w * wlo.w;
  const half d1_1 = h1_1.x * whi.x + h1_1.y * whi.y + h1_1.z * whi.z + h1_1.w * whi.w;
  acc0 = fma(float(d0_0) + float(d0_1), sc, acc0);
  acc1 = fma(float(d1_0) + float(d1_1), sc, acc1);
}

inline void pretiled_m2_epilogue_row(
    float a0,
    float a1,
    device bfloat *y,
    uint row_idx,
    uint n,
    uint lane) {
  const float v0 = simd_sum(a0);
  const float v1 = simd_sum(a1);
  if (lane == 0u && row_idx < n) {
    y[ulong(row_idx)] = bfloat(v0);
    y[ulong(n) + ulong(row_idx)] = bfloat(v1);
  }
}

inline void pretiled_m2_epilogue_add_sq_row(
    float a0,
    float a1,
    device bfloat *y,
    uint row_idx,
    uint n,
    uint lane,
    thread float &sg_sq0,
    thread float &sg_sq1) {
  const float v0 = simd_sum(a0);
  const float v1 = simd_sum(a1);
  if (lane == 0u && row_idx < n) {
    const ulong c0 = ulong(row_idx);
    const ulong c1 = ulong(n) + c0;
    const bfloat hn0 = bfloat(float(y[c0]) + float(bfloat(v0)));
    const bfloat hn1 = bfloat(float(y[c1]) + float(bfloat(v1)));
    y[c0] = hn0;
    y[c1] = hn1;
    const float hf0 = float(hn0);
    const float hf1 = float(hn1);
    sg_sq0 = fma(hf0, hf0, sg_sq0);
    sg_sq1 = fma(hf1, hf1, sg_sq1);
  }
}

inline void pretiled_m2_epilogue_swiglu_row(
    float g0_acc,
    float u0_acc,
    float g1_acc,
    float u1_acc,
    device bfloat *y,
    uint row_idx,
    uint n,
    uint lane) {
  const float g0 = simd_sum(g0_acc);
  const float u0 = simd_sum(u0_acc);
  const float g1 = simd_sum(g1_acc);
  const float u1 = simd_sum(u1_acc);
  if (lane == 0u && row_idx < n) {
    const float gv0 = float(bfloat(g0));
    const float uv0 = float(bfloat(u0));
    const float gv1 = float(bfloat(g1));
    const float uv1 = float(bfloat(u1));
    y[ulong(row_idx)] = bfloat((gv0 / (1.0f + exp(-gv0))) * uv0);
    y[ulong(n) + ulong(row_idx)] = bfloat((gv1 / (1.0f + exp(-gv1))) * uv1);
  }
}

template <uint ROWS_SG, uint BN, bool PROLOGUE_NORM = false, bool EPILOGUE_ADD_SQ = false>
inline void pretiled_qmv_mxfp4_kl32_m2_impl(
    device const bfloat *x,
    device const uchar *nax_data,
    device const uchar *nax_scale,
    device bfloat *y,
    uint n,
    uint k,
    uint tg_local,
    uint sg,
    uint lane,
    device const bfloat *rms_w = nullptr,
    float inv_rms0 = 1.0f,
    float inv_rms1 = 1.0f,
    float w_off = 0.0f,
    device atomic_float *rms_sq_buf = nullptr,
    uint sq_slot = 0u) {
  static_assert(BN == 16u || BN == 32u, "BN must be 16 or 32");
  static_assert(ROWS_SG == 2u || ROWS_SG == 4u, "ROWS_SG must be 2 or 4");
  const uint row_sg = (tg_local * 2u + sg) * ROWS_SG;
  if (row_sg >= n) {
    return;
  }
  const bool do_fwht = (k & 0x80000000u) != 0u;
  const uint k_eff = k & 0x3FFFFFFFu;
  const uint k_blocks = k_eff / 128u;
  const uint tile_n = row_sg / BN;
  const uint rn0 = row_sg % BN;
  constexpr ulong tile_code_bytes = ulong(BN) * 64ul;
  constexpr ulong tile_scale_bytes = ulong(BN) * 4ul;

  const uint b_sub = lane >> 4u;
  const uint k_l = lane & 15u;
  const uint sub32 = k_l >> 2u;

  float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
  float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;

  const float4 off4 = float4(w_off);
  const ulong tile_base = ulong(tile_n) * k_blocks;
  device const bfloat *x1 = x + ulong(k_eff);
  for (uint blk = 0u; blk < k_blocks; blk += 2u) {
    const uint cur_blk = blk + b_sub;
    float4 f0_0 = float4(0.0f), f0_1 = float4(0.0f);
    float4 f1_0 = float4(0.0f), f1_1 = float4(0.0f);
    if (cur_blk < k_blocks) {
      const ulong x_off = ulong(cur_blk) * 128ul + ulong(k_l) * 8ul;
      device const bfloat4 *xb0 = reinterpret_cast<device const bfloat4 *>(x + x_off);
      device const bfloat4 *xb1 = reinterpret_cast<device const bfloat4 *>(x1 + x_off);
      if constexpr (PROLOGUE_NORM) {
        device const bfloat4 *wb = reinterpret_cast<device const bfloat4 *>(rms_w + x_off);
        const float4 w0 = float4(wb[0]) + off4;
        const float4 w1 = float4(wb[1]) + off4;
        f0_0 = float4(bfloat4(float4(xb0[0]) * inv_rms0 * w0));
        f0_1 = float4(bfloat4(float4(xb0[1]) * inv_rms0 * w1));
        f1_0 = float4(bfloat4(float4(xb1[0]) * inv_rms1 * w0));
        f1_1 = float4(bfloat4(float4(xb1[1]) * inv_rms1 * w1));
      } else {
        (void)rms_w; (void)inv_rms0; (void)inv_rms1; (void)off4;
        f0_0 = float4(xb0[0]);
        f0_1 = float4(xb0[1]);
        f1_0 = float4(xb1[0]);
        f1_1 = float4(xb1[1]);
      }
    }
    if (do_fwht) {
      omlx_m5_fwht64_sub8(f0_0, f0_1, lane);
      omlx_m5_fwht64_sub8(f1_0, f1_1, lane);
    }
    if (cur_blk < k_blocks) {
      const half4 h0_0 = half4(f0_0);
      const half4 h0_1 = half4(f0_1);
      const half4 h1_0 = half4(f1_0);
      const half4 h1_1 = half4(f1_1);
      const ulong idx = tile_base + cur_blk;
      device const uchar *d_tile = nax_data + idx * tile_code_bytes;
      device const uchar *s_tile = nax_scale + idx * tile_scale_bytes;
      pretiled_mxfp4_m2_row_fma(d_tile, s_tile, rn0 + 0u, k_l, sub32, h0_0, h0_1, h1_0, h1_1, a0_0, a1_0);
      pretiled_mxfp4_m2_row_fma(d_tile, s_tile, rn0 + 1u, k_l, sub32, h0_0, h0_1, h1_0, h1_1, a0_1, a1_1);
      if constexpr (ROWS_SG == 4u) {
        pretiled_mxfp4_m2_row_fma(d_tile, s_tile, rn0 + 2u, k_l, sub32, h0_0, h0_1, h1_0, h1_1, a0_2, a1_2);
        pretiled_mxfp4_m2_row_fma(d_tile, s_tile, rn0 + 3u, k_l, sub32, h0_0, h0_1, h1_0, h1_1, a0_3, a1_3);
      }
    }
  }

  if constexpr (EPILOGUE_ADD_SQ) {
    float sg_sq0 = 0.0f;
    float sg_sq1 = 0.0f;
    pretiled_m2_epilogue_add_sq_row(a0_0, a1_0, y, row_sg + 0u, n, lane, sg_sq0, sg_sq1);
    pretiled_m2_epilogue_add_sq_row(a0_1, a1_1, y, row_sg + 1u, n, lane, sg_sq0, sg_sq1);
    if constexpr (ROWS_SG == 4u) {
      pretiled_m2_epilogue_add_sq_row(a0_2, a1_2, y, row_sg + 2u, n, lane, sg_sq0, sg_sq1);
      pretiled_m2_epilogue_add_sq_row(a0_3, a1_3, y, row_sg + 3u, n, lane, sg_sq0, sg_sq1);
    }
    if (lane == 0u) {
      atomic_fetch_add_explicit(&rms_sq_buf[sq_slot * 2u + 0u], sg_sq0, memory_order_relaxed);
      atomic_fetch_add_explicit(&rms_sq_buf[sq_slot * 2u + 1u], sg_sq1, memory_order_relaxed);
    }
  } else {
    (void)rms_sq_buf; (void)sq_slot;
    pretiled_m2_epilogue_row(a0_0, a1_0, y, row_sg + 0u, n, lane);
    pretiled_m2_epilogue_row(a0_1, a1_1, y, row_sg + 1u, n, lane);
    if constexpr (ROWS_SG == 4u) {
      pretiled_m2_epilogue_row(a0_2, a1_2, y, row_sg + 2u, n, lane);
      pretiled_m2_epilogue_row(a0_3, a1_3, y, row_sg + 3u, n, lane);
    }
  }
}

template <uint ROWS_SG, uint BN, bool PROLOGUE_NORM = false>
inline void pretiled_qmv_mxfp4_swiglu_kl32_m2_impl(
    device const bfloat *x,
    device const uchar *g_data,
    device const uchar *g_scale,
    device const uchar *u_data,
    device const uchar *u_scale,
    device bfloat *y,
    uint n,
    uint k,
    uint tg_local,
    uint sg,
    uint lane,
    device const bfloat *rms_w = nullptr,
    float inv_rms0 = 1.0f,
    float inv_rms1 = 1.0f,
    float w_off = 0.0f) {
  static_assert(BN == 16u || BN == 32u, "BN must be 16 or 32");
  static_assert(ROWS_SG == 2u || ROWS_SG == 4u, "ROWS_SG must be 2 or 4");
  const uint row_sg = (tg_local * 2u + sg) * ROWS_SG;
  if (row_sg >= n) {
    return;
  }
  const bool do_fwht = (k & 0x80000000u) != 0u;
  const uint k_eff = k & 0x3FFFFFFFu;
  const uint k_blocks = k_eff / 128u;
  const uint tile_n = row_sg / BN;
  const uint rn0 = row_sg % BN;
  constexpr ulong tile_code_bytes = ulong(BN) * 64ul;
  constexpr ulong tile_scale_bytes = ulong(BN) * 4ul;

  const uint b_sub = lane >> 4u;
  const uint k_l = lane & 15u;
  const uint sub32 = k_l >> 2u;

  float ag0_0 = 0.0f, ag0_1 = 0.0f, ag0_2 = 0.0f, ag0_3 = 0.0f;
  float ag1_0 = 0.0f, ag1_1 = 0.0f, ag1_2 = 0.0f, ag1_3 = 0.0f;
  float au0_0 = 0.0f, au0_1 = 0.0f, au0_2 = 0.0f, au0_3 = 0.0f;
  float au1_0 = 0.0f, au1_1 = 0.0f, au1_2 = 0.0f, au1_3 = 0.0f;

  const float4 off4 = float4(w_off);
  const ulong tile_base = ulong(tile_n) * k_blocks;
  device const bfloat *x1 = x + ulong(k_eff);
  for (uint blk = 0u; blk < k_blocks; blk += 2u) {
    const uint cur_blk = blk + b_sub;
    float4 f0_0 = float4(0.0f), f0_1 = float4(0.0f);
    float4 f1_0 = float4(0.0f), f1_1 = float4(0.0f);
    if (cur_blk < k_blocks) {
      const ulong x_off = ulong(cur_blk) * 128ul + ulong(k_l) * 8ul;
      device const bfloat4 *xb0 = reinterpret_cast<device const bfloat4 *>(x + x_off);
      device const bfloat4 *xb1 = reinterpret_cast<device const bfloat4 *>(x1 + x_off);
      if constexpr (PROLOGUE_NORM) {
        device const bfloat4 *wb = reinterpret_cast<device const bfloat4 *>(rms_w + x_off);
        const float4 w0 = float4(wb[0]) + off4;
        const float4 w1 = float4(wb[1]) + off4;
        f0_0 = float4(bfloat4(float4(xb0[0]) * inv_rms0 * w0));
        f0_1 = float4(bfloat4(float4(xb0[1]) * inv_rms0 * w1));
        f1_0 = float4(bfloat4(float4(xb1[0]) * inv_rms1 * w0));
        f1_1 = float4(bfloat4(float4(xb1[1]) * inv_rms1 * w1));
      } else {
        (void)rms_w; (void)inv_rms0; (void)inv_rms1; (void)off4;
        f0_0 = float4(xb0[0]);
        f0_1 = float4(xb0[1]);
        f1_0 = float4(xb1[0]);
        f1_1 = float4(xb1[1]);
      }
    }
    if (do_fwht) {
      omlx_m5_fwht64_sub8(f0_0, f0_1, lane);
      omlx_m5_fwht64_sub8(f1_0, f1_1, lane);
    }
    if (cur_blk < k_blocks) {
      const half4 h0_0 = half4(f0_0);
      const half4 h0_1 = half4(f0_1);
      const half4 h1_0 = half4(f1_0);
      const half4 h1_1 = half4(f1_1);
      const ulong idx = tile_base + cur_blk;
      device const uchar *gd_tile = g_data + idx * tile_code_bytes;
      device const uchar *gs_tile = g_scale + idx * tile_scale_bytes;
      device const uchar *ud_tile = u_data + idx * tile_code_bytes;
      device const uchar *us_tile = u_scale + idx * tile_scale_bytes;
      pretiled_mxfp4_m2_row_fma(gd_tile, gs_tile, rn0 + 0u, k_l, sub32, h0_0, h0_1, h1_0, h1_1, ag0_0, ag1_0);
      pretiled_mxfp4_m2_row_fma(ud_tile, us_tile, rn0 + 0u, k_l, sub32, h0_0, h0_1, h1_0, h1_1, au0_0, au1_0);
      pretiled_mxfp4_m2_row_fma(gd_tile, gs_tile, rn0 + 1u, k_l, sub32, h0_0, h0_1, h1_0, h1_1, ag0_1, ag1_1);
      pretiled_mxfp4_m2_row_fma(ud_tile, us_tile, rn0 + 1u, k_l, sub32, h0_0, h0_1, h1_0, h1_1, au0_1, au1_1);
      if constexpr (ROWS_SG == 4u) {
        pretiled_mxfp4_m2_row_fma(gd_tile, gs_tile, rn0 + 2u, k_l, sub32, h0_0, h0_1, h1_0, h1_1, ag0_2, ag1_2);
        pretiled_mxfp4_m2_row_fma(ud_tile, us_tile, rn0 + 2u, k_l, sub32, h0_0, h0_1, h1_0, h1_1, au0_2, au1_2);
        pretiled_mxfp4_m2_row_fma(gd_tile, gs_tile, rn0 + 3u, k_l, sub32, h0_0, h0_1, h1_0, h1_1, ag0_3, ag1_3);
        pretiled_mxfp4_m2_row_fma(ud_tile, us_tile, rn0 + 3u, k_l, sub32, h0_0, h0_1, h1_0, h1_1, au0_3, au1_3);
      }
    }
  }

  pretiled_m2_epilogue_swiglu_row(ag0_0, au0_0, ag1_0, au1_0, y, row_sg + 0u, n, lane);
  pretiled_m2_epilogue_swiglu_row(ag0_1, au0_1, ag1_1, au1_1, y, row_sg + 1u, n, lane);
  if constexpr (ROWS_SG == 4u) {
    pretiled_m2_epilogue_swiglu_row(ag0_2, au0_2, ag1_2, au1_2, y, row_sg + 2u, n, lane);
    pretiled_m2_epilogue_swiglu_row(ag0_3, au0_3, ag1_3, au1_3, y, row_sg + 3u, n, lane);
  }
}

#define OMLX_M5_DEF_MXFP4_M2_SPEC(NAME, ROWS_SG, BN)                            \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *nax_data [[buffer(1)]],                               \
      device const uchar *nax_scale [[buffer(2)]],                              \
      device bfloat *y [[buffer(3)]],                                           \
      constant QmvT1Params &p [[buffer(4)]],                                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    pretiled_qmv_mxfp4_kl32_m2_impl<ROWS_SG, BN>(                               \
        x, nax_data, nax_scale, y, p.output_features, p.input_features, tg.x,   \
        sg, lane);                                                              \
  }

OMLX_M5_DEF_MXFP4_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_bn16_r4_m2, 4u, 16u)
OMLX_M5_DEF_MXFP4_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_bn16_r2_m2, 2u, 16u)
OMLX_M5_DEF_MXFP4_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_bn32_r4_m2, 4u, 32u)

#define OMLX_M5_DEF_MXFP4_SWIGLU_M2_SPEC(NAME, ROWS_SG, BN)                     \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *g_data [[buffer(1)]],                                 \
      device const uchar *g_scale [[buffer(2)]],                                \
      device const uchar *u_data [[buffer(3)]],                                 \
      device const uchar *u_scale [[buffer(4)]],                                \
      device bfloat *y [[buffer(5)]],                                           \
      constant QmvT1Params &p [[buffer(6)]],                                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    pretiled_qmv_mxfp4_swiglu_kl32_m2_impl<ROWS_SG, BN>(                        \
        x, g_data, g_scale, u_data, u_scale, y, p.output_features,              \
        p.input_features, tg.x, sg, lane);                                      \
  }

OMLX_M5_DEF_MXFP4_SWIGLU_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r4_m2, 4u, 16u)
OMLX_M5_DEF_MXFP4_SWIGLU_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r2_m2, 2u, 16u)

#define OMLX_M5_DEF_MXFP4_CONCAT3_M2_SPEC(NAME, ROWS_SG)                        \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *d0 [[buffer(1)]],                                     \
      device const uchar *s0 [[buffer(2)]],                                     \
      device const uchar *d1 [[buffer(3)]],                                     \
      device const uchar *s1 [[buffer(4)]],                                     \
      device const uchar *d2 [[buffer(5)]],                                     \
      device const uchar *s2 [[buffer(6)]],                                     \
      device bfloat *y0 [[buffer(7)]],                                          \
      device bfloat *y1 [[buffer(8)]],                                          \
      device bfloat *y2 [[buffer(9)]],                                          \
      constant PretiledQmvMxfp4Params &p [[buffer(10)]],                        \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    constexpr uint RPTG = (ROWS_SG) * 2u;                                       \
    const uint tgs0 = (p.n0 + RPTG - 1u) / RPTG;                                \
    const uint tgs1 = (p.n1 + RPTG - 1u) / RPTG;                                \
    device const uchar *ds = (tg.x < tgs0) ? d0 : ((tg.x < tgs0 + tgs1) ? d1 : d2); \
    device const uchar *ss = (tg.x < tgs0) ? s0 : ((tg.x < tgs0 + tgs1) ? s1 : s2); \
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2);  \
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2); \
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1)); \
    pretiled_qmv_mxfp4_kl32_m2_impl<ROWS_SG, 16u>(x, ds, ss, ys, ns, p.k, tgl, sg, lane); \
  }

OMLX_M5_DEF_MXFP4_CONCAT3_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r4_m2, 4u)
OMLX_M5_DEF_MXFP4_CONCAT3_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r2_m2, 2u)

#define OMLX_M5_DEF_MXFP4_CONCAT4_M2_SPEC(NAME, ROWS_SG)                        \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *d0 [[buffer(1)]],                                     \
      device const uchar *s0 [[buffer(2)]],                                     \
      device const uchar *d1 [[buffer(3)]],                                     \
      device const uchar *s1 [[buffer(4)]],                                     \
      device const uchar *d2 [[buffer(5)]],                                     \
      device const uchar *s2 [[buffer(6)]],                                     \
      device const uchar *d3 [[buffer(7)]],                                     \
      device const uchar *s3 [[buffer(8)]],                                     \
      device bfloat *y0 [[buffer(9)]],                                          \
      device bfloat *y1 [[buffer(10)]],                                         \
      device bfloat *y2 [[buffer(11)]],                                         \
      device bfloat *y3 [[buffer(12)]],                                         \
      constant PretiledQmvConcat4Mxfp4Params &p [[buffer(13)]],                 \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    constexpr uint RPTG = (ROWS_SG) * 2u;                                       \
    const uint tgs0 = (p.n0 + RPTG - 1u) / RPTG;                                \
    const uint tgs1 = (p.n1 + RPTG - 1u) / RPTG;                                \
    const uint tgs2 = (p.n2 + RPTG - 1u) / RPTG;                                \
    device const uchar *ds = d3;                                                \
    device const uchar *ss = s3;                                                \
    device bfloat *ys = y3;                                                     \
    uint ns = p.n3;                                                             \
    uint tgl = tg.x - tgs0 - tgs1 - tgs2;                                       \
    if (tg.x < tgs0) { ds = d0; ss = s0; ys = y0; ns = p.n0; tgl = tg.x; }      \
    else if (tg.x < tgs0 + tgs1) { ds = d1; ss = s1; ys = y1; ns = p.n1; tgl = tg.x - tgs0; } \
    else if (tg.x < tgs0 + tgs1 + tgs2) { ds = d2; ss = s2; ys = y2; ns = p.n2; tgl = tg.x - tgs0 - tgs1; } \
    pretiled_qmv_mxfp4_kl32_m2_impl<ROWS_SG, 16u>(x, ds, ss, ys, ns, p.k, tgl, sg, lane); \
  }

OMLX_M5_DEF_MXFP4_CONCAT4_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat4_r4_m2, 4u)
OMLX_M5_DEF_MXFP4_CONCAT4_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat4_r2_m2, 2u)

#define OMLX_M5_DEF_MXFP4_ADD_SQ_M2_SPEC(NAME, ROWS_SG, BN)                     \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *nax_data [[buffer(1)]],                               \
      device const uchar *nax_scale [[buffer(2)]],                              \
      device bfloat *hidden_inout [[buffer(3)]],                                \
      device atomic_float *rms_sq_buf [[buffer(4)]],                            \
      constant PretiledQmvFusedNormParams &p [[buffer(5)]],                     \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    pretiled_qmv_mxfp4_kl32_m2_impl<ROWS_SG, BN, false, true>(                  \
        x, nax_data, nax_scale, hidden_inout, p.n0, p.k, tg.x, sg, lane,        \
        nullptr, 1.0f, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);                      \
  }

OMLX_M5_DEF_MXFP4_ADD_SQ_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_bn16_r4_add_sq_m2, 4u, 16u)
OMLX_M5_DEF_MXFP4_ADD_SQ_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_bn16_r2_add_sq_m2, 2u, 16u)

#define OMLX_M5_DEF_MXFP4_NORM_M2_SPEC(NAME, ROWS_SG, BN)                       \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *nax_data [[buffer(3)]],                               \
      device const uchar *nax_scale [[buffer(4)]],                              \
      device bfloat *y [[buffer(5)]],                                           \
      constant PretiledQmvFusedNormParams &p [[buffer(6)]],                     \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    const float inv0 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 0u] * p.inv_k + p.eps); \
    const float inv1 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 1u] * p.inv_k + p.eps); \
    pretiled_qmv_mxfp4_kl32_m2_impl<ROWS_SG, BN, true, false>(                  \
        hidden, nax_data, nax_scale, y, p.n0, p.k, tg.x, sg, lane,              \
        rms_w, inv0, inv1, p.w_off);                                            \
  }

OMLX_M5_DEF_MXFP4_NORM_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_bn32_r4_norm_m2, 4u, 32u)
OMLX_M5_DEF_MXFP4_NORM_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_bn16_r4_norm_m2, 4u, 16u)

#define OMLX_M5_DEF_MXFP4_SWIGLU_NORM_M2_SPEC(NAME, ROWS_SG, BN)                \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *gd [[buffer(3)]],                                     \
      device const uchar *gs [[buffer(4)]],                                     \
      device const uchar *ud [[buffer(5)]],                                     \
      device const uchar *us [[buffer(6)]],                                     \
      device bfloat *y [[buffer(7)]],                                           \
      constant PretiledQmvFusedNormParams &p [[buffer(8)]],                     \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    const float inv0 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 0u] * p.inv_k + p.eps); \
    const float inv1 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 1u] * p.inv_k + p.eps); \
    pretiled_qmv_mxfp4_swiglu_kl32_m2_impl<ROWS_SG, BN, true>(                  \
        hidden, gd, gs, ud, us, y, p.n0, p.k, tg.x, sg, lane,                   \
        rms_w, inv0, inv1, p.w_off);                                            \
  }

OMLX_M5_DEF_MXFP4_SWIGLU_NORM_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r4_norm_m2, 4u, 16u)
OMLX_M5_DEF_MXFP4_SWIGLU_NORM_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_swiglu_bn16_r2_norm_m2, 2u, 16u)

#define OMLX_M5_DEF_MXFP4_CONCAT3_NORM_M2_SPEC(NAME, ROWS_SG)                   \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *d0 [[buffer(3)]],                                     \
      device const uchar *s0 [[buffer(4)]],                                     \
      device const uchar *d1 [[buffer(5)]],                                     \
      device const uchar *s1 [[buffer(6)]],                                     \
      device const uchar *d2 [[buffer(7)]],                                     \
      device const uchar *s2 [[buffer(8)]],                                     \
      device bfloat *y0 [[buffer(9)]],                                          \
      device bfloat *y1 [[buffer(10)]],                                         \
      device bfloat *y2 [[buffer(11)]],                                         \
      constant PretiledQmvFusedNormParams &p [[buffer(12)]],                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    constexpr uint RPTG = (ROWS_SG) * 2u;                                       \
    const uint tgs0 = (p.n0 + RPTG - 1u) / RPTG;                                \
    const uint tgs1 = (p.n1 + RPTG - 1u) / RPTG;                                \
    device const uchar *ds = (tg.x < tgs0) ? d0 : ((tg.x < tgs0 + tgs1) ? d1 : d2); \
    device const uchar *ss = (tg.x < tgs0) ? s0 : ((tg.x < tgs0 + tgs1) ? s1 : s2); \
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2);  \
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2); \
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1)); \
    const float inv0 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 0u] * p.inv_k + p.eps); \
    const float inv1 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 1u] * p.inv_k + p.eps); \
    pretiled_qmv_mxfp4_kl32_m2_impl<ROWS_SG, 16u, true, false>(                 \
        hidden, ds, ss, ys, ns, p.k, tgl, sg, lane, rms_w, inv0, inv1, p.w_off); \
  }

OMLX_M5_DEF_MXFP4_CONCAT3_NORM_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r4_norm_m2, 4u)
OMLX_M5_DEF_MXFP4_CONCAT3_NORM_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat3_bn16_r2_norm_m2, 2u)

#define OMLX_M5_DEF_MXFP4_CONCAT4_NORM_M2_SPEC(NAME, ROWS_SG)                   \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *d0 [[buffer(3)]],                                     \
      device const uchar *s0 [[buffer(4)]],                                     \
      device const uchar *d1 [[buffer(5)]],                                     \
      device const uchar *s1 [[buffer(6)]],                                     \
      device const uchar *d2 [[buffer(7)]],                                     \
      device const uchar *s2 [[buffer(8)]],                                     \
      device const uchar *d3 [[buffer(9)]],                                     \
      device const uchar *s3 [[buffer(10)]],                                    \
      device bfloat *y0 [[buffer(11)]],                                         \
      device bfloat *y1 [[buffer(12)]],                                         \
      device bfloat *y2 [[buffer(13)]],                                         \
      device bfloat *y3 [[buffer(14)]],                                         \
      constant PretiledQmvFusedNormParams &p [[buffer(15)]],                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    constexpr uint RPTG = (ROWS_SG) * 2u;                                       \
    const uint tgs0 = (p.n0 + RPTG - 1u) / RPTG;                                \
    const uint tgs1 = (p.n1 + RPTG - 1u) / RPTG;                                \
    const uint tgs2 = (p.n2 + RPTG - 1u) / RPTG;                                \
    device const uchar *ds = d3;                                                \
    device const uchar *ss = s3;                                                \
    device bfloat *ys = y3;                                                     \
    uint ns = p.n3;                                                             \
    uint tgl = tg.x - tgs0 - tgs1 - tgs2;                                       \
    if (tg.x < tgs0) { ds = d0; ss = s0; ys = y0; ns = p.n0; tgl = tg.x; }      \
    else if (tg.x < tgs0 + tgs1) { ds = d1; ss = s1; ys = y1; ns = p.n1; tgl = tg.x - tgs0; } \
    else if (tg.x < tgs0 + tgs1 + tgs2) { ds = d2; ss = s2; ys = y2; ns = p.n2; tgl = tg.x - tgs0 - tgs1; } \
    const float inv0 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 0u] * p.inv_k + p.eps); \
    const float inv1 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 1u] * p.inv_k + p.eps); \
    pretiled_qmv_mxfp4_kl32_m2_impl<ROWS_SG, 16u, true, false>(                 \
        hidden, ds, ss, ys, ns, p.k, tgl, sg, lane, rms_w, inv0, inv1, p.w_off); \
  }

OMLX_M5_DEF_MXFP4_CONCAT4_NORM_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat4_r4_norm_m2, 4u)
OMLX_M5_DEF_MXFP4_CONCAT4_NORM_M2_SPEC(omlx_m5_pretiled_qmv_mxfp4_concat4_r2_norm_m2, 2u)

// --- Affine M=2 ONLY Scalar KL32-BM2 (r2, zero-spill, Q4/Q5/Q6/Q8 + Q5High1 + sb_tiled) ---
__attribute__((always_inline)) inline void pretiled_affine_fma4_m2(
    uint q0, uint q1, uint q2, uint q3,
    float4 x0, float4 x1,
    thread float &d0, thread float &d1) {
  const float w0 = float(as_type<half>(ushort(0x6400u | q0)));
  const float w1 = float(as_type<half>(ushort(0x6400u | q1)));
  const float w2 = float(as_type<half>(ushort(0x6400u | q2)));
  const float w3 = float(as_type<half>(ushort(0x6400u | q3)));
  d0 = fma(x0.x, w0, d0); d1 = fma(x1.x, w0, d1);
  d0 = fma(x0.y, w1, d0); d1 = fma(x1.y, w1, d1);
  d0 = fma(x0.z, w2, d0); d1 = fma(x1.z, w2, d1);
  d0 = fma(x0.w, w3, d0); d1 = fma(x1.w, w3, d1);
}

template <uint BITS>
__attribute__((always_inline)) inline float2 pretiled_affine_dot16_f32_m2(
    device const uchar *c_tile,
    device const uchar *h_tile,
    uint rn,
    uint k_l,
    float4 x0_0, float4 x0_1, float4 x0_2, float4 x0_3,
    float4 x1_0, float4 x1_1, float4 x1_2, float4 x1_3) {
  float d0 = 0.0f;
  float d1 = 0.0f;
  if constexpr (BITS == 4u) {
    (void)h_tile;
    const uint2 cw =
        reinterpret_cast<device const uint2 *>(c_tile + ulong(rn) * 32ul)[k_l];
    pretiled_affine_fma4_m2(cw.x & 0xFu, (cw.x >> 4u) & 0xFu, (cw.x >> 8u) & 0xFu, (cw.x >> 12u) & 0xFu, x0_0, x1_0, d0, d1);
    pretiled_affine_fma4_m2((cw.x >> 16u) & 0xFu, (cw.x >> 20u) & 0xFu, (cw.x >> 24u) & 0xFu, (cw.x >> 28u) & 0xFu, x0_1, x1_1, d0, d1);
    pretiled_affine_fma4_m2(cw.y & 0xFu, (cw.y >> 4u) & 0xFu, (cw.y >> 8u) & 0xFu, (cw.y >> 12u) & 0xFu, x0_2, x1_2, d0, d1);
    pretiled_affine_fma4_m2((cw.y >> 16u) & 0xFu, (cw.y >> 20u) & 0xFu, (cw.y >> 24u) & 0xFu, (cw.y >> 28u) & 0xFu, x0_3, x1_3, d0, d1);
  } else if constexpr (BITS == 6u) {
    const uint2 cw =
        reinterpret_cast<device const uint2 *>(c_tile + ulong(rn) * 32ul)[k_l];
    const uint hw =
        reinterpret_cast<device const uint *>(h_tile + ulong(rn) * 16ul)[k_l];
    pretiled_affine_fma4_m2(
        (cw.x & 0xFu) | ((hw & 3u) << 4u),
        ((cw.x >> 4u) & 0xFu) | (((hw >> 2u) & 3u) << 4u),
        ((cw.x >> 8u) & 0xFu) | (((hw >> 4u) & 3u) << 4u),
        ((cw.x >> 12u) & 0xFu) | (((hw >> 6u) & 3u) << 4u),
        x0_0, x1_0, d0, d1);
    pretiled_affine_fma4_m2(
        ((cw.x >> 16u) & 0xFu) | (((hw >> 8u) & 3u) << 4u),
        ((cw.x >> 20u) & 0xFu) | (((hw >> 10u) & 3u) << 4u),
        ((cw.x >> 24u) & 0xFu) | (((hw >> 12u) & 3u) << 4u),
        ((cw.x >> 28u) & 0xFu) | (((hw >> 14u) & 3u) << 4u),
        x0_1, x1_1, d0, d1);
    pretiled_affine_fma4_m2(
        (cw.y & 0xFu) | (((hw >> 16u) & 3u) << 4u),
        ((cw.y >> 4u) & 0xFu) | (((hw >> 18u) & 3u) << 4u),
        ((cw.y >> 8u) & 0xFu) | (((hw >> 20u) & 3u) << 4u),
        ((cw.y >> 12u) & 0xFu) | (((hw >> 22u) & 3u) << 4u),
        x0_2, x1_2, d0, d1);
    pretiled_affine_fma4_m2(
        ((cw.y >> 16u) & 0xFu) | (((hw >> 24u) & 3u) << 4u),
        ((cw.y >> 20u) & 0xFu) | (((hw >> 26u) & 3u) << 4u),
        ((cw.y >> 24u) & 0xFu) | (((hw >> 28u) & 3u) << 4u),
        ((cw.y >> 28u) & 0xFu) | (((hw >> 30u) & 3u) << 4u),
        x0_3, x1_3, d0, d1);
  } else if constexpr (BITS == 5u) {
    const uint2 cw =
        reinterpret_cast<device const uint2 *>(c_tile + ulong(rn) * 32ul)[k_l];
    const uint hw = uint(
        reinterpret_cast<device const ushort *>(h_tile + ulong(rn) * 8ul)[k_l]);
    pretiled_affine_fma4_m2(
        (cw.x & 0xFu) | ((hw & 1u) << 4u),
        ((cw.x >> 4u) & 0xFu) | (((hw >> 1u) & 1u) << 4u),
        ((cw.x >> 8u) & 0xFu) | (((hw >> 2u) & 1u) << 4u),
        ((cw.x >> 12u) & 0xFu) | (((hw >> 3u) & 1u) << 4u),
        x0_0, x1_0, d0, d1);
    pretiled_affine_fma4_m2(
        ((cw.x >> 16u) & 0xFu) | (((hw >> 4u) & 1u) << 4u),
        ((cw.x >> 20u) & 0xFu) | (((hw >> 5u) & 1u) << 4u),
        ((cw.x >> 24u) & 0xFu) | (((hw >> 6u) & 1u) << 4u),
        ((cw.x >> 28u) & 0xFu) | (((hw >> 7u) & 1u) << 4u),
        x0_1, x1_1, d0, d1);
    pretiled_affine_fma4_m2(
        (cw.y & 0xFu) | (((hw >> 8u) & 1u) << 4u),
        ((cw.y >> 4u) & 0xFu) | (((hw >> 9u) & 1u) << 4u),
        ((cw.y >> 8u) & 0xFu) | (((hw >> 10u) & 1u) << 4u),
        ((cw.y >> 12u) & 0xFu) | (((hw >> 11u) & 1u) << 4u),
        x0_2, x1_2, d0, d1);
    pretiled_affine_fma4_m2(
        ((cw.y >> 16u) & 0xFu) | (((hw >> 12u) & 1u) << 4u),
        ((cw.y >> 20u) & 0xFu) | (((hw >> 13u) & 1u) << 4u),
        ((cw.y >> 24u) & 0xFu) | (((hw >> 14u) & 1u) << 4u),
        ((cw.y >> 28u) & 0xFu) | (((hw >> 15u) & 1u) << 4u),
        x0_3, x1_3, d0, d1);
  } else if constexpr (BITS == 8u) {
    (void)h_tile;
    const uint4 cb =
        reinterpret_cast<device const uint4 *>(c_tile + ulong(rn) * 64ul)[k_l];
    pretiled_affine_fma4_m2(cb.x & 0xFFu, (cb.x >> 8u) & 0xFFu, (cb.x >> 16u) & 0xFFu, cb.x >> 24u, x0_0, x1_0, d0, d1);
    pretiled_affine_fma4_m2(cb.y & 0xFFu, (cb.y >> 8u) & 0xFFu, (cb.y >> 16u) & 0xFFu, cb.y >> 24u, x0_1, x1_1, d0, d1);
    pretiled_affine_fma4_m2(cb.z & 0xFFu, (cb.z >> 8u) & 0xFFu, (cb.z >> 16u) & 0xFFu, cb.z >> 24u, x0_2, x1_2, d0, d1);
    pretiled_affine_fma4_m2(cb.w & 0xFFu, (cb.w >> 8u) & 0xFFu, (cb.w >> 16u) & 0xFFu, cb.w >> 24u, x0_3, x1_3, d0, d1);
  } else {
    const uint lw =
        reinterpret_cast<device const uint *>(c_tile + ulong(rn) * 16ul)[k_l];
    const uint hw = uint(
        reinterpret_cast<device const ushort *>(h_tile + ulong(rn) * 8ul)[k_l]);
    pretiled_affine_fma4_m2(
        (lw & 3u) | ((hw & 1u) << 2u),
        ((lw >> 2u) & 3u) | (((hw >> 1u) & 1u) << 2u),
        ((lw >> 4u) & 3u) | (((hw >> 2u) & 1u) << 2u),
        ((lw >> 6u) & 3u) | (((hw >> 3u) & 1u) << 2u),
        x0_0, x1_0, d0, d1);
    pretiled_affine_fma4_m2(
        ((lw >> 8u) & 3u) | (((hw >> 4u) & 1u) << 2u),
        ((lw >> 10u) & 3u) | (((hw >> 5u) & 1u) << 2u),
        ((lw >> 12u) & 3u) | (((hw >> 6u) & 1u) << 2u),
        ((lw >> 14u) & 3u) | (((hw >> 7u) & 1u) << 2u),
        x0_1, x1_1, d0, d1);
    pretiled_affine_fma4_m2(
        ((lw >> 16u) & 3u) | (((hw >> 8u) & 1u) << 2u),
        ((lw >> 18u) & 3u) | (((hw >> 9u) & 1u) << 2u),
        ((lw >> 20u) & 3u) | (((hw >> 10u) & 1u) << 2u),
        ((lw >> 22u) & 3u) | (((hw >> 11u) & 1u) << 2u),
        x0_2, x1_2, d0, d1);
    pretiled_affine_fma4_m2(
        ((lw >> 24u) & 3u) | (((hw >> 12u) & 1u) << 2u),
        ((lw >> 26u) & 3u) | (((hw >> 13u) & 1u) << 2u),
        ((lw >> 28u) & 3u) | (((hw >> 14u) & 1u) << 2u),
        ((lw >> 30u) & 3u) | (((hw >> 15u) & 1u) << 2u),
        x0_3, x1_3, d0, d1);
  }
  return float2(d0, d1);
}

__attribute__((always_inline)) inline float2 pretiled_affine_dot16_f32_m2_dyn(
    device const uchar *c_tile,
    device const uchar *h_tile,
    uint rn,
    uint bits,
    uint k_l,
    float4 x0_0, float4 x0_1, float4 x0_2, float4 x0_3,
    float4 x1_0, float4 x1_1, float4 x1_2, float4 x1_3) {
  if (bits == 4u) return pretiled_affine_dot16_f32_m2<4u>(c_tile, h_tile, rn, k_l, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3);
  if (bits == 6u) return pretiled_affine_dot16_f32_m2<6u>(c_tile, h_tile, rn, k_l, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3);
  if (bits == 8u) return pretiled_affine_dot16_f32_m2<8u>(c_tile, h_tile, rn, k_l, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3);
  if (bits == 5u) return pretiled_affine_dot16_f32_m2<5u>(c_tile, h_tile, rn, k_l, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3);
  return pretiled_affine_dot16_f32_m2<3u>(c_tile, h_tile, rn, k_l, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3);
}

template <bool PROLOGUE_NORM>
__attribute__((always_inline)) inline void pretiled_affine_load16_m2(
    device const bfloat *x0,
    device const bfloat *x1,
    device const bfloat *rms_w,
    ulong group_base,
    uint k_l,
    uint lane,
    bool valid_g,
    bool is_lane_v2,
    bool do_fwht,
    float inv_rms0,
    float inv_rms1,
    float4 off4,
    thread float4 &x0_0, thread float4 &x0_1, thread float4 &x0_2, thread float4 &x0_3,
    thread float4 &x1_0, thread float4 &x1_1, thread float4 &x1_2, thread float4 &x1_3,
    thread float &lsum0, thread float &lsum1) {
  float4 v0[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
  float4 v1[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
  if (valid_g) {
    if (is_lane_v2) {
      device const bfloat4 *xb0 = reinterpret_cast<device const bfloat4 *>(x0 + group_base);
      device const bfloat4 *xb1 = reinterpret_cast<device const bfloat4 *>(x1 + group_base);
      if constexpr (PROLOGUE_NORM) {
        device const bfloat4 *wb4 = reinterpret_cast<device const bfloat4 *>(rms_w + group_base);
#pragma clang loop unroll(full)
        for (uint s = 0u; s < 4u; ++s) {
          const uint idx4 = s * 4u + k_l;
          const float4 w = float4(wb4[idx4]) + off4;
          v0[s] = float4(bfloat4(float4(xb0[idx4]) * inv_rms0 * w));
          v1[s] = float4(bfloat4(float4(xb1[idx4]) * inv_rms1 * w));
        }
      } else {
        (void)rms_w; (void)inv_rms0; (void)inv_rms1; (void)off4;
#pragma clang loop unroll(full)
        for (uint s = 0u; s < 4u; ++s) {
          const uint idx4 = s * 4u + k_l;
          v0[s] = float4(xb0[idx4]);
          v1[s] = float4(xb1[idx4]);
        }
      }
    } else {
      const ulong x_off = group_base + ulong(k_l) * 16ul;
      device const bfloat4 *xb0 = reinterpret_cast<device const bfloat4 *>(x0 + x_off);
      device const bfloat4 *xb1 = reinterpret_cast<device const bfloat4 *>(x1 + x_off);
      if constexpr (PROLOGUE_NORM) {
        device const bfloat4 *wb4 = reinterpret_cast<device const bfloat4 *>(rms_w + x_off);
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 4u; ++i) {
          const float4 w = float4(wb4[i]) + off4;
          v0[i] = float4(bfloat4(float4(xb0[i]) * inv_rms0 * w));
          v1[i] = float4(bfloat4(float4(xb1[i]) * inv_rms1 * w));
        }
      } else {
        (void)rms_w; (void)inv_rms0; (void)inv_rms1; (void)off4;
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 4u; ++i) {
          v0[i] = float4(xb0[i]);
          v1[i] = float4(xb1[i]);
        }
      }
    }
  }
  if (do_fwht) {
    if (is_lane_v2) {
      omlx_m5_fwht64_sub4_pi64(v0, lane);
      omlx_m5_fwht64_sub4_pi64(v1, lane);
    } else {
      omlx_m5_fwht64_sub4_linear(v0, lane);
      omlx_m5_fwht64_sub4_linear(v1, lane);
    }
  }
  x0_0 = v0[0]; x0_1 = v0[1]; x0_2 = v0[2]; x0_3 = v0[3];
  x1_0 = v1[0]; x1_1 = v1[1]; x1_2 = v1[2]; x1_3 = v1[3];
  lsum0 = ((x0_0.x + x0_0.y) + (x0_0.z + x0_0.w)) +
          ((x0_1.x + x0_1.y) + (x0_1.z + x0_1.w)) +
          ((x0_2.x + x0_2.y) + (x0_2.z + x0_2.w)) +
          ((x0_3.x + x0_3.y) + (x0_3.z + x0_3.w));
  lsum1 = ((x1_0.x + x1_0.y) + (x1_0.z + x1_0.w)) +
          ((x1_1.x + x1_1.y) + (x1_1.z + x1_1.w)) +
          ((x1_2.x + x1_2.y) + (x1_2.z + x1_2.w)) +
          ((x1_3.x + x1_3.y) + (x1_3.z + x1_3.w));
}

template <uint BITS, uint BN, bool PROLOGUE_NORM = false, bool EPILOGUE_ADD_SQ = false>
inline void pretiled_qmv_affine_kl32_m2_impl(
    device const bfloat *x,
    device const uchar *nax_codes,
    device const uchar *nax_high,
    device const uint *nax_sb,
    device bfloat *y,
    uint n,
    uint k,
    uint tg_local,
    uint sg,
    uint lane,
    device const bfloat *rms_w = nullptr,
    float inv_rms0 = 1.0f,
    float inv_rms1 = 1.0f,
    float w_off = 0.0f,
    device atomic_float *rms_sq_buf = nullptr,
    uint sq_slot = 0u) {
  static_assert(BN == 16u || BN == 32u, "BN must be 16 or 32");
  const uint row_sg = (tg_local * 2u + sg) * 2u;
  if (row_sg >= n) {
    return;
  }
  const bool is_lane_v2 = (k & 0x40000000u) != 0u;
  const bool do_fwht = (k & 0x80000000u) != 0u;
  const uint k_eff = k & 0x3FFFFFFFu;
  const uint groups = k_eff / 64u;
  const uint tile_n = row_sg / BN;
  const uint rn0 = row_sg % BN;
  constexpr ulong code_tile_bytes =
      ulong(BN) * ((BITS == 3u) ? 16ul : ((BITS == 8u) ? 64ul : 32ul));
  constexpr ulong high_tile_bytes =
      ulong(BN) * ((BITS == 3u || BITS == 5u) ? 8ul : ((BITS == 6u) ? 16ul : 0ul));

  const uint g_sub = lane >> 2u;
  const uint k_l = lane & 3u;
  float a0_0 = 0.0f, a0_1 = 0.0f;
  float a1_0 = 0.0f, a1_1 = 0.0f;

  const float4 off4 = float4(w_off);
  const ulong tile_base = ulong(tile_n) * groups;
  device const bfloat *x1 = x + ulong(k_eff);
  for (uint g = 0u; g < groups; g += 8u) {
    const uint cur_g = g + g_sub;
    float4 x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3;
    float lsum0, lsum1;
    pretiled_affine_load16_m2<PROLOGUE_NORM>(
        x, x1, rms_w, ulong(cur_g) * 64ul, k_l, lane, cur_g < groups,
        is_lane_v2, do_fwht, inv_rms0, inv_rms1, off4,
        x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, lsum0, lsum1);
    if (cur_g < groups) {
      const ulong idx = tile_base + cur_g;
      device const uchar *c_tile = nax_codes + idx * code_tile_bytes;
      device const uchar *h_tile = nax_high + idx * high_tile_bytes;
      const uint2 sb2 = *reinterpret_cast<device const uint2 *>(nax_sb + idx * ulong(BN) + rn0);
      {
        const float sc = as_type<float>((sb2.x & 0xFFFFu) << 16u);
        const float bi = as_type<float>(sb2.x & 0xFFFF0000u) - 1024.0f * sc;
        const float2 dv = pretiled_affine_dot16_f32_m2<BITS>(c_tile, h_tile, rn0 + 0u, k_l, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3);
        a0_0 = fma(sc, dv.x, fma(bi, lsum0, a0_0));
        a1_0 = fma(sc, dv.y, fma(bi, lsum1, a1_0));
      }
      {
        const float sc = as_type<float>((sb2.y & 0xFFFFu) << 16u);
        const float bi = as_type<float>(sb2.y & 0xFFFF0000u) - 1024.0f * sc;
        const float2 dv = pretiled_affine_dot16_f32_m2<BITS>(c_tile, h_tile, rn0 + 1u, k_l, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3);
        a0_1 = fma(sc, dv.x, fma(bi, lsum0, a0_1));
        a1_1 = fma(sc, dv.y, fma(bi, lsum1, a1_1));
      }
    }
  }

  if constexpr (EPILOGUE_ADD_SQ) {
    float sg_sq0 = 0.0f;
    float sg_sq1 = 0.0f;
    pretiled_m2_epilogue_add_sq_row(a0_0, a1_0, y, row_sg + 0u, n, lane, sg_sq0, sg_sq1);
    pretiled_m2_epilogue_add_sq_row(a0_1, a1_1, y, row_sg + 1u, n, lane, sg_sq0, sg_sq1);
    if (lane == 0u) {
      atomic_fetch_add_explicit(&rms_sq_buf[sq_slot * 2u + 0u], sg_sq0, memory_order_relaxed);
      atomic_fetch_add_explicit(&rms_sq_buf[sq_slot * 2u + 1u], sg_sq1, memory_order_relaxed);
    }
  } else {
    (void)rms_sq_buf; (void)sq_slot;
    pretiled_m2_epilogue_row(a0_0, a1_0, y, row_sg + 0u, n, lane);
    pretiled_m2_epilogue_row(a0_1, a1_1, y, row_sg + 1u, n, lane);
  }
}

template <uint BITS, bool PROLOGUE_NORM = false>
inline void pretiled_qmv_affine_swiglu_kl32_m2_impl(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const uint *g_sb,
    device const uchar *u_codes,
    device const uchar *u_high,
    device const uint *u_sb,
    device bfloat *y,
    uint n,
    uint k,
    uint tg_local,
    uint sg,
    uint lane,
    device const bfloat *rms_w = nullptr,
    float inv_rms0 = 1.0f,
    float inv_rms1 = 1.0f,
    float w_off = 0.0f) {
  constexpr uint BN = 16u;
  const uint row_sg = (tg_local * 2u + sg) * 2u;
  if (row_sg >= n) {
    return;
  }
  const bool is_lane_v2 = (k & 0x40000000u) != 0u;
  const bool do_fwht = (k & 0x80000000u) != 0u;
  const uint k_eff = k & 0x3FFFFFFFu;
  const uint groups = k_eff / 64u;
  const uint tile_n = row_sg / BN;
  const uint rn0 = row_sg % BN;
  constexpr ulong c_tile_bytes =
      ulong(BN) * ((BITS == 3u) ? 16ul : ((BITS == 8u) ? 64ul : 32ul));
  constexpr ulong h_tile_bytes =
      ulong(BN) * ((BITS == 3u || BITS == 5u) ? 8ul : ((BITS == 6u) ? 16ul : 0ul));

  const uint g_sub = lane >> 2u;
  const uint k_l = lane & 3u;
  float ag0_0 = 0.0f, ag0_1 = 0.0f, ag1_0 = 0.0f, ag1_1 = 0.0f;
  float au0_0 = 0.0f, au0_1 = 0.0f, au1_0 = 0.0f, au1_1 = 0.0f;

  const float4 off4 = float4(w_off);
  const ulong tile_base = ulong(tile_n) * groups;
  device const bfloat *x1 = x + ulong(k_eff);
  for (uint g = 0u; g < groups; g += 8u) {
    const uint cur_g = g + g_sub;
    float4 x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3;
    float lsum0, lsum1;
    pretiled_affine_load16_m2<PROLOGUE_NORM>(
        x, x1, rms_w, ulong(cur_g) * 64ul, k_l, lane, cur_g < groups,
        is_lane_v2, do_fwht, inv_rms0, inv_rms1, off4,
        x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, lsum0, lsum1);
    if (cur_g < groups) {
      const ulong idx = tile_base + cur_g;
      device const uchar *gc_t = g_codes + idx * c_tile_bytes;
      device const uchar *gh_t = g_high + idx * h_tile_bytes;
      device const uchar *uc_t = u_codes + idx * c_tile_bytes;
      device const uchar *uh_t = u_high + idx * h_tile_bytes;
      const uint2 gp2 = *reinterpret_cast<device const uint2 *>(g_sb + idx * ulong(BN) + rn0);
      const uint2 up2 = *reinterpret_cast<device const uint2 *>(u_sb + idx * ulong(BN) + rn0);
      {
        const float gsc = as_type<float>((gp2.x & 0xFFFFu) << 16u);
        const float gbi = as_type<float>(gp2.x & 0xFFFF0000u) - 1024.0f * gsc;
        const float usc = as_type<float>((up2.x & 0xFFFFu) << 16u);
        const float ubi = as_type<float>(up2.x & 0xFFFF0000u) - 1024.0f * usc;
        const float2 dg = pretiled_affine_dot16_f32_m2<BITS>(gc_t, gh_t, rn0 + 0u, k_l, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3);
        const float2 du = pretiled_affine_dot16_f32_m2<BITS>(uc_t, uh_t, rn0 + 0u, k_l, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3);
        ag0_0 = fma(gsc, dg.x, fma(gbi, lsum0, ag0_0));
        ag1_0 = fma(gsc, dg.y, fma(gbi, lsum1, ag1_0));
        au0_0 = fma(usc, du.x, fma(ubi, lsum0, au0_0));
        au1_0 = fma(usc, du.y, fma(ubi, lsum1, au1_0));
      }
      {
        const float gsc = as_type<float>((gp2.y & 0xFFFFu) << 16u);
        const float gbi = as_type<float>(gp2.y & 0xFFFF0000u) - 1024.0f * gsc;
        const float usc = as_type<float>((up2.y & 0xFFFFu) << 16u);
        const float ubi = as_type<float>(up2.y & 0xFFFF0000u) - 1024.0f * usc;
        const float2 dg = pretiled_affine_dot16_f32_m2<BITS>(gc_t, gh_t, rn0 + 1u, k_l, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3);
        const float2 du = pretiled_affine_dot16_f32_m2<BITS>(uc_t, uh_t, rn0 + 1u, k_l, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3);
        ag0_1 = fma(gsc, dg.x, fma(gbi, lsum0, ag0_1));
        ag1_1 = fma(gsc, dg.y, fma(gbi, lsum1, ag1_1));
        au0_1 = fma(usc, du.x, fma(ubi, lsum0, au0_1));
        au1_1 = fma(usc, du.y, fma(ubi, lsum1, au1_1));
      }
    }
  }
  pretiled_m2_epilogue_swiglu_row(ag0_0, au0_0, ag1_0, au1_0, y, row_sg + 0u, n, lane);
  pretiled_m2_epilogue_swiglu_row(ag0_1, au0_1, ag1_1, au1_1, y, row_sg + 1u, n, lane);
}

template <bool PROLOGUE_NORM = false>
inline void pretiled_qmv_affine_swiglu_kl32_m2_mixed_fallback(
    device const bfloat *x,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const uint *g_sb,
    device const uchar *u_codes,
    device const uchar *u_high,
    device const uint *u_sb,
    device bfloat *y,
    uint n,
    uint k,
    uint g_bits,
    uint u_bits,
    uint tg_local,
    uint sg,
    uint lane,
    device const bfloat *rms_w = nullptr,
    float inv_rms0 = 1.0f,
    float inv_rms1 = 1.0f,
    float w_off = 0.0f) {
  const uint bn = (n >= 65536u) ? 32u : 16u;
  const uint row_sg = (tg_local * 2u + sg) * 2u;
  if (row_sg >= n) return;
  const bool is_lane_v2 = (k & 0x40000000u) != 0u;
  const bool do_fwht = (k & 0x80000000u) != 0u;
  const uint k_eff = k & 0x3FFFFFFFu;
  const uint groups = k_eff / 64u;
  const uint tile_n = row_sg / bn;
  const uint rn0 = row_sg % bn;
  const ulong gc_tile_bytes = ulong(bn) * ((g_bits == 3u) ? 16ul : ((g_bits == 8u) ? 64ul : 32ul));
  const ulong gh_tile_bytes = ulong(bn) * ((g_bits == 3u || g_bits == 5u) ? 8ul : ((g_bits == 6u) ? 16ul : 0ul));
  const ulong uc_tile_bytes = ulong(bn) * ((u_bits == 3u) ? 16ul : ((u_bits == 8u) ? 64ul : 32ul));
  const ulong uh_tile_bytes = ulong(bn) * ((u_bits == 3u || u_bits == 5u) ? 8ul : ((u_bits == 6u) ? 16ul : 0ul));

  const uint g_sub = lane >> 2u;
  const uint k_l = lane & 3u;
  float ag0_0 = 0.0f, ag0_1 = 0.0f, ag1_0 = 0.0f, ag1_1 = 0.0f;
  float au0_0 = 0.0f, au0_1 = 0.0f, au1_0 = 0.0f, au1_1 = 0.0f;

  const float4 off4 = float4(w_off);
  const ulong tile_base = ulong(tile_n) * groups;
  device const bfloat *x1 = x + ulong(k_eff);
  for (uint g = 0u; g < groups; g += 8u) {
    const uint cur_g = g + g_sub;
    float4 x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3;
    float lsum0, lsum1;
    pretiled_affine_load16_m2<PROLOGUE_NORM>(
        x, x1, rms_w, ulong(cur_g) * 64ul, k_l, lane, cur_g < groups,
        is_lane_v2, do_fwht, inv_rms0, inv_rms1, off4,
        x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3, lsum0, lsum1);
    if (cur_g < groups) {
      const ulong idx = tile_base + cur_g;
      device const uchar *gc_t = g_codes + idx * gc_tile_bytes;
      device const uchar *gh_t = g_high + idx * gh_tile_bytes;
      device const uchar *uc_t = u_codes + idx * uc_tile_bytes;
      device const uchar *uh_t = u_high + idx * uh_tile_bytes;
      device const uint *gsb_r = g_sb + idx * ulong(bn) + rn0;
      device const uint *usb_r = u_sb + idx * ulong(bn) + rn0;
      {
        const uint gp = gsb_r[0];
        const uint up = usb_r[0];
        const float gsc = as_type<float>((gp & 0xFFFFu) << 16u);
        const float gbi = as_type<float>(gp & 0xFFFF0000u) - 1024.0f * gsc;
        const float usc = as_type<float>((up & 0xFFFFu) << 16u);
        const float ubi = as_type<float>(up & 0xFFFF0000u) - 1024.0f * usc;
        const float2 dg = pretiled_affine_dot16_f32_m2_dyn(gc_t, gh_t, rn0 + 0u, g_bits, k_l, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3);
        const float2 du = pretiled_affine_dot16_f32_m2_dyn(uc_t, uh_t, rn0 + 0u, u_bits, k_l, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3);
        ag0_0 = fma(gsc, dg.x, fma(gbi, lsum0, ag0_0));
        ag1_0 = fma(gsc, dg.y, fma(gbi, lsum1, ag1_0));
        au0_0 = fma(usc, du.x, fma(ubi, lsum0, au0_0));
        au1_0 = fma(usc, du.y, fma(ubi, lsum1, au1_0));
      }
      {
        const uint gp = gsb_r[1];
        const uint up = usb_r[1];
        const float gsc = as_type<float>((gp & 0xFFFFu) << 16u);
        const float gbi = as_type<float>(gp & 0xFFFF0000u) - 1024.0f * gsc;
        const float usc = as_type<float>((up & 0xFFFFu) << 16u);
        const float ubi = as_type<float>(up & 0xFFFF0000u) - 1024.0f * usc;
        const float2 dg = pretiled_affine_dot16_f32_m2_dyn(gc_t, gh_t, rn0 + 1u, g_bits, k_l, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3);
        const float2 du = pretiled_affine_dot16_f32_m2_dyn(uc_t, uh_t, rn0 + 1u, u_bits, k_l, x0_0, x0_1, x0_2, x0_3, x1_0, x1_1, x1_2, x1_3);
        ag0_1 = fma(gsc, dg.x, fma(gbi, lsum0, ag0_1));
        ag1_1 = fma(gsc, dg.y, fma(gbi, lsum1, ag1_1));
        au0_1 = fma(usc, du.x, fma(ubi, lsum0, au0_1));
        au1_1 = fma(usc, du.y, fma(ubi, lsum1, au1_1));
      }
    }
  }
  pretiled_m2_epilogue_swiglu_row(ag0_0, au0_0, ag1_0, au1_0, y, row_sg + 0u, n, lane);
  pretiled_m2_epilogue_swiglu_row(ag0_1, au0_1, ag1_1, au1_1, y, row_sg + 1u, n, lane);
}

template <bool PROLOGUE_NORM = false>
inline void pretiled_qmv_affine_kl32_m2_bn16_dispatch(
    device const bfloat *x, device const uchar *c, device const uchar *h,
    device const uint *sb, device bfloat *y, uint n, uint k, uint bits,
    uint tg_l, uint sg, uint lane,
    device const bfloat *rms_w = nullptr, float inv0 = 1.0f, float inv1 = 1.0f, float w_off = 0.0f) {
  if (bits == 4u) pretiled_qmv_affine_kl32_m2_impl<4u, 16u, PROLOGUE_NORM, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv0, inv1, w_off);
  else if (bits == 5u) pretiled_qmv_affine_kl32_m2_impl<5u, 16u, PROLOGUE_NORM, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv0, inv1, w_off);
  else if (bits == 6u) pretiled_qmv_affine_kl32_m2_impl<6u, 16u, PROLOGUE_NORM, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv0, inv1, w_off);
  else if (bits == 8u) pretiled_qmv_affine_kl32_m2_impl<8u, 16u, PROLOGUE_NORM, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv0, inv1, w_off);
  else pretiled_qmv_affine_kl32_m2_impl<3u, 16u, PROLOGUE_NORM, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv0, inv1, w_off);
}

template <bool PROLOGUE_NORM = false>
inline void pretiled_qmv_affine_kl32_m2_bn16_q456_dispatch(
    device const bfloat *x, device const uchar *c, device const uchar *h,
    device const uint *sb, device bfloat *y, uint n, uint k, uint bits,
    uint tg_l, uint sg, uint lane,
    device const bfloat *rms_w = nullptr, float inv0 = 1.0f, float inv1 = 1.0f, float w_off = 0.0f) {
  if (bits == 5u) pretiled_qmv_affine_kl32_m2_impl<5u, 16u, PROLOGUE_NORM, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv0, inv1, w_off);
  else if (bits == 4u) pretiled_qmv_affine_kl32_m2_impl<4u, 16u, PROLOGUE_NORM, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv0, inv1, w_off);
  else pretiled_qmv_affine_kl32_m2_impl<6u, 16u, PROLOGUE_NORM, false>(x, c, h, sb, y, n, k, tg_l, sg, lane, rms_w, inv0, inv1, w_off);
}

#define OMLX_M5_DEF_AFFINE_M2_SPEC(NAME, BITS, BN)                              \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *c [[buffer(1)]],                                      \
      device const uchar *h [[buffer(2)]],                                      \
      device const uint *sb [[buffer(3)]],                                      \
      device bfloat *y [[buffer(4)]],                                           \
      constant PretiledQmvAffineParams &p [[buffer(5)]],                        \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    pretiled_qmv_affine_kl32_m2_impl<BITS, BN>(                                 \
        x, c, h, sb, y, p.n0, p.k, tg.x, sg, lane);                             \
  }

OMLX_M5_DEF_AFFINE_M2_SPEC(omlx_m5_pretiled_qmv_affine_q4_bn16_r2_m2, 4u, 16u)
OMLX_M5_DEF_AFFINE_M2_SPEC(omlx_m5_pretiled_qmv_affine_q5_bn16_r2_m2, 5u, 16u)
OMLX_M5_DEF_AFFINE_M2_SPEC(omlx_m5_pretiled_qmv_affine_q6_bn16_r2_m2, 6u, 16u)
OMLX_M5_DEF_AFFINE_M2_SPEC(omlx_m5_pretiled_qmv_affine_q8_bn16_r2_m2, 8u, 16u)
OMLX_M5_DEF_AFFINE_M2_SPEC(omlx_m5_pretiled_qmv_affine_q4_bn32_r2_m2, 4u, 32u)
OMLX_M5_DEF_AFFINE_M2_SPEC(omlx_m5_pretiled_qmv_affine_q5_bn32_r2_m2, 5u, 32u)
OMLX_M5_DEF_AFFINE_M2_SPEC(omlx_m5_pretiled_qmv_affine_q6_bn32_r2_m2, 6u, 32u)
OMLX_M5_DEF_AFFINE_M2_SPEC(omlx_m5_pretiled_qmv_affine_q8_bn32_r2_m2, 8u, 32u)

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_affine_r2_m2(
    device const bfloat *x [[buffer(0)]],
    device const uchar *c [[buffer(1)]],
    device const uchar *h [[buffer(2)]],
    device const uint *sb [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant PretiledQmvAffineParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint bits = p.bits & 0xFFu;
  if (p.n0 >= 65536u) {
    if (bits == 6u) pretiled_qmv_affine_kl32_m2_impl<6u, 32u>(x, c, h, sb, y, p.n0, p.k, tg.x, sg, lane);
    else if (bits == 4u) pretiled_qmv_affine_kl32_m2_impl<4u, 32u>(x, c, h, sb, y, p.n0, p.k, tg.x, sg, lane);
    else if (bits == 5u) pretiled_qmv_affine_kl32_m2_impl<5u, 32u>(x, c, h, sb, y, p.n0, p.k, tg.x, sg, lane);
    else if (bits == 8u) pretiled_qmv_affine_kl32_m2_impl<8u, 32u>(x, c, h, sb, y, p.n0, p.k, tg.x, sg, lane);
    else pretiled_qmv_affine_kl32_m2_impl<3u, 32u>(x, c, h, sb, y, p.n0, p.k, tg.x, sg, lane);
  } else {
    pretiled_qmv_affine_kl32_m2_bn16_dispatch<false>(x, c, h, sb, y, p.n0, p.k, bits, tg.x, sg, lane);
  }
}

#define OMLX_M5_DEF_AFFINE_SWIGLU_M2_SPEC(NAME, BITS)                           \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *gc [[buffer(1)]],                                     \
      device const uchar *gh [[buffer(2)]],                                     \
      device const uint *gsb [[buffer(3)]],                                     \
      device const uchar *uc [[buffer(4)]],                                     \
      device const uchar *uh [[buffer(5)]],                                     \
      device const uint *usb [[buffer(6)]],                                     \
      device bfloat *y [[buffer(7)]],                                           \
      constant PretiledQmvAffineParams &p [[buffer(8)]],                        \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    pretiled_qmv_affine_swiglu_kl32_m2_impl<BITS>(                              \
        x, gc, gh, gsb, uc, uh, usb, y, p.n0, p.k, tg.x, sg, lane);             \
  }

OMLX_M5_DEF_AFFINE_SWIGLU_M2_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q4_r2_m2, 4u)
OMLX_M5_DEF_AFFINE_SWIGLU_M2_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q5_r2_m2, 5u)
OMLX_M5_DEF_AFFINE_SWIGLU_M2_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q6_r2_m2, 6u)
OMLX_M5_DEF_AFFINE_SWIGLU_M2_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q8_r2_m2, 8u)

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_affine_swiglu_r2_m2(
    device const bfloat *x [[buffer(0)]],
    device const uchar *gc [[buffer(1)]],
    device const uchar *gh [[buffer(2)]],
    device const uint *gsb [[buffer(3)]],
    device const uchar *uc [[buffer(4)]],
    device const uchar *uh [[buffer(5)]],
    device const uint *usb [[buffer(6)]],
    device bfloat *y [[buffer(7)]],
    constant PretiledQmvAffineParams &p [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g_bits = p.bits & 0xFFu;
  const uint u_hi = (p.bits >> 8u) & 0xFFu;
  const uint u_bits = (u_hi != 0u) ? u_hi : g_bits;
  pretiled_qmv_affine_swiglu_kl32_m2_mixed_fallback<false>(
      x, gc, gh, gsb, uc, uh, usb, y, p.n0, p.k, g_bits, u_bits, tg.x, sg, lane);
}

#define OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_M2_SPEC(NAME, BITS)                     \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *c0 [[buffer(1)]],                                     \
      device const uchar *h0 [[buffer(2)]],                                     \
      device const uint *sb0 [[buffer(3)]],                                     \
      device const uchar *c1 [[buffer(4)]],                                     \
      device const uchar *h1 [[buffer(5)]],                                     \
      device const uint *sb1 [[buffer(6)]],                                     \
      device const uchar *c2 [[buffer(7)]],                                     \
      device const uchar *h2 [[buffer(8)]],                                     \
      device const uint *sb2 [[buffer(9)]],                                     \
      device bfloat *y0 [[buffer(10)]],                                         \
      device bfloat *y1 [[buffer(11)]],                                         \
      device bfloat *y2 [[buffer(12)]],                                         \
      constant PretiledQmvAffineParams &p [[buffer(13)]],                       \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    const uint tgs0 = (p.n0 + 3u) / 4u;                                         \
    const uint tgs1 = (p.n1 + 3u) / 4u;                                         \
    device const uchar *cs = (tg.x < tgs0) ? c0 : ((tg.x < tgs0 + tgs1) ? c1 : c2); \
    device const uchar *hs = (tg.x < tgs0) ? h0 : ((tg.x < tgs0 + tgs1) ? h1 : h2); \
    device const uint *sbs = (tg.x < tgs0) ? sb0 : ((tg.x < tgs0 + tgs1) ? sb1 : sb2); \
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2);  \
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2); \
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1)); \
    pretiled_qmv_affine_kl32_m2_impl<BITS, 16u>(x, cs, hs, sbs, ys, ns, p.k, tgl, sg, lane); \
  }

OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_M2_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q4_r2_m2, 4u)
OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_M2_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q5_r2_m2, 5u)
OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_M2_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q6_r2_m2, 6u)

#define OMLX_M5_DEF_AFFINE_CONCAT3_M2_SPEC(NAME, DISPATCH_FN)                   \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *c0 [[buffer(1)]],                                     \
      device const uchar *h0 [[buffer(2)]],                                     \
      device const uint *sb0 [[buffer(3)]],                                     \
      device const uchar *c1 [[buffer(4)]],                                     \
      device const uchar *h1 [[buffer(5)]],                                     \
      device const uint *sb1 [[buffer(6)]],                                     \
      device const uchar *c2 [[buffer(7)]],                                     \
      device const uchar *h2 [[buffer(8)]],                                     \
      device const uint *sb2 [[buffer(9)]],                                     \
      device bfloat *y0 [[buffer(10)]],                                         \
      device bfloat *y1 [[buffer(11)]],                                         \
      device bfloat *y2 [[buffer(12)]],                                         \
      constant PretiledQmvAffineParams &p [[buffer(13)]],                       \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    const uint b0 = p.bits & 0xFFu;                                             \
    const uint b1_r = (p.bits >> 8u) & 0xFFu;                                   \
    const uint b2_r = (p.bits >> 16u) & 0xFFu;                                  \
    const uint b1 = (b1_r != 0u) ? b1_r : b0;                                   \
    const uint b2 = (b2_r != 0u) ? b2_r : b0;                                   \
    const uint tgs0 = (p.n0 + 3u) / 4u;                                         \
    const uint tgs1 = (p.n1 + 3u) / 4u;                                         \
    device const uchar *cs = (tg.x < tgs0) ? c0 : ((tg.x < tgs0 + tgs1) ? c1 : c2); \
    device const uchar *hs = (tg.x < tgs0) ? h0 : ((tg.x < tgs0 + tgs1) ? h1 : h2); \
    device const uint *sbs = (tg.x < tgs0) ? sb0 : ((tg.x < tgs0 + tgs1) ? sb1 : sb2); \
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2);  \
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2); \
    const uint bs = (tg.x < tgs0) ? b0 : ((tg.x < tgs0 + tgs1) ? b1 : b2);      \
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1)); \
    DISPATCH_FN<false>(x, cs, hs, sbs, ys, ns, p.k, bs, tgl, sg, lane);         \
  }

OMLX_M5_DEF_AFFINE_CONCAT3_M2_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q456_r2_m2, pretiled_qmv_affine_kl32_m2_bn16_q456_dispatch)
OMLX_M5_DEF_AFFINE_CONCAT3_M2_SPEC(omlx_m5_pretiled_qmv_affine_concat3_r2_m2, pretiled_qmv_affine_kl32_m2_bn16_dispatch)

#define OMLX_M5_DEF_AFFINE_CONCAT4_M2_SPEC(NAME, DISPATCH_FN)                   \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *c0 [[buffer(1)]],                                     \
      device const uchar *h0 [[buffer(2)]],                                     \
      device const uint *sb0 [[buffer(3)]],                                     \
      device const uchar *c1 [[buffer(4)]],                                     \
      device const uchar *h1 [[buffer(5)]],                                     \
      device const uint *sb1 [[buffer(6)]],                                     \
      device const uchar *c2 [[buffer(7)]],                                     \
      device const uchar *h2 [[buffer(8)]],                                     \
      device const uint *sb2 [[buffer(9)]],                                     \
      device const uchar *c3 [[buffer(10)]],                                    \
      device const uchar *h3 [[buffer(11)]],                                    \
      device const uint *sb3 [[buffer(12)]],                                    \
      device bfloat *y0 [[buffer(13)]],                                         \
      device bfloat *y1 [[buffer(14)]],                                         \
      device bfloat *y2 [[buffer(15)]],                                         \
      device bfloat *y3 [[buffer(16)]],                                         \
      constant PretiledQmvConcat4AffineParams &p [[buffer(17)]],                \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    const uint b0 = p.bits & 0xFFu;                                             \
    const uint b1_r = (p.bits >> 8u) & 0xFFu;                                   \
    const uint b2_r = (p.bits >> 16u) & 0xFFu;                                  \
    const uint b3_r = (p.bits >> 24u) & 0xFFu;                                  \
    const uint b1 = (b1_r != 0u) ? b1_r : b0;                                   \
    const uint b2 = (b2_r != 0u) ? b2_r : b0;                                   \
    const uint b3 = (b3_r != 0u) ? b3_r : b0;                                   \
    const uint tgs0 = (p.n0 + 3u) / 4u;                                         \
    const uint tgs1 = (p.n1 + 3u) / 4u;                                         \
    const uint tgs2 = (p.n2 + 3u) / 4u;                                         \
    device const uchar *cs = c3;                                                \
    device const uchar *hs = h3;                                                \
    device const uint *sbs = sb3;                                               \
    device bfloat *ys = y3;                                                     \
    uint ns = p.n3;                                                             \
    uint bs = b3;                                                               \
    uint tgl = tg.x - tgs0 - tgs1 - tgs2;                                       \
    if (tg.x < tgs0) { cs = c0; hs = h0; sbs = sb0; ys = y0; ns = p.n0; bs = b0; tgl = tg.x; } \
    else if (tg.x < tgs0 + tgs1) { cs = c1; hs = h1; sbs = sb1; ys = y1; ns = p.n1; bs = b1; tgl = tg.x - tgs0; } \
    else if (tg.x < tgs0 + tgs1 + tgs2) { cs = c2; hs = h2; sbs = sb2; ys = y2; ns = p.n2; bs = b2; tgl = tg.x - tgs0 - tgs1; } \
    DISPATCH_FN<false>(x, cs, hs, sbs, ys, ns, p.k, bs, tgl, sg, lane);         \
  }

OMLX_M5_DEF_AFFINE_CONCAT4_M2_SPEC(omlx_m5_pretiled_qmv_affine_concat4_q456_r2_m2, pretiled_qmv_affine_kl32_m2_bn16_q456_dispatch)
OMLX_M5_DEF_AFFINE_CONCAT4_M2_SPEC(omlx_m5_pretiled_qmv_affine_concat4_r2_m2, pretiled_qmv_affine_kl32_m2_bn16_dispatch)

#define OMLX_M5_DEF_AFFINE_ADD_SQ_M2_SPEC(NAME, BITS, BN)                       \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *x [[buffer(0)]],                                     \
      device const uchar *c [[buffer(1)]],                                      \
      device const uchar *h [[buffer(2)]],                                      \
      device const uint *sb [[buffer(3)]],                                      \
      device bfloat *hidden_inout [[buffer(4)]],                                \
      device atomic_float *rms_sq_buf [[buffer(5)]],                            \
      constant PretiledQmvFusedNormParams &p [[buffer(6)]],                     \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    pretiled_qmv_affine_kl32_m2_impl<BITS, BN, false, true>(                    \
        x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane,                   \
        nullptr, 1.0f, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);                      \
  }

OMLX_M5_DEF_AFFINE_ADD_SQ_M2_SPEC(omlx_m5_pretiled_qmv_affine_q4_bn16_r2_add_sq_m2, 4u, 16u)
OMLX_M5_DEF_AFFINE_ADD_SQ_M2_SPEC(omlx_m5_pretiled_qmv_affine_q5_bn16_r2_add_sq_m2, 5u, 16u)
OMLX_M5_DEF_AFFINE_ADD_SQ_M2_SPEC(omlx_m5_pretiled_qmv_affine_q6_bn16_r2_add_sq_m2, 6u, 16u)
OMLX_M5_DEF_AFFINE_ADD_SQ_M2_SPEC(omlx_m5_pretiled_qmv_affine_q8_bn16_r2_add_sq_m2, 8u, 16u)

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_affine_r2_add_sq_m2(
    device const bfloat *x [[buffer(0)]],
    device const uchar *c [[buffer(1)]],
    device const uchar *h [[buffer(2)]],
    device const uint *sb [[buffer(3)]],
    device bfloat *hidden_inout [[buffer(4)]],
    device atomic_float *rms_sq_buf [[buffer(5)]],
    constant PretiledQmvFusedNormParams &p [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint bits = p.bits & 0xFFu;
  if (bits == 4u) pretiled_qmv_affine_kl32_m2_impl<4u, 16u, false, true>(x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane, nullptr, 1.0f, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);
  else if (bits == 5u) pretiled_qmv_affine_kl32_m2_impl<5u, 16u, false, true>(x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane, nullptr, 1.0f, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);
  else if (bits == 6u) pretiled_qmv_affine_kl32_m2_impl<6u, 16u, false, true>(x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane, nullptr, 1.0f, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);
  else if (bits == 8u) pretiled_qmv_affine_kl32_m2_impl<8u, 16u, false, true>(x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane, nullptr, 1.0f, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);
  else pretiled_qmv_affine_kl32_m2_impl<3u, 16u, false, true>(x, c, h, sb, hidden_inout, p.n0, p.k, tg.x, sg, lane, nullptr, 1.0f, 1.0f, 0.0f, rms_sq_buf, p.sq_slot);
}

#define OMLX_M5_DEF_AFFINE_NORM_M2_SPEC(NAME, BITS, BN)                         \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *c [[buffer(3)]],                                      \
      device const uchar *h [[buffer(4)]],                                      \
      device const uint *sb [[buffer(5)]],                                      \
      device bfloat *y [[buffer(6)]],                                           \
      constant PretiledQmvFusedNormParams &p [[buffer(7)]],                     \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    const float inv0 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 0u] * p.inv_k + p.eps); \
    const float inv1 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 1u] * p.inv_k + p.eps); \
    pretiled_qmv_affine_kl32_m2_impl<BITS, BN, true, false>(                    \
        hidden, c, h, sb, y, p.n0, p.k, tg.x, sg, lane, rms_w, inv0, inv1, p.w_off); \
  }

OMLX_M5_DEF_AFFINE_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_q4_bn32_r2_norm_m2, 4u, 32u)
OMLX_M5_DEF_AFFINE_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_q5_bn32_r2_norm_m2, 5u, 32u)
OMLX_M5_DEF_AFFINE_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_q6_bn32_r2_norm_m2, 6u, 32u)
OMLX_M5_DEF_AFFINE_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_q8_bn32_r2_norm_m2, 8u, 32u)
OMLX_M5_DEF_AFFINE_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_q4_bn16_r2_norm_m2, 4u, 16u)
OMLX_M5_DEF_AFFINE_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_q5_bn16_r2_norm_m2, 5u, 16u)
OMLX_M5_DEF_AFFINE_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_q6_bn16_r2_norm_m2, 6u, 16u)
OMLX_M5_DEF_AFFINE_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_q8_bn16_r2_norm_m2, 8u, 16u)

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_affine_r2_norm_m2(
    device const bfloat *hidden [[buffer(0)]],
    device const bfloat *rms_w [[buffer(1)]],
    device const float *rms_sq_buf [[buffer(2)]],
    device const uchar *c [[buffer(3)]],
    device const uchar *h [[buffer(4)]],
    device const uint *sb [[buffer(5)]],
    device bfloat *y [[buffer(6)]],
    constant PretiledQmvFusedNormParams &p [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint bits = p.bits & 0xFFu;
  const float inv0 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 0u] * p.inv_k + p.eps);
  const float inv1 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 1u] * p.inv_k + p.eps);
  if (p.n0 >= 65536u) {
    if (bits == 6u) pretiled_qmv_affine_kl32_m2_impl<6u, 32u, true, false>(hidden, c, h, sb, y, p.n0, p.k, tg.x, sg, lane, rms_w, inv0, inv1, p.w_off);
    else if (bits == 4u) pretiled_qmv_affine_kl32_m2_impl<4u, 32u, true, false>(hidden, c, h, sb, y, p.n0, p.k, tg.x, sg, lane, rms_w, inv0, inv1, p.w_off);
    else if (bits == 5u) pretiled_qmv_affine_kl32_m2_impl<5u, 32u, true, false>(hidden, c, h, sb, y, p.n0, p.k, tg.x, sg, lane, rms_w, inv0, inv1, p.w_off);
    else if (bits == 8u) pretiled_qmv_affine_kl32_m2_impl<8u, 32u, true, false>(hidden, c, h, sb, y, p.n0, p.k, tg.x, sg, lane, rms_w, inv0, inv1, p.w_off);
    else pretiled_qmv_affine_kl32_m2_impl<3u, 32u, true, false>(hidden, c, h, sb, y, p.n0, p.k, tg.x, sg, lane, rms_w, inv0, inv1, p.w_off);
  } else {
    pretiled_qmv_affine_kl32_m2_bn16_dispatch<true>(hidden, c, h, sb, y, p.n0, p.k, bits, tg.x, sg, lane, rms_w, inv0, inv1, p.w_off);
  }
}

#define OMLX_M5_DEF_AFFINE_SWIGLU_NORM_M2_SPEC(NAME, BITS)                      \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *gc [[buffer(3)]],                                     \
      device const uchar *gh [[buffer(4)]],                                     \
      device const uint *gsb [[buffer(5)]],                                     \
      device const uchar *uc [[buffer(6)]],                                     \
      device const uchar *uh [[buffer(7)]],                                     \
      device const uint *usb [[buffer(8)]],                                     \
      device bfloat *y [[buffer(9)]],                                           \
      constant PretiledQmvFusedNormParams &p [[buffer(10)]],                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    const float inv0 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 0u] * p.inv_k + p.eps); \
    const float inv1 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 1u] * p.inv_k + p.eps); \
    pretiled_qmv_affine_swiglu_kl32_m2_impl<BITS, true>(                        \
        hidden, gc, gh, gsb, uc, uh, usb, y, p.n0, p.k, tg.x, sg, lane,         \
        rms_w, inv0, inv1, p.w_off);                                            \
  }

OMLX_M5_DEF_AFFINE_SWIGLU_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q4_r2_norm_m2, 4u)
OMLX_M5_DEF_AFFINE_SWIGLU_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q5_r2_norm_m2, 5u)
OMLX_M5_DEF_AFFINE_SWIGLU_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q6_r2_norm_m2, 6u)
OMLX_M5_DEF_AFFINE_SWIGLU_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_swiglu_q8_r2_norm_m2, 8u)

[[max_total_threads_per_threadgroup(64)]]
kernel void omlx_m5_pretiled_qmv_affine_swiglu_r2_norm_m2(
    device const bfloat *hidden [[buffer(0)]],
    device const bfloat *rms_w [[buffer(1)]],
    device const float *rms_sq_buf [[buffer(2)]],
    device const uchar *gc [[buffer(3)]],
    device const uchar *gh [[buffer(4)]],
    device const uint *gsb [[buffer(5)]],
    device const uchar *uc [[buffer(6)]],
    device const uchar *uh [[buffer(7)]],
    device const uint *usb [[buffer(8)]],
    device bfloat *y [[buffer(9)]],
    constant PretiledQmvFusedNormParams &p [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g_bits = p.bits & 0xFFu;
  const uint u_hi = (p.bits >> 8u) & 0xFFu;
  const uint u_bits = (u_hi != 0u) ? u_hi : g_bits;
  const float inv0 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 0u] * p.inv_k + p.eps);
  const float inv1 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 1u] * p.inv_k + p.eps);
  pretiled_qmv_affine_swiglu_kl32_m2_mixed_fallback<true>(
      hidden, gc, gh, gsb, uc, uh, usb, y, p.n0, p.k, g_bits, u_bits, tg.x, sg,
      lane, rms_w, inv0, inv1, p.w_off);
}

#define OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_NORM_M2_SPEC(NAME, BITS)                \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *c0 [[buffer(3)]],                                     \
      device const uchar *h0 [[buffer(4)]],                                     \
      device const uint *sb0 [[buffer(5)]],                                     \
      device const uchar *c1 [[buffer(6)]],                                     \
      device const uchar *h1 [[buffer(7)]],                                     \
      device const uint *sb1 [[buffer(8)]],                                     \
      device const uchar *c2 [[buffer(9)]],                                     \
      device const uchar *h2 [[buffer(10)]],                                    \
      device const uint *sb2 [[buffer(11)]],                                    \
      device bfloat *y0 [[buffer(12)]],                                         \
      device bfloat *y1 [[buffer(13)]],                                         \
      device bfloat *y2 [[buffer(14)]],                                         \
      constant PretiledQmvFusedNormParams &p [[buffer(15)]],                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    const uint tgs0 = (p.n0 + 3u) / 4u;                                         \
    const uint tgs1 = (p.n1 + 3u) / 4u;                                         \
    device const uchar *cs = (tg.x < tgs0) ? c0 : ((tg.x < tgs0 + tgs1) ? c1 : c2); \
    device const uchar *hs = (tg.x < tgs0) ? h0 : ((tg.x < tgs0 + tgs1) ? h1 : h2); \
    device const uint *sbs = (tg.x < tgs0) ? sb0 : ((tg.x < tgs0 + tgs1) ? sb1 : sb2); \
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2);  \
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2); \
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1)); \
    const float inv0 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 0u] * p.inv_k + p.eps); \
    const float inv1 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 1u] * p.inv_k + p.eps); \
    pretiled_qmv_affine_kl32_m2_impl<BITS, 16u, true, false>(                   \
        hidden, cs, hs, sbs, ys, ns, p.k, tgl, sg, lane, rms_w, inv0, inv1, p.w_off); \
  }

OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q4_r2_norm_m2, 4u)
OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q5_r2_norm_m2, 5u)
OMLX_M5_DEF_AFFINE_CONCAT3_HOMO_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q6_r2_norm_m2, 6u)

#define OMLX_M5_DEF_AFFINE_CONCAT3_NORM_M2_SPEC(NAME, DISPATCH_FN)              \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *c0 [[buffer(3)]],                                     \
      device const uchar *h0 [[buffer(4)]],                                     \
      device const uint *sb0 [[buffer(5)]],                                     \
      device const uchar *c1 [[buffer(6)]],                                     \
      device const uchar *h1 [[buffer(7)]],                                     \
      device const uint *sb1 [[buffer(8)]],                                     \
      device const uchar *c2 [[buffer(9)]],                                     \
      device const uchar *h2 [[buffer(10)]],                                    \
      device const uint *sb2 [[buffer(11)]],                                    \
      device bfloat *y0 [[buffer(12)]],                                         \
      device bfloat *y1 [[buffer(13)]],                                         \
      device bfloat *y2 [[buffer(14)]],                                         \
      constant PretiledQmvFusedNormParams &p [[buffer(15)]],                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    const uint b0 = p.bits & 0xFFu;                                             \
    const uint b1_r = (p.bits >> 8u) & 0xFFu;                                   \
    const uint b2_r = (p.bits >> 16u) & 0xFFu;                                  \
    const uint b1 = (b1_r != 0u) ? b1_r : b0;                                   \
    const uint b2 = (b2_r != 0u) ? b2_r : b0;                                   \
    const uint tgs0 = (p.n0 + 3u) / 4u;                                         \
    const uint tgs1 = (p.n1 + 3u) / 4u;                                         \
    device const uchar *cs = (tg.x < tgs0) ? c0 : ((tg.x < tgs0 + tgs1) ? c1 : c2); \
    device const uchar *hs = (tg.x < tgs0) ? h0 : ((tg.x < tgs0 + tgs1) ? h1 : h2); \
    device const uint *sbs = (tg.x < tgs0) ? sb0 : ((tg.x < tgs0 + tgs1) ? sb1 : sb2); \
    device bfloat *ys = (tg.x < tgs0) ? y0 : ((tg.x < tgs0 + tgs1) ? y1 : y2);  \
    const uint ns = (tg.x < tgs0) ? p.n0 : ((tg.x < tgs0 + tgs1) ? p.n1 : p.n2); \
    const uint bs = (tg.x < tgs0) ? b0 : ((tg.x < tgs0 + tgs1) ? b1 : b2);      \
    const uint tgl = (tg.x < tgs0) ? tg.x : ((tg.x < tgs0 + tgs1) ? (tg.x - tgs0) : (tg.x - tgs0 - tgs1)); \
    const float inv0 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 0u] * p.inv_k + p.eps); \
    const float inv1 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 1u] * p.inv_k + p.eps); \
    DISPATCH_FN<true>(hidden, cs, hs, sbs, ys, ns, p.k, bs, tgl, sg, lane, rms_w, inv0, inv1, p.w_off); \
  }

OMLX_M5_DEF_AFFINE_CONCAT3_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_concat3_q456_r2_norm_m2, pretiled_qmv_affine_kl32_m2_bn16_q456_dispatch)
OMLX_M5_DEF_AFFINE_CONCAT3_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_concat3_r2_norm_m2, pretiled_qmv_affine_kl32_m2_bn16_dispatch)

#define OMLX_M5_DEF_AFFINE_CONCAT4_NORM_M2_SPEC(NAME, DISPATCH_FN)              \
  [[max_total_threads_per_threadgroup(64)]]                                     \
  kernel void NAME(                                                             \
      device const bfloat *hidden [[buffer(0)]],                                \
      device const bfloat *rms_w [[buffer(1)]],                                 \
      device const float *rms_sq_buf [[buffer(2)]],                             \
      device const uchar *c0 [[buffer(3)]],                                     \
      device const uchar *h0 [[buffer(4)]],                                     \
      device const uint *sb0 [[buffer(5)]],                                     \
      device const uchar *c1 [[buffer(6)]],                                     \
      device const uchar *h1 [[buffer(7)]],                                     \
      device const uint *sb1 [[buffer(8)]],                                     \
      device const uchar *c2 [[buffer(9)]],                                     \
      device const uchar *h2 [[buffer(10)]],                                    \
      device const uint *sb2 [[buffer(11)]],                                    \
      device const uchar *c3 [[buffer(12)]],                                    \
      device const uchar *h3 [[buffer(13)]],                                    \
      device const uint *sb3 [[buffer(14)]],                                    \
      device bfloat *y0 [[buffer(15)]],                                         \
      device bfloat *y1 [[buffer(16)]],                                         \
      device bfloat *y2 [[buffer(17)]],                                         \
      device bfloat *y3 [[buffer(18)]],                                         \
      constant PretiledQmvFusedNormParams &p [[buffer(19)]],                    \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                               \
      uint3 tg [[threadgroup_position_in_grid]]) {                              \
    const uint b0 = p.bits & 0xFFu;                                             \
    const uint b1_r = (p.bits >> 8u) & 0xFFu;                                   \
    const uint b2_r = (p.bits >> 16u) & 0xFFu;                                  \
    const uint b3_r = (p.bits >> 24u) & 0xFFu;                                  \
    const uint b1 = (b1_r != 0u) ? b1_r : b0;                                   \
    const uint b2 = (b2_r != 0u) ? b2_r : b0;                                   \
    const uint b3 = (b3_r != 0u) ? b3_r : b0;                                   \
    const uint tgs0 = (p.n0 + 3u) / 4u;                                         \
    const uint tgs1 = (p.n1 + 3u) / 4u;                                         \
    const uint tgs2 = (p.n2 + 3u) / 4u;                                         \
    device const uchar *cs = c3;                                                \
    device const uchar *hs = h3;                                                \
    device const uint *sbs = sb3;                                               \
    device bfloat *ys = y3;                                                     \
    uint ns = p.n3;                                                             \
    uint bs = b3;                                                               \
    uint tgl = tg.x - tgs0 - tgs1 - tgs2;                                       \
    if (tg.x < tgs0) { cs = c0; hs = h0; sbs = sb0; ys = y0; ns = p.n0; bs = b0; tgl = tg.x; } \
    else if (tg.x < tgs0 + tgs1) { cs = c1; hs = h1; sbs = sb1; ys = y1; ns = p.n1; bs = b1; tgl = tg.x - tgs0; } \
    else if (tg.x < tgs0 + tgs1 + tgs2) { cs = c2; hs = h2; sbs = sb2; ys = y2; ns = p.n2; bs = b2; tgl = tg.x - tgs0 - tgs1; } \
    const float inv0 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 0u] * p.inv_k + p.eps); \
    const float inv1 = rsqrt(rms_sq_buf[p.sq_slot * 2u + 1u] * p.inv_k + p.eps); \
    DISPATCH_FN<true>(hidden, cs, hs, sbs, ys, ns, p.k, bs, tgl, sg, lane, rms_w, inv0, inv1, p.w_off); \
  }

OMLX_M5_DEF_AFFINE_CONCAT4_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_concat4_q456_r2_norm_m2, pretiled_qmv_affine_kl32_m2_bn16_q456_dispatch)
OMLX_M5_DEF_AFFINE_CONCAT4_NORM_M2_SPEC(omlx_m5_pretiled_qmv_affine_concat4_r2_norm_m2, pretiled_qmv_affine_kl32_m2_bn16_dispatch)
