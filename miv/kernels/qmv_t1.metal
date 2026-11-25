#include "fp4_codec.h"
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

struct QmvT1Params {
  uint output_features;
  uint input_features;
  uint padded_input_features;
  uint reserved;
};

template <uint GroupSize>
inline void activation_group_sum_t1(device const bfloat *input,
                                    device float *group_sums, uint lane,
                                    uint group) {
  float partial = 0.0f;
  const uint begin = group * GroupSize;
  for (uint column = begin + lane; column < begin + GroupSize; column += 32) {
    partial += float(input[column]);
  }
  partial = simd_sum(partial);
  if (lane == 0) {
    group_sums[group] = partial;
  }
}

template <uint Bits>
inline uint affine_code(device const uchar *base, device const uchar *high,
                        uint column) {
  if constexpr (Bits == 8) {
    return uint(base[column]);
  } else {
    const uint value = (uint(base[column / 2]) >> ((column & 1u) * 4u)) & 0x0fu;
    if constexpr (Bits == 5) {
      return value | (((uint(high[column / 8]) >> (column & 7u)) & 1u) << 4u);
    } else if constexpr (Bits == 6) {
      return value |
             (((uint(high[column / 4]) >> ((column & 3u) * 2u)) & 3u) << 4u);
    } else {
      return value;
    }
  }
}

template <uint Bits, uint GroupSize>
inline void
affine_qmv_t1(device const bfloat *input, device const uchar *base_codes,
              device const uchar *high_codes, device const bfloat *scales,
              device const bfloat *biases, device const float *group_sums,
              device bfloat *output, constant QmvT1Params &params, uint lane,
              uint simdgroup, uint tg_index) {
  constexpr uint simdgroups_per_threadgroup = 4;
  const uint row = tg_index * simdgroups_per_threadgroup + simdgroup;
  if (row >= params.output_features) {
    return;
  }
  const ulong base_row_bytes = Bits == 8 ? params.padded_input_features
                                         : params.padded_input_features / 2;
  const ulong high_row_bytes =
      Bits == 5 ? params.padded_input_features / 8
                : (Bits == 6 ? params.padded_input_features / 4 : 0);
  const ulong groups_per_row = params.padded_input_features / GroupSize;
  device const uchar *base = base_codes + ulong(row) * base_row_bytes;
  device const uchar *high = high_codes + ulong(row) * high_row_bytes;
  device const bfloat *row_scales = scales + ulong(row) * groups_per_row;
  device const bfloat *row_biases = biases + ulong(row) * groups_per_row;
  const uint logical_groups = params.input_features / GroupSize;
  float result = 0.0f;
  for (uint group = 0; group < logical_groups; ++group) {
    const uint begin = group * GroupSize;
    float qdot = 0.0f;
    for (uint column = begin + lane; column < begin + GroupSize; column += 32) {
      const float x = float(input[column]);
      qdot = fma(x, float(affine_code<Bits>(base, high, column)), qdot);
    }
    qdot = simd_sum(qdot);
    if (lane == 0) {
      result = fma(float(row_scales[group]), qdot, result);
      result = fma(float(row_biases[group]), group_sums[group], result);
    }
  }
  if (lane == 0) {
    output[row] = bfloat(result);
  }
}

// Candidate for large output matrices: accumulate lane-local group
// contributions and perform one SIMD reduction per row. The original path
// remains available until model-bound numerical and timing gates qualify this
// ordering.
template <uint Bits, uint GroupSize>
inline void affine_qmv_t1_lane_accum(
    device const bfloat *input, device const uchar *base_codes,
    device const uchar *high_codes, device const bfloat *scales,
    device const bfloat *biases, device const float *group_sums,
    device bfloat *output, constant QmvT1Params &params, uint lane,
    uint simdgroup, uint tg_index) {
  const uint row = tg_index * 4 + simdgroup;
  if (row >= params.output_features)
    return;
  const ulong base_stride = Bits == 8 ? params.padded_input_features
                                      : params.padded_input_features / 2;
  const ulong high_stride = Bits == 5   ? params.padded_input_features / 8
                            : Bits == 6 ? params.padded_input_features / 4
                                        : 0;
  device const uchar *base = base_codes + ulong(row) * base_stride;
  device const uchar *high = high_codes + ulong(row) * high_stride;
  const ulong groups_per_row = params.padded_input_features / GroupSize;
  device const bfloat *row_scales = scales + ulong(row) * groups_per_row;
  device const bfloat *row_biases = biases + ulong(row) * groups_per_row;
  float dot = 0.0f, correction = 0.0f;
  for (uint group = 0; group < params.input_features / GroupSize; ++group) {
    float local = 0.0f;
    const uint begin = group * GroupSize;
    for (uint column = begin + lane; column < begin + GroupSize; column += 32)
      local = fma(float(input[column]),
                  float(affine_code<Bits>(base, high, column)), local);
    dot = fma(float(row_scales[group]), local, dot);
    if (lane == 0)
      correction = fma(float(row_biases[group]), group_sums[group], correction);
  }
  dot = simd_sum(dot);
  if (lane == 0)
    output[row] = bfloat(dot + correction);
}

inline float decode_e2m1(uint code) { return omlx_m5_decode_fp4(code); }

inline float decode_e8m0(uchar code) {
  const uint bits = code == 0 ? 0x00400000u : uint(code) << 23;
  return as_type<float>(bits);
}

inline void mxfp4_qmv_t1(device const bfloat *input, device const uchar *codes,
                         device const uchar *scales, device bfloat *output,
                         constant QmvT1Params &params, uint lane,
                         uint simdgroup, uint tg_index) {
  constexpr uint simdgroups_per_threadgroup = 4;
  constexpr uint group_size = 32;
  const uint row = tg_index * simdgroups_per_threadgroup + simdgroup;
  if (row >= params.output_features) {
    return;
  }
  const ulong code_row_bytes = params.padded_input_features / 2;
  const ulong groups_per_row = params.padded_input_features / group_size;
  device const uchar *row_codes = codes + ulong(row) * code_row_bytes;
  device const uchar *row_scales = scales + ulong(row) * groups_per_row;
  const uint logical_groups = params.input_features / group_size;
  float result = 0.0f;
  for (uint group = 0; group < logical_groups; ++group) {
    const uint column = group * group_size + lane;
    const uchar packed = row_codes[column / 2];
    const uint code = (uint(packed) >> ((column & 1u) * 4u)) & 0x0fu;
    float partial = float(input[column]) * decode_e2m1(code);
    partial = simd_sum(partial);
    if (lane == 0) {
      result = fma(decode_e8m0(row_scales[group]), partial, result);
    }
  }
  if (lane == 0) {
    output[row] = bfloat(result);
  }
}

inline void mxfp4_qmv_t1_lane_accum(device const bfloat *input,
                                    device const uchar *codes,
                                    device const uchar *scales,
                                    device bfloat *output,
                                    constant QmvT1Params &params, uint lane,
                                    uint simdgroup, uint tg_index) {
  const uint row = tg_index * 4 + simdgroup;
  if (row >= params.output_features)
    return;
  const ulong groups_per_row = params.padded_input_features / 32;
  device const uchar *row_codes =
      codes + ulong(row) * (params.padded_input_features / 2);
  device const uchar *row_scales = scales + ulong(row) * groups_per_row;
  float dot = 0.0f;
  for (uint group = 0; group < params.input_features / 32; ++group) {
    const uint column = group * 32 + lane;
    const uchar packed = row_codes[column / 2];
    const uint code = (uint(packed) >> ((column & 1u) * 4u)) & 0x0fu;
    dot = fma(float(input[column]),
              decode_e2m1(code) * decode_e8m0(row_scales[group]), dot);
  }
  dot = simd_sum(dot);
  if (lane == 0)
    output[row] = bfloat(dot);
}

inline void bf16_qmv_t1(device const bfloat *input,
                        device const bfloat *weights, device bfloat *output,
                        constant QmvT1Params &params, uint lane, uint simdgroup,
                        uint tg_index) {
  constexpr uint simdgroups_per_threadgroup = 4;
  const uint row = tg_index * simdgroups_per_threadgroup + simdgroup;
  if (row >= params.output_features) {
    return;
  }
  const ulong row_stride = ulong(
      params.padded_input_features ? params.padded_input_features
                                   : params.input_features);
  device const bfloat *weight = weights + ulong(row) * row_stride;
  float result = 0.0f;
  for (uint column = lane; column < params.input_features; column += 32) {
    result = fma(float(input[column]), float(weight[column]), result);
  }
  result = simd_sum(result);
  if (lane == 0) {
    output[row] = bfloat(result);
  }
}

#define DEFINE_AFFINE_QMV(bits, group_size)                                    \
  kernel void omlx_m5_oq_q##bits##_g##group_size##_qmv_t1(                     \
      device const bfloat *input [[buffer(0)]],                                \
      device const uchar *base_codes [[buffer(1)]],                            \
      device const uchar *high_codes [[buffer(2)]],                            \
      device const bfloat *scales [[buffer(3)]],                               \
      device const bfloat *biases [[buffer(4)]],                               \
      device const float *group_sums [[buffer(5)]],                            \
      device bfloat *output [[buffer(6)]],                                     \
      constant QmvT1Params &params [[buffer(7)]],                              \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simdgroup [[simdgroup_index_in_threadgroup]],                       \
      uint3 tg_position [[threadgroup_position_in_grid]]) {                    \
    affine_qmv_t1<bits, group_size>(input, base_codes, high_codes, scales,     \
                                    biases, group_sums, output, params, lane,  \
                                    simdgroup, tg_position.x);                 \
  }

#define DEFINE_AFFINE_QMV_LANE_ACCUM(bits, group_size)                         \
  kernel void omlx_m5_oq_q##bits##_g##group_size##_qmv_t1_lane_accum(          \
      device const bfloat *input [[buffer(0)]],                                \
      device const uchar *base_codes [[buffer(1)]],                            \
      device const uchar *high_codes [[buffer(2)]],                            \
      device const bfloat *scales [[buffer(3)]],                               \
      device const bfloat *biases [[buffer(4)]],                               \
      device const float *group_sums [[buffer(5)]],                            \
      device bfloat *output [[buffer(6)]],                                     \
      constant QmvT1Params &params [[buffer(7)]],                              \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simdgroup [[simdgroup_index_in_threadgroup]],                       \
      uint3 tg_position [[threadgroup_position_in_grid]]) {                    \
    affine_qmv_t1_lane_accum<bits, group_size>(                                \
        input, base_codes, high_codes, scales, biases, group_sums, output,     \
        params, lane, simdgroup, tg_position.x);                               \
  }

kernel void
omlx_m5_activation_group_sum_g64(device const bfloat *input [[buffer(0)]],
                                 device float *group_sums [[buffer(1)]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint3 tg_position
                                 [[threadgroup_position_in_grid]]) {
  activation_group_sum_t1<64>(input, group_sums, lane, tg_position.x);
}

kernel void
omlx_m5_activation_group_sum_g128(device const bfloat *input [[buffer(0)]],
                                  device float *group_sums [[buffer(1)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint3 tg_position
                                  [[threadgroup_position_in_grid]]) {
  activation_group_sum_t1<128>(input, group_sums, lane, tg_position.x);
}

DEFINE_AFFINE_QMV(4, 64)
DEFINE_AFFINE_QMV(4, 128)
DEFINE_AFFINE_QMV(5, 64)
DEFINE_AFFINE_QMV(5, 128)
DEFINE_AFFINE_QMV(6, 64)
DEFINE_AFFINE_QMV(6, 128)
DEFINE_AFFINE_QMV(8, 64)
DEFINE_AFFINE_QMV(8, 128)
DEFINE_AFFINE_QMV_LANE_ACCUM(4, 64)
DEFINE_AFFINE_QMV_LANE_ACCUM(4, 128)
DEFINE_AFFINE_QMV_LANE_ACCUM(5, 64)
DEFINE_AFFINE_QMV_LANE_ACCUM(5, 128)
DEFINE_AFFINE_QMV_LANE_ACCUM(6, 64)
DEFINE_AFFINE_QMV_LANE_ACCUM(6, 128)
DEFINE_AFFINE_QMV_LANE_ACCUM(8, 64)
DEFINE_AFFINE_QMV_LANE_ACCUM(8, 128)

kernel void omlx_m5_mxfp4_qmv_t1(device const bfloat *input [[buffer(0)]],
                                 device const uchar *codes [[buffer(1)]],
                                 device const uchar *scales [[buffer(2)]],
                                 device bfloat *output [[buffer(3)]],
                                 constant QmvT1Params &params [[buffer(4)]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint simdgroup
                                 [[simdgroup_index_in_threadgroup]],
                                 uint3 tg_position
                                 [[threadgroup_position_in_grid]]) {
  mxfp4_qmv_t1(input, codes, scales, output, params, lane, simdgroup,
               tg_position.x);
}

kernel void omlx_m5_mxfp4_qmv_t1_lane_accum(
    device const bfloat *input [[buffer(0)]],
    device const uchar *codes [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant QmvT1Params &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  mxfp4_qmv_t1_lane_accum(input, codes, scales, output, params, lane, simdgroup,
                          tg_position.x);
}

kernel void omlx_m5_bf16_qmv_t1(device const bfloat *input [[buffer(0)]],
                                device const bfloat *weights [[buffer(1)]],
                                device bfloat *output [[buffer(2)]],
                                constant QmvT1Params &params [[buffer(3)]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint simdgroup
                                [[simdgroup_index_in_threadgroup]],
                                uint3 tg_position
                                [[threadgroup_position_in_grid]]) {
  bf16_qmv_t1(input, weights, output, params, lane, simdgroup, tg_position.x);
}
