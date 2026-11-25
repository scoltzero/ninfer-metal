#include "fp4_codec.h"
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

struct Mxfp4SmallTParams {
  uint output_features;
  uint input_features;
  uint padded_input_features;
  uint tokens;
};

inline float decode_e2m1(uint code) { return omlx_m5_decode_fp4(code); }

inline float decode_e8m0(uchar code) {
  const uint bits = code == 0 ? 0x00400000u : uint(code) << 23;
  return as_type<float>(bits);
}

inline uchar encode_e8m0_round_up(float value) {
  if (!(value > 0.0f)) {
    return 0;
  }
  const int exponent = clamp(int(ceil(log2(value))), -127, 127);
  return uchar(exponent + 127);
}

inline uchar encode_e2m1(float value) {
  const uchar sign = signbit(value) ? 8u : 0u;
  const float magnitude = abs(value);
  uchar code = 0;
  if (magnitude > 5.0f) {
    code = 7;
  } else if (magnitude >= 3.5f) {
    code = 6;
  } else if (magnitude > 2.5f) {
    code = 5;
  } else if (magnitude >= 1.75f) {
    code = 4;
  } else if (magnitude > 1.25f) {
    code = 3;
  } else if (magnitude >= 0.75f) {
    code = 2;
  } else if (magnitude > 0.25f) {
    code = 1;
  }
  return code | sign;
}

kernel void omlx_m5_mxfp4_quantize_a4(device const bfloat *input [[buffer(0)]],
                                      device uchar *output_codes [[buffer(1)]],
                                      device uchar *output_scales [[buffer(2)]],
                                      constant Mxfp4SmallTParams &params
                                      [[buffer(3)]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint3 tg_position
                                      [[threadgroup_position_in_grid]]) {
  constexpr uint group_size = 32;
  const uint groups_per_token = params.input_features / group_size;
  if (groups_per_token == 0)
    return;
  const uint token_group = tg_position.x;
  const uint token = token_group / groups_per_token;
  if (token >= params.tokens)
    return;
  const uint group = token_group - token * groups_per_token;
  const uint column = group * group_size + lane;
  const bool valid_col = column < params.input_features;
  const float value =
      valid_col ? float(input[ulong(token) * params.input_features + column])
                : 0.0f;
  const float maximum = simd_max(abs(value));
  const uchar scale_code = encode_e8m0_round_up(maximum / 6.0f);
  const float scale = decode_e8m0(scale_code);
  const uchar code = encode_e2m1(value / scale);
  if (lane == 0) {
    output_scales[token_group] = scale_code;
  }
  const uint high = simd_shuffle_down(uint(code), 1);
  if ((lane & 1u) == 0 && valid_col) {
    const ulong packed_index =
        ulong(token) * (params.input_features / 2) + ulong(group) * 16 +
        lane / 2;
    output_codes[packed_index] = code | uchar(high << 4);
  }
}

template <uint Capacity>
inline void
mxfp4_w4a16_row(device const bfloat *input, device const uchar *weight_codes,
                device const uchar *weight_scales, device bfloat *output,
                constant Mxfp4SmallTParams &params, uint lane, uint simdgroup,
                uint threadgroup_index) {
  constexpr uint simdgroups_per_threadgroup = 4;
  constexpr uint group_size = 32;
  const uint row = threadgroup_index * simdgroups_per_threadgroup + simdgroup;
  if (row >= params.output_features) {
    return;
  }
  const ulong code_row_bytes = params.padded_input_features / 2;
  const ulong groups_per_row = params.padded_input_features / group_size;
  const uint logical_groups = params.input_features / group_size;
  device const uchar *row_codes = weight_codes + ulong(row) * code_row_bytes;
  device const uchar *row_scales = weight_scales + ulong(row) * groups_per_row;
  float result[Capacity];
  for (uint token = 0; token < Capacity; ++token) {
    result[token] = 0.0f;
  }
  for (uint group = 0; group < logical_groups; ++group) {
    const uint column = group * group_size + lane;
    const uchar packed = row_codes[column / 2];
    const uint code = (uint(packed) >> ((column & 1u) * 4u)) & 0x0fu;
    const float weight = decode_e2m1(code) * decode_e8m0(row_scales[group]);
    for (uint token = 0; token < params.tokens; ++token) {
      float partial =
          float(input[ulong(token) * params.input_features + column]) * weight;
      partial = simd_sum(partial);
      if (lane == 0) {
        result[token] += partial;
      }
    }
  }
  if (lane == 0) {
    for (uint token = 0; token < params.tokens; ++token) {
      output[ulong(token) * params.output_features + row] =
          bfloat(result[token]);
    }
  }
}

inline void mxfp4_w4a16_lane_accum_c8(device const bfloat *input,
                                      device const uchar *weight_codes,
                                      device const uchar *weight_scales,
                                      device bfloat *output,
                                      constant Mxfp4SmallTParams &params,
                                      uint lane, uint simdgroup,
                                      uint threadgroup_index) {
  const uint row = threadgroup_index * 4 + simdgroup;
  if (row >= params.output_features)
    return;
  const ulong groups_per_row = params.padded_input_features / 32;
  device const uchar *row_codes =
      weight_codes + ulong(row) * (params.padded_input_features / 2);
  device const uchar *row_scales = weight_scales + ulong(row) * groups_per_row;
  float dot[8] = {0};
  for (uint group = 0; group < params.input_features / 32; ++group) {
    const uint column = group * 32 + lane;
    const uchar packed = row_codes[column / 2];
    const uint code = (uint(packed) >> ((column & 1u) * 4u)) & 0x0fu;
    const float weight = decode_e2m1(code) * decode_e8m0(row_scales[group]);
    for (uint token = 0; token < params.tokens; ++token)
      dot[token] =
          fma(float(input[ulong(token) * params.input_features + column]),
              weight, dot[token]);
  }
  for (uint token = 0; token < params.tokens; ++token) {
    const float total = simd_sum(dot[token]);
    if (lane == 0)
      output[ulong(token) * params.output_features + row] = bfloat(total);
  }
}

template <uint Capacity>
inline void mxfp4_w4a4_row(device const uchar *activation_codes,
                           device const uchar *activation_scales,
                           device const uchar *weight_codes,
                           device const uchar *weight_scales,
                           device bfloat *output,
                           constant Mxfp4SmallTParams &params, uint lane,
                           uint simdgroup, uint threadgroup_index) {
  constexpr uint simdgroups_per_threadgroup = 4;
  constexpr uint group_size = 32;
  const uint row = threadgroup_index * simdgroups_per_threadgroup + simdgroup;
  if (row >= params.output_features) {
    return;
  }
  const ulong weight_code_row_bytes = params.padded_input_features / 2;
  const ulong weight_groups_per_row = params.padded_input_features / group_size;
  const ulong activation_code_row_bytes = params.input_features / 2;
  const uint logical_groups = params.input_features / group_size;
  device const uchar *row_codes =
      weight_codes + ulong(row) * weight_code_row_bytes;
  device const uchar *row_scales =
      weight_scales + ulong(row) * weight_groups_per_row;
  float result[Capacity];
  for (uint token = 0; token < Capacity; ++token) {
    result[token] = 0.0f;
  }
  for (uint group = 0; group < logical_groups; ++group) {
    const uint column = group * group_size + lane;
    const uchar weight_packed = row_codes[column / 2];
    const uint weight_code =
        (uint(weight_packed) >> ((column & 1u) * 4u)) & 0x0fu;
    const float weight_value = decode_e2m1(weight_code);
    const float weight_scale = decode_e8m0(row_scales[group]);
    float scale_product[Capacity];
    for (uint token = 0; token < params.tokens; ++token) {
      scale_product[token] =
          weight_scale *
          decode_e8m0(activation_scales[ulong(token) * logical_groups + group]);
    }
    for (uint token = 0; token < params.tokens; ++token) {
      const ulong activation_byte =
          ulong(token) * activation_code_row_bytes + column / 2;
      const uchar activation_packed = activation_codes[activation_byte];
      const uint activation_code =
          (uint(activation_packed) >> ((column & 1u) * 4u)) & 0x0fu;
      float partial = weight_value * decode_e2m1(activation_code);
      partial = simd_sum(partial);
      if (lane == 0) {
        result[token] = fma(scale_product[token], partial, result[token]);
      }
    }
  }
  if (lane == 0) {
    for (uint token = 0; token < params.tokens; ++token) {
      output[ulong(token) * params.output_features + row] =
          bfloat(result[token]);
    }
  }
}

#define DEFINE_W4A16(capacity)                                                 \
  kernel void omlx_m5_mxfp4_w4a16_c##capacity(                                 \
      device const bfloat *input [[buffer(0)]],                                \
      device const uchar *weight_codes [[buffer(1)]],                          \
      device const uchar *weight_scales [[buffer(2)]],                         \
      device bfloat *output [[buffer(3)]],                                     \
      constant Mxfp4SmallTParams &params [[buffer(4)]],                        \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simdgroup [[simdgroup_index_in_threadgroup]],                       \
      uint3 tg_position [[threadgroup_position_in_grid]]) {                    \
    mxfp4_w4a16_row<capacity>(input, weight_codes, weight_scales, output,      \
                              params, lane, simdgroup, tg_position.x);         \
  }

kernel void omlx_m5_mxfp4_w4a16_c8_lane_accum(
    device const bfloat *input [[buffer(0)]],
    device const uchar *weight_codes [[buffer(1)]],
    device const uchar *weight_scales [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant Mxfp4SmallTParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  mxfp4_w4a16_lane_accum_c8(input, weight_codes, weight_scales, output, params,
                            lane, simdgroup, tg_position.x);
}

#define DEFINE_W4A4(capacity)                                                  \
  kernel void omlx_m5_mxfp4_w4a4_c##capacity(                                  \
      device const uchar *activation_codes [[buffer(0)]],                      \
      device const uchar *activation_scales [[buffer(1)]],                     \
      device const uchar *weight_codes [[buffer(2)]],                          \
      device const uchar *weight_scales [[buffer(3)]],                         \
      device bfloat *output [[buffer(4)]],                                     \
      constant Mxfp4SmallTParams &params [[buffer(5)]],                        \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simdgroup [[simdgroup_index_in_threadgroup]],                       \
      uint3 tg_position [[threadgroup_position_in_grid]]) {                    \
    mxfp4_w4a4_row<capacity>(activation_codes, activation_scales,              \
                             weight_codes, weight_scales, output, params,      \
                             lane, simdgroup, tg_position.x);                  \
  }

DEFINE_W4A16(2)
DEFINE_W4A16(4)
DEFINE_W4A16(8)
DEFINE_W4A16(12)
DEFINE_W4A4(2)
DEFINE_W4A4(4)
DEFINE_W4A4(8)
DEFINE_W4A4(12)
