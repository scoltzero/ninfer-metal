#include "fp4_codec.h"
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

struct QmmBroadParams {
  uint output_features;
  uint input_features;
  uint padded_input_features;
  uint tokens;
};

inline float broad_decode_e2m1(uint code) { return omlx_m5_decode_fp4(code); }

inline float broad_decode_e8m0(uchar code) {
  const uint bits = code == 0 ? 0x00400000u : uint(code) << 23;
  return as_type<float>(bits);
}

template <uint Bits>
inline uint broad_affine_code(device const uchar *base,
                              device const uchar *high, uint column) {
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

template <uint GroupSize>
inline void
broad_group_sum(device const bfloat *input, device float *group_sums,
                constant QmmBroadParams &params, uint lane, uint token_group) {
  const uint groups = params.input_features / GroupSize;
  const uint token = token_group / groups;
  if (token >= params.tokens)
    return;
  const uint group = token_group - token * groups;
  const uint begin = group * GroupSize;
  float partial = 0.0f;
  for (uint column = begin + lane; column < begin + GroupSize; column += 32) {
    partial += float(input[token * params.input_features + column]);
  }
  partial = simd_sum(partial);
  if (lane == 0) {
    group_sums[token_group] = partial;
  }
}

template <uint Bits, uint GroupSize>
inline void
affine_broad_t16(device const bfloat *input, device const uchar *base_codes,
                 device const uchar *high_codes, device const bfloat *scales,
                 device const bfloat *biases, device const float *group_sums,
                 device bfloat *output, constant QmmBroadParams &params,
                 threadgroup bfloat *shared_x, uint thread_index, uint lane,
                 uint simdgroup, uint row_block, uint token_block) {
  constexpr uint tile_t = 16;
  constexpr uint tile_k = 128;
  constexpr uint rows_per_threadgroup = 4;
  const uint row = row_block * rows_per_threadgroup + simdgroup;
  const uint token_begin = token_block * tile_t;
  if (token_begin >= params.tokens)
    return;
  const uint live_tokens = min(tile_t, params.tokens - token_begin);
  const bool valid_row = row < params.output_features;
  const uint base_row_bytes = Bits == 8 ? params.padded_input_features
                                        : params.padded_input_features / 2;
  const uint high_row_bytes =
      Bits == 5 ? params.padded_input_features / 8
                : (Bits == 6 ? params.padded_input_features / 4 : 0);
  const uint groups_per_row = params.padded_input_features / GroupSize;
  const uint logical_groups = params.input_features / GroupSize;
  device const uchar *base =
      valid_row ? base_codes + row * base_row_bytes : base_codes;
  device const uchar *high =
      valid_row ? high_codes + row * high_row_bytes : high_codes;
  device const bfloat *row_scales =
      valid_row ? scales + row * groups_per_row : scales;
  device const bfloat *row_biases =
      valid_row ? biases + row * groups_per_row : biases;
  float result[tile_t];
  for (uint token = 0; token < tile_t; ++token) {
    result[token] = 0.0f;
  }

  for (uint tile_begin = 0; tile_begin < params.input_features;
       tile_begin += tile_k) {
    for (uint token = 0; token < tile_t; ++token) {
      const uint column = tile_begin + thread_index;
      shared_x[token * tile_k + thread_index] =
          token < live_tokens && column < params.input_features
              ? input[(token_begin + token) * params.input_features + column]
              : bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid_row) {
      constexpr uint groups_per_tile = tile_k / GroupSize;
      for (uint local_group = 0; local_group < groups_per_tile; ++local_group) {
        const uint group = tile_begin / GroupSize + local_group;
        if (group >= logical_groups) {
          continue;
        }
        const float scale = lane == 0 ? float(row_scales[group]) : 0.0f;
        const float bias = lane == 0 ? float(row_biases[group]) : 0.0f;
        float qdot[tile_t];
        for (uint token = 0; token < tile_t; ++token) {
          qdot[token] = 0.0f;
        }
        for (uint local = lane; local < GroupSize; local += 32) {
          const uint column = tile_begin + local_group * GroupSize + local;
          const float q = float(broad_affine_code<Bits>(base, high, column));
          for (uint token = 0; token < live_tokens; ++token) {
            qdot[token] = fma(
                float(
                    shared_x[token * tile_k + local_group * GroupSize + local]),
                q, qdot[token]);
          }
        }
        for (uint token = 0; token < live_tokens; ++token) {
          qdot[token] = simd_sum(qdot[token]);
          if (lane == 0) {
            result[token] = fma(scale, qdot[token], result[token]);
            result[token] =
                fma(bias,
                    group_sums[(token_begin + token) * logical_groups + group],
                    result[token]);
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (valid_row && lane == 0) {
    for (uint token = 0; token < live_tokens; ++token) {
      output[(token_begin + token) * params.output_features + row] =
          bfloat(result[token]);
    }
  }
}

inline void mxfp4_w4a16_broad_t16(device const bfloat *input,
                                  device const uchar *weight_codes,
                                  device const uchar *weight_scales,
                                  device bfloat *output,
                                  constant QmmBroadParams &params,
                                  threadgroup bfloat *shared_x,
                                  uint thread_index, uint lane, uint simdgroup,
                                  uint row_block, uint token_block) {
  constexpr uint tile_t = 16;
  constexpr uint tile_k = 128;
  constexpr uint group_size = 32;
  constexpr uint rows_per_threadgroup = 4;
  const uint row = row_block * rows_per_threadgroup + simdgroup;
  const uint token_begin = token_block * tile_t;
  if (token_begin >= params.tokens)
    return;
  const uint live_tokens = min(tile_t, params.tokens - token_begin);
  const bool valid_row = row < params.output_features;
  const uint code_row_bytes = params.padded_input_features / 2;
  const uint groups_per_row = params.padded_input_features / group_size;
  const uint logical_groups = params.input_features / group_size;
  device const uchar *row_codes =
      valid_row ? weight_codes + row * code_row_bytes : weight_codes;
  device const uchar *row_scales =
      valid_row ? weight_scales + row * groups_per_row : weight_scales;
  float result[tile_t];
  for (uint token = 0; token < tile_t; ++token) {
    result[token] = 0.0f;
  }
  for (uint tile_begin = 0; tile_begin < params.input_features;
       tile_begin += tile_k) {
    for (uint token = 0; token < tile_t; ++token) {
      const uint column = tile_begin + thread_index;
      shared_x[token * tile_k + thread_index] =
          token < live_tokens && column < params.input_features
              ? input[(token_begin + token) * params.input_features + column]
              : bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid_row) {
      for (uint local_group = 0; local_group < tile_k / group_size;
           ++local_group) {
        const uint group = tile_begin / group_size + local_group;
        if (group >= logical_groups) {
          continue;
        }
        const uint column = tile_begin + local_group * group_size + lane;
        const uchar packed = row_codes[column / 2];
        const uint code = (uint(packed) >> ((column & 1u) * 4u)) & 0x0fu;
        const float weight =
            broad_decode_e2m1(code) * broad_decode_e8m0(row_scales[group]);
        for (uint token = 0; token < live_tokens; ++token) {
          float partial =
              float(
                  shared_x[token * tile_k + local_group * group_size + lane]) *
              weight;
          partial = simd_sum(partial);
          if (lane == 0) {
            result[token] += partial;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (valid_row && lane == 0) {
    for (uint token = 0; token < live_tokens; ++token) {
      output[(token_begin + token) * params.output_features + row] =
          bfloat(result[token]);
    }
  }
}

inline void mxfp4_w4a4_broad_t16(
    device const uchar *activation_codes, device const uchar *activation_scales,
    device const uchar *weight_codes, device const uchar *weight_scales,
    device bfloat *output, constant QmmBroadParams &params, uint lane,
    uint simdgroup, uint row_block, uint token_block) {
  constexpr uint tile_t = 16;
  constexpr uint group_size = 32;
  constexpr uint rows_per_threadgroup = 4;
  const uint row = row_block * rows_per_threadgroup + simdgroup;
  if (row >= params.output_features) {
    return;
  }
  const uint token_begin = token_block * tile_t;
  if (token_begin >= params.tokens)
    return;
  const uint live_tokens = min(tile_t, params.tokens - token_begin);
  const uint weight_code_row_bytes = params.padded_input_features / 2;
  const uint weight_groups_per_row = params.padded_input_features / group_size;
  const uint activation_code_row_bytes = params.input_features / 2;
  const uint logical_groups = params.input_features / group_size;
  device const uchar *row_codes = weight_codes + row * weight_code_row_bytes;
  device const uchar *row_scales = weight_scales + row * weight_groups_per_row;
  float result[tile_t];
  for (uint token = 0; token < tile_t; ++token) {
    result[token] = 0.0f;
  }
  for (uint group = 0; group < logical_groups; ++group) {
    const uint column = group * group_size + lane;
    const uchar packed = row_codes[column / 2];
    const uint code = (uint(packed) >> ((column & 1u) * 4u)) & 0x0fu;
    const float weight_value = broad_decode_e2m1(code);
    const float weight_scale = broad_decode_e8m0(row_scales[group]);
    for (uint token = 0; token < live_tokens; ++token) {
      const uint absolute_token = token_begin + token;
      const uchar activation_packed =
          activation_codes[absolute_token * activation_code_row_bytes +
                           column / 2];
      const uint activation_code =
          (uint(activation_packed) >> ((column & 1u) * 4u)) & 0x0fu;
      const float scale =
          weight_scale *
          broad_decode_e8m0(
              activation_scales[absolute_token * logical_groups + group]);
      float partial = weight_value * broad_decode_e2m1(activation_code);
      partial = simd_sum(partial);
      if (lane == 0) {
        result[token] = fma(scale, partial, result[token]);
      }
    }
  }
  if (lane == 0) {
    for (uint token = 0; token < live_tokens; ++token) {
      output[(token_begin + token) * params.output_features + row] =
          bfloat(result[token]);
    }
  }
}

kernel void omlx_m5_broad_group_sum_g64(device const bfloat *input
                                        [[buffer(0)]],
                                        device float *group_sums [[buffer(1)]],
                                        constant QmmBroadParams &params
                                        [[buffer(2)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint3 tg_position
                                        [[threadgroup_position_in_grid]]) {
  broad_group_sum<64>(input, group_sums, params, lane, tg_position.x);
}

kernel void
omlx_m5_broad_group_sum_g128(device const bfloat *input [[buffer(0)]],
                             device float *group_sums [[buffer(1)]],
                             constant QmmBroadParams &params [[buffer(2)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint3 tg_position
                             [[threadgroup_position_in_grid]]) {
  broad_group_sum<128>(input, group_sums, params, lane, tg_position.x);
}

#define DEFINE_AFFINE_BROAD(bits, group_size)                                  \
  kernel void omlx_m5_oq_q##bits##_g##group_size##_broad_t16(                  \
      device const bfloat *input [[buffer(0)]],                                \
      device const uchar *base_codes [[buffer(1)]],                            \
      device const uchar *high_codes [[buffer(2)]],                            \
      device const bfloat *scales [[buffer(3)]],                               \
      device const bfloat *biases [[buffer(4)]],                               \
      device const float *group_sums [[buffer(5)]],                            \
      device bfloat *output [[buffer(6)]],                                     \
      constant QmmBroadParams &params [[buffer(7)]],                           \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simdgroup [[simdgroup_index_in_threadgroup]],                       \
      uint3 tg_position [[threadgroup_position_in_grid]]) {                    \
    threadgroup bfloat shared_x[16 * 128];                                     \
    affine_broad_t16<bits, group_size>(                                        \
        input, base_codes, high_codes, scales, biases, group_sums, output,     \
        params, shared_x, thread_index, lane, simdgroup, tg_position.x,        \
        tg_position.y);                                                        \
  }

DEFINE_AFFINE_BROAD(4, 64)
DEFINE_AFFINE_BROAD(4, 128)
DEFINE_AFFINE_BROAD(5, 64)
DEFINE_AFFINE_BROAD(5, 128)
DEFINE_AFFINE_BROAD(6, 64)
DEFINE_AFFINE_BROAD(6, 128)
DEFINE_AFFINE_BROAD(8, 64)
DEFINE_AFFINE_BROAD(8, 128)

kernel void
omlx_m5_mxfp4_w4a16_broad_t16(device const bfloat *input [[buffer(0)]],
                              device const uchar *weight_codes [[buffer(1)]],
                              device const uchar *weight_scales [[buffer(2)]],
                              device bfloat *output [[buffer(3)]],
                              constant QmmBroadParams &params [[buffer(4)]],
                              uint thread_index [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simdgroup [[simdgroup_index_in_threadgroup]],
                              uint3 tg_position
                              [[threadgroup_position_in_grid]]) {
  threadgroup bfloat shared_x[16 * 128];
  mxfp4_w4a16_broad_t16(input, weight_codes, weight_scales, output, params,
                        shared_x, thread_index, lane, simdgroup, tg_position.x,
                        tg_position.y);
}

kernel void omlx_m5_mxfp4_w4a4_broad_t16(
    device const uchar *activation_codes [[buffer(0)]],
    device const uchar *activation_scales [[buffer(1)]],
    device const uchar *weight_codes [[buffer(2)]],
    device const uchar *weight_scales [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant QmmBroadParams &params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  mxfp4_w4a4_broad_t16(activation_codes, activation_scales, weight_codes,
                       weight_scales, output, params, lane, simdgroup,
                       tg_position.x, tg_position.y);
}
