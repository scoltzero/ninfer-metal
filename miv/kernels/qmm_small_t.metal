#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

struct QmmSmallTParams {
  uint output_features;
  uint input_features;
  uint padded_input_features;
  uint tokens;
};

template <uint GroupSize>
inline void activation_group_sum_small_t(device const bfloat *input,
                                         device float *group_sums,
                                         constant QmmSmallTParams &params,
                                         uint lane, uint token_group) {
  const uint groups = params.input_features / GroupSize;
  const uint token = token_group / groups;
  if (token >= params.tokens)
    return;
  const uint group = token_group - token * groups;
  float partial = 0.0f;
  const uint begin = group * GroupSize;
  for (uint column = begin + lane; column < begin + GroupSize; column += 32) {
    partial += float(input[token * params.input_features + column]);
  }
  partial = simd_sum(partial);
  if (lane == 0) {
    group_sums[token_group] = partial;
  }
}

template <uint Bits>
inline uint affine_code(device const uchar *base, device const uchar *high,
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

template <uint Bits, uint GroupSize>
inline void affine_row_geometry(constant QmmSmallTParams &params, uint row,
                                thread uint &base_row_bytes,
                                thread uint &high_row_bytes,
                                thread uint &groups_per_row) {
  base_row_bytes = Bits == 8 ? params.padded_input_features
                             : params.padded_input_features / 2;
  high_row_bytes = Bits == 5
                       ? params.padded_input_features / 8
                       : (Bits == 6 ? params.padded_input_features / 4 : 0);
  groups_per_row = params.padded_input_features / GroupSize;
  (void)row;
}

template <uint Bits, uint GroupSize, uint Capacity>
inline void
affine_row_split(device const bfloat *input, device const uchar *base_codes,
                 device const uchar *high_codes, device const bfloat *scales,
                 device const bfloat *biases, device const float *group_sums,
                 device bfloat *output, constant QmmSmallTParams &params,
                 uint lane, uint simdgroup, uint threadgroup_index) {
  constexpr uint simdgroups_per_threadgroup = 4;
  const uint row = threadgroup_index * simdgroups_per_threadgroup + simdgroup;
  if (row >= params.output_features) {
    return;
  }
  uint base_row_bytes = 0;
  uint high_row_bytes = 0;
  uint groups_per_row = 0;
  affine_row_geometry<Bits, GroupSize>(params, row, base_row_bytes,
                                       high_row_bytes, groups_per_row);
  device const uchar *base = base_codes + row * base_row_bytes;
  device const uchar *high = high_codes + row * high_row_bytes;
  device const bfloat *row_scales = scales + row * groups_per_row;
  device const bfloat *row_biases = biases + row * groups_per_row;
  const uint logical_groups = params.input_features / GroupSize;
  float result[Capacity];
  for (uint token = 0; token < Capacity; ++token) {
    result[token] = 0.0f;
  }
  for (uint group = 0; group < logical_groups; ++group) {
    // The affine metadata is also loaded once per group and reused across T.
    const float scale = lane == 0 ? float(row_scales[group]) : 0.0f;
    const float bias = lane == 0 ? float(row_biases[group]) : 0.0f;
    float qdot[Capacity];
    for (uint token = 0; token < Capacity; ++token) {
      qdot[token] = 0.0f;
    }
    const uint begin = group * GroupSize;
    for (uint column = begin + lane; column < begin + GroupSize; column += 32) {
      // Decode once, then reuse the value across every live token column.
      const float q = float(affine_code<Bits>(base, high, column));
      for (uint token = 0; token < params.tokens; ++token) {
        qdot[token] = fma(float(input[token * params.input_features + column]),
                          q, qdot[token]);
      }
    }
    for (uint token = 0; token < params.tokens; ++token) {
      qdot[token] = simd_sum(qdot[token]);
      if (lane == 0) {
        result[token] = fma(scale, qdot[token], result[token]);
        result[token] = fma(bias, group_sums[token * logical_groups + group],
                            result[token]);
      }
    }
  }
  if (lane == 0) {
    for (uint token = 0; token < params.tokens; ++token) {
      output[token * params.output_features + row] = bfloat(result[token]);
    }
  }
}

// Candidate T=8 path: keep lane-local accumulators and reduce once per output
// token, rather than once per quantization group and token.
template <uint Bits, uint GroupSize>
inline void affine_row_lane_accum_c8(
    device const bfloat *input, device const uchar *base_codes,
    device const uchar *high_codes, device const bfloat *scales,
    device const bfloat *biases, device const float *group_sums,
    device bfloat *output, constant QmmSmallTParams &params, uint lane,
    uint simdgroup, uint threadgroup_index) {
  const uint row = threadgroup_index * 4 + simdgroup;
  if (row >= params.output_features)
    return;
  uint base_stride = 0, high_stride = 0, groups_per_row = 0;
  affine_row_geometry<Bits, GroupSize>(params, row, base_stride, high_stride,
                                       groups_per_row);
  device const uchar *base = base_codes + row * base_stride;
  device const uchar *high = high_codes + row * high_stride;
  device const bfloat *row_scales = scales + row * groups_per_row;
  device const bfloat *row_biases = biases + row * groups_per_row;
  const uint logical_groups = params.input_features / GroupSize;
  float dot[8] = {0}, bias[8] = {0};
  for (uint group = 0; group < logical_groups; ++group) {
    const float scale = float(row_scales[group]);
    const uint begin = group * GroupSize;
    for (uint column = begin + lane; column < begin + GroupSize; column += 32) {
      const float q = float(affine_code<Bits>(base, high, column));
      for (uint token = 0; token < params.tokens; ++token)
        dot[token] = fma(float(input[token * params.input_features + column]),
                         q * scale, dot[token]);
    }
    if (lane == 0)
      for (uint token = 0; token < params.tokens; ++token)
        bias[token] =
            fma(float(row_biases[group]),
                group_sums[token * logical_groups + group], bias[token]);
  }
  for (uint token = 0; token < params.tokens; ++token) {
    const float total = simd_sum(dot[token]);
    if (lane == 0)
      output[token * params.output_features + row] =
          bfloat(total + bias[token]);
  }
}

template <uint Bits, uint GroupSize, uint Capacity>
inline void affine_split_k_partial(
    device const bfloat *input, device const uchar *base_codes,
    device const uchar *high_codes, device const bfloat *scales,
    device const bfloat *biases, device const float *group_sums,
    device float *partials, constant QmmSmallTParams &params, uint lane,
    uint split, uint row) {
  constexpr uint split_count = 4;
  if (row >= params.output_features) {
    return;
  }
  uint base_row_bytes = 0;
  uint high_row_bytes = 0;
  uint groups_per_row = 0;
  affine_row_geometry<Bits, GroupSize>(params, row, base_row_bytes,
                                       high_row_bytes, groups_per_row);
  device const uchar *base = base_codes + row * base_row_bytes;
  device const uchar *high = high_codes + row * high_row_bytes;
  device const bfloat *row_scales = scales + row * groups_per_row;
  device const bfloat *row_biases = biases + row * groups_per_row;
  const uint logical_groups = params.input_features / GroupSize;
  float result[Capacity];
  for (uint token = 0; token < Capacity; ++token) {
    result[token] = 0.0f;
  }
  for (uint group = split; group < logical_groups; group += split_count) {
    const float scale = lane == 0 ? float(row_scales[group]) : 0.0f;
    const float bias = lane == 0 ? float(row_biases[group]) : 0.0f;
    float qdot[Capacity];
    for (uint token = 0; token < Capacity; ++token) {
      qdot[token] = 0.0f;
    }
    const uint begin = group * GroupSize;
    for (uint column = begin + lane; column < begin + GroupSize; column += 32) {
      const float q = float(affine_code<Bits>(base, high, column));
      for (uint token = 0; token < params.tokens; ++token) {
        qdot[token] = fma(float(input[token * params.input_features + column]),
                          q, qdot[token]);
      }
    }
    for (uint token = 0; token < params.tokens; ++token) {
      qdot[token] = simd_sum(qdot[token]);
      if (lane == 0) {
        result[token] = fma(scale, qdot[token], result[token]);
        result[token] = fma(bias, group_sums[token * logical_groups + group],
                            result[token]);
      }
    }
  }
  if (lane == 0) {
    const uint partial_base = (row * split_count + split) * Capacity;
    for (uint token = 0; token < params.tokens; ++token) {
      partials[partial_base + token] = result[token];
    }
  }
}

template <uint Capacity>
inline void reduce_split_k(device const float *partials, device bfloat *output,
                           constant QmmSmallTParams &params, uint index) {
  constexpr uint split_count = 4;
  const uint output_count = params.tokens * params.output_features;
  if (index >= output_count) {
    return;
  }
  const uint token = index / params.output_features;
  const uint row = index - token * params.output_features;
  float result = 0.0f;
  for (uint split = 0; split < split_count; ++split) {
    result += partials[(row * split_count + split) * Capacity + token];
  }
  output[index] = bfloat(result);
}

template <uint Bits, uint GroupSize>
inline void affine_cooperative_row_block(
    device const bfloat *input, device const uchar *base_codes,
    device const uchar *high_codes, device const bfloat *scales,
    device const bfloat *biases, device const float *group_sums,
    device bfloat *output, constant QmmSmallTParams &params,
    threadgroup bfloat *shared_x, uint thread_index, uint lane, uint simdgroup,
    uint threadgroup_index) {
  constexpr uint capacity = 12;
  constexpr uint tile_k = 128;
  constexpr uint simdgroups_per_threadgroup = 4;
  const uint row = threadgroup_index * simdgroups_per_threadgroup + simdgroup;
  const bool valid_row = row < params.output_features;
  uint base_row_bytes = 0;
  uint high_row_bytes = 0;
  uint groups_per_row = 0;
  affine_row_geometry<Bits, GroupSize>(params, row, base_row_bytes,
                                       high_row_bytes, groups_per_row);
  device const uchar *base =
      valid_row ? base_codes + row * base_row_bytes : base_codes;
  device const uchar *high =
      valid_row ? high_codes + row * high_row_bytes : high_codes;
  device const bfloat *row_scales =
      valid_row ? scales + row * groups_per_row : scales;
  device const bfloat *row_biases =
      valid_row ? biases + row * groups_per_row : biases;
  const uint logical_groups = params.input_features / GroupSize;
  float result[capacity];
  for (uint token = 0; token < capacity; ++token) {
    result[token] = 0.0f;
  }

  for (uint tile_begin = 0; tile_begin < params.input_features;
       tile_begin += tile_k) {
    for (uint token = 0; token < params.tokens; ++token) {
      const uint column = tile_begin + thread_index;
      shared_x[token * tile_k + thread_index] =
          column < params.input_features
              ? input[token * params.input_features + column]
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
        float qdot[capacity];
        for (uint token = 0; token < capacity; ++token) {
          qdot[token] = 0.0f;
        }
        for (uint local = lane; local < GroupSize; local += 32) {
          const uint column = tile_begin + local_group * GroupSize + local;
          const float q = float(affine_code<Bits>(base, high, column));
          for (uint token = 0; token < params.tokens; ++token) {
            qdot[token] = fma(
                float(
                    shared_x[token * tile_k + local_group * GroupSize + local]),
                q, qdot[token]);
          }
        }
        for (uint token = 0; token < params.tokens; ++token) {
          qdot[token] = simd_sum(qdot[token]);
          if (lane == 0) {
            result[token] = fma(scale, qdot[token], result[token]);
            result[token] =
                fma(bias, group_sums[token * logical_groups + group],
                    result[token]);
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (valid_row && lane == 0) {
    for (uint token = 0; token < params.tokens; ++token) {
      output[token * params.output_features + row] = bfloat(result[token]);
    }
  }
}

kernel void omlx_m5_activation_group_sum_small_t_g64(
    device const bfloat *input [[buffer(0)]],
    device float *group_sums [[buffer(1)]],
    constant QmmSmallTParams &params [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  activation_group_sum_small_t<64>(input, group_sums, params, lane,
                                   tg_position.x);
}

kernel void omlx_m5_activation_group_sum_small_t_g128(
    device const bfloat *input [[buffer(0)]],
    device float *group_sums [[buffer(1)]],
    constant QmmSmallTParams &params [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  activation_group_sum_small_t<128>(input, group_sums, params, lane,
                                    tg_position.x);
}

#define DEFINE_ROW(bits, group_size, capacity)                                 \
  kernel void omlx_m5_oq_q##bits##_g##group_size##_row_c##capacity(            \
      device const bfloat *input [[buffer(0)]],                                \
      device const uchar *base_codes [[buffer(1)]],                            \
      device const uchar *high_codes [[buffer(2)]],                            \
      device const bfloat *scales [[buffer(3)]],                               \
      device const bfloat *biases [[buffer(4)]],                               \
      device const float *group_sums [[buffer(5)]],                            \
      device bfloat *output [[buffer(6)]],                                     \
      constant QmmSmallTParams &params [[buffer(7)]],                          \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simdgroup [[simdgroup_index_in_threadgroup]],                       \
      uint3 tg_position [[threadgroup_position_in_grid]]) {                    \
    affine_row_split<bits, group_size, capacity>(                              \
        input, base_codes, high_codes, scales, biases, group_sums, output,     \
        params, lane, simdgroup, tg_position.x);                               \
  }

#define DEFINE_ROW_LANE_ACCUM(bits, group_size)                                \
  kernel void omlx_m5_oq_q##bits##_g##group_size##_row_c8_lane_accum(          \
      device const bfloat *input [[buffer(0)]],                                \
      device const uchar *base_codes [[buffer(1)]],                            \
      device const uchar *high_codes [[buffer(2)]],                            \
      device const bfloat *scales [[buffer(3)]],                               \
      device const bfloat *biases [[buffer(4)]],                               \
      device const float *group_sums [[buffer(5)]],                            \
      device bfloat *output [[buffer(6)]],                                     \
      constant QmmSmallTParams &params [[buffer(7)]],                          \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simdgroup [[simdgroup_index_in_threadgroup]],                       \
      uint3 tg_position [[threadgroup_position_in_grid]]) {                    \
    affine_row_lane_accum_c8<bits, group_size>(                                \
        input, base_codes, high_codes, scales, biases, group_sums, output,     \
        params, lane, simdgroup, tg_position.x);                               \
  }

#define DEFINE_SPLIT(bits, group_size, capacity)                               \
  kernel void omlx_m5_oq_q##bits##_g##group_size##_split_c##capacity(          \
      device const bfloat *input [[buffer(0)]],                                \
      device const uchar *base_codes [[buffer(1)]],                            \
      device const uchar *high_codes [[buffer(2)]],                            \
      device const bfloat *scales [[buffer(3)]],                               \
      device const bfloat *biases [[buffer(4)]],                               \
      device const float *group_sums [[buffer(5)]],                            \
      device float *partials [[buffer(6)]],                                    \
      constant QmmSmallTParams &params [[buffer(7)]],                          \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint split [[simdgroup_index_in_threadgroup]],                           \
      uint3 tg_position [[threadgroup_position_in_grid]]) {                    \
    affine_split_k_partial<bits, group_size, capacity>(                        \
        input, base_codes, high_codes, scales, biases, group_sums, partials,   \
        params, lane, split, tg_position.x);                                   \
  }

#define DEFINE_COOP(bits, group_size)                                          \
  kernel void omlx_m5_oq_q##bits##_g##group_size##_coop_c12(                   \
      device const bfloat *input [[buffer(0)]],                                \
      device const uchar *base_codes [[buffer(1)]],                            \
      device const uchar *high_codes [[buffer(2)]],                            \
      device const bfloat *scales [[buffer(3)]],                               \
      device const bfloat *biases [[buffer(4)]],                               \
      device const float *group_sums [[buffer(5)]],                            \
      device bfloat *output [[buffer(6)]],                                     \
      constant QmmSmallTParams &params [[buffer(7)]],                          \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simdgroup [[simdgroup_index_in_threadgroup]],                       \
      uint3 tg_position [[threadgroup_position_in_grid]]) {                    \
    threadgroup bfloat shared_x[12 * 128];                                     \
    affine_cooperative_row_block<bits, group_size>(                            \
        input, base_codes, high_codes, scales, biases, group_sums, output,     \
        params, shared_x, thread_index, lane, simdgroup, tg_position.x);       \
  }

#define DEFINE_ROWS(bits, group_size)                                          \
  DEFINE_ROW(bits, group_size, 2)                                              \
  DEFINE_ROW(bits, group_size, 4)                                              \
  DEFINE_ROW(bits, group_size, 8)

#define DEFINE_SPLITS(bits, group_size)                                        \
  DEFINE_SPLIT(bits, group_size, 2)                                            \
  DEFINE_SPLIT(bits, group_size, 4)                                            \
  DEFINE_SPLIT(bits, group_size, 8)

DEFINE_ROWS(4, 64)
DEFINE_ROWS(4, 128)
DEFINE_ROWS(5, 64)
DEFINE_ROWS(5, 128)
DEFINE_ROWS(6, 64)
DEFINE_ROWS(6, 128)
DEFINE_ROWS(8, 64)
DEFINE_ROWS(8, 128)

DEFINE_ROW_LANE_ACCUM(4, 64)
DEFINE_ROW_LANE_ACCUM(4, 128)
DEFINE_ROW_LANE_ACCUM(5, 64)
DEFINE_ROW_LANE_ACCUM(5, 128)
DEFINE_ROW_LANE_ACCUM(6, 64)
DEFINE_ROW_LANE_ACCUM(6, 128)
DEFINE_ROW_LANE_ACCUM(8, 64)
DEFINE_ROW_LANE_ACCUM(8, 128)

DEFINE_SPLITS(5, 64)
DEFINE_SPLITS(5, 128)
DEFINE_SPLITS(6, 64)
DEFINE_SPLITS(6, 128)

DEFINE_COOP(4, 64)
DEFINE_COOP(4, 128)
DEFINE_COOP(5, 64)
DEFINE_COOP(5, 128)
DEFINE_COOP(6, 64)
DEFINE_COOP(6, 128)
DEFINE_COOP(8, 64)
DEFINE_COOP(8, 128)

#define DEFINE_REDUCE(capacity)                                                \
  kernel void omlx_m5_oq_split_reduce_c##capacity(                             \
      device const float *partials [[buffer(0)]],                              \
      device bfloat *output [[buffer(1)]],                                     \
      constant QmmSmallTParams &params [[buffer(2)]],                          \
      uint3 position [[thread_position_in_grid]]) {                            \
    reduce_split_k<capacity>(partials, output, params, position.x);            \
  }

DEFINE_REDUCE(2)
DEFINE_REDUCE(4)
DEFINE_REDUCE(8)
