#include "fp4_codec.h"
// MXFP4 small-T TensorOps candidates for macOS 26.2+ compilation and
// macOS 27 target execution. Raw W4/A4 planes are consumed directly by the
// kernel; decoded tiles live only in threadgroup memory.

#if __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)

// Include order is part of MLX's Metal header contract.
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/fp_quantized_nax.h"
// clang-format on

using namespace metal;
using namespace mlx::steel;

struct Mxfp4TensorOpsParams {
  uint output_features;
  uint input_features;
  uint padded_input_features;
  uint tokens;
};

inline float omlx_m5_decode_e2m1(uint code) { return omlx_m5_decode_fp4(code); }

inline float omlx_m5_decode_e8m0(uchar code) {
  const uint bits = code == 0 ? 0x00400000u : uint(code) << 23;
  return as_type<float>(bits);
}

[[kernel]] void omlx_m5_mxfp4_w4a16_tensorops(
    const device uint32_t *weights [[buffer(0)]],
    const device uint8_t *weight_scales [[buffer(1)]],
    const device bfloat *input [[buffer(2)]],
    device bfloat *output [[buffer(3)]], const constant int &K [[buffer(4)]],
    const constant int &N [[buffer(5)]], const constant int &M [[buffer(6)]],
    const constant int &in_features [[buffer(7)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  const uint output_begin = threadgroup_position.x * 32u;
  const uint token_begin = threadgroup_position.y * 16u;
  if (token_begin >= uint(M) || output_begin >= uint(N))
    return;
  if (in_features == K) {
    constexpr int BK = 64;
    constexpr int BN = 32;
    constexpr int BK_padded = BK + 16 / sizeof(bfloat);
    threadgroup bfloat weight_tile[BN * BK_padded];
    fp_qmm_t_impl<bfloat, 32, 4, false, false, 16, BK, BN, 1, 1>(
        weights, weight_scales, nullptr, input, output, weight_tile, K, N, M,
        threadgroup_position, thread_index, simdgroup_index, simd_lane);
    return;
  }

  constexpr uint tile_m = 16;
  constexpr uint tile_n = 32;
  constexpr uint tile_k = 32;
  threadgroup bfloat activation_tile[tile_m * tile_k];
  threadgroup bfloat padded_weight_tile[tile_k * tile_n];
  device const uchar *weight_codes =
      reinterpret_cast<device const uchar *>(weights);

  NAXTile<float, 1, 2> result;
  result.clear();
  const uint logical_k = uint(in_features);
  const uint padded_k = uint(K);
  const uint logical_groups = logical_k / tile_k;
  const ulong weight_code_row_bytes = ulong(padded_k) / 2;
  const ulong weight_groups_per_row = ulong(padded_k) / tile_k;

  for (uint group = 0; group < logical_groups; ++group) {
    for (uint index = simd_lane; index < tile_m * tile_k; index += 32) {
      const uint token = index / tile_k;
      const uint local_k = index - token * tile_k;
      bfloat value = bfloat(0.0f);
      if (token_begin + token < uint(M)) {
        const uint column = group * tile_k + local_k;
        value = input[ulong(token_begin + token) * logical_k + column];
      }
      activation_tile[index] = value;
    }
    for (uint index = simd_lane; index < tile_k * tile_n; index += 32) {
      const uint local_k = index / tile_n;
      const uint local_n = index - local_k * tile_n;
      const uint row = output_begin + local_n;
      bfloat value = bfloat(0.0f);
      if (row < uint(N)) {
        const uint column = group * tile_k + local_k;
        const uchar packed =
            weight_codes[ulong(row) * weight_code_row_bytes + column / 2];
        const uint code = (uint(packed) >> ((column & 1u) * 4u)) & 0x0fu;
        const float scale = omlx_m5_decode_e8m0(
            weight_scales[ulong(row) * weight_groups_per_row + group]);
        value = bfloat(omlx_m5_decode_e2m1(code) * scale);
      }
      padded_weight_tile[index] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    NAXTile<bfloat, 1, 2> activation_fragment;
    NAXTile<bfloat, 2, 2> weight_fragment;
    activation_fragment.template load<bfloat, tile_k, 1>(activation_tile);
    weight_fragment.template load<bfloat, tile_n, 1>(padded_weight_tile);
    tile_matmad_nax(result, activation_fragment, metal::bool_constant<false>{},
                    weight_fragment, metal::bool_constant<false>{});
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const short valid_rows = short(min(tile_m, uint(M) - token_begin));
  const short valid_columns = short(min(tile_n, uint(N) - output_begin));
  result.store_safe(output + ulong(token_begin) * uint(N) + output_begin,
                    uint(N), short2(valid_columns, valid_rows));
}

[[kernel]] void omlx_m5_mxfp4_w4a4_tensorops(
    device const uchar *weight_codes [[buffer(0)]],
    device const uchar *weight_scales [[buffer(1)]],
    device const uchar *activation_codes [[buffer(2)]],
    device const uchar *activation_scales [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant Mxfp4TensorOpsParams &params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  constexpr uint tile_m = 16;
  constexpr uint tile_n = 32;
  constexpr uint tile_k = 32;
  const uint output_begin = tg_position.x * tile_n;
  const uint token_begin = tg_position.y * tile_m;
  if (token_begin >= params.tokens || output_begin >= params.output_features)
    return;
  threadgroup bfloat activation_tile[tile_m * tile_k];
  threadgroup bfloat weight_tile[tile_k * tile_n];

  NAXTile<float, 1, 2> result;
  result.clear();
  const ulong activation_code_row_bytes = params.input_features / 2;
  const ulong activation_groups_per_row = params.input_features / tile_k;
  const ulong weight_code_row_bytes = params.padded_input_features / 2;
  const ulong weight_groups_per_row = params.padded_input_features / tile_k;

  for (uint group = 0; group < uint(activation_groups_per_row); ++group) {
    for (uint index = lane; index < tile_m * tile_k; index += 32) {
      const uint token = index / tile_k;
      const uint local_k = index - token * tile_k;
      bfloat value = bfloat(0.0f);
      if (token_begin + token < params.tokens) {
        const uint column = group * tile_k + local_k;
        const uchar packed = activation_codes[ulong(token_begin + token) *
                                                  activation_code_row_bytes +
                                              column / 2];
        const uint code = (uint(packed) >> ((column & 1u) * 4u)) & 0x0fu;
        const float scale = omlx_m5_decode_e8m0(
            activation_scales[ulong(token_begin + token) *
                                  activation_groups_per_row +
                              group]);
        value = bfloat(omlx_m5_decode_e2m1(code) * scale);
      }
      activation_tile[index] = value;
    }
    for (uint index = lane; index < tile_k * tile_n; index += 32) {
      const uint local_k = index / tile_n;
      const uint local_n = index - local_k * tile_n;
      const uint row = output_begin + local_n;
      bfloat value = bfloat(0.0f);
      if (row < params.output_features) {
        const uint column = group * tile_k + local_k;
        const uchar packed =
            weight_codes[ulong(row) * weight_code_row_bytes + column / 2];
        const uint code = (uint(packed) >> ((column & 1u) * 4u)) & 0x0fu;
        const float scale = omlx_m5_decode_e8m0(
            weight_scales[ulong(row) * weight_groups_per_row + group]);
        value = bfloat(omlx_m5_decode_e2m1(code) * scale);
      }
      weight_tile[index] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    NAXTile<bfloat, 1, 2> activation_fragment;
    NAXTile<bfloat, 2, 2> weight_fragment;
    activation_fragment.template load<bfloat, tile_k, 1>(activation_tile);
    weight_fragment.template load<bfloat, tile_n, 1>(weight_tile);
    tile_matmad_nax(result, activation_fragment, metal::bool_constant<false>{},
                    weight_fragment, metal::bool_constant<false>{});
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (token_begin >= params.tokens || output_begin >= params.output_features)
    return;
  const short valid_rows = short(min(tile_m, params.tokens - token_begin));
  const short valid_columns =
      short(min(tile_n, params.output_features - output_begin));
  result.store_safe(output + ulong(token_begin) * params.output_features +
                        output_begin,
                    params.output_features, short2(valid_columns, valid_rows));
}

#else
#error "MetalPerformancePrimitives tensor operations are unavailable"
#endif
