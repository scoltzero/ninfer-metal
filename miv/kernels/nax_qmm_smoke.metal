// D0 compile-only canary for the MLX Metal 4 quantized NAX interface.
// The MLX headers are MIT licensed. This kernel is not a performance result.

#if __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)

// Include order is part of MLX's Metal header contract.
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"

#define define_omlx_m5_nax_qmm(bits)                                          \
  template <typename T, const int BM, const int BK, const int BN,             \
            const int WM, const int WN>                                       \
  [[kernel]] void omlx_m5_q##bits##_affine_g64_nax(                          \
      const device uint32_t* weights [[buffer(0)]],                           \
      const device T* scales [[buffer(1)]],                                   \
      const device T* biases [[buffer(2)]],                                   \
      const device T* input [[buffer(3)]],                                    \
      device T* output [[buffer(4)]],                                         \
      const constant int& K [[buffer(5)]],                                    \
      const constant int& N [[buffer(6)]],                                    \
      const constant int& M [[buffer(7)]],                                    \
      uint3 threadgroup_position [[threadgroup_position_in_grid]],             \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint simdgroup_index [[simdgroup_index_in_threadgroup]],                \
      uint simd_lane [[thread_index_in_simdgroup]]) {                         \
    constexpr int BK_padded = BK + 16 / sizeof(T);                            \
    threadgroup T weight_tile[BN * BK_padded];                                \
    qmm_t_nax_tgp_impl<T, 64, bits, true, BM, BK, BN, WM, WN>(               \
        weights, scales, biases, input, output, weight_tile, K, N, M,         \
        threadgroup_position, thread_index, simdgroup_index, simd_lane);       \
  }

#define instantiate_omlx_m5_nax_qmm(bits, type)                               \
  instantiate_kernel(                                                         \
      "omlx_m5_q" #bits "_affine_g64_nax_" #type,                           \
      omlx_m5_q##bits##_affine_g64_nax, type, 64, 64, 64, 2, 2)

define_omlx_m5_nax_qmm(4)
define_omlx_m5_nax_qmm(5)

instantiate_omlx_m5_nax_qmm(4, float16_t);
instantiate_omlx_m5_nax_qmm(4, bfloat16_t);
instantiate_omlx_m5_nax_qmm(5, float16_t);
instantiate_omlx_m5_nax_qmm(5, bfloat16_t);
// clang-format on

#else
#error "MetalPerformancePrimitives tensor operations are unavailable"
#endif
