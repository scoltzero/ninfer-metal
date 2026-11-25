#include "fp4_codec.h"
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

struct MatrixParams {
  uint n, k, padded_k, tokens;
};
template <uint Bits>
inline uint matrix_code(device const uchar *base, device const uchar *high,
                        uint row, uint col, uint padded) {
  if constexpr (Bits == 8)
    return base[ulong(row) * padded + col];
  uint low =
      (base[ulong(row) * (padded / 2) + col / 2] >> ((col & 1) * 4)) & 15;
  if constexpr (Bits == 5)
    return low |
           (((high[ulong(row) * (padded / 8) + col / 8] >> (col & 7)) & 1)
            << 4);
  if constexpr (Bits == 6)
    return low | (((high[ulong(row) * (padded / 4) + col / 4] >>
                    ((col & 3) * 2)) &
                   3)
                  << 4);
  return low;
}
template <uint Bits, uint Group>
inline float matrix_weight(device const uchar *base, device const uchar *high,
                           device const uchar *scale, device const bfloat *bias,
                           uint row, uint col, uint padded) {
  uint code = matrix_code<Bits>(base, high, row, col, padded);
  if constexpr (Bits == 0) {
    uchar s = scale[ulong(row) * (padded / 32) + col / 32];
    float factor = as_type<float>(s == 0 ? 0x00400000u : uint(s) << 23);
    return omlx_m5_decode_fp4(code) * factor;
  } else {
    device const bfloat *scales =
        reinterpret_cast<device const bfloat *>(scale);
    ulong index = ulong(row) * (padded / Group) + col / Group;
    return fma(float(code), float(scales[index]), float(bias[index]));
  }
}

// All dequantization stays inside the output tile. Float operands preserve
// represented affine values without an extra BF16/FP16 staging rounding.
template <uint Bits, uint Group>
inline void quantized_matrix(device const bfloat *x, device const uchar *base,
                             device const uchar *high,
                             device const uchar *scale,
                             device const bfloat *bias, device bfloat *out,
                             constant MatrixParams &p, threadgroup float *a,
                             threadgroup float *b, threadgroup float *c,
                             uint tid, uint sg, uint2 group) {
  uint m0 = group.y * 16, n0 = group.x * 32;
  if (m0 >= p.tokens || n0 >= p.n)
    return;
  simdgroup_float8x8 acc0(0), acc1(0);
  uint mr = (sg / 2) * 8, nr = (sg % 2) * 16;
  for (uint k0 = 0; k0 < p.k; k0 += 32) {
    for (uint i = tid; i < 16 * 32; i += 128) {
      uint row = i / 32, col = i % 32;
      a[i] = m0 + row < p.tokens && k0 + col < p.k
                 ? float(x[ulong(m0 + row) * p.k + k0 + col])
                 : 0;
    }
    for (uint i = tid; i < 32 * 32; i += 128) {
      uint col = i / 32, row = i % 32;
      b[i] = n0 + row < p.n && k0 + col < p.k
                 ? matrix_weight<Bits, Group>(base, high, scale, bias, n0 + row,
                                              k0 + col, p.padded_k)
                 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 32; j += 8) {
      simdgroup_float8x8 ax, b0, b1;
      simdgroup_load(ax, a + mr * 32 + j, 32);
      simdgroup_load(b0, b + j * 32 + nr, 32);
      simdgroup_load(b1, b + j * 32 + nr + 8, 32);
      simdgroup_multiply_accumulate(acc0, ax, b0, acc0);
      simdgroup_multiply_accumulate(acc1, ax, b1, acc1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  simdgroup_store(acc0, c + mr * 32 + nr, 32);
  simdgroup_store(acc1, c + mr * 32 + nr + 8, 32);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 16 * 32; i += 128) {
    uint row = i / 32, col = i % 32;
    if (m0 + row < p.tokens && n0 + col < p.n)
      out[ulong(m0 + row) * p.n + n0 + col] = bfloat(c[i]);
  }
}
#define DEFINE_MATRIX(BITS, GROUP)                                             \
  kernel void omlx_m5_matrix_q##BITS##_g##GROUP(                               \
      device const bfloat *x [[buffer(0)]],                                    \
      device const uchar *base [[buffer(1)]],                                  \
      device const uchar *high [[buffer(2)]],                                  \
      device const uchar *scale [[buffer(3)]],                                 \
      device const bfloat *bias [[buffer(4)]],                                 \
      device bfloat *out [[buffer(5)]],                                        \
      constant MatrixParams &p [[buffer(6)]],                                  \
      uint tid [[thread_index_in_threadgroup]],                                \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint2 group [[threadgroup_position_in_grid]]) {                          \
    threadgroup float a[16 * 32], b[32 * 32], c[16 * 32];                      \
    quantized_matrix<BITS, GROUP>(x, base, high, scale, bias, out, p, a, b, c, \
                                  tid, sg, group);                             \
  }
DEFINE_MATRIX(4, 64)
DEFINE_MATRIX(4, 128)
DEFINE_MATRIX(5, 64)
DEFINE_MATRIX(5, 128)
DEFINE_MATRIX(6, 64)
DEFINE_MATRIX(6, 128)
DEFINE_MATRIX(8, 64)
DEFINE_MATRIX(8, 128)
DEFINE_MATRIX(0, 32)
