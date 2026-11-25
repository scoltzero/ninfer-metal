#include <metal_stdlib>

using namespace metal;

template <uint Bits>
inline uint unpack_code(device const uchar *packed, uint column) {
  const uint bit_offset = column * Bits;
  const uint byte_offset = bit_offset / 8;
  const uint shift = bit_offset % 8;
  uint value = uint(packed[byte_offset]);
  // q4 can end exactly at a byte boundary; q5 may straddle two bytes.
  if (shift + Bits > 8) {
    value |= uint(packed[byte_offset + 1]) << 8;
  }
  return (value >> shift) & ((1u << Bits) - 1u);
}

template <uint Bits>
inline void reference_qmm(device const float *input,
                          device const uchar *weights,
                          device const float *scales,
                          device const float *biases, device float *output,
                          uint m, uint n, uint k, uint2 position) {
  const uint output_column = position.x;
  const uint input_row = position.y;
  if (input_row >= m || output_column >= n) {
    return;
  }
  constexpr uint group_size = 64;
  const uint groups = k / group_size;
  const uint row_bytes = (k * Bits) / 8;
  device const uchar *packed = weights + output_column * row_bytes;
  float sum = 0.0f;
  for (uint column = 0; column < k; ++column) {
    const uint group = column / group_size;
    const float weight =
        fma(scales[output_column * groups + group],
            float(unpack_code<Bits>(packed, column)),
            biases[output_column * groups + group]);
    sum = fma(input[input_row * k + column], weight, sum);
  }
  output[input_row * n + output_column] = sum;
}

kernel void omlx_m5_reference_qmm_q4(
    device const float *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *scales [[buffer(2)]],
    device const float *biases [[buffer(3)]],
    device float *output [[buffer(4)]], constant uint &m [[buffer(5)]],
    constant uint &n [[buffer(6)]], constant uint &k [[buffer(7)]],
    uint2 position [[thread_position_in_grid]]) {
  reference_qmm<4>(input, weights, scales, biases, output, m, n, k,
                   position);
}

kernel void omlx_m5_reference_qmm_q5(
    device const float *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *scales [[buffer(2)]],
    device const float *biases [[buffer(3)]],
    device float *output [[buffer(4)]], constant uint &m [[buffer(5)]],
    constant uint &n [[buffer(6)]], constant uint &k [[buffer(7)]],
    uint2 position [[thread_position_in_grid]]) {
  reference_qmm<5>(input, weights, scales, biases, output, m, n, k,
                   position);
}
