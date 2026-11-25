#include <metal_stdlib>

using namespace metal;

kernel void omlx_m5_copy_u32(device const uint *source [[buffer(0)]],
                             device uint *destination [[buffer(1)]],
                             constant uint &element_count [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
  if (index < element_count) {
    destination[index] = source[index];
  }
}
