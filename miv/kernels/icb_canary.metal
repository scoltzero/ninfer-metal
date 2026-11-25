#include <metal_stdlib>

using namespace metal;

struct IcbCanaryParameters {
  uint addend;
  uint multiplier;
  uint xor_mask;
  uint element_count;
};

kernel void omlx_m5_icb_stage_a(
    device const uint *source [[buffer(0)]],
    device uint *middle [[buffer(1)]],
    constant IcbCanaryParameters &parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < parameters.element_count) {
    middle[index] = source[index] + parameters.addend;
  }
}

kernel void omlx_m5_icb_stage_b(
    device const uint *middle [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant IcbCanaryParameters &parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < parameters.element_count) {
    output[index] =
        (middle[index] * parameters.multiplier) ^ parameters.xor_mask;
  }
}
