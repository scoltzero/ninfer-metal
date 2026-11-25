#pragma once

// E2M1 magnitudes times two: {0,1,2,3,4,6,8,12}. Keeping the complete codec
// in an immediate avoids an implicit global-constant resource. That resource
// produced zero LUT values in Metal 4 pipelines under GPU Shader Validation.
inline float omlx_m5_decode_fp4(uint code) {
  const float magnitude =
      0.5f * float((0xc8643210u >> ((code & 7u) * 4u)) & 15u);
  return (code & 8u) == 0 ? magnitude : -magnitude;
}
