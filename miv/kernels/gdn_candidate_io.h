// SPDX-License-Identifier: Apache-2.0
// Diagnostic adapters preserve the current native normalization convention.
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
struct GdnCandidateParams { uint tokens, key_heads, value_heads, conv_width; };

kernel void omlx_m5_gdn_prepare_candidates(
    device const bfloat *convolved [[buffer(0)]],
    device const bfloat *a [[buffer(1)]], device const bfloat *b [[buffer(2)]],
    device const bfloat *a_log [[buffer(3)]], device const bfloat *dt_bias [[buffer(4)]],
    device float *q [[buffer(5)]], device float *k [[buffer(6)]],
    device float *v [[buffer(7)]], device float *g [[buffer(8)]],
    device float *beta [[buffer(9)]],
    constant GdnCandidateParams &p [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint token = tg.y / p.value_heads, head = tg.y % p.value_heads;
  if (token >= p.tokens) return;
  const uint key_width = p.key_heads * 128;
  device const bfloat *row = convolved + token * p.conv_width;
  if (head < p.key_heads) {
    float q2 = 0, k2 = 0;
    for (uint d = lane; d < 128; d += 32) {
      float qv = float(row[head * 128 + d]);
      float kv = float(row[key_width + head * 128 + d]);
      q2 = fma(qv, qv, q2); k2 = fma(kv, kv, k2);
    }
    const float iq = rsqrt(simd_sum(q2) / 128.0f + 1.0e-6f);
    const float ik = rsqrt(simd_sum(k2) / 128.0f + 1.0e-6f);
    const float scale = rsqrt(128.0f);
    for (uint d = lane; d < 128; d += 32) {
      q[(token * p.key_heads + head) * 128 + d] =
          float(row[head * 128 + d]) * iq * scale * scale;
      k[(token * p.key_heads + head) * 128 + d] =
          float(row[key_width + head * 128 + d]) * ik * scale;
    }
  }
  for (uint d = lane; d < 128; d += 32)
    v[(token * p.value_heads + head) * 128 + d] = float(row[2 * key_width + head * 128 + d]);
  if (lane == 0) {
    const float activation = float(a[token * p.value_heads + head]) + float(dt_bias[head]);
    const float sp = activation > 20.0f ? activation : log(1.0f + exp(activation));
    g[token * p.value_heads + head] = exp(-exp(float(a_log[head])) * sp);
    beta[token * p.value_heads + head] = 1.0f / (1.0f + exp(-float(b[token * p.value_heads + head])));
  }
}
kernel void omlx_m5_gdn_candidate_output(
    device const float *y [[buffer(0)]], device bfloat *output [[buffer(1)]],
    constant GdnCandidateParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < p.tokens * p.value_heads * 128) output[i] = bfloat(y[i]);
}
