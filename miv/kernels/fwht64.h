#pragma once

#include <metal_stdlib>

using namespace metal;

constant constexpr ulong kApple10MaxPrefillTgSramBytes = 16384ul;
constant constexpr ulong kApple10HighOccTgSramBytes = 10240ul;

// Involutive intra-row Apple10 NAX quartet permutation on a 64-element group:
// Swaps bits [5:4] and [3:2], leaving bits [1:0] untouched.
// Maps logical k = (s << 4) | (q << 2) | c to stored slot = (q << 4) | (s << 2) | c
// so lane fn = 4*q reads all 16 elements across s = 0..3 contiguously.
inline uint omlx_m5_pi64(uint off) {
  return ((off & 12u) << 2u) | ((off >> 2u) & 12u) | (off & 3u);
}

// Same permutation on 4-element (bfloat4) chunk indices c4 = off >> 2 in [0..15].
inline uint omlx_m5_pi16(uint c4) {
  return ((c4 & 3u) << 2u) | ((c4 >> 2u) & 3u);
}

// Case A: Full 32-lane SIMDgroup FWHT-64 (2 elements per lane: v0 = x[2*lane], v1 = x[2*lane+1]).
// Normalized H_64 = (1/8) * H_tilde_64 (orthogonal & involutive: H_64^2 = I_64).
// 0 B Threadgroup SRAM, 0 barriers, 5 simd_shuffle_xor steps.
inline void omlx_m5_fwht64_sg32(thread float &v0, thread float &v1, uint lane) {
  // Stage 0 (stride 1, in-register)
  const float s0 = v0 + v1;
  const float d0 = v0 - v1;
  v0 = s0;
  v1 = d0;
  // Stages 1..5 (strides 2, 4, 8, 16, 32 via simd_shuffle_xor masks 1, 2, 4, 8, 16)
  #pragma clang loop unroll(full)
  for (uint d = 0u; d < 5u; ++d) {
    const uint mask = 1u << d;
    const bool lo = ((lane & mask) == 0u);
    const float p0 = simd_shuffle_xor(v0, ushort(mask));
    const float p1 = simd_shuffle_xor(v1, ushort(mask));
    v0 = lo ? (v0 + p0) : (p0 - v0);
    v1 = lo ? (v1 + p1) : (p1 - v1);
  }
  v0 *= 0.125f;
  v1 *= 0.125f;
}

// Case B: 8-lane aligned subgroup FWHT-64 (8 elements per lane: lo = x[8*l64 + 0..3], hi = x[8*l64 + 4..7]).
// Used in pretiled_qmv_mxfp4 (T=1/M=2) and moe9_mxfp4_kl16_down_* (B=1..8).
// 0 B Threadgroup SRAM, 0 barriers, only 3 simd_shuffle_xor steps (masks 1, 2, 4).
inline void omlx_m5_fwht64_sub8(thread float4 &lo, thread float4 &hi, uint lane) {
  // Stage 0 (stride 1)
  lo = float4(lo.x + lo.y, lo.x - lo.y, lo.z + lo.w, lo.z - lo.w);
  hi = float4(hi.x + hi.y, hi.x - hi.y, hi.z + hi.w, hi.z - hi.w);
  // Stage 1 (stride 2)
  lo = float4(lo.xy + lo.zw, lo.xy - lo.zw);
  hi = float4(hi.xy + hi.zw, hi.xy - hi.zw);
  // Stage 2 (stride 4)
  const float4 a = lo + hi;
  const float4 b = lo - hi;
  lo = a;
  hi = b;
  // Stages 3..5 (strides 8, 16, 32 across aligned 8-lane subgroup)
  #pragma clang loop unroll(full)
  for (uint d = 0u; d < 3u; ++d) {
    const uint mask = 1u << d;
    const bool is_lo = ((lane & mask) == 0u);
    const float4 plo = simd_shuffle_xor(lo, ushort(mask));
    const float4 phi = simd_shuffle_xor(hi, ushort(mask));
    lo = is_lo ? (lo + plo) : (plo - lo);
    hi = is_lo ? (hi + phi) : (phi - hi);
  }
  lo *= 0.125f;
  hi *= 0.125f;
}

// Case C1: 4-lane aligned subgroup FWHT-64 in pi64 quartet order.
// Lane k_l = lane & 3 holds v[s] = x[g*64 + s*16 + k_l*4 + 0..3] for s = 0..3.
// Output v[s] sits directly at pi64 slot 16*k_l + s*4 + 0..3.
// 0 B SRAM, 0 barriers, only 2 simd_shuffle_xor steps (masks 1, 2).
inline void omlx_m5_fwht64_sub4_pi64(thread float4 (&v)[4], uint lane) {
  // Bits 0 & 1 (strides 1 & 2, inside each float4)
  #pragma clang loop unroll(full)
  for (uint j = 0u; j < 4u; ++j) {
    v[j] = float4(v[j].x + v[j].y, v[j].x - v[j].y, v[j].z + v[j].w, v[j].z - v[j].w);
    v[j] = float4(v[j].xy + v[j].zw, v[j].xy - v[j].zw);
  }
  // Bits 4 & 5 (strides 16 & 32, across v[0..3] in the same lane)
  const float4 a0 = v[0] + v[1];
  const float4 a1 = v[0] - v[1];
  const float4 a2 = v[2] + v[3];
  const float4 a3 = v[2] - v[3];
  v[0] = a0 + a2;
  v[1] = a1 + a3;
  v[2] = a0 - a2;
  v[3] = a1 - a3;
  // Bits 2 & 3 (strides 4 & 8, across aligned 4-lane subgroup)
  #pragma clang loop unroll(full)
  for (uint d = 0u; d < 2u; ++d) {
    const uint mask = 1u << d;
    const bool is_lo = ((lane & mask) == 0u);
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
      const float4 p = simd_shuffle_xor(v[j], ushort(mask));
      v[j] = is_lo ? (v[j] + p) : (p - v[j]);
    }
  }
  #pragma clang loop unroll(full)
  for (uint j = 0u; j < 4u; ++j) {
    v[j] *= 0.125f;
  }
}

// Case C2: 4-lane aligned subgroup FWHT-64 in linear order (for v1 pretiled fallback).
// Lane k_l = lane & 3 holds v[j] = x[g*64 + k_l*16 + j*4 + 0..3] for j = 0..3.
inline void omlx_m5_fwht64_sub4_linear(thread float4 (&v)[4], uint lane) {
  // Bits 0 & 1 (strides 1 & 2, inside each float4)
  #pragma clang loop unroll(full)
  for (uint j = 0u; j < 4u; ++j) {
    v[j] = float4(v[j].x + v[j].y, v[j].x - v[j].y, v[j].z + v[j].w, v[j].z - v[j].w);
    v[j] = float4(v[j].xy + v[j].zw, v[j].xy - v[j].zw);
  }
  // Bits 2 & 3 (strides 4 & 8, across v[0..3] in the same lane)
  const float4 a0 = v[0] + v[1];
  const float4 a1 = v[0] - v[1];
  const float4 a2 = v[2] + v[3];
  const float4 a3 = v[2] - v[3];
  v[0] = a0 + a2;
  v[1] = a1 + a3;
  v[2] = a0 - a2;
  v[3] = a1 - a3;
  // Bits 4 & 5 (strides 16 & 32, across aligned 4-lane subgroup)
  #pragma clang loop unroll(full)
  for (uint d = 0u; d < 2u; ++d) {
    const uint mask = 1u << d;
    const bool is_lo = ((lane & mask) == 0u);
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
      const float4 p = simd_shuffle_xor(v[j], ushort(mask));
      v[j] = is_lo ? (v[j] + p) : (p - v[j]);
    }
  }
  #pragma clang loop unroll(full)
  for (uint j = 0u; j < 4u; ++j) {
    v[j] *= 0.125f;
  }
}

// Bit-exact match to w4a8_quantize_pair_g64:
// Quantizes 64 FP32 values (2 per lane) to symmetric INT8 [-127, 127],
// storing *sa_dst = amax / 127.0f and *ra_dst = float(sum_i qa_i) (UNSCALED!).
inline void omlx_m5_quantize_pair_g64_f32(
    float v0,
    float v1,
    uint lane,
    device int8_t *qa_dst,
    device float *sa_dst,
    device float *ra_dst) {
  const float amax = simd_max(max(fabs(v0), fabs(v1)));
  const float sa = (amax > 0.0f) ? (amax * (1.0f / 127.0f)) : 0.0f;
  const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
  const int q0 = clamp(int(rint(v0 * inv)), -127, 127);
  const int q1 = clamp(int(rint(v1 * inv)), -127, 127);
  qa_dst[lane * 2u + 0u] = int8_t(q0);
  qa_dst[lane * 2u + 1u] = int8_t(q1);
  const int r_sum = simd_sum(q0 + q1);
  if (lane == 0u) {
    *sa_dst = sa;
    *ra_dst = float(r_sum);
  }
}
