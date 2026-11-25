#ifndef OMLX_M5_COMMON_MATH_H
#define OMLX_M5_COMMON_MATH_H

#include <metal_simdgroup>
#include <metal_stdlib>

// Shared FP4 E2M1 / UE8M0, sub-byte bitplane expansion, SiLU, row-scale, and
// threadgroup argmax/top-k utilities across m5rt Metal 4 shaders.

inline float e8m0_to_float(uchar code) {
  const uint bits = code == 0u ? 0x00400000u : uint(code) << 23u;
  return as_type<float>(bits);
}

inline float nax_decode_e8m0(uchar code) { return e8m0_to_float(code); }

inline uchar nax_encode_e8m0_round_up(float value) {
  if (!(value > 0.0f) || !metal::isfinite(value)) {
    return 0u;
  }
  const int exponent = metal::clamp(int(metal::ceil(metal::log2(value))), -127, 127);
  return uchar(exponent + 127);
}

inline float fp4_nibble_to_float(uint code) {
  const float magnitude =
      0.5f * float((0xc8643210u >> ((code & 7u) * 4u)) & 15u);
  return (code & 8u) == 0u ? magnitude : -magnitude;
}

inline uchar nax_encode_e2m1(float value) {
  if (!metal::isfinite(value)) {
    return 0u;
  }
  const uchar sign = metal::signbit(value) ? 8u : 0u;
  const float magnitude = metal::fabs(value);
  uchar code = 0u;
  if (magnitude > 5.0f) {
    code = 7u;
  } else if (magnitude >= 3.5f) {
    code = 6u;
  } else if (magnitude > 2.5f) {
    code = 5u;
  } else if (magnitude >= 1.75f) {
    code = 4u;
  } else if (magnitude > 1.25f) {
    code = 3u;
  } else if (magnitude >= 0.75f) {
    code = 2u;
  } else if (magnitude > 0.25f) {
    code = 1u;
  }
  return uchar(code | sign);
}

// Expand 4 single-bit flags (bits 0..3 of nib) into 4 two-bit fields in a byte
// (for metal::uint2b_format).
inline uchar expand_4x1bit_to_4x2bit(uint nib) {
  return uchar((nib & 1u) | ((nib & 2u) << 1u) | ((nib & 4u) << 2u) |
               ((nib & 8u) << 3u));
}

inline float nax_silu_f32(float gv) {
  return gv / (1.0f + metal::exp(-gv));
}

inline void nax_compute_row_scale_f16(float gmax, thread float &rs,
                                      thread float &inv_rs) {
  rs = 1.0f;
  inv_rs = 1.0f;
  if (gmax > 16384.0f) {
    const int e = metal::ilogb(gmax) - 13;
    rs = metal::ldexp(1.0f, e);
    inv_rs = metal::ldexp(1.0f, -e);
  }
}

inline void reduce_argmax_256(float score, uint id,
                              threadgroup float *tg_scores,
                              threadgroup uint *tg_ids, uint tid,
                              thread float &best_score,
                              thread uint &best_id) {
  tg_scores[tid] = score;
  tg_ids[tid] = id;
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      const float other_s = tg_scores[tid + stride];
      const uint other_id = tg_ids[tid + stride];
      const float cur_s = tg_scores[tid];
      const uint cur_id = tg_ids[tid];
      if (other_s > cur_s || (other_s == cur_s && other_id < cur_id)) {
        tg_scores[tid] = other_s;
        tg_ids[tid] = other_id;
      }
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  best_score = tg_scores[0];
  best_id = tg_ids[0];
}

template <uint MaxK>
inline void insert_top_k(thread float (&scores)[MaxK], thread uint (&ids)[MaxK],
                         uint top_k, float score, uint id) {
  for (uint i = 0u; i < top_k; ++i) {
    if (score > scores[i] || (score == scores[i] && id < ids[i])) {
      for (uint j = top_k - 1u; j > i; --j) {
        scores[j] = scores[j - 1u];
        ids[j] = ids[j - 1u];
      }
      scores[i] = score;
      ids[i] = id;
      break;
    }
  }
}

#endif // OMLX_M5_COMMON_MATH_H
