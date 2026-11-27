// Module 3: Hybrid 48-GDN + 16-Full-Attn Cross-Chunk Reuse (`GDN-CacheBlend`)
// Compiled with `-mmacosx-version-min=27.0` (Metal 4.1, NO `-std=metal4.0`).
//
// Kernels:
//   1. `omlx_m5_blend_extract_kv_prerope_affine4_g128`
//      Packs pre-RoPE K (post k_norm) and V into compact 144 B/vec `affine4_g128`.
//   2. `omlx_m5_blend_instantiate_kv_affine4_g128`
//      Materializes cached chunk KV into KvPagePool at arbitrary `p_new`
//      (`p_new % 64 != 0` supported): copies 100% of V and 50% of K (`[128..256)`)
//      bit-exact raw with ZERO re-quantization, rotates K `[0..rotary_dim)` by
//      `RoPE(p_new + t)`, and re-quantizes Group 0 `[0..128)` only.
//   3. `omlx_m5_blend_refresh_kv_page_summary_affine4_g128`
//      Recomputes exact 8-subband FP16 min/max `KvAffine4PageSummary` headers on
//      touched physical KV pages (race-free across misaligned/scattered writes).
//   4. `omlx_m5_gdn_extract_transition_m_c64`, `omlx_m5_gdn_pack_operator`,
//      `omlx_m5_gdn_compute_cum_decay_f16`, and `omlx_m5_gdn_state_compose_nax`
//      Exact affine-linear GDN interior state transition:
//        `S_out = S_in * M_C + B_C + DeltaS_hkvd`
//      using NAX `matmul2d` (`8 KiB` static TG SRAM, 192 TGs on `Hv = 48`),
//      supporting both BF16 and compact FP8 E4M3 + per-row FP16 scale operators.
//   5. `omlx_m5_blend_quantize_int8_g128_bf16`, `omlx_m5_blend_dequantize_int8_g128_bf16`,
//      `omlx_m5_blend_hkvd_l2_score`, and `omlx_m5_blend_hkvd_topk_select`
//      Exact L2 deviation `||AttnMix_stitched^(3)(t) - AttnMix_cached^(3)(t)||_2`
//      + exact 4-pass MSB radix top-k selection outputting strictly ascending
//      `active_idx[0 .. M_active)` and `active_pos[0 .. M_active)`.
//   6. `omlx_m5_blend_gather_rows_bf16`, `omlx_m5_blend_scatter_rows_bf16`, and
//      `omlx_m5_attention_qk_norm_rope_kv_scatter_affine4_g128`
//      Contiguous `[M_active, H]` gather/scatter and scattered `KvPagePool`
//      writeback at `active_pos[i]` so Route A (`bf16`/`w4a8`), Dual-ANE, and
//      Route B (`exact_nax`/`sage2`/`minference`) execute unchanged.

#include "common_math.h"
#include <metal_packed_numeric>
#include <metal_simdgroup>
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct KvAffine4VecMeta {
  half scale[4];
  half zero[4];
};

struct KvAffine4PageSummary {
  half subband_min[8];
  half subband_max[8];
};

struct BlendAffine4Vec256 {
  uchar codes_g0[64];
  uchar codes_g1[64];
  KvAffine4VecMeta meta;
};

struct BlendExtractPreRopeKvParams {
  uint key_value_heads;
  uint head_dim;
  uint rotary_dim;
  uint tokens;
  uint key_token_stride;
  uint value_token_stride;
  uint apply_k_norm;
  uint write_rotary_bf16;
  float epsilon;
  float weight_offset;
  uint _pad0;
  uint _pad1;
};

struct BlendInstantiateKvParams {
  uint key_value_heads;
  uint head_dim;
  uint rotary_dim;
  uint chunk_tokens;
  uint dst_start_pos;
  uint page_tokens;
  float theta;
  uint src_token_offset;
  uint use_rotary_bf16;
  uint _pad0;
  uint _pad1;
  uint _pad2;
};

struct BlendRefreshPageSummaryParams {
  uint key_value_heads;
  uint head_dim;
  uint page_tokens;
  uint first_logical_page;
  uint num_logical_pages;
  uint total_valid_tokens;
};

struct BlendGdnExtractMParams {
  uint key_heads;
  uint value_heads;
  uint key_dim;
  uint value_dim;
  uint tokens;
  uint _pad0;
};

struct BlendGdnPackOpParams {
  uint value_heads;
  uint dim;
  uint op_format; // 0 = BF16, 1 = FP8 E4M3 + per-row FP16 scale
  uint _pad0;
};

struct BlendGdnStateComposeParams {
  uint key_heads;
  uint value_heads;
  uint dim;
  uint op_format; // 0 = BF16, 1 = FP8 E4M3 + per-row FP16 scale
  uint chunk_tokens;
  uint num_hkvd_perturb;
  uint apply_bias;
  uint _pad0;
};

struct BlendHkvdSelectParams {
  uint total_tokens;
  uint mix_width;
  uint cached_is_int8;
  uint compute_l2_inline;
  uint explicit_hkvd_k;
  float hkvd_ratio;
  uint _pad0;
  uint _pad1;
};

struct BlendGatherScatterParams {
  uint max_active_tokens;
  uint width;
  uint use_device_count;
  uint residual_add;
};

struct BlendQkNormRopeScatterParams {
  uint query_heads;
  uint key_value_heads;
  uint head_dim;
  uint rotary_dim;
  uint max_active_tokens;
  uint query_token_stride;
  uint query_head_stride;
  uint key_token_stride;
  uint key_head_stride;
  uint value_token_stride;
  uint page_tokens;
  uint use_device_count;
  float epsilon;
  float theta;
  float weight_offset;
  uint _pad0;
};

struct BlendInt8QuantParams {
  uint tokens;
  uint width;
};

struct BlendGdnCumDecayParams {
  uint value_heads;
  uint tokens;
};

namespace blend_detail {

constant constexpr uint kChunkSize = 64u;
constant constexpr uint kKeyDim = 128u;
constant constexpr uint kValDim = 128u;
constant constexpr ulong kQkPairBfloats = 2ul * ulong(kChunkSize) * ulong(kKeyDim);
constant constexpr ulong kCtrlStrideBytes = 17152ul;
constant constexpr ulong kCtrlPrefByteOff = 16384ul;
constant constexpr ulong kCtrlSuffByteOff = 16640ul;
constant constexpr ulong kCtrlMetaByteOff = 16896ul;
constant constexpr uint kTokenMandatory = 1u;
constant constexpr uint kTokenInteriorCandidate = 2u;

inline ulong page_byte_stride(uint page_tokens, uint kv_heads, uint head_dim) {
  const ulong vec_bytes = ulong(head_dim / 2u) + 16ul;
  const ulong head_bytes = ulong(page_tokens) * vec_bytes + 32ul;
  return ulong(kv_heads) * head_bytes;
}

// Bit-exact match to `affine4_kv::quantize_vec4_g128_core` in attention_affine4.metal.
inline void quantize_vec4_g128_core(float4 v, device ushort *dst16, uint lane,
                                    thread half &out_s, thread half &out_z) {
  constexpr float kHalfMax = 65504.0f;
  v = clamp(v, float4(-kHalfMax), float4(kHalfMax));
  const float lane_lo = min(min(v.x, v.y), min(v.z, v.w));
  const float lane_hi = max(max(v.x, v.y), max(v.z, v.w));
  const float gmin = clamp(simd_min(lane_lo), -kHalfMax, kHalfMax);
  const float gmax = clamp(simd_max(lane_hi), -kHalfMax, kHalfMax);
  const half z_h = half(gmin);
  const float z_f = float(z_h);
  const float span = max(0.0f, gmax - z_f);
  const half s_h = half(min(span * (1.0f / 15.0f), kHalfMax));
  const float s_f = float(s_h);
  const float inv_s = (s_f > 0.0f) ? precise::divide(1.0f, s_f) : 0.0f;

  const uint q0 = uint(clamp(rint((v.x - z_f) * inv_s), 0.0f, 15.0f));
  const uint q1 = uint(clamp(rint((v.y - z_f) * inv_s), 0.0f, 15.0f));
  const uint q2 = uint(clamp(rint((v.z - z_f) * inv_s), 0.0f, 15.0f));
  const uint q3 = uint(clamp(rint((v.w - z_f) * inv_s), 0.0f, 15.0f));
  dst16[lane] = ushort(q0 | (q1 << 4u) | (q2 << 8u) | (q3 << 12u));
  out_s = s_h;
  out_z = z_h;
}

inline void quantize_vec_g128_device(device const bfloat *src_vec,
                                     device uchar *codes_head,
                                     device KvAffine4VecMeta *meta_head,
                                     uint page_tokens, uint within,
                                     uint head_dim, uint lane) {
  half s0 = half(0.0h), z0 = half(0.0h);
  half s1 = half(0.0h), z1 = half(0.0h);
  {
    device const bfloat4 *src4 =
        reinterpret_cast<device const bfloat4 *>(src_vec);
    device ushort *dst16 =
        reinterpret_cast<device ushort *>(codes_head + ulong(within) * 64ul);
    quantize_vec4_g128_core(float4(src4[lane]), dst16, lane, s0, z0);
  }
  if (head_dim == 256u) {
    device const bfloat4 *src4 =
        reinterpret_cast<device const bfloat4 *>(src_vec + 128u);
    device ushort *dst16 = reinterpret_cast<device ushort *>(
        codes_head + (ulong(page_tokens) + ulong(within)) * 64ul);
    quantize_vec4_g128_core(float4(src4[lane]), dst16, lane, s1, z1);
  }
  if (lane == 0u) {
    device KvAffine4VecMeta &m = meta_head[within];
    m.scale[0] = s0;
    m.scale[1] = s1;
    m.scale[2] = half(0.0h);
    m.scale[3] = half(0.0h);
    m.zero[0] = z0;
    m.zero[1] = z1;
    m.zero[2] = half(0.0h);
    m.zero[3] = half(0.0h);
  }
}

inline void quantize_vec_g128_tg(threadgroup const bfloat *src_vec,
                                 device uchar *codes_head,
                                 device KvAffine4VecMeta *meta_head,
                                 uint page_tokens, uint within, uint head_dim,
                                 uint lane) {
  half s0 = half(0.0h), z0 = half(0.0h);
  half s1 = half(0.0h), z1 = half(0.0h);
  {
    threadgroup const bfloat4 *src4 =
        reinterpret_cast<threadgroup const bfloat4 *>(src_vec);
    device ushort *dst16 =
        reinterpret_cast<device ushort *>(codes_head + ulong(within) * 64ul);
    quantize_vec4_g128_core(float4(src4[lane]), dst16, lane, s0, z0);
  }
  if (head_dim == 256u) {
    threadgroup const bfloat4 *src4 =
        reinterpret_cast<threadgroup const bfloat4 *>(src_vec + 128u);
    device ushort *dst16 = reinterpret_cast<device ushort *>(
        codes_head + (ulong(page_tokens) + ulong(within)) * 64ul);
    quantize_vec4_g128_core(float4(src4[lane]), dst16, lane, s1, z1);
  }
  if (lane == 0u) {
    device KvAffine4VecMeta &m = meta_head[within];
    m.scale[0] = s0;
    m.scale[1] = s1;
    m.scale[2] = half(0.0h);
    m.scale[3] = half(0.0h);
    m.zero[0] = z0;
    m.zero[1] = z1;
    m.zero[2] = half(0.0h);
    m.zero[3] = half(0.0h);
  }
}

// Exact OCP FP8 E4M3 (bias=7, max finite = 448.0, no inf, 0x7F/0xFF NaN) codec
// portable across both Apple9 and Apple10, matching Metal 4.1 `metal_fp8_e4m3_format`.
inline uchar encode_fp8_e4m3(float x) {
  if (!isfinite(x)) {
    return 0u;
  }
  const uchar sign = signbit(x) ? 0x80u : 0u;
  float ax = fabs(x);
  if (ax >= 448.0f) {
    return uchar(sign | 0x7Eu); // max normal = 448.0
  }
  if (ax < 1.953125e-3f) { // < 2^-9 -> subnormal or zero
    const uint sub = uint(clamp(rint(ax * 512.0f), 0.0f, 7.0f));
    return uchar(sign | sub);
  }
  const uint bits = as_type<uint>(ax);
  const int exp_unbiased = int((bits >> 23u) & 0xFFu) - 127;
  const uint mant23 = bits & 0x7FFFFFu;
  if (exp_unbiased < -6) {
    // Subnormal range [2^-9 .. 2^-6): step = 2^-9
    const uint sub = uint(clamp(rint(ax * 512.0f), 0.0f, 7.0f));
    if (sub >= 8u) {
      return uchar(sign | 0x08u);
    }
    return uchar(sign | sub);
  }
  int e_field = exp_unbiased + 7; // [1 .. 15]
  // Round-to-nearest-even from 23-bit mantissa to 3-bit mantissa.
  const uint m3 = mant23 >> 20u;
  const uint rem = mant23 & 0xFFFFFu;
  const uint half_lsb = 0x80000u;
  uint m_round = m3 + ((rem > half_lsb || (rem == half_lsb && (m3 & 1u) != 0u)) ? 1u : 0u);
  if (m_round >= 8u) {
    m_round = 0u;
    ++e_field;
  }
  if (e_field >= 15 && m_round >= 7u) {
    return uchar(sign | 0x7Eu); // saturate to 448.0 (avoid NaN 0x7F)
  }
  return uchar(sign | (uint(e_field) << 3u) | m_round);
}

inline float decode_fp8_e4m3(uchar code) {
  const uint sign = (uint(code) & 0x80u) << 24u;
  const uint e = (uint(code) >> 3u) & 0x0Fu;
  const uint m = uint(code) & 0x07u;
  if (e == 0u) {
    const float val = float(m) * (1.0f / 512.0f);
    return (sign != 0u) ? -val : val;
  }
  if (e == 15u && m == 7u) {
    return 0.0f;
  }
  const uint f32_exp = (e + 120u) << 23u;
  const uint f32_mant = m << 20u;
  return as_type<float>(sign | f32_exp | f32_mant);
}

inline uint float_nonneg_to_sortable_u32(float v) {
  if (!isfinite(v) || v <= 0.0f) {
    return 0u;
  }
  return as_type<uint>(v);
}

} // namespace blend_detail

// ============================================================================
// 1. Extract pre-RoPE K (optionally applying k_norm) + V into BlendAffine4Vec256
//    Launch: grid = {32, tokens * kv_heads}, tptg = {32}
// ============================================================================
kernel void omlx_m5_blend_extract_kv_prerope_affine4_g128(
    device const bfloat *key [[buffer(0)]],
    device const bfloat *value [[buffer(1)]],
    device const bfloat *key_norm [[buffer(2)]],
    device BlendAffine4Vec256 *dst_k_pre [[buffer(3)]],
    device half *dst_k_rotary_f16 [[buffer(4)]],
    device BlendAffine4Vec256 *dst_v [[buffer(5)]],
    constant BlendExtractPreRopeKvParams &params [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint kvh = tg.y % params.key_value_heads;
  const uint token = tg.y / params.key_value_heads;
  if (token >= params.tokens || params.head_dim != 256u) {
    return;
  }

  device const bfloat *k_src = key + ulong(token) * params.key_token_stride +
                               ulong(kvh) * params.head_dim;
  device const bfloat *v_src = value + ulong(token) * params.value_token_stride +
                               ulong(kvh) * params.head_dim;

  const ulong dst_idx = ulong(token) * ulong(params.key_value_heads) + ulong(kvh);
  device BlendAffine4Vec256 &dk = dst_k_pre[dst_idx];
  device BlendAffine4Vec256 &dv = dst_v[dst_idx];

  // Key: optionally apply exact 0-ULP k_norm (matching attention_affine4.metal
  // strided reduction order, NO RoPE!) then quantize g0 & g1.
  float4 k_g0_f32 = float4(*reinterpret_cast<device const bfloat4 *>(k_src + lane * 4u));
  float4 k_g1 = float4(*reinterpret_cast<device const bfloat4 *>(k_src + 128u + lane * 4u));
  if (params.apply_k_norm != 0u && params.epsilon >= 0.0f) {
    float square_sum = 0.0f;
    for (uint column = lane; column < 256u; column += 32u) {
      const float val = float(k_src[column]);
      square_sum = fma(val, val, square_sum);
    }
    const float inv_rms = precise::rsqrt(
        precise::divide(simd_sum(square_sum), 256.0f) + params.epsilon);
    const float4 n0 =
        float4(*reinterpret_cast<device const bfloat4 *>(key_norm + lane * 4u)) +
        float4(params.weight_offset);
    const float4 n1 =
        float4(*reinterpret_cast<device const bfloat4 *>(key_norm + 128u + lane * 4u)) +
        float4(params.weight_offset);
    k_g0_f32 = k_g0_f32 * inv_rms * n0;
    k_g1 = float4(bfloat4(k_g1 * inv_rms * n1));
  }
  const float4 k_g0 = float4(bfloat4(k_g0_f32));

  // Optional single-quantize rotary sidecar: preserve 10-bit-mantissa FP16
  // post-k_norm values for dims [0 .. rotary_dim) (`rotary_dim = 64` -> 16 lanes),
  // and exact [rotary_dim .. 128) min/max bounds in unused meta.{zero,scale}[2]
  // so Group-0 requantization after RoPE achieves ~0.99995+ parity vs fresh append!
  if (params.write_rotary_bf16 != 0u && lane * 4u < params.rotary_dim) {
    device half4 *rot_dst = reinterpret_cast<device half4 *>(
        dst_k_rotary_f16 + dst_idx * ulong(params.rotary_dim) + ulong(lane) * 4ul);
    *rot_dst = half4(clamp(k_g0_f32, float4(-65504.0f), float4(65504.0f)));
  }
  const float k0_lo = min(min(k_g0.x, k_g0.y), min(k_g0.z, k_g0.w));
  const float k0_hi = max(max(k_g0.x, k_g0.y), max(k_g0.z, k_g0.w));
  const float unrot_g0_min = simd_min(lane * 4u >= params.rotary_dim ? k0_lo : 65504.0f);
  const float unrot_g0_max = simd_max(lane * 4u >= params.rotary_dim ? k0_hi : -65504.0f);

  half ks0 = half(0.0h), kz0 = half(0.0h);
  half ks1 = half(0.0h), kz1 = half(0.0h);
  blend_detail::quantize_vec4_g128_core(
      k_g0, reinterpret_cast<device ushort *>(dk.codes_g0), lane, ks0, kz0);
  blend_detail::quantize_vec4_g128_core(
      k_g1, reinterpret_cast<device ushort *>(dk.codes_g1), lane, ks1, kz1);
  if (lane == 0u) {
    dk.meta.scale[0] = ks0;
    dk.meta.scale[1] = ks1;
    dk.meta.scale[2] =
        (params.rotary_dim < 128u) ? half(clamp(unrot_g0_max, -65504.0f, 65504.0f)) : half(0.0h);
    dk.meta.scale[3] = half(0.0h);
    dk.meta.zero[0] = kz0;
    dk.meta.zero[1] = kz1;
    dk.meta.zero[2] =
        (params.rotary_dim < 128u) ? half(clamp(unrot_g0_min, -65504.0f, 65504.0f)) : half(0.0h);
    dk.meta.zero[3] = half(0.0h);
  }

  // Value: quantize directly into BlendAffine4Vec256.
  const float4 v_g0 =
      float4(*reinterpret_cast<device const bfloat4 *>(v_src + lane * 4u));
  const float4 v_g1 =
      float4(*reinterpret_cast<device const bfloat4 *>(v_src + 128u + lane * 4u));
  half vs0 = half(0.0h), vz0 = half(0.0h);
  half vs1 = half(0.0h), vz1 = half(0.0h);
  blend_detail::quantize_vec4_g128_core(
      v_g0, reinterpret_cast<device ushort *>(dv.codes_g0), lane, vs0, vz0);
  blend_detail::quantize_vec4_g128_core(
      v_g1, reinterpret_cast<device ushort *>(dv.codes_g1), lane, vs1, vz1);
  if (lane == 0u) {
    dv.meta.scale[0] = vs0;
    dv.meta.scale[1] = vs1;
    dv.meta.scale[2] = half(0.0h);
    dv.meta.scale[3] = half(0.0h);
    dv.meta.zero[0] = vz0;
    dv.meta.zero[1] = vz1;
    dv.meta.zero[2] = half(0.0h);
    dv.meta.zero[3] = half(0.0h);
  }
}

// ============================================================================
// 2. Instantiate Cached Chunk KV into KvPagePool at Arbitrary `p_new`
//    - Arbitrary `p_new % 64 != 0` fully supported.
//    - 100% of V copied bit-exact raw (ZERO re-quantization!).
//    - 50% of K (`[128..256)`, Group 1) copied bit-exact raw (ZERO re-quant!).
//    - Only Group 0 (`[0..128)`) unpacks, applies RoPE(p_new + t) to `[0..rotary_dim)`
//      (`rotary_dim = 64 <= 128`), and re-quantizes Group 0!
//    Launch: grid = {32, chunk_tokens * kv_heads}, tptg = {32}
//    Static TG SRAM: 512 B (`tg_g0[128]` float).
// ============================================================================
kernel void omlx_m5_blend_instantiate_kv_affine4_g128(
    device const BlendAffine4Vec256 *src_k_pre [[buffer(0)]],
    device const half *src_k_rotary_f16 [[buffer(1)]],
    device const BlendAffine4Vec256 *src_v [[buffer(2)]],
    device uchar *cache_key [[buffer(3)]],
    device uchar *cache_value [[buffer(4)]],
    device const uint *page_table [[buffer(5)]],
    constant BlendInstantiateKvParams &params [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float tg_g0[128];

  const uint kvh = tg.y % params.key_value_heads;
  const uint t = tg.y / params.key_value_heads;
  if (t >= params.chunk_tokens || params.head_dim != 256u ||
      params.rotary_dim > 128u || (params.rotary_dim & 1u) != 0u) {
    return;
  }

  const uint logical_pos = params.dst_start_pos + t;
  const uint logical_page = logical_pos / params.page_tokens;
  const uint within = logical_pos - logical_page * params.page_tokens;
  const uint phys_page = page_table[logical_page];

  const ulong page_bytes = blend_detail::page_byte_stride(
      params.page_tokens, params.key_value_heads, params.head_dim);
  const ulong head_code_bytes = ulong(params.page_tokens) * 128ul;
  const ulong all_codes_bytes = ulong(params.key_value_heads) * head_code_bytes;

  device uchar *k_page = cache_key + ulong(phys_page) * page_bytes;
  device uchar *v_page = cache_value + ulong(phys_page) * page_bytes;
  device uchar *k_codes_head = k_page + ulong(kvh) * head_code_bytes;
  device uchar *v_codes_head = v_page + ulong(kvh) * head_code_bytes;
  device KvAffine4VecMeta *k_meta_head =
      reinterpret_cast<device KvAffine4VecMeta *>(k_page + all_codes_bytes) +
      ulong(kvh) * params.page_tokens;
  device KvAffine4VecMeta *v_meta_head =
      reinterpret_cast<device KvAffine4VecMeta *>(v_page + all_codes_bytes) +
      ulong(kvh) * params.page_tokens;

  const ulong src_idx =
      ulong(params.src_token_offset + t) * ulong(params.key_value_heads) +
      ulong(kvh);
  device const BlendAffine4Vec256 &sk = src_k_pre[src_idx];
  device const BlendAffine4Vec256 &sv = src_v[src_idx];

  // A. V: 100% raw bit-exact copy (Group 0 + Group 1 + Meta). ZERO re-quant!
  reinterpret_cast<device ushort *>(v_codes_head + ulong(within) * 64ul)[lane] =
      reinterpret_cast<device const ushort *>(sv.codes_g0)[lane];
  reinterpret_cast<device ushort *>(
      v_codes_head + (ulong(params.page_tokens) + ulong(within)) * 64ul)[lane] =
      reinterpret_cast<device const ushort *>(sv.codes_g1)[lane];
  if (lane == 0u) {
    v_meta_head[within] = sv.meta;
  }

  // B. K Group 1 [128..256): 100% raw bit-exact copy (RoPE-free!). ZERO re-quant!
  reinterpret_cast<device ushort *>(
      k_codes_head + (ulong(params.page_tokens) + ulong(within)) * 64ul)[lane] =
      reinterpret_cast<device const ushort *>(sk.codes_g1)[lane];

  // C. K Group 0 [0..128): load rotary dims [0..rotary_dim) from FP16 rotary
  //    sidecar when enabled (true single-quantize RoPE!) and remaining dims from
  //    affine4 codes_g0, apply exact FP32 RoPE(p_new + t), and quantize Group 0!
  if (params.use_rotary_bf16 != 0u && lane * 4u < params.rotary_dim) {
    const float4 r4 = float4(*reinterpret_cast<device const half4 *>(
        src_k_rotary_f16 + src_idx * ulong(params.rotary_dim) +
        ulong(lane) * 4ul));
    tg_g0[lane * 4u + 0u] = r4.x;
    tg_g0[lane * 4u + 1u] = r4.y;
    tg_g0[lane * 4u + 2u] = r4.z;
    tg_g0[lane * 4u + 3u] = r4.w;
  } else {
    const ushort packed_k0 =
        reinterpret_cast<device const ushort *>(sk.codes_g0)[lane];
    const float sk0 = float(sk.meta.scale[0]);
    const float zk0 = float(sk.meta.zero[0]);
    tg_g0[lane * 4u + 0u] = fma(float((packed_k0 >> 0u) & 0xFu), sk0, zk0);
    tg_g0[lane * 4u + 1u] = fma(float((packed_k0 >> 4u) & 0xFu), sk0, zk0);
    tg_g0[lane * 4u + 2u] = fma(float((packed_k0 >> 8u) & 0xFu), sk0, zk0);
    tg_g0[lane * 4u + 3u] = fma(float((packed_k0 >> 12u) & 0xFu), sk0, zk0);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  const uint half_rotary = params.rotary_dim / 2u;
  const float position = float(logical_pos);
  for (uint pair = lane; pair < half_rotary; pair += 32u) {
    const uint i0 = pair;
    const uint i1 = pair + half_rotary;
    const float x0 = tg_g0[i0];
    const float x1 = tg_g0[i1];
    const float exponent =
        precise::divide(float(2u * pair), float(params.rotary_dim));
    const float angle =
        precise::divide(position, precise::pow(params.theta, exponent));
    const float c = precise::cos(angle);
    const float s = precise::sin(angle);
    // Round through bfloat to match `attention_affine4.metal:1297-1298` to 0 ULP.
    tg_g0[i0] = float(bfloat(fma(-x1, s, x0 * c)));
    tg_g0[i1] = float(bfloat(fma(x0, s, x1 * c)));
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  constexpr float kHalfMax = 65504.0f;
  float4 rot_v(tg_g0[lane * 4u + 0u], tg_g0[lane * 4u + 1u],
               tg_g0[lane * 4u + 2u], tg_g0[lane * 4u + 3u]);
  rot_v = clamp(rot_v, float4(-kHalfMax), float4(kHalfMax));
  float lane_lo = min(min(rot_v.x, rot_v.y), min(rot_v.z, rot_v.w));
  float lane_hi = max(max(rot_v.x, rot_v.y), max(rot_v.z, rot_v.w));
  if (params.use_rotary_bf16 != 0u && params.rotary_dim < 128u &&
      lane * 4u >= params.rotary_dim) {
    // Use exact pre-quantization [rotary_dim..128) min/max stored in meta[2]
    // so rotated Group-0 scale/zero matches fresh kv_append to 0 ULP!
    const float unrot_lo = float(sk.meta.zero[2]);
    const float unrot_hi = float(sk.meta.scale[2]);
    rot_v = clamp(rot_v, float4(unrot_lo), float4(unrot_hi));
    lane_lo = unrot_lo;
    lane_hi = unrot_hi;
  }
  const float gmin = clamp(simd_min(lane_lo), -kHalfMax, kHalfMax);
  const float gmax = clamp(simd_max(lane_hi), -kHalfMax, kHalfMax);
  const half new_z0 = half(gmin);
  const float z_f = float(new_z0);
  const float span = max(0.0f, gmax - z_f);
  const half new_s0 = half(min(span * (1.0f / 15.0f), kHalfMax));
  const float s_f = float(new_s0);
  const float inv_s = (s_f > 0.0f) ? precise::divide(1.0f, s_f) : 0.0f;
  const uint q0 = uint(clamp(rint((rot_v.x - z_f) * inv_s), 0.0f, 15.0f));
  const uint q1 = uint(clamp(rint((rot_v.y - z_f) * inv_s), 0.0f, 15.0f));
  const uint q2 = uint(clamp(rint((rot_v.z - z_f) * inv_s), 0.0f, 15.0f));
  const uint q3 = uint(clamp(rint((rot_v.w - z_f) * inv_s), 0.0f, 15.0f));
  reinterpret_cast<device ushort *>(k_codes_head + ulong(within) * 64ul)[lane] =
      ushort(q0 | (q1 << 4u) | (q2 << 8u) | (q3 << 12u));
  if (lane == 0u) {
    device KvAffine4VecMeta &km = k_meta_head[within];
    km.scale[0] = new_s0;
    km.scale[1] = sk.meta.scale[1]; // Preserved bit-exact from Group 1!
    km.scale[2] = half(0.0h);
    km.scale[3] = half(0.0h);
    km.zero[0] = new_z0;
    km.zero[1] = sk.meta.zero[1];   // Preserved bit-exact from Group 1!
    km.zero[2] = half(0.0h);
    km.zero[3] = half(0.0h);
  }
}

// ============================================================================
// 3. Refresh KvAffine4PageSummary (8 subbands x 32 dims) on Touched Pages
//    Launch: grid = {32, num_logical_pages * kv_heads}, tptg = {32}
//    Static TG SRAM: 0 B.
// ============================================================================
kernel void omlx_m5_blend_refresh_kv_page_summary_affine4_g128(
    device uchar *cache_key [[buffer(0)]],
    device uchar *cache_value [[buffer(1)]],
    device const uint *page_table [[buffer(2)]],
    constant BlendRefreshPageSummaryParams &params [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint kvh = tg.y % params.key_value_heads;
  const uint rel_page = tg.y / params.key_value_heads;
  if (rel_page >= params.num_logical_pages || params.head_dim != 256u) {
    return;
  }

  const uint logical_page = params.first_logical_page + rel_page;
  const uint page_tok_base = logical_page * params.page_tokens;
  const uint valid = (params.total_valid_tokens > page_tok_base)
                         ? min(params.page_tokens,
                               params.total_valid_tokens - page_tok_base)
                         : 0u;
  const uint phys_page = page_table[logical_page];

  const ulong page_bytes = blend_detail::page_byte_stride(
      params.page_tokens, params.key_value_heads, params.head_dim);
  const ulong head_code_bytes = ulong(params.page_tokens) * 128ul;
  const ulong all_codes_bytes = ulong(params.key_value_heads) * head_code_bytes;
  const ulong all_meta_bytes =
      ulong(params.key_value_heads) * ulong(params.page_tokens) * 16ul;

  auto refresh_plane = [&](device uchar *plane_base) {
    device uchar *page_ptr = plane_base + ulong(phys_page) * page_bytes;
    device const uchar *codes_head = page_ptr + ulong(kvh) * head_code_bytes;
    device const KvAffine4VecMeta *meta_head =
        reinterpret_cast<device const KvAffine4VecMeta *>(page_ptr +
                                                          all_codes_bytes) +
        ulong(kvh) * params.page_tokens;
    device KvAffine4PageSummary *summary =
        reinterpret_cast<device KvAffine4PageSummary *>(
            page_ptr + all_codes_bytes + all_meta_bytes) +
        ulong(kvh);

    for (uint sb = 0u; sb < 8u; ++sb) {
      const uint g = sb >> 2u;            // 0 or 1
      const uint dim_in_g = (sb & 3u) * 32u + lane;
      const uint byte_in_vec = dim_in_g >> 1u;
      const uint nib_shift = (dim_in_g & 1u) * 4u;
      float lo = 65504.0f;
      float hi = -65504.0f;
      for (uint w = 0u; w < valid; ++w) {
        const uchar packed =
            codes_head[(ulong(g) * ulong(params.page_tokens) + ulong(w)) * 64ul +
                       byte_in_vec];
        const float q = float((uint(packed) >> nib_shift) & 0xFu);
        const float val =
            fma(q, float(meta_head[w].scale[g]), float(meta_head[w].zero[g]));
        lo = min(lo, val);
        hi = max(hi, val);
      }
      const float sb_min = (valid > 0u) ? simd_min(lo) : 0.0f;
      const float sb_max = (valid > 0u) ? simd_max(hi) : 0.0f;
      if (lane == 0u) {
        summary->subband_min[sb] = half(clamp(sb_min, -65504.0f, 65504.0f));
        summary->subband_max[sb] = half(clamp(sb_max, -65504.0f, 65504.0f));
      }
    }
  };

  refresh_plane(cache_key);
  refresh_plane(cache_value);
}

// ============================================================================
// 4. GDN Exact Affine Transition Extraction (`M_C` via S_init = I_128, V = 0)
//    Consumes `qk_ws` and `ctrl_ws` produced by `omlx_m5_gdn_chunk_prepare_c64`
//    over interior `[W_bnd, L_m)` tokens.
//    Launch: grid = {128, value_heads, 2}, tptg = {128} (4 SGs, each owns 16 rows)
//    Static TG SRAM: 4 * 64 * 16 * 2 B = 8,192 B (8.0 KiB <= 16 KiB!).
// ============================================================================
// Zero-spill GDN Exact Affine Transition Extraction (`M_C` via S_init = I_128, V = 0):
// Uses Apple10 register-fragment NAX (`nax_mma_16x16x32` / `nax_mma_16x32x16_nt`, 0 B spills!)
// with a portable Apple9 pure-FP32 fallback (also 0 B spills!). Static TG SRAM = 9,216 B.
[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_extract_transition_m_c64(
    device const bfloat *qk_ws [[buffer(0)]],
    device const uchar *ctrl_ws [[buffer(1)]],
    device float *m_out_f32 [[buffer(2)]],
    constant BlendGdnExtractMParams &params [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint kValSgDim = 16u;
  constexpr uint kWorkPad = 72u; // 64 + 8 (16-byte bfloat4 aligned)
  struct BlendExtractMTgStorage {
    bfloat work[4u * kValSgDim * kWorkPad]; // 9,216 B (9.00 KiB <= 16.00 KiB!)
  };
  static_assert(sizeof(BlendExtractMTgStorage) <= kApple10MaxPrefillTgSramBytes,
                "BlendExtractMTgStorage exceeds 16,384 B!");
  alignas(16) threadgroup BlendExtractMTgStorage tg_mem;

  const uint hv = tg.y;
  const uint half_idx = tg.z;
  if (hv >= params.value_heads || half_idx >= 2u || sg >= 4u ||
      params.key_dim != blend_detail::kKeyDim ||
      params.value_dim != blend_detail::kValDim) {
    return;
  }

  const uint repeat = params.value_heads / params.key_heads;
  const uint hk = hv / repeat;
  const uint dv_sg_base = half_idx * 64u + sg * kValSgDim;
  device float *m_sg =
      m_out_f32 + (ulong(hv) * ulong(blend_detail::kValDim) + ulong(dv_sg_base)) *
                      ulong(blend_detail::kKeyDim);

  // Initialize M_sg = I_128 rows [dv_sg_base .. +16) directly in m_out_f32.
  for (uint row = 0u; row < kValSgDim; ++row) {
    const uint g_row = dv_sg_base + row;
    for (uint col = lane; col < blend_detail::kKeyDim; col += 32u) {
      m_sg[ulong(row) * blend_detail::kKeyDim + col] =
          (g_row == col) ? 1.0f : 0.0f;
    }
  }

  threadgroup bfloat *my_work = tg_mem.work + sg * (kValSgDim * kWorkPad);

  // Check Apple10 16x16x32 cooperative tensor lane layout at compile time on Apple10.
  bool is_apple10 = false;
  {
    constexpr auto desc_chk = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_chk, execution_simdgroup> chk_op;
    auto ct_chk =
        chk_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    const auto c0 = ct_chk.get_multidimensional_index(ushort(0u));
    const auto c2 = ct_chk.get_multidimensional_index(ushort(2u));
    is_apple10 = (int(c2[0]) - int(c0[0]) == 2) && (int(c2[1]) == int(c0[1]));
  }

  if (is_apple10) {
    const ushort2 lc = nax_apple10_lane_coord(ushort(lane));
    const uint fm = uint(lc.x);
    const uint fn = uint(lc.y);
    const vec<bfloat, 8> kZeroBf8(bfloat(0.0f));

    for (uint c = 0u; c * blend_detail::kChunkSize < params.tokens; ++c) {
      const uint t0 = c * blend_detail::kChunkSize;
      const uint clen = min(blend_detail::kChunkSize, params.tokens - t0);
      const uint n_rblks = (clen + 15u) / 16u;

      device const bfloat *ws_nk =
          qk_ws +
          (ulong(c) * ulong(params.key_heads) + ulong(hk)) *
              blend_detail::kQkPairBfloats;
      device const uchar *ctrl_raw =
          ctrl_ws +
          (ulong(c) * ulong(params.value_heads) + ulong(hv)) *
              blend_detail::kCtrlStrideBytes;
      device const bfloat *ctrl_solve =
          reinterpret_cast<device const bfloat *>(ctrl_raw);
      device const float *ctrl_pref =
          reinterpret_cast<device const float *>(ctrl_raw + blend_detail::kCtrlPrefByteOff);
      device const float *ctrl_suff =
          reinterpret_cast<device const float *>(ctrl_raw + blend_detail::kCtrlSuffByteOff);
      const float decay_last =
          reinterpret_cast<device const float *>(ctrl_raw + blend_detail::kCtrlMetaByteOff)[0];

      // 1 & 2. pred = nk @ M_sg^T -> rhsT[dv=0..15][r=0..63] = bfloat(-pref[r] * pred)
      for (uint rb = 0u; rb < 4u; ++rb) {
        const uint r_base = rb * 16u;
        vec<float, 8> pred(0.0f);
        if (rb < n_rblks) {
          for (uint k0 = 0u; k0 < blend_detail::kKeyDim; k0 += 32u) {
            const bfloat4 a0_0 = *reinterpret_cast<device const bfloat4 *>(
                ws_nk + ulong(r_base + fm) * blend_detail::kKeyDim + k0 + fn);
            const bfloat4 a0_1 = *reinterpret_cast<device const bfloat4 *>(
                ws_nk + ulong(r_base + fm + 8u) * blend_detail::kKeyDim + k0 + fn);
            const bfloat4 a1_0 = *reinterpret_cast<device const bfloat4 *>(
                ws_nk + ulong(r_base + fm) * blend_detail::kKeyDim + k0 + 16u + fn);
            const bfloat4 a1_1 = *reinterpret_cast<device const bfloat4 *>(
                ws_nk + ulong(r_base + fm + 8u) * blend_detail::kKeyDim + k0 + 16u + fn);
            const vec<bfloat, 8> a0(a0_0.x, a0_0.y, a0_0.z, a0_0.w,
                                    a0_1.x, a0_1.y, a0_1.z, a0_1.w);
            const vec<bfloat, 8> a1(a1_0.x, a1_0.y, a1_0.z, a1_0.w,
                                    a1_1.x, a1_1.y, a1_1.z, a1_1.w);

            const bfloat4 m0_0 = bfloat4(*reinterpret_cast<device const float4 *>(
                m_sg + ulong(fm) * blend_detail::kKeyDim + k0 + fn));
            const bfloat4 m0_1 = bfloat4(*reinterpret_cast<device const float4 *>(
                m_sg + ulong(fm + 8u) * blend_detail::kKeyDim + k0 + fn));
            const bfloat4 m1_0 = bfloat4(*reinterpret_cast<device const float4 *>(
                m_sg + ulong(fm) * blend_detail::kKeyDim + k0 + 16u + fn));
            const bfloat4 m1_1 = bfloat4(*reinterpret_cast<device const float4 *>(
                m_sg + ulong(fm + 8u) * blend_detail::kKeyDim + k0 + 16u + fn));
            const vec<bfloat, 8> b0(m0_0.x, m0_0.y, m0_0.z, m0_0.w,
                                    m0_1.x, m0_1.y, m0_1.z, m0_1.w);
            const vec<bfloat, 8> b1(m1_0.x, m1_0.y, m1_0.z, m1_0.w,
                                    m1_1.x, m1_1.y, m1_1.z, m1_1.w);
            nax_mma_16x16x32<bfloat, true>(pred, a0, a1, b0, b1);
          }
        }
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint r = r_base + fm + h * 8u;
          const float pref = (r < clen) ? ctrl_pref[r] : 0.0f;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            my_work[(fn + j) * kWorkPad + r] =
                (r < clen) ? bfloat(-pref * pred[h * 4u + j]) : bfloat(0.0f);
          }
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);

      // 3. Lower-triangular delta = solve @ rhs in REVERSE rb = 3..0 so my_work
      //    overwrites rhsT in-place with deltaSuffT[dv][r] = bfloat(suff[r] * delta)!
      for (int rb_i = 3; rb_i >= 0; --rb_i) {
        const uint rb = uint(rb_i);
        const uint r_base = rb * 16u;
        vec<float, 8> delta(0.0f);
        if (rb < n_rblks) {
          for (uint kt = 0u; kt <= rb; kt += 2u) {
            const uint k0 = kt * 16u;
            const bfloat4 s0_0 = *reinterpret_cast<device const bfloat4 *>(
                ctrl_solve + ulong(r_base + fm) * blend_detail::kChunkSize + k0 + fn);
            const bfloat4 s0_1 = *reinterpret_cast<device const bfloat4 *>(
                ctrl_solve + ulong(r_base + fm + 8u) * blend_detail::kChunkSize + k0 + fn);
            const vec<bfloat, 8> a0(s0_0.x, s0_0.y, s0_0.z, s0_0.w,
                                    s0_1.x, s0_1.y, s0_1.z, s0_1.w);
            const bfloat4 r0_0 = *reinterpret_cast<const threadgroup bfloat4 *>(
                my_work + fm * kWorkPad + k0 + fn);
            const bfloat4 r0_1 = *reinterpret_cast<const threadgroup bfloat4 *>(
                my_work + (fm + 8u) * kWorkPad + k0 + fn);
            const vec<bfloat, 8> b0(r0_0.x, r0_0.y, r0_0.z, r0_0.w,
                                    r0_1.x, r0_1.y, r0_1.z, r0_1.w);

            const bool has_k1 = (kt + 1u <= rb);
            vec<bfloat, 8> a1 = kZeroBf8;
            vec<bfloat, 8> b1 = kZeroBf8;
            if (has_k1) {
              const bfloat4 s1_0 = *reinterpret_cast<device const bfloat4 *>(
                  ctrl_solve + ulong(r_base + fm) * blend_detail::kChunkSize + k0 + 16u + fn);
              const bfloat4 s1_1 = *reinterpret_cast<device const bfloat4 *>(
                  ctrl_solve + ulong(r_base + fm + 8u) * blend_detail::kChunkSize + k0 + 16u + fn);
              a1 = vec<bfloat, 8>(s1_0.x, s1_0.y, s1_0.z, s1_0.w,
                                  s1_1.x, s1_1.y, s1_1.z, s1_1.w);
              const bfloat4 r1_0 = *reinterpret_cast<const threadgroup bfloat4 *>(
                  my_work + fm * kWorkPad + k0 + 16u + fn);
              const bfloat4 r1_1 = *reinterpret_cast<const threadgroup bfloat4 *>(
                  my_work + (fm + 8u) * kWorkPad + k0 + 16u + fn);
              b1 = vec<bfloat, 8>(r1_0.x, r1_0.y, r1_0.z, r1_0.w,
                                  r1_1.x, r1_1.y, r1_1.z, r1_1.w);
            }
            nax_mma_16x16x32<bfloat, true>(delta, a0, a1, b0, b1);
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint r = r_base + fm + h * 8u;
          const float suff = (r < clen) ? ctrl_suff[r] : 0.0f;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            my_work[(fn + j) * kWorkPad + r] =
                (r < clen) ? bfloat(suff * delta[h * 4u + j]) : bfloat(0.0f);
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }

      // 4. Update M_sg[16, 128] = decay_last * M_sg + deltaSuffT[16, 64] @ nk[64, 128]
      //    one [16, 32] slice at a time via nax_mma_16x32x16_nn!
      for (uint k0 = 0u; k0 < blend_detail::kKeyDim; k0 += 32u) {
        vec<float, 8> ms0, ms1;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const float4 f0 = *reinterpret_cast<device const float4 *>(
              m_sg + ulong(fm + h * 8u) * blend_detail::kKeyDim + k0 + fn) *
              decay_last;
          const float4 f1 = *reinterpret_cast<device const float4 *>(
              m_sg + ulong(fm + h * 8u) * blend_detail::kKeyDim + k0 + 16u + fn) *
              decay_last;
          ms0[h * 4u + 0u] = f0[0]; ms0[h * 4u + 1u] = f0[1];
          ms0[h * 4u + 2u] = f0[2]; ms0[h * 4u + 3u] = f0[3];
          ms1[h * 4u + 0u] = f1[0]; ms1[h * 4u + 1u] = f1[1];
          ms1[h * 4u + 2u] = f1[2]; ms1[h * 4u + 3u] = f1[3];
        }
        for (uint rb = 0u; rb < n_rblks; ++rb) {
          const uint r_base = rb * 16u;
          const bfloat4 d0 = *reinterpret_cast<const threadgroup bfloat4 *>(
              my_work + fm * kWorkPad + r_base + fn);
          const bfloat4 d1 = *reinterpret_cast<const threadgroup bfloat4 *>(
              my_work + (fm + 8u) * kWorkPad + r_base + fn);
          const vec<bfloat, 8> a_ds(d0.x, d0.y, d0.z, d0.w,
                                    d1.x, d1.y, d1.z, d1.w);
          const bfloat4 n0_0 = *reinterpret_cast<device const bfloat4 *>(
              ws_nk + ulong(r_base + fm) * blend_detail::kKeyDim + k0 + fn);
          const bfloat4 n0_1 = *reinterpret_cast<device const bfloat4 *>(
              ws_nk + ulong(r_base + fm + 8u) * blend_detail::kKeyDim + k0 + fn);
          const bfloat4 n1_0 = *reinterpret_cast<device const bfloat4 *>(
              ws_nk + ulong(r_base + fm) * blend_detail::kKeyDim + k0 + 16u + fn);
          const bfloat4 n1_1 = *reinterpret_cast<device const bfloat4 *>(
              ws_nk + ulong(r_base + fm + 8u) * blend_detail::kKeyDim + k0 + 16u + fn);
          const vec<bfloat, 8> b0(n0_0.x, n0_0.y, n0_0.z, n0_0.w,
                                  n0_1.x, n0_1.y, n0_1.z, n0_1.w);
          const vec<bfloat, 8> b1(n1_0.x, n1_0.y, n1_0.z, n1_0.w,
                                  n1_1.x, n1_1.y, n1_1.z, n1_1.w);
          nax_mma_16x32x16_nn<bfloat>(ms0, ms1, a_ds, b0, b1);
        }
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          *reinterpret_cast<device float4 *>(
              m_sg + ulong(fm + h * 8u) * blend_detail::kKeyDim + k0 + fn) =
              float4(ms0[h * 4u + 0u], ms0[h * 4u + 1u],
                     ms0[h * 4u + 2u], ms0[h * 4u + 3u]);
          *reinterpret_cast<device float4 *>(
              m_sg + ulong(fm + h * 8u) * blend_detail::kKeyDim + k0 + 16u + fn) =
              float4(ms1[h * 4u + 0u], ms1[h * 4u + 1u],
                     ms1[h * 4u + 2u], ms1[h * 4u + 3u]);
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    return;
  }

  // Portable Apple9 (M4) zero-spill FP32 SIMD fallback (exact same recurrence math).
  for (uint c = 0u; c * blend_detail::kChunkSize < params.tokens; ++c) {
    const uint t0 = c * blend_detail::kChunkSize;
    const uint clen = min(blend_detail::kChunkSize, params.tokens - t0);
    device const bfloat *ws_nk =
        qk_ws +
        (ulong(c) * ulong(params.key_heads) + ulong(hk)) *
            blend_detail::kQkPairBfloats;
    device const uchar *ctrl_raw =
        ctrl_ws +
        (ulong(c) * ulong(params.value_heads) + ulong(hv)) *
            blend_detail::kCtrlStrideBytes;
    device const bfloat *ctrl_solve =
        reinterpret_cast<device const bfloat *>(ctrl_raw);
    device const float *ctrl_pref =
        reinterpret_cast<device const float *>(ctrl_raw + blend_detail::kCtrlPrefByteOff);
    device const float *ctrl_suff =
        reinterpret_cast<device const float *>(ctrl_raw + blend_detail::kCtrlSuffByteOff);
    const float decay_last =
        reinterpret_cast<device const float *>(ctrl_raw + blend_detail::kCtrlMetaByteOff)[0];

    for (uint dv = 0u; dv < kValSgDim; ++dv) {
      device float *m_row = m_sg + ulong(dv) * blend_detail::kKeyDim;
      for (uint r = 0u; r < clen; ++r) {
        device const bfloat *nk_r = ws_nk + ulong(r) * blend_detail::kKeyDim;
        float acc = 0.0f;
        for (uint k = lane; k < blend_detail::kKeyDim; k += 32u) {
          acc = fma(float(nk_r[k]), float(bfloat(m_row[k])), acc);
        }
        const float pred = simd_sum(acc);
        if (lane == 0u) {
          my_work[dv * kWorkPad + r] = bfloat(-ctrl_pref[r] * pred);
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (int r_i = int(clen) - 1; r_i >= 0; --r_i) {
        const uint r = uint(r_i);
        float acc = 0.0f;
        for (uint j = lane; j <= r; j += 32u) {
          acc = fma(float(ctrl_solve[r * blend_detail::kChunkSize + j]),
                    float(my_work[dv * kWorkPad + j]), acc);
        }
        const float delta = simd_sum(acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
          my_work[dv * kWorkPad + r] = bfloat(ctrl_suff[r] * delta);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
      for (uint k = lane; k < blend_detail::kKeyDim; k += 32u) {
        float val = m_row[k] * decay_last;
        for (uint r = 0u; r < clen; ++r) {
          val = fma(float(my_work[dv * kWorkPad + r]),
                    float(ws_nk[ulong(r) * blend_detail::kKeyDim + k]), val);
        }
        m_row[k] = val;
      }
    }
  }
}

// Pack FP32 [Hv, 128, 128] operator (M_C or B_C) into BF16 or FP8 E4M3 + per-row FP16 scale.
// Launch: grid = {32, Hv * 128}, tptg = {32}. Static TG SRAM: 0 B.
kernel void omlx_m5_gdn_pack_operator(
    device const float *src_f32 [[buffer(0)]],
    device uchar *dst_data [[buffer(1)]],
    device half *dst_row_scales [[buffer(2)]],
    constant BlendGdnPackOpParams &params [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint flat_row = tg.y;
  if (flat_row >= params.value_heads * params.dim || params.dim != 128u) {
    return;
  }
  device const float4 *row_in =
      reinterpret_cast<device const float4 *>(src_f32 + ulong(flat_row) * 128ul);
  const float4 v = row_in[lane];
  const float local_max = max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w)));
  const float row_max = simd_max(local_max);

  if (params.op_format == 0u) {
    // BF16 with unit FP16 row scale (exact FP32 exponent range).
    device bfloat4 *row_out =
        reinterpret_cast<device bfloat4 *>(dst_data + ulong(flat_row) * 256ul);
    row_out[lane] = bfloat4(v);
    if (lane == 0u) {
      dst_row_scales[flat_row] = half(1.0h);
    }
  } else if (params.op_format == 2u) {
    // High-precision FP16 (10-bit mantissa) with per-row FP16 scale.
    const half s_h = (row_max > 16384.0f) ? half(row_max / 16384.0f) : half(1.0h);
    const float s_f = float(s_h);
    const float inv_s = (s_f > 0.0f) ? precise::divide(1.0f, s_f) : 0.0f;
    device half4 *row_out =
        reinterpret_cast<device half4 *>(dst_data + ulong(flat_row) * 256ul);
    row_out[lane] = half4(clamp(v * inv_s, float4(-65504.0f), float4(65504.0f)));
    if (lane == 0u) {
      dst_row_scales[flat_row] = s_h;
    }
  } else {
    // Compact FP8 E4M3 with per-row FP16 scale = max_abs / 448.0f.
    const half s_h = (row_max > 0.0f) ? half(max(row_max / 448.0f, 6.103515625e-5f))
                                      : half(0.0h);
    const float s_f = float(s_h);
    const float inv_s = (s_f > 0.0f) ? precise::divide(1.0f, s_f) : 0.0f;
    const float4 scaled = v * inv_s;
    device uchar4 *row_out =
        reinterpret_cast<device uchar4 *>(dst_data + ulong(flat_row) * 128ul);
    row_out[lane] = uchar4(
        blend_detail::encode_fp8_e4m3(scaled.x),
        blend_detail::encode_fp8_e4m3(scaled.y),
        blend_detail::encode_fp8_e4m3(scaled.z),
        blend_detail::encode_fp8_e4m3(scaled.w));
    if (lane == 0u) {
      dst_row_scales[flat_row] = s_h;
    }
  }
}

// Compute cumulative log-decay G_t[tokens, Hv] in FP16 for interior HKVD perturbation.
// Launch: grid = {32, value_heads}, tptg = {32} (1 lane scans or 1 thread/head).
kernel void omlx_m5_gdn_compute_cum_decay_f16(
    device const bfloat *a [[buffer(0)]],
    device const bfloat *a_log [[buffer(1)]],
    device const bfloat *dt_bias [[buffer(2)]],
    device half *cum_decay_f16 [[buffer(3)]],
    constant BlendGdnCumDecayParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const uint hv = gid;
  if (hv >= params.value_heads) {
    return;
  }
  const float neg_exp_a = -exp(float(a_log[hv]));
  const float dt_b = float(dt_bias[hv]);
  float g_cum = 0.0f;
  for (uint t = 0u; t < params.tokens; ++t) {
    const ulong idx = ulong(t) * ulong(params.value_heads) + ulong(hv);
    const float act = float(a[idx]) + dt_b;
    const float sp = act > 20.0f ? act : log(1.0f + exp(act));
    const float g = min(0.0f, neg_exp_a * sp);
    g_cum = min(0.0f, g_cum + g);
    cum_decay_f16[idx] = half(clamp(g_cum, -65504.0f, 0.0f));
  }
}

// ============================================================================
// 5. NAX GDN State Composition Kernel (`omlx_m5_gdn_state_compose_nax`)
//    Advances [Hv, 128, 128] FP32 recurrent state across cached chunk interior:
//      S_out = S_in * M_C + B_C + sum_{p in HKVD} exp(G_end - G_{t_p}) *
//              (beta_new * v_new * k_new^T - beta_old * v_old * k_old^T)
//    - Safe for both out-of-place AND in-place (`state_in == state_out`) because
//      row `dv` of `S_out` depends only on row `dv` of `S_in`!
//    - Uses IEEE FP32 NAX 32x32x32 `matmul2d` (`execution_simdgroups<4>`).
//    - Launch: grid = {128, value_heads, 4}, tptg = {128} -> 192 TGs on Hv=48!
//    - Static TG SRAM: 2 * 32 * 32 * 4 B = 8,192 B (8.0 KiB!).
// ============================================================================
[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_state_compose_nax(
    device const float *state_in [[buffer(0)]],
    device const uchar *m_data [[buffer(1)]],
    device const half *m_row_scales [[buffer(2)]],
    device const uchar *b_data [[buffer(3)]],
    device const half *b_row_scales [[buffer(4)]],
    device const half *cum_decay_f16 [[buffer(5)]],
    device const uint *hkvd_t_local [[buffer(6)]],
    device const bfloat *hkvd_k_new [[buffer(7)]],
    device const bfloat *hkvd_v_new [[buffer(8)]],
    device const float *hkvd_beta_new [[buffer(9)]],
    device const bfloat *hkvd_k_old [[buffer(10)]],
    device const bfloat *hkvd_v_old [[buffer(11)]],
    device const float *hkvd_beta_old [[buffer(12)]],
    device float *state_out [[buffer(13)]],
    constant BlendGdnStateComposeParams &params [[buffer(14)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  struct BlendComposeTgStorage {
    float a[32u * 32u]; // 4 KiB
    float b[32u * 32u]; // 4 KiB (total 8.0 KiB!)
  };
  static_assert(sizeof(BlendComposeTgStorage) <= kApple10MaxPrefillTgSramBytes,
                "BlendComposeTgStorage exceeds 16,384 B!");
  alignas(16) threadgroup BlendComposeTgStorage tg_mem;
  threadgroup float *tg_a = tg_mem.a;
  threadgroup float *tg_b = tg_mem.b;

  const uint hv = tg.y;
  const uint row_tile = tg.z; // 0..3 (each tile = 32 rows of Dv=128)
  if (hv >= params.value_heads || row_tile >= 4u || sg >= 4u ||
      params.dim != 128u) {
    return;
  }

  const uint tid = sg * 32u + lane; // 0..127
  const uint dv_base = row_tile * 32u;
  const ulong head_mat_off = ulong(hv) * 128ul * 128ul;
  const ulong head_row_off = ulong(hv) * 128ul;
  device const float *s_in_hv = state_in + head_mat_off;
  device float *s_out_hv = state_out + head_mat_off;

  // Each of the 4 SIMDgroups owns 8 rows x 128 cols via 16 simdgroup_float8x8
  // tiles (32 floats/lane total -> 0 cooperative_tensor allocas, 0 B spills,
  // 100% in-place `state_in == state_out` safe!).
  simdgroup_float8x8 acc[16];
  #pragma clang loop unroll(full)
  for (uint t = 0u; t < 16u; ++t) {
    acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  }

  const uint ld_row = tid >> 2u;        // 0..31
  const uint ld_col8 = (tid & 3u) * 8u; // 0, 8, 16, 24
  const uint sg_row_off = sg * 8u;

  for (uint k0 = 0u; k0 < 128u; k0 += 32u) {
    device const float4 *a_src = reinterpret_cast<device const float4 *>(
        s_in_hv + ulong(dv_base + ld_row) * 128ul + k0 + ld_col8);
    *reinterpret_cast<threadgroup float4 *>(tg_a + ld_row * 32u + ld_col8) = a_src[0];
    *reinterpret_cast<threadgroup float4 *>(tg_a + ld_row * 32u + ld_col8 + 4u) = a_src[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint nt = 0u; nt < 4u; ++nt) {
      const uint n0 = nt * 32u;
      const uint m_row = k0 + ld_row;
      const ulong m_row_flat = head_row_off + m_row;
      const float rs = float(m_row_scales[m_row_flat]);
      if (params.op_format == 0u) {
        device const bfloat4 *m_bf = reinterpret_cast<device const bfloat4 *>(
            m_data + (head_mat_off + ulong(m_row) * 128ul + n0 + ld_col8) * 2ul);
        *reinterpret_cast<threadgroup float4 *>(tg_b + ld_row * 32u + ld_col8) =
            float4(m_bf[0]) * rs;
        *reinterpret_cast<threadgroup float4 *>(tg_b + ld_row * 32u + ld_col8 + 4u) =
            float4(m_bf[1]) * rs;
      } else if (params.op_format == 2u) {
        device const half4 *m_hf = reinterpret_cast<device const half4 *>(
            m_data + (head_mat_off + ulong(m_row) * 128ul + n0 + ld_col8) * 2ul);
        *reinterpret_cast<threadgroup float4 *>(tg_b + ld_row * 32u + ld_col8) =
            float4(m_hf[0]) * rs;
        *reinterpret_cast<threadgroup float4 *>(tg_b + ld_row * 32u + ld_col8 + 4u) =
            float4(m_hf[1]) * rs;
      } else {
        device const uchar4 *m_u8 = reinterpret_cast<device const uchar4 *>(
            m_data + head_mat_off + ulong(m_row) * 128ul + n0 + ld_col8);
        const uchar4 q0 = m_u8[0];
        const uchar4 q1 = m_u8[1];
        *reinterpret_cast<threadgroup float4 *>(tg_b + ld_row * 32u + ld_col8) =
            float4(blend_detail::decode_fp8_e4m3(q0.x),
                   blend_detail::decode_fp8_e4m3(q0.y),
                   blend_detail::decode_fp8_e4m3(q0.z),
                   blend_detail::decode_fp8_e4m3(q0.w)) *
            rs;
        *reinterpret_cast<threadgroup float4 *>(tg_b + ld_row * 32u + ld_col8 + 4u) =
            float4(blend_detail::decode_fp8_e4m3(q1.x),
                   blend_detail::decode_fp8_e4m3(q1.y),
                   blend_detail::decode_fp8_e4m3(q1.z),
                   blend_detail::decode_fp8_e4m3(q1.w)) *
            rs;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      #pragma clang loop unroll(full)
      for (uint ks = 0u; ks < 4u; ++ks) {
        simdgroup_float8x8 ma;
        simdgroup_load(ma, tg_a + sg_row_off * 32u + ks * 8u, 32u);
        #pragma clang loop unroll(full)
        for (uint ns = 0u; ns < 4u; ++ns) {
          simdgroup_float8x8 mb;
          simdgroup_load(mb, tg_b + (ks * 8u) * 32u + ns * 8u, 32u);
          simdgroup_multiply_accumulate(acc[nt * 4u + ns], ma, mb,
                                        acc[nt * 4u + ns]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // Finalize each 32x32 N-tile via tg_a (reused as [32x32] FP32 output stage),
  // adding optional B_C + decayed rank-1 HKVD perturbations, then write S_out.
  const uint repeat = params.value_heads / max(1u, params.key_heads);
  const uint hk = hv / repeat;
  const float g_end =
      (params.num_hkvd_perturb > 0u && params.chunk_tokens > 0u)
          ? float(cum_decay_f16[ulong(params.chunk_tokens - 1u) *
                                    ulong(params.value_heads) +
                                ulong(hv)])
          : 0.0f;

  for (uint nt = 0u; nt < 4u; ++nt) {
    const uint n0 = nt * 32u;
    #pragma clang loop unroll(full)
    for (uint ns = 0u; ns < 4u; ++ns) {
      simdgroup_store(acc[nt * 4u + ns], tg_a + sg_row_off * 32u + ns * 8u, 32u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint dv = dv_base + ld_row;
    const float bs =
        (params.apply_bias != 0u) ? float(b_row_scales[head_row_off + dv]) : 0.0f;
    for (uint j = 0u; j < 8u; ++j) {
      const uint dk = n0 + ld_col8 + j;
      float val = tg_a[ld_row * 32u + ld_col8 + j];
      if (params.apply_bias != 0u) {
        const ulong elem_idx = head_mat_off + ulong(dv) * 128ul + dk;
        if (params.op_format == 0u) {
          const bfloat b_val =
              reinterpret_cast<device const bfloat *>(b_data)[elem_idx];
          val = fma(float(b_val), bs, val);
        } else if (params.op_format == 2u) {
          const half b_val =
              reinterpret_cast<device const half *>(b_data)[elem_idx];
          val = fma(float(b_val), bs, val);
        } else {
          val = fma(blend_detail::decode_fp8_e4m3(b_data[elem_idx]), bs, val);
        }
      }
      for (uint p = 0u; p < params.num_hkvd_perturb; ++p) {
        const uint t_loc = hkvd_t_local[p];
        const float g_t =
            float(cum_decay_f16[ulong(t_loc) * ulong(params.value_heads) +
                                ulong(hv)]);
        const float w_dec = exp(min(0.0f, g_end - g_t));
        const ulong hv_off = ulong(p) * ulong(params.value_heads) + ulong(hv);
        const ulong hk_off = ulong(p) * ulong(params.key_heads) + ulong(hk);
        const float bn = hkvd_beta_new[hv_off];
        const float vn = float(hkvd_v_new[hv_off * 128ul + dv]);
        const float kn = float(hkvd_k_new[hk_off * 128ul + dk]);
        const float bo = hkvd_beta_old[hv_off];
        const float vo = float(hkvd_v_old[hv_off * 128ul + dv]);
        const float ko = float(hkvd_k_old[hk_off * 128ul + dk]);
        val = fma(w_dec, fma(bn * vn, kn, -(bo * vo * ko)), val);
      }
      s_out_hv[ulong(dv) * 128ul + dk] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ============================================================================
// 6. Compact `int8_g128` Quantize / Dequantize (for Layer-3 HKVD Probe Planes)
//    Launch: grid = {32, tokens * (width / 128)}, tptg = {32}. 0 B TG SRAM.
// ============================================================================
kernel void omlx_m5_blend_quantize_int8_g128_bf16(
    device const bfloat *src [[buffer(0)]],
    device char *dst_i8 [[buffer(1)]],
    device half *dst_scales [[buffer(2)]],
    constant BlendInt8QuantParams &params [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint groups_per_row = params.width / 128u;
  const uint g_flat = tg.y;
  if (groups_per_row == 0u || g_flat >= params.tokens * groups_per_row) {
    return;
  }
  const ulong elem_base = ulong(g_flat) * 128ul + ulong(lane) * 4ul;
  const float4 v =
      float4(*reinterpret_cast<device const bfloat4 *>(src + elem_base));
  const float lmax = max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w)));
  const float gmax = simd_max(lmax);
  const half s_h = (gmax > 0.0f) ? half(max(gmax / 127.0f, 6.103515625e-5f))
                                 : half(0.0h);
  const float s_f = float(s_h);
  const float inv_s = (s_f > 0.0f) ? precise::divide(1.0f, s_f) : 0.0f;
  const float4 q = clamp(rint(v * inv_s), float4(-127.0f), float4(127.0f));
  *reinterpret_cast<device char4 *>(dst_i8 + elem_base) =
      char4(char(int(q.x)), char(int(q.y)), char(int(q.z)), char(int(q.w)));
  if (lane == 0u) {
    dst_scales[g_flat] = s_h;
  }
}

kernel void omlx_m5_blend_dequantize_int8_g128_bf16(
    device const char *src_i8 [[buffer(0)]],
    device const half *src_scales [[buffer(1)]],
    device bfloat *dst [[buffer(2)]],
    constant BlendInt8QuantParams &params [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint groups_per_row = params.width / 128u;
  const uint g_flat = tg.y;
  if (groups_per_row == 0u || g_flat >= params.tokens * groups_per_row) {
    return;
  }
  const float s = float(src_scales[g_flat]);
  const ulong elem_base = ulong(g_flat) * 128ul + ulong(lane) * 4ul;
  const char4 q = *reinterpret_cast<device const char4 *>(src_i8 + elem_base);
  *reinterpret_cast<device bfloat4 *>(dst + elem_base) =
      bfloat4(float4(float(q.x), float(q.y), float(q.z), float(q.w)) * s);
}

// ============================================================================
// 7. Layer-3 HKVD L2 Deviation Scoring + Exact Sorted Top-K Selection
// ============================================================================

// Parallel L2 score pass (128 threads/TG, 1 TG per token) for large prompts.
// Static TG SRAM: 16 B (`tg_sg[4]`).
[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_blend_hkvd_l2_score(
    device const bfloat *mix_stitched [[buffer(0)]],
    device const uchar *mix_cached_raw [[buffer(1)]],
    device const half *mix_cached_scales [[buffer(2)]],
    device const uint *token_flags [[buffer(3)]],
    device float *l2_scores [[buffer(4)]],
    constant BlendHkvdSelectParams &params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float tg_sg[4];
  const uint t = tg.y;
  if (t >= params.total_tokens || sg >= 4u) {
    return;
  }
  const uint flag = token_flags[t];
  if (flag != blend_detail::kTokenInteriorCandidate) {
    if (sg == 0u && lane == 0u) {
      l2_scores[t] = (flag == blend_detail::kTokenMandatory) ? INFINITY : 0.0f;
    }
    return;
  }

  const uint tid = sg * 32u + lane;
  const ulong row_off = ulong(t) * ulong(params.mix_width);
  const ulong scale_row_off = ulong(t) * ulong(params.mix_width / 128u);
  float acc = 0.0f;

  if (params.cached_is_int8 != 0u) {
    device const char *cached_i8 =
        reinterpret_cast<device const char *>(mix_cached_raw) + row_off;
    device const half *scales = mix_cached_scales + scale_row_off;
    for (uint d4 = tid * 4u; d4 < params.mix_width; d4 += 512u) {
      const float4 st = float4(
          *reinterpret_cast<device const bfloat4 *>(mix_stitched + row_off + d4));
      const char4 ci =
          *reinterpret_cast<device const char4 *>(cached_i8 + d4);
      const float sc = float(scales[d4 >> 7u]);
      const float4 diff =
          st - float4(float(ci.x), float(ci.y), float(ci.z), float(ci.w)) * sc;
      acc += dot(diff, diff);
    }
  } else {
    device const bfloat *cached_bf =
        reinterpret_cast<device const bfloat *>(mix_cached_raw) + row_off;
    for (uint d4 = tid * 4u; d4 < params.mix_width; d4 += 512u) {
      const float4 st = float4(
          *reinterpret_cast<device const bfloat4 *>(mix_stitched + row_off + d4));
      const float4 cb =
          float4(*reinterpret_cast<device const bfloat4 *>(cached_bf + d4));
      const float4 diff = st - cb;
      acc += dot(diff, diff);
    }
  }

  const float sg_sum = simd_sum(acc);
  if (lane == 0u) {
    tg_sg[sg] = sg_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane == 0u) {
    const float total_sq = (tg_sg[0] + tg_sg[1]) + (tg_sg[2] + tg_sg[3]);
    l2_scores[t] = sqrt(max(0.0f, total_sq));
  }
}

// Exact HKVD Top-K Selection -> Strictly Ascending `active_idx` & `active_pos`!
// - Supports BOTH:
//   (a) `compute_l2_inline == 0`: consumes precomputed `l2_scores` from
//       `omlx_m5_blend_hkvd_l2_score` (O(T) across arbitrary prompt lengths up to 256K).
//   (b) `compute_l2_inline == 1`: computes `||Mix_stitched - Mix_cached||_2` inline
//       in a single dispatch before selecting top-k!
// - Launch: grid = {256, 1, 1}, tptg = {256, 1, 1} (1 TG, ~2.5 KiB static TG SRAM).
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_blend_hkvd_topk_select(
    device const bfloat *mix_stitched [[buffer(0)]],
    device const uchar *mix_cached_raw [[buffer(1)]],
    device const half *mix_cached_scales [[buffer(2)]],
    device const uint *token_flags [[buffer(3)]],
    device const uint *prompt_positions [[buffer(4)]],
    device float *l2_scores [[buffer(5)]],
    device uint *active_idx [[buffer(6)]],
    device uint *active_pos [[buffer(7)]],
    device uint *active_count [[buffer(8)]],
    constant BlendHkvdSelectParams &params [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint tg_hist[256];
  threadgroup uint tg_sg_counts[8];
  threadgroup uint tg_sg_ties[8];
  threadgroup uint tg_shared[8]; // [0]=n_cand, [1]=k_rem, [2]=pref_val, [3]=pref_mask, [4]=ties_rem, [5]=out_base

  const uint tid = sg * 32u + lane;
  const uint total_tokens = params.total_tokens;

  // Optional inline L2 deviation computation (single-dispatch path).
  if (params.compute_l2_inline != 0u) {
    for (uint t = tid; t < total_tokens; t += 256u) {
      const uint flag = token_flags[t];
      if (flag != blend_detail::kTokenInteriorCandidate) {
        l2_scores[t] = (flag == blend_detail::kTokenMandatory) ? INFINITY : 0.0f;
        continue;
      }
      const ulong row_off = ulong(t) * ulong(params.mix_width);
      const ulong scale_row_off = ulong(t) * ulong(params.mix_width / 128u);
      float sq = 0.0f;
      if (params.cached_is_int8 != 0u) {
        device const char *cached_i8 =
            reinterpret_cast<device const char *>(mix_cached_raw) + row_off;
        device const half *scales = mix_cached_scales + scale_row_off;
        for (uint d = 0u; d < params.mix_width; d += 4u) {
          const float4 st = float4(
              *reinterpret_cast<device const bfloat4 *>(mix_stitched + row_off + d));
          const char4 ci =
              *reinterpret_cast<device const char4 *>(cached_i8 + d);
          const float sc = float(scales[d >> 7u]);
          const float4 diff =
              st - float4(float(ci.x), float(ci.y), float(ci.z), float(ci.w)) * sc;
          sq += dot(diff, diff);
        }
      } else {
        device const bfloat *cached_bf =
            reinterpret_cast<device const bfloat *>(mix_cached_raw) + row_off;
        for (uint d = 0u; d < params.mix_width; d += 4u) {
          const float4 st = float4(
              *reinterpret_cast<device const bfloat4 *>(mix_stitched + row_off + d));
          const float4 cb =
              float4(*reinterpret_cast<device const bfloat4 *>(cached_bf + d));
          const float4 diff = st - cb;
          sq += dot(diff, diff);
        }
      }
      l2_scores[t] = sqrt(max(0.0f, sq));
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }

  // 1. Count interior candidates across [0, total_tokens).
  uint local_cand = 0u;
  for (uint t = tid; t < total_tokens; t += 256u) {
    if (token_flags[t] == blend_detail::kTokenInteriorCandidate) {
      ++local_cand;
    }
  }
  const uint sg_cand = simd_sum(local_cand);
  if (lane == 0u) {
    tg_sg_counts[sg] = sg_cand;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    uint n_cand = 0u;
    for (uint s = 0u; s < 8u; ++s) {
      n_cand += tg_sg_counts[s];
    }
    uint k_sel = 0u;
    if (n_cand > 0u) {
      if (params.explicit_hkvd_k > 0u) {
        k_sel = min(n_cand, params.explicit_hkvd_k);
      } else if (params.hkvd_ratio > 0.0f) {
        k_sel = min(n_cand, uint(ceil(float(n_cand) * params.hkvd_ratio)));
      }
    }
    tg_shared[0] = n_cand;
    tg_shared[1] = k_sel;
    tg_shared[2] = 0u; // prefix_val
    tg_shared[3] = 0u; // prefix_mask
    tg_shared[4] = 0u; // ties_rem
    tg_shared[5] = 0u; // out_base
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint n_cand = tg_shared[0];
  const uint k_sel_total = tg_shared[1];

  // 2. Exact 4-pass 8-bit MSB radix select for the k_sel_total-th largest L2 key.
  if (k_sel_total > 0u && k_sel_total < n_cand) {
    for (uint pass = 0u; pass < 4u; ++pass) {
      const uint shift = 24u - pass * 8u;
      atomic_store_explicit(&tg_hist[tid], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const uint pref_val = tg_shared[2];
      const uint pref_mask = tg_shared[3];
      for (uint t = tid; t < total_tokens; t += 256u) {
        if (token_flags[t] == blend_detail::kTokenInteriorCandidate) {
          const uint key = blend_detail::float_nonneg_to_sortable_u32(l2_scores[t]);
          if ((key & pref_mask) == pref_val) {
            const uint bucket = (key >> shift) & 0xFFu;
            atomic_fetch_add_explicit(&tg_hist[bucket], 1u, memory_order_relaxed);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (tid == 0u) {
        uint rem = tg_shared[1];
        uint chosen = 0u;
        for (int b = 255; b >= 0; --b) {
          const uint cnt = atomic_load_explicit(&tg_hist[b], memory_order_relaxed);
          if (rem <= cnt) {
            chosen = uint(b);
            break;
          }
          rem -= cnt;
        }
        tg_shared[1] = rem;
        tg_shared[2] = pref_val | (chosen << shift);
        tg_shared[3] = pref_mask | (0xFFu << shift);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
      tg_shared[4] = tg_shared[1]; // exact ties remaining at threshold_key
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint threshold_key = tg_shared[2];
  uint ties_rem = tg_shared[4];
  uint out_base = 0u;

  // 3. Strictly ascending sorted prefix-sum compaction into `active_idx` & `active_pos`.
  for (uint base = 0u; base < total_tokens; base += 256u) {
    const uint t = base + tid;
    bool is_tie = false;
    bool sel_non_tie = false;

    if (t < total_tokens) {
      const uint flag = token_flags[t];
      if (flag == blend_detail::kTokenMandatory) {
        sel_non_tie = true;
      } else if (flag == blend_detail::kTokenInteriorCandidate && k_sel_total > 0u) {
        if (k_sel_total >= n_cand) {
          sel_non_tie = true;
        } else {
          const uint key = blend_detail::float_nonneg_to_sortable_u32(l2_scores[t]);
          if (key > threshold_key) {
            sel_non_tie = true;
          } else if (key == threshold_key) {
            is_tie = true;
          }
        }
      }
    }

    // Prefix sum of exact-threshold ties in this 256-token tile (deterministic index order).
    const uint tie_bit = is_tie ? 1u : 0u;
    const uint tie_lane_off = simd_prefix_exclusive_sum(tie_bit);
    const uint tie_sg_tot = simd_sum(tie_bit);
    if (lane == 0u) {
      tg_sg_ties[sg] = tie_sg_tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tie_sg_base = 0u;
    uint tile_ties = 0u;
    for (uint s = 0u; s < 8u; ++s) {
      if (s < sg) {
        tie_sg_base += tg_sg_ties[s];
      }
      tile_ties += tg_sg_ties[s];
    }
    const uint tie_tile_rank = tie_sg_base + tie_lane_off;
    const bool selected = sel_non_tie || (is_tie && tie_tile_rank < ties_rem);

    // Prefix sum of selected tokens in this 256-token tile.
    const uint sel_bit = selected ? 1u : 0u;
    const uint sel_lane_off = simd_prefix_exclusive_sum(sel_bit);
    const uint sel_sg_tot = simd_sum(sel_bit);
    if (lane == 0u) {
      tg_sg_counts[sg] = sel_sg_tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sel_sg_base = 0u;
    uint tile_sel = 0u;
    for (uint s = 0u; s < 8u; ++s) {
      if (s < sg) {
        sel_sg_base += tg_sg_counts[s];
      }
      tile_sel += tg_sg_counts[s];
    }
    if (selected) {
      const uint out_slot = out_base + sel_sg_base + sel_lane_off;
      active_idx[out_slot] = t;
      active_pos[out_slot] = prompt_positions[t];
    }

    out_base += tile_sel;
    ties_rem = (ties_rem > tile_ties) ? (ties_rem - tile_ties) : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0u) {
    active_count[0] = out_base;
  }
}

// ============================================================================
// 8. Contiguous Row Gather & Scatter (BF16) for Packed `[M_active, H]` Compute
//    Launch: grid = {ceil(width/4) [or width if width%4!=0], max_active_tokens},
//            tptg = {128, 1}
// ============================================================================
kernel void omlx_m5_blend_gather_rows_bf16(
    device const bfloat *src [[buffer(0)]],
    device const uint *active_idx [[buffer(1)]],
    device const uint *active_count [[buffer(2)]],
    device bfloat *dst [[buffer(3)]],
    constant BlendGatherScatterParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint active_n =
      params.use_device_count != 0u ? active_count[0] : params.max_active_tokens;
  const uint row = gid.y;
  if (row >= active_n) {
    return;
  }
  const uint src_row = active_idx[row];
  if ((params.width & 3u) == 0u) {
    const uint c4 = gid.x;
    if (c4 * 4u >= params.width) {
      return;
    }
    const ulong src_off = ulong(src_row) * ulong(params.width) + ulong(c4) * 4ul;
    const ulong dst_off = ulong(row) * ulong(params.width) + ulong(c4) * 4ul;
    *reinterpret_cast<device bfloat4 *>(dst + dst_off) =
        *reinterpret_cast<device const bfloat4 *>(src + src_off);
  } else {
    const uint col = gid.x;
    if (col >= params.width) {
      return;
    }
    dst[ulong(row) * ulong(params.width) + ulong(col)] =
        src[ulong(src_row) * ulong(params.width) + ulong(col)];
  }
}

kernel void omlx_m5_blend_scatter_rows_bf16(
    device const bfloat *src [[buffer(0)]],
    device const uint *active_idx [[buffer(1)]],
    device const uint *active_count [[buffer(2)]],
    device bfloat *dst [[buffer(3)]],
    constant BlendGatherScatterParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint active_n =
      params.use_device_count != 0u ? active_count[0] : params.max_active_tokens;
  const uint row = gid.y;
  if (row >= active_n) {
    return;
  }
  const uint dst_row = active_idx[row];
  if ((params.width & 3u) == 0u) {
    const uint c4 = gid.x;
    if (c4 * 4u >= params.width) {
      return;
    }
    const ulong src_off = ulong(row) * ulong(params.width) + ulong(c4) * 4ul;
    const ulong dst_off = ulong(dst_row) * ulong(params.width) + ulong(c4) * 4ul;
    const bfloat4 v = *reinterpret_cast<device const bfloat4 *>(src + src_off);
    if (params.residual_add != 0u) {
      const bfloat4 cur = *reinterpret_cast<device const bfloat4 *>(dst + dst_off);
      *reinterpret_cast<device bfloat4 *>(dst + dst_off) =
          bfloat4(float4(cur) + float4(v));
    } else {
      *reinterpret_cast<device bfloat4 *>(dst + dst_off) = v;
    }
  } else {
    const uint col = gid.x;
    if (col >= params.width) {
      return;
    }
    const ulong dst_off = ulong(dst_row) * ulong(params.width) + ulong(col);
    const bfloat v = src[ulong(row) * ulong(params.width) + ulong(col)];
    dst[dst_off] =
        (params.residual_add != 0u) ? bfloat(float(dst[dst_off]) + float(v)) : v;
  }
}

// ============================================================================
// 9. Scattered QK-Norm + RoPE(active_pos[i]) + Affine4Group128 KV Scatter
//    Writes recomputed K_active, V_active directly into KvPagePool at arbitrary
//    scattered logical positions `active_pos[i]`, while keeping Q_active
//    contiguous `[M_active, query_heads, head_dim]` for unchanged Route B!
//    Launch: grid = {32, max_active_tokens * (query_heads + key_value_heads)},
//            tptg = {32}. Static TG SRAM: 512 B (`tg_vec[256]`).
// ============================================================================
kernel void omlx_m5_attention_qk_norm_rope_kv_scatter_affine4_g128(
    device bfloat *query [[buffer(0)]],
    device bfloat *key [[buffer(1)]],
    device const bfloat *value [[buffer(2)]],
    device const bfloat *query_norm [[buffer(3)]],
    device const bfloat *key_norm [[buffer(4)]],
    device uchar *cache_key [[buffer(5)]],
    device uchar *cache_value [[buffer(6)]],
    device const uint *page_table [[buffer(7)]],
    device const uint *active_pos [[buffer(8)]],
    device const uint *active_count [[buffer(9)]],
    constant BlendQkNormRopeScatterParams &params [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  threadgroup bfloat tg_vec[256];

  const uint heads_per_token = params.query_heads + params.key_value_heads;
  const uint logical_head = tg_position.y;
  const uint token = logical_head / heads_per_token;
  const uint active_n =
      params.use_device_count != 0u ? active_count[0] : params.max_active_tokens;
  if (token >= active_n || (params.head_dim != 256u && params.head_dim != 128u)) {
    return;
  }

  const uint local_head = logical_head - token * heads_per_token;
  const bool is_query = local_head < params.query_heads;
  const uint head = is_query ? local_head : (local_head - params.query_heads);
  device bfloat *row =
      is_query ? query + ulong(token) * params.query_token_stride +
                     ulong(head) * params.query_head_stride
               : key + ulong(token) * params.key_token_stride +
                     ulong(head) * params.key_head_stride;
  device const bfloat *norm = is_query ? query_norm : key_norm;

  const bool has_qk_norm = (params.epsilon >= 0.0f);
  float inverse_rms = 1.0f;
  if (has_qk_norm) {
    float square_sum = 0.0f;
    for (uint column = lane; column < params.head_dim; column += 32u) {
      const float val = float(row[column]);
      square_sum = fma(val, val, square_sum);
    }
    inverse_rms = precise::rsqrt(
        precise::divide(simd_sum(square_sum), float(params.head_dim)) +
        params.epsilon);
  }

  const uint half_rotary = params.rotary_dim / 2u;
  const uint logical_pos = active_pos[token];
  const float position = float(logical_pos);

  for (uint pair = lane; pair < half_rotary; pair += 32u) {
    const uint first_index = pair;
    const uint second_index = pair + half_rotary;
    const float first =
        has_qk_norm ? (float(row[first_index]) * inverse_rms *
                       (float(norm[first_index]) + params.weight_offset))
                    : float(row[first_index]);
    const float second =
        has_qk_norm ? (float(row[second_index]) * inverse_rms *
                       (float(norm[second_index]) + params.weight_offset))
                    : float(row[second_index]);
    const float exponent =
        precise::divide(float(2u * pair), float(params.rotary_dim));
    const float angle =
        precise::divide(position, precise::pow(params.theta, exponent));
    const float cosine = precise::cos(angle);
    const float sine = precise::sin(angle);
    const bfloat o0 = bfloat(fma(-second, sine, first * cosine));
    const bfloat o1 = bfloat(fma(first, sine, second * cosine));
    row[first_index] = o0;
    row[second_index] = o1;
    if (!is_query) {
      tg_vec[first_index] = o0;
      tg_vec[second_index] = o1;
    }
  }
  for (uint column = params.rotary_dim + lane; column < params.head_dim;
       column += 32u) {
    const bfloat normed =
        has_qk_norm ? bfloat(float(row[column]) * inverse_rms *
                             (float(norm[column]) + params.weight_offset))
                    : row[column];
    row[column] = normed;
    if (!is_query) {
      tg_vec[column] = normed;
    }
  }

  if (!is_query) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint logical_page = logical_pos / params.page_tokens;
    const uint within = logical_pos - logical_page * params.page_tokens;
    const uint phys_page = page_table[logical_page];
    const ulong page_bytes = blend_detail::page_byte_stride(
        params.page_tokens, params.key_value_heads, params.head_dim);
    const ulong head_code_bytes =
        ulong(params.page_tokens) * ulong(params.head_dim / 2u);
    const ulong all_codes_bytes =
        ulong(params.key_value_heads) * head_code_bytes;

    device uchar *k_page = cache_key + ulong(phys_page) * page_bytes;
    device uchar *v_page = cache_value + ulong(phys_page) * page_bytes;
    device const bfloat *v_src =
        value + ulong(token) * params.value_token_stride +
        ulong(head) * params.head_dim;

    blend_detail::quantize_vec_g128_tg(
        tg_vec, k_page + ulong(head) * head_code_bytes,
        reinterpret_cast<device KvAffine4VecMeta *>(k_page + all_codes_bytes) +
            ulong(head) * params.page_tokens,
        params.page_tokens, within, params.head_dim, lane);
    blend_detail::quantize_vec_g128_device(
        v_src, v_page + ulong(head) * head_code_bytes,
        reinterpret_cast<device KvAffine4VecMeta *>(v_page + all_codes_bytes) +
            ulong(head) * params.page_tokens,
        params.page_tokens, within, params.head_dim, lane);
  }
}
