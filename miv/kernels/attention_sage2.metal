// Module 2 Route B.1: SageAttention2 Prefill Kernels (Metal 4.1 / MacOSX27.2 SDK)
//   - Compiled with -mmacosx-version-min=27.0 -std=metal4.1 (unlocks metal_fp8_e4m3_format
//     & metal_fp8_ue8m0_format!).
//   - Built on Lossless Base 2.1 (Multi-Tier GQA Packing WM in {4, 6, 8}) +
//     Base 2.2 (LPT Reverse q_blk + kvh-minor) + CacheBlend explicit positions.
//   - Post-RoPE per-tile channel-mean smoothing mu_{K,b,p} with strict r >= valid_k
//     zeroing (blocks unwritten page-tail NaN/Inf poisoning), exact +Delta_{i,b,p}
//     scalar shift restoration BEFORE causal masking, sink-safe per-token s_{K,j,p}
//     INT8 x INT8 -> INT32 (16x32x128 NAX matmul2d).
//   - Dual P x V fast paths behind template bool UseMxfp8PV:
//       (a) UseMxfp8PV == true:  P * 448.0f (fp8_e4m3) x V^T [128, 32] MXFP8
//                                (tensor_blockwise<..., ue8m0, 32, 1>) -> float
//       (b) UseMxfp8PV == false: full-rate FP16 (affine4) / BF16 (bf16) NAX P x V -> float
//   - Optional Route B.2 MInference sparse block indexing (SparseMInference == true).
//   - Threadgroup SRAM strictly <= 27.0 KiB (<= 28.5 KiB <= 32.0 KiB hardware cap).
#include <metal_simdgroup>
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#if !defined(__HAVE_METAL_FP8_E4M3_FORMAT_TYPE__) || !defined(__HAVE_METAL_FP8_UE8M0_FORMAT_TYPE__)
#error "attention_sage2.metal requires Metal 4.1 (-std=metal4.1 -mmacosx-version-min=27.0)"
#endif

using namespace metal;
using namespace mpp::tensor_ops;

#define SAGE2_UNROLL _Pragma("clang loop unroll(full)")
#define SAGE2_ALWAYS_INLINE __attribute__((always_inline))

struct PagedAttentionParams {
  uint query_heads;
  uint key_value_heads;
  uint head_dim;
  uint tokens;
  uint cache_start;
  uint cache_capacity;
  uint page_tokens;
  uint page_count;
  uint query_token_stride;
  uint query_head_stride;
  uint key_token_stride;
  uint value_token_stride;
  uint gate_token_stride;
  uint gate_head_stride;
};

struct Sage2AttentionParams {
  PagedAttentionParams base;
  uint max_q_blocks;
  uint active_stride;
  uint flags;
  uint reserved;
};

struct KvAffine4VecMeta {
  half scale[4];
  half zero[4];
};

namespace sage2 {

constant constexpr float kNegInf = -3.402823466e+38f;
constant constexpr float kNegInfHalf = -1.701411733e+38f;
constant constexpr float kLog2e = 1.4426950408889634f;
constant constexpr float kFp8E4M3Max = 448.0f;
constant constexpr float kInvFp8E4M3Max = 1.0f / 448.0f;

using TgFp8Plain =
    tensor<threadgroup metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline>;
using TgUe8ScaleTag =
    tensor_blockwise<tensor_plane_scales, threadgroup metal_fp8_ue8m0_format, 32, 1>;
using TgMxfp8VT =
    tensor<threadgroup metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline, TgUe8ScaleTag>;

SAGE2_ALWAYS_INLINE
static inline bfloat4 clean_bf4(bfloat4 v) {
  const ushort4 bits = as_type<ushort4>(v);
  return select(v, bfloat4(bfloat(0.0f)),
                (bits & ushort4(0x7fffu)) > ushort4(0x7f80u));
}

SAGE2_ALWAYS_INLINE
static inline float4 clean_f4(float4 v) {
  const uint4 bits = as_type<uint4>(v);
  // Filter both NaN and Inf safely even under -ffast-math.
  return select(v, float4(0.0f), (bits & uint4(0x7f800000u)) == uint4(0x7f800000u));
}

SAGE2_ALWAYS_INLINE
static inline void fast_divmod_g(uint r, uint g, thread uint &q,
                                 thread uint &rem) {
  if (g == 6u) {
    q = (r * 43u) >> 8u;
    rem = r - q * 6u;
  } else if (g == 8u) {
    q = r >> 3u;
    rem = r & 7u;
  } else if (g == 4u) {
    q = r >> 2u;
    rem = r & 3u;
  } else if (g == 2u) {
    q = r >> 1u;
    rem = r & 1u;
  } else if (g == 1u) {
    q = r;
    rem = 0u;
  } else {
    q = r / g;
    rem = r - q * g;
  }
}

SAGE2_ALWAYS_INLINE
static inline ulong affine4_page_byte_stride(uint page_tokens, uint kv_heads,
                                             uint head_dim) {
  const ulong vec_bytes = ulong(head_dim / 2u) + 16ul;
  const ulong head_bytes = ulong(page_tokens) * vec_bytes + 32ul;
  return ulong(kv_heads) * head_bytes;
}

// Compute UE8M0 scale exponent e in [1, 253] such that 2^(e - 127) >= vmax / 448.0f
// and exact power-of-two inverse scale inv_s = 2^(127 - e).
SAGE2_ALWAYS_INLINE
static inline uchar compute_ue8m0_scale(float vmax, thread float &inv_scale) {
  if (!(vmax > 0.0f)) {
    inv_scale = 0.0f;
    return uchar(127u);
  }
  const float needed = vmax * kInvFp8E4M3Max;
  const uint bits = as_type<uint>(needed);
  uint exp = (bits >> 23u) & 0xFFu;
  const uint mant = bits & 0x7FFFFFu;
  if (mant != 0u && exp < 253u) {
    ++exp;
  }
  exp = clamp(exp, 1u, 253u);
  inv_scale = as_type<float>((254u - exp) << 23u);
  return uchar(exp);
}

// Unpack a single 4-channel vector of K or V at row r (0..31), d_part p (0..1),
// col_v4 (0..31). Strictly returns 0.0f without touching device memory when r >= valid_k!
template <bool IsAffine4>
SAGE2_ALWAYS_INLINE
static inline float4 load_kv_vec4_safe(
    device const uchar *cache_raw, uint phys_page, uint sub_off,
    uint page_tokens, ulong page_bytes, ulong kvh_code_base,
    ulong all_codes_bytes, ulong bf16_kv_stride, uint kvh, uint p_g, uint r,
    uint col_v4, short valid_k) {
  if (short(r) >= valid_k) {
    return float4(0.0f);
  }
  if constexpr (IsAffine4) {
    const uint within = sub_off + r;
    device const uchar *page_ptr = cache_raw + ulong(phys_page) * page_bytes;
    device const ushort *codes = reinterpret_cast<device const ushort *>(
        page_ptr + kvh_code_base + ulong(p_g) * (64ul * 64ul));
    device const KvAffine4VecMeta *meta =
        reinterpret_cast<device const KvAffine4VecMeta *>(page_ptr +
                                                          all_codes_bytes) +
        ulong(kvh) * page_tokens;
    const ushort pk = codes[within * 32u + col_v4];
    const float s = float(meta[within].scale[p_g]);
    const float z = float(meta[within].zero[p_g]);
    const float4 q4 = float4(float(pk & 0xFu), float((pk >> 4u) & 0xFu),
                             float((pk >> 8u) & 0xFu), float(pk >> 12u));
    return clean_f4(fma(q4, float4(s), float4(z)));
  } else {
    device const bfloat *cache_bf =
        reinterpret_cast<device const bfloat *>(cache_raw);
    const ulong phys_row = ulong(phys_page) * page_tokens + sub_off + r;
    device const bfloat4 *row_v4 = reinterpret_cast<device const bfloat4 *>(
        cache_bf + phys_row * bf16_kv_stride + ulong(kvh) * 256ul + p_g * 128u);
    return float4(clean_bf4(row_v4[col_v4]));
  }
}

// Cooperatively unpack K[kb], zero-mask r >= valid_k, compute exact per-tile
// channel mean mu_{K,b,p} -> tg_mu_k[2][128], subtract mu, compute per-token
// s_{K,j,p} -> tg_sk[2][32], and quantize K_tilde -> tg_k_i8[2][32][128].
// Uses first 256 threads (r_chunk in 0..3, p in 0..1, col_v4 in 0..31);
// all 32 lanes of each SIMDgroup share (r_chunk, p) for 0-barrier simd_max!
template <bool IsAffine4>
SAGE2_ALWAYS_INLINE
static inline void prepare_k_smoothed_int8(
    device const uchar *cache_key, uint phys_page, uint sub_off,
    uint page_tokens, ulong page_bytes, ulong kvh_code_base,
    ulong all_codes_bytes, ulong bf16_kv_stride, uint kvh, short valid_k,
    uint flat_tid, threadgroup uint4 *tg_stage, threadgroup float *tg_mu_k,
    threadgroup float *tg_sk) {
  // Layout inside tg_stage during prepare_k:
  //   [0 .. 8191]:    tg_k_i8[2][32][128] (8,192 B)
  //   [8192 .. 12287]: tg_mu_tmp[4][64] float4 (4,096 B, dead before tg_q_stage!)
  threadgroup float4 *tg_mu_tmp =
      reinterpret_cast<threadgroup float4 *>(tg_stage) + 512u; // offset 8192 B
  threadgroup char4 *tg_k_i8_v4 =
      reinterpret_cast<threadgroup char4 *>(tg_stage); // offset 0 B

  float4 k_reg[8];
  const bool active = (flat_tid < 256u);
  const uint r_chunk = flat_tid >> 6u;       // 0..3
  const uint v4_id = flat_tid & 63u;         // 0..63
  const uint p_g = v4_id >> 5u;              // 0..1
  const uint col_v4 = v4_id & 31u;           // 0..31 (= lane!)

  if (active) {
    float4 sum4(0.0f);
    SAGE2_UNROLL
    for (uint rr = 0u; rr < 8u; ++rr) {
      const uint r = r_chunk * 8u + rr;
      k_reg[rr] = load_kv_vec4_safe<IsAffine4>(
          cache_key, phys_page, sub_off, page_tokens, page_bytes, kvh_code_base,
          all_codes_bytes, bf16_kv_stride, kvh, p_g, r, col_v4, valid_k);
      sum4 += k_reg[rr];
    }
    tg_mu_tmp[r_chunk * 64u + v4_id] = sum4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (flat_tid < 64u) {
    const float inv_vk = 1.0f / float(max(int(valid_k), 1));
    const float4 total = (tg_mu_tmp[0u * 64u + flat_tid] +
                          tg_mu_tmp[1u * 64u + flat_tid]) +
                         (tg_mu_tmp[2u * 64u + flat_tid] +
                          tg_mu_tmp[3u * 64u + flat_tid]);
    reinterpret_cast<threadgroup float4 *>(tg_mu_k)[flat_tid] = total * inv_vk;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (active) {
    const float4 mu4 = reinterpret_cast<threadgroup float4 *>(tg_mu_k)[v4_id];
    SAGE2_UNROLL
    for (uint rr = 0u; rr < 8u; ++rr) {
      const uint r = r_chunk * 8u + rr;
      const float4 kt = (short(r) < valid_k) ? (k_reg[rr] - mu4) : float4(0.0f);
      const float4 ak = fabs(kt);
      const float lane_max = max(max(ak.x, ak.y), max(ak.z, ak.w));
      const float row_max = simd_max(lane_max);
      const float sk = (row_max > 0.0f) ? (row_max * (1.0f / 127.0f)) : 0.0f;
      const float inv_sk = (row_max > 0.0f) ? (127.0f / row_max) : 0.0f;
      if (col_v4 == 0u) {
        tg_sk[p_g * 32u + r] = sk;
      }
      const float4 qf = clamp(rint(kt * inv_sk), float4(-127.0f), float4(127.0f));
      const char4 kq(char(int(qf.x)), char(int(qf.y)), char(int(qf.z)),
                     char(int(qf.w)));
      tg_k_i8_v4[p_g * 1024u + r * 32u + col_v4] = kq;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Cooperatively unpack V[kb] -> transposed MXFP8 V^T [2][128][32] + UE8M0 scales [2][128].
// Places tg_vt_fp8 at tg_stage[0..8191], tg_vt_sc at [8192..8447],
// temporary vmax at [8448..12543]; leaves tg_p_fp8 at [16384..20479] untouched!
template <bool IsAffine4>
SAGE2_ALWAYS_INLINE
static inline void prepare_v_mxfp8(
    device const uchar *cache_value, uint phys_page, uint sub_off,
    uint page_tokens, ulong page_bytes, ulong kvh_code_base,
    ulong all_codes_bytes, ulong bf16_kv_stride, uint kvh, short valid_k,
    uint flat_tid, threadgroup uint4 *tg_stage) {
  threadgroup uchar *stage_b = reinterpret_cast<threadgroup uchar *>(tg_stage);
  threadgroup uchar *tg_vt_fp8 = stage_b;              // [0 .. 8191]
  threadgroup uchar *tg_vt_sc = stage_b + 8192u;       // [8192 .. 8447]
  threadgroup float4 *tg_vmax_tmp =
      reinterpret_cast<threadgroup float4 *>(stage_b + 8448u); // [8448 .. 12543]

  float4 v_reg[8];
  const bool active = (flat_tid < 256u);
  const uint r_chunk = flat_tid >> 6u; // 0..3
  const uint v4_id = flat_tid & 63u;   // 0..63
  const uint p_g = v4_id >> 5u;        // 0..1
  const uint col_v4 = v4_id & 31u;     // 0..31

  if (active) {
    float4 lmax(0.0f);
    SAGE2_UNROLL
    for (uint rr = 0u; rr < 8u; ++rr) {
      const uint r = r_chunk * 8u + rr;
      v_reg[rr] = load_kv_vec4_safe<IsAffine4>(
          cache_value, phys_page, sub_off, page_tokens, page_bytes,
          kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, p_g, r, col_v4,
          valid_k);
      lmax = max(lmax, fabs(v_reg[rr]));
    }
    tg_vmax_tmp[r_chunk * 64u + v4_id] = lmax;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (active) {
    const float4 gmax = max(
        max(tg_vmax_tmp[0u * 64u + v4_id], tg_vmax_tmp[1u * 64u + v4_id]),
        max(tg_vmax_tmp[2u * 64u + v4_id], tg_vmax_tmp[3u * 64u + v4_id]));

    float inv_sc[4];
    uchar exp_u8[4];
    exp_u8[0] = compute_ue8m0_scale(gmax.x, inv_sc[0]);
    exp_u8[1] = compute_ue8m0_scale(gmax.y, inv_sc[1]);
    exp_u8[2] = compute_ue8m0_scale(gmax.z, inv_sc[2]);
    exp_u8[3] = compute_ue8m0_scale(gmax.w, inv_sc[3]);

    const uint d_base = col_v4 * 4u;
    if (r_chunk == 0u) {
      tg_vt_sc[p_g * 128u + d_base + 0u] = exp_u8[0];
      tg_vt_sc[p_g * 128u + d_base + 1u] = exp_u8[1];
      tg_vt_sc[p_g * 128u + d_base + 2u] = exp_u8[2];
      tg_vt_sc[p_g * 128u + d_base + 3u] = exp_u8[3];
    }

    // Write 4 channels x 8 rows = 32 FP8 bytes as 8 aligned packed_uchar4 stores.
    const uint k_base_v4 = r_chunk * 2u; // 0, 2, 4, 6 (in units of 4 bytes along dim0=32)
    {
      const float4 lo0 = float4(v_reg[0].x, v_reg[1].x, v_reg[2].x, v_reg[3].x) * inv_sc[0];
      const float4 hi0 = float4(v_reg[4].x, v_reg[5].x, v_reg[6].x, v_reg[7].x) * inv_sc[0];
      threadgroup packed_uchar4 *dst0 = reinterpret_cast<threadgroup packed_uchar4 *>(
          tg_vt_fp8 + (p_g * 128u + d_base + 0u) * 32u);
      dst0[k_base_v4 + 0u] = pack<metal_fp8_e4m3_format>(lo0).as_storage_type();
      dst0[k_base_v4 + 1u] = pack<metal_fp8_e4m3_format>(hi0).as_storage_type();
    }
    {
      const float4 lo1 = float4(v_reg[0].y, v_reg[1].y, v_reg[2].y, v_reg[3].y) * inv_sc[1];
      const float4 hi1 = float4(v_reg[4].y, v_reg[5].y, v_reg[6].y, v_reg[7].y) * inv_sc[1];
      threadgroup packed_uchar4 *dst1 = reinterpret_cast<threadgroup packed_uchar4 *>(
          tg_vt_fp8 + (p_g * 128u + d_base + 1u) * 32u);
      dst1[k_base_v4 + 0u] = pack<metal_fp8_e4m3_format>(lo1).as_storage_type();
      dst1[k_base_v4 + 1u] = pack<metal_fp8_e4m3_format>(hi1).as_storage_type();
    }
    {
      const float4 lo2 = float4(v_reg[0].z, v_reg[1].z, v_reg[2].z, v_reg[3].z) * inv_sc[2];
      const float4 hi2 = float4(v_reg[4].z, v_reg[5].z, v_reg[6].z, v_reg[7].z) * inv_sc[2];
      threadgroup packed_uchar4 *dst2 = reinterpret_cast<threadgroup packed_uchar4 *>(
          tg_vt_fp8 + (p_g * 128u + d_base + 2u) * 32u);
      dst2[k_base_v4 + 0u] = pack<metal_fp8_e4m3_format>(lo2).as_storage_type();
      dst2[k_base_v4 + 1u] = pack<metal_fp8_e4m3_format>(hi2).as_storage_type();
    }
    {
      const float4 lo3 = float4(v_reg[0].w, v_reg[1].w, v_reg[2].w, v_reg[3].w) * inv_sc[3];
      const float4 hi3 = float4(v_reg[4].w, v_reg[5].w, v_reg[6].w, v_reg[7].w) * inv_sc[3];
      threadgroup packed_uchar4 *dst3 = reinterpret_cast<threadgroup packed_uchar4 *>(
          tg_vt_fp8 + (p_g * 128u + d_base + 3u) * 32u);
      dst3[k_base_v4 + 0u] = pack<metal_fp8_e4m3_format>(lo3).as_storage_type();
      dst3[k_base_v4 + 1u] = pack<metal_fp8_e4m3_format>(hi3).as_storage_type();
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Cooperatively unpack V[kb] -> row-major FP16 (affine4) or BF16 (bf16) [2][32][128]
// at tg_stage[0..16383]. Strictly zeroes r >= valid_k!
template <bool IsAffine4, uint N_THR>
SAGE2_ALWAYS_INLINE
static inline void prepare_v_f16_bf16(
    device const uchar *cache_value, uint phys_page, uint sub_off,
    uint page_tokens, ulong page_bytes, ulong kvh_code_base,
    ulong all_codes_bytes, ulong bf16_kv_stride, uint kvh, short valid_k,
    uint flat_tid, threadgroup uint4 *tg_stage) {
  constexpr uint kIters = (2048u + N_THR - 1u) / N_THR;
  SAGE2_UNROLL
  for (uint it = 0u; it < kIters; ++it) {
    const uint idx = flat_tid + it * N_THR;
    if (idx < 2048u) {
      const uint p_g = idx >> 10u;
      const uint r = (idx >> 5u) & 31u;
      const uint col_v4 = idx & 31u;
      const float4 vf = load_kv_vec4_safe<IsAffine4>(
          cache_value, phys_page, sub_off, page_tokens, page_bytes,
          kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, p_g, r, col_v4,
          valid_k);
      if constexpr (IsAffine4) {
        reinterpret_cast<threadgroup half4 *>(tg_stage)[idx] = half4(vf);
      } else {
        reinterpret_cast<threadgroup bfloat4 *>(tg_stage)[idx] = bfloat4(vf);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <typename CoopT>
SAGE2_ALWAYS_INLINE
static inline void sage2_store_ct_output(
    thread CoopT &ct_O, device const bfloat *gate, device bfloat *output,
    const threadgroup float *tg_fac, constant PagedAttentionParams &p, uint tm,
    uint active_m_rows, uint repeat, uint kvh, uint q_base, uint d_offset) {
  constexpr uint BD = 256u;
  SAGE2_UNROLL
  for (ushort i = 0; i < ct_O.get_capacity(); ++i) {
    if (ct_O.is_valid_element(i)) {
      const auto idx = ct_O.get_multidimensional_index(i);
      const uint col = uint(idx[0]);
      const uint row = uint(idx[1]);
      const uint rm_o = tm + row;
      if (rm_o < active_m_rows) {
        uint tok_o = 0u, g_o = 0u;
        fast_divmod_g(rm_o, repeat, tok_o, g_o);
        const uint qh_o = kvh * repeat + g_o;
        const uint d_abs = d_offset + col;
        const float rcp = tg_fac[rm_o];
        const float g_val = float(
            gate[ulong(q_base + tok_o) * p.gate_token_stride +
                 ulong(qh_o) * p.gate_head_stride + d_abs]);
        const float out_val = (ct_O[i] * rcp) / (1.0f + exp(-g_val));
        output[(ulong(q_base + tok_o) * p.query_heads + qh_o) * BD + d_abs] =
            bfloat(out_val);
      }
    }
  }
}

template <uint WM, bool IsAffine4, bool UseMxfp8PV, bool SparseMInference>
SAGE2_ALWAYS_INLINE
static inline void paged_flash_prefill_sage2_hd256_impl(
    device const bfloat *query, device const bfloat *gate,
    device const uchar *cache_key, device const uchar *cache_value,
    device bfloat *output, device const uint *page_table,
    device const uint *positions, device const ushort *active_indices,
    device const uint *active_counts, constant Sage2AttentionParams &sp,
    threadgroup uint4 *tg_stage, threadgroup float *tg_scores,
    threadgroup float *tg_mu_k, threadgroup float *tg_sk,
    threadgroup float *tg_sq, threadgroup float *tg_delta,
    threadgroup float *tg_fac, ushort lane, ushort sg, uint3 tg) {
  constexpr uint BD = 256u;
  constexpr uint BK = 32u;
  constexpr uint WN = 2u;
  constexpr uint N_THR = WM * WN * 32u;
  constexpr uint M_TILE = WM * 16u;
  constexpr uint STAGE_RG = 2u;
  constexpr uint NUM_WAVES = WM / STAGE_RG;

  const PagedAttentionParams p = sp.base;
  if (p.tokens == 0u || p.query_heads == 0u || p.key_value_heads == 0u) {
    return;
  }
  const uint repeat = p.query_heads / p.key_value_heads;
  if (repeat == 0u || repeat > M_TILE) {
    return;
  }
  const uint bm_tok = M_TILE / repeat;
  const uint n_q_blocks = (p.tokens + bm_tok - 1u) / bm_tok;

  // Base 2.2: LPT Reverse q_blk + kvh-minor mapping.
  const uint q_blk_rev = tg.y / p.key_value_heads;
  const uint kvh = tg.y - q_blk_rev * p.key_value_heads;
  if (q_blk_rev >= n_q_blocks) {
    return;
  }
  const uint q_blk = (n_q_blocks - 1u) - q_blk_rev;
  const uint q_base = q_blk * bm_tok;
  const uint active_tok = min(bm_tok, p.tokens - q_base);
  const uint active_m_rows = active_tok * repeat;

  const uint flat_tid = uint(sg) * 32u + uint(lane);
  threadgroup uint *tg_q_pos = reinterpret_cast<threadgroup uint *>(tg_stage);
  if (flat_tid < active_tok) {
    tg_q_pos[flat_tid] = positions[q_base + flat_tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint max_valid_kv = 0u;
  uint min_q_pos = 0xffffffffu;
  for (uint r = 0u; r < active_tok; ++r) {
    const uint pos = tg_q_pos[r];
    max_valid_kv = max(max_valid_kv, pos + 1u);
    min_q_pos = min(min_q_pos, pos);
  }
  const uint nk_blocks = (max_valid_kv + BK - 1u) / BK;
  const uint kb_min_causal = min_q_pos / BK;

  const uint row_group = uint(sg >> 1u); // 0 .. WM-1
  const uint d_part = uint(sg & 1u);     // 0 .. 1
  const uint my_wave = row_group / STAGE_RG;
  const uint rg_local = row_group - my_wave * STAGE_RG;
  const uint tm = row_group * 16u;
  const uint d_offset = d_part * 128u;
  const bool warp_full_rows = (int(active_m_rows) - int(tm) >= 16);

  // Each lane owns 1 half-row (64 contiguous channels) of its 16x128 Q tile:
  //   q_r_in16 = lane >> 1 (0..15), q_half = lane & 1 (0..1)
  const uint q_r_in16 = uint(lane) >> 1u;
  const uint q_half = uint(lane) & 1u;
  const uint r_m = tm + q_r_in16;
  const bool qv = (r_m < active_m_rows);
  uint tok_l = 0u, g_l = 0u;
  if (qv) {
    fast_divmod_g(r_m, repeat, tok_l, g_l);
  }
  const uint qh_l = kvh * repeat + g_l;
  const uint qpos_l = qv ? tg_q_pos[tok_l] : 0u;

  // Guard reuse of tg_stage after reading tg_q_pos!
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Prologue: Quantize Q once into 16 char4 registers (64 B/lane = 16 VGPRs)
  // and store per-row, per-d_part s_Q in tg_sq[d_part * M_TILE + r_m].
  char4 q_i8_reg[16];
  {
    float4 q_f4[16];
    float lmax = 0.0f;
    if (qv) {
      device const bfloat4 *q_ptr = reinterpret_cast<device const bfloat4 *>(
          query + ulong(q_base + tok_l) * p.query_token_stride +
          ulong(qh_l) * p.query_head_stride + d_offset + q_half * 64u);
      SAGE2_UNROLL
      for (uint k = 0u; k < 16u; ++k) {
        const float4 v = float4(clean_bf4(q_ptr[k]));
        q_f4[k] = v;
        const float4 av = fabs(v);
        lmax = max(lmax, max(max(av.x, av.y), max(av.z, av.w)));
      }
    } else {
      SAGE2_UNROLL
      for (uint k = 0u; k < 16u; ++k) {
        q_f4[k] = float4(0.0f);
      }
    }
    const float q_absmax = max(lmax, simd_shuffle_xor(lmax, ushort(1)));
    const float sq = (q_absmax > 0.0f) ? (q_absmax * (1.0f / 127.0f)) : 0.0f;
    const float inv_sq = (q_absmax > 0.0f) ? (127.0f / q_absmax) : 0.0f;
    if (q_half == 0u) {
      tg_sq[d_part * M_TILE + r_m] = sq;
    }
    SAGE2_UNROLL
    for (uint k = 0u; k < 16u; ++k) {
      const float4 qf =
          clamp(rint(q_f4[k] * inv_sq), float4(-127.0f), float4(127.0f));
      q_i8_reg[k] =
          char4(char(int(qf.x)), char(int(qf.y)), char(int(qf.z)), char(int(qf.w)));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Set up persistent FP32 output accumulator cooperative_tensor ct_O [16, 128].
  constexpr auto qk_desc = matmul2d_descriptor(
      16, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<qk_desc, execution_simdgroup> qk_op;

  using PVValT = typename conditional<IsAffine4, half, bfloat>::type;
  using PVVal4 = typename conditional<IsAffine4, half4, bfloat4>::type;

  constexpr auto pv_desc_mx = matmul2d_descriptor(
      16, 128, 32, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc_mx, execution_simdgroup> pv_op_mx;

  constexpr auto pv_desc_f16 = matmul2d_descriptor(
      16, 128, 32, false, false, true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc_f16, execution_simdgroup> pv_op_f16;

  threadgroup uchar *stage_b = reinterpret_cast<threadgroup uchar *>(tg_stage);
  // Dummy tensors solely to instantiate ct_O with exact matching type.
  TgFp8Plain tP_mx_dummy(stage_b + 16384u, dextents<int32_t, 2>(32, 16));
  TgMxfp8VT tVT_mx_dummy(stage_b, dextents<int32_t, 2>(32, 128),
                         TgUe8ScaleTag(stage_b + 8192u));
  auto ct_O_mx = pv_op_mx.template get_destination_cooperative_tensor<
      decltype(tP_mx_dummy), decltype(tVT_mx_dummy), float>();

  tensor<threadgroup PVValT, dextents<int32_t, 2>, tensor_inline> tP_f16_dummy(
      reinterpret_cast<threadgroup PVValT *>(stage_b + 16384u),
      dextents<int32_t, 2>(32, 16));
  tensor<threadgroup PVValT, dextents<int32_t, 2>, tensor_inline> tV_f16_dummy(
      reinterpret_cast<threadgroup PVValT *>(stage_b),
      dextents<int32_t, 2>(128, 32));
  auto ct_O_f16 = pv_op_f16.template get_destination_cooperative_tensor<
      decltype(tP_f16_dummy), decltype(tV_f16_dummy), float>();

  if constexpr (UseMxfp8PV) {
    SAGE2_UNROLL
    for (ushort i = 0; i < ct_O_mx.get_capacity(); ++i) {
      ct_O_mx[i] = 0.0f;
    }
  } else {
    SAGE2_UNROLL
    for (ushort i = 0; i < ct_O_f16.get_capacity(); ++i) {
      ct_O_f16[i] = 0.0f;
    }
  }

  // Softmax state on d_part == 0 (1 row per lane: q_r_in16, 16 cols per lane).
  float max_s = kNegInf;
  float sum_s = 0.0f;
  const float scale2 = 0.0625f * kLog2e;

  const uint half_blocks_per_page = p.page_tokens / BK;
  const ulong page_bytes =
      affine4_page_byte_stride(p.page_tokens, p.key_value_heads, BD);
  const ulong head_code_bytes = ulong(p.page_tokens) * 128ul;
  const ulong all_codes_bytes = ulong(p.key_value_heads) * head_code_bytes;
  const ulong kvh_code_base = ulong(kvh) * head_code_bytes;
  const ulong bf16_kv_stride = ulong(p.key_value_heads) * BD;

  // Sparse MInference block loop count (or full dense nk_blocks).
  const uint n_steps =
      SparseMInference ? active_counts[kvh * sp.max_q_blocks + q_blk] : nk_blocks;

  for (uint step = 0u; step < n_steps; ++step) {
    const uint kb =
        SparseMInference
            ? uint(active_indices[(kvh * sp.max_q_blocks + q_blk) *
                                      sp.active_stride +
                                  step])
            : step;
    if (kb >= nk_blocks) {
      continue;
    }

    const uint lpage = kb / half_blocks_per_page;
    const uint sub_off = (kb % half_blocks_per_page) * BK;
    const uint phys_page = page_table[lpage];
    const uint kb_col_base = kb * BK;
    const short valid_k = short(min(BK, max_valid_kv - kb_col_base));
    const bool full_causal_free =
        (kb < kb_min_causal) && (valid_k == 32) && warp_full_rows;

    // Step 1: Cooperative K unpack -> zero-masked mu_{K,b,p} -> per-token s_{K,j,p} -> tg_k_i8
    prepare_k_smoothed_int8<IsAffine4>(
        cache_key, phys_page, sub_off, p.page_tokens, page_bytes, kvh_code_base,
        all_codes_bytes, bf16_kv_stride, kvh, valid_k, flat_tid, tg_stage,
        tg_mu_k, tg_sk);

    // For UseMxfp8PV == false, hold this lane's 16 P values in 4 PVVal4 regs on d_part == 0.
    PVVal4 p_reg_f16[4];

    // Step 2: Wave-staged INT8 Q x K^T -> Exact +Delta_{i,b,p} restoration -> Online Softmax
    threadgroup int8_t *tg_k_i8_part =
        reinterpret_cast<threadgroup int8_t *>(stage_b) + d_part * (32u * 128u);
    threadgroup int8_t *tg_q_stage_base =
        reinterpret_cast<threadgroup int8_t *>(stage_b + 8192u);
    threadgroup uchar *tg_p_fp8_all = stage_b + 16384u;

    SAGE2_UNROLL
    for (uint wave = 0u; wave < NUM_WAVES; ++wave) {
      const bool in_wave = (my_wave == wave);

      // 2A: Active SGs in wave stage Q_i8 from registers -> tg_q_stage AND compute
      // unscaled delta = dot(float(q_i8), mu_{K,b,p})!
      if (in_wave) {
        threadgroup char4 *q_dst = reinterpret_cast<threadgroup char4 *>(
            tg_q_stage_base + (rg_local * 2u + d_part) * (16u * 128u) +
            q_r_in16 * 128u + q_half * 64u);
        const threadgroup float4 *mu_src =
            reinterpret_cast<const threadgroup float4 *>(
                tg_mu_k + d_part * 128u + q_half * 64u);
        float d_acc = 0.0f;
        SAGE2_UNROLL
        for (uint k = 0u; k < 16u; ++k) {
          const char4 q4 = q_i8_reg[k];
          q_dst[k] = q4;
          const float4 qf(float(q4.x), float(q4.y), float(q4.z), float(q4.w));
          d_acc += dot(qf, mu_src[k]);
        }
        d_acc += simd_shuffle_xor(d_acc, ushort(1));
        if (q_half == 0u) {
          tg_delta[d_part * (STAGE_RG * 16u) + rg_local * 16u + q_r_in16] =
              d_acc;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // 2B: Active SGs run single-op 16x32x128 INT8 NAX matmul2d, de-scale with
      // per-token s_{K,j,p} + exact delta restoration; d_part==1 writes to tg_scores.
      threadgroup int8_t *my_tg_q =
          tg_q_stage_base + (rg_local * 2u + d_part) * (16u * 128u);
      tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ(
          my_tg_q, dextents<int32_t, 2>(128, 16));
      tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tK(
          tg_k_i8_part, dextents<int32_t, 2>(128, 32));
      auto ct_s = qk_op.template get_destination_cooperative_tensor<
          decltype(tQ), decltype(tK), int32_t>();

      if (in_wave) {
        qk_op.run(tQ, tK, ct_s);
        if (d_part == 1u) {
          SAGE2_UNROLL
          for (ushort i = 0; i < ct_s.get_capacity(); ++i) {
            if (ct_s.is_valid_element(i)) {
              const auto idx = ct_s.get_multidimensional_index(i);
              const uint col = uint(idx[0]);
              const uint row = uint(idx[1]);
              const float sq = tg_sq[1u * M_TILE + tm + row];
              const float sk = tg_sk[1u * 32u + col];
              const float del = tg_delta[1u * (STAGE_RG * 16u) + rg_local * 16u + row];
              tg_scores[rg_local * 512u + row * 32u + col] =
                  sq * fma(float(ct_s[i]), sk, del);
            }
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // 2C: d_part == 0 adds its de-scaled +Delta_0 contribution to tg_scores,
      // then executes Online Softmax (-50% exp2!) with causal mask applied AFTER +Delta!
      if (in_wave && d_part == 0u) {
        SAGE2_UNROLL
        for (ushort i = 0; i < ct_s.get_capacity(); ++i) {
          if (ct_s.is_valid_element(i)) {
            const auto idx = ct_s.get_multidimensional_index(i);
            const uint col = uint(idx[0]);
            const uint row = uint(idx[1]);
            const float sq = tg_sq[0u * M_TILE + tm + row];
            const float sk = tg_sk[0u * 32u + col];
            const float del = tg_delta[0u * (STAGE_RG * 16u) + rg_local * 16u + row];
            tg_scores[rg_local * 512u + row * 32u + col] +=
                sq * fma(float(ct_s[i]), sk, del);
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Linear half-row Softmax: 1 row (q_r_in16) x 16 contiguous cols per lane!
        const threadgroup float4 *s_row4 =
            reinterpret_cast<const threadgroup float4 *>(
                tg_scores + rg_local * 512u + q_r_in16 * 32u + q_half * 16u);
        float4 s0 = s_row4[0] * scale2;
        float4 s1 = s_row4[1] * scale2;
        float4 s2 = s_row4[2] * scale2;
        float4 s3 = s_row4[3] * scale2;

        if (!full_causal_free) {
          const uint c_base = kb_col_base + q_half * 16u;
          const uint4 c0 = uint4(c_base) + uint4(0u, 1u, 2u, 3u);
          const uint4 c1 = c0 + uint4(4u);
          const uint4 c2 = c0 + uint4(8u);
          const uint4 c3 = c0 + uint4(12u);
          s0 = select(float4(kNegInf), s0,
                      qv && (c0 < max_valid_kv) && (c0 <= qpos_l));
          s1 = select(float4(kNegInf), s1,
                      qv && (c1 < max_valid_kv) && (c1 <= qpos_l));
          s2 = select(float4(kNegInf), s2,
                      qv && (c2 < max_valid_kv) && (c2 <= qpos_l));
          s3 = select(float4(kNegInf), s3,
                      qv && (c3 < max_valid_kv) && (c3 <= qpos_l));
        }

        const float4 m01 = max(s0, s1);
        const float4 m23 = max(s2, s3);
        const float4 m4 = max(m01, m23);
        float m_loc = max(max(m4.x, m4.y), max(m4.z, m4.w));
        m_loc = max(m_loc, simd_shuffle_xor(m_loc, ushort(1)));
        const float new_m = max(max_s, m_loc);
        const float fac =
            (max_s <= kNegInfHalf) ? 0.0f : fast::exp2(max_s - new_m);
        max_s = new_m;
        sum_s *= fac;

        float4 p0, p1, p2, p3;
        if (full_causal_free) {
          p0 = fast::exp2(s0 - new_m);
          p1 = fast::exp2(s1 - new_m);
          p2 = fast::exp2(s2 - new_m);
          p3 = fast::exp2(s3 - new_m);
        } else {
          const bool emp = (new_m <= kNegInfHalf);
          p0 = select(fast::exp2(s0 - new_m), float4(0.0f),
                      emp || (s0 <= kNegInfHalf));
          p1 = select(fast::exp2(s1 - new_m), float4(0.0f),
                      emp || (s1 <= kNegInfHalf));
          p2 = select(fast::exp2(s2 - new_m), float4(0.0f),
                      emp || (s2 <= kNegInfHalf));
          p3 = select(fast::exp2(s3 - new_m), float4(0.0f),
                      emp || (s3 <= kNegInfHalf));
        }

        const float4 ps = (p0 + p1) + (p2 + p3);
        float tsum = (ps.x + ps.y) + (ps.z + ps.w);
        tsum += simd_shuffle_xor(tsum, ushort(1));
        sum_s += tsum;

        if (q_half == 0u) {
          tg_fac[r_m] = fac;
        }

        if constexpr (UseMxfp8PV) {
          // Pack P * 448.0f directly to tg_p_fp8 [WM * 16][32] at tg_stage[16384..20479].
          threadgroup packed_uchar4 *p_dst =
              reinterpret_cast<threadgroup packed_uchar4 *>(
                  tg_p_fp8_all + r_m * 32u + q_half * 16u);
          p_dst[0] =
              pack<metal_fp8_e4m3_format>(p0 * kFp8E4M3Max).as_storage_type();
          p_dst[1] =
              pack<metal_fp8_e4m3_format>(p1 * kFp8E4M3Max).as_storage_type();
          p_dst[2] =
              pack<metal_fp8_e4m3_format>(p2 * kFp8E4M3Max).as_storage_type();
          p_dst[3] =
              pack<metal_fp8_e4m3_format>(p3 * kFp8E4M3Max).as_storage_type();
        } else {
          p_reg_f16[0] = PVVal4(p0);
          p_reg_f16[1] = PVVal4(p1);
          p_reg_f16[2] = PVVal4(p2);
          p_reg_f16[3] = PVVal4(p3);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 3: P x V accumulation into register ct_O!
    if constexpr (UseMxfp8PV) {
      // Unpack V[kb] -> transposed MXFP8 V^T [2][128][32] + UE8M0 scales [2][128]
      // (leaves tg_p_fp8_all at [16384..20479] untouched!).
      prepare_v_mxfp8<IsAffine4>(
          cache_value, phys_page, sub_off, p.page_tokens, page_bytes,
          kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, valid_k,
          flat_tid, tg_stage);

      // Rescale ct_O by per-row fac before multiply_accumulate.
      SAGE2_UNROLL
      for (ushort i = 0; i < ct_O_mx.get_capacity(); ++i) {
        if (ct_O_mx.is_valid_element(i)) {
          const auto idx = ct_O_mx.get_multidimensional_index(i);
          ct_O_mx[i] *= tg_fac[tm + uint(idx[1])];
        }
      }

      // All WM*2 SGs execute MXFP8 P x V^T concurrently in one shot!
      TgFp8Plain tP(tg_p_fp8_all + tm * 32u, dextents<int32_t, 2>(32, 16));
      TgMxfp8VT tVT(stage_b + d_part * (128u * 32u),
                    dextents<int32_t, 2>(32, 128),
                    TgUe8ScaleTag(stage_b + 8192u + d_part * 128u));
      pv_op_mx.run(tP, tVT, ct_O_mx);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      // Unpack V[kb] -> row-major FP16/BF16 [2][32][128] at tg_stage[0..16383].
      prepare_v_f16_bf16<IsAffine4, N_THR>(
          cache_value, phys_page, sub_off, p.page_tokens, page_bytes,
          kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, valid_k,
          flat_tid, tg_stage);

      // Rescale ct_O by per-row fac.
      SAGE2_UNROLL
      for (ushort i = 0; i < ct_O_f16.get_capacity(); ++i) {
        if (ct_O_f16.is_valid_element(i)) {
          const auto idx = ct_O_f16.get_multidimensional_index(i);
          ct_O_f16[i] *= tg_fac[tm + uint(idx[1])];
        }
      }

      // Stage P [STAGE_RG][16][32] wave-by-wave at tg_stage[16384..18431].
      threadgroup PVValT *tg_p_wave =
          reinterpret_cast<threadgroup PVValT *>(stage_b + 16384u);
      threadgroup PVValT *tg_v_part =
          reinterpret_cast<threadgroup PVValT *>(stage_b) +
          d_part * (32u * 128u);

      SAGE2_UNROLL
      for (uint wave = 0u; wave < NUM_WAVES; ++wave) {
        const bool in_wave = (my_wave == wave);
        if (in_wave && d_part == 0u) {
          threadgroup PVVal4 *p_dst =
              reinterpret_cast<threadgroup PVVal4 *>(
                  tg_p_wave + rg_local * 512u + q_r_in16 * 32u + q_half * 16u);
          p_dst[0] = p_reg_f16[0];
          p_dst[1] = p_reg_f16[1];
          p_dst[2] = p_reg_f16[2];
          p_dst[3] = p_reg_f16[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (in_wave) {
          tensor<threadgroup PVValT, dextents<int32_t, 2>, tensor_inline> tP(
              tg_p_wave + rg_local * 512u, dextents<int32_t, 2>(32, 16));
          tensor<threadgroup PVValT, dextents<int32_t, 2>, tensor_inline> tV(
              tg_v_part, dextents<int32_t, 2>(128, 32));
          pv_op_f16.run(tP, tV, ct_O_f16);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }

  // Step 4: Epilogue — broadcast final 1 / (scale_p * sum_s) via tg_fac and store gated BF16 output.
  if (d_part == 0u && q_half == 0u) {
    if constexpr (UseMxfp8PV) {
      tg_fac[r_m] = (sum_s > 0.0f) ? (kInvFp8E4M3Max / sum_s) : 0.0f;
    } else {
      tg_fac[r_m] = (sum_s > 0.0f) ? (1.0f / sum_s) : 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if constexpr (UseMxfp8PV) {
    sage2_store_ct_output(ct_O_mx, gate, output, tg_fac, sp.base, tm,
                          active_m_rows, repeat, kvh, q_base, d_offset);
  } else {
    sage2_store_ct_output(ct_O_f16, gate, output, tg_fac, sp.base, tm,
                          active_m_rows, repeat, kvh, q_base, d_offset);
  }
}

// ============================================================================
// High-Throughput WM=4 (NUM_WAVES=1) FP16/BF16-PV SageAttention2 Engine
//   - 100% resident tg_q_i8 (16 KiB) + time-shared tg_work (8 KiB)
//   - Total static Threadgroup SRAM = 26.5 KiB (<= 28.5 KiB hardware budget)
//   - All 8 SGs (4 row_groups x 2 d_parts) execute INT8 16x32x128 QK concurrently
//     with fused dot(q_i8, mu_k) delta restoration and hoisted K pointers
//   - Apple10 (M5): Direct-L1 Frag16x16 P x V (0 V staging, 0 PV barriers,
//     7 barriers/block total vs 24 in wave-staged path)
//   - Portable fallback (Apple9): 16x32x32 4-subtile NAX P x V in tg_work
// ============================================================================
SAGE2_ALWAYS_INLINE
static inline short2 sage2_frag_coord(ushort lane) {
  const short qid = short(lane >> 2u);
  const short fm = (qid & 4) | short((lane >> 1u) & 3u);
  const short fn = ((qid & 2) | short(lane & 1u)) * 4;
  return short2(fn, fm);
}

template <typename T>
SAGE2_ALWAYS_INLINE
static inline bool sage2_has_apple10_pv_layout(ushort lane) {
  const short2 sc = sage2_frag_coord(lane);
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, false, true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
  auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>,
      float>();
  const auto a0 = ct_a.get_multidimensional_index(ushort(0u));
  const auto a2 = ct_a.get_multidimensional_index(ushort(2u));
  const auto a4 = ct_a.get_multidimensional_index(ushort(4u));
  const auto b0 = ct_b.get_multidimensional_index(ushort(0u));
  const auto b8 = ct_b.get_multidimensional_index(ushort(8u));
  const auto c0 = ct_c.get_multidimensional_index(ushort(0u));
  const auto c8 = ct_c.get_multidimensional_index(ushort(8u));
  return (int(a0[0]) == int(sc.x)) && (int(a0[1]) == int(sc.y)) &&
         (int(a2[0]) == int(sc.x) + 2) && (int(a2[1]) == int(sc.y)) &&
         (int(a4[0]) == int(sc.x)) && (int(a4[1]) == int(sc.y) + 8) &&
         (int(b0[0]) == int(sc.x)) && (int(b0[1]) == int(sc.y)) &&
         (int(b8[0]) == int(sc.x) + 16) && (int(b8[1]) == int(sc.y)) &&
         (int(c0[0]) == int(sc.x)) && (int(c0[1]) == int(sc.y)) &&
         (int(c8[0]) == int(sc.x) + 16) && (int(c8[1]) == int(sc.y));
}

SAGE2_ALWAYS_INLINE
static inline vec<half, 8> sage2_dequant_v_ptr_h(device const ushort *p0,
                                                 device const ushort *p1,
                                                 half s0, half z0, half s1,
                                                 half z1, uint col_off) {
  const ushort pk0 = p0[col_off];
  const ushort pk1 = p1[col_off];
  const half4 q0 = half4(ushort4(pk0 & 0xFu, (pk0 >> 4u) & 0xFu,
                                 (pk0 >> 8u) & 0xFu, pk0 >> 12u));
  const half4 q1 = half4(ushort4(pk1 & 0xFu, (pk1 >> 4u) & 0xFu,
                                 (pk1 >> 8u) & 0xFu, pk1 >> 12u));
  const half4 v0 = fma(q0, half4(s0), half4(z0));
  const half4 v1 = fma(q1, half4(s1), half4(z1));
  return vec<half, 8>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
}

SAGE2_ALWAYS_INLINE
static inline vec<bfloat, 8> sage2_load_v_ptr_bf(device const bfloat *p0,
                                                 device const bfloat *p1,
                                                 bool ok0, bool ok1,
                                                 uint col_off_elems) {
  const bfloat4 v0 =
      ok0 ? clean_bf4(*reinterpret_cast<device const bfloat4 *>(p0 +
                                                                col_off_elems))
          : bfloat4(bfloat(0.0f));
  const bfloat4 v1 =
      ok1 ? clean_bf4(*reinterpret_cast<device const bfloat4 *>(p1 +
                                                                col_off_elems))
          : bfloat4(bfloat(0.0f));
  return vec<bfloat, 8>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
}

template <typename T>
SAGE2_ALWAYS_INLINE
static inline void sage2_mma_1x2_pv(thread vec<float, 8> &c0,
                                    thread vec<float, 8> &c1,
                                    const thread vec<T, 8> &a,
                                    const thread vec<T, 8> &b0,
                                    const thread vec<T, 8> &b1) {
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, false, true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
  auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>,
      float>();
  SAGE2_UNROLL
  for (short i = 0; i < 8; ++i) {
    ct_a[i] = a[i];
    ct_b[i] = b0[i];
    ct_b[8 + i] = b1[i];
    ct_c[i] = c0[i];
    ct_c[8 + i] = c1[i];
  }
  op.run(ct_a, ct_b, ct_c);
  SAGE2_UNROLL
  for (short i = 0; i < 8; ++i) {
    c0[i] = ct_c[i];
    c1[i] = ct_c[8 + i];
  }
}

// Fast 256-thread K unpack + channel-mean smoothing + fused dot(q_i8, mu_k) delta!
// Uses ONLY tg_work[8192 B] (tg_mu_tmp[0..4095] -> tg_k_i8[0..8191]) + resident tg_q_i8.
template <bool IsAffine4>
SAGE2_ALWAYS_INLINE
static inline void prepare_k_and_delta_wm4(
    device const uchar *cache_key, uint phys_page, uint sub_off,
    uint page_tokens, ulong page_bytes, ulong kvh_code_base,
    ulong all_codes_bytes, ulong bf16_kv_stride, uint kvh, short valid_k,
    uint flat_tid, uint row_group, uint d_part, uint q_r_in16, uint q_half,
    const threadgroup uchar *tg_q_i8_b, threadgroup uint4 *tg_work,
    threadgroup float *tg_mu_k, threadgroup float *tg_sk,
    threadgroup float *tg_delta) {
  threadgroup float4 *tg_mu_tmp =
      reinterpret_cast<threadgroup float4 *>(tg_work); // [0 .. 4095]
  threadgroup char4 *tg_k_i8_v4 =
      reinterpret_cast<threadgroup char4 *>(tg_work); // [0 .. 8191]

  float4 k_reg[8];
  const uint r_chunk = flat_tid >> 6u; // 0..3
  const uint v4_id = flat_tid & 63u;   // 0..63
  const uint p_g = v4_id >> 5u;        // 0..1
  const uint col_v4 = v4_id & 31u;     // 0..31 (= lane!)

  // Hoist invariant page / head pointers out of the 8-row loop.
  float4 sum4(0.0f);
  if constexpr (IsAffine4) {
    device const uchar *page_ptr = cache_key + ulong(phys_page) * page_bytes;
    device const ushort *codes = reinterpret_cast<device const ushort *>(
        page_ptr + kvh_code_base + ulong(p_g) * (64ul * 64ul));
    device const KvAffine4VecMeta *meta =
        reinterpret_cast<device const KvAffine4VecMeta *>(page_ptr +
                                                          all_codes_bytes) +
        ulong(kvh) * page_tokens;
    SAGE2_UNROLL
    for (uint rr = 0u; rr < 8u; ++rr) {
      const uint r = r_chunk * 8u + rr;
      if (short(r) < valid_k) {
        const uint within = sub_off + r;
        const ushort pk = codes[within * 32u + col_v4];
        const float s = float(meta[within].scale[p_g]);
        const float z = float(meta[within].zero[p_g]);
        const float4 q4 =
            float4(float(pk & 0xFu), float((pk >> 4u) & 0xFu),
                   float((pk >> 8u) & 0xFu), float(pk >> 12u));
        k_reg[rr] = clean_f4(fma(q4, float4(s), float4(z)));
      } else {
        k_reg[rr] = float4(0.0f);
      }
      sum4 += k_reg[rr];
    }
  } else {
    device const bfloat *cache_bf =
        reinterpret_cast<device const bfloat *>(cache_key);
    const ulong base_row = ulong(phys_page) * page_tokens + sub_off;
    const ulong head_off = ulong(kvh) * 256ul + p_g * 128u;
    SAGE2_UNROLL
    for (uint rr = 0u; rr < 8u; ++rr) {
      const uint r = r_chunk * 8u + rr;
      if (short(r) < valid_k) {
        device const bfloat4 *row_v4 = reinterpret_cast<device const bfloat4 *>(
            cache_bf + (base_row + r) * bf16_kv_stride + head_off);
        k_reg[rr] = float4(clean_bf4(row_v4[col_v4]));
      } else {
        k_reg[rr] = float4(0.0f);
      }
      sum4 += k_reg[rr];
    }
  }
  tg_mu_tmp[r_chunk * 64u + v4_id] = sum4;
  threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier #1

  if (flat_tid < 64u) {
    const float inv_vk = 1.0f / float(max(int(valid_k), 1));
    const float4 total = (tg_mu_tmp[0u * 64u + flat_tid] +
                          tg_mu_tmp[1u * 64u + flat_tid]) +
                         (tg_mu_tmp[2u * 64u + flat_tid] +
                          tg_mu_tmp[3u * 64u + flat_tid]);
    reinterpret_cast<threadgroup float4 *>(tg_mu_k)[flat_tid] = total * inv_vk;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier #2

  // tg_mu_tmp is now dead; reuse tg_work[0..8191] for tg_k_i8[2][32][128].
  const float4 mu4 = reinterpret_cast<threadgroup float4 *>(tg_mu_k)[v4_id];
  SAGE2_UNROLL
  for (uint rr = 0u; rr < 8u; ++rr) {
    const uint r = r_chunk * 8u + rr;
    const float4 kt = (short(r) < valid_k) ? (k_reg[rr] - mu4) : float4(0.0f);
    const float4 ak = fabs(kt);
    const float lane_max = max(max(ak.x, ak.y), max(ak.z, ak.w));
    const float row_max = simd_max(lane_max);
    const float sk = (row_max > 0.0f) ? (row_max * (1.0f / 127.0f)) : 0.0f;
    const float inv_sk = (row_max > 0.0f) ? (127.0f / row_max) : 0.0f;
    if (col_v4 == 0u) {
      tg_sk[p_g * 32u + r] = sk;
    }
    const float4 qf = clamp(rint(kt * inv_sk), float4(-127.0f), float4(127.0f));
    const char4 kq(char(int(qf.x)), char(int(qf.y)), char(int(qf.z)),
                   char(int(qf.w)));
    tg_k_i8_v4[p_g * 1024u + r * 32u + col_v4] = kq;
  }

  // Fused exact Delta = dot(float(q_i8), mu_{K,b,p}) BEFORE Barrier #3!
  {
    const threadgroup char4 *q_src =
        reinterpret_cast<const threadgroup char4 *>(
            tg_q_i8_b + (row_group * 2u + d_part) * (16u * 128u) +
            q_r_in16 * 128u + q_half * 64u);
    const threadgroup float4 *mu_src =
        reinterpret_cast<const threadgroup float4 *>(
            tg_mu_k + d_part * 128u + q_half * 64u);
    float d_acc = 0.0f;
    SAGE2_UNROLL
    for (uint k = 0u; k < 16u; ++k) {
      const char4 q4 = q_src[k];
      const float4 qf(float(q4.x), float(q4.y), float(q4.z), float(q4.w));
      d_acc += dot(qf, mu_src[k]);
    }
    d_acc += simd_shuffle_xor(d_acc, ushort(1));
    if (q_half == 0u) {
      tg_delta[d_part * 64u + row_group * 16u + q_r_in16] = d_acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier #3
}

template <bool IsAffine4, bool SparseMInference>
SAGE2_ALWAYS_INLINE
static inline void paged_flash_prefill_sage2_wm4_fp16_impl(
    device const bfloat *query, device const bfloat *gate,
    device const uchar *cache_key, device const uchar *cache_value,
    device bfloat *output, device const uint *page_table,
    device const uint *positions, device const ushort *active_indices,
    device const uint *active_counts, constant Sage2AttentionParams &sp,
    threadgroup uint4 *tg_q_i8, threadgroup uint4 *tg_work,
    threadgroup float *tg_mu_k, threadgroup float *tg_sk,
    threadgroup float *tg_sq, threadgroup float *tg_delta,
    threadgroup float *tg_fac, ushort lane, ushort sg, uint3 tg) {
  constexpr uint BD = 256u;
  constexpr uint BK = 32u;
  constexpr uint WM = 4u;
  constexpr uint M_TILE = WM * 16u; // 64
  using PVValT = typename conditional<IsAffine4, half, bfloat>::type;
  using PVVal4 = typename conditional<IsAffine4, half4, bfloat4>::type;

  const PagedAttentionParams p = sp.base;
  if (p.tokens == 0u || p.query_heads == 0u || p.key_value_heads == 0u) {
    return;
  }
  const uint repeat = p.query_heads / p.key_value_heads;
  if (repeat == 0u || repeat > M_TILE) {
    return;
  }
  const uint bm_tok = M_TILE / repeat;
  const uint n_q_blocks = (p.tokens + bm_tok - 1u) / bm_tok;

  // Base 2.2: LPT Reverse q_blk + kvh-minor mapping.
  const uint q_blk_rev = tg.y / p.key_value_heads;
  const uint kvh = tg.y - q_blk_rev * p.key_value_heads;
  if (q_blk_rev >= n_q_blocks) {
    return;
  }
  const uint q_blk = (n_q_blocks - 1u) - q_blk_rev;
  const uint q_base = q_blk * bm_tok;
  const uint active_tok = min(bm_tok, p.tokens - q_base);
  const uint active_m_rows = active_tok * repeat;

  const uint flat_tid = uint(sg) * 32u + uint(lane);
  threadgroup uint *tg_q_pos = reinterpret_cast<threadgroup uint *>(tg_work);
  if (flat_tid < active_tok) {
    tg_q_pos[flat_tid] = positions[q_base + flat_tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint max_valid_kv = 0u;
  uint min_q_pos = 0xffffffffu;
  for (uint r = 0u; r < active_tok; ++r) {
    const uint pos = tg_q_pos[r];
    max_valid_kv = max(max_valid_kv, pos + 1u);
    min_q_pos = min(min_q_pos, pos);
  }
  const uint nk_blocks = (max_valid_kv + BK - 1u) / BK;
  const uint kb_min_causal = min_q_pos / BK;

  const uint row_group = uint(sg >> 1u); // 0..3
  const uint d_part = uint(sg & 1u);     // 0..1
  const uint tm = row_group * 16u;
  const uint d_offset = d_part * 128u;
  const bool warp_full_rows = (int(active_m_rows) - int(tm) >= 16);

  const uint q_r_in16 = uint(lane) >> 1u; // 0..15
  const uint q_half = uint(lane) & 1u;    // 0..1
  const uint r_m = tm + q_r_in16;
  const bool qv = (r_m < active_m_rows);
  uint tok_l = 0u, g_l = 0u;
  if (qv) {
    fast_divmod_g(r_m, repeat, tok_l, g_l);
  }
  const uint qh_l = kvh * repeat + g_l;
  const uint qpos_l = qv ? tg_q_pos[tok_l] : 0u;

  // Guard reuse of tg_work after reading tg_q_pos!
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Prologue: Quantize Q ONCE directly into 100% resident tg_q_i8 [4][2][16][128] (16 KiB)!
  threadgroup uchar *tg_q_i8_b = reinterpret_cast<threadgroup uchar *>(tg_q_i8);
  {
    float4 q_f4[16];
    float lmax = 0.0f;
    if (qv) {
      device const bfloat4 *q_ptr = reinterpret_cast<device const bfloat4 *>(
          query + ulong(q_base + tok_l) * p.query_token_stride +
          ulong(qh_l) * p.query_head_stride + d_offset + q_half * 64u);
      SAGE2_UNROLL
      for (uint k = 0u; k < 16u; ++k) {
        const float4 v = float4(clean_bf4(q_ptr[k]));
        q_f4[k] = v;
        const float4 av = fabs(v);
        lmax = max(lmax, max(max(av.x, av.y), max(av.z, av.w)));
      }
    } else {
      SAGE2_UNROLL
      for (uint k = 0u; k < 16u; ++k) {
        q_f4[k] = float4(0.0f);
      }
    }
    const float q_absmax = max(lmax, simd_shuffle_xor(lmax, ushort(1)));
    const float sq = (q_absmax > 0.0f) ? (q_absmax * (1.0f / 127.0f)) : 0.0f;
    const float inv_sq = (q_absmax > 0.0f) ? (127.0f / q_absmax) : 0.0f;
    if (q_half == 0u) {
      tg_sq[d_part * M_TILE + r_m] = sq;
    }
    threadgroup char4 *q_dst = reinterpret_cast<threadgroup char4 *>(
        tg_q_i8_b + (row_group * 2u + d_part) * (16u * 128u) +
        q_r_in16 * 128u + q_half * 64u);
    SAGE2_UNROLL
    for (uint k = 0u; k < 16u; ++k) {
      const float4 qf =
          clamp(rint(q_f4[k] * inv_sq), float4(-127.0f), float4(127.0f));
      q_dst[k] = char4(char(int(qf.x)), char(int(qf.y)), char(int(qf.z)),
                       char(int(qf.w)));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr auto qk_desc = matmul2d_descriptor(
      16, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<qk_desc, execution_simdgroup> qk_op;

  float max_s = kNegInf;
  float sum_s = 0.0f;
  const float scale2 = 0.0625f * kLog2e;

  const uint half_blocks_per_page = p.page_tokens / BK;
  const ulong page_bytes =
      affine4_page_byte_stride(p.page_tokens, p.key_value_heads, BD);
  const ulong head_code_bytes = ulong(p.page_tokens) * 128ul;
  const ulong all_codes_bytes = ulong(p.key_value_heads) * head_code_bytes;
  const ulong kvh_code_base = ulong(kvh) * head_code_bytes;
  const ulong group_code_offset =
      kvh_code_base + ulong(d_part) * (64ul * 64ul);
  const ulong bf16_kv_stride = ulong(p.key_value_heads) * BD;

  threadgroup uchar *work_b = reinterpret_cast<threadgroup uchar *>(tg_work);
  threadgroup int8_t *my_tg_q = reinterpret_cast<threadgroup int8_t *>(
      tg_q_i8_b + (row_group * 2u + d_part) * (16u * 128u));
  threadgroup int8_t *tg_k_i8_part =
      reinterpret_cast<threadgroup int8_t *>(work_b) + d_part * (32u * 128u);
  threadgroup float *tg_scores = reinterpret_cast<threadgroup float *>(work_b);

  const uint n_steps =
      SparseMInference ? active_counts[kvh * sp.max_q_blocks + q_blk] : nk_blocks;

  if (sage2_has_apple10_pv_layout<PVValT>(lane)) {
    // ------------------------------------------------------------------------
    // Apple10 (M5) Direct-L1 Frag16x16 P x V Fast Path (0 V staging, 7 barriers/blk)
    // ------------------------------------------------------------------------
    using Frag8 = vec<PVValT, 8>;
    using F8 = vec<float, 8>;
    const short2 sc = sage2_frag_coord(lane);
    const short sm = sc.y;
    const short sn = sc.x;
    const uint sn_u16 = uint(sn >> 2u); // 0..3

    F8 O0(0.0f), O1(0.0f), O2(0.0f), O3(0.0f);
    F8 O4(0.0f), O5(0.0f), O6(0.0f), O7(0.0f);

    // Each row_group stores its 16x32 PVValT P tile in the first 1024 B of its
    // own private 2048 B tg_scores slice inside tg_work (zero cross-RG conflict!).
    threadgroup PVValT *my_tg_p =
        reinterpret_cast<threadgroup PVValT *>(work_b + row_group * 2048u);

    for (uint step = 0u; step < n_steps; ++step) {
      const uint kb =
          SparseMInference
              ? uint(active_indices[(kvh * sp.max_q_blocks + q_blk) *
                                        sp.active_stride +
                                    step])
              : step;
      if (kb >= nk_blocks) {
        continue;
      }

      const uint lpage = kb / half_blocks_per_page;
      const uint sub_off = (kb % half_blocks_per_page) * BK;
      const uint phys_page = page_table[lpage];
      const uint kb_col_base = kb * BK;
      const short valid_k = short(min(BK, max_valid_kv - kb_col_base));
      const bool full_causal_free =
          (kb < kb_min_causal) && (valid_k == 32) && warp_full_rows;

      // Step 1: Cooperative K unpack -> mu_k -> tg_k_i8 + fused Delta (Barriers #1, #2, #3)
      prepare_k_and_delta_wm4<IsAffine4>(
          cache_key, phys_page, sub_off, p.page_tokens, page_bytes,
          kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, valid_k,
          flat_tid, row_group, d_part, q_r_in16, q_half, tg_q_i8_b, tg_work,
          tg_mu_k, tg_sk, tg_delta);

      // Step 2A: All 8 SGs execute 16x32x128 INT8 NAX Q x K^T concurrently!
      tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ(
          my_tg_q, dextents<int32_t, 2>(128, 16));
      tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tK(
          tg_k_i8_part, dextents<int32_t, 2>(128, 32));
      auto ct_s = qk_op.template get_destination_cooperative_tensor<
          decltype(tQ), decltype(tK), int32_t>();
      qk_op.run(tQ, tK, ct_s);

      // Wait until all 8 SGs finish reading tg_k_i8 before reusing tg_work for tg_scores!
      threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier #4

      // Step 2B: d_part == 1 writes de-scaled S_1 + Delta_1 to tg_scores[row_group]
      if (d_part == 1u) {
        SAGE2_UNROLL
        for (ushort i = 0; i < ct_s.get_capacity(); ++i) {
          if (ct_s.is_valid_element(i)) {
            const auto idx = ct_s.get_multidimensional_index(i);
            const uint col = uint(idx[0]);
            const uint row = uint(idx[1]);
            const float sq = tg_sq[1u * M_TILE + tm + row];
            const float sk = tg_sk[1u * 32u + col];
            const float del = tg_delta[1u * M_TILE + tm + row];
            tg_scores[row_group * 512u + row * 32u + col] =
                sq * fma(float(ct_s[i]), sk, del);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier #5

      // Step 2C: d_part == 0 adds S_0 + Delta_0, executes Online Softmax, and writes P -> my_tg_p!
      if (d_part == 0u) {
        SAGE2_UNROLL
        for (ushort i = 0; i < ct_s.get_capacity(); ++i) {
          if (ct_s.is_valid_element(i)) {
            const auto idx = ct_s.get_multidimensional_index(i);
            const uint col = uint(idx[0]);
            const uint row = uint(idx[1]);
            const float sq = tg_sq[0u * M_TILE + tm + row];
            const float sk = tg_sk[0u * 32u + col];
            const float del = tg_delta[0u * M_TILE + tm + row];
            tg_scores[row_group * 512u + row * 32u + col] +=
                sq * fma(float(ct_s[i]), sk, del);
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup float4 *s_row4 =
            reinterpret_cast<const threadgroup float4 *>(
                tg_scores + row_group * 512u + q_r_in16 * 32u + q_half * 16u);
        float4 s0 = s_row4[0] * scale2;
        float4 s1 = s_row4[1] * scale2;
        float4 s2 = s_row4[2] * scale2;
        float4 s3 = s_row4[3] * scale2;

        if (!full_causal_free) {
          const uint c_base = kb_col_base + q_half * 16u;
          const uint4 c0 = uint4(c_base) + uint4(0u, 1u, 2u, 3u);
          const uint4 c1 = c0 + uint4(4u);
          const uint4 c2 = c0 + uint4(8u);
          const uint4 c3 = c0 + uint4(12u);
          s0 = select(float4(kNegInf), s0,
                      qv && (c0 < max_valid_kv) && (c0 <= qpos_l));
          s1 = select(float4(kNegInf), s1,
                      qv && (c1 < max_valid_kv) && (c1 <= qpos_l));
          s2 = select(float4(kNegInf), s2,
                      qv && (c2 < max_valid_kv) && (c2 <= qpos_l));
          s3 = select(float4(kNegInf), s3,
                      qv && (c3 < max_valid_kv) && (c3 <= qpos_l));
        }

        const float4 m4 = max(max(s0, s1), max(s2, s3));
        float m_loc = max(max(m4.x, m4.y), max(m4.z, m4.w));
        m_loc = max(m_loc, simd_shuffle_xor(m_loc, ushort(1)));
        const float new_m = max(max_s, m_loc);
        const float fac =
            (max_s <= kNegInfHalf) ? 0.0f : fast::exp2(max_s - new_m);
        max_s = new_m;
        sum_s *= fac;

        float4 p0, p1, p2, p3;
        if (full_causal_free) {
          p0 = fast::exp2(s0 - new_m);
          p1 = fast::exp2(s1 - new_m);
          p2 = fast::exp2(s2 - new_m);
          p3 = fast::exp2(s3 - new_m);
        } else {
          const bool emp = (new_m <= kNegInfHalf);
          p0 = select(fast::exp2(s0 - new_m), float4(0.0f),
                      emp || (s0 <= kNegInfHalf));
          p1 = select(fast::exp2(s1 - new_m), float4(0.0f),
                      emp || (s1 <= kNegInfHalf));
          p2 = select(fast::exp2(s2 - new_m), float4(0.0f),
                      emp || (s2 <= kNegInfHalf));
          p3 = select(fast::exp2(s3 - new_m), float4(0.0f),
                      emp || (s3 <= kNegInfHalf));
        }

        const float4 ps = (p0 + p1) + (p2 + p3);
        float tsum = (ps.x + ps.y) + (ps.z + ps.w);
        tsum += simd_shuffle_xor(tsum, ushort(1));
        sum_s += tsum;

        if (q_half == 0u) {
          tg_fac[r_m] = fac;
        }

        // Wait until all 32 lanes of this d_part==0 SG finish reading its tg_scores slice
        // before overwriting the lower 1024 B of that same slice with P[16][32]!
        simdgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup PVVal4 *p_dst = reinterpret_cast<threadgroup PVVal4 *>(
            my_tg_p + q_r_in16 * 32u + q_half * 16u);
        p_dst[0] = PVVal4(p0);
        p_dst[1] = PVVal4(p1);
        p_dst[2] = PVVal4(p2);
        p_dst[3] = PVVal4(p3);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier #6

      // Step 3: Load P0, P1 & fac into registers, then Barrier #7 frees tg_work/tg_fac!
      const float fac0 = tg_fac[tm + uint(sm)];
      const float fac1 = tg_fac[tm + 8u + uint(sm)];
      const PVVal4 hp00 = *reinterpret_cast<const threadgroup PVVal4 *>(
          my_tg_p + uint(sm) * 32u + uint(sn));
      const PVVal4 hp01 = *reinterpret_cast<const threadgroup PVVal4 *>(
          my_tg_p + uint(sm + 8) * 32u + uint(sn));
      const PVVal4 hp10 = *reinterpret_cast<const threadgroup PVVal4 *>(
          my_tg_p + uint(sm) * 32u + 16u + uint(sn));
      const PVVal4 hp11 = *reinterpret_cast<const threadgroup PVVal4 *>(
          my_tg_p + uint(sm + 8) * 32u + 16u + uint(sn));
      const Frag8 P0(hp00.x, hp00.y, hp00.z, hp00.w, hp01.x, hp01.y, hp01.z,
                     hp01.w);
      const Frag8 P1(hp10.x, hp10.y, hp10.z, hp10.w, hp11.x, hp11.y, hp11.z,
                     hp11.w);
      threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier #7

      // Rescale O0..O7 in registers.
      const F8 fac_vec(fac0, fac0, fac0, fac0, fac1, fac1, fac1, fac1);
      O0 *= fac_vec;
      O1 *= fac_vec;
      O2 *= fac_vec;
      O3 *= fac_vec;
      O4 *= fac_vec;
      O5 *= fac_vec;
      O6 *= fac_vec;
      O7 *= fac_vec;

      // Direct-L1 Frag16x16 P x V (ZERO V threadgroup staging, ZERO PV barriers!)
      if constexpr (IsAffine4) {
        device const uchar *v_page_ptr =
            cache_value + ulong(phys_page) * page_bytes;
        device const ushort *v_codes_u16 =
            reinterpret_cast<device const ushort *>(v_page_ptr +
                                                    group_code_offset);
        device const KvAffine4VecMeta *v_meta =
            reinterpret_cast<device const KvAffine4VecMeta *>(v_page_ptr +
                                                              all_codes_bytes) +
            ulong(kvh) * p.page_tokens;

        const uint rv00 = sub_off + uint(sm);
        const uint rv01 = rv00 + 8u;
        const uint rv10 = rv00 + 16u;
        const uint rv11 = rv00 + 24u;
        half sv00 = v_meta[rv00].scale[d_part];
        half zv00 = v_meta[rv00].zero[d_part];
        half sv01 = v_meta[rv01].scale[d_part];
        half zv01 = v_meta[rv01].zero[d_part];
        half sv10 = v_meta[rv10].scale[d_part];
        half zv10 = v_meta[rv10].zero[d_part];
        half sv11 = v_meta[rv11].scale[d_part];
        half zv11 = v_meta[rv11].zero[d_part];
        if (valid_k < 32) {
          if (!(sm < valid_k)) { sv00 = half(0.0h); zv00 = half(0.0h); }
          if (!(short(sm + 8) < valid_k)) { sv01 = half(0.0h); zv01 = half(0.0h); }
          if (!(short(sm + 16) < valid_k)) { sv10 = half(0.0h); zv10 = half(0.0h); }
          if (!(short(sm + 24) < valid_k)) { sv11 = half(0.0h); zv11 = half(0.0h); }
        }
        device const ushort *vp00 = v_codes_u16 + rv00 * 32u + sn_u16;
        device const ushort *vp01 = v_codes_u16 + rv01 * 32u + sn_u16;
        device const ushort *vp10 = v_codes_u16 + rv10 * 32u + sn_u16;
        device const ushort *vp11 = v_codes_u16 + rv11 * 32u + sn_u16;

        simdgroup_barrier(mem_flags::mem_none);
#define SAGE2_PV_PAIR_AFF4(OA, OB, D_IDX)                                      \
  do {                                                                         \
    const uint c0 = uint(D_IDX) * 4u;                                          \
    const uint c1 = uint((D_IDX) + 1) * 4u;                                    \
    {                                                                          \
      const Frag8 V0 =                                                         \
          sage2_dequant_v_ptr_h(vp00, vp01, sv00, zv00, sv01, zv01, c0);       \
      const Frag8 V1 =                                                         \
          sage2_dequant_v_ptr_h(vp00, vp01, sv00, zv00, sv01, zv01, c1);       \
      sage2_mma_1x2_pv<half>(OA, OB, P0, V0, V1);                              \
    }                                                                          \
    {                                                                          \
      const Frag8 V0 =                                                         \
          sage2_dequant_v_ptr_h(vp10, vp11, sv10, zv10, sv11, zv11, c0);       \
      const Frag8 V1 =                                                         \
          sage2_dequant_v_ptr_h(vp10, vp11, sv10, zv10, sv11, zv11, c1);       \
      sage2_mma_1x2_pv<half>(OA, OB, P1, V0, V1);                              \
    }                                                                          \
  } while (false)

        SAGE2_PV_PAIR_AFF4(O0, O1, 0);
        SAGE2_PV_PAIR_AFF4(O2, O3, 2);
        SAGE2_PV_PAIR_AFF4(O4, O5, 4);
        SAGE2_PV_PAIR_AFF4(O6, O7, 6);
#undef SAGE2_PV_PAIR_AFF4
      } else {
        device const bfloat *v_bf =
            reinterpret_cast<device const bfloat *>(cache_value);
        const ulong base_row = ulong(phys_page) * p.page_tokens + sub_off;
        const ulong col_base = ulong(kvh) * BD + d_offset + uint(sn);
        device const bfloat *vp00 =
            v_bf + (base_row + uint(sm)) * bf16_kv_stride + col_base;
        device const bfloat *vp01 =
            v_bf + (base_row + uint(sm + 8)) * bf16_kv_stride + col_base;
        device const bfloat *vp10 =
            v_bf + (base_row + uint(sm + 16)) * bf16_kv_stride + col_base;
        device const bfloat *vp11 =
            v_bf + (base_row + uint(sm + 24)) * bf16_kv_stride + col_base;
        const bool ok00 = (sm < valid_k);
        const bool ok01 = (short(sm + 8) < valid_k);
        const bool ok10 = (short(sm + 16) < valid_k);
        const bool ok11 = (short(sm + 24) < valid_k);

        simdgroup_barrier(mem_flags::mem_none);
#define SAGE2_PV_PAIR_BF16(OA, OB, D_IDX)                                      \
  do {                                                                         \
    const uint c0 = uint(D_IDX) * 16u;                                         \
    const uint c1 = uint((D_IDX) + 1) * 16u;                                   \
    {                                                                          \
      const Frag8 V0 = sage2_load_v_ptr_bf(vp00, vp01, ok00, ok01, c0);        \
      const Frag8 V1 = sage2_load_v_ptr_bf(vp00, vp01, ok00, ok01, c1);        \
      sage2_mma_1x2_pv<bfloat>(OA, OB, P0, V0, V1);                            \
    }                                                                          \
    {                                                                          \
      const Frag8 V0 = sage2_load_v_ptr_bf(vp10, vp11, ok10, ok11, c0);        \
      const Frag8 V1 = sage2_load_v_ptr_bf(vp10, vp11, ok10, ok11, c1);        \
      sage2_mma_1x2_pv<bfloat>(OA, OB, P1, V0, V1);                            \
    }                                                                          \
  } while (false)

        SAGE2_PV_PAIR_BF16(O0, O1, 0);
        SAGE2_PV_PAIR_BF16(O2, O3, 2);
        SAGE2_PV_PAIR_BF16(O4, O5, 4);
        SAGE2_PV_PAIR_BF16(O6, O7, 6);
#undef SAGE2_PV_PAIR_BF16
      }
    }

    // Epilogue: vectorized bfloat4 gated output store!
    if (d_part == 0u && q_half == 0u) {
      tg_fac[r_m] = (sum_s > 0.0f) ? (1.0f / sum_s) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rm0 = tm + uint(sm);
    const uint rm1 = tm + 8u + uint(sm);
    const bool v0 = (rm0 < active_m_rows);
    const bool v1 = (rm1 < active_m_rows);
    uint tok0 = 0u, g0 = 0u, tok1 = 0u, g1 = 0u;
    if (v0) {
      fast_divmod_g(rm0, repeat, tok0, g0);
    }
    if (v1) {
      fast_divmod_g(rm1, repeat, tok1, g1);
    }
    const float rcp0 = v0 ? tg_fac[rm0] : 0.0f;
    const float rcp1 = v1 ? tg_fac[rm1] : 0.0f;
    const uint qh0 = kvh * repeat + g0;
    const uint qh1 = kvh * repeat + g1;
    device const bfloat *g_ptr0 =
        gate + ulong(q_base + tok0) * p.gate_token_stride +
        ulong(qh0) * p.gate_head_stride + d_offset + uint(sn);
    device const bfloat *g_ptr1 =
        gate + ulong(q_base + tok1) * p.gate_token_stride +
        ulong(qh1) * p.gate_head_stride + d_offset + uint(sn);
    device bfloat *o_ptr0 =
        output + (ulong(q_base + tok0) * p.query_heads + qh0) * BD + d_offset +
        uint(sn);
    device bfloat *o_ptr1 =
        output + (ulong(q_base + tok1) * p.query_heads + qh1) * BD + d_offset +
        uint(sn);

#define SAGE2_STORE_O_FRAG(OREG, D_IDX)                                        \
  do {                                                                         \
    const uint off = uint(D_IDX) * 16u;                                        \
    if (v0) {                                                                  \
      const float4 gv =                                                        \
          float4(*reinterpret_cast<device const bfloat4 *>(g_ptr0 + off));     \
      const float4 ov = (float4(OREG[0], OREG[1], OREG[2], OREG[3]) * rcp0) /  \
                        (float4(1.0f) + exp(-gv));                             \
      *reinterpret_cast<device bfloat4 *>(o_ptr0 + off) = bfloat4(ov);         \
    }                                                                          \
    if (v1) {                                                                  \
      const float4 gv =                                                        \
          float4(*reinterpret_cast<device const bfloat4 *>(g_ptr1 + off));     \
      const float4 ov = (float4(OREG[4], OREG[5], OREG[6], OREG[7]) * rcp1) /  \
                        (float4(1.0f) + exp(-gv));                             \
      *reinterpret_cast<device bfloat4 *>(o_ptr1 + off) = bfloat4(ov);         \
    }                                                                          \
  } while (false)

    SAGE2_STORE_O_FRAG(O0, 0);
    SAGE2_STORE_O_FRAG(O1, 1);
    SAGE2_STORE_O_FRAG(O2, 2);
    SAGE2_STORE_O_FRAG(O3, 3);
    SAGE2_STORE_O_FRAG(O4, 4);
    SAGE2_STORE_O_FRAG(O5, 5);
    SAGE2_STORE_O_FRAG(O6, 6);
    SAGE2_STORE_O_FRAG(O7, 7);
#undef SAGE2_STORE_O_FRAG
    return;
  }

  // --------------------------------------------------------------------------
  // Portable Fallback (Apple9 / non-Apple10): NUM_WAVES=1 with 4x 16x32x32 PV
  // inside tg_work (tg_v_sub[0..4095] + tg_p_all[4096..8191]).
  // --------------------------------------------------------------------------
  constexpr auto pv_sub_desc = matmul2d_descriptor(
      16, 32, 32, false, false, true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_sub_desc, execution_simdgroup> pv_sub_op;

  threadgroup PVValT *tg_v_sub_base =
      reinterpret_cast<threadgroup PVValT *>(work_b); // [0 .. 4095] (2x32x32)
  threadgroup PVValT *tg_p_all = reinterpret_cast<threadgroup PVValT *>(
      work_b + 4096u); // [4096 .. 8191] (4x16x32)

  tensor<threadgroup PVValT, dextents<int32_t, 2>, tensor_inline> tP_d(
      tg_p_all, dextents<int32_t, 2>(32, 16));
  tensor<threadgroup PVValT, dextents<int32_t, 2>, tensor_inline> tV_d(
      tg_v_sub_base, dextents<int32_t, 2>(32, 32));
  auto ct_O0 = pv_sub_op.template get_destination_cooperative_tensor<
      decltype(tP_d), decltype(tV_d), float>();
  auto ct_O1 = pv_sub_op.template get_destination_cooperative_tensor<
      decltype(tP_d), decltype(tV_d), float>();
  auto ct_O2 = pv_sub_op.template get_destination_cooperative_tensor<
      decltype(tP_d), decltype(tV_d), float>();
  auto ct_O3 = pv_sub_op.template get_destination_cooperative_tensor<
      decltype(tP_d), decltype(tV_d), float>();
  SAGE2_UNROLL
  for (ushort i = 0; i < ct_O0.get_capacity(); ++i) {
    ct_O0[i] = 0.0f;
    ct_O1[i] = 0.0f;
    ct_O2[i] = 0.0f;
    ct_O3[i] = 0.0f;
  }

  for (uint step = 0u; step < n_steps; ++step) {
    const uint kb =
        SparseMInference
            ? uint(active_indices[(kvh * sp.max_q_blocks + q_blk) *
                                      sp.active_stride +
                                  step])
            : step;
    if (kb >= nk_blocks) {
      continue;
    }

    const uint lpage = kb / half_blocks_per_page;
    const uint sub_off = (kb % half_blocks_per_page) * BK;
    const uint phys_page = page_table[lpage];
    const uint kb_col_base = kb * BK;
    const short valid_k = short(min(BK, max_valid_kv - kb_col_base));
    const bool full_causal_free =
        (kb < kb_min_causal) && (valid_k == 32) && warp_full_rows;

    prepare_k_and_delta_wm4<IsAffine4>(
        cache_key, phys_page, sub_off, p.page_tokens, page_bytes, kvh_code_base,
        all_codes_bytes, bf16_kv_stride, kvh, valid_k, flat_tid, row_group,
        d_part, q_r_in16, q_half, tg_q_i8_b, tg_work, tg_mu_k, tg_sk,
        tg_delta);

    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ(
        my_tg_q, dextents<int32_t, 2>(128, 16));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tK(
        tg_k_i8_part, dextents<int32_t, 2>(128, 32));
    auto ct_s = qk_op.template get_destination_cooperative_tensor<
        decltype(tQ), decltype(tK), int32_t>();
    qk_op.run(tQ, tK, ct_s);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (d_part == 1u) {
      SAGE2_UNROLL
      for (ushort i = 0; i < ct_s.get_capacity(); ++i) {
        if (ct_s.is_valid_element(i)) {
          const auto idx = ct_s.get_multidimensional_index(i);
          const uint col = uint(idx[0]);
          const uint row = uint(idx[1]);
          const float sq = tg_sq[1u * M_TILE + tm + row];
          const float sk = tg_sk[1u * 32u + col];
          const float del = tg_delta[1u * M_TILE + tm + row];
          tg_scores[row_group * 512u + row * 32u + col] =
              sq * fma(float(ct_s[i]), sk, del);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    PVVal4 p_reg[4];
    if (d_part == 0u) {
      SAGE2_UNROLL
      for (ushort i = 0; i < ct_s.get_capacity(); ++i) {
        if (ct_s.is_valid_element(i)) {
          const auto idx = ct_s.get_multidimensional_index(i);
          const uint col = uint(idx[0]);
          const uint row = uint(idx[1]);
          const float sq = tg_sq[0u * M_TILE + tm + row];
          const float sk = tg_sk[0u * 32u + col];
          const float del = tg_delta[0u * M_TILE + tm + row];
          tg_scores[row_group * 512u + row * 32u + col] +=
              sq * fma(float(ct_s[i]), sk, del);
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);

      const threadgroup float4 *s_row4 =
          reinterpret_cast<const threadgroup float4 *>(
              tg_scores + row_group * 512u + q_r_in16 * 32u + q_half * 16u);
      float4 s0 = s_row4[0] * scale2;
      float4 s1 = s_row4[1] * scale2;
      float4 s2 = s_row4[2] * scale2;
      float4 s3 = s_row4[3] * scale2;

      if (!full_causal_free) {
        const uint c_base = kb_col_base + q_half * 16u;
        const uint4 c0 = uint4(c_base) + uint4(0u, 1u, 2u, 3u);
        const uint4 c1 = c0 + uint4(4u);
        const uint4 c2 = c0 + uint4(8u);
        const uint4 c3 = c0 + uint4(12u);
        s0 = select(float4(kNegInf), s0,
                    qv && (c0 < max_valid_kv) && (c0 <= qpos_l));
        s1 = select(float4(kNegInf), s1,
                    qv && (c1 < max_valid_kv) && (c1 <= qpos_l));
        s2 = select(float4(kNegInf), s2,
                    qv && (c2 < max_valid_kv) && (c2 <= qpos_l));
        s3 = select(float4(kNegInf), s3,
                    qv && (c3 < max_valid_kv) && (c3 <= qpos_l));
      }

      const float4 m4 = max(max(s0, s1), max(s2, s3));
      float m_loc = max(max(m4.x, m4.y), max(m4.z, m4.w));
      m_loc = max(m_loc, simd_shuffle_xor(m_loc, ushort(1)));
      const float new_m = max(max_s, m_loc);
      const float fac =
          (max_s <= kNegInfHalf) ? 0.0f : fast::exp2(max_s - new_m);
      max_s = new_m;
      sum_s *= fac;

      float4 p0, p1, p2, p3;
      if (full_causal_free) {
        p0 = fast::exp2(s0 - new_m);
        p1 = fast::exp2(s1 - new_m);
        p2 = fast::exp2(s2 - new_m);
        p3 = fast::exp2(s3 - new_m);
      } else {
        const bool emp = (new_m <= kNegInfHalf);
        p0 = select(fast::exp2(s0 - new_m), float4(0.0f),
                    emp || (s0 <= kNegInfHalf));
        p1 = select(fast::exp2(s1 - new_m), float4(0.0f),
                    emp || (s1 <= kNegInfHalf));
        p2 = select(fast::exp2(s2 - new_m), float4(0.0f),
                    emp || (s2 <= kNegInfHalf));
        p3 = select(fast::exp2(s3 - new_m), float4(0.0f),
                    emp || (s3 <= kNegInfHalf));
      }

      const float4 ps = (p0 + p1) + (p2 + p3);
      float tsum = (ps.x + ps.y) + (ps.z + ps.w);
      tsum += simd_shuffle_xor(tsum, ushort(1));
      sum_s += tsum;

      if (q_half == 0u) {
        tg_fac[r_m] = fac;
      }
      p_reg[0] = PVVal4(p0);
      p_reg[1] = PVVal4(p1);
      p_reg[2] = PVVal4(p2);
      p_reg[3] = PVVal4(p3);
    }
    // Wait until all d_part==0 SGs finish reading tg_scores before writing tg_p_all[4096..8191]!
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (d_part == 0u) {
      threadgroup PVVal4 *p_dst = reinterpret_cast<threadgroup PVVal4 *>(
          tg_p_all + row_group * 512u + q_r_in16 * 32u + q_half * 16u);
      p_dst[0] = p_reg[0];
      p_dst[1] = p_reg[1];
      p_dst[2] = p_reg[2];
      p_dst[3] = p_reg[3];
    }

    SAGE2_UNROLL
    for (ushort i = 0; i < ct_O0.get_capacity(); ++i) {
      if (ct_O0.is_valid_element(i)) {
        const auto idx = ct_O0.get_multidimensional_index(i);
        const float f = tg_fac[tm + uint(idx[1])];
        ct_O0[i] *= f;
        ct_O1[i] *= f;
        ct_O2[i] *= f;
        ct_O3[i] *= f;
      }
    }

    tensor<threadgroup PVValT, dextents<int32_t, 2>, tensor_inline> tP(
        tg_p_all + row_group * 512u, dextents<int32_t, 2>(32, 16));
    tensor<threadgroup PVValT, dextents<int32_t, 2>, tensor_inline> tV_sub(
        tg_v_sub_base + d_part * (32u * 32u), dextents<int32_t, 2>(32, 32));

#define SAGE2_PORTABLE_PV_SUB(SUB_IDX, CT_O)                                   \
  do {                                                                         \
    for (uint it = 0u; it < 2u; ++it) {                                        \
      const uint idx = flat_tid + it * 256u;                                   \
      const uint p_g = idx >> 8u;                                              \
      const uint r = (idx >> 3u) & 31u;                                        \
      const uint c4 = idx & 7u;                                                \
      const float4 vf = load_kv_vec4_safe<IsAffine4>(                          \
          cache_value, phys_page, sub_off, p.page_tokens, page_bytes,          \
          kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, p_g, r,         \
          uint(SUB_IDX) * 8u + c4, valid_k);                                   \
      reinterpret_cast<threadgroup PVVal4 *>(tg_v_sub_base)[idx] = PVVal4(vf); \
    }                                                                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    pv_sub_op.run(tP, tV_sub, CT_O);                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
  } while (false)

    SAGE2_PORTABLE_PV_SUB(0, ct_O0);
    SAGE2_PORTABLE_PV_SUB(1, ct_O1);
    SAGE2_PORTABLE_PV_SUB(2, ct_O2);
    SAGE2_PORTABLE_PV_SUB(3, ct_O3);
#undef SAGE2_PORTABLE_PV_SUB
  }

  if (d_part == 0u && q_half == 0u) {
    tg_fac[r_m] = (sum_s > 0.0f) ? (1.0f / sum_s) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  sage2_store_ct_output(ct_O0, gate, output, tg_fac, sp.base, tm, active_m_rows,
                        repeat, kvh, q_base, d_offset + 0u);
  sage2_store_ct_output(ct_O1, gate, output, tg_fac, sp.base, tm, active_m_rows,
                        repeat, kvh, q_base, d_offset + 32u);
  sage2_store_ct_output(ct_O2, gate, output, tg_fac, sp.base, tm, active_m_rows,
                        repeat, kvh, q_base, d_offset + 64u);
  sage2_store_ct_output(ct_O3, gate, output, tg_fac, sp.base, tm, active_m_rows,
                        repeat, kvh, q_base, d_offset + 96u);
}

} // namespace sage2

// ============================================================================
// Diagnostic Identity & Lane-Coordinate Probe Kernels (Verified in Unit Tests)
// ============================================================================
kernel void omlx_m5_sage2_probe_int8_16x32x128(
    device const int8_t *in_q [[buffer(0)]],
    device const int8_t *in_k [[buffer(1)]],
    device int32_t *out_s [[buffer(2)]],
    device uint *out_coords [[buffer(3)]],
    constant uint &dummy [[buffer(4)]],
    ushort lane [[thread_index_in_simdgroup]]) {
  if (dummy != 0u) return;
  threadgroup int8_t tg_q[16 * 128];
  threadgroup int8_t tg_k[32 * 128];
  for (uint i = lane; i < 16u * 128u; i += 32u) tg_q[i] = in_q[i];
  for (uint i = lane; i < 32u * 128u; i += 32u) tg_k[i] = in_k[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ(
      tg_q, dextents<int32_t, 2>(128, 16));
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tK(
      tg_k, dextents<int32_t, 2>(128, 32));
  auto ct_s =
      op.get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
  op.run(tQ, tK, ct_s);

  const ushort cap = ct_s.get_capacity();
  for (ushort i = 0; i < cap; ++i) {
    if (ct_s.is_valid_element(i)) {
      const auto idx = ct_s.get_multidimensional_index(i);
      const uint col = uint(idx[0]);
      const uint row = uint(idx[1]);
      out_s[row * 32u + col] = ct_s[i];
      if (i < 16u) {
        out_coords[(uint(lane) * 16u + uint(i)) * 2u + 0u] = row;
        out_coords[(uint(lane) * 16u + uint(i)) * 2u + 1u] = col;
      }
    }
  }
}

kernel void omlx_m5_sage2_probe_mxfp8_16x128x32(
    device const float *in_p [[buffer(0)]],
    device const float *in_v [[buffer(1)]],
    device float *out_o [[buffer(2)]],
    constant uint &dummy [[buffer(3)]],
    ushort lane [[thread_index_in_simdgroup]]) {
  if (dummy != 0u) return;
  threadgroup uchar tg_p[16 * 32];
  threadgroup uchar tg_vt[128 * 32];
  threadgroup uchar tg_sc[128];

  // Pack P * 448.0f -> FP8 E4M3
  for (uint idx = lane; idx < 16u * 8u; idx += 32u) {
    const uint base = idx * 4u;
    const float4 v(in_p[base], in_p[base + 1u], in_p[base + 2u],
                   in_p[base + 3u]);
    reinterpret_cast<threadgroup packed_uchar4 *>(tg_p)[idx] =
        pack<metal_fp8_e4m3_format>(v * sage2::kFp8E4M3Max).as_storage_type();
  }
  // Transpose V [32, 128] -> V^T [128, 32] with per-channel UE8M0 scale
  for (uint d = lane; d < 128u; d += 32u) {
    float vmax = 0.0f;
    for (uint k = 0u; k < 32u; ++k) {
      vmax = max(vmax, fabs(in_v[k * 128u + d]));
    }
    float inv_sc = 0.0f;
    tg_sc[d] = sage2::compute_ue8m0_scale(vmax, inv_sc);
    threadgroup packed_uchar4 *dst =
        reinterpret_cast<threadgroup packed_uchar4 *>(tg_vt + d * 32u);
    for (uint k4 = 0u; k4 < 8u; ++k4) {
      const uint k = k4 * 4u;
      const float4 vv(in_v[(k + 0u) * 128u + d], in_v[(k + 1u) * 128u + d],
                      in_v[(k + 2u) * 128u + d], in_v[(k + 3u) * 128u + d]);
      dst[k4] = pack<metal_fp8_e4m3_format>(vv * inv_sc).as_storage_type();
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr auto desc = matmul2d_descriptor(
      16, 128, 32, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  sage2::TgFp8Plain tP(tg_p, dextents<int32_t, 2>(32, 16));
  sage2::TgMxfp8VT tVT(tg_vt, dextents<int32_t, 2>(32, 128),
                       sage2::TgUe8ScaleTag(tg_sc));
  auto ct_o =
      op.get_destination_cooperative_tensor<decltype(tP), decltype(tVT), float>();
  for (ushort i = 0; i < ct_o.get_capacity(); ++i) ct_o[i] = 0.0f;
  op.run(tP, tVT, ct_o);
  for (ushort i = 0; i < ct_o.get_capacity(); ++i) {
    if (ct_o.is_valid_element(i)) {
      const auto idx = ct_o.get_multidimensional_index(i);
      out_o[uint(idx[1]) * 128u + uint(idx[0])] =
          ct_o[i] * sage2::kInvFp8E4M3Max;
    }
  }
}

// ============================================================================
// SageAttention2 Production Kernels (Dense & MInference Sparse; MXFP8 & FP16/BF16 PV)
// SRAM strictly <= 27.0 KiB across all tiers (WM in {4, 6, 8})!
// ============================================================================
#define DEFINE_SAGE2_KERNEL(NAME, WM_VAL, IS_AFF4, USE_MXFP8, SPARSE_MINF)     \
  kernel void NAME(                                                            \
      device const bfloat *query [[buffer(0)]],                                \
      device const bfloat *gate [[buffer(1)]],                                 \
      device const uchar *cache_key [[buffer(2)]],                             \
      device const uchar *cache_value [[buffer(3)]],                           \
      device bfloat *output [[buffer(4)]],                                     \
      device const uint *page_table [[buffer(5)]],                             \
      device const uint *positions [[buffer(6)]],                              \
      device const ushort *active_indices [[buffer(8)]],                       \
      device const uint *active_counts [[buffer(9)]],                          \
      constant Sage2AttentionParams &sp [[buffer(10)]],                        \
      ushort lane [[thread_index_in_simdgroup]],                               \
      ushort sg [[simdgroup_index_in_threadgroup]],                            \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup uint4 tg_stage[20480u / 16u];                                  \
    threadgroup float tg_scores[2u * 16u * 32u];                               \
    threadgroup float tg_mu_k[2u * 128u];                                      \
    threadgroup float tg_sk[2u * 32u];                                         \
    threadgroup float tg_sq[2u * (WM_VAL) * 16u];                              \
    threadgroup float tg_delta[2u * 2u * 16u];                                 \
    threadgroup float tg_fac[(WM_VAL) * 16u];                                  \
    sage2::paged_flash_prefill_sage2_hd256_impl<WM_VAL, IS_AFF4, USE_MXFP8,    \
                                                SPARSE_MINF>(                  \
        query, gate, cache_key, cache_value, output, page_table, positions,    \
        active_indices, active_counts, sp, tg_stage, tg_scores, tg_mu_k,       \
        tg_sk, tg_sq, tg_delta, tg_fac, lane, sg, tg);                         \
  }

#define DEFINE_SAGE2_WM4_FP16_KERNEL(NAME, IS_AFF4, SPARSE_MINF)               \
  kernel void NAME(                                                            \
      device const bfloat *query [[buffer(0)]],                                \
      device const bfloat *gate [[buffer(1)]],                                 \
      device const uchar *cache_key [[buffer(2)]],                             \
      device const uchar *cache_value [[buffer(3)]],                           \
      device bfloat *output [[buffer(4)]],                                     \
      device const uint *page_table [[buffer(5)]],                             \
      device const uint *positions [[buffer(6)]],                              \
      device const ushort *active_indices [[buffer(8)]],                       \
      device const uint *active_counts [[buffer(9)]],                          \
      constant Sage2AttentionParams &sp [[buffer(10)]],                        \
      ushort lane [[thread_index_in_simdgroup]],                               \
      ushort sg [[simdgroup_index_in_threadgroup]],                            \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup uint4 tg_q_i8[16384u / 16u];                                   \
    threadgroup uint4 tg_work[8192u / 16u];                                    \
    threadgroup float tg_mu_k[2u * 128u];                                      \
    threadgroup float tg_sk[2u * 32u];                                         \
    threadgroup float tg_sq[2u * 64u];                                         \
    threadgroup float tg_delta[2u * 64u];                                      \
    threadgroup float tg_fac[64u];                                             \
    sage2::paged_flash_prefill_sage2_wm4_fp16_impl<IS_AFF4, SPARSE_MINF>(      \
        query, gate, cache_key, cache_value, output, page_table, positions,    \
        active_indices, active_counts, sp, tg_q_i8, tg_work, tg_mu_k, tg_sk,   \
        tg_sq, tg_delta, tg_fac, lane, sg, tg);                                \
  }

// Affine4Group128 KV — Dense SageAttention2 (MXFP8 PV & FP16 PV)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_mxfp8_affine4_g128_hd256_wm4, 4u, true, true, false)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_mxfp8_affine4_g128_hd256_wm6, 6u, true, true, false)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_mxfp8_affine4_g128_hd256_wm8, 8u, true, true, false)
DEFINE_SAGE2_WM4_FP16_KERNEL(omlx_m5_paged_flash_prefill_sage2_fp16_affine4_g128_hd256_wm4, true, false)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_fp16_affine4_g128_hd256_wm6, 6u, true, false, false)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_fp16_affine4_g128_hd256_wm8, 8u, true, false, false)

// Affine4Group128 KV — Sparse MInference + SageAttention2 (MXFP8 PV & FP16 PV)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_affine4_g128_hd256_wm4, 4u, true, true, true)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_affine4_g128_hd256_wm6, 6u, true, true, true)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_affine4_g128_hd256_wm8, 8u, true, true, true)
DEFINE_SAGE2_WM4_FP16_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_fp16_affine4_g128_hd256_wm4, true, true)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_fp16_affine4_g128_hd256_wm6, 6u, true, false, true)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_fp16_affine4_g128_hd256_wm8, 8u, true, false, true)

// BF16 KV — Dense SageAttention2 (MXFP8 PV & BF16 PV)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_mxfp8_bf16_hd256_wm4, 4u, false, true, false)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_mxfp8_bf16_hd256_wm6, 6u, false, true, false)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_mxfp8_bf16_hd256_wm8, 8u, false, true, false)
DEFINE_SAGE2_WM4_FP16_KERNEL(omlx_m5_paged_flash_prefill_sage2_fp16_bf16_hd256_wm4, false, false)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_fp16_bf16_hd256_wm6, 6u, false, false, false)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_fp16_bf16_hd256_wm8, 8u, false, false, false)

// BF16 KV — Sparse MInference + SageAttention2 (MXFP8 PV & BF16 PV)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_bf16_hd256_wm4, 4u, false, true, true)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_bf16_hd256_wm6, 6u, false, true, true)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_bf16_hd256_wm8, 8u, false, true, true)
DEFINE_SAGE2_WM4_FP16_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_fp16_bf16_hd256_wm4, false, true)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_fp16_bf16_hd256_wm6, 6u, false, false, true)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_fp16_bf16_hd256_wm8, 8u, false, false, true)

#undef DEFINE_SAGE2_WM4_FP16_KERNEL
#undef DEFINE_SAGE2_KERNEL
