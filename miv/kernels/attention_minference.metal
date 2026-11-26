// Module 2 Route B.2: MInference Vertical-Slash Block-Sparse Index Builder
// (Metal 4.1 / MacOSX27.2 SDK)
//   - Last-W_probe=64 INT8 smoothed Q x K^T probe with exact +Delta_{i,b,p}
//     restoration and per-token s_{K,j,p}.
//   - GQA max_{g=0..G-1} pooling over exact softmax-normalized probe
//     probabilities so packed kvh TGs share one active block stream.
//   - Mandatory N_sink=4 (128 tok) + N_local=32 (1024 tok) retention.
//   - 1 KiB Threadgroup SRAM bitmask[256] + parallel popcount/ctz builder
//     emitting strictly ascending uint16_t active_indices per (kvh, q_blk).
//   - Reads explicit positions buffer (100% CacheBlend-compatible!).
#include <metal_atomic>
#include <metal_simdgroup>
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define MINF_UNROLL _Pragma("clang loop unroll(full)")
#define MINF_ALWAYS_INLINE __attribute__((always_inline))

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

struct MInferenceParams {
  PagedAttentionParams base;
  uint w_probe;
  uint q_probe_start;
  uint split_count;
  uint split_blocks;
  uint max_kv_blocks;
  uint bm_tok;
  uint n_q_blocks;
  uint max_q_blocks;
  uint active_stride;
  uint n_sink;
  uint n_local;
  uint k_vertical;
  uint k_slash;
  uint top_stride;
};

struct KvAffine4VecMeta {
  half scale[4];
  half zero[4];
};

namespace minf {

constant constexpr float kNegInf = -3.402823466e+38f;
constant constexpr float kNegInfHalf = -1.701411733e+38f;
constant constexpr float kLog2e = 1.4426950408889634f;

MINF_ALWAYS_INLINE
static inline bfloat4 clean_bf4(bfloat4 v) {
  const ushort4 bits = as_type<ushort4>(v);
  return select(v, bfloat4(bfloat(0.0f)),
                (bits & ushort4(0x7fffu)) > ushort4(0x7f80u));
}

MINF_ALWAYS_INLINE
static inline float4 clean_f4(float4 v) {
  const uint4 bits = as_type<uint4>(v);
  return select(v, float4(0.0f),
                (bits & uint4(0x7f800000u)) == uint4(0x7f800000u));
}

MINF_ALWAYS_INLINE
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

MINF_ALWAYS_INLINE
static inline ulong affine4_page_byte_stride(uint page_tokens, uint kv_heads,
                                             uint head_dim) {
  const ulong vec_bytes = ulong(head_dim / 2u) + 16ul;
  const ulong head_bytes = ulong(page_tokens) * vec_bytes + 32ul;
  return ulong(kv_heads) * head_bytes;
}

template <bool IsAffine4>
MINF_ALWAYS_INLINE
static inline float4 load_k_vec4_safe(
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

template <bool IsAffine4>
MINF_ALWAYS_INLINE
static inline void probe_prepare_k_int8(
    device const uchar *cache_key, uint phys_page, uint sub_off,
    uint page_tokens, ulong page_bytes, ulong kvh_code_base,
    ulong all_codes_bytes, ulong bf16_kv_stride, uint kvh, short valid_k,
    uint flat_tid, threadgroup uint4 *tg_stage, threadgroup float *tg_mu_k,
    threadgroup float *tg_sk) {
  threadgroup float4 *tg_mu_tmp =
      reinterpret_cast<threadgroup float4 *>(tg_stage) + 512u; // offset 8192 B
  threadgroup char4 *tg_k_i8_v4 =
      reinterpret_cast<threadgroup char4 *>(tg_stage); // offset 0 B

  float4 k_reg[8];
  const uint r_chunk = flat_tid >> 6u; // 0..3 (256 threads)
  const uint v4_id = flat_tid & 63u;   // 0..63
  const uint p_g = v4_id >> 5u;        // 0..1
  const uint col_v4 = v4_id & 31u;     // 0..31 (= lane)

  float4 sum4(0.0f);
  MINF_UNROLL
  for (uint rr = 0u; rr < 8u; ++rr) {
    const uint r = r_chunk * 8u + rr;
    k_reg[rr] = load_k_vec4_safe<IsAffine4>(
        cache_key, phys_page, sub_off, page_tokens, page_bytes, kvh_code_base,
        all_codes_bytes, bf16_kv_stride, kvh, p_g, r, col_v4, valid_k);
    sum4 += k_reg[rr];
  }
  tg_mu_tmp[r_chunk * 64u + v4_id] = sum4;
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

  const float4 mu4 = reinterpret_cast<threadgroup float4 *>(tg_mu_k)[v4_id];
  MINF_UNROLL
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
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Unified INT8 Q_probe x K^T probe engine (WM=4, 256 threads, <= 24.5 KiB SRAM).
// PassMode == 0: compute per-split (max_s, sum_s) -> probe_m_split, probe_l_split.
// PassMode == 1: read global (M_global, inv_Z), pool GQA max_{g=0..G-1}, atomic-add
//                normalized probabilities into V_score and D_score.
template <bool IsAffine4, uint PassMode>
MINF_ALWAYS_INLINE
static inline void minference_probe_impl(
    device const bfloat *query, device const uchar *cache_key,
    device const uint *page_table, device const uint *positions,
    device float *probe_m_split, device float *probe_l_split,
    device const float *probe_m_global, device const float *probe_inv_z,
    device atomic<float> *v_score, device atomic<float> *d_score,
    constant MInferenceParams &mp, threadgroup uint4 *tg_stage,
    threadgroup float *tg_scores, threadgroup float *tg_mu_k,
    threadgroup float *tg_sk, threadgroup float *tg_sq,
    threadgroup float *tg_delta, threadgroup uint *tg_q_pos,
    threadgroup float *tg_pool_buf, ushort lane, ushort sg, uint3 tg) {
  constexpr uint BD = 256u;
  constexpr uint BK = 32u;
  constexpr uint M_TILE = 64u;
  constexpr uint STAGE_RG = 2u;
  constexpr uint NUM_WAVES = 2u;

  const PagedAttentionParams p = mp.base;
  if (mp.w_probe == 0u || p.query_heads == 0u || p.key_value_heads == 0u) {
    return;
  }
  const uint repeat = p.query_heads / p.key_value_heads;
  if (repeat == 0u || repeat > M_TILE) {
    return;
  }
  const uint bm_tok_p = M_TILE / repeat;
  const uint n_probe_qblks = (mp.w_probe + bm_tok_p - 1u) / bm_tok_p;

  const uint split = tg.x;
  const uint p_blk = tg.y / p.key_value_heads;
  const uint kvh = tg.y - p_blk * p.key_value_heads;
  if (split >= mp.split_count || p_blk >= n_probe_qblks) {
    return;
  }

  const uint w_base = p_blk * bm_tok_p; // offset inside [0, w_probe)
  const uint q_base = mp.q_probe_start + w_base;
  const uint active_tok = min(bm_tok_p, mp.w_probe - w_base);
  const uint active_m_rows = active_tok * repeat;
  const uint flat_tid = uint(sg) * 32u + uint(lane);

  if (flat_tid < active_tok) {
    tg_q_pos[flat_tid] = positions[q_base + flat_tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint max_valid_kv = 0u;
  for (uint r = 0u; r < active_tok; ++r) {
    max_valid_kv = max(max_valid_kv, tg_q_pos[r] + 1u);
  }
  const uint nk_blocks = min((max_valid_kv + BK - 1u) / BK, mp.max_kv_blocks);
  const uint kb_start = split * mp.split_blocks;
  const uint kb_end = min(nk_blocks, kb_start + mp.split_blocks);

  const uint row_group = uint(sg >> 1u); // 0..3
  const uint d_part = uint(sg & 1u);     // 0..1
  const uint my_wave = row_group / STAGE_RG;
  const uint rg_local = row_group - my_wave * STAGE_RG;
  const uint tm = row_group * 16u;
  const uint d_offset = d_part * 128u;

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

  // Empty split fast exit!
  if (kb_start >= kb_end) {
    if constexpr (PassMode == 0u) {
      if (d_part == 0u && q_half == 0u && qv) {
        const ulong rec =
            (ulong(w_base + tok_l) * p.query_heads + qh_l) * mp.split_count +
            split;
        probe_m_split[rec] = kNegInf;
        probe_l_split[rec] = 0.0f;
      }
    }
    return;
  }

  // Quantize probe Q once into 16 char4 registers.
  char4 q_i8_reg[16];
  {
    float4 q_f4[16];
    float lmax = 0.0f;
    if (qv) {
      device const bfloat4 *q_ptr = reinterpret_cast<device const bfloat4 *>(
          query + ulong(q_base + tok_l) * p.query_token_stride +
          ulong(qh_l) * p.query_head_stride + d_offset + q_half * 64u);
      MINF_UNROLL
      for (uint k = 0u; k < 16u; ++k) {
        const float4 v = float4(clean_bf4(q_ptr[k]));
        q_f4[k] = v;
        const float4 av = fabs(v);
        lmax = max(lmax, max(max(av.x, av.y), max(av.z, av.w)));
      }
    } else {
      MINF_UNROLL
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
    MINF_UNROLL
    for (uint k = 0u; k < 16u; ++k) {
      const float4 qf =
          clamp(rint(q_f4[k] * inv_sq), float4(-127.0f), float4(127.0f));
      q_i8_reg[k] = char4(char(int(qf.x)), char(int(qf.y)), char(int(qf.z)),
                          char(int(qf.w)));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr auto qk_desc = matmul2d_descriptor(
      16, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<qk_desc, execution_simdgroup> qk_op;

  float max_s = kNegInf;
  float sum_s = 0.0f;
  float g_m_row = kNegInf;
  float g_invz_row = 0.0f;
  if constexpr (PassMode == 1u) {
    if (d_part == 0u && qv) {
      const ulong row_id = ulong(w_base + tok_l) * p.query_heads + qh_l;
      g_m_row = probe_m_global[row_id];
      g_invz_row = probe_inv_z[row_id];
    }
  }

  const float scale2 = 0.0625f * kLog2e;
  const uint half_blocks_per_page = p.page_tokens / BK;
  const ulong page_bytes =
      affine4_page_byte_stride(p.page_tokens, p.key_value_heads, BD);
  const ulong head_code_bytes = ulong(p.page_tokens) * 128ul;
  const ulong all_codes_bytes = ulong(p.key_value_heads) * head_code_bytes;
  const ulong kvh_code_base = ulong(kvh) * head_code_bytes;
  const ulong bf16_kv_stride = ulong(p.key_value_heads) * BD;

  threadgroup uchar *stage_b = reinterpret_cast<threadgroup uchar *>(tg_stage);
  threadgroup int8_t *tg_k_i8_part =
      reinterpret_cast<threadgroup int8_t *>(stage_b) + d_part * (32u * 128u);
  threadgroup int8_t *tg_q_stage_base =
      reinterpret_cast<threadgroup int8_t *>(stage_b + 8192u);
  threadgroup float *tg_v_row = tg_pool_buf;         // [64]
  threadgroup float *tg_dhi_row = tg_pool_buf + 64u; // [64]
  threadgroup float *tg_dlo_row = tg_pool_buf + 128u; // [64]

  for (uint kb = kb_start; kb < kb_end; ++kb) {
    const uint lpage = kb / half_blocks_per_page;
    const uint sub_off = (kb % half_blocks_per_page) * BK;
    const uint phys_page = page_table[lpage];
    const uint kb_col_base = kb * BK;
    const short valid_k = short(min(BK, max_valid_kv - kb_col_base));

    probe_prepare_k_int8<IsAffine4>(
        cache_key, phys_page, sub_off, p.page_tokens, page_bytes, kvh_code_base,
        all_codes_bytes, bf16_kv_stride, kvh, valid_k, flat_tid, tg_stage,
        tg_mu_k, tg_sk);

    MINF_UNROLL
    for (uint wave = 0u; wave < NUM_WAVES; ++wave) {
      const bool in_wave = (my_wave == wave);
      if (in_wave) {
        threadgroup char4 *q_dst = reinterpret_cast<threadgroup char4 *>(
            tg_q_stage_base + (rg_local * 2u + d_part) * (16u * 128u) +
            q_r_in16 * 128u + q_half * 64u);
        const threadgroup float4 *mu_src =
            reinterpret_cast<const threadgroup float4 *>(
                tg_mu_k + d_part * 128u + q_half * 64u);
        float d_acc = 0.0f;
        MINF_UNROLL
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
          MINF_UNROLL
          for (ushort i = 0; i < ct_s.get_capacity(); ++i) {
            if (ct_s.is_valid_element(i)) {
              const auto idx = ct_s.get_multidimensional_index(i);
              const uint col = uint(idx[0]);
              const uint row = uint(idx[1]);
              const float sq = tg_sq[1u * M_TILE + tm + row];
              const float sk = tg_sk[1u * 32u + col];
              const float del =
                  tg_delta[1u * (STAGE_RG * 16u) + rg_local * 16u + row];
              tg_scores[rg_local * 512u + row * 32u + col] =
                  sq * fma(float(ct_s[i]), sk, del);
            }
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (in_wave && d_part == 0u) {
        MINF_UNROLL
        for (ushort i = 0; i < ct_s.get_capacity(); ++i) {
          if (ct_s.is_valid_element(i)) {
            const auto idx = ct_s.get_multidimensional_index(i);
            const uint col = uint(idx[0]);
            const uint row = uint(idx[1]);
            const float sq = tg_sq[0u * M_TILE + tm + row];
            const float sk = tg_sk[0u * 32u + col];
            const float del =
                tg_delta[0u * (STAGE_RG * 16u) + rg_local * 16u + row];
            tg_scores[rg_local * 512u + row * 32u + col] +=
                sq * fma(float(ct_s[i]), sk, del);
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup float4 *s_row4 =
            reinterpret_cast<const threadgroup float4 *>(
                tg_scores + rg_local * 512u + q_r_in16 * 32u + q_half * 16u);
        const uint c_base = kb_col_base + q_half * 16u;
        const uint4 c0 = uint4(c_base) + uint4(0u, 1u, 2u, 3u);
        const uint4 c1 = c0 + uint4(4u);
        const uint4 c2 = c0 + uint4(8u);
        const uint4 c3 = c0 + uint4(12u);
        const bool4 ok0 = qv && (c0 < max_valid_kv) && (c0 <= qpos_l);
        const bool4 ok1 = qv && (c1 < max_valid_kv) && (c1 <= qpos_l);
        const bool4 ok2 = qv && (c2 < max_valid_kv) && (c2 <= qpos_l);
        const bool4 ok3 = qv && (c3 < max_valid_kv) && (c3 <= qpos_l);
        const float4 s0 = select(float4(kNegInf), s_row4[0] * scale2, ok0);
        const float4 s1 = select(float4(kNegInf), s_row4[1] * scale2, ok1);
        const float4 s2 = select(float4(kNegInf), s_row4[2] * scale2, ok2);
        const float4 s3 = select(float4(kNegInf), s_row4[3] * scale2, ok3);

        if constexpr (PassMode == 0u) {
          const float4 m4 = max(max(s0, s1), max(s2, s3));
          float m_loc = max(max(m4.x, m4.y), max(m4.z, m4.w));
          m_loc = max(m_loc, simd_shuffle_xor(m_loc, ushort(1)));
          const float new_m = max(max_s, m_loc);
          const float fac =
              (max_s <= kNegInfHalf) ? 0.0f : fast::exp2(max_s - new_m);
          max_s = new_m;
          sum_s *= fac;
          const bool emp = (new_m <= kNegInfHalf);
          const float4 p0 = select(fast::exp2(s0 - new_m), float4(0.0f),
                                   emp || (s0 <= kNegInfHalf));
          const float4 p1 = select(fast::exp2(s1 - new_m), float4(0.0f),
                                   emp || (s1 <= kNegInfHalf));
          const float4 p2 = select(fast::exp2(s2 - new_m), float4(0.0f),
                                   emp || (s2 <= kNegInfHalf));
          const float4 p3 = select(fast::exp2(s3 - new_m), float4(0.0f),
                                   emp || (s3 <= kNegInfHalf));
          const float4 ps = (p0 + p1) + (p2 + p3);
          float tsum = (ps.x + ps.y) + (ps.z + ps.w);
          tsum += simd_shuffle_xor(tsum, ushort(1));
          sum_s += tsum;
        } else {
          // PassMode == 1: Exact normalized probabilities -> block & 2-bin slash sums!
          const bool emp = (g_m_row <= kNegInfHalf) || !(g_invz_row > 0.0f);
          const float4 p0 = select(fast::exp2(s0 - g_m_row) * g_invz_row,
                                   float4(0.0f), emp || (s0 <= kNegInfHalf));
          const float4 p1 = select(fast::exp2(s1 - g_m_row) * g_invz_row,
                                   float4(0.0f), emp || (s1 <= kNegInfHalf));
          const float4 p2 = select(fast::exp2(s2 - g_m_row) * g_invz_row,
                                   float4(0.0f), emp || (s2 <= kNegInfHalf));
          const float4 p3 = select(fast::exp2(s3 - g_m_row) * g_invz_row,
                                   float4(0.0f), emp || (s3 <= kNegInfHalf));

          const uint split_c =
              (qpos_l >= kb_col_base) ? (((qpos_l - kb_col_base) & 31u) + 1u)
                                      : 0u;
          const uint j_base = q_half * 16u;
          const uint4 j0 = uint4(j_base) + uint4(0u, 1u, 2u, 3u);
          const uint4 j1 = j0 + uint4(4u);
          const uint4 j2 = j0 + uint4(8u);
          const uint4 j3 = j0 + uint4(12u);

          const float4 hi_s = select(float4(0.0f), p0, j0 < split_c) +
                              select(float4(0.0f), p1, j1 < split_c) +
                              select(float4(0.0f), p2, j2 < split_c) +
                              select(float4(0.0f), p3, j3 < split_c);
          const float4 lo_s = select(float4(0.0f), p0, j0 >= split_c) +
                              select(float4(0.0f), p1, j1 >= split_c) +
                              select(float4(0.0f), p2, j2 >= split_c) +
                              select(float4(0.0f), p3, j3 >= split_c);

          float dhi = (hi_s.x + hi_s.y) + (hi_s.z + hi_s.w);
          float dlo = (lo_s.x + lo_s.y) + (lo_s.z + lo_s.w);
          dhi += simd_shuffle_xor(dhi, ushort(1));
          dlo += simd_shuffle_xor(dlo, ushort(1));
          if (q_half == 0u) {
            tg_v_row[r_m] = dhi + dlo;
            tg_dhi_row[r_m] = dhi;
            tg_dlo_row[r_m] = dlo;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // PassMode == 1: GQA max_{g=0..G-1} pooling across the repeat heads for each token!
    if constexpr (PassMode == 1u) {
      if (flat_tid < active_tok) {
        const uint pos_w = tg_q_pos[flat_tid];
        if (pos_w >= kb_col_base) {
          float v_max = 0.0f;
          float dhi_max = 0.0f;
          float dlo_max = 0.0f;
          const uint r_base = flat_tid * repeat;
          for (uint g = 0u; g < repeat; ++g) {
            v_max = max(v_max, tg_v_row[r_base + g]);
            dhi_max = max(dhi_max, tg_dhi_row[r_base + g]);
            dlo_max = max(dlo_max, tg_dlo_row[r_base + g]);
          }
          const ulong head_off = ulong(kvh) * mp.max_kv_blocks;
          if (v_max > 0.0f) {
            atomic_fetch_add_explicit(v_score + head_off + kb, v_max,
                                      memory_order_relaxed);
          }
          const uint delta_hi = (pos_w - kb_col_base) >> 5u;
          if (dhi_max > 0.0f && delta_hi < mp.max_kv_blocks) {
            atomic_fetch_add_explicit(d_score + head_off + delta_hi, dhi_max,
                                      memory_order_relaxed);
          }
          if (dlo_max > 0.0f && delta_hi > 0u &&
              (delta_hi - 1u) < mp.max_kv_blocks) {
            atomic_fetch_add_explicit(d_score + head_off + (delta_hi - 1u),
                                      dlo_max, memory_order_relaxed);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  if constexpr (PassMode == 0u) {
    if (d_part == 0u && q_half == 0u && qv) {
      const ulong rec =
          (ulong(w_base + tok_l) * p.query_heads + qh_l) * mp.split_count +
          split;
      probe_m_split[rec] = max_s;
      probe_l_split[rec] = sum_s;
    }
  }
}

} // namespace minf

// ============================================================================
// Stage 1A: Probe Split-KV Partition Stats (Affine4 & BF16 KV)
// Grid: {split_count * 256, ceil(w_probe / bm_tok_p) * kvh, 1}, tptg: {256, 1, 1}
// ============================================================================
#define DEFINE_MINF_PROBE_STATS(NAME, IS_AFF4)                                 \
  kernel void NAME(                                                            \
      device const bfloat *query [[buffer(0)]],                                \
      device const uchar *cache_key [[buffer(1)]],                             \
      device const uint *page_table [[buffer(2)]],                             \
      device const uint *positions [[buffer(3)]],                              \
      device float *probe_m_split [[buffer(4)]],                               \
      device float *probe_l_split [[buffer(5)]],                               \
      constant MInferenceParams &mp [[buffer(6)]],                             \
      ushort lane [[thread_index_in_simdgroup]],                               \
      ushort sg [[simdgroup_index_in_threadgroup]],                            \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup uint4 tg_stage[16384u / 16u];                                  \
    threadgroup float tg_scores[2u * 16u * 32u];                               \
    threadgroup float tg_mu_k[2u * 128u];                                      \
    threadgroup float tg_sk[2u * 32u];                                         \
    threadgroup float tg_sq[2u * 64u];                                         \
    threadgroup float tg_delta[2u * 32u];                                      \
    threadgroup uint tg_q_pos[64u];                                            \
    threadgroup float tg_pool_unused[1u];                                      \
    minf::minference_probe_impl<IS_AFF4, 0u>(                                  \
        query, cache_key, page_table, positions, probe_m_split, probe_l_split, \
        nullptr, nullptr, nullptr, nullptr, mp, tg_stage, tg_scores, tg_mu_k,  \
        tg_sk, tg_sq, tg_delta, tg_q_pos, tg_pool_unused, lane, sg, tg);       \
  }

DEFINE_MINF_PROBE_STATS(omlx_m5_minference_probe_stats_affine4_g128_hd256, true)
DEFINE_MINF_PROBE_STATS(omlx_m5_minference_probe_stats_bf16_hd256, false)
#undef DEFINE_MINF_PROBE_STATS

// ============================================================================
// Stage 1B: Reduce Split Stats -> Exact (M_global, inv_Z) & Zero V_score/D_score
// Grid: {max(w_probe * qh, kvh * max_kv_blocks), 1, 1}, tptg: {256, 1, 1}
// ============================================================================
kernel void omlx_m5_minference_probe_reduce_and_clear(
    device const float *probe_m_split [[buffer(0)]],
    device const float *probe_l_split [[buffer(1)]],
    device float *probe_m_global [[buffer(2)]],
    device float *probe_inv_z [[buffer(3)]],
    device float *v_score [[buffer(4)]],
    device float *d_score [[buffer(5)]],
    constant MInferenceParams &mp [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total_scores = mp.base.key_value_heads * mp.max_kv_blocks;
  if (gid < total_scores) {
    v_score[gid] = 0.0f;
    d_score[gid] = 0.0f;
  }
  const uint total_rows = mp.w_probe * mp.base.query_heads;
  if (gid < total_rows) {
    const ulong base_rec = ulong(gid) * mp.split_count;
    float gm = minf::kNegInf;
    for (uint s = 0u; s < mp.split_count; ++s) {
      const float m = probe_m_split[base_rec + s];
      if (m > minf::kNegInfHalf) {
        gm = max(gm, m);
      }
    }
    float gz = 0.0f;
    if (gm > minf::kNegInfHalf) {
      for (uint s = 0u; s < mp.split_count; ++s) {
        const float m = probe_m_split[base_rec + s];
        if (m > minf::kNegInfHalf) {
          gz += probe_l_split[base_rec + s] * fast::exp2(m - gm);
        }
      }
    }
    probe_m_global[gid] = gm;
    probe_inv_z[gid] = (gz > 0.0f) ? (1.0f / gz) : 0.0f;
  }
}

// ============================================================================
// Stage 1C: Normalized Probe Accumulation with GQA max_{g=0..G-1} Pooling
// Grid: {split_count * 256, ceil(w_probe / bm_tok_p) * kvh, 1}, tptg: {256, 1, 1}
// ============================================================================
#define DEFINE_MINF_PROBE_SCORE(NAME, IS_AFF4)                                 \
  kernel void NAME(                                                            \
      device const bfloat *query [[buffer(0)]],                                \
      device const uchar *cache_key [[buffer(1)]],                             \
      device const uint *page_table [[buffer(2)]],                             \
      device const uint *positions [[buffer(3)]],                              \
      device const float *probe_m_global [[buffer(4)]],                        \
      device const float *probe_inv_z [[buffer(5)]],                           \
      device atomic<float> *v_score [[buffer(6)]],                             \
      device atomic<float> *d_score [[buffer(7)]],                             \
      constant MInferenceParams &mp [[buffer(8)]],                             \
      ushort lane [[thread_index_in_simdgroup]],                               \
      ushort sg [[simdgroup_index_in_threadgroup]],                            \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup uint4 tg_stage[16384u / 16u];                                  \
    threadgroup float tg_scores[2u * 16u * 32u];                               \
    threadgroup float tg_mu_k[2u * 128u];                                      \
    threadgroup float tg_sk[2u * 32u];                                         \
    threadgroup float tg_sq[2u * 64u];                                         \
    threadgroup float tg_delta[2u * 32u];                                      \
    threadgroup uint tg_q_pos[64u];                                            \
    threadgroup float tg_pool_buf[3u * 64u];                                   \
    minf::minference_probe_impl<IS_AFF4, 1u>(                                  \
        query, cache_key, page_table, positions, nullptr, nullptr,             \
        probe_m_global, probe_inv_z, v_score, d_score, mp, tg_stage,           \
        tg_scores, tg_mu_k, tg_sk, tg_sq, tg_delta, tg_q_pos, tg_pool_buf,     \
        lane, sg, tg);                                                         \
  }

DEFINE_MINF_PROBE_SCORE(omlx_m5_minference_probe_score_affine4_g128_hd256, true)
DEFINE_MINF_PROBE_SCORE(omlx_m5_minference_probe_score_bf16_hd256, false)
#undef DEFINE_MINF_PROBE_SCORE

// ============================================================================
// Stage 2A: Exact Per-KVH Top-K_V Vertical & Top-K_S Slash Selection
// Grid: {256, key_value_heads, 1}, tptg: {256, 1, 1}
// ============================================================================
kernel void omlx_m5_minference_select_topk(
    device const float *v_score [[buffer(0)]],
    device const float *d_score [[buffer(1)]],
    device const uint *positions [[buffer(2)]],
    device ushort *top_v_idx [[buffer(3)]],
    device ushort *top_s_lag [[buffer(4)]],
    device uint *top_counts [[buffer(5)]],
    constant MInferenceParams &mp [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint kvh = tg.y;
  if (kvh >= mp.base.key_value_heads) {
    return;
  }

  threadgroup float tg_w_val[8];
  threadgroup uint tg_w_idx[8];
  threadgroup uint tg_nk_blocks[1];
  // 1 KiB threadgroup exclusion bitmask (0 device writes, 100% coherent across SGs!).
  threadgroup uint tg_excl[256];

  // Find max position across all tokens to bound nk_blocks.
  uint local_max_pos = 0u;
  for (uint t = tid; t < mp.base.tokens; t += 256u) {
    local_max_pos = max(local_max_pos, positions[t]);
  }
  uint wm = simd_max(local_max_pos);
  if (lane == 0u) {
    tg_w_idx[sg] = wm;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    uint gmax_pos = 0u;
    for (uint s = 0u; s < 8u; ++s) {
      gmax_pos = max(gmax_pos, tg_w_idx[s]);
    }
    tg_nk_blocks[0] = min((gmax_pos + 32u) / 32u, mp.max_kv_blocks);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint nk_blocks = tg_nk_blocks[0];
  device const float *vs = v_score + ulong(kvh) * mp.max_kv_blocks;
  device const float *ds = d_score + ulong(kvh) * mp.max_kv_blocks;
  device ushort *out_v = top_v_idx + ulong(kvh) * mp.top_stride;
  device ushort *out_s = top_s_lag + ulong(kvh) * mp.top_stride;

  // Mask out mandatory sink blocks [0, n_sink) so Top-K_V is spent on interior blocks.
  {
    const uint bit_base = tid * 32u;
    uint mask = 0u;
    if (bit_base < mp.n_sink) {
      const uint rem = mp.n_sink - bit_base;
      mask = (rem >= 32u) ? 0xffffffffu : ((1u << rem) - 1u);
    }
    tg_excl[tid] = mask;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Select Top-K_V vertical blocks via parallel 256-thread argmax.
  uint actual_v = 0u;
  const uint max_kv = min(mp.k_vertical, mp.top_stride);
  for (uint k = 0u; k < max_kv; ++k) {
    float best_v = -1.0f;
    uint best_i = 0xffffffffu;
    for (uint i = tid; i < nk_blocks; i += 256u) {
      if ((tg_excl[i >> 5u] & (1u << (i & 31u))) == 0u) {
        const float val = vs[i];
        if (val > best_v) {
          best_v = val;
          best_i = i;
        }
      }
    }
    // Warp argmax (break ties by lowest index).
    for (ushort off = 16u; off > 0u; off >>= 1u) {
      const float ov = simd_shuffle_down(best_v, off);
      const uint oi = simd_shuffle_down(best_i, off);
      if (ov > best_v || (ov == best_v && oi < best_i)) {
        best_v = ov;
        best_i = oi;
      }
    }
    if (lane == 0u) {
      tg_w_val[sg] = best_v;
      tg_w_idx[sg] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float gv = -1.0f;
      uint gi = 0xffffffffu;
      for (uint s = 0u; s < 8u; ++s) {
        if (tg_w_val[s] > gv || (tg_w_val[s] == gv && tg_w_idx[s] < gi)) {
          gv = tg_w_val[s];
          gi = tg_w_idx[s];
        }
      }
      if (gv > 0.0f && gi < nk_blocks) {
        out_v[actual_v++] = ushort(gi);
        tg_excl[gi >> 5u] |= (1u << (gi & 31u));
      }
      tg_w_idx[0] = (gv > 0.0f && gi < nk_blocks) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tg_w_idx[0] == 0u) {
      break;
    }
  }

  // Mask out mandatory local lags [0, n_local] so Top-K_S is spent on non-local diagonals.
  {
    const uint local_cut = mp.n_local + 1u;
    const uint bit_base = tid * 32u;
    uint mask = 0u;
    if (bit_base < local_cut) {
      const uint rem = local_cut - bit_base;
      mask = (rem >= 32u) ? 0xffffffffu : ((1u << rem) - 1u);
    }
    tg_excl[tid] = mask;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Select Top-K_S slash lags via parallel 256-thread argmax.
  uint actual_s = 0u;
  const uint max_ks = min(mp.k_slash, mp.top_stride);
  for (uint k = 0u; k < max_ks; ++k) {
    float best_v = -1.0f;
    uint best_i = 0xffffffffu;
    for (uint i = tid; i < nk_blocks; i += 256u) {
      if ((tg_excl[i >> 5u] & (1u << (i & 31u))) == 0u) {
        const float val = ds[i];
        if (val > best_v) {
          best_v = val;
          best_i = i;
        }
      }
    }
    for (ushort off = 16u; off > 0u; off >>= 1u) {
      const float ov = simd_shuffle_down(best_v, off);
      const uint oi = simd_shuffle_down(best_i, off);
      if (ov > best_v || (ov == best_v && oi < best_i)) {
        best_v = ov;
        best_i = oi;
      }
    }
    if (lane == 0u) {
      tg_w_val[sg] = best_v;
      tg_w_idx[sg] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float gv = -1.0f;
      uint gi = 0xffffffffu;
      for (uint s = 0u; s < 8u; ++s) {
        if (tg_w_val[s] > gv || (tg_w_val[s] == gv && tg_w_idx[s] < gi)) {
          gv = tg_w_val[s];
          gi = tg_w_idx[s];
        }
      }
      if (gv > 0.0f && gi < nk_blocks) {
        out_s[actual_s++] = ushort(gi);
        tg_excl[gi >> 5u] |= (1u << (gi & 31u));
      }
      tg_w_idx[0] = (gv > 0.0f && gi < nk_blocks) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tg_w_idx[0] == 0u) {
      break;
    }
  }

  if (tid == 0u) {
    top_counts[kvh * 2u + 0u] = actual_v;
    top_counts[kvh * 2u + 1u] = actual_s;
  }
}

// ============================================================================
// Stage 2B: 1 KiB SRAM Bitmask + ctz Sorted Block-Index Builder
// Grid: {256, n_q_blocks * key_value_heads, 1}, tptg: {256, 1, 1}
// ============================================================================
kernel void omlx_m5_minference_build_indices(
    device const uint *positions [[buffer(0)]],
    device const ushort *top_v_idx [[buffer(1)]],
    device const ushort *top_s_lag [[buffer(2)]],
    device const uint *top_counts [[buffer(3)]],
    device ushort *active_indices [[buffer(4)]],
    device uint *active_counts [[buffer(5)]],
    constant MInferenceParams &mp [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  // Exact 1,024 B (1 KiB) SRAM bitmask covering up to 8,192 KV blocks (256K tok)!
  threadgroup atomic_uint bitmask[256];
  threadgroup uint warp_totals[8];
  threadgroup uint tg_minmax[2];

  const uint kvh = tg.y % mp.base.key_value_heads;
  const uint q_blk = tg.y / mp.base.key_value_heads;
  if (q_blk >= mp.n_q_blocks || kvh >= mp.base.key_value_heads) {
    return;
  }

  atomic_store_explicit(&bitmask[tid], 0u, memory_order_relaxed);

  const uint q_base = q_blk * mp.bm_tok;
  const uint active_tok = min(mp.bm_tok, mp.base.tokens - q_base);

  if (tid == 0u) {
    uint lo = 0xffffffffu;
    uint hi = 0u;
    for (uint r = 0u; r < active_tok; ++r) {
      const uint pos = positions[q_base + r];
      lo = min(lo, pos);
      hi = max(hi, pos);
    }
    tg_minmax[0] = lo;
    tg_minmax[1] = hi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint min_q_pos = tg_minmax[0];
  const uint max_q_pos = tg_minmax[1];
  const uint nk_blocks_blk = min((max_q_pos + 32u) / 32u, mp.max_kv_blocks);
  const uint min_kb = min_q_pos >> 5u;

  // 1. Mandatory attention sink blocks: [0, N_sink)
  const uint sink_end = min(mp.n_sink, nk_blocks_blk);
  for (uint b = tid; b < sink_end; b += 256u) {
    atomic_fetch_or_explicit(&bitmask[b >> 5u], 1u << (b & 31u),
                             memory_order_relaxed);
  }

  // 2. Mandatory local sliding-window diagonal: [max(0, min_kb - N_local), nk_blocks_blk)
  const uint local_start = (min_kb > mp.n_local) ? (min_kb - mp.n_local) : 0u;
  for (uint b = local_start + tid; b < nk_blocks_blk; b += 256u) {
    atomic_fetch_or_explicit(&bitmask[b >> 5u], 1u << (b & 31u),
                             memory_order_relaxed);
  }

  // 3. Global Top-K_V vertical blocks
  const uint actual_v = top_counts[kvh * 2u + 0u];
  device const ushort *v_list = top_v_idx + ulong(kvh) * mp.top_stride;
  for (uint i = tid; i < actual_v; i += 256u) {
    const uint b = uint(v_list[i]);
    if (b < nk_blocks_blk) {
      atomic_fetch_or_explicit(&bitmask[b >> 5u], 1u << (b & 31u),
                               memory_order_relaxed);
    }
  }

  // 4. Per-q_blk Top-K_S slash diagonal blocks (straddle-safe across [min_q_pos, max_q_pos])
  const uint actual_s = top_counts[kvh * 2u + 1u];
  device const ushort *s_list = top_s_lag + ulong(kvh) * mp.top_stride;
  for (uint i = tid; i < actual_s; i += 256u) {
    const uint lag = uint(s_list[i]);
    const uint lag_lo_off = lag * 32u;
    if (max_q_pos >= lag_lo_off) {
      const uint k_hi = max_q_pos - lag_lo_off;
      const uint lag_hi_off = (lag + 1u) * 32u - 1u;
      const uint k_lo = (min_q_pos > lag_hi_off) ? (min_q_pos - lag_hi_off) : 0u;
      const uint kb_lo = k_lo >> 5u;
      const uint kb_hi = min(nk_blocks_blk - 1u, k_hi >> 5u);
      const uint kb_lim = min(kb_hi, kb_lo + 63u);
      for (uint b = kb_lo; b <= kb_lim; ++b) {
        atomic_fetch_or_explicit(&bitmask[b >> 5u], 1u << (b & 31u),
                                 memory_order_relaxed);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 5. Parallel popcount prefix sum + ctz monotonic sorted emit!
  uint w = atomic_load_explicit(&bitmask[tid], memory_order_relaxed);
  const uint cnt = popcount(w);
  const uint lane_excl = simd_prefix_exclusive_sum(cnt);
  if (lane == 31u) {
    warp_totals[sg] = lane_excl + cnt;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint warp_base = 0u;
  for (uint s = 0u; s < sg; ++s) {
    warp_base += warp_totals[s];
  }
  uint out_off = warp_base + lane_excl;

  const ulong slot = ulong(kvh) * mp.max_q_blocks + q_blk;
  device ushort *out_idx = active_indices + slot * mp.active_stride;
  while (w != 0u) {
    const uint bit = ctz(w);
    w &= (w - 1u);
    if (out_off < mp.active_stride) {
      out_idx[out_off] = ushort(tid * 32u + bit);
    }
    ++out_off;
  }

  if (tid == 255u) {
    active_counts[slot] = min(out_off, mp.active_stride);
  }
}
