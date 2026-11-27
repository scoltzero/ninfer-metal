// Phase 2 Work Package 2.2: MInference Vertical-Slash Block-Sparse Index Builder
// (Metal 4.1 / MacOSX27.2 SDK — Zero-Spill WN=4 High-Occupancy Architecture)
//   - Unified INT8-NAX probe kernels (probe_stats_i8 & probe_accum_i8) reading
//     pre-smoothed (K_int8[kvh][kb][4][32][64], mu_K[kvh][4][pad_kb][64],
//     s_K[kvh][kb][4][32], valid_kv) directly from sage2_kv_ws (0 redundant K unpacks!).
//   - WN=4 (WM=2, 256 thr, M_TILE=32): ct_q = 8 GPRs, 2-slot float4 reduction,
//     amortized 16-block native Metal 4.1 NAX Delta (matmul2d 16x16x64),
//     Apple10 in-register Frag16x16 fast path + portable Apple9 fallback.
//   - Strict <= 16.0 KiB (16,384 B) SRAM cap & 0 B AGX spills across ALL 5 kernels!
//     (probe_stats_i8: 10,880 B / 10.625 KiB -> 3 TG/core; probe_accum_i8: 11,264 B / 11.0 KiB).
#include <metal_atomic>
#include <metal_simdgroup>
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define MINF_UNROLL _Pragma("clang loop unroll(full)")
#define MINF_NOUNROLL _Pragma("clang loop unroll(disable)")
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
  uint ws_max_kv_blocks;
  uint ws_pad_kv_blocks;
  uint reserved0;
  uint reserved1;
};

namespace minf {

constant constexpr float kNegInf = -3.402823466e+38f;
constant constexpr float kNegInfHalf = -1.701411733e+38f;
constant constexpr float kLog2e = 1.4426950408889634f;

// ============================================================================
// Threadgroup SRAM Layouts & Compile-Time <= 16.0 KiB (16,384 B) Enforcement
// ============================================================================
template <uint PassMode>
struct MinfProbeTgStorage;

template <>
struct MinfProbeTgStorage<0u> {
  alignas(16) float s_xchg[2048u];  // 8,192 B (also 8x16x64 tQ_sg & 2x3x256 ctD)
  alignas(16) float del_tile[512u]; // 2,048 B (32 rows x 16 blocks TOTAL Delta)
  alignas(16) float sq[128u];       //   512 B (4 d_parts x 32 rows)
  alignas(16) uint q_pos[32u];      //   128 B
};

template <>
struct MinfProbeTgStorage<1u> {
  alignas(16) float s_xchg[2048u];  // 8,192 B
  alignas(16) float del_tile[512u]; // 2,048 B
  alignas(16) float sq[128u];       //   512 B
  alignas(16) uint q_pos[32u];      //   128 B
  alignas(16) float pool_buf[96u];  //   384 B (v_row[32], dhi[32], dlo[32])
};

struct MinfSelectTopkTg {
  float w_val[8];
  uint w_idx[8];
  uint nk_blocks[1];
  uint excl[256];
};

struct MinfBuildIndicesTg {
  atomic_uint bitmask[256];
  uint warp_totals[8];
  uint minmax[2];
  uint q_pos[128];
};

static_assert(sizeof(MinfProbeTgStorage<0u>) == 10880u &&
                  sizeof(MinfProbeTgStorage<0u>) <= 16384u,
              "WP2.2: MInference probe_stats_i8 SRAM must be <= 16 KiB");
static_assert(sizeof(MinfProbeTgStorage<1u>) == 11264u &&
                  sizeof(MinfProbeTgStorage<1u>) <= 16384u,
              "WP2.2: MInference probe_accum_i8 SRAM must be <= 16 KiB");
static_assert(sizeof(MinfSelectTopkTg) <= 16384u,
              "WP2.2: MInference select_topk SRAM must be <= 16 KiB");
static_assert(sizeof(MinfBuildIndicesTg) <= 16384u,
              "WP2.2: MInference build_indices SRAM must be <= 16 KiB");

MINF_ALWAYS_INLINE
static inline bfloat4 clean_bf4(bfloat4 v) {
  const ushort4 bits = as_type<ushort4>(v);
  return select(v, bfloat4(bfloat(0.0f)),
                (bits & ushort4(0x7fffu)) > ushort4(0x7f80u));
}

MINF_ALWAYS_INLINE
static inline void fast_divmod_g(uint r, uint g, thread uint &q,
                                 thread uint &rem) {
  if (g == 6u) {
    q = mulhi(r, 0x2AAAAAABu);
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
static inline short2 minf_frag_coord(ushort lane) {
  const ushort qid = lane >> 2u;
  const ushort fm = (qid & 4u) | ((lane >> 1u) & 3u);
  const ushort fn = ((qid & 2u) | (lane & 1u)) << 2u;
  return short2(short(fn), short(fm));
}

// Compile-time const-foldable Apple10 (M5) NAX layout predicate (identical idiom
// to gdn_chunk_nax.metal / attention_sage2.metal) so AGX LLVM DCEs the Apple9
// fallback on Apple10 and guarantees 0 B spills!
MINF_ALWAYS_INLINE
static inline bool minf_has_apple10_nax_layout() {
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  const auto c0 = ct_a.get_multidimensional_index(ushort(0u));
  const auto c2 = ct_a.get_multidimensional_index(ushort(2u));
  return (int(c2[0]) - int(c0[0]) == 2) && (int(c2[1]) == int(c0[1]));
}

// Unified INT8-NAX Probe Engine (WM=2, WN=4 -> 256 threads, <= 11.0 KiB, 0 Spills!)
template <uint PassMode>
MINF_ALWAYS_INLINE
static inline void minference_probe_i8_impl(
    device const bfloat *query, device const int8_t *k_int8,
    device const float *mu_k, device const float *s_k,
    device const uint *valid_kv_buf, device const uint *positions,
    device float *probe_m_split, device float *probe_l_split,
    device const float *probe_m_global, device const float *probe_inv_z,
    device atomic<float> *v_score, device atomic<float> *d_score,
    constant MInferenceParams &mp,
    threadgroup MinfProbeTgStorage<PassMode> &tg_mem, ushort lane, ushort sg,
    uint3 tg) {
  constexpr uint BK = 32u;
  constexpr uint M_TILE = 32u;

  const PagedAttentionParams p = mp.base;
  if (p.tokens == 0u || mp.w_probe == 0u || p.query_heads == 0u ||
      p.key_value_heads == 0u) {
    return;
  }
  const uint live_w_probe = min(mp.w_probe, p.tokens);
  const uint live_q_probe_start = p.tokens - live_w_probe;
  const uint repeat = p.query_heads / p.key_value_heads;
  if (repeat == 0u || repeat > M_TILE) {
    return;
  }
  const uint bm_tok_p = M_TILE / repeat;
  const uint n_probe_qblks = (live_w_probe + bm_tok_p - 1u) / bm_tok_p;

  const uint split = tg.x;
  const uint p_blk = tg.y / p.key_value_heads;
  const uint kvh = tg.y - p_blk * p.key_value_heads;
  if (split >= mp.split_count || p_blk >= n_probe_qblks) {
    return;
  }

  const uint w_base = p_blk * bm_tok_p;
  const uint q_base = live_q_probe_start + w_base;
  const uint active_tok = min(bm_tok_p, live_w_probe - w_base);
  const uint active_m_rows = active_tok * repeat;
  const uint flat_tid = uint(sg) * 32u + uint(lane);

  if (flat_tid < active_tok) {
    tg_mem.q_pos[flat_tid] = positions[q_base + flat_tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint max_valid_kv = 0u;
  for (uint r = 0u; r < active_tok; ++r) {
    max_valid_kv = max(max_valid_kv, tg_mem.q_pos[r] + 1u);
  }
  const uint global_valid_kv = valid_kv_buf[0];
  max_valid_kv = min(max_valid_kv, global_valid_kv);
  const uint nk_blocks = min(
      min((max_valid_kv + BK - 1u) / BK, mp.max_kv_blocks),
      mp.ws_max_kv_blocks);
  const uint kb_start = split * mp.split_blocks;
  const uint kb_end = min(nk_blocks, kb_start + mp.split_blocks);

  const uint rg = uint(sg) >> 2u; // 0..1
  const uint dp = uint(sg) & 3u;  // 0..3 (D_SLICE=64)
  const uint tm = rg * 16u;
  const uint d_offset = dp * 64u;
  const uint lane_u = uint(lane);

  const short2 sc = minf_frag_coord(lane);
  const short sm = sc.y;
  const short sn = sc.x;
  const uint sn_idx = uint(sn) >> 2u;

  // Empty split fast exit!
  if (kb_start >= kb_end) {
    if constexpr (PassMode == 0u) {
      if (dp == 0u && (lane_u & 1u) == 0u) {
        const uint rm = tm + (lane_u >> 1u);
        if (rm < active_m_rows) {
          uint tok_l = 0u, g_l = 0u;
          fast_divmod_g(rm, repeat, tok_l, g_l);
          const uint qh_l = kvh * repeat + g_l;
          const ulong rec =
              (ulong(w_base + tok_l) * p.query_heads + qh_l) * mp.split_count +
              split;
          probe_m_split[rec] = kNegInf;
          probe_l_split[rec] = 0.0f;
        }
      }
    }
    return;
  }

  const float scale2 = 0.0625f * kLog2e;

  // Prologue: Quantize probe Q[16, 64] -> stage all 8 SGs in s_xchg[8,192 B] -> ct_q (8 GPRs!)
  constexpr auto qk_desc = matmul2d_descriptor(
      16, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto mu_desc = matmul2d_descriptor(
      16, 16, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<qk_desc, execution_simdgroup> qk_op;
  matmul2d<mu_desc, execution_simdgroup> mu_op;

  auto ct_q =
      qk_op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();

  {
    const uint q_r_in16 = lane_u >> 1u;
    const uint q_half = lane_u & 1u;
    const uint rm = tm + q_r_in16;
    const bool qv = (rm < active_m_rows);
    uint tok_l = 0u, g_l = 0u;
    if (qv) fast_divmod_g(rm, repeat, tok_l, g_l);
    const uint qh_l = kvh * repeat + g_l;
    device const bfloat4 *q_ptr =
        qv ? reinterpret_cast<device const bfloat4 *>(
                 query + ulong(q_base + tok_l) * p.query_token_stride +
                 ulong(qh_l) * p.query_head_stride + d_offset + q_half * 32u)
           : nullptr;

    float lmax = 0.0f;
    if (qv) {
      MINF_UNROLL
      for (uint k = 0u; k < 8u; ++k) {
        const float4 av = fabs(float4(clean_bf4(q_ptr[k])));
        lmax = max(lmax, max(max(av.x, av.y), max(av.z, av.w)));
      }
    }
    const float q_absmax = max(lmax, simd_shuffle_xor(lmax, ushort(1)));
    const float sq_scaled =
        (q_absmax > 0.0f) ? (q_absmax * (scale2 / 127.0f)) : 0.0f;
    const float inv_sq = (q_absmax > 0.0f) ? (127.0f / q_absmax) : 0.0f;
    if (q_half == 0u) {
      tg_mem.sq[dp * M_TILE + rm] = sq_scaled;
    }

    threadgroup uchar *q_base_tg =
        reinterpret_cast<threadgroup uchar *>(tg_mem.s_xchg) + uint(sg) * 1024u;
    threadgroup char4 *q_dst = reinterpret_cast<threadgroup char4 *>(
        q_base_tg + q_r_in16 * 64u + q_half * 32u);
    if (qv) {
      MINF_UNROLL
      for (uint k = 0u; k < 8u; ++k) {
        const float4 vf = float4(clean_bf4(q_ptr[k]));
        const float4 qf =
            clamp(rint(vf * inv_sq), float4(-127.0f), float4(127.0f));
        q_dst[k] = char4(char(int(qf.x)), char(int(qf.y)), char(int(qf.z)),
                         char(int(qf.w)));
      }
    } else {
      MINF_UNROLL
      for (uint k = 0u; k < 8u; ++k) {
        q_dst[k] = char4(0);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ_tg(
        reinterpret_cast<threadgroup int8_t *>(q_base_tg),
        dextents<int32_t, 2>(64, 16));
    ct_q.load(tQ_tg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  device const int8_t *k_int8_head =
      k_int8 + ulong(kvh) * mp.ws_max_kv_blocks * (4ul * 32ul * 64ul);
  device const float *s_k_head =
      s_k + ulong(kvh) * mp.ws_max_kv_blocks * (4ul * 32ul);

  // Portable Apple9 fallback (early return -> completely DCE'd on Apple10!)
  if (!minf_has_apple10_nax_layout()) {
    const uint q_r_lin = lane_u >> 1u;
    const uint q_h_lin = lane_u & 1u;
    const uint rm_lin = tm + q_r_lin;
    const bool qv_lin = (rm_lin < active_m_rows);
    uint tok_lin = 0u, g_lin = 0u;
    if (qv_lin) fast_divmod_g(rm_lin, repeat, tok_lin, g_lin);
    const uint qh_lin = kvh * repeat + g_lin;
    const uint qpos_lin = qv_lin ? tg_mem.q_pos[tok_lin] : 0u;

    float m0 = kNegInf;
    float l0 = 0.0f;
    float gm_lin = kNegInf, ginv_lin = 0.0f;
    if constexpr (PassMode == 1u) {
      if (dp == 0u && qv_lin) {
        const ulong r_idx = ulong(w_base + tok_lin) * p.query_heads + qh_lin;
        gm_lin = probe_m_global[r_idx];
        ginv_lin = probe_inv_z[r_idx];
      }
    }

    for (uint kb = kb_start; kb < kb_end; ++kb) {
      if ((kb & 15u) == 0u || kb == kb_start) {
        const uint kb_tile = kb & ~15u;
        threadgroup int8_t *my_q_tg =
            reinterpret_cast<threadgroup int8_t *>(tg_mem.s_xchg) +
            uint(sg) * 1024u;
        tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ_sg(
            my_q_tg, dextents<int32_t, 2>(64, 16));
        ct_q.store(tQ_sg);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        device float *mu_ptr = const_cast<device float *>(
            mu_k + ((ulong(kvh) * 4ul + dp) * mp.ws_pad_kv_blocks + kb_tile) *
                       64ul);
        tensor<device float, dextents<int32_t, 2>, tensor_inline> tMu(
            mu_ptr, dextents<int32_t, 2>(64, 16));
        auto ctD = mu_op.template get_destination_cooperative_tensor<
            decltype(tQ_sg), decltype(tMu), float>();
        mu_op.run(tQ_sg, tMu, ctD);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        MINF_NOUNROLL
        for (uint dp_g = 0u; dp_g < 4u; ++dp_g) {
          if (dp == dp_g) {
            MINF_UNROLL
            for (ushort i = 0; i < ctD.get_capacity(); ++i) {
              if (ctD.is_valid_element(i)) {
                const auto idx = ctD.get_multidimensional_index(i);
                const uint col = uint(idx[0]);
                const uint row = uint(idx[1]);
                const float val = ctD[i] * tg_mem.sq[dp * M_TILE + tm + row];
                const uint off = (tm + row) * 16u + col;
                if (dp_g == 0u) tg_mem.del_tile[off] = val;
                else tg_mem.del_tile[off] += val;
              }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }

      const uint kb_col_base = kb * BK;
      const uint b16 = kb & 15u;
      tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK_dev(
          const_cast<device int8_t *>(k_int8_head +
                                      (ulong(kb) * 4ul + dp) * 2048ul),
          dextents<int32_t, 2>(64, 32));
      auto ct_s = qk_op.template get_destination_cooperative_tensor<
          remove_addrspace_t<decltype(ct_q)>, decltype(tK_dev), int32_t>();
      qk_op.run(ct_q, tK_dev, ct_s);
      device const float *sk_ptr = s_k_head + (ulong(kb) * 4ul + dp) * 32ul;

      MINF_NOUNROLL
      for (uint dp_g = 0u; dp_g < 4u; ++dp_g) {
        if (dp == dp_g) {
          MINF_UNROLL
          for (ushort i = 0; i < ct_s.get_capacity(); ++i) {
            if (ct_s.is_valid_element(i)) {
              const auto idx = ct_s.get_multidimensional_index(i);
              const uint col = uint(idx[0]);
              const uint row = uint(idx[1]);
              const float v = float(ct_s[i]) *
                              tg_mem.sq[dp * M_TILE + tm + row] * sk_ptr[col];
              const uint off = rg * 512u + row * 32u + col;
              if (dp_g == 0u)
                tg_mem.s_xchg[off] =
                    v + tg_mem.del_tile[(tm + row) * 16u + b16];
              else
                tg_mem.s_xchg[off] += v;
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      if (dp == 0u) {
        const threadgroup float4 *s_row4 =
            reinterpret_cast<const threadgroup float4 *>(
                tg_mem.s_xchg + rg * 512u + q_r_lin * 32u + q_h_lin * 16u);
        const uint c_base = kb_col_base + q_h_lin * 16u;
        const uint4 c0 = uint4(c_base) + uint4(0u, 1u, 2u, 3u);
        const float4 s0 = select(float4(kNegInf), s_row4[0],
                                 qv_lin && (c0 < max_valid_kv) && (c0 <= qpos_lin));
        const float4 s1 = select(float4(kNegInf), s_row4[1],
                                 qv_lin && ((c0 + 4u) < max_valid_kv) &&
                                     ((c0 + 4u) <= qpos_lin));
        const float4 s2 = select(float4(kNegInf), s_row4[2],
                                 qv_lin && ((c0 + 8u) < max_valid_kv) &&
                                     ((c0 + 8u) <= qpos_lin));
        const float4 s3 = select(float4(kNegInf), s_row4[3],
                                 qv_lin && ((c0 + 12u) < max_valid_kv) &&
                                     ((c0 + 12u) <= qpos_lin));

        if constexpr (PassMode == 0u) {
          const float4 m4 = max(max(s0, s1), max(s2, s3));
          float m_loc = max(max(m4.x, m4.y), max(m4.z, m4.w));
          m_loc = max(m_loc, simd_shuffle_xor(m_loc, ushort(1)));
          const float nm = max(m0, m_loc);
          const float fac = (m0 <= kNegInfHalf) ? 0.0f : fast::exp2(m0 - nm);
          m0 = nm;
          const bool emp = (nm <= kNegInfHalf);
          const float4 p0 =
              select(fast::exp2(s0 - nm), float4(0.0f), emp || (s0 <= kNegInfHalf));
          const float4 p1 =
              select(fast::exp2(s1 - nm), float4(0.0f), emp || (s1 <= kNegInfHalf));
          const float4 p2 =
              select(fast::exp2(s2 - nm), float4(0.0f), emp || (s2 <= kNegInfHalf));
          const float4 p3 =
              select(fast::exp2(s3 - nm), float4(0.0f), emp || (s3 <= kNegInfHalf));
          const float4 ps = (p0 + p1) + (p2 + p3);
          float tsum = (ps.x + ps.y) + (ps.z + ps.w);
          tsum += simd_shuffle_xor(tsum, ushort(1));
          l0 = fma(l0, fac, tsum);
        } else {
          const bool emp = (gm_lin <= kNegInfHalf) || !(ginv_lin > 0.0f);
          const float4 p0 = select(fast::exp2(s0 - gm_lin) * ginv_lin,
                                   float4(0.0f), emp || (s0 <= kNegInfHalf));
          const float4 p1 = select(fast::exp2(s1 - gm_lin) * ginv_lin,
                                   float4(0.0f), emp || (s1 <= kNegInfHalf));
          const float4 p2 = select(fast::exp2(s2 - gm_lin) * ginv_lin,
                                   float4(0.0f), emp || (s2 <= kNegInfHalf));
          const float4 p3 = select(fast::exp2(s3 - gm_lin) * ginv_lin,
                                   float4(0.0f), emp || (s3 <= kNegInfHalf));
          const uint split_c = (qpos_lin >= kb_col_base)
                                   ? (((qpos_lin - kb_col_base) & 31u) + 1u)
                                   : 0u;
          const uint4 j0 = uint4(q_h_lin * 16u) + uint4(0u, 1u, 2u, 3u);
          const float4 hi_s = select(float4(0.0f), p0, j0 < split_c) +
                              select(float4(0.0f), p1, (j0 + 4u) < split_c) +
                              select(float4(0.0f), p2, (j0 + 8u) < split_c) +
                              select(float4(0.0f), p3, (j0 + 12u) < split_c);
          const float4 lo_s = select(float4(0.0f), p0, j0 >= split_c) +
                              select(float4(0.0f), p1, (j0 + 4u) >= split_c) +
                              select(float4(0.0f), p2, (j0 + 8u) >= split_c) +
                              select(float4(0.0f), p3, (j0 + 12u) >= split_c);
          float dhi = (hi_s.x + hi_s.y) + (hi_s.z + hi_s.w);
          float dlo = (lo_s.x + lo_s.y) + (lo_s.z + lo_s.w);
          dhi += simd_shuffle_xor(dhi, ushort(1));
          dlo += simd_shuffle_xor(dlo, ushort(1));
          if (q_h_lin == 0u) {
            tg_mem.pool_buf[rm_lin] = dhi + dlo;
            tg_mem.pool_buf[32u + rm_lin] = dhi;
            tg_mem.pool_buf[64u + rm_lin] = dlo;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (PassMode == 1u) {
        if (flat_tid < active_tok) {
          const uint pos_w = tg_mem.q_pos[flat_tid];
          if (pos_w >= kb_col_base) {
            float v_max = 0.0f;
            float dhi_max = 0.0f;
            float dlo_max = 0.0f;
            const uint r_base = flat_tid * repeat;
            for (uint g = 0u; g < repeat; ++g) {
              v_max = max(v_max, tg_mem.pool_buf[r_base + g]);
              dhi_max = max(dhi_max, tg_mem.pool_buf[32u + r_base + g]);
              dlo_max = max(dlo_max, tg_mem.pool_buf[64u + r_base + g]);
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
      }
    }

    if constexpr (PassMode == 0u) {
      if (dp == 0u && q_h_lin == 0u && qv_lin) {
        const ulong rec =
            (ulong(w_base + tok_lin) * p.query_heads + qh_lin) * mp.split_count +
            split;
        probe_m_split[rec] = m0;
        probe_l_split[rec] = l0;
      }
    }
    return;
  }

  // Apple10 Zero-Spill Fast Path (0 Apple9 locals, 0 get_multidimensional_index!)
  const float sq0 = tg_mem.sq[dp * M_TILE + tm + uint(sm)];
  const float sq1 = tg_mem.sq[dp * M_TILE + tm + 8u + uint(sm)];

  const uint rm0 = tm + uint(sm);
  const uint rm1 = tm + 8u + uint(sm);
  const bool qv0 = (rm0 < active_m_rows);
  const bool qv1 = (rm1 < active_m_rows);
  uint tok0 = 0u, g0 = 0u, tok1 = 0u, g1 = 0u;
  if (qv0) fast_divmod_g(rm0, repeat, tok0, g0);
  if (qv1) fast_divmod_g(rm1, repeat, tok1, g1);
  const uint qh0 = kvh * repeat + g0;
  const uint qh1 = kvh * repeat + g1;
  const uint qpos0 = qv0 ? tg_mem.q_pos[tok0] : 0u;
  const uint qpos1 = qv1 ? tg_mem.q_pos[tok1] : 0u;

  float m0 = kNegInf, m1 = kNegInf;
  float l0 = 0.0f, l1 = 0.0f;
  float gm0 = kNegInf, ginv0 = 0.0f, gm1 = kNegInf, ginv1 = 0.0f;
  if constexpr (PassMode == 1u) {
    if (dp == 0u) {
      if (qv0) {
        const ulong r0_idx = ulong(w_base + tok0) * p.query_heads + qh0;
        gm0 = probe_m_global[r0_idx];
        ginv0 = probe_inv_z[r0_idx];
      }
      if (qv1) {
        const ulong r1_idx = ulong(w_base + tok1) * p.query_heads + qh1;
        gm1 = probe_m_global[r1_idx];
        ginv1 = probe_inv_z[r1_idx];
      }
    }
  }

  for (uint kb = kb_start; kb < kb_end; ++kb) {
    // Amortized 16-block NAX Delta via native matmul2d(16, 16, 64) into del_tile!
    if ((kb & 15u) == 0u || kb == kb_start) {
      const uint kb_tile = kb & ~15u;
      threadgroup int8_t *my_q_tg =
          reinterpret_cast<threadgroup int8_t *>(tg_mem.s_xchg) +
          uint(sg) * 1024u;
      tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ_sg(
          my_q_tg, dextents<int32_t, 2>(64, 16));
      ct_q.store(tQ_sg);
      simdgroup_barrier(mem_flags::mem_threadgroup);

      device float *mu_ptr = const_cast<device float *>(
          mu_k + ((ulong(kvh) * 4ul + dp) * mp.ws_pad_kv_blocks + kb_tile) *
                     64ul);
      tensor<device float, dextents<int32_t, 2>, tensor_inline> tMu(
          mu_ptr, dextents<int32_t, 2>(64, 16));
      auto ctD = mu_op.template get_destination_cooperative_tensor<
          decltype(tQ_sg), decltype(tMu), float>();
      mu_op.run(tQ_sg, tMu, ctD);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (dp > 0u) {
        threadgroup float4 *dst = reinterpret_cast<threadgroup float4 *>(
            tg_mem.s_xchg + (rg * 3u + (dp - 1u)) * 256u + lane_u * 8u);
        dst[0] = float4(ctD[0], ctD[1], ctD[2], ctD[3]) * sq0;
        dst[1] = float4(ctD[4], ctD[5], ctD[6], ctD[7]) * sq1;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (dp == 0u) {
        float4 d0 = float4(ctD[0], ctD[1], ctD[2], ctD[3]) * sq0;
        float4 d1 = float4(ctD[4], ctD[5], ctD[6], ctD[7]) * sq1;
        MINF_NOUNROLL
        for (uint k = 0u; k < 3u; ++k) {
          const threadgroup float4 *src =
              reinterpret_cast<const threadgroup float4 *>(
                  tg_mem.s_xchg + (rg * 3u + k) * 256u + lane_u * 8u);
          d0 += src[0];
          d1 += src[1];
        }
        threadgroup float *dr0 = tg_mem.del_tile + (tm + uint(sm)) * 16u;
        threadgroup float *dr1 = tg_mem.del_tile + (tm + uint(sm + 8)) * 16u;
        *reinterpret_cast<threadgroup float2 *>(dr0 + sn_idx * 2u) = d0.xy;
        *reinterpret_cast<threadgroup float2 *>(dr0 + 8u + sn_idx * 2u) = d0.zw;
        *reinterpret_cast<threadgroup float2 *>(dr1 + sn_idx * 2u) = d1.xy;
        *reinterpret_cast<threadgroup float2 *>(dr1 + 8u + sn_idx * 2u) = d1.zw;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint kb_col_base = kb * BK;
    const uint b16 = kb & 15u;

    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK_dev(
        const_cast<device int8_t *>(k_int8_head +
                                    (ulong(kb) * 4ul + dp) * 2048ul),
        dextents<int32_t, 2>(64, 32));
    auto ct_s = qk_op.template get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(ct_q)>, decltype(tK_dev), int32_t>();
    qk_op.run(ct_q, tK_dev, ct_s);

    device const float *sk_ptr = s_k_head + (ulong(kb) * 4ul + dp) * 32ul;
    const float4 sk0 =
        *reinterpret_cast<device const float4 *>(sk_ptr + uint(sn));
    const float4 sk1 =
        *reinterpret_cast<device const float4 *>(sk_ptr + 16u + uint(sn));
    if (dp == 1u || dp == 3u) {
      threadgroup float4 *dst = reinterpret_cast<threadgroup float4 *>(
          tg_mem.s_xchg + (rg * 2u + (dp >> 1u)) * 512u + lane_u * 16u);
      dst[0] = float4(ct_s[0], ct_s[1], ct_s[2], ct_s[3]) * (sq0 * sk0);
      dst[1] = float4(ct_s[4], ct_s[5], ct_s[6], ct_s[7]) * (sq1 * sk0);
      dst[2] = float4(ct_s[8], ct_s[9], ct_s[10], ct_s[11]) * (sq0 * sk1);
      dst[3] = float4(ct_s[12], ct_s[13], ct_s[14], ct_s[15]) * (sq1 * sk1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dp == 2u) {
      threadgroup float4 *p1 = reinterpret_cast<threadgroup float4 *>(
          tg_mem.s_xchg + (rg * 2u + 1u) * 512u + lane_u * 16u);
      p1[0] += float4(ct_s[0], ct_s[1], ct_s[2], ct_s[3]) * (sq0 * sk0);
      p1[1] += float4(ct_s[4], ct_s[5], ct_s[6], ct_s[7]) * (sq1 * sk0);
      p1[2] += float4(ct_s[8], ct_s[9], ct_s[10], ct_s[11]) * (sq0 * sk1);
      p1[3] += float4(ct_s[12], ct_s[13], ct_s[14], ct_s[15]) * (sq1 * sk1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (dp == 0u) {
      const threadgroup float4 *p0 =
          reinterpret_cast<const threadgroup float4 *>(
              tg_mem.s_xchg + (rg * 2u + 0u) * 512u + lane_u * 16u);
      const threadgroup float4 *p1 =
          reinterpret_cast<const threadgroup float4 *>(
              tg_mem.s_xchg + (rg * 2u + 1u) * 512u + lane_u * 16u);
      const uint c_lo = kb_col_base + uint(sn);
      const uint4 c0v = uint4(c_lo) + uint4(0u, 1u, 2u, 3u);
      const uint4 c1v = c0v + uint4(16u);

      MINF_UNROLL
      for (uint r_sub = 0u; r_sub < 2u; ++r_sub) {
        const uint r_loc = (r_sub == 0u) ? uint(sm) : uint(sm + 8);
        const bool rv = (r_sub == 0u) ? qv0 : qv1;
        const uint rp = (r_sub == 0u) ? qpos0 : qpos1;
        const float dv = tg_mem.del_tile[(tm + r_loc) * 16u + b16];
        float4 a, b;
        if (r_sub == 0u) {
          a = fma(float4(ct_s[0], ct_s[1], ct_s[2], ct_s[3]), sq0 * sk0, dv) +
              p0[0] + p1[0];
          b = fma(float4(ct_s[8], ct_s[9], ct_s[10], ct_s[11]), sq0 * sk1, dv) +
              p0[2] + p1[2];
        } else {
          a = fma(float4(ct_s[4], ct_s[5], ct_s[6], ct_s[7]), sq1 * sk0, dv) +
              p0[1] + p1[1];
          b = fma(float4(ct_s[12], ct_s[13], ct_s[14], ct_s[15]), sq1 * sk1,
                  dv) +
              p0[3] + p1[3];
        }
        a = select(float4(kNegInf), a,
                   rv && (c0v < max_valid_kv) && (c0v <= rp));
        b = select(float4(kNegInf), b,
                   rv && (c1v < max_valid_kv) && (c1v <= rp));

        if constexpr (PassMode == 0u) {
          const float4 mx = max(a, b);
          float rmax = max(max(mx.x, mx.y), max(mx.z, mx.w));
          rmax = max(rmax, simd_shuffle_xor(rmax, ushort(1)));
          rmax = max(rmax, simd_shuffle_xor(rmax, ushort(8)));
          const float old_m = (r_sub == 0u) ? m0 : m1;
          const float nm = max(old_m, rmax);
          const float fac =
              (old_m <= kNegInfHalf) ? 0.0f : fast::exp2(old_m - nm);
          if (r_sub == 0u) m0 = nm;
          else m1 = nm;
          const bool emp = (nm <= kNegInfHalf);
          a = select(fast::exp2(a - nm), float4(0.0f),
                     emp || (a <= kNegInfHalf));
          b = select(fast::exp2(b - nm), float4(0.0f),
                     emp || (b <= kNegInfHalf));
          const float4 ps = a + b;
          float ts = (ps.x + ps.y) + (ps.z + ps.w);
          ts += simd_shuffle_xor(ts, ushort(1));
          ts += simd_shuffle_xor(ts, ushort(8));
          if (r_sub == 0u) l0 = fma(l0, fac, ts);
          else l1 = fma(l1, fac, ts);
        } else {
          const float gm = (r_sub == 0u) ? gm0 : gm1;
          const float ginv = (r_sub == 0u) ? ginv0 : ginv1;
          const bool emp = (gm <= kNegInfHalf) || !(ginv > 0.0f);
          a = select(fast::exp2(a - gm) * ginv, float4(0.0f),
                     emp || (a <= kNegInfHalf));
          b = select(fast::exp2(b - gm) * ginv, float4(0.0f),
                     emp || (b <= kNegInfHalf));

          const uint split_c = (rp >= kb_col_base)
                                   ? (((rp - kb_col_base) & 31u) + 1u)
                                   : 0u;
          const uint4 j0 = uint4(uint(sn)) + uint4(0u, 1u, 2u, 3u);
          const uint4 j1 = j0 + uint4(16u);
          const float4 hi_s = select(float4(0.0f), a, j0 < split_c) +
                              select(float4(0.0f), b, j1 < split_c);
          const float4 lo_s = select(float4(0.0f), a, j0 >= split_c) +
                              select(float4(0.0f), b, j1 >= split_c);
          float dhi = (hi_s.x + hi_s.y) + (hi_s.z + hi_s.w);
          float dlo = (lo_s.x + lo_s.y) + (lo_s.z + lo_s.w);
          dhi += simd_shuffle_xor(dhi, ushort(1));
          dhi += simd_shuffle_xor(dhi, ushort(8));
          dlo += simd_shuffle_xor(dlo, ushort(1));
          dlo += simd_shuffle_xor(dlo, ushort(8));
          if (sn == 0) {
            const uint rm_w = tm + r_loc;
            tg_mem.pool_buf[rm_w] = dhi + dlo;
            tg_mem.pool_buf[32u + rm_w] = dhi;
            tg_mem.pool_buf[64u + rm_w] = dlo;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if constexpr (PassMode == 1u) {
      if (flat_tid < active_tok) {
        const uint pos_w = tg_mem.q_pos[flat_tid];
        if (pos_w >= kb_col_base) {
          float v_max = 0.0f;
          float dhi_max = 0.0f;
          float dlo_max = 0.0f;
          const uint r_base = flat_tid * repeat;
          for (uint g = 0u; g < repeat; ++g) {
            v_max = max(v_max, tg_mem.pool_buf[r_base + g]);
            dhi_max = max(dhi_max, tg_mem.pool_buf[32u + r_base + g]);
            dlo_max = max(dlo_max, tg_mem.pool_buf[64u + r_base + g]);
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
    }
  }

  if constexpr (PassMode == 0u) {
    if (dp == 0u && sn == 0) {
      if (qv0) {
        const ulong rec0 =
            (ulong(w_base + tok0) * p.query_heads + qh0) * mp.split_count +
            split;
        probe_m_split[rec0] = m0;
        probe_l_split[rec0] = l0;
      }
      if (qv1) {
        const ulong rec1 =
            (ulong(w_base + tok1) * p.query_heads + qh1) * mp.split_count +
            split;
        probe_m_split[rec1] = m1;
        probe_l_split[rec1] = l1;
      }
    }
  }
}

} // namespace minf

// ============================================================================
// Stage 1A: Unified INT8-NAX Probe Split-KV Partition Stats (10.625 KiB SRAM, 0 Spills)
// ============================================================================
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_minference_probe_stats_i8(
    device const bfloat *query [[buffer(0)]],
    device const int8_t *k_int8 [[buffer(1)]],
    device const float *mu_k [[buffer(2)]],
    device const float *s_k [[buffer(3)]],
    device const uint *valid_kv_buf [[buffer(4)]],
    device const uint *positions [[buffer(5)]],
    device float *probe_m_split [[buffer(6)]],
    device float *probe_l_split [[buffer(7)]],
    constant MInferenceParams &mp [[buffer(8)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup minf::MinfProbeTgStorage<0u> tg_mem;
  minf::minference_probe_i8_impl<0u>(
      query, k_int8, mu_k, s_k, valid_kv_buf, positions, probe_m_split,
      probe_l_split, nullptr, nullptr, nullptr, nullptr, mp, tg_mem, lane, sg,
      tg);
}

// ============================================================================
// Stage 1B: Reduce Split Stats -> Exact (M_global, inv_Z) & Zero V_score/D_score
// ============================================================================
[[max_total_threads_per_threadgroup(256)]]
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
  const uint live_w_probe = min(mp.w_probe, mp.base.tokens);
  const uint total_rows = live_w_probe * mp.base.query_heads;
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
// Stage 1C: Unified INT8-NAX Normalized Probe Accumulation (11.00 KiB SRAM, 0 Spills)
// ============================================================================
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_minference_probe_accum_i8(
    device const bfloat *query [[buffer(0)]],
    device const int8_t *k_int8 [[buffer(1)]],
    device const float *mu_k [[buffer(2)]],
    device const float *s_k [[buffer(3)]],
    device const uint *valid_kv_buf [[buffer(4)]],
    device const uint *positions [[buffer(5)]],
    device const float *probe_m_global [[buffer(6)]],
    device const float *probe_inv_z [[buffer(7)]],
    device atomic<float> *v_score [[buffer(8)]],
    device atomic<float> *d_score [[buffer(9)]],
    constant MInferenceParams &mp [[buffer(10)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup minf::MinfProbeTgStorage<1u> tg_mem;
  minf::minference_probe_i8_impl<1u>(
      query, k_int8, mu_k, s_k, valid_kv_buf, positions, nullptr, nullptr,
      probe_m_global, probe_inv_z, v_score, d_score, mp, tg_mem, lane, sg, tg);
}

// ============================================================================
// Stage 2A: Exact Per-KVH Top-K_V Vertical & Top-K_S Slash Selection (1.07 KiB)
// ============================================================================
[[max_total_threads_per_threadgroup(256)]]
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
  if (mp.base.tokens == 0u) {
    if (tid == 0u) {
      top_counts[kvh * 2u + 0u] = 0u;
      top_counts[kvh * 2u + 1u] = 0u;
    }
    return;
  }

  threadgroup minf::MinfSelectTopkTg tg_mem;

  uint local_max_pos = 0u;
  for (uint t = tid; t < mp.base.tokens; t += 256u) {
    local_max_pos = max(local_max_pos, positions[t]);
  }
  uint wm = simd_max(local_max_pos);
  if (lane == 0u) {
    tg_mem.w_idx[sg] = wm;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    uint gmax_pos = 0u;
    for (uint s = 0u; s < 8u; ++s) {
      gmax_pos = max(gmax_pos, tg_mem.w_idx[s]);
    }
    tg_mem.nk_blocks[0] = min((gmax_pos + 32u) / 32u, mp.max_kv_blocks);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint nk_blocks = tg_mem.nk_blocks[0];
  device const float *vs = v_score + ulong(kvh) * mp.max_kv_blocks;
  device const float *ds = d_score + ulong(kvh) * mp.max_kv_blocks;
  device ushort *out_v = top_v_idx + ulong(kvh) * mp.top_stride;
  device ushort *out_s = top_s_lag + ulong(kvh) * mp.top_stride;

  {
    const uint bit_base = tid * 32u;
    uint mask = 0u;
    if (bit_base < mp.n_sink) {
      const uint rem = mp.n_sink - bit_base;
      mask = (rem >= 32u) ? 0xffffffffu : ((1u << rem) - 1u);
    }
    tg_mem.excl[tid] = mask;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint actual_v = 0u;
  const uint max_kv = min(mp.k_vertical, mp.top_stride);
  for (uint k = 0u; k < max_kv; ++k) {
    float best_v = -1.0f;
    uint best_i = 0xffffffffu;
    for (uint i = tid; i < nk_blocks; i += 256u) {
      if ((tg_mem.excl[i >> 5u] & (1u << (i & 31u))) == 0u) {
        const float val = vs[i];
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
      tg_mem.w_val[sg] = best_v;
      tg_mem.w_idx[sg] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float gv = -1.0f;
      uint gi = 0xffffffffu;
      for (uint s = 0u; s < 8u; ++s) {
        if (tg_mem.w_val[s] > gv ||
            (tg_mem.w_val[s] == gv && tg_mem.w_idx[s] < gi)) {
          gv = tg_mem.w_val[s];
          gi = tg_mem.w_idx[s];
        }
      }
      if (gv > 0.0f && gi < nk_blocks) {
        out_v[actual_v++] = ushort(gi);
        tg_mem.excl[gi >> 5u] |= (1u << (gi & 31u));
      }
      tg_mem.w_idx[0] = (gv > 0.0f && gi < nk_blocks) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tg_mem.w_idx[0] == 0u) {
      break;
    }
  }

  {
    const uint local_cut = mp.n_local + 1u;
    const uint bit_base = tid * 32u;
    uint mask = 0u;
    if (bit_base < local_cut) {
      const uint rem = local_cut - bit_base;
      mask = (rem >= 32u) ? 0xffffffffu : ((1u << rem) - 1u);
    }
    tg_mem.excl[tid] = mask;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint actual_s = 0u;
  const uint max_ks = min(mp.k_slash, mp.top_stride);
  for (uint k = 0u; k < max_ks; ++k) {
    float best_v = -1.0f;
    uint best_i = 0xffffffffu;
    for (uint i = tid; i < nk_blocks; i += 256u) {
      if ((tg_mem.excl[i >> 5u] & (1u << (i & 31u))) == 0u) {
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
      tg_mem.w_val[sg] = best_v;
      tg_mem.w_idx[sg] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float gv = -1.0f;
      uint gi = 0xffffffffu;
      for (uint s = 0u; s < 8u; ++s) {
        if (tg_mem.w_val[s] > gv ||
            (tg_mem.w_val[s] == gv && tg_mem.w_idx[s] < gi)) {
          gv = tg_mem.w_val[s];
          gi = tg_mem.w_idx[s];
        }
      }
      if (gv > 0.0f && gi < nk_blocks) {
        out_s[actual_s++] = ushort(gi);
        tg_mem.excl[gi >> 5u] |= (1u << (gi & 31u));
      }
      tg_mem.w_idx[0] = (gv > 0.0f && gi < nk_blocks) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tg_mem.w_idx[0] == 0u) {
      break;
    }
  }

  if (tid == 0u) {
    top_counts[kvh * 2u + 0u] = actual_v;
    top_counts[kvh * 2u + 1u] = actual_s;
  }
}

// ============================================================================
// Stage 2B: 1 KiB SRAM Bitmask + ctz Sorted Block-Index Builder (1.54 KiB)
// ============================================================================
[[max_total_threads_per_threadgroup(256)]]
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
  threadgroup minf::MinfBuildIndicesTg tg_mem;

  if (mp.base.tokens == 0u || mp.base.key_value_heads == 0u ||
      mp.bm_tok == 0u) {
    return;
  }
  const uint live_n_q_blocks = (mp.base.tokens + mp.bm_tok - 1u) / mp.bm_tok;
  const uint kvh = tg.y % mp.base.key_value_heads;
  const uint q_blk = tg.y / mp.base.key_value_heads;
  if (q_blk >= live_n_q_blocks || q_blk >= mp.max_q_blocks ||
      kvh >= mp.base.key_value_heads) {
    return;
  }

  atomic_store_explicit(&tg_mem.bitmask[tid], 0u, memory_order_relaxed);

  const uint q_base = q_blk * mp.bm_tok;
  const uint active_tok = min(mp.bm_tok, mp.base.tokens - q_base);
  const uint capped_tok = min(active_tok, 128u);

  if (tid < capped_tok) {
    tg_mem.q_pos[tid] = positions[q_base + tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    uint lo = 0xffffffffu;
    uint hi = 0u;
    for (uint r = 0u; r < capped_tok; ++r) {
      const uint pos = tg_mem.q_pos[r];
      lo = min(lo, pos);
      hi = max(hi, pos);
    }
    tg_mem.minmax[0] = lo;
    tg_mem.minmax[1] = hi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint min_q_pos = tg_mem.minmax[0];
  const uint max_q_pos = tg_mem.minmax[1];
  const uint nk_blocks_blk = min((max_q_pos + 32u) / 32u, mp.max_kv_blocks);
  const uint min_kb = min_q_pos >> 5u;
  const uint max_kb = max_q_pos >> 5u;
  const bool dense_local_span = (max_kb - min_kb) <= (2u * mp.n_local);

  const uint sink_end = min(mp.n_sink, nk_blocks_blk);
  for (uint b = tid; b < sink_end; b += 256u) {
    atomic_fetch_or_explicit(&tg_mem.bitmask[b >> 5u], 1u << (b & 31u),
                             memory_order_relaxed);
  }

  if (dense_local_span) {
    const uint local_start = (min_kb > mp.n_local) ? (min_kb - mp.n_local) : 0u;
    for (uint b = local_start + tid; b < nk_blocks_blk; b += 256u) {
      atomic_fetch_or_explicit(&tg_mem.bitmask[b >> 5u], 1u << (b & 31u),
                               memory_order_relaxed);
    }
  } else if (nk_blocks_blk > 0u) {
    for (uint r = 0u; r < capped_tok; ++r) {
      const uint kb_r = tg_mem.q_pos[r] >> 5u;
      const uint loc_lo = (kb_r > mp.n_local) ? (kb_r - mp.n_local) : 0u;
      const uint loc_hi = min(nk_blocks_blk - 1u, kb_r);
      for (uint b = loc_lo + tid; b <= loc_hi; b += 256u) {
        atomic_fetch_or_explicit(&tg_mem.bitmask[b >> 5u], 1u << (b & 31u),
                                 memory_order_relaxed);
      }
    }
  }

  const uint actual_v = top_counts[kvh * 2u + 0u];
  device const ushort *v_list = top_v_idx + ulong(kvh) * mp.top_stride;
  for (uint i = tid; i < actual_v; i += 256u) {
    const uint b = uint(v_list[i]);
    if (b < nk_blocks_blk) {
      atomic_fetch_or_explicit(&tg_mem.bitmask[b >> 5u], 1u << (b & 31u),
                               memory_order_relaxed);
    }
  }

  const uint actual_s = top_counts[kvh * 2u + 1u];
  device const ushort *s_list = top_s_lag + ulong(kvh) * mp.top_stride;
  if (nk_blocks_blk > 0u) {
    if (dense_local_span) {
      for (uint i = tid; i < actual_s; i += 256u) {
        const uint lag = uint(s_list[i]);
        const uint lag_lo_off = lag * 32u;
        if (max_q_pos >= lag_lo_off) {
          const uint k_hi = max_q_pos - lag_lo_off;
          const uint lag_hi_off = (lag + 1u) * 32u - 1u;
          const uint k_lo =
              (min_q_pos > lag_hi_off) ? (min_q_pos - lag_hi_off) : 0u;
          const uint kb_lo = k_lo >> 5u;
          const uint kb_hi = min(nk_blocks_blk - 1u, k_hi >> 5u);
          for (uint b = kb_lo; b <= kb_hi; ++b) {
            atomic_fetch_or_explicit(&tg_mem.bitmask[b >> 5u], 1u << (b & 31u),
                                     memory_order_relaxed);
          }
        }
      }
    } else {
      for (uint i = tid; i < actual_s; i += 256u) {
        const uint lag = uint(s_list[i]);
        const uint lag_lo_off = lag * 32u;
        const uint lag_hi_off = (lag + 1u) * 32u - 1u;
        for (uint r = 0u; r < capped_tok; ++r) {
          const uint qp = tg_mem.q_pos[r];
          if (qp >= lag_lo_off) {
            const uint k_hi = qp - lag_lo_off;
            const uint k_lo = (qp > lag_hi_off) ? (qp - lag_hi_off) : 0u;
            const uint kb_lo = k_lo >> 5u;
            const uint kb_hi = min(nk_blocks_blk - 1u, k_hi >> 5u);
            for (uint b = kb_lo; b <= kb_hi; ++b) {
              atomic_fetch_or_explicit(&tg_mem.bitmask[b >> 5u],
                                       1u << (b & 31u), memory_order_relaxed);
            }
          }
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint w = atomic_load_explicit(&tg_mem.bitmask[tid], memory_order_relaxed);
  const uint cnt = popcount(w);
  const uint lane_excl = simd_prefix_exclusive_sum(cnt);
  if (lane == 31u) {
    tg_mem.warp_totals[sg] = lane_excl + cnt;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint warp_base = 0u;
  for (uint s = 0u; s < sg; ++s) {
    warp_base += tg_mem.warp_totals[s];
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
