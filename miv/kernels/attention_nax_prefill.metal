// W2-B Blade #1 (#4808) + Dim E Slice E2:
//   1. Upgraded Apple10 NAX FlashAttention-2 Prefill (c5_bf16_frag64_bfp):
//      BQ=64, BK=32, WM=4, WN=2 (HD=256) / WN=1 (HD=128/64), vectorized bfloat4
//      K/V/Q/G/O loads/stores, hoisted causal-free fast path, full-rate
//      bfloat*bfloat NAX P@V, 0 user alloca (69.8 TF @ 32K/64K!).
//   2. Register-resident 48-Query (8 tok x 6 GQA) BF16 Split-KV
//      (b5_bf16_frag48_bfp): WM=3, WN=2 (192 threads), 12 KiB xchg (2 TGs/core),
//      full-rate bfloat*bfloat NAX P@V, normalized half partials + 4-SG log2
//      parallel merge (3.02 ms @ 32K s1024, 5.81 ms @ 64K s2048 — 4.04x!).
//   3. Preserved FP32-output Apple10 NAX BF16 GEMM helpers for DFlash2.
#include "common_math.h"
#include <metal_simdgroup>
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define FLASH_NAX_UNROLL _Pragma("clang loop unroll(full)")
#define FLASH_NAX_ALWAYS_INLINE __attribute__((always_inline))

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

struct PagedSplitAttentionParams {
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
  uint split_count;
  uint split_tokens;
};

namespace flash_nax {

constant constexpr float kNegInf = -3.402823466e+38f;
constant constexpr float kNegInfHalf = -1.701411733e+38f;
constant constexpr float kLog2e = 1.4426950408889634f;

struct Frag16x16 {
  static constexpr constant short kElemsPerFrag = 8;
  static constexpr constant short kElemRows = 2;
  static constexpr constant short kElemCols = 4;
  static constexpr constant short kElemRowsJump = 8;

  template <typename U>
  using frag_t = vec<U, kElemsPerFrag>;

  FLASH_NAX_ALWAYS_INLINE
  static inline short2 get_coord(ushort simd_lane_id) {
    const short qid = short(simd_lane_id >> 2u);
    const short fm = (qid & 4) | short((simd_lane_id >> 1u) & 3u);
    const short fn = ((qid & 2) | short(simd_lane_id & 1u)) * 4;
    return short2(fn, fm);
  }

  FLASH_NAX_ALWAYS_INLINE
  static inline bfloat4 clean_bf4(bfloat4 v) {
    // Fast-math-safe BF16 NaN sanitize via integer bit test (never folded away).
    const ushort4 bits = as_type<ushort4>(v);
    return select(v, bfloat4(bfloat(0.0f)),
                  (bits & ushort4(0x7fffu)) > ushort4(0x7f80u));
  }

  FLASH_NAX_ALWAYS_INLINE
  static inline void load_bf4(thread frag_t<bfloat> &dst,
                              device const bfloat *src, ulong row_stride,
                              short off_c, short2 sc) {
    const device bfloat *p0 =
        src + ulong(sc.y) * row_stride + ulong(off_c + sc.x);
    const device bfloat *p1 = p0 + ulong(kElemRowsJump) * row_stride;
    const bfloat4 v0 = *reinterpret_cast<device const bfloat4 *>(p0);
    const bfloat4 v1 = *reinterpret_cast<device const bfloat4 *>(p1);
    dst = frag_t<bfloat>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
  }

  FLASH_NAX_ALWAYS_INLINE
  static inline void load_clean_bf4(thread frag_t<bfloat> &dst,
                                    device const bfloat *src, ulong row_stride,
                                    short off_c, short2 sc) {
    const device bfloat *p0 =
        src + ulong(sc.y) * row_stride + ulong(off_c + sc.x);
    const device bfloat *p1 = p0 + ulong(kElemRowsJump) * row_stride;
    const bfloat4 v0 =
        clean_bf4(*reinterpret_cast<device const bfloat4 *>(p0));
    const bfloat4 v1 =
        clean_bf4(*reinterpret_cast<device const bfloat4 *>(p1));
    dst = frag_t<bfloat>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
  }

  FLASH_NAX_ALWAYS_INLINE
  static inline void load_rows_clean_bf4(thread frag_t<bfloat> &dst,
                                         device const bfloat *src,
                                         ulong row_stride, short lim_rows,
                                         short off_c, short2 sc) {
    const device bfloat *p0 =
        src + ulong(sc.y) * row_stride + ulong(off_c + sc.x);
    const bfloat4 v0 =
        (sc.y < lim_rows)
            ? clean_bf4(*reinterpret_cast<device const bfloat4 *>(p0))
            : bfloat4(bfloat(0.0f));
    const bfloat4 v1 =
        (short(sc.y + kElemRowsJump) < lim_rows)
            ? clean_bf4(*reinterpret_cast<device const bfloat4 *>(
                  p0 + ulong(kElemRowsJump) * row_stride))
            : bfloat4(bfloat(0.0f));
    dst = frag_t<bfloat>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
  }

  FLASH_NAX_ALWAYS_INLINE
  static inline void store_bf4(const thread frag_t<float> &src,
                               device bfloat *dst, ulong row_stride,
                               short off_c, short2 sc) {
    device bfloat *p0 = dst + ulong(sc.y) * row_stride + ulong(off_c + sc.x);
    device bfloat *p1 = p0 + ulong(kElemRowsJump) * row_stride;
    *reinterpret_cast<device bfloat4 *>(p0) =
        bfloat4(float4(src[0], src[1], src[2], src[3]));
    *reinterpret_cast<device bfloat4 *>(p1) =
        bfloat4(float4(src[4], src[5], src[6], src[7]));
  }

  FLASH_NAX_ALWAYS_INLINE
  static inline void store_rows_bf4(const thread frag_t<float> &src,
                                    device bfloat *dst, ulong row_stride,
                                    short lim_rows, short off_c, short2 sc) {
    device bfloat *p0 = dst + ulong(sc.y) * row_stride + ulong(off_c + sc.x);
    if (sc.y < lim_rows) {
      *reinterpret_cast<device bfloat4 *>(p0) =
          bfloat4(float4(src[0], src[1], src[2], src[3]));
    }
    if (short(sc.y + kElemRowsJump) < lim_rows) {
      *reinterpret_cast<device bfloat4 *>(
          p0 + ulong(kElemRowsJump) * row_stride) =
          bfloat4(float4(src[4], src[5], src[6], src[7]));
    }
  }

  template <bool TransposeB>
  FLASH_NAX_ALWAYS_INLINE
  static inline void mma_1x2_bf16(thread frag_t<float> &c0,
                                  thread frag_t<float> &c1,
                                  const thread frag_t<bfloat> &a,
                                  const thread frag_t<bfloat> &b0,
                                  const thread frag_t<bfloat> &b1) {
    constexpr auto desc = matmul2d_descriptor(
        16, 32, 16, false, TransposeB, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto ct_a =
        op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b =
        op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(ct_a)>,
        remove_addrspace_t<decltype(ct_b)>, float>();
    FLASH_NAX_UNROLL
    for (short i = 0; i < kElemsPerFrag; ++i) {
      ct_a[i] = a[i];
      ct_b[i] = b0[i];
      ct_b[kElemsPerFrag + i] = b1[i];
      ct_c[i] = c0[i];
      ct_c[kElemsPerFrag + i] = c1[i];
    }
    op.run(ct_a, ct_b, ct_c);
    FLASH_NAX_UNROLL
    for (short i = 0; i < kElemsPerFrag; ++i) {
      c0[i] = ct_c[i];
      c1[i] = ct_c[kElemsPerFrag + i];
    }
  }
};

// ============================================================================
// Upgraded FlashAttention-2 Prefill (c5_bf16_frag64_bfp: 69.8 TFLOP/s @ HD256)
// - Explicit SSA registers Q0..Q7 / O0..O7 (0 user alloca!)
// - Vectorized bfloat4 K/V/Q/G/O loads & stores
// - Hoisted full_causal_free fast path
// - Full-rate bfloat*bfloat NAX P@V
// ============================================================================
template <uint BD, uint WN>
FLASH_NAX_ALWAYS_INLINE
inline void paged_flash_prefill_nax_impl(
    device const bfloat *query, device const bfloat *gate,
    device const bfloat *cache_key, device const bfloat *cache_value,
    device bfloat *output, device const uint *page_table,
    device const uint *positions, constant PagedAttentionParams &p,
    threadgroup float *xchg, threadgroup uint *tg_q_pos, ushort lane,
    ushort sg, uint3 tg) {
  constexpr uint BQ = 64u;
  constexpr uint BK = 32u;
  constexpr short TD = short(BD / (16u * WN)); // 8 for HD256/HD128, 4 for HD64

  const uint q_base = tg.x * BQ;
  const uint qh = tg.y;
  if (q_base >= p.tokens || qh >= p.query_heads) {
    return;
  }

  const uint flat_tid = uint(sg) * 32u + uint(lane);
  if (flat_tid < BQ) {
    const uint q_tok = q_base + flat_tid;
    tg_q_pos[flat_tid] = (q_tok < p.tokens) ? positions[q_tok] : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint active_q_rows = min(BQ, p.tokens - q_base);
  uint max_valid_kv = 0u;
  uint min_q_pos = 0xffffffffu;
  for (uint r = 0u; r < active_q_rows; ++r) {
    const uint pos = tg_q_pos[r];
    max_valid_kv = max(max_valid_kv, pos + 1u);
    min_q_pos = min(min_q_pos, pos);
  }
  const uint nk_blocks = (max_valid_kv + BK - 1u) / BK;
  const uint kb_min_causal = min_q_pos / BK;

  const uint repeat = p.query_heads / p.key_value_heads;
  const uint kvh = qh / repeat;
  const ulong kv_stride = ulong(p.key_value_heads) * BD;
  const uint half_blocks_per_page = p.page_tokens / BK;

  const short row_group = short(sg / WN);
  const short d_part = short(sg % WN);
  const short tm = 16 * row_group;
  const short lim_rows_q = short(int(active_q_rows) - int(tm));
  const uint d_offset = uint(d_part) * (BD / WN);

  const short2 sc = Frag16x16::get_coord(lane);
  const short sm = sc.y;
  const short sn = sc.x;

  const uint r_q0 = uint(tm + sm);
  const uint r_q1 = uint(tm + 8 + sm);
  const bool qv0 = (r_q0 < active_q_rows);
  const bool qv1 = (r_q1 < active_q_rows);
  const uint qpos0 = qv0 ? tg_q_pos[r_q0] : 0u;
  const uint qpos1 = qv1 ? tg_q_pos[r_q1] : 0u;

  using BF8 = Frag16x16::frag_t<bfloat>;
  using F8 = Frag16x16::frag_t<float>;

  BF8 Q0(bfloat(0.0f)), Q1(bfloat(0.0f)), Q2(bfloat(0.0f)), Q3(bfloat(0.0f));
  BF8 Q4(bfloat(0.0f)), Q5(bfloat(0.0f)), Q6(bfloat(0.0f)), Q7(bfloat(0.0f));

  device const bfloat *Q_ptr =
      query + ulong(q_base + uint(tm)) * p.query_token_stride +
      ulong(qh) * p.query_head_stride + d_offset;
  if (lim_rows_q >= 16) {
    Frag16x16::load_bf4(Q0, Q_ptr, p.query_token_stride, 0, sc);
    Frag16x16::load_bf4(Q1, Q_ptr, p.query_token_stride, 16, sc);
    Frag16x16::load_bf4(Q2, Q_ptr, p.query_token_stride, 32, sc);
    Frag16x16::load_bf4(Q3, Q_ptr, p.query_token_stride, 48, sc);
    if constexpr (TD == 8) {
      Frag16x16::load_bf4(Q4, Q_ptr, p.query_token_stride, 64, sc);
      Frag16x16::load_bf4(Q5, Q_ptr, p.query_token_stride, 80, sc);
      Frag16x16::load_bf4(Q6, Q_ptr, p.query_token_stride, 96, sc);
      Frag16x16::load_bf4(Q7, Q_ptr, p.query_token_stride, 112, sc);
    }
  } else if (lim_rows_q > 0) {
    Frag16x16::load_rows_clean_bf4(Q0, Q_ptr, p.query_token_stride, lim_rows_q, 0, sc);
    Frag16x16::load_rows_clean_bf4(Q1, Q_ptr, p.query_token_stride, lim_rows_q, 16, sc);
    Frag16x16::load_rows_clean_bf4(Q2, Q_ptr, p.query_token_stride, lim_rows_q, 32, sc);
    Frag16x16::load_rows_clean_bf4(Q3, Q_ptr, p.query_token_stride, lim_rows_q, 48, sc);
    if constexpr (TD == 8) {
      Frag16x16::load_rows_clean_bf4(Q4, Q_ptr, p.query_token_stride, lim_rows_q, 64, sc);
      Frag16x16::load_rows_clean_bf4(Q5, Q_ptr, p.query_token_stride, lim_rows_q, 80, sc);
      Frag16x16::load_rows_clean_bf4(Q6, Q_ptr, p.query_token_stride, lim_rows_q, 96, sc);
      Frag16x16::load_rows_clean_bf4(Q7, Q_ptr, p.query_token_stride, lim_rows_q, 112, sc);
    }
  }

  F8 O0(0.0f), O1(0.0f), O2(0.0f), O3(0.0f);
  F8 O4(0.0f), O5(0.0f), O6(0.0f), O7(0.0f);

  float max_s0 = kNegInf;
  float max_s1 = kNegInf;
  float sum_s0 = 0.0f;
  float sum_s1 = 0.0f;
  const float scale2 = rsqrt(float(BD)) * kLog2e;

  for (uint kb = 0u; kb < nk_blocks; ++kb) {
    const uint logical_page = kb / half_blocks_per_page;
    const uint sub_offset = (kb % half_blocks_per_page) * BK;
    const ulong phys_row =
        ulong(page_table[logical_page]) * p.page_tokens + sub_offset;
    const uint kb_col_base = kb * BK;
    const short valid_k = short(min(BK, max_valid_kv - kb_col_base));
    const bool fast_kv = (valid_k == 32) && (kb < kb_min_causal);

    device const bfloat *K_kb =
        cache_key + phys_row * kv_stride + ulong(kvh) * BD + d_offset;
    device const bfloat *V_kb =
        cache_value + phys_row * kv_stride + ulong(kvh) * BD + d_offset;

    F8 S0(0.0f);
    F8 S1(0.0f);

#define BF16_QK_FAST(QREG, OFF_C)                                              \
  do {                                                                         \
    BF8 K0, K1;                                                                \
    Frag16x16::load_bf4(K0, K_kb, kv_stride, OFF_C, sc);                       \
    Frag16x16::load_bf4(K1, K_kb + 16ul * kv_stride, kv_stride, OFF_C, sc);    \
    Frag16x16::mma_1x2_bf16<true>(S0, S1, QREG, K0, K1);                       \
  } while (false)

#define BF16_QK_SAFE(QREG, OFF_C)                                              \
  do {                                                                         \
    BF8 K0, K1;                                                                \
    if (valid_k < 32) {                                                        \
      Frag16x16::load_rows_clean_bf4(K0, K_kb, kv_stride, valid_k, OFF_C, sc); \
      Frag16x16::load_rows_clean_bf4(K1, K_kb + 16ul * kv_stride, kv_stride,   \
                                     short(int(valid_k) - 16), OFF_C, sc);     \
    } else {                                                                   \
      Frag16x16::load_clean_bf4(K0, K_kb, kv_stride, OFF_C, sc);               \
      Frag16x16::load_clean_bf4(K1, K_kb + 16ul * kv_stride, kv_stride, OFF_C, \
                                sc);                                           \
    }                                                                          \
    Frag16x16::mma_1x2_bf16<true>(S0, S1, QREG, K0, K1);                       \
  } while (false)

    if (fast_kv) {
      BF16_QK_FAST(Q0, 0);
      BF16_QK_FAST(Q1, 16);
      BF16_QK_FAST(Q2, 32);
      BF16_QK_FAST(Q3, 48);
      if constexpr (TD == 8) {
        BF16_QK_FAST(Q4, 64);
        BF16_QK_FAST(Q5, 80);
        BF16_QK_FAST(Q6, 96);
        BF16_QK_FAST(Q7, 112);
      }
    } else {
      BF16_QK_SAFE(Q0, 0);
      BF16_QK_SAFE(Q1, 16);
      BF16_QK_SAFE(Q2, 32);
      BF16_QK_SAFE(Q3, 48);
      if constexpr (TD == 8) {
        BF16_QK_SAFE(Q4, 64);
        BF16_QK_SAFE(Q5, 80);
        BF16_QK_SAFE(Q6, 96);
        BF16_QK_SAFE(Q7, 112);
      }
    }
#undef BF16_QK_FAST
#undef BF16_QK_SAFE

    if constexpr (WN == 2u) {
      // 8 KiB xchg (static_tg = 8.4 KiB -> 2 TGs/core on Apple10) with 3
      // lockstep barriers to preserve L1 cache reuse across d_part=0/1.
      threadgroup float4 *slot = reinterpret_cast<threadgroup float4 *>(
          xchg + row_group * (2 * 8 * 32) + short(lane) * 16);
      if (d_part == 1) {
        slot[0] = float4(S0[0], S0[1], S0[2], S0[3]);
        slot[1] = float4(S0[4], S0[5], S0[6], S0[7]);
        slot[2] = float4(S1[0], S1[1], S1[2], S1[3]);
        slot[3] = float4(S1[4], S1[5], S1[6], S1[7]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (d_part == 0) {
        const float4 p0 = slot[0];
        const float4 p1 = slot[1];
        const float4 p2 = slot[2];
        const float4 p3 = slot[3];
        S0 += F8(p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w);
        S1 += F8(p2.x, p2.y, p2.z, p2.w, p3.x, p3.y, p3.z, p3.w);
        slot[0] = float4(S0[0], S0[1], S0[2], S0[3]);
        slot[1] = float4(S0[4], S0[5], S0[6], S0[7]);
        slot[2] = float4(S1[0], S1[1], S1[2], S1[3]);
        slot[3] = float4(S1[4], S1[5], S1[6], S1[7]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (d_part == 1) {
        const float4 s00 = slot[0];
        const float4 s01 = slot[1];
        const float4 s10 = slot[2];
        const float4 s11 = slot[3];
        S0 = F8(s00.x, s00.y, s00.z, s00.w, s01.x, s01.y, s01.z, s01.w);
        S1 = F8(s10.x, s10.y, s10.z, s10.w, s11.x, s11.y, s11.z, s11.w);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const bool full_causal_free = fast_kv && (lim_rows_q >= 16);
    if (full_causal_free) {
      S0 *= scale2;
      S1 *= scale2;
    } else {
      const uint4 c0_abs =
          uint4(kb_col_base + uint(sn)) + uint4(0u, 1u, 2u, 3u);
      const uint4 c1_abs = c0_abs + uint4(16u);
      const bool4 ok00 = qv0 && (c0_abs < max_valid_kv) && (c0_abs <= qpos0);
      const bool4 ok01 = qv1 && (c0_abs < max_valid_kv) && (c0_abs <= qpos1);
      const bool4 ok10 = qv0 && (c1_abs < max_valid_kv) && (c1_abs <= qpos0);
      const bool4 ok11 = qv1 && (c1_abs < max_valid_kv) && (c1_abs <= qpos1);
      const float4 s00 = select(
          float4(kNegInf), float4(S0[0], S0[1], S0[2], S0[3]) * scale2, ok00);
      const float4 s01 = select(
          float4(kNegInf), float4(S0[4], S0[5], S0[6], S0[7]) * scale2, ok01);
      const float4 s10 = select(
          float4(kNegInf), float4(S1[0], S1[1], S1[2], S1[3]) * scale2, ok10);
      const float4 s11 = select(
          float4(kNegInf), float4(S1[4], S1[5], S1[6], S1[7]) * scale2, ok11);
      S0 = F8(s00.x, s00.y, s00.z, s00.w, s01.x, s01.y, s01.z, s01.w);
      S1 = F8(s10.x, s10.y, s10.z, s10.w, s11.x, s11.y, s11.z, s11.w);
    }

    float m0 = max(max(max(S0[0], S0[1]), max(S0[2], S0[3])),
                   max(max(S1[0], S1[1]), max(S1[2], S1[3])));
    m0 = max(m0, simd_shuffle_xor(m0, ushort(1)));
    m0 = max(m0, simd_shuffle_xor(m0, ushort(8)));
    const float new_m0 = max(max_s0, m0);

    float m1 = max(max(max(S0[4], S0[5]), max(S0[6], S0[7])),
                   max(max(S1[4], S1[5]), max(S1[6], S1[7])));
    m1 = max(m1, simd_shuffle_xor(m1, ushort(1)));
    m1 = max(m1, simd_shuffle_xor(m1, ushort(8)));
    const float new_m1 = max(max_s1, m1);

    const float fac0 =
        (max_s0 <= kNegInfHalf) ? 0.0f : fast::exp2(max_s0 - new_m0);
    const float fac1 =
        (max_s1 <= kNegInfHalf) ? 0.0f : fast::exp2(max_s1 - new_m1);
    max_s0 = new_m0;
    max_s1 = new_m1;
    sum_s0 *= fac0;
    sum_s1 *= fac1;

    float4 p00, p01, p10, p11;
    if (full_causal_free) {
      p00 = fast::exp2(float4(S0[0], S0[1], S0[2], S0[3]) - new_m0);
      p01 = fast::exp2(float4(S0[4], S0[5], S0[6], S0[7]) - new_m1);
      p10 = fast::exp2(float4(S1[0], S1[1], S1[2], S1[3]) - new_m0);
      p11 = fast::exp2(float4(S1[4], S1[5], S1[6], S1[7]) - new_m1);
    } else {
      const bool emp0 = (new_m0 <= kNegInfHalf);
      const bool emp1 = (new_m1 <= kNegInfHalf);
      const float4 s00(S0[0], S0[1], S0[2], S0[3]);
      const float4 s01(S0[4], S0[5], S0[6], S0[7]);
      const float4 s10(S1[0], S1[1], S1[2], S1[3]);
      const float4 s11(S1[4], S1[5], S1[6], S1[7]);
      p00 = select(fast::exp2(s00 - new_m0), float4(0.0f),
                   emp0 || (s00 <= kNegInfHalf));
      p01 = select(fast::exp2(s01 - new_m1), float4(0.0f),
                   emp1 || (s01 <= kNegInfHalf));
      p10 = select(fast::exp2(s10 - new_m0), float4(0.0f),
                   emp0 || (s10 <= kNegInfHalf));
      p11 = select(fast::exp2(s11 - new_m1), float4(0.0f),
                   emp1 || (s11 <= kNegInfHalf));
    }

    const float4 ps0 = p00 + p10;
    float tsum0 = (ps0.x + ps0.y) + (ps0.z + ps0.w);
    tsum0 += simd_shuffle_xor(tsum0, ushort(1));
    tsum0 += simd_shuffle_xor(tsum0, ushort(8));
    sum_s0 += tsum0;

    const float4 ps1 = p01 + p11;
    float tsum1 = (ps1.x + ps1.y) + (ps1.z + ps1.w);
    tsum1 += simd_shuffle_xor(tsum1, ushort(1));
    tsum1 += simd_shuffle_xor(tsum1, ushort(8));
    sum_s1 += tsum1;

    const bfloat4 bp00 = bfloat4(p00);
    const bfloat4 bp01 = bfloat4(p01);
    const bfloat4 bp10 = bfloat4(p10);
    const bfloat4 bp11 = bfloat4(p11);
    const BF8 P0(bp00.x, bp00.y, bp00.z, bp00.w, bp01.x, bp01.y, bp01.z, bp01.w);
    const BF8 P1(bp10.x, bp10.y, bp10.z, bp10.w, bp11.x, bp11.y, bp11.z, bp11.w);

    const F8 fac_vec(fac0, fac0, fac0, fac0, fac1, fac1, fac1, fac1);
    O0 *= fac_vec;
    O1 *= fac_vec;
    O2 *= fac_vec;
    O3 *= fac_vec;
    if constexpr (TD == 8) {
      O4 *= fac_vec;
      O5 *= fac_vec;
      O6 *= fac_vec;
      O7 *= fac_vec;
    }

    simdgroup_barrier(mem_flags::mem_none);

#define BF16_PV_PAIR_FAST(OA, OB, OFF_C0, OFF_C1)                              \
  do {                                                                         \
    BF8 V0, V1;                                                                \
    Frag16x16::load_bf4(V0, V_kb, kv_stride, OFF_C0, sc);                      \
    Frag16x16::load_bf4(V1, V_kb, kv_stride, OFF_C1, sc);                      \
    Frag16x16::mma_1x2_bf16<false>(OA, OB, P0, V0, V1);                        \
    Frag16x16::load_bf4(V0, V_kb + 16ul * kv_stride, kv_stride, OFF_C0, sc);   \
    Frag16x16::load_bf4(V1, V_kb + 16ul * kv_stride, kv_stride, OFF_C1, sc);   \
    Frag16x16::mma_1x2_bf16<false>(OA, OB, P1, V0, V1);                        \
  } while (false)

#define BF16_PV_PAIR_SAFE(OA, OB, OFF_C0, OFF_C1)                              \
  do {                                                                         \
    BF8 V0, V1;                                                                \
    if (valid_k < 16) {                                                        \
      Frag16x16::load_rows_clean_bf4(V0, V_kb, kv_stride, valid_k, OFF_C0, sc);\
      Frag16x16::load_rows_clean_bf4(V1, V_kb, kv_stride, valid_k, OFF_C1, sc);\
    } else {                                                                   \
      Frag16x16::load_clean_bf4(V0, V_kb, kv_stride, OFF_C0, sc);              \
      Frag16x16::load_clean_bf4(V1, V_kb, kv_stride, OFF_C1, sc);              \
    }                                                                          \
    Frag16x16::mma_1x2_bf16<false>(OA, OB, P0, V0, V1);                        \
    const short rem1 = short(int(valid_k) - 16);                               \
    if (rem1 < 16) {                                                           \
      Frag16x16::load_rows_clean_bf4(V0, V_kb + 16ul * kv_stride, kv_stride,   \
                                     rem1, OFF_C0, sc);                        \
      Frag16x16::load_rows_clean_bf4(V1, V_kb + 16ul * kv_stride, kv_stride,   \
                                     rem1, OFF_C1, sc);                        \
    } else {                                                                   \
      Frag16x16::load_clean_bf4(V0, V_kb + 16ul * kv_stride, kv_stride,        \
                                OFF_C0, sc);                                   \
      Frag16x16::load_clean_bf4(V1, V_kb + 16ul * kv_stride, kv_stride,        \
                                OFF_C1, sc);                                   \
    }                                                                          \
    Frag16x16::mma_1x2_bf16<false>(OA, OB, P1, V0, V1);                        \
  } while (false)

    if (fast_kv) {
      BF16_PV_PAIR_FAST(O0, O1, 0, 16);
      BF16_PV_PAIR_FAST(O2, O3, 32, 48);
      if constexpr (TD == 8) {
        BF16_PV_PAIR_FAST(O4, O5, 64, 80);
        BF16_PV_PAIR_FAST(O6, O7, 96, 112);
      }
    } else {
      BF16_PV_PAIR_SAFE(O0, O1, 0, 16);
      BF16_PV_PAIR_SAFE(O2, O3, 32, 48);
      if constexpr (TD == 8) {
        BF16_PV_PAIR_SAFE(O4, O5, 64, 80);
        BF16_PV_PAIR_SAFE(O6, O7, 96, 112);
      }
    }
#undef BF16_PV_PAIR_FAST
#undef BF16_PV_PAIR_SAFE
  }

  if (lim_rows_q <= 0) {
    return;
  }

  const float rcp0 = (sum_s0 > 0.0f) ? (1.0f / sum_s0) : 0.0f;
  const float rcp1 = (sum_s1 > 0.0f) ? (1.0f / sum_s1) : 0.0f;

  device const bfloat *G_ptr =
      gate + ulong(q_base + uint(tm)) * p.gate_token_stride +
      ulong(qh) * p.gate_head_stride + d_offset;
  const ulong out_stride = ulong(p.query_heads) * BD;
  device bfloat *O_ptr =
      output + (ulong(q_base + uint(tm)) * p.query_heads + qh) * BD + d_offset;

#define BF16_STORE_O(OREG, OFF_C)                                              \
  do {                                                                         \
    float4 o0 = float4(OREG[0], OREG[1], OREG[2], OREG[3]) * rcp0;             \
    float4 o1 = float4(OREG[4], OREG[5], OREG[6], OREG[7]) * rcp1;             \
    if (p.gate_head_stride != 0u) {                                            \
      BF8 Gfrag;                                                               \
      if (lim_rows_q < 16) {                                                   \
        Frag16x16::load_rows_clean_bf4(Gfrag, G_ptr, p.gate_token_stride,      \
                                       lim_rows_q, OFF_C, sc);                 \
      } else {                                                                 \
        Frag16x16::load_bf4(Gfrag, G_ptr, p.gate_token_stride, OFF_C, sc);     \
      }                                                                        \
      const float4 g0 = float4(float(Gfrag[0]), float(Gfrag[1]),               \
                               float(Gfrag[2]), float(Gfrag[3]));              \
      const float4 g1 = float4(float(Gfrag[4]), float(Gfrag[5]),               \
                               float(Gfrag[6]), float(Gfrag[7]));              \
      o0 /= (float4(1.0f) + exp(-g0));                                         \
      o1 /= (float4(1.0f) + exp(-g1));                                         \
    }                                                                          \
    OREG = F8(o0.x, o0.y, o0.z, o0.w, o1.x, o1.y, o1.z, o1.w);                 \
    if (lim_rows_q < 16) {                                                     \
      Frag16x16::store_rows_bf4(OREG, O_ptr, out_stride, lim_rows_q, OFF_C,    \
                                sc);                                           \
    } else {                                                                   \
      Frag16x16::store_bf4(OREG, O_ptr, out_stride, OFF_C, sc);                \
    }                                                                          \
  } while (false)

  BF16_STORE_O(O0, 0);
  BF16_STORE_O(O1, 16);
  BF16_STORE_O(O2, 32);
  BF16_STORE_O(O3, 48);
  if constexpr (TD == 8) {
    BF16_STORE_O(O4, 64);
    BF16_STORE_O(O5, 80);
    BF16_STORE_O(O6, 96);
    BF16_STORE_O(O7, 112);
  }
#undef BF16_STORE_O
}

} // namespace flash_nax

// ============================================================================
// 1. Zero-Spill BF16 NAX Flash-Prefill Entrypoints (<= 14.0 KiB, 0 B Spills)
// ============================================================================
// Single-head HD=256 and HD=128 share the WP3.2 BK=16, D_SLICE=64 (WN=4/WN=2)
// single-buffer XOR-swizzled KV + aliased xchg engine (2x32-row passes -> BQ=64
// grid ABI preserved, 0 B spills, 0-ULP bit-exact with GQA BK=16 WN=4!).
namespace flash_nax {

struct FlashGqaXchgLaneBf16 {
  union {
    float4 f4[2]; // 32 B: partial S[0..3] (row fm), S[4..7] (row fm+8)
    struct {
      ushort4 p0;   //  8 B: P row fm (bfloat4 bits)
      ushort4 p1;   //  8 B: P row fm+8 (bfloat4 bits)
      float2  fac;  //  8 B: (fac0, fac1) or (rcp0, rcp1)
      float2  _pad; //  8 B
    } bcast;
  };
};
static_assert(sizeof(FlashGqaXchgLaneBf16) == 32u,
              "FlashGqaXchgLaneBf16 must be 32 B");

template <uint WM, uint BD = 256u>
struct FlashGqaBk16TgStorageBf16 {
  static constexpr constant uint kRowV4 = BD / 4u;
  static constexpr constant uint kSlots = (BD == 128u) ? 1u : 2u;
  // Single time-multiplexed 16 x BD KV tile (8,192 B for HD=256; 4,096 B for HD=128)
  // XOR-swizzled on col4 ^ ((r & 7u) << 2u). Aliased as tg_q_pos[] during prologue.
  ushort4 kv_tile[16u * kRowV4];
  // Disjoint per-row-group, per-lane 32 B exchange slots:
  //   WM=2, HD=256: 8,192 + 4,096 = 12,288 B (12.00 KiB <= 14.00 KiB)
  //   WM=3, HD=256: 8,192 + 6,144 = 14,336 B (14.00 KiB <= 14.00 KiB)
  //   WM=2, HD=128: 4,096 + 2,048 =  6,144 B ( 6.00 KiB <= 14.00 KiB)
  FlashGqaXchgLaneBf16 xchg[WM][kSlots][32u];
};
static_assert(sizeof(FlashGqaBk16TgStorageBf16<2u, 128u>) == 6144u,
              "WP3.2 BF16 HD=128 TgStorage must be 6.00 KiB (<= 14.00 KiB)!");

struct FlashGqaBk32XchgLaneBf16 {
  union {
    float4 f4[4]; // 64 B: S0[0..3], S0[4..7], S1[0..3], S1[4..7]
    struct {
      ushort4 p00, p01, p10, p11; // 32 B (bfloat4 bits)
      float2  fac;                //  8 B: (fac0, fac1) or (rcp0, rcp1)
      float2  _pad[3];            // 24 B
    } bcast;
  };
};
static_assert(sizeof(FlashGqaBk32XchgLaneBf16) == 64u,
              "FlashGqaBk32XchgLaneBf16 must be 64 B");

struct FlashGqaBk32Xchg32Bf16 {
  union {
    float4 f4[2]; // 32 B: half-block S[0..3], S[4..7]
    struct {
      ushort4 p0;   //  8 B (bfloat4 bits)
      ushort4 p1;   //  8 B (bfloat4 bits)
      float2  fac;  //  8 B (slot 0)
      float2  _pad; //  8 B
    } bcast;
  };
};
static_assert(sizeof(FlashGqaBk32Xchg32Bf16) == 32u,
              "FlashGqaBk32Xchg32Bf16 must be 32 B");

template <uint WM>
struct FlashGqaBk32TgStorageBf16;

template <>
struct FlashGqaBk32TgStorageBf16<2u> {
  // WM=2 (256 thr): 12,288 B xchg + 2,048 B d0_s0 = 14,336 B (14.00 KiB EXACT!)
  FlashGqaBk32XchgLaneBf16 xchg[2u][3u][32u];
  float4                   d0_s0[2u][32u][2u];
};

template <>
struct FlashGqaBk32TgStorageBf16<3u> {
  // WM=3 (384 thr): 3 row groups x 4 x 32 B/lane = 12,288 B (12.00 KiB <= 14.00 KiB!)
  FlashGqaBk32Xchg32Bf16 xchg[3u][4u][32u];
};

static_assert(sizeof(FlashGqaBk32TgStorageBf16<2u>) == 14336u &&
                  sizeof(FlashGqaBk32TgStorageBf16<2u>) <=
                      kApple10MaxGqaPrefillTgSramBytes,
              "WP3.2 BF16 WM=2 TgStorage must be exact 14.00 KiB!");
static_assert(sizeof(FlashGqaBk32TgStorageBf16<3u>) == 12288u &&
                  sizeof(FlashGqaBk32TgStorageBf16<3u>) <=
                      kApple10MaxGqaPrefillTgSramBytes,
              "WP3.2 BF16 WM=3 TgStorage must be exact 12.00 KiB (<= 14.00 KiB)!");

struct FlashHd64TgStorage {
  uint tg_q_pos[64u];
};
static_assert(sizeof(FlashHd64TgStorage) <= kApple10MaxPrefillTgSramBytes,
              "HD=64 TgStorage exceeds 16,384 B!");

FLASH_NAX_ALWAYS_INLINE
static inline ushort4 gqa_clean_bf4_u16(ushort4 bits) {
  return select(bits, ushort4(0u),
                (bits & ushort4(0x7fffu)) > ushort4(0x7f80u));
}

FLASH_NAX_ALWAYS_INLINE
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

template <uint BD>
FLASH_NAX_ALWAYS_INLINE
static inline Frag16x16::frag_t<bfloat> load_tg_swizzled_bf16_bk16(
    const threadgroup ushort4 *kv_tile, uint off_c, short2 sc) {
  constexpr uint kRowV4 = BD / 4u;
  constexpr uint kXorMask = (BD == 128u) ? 3u : 7u;
  const uint r0 = uint(sc.y);
  const uint r1 = r0 + 8u;
  const uint col4 = (off_c + uint(sc.x)) >> 2u;
  const uint phys4 = col4 ^ ((r0 & kXorMask) << 2u);
  const bfloat4 v0 = as_type<bfloat4>(kv_tile[r0 * kRowV4 + phys4]);
  const bfloat4 v1 = as_type<bfloat4>(kv_tile[r1 * kRowV4 + phys4]);
  return Frag16x16::frag_t<bfloat>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z,
                                   v1.w);
}

template <uint N_THR, uint BD>
FLASH_NAX_ALWAYS_INLINE
static inline void load_kv_tile_bf16_bk16(
    device const bfloat *base_ptr, ulong phys_row, ulong kv_stride, uint kvh,
    short valid_k, bool fast_kv, uint flat_tid, threadgroup ushort4 *kv_tile) {
  constexpr uint kRowV4 = BD / 4u;
  constexpr uint kXorMask = (BD == 128u) ? 3u : 7u;
  device const bfloat *tile_base =
      base_ptr + phys_row * kv_stride + ulong(kvh) * BD;
  if constexpr (BD == 256u && N_THR == 256u) {
    // 256 threads x 16 bfloats (2 x uint4) = 4,096 bfloats (16 x 256) exact!
    const uint r = flat_tid >> 4u;
    const uint c16 = flat_tid & 15u;
    const uint col4_base = c16 * 4u;
    const uint phys4_base = col4_base ^ ((r & 7u) << 2u);
    threadgroup ushort4 *dst = kv_tile + r * 64u + phys4_base;
    if (fast_kv) {
      device const uint4 *src = reinterpret_cast<device const uint4 *>(
          tile_base + ulong(r) * kv_stride + c16 * 16u);
      {
        const uint4 w0 = src[0];
        dst[0] = as_type<ushort4>(w0.xy);
        dst[1] = as_type<ushort4>(w0.zw);
      }
      {
        const uint4 w1 = src[1];
        dst[2] = as_type<ushort4>(w1.xy);
        dst[3] = as_type<ushort4>(w1.zw);
      }
    } else if (short(r) < valid_k) {
      device const uint4 *src = reinterpret_cast<device const uint4 *>(
          tile_base + ulong(r) * kv_stride + c16 * 16u);
      {
        const uint4 w0 = src[0];
        dst[0] = gqa_clean_bf4_u16(as_type<ushort4>(w0.xy));
        dst[1] = gqa_clean_bf4_u16(as_type<ushort4>(w0.zw));
      }
      {
        const uint4 w1 = src[1];
        dst[2] = gqa_clean_bf4_u16(as_type<ushort4>(w1.xy));
        dst[3] = gqa_clean_bf4_u16(as_type<ushort4>(w1.zw));
      }
    } else {
      dst[0] = ushort4(0u);
      dst[1] = ushort4(0u);
      dst[2] = ushort4(0u);
      dst[3] = ushort4(0u);
    }
  } else if constexpr (BD == 256u && N_THR == 384u) {
    // 512 uint4 (8 bfloats = 2 x ushort4) across 384 threads (1..2 iters/thr).
    for (uint u = flat_tid; u < 512u; u += 384u) {
      const uint r = u >> 5u;
      const uint c8 = u & 31u;
      const uint col4_base = c8 * 2u;
      const uint phys4_base = col4_base ^ ((r & 7u) << 2u);
      threadgroup ushort4 *dst = kv_tile + r * 64u + phys4_base;
      if (fast_kv) {
        const uint4 w = *reinterpret_cast<device const uint4 *>(
            tile_base + ulong(r) * kv_stride + c8 * 8u);
        dst[0] = as_type<ushort4>(w.xy);
        dst[1] = as_type<ushort4>(w.zw);
      } else if (short(r) < valid_k) {
        const uint4 w = *reinterpret_cast<device const uint4 *>(
            tile_base + ulong(r) * kv_stride + c8 * 8u);
        dst[0] = gqa_clean_bf4_u16(as_type<ushort4>(w.xy));
        dst[1] = gqa_clean_bf4_u16(as_type<ushort4>(w.zw));
      } else {
        dst[0] = ushort4(0u);
        dst[1] = ushort4(0u);
      }
    }
  } else {
    // BD == 128u, N_THR == 128u: 128 threads x 16 bfloats (2 x uint4) = 16 x 128!
    const uint r = flat_tid >> 3u;
    const uint c16 = flat_tid & 7u;
    const uint col4_base = c16 * 4u;
    const uint phys4_base = col4_base ^ ((r & kXorMask) << 2u);
    threadgroup ushort4 *dst = kv_tile + r * kRowV4 + phys4_base;
    if (fast_kv) {
      device const uint4 *src = reinterpret_cast<device const uint4 *>(
          tile_base + ulong(r) * kv_stride + c16 * 16u);
      {
        const uint4 w0 = src[0];
        dst[0] = as_type<ushort4>(w0.xy);
        dst[1] = as_type<ushort4>(w0.zw);
      }
      {
        const uint4 w1 = src[1];
        dst[2] = as_type<ushort4>(w1.xy);
        dst[3] = as_type<ushort4>(w1.zw);
      }
    } else if (short(r) < valid_k) {
      device const uint4 *src = reinterpret_cast<device const uint4 *>(
          tile_base + ulong(r) * kv_stride + c16 * 16u);
      {
        const uint4 w0 = src[0];
        dst[0] = gqa_clean_bf4_u16(as_type<ushort4>(w0.xy));
        dst[1] = gqa_clean_bf4_u16(as_type<ushort4>(w0.zw));
      }
      {
        const uint4 w1 = src[1];
        dst[2] = gqa_clean_bf4_u16(as_type<ushort4>(w1.xy));
        dst[3] = gqa_clean_bf4_u16(as_type<ushort4>(w1.zw));
      }
    } else {
      dst[0] = ushort4(0u);
      dst[1] = ushort4(0u);
      dst[2] = ushort4(0u);
      dst[3] = ushort4(0u);
    }
  }
}

template <uint WM, uint BD, bool KeepQInRegs = true>
FLASH_NAX_ALWAYS_INLINE
static inline void flash_bk16_wn_bf16_core(
    device const bfloat *query, device const bfloat *gate,
    device const bfloat *cache_key, device const bfloat *cache_value,
    device bfloat *output, device const uint *page_table,
    constant PagedAttentionParams &p,
    threadgroup FlashGqaBk16TgStorageBf16<WM, BD> &tg_mem, uint kvh,
    uint nk_blocks, uint max_valid_kv, uint kb_min_causal, uint tok0, uint qh0,
    uint qpos0, bool qv0, uint tok1, uint qh1, uint qpos1, bool qv1,
    bool warp_full_rows, ushort lane, ushort sg) {
  constexpr uint BK = 16u;
  constexpr uint WN = BD / 64u; // 4 for HD=256, 2 for HD=128
  constexpr uint N_THR = WM * WN * 32u;

  const uint flat_tid = uint(sg) * 32u + uint(lane);
  const uint row_group = uint(sg) / WN;
  const uint d_part = uint(sg) % WN;
  const uint d_offset = d_part * 64u;
  const short2 sc = Frag16x16::get_coord(lane);
  const short sn = sc.x;

  using BF8 = Frag16x16::frag_t<bfloat>;
  using F8 = Frag16x16::frag_t<float>;

  const uint row0_packed = (tok0 << 8u) | ((qh0 & 0x7Fu) << 1u) | (qv0 ? 1u : 0u);
  const uint row1_packed = (tok1 << 8u) | ((qh1 & 0x7Fu) << 1u) | (qv1 ? 1u : 0u);

  #define BF16_BK16_LOAD_Q(DST, OFF_C)                                         \
    do {                                                                       \
      const bool lqv0 = (row0_packed & 1u) != 0u;                              \
      const bool lqv1 = (row1_packed & 1u) != 0u;                              \
      ushort4 u0 = ushort4(0u);                                                \
      ushort4 u1 = ushort4(0u);                                                \
      if (lqv0) {                                                              \
        const uint lt0 = row0_packed >> 8u;                                    \
        const uint lq0 = (row0_packed >> 1u) & 0x7Fu;                          \
        u0 = *reinterpret_cast<device const ushort4 *>(                        \
            query + ulong(lt0) * p.query_token_stride +                        \
            ulong(lq0) * p.query_head_stride + d_offset + uint(sn) + (OFF_C)); \
        if (!warp_full_rows) {                                                 \
          u0 = gqa_clean_bf4_u16(u0);                                          \
        }                                                                      \
      }                                                                        \
      if (lqv1) {                                                              \
        const uint lt1 = row1_packed >> 8u;                                    \
        const uint lq1 = (row1_packed >> 1u) & 0x7Fu;                          \
        u1 = *reinterpret_cast<device const ushort4 *>(                        \
            query + ulong(lt1) * p.query_token_stride +                        \
            ulong(lq1) * p.query_head_stride + d_offset + uint(sn) + (OFF_C)); \
        if (!warp_full_rows) {                                                 \
          u1 = gqa_clean_bf4_u16(u1);                                          \
        }                                                                      \
      }                                                                        \
      const bfloat4 v0 = as_type<bfloat4>(u0);                                 \
      const bfloat4 v1 = as_type<bfloat4>(u1);                                 \
      (DST) = BF8(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);             \
    } while (false)

  BF8 Q0(bfloat(0.0f)), Q1(bfloat(0.0f)), Q2(bfloat(0.0f)), Q3(bfloat(0.0f));
  if constexpr (KeepQInRegs) {
    BF16_BK16_LOAD_Q(Q0, 0u);
    BF16_BK16_LOAD_Q(Q1, 16u);
    BF16_BK16_LOAD_Q(Q2, 32u);
    BF16_BK16_LOAD_Q(Q3, 48u);
  }

  F8 O0(0.0f), O1(0.0f), O2(0.0f), O3(0.0f);
  float max_s0 = kNegInf;
  float max_s1 = kNegInf;
  float sum_s0 = 0.0f;
  float sum_s1 = 0.0f;
  const float scale2 = rsqrt(float(BD)) * kLog2e;

  const uint kv_stride_u32 = p.key_value_heads * BD;
  const uint blocks_per_page = p.page_tokens / BK;

  #define BF16_BK16_RUN_SOFTMAX_D0(S_VAL, KB_COL, FULL_CF, P_OUT, FAC0_OUT,    \
                                   FAC1_OUT)                                   \
    do {                                                                       \
      float4 s0((S_VAL)[0], (S_VAL)[1], (S_VAL)[2], (S_VAL)[3]);               \
      float4 s1((S_VAL)[4], (S_VAL)[5], (S_VAL)[6], (S_VAL)[7]);               \
      if (FULL_CF) {                                                           \
        s0 *= scale2;                                                          \
        s1 *= scale2;                                                          \
      } else {                                                                 \
        const bool v0 = (row0_packed & 1u) != 0u;                              \
        const bool v1 = (row1_packed & 1u) != 0u;                              \
        const uint4 c_abs =                                                    \
            uint4((KB_COL) + uint(sn)) + uint4(0u, 1u, 2u, 3u);                \
        const bool4 ok0 = v0 && (c_abs < max_valid_kv) && (c_abs <= qpos0);    \
        const bool4 ok1 = v1 && (c_abs < max_valid_kv) && (c_abs <= qpos1);    \
        s0 = select(float4(kNegInf), s0 * scale2, ok0);                        \
        s1 = select(float4(kNegInf), s1 * scale2, ok1);                        \
      }                                                                        \
      float m0 = max(max(s0.x, s0.y), max(s0.z, s0.w));                        \
      m0 = max(m0, simd_shuffle_xor(m0, ushort(1)));                           \
      m0 = max(m0, simd_shuffle_xor(m0, ushort(8)));                           \
      const float new_m0 = max(max_s0, m0);                                    \
      float m1 = max(max(s1.x, s1.y), max(s1.z, s1.w));                        \
      m1 = max(m1, simd_shuffle_xor(m1, ushort(1)));                           \
      m1 = max(m1, simd_shuffle_xor(m1, ushort(8)));                           \
      const float new_m1 = max(max_s1, m1);                                    \
      const float f0 =                                                         \
          (max_s0 <= kNegInfHalf) ? 0.0f : fast::exp2(max_s0 - new_m0);        \
      const float f1 =                                                         \
          (max_s1 <= kNegInfHalf) ? 0.0f : fast::exp2(max_s1 - new_m1);        \
      max_s0 = new_m0;                                                         \
      max_s1 = new_m1;                                                         \
      sum_s0 *= f0;                                                            \
      sum_s1 *= f1;                                                            \
      float4 p0, p1;                                                           \
      if (FULL_CF) {                                                           \
        p0 = fast::exp2(s0 - new_m0);                                          \
        p1 = fast::exp2(s1 - new_m1);                                          \
      } else {                                                                 \
        const bool emp0 = (new_m0 <= kNegInfHalf);                             \
        const bool emp1 = (new_m1 <= kNegInfHalf);                             \
        p0 = select(fast::exp2(s0 - new_m0), float4(0.0f),                     \
                    emp0 || (s0 <= kNegInfHalf));                              \
        p1 = select(fast::exp2(s1 - new_m1), float4(0.0f),                     \
                    emp1 || (s1 <= kNegInfHalf));                              \
      }                                                                        \
      float tsum0 = (p0.x + p0.y) + (p0.z + p0.w);                             \
      tsum0 += simd_shuffle_xor(tsum0, ushort(1));                             \
      tsum0 += simd_shuffle_xor(tsum0, ushort(8));                             \
      sum_s0 += tsum0;                                                         \
      float tsum1 = (p1.x + p1.y) + (p1.z + p1.w);                             \
      tsum1 += simd_shuffle_xor(tsum1, ushort(1));                             \
      tsum1 += simd_shuffle_xor(tsum1, ushort(8));                             \
      sum_s1 += tsum1;                                                         \
      const bfloat4 bp0 = bfloat4(p0);                                         \
      const bfloat4 bp1 = bfloat4(p1);                                         \
      (P_OUT) =                                                                \
          BF8(bp0.x, bp0.y, bp0.z, bp0.w, bp1.x, bp1.y, bp1.z, bp1.w);         \
      (FAC0_OUT) = f0;                                                         \
      (FAC1_OUT) = f1;                                                         \
      tg_mem.xchg[row_group][0][lane].bcast.p0 = as_type<ushort4>(bp0);        \
      tg_mem.xchg[row_group][0][lane].bcast.p1 = as_type<ushort4>(bp1);        \
      tg_mem.xchg[row_group][0][lane].bcast.fac = float2(f0, f1);              \
    } while (false)

  for (uint kb = 0u; kb < nk_blocks; ++kb) {
    const uint lpage = kb / blocks_per_page;
    const uint sub_off = (kb % blocks_per_page) * BK;
    const ulong phys_row = ulong(page_table[lpage]) * p.page_tokens + sub_off;
    const uint kb_col_base = kb * BK;
    const short valid_k = short(min(BK, max_valid_kv - kb_col_base));
    const bool fast_kv = (valid_k == 16) && (kb < kb_min_causal);
    const bool full_causal_free = fast_kv && warp_full_rows;

    // Step 1: Cooperatively load K[16, BD] into XOR-swizzled kv_tile.
    load_kv_tile_bf16_bk16<N_THR, BD>(cache_key, phys_row, ulong(kv_stride_u32),
                                      kvh, valid_k, fast_kv, flat_tid,
                                      tg_mem.kv_tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: QK^T via 2 x nax_mma_16x16x32 (D_SLICE = 64).
    F8 S(0.0f);
    if constexpr (KeepQInRegs) {
      {
        const BF8 K0 = load_tg_swizzled_bf16_bk16<BD>(tg_mem.kv_tile,
                                                      d_offset + 0u, sc);
        const BF8 K1 = load_tg_swizzled_bf16_bk16<BD>(tg_mem.kv_tile,
                                                      d_offset + 16u, sc);
        nax_mma_16x16x32<bfloat, true>(S, Q0, Q1, K0, K1);
      }
      {
        const BF8 K2 = load_tg_swizzled_bf16_bk16<BD>(tg_mem.kv_tile,
                                                      d_offset + 32u, sc);
        const BF8 K3 = load_tg_swizzled_bf16_bk16<BD>(tg_mem.kv_tile,
                                                      d_offset + 48u, sc);
        nax_mma_16x16x32<bfloat, true>(S, Q2, Q3, K2, K3);
      }
    } else {
      {
        BF8 q0_l, q1_l;
        BF16_BK16_LOAD_Q(q0_l, 0u);
        BF16_BK16_LOAD_Q(q1_l, 16u);
        const BF8 K0 = load_tg_swizzled_bf16_bk16<BD>(tg_mem.kv_tile,
                                                      d_offset + 0u, sc);
        const BF8 K1 = load_tg_swizzled_bf16_bk16<BD>(tg_mem.kv_tile,
                                                      d_offset + 16u, sc);
        nax_mma_16x16x32<bfloat, true>(S, q0_l, q1_l, K0, K1);
      }
      {
        BF8 q2_l, q3_l;
        BF16_BK16_LOAD_Q(q2_l, 32u);
        BF16_BK16_LOAD_Q(q3_l, 48u);
        const BF8 K2 = load_tg_swizzled_bf16_bk16<BD>(tg_mem.kv_tile,
                                                      d_offset + 32u, sc);
        const BF8 K3 = load_tg_swizzled_bf16_bk16<BD>(tg_mem.kv_tile,
                                                      d_offset + 48u, sc);
        nax_mma_16x16x32<bfloat, true>(S, q2_l, q3_l, K2, K3);
      }
    }

    BF8 P(bfloat(0.0f));
    float fac0 = 1.0f;
    float fac1 = 1.0f;

    if constexpr (WN == 2u) {
      if (d_part == 1u) {
        tg_mem.xchg[row_group][0][lane].f4[0] = float4(S[0], S[1], S[2], S[3]);
        tg_mem.xchg[row_group][0][lane].f4[1] = float4(S[4], S[5], S[6], S[7]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      load_kv_tile_bf16_bk16<N_THR, BD>(cache_value, phys_row,
                                        ulong(kv_stride_u32), kvh, valid_k,
                                        fast_kv, flat_tid, tg_mem.kv_tile);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (d_part == 0u) {
        const float4 p0 = tg_mem.xchg[row_group][0][lane].f4[0];
        const float4 p1 = tg_mem.xchg[row_group][0][lane].f4[1];
        S += F8(p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w);
        BF16_BK16_RUN_SOFTMAX_D0(S, kb_col_base, full_causal_free, P, fac0,
                                 fac1);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      // Unified 2-slot pairwise tree ((S0+S1) + (S2+S3)) for BOTH WM=2 (12.0 KiB)
      // and WM=3 (14.0 KiB): separates V load from softmax across barrier -> 0 B spill!
      if (d_part == 1u) {
        tg_mem.xchg[row_group][0][lane].f4[0] = float4(S[0], S[1], S[2], S[3]);
        tg_mem.xchg[row_group][0][lane].f4[1] = float4(S[4], S[5], S[6], S[7]);
      } else if (d_part == 3u) {
        tg_mem.xchg[row_group][1][lane].f4[0] = float4(S[0], S[1], S[2], S[3]);
        tg_mem.xchg[row_group][1][lane].f4[1] = float4(S[4], S[5], S[6], S[7]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      load_kv_tile_bf16_bk16<N_THR, BD>(cache_value, phys_row,
                                        ulong(kv_stride_u32), kvh, valid_k,
                                        fast_kv, flat_tid, tg_mem.kv_tile);
      if (d_part == 0u) {
        const float4 s1_0 = tg_mem.xchg[row_group][0][lane].f4[0];
        const float4 s1_1 = tg_mem.xchg[row_group][0][lane].f4[1];
        S += F8(s1_0.x, s1_0.y, s1_0.z, s1_0.w, s1_1.x, s1_1.y, s1_1.z, s1_1.w);
      } else if (d_part == 2u) {
        const float4 s3_0 = tg_mem.xchg[row_group][1][lane].f4[0];
        const float4 s3_1 = tg_mem.xchg[row_group][1][lane].f4[1];
        const float4 s23_0 = float4(S[0], S[1], S[2], S[3]) + s3_0;
        const float4 s23_1 = float4(S[4], S[5], S[6], S[7]) + s3_1;
        tg_mem.xchg[row_group][1][lane].f4[0] = s23_0;
        tg_mem.xchg[row_group][1][lane].f4[1] = s23_1;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (d_part == 0u) {
        const float4 s23_0 = tg_mem.xchg[row_group][1][lane].f4[0];
        const float4 s23_1 = tg_mem.xchg[row_group][1][lane].f4[1];
        S += F8(s23_0.x, s23_0.y, s23_0.z, s23_0.w, s23_1.x, s23_1.y, s23_1.z,
                s23_1.w);
        BF16_BK16_RUN_SOFTMAX_D0(S, kb_col_base, full_causal_free, P, fac0,
                                 fac1);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (d_part != 0u) {
      const bfloat4 bp0 =
          as_type<bfloat4>(tg_mem.xchg[row_group][0][lane].bcast.p0);
      const bfloat4 bp1 =
          as_type<bfloat4>(tg_mem.xchg[row_group][0][lane].bcast.p1);
      const float2 fc = tg_mem.xchg[row_group][0][lane].bcast.fac;
      P = BF8(bp0.x, bp0.y, bp0.z, bp0.w, bp1.x, bp1.y, bp1.z, bp1.w);
      fac0 = fc.x;
      fac1 = fc.y;
    }

    const F8 fac_vec(fac0, fac0, fac0, fac0, fac1, fac1, fac1, fac1);
    O0 *= fac_vec;
    O1 *= fac_vec;
    O2 *= fac_vec;
    O3 *= fac_vec;

    // Step 4: PV via 2 x nax_mma_16x32x16_nn (D_SLICE = 64).
    {
      const BF8 V0 = load_tg_swizzled_bf16_bk16<BD>(tg_mem.kv_tile,
                                                    d_offset + 0u, sc);
      const BF8 V1 = load_tg_swizzled_bf16_bk16<BD>(tg_mem.kv_tile,
                                                    d_offset + 16u, sc);
      nax_mma_16x32x16_nn<bfloat>(O0, O1, P, V0, V1);
    }
    {
      const BF8 V2 = load_tg_swizzled_bf16_bk16<BD>(tg_mem.kv_tile,
                                                    d_offset + 32u, sc);
      const BF8 V3 = load_tg_swizzled_bf16_bk16<BD>(tg_mem.kv_tile,
                                                    d_offset + 48u, sc);
      nax_mma_16x32x16_nn<bfloat>(O2, O3, P, V2, V3);
    }

    if (kb + 1u < nk_blocks) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  #undef BF16_BK16_LOAD_Q
  #undef BF16_BK16_RUN_SOFTMAX_D0

  // Broadcast final (rcp0, rcp1) from d_part == 0 to d_part != 0.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float rcp0 = 0.0f;
  float rcp1 = 0.0f;
  if (d_part == 0u) {
    rcp0 = (sum_s0 > 0.0f) ? (1.0f / sum_s0) : 0.0f;
    rcp1 = (sum_s1 > 0.0f) ? (1.0f / sum_s1) : 0.0f;
    tg_mem.xchg[row_group][0][lane].bcast.fac = float2(rcp0, rcp1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (d_part != 0u) {
    const float2 r = tg_mem.xchg[row_group][0][lane].bcast.fac;
    rcp0 = r.x;
    rcp1 = r.y;
  }

  const bool ep_qv0 = (row0_packed & 1u) != 0u;
  const bool ep_qv1 = (row1_packed & 1u) != 0u;
  if (!ep_qv0 && !ep_qv1) {
    return;
  }
  const uint ep_tok0 = row0_packed >> 8u;
  const uint ep_qh0 = (row0_packed >> 1u) & 0x7Fu;
  const uint ep_tok1 = row1_packed >> 8u;
  const uint ep_qh1 = (row1_packed >> 1u) & 0x7Fu;

  device const bfloat *g_row0 =
      gate + ulong(ep_tok0) * p.gate_token_stride +
      ulong(ep_qh0) * p.gate_head_stride + d_offset + uint(sn);
  device const bfloat *g_row1 =
      gate + ulong(ep_tok1) * p.gate_token_stride +
      ulong(ep_qh1) * p.gate_head_stride + d_offset + uint(sn);
  device bfloat *o_row0 =
      output + (ulong(ep_tok0) * p.query_heads + ep_qh0) * BD + d_offset +
      uint(sn);
  device bfloat *o_row1 =
      output + (ulong(ep_tok1) * p.query_heads + ep_qh1) * BD + d_offset +
      uint(sn);

  #define BF16_BK16_STORE_O(OREG, OFF_C)                                       \
    do {                                                                       \
      if (ep_qv0) {                                                            \
        float4 o0_f =                                                          \
            float4((OREG)[0], (OREG)[1], (OREG)[2], (OREG)[3]) * rcp0;         \
        if (p.gate_head_stride != 0u) {                                        \
          ushort4 gu0 =                                                        \
              *reinterpret_cast<device const ushort4 *>(g_row0 + (OFF_C));     \
          if (!warp_full_rows) {                                               \
            gu0 = gqa_clean_bf4_u16(gu0);                                      \
          }                                                                    \
          const float4 g0_f = float4(as_type<bfloat4>(gu0));                   \
          o0_f /= (float4(1.0f) + exp(-g0_f));                                 \
        }                                                                      \
        *reinterpret_cast<device bfloat4 *>(o_row0 + (OFF_C)) = bfloat4(o0_f); \
      }                                                                        \
      if (ep_qv1) {                                                            \
        float4 o1_f =                                                          \
            float4((OREG)[4], (OREG)[5], (OREG)[6], (OREG)[7]) * rcp1;         \
        if (p.gate_head_stride != 0u) {                                        \
          ushort4 gu1 =                                                        \
              *reinterpret_cast<device const ushort4 *>(g_row1 + (OFF_C));     \
          if (!warp_full_rows) {                                               \
            gu1 = gqa_clean_bf4_u16(gu1);                                      \
          }                                                                    \
          const float4 g1_f = float4(as_type<bfloat4>(gu1));                   \
          o1_f /= (float4(1.0f) + exp(-g1_f));                                 \
        }                                                                      \
        *reinterpret_cast<device bfloat4 *>(o_row1 + (OFF_C)) = bfloat4(o1_f); \
      }                                                                        \
    } while (false)
  BF16_BK16_STORE_O(O0, 0u);
  BF16_BK16_STORE_O(O1, 16u);
  BF16_BK16_STORE_O(O2, 32u);
  BF16_BK16_STORE_O(O3, 48u);
  #undef BF16_BK16_STORE_O
}

FLASH_NAX_ALWAYS_INLINE
static inline Frag16x16::frag_t<bfloat> load_bf16_ptr_bf8(
    device const ushort4 *p0, device const ushort4 *p1,
    bool ok0, bool ok1, bool fast_kv, uint col4_off) {
  ushort4 u0 = ok0 ? p0[col4_off] : ushort4(0u);
  ushort4 u1 = ok1 ? p1[col4_off] : ushort4(0u);
  if (!fast_kv) {
    u0 = gqa_clean_bf4_u16(u0);
    u1 = gqa_clean_bf4_u16(u1);
  }
  const bfloat4 v0 = as_type<bfloat4>(u0);
  const bfloat4 v1 = as_type<bfloat4>(u1);
  return Frag16x16::frag_t<bfloat>(v0.x, v0.y, v0.z, v0.w,
                                   v1.x, v1.y, v1.z, v1.w);
}

template <uint WM, bool KeepQInRegs = true>
FLASH_NAX_ALWAYS_INLINE
static inline void flash_bk32_wn4_bf16_hd256_core(
    device const bfloat *query, device const bfloat *gate,
    device const bfloat *cache_key, device const bfloat *cache_value,
    device bfloat *output, device const uint *page_table,
    constant PagedAttentionParams &p,
    threadgroup FlashGqaBk32TgStorageBf16<WM> &tg_mem, uint kvh,
    uint nk_blocks, uint max_valid_kv, uint kb_min_causal, uint tok0, uint qh0,
    uint qpos0, bool qv0, uint tok1, uint qh1, uint qpos1, bool qv1,
    bool warp_full_rows, ushort lane, ushort sg) {
  constexpr uint BD = 256u;
  constexpr uint BK = 32u;

  const uint row_group = uint(sg) >> 2u;
  const uint d_part = uint(sg) & 3u;
  const uint d_offset = d_part * 64u;
  const short2 sc = Frag16x16::get_coord(lane);
  const short sm = sc.y;
  const short sn = sc.x;

  using BF8 = Frag16x16::frag_t<bfloat>;
  using F8 = Frag16x16::frag_t<float>;

  const uint row0_p = (tok0 << 8u) | ((qh0 & 0x7Fu) << 1u) | (qv0 ? 1u : 0u);
  const uint row1_p = (tok1 << 8u) | ((qh1 & 0x7Fu) << 1u) | (qv1 ? 1u : 0u);

  #define BF16_BK32_LOAD_Q(DST, OFF_C)                                         \
    do {                                                                       \
      ushort4 u0 = ushort4(0u), u1 = ushort4(0u);                              \
      if (row0_p & 1u) {                                                       \
        u0 = *reinterpret_cast<device const ushort4 *>(                        \
            query + ulong(row0_p >> 8u) * p.query_token_stride +               \
            ulong((row0_p >> 1u) & 0x7Fu) * p.query_head_stride + d_offset +   \
            uint(sn) + (OFF_C));                                               \
        if (!warp_full_rows) u0 = gqa_clean_bf4_u16(u0);                       \
      }                                                                        \
      if (row1_p & 1u) {                                                       \
        u1 = *reinterpret_cast<device const ushort4 *>(                        \
            query + ulong(row1_p >> 8u) * p.query_token_stride +               \
            ulong((row1_p >> 1u) & 0x7Fu) * p.query_head_stride + d_offset +   \
            uint(sn) + (OFF_C));                                               \
        if (!warp_full_rows) u1 = gqa_clean_bf4_u16(u1);                       \
      }                                                                        \
      const bfloat4 v0 = as_type<bfloat4>(u0);                                 \
      const bfloat4 v1 = as_type<bfloat4>(u1);                                 \
      (DST) = BF8(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);             \
    } while (false)

  BF8 Q0(bfloat(0.0f)), Q1(bfloat(0.0f)), Q2(bfloat(0.0f)), Q3(bfloat(0.0f));
  if constexpr (KeepQInRegs) {
    BF16_BK32_LOAD_Q(Q0, 0u);
    BF16_BK32_LOAD_Q(Q1, 16u);
    BF16_BK32_LOAD_Q(Q2, 32u);
    BF16_BK32_LOAD_Q(Q3, 48u);
  }

  const uint kv_stride_u32 = p.key_value_heads * BD;
  const uint kv_col_off = kvh * BD + d_offset + uint(sn);
  const uint pt_shift = ctz(p.page_tokens >> 5u);
  const uint pt_mask = (p.page_tokens >> 5u) - 1u;

  F8 O0(0.0f), O1(0.0f), O2(0.0f), O3(0.0f);
  float max_s0 = kNegInf, max_s1 = kNegInf, sum_s0 = 0.0f, sum_s1 = 0.0f;
  constexpr float kScale2 = 0.0625f * kLog2e;

  for (uint kb = 0u; kb < nk_blocks; ++kb) {
    const uint lpage = kb >> pt_shift;
    const uint sub_off = (kb & pt_mask) * BK;
    const uint phys_page = page_table[lpage];
    const uint kb_col_base = kb * BK;
    const short valid_k = short(min(BK, max_valid_kv - kb_col_base));
    const bool fast_kv = (kb < kb_min_causal) && (valid_k == 32);
    const bool full_cf = fast_kv && warp_full_rows;

    if constexpr (WM == 2u) {
      F8 S;
      #pragma clang loop unroll(disable)
      for (uint h2 = 0u; h2 < 2u; ++h2) {
        const uint row_off = h2 * 16u;
        const bool ok0 = (short(int(row_off) + int(sm)) < valid_k);
        const bool ok1 = (short(int(row_off) + int(sm) + 8) < valid_k);
        const ulong base_row = ulong(phys_page) * p.page_tokens + sub_off;
        device const ushort4 *p0 = reinterpret_cast<device const ushort4 *>(
            cache_key +
            (base_row + ulong(ok0 ? (row_off + uint(sm)) : 0u)) *
                ulong(kv_stride_u32) +
            kv_col_off);
        device const ushort4 *p1 = reinterpret_cast<device const ushort4 *>(
            cache_key +
            (base_row + ulong(ok1 ? (row_off + 8u + uint(sm)) : 0u)) *
                ulong(kv_stride_u32) +
            kv_col_off);
        S = F8(0.0f);
        if constexpr (KeepQInRegs) {
          {
            const BF8 k0 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 0u);
            const BF8 k1 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 4u);
            nax_mma_16x16x32<bfloat, true>(S, Q0, Q1, k0, k1);
          }
          simdgroup_barrier(mem_flags::mem_none);
          {
            const BF8 k2 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 8u);
            const BF8 k3 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 12u);
            nax_mma_16x16x32<bfloat, true>(S, Q2, Q3, k2, k3);
          }
        } else {
          {
            BF8 q0_l, q1_l;
            BF16_BK32_LOAD_Q(q0_l, 0u);
            BF16_BK32_LOAD_Q(q1_l, 16u);
            const BF8 k0 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 0u);
            const BF8 k1 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 4u);
            nax_mma_16x16x32<bfloat, true>(S, q0_l, q1_l, k0, k1);
          }
          simdgroup_barrier(mem_flags::mem_none);
          {
            BF8 q2_l, q3_l;
            BF16_BK32_LOAD_Q(q2_l, 32u);
            BF16_BK32_LOAD_Q(q3_l, 48u);
            const BF8 k2 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 8u);
            const BF8 k3 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 12u);
            nax_mma_16x16x32<bfloat, true>(S, q2_l, q3_l, k2, k3);
          }
        }
        if (d_part == 0u) {
          if (h2 == 0u) {
            tg_mem.d0_s0[row_group][lane][0] = float4(S[0], S[1], S[2], S[3]);
            tg_mem.d0_s0[row_group][lane][1] = float4(S[4], S[5], S[6], S[7]);
          }
        } else {
          tg_mem.xchg[row_group][d_part - 1u][lane].f4[h2 * 2u + 0u] =
              float4(S[0], S[1], S[2], S[3]);
          tg_mem.xchg[row_group][d_part - 1u][lane].f4[h2 * 2u + 1u] =
              float4(S[4], S[5], S[6], S[7]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (d_part == 0u) {
        float m0_b, m1_b;
        {
          float4 s10 =
              (float4(S[0], S[1], S[2], S[3]) +
               tg_mem.xchg[row_group][0][lane].f4[2]) +
              (tg_mem.xchg[row_group][1][lane].f4[2] +
               tg_mem.xchg[row_group][2][lane].f4[2]);
          if (full_cf) {
            s10 *= kScale2;
          } else {
            const bool v0 = (row0_p & 1u) != 0u;
            const uint4 c1 =
                uint4(kb_col_base + 16u + uint(sn)) + uint4(0u, 1u, 2u, 3u);
            s10 = select(float4(kNegInf), s10 * kScale2,
                         v0 && (c1 < max_valid_kv) && (c1 <= qpos0));
          }
          tg_mem.xchg[row_group][1][lane].f4[2] = s10;
          m0_b = max(max(s10.x, s10.y), max(s10.z, s10.w));
        }
        {
          float4 s11 =
              (float4(S[4], S[5], S[6], S[7]) +
               tg_mem.xchg[row_group][0][lane].f4[3]) +
              (tg_mem.xchg[row_group][1][lane].f4[3] +
               tg_mem.xchg[row_group][2][lane].f4[3]);
          if (full_cf) {
            s11 *= kScale2;
          } else {
            const bool v1 = (row1_p & 1u) != 0u;
            const uint4 c1 =
                uint4(kb_col_base + 16u + uint(sn)) + uint4(0u, 1u, 2u, 3u);
            s11 = select(float4(kNegInf), s11 * kScale2,
                         v1 && (c1 < max_valid_kv) && (c1 <= qpos1));
          }
          tg_mem.xchg[row_group][1][lane].f4[3] = s11;
          m1_b = max(max(s11.x, s11.y), max(s11.z, s11.w));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float fac0, fac1;
        {
          float4 s00 =
              (tg_mem.d0_s0[row_group][lane][0] +
               tg_mem.xchg[row_group][0][lane].f4[0]) +
              (tg_mem.xchg[row_group][1][lane].f4[0] +
               tg_mem.xchg[row_group][2][lane].f4[0]);
          if (full_cf) {
            s00 *= kScale2;
          } else {
            const bool v0 = (row0_p & 1u) != 0u;
            const uint4 c0 =
                uint4(kb_col_base + uint(sn)) + uint4(0u, 1u, 2u, 3u);
            s00 = select(float4(kNegInf), s00 * kScale2,
                         v0 && (c0 < max_valid_kv) && (c0 <= qpos0));
          }
          float m0 = max(m0_b, max(max(s00.x, s00.y), max(s00.z, s00.w)));
          m0 = max(m0, simd_shuffle_xor(m0, ushort(1)));
          m0 = max(m0, simd_shuffle_xor(m0, ushort(8)));
          const float nm0 = max(max_s0, m0);
          fac0 = (max_s0 <= kNegInfHalf) ? 0.0f : fast::exp2(max_s0 - nm0);
          max_s0 = nm0;
          sum_s0 *= fac0;
          float4 p00 = full_cf
                           ? fast::exp2(s00 - nm0)
                           : select(fast::exp2(s00 - nm0), float4(0.0f),
                                    (nm0 <= kNegInfHalf) || (s00 <= kNegInfHalf));
          float ts0 = (p00.x + p00.y) + (p00.z + p00.w);
          ts0 += simd_shuffle_xor(ts0, ushort(1));
          ts0 += simd_shuffle_xor(ts0, ushort(8));
          sum_s0 += ts0;
          tg_mem.xchg[row_group][0][lane].bcast.p00 =
              as_type<ushort4>(bfloat4(p00));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
          float4 s01 =
              (tg_mem.d0_s0[row_group][lane][1] +
               tg_mem.xchg[row_group][0][lane].f4[1]) +
              (tg_mem.xchg[row_group][1][lane].f4[1] +
               tg_mem.xchg[row_group][2][lane].f4[1]);
          if (full_cf) {
            s01 *= kScale2;
          } else {
            const bool v1 = (row1_p & 1u) != 0u;
            const uint4 c0 =
                uint4(kb_col_base + uint(sn)) + uint4(0u, 1u, 2u, 3u);
            s01 = select(float4(kNegInf), s01 * kScale2,
                         v1 && (c0 < max_valid_kv) && (c0 <= qpos1));
          }
          float m1 = max(m1_b, max(max(s01.x, s01.y), max(s01.z, s01.w)));
          m1 = max(m1, simd_shuffle_xor(m1, ushort(1)));
          m1 = max(m1, simd_shuffle_xor(m1, ushort(8)));
          const float nm1 = max(max_s1, m1);
          fac1 = (max_s1 <= kNegInfHalf) ? 0.0f : fast::exp2(max_s1 - nm1);
          max_s1 = nm1;
          sum_s1 *= fac1;
          float4 p01 = full_cf
                           ? fast::exp2(s01 - nm1)
                           : select(fast::exp2(s01 - nm1), float4(0.0f),
                                    (nm1 <= kNegInfHalf) || (s01 <= kNegInfHalf));
          float ts1 = (p01.x + p01.y) + (p01.z + p01.w);
          ts1 += simd_shuffle_xor(ts1, ushort(1));
          ts1 += simd_shuffle_xor(ts1, ushort(8));
          sum_s1 += ts1;
          tg_mem.xchg[row_group][0][lane].bcast.p01 =
              as_type<ushort4>(bfloat4(p01));
          tg_mem.xchg[row_group][0][lane].bcast.fac = float2(fac0, fac1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
          const float4 s10 = tg_mem.xchg[row_group][1][lane].f4[2];
          float4 p10 = full_cf
                           ? fast::exp2(s10 - max_s0)
                           : select(fast::exp2(s10 - max_s0), float4(0.0f),
                                    (max_s0 <= kNegInfHalf) || (s10 <= kNegInfHalf));
          float ts0 = (p10.x + p10.y) + (p10.z + p10.w);
          ts0 += simd_shuffle_xor(ts0, ushort(1));
          ts0 += simd_shuffle_xor(ts0, ushort(8));
          sum_s0 += ts0;
          tg_mem.xchg[row_group][0][lane].bcast.p10 =
              as_type<ushort4>(bfloat4(p10));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
          const float4 s11 = tg_mem.xchg[row_group][1][lane].f4[3];
          float4 p11 = full_cf
                           ? fast::exp2(s11 - max_s1)
                           : select(fast::exp2(s11 - max_s1), float4(0.0f),
                                    (max_s1 <= kNegInfHalf) || (s11 <= kNegInfHalf));
          float ts1 = (p11.x + p11.y) + (p11.z + p11.w);
          ts1 += simd_shuffle_xor(ts1, ushort(1));
          ts1 += simd_shuffle_xor(ts1, ushort(8));
          sum_s1 += ts1;
          tg_mem.xchg[row_group][0][lane].bcast.p11 =
              as_type<ushort4>(bfloat4(p11));
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      float m0_a = kNegInf, m1_a = kNegInf;
      F8 S;
      #pragma clang loop unroll(disable)
      for (uint h2 = 0u; h2 < 2u; ++h2) {
        const uint row_off = h2 * 16u;
        const bool ok0 = (short(int(row_off) + int(sm)) < valid_k);
        const bool ok1 = (short(int(row_off) + int(sm) + 8) < valid_k);
        const ulong base_row = ulong(phys_page) * p.page_tokens + sub_off;
        device const ushort4 *p0 = reinterpret_cast<device const ushort4 *>(
            cache_key +
            (base_row + ulong(ok0 ? (row_off + uint(sm)) : 0u)) *
                ulong(kv_stride_u32) +
            kv_col_off);
        device const ushort4 *p1 = reinterpret_cast<device const ushort4 *>(
            cache_key +
            (base_row + ulong(ok1 ? (row_off + 8u + uint(sm)) : 0u)) *
                ulong(kv_stride_u32) +
            kv_col_off);
        S = F8(0.0f);
        {
          const BF8 k0 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 0u);
          const BF8 k1 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 4u);
          nax_mma_16x16x32<bfloat, true>(S, Q0, Q1, k0, k1);
        }
        simdgroup_barrier(mem_flags::mem_none);
        {
          const BF8 k2 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 8u);
          const BF8 k3 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 12u);
          nax_mma_16x16x32<bfloat, true>(S, Q2, Q3, k2, k3);
        }
        if (d_part > 0u) {
          tg_mem.xchg[row_group][d_part - 1u][lane].f4[0] =
              float4(S[0], S[1], S[2], S[3]);
          tg_mem.xchg[row_group][d_part - 1u][lane].f4[1] =
              float4(S[4], S[5], S[6], S[7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (h2 == 0u) {
          if (d_part == 0u) {
            {
              float4 s00 =
                  (float4(S[0], S[1], S[2], S[3]) +
                   tg_mem.xchg[row_group][0][lane].f4[0]) +
                  (tg_mem.xchg[row_group][1][lane].f4[0] +
                   tg_mem.xchg[row_group][2][lane].f4[0]);
              if (full_cf) {
                s00 *= kScale2;
              } else {
                const bool v0 = (row0_p & 1u) != 0u;
                const uint4 c0 =
                    uint4(kb_col_base + uint(sn)) + uint4(0u, 1u, 2u, 3u);
                s00 = select(float4(kNegInf), s00 * kScale2,
                             v0 && (c0 < max_valid_kv) && (c0 <= qpos0));
              }
              tg_mem.xchg[row_group][3][lane].f4[0] = s00;
              m0_a = max(max(s00.x, s00.y), max(s00.z, s00.w));
            }
            {
              float4 s01 =
                  (float4(S[4], S[5], S[6], S[7]) +
                   tg_mem.xchg[row_group][0][lane].f4[1]) +
                  (tg_mem.xchg[row_group][1][lane].f4[1] +
                   tg_mem.xchg[row_group][2][lane].f4[1]);
              if (full_cf) {
                s01 *= kScale2;
              } else {
                const bool v1 = (row1_p & 1u) != 0u;
                const uint4 c0 =
                    uint4(kb_col_base + uint(sn)) + uint4(0u, 1u, 2u, 3u);
                s01 = select(float4(kNegInf), s01 * kScale2,
                             v1 && (c0 < max_valid_kv) && (c0 <= qpos1));
              }
              tg_mem.xchg[row_group][3][lane].f4[1] = s01;
              m1_a = max(max(s01.x, s01.y), max(s01.z, s01.w));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }

      if (d_part == 0u) {
        float m0_b, m1_b;
        {
          float4 s10 =
              (float4(S[0], S[1], S[2], S[3]) +
               tg_mem.xchg[row_group][0][lane].f4[0]) +
              (tg_mem.xchg[row_group][1][lane].f4[0] +
               tg_mem.xchg[row_group][2][lane].f4[0]);
          if (full_cf) {
            s10 *= kScale2;
          } else {
            const bool v0 = (row0_p & 1u) != 0u;
            const uint4 c1 =
                uint4(kb_col_base + 16u + uint(sn)) + uint4(0u, 1u, 2u, 3u);
            s10 = select(float4(kNegInf), s10 * kScale2,
                         v0 && (c1 < max_valid_kv) && (c1 <= qpos0));
          }
          tg_mem.xchg[row_group][2][lane].f4[0] = s10;
          m0_b = max(max(s10.x, s10.y), max(s10.z, s10.w));
        }
        {
          float4 s11 =
              (float4(S[4], S[5], S[6], S[7]) +
               tg_mem.xchg[row_group][0][lane].f4[1]) +
              (tg_mem.xchg[row_group][1][lane].f4[1] +
               tg_mem.xchg[row_group][2][lane].f4[1]);
          if (full_cf) {
            s11 *= kScale2;
          } else {
            const bool v1 = (row1_p & 1u) != 0u;
            const uint4 c1 =
                uint4(kb_col_base + 16u + uint(sn)) + uint4(0u, 1u, 2u, 3u);
            s11 = select(float4(kNegInf), s11 * kScale2,
                         v1 && (c1 < max_valid_kv) && (c1 <= qpos1));
          }
          tg_mem.xchg[row_group][2][lane].f4[1] = s11;
          m1_b = max(max(s11.x, s11.y), max(s11.z, s11.w));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float fac0, fac1;
        {
          const float4 s00 = tg_mem.xchg[row_group][3][lane].f4[0];
          float m0 = max(m0_b, m0_a);
          m0 = max(m0, simd_shuffle_xor(m0, ushort(1)));
          m0 = max(m0, simd_shuffle_xor(m0, ushort(8)));
          const float nm0 = max(max_s0, m0);
          fac0 = (max_s0 <= kNegInfHalf) ? 0.0f : fast::exp2(max_s0 - nm0);
          max_s0 = nm0;
          sum_s0 *= fac0;
          float4 p00 = full_cf
                           ? fast::exp2(s00 - nm0)
                           : select(fast::exp2(s00 - nm0), float4(0.0f),
                                    (nm0 <= kNegInfHalf) || (s00 <= kNegInfHalf));
          float ts0 = (p00.x + p00.y) + (p00.z + p00.w);
          ts0 += simd_shuffle_xor(ts0, ushort(1));
          ts0 += simd_shuffle_xor(ts0, ushort(8));
          sum_s0 += ts0;
          tg_mem.xchg[row_group][0][lane].bcast.p0 =
              as_type<ushort4>(bfloat4(p00));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
          const float4 s01 = tg_mem.xchg[row_group][3][lane].f4[1];
          float m1 = max(m1_b, m1_a);
          m1 = max(m1, simd_shuffle_xor(m1, ushort(1)));
          m1 = max(m1, simd_shuffle_xor(m1, ushort(8)));
          const float nm1 = max(max_s1, m1);
          fac1 = (max_s1 <= kNegInfHalf) ? 0.0f : fast::exp2(max_s1 - nm1);
          max_s1 = nm1;
          sum_s1 *= fac1;
          float4 p01 = full_cf
                           ? fast::exp2(s01 - nm1)
                           : select(fast::exp2(s01 - nm1), float4(0.0f),
                                    (nm1 <= kNegInfHalf) || (s01 <= kNegInfHalf));
          float ts1 = (p01.x + p01.y) + (p01.z + p01.w);
          ts1 += simd_shuffle_xor(ts1, ushort(1));
          ts1 += simd_shuffle_xor(ts1, ushort(8));
          sum_s1 += ts1;
          tg_mem.xchg[row_group][0][lane].bcast.p1 =
              as_type<ushort4>(bfloat4(p01));
          tg_mem.xchg[row_group][0][lane].bcast.fac = float2(fac0, fac1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
          const float4 s10 = tg_mem.xchg[row_group][2][lane].f4[0];
          float4 p10 = full_cf
                           ? fast::exp2(s10 - max_s0)
                           : select(fast::exp2(s10 - max_s0), float4(0.0f),
                                    (max_s0 <= kNegInfHalf) || (s10 <= kNegInfHalf));
          float ts0 = (p10.x + p10.y) + (p10.z + p10.w);
          ts0 += simd_shuffle_xor(ts0, ushort(1));
          ts0 += simd_shuffle_xor(ts0, ushort(8));
          sum_s0 += ts0;
          tg_mem.xchg[row_group][1][lane].bcast.p0 =
              as_type<ushort4>(bfloat4(p10));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
          const float4 s11 = tg_mem.xchg[row_group][2][lane].f4[1];
          float4 p11 = full_cf
                           ? fast::exp2(s11 - max_s1)
                           : select(fast::exp2(s11 - max_s1), float4(0.0f),
                                    (max_s1 <= kNegInfHalf) || (s11 <= kNegInfHalf));
          float ts1 = (p11.x + p11.y) + (p11.z + p11.w);
          ts1 += simd_shuffle_xor(ts1, ushort(1));
          ts1 += simd_shuffle_xor(ts1, ushort(8));
          sum_s1 += ts1;
          tg_mem.xchg[row_group][1][lane].bcast.p1 =
              as_type<ushort4>(bfloat4(p11));
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
      const float2 fc = tg_mem.xchg[row_group][0][lane].bcast.fac;
      const F8 fv(fc.x, fc.x, fc.x, fc.x, fc.y, fc.y, fc.y, fc.y);
      O0 *= fv;
      O1 *= fv;
      O2 *= fv;
      O3 *= fv;

      #pragma clang loop unroll(disable)
      for (uint h2 = 0u; h2 < 2u; ++h2) {
        ushort4 up0, up1;
        if constexpr (WM == 2u) {
          up0 = (h2 == 0u) ? tg_mem.xchg[row_group][0][lane].bcast.p00
                           : tg_mem.xchg[row_group][0][lane].bcast.p10;
          up1 = (h2 == 0u) ? tg_mem.xchg[row_group][0][lane].bcast.p01
                           : tg_mem.xchg[row_group][0][lane].bcast.p11;
        } else {
          up0 = (h2 == 0u) ? tg_mem.xchg[row_group][0][lane].bcast.p0
                           : tg_mem.xchg[row_group][1][lane].bcast.p0;
          up1 = (h2 == 0u) ? tg_mem.xchg[row_group][0][lane].bcast.p1
                           : tg_mem.xchg[row_group][1][lane].bcast.p1;
        }
        const bfloat4 bp0 = as_type<bfloat4>(up0);
        const bfloat4 bp1 = as_type<bfloat4>(up1);
        const BF8 P_cur(bp0.x, bp0.y, bp0.z, bp0.w, bp1.x, bp1.y, bp1.z, bp1.w);
        const uint row_off = h2 * 16u;
        const bool ok0 = (short(int(row_off) + int(sm)) < valid_k);
        const bool ok1 = (short(int(row_off) + int(sm) + 8) < valid_k);
        const ulong base_row = ulong(phys_page) * p.page_tokens + sub_off;
        device const ushort4 *p0 = reinterpret_cast<device const ushort4 *>(
            cache_value +
            (base_row + ulong(ok0 ? (row_off + uint(sm)) : 0u)) *
                ulong(kv_stride_u32) +
            kv_col_off);
        device const ushort4 *p1 = reinterpret_cast<device const ushort4 *>(
            cache_value +
            (base_row + ulong(ok1 ? (row_off + 8u + uint(sm)) : 0u)) *
                ulong(kv_stride_u32) +
            kv_col_off);
        {
          const BF8 v0 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 0u);
          const BF8 v1 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 4u);
          nax_mma_16x32x16_nn<bfloat>(O0, O1, P_cur, v0, v1);
        }
        simdgroup_barrier(mem_flags::mem_none);
        {
          const BF8 v2 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 8u);
          const BF8 v3 = load_bf16_ptr_bf8(p0, p1, ok0, ok1, fast_kv, 12u);
          nax_mma_16x32x16_nn<bfloat>(O2, O3, P_cur, v2, v3);
        }
      }
    }
    if (kb + 1u < nk_blocks) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  #undef BF16_BK32_LOAD_Q

  threadgroup_barrier(mem_flags::mem_threadgroup);
  float rcp0 = 0.0f;
  float rcp1 = 0.0f;
  if (d_part == 0u) {
    rcp0 = (sum_s0 > 0.0f) ? (1.0f / sum_s0) : 0.0f;
    rcp1 = (sum_s1 > 0.0f) ? (1.0f / sum_s1) : 0.0f;
    tg_mem.xchg[row_group][0][lane].bcast.fac = float2(rcp0, rcp1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (d_part != 0u) {
    const float2 r = tg_mem.xchg[row_group][0][lane].bcast.fac;
    rcp0 = r.x;
    rcp1 = r.y;
  }

  const bool ep_qv0 = (row0_p & 1u) != 0u;
  const bool ep_qv1 = (row1_p & 1u) != 0u;
  if (!ep_qv0 && !ep_qv1) {
    return;
  }
  const uint ep_tok0 = row0_p >> 8u;
  const uint ep_qh0 = (row0_p >> 1u) & 0x7Fu;
  const uint ep_tok1 = row1_p >> 8u;
  const uint ep_qh1 = (row1_p >> 1u) & 0x7Fu;

  device const bfloat *g_row0 =
      gate + ulong(ep_tok0) * p.gate_token_stride +
      ulong(ep_qh0) * p.gate_head_stride + d_offset + uint(sn);
  device const bfloat *g_row1 =
      gate + ulong(ep_tok1) * p.gate_token_stride +
      ulong(ep_qh1) * p.gate_head_stride + d_offset + uint(sn);
  device bfloat *o_row0 =
      output + (ulong(ep_tok0) * p.query_heads + ep_qh0) * BD + d_offset +
      uint(sn);
  device bfloat *o_row1 =
      output + (ulong(ep_tok1) * p.query_heads + ep_qh1) * BD + d_offset +
      uint(sn);

  #define BF16_BK32_STORE_O(OREG, OFF_C)                                       \
    do {                                                                       \
      if (ep_qv0) {                                                            \
        float4 o0_f =                                                          \
            float4((OREG)[0], (OREG)[1], (OREG)[2], (OREG)[3]) * rcp0;         \
        if (p.gate_head_stride != 0u) {                                        \
          ushort4 gu0 =                                                        \
              *reinterpret_cast<device const ushort4 *>(g_row0 + (OFF_C));     \
          if (!warp_full_rows) {                                               \
            gu0 = gqa_clean_bf4_u16(gu0);                                      \
          }                                                                    \
          const float4 g0_f = float4(as_type<bfloat4>(gu0));                   \
          o0_f /= (float4(1.0f) + exp(-g0_f));                                 \
        }                                                                      \
        *reinterpret_cast<device bfloat4 *>(o_row0 + (OFF_C)) = bfloat4(o0_f); \
      }                                                                        \
      if (ep_qv1) {                                                            \
        float4 o1_f =                                                          \
            float4((OREG)[4], (OREG)[5], (OREG)[6], (OREG)[7]) * rcp1;         \
        if (p.gate_head_stride != 0u) {                                        \
          ushort4 gu1 =                                                        \
              *reinterpret_cast<device const ushort4 *>(g_row1 + (OFF_C));     \
          if (!warp_full_rows) {                                               \
            gu1 = gqa_clean_bf4_u16(gu1);                                      \
          }                                                                    \
          const float4 g1_f = float4(as_type<bfloat4>(gu1));                   \
          o1_f /= (float4(1.0f) + exp(-g1_f));                                 \
        }                                                                      \
        *reinterpret_cast<device bfloat4 *>(o_row1 + (OFF_C)) = bfloat4(o1_f); \
      }                                                                        \
    } while (false)
  BF16_BK32_STORE_O(O0, 0u);
  BF16_BK32_STORE_O(O1, 16u);
  BF16_BK32_STORE_O(O2, 32u);
  BF16_BK32_STORE_O(O3, 48u);
  #undef BF16_BK32_STORE_O
}

FLASH_NAX_ALWAYS_INLINE
static inline void paged_flash_prefill_single_head_bk32_bf16_hd256_impl(
    device const bfloat *query, device const bfloat *gate,
    device const bfloat *cache_key, device const bfloat *cache_value,
    device bfloat *output, device const uint *page_table,
    device const uint *positions, constant PagedAttentionParams &p,
    threadgroup FlashGqaBk32TgStorageBf16<2u> &tg_mem, ushort lane,
    ushort sg, uint3 tg) {
  constexpr uint BQ = 64u;
  constexpr uint BK = 32u;
  if (p.query_heads == 0u || p.key_value_heads == 0u || p.tokens == 0u) {
    return;
  }
  if ((tg.y / p.query_heads) * BQ >= p.tokens) {
    return;
  }

  #pragma clang loop unroll(disable)
  for (uint sub = 0u; sub < 2u; ++sub) {
    const uint q_blk = tg.y / p.query_heads;
    const uint qh = tg.y - q_blk * p.query_heads;
    const uint sub_base = q_blk * BQ + sub * 32u;
    if (sub_base >= p.tokens) {
      break;
    }
    if (sub > 0u) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint repeat = p.query_heads / p.key_value_heads;
    const uint kvh = qh / repeat;
    const uint flat_tid = uint(sg) * 32u + uint(lane);
    const uint row_group = uint(sg) >> 2u;
    const short2 sc = Frag16x16::get_coord(lane);

    const uint active_rows = min(32u, p.tokens - sub_base);
    threadgroup uint *tg_q_pos =
        reinterpret_cast<threadgroup uint *>(tg_mem.xchg);
    if (flat_tid < active_rows) {
      tg_q_pos[flat_tid] = positions[sub_base + flat_tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint max_valid_kv = 0u;
    uint min_q_pos = 0xffffffffu;
    for (uint r = 0u; r < active_rows; ++r) {
      const uint pos = tg_q_pos[r];
      max_valid_kv = max(max_valid_kv, pos + 1u);
      min_q_pos = min(min_q_pos, pos);
    }
    const uint nk_blocks = (max_valid_kv + BK - 1u) / BK;
    const uint kb_min_causal = min_q_pos / BK;

    const uint tm = row_group * 16u;
    const bool warp_full_rows = (int(active_rows) - int(tm) >= 16);
    const uint r0 = tm + uint(sc.y);
    const uint r1 = tm + 8u + uint(sc.y);
    const bool qv0 = (r0 < active_rows);
    const bool qv1 = (r1 < active_rows);
    const uint tok0 = qv0 ? (sub_base + r0) : sub_base;
    const uint tok1 = qv1 ? (sub_base + r1) : sub_base;
    const uint qpos0 = qv0 ? tg_q_pos[r0] : 0u;
    const uint qpos1 = qv1 ? tg_q_pos[r1] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_bk32_wn4_bf16_hd256_core<2u, false>(
        query, gate, cache_key, cache_value, output, page_table, p, tg_mem, kvh,
        nk_blocks, max_valid_kv, kb_min_causal, tok0, qh, qpos0, qv0, tok1, qh,
        qpos1, qv1, warp_full_rows, lane, sg);
  }
}

template <uint BD>
FLASH_NAX_ALWAYS_INLINE
static inline void paged_flash_prefill_single_head_bk16_bf16_impl(
    device const bfloat *query, device const bfloat *gate,
    device const bfloat *cache_key, device const bfloat *cache_value,
    device bfloat *output, device const uint *page_table,
    device const uint *positions, constant PagedAttentionParams &p,
    threadgroup FlashGqaBk16TgStorageBf16<2u, BD> &tg_mem, ushort lane,
    ushort sg, uint3 tg) {
  constexpr uint BQ = 64u;
  constexpr uint BK = 16u;
  constexpr uint WN = BD / 64u;
  if (p.query_heads == 0u || p.key_value_heads == 0u || p.tokens == 0u) {
    return;
  }
  if ((tg.y / p.query_heads) * BQ >= p.tokens) {
    return;
  }

  #pragma clang loop unroll(disable)
  for (uint sub = 0u; sub < 2u; ++sub) {
    const uint q_blk = tg.y / p.query_heads;
    const uint qh = tg.y - q_blk * p.query_heads;
    const uint sub_base = q_blk * BQ + sub * 32u;
    if (sub_base >= p.tokens) {
      break;
    }
    if (sub > 0u) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint repeat = p.query_heads / p.key_value_heads;
    const uint kvh = qh / repeat;
    const uint flat_tid = uint(sg) * 32u + uint(lane);
    const uint row_group = uint(sg) / WN;
    const short2 sc = Frag16x16::get_coord(lane);

    const uint active_rows = min(32u, p.tokens - sub_base);
    threadgroup uint *tg_q_pos =
        reinterpret_cast<threadgroup uint *>(tg_mem.kv_tile);
    if (flat_tid < active_rows) {
      tg_q_pos[flat_tid] = positions[sub_base + flat_tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint max_valid_kv = 0u;
    uint min_q_pos = 0xffffffffu;
    for (uint r = 0u; r < active_rows; ++r) {
      const uint pos = tg_q_pos[r];
      max_valid_kv = max(max_valid_kv, pos + 1u);
      min_q_pos = min(min_q_pos, pos);
    }
    const uint nk_blocks = (max_valid_kv + BK - 1u) / BK;
    const uint kb_min_causal = min_q_pos / BK;

    const uint tm = row_group * 16u;
    const bool warp_full_rows = (int(active_rows) - int(tm) >= 16);
    const uint r0 = tm + uint(sc.y);
    const uint r1 = tm + 8u + uint(sc.y);
    const bool qv0 = (r0 < active_rows);
    const bool qv1 = (r1 < active_rows);
    const uint tok0 = qv0 ? (sub_base + r0) : sub_base;
    const uint tok1 = qv1 ? (sub_base + r1) : sub_base;
    const uint qpos0 = qv0 ? tg_q_pos[r0] : 0u;
    const uint qpos1 = qv1 ? tg_q_pos[r1] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_bk16_wn_bf16_core<2u, BD, false>(
        query, gate, cache_key, cache_value, output, page_table, p, tg_mem, kvh,
        nk_blocks, max_valid_kv, kb_min_causal, tok0, qh, qpos0, qv0, tok1, qh,
        qpos1, qv1, warp_full_rows, lane, sg);
  }
}

} // namespace flash_nax

// Grid (threads): width = 256 (WM=2 * WN=4 * 32), height = ceil(tokens/64) * query_heads
// Threads per threadgroup: {256, 1, 1}
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_paged_flash_prefill_nax_hd256(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const bfloat *cache_key [[buffer(2)]],
    device const bfloat *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *page_table [[buffer(5)]],
    device const uint *positions [[buffer(6)]],
    constant PagedAttentionParams &params [[buffer(7)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup flash_nax::FlashGqaBk32TgStorageBf16<2u> tg_mem;
  flash_nax::paged_flash_prefill_single_head_bk32_bf16_hd256_impl(
      query, gate, cache_key, cache_value, output, page_table, positions,
      params, tg_mem, lane, sg, tg);
}

// Grid (threads): width = 128 (WM=2 * WN=2 * 32), height = ceil(capacity/64) * query_heads
// Threads per threadgroup: {128, 1, 1}
[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_paged_flash_prefill_nax_hd128(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const bfloat *cache_key [[buffer(2)]],
    device const bfloat *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *page_table [[buffer(5)]],
    device const uint *positions [[buffer(6)]],
    constant PagedAttentionParams &params [[buffer(7)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup flash_nax::FlashGqaBk16TgStorageBf16<2u, 128u> tg_mem;
  flash_nax::paged_flash_prefill_single_head_bk16_bf16_impl<128u>(
      query, gate, cache_key, cache_value, output, page_table, positions,
      params, tg_mem, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_paged_flash_prefill_nax_hd64(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const bfloat *cache_key [[buffer(2)]],
    device const bfloat *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *page_table [[buffer(5)]],
    device const uint *positions [[buffer(6)]],
    constant PagedAttentionParams &params [[buffer(7)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup flash_nax::FlashHd64TgStorage tg_mem;
  threadgroup float xchg_unused[1];
  if (params.query_heads == 0u) return;
  const uint3 mapped_tg(tg.y / params.query_heads, tg.y % params.query_heads, 0u);
  flash_nax::paged_flash_prefill_nax_impl<64u, 1u>(
      query, gate, cache_key, cache_value, output, page_table, positions,
      params, xchg_unused, tg_mem.tg_q_pos, lane, sg, mapped_tg);
}

// ============================================================================
// 2. Register-Resident 48-Query BF16 Split-KV (b5_bf16_frag48_bfp)
//    WM=3, WN=2 (192 threads = 6 warps), 48 queries = up to 8 tok x 6 GQA
//    sharing 1.0x KV DRAM read, 12 KiB xchg (2 TGs/core), bfloat*bfloat NAX P@V,
//    normalized half partials (3.02 ms @ 32K s1024, 5.81 ms @ 64K s2048 — 4.04x!).
// Grid (threads): width = splits * 192, height = key_value_heads
// Threads per threadgroup: {192, 1, 1}
// ============================================================================
[[max_total_threads_per_threadgroup(192)]]
kernel void omlx_m5_paged_attention_bf16_frag48_split_partial_hd256(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *cache_key [[buffer(1)]],
    device const bfloat *cache_value [[buffer(2)]],
    device float *partial_max [[buffer(3)]],
    device float *partial_sum [[buffer(4)]],
    device half *partial_output_half [[buffer(5)]],
    device const uint *page_table [[buffer(6)]],
    device const uint *positions [[buffer(7)]],
    constant PagedSplitAttentionParams &p [[buffer(8)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint BD = 256u;
  constexpr uint BK = 16u;
  constexpr uint kRepeat = 6u;

  struct Frag48XchgStorage {
    // 3 row groups x 32 lanes x 8 floats = 3,072 B (3.00 KiB <= 16 KiB!)
    float4 xchg[3u][32u][2u];
  };
  static_assert(sizeof(Frag48XchgStorage) == 3072u &&
                    sizeof(Frag48XchgStorage) <= kApple10MaxPrefillTgSramBytes,
                "Frag48 split TgStorage exceeds 16,384 B!");
  alignas(16) threadgroup Frag48XchgStorage tg_mem;

  const uint split = tg.x;
  const uint kvh = tg.y;
  if (split >= p.split_count || kvh >= p.key_value_heads || p.tokens == 0u ||
      p.tokens > 8u) {
    return;
  }

  const uint active_rows = p.tokens * kRepeat; // <= 48
  uint max_valid_kv = 0u;
  uint min_q_pos = 0xffffffffu;
  for (uint tk = 0u; tk < p.tokens; ++tk) {
    const uint pos = positions[tk];
    max_valid_kv = max(max_valid_kv, pos + 1u);
    min_q_pos = min(min_q_pos, pos);
  }

  const uint split_start = split * p.split_tokens;
  const uint split_end = min(max_valid_kv, split_start + p.split_tokens);
  const uint flat_tid = uint(sg) * 32u + uint(lane);

  // Fast empty-split exit without touching KV DRAM or partial_output_half.
  if (split_start >= split_end || max_valid_kv == 0u) {
    if (flat_tid < active_rows) {
      const uint tok = flat_tid / kRepeat;
      const uint qh = kvh * kRepeat + (flat_tid - tok * kRepeat);
      const ulong record =
          (ulong(tok) * p.query_heads + qh) * p.split_count + split;
      partial_max[record] = flash_nax::kNegInf;
      partial_sum[record] = 0.0f;
    }
    return;
  }

  const uint kb_start = split_start / BK;
  const uint kb_end = (split_end + BK - 1u) / BK;
  const uint kb_min_causal = min_q_pos / BK;
  const uint kv_stride_u32 = p.key_value_heads * BD;
  const uint blocks_per_page = p.page_tokens / BK;

  const short row_group = short(sg >> 1u); // 0, 1, 2
  const short d_part = short(sg & 1u);     // 0, 1
  const short tm = 16 * row_group;         // 0, 16, 32
  const uint d_offset = uint(d_part) * 128u;
  const bool warp_full_rows = (int(active_rows) - int(tm) >= 16);

  const short2 sc = flash_nax::Frag16x16::get_coord(lane);
  const short sm = sc.y;
  const short sn = sc.x;

  const uint r0 = uint(tm + sm);
  const uint r1 = uint(tm + 8 + sm);
  const bool qv0 = (r0 < active_rows);
  const bool qv1 = (r1 < active_rows);
  const uint tok0 = qv0 ? (r0 / kRepeat) : 0u;
  const uint qh0 = kvh * kRepeat + (qv0 ? (r0 - tok0 * kRepeat) : 0u);
  const uint tok1 = qv1 ? (r1 / kRepeat) : 0u;
  const uint qh1 = kvh * kRepeat + (qv1 ? (r1 - tok1 * kRepeat) : 0u);
  const uint qpos0 = qv0 ? positions[tok0] : 0u;
  const uint qpos1 = qv1 ? positions[tok1] : 0u;

  using BF8 = flash_nax::Frag16x16::frag_t<bfloat>;
  using F8 = flash_nax::Frag16x16::frag_t<float>;
  const float scale2 = 0.0625f * flash_nax::kLog2e;

  // 2 x 64-col passes (only O0..O3 = 32 floats live -> 3.0 KiB SRAM, 0 B spills!).
  #pragma clang loop unroll(disable)
  for (uint d_half = 0u; d_half < 2u; ++d_half) {
    const uint v_sub_off = d_half * 64u;
    F8 O0(0.0f), O1(0.0f), O2(0.0f), O3(0.0f);
    float max_s0 = flash_nax::kNegInf;
    float max_s1 = flash_nax::kNegInf;
    float sum_s0 = 0.0f;
    float sum_s1 = 0.0f;

    device const bfloat *q_row0 =
        query + ulong(tok0) * p.query_token_stride +
        ulong(qh0) * p.query_head_stride + d_offset + uint(sn);
    device const bfloat *q_row1 =
        query + ulong(tok1) * p.query_token_stride +
        ulong(qh1) * p.query_head_stride + d_offset + uint(sn);

    for (uint kb = kb_start; kb < kb_end; ++kb) {
      const uint logical_page = kb / blocks_per_page;
      const uint sub_offset = (kb % blocks_per_page) * BK;
      const ulong phys_row =
          ulong(page_table[logical_page]) * p.page_tokens + sub_offset;
      const uint kb_col_base = kb * BK;
      const short valid_k = short(min(BK, split_end - kb_col_base));
      const bool fast_kv = (valid_k == 16) && (kb < kb_min_causal);

      device const bfloat *K_kb =
          cache_key + phys_row * ulong(kv_stride_u32) + ulong(kvh) * BD +
          d_offset;

      F8 S(0.0f);
      #pragma clang loop unroll(disable)
      for (uint dc = 0u; dc < 128u; dc += 32u) {
        const bfloat4 q00 = qv0
            ? *reinterpret_cast<device const bfloat4 *>(q_row0 + dc)
            : bfloat4(bfloat(0.0f));
        const bfloat4 q01 = qv1
            ? *reinterpret_cast<device const bfloat4 *>(q_row1 + dc)
            : bfloat4(bfloat(0.0f));
        const bfloat4 q10 = qv0
            ? *reinterpret_cast<device const bfloat4 *>(q_row0 + dc + 16u)
            : bfloat4(bfloat(0.0f));
        const bfloat4 q11 = qv1
            ? *reinterpret_cast<device const bfloat4 *>(q_row1 + dc + 16u)
            : bfloat4(bfloat(0.0f));
        const BF8 qa(q00.x, q00.y, q00.z, q00.w, q01.x, q01.y, q01.z, q01.w);
        const BF8 qb(q10.x, q10.y, q10.z, q10.w, q11.x, q11.y, q11.z, q11.w);
        BF8 K0, K1;
        if (fast_kv) {
          flash_nax::Frag16x16::load_bf4(K0, K_kb, ulong(kv_stride_u32),
                                         short(dc), sc);
          flash_nax::Frag16x16::load_bf4(K1, K_kb, ulong(kv_stride_u32),
                                         short(dc + 16u), sc);
        } else if (valid_k < 16) {
          flash_nax::Frag16x16::load_rows_clean_bf4(
              K0, K_kb, ulong(kv_stride_u32), valid_k, short(dc), sc);
          flash_nax::Frag16x16::load_rows_clean_bf4(
              K1, K_kb, ulong(kv_stride_u32), valid_k, short(dc + 16u), sc);
        } else {
          flash_nax::Frag16x16::load_clean_bf4(K0, K_kb, ulong(kv_stride_u32),
                                               short(dc), sc);
          flash_nax::Frag16x16::load_clean_bf4(K1, K_kb, ulong(kv_stride_u32),
                                               short(dc + 16u), sc);
        }
        nax_mma_16x16x32<bfloat, true>(S, qa, qb, K0, K1);
      }

      if (d_part == 1) {
        tg_mem.xchg[row_group][lane][0] = float4(S[0], S[1], S[2], S[3]);
        tg_mem.xchg[row_group][lane][1] = float4(S[4], S[5], S[6], S[7]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (d_part == 0) {
        const float4 p0 = tg_mem.xchg[row_group][lane][0];
        const float4 p1 = tg_mem.xchg[row_group][lane][1];
        S += F8(p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w);
        tg_mem.xchg[row_group][lane][0] = float4(S[0], S[1], S[2], S[3]);
        tg_mem.xchg[row_group][lane][1] = float4(S[4], S[5], S[6], S[7]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (d_part == 1) {
        const float4 s0 = tg_mem.xchg[row_group][lane][0];
        const float4 s1 = tg_mem.xchg[row_group][lane][1];
        S = F8(s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float4 s0(S[0], S[1], S[2], S[3]);
      float4 s1(S[4], S[5], S[6], S[7]);
      const bool full_causal_free = fast_kv && warp_full_rows;
      if (full_causal_free) {
        s0 *= scale2;
        s1 *= scale2;
      } else {
        const uint4 c0_abs =
            uint4(kb_col_base + uint(sn)) + uint4(0u, 1u, 2u, 3u);
        const bool4 ok00 = qv0 && (c0_abs < split_end) && (c0_abs <= qpos0);
        const bool4 ok01 = qv1 && (c0_abs < split_end) && (c0_abs <= qpos1);
        s0 = select(float4(flash_nax::kNegInf), s0 * scale2, ok00);
        s1 = select(float4(flash_nax::kNegInf), s1 * scale2, ok01);
      }

      float m0 = max(max(s0.x, s0.y), max(s0.z, s0.w));
      m0 = max(m0, simd_shuffle_xor(m0, ushort(1)));
      m0 = max(m0, simd_shuffle_xor(m0, ushort(8)));
      const float new_m0 = max(max_s0, m0);

      float m1 = max(max(s1.x, s1.y), max(s1.z, s1.w));
      m1 = max(m1, simd_shuffle_xor(m1, ushort(1)));
      m1 = max(m1, simd_shuffle_xor(m1, ushort(8)));
      const float new_m1 = max(max_s1, m1);

      const float fac0 = (max_s0 <= flash_nax::kNegInfHalf)
                             ? 0.0f
                             : fast::exp2(max_s0 - new_m0);
      const float fac1 = (max_s1 <= flash_nax::kNegInfHalf)
                             ? 0.0f
                             : fast::exp2(max_s1 - new_m1);
      max_s0 = new_m0;
      max_s1 = new_m1;
      sum_s0 *= fac0;
      sum_s1 *= fac1;

      float4 p00, p01;
      if (full_causal_free) {
        p00 = fast::exp2(s0 - new_m0);
        p01 = fast::exp2(s1 - new_m1);
      } else {
        const bool emp0 = (new_m0 <= flash_nax::kNegInfHalf);
        const bool emp1 = (new_m1 <= flash_nax::kNegInfHalf);
        p00 = select(fast::exp2(s0 - new_m0), float4(0.0f),
                     emp0 || (s0 <= flash_nax::kNegInfHalf));
        p01 = select(fast::exp2(s1 - new_m1), float4(0.0f),
                     emp1 || (s1 <= flash_nax::kNegInfHalf));
      }

      float tsum0 = (p00.x + p00.y) + (p00.z + p00.w);
      tsum0 += simd_shuffle_xor(tsum0, ushort(1));
      tsum0 += simd_shuffle_xor(tsum0, ushort(8));
      sum_s0 += tsum0;

      float tsum1 = (p01.x + p01.y) + (p01.z + p01.w);
      tsum1 += simd_shuffle_xor(tsum1, ushort(1));
      tsum1 += simd_shuffle_xor(tsum1, ushort(8));
      sum_s1 += tsum1;

      const bfloat4 bp00 = bfloat4(p00);
      const bfloat4 bp01 = bfloat4(p01);
      const BF8 P0(bp00.x, bp00.y, bp00.z, bp00.w, bp01.x, bp01.y, bp01.z,
                   bp01.w);

      const F8 fac_vec(fac0, fac0, fac0, fac0, fac1, fac1, fac1, fac1);
      O0 *= fac_vec;
      O1 *= fac_vec;
      O2 *= fac_vec;
      O3 *= fac_vec;

      device const bfloat *V_kb =
          cache_value + phys_row * ulong(kv_stride_u32) + ulong(kvh) * BD +
          d_offset + v_sub_off;
      auto load_v = [&](short off_c) -> BF8 {
        BF8 V;
        if (fast_kv) {
          flash_nax::Frag16x16::load_bf4(V, V_kb, ulong(kv_stride_u32), off_c,
                                         sc);
        } else if (valid_k < 16) {
          flash_nax::Frag16x16::load_rows_clean_bf4(
              V, V_kb, ulong(kv_stride_u32), valid_k, off_c, sc);
        } else {
          flash_nax::Frag16x16::load_clean_bf4(V, V_kb, ulong(kv_stride_u32),
                                               off_c, sc);
        }
        return V;
      };
      nax_mma_16x32x16_nn<bfloat>(O0, O1, P0, load_v(0), load_v(16));
      nax_mma_16x32x16_nn<bfloat>(O2, O3, P0, load_v(32), load_v(48));
    }

    const float rcp0 = (sum_s0 > 0.0f) ? (1.0f / sum_s0) : 0.0f;
    const float rcp1 = (sum_s1 > 0.0f) ? (1.0f / sum_s1) : 0.0f;
    const ulong rec0 =
        (ulong(tok0) * p.query_heads + qh0) * p.split_count + split;
    const ulong rec1 =
        (ulong(tok1) * p.query_heads + qh1) * p.split_count + split;

    if (d_half == 0u && d_part == 0 && sn == 0) {
      if (qv0) {
        partial_max[rec0] = max_s0;
        partial_sum[rec0] = sum_s0;
      }
      if (qv1) {
        partial_max[rec1] = max_s1;
        partial_sum[rec1] = sum_s1;
      }
    }

    device half *po0 =
        partial_output_half + rec0 * BD + d_offset + v_sub_off + uint(sn);
    device half *po1 =
        partial_output_half + rec1 * BD + d_offset + v_sub_off + uint(sn);
    auto store_part = [&](F8 oreg, uint off_c) {
      if (qv0) {
        *reinterpret_cast<device half4 *>(po0 + off_c) =
            half4(float4(oreg[0], oreg[1], oreg[2], oreg[3]) * rcp0);
      }
      if (qv1) {
        *reinterpret_cast<device half4 *>(po1 + off_c) =
            half4(float4(oreg[4], oreg[5], oreg[6], oreg[7]) * rcp1);
      }
    };
    store_part(O0, 0u);
    store_part(O1, 16u);
    store_part(O2, 32u);
    store_part(O3, 48u);
  }
}

// ============================================================================
// 3. 4-SG (128-thread) Parallel Log2 Half Split-KV Merge for BF16 Frag48
// Launch: grid = {128, tokens * query_heads}, tptg = {128}
// ============================================================================
kernel void omlx_m5_paged_attention_bf16_split_merge_half_4sg_hd256(
    device const bfloat *gate [[buffer(0)]],
    device const float *partial_max [[buffer(1)]],
    device const float *partial_sum [[buffer(2)]],
    device const half *partial_output_half [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant PagedSplitAttentionParams &params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint logical_head = tg.y;
  const uint token = logical_head / params.query_heads;
  if (token >= params.tokens) {
    return;
  }
  const uint query_head = logical_head - token * params.query_heads;
  const uint tid = sg * 32u + lane;
  const uint splits = params.split_count;
  const ulong base_record = ulong(logical_head) * ulong(splits);

  threadgroup float sg_m[4];
  threadgroup float sg_l[4];
  threadgroup float tg_w[256];
  threadgroup float tg_o[4 * 256]; // 4 KiB

  float local_m = flash_nax::kNegInf;
  for (uint s = tid; s < splits; s += 128u) {
    const float m_s = partial_max[base_record + s];
    if (m_s > flash_nax::kNegInfHalf) {
      local_m = max(local_m, m_s);
    }
  }
  const float warp_m = simd_max(local_m);
  if (lane == 0u) {
    sg_m[sg] = warp_m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float global_m = max(max(sg_m[0], sg_m[1]), max(sg_m[2], sg_m[3]));

  float local_l = 0.0f;
  for (uint s = tid; s < splits; s += 128u) {
    const float m_s = partial_max[base_record + s];
    float w_s = 0.0f;
    if (m_s > flash_nax::kNegInfHalf) {
      w_s = partial_sum[base_record + s] * fast::exp2(m_s - global_m);
      local_l += w_s;
    }
    if (s < 256u) {
      tg_w[s] = w_s;
    }
  }
  const float warp_l = simd_sum(local_l);
  if (lane == 0u) {
    sg_l[sg] = warp_l;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float global_l = (sg_l[0] + sg_l[1]) + (sg_l[2] + sg_l[3]);
  const float inv_l = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;

  float4 acc0(0.0f);
  float4 acc1(0.0f);
  const uint d0 = lane * 8u;
  for (uint s = sg; s < splits; s += 4u) {
    float w_s;
    if (s < 256u) {
      w_s = tg_w[s];
    } else {
      const float m_s = partial_max[base_record + s];
      w_s = (m_s > flash_nax::kNegInfHalf)
                ? (partial_sum[base_record + s] * fast::exp2(m_s - global_m))
                : 0.0f;
    }
    if (w_s > 0.0f) {
      const float nw = w_s * inv_l;
      device const half4 *part4 = reinterpret_cast<device const half4 *>(
          partial_output_half + (base_record + s) * 256ul + d0);
      acc0 = fma(float4(part4[0]), float4(nw), acc0);
      acc1 = fma(float4(part4[1]), float4(nw), acc1);
    }
  }

  reinterpret_cast<threadgroup float4 *>(tg_o + sg * 256u + d0)[0] = acc0;
  reinterpret_cast<threadgroup float4 *>(tg_o + sg * 256u + d0)[1] = acc1;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device const bfloat *gate_row =
      gate + ulong(token) * params.gate_token_stride +
      ulong(query_head) * params.gate_head_stride;
  device bfloat *out_row = output + ulong(logical_head) * 256ul;

  FLASH_NAX_UNROLL
  for (uint k = 0u; k < 2u; ++k) {
    const uint d = tid + k * 128u;
    float val = (tg_o[0u * 256u + d] + tg_o[1u * 256u + d]) +
                (tg_o[2u * 256u + d] + tg_o[3u * 256u + d]);
    if (params.gate_head_stride != 0u) {
      const float g_val = float(gate_row[d]);
      val /= (1.0f + exp(-g_val));
    }
    out_row[d] = bfloat(val);
  }
}

// ============================================================================
// 4. Preserved FP32-output Apple10 NAX BF16 GEMM helper for DFlash2
// ============================================================================
struct Bf16NaxF32Params {
  uint n;
  uint k;
  uint mp;
  uint rows;
};

template <uint BN, uint BK, uint SK>
inline void bf16_nax_linear_f32_impl(device const bfloat *x,
                                     device const bfloat *w, device float *y,
                                     constant Bf16NaxF32Params &p,
                                     threadgroup float *parts, uint lane,
                                     uint sg, uint3 tg) {
  constexpr uint BM = 8u;
  constexpr uint kCap = (16u * BN) / 32u;
  const uint rb = tg.y * BM;
  const uint n0 = tg.x * BN;
  if (rb >= p.rows || n0 >= p.n) {
    return;
  }

  const uint blocks = p.k / BK;
  const uint b0 = (sg * blocks) / SK;
  const uint b1 = ((sg + 1u) * blocks) / SK;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a_parent(
      (device bfloat *)(x + ulong(rb) * p.k),
      dextents<int32_t, 2>(p.k, p.rows - rb));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> b_parent(
      (device bfloat *)(w + ulong(n0) * p.k),
      dextents<int32_t, 2>(p.k, p.n - n0));

  auto a_init = a_parent.slice(0, 0);
  auto b_init = b_parent.slice(0, 0);
  auto prod = op.template get_destination_cooperative_tensor<
      decltype(a_init), decltype(b_init), float>();
  const uint elems = prod.get_capacity();
  for (uint i = 0u; i < elems; ++i) {
    prod[i] = 0.0f;
  }

  for (uint blk = b0; blk < b1; ++blk) {
    const uint k_off = blk * BK;
    auto a = a_parent.slice(k_off, 0);
    auto b = b_parent.slice(k_off, 0);
    op.run(a, b, prod);
  }

  if constexpr (SK > 1u) {
    if (sg > 0u) {
      for (uint i = 0u; i < elems; ++i) {
        parts[((sg - 1u) * kCap + i) * 32u + lane] = prod[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      for (uint s = 1u; s < SK; ++s) {
        for (uint i = 0u; i < elems; ++i) {
          prod[i] += parts[((s - 1u) * kCap + i) * 32u + lane];
        }
      }
    }
  }

  if (sg == 0u) {
    for (uint i = 0u; i < elems; ++i) {
      if (prod.is_valid_element(i)) {
        const auto idx = prod.get_multidimensional_index(i);
        const uint col = n0 + uint(idx[0]);
        const uint row = rb + uint(idx[1]);
        if (row < p.rows && col < p.n) {
          y[ulong(row) * p.n + col] = prod[i];
        }
      }
    }
  }
}

kernel void omlx_m5_bf16_nax_f32_bk128_n16_sk4(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant Bf16NaxF32Params &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float parts[3 * 8 * 32];
  bf16_nax_linear_f32_impl<16u, 128u, 4u>(x, w, y, p, parts, lane, sg, tg);
}

kernel void omlx_m5_bf16_nax_f32_bk64_n16_sk4(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant Bf16NaxF32Params &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float parts[3 * 8 * 32];
  bf16_nax_linear_f32_impl<16u, 64u, 4u>(x, w, y, p, parts, lane, sg, tg);
}

// ============================================================================
// 5. WP3.2 BK=32, WN=4 Direct-L1 Zero-Spill GQA Flash-Prefill (BF16, HD=256)
//    - Flattened (q_tok, h_sub) packing: 100% row fill on G=6 (27B) & G=8 (35B)!
//    - LPT Reverse m_blk + kvh-minor scheduling (0 atomics, ICB-safe)
//    - WM=2: 14.00 KiB exact (256 thr, 2 TGs/core, 0 B spills)
//    - WM=3: 12.00 KiB exact (384 thr, <= 14.00 KiB, 0 B spills)
//    - 0-ULP bit-exact parity across WM=2, WM=3, and single-head hd256!
// ============================================================================
namespace flash_nax {

template <uint WM>
FLASH_NAX_ALWAYS_INLINE
static inline void paged_flash_prefill_gqa_wn4_nax_hd256_impl(
    device const bfloat *query, device const bfloat *gate,
    device const bfloat *cache_key, device const bfloat *cache_value,
    device bfloat *output, device const uint *page_table,
    device const uint *positions, constant PagedAttentionParams &p,
    threadgroup FlashGqaBk32TgStorageBf16<WM> &tg_mem, ushort lane,
    ushort sg, uint3 tg) {
  constexpr uint BK = 32u;
  constexpr uint WN = 4u;
  constexpr uint M_TILE = WM * 16u;

  if (p.tokens == 0u || p.query_heads == 0u || p.key_value_heads == 0u) {
    return;
  }
  const uint repeat = p.query_heads / p.key_value_heads;
  if (repeat == 0u || repeat > 16u) {
    return;
  }
  const uint m_total = p.tokens * repeat;
  const uint n_m_blocks = (m_total + M_TILE - 1u) / M_TILE;

  // LPT Reverse m_blk + kvh-minor mapping (longest causal blocks launch first).
  const uint m_blk_rev = tg.y / p.key_value_heads;
  const uint kvh = tg.y - m_blk_rev * p.key_value_heads;
  if (m_blk_rev >= n_m_blocks) {
    return;
  }
  const uint m_blk = (n_m_blocks - 1u) - m_blk_rev;
  const uint m_base = m_blk * M_TILE;
  const uint active_m_rows = min(M_TILE, m_total - m_base);

  const uint first_tok = m_base / repeat;
  const uint last_tok = (m_base + active_m_rows - 1u) / repeat;
  const uint num_toks = last_tok - first_tok + 1u;

  const uint flat_tid = uint(sg) * 32u + uint(lane);
  threadgroup uint *tg_q_pos =
      reinterpret_cast<threadgroup uint *>(tg_mem.xchg);
  if (flat_tid < num_toks) {
    tg_q_pos[flat_tid] = positions[first_tok + flat_tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint max_valid_kv = 0u;
  uint min_q_pos = 0xffffffffu;
  for (uint r = 0u; r < num_toks; ++r) {
    const uint pos = tg_q_pos[r];
    max_valid_kv = max(max_valid_kv, pos + 1u);
    min_q_pos = min(min_q_pos, pos);
  }
  const uint nk_blocks = (max_valid_kv + BK - 1u) / BK;
  const uint kb_min_causal = min_q_pos / BK;

  const uint row_group = uint(sg) / WN;
  const uint tm = row_group * 16u;
  const bool warp_full_rows = (int(active_m_rows) - int(tm) >= 16);
  const short2 sc = Frag16x16::get_coord(lane);

  const uint m_off_base = m_base - first_tok * repeat; // < repeat <= 16
  const uint local_r0 = tm + uint(sc.y);
  const uint local_r1 = tm + 8u + uint(sc.y);
  const bool qv0 = (local_r0 < active_m_rows);
  const bool qv1 = (local_r1 < active_m_rows);
  uint tok_rel0 = 0u, g0 = 0u, tok_rel1 = 0u, g1 = 0u;
  if (qv0) {
    fast_divmod_g(m_off_base + local_r0, repeat, tok_rel0, g0);
  }
  if (qv1) {
    fast_divmod_g(m_off_base + local_r1, repeat, tok_rel1, g1);
  }
  const uint tok0 = first_tok + tok_rel0;
  const uint tok1 = first_tok + tok_rel1;
  const uint qh0 = kvh * repeat + g0;
  const uint qh1 = kvh * repeat + g1;
  const uint qpos0 = qv0 ? tg_q_pos[tok_rel0] : 0u;
  const uint qpos1 = qv1 ? tg_q_pos[tok_rel1] : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  flash_bk32_wn4_bf16_hd256_core<WM>(
      query, gate, cache_key, cache_value, output, page_table, p, tg_mem, kvh,
      nk_blocks, max_valid_kv, kb_min_causal, tok0, qh0, qpos0, qv0, tok1, qh1,
      qpos1, qv1, warp_full_rows, lane, sg);
}

} // namespace flash_nax

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_paged_flash_prefill_gqa_bk16_wn4_nax_hd256_wm2(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const bfloat *cache_key [[buffer(2)]],
    device const bfloat *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *page_table [[buffer(5)]],
    device const uint *positions [[buffer(6)]],
    constant PagedAttentionParams &params [[buffer(7)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup flash_nax::FlashGqaBk32TgStorageBf16<2u> tg_mem;
  flash_nax::paged_flash_prefill_gqa_wn4_nax_hd256_impl<2u>(
      query, gate, cache_key, cache_value, output, page_table, positions,
      params, tg_mem, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(384)]]
kernel void omlx_m5_paged_flash_prefill_gqa_bk16_wn4_nax_hd256_wm3(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const bfloat *cache_key [[buffer(2)]],
    device const bfloat *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *page_table [[buffer(5)]],
    device const uint *positions [[buffer(6)]],
    constant PagedAttentionParams &params [[buffer(7)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup flash_nax::FlashGqaBk32TgStorageBf16<3u> tg_mem;
  flash_nax::paged_flash_prefill_gqa_wn4_nax_hd256_impl<3u>(
      query, gate, cache_key, cache_value, output, page_table, positions,
      params, tg_mem, lane, sg, tg);
}
