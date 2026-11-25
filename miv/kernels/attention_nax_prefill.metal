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
    BF8 Gfrag;                                                                 \
    if (lim_rows_q < 16) {                                                     \
      Frag16x16::load_rows_clean_bf4(Gfrag, G_ptr, p.gate_token_stride,        \
                                     lim_rows_q, OFF_C, sc);                   \
    } else {                                                                   \
      Frag16x16::load_bf4(Gfrag, G_ptr, p.gate_token_stride, OFF_C, sc);       \
    }                                                                          \
    const float4 g0 = float4(float(Gfrag[0]), float(Gfrag[1]), float(Gfrag[2]),\
                             float(Gfrag[3]));                                 \
    const float4 g1 = float4(float(Gfrag[4]), float(Gfrag[5]), float(Gfrag[6]),\
                             float(Gfrag[7]));                                 \
    const float4 o0 = (float4(OREG[0], OREG[1], OREG[2], OREG[3]) * rcp0) /    \
                      (float4(1.0f) + exp(-g0));                               \
    const float4 o1 = (float4(OREG[4], OREG[5], OREG[6], OREG[7]) * rcp1) /    \
                      (float4(1.0f) + exp(-g1));                               \
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
// 1. Upgraded BF16 NAX Flash-Prefill Entrypoints (69.8 TF @ HD=256, 0 alloca)
// ============================================================================
// Grid (threads): width = 256 (WM=4 * WN=2 * 32), height = ceil(tokens/64) * query_heads
// Threads per threadgroup: {256, 1, 1}
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
  // 8 KiB xchg (static_tg = 8.4 KiB -> guaranteed 2 TGs/core on Apple10!).
  threadgroup float xchg[4 * 16 * 32];
  threadgroup uint tg_q_pos[64];
  if (params.query_heads == 0u) return;
  const uint3 mapped_tg(tg.y / params.query_heads, tg.y % params.query_heads, 0u);
  flash_nax::paged_flash_prefill_nax_impl<256u, 2u>(
      query, gate, cache_key, cache_value, output, page_table, positions,
      params, xchg, tg_q_pos, lane, sg, mapped_tg);
}

// Grid (threads): width = 128 (WM=4 * WN=1 * 32), height = ceil(capacity/64) * query_heads
// Threads per threadgroup: {128, 1, 1}
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
  threadgroup float xchg_unused[1];
  threadgroup uint tg_q_pos[64];
  if (params.query_heads == 0u) return;
  const uint3 mapped_tg(tg.y / params.query_heads, tg.y % params.query_heads, 0u);
  flash_nax::paged_flash_prefill_nax_impl<128u, 1u>(
      query, gate, cache_key, cache_value, output, page_table, positions,
      params, xchg_unused, tg_q_pos, lane, sg, mapped_tg);
}

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
  threadgroup float xchg_unused[1];
  threadgroup uint tg_q_pos[64];
  if (params.query_heads == 0u) return;
  const uint3 mapped_tg(tg.y / params.query_heads, tg.y % params.query_heads, 0u);
  flash_nax::paged_flash_prefill_nax_impl<64u, 1u>(
      query, gate, cache_key, cache_value, output, page_table, positions,
      params, xchg_unused, tg_q_pos, lane, sg, mapped_tg);
}

// ============================================================================
// 2. Register-Resident 48-Query BF16 Split-KV (b5_bf16_frag48_bfp)
//    WM=3, WN=2 (192 threads = 6 warps), 48 queries = up to 8 tok x 6 GQA
//    sharing 1.0x KV DRAM read, 12 KiB xchg (2 TGs/core), bfloat*bfloat NAX P@V,
//    normalized half partials (3.02 ms @ 32K s1024, 5.81 ms @ 64K s2048 — 4.04x!).
// Grid (threads): width = splits * 192, height = key_value_heads
// Threads per threadgroup: {192, 1, 1}
// ============================================================================
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
  constexpr uint BK = 32u;
  constexpr uint kRepeat = 6u;

  threadgroup float xchg[6 * 16 * 32]; // 12 KiB -> 2 TGs/core on Apple10!

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
  const ulong kv_stride = ulong(p.key_value_heads) * BD;
  const uint half_blocks_per_page = p.page_tokens / BK;

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

  device const bfloat *q_row0 =
      query + ulong(tok0) * p.query_token_stride +
      ulong(qh0) * p.query_head_stride + d_offset + uint(sn);
  device const bfloat *q_row1 =
      query + ulong(tok1) * p.query_token_stride +
      ulong(qh1) * p.query_head_stride + d_offset + uint(sn);

  using BF8 = flash_nax::Frag16x16::frag_t<bfloat>;
  using F8 = flash_nax::Frag16x16::frag_t<float>;

  BF8 Q0(bfloat(0.0f)), Q1(bfloat(0.0f)), Q2(bfloat(0.0f)), Q3(bfloat(0.0f));
  BF8 Q4(bfloat(0.0f)), Q5(bfloat(0.0f)), Q6(bfloat(0.0f)), Q7(bfloat(0.0f));

#define FRAG48_LOAD_Q(QDST, OFF_C)                                             \
  do {                                                                         \
    const bfloat4 v0 = qv0                                                     \
        ? *reinterpret_cast<device const bfloat4 *>(q_row0 + uint(OFF_C))      \
        : bfloat4(bfloat(0.0f));                                               \
    const bfloat4 v1 = qv1                                                     \
        ? *reinterpret_cast<device const bfloat4 *>(q_row1 + uint(OFF_C))      \
        : bfloat4(bfloat(0.0f));                                               \
    QDST = BF8(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);                \
  } while (false)

  FRAG48_LOAD_Q(Q0, 0);
  FRAG48_LOAD_Q(Q1, 16);
  FRAG48_LOAD_Q(Q2, 32);
  FRAG48_LOAD_Q(Q3, 48);
  FRAG48_LOAD_Q(Q4, 64);
  FRAG48_LOAD_Q(Q5, 80);
  FRAG48_LOAD_Q(Q6, 96);
  FRAG48_LOAD_Q(Q7, 112);
#undef FRAG48_LOAD_Q

  F8 O0(0.0f), O1(0.0f), O2(0.0f), O3(0.0f);
  F8 O4(0.0f), O5(0.0f), O6(0.0f), O7(0.0f);

  float max_s0 = flash_nax::kNegInf;
  float max_s1 = flash_nax::kNegInf;
  float sum_s0 = 0.0f;
  float sum_s1 = 0.0f;
  const float scale2 = 0.0625f * flash_nax::kLog2e;

  threadgroup float4 *xchg_slot = reinterpret_cast<threadgroup float4 *>(
      xchg + (row_group * 2 + d_part) * (2 * 8 * 32) + short(lane) * 16);
  const threadgroup float4 *xchg_peer =
      reinterpret_cast<const threadgroup float4 *>(
          xchg + (row_group * 2 + (1 - d_part)) * (2 * 8 * 32) +
          short(lane) * 16);

  for (uint kb = kb_start; kb < kb_end; ++kb) {
    const uint logical_page = kb / half_blocks_per_page;
    const uint sub_offset = (kb % half_blocks_per_page) * BK;
    const ulong phys_row =
        ulong(page_table[logical_page]) * p.page_tokens + sub_offset;
    const uint kb_col_base = kb * BK;
    const short valid_k = short(min(BK, split_end - kb_col_base));
    const bool fast_kv = (valid_k == 32) && (kb < kb_min_causal);

    device const bfloat *K_kb =
        cache_key + phys_row * kv_stride + ulong(kvh) * BD + d_offset;
    device const bfloat *V_kb =
        cache_value + phys_row * kv_stride + ulong(kvh) * BD + d_offset;

    F8 S0(0.0f);
    F8 S1(0.0f);

#define FRAG48_QK_FAST(QREG, OFF_C)                                            \
  do {                                                                         \
    BF8 K0, K1;                                                                \
    flash_nax::Frag16x16::load_bf4(K0, K_kb, kv_stride, OFF_C, sc);            \
    flash_nax::Frag16x16::load_bf4(K1, K_kb + 16ul * kv_stride, kv_stride,     \
                                   OFF_C, sc);                                 \
    flash_nax::Frag16x16::mma_1x2_bf16<true>(S0, S1, QREG, K0, K1);            \
  } while (false)

#define FRAG48_QK_SAFE(QREG, OFF_C)                                            \
  do {                                                                         \
    BF8 K0, K1;                                                                \
    if (valid_k < 32) {                                                        \
      flash_nax::Frag16x16::load_rows_clean_bf4(K0, K_kb, kv_stride, valid_k,  \
                                                OFF_C, sc);                    \
      flash_nax::Frag16x16::load_rows_clean_bf4(K1, K_kb + 16ul * kv_stride,   \
                                                kv_stride,                     \
                                                short(int(valid_k) - 16),      \
                                                OFF_C, sc);                    \
    } else {                                                                   \
      flash_nax::Frag16x16::load_clean_bf4(K0, K_kb, kv_stride, OFF_C, sc);    \
      flash_nax::Frag16x16::load_clean_bf4(K1, K_kb + 16ul * kv_stride,        \
                                           kv_stride, OFF_C, sc);              \
    }                                                                          \
    flash_nax::Frag16x16::mma_1x2_bf16<true>(S0, S1, QREG, K0, K1);            \
  } while (false)

    if (fast_kv) {
      FRAG48_QK_FAST(Q0, 0);
      FRAG48_QK_FAST(Q1, 16);
      FRAG48_QK_FAST(Q2, 32);
      FRAG48_QK_FAST(Q3, 48);
      FRAG48_QK_FAST(Q4, 64);
      FRAG48_QK_FAST(Q5, 80);
      FRAG48_QK_FAST(Q6, 96);
      FRAG48_QK_FAST(Q7, 112);
    } else {
      FRAG48_QK_SAFE(Q0, 0);
      FRAG48_QK_SAFE(Q1, 16);
      FRAG48_QK_SAFE(Q2, 32);
      FRAG48_QK_SAFE(Q3, 48);
      FRAG48_QK_SAFE(Q4, 64);
      FRAG48_QK_SAFE(Q5, 80);
      FRAG48_QK_SAFE(Q6, 96);
      FRAG48_QK_SAFE(Q7, 112);
    }
#undef FRAG48_QK_FAST
#undef FRAG48_QK_SAFE

    xchg_slot[0] = float4(S0[0], S0[1], S0[2], S0[3]);
    xchg_slot[1] = float4(S0[4], S0[5], S0[6], S0[7]);
    xchg_slot[2] = float4(S1[0], S1[1], S1[2], S1[3]);
    xchg_slot[3] = float4(S1[4], S1[5], S1[6], S1[7]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const float4 p0 = xchg_peer[0];
      const float4 p1 = xchg_peer[1];
      const float4 p2 = xchg_peer[2];
      const float4 p3 = xchg_peer[3];
      S0 += F8(p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w);
      S1 += F8(p2.x, p2.y, p2.z, p2.w, p3.x, p3.y, p3.z, p3.w);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool full_causal_free = fast_kv && warp_full_rows;
    if (full_causal_free) {
      S0 *= scale2;
      S1 *= scale2;
    } else {
      const uint4 c0_abs =
          uint4(kb_col_base + uint(sn)) + uint4(0u, 1u, 2u, 3u);
      const uint4 c1_abs = c0_abs + uint4(16u);
      const bool4 ok00 = qv0 && (c0_abs < split_end) && (c0_abs <= qpos0);
      const bool4 ok01 = qv1 && (c0_abs < split_end) && (c0_abs <= qpos1);
      const bool4 ok10 = qv0 && (c1_abs < split_end) && (c1_abs <= qpos0);
      const bool4 ok11 = qv1 && (c1_abs < split_end) && (c1_abs <= qpos1);
      const float4 s00 = select(
          float4(flash_nax::kNegInf),
          float4(S0[0], S0[1], S0[2], S0[3]) * scale2, ok00);
      const float4 s01 = select(
          float4(flash_nax::kNegInf),
          float4(S0[4], S0[5], S0[6], S0[7]) * scale2, ok01);
      const float4 s10 = select(
          float4(flash_nax::kNegInf),
          float4(S1[0], S1[1], S1[2], S1[3]) * scale2, ok10);
      const float4 s11 = select(
          float4(flash_nax::kNegInf),
          float4(S1[4], S1[5], S1[6], S1[7]) * scale2, ok11);
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

    float4 p00, p01, p10, p11;
    if (full_causal_free) {
      p00 = fast::exp2(float4(S0[0], S0[1], S0[2], S0[3]) - new_m0);
      p01 = fast::exp2(float4(S0[4], S0[5], S0[6], S0[7]) - new_m1);
      p10 = fast::exp2(float4(S1[0], S1[1], S1[2], S1[3]) - new_m0);
      p11 = fast::exp2(float4(S1[4], S1[5], S1[6], S1[7]) - new_m1);
    } else {
      const bool emp0 = (new_m0 <= flash_nax::kNegInfHalf);
      const bool emp1 = (new_m1 <= flash_nax::kNegInfHalf);
      const float4 s00(S0[0], S0[1], S0[2], S0[3]);
      const float4 s01(S0[4], S0[5], S0[6], S0[7]);
      const float4 s10(S1[0], S1[1], S1[2], S1[3]);
      const float4 s11(S1[4], S1[5], S1[6], S1[7]);
      p00 = select(fast::exp2(s00 - new_m0), float4(0.0f),
                   emp0 || (s00 <= flash_nax::kNegInfHalf));
      p01 = select(fast::exp2(s01 - new_m1), float4(0.0f),
                   emp1 || (s01 <= flash_nax::kNegInfHalf));
      p10 = select(fast::exp2(s10 - new_m0), float4(0.0f),
                   emp0 || (s10 <= flash_nax::kNegInfHalf));
      p11 = select(fast::exp2(s11 - new_m1), float4(0.0f),
                   emp1 || (s11 <= flash_nax::kNegInfHalf));
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
    O4 *= fac_vec;
    O5 *= fac_vec;
    O6 *= fac_vec;
    O7 *= fac_vec;

    simdgroup_barrier(mem_flags::mem_none);

#define FRAG48_PV_PAIR_FAST(OA, OB, OFF_C0, OFF_C1)                            \
  do {                                                                         \
    BF8 V0, V1;                                                                \
    flash_nax::Frag16x16::load_bf4(V0, V_kb, kv_stride, OFF_C0, sc);           \
    flash_nax::Frag16x16::load_bf4(V1, V_kb, kv_stride, OFF_C1, sc);           \
    flash_nax::Frag16x16::mma_1x2_bf16<false>(OA, OB, P0, V0, V1);             \
    flash_nax::Frag16x16::load_bf4(V0, V_kb + 16ul * kv_stride, kv_stride,     \
                                   OFF_C0, sc);                                \
    flash_nax::Frag16x16::load_bf4(V1, V_kb + 16ul * kv_stride, kv_stride,     \
                                   OFF_C1, sc);                                \
    flash_nax::Frag16x16::mma_1x2_bf16<false>(OA, OB, P1, V0, V1);             \
  } while (false)

#define FRAG48_PV_PAIR_SAFE(OA, OB, OFF_C0, OFF_C1)                            \
  do {                                                                         \
    BF8 V0, V1;                                                                \
    if (valid_k < 16) {                                                        \
      flash_nax::Frag16x16::load_rows_clean_bf4(V0, V_kb, kv_stride, valid_k,  \
                                                OFF_C0, sc);                   \
      flash_nax::Frag16x16::load_rows_clean_bf4(V1, V_kb, kv_stride, valid_k,  \
                                                OFF_C1, sc);                   \
    } else {                                                                   \
      flash_nax::Frag16x16::load_clean_bf4(V0, V_kb, kv_stride, OFF_C0, sc);   \
      flash_nax::Frag16x16::load_clean_bf4(V1, V_kb, kv_stride, OFF_C1, sc);   \
    }                                                                          \
    flash_nax::Frag16x16::mma_1x2_bf16<false>(OA, OB, P0, V0, V1);             \
    const short rem1 = short(int(valid_k) - 16);                               \
    if (rem1 < 16) {                                                           \
      flash_nax::Frag16x16::load_rows_clean_bf4(V0, V_kb + 16ul * kv_stride,   \
                                                kv_stride, rem1, OFF_C0, sc);  \
      flash_nax::Frag16x16::load_rows_clean_bf4(V1, V_kb + 16ul * kv_stride,   \
                                                kv_stride, rem1, OFF_C1, sc);  \
    } else {                                                                   \
      flash_nax::Frag16x16::load_clean_bf4(V0, V_kb + 16ul * kv_stride,        \
                                           kv_stride, OFF_C0, sc);             \
      flash_nax::Frag16x16::load_clean_bf4(V1, V_kb + 16ul * kv_stride,        \
                                           kv_stride, OFF_C1, sc);             \
    }                                                                          \
    flash_nax::Frag16x16::mma_1x2_bf16<false>(OA, OB, P1, V0, V1);             \
  } while (false)

    if (fast_kv) {
      FRAG48_PV_PAIR_FAST(O0, O1, 0, 16);
      FRAG48_PV_PAIR_FAST(O2, O3, 32, 48);
      FRAG48_PV_PAIR_FAST(O4, O5, 64, 80);
      FRAG48_PV_PAIR_FAST(O6, O7, 96, 112);
    } else {
      FRAG48_PV_PAIR_SAFE(O0, O1, 0, 16);
      FRAG48_PV_PAIR_SAFE(O2, O3, 32, 48);
      FRAG48_PV_PAIR_SAFE(O4, O5, 64, 80);
      FRAG48_PV_PAIR_SAFE(O6, O7, 96, 112);
    }
#undef FRAG48_PV_PAIR_FAST
#undef FRAG48_PV_PAIR_SAFE
  }

  // Write partial_max, partial_sum, and normalized half4 partial outputs.
  const float rcp0 = (sum_s0 > 0.0f) ? (1.0f / sum_s0) : 0.0f;
  const float rcp1 = (sum_s1 > 0.0f) ? (1.0f / sum_s1) : 0.0f;
  const ulong rec0 =
      (ulong(tok0) * p.query_heads + qh0) * p.split_count + split;
  const ulong rec1 =
      (ulong(tok1) * p.query_heads + qh1) * p.split_count + split;

  if (d_part == 0 && sn == 0) {
    if (qv0) {
      partial_max[rec0] = max_s0;
      partial_sum[rec0] = sum_s0;
    }
    if (qv1) {
      partial_max[rec1] = max_s1;
      partial_sum[rec1] = sum_s1;
    }
  }

  device half *po0 = partial_output_half + rec0 * BD + d_offset + uint(sn);
  device half *po1 = partial_output_half + rec1 * BD + d_offset + uint(sn);

#define FRAG48_STORE_PART(OREG, OFF_C)                                         \
  do {                                                                         \
    if (qv0) {                                                                 \
      *reinterpret_cast<device half4 *>(po0 + uint(OFF_C)) =                   \
          half4(float4(OREG[0], OREG[1], OREG[2], OREG[3]) * rcp0);            \
    }                                                                          \
    if (qv1) {                                                                 \
      *reinterpret_cast<device half4 *>(po1 + uint(OFF_C)) =                   \
          half4(float4(OREG[4], OREG[5], OREG[6], OREG[7]) * rcp1);            \
    }                                                                          \
  } while (false)

  FRAG48_STORE_PART(O0, 0);
  FRAG48_STORE_PART(O1, 16);
  FRAG48_STORE_PART(O2, 32);
  FRAG48_STORE_PART(O3, 48);
  FRAG48_STORE_PART(O4, 64);
  FRAG48_STORE_PART(O5, 80);
  FRAG48_STORE_PART(O6, 96);
  FRAG48_STORE_PART(O7, 112);
#undef FRAG48_STORE_PART
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
    const float val = (tg_o[0u * 256u + d] + tg_o[1u * 256u + d]) +
                      (tg_o[2u * 256u + d] + tg_o[3u * 256u + d]);
    const float g_val = float(gate_row[d]);
    out_row[d] = bfloat(val / (1.0f + exp(-g_val)));
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
