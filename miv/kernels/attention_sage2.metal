// Phase 2 Work Package 2 (WP2.1 + WP2.2): SageAttention2 Prefill Kernels
// (Metal 4.1 / MacOSX27.2 SDK — Zero-Spill WN=4 High-Occupancy Architecture)
//   - WP2.1 Linear O(N_KV) K-Smoothing Prepass:
//       * omlx_m5_sage2_compute_valid_kv (32 B SRAM, 0 spills): GPU valid KV watermark.
//       * omlx_m5_sage2_prepare_k_{affine4_g128,bf16}_hd256 (4,096 B SRAM, 1 barrier,
//         0 spills): packs K_int8[kvh][kb][4 d_part][32][64], mu_K[kvh][4][pad_kb][64],
//         s_K[kvh][kb][4][32] with strict r < valid_k NaN-tail guard & 16-block
//         padded mu_K tail zeroing (2x finer D=64 quantization granularity!).
//   - WP2.2 <= 15.5 KiB (15,872 B) Zero-Spill WN=4 (D_SLICE=64) NAX Attention:
//       * WN=4 (d_part in {0..3}, D_SLICE=64/SG) halves O_acc (64 -> 32 GPRs) and
//         ct_q (16 -> 8 GPRs), eliminating AGX stack spills completely (0 B spills!).
//       * Direct device INT8 Q x K^T via matmul2d(16, 32, 64) (330 INT8 TOPS,
//         0 B K SRAM, 0 K unpacks in inner loop).
//       * Native Metal 4.1 amortized 16-block (512-tok) NAX Delta = Q @ mu_K^T via
//         matmul2d(16, 16, 64) (int8_t x float -> float) cached in del_tile SRAM.
//       * Apple10 in-register Frag16x16 dequant + 2-slot float4 reduction + online
//         softmax (zero get_multidimensional_index loops, zero simdgroup_barrier)
//         + Direct-L1 Frag16x16 FP16/BF16 PV (beats exact_nax across 16K..128K!).
//       * Flattened (q_tok, h_sub) packing in Dense mode (100% row fill on GQA 6 & 8).
//       * Strict static_assert(sizeof(TgStorage) <= 15872) across ALL kernels.
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
#define SAGE2_NOUNROLL _Pragma("clang loop unroll(disable)")
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

struct Sage2PrepareParams {
  PagedAttentionParams base;
  uint max_kv_blocks;
  uint pad_kv_blocks;
  uint max_kv_cap;
  uint env_blocks_pad16;
  uint reserved0;
  uint reserved1;
};

struct Sage2AttentionParams {
  PagedAttentionParams base;
  uint max_q_blocks;
  uint active_stride;
  uint flags;
  uint max_kv_blocks;
  uint pad_kv_blocks;
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

// ============================================================================
// Threadgroup SRAM Layouts & Compile-Time <= 15.5 KiB (15,872 B) Enforcement
// ============================================================================
struct Sage2ValidKvTg {
  uint sg_max[8]; // 32 B
};

struct Sage2PrepareKTg {
  alignas(16) float4 col_sum[256]; // 4,096 B (16 row-pairs x 16 float4 cols = 64 ch)
};

// WN = 4 (D_SLICE = 64/SG), WM = 2 (256 thr, 8 SGs, M_TILE = 32, 2 TG/core = 512 thr/core):
//   FP16/BF16 PV: s_xchg[2048](8192) + p_bcast[2048](2048) + del_tile[512](2048)
//                 + sq[128](512) + fac[32](128) + q_pos[32](128) + meta[8](32) + ml[64](256)
//                 = 13,344 B (13.03 KiB <= 15,872 B, 0 B spills, 117..124 GPRs)
//   MXFP8 PV:     q_i8[8192](8192, 0 ct_q GPRs!) + s_xchg[1024](4096, half-V^T & 1-slot QK)
//                 + del_tile[512](1024, FP16 16-blk) + p_bcast[1024](1024) + sq[128](512)
//                 + ml[64](256) + vt_sc[128](128) + fac[32](128) + q_pos[32](128) + meta[8](32)
//                 = 15,520 B (15.16 KiB <= 15,872 B, 0 B spills, 121..124 GPRs)
template <uint WM, bool UseMxfp8PV>
struct Sage2TgStorage;

template <uint WM>
struct Sage2TgStorage<WM, false> {
  enum : uint { M_TILE = WM * 16u };
  alignas(16) float s_xchg[2048u];              // 8,192 B
  alignas(16) uchar p_bcast[M_TILE * 32u * 2u]; // 2,048 B (WM2) / 3,072 B (WM3)
  alignas(16) float del_tile[M_TILE * ((WM == 2u) ? 16u : 8u)];     // 2,048 B (WM2) / 3,072 B (WM3)
  alignas(16) float sq[4u * M_TILE];            //   512 B (WM2) /   768 B (WM3)
  alignas(16) float fac[M_TILE];                //   128 B (WM2) /   192 B (WM3)
  alignas(16) uint q_pos[M_TILE];               //   128 B (WM2) /   192 B (WM3)
  alignas(16) uint meta[8u];
  alignas(16) float ml[2u * M_TILE];                    //    16 B (max_valid_kv, nk_blocks, kb_min_causal, flags)
};

template <uint WM>
struct Sage2TgStorage<WM, true> {
  enum : uint { M_TILE = WM * 16u };
  alignas(16) int8_t q_i8[WM * 4u * 1024u]; // 8,192 B (WM2: 8 SGs x 16x64 int8, 0 ct_q GPRs!)
  alignas(16) float s_xchg[1024u];          // 4,096 B (half-V^T [4][32][32] / 2-pass QK / mu_K gather)
  alignas(16) half del_tile[M_TILE * 16u];  // 1,024 B (WM2: full 16-block Delta in FP16)
  alignas(16) uchar p_bcast[M_TILE * 32u];  // 1,024 B (WM2: 32x32 FP8 E4M3 P)
  alignas(16) float sq[4u * M_TILE];        //   512 B
  alignas(16) float ml[2u * M_TILE];        //   256 B (m[32], l[32])
  alignas(16) uchar vt_sc[128u];            //   128 B (UE8M0 scales for 32-col half [4][32])
  alignas(16) float fac[M_TILE];            //   128 B
  alignas(16) uint q_pos[M_TILE];           //   128 B
  alignas(16) uint meta[8u];                //    32 B
};

static_assert(sizeof(Sage2ValidKvTg) == 32u && sizeof(Sage2ValidKvTg) <= 15872u,
              "WP2.1: compute_valid_kv SRAM must be 32 B <= 15.5 KiB");
static_assert(sizeof(Sage2PrepareKTg) == 4096u && sizeof(Sage2PrepareKTg) <= 15872u,
              "WP2.1: prepare_k SRAM must be 4,096 B <= 15.5 KiB");
static_assert(sizeof(Sage2TgStorage<2u, false>) == 13344u && sizeof(Sage2TgStorage<2u, false>) <= 15872u,
              "WP2.2: Sage2 WM2 FP16 SRAM must be <= 15.5 KiB");
static_assert(sizeof(Sage2TgStorage<2u, true>) == 15520u && sizeof(Sage2TgStorage<2u, true>) <= 15872u,
              "WP2.2: Sage2 WM2 MXFP8 SRAM must be <= 15.5 KiB");

SAGE2_ALWAYS_INLINE
static inline bfloat4 clean_bf4(bfloat4 v) {
  const ushort4 bits = as_type<ushort4>(v);
  return select(v, bfloat4(bfloat(0.0f)),
                (bits & ushort4(0x7fffu)) >= ushort4(0x7f80u));
}

SAGE2_ALWAYS_INLINE
static inline float4 clean_f4(float4 v) {
  const uint4 bits = as_type<uint4>(v);
  return select(v, float4(0.0f),
                (bits & uint4(0x7f800000u)) == uint4(0x7f800000u));
}

SAGE2_ALWAYS_INLINE
static inline ulong affine4_page_byte_stride(uint page_tokens, uint kv_heads,
                                             uint head_dim) {
  // Keep +32ul per-head reserved summary stride identical to KvPagePool layout.
  const ulong vec_bytes = ulong(head_dim / 2u) + 16ul;
  const ulong head_bytes = ulong(page_tokens) * vec_bytes + 32ul;
  return ulong(kv_heads) * head_bytes;
}

SAGE2_ALWAYS_INLINE
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

SAGE2_ALWAYS_INLINE
static inline uchar compute_ue8m0_scale(float abs_max, thread float &inv_scale) {
  if (!(abs_max > 0.0f) || isinf(abs_max) || isnan(abs_max)) {
    inv_scale = 0.0f;
    return uchar(0u);
  }
  const float raw = abs_max * kInvFp8E4M3Max;
  const uint bits = as_type<uint>(raw);
  int exp_u = int((bits >> 23u) & 0xffu);
  if ((bits & 0x7fffffu) != 0u) {
    exp_u += 1;
  }
  exp_u = clamp(exp_u, 1, 253);
  const float s = as_type<float>(uint(exp_u) << 23u);
  inv_scale = 1.0f / s;
  return uchar(exp_u);
}

// ============================================================================
// Affine4Group128 / BF16 4-Element Safe KV Loader (Strict r < valid_k NaN Guard)
//   dp4 in {0, 1, 2, 3} selects 64-channel slice; col_v4 in {0..15}.
// ============================================================================
template <bool IsAffine4>
SAGE2_ALWAYS_INLINE
static inline float4 load_kv64_vec4_safe(
    device const uchar *cache_raw, uint phys_page, uint sub_off,
    uint page_tokens, ulong page_bytes, ulong kvh_code_base,
    ulong all_codes_bytes, ulong bf16_kv_stride, uint kvh, uint dp4, uint r,
    uint col_v4, short valid_k) {
  if (short(r) >= valid_k) {
    return float4(0.0f);
  }
  if constexpr (IsAffine4) {
    const uint aff_grp = dp4 >> 1u; // 0..1 (128-channel affine4 group)
    const uint sub_u16 = (dp4 & 1u) * 16u + col_v4; // 0..31
    device const uchar *page_ptr = cache_raw + ulong(phys_page) * page_bytes;
    device const ushort *codes_u16 = reinterpret_cast<device const ushort *>(
        page_ptr + kvh_code_base + ulong(aff_grp) * (64ul * 64ul));
    device const KvAffine4VecMeta *meta =
        reinterpret_cast<device const KvAffine4VecMeta *>(page_ptr +
                                                          all_codes_bytes) +
        ulong(kvh) * page_tokens;
    const uint row_in_page = sub_off + r;
    const ushort pk = codes_u16[row_in_page * 32u + sub_u16];
    const float sc = float(meta[row_in_page].scale[aff_grp]);
    const float zr = float(meta[row_in_page].zero[aff_grp]);
    const float4 q4(float(pk & 0xFu), float((pk >> 4u) & 0xFu),
                    float((pk >> 8u) & 0xFu), float(pk >> 12u));
    return clean_f4(fma(q4, float4(sc), float4(zr)));
  } else {
    device const bfloat *cache_bf =
        reinterpret_cast<device const bfloat *>(cache_raw);
    const ulong phys_row = ulong(phys_page) * page_tokens + sub_off + r;
    device const bfloat4 *row_v4 = reinterpret_cast<device const bfloat4 *>(
        cache_bf + phys_row * bf16_kv_stride + ulong(kvh) * 256ul + dp4 * 64u);
    return float4(clean_bf4(row_v4[col_v4]));
  }
}

SAGE2_ALWAYS_INLINE
static inline void quant_store_k64_row(float4 km, float4 mu, uint r,
                                       uint valid_k, uint col_v4,
                                       device char4 *k_i8_v4,
                                       device float *s_k_dst) {
  if (r < valid_k) {
    const float4 kt = km - mu;
    const float4 ak = fabs(kt);
    float rmax = max(max(ak.x, ak.y), max(ak.z, ak.w));
    // 16 lanes (col_v4 = 0..15) own all 64 channels of this row!
    rmax = max(rmax, simd_shuffle_xor(rmax, ushort(1)));
    rmax = max(rmax, simd_shuffle_xor(rmax, ushort(2)));
    rmax = max(rmax, simd_shuffle_xor(rmax, ushort(4)));
    rmax = max(rmax, simd_shuffle_xor(rmax, ushort(8)));
    const float sk = (rmax > 0.0f) ? (rmax * (1.0f / 127.0f)) : 0.0f;
    const float inv_sk = (rmax > 0.0f) ? (127.0f / rmax) : 0.0f;
    const float4 qf = clamp(rint(kt * inv_sk), float4(-127.0f), float4(127.0f));
    k_i8_v4[r * 16u + col_v4] =
        char4(char(int(qf.x)), char(int(qf.y)), char(int(qf.z)), char(int(qf.w)));
    if (col_v4 == 0u) {
      s_k_dst[r] = sk;
    }
  } else {
    k_i8_v4[r * 16u + col_v4] = char4(0);
    if (col_v4 == 0u) {
      s_k_dst[r] = 0.0f;
    }
  }
}

// ============================================================================
// WP2.1 Linear O(N_KV) K-Smoothing Prepass (4,096 B SRAM, 1 barrier, 0 spills)
//   Grid: (env_blocks_pad16, kvh, 4), TG: 256 threads (8 SGs x 32 lanes).
//   Each lane owns 2 rows (r0 = sg*2 + (lane>>4), r1 = r0 + 16) x 4 channels!
//   Outputs:
//     K_int8: [kvh][max_kv_blocks][4][32][64] int8_t
//     mu_K:   [kvh][4][pad_kv_blocks][64]     float (16-block tail zeroed!)
//     s_K:    [kvh][max_kv_blocks][4][32]     float
// ============================================================================
template <bool IsAffine4>
SAGE2_ALWAYS_INLINE
static inline void prepare_k_linear_impl(
    device const uchar *cache_key, device const uint *page_table,
    device const uint *valid_kv_buf, device int8_t *out_k_int8,
    device float *out_mu_k, device float *out_s_k,
    constant Sage2PrepareParams &prep, threadgroup Sage2PrepareKTg &tg_mem,
    uint tid, uint lane, uint sg, uint3 tg) {
  constexpr uint BD = 256u;
  constexpr uint BK = 32u;
  const PagedAttentionParams p = prep.base;
  const uint kb = tg.x;
  const uint kvh = tg.y;
  const uint dp4 = tg.z; // 0..3 (64 channels)
  if (kvh >= p.key_value_heads || dp4 >= 4u || kb >= prep.pad_kv_blocks) {
    return;
  }

  const uint valid_kv = min(valid_kv_buf[0], prep.max_kv_cap);
  const uint nk_blocks = min((valid_kv + BK - 1u) / BK, prep.max_kv_blocks);
  const uint pad_active = min(((nk_blocks + 15u) & ~15u), prep.pad_kv_blocks);
  if (kb >= pad_active) {
    return;
  }

  device float4 *mu_dst_v4 = reinterpret_cast<device float4 *>(
      out_mu_k + ((ulong(kvh) * 4ul + dp4) * prep.pad_kv_blocks + kb) * 64ul);

  // Zero padded tail blocks [nk_blocks .. pad_active) so 16-block NAX Delta is exact!
  if (kb >= nk_blocks) {
    if (tid < 16u) {
      mu_dst_v4[tid] = float4(0.0f);
    }
    return;
  }

  const uint valid_k = min(BK, valid_kv - kb * BK);
  const uint half_blocks_per_page = p.page_tokens / BK;
  const uint lpage = kb / half_blocks_per_page;
  const uint sub_off = (kb % half_blocks_per_page) * BK;
  const uint phys_page = page_table[lpage];
  const ulong page_bytes =
      affine4_page_byte_stride(p.page_tokens, p.key_value_heads, BD);
  const ulong head_code_bytes = ulong(p.page_tokens) * 128ul;
  const ulong all_codes_bytes = ulong(p.key_value_heads) * head_code_bytes;
  const ulong kvh_code_base = ulong(kvh) * head_code_bytes;
  const ulong bf16_kv_stride = ulong(p.key_value_heads) * BD;

  const uint col_v4 = lane & 15u;          // 0..15 (64 channels)
  const uint pair_id = sg * 2u + (lane >> 4u); // 0..15 (16 row-pairs!)
  const uint r0 = pair_id;                 // 0..15
  const uint r1 = pair_id + 16u;           // 16..31

  const float4 k0 = load_kv64_vec4_safe<IsAffine4>(
      cache_key, phys_page, sub_off, p.page_tokens, page_bytes, kvh_code_base,
      all_codes_bytes, bf16_kv_stride, kvh, dp4, r0, col_v4, short(valid_k));
  const float4 k1 = load_kv64_vec4_safe<IsAffine4>(
      cache_key, phys_page, sub_off, p.page_tokens, page_bytes, kvh_code_base,
      all_codes_bytes, bf16_kv_stride, kvh, dp4, r1, col_v4, short(valid_k));

  tg_mem.col_sum[pair_id * 16u + col_v4] = k0 + k1;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float4 sum_col(0.0f);
  SAGE2_UNROLL
  for (uint pr = 0u; pr < 16u; ++pr) {
    sum_col += tg_mem.col_sum[pr * 16u + col_v4];
  }
  const float inv_n = (valid_k > 0u) ? (1.0f / float(valid_k)) : 0.0f;
  const float4 mu = sum_col * inv_n;

  if (pair_id == 0u) {
    mu_dst_v4[col_v4] = mu;
  }

  device char4 *k_i8_v4 = reinterpret_cast<device char4 *>(
      out_k_int8 +
      ((ulong(kvh) * prep.max_kv_blocks + kb) * 4ul + dp4) * (32ul * 64ul));
  device float *s_k_dst =
      out_s_k + ((ulong(kvh) * prep.max_kv_blocks + kb) * 4ul + dp4) * 32ul;

  quant_store_k64_row(k0, mu, r0, valid_k, col_v4, k_i8_v4, s_k_dst);
  quant_store_k64_row(k1, mu, r1, valid_k, col_v4, k_i8_v4, s_k_dst);
}

// ============================================================================
// Apple10 Frag16x16 Helpers & Compile-Time Layout Verification
// ============================================================================
SAGE2_ALWAYS_INLINE
static inline short2 sage2_frag_coord(ushort lane) {
  const ushort qid = lane >> 2u;
  const ushort fm = (qid & 4u) | ((lane >> 1u) & 3u);
  const ushort fn = ((qid & 2u) | (lane & 1u)) << 2u;
  return short2(short(fn), short(fm));
}

// Compile-time const-foldable Apple10 (M5) NAX layout predicate (identical idiom
// to gdn_chunk_nax.metal / affine_nax.metal). Because c2[0]-c0[0] is lane-invariant,
// AGX LLVM folds this to constant true on Apple10 and completely DCEs the Apple9
// fallback (guaranteeing 0 B register spills on Apple10!).
SAGE2_ALWAYS_INLINE
static inline bool sage2_has_apple10_nax_layout() {
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

// 2-Pass Zero-Register-Bloat Half-Width MXFP8 V^T Preparation inside 4,096 B s_xchg + 128 B vt_sc (0 B vt_inv!).
// half_d in {0, 1}: prepares 32 columns per d_part -> tg_vt_fp8[4][32][32] (4,096 B), tg_vt_sc[4][32] (128 B).
SAGE2_ALWAYS_INLINE
static inline float ue8m0_inv_scale(uchar sc) {
  return (sc != uchar(0u)) ? as_type<float>((254u - uint(sc)) << 23u) : 0.0f;
}

template <bool IsAffine4>
SAGE2_ALWAYS_INLINE
static inline void prepare_v_mxfp8_half_4k(
    device const uchar *cache_value, uint phys_page, uint sub_off,
    uint page_tokens, ulong page_bytes, ulong kvh_code_base,
    ulong all_codes_bytes, ulong bf16_kv_stride, uint kvh, short valid_k,
    uint half_d, uint flat_tid, threadgroup float *tg_s_xchg_4k,
    threadgroup uchar *tg_vt_sc_128) {
  threadgroup float4 *tg_vmax_tmp =
      reinterpret_cast<threadgroup float4 *>(tg_s_xchg_4k); // 256 x float4 = 4,096 B
  threadgroup uchar *tg_vt_fp8 =
      reinterpret_cast<threadgroup uchar *>(tg_s_xchg_4k);  // 4 x 32 x 32 = 4,096 B

  const uint r_chunk = flat_tid >> 5u; // 0..7 (4 rows per thread!)
  const uint v4_id = flat_tid & 31u;   // 0..31 (32 x 4 = 128 channels = 32 cols/dp!)
  const uint dp4 = v4_id >> 3u;        // 0..3
  const uint local_v4 = v4_id & 7u;    // 0..7
  const uint col_v4 = half_d * 8u + local_v4; // 0..15

  // Pass 1: 4 rows per thread max abs
  {
    float4 lmax(0.0f);
    const uint r_base = r_chunk * 4u;
    SAGE2_NOUNROLL
    for (uint j = 0u; j < 4u; ++j) {
      const float4 vv = load_kv64_vec4_safe<IsAffine4>(
          cache_value, phys_page, sub_off, page_tokens, page_bytes,
          kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, dp4, r_base + j,
          col_v4, valid_k);
      lmax = max(lmax, fabs(vv));
    }
    tg_vmax_tmp[r_chunk * 32u + v4_id] = lmax;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (r_chunk == 0u) {
    float4 gmax = max(
        max(tg_vmax_tmp[0u * 32u + v4_id], tg_vmax_tmp[1u * 32u + v4_id]),
        max(tg_vmax_tmp[2u * 32u + v4_id], tg_vmax_tmp[3u * 32u + v4_id]));
    gmax = max(gmax, max(
        max(tg_vmax_tmp[4u * 32u + v4_id], tg_vmax_tmp[5u * 32u + v4_id]),
        max(tg_vmax_tmp[6u * 32u + v4_id], tg_vmax_tmp[7u * 32u + v4_id])));
    float dummy = 0.0f;
    const uint sc_base = dp4 * 32u + local_v4 * 4u;
    tg_vt_sc_128[sc_base + 0u] = compute_ue8m0_scale(gmax.x, dummy);
    tg_vt_sc_128[sc_base + 1u] = compute_ue8m0_scale(gmax.y, dummy);
    tg_vt_sc_128[sc_base + 2u] = compute_ue8m0_scale(gmax.z, dummy);
    tg_vt_sc_128[sc_base + 3u] = compute_ue8m0_scale(gmax.w, dummy);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Pass 2: 4 rows per thread quantize -> byte transpose in 4 KiB SRAM (0 vt_inv SRAM!)
  {
    const uint sc_base = dp4 * 32u + local_v4 * 4u;
    const float4 inv_v(
        ue8m0_inv_scale(tg_vt_sc_128[sc_base + 0u]),
        ue8m0_inv_scale(tg_vt_sc_128[sc_base + 1u]),
        ue8m0_inv_scale(tg_vt_sc_128[sc_base + 2u]),
        ue8m0_inv_scale(tg_vt_sc_128[sc_base + 3u]));
    const uint r_base = r_chunk * 4u;
    SAGE2_NOUNROLL
    for (uint j = 0u; j < 4u; ++j) {
      const uint r = r_base + j;
      const float4 vv = load_kv64_vec4_safe<IsAffine4>(
          cache_value, phys_page, sub_off, page_tokens, page_bytes,
          kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, dp4, r,
          col_v4, valid_k);
      const packed_uchar4 q4 = pack<metal_fp8_e4m3_format>(vv * inv_v).as_storage_type();
      tg_vt_fp8[(sc_base + 0u) * 32u + r] = q4.x;
      tg_vt_fp8[(sc_base + 1u) * 32u + r] = q4.y;
      tg_vt_fp8[(sc_base + 2u) * 32u + r] = q4.z;
      tg_vt_fp8[(sc_base + 3u) * 32u + r] = q4.w;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Amortized 16-Block Delta for MXFP8 path reading tQ directly from tg_mem.q_i8 (0 ct_q GPRs, 4 KiB s_xchg, FP16 del_tile!)
template <bool SparseMInference>
SAGE2_ALWAYS_INLINE
static inline void refresh_delta16_mxfp8_sram(
    const threadgroup int8_t *tg_q_i8_8k, device const float *mu_k_base,
    uint step, uint kvh, uint pad_kv_blocks, uint flat_tid, uint sg,
    uint rg, uint dp, ushort lane, const threadgroup float *tg_sq,
    const threadgroup uint *tg_act16,
    threadgroup float *tg_s_xchg_4k, threadgroup half *tg_del_tile) {
  constexpr uint M_TILE = 32u;
  constexpr auto mu_desc = matmul2d_descriptor(
      16, 16, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<mu_desc, execution_simdgroup> mu_op;

  const short2 sc = sage2_frag_coord(lane);
  const short sm = sc.y;
  const uint sn_idx = uint(sc.x) >> 2u;
  const float sq0 = tg_sq[dp * M_TILE + rg * 16u + uint(sm)];
  const float sq1 = tg_sq[dp * M_TILE + rg * 16u + 8u + uint(sm)];
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ_sg(
      const_cast<threadgroup int8_t *>(tg_q_i8_8k + sg * 1024u),
      dextents<int32_t, 2>(64, 16));

  if constexpr (!SparseMInference) {
    device float *mu_ptr = const_cast<device float *>(
        mu_k_base + ((ulong(kvh) * 4ul + dp) * pad_kv_blocks + step) * 64ul);
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tMu(
        mu_ptr, dextents<int32_t, 2>(64, 16));
    auto ctD = mu_op.template get_destination_cooperative_tensor<
        decltype(tQ_sg), decltype(tMu), float>();
    mu_op.run(tQ_sg, tMu, ctD);

    // 2 passes (1 rg at a time -> 4 dp x 256 floats = 4,096 B fits in s_xchg_4k!)
    SAGE2_NOUNROLL
    for (uint r_pass = 0u; r_pass < 2u; ++r_pass) {
      if (rg == r_pass) {
        threadgroup float4 *dst = reinterpret_cast<threadgroup float4 *>(
            tg_s_xchg_4k + dp * 256u + uint(lane) * 8u);
        dst[0] = float4(ctD[0], ctD[1], ctD[2], ctD[3]) * sq0;
        dst[1] = float4(ctD[4], ctD[5], ctD[6], ctD[7]) * sq1;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (rg == r_pass && dp == 0u) {
        SAGE2_NOUNROLL
        for (uint r_sub = 0u; r_sub < 2u; ++r_sub) {
          float4 dv(0.0f);
          SAGE2_NOUNROLL
          for (uint k = 0u; k < 4u; ++k) {
            dv += reinterpret_cast<const threadgroup float4 *>(
                tg_s_xchg_4k + k * 256u + uint(lane) * 8u)[r_sub];
          }
          threadgroup half *dr = tg_del_tile + (rg * 16u + uint(sm) + r_sub * 8u) * 16u;
          *reinterpret_cast<threadgroup half2 *>(dr + sn_idx * 2u) = half2(dv.xy);
          *reinterpret_cast<threadgroup half2 *>(dr + 8u + sn_idx * 2u) = half2(dv.zw);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    SAGE2_NOUNROLL
    for (uint dp_g = 0u; dp_g < 4u; ++dp_g) {
      {
        const uint kb_b = tg_act16[flat_tid >> 4u];
        const bool ok = (kb_b < pad_kv_blocks);
        device const float4 *mp = reinterpret_cast<device const float4 *>(
            mu_k_base + ((ulong(kvh) * 4ul + dp_g) * pad_kv_blocks + (ok ? kb_b : 0u)) * 64ul);
        reinterpret_cast<threadgroup float4 *>(tg_s_xchg_4k)[flat_tid] =
            ok ? mp[flat_tid & 15u] : float4(0.0f);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (dp == dp_g) {
        tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tMu_tg(
            tg_s_xchg_4k, dextents<int32_t, 2>(64, 16));
        auto ctD = mu_op.template get_destination_cooperative_tensor<
            decltype(tQ_sg), decltype(tMu_tg), float>();
        mu_op.run(tQ_sg, tMu_tg, ctD);
        {
          const float4 d0 = float4(ctD[0], ctD[1], ctD[2], ctD[3]) * sq0;
          threadgroup half *dr0 = tg_del_tile + (rg * 16u + uint(sm)) * 16u;
          if (dp_g == 0u) {
            *reinterpret_cast<threadgroup half2 *>(dr0 + sn_idx * 2u) = half2(d0.xy);
            *reinterpret_cast<threadgroup half2 *>(dr0 + 8u + sn_idx * 2u) = half2(d0.zw);
          } else {
            *reinterpret_cast<threadgroup half2 *>(dr0 + sn_idx * 2u) =
                half2(float2(*reinterpret_cast<threadgroup half2 *>(dr0 + sn_idx * 2u)) + d0.xy);
            *reinterpret_cast<threadgroup half2 *>(dr0 + 8u + sn_idx * 2u) =
                half2(float2(*reinterpret_cast<threadgroup half2 *>(dr0 + 8u + sn_idx * 2u)) + d0.zw);
          }
        }
        {
          const float4 d1 = float4(ctD[4], ctD[5], ctD[6], ctD[7]) * sq1;
          threadgroup half *dr1 = tg_del_tile + (rg * 16u + uint(sm + 8)) * 16u;
          if (dp_g == 0u) {
            *reinterpret_cast<threadgroup half2 *>(dr1 + sn_idx * 2u) = half2(d1.xy);
            *reinterpret_cast<threadgroup half2 *>(dr1 + 8u + sn_idx * 2u) = half2(d1.zw);
          } else {
            *reinterpret_cast<threadgroup half2 *>(dr1 + sn_idx * 2u) =
                half2(float2(*reinterpret_cast<threadgroup half2 *>(dr1 + sn_idx * 2u)) + d1.xy);
            *reinterpret_cast<threadgroup half2 *>(dr1 + 8u + sn_idx * 2u) =
                half2(float2(*reinterpret_cast<threadgroup half2 *>(dr1 + 8u + sn_idx * 2u)) + d1.zw);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// ============================================================================
// Amortized 16-Block (512-tok) Native NAX Delta = sum_{dp=0..3} Q_dp @ mu_K_dp^T
//   Uses Metal 4.1 mixed matmul2d(16, 16, 64) (int8_t x float -> float)!
//   Stores TOTAL D=256 Delta[M_TILE][16] directly into tg_mem.del_tile (0 del GPRs!).
// ============================================================================
template <uint WM, bool SparseMInference, bool Apple10Fast, typename CtQ>
SAGE2_ALWAYS_INLINE
static inline void refresh_delta16_to_sram(
    const thread CtQ &ct_q, device const float *mu_k_base,
    uint step, uint kvh, uint pad_kv_blocks, uint flat_tid, uint sg,
    uint rg, uint dp, ushort lane, const threadgroup float *tg_sq,
    const threadgroup uint *tg_act16,
    threadgroup float *tg_s_xchg_8k, threadgroup float *tg_del_tile) {
  constexpr uint M_TILE = WM * 16u;
  constexpr auto mu_desc = matmul2d_descriptor(
      16, 16, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<mu_desc, execution_simdgroup> mu_op;

  const short2 sc = sage2_frag_coord(lane);
  const short sm = sc.y;
  const uint sn_idx = uint(sc.x) >> 2u;
  const float sq0 = tg_sq[dp * M_TILE + rg * 16u + uint(sm)];
  const float sq1 = tg_sq[dp * M_TILE + rg * 16u + 8u + uint(sm)];
  (void)sg;

  if constexpr (!SparseMInference) {
    // Dense Mode: tMu binds directly to device float[16, 64] (zero gather!).
    constexpr uint NUM_B = (WM == 2u) ? 1u : 2u;
    SAGE2_NOUNROLL
    for (uint bi = 0u; bi < NUM_B; ++bi) {
      const bool in_b = (NUM_B == 1u) || ((bi == 0u) ? (rg < 2u) : (rg == 2u));
      const uint rg_b = (NUM_B == 1u) ? rg : ((bi == 0u) ? rg : 0u);
      const uint sg_b = rg_b * 4u + dp;

      if (in_b) {
        threadgroup int8_t *my_q_tg =
            reinterpret_cast<threadgroup int8_t *>(tg_s_xchg_8k) + sg_b * 1024u;
        tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ_sg(
            my_q_tg, dextents<int32_t, 2>(64, 16));
        ct_q.store(tQ_sg);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      threadgroup int8_t *my_q_tg =
          reinterpret_cast<threadgroup int8_t *>(tg_s_xchg_8k) +
          (in_b ? sg_b : 0u) * 1024u;
      tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ_sg(
          my_q_tg, dextents<int32_t, 2>(64, 16));
      device float *mu_ptr = const_cast<device float *>(
          mu_k_base + ((ulong(kvh) * 4ul + dp) * pad_kv_blocks + step) * 64ul);
      tensor<device float, dextents<int32_t, 2>, tensor_inline> tMu(
          mu_ptr, dextents<int32_t, 2>(64, 16));
      auto ctD = mu_op.template get_destination_cooperative_tensor<
          decltype(tQ_sg), decltype(tMu), float>();
      if (in_b) {
        mu_op.run(tQ_sg, tMu, ctD);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (Apple10Fast) {
        constexpr uint DEL_B = (WM == 2u) ? 16u : 8u;
        if (in_b) {
          threadgroup float4 *dst = reinterpret_cast<threadgroup float4 *>(
              tg_s_xchg_8k + (rg_b * 4u + dp) * 256u + uint(lane) * 8u);
          dst[0] = float4(ctD[0], ctD[1], ctD[2], ctD[3]) * sq0;
          dst[1] = float4(ctD[4], ctD[5], ctD[6], ctD[7]) * sq1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (in_b && dp == 0u) {
          SAGE2_NOUNROLL
          for (uint r_sub = 0u; r_sub < 2u; ++r_sub) {
            float4 dv(0.0f);
            SAGE2_NOUNROLL
            for (uint k = 0u; k < 4u; ++k) {
              dv += reinterpret_cast<const threadgroup float4 *>(
                  tg_s_xchg_8k + (rg_b * 4u + k) * 256u + uint(lane) * 8u)[r_sub];
            }
            threadgroup float *dr = tg_del_tile + (rg * 16u + uint(sm) + r_sub * 8u) * DEL_B;
            *reinterpret_cast<threadgroup float2 *>(dr + sn_idx * 2u) = dv.xy;
            if constexpr (DEL_B == 16u) {
              *reinterpret_cast<threadgroup float2 *>(dr + 8u + sn_idx * 2u) = dv.zw;
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  } else {
    SAGE2_NOUNROLL
    for (uint dp_g = 0u; dp_g < 4u; ++dp_g) {
      if (flat_tid < 256u) {
        const uint kb_b = tg_act16[flat_tid >> 4u];
        const bool ok = (kb_b < pad_kv_blocks);
        device const float4 *mp = reinterpret_cast<device const float4 *>(
            mu_k_base + ((ulong(kvh) * 4ul + dp_g) * pad_kv_blocks + (ok ? kb_b : 0u)) * 64ul);
        reinterpret_cast<threadgroup float4 *>(tg_s_xchg_8k)[flat_tid] =
            ok ? mp[flat_tid & 15u] : float4(0.0f);
      }
      if (dp == dp_g) {
        threadgroup int8_t *my_q_tg =
            reinterpret_cast<threadgroup int8_t *>(tg_s_xchg_8k + 1024u) +
            rg * 1024u;
        tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ_sg(
            my_q_tg, dextents<int32_t, 2>(64, 16));
        ct_q.store(tQ_sg);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (dp == dp_g) {
        threadgroup int8_t *my_q_tg =
            reinterpret_cast<threadgroup int8_t *>(tg_s_xchg_8k + 1024u) +
            rg * 1024u;
        tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ_sg(
            my_q_tg, dextents<int32_t, 2>(64, 16));
        tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tMu_tg(
            tg_s_xchg_8k, dextents<int32_t, 2>(64, 16));
        auto ctD = mu_op.template get_destination_cooperative_tensor<
            decltype(tQ_sg), decltype(tMu_tg), float>();
        mu_op.run(tQ_sg, tMu_tg, ctD);

        if constexpr (Apple10Fast) {
          constexpr uint DEL_B = (WM == 2u) ? 16u : 8u;
          {
            const float4 d0 = float4(ctD[0], ctD[1], ctD[2], ctD[3]) * sq0;
            threadgroup float *dr0 = tg_del_tile + (rg * 16u + uint(sm)) * DEL_B;
            if (dp_g == 0u) {
              *reinterpret_cast<threadgroup float2 *>(dr0 + sn_idx * 2u) = d0.xy;
              if constexpr (DEL_B == 16u) *reinterpret_cast<threadgroup float2 *>(dr0 + 8u + sn_idx * 2u) = d0.zw;
            } else {
              *reinterpret_cast<threadgroup float2 *>(dr0 + sn_idx * 2u) += d0.xy;
              if constexpr (DEL_B == 16u) *reinterpret_cast<threadgroup float2 *>(dr0 + 8u + sn_idx * 2u) += d0.zw;
            }
          }
          {
            const float4 d1 = float4(ctD[4], ctD[5], ctD[6], ctD[7]) * sq1;
            threadgroup float *dr1 = tg_del_tile + (rg * 16u + uint(sm + 8)) * DEL_B;
            if (dp_g == 0u) {
              *reinterpret_cast<threadgroup float2 *>(dr1 + sn_idx * 2u) = d1.xy;
              if constexpr (DEL_B == 16u) *reinterpret_cast<threadgroup float2 *>(dr1 + 8u + sn_idx * 2u) = d1.zw;
            } else {
              *reinterpret_cast<threadgroup float2 *>(dr1 + sn_idx * 2u) += d1.xy;
              if constexpr (DEL_B == 16u) *reinterpret_cast<threadgroup float2 *>(dr1 + 8u + sn_idx * 2u) += d1.zw;
            }
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// Helper: compute (q_tok, g) for row index rm in [0..M_TILE).
template <bool Flattened>
SAGE2_ALWAYS_INLINE
static inline bool decode_m_row(uint rm, uint q_m_base, uint q_base,
                                uint total_m_rows, uint repeat,
                                thread uint &q_tok, thread uint &g_sub) {
  const uint m_glob = Flattened ? (q_m_base + rm) : rm;
  if (m_glob >= total_m_rows) {
    q_tok = 0u;
    g_sub = 0u;
    return false;
  }
  uint tok_local = 0u;
  fast_divmod_g(m_glob, repeat, tok_local, g_sub);
  q_tok = Flattened ? tok_local : (q_base + tok_local);
  return true;
}

// ============================================================================

SAGE2_ALWAYS_INLINE
static inline float decode_fp8_e4m3_scalar(uchar b) {
  const uint sign = uint(b >> 7u);
  const uint exp = uint((b >> 3u) & 0xFu);
  const uint man = uint(b & 0x7u);
  if (exp == 15u && man == 7u) return 0.0f;
  float v = (exp == 0u) ? ldexp(float(man) * 0.125f, -6)
                        : ldexp(1.0f + float(man) * 0.125f, int(exp) - 7);
  return (sign != 0u) ? -v : v;
}

template <uint WM, bool IsAffine4, bool UseMxfp8PV, bool SparseMInference>
SAGE2_ALWAYS_INLINE
static inline void paged_flash_prefill_sage2_apple9_fallback(
    device const bfloat *query, device const bfloat *gate,
    device const int8_t *k_int8, device const uchar *cache_value,
    device bfloat *output, device const uint *page_table,
    device const float *mu_k, device const float *s_k,
    device const ushort *active_indices, constant Sage2AttentionParams &sp,
    threadgroup Sage2TgStorage<WM, UseMxfp8PV> &tg_mem, ushort lane, ushort sg) {
  constexpr uint BD = 256u;
  constexpr uint BK = 32u;
  constexpr uint M_TILE = WM * 16u;
  constexpr bool kFlattened = !SparseMInference;
  using PVValT = typename conditional<IsAffine4, half, bfloat>::type;

  const uint flat_tid = uint(sg) * 32u + uint(lane);
  const uint rg = uint(sg) >> 2u;
  const uint dp = uint(sg) & 3u;
  const uint tm = rg * 16u;
  const uint lane_u = uint(lane);
  const uint kvh_cnt = sp.base.key_value_heads;
  const uint kvh = tg_mem.meta[5];
  const uint q_blk = tg_mem.meta[6];
  const uint repeat = sp.base.query_heads / kvh_cnt;
  const uint bm_tok = M_TILE / repeat;
  const uint q_m_base = q_blk * M_TILE;
  const uint q_base = kFlattened ? (q_m_base / repeat) : (q_blk * bm_tok);
  const uint total_m_rows =
      kFlattened ? (sp.base.tokens * repeat)
                 : (min(bm_tok, sp.base.tokens - q_base) * repeat);

  // Quantize Q into local 64-ch int8 per lane (r_me = lane>>1, h_me = lane&1 -> 32 ch)
  // Wait: even simpler! Reload q_i8 from tg_mem.q_i8 (UseMxfp8PV) OR quantize directly from query (0 ct_q)!
  const uint r_me = lane_u >> 1u;
  const uint h_me = lane_u & 1u;
  float4 o0(0.0f), o1(0.0f), o2(0.0f), o3(0.0f);
  float4 o4(0.0f), o5(0.0f), o6(0.0f), o7(0.0f);

  if (flat_tid < M_TILE) {
    tg_mem.ml[flat_tid] = kNegInf;
    tg_mem.ml[M_TILE + flat_tid] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Use simple exact INT8-Q x INT8-K + Delta reference loop on Apple9 (zero get_multidimensional_index, zero 8K scratch assumption!)
  constexpr auto qk_desc = matmul2d_descriptor(
      16, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<qk_desc, execution_simdgroup> qk_op;

  for (uint step = 0u; step < tg_mem.meta[3]; ++step) {
    const uint nk_blocks = tg_mem.meta[1];
    const uint kb =
        SparseMInference ? uint(active_indices[tg_mem.meta[4] + step]) : step;
    if (kb >= nk_blocks) continue;

    // 1. Compute Q_i8 (per-row) into s_xchg? Even simpler: compute QK + Delta directly for each (rg, dp_g)!
    // Wait: for UseMxfp8PV, tg_mem.q_i8[8 KiB] ALREADY holds quantized Q!
    // For !UseMxfp8PV, re-quantize into s_xchg[1024..2047] (4 KiB) 1 rg at a time!
    SAGE2_NOUNROLL
    for (uint rg_p = 0u; rg_p < WM; ++rg_p) {
      SAGE2_NOUNROLL
      for (uint dp_g = 0u; dp_g < 4u; ++dp_g) {
        if constexpr (!UseMxfp8PV) {
          if (rg == rg_p && dp == dp_g) {
            const uint rm = rg_p * 16u + r_me;
            uint q_tok = 0u, g_sub = 0u;
            const bool qv = decode_m_row<kFlattened>(rm, q_m_base, q_base, total_m_rows, repeat, q_tok, g_sub);
            const float sq_val = tg_mem.sq[dp_g * M_TILE + rm];
            const float inv_sq = (sq_val > 0.0f) ? ((0.0625f * kLog2e) / sq_val) : 0.0f;
            threadgroup char4 *q_dst = reinterpret_cast<threadgroup char4 *>(
                reinterpret_cast<threadgroup uchar *>(tg_mem.s_xchg + 1536u) + r_me * 64u + h_me * 32u);
            if (qv) {
              const uint qh_l = kvh * repeat + g_sub;
              device const bfloat4 *qp = reinterpret_cast<device const bfloat4 *>(
                  query + ulong(q_tok) * sp.base.query_token_stride +
                  ulong(qh_l) * sp.base.query_head_stride + dp_g * 64u + h_me * 32u);
              SAGE2_UNROLL for (uint k = 0u; k < 8u; ++k) {
                const float4 qf = clamp(rint(float4(clean_bf4(qp[k])) * inv_sq), float4(-127.0f), float4(127.0f));
                q_dst[k] = char4(char(int(qf.x)), char(int(qf.y)), char(int(qf.z)), char(int(qf.w)));
              }
            } else {
              SAGE2_UNROLL for (uint k = 0u; k < 8u; ++k) q_dst[k] = char4(0);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ_sg(
                reinterpret_cast<threadgroup int8_t *>(tg_mem.s_xchg + 1536u), dextents<int32_t, 2>(64, 16));
            tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK_dev(
                const_cast<device int8_t *>(k_int8 + ((ulong(kvh) * sp.max_kv_blocks + kb) * 4ul + dp_g) * 2048ul),
                dextents<int32_t, 2>(64, 32));
            threadgroup int32_t *s_i32 = reinterpret_cast<threadgroup int32_t *>(tg_mem.s_xchg + 1024u);
            tensor<threadgroup int32_t, dextents<int32_t, 2>, tensor_inline> tS(s_i32, dextents<int32_t, 2>(32, 16));
            qk_op.run(tQ_sg, tK_dev, tS);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            // Also compute exact Delta_dp_g for this kb (16 rows x 64 ch dot mu_k)
            device const float *mu_ptr = mu_k + ((ulong(kvh) * 4ul + dp_g) * sp.pad_kv_blocks + kb) * 64ul;
            const threadgroup int8_t *q_row_ptr = reinterpret_cast<const threadgroup int8_t *>(tg_mem.s_xchg + 1536u) + r_me * 64u + h_me * 32u;
            float d_part = 0.0f;
            for (uint c = 0u; c < 32u; ++c) d_part = fma(float(q_row_ptr[c]), mu_ptr[h_me * 32u + c], d_part);
            d_part += simd_shuffle_xor(d_part, ushort(1));
            if (h_me == 0u) {
              const float dv = d_part * sq_val;
              if (dp_g == 0u) tg_mem.del_tile[rm * 16u] = dv;
              else tg_mem.del_tile[rm * 16u] += dv;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            device const float *sk_ptr = s_k + ((ulong(kvh) * sp.max_kv_blocks + kb) * 4ul + dp_g) * 32ul;
            for (uint e = lane_u; e < 512u; e += 32u) {
              const uint r = e >> 5u;
              const uint c = e & 31u;
              const float val = float(s_i32[e]) * tg_mem.sq[dp_g * M_TILE + rg_p * 16u + r] * sk_ptr[c];
              if (dp_g == 0u) tg_mem.s_xchg[rg_p * 512u + e] = val;
              else tg_mem.s_xchg[rg_p * 512u + e] += val;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
          if (rg == rg_p && dp == dp_g) {
            threadgroup int8_t *q_sg_ptr = tg_mem.q_i8 + (rg_p * 4u + dp_g) * 1024u;
            tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ_sg(
                q_sg_ptr, dextents<int32_t, 2>(64, 16));
            tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK_dev(
                const_cast<device int8_t *>(k_int8 + ((ulong(kvh) * sp.max_kv_blocks + kb) * 4ul + dp_g) * 2048ul),
                dextents<int32_t, 2>(64, 32));
            threadgroup int32_t *s_i32 = reinterpret_cast<threadgroup int32_t *>(tg_mem.s_xchg + 512u);
            tensor<threadgroup int32_t, dextents<int32_t, 2>, tensor_inline> tS(s_i32, dextents<int32_t, 2>(32, 16));
            qk_op.run(tQ_sg, tK_dev, tS);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint rm = rg_p * 16u + r_me;
            device const float *mu_ptr = mu_k + ((ulong(kvh) * 4ul + dp_g) * sp.pad_kv_blocks + kb) * 64ul;
            float d_part = 0.0f;
            for (uint c = 0u; c < 32u; ++c) d_part = fma(float(q_sg_ptr[r_me * 64u + h_me * 32u + c]), mu_ptr[h_me * 32u + c], d_part);
            d_part += simd_shuffle_xor(d_part, ushort(1));
            if (h_me == 0u) {
              const float dv = d_part * tg_mem.sq[dp_g * M_TILE + rm];
              if (dp_g == 0u) tg_mem.del_tile[rm * 16u] = half(dv);
              else tg_mem.del_tile[rm * 16u] = half(float(tg_mem.del_tile[rm * 16u]) + dv);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            device const float *sk_ptr = s_k + ((ulong(kvh) * sp.max_kv_blocks + kb) * 4ul + dp_g) * 32ul;
            for (uint e = lane_u; e < 512u; e += 32u) {
              const uint r = e >> 5u;
              const uint c = e & 31u;
              const float val = float(s_i32[e]) * tg_mem.sq[dp_g * M_TILE + rg_p * 16u + r] * sk_ptr[c];
              if (dp_g == 0u) tg_mem.s_xchg[e] = val;
              else tg_mem.s_xchg[e] += val;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }

      // Softmax for rg_p immediately (so s_xchg[0..511] can be reused for next rg_p in UseMxfp8PV!)
      if (rg == rg_p && dp == 0u) {
        const uint max_valid_kv = tg_mem.meta[0];
        const uint kb_col_base = kb * BK;
        const threadgroup float *s_row_base = tg_mem.s_xchg + (UseMxfp8PV ? 0u : rg_p * 512u);
        for (uint r = 0u; r < 16u; ++r) {
          const uint c = lane_u;
          float s = s_row_base[r * 32u + c] + float(tg_mem.del_tile[(tm + r) * 16u]);
          const uint qp_raw = tg_mem.q_pos[tm + r];
          const bool rv = (qp_raw & 0x80000000u) != 0u;
          const uint rp = qp_raw & 0x7fffffffu;
          if (!(rv && (kb_col_base + c < max_valid_kv) && (kb_col_base + c <= rp))) {
            s = kNegInf;
          }
          const float rmax = simd_max(s);
          const float old_m = tg_mem.ml[tm + r];
          const float nm = max(old_m, rmax);
          const float fac = (old_m <= kNegInfHalf) ? 0.0f : fast::exp2(old_m - nm);
          const float pv = (nm <= kNegInfHalf || s <= kNegInfHalf) ? 0.0f : fast::exp2(s - nm);
          const float ts = simd_sum(pv);
          if (c == 0u) {
            tg_mem.ml[tm + r] = nm;
            tg_mem.ml[M_TILE + tm + r] = fma(tg_mem.ml[M_TILE + tm + r], fac, ts);
            tg_mem.fac[tm + r] = fac;
          }
          if constexpr (UseMxfp8PV) {
            tg_mem.p_bcast[(tm + r) * 32u + c] =
                pack<metal_fp8_e4m3_format>(float4(pv * kFp8E4M3Max)).as_storage_type().x;
          } else {
            reinterpret_cast<threadgroup PVValT *>(tg_mem.p_bcast)[(tm + r) * 32u + c] = PVValT(pv);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. Rescale + PV Accumulate
    {
      const float fv = tg_mem.fac[tm + r_me];
      o0 *= fv; o1 *= fv; o2 *= fv; o3 *= fv;
      o4 *= fv; o5 *= fv; o6 *= fv; o7 *= fv;

      const uint hbp = sp.base.page_tokens / BK;
      const uint phys_page = page_table[kb / hbp];
      const uint sub_off = (kb % hbp) * BK;
      const short valid_k = short(min(BK, tg_mem.meta[0] - kb * BK));
      const ulong page_bytes = IsAffine4 ? affine4_page_byte_stride(sp.base.page_tokens, kvh_cnt, BD) : 0ul;
      const ulong all_codes_bytes = IsAffine4 ? (ulong(kvh_cnt) * ulong(sp.base.page_tokens) * 128ul) : 0ul;
      const ulong kvh_code_base = IsAffine4 ? (ulong(kvh) * ulong(sp.base.page_tokens) * 128ul) : 0ul;
      const ulong bf16_kv_stride = IsAffine4 ? 0ul : (ulong(kvh_cnt) * BD);

      if constexpr (UseMxfp8PV) {
        SAGE2_NOUNROLL
        for (uint half_d = 0u; half_d < 2u; ++half_d) {
          prepare_v_mxfp8_half_4k<IsAffine4>(
              cache_value, phys_page, sub_off, sp.base.page_tokens, page_bytes,
              kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, valid_k,
              half_d, flat_tid, tg_mem.s_xchg, tg_mem.vt_sc);
          if (h_me == half_d) {
            const threadgroup uchar *vt_fp8 = reinterpret_cast<const threadgroup uchar *>(tg_mem.s_xchg);
            const uint sc_base = dp * 32u;
            const auto load_vt4 = [&](uint d4_idx, uint k) -> float4 {
              const uint d0 = sc_base + d4_idx * 4u;
              const float s0 = as_type<float>(uint(tg_mem.vt_sc[d0 + 0u]) << 23u);
              const float s1 = as_type<float>(uint(tg_mem.vt_sc[d0 + 1u]) << 23u);
              const float s2 = as_type<float>(uint(tg_mem.vt_sc[d0 + 2u]) << 23u);
              const float s3 = as_type<float>(uint(tg_mem.vt_sc[d0 + 3u]) << 23u);
              return float4(
                  decode_fp8_e4m3_scalar(vt_fp8[(d0 + 0u) * 32u + k]) * s0,
                  decode_fp8_e4m3_scalar(vt_fp8[(d0 + 1u) * 32u + k]) * s1,
                  decode_fp8_e4m3_scalar(vt_fp8[(d0 + 2u) * 32u + k]) * s2,
                  decode_fp8_e4m3_scalar(vt_fp8[(d0 + 3u) * 32u + k]) * s3);
            };
            for (uint k = 0u; k < 32u; ++k) {
              const float pk = decode_fp8_e4m3_scalar(tg_mem.p_bcast[(tm + r_me) * 32u + k]);
              if (pk != 0.0f) {
                o0 = fma(float4(pk), load_vt4(0u, k), o0);
                o1 = fma(float4(pk), load_vt4(1u, k), o1);
                o2 = fma(float4(pk), load_vt4(2u, k), o2);
                o3 = fma(float4(pk), load_vt4(3u, k), o3);
                o4 = fma(float4(pk), load_vt4(4u, k), o4);
                o5 = fma(float4(pk), load_vt4(5u, k), o5);
                o6 = fma(float4(pk), load_vt4(6u, k), o6);
                o7 = fma(float4(pk), load_vt4(7u, k), o7);
              }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      } else {
        const threadgroup PVValT *p_row =
            reinterpret_cast<const threadgroup PVValT *>(tg_mem.p_bcast) + (tm + r_me) * 32u;
        const uint col_v4_base = h_me * 8u;
        for (uint k = 0u; k < 32u; ++k) {
          const float pk = float(p_row[k]);
          if (pk != 0.0f) {
            o0 = fma(float4(pk), load_kv64_vec4_safe<IsAffine4>(cache_value, phys_page, sub_off, sp.base.page_tokens, page_bytes, kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, dp, k, col_v4_base + 0u, valid_k), o0);
            o1 = fma(float4(pk), load_kv64_vec4_safe<IsAffine4>(cache_value, phys_page, sub_off, sp.base.page_tokens, page_bytes, kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, dp, k, col_v4_base + 1u, valid_k), o1);
            o2 = fma(float4(pk), load_kv64_vec4_safe<IsAffine4>(cache_value, phys_page, sub_off, sp.base.page_tokens, page_bytes, kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, dp, k, col_v4_base + 2u, valid_k), o2);
            o3 = fma(float4(pk), load_kv64_vec4_safe<IsAffine4>(cache_value, phys_page, sub_off, sp.base.page_tokens, page_bytes, kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, dp, k, col_v4_base + 3u, valid_k), o3);
            o4 = fma(float4(pk), load_kv64_vec4_safe<IsAffine4>(cache_value, phys_page, sub_off, sp.base.page_tokens, page_bytes, kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, dp, k, col_v4_base + 4u, valid_k), o4);
            o5 = fma(float4(pk), load_kv64_vec4_safe<IsAffine4>(cache_value, phys_page, sub_off, sp.base.page_tokens, page_bytes, kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, dp, k, col_v4_base + 5u, valid_k), o5);
            o6 = fma(float4(pk), load_kv64_vec4_safe<IsAffine4>(cache_value, phys_page, sub_off, sp.base.page_tokens, page_bytes, kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, dp, k, col_v4_base + 6u, valid_k), o6);
            o7 = fma(float4(pk), load_kv64_vec4_safe<IsAffine4>(cache_value, phys_page, sub_off, sp.base.page_tokens, page_bytes, kvh_code_base, all_codes_bytes, bf16_kv_stride, kvh, dp, k, col_v4_base + 7u, valid_k), o7);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }

  // Epilogue
  uint tok = 0u, g_sub = 0u;
  if (decode_m_row<kFlattened>(tm + r_me, q_m_base, q_base, total_m_rows, repeat, tok, g_sub)) {
    const float l_val = tg_mem.ml[M_TILE + tm + r_me];
    const float rcp = (l_val > 0.0f) ? ((UseMxfp8PV ? kInvFp8E4M3Max : 1.0f) / l_val) : 0.0f;
    const uint qh_o = kvh * repeat + g_sub;
    const uint d_off = dp * 64u + h_me * 32u;
    device const bfloat4 *gp = reinterpret_cast<device const bfloat4 *>(
        gate + ulong(tok) * sp.base.gate_token_stride + ulong(qh_o) * sp.base.gate_head_stride + d_off);
    device bfloat4 *op = reinterpret_cast<device bfloat4 *>(
        output + (ulong(tok) * sp.base.query_heads + qh_o) * BD + d_off);
    op[0] = bfloat4((o0 * rcp) / (float4(1.0f) + fast::exp(-float4(gp[0]))));
    op[1] = bfloat4((o1 * rcp) / (float4(1.0f) + fast::exp(-float4(gp[1]))));
    op[2] = bfloat4((o2 * rcp) / (float4(1.0f) + fast::exp(-float4(gp[2]))));
    op[3] = bfloat4((o3 * rcp) / (float4(1.0f) + fast::exp(-float4(gp[3]))));
    op[4] = bfloat4((o4 * rcp) / (float4(1.0f) + fast::exp(-float4(gp[4]))));
    op[5] = bfloat4((o5 * rcp) / (float4(1.0f) + fast::exp(-float4(gp[5]))));
    op[6] = bfloat4((o6 * rcp) / (float4(1.0f) + fast::exp(-float4(gp[6]))));
    op[7] = bfloat4((o7 * rcp) / (float4(1.0f) + fast::exp(-float4(gp[7]))));
  }
}

// Unified WP2.2 Zero-Spill Production Engine (WN=4, WM in {2, 3})
// ============================================================================
template <uint WM, bool IsAffine4, bool UseMxfp8PV, bool SparseMInference>
SAGE2_ALWAYS_INLINE
static inline void paged_flash_prefill_sage2_hd256_impl(
    device const bfloat *query, device const bfloat *gate,
    device const int8_t *k_int8, device const uchar *cache_value,
    device bfloat *output, device const uint *page_table,
    device const uint *positions, device const float *mu_k,
    device const float *s_k, device const uint *valid_kv_buf,
    device const ushort *active_indices, device const uint *active_counts,
    constant Sage2AttentionParams &sp,
    threadgroup Sage2TgStorage<WM, UseMxfp8PV> &tg_mem, ushort lane, ushort sg,
    uint3 tg) {
  constexpr uint BD = 256u;
  constexpr uint BK = 32u;
  constexpr uint M_TILE = WM * 16u;
  constexpr bool kFlattened = !SparseMInference;
  using PVValT = typename conditional<IsAffine4, half, bfloat>::type;
  using PVVal4 = typename conditional<IsAffine4, half4, bfloat4>::type;

  if (sp.base.tokens == 0u || sp.base.query_heads == 0u ||
      sp.base.key_value_heads == 0u) {
    return;
  }
  const uint kvh_cnt = sp.base.key_value_heads;
  const uint repeat_init = sp.base.query_heads / kvh_cnt;
  if (repeat_init == 0u || repeat_init > M_TILE) {
    return;
  }

  // LPT Reverse q_blk + kvh-minor mapping.
  const uint q_blk_rev = tg.y / kvh_cnt;
  const uint kvh = tg.y - q_blk_rev * kvh_cnt;
  const uint n_q_blocks =
      kFlattened ? ((sp.base.tokens * repeat_init + M_TILE - 1u) / M_TILE)
                 : ((sp.base.tokens + (M_TILE / repeat_init) - 1u) /
                    (M_TILE / repeat_init));
  if (q_blk_rev >= n_q_blocks) {
    return;
  }
  const uint q_blk = (n_q_blocks - 1u) - q_blk_rev;
  const uint flat_tid = uint(sg) * 32u + uint(lane);
  const uint rg = uint(sg) >> 2u; // 0 .. WM-1
  const uint dp = uint(sg) & 3u;  // 0 .. 3 (D_SLICE = 64)
  const uint tm = rg * 16u;
  const uint lane_u = uint(lane);

  // Scoped Prologue: populate per-row packed tg_mem.q_pos[M_TILE], tg_mem.meta[7], tg_mem.sq[4*M_TILE].
  constexpr auto qk_desc = matmul2d_descriptor(
      16, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<qk_desc, execution_simdgroup> qk_op;

  {
    const uint bm_tok = M_TILE / repeat_init;
    const uint total_m_all = sp.base.tokens * repeat_init;
    const uint q_m_base = q_blk * M_TILE;
    const uint q_base = kFlattened ? (q_m_base / repeat_init) : (q_blk * bm_tok);
    const uint total_m_rows =
        kFlattened ? total_m_all : (min(bm_tok, sp.base.tokens - q_base) * repeat_init);

    if (flat_tid < M_TILE) {
      uint qt = 0u, gu = 0u;
      const bool rv = decode_m_row<kFlattened>(flat_tid, q_m_base, q_base,
                                               total_m_rows, repeat_init, qt, gu);
      tg_mem.q_pos[flat_tid] = rv ? (positions[qt] | 0x80000000u) : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (flat_tid == 0u) {
      uint max_vkv = 0u;
      uint min_qp = 0x7fffffffu;
      for (uint r = 0u; r < M_TILE; ++r) {
        const uint qp_raw = tg_mem.q_pos[r];
        if ((qp_raw & 0x80000000u) != 0u) {
          const uint pos = qp_raw & 0x7fffffffu;
          max_vkv = max(max_vkv, pos + 1u);
          min_qp = min(min_qp, pos);
        }
      }
      max_vkv = min(max_vkv, valid_kv_buf[0]);
      const uint nk = min((max_vkv + BK - 1u) / BK, sp.max_kv_blocks);
      tg_mem.meta[0] = max_vkv;
      tg_mem.meta[1] = nk;
      tg_mem.meta[2] = min_qp / BK;
      tg_mem.meta[3] =
          SparseMInference ? active_counts[ulong(kvh) * sp.max_q_blocks + q_blk]
                           : nk;
      tg_mem.meta[4] = SparseMInference ? uint((ulong(kvh) * sp.max_q_blocks + q_blk) * sp.active_stride) : 0u;
      tg_mem.meta[5] = kvh;
      tg_mem.meta[6] = q_blk;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint q_r_in16 = lane_u >> 1u;
    const uint q_half = lane_u & 1u;
    const uint rm = tm + q_r_in16;
    uint q_tok = 0u, g_sub = 0u;
    const bool qv = decode_m_row<kFlattened>(rm, q_m_base, q_base, total_m_rows,
                                             repeat_init, q_tok, g_sub);
    const uint qh_l = kvh * repeat_init + g_sub;
    device const bfloat4 *q_ptr =
        qv ? reinterpret_cast<device const bfloat4 *>(
                 query + ulong(q_tok) * sp.base.query_token_stride +
                 ulong(qh_l) * sp.base.query_head_stride + dp * 64u +
                 q_half * 32u)
           : nullptr;

    float lmax = 0.0f;
    if (qv) {
      SAGE2_UNROLL
      for (uint k = 0u; k < 8u; ++k) {
        const float4 av = fabs(float4(clean_bf4(q_ptr[k])));
        lmax = max(lmax, max(max(av.x, av.y), max(av.z, av.w)));
      }
    }
    const float scale2 = 0.0625f * kLog2e;
    const float q_absmax = max(lmax, simd_shuffle_xor(lmax, ushort(1)));
    const float sq_scaled =
        (q_absmax > 0.0f) ? (q_absmax * (scale2 / 127.0f)) : 0.0f;
    const float inv_sq = (q_absmax > 0.0f) ? (127.0f / q_absmax) : 0.0f;
    if (q_half == 0u) {
      tg_mem.sq[dp * M_TILE + rm] = sq_scaled;
    }

    // Write quantized Q directly into tg_mem.q_i8 (UseMxfp8PV) or tg_mem.s_xchg (!UseMxfp8PV)
    threadgroup uchar *q_base_tg = nullptr;
    if constexpr (UseMxfp8PV) {
      q_base_tg = reinterpret_cast<threadgroup uchar *>(tg_mem.q_i8) + uint(sg) * 1024u;
    } else {
      q_base_tg = reinterpret_cast<threadgroup uchar *>(tg_mem.s_xchg) + uint(sg) * 1024u;
    }
    threadgroup char4 *q_dst = reinterpret_cast<threadgroup char4 *>(
        q_base_tg + q_r_in16 * 64u + q_half * 32u);
    if (qv) {
      SAGE2_UNROLL
      for (uint k = 0u; k < 8u; ++k) {
        const float4 vf = float4(clean_bf4(q_ptr[k]));
        const float4 qf =
            clamp(rint(vf * inv_sq), float4(-127.0f), float4(127.0f));
        q_dst[k] = char4(char(int(qf.x)), char(int(qf.y)), char(int(qf.z)),
                         char(int(qf.w)));
      }
    } else {
      SAGE2_UNROLL
      for (uint k = 0u; k < 8u; ++k) {
        q_dst[k] = char4(0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (!sage2_has_apple10_nax_layout()) {
    paged_flash_prefill_sage2_apple9_fallback<WM, IsAffine4, UseMxfp8PV, SparseMInference>(
        query, gate, k_int8, cache_value, output, page_table, mu_k, s_k,
        active_indices, sp, tg_mem, lane, sg);
    return;
  }

  const short2 sc = sage2_frag_coord(lane);
  const short sm = sc.y;
  const short sn = sc.x;

  // ==========================================================================
  // Path A: MXFP8 P x V (UseMxfp8PV == true, 15.16 KiB, 0 ct_q GPRs, 0 Spills!)
  // ==========================================================================
  if constexpr (UseMxfp8PV) {
    constexpr auto pv_desc_mx = matmul2d_descriptor(
        16, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc_mx, execution_simdgroup> pv_op_mx;

    vec<float, 8> O0(0.0f), O1(0.0f), O2(0.0f), O3(0.0f);

    if (dp == 0u && sn == 0) {
      tg_mem.ml[tm + uint(sm)] = kNegInf;
      tg_mem.ml[tm + 8u + uint(sm)] = kNegInf;
      tg_mem.ml[M_TILE + tm + uint(sm)] = 0.0f;
      tg_mem.ml[M_TILE + tm + 8u + uint(sm)] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 0u; step < tg_mem.meta[3]; ++step) {
      const uint nk_blocks = tg_mem.meta[1];
      if ((step & 15u) == 0u) {
        if constexpr (SparseMInference) {
          if (flat_tid < 16u) {
            const uint s_idx = step + flat_tid;
            uint kb_b = 0xffffffffu;
            if (s_idx < tg_mem.meta[3]) {
              const uint c = uint(active_indices[tg_mem.meta[4] + s_idx]);
              if (c < nk_blocks) kb_b = c;
            }
            reinterpret_cast<threadgroup uint *>(tg_mem.fac)[flat_tid] = kb_b;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        refresh_delta16_mxfp8_sram<SparseMInference>(
            tg_mem.q_i8, mu_k, step, tg_mem.meta[5], sp.pad_kv_blocks,
            flat_tid, uint(sg), rg, dp, lane, tg_mem.sq,
            reinterpret_cast<const threadgroup uint *>(tg_mem.fac),
            tg_mem.s_xchg, tg_mem.del_tile);
      }

      const uint kb =
          SparseMInference ? uint(active_indices[tg_mem.meta[4] + step]) : step;
      if (kb >= nk_blocks) {
        continue;
      }

      // 1. Direct SRAM tQ_tg x device tK_dev INT8 QK + 2-pass 4 KiB reduction + Softmax
      {
        const uint max_valid_kv = tg_mem.meta[0];
        const uint kb_col_base = kb * BK;
        const bool full_causal_free =
            (kb < tg_mem.meta[2]) && (max_valid_kv >= kb_col_base + BK) &&
            ((tg_mem.q_pos[tm + 15u] & 0x80000000u) != 0u);

        tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ_tg(
            tg_mem.q_i8 + uint(sg) * 1024u, dextents<int32_t, 2>(64, 16));
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK_dev(
            const_cast<device int8_t *>(
                k_int8 + ((ulong(tg_mem.meta[5]) * sp.max_kv_blocks + kb) * 4ul + dp) * 2048ul),
            dextents<int32_t, 2>(64, 32));
        auto ct_s = qk_op.template get_destination_cooperative_tensor<
            decltype(tQ_tg), decltype(tK_dev), int32_t>();
        qk_op.run(tQ_tg, tK_dev, ct_s);

        const float sq0 = tg_mem.sq[dp * M_TILE + tm + uint(sm)];
        const float sq1 = tg_mem.sq[dp * M_TILE + tm + 8u + uint(sm)];
        device const float *sk_ptr =
            s_k + ((ulong(tg_mem.meta[5]) * sp.max_kv_blocks + kb) * 4ul + dp) * 32ul;
        const float4 sk0 = *reinterpret_cast<device const float4 *>(sk_ptr + uint(sn));
        const float4 sk1 = *reinterpret_cast<device const float4 *>(sk_ptr + 16u + uint(sn));
        const uint b16 = step & 15u;

        {
          threadgroup float4 *slot = reinterpret_cast<threadgroup float4 *>(
              tg_mem.s_xchg + rg * 512u + lane_u * 16u);
          if (dp == 1u) {
            slot[0] = float4(ct_s[0], ct_s[1], ct_s[2], ct_s[3]) * (sq0 * sk0);
            slot[1] = float4(ct_s[4], ct_s[5], ct_s[6], ct_s[7]) * (sq1 * sk0);
            slot[2] = float4(ct_s[8], ct_s[9], ct_s[10], ct_s[11]) * (sq0 * sk1);
            slot[3] = float4(ct_s[12], ct_s[13], ct_s[14], ct_s[15]) * (sq1 * sk1);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (dp == 2u) {
            slot[0] += float4(ct_s[0], ct_s[1], ct_s[2], ct_s[3]) * (sq0 * sk0);
            slot[1] += float4(ct_s[4], ct_s[5], ct_s[6], ct_s[7]) * (sq1 * sk0);
            slot[2] += float4(ct_s[8], ct_s[9], ct_s[10], ct_s[11]) * (sq0 * sk1);
            slot[3] += float4(ct_s[12], ct_s[13], ct_s[14], ct_s[15]) * (sq1 * sk1);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (dp == 3u) {
            slot[0] += float4(ct_s[0], ct_s[1], ct_s[2], ct_s[3]) * (sq0 * sk0);
            slot[1] += float4(ct_s[4], ct_s[5], ct_s[6], ct_s[7]) * (sq1 * sk0);
            slot[2] += float4(ct_s[8], ct_s[9], ct_s[10], ct_s[11]) * (sq0 * sk1);
            slot[3] += float4(ct_s[12], ct_s[13], ct_s[14], ct_s[15]) * (sq1 * sk1);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (dp == 0u) {
            slot[0] += float4(ct_s[0], ct_s[1], ct_s[2], ct_s[3]) * (sq0 * sk0);
            slot[1] += float4(ct_s[4], ct_s[5], ct_s[6], ct_s[7]) * (sq1 * sk0);
            slot[2] += float4(ct_s[8], ct_s[9], ct_s[10], ct_s[11]) * (sq0 * sk1);
            slot[3] += float4(ct_s[12], ct_s[13], ct_s[14], ct_s[15]) * (sq1 * sk1);
          }
        }
        // ct_s, sq0, sq1, sk0, sk1 are ALL DEAD before Softmax!
        if (dp == 0u) {
          const uint c_lo = kb_col_base + uint(sn);
          const threadgroup float4 *slot = reinterpret_cast<const threadgroup float4 *>(
              tg_mem.s_xchg + rg * 512u + lane_u * 16u);
          SAGE2_NOUNROLL
          for (uint r_sub = 0u; r_sub < 2u; ++r_sub) {
            const uint r_loc = (r_sub == 0u) ? uint(sm) : uint(sm + 8);
            const float dv = float(tg_mem.del_tile[(tm + r_loc) * 16u + b16]);
            float4 a = ((r_sub == 0u) ? slot[0] : slot[1]) + dv;
            float4 b = ((r_sub == 0u) ? slot[2] : slot[3]) + dv;
            if (!full_causal_free) {
              const uint qp_raw = tg_mem.q_pos[tm + r_loc];
              const bool rv = (qp_raw & 0x80000000u) != 0u;
              const uint rp = qp_raw & 0x7fffffffu;
              const uint4 c0v = uint4(c_lo) + uint4(0u, 1u, 2u, 3u);
              const uint4 c1v = c0v + uint4(16u);
              a = select(float4(kNegInf), a, rv && (c0v < max_valid_kv) && (c0v <= rp));
              b = select(float4(kNegInf), b, rv && (c1v < max_valid_kv) && (c1v <= rp));
            }
            const float4 mx = max(a, b);
            float rmax = max(max(mx.x, mx.y), max(mx.z, mx.w));
            rmax = max(rmax, simd_shuffle_xor(rmax, ushort(1)));
            rmax = max(rmax, simd_shuffle_xor(rmax, ushort(8)));
            const float old_m = tg_mem.ml[tm + r_loc];
            const float nm = max(old_m, rmax);
            const float fac = (old_m <= kNegInfHalf) ? 0.0f : fast::exp2(old_m - nm);
            if (sn == 0) tg_mem.ml[tm + r_loc] = nm;
            if (full_causal_free) {
              a = fast::exp2(a - nm);
              b = fast::exp2(b - nm);
            } else {
              const bool emp = (nm <= kNegInfHalf);
              a = select(fast::exp2(a - nm), float4(0.0f), emp || (a <= kNegInfHalf));
              b = select(fast::exp2(b - nm), float4(0.0f), emp || (b <= kNegInfHalf));
            }
            const float4 ps = a + b;
            float ts = (ps.x + ps.y) + (ps.z + ps.w);
            ts += simd_shuffle_xor(ts, ushort(1));
            ts += simd_shuffle_xor(ts, ushort(8));
            if (sn == 0) tg_mem.ml[M_TILE + tm + r_loc] = fma(tg_mem.ml[M_TILE + tm + r_loc], fac, ts);
            threadgroup uchar *my_p = tg_mem.p_bcast + (tm + r_loc) * 32u;
            *reinterpret_cast<threadgroup packed_uchar4 *>(my_p + uint(sn)) =
                pack<metal_fp8_e4m3_format>(a * kFp8E4M3Max).as_storage_type();
            *reinterpret_cast<threadgroup packed_uchar4 *>(my_p + 16u + uint(sn)) =
                pack<metal_fp8_e4m3_format>(b * kFp8E4M3Max).as_storage_type();
            if (sn == 0) tg_mem.fac[tm + r_loc] = fac;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      // 2. Rescale O0..O3 & Half-by-Half MXFP8 V^T Prepare (4 KiB s_xchg) + PV!
      {
        const float fac0 = tg_mem.fac[tm + uint(sm)];
        const float fac1 = tg_mem.fac[tm + 8u + uint(sm)];
        const vec<float, 8> fv(fac0, fac0, fac0, fac0, fac1, fac1, fac1, fac1);
        O0 *= fv; O1 *= fv; O2 *= fv; O3 *= fv;

        const uint hbp = sp.base.page_tokens / BK;
        const uint phys_page = page_table[kb / hbp];
        const uint sub_off = (kb % hbp) * BK;
        const short valid_k = short(min(BK, tg_mem.meta[0] - kb * BK));
        const ulong page_bytes = IsAffine4 ? affine4_page_byte_stride(sp.base.page_tokens, kvh_cnt, BD) : 0ul;
        const ulong all_codes_bytes = IsAffine4 ? (ulong(kvh_cnt) * ulong(sp.base.page_tokens) * 128ul) : 0ul;
        const ulong kvh_code_base = IsAffine4 ? (ulong(tg_mem.meta[5]) * ulong(sp.base.page_tokens) * 128ul) : 0ul;
        const ulong bf16_kv_stride = IsAffine4 ? 0ul : (ulong(kvh_cnt) * BD);
        threadgroup uchar *vt_base = reinterpret_cast<threadgroup uchar *>(tg_mem.s_xchg);

        prepare_v_mxfp8_half_4k<IsAffine4>(
            cache_value, phys_page, sub_off, sp.base.page_tokens, page_bytes,
            kvh_code_base, all_codes_bytes, bf16_kv_stride, tg_mem.meta[5], valid_k,
            0u, flat_tid, tg_mem.s_xchg, tg_mem.vt_sc);
        {
          TgFp8Plain tP0(tg_mem.p_bcast + tm * 32u, dextents<int32_t, 2>(32, 16));
          TgMxfp8VT tVT0(vt_base + dp * 1024u, dextents<int32_t, 2>(32, 32), TgUe8ScaleTag(tg_mem.vt_sc + dp * 32u));
          auto c01 = pv_op_mx.template get_destination_cooperative_tensor<decltype(tP0), decltype(tVT0), float>();
          SAGE2_UNROLL for (ushort i = 0u; i < 8u; ++i) { c01[i] = O0[i]; c01[8u + i] = O1[i]; }
          pv_op_mx.run(tP0, tVT0, c01);
          SAGE2_UNROLL for (ushort i = 0u; i < 8u; ++i) { O0[i] = c01[i]; O1[i] = c01[8u + i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        prepare_v_mxfp8_half_4k<IsAffine4>(
            cache_value, phys_page, sub_off, sp.base.page_tokens, page_bytes,
            kvh_code_base, all_codes_bytes, bf16_kv_stride, tg_mem.meta[5], valid_k,
            1u, flat_tid, tg_mem.s_xchg, tg_mem.vt_sc);
        {
          TgFp8Plain tP1(tg_mem.p_bcast + tm * 32u, dextents<int32_t, 2>(32, 16));
          TgMxfp8VT tVT1(vt_base + dp * 1024u, dextents<int32_t, 2>(32, 32), TgUe8ScaleTag(tg_mem.vt_sc + dp * 32u));
          auto c23 = pv_op_mx.template get_destination_cooperative_tensor<decltype(tP1), decltype(tVT1), float>();
          SAGE2_UNROLL for (ushort i = 0u; i < 8u; ++i) { c23[i] = O2[i]; c23[8u + i] = O3[i]; }
          pv_op_mx.run(tP1, tVT1, c23);
          SAGE2_UNROLL for (ushort i = 0u; i < 8u; ++i) { O2[i] = c23[i]; O3[i] = c23[8u + i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    if (dp == 0u && sn == 0) {
      tg_mem.fac[tm + uint(sm)] = (tg_mem.ml[M_TILE + tm + uint(sm)] > 0.0f) ? (kInvFp8E4M3Max / tg_mem.ml[M_TILE + tm + uint(sm)]) : 0.0f;
      tg_mem.fac[tm + 8u + uint(sm)] =
          (tg_mem.ml[M_TILE + tm + 8u + uint(sm)] > 0.0f) ? (kInvFp8E4M3Max / tg_mem.ml[M_TILE + tm + 8u + uint(sm)]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
      const uint kvh_e = tg_mem.meta[5];
      const uint q_blk_e = tg_mem.meta[6];
      const uint repeat = sp.base.query_heads / kvh_cnt;
      const uint bm_tok = M_TILE / repeat;
      const uint q_m_base = q_blk_e * M_TILE;
      const uint q_base = kFlattened ? (q_m_base / repeat) : (q_blk_e * bm_tok);
      const uint total_m_rows =
          kFlattened ? (sp.base.tokens * repeat)
                     : (min(bm_tok, sp.base.tokens - q_base) * repeat);
      const uint d_offset = dp * 64u;
      {
        uint tok = 0u, g_sub = 0u;
        if (decode_m_row<kFlattened>(tm + uint(sm), q_m_base, q_base, total_m_rows, repeat, tok, g_sub)) {
          const float rcp = tg_mem.fac[tm + uint(sm)];
          const uint qh_o = kvh_e * repeat + g_sub;
          device const bfloat *gp = gate + ulong(tok) * sp.base.gate_token_stride + ulong(qh_o) * sp.base.gate_head_stride + d_offset + uint(sn);
          device bfloat *op = output + (ulong(tok) * sp.base.query_heads + qh_o) * BD + d_offset + uint(sn);
          *reinterpret_cast<device bfloat4 *>(op +  0u) = bfloat4((float4(O0[0],O0[1],O0[2],O0[3]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp +  0u)))));
          *reinterpret_cast<device bfloat4 *>(op + 16u) = bfloat4((float4(O1[0],O1[1],O1[2],O1[3]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp + 16u)))));
          *reinterpret_cast<device bfloat4 *>(op + 32u) = bfloat4((float4(O2[0],O2[1],O2[2],O2[3]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp + 32u)))));
          *reinterpret_cast<device bfloat4 *>(op + 48u) = bfloat4((float4(O3[0],O3[1],O3[2],O3[3]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp + 48u)))));
        }
      }
      {
        uint tok = 0u, g_sub = 0u;
        if (decode_m_row<kFlattened>(tm + 8u + uint(sm), q_m_base, q_base, total_m_rows, repeat, tok, g_sub)) {
          const float rcp = tg_mem.fac[tm + 8u + uint(sm)];
          const uint qh_o = kvh_e * repeat + g_sub;
          device const bfloat *gp = gate + ulong(tok) * sp.base.gate_token_stride + ulong(qh_o) * sp.base.gate_head_stride + d_offset + uint(sn);
          device bfloat *op = output + (ulong(tok) * sp.base.query_heads + qh_o) * BD + d_offset + uint(sn);
          *reinterpret_cast<device bfloat4 *>(op +  0u) = bfloat4((float4(O0[4],O0[5],O0[6],O0[7]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp +  0u)))));
          *reinterpret_cast<device bfloat4 *>(op + 16u) = bfloat4((float4(O1[4],O1[5],O1[6],O1[7]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp + 16u)))));
          *reinterpret_cast<device bfloat4 *>(op + 32u) = bfloat4((float4(O2[4],O2[5],O2[6],O2[7]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp + 32u)))));
          *reinterpret_cast<device bfloat4 *>(op + 48u) = bfloat4((float4(O3[4],O3[5],O3[6],O3[7]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp + 48u)))));
        }
      }
    }
    return;
  }

  else {
  // Load ct_q ONCE for Path B (FP16/BF16 PV) from tg_mem.s_xchg (0 B extra SRAM!)
  auto ct_q =
      qk_op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  {
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ_tg(
        reinterpret_cast<threadgroup int8_t *>(tg_mem.s_xchg) + uint(sg) * 1024u,
        dextents<int32_t, 2>(64, 16));
    ct_q.load(tQ_tg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ==========================================================================
  // Path B: FP16/BF16 P x V (UseMxfp8PV == false, Primary 86-91 TFLOPS Path!)
  // ==========================================================================
  threadgroup PVValT *tg_p_base =
      reinterpret_cast<threadgroup PVValT *>(tg_mem.p_bcast);

  {
    // ------------------------------------------------------------------------
    // Apple10 (M5 Ultra) Zero-Spill WN=4 In-Register Frag16x16 Engine:
    //   * O0..O3 = ONLY 32 GPRs, ct_q = ONLY 8 GPRs, 0 hoisted loop-invariants!
    //   * Direct Frag16x16 ct_s + 2-slot float4 reduction + in-register softmax
    //   * Direct-L1 Frag16x16 PV (zero V staging, zero PV barriers!)
    // ------------------------------------------------------------------------
    using Frag8 = vec<PVValT, 8>;
    using F8 = vec<float, 8>;

    F8 O0(0.0f), O1(0.0f), O2(0.0f), O3(0.0f);
    float m0 = kNegInf, m1 = kNegInf, l0 = 0.0f, l1 = 0.0f;

    for (uint step = 0u; step < tg_mem.meta[3]; ++step) {
      const uint nk_blocks = tg_mem.meta[1];
      if ((step & ((WM == 2u) ? 15u : 7u)) == 0u) {
        if constexpr (SparseMInference) {
          if (flat_tid < 16u) {
            const uint s_idx = step + flat_tid;
            uint kb_b = 0xffffffffu;
            if (s_idx < tg_mem.meta[3]) {
              const uint c = uint(active_indices[tg_mem.meta[4] + s_idx]);
              if (c < nk_blocks) kb_b = c;
            }
            reinterpret_cast<threadgroup uint *>(tg_mem.fac)[flat_tid] = kb_b;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        refresh_delta16_to_sram<WM, SparseMInference, true>(
            ct_q, mu_k, step, tg_mem.meta[5], sp.pad_kv_blocks,
            flat_tid, uint(sg), rg, dp, lane, tg_mem.sq,
            reinterpret_cast<const threadgroup uint *>(tg_mem.fac),
            tg_mem.s_xchg, tg_mem.del_tile);
      }

      const uint kb =
          SparseMInference
              ? uint(active_indices[tg_mem.meta[4] + step])
              : step;
      if (kb >= nk_blocks) {
        continue;
      }

      // 1. Direct device INT8 QK (16x32x64) + 2-slot reduction + In-Register Softmax!
      {
        const uint max_valid_kv = tg_mem.meta[0];
        const uint kb_col_base = kb * BK;
        const bool full_causal_free =
            (kb < tg_mem.meta[2]) && (max_valid_kv >= kb_col_base + BK) &&
            ((tg_mem.q_pos[tm + 15u] & 0x80000000u) != 0u);

        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK_dev(
            const_cast<device int8_t *>(
                k_int8 + ((ulong(tg_mem.meta[5]) * sp.max_kv_blocks + kb) * 4ul + dp) *
                             2048ul),
            dextents<int32_t, 2>(64, 32));
        auto ct_s = qk_op.template get_destination_cooperative_tensor<
            remove_addrspace_t<decltype(ct_q)>, decltype(tK_dev), int32_t>();
        qk_op.run(ct_q, tK_dev, ct_s);

        const float sq0 = tg_mem.sq[dp * M_TILE + tm + uint(sm)];
        const float sq1 = tg_mem.sq[dp * M_TILE + tm + 8u + uint(sm)];
        device const float *sk_ptr =
            s_k + ((ulong(tg_mem.meta[5]) * sp.max_kv_blocks + kb) * 4ul + dp) * 32ul;
        const float4 sk0 =
            *reinterpret_cast<device const float4 *>(sk_ptr + uint(sn));
        const float4 sk1 =
            *reinterpret_cast<device const float4 *>(sk_ptr + 16u + uint(sn));
        constexpr uint DEL_B = (WM == 2u) ? 16u : 8u; const uint b16 = step & (DEL_B - 1u);

        {
          {
            threadgroup half4 *dst = reinterpret_cast<threadgroup half4 *>(
                tg_mem.s_xchg + (rg * 4u + dp) * 256u + lane_u * 8u);
            dst[0] = half4(float4(ct_s[0], ct_s[1], ct_s[2], ct_s[3]) * (sq0 * sk0));
            dst[1] = half4(float4(ct_s[4], ct_s[5], ct_s[6], ct_s[7]) * (sq1 * sk0));
            dst[2] = half4(float4(ct_s[8], ct_s[9], ct_s[10], ct_s[11]) * (sq0 * sk1));
            dst[3] = half4(float4(ct_s[12], ct_s[13], ct_s[14], ct_s[15]) * (sq1 * sk1));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (dp == 0u) {
            threadgroup PVVal4 *pdst = reinterpret_cast<threadgroup PVVal4 *>(
                tg_p_base + rg * 512u + lane_u * 16u);
            const uint c_lo = kb_col_base + uint(sn);

            SAGE2_UNROLL
            for (uint r_sub = 0u; r_sub < 2u; ++r_sub) {
              const uint r_loc = (r_sub == 0u) ? uint(sm) : uint(sm + 8);
              const float dv = tg_mem.del_tile[(tm + r_loc) * DEL_B + b16];
              const threadgroup half4 *h0 = reinterpret_cast<const threadgroup half4 *>(tg_mem.s_xchg + (rg * 4u + 0u) * 256u + lane_u * 8u);
              const threadgroup half4 *h1 = reinterpret_cast<const threadgroup half4 *>(tg_mem.s_xchg + (rg * 4u + 1u) * 256u + lane_u * 8u);
              const threadgroup half4 *h2 = reinterpret_cast<const threadgroup half4 *>(tg_mem.s_xchg + (rg * 4u + 2u) * 256u + lane_u * 8u);
              const threadgroup half4 *h3 = reinterpret_cast<const threadgroup half4 *>(tg_mem.s_xchg + (rg * 4u + 3u) * 256u + lane_u * 8u);
              const uint ia = (r_sub == 0u) ? 0u : 1u;
              const uint ib = (r_sub == 0u) ? 2u : 3u;
              float4 a = (float4(h0[ia]) + float4(h1[ia])) + (float4(h2[ia]) + float4(h3[ia])) + dv;
              float4 b = (float4(h0[ib]) + float4(h1[ib])) + (float4(h2[ib]) + float4(h3[ib])) + dv;

              if (!full_causal_free) {
                const uint qp_raw = tg_mem.q_pos[tm + r_loc];
                const bool rv = (qp_raw & 0x80000000u) != 0u;
                const uint rp = qp_raw & 0x7fffffffu;
                const uint4 c0v = uint4(c_lo) + uint4(0u, 1u, 2u, 3u);
                const uint4 c1v = c0v + uint4(16u);
                a = select(float4(kNegInf), a,
                           rv && (c0v < max_valid_kv) && (c0v <= rp));
                b = select(float4(kNegInf), b,
                           rv && (c1v < max_valid_kv) && (c1v <= rp));
              }
              const float4 mx = max(a, b);
              float rmax = max(max(mx.x, mx.y), max(mx.z, mx.w));
              rmax = max(rmax, simd_shuffle_xor(rmax, ushort(1)));
              rmax = max(rmax, simd_shuffle_xor(rmax, ushort(8)));
              const float old_m = (r_sub == 0u) ? m0 : m1;
              const float nm = max(old_m, rmax);
              const float fac =
                  (old_m <= kNegInfHalf) ? 0.0f : fast::exp2(old_m - nm);
              if (r_sub == 0u) m0 = nm; else m1 = nm;

              if (full_causal_free) {
                a = fast::exp2(a - nm);
                b = fast::exp2(b - nm);
              } else {
                const bool emp = (nm <= kNegInfHalf);
                a = select(fast::exp2(a - nm), float4(0.0f),
                           emp || (a <= kNegInfHalf));
                b = select(fast::exp2(b - nm), float4(0.0f),
                           emp || (b <= kNegInfHalf));
              }

              const float4 ps = a + b;
              float ts = (ps.x + ps.y) + (ps.z + ps.w);
              ts += simd_shuffle_xor(ts, ushort(1));
              ts += simd_shuffle_xor(ts, ushort(8));
              if (r_sub == 0u) l0 = fma(l0, fac, ts); else l1 = fma(l1, fac, ts);

              if (r_sub == 0u) {
                pdst[0] = PVVal4(a);
                pdst[2] = PVVal4(b);
              } else {
                pdst[1] = PVVal4(a);
                pdst[3] = PVVal4(b);
              }
              if (sn == 0) tg_mem.fac[tm + r_loc] = fac;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }

      // 2. Rescale O0..O3 (ONLY 32 GPRs!)
      {
        const float fac0 = tg_mem.fac[tm + uint(sm)];
        const float fac1 = tg_mem.fac[tm + 8u + uint(sm)];
        const F8 fac_vec(fac0, fac0, fac0, fac0, fac1, fac1, fac1, fac1);
        O0 *= fac_vec;
        O1 *= fac_vec;
        O2 *= fac_vec;
        O3 *= fac_vec;
      }

      // 3. Direct-L1 Frag16x16 P x V (kh=0..1, 0 V staging, 0 PV barriers!)
      {
        const uint hbp = sp.base.page_tokens / BK;
        const uint phys_page = page_table[kb / hbp];
        const uint sub_off = (kb % hbp) * BK;
        const short valid_k = short(min(BK, tg_mem.meta[0] - kb * BK));
        const threadgroup PVVal4 *psrc =
            reinterpret_cast<const threadgroup PVVal4 *>(
                tg_p_base + rg * 512u + lane_u * 16u);
        if constexpr (IsAffine4) {
          const uint aff_grp = dp >> 1u;
          const uint base_u16 = (dp & 1u) * 16u + (uint(sn) >> 2u);
          const ulong page_bytes =
              affine4_page_byte_stride(sp.base.page_tokens, kvh_cnt, BD);
          const ulong head_code_bytes = ulong(sp.base.page_tokens) * 128ul;
          device const uchar *v_page_ptr =
              cache_value + ulong(phys_page) * page_bytes;
          device const ushort *v_codes_u16 =
              reinterpret_cast<device const ushort *>(
                  v_page_ptr + ulong(tg_mem.meta[5]) * head_code_bytes +
                  ulong(aff_grp) * (64ul * 64ul));
          device const KvAffine4VecMeta *v_meta =
              reinterpret_cast<device const KvAffine4VecMeta *>(
                  v_page_ptr + ulong(kvh_cnt) * head_code_bytes) +
              ulong(tg_mem.meta[5]) * sp.base.page_tokens;
          const uint rv00 = sub_off + uint(sm);

          SAGE2_UNROLL
          for (uint kh = 0u; kh < 2u; ++kh) {
            const PVVal4 hp0 = psrc[kh * 2u + 0u];
            const PVVal4 hp1 = psrc[kh * 2u + 1u];
            const Frag8 P(hp0.x, hp0.y, hp0.z, hp0.w, hp1.x, hp1.y, hp1.z, hp1.w);
            const uint r_base = rv00 + kh * 16u;
            const short r_rel = sm + short(kh * 16u);
            half sv0 = v_meta[r_base].scale[aff_grp];
            half zv0 = v_meta[r_base].zero[aff_grp];
            half sv1 = v_meta[r_base + 8u].scale[aff_grp];
            half zv1 = v_meta[r_base + 8u].zero[aff_grp];
            if (valid_k < 32) {
              if (!(r_rel < valid_k)) {
                sv0 = half(0.0h);
                zv0 = half(0.0h);
              }
              if (!(short(r_rel + 8) < valid_k)) {
                sv1 = half(0.0h);
                zv1 = half(0.0h);
              }
            }
            device const ushort *vp0 = v_codes_u16 + r_base * 32u + base_u16;
            device const ushort *vp1 =
                v_codes_u16 + (r_base + 8u) * 32u + base_u16;
            sage2_mma_1x2_pv<half>(
                O0, O1, P,
                sage2_dequant_v_ptr_h(vp0, vp1, sv0, zv0, sv1, zv1, 0u),
                sage2_dequant_v_ptr_h(vp0, vp1, sv0, zv0, sv1, zv1, 4u));
            sage2_mma_1x2_pv<half>(
                O2, O3, P,
                sage2_dequant_v_ptr_h(vp0, vp1, sv0, zv0, sv1, zv1, 8u),
                sage2_dequant_v_ptr_h(vp0, vp1, sv0, zv0, sv1, zv1, 12u));
          }
        } else {
          device const bfloat *v_bf =
              reinterpret_cast<device const bfloat *>(cache_value);
          const ulong bf16_kv_stride = ulong(kvh_cnt) * BD;
          const ulong base_row = ulong(phys_page) * sp.base.page_tokens + sub_off;
          const ulong col_base = ulong(tg_mem.meta[5]) * BD + dp * 64u + uint(sn);

          SAGE2_NOUNROLL
          for (uint kh = 0u; kh < 2u; ++kh) {
            const PVVal4 hp0 = psrc[kh * 2u + 0u];
            const PVVal4 hp1 = psrc[kh * 2u + 1u];
            const Frag8 P(hp0.x, hp0.y, hp0.z, hp0.w, hp1.x, hp1.y, hp1.z, hp1.w);
            const short r_rel = sm + short(kh * 16u);
            const bool ok0 = (r_rel < valid_k);
            const bool ok1 = (short(r_rel + 8) < valid_k);
            device const bfloat *vp0 =
                v_bf + (base_row + uint(r_rel)) * bf16_kv_stride + col_base;
            device const bfloat *vp1 =
                v_bf + (base_row + uint(r_rel + 8)) * bf16_kv_stride + col_base;
            sage2_mma_1x2_pv<bfloat>(
                O0, O1, P, sage2_load_v_ptr_bf(vp0, vp1, ok0, ok1, 0u),
                sage2_load_v_ptr_bf(vp0, vp1, ok0, ok1, 16u));
            sage2_mma_1x2_pv<bfloat>(
                O2, O3, P, sage2_load_v_ptr_bf(vp0, vp1, ok0, ok1, 32u),
                sage2_load_v_ptr_bf(vp0, vp1, ok0, ok1, 48u));
          }
        }
      }
    }

    // Epilogue: normalize by 1/l & apply Qwen3.5 sigmoid output gate!
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dp == 0u && sn == 0) {
      tg_mem.fac[tm + uint(sm)] = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
      tg_mem.fac[tm + 8u + uint(sm)] = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
      const uint kvh_e = tg_mem.meta[5];
      const uint q_blk_e = tg_mem.meta[6];
      const uint repeat = sp.base.query_heads / kvh_cnt;
      const uint bm_tok = M_TILE / repeat;
      const uint q_m_base = q_blk_e * M_TILE;
      const uint q_base = kFlattened ? (q_m_base / repeat) : (q_blk_e * bm_tok);
      const uint total_m_rows =
          kFlattened ? (sp.base.tokens * repeat)
                     : (min(bm_tok, sp.base.tokens - q_base) * repeat);
      const uint d_offset = dp * 64u;
      {
        uint tok = 0u, g_sub = 0u;
        if (decode_m_row<kFlattened>(tm + uint(sm), q_m_base, q_base, total_m_rows, repeat, tok, g_sub)) {
          const float rcp = tg_mem.fac[tm + uint(sm)];
          const uint qh_o = kvh_e * repeat + g_sub;
          device const bfloat *gp = gate + ulong(tok) * sp.base.gate_token_stride + ulong(qh_o) * sp.base.gate_head_stride + d_offset + uint(sn);
          device bfloat *op = output + (ulong(tok) * sp.base.query_heads + qh_o) * BD + d_offset + uint(sn);
          *reinterpret_cast<device bfloat4 *>(op +  0u) = bfloat4((float4(O0[0],O0[1],O0[2],O0[3]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp +  0u)))));
          *reinterpret_cast<device bfloat4 *>(op + 16u) = bfloat4((float4(O1[0],O1[1],O1[2],O1[3]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp + 16u)))));
          *reinterpret_cast<device bfloat4 *>(op + 32u) = bfloat4((float4(O2[0],O2[1],O2[2],O2[3]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp + 32u)))));
          *reinterpret_cast<device bfloat4 *>(op + 48u) = bfloat4((float4(O3[0],O3[1],O3[2],O3[3]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp + 48u)))));
        }
      }
      {
        uint tok = 0u, g_sub = 0u;
        if (decode_m_row<kFlattened>(tm + 8u + uint(sm), q_m_base, q_base, total_m_rows, repeat, tok, g_sub)) {
          const float rcp = tg_mem.fac[tm + 8u + uint(sm)];
          const uint qh_o = kvh_e * repeat + g_sub;
          device const bfloat *gp = gate + ulong(tok) * sp.base.gate_token_stride + ulong(qh_o) * sp.base.gate_head_stride + d_offset + uint(sn);
          device bfloat *op = output + (ulong(tok) * sp.base.query_heads + qh_o) * BD + d_offset + uint(sn);
          *reinterpret_cast<device bfloat4 *>(op +  0u) = bfloat4((float4(O0[4],O0[5],O0[6],O0[7]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp +  0u)))));
          *reinterpret_cast<device bfloat4 *>(op + 16u) = bfloat4((float4(O1[4],O1[5],O1[6],O1[7]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp + 16u)))));
          *reinterpret_cast<device bfloat4 *>(op + 32u) = bfloat4((float4(O2[4],O2[5],O2[6],O2[7]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp + 32u)))));
          *reinterpret_cast<device bfloat4 *>(op + 48u) = bfloat4((float4(O3[4],O3[5],O3[6],O3[7]) * rcp) / (float4(1.0f) + fast::exp(-float4(*reinterpret_cast<device const bfloat4 *>(gp + 48u)))));
        }
      }
    }
    return;
  }
  }

}

} // namespace sage2

// ============================================================================
// WP2.1 Linear O(N_KV) K-Smoothing Prepass Entry Points
// ============================================================================
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_sage2_compute_valid_kv(
    device const uint *positions [[buffer(0)]],
    device uint *valid_kv_out [[buffer(1)]],
    constant Sage2PrepareParams &prep [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup sage2::Sage2ValidKvTg tg_mem;
  const uint tokens = prep.base.tokens;
  uint lmax = 0u;
  for (uint i = tid; i < tokens; i += 256u) {
    lmax = max(lmax, positions[i] + 1u);
  }
  const uint smax = simd_max(lmax);
  if (lane == 0u) {
    tg_mem.sg_max[sg] = smax;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    uint gmax = 0u;
    SAGE2_UNROLL
    for (uint s = 0u; s < 8u; ++s) {
      gmax = max(gmax, tg_mem.sg_max[s]);
    }
    valid_kv_out[0] = min(gmax, prep.max_kv_cap);
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_sage2_prepare_k_affine4_g128_hd256(
    device const uchar *cache_key [[buffer(0)]],
    device const uint *page_table [[buffer(1)]],
    device const uint *valid_kv_buf [[buffer(2)]],
    device int8_t *out_k_int8 [[buffer(3)]],
    device float *out_mu_k [[buffer(4)]],
    device float *out_s_k [[buffer(5)]],
    constant Sage2PrepareParams &prep [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup sage2::Sage2PrepareKTg tg_mem;
  sage2::prepare_k_linear_impl<true>(cache_key, page_table, valid_kv_buf,
                                     out_k_int8, out_mu_k, out_s_k, prep,
                                     tg_mem, tid, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_sage2_prepare_k_bf16_hd256(
    device const uchar *cache_key [[buffer(0)]],
    device const uint *page_table [[buffer(1)]],
    device const uint *valid_kv_buf [[buffer(2)]],
    device int8_t *out_k_int8 [[buffer(3)]],
    device float *out_mu_k [[buffer(4)]],
    device float *out_s_k [[buffer(5)]],
    constant Sage2PrepareParams &prep [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup sage2::Sage2PrepareKTg tg_mem;
  sage2::prepare_k_linear_impl<false>(cache_key, page_table, valid_kv_buf,
                                      out_k_int8, out_mu_k, out_s_k, prep,
                                      tg_mem, tid, lane, sg, tg);
}

// ============================================================================
// Diagnostic Identity & Lane-Coordinate Probe Kernels (Zero Spills!)
// ============================================================================
[[max_total_threads_per_threadgroup(32)]]
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
  tensor<device int32_t, dextents<int32_t, 2>, tensor_inline> tOut(
      out_s, dextents<int32_t, 2>(32, 16));
  op.run(tQ, tK, tOut);

  const short2 sc = sage2::sage2_frag_coord(lane);
  const uint fm = uint(sc.y);
  const uint fn = uint(sc.x);
  const uint cbase = uint(lane) * 32u;
  SAGE2_UNROLL
  for (ushort i = 0u; i < 16u; ++i) {
    out_coords[cbase + uint(i) * 2u + 0u] = fm + (((i & 4u) != 0u) ? 8u : 0u);
    out_coords[cbase + uint(i) * 2u + 1u] =
        fn + (((i & 8u) != 0u) ? 16u : 0u) + uint(i & 3u);
  }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_sage2_probe_delta_16x16x128(
    device const int8_t *in_q [[buffer(0)]],
    device const float *in_mu [[buffer(1)]],
    device float *out_d [[buffer(2)]],
    device uint *out_flags [[buffer(3)]],
    constant uint &dummy [[buffer(4)]],
    ushort lane [[thread_index_in_simdgroup]]) {
  if (dummy != 0u) return;
  threadgroup int8_t tg_q[16 * 128];
  for (uint i = lane; i < 16u * 128u; i += 32u) tg_q[i] = in_q[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr auto mu_desc = matmul2d_descriptor(
      16, 16, 128, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<mu_desc, execution_simdgroup> mu_op;
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ(
      tg_q, dextents<int32_t, 2>(128, 16));
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tMu(
      const_cast<device float *>(in_mu), dextents<int32_t, 2>(128, 16));
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tOut(
      out_d, dextents<int32_t, 2>(16, 16));
  mu_op.run(tQ, tMu, tOut);
  if (lane == 0u) {
    out_flags[0] = 1u;
    out_flags[1] = 64u;
    out_flags[2] = 64u;
    out_flags[3] = 8u;
  }
}

[[max_total_threads_per_threadgroup(32)]]
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

  for (uint idx = lane; idx < 16u * 8u; idx += 32u) {
    const uint base = idx * 4u;
    const float4 v(in_p[base], in_p[base + 1u], in_p[base + 2u],
                   in_p[base + 3u]);
    reinterpret_cast<threadgroup packed_uchar4 *>(tg_p)[idx] =
        pack<metal_fp8_e4m3_format>(v * sage2::kFp8E4M3Max).as_storage_type();
  }
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

  sage2::TgFp8Plain tP(tg_p, dextents<int32_t, 2>(32, 16));
  sage2::TgMxfp8VT tVT(tg_vt, dextents<int32_t, 2>(32, 128),
                       sage2::TgUe8ScaleTag(tg_sc));
  constexpr auto desc_mul = matmul2d_descriptor(
      16, 128, 32, false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<desc_mul, execution_simdgroup> op_mul;
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tOut(
      out_o, dextents<int32_t, 2>(128, 16));
  op_mul.run(tP, tVT, tOut);
  simdgroup_barrier(mem_flags::mem_device);
  for (uint i = lane; i < 16u * 128u; i += 32u) {
    out_o[i] *= sage2::kInvFp8E4M3Max;
  }
}

// ============================================================================
// 8 Production SageAttention2 Entry Points (WM=2, WN=4, 13.03 KiB FP16 / 15.16 KiB MXFP8, 0 B Spills)
// ============================================================================
#define DEFINE_SAGE2_KERNEL(NAME, WM_VAL, MAX_THR, IS_AFF4, USE_MXFP8, SPARSE) \
  [[max_total_threads_per_threadgroup(MAX_THR)]]                               \
  kernel void NAME(                                                            \
      device const bfloat *query [[buffer(0)]],                                \
      device const bfloat *gate [[buffer(1)]],                                 \
      device const int8_t *k_int8 [[buffer(2)]],                               \
      device const uchar *cache_value [[buffer(3)]],                           \
      device bfloat *output [[buffer(4)]],                                     \
      device const uint *page_table [[buffer(5)]],                             \
      device const uint *positions [[buffer(6)]],                              \
      device const float *mu_k [[buffer(7)]],                                  \
      device const float *s_k [[buffer(8)]],                                   \
      device const uint *valid_kv_buf [[buffer(9)]],                           \
      device const ushort *active_indices [[buffer(10)]],                      \
      device const uint *active_counts [[buffer(11)]],                         \
      constant Sage2AttentionParams &sp [[buffer(12)]],                        \
      ushort lane [[thread_index_in_simdgroup]],                               \
      ushort sg [[simdgroup_index_in_threadgroup]],                            \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup sage2::Sage2TgStorage<WM_VAL, USE_MXFP8> tg_mem;               \
    sage2::paged_flash_prefill_sage2_hd256_impl<WM_VAL, IS_AFF4, USE_MXFP8,    \
                                                SPARSE>(                       \
        query, gate, k_int8, cache_value, output, page_table, positions, mu_k, \
        s_k, valid_kv_buf, active_indices, active_counts, sp, tg_mem, lane,    \
        sg, tg);                                                               \
  }


DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_fp16_affine4_g128_hd256_wm2, 2u, 256, true, false, false)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_fp16_affine4_g128_hd256_wm2, 2u, 256, true, false, true)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_mxfp8_affine4_g128_hd256_wm2, 2u, 256, true, true, false)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_affine4_g128_hd256_wm2, 2u, 256, true, true, true)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_fp16_bf16_hd256_wm2, 2u, 256, false, false, false)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_fp16_bf16_hd256_wm2, 2u, 256, false, false, true)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_mxfp8_bf16_hd256_wm2, 2u, 256, false, true, false)
DEFINE_SAGE2_KERNEL(omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_bf16_hd256_wm2, 2u, 256, false, true, true)

#undef DEFINE_SAGE2_KERNEL
