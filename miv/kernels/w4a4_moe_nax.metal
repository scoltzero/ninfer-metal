// Phase 3 WP 3A: High-Throughput MoE MXFP4 W4A4 Prefill Kernels (Zero .miv v2 Repack)
// (Fused Gate + Up + SwiGLU [Mode A BF16 / Mode B FWHT-64 + A4] & W4A4 Down Scatter)
// Executed on Apple10's 251.5 TOPS INT8 x INT8 -> INT32 NAX TensorOps pipe via dual
// per-row |de| <= 3 shift-merge + 2 KiB constant-buffer pre-shifted byte LUT:
// - All TG SRAM strictly <= 10,240 B (3 TGs/core on Apple10 32 KiB/core SRAM), 0 B spills:
//   * MoE MXFP4 W4A4 SwiGLU (Mode A & Mode B, 128t, BK=64, 32x32/16x32 i8xi8): 9,520 B (9.30 KiB)
//   * MoE MXFP4 W4A4 Down Scatter (128t, BK=64, 32x32/16x32 i8xi8):            9,264 B (9.05 KiB)
// - Consumes 3D untiled expert weights (`mxfp4_expert_rowsplit_k128_v1`),
//   tiled input `(a4_codes, a4_scales)` from RMSNorm+A4, and reuses existing
//   `MoeNaxWorkspace::mid_qa` (row stride `f` bytes: 64B disjoint tile slots where hn=0 stashes
//   32 bfloats and epilogue overwrites [0..31] with 32B MXFP4 codes) & `mid_sa` (`[row, f/32]`).

#include "common_math.h"
#include "fwht64.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct W4A4MoeGemmParams {
  uint n;          // output features (f for SwiGLU, h for Down Scatter)
  uint k;          // input features  (h for SwiGLU, f for Down Scatter)
  uint padded_k;   // untiled weight padded_k (>= k, % 128 == 0)
  uint max_tiles;  // bit 31: use_gx32, bits 0..30: max_tiles
  uint mp;         // input padded rows (for tiled a4_codes/scales)
  uint reserved;   // Mode B SwiGLU: bits 0..30 = mid_max_rows, bit 31 = skip_fwht64 (0 => FWHT-64)
};

namespace {

constant constexpr ulong kW4A4MoeHighOccSramBytes = 10240ul;
constant constexpr uint kW4A4MoeStride = 64u;
constant constexpr uint kW4A4MoeEpiStride = 68u;

// Constant-buffer 2 KiB pre-shifted byte LUT for lossless MXFP4 (2 * E2M1 << sh, sh in 0..3).
// Passed via Encoder::dynamic(...) (icb=1 + MTL_SHADER_VALIDATION=1 clean, 0 B extra SRAM).
inline void w4a4_moe_unpack_lut_sh_u4(
    uint4 raw, uint sh, constant ushort *mxfp4_lut,
    thread uint4 &o0, thread uint4 &o1) {
  constant ushort *lut = mxfp4_lut + (sh << 8u);
  o0.x = uint(lut[(raw.x >>  0u) & 0xFFu]) | (uint(lut[(raw.x >>  8u) & 0xFFu]) << 16u);
  o0.y = uint(lut[(raw.x >> 16u) & 0xFFu]) | (uint(lut[(raw.x >> 24u) & 0xFFu]) << 16u);
  o0.z = uint(lut[(raw.y >>  0u) & 0xFFu]) | (uint(lut[(raw.y >>  8u) & 0xFFu]) << 16u);
  o0.w = uint(lut[(raw.y >> 16u) & 0xFFu]) | (uint(lut[(raw.y >> 24u) & 0xFFu]) << 16u);
  o1.x = uint(lut[(raw.z >>  0u) & 0xFFu]) | (uint(lut[(raw.z >>  8u) & 0xFFu]) << 16u);
  o1.y = uint(lut[(raw.z >> 16u) & 0xFFu]) | (uint(lut[(raw.z >> 24u) & 0xFFu]) << 16u);
  o1.z = uint(lut[(raw.w >>  0u) & 0xFFu]) | (uint(lut[(raw.w >>  8u) & 0xFFu]) << 16u);
  o1.w = uint(lut[(raw.w >> 16u) & 0xFFu]) | (uint(lut[(raw.w >> 24u) & 0xFFu]) << 16u);
}

struct W4A4MoeSwiTgStorage {
  union {
    struct {
      alignas(16) int8_t As_i8[64u * kW4A4MoeStride]; // 4,096 B
      alignas(16) int8_t Ws_i8[64u * kW4A4MoeStride]; // 4,096 B
      float Sa0_s[64u];                               //   256 B
      float Sa1_s[64u];                               //   256 B
      float Sw0_s[64u];                               //   256 B
      float Sw1_s[64u];                               //   256 B
      uint slow_a[4u];                                //    16 B
      uint slow_w[4u];                                //    16 B
    } kloop;                                          // 9,248 B
    alignas(16) bfloat SwiBuf[64u * kW4A4MoeEpiStride]; // 8,704 B
  };
  uint tg_tok[64u];                                   //   256 B
  uint tg_meta[4u];                                   //    16 B {expert, rb, valid_rows, tg_n0_base}
};
static_assert(sizeof(W4A4MoeSwiTgStorage) == 9520ul,
              "W4A4MoeSwiTgStorage must be 9,520 B (9.30 KiB)");
static_assert(sizeof(W4A4MoeSwiTgStorage) <= kW4A4MoeHighOccSramBytes,
              "W4A4MoeSwiTgStorage exceeds 10.0 KiB");

struct W4A4MoeDownTgStorage {
  alignas(16) int8_t As_i8[64u * kW4A4MoeStride]; // 4,096 B
  alignas(16) int8_t Ws_i8[64u * kW4A4MoeStride]; // 4,096 B
  float Sa0_s[64u];                               //   256 B
  float Sa1_s[64u];                               //   256 B
  float Sw0_s[64u];                               //   256 B
  float Sw1_s[64u];                               //   256 B
  uint slow_a[4u];                                //    16 B
  uint slow_w[4u];                                //    16 B
  uint tg_meta[4u];                               //    16 B {expert, rb, valid_rows, tg_n0}
};
static_assert(sizeof(W4A4MoeDownTgStorage) == 9264ul,
              "W4A4MoeDownTgStorage must be 9,264 B (9.05 KiB)");
static_assert(sizeof(W4A4MoeDownTgStorage) <= kW4A4MoeHighOccSramBytes,
              "W4A4MoeDownTgStorage exceeds 10.0 KiB");

inline bool w4a4_moe_has_apple10_layout() {
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 32, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto b = op.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  return (int(b.get_multidimensional_index(ushort(2u))[0]) -
          int(b.get_multidimensional_index(ushort(0u))[0])) == 2;
}

inline float w4a4_moe_e8m0_safe(uchar code) {
  if (code == 255u) return 0.0f;
  return e8m0_to_float(code);
}

inline void w4a4_moe_quantize_64_sg32(
    float v0,
    float v1,
    bool apply_fwht64,
    uint lane,
    device uchar *dst_codes32,
    device uchar *dst_scales2) {
  if (apply_fwht64) {
    omlx_m5_fwht64_sg32(v0, v1, lane);
  }
  float m = fmax(fabs(v0), fabs(v1));
  m = fmax(m, simd_shuffle_xor(m, 1u));
  m = fmax(m, simd_shuffle_xor(m, 2u));
  m = fmax(m, simd_shuffle_xor(m, 4u));
  m = fmax(m, simd_shuffle_xor(m, 8u));

  const uchar sc = nax_encode_e8m0_round_up(m / 6.0f);
  const float scale = nax_decode_e8m0(sc);
  const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
  const uchar c0 = nax_encode_e2m1(v0 * inv_scale);
  const uchar c1 = nax_encode_e2m1(v1 * inv_scale);
  dst_codes32[lane] = uchar(c0 | (c1 << 4u));
  if ((lane & 15u) == 0u) {
    dst_scales2[lane >> 4u] = sc;
  }
}

// Writes 32B MXFP4 codes into row-major mid_qa (row stride = f_dim bytes, disjoint 64B slots per
// col_tile so hn=0 64B bfloat stash and final 32B FP4 write never overlap across TGs!).
template <uint RowsPerSg>
inline void w4a4_moe_swiglu_epilogue_a4_phase(
    const threadgroup bfloat *SwiBuf,
    uint rb,
    uint base_row_offset,
    uint valid_rows,
    uint rb_8ceil,
    uint tg_n0_base,
    device uchar *mid_qa,
    device uchar *mid_sa,
    uint f_dim,
    bool apply_fwht64,
    uint lane,
    uint sg) {
  const uint base_r = sg * RowsPerSg;
  const ulong row_c_stride = ulong(f_dim);
  const ulong row_s_stride = ulong(f_dim >> 5u);
  const ulong col_c_off = ulong(tg_n0_base);
  const ulong col_s_off = ulong(tg_n0_base >> 5u);

  #pragma clang loop unroll(disable)
  for (uint rr = 0u; rr < RowsPerSg; ++rr) {
    const uint r_l = base_r + rr;
    const uint rm = base_row_offset + r_l;
    if (rm < valid_rows) {
      const bfloat2 s2 =
          reinterpret_cast<const threadgroup bfloat2 *>(SwiBuf + r_l * kW4A4MoeEpiStride)[lane];
      w4a4_moe_quantize_64_sg32(
          float(s2.x), float(s2.y), apply_fwht64, lane,
          mid_qa + ulong(rb + rm) * row_c_stride + col_c_off,
          mid_sa + ulong(rb + rm) * row_s_stride + col_s_off);
    } else if (rm < rb_8ceil) {
      mid_qa[ulong(rb + rm) * row_c_stride + col_c_off + lane] = 0u;
      if ((lane & 15u) == 0u) {
        mid_sa[ulong(rb + rm) * row_s_stride + col_s_off + (lane >> 4u)] = 0x7Fu;
      }
    }
  }
}

// ============================================================================
// MoE MXFP4 W4A4 SwiGLU (BM=64, OUT_N=32 Mode A / 64 Mode B, 128t, i8xi8 shift-merge)
// ============================================================================
template <bool OutputA4>
inline void moe_mxfp4_w4a4_nax_swiglu_bm64_out64_impl(
    device const uchar *a_codes,
    device const uchar *a_scales,
    device const uint *tile_assign,
    device const uint *tile_expert,
    device const uint *tile_valid_rows,
    device const uint *total_tiles,
    device const uchar *g_codes,
    device const uchar *g_scales,
    device const uchar *u_codes,
    device const uchar *u_scales,
    device bfloat *mid_bf16,
    device uchar *mid_qa,
    device uchar *mid_sa,
    constant W4A4MoeGemmParams &p,
    constant ushort *mxfp4_lut,
    threadgroup W4A4MoeSwiTgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM_TG = 64u;
  constexpr uint OUT_N_TG = OutputA4 ? 64u : 32u;
  constexpr uint kPasses = OutputA4 ? 2u : 1u;

  const uint n_cols = (p.n + OUT_N_TG - 1u) / OUT_N_TG;
  const bool use_gx32 = (p.max_tiles >> 31u) != 0u;
  const uint m_tile = use_gx32 ? ((tg.y / n_cols) * 32u + tg.x) : tg.y;
  const uint col_tile = use_gx32 ? (tg.y % n_cols) : tg.x;
  const uint tg_n0_base = col_tile * OUT_N_TG;
  if (m_tile >= total_tiles[1] || col_tile >= n_cols || tg_n0_base >= p.n) {
    return;
  }
  const uint valid_rows = tile_valid_rows[m_tile];
  if (valid_rows == 0u) {
    return;
  }
  const uint packed = tile_expert[m_tile];
  const uint expert = packed & 0xFFFFu;
  const uint rb = (packed >> 16u) * 8u;

  {
    const uint tid = sg * 32u + lane;
    if (tid < BM_TG) {
      uint tok = 0xFFFFFFFFu;
      if (tid < valid_rows) {
        const uint assign = tile_assign[rb + tid];
        if (assign != 0xFFFFFFFFu) {
          tok = assign >> 3u;
        }
      }
      tg_mem.tg_tok[tid] = tok;
    }
    if (tid == 0u) {
      tg_mem.tg_meta[0] = expert;
      tg_mem.tg_meta[1] = rb;
      tg_mem.tg_meta[2] = valid_rows;
      tg_mem.tg_meta[3] = tg_n0_base;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint k_tiles128 = p.k >> 7u;
  const uint k_blocks = p.k >> 6u;
  const ulong c_stride = ulong(p.padded_k >> 1u);
  const ulong s_stride = ulong(p.padded_k >> 5u);

  if (w4a4_moe_has_apple10_layout()) {
    constexpr auto desc16_m = matmul2d_descriptor(
        16, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply);
    constexpr auto desc16_a = matmul2d_descriptor(
        16, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto desc32_m = matmul2d_descriptor(
        32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply);
    constexpr auto desc32_a = matmul2d_descriptor(
        32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc16_m, execution_simdgroup> op16_m;
    matmul2d<desc16_a, execution_simdgroup> op16_a;
    matmul2d<desc32_m, execution_simdgroup> op32_m;
    matmul2d<desc32_a, execution_simdgroup> op32_a;
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
        tg_mem.kloop.As_i8, dextents<int32_t, 2>(int(kW4A4MoeStride), 64));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
        tg_mem.kloop.Ws_i8, dextents<int32_t, 2>(int(kW4A4MoeStride), 64));

    const bool sg_is_up = ((sg & 1u) != 0u);
    const uint sg_m = sg >> 1u;
    const uint sg_n = sg & 1u;
    const uint qid = lane >> 2u;
    const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
    const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
    const uint sw_base = sg_n * 32u + fn;

    // Fast path for valid_rows <= 32: half A loads/unpacks, 16x32x32 MMA, 16-reg acc0!
    if (tg_mem.tg_meta[2] <= 32u) {
      #pragma clang loop unroll(full)
      for (uint hn = 0u; hn < kPasses; ++hn) {
        const uint w_col = tg_mem.tg_meta[3] + hn * 32u + (sg >> 1u) * 16u + (lane >> 1u);
        const bool w_live = (w_col < p.n);
        const ulong w_exp_row = ulong(tg_mem.tg_meta[0]) * ulong(p.n) + ulong(w_col);

        float acc0[16];
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) acc0[i] = 0.0f;

        for (uint kb = 0u; kb < k_blocks; ++kb) {
          const uint blk128 = kb >> 1u;
          const uint half64 = kb & 1u;
          const uint tid = sg * 32u + lane;
          const uint r = tid >> 1u;
          const uint col32 = tid & 1u;
          const uint sub32 = half64 * 2u + col32;
          const uint w_row = sg * 16u + (lane >> 1u);

          uint4 raw_a = uint4(0u);
          uchar2 ea = uchar2(0x7Fu, 0x7Fu);
          bool a_live = false;
          if (r < 32u) {
            const uint tok = tg_mem.tg_tok[r];
            if (tok != 0xFFFFFFFFu) {
              a_live = true;
              const ulong ai = ulong(tok >> 4u) * ulong(k_tiles128) + ulong(blk128);
              const ulong rm = ulong(tok & 15u);
              raw_a = reinterpret_cast<device const uint4 *>(a_codes + ai * 1024ul + rm * 64ul)[sub32];
              ea = *reinterpret_cast<device const uchar2 *>(
                  a_scales + ai * 64ul + rm * 4ul + ulong(half64) * 2ul);
            }
          }
          uint4 raw_w = uint4(0u);
          uchar2 ew = uchar2(0x7Fu, 0x7Fu);
          if (w_live) {
            const ulong c_off = w_exp_row * c_stride + ulong(kb * 2u + col32) * 16ul;
            const ulong s_off = w_exp_row * s_stride + ulong(kb) * 2ul;
            if (!sg_is_up) {
              raw_w = *reinterpret_cast<device const uint4 *>(g_codes + c_off);
              ew = *reinterpret_cast<device const uchar2 *>(g_scales + s_off);
            } else {
              raw_w = *reinterpret_cast<device const uint4 *>(u_codes + c_off);
              ew = *reinterpret_cast<device const uchar2 *>(u_scales + s_off);
            }
          }

          const int dea = (a_live && ea.x != 255u && ea.y != 255u)
              ? (int(ea.y) - int(ea.x)) : (a_live ? 255 : 0);
          const bool fa = (abs(dea) <= 3);
          const uint sha = fa
              ? ((col32 == 0u) ? ((dea < 0) ? uint(-dea) : 0u) : ((dea > 0) ? uint(dea) : 0u))
              : 0u;
          const uint sg_slow_a = simd_any(!fa) ? 1u : 0u;

          const int dew = (w_live && ew.x != 255u && ew.y != 255u)
              ? (int(ew.y) - int(ew.x)) : (w_live ? 255 : 0);
          const bool fw = (abs(dew) <= 3);
          const uint shw = fw
              ? ((col32 == 0u) ? ((dew < 0) ? uint(-dew) : 0u) : ((dew > 0) ? uint(dew) : 0u))
              : 0u;
          const uint sg_slow_w = simd_any(!fw) ? 1u : 0u;

          uint4 au0, au1, wu0, wu1;
          if (r < 32u) w4a4_moe_unpack_lut_sh_u4(raw_a, sha, mxfp4_lut, au0, au1);
          w4a4_moe_unpack_lut_sh_u4(raw_w, shw, mxfp4_lut, wu0, wu1);

          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (r < 32u) {
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + r * kW4A4MoeStride + col32 * 32u + 0u) = au0;
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + r * kW4A4MoeStride + col32 * 32u + 16u) = au1;
            if (col32 == 0u) {
              if (fa) {
                const float sa = a_live ? (0.5f * w4a4_moe_e8m0_safe(min(ea.x, ea.y))) : 0.0f;
                tg_mem.kloop.Sa0_s[r] = sa;
                tg_mem.kloop.Sa1_s[r] = sa;
              } else {
                tg_mem.kloop.Sa0_s[r] = a_live ? (0.5f * w4a4_moe_e8m0_safe(ea.x)) : 0.0f;
                tg_mem.kloop.Sa1_s[r] = a_live ? (0.5f * w4a4_moe_e8m0_safe(ea.y)) : 0.0f;
              }
            }
          }
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A4MoeStride + col32 * 32u + 0u) = wu0;
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A4MoeStride + col32 * 32u + 16u) = wu1;
          if (col32 == 0u) {
            if (fw) {
              const float sw = w_live ? (0.5f * w4a4_moe_e8m0_safe(min(ew.x, ew.y))) : 0.0f;
              tg_mem.kloop.Sw0_s[w_row] = sw;
              tg_mem.kloop.Sw1_s[w_row] = sw;
            } else {
              tg_mem.kloop.Sw0_s[w_row] = w_live ? (0.5f * w4a4_moe_e8m0_safe(ew.x)) : 0.0f;
              tg_mem.kloop.Sw1_s[w_row] = w_live ? (0.5f * w4a4_moe_e8m0_safe(ew.y)) : 0.0f;
            }
          }
          if (lane == 0u) {
            tg_mem.kloop.slow_a[sg] = sg_slow_a;
            tg_mem.kloop.slow_w[sg] = sg_slow_w;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          const uint m_off = sg_m * 16u;
          auto a0 = tA.slice(0, int(m_off));
          auto b0 = tB.slice(0, int(sg_n * 32u));
          auto ci = op16_m.template get_destination_cooperative_tensor<
              remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
          const uint sg_slow = tg_mem.kloop.slow_a[sg_m] |
                               tg_mem.kloop.slow_w[sg_n * 2u] | tg_mem.kloop.slow_w[sg_n * 2u + 1u];
          if (sg_slow == 0u) {
            auto a1 = tA.slice(32, int(m_off));
            auto b1 = tB.slice(32, int(sg_n * 32u));
            op16_m.run(a0, b0, ci);
            op16_a.run(a1, b1, ci);
            const float sa0 = tg_mem.kloop.Sa0_s[m_off + fm];
            const float sa1 = tg_mem.kloop.Sa0_s[m_off + 8u + fm];
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) {
              const float sw0 = tg_mem.kloop.Sw0_s[sw_base + c];
              const float sw1 = tg_mem.kloop.Sw0_s[sw_base + 16u + c];
              acc0[ 0u + c] = fma(sa0 * sw0, float(ci[ 0u + c]), acc0[ 0u + c]);
              acc0[ 4u + c] = fma(sa1 * sw0, float(ci[ 4u + c]), acc0[ 4u + c]);
              acc0[ 8u + c] = fma(sa0 * sw1, float(ci[ 8u + c]), acc0[ 8u + c]);
              acc0[12u + c] = fma(sa1 * sw1, float(ci[12u + c]), acc0[12u + c]);
            }
          } else {
            #pragma clang loop unroll(disable)
            for (uint hk = 0u; hk < 2u; ++hk) {
              auto ah = tA.slice(int(hk * 32u), int(m_off));
              auto bh = tB.slice(int(hk * 32u), int(sg_n * 32u));
              op16_m.run(ah, bh, ci);
              threadgroup const float *sap = (hk == 0u) ? tg_mem.kloop.Sa0_s : tg_mem.kloop.Sa1_s;
              threadgroup const float *swp = (hk == 0u) ? tg_mem.kloop.Sw0_s : tg_mem.kloop.Sw1_s;
              const float sa0 = sap[m_off + fm];
              const float sa1 = sap[m_off + 8u + fm];
              #pragma clang loop unroll(full)
              for (uint c = 0u; c < 4u; ++c) {
                const float sw0 = swp[sw_base + c];
                const float sw1 = swp[sw_base + 16u + c];
                acc0[ 0u + c] = fma(sa0 * sw0, float(ci[ 0u + c]), acc0[ 0u + c]);
                acc0[ 4u + c] = fma(sa1 * sw0, float(ci[ 4u + c]), acc0[ 4u + c]);
                acc0[ 8u + c] = fma(sa0 * sw1, float(ci[ 8u + c]), acc0[ 8u + c]);
                acc0[12u + c] = fma(sa1 * sw1, float(ci[12u + c]), acc0[12u + c]);
              }
            }
          }
        }

        const uint rb_cur = tg_mem.tg_meta[1];
        const uint vr = tg_mem.tg_meta[2];
        const uint n0_base = tg_mem.tg_meta[3];
        const uint m_off = sg_m * 16u;
        const uint n0_sg = n0_base + hn * 32u + sg_n * 16u;

        if constexpr (!OutputA4) {
          (void)mid_qa; (void)mid_sa;
          const uint rb_8c = (vr + 7u) & ~7u;
          if (m_off < vr && n0_sg + fn < p.n) {
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint rm = m_off + fm + h * 8u;
              if (rm < vr) {
                const uint e = h * 4u;
                bfloat4 out_b;
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  out_b[j] = bfloat(nax_silu_f32(float(bfloat(acc0[e + j]))) *
                                    float(bfloat(acc0[8u + e + j])));
                }
                *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb_cur + rm) * p.n + n0_sg + fn) = out_b;
              } else if (rm < rb_8c) {
                *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb_cur + rm) * p.n + n0_sg + fn) = bfloat4(bfloat(0.0f));
              }
            }
          }
        } else {
          (void)mid_bf16;
          const ulong stash_col_off = ulong(n0_base) + ulong(sg_n * 16u + fn) * 2ul;
          if (hn == 0u) {
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint rm = m_off + fm + h * 8u;
              if (rm < vr) {
                bfloat4 out_b = bfloat4(bfloat(0.0f));
                if (n0_sg + fn < p.n) {
                  const uint e = h * 4u;
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    out_b[j] = (n0_sg + fn + j < p.n)
                        ? bfloat(nax_silu_f32(float(bfloat(acc0[e + j]))) *
                                 float(bfloat(acc0[8u + e + j])))
                        : bfloat(0.0f);
                  }
                }
                *reinterpret_cast<device bfloat4 *>(mid_qa + ulong(rb_cur + rm) * p.n + stash_col_off) = out_b;
              }
            }
          } else {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint rm = m_off + fm + h * 8u;
              bfloat4 s0 = bfloat4(bfloat(0.0f));
              bfloat4 s1 = bfloat4(bfloat(0.0f));
              if (rm < vr) {
                s0 = *reinterpret_cast<device const bfloat4 *>(mid_qa + ulong(rb_cur + rm) * p.n + stash_col_off);
                if (n0_sg + fn < p.n) {
                  const uint e = h * 4u;
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    s1[j] = (n0_sg + fn + j < p.n)
                        ? bfloat(nax_silu_f32(float(bfloat(acc0[e + j]))) *
                                 float(bfloat(acc0[8u + e + j])))
                        : bfloat(0.0f);
                  }
                }
              }
              *reinterpret_cast<threadgroup bfloat4 *>(tg_mem.SwiBuf + rm * kW4A4MoeEpiStride + sg_n * 16u + fn) = s0;
              *reinterpret_cast<threadgroup bfloat4 *>(tg_mem.SwiBuf + rm * kW4A4MoeEpiStride + 32u + sg_n * 16u + fn) = s1;
            }
          }
        }
      }
      if constexpr (OutputA4) {
        const uint rb_cur = tg_mem.tg_meta[1];
        const uint vr = tg_mem.tg_meta[2];
        const uint n0_base = tg_mem.tg_meta[3];
        const uint rb_8c = (vr + 7u) & ~7u;
        const bool apply_fwht64 = ((p.reserved >> 31u) == 0u);
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        w4a4_moe_swiglu_epilogue_a4_phase<8u>(
            tg_mem.SwiBuf, rb_cur, 0u, vr, rb_8c, n0_base,
            mid_qa, mid_sa, p.n, apply_fwht64, lane, sg);
      }
      return;
    }

    // Full-64 path (valid_rows > 32): 32x32x32 i8xi8 dual shift-merge!
    #pragma clang loop unroll(full)
    for (uint hn = 0u; hn < kPasses; ++hn) {
      const uint w_col = tg_mem.tg_meta[3] + hn * 32u + (sg >> 1u) * 16u + (lane >> 1u);
      const bool w_live = (w_col < p.n);
      const ulong w_exp_row = ulong(tg_mem.tg_meta[0]) * ulong(p.n) + ulong(w_col);

      float acc_f32[32];
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 32u; ++i) acc_f32[i] = 0.0f;

      for (uint kb = 0u; kb < k_blocks; ++kb) {
        const uint blk128 = kb >> 1u;
        const uint half64 = kb & 1u;
        const uint tid = sg * 32u + lane;
        const uint r = tid >> 1u;
        const uint col32 = tid & 1u;
        const uint sub32 = half64 * 2u + col32;
        const uint w_row = sg * 16u + (lane >> 1u);

        const uint tok = tg_mem.tg_tok[r];
        const bool a_live = (tok != 0xFFFFFFFFu);
        uint4 raw_a = uint4(0u);
        uchar2 ea = uchar2(0x7Fu, 0x7Fu);
        if (a_live) {
          const ulong ai = ulong(tok >> 4u) * ulong(k_tiles128) + ulong(blk128);
          const ulong rm = ulong(tok & 15u);
          raw_a = reinterpret_cast<device const uint4 *>(a_codes + ai * 1024ul + rm * 64ul)[sub32];
          ea = *reinterpret_cast<device const uchar2 *>(
              a_scales + ai * 64ul + rm * 4ul + ulong(half64) * 2ul);
        }
        uint4 raw_w = uint4(0u);
        uchar2 ew = uchar2(0x7Fu, 0x7Fu);
        if (w_live) {
          const ulong c_off = w_exp_row * c_stride + ulong(kb * 2u + col32) * 16ul;
          const ulong s_off = w_exp_row * s_stride + ulong(kb) * 2ul;
          if (!sg_is_up) {
            raw_w = *reinterpret_cast<device const uint4 *>(g_codes + c_off);
            ew = *reinterpret_cast<device const uchar2 *>(g_scales + s_off);
          } else {
            raw_w = *reinterpret_cast<device const uint4 *>(u_codes + c_off);
            ew = *reinterpret_cast<device const uchar2 *>(u_scales + s_off);
          }
        }

        const int dea = (a_live && ea.x != 255u && ea.y != 255u)
            ? (int(ea.y) - int(ea.x)) : (a_live ? 255 : 0);
        const bool fa = (abs(dea) <= 3);
        const uint sha = fa
            ? ((col32 == 0u) ? ((dea < 0) ? uint(-dea) : 0u) : ((dea > 0) ? uint(dea) : 0u))
            : 0u;
        const uint sg_slow_a = simd_any(!fa) ? 1u : 0u;

        const int dew = (w_live && ew.x != 255u && ew.y != 255u)
            ? (int(ew.y) - int(ew.x)) : (w_live ? 255 : 0);
        const bool fw = (abs(dew) <= 3);
        const uint shw = fw
            ? ((col32 == 0u) ? ((dew < 0) ? uint(-dew) : 0u) : ((dew > 0) ? uint(dew) : 0u))
            : 0u;
        const uint sg_slow_w = simd_any(!fw) ? 1u : 0u;

        uint4 au0, au1, wu0, wu1;
        w4a4_moe_unpack_lut_sh_u4(raw_a, sha, mxfp4_lut, au0, au1);
        w4a4_moe_unpack_lut_sh_u4(raw_w, shw, mxfp4_lut, wu0, wu1);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + r * kW4A4MoeStride + col32 * 32u + 0u) = au0;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + r * kW4A4MoeStride + col32 * 32u + 16u) = au1;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A4MoeStride + col32 * 32u + 0u) = wu0;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A4MoeStride + col32 * 32u + 16u) = wu1;
        if (col32 == 0u) {
          if (fa) {
            const float sa = a_live ? (0.5f * w4a4_moe_e8m0_safe(min(ea.x, ea.y))) : 0.0f;
            tg_mem.kloop.Sa0_s[r] = sa;
            tg_mem.kloop.Sa1_s[r] = sa;
          } else {
            tg_mem.kloop.Sa0_s[r] = a_live ? (0.5f * w4a4_moe_e8m0_safe(ea.x)) : 0.0f;
            tg_mem.kloop.Sa1_s[r] = a_live ? (0.5f * w4a4_moe_e8m0_safe(ea.y)) : 0.0f;
          }
          if (fw) {
            const float sw = w_live ? (0.5f * w4a4_moe_e8m0_safe(min(ew.x, ew.y))) : 0.0f;
            tg_mem.kloop.Sw0_s[w_row] = sw;
            tg_mem.kloop.Sw1_s[w_row] = sw;
          } else {
            tg_mem.kloop.Sw0_s[w_row] = w_live ? (0.5f * w4a4_moe_e8m0_safe(ew.x)) : 0.0f;
            tg_mem.kloop.Sw1_s[w_row] = w_live ? (0.5f * w4a4_moe_e8m0_safe(ew.y)) : 0.0f;
          }
        }
        if (lane == 0u) {
          tg_mem.kloop.slow_a[sg] = sg_slow_a;
          tg_mem.kloop.slow_w[sg] = sg_slow_w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint m_off = sg_m * 32u;
        auto a0 = tA.slice(0, int(m_off));
        auto b0 = tB.slice(0, int(sg_n * 32u));
        auto ci = op32_m.template get_destination_cooperative_tensor<
            remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
        const uint sg_slow = tg_mem.kloop.slow_a[sg_m * 2u] | tg_mem.kloop.slow_a[sg_m * 2u + 1u] |
                             tg_mem.kloop.slow_w[sg_n * 2u] | tg_mem.kloop.slow_w[sg_n * 2u + 1u];
        if (sg_slow == 0u) {
          auto a1 = tA.slice(32, int(m_off));
          auto b1 = tB.slice(32, int(sg_n * 32u));
          op32_m.run(a0, b0, ci);
          op32_a.run(a1, b1, ci);
          const float sa0 = tg_mem.kloop.Sa0_s[m_off + fm];
          const float sa1 = tg_mem.kloop.Sa0_s[m_off + 8u + fm];
          const float sa2 = tg_mem.kloop.Sa0_s[m_off + 16u + fm];
          const float sa3 = tg_mem.kloop.Sa0_s[m_off + 24u + fm];
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            const float sw0 = tg_mem.kloop.Sw0_s[sw_base + c];
            const float sw1 = tg_mem.kloop.Sw0_s[sw_base + 16u + c];
            acc_f32[ 0u + c] = fma(sa0 * sw0, float(ci[ 0u + c]), acc_f32[ 0u + c]);
            acc_f32[ 4u + c] = fma(sa1 * sw0, float(ci[ 4u + c]), acc_f32[ 4u + c]);
            acc_f32[ 8u + c] = fma(sa0 * sw1, float(ci[ 8u + c]), acc_f32[ 8u + c]);
            acc_f32[12u + c] = fma(sa1 * sw1, float(ci[12u + c]), acc_f32[12u + c]);
            acc_f32[16u + c] = fma(sa2 * sw0, float(ci[16u + c]), acc_f32[16u + c]);
            acc_f32[20u + c] = fma(sa3 * sw0, float(ci[20u + c]), acc_f32[20u + c]);
            acc_f32[24u + c] = fma(sa2 * sw1, float(ci[24u + c]), acc_f32[24u + c]);
            acc_f32[28u + c] = fma(sa3 * sw1, float(ci[28u + c]), acc_f32[28u + c]);
          }
        } else {
          #pragma clang loop unroll(disable)
          for (uint hk = 0u; hk < 2u; ++hk) {
            auto ah = tA.slice(int(hk * 32u), int(m_off));
            auto bh = tB.slice(int(hk * 32u), int(sg_n * 32u));
            op32_m.run(ah, bh, ci);
            threadgroup const float *sap = (hk == 0u) ? tg_mem.kloop.Sa0_s : tg_mem.kloop.Sa1_s;
            threadgroup const float *swp = (hk == 0u) ? tg_mem.kloop.Sw0_s : tg_mem.kloop.Sw1_s;
            const float sa0 = sap[m_off + fm];
            const float sa1 = sap[m_off + 8u + fm];
            const float sa2 = sap[m_off + 16u + fm];
            const float sa3 = sap[m_off + 24u + fm];
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) {
              const float sw0 = swp[sw_base + c];
              const float sw1 = swp[sw_base + 16u + c];
              acc_f32[ 0u + c] = fma(sa0 * sw0, float(ci[ 0u + c]), acc_f32[ 0u + c]);
              acc_f32[ 4u + c] = fma(sa1 * sw0, float(ci[ 4u + c]), acc_f32[ 4u + c]);
              acc_f32[ 8u + c] = fma(sa0 * sw1, float(ci[ 8u + c]), acc_f32[ 8u + c]);
              acc_f32[12u + c] = fma(sa1 * sw1, float(ci[12u + c]), acc_f32[12u + c]);
              acc_f32[16u + c] = fma(sa2 * sw0, float(ci[16u + c]), acc_f32[16u + c]);
              acc_f32[20u + c] = fma(sa3 * sw0, float(ci[20u + c]), acc_f32[20u + c]);
              acc_f32[24u + c] = fma(sa2 * sw1, float(ci[24u + c]), acc_f32[24u + c]);
              acc_f32[28u + c] = fma(sa3 * sw1, float(ci[28u + c]), acc_f32[28u + c]);
            }
          }
        }
      }

      const uint rb_cur = tg_mem.tg_meta[1];
      const uint vr = tg_mem.tg_meta[2];
      const uint n0_base = tg_mem.tg_meta[3];
      const uint m_off = sg_m * 32u;
      const uint n0_sg = n0_base + hn * 32u + sg_n * 16u;

      if constexpr (!OutputA4) {
        (void)mid_qa; (void)mid_sa;
        const uint rb_8c = (vr + 7u) & ~7u;
        if (m_off < vr && n0_sg + fn < p.n) {
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            if (m_off + mm * 16u < vr) {
              #pragma clang loop unroll(full)
              for (uint h = 0u; h < 2u; ++h) {
                const uint rm = m_off + mm * 16u + fm + h * 8u;
                if (rm < vr) {
                  const uint e = mm * 16u + h * 4u;
                  bfloat4 out_b;
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    out_b[j] = bfloat(nax_silu_f32(float(bfloat(acc_f32[e + j]))) *
                                      float(bfloat(acc_f32[8u + e + j])));
                  }
                  *reinterpret_cast<device bfloat4 *>(
                      mid_bf16 + ulong(rb_cur + rm) * p.n + n0_sg + fn) = out_b;
                } else if (rm < rb_8c) {
                  *reinterpret_cast<device bfloat4 *>(
                      mid_bf16 + ulong(rb_cur + rm) * p.n + n0_sg + fn) = bfloat4(bfloat(0.0f));
                }
              }
            }
          }
        }
      } else {
        (void)mid_bf16;
        const ulong stash_col_off = ulong(n0_base) + ulong(sg_n * 16u + fn) * 2ul;
        if (hn == 0u) {
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint rm = m_off + mm * 16u + fm + h * 8u;
              if (rm < vr) {
                bfloat4 out_b = bfloat4(bfloat(0.0f));
                if (n0_sg + fn < p.n) {
                  const uint e = mm * 16u + h * 4u;
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    out_b[j] = (n0_sg + fn + j < p.n)
                        ? bfloat(nax_silu_f32(float(bfloat(acc_f32[e + j]))) *
                                 float(bfloat(acc_f32[8u + e + j])))
                        : bfloat(0.0f);
                  }
                }
                *reinterpret_cast<device bfloat4 *>(
                    mid_qa + ulong(rb_cur + rm) * p.n + stash_col_off) = out_b;
              }
            }
          }
        } else {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint rm = m_off + mm * 16u + fm + h * 8u;
              bfloat4 s0 = bfloat4(bfloat(0.0f));
              bfloat4 s1 = bfloat4(bfloat(0.0f));
              if (rm < vr) {
                s0 = *reinterpret_cast<device const bfloat4 *>(
                    mid_qa + ulong(rb_cur + rm) * p.n + stash_col_off);
                if (n0_sg + fn < p.n) {
                  const uint e = mm * 16u + h * 4u;
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    s1[j] = (n0_sg + fn + j < p.n)
                        ? bfloat(nax_silu_f32(float(bfloat(acc_f32[e + j]))) *
                                 float(bfloat(acc_f32[8u + e + j])))
                        : bfloat(0.0f);
                  }
                }
              }
              *reinterpret_cast<threadgroup bfloat4 *>(
                  tg_mem.SwiBuf + rm * kW4A4MoeEpiStride + sg_n * 16u + fn) = s0;
              *reinterpret_cast<threadgroup bfloat4 *>(
                  tg_mem.SwiBuf + rm * kW4A4MoeEpiStride + 32u + sg_n * 16u + fn) = s1;
            }
          }
        }
      }
    }

    if constexpr (OutputA4) {
      const uint rb_cur = tg_mem.tg_meta[1];
      const uint vr = tg_mem.tg_meta[2];
      const uint n0_base = tg_mem.tg_meta[3];
      const uint rb_8c = (vr + 7u) & ~7u;
      const bool apply_fwht64 = ((p.reserved >> 31u) == 0u);
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      w4a4_moe_swiglu_epilogue_a4_phase<16u>(
          tg_mem.SwiBuf, rb_cur, 0u, vr, rb_8c, n0_base,
          mid_qa, mid_sa, p.n, apply_fwht64, lane, sg);
    }
    return;
  }

  // Portable Apple9 fallback
  const uint tid = sg * 32u + lane;
  const bool is_up = ((sg & 1u) != 0u);
  const uint w_row = sg * 16u + (lane >> 1u);
  const uint w_col_in32 = (sg >> 1u) * 16u + (lane >> 1u);
  const uint r_st = tid >> 1u;
  const uint col32 = tid & 1u;
  const uint c_l = tid & 31u;
  const uint stripe4 = (tid >> 5u) * 4u;
  const uint ws_g_row = ((c_l >> 4u) * 32u) + (c_l & 15u);
  const uint ws_u_row = ws_g_row + 16u;
  const uint rb_8ceil = (valid_rows + 7u) & ~7u;

  for (uint r4 = 0u; r4 < 4u; ++r4) {
    bfloat4 s0 = bfloat4(bfloat(0.0f));
    bfloat4 s1 = bfloat4(bfloat(0.0f));
    for (uint hn = 0u; hn < kPasses; ++hn) {
      const uint tg_n0 = tg_n0_base + hn * 32u;
      const uint w_col = tg_n0 + w_col_in32;
      const uint col = tg_n0 + c_l;
      const bool col_live = (col < p.n);
      float4 g_acc = float4(0.0f);
      float4 u_acc = float4(0.0f);

      for (uint kb = 0u; kb < k_blocks; ++kb) {
        const uint blk128 = kb >> 1u;
        const uint half64 = kb & 1u;
        const uint sub32 = half64 * 2u + col32;
        const uint tok = tg_mem.tg_tok[r_st];

        uint4 raw_a = uint4(0u);
        float sa_v = 0.0f;
        if (tok != 0xFFFFFFFFu) {
          const ulong ai = ulong(tok >> 4u) * ulong(k_tiles128) + ulong(blk128);
          const ulong rm = ulong(tok & 15u);
          raw_a = reinterpret_cast<device const uint4 *>(a_codes + ai * 1024ul + rm * 64ul)[sub32];
          sa_v = 0.5f * w4a4_moe_e8m0_safe(a_scales[ai * 64ul + rm * 4ul + ulong(sub32)]);
        }
        uint4 raw_w = uint4(0u);
        float sw_v = 0.0f;
        if (w_col < p.n) {
          const ulong w_exp_row = ulong(expert) * ulong(p.n) + ulong(w_col);
          const ulong c_off = w_exp_row * c_stride + ulong(kb * 2u + col32) * 16ul;
          const ulong s_off = w_exp_row * s_stride + ulong(kb * 2u + col32);
          if (!is_up) {
            raw_w = *reinterpret_cast<device const uint4 *>(g_codes + c_off);
            sw_v = 0.5f * w4a4_moe_e8m0_safe(g_scales[s_off]);
          } else {
            raw_w = *reinterpret_cast<device const uint4 *>(u_codes + c_off);
            sw_v = 0.5f * w4a4_moe_e8m0_safe(u_scales[s_off]);
          }
        }
        uint4 au0, au1, wu0, wu1;
        w4a4_moe_unpack_lut_sh_u4(raw_a, 0u, mxfp4_lut, au0, au1);
        w4a4_moe_unpack_lut_sh_u4(raw_w, 0u, mxfp4_lut, wu0, wu1);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + r_st * kW4A4MoeStride + col32 * 32u + 0u) = au0;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + r_st * kW4A4MoeStride + col32 * 32u + 16u) = au1;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A4MoeStride + col32 * 32u + 0u) = wu0;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A4MoeStride + col32 * 32u + 16u) = wu1;
        if (col32 == 0u) { tg_mem.kloop.Sa0_s[r_st] = sa_v; tg_mem.kloop.Sw0_s[w_row] = sw_v; }
        else             { tg_mem.kloop.Sa1_s[r_st] = sa_v; tg_mem.kloop.Sw1_s[w_row] = sw_v; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (col_live && r4 * 16u + stripe4 < valid_rows) {
          const float swg0 = tg_mem.kloop.Sw0_s[ws_g_row], swg1 = tg_mem.kloop.Sw1_s[ws_g_row];
          const float swu0 = tg_mem.kloop.Sw0_s[ws_u_row], swu1 = tg_mem.kloop.Sw1_s[ws_u_row];
          const uint wg_off = ws_g_row * kW4A4MoeStride;
          const uint wu_off = ws_u_row * kW4A4MoeStride;
          for (uint rr = 0u; rr < 4u; ++rr) {
            const uint r_l = r4 * 16u + stripe4 + rr;
            if (r_l < valid_rows) {
              int32_t dg0 = 0, dg1 = 0, du0 = 0, du1 = 0;
              const uint a_off = r_l * kW4A4MoeStride;
              for (uint k = 0u; k < 32u; ++k) {
                const int32_t a0 = int32_t(tg_mem.kloop.As_i8[a_off + k]);
                const int32_t a1 = int32_t(tg_mem.kloop.As_i8[a_off + 32u + k]);
                dg0 += a0 * int32_t(tg_mem.kloop.Ws_i8[wg_off + k]);
                dg1 += a1 * int32_t(tg_mem.kloop.Ws_i8[wg_off + 32u + k]);
                du0 += a0 * int32_t(tg_mem.kloop.Ws_i8[wu_off + k]);
                du1 += a1 * int32_t(tg_mem.kloop.Ws_i8[wu_off + 32u + k]);
              }
              const float sa0 = tg_mem.kloop.Sa0_s[r_l], sa1 = tg_mem.kloop.Sa1_s[r_l];
              g_acc[rr] = fma(sa0 * swg0, float(dg0), fma(sa1 * swg1, float(dg1), g_acc[rr]));
              u_acc[rr] = fma(sa0 * swu0, float(du0), fma(sa1 * swu1, float(du1), u_acc[rr]));
            }
          }
        }
      }
      bfloat4 sv4 = bfloat4(bfloat(0.0f));
      for (uint rr = 0u; rr < 4u; ++rr) {
        const uint rm = r4 * 16u + stripe4 + rr;
        const bfloat sv = (col_live && rm < valid_rows)
            ? bfloat(nax_silu_f32(float(bfloat(g_acc[rr]))) * float(bfloat(u_acc[rr])))
            : bfloat(0.0f);
        sv4[rr] = sv;
        if constexpr (!OutputA4) {
          if (col_live && rm < rb_8ceil) {
            mid_bf16[ulong(rb + rm) * p.n + col] = sv;
          }
        }
      }
      if (hn == 0u) s0 = sv4; else s1 = sv4;
    }
    if constexpr (OutputA4) {
      const bool apply_fwht64 = ((p.reserved >> 31u) == 0u);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint rr = 0u; rr < 4u; ++rr) {
        tg_mem.SwiBuf[(stripe4 + rr) * kW4A4MoeEpiStride + c_l] = s0[rr];
        tg_mem.SwiBuf[(stripe4 + rr) * kW4A4MoeEpiStride + 32u + c_l] = s1[rr];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      w4a4_moe_swiglu_epilogue_a4_phase<4u>(
          tg_mem.SwiBuf, rb, r4 * 16u, valid_rows, rb_8ceil, tg_n0_base,
          mid_qa, mid_sa, p.n, apply_fwht64, lane, sg);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// ============================================================================
// MoE MXFP4 W4A4 Down Scatter (BM=64, BN=64, BK=64, 128t, i8xi8 dual shift-merge)
// Consumes row-major mid_qa (row stride = k bytes, 32B codes per 64B slot) + mid_sa [row, k/32].
// ============================================================================
inline void moe_mxfp4_w4a4_nax_down_scatter_bm64_bn64_impl(
    device const uchar *mid_qa,
    device const uchar *mid_sa,
    device const uint *tile_expert,
    device const uint *tile_valid_rows,
    device const uint *tile_assign,
    device const uint *total_tiles,
    device const bfloat *route_scores,
    device const uchar *d_codes,
    device const uchar *d_scales,
    device bfloat *expert_output,
    constant W4A4MoeGemmParams &p,
    constant ushort *mxfp4_lut,
    threadgroup W4A4MoeDownTgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BN_TG = 64u;

  const uint n_cols = (p.n + BN_TG - 1u) / BN_TG;
  const bool use_gx32 = (p.max_tiles >> 31u) != 0u;
  const uint m_tile = use_gx32 ? ((tg.y / n_cols) * 32u + tg.x) : tg.y;
  const uint col_tile = use_gx32 ? (tg.y % n_cols) : tg.x;
  const uint tg_n0 = col_tile * BN_TG;
  if (m_tile >= total_tiles[1] || col_tile >= n_cols || tg_n0 >= p.n) {
    return;
  }
  const uint valid_rows = tile_valid_rows[m_tile];
  if (valid_rows == 0u) {
    return;
  }
  const uint packed = tile_expert[m_tile];
  const uint expert = packed & 0xFFFFu;
  const uint rb = (packed >> 16u) * 8u;
  const uint tid = sg * 32u + lane;

  const uint k_blocks = p.k >> 6u;
  const ulong mid_c_stride = ulong(p.k);
  const ulong mid_s_stride = ulong(p.k >> 5u);
  const ulong w_c_stride = ulong(p.padded_k >> 1u);
  const ulong w_s_stride = ulong(p.padded_k >> 5u);

  if (w4a4_moe_has_apple10_layout()) {
    constexpr auto desc16_m = matmul2d_descriptor(
        16, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply);
    constexpr auto desc16_a = matmul2d_descriptor(
        16, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto desc32_m = matmul2d_descriptor(
        32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply);
    constexpr auto desc32_a = matmul2d_descriptor(
        32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc16_m, execution_simdgroup> op16_m;
    matmul2d<desc16_a, execution_simdgroup> op16_a;
    matmul2d<desc32_m, execution_simdgroup> op32_m;
    matmul2d<desc32_a, execution_simdgroup> op32_a;
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
        tg_mem.As_i8, dextents<int32_t, 2>(int(kW4A4MoeStride), 64));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
        tg_mem.Ws_i8, dextents<int32_t, 2>(int(kW4A4MoeStride), 64));

    const bool compact32 = (valid_rows <= 32u);
    const uint sg_m = sg >> 1u;
    const uint sg_n = sg & 1u;
    const uint m_off = sg_m * (compact32 ? 16u : 32u);
    const uint n0_sg = tg_n0 + sg_n * 32u;
    const bool sg_active = (m_off < valid_rows) && (n0_sg < p.n);

    const uint r = tid >> 1u;
    const uint col32 = tid & 1u;
    const uint w_col = tg_n0 + r;
    const bool w_live = (w_col < p.n);
    const ulong w_exp_row = ulong(expert) * ulong(p.n) + ulong(w_col);
    const uint qid = lane >> 2u;
    const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
    const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
    const uint sw_base = sg_n * 32u + fn;

    if (compact32) {
      float acc0[16];
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) acc0[i] = 0.0f;

      for (uint kb = 0u; kb < k_blocks; ++kb) {
        const bool a_live = (r < valid_rows);
        uint4 raw_a = uint4(0u);
        uchar2 ea = uchar2(0x7Fu, 0x7Fu);
        if (a_live) {
          const ulong row_idx = ulong(rb + r);
          raw_a = *reinterpret_cast<device const uint4 *>(
              mid_qa + row_idx * mid_c_stride + ulong(kb) * 64ul + ulong(col32) * 16ul);
          ea = *reinterpret_cast<device const uchar2 *>(
              mid_sa + row_idx * mid_s_stride + ulong(kb) * 2ul);
        }
        const uint4 raw_w = w_live
            ? *reinterpret_cast<device const uint4 *>(
                  d_codes + w_exp_row * w_c_stride + ulong(kb * 2u + col32) * 16ul)
            : uint4(0u);
        const uchar2 ew = w_live
            ? *reinterpret_cast<device const uchar2 *>(
                  d_scales + w_exp_row * w_s_stride + ulong(kb) * 2ul)
            : uchar2(0x7Fu, 0x7Fu);

        const int dea = (a_live && ea.x != 255u && ea.y != 255u)
            ? (int(ea.y) - int(ea.x)) : (a_live ? 255 : 0);
        const bool fa = (abs(dea) <= 3);
        const uint sha = fa
            ? ((col32 == 0u) ? ((dea < 0) ? uint(-dea) : 0u) : ((dea > 0) ? uint(dea) : 0u))
            : 0u;
        const uint sg_slow_a = simd_any(!fa) ? 1u : 0u;

        const int dew = (w_live && ew.x != 255u && ew.y != 255u)
            ? (int(ew.y) - int(ew.x)) : (w_live ? 255 : 0);
        const bool fw = (abs(dew) <= 3);
        const uint shw = fw
            ? ((col32 == 0u) ? ((dew < 0) ? uint(-dew) : 0u) : ((dew > 0) ? uint(dew) : 0u))
            : 0u;
        const uint sg_slow_w = simd_any(!fw) ? 1u : 0u;

        uint4 au0, au1, wu0, wu1;
        if (r < 32u) w4a4_moe_unpack_lut_sh_u4(raw_a, sha, mxfp4_lut, au0, au1);
        w4a4_moe_unpack_lut_sh_u4(raw_w, shw, mxfp4_lut, wu0, wu1);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (r < 32u) {
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + r * kW4A4MoeStride + col32 * 32u + 0u) = au0;
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + r * kW4A4MoeStride + col32 * 32u + 16u) = au1;
          if (col32 == 0u) {
            if (fa) {
              const float sa = a_live ? (0.5f * w4a4_moe_e8m0_safe(min(ea.x, ea.y))) : 0.0f;
              tg_mem.Sa0_s[r] = sa;
              tg_mem.Sa1_s[r] = sa;
            } else {
              tg_mem.Sa0_s[r] = a_live ? (0.5f * w4a4_moe_e8m0_safe(ea.x)) : 0.0f;
              tg_mem.Sa1_s[r] = a_live ? (0.5f * w4a4_moe_e8m0_safe(ea.y)) : 0.0f;
            }
          }
        }
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + r * kW4A4MoeStride + col32 * 32u + 0u) = wu0;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + r * kW4A4MoeStride + col32 * 32u + 16u) = wu1;
        if (col32 == 0u) {
          if (fw) {
            const float sw = w_live ? (0.5f * w4a4_moe_e8m0_safe(min(ew.x, ew.y))) : 0.0f;
            tg_mem.Sw0_s[r] = sw;
            tg_mem.Sw1_s[r] = sw;
          } else {
            tg_mem.Sw0_s[r] = w_live ? (0.5f * w4a4_moe_e8m0_safe(ew.x)) : 0.0f;
            tg_mem.Sw1_s[r] = w_live ? (0.5f * w4a4_moe_e8m0_safe(ew.y)) : 0.0f;
          }
        }
        if (lane == 0u) {
          tg_mem.slow_a[sg] = sg_slow_a;
          tg_mem.slow_w[sg] = sg_slow_w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
          auto a0 = tA.slice(0, int(m_off));
          auto b0 = tB.slice(0, int(sg_n * 32u));
          auto ci = op16_m.template get_destination_cooperative_tensor<
              remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
          const uint sg_slow = tg_mem.slow_a[sg_m] |
                               tg_mem.slow_w[sg_n * 2u] | tg_mem.slow_w[sg_n * 2u + 1u];
          if (sg_slow == 0u) {
            auto a1 = tA.slice(32, int(m_off));
            auto b1 = tB.slice(32, int(sg_n * 32u));
            op16_m.run(a0, b0, ci);
            op16_a.run(a1, b1, ci);
            const float sa0 = tg_mem.Sa0_s[m_off + fm];
            const float sa1 = tg_mem.Sa0_s[m_off + 8u + fm];
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) {
              const float sw0 = tg_mem.Sw0_s[sw_base + c];
              const float sw1 = tg_mem.Sw0_s[sw_base + 16u + c];
              acc0[ 0u + c] = fma(sa0 * sw0, float(ci[ 0u + c]), acc0[ 0u + c]);
              acc0[ 4u + c] = fma(sa1 * sw0, float(ci[ 4u + c]), acc0[ 4u + c]);
              acc0[ 8u + c] = fma(sa0 * sw1, float(ci[ 8u + c]), acc0[ 8u + c]);
              acc0[12u + c] = fma(sa1 * sw1, float(ci[12u + c]), acc0[12u + c]);
            }
          } else {
            #pragma clang loop unroll(disable)
            for (uint hk = 0u; hk < 2u; ++hk) {
              auto ah = tA.slice(int(hk * 32u), int(m_off));
              auto bh = tB.slice(int(hk * 32u), int(sg_n * 32u));
              op16_m.run(ah, bh, ci);
              threadgroup const float *sap = (hk == 0u) ? tg_mem.Sa0_s : tg_mem.Sa1_s;
              threadgroup const float *swp = (hk == 0u) ? tg_mem.Sw0_s : tg_mem.Sw1_s;
              const float sa0 = sap[m_off + fm];
              const float sa1 = sap[m_off + 8u + fm];
              #pragma clang loop unroll(full)
              for (uint c = 0u; c < 4u; ++c) {
                const float sw0 = swp[sw_base + c];
                const float sw1 = swp[sw_base + 16u + c];
                acc0[ 0u + c] = fma(sa0 * sw0, float(ci[ 0u + c]), acc0[ 0u + c]);
                acc0[ 4u + c] = fma(sa1 * sw0, float(ci[ 4u + c]), acc0[ 4u + c]);
                acc0[ 8u + c] = fma(sa0 * sw1, float(ci[ 8u + c]), acc0[ 8u + c]);
                acc0[12u + c] = fma(sa1 * sw1, float(ci[12u + c]), acc0[12u + c]);
              }
            }
          }
        }
      }
      if (sg_active) {
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint rm = m_off + fm + h * 8u;
          if (rm < valid_rows) {
            const uint assign = tile_assign[rb + rm];
            const float score = float(route_scores[assign]);
            #pragma clang loop unroll(full)
            for (uint u = 0u; u < 2u; ++u) {
              const uint col = n0_sg + u * 16u + fn;
              if (col < p.n) {
                const uint e = u * 8u + h * 4u;
                *reinterpret_cast<device bfloat4 *>(expert_output + ulong(assign) * p.n + col) =
                    bfloat4(bfloat(acc0[e + 0u] * score), bfloat(acc0[e + 1u] * score),
                            bfloat(acc0[e + 2u] * score), bfloat(acc0[e + 3u] * score));
              }
            }
          }
        }
      }
      return;
    }

    float acc_f32[32];
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 32u; ++i) acc_f32[i] = 0.0f;

    for (uint kb = 0u; kb < k_blocks; ++kb) {
      const bool a_live = (r < valid_rows);
      uint4 raw_a = uint4(0u);
      uchar2 ea = uchar2(0x7Fu, 0x7Fu);
      if (a_live) {
        const ulong row_idx = ulong(rb + r);
        raw_a = *reinterpret_cast<device const uint4 *>(
            mid_qa + row_idx * mid_c_stride + ulong(kb) * 64ul + ulong(col32) * 16ul);
        ea = *reinterpret_cast<device const uchar2 *>(
            mid_sa + row_idx * mid_s_stride + ulong(kb) * 2ul);
      }
      const uint4 raw_w = w_live
          ? *reinterpret_cast<device const uint4 *>(
                d_codes + w_exp_row * w_c_stride + ulong(kb * 2u + col32) * 16ul)
          : uint4(0u);
      const uchar2 ew = w_live
          ? *reinterpret_cast<device const uchar2 *>(
                d_scales + w_exp_row * w_s_stride + ulong(kb) * 2ul)
          : uchar2(0x7Fu, 0x7Fu);

      const int dea = (a_live && ea.x != 255u && ea.y != 255u)
          ? (int(ea.y) - int(ea.x)) : (a_live ? 255 : 0);
      const bool fa = (abs(dea) <= 3);
      const uint sha = fa
          ? ((col32 == 0u) ? ((dea < 0) ? uint(-dea) : 0u) : ((dea > 0) ? uint(dea) : 0u))
          : 0u;
      const uint sg_slow_a = simd_any(!fa) ? 1u : 0u;

      const int dew = (w_live && ew.x != 255u && ew.y != 255u)
          ? (int(ew.y) - int(ew.x)) : (w_live ? 255 : 0);
      const bool fw = (abs(dew) <= 3);
      const uint shw = fw
          ? ((col32 == 0u) ? ((dew < 0) ? uint(-dew) : 0u) : ((dew > 0) ? uint(dew) : 0u))
          : 0u;
      const uint sg_slow_w = simd_any(!fw) ? 1u : 0u;

      uint4 au0, au1, wu0, wu1;
      w4a4_moe_unpack_lut_sh_u4(raw_a, sha, mxfp4_lut, au0, au1);
      w4a4_moe_unpack_lut_sh_u4(raw_w, shw, mxfp4_lut, wu0, wu1);

      threadgroup_barrier(mem_flags::mem_threadgroup);
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + r * kW4A4MoeStride + col32 * 32u + 0u) = au0;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + r * kW4A4MoeStride + col32 * 32u + 16u) = au1;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + r * kW4A4MoeStride + col32 * 32u + 0u) = wu0;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + r * kW4A4MoeStride + col32 * 32u + 16u) = wu1;
      if (col32 == 0u) {
        if (fa) {
          const float sa = a_live ? (0.5f * w4a4_moe_e8m0_safe(min(ea.x, ea.y))) : 0.0f;
          tg_mem.Sa0_s[r] = sa;
          tg_mem.Sa1_s[r] = sa;
        } else {
          tg_mem.Sa0_s[r] = a_live ? (0.5f * w4a4_moe_e8m0_safe(ea.x)) : 0.0f;
          tg_mem.Sa1_s[r] = a_live ? (0.5f * w4a4_moe_e8m0_safe(ea.y)) : 0.0f;
        }
        if (fw) {
          const float sw = w_live ? (0.5f * w4a4_moe_e8m0_safe(min(ew.x, ew.y))) : 0.0f;
          tg_mem.Sw0_s[r] = sw;
          tg_mem.Sw1_s[r] = sw;
        } else {
          tg_mem.Sw0_s[r] = w_live ? (0.5f * w4a4_moe_e8m0_safe(ew.x)) : 0.0f;
          tg_mem.Sw1_s[r] = w_live ? (0.5f * w4a4_moe_e8m0_safe(ew.y)) : 0.0f;
        }
      }
      if (lane == 0u) {
        tg_mem.slow_a[sg] = sg_slow_a;
        tg_mem.slow_w[sg] = sg_slow_w;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_active) {
        auto a0 = tA.slice(0, int(m_off));
        auto b0 = tB.slice(0, int(sg_n * 32u));
        auto ci = op32_m.template get_destination_cooperative_tensor<
            remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
        const uint sg_slow = tg_mem.slow_a[sg_m * 2u] | tg_mem.slow_a[sg_m * 2u + 1u] |
                             tg_mem.slow_w[sg_n * 2u] | tg_mem.slow_w[sg_n * 2u + 1u];
        if (sg_slow == 0u) {
          auto a1 = tA.slice(32, int(m_off));
          auto b1 = tB.slice(32, int(sg_n * 32u));
          op32_m.run(a0, b0, ci);
          op32_a.run(a1, b1, ci);
          const float sa0 = tg_mem.Sa0_s[m_off + fm];
          const float sa1 = tg_mem.Sa0_s[m_off + 8u + fm];
          const float sa2 = tg_mem.Sa0_s[m_off + 16u + fm];
          const float sa3 = tg_mem.Sa0_s[m_off + 24u + fm];
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            const float sw0 = tg_mem.Sw0_s[sw_base + c];
            const float sw1 = tg_mem.Sw0_s[sw_base + 16u + c];
            acc_f32[ 0u + c] = fma(sa0 * sw0, float(ci[ 0u + c]), acc_f32[ 0u + c]);
            acc_f32[ 4u + c] = fma(sa1 * sw0, float(ci[ 4u + c]), acc_f32[ 4u + c]);
            acc_f32[ 8u + c] = fma(sa0 * sw1, float(ci[ 8u + c]), acc_f32[ 8u + c]);
            acc_f32[12u + c] = fma(sa1 * sw1, float(ci[12u + c]), acc_f32[12u + c]);
            acc_f32[16u + c] = fma(sa2 * sw0, float(ci[16u + c]), acc_f32[16u + c]);
            acc_f32[20u + c] = fma(sa3 * sw0, float(ci[20u + c]), acc_f32[20u + c]);
            acc_f32[24u + c] = fma(sa2 * sw1, float(ci[24u + c]), acc_f32[24u + c]);
            acc_f32[28u + c] = fma(sa3 * sw1, float(ci[28u + c]), acc_f32[28u + c]);
          }
        } else {
          #pragma clang loop unroll(disable)
          for (uint hk = 0u; hk < 2u; ++hk) {
            auto ah = tA.slice(int(hk * 32u), int(m_off));
            auto bh = tB.slice(int(hk * 32u), int(sg_n * 32u));
            op32_m.run(ah, bh, ci);
            threadgroup const float *sap = (hk == 0u) ? tg_mem.Sa0_s : tg_mem.Sa1_s;
            threadgroup const float *swp = (hk == 0u) ? tg_mem.Sw0_s : tg_mem.Sw1_s;
            const float sa0 = sap[m_off + fm];
            const float sa1 = sap[m_off + 8u + fm];
            const float sa2 = sap[m_off + 16u + fm];
            const float sa3 = sap[m_off + 24u + fm];
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) {
              const float sw0 = swp[sw_base + c];
              const float sw1 = swp[sw_base + 16u + c];
              acc_f32[ 0u + c] = fma(sa0 * sw0, float(ci[ 0u + c]), acc_f32[ 0u + c]);
              acc_f32[ 4u + c] = fma(sa1 * sw0, float(ci[ 4u + c]), acc_f32[ 4u + c]);
              acc_f32[ 8u + c] = fma(sa0 * sw1, float(ci[ 8u + c]), acc_f32[ 8u + c]);
              acc_f32[12u + c] = fma(sa1 * sw1, float(ci[12u + c]), acc_f32[12u + c]);
              acc_f32[16u + c] = fma(sa2 * sw0, float(ci[16u + c]), acc_f32[16u + c]);
              acc_f32[20u + c] = fma(sa3 * sw0, float(ci[20u + c]), acc_f32[20u + c]);
              acc_f32[24u + c] = fma(sa2 * sw1, float(ci[24u + c]), acc_f32[24u + c]);
              acc_f32[28u + c] = fma(sa3 * sw1, float(ci[28u + c]), acc_f32[28u + c]);
            }
          }
        }
      }
    }

    if (sg_active) {
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        if (m_off + mm * 16u < valid_rows) {
          #pragma clang loop unroll(full)
          for (uint h = 0u; h < 2u; ++h) {
            const uint rm = m_off + mm * 16u + fm + h * 8u;
            if (rm < valid_rows) {
              const uint assign = tile_assign[rb + rm];
              const float score = float(route_scores[assign]);
              #pragma clang loop unroll(full)
              for (uint u = 0u; u < 2u; ++u) {
                const uint col = n0_sg + u * 16u + fn;
                if (col < p.n) {
                  const uint e = mm * 16u + u * 8u + h * 4u;
                  *reinterpret_cast<device bfloat4 *>(expert_output + ulong(assign) * p.n + col) =
                      bfloat4(bfloat(acc_f32[e + 0u] * score), bfloat(acc_f32[e + 1u] * score),
                              bfloat(acc_f32[e + 2u] * score), bfloat(acc_f32[e + 3u] * score));
                }
              }
            }
          }
        }
      }
    }
    return;
  }

  // Portable Apple9 fallback
  const uint c_l = tid & 63u;
  const uint r_base = (tid >> 6u) * 32u;
  const uint col = tg_n0 + c_l;
  const bool col_live = (col < p.n);
  const uint r_st = tid >> 1u;
  const uint col32 = tid & 1u;

  for (uint r4 = 0u; r4 < 8u; ++r4) {
    float4 acc_a9 = float4(0.0f);
    for (uint kb = 0u; kb < k_blocks; ++kb) {
      const uint w_col = tg_n0 + r_st;
      uint4 raw_a = uint4(0u);
      float sa_v = 0.0f;
      if (r_st < valid_rows) {
        const ulong row_idx = ulong(rb + r_st);
        raw_a = *reinterpret_cast<device const uint4 *>(
            mid_qa + row_idx * mid_c_stride + ulong(kb) * 64ul + ulong(col32) * 16ul);
        sa_v = 0.5f * w4a4_moe_e8m0_safe(mid_sa[row_idx * mid_s_stride + ulong(kb * 2u + col32)]);
      }
      uint4 raw_w = uint4(0u);
      float sw_v = 0.0f;
      if (w_col < p.n) {
        const ulong w_exp_row = ulong(expert) * ulong(p.n) + ulong(w_col);
        raw_w = *reinterpret_cast<device const uint4 *>(
            d_codes + w_exp_row * w_c_stride + ulong(kb * 2u + col32) * 16ul);
        sw_v = 0.5f * w4a4_moe_e8m0_safe(d_scales[w_exp_row * w_s_stride + ulong(kb * 2u + col32)]);
      }
      uint4 au0, au1, wu0, wu1;
      w4a4_moe_unpack_lut_sh_u4(raw_a, 0u, mxfp4_lut, au0, au1);
      w4a4_moe_unpack_lut_sh_u4(raw_w, 0u, mxfp4_lut, wu0, wu1);

      threadgroup_barrier(mem_flags::mem_threadgroup);
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + r_st * kW4A4MoeStride + col32 * 32u + 0u) = au0;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + r_st * kW4A4MoeStride + col32 * 32u + 16u) = au1;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + r_st * kW4A4MoeStride + col32 * 32u + 0u) = wu0;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + r_st * kW4A4MoeStride + col32 * 32u + 16u) = wu1;
      if (col32 == 0u) { tg_mem.Sa0_s[r_st] = sa_v; tg_mem.Sw0_s[r_st] = sw_v; }
      else             { tg_mem.Sa1_s[r_st] = sa_v; tg_mem.Sw1_s[r_st] = sw_v; }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (col_live && r_base + r4 * 4u < valid_rows) {
        const float sw0 = tg_mem.Sw0_s[c_l], sw1 = tg_mem.Sw1_s[c_l];
        const uint w_off = c_l * kW4A4MoeStride;
        for (uint rr = 0u; rr < 4u; ++rr) {
          const uint r_l = r_base + r4 * 4u + rr;
          if (r_l < valid_rows) {
            int32_t d0 = 0, d1 = 0;
            const uint a_off = r_l * kW4A4MoeStride;
            for (uint k = 0u; k < 32u; ++k) {
              d0 += int32_t(tg_mem.As_i8[a_off + k]) * int32_t(tg_mem.Ws_i8[w_off + k]);
              d1 += int32_t(tg_mem.As_i8[a_off + 32u + k]) * int32_t(tg_mem.Ws_i8[w_off + 32u + k]);
            }
            acc_a9[rr] = fma(tg_mem.Sa0_s[r_l] * sw0, float(d0),
                             fma(tg_mem.Sa1_s[r_l] * sw1, float(d1), acc_a9[rr]));
          }
        }
      }
    }
    if (col_live) {
      for (uint rr = 0u; rr < 4u; ++rr) {
        const uint rm = r_base + r4 * 4u + rr;
        if (rm < valid_rows) {
          const uint assign = tile_assign[rb + rm];
          const float score = float(route_scores[assign]);
          expert_output[ulong(assign) * p.n + col] = bfloat(acc_a9[rr] * score);
        }
      }
    }
  }
}

} // namespace

// ============================================================================
// MoE MXFP4 W4A4 Kernel Entry Points (<= 9,520 B TG SRAM, 0 B Spills)
// ============================================================================

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_moe_mxfp4_w4a4_nax_swiglu_bm64_out64_sk1(
    device const uchar *a_codes [[buffer(0)]],
    device const uchar *a_scales [[buffer(1)]],
    device const uint *tile_assign [[buffer(2)]],
    device const uint *tile_expert [[buffer(3)]],
    device const uint *tile_valid_rows [[buffer(4)]],
    device const uint *total_tiles [[buffer(5)]],
    device const uchar *g_codes [[buffer(6)]],
    device const uchar *g_scales [[buffer(7)]],
    device const uchar *u_codes [[buffer(8)]],
    device const uchar *u_scales [[buffer(9)]],
    device bfloat *mid_bf16 [[buffer(10)]],
    constant W4A4MoeGemmParams &p [[buffer(11)]],
    constant ushort *mxfp4_lut [[buffer(12)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup W4A4MoeSwiTgStorage tg_mem;
  moe_mxfp4_w4a4_nax_swiglu_bm64_out64_impl<false>(
      a_codes, a_scales, tile_assign, tile_expert, tile_valid_rows, total_tiles,
      g_codes, g_scales, u_codes, u_scales, mid_bf16, nullptr, nullptr,
      p, mxfp4_lut, tg_mem, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_moe_mxfp4_w4a4_nax_swiglu_bm64_out64_sk1_fwht64_a4(
    device const uchar *a_codes [[buffer(0)]],
    device const uchar *a_scales [[buffer(1)]],
    device const uint *tile_assign [[buffer(2)]],
    device const uint *tile_expert [[buffer(3)]],
    device const uint *tile_valid_rows [[buffer(4)]],
    device const uint *total_tiles [[buffer(5)]],
    device const uchar *g_codes [[buffer(6)]],
    device const uchar *g_scales [[buffer(7)]],
    device const uchar *u_codes [[buffer(8)]],
    device const uchar *u_scales [[buffer(9)]],
    device uchar *mid_qa [[buffer(10)]],
    device uchar *mid_sa [[buffer(11)]],
    constant W4A4MoeGemmParams &p [[buffer(12)]],
    constant ushort *mxfp4_lut [[buffer(13)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup W4A4MoeSwiTgStorage tg_mem;
  moe_mxfp4_w4a4_nax_swiglu_bm64_out64_impl<true>(
      a_codes, a_scales, tile_assign, tile_expert, tile_valid_rows, total_tiles,
      g_codes, g_scales, u_codes, u_scales, nullptr, mid_qa, mid_sa,
      p, mxfp4_lut, tg_mem, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_moe_mxfp4_w4a4_nax_down_scatter_bm64_bn64_sk1(
    device const uchar *mid_qa [[buffer(0)]],
    device const uchar *mid_sa [[buffer(1)]],
    device const uint *tile_expert [[buffer(2)]],
    device const uint *tile_valid_rows [[buffer(3)]],
    device const uint *tile_assign [[buffer(4)]],
    device const uint *total_tiles [[buffer(5)]],
    device const bfloat *route_scores [[buffer(6)]],
    device const uchar *d_codes [[buffer(7)]],
    device const uchar *d_scales [[buffer(8)]],
    device bfloat *expert_output [[buffer(9)]],
    constant W4A4MoeGemmParams &p [[buffer(10)]],
    constant ushort *mxfp4_lut [[buffer(11)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup W4A4MoeDownTgStorage tg_mem;
  moe_mxfp4_w4a4_nax_down_scatter_bm64_bn64_impl(
      mid_qa, mid_sa, tile_expert, tile_valid_rows, tile_assign, total_tiles,
      route_scores, d_codes, d_scales, expert_output, p, mxfp4_lut, tg_mem, lane, sg, tg);
}
