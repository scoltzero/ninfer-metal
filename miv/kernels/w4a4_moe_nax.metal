// Phase 3 WP 3A: Hardware-Native MoE MXFP4 W4A4 Prefill Kernels (Zero .miv v2 Repack)
// (Fused Gate + Up + SwiGLU [Mode A BF16 / Mode B FWHT-64 + A4] & W4A4 Down Scatter)
// Metal 4.1 NAX TensorOps (`matmul2d_descriptor(16, 32, 32)` on contiguous K=32 planes).
// - All TG SRAM strictly <= 10,240 B (3 TGs/core on Apple10 32 KiB/core SRAM), 0 B spills:
//   * MoE MXFP4 W4A4 SwiGLU (Mode A & Mode B, 128t, 2x K=32 + on-chip Swi0): 8,720 B (8.52 KiB)
//   * MoE MXFP4 W4A4 Down Scatter (128t, 4x K=32 planes):                    8,720 B (8.52 KiB)
// - Consumes 3D untiled expert weights (`mxfp4_expert_rowsplit_k128_v1`),
//   tiled input `(a4_codes, a4_scales)` from RMSNorm+A4, and reuses existing
//   `MoeNaxWorkspace::mid_qa` (row-major `[row, f/2]` FP4 codes) & `mid_sa` (`[row, f/32]` UE8M0).

#include "common_math.h"
#include "fwht64.h"
#include <metal_tensor>
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

using TgUe8 = tensor_blockwise<tensor_plane_scales, threadgroup metal_fp8_ue8m0_format, 32, 1>;
using TgFp4 = tensor<threadgroup metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline, TgUe8>;

struct W4A4MoeSwiTgStorage {
  union {
    struct {
      alignas(16) uchar As_c[2u][64u * 16u]; // 2,048 B
      alignas(16) uchar Ws_c[2u][64u * 16u]; // 2,048 B
      alignas(16) uchar As_s[2u][64u];       //   128 B
      alignas(16) uchar Ws_s[2u][64u];       //   128 B
    } kloop;                                 // 4,352 B
    alignas(16) bfloat Swi1[64u * 32u];      // 4,096 B
  };
  alignas(16) bfloat Swi0[64u * 32u];        // 4,096 B
  uint tg_tok[64u];                          //   256 B
  uint tg_meta[4u];                          //    16 B {expert, rb, valid_rows, tg_n0_base}
};
static_assert(sizeof(W4A4MoeSwiTgStorage) == 8720ul,
              "W4A4MoeSwiTgStorage must be 8,720 B (8.52 KiB)");
static_assert(sizeof(W4A4MoeSwiTgStorage) <= kW4A4MoeHighOccSramBytes,
              "W4A4MoeSwiTgStorage exceeds 10.0 KiB");

struct W4A4MoeDownTgStorage {
  alignas(16) uchar As_c[4u][64u * 16u]; // 4,096 B
  alignas(16) uchar Ws_c[4u][64u * 16u]; // 4,096 B
  alignas(16) uchar As_s[4u][64u];       //   256 B
  alignas(16) uchar Ws_s[4u][64u];       //   256 B
  uint tg_meta[4u];                      //    16 B {expert, rb, valid_rows, tg_n0}
};
static_assert(sizeof(W4A4MoeDownTgStorage) == 8720ul,
              "W4A4MoeDownTgStorage must be 8,720 B (8.52 KiB)");
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

inline float w4a4_moe_dot32_fp4_tg(
    const threadgroup uchar *a16,
    uchar sa_code,
    const threadgroup uchar *w16,
    uchar sw_code) {
  const float sa = w4a4_moe_e8m0_safe(sa_code);
  const float sw = w4a4_moe_e8m0_safe(sw_code);
  float acc = 0.0f;
  #pragma clang loop unroll(full)
  for (uint bi = 0u; bi < 16u; ++bi) {
    const uint ab = uint(a16[bi]);
    const uint wb = uint(w16[bi]);
    acc = fma(fp4_nibble_to_float(ab & 0xFu), fp4_nibble_to_float(wb & 0xFu), acc);
    acc = fma(fp4_nibble_to_float(ab >> 4u),  fp4_nibble_to_float(wb >> 4u),  acc);
  }
  return acc * (sa * sw);
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

template <uint RowsPerSg>
inline void w4a4_moe_swiglu_epilogue_a4_phase(
    const threadgroup bfloat *Swi0,
    const threadgroup bfloat *Swi1,
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
  const bool in_hi_half = (lane >= 16u);
  const uint sub_lane = lane & 15u;
  const ulong row_c_stride = ulong(f_dim >> 1u);
  const ulong row_s_stride = ulong(f_dim >> 5u);
  const ulong col_c_off = ulong(tg_n0_base >> 1u);
  const ulong col_s_off = ulong(tg_n0_base >> 5u);

  #pragma clang loop unroll(disable)
  for (uint rr = 0u; rr < RowsPerSg; ++rr) {
    const uint r_l = base_r + rr;
    const uint rm = base_row_offset + r_l;
    if (rm < valid_rows) {
      const bfloat2 s2 = in_hi_half
          ? reinterpret_cast<const threadgroup bfloat2 *>(Swi1 + r_l * 32u)[sub_lane]
          : reinterpret_cast<const threadgroup bfloat2 *>(Swi0 + r_l * 32u)[sub_lane];
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
// MoE Native MXFP4 W4A4 SwiGLU (BM=64, OUT_N=32 Mode A / 64 Mode B, 128 threads)
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
  const uint k_blocks64 = p.k >> 6u;
  const ulong c_stride = ulong(p.padded_k >> 1u);
  const ulong s_stride = ulong(p.padded_k >> 5u);

  if (w4a4_moe_has_apple10_layout()) {
    constexpr auto desc = matmul2d_descriptor(
        16, 32, 32, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    TgFp4 ta_init(tg_mem.kloop.As_c[0], dextents<int32_t, 2>(32, 16), TgUe8(tg_mem.kloop.As_s[0]));
    TgFp4 tb_init(tg_mem.kloop.Ws_c[0], dextents<int32_t, 2>(32, 32), TgUe8(tg_mem.kloop.Ws_s[0]));

    const bool sg_is_up = ((sg & 1u) != 0u);
    const uint tid = sg * 32u + lane;
    const uint sg_n = sg & 1u;
    const bool compact32 = (tg_mem.tg_meta[2] <= 32u);
    const uint m_off = (sg >> 1u) * (compact32 ? 16u : 32u);

    #pragma clang loop unroll(full)
    for (uint hn = 0u; hn < kPasses; ++hn) {
      auto c0 = op.template get_destination_cooperative_tensor<TgFp4, TgFp4, float>();
      auto c1 = op.template get_destination_cooperative_tensor<TgFp4, TgFp4, float>();
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) {
        c0[i] = 0.0f;
        c1[i] = 0.0f;
      }

      for (uint k64 = 0u; k64 < k_blocks64; ++k64) {
        const uint blk128 = k64 >> 1u;
        const uint half64 = k64 & 1u;
        const uint r = tid >> 1u;
        const uint h = tid & 1u;
        const uint tok = (r < BM_TG) ? tg_mem.tg_tok[r] : 0xFFFFFFFFu;
        const uint w_row = sg * 16u + (lane >> 1u);
        const uint w_col = tg_mem.tg_meta[3] + hn * 32u + (sg >> 1u) * 16u + (lane >> 1u);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tok != 0xFFFFFFFFu) {
          const ulong ai = ulong(tok >> 4u) * ulong(k_tiles128) + ulong(blk128);
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_c[h] + r * 16u) =
              reinterpret_cast<device const uint4 *>(
                  a_codes + ai * 1024ul + ulong(tok & 15u) * 64ul + ulong(half64) * 32ul)[h];
          tg_mem.kloop.As_s[h][r] =
              a_scales[ai * 64ul + ulong(tok & 15u) * 4ul + ulong(half64) * 2ul + ulong(h)];
        } else {
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_c[h] + r * 16u) = uint4(0u);
          tg_mem.kloop.As_s[h][r] = 0x7Fu;
        }
        if (w_col < p.n) {
          const ulong w_exp_row = ulong(tg_mem.tg_meta[0]) * ulong(p.n) + ulong(w_col);
          const ulong c_off = w_exp_row * c_stride + ulong(k64) * 32ul;
          const ulong s_off = w_exp_row * s_stride + ulong(k64) * 2ul + ulong(h);
          if (!sg_is_up) {
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_c[h] + w_row * 16u) =
                reinterpret_cast<device const uint4 *>(g_codes + c_off)[h];
            tg_mem.kloop.Ws_s[h][w_row] = g_scales[s_off];
          } else {
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_c[h] + w_row * 16u) =
                reinterpret_cast<device const uint4 *>(u_codes + c_off)[h];
            tg_mem.kloop.Ws_s[h][w_row] = u_scales[s_off];
          }
        } else {
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_c[h] + w_row * 16u) = uint4(0u);
          tg_mem.kloop.Ws_s[h][w_row] = 0x7Fu;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(disable)
        for (uint pl = 0u; pl < 2u; ++pl) {
          TgFp4 tA0(tg_mem.kloop.As_c[pl] + m_off * 16u,
                    dextents<int32_t, 2>(32, 16), TgUe8(tg_mem.kloop.As_s[pl] + m_off));
          TgFp4 tB(tg_mem.kloop.Ws_c[pl] + sg_n * 32u * 16u,
                   dextents<int32_t, 2>(32, 32), TgUe8(tg_mem.kloop.Ws_s[pl] + sg_n * 32u));
          op.run(tA0, tB, c0);
          if (!compact32) {
            TgFp4 tA1(tg_mem.kloop.As_c[pl] + (m_off + 16u) * 16u,
                      dextents<int32_t, 2>(32, 16), TgUe8(tg_mem.kloop.As_s[pl] + m_off + 16u));
            op.run(tA1, tB, c1);
          }
        }
      }

      const uint qid = lane >> 2u;
      const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
      const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
      const uint rb_cur = tg_mem.tg_meta[1];
      const uint vr = tg_mem.tg_meta[2];
      const uint n0_base = tg_mem.tg_meta[3];
      const uint n0_sg = n0_base + hn * 32u + sg_n * 16u;

      if constexpr (!OutputA4) {
        (void)mid_qa; (void)mid_sa;
        const uint rb_8c = (vr + 7u) & ~7u;
        if (m_off < vr && n0_sg + fn < p.n) {
          #pragma clang loop unroll(full)
          for (uint h2 = 0u; h2 < 2u; ++h2) {
            const uint rm0 = m_off + fm + h2 * 8u;
            const uint e = h2 * 4u;
            if (rm0 < vr) {
              bfloat4 ob0;
              #pragma clang loop unroll(full)
              for (uint j = 0u; j < 4u; ++j) {
                ob0[j] = bfloat(nax_silu_f32(float(bfloat(c0[e + j]))) *
                                float(bfloat(c0[8u + e + j])));
              }
              *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb_cur + rm0) * p.n + n0_sg + fn) = ob0;
            } else if (rm0 < rb_8c) {
              *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb_cur + rm0) * p.n + n0_sg + fn) = bfloat4(bfloat(0.0f));
            }
            if (!compact32 && (m_off + 16u < vr)) {
              const uint rm1 = rm0 + 16u;
              if (rm1 < vr) {
                bfloat4 ob1;
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  ob1[j] = bfloat(nax_silu_f32(float(bfloat(c1[e + j]))) *
                                  float(bfloat(c1[8u + e + j])));
                }
                *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb_cur + rm1) * p.n + n0_sg + fn) = ob1;
              } else if (rm1 < rb_8c) {
                *reinterpret_cast<device bfloat4 *>(mid_bf16 + ulong(rb_cur + rm1) * p.n + n0_sg + fn) = bfloat4(bfloat(0.0f));
              }
            }
          }
        }
      } else {
        (void)mid_bf16;
        if (hn == 1u) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup bfloat *dst_swi = (hn == 0u) ? tg_mem.Swi0 : tg_mem.Swi1;
        #pragma clang loop unroll(full)
        for (uint h2 = 0u; h2 < 2u; ++h2) {
          const uint rm0 = m_off + fm + h2 * 8u;
          const uint e = h2 * 4u;
          bfloat4 ob0 = bfloat4(bfloat(0.0f));
          if (rm0 < vr && n0_sg + fn < p.n) {
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) {
              ob0[j] = (n0_sg + fn + j < p.n)
                  ? bfloat(nax_silu_f32(float(bfloat(c0[e + j]))) *
                           float(bfloat(c0[8u + e + j])))
                  : bfloat(0.0f);
            }
          }
          *reinterpret_cast<threadgroup bfloat4 *>(dst_swi + rm0 * 32u + sg_n * 16u + fn) = ob0;
          if (!compact32) {
            const uint rm1 = rm0 + 16u;
            bfloat4 ob1 = bfloat4(bfloat(0.0f));
            if (rm1 < vr && n0_sg + fn < p.n) {
              #pragma clang loop unroll(full)
              for (uint j = 0u; j < 4u; ++j) {
                ob1[j] = (n0_sg + fn + j < p.n)
                    ? bfloat(nax_silu_f32(float(bfloat(c1[e + j]))) *
                             float(bfloat(c1[8u + e + j])))
                    : bfloat(0.0f);
              }
            }
            *reinterpret_cast<threadgroup bfloat4 *>(dst_swi + rm1 * 32u + sg_n * 16u + fn) = ob1;
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
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (compact32) {
        w4a4_moe_swiglu_epilogue_a4_phase<8u>(
            tg_mem.Swi0, tg_mem.Swi1, rb_cur, 0u, vr, rb_8c, n0_base,
            mid_qa, mid_sa, p.n, apply_fwht64, lane, sg);
      } else {
        w4a4_moe_swiglu_epilogue_a4_phase<16u>(
            tg_mem.Swi0, tg_mem.Swi1, rb_cur, 0u, vr, rb_8c, n0_base,
            mid_qa, mid_sa, p.n, apply_fwht64, lane, sg);
      }
    }
    return;
  }

  // Portable Apple9 fallback
  const uint tid = sg * 32u + lane;
  const bool is_up = ((sg & 1u) != 0u);
  const uint w_row = sg * 16u + (lane >> 1u);
  const uint w_col_in32 = (sg >> 1u) * 16u + (lane >> 1u);
  const uint r_st = tid >> 1u;
  const uint h_st = tid & 1u;
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

      for (uint k64 = 0u; k64 < k_blocks64; ++k64) {
        const uint blk128 = k64 >> 1u;
        const uint half64 = k64 & 1u;
        const uint tok = tg_mem.tg_tok[r_st];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tok != 0xFFFFFFFFu) {
          const ulong ai = ulong(tok >> 4u) * ulong(k_tiles128) + ulong(blk128);
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_c[h_st] + r_st * 16u) =
              reinterpret_cast<device const uint4 *>(
                  a_codes + ai * 1024ul + ulong(tok & 15u) * 64ul + ulong(half64) * 32ul)[h_st];
          tg_mem.kloop.As_s[h_st][r_st] =
              a_scales[ai * 64ul + ulong(tok & 15u) * 4ul + ulong(half64) * 2ul + ulong(h_st)];
        } else {
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_c[h_st] + r_st * 16u) = uint4(0u);
          tg_mem.kloop.As_s[h_st][r_st] = 0x7Fu;
        }
        if (w_col < p.n) {
          const ulong w_exp_row = ulong(expert) * ulong(p.n) + ulong(w_col);
          const ulong c_off = w_exp_row * c_stride + ulong(k64) * 32ul;
          const ulong s_off = w_exp_row * s_stride + ulong(k64) * 2ul + ulong(h_st);
          if (!is_up) {
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_c[h_st] + w_row * 16u) =
                reinterpret_cast<device const uint4 *>(g_codes + c_off)[h_st];
            tg_mem.kloop.Ws_s[h_st][w_row] = g_scales[s_off];
          } else {
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_c[h_st] + w_row * 16u) =
                reinterpret_cast<device const uint4 *>(u_codes + c_off)[h_st];
            tg_mem.kloop.Ws_s[h_st][w_row] = u_scales[s_off];
          }
        } else {
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_c[h_st] + w_row * 16u) = uint4(0u);
          tg_mem.kloop.Ws_s[h_st][w_row] = 0x7Fu;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (col_live && r4 * 16u + stripe4 < valid_rows) {
          for (uint rr = 0u; rr < 4u; ++rr) {
            const uint r_l = r4 * 16u + stripe4 + rr;
            if (r_l < valid_rows) {
              for (uint pl = 0u; pl < 2u; ++pl) {
                g_acc[rr] += w4a4_moe_dot32_fp4_tg(
                    tg_mem.kloop.As_c[pl] + r_l * 16u, tg_mem.kloop.As_s[pl][r_l],
                    tg_mem.kloop.Ws_c[pl] + ws_g_row * 16u, tg_mem.kloop.Ws_s[pl][ws_g_row]);
                u_acc[rr] += w4a4_moe_dot32_fp4_tg(
                    tg_mem.kloop.As_c[pl] + r_l * 16u, tg_mem.kloop.As_s[pl][r_l],
                    tg_mem.kloop.Ws_c[pl] + ws_u_row * 16u, tg_mem.kloop.Ws_s[pl][ws_u_row]);
              }
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
        tg_mem.Swi0[(stripe4 + rr) * 32u + c_l] = s0[rr];
        tg_mem.Swi0[(16u + stripe4 + rr) * 32u + c_l] = s1[rr];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      w4a4_moe_swiglu_epilogue_a4_phase<4u>(
          tg_mem.Swi0, tg_mem.Swi0 + 16u * 32u, rb, r4 * 16u, valid_rows, rb_8ceil, tg_n0_base,
          mid_qa, mid_sa, p.n, apply_fwht64, lane, sg);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// ============================================================================
// MoE Native MXFP4 W4A4 Down Scatter (BM=64, BN=64, BK=128 as 4x32 planes, 128t)
// Consumes row-major mid_qa [row, k/2] + mid_sa [row, k/32] -> weighted scatter.
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

  if (tid == 0u) {
    tg_mem.tg_meta[0] = expert;
    tg_mem.tg_meta[1] = rb;
    tg_mem.tg_meta[2] = valid_rows;
    tg_mem.tg_meta[3] = tg_n0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint k_tiles128 = p.k >> 7u;
  const ulong mid_c_stride = ulong(p.k >> 1u);
  const ulong mid_s_stride = ulong(p.k >> 5u);
  const ulong w_c_stride = ulong(p.padded_k >> 1u);
  const ulong w_s_stride = ulong(p.padded_k >> 5u);

  if (w4a4_moe_has_apple10_layout()) {
    constexpr auto desc = matmul2d_descriptor(
        16, 32, 32, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    TgFp4 ta_init(tg_mem.As_c[0], dextents<int32_t, 2>(32, 16), TgUe8(tg_mem.As_s[0]));
    TgFp4 tb_init(tg_mem.Ws_c[0], dextents<int32_t, 2>(32, 32), TgUe8(tg_mem.Ws_s[0]));
    auto c0 = op.template get_destination_cooperative_tensor<TgFp4, TgFp4, float>();
    auto c1 = op.template get_destination_cooperative_tensor<TgFp4, TgFp4, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      c0[i] = 0.0f;
      c1[i] = 0.0f;
    }

    const bool compact32 = (tg_mem.tg_meta[2] <= 32u);
    const uint m_off = (sg >> 1u) * (compact32 ? 16u : 32u);
    const uint sg_n = sg & 1u;

    for (uint kt = 0u; kt < k_tiles128; ++kt) {
      const uint r = tid >> 1u;
      const uint h = tid & 1u;
      const uint vr = tg_mem.tg_meta[2];
      const uint w_col = tg_mem.tg_meta[3] + r;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (r < vr) {
        const ulong row_idx = ulong(tg_mem.tg_meta[1] + r);
        device const uint4 *ap = reinterpret_cast<device const uint4 *>(
            mid_qa + row_idx * mid_c_stride + ulong(kt) * 64ul);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_c[h * 2u + 0u] + r * 16u) = ap[h * 2u + 0u];
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_c[h * 2u + 1u] + r * 16u) = ap[h * 2u + 1u];
        const ushort s2 = reinterpret_cast<device const ushort *>(
            mid_sa + row_idx * mid_s_stride + ulong(kt) * 4ul)[h];
        tg_mem.As_s[h * 2u + 0u][r] = uchar(s2 & 0xFFu);
        tg_mem.As_s[h * 2u + 1u][r] = uchar(s2 >> 8u);
      } else {
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_c[h * 2u + 0u] + r * 16u) = uint4(0u);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_c[h * 2u + 1u] + r * 16u) = uint4(0u);
        tg_mem.As_s[h * 2u + 0u][r] = 0x7Fu;
        tg_mem.As_s[h * 2u + 1u][r] = 0x7Fu;
      }
      if (w_col < p.n) {
        const ulong w_exp_row = ulong(tg_mem.tg_meta[0]) * ulong(p.n) + ulong(w_col);
        device const uint4 *wp = reinterpret_cast<device const uint4 *>(
            d_codes + w_exp_row * w_c_stride + ulong(kt) * 64ul);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_c[h * 2u + 0u] + r * 16u) = wp[h * 2u + 0u];
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_c[h * 2u + 1u] + r * 16u) = wp[h * 2u + 1u];
        const ushort s2 = reinterpret_cast<device const ushort *>(
            d_scales + w_exp_row * w_s_stride + ulong(kt) * 4ul)[h];
        tg_mem.Ws_s[h * 2u + 0u][r] = uchar(s2 & 0xFFu);
        tg_mem.Ws_s[h * 2u + 1u][r] = uchar(s2 >> 8u);
      } else {
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_c[h * 2u + 0u] + r * 16u) = uint4(0u);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_c[h * 2u + 1u] + r * 16u) = uint4(0u);
        tg_mem.Ws_s[h * 2u + 0u][r] = 0x7Fu;
        tg_mem.Ws_s[h * 2u + 1u][r] = 0x7Fu;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      #pragma clang loop unroll(disable)
      for (uint pl = 0u; pl < 4u; ++pl) {
        TgFp4 tA0(tg_mem.As_c[pl] + m_off * 16u,
                  dextents<int32_t, 2>(32, 16), TgUe8(tg_mem.As_s[pl] + m_off));
        TgFp4 tB(tg_mem.Ws_c[pl] + sg_n * 32u * 16u,
                 dextents<int32_t, 2>(32, 32), TgUe8(tg_mem.Ws_s[pl] + sg_n * 32u));
        op.run(tA0, tB, c0);
        if (!compact32) {
          TgFp4 tA1(tg_mem.As_c[pl] + (m_off + 16u) * 16u,
                    dextents<int32_t, 2>(32, 16), TgUe8(tg_mem.As_s[pl] + m_off + 16u));
          op.run(tA1, tB, c1);
        }
      }
    }

    const uint rb_cur = tg_mem.tg_meta[1];
    const uint vr = tg_mem.tg_meta[2];
    const uint n0_sg = tg_mem.tg_meta[3] + sg_n * 32u;
    if (m_off < vr && n0_sg < p.n) {
      const uint qid = lane >> 2u;
      const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
      const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
      #pragma clang loop unroll(full)
      for (uint h2 = 0u; h2 < 2u; ++h2) {
        const uint rm0 = m_off + fm + h2 * 8u;
        if (rm0 < vr) {
          const uint assign = tile_assign[rb_cur + rm0];
          const float score = float(route_scores[assign]);
          #pragma clang loop unroll(full)
          for (uint u = 0u; u < 2u; ++u) {
            const uint col = n0_sg + u * 16u + fn;
            if (col < p.n) {
              const uint e = u * 8u + h2 * 4u;
              *reinterpret_cast<device bfloat4 *>(expert_output + ulong(assign) * p.n + col) =
                  bfloat4(bfloat(c0[e + 0u] * score), bfloat(c0[e + 1u] * score),
                          bfloat(c0[e + 2u] * score), bfloat(c0[e + 3u] * score));
            }
          }
        }
        if (!compact32) {
          const uint rm1 = rm0 + 16u;
          if (rm1 < vr) {
            const uint assign = tile_assign[rb_cur + rm1];
            const float score = float(route_scores[assign]);
            #pragma clang loop unroll(full)
            for (uint u = 0u; u < 2u; ++u) {
              const uint col = n0_sg + u * 16u + fn;
              if (col < p.n) {
                const uint e = u * 8u + h2 * 4u;
                *reinterpret_cast<device bfloat4 *>(expert_output + ulong(assign) * p.n + col) =
                    bfloat4(bfloat(c1[e + 0u] * score), bfloat(c1[e + 1u] * score),
                            bfloat(c1[e + 2u] * score), bfloat(c1[e + 3u] * score));
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
  const uint h_st = tid & 1u;

  for (uint r4 = 0u; r4 < 8u; ++r4) {
    float4 acc_a9 = float4(0.0f);
    for (uint kt = 0u; kt < k_tiles128; ++kt) {
      const uint w_col = tg_n0 + r_st;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (r_st < valid_rows) {
        const ulong row_idx = ulong(rb + r_st);
        device const uint4 *ap = reinterpret_cast<device const uint4 *>(
            mid_qa + row_idx * mid_c_stride + ulong(kt) * 64ul);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_c[h_st * 2u + 0u] + r_st * 16u) = ap[h_st * 2u + 0u];
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_c[h_st * 2u + 1u] + r_st * 16u) = ap[h_st * 2u + 1u];
        const ushort s2 = reinterpret_cast<device const ushort *>(
            mid_sa + row_idx * mid_s_stride + ulong(kt) * 4ul)[h_st];
        tg_mem.As_s[h_st * 2u + 0u][r_st] = uchar(s2 & 0xFFu);
        tg_mem.As_s[h_st * 2u + 1u][r_st] = uchar(s2 >> 8u);
      } else {
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_c[h_st * 2u + 0u] + r_st * 16u) = uint4(0u);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_c[h_st * 2u + 1u] + r_st * 16u) = uint4(0u);
        tg_mem.As_s[h_st * 2u + 0u][r_st] = 0x7Fu;
        tg_mem.As_s[h_st * 2u + 1u][r_st] = 0x7Fu;
      }
      if (w_col < p.n) {
        const ulong w_exp_row = ulong(expert) * ulong(p.n) + ulong(w_col);
        device const uint4 *wp = reinterpret_cast<device const uint4 *>(
            d_codes + w_exp_row * w_c_stride + ulong(kt) * 64ul);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_c[h_st * 2u + 0u] + r_st * 16u) = wp[h_st * 2u + 0u];
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_c[h_st * 2u + 1u] + r_st * 16u) = wp[h_st * 2u + 1u];
        const ushort s2 = reinterpret_cast<device const ushort *>(
            d_scales + w_exp_row * w_s_stride + ulong(kt) * 4ul)[h_st];
        tg_mem.Ws_s[h_st * 2u + 0u][r_st] = uchar(s2 & 0xFFu);
        tg_mem.Ws_s[h_st * 2u + 1u][r_st] = uchar(s2 >> 8u);
      } else {
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_c[h_st * 2u + 0u] + r_st * 16u) = uint4(0u);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_c[h_st * 2u + 1u] + r_st * 16u) = uint4(0u);
        tg_mem.Ws_s[h_st * 2u + 0u][r_st] = 0x7Fu;
        tg_mem.Ws_s[h_st * 2u + 1u][r_st] = 0x7Fu;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (col_live && r_base + r4 * 4u < valid_rows) {
        for (uint rr = 0u; rr < 4u; ++rr) {
          const uint r_l = r_base + r4 * 4u + rr;
          if (r_l < valid_rows) {
            float sum = 0.0f;
            for (uint pl = 0u; pl < 4u; ++pl) {
              sum += w4a4_moe_dot32_fp4_tg(
                  tg_mem.As_c[pl] + r_l * 16u, tg_mem.As_s[pl][r_l],
                  tg_mem.Ws_c[pl] + c_l * 16u, tg_mem.Ws_s[pl][c_l]);
            }
            acc_a9[rr] += sum;
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
// MoE Native MXFP4 W4A4 Kernel Entry Points (<= 8,720 B TG SRAM, 0 B Spills)
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
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup W4A4MoeSwiTgStorage tg_mem;
  moe_mxfp4_w4a4_nax_swiglu_bm64_out64_impl<false>(
      a_codes, a_scales, tile_assign, tile_expert, tile_valid_rows, total_tiles,
      g_codes, g_scales, u_codes, u_scales, mid_bf16, nullptr, nullptr,
      p, tg_mem, lane, sg, tg);
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
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup W4A4MoeSwiTgStorage tg_mem;
  moe_mxfp4_w4a4_nax_swiglu_bm64_out64_impl<true>(
      a_codes, a_scales, tile_assign, tile_expert, tile_valid_rows, total_tiles,
      g_codes, g_scales, u_codes, u_scales, nullptr, mid_qa, mid_sa,
      p, tg_mem, lane, sg, tg);
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
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup W4A4MoeDownTgStorage tg_mem;
  moe_mxfp4_w4a4_nax_down_scatter_bm64_bn64_impl(
      mid_qa, mid_sa, tile_expert, tile_valid_rows, tile_assign, total_tiles,
      route_scores, d_codes, d_scales, expert_output, p, tg_mem, lane, sg, tg);
}
