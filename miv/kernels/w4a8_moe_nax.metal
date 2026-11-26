// Route A: MoE Prefill W4A8 Fused Gate + Up + SwiGLU (BM=64, BN_out=64, BK=64)
// Consumes shared GS64 INT8 (Qa, Sa, Ra) gathered directly into 5.0 KiB As_i8 SRAM
// (2x less SLC gather traffic than BF16), paired with untiled MoE MXFP4 or oq_affine
// (q3/q4/q5/q6/q8 x g64/g128) weights -> BF16 mid_bf16 for W4A16 down_scatter.
// Uses Metal 4 NAX TensorOps `matmul2d_descriptor(16, 32, 32)` int8_t x int8_t -> int32_t.

#include "common_math.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct W4A8MoeGemmParams {
  uint n;          // f (intermediate size)
  uint k;          // h (hidden size)
  uint padded_k;   // untiled weight padded_k (>= k)
  uint max_tiles;  // bit 31: use_gx32, bits 0..30: max_tiles
  uint mp;         // sa/ra row stride: sa[g * mp + tok]
  uint reserved;
};

namespace {

constant constexpr uint kW4A8MoeBK = 64u;
constant constexpr uint kW4A8MoeStrideI8 = 80u; // 64 + 16 B (16B aligned, bank-conflict-free)

inline int8_t w4a8_moe_mxfp4_e2m1_to_i8_x2(uint nib) {
  const int mag2 = int((0xC8643210u >> ((nib & 7u) * 4u)) & 0xFu);
  return int8_t((nib & 8u) ? -mag2 : mag2);
}

inline void w4a8_moe_mxfp4_unpack_raw_32_i8(uint4 raw, threadgroup char4 *w_dst4) {
  #pragma clang loop unroll(full)
  for (uint w = 0u; w < 4u; ++w) {
    const uint wd = raw[w];
    w_dst4[w * 2u + 0u] = char4(
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >>  0u) & 0xFu),
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >>  4u) & 0xFu),
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >>  8u) & 0xFu),
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >> 12u) & 0xFu));
    w_dst4[w * 2u + 1u] = char4(
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >> 16u) & 0xFu),
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >> 20u) & 0xFu),
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >> 24u) & 0xFu),
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >> 28u) & 0xFu));
  }
}

struct W4A8MoeAffineRaw32 {
  float sw;
  float bw;
  uint4 r0;
  uint4 r1;
};

template <uint Bits, uint GroupSize>
inline W4A8MoeAffineRaw32 w4a8_moe_fetch_affine_32(
    device const uchar *codes,
    device const uchar *high,
    device const bfloat *scales,
    device const bfloat *biases,
    ulong expert_row,
    uint padded_k,
    uint g,
    uint half_k,
    bool row_live,
    bool load_sb) {
  W4A8MoeAffineRaw32 out = {0.0f, 0.0f, uint4(0u), uint4(0u)};
  if (!row_live) {
    return out;
  }
  const uint k_col = g * kW4A8MoeBK + half_k * 32u;
  if (load_sb) {
    const uint sb_g = (GroupSize == 128u) ? (g >> 1u) : g;
    const ulong sb_idx = expert_row * ulong(padded_k / GroupSize) + ulong(sb_g);
    out.sw = float(scales[sb_idx]);
    out.bw = float(biases[sb_idx]);
    if constexpr (Bits == 8u) {
      out.bw = fma(128.0f, out.sw, out.bw);
    }
  }
  if constexpr (Bits == 4u) {
    out.r0 = *reinterpret_cast<device const uint4 *>(
        codes + expert_row * ulong(padded_k / 2u) + ulong(k_col / 2u));
  } else if constexpr (Bits == 8u) {
    device const uint4 *p4 = reinterpret_cast<device const uint4 *>(
        codes + expert_row * ulong(padded_k) + ulong(k_col));
    out.r0 = p4[0];
    out.r1 = p4[1];
  } else if constexpr (Bits == 6u) {
    out.r0 = *reinterpret_cast<device const uint4 *>(
        codes + expert_row * ulong(padded_k / 2u) + ulong(k_col / 2u));
    const uint2 hi2 = *reinterpret_cast<device const uint2 *>(
        high + expert_row * ulong(padded_k / 4u) + ulong(k_col / 4u));
    out.r1 = uint4(hi2.x, hi2.y, 0u, 0u);
  } else if constexpr (Bits == 5u) {
    out.r0 = *reinterpret_cast<device const uint4 *>(
        codes + expert_row * ulong(padded_k / 2u) + ulong(k_col / 2u));
    const uint hi1 = *reinterpret_cast<device const uint *>(
        high + expert_row * ulong(padded_k / 8u) + ulong(k_col / 8u));
    out.r1 = uint4(hi1, 0u, 0u, 0u);
  } else {
    const uint2 lo2 = *reinterpret_cast<device const uint2 *>(
        codes + expert_row * ulong(padded_k / 4u) + ulong(k_col / 4u));
    const uint hi1 = *reinterpret_cast<device const uint *>(
        high + expert_row * ulong(padded_k / 8u) + ulong(k_col / 8u));
    out.r0 = uint4(lo2.x, lo2.y, hi1, 0u);
  }
  return out;
}

template <uint Bits>
inline void w4a8_moe_unpack_affine_32_i8(
    const W4A8MoeAffineRaw32 raw,
    threadgroup char4 *w_dst4) {
  if constexpr (Bits == 4u) {
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      const uint wd = raw.r0[w];
      w_dst4[w * 2u + 0u] = char4(
          int8_t((wd >>  0u) & 0xFu), int8_t((wd >>  4u) & 0xFu),
          int8_t((wd >>  8u) & 0xFu), int8_t((wd >> 12u) & 0xFu));
      w_dst4[w * 2u + 1u] = char4(
          int8_t((wd >> 16u) & 0xFu), int8_t((wd >> 20u) & 0xFu),
          int8_t((wd >> 24u) & 0xFu), int8_t((wd >> 28u) & 0xFu));
    }
  } else if constexpr (Bits == 8u) {
    const uint4 s0 = raw.r0 ^ uint4(0x80808080u);
    const uint4 s1 = raw.r1 ^ uint4(0x80808080u);
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      w_dst4[w]      = as_type<char4>(s0[w]);
      w_dst4[4u + w] = as_type<char4>(s1[w]);
    }
  } else if constexpr (Bits == 6u) {
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      const uint wd = raw.r0[w];
      const uint h16 = (w < 2u) ? (raw.r1.x >> (w * 16u)) : (raw.r1.y >> ((w - 2u) * 16u));
      w_dst4[w * 2u + 0u] = char4(
          int8_t(((wd >>  0u) & 0xFu) | (((h16 >> 0u) & 3u) << 4u)),
          int8_t(((wd >>  4u) & 0xFu) | (((h16 >> 2u) & 3u) << 4u)),
          int8_t(((wd >>  8u) & 0xFu) | (((h16 >> 4u) & 3u) << 4u)),
          int8_t(((wd >> 12u) & 0xFu) | (((h16 >> 6u) & 3u) << 4u)));
      w_dst4[w * 2u + 1u] = char4(
          int8_t(((wd >> 16u) & 0xFu) | (((h16 >>  8u) & 3u) << 4u)),
          int8_t(((wd >> 20u) & 0xFu) | (((h16 >> 10u) & 3u) << 4u)),
          int8_t(((wd >> 24u) & 0xFu) | (((h16 >> 12u) & 3u) << 4u)),
          int8_t(((wd >> 28u) & 0xFu) | (((h16 >> 14u) & 3u) << 4u)));
    }
  } else if constexpr (Bits == 5u) {
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      const uint wd = raw.r0[w];
      const uint hb = (raw.r1.x >> (w * 8u)) & 0xFFu;
      w_dst4[w * 2u + 0u] = char4(
          int8_t(((wd >>  0u) & 0xFu) | (((hb >> 0u) & 1u) << 4u)),
          int8_t(((wd >>  4u) & 0xFu) | (((hb >> 1u) & 1u) << 4u)),
          int8_t(((wd >>  8u) & 0xFu) | (((hb >> 2u) & 1u) << 4u)),
          int8_t(((wd >> 12u) & 0xFu) | (((hb >> 3u) & 1u) << 4u)));
      w_dst4[w * 2u + 1u] = char4(
          int8_t(((wd >> 16u) & 0xFu) | (((hb >> 4u) & 1u) << 4u)),
          int8_t(((wd >> 20u) & 0xFu) | (((hb >> 5u) & 1u) << 4u)),
          int8_t(((wd >> 24u) & 0xFu) | (((hb >> 6u) & 1u) << 4u)),
          int8_t(((wd >> 28u) & 0xFu) | (((hb >> 7u) & 1u) << 4u)));
    }
  } else {
    #pragma clang loop unroll(full)
    for (uint q16 = 0u; q16 < 2u; ++q16) {
      const uint lw = (q16 == 0u) ? raw.r0.x : raw.r0.y;
      const uint hw = raw.r0.z >> (q16 * 16u);
      #pragma clang loop unroll(full)
      for (uint c4 = 0u; c4 < 4u; ++c4) {
        const uint l8 = lw >> (c4 * 8u);
        const uint h4 = hw >> (c4 * 4u);
        w_dst4[q16 * 4u + c4] = char4(
            int8_t(((l8 >> 0u) & 3u) | (((h4 >> 0u) & 1u) << 2u)),
            int8_t(((l8 >> 2u) & 3u) | (((h4 >> 1u) & 1u) << 2u)),
            int8_t(((l8 >> 4u) & 3u) | (((h4 >> 2u) & 1u) << 2u)),
            int8_t(((l8 >> 6u) & 3u) | (((h4 >> 3u) & 1u) << 2u)));
      }
    }
  }
}

template <uint Bits, uint GroupSize>
inline void moe_affine_w4a8_nax_swiglu_bm64_out64_impl(
    device const int8_t *qa,
    device const float *sa,
    device const float *ra,
    device const uint *tile_assign,
    device const uint *tile_expert,
    device const uint *tile_valid_rows,
    device const uint *total_tiles,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const bfloat *g_scales,
    device const bfloat *g_biases,
    device const uchar *u_codes,
    device const uchar *u_high,
    device const bfloat *u_scales,
    device const bfloat *u_biases,
    device bfloat *mid_bf16,
    constant W4A8MoeGemmParams &p,
    threadgroup int8_t *As_i8,
    threadgroup int8_t *Ws_i8,
    threadgroup float *Sa_s,
    threadgroup float *Ra_s,
    threadgroup float *R128_s,
    threadgroup float *Sw_s,
    threadgroup float *Bw_s,
    threadgroup uint *tg_tok,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM_TG = 64u;
  constexpr uint OUT_N_TG = 64u;
  constexpr uint WS_ROWS = 128u;

  const uint n_cols = (p.n + OUT_N_TG - 1u) / OUT_N_TG;
  const bool use_gx32 = (p.max_tiles >> 31u) != 0u;
  const uint m_tile = use_gx32 ? ((tg.y / n_cols) * 32u + tg.x) : tg.y;
  const uint col_tile = use_gx32 ? (tg.y % n_cols) : tg.x;
  const uint tg_n0 = col_tile * OUT_N_TG;
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

  if (tid < BM_TG) {
    uint tok = 0xFFFFFFFFu;
    if (tid < valid_rows) {
      const uint assign = tile_assign[rb + tid];
      if (assign != 0xFFFFFFFFu) {
        tok = assign >> 3u;
      }
    }
    tg_tok[tid] = tok;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg_m = sg >> 2u;
  const uint sg_n = sg & 3u;
  const bool compact32 = (valid_rows <= 32u);
  const uint m_off = sg_m * (compact32 ? 16u : 32u);
  const uint n0_sg = tg_n0 + sg_n * 16u;
  const bool sg_active = (m_off < valid_rows) && (n0_sg < p.n);
  const uint groups = p.k / kW4A8MoeBK;

  constexpr auto desc_i8 = matmul2d_descriptor(
      16, 32, 32, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_i8, execution_simdgroup> op_i8;
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
      As_i8, dextents<int32_t, 2>(int(kW4A8MoeStrideI8), int(BM_TG)));
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
      Ws_i8, dextents<int32_t, 2>(int(kW4A8MoeStrideI8), int(WS_ROWS)));

  auto a_proto = tA.slice(0, int(m_off));
  auto b_proto = tB.slice(0, int(sg_n * 32u));
  auto acc_i32 = op_i8.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(a_proto)>,
      remove_addrspace_t<decltype(b_proto)>,
      int32_t>();

  float acc_f32[2][16];
  float a128_0[2][16];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < 2u; ++mm) {
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      acc_f32[mm][i] = 0.0f;
      a128_0[mm][i] = 0.0f;
    }
  }

  const uint a_r = tid >> 2u;
  const uint a_col16 = (tid & 3u) * 16u;
  const uint a_tok = tg_tok[a_r];
  const bool a_live = (a_r < valid_rows) && (a_tok != 0xFFFFFFFFu);

  const uint w_row = tid >> 1u;
  const uint half_k = tid & 1u;
  const bool is_up = ((w_row >> 4u) & 1u) != 0u;
  const uint out_col = tg_n0 + ((w_row >> 5u) * 16u) + (w_row & 15u);
  const bool w_row_live = (out_col < p.n);
  const ulong expert_row = ulong(expert) * p.n + out_col;
  device const uchar *sel_codes = is_up ? u_codes : g_codes;
  device const uchar *sel_high = is_up ? u_high : g_high;
  device const bfloat *sel_scales = is_up ? u_scales : g_scales;
  device const bfloat *sel_biases = is_up ? u_biases : g_biases;
  threadgroup char4 *w_dst4 =
      reinterpret_cast<threadgroup char4 *>(Ws_i8 + w_row * kW4A8MoeStrideI8 + half_k * 32u);

  for (uint g = 0u; g < groups; ++g) {
    const uint k0 = g * kW4A8MoeBK;
    const bool need_sb = (half_k == 0u) && ((GroupSize == 64u) || ((g & 1u) == 0u));

    const uint4 qa_v = a_live
        ? *reinterpret_cast<device const uint4 *>(qa + ulong(a_tok) * p.k + k0 + a_col16)
        : uint4(0u);
    const W4A8MoeAffineRaw32 raw_w = w4a8_moe_fetch_affine_32<Bits, GroupSize>(
        sel_codes, sel_high, sel_scales, sel_biases, expert_row, p.padded_k,
        g, half_k, w_row_live, need_sb);

    float sa_v = 0.0f, ra_v = 0.0f;
    if (tid < BM_TG) {
      const uint tok_s = tg_tok[tid];
      if (tid < valid_rows && tok_s != 0xFFFFFFFFu) {
        sa_v = sa[ulong(g) * p.mp + tok_s];
        ra_v = ra[ulong(g) * p.mp + tok_s];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    *reinterpret_cast<threadgroup uint4 *>(As_i8 + a_r * kW4A8MoeStrideI8 + a_col16) = qa_v;
    w4a8_moe_unpack_affine_32_i8<Bits>(raw_w, w_dst4);
    if (tid < BM_TG) {
      Sa_s[tid] = sa_v;
      Ra_s[tid] = ra_v;
      if constexpr (GroupSize == 128u) {
        if ((g & 1u) == 0u) {
          R128_s[tid] = sa_v * ra_v;
        } else {
          R128_s[tid] = fma(sa_v, ra_v, R128_s[tid]);
        }
      }
    }
    if (need_sb) {
      Sw_s[w_row] = raw_w.sw;
      Bw_s[w_row] = raw_w.bw;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_active) {
      const int col_base = int(sg_n * 32u);
      auto b0 = tB.slice(0, col_base);
      auto b1 = tB.slice(32, col_base);
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            acc_i32[i] = 0;
          }
          const int row_base = int(m_off + mm * 16u);
          auto a0 = tA.slice(0, row_base);
          auto a1 = tA.slice(32, row_base);
          op_i8.run(a0, b0, acc_i32);
          op_i8.run(a1, b1, acc_i32);
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            if (acc_i32.is_valid_element(i)) {
              const auto idx = acc_i32.get_multidimensional_index(i);
              const uint c_l = sg_n * 32u + uint(idx[0]);
              const uint r_l = m_off + mm * 16u + uint(idx[1]);
              if constexpr (GroupSize == 64u) {
                acc_f32[mm][i] = fma(
                    Sa_s[r_l],
                    fma(Sw_s[c_l], float(acc_i32[i]), Bw_s[c_l] * Ra_s[r_l]),
                    acc_f32[mm][i]);
              } else {
                if ((g & 1u) == 0u) {
                  a128_0[mm][i] = Sa_s[r_l] * float(acc_i32[i]);
                } else {
                  const float a_tot = fma(Sa_s[r_l], float(acc_i32[i]), a128_0[mm][i]);
                  acc_f32[mm][i] = fma(
                      Sw_s[c_l], a_tot,
                      fma(Bw_s[c_l], R128_s[r_l], acc_f32[mm][i]));
                }
              }
            }
          }
        }
      }
    }
  }

  // Portable 0-ULP Qwen SwiGLU epilogue + 8-row-aligned zero-pad reusing dead Ws_i8 (10 KiB >= 8 KiB).
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg_active) {
    threadgroup bfloat *sg_buf = reinterpret_cast<threadgroup bfloat *>(Ws_i8) + sg * 512u;
    const uint rb_8ceil = (valid_rows + 7u) & ~7u;
    const uint r_out_base = (lane >> 2u) * 2u; // 0, 2, ..., 14
    const uint c_out4 = (lane & 3u) * 4u;      // 0, 4, 8, 12
    const uint col = n0_sg + c_out4;
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm) {
      if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          if (acc_i32.is_valid_element(i)) {
            const auto idx = acc_i32.get_multidimensional_index(i);
            sg_buf[uint(idx[1]) * 32u + uint(idx[0])] = bfloat(acc_f32[mm][i]);
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (col < p.n) {
          #pragma clang loop unroll(full)
          for (uint dr = 0u; dr < 2u; ++dr) {
            const uint r = r_out_base + dr;
            const uint rm = m_off + mm * 16u + r;
            if (rm < valid_rows) {
              const bfloat4 g4 = *reinterpret_cast<const threadgroup bfloat4 *>(
                  sg_buf + r * 32u + c_out4);
              const bfloat4 u4 = *reinterpret_cast<const threadgroup bfloat4 *>(
                  sg_buf + r * 32u + 16u + c_out4);
              bfloat4 out_b;
              #pragma clang loop unroll(full)
              for (uint j = 0u; j < 4u; ++j) {
                out_b[j] = bfloat(nax_silu_f32(float(g4[j])) * float(u4[j]));
              }
              *reinterpret_cast<device bfloat4 *>(
                  mid_bf16 + ulong(rb + rm) * p.n + col) = out_b;
            } else if (rm < rb_8ceil) {
              *reinterpret_cast<device bfloat4 *>(
                  mid_bf16 + ulong(rb + rm) * p.n + col) = bfloat4(bfloat(0.0f));
            }
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }
}

} // namespace

// ============================================================================
// MoE MXFP4 W4A8 Fused Gate + Up + SwiGLU (BM=64, BN_out=64, k_sub=32, 16.5 KiB SRAM)
// ============================================================================
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_moe_mxfp4_w4a8_nax_swiglu_bm64_out64_sk1(
    device const int8_t *qa [[buffer(0)]],
    device const float *sa [[buffer(1)]],
    device const uint *tile_assign [[buffer(2)]],
    device const uint *tile_expert [[buffer(3)]],
    device const uint *tile_valid_rows [[buffer(4)]],
    device const uint *total_tiles [[buffer(5)]],
    device const uchar *g_codes [[buffer(6)]],
    device const uchar *g_scales [[buffer(7)]],
    device const uchar *u_codes [[buffer(8)]],
    device const uchar *u_scales [[buffer(9)]],
    device bfloat *mid_bf16 [[buffer(10)]],
    constant W4A8MoeGemmParams &p [[buffer(11)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint BM_TG = 64u;
  constexpr uint OUT_N_TG = 64u;
  constexpr uint WS_ROWS = 128u;

  alignas(16) threadgroup int8_t As_i8[BM_TG * kW4A8MoeStrideI8];
  alignas(16) threadgroup int8_t Ws_i8[WS_ROWS * kW4A8MoeStrideI8];
  threadgroup float Sa_s[BM_TG];
  threadgroup float Sw0_s[WS_ROWS];
  threadgroup float Sw1_s[WS_ROWS];
  threadgroup uint tg_tok[BM_TG];

  const uint n_cols = (p.n + OUT_N_TG - 1u) / OUT_N_TG;
  const bool use_gx32 = (p.max_tiles >> 31u) != 0u;
  const uint m_tile = use_gx32 ? ((tg.y / n_cols) * 32u + tg.x) : tg.y;
  const uint col_tile = use_gx32 ? (tg.y % n_cols) : tg.x;
  const uint tg_n0 = col_tile * OUT_N_TG;
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

  if (tid < BM_TG) {
    uint tok = 0xFFFFFFFFu;
    if (tid < valid_rows) {
      const uint assign = tile_assign[rb + tid];
      if (assign != 0xFFFFFFFFu) {
        tok = assign >> 3u;
      }
    }
    tg_tok[tid] = tok;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg_m = sg >> 2u;
  const uint sg_n = sg & 3u;
  const bool compact32 = (valid_rows <= 32u);
  const uint m_off = sg_m * (compact32 ? 16u : 32u);
  const uint n0_sg = tg_n0 + sg_n * 16u;
  const bool sg_active = (m_off < valid_rows) && (n0_sg < p.n);
  const uint k_blocks = p.k / kW4A8MoeBK;

  constexpr auto desc_i8 = matmul2d_descriptor(
      16, 32, 32, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_i8, execution_simdgroup> op_i8;
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
      As_i8, dextents<int32_t, 2>(int(kW4A8MoeStrideI8), int(BM_TG)));
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
      Ws_i8, dextents<int32_t, 2>(int(kW4A8MoeStrideI8), int(WS_ROWS)));

  auto a_proto = tA.slice(0, int(m_off));
  auto b_proto = tB.slice(0, int(sg_n * 32u));
  auto acc_i32 = op_i8.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(a_proto)>,
      remove_addrspace_t<decltype(b_proto)>,
      int32_t>();

  float acc_f32[2][16];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < 2u; ++mm) {
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      acc_f32[mm][i] = 0.0f;
    }
  }

  const uint a_r = tid >> 2u;
  const uint a_col16 = (tid & 3u) * 16u;
  const uint a_tok = tg_tok[a_r];
  const bool a_live = (a_r < valid_rows) && (a_tok != 0xFFFFFFFFu);

  const uint w_row = tid >> 1u;
  const uint col32 = tid & 1u;
  const bool is_up = ((w_row >> 4u) & 1u) != 0u;
  const uint out_col = tg_n0 + ((w_row >> 5u) * 16u) + (w_row & 15u);
  const bool w_row_live = (out_col < p.n);
  const ulong expert_row = ulong(expert) * p.n + out_col;
  device const uchar *sel_codes = is_up ? u_codes : g_codes;
  device const uchar *sel_scales = is_up ? u_scales : g_scales;
  threadgroup char4 *w_dst4 =
      reinterpret_cast<threadgroup char4 *>(Ws_i8 + w_row * kW4A8MoeStrideI8 + col32 * 32u);

  for (uint kb = 0u; kb < k_blocks; ++kb) {
    const uint k0 = kb * kW4A8MoeBK;
    const uint k_col = k0 + col32 * 32u;

    const uint4 qa_v = a_live
        ? *reinterpret_cast<device const uint4 *>(qa + ulong(a_tok) * p.k + k0 + a_col16)
        : uint4(0u);
    uint4 raw_w = uint4(0u);
    float sw_v = 0.0f;
    if (w_row_live) {
      raw_w = *reinterpret_cast<device const uint4 *>(
          sel_codes + expert_row * ulong(p.padded_k / 2u) + ulong(k_col / 2u));
      sw_v = 0.5f * e8m0_to_float(
          sel_scales[expert_row * ulong(p.padded_k / 32u) + ulong(k_col / 32u)]);
    }
    float sa_v = 0.0f;
    if (tid < BM_TG) {
      const uint tok_s = tg_tok[tid];
      if (tid < valid_rows && tok_s != 0xFFFFFFFFu) {
        sa_v = sa[ulong(kb) * p.mp + tok_s];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    *reinterpret_cast<threadgroup uint4 *>(As_i8 + a_r * kW4A8MoeStrideI8 + a_col16) = qa_v;
    w4a8_moe_mxfp4_unpack_raw_32_i8(raw_w, w_dst4);
    if (tid < BM_TG) {
      Sa_s[tid] = sa_v;
    }
    if (col32 == 0u) {
      Sw0_s[w_row] = sw_v;
    } else {
      Sw1_s[w_row] = sw_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_active) {
      const int col_base = int(sg_n * 32u);
      auto b0 = tB.slice(0, col_base);
      auto b1 = tB.slice(32, col_base);
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
          const int row_base = int(m_off + mm * 16u);
          auto a0 = tA.slice(0, row_base);
          auto a1 = tA.slice(32, row_base);
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            acc_i32[i] = 0;
          }
          op_i8.run(a0, b0, acc_i32);
          float p0[16];
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            if (acc_i32.is_valid_element(i)) {
              const auto idx = acc_i32.get_multidimensional_index(i);
              p0[i] = float(acc_i32[i]) * Sw0_s[sg_n * 32u + uint(idx[0])];
            } else {
              p0[i] = 0.0f;
            }
          }
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            acc_i32[i] = 0;
          }
          op_i8.run(a1, b1, acc_i32);
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            if (acc_i32.is_valid_element(i)) {
              const auto idx = acc_i32.get_multidimensional_index(i);
              const uint c_l = sg_n * 32u + uint(idx[0]);
              const uint r_l = m_off + mm * 16u + uint(idx[1]);
              const float partial64 = fma(float(acc_i32[i]), Sw1_s[c_l], p0[i]);
              acc_f32[mm][i] = fma(Sa_s[r_l], partial64, acc_f32[mm][i]);
            }
          }
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg_active) {
    threadgroup bfloat *sg_buf = reinterpret_cast<threadgroup bfloat *>(Ws_i8) + sg * 512u;
    const uint rb_8ceil = (valid_rows + 7u) & ~7u;
    const uint r_out_base = (lane >> 2u) * 2u;
    const uint c_out4 = (lane & 3u) * 4u;
    const uint col = n0_sg + c_out4;
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm) {
      if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          if (acc_i32.is_valid_element(i)) {
            const auto idx = acc_i32.get_multidimensional_index(i);
            sg_buf[uint(idx[1]) * 32u + uint(idx[0])] = bfloat(acc_f32[mm][i]);
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (col < p.n) {
          #pragma clang loop unroll(full)
          for (uint dr = 0u; dr < 2u; ++dr) {
            const uint r = r_out_base + dr;
            const uint rm = m_off + mm * 16u + r;
            if (rm < valid_rows) {
              const bfloat4 g4 = *reinterpret_cast<const threadgroup bfloat4 *>(
                  sg_buf + r * 32u + c_out4);
              const bfloat4 u4 = *reinterpret_cast<const threadgroup bfloat4 *>(
                  sg_buf + r * 32u + 16u + c_out4);
              bfloat4 out_b;
              #pragma clang loop unroll(full)
              for (uint j = 0u; j < 4u; ++j) {
                out_b[j] = bfloat(nax_silu_f32(float(g4[j])) * float(u4[j]));
              }
              *reinterpret_cast<device bfloat4 *>(
                  mid_bf16 + ulong(rb + rm) * p.n + col) = out_b;
            } else if (rm < rb_8ceil) {
              *reinterpret_cast<device bfloat4 *>(
                  mid_bf16 + ulong(rb + rm) * p.n + col) = bfloat4(bfloat(0.0f));
            }
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }
}

// ============================================================================
// MoE Affine W4A8 Fused Gate + Up + SwiGLU (q3/q4/q5/q6/q8 x g64/g128, 17 KiB SRAM)
// ============================================================================
#define OMLX_M5_MOE_AFFINE_W4A8_NO_HIGH(BITS, GS)                              \
  [[max_total_threads_per_threadgroup(256)]]                                   \
  kernel void omlx_m5_moe_affine_w4a8_nax_swiglu_q##BITS##_g##GS##_bm64_out64_sk1( \
      device const int8_t *qa [[buffer(0)]],                                   \
      device const float *sa [[buffer(1)]],                                    \
      device const float *ra [[buffer(2)]],                                    \
      device const uint *tile_assign [[buffer(3)]],                            \
      device const uint *tile_expert [[buffer(4)]],                            \
      device const uint *tile_valid_rows [[buffer(5)]],                        \
      device const uint *total_tiles [[buffer(6)]],                            \
      device const uchar *g_codes [[buffer(7)]],                               \
      device const bfloat *g_scales [[buffer(8)]],                             \
      device const bfloat *g_biases [[buffer(9)]],                             \
      device const uchar *u_codes [[buffer(10)]],                              \
      device const bfloat *u_scales [[buffer(11)]],                            \
      device const bfloat *u_biases [[buffer(12)]],                            \
      device bfloat *mid_bf16 [[buffer(13)]],                                  \
      constant W4A8MoeGemmParams &p [[buffer(14)]],                            \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup int8_t As_i8[64u * kW4A8MoeStrideI8];              \
    alignas(16) threadgroup int8_t Ws_i8[128u * kW4A8MoeStrideI8];             \
    threadgroup float Sa_s[64u];                                               \
    threadgroup float Ra_s[64u];                                               \
    threadgroup float R128_s[(GS) == 128u ? 64u : 1u];                         \
    threadgroup float Sw_s[128u];                                              \
    threadgroup float Bw_s[128u];                                              \
    threadgroup uint tg_tok[64u];                                              \
    moe_affine_w4a8_nax_swiglu_bm64_out64_impl<BITS##u, GS##u>(                \
        qa, sa, ra, tile_assign, tile_expert, tile_valid_rows, total_tiles,    \
        g_codes, nullptr, g_scales, g_biases, u_codes, nullptr, u_scales,      \
        u_biases, mid_bf16, p, As_i8, Ws_i8, Sa_s, Ra_s, R128_s, Sw_s, Bw_s,   \
        tg_tok, lane, sg, tg);                                                 \
  }

#define OMLX_M5_MOE_AFFINE_W4A8_WITH_HIGH(BITS, GS)                            \
  [[max_total_threads_per_threadgroup(256)]]                                   \
  kernel void omlx_m5_moe_affine_w4a8_nax_swiglu_q##BITS##_g##GS##_bm64_out64_sk1( \
      device const int8_t *qa [[buffer(0)]],                                   \
      device const float *sa [[buffer(1)]],                                    \
      device const float *ra [[buffer(2)]],                                    \
      device const uint *tile_assign [[buffer(3)]],                            \
      device const uint *tile_expert [[buffer(4)]],                            \
      device const uint *tile_valid_rows [[buffer(5)]],                        \
      device const uint *total_tiles [[buffer(6)]],                            \
      device const uchar *g_codes [[buffer(7)]],                               \
      device const uchar *g_high [[buffer(8)]],                                \
      device const bfloat *g_scales [[buffer(9)]],                             \
      device const bfloat *g_biases [[buffer(10)]],                            \
      device const uchar *u_codes [[buffer(11)]],                              \
      device const uchar *u_high [[buffer(12)]],                               \
      device const bfloat *u_scales [[buffer(13)]],                            \
      device const bfloat *u_biases [[buffer(14)]],                            \
      device bfloat *mid_bf16 [[buffer(15)]],                                  \
      constant W4A8MoeGemmParams &p [[buffer(16)]],                            \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup int8_t As_i8[64u * kW4A8MoeStrideI8];              \
    alignas(16) threadgroup int8_t Ws_i8[128u * kW4A8MoeStrideI8];             \
    threadgroup float Sa_s[64u];                                               \
    threadgroup float Ra_s[64u];                                               \
    threadgroup float R128_s[(GS) == 128u ? 64u : 1u];                         \
    threadgroup float Sw_s[128u];                                              \
    threadgroup float Bw_s[128u];                                              \
    threadgroup uint tg_tok[64u];                                              \
    moe_affine_w4a8_nax_swiglu_bm64_out64_impl<BITS##u, GS##u>(                \
        qa, sa, ra, tile_assign, tile_expert, tile_valid_rows, total_tiles,    \
        g_codes, g_high, g_scales, g_biases, u_codes, u_high, u_scales,        \
        u_biases, mid_bf16, p, As_i8, Ws_i8, Sa_s, Ra_s, R128_s, Sw_s, Bw_s,   \
        tg_tok, lane, sg, tg);                                                 \
  }

OMLX_M5_MOE_AFFINE_W4A8_WITH_HIGH(3, 64)
OMLX_M5_MOE_AFFINE_W4A8_WITH_HIGH(3, 128)
OMLX_M5_MOE_AFFINE_W4A8_NO_HIGH(4, 64)
OMLX_M5_MOE_AFFINE_W4A8_NO_HIGH(4, 128)
OMLX_M5_MOE_AFFINE_W4A8_WITH_HIGH(5, 64)
OMLX_M5_MOE_AFFINE_W4A8_WITH_HIGH(5, 128)
OMLX_M5_MOE_AFFINE_W4A8_WITH_HIGH(6, 64)
OMLX_M5_MOE_AFFINE_W4A8_WITH_HIGH(6, 128)
OMLX_M5_MOE_AFFINE_W4A8_NO_HIGH(8, 64)
OMLX_M5_MOE_AFFINE_W4A8_NO_HIGH(8, 128)
