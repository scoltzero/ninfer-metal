// Route A (WP1 Part B): Compact TG-Staged High-Occupancy MoE Prefill W4A8
// (Fused Gate + Up + SwiGLU [Mode A BF16 / Mode B FWHT-64 + A8] & W4A8 Down Scatter)
// Metal 4 NAX TensorOps (`matmul2d_descriptor(16, 32, 32)` int8_t x int8_t -> int32_t).
// - All TG SRAM strictly <= 10.0 KiB (3 TGs/core on Apple10 32 KiB/core SRAM), 0 register spills:
//   * MoE Affine SwiGLU (128t, BK=64):       9,216 B (9.00 KiB -> 3 TGs/core!)
//   * MoE MXFP4  SwiGLU (128t, BK=64):      10,240 B (10.00 KiB -> 3 TGs/core!)
//   * MoE Affine Down Scatter (128t, BK=64): 8,960 B (8.75 KiB -> 3 TGs/core!)
//   * MoE MXFP4  Down Scatter (128t, BK=64): 9,984 B (9.75 KiB -> 3 TGs/core!)
// - Directly consumes 3D untiled expert weights (`*_expert_rowsplit_k128_v1`).

#include "common_math.h"
#include "fwht64.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct W4A8MoeGemmParams {
  uint n;          // output features (f for SwiGLU, h for Down Scatter)
  uint k;          // input features  (h for SwiGLU, f for Down Scatter)
  uint padded_k;   // untiled weight padded_k (>= k)
  uint max_tiles;  // bit 31: use_gx32, bits 0..30: max_tiles
  uint mp;         // input sa/ra row stride: sa[g * mp + row]
  uint reserved;   // Mode B SwiGLU: bits 0..30 = mid_mp, bit 31 = skip_fwht64 (0 => apply FWHT-64)
};

namespace {

constant constexpr uint kW4A8MoeBK = 64u;
constant constexpr uint kW4A8MoeLinStride = 64u;
constant constexpr uint kW4A8MoeMxStride = 64u;
constant constexpr uint kW4A8MoeEpiStride = 68u;

struct W4A8MoeAffineSwiTgStorage {
  union {
    struct {
      alignas(16) int8_t As_i8[64u * kW4A8MoeLinStride]; // 4,096 B
      alignas(16) int8_t Ws_i8[64u * kW4A8MoeLinStride]; // 4,096 B
      float Sa_s[64u];                                   //   256 B
      float Ra_s[64u];                                   //   256 B
      uint Sb_s[64u];                                    //   256 B (packed bfloat2)
    } kloop;
    alignas(16) bfloat SwiBuf[64u * kW4A8MoeEpiStride];  // 8,704 B
  };
  uint tg_tok[64u];                                      //   256 B
  uint tg_meta[4u];                                      //    16 B {expert, rb, valid_rows, tg_n0_base}
};
static_assert(sizeof(W4A8MoeAffineSwiTgStorage) == 9232ul,
              "W4A8MoeAffineSwiTgStorage must be 9,232 B (9.02 KiB)");
static_assert(sizeof(W4A8MoeAffineSwiTgStorage) <= kApple10HighOccTgSramBytes,
              "W4A8MoeAffineSwiTgStorage exceeds 10.0 KiB");

struct W4A8MoeMxfp4SwiTgStorage {
  union {
    struct {
      alignas(16) int8_t As_i8[64u * kW4A8MoeMxStride]; // 4,096 B
      alignas(16) int8_t Ws_i8[64u * kW4A8MoeMxStride]; // 4,096 B
      float Sa_s[64u];                                  //   256 B
      float Sw0_s[64u];                                 //   256 B
      float Sw1_s[64u];                                 //   256 B
      uint slow_flag[2u];                               //     8 B
    } kloop;
    alignas(16) bfloat SwiBuf[64u * kW4A8MoeEpiStride]; // 8,704 B
  };
  uint tg_tok[64u];                                     //   256 B
  uint tg_meta[4u];                                     //    16 B {expert, rb, valid_rows, tg_n0_base}
};
static_assert(sizeof(W4A8MoeMxfp4SwiTgStorage) == 9248ul,
              "W4A8MoeMxfp4SwiTgStorage must be 9,248 B (9.03 KiB)");
static_assert(sizeof(W4A8MoeMxfp4SwiTgStorage) <= kApple10HighOccTgSramBytes,
              "W4A8MoeMxfp4SwiTgStorage exceeds 10.0 KiB");

struct W4A8MoeDownAffineTgStorage {
  alignas(16) int8_t As_i8[64u * kW4A8MoeLinStride]; // 4,096 B
  alignas(16) int8_t Ws_i8[64u * kW4A8MoeLinStride]; // 4,096 B
  float Sa_s[64u];                                   //   256 B
  float Ra_s[64u];                                   //   256 B
  uint Sb_s[64u];                                    //   256 B
};
static_assert(sizeof(W4A8MoeDownAffineTgStorage) == 8960ul,
              "W4A8MoeDownAffineTgStorage must be 8,960 B (8.75 KiB)");
static_assert(sizeof(W4A8MoeDownAffineTgStorage) <= kApple10HighOccTgSramBytes,
              "W4A8MoeDownAffineTgStorage exceeds 10.0 KiB");

struct W4A8MoeDownMxfp4TgStorage {
  alignas(16) int8_t As_i8[64u * kW4A8MoeMxStride]; // 4,096 B
  alignas(16) int8_t Ws_i8[64u * kW4A8MoeMxStride]; // 4,096 B
  float Sa_s[64u];                                  //   256 B
  float Sw0_s[64u];                                 //   256 B
  float Sw1_s[64u];                                 //   256 B
  uint slow_flag[2u];                               //     8 B
};
static_assert(sizeof(W4A8MoeDownMxfp4TgStorage) == 8976ul,
              "W4A8MoeDownMxfp4TgStorage must be 8,976 B (8.77 KiB)");
static_assert(sizeof(W4A8MoeDownMxfp4TgStorage) <= kApple10HighOccTgSramBytes,
              "W4A8MoeDownMxfp4TgStorage exceeds 10.0 KiB");

inline bool w4a8_moe_has_apple10_i8_layout() {
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 32, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto b = op.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  ushort i0 = 0u;
  ushort i2 = 2u;
  return (int(b.get_multidimensional_index(i2)[0]) -
          int(b.get_multidimensional_index(i0)[0])) == 2;
}

inline int8_t w4a8_moe_mxfp4_e2m1_to_i8_x2(uint nib) {
  const int mag2 = int((0xC8643210u >> ((nib & 7u) * 4u)) & 0xFu);
  return int8_t((nib & 8u) ? -mag2 : mag2);
}

#define W4A8_MOE_MX_NIB(n, s) \
  (int8_t(((n) & 8) ? -((((0xC8643210u >> (((n) & 7) * 4)) & 0xFu) << (s))) \
                    :  ((((0xC8643210u >> (((n) & 7) * 4)) & 0xFu) << (s)))))
#define W4A8_MOE_MX_PAIR(b, s) \
  ushort(uchar(W4A8_MOE_MX_NIB((b) & 15, s)) | (ushort(uchar(W4A8_MOE_MX_NIB(((b) >> 4) & 15, s))) << 8))
#define W4A8_MOE_MX_ROW16(hi, s) \
  W4A8_MOE_MX_PAIR((hi)*16+0, s),  W4A8_MOE_MX_PAIR((hi)*16+1, s),  W4A8_MOE_MX_PAIR((hi)*16+2, s),  W4A8_MOE_MX_PAIR((hi)*16+3, s), \
  W4A8_MOE_MX_PAIR((hi)*16+4, s),  W4A8_MOE_MX_PAIR((hi)*16+5, s),  W4A8_MOE_MX_PAIR((hi)*16+6, s),  W4A8_MOE_MX_PAIR((hi)*16+7, s), \
  W4A8_MOE_MX_PAIR((hi)*16+8, s),  W4A8_MOE_MX_PAIR((hi)*16+9, s),  W4A8_MOE_MX_PAIR((hi)*16+10, s), W4A8_MOE_MX_PAIR((hi)*16+11, s), \
  W4A8_MOE_MX_PAIR((hi)*16+12, s), W4A8_MOE_MX_PAIR((hi)*16+13, s), W4A8_MOE_MX_PAIR((hi)*16+14, s), W4A8_MOE_MX_PAIR((hi)*16+15, s)
#define W4A8_MOE_MX_TBL256(s) \
  { W4A8_MOE_MX_ROW16(0, s),  W4A8_MOE_MX_ROW16(1, s),  W4A8_MOE_MX_ROW16(2, s),  W4A8_MOE_MX_ROW16(3, s),  \
    W4A8_MOE_MX_ROW16(4, s),  W4A8_MOE_MX_ROW16(5, s),  W4A8_MOE_MX_ROW16(6, s),  W4A8_MOE_MX_ROW16(7, s),  \
    W4A8_MOE_MX_ROW16(8, s),  W4A8_MOE_MX_ROW16(9, s),  W4A8_MOE_MX_ROW16(10, s), W4A8_MOE_MX_ROW16(11, s), \
    W4A8_MOE_MX_ROW16(12, s), W4A8_MOE_MX_ROW16(13, s), W4A8_MOE_MX_ROW16(14, s), W4A8_MOE_MX_ROW16(15, s) }

constant ushort kW4A8MoeMxfp4ByteLutSh[4][256] = {
    W4A8_MOE_MX_TBL256(0), W4A8_MOE_MX_TBL256(1), W4A8_MOE_MX_TBL256(2), W4A8_MOE_MX_TBL256(3)
};

#undef W4A8_MOE_MX_TBL256
#undef W4A8_MOE_MX_ROW16
#undef W4A8_MOE_MX_PAIR
#undef W4A8_MOE_MX_NIB

inline float w4a8_moe_e8m0_safe(uchar code) {
  if (code == 255u) return 0.0f;
  return e8m0_to_float(code);
}

inline void w4a8_moe_unpack_mxfp4_lut_sh_u4(
    uint4 raw_w, uint sh, thread uint4 &wu0, thread uint4 &wu1) {
  constant ushort *lut = kW4A8MoeMxfp4ByteLutSh[sh];
  wu0.x = uint(lut[(raw_w.x >>  0u) & 0xFFu]) | (uint(lut[(raw_w.x >>  8u) & 0xFFu]) << 16u);
  wu0.y = uint(lut[(raw_w.x >> 16u) & 0xFFu]) | (uint(lut[(raw_w.x >> 24u) & 0xFFu]) << 16u);
  wu0.z = uint(lut[(raw_w.y >>  0u) & 0xFFu]) | (uint(lut[(raw_w.y >>  8u) & 0xFFu]) << 16u);
  wu0.w = uint(lut[(raw_w.y >> 16u) & 0xFFu]) | (uint(lut[(raw_w.y >> 24u) & 0xFFu]) << 16u);
  wu1.x = uint(lut[(raw_w.z >>  0u) & 0xFFu]) | (uint(lut[(raw_w.z >>  8u) & 0xFFu]) << 16u);
  wu1.y = uint(lut[(raw_w.z >> 16u) & 0xFFu]) | (uint(lut[(raw_w.z >> 24u) & 0xFFu]) << 16u);
  wu1.z = uint(lut[(raw_w.w >>  0u) & 0xFFu]) | (uint(lut[(raw_w.w >>  8u) & 0xFFu]) << 16u);
  wu1.w = uint(lut[(raw_w.w >> 16u) & 0xFFu]) | (uint(lut[(raw_w.w >> 24u) & 0xFFu]) << 16u);
}

inline void w4a8_moe_store_as_mx_u4(threadgroup int8_t *As_i8, uint a_r, uint a_col16, uint4 qa_v) {
  *reinterpret_cast<threadgroup uint4 *>(As_i8 + a_r * kW4A8MoeMxStride + a_col16) = qa_v;
}

template <uint Bits>
inline void w4a8_moe_decode_sb(uint sb_u, thread float &sw, thread float &bw) {
  const vec<bfloat, 2> pair = as_type<vec<bfloat, 2>>(sb_u);
  sw = float(pair[0]);
  bw = float(pair[1]);
  if constexpr (Bits == 8u) {
    bw = fma(128.0f, sw, bw);
  }
}

template <uint GroupSize>
inline uint w4a8_moe_fetch_sb_packed(
    device const bfloat *scales,
    device const bfloat *biases,
    ulong expert_row,
    bool row_live,
    uint padded_k,
    uint g) {
  if (!row_live) {
    return 0u;
  }
  const uint sb_g = (GroupSize == 128u) ? (g >> 1u) : g;
  const ulong sb_idx = expert_row * ulong(padded_k / GroupSize) + ulong(sb_g);
  return as_type<uint>(vec<bfloat, 2>(scales[sb_idx], biases[sb_idx]));
}

// Coalesced linear-K untiled expert 32-element fetch & unpack to Ws_i8 (8 char4s).
struct W4A8MoeAffRaw32 {
  uint4 r0;
  uint4 r1;
};

template <uint Bits>
inline W4A8MoeAffRaw32 w4a8_moe_fetch_affine_32(
    device const uchar *codes,
    device const uchar *high,
    ulong expert_row,
    bool row_live,
    uint padded_k,
    uint k_off32) {
  W4A8MoeAffRaw32 out = {uint4(0u), uint4(0u)};
  if (!row_live) {
    return out;
  }
  if constexpr (Bits == 4u) {
    out.r0 = *reinterpret_cast<device const uint4 *>(
        codes + expert_row * ulong(padded_k >> 1u) + ulong(k_off32 >> 1u));
  } else if constexpr (Bits == 5u) {
    out.r0 = *reinterpret_cast<device const uint4 *>(
        codes + expert_row * ulong(padded_k >> 1u) + ulong(k_off32 >> 1u));
    out.r1.x = *reinterpret_cast<device const uint *>(
        high + expert_row * ulong(padded_k >> 3u) + ulong(k_off32 >> 3u));
  } else if constexpr (Bits == 6u) {
    out.r0 = *reinterpret_cast<device const uint4 *>(
        codes + expert_row * ulong(padded_k >> 1u) + ulong(k_off32 >> 1u));
    const uint2 h2 = *reinterpret_cast<device const uint2 *>(
        high + expert_row * ulong(padded_k >> 2u) + ulong(k_off32 >> 2u));
    out.r1 = uint4(h2.x, h2.y, 0u, 0u);
  } else if constexpr (Bits == 3u) {
    const uint2 lo2 = *reinterpret_cast<device const uint2 *>(
        codes + expert_row * ulong(padded_k >> 2u) + ulong(k_off32 >> 2u));
    const uint hi1 = *reinterpret_cast<device const uint *>(
        high + expert_row * ulong(padded_k >> 3u) + ulong(k_off32 >> 3u));
    out.r0 = uint4(lo2.x, lo2.y, hi1, 0u);
  } else {
    device const uint4 *p4 = reinterpret_cast<device const uint4 *>(
        codes + expert_row * ulong(padded_k) + ulong(k_off32));
    out.r0 = p4[0];
    out.r1 = p4[1];
  }
  return out;
}

template <uint Bits>
inline void w4a8_moe_unpack_affine_32_i8(W4A8MoeAffRaw32 raw, threadgroup char4 *dst4) {
  if constexpr (Bits == 4u) {
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      const uint wd = raw.r0[w];
      dst4[w * 2u + 0u] = char4(int8_t((wd >>  0u) & 0xFu), int8_t((wd >>  4u) & 0xFu),
                                int8_t((wd >>  8u) & 0xFu), int8_t((wd >> 12u) & 0xFu));
      dst4[w * 2u + 1u] = char4(int8_t((wd >> 16u) & 0xFu), int8_t((wd >> 20u) & 0xFu),
                                int8_t((wd >> 24u) & 0xFu), int8_t((wd >> 28u) & 0xFu));
    }
  } else if constexpr (Bits == 5u) {
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      const uint wd = raw.r0[w];
      const uint hb = (raw.r1.x >> (w * 8u)) & 0xFFu;
      dst4[w * 2u + 0u] = char4(
          int8_t(((wd >>  0u) & 0xFu) | (((hb >> 0u) & 1u) << 4u)),
          int8_t(((wd >>  4u) & 0xFu) | (((hb >> 1u) & 1u) << 4u)),
          int8_t(((wd >>  8u) & 0xFu) | (((hb >> 2u) & 1u) << 4u)),
          int8_t(((wd >> 12u) & 0xFu) | (((hb >> 3u) & 1u) << 4u)));
      dst4[w * 2u + 1u] = char4(
          int8_t(((wd >> 16u) & 0xFu) | (((hb >> 4u) & 1u) << 4u)),
          int8_t(((wd >> 20u) & 0xFu) | (((hb >> 5u) & 1u) << 4u)),
          int8_t(((wd >> 24u) & 0xFu) | (((hb >> 6u) & 1u) << 4u)),
          int8_t(((wd >> 28u) & 0xFu) | (((hb >> 7u) & 1u) << 4u)));
    }
  } else if constexpr (Bits == 6u) {
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      const uint wd = raw.r0[w];
      const uint h16 = (w < 2u) ? (raw.r1.x >> (w * 16u)) : (raw.r1.y >> ((w - 2u) * 16u));
      dst4[w * 2u + 0u] = char4(
          int8_t(((wd >>  0u) & 0xFu) | (((h16 >>  0u) & 3u) << 4u)),
          int8_t(((wd >>  4u) & 0xFu) | (((h16 >>  2u) & 3u) << 4u)),
          int8_t(((wd >>  8u) & 0xFu) | (((h16 >>  4u) & 3u) << 4u)),
          int8_t(((wd >> 12u) & 0xFu) | (((h16 >>  6u) & 3u) << 4u)));
      dst4[w * 2u + 1u] = char4(
          int8_t(((wd >> 16u) & 0xFu) | (((h16 >>  8u) & 3u) << 4u)),
          int8_t(((wd >> 20u) & 0xFu) | (((h16 >> 10u) & 3u) << 4u)),
          int8_t(((wd >> 24u) & 0xFu) | (((h16 >> 12u) & 3u) << 4u)),
          int8_t(((wd >> 28u) & 0xFu) | (((h16 >> 14u) & 3u) << 4u)));
    }
  } else if constexpr (Bits == 3u) {
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      const uint l16 = ((w < 2u) ? raw.r0.x : raw.r0.y) >> ((w & 1u) * 16u);
      const uint h8 = (raw.r0.z >> (w * 8u)) & 0xFFu;
      dst4[w * 2u + 0u] = char4(
          int8_t(((l16 >>  0u) & 3u) | (((h8 >> 0u) & 1u) << 2u)),
          int8_t(((l16 >>  2u) & 3u) | (((h8 >> 1u) & 1u) << 2u)),
          int8_t(((l16 >>  4u) & 3u) | (((h8 >> 2u) & 1u) << 2u)),
          int8_t(((l16 >>  6u) & 3u) | (((h8 >> 3u) & 1u) << 2u)));
      dst4[w * 2u + 1u] = char4(
          int8_t(((l16 >>  8u) & 3u) | (((h8 >> 4u) & 1u) << 2u)),
          int8_t(((l16 >> 10u) & 3u) | (((h8 >> 5u) & 1u) << 2u)),
          int8_t(((l16 >> 12u) & 3u) | (((h8 >> 6u) & 1u) << 2u)),
          int8_t(((l16 >> 14u) & 3u) | (((h8 >> 7u) & 1u) << 2u)));
    }
  } else {
    const uint4 a0 = raw.r0 ^ uint4(0x80808080u);
    const uint4 a1 = raw.r1 ^ uint4(0x80808080u);
    dst4[0] = as_type<char4>(a0.x); dst4[1] = as_type<char4>(a0.y);
    dst4[2] = as_type<char4>(a0.z); dst4[3] = as_type<char4>(a0.w);
    dst4[4] = as_type<char4>(a1.x); dst4[5] = as_type<char4>(a1.y);
    dst4[6] = as_type<char4>(a1.z); dst4[7] = as_type<char4>(a1.w);
  }
}

inline void w4a8_moe_unpack_mxfp4_32_i8(uint4 raw_w, threadgroup char4 *dst4) {
  #pragma clang loop unroll(full)
  for (uint w = 0u; w < 4u; ++w) {
    const uint wd = raw_w[w];
    dst4[w * 2u + 0u] = char4(
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >>  0u) & 0xFu),
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >>  4u) & 0xFu),
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >>  8u) & 0xFu),
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >> 12u) & 0xFu));
    dst4[w * 2u + 1u] = char4(
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >> 16u) & 0xFu),
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >> 20u) & 0xFu),
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >> 24u) & 0xFu),
        w4a8_moe_mxfp4_e2m1_to_i8_x2((wd >> 28u) & 0xFu));
  }
}

// Shared MoE Mode B epilogue: SwiBuf -> (optional FWHT-64) + GS64 INT8 (mid_qa, mid_sa, mid_ra)
template <uint RowsPerSg>
inline void w4a8_moe_swiglu_epilogue_a8_phase(
    threadgroup bfloat *SwiBuf,
    uint rb,
    uint rm_base,
    uint valid_rows,
    uint rb_8ceil,
    uint tg_n0,
    device int8_t *mid_qa,
    device float *mid_sa,
    device float *mid_ra,
    uint f_dim,
    uint mid_mp,
    bool apply_fwht64,
    uint lane,
    uint sg) {
  const uint g_col = tg_n0 / 64u;
  const uint base_r = sg * RowsPerSg;
  #pragma clang loop unroll(disable)
  for (uint rr = 0u; rr < RowsPerSg; ++rr) {
    const uint r_l = base_r + rr;
    const uint rm = rm_base + r_l;
    const uint out_row = rb + rm;
    const ulong q_off = ulong(out_row) * ulong(f_dim) + ulong(g_col) * 64ul;
    if (rm < valid_rows) {
      const bfloat2 s2 =
          reinterpret_cast<const threadgroup bfloat2 *>(SwiBuf + r_l * kW4A8MoeEpiStride)[lane];
      float v0 = float(s2.x);
      float v1 = float(s2.y);
      if (apply_fwht64) {
        omlx_m5_fwht64_sg32(v0, v1, lane);
      }
      omlx_m5_quantize_pair_g64_f32(
          v0, v1, lane, mid_qa + q_off,
          mid_sa + ulong(g_col) * mid_mp + out_row,
          mid_ra + ulong(g_col) * mid_mp + out_row);
    } else if (rm < rb_8ceil) {
      reinterpret_cast<device char2 *>(mid_qa + q_off)[lane] = char2(0);
      if (lane == 0u) {
        mid_sa[ulong(g_col) * mid_mp + out_row] = 0.0f;
        mid_ra[ulong(g_col) * mid_mp + out_row] = 0.0f;
      }
    }
  }
}

// ============================================================================
// MoE Affine Compact TG-Staged W4A8 Fused Gate + Up + SwiGLU (BM=64, BK=64)
// 4 SGs = 128 threads (WM=2, WN=2, SM=32/compact32), 9,216 B (9.00 KiB <= 10.0 KiB) SRAM -> 3 TGs/core, 0 spills!
// - Mode A (BF16): 1-pass BN_out=32
// - Mode B (Fused FWHT-64 + GS64 INT8): 2-pass BN_out=64 with 4 KiB L1-cached mid_qa stash
// ============================================================================
template <uint Bits, uint GroupSize, bool OutputA8>
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
    device int8_t *mid_qa,
    device float *mid_sa,
    device float *mid_ra,
    constant W4A8MoeGemmParams &p,
    threadgroup W4A8MoeAffineSwiTgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM_TG = 64u;
  constexpr uint OUT_N_TG = OutputA8 ? 64u : 32u;
  constexpr uint kPasses = OutputA8 ? 2u : 1u;

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

  if (w4a8_moe_has_apple10_i8_layout()) {
    constexpr auto desc_i8 = matmul2d_descriptor(
        16, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_i8, execution_simdgroup> op_i8;
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
        tg_mem.kloop.As_i8, dextents<int32_t, 2>(int(kW4A8MoeLinStride), 64));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
        tg_mem.kloop.Ws_i8, dextents<int32_t, 2>(int(kW4A8MoeLinStride), 64));

    const bool sg_is_up = ((sg & 1u) != 0u);
    const uint groups = p.k >> 6u;

    // Fast path for valid_rows <= 32 (e.g. T <= 1024 on E=256, top_k=8):
    // Half qa loads (no qa1), half Sa/Ra loads, half NAX MMAs (mm=0 only), 16-reg acc0[16]!
    // Skip duplicate outer specialization only for q3 Mode B (keeps 0 B AGX register spills
    // while its 64-row loop still respects compact32).
    if constexpr (!(Bits == 3u && OutputA8)) {
    if (tg_mem.tg_meta[2] <= 32u) {
      #pragma clang loop unroll(full)
      for (uint hn = 0u; hn < kPasses; ++hn) {
        const uint w_col = tg_mem.tg_meta[3] + hn * 32u + (sg >> 1u) * 16u + (lane >> 1u);
        const bool w_live = (w_col < p.n);
        const ulong w_exp_row = ulong(tg_mem.tg_meta[0]) * ulong(p.n) + ulong(w_col);

        float acc0[16];
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) acc0[i] = 0.0f;

        for (uint g = 0u; g < groups; ++g) {
          const uint k0 = g * kW4A8MoeBK;
          const uint tid = sg * 32u + lane;
          const uint a_col16 = (lane & 3u) * 16u;
          const uint a_r0 = tid >> 2u;
          const uint half_k = lane & 1u;
          const uint w_row = sg * 16u + (lane >> 1u);

          const uint t0 = tg_mem.tg_tok[a_r0];
          const uint4 qa0 = (t0 != 0xFFFFFFFFu)
              ? *reinterpret_cast<device const uint4 *>(qa + ulong(t0) * p.k + k0 + a_col16)
              : uint4(0u);
          if constexpr (Bits == 3u) {
            const uint2 lo2 = w_live
                ? *reinterpret_cast<device const uint2 *>(
                      ((!sg_is_up) ? g_codes : u_codes) +
                      w_exp_row * ulong(p.padded_k >> 2u) + ulong((k0 + half_k * 32u) >> 2u))
                : uint2(0u);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r0 * kW4A8MoeLinStride + a_col16) = qa0;
            if (half_k == 0u) {
              tg_mem.kloop.Sb_s[w_row] = (!sg_is_up)
                  ? w4a8_moe_fetch_sb_packed<GroupSize>(g_scales, g_biases, w_exp_row, w_live, p.padded_k, g)
                  : w4a8_moe_fetch_sb_packed<GroupSize>(u_scales, u_biases, w_exp_row, w_live, p.padded_k, g);
            }
            const uint hi1 = w_live
                ? *reinterpret_cast<device const uint *>(
                      ((!sg_is_up) ? g_high : u_high) +
                      w_exp_row * ulong(p.padded_k >> 3u) + ulong((k0 + half_k * 32u) >> 3u))
                : 0u;
            const W4A8MoeAffRaw32 raw_w = {uint4(lo2.x, lo2.y, hi1, 0u), uint4(0u)};
            w4a8_moe_unpack_affine_32_i8<3u>(
                raw_w, reinterpret_cast<threadgroup char4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MoeLinStride + half_k * 32u));
          } else {
            W4A8MoeAffRaw32 raw_w;
            if (!sg_is_up) {
              raw_w = w4a8_moe_fetch_affine_32<Bits>(
                  g_codes, g_high, w_exp_row, w_live, p.padded_k, k0 + half_k * 32u);
            } else {
              raw_w = w4a8_moe_fetch_affine_32<Bits>(
                  u_codes, u_high, w_exp_row, w_live, p.padded_k, k0 + half_k * 32u);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r0 * kW4A8MoeLinStride + a_col16) = qa0;
            if (half_k == 0u) {
              tg_mem.kloop.Sb_s[w_row] = (!sg_is_up)
                  ? w4a8_moe_fetch_sb_packed<GroupSize>(g_scales, g_biases, w_exp_row, w_live, p.padded_k, g)
                  : w4a8_moe_fetch_sb_packed<GroupSize>(u_scales, u_biases, w_exp_row, w_live, p.padded_k, g);
            }
            w4a8_moe_unpack_affine_32_i8<Bits>(
                raw_w, reinterpret_cast<threadgroup char4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MoeLinStride + half_k * 32u));
          }
          if (tid < 32u) {
            const uint st = tg_mem.tg_tok[tid];
            tg_mem.kloop.Sa_s[tid] = (st != 0xFFFFFFFFu) ? sa[ulong(g) * p.mp + st] : 0.0f;
            tg_mem.kloop.Ra_s[tid] = (st != 0xFFFFFFFFu) ? ra[ulong(g) * p.mp + st] : 0.0f;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          const uint m_off = (sg >> 1u) * 16u;
          const uint sg_n = sg & 1u;
          const uint qid = lane >> 2u;
          const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
          const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
          auto a0 = tA.slice(0, int(m_off));
          auto a1 = tA.slice(32, int(m_off));
          auto b0 = tB.slice(0, int(sg_n * 32u));
          auto b1 = tB.slice(32, int(sg_n * 32u));
          auto ci = op_i8.template get_destination_cooperative_tensor<
              remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) ci[i] = 0;
          op_i8.run(a0, b0, ci);
          op_i8.run(a1, b1, ci);

          const uint r0 = m_off + fm;
          const float sa0 = tg_mem.kloop.Sa_s[r0], sa1 = tg_mem.kloop.Sa_s[r0 + 8u];
          const float ra0 = tg_mem.kloop.Ra_s[r0], ra1 = tg_mem.kloop.Ra_s[r0 + 8u];
          const uint col_sb = sg_n * 32u + fn;
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            float swg, bwg, swu, bwu;
            w4a8_moe_decode_sb<Bits>(tg_mem.kloop.Sb_s[col_sb + c], swg, bwg);
            w4a8_moe_decode_sb<Bits>(tg_mem.kloop.Sb_s[col_sb + 16u + c], swu, bwu);
            acc0[0u + c]  = fma(sa0, fma(swg, float(ci[0u + c]),  bwg * ra0), acc0[0u + c]);
            acc0[4u + c]  = fma(sa1, fma(swg, float(ci[4u + c]),  bwg * ra1), acc0[4u + c]);
            acc0[8u + c]  = fma(sa0, fma(swu, float(ci[8u + c]),  bwu * ra0), acc0[8u + c]);
            acc0[12u + c] = fma(sa1, fma(swu, float(ci[12u + c]), bwu * ra1), acc0[12u + c]);
          }
        }

        const uint sg_n = sg & 1u;
        const uint qid = lane >> 2u;
        const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
        const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
        const uint rb_cur = tg_mem.tg_meta[1];
        const uint vr = tg_mem.tg_meta[2];
        const uint n0_base = tg_mem.tg_meta[3];
        const uint m_off = (sg >> 1u) * 16u;
        const uint n0_sg = n0_base + hn * 32u + sg_n * 16u;

        if constexpr (!OutputA8) {
          (void)mid_qa; (void)mid_sa; (void)mid_ra;
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
              *reinterpret_cast<threadgroup bfloat4 *>(tg_mem.SwiBuf + rm * kW4A8MoeEpiStride + sg_n * 16u + fn) = s0;
              *reinterpret_cast<threadgroup bfloat4 *>(tg_mem.SwiBuf + rm * kW4A8MoeEpiStride + 32u + sg_n * 16u + fn) = s1;
            }
          }
        }
      }
      if constexpr (OutputA8) {
        const uint rb_cur = tg_mem.tg_meta[1];
        const uint vr = tg_mem.tg_meta[2];
        const uint n0_base = tg_mem.tg_meta[3];
        const uint rb_8c = (vr + 7u) & ~7u;
        const uint mid_mp = p.reserved & 0x7FFFFFFFu;
        const bool apply_fwht64 = ((p.reserved >> 31u) == 0u);
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        w4a8_moe_swiglu_epilogue_a8_phase<8u>(
            tg_mem.SwiBuf, rb_cur, 0u, vr, rb_8c, n0_base,
            mid_qa, mid_sa, mid_ra, p.n, mid_mp, apply_fwht64, lane, sg);
      }
      return;
    }
    }

    #pragma clang loop unroll(full)
    for (uint hn = 0u; hn < kPasses; ++hn) {
      const uint w_col = tg_mem.tg_meta[3] + hn * 32u + (sg >> 1u) * 16u + (lane >> 1u);
      const bool w_live = (w_col < p.n);
      const ulong w_exp_row = ulong(tg_mem.tg_meta[0]) * ulong(p.n) + ulong(w_col);

      float acc_f32[2][16];
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) acc_f32[mm][i] = 0.0f;
      }

      for (uint g = 0u; g < groups; ++g) {
        const uint k0 = g * kW4A8MoeBK;
        const uint tid = sg * 32u + lane;
        const uint a_col16 = (lane & 3u) * 16u;
        const uint a_r0 = tid >> 2u;
        const uint a_r1 = a_r0 + 32u;
        const uint half_k = lane & 1u;
        const uint w_row = sg * 16u + (lane >> 1u);

        const uint t0 = tg_mem.tg_tok[a_r0];
        const uint t1 = tg_mem.tg_tok[a_r1];
        const uint4 qa0 = (t0 != 0xFFFFFFFFu)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(t0) * p.k + k0 + a_col16)
            : uint4(0u);
        const uint4 qa1 = (t1 != 0xFFFFFFFFu)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(t1) * p.k + k0 + a_col16)
            : uint4(0u);
        if constexpr (Bits == 3u) {
          const uint2 lo2 = w_live
              ? *reinterpret_cast<device const uint2 *>(
                    ((!sg_is_up) ? g_codes : u_codes) +
                    w_exp_row * ulong(p.padded_k >> 2u) + ulong((k0 + half_k * 32u) >> 2u))
              : uint2(0u);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r0 * kW4A8MoeLinStride + a_col16) = qa0;
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r1 * kW4A8MoeLinStride + a_col16) = qa1;
          if (half_k == 0u) {
            tg_mem.kloop.Sb_s[w_row] = (!sg_is_up)
                ? w4a8_moe_fetch_sb_packed<GroupSize>(g_scales, g_biases, w_exp_row, w_live, p.padded_k, g)
                : w4a8_moe_fetch_sb_packed<GroupSize>(u_scales, u_biases, w_exp_row, w_live, p.padded_k, g);
          }
          const uint hi1 = w_live
              ? *reinterpret_cast<device const uint *>(
                    ((!sg_is_up) ? g_high : u_high) +
                    w_exp_row * ulong(p.padded_k >> 3u) + ulong((k0 + half_k * 32u) >> 3u))
              : 0u;
          const W4A8MoeAffRaw32 raw_w = {uint4(lo2.x, lo2.y, hi1, 0u), uint4(0u)};
          w4a8_moe_unpack_affine_32_i8<3u>(
              raw_w, reinterpret_cast<threadgroup char4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MoeLinStride + half_k * 32u));
        } else if constexpr (Bits == 5u) {
          const uint4 r0_5 = w_live
              ? *reinterpret_cast<device const uint4 *>(
                    ((!sg_is_up) ? g_codes : u_codes) +
                    w_exp_row * ulong(p.padded_k >> 1u) + ulong((k0 + half_k * 32u) >> 1u))
              : uint4(0u);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r0 * kW4A8MoeLinStride + a_col16) = qa0;
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r1 * kW4A8MoeLinStride + a_col16) = qa1;
          if (half_k == 0u) {
            tg_mem.kloop.Sb_s[w_row] = (!sg_is_up)
                ? w4a8_moe_fetch_sb_packed<GroupSize>(g_scales, g_biases, w_exp_row, w_live, p.padded_k, g)
                : w4a8_moe_fetch_sb_packed<GroupSize>(u_scales, u_biases, w_exp_row, w_live, p.padded_k, g);
          }
          const uint hi1 = w_live
              ? *reinterpret_cast<device const uint *>(
                    ((!sg_is_up) ? g_high : u_high) +
                    w_exp_row * ulong(p.padded_k >> 3u) + ulong((k0 + half_k * 32u) >> 3u))
              : 0u;
          const W4A8MoeAffRaw32 raw_w = {r0_5, uint4(hi1, 0u, 0u, 0u)};
          w4a8_moe_unpack_affine_32_i8<5u>(
              raw_w, reinterpret_cast<threadgroup char4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MoeLinStride + half_k * 32u));
        } else {
          W4A8MoeAffRaw32 raw_w;
          if (!sg_is_up) {
            raw_w = w4a8_moe_fetch_affine_32<Bits>(
                g_codes, g_high, w_exp_row, w_live, p.padded_k, k0 + half_k * 32u);
          } else {
            raw_w = w4a8_moe_fetch_affine_32<Bits>(
                u_codes, u_high, w_exp_row, w_live, p.padded_k, k0 + half_k * 32u);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r0 * kW4A8MoeLinStride + a_col16) = qa0;
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r1 * kW4A8MoeLinStride + a_col16) = qa1;
          if (half_k == 0u) {
            tg_mem.kloop.Sb_s[w_row] = (!sg_is_up)
                ? w4a8_moe_fetch_sb_packed<GroupSize>(g_scales, g_biases, w_exp_row, w_live, p.padded_k, g)
                : w4a8_moe_fetch_sb_packed<GroupSize>(u_scales, u_biases, w_exp_row, w_live, p.padded_k, g);
          }
          w4a8_moe_unpack_affine_32_i8<Bits>(
              raw_w, reinterpret_cast<threadgroup char4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MoeLinStride + half_k * 32u));
        }
        if (tid < BM_TG) {
          const uint st = tg_mem.tg_tok[tid];
          tg_mem.kloop.Sa_s[tid] = (st != 0xFFFFFFFFu) ? sa[ulong(g) * p.mp + st] : 0.0f;
          tg_mem.kloop.Ra_s[tid] = (st != 0xFFFFFFFFu) ? ra[ulong(g) * p.mp + st] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint m_off = (sg >> 1u) * ((tg_mem.tg_meta[2] <= 32u) ? 16u : 32u);
        const uint sg_n = sg & 1u;
        const uint qid = lane >> 2u;
        const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
        const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
        auto b0 = tB.slice(0, int(sg_n * 32u));
        auto b1 = tB.slice(32, int(sg_n * 32u));
        #pragma clang loop unroll(full)
        for (uint mm = 0u; mm < 2u; ++mm) {
          auto a0 = tA.slice(0, int(m_off + mm * 16u));
          auto a1 = tA.slice(32, int(m_off + mm * 16u));
          auto ci = op_i8.template get_destination_cooperative_tensor<
              remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) ci[i] = 0;
          op_i8.run(a0, b0, ci);
          op_i8.run(a1, b1, ci);

          const uint r0 = m_off + mm * 16u + fm;
          const float sa0 = tg_mem.kloop.Sa_s[r0], sa1 = tg_mem.kloop.Sa_s[r0 + 8u];
          const float ra0 = tg_mem.kloop.Ra_s[r0], ra1 = tg_mem.kloop.Ra_s[r0 + 8u];
          const uint col_sb = sg_n * 32u + fn;
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            float swg, bwg, swu, bwu;
            w4a8_moe_decode_sb<Bits>(tg_mem.kloop.Sb_s[col_sb + c], swg, bwg);
            w4a8_moe_decode_sb<Bits>(tg_mem.kloop.Sb_s[col_sb + 16u + c], swu, bwu);
            acc_f32[mm][0u + c]  = fma(sa0, fma(swg, float(ci[0u + c]),  bwg * ra0), acc_f32[mm][0u + c]);
            acc_f32[mm][4u + c]  = fma(sa1, fma(swg, float(ci[4u + c]),  bwg * ra1), acc_f32[mm][4u + c]);
            acc_f32[mm][8u + c]  = fma(sa0, fma(swu, float(ci[8u + c]),  bwu * ra0), acc_f32[mm][8u + c]);
            acc_f32[mm][12u + c] = fma(sa1, fma(swu, float(ci[12u + c]), bwu * ra1), acc_f32[mm][12u + c]);
          }
        }
      }

      const uint sg_n = sg & 1u;
      const uint qid = lane >> 2u;
      const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
      const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
      const uint rb_cur = tg_mem.tg_meta[1];
      const uint vr = tg_mem.tg_meta[2];
      const uint n0_base = tg_mem.tg_meta[3];
      const bool compact32 = (vr <= 32u);
      const uint m_off = (sg >> 1u) * (compact32 ? 16u : 32u);
      const uint n0_sg = n0_base + hn * 32u + sg_n * 16u;

      if constexpr (!OutputA8) {
        (void)mid_qa; (void)mid_sa; (void)mid_ra;
        const uint rb_8c = (vr + 7u) & ~7u;
        if (m_off < vr && n0_sg + fn < p.n) {
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            if ((mm == 0u || !compact32) && (m_off + mm * 16u < vr)) {
              #pragma clang loop unroll(full)
              for (uint h = 0u; h < 2u; ++h) {
                const uint rm = m_off + mm * 16u + fm + h * 8u;
                if (rm < vr) {
                  const uint e = h * 4u;
                  bfloat4 out_b;
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    out_b[j] = bfloat(nax_silu_f32(float(bfloat(acc_f32[mm][e + j]))) *
                                      float(bfloat(acc_f32[mm][8u + e + j])));
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
            if (mm == 0u || !compact32) {
              #pragma clang loop unroll(full)
              for (uint h = 0u; h < 2u; ++h) {
                const uint rm = m_off + mm * 16u + fm + h * 8u;
                if (rm < vr) {
                  bfloat4 out_b = bfloat4(bfloat(0.0f));
                  if (n0_sg + fn < p.n) {
                    const uint e = h * 4u;
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 4u; ++j) {
                      out_b[j] = (n0_sg + fn + j < p.n)
                          ? bfloat(nax_silu_f32(float(bfloat(acc_f32[mm][e + j]))) *
                                   float(bfloat(acc_f32[mm][8u + e + j])))
                          : bfloat(0.0f);
                    }
                  }
                  *reinterpret_cast<device bfloat4 *>(
                      mid_qa + ulong(rb_cur + rm) * p.n + stash_col_off) = out_b;
                }
              }
            }
          }
        } else {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            if (mm == 0u || !compact32) {
              #pragma clang loop unroll(full)
              for (uint h = 0u; h < 2u; ++h) {
                const uint rm = m_off + mm * 16u + fm + h * 8u;
                bfloat4 s0 = bfloat4(bfloat(0.0f));
                bfloat4 s1 = bfloat4(bfloat(0.0f));
                if (rm < vr) {
                  s0 = *reinterpret_cast<device const bfloat4 *>(
                      mid_qa + ulong(rb_cur + rm) * p.n + stash_col_off);
                  if (n0_sg + fn < p.n) {
                    const uint e = h * 4u;
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 4u; ++j) {
                      s1[j] = (n0_sg + fn + j < p.n)
                          ? bfloat(nax_silu_f32(float(bfloat(acc_f32[mm][e + j]))) *
                                   float(bfloat(acc_f32[mm][8u + e + j])))
                          : bfloat(0.0f);
                    }
                  }
                }
                *reinterpret_cast<threadgroup bfloat4 *>(
                    tg_mem.SwiBuf + rm * kW4A8MoeEpiStride + sg_n * 16u + fn) = s0;
                *reinterpret_cast<threadgroup bfloat4 *>(
                    tg_mem.SwiBuf + rm * kW4A8MoeEpiStride + 32u + sg_n * 16u + fn) = s1;
              }
            }
          }
        }
      }
    }

    if constexpr (OutputA8) {
      const uint rb_cur = tg_mem.tg_meta[1];
      const uint vr = tg_mem.tg_meta[2];
      const uint n0_base = tg_mem.tg_meta[3];
      const uint rb_8c = (vr + 7u) & ~7u;
      const uint mid_mp = p.reserved & 0x7FFFFFFFu;
      const bool apply_fwht64 = ((p.reserved >> 31u) == 0u);
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      w4a8_moe_swiglu_epilogue_a8_phase<16u>(
          tg_mem.SwiBuf, rb_cur, 0u, vr, rb_8c, n0_base,
          mid_qa, mid_sa, mid_ra, p.n, mid_mp, apply_fwht64, lane, sg);
    }
    return;
  }

  // Portable Apple9 fallback (zero stack allocas: 4 rows/pass in float4 + 16-row SwiBuf Mode B)
  const uint tid = sg * 32u + lane;
  const uint groups = p.k >> 6u;
  const uint rb_8ceil = (valid_rows + 7u) & ~7u;
  const uint w_row = tid >> 1u;
  const uint half_k = tid & 1u;
  const bool is_up = ((sg & 1u) != 0u);
  const uint w_col_in32 = (sg >> 1u) * 16u + (lane >> 1u);
  threadgroup char4 *w_dst4 =
      reinterpret_cast<threadgroup char4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MoeLinStride + half_k * 32u);
  const uint a_col16 = (tid & 3u) * 16u;
  const uint a_r0 = tid >> 2u;
  const uint a_r1 = a_r0 + 32u;
  device const uchar *wc = is_up ? u_codes : g_codes;
  device const uchar *wh = is_up ? u_high : g_high;
  device const bfloat *ws = is_up ? u_scales : g_scales;
  device const bfloat *wb = is_up ? u_biases : g_biases;
  const uint c_l = tid & 31u;
  const uint stripe4 = (tid >> 5u) * 4u;
  const uint ws_g_row = ((c_l >> 4u) * 32u) + (c_l & 15u);
  const uint ws_u_row = ws_g_row + 16u;

  for (uint r4 = 0u; r4 < 4u; ++r4) {
    bfloat4 s0 = bfloat4(bfloat(0.0f));
    bfloat4 s1 = bfloat4(bfloat(0.0f));
    for (uint hn = 0u; hn < kPasses; ++hn) {
      const uint tg_n0 = tg_n0_base + hn * 32u;
      const uint w_col = tg_n0 + w_col_in32;
      const bool w_live = (w_col < p.n);
      const ulong w_exp_row = ulong(expert) * ulong(p.n) + ulong(w_col);
      const uint col = tg_n0 + c_l;
      const bool col_live = (col < p.n);
      float4 g_acc = float4(0.0f);
      float4 u_acc = float4(0.0f);

      for (uint g = 0u; g < groups; ++g) {
        const uint k0 = g * kW4A8MoeBK;
        const uint t0 = tg_mem.tg_tok[a_r0];
        const uint t1 = tg_mem.tg_tok[a_r1];
        const uint4 qa0 = (t0 != 0xFFFFFFFFu)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(t0) * p.k + k0 + a_col16)
            : uint4(0u);
        const uint4 qa1 = (t1 != 0xFFFFFFFFu)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(t1) * p.k + k0 + a_col16)
            : uint4(0u);
        const W4A8MoeAffRaw32 raw_w = w4a8_moe_fetch_affine_32<Bits>(
            wc, wh, w_exp_row, w_live, p.padded_k, k0 + half_k * 32u);
        const uint sb_v = (half_k == 0u)
            ? w4a8_moe_fetch_sb_packed<GroupSize>(ws, wb, w_exp_row, w_live, p.padded_k, g)
            : 0u;
        float sa_v = 0.0f, ra_v = 0.0f;
        if (tid < BM_TG) {
          const uint st = tg_mem.tg_tok[tid];
          if (st != 0xFFFFFFFFu) {
            sa_v = sa[ulong(g) * p.mp + st];
            ra_v = ra[ulong(g) * p.mp + st];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r0 * kW4A8MoeLinStride + a_col16) = qa0;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r1 * kW4A8MoeLinStride + a_col16) = qa1;
        w4a8_moe_unpack_affine_32_i8<Bits>(raw_w, w_dst4);
        if (half_k == 0u) tg_mem.kloop.Sb_s[w_row] = sb_v;
        if (tid < BM_TG) { tg_mem.kloop.Sa_s[tid] = sa_v; tg_mem.kloop.Ra_s[tid] = ra_v; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (col_live && r4 * 16u + stripe4 < valid_rows) {
          float sw_g, bw_g, sw_u, bw_u;
          w4a8_moe_decode_sb<Bits>(tg_mem.kloop.Sb_s[ws_g_row], sw_g, bw_g);
          w4a8_moe_decode_sb<Bits>(tg_mem.kloop.Sb_s[ws_u_row], sw_u, bw_u);
          for (uint rr = 0u; rr < 4u; ++rr) {
            const uint r_l = r4 * 16u + stripe4 + rr;
            if (r_l < valid_rows) {
              int32_t dg = 0, du = 0;
              for (uint k = 0u; k < kW4A8MoeBK; ++k) {
                const int32_t av = int32_t(tg_mem.kloop.As_i8[r_l * kW4A8MoeLinStride + k]);
                dg += av * int32_t(tg_mem.kloop.Ws_i8[ws_g_row * kW4A8MoeLinStride + k]);
                du += av * int32_t(tg_mem.kloop.Ws_i8[ws_u_row * kW4A8MoeLinStride + k]);
              }
              const float sa_r = tg_mem.kloop.Sa_s[r_l];
              const float ra_r = tg_mem.kloop.Ra_s[r_l];
              g_acc[rr] = fma(sa_r, fma(sw_g, float(dg), bw_g * ra_r), g_acc[rr]);
              u_acc[rr] = fma(sa_r, fma(sw_u, float(du), bw_u * ra_r), u_acc[rr]);
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
        if constexpr (!OutputA8) {
          if (col_live && rm < rb_8ceil) {
            mid_bf16[ulong(rb + rm) * p.n + col] = sv;
          }
        }
      }
      if (hn == 0u) s0 = sv4; else s1 = sv4;
    }
    if constexpr (OutputA8) {
      const uint mid_mp = p.reserved & 0x7FFFFFFFu;
      const bool apply_fwht64 = ((p.reserved >> 31u) == 0u);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint rr = 0u; rr < 4u; ++rr) {
        tg_mem.SwiBuf[(stripe4 + rr) * kW4A8MoeEpiStride + c_l] = s0[rr];
        tg_mem.SwiBuf[(stripe4 + rr) * kW4A8MoeEpiStride + 32u + c_l] = s1[rr];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      w4a8_moe_swiglu_epilogue_a8_phase<4u>(
          tg_mem.SwiBuf, rb, r4 * 16u, valid_rows, rb_8ceil, tg_n0_base,
          mid_qa, mid_sa, mid_ra, p.n, mid_mp, apply_fwht64, lane, sg);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// ============================================================================
// MoE MXFP4 Compact TG-Staged W4A8 Fused Gate + Up + SwiGLU (BM=64, BK=64)
// 4 SGs = 128 threads (WM=2, WN=2, SM=32/compact32), 10,240 B (10.00 KiB <= 10.0 KiB) SRAM -> 3 TGs/core, 0 spills!
// ============================================================================
template <bool OutputA8>
inline void moe_mxfp4_w4a8_nax_swiglu_bm64_out64_impl(
    device const int8_t *qa,
    device const float *sa,
    device const uint *tile_assign,
    device const uint *tile_expert,
    device const uint *tile_valid_rows,
    device const uint *total_tiles,
    device const uchar *g_codes,
    device const uchar *g_scales,
    device const uchar *u_codes,
    device const uchar *u_scales,
    device bfloat *mid_bf16,
    device int8_t *mid_qa,
    device float *mid_sa,
    device float *mid_ra,
    constant W4A8MoeGemmParams &p,
    threadgroup W4A8MoeMxfp4SwiTgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM_TG = 64u;
  constexpr uint OUT_N_TG = OutputA8 ? 64u : 32u;
  constexpr uint kPasses = OutputA8 ? 2u : 1u;

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

  if (w4a8_moe_has_apple10_i8_layout()) {
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
        tg_mem.kloop.As_i8, dextents<int32_t, 2>(int(kW4A8MoeMxStride), 64));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
        tg_mem.kloop.Ws_i8, dextents<int32_t, 2>(int(kW4A8MoeMxStride), 64));

    const bool sg_is_up = ((sg & 1u) != 0u);
    const uint k_blocks = p.k >> 6u;

    // Fast path for valid_rows <= 32: half qa loads, half Sa loads, 16x32x32 shift-merge LUT!
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
          const uint k0 = kb * kW4A8MoeBK;
          const uint tid = sg * 32u + lane;
          const uint a_col16 = (lane & 3u) * 16u;
          const uint a_r0 = tid >> 2u;
          const uint col32 = lane & 1u;
          const uint w_row = sg * 16u + (lane >> 1u);
          const uint sub32_abs = kb * 2u + col32;

          const uint t0 = tg_mem.tg_tok[a_r0];
          const uint4 qa0 = (t0 != 0xFFFFFFFFu)
              ? *reinterpret_cast<device const uint4 *>(qa + ulong(t0) * p.k + k0 + a_col16)
              : uint4(0u);
          uint4 raw_w = uint4(0u);
          uchar2 ep = uchar2(0u, 0u);
          if (w_live) {
            const ulong c_off = w_exp_row * ulong(p.padded_k >> 1u) + ulong(sub32_abs) * 16ul;
            const ulong s_off = w_exp_row * ulong(p.padded_k >> 5u) + ulong(kb) * 2ul;
            if (!sg_is_up) {
              raw_w = *reinterpret_cast<device const uint4 *>(g_codes + c_off);
              ep = *reinterpret_cast<device const uchar2 *>(g_scales + s_off);
            } else {
              raw_w = *reinterpret_cast<device const uint4 *>(u_codes + c_off);
              ep = *reinterpret_cast<device const uchar2 *>(u_scales + s_off);
            }
          }
          float sa_v = 0.0f;
          if (tid < 32u) {
            const uint st = tg_mem.tg_tok[tid];
            if (st != 0xFFFFFFFFu) sa_v = sa[ulong(kb) * p.mp + st];
          }
          const int de = (w_live && ep.x != 255u && ep.y != 255u)
              ? (int(ep.y) - int(ep.x)) : (w_live ? 255 : 0);
          const bool row_fast = (abs(de) <= 3);
          const uint sh = row_fast
              ? ((col32 == 0u) ? ((de < 0) ? uint(-de) : 0u) : ((de > 0) ? uint(de) : 0u))
              : 0u;
          uint4 wu0, wu1;
          w4a8_moe_unpack_mxfp4_lut_sh_u4(raw_w, sh, wu0, wu1);

          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (tid < 2u) tg_mem.kloop.slow_flag[tid] = 0u;
          w4a8_moe_store_as_mx_u4(tg_mem.kloop.As_i8, a_r0, a_col16, qa0);
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MoeMxStride + col32 * 32u + 0u) = wu0;
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MoeMxStride + col32 * 32u + 16u) = wu1;
          if (col32 == 0u) {
            if (row_fast) {
              const float sw = w_live ? (0.5f * w4a8_moe_e8m0_safe(min(ep.x, ep.y))) : 0.0f;
              tg_mem.kloop.Sw0_s[w_row] = sw;
              tg_mem.kloop.Sw1_s[w_row] = sw;
            } else {
              tg_mem.kloop.Sw0_s[w_row] = w_live ? (0.5f * w4a8_moe_e8m0_safe(ep.x)) : 0.0f;
              tg_mem.kloop.Sw1_s[w_row] = w_live ? (0.5f * w4a8_moe_e8m0_safe(ep.y)) : 0.0f;
              tg_mem.kloop.slow_flag[w_row >> 5u] = 1u;
            }
          }
          if (tid < 32u) tg_mem.kloop.Sa_s[tid] = sa_v;
          threadgroup_barrier(mem_flags::mem_threadgroup);

          const uint m_off = (sg >> 1u) * 16u;
          const uint sg_n = sg & 1u;
          const uint qid = lane >> 2u;
          const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
          const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
          const uint sw_base = sg_n * 32u + fn;
          const float sa0 = tg_mem.kloop.Sa_s[m_off + fm];
          const float sa1 = tg_mem.kloop.Sa_s[m_off + 8u + fm];
          auto a0 = tA.slice(0, int(m_off));
          auto b0 = tB.slice(0, int(sg_n * 32u));
          auto ci = op16_m.template get_destination_cooperative_tensor<
              remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
          if (tg_mem.kloop.slow_flag[sg_n] == 0u) {
            auto a1 = tA.slice(32, int(m_off));
            auto b1 = tB.slice(32, int(sg_n * 32u));
            op16_m.run(a0, b0, ci);
            op16_a.run(a1, b1, ci);
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
              threadgroup const float *swp = (hk == 0u) ? tg_mem.kloop.Sw0_s : tg_mem.kloop.Sw1_s;
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

        const uint sg_n = sg & 1u;
        const uint qid = lane >> 2u;
        const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
        const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
        const uint rb_cur = tg_mem.tg_meta[1];
        const uint vr = tg_mem.tg_meta[2];
        const uint n0_base = tg_mem.tg_meta[3];
        const uint m_off = (sg >> 1u) * 16u;
        const uint n0_sg = n0_base + hn * 32u + sg_n * 16u;

        if constexpr (!OutputA8) {
          (void)mid_qa; (void)mid_sa; (void)mid_ra;
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
              *reinterpret_cast<threadgroup bfloat4 *>(tg_mem.SwiBuf + rm * kW4A8MoeEpiStride + sg_n * 16u + fn) = s0;
              *reinterpret_cast<threadgroup bfloat4 *>(tg_mem.SwiBuf + rm * kW4A8MoeEpiStride + 32u + sg_n * 16u + fn) = s1;
            }
          }
        }
      }
      if constexpr (OutputA8) {
        const uint rb_cur = tg_mem.tg_meta[1];
        const uint vr = tg_mem.tg_meta[2];
        const uint n0_base = tg_mem.tg_meta[3];
        const uint rb_8c = (vr + 7u) & ~7u;
        const uint mid_mp = p.reserved & 0x7FFFFFFFu;
        const bool apply_fwht64 = ((p.reserved >> 31u) == 0u);
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        w4a8_moe_swiglu_epilogue_a8_phase<8u>(
            tg_mem.SwiBuf, rb_cur, 0u, vr, rb_8c, n0_base,
            mid_qa, mid_sa, mid_ra, p.n, mid_mp, apply_fwht64, lane, sg);
      }
      return;
    }

    #pragma clang loop unroll(full)
    for (uint hn = 0u; hn < kPasses; ++hn) {
      const uint w_col = tg_mem.tg_meta[3] + hn * 32u + (sg >> 1u) * 16u + (lane >> 1u);
      const bool w_live = (w_col < p.n);
      const ulong w_exp_row = ulong(tg_mem.tg_meta[0]) * ulong(p.n) + ulong(w_col);

      float acc_f32[32];
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 32u; ++i) acc_f32[i] = 0.0f;

      for (uint kb = 0u; kb < k_blocks; ++kb) {
        const uint k0 = kb * kW4A8MoeBK;
        const uint tid = sg * 32u + lane;
        const uint a_col16 = (lane & 3u) * 16u;
        const uint a_r0 = tid >> 2u;
        const uint a_r1 = a_r0 + 32u;
        const uint col32 = lane & 1u;
        const uint w_row = sg * 16u + (lane >> 1u);
        const uint sub32_abs = kb * 2u + col32;

        const uint t0 = tg_mem.tg_tok[a_r0];
        const uint t1 = tg_mem.tg_tok[a_r1];
        const uint4 qa0 = (t0 != 0xFFFFFFFFu)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(t0) * p.k + k0 + a_col16)
            : uint4(0u);
        const uint4 qa1 = (t1 != 0xFFFFFFFFu)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(t1) * p.k + k0 + a_col16)
            : uint4(0u);
        uint4 raw_w = uint4(0u);
        uchar2 ep = uchar2(0u, 0u);
        if (w_live) {
          const ulong c_off = w_exp_row * ulong(p.padded_k >> 1u) + ulong(sub32_abs) * 16ul;
          const ulong s_off = w_exp_row * ulong(p.padded_k >> 5u) + ulong(kb) * 2ul;
          if (!sg_is_up) {
            raw_w = *reinterpret_cast<device const uint4 *>(g_codes + c_off);
            ep = *reinterpret_cast<device const uchar2 *>(g_scales + s_off);
          } else {
            raw_w = *reinterpret_cast<device const uint4 *>(u_codes + c_off);
            ep = *reinterpret_cast<device const uchar2 *>(u_scales + s_off);
          }
        }
        float sa_v = 0.0f;
        if (tid < BM_TG) {
          const uint st = tg_mem.tg_tok[tid];
          if (st != 0xFFFFFFFFu) sa_v = sa[ulong(kb) * p.mp + st];
        }
        const int de = (w_live && ep.x != 255u && ep.y != 255u)
            ? (int(ep.y) - int(ep.x)) : (w_live ? 255 : 0);
        const bool row_fast = (abs(de) <= 3);
        const uint sh = row_fast
            ? ((col32 == 0u) ? ((de < 0) ? uint(-de) : 0u) : ((de > 0) ? uint(de) : 0u))
            : 0u;
        uint4 wu0, wu1;
        w4a8_moe_unpack_mxfp4_lut_sh_u4(raw_w, sh, wu0, wu1);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 2u) tg_mem.kloop.slow_flag[tid] = 0u;
        w4a8_moe_store_as_mx_u4(tg_mem.kloop.As_i8, a_r0, a_col16, qa0);
        w4a8_moe_store_as_mx_u4(tg_mem.kloop.As_i8, a_r1, a_col16, qa1);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MoeMxStride + col32 * 32u + 0u) = wu0;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MoeMxStride + col32 * 32u + 16u) = wu1;
        if (col32 == 0u) {
          if (row_fast) {
            const float sw = w_live ? (0.5f * w4a8_moe_e8m0_safe(min(ep.x, ep.y))) : 0.0f;
            tg_mem.kloop.Sw0_s[w_row] = sw;
            tg_mem.kloop.Sw1_s[w_row] = sw;
          } else {
            tg_mem.kloop.Sw0_s[w_row] = w_live ? (0.5f * w4a8_moe_e8m0_safe(ep.x)) : 0.0f;
            tg_mem.kloop.Sw1_s[w_row] = w_live ? (0.5f * w4a8_moe_e8m0_safe(ep.y)) : 0.0f;
            tg_mem.kloop.slow_flag[w_row >> 5u] = 1u;
          }
        }
        if (tid < BM_TG) tg_mem.kloop.Sa_s[tid] = sa_v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint m_off = (sg >> 1u) * 32u;
        const uint sg_n = sg & 1u;
        const uint qid = lane >> 2u;
        const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
        const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
        const uint sw_base = sg_n * 32u + fn;
        const float sa0 = tg_mem.kloop.Sa_s[m_off + fm];
        const float sa1 = tg_mem.kloop.Sa_s[m_off + 8u + fm];
        const float sa2 = tg_mem.kloop.Sa_s[m_off + 16u + fm];
        const float sa3 = tg_mem.kloop.Sa_s[m_off + 24u + fm];
        auto a0 = tA.slice(0, int(m_off));
        auto b0 = tB.slice(0, int(sg_n * 32u));
        auto ci = op32_m.template get_destination_cooperative_tensor<
            remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
        if (tg_mem.kloop.slow_flag[sg_n] == 0u) {
          auto a1 = tA.slice(32, int(m_off));
          auto b1 = tB.slice(32, int(sg_n * 32u));
          op32_m.run(a0, b0, ci);
          op32_a.run(a1, b1, ci);
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
            threadgroup const float *swp = (hk == 0u) ? tg_mem.kloop.Sw0_s : tg_mem.kloop.Sw1_s;
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

      const uint sg_n = sg & 1u;
      const uint qid = lane >> 2u;
      const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
      const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
      const uint rb_cur = tg_mem.tg_meta[1];
      const uint vr = tg_mem.tg_meta[2];
      const uint n0_base = tg_mem.tg_meta[3];
      const uint m_off = (sg >> 1u) * 32u;
      const uint n0_sg = n0_base + hn * 32u + sg_n * 16u;

      if constexpr (!OutputA8) {
        (void)mid_qa; (void)mid_sa; (void)mid_ra;
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
                  tg_mem.SwiBuf + rm * kW4A8MoeEpiStride + sg_n * 16u + fn) = s0;
              *reinterpret_cast<threadgroup bfloat4 *>(
                  tg_mem.SwiBuf + rm * kW4A8MoeEpiStride + 32u + sg_n * 16u + fn) = s1;
            }
          }
        }
      }
    }

    if constexpr (OutputA8) {
      const uint rb_cur = tg_mem.tg_meta[1];
      const uint vr = tg_mem.tg_meta[2];
      const uint n0_base = tg_mem.tg_meta[3];
      const uint rb_8c = (vr + 7u) & ~7u;
      const uint mid_mp = p.reserved & 0x7FFFFFFFu;
      const bool apply_fwht64 = ((p.reserved >> 31u) == 0u);
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      w4a8_moe_swiglu_epilogue_a8_phase<16u>(
          tg_mem.SwiBuf, rb_cur, 0u, vr, rb_8c, n0_base,
          mid_qa, mid_sa, mid_ra, p.n, mid_mp, apply_fwht64, lane, sg);
    }
    return;
  }

  // Portable Apple9 fallback (zero stack allocas: 4 rows/pass in float4 + 16-row SwiBuf Mode B)
  const uint tid = sg * 32u + lane;
  const uint k_blocks = p.k >> 6u;
  const ulong c_stride = ulong(p.padded_k >> 1u);
  const ulong s_stride = ulong(p.padded_k >> 5u);
  const uint rb_8ceil = (valid_rows + 7u) & ~7u;
  const uint w_row = tid >> 1u;
  const uint col32 = tid & 1u;
  const bool is_up = ((sg & 1u) != 0u);
  const uint w_col_in32 = (sg >> 1u) * 16u + (lane >> 1u);
  threadgroup char4 *w_dst4 =
      reinterpret_cast<threadgroup char4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MoeMxStride + col32 * 32u);
  const uint a_col16 = (tid & 3u) * 16u;
  const uint a_r0 = tid >> 2u;
  const uint a_r1 = a_r0 + 32u;
  const uint c_l = tid & 31u;
  const uint stripe4 = (tid >> 5u) * 4u;
  const uint ws_g_row = ((c_l >> 4u) * 32u) + (c_l & 15u);
  const uint ws_u_row = ws_g_row + 16u;

  for (uint r4 = 0u; r4 < 4u; ++r4) {
    bfloat4 s0 = bfloat4(bfloat(0.0f));
    bfloat4 s1 = bfloat4(bfloat(0.0f));
    for (uint hn = 0u; hn < kPasses; ++hn) {
      const uint tg_n0 = tg_n0_base + hn * 32u;
      const uint w_col = tg_n0 + w_col_in32;
      const bool w_live = (w_col < p.n);
      const ulong w_exp_row = ulong(expert) * ulong(p.n) + ulong(w_col);
      device const uchar *wc_row = w_live ? ((is_up ? u_codes : g_codes) + w_exp_row * c_stride) : nullptr;
      device const uchar *ws_row = w_live ? ((is_up ? u_scales : g_scales) + w_exp_row * s_stride) : nullptr;
      const uint col = tg_n0 + c_l;
      const bool col_live = (col < p.n);
      float4 g_acc = float4(0.0f);
      float4 u_acc = float4(0.0f);

      for (uint kb = 0u; kb < k_blocks; ++kb) {
        const uint k0 = kb * kW4A8MoeBK;
        const uint sub32_abs = kb * 2u + col32;
        const uint t0 = tg_mem.tg_tok[a_r0];
        const uint t1 = tg_mem.tg_tok[a_r1];
        const uint4 qa0 = (t0 != 0xFFFFFFFFu)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(t0) * p.k + k0 + a_col16)
            : uint4(0u);
        const uint4 qa1 = (t1 != 0xFFFFFFFFu)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(t1) * p.k + k0 + a_col16)
            : uint4(0u);
        const uint4 raw_w = wc_row
            ? *reinterpret_cast<device const uint4 *>(wc_row + ulong(sub32_abs) * 16ul)
            : uint4(0u);
        const float sw_v = ws_row
            ? (0.5f * e8m0_to_float(ws_row[sub32_abs]))
            : 0.0f;
        float sa_v = 0.0f;
        if (tid < BM_TG) {
          const uint st = tg_mem.tg_tok[tid];
          if (st != 0xFFFFFFFFu) sa_v = sa[ulong(kb) * p.mp + st];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        w4a8_moe_store_as_mx_u4(tg_mem.kloop.As_i8, a_r0, a_col16, qa0);
        w4a8_moe_store_as_mx_u4(tg_mem.kloop.As_i8, a_r1, a_col16, qa1);
        w4a8_moe_unpack_mxfp4_32_i8(raw_w, w_dst4);
        if (col32 == 0u) tg_mem.kloop.Sw0_s[w_row] = sw_v; else tg_mem.kloop.Sw1_s[w_row] = sw_v;
        if (tid < BM_TG) tg_mem.kloop.Sa_s[tid] = sa_v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (col_live && r4 * 16u + stripe4 < valid_rows) {
          const float swg0 = tg_mem.kloop.Sw0_s[ws_g_row], swg1 = tg_mem.kloop.Sw1_s[ws_g_row];
          const float swu0 = tg_mem.kloop.Sw0_s[ws_u_row], swu1 = tg_mem.kloop.Sw1_s[ws_u_row];
          for (uint rr = 0u; rr < 4u; ++rr) {
            const uint r_l = r4 * 16u + stripe4 + rr;
            if (r_l < valid_rows) {
              int32_t dg0 = 0, dg1 = 0, du0 = 0, du1 = 0;
              const uint a_off = r_l * kW4A8MoeMxStride;
              const uint wg_off = ws_g_row * kW4A8MoeMxStride;
              const uint wu_off = ws_u_row * kW4A8MoeMxStride;
              for (uint k = 0u; k < 32u; ++k) {
                const int32_t a0 = int32_t(tg_mem.kloop.As_i8[a_off + k]);
                const int32_t a1 = int32_t(tg_mem.kloop.As_i8[a_off + 32u + k]);
                dg0 += a0 * int32_t(tg_mem.kloop.Ws_i8[wg_off + k]);
                dg1 += a1 * int32_t(tg_mem.kloop.Ws_i8[wg_off + 32u + k]);
                du0 += a0 * int32_t(tg_mem.kloop.Ws_i8[wu_off + k]);
                du1 += a1 * int32_t(tg_mem.kloop.Ws_i8[wu_off + 32u + k]);
              }
              const float sa_r = tg_mem.kloop.Sa_s[r_l];
              g_acc[rr] = fma(sa_r, fma(swg0, float(dg0), swg1 * float(dg1)), g_acc[rr]);
              u_acc[rr] = fma(sa_r, fma(swu0, float(du0), swu1 * float(du1)), u_acc[rr]);
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
        if constexpr (!OutputA8) {
          if (col_live && rm < rb_8ceil) {
            mid_bf16[ulong(rb + rm) * p.n + col] = sv;
          }
        }
      }
      if (hn == 0u) s0 = sv4; else s1 = sv4;
    }
    if constexpr (OutputA8) {
      const uint mid_mp = p.reserved & 0x7FFFFFFFu;
      const bool apply_fwht64 = ((p.reserved >> 31u) == 0u);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint rr = 0u; rr < 4u; ++rr) {
        tg_mem.SwiBuf[(stripe4 + rr) * kW4A8MoeEpiStride + c_l] = s0[rr];
        tg_mem.SwiBuf[(stripe4 + rr) * kW4A8MoeEpiStride + 32u + c_l] = s1[rr];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      w4a8_moe_swiglu_epilogue_a8_phase<4u>(
          tg_mem.SwiBuf, rb, r4 * 16u, valid_rows, rb_8ceil, tg_n0_base,
          mid_qa, mid_sa, mid_ra, p.n, mid_mp, apply_fwht64, lane, sg);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// ============================================================================
// WP1 Part B: MoE Compact TG-Staged W4A8 Down Scatter (BM=64, BN=64, 4 SGs = 128 threads)
// Affine: 8,960 B (8.75 KiB), MXFP4: 9,984 B (9.75 KiB) -> 3 TGs/core, 0 spills!
// ============================================================================
template <uint Bits, uint GroupSize>
inline void moe_affine_w4a8_nax_down_scatter_bm64_bn64_impl(
    device const int8_t *mid_qa,
    device const float *mid_sa,
    device const float *mid_ra,
    device const uint *tile_expert,
    device const uint *tile_valid_rows,
    device const uint *tile_assign,
    device const uint *total_tiles,
    device const bfloat *route_scores,
    device const uchar *d_codes,
    device const uchar *d_high,
    device const bfloat *d_scales,
    device const bfloat *d_biases,
    device bfloat *expert_output,
    constant W4A8MoeGemmParams &p,
    threadgroup W4A8MoeDownAffineTgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM_TG = 64u;
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

  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const bool compact32 = (valid_rows <= 32u);
  const uint m_off = sg_m * (compact32 ? 16u : 32u);
  const uint n0_sg = tg_n0 + sg_n * 32u;
  const bool sg_active = (m_off < valid_rows) && (n0_sg < p.n);
  const uint groups = p.k / kW4A8MoeBK;

  const uint w_row = tid >> 1u; // 0..63
  const uint half_k = tid & 1u; // 0 or 1 (32 weights/thr)
  const uint w_col = tg_n0 + w_row;
  const bool w_live = (w_col < p.n);
  const ulong w_exp_row = ulong(expert) * ulong(p.n) + ulong(w_col);
  threadgroup char4 *w_dst4 =
      reinterpret_cast<threadgroup char4 *>(tg_mem.Ws_i8 + w_row * kW4A8MoeLinStride + half_k * 32u);

  const uint a_col16 = (tid & 3u) * 16u;
  const uint a_r0 = tid >> 2u;
  const uint a_r1 = a_r0 + 32u;

  if (w4a8_moe_has_apple10_i8_layout()) {
    constexpr auto desc_i8 = matmul2d_descriptor(
        16, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_i8, execution_simdgroup> op_i8;
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
        tg_mem.As_i8, dextents<int32_t, 2>(int(kW4A8MoeLinStride), 64));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
        tg_mem.Ws_i8, dextents<int32_t, 2>(int(kW4A8MoeLinStride), 64));

    float acc_f32[2][16];
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm) {
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) acc_f32[mm][i] = 0.0f;
    }
    const uint qid = lane >> 2u;
    const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
    const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

    for (uint g = 0u; g < groups; ++g) {
      const uint k0 = g * kW4A8MoeBK;
      const uint4 qa0 = (a_r0 < valid_rows)
          ? *reinterpret_cast<device const uint4 *>(mid_qa + ulong(rb + a_r0) * p.k + k0 + a_col16)
          : uint4(0u);
      const uint4 qa1 = (a_r1 < valid_rows)
          ? *reinterpret_cast<device const uint4 *>(mid_qa + ulong(rb + a_r1) * p.k + k0 + a_col16)
          : uint4(0u);
      const W4A8MoeAffRaw32 raw_w = w4a8_moe_fetch_affine_32<Bits>(
          d_codes, d_high, w_exp_row, w_live, p.padded_k, k0 + half_k * 32u);
      const uint sb_v = (half_k == 0u)
          ? w4a8_moe_fetch_sb_packed<GroupSize>(d_scales, d_biases, w_exp_row, w_live, p.padded_k, g)
          : 0u;
      float sa_v = 0.0f, ra_v = 0.0f;
      if (tid < BM_TG && tid < valid_rows) {
        sa_v = mid_sa[ulong(g) * p.mp + rb + tid];
        ra_v = mid_ra[ulong(g) * p.mp + rb + tid];
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + a_r0 * kW4A8MoeLinStride + a_col16) = qa0;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + a_r1 * kW4A8MoeLinStride + a_col16) = qa1;
      w4a8_moe_unpack_affine_32_i8<Bits>(raw_w, w_dst4);
      if (half_k == 0u) tg_mem.Sb_s[w_row] = sb_v;
      if (tid < BM_TG) {
        tg_mem.Sa_s[tid] = sa_v;
        tg_mem.Ra_s[tid] = ra_v;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_active) {
        auto b0 = tB.slice(0, int(sg_n * 32u));
        auto b1 = tB.slice(32, int(sg_n * 32u));
        #pragma clang loop unroll(full)
        for (uint mm = 0u; mm < 2u; ++mm) {
          if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
            auto a0 = tA.slice(0, int(m_off + mm * 16u));
            auto a1 = tA.slice(32, int(m_off + mm * 16u));
            auto ci = op_i8.template get_destination_cooperative_tensor<
                remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
            #pragma clang loop unroll(full)
            for (ushort i = 0u; i < 16u; ++i) ci[i] = 0;
            op_i8.run(a0, b0, ci);
            op_i8.run(a1, b1, ci);

            const uint r0 = m_off + mm * 16u + fm;
            const uint r1 = r0 + 8u;
            const float sa0 = tg_mem.Sa_s[r0], sa1 = tg_mem.Sa_s[r1];
            const float ra0 = tg_mem.Ra_s[r0], ra1 = tg_mem.Ra_s[r1];
            #pragma clang loop unroll(full)
            for (uint u = 0u; u < 2u; ++u) {
              const uint c_base = sg_n * 32u + u * 16u + fn;
              #pragma clang loop unroll(full)
              for (uint c = 0u; c < 4u; ++c) {
                float sw, bw;
                w4a8_moe_decode_sb<Bits>(tg_mem.Sb_s[c_base + c], sw, bw);
                const uint i0 = u * 8u + c;
                const uint i1 = u * 8u + 4u + c;
                acc_f32[mm][i0] = fma(sa0, fma(sw, float(ci[i0]), bw * ra0), acc_f32[mm][i0]);
                acc_f32[mm][i1] = fma(sa1, fma(sw, float(ci[i1]), bw * ra1), acc_f32[mm][i1]);
              }
            }
          }
        }
      }
    }

    if (sg_active) {
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        if ((mm == 0u || !compact32) && (m_off + mm * 16u < valid_rows)) {
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
                  const uint e = u * 8u + h * 4u;
                  const bfloat4 out_v = bfloat4(
                      bfloat(acc_f32[mm][e + 0u] * score),
                      bfloat(acc_f32[mm][e + 1u] * score),
                      bfloat(acc_f32[mm][e + 2u] * score),
                      bfloat(acc_f32[mm][e + 3u] * score));
                  *reinterpret_cast<device bfloat4 *>(
                      expert_output + ulong(assign) * p.n + col) = out_v;
                }
              }
            }
          }
        }
      }
    }
    return;
  }

  // Portable Apple9 fallback (128 threads: 64 cols x 2 row-stripes of 32 rows)
  const uint c_l = tid & 63u;
  const uint r_base = (tid >> 6u) * 32u;
  const uint col = tg_n0 + c_l;
  const bool col_live = (col < p.n);
  float acc_a9[32];
  for (uint rr = 0u; rr < 32u; ++rr) acc_a9[rr] = 0.0f;

  for (uint g = 0u; g < groups; ++g) {
    const uint k0 = g * kW4A8MoeBK;
    const uint4 qa0 = (a_r0 < valid_rows)
        ? *reinterpret_cast<device const uint4 *>(mid_qa + ulong(rb + a_r0) * p.k + k0 + a_col16)
        : uint4(0u);
    const uint4 qa1 = (a_r1 < valid_rows)
        ? *reinterpret_cast<device const uint4 *>(mid_qa + ulong(rb + a_r1) * p.k + k0 + a_col16)
        : uint4(0u);
    const W4A8MoeAffRaw32 raw_w = w4a8_moe_fetch_affine_32<Bits>(
        d_codes, d_high, w_exp_row, w_live, p.padded_k, k0 + half_k * 32u);
    const uint sb_v = (half_k == 0u)
        ? w4a8_moe_fetch_sb_packed<GroupSize>(d_scales, d_biases, w_exp_row, w_live, p.padded_k, g)
        : 0u;
    float sa_v = 0.0f, ra_v = 0.0f;
    if (tid < BM_TG && tid < valid_rows) {
      sa_v = mid_sa[ulong(g) * p.mp + rb + tid];
      ra_v = mid_ra[ulong(g) * p.mp + rb + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + a_r0 * kW4A8MoeLinStride + a_col16) = qa0;
    *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + a_r1 * kW4A8MoeLinStride + a_col16) = qa1;
    w4a8_moe_unpack_affine_32_i8<Bits>(raw_w, w_dst4);
    if (half_k == 0u) tg_mem.Sb_s[w_row] = sb_v;
    if (tid < BM_TG) { tg_mem.Sa_s[tid] = sa_v; tg_mem.Ra_s[tid] = ra_v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (col_live && r_base < valid_rows) {
      float sw, bw;
      w4a8_moe_decode_sb<Bits>(tg_mem.Sb_s[c_l], sw, bw);
      for (uint rr = 0u; rr < 32u; ++rr) {
        const uint r_l = r_base + rr;
        if (r_l < valid_rows) {
          int32_t dot = 0;
          for (uint k = 0u; k < kW4A8MoeBK; ++k) {
            dot += int32_t(tg_mem.As_i8[r_l * kW4A8MoeLinStride + k]) *
                   int32_t(tg_mem.Ws_i8[c_l * kW4A8MoeLinStride + k]);
          }
          acc_a9[rr] = fma(tg_mem.Sa_s[r_l], fma(sw, float(dot), bw * tg_mem.Ra_s[r_l]), acc_a9[rr]);
        }
      }
    }
  }
  if (col_live) {
    for (uint rr = 0u; rr < 32u; ++rr) {
      const uint rm = r_base + rr;
      if (rm < valid_rows) {
        const uint assign = tile_assign[rb + rm];
        const float score = float(route_scores[assign]);
        expert_output[ulong(assign) * p.n + col] = bfloat(acc_a9[rr] * score);
      }
    }
  }
}

inline void moe_mxfp4_w4a8_nax_down_scatter_bm64_bn64_impl(
    device const int8_t *mid_qa,
    device const float *mid_sa,
    device const uint *tile_expert,
    device const uint *tile_valid_rows,
    device const uint *tile_assign,
    device const uint *total_tiles,
    device const bfloat *route_scores,
    device const uchar *d_codes,
    device const uchar *d_scales,
    device bfloat *expert_output,
    constant W4A8MoeGemmParams &p,
    threadgroup W4A8MoeDownMxfp4TgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM_TG = 64u;
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

  if (w4a8_moe_has_apple10_i8_layout()) {
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
        tg_mem.As_i8, dextents<int32_t, 2>(int(kW4A8MoeMxStride), 64));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
        tg_mem.Ws_i8, dextents<int32_t, 2>(int(kW4A8MoeMxStride), 64));

    const bool compact32 = (valid_rows <= 32u);
    const uint m_off = (sg >> 1u) * (compact32 ? 16u : 32u);
    const uint k_blocks = p.k >> 6u;
    const uint w_row = tid >> 1u;
    const uint col32 = tid & 1u;
    const uint w_col = tg_n0 + w_row;
    const bool w_live = (w_col < p.n);
    const ulong w_exp_row = ulong(expert) * ulong(p.n) + ulong(w_col);
    const uint a_col16 = (tid & 3u) * 16u;
    const uint a_r0 = tid >> 2u;
    const uint a_r1 = a_r0 + 32u;
    const uint sg_n = sg & 1u;
    const uint n0_sg = tg_n0 + sg_n * 32u;
    const bool sg_active = (m_off < valid_rows) && (n0_sg < p.n);
    const uint qid = lane >> 2u;
    const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
    const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
    const uint sw_base = sg_n * 32u + fn;

    if (compact32) {
      float acc0[16];
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) acc0[i] = 0.0f;

      for (uint kb = 0u; kb < k_blocks; ++kb) {
        const uint k0 = kb * kW4A8MoeBK;
        const uint sub32_abs = kb * 2u + col32;
        const uint4 qa0 = (a_r0 < valid_rows)
            ? *reinterpret_cast<device const uint4 *>(mid_qa + ulong(rb + a_r0) * p.k + k0 + a_col16)
            : uint4(0u);
        const uint4 raw_w = w_live
            ? *reinterpret_cast<device const uint4 *>(
                  d_codes + w_exp_row * ulong(p.padded_k >> 1u) + ulong(sub32_abs) * 16ul)
            : uint4(0u);
        const uchar2 ep = w_live
            ? *reinterpret_cast<device const uchar2 *>(
                  d_scales + w_exp_row * ulong(p.padded_k >> 5u) + ulong(kb) * 2ul)
            : uchar2(0u, 0u);
        const float sa_v = (tid < 32u && tid < valid_rows) ? mid_sa[ulong(kb) * p.mp + rb + tid] : 0.0f;
        const int de = (w_live && ep.x != 255u && ep.y != 255u)
            ? (int(ep.y) - int(ep.x)) : (w_live ? 255 : 0);
        const bool row_fast = (abs(de) <= 3);
        const uint sh = row_fast
            ? ((col32 == 0u) ? ((de < 0) ? uint(-de) : 0u) : ((de > 0) ? uint(de) : 0u))
            : 0u;
        uint4 wu0, wu1;
        w4a8_moe_unpack_mxfp4_lut_sh_u4(raw_w, sh, wu0, wu1);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 2u) tg_mem.slow_flag[tid] = 0u;
        w4a8_moe_store_as_mx_u4(tg_mem.As_i8, a_r0, a_col16, qa0);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + w_row * kW4A8MoeMxStride + col32 * 32u + 0u) = wu0;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + w_row * kW4A8MoeMxStride + col32 * 32u + 16u) = wu1;
        if (col32 == 0u) {
          if (row_fast) {
            const float sw = w_live ? (0.5f * w4a8_moe_e8m0_safe(min(ep.x, ep.y))) : 0.0f;
            tg_mem.Sw0_s[w_row] = sw;
            tg_mem.Sw1_s[w_row] = sw;
          } else {
            tg_mem.Sw0_s[w_row] = w_live ? (0.5f * w4a8_moe_e8m0_safe(ep.x)) : 0.0f;
            tg_mem.Sw1_s[w_row] = w_live ? (0.5f * w4a8_moe_e8m0_safe(ep.y)) : 0.0f;
            tg_mem.slow_flag[w_row >> 5u] = 1u;
          }
        }
        if (tid < 32u) tg_mem.Sa_s[tid] = sa_v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
          auto a0 = tA.slice(0, int(m_off));
          auto b0 = tB.slice(0, int(sg_n * 32u));
          auto ci = op16_m.template get_destination_cooperative_tensor<
              remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
          const float sa0 = tg_mem.Sa_s[m_off + fm];
          const float sa1 = tg_mem.Sa_s[m_off + 8u + fm];
          if (tg_mem.slow_flag[sg_n] == 0u) {
            auto a1 = tA.slice(32, int(m_off));
            auto b1 = tB.slice(32, int(sg_n * 32u));
            op16_m.run(a0, b0, ci);
            op16_a.run(a1, b1, ci);
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
              threadgroup const float *swp = (hk == 0u) ? tg_mem.Sw0_s : tg_mem.Sw1_s;
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
                const bfloat4 out_v = bfloat4(
                    bfloat(acc0[e + 0u] * score), bfloat(acc0[e + 1u] * score),
                    bfloat(acc0[e + 2u] * score), bfloat(acc0[e + 3u] * score));
                *reinterpret_cast<device bfloat4 *>(expert_output + ulong(assign) * p.n + col) = out_v;
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
      const uint k0 = kb * kW4A8MoeBK;
      const uint sub32_abs = kb * 2u + col32;

      const uint4 qa0 = (a_r0 < valid_rows)
          ? *reinterpret_cast<device const uint4 *>(mid_qa + ulong(rb + a_r0) * p.k + k0 + a_col16)
          : uint4(0u);
      const uint4 qa1 = (a_r1 < valid_rows)
          ? *reinterpret_cast<device const uint4 *>(mid_qa + ulong(rb + a_r1) * p.k + k0 + a_col16)
          : uint4(0u);
      const uint4 raw_w = w_live
          ? *reinterpret_cast<device const uint4 *>(
                d_codes + w_exp_row * ulong(p.padded_k >> 1u) + ulong(sub32_abs) * 16ul)
          : uint4(0u);
      const uchar2 ep = w_live
          ? *reinterpret_cast<device const uchar2 *>(
                d_scales + w_exp_row * ulong(p.padded_k >> 5u) + ulong(kb) * 2ul)
          : uchar2(0u, 0u);
      const float sa_v = (tid < BM_TG && tid < valid_rows) ? mid_sa[ulong(kb) * p.mp + rb + tid] : 0.0f;
      const int de = (w_live && ep.x != 255u && ep.y != 255u)
          ? (int(ep.y) - int(ep.x)) : (w_live ? 255 : 0);
      const bool row_fast = (abs(de) <= 3);
      const uint sh = row_fast
          ? ((col32 == 0u) ? ((de < 0) ? uint(-de) : 0u) : ((de > 0) ? uint(de) : 0u))
          : 0u;
      uint4 wu0, wu1;
      w4a8_moe_unpack_mxfp4_lut_sh_u4(raw_w, sh, wu0, wu1);

      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < 2u) tg_mem.slow_flag[tid] = 0u;
      w4a8_moe_store_as_mx_u4(tg_mem.As_i8, a_r0, a_col16, qa0);
      w4a8_moe_store_as_mx_u4(tg_mem.As_i8, a_r1, a_col16, qa1);
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + w_row * kW4A8MoeMxStride + col32 * 32u + 0u) = wu0;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + w_row * kW4A8MoeMxStride + col32 * 32u + 16u) = wu1;
      if (col32 == 0u) {
        if (row_fast) {
          const float sw = w_live ? (0.5f * w4a8_moe_e8m0_safe(min(ep.x, ep.y))) : 0.0f;
          tg_mem.Sw0_s[w_row] = sw;
          tg_mem.Sw1_s[w_row] = sw;
        } else {
          tg_mem.Sw0_s[w_row] = w_live ? (0.5f * w4a8_moe_e8m0_safe(ep.x)) : 0.0f;
          tg_mem.Sw1_s[w_row] = w_live ? (0.5f * w4a8_moe_e8m0_safe(ep.y)) : 0.0f;
          tg_mem.slow_flag[w_row >> 5u] = 1u;
        }
      }
      if (tid < BM_TG) tg_mem.Sa_s[tid] = sa_v;
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_active) {
        auto a0 = tA.slice(0, int(m_off));
        auto b0 = tB.slice(0, int(sg_n * 32u));
        auto ci = op32_m.template get_destination_cooperative_tensor<
            remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
        const float sa0 = tg_mem.Sa_s[m_off + fm];
        const float sa1 = tg_mem.Sa_s[m_off + 8u + fm];
        const float sa2 = tg_mem.Sa_s[m_off + 16u + fm];
        const float sa3 = tg_mem.Sa_s[m_off + 24u + fm];
        if (tg_mem.slow_flag[sg_n] == 0u) {
          auto a1 = tA.slice(32, int(m_off));
          auto b1 = tB.slice(32, int(sg_n * 32u));
          op32_m.run(a0, b0, ci);
          op32_a.run(a1, b1, ci);
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
            threadgroup const float *swp = (hk == 0u) ? tg_mem.Sw0_s : tg_mem.Sw1_s;
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
                  const bfloat4 out_v = bfloat4(
                      bfloat(acc_f32[e + 0u] * score),
                      bfloat(acc_f32[e + 1u] * score),
                      bfloat(acc_f32[e + 2u] * score),
                      bfloat(acc_f32[e + 3u] * score));
                  *reinterpret_cast<device bfloat4 *>(
                      expert_output + ulong(assign) * p.n + col) = out_v;
                }
              }
            }
          }
        }
      }
    }
    return;
  }

  // Portable Apple9 fallback (zero stack allocas: 4 rows/pass in float4)
  const uint k_blocks = p.k >> 6u;
  const uint w_row = tid >> 1u;
  const uint col32 = tid & 1u;
  const uint w_col = tg_n0 + w_row;
  const bool w_live = (w_col < p.n);
  const ulong w_exp_row = ulong(expert) * ulong(p.n) + ulong(w_col);
  const ulong c_stride = ulong(p.padded_k >> 1u);
  const ulong s_stride = ulong(p.padded_k >> 5u);
  threadgroup char4 *w_dst4 =
      reinterpret_cast<threadgroup char4 *>(tg_mem.Ws_i8 + w_row * kW4A8MoeMxStride + col32 * 32u);
  const uint a_col16 = (tid & 3u) * 16u;
  const uint a_r0 = tid >> 2u;
  const uint a_r1 = a_r0 + 32u;
  device const uchar *wc_row = w_live ? (d_codes + w_exp_row * c_stride) : nullptr;
  device const uchar *ws_row = w_live ? (d_scales + w_exp_row * s_stride) : nullptr;
  const uint c_l = tid & 63u;
  const uint r_base = (tid >> 6u) * 32u;
  const uint col = tg_n0 + c_l;
  const bool col_live = (col < p.n);

  for (uint r4 = 0u; r4 < 8u; ++r4) {
    float4 acc_a9 = float4(0.0f);
    for (uint kb = 0u; kb < k_blocks; ++kb) {
      const uint k0 = kb * kW4A8MoeBK;
      const uint sub32_abs = kb * 2u + col32;
      const uint4 qa0 = (a_r0 < valid_rows)
          ? *reinterpret_cast<device const uint4 *>(mid_qa + ulong(rb + a_r0) * p.k + k0 + a_col16)
          : uint4(0u);
      const uint4 qa1 = (a_r1 < valid_rows)
          ? *reinterpret_cast<device const uint4 *>(mid_qa + ulong(rb + a_r1) * p.k + k0 + a_col16)
          : uint4(0u);
      const uint4 raw_w = wc_row
          ? *reinterpret_cast<device const uint4 *>(wc_row + ulong(sub32_abs) * 16ul)
          : uint4(0u);
      const float sw_v = ws_row
          ? (0.5f * e8m0_to_float(ws_row[sub32_abs]))
          : 0.0f;
      const float sa_v = (tid < BM_TG && tid < valid_rows) ? mid_sa[ulong(kb) * p.mp + rb + tid] : 0.0f;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      w4a8_moe_store_as_mx_u4(tg_mem.As_i8, a_r0, a_col16, qa0);
      w4a8_moe_store_as_mx_u4(tg_mem.As_i8, a_r1, a_col16, qa1);
      w4a8_moe_unpack_mxfp4_32_i8(raw_w, w_dst4);
      if (col32 == 0u) tg_mem.Sw0_s[w_row] = sw_v; else tg_mem.Sw1_s[w_row] = sw_v;
      if (tid < BM_TG) tg_mem.Sa_s[tid] = sa_v;
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (col_live && r_base + r4 * 4u < valid_rows) {
        const float sw0 = tg_mem.Sw0_s[c_l], sw1 = tg_mem.Sw1_s[c_l];
        for (uint rr = 0u; rr < 4u; ++rr) {
          const uint r_l = r_base + r4 * 4u + rr;
          if (r_l < valid_rows) {
            int32_t d0 = 0, d1 = 0;
            const uint a_off = r_l * kW4A8MoeMxStride;
            const uint w_off = c_l * kW4A8MoeMxStride;
            for (uint k = 0u; k < 32u; ++k) {
              d0 += int32_t(tg_mem.As_i8[a_off + k]) * int32_t(tg_mem.Ws_i8[w_off + k]);
              d1 += int32_t(tg_mem.As_i8[a_off + 32u + k]) * int32_t(tg_mem.Ws_i8[w_off + 32u + k]);
            }
            acc_a9[rr] = fma(tg_mem.Sa_s[r_l], fma(sw0, float(d0), sw1 * float(d1)), acc_a9[rr]);
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
// MoE Compact TG-Staged W4A8 Kernel Entry Points (<= 10.00 KiB TG SRAM)
// ============================================================================

[[max_total_threads_per_threadgroup(128)]]
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
  threadgroup W4A8MoeMxfp4SwiTgStorage tg_mem;
  moe_mxfp4_w4a8_nax_swiglu_bm64_out64_impl<false>(
      qa, sa, tile_assign, tile_expert, tile_valid_rows, total_tiles,
      g_codes, g_scales, u_codes, u_scales, mid_bf16, nullptr, nullptr, nullptr,
      p, tg_mem, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_moe_mxfp4_w4a8_nax_swiglu_bm64_out64_sk1_fwht64_a8(
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
    device int8_t *mid_qa [[buffer(10)]],
    device float *mid_sa [[buffer(11)]],
    device float *mid_ra [[buffer(12)]],
    constant W4A8MoeGemmParams &p [[buffer(13)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup W4A8MoeMxfp4SwiTgStorage tg_mem;
  moe_mxfp4_w4a8_nax_swiglu_bm64_out64_impl<true>(
      qa, sa, tile_assign, tile_expert, tile_valid_rows, total_tiles,
      g_codes, g_scales, u_codes, u_scales, nullptr, mid_qa, mid_sa, mid_ra,
      p, tg_mem, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_moe_mxfp4_w4a8_nax_down_scatter_bm64_bn64_sk1(
    device const int8_t *mid_qa [[buffer(0)]],
    device const float *mid_sa [[buffer(1)]],
    device const uint *tile_expert [[buffer(2)]],
    device const uint *tile_valid_rows [[buffer(3)]],
    device const uint *tile_assign [[buffer(4)]],
    device const uint *total_tiles [[buffer(5)]],
    device const bfloat *route_scores [[buffer(6)]],
    device const uchar *d_codes [[buffer(7)]],
    device const uchar *d_scales [[buffer(8)]],
    device bfloat *expert_output [[buffer(9)]],
    constant W4A8MoeGemmParams &p [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup W4A8MoeDownMxfp4TgStorage tg_mem;
  moe_mxfp4_w4a8_nax_down_scatter_bm64_bn64_impl(
      mid_qa, mid_sa, tile_expert, tile_valid_rows, tile_assign, total_tiles,
      route_scores, d_codes, d_scales, expert_output, p, tg_mem, lane, sg, tg);
}

#define OMLX_M5_MOE_AFFINE_W4A8_NO_HIGH(BITS, GS)                              \
  [[max_total_threads_per_threadgroup(128)]]                                   \
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
    threadgroup W4A8MoeAffineSwiTgStorage tg_mem;                              \
    moe_affine_w4a8_nax_swiglu_bm64_out64_impl<BITS##u, GS##u, false>(         \
        qa, sa, ra, tile_assign, tile_expert, tile_valid_rows, total_tiles,    \
        g_codes, nullptr, g_scales, g_biases, u_codes, nullptr, u_scales,      \
        u_biases, mid_bf16, nullptr, nullptr, nullptr, p, tg_mem,              \
        lane, sg, tg);                                                         \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_moe_affine_w4a8_nax_swiglu_q##BITS##_g##GS##_bm64_out64_sk1_fwht64_a8( \
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
      device int8_t *mid_qa [[buffer(13)]],                                    \
      device float *mid_sa [[buffer(14)]],                                     \
      device float *mid_ra [[buffer(15)]],                                     \
      constant W4A8MoeGemmParams &p [[buffer(16)]],                            \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup W4A8MoeAffineSwiTgStorage tg_mem;                              \
    moe_affine_w4a8_nax_swiglu_bm64_out64_impl<BITS##u, GS##u, true>(          \
        qa, sa, ra, tile_assign, tile_expert, tile_valid_rows, total_tiles,    \
        g_codes, nullptr, g_scales, g_biases, u_codes, nullptr, u_scales,      \
        u_biases, nullptr, mid_qa, mid_sa, mid_ra, p, tg_mem, lane, sg, tg);   \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_moe_affine_w4a8_nax_down_scatter_q##BITS##_g##GS##_bm64_bn64_sk1( \
      device const int8_t *mid_qa [[buffer(0)]],                               \
      device const float *mid_sa [[buffer(1)]],                                \
      device const float *mid_ra [[buffer(2)]],                                \
      device const uint *tile_expert [[buffer(3)]],                            \
      device const uint *tile_valid_rows [[buffer(4)]],                        \
      device const uint *tile_assign [[buffer(5)]],                            \
      device const uint *total_tiles [[buffer(6)]],                            \
      device const bfloat *route_scores [[buffer(7)]],                         \
      device const uchar *d_codes [[buffer(8)]],                               \
      device const bfloat *d_scales [[buffer(9)]],                             \
      device const bfloat *d_biases [[buffer(10)]],                            \
      device bfloat *expert_output [[buffer(11)]],                             \
      constant W4A8MoeGemmParams &p [[buffer(12)]],                            \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup W4A8MoeDownAffineTgStorage tg_mem;                             \
    moe_affine_w4a8_nax_down_scatter_bm64_bn64_impl<BITS##u, GS##u>(           \
        mid_qa, mid_sa, mid_ra, tile_expert, tile_valid_rows, tile_assign,     \
        total_tiles, route_scores, d_codes, nullptr, d_scales, d_biases,       \
        expert_output, p, tg_mem, lane, sg, tg);                               \
  }

#define OMLX_M5_MOE_AFFINE_W4A8_WITH_HIGH(BITS, GS)                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
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
    threadgroup W4A8MoeAffineSwiTgStorage tg_mem;                              \
    moe_affine_w4a8_nax_swiglu_bm64_out64_impl<BITS##u, GS##u, false>(         \
        qa, sa, ra, tile_assign, tile_expert, tile_valid_rows, total_tiles,    \
        g_codes, g_high, g_scales, g_biases, u_codes, u_high, u_scales,        \
        u_biases, mid_bf16, nullptr, nullptr, nullptr, p, tg_mem,              \
        lane, sg, tg);                                                         \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_moe_affine_w4a8_nax_swiglu_q##BITS##_g##GS##_bm64_out64_sk1_fwht64_a8( \
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
      device int8_t *mid_qa [[buffer(15)]],                                    \
      device float *mid_sa [[buffer(16)]],                                     \
      device float *mid_ra [[buffer(17)]],                                     \
      constant W4A8MoeGemmParams &p [[buffer(18)]],                            \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup W4A8MoeAffineSwiTgStorage tg_mem;                              \
    moe_affine_w4a8_nax_swiglu_bm64_out64_impl<BITS##u, GS##u, true>(          \
        qa, sa, ra, tile_assign, tile_expert, tile_valid_rows, total_tiles,    \
        g_codes, g_high, g_scales, g_biases, u_codes, u_high, u_scales,        \
        u_biases, nullptr, mid_qa, mid_sa, mid_ra, p, tg_mem, lane, sg, tg);   \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void omlx_m5_moe_affine_w4a8_nax_down_scatter_q##BITS##_g##GS##_bm64_bn64_sk1( \
      device const int8_t *mid_qa [[buffer(0)]],                               \
      device const float *mid_sa [[buffer(1)]],                                \
      device const float *mid_ra [[buffer(2)]],                                \
      device const uint *tile_expert [[buffer(3)]],                            \
      device const uint *tile_valid_rows [[buffer(4)]],                        \
      device const uint *tile_assign [[buffer(5)]],                            \
      device const uint *total_tiles [[buffer(6)]],                            \
      device const bfloat *route_scores [[buffer(7)]],                         \
      device const uchar *d_codes [[buffer(8)]],                               \
      device const uchar *d_high [[buffer(9)]],                                \
      device const bfloat *d_scales [[buffer(10)]],                            \
      device const bfloat *d_biases [[buffer(11)]],                            \
      device bfloat *expert_output [[buffer(12)]],                             \
      constant W4A8MoeGemmParams &p [[buffer(13)]],                            \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    threadgroup W4A8MoeDownAffineTgStorage tg_mem;                             \
    moe_affine_w4a8_nax_down_scatter_bm64_bn64_impl<BITS##u, GS##u>(           \
        mid_qa, mid_sa, mid_ra, tile_expert, tile_valid_rows, tile_assign,     \
        total_tiles, route_scores, d_codes, d_high, d_scales, d_biases,        \
        expert_output, p, tg_mem, lane, sg, tg);                               \
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
