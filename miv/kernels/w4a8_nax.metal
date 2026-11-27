// Route A (WP1 Part B + WP 3B): Compact TG-Staged High-Occupancy Prefill W4A8
// (INT8 GS64 Dynamic Activation x oQ4e lane_v2 / MXFP4 Weight)
// Metal 4 NAX TensorOps:
// - Default oQ4e channel: 2x `matmul2d_descriptor(32, 32, 32)` int8_t x int8_t -> int32_t
//   with hoisted Sb decode & precomputed pointer fast path (166–173 TFLOPS on M5 Ultra).
// - Opt-in oQ4e q4 native channel (`OMLX_M5_OQ4_INT4B=1`): `pi64`-permuted `As_pi64` x
//   `Ws_i4` (`raw ^ 0x88888888u`, `bw_tilde = fma(8.0f, sw, bw)`) native `int8_t x int4b_format`.
// - All TG SRAM strictly <= 10.0 KiB (3 TGs/core on Apple10 32 KiB/core SRAM), 0 register spills:
//   * Affine Linear & SwiGLU (128t, BK=64, stride 64): 8,960 / 8,976 B (8.75–8.77 KiB)
//   * Affine q4 int4b Linear & SwiGLU (128t, BK=64):   6,912 / 8,720 B (6.75–8.52 KiB)
//   * MXFP4  Linear & SwiGLU (128t, BK=64, stride 72): 9,984 B (9.75 KiB -> 3 TGs/core)
// - Consumes canonical `oq_nax_lane_g{64,128}_v2` (pi64 intra-row quartet order)
//   and `mx_nax_tiled_bk128_v1` (MXFP4 E2M1 + UE8M0).
// - Compile-time Apple10 fast path + portable Apple9 fallback (`w4a8_has_apple10_i8_layout()`).

#include "common_math.h"
#include "fwht64.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct W4A8RmsNormParams {
  uint k;
  uint rows;          // word 1 (matches dynamic token patch convention of RMSNorm)
  uint mp;
  float epsilon;
  float weight_offset;
  uint write_bf16;
};

struct W4A8QuantParams {
  uint k;
  uint mp;
  uint reserved;
  uint rows;          // word 3 (matches dynamic token patch convention)
};

struct W4A8GdnNormGateParams {
  uint key_heads;
  uint value_heads;
  uint key_dim;
  uint value_dim;
  uint conv_width;
  uint conv_kernel;
  uint tokens;        // word 6 (matches GdnChunkMegaParams token patch convention)
  float epsilon;
  uint mp;
  uint write_bf16;
};

struct W4A8LinearParams {
  uint n;             // slice column count (N or N - row_offset)
  uint k;
  uint mp;            // sa/ra row stride: sa[g * mp + row]
  uint rows;          // word 3 (dynamic token patch)
  uint padded_n;      // full weight padded_n (for affine sb stride)
  uint row_offset;    // % 64 == 0 (starting output col in weight & y)
  uint out_stride;    // y/qa_out row stride in elements (0 => row_offset + n)
  uint reserved;      // bit 0: W4A8_EPI_ALSO_WRITE_BF16 (for Mode B SwiGLU+FWHT64+A8)
};

namespace {

constant constexpr uint kW4A8BK = 64u;
constant constexpr uint kW4A8LinStride = 64u;   // 16B-aligned (uint4 stores)
constant constexpr uint kW4A8MxStride = 64u;    // 16B-aligned (uint4 stores, 8,992 B TG SRAM)
constant constexpr uint kW4A8EpiStride = 68u;   // 64 + 4 bfloat (8B-aligned)

struct W4A8AffineLinTgStorage {
  alignas(16) int8_t As_i8[64u * kW4A8LinStride]; // 4,096 B
  alignas(16) int8_t Ws_i8[64u * kW4A8LinStride]; // 4,096 B
  float Sa_s[64u];                                //   256 B
  float Ra_s[64u];                                //   256 B
  uint Sb_s[64u];                                 //   256 B (packed bfloat2)
};
static_assert(sizeof(W4A8AffineLinTgStorage) == 8960ul,
              "W4A8AffineLinTgStorage must be 8,960 B (8.75 KiB)");
static_assert(sizeof(W4A8AffineLinTgStorage) <= kApple10HighOccTgSramBytes,
              "W4A8AffineLinTgStorage exceeds 10.0 KiB budget");

struct W4A8AffineSwiTgStorage {
  union {
    struct {
      alignas(16) int8_t As_i8[64u * kW4A8LinStride]; // 4,096 B
      alignas(16) int8_t Ws_i8[64u * kW4A8LinStride]; // 4,096 B
      float Sa_s[64u];                                //   256 B
      float Ra_s[64u];                                //   256 B
      uint Sb_s[64u];                                 //   256 B (packed bfloat2)
    } kloop;
    alignas(16) bfloat SwiBuf[64u * kW4A8EpiStride];  // 8,704 B (Mode B 1-phase epilogue)
  };
  uint tg_meta[4u];                                   //    16 B {rb_tg, n0_tg_base}
};
static_assert(sizeof(W4A8AffineSwiTgStorage) == 8976ul,
              "W4A8AffineSwiTgStorage must be 8,976 B (8.77 KiB)");
static_assert(sizeof(W4A8AffineSwiTgStorage) <= kApple10HighOccTgSramBytes,
              "W4A8AffineSwiTgStorage exceeds 10.0 KiB budget");

// WP 3B Native int8_t x int4b_format TG storage (pi64-permuted As_pi64 + packed signed Ws_i4).
struct W4A8AffineQ4I4LinTgStorage {
  alignas(16) int8_t As_pi64[64u * kW4A8LinStride]; // 4,096 B
  alignas(16) uchar Ws_i4[64u * 32u];               // 2,048 B (64 rows x 64 int4b)
  float Sa_s[64u];                                  //   256 B
  float Ra_s[64u];                                  //   256 B
  uint Sb_s[64u];                                   //   256 B (packed bfloat2)
};
static_assert(sizeof(W4A8AffineQ4I4LinTgStorage) == 6912ul,
              "W4A8AffineQ4I4LinTgStorage must be 6,912 B (6.75 KiB)");
static_assert(sizeof(W4A8AffineQ4I4LinTgStorage) <= kApple10HighOccTgSramBytes,
              "W4A8AffineQ4I4LinTgStorage exceeds 10.0 KiB budget");

struct W4A8AffineQ4I4SwiTgStorage {
  union {
    struct {
      alignas(16) int8_t As_pi64[64u * kW4A8LinStride]; // 4,096 B
      alignas(16) uchar Ws_i4[64u * 32u];               // 2,048 B
      float Sa_s[64u];                                  //   256 B
      float Ra_s[64u];                                  //   256 B
      uint Sb_s[64u];                                   //   256 B
    } kloop;                                            // 6,912 B
    alignas(16) bfloat SwiBuf[64u * kW4A8EpiStride];    // 8,704 B
  };
  uint tg_meta[4u];                                     //    16 B {rb_tg, n0_tg_base}
};
static_assert(sizeof(W4A8AffineQ4I4SwiTgStorage) == 8720ul,
              "W4A8AffineQ4I4SwiTgStorage must be 8,720 B (8.52 KiB)");
static_assert(sizeof(W4A8AffineQ4I4SwiTgStorage) <= kApple10HighOccTgSramBytes,
              "W4A8AffineQ4I4SwiTgStorage exceeds 10.0 KiB budget");

struct W4A8Mxfp4TgStorage {
  union {
    struct {
      alignas(16) int8_t As_i8[64u * kW4A8MxStride]; // 4,096 B
      alignas(16) int8_t Ws_i8[64u * kW4A8MxStride]; // 4,096 B
      float Sa_s[64u];                               //   256 B
      float Sw0_s[64u];                              //   256 B
      float Sw1_s[64u];                              //   256 B
      uint slow_flag[4u];                            //    16 B (per-SG race-free shift-merge flag)
    } kloop;
    alignas(16) bfloat SwiBuf[64u * kW4A8EpiStride]; // 8,704 B (Mode B 1-phase epilogue)
  };
  uint tg_meta[4u];                                  //    16 B {rb_tg, n0_tg_base}
};
static_assert(sizeof(W4A8Mxfp4TgStorage) == 8992ul,
              "W4A8Mxfp4TgStorage must be 8,992 B (8.78 KiB)");
static_assert(sizeof(W4A8Mxfp4TgStorage) <= kApple10HighOccTgSramBytes,
              "W4A8Mxfp4TgStorage exceeds 10.0 KiB budget");

// Compile-time constant on Apple10 vs Apple9 PSO specialization.
inline bool w4a8_has_apple10_i8_layout() {
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

inline uint2 w4a8_swizzle_tg_group_m8(uint2 tg_xy, uint num_pid_n, uint num_pid_m) {
  constexpr uint GROUP_M = 8u;
  if (num_pid_n == 0u || num_pid_m == 0u) {
    return tg_xy;
  }
  const uint pid = tg_xy.y * num_pid_n + tg_xy.x;
  const uint num_pid_in_group = GROUP_M * num_pid_n;
  const uint group_id = pid / num_pid_in_group;
  const uint first_pid_m = group_id * GROUP_M;
  const uint group_size_m = min(num_pid_m - first_pid_m, GROUP_M);
  const uint pid_m = first_pid_m + (pid % group_size_m);
  const uint pid_n = (pid % num_pid_in_group) / group_size_m;
  return uint2(pid_n, pid_m);
}

inline int8_t w4a8_mxfp4_e2m1_to_i8_x2(uint nib) {
  const int mag2 = int((0xC8643210u >> ((nib & 7u) * 4u)) & 0xFu);
  return int8_t((nib & 8u) ? -mag2 : mag2);
}

inline void w4a8_quantize_pair_g64(
    float2 fo,
    uint lane,
    device int8_t *qa_grp,
    device float *sa_dst,
    device float *ra_dst) {
  const float amax = simd_max(fmax(fabs(fo.x), fabs(fo.y)));
  const float sa = amax * (1.0f / 127.0f);
  const float inv_sa = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
  const int q0 = int(clamp(rint(fo.x * inv_sa), -127.0f, 127.0f));
  const int q1 = int(clamp(rint(fo.y * inv_sa), -127.0f, 127.0f));
  reinterpret_cast<device char2 *>(qa_grp)[lane] = char2(int8_t(q0), int8_t(q1));
  const int r_sum = simd_sum(q0 + q1);
  if (lane == 0u) {
    *sa_dst = sa;
    *ra_dst = float(r_sum);
  }
}

template <uint Mode>
inline void w4a8_decode_sb(uint sb_u, thread float &sw, thread float &bw) {
  const vec<bfloat, 2> pair = as_type<vec<bfloat, 2>>(sb_u);
  sw = float(pair[0]);
  bw = float(pair[1]);
  if constexpr (Mode == 3u) {
    bw = fma(128.0f, sw, bw);
  }
}

// ============================================================================
// Coalesced oQ4e lane_v2 (pi64) Fetchers & Linear-K Ws_i8 Unpackers
// Each thread loads half_k in {0, 1} = 32 contiguous stored elements
// (q0 = half_k*2, q1 = q0+1 across all 4 s=0..3 steps) with ONE coalesced load.
// ============================================================================
template <uint Mode>
struct W4A8OqRaw32 {
  uint sb_u;
  uint4 r0;
  uint4 r1;
};

template <uint Mode, uint BN_STORED>
inline W4A8OqRaw32<Mode> w4a8_oq_fetch_raw_32_fast(
    device const uchar *wc_row,
    device const uchar *wh_row,
    device const uint *sb_col_base,
    uint padded_n,
    uint g,
    uint half_k,
    bool load_sb) {
  W4A8OqRaw32<Mode> out;
  out.sb_u = load_sb ? sb_col_base[ulong(g) * ulong(padded_n)] : 0u;
  out.r1 = uint4(0u);
  if constexpr (Mode == 0u) {
    out.r0 = reinterpret_cast<device const uint4 *>(
        wc_row + ulong(g) * (ulong(BN_STORED) * 32ul))[half_k];
  } else if constexpr (Mode == 1u) {
    out.r0 = reinterpret_cast<device const uint4 *>(
        wc_row + ulong(g) * (ulong(BN_STORED) * 32ul))[half_k];
    const uint2 hi2 = reinterpret_cast<device const uint2 *>(
        wh_row + ulong(g) * (ulong(BN_STORED) * 16ul))[half_k];
    out.r1 = uint4(hi2.x, hi2.y, 0u, 0u);
  } else if constexpr (Mode == 2u) {
    const uint2 lo2 = reinterpret_cast<device const uint2 *>(
        wc_row + ulong(g) * (ulong(BN_STORED) * 16ul))[half_k];
    const uint hi1 = reinterpret_cast<device const uint *>(
        wh_row + ulong(g) * (ulong(BN_STORED) * 8ul))[half_k];
    out.r0 = uint4(lo2.x, lo2.y, hi1, 0u);
  } else {
    device const uint4 *p4 = reinterpret_cast<device const uint4 *>(
        wc_row + ulong(g) * (ulong(BN_STORED) * 64ul)) + half_k * 2u;
    out.r0 = p4[0];
    out.r1 = p4[1];
  }
  return out;
}

template <uint Mode, uint BN_STORED>
inline W4A8OqRaw32<Mode> w4a8_oq_fetch_raw_32_checked(
    device const uchar *codes,
    device const uchar *high_plane,
    device const uint *sb,
    uint global_col,
    uint col_limit,
    uint padded_n,
    uint groups,
    uint g,
    uint half_k,
    bool load_sb) {
  W4A8OqRaw32<Mode> out = {0u, uint4(0u), uint4(0u)};
  if (global_col >= col_limit) {
    return out;
  }
  const uint tile_n = global_col / BN_STORED;
  const uint rn = global_col % BN_STORED;
  const ulong tile_idx = ulong(tile_n) * ulong(groups) + ulong(g);
  out.sb_u = load_sb ? sb[ulong(g) * ulong(padded_n) + global_col] : 0u;
  if constexpr (Mode == 0u) {
    out.r0 = reinterpret_cast<device const uint4 *>(
        codes + tile_idx * (ulong(BN_STORED) * 32ul) + ulong(rn) * 32ul)[half_k];
  } else if constexpr (Mode == 1u) {
    out.r0 = reinterpret_cast<device const uint4 *>(
        codes + tile_idx * (ulong(BN_STORED) * 32ul) + ulong(rn) * 32ul)[half_k];
    const uint2 hi2 = reinterpret_cast<device const uint2 *>(
        high_plane + tile_idx * (ulong(BN_STORED) * 16ul) + ulong(rn) * 16ul)[half_k];
    out.r1 = uint4(hi2.x, hi2.y, 0u, 0u);
  } else if constexpr (Mode == 2u) {
    const uint2 lo2 = reinterpret_cast<device const uint2 *>(
        codes + tile_idx * (ulong(BN_STORED) * 16ul) + ulong(rn) * 16ul)[half_k];
    const uint hi1 = reinterpret_cast<device const uint *>(
        high_plane + tile_idx * (ulong(BN_STORED) * 8ul) + ulong(rn) * 8ul)[half_k];
    out.r0 = uint4(lo2.x, lo2.y, hi1, 0u);
  } else {
    device const uint4 *p4 = reinterpret_cast<device const uint4 *>(
        codes + tile_idx * (ulong(BN_STORED) * 64ul) + ulong(rn) * 64ul) + half_k * 2u;
    out.r0 = p4[0];
    out.r1 = p4[1];
  }
  return out;
}

// Unpack 32 lane_v2 (pi64) weights into logical linear-K positions in a 64-element Ws_i8 row.
template <uint Mode>
inline void w4a8_oq_unpack_raw_32_to_ws64(
    const W4A8OqRaw32<Mode> raw,
    threadgroup char4 *w_row_dst4,
    uint half_k) {
  const uint base_c4 = half_k * 8u;
  if constexpr (Mode == 0u) {
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      const uint wd = raw.r0[w];
      w_row_dst4[omlx_m5_pi16(base_c4 + w * 2u + 0u)] = char4(
          int8_t((wd >>  0u) & 0xFu), int8_t((wd >>  4u) & 0xFu),
          int8_t((wd >>  8u) & 0xFu), int8_t((wd >> 12u) & 0xFu));
      w_row_dst4[omlx_m5_pi16(base_c4 + w * 2u + 1u)] = char4(
          int8_t((wd >> 16u) & 0xFu), int8_t((wd >> 20u) & 0xFu),
          int8_t((wd >> 24u) & 0xFu), int8_t((wd >> 28u) & 0xFu));
    }
  } else if constexpr (Mode == 1u) {
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      const uint wd = raw.r0[w];
      const uint h16 = (w < 2u) ? (raw.r1.x >> (w * 16u)) : (raw.r1.y >> ((w - 2u) * 16u));
      w_row_dst4[omlx_m5_pi16(base_c4 + w * 2u + 0u)] = char4(
          int8_t(((wd >>  0u) & 0xFu) | (((h16 >>  0u) & 3u) << 4u)),
          int8_t(((wd >>  4u) & 0xFu) | (((h16 >>  2u) & 3u) << 4u)),
          int8_t(((wd >>  8u) & 0xFu) | (((h16 >>  4u) & 3u) << 4u)),
          int8_t(((wd >> 12u) & 0xFu) | (((h16 >>  6u) & 3u) << 4u)));
      w_row_dst4[omlx_m5_pi16(base_c4 + w * 2u + 1u)] = char4(
          int8_t(((wd >> 16u) & 0xFu) | (((h16 >>  8u) & 3u) << 4u)),
          int8_t(((wd >> 20u) & 0xFu) | (((h16 >> 10u) & 3u) << 4u)),
          int8_t(((wd >> 24u) & 0xFu) | (((h16 >> 12u) & 3u) << 4u)),
          int8_t(((wd >> 28u) & 0xFu) | (((h16 >> 14u) & 3u) << 4u)));
    }
  } else if constexpr (Mode == 2u) {
    #pragma clang loop unroll(full)
    for (uint q16 = 0u; q16 < 2u; ++q16) {
      const uint lword = (q16 == 0u) ? raw.r0.x : raw.r0.y;
      const uint h16 = (raw.r0.z >> (q16 * 16u)) & 0xFFFFu;
      #pragma clang loop unroll(full)
      for (uint half8 = 0u; half8 < 2u; ++half8) {
        const uint l16 = lword >> (half8 * 16u);
        const uint h8 = h16 >> (half8 * 8u);
        const uint c4 = base_c4 + (q16 * 2u + half8) * 2u;
        w_row_dst4[omlx_m5_pi16(c4 + 0u)] = char4(
            int8_t(((l16 >>  0u) & 3u) | (((h8 >> 0u) & 1u) << 2u)),
            int8_t(((l16 >>  2u) & 3u) | (((h8 >> 1u) & 1u) << 2u)),
            int8_t(((l16 >>  4u) & 3u) | (((h8 >> 2u) & 1u) << 2u)),
            int8_t(((l16 >>  6u) & 3u) | (((h8 >> 3u) & 1u) << 2u)));
        w_row_dst4[omlx_m5_pi16(c4 + 1u)] = char4(
            int8_t(((l16 >>  8u) & 3u) | (((h8 >> 4u) & 1u) << 2u)),
            int8_t(((l16 >> 10u) & 3u) | (((h8 >> 5u) & 1u) << 2u)),
            int8_t(((l16 >> 12u) & 3u) | (((h8 >> 6u) & 1u) << 2u)),
            int8_t(((l16 >> 14u) & 3u) | (((h8 >> 7u) & 1u) << 2u)));
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (uint q16 = 0u; q16 < 2u; ++q16) {
      const uint4 pk = ((q16 == 0u) ? raw.r0 : raw.r1) ^ uint4(0x80808080u);
      #pragma clang loop unroll(full)
      for (uint half8 = 0u; half8 < 2u; ++half8) {
        const uint c4 = base_c4 + (q16 * 2u + half8) * 2u;
        w_row_dst4[omlx_m5_pi16(c4 + 0u)] = as_type<char4>(pk[half8 * 2u + 0u]);
        w_row_dst4[omlx_m5_pi16(c4 + 1u)] = as_type<char4>(pk[half8 * 2u + 1u]);
      }
    }
  }
}

inline void w4a8_mxfp4_unpack_raw_32_i8(uint4 raw_w, threadgroup char4 *w_dst4) {
  #pragma clang loop unroll(full)
  for (uint w = 0u; w < 4u; ++w) {
    const uint wd = raw_w[w];
    w_dst4[w * 2u + 0u] = char4(
        w4a8_mxfp4_e2m1_to_i8_x2((wd >>  0u) & 0xFu),
        w4a8_mxfp4_e2m1_to_i8_x2((wd >>  4u) & 0xFu),
        w4a8_mxfp4_e2m1_to_i8_x2((wd >>  8u) & 0xFu),
        w4a8_mxfp4_e2m1_to_i8_x2((wd >> 12u) & 0xFu));
    w_dst4[w * 2u + 1u] = char4(
        w4a8_mxfp4_e2m1_to_i8_x2((wd >> 16u) & 0xFu),
        w4a8_mxfp4_e2m1_to_i8_x2((wd >> 20u) & 0xFu),
        w4a8_mxfp4_e2m1_to_i8_x2((wd >> 24u) & 0xFu),
        w4a8_mxfp4_e2m1_to_i8_x2((wd >> 28u) & 0xFu));
  }
}

// Constant-buffer 2 KiB pre-shifted byte LUT for lossless MXFP4 (2 * E2M1 << sh, sh in 0..3).
// Passed via Encoder::dynamic(...) (icb=1 + MTL_SHADER_VALIDATION=1 clean, 0 B extra SRAM).
inline float w4a8_mxfp4_e8m0_safe(uchar code) {
  if (code == 255u) return 0.0f;
  return e8m0_to_float(code);
}

inline void w4a8_mxfp4_unpack_lut_sh_u4(
    uint4 raw_w, uint sh, constant ushort *mxfp4_lut,
    thread uint4 &wu0, thread uint4 &wu1) {
  constant ushort *lut = mxfp4_lut + (sh << 8u);
  wu0.x = uint(lut[(raw_w.x >>  0u) & 0xFFu]) | (uint(lut[(raw_w.x >>  8u) & 0xFFu]) << 16u);
  wu0.y = uint(lut[(raw_w.x >> 16u) & 0xFFu]) | (uint(lut[(raw_w.x >> 24u) & 0xFFu]) << 16u);
  wu0.z = uint(lut[(raw_w.y >>  0u) & 0xFFu]) | (uint(lut[(raw_w.y >>  8u) & 0xFFu]) << 16u);
  wu0.w = uint(lut[(raw_w.y >> 16u) & 0xFFu]) | (uint(lut[(raw_w.y >> 24u) & 0xFFu]) << 16u);
  wu1.x = uint(lut[(raw_w.z >>  0u) & 0xFFu]) | (uint(lut[(raw_w.z >>  8u) & 0xFFu]) << 16u);
  wu1.y = uint(lut[(raw_w.z >> 16u) & 0xFFu]) | (uint(lut[(raw_w.z >> 24u) & 0xFFu]) << 16u);
  wu1.z = uint(lut[(raw_w.w >>  0u) & 0xFFu]) | (uint(lut[(raw_w.w >>  8u) & 0xFFu]) << 16u);
  wu1.w = uint(lut[(raw_w.w >> 16u) & 0xFFu]) | (uint(lut[(raw_w.w >> 24u) & 0xFFu]) << 16u);
}

inline void w4a8_store_as_mx_u4(threadgroup int8_t *As_i8, uint a_r, uint a_col16, uint4 qa_v) {
  *reinterpret_cast<threadgroup uint4 *>(As_i8 + a_r * kW4A8MxStride + a_col16) = qa_v;
}

// ============================================================================
// Mode B Shared Fused Epilogue: SwiGLU + FWHT-64 + GS64 INT8 (Aliases Dead kloop)
// ============================================================================
template <uint RowsPerSg>
inline void w4a8_swiglu_epilogue_fwht64_a8_phase(
    threadgroup bfloat *SwiBuf,
    uint rb_half,
    uint n0_tg,
    device int8_t *qa_out,
    device float *sa_out,
    device float *ra_out,
    bool also_bf16,
    device bfloat *y_bf16,
    constant W4A8LinearParams &p,
    uint out_stride,
    uint lane,
    uint sg) {
  const uint g_col = (p.row_offset + n0_tg) / 64u;
  const uint base_r = sg * RowsPerSg;
  const uint lcol = n0_tg + lane * 2u;
  const bool aligned2 = ((p.row_offset & 1u) == 0u) && ((out_stride & 1u) == 0u);
  #pragma clang loop unroll(disable)
  for (uint rr = 0u; rr < RowsPerSg; ++rr) {
    const uint r_l = base_r + rr;
    const uint row = rb_half + r_l;
    const ulong q_off = ulong(row) * ulong(out_stride) + ulong(g_col) * 64ul;
    if (row < p.rows) {
      const bfloat2 s2 =
          reinterpret_cast<const threadgroup bfloat2 *>(SwiBuf + r_l * kW4A8EpiStride)[lane];
      if (also_bf16) {
        const ulong y_off = ulong(row) * ulong(out_stride) + ulong(p.row_offset + lcol);
        if (lcol + 1u < p.n) {
          if (aligned2) {
            *reinterpret_cast<device bfloat2 *>(y_bf16 + y_off) = s2;
          } else {
            y_bf16[y_off + 0ul] = s2.x;
            y_bf16[y_off + 1ul] = s2.y;
          }
        } else if (lcol < p.n) {
          y_bf16[y_off] = s2.x;
        }
      }
      float v0 = float(s2.x);
      float v1 = float(s2.y);
      omlx_m5_fwht64_sg32(v0, v1, lane);
      omlx_m5_quantize_pair_g64_f32(
          v0, v1, lane, qa_out + q_off,
          sa_out + ulong(g_col) * p.mp + row,
          ra_out + ulong(g_col) * p.mp + row);
    } else if (row < p.mp) {
      reinterpret_cast<device char2 *>(qa_out + q_off)[lane] = char2(0);
      if (lane == 0u) {
        sa_out[ulong(g_col) * p.mp + row] = 0.0f;
        ra_out[ulong(g_col) * p.mp + row] = 0.0f;
      }
    }
  }
}

// ============================================================================
// Dense Affine Compact TG-Staged W4A8 Linear (BM=64, BN=64, BK=64, 4 SGs = 128 threads)
// 8,960 B (8.75 KiB <= 10.0 KiB) SRAM -> 3 TGs/core, 0 spills, 163+ TFLOPS!
// ============================================================================
template <uint Mode, uint GroupSize, uint BN_STORED>
inline void affine_w4a8_nax_linear_impl(
    device const int8_t *qa,
    device const float *sa,
    device const float *ra,
    device const uchar *codes,
    device const uchar *high_plane,
    device const uint *sb,
    device bfloat *y,
    constant W4A8LinearParams &p,
    threadgroup W4A8AffineLinTgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 grid_tgs) {
  (void)GroupSize;
  constexpr uint BM_TG = 64u;
  constexpr uint BN_TG = 64u;
  constexpr uint SM = 32u;
  constexpr uint SN = 32u;

  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  const uint2 pid_nm = w4a8_swizzle_tg_group_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg = pid_nm.x * BN_TG;
  if (rb_tg >= p.rows || n0_tg >= p.n) {
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * SN;
  const bool sg_active = (rb_sg < p.rows) && (n0_sg < p.n);
  const uint groups = p.k / kW4A8BK;
  const uint col_limit = p.row_offset + p.n;

  const uint w_row = tid >> 1u;
  const uint half_k = tid & 1u;
  const uint gcol_w = p.row_offset + n0_tg + w_row;
  threadgroup char4 *w_row_dst4 =
      reinterpret_cast<threadgroup char4 *>(tg_mem.Ws_i8 + w_row * kW4A8LinStride);
  const uint a_col16 = (tid & 3u) * 16u;
  const uint a_r0 = tid >> 2u;
  const uint a_r1 = a_r0 + 32u;
  const bool full_tile = (rb_tg + BM_TG <= p.rows) && (n0_tg + BN_TG <= p.n);

  constexpr ulong kCodeBytes = (Mode == 2u) ? 16ul : (Mode == 3u) ? 64ul : 32ul;
  constexpr ulong kHighBytes = (Mode == 1u) ? 16ul : (Mode == 2u) ? 8ul : 0ul;
  device const uchar *wc_row = nullptr;
  device const uchar *wh_row = nullptr;
  device const uint *sb_col = nullptr;
  if (full_tile) {
    const uint tile_n = gcol_w / BN_STORED;
    const uint rn = gcol_w % BN_STORED;
    const ulong tile_base = ulong(tile_n) * ulong(groups);
    wc_row = codes + tile_base * (ulong(BN_STORED) * kCodeBytes) + ulong(rn) * kCodeBytes;
    wh_row = (kHighBytes > 0ul)
        ? (high_plane + tile_base * (ulong(BN_STORED) * kHighBytes) + ulong(rn) * kHighBytes)
        : nullptr;
    sb_col = sb + gcol_w;
  }

  if (w4a8_has_apple10_i8_layout()) {
    constexpr auto desc_i8 = matmul2d_descriptor(
        32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_i8, execution_simdgroup> op_i8;
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
        tg_mem.As_i8, dextents<int32_t, 2>(int(kW4A8LinStride), 64));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
        tg_mem.Ws_i8, dextents<int32_t, 2>(int(kW4A8LinStride), 64));

    float acc_f32[32];
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 32u; ++i) {
      acc_f32[i] = 0.0f;
    }
    const uint qid = lane >> 2u;
    const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
    const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

    for (uint g = 0u; g < groups; ++g) {
      const uint k0 = g * kW4A8BK;
      uint4 qa0, qa1;
      W4A8OqRaw32<Mode> raw_w;
      if (full_tile) {
        qa0 = *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0 + a_col16);
        qa1 = *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0 + a_col16);
        raw_w = w4a8_oq_fetch_raw_32_fast<Mode, BN_STORED>(
            wc_row, wh_row, sb_col, p.padded_n, g, half_k, half_k == 0u);
      } else {
        qa0 = (rb_tg + a_r0 < p.rows)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0 + a_col16)
            : uint4(0u);
        qa1 = (rb_tg + a_r1 < p.rows)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0 + a_col16)
            : uint4(0u);
        raw_w = w4a8_oq_fetch_raw_32_checked<Mode, BN_STORED>(
            codes, high_plane, sb, gcol_w, col_limit, p.padded_n, groups, g, half_k, half_k == 0u);
      }
      float sa_v = 0.0f;
      float ra_v = 0.0f;
      if (tid < BM_TG) {
        const uint r_g = rb_tg + tid;
        if (r_g < p.rows) {
          sa_v = sa[ulong(g) * p.mp + r_g];
          ra_v = ra[ulong(g) * p.mp + r_g];
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + a_r0 * kW4A8LinStride + a_col16) = qa0;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + a_r1 * kW4A8LinStride + a_col16) = qa1;
      w4a8_oq_unpack_raw_32_to_ws64<Mode>(raw_w, w_row_dst4, half_k);
      if (half_k == 0u) {
        tg_mem.Sb_s[w_row] = raw_w.sb_u;
      }
      if (tid < BM_TG) {
        tg_mem.Sa_s[tid] = sa_v;
        tg_mem.Ra_s[tid] = ra_v;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_active) {
        auto a0 = tA.slice(0, int(sg_m * SM));
        auto a1 = tA.slice(32, int(sg_m * SM));
        auto b0 = tB.slice(0, int(sg_n * SN));
        auto b1 = tB.slice(32, int(sg_n * SN));
        auto ci = op_i8.template get_destination_cooperative_tensor<
            remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 32u; ++i) {
          ci[i] = 0;
        }
        op_i8.run(a0, b0, ci);
        op_i8.run(a1, b1, ci);

        const uint r_base = sg_m * SM + fm;
        const float sa0 = tg_mem.Sa_s[r_base];
        const float sa1 = tg_mem.Sa_s[r_base + 8u];
        const float sa2 = tg_mem.Sa_s[r_base + 16u];
        const float sa3 = tg_mem.Sa_s[r_base + 24u];
        const float ra0 = tg_mem.Ra_s[r_base];
        const float ra1 = tg_mem.Ra_s[r_base + 8u];
        const float ra2 = tg_mem.Ra_s[r_base + 16u];
        const float ra3 = tg_mem.Ra_s[r_base + 24u];
        #pragma clang loop unroll(full)
        for (uint u = 0u; u < 2u; ++u) {
          const uint c_base = sg_n * SN + u * 16u + fn;
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            float sw, bw;
            w4a8_decode_sb<Mode>(tg_mem.Sb_s[c_base + c], sw, bw);
            const uint b0i = u * 8u + c;
            const uint b1i = 16u + u * 8u + c;
            acc_f32[b0i]      = fma(sa0, fma(sw, float(ci[b0i]),      bw * ra0), acc_f32[b0i]);
            acc_f32[b0i + 4u] = fma(sa1, fma(sw, float(ci[b0i + 4u]), bw * ra1), acc_f32[b0i + 4u]);
            acc_f32[b1i]      = fma(sa2, fma(sw, float(ci[b1i]),      bw * ra2), acc_f32[b1i]);
            acc_f32[b1i + 4u] = fma(sa3, fma(sw, float(ci[b1i + 4u]), bw * ra3), acc_f32[b1i + 4u]);
          }
        }
      }
    }

    if (sg_active) {
      const bool aligned_out = ((out_stride & 3u) == 0u) && ((p.row_offset & 3u) == 0u);
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        if (rb_sg + mm * 16u < p.rows) {
          #pragma clang loop unroll(full)
          for (uint h = 0u; h < 2u; ++h) {
            const uint row = rb_sg + mm * 16u + fm + h * 8u;
            if (row < p.rows) {
              #pragma clang loop unroll(full)
              for (uint u = 0u; u < 2u; ++u) {
                const uint lcol = n0_sg + u * 16u + fn;
                const uint gcol = p.row_offset + lcol;
                if (lcol < p.n) {
                  const uint e = mm * 16u + u * 8u + h * 4u;
                  const bfloat4 out_v = bfloat4(
                      bfloat(acc_f32[e + 0u]), bfloat(acc_f32[e + 1u]),
                      bfloat(acc_f32[e + 2u]), bfloat(acc_f32[e + 3u]));
                  if (aligned_out && lcol + 4u <= p.n) {
                    *reinterpret_cast<device bfloat4 *>(
                        y + ulong(row) * ulong(out_stride) + ulong(gcol)) = out_v;
                  } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 4u; ++j) {
                      if (lcol + j < p.n) {
                        y[ulong(row) * ulong(out_stride) + ulong(gcol + j)] = out_v[j];
                      }
                    }
                  }
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
  const uint lcol = n0_tg + c_l;
  const uint gcol = p.row_offset + lcol;
  const bool col_live = (lcol < p.n);
  float acc_a9[32];
  for (uint rr = 0u; rr < 32u; ++rr) acc_a9[rr] = 0.0f;

  for (uint g = 0u; g < groups; ++g) {
    const uint k0 = g * kW4A8BK;
    const uint4 qa0 = (rb_tg + a_r0 < p.rows)
        ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0 + a_col16)
        : uint4(0u);
    const uint4 qa1 = (rb_tg + a_r1 < p.rows)
        ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0 + a_col16)
        : uint4(0u);
    const W4A8OqRaw32<Mode> raw_w = w4a8_oq_fetch_raw_32_checked<Mode, BN_STORED>(
        codes, high_plane, sb, gcol_w, col_limit, p.padded_n, groups, g, half_k, half_k == 0u);
    float sa_v = 0.0f, ra_v = 0.0f;
    if (tid < BM_TG && rb_tg + tid < p.rows) {
      sa_v = sa[ulong(g) * p.mp + rb_tg + tid];
      ra_v = ra[ulong(g) * p.mp + rb_tg + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + a_r0 * kW4A8LinStride + a_col16) = qa0;
    *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + a_r1 * kW4A8LinStride + a_col16) = qa1;
    w4a8_oq_unpack_raw_32_to_ws64<Mode>(raw_w, w_row_dst4, half_k);
    if (half_k == 0u) tg_mem.Sb_s[w_row] = raw_w.sb_u;
    if (tid < BM_TG) { tg_mem.Sa_s[tid] = sa_v; tg_mem.Ra_s[tid] = ra_v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (col_live) {
      float sw = 0.0f, bw = 0.0f;
      w4a8_decode_sb<Mode>(tg_mem.Sb_s[c_l], sw, bw);
      for (uint rr = 0u; rr < 32u; ++rr) {
        const uint r_l = r_base + rr;
        if (rb_tg + r_l < p.rows) {
          int32_t dot = 0;
          for (uint k = 0u; k < kW4A8BK; ++k) {
            dot += int32_t(tg_mem.As_i8[r_l * kW4A8LinStride + k]) *
                   int32_t(tg_mem.Ws_i8[c_l * kW4A8LinStride + k]);
          }
          acc_a9[rr] = fma(tg_mem.Sa_s[r_l], fma(sw, float(dot), bw * tg_mem.Ra_s[r_l]), acc_a9[rr]);
        }
      }
    }
  }
  if (col_live) {
    for (uint rr = 0u; rr < 32u; ++rr) {
      const uint row = rb_tg + r_base + rr;
      if (row < p.rows) {
        y[ulong(row) * ulong(out_stride) + ulong(gcol)] = bfloat(acc_a9[rr]);
      }
    }
  }
}

// ============================================================================
// Dense Affine Compact TG-Staged W4A8 Fused Gate + Up + SwiGLU (BM=64, BK=64)
// 4 SGs = 128 threads (WM=2, WN=2, SM=32), 8,960 B (8.75 KiB <= 10.0 KiB) SRAM -> 3 TGs/core, 0 spills!
// - Mode A (BF16): 1-pass BN_out=32 (2x wave parallelism, zero wave-tail penalty)
// - Mode B (Fused FWHT-64 + GS64 INT8): 2-pass BN_out=64 with 4 KiB L1-cached qa_out stash
// ============================================================================
template <uint ModeG, uint ModeU, uint GroupSize, bool FusedFwht64A8>
inline void affine_w4a8_nax_swiglu_out64_impl(
    device const int8_t *qa,
    device const float *sa,
    device const float *ra,
    device const uchar *g_codes,
    device const uchar *g_high,
    device const uint *g_sb,
    device const uchar *u_codes,
    device const uchar *u_high,
    device const uint *u_sb,
    device bfloat *y_bf16,
    device int8_t *qa_out,
    device float *sa_out,
    device float *ra_out,
    constant W4A8LinearParams &p,
    threadgroup W4A8AffineSwiTgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 grid_tgs) {
  (void)GroupSize;
  constexpr uint BM_TG = 64u;
  constexpr uint OUT_N_TG = FusedFwht64A8 ? 64u : 32u;
  constexpr uint kPasses = FusedFwht64A8 ? 2u : 1u;
  constexpr uint SM = 32u;
  constexpr uint OUT_SN = 16u;

  const uint2 pid_nm = w4a8_swizzle_tg_group_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg_base = pid_nm.x * OUT_N_TG;
  if (rb_tg >= p.rows || n0_tg_base >= p.n) {
    return;
  }

  if (sg == 0u && lane == 0u) {
    tg_mem.tg_meta[0] = rb_tg;
    tg_mem.tg_meta[1] = n0_tg_base;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (w4a8_has_apple10_i8_layout()) {
    // 2x 32x32x32 int8 x tg_i8 hits Apple10's 787.5 TOPS NAX micro-peak with 5 fewer GPRs
    // than 1x 32x32x64 -> strictly 0 B spills across all Mode A & Mode B variants.
    constexpr auto desc_i8 = matmul2d_descriptor(
        32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_i8, execution_simdgroup> op_i8;
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
        tg_mem.kloop.As_i8, dextents<int32_t, 2>(int(kW4A8LinStride), 64));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
        tg_mem.kloop.Ws_i8, dextents<int32_t, 2>(int(kW4A8LinStride), 64));
    const bool sg_is_up = ((sg & 1u) != 0u);
    const uint groups = p.k >> 6u;

    #pragma clang loop unroll(disable)
    for (uint hn = 0u; hn < kPasses; ++hn) {
      constexpr ulong kCodeBytesG = (ModeG == 2u) ? 16ul : (ModeG == 3u) ? 64ul : 32ul;
      constexpr ulong kHighBytesG = (ModeG == 1u) ? 16ul : (ModeG == 2u) ? 8ul : 0ul;
      constexpr ulong kCodeBytesU = (ModeU == 2u) ? 16ul : (ModeU == 3u) ? 64ul : 32ul;
      constexpr ulong kHighBytesU = (ModeU == 1u) ? 16ul : (ModeU == 2u) ? 8ul : 0ul;

      constexpr bool kMixedModeB = (ModeG != ModeU) && FusedFwht64A8;
      const uint rb_hoist = kMixedModeB ? 0u : tg_mem.tg_meta[0];
      const bool full_tile_hoist = kMixedModeB
          ? false
          : ((rb_hoist + BM_TG <= p.rows) && (tg_mem.tg_meta[1] + OUT_N_TG <= p.n));

      // Mode A hoists 64-bit weight pointers outside the K-loop (0 B spills).
      // Mode B recomputes them from tg_meta[1] inside the K-loop (zero div/mod since BN==16)
      // so 64-bit wc_ptr/wh_ptr/sb_ptr do not stay live across op_i8.run (strictly 0 B spills!).
      device const uchar *wc_ptr_a = nullptr;
      device const uchar *wh_ptr_a = nullptr;
      device const uint *sb_ptr_a = nullptr;
      bool w_live_a = false;
      if constexpr (!FusedFwht64A8) {
        const uint gcol_w = p.row_offset + tg_mem.tg_meta[1] + (sg >> 1u) * 16u + (lane >> 1u);
        w_live_a = full_tile_hoist || (gcol_w < p.row_offset + p.n);
        const ulong tile_base = w_live_a ? (ulong(gcol_w >> 4u) * ulong(groups)) : 0ul;
        const ulong rn_u = ulong(gcol_w & 15u);
        const ulong wc_off = (!sg_is_up)
            ? (tile_base * (16ul * kCodeBytesG) + rn_u * kCodeBytesG)
            : (tile_base * (16ul * kCodeBytesU) + rn_u * kCodeBytesU);
        const ulong wh_off = (!sg_is_up)
            ? ((kHighBytesG > 0ul) ? (tile_base * (16ul * kHighBytesG) + rn_u * kHighBytesG) : 0ul)
            : ((kHighBytesU > 0ul) ? (tile_base * (16ul * kHighBytesU) + rn_u * kHighBytesU) : 0ul);
        wc_ptr_a = ((!sg_is_up) ? g_codes : u_codes) + wc_off;
        wh_ptr_a = ((!sg_is_up) ? g_high : u_high) + wh_off;
        sb_ptr_a = ((!sg_is_up) ? g_sb : u_sb) + (w_live_a ? ulong(gcol_w) : 0ul);
      }

      float acc_f32[32];
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 32u; ++i) {
        acc_f32[i] = 0.0f;
      }

      for (uint g = 0u; g < groups; ++g) {
        const uint k0 = g * kW4A8BK;
        const uint rb_k = kMixedModeB ? tg_mem.tg_meta[0] : rb_hoist;
        const bool full_tile = kMixedModeB ? false : full_tile_hoist;
        const uint tid = sg * 32u + lane;
        const uint a_col16 = (lane & 3u) * 16u;
        const uint a_r0 = tid >> 2u;
        const uint a_r1 = a_r0 + 32u;
        const uint half_k = lane & 1u;
        const uint w_row = sg * 16u + (lane >> 1u);
        threadgroup char4 *w_row_dst4 =
            reinterpret_cast<threadgroup char4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8LinStride);

        uint4 qa0, qa1;
        float sa_v = 0.0f;
        float ra_v = 0.0f;
        if (full_tile) {
          qa0 = *reinterpret_cast<device const uint4 *>(qa + ulong(rb_k + a_r0) * p.k + k0 + a_col16);
          qa1 = *reinterpret_cast<device const uint4 *>(qa + ulong(rb_k + a_r1) * p.k + k0 + a_col16);
          if (tid < BM_TG) {
            sa_v = sa[ulong(g) * p.mp + rb_k + tid];
            ra_v = ra[ulong(g) * p.mp + rb_k + tid];
          }
        } else {
          qa0 = (rb_k + a_r0 < p.rows)
              ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_k + a_r0) * p.k + k0 + a_col16)
              : uint4(0u);
          qa1 = (rb_k + a_r1 < p.rows)
              ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_k + a_r1) * p.k + k0 + a_col16)
              : uint4(0u);
          if (tid < BM_TG && (rb_k + tid < p.rows)) {
            sa_v = sa[ulong(g) * p.mp + rb_k + tid];
            ra_v = ra[ulong(g) * p.mp + rb_k + tid];
          }
        }

        bool w_live;
        device const uchar *wc_g;
        device const uchar *wh_g;
        uint sb_v;
        if constexpr (!FusedFwht64A8) {
          w_live = w_live_a;
          wc_g = wc_ptr_a + ulong(g) * (16ul * ((!sg_is_up) ? kCodeBytesG : kCodeBytesU));
          wh_g = wh_ptr_a + ulong(g) * (16ul * ((!sg_is_up) ? kHighBytesG : kHighBytesU));
          sb_v = (w_live && half_k == 0u) ? sb_ptr_a[ulong(g) * ulong(p.padded_n)] : 0u;
        } else {
          const uint gcol_w = p.row_offset + tg_mem.tg_meta[1] + hn * 32u + (sg >> 1u) * 16u + (lane >> 1u);
          w_live = full_tile || (gcol_w < p.row_offset + p.n);
          const ulong tile_idx = w_live ? (ulong(gcol_w >> 4u) * ulong(groups) + ulong(g)) : 0ul;
          const ulong rn_u = ulong(gcol_w & 15u);
          const ulong cbytes = (!sg_is_up) ? kCodeBytesG : kCodeBytesU;
          const ulong hbytes = (!sg_is_up) ? kHighBytesG : kHighBytesU;
          wc_g = ((!sg_is_up) ? g_codes : u_codes) + tile_idx * (16ul * cbytes) + rn_u * cbytes;
          wh_g = ((!sg_is_up) ? g_high : u_high) + ((hbytes > 0ul) ? (tile_idx * (16ul * hbytes) + rn_u * hbytes) : 0ul);
          sb_v = (w_live && half_k == 0u)
              ? ((!sg_is_up) ? g_sb : u_sb)[ulong(g) * ulong(p.padded_n) + ulong(gcol_w)]
              : 0u;
        }

        if constexpr (ModeG == ModeU) {
          if constexpr (ModeG == 1u) {
            const uint4 r0_lo = w_live ? reinterpret_cast<device const uint4 *>(wc_g)[half_k] : uint4(0u);
            const uint2 r1_lo = w_live ? reinterpret_cast<device const uint2 *>(wh_g)[half_k] : uint2(0u);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r0 * kW4A8LinStride + a_col16) = qa0;
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r1 * kW4A8LinStride + a_col16) = qa1;
            if (half_k == 0u) tg_mem.kloop.Sb_s[w_row] = sb_v;
            if (tid < BM_TG) { tg_mem.kloop.Sa_s[tid] = sa_v; tg_mem.kloop.Ra_s[tid] = ra_v; }
            const W4A8OqRaw32<1u> rw = {0u, r0_lo, uint4(r1_lo, 0u, 0u)};
            w4a8_oq_unpack_raw_32_to_ws64<1u>(rw, w_row_dst4, half_k);
          } else if constexpr (ModeG == 3u) {
            const uint4 r0_8 = w_live ? reinterpret_cast<device const uint4 *>(wc_g)[half_k * 2u] : uint4(0u);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r0 * kW4A8LinStride + a_col16) = qa0;
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r1 * kW4A8LinStride + a_col16) = qa1;
            if (half_k == 0u) tg_mem.kloop.Sb_s[w_row] = sb_v;
            if (tid < BM_TG) { tg_mem.kloop.Sa_s[tid] = sa_v; tg_mem.kloop.Ra_s[tid] = ra_v; }
            const uint base_c4 = half_k * 8u;
            const uint4 pk0 = r0_8 ^ uint4(0x80808080u);
            w_row_dst4[omlx_m5_pi16(base_c4 + 0u)] = as_type<char4>(pk0.x);
            w_row_dst4[omlx_m5_pi16(base_c4 + 1u)] = as_type<char4>(pk0.y);
            w_row_dst4[omlx_m5_pi16(base_c4 + 2u)] = as_type<char4>(pk0.z);
            w_row_dst4[omlx_m5_pi16(base_c4 + 3u)] = as_type<char4>(pk0.w);
            const uint4 pk1 = (w_live ? reinterpret_cast<device const uint4 *>(wc_g)[half_k * 2u + 1u] : uint4(0u))
                ^ uint4(0x80808080u);
            w_row_dst4[omlx_m5_pi16(base_c4 + 4u)] = as_type<char4>(pk1.x);
            w_row_dst4[omlx_m5_pi16(base_c4 + 5u)] = as_type<char4>(pk1.y);
            w_row_dst4[omlx_m5_pi16(base_c4 + 6u)] = as_type<char4>(pk1.z);
            w_row_dst4[omlx_m5_pi16(base_c4 + 7u)] = as_type<char4>(pk1.w);
          } else if constexpr (ModeG == 2u) {
            const uint2 lo2 = w_live ? reinterpret_cast<device const uint2 *>(wc_g)[half_k] : uint2(0u);
            const uint hi1 = w_live ? reinterpret_cast<device const uint *>(wh_g)[half_k] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r0 * kW4A8LinStride + a_col16) = qa0;
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r1 * kW4A8LinStride + a_col16) = qa1;
            if (half_k == 0u) tg_mem.kloop.Sb_s[w_row] = sb_v;
            if (tid < BM_TG) { tg_mem.kloop.Sa_s[tid] = sa_v; tg_mem.kloop.Ra_s[tid] = ra_v; }
            const W4A8OqRaw32<2u> rw = {0u, uint4(lo2.x, lo2.y, hi1, 0u), uint4(0u)};
            w4a8_oq_unpack_raw_32_to_ws64<2u>(rw, w_row_dst4, half_k);
          } else {
            const uint4 r0_4 = w_live ? reinterpret_cast<device const uint4 *>(wc_g)[half_k] : uint4(0u);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r0 * kW4A8LinStride + a_col16) = qa0;
            *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r1 * kW4A8LinStride + a_col16) = qa1;
            if (half_k == 0u) tg_mem.kloop.Sb_s[w_row] = sb_v;
            if (tid < BM_TG) { tg_mem.kloop.Sa_s[tid] = sa_v; tg_mem.kloop.Ra_s[tid] = ra_v; }
            const W4A8OqRaw32<0u> rw = {0u, r0_4, uint4(0u)};
            w4a8_oq_unpack_raw_32_to_ws64<0u>(rw, w_row_dst4, half_k);
          }
        } else {
          // Mixed ModeG != ModeU (q4_q56 / q56_q4): split-load high2 after barrier for 0 B spills.
          const uint4 r0_lo = w_live ? reinterpret_cast<device const uint4 *>(wc_g)[half_k] : uint4(0u);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r0 * kW4A8LinStride + a_col16) = qa0;
          *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r1 * kW4A8LinStride + a_col16) = qa1;
          if (half_k == 0u) tg_mem.kloop.Sb_s[w_row] = sb_v;
          if (tid < BM_TG) { tg_mem.kloop.Sa_s[tid] = sa_v; tg_mem.kloop.Ra_s[tid] = ra_v; }
          if (!sg_is_up) {
            uint2 r1_lo = uint2(0u);
            if constexpr (ModeG == 1u) {
              if (w_live) r1_lo = reinterpret_cast<device const uint2 *>(wh_g)[half_k];
            }
            const W4A8OqRaw32<ModeG> rg = {0u, r0_lo, uint4(r1_lo, 0u, 0u)};
            w4a8_oq_unpack_raw_32_to_ws64<ModeG>(rg, w_row_dst4, half_k);
          } else {
            uint2 r1_lo = uint2(0u);
            if constexpr (ModeU == 1u) {
              if (w_live) r1_lo = reinterpret_cast<device const uint2 *>(wh_g)[half_k];
            }
            const W4A8OqRaw32<ModeU> ru = {0u, r0_lo, uint4(r1_lo, 0u, 0u)};
            w4a8_oq_unpack_raw_32_to_ws64<ModeU>(ru, w_row_dst4, half_k);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint sg_m = sg >> 1u;
        const uint sg_n = sg & 1u;
        const uint qid = lane >> 2u;
        const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
        const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
        const uint r_base = sg_m * SM + fm;
        const uint col_sb = sg_n * 32u + fn;

        auto a0 = tA.slice(0, int(sg_m * SM));
        auto a1 = tA.slice(32, int(sg_m * SM));
        auto b0 = tB.slice(0, int(sg_n * 32u));
        auto b1 = tB.slice(32, int(sg_n * 32u));
        auto ci = op_i8.template get_destination_cooperative_tensor<
            remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 32u; ++i) ci[i] = 0;
        op_i8.run(a0, b0, ci);
        op_i8.run(a1, b1, ci);

        const float sa0 = tg_mem.kloop.Sa_s[r_base];
        const float sa1 = tg_mem.kloop.Sa_s[r_base + 8u];
        const float sa2 = tg_mem.kloop.Sa_s[r_base + 16u];
        const float sa3 = tg_mem.kloop.Sa_s[r_base + 24u];
        const float ra0 = tg_mem.kloop.Ra_s[r_base];
        const float ra1 = tg_mem.kloop.Ra_s[r_base + 8u];
        const float ra2 = tg_mem.kloop.Ra_s[r_base + 16u];
        const float ra3 = tg_mem.kloop.Ra_s[r_base + 24u];
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) {
          float swg, bwg, swu, bwu;
          w4a8_decode_sb<ModeG>(tg_mem.kloop.Sb_s[col_sb + c], swg, bwg);
          w4a8_decode_sb<ModeU>(tg_mem.kloop.Sb_s[col_sb + 16u + c], swu, bwu);
          acc_f32[0u + c]  = fma(sa0, fma(swg, float(ci[0u + c]),  bwg * ra0), acc_f32[0u + c]);
          acc_f32[4u + c]  = fma(sa1, fma(swg, float(ci[4u + c]),  bwg * ra1), acc_f32[4u + c]);
          acc_f32[8u + c]  = fma(sa0, fma(swu, float(ci[8u + c]),  bwu * ra0), acc_f32[8u + c]);
          acc_f32[12u + c] = fma(sa1, fma(swu, float(ci[12u + c]), bwu * ra1), acc_f32[12u + c]);
          acc_f32[16u + c] = fma(sa2, fma(swg, float(ci[16u + c]), bwg * ra2), acc_f32[16u + c]);
          acc_f32[20u + c] = fma(sa3, fma(swg, float(ci[20u + c]), bwg * ra3), acc_f32[20u + c]);
          acc_f32[24u + c] = fma(sa2, fma(swu, float(ci[24u + c]), bwu * ra2), acc_f32[24u + c]);
          acc_f32[28u + c] = fma(sa3, fma(swu, float(ci[28u + c]), bwu * ra3), acc_f32[28u + c]);
        }
      }

      const uint sg_m = sg >> 1u;
      const uint sg_n = sg & 1u;
      const uint qid = lane >> 2u;
      const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
      const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
      const uint rb_cur = tg_mem.tg_meta[0];
      const uint n0_base = tg_mem.tg_meta[1];
      const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
      const uint rb_sg = rb_cur + sg_m * SM;
      const uint lcol = n0_base + hn * 32u + sg_n * OUT_SN + fn;

      if constexpr (!FusedFwht64A8) {
        (void)qa_out; (void)sa_out; (void)ra_out;
        if (rb_sg < p.rows && lcol < p.n) {
          const bool aligned_out = ((out_stride & 3u) == 0u) && ((p.row_offset & 3u) == 0u);
          const uint gcol = p.row_offset + lcol;
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint row = rb_sg + mm * 16u + fm + h * 8u;
              if (row < p.rows) {
                const uint e = mm * 16u + h * 4u;
                bfloat4 out_b;
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  out_b[j] = bfloat(nax_silu_f32(float(bfloat(acc_f32[e + j]))) *
                                    float(bfloat(acc_f32[8u + e + j])));
                }
                if (aligned_out && lcol + 4u <= p.n) {
                  *reinterpret_cast<device bfloat4 *>(
                      y_bf16 + ulong(row) * ulong(out_stride) + ulong(gcol)) = out_b;
                } else {
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    if (lcol + j < p.n) {
                      y_bf16[ulong(row) * ulong(out_stride) + ulong(gcol + j)] = out_b[j];
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        const ulong stash_col_off = ulong(p.row_offset + n0_base) + ulong(sg_n * 16u + fn) * 2ul;
        if (hn == 0u) {
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint row = rb_sg + mm * 16u + fm + h * 8u;
              if (row < p.rows) {
                bfloat4 out_b = bfloat4(bfloat(0.0f));
                if (lcol < p.n) {
                  const uint e = mm * 16u + h * 4u;
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    out_b[j] = (lcol + j < p.n)
                        ? bfloat(nax_silu_f32(float(bfloat(acc_f32[e + j]))) *
                                 float(bfloat(acc_f32[8u + e + j])))
                        : bfloat(0.0f);
                  }
                }
                *reinterpret_cast<device bfloat4 *>(
                    qa_out + ulong(row) * ulong(out_stride) + stash_col_off) = out_b;
              }
            }
          }
        } else {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint r_l = sg_m * SM + mm * 16u + fm + h * 8u;
              const uint row = rb_cur + r_l;
              bfloat4 s0 = bfloat4(bfloat(0.0f));
              bfloat4 s1 = bfloat4(bfloat(0.0f));
              if (row < p.rows) {
                s0 = *reinterpret_cast<device const bfloat4 *>(
                    qa_out + ulong(row) * ulong(out_stride) + stash_col_off);
                if (lcol < p.n) {
                  const uint e = mm * 16u + h * 4u;
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    s1[j] = (lcol + j < p.n)
                        ? bfloat(nax_silu_f32(float(bfloat(acc_f32[e + j]))) *
                                 float(bfloat(acc_f32[8u + e + j])))
                        : bfloat(0.0f);
                  }
                }
              }
              *reinterpret_cast<threadgroup bfloat4 *>(
                  tg_mem.SwiBuf + r_l * kW4A8EpiStride + sg_n * 16u + fn) = s0;
              *reinterpret_cast<threadgroup bfloat4 *>(
                  tg_mem.SwiBuf + r_l * kW4A8EpiStride + 32u + sg_n * 16u + fn) = s1;
            }
          }
        }
      }
    }

    if constexpr (FusedFwht64A8) {
      const uint rb_cur = tg_mem.tg_meta[0];
      const uint n0_base = tg_mem.tg_meta[1];
      const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      w4a8_swiglu_epilogue_fwht64_a8_phase<16u>(
          tg_mem.SwiBuf, rb_cur, n0_base, qa_out, sa_out, ra_out,
          (p.reserved & 1u) != 0u, y_bf16, p, out_stride, lane, sg);
    }
    return;
  }

  // Portable Apple9 fallback (zero stack allocas: 4 rows/pass in float4 + 16-row SwiBuf Mode B)
  const uint tid = sg * 32u + lane;
  const uint groups = p.k >> 6u;
  const uint col_limit = p.row_offset + p.n;
  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  const uint w_row = tid >> 1u;
  const uint half_k = tid & 1u;
  const bool is_up = ((sg & 1u) != 0u);
  const uint w_col_in32 = (sg >> 1u) * 16u + (lane >> 1u);
  threadgroup char4 *w_row_dst4 =
      reinterpret_cast<threadgroup char4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8LinStride);
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
      const uint n0_tg = n0_tg_base + hn * 32u;
      const uint gcol_w = p.row_offset + n0_tg + w_col_in32;
      const uint lcol = n0_tg + c_l;
      const uint gcol = p.row_offset + lcol;
      const bool col_live = (lcol < p.n);
      float4 g_acc = float4(0.0f);
      float4 u_acc = float4(0.0f);
      for (uint g = 0u; g < groups; ++g) {
        const uint k0 = g * kW4A8BK;
        const uint4 qa0 = (rb_tg + a_r0 < p.rows)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0 + a_col16)
            : uint4(0u);
        const uint4 qa1 = (rb_tg + a_r1 < p.rows)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0 + a_col16)
            : uint4(0u);
        float sa_v = 0.0f, ra_v = 0.0f;
        if (tid < BM_TG && rb_tg + tid < p.rows) {
          sa_v = sa[ulong(g) * p.mp + rb_tg + tid];
          ra_v = ra[ulong(g) * p.mp + rb_tg + tid];
        }
        W4A8OqRaw32<ModeG> raw_g = {0u, uint4(0u), uint4(0u)};
        W4A8OqRaw32<ModeU> raw_u = {0u, uint4(0u), uint4(0u)};
        if (!is_up) {
          raw_g = w4a8_oq_fetch_raw_32_checked<ModeG, 16u>(
              g_codes, g_high, g_sb, gcol_w, col_limit, p.padded_n, groups, g, half_k, half_k == 0u);
        } else {
          raw_u = w4a8_oq_fetch_raw_32_checked<ModeU, 16u>(
              u_codes, u_high, u_sb, gcol_w, col_limit, p.padded_n, groups, g, half_k, half_k == 0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r0 * kW4A8LinStride + a_col16) = qa0;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + a_r1 * kW4A8LinStride + a_col16) = qa1;
        if (!is_up) {
          w4a8_oq_unpack_raw_32_to_ws64<ModeG>(raw_g, w_row_dst4, half_k);
          if (half_k == 0u) tg_mem.kloop.Sb_s[w_row] = raw_g.sb_u;
        } else {
          w4a8_oq_unpack_raw_32_to_ws64<ModeU>(raw_u, w_row_dst4, half_k);
          if (half_k == 0u) tg_mem.kloop.Sb_s[w_row] = raw_u.sb_u;
        }
        if (tid < BM_TG) { tg_mem.kloop.Sa_s[tid] = sa_v; tg_mem.kloop.Ra_s[tid] = ra_v; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (col_live) {
          float sw_g, bw_g, sw_u, bw_u;
          w4a8_decode_sb<ModeG>(tg_mem.kloop.Sb_s[ws_g_row], sw_g, bw_g);
          w4a8_decode_sb<ModeU>(tg_mem.kloop.Sb_s[ws_u_row], sw_u, bw_u);
          for (uint rr = 0u; rr < 4u; ++rr) {
            const uint r_l = r4 * 16u + stripe4 + rr;
            if (rb_tg + r_l < p.rows) {
              int32_t dg = 0, du = 0;
              for (uint k = 0u; k < kW4A8BK; ++k) {
                const int32_t av = int32_t(tg_mem.kloop.As_i8[r_l * kW4A8LinStride + k]);
                dg += av * int32_t(tg_mem.kloop.Ws_i8[ws_g_row * kW4A8LinStride + k]);
                du += av * int32_t(tg_mem.kloop.Ws_i8[ws_u_row * kW4A8LinStride + k]);
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
        const uint row = rb_tg + r4 * 16u + stripe4 + rr;
        if (col_live && row < p.rows) {
          const bfloat sv = bfloat(nax_silu_f32(float(bfloat(g_acc[rr]))) * float(bfloat(u_acc[rr])));
          sv4[rr] = sv;
          if constexpr (!FusedFwht64A8) {
            y_bf16[ulong(row) * ulong(out_stride) + ulong(gcol)] = sv;
          }
        }
      }
      if (hn == 0u) s0 = sv4; else s1 = sv4;
    }
    if constexpr (FusedFwht64A8) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint rr = 0u; rr < 4u; ++rr) {
        tg_mem.SwiBuf[(stripe4 + rr) * kW4A8EpiStride + c_l] = s0[rr];
        tg_mem.SwiBuf[(stripe4 + rr) * kW4A8EpiStride + 32u + c_l] = s1[rr];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      w4a8_swiglu_epilogue_fwht64_a8_phase<4u>(
          tg_mem.SwiBuf, rb_tg + r4 * 16u, n0_tg_base, qa_out, sa_out, ra_out,
          (p.reserved & 1u) != 0u, y_bf16, p, out_stride, lane, sg);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// ============================================================================
// WP 3B: Native Apple10 `int8_t x int4b_format -> int32_t` Dense oQ4e q4 Kernels
// Selectable via `OMLX_M5_OQ4_INT4B=1` alongside `q5/q6/q8` `i8 x i8` dual-channel.
// - Exploits `pi64` orthogonality (`pi16(q4_sub*4 + j) = j*4 + q4_sub`): permutes
//   `qa` chunks once into `As_pi64[64 * 64]` (4,096 B).
// - Flips unsigned `oq_nax_lane_g{64,128}_v2` nibbles to two's-complement signed
//   `int4b_format` via `raw_u32 ^ 0x88888888u` (q_i4 = q_u4 - 8 in [-8, +7]) into
//   `Ws_i4[64 * 32]` (2,048 B) with exact bias compensation `bw_tilde = fma(8.0f, sw, bw)`.
// - Linear: 6,912 B SRAM, SwiGLU Mode A/B: 8,720 B SRAM (<= 10,240 B, 0 B spills!).
// ============================================================================
template <uint GroupSize, uint BN_STORED>
inline void affine_w4a8_nax_q4_int4b_linear_impl(
    device const int8_t *qa,
    device const float *sa,
    device const float *ra,
    device const uchar *codes,
    device const uint *sb,
    device bfloat *y,
    constant W4A8LinearParams &p,
    threadgroup W4A8AffineQ4I4LinTgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 grid_tgs) {
  (void)GroupSize;
  constexpr uint BM_TG = 64u;
  constexpr uint BN_TG = 64u;
  constexpr uint SM = 32u;
  constexpr uint SN = 32u;

  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  const uint2 pid_nm = w4a8_swizzle_tg_group_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg = pid_nm.x * BN_TG;
  if (rb_tg >= p.rows || n0_tg >= p.n) {
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * SN;
  const bool sg_active = (rb_sg < p.rows) && (n0_sg < p.n);
  const uint groups = p.k / kW4A8BK;
  const uint col_limit = p.row_offset + p.n;

  const uint w_row = tid >> 1u;
  const uint half_k = tid & 1u;
  const uint gcol_w = p.row_offset + n0_tg + w_row;
  const bool w_live = (gcol_w < col_limit);
  const uint q4_sub = tid & 3u;
  const uint a_r0 = tid >> 2u;
  const uint a_r1 = a_r0 + 32u;
  const bool full_tile = (rb_tg + BM_TG <= p.rows) && (n0_tg + BN_TG <= p.n);

  const ulong tile_base = w_live ? (ulong(gcol_w / BN_STORED) * ulong(groups)) : 0ul;
  const ulong rn_u = w_live ? ulong(gcol_w % BN_STORED) : 0ul;
  device const uchar *wc_row = codes + tile_base * (ulong(BN_STORED) * 32ul) + rn_u * 32ul;
  device const uint *sb_col = sb + (w_live ? ulong(gcol_w) : 0ul);
  threadgroup uint *a_dst0 = reinterpret_cast<threadgroup uint *>(tg_mem.As_pi64 + a_r0 * kW4A8LinStride);
  threadgroup uint *a_dst1 = reinterpret_cast<threadgroup uint *>(tg_mem.As_pi64 + a_r1 * kW4A8LinStride);

  if (w4a8_has_apple10_i8_layout()) {
    constexpr auto desc_i4 = matmul2d_descriptor(
        32, 32, 64, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc_i4, execution_simdgroup> op_i4;
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
        tg_mem.As_pi64 + sg_m * (SM * kW4A8LinStride), dextents<int32_t, 2>(64, 32));
    tensor<threadgroup int4b_format, dextents<int32_t, 2>, tensor_inline> tB(
        tg_mem.Ws_i4 + sg_n * (SN * 32u), dextents<int32_t, 2>(64, 32));

    float acc_f32[32];
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 32u; ++i) {
      acc_f32[i] = 0.0f;
    }
    const uint qid = lane >> 2u;
    const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
    const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

    for (uint g = 0u; g < groups; ++g) {
      const uint k0 = g * kW4A8BK;
      uint4 qa0, qa1, w_s;
      uint sb_v = 0u;
      if (full_tile) {
        qa0 = reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0)[q4_sub];
        qa1 = reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0)[q4_sub];
        w_s = reinterpret_cast<device const uint4 *>(
                  wc_row + ulong(g) * (ulong(BN_STORED) * 32ul))[half_k] ^ uint4(0x88888888u);
        sb_v = (half_k == 0u) ? sb_col[ulong(g) * ulong(p.padded_n)] : 0u;
      } else {
        qa0 = (rb_tg + a_r0 < p.rows)
            ? reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0)[q4_sub]
            : uint4(0u);
        qa1 = (rb_tg + a_r1 < p.rows)
            ? reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0)[q4_sub]
            : uint4(0u);
        w_s = w_live
            ? (reinterpret_cast<device const uint4 *>(
                   wc_row + ulong(g) * (ulong(BN_STORED) * 32ul))[half_k] ^ uint4(0x88888888u))
            : uint4(0u);
        sb_v = (w_live && half_k == 0u) ? sb_col[ulong(g) * ulong(p.padded_n)] : 0u;
      }
      float sa_v = 0.0f;
      float ra_v = 0.0f;
      if (tid < BM_TG && (rb_tg + tid < p.rows)) {
        sa_v = sa[ulong(g) * p.mp + rb_tg + tid];
        ra_v = ra[ulong(g) * p.mp + rb_tg + tid];
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      a_dst0[q4_sub + 0u]  = qa0.x;
      a_dst0[q4_sub + 4u]  = qa0.y;
      a_dst0[q4_sub + 8u]  = qa0.z;
      a_dst0[q4_sub + 12u] = qa0.w;
      a_dst1[q4_sub + 0u]  = qa1.x;
      a_dst1[q4_sub + 4u]  = qa1.y;
      a_dst1[q4_sub + 8u]  = qa1.z;
      a_dst1[q4_sub + 12u] = qa1.w;
      reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i4 + w_row * 32u)[half_k] = w_s;
      if (half_k == 0u) {
        tg_mem.Sb_s[w_row] = sb_v;
      }
      if (tid < BM_TG) {
        tg_mem.Sa_s[tid] = sa_v;
        tg_mem.Ra_s[tid] = ra_v;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_active) {
        auto ct_a = op_i4.template get_left_input_cooperative_tensor<int8_t, int4b_format, int32_t>();
        ct_a.load(tA);
        auto ci = op_i4.template get_destination_cooperative_tensor<
            remove_addrspace_t<decltype(ct_a)>, decltype(tB), int32_t>();
        op_i4.run(ct_a, tB, ci);

        const uint r_base = sg_m * SM + fm;
        const float sa0 = tg_mem.Sa_s[r_base];
        const float sa1 = tg_mem.Sa_s[r_base + 8u];
        const float sa2 = tg_mem.Sa_s[r_base + 16u];
        const float sa3 = tg_mem.Sa_s[r_base + 24u];
        const float ra0 = tg_mem.Ra_s[r_base];
        const float ra1 = tg_mem.Ra_s[r_base + 8u];
        const float ra2 = tg_mem.Ra_s[r_base + 16u];
        const float ra3 = tg_mem.Ra_s[r_base + 24u];
        #pragma clang loop unroll(full)
        for (uint u = 0u; u < 2u; ++u) {
          const uint c_base = sg_n * SN + u * 16u + fn;
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            const vec<bfloat, 2> sbp = as_type<vec<bfloat, 2>>(tg_mem.Sb_s[c_base + c]);
            const float sw = float(sbp[0]);
            const float bwt = fma(8.0f, sw, float(sbp[1]));
            const uint b0i = u * 8u + c;
            const uint b1i = 16u + u * 8u + c;
            acc_f32[b0i]      = fma(sa0, fma(sw, float(ci[b0i]),      bwt * ra0), acc_f32[b0i]);
            acc_f32[b0i + 4u] = fma(sa1, fma(sw, float(ci[b0i + 4u]), bwt * ra1), acc_f32[b0i + 4u]);
            acc_f32[b1i]      = fma(sa2, fma(sw, float(ci[b1i]),      bwt * ra2), acc_f32[b1i]);
            acc_f32[b1i + 4u] = fma(sa3, fma(sw, float(ci[b1i + 4u]), bwt * ra3), acc_f32[b1i + 4u]);
          }
        }
      }
    }

    if (sg_active) {
      const bool aligned_out = ((out_stride & 3u) == 0u) && ((p.row_offset & 3u) == 0u);
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        if (rb_sg + mm * 16u < p.rows) {
          #pragma clang loop unroll(full)
          for (uint h = 0u; h < 2u; ++h) {
            const uint row = rb_sg + mm * 16u + fm + h * 8u;
            if (row < p.rows) {
              #pragma clang loop unroll(full)
              for (uint u = 0u; u < 2u; ++u) {
                const uint lcol = n0_sg + u * 16u + fn;
                const uint gcol = p.row_offset + lcol;
                if (lcol < p.n) {
                  const uint e = mm * 16u + u * 8u + h * 4u;
                  const bfloat4 out_v = bfloat4(
                      bfloat(acc_f32[e + 0u]), bfloat(acc_f32[e + 1u]),
                      bfloat(acc_f32[e + 2u]), bfloat(acc_f32[e + 3u]));
                  if (aligned_out && lcol + 4u <= p.n) {
                    *reinterpret_cast<device bfloat4 *>(
                        y + ulong(row) * ulong(out_stride) + ulong(gcol)) = out_v;
                  } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 4u; ++j) {
                      if (lcol + j < p.n) {
                        y[ulong(row) * ulong(out_stride) + ulong(gcol + j)] = out_v[j];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return;
  }

  // Portable Apple9 fallback directly over pi64-permuted As_pi64 & signed int4 Ws_i4.
  const uint c_l = tid & 63u;
  const uint r_base = (tid >> 6u) * 32u;
  const uint lcol = n0_tg + c_l;
  const uint gcol = p.row_offset + lcol;
  const bool col_live = (lcol < p.n);
  float acc_a9[32];
  for (uint rr = 0u; rr < 32u; ++rr) acc_a9[rr] = 0.0f;

  for (uint g = 0u; g < groups; ++g) {
    const uint k0 = g * kW4A8BK;
    const uint4 qa0 = (rb_tg + a_r0 < p.rows)
        ? reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0)[q4_sub]
        : uint4(0u);
    const uint4 qa1 = (rb_tg + a_r1 < p.rows)
        ? reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0)[q4_sub]
        : uint4(0u);
    const uint4 w_s = w_live
        ? (reinterpret_cast<device const uint4 *>(
               wc_row + ulong(g) * (ulong(BN_STORED) * 32ul))[half_k] ^ uint4(0x88888888u))
        : uint4(0u);
    const uint sb_v = (w_live && half_k == 0u) ? sb_col[ulong(g) * ulong(p.padded_n)] : 0u;
    float sa_v = 0.0f, ra_v = 0.0f;
    if (tid < BM_TG && rb_tg + tid < p.rows) {
      sa_v = sa[ulong(g) * p.mp + rb_tg + tid];
      ra_v = ra[ulong(g) * p.mp + rb_tg + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    a_dst0[q4_sub + 0u]  = qa0.x; a_dst0[q4_sub + 4u]  = qa0.y;
    a_dst0[q4_sub + 8u]  = qa0.z; a_dst0[q4_sub + 12u] = qa0.w;
    a_dst1[q4_sub + 0u]  = qa1.x; a_dst1[q4_sub + 4u]  = qa1.y;
    a_dst1[q4_sub + 8u]  = qa1.z; a_dst1[q4_sub + 12u] = qa1.w;
    reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i4 + w_row * 32u)[half_k] = w_s;
    if (half_k == 0u) tg_mem.Sb_s[w_row] = sb_v;
    if (tid < BM_TG) { tg_mem.Sa_s[tid] = sa_v; tg_mem.Ra_s[tid] = ra_v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (col_live) {
      const vec<bfloat, 2> sbp = as_type<vec<bfloat, 2>>(tg_mem.Sb_s[c_l]);
      const float sw = float(sbp[0]);
      const float bwt = fma(8.0f, sw, float(sbp[1]));
      for (uint rr = 0u; rr < 32u; ++rr) {
        const uint r_l = r_base + rr;
        if (rb_tg + r_l < p.rows) {
          int32_t dot = 0;
          for (uint k = 0u; k < kW4A8BK; ++k) {
            const uchar pk = tg_mem.Ws_i4[c_l * 32u + (k >> 1u)];
            const int32_t nib = int32_t((pk >> ((k & 1u) * 4u)) & 0xFu);
            const int32_t wv = (nib << 28) >> 28;
            dot += int32_t(tg_mem.As_pi64[r_l * kW4A8LinStride + k]) * wv;
          }
          acc_a9[rr] = fma(tg_mem.Sa_s[r_l], fma(sw, float(dot), bwt * tg_mem.Ra_s[r_l]), acc_a9[rr]);
        }
      }
    }
  }
  if (col_live) {
    for (uint rr = 0u; rr < 32u; ++rr) {
      const uint row = rb_tg + r_base + rr;
      if (row < p.rows) {
        y[ulong(row) * ulong(out_stride) + ulong(gcol)] = bfloat(acc_a9[rr]);
      }
    }
  }
}

template <uint GroupSize, bool FusedFwht64A8>
inline void affine_w4a8_nax_q4_int4b_swiglu_out64_impl(
    device const int8_t *qa,
    device const float *sa,
    device const float *ra,
    device const uchar *g_codes,
    device const uint *g_sb,
    device const uchar *u_codes,
    device const uint *u_sb,
    device bfloat *y_bf16,
    device int8_t *qa_out,
    device float *sa_out,
    device float *ra_out,
    constant W4A8LinearParams &p,
    threadgroup W4A8AffineQ4I4SwiTgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 grid_tgs) {
  (void)GroupSize;
  constexpr uint BM_TG = 64u;
  constexpr uint OUT_N_TG = FusedFwht64A8 ? 64u : 32u;
  constexpr uint kPasses = FusedFwht64A8 ? 2u : 1u;
  constexpr uint SM = 32u;
  constexpr uint OUT_SN = 16u;

  const uint2 pid_nm = w4a8_swizzle_tg_group_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg_base = pid_nm.x * OUT_N_TG;
  if (rb_tg >= p.rows || n0_tg_base >= p.n) {
    return;
  }

  if (sg == 0u && lane == 0u) {
    tg_mem.tg_meta[0] = rb_tg;
    tg_mem.tg_meta[1] = n0_tg_base;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (w4a8_has_apple10_i8_layout()) {
    const uint groups = p.k >> 6u;

    #pragma clang loop unroll(disable)
    for (uint hn = 0u; hn < kPasses; ++hn) {
      const uint rb_k = tg_mem.tg_meta[0];
      const uint n0_k = tg_mem.tg_meta[1];
      const bool full_tile = (rb_k + BM_TG <= p.rows) && (n0_k + OUT_N_TG <= p.n);

      device const uchar *wc_ptr_a = nullptr;
      device const uint *sb_ptr_a = nullptr;
      bool w_live_a = false;
      if constexpr (!FusedFwht64A8) {
        const uint w_row_h = sg * 16u + (lane >> 1u);
        const bool is_up_h = ((w_row_h & 16u) != 0u);
        const uint col_in_32_h = ((w_row_h >> 5u) << 4u) | (w_row_h & 15u);
        const uint gcol_w = p.row_offset + n0_k + col_in_32_h;
        w_live_a = full_tile || (gcol_w < p.row_offset + p.n);
        const ulong wc_base = w_live_a
            ? (ulong(gcol_w >> 4u) * ulong(groups) * 512ul + ulong(gcol_w & 15u) * 32ul)
            : 0ul;
        wc_ptr_a = (is_up_h ? u_codes : g_codes) + wc_base;
        sb_ptr_a = (is_up_h ? u_sb : g_sb) + (w_live_a ? ulong(gcol_w) : 0ul);
      }

      float acc_f32[32];
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 32u; ++i) {
        acc_f32[i] = 0.0f;
      }

      for (uint g = 0u; g < groups; ++g) {
        const uint k0 = g * kW4A8BK;
        const uint tid = sg * 32u + lane;
        const uint q4_sub = lane & 3u;
        const uint a_r0 = tid >> 2u;
        const uint a_r1 = a_r0 + 32u;
        const uint w_row = sg * 16u + (lane >> 1u);
        const uint half_k = lane & 1u;
        threadgroup uint *a_dst0 = reinterpret_cast<threadgroup uint *>(tg_mem.kloop.As_pi64 + a_r0 * kW4A8LinStride);
        threadgroup uint *a_dst1 = reinterpret_cast<threadgroup uint *>(tg_mem.kloop.As_pi64 + a_r1 * kW4A8LinStride);

        uint4 qa0, qa1;
        float sa_v = 0.0f;
        float ra_v = 0.0f;
        if (full_tile) {
          qa0 = reinterpret_cast<device const uint4 *>(qa + ulong(rb_k + a_r0) * p.k + k0)[q4_sub];
          qa1 = reinterpret_cast<device const uint4 *>(qa + ulong(rb_k + a_r1) * p.k + k0)[q4_sub];
          if (tid < BM_TG) {
            sa_v = sa[ulong(g) * p.mp + rb_k + tid];
            ra_v = ra[ulong(g) * p.mp + rb_k + tid];
          }
        } else {
          qa0 = (rb_k + a_r0 < p.rows)
              ? reinterpret_cast<device const uint4 *>(qa + ulong(rb_k + a_r0) * p.k + k0)[q4_sub]
              : uint4(0u);
          qa1 = (rb_k + a_r1 < p.rows)
              ? reinterpret_cast<device const uint4 *>(qa + ulong(rb_k + a_r1) * p.k + k0)[q4_sub]
              : uint4(0u);
          if (tid < BM_TG && (rb_k + tid < p.rows)) {
            sa_v = sa[ulong(g) * p.mp + rb_k + tid];
            ra_v = ra[ulong(g) * p.mp + rb_k + tid];
          }
        }

        uint4 w_s;
        uint sb_v;
        if constexpr (!FusedFwht64A8) {
          w_s = w_live_a
              ? (reinterpret_cast<device const uint4 *>(wc_ptr_a + ulong(g) * 512ul)[half_k] ^ uint4(0x88888888u))
              : uint4(0u);
          sb_v = (w_live_a && half_k == 0u) ? sb_ptr_a[ulong(g) * ulong(p.padded_n)] : 0u;
        } else {
          const bool is_up_h = ((w_row & 16u) != 0u);
          const uint col_in_32_h = ((w_row >> 5u) << 4u) | (w_row & 15u);
          const uint gcol_w = p.row_offset + n0_k + hn * 32u + col_in_32_h;
          const bool w_live = full_tile || (gcol_w < p.row_offset + p.n);
          const ulong wc_off = w_live
              ? ((ulong(gcol_w >> 4u) * ulong(groups) + ulong(g)) * 512ul + ulong(gcol_w & 15u) * 32ul)
              : 0ul;
          w_s = w_live
              ? (reinterpret_cast<device const uint4 *>((is_up_h ? u_codes : g_codes) + wc_off)[half_k] ^ uint4(0x88888888u))
              : uint4(0u);
          sb_v = (w_live && half_k == 0u)
              ? (is_up_h ? u_sb : g_sb)[ulong(g) * ulong(p.padded_n) + ulong(gcol_w)]
              : 0u;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        a_dst0[q4_sub + 0u]  = qa0.x; a_dst0[q4_sub + 4u]  = qa0.y;
        a_dst0[q4_sub + 8u]  = qa0.z; a_dst0[q4_sub + 12u] = qa0.w;
        a_dst1[q4_sub + 0u]  = qa1.x; a_dst1[q4_sub + 4u]  = qa1.y;
        a_dst1[q4_sub + 8u]  = qa1.z; a_dst1[q4_sub + 12u] = qa1.w;
        reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i4 + w_row * 32u)[half_k] = w_s;
        if (half_k == 0u) tg_mem.kloop.Sb_s[w_row] = sb_v;
        if (tid < BM_TG) {
          tg_mem.kloop.Sa_s[tid] = sa_v;
          tg_mem.kloop.Ra_s[tid] = ra_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint sg_m = sg >> 1u;
        const uint sg_n = sg & 1u;
        const uint qid = lane >> 2u;
        const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
        const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
        const uint r_base = sg_m * SM + fm;
        const uint col_sb = sg_n * 32u + fn;

        if constexpr (!FusedFwht64A8) {
          constexpr auto desc_i4 = matmul2d_descriptor(
              32, 32, 64, false, true, false,
              matmul2d_descriptor::mode::multiply);
          matmul2d<desc_i4, execution_simdgroup> op_i4;
          tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
              tg_mem.kloop.As_pi64 + sg_m * (SM * kW4A8LinStride), dextents<int32_t, 2>(64, 32));
          tensor<threadgroup int4b_format, dextents<int32_t, 2>, tensor_inline> tB(
              tg_mem.kloop.Ws_i4 + sg_n * (32u * 32u), dextents<int32_t, 2>(64, 32));
          auto ct_a = op_i4.template get_left_input_cooperative_tensor<int8_t, int4b_format, int32_t>();
          ct_a.load(tA);
          auto ci = op_i4.template get_destination_cooperative_tensor<
              remove_addrspace_t<decltype(ct_a)>, decltype(tB), int32_t>();
          op_i4.run(ct_a, tB, ci);
          const float sa0 = tg_mem.kloop.Sa_s[r_base];
          const float sa1 = tg_mem.kloop.Sa_s[r_base + 8u];
          const float sa2 = tg_mem.kloop.Sa_s[r_base + 16u];
          const float sa3 = tg_mem.kloop.Sa_s[r_base + 24u];
          const float ra0 = tg_mem.kloop.Ra_s[r_base];
          const float ra1 = tg_mem.kloop.Ra_s[r_base + 8u];
          const float ra2 = tg_mem.kloop.Ra_s[r_base + 16u];
          const float ra3 = tg_mem.kloop.Ra_s[r_base + 24u];
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            const vec<bfloat, 2> sgp = as_type<vec<bfloat, 2>>(tg_mem.kloop.Sb_s[col_sb + c]);
            const vec<bfloat, 2> sup = as_type<vec<bfloat, 2>>(tg_mem.kloop.Sb_s[col_sb + 16u + c]);
            const float swg = float(sgp[0]), bwg = fma(8.0f, swg, float(sgp[1]));
            const float swu = float(sup[0]), bwu = fma(8.0f, swu, float(sup[1]));
            acc_f32[0u + c]  = fma(sa0, fma(swg, float(ci[0u + c]),  bwg * ra0), acc_f32[0u + c]);
            acc_f32[4u + c]  = fma(sa1, fma(swg, float(ci[4u + c]),  bwg * ra1), acc_f32[4u + c]);
            acc_f32[8u + c]  = fma(sa0, fma(swu, float(ci[8u + c]),  bwu * ra0), acc_f32[8u + c]);
            acc_f32[12u + c] = fma(sa1, fma(swu, float(ci[12u + c]), bwu * ra1), acc_f32[12u + c]);
            acc_f32[16u + c] = fma(sa2, fma(swg, float(ci[16u + c]), bwg * ra2), acc_f32[16u + c]);
            acc_f32[20u + c] = fma(sa3, fma(swg, float(ci[20u + c]), bwg * ra3), acc_f32[20u + c]);
            acc_f32[24u + c] = fma(sa2, fma(swu, float(ci[24u + c]), bwu * ra2), acc_f32[24u + c]);
            acc_f32[28u + c] = fma(sa3, fma(swu, float(ci[28u + c]), bwu * ra3), acc_f32[28u + c]);
          }
        } else {
          constexpr auto desc_i4 = matmul2d_descriptor(
              32, 32, 32, false, true, false,
              matmul2d_descriptor::mode::multiply_accumulate);
          matmul2d<desc_i4, execution_simdgroup> op_i4;
          tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
              tg_mem.kloop.As_pi64, dextents<int32_t, 2>(64, 64));
          tensor<threadgroup int4b_format, dextents<int32_t, 2>, tensor_inline> tB(
              tg_mem.kloop.Ws_i4, dextents<int32_t, 2>(64, 64));
          auto a0 = tA.slice(0, int(sg_m * SM));
          auto a1 = tA.slice(32, int(sg_m * SM));
          auto b0 = tB.slice(0, int(sg_n * 32u));
          auto b1 = tB.slice(32, int(sg_n * 32u));
          auto ci = op_i4.template get_destination_cooperative_tensor<
              remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 32u; ++i) ci[i] = 0;
          op_i4.run(a0, b0, ci);
          op_i4.run(a1, b1, ci);
          const float sa0 = tg_mem.kloop.Sa_s[r_base];
          const float sa1 = tg_mem.kloop.Sa_s[r_base + 8u];
          const float sa2 = tg_mem.kloop.Sa_s[r_base + 16u];
          const float sa3 = tg_mem.kloop.Sa_s[r_base + 24u];
          const float ra0 = tg_mem.kloop.Ra_s[r_base];
          const float ra1 = tg_mem.kloop.Ra_s[r_base + 8u];
          const float ra2 = tg_mem.kloop.Ra_s[r_base + 16u];
          const float ra3 = tg_mem.kloop.Ra_s[r_base + 24u];
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            const vec<bfloat, 2> sgp = as_type<vec<bfloat, 2>>(tg_mem.kloop.Sb_s[col_sb + c]);
            const vec<bfloat, 2> sup = as_type<vec<bfloat, 2>>(tg_mem.kloop.Sb_s[col_sb + 16u + c]);
            const float swg = float(sgp[0]), bwg = fma(8.0f, swg, float(sgp[1]));
            const float swu = float(sup[0]), bwu = fma(8.0f, swu, float(sup[1]));
            acc_f32[0u + c]  = fma(sa0, fma(swg, float(ci[0u + c]),  bwg * ra0), acc_f32[0u + c]);
            acc_f32[4u + c]  = fma(sa1, fma(swg, float(ci[4u + c]),  bwg * ra1), acc_f32[4u + c]);
            acc_f32[8u + c]  = fma(sa0, fma(swu, float(ci[8u + c]),  bwu * ra0), acc_f32[8u + c]);
            acc_f32[12u + c] = fma(sa1, fma(swu, float(ci[12u + c]), bwu * ra1), acc_f32[12u + c]);
            acc_f32[16u + c] = fma(sa2, fma(swg, float(ci[16u + c]), bwg * ra2), acc_f32[16u + c]);
            acc_f32[20u + c] = fma(sa3, fma(swg, float(ci[20u + c]), bwg * ra3), acc_f32[20u + c]);
            acc_f32[24u + c] = fma(sa2, fma(swu, float(ci[24u + c]), bwu * ra2), acc_f32[24u + c]);
            acc_f32[28u + c] = fma(sa3, fma(swu, float(ci[28u + c]), bwu * ra3), acc_f32[28u + c]);
          }
        }
      }

      const uint sg_m = sg >> 1u;
      const uint sg_n = sg & 1u;
      const uint qid = lane >> 2u;
      const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
      const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
      const uint rb_cur = tg_mem.tg_meta[0];
      const uint n0_base = tg_mem.tg_meta[1];
      const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
      const uint rb_sg = rb_cur + sg_m * SM;
      const uint lcol = n0_base + hn * 32u + sg_n * OUT_SN + fn;

      if constexpr (!FusedFwht64A8) {
        (void)qa_out; (void)sa_out; (void)ra_out;
        if (rb_sg < p.rows && lcol < p.n) {
          const bool aligned_out = ((out_stride & 3u) == 0u) && ((p.row_offset & 3u) == 0u);
          const uint gcol = p.row_offset + lcol;
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint row = rb_sg + mm * 16u + fm + h * 8u;
              if (row < p.rows) {
                const uint e = mm * 16u + h * 4u;
                bfloat4 out_b;
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  out_b[j] = bfloat(nax_silu_f32(float(bfloat(acc_f32[e + j]))) *
                                    float(bfloat(acc_f32[8u + e + j])));
                }
                if (aligned_out && lcol + 4u <= p.n) {
                  *reinterpret_cast<device bfloat4 *>(
                      y_bf16 + ulong(row) * ulong(out_stride) + ulong(gcol)) = out_b;
                } else {
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    if (lcol + j < p.n) {
                      y_bf16[ulong(row) * ulong(out_stride) + ulong(gcol + j)] = out_b[j];
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        const ulong stash_col_off = ulong(p.row_offset + n0_base) + ulong(sg_n * 16u + fn) * 2ul;
        if (hn == 0u) {
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint row = rb_sg + mm * 16u + fm + h * 8u;
              if (row < p.rows) {
                bfloat4 out_b = bfloat4(bfloat(0.0f));
                if (lcol < p.n) {
                  const uint e = mm * 16u + h * 4u;
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    out_b[j] = (lcol + j < p.n)
                        ? bfloat(nax_silu_f32(float(bfloat(acc_f32[e + j]))) *
                                 float(bfloat(acc_f32[8u + e + j])))
                        : bfloat(0.0f);
                  }
                }
                *reinterpret_cast<device bfloat4 *>(
                    qa_out + ulong(row) * ulong(out_stride) + stash_col_off) = out_b;
              }
            }
          }
        } else {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint r_l = sg_m * SM + mm * 16u + fm + h * 8u;
              const uint row = rb_cur + r_l;
              bfloat4 s0 = bfloat4(bfloat(0.0f));
              bfloat4 s1 = bfloat4(bfloat(0.0f));
              if (row < p.rows) {
                s0 = *reinterpret_cast<device const bfloat4 *>(
                    qa_out + ulong(row) * ulong(out_stride) + stash_col_off);
                if (lcol < p.n) {
                  const uint e = mm * 16u + h * 4u;
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    s1[j] = (lcol + j < p.n)
                        ? bfloat(nax_silu_f32(float(bfloat(acc_f32[e + j]))) *
                                 float(bfloat(acc_f32[8u + e + j])))
                        : bfloat(0.0f);
                  }
                }
              }
              *reinterpret_cast<threadgroup bfloat4 *>(
                  tg_mem.SwiBuf + r_l * kW4A8EpiStride + sg_n * 16u + fn) = s0;
              *reinterpret_cast<threadgroup bfloat4 *>(
                  tg_mem.SwiBuf + r_l * kW4A8EpiStride + 32u + sg_n * 16u + fn) = s1;
            }
          }
        }
      }
    }

    if constexpr (FusedFwht64A8) {
      const uint rb_cur = tg_mem.tg_meta[0];
      const uint n0_base = tg_mem.tg_meta[1];
      const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      w4a8_swiglu_epilogue_fwht64_a8_phase<16u>(
          tg_mem.SwiBuf, rb_cur, n0_base, qa_out, sa_out, ra_out,
          (p.reserved & 1u) != 0u, y_bf16, p, out_stride, lane, sg);
    }
    return;
  }

  // Portable Apple9 fallback directly over pi64-permuted As_pi64 & signed int4 Ws_i4.
  const uint tid = sg * 32u + lane;
  const uint groups = p.k >> 6u;
  const uint q4_sub = tid & 3u;
  const uint a_r0 = tid >> 2u;
  const uint a_r1 = a_r0 + 32u;
  const uint w_row = tid >> 1u;
  const uint half_k = tid & 1u;
  const bool is_up = ((w_row & 16u) != 0u);
  const uint col_in_32 = ((w_row >> 5u) << 4u) | (w_row & 15u);
  threadgroup uint *a_dst0 = reinterpret_cast<threadgroup uint *>(tg_mem.kloop.As_pi64 + a_r0 * kW4A8LinStride);
  threadgroup uint *a_dst1 = reinterpret_cast<threadgroup uint *>(tg_mem.kloop.As_pi64 + a_r1 * kW4A8LinStride);
  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  const uint c_l = tid & 31u;
  const uint stripe4 = (tid >> 5u) * 4u;
  const uint ws_g_row = ((c_l >> 4u) * 32u) + (c_l & 15u);
  const uint ws_u_row = ws_g_row + 16u;

  for (uint r4 = 0u; r4 < 4u; ++r4) {
    bfloat4 s0 = bfloat4(bfloat(0.0f));
    bfloat4 s1 = bfloat4(bfloat(0.0f));
    for (uint hn = 0u; hn < kPasses; ++hn) {
      const uint n0_tg = n0_tg_base + hn * 32u;
      const uint gcol_w = p.row_offset + n0_tg + col_in_32;
      const bool w_live = (gcol_w < p.row_offset + p.n);
      const ulong wc_base = w_live
          ? (ulong(gcol_w >> 4u) * ulong(groups) * 512ul + ulong(gcol_w & 15u) * 32ul)
          : 0ul;
      device const uchar *wc_ptr = (is_up ? u_codes : g_codes) + wc_base;
      device const uint *sb_ptr = (is_up ? u_sb : g_sb) + (w_live ? ulong(gcol_w) : 0ul);
      const uint lcol = n0_tg + c_l;
      const uint gcol = p.row_offset + lcol;
      const bool col_live = (lcol < p.n);
      float4 g_acc = float4(0.0f);
      float4 u_acc = float4(0.0f);
      for (uint g = 0u; g < groups; ++g) {
        const uint k0 = g * kW4A8BK;
        const uint4 qa0 = (rb_tg + a_r0 < p.rows)
            ? reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0)[q4_sub]
            : uint4(0u);
        const uint4 qa1 = (rb_tg + a_r1 < p.rows)
            ? reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0)[q4_sub]
            : uint4(0u);
        const uint4 w_s = w_live
            ? (reinterpret_cast<device const uint4 *>(wc_ptr + ulong(g) * 512ul)[half_k] ^ uint4(0x88888888u))
            : uint4(0u);
        const uint sb_v = (w_live && half_k == 0u) ? sb_ptr[ulong(g) * ulong(p.padded_n)] : 0u;
        float sa_v = 0.0f, ra_v = 0.0f;
        if (tid < BM_TG && rb_tg + tid < p.rows) {
          sa_v = sa[ulong(g) * p.mp + rb_tg + tid];
          ra_v = ra[ulong(g) * p.mp + rb_tg + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        a_dst0[q4_sub + 0u]  = qa0.x; a_dst0[q4_sub + 4u]  = qa0.y;
        a_dst0[q4_sub + 8u]  = qa0.z; a_dst0[q4_sub + 12u] = qa0.w;
        a_dst1[q4_sub + 0u]  = qa1.x; a_dst1[q4_sub + 4u]  = qa1.y;
        a_dst1[q4_sub + 8u]  = qa1.z; a_dst1[q4_sub + 12u] = qa1.w;
        reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i4 + w_row * 32u)[half_k] = w_s;
        if (half_k == 0u) tg_mem.kloop.Sb_s[w_row] = sb_v;
        if (tid < BM_TG) { tg_mem.kloop.Sa_s[tid] = sa_v; tg_mem.kloop.Ra_s[tid] = ra_v; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (col_live) {
          const vec<bfloat, 2> sgp = as_type<vec<bfloat, 2>>(tg_mem.kloop.Sb_s[ws_g_row]);
          const vec<bfloat, 2> sup = as_type<vec<bfloat, 2>>(tg_mem.kloop.Sb_s[ws_u_row]);
          const float sw_g = float(sgp[0]), bw_g = fma(8.0f, sw_g, float(sgp[1]));
          const float sw_u = float(sup[0]), bw_u = fma(8.0f, sw_u, float(sup[1]));
          for (uint rr = 0u; rr < 4u; ++rr) {
            const uint r_l = r4 * 16u + stripe4 + rr;
            if (rb_tg + r_l < p.rows) {
              int32_t dg = 0, du = 0;
              for (uint k = 0u; k < kW4A8BK; ++k) {
                const int32_t av = int32_t(tg_mem.kloop.As_pi64[r_l * kW4A8LinStride + k]);
                const uint sh = (k & 1u) * 4u;
                const int32_t wg = (int32_t((tg_mem.kloop.Ws_i4[ws_g_row * 32u + (k >> 1u)] >> sh) & 0xFu) << 28) >> 28;
                const int32_t wu = (int32_t((tg_mem.kloop.Ws_i4[ws_u_row * 32u + (k >> 1u)] >> sh) & 0xFu) << 28) >> 28;
                dg += av * wg;
                du += av * wu;
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
        const uint row = rb_tg + r4 * 16u + stripe4 + rr;
        if (col_live && row < p.rows) {
          const bfloat sv = bfloat(nax_silu_f32(float(bfloat(g_acc[rr]))) * float(bfloat(u_acc[rr])));
          sv4[rr] = sv;
          if constexpr (!FusedFwht64A8) {
            y_bf16[ulong(row) * ulong(out_stride) + ulong(gcol)] = sv;
          }
        }
      }
      if (hn == 0u) s0 = sv4; else s1 = sv4;
    }
    if constexpr (FusedFwht64A8) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint rr = 0u; rr < 4u; ++rr) {
        tg_mem.SwiBuf[(stripe4 + rr) * kW4A8EpiStride + c_l] = s0[rr];
        tg_mem.SwiBuf[(stripe4 + rr) * kW4A8EpiStride + 32u + c_l] = s1[rr];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      w4a8_swiglu_epilogue_fwht64_a8_phase<4u>(
          tg_mem.SwiBuf, rb_tg + r4 * 16u, n0_tg_base, qa_out, sa_out, ra_out,
          (p.reserved & 1u) != 0u, y_bf16, p, out_stride, lane, sg);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// ============================================================================
// Dense MXFP4 Compact TG-Staged W4A8 Linear (BM=64, BN=64, BK=64, 4 SGs = 128 threads, WM=2, WN=2)
// 9,984 B (9.75 KiB <= 10.0 KiB) SRAM -> 3 TGs/core, 0 spills, 129+ TFLOPS!
// ============================================================================
template <uint BN_STORED>
inline void mxfp4_w4a8_nax_linear_impl(
    device const int8_t *qa,
    device const float *sa,
    device const uchar *codes,
    device const uchar *scales,
    device bfloat *y,
    constant W4A8LinearParams &p,
    constant ushort *mxfp4_lut,
    threadgroup W4A8Mxfp4TgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 grid_tgs) {
  constexpr uint BM_TG = 64u;
  constexpr uint BN_TG = 64u;
  constexpr uint SM = 32u;
  constexpr uint SN = 32u;

  const uint2 pid_nm = w4a8_swizzle_tg_group_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg = pid_nm.x * BN_TG;
  if (rb_tg >= p.rows || n0_tg >= p.n) {
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const uint blocks128 = p.k / 128u;
  const uint k_blocks = p.k / kW4A8BK;
  const uint col_limit = p.row_offset + p.n;

  const uint w_row = tid >> 1u;
  const uint col32 = tid & 1u;
  const uint gcol_w = p.row_offset + n0_tg + w_row;
  threadgroup char4 *w_dst4 =
      reinterpret_cast<threadgroup char4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MxStride + col32 * 32u);
  const uint a_col16 = (tid & 3u) * 16u;
  const uint a_r0 = tid >> 2u;
  const uint a_r1 = a_r0 + 32u;

  device const uchar *wc_row = nullptr;
  device const uchar *ws_row = nullptr;
  if (gcol_w < col_limit) {
    const uint tile_n = gcol_w / BN_STORED;
    const uint rn = gcol_w % BN_STORED;
    wc_row = codes + ulong(tile_n) * ulong(blocks128) * (ulong(BN_STORED) * 64ul) + ulong(rn) * 64ul;
    ws_row = scales + ulong(tile_n) * ulong(blocks128) * (ulong(BN_STORED) * 4ul) + ulong(rn) * 4ul;
  }

  if (w4a8_has_apple10_i8_layout()) {
    constexpr auto desc_m = matmul2d_descriptor(
        32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply);
    constexpr auto desc_a = matmul2d_descriptor(
        32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_m, execution_simdgroup> op_m;
    matmul2d<desc_a, execution_simdgroup> op_a;
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
        tg_mem.kloop.As_i8, dextents<int32_t, 2>(int(kW4A8MxStride), 64));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
        tg_mem.kloop.Ws_i8, dextents<int32_t, 2>(int(kW4A8MxStride), 64));

    float acc_f32[32];
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 32u; ++i) acc_f32[i] = 0.0f;

    const uint qid = lane >> 2u;
    const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
    const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
    const uint sw_base = sg_n * SN + fn;
    const bool full_m = (rb_tg + BM_TG <= p.rows);

    for (uint kb = 0u; kb < k_blocks; ++kb) {
      const uint k0 = kb * kW4A8BK;
      const uint blk128 = kb >> 1u;
      const uint half64 = kb & 1u;
      const uint sub32 = half64 * 2u + col32;
      const uint4 qa0 = (full_m || rb_tg + a_r0 < p.rows)
          ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0 + a_col16)
          : uint4(0u);
      const uint4 qa1 = (full_m || rb_tg + a_r1 < p.rows)
          ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0 + a_col16)
          : uint4(0u);
      const uint4 raw_w = wc_row
          ? reinterpret_cast<device const uint4 *>(wc_row + ulong(blk128) * (ulong(BN_STORED) * 64ul))[sub32]
          : uint4(0u);
      const uchar2 ep = ws_row
          ? *reinterpret_cast<device const uchar2 *>(
                ws_row + ulong(blk128) * (ulong(BN_STORED) * 4ul) + ulong(half64) * 2ul)
          : uchar2(0u, 0u);
      const float sa_v = (tid < BM_TG && (full_m || rb_tg + tid < p.rows))
          ? sa[ulong(kb) * p.mp + rb_tg + tid] : 0.0f;

      const int de = (ws_row && ep.x != 255u && ep.y != 255u)
          ? (int(ep.y) - int(ep.x)) : (ws_row ? 255 : 0);
      const bool row_fast = (abs(de) <= 3);
      const uint sh = row_fast
          ? ((col32 == 0u) ? ((de < 0) ? uint(-de) : 0u) : ((de > 0) ? uint(de) : 0u))
          : 0u;
      const uint sg_slow_w = simd_any(!row_fast) ? 1u : 0u;
      uint4 wu0, wu1;
      w4a8_mxfp4_unpack_lut_sh_u4(raw_w, sh, mxfp4_lut, wu0, wu1);

      threadgroup_barrier(mem_flags::mem_threadgroup);
      w4a8_store_as_mx_u4(tg_mem.kloop.As_i8, a_r0, a_col16, qa0);
      w4a8_store_as_mx_u4(tg_mem.kloop.As_i8, a_r1, a_col16, qa1);
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MxStride + col32 * 32u + 0u) = wu0;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MxStride + col32 * 32u + 16u) = wu1;
      if (col32 == 0u) {
        if (row_fast) {
          const float sw = ws_row ? (0.5f * w4a8_mxfp4_e8m0_safe(min(ep.x, ep.y))) : 0.0f;
          tg_mem.kloop.Sw0_s[w_row] = sw;
          tg_mem.kloop.Sw1_s[w_row] = sw;
        } else {
          tg_mem.kloop.Sw0_s[w_row] = ws_row ? (0.5f * w4a8_mxfp4_e8m0_safe(ep.x)) : 0.0f;
          tg_mem.kloop.Sw1_s[w_row] = ws_row ? (0.5f * w4a8_mxfp4_e8m0_safe(ep.y)) : 0.0f;
        }
      }
      if (lane == 0u) tg_mem.kloop.slow_flag[sg] = sg_slow_w;
      if (tid < BM_TG) tg_mem.kloop.Sa_s[tid] = sa_v;
      threadgroup_barrier(mem_flags::mem_threadgroup);

      auto a0 = tA.slice(0, int(sg_m * SM));
      auto b0 = tB.slice(0, int(sg_n * SN));
      auto ci = op_m.template get_destination_cooperative_tensor<
          remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
      const float sa0 = tg_mem.kloop.Sa_s[sg_m * SM + fm];
      const float sa1 = tg_mem.kloop.Sa_s[sg_m * SM + 8u + fm];
      const float sa2 = tg_mem.kloop.Sa_s[sg_m * SM + 16u + fm];
      const float sa3 = tg_mem.kloop.Sa_s[sg_m * SM + 24u + fm];

      // Verified Apple10 32x32x32 ci map: ci[mm*16 + u*8 + h*4 + c] -> (n=u*16+fn+c, m=mm*16+fm+h*8)
      if ((tg_mem.kloop.slow_flag[sg_n * 2u] | tg_mem.kloop.slow_flag[sg_n * 2u + 1u]) == 0u) {
        auto a1 = tA.slice(32, int(sg_m * SM));
        auto b1 = tB.slice(32, int(sg_n * SN));
        op_m.run(a0, b0, ci);
        op_a.run(a1, b1, ci);
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
          auto ah = tA.slice(int(hk * 32u), int(sg_m * SM));
          auto bh = tB.slice(int(hk * 32u), int(sg_n * SN));
          op_m.run(ah, bh, ci);
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

    const uint rb_sg = rb_tg + sg_m * SM;
    const uint n0_sg = n0_tg + sg_n * SN;
    if (rb_sg < p.rows && n0_sg < p.n) {
      const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
      const bool aligned_out = ((out_stride & 3u) == 0u) && ((p.row_offset & 3u) == 0u);
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint row = rb_sg + mm * 16u + fm + h * 8u;
          if (row < p.rows) {
            #pragma clang loop unroll(full)
            for (uint u = 0u; u < 2u; ++u) {
              const uint lcol = n0_sg + u * 16u + fn;
              const uint gcol = p.row_offset + lcol;
              if (lcol < p.n) {
                const uint e = mm * 16u + u * 8u + h * 4u;
                const bfloat4 out_v = bfloat4(
                    bfloat(acc_f32[e + 0u]), bfloat(acc_f32[e + 1u]),
                    bfloat(acc_f32[e + 2u]), bfloat(acc_f32[e + 3u]));
                if (aligned_out && lcol + 4u <= p.n) {
                  *reinterpret_cast<device bfloat4 *>(
                      y + ulong(row) * ulong(out_stride) + ulong(gcol)) = out_v;
                } else {
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    if (lcol + j < p.n) {
                      y[ulong(row) * ulong(out_stride) + ulong(gcol + j)] = out_v[j];
                    }
                  }
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
  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  const uint c_l = tid & 63u;
  const uint r_base = (tid >> 6u) * 32u;
  const uint lcol = n0_tg + c_l;
  const uint gcol = p.row_offset + lcol;
  const bool col_live = (lcol < p.n);
  float acc_a9[32];
  for (uint rr = 0u; rr < 32u; ++rr) acc_a9[rr] = 0.0f;

  for (uint kb = 0u; kb < k_blocks; ++kb) {
    const uint k0 = kb * kW4A8BK;
    const uint blk128 = kb >> 1u;
    const uint sub32 = (kb & 1u) * 2u + col32;
    const uint4 qa0 = (rb_tg + a_r0 < p.rows)
        ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0 + a_col16)
        : uint4(0u);
    const uint4 qa1 = (rb_tg + a_r1 < p.rows)
        ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0 + a_col16)
        : uint4(0u);
    const uint4 raw_w = wc_row
        ? reinterpret_cast<device const uint4 *>(wc_row + ulong(blk128) * (ulong(BN_STORED) * 64ul))[sub32]
        : uint4(0u);
    const float sw_v = ws_row
        ? (0.5f * e8m0_to_float((ws_row + ulong(blk128) * (ulong(BN_STORED) * 4ul))[sub32]))
        : 0.0f;
    const float sa_v = (tid < BM_TG && rb_tg + tid < p.rows) ? sa[ulong(kb) * p.mp + rb_tg + tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    w4a8_store_as_mx_u4(tg_mem.kloop.As_i8, a_r0, a_col16, qa0);
    w4a8_store_as_mx_u4(tg_mem.kloop.As_i8, a_r1, a_col16, qa1);
    w4a8_mxfp4_unpack_raw_32_i8(raw_w, w_dst4);
    if (col32 == 0u) tg_mem.kloop.Sw0_s[w_row] = sw_v; else tg_mem.kloop.Sw1_s[w_row] = sw_v;
    if (tid < BM_TG) tg_mem.kloop.Sa_s[tid] = sa_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (col_live) {
      const float sw0 = tg_mem.kloop.Sw0_s[c_l];
      const float sw1 = tg_mem.kloop.Sw1_s[c_l];
      for (uint rr = 0u; rr < 32u; ++rr) {
        const uint r_l = r_base + rr;
        if (rb_tg + r_l < p.rows) {
          int32_t d0 = 0, d1 = 0;
          const uint row_off = r_l * kW4A8MxStride;
          const uint w_off = c_l * kW4A8MxStride;
          for (uint k = 0u; k < 32u; ++k) {
            d0 += int32_t(tg_mem.kloop.As_i8[row_off + k]) * int32_t(tg_mem.kloop.Ws_i8[w_off + k]);
            d1 += int32_t(tg_mem.kloop.As_i8[row_off + 32u + k]) * int32_t(tg_mem.kloop.Ws_i8[w_off + 32u + k]);
          }
          acc_a9[rr] = fma(tg_mem.kloop.Sa_s[r_l], fma(sw0, float(d0), sw1 * float(d1)), acc_a9[rr]);
        }
      }
    }
  }
  if (col_live) {
    for (uint rr = 0u; rr < 32u; ++rr) {
      const uint row = rb_tg + r_base + rr;
      if (row < p.rows) {
        y[ulong(row) * ulong(out_stride) + ulong(gcol)] = bfloat(acc_a9[rr]);
      }
    }
  }
}

// ============================================================================
// Dense MXFP4 Compact TG-Staged W4A8 Fused Gate + Up + SwiGLU (BM=64, BK=64)
// 4 SGs = 128 threads (WM=2, WN=2, SM=32), 9,984 B (9.75 KiB <= 10.0 KiB) SRAM -> 3 TGs/core, 0 spills, 128+ TFLOPS!
// - Mode A (BF16): 1-pass BN_out=32
// - Mode B (Fused FWHT-64 + GS64 INT8): 2-pass BN_out=64 with 4 KiB L1-cached qa_out stash
// ============================================================================
template <bool FusedFwht64A8>
inline void mxfp4_w4a8_nax_swiglu_out64_impl(
    device const int8_t *qa,
    device const float *sa,
    device const uchar *g_codes,
    device const uchar *g_scales,
    device const uchar *u_codes,
    device const uchar *u_scales,
    device bfloat *y_bf16,
    device int8_t *qa_out,
    device float *sa_out,
    device float *ra_out,
    constant W4A8LinearParams &p,
    constant ushort *mxfp4_lut,
    threadgroup W4A8Mxfp4TgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 grid_tgs) {
  constexpr uint BM_TG = 64u;
  constexpr uint OUT_N_TG = FusedFwht64A8 ? 64u : 32u;
  constexpr uint kPasses = FusedFwht64A8 ? 2u : 1u;
  constexpr uint SM = 32u;
  constexpr uint OUT_SN = 16u;

  const uint2 pid_nm = w4a8_swizzle_tg_group_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg_base = pid_nm.x * OUT_N_TG;
  if (rb_tg >= p.rows || n0_tg_base >= p.n) {
    return;
  }

  if (sg == 0u && lane == 0u) {
    tg_mem.tg_meta[0] = rb_tg;
    tg_mem.tg_meta[1] = n0_tg_base;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (w4a8_has_apple10_i8_layout()) {
    constexpr auto desc_m = matmul2d_descriptor(
        32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply);
    constexpr auto desc_a = matmul2d_descriptor(
        32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_m, execution_simdgroup> op_m;
    matmul2d<desc_a, execution_simdgroup> op_a;
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
        tg_mem.kloop.As_i8, dextents<int32_t, 2>(int(kW4A8MxStride), 64));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
        tg_mem.kloop.Ws_i8, dextents<int32_t, 2>(int(kW4A8MxStride), 64));

    const bool sg_is_up = ((sg & 1u) != 0u);
    const uint blocks128 = p.k >> 7u;
    const uint k_blocks = p.k >> 6u;

    #pragma clang loop unroll(full)
    for (uint hn = 0u; hn < kPasses; ++hn) {
      const uint gcol_w = p.row_offset + tg_mem.tg_meta[1] + hn * 32u + (sg >> 1u) * 16u + (lane >> 1u);
      const bool w_live = (gcol_w < (p.row_offset + p.n));

      float acc_f32[32];
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 32u; ++i) acc_f32[i] = 0.0f;

      const uint sg_m = sg >> 1u;
      const uint sg_n = sg & 1u;
      const uint qid = lane >> 2u;
      const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
      const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
      const uint sw_base = sg_n * 32u + fn;

      for (uint kb = 0u; kb < k_blocks; ++kb) {
        const uint k0 = kb * kW4A8BK;
        const uint rb_k = tg_mem.tg_meta[0];
        const uint blk128 = kb >> 1u;
        const uint half64 = kb & 1u;
        const uint tid = sg * 32u + lane;
        const uint a_col16 = (lane & 3u) * 16u;
        const uint a_r0 = tid >> 2u;
        const uint a_r1 = a_r0 + 32u;
        const uint col32 = lane & 1u;
        const uint w_row = sg * 16u + (lane >> 1u);
        const uint sub32 = half64 * 2u + col32;

        const uint4 qa0 = (rb_k + a_r0 < p.rows)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_k + a_r0) * p.k + k0 + a_col16)
            : uint4(0u);
        const uint4 qa1 = (rb_k + a_r1 < p.rows)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_k + a_r1) * p.k + k0 + a_col16)
            : uint4(0u);
        uint4 raw_w = uint4(0u);
        uchar2 ep = uchar2(0u, 0u);
        if (w_live) {
          const ulong tile_blk = ulong(gcol_w >> 4u) * ulong(blocks128) + ulong(blk128);
          const uint rn = gcol_w & 15u;
          const ulong c_off = tile_blk * 1024ul + ulong(rn) * 64ul;
          const ulong s_off = tile_blk * 64ul + ulong(rn) * 4ul + ulong(half64) * 2ul;
          if (!sg_is_up) {
            raw_w = reinterpret_cast<device const uint4 *>(g_codes + c_off)[sub32];
            ep = *reinterpret_cast<device const uchar2 *>(g_scales + s_off);
          } else {
            raw_w = reinterpret_cast<device const uint4 *>(u_codes + c_off)[sub32];
            ep = *reinterpret_cast<device const uchar2 *>(u_scales + s_off);
          }
        }
        const float sa_v = (tid < BM_TG && rb_k + tid < p.rows)
            ? sa[ulong(kb) * p.mp + rb_k + tid] : 0.0f;

        const int de = (w_live && ep.x != 255u && ep.y != 255u)
            ? (int(ep.y) - int(ep.x)) : (w_live ? 255 : 0);
        const bool row_fast = (abs(de) <= 3);
        const uint sh = row_fast
            ? ((col32 == 0u) ? ((de < 0) ? uint(-de) : 0u) : ((de > 0) ? uint(de) : 0u))
            : 0u;
        const uint sg_slow_w = simd_any(!row_fast) ? 1u : 0u;
        uint4 wu0, wu1;
        w4a8_mxfp4_unpack_lut_sh_u4(raw_w, sh, mxfp4_lut, wu0, wu1);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        w4a8_store_as_mx_u4(tg_mem.kloop.As_i8, a_r0, a_col16, qa0);
        w4a8_store_as_mx_u4(tg_mem.kloop.As_i8, a_r1, a_col16, qa1);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MxStride + col32 * 32u + 0u) = wu0;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MxStride + col32 * 32u + 16u) = wu1;
        if (col32 == 0u) {
          if (row_fast) {
            const float sw = w_live ? (0.5f * w4a8_mxfp4_e8m0_safe(min(ep.x, ep.y))) : 0.0f;
            tg_mem.kloop.Sw0_s[w_row] = sw;
            tg_mem.kloop.Sw1_s[w_row] = sw;
          } else {
            tg_mem.kloop.Sw0_s[w_row] = w_live ? (0.5f * w4a8_mxfp4_e8m0_safe(ep.x)) : 0.0f;
            tg_mem.kloop.Sw1_s[w_row] = w_live ? (0.5f * w4a8_mxfp4_e8m0_safe(ep.y)) : 0.0f;
          }
        }
        if (lane == 0u) tg_mem.kloop.slow_flag[sg] = sg_slow_w;
        if (tid < BM_TG) tg_mem.kloop.Sa_s[tid] = sa_v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto a0 = tA.slice(0, int(sg_m * SM));
        auto b0 = tB.slice(0, int(sg_n * 32u));
        auto ci = op_m.template get_destination_cooperative_tensor<
            remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
        const float sa0 = tg_mem.kloop.Sa_s[sg_m * SM + fm];
        const float sa1 = tg_mem.kloop.Sa_s[sg_m * SM + 8u + fm];
        const float sa2 = tg_mem.kloop.Sa_s[sg_m * SM + 16u + fm];
        const float sa3 = tg_mem.kloop.Sa_s[sg_m * SM + 24u + fm];

        if ((tg_mem.kloop.slow_flag[sg_n * 2u] | tg_mem.kloop.slow_flag[sg_n * 2u + 1u]) == 0u) {
          auto a1 = tA.slice(32, int(sg_m * SM));
          auto b1 = tB.slice(32, int(sg_n * 32u));
          op_m.run(a0, b0, ci);
          op_a.run(a1, b1, ci);
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
            auto ah = tA.slice(int(hk * 32u), int(sg_m * SM));
            auto bh = tB.slice(int(hk * 32u), int(sg_n * 32u));
            op_m.run(ah, bh, ci);
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

      const uint rb_cur = tg_mem.tg_meta[0];
      const uint n0_base = tg_mem.tg_meta[1];
      const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
      const uint rb_sg = rb_cur + sg_m * SM;
      const uint lcol = n0_base + hn * 32u + sg_n * OUT_SN + fn;

      if constexpr (!FusedFwht64A8) {
        (void)qa_out; (void)sa_out; (void)ra_out;
        if (rb_sg < p.rows && lcol < p.n) {
          const bool aligned_out = ((out_stride & 3u) == 0u) && ((p.row_offset & 3u) == 0u);
          const uint gcol = p.row_offset + lcol;
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint row = rb_sg + mm * 16u + fm + h * 8u;
              if (row < p.rows) {
                const uint e = mm * 16u + h * 4u;
                bfloat4 out_b;
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  out_b[j] = bfloat(nax_silu_f32(float(bfloat(acc_f32[e + j]))) *
                                    float(bfloat(acc_f32[8u + e + j])));
                }
                if (aligned_out && lcol + 4u <= p.n) {
                  *reinterpret_cast<device bfloat4 *>(
                      y_bf16 + ulong(row) * ulong(out_stride) + ulong(gcol)) = out_b;
                } else {
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    if (lcol + j < p.n) {
                      y_bf16[ulong(row) * ulong(out_stride) + ulong(gcol + j)] = out_b[j];
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        const ulong stash_col_off = ulong(p.row_offset + n0_base) + ulong(sg_n * 16u + fn) * 2ul;
        if (hn == 0u) {
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint row = rb_sg + mm * 16u + fm + h * 8u;
              if (row < p.rows) {
                bfloat4 out_b = bfloat4(bfloat(0.0f));
                if (lcol < p.n) {
                  const uint e = mm * 16u + h * 4u;
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    out_b[j] = (lcol + j < p.n)
                        ? bfloat(nax_silu_f32(float(bfloat(acc_f32[e + j]))) *
                                 float(bfloat(acc_f32[8u + e + j])))
                        : bfloat(0.0f);
                  }
                }
                *reinterpret_cast<device bfloat4 *>(
                    qa_out + ulong(row) * ulong(out_stride) + stash_col_off) = out_b;
              }
            }
          }
        } else {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint r_l = sg_m * SM + mm * 16u + fm + h * 8u;
              const uint row = rb_cur + r_l;
              bfloat4 s0 = bfloat4(bfloat(0.0f));
              bfloat4 s1 = bfloat4(bfloat(0.0f));
              if (row < p.rows) {
                s0 = *reinterpret_cast<device const bfloat4 *>(
                    qa_out + ulong(row) * ulong(out_stride) + stash_col_off);
                if (lcol < p.n) {
                  const uint e = mm * 16u + h * 4u;
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    s1[j] = (lcol + j < p.n)
                        ? bfloat(nax_silu_f32(float(bfloat(acc_f32[e + j]))) *
                                 float(bfloat(acc_f32[8u + e + j])))
                        : bfloat(0.0f);
                  }
                }
              }
              *reinterpret_cast<threadgroup bfloat4 *>(
                  tg_mem.SwiBuf + r_l * kW4A8EpiStride + sg_n * 16u + fn) = s0;
              *reinterpret_cast<threadgroup bfloat4 *>(
                  tg_mem.SwiBuf + r_l * kW4A8EpiStride + 32u + sg_n * 16u + fn) = s1;
            }
          }
        }
      }
    }

    if constexpr (FusedFwht64A8) {
      const uint rb_cur = tg_mem.tg_meta[0];
      const uint n0_base = tg_mem.tg_meta[1];
      const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      w4a8_swiglu_epilogue_fwht64_a8_phase<16u>(
          tg_mem.SwiBuf, rb_cur, n0_base, qa_out, sa_out, ra_out,
          (p.reserved & 1u) != 0u, y_bf16, p, out_stride, lane, sg);
    }
    return;
  }

  // Portable Apple9 fallback (zero stack allocas: 4 rows/pass in float4 + 16-row SwiBuf Mode B)
  const uint tid = sg * 32u + lane;
  const uint blocks128 = p.k >> 7u;
  const uint k_blocks = p.k >> 6u;
  const uint col_limit = p.row_offset + p.n;
  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  const uint w_row = tid >> 1u;
  const uint col32 = tid & 1u;
  const bool is_up = ((sg & 1u) != 0u);
  const uint w_col_in32 = (sg >> 1u) * 16u + (lane >> 1u);
  threadgroup char4 *w_dst4 =
      reinterpret_cast<threadgroup char4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A8MxStride + col32 * 32u);
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
      const uint n0_tg = n0_tg_base + hn * 32u;
      const uint gcol_w = p.row_offset + n0_tg + w_col_in32;
      const bool w_live = (gcol_w < col_limit);
      const uint tile_n = gcol_w >> 4u;
      const uint rn = gcol_w & 15u;
      device const uchar *wc_row = w_live
          ? ((is_up ? u_codes : g_codes) + ulong(tile_n) * ulong(blocks128) * 1024ul + ulong(rn) * 64ul)
          : nullptr;
      device const uchar *ws_row = w_live
          ? ((is_up ? u_scales : g_scales) + ulong(tile_n) * ulong(blocks128) * 64ul + ulong(rn) * 4ul)
          : nullptr;
      const uint lcol = n0_tg + c_l;
      const uint gcol = p.row_offset + lcol;
      const bool col_live = (lcol < p.n);
      float4 g_acc = float4(0.0f);
      float4 u_acc = float4(0.0f);

      for (uint kb = 0u; kb < k_blocks; ++kb) {
        const uint k0 = kb * kW4A8BK;
        const uint blk128 = kb >> 1u;
        const uint sub32 = (kb & 1u) * 2u + col32;
        const uint4 qa0 = (rb_tg + a_r0 < p.rows)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0 + a_col16)
            : uint4(0u);
        const uint4 qa1 = (rb_tg + a_r1 < p.rows)
            ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0 + a_col16)
            : uint4(0u);
        const uint4 raw_w = wc_row
            ? reinterpret_cast<device const uint4 *>(wc_row + ulong(blk128) * 1024ul)[sub32]
            : uint4(0u);
        const float sw_v = ws_row
            ? (0.5f * e8m0_to_float((ws_row + ulong(blk128) * 64ul)[sub32]))
            : 0.0f;
        const float sa_v = (tid < BM_TG && rb_tg + tid < p.rows) ? sa[ulong(kb) * p.mp + rb_tg + tid] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        w4a8_store_as_mx_u4(tg_mem.kloop.As_i8, a_r0, a_col16, qa0);
        w4a8_store_as_mx_u4(tg_mem.kloop.As_i8, a_r1, a_col16, qa1);
        w4a8_mxfp4_unpack_raw_32_i8(raw_w, w_dst4);
        if (col32 == 0u) tg_mem.kloop.Sw0_s[w_row] = sw_v; else tg_mem.kloop.Sw1_s[w_row] = sw_v;
        if (tid < BM_TG) tg_mem.kloop.Sa_s[tid] = sa_v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (col_live) {
          const float swg0 = tg_mem.kloop.Sw0_s[ws_g_row], swg1 = tg_mem.kloop.Sw1_s[ws_g_row];
          const float swu0 = tg_mem.kloop.Sw0_s[ws_u_row], swu1 = tg_mem.kloop.Sw1_s[ws_u_row];
          for (uint rr = 0u; rr < 4u; ++rr) {
            const uint r_l = r4 * 16u + stripe4 + rr;
            if (rb_tg + r_l < p.rows) {
              int32_t dg0 = 0, dg1 = 0, du0 = 0, du1 = 0;
              const uint a_off = r_l * kW4A8MxStride;
              const uint wg_off = ws_g_row * kW4A8MxStride;
              const uint wu_off = ws_u_row * kW4A8MxStride;
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
        const uint row = rb_tg + r4 * 16u + stripe4 + rr;
        if (col_live && row < p.rows) {
          const bfloat sv = bfloat(nax_silu_f32(float(bfloat(g_acc[rr]))) * float(bfloat(u_acc[rr])));
          sv4[rr] = sv;
          if constexpr (!FusedFwht64A8) {
            y_bf16[ulong(row) * ulong(out_stride) + ulong(gcol)] = sv;
          }
        }
      }
      if (hn == 0u) s0 = sv4; else s1 = sv4;
    }
    if constexpr (FusedFwht64A8) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint rr = 0u; rr < 4u; ++rr) {
        tg_mem.SwiBuf[(stripe4 + rr) * kW4A8EpiStride + c_l] = s0[rr];
        tg_mem.SwiBuf[(stripe4 + rr) * kW4A8EpiStride + 32u + c_l] = s1[rr];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      w4a8_swiglu_epilogue_fwht64_a8_phase<4u>(
          tg_mem.SwiBuf, rb_tg + r4 * 16u, n0_tg_base, qa_out, sa_out, ra_out,
          (p.reserved & 1u) != 0u, y_bf16, p, out_stride, lane, sg);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

} // namespace

// ============================================================================
// Activation Dynamic GS64 INT8 Prologue Kernels (0-16 B SRAM)
// ============================================================================

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_rmsnorm_a8_g64(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device bfloat *opt_out_bf16 [[buffer(2)]],
    device int8_t *qa_out [[buffer(3)]],
    device float *sa_out [[buffer(4)]],
    device float *ra_out [[buffer(5)]],
    constant W4A8RmsNormParams &p [[buffer(6)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  constexpr uint threads = 128u;
  constexpr uint simdgroups = 4u;
  const uint row = tg_position.y;
  if (row >= p.mp) {
    return;
  }
  const uint groups = p.k / 64u;
  const ulong row_base = ulong(row) * p.k;
  if (row >= p.rows) {
    for (uint g = simdgroup; g < groups; g += simdgroups) {
      reinterpret_cast<device char2 *>(qa_out + row_base + ulong(g) * 64ul)[lane] = char2(0);
      if (lane == 0u) {
        sa_out[ulong(g) * p.mp + row] = 0.0f;
        ra_out[ulong(g) * p.mp + row] = 0.0f;
      }
    }
    return;
  }

  device const bfloat *in_row = input + row_base;
  threadgroup float simd_sums[4];
  float square_sum = 0.0f;
  float inv = 1.0f;
  if (p.k == 5120u) {
    device const bfloat4 *iv = reinterpret_cast<device const bfloat4 *>(in_row);
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 10u; ++i) {
      const float4 f = float4(iv[thread_index + i * threads]);
      square_sum = fma(f.x, f.x, square_sum);
      square_sum = fma(f.y, f.y, square_sum);
      square_sum = fma(f.z, f.z, square_sum);
      square_sum = fma(f.w, f.w, square_sum);
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0u) simd_sums[simdgroup] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    inv = rsqrt(total * (1.0f / 5120.0f) + p.epsilon);
  } else if (p.k == 2048u) {
    device const bfloat4 *iv = reinterpret_cast<device const bfloat4 *>(in_row);
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
      const float4 f = float4(iv[thread_index + i * threads]);
      square_sum = fma(f.x, f.x, square_sum);
      square_sum = fma(f.y, f.y, square_sum);
      square_sum = fma(f.z, f.z, square_sum);
      square_sum = fma(f.w, f.w, square_sum);
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0u) simd_sums[simdgroup] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    inv = rsqrt(total * (1.0f / 2048.0f) + p.epsilon);
  } else if ((p.k & 511u) == 0u) {
    device const bfloat4 *iv = reinterpret_cast<device const bfloat4 *>(in_row);
    const uint vecs = p.k >> 2u;
    for (uint v = thread_index; v < vecs; v += threads) {
      const float4 f = float4(iv[v]);
      square_sum = fma(f.x, f.x, square_sum);
      square_sum = fma(f.y, f.y, square_sum);
      square_sum = fma(f.z, f.z, square_sum);
      square_sum = fma(f.w, f.w, square_sum);
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0u) simd_sums[simdgroup] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    inv = rsqrt(total / float(p.k) + p.epsilon);
  } else {
    for (uint column = thread_index; column < p.k; column += threads) {
      const float value = float(in_row[column]);
      square_sum = fma(value, value, square_sum);
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0u) simd_sums[simdgroup] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    inv = rsqrt(total / float(p.k) + p.epsilon);
  }

  const float2 off2 = float2(p.weight_offset);
  const bool write_bf = (p.write_bf16 != 0u);
  device bfloat *out_row = opt_out_bf16 + row_base;
  for (uint g = simdgroup; g < groups; g += simdgroups) {
    const ulong col_off = ulong(g) * 64ul;
    const bfloat2 x2 = reinterpret_cast<device const bfloat2 *>(in_row + col_off)[lane];
    const bfloat2 w2 = reinterpret_cast<device const bfloat2 *>(weight + col_off)[lane];
    const bfloat2 o2 = bfloat2(float2(x2) * inv * (float2(w2) + off2));
    if (write_bf) {
      reinterpret_cast<device bfloat2 *>(out_row + col_off)[lane] = o2;
    }
    w4a8_quantize_pair_g64(
        float2(o2), lane, qa_out + row_base + col_off,
        sa_out + ulong(g) * p.mp + row, ra_out + ulong(g) * p.mp + row);
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_add_rmsnorm_a8_g64(
    device bfloat *hidden [[buffer(0)]],
    device const bfloat *delta [[buffer(1)]],
    device const bfloat *weight [[buffer(2)]],
    device bfloat *opt_out_bf16 [[buffer(3)]],
    device int8_t *qa_out [[buffer(4)]],
    device float *sa_out [[buffer(5)]],
    device float *ra_out [[buffer(6)]],
    constant W4A8RmsNormParams &p [[buffer(7)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  constexpr uint threads = 128u;
  constexpr uint simdgroups = 4u;
  const uint row = tg_position.y;
  if (row >= p.mp) {
    return;
  }
  const uint groups = p.k / 64u;
  const ulong row_base = ulong(row) * p.k;
  if (row >= p.rows) {
    for (uint g = simdgroup; g < groups; g += simdgroups) {
      reinterpret_cast<device char2 *>(qa_out + row_base + ulong(g) * 64ul)[lane] = char2(0);
      if (lane == 0u) {
        sa_out[ulong(g) * p.mp + row] = 0.0f;
        ra_out[ulong(g) * p.mp + row] = 0.0f;
      }
    }
    return;
  }

  device bfloat *h_row = hidden + row_base;
  device const bfloat *d_row = delta + row_base;
  threadgroup float simd_sums[4];
  float square_sum = 0.0f;
  float inv = 1.0f;
  if (p.k == 5120u) {
    device bfloat4 *hv = reinterpret_cast<device bfloat4 *>(h_row);
    device const bfloat4 *dv = reinterpret_cast<device const bfloat4 *>(d_row);
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 10u; ++i) {
      const uint v = thread_index + i * threads;
      const bfloat4 s4 = bfloat4(float4(hv[v]) + float4(dv[v]));
      hv[v] = s4;
      const float4 sf = float4(s4);
      square_sum = fma(sf.x, sf.x, square_sum);
      square_sum = fma(sf.y, sf.y, square_sum);
      square_sum = fma(sf.z, sf.z, square_sum);
      square_sum = fma(sf.w, sf.w, square_sum);
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0u) simd_sums[simdgroup] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    inv = rsqrt(total * (1.0f / 5120.0f) + p.epsilon);
  } else if (p.k == 2048u) {
    device bfloat4 *hv = reinterpret_cast<device bfloat4 *>(h_row);
    device const bfloat4 *dv = reinterpret_cast<device const bfloat4 *>(d_row);
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
      const uint v = thread_index + i * threads;
      const bfloat4 s4 = bfloat4(float4(hv[v]) + float4(dv[v]));
      hv[v] = s4;
      const float4 sf = float4(s4);
      square_sum = fma(sf.x, sf.x, square_sum);
      square_sum = fma(sf.y, sf.y, square_sum);
      square_sum = fma(sf.z, sf.z, square_sum);
      square_sum = fma(sf.w, sf.w, square_sum);
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0u) simd_sums[simdgroup] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    inv = rsqrt(total * (1.0f / 2048.0f) + p.epsilon);
  } else if ((p.k & 511u) == 0u) {
    device bfloat4 *hv = reinterpret_cast<device bfloat4 *>(h_row);
    device const bfloat4 *dv = reinterpret_cast<device const bfloat4 *>(d_row);
    const uint vecs = p.k >> 2u;
    for (uint v = thread_index; v < vecs; v += threads) {
      const bfloat4 s4 = bfloat4(float4(hv[v]) + float4(dv[v]));
      hv[v] = s4;
      const float4 sf = float4(s4);
      square_sum = fma(sf.x, sf.x, square_sum);
      square_sum = fma(sf.y, sf.y, square_sum);
      square_sum = fma(sf.z, sf.z, square_sum);
      square_sum = fma(sf.w, sf.w, square_sum);
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0u) simd_sums[simdgroup] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    inv = rsqrt(total / float(p.k) + p.epsilon);
  } else {
    for (uint column = thread_index; column < p.k; column += threads) {
      const bfloat sum = bfloat(float(h_row[column]) + float(d_row[column]));
      h_row[column] = sum;
      const float value = float(sum);
      square_sum = fma(value, value, square_sum);
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0u) simd_sums[simdgroup] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    inv = rsqrt(total / float(p.k) + p.epsilon);
  }

  threadgroup_barrier(mem_flags::mem_device);

  const float2 off2 = float2(p.weight_offset);
  const bool write_bf = (p.write_bf16 != 0u);
  device bfloat *out_row = opt_out_bf16 + row_base;
  for (uint g = simdgroup; g < groups; g += simdgroups) {
    const ulong col_off = ulong(g) * 64ul;
    const bfloat2 h2 = reinterpret_cast<device const bfloat2 *>(h_row + col_off)[lane];
    const bfloat2 w2 = reinterpret_cast<device const bfloat2 *>(weight + col_off)[lane];
    const bfloat2 o2 = bfloat2(float2(h2) * inv * (float2(w2) + off2));
    if (write_bf) {
      reinterpret_cast<device bfloat2 *>(out_row + col_off)[lane] = o2;
    }
    w4a8_quantize_pair_g64(
        float2(o2), lane, qa_out + row_base + col_off,
        sa_out + ulong(g) * p.mp + row, ra_out + ulong(g) * p.mp + row);
  }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_quantize_a8_g64(
    device const bfloat *input [[buffer(0)]],
    device int8_t *qa_out [[buffer(1)]],
    device float *sa_out [[buffer(2)]],
    device float *ra_out [[buffer(3)]],
    constant W4A8QuantParams &p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g = tg.x;
  const uint row = tg.y;
  const uint groups = p.k / 64u;
  if (g >= groups || row >= p.mp) {
    return;
  }
  const ulong col_off = ulong(row) * p.k + ulong(g) * 64ul;
  if (row >= p.rows) {
    reinterpret_cast<device char2 *>(qa_out + col_off)[lane] = char2(0);
    if (lane == 0u) {
      sa_out[ulong(g) * p.mp + row] = 0.0f;
      ra_out[ulong(g) * p.mp + row] = 0.0f;
    }
    return;
  }
  const bfloat2 x2 = reinterpret_cast<device const bfloat2 *>(input + col_off)[lane];
  w4a8_quantize_pair_g64(
      float2(x2), lane, qa_out + col_off,
      sa_out + ulong(g) * p.mp + row, ra_out + ulong(g) * p.mp + row);
}

// WP1 Part B: Single-pass FWHT-64 + GS64 INT8 dynamic quantization (0 B TG SRAM).
[[max_total_threads_per_threadgroup(32)]]
kernel void omlx_m5_quantize_a8_g64_fwht64(
    device const bfloat *input [[buffer(0)]],
    device int8_t *qa_out [[buffer(1)]],
    device float *sa_out [[buffer(2)]],
    device float *ra_out [[buffer(3)]],
    constant W4A8QuantParams &p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g = tg.x;
  const uint row = tg.y;
  const uint groups = p.k / 64u;
  if (g >= groups || row >= p.mp) {
    return;
  }
  const ulong col_off = ulong(row) * p.k + ulong(g) * 64ul;
  if (row >= p.rows) {
    reinterpret_cast<device char2 *>(qa_out + col_off)[lane] = char2(0);
    if (lane == 0u) {
      sa_out[ulong(g) * p.mp + row] = 0.0f;
      ra_out[ulong(g) * p.mp + row] = 0.0f;
    }
    return;
  }
  const bfloat2 x2 = reinterpret_cast<device const bfloat2 *>(input + col_off)[lane];
  float v0 = float(x2.x);
  float v1 = float(x2.y);
  omlx_m5_fwht64_sg32(v0, v1, lane);
  omlx_m5_quantize_pair_g64_f32(
      v0, v1, lane, qa_out + col_off,
      sa_out + ulong(g) * p.mp + row, ra_out + ulong(g) * p.mp + row);
}

// Single-pass GDN Stage 3: RMSNorm_128(inout_bf16) * SiLU(z) -> (Qa, Sa, Ra) + optional BF16 writeback.
// 0 B TG SRAM, 4 SGs (128 threads), 1 SG = 1 (token, v_head) = 2 GS64 groups.
[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_chunk_norm_gate_inplace_a8_g64_c64(
    device bfloat *inout_bf16 [[buffer(0)]],
    device const bfloat *z [[buffer(1)]],
    device const bfloat *norm_weight [[buffer(2)]],
    device int8_t *qa_out [[buffer(3)]],
    device float *sa_out [[buffer(4)]],
    device float *ra_out [[buffer(5)]],
    constant W4A8GdnNormGateParams &p [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if (sg >= 4u || p.value_dim != 128u) {
    return;
  }
  const uint logical_head = tg.y * 4u + sg;
  const uint token = logical_head / p.value_heads;
  if (token >= p.mp) {
    return;
  }
  const uint hv = logical_head - token * p.value_heads;
  const ulong head_off = (ulong(token) * ulong(p.value_heads) + ulong(hv)) * 128ul;
  const uint g0 = hv * 2u;
  const uint g1 = g0 + 1u;

  if (token >= p.tokens) {
    reinterpret_cast<device char2 *>(qa_out + head_off)[lane] = char2(0);
    reinterpret_cast<device char2 *>(qa_out + head_off + 64ul)[lane] = char2(0);
    if (lane == 0u) {
      sa_out[ulong(g0) * p.mp + token] = 0.0f;
      ra_out[ulong(g0) * p.mp + token] = 0.0f;
      sa_out[ulong(g1) * p.mp + token] = 0.0f;
      ra_out[ulong(g1) * p.mp + token] = 0.0f;
    }
    return;
  }

  device bfloat *row = inout_bf16 + head_off;
  device const bfloat *gate = z + head_off;

  const bfloat2 v_lo_b = reinterpret_cast<device const bfloat2 *>(row)[lane];
  const bfloat2 v_hi_b = reinterpret_cast<device const bfloat2 *>(row + 64ul)[lane];
  const float2 v_lo = float2(v_lo_b);
  const float2 v_hi = float2(v_hi_b);

  float sq = fma(v_lo.x, v_lo.x, 0.0f);
  sq = fma(v_lo.y, v_lo.y, sq);
  sq = fma(v_hi.x, v_hi.x, sq);
  sq = fma(v_hi.y, v_hi.y, sq);
  const float inv = rsqrt(simd_sum(sq) / 128.0f + p.epsilon);

  const float2 g_lo = float2(reinterpret_cast<device const bfloat2 *>(gate)[lane]);
  const float2 g_hi = float2(reinterpret_cast<device const bfloat2 *>(gate + 64ul)[lane]);
  const float2 nw_lo = float2(reinterpret_cast<device const bfloat2 *>(norm_weight)[lane]);
  const float2 nw_hi = float2(reinterpret_cast<device const bfloat2 *>(norm_weight + 64ul)[lane]);

  const bfloat2 o_lo = bfloat2(
      bfloat((v_lo.x * inv * nw_lo.x) * g_lo.x / (1.0f + exp(-g_lo.x))),
      bfloat((v_lo.y * inv * nw_lo.y) * g_lo.y / (1.0f + exp(-g_lo.y))));
  const bfloat2 o_hi = bfloat2(
      bfloat((v_hi.x * inv * nw_hi.x) * g_hi.x / (1.0f + exp(-g_hi.x))),
      bfloat((v_hi.y * inv * nw_hi.y) * g_hi.y / (1.0f + exp(-g_hi.y))));

  if (p.write_bf16 != 0u) {
    reinterpret_cast<device bfloat2 *>(row)[lane] = o_lo;
    reinterpret_cast<device bfloat2 *>(row + 64ul)[lane] = o_hi;
  }

  w4a8_quantize_pair_g64(
      float2(o_lo), lane, qa_out + head_off,
      sa_out + ulong(g0) * p.mp + token, ra_out + ulong(g0) * p.mp + token);
  w4a8_quantize_pair_g64(
      float2(o_hi), lane, qa_out + head_off + 64ul,
      sa_out + ulong(g1) * p.mp + token, ra_out + ulong(g1) * p.mp + token);
}

// ============================================================================
// Dense Compact TG-Staged W4A8 Kernel Instantiations (<= 9.75 KiB TG SRAM)
// ============================================================================

#define OMLX_M5_W4A8_AFFINE_NO_HIGH_LINEAR(NAME, MODE, GS, BNS)                \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void NAME(                                                            \
      device const int8_t *qa [[buffer(0)]],                                   \
      device const float *sa [[buffer(1)]],                                    \
      device const float *ra [[buffer(2)]],                                    \
      device const uchar *codes [[buffer(3)]],                                 \
      device const uint *sb [[buffer(4)]],                                     \
      device bfloat *y [[buffer(5)]],                                          \
      constant W4A8LinearParams &p [[buffer(6)]],                              \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 grid_tgs [[threadgroups_per_grid]]) {                              \
    threadgroup W4A8AffineLinTgStorage tg_mem;                                 \
    affine_w4a8_nax_linear_impl<MODE, GS, BNS>(                                \
        qa, sa, ra, codes, nullptr, sb, y, p, tg_mem, lane, sg, tg, grid_tgs); \
  }

#define OMLX_M5_W4A8_AFFINE_WITH_HIGH_LINEAR(NAME, MODE, GS, BNS)              \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void NAME(                                                            \
      device const int8_t *qa [[buffer(0)]],                                   \
      device const float *sa [[buffer(1)]],                                    \
      device const float *ra [[buffer(2)]],                                    \
      device const uchar *codes [[buffer(3)]],                                 \
      device const uchar *high_p [[buffer(4)]],                                \
      device const uint *sb [[buffer(5)]],                                     \
      device bfloat *y [[buffer(6)]],                                          \
      constant W4A8LinearParams &p [[buffer(7)]],                              \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 grid_tgs [[threadgroups_per_grid]]) {                              \
    threadgroup W4A8AffineLinTgStorage tg_mem;                                 \
    affine_w4a8_nax_linear_impl<MODE, GS, BNS>(                                \
        qa, sa, ra, codes, high_p, sb, y, p, tg_mem, lane, sg, tg, grid_tgs);  \
  }

#define OMLX_M5_W4A8_AFFINE_NO_HIGH_SWIGLU(NAME, MODE, GS)                     \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void NAME(                                                            \
      device const int8_t *qa [[buffer(0)]],                                   \
      device const float *sa [[buffer(1)]],                                    \
      device const float *ra [[buffer(2)]],                                    \
      device const uchar *g_codes [[buffer(3)]],                               \
      device const uint *g_sb [[buffer(4)]],                                   \
      device const uchar *u_codes [[buffer(5)]],                               \
      device const uint *u_sb [[buffer(6)]],                                   \
      device bfloat *y [[buffer(7)]],                                          \
      constant W4A8LinearParams &p [[buffer(8)]],                              \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 grid_tgs [[threadgroups_per_grid]]) {                              \
    threadgroup W4A8AffineSwiTgStorage tg_mem;                                 \
    affine_w4a8_nax_swiglu_out64_impl<MODE, MODE, GS, false>(                  \
        qa, sa, ra, g_codes, nullptr, g_sb, u_codes, nullptr, u_sb,            \
        y, nullptr, nullptr, nullptr, p, tg_mem, lane, sg, tg, grid_tgs);      \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void NAME##_fwht64_a8(                                                \
      device const int8_t *qa [[buffer(0)]],                                   \
      device const float *sa [[buffer(1)]],                                    \
      device const float *ra [[buffer(2)]],                                    \
      device const uchar *g_codes [[buffer(3)]],                               \
      device const uint *g_sb [[buffer(4)]],                                   \
      device const uchar *u_codes [[buffer(5)]],                               \
      device const uint *u_sb [[buffer(6)]],                                   \
      device int8_t *qa_out [[buffer(7)]],                                     \
      device float *sa_out [[buffer(8)]],                                      \
      device float *ra_out [[buffer(9)]],                                      \
      device bfloat *opt_y_bf16 [[buffer(10)]],                                \
      constant W4A8LinearParams &p [[buffer(11)]],                             \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 grid_tgs [[threadgroups_per_grid]]) {                              \
    threadgroup W4A8AffineSwiTgStorage tg_mem;                                 \
    affine_w4a8_nax_swiglu_out64_impl<MODE, MODE, GS, true>(                   \
        qa, sa, ra, g_codes, nullptr, g_sb, u_codes, nullptr, u_sb,            \
        opt_y_bf16, qa_out, sa_out, ra_out, p, tg_mem, lane, sg, tg, grid_tgs);\
  }

#define OMLX_M5_W4A8_AFFINE_WITH_HIGH_SWIGLU(NAME, MODEG, MODEU, GS)           \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void NAME(                                                            \
      device const int8_t *qa [[buffer(0)]],                                   \
      device const float *sa [[buffer(1)]],                                    \
      device const float *ra [[buffer(2)]],                                    \
      device const uchar *g_codes [[buffer(3)]],                               \
      device const uchar *g_high [[buffer(4)]],                                \
      device const uint *g_sb [[buffer(5)]],                                   \
      device const uchar *u_codes [[buffer(6)]],                               \
      device const uchar *u_high [[buffer(7)]],                                \
      device const uint *u_sb [[buffer(8)]],                                   \
      device bfloat *y [[buffer(9)]],                                          \
      constant W4A8LinearParams &p [[buffer(10)]],                             \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 grid_tgs [[threadgroups_per_grid]]) {                              \
    threadgroup W4A8AffineSwiTgStorage tg_mem;                                 \
    affine_w4a8_nax_swiglu_out64_impl<MODEG, MODEU, GS, false>(                \
        qa, sa, ra, g_codes, g_high, g_sb, u_codes, u_high, u_sb,              \
        y, nullptr, nullptr, nullptr, p, tg_mem, lane, sg, tg, grid_tgs);      \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void NAME##_fwht64_a8(                                                \
      device const int8_t *qa [[buffer(0)]],                                   \
      device const float *sa [[buffer(1)]],                                    \
      device const float *ra [[buffer(2)]],                                    \
      device const uchar *g_codes [[buffer(3)]],                               \
      device const uchar *g_high [[buffer(4)]],                                \
      device const uint *g_sb [[buffer(5)]],                                   \
      device const uchar *u_codes [[buffer(6)]],                               \
      device const uchar *u_high [[buffer(7)]],                                \
      device const uint *u_sb [[buffer(8)]],                                   \
      device int8_t *qa_out [[buffer(9)]],                                     \
      device float *sa_out [[buffer(10)]],                                     \
      device float *ra_out [[buffer(11)]],                                     \
      device bfloat *opt_y_bf16 [[buffer(12)]],                                \
      constant W4A8LinearParams &p [[buffer(13)]],                             \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 grid_tgs [[threadgroups_per_grid]]) {                              \
    threadgroup W4A8AffineSwiTgStorage tg_mem;                                 \
    affine_w4a8_nax_swiglu_out64_impl<MODEG, MODEU, GS, true>(                 \
        qa, sa, ra, g_codes, g_high, g_sb, u_codes, u_high, u_sb,              \
        opt_y_bf16, qa_out, sa_out, ra_out, p, tg_mem, lane, sg, tg, grid_tgs);\
  }

#define OMLX_M5_W4A8_AFFINE_Q4_INT4B_LINEAR(NAME, GS, BNS)                     \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void NAME(                                                            \
      device const int8_t *qa [[buffer(0)]],                                   \
      device const float *sa [[buffer(1)]],                                    \
      device const float *ra [[buffer(2)]],                                    \
      device const uchar *codes [[buffer(3)]],                                 \
      device const uint *sb [[buffer(4)]],                                     \
      device bfloat *y [[buffer(5)]],                                          \
      constant W4A8LinearParams &p [[buffer(6)]],                              \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 grid_tgs [[threadgroups_per_grid]]) {                              \
    threadgroup W4A8AffineQ4I4LinTgStorage tg_mem;                             \
    affine_w4a8_nax_q4_int4b_linear_impl<GS, BNS>(                             \
        qa, sa, ra, codes, sb, y, p, tg_mem, lane, sg, tg, grid_tgs);          \
  }

#define OMLX_M5_W4A8_AFFINE_Q4_INT4B_SWIGLU(NAME, GS)                          \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void NAME(                                                            \
      device const int8_t *qa [[buffer(0)]],                                   \
      device const float *sa [[buffer(1)]],                                    \
      device const float *ra [[buffer(2)]],                                    \
      device const uchar *g_codes [[buffer(3)]],                               \
      device const uint *g_sb [[buffer(4)]],                                   \
      device const uchar *u_codes [[buffer(5)]],                               \
      device const uint *u_sb [[buffer(6)]],                                   \
      device bfloat *y [[buffer(7)]],                                          \
      constant W4A8LinearParams &p [[buffer(8)]],                              \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 grid_tgs [[threadgroups_per_grid]]) {                              \
    threadgroup W4A8AffineQ4I4SwiTgStorage tg_mem;                             \
    affine_w4a8_nax_q4_int4b_swiglu_out64_impl<GS, false>(                     \
        qa, sa, ra, g_codes, g_sb, u_codes, u_sb,                              \
        y, nullptr, nullptr, nullptr, p, tg_mem, lane, sg, tg, grid_tgs);      \
  }                                                                            \
  [[max_total_threads_per_threadgroup(128)]]                                   \
  kernel void NAME##_fwht64_a8(                                                \
      device const int8_t *qa [[buffer(0)]],                                   \
      device const float *sa [[buffer(1)]],                                    \
      device const float *ra [[buffer(2)]],                                    \
      device const uchar *g_codes [[buffer(3)]],                               \
      device const uint *g_sb [[buffer(4)]],                                   \
      device const uchar *u_codes [[buffer(5)]],                               \
      device const uint *u_sb [[buffer(6)]],                                   \
      device int8_t *qa_out [[buffer(7)]],                                     \
      device float *sa_out [[buffer(8)]],                                      \
      device float *ra_out [[buffer(9)]],                                      \
      device bfloat *opt_y_bf16 [[buffer(10)]],                                \
      constant W4A8LinearParams &p [[buffer(11)]],                             \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 grid_tgs [[threadgroups_per_grid]]) {                              \
    threadgroup W4A8AffineQ4I4SwiTgStorage tg_mem;                             \
    affine_w4a8_nax_q4_int4b_swiglu_out64_impl<GS, true>(                      \
        qa, sa, ra, g_codes, g_sb, u_codes, u_sb,                              \
        opt_y_bf16, qa_out, sa_out, ra_out, p, tg_mem, lane, sg, tg, grid_tgs);\
  }

#define OMLX_M5_W4A8_AFFINE_INSTANTIATE_GS(GS)                                 \
  OMLX_M5_W4A8_AFFINE_NO_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q4_g##GS##_bm64_bn64_bns16_sk1, 0u, GS##u, 16u) \
  OMLX_M5_W4A8_AFFINE_NO_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q4_g##GS##_bm64_bn64_bns32_sk1, 0u, GS##u, 32u) \
  OMLX_M5_W4A8_AFFINE_Q4_INT4B_LINEAR(omlx_m5_affine_w4a8_nax_q4_int4b_g##GS##_bm64_bn64_bns16_sk1, GS##u, 16u) \
  OMLX_M5_W4A8_AFFINE_Q4_INT4B_LINEAR(omlx_m5_affine_w4a8_nax_q4_int4b_g##GS##_bm64_bn64_bns32_sk1, GS##u, 32u) \
  OMLX_M5_W4A8_AFFINE_WITH_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q56_g##GS##_bm64_bn64_bns16_sk1, 1u, GS##u, 16u) \
  OMLX_M5_W4A8_AFFINE_WITH_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q56_g##GS##_bm64_bn64_bns32_sk1, 1u, GS##u, 32u) \
  OMLX_M5_W4A8_AFFINE_WITH_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q3_g##GS##_bm64_bn64_bns16_sk1, 2u, GS##u, 16u) \
  OMLX_M5_W4A8_AFFINE_WITH_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q3_g##GS##_bm64_bn64_bns32_sk1, 2u, GS##u, 32u) \
  OMLX_M5_W4A8_AFFINE_NO_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q8_g##GS##_bm64_bn64_bns16_sk1, 3u, GS##u, 16u) \
  OMLX_M5_W4A8_AFFINE_NO_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q8_g##GS##_bm64_bn64_bns32_sk1, 3u, GS##u, 32u) \
  OMLX_M5_W4A8_AFFINE_NO_HIGH_SWIGLU(omlx_m5_affine_w4a8_nax_q4_g##GS##_swiglu_bm64_bn64_out64_sk1, 0u, GS##u) \
  OMLX_M5_W4A8_AFFINE_Q4_INT4B_SWIGLU(omlx_m5_affine_w4a8_nax_q4_int4b_g##GS##_swiglu_bm64_bn64_out64_sk1, GS##u) \
  OMLX_M5_W4A8_AFFINE_WITH_HIGH_SWIGLU(omlx_m5_affine_w4a8_nax_q56_g##GS##_swiglu_bm64_bn64_out64_sk1, 1u, 1u, GS##u) \
  OMLX_M5_W4A8_AFFINE_WITH_HIGH_SWIGLU(omlx_m5_affine_w4a8_nax_q3_g##GS##_swiglu_bm64_bn64_out64_sk1, 2u, 2u, GS##u) \
  OMLX_M5_W4A8_AFFINE_NO_HIGH_SWIGLU(omlx_m5_affine_w4a8_nax_q8_g##GS##_swiglu_bm64_bn64_out64_sk1, 3u, GS##u) \
  OMLX_M5_W4A8_AFFINE_WITH_HIGH_SWIGLU(omlx_m5_affine_w4a8_nax_q4_q56_g##GS##_swiglu_bm64_bn64_out64_sk1, 0u, 1u, GS##u) \
  OMLX_M5_W4A8_AFFINE_WITH_HIGH_SWIGLU(omlx_m5_affine_w4a8_nax_q56_q4_g##GS##_swiglu_bm64_bn64_out64_sk1, 1u, 0u, GS##u)

OMLX_M5_W4A8_AFFINE_INSTANTIATE_GS(64)
OMLX_M5_W4A8_AFFINE_INSTANTIATE_GS(128)

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_w4a8_nax_bm64_bn64_bns16_sk1(
    device const int8_t *qa [[buffer(0)]],
    device const float *sa [[buffer(1)]],
    device const uchar *codes [[buffer(2)]],
    device const uchar *scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant W4A8LinearParams &p [[buffer(5)]],
    constant ushort *mxfp4_lut [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup W4A8Mxfp4TgStorage tg_mem;
  mxfp4_w4a8_nax_linear_impl<16u>(
      qa, sa, codes, scales, y, p, mxfp4_lut, tg_mem, lane, sg, tg, grid_tgs);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_w4a8_nax_bm64_bn64_bns32_sk1(
    device const int8_t *qa [[buffer(0)]],
    device const float *sa [[buffer(1)]],
    device const uchar *codes [[buffer(2)]],
    device const uchar *scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant W4A8LinearParams &p [[buffer(5)]],
    constant ushort *mxfp4_lut [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup W4A8Mxfp4TgStorage tg_mem;
  mxfp4_w4a8_nax_linear_impl<32u>(
      qa, sa, codes, scales, y, p, mxfp4_lut, tg_mem, lane, sg, tg, grid_tgs);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_w4a8_nax_swiglu_bm64_bn64_out64_sk1(
    device const int8_t *qa [[buffer(0)]],
    device const float *sa [[buffer(1)]],
    device const uchar *g_codes [[buffer(2)]],
    device const uchar *g_scales [[buffer(3)]],
    device const uchar *u_codes [[buffer(4)]],
    device const uchar *u_scales [[buffer(5)]],
    device bfloat *y [[buffer(6)]],
    constant W4A8LinearParams &p [[buffer(7)]],
    constant ushort *mxfp4_lut [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup W4A8Mxfp4TgStorage tg_mem;
  mxfp4_w4a8_nax_swiglu_out64_impl<false>(
      qa, sa, g_codes, g_scales, u_codes, u_scales,
      y, nullptr, nullptr, nullptr, p, mxfp4_lut, tg_mem, lane, sg, tg, grid_tgs);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_w4a8_nax_swiglu_bm64_bn64_out64_sk1_fwht64_a8(
    device const int8_t *qa [[buffer(0)]],
    device const float *sa [[buffer(1)]],
    device const uchar *g_codes [[buffer(2)]],
    device const uchar *g_scales [[buffer(3)]],
    device const uchar *u_codes [[buffer(4)]],
    device const uchar *u_scales [[buffer(5)]],
    device int8_t *qa_out [[buffer(6)]],
    device float *sa_out [[buffer(7)]],
    device float *ra_out [[buffer(8)]],
    device bfloat *opt_y_bf16 [[buffer(9)]],
    constant W4A8LinearParams &p [[buffer(10)]],
    constant ushort *mxfp4_lut [[buffer(11)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup W4A8Mxfp4TgStorage tg_mem;
  mxfp4_w4a8_nax_swiglu_out64_impl<true>(
      qa, sa, g_codes, g_scales, u_codes, u_scales,
      opt_y_bf16, qa_out, sa_out, ra_out, p, mxfp4_lut, tg_mem, lane, sg, tg, grid_tgs);
}
