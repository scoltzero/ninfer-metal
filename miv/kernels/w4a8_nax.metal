// Route A: Prefill W4A8 (INT8 GS64 Dynamic Activation x Lossless Unpacked oQ4e / MXFP4 Weight)
// Metal 4 NAX TensorOps (`matmul2d_descriptor(16, 32, 32)` int8_t x int8_t -> int32_t).
// Zero `.miv` reconversion: directly consumes `oq_nax_tiled_g64_v1` / `oq_nax_tiled_g128_v1`
// (q3/q4/q5/q6/q8) and `mx_nax_tiled_bk128_v1` (MXFP4 E2M1 + UE8M0).
// Supports Dual-ANE output-channel slicing (`row_offset % 64 == 0`, `out_stride`).

#include "common_math.h"
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
  uint out_stride;    // y row stride in elements (0 => row_offset + n)
  uint reserved;
};

namespace {

constant constexpr uint kW4A8BK = 64u;
constant constexpr uint kW4A8StrideI8 = 80u; // 64 + 16 B (16B-aligned, bank-conflict-free)

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
struct W4A8AffineRaw32 {
  uint sb_u;
  uint4 r0;
  uint4 r1;
};

template <uint Mode, uint BN_STORED>
inline W4A8AffineRaw32<Mode> w4a8_affine_fetch_raw_32_fast(
    device const uchar *wc_row,
    device const uchar *wh_row,
    device const uint *sb_col_base,
    uint padded_n,
    uint g,
    uint half_k,
    bool load_sb) {
  W4A8AffineRaw32<Mode> out;
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
inline W4A8AffineRaw32<Mode> w4a8_affine_fetch_raw_32_checked(
    device const uchar *codes,
    device const uchar *high_plane,
    device const uint *sb,
    uint col,
    uint col_limit,
    uint padded_n,
    uint groups,
    uint g,
    uint half_k,
    bool load_sb) {
  W4A8AffineRaw32<Mode> out = {0u, uint4(0u), uint4(0u)};
  if (col >= col_limit) {
    return out;
  }
  const uint tile_n = col / BN_STORED;
  const uint rn = col % BN_STORED;
  const ulong tile_base = ulong(tile_n) * ulong(groups);
  if (load_sb) {
    out.sb_u = sb[ulong(g) * ulong(padded_n) + col];
  }
  if constexpr (Mode == 0u) {
    out.r0 = reinterpret_cast<device const uint4 *>(
        codes + (tile_base + ulong(g)) * (ulong(BN_STORED) * 32ul) + ulong(rn) * 32ul)[half_k];
  } else if constexpr (Mode == 1u) {
    out.r0 = reinterpret_cast<device const uint4 *>(
        codes + (tile_base + ulong(g)) * (ulong(BN_STORED) * 32ul) + ulong(rn) * 32ul)[half_k];
    const uint2 hi2 = reinterpret_cast<device const uint2 *>(
        high_plane + (tile_base + ulong(g)) * (ulong(BN_STORED) * 16ul) + ulong(rn) * 16ul)[half_k];
    out.r1 = uint4(hi2.x, hi2.y, 0u, 0u);
  } else if constexpr (Mode == 2u) {
    const uint2 lo2 = reinterpret_cast<device const uint2 *>(
        codes + (tile_base + ulong(g)) * (ulong(BN_STORED) * 16ul) + ulong(rn) * 16ul)[half_k];
    const uint hi1 = reinterpret_cast<device const uint *>(
        high_plane + (tile_base + ulong(g)) * (ulong(BN_STORED) * 8ul) + ulong(rn) * 8ul)[half_k];
    out.r0 = uint4(lo2.x, lo2.y, hi1, 0u);
  } else {
    device const uint4 *p4 = reinterpret_cast<device const uint4 *>(
        codes + (tile_base + ulong(g)) * (ulong(BN_STORED) * 64ul) + ulong(rn) * 64ul) + half_k * 2u;
    out.r0 = p4[0];
    out.r1 = p4[1];
  }
  return out;
}

template <uint Mode>
inline void w4a8_affine_unpack_raw_32_i8(
    const W4A8AffineRaw32<Mode> raw,
    threadgroup char4 *w_dst4) {
  if constexpr (Mode == 0u) {
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
  } else if constexpr (Mode == 1u) {
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
  } else if constexpr (Mode == 2u) {
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
  } else {
    // q8 unsigned [0..255] -> signed [-128..127] via XOR 0x80; compensated by Bw' = fma(128, Sw, Bw).
    const uint4 s0 = raw.r0 ^ uint4(0x80808080u);
    const uint4 s1 = raw.r1 ^ uint4(0x80808080u);
    #pragma clang loop unroll(full)
    for (uint w = 0u; w < 4u; ++w) {
      w_dst4[w]      = as_type<char4>(s0[w]);
      w_dst4[4u + w] = as_type<char4>(s1[w]);
    }
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

template <uint BN_STORED>
inline void w4a8_mxfp4_fetch_raw_32(
    device const uchar *codes,
    device const uchar *scales,
    uint col,
    uint col_limit,
    uint blocks128,
    uint blk128,
    uint sub32,
    thread uint4 &raw_out,
    thread float &sw_out) {
  if (col < col_limit) {
    const uint tile_n = col / BN_STORED;
    const uint rn = col % BN_STORED;
    const ulong tile_idx = ulong(tile_n) * ulong(blocks128) + ulong(blk128);
    raw_out = reinterpret_cast<device const uint4 *>(
        codes + tile_idx * (ulong(BN_STORED) * 64ul) + ulong(rn) * 64ul)[sub32];
    sw_out = 0.5f * e8m0_to_float(
        scales[tile_idx * (ulong(BN_STORED) * 4ul) + ulong(rn) * 4ul + sub32]);
  } else {
    raw_out = uint4(0u);
    sw_out = 0.0f;
  }
}

inline void w4a8_mxfp4_unpack_raw_32_i8(uint4 raw, threadgroup char4 *w_dst4) {
  #pragma clang loop unroll(full)
  for (uint w = 0u; w < 4u; ++w) {
    const uint wd = raw[w];
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

// ============================================================================
// Dense Affine W4A8 Linear (BM=64, BN=64, BK=64, 4 SGs = 128 threads, 11.5 KiB SRAM)
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
    threadgroup int8_t *As_i8,
    threadgroup int8_t *Ws_i8,
    threadgroup float *Sa_s,
    threadgroup float *Ra_s,
    threadgroup float *R128_s,
    threadgroup float *Sw_s,
    threadgroup float *Bw_s,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM_TG = 64u;
  constexpr uint BN_TG = 64u;
  constexpr uint SM = 32u;
  constexpr uint SN = 32u;

  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  const uint num_pid_n = (p.n + BN_TG - 1u) / BN_TG;
  const uint num_pid_m = (p.mp + BM_TG - 1u) / BM_TG;
  const uint2 pid_nm = w4a8_swizzle_tg_group_m8(tg.xy, num_pid_n, num_pid_m);
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

  constexpr auto desc_i8 = matmul2d_descriptor(
      16, 32, 32, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_i8, execution_simdgroup> op_i8;
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
      As_i8, dextents<int32_t, 2>(int(kW4A8StrideI8), int(BM_TG)));
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
      Ws_i8, dextents<int32_t, 2>(int(kW4A8StrideI8), int(BN_TG)));

  auto a_proto = tA.slice(0, int(sg_m * SM));
  auto b_proto = tB.slice(0, int(sg_n * SN));
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

  const uint w_row = tid >> 1u;
  const uint half_k = tid & 1u;
  const uint global_col = p.row_offset + n0_tg + w_row;
  threadgroup char4 *w_dst4 =
      reinterpret_cast<threadgroup char4 *>(Ws_i8 + w_row * kW4A8StrideI8 + half_k * 32u);

  const uint a_col16 = (tid & 3u) * 16u;
  const uint a_r0 = tid >> 2u;       // 0..31
  const uint a_r1 = a_r0 + 32u;      // 32..63

  const bool full_tile = (rb_tg + BM_TG <= p.rows) && (n0_tg + BN_TG <= p.n);
  constexpr ulong kCodeBytes = (Mode == 2u) ? 16ul : (Mode == 3u) ? 64ul : 32ul;
  constexpr ulong kHighBytes = (Mode == 1u) ? 16ul : (Mode == 2u) ? 8ul : 0ul;
  device const uchar *wc_row = nullptr;
  device const uchar *wh_row = nullptr;
  device const uint *sb_col = nullptr;
  if (full_tile) {
    const uint tile_n = global_col / BN_STORED;
    const uint rn = global_col % BN_STORED;
    const ulong tile_base = ulong(tile_n) * ulong(groups);
    wc_row = codes + tile_base * (ulong(BN_STORED) * kCodeBytes) + ulong(rn) * kCodeBytes;
    wh_row = (kHighBytes > 0ul)
        ? (high_plane + tile_base * (ulong(BN_STORED) * kHighBytes) + ulong(rn) * kHighBytes)
        : nullptr;
    sb_col = sb + global_col;
  }

  for (uint g = 0u; g < groups; ++g) {
    const uint k0 = g * kW4A8BK;
    const bool need_sb = (half_k == 0u) && ((GroupSize == 64u) || ((g & 1u) == 0u));

    uint4 qa0, qa1;
    W4A8AffineRaw32<Mode> raw_w;
    if (full_tile) {
      qa0 = *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0 + a_col16);
      qa1 = *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0 + a_col16);
      raw_w = w4a8_affine_fetch_raw_32_fast<Mode, BN_STORED>(
          wc_row, wh_row, sb_col, p.padded_n, g, half_k, need_sb);
    } else {
      qa0 = (rb_tg + a_r0 < p.rows)
          ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0 + a_col16)
          : uint4(0u);
      qa1 = (rb_tg + a_r1 < p.rows)
          ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0 + a_col16)
          : uint4(0u);
      raw_w = w4a8_affine_fetch_raw_32_checked<Mode, BN_STORED>(
          codes, high_plane, sb, global_col, col_limit, p.padded_n, groups, g, half_k, need_sb);
    }
    float sa_v = 0.0f, ra_v = 0.0f;
    if (tid < BM_TG) {
      const uint r_g = rb_tg + tid;
      if (r_g < p.rows) {
        sa_v = sa[ulong(g) * p.mp + r_g];
        ra_v = ra[ulong(g) * p.mp + r_g];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    *reinterpret_cast<threadgroup uint4 *>(As_i8 + a_r0 * kW4A8StrideI8 + a_col16) = qa0;
    *reinterpret_cast<threadgroup uint4 *>(As_i8 + a_r1 * kW4A8StrideI8 + a_col16) = qa1;
    w4a8_affine_unpack_raw_32_i8<Mode>(raw_w, w_dst4);
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
      float sw, bw;
      w4a8_decode_sb<Mode>(raw_w.sb_u, sw, bw);
      Sw_s[w_row] = sw;
      Bw_s[w_row] = bw;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_active) {
      const int col_base = int(sg_n * SN);
      auto b0 = tB.slice(0, col_base);
      auto b1 = tB.slice(32, col_base);
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        if (full_tile || rb_sg + mm * 16u < p.rows) {
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            acc_i32[i] = 0;
          }
          const int row_base = int(sg_m * SM + mm * 16u);
          auto a0 = tA.slice(0, row_base);
          auto a1 = tA.slice(32, row_base);
          op_i8.run(a0, b0, acc_i32);
          op_i8.run(a1, b1, acc_i32);
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            if (acc_i32.is_valid_element(i)) {
              const auto idx = acc_i32.get_multidimensional_index(i);
              const uint c_l = sg_n * SN + uint(idx[0]);
              const uint r_l = sg_m * SM + mm * 16u + uint(idx[1]);
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

  // Coalesced portable epilogue reusing dead Ws_i8 (5,120 B >= 4 SGs * 16 * 32 * 2 B = 4,096 B).
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg_active) {
    threadgroup bfloat *sg_buf = reinterpret_cast<threadgroup bfloat *>(Ws_i8) + sg * 512u;
    const bool aligned_out = ((out_stride & 3u) == 0u) && ((p.row_offset & 3u) == 0u);
    const uint r_out_base = (lane >> 3u) * 4u; // 0, 4, 8, 12
    const uint c_out4 = (lane & 7u) * 4u;      // 0, 4, ..., 28
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm) {
      if (rb_sg + mm * 16u < p.rows) {
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          if (acc_i32.is_valid_element(i)) {
            const auto idx = acc_i32.get_multidimensional_index(i);
            sg_buf[uint(idx[1]) * 32u + uint(idx[0])] = bfloat(acc_f32[mm][i]);
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint lcol = n0_sg + c_out4;
        const uint gcol = p.row_offset + lcol;
        if (lcol < p.n) {
          #pragma clang loop unroll(full)
          for (uint dr = 0u; dr < 4u; ++dr) {
            const uint r = r_out_base + dr;
            const uint row = rb_sg + mm * 16u + r;
            if (row < p.rows) {
              const bfloat4 v4 = *reinterpret_cast<const threadgroup bfloat4 *>(
                  sg_buf + r * 32u + c_out4);
              if (aligned_out && lcol + 4u <= p.n) {
                *reinterpret_cast<device bfloat4 *>(y + ulong(row) * ulong(out_stride) + ulong(gcol)) = v4;
              } else {
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  if (lcol + j < p.n) {
                    y[ulong(row) * ulong(out_stride) + ulong(gcol + j)] = v4[j];
                  }
                }
              }
            }
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }
}

// ============================================================================
// Dense Affine W4A8 Fused Gate + Up + SwiGLU (BM=64, BN_out=64, BK=64)
// 8 SGs = 256 threads (WM=2, WN=4), SRAM = 16.75 KiB <= 32 KiB
// ============================================================================
template <uint ModeG, uint ModeU, uint GroupSize>
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
    device bfloat *y,
    constant W4A8LinearParams &p,
    threadgroup int8_t *As_i8,
    threadgroup int8_t *Ws_i8,
    threadgroup float *Sa_s,
    threadgroup float *Ra_s,
    threadgroup float *R128_s,
    threadgroup float *Sw_s,
    threadgroup float *Bw_s,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM_TG = 64u;
  constexpr uint OUT_N_TG = 64u;
  constexpr uint WS_ROWS = 128u;
  constexpr uint SM = 32u;
  constexpr uint OUT_SN = 16u;

  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  const uint num_pid_n = (p.n + OUT_N_TG - 1u) / OUT_N_TG;
  const uint num_pid_m = (p.mp + BM_TG - 1u) / BM_TG;
  const uint2 pid_nm = w4a8_swizzle_tg_group_m8(tg.xy, num_pid_n, num_pid_m);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg = pid_nm.x * OUT_N_TG;
  if (rb_tg >= p.rows || n0_tg >= p.n) {
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint sg_m = sg >> 2u;
  const uint sg_n = sg & 3u;
  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * OUT_SN;
  const bool sg_active = (rb_sg < p.rows) && (n0_sg < p.n);
  const uint groups = p.k / kW4A8BK;
  const uint col_limit = p.row_offset + p.n;

  constexpr auto desc_i8 = matmul2d_descriptor(
      16, 32, 32, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_i8, execution_simdgroup> op_i8;
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
      As_i8, dextents<int32_t, 2>(int(kW4A8StrideI8), int(BM_TG)));
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
      Ws_i8, dextents<int32_t, 2>(int(kW4A8StrideI8), int(WS_ROWS)));

  auto a_proto = tA.slice(0, int(sg_m * SM));
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

  const uint w_row = tid >> 1u;
  const uint half_k = tid & 1u;
  const bool is_up = ((w_row >> 4u) & 1u) != 0u;
  const uint local_col = n0_tg + ((w_row >> 5u) * 16u) + (w_row & 15u);
  const uint global_col = p.row_offset + local_col;
  threadgroup char4 *w_dst4 =
      reinterpret_cast<threadgroup char4 *>(Ws_i8 + w_row * kW4A8StrideI8 + half_k * 32u);

  const uint a_col16 = (tid & 3u) * 16u;
  const uint a_r = tid >> 2u; // 0..63

  const bool full_tile = (rb_tg + BM_TG <= p.rows) && (n0_tg + OUT_N_TG <= p.n);
  constexpr ulong kCodeBytesG = (ModeG == 2u) ? 16ul : (ModeG == 3u) ? 64ul : 32ul;
  constexpr ulong kHighBytesG = (ModeG == 1u) ? 16ul : (ModeG == 2u) ? 8ul : 0ul;
  constexpr ulong kCodeBytesU = (ModeU == 2u) ? 16ul : (ModeU == 3u) ? 64ul : 32ul;
  constexpr ulong kHighBytesU = (ModeU == 1u) ? 16ul : (ModeU == 2u) ? 8ul : 0ul;
  device const uchar *wc_g = nullptr;
  device const uchar *wh_g = nullptr;
  device const uchar *wc_u = nullptr;
  device const uchar *wh_u = nullptr;
  device const uint *sb_g = nullptr;
  device const uint *sb_u = nullptr;
  if (full_tile) {
    const uint tile_n = global_col >> 4u;
    const uint rn = global_col & 15u;
    const ulong tile_base = ulong(tile_n) * ulong(groups);
    wc_g = g_codes + tile_base * (16ul * kCodeBytesG) + ulong(rn) * kCodeBytesG;
    wh_g = (kHighBytesG > 0ul)
        ? (g_high + tile_base * (16ul * kHighBytesG) + ulong(rn) * kHighBytesG)
        : nullptr;
    wc_u = u_codes + tile_base * (16ul * kCodeBytesU) + ulong(rn) * kCodeBytesU;
    wh_u = (kHighBytesU > 0ul)
        ? (u_high + tile_base * (16ul * kHighBytesU) + ulong(rn) * kHighBytesU)
        : nullptr;
    sb_g = g_sb + global_col;
    sb_u = u_sb + global_col;
  }

  for (uint g = 0u; g < groups; ++g) {
    const uint k0 = g * kW4A8BK;
    const bool need_sb = (half_k == 0u) && ((GroupSize == 64u) || ((g & 1u) == 0u));

    uint4 qa_v;
    W4A8AffineRaw32<ModeG> raw_g;
    W4A8AffineRaw32<ModeU> raw_u;
    if (full_tile) {
      qa_v = *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r) * p.k + k0 + a_col16);
      if (!is_up) {
        raw_g = w4a8_affine_fetch_raw_32_fast<ModeG, 16u>(wc_g, wh_g, sb_g, p.padded_n, g, half_k, need_sb);
      } else {
        raw_u = w4a8_affine_fetch_raw_32_fast<ModeU, 16u>(wc_u, wh_u, sb_u, p.padded_n, g, half_k, need_sb);
      }
    } else {
      qa_v = (rb_tg + a_r < p.rows)
          ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r) * p.k + k0 + a_col16)
          : uint4(0u);
      if (!is_up) {
        raw_g = w4a8_affine_fetch_raw_32_checked<ModeG, 16u>(
            g_codes, g_high, g_sb, global_col, col_limit, p.padded_n, groups, g, half_k, need_sb);
      } else {
        raw_u = w4a8_affine_fetch_raw_32_checked<ModeU, 16u>(
            u_codes, u_high, u_sb, global_col, col_limit, p.padded_n, groups, g, half_k, need_sb);
      }
    }
    float sa_v = 0.0f, ra_v = 0.0f;
    if (tid < BM_TG) {
      const uint r_g = rb_tg + tid;
      if (r_g < p.rows) {
        sa_v = sa[ulong(g) * p.mp + r_g];
        ra_v = ra[ulong(g) * p.mp + r_g];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    *reinterpret_cast<threadgroup uint4 *>(As_i8 + a_r * kW4A8StrideI8 + a_col16) = qa_v;
    if (!is_up) {
      w4a8_affine_unpack_raw_32_i8<ModeG>(raw_g, w_dst4);
    } else {
      w4a8_affine_unpack_raw_32_i8<ModeU>(raw_u, w_dst4);
    }
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
      float sw, bw;
      if (!is_up) {
        w4a8_decode_sb<ModeG>(raw_g.sb_u, sw, bw);
      } else {
        w4a8_decode_sb<ModeU>(raw_u.sb_u, sw, bw);
      }
      Sw_s[w_row] = sw;
      Bw_s[w_row] = bw;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_active) {
      const int col_base = int(sg_n * 32u);
      auto b0 = tB.slice(0, col_base);
      auto b1 = tB.slice(32, col_base);
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        if (full_tile || rb_sg + mm * 16u < p.rows) {
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            acc_i32[i] = 0;
          }
          const int row_base = int(sg_m * SM + mm * 16u);
          auto a0 = tA.slice(0, row_base);
          auto a1 = tA.slice(32, row_base);
          op_i8.run(a0, b0, acc_i32);
          op_i8.run(a1, b1, acc_i32);
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            if (acc_i32.is_valid_element(i)) {
              const auto idx = acc_i32.get_multidimensional_index(i);
              const uint c_l = sg_n * 32u + uint(idx[0]);
              const uint r_l = sg_m * SM + mm * 16u + uint(idx[1]);
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

  // Portable 0-ULP Qwen SwiGLU epilogue reusing dead Ws_i8 (10,240 B >= 8 SGs * 16 * 32 * 2 B = 8,192 B).
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg_active) {
    threadgroup bfloat *sg_buf = reinterpret_cast<threadgroup bfloat *>(Ws_i8) + sg * 512u;
    const bool aligned_out = ((out_stride & 3u) == 0u) && ((p.row_offset & 3u) == 0u);
    const uint r_out_base = (lane >> 2u) * 2u; // 0, 2, ..., 14
    const uint c_out4 = (lane & 3u) * 4u;      // 0, 4, 8, 12
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm) {
      if (rb_sg + mm * 16u < p.rows) {
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          if (acc_i32.is_valid_element(i)) {
            const auto idx = acc_i32.get_multidimensional_index(i);
            sg_buf[uint(idx[1]) * 32u + uint(idx[0])] = bfloat(acc_f32[mm][i]);
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint lcol = n0_sg + c_out4;
        const uint gcol = p.row_offset + lcol;
        if (lcol < p.n) {
          #pragma clang loop unroll(full)
          for (uint dr = 0u; dr < 2u; ++dr) {
            const uint r = r_out_base + dr;
            const uint row = rb_sg + mm * 16u + r;
            if (row < p.rows) {
              const bfloat4 g4 = *reinterpret_cast<const threadgroup bfloat4 *>(
                  sg_buf + r * 32u + c_out4);
              const bfloat4 u4 = *reinterpret_cast<const threadgroup bfloat4 *>(
                  sg_buf + r * 32u + 16u + c_out4);
              bfloat4 out_b;
              #pragma clang loop unroll(full)
              for (uint j = 0u; j < 4u; ++j) {
                out_b[j] = bfloat(nax_silu_f32(float(g4[j])) * float(u4[j]));
              }
              if (aligned_out && lcol + 4u <= p.n) {
                *reinterpret_cast<device bfloat4 *>(y + ulong(row) * ulong(out_stride) + ulong(gcol)) = out_b;
              } else {
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  if (lcol + j < p.n) {
                    y[ulong(row) * ulong(out_stride) + ulong(gcol + j)] = out_b[j];
                  }
                }
              }
            }
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }
}

// ============================================================================
// Dense MXFP4 W4A8 Linear (BM=64, BN=64, BK=64, k_sub=32, 4 SGs = 128 threads, 10.75 KiB SRAM)
// ============================================================================
template <uint BN_STORED>
inline void mxfp4_w4a8_nax_linear_impl(
    device const int8_t *qa,
    device const float *sa,
    device const uchar *codes,
    device const uchar *scales,
    device bfloat *y,
    constant W4A8LinearParams &p,
    threadgroup int8_t *As_i8,
    threadgroup int8_t *Ws_i8,
    threadgroup float *Sa_s,
    threadgroup float *Sw0_s,
    threadgroup float *Sw1_s,
    uint lane,
    uint sg,
    uint3 tg) {
  constexpr uint BM_TG = 64u;
  constexpr uint BN_TG = 64u;
  constexpr uint SM = 32u;
  constexpr uint SN = 32u;

  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  const uint num_pid_n = (p.n + BN_TG - 1u) / BN_TG;
  const uint num_pid_m = (p.mp + BM_TG - 1u) / BM_TG;
  const uint2 pid_nm = w4a8_swizzle_tg_group_m8(tg.xy, num_pid_n, num_pid_m);
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
  const uint blocks128 = p.k / 128u;
  const uint k_blocks = p.k / kW4A8BK;
  const uint col_limit = p.row_offset + p.n;

  constexpr auto desc_i8 = matmul2d_descriptor(
      16, 32, 32, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_i8, execution_simdgroup> op_i8;
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
      As_i8, dextents<int32_t, 2>(int(kW4A8StrideI8), int(BM_TG)));
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
      Ws_i8, dextents<int32_t, 2>(int(kW4A8StrideI8), int(BN_TG)));

  auto a_proto = tA.slice(0, int(sg_m * SM));
  auto b_proto = tB.slice(0, int(sg_n * SN));
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

  const uint w_row = tid >> 1u;
  const uint col32 = tid & 1u;
  const uint global_col = p.row_offset + n0_tg + w_row;
  threadgroup char4 *w_dst4 =
      reinterpret_cast<threadgroup char4 *>(Ws_i8 + w_row * kW4A8StrideI8 + col32 * 32u);

  const uint a_col16 = (tid & 3u) * 16u;
  const uint a_r0 = tid >> 2u;
  const uint a_r1 = a_r0 + 32u;

  const bool full_tile = (rb_tg + BM_TG <= p.rows) && (n0_tg + BN_TG <= p.n);
  device const uchar *wc_row = nullptr;
  device const uchar *ws_row = nullptr;
  if (full_tile) {
    const uint tile_n = global_col / BN_STORED;
    const uint rn = global_col % BN_STORED;
    wc_row = codes + ulong(tile_n) * ulong(blocks128) * (ulong(BN_STORED) * 64ul) + ulong(rn) * 64ul;
    ws_row = scales + ulong(tile_n) * ulong(blocks128) * (ulong(BN_STORED) * 4ul) + ulong(rn) * 4ul;
  }

  for (uint kb = 0u; kb < k_blocks; ++kb) {
    const uint k0 = kb * kW4A8BK;
    const uint blk128 = kb >> 1u;
    const uint sub32 = (kb & 1u) * 2u + col32;

    uint4 qa0, qa1, raw_w;
    float sw_v;
    if (full_tile) {
      qa0 = *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0 + a_col16);
      qa1 = *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0 + a_col16);
      raw_w = reinterpret_cast<device const uint4 *>(
          wc_row + ulong(blk128) * (ulong(BN_STORED) * 64ul))[sub32];
      sw_v = 0.5f * e8m0_to_float(
          (ws_row + ulong(blk128) * (ulong(BN_STORED) * 4ul))[sub32]);
    } else {
      qa0 = (rb_tg + a_r0 < p.rows)
          ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0 + a_col16)
          : uint4(0u);
      qa1 = (rb_tg + a_r1 < p.rows)
          ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0 + a_col16)
          : uint4(0u);
      w4a8_mxfp4_fetch_raw_32<BN_STORED>(
          codes, scales, global_col, col_limit, blocks128, blk128, sub32, raw_w, sw_v);
    }
    float sa_v = 0.0f;
    if (tid < BM_TG) {
      const uint r_g = rb_tg + tid;
      if (r_g < p.rows) {
        sa_v = sa[ulong(kb) * p.mp + r_g];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    *reinterpret_cast<threadgroup uint4 *>(As_i8 + a_r0 * kW4A8StrideI8 + a_col16) = qa0;
    *reinterpret_cast<threadgroup uint4 *>(As_i8 + a_r1 * kW4A8StrideI8 + a_col16) = qa1;
    w4a8_mxfp4_unpack_raw_32_i8(raw_w, w_dst4);
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
      const int col_base = int(sg_n * SN);
      auto b0 = tB.slice(0, col_base);
      auto b1 = tB.slice(32, col_base);
      #pragma clang loop unroll(full)
      for (uint mm = 0u; mm < 2u; ++mm) {
        if (full_tile || rb_sg + mm * 16u < p.rows) {
          const int row_base = int(sg_m * SM + mm * 16u);
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
              p0[i] = float(acc_i32[i]) * Sw0_s[sg_n * SN + uint(idx[0])];
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
              const uint c_l = sg_n * SN + uint(idx[0]);
              const uint r_l = sg_m * SM + mm * 16u + uint(idx[1]);
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
    const bool aligned_out = ((out_stride & 3u) == 0u) && ((p.row_offset & 3u) == 0u);
    const uint r_out_base = (lane >> 3u) * 4u;
    const uint c_out4 = (lane & 7u) * 4u;
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm) {
      if (rb_sg + mm * 16u < p.rows) {
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          if (acc_i32.is_valid_element(i)) {
            const auto idx = acc_i32.get_multidimensional_index(i);
            sg_buf[uint(idx[1]) * 32u + uint(idx[0])] = bfloat(acc_f32[mm][i]);
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint lcol = n0_sg + c_out4;
        const uint gcol = p.row_offset + lcol;
        if (lcol < p.n) {
          #pragma clang loop unroll(full)
          for (uint dr = 0u; dr < 4u; ++dr) {
            const uint r = r_out_base + dr;
            const uint row = rb_sg + mm * 16u + r;
            if (row < p.rows) {
              const bfloat4 v4 = *reinterpret_cast<const threadgroup bfloat4 *>(
                  sg_buf + r * 32u + c_out4);
              if (aligned_out && lcol + 4u <= p.n) {
                *reinterpret_cast<device bfloat4 *>(y + ulong(row) * ulong(out_stride) + ulong(gcol)) = v4;
              } else {
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  if (lcol + j < p.n) {
                    y[ulong(row) * ulong(out_stride) + ulong(gcol + j)] = v4[j];
                  }
                }
              }
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
// Single-Pass Activation Norm + GS64 INT8 Quantization + Ra Kernels
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
// Dense W4A8 Kernel Instantiations (oQ4e q3/q4/q56/q8 x g64/g128 & MXFP4)
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
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup int8_t As_i8[64u * kW4A8StrideI8];                 \
    alignas(16) threadgroup int8_t Ws_i8[64u * kW4A8StrideI8];                 \
    threadgroup float Sa_s[64u];                                               \
    threadgroup float Ra_s[64u];                                               \
    threadgroup float R128_s[(GS) == 128u ? 64u : 1u];                         \
    threadgroup float Sw_s[64u];                                               \
    threadgroup float Bw_s[64u];                                               \
    affine_w4a8_nax_linear_impl<MODE, GS, BNS>(                                \
        qa, sa, ra, codes, nullptr, sb, y, p,                                  \
        As_i8, Ws_i8, Sa_s, Ra_s, R128_s, Sw_s, Bw_s, lane, sg, tg);           \
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
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup int8_t As_i8[64u * kW4A8StrideI8];                 \
    alignas(16) threadgroup int8_t Ws_i8[64u * kW4A8StrideI8];                 \
    threadgroup float Sa_s[64u];                                               \
    threadgroup float Ra_s[64u];                                               \
    threadgroup float R128_s[(GS) == 128u ? 64u : 1u];                         \
    threadgroup float Sw_s[64u];                                               \
    threadgroup float Bw_s[64u];                                               \
    affine_w4a8_nax_linear_impl<MODE, GS, BNS>(                                \
        qa, sa, ra, codes, high_p, sb, y, p,                                   \
        As_i8, Ws_i8, Sa_s, Ra_s, R128_s, Sw_s, Bw_s, lane, sg, tg);           \
  }

#define OMLX_M5_W4A8_AFFINE_NO_HIGH_SWIGLU(NAME, MODE, GS)                     \
  [[max_total_threads_per_threadgroup(256)]]                                   \
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
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup int8_t As_i8[64u * kW4A8StrideI8];                 \
    alignas(16) threadgroup int8_t Ws_i8[128u * kW4A8StrideI8];                \
    threadgroup float Sa_s[64u];                                               \
    threadgroup float Ra_s[64u];                                               \
    threadgroup float R128_s[(GS) == 128u ? 64u : 1u];                         \
    threadgroup float Sw_s[128u];                                              \
    threadgroup float Bw_s[128u];                                              \
    affine_w4a8_nax_swiglu_out64_impl<MODE, MODE, GS>(                         \
        qa, sa, ra, g_codes, nullptr, g_sb, u_codes, nullptr, u_sb, y, p,      \
        As_i8, Ws_i8, Sa_s, Ra_s, R128_s, Sw_s, Bw_s, lane, sg, tg);           \
  }

#define OMLX_M5_W4A8_AFFINE_WITH_HIGH_SWIGLU(NAME, MODEG, MODEU, GS)           \
  [[max_total_threads_per_threadgroup(256)]]                                   \
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
      uint3 tg [[threadgroup_position_in_grid]]) {                             \
    alignas(16) threadgroup int8_t As_i8[64u * kW4A8StrideI8];                 \
    alignas(16) threadgroup int8_t Ws_i8[128u * kW4A8StrideI8];                \
    threadgroup float Sa_s[64u];                                               \
    threadgroup float Ra_s[64u];                                               \
    threadgroup float R128_s[(GS) == 128u ? 64u : 1u];                         \
    threadgroup float Sw_s[128u];                                              \
    threadgroup float Bw_s[128u];                                              \
    affine_w4a8_nax_swiglu_out64_impl<MODEG, MODEU, GS>(                       \
        qa, sa, ra, g_codes, g_high, g_sb, u_codes, u_high, u_sb, y, p,        \
        As_i8, Ws_i8, Sa_s, Ra_s, R128_s, Sw_s, Bw_s, lane, sg, tg);           \
  }

#define OMLX_M5_W4A8_AFFINE_INSTANTIATE_GS(GS)                                 \
  OMLX_M5_W4A8_AFFINE_NO_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q4_g##GS##_bm64_bn64_bns16_sk1, 0u, GS##u, 16u) \
  OMLX_M5_W4A8_AFFINE_NO_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q4_g##GS##_bm64_bn64_bns32_sk1, 0u, GS##u, 32u) \
  OMLX_M5_W4A8_AFFINE_WITH_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q56_g##GS##_bm64_bn64_bns16_sk1, 1u, GS##u, 16u) \
  OMLX_M5_W4A8_AFFINE_WITH_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q56_g##GS##_bm64_bn64_bns32_sk1, 1u, GS##u, 32u) \
  OMLX_M5_W4A8_AFFINE_WITH_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q3_g##GS##_bm64_bn64_bns16_sk1, 2u, GS##u, 16u) \
  OMLX_M5_W4A8_AFFINE_WITH_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q3_g##GS##_bm64_bn64_bns32_sk1, 2u, GS##u, 32u) \
  OMLX_M5_W4A8_AFFINE_NO_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q8_g##GS##_bm64_bn64_bns16_sk1, 3u, GS##u, 16u) \
  OMLX_M5_W4A8_AFFINE_NO_HIGH_LINEAR(omlx_m5_affine_w4a8_nax_q8_g##GS##_bm64_bn64_bns32_sk1, 3u, GS##u, 32u) \
  OMLX_M5_W4A8_AFFINE_NO_HIGH_SWIGLU(omlx_m5_affine_w4a8_nax_q4_g##GS##_swiglu_bm64_bn64_out64_sk1, 0u, GS##u) \
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
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup int8_t As_i8[64u * kW4A8StrideI8];
  alignas(16) threadgroup int8_t Ws_i8[64u * kW4A8StrideI8];
  threadgroup float Sa_s[64u];
  threadgroup float Sw0_s[64u];
  threadgroup float Sw1_s[64u];
  mxfp4_w4a8_nax_linear_impl<16u>(
      qa, sa, codes, scales, y, p, As_i8, Ws_i8, Sa_s, Sw0_s, Sw1_s, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_w4a8_nax_bm64_bn64_bns32_sk1(
    device const int8_t *qa [[buffer(0)]],
    device const float *sa [[buffer(1)]],
    device const uchar *codes [[buffer(2)]],
    device const uchar *scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant W4A8LinearParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup int8_t As_i8[64u * kW4A8StrideI8];
  alignas(16) threadgroup int8_t Ws_i8[64u * kW4A8StrideI8];
  threadgroup float Sa_s[64u];
  threadgroup float Sw0_s[64u];
  threadgroup float Sw1_s[64u];
  mxfp4_w4a8_nax_linear_impl<32u>(
      qa, sa, codes, scales, y, p, As_i8, Ws_i8, Sa_s, Sw0_s, Sw1_s, lane, sg, tg);
}

// Dense MXFP4 W4A8 Fused Gate + Up + SwiGLU (BM=64, BN_out=64, BK=64, k_sub=32, 8 SGs = 256 threads, 16.25 KiB SRAM)
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_mxfp4_w4a8_nax_swiglu_bm64_bn64_out64_sk1(
    device const int8_t *qa [[buffer(0)]],
    device const float *sa [[buffer(1)]],
    device const uchar *g_codes [[buffer(2)]],
    device const uchar *g_scales [[buffer(3)]],
    device const uchar *u_codes [[buffer(4)]],
    device const uchar *u_scales [[buffer(5)]],
    device bfloat *y [[buffer(6)]],
    constant W4A8LinearParams &p [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint BM_TG = 64u;
  constexpr uint OUT_N_TG = 64u;
  constexpr uint WS_ROWS = 128u;
  constexpr uint SM = 32u;
  constexpr uint OUT_SN = 16u;

  alignas(16) threadgroup int8_t As_i8[BM_TG * kW4A8StrideI8];
  alignas(16) threadgroup int8_t Ws_i8[WS_ROWS * kW4A8StrideI8];
  threadgroup float Sa_s[BM_TG];
  threadgroup float Sw0_s[WS_ROWS];
  threadgroup float Sw1_s[WS_ROWS];

  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  const uint num_pid_n = (p.n + OUT_N_TG - 1u) / OUT_N_TG;
  const uint num_pid_m = (p.mp + BM_TG - 1u) / BM_TG;
  const uint2 pid_nm = w4a8_swizzle_tg_group_m8(tg.xy, num_pid_n, num_pid_m);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg = pid_nm.x * OUT_N_TG;
  if (rb_tg >= p.rows || n0_tg >= p.n) {
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint sg_m = sg >> 2u;
  const uint sg_n = sg & 3u;
  const uint rb_sg = rb_tg + sg_m * SM;
  const uint n0_sg = n0_tg + sg_n * OUT_SN;
  const bool sg_active = (rb_sg < p.rows) && (n0_sg < p.n);
  const uint blocks128 = p.k / 128u;
  const uint k_blocks = p.k / kW4A8BK;
  const uint col_limit = p.row_offset + p.n;

  constexpr auto desc_i8 = matmul2d_descriptor(
      16, 32, 32, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_i8, execution_simdgroup> op_i8;
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
      As_i8, dextents<int32_t, 2>(int(kW4A8StrideI8), int(BM_TG)));
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
      Ws_i8, dextents<int32_t, 2>(int(kW4A8StrideI8), int(WS_ROWS)));

  auto a_proto = tA.slice(0, int(sg_m * SM));
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

  const uint w_row = tid >> 1u;
  const uint col32 = tid & 1u;
  const bool is_up = ((w_row >> 4u) & 1u) != 0u;
  const uint local_col = n0_tg + ((w_row >> 5u) * 16u) + (w_row & 15u);
  const uint global_col = p.row_offset + local_col;
  threadgroup char4 *w_dst4 =
      reinterpret_cast<threadgroup char4 *>(Ws_i8 + w_row * kW4A8StrideI8 + col32 * 32u);

  const uint a_col16 = (tid & 3u) * 16u;
  const uint a_r = tid >> 2u;

  const bool full_tile = (rb_tg + BM_TG <= p.rows) && (n0_tg + OUT_N_TG <= p.n);
  device const uchar *wc_row = nullptr;
  device const uchar *ws_row = nullptr;
  if (full_tile) {
    const uint tile_n = global_col >> 4u;
    const uint rn = global_col & 15u;
    wc_row = (is_up ? u_codes : g_codes) + ulong(tile_n) * ulong(blocks128) * 1024ul + ulong(rn) * 64ul;
    ws_row = (is_up ? u_scales : g_scales) + ulong(tile_n) * ulong(blocks128) * 64ul + ulong(rn) * 4ul;
  }

  for (uint kb = 0u; kb < k_blocks; ++kb) {
    const uint k0 = kb * kW4A8BK;
    const uint blk128 = kb >> 1u;
    const uint sub32 = (kb & 1u) * 2u + col32;

    uint4 qa_v, raw_w;
    float sw_v;
    if (full_tile) {
      qa_v = *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r) * p.k + k0 + a_col16);
      raw_w = reinterpret_cast<device const uint4 *>(wc_row + ulong(blk128) * 1024ul)[sub32];
      sw_v = 0.5f * e8m0_to_float((ws_row + ulong(blk128) * 64ul)[sub32]);
    } else {
      qa_v = (rb_tg + a_r < p.rows)
          ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r) * p.k + k0 + a_col16)
          : uint4(0u);
      w4a8_mxfp4_fetch_raw_32<16u>(
          is_up ? u_codes : g_codes, is_up ? u_scales : g_scales,
          global_col, col_limit, blocks128, blk128, sub32, raw_w, sw_v);
    }
    float sa_v = 0.0f;
    if (tid < BM_TG) {
      const uint r_g = rb_tg + tid;
      if (r_g < p.rows) {
        sa_v = sa[ulong(kb) * p.mp + r_g];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    *reinterpret_cast<threadgroup uint4 *>(As_i8 + a_r * kW4A8StrideI8 + a_col16) = qa_v;
    w4a8_mxfp4_unpack_raw_32_i8(raw_w, w_dst4);
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
        if (full_tile || rb_sg + mm * 16u < p.rows) {
          const int row_base = int(sg_m * SM + mm * 16u);
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
              const uint r_l = sg_m * SM + mm * 16u + uint(idx[1]);
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
    const bool aligned_out = ((out_stride & 3u) == 0u) && ((p.row_offset & 3u) == 0u);
    const uint r_out_base = (lane >> 2u) * 2u;
    const uint c_out4 = (lane & 3u) * 4u;
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm) {
      if (rb_sg + mm * 16u < p.rows) {
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          if (acc_i32.is_valid_element(i)) {
            const auto idx = acc_i32.get_multidimensional_index(i);
            sg_buf[uint(idx[1]) * 32u + uint(idx[0])] = bfloat(acc_f32[mm][i]);
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint lcol = n0_sg + c_out4;
        const uint gcol = p.row_offset + lcol;
        if (lcol < p.n) {
          #pragma clang loop unroll(full)
          for (uint dr = 0u; dr < 2u; ++dr) {
            const uint r = r_out_base + dr;
            const uint row = rb_sg + mm * 16u + r;
            if (row < p.rows) {
              const bfloat4 g4 = *reinterpret_cast<const threadgroup bfloat4 *>(
                  sg_buf + r * 32u + c_out4);
              const bfloat4 u4 = *reinterpret_cast<const threadgroup bfloat4 *>(
                  sg_buf + r * 32u + 16u + c_out4);
              bfloat4 out_b;
              #pragma clang loop unroll(full)
              for (uint j = 0u; j < 4u; ++j) {
                out_b[j] = bfloat(nax_silu_f32(float(g4[j])) * float(u4[j]));
              }
              if (aligned_out && lcol + 4u <= p.n) {
                *reinterpret_cast<device bfloat4 *>(y + ulong(row) * ulong(out_stride) + ulong(gcol)) = out_b;
              } else {
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  if (lcol + j < p.n) {
                    y[ulong(row) * ulong(out_stride) + ulong(gcol + j)] = out_b[j];
                  }
                }
              }
            }
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }
}
