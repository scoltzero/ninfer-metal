// WP 3A M5 Ultra Live Microbenchmark Kernels:
//   - Baseline Phase 2: 128-thread Ws_i8 (16x32x32 x 2, 10,000 B SRAM)
//   - Candidate alpha-256t (tA & ctA, stride 72 [4,864 B] & stride 64 [4,352 B]):
//       Zero-Ws shift-merge, simd_all(|de| <= 3) single matmul2d(16,32,64) + 1 rescale,
//       uniform 2x matmul2d(16,32,32) fallback when simd_any(|de| > 3).
//   - Candidate alpha-512t (BM=64, BN=128, WM=4, WN=4, 4,352 B SRAM)
//   - Candidate alpha-512t-m128 (BM=128, BN=64, WM=8, WN=2, 8,704 B SRAM)
//   - Candidate beta-256t (TG MXFP8 x TG MXFP8 -> float, 8,448 B SRAM) + MXFP8 quantizer
//   - Candidate beta-256t-dva (DV MXFP8 x TG MXFP8 -> float, 4,224 B SRAM)
//   - Reference Native W4A4 (128t & 256t, 0 B SRAM, DvFp4 x DvFp4)

#include "../kernels/common_math.h"
#include "../kernels/fwht64.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct BenchGemmParams {
  uint n;
  uint k;
  uint mp;
  uint rows;
  uint padded_n;
  uint row_offset;
  uint out_stride;
  uint reserved;
};

struct BenchQuantParams {
  uint k;
  uint mp;
  uint reserved;
  uint rows;
};

namespace {

inline uint2 bench_swizzle_m8(uint2 tg_xy, uint num_pid_n, uint num_pid_m) {
  constexpr uint GROUP_M = 8u;
  if (num_pid_n == 0u || num_pid_m == 0u) return tg_xy;
  const uint pid = tg_xy.y * num_pid_n + tg_xy.x;
  const uint num_pid_in_group = GROUP_M * num_pid_n;
  const uint group_id = pid / num_pid_in_group;
  const uint first_pid_m = group_id * GROUP_M;
  const uint group_size_m = min(num_pid_m - first_pid_m, GROUP_M);
  const uint pid_m = first_pid_m + (pid % group_size_m);
  const uint pid_n = (pid % num_pid_in_group) / group_size_m;
  return uint2(pid_n, pid_m);
}

inline float bench_e8m0_safe(uchar code) {
  if (code == 255u) return 0.0f;
  const uint bits = (code == 0u) ? 0x00400000u : (uint(code) << 23u);
  return as_type<float>(bits);
}

inline int8_t bench_e2m1_x2_sh(uint nib, uint sh) {
  const int mag = int(((0xC8643210u >> ((nib & 7u) * 4u)) & 0xFu) << sh);
  return int8_t((nib & 8u) ? -mag : mag);
}

inline uchar bench_e2m1_to_fp8_e4m3(uint nib) {
  // E2M1 {0, 0.5, 1, 1.5, 2, 3, 4, 6} -> exact FP8 E4M3 (bias=7):
  // {0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C} | ((nib & 8) << 4)
  const uchar mag = uchar((0x4C4844403C383000ul >> ((nib & 7u) * 8u)) & 0xFFul);
  return uchar(mag | ((nib & 8u) << 4u));
}

inline void bench_unpack_mxfp4_32_i8(uint4 raw_w, threadgroup char4 *dst4) {
  #pragma clang loop unroll(full)
  for (uint wi = 0u; wi < 4u; ++wi) {
    const uint wd = raw_w[wi];
    dst4[wi * 2u + 0u] = char4(
        bench_e2m1_x2_sh((wd >>  0u) & 0xFu, 0u),
        bench_e2m1_x2_sh((wd >>  4u) & 0xFu, 0u),
        bench_e2m1_x2_sh((wd >>  8u) & 0xFu, 0u),
        bench_e2m1_x2_sh((wd >> 12u) & 0xFu, 0u));
    dst4[wi * 2u + 1u] = char4(
        bench_e2m1_x2_sh((wd >> 16u) & 0xFu, 0u),
        bench_e2m1_x2_sh((wd >> 20u) & 0xFu, 0u),
        bench_e2m1_x2_sh((wd >> 24u) & 0xFu, 0u),
        bench_e2m1_x2_sh((wd >> 28u) & 0xFu, 0u));
  }
}

inline uchar4 bench_unpack_4nib_fp8(ushort pk) {
  return uchar4(
      bench_e2m1_to_fp8_e4m3((uint(pk) >>  0u) & 0xFu),
      bench_e2m1_to_fp8_e4m3((uint(pk) >>  4u) & 0xFu),
      bench_e2m1_to_fp8_e4m3((uint(pk) >>  8u) & 0xFu),
      bench_e2m1_to_fp8_e4m3((uint(pk) >> 12u) & 0xFu));
}

struct BaselineTg {
  alignas(16) int8_t As_i8[64u * 72u];
  alignas(16) int8_t Ws_i8[64u * 72u];
  float Sa_s[64u];
  float Sw0_s[64u];
  float Sw1_s[64u];
};

template <uint BM, uint AS_STRIDE>
struct AlphaTg {
  alignas(16) int8_t As_i8[BM * AS_STRIDE];
  alignas(16) float Sa_s[BM];
};
static_assert(sizeof(AlphaTg<64u, 72u>) == 4864ul, "AlphaTg<64,72> == 4864 B");
static_assert(sizeof(AlphaTg<64u, 64u>) == 4352ul, "AlphaTg<64,64> == 4352 B");
static_assert(sizeof(AlphaTg<128u, 64u>) == 8704ul, "AlphaTg<128,64> == 8704 B");

struct BetaTg {
  alignas(16) uchar As_fp8[64u * 64u]; // 4096 B
  alignas(16) uchar Ws_fp8[64u * 64u]; // 4096 B
  alignas(16) uchar As_sc[64u * 2u];   //  128 B
  alignas(16) uchar Ws_sc[64u * 2u];   //  128 B
};
static_assert(sizeof(BetaTg) == 8448ul, "BetaTg == 8448 B");

struct BetaDvaTg {
  alignas(16) uchar Ws_fp8[64u * 64u]; // 4096 B
  alignas(16) uchar Ws_sc[64u * 2u];   //  128 B
};
static_assert(sizeof(BetaDvaTg) == 4224ul, "BetaDvaTg == 4224 B");

using DvUe8 = tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>;
using TgUe8 = tensor_blockwise<tensor_plane_scales, threadgroup metal_fp8_ue8m0_format, 32, 1>;
using DvMxfp8 = tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline, DvUe8>;
using TgMxfp8 = tensor<threadgroup metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline, TgUe8>;
using DvMxfp4 = tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline, DvUe8>;

// Core Candidate alpha template:
//   WM*WN SGs (256t: WM=4,WN=2; 512t-n128: WM=4,WN=4; 512t-m128: WM=8,WN=2)
//   UseRegisterCtA: true => direct char4 load into register ct_a; false => tA threadgroup tensor
template <uint WM, uint WN, uint AS_STRIDE, bool UseRegisterCtA>
inline void bench_alpha_zero_ws_impl(
    device const int8_t *qa,
    device const float *sa,
    device const uchar *codes,
    device const uchar *scales,
    device bfloat *y,
    constant BenchGemmParams &p,
    threadgroup AlphaTg<WM * 16u, AS_STRIDE> &tg_mem,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 grid_tgs) {
  constexpr uint BM_TG = WM * 16u;
  constexpr uint BN_TG = WN * 32u;

  const uint2 pid_nm = bench_swizzle_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg = pid_nm.x * BN_TG;
  if (rb_tg >= p.rows || n0_tg >= p.n) return;

  const uint tid = sg * 32u + lane;
  const uint sg_m = sg / WN;
  const uint sg_n = sg % WN;
  const uint n0_sg = n0_tg + sg_n * 32u;
  const uint blocks128 = p.k >> 7u;
  const uint k_blocks = p.k >> 6u;
  const uint col_limit = p.row_offset + p.n;

  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) | ((lane >> 1u) & 3u);
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
  const uint fn_byte = fn >> 1u;

  // Precompute per-lane (1 of 32 N-rows for scales) and 4 weight code row pointers (fm + nb*8)
  const uint gcol_lane = p.row_offset + n0_sg + lane;
  const bool live_lane = (gcol_lane < col_limit);
  device const uchar *ws_lane = live_lane
      ? (scales + ulong(gcol_lane >> 4u) * ulong(blocks128) * 64ul + ulong(gcol_lane & 15u) * 4ul)
      : nullptr;

  device const uchar *wc_nb[4];
  #pragma clang loop unroll(full)
  for (uint nb = 0u; nb < 4u; ++nb) {
    const uint gcol_w = p.row_offset + n0_sg + fm + nb * 8u;
    wc_nb[nb] = (gcol_w < col_limit)
        ? (codes + ulong(gcol_w >> 4u) * ulong(blocks128) * 1024ul + ulong(gcol_w & 15u) * 64ul + fn_byte)
        : nullptr;
  }

  float acc[16];
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 16u; ++i) acc[i] = 0.0f;

  // Staging indices for As_i8 [BM_TG, 64]:
  // 4 threads per row (each loads 16B = uint4). With BM_TG*4 <= THREADS, 1 load/thread!
  constexpr uint kAStagers = BM_TG * 4u;
  const uint a_row = tid >> 2u;
  const uint a_col16 = (tid & 3u) * 16u;

  for (uint kb = 0u; kb < k_blocks; ++kb) {
    const uint k0 = kb * 64u;
    const uint blk128 = kb >> 1u;
    const uint half64 = kb & 1u;

    uint4 qa_v = uint4(0u);
    if (tid < kAStagers && (rb_tg + a_row) < p.rows) {
      qa_v = *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_row) * p.k + k0 + a_col16);
    }
    const float sa_v = (tid < BM_TG && (rb_tg + tid) < p.rows)
        ? sa[ulong(kb) * p.mp + rb_tg + tid]
        : 0.0f;

    // Fetch UE8M0 scale pair for this lane's N-row (1 coalesced uchar2 per lane!)
    uchar2 ep_lane = uchar2(0u, 0u);
    if (ws_lane) {
      ep_lane = *reinterpret_cast<device const uchar2 *>(
          ws_lane + ulong(blk128) * 64ul + ulong(half64) * 2ul);
    }
    const bool valid_e = (!live_lane) || ((ep_lane.x != 255u) && (ep_lane.y != 255u));
    const int de_lane = live_lane ? (valid_e ? (int(ep_lane.y) - int(ep_lane.x)) : 255) : 0;
    const bool fast_k64 = simd_all(abs(de_lane) <= 3);

    const uint sh0_lane = (fast_k64 && de_lane < 0) ? uint(-de_lane) : 0u;
    const uint sh1_lane = (fast_k64 && de_lane > 0) ? uint(de_lane) : 0u;
    const uint sh_pack_lane = sh0_lane | (sh1_lane << 4u);

    // Fetch & shift-decode 4 N-rows x 4 K-quarters directly into registers (ZERO Ws SRAM!)
    char4 w_reg[4][4];
    const ulong blk_c_off = ulong(blk128) * 1024ul + ulong(half64) * 32ul;
    #pragma clang loop unroll(full)
    for (uint nb = 0u; nb < 4u; ++nb) {
      const uint sh_nb = simd_shuffle(sh_pack_lane, ushort(fm + nb * 8u));
      const uint sh0 = sh_nb & 0xFu;
      const uint sh1 = sh_nb >> 4u;
      device const uchar *wp = wc_nb[nb];
      ushort pk0 = 0u, pk1 = 0u, pk2 = 0u, pk3 = 0u;
      if (wp) {
        device const uchar *wb = wp + blk_c_off;
        pk0 = *reinterpret_cast<device const ushort *>(wb + 0u);
        pk1 = *reinterpret_cast<device const ushort *>(wb + 8u);
        pk2 = *reinterpret_cast<device const ushort *>(wb + 16u);
        pk3 = *reinterpret_cast<device const ushort *>(wb + 24u);
      }
      w_reg[0][nb] = char4(
          bench_e2m1_x2_sh((uint(pk0) >>  0u) & 0xFu, sh0),
          bench_e2m1_x2_sh((uint(pk0) >>  4u) & 0xFu, sh0),
          bench_e2m1_x2_sh((uint(pk0) >>  8u) & 0xFu, sh0),
          bench_e2m1_x2_sh((uint(pk0) >> 12u) & 0xFu, sh0));
      w_reg[1][nb] = char4(
          bench_e2m1_x2_sh((uint(pk1) >>  0u) & 0xFu, sh0),
          bench_e2m1_x2_sh((uint(pk1) >>  4u) & 0xFu, sh0),
          bench_e2m1_x2_sh((uint(pk1) >>  8u) & 0xFu, sh0),
          bench_e2m1_x2_sh((uint(pk1) >> 12u) & 0xFu, sh0));
      w_reg[2][nb] = char4(
          bench_e2m1_x2_sh((uint(pk2) >>  0u) & 0xFu, sh1),
          bench_e2m1_x2_sh((uint(pk2) >>  4u) & 0xFu, sh1),
          bench_e2m1_x2_sh((uint(pk2) >>  8u) & 0xFu, sh1),
          bench_e2m1_x2_sh((uint(pk2) >> 12u) & 0xFu, sh1));
      w_reg[3][nb] = char4(
          bench_e2m1_x2_sh((uint(pk3) >>  0u) & 0xFu, sh1),
          bench_e2m1_x2_sh((uint(pk3) >>  4u) & 0xFu, sh1),
          bench_e2m1_x2_sh((uint(pk3) >>  8u) & 0xFu, sh1),
          bench_e2m1_x2_sh((uint(pk3) >> 12u) & 0xFu, sh1));
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < kAStagers) {
      if constexpr (AS_STRIDE == 64u) {
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + a_row * AS_STRIDE + a_col16) = qa_v;
      } else {
        *reinterpret_cast<threadgroup uint2 *>(tg_mem.As_i8 + a_row * AS_STRIDE + a_col16 + 0u) = qa_v.xy;
        *reinterpret_cast<threadgroup uint2 *>(tg_mem.As_i8 + a_row * AS_STRIDE + a_col16 + 8u) = qa_v.zw;
      }
    }
    if (tid < BM_TG) tg_mem.Sa_s[tid] = sa_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float sa0 = tg_mem.Sa_s[sg_m * 16u + fm];
    const float sa1 = tg_mem.Sa_s[sg_m * 16u + 8u + fm];

    if (fast_k64) {
      const float sw_lane = live_lane ? (0.5f * bench_e8m0_safe(min(ep_lane.x, ep_lane.y))) : 0.0f;
      if constexpr (UseRegisterCtA) {
        // Both inputs in register cooperative_tensors: K must be 32, so chain
        // mode::multiply (k=0..31) -> mode::multiply_accumulate (k=32..63) + 1 FP32 rescale!
        constexpr auto desc32_m = matmul2d_descriptor(
            16, 32, 32, false, true, false,
            matmul2d_descriptor::mode::multiply);
        constexpr auto desc32_a = matmul2d_descriptor(
            16, 32, 32, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc32_m, execution_simdgroup> op32_m;
        matmul2d<desc32_a, execution_simdgroup> op32_a;
        const threadgroup int8_t *as_r0 = tg_mem.As_i8 + (sg_m * 16u + fm) * AS_STRIDE + fn;
        const threadgroup int8_t *as_r1 = as_r0 + 8u * AS_STRIDE;

        auto ca0 = op32_m.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
        auto cb0 = op32_m.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < 2u; ++kk) {
          const char4 v0 = *reinterpret_cast<const threadgroup char4 *>(as_r0 + kk * 16u);
          const char4 v1 = *reinterpret_cast<const threadgroup char4 *>(as_r1 + kk * 16u);
          const uint base_a = kk * 8u;
          ca0[base_a + 0u] = v0[0]; ca0[base_a + 1u] = v0[1]; ca0[base_a + 2u] = v0[2]; ca0[base_a + 3u] = v0[3];
          ca0[base_a + 4u] = v1[0]; ca0[base_a + 5u] = v1[1]; ca0[base_a + 6u] = v1[2]; ca0[base_a + 7u] = v1[3];
          #pragma clang loop unroll(full)
          for (uint nb = 0u; nb < 4u; ++nb) {
            const uint base_b = kk * 16u + nb * 4u;
            cb0[base_b + 0u] = w_reg[kk][nb][0];
            cb0[base_b + 1u] = w_reg[kk][nb][1];
            cb0[base_b + 2u] = w_reg[kk][nb][2];
            cb0[base_b + 3u] = w_reg[kk][nb][3];
          }
        }
        auto ci = op32_m.template get_destination_cooperative_tensor<
            remove_addrspace_t<decltype(ca0)>, remove_addrspace_t<decltype(cb0)>, int32_t>();
        op32_m.run(ca0, cb0, ci);

        auto ca1 = op32_a.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
        auto cb1 = op32_a.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < 2u; ++kk) {
          const char4 v0 = *reinterpret_cast<const threadgroup char4 *>(as_r0 + (kk + 2u) * 16u);
          const char4 v1 = *reinterpret_cast<const threadgroup char4 *>(as_r1 + (kk + 2u) * 16u);
          const uint base_a = kk * 8u;
          ca1[base_a + 0u] = v0[0]; ca1[base_a + 1u] = v0[1]; ca1[base_a + 2u] = v0[2]; ca1[base_a + 3u] = v0[3];
          ca1[base_a + 4u] = v1[0]; ca1[base_a + 5u] = v1[1]; ca1[base_a + 6u] = v1[2]; ca1[base_a + 7u] = v1[3];
          #pragma clang loop unroll(full)
          for (uint nb = 0u; nb < 4u; ++nb) {
            const uint base_b = kk * 16u + nb * 4u;
            cb1[base_b + 0u] = w_reg[kk + 2u][nb][0];
            cb1[base_b + 1u] = w_reg[kk + 2u][nb][1];
            cb1[base_b + 2u] = w_reg[kk + 2u][nb][2];
            cb1[base_b + 3u] = w_reg[kk + 2u][nb][3];
          }
        }
        op32_a.run(ca1, cb1, ci);

        #pragma clang loop unroll(full)
        for (uint u = 0u; u < 2u; ++u) {
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            const float sw = simd_shuffle(sw_lane, ushort(u * 16u + fn + c));
            acc[u * 8u + c]      = fma(sa0 * sw, float(ci[u * 8u + c]),      acc[u * 8u + c]);
            acc[u * 8u + 4u + c] = fma(sa1 * sw, float(ci[u * 8u + 4u + c]), acc[u * 8u + 4u + c]);
          }
        }
      } else {
        constexpr auto desc64 = matmul2d_descriptor(
            16, 32, 64, false, true, false,
            matmul2d_descriptor::mode::multiply);
        matmul2d<desc64, execution_simdgroup> op64;
        auto cb = op64.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < 4u; ++kk) {
          #pragma clang loop unroll(full)
          for (uint nb = 0u; nb < 4u; ++nb) {
            const uint base_i = kk * 16u + nb * 4u;
            cb[base_i + 0u] = w_reg[kk][nb][0];
            cb[base_i + 1u] = w_reg[kk][nb][1];
            cb[base_i + 2u] = w_reg[kk][nb][2];
            cb[base_i + 3u] = w_reg[kk][nb][3];
          }
        }
        tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA_full(
            tg_mem.As_i8, dextents<int32_t, 2>(int(AS_STRIDE), int(BM_TG)));
        auto tA = tA_full.slice(0, int(sg_m * 16u));
        auto ci = op64.template get_destination_cooperative_tensor<
            remove_addrspace_t<decltype(tA)>, remove_addrspace_t<decltype(cb)>, int32_t>();
        op64.run(tA, cb, ci);
        #pragma clang loop unroll(full)
        for (uint u = 0u; u < 2u; ++u) {
          #pragma clang loop unroll(full)
          for (uint c = 0u; c < 4u; ++c) {
            const float sw = simd_shuffle(sw_lane, ushort(u * 16u + fn + c));
            acc[u * 8u + c]      = fma(sa0 * sw, float(ci[u * 8u + c]),      acc[u * 8u + c]);
            acc[u * 8u + 4u + c] = fma(sa1 * sw, float(ci[u * 8u + 4u + c]), acc[u * 8u + 4u + c]);
          }
        }
      }
    } else {
      // Uniform fallback when simd_any(|de| > 3): 2 x matmul2d(16, 32, 32) with unshifted weights!
      const float sw0_lane = live_lane ? (0.5f * bench_e8m0_safe(ep_lane.x)) : 0.0f;
      const float sw1_lane = live_lane ? (0.5f * bench_e8m0_safe(ep_lane.y)) : 0.0f;
      constexpr auto desc32 = matmul2d_descriptor(
          16, 32, 32, false, true, false,
          matmul2d_descriptor::mode::multiply);
      matmul2d<desc32, execution_simdgroup> op32;
      #pragma clang loop unroll(full)
      for (uint h32 = 0u; h32 < 2u; ++h32) {
        auto cb = op32.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < 2u; ++kk) {
          #pragma clang loop unroll(full)
          for (uint nb = 0u; nb < 4u; ++nb) {
            const uint base_i = kk * 16u + nb * 4u;
            cb[base_i + 0u] = w_reg[h32 * 2u + kk][nb][0];
            cb[base_i + 1u] = w_reg[h32 * 2u + kk][nb][1];
            cb[base_i + 2u] = w_reg[h32 * 2u + kk][nb][2];
            cb[base_i + 3u] = w_reg[h32 * 2u + kk][nb][3];
          }
        }
        const float sw_cur_lane = (h32 == 0u) ? sw0_lane : sw1_lane;
        if constexpr (UseRegisterCtA) {
          auto ca = op32.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
          const threadgroup int8_t *as_r0 = tg_mem.As_i8 + (sg_m * 16u + fm) * AS_STRIDE + h32 * 32u + fn;
          const threadgroup int8_t *as_r1 = as_r0 + 8u * AS_STRIDE;
          #pragma clang loop unroll(full)
          for (uint kk = 0u; kk < 2u; ++kk) {
            const char4 v0 = *reinterpret_cast<const threadgroup char4 *>(as_r0 + kk * 16u);
            const char4 v1 = *reinterpret_cast<const threadgroup char4 *>(as_r1 + kk * 16u);
            const uint base_i = kk * 8u;
            ca[base_i + 0u] = v0[0]; ca[base_i + 1u] = v0[1]; ca[base_i + 2u] = v0[2]; ca[base_i + 3u] = v0[3];
            ca[base_i + 4u] = v1[0]; ca[base_i + 5u] = v1[1]; ca[base_i + 6u] = v1[2]; ca[base_i + 7u] = v1[3];
          }
          auto ci = op32.template get_destination_cooperative_tensor<
              remove_addrspace_t<decltype(ca)>, remove_addrspace_t<decltype(cb)>, int32_t>();
          op32.run(ca, cb, ci);
          #pragma clang loop unroll(full)
          for (uint u = 0u; u < 2u; ++u) {
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) {
              const float sw = simd_shuffle(sw_cur_lane, ushort(u * 16u + fn + c));
              acc[u * 8u + c]      = fma(sa0 * sw, float(ci[u * 8u + c]),      acc[u * 8u + c]);
              acc[u * 8u + 4u + c] = fma(sa1 * sw, float(ci[u * 8u + 4u + c]), acc[u * 8u + 4u + c]);
            }
          }
        } else {
          tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA_full(
              tg_mem.As_i8, dextents<int32_t, 2>(int(AS_STRIDE), int(BM_TG)));
          auto tA = tA_full.slice(int(h32 * 32u), int(sg_m * 16u));
          auto ci = op32.template get_destination_cooperative_tensor<
              remove_addrspace_t<decltype(tA)>, remove_addrspace_t<decltype(cb)>, int32_t>();
          op32.run(tA, cb, ci);
          #pragma clang loop unroll(full)
          for (uint u = 0u; u < 2u; ++u) {
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; ++c) {
              const float sw = simd_shuffle(sw_cur_lane, ushort(u * 16u + fn + c));
              acc[u * 8u + c]      = fma(sa0 * sw, float(ci[u * 8u + c]),      acc[u * 8u + c]);
              acc[u * 8u + 4u + c] = fma(sa1 * sw, float(ci[u * 8u + 4u + c]), acc[u * 8u + 4u + c]);
            }
          }
        }
      }
    }
  }

  const uint rb_sg = rb_tg + sg_m * 16u;
  if (rb_sg < p.rows && n0_sg < p.n) {
    const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint row = rb_sg + fm + h * 8u;
      if (row < p.rows) {
        #pragma clang loop unroll(full)
        for (uint u = 0u; u < 2u; ++u) {
          const uint lcol = n0_sg + u * 16u + fn;
          if (lcol < p.n) {
            const uint e = u * 8u + h * 4u;
            *reinterpret_cast<device bfloat4 *>(y + ulong(row) * out_stride + p.row_offset + lcol) =
                bfloat4(bfloat(acc[e + 0u]), bfloat(acc[e + 1u]), bfloat(acc[e + 2u]), bfloat(acc[e + 3u]));
          }
        }
      }
    }
  }
}

} // namespace

// ============================================================================
// Quantizer: BF16 -> A8 GS64 (for Baseline & Candidate alpha)
// ============================================================================
[[max_total_threads_per_threadgroup(32)]]
kernel void bench_quantize_a8_g64(
    device const bfloat *input [[buffer(0)]],
    device int8_t *qa_out [[buffer(1)]],
    device float *sa_out [[buffer(2)]],
    constant BenchQuantParams &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g = tg.x;
  const uint row = tg.y;
  const uint groups = p.k >> 6u;
  if (g >= groups || row >= p.mp) return;
  const ulong col_off = ulong(row) * p.k + ulong(g) * 64ul;
  if (row >= p.rows) {
    reinterpret_cast<device char2 *>(qa_out + col_off)[lane] = char2(0);
    if (lane == 0u) sa_out[ulong(g) * p.mp + row] = 0.0f;
    return;
  }
  const float2 fo = float2(reinterpret_cast<device const bfloat2 *>(input + col_off)[lane]);
  const float amax = simd_max(fmax(fabs(fo.x), fabs(fo.y)));
  const float sa = amax * (1.0f / 127.0f);
  const float inv_sa = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
  const int q0 = int(clamp(rint(fo.x * inv_sa), -127.0f, 127.0f));
  const int q1 = int(clamp(rint(fo.y * inv_sa), -127.0f, 127.0f));
  reinterpret_cast<device char2 *>(qa_out + col_off)[lane] = char2(int8_t(q0), int8_t(q1));
  if (lane == 0u) sa_out[ulong(g) * p.mp + row] = sa;
}

// ============================================================================
// Quantizer: BF16 -> MXFP8 E4M3 + UE8M0 (g=32, for Candidate beta)
// Row-major codes [mp, K] uchar + row-major scales [mp, K/32] uchar
// ============================================================================
[[max_total_threads_per_threadgroup(128)]]
kernel void bench_quantize_mxfp8_g32(
    device const bfloat *input [[buffer(0)]],
    device uchar *fp8_out [[buffer(1)]],
    device uchar *sc_out [[buffer(2)]],
    constant BenchQuantParams &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  // 1 SG = 1 group of 128 elements (4 x g=32) for 1 row -> 4 elements/lane!
  const uint g128 = tg.x * 4u + sg;
  const uint row = tg.y;
  const uint groups128 = p.k >> 7u;
  if (g128 >= groups128 || row >= p.mp) return;
  const ulong col_off = ulong(row) * p.k + ulong(g128) * 128ul;
  const ulong sc_off = ulong(row) * (p.k >> 5u) + ulong(g128) * 4ul;
  if (row >= p.rows) {
    reinterpret_cast<device uint *>(fp8_out + col_off)[lane] = 0u;
    if (lane < 4u) sc_out[sc_off + lane] = 0u;
    return;
  }
  const float4 x = float4(reinterpret_cast<device const bfloat4 *>(input + col_off)[lane]);
  // 8 lanes per g=32 sub-group (8 * 4 = 32 elems)!
  float m = fmax(fmax(fabs(x.x), fabs(x.y)), fmax(fabs(x.z), fabs(x.w)));
  m = fmax(m, simd_shuffle_xor(m, 1u));
  m = fmax(m, simd_shuffle_xor(m, 2u));
  m = fmax(m, simd_shuffle_xor(m, 4u));
  const uchar sc = nax_encode_e8m0_round_up(m / 448.0f);
  const float inv_s = (m > 0.0f) ? (1.0f / e8m0_to_float(sc)) : 0.0f;
  const packed_uchar4 q4 = pack<metal_fp8_e4m3_format, rounding_mode::to_nearest_even, saturation_mode::saturate>(
      x * inv_s).as_storage_type();
  reinterpret_cast<device packed_uchar4 *>(fp8_out + col_off)[lane] = q4;
  if ((lane & 7u) == 0u) {
    sc_out[sc_off + (lane >> 3u)] = sc;
  }
}

// ============================================================================
// Baseline: Phase 2 128-thread Ws_i8 (16x32x32 x 2)
// ============================================================================
[[max_total_threads_per_threadgroup(128)]]
kernel void bench_baseline_ws_128t(
    device const int8_t *qa [[buffer(0)]],
    device const float *sa [[buffer(1)]],
    device const uchar *codes [[buffer(2)]],
    device const uchar *scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant BenchGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup BaselineTg tg_mem;
  const uint2 pid_nm = bench_swizzle_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * 64u;
  const uint n0_tg = pid_nm.x * 64u;
  if (rb_tg >= p.rows || n0_tg >= p.n) return;

  const uint tid = sg * 32u + lane;
  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const uint blocks128 = p.k >> 7u;
  const uint k_blocks = p.k >> 6u;
  const uint col_limit = p.row_offset + p.n;
  const uint w_row = tid >> 1u;
  const uint col32 = tid & 1u;
  const uint gcol_w = p.row_offset + n0_tg + w_row;
  threadgroup char4 *w_dst4 =
      reinterpret_cast<threadgroup char4 *>(tg_mem.Ws_i8 + w_row * 72u + col32 * 32u);
  const uint a_col16 = (tid & 3u) * 16u;
  const uint a_r0 = tid >> 2u;
  const uint a_r1 = a_r0 + 32u;
  device const uchar *wc_row = (gcol_w < col_limit)
      ? (codes + ulong(gcol_w >> 4u) * ulong(blocks128) * 1024ul + ulong(gcol_w & 15u) * 64ul)
      : nullptr;
  device const uchar *ws_row = (gcol_w < col_limit)
      ? (scales + ulong(gcol_w >> 4u) * ulong(blocks128) * 64ul + ulong(gcol_w & 15u) * 4ul)
      : nullptr;

  constexpr auto desc_i8 = matmul2d_descriptor(
      16, 32, 32, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_i8, execution_simdgroup> op_i8;
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
      tg_mem.As_i8, dextents<int32_t, 2>(72, 64));
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
      tg_mem.Ws_i8, dextents<int32_t, 2>(72, 64));

  float acc_f32[2][16];
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < 2u; ++mm)
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) acc_f32[mm][i] = 0.0f;

  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) | ((lane >> 1u) & 3u);
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

  for (uint kb = 0u; kb < k_blocks; ++kb) {
    const uint k0 = kb * 64u;
    const uint blk128 = kb >> 1u;
    const uint sub32 = (kb & 1u) * 2u + col32;
    const uint4 qa0 = (rb_tg + a_r0 < p.rows)
        ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0 + a_col16) : uint4(0u);
    const uint4 qa1 = (rb_tg + a_r1 < p.rows)
        ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r1) * p.k + k0 + a_col16) : uint4(0u);
    const uint4 raw_w = wc_row
        ? reinterpret_cast<device const uint4 *>(wc_row + ulong(blk128) * 1024ul)[sub32] : uint4(0u);
    const float sw_v = ws_row
        ? (0.5f * bench_e8m0_safe((ws_row + ulong(blk128) * 64ul)[sub32])) : 0.0f;
    const float sa_v = (tid < 64u && rb_tg + tid < p.rows) ? sa[ulong(kb) * p.mp + rb_tg + tid] : 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    *reinterpret_cast<threadgroup uint2 *>(tg_mem.As_i8 + a_r0 * 72u + a_col16 + 0u) = qa0.xy;
    *reinterpret_cast<threadgroup uint2 *>(tg_mem.As_i8 + a_r0 * 72u + a_col16 + 8u) = qa0.zw;
    *reinterpret_cast<threadgroup uint2 *>(tg_mem.As_i8 + a_r1 * 72u + a_col16 + 0u) = qa1.xy;
    *reinterpret_cast<threadgroup uint2 *>(tg_mem.As_i8 + a_r1 * 72u + a_col16 + 8u) = qa1.zw;
    bench_unpack_mxfp4_32_i8(raw_w, w_dst4);
    if (col32 == 0u) tg_mem.Sw0_s[w_row] = sw_v; else tg_mem.Sw1_s[w_row] = sw_v;
    if (tid < 64u) tg_mem.Sa_s[tid] = sa_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto b0 = tB.slice(0, int(sg_n * 32u));
    auto b1 = tB.slice(32, int(sg_n * 32u));
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm) {
      auto a0 = tA.slice(0, int(sg_m * 32u + mm * 16u));
      auto a1 = tA.slice(32, int(sg_m * 32u + mm * 16u));
      auto ci = op_i8.template get_destination_cooperative_tensor<
          remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
      const float sa0 = tg_mem.Sa_s[sg_m * 32u + mm * 16u + fm];
      const float sa1 = tg_mem.Sa_s[sg_m * 32u + mm * 16u + 8u + fm];
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) ci[i] = 0;
      op_i8.run(a0, b0, ci);
      #pragma clang loop unroll(full)
      for (uint u = 0u; u < 2u; ++u)
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) {
          const float sw = tg_mem.Sw0_s[sg_n * 32u + u * 16u + fn + c];
          acc_f32[mm][u * 8u + c]      = fma(sa0 * sw, float(ci[u * 8u + c]),      acc_f32[mm][u * 8u + c]);
          acc_f32[mm][u * 8u + 4u + c] = fma(sa1 * sw, float(ci[u * 8u + 4u + c]), acc_f32[mm][u * 8u + 4u + c]);
        }
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) ci[i] = 0;
      op_i8.run(a1, b1, ci);
      #pragma clang loop unroll(full)
      for (uint u = 0u; u < 2u; ++u)
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) {
          const float sw = tg_mem.Sw1_s[sg_n * 32u + u * 16u + fn + c];
          acc_f32[mm][u * 8u + c]      = fma(sa0 * sw, float(ci[u * 8u + c]),      acc_f32[mm][u * 8u + c]);
          acc_f32[mm][u * 8u + 4u + c] = fma(sa1 * sw, float(ci[u * 8u + 4u + c]), acc_f32[mm][u * 8u + 4u + c]);
        }
    }
  }

  const uint rb_sg = rb_tg + sg_m * 32u;
  const uint n0_sg = n0_tg + sg_n * 32u;
  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  if (rb_sg < p.rows && n0_sg < p.n) {
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm)
      #pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const uint row = rb_sg + mm * 16u + fm + h * 8u;
        if (row < p.rows) {
          #pragma clang loop unroll(full)
          for (uint u = 0u; u < 2u; ++u) {
            const uint lcol = n0_sg + u * 16u + fn;
            if (lcol < p.n) {
              const uint e = u * 8u + h * 4u;
              *reinterpret_cast<device bfloat4 *>(y + ulong(row) * out_stride + p.row_offset + lcol) =
                  bfloat4(bfloat(acc_f32[mm][e + 0u]), bfloat(acc_f32[mm][e + 1u]),
                          bfloat(acc_f32[mm][e + 2u]), bfloat(acc_f32[mm][e + 3u]));
            }
          }
        }
      }
  }
}

// ============================================================================
// Candidate alpha variants
// ============================================================================
[[max_total_threads_per_threadgroup(256)]]
kernel void bench_alpha_256t_cta_s64(
    device const int8_t *qa [[buffer(0)]], device const float *sa [[buffer(1)]],
    device const uchar *codes [[buffer(2)]], device const uchar *scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]], constant BenchGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup AlphaTg<64u, 64u> tg_mem;
  bench_alpha_zero_ws_impl<4u, 2u, 64u, true>(qa, sa, codes, scales, y, p, tg_mem, lane, sg, tg, grid_tgs);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void bench_alpha_256t_ta_s64(
    device const int8_t *qa [[buffer(0)]], device const float *sa [[buffer(1)]],
    device const uchar *codes [[buffer(2)]], device const uchar *scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]], constant BenchGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup AlphaTg<64u, 64u> tg_mem;
  bench_alpha_zero_ws_impl<4u, 2u, 64u, false>(qa, sa, codes, scales, y, p, tg_mem, lane, sg, tg, grid_tgs);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void bench_alpha_256t_cta_s72(
    device const int8_t *qa [[buffer(0)]], device const float *sa [[buffer(1)]],
    device const uchar *codes [[buffer(2)]], device const uchar *scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]], constant BenchGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup AlphaTg<64u, 72u> tg_mem;
  bench_alpha_zero_ws_impl<4u, 2u, 72u, true>(qa, sa, codes, scales, y, p, tg_mem, lane, sg, tg, grid_tgs);
}

[[max_total_threads_per_threadgroup(512)]]
kernel void bench_alpha_512t_n128_cta_s64(
    device const int8_t *qa [[buffer(0)]], device const float *sa [[buffer(1)]],
    device const uchar *codes [[buffer(2)]], device const uchar *scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]], constant BenchGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup AlphaTg<64u, 64u> tg_mem;
  bench_alpha_zero_ws_impl<4u, 4u, 64u, true>(qa, sa, codes, scales, y, p, tg_mem, lane, sg, tg, grid_tgs);
}

[[max_total_threads_per_threadgroup(512)]]
kernel void bench_alpha_512t_m128_cta_s64(
    device const int8_t *qa [[buffer(0)]], device const float *sa [[buffer(1)]],
    device const uchar *codes [[buffer(2)]], device const uchar *scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]], constant BenchGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup AlphaTg<128u, 64u> tg_mem;
  bench_alpha_zero_ws_impl<8u, 2u, 64u, true>(qa, sa, codes, scales, y, p, tg_mem, lane, sg, tg, grid_tgs);
}

// ============================================================================
// Candidate beta: 256-thread Native MXFP8 x MXFP8 -> float (8,448 B TG SRAM)
// ============================================================================
[[max_total_threads_per_threadgroup(256)]]
kernel void bench_beta_mxfp8_tg_256t(
    device const uchar *a_fp8 [[buffer(0)]],
    device const uchar *a_sc [[buffer(1)]],
    device const uchar *w_codes [[buffer(2)]],
    device const uchar *w_scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant BenchGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup BetaTg tg_mem;
  const uint2 pid_nm = bench_swizzle_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * 64u;
  const uint n0_tg = pid_nm.x * 64u;
  if (rb_tg >= p.rows || n0_tg >= p.n) return;

  const uint tid = sg * 32u + lane;
  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const uint blocks128 = p.k >> 7u;
  const uint k_blocks = p.k >> 6u;
  const uint sc_stride = p.k >> 5u;
  const uint col_limit = p.row_offset + p.n;

  const uint row64 = tid >> 2u;   // 0..63
  const uint sub4 = tid & 3u;     // 0..3 -> 16 elements = 8 bytes of E2M1 / 16 bytes of FP8
  const uint gcol_w = p.row_offset + n0_tg + row64;
  device const uchar *wc_row = (gcol_w < col_limit)
      ? (w_codes + ulong(gcol_w >> 4u) * ulong(blocks128) * 1024ul + ulong(gcol_w & 15u) * 64ul)
      : nullptr;
  device const uchar *ws_row = (gcol_w < col_limit)
      ? (w_scales + ulong(gcol_w >> 4u) * ulong(blocks128) * 64ul + ulong(gcol_w & 15u) * 4ul)
      : nullptr;

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 64, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  vec<float, 8> acc0(0.0f), acc1(0.0f);

  for (uint kb = 0u; kb < k_blocks; ++kb) {
    const uint k0 = kb * 64u;
    const uint blk128 = kb >> 1u;
    const uint half64 = kb & 1u;

    const uint4 qa_v = (rb_tg + row64 < p.rows)
        ? *reinterpret_cast<device const uint4 *>(a_fp8 + ulong(rb_tg + row64) * p.k + k0 + sub4 * 16u)
        : uint4(0u);
    const uchar2 sa_v = (sub4 == 0u && rb_tg + row64 < p.rows)
        ? *reinterpret_cast<device const uchar2 *>(a_sc + ulong(rb_tg + row64) * sc_stride + kb * 2u)
        : uchar2(0u, 0u);
    const ushort4 raw_w = wc_row
        ? reinterpret_cast<device const ushort4 *>(wc_row + ulong(blk128) * 1024ul + half64 * 32ul)[sub4]
        : ushort4(0u);
    const uchar2 sw_v = (sub4 == 0u && ws_row)
        ? *reinterpret_cast<device const uchar2 *>(ws_row + ulong(blk128) * 64ul + half64 * 2ul)
        : uchar2(0u, 0u);

    const uchar4 f8_0 = bench_unpack_4nib_fp8(raw_w.x);
    const uchar4 f8_1 = bench_unpack_4nib_fp8(raw_w.y);
    const uchar4 f8_2 = bench_unpack_4nib_fp8(raw_w.z);
    const uchar4 f8_3 = bench_unpack_4nib_fp8(raw_w.w);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_fp8 + row64 * 64u + sub4 * 16u) = qa_v;
    threadgroup uchar4 *wd = reinterpret_cast<threadgroup uchar4 *>(tg_mem.Ws_fp8 + row64 * 64u + sub4 * 16u);
    wd[0] = f8_0; wd[1] = f8_1; wd[2] = f8_2; wd[3] = f8_3;
    if (sub4 == 0u) {
      *reinterpret_cast<threadgroup uchar2 *>(tg_mem.As_sc + row64 * 2u) = sa_v;
      *reinterpret_cast<threadgroup uchar2 *>(tg_mem.Ws_sc + row64 * 2u) = sw_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    TgMxfp8 tA(tg_mem.As_fp8 + sg_m * (16u * 64u), dextents<int32_t, 2>(64, 16),
               TgUe8(tg_mem.As_sc + sg_m * 32u));
    TgMxfp8 tB(tg_mem.Ws_fp8 + sg_n * (32u * 64u), dextents<int32_t, 2>(64, 32),
               TgUe8(tg_mem.Ws_sc + sg_n * 64u));
    auto c = op.get_destination_cooperative_tensor<TgMxfp8, TgMxfp8, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 8u; ++i) { c[i] = acc0[i]; c[8u + i] = acc1[i]; }
    op.run(tA, tB, c);
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 8u; ++i) { acc0[i] = c[i]; acc1[i] = c[8u + i]; }
  }

  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) | ((lane >> 1u) & 3u);
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
  const uint rb_sg = rb_tg + sg_m * 16u;
  const uint n0_sg = n0_tg + sg_n * 32u;
  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  if (rb_sg < p.rows && n0_sg < p.n) {
    const uint r0 = rb_sg + fm;
    const uint r1 = r0 + 8u;
    const uint c0 = n0_sg + fn;
    const uint c1 = c0 + 16u;
    if (r0 < p.rows && c0 < p.n)
      *reinterpret_cast<device bfloat4 *>(y + ulong(r0) * out_stride + p.row_offset + c0) =
          bfloat4(bfloat(acc0[0]), bfloat(acc0[1]), bfloat(acc0[2]), bfloat(acc0[3]));
    if (r1 < p.rows && c0 < p.n)
      *reinterpret_cast<device bfloat4 *>(y + ulong(r1) * out_stride + p.row_offset + c0) =
          bfloat4(bfloat(acc0[4]), bfloat(acc0[5]), bfloat(acc0[6]), bfloat(acc0[7]));
    if (r0 < p.rows && c1 < p.n)
      *reinterpret_cast<device bfloat4 *>(y + ulong(r0) * out_stride + p.row_offset + c1) =
          bfloat4(bfloat(acc1[0]), bfloat(acc1[1]), bfloat(acc1[2]), bfloat(acc1[3]));
    if (r1 < p.rows && c1 < p.n)
      *reinterpret_cast<device bfloat4 *>(y + ulong(r1) * out_stride + p.row_offset + c1) =
          bfloat4(bfloat(acc1[4]), bfloat(acc1[5]), bfloat(acc1[6]), bfloat(acc1[7]));
  }
}

// Candidate beta-dva: Direct device MXFP8 A x TG MXFP8 W (4,224 B SRAM -> 7 TGs/core!)
[[max_total_threads_per_threadgroup(256)]]
kernel void bench_beta_mxfp8_dva_256t(
    device const uchar *a_fp8 [[buffer(0)]],
    device const uchar *a_sc [[buffer(1)]],
    device const uchar *w_codes [[buffer(2)]],
    device const uchar *w_scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant BenchGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup BetaDvaTg tg_mem;
  const uint2 pid_nm = bench_swizzle_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * 64u;
  const uint n0_tg = pid_nm.x * 64u;
  if (rb_tg >= p.rows || n0_tg >= p.n) return;

  const uint tid = sg * 32u + lane;
  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const uint blocks128 = p.k >> 7u;
  const uint k_blocks = p.k >> 6u;
  const uint col_limit = p.row_offset + p.n;

  const uint row64 = tid >> 2u;
  const uint sub4 = tid & 3u;
  const uint gcol_w = p.row_offset + n0_tg + row64;
  device const uchar *wc_row = (gcol_w < col_limit)
      ? (w_codes + ulong(gcol_w >> 4u) * ulong(blocks128) * 1024ul + ulong(gcol_w & 15u) * 64ul)
      : nullptr;
  device const uchar *ws_row = (gcol_w < col_limit)
      ? (w_scales + ulong(gcol_w >> 4u) * ulong(blocks128) * 64ul + ulong(gcol_w & 15u) * 4ul)
      : nullptr;

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 64, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  vec<float, 8> acc0(0.0f), acc1(0.0f);
  const uint rb_sg = rb_tg + sg_m * 16u;

  for (uint kb = 0u; kb < k_blocks; ++kb) {
    const uint blk128 = kb >> 1u;
    const uint half64 = kb & 1u;
    const ushort4 raw_w = wc_row
        ? reinterpret_cast<device const ushort4 *>(wc_row + ulong(blk128) * 1024ul + half64 * 32ul)[sub4]
        : ushort4(0u);
    const uchar2 sw_v = (sub4 == 0u && ws_row)
        ? *reinterpret_cast<device const uchar2 *>(ws_row + ulong(blk128) * 64ul + half64 * 2ul)
        : uchar2(0u, 0u);

    const uchar4 f8_0 = bench_unpack_4nib_fp8(raw_w.x);
    const uchar4 f8_1 = bench_unpack_4nib_fp8(raw_w.y);
    const uchar4 f8_2 = bench_unpack_4nib_fp8(raw_w.z);
    const uchar4 f8_3 = bench_unpack_4nib_fp8(raw_w.w);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uchar4 *wd = reinterpret_cast<threadgroup uchar4 *>(tg_mem.Ws_fp8 + row64 * 64u + sub4 * 16u);
    wd[0] = f8_0; wd[1] = f8_1; wd[2] = f8_2; wd[3] = f8_3;
    if (sub4 == 0u) *reinterpret_cast<threadgroup uchar2 *>(tg_mem.Ws_sc + row64 * 2u) = sw_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    DvMxfp8 tA_full(const_cast<device uchar *>(a_fp8), dextents<int32_t, 2>(int(p.k), int(p.mp)),
                    DvUe8(const_cast<device uchar *>(a_sc)));
    auto tA = tA_full.slice(int(kb * 64u), int(rb_sg));
    TgMxfp8 tB(tg_mem.Ws_fp8 + sg_n * (32u * 64u), dextents<int32_t, 2>(64, 32),
               TgUe8(tg_mem.Ws_sc + sg_n * 64u));
    auto c = op.template get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(tA)>, TgMxfp8, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 8u; ++i) { c[i] = acc0[i]; c[8u + i] = acc1[i]; }
    op.run(tA, tB, c);
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 8u; ++i) { acc0[i] = c[i]; acc1[i] = c[8u + i]; }
  }

  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) | ((lane >> 1u) & 3u);
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
  const uint n0_sg = n0_tg + sg_n * 32u;
  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  if (rb_sg < p.rows && n0_sg < p.n) {
    const uint r0 = rb_sg + fm;
    const uint r1 = r0 + 8u;
    const uint c0 = n0_sg + fn;
    const uint c1 = c0 + 16u;
    if (r0 < p.rows && c0 < p.n)
      *reinterpret_cast<device bfloat4 *>(y + ulong(r0) * out_stride + p.row_offset + c0) =
          bfloat4(bfloat(acc0[0]), bfloat(acc0[1]), bfloat(acc0[2]), bfloat(acc0[3]));
    if (r1 < p.rows && c0 < p.n)
      *reinterpret_cast<device bfloat4 *>(y + ulong(r1) * out_stride + p.row_offset + c0) =
          bfloat4(bfloat(acc0[4]), bfloat(acc0[5]), bfloat(acc0[6]), bfloat(acc0[7]));
    if (r0 < p.rows && c1 < p.n)
      *reinterpret_cast<device bfloat4 *>(y + ulong(r0) * out_stride + p.row_offset + c1) =
          bfloat4(bfloat(acc1[0]), bfloat(acc1[1]), bfloat(acc1[2]), bfloat(acc1[3]));
    if (r1 < p.rows && c1 < p.n)
      *reinterpret_cast<device bfloat4 *>(y + ulong(r1) * out_stride + p.row_offset + c1) =
          bfloat4(bfloat(acc1[4]), bfloat(acc1[5]), bfloat(acc1[6]), bfloat(acc1[7]));
  }
}

// ============================================================================
// Reference Native W4A4 (Zero-Spill 0 B SRAM: 128t & 256t DvFp4 x DvFp4)
// ============================================================================
[[max_total_threads_per_threadgroup(128)]]
kernel void bench_w4a4_zero_spill_128t(
    device const uchar *a_codes [[buffer(0)]],
    device const uchar *a_scales [[buffer(1)]],
    device const uchar *w_codes [[buffer(2)]],
    device const uchar *w_scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant BenchGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  const uint2 pid_nm = bench_swizzle_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * 64u;
  const uint n0_tg = pid_nm.x * 64u;
  if (rb_tg >= p.rows || n0_tg >= p.n) return;
  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const uint rb_sg = rb_tg + sg_m * 32u;
  const uint n0_sg = n0_tg + sg_n * 32u;
  if (rb_sg >= p.rows || n0_sg >= p.n) return;

  constexpr auto desc = matmul2d_descriptor(
      16, 16, 128, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  vec<float, 8> acc[2][2];
  acc[0][0] = vec<float, 8>(0.0f); acc[0][1] = vec<float, 8>(0.0f);
  acc[1][0] = vec<float, 8>(0.0f); acc[1][1] = vec<float, 8>(0.0f);

  const uint k_tiles = p.k >> 7u;
  const uint m0_tile = rb_sg >> 4u;
  const uint g_n0_tile = (p.row_offset + n0_sg) >> 4u;
  const bool has_m1 = (rb_sg + 16u < p.rows);
  const bool has_n1 = (n0_sg + 16u < p.n);

  for (uint kt = 0u; kt < k_tiles; ++kt) {
    const ulong a0_idx = ulong(m0_tile) * k_tiles + kt;
    const ulong a1_idx = ulong(m0_tile + 1u) * k_tiles + kt;
    const ulong b0_idx = ulong(g_n0_tile) * k_tiles + kt;
    const ulong b1_idx = ulong(g_n0_tile + 1u) * k_tiles + kt;
    DvMxfp4 a0(const_cast<device uchar *>(a_codes + a0_idx * 1024ul), dextents<int32_t, 2>(128, 16),
               DvUe8(const_cast<device uchar *>(a_scales + a0_idx * 64ul)));
    DvMxfp4 b0(const_cast<device uchar *>(w_codes + b0_idx * 1024ul), dextents<int32_t, 2>(128, 16),
               DvUe8(const_cast<device uchar *>(w_scales + b0_idx * 64ul)));
    {
      auto c = op.get_destination_cooperative_tensor<DvMxfp4, DvMxfp4, float>();
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < 8; ++i) c[i] = acc[0][0][i];
      op.run(a0, b0, c);
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < 8; ++i) acc[0][0][i] = c[i];
    }
    if (has_n1) {
      DvMxfp4 b1(const_cast<device uchar *>(w_codes + b1_idx * 1024ul), dextents<int32_t, 2>(128, 16),
                 DvUe8(const_cast<device uchar *>(w_scales + b1_idx * 64ul)));
      {
        auto c = op.get_destination_cooperative_tensor<DvMxfp4, DvMxfp4, float>();
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < 8; ++i) c[i] = acc[0][1][i];
        op.run(a0, b1, c);
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < 8; ++i) acc[0][1][i] = c[i];
      }
      if (has_m1) {
        DvMxfp4 a1(const_cast<device uchar *>(a_codes + a1_idx * 1024ul), dextents<int32_t, 2>(128, 16),
                   DvUe8(const_cast<device uchar *>(a_scales + a1_idx * 64ul)));
        auto c = op.get_destination_cooperative_tensor<DvMxfp4, DvMxfp4, float>();
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < 8; ++i) c[i] = acc[1][1][i];
        op.run(a1, b1, c);
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < 8; ++i) acc[1][1][i] = c[i];
      }
    }
    if (has_m1) {
      DvMxfp4 a1(const_cast<device uchar *>(a_codes + a1_idx * 1024ul), dextents<int32_t, 2>(128, 16),
                 DvUe8(const_cast<device uchar *>(a_scales + a1_idx * 64ul)));
      auto c = op.get_destination_cooperative_tensor<DvMxfp4, DvMxfp4, float>();
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < 8; ++i) c[i] = acc[1][0][i];
      op.run(a1, b0, c);
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < 8; ++i) acc[1][0][i] = c[i];
    }
  }
  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) | ((lane >> 1u) & 3u);
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  #pragma clang loop unroll(full)
  for (uint mm = 0u; mm < 2u; ++mm)
    #pragma clang loop unroll(full)
    for (uint nn = 0u; nn < 2u; ++nn) {
      const uint r0 = rb_sg + mm * 16u + fm;
      const uint c0 = n0_sg + nn * 16u + fn;
      if (r0 < p.rows && c0 < p.n)
        *reinterpret_cast<device bfloat4 *>(y + ulong(r0) * out_stride + p.row_offset + c0) =
            bfloat4(bfloat(acc[mm][nn][0]), bfloat(acc[mm][nn][1]), bfloat(acc[mm][nn][2]), bfloat(acc[mm][nn][3]));
      if (r0 + 8u < p.rows && c0 < p.n)
        *reinterpret_cast<device bfloat4 *>(y + ulong(r0 + 8u) * out_stride + p.row_offset + c0) =
            bfloat4(bfloat(acc[mm][nn][4]), bfloat(acc[mm][nn][5]), bfloat(acc[mm][nn][6]), bfloat(acc[mm][nn][7]));
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void bench_w4a4_zero_spill_256t(
    device const uchar *a_codes [[buffer(0)]],
    device const uchar *a_scales [[buffer(1)]],
    device const uchar *w_codes [[buffer(2)]],
    device const uchar *w_scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant BenchGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  const uint2 pid_nm = bench_swizzle_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * 64u;
  const uint n0_tg = pid_nm.x * 64u;
  if (rb_tg >= p.rows || n0_tg >= p.n) return;
  const uint sg_m = sg >> 1u; // 0..3 -> SM=16
  const uint sg_n = sg & 1u;  // 0..1 -> SN=32
  const uint rb_sg = rb_tg + sg_m * 16u;
  const uint n0_sg = n0_tg + sg_n * 32u;
  if (rb_sg >= p.rows || n0_sg >= p.n) return;

  constexpr auto desc = matmul2d_descriptor(
      16, 16, 128, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  vec<float, 8> acc0(0.0f), acc1(0.0f);

  const uint k_tiles = p.k >> 7u;
  const uint m_tile = rb_sg >> 4u;
  const uint g_n0_tile = (p.row_offset + n0_sg) >> 4u;
  const bool has_n1 = (n0_sg + 16u < p.n);

  for (uint kt = 0u; kt < k_tiles; ++kt) {
    const ulong a_idx = ulong(m_tile) * k_tiles + kt;
    const ulong b0_idx = ulong(g_n0_tile) * k_tiles + kt;
    const ulong b1_idx = ulong(g_n0_tile + 1u) * k_tiles + kt;
    DvMxfp4 a0(const_cast<device uchar *>(a_codes + a_idx * 1024ul), dextents<int32_t, 2>(128, 16),
               DvUe8(const_cast<device uchar *>(a_scales + a_idx * 64ul)));
    DvMxfp4 b0(const_cast<device uchar *>(w_codes + b0_idx * 1024ul), dextents<int32_t, 2>(128, 16),
               DvUe8(const_cast<device uchar *>(w_scales + b0_idx * 64ul)));
    {
      auto c = op.get_destination_cooperative_tensor<DvMxfp4, DvMxfp4, float>();
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < 8; ++i) c[i] = acc0[i];
      op.run(a0, b0, c);
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < 8; ++i) acc0[i] = c[i];
    }
    if (has_n1) {
      DvMxfp4 b1(const_cast<device uchar *>(w_codes + b1_idx * 1024ul), dextents<int32_t, 2>(128, 16),
                 DvUe8(const_cast<device uchar *>(w_scales + b1_idx * 64ul)));
      auto c = op.get_destination_cooperative_tensor<DvMxfp4, DvMxfp4, float>();
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < 8; ++i) c[i] = acc1[i];
      op.run(a0, b1, c);
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < 8; ++i) acc1[i] = c[i];
    }
  }
  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) | ((lane >> 1u) & 3u);
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  const uint r0 = rb_sg + fm;
  const uint r1 = r0 + 8u;
  const uint c0 = n0_sg + fn;
  const uint c1 = c0 + 16u;
  if (r0 < p.rows && c0 < p.n)
    *reinterpret_cast<device bfloat4 *>(y + ulong(r0) * out_stride + p.row_offset + c0) =
        bfloat4(bfloat(acc0[0]), bfloat(acc0[1]), bfloat(acc0[2]), bfloat(acc0[3]));
  if (r1 < p.rows && c0 < p.n)
    *reinterpret_cast<device bfloat4 *>(y + ulong(r1) * out_stride + p.row_offset + c0) =
        bfloat4(bfloat(acc0[4]), bfloat(acc0[5]), bfloat(acc0[6]), bfloat(acc0[7]));
  if (r0 < p.rows && c1 < p.n)
    *reinterpret_cast<device bfloat4 *>(y + ulong(r0) * out_stride + p.row_offset + c1) =
        bfloat4(bfloat(acc1[0]), bfloat(acc1[1]), bfloat(acc1[2]), bfloat(acc1[3]));
  if (r1 < p.rows && c1 < p.n)
    *reinterpret_cast<device bfloat4 *>(y + ulong(r1) * out_stride + p.row_offset + c1) =
        bfloat4(bfloat(acc1[4]), bfloat(acc1[5]), bfloat(acc1[6]), bfloat(acc1[7]));
}

[[max_total_threads_per_threadgroup(32)]]
kernel void bench_quantize_a4_bm16_bk128(
    device const bfloat *src [[buffer(0)]],
    device uchar *dst_codes [[buffer(1)]],
    device uchar *dst_scales [[buffer(2)]],
    constant BenchQuantParams &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint g = tg.x;
  const uint row = tg.y;
  const uint groups = p.k >> 5u;
  if (g >= groups || row >= p.mp) return;
  const uint k_blocks = p.k >> 7u;
  const ulong tile_idx = ulong(row >> 4u) * ulong(k_blocks) + ulong(g >> 2u);
  const uint rm = row & 15u;
  const uint sub_g = g & 3u;
  device uchar *c_out = dst_codes + tile_idx * 1024ul + ulong(rm) * 64ul + ulong(sub_g) * 16ul;
  device uchar *s_out = dst_scales + tile_idx * 64ul + ulong(rm) * 4ul + ulong(sub_g);
  if (row >= p.rows) {
    if ((lane & 1u) == 0u) c_out[lane >> 1u] = 0u;
    if (lane == 0u) *s_out = 0x7fu;
    return;
  }
  const float v = float(src[ulong(row) * p.k + g * 32u + lane]);
  const float maximum = simd_max(fabs(v));
  const uchar sc = nax_encode_e8m0_round_up(maximum / 6.0f);
  const float scale = nax_decode_e8m0(sc);
  const float inv_scale = scale > 0.0f ? (1.0f / scale) : 0.0f;
  const uchar c_lo = nax_encode_e2m1(v * inv_scale);
  const uchar c_hi = simd_shuffle_down(c_lo, 1u);
  if ((lane & 1u) == 0u) c_out[lane >> 1u] = uchar(c_lo | (c_hi << 4u));
  if (lane == 0u) *s_out = sc;
}

// ============================================================================
// Winning MXFP4 W4A8 Architecture (128t, BM=64, BN=64, BK=64, stride=64, 8,976 B SRAM):
//   - 2 KiB constant-cache pre-shifted byte LUT (kBenchE2m1ByteLutSh[4][256])
//   - Per-row |de| <= 3 shift-merge (100% bit-exact for |de| <= 3 & |de| > 3, zero re-unpack)
//   - Single 32x32x32 NAX op (mode::multiply + multiply_accumulate -> 1 rescale/K=64)
//   - Exact verified Apple10 32x32 ci map (mm=i>>4, u=(i>>3)&1, h=(i>>2)&1, c=i&3)
//   - Outer blk128 loop (1 uchar4 scale load per 128 K-elements)
// ============================================================================
#define B_E2M1_NIB(n, s) (int8_t(((n) & 8) ? -((((0xC8643210u >> (((n) & 7) * 4)) & 0xFu) << (s))) \
                                           :  ((((0xC8643210u >> (((n) & 7) * 4)) & 0xFu) << (s)))))
#define B_E2M1_PAIR(b, s) ushort(uchar(B_E2M1_NIB((b) & 15, s)) | (ushort(uchar(B_E2M1_NIB(((b) >> 4) & 15, s))) << 8))
#define B_E2M1_ROW16(hi, s) \
  B_E2M1_PAIR((hi)*16+0, s), B_E2M1_PAIR((hi)*16+1, s), B_E2M1_PAIR((hi)*16+2, s), B_E2M1_PAIR((hi)*16+3, s), \
  B_E2M1_PAIR((hi)*16+4, s), B_E2M1_PAIR((hi)*16+5, s), B_E2M1_PAIR((hi)*16+6, s), B_E2M1_PAIR((hi)*16+7, s), \
  B_E2M1_PAIR((hi)*16+8, s), B_E2M1_PAIR((hi)*16+9, s), B_E2M1_PAIR((hi)*16+10, s), B_E2M1_PAIR((hi)*16+11, s), \
  B_E2M1_PAIR((hi)*16+12, s), B_E2M1_PAIR((hi)*16+13, s), B_E2M1_PAIR((hi)*16+14, s), B_E2M1_PAIR((hi)*16+15, s)
#define B_E2M1_TBL256(s) \
  { B_E2M1_ROW16(0, s), B_E2M1_ROW16(1, s), B_E2M1_ROW16(2, s), B_E2M1_ROW16(3, s), \
    B_E2M1_ROW16(4, s), B_E2M1_ROW16(5, s), B_E2M1_ROW16(6, s), B_E2M1_ROW16(7, s), \
    B_E2M1_ROW16(8, s), B_E2M1_ROW16(9, s), B_E2M1_ROW16(10, s), B_E2M1_ROW16(11, s), \
    B_E2M1_ROW16(12, s), B_E2M1_ROW16(13, s), B_E2M1_ROW16(14, s), B_E2M1_ROW16(15, s) }

constant ushort kBenchE2m1ByteLutSh[4][256] = {
    B_E2M1_TBL256(0), B_E2M1_TBL256(1), B_E2M1_TBL256(2), B_E2M1_TBL256(3)
};
constant char2 kBenchE2m1Lut512[256] = {
#define B_C2(b) char2(B_E2M1_NIB((b) & 15, 0), B_E2M1_NIB(((b) >> 4) & 15, 0))
#define B_C2_R16(hi) \
  B_C2((hi)*16+0), B_C2((hi)*16+1), B_C2((hi)*16+2), B_C2((hi)*16+3), \
  B_C2((hi)*16+4), B_C2((hi)*16+5), B_C2((hi)*16+6), B_C2((hi)*16+7), \
  B_C2((hi)*16+8), B_C2((hi)*16+9), B_C2((hi)*16+10), B_C2((hi)*16+11), \
  B_C2((hi)*16+12), B_C2((hi)*16+13), B_C2((hi)*16+14), B_C2((hi)*16+15)
  B_C2_R16(0), B_C2_R16(1), B_C2_R16(2), B_C2_R16(3),
  B_C2_R16(4), B_C2_R16(5), B_C2_R16(6), B_C2_R16(7),
  B_C2_R16(8), B_C2_R16(9), B_C2_R16(10), B_C2_R16(11),
  B_C2_R16(12), B_C2_R16(13), B_C2_R16(14), B_C2_R16(15)
#undef B_C2_R16
#undef B_C2
};

inline uint bench_unpack4_lut512_sh(ushort raw16, int8_t sh) {
  const char2 p0 = kBenchE2m1Lut512[raw16 & 0xFFu];
  const char2 p1 = kBenchE2m1Lut512[raw16 >> 8u];
  const char4 v = char4(p0.x, p0.y, p1.x, p1.y) << sh;
  return as_type<uint>(v);
}

struct WinSmLutTg {
  alignas(16) int8_t As_i8[64u * 64u]; // 4096 B
  alignas(16) int8_t Ws_i8[64u * 64u]; // 4096 B
  float Sa_s[64u];                     //  256 B
  float Sw0_s[64u];                    //  256 B
  float Sw1_s[64u];                    //  256 B
  uint  slow_flag[2u];                 //    8 B
};
static_assert(sizeof(WinSmLutTg) == 8976ul, "WinSmLutTg == 8976 B");

template <bool UseLut512>
inline void bench_alpha_win_sm_impl(
    device const int8_t *qa,
    device const float *sa,
    device const uchar *codes,
    device const uchar *scales,
    device bfloat *y,
    constant BenchGemmParams &p,
    threadgroup WinSmLutTg &tg_mem,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 grid_tgs) {
  const uint2 pid = bench_swizzle_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid.y * 64u, n0_tg = pid.x * 64u;
  if (rb_tg >= p.rows || n0_tg >= p.n) return;

  const uint tid = sg * 32u + lane;
  const uint sg_m = sg >> 1u, sg_n = sg & 1u;
  const uint blocks128 = p.k >> 7u;
  const uint col_limit = p.row_offset + p.n;
  const uint w_row = tid >> 1u, w_sub = tid & 1u;
  const uint gw = p.row_offset + n0_tg + w_row;
  device const uchar *wc_row = (gw < col_limit)
      ? (codes + ulong(gw >> 4u) * ulong(blocks128) * 1024ul + ulong(gw & 15u) * 64ul) : nullptr;
  device const uchar *ws_row = (gw < col_limit)
      ? (scales + ulong(gw >> 4u) * ulong(blocks128) * 64ul + ulong(gw & 15u) * 4ul) : nullptr;
  const uint a_r0 = tid >> 2u, a_col16 = (tid & 3u) * 16u;
  const bool full_m = (rb_tg + 64u <= p.rows);

  constexpr auto desc_m = matmul2d_descriptor(32, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto desc_a = matmul2d_descriptor(32, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_m, execution_simdgroup> op_m;
  matmul2d<desc_a, execution_simdgroup> op_a;
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(tg_mem.As_i8, dextents<int32_t, 2>(64, 64));
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(tg_mem.Ws_i8, dextents<int32_t, 2>(64, 64));

  float acc[32];
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 32u; ++i) acc[i] = 0.0f;

  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) | ((lane >> 1u) & 3u);
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
  const uint sw_base = sg_n * 32u + fn;

  auto run_k64 = [&](uint kb, uchar2 ep, uint4 raw16) {
    const uint k0 = kb * 64u;
    const uint4 qa0 = (full_m || rb_tg + a_r0 < p.rows)
        ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0) * p.k + k0 + a_col16) : uint4(0u);
    const uint4 qa1 = (full_m || rb_tg + a_r0 + 32u < p.rows)
        ? *reinterpret_cast<device const uint4 *>(qa + ulong(rb_tg + a_r0 + 32u) * p.k + k0 + a_col16) : uint4(0u);
    const float sa_v = (tid < 64u && (full_m || rb_tg + tid < p.rows)) ? sa[ulong(kb) * p.mp + rb_tg + tid] : 0.0f;

    const int de = (ws_row && ep.x != 255u && ep.y != 255u) ? (int(ep.y) - int(ep.x)) : (ws_row ? 255 : 0);
    const bool row_fast = (abs(de) <= 3);
    const uint sh = row_fast ? ((w_sub == 0u) ? ((de < 0) ? uint(-de) : 0u) : ((de > 0) ? uint(de) : 0u)) : 0u;

    uint4 wu0, wu1;
    if constexpr (UseLut512) {
      const int8_t sh8 = int8_t(sh);
      wu0.x = bench_unpack4_lut512_sh(ushort(raw16.x), sh8);
      wu0.y = bench_unpack4_lut512_sh(ushort(raw16.x >> 16u), sh8);
      wu0.z = bench_unpack4_lut512_sh(ushort(raw16.y), sh8);
      wu0.w = bench_unpack4_lut512_sh(ushort(raw16.y >> 16u), sh8);
      wu1.x = bench_unpack4_lut512_sh(ushort(raw16.z), sh8);
      wu1.y = bench_unpack4_lut512_sh(ushort(raw16.z >> 16u), sh8);
      wu1.z = bench_unpack4_lut512_sh(ushort(raw16.w), sh8);
      wu1.w = bench_unpack4_lut512_sh(ushort(raw16.w >> 16u), sh8);
    } else {
      constant ushort *lut = kBenchE2m1ByteLutSh[sh];
      wu0.x = uint(lut[(raw16.x >>  0u) & 0xFFu]) | (uint(lut[(raw16.x >>  8u) & 0xFFu]) << 16u);
      wu0.y = uint(lut[(raw16.x >> 16u) & 0xFFu]) | (uint(lut[(raw16.x >> 24u) & 0xFFu]) << 16u);
      wu0.z = uint(lut[(raw16.y >>  0u) & 0xFFu]) | (uint(lut[(raw16.y >>  8u) & 0xFFu]) << 16u);
      wu0.w = uint(lut[(raw16.y >> 16u) & 0xFFu]) | (uint(lut[(raw16.y >> 24u) & 0xFFu]) << 16u);
      wu1.x = uint(lut[(raw16.z >>  0u) & 0xFFu]) | (uint(lut[(raw16.z >>  8u) & 0xFFu]) << 16u);
      wu1.y = uint(lut[(raw16.z >> 16u) & 0xFFu]) | (uint(lut[(raw16.z >> 24u) & 0xFFu]) << 16u);
      wu1.z = uint(lut[(raw16.w >>  0u) & 0xFFu]) | (uint(lut[(raw16.w >>  8u) & 0xFFu]) << 16u);
      wu1.w = uint(lut[(raw16.w >> 16u) & 0xFFu]) | (uint(lut[(raw16.w >> 24u) & 0xFFu]) << 16u);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2u) tg_mem.slow_flag[tid] = 0u;
    *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + a_r0 * 64u + a_col16) = qa0;
    *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + (a_r0 + 32u) * 64u + a_col16) = qa1;
    *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + w_row * 64u + w_sub * 32u + 0u) = wu0;
    *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + w_row * 64u + w_sub * 32u + 16u) = wu1;
    if (w_sub == 0u) {
      if (row_fast) {
        const float sw = ws_row ? (0.5f * bench_e8m0_safe(min(ep.x, ep.y))) : 0.0f;
        tg_mem.Sw0_s[w_row] = sw;
        tg_mem.Sw1_s[w_row] = sw;
      } else {
        tg_mem.Sw0_s[w_row] = ws_row ? (0.5f * bench_e8m0_safe(ep.x)) : 0.0f;
        tg_mem.Sw1_s[w_row] = ws_row ? (0.5f * bench_e8m0_safe(ep.y)) : 0.0f;
        tg_mem.slow_flag[w_row >> 5u] = 1u;
      }
    }
    if (tid < 64u) tg_mem.Sa_s[tid] = sa_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto a0 = tA.slice(0, int(sg_m * 32u));
    auto b0 = tB.slice(0, int(sg_n * 32u));
    auto ci = op_m.template get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();
    const float sa0 = tg_mem.Sa_s[sg_m * 32u + fm];
    const float sa1 = tg_mem.Sa_s[sg_m * 32u + 8u + fm];
    const float sa2 = tg_mem.Sa_s[sg_m * 32u + 16u + fm];
    const float sa3 = tg_mem.Sa_s[sg_m * 32u + 24u + fm];

    if (tg_mem.slow_flag[sg_n] == 0u) {
      auto a1 = tA.slice(32, int(sg_m * 32u));
      auto b1 = tB.slice(32, int(sg_n * 32u));
      op_m.run(a0, b0, ci);
      op_a.run(a1, b1, ci);
      #pragma clang loop unroll(full)
      for (uint c = 0u; c < 4u; ++c) {
        const float sw0 = tg_mem.Sw0_s[sw_base + c];
        const float sw1 = tg_mem.Sw0_s[sw_base + 16u + c];
        acc[ 0u + c] = fma(sa0 * sw0, float(ci[ 0u + c]), acc[ 0u + c]);
        acc[ 4u + c] = fma(sa1 * sw0, float(ci[ 4u + c]), acc[ 4u + c]);
        acc[ 8u + c] = fma(sa0 * sw1, float(ci[ 8u + c]), acc[ 8u + c]);
        acc[12u + c] = fma(sa1 * sw1, float(ci[12u + c]), acc[12u + c]);
        acc[16u + c] = fma(sa2 * sw0, float(ci[16u + c]), acc[16u + c]);
        acc[20u + c] = fma(sa3 * sw0, float(ci[20u + c]), acc[20u + c]);
        acc[24u + c] = fma(sa2 * sw1, float(ci[24u + c]), acc[24u + c]);
        acc[28u + c] = fma(sa3 * sw1, float(ci[28u + c]), acc[28u + c]);
      }
    } else {
      #pragma clang loop unroll(disable)
      for (uint hk = 0u; hk < 2u; ++hk) {
        auto ah = tA.slice(int(hk * 32u), int(sg_m * 32u));
        auto bh = tB.slice(int(hk * 32u), int(sg_n * 32u));
        op_m.run(ah, bh, ci);
        threadgroup const float *swp = (hk == 0u) ? tg_mem.Sw0_s : tg_mem.Sw1_s;
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) {
          const float sw0 = swp[sw_base + c];
          const float sw1 = swp[sw_base + 16u + c];
          acc[ 0u + c] = fma(sa0 * sw0, float(ci[ 0u + c]), acc[ 0u + c]);
          acc[ 4u + c] = fma(sa1 * sw0, float(ci[ 4u + c]), acc[ 4u + c]);
          acc[ 8u + c] = fma(sa0 * sw1, float(ci[ 8u + c]), acc[ 8u + c]);
          acc[12u + c] = fma(sa1 * sw1, float(ci[12u + c]), acc[12u + c]);
          acc[16u + c] = fma(sa2 * sw0, float(ci[16u + c]), acc[16u + c]);
          acc[20u + c] = fma(sa3 * sw0, float(ci[20u + c]), acc[20u + c]);
          acc[24u + c] = fma(sa2 * sw1, float(ci[24u + c]), acc[24u + c]);
          acc[28u + c] = fma(sa3 * sw1, float(ci[28u + c]), acc[28u + c]);
        }
      }
    }
  };

  for (uint blk128 = 0u; blk128 < blocks128; ++blk128) {
    const uchar4 ep4 = ws_row ? *reinterpret_cast<device const uchar4 *>(ws_row + ulong(blk128) * 64ul) : uchar4(0u);
    device const uint4 *wcp = wc_row ? reinterpret_cast<device const uint4 *>(wc_row + ulong(blk128) * 1024ul) : nullptr;
    run_k64(blk128 * 2u + 0u, ep4.xy, wcp ? wcp[w_sub] : uint4(0u));
    run_k64(blk128 * 2u + 1u, ep4.zw, wcp ? wcp[2u + w_sub] : uint4(0u));
  }

  const uint rb_sg = rb_tg + sg_m * 32u, n0_sg = n0_tg + sg_n * 32u;
  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  if (rb_sg < p.rows && n0_sg < p.n) {
    #pragma clang loop unroll(full)
    for (uint mm = 0u; mm < 2u; ++mm)
      #pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const uint row = rb_sg + mm * 16u + fm + h * 8u;
        if (row < p.rows) {
          #pragma clang loop unroll(full)
          for (uint u = 0u; u < 2u; ++u) {
            const uint lcol = n0_sg + u * 16u + fn;
            if (lcol < p.n) {
              const uint e = mm * 16u + u * 8u + h * 4u;
              *reinterpret_cast<device bfloat4 *>(y + ulong(row) * out_stride + p.row_offset + lcol) =
                  bfloat4(bfloat(acc[e+0]), bfloat(acc[e+1]), bfloat(acc[e+2]), bfloat(acc[e+3]));
            }
          }
        }
      }
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void bench_alpha_win_sm_lut_32x32(
    device const int8_t *qa [[buffer(0)]], device const float *sa [[buffer(1)]],
    device const uchar *codes [[buffer(2)]], device const uchar *scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]], constant BenchGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup WinSmLutTg tg_mem;
  bench_alpha_win_sm_impl<false>(qa, sa, codes, scales, y, p, tg_mem, lane, sg, tg, grid_tgs);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void bench_alpha_win_sm_lut512_32x32(
    device const int8_t *qa [[buffer(0)]], device const float *sa [[buffer(1)]],
    device const uchar *codes [[buffer(2)]], device const uchar *scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]], constant BenchGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup WinSmLutTg tg_mem;
  bench_alpha_win_sm_impl<true>(qa, sa, codes, scales, y, p, tg_mem, lane, sg, tg, grid_tgs);
}

// ============================================================================
// Winning Native MXFP4 W4A4 Architecture (128t, BM=64, BN=64, BK=128 as 4x32 planes, 8,720 B, 0 spills!)
// ============================================================================
using TgMxfp4 = tensor<threadgroup metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline, TgUe8>;

struct WinW4A4Tg {
  alignas(16) uchar As_c[4u][64u * 16u]; // 4096 B
  alignas(16) uchar Ws_c[4u][64u * 16u]; // 4096 B
  alignas(16) uchar As_s[4u][64u];       //  256 B
  alignas(16) uchar Ws_s[4u][64u];       //  256 B
  uint meta[2u];                         //    8 B (padded to 16 B)
};
static_assert(sizeof(WinW4A4Tg) == 8720ul, "WinW4A4Tg == 8720 B");

[[max_total_threads_per_threadgroup(128)]]
kernel void bench_w4a4_win_tg_128p(
    device const uchar *a_codes [[buffer(0)]],
    device const uchar *a_scales [[buffer(1)]],
    device const uchar *w_codes [[buffer(2)]],
    device const uchar *w_scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant BenchGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup WinW4A4Tg tg_mem;
  const uint2 pid = bench_swizzle_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid.y * 64u, n0_tg = pid.x * 64u;
  if (rb_tg >= p.rows || n0_tg >= p.n) return;
  if (sg == 0u && lane == 0u) { tg_mem.meta[0] = rb_tg; tg_mem.meta[1] = n0_tg; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 32, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  TgMxfp4 ta_init(tg_mem.As_c[0], dextents<int32_t, 2>(32, 16), TgUe8(tg_mem.As_s[0]));
  TgMxfp4 tb_init(tg_mem.Ws_c[0], dextents<int32_t, 2>(32, 32), TgUe8(tg_mem.Ws_s[0]));
  auto c0 = op.get_destination_cooperative_tensor<TgMxfp4, TgMxfp4, float>();
  auto c1 = op.get_destination_cooperative_tensor<TgMxfp4, TgMxfp4, float>();
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 16u; ++i) { c0[i] = 0.0f; c1[i] = 0.0f; }

  const uint k_tiles = p.k >> 7u;
  const uint tid = sg * 32u + lane;
  const uint sg_m = sg >> 1u, sg_n = sg & 1u;

  for (uint kt = 0u; kt < k_tiles; ++kt) {
    const uint r = tid >> 1u, h = tid & 1u;
    const uint gr = tg_mem.meta[0] + r;
    const uint gw = p.row_offset + tg_mem.meta[1] + r;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gr < p.mp) {
      const ulong ai = ulong(gr >> 4u) * k_tiles + kt;
      device const uint4 *ap = reinterpret_cast<device const uint4 *>(a_codes + ai * 1024ul + (gr & 15u) * 64ul);
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_c[h * 2u + 0u] + r * 16u) = ap[h * 2u + 0u];
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_c[h * 2u + 1u] + r * 16u) = ap[h * 2u + 1u];
      const ushort s2 = reinterpret_cast<device const ushort *>(a_scales + ai * 64ul + (gr & 15u) * 4ul)[h];
      tg_mem.As_s[h * 2u + 0u][r] = uchar(s2 & 0xFFu);
      tg_mem.As_s[h * 2u + 1u][r] = uchar(s2 >> 8u);
    } else {
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_c[h * 2u + 0u] + r * 16u) = uint4(0u);
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_c[h * 2u + 1u] + r * 16u) = uint4(0u);
      tg_mem.As_s[h * 2u + 0u][r] = 0x7Fu;
      tg_mem.As_s[h * 2u + 1u][r] = 0x7Fu;
    }
    if (gw < p.row_offset + p.n) {
      const ulong bi = ulong(gw >> 4u) * k_tiles + kt;
      device const uint4 *wp = reinterpret_cast<device const uint4 *>(w_codes + bi * 1024ul + (gw & 15u) * 64ul);
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_c[h * 2u + 0u] + r * 16u) = wp[h * 2u + 0u];
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_c[h * 2u + 1u] + r * 16u) = wp[h * 2u + 1u];
      const ushort s2 = reinterpret_cast<device const ushort *>(w_scales + bi * 64ul + (gw & 15u) * 4ul)[h];
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
      TgMxfp4 tA0(tg_mem.As_c[pl] + (sg_m * 32u + 0u) * 16u,  dextents<int32_t, 2>(32, 16), TgUe8(tg_mem.As_s[pl] + sg_m * 32u));
      TgMxfp4 tA1(tg_mem.As_c[pl] + (sg_m * 32u + 16u) * 16u, dextents<int32_t, 2>(32, 16), TgUe8(tg_mem.As_s[pl] + sg_m * 32u + 16u));
      TgMxfp4 tB(tg_mem.Ws_c[pl] + sg_n * 32u * 16u, dextents<int32_t, 2>(32, 32), TgUe8(tg_mem.Ws_s[pl] + sg_n * 32u));
      op.run(tA0, tB, c0);
      op.run(tA1, tB, c1);
    }
  }

  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) | ((lane >> 1u) & 3u);
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
  const uint rb_sg = tg_mem.meta[0] + sg_m * 32u, n0_sg = tg_mem.meta[1] + sg_n * 32u;
  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;
  if (rb_sg < p.rows && n0_sg < p.n) {
    #pragma clang loop unroll(full)
    for (uint h2 = 0u; h2 < 2u; ++h2) {
      const uint r0 = rb_sg + fm + h2 * 8u, r1 = r0 + 16u;
      #pragma clang loop unroll(full)
      for (uint u = 0u; u < 2u; ++u) {
        const uint lc = n0_sg + u * 16u + fn;
        if (lc < p.n) {
          const uint e = u * 8u + h2 * 4u;
          if (r0 < p.rows)
            *reinterpret_cast<device bfloat4 *>(y + ulong(r0) * out_stride + p.row_offset + lc) =
                bfloat4(bfloat(c0[e]), bfloat(c0[e+1]), bfloat(c0[e+2]), bfloat(c0[e+3]));
          if (r1 < p.rows)
            *reinterpret_cast<device bfloat4 *>(y + ulong(r1) * out_stride + p.row_offset + lc) =
                bfloat4(bfloat(c1[e]), bfloat(c1[e+1]), bfloat(c1[e+2]), bfloat(c1[e+3]));
        }
      }
    }
  }
}
