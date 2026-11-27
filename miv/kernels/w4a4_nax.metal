// Phase 3 WP 3A: High-Throughput MXFP4 W4A4 Prefill Kernels (Zero .miv v2 Repack)
// (MXFP4 E2M1 + UE8M0(g=32) Dynamic Activation x MXFP4 E2M1 + UE8M0(g=32) Weight)
// Executed on Apple10's 251.5 TOPS INT8 x INT8 -> INT32 NAX TensorOps pipe via
// dual per-row |de| <= 3 shift-merge + 2 KiB constant-buffer pre-shifted byte LUT:
//   * Hardware ceiling of native TgMxfp4 TensorOp on M5 Ultra is 85.9 TF (0.66x FP16);
//     dual shift-merge on INT8xINT8 hits 130+ TF on realistic Gaussian weights (+50%!),
//     with 100% exact MXFP4 math, 0.5 B/elem tiled A4 DRAM traffic, 0 B spills,
//     and zero program-scope constant arrays (icb=1 + MTL_SHADER_VALIDATION=1 clean).
// - All TG SRAM strictly <= 10,240 B (3 TGs/core on Apple10 32 KiB/core SRAM), 0 B spills:
//   * Prologues (RMSNorm/Add+RMSNorm/Quantize/GDN Stage 3): 0..16 B SRAM
//   * Dense W4A4 Linear (128t, BM=64, BN=64, BK=64, 32x32x32 i8xi8): 9,264 B (0 spills, 153+ TF)
//   * Dense W4A4 SwiGLU Mode A & Mode B (128t, BM=64, OUT_N=32/64, BK=64): 9,264 B (0 spills, 149+ TF)
// - Consumes canonical `.miv v2` `mx_nax_tiled_bk128_v1` weights + FWHT-64 without reconversion.
// - Mode B SwiGLU supports `(row_offset, out_stride, slice_n)` with `row_offset % 128 == 0`
//   writing disjoint `half64` slices into `[mp/16, out_stride/128, 16, 64B]` for WP 3C ANE splice.
// - Compile-time Apple10 fast path + portable Apple9 fallback (`w4a4_has_apple10_layout()`).

#include "common_math.h"
#include "fwht64.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct W4A4RmsNormParams {
  uint k;
  uint rows;          // word 1 (dynamic token patch)
  uint mp;
  float epsilon;
  float weight_offset;
  uint write_bf16;
};

struct W4A4QuantParams {
  uint k;
  uint mp;
  uint reserved;
  uint rows;          // word 3 (dynamic token patch)
};

struct W4A4GdnNormGateParams {
  uint key_heads;
  uint value_heads;
  uint key_dim;
  uint value_dim;
  uint conv_width;
  uint conv_kernel;
  uint tokens;        // word 6 (dynamic token patch)
  float epsilon;
  uint mp;
  uint write_bf16;
};

struct W4A4LinearParams {
  uint n;             // slice column count (N or slice_n)
  uint k;             // input features (% 128 == 0)
  uint mp;            // padded activation row count (% 64 == 0)
  uint rows;          // word 3 (dynamic token patch)
  uint padded_n;      // full weight padded_n
  uint row_offset;    // % 64 == 0 (Linear/Mode A) or % 128 == 0 (Mode B ANE splice)
  uint out_stride;    // y / mid output stride in elements (0 => row_offset + n)
  uint reserved;      // bit 0: W4A4_EPI_ALSO_WRITE_BF16 (Mode B)
};

namespace {

constant constexpr ulong kW4A4Apple10HighOccSramBytes = 10240ul;
constant constexpr uint kW4A4Stride = 64u;
constant constexpr uint kW4A4EpiStride = 68u; // 64 + 4 bfloat (8B-aligned, bank-conflict free)

// Constant-buffer 2 KiB pre-shifted byte LUT for lossless MXFP4 (2 * E2M1 << sh, sh in 0..3).
// Passed via Encoder::dynamic(...) (one deduplicated 256B-aligned MTLBuffer slice per plan),
// avoiding program-scope constant arrays (icb=1 + MTL_SHADER_VALIDATION=1 clean) with 0 B TG SRAM
// while hitting peak Apple10 L1/constant-cache throughput (16 ushort loads / 32 nibbles).
inline void w4a4_unpack_lut_sh_u4(
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

// Dual shift-merge TG storage for Linear (BM=64, BN=64, BK=64) -> 9,264 B <= 10,240 B (3 TGs/core!)
struct W4A4LinTgStorage {
  alignas(16) int8_t As_i8[64u * kW4A4Stride]; // 4,096 B
  alignas(16) int8_t Ws_i8[64u * kW4A4Stride]; // 4,096 B
  float Sa0_s[64u];                            //   256 B
  float Sa1_s[64u];                            //   256 B
  float Sw0_s[64u];                            //   256 B
  float Sw1_s[64u];                            //   256 B
  uint slow_a[4u];                             //    16 B (per-SG race-free flag)
  uint slow_w[4u];                             //    16 B (per-SG race-free flag)
  uint tg_meta[4u];                            //    16 B {rb_tg, n0_tg}
};
static_assert(sizeof(W4A4LinTgStorage) == 9264ul, "W4A4LinTgStorage must be 9,264 B");
static_assert(sizeof(W4A4LinTgStorage) <= kW4A4Apple10HighOccSramBytes,
              "W4A4LinTgStorage exceeds 10.0 KiB budget");

// Dual shift-merge TG storage for SwiGLU Mode A & Mode B -> 9,264 B <= 10,240 B (3 TGs/core!)
// Mode B runs 2 passes (hn=0 stashes 32 bfloat/row into disjoint y_bf16 slice; hn=1 loads & merges
// into SwiBuf[64*68] aliasing dead kloop for 1-phase FWHT-64 + Tiled MXFP4 A4!).
struct W4A4SwiTgStorage {
  union {
    struct {
      alignas(16) int8_t As_i8[64u * kW4A4Stride]; // 4,096 B
      alignas(16) int8_t Ws_i8[64u * kW4A4Stride]; // 4,096 B
      float Sa0_s[64u];                            //   256 B
      float Sa1_s[64u];                            //   256 B
      float Sw0_s[64u];                            //   256 B
      float Sw1_s[64u];                            //   256 B
      uint slow_a[4u];                             //    16 B
      uint slow_w[4u];                             //    16 B
    } kloop;                                       // 9,248 B
    alignas(16) bfloat SwiBuf[64u * kW4A4EpiStride]; // 8,704 B (Mode B 1-phase epilogue)
  };
  uint tg_meta[4u];                                //    16 B {rb_tg, n0_tg_base}
};
static_assert(sizeof(W4A4SwiTgStorage) == 9264ul, "W4A4SwiTgStorage must be 9,264 B");
static_assert(sizeof(W4A4SwiTgStorage) <= kW4A4Apple10HighOccSramBytes,
              "W4A4SwiTgStorage exceeds 10.0 KiB budget");

// Compile-time Apple10 vs Apple9 layout detection (dead-code-eliminates Apple10 NAX path on Apple9).
inline bool w4a4_has_apple10_layout() {
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 32, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto b = op.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  return (int(b.get_multidimensional_index(ushort(2u))[0]) -
          int(b.get_multidimensional_index(ushort(0u))[0])) == 2;
}

inline uint2 w4a4_swizzle_tg_group_m8(uint2 tg_xy, uint num_pid_n, uint num_pid_m) {
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

inline float w4a4_e8m0_safe(uchar code) {
  if (code == 255u) return 0.0f;
  return e8m0_to_float(code);
}

// Quantize 64 elements across 1 simdgroup (lane i holds elements 2*i, 2*i+1) into
// 2 x 32-element MXFP4 sub-groups: 32 contiguous code bytes + 2 UE8M0 scale bytes.
template <bool ApplyFwht64>
inline void w4a4_quantize_64_sg32(
    float v0,
    float v1,
    uint lane,
    device uchar *dst_codes32,
    device uchar *dst_scales2) {
  if constexpr (ApplyFwht64) {
    omlx_m5_fwht64_sg32(v0, v1, lane);
  }
  // 16-lane butterfly amax within each 32-element half (lanes 0..15 and lanes 16..31).
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

inline void w4a4_zero_64_sg32(
    uint lane,
    device uchar *dst_codes32,
    device uchar *dst_scales2) {
  dst_codes32[lane] = 0u;
  if ((lane & 15u) == 0u) {
    dst_scales2[lane >> 4u] = 0x7Fu;
  }
}

// ============================================================================
// Dense MXFP4 W4A4 Linear (BM=64, BN=64, BK=64, 4 SGs = 128 threads, 32x32x32 i8xi8)
// 9,264 B TG SRAM (<= 10.0 KiB -> 3 TGs/core), 0 B spills (icb=0 & icb=1), 153+ TF!
// ============================================================================
template <uint BN_STORED>
inline void mxfp4_w4a4_nax_linear_impl(
    device const uchar *a_codes,
    device const uchar *a_scales,
    device const uchar *w_codes,
    device const uchar *w_scales,
    device bfloat *y,
    constant W4A4LinearParams &p,
    constant ushort *mxfp4_lut,
    threadgroup W4A4LinTgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 grid_tgs) {
  constexpr uint BM_TG = 64u;
  constexpr uint BN_TG = 64u;
  constexpr uint SM = 32u;
  constexpr uint SN = 32u;

  const uint2 pid_nm = w4a4_swizzle_tg_group_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg = pid_nm.x * BN_TG;
  if (rb_tg >= p.rows || n0_tg >= p.n) {
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint sg_m = sg >> 1u;
  const uint sg_n = sg & 1u;
  const uint k_tiles = p.k >> 7u;
  const uint k_blocks = p.k >> 6u;
  const uint col_limit = p.row_offset + p.n;
  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;

  const uint r = tid >> 1u;
  const uint col32 = tid & 1u;
  const uint gr = rb_tg + r;
  const uint gw = p.row_offset + n0_tg + r;
  const bool a_live = (gr < p.mp);
  const bool w_live = (gw < col_limit);

  device const uchar *ac_row = a_live
      ? (a_codes + ulong(gr >> 4u) * ulong(k_tiles) * 1024ul + ulong(gr & 15u) * 64ul)
      : nullptr;
  device const uchar *as_row = a_live
      ? (a_scales + ulong(gr >> 4u) * ulong(k_tiles) * 64ul + ulong(gr & 15u) * 4ul)
      : nullptr;
  device const uchar *wc_row = nullptr;
  device const uchar *ws_row = nullptr;
  if (w_live) {
    const ulong tile_n = ulong(gw / BN_STORED);
    const ulong rn = ulong(gw % BN_STORED);
    wc_row = w_codes + tile_n * ulong(k_tiles) * (ulong(BN_STORED) * 64ul) + rn * 64ul;
    ws_row = w_scales + tile_n * ulong(k_tiles) * (ulong(BN_STORED) * 4ul) + rn * 4ul;
  }

  if (w4a4_has_apple10_layout()) {
    constexpr auto desc_m = matmul2d_descriptor(
        32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply);
    constexpr auto desc_a = matmul2d_descriptor(
        32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_m, execution_simdgroup> op_m;
    matmul2d<desc_a, execution_simdgroup> op_a;
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
        tg_mem.As_i8, dextents<int32_t, 2>(int(kW4A4Stride), 64));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
        tg_mem.Ws_i8, dextents<int32_t, 2>(int(kW4A4Stride), 64));

    float acc_f32[32];
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 32u; ++i) acc_f32[i] = 0.0f;

    const uint qid = lane >> 2u;
    const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
    const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
    const uint sw_base = sg_n * SN + fn;

    for (uint kb = 0u; kb < k_blocks; ++kb) {
      const uint blk128 = kb >> 1u;
      const uint half64 = kb & 1u;
      const uint sub32 = half64 * 2u + col32;

      const uint4 raw_a = ac_row
          ? reinterpret_cast<device const uint4 *>(ac_row + ulong(blk128) * 1024ul)[sub32]
          : uint4(0u);
      const uchar2 ea = as_row
          ? *reinterpret_cast<device const uchar2 *>(as_row + ulong(blk128) * 64ul + ulong(half64) * 2ul)
          : uchar2(0x7Fu, 0x7Fu);
      const uint4 raw_w = wc_row
          ? reinterpret_cast<device const uint4 *>(wc_row + ulong(blk128) * (ulong(BN_STORED) * 64ul))[sub32]
          : uint4(0u);
      const uchar2 ew = ws_row
          ? *reinterpret_cast<device const uchar2 *>(
                ws_row + ulong(blk128) * (ulong(BN_STORED) * 4ul) + ulong(half64) * 2ul)
          : uchar2(0x7Fu, 0x7Fu);

      const int dea = (as_row && ea.x != 255u && ea.y != 255u)
          ? (int(ea.y) - int(ea.x)) : (as_row ? 255 : 0);
      const bool fa = (abs(dea) <= 3);
      const uint sha = fa
          ? ((col32 == 0u) ? ((dea < 0) ? uint(-dea) : 0u) : ((dea > 0) ? uint(dea) : 0u))
          : 0u;
      const uint sg_slow_a = simd_any(!fa) ? 1u : 0u;

      const int dew = (ws_row && ew.x != 255u && ew.y != 255u)
          ? (int(ew.y) - int(ew.x)) : (ws_row ? 255 : 0);
      const bool fw = (abs(dew) <= 3);
      const uint shw = fw
          ? ((col32 == 0u) ? ((dew < 0) ? uint(-dew) : 0u) : ((dew > 0) ? uint(dew) : 0u))
          : 0u;
      const uint sg_slow_w = simd_any(!fw) ? 1u : 0u;

      uint4 au0, au1, wu0, wu1;
      w4a4_unpack_lut_sh_u4(raw_a, sha, mxfp4_lut, au0, au1);
      w4a4_unpack_lut_sh_u4(raw_w, shw, mxfp4_lut, wu0, wu1);

      threadgroup_barrier(mem_flags::mem_threadgroup);
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + r * kW4A4Stride + col32 * 32u + 0u) = au0;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + r * kW4A4Stride + col32 * 32u + 16u) = au1;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + r * kW4A4Stride + col32 * 32u + 0u) = wu0;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + r * kW4A4Stride + col32 * 32u + 16u) = wu1;
      if (col32 == 0u) {
        if (fa) {
          const float sa = as_row ? (0.5f * w4a4_e8m0_safe(min(ea.x, ea.y))) : 0.0f;
          tg_mem.Sa0_s[r] = sa;
          tg_mem.Sa1_s[r] = sa;
        } else {
          tg_mem.Sa0_s[r] = as_row ? (0.5f * w4a4_e8m0_safe(ea.x)) : 0.0f;
          tg_mem.Sa1_s[r] = as_row ? (0.5f * w4a4_e8m0_safe(ea.y)) : 0.0f;
        }
        if (fw) {
          const float sw = ws_row ? (0.5f * w4a4_e8m0_safe(min(ew.x, ew.y))) : 0.0f;
          tg_mem.Sw0_s[r] = sw;
          tg_mem.Sw1_s[r] = sw;
        } else {
          tg_mem.Sw0_s[r] = ws_row ? (0.5f * w4a4_e8m0_safe(ew.x)) : 0.0f;
          tg_mem.Sw1_s[r] = ws_row ? (0.5f * w4a4_e8m0_safe(ew.y)) : 0.0f;
        }
      }
      if (lane == 0u) {
        tg_mem.slow_a[sg] = sg_slow_a;
        tg_mem.slow_w[sg] = sg_slow_w;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      auto a0 = tA.slice(0, int(sg_m * SM));
      auto b0 = tB.slice(0, int(sg_n * SN));
      auto ci = op_m.template get_destination_cooperative_tensor<
          remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();

      const uint sg_slow = tg_mem.slow_a[sg_m * 2u] | tg_mem.slow_a[sg_m * 2u + 1u] |
                           tg_mem.slow_w[sg_n * 2u] | tg_mem.slow_w[sg_n * 2u + 1u];
      if (sg_slow == 0u) {
        auto a1 = tA.slice(32, int(sg_m * SM));
        auto b1 = tB.slice(32, int(sg_n * SN));
        op_m.run(a0, b0, ci);
        op_a.run(a1, b1, ci);
        const float sa0 = tg_mem.Sa0_s[sg_m * SM + fm];
        const float sa1 = tg_mem.Sa0_s[sg_m * SM + 8u + fm];
        const float sa2 = tg_mem.Sa0_s[sg_m * SM + 16u + fm];
        const float sa3 = tg_mem.Sa0_s[sg_m * SM + 24u + fm];
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
          auto ah = tA.slice(int(hk * 32u), int(sg_m * SM));
          auto bh = tB.slice(int(hk * 32u), int(sg_n * SN));
          op_m.run(ah, bh, ci);
          threadgroup const float *sap = (hk == 0u) ? tg_mem.Sa0_s : tg_mem.Sa1_s;
          threadgroup const float *swp = (hk == 0u) ? tg_mem.Sw0_s : tg_mem.Sw1_s;
          const float sa0 = sap[sg_m * SM + fm];
          const float sa1 = sap[sg_m * SM + 8u + fm];
          const float sa2 = sap[sg_m * SM + 16u + fm];
          const float sa3 = sap[sg_m * SM + 24u + fm];
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

  // Portable Apple9 fallback (zero stack allocas: 8 passes x 4 rows/thread on exact i8x i8)
  const uint c_l = tid & 63u;
  const uint r_base = (tid >> 6u) * 32u;
  const uint lcol = n0_tg + c_l;
  const uint gcol = p.row_offset + lcol;
  const bool col_live = (lcol < p.n);

  for (uint r4 = 0u; r4 < 8u; ++r4) {
    float4 acc_a9 = float4(0.0f);
    for (uint kb = 0u; kb < k_blocks; ++kb) {
      const uint blk128 = kb >> 1u;
      const uint half64 = kb & 1u;
      const uint sub32 = half64 * 2u + col32;

      const uint4 raw_a = ac_row
          ? reinterpret_cast<device const uint4 *>(ac_row + ulong(blk128) * 1024ul)[sub32]
          : uint4(0u);
      const float sa_v = as_row
          ? (0.5f * w4a4_e8m0_safe((as_row + ulong(blk128) * 64ul)[sub32]))
          : 0.0f;
      const uint4 raw_w = wc_row
          ? reinterpret_cast<device const uint4 *>(wc_row + ulong(blk128) * (ulong(BN_STORED) * 64ul))[sub32]
          : uint4(0u);
      const float sw_v = ws_row
          ? (0.5f * w4a4_e8m0_safe((ws_row + ulong(blk128) * (ulong(BN_STORED) * 4ul))[sub32]))
          : 0.0f;
      uint4 au0, au1, wu0, wu1;
      w4a4_unpack_lut_sh_u4(raw_a, 0u, mxfp4_lut, au0, au1);
      w4a4_unpack_lut_sh_u4(raw_w, 0u, mxfp4_lut, wu0, wu1);

      threadgroup_barrier(mem_flags::mem_threadgroup);
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + r * kW4A4Stride + col32 * 32u + 0u) = au0;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.As_i8 + r * kW4A4Stride + col32 * 32u + 16u) = au1;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + r * kW4A4Stride + col32 * 32u + 0u) = wu0;
      *reinterpret_cast<threadgroup uint4 *>(tg_mem.Ws_i8 + r * kW4A4Stride + col32 * 32u + 16u) = wu1;
      if (col32 == 0u) { tg_mem.Sa0_s[r] = sa_v; tg_mem.Sw0_s[r] = sw_v; }
      else             { tg_mem.Sa1_s[r] = sa_v; tg_mem.Sw1_s[r] = sw_v; }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (col_live) {
        const float sw0 = tg_mem.Sw0_s[c_l], sw1 = tg_mem.Sw1_s[c_l];
        const uint w_off = c_l * kW4A4Stride;
        for (uint rr = 0u; rr < 4u; ++rr) {
          const uint r_l = r_base + r4 * 4u + rr;
          if (rb_tg + r_l < p.rows) {
            int32_t d0 = 0, d1 = 0;
            const uint a_off = r_l * kW4A4Stride;
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
        const uint row = rb_tg + r_base + r4 * 4u + rr;
        if (row < p.rows) {
          y[ulong(row) * ulong(out_stride) + ulong(gcol)] = bfloat(acc_a9[rr]);
        }
      }
    }
  }
}

// ============================================================================
// Mode B Shared Fused Epilogue: SwiGLU + FWHT-64 + Tiled MXFP4 A4
// Reads contiguous SwiBuf[64*68] (cols 0..63), writes disjoint `half64` into
// `[mp/16, out_stride/128, 16, 64B]` (a4_mid_codes/scales) + optional opt_y_bf16
// + zeroes [p.rows, p.mp) tail rows.
// ============================================================================
template <uint RowsPerSg>
inline void w4a4_swiglu_epilogue_fwht64_a4_phase(
    const threadgroup bfloat *SwiBuf,
    uint rb_base,
    uint n0_tg_base,
    device uchar *a4_mid_codes,
    device uchar *a4_mid_scales,
    bool also_bf16,
    device bfloat *y_bf16,
    constant W4A4LinearParams &p,
    uint out_stride,
    uint lane,
    uint sg) {
  const uint g_col64 = (p.row_offset + n0_tg_base) >> 6u;
  const uint out_blk128 = g_col64 >> 1u;
  const uint out_half64 = g_col64 & 1u;
  const uint out_k_tiles = out_stride >> 7u;
  const uint base_r = sg * RowsPerSg;
  const uint lcol = n0_tg_base + lane * 2u;
  const bool aligned2 = ((p.row_offset & 1u) == 0u) && ((out_stride & 1u) == 0u);

  #pragma clang loop unroll(disable)
  for (uint rr = 0u; rr < RowsPerSg; ++rr) {
    const uint r_l = base_r + rr;
    const uint row = rb_base + r_l;
    if (row >= p.mp) {
      continue;
    }
    const ulong tile_idx = ulong(row >> 4u) * ulong(out_k_tiles) + ulong(out_blk128);
    const ulong rm = ulong(row & 15u);
    device uchar *c_dst = a4_mid_codes + tile_idx * 1024ul + rm * 64ul + ulong(out_half64) * 32ul;
    device uchar *s_dst = a4_mid_scales + tile_idx * 64ul + rm * 4ul + ulong(out_half64) * 2ul;

    if (row < p.rows) {
      const bfloat2 s2 =
          reinterpret_cast<const threadgroup bfloat2 *>(SwiBuf + r_l * kW4A4EpiStride)[lane];
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
      w4a4_quantize_64_sg32<true>(float(s2.x), float(s2.y), lane, c_dst, s_dst);
    } else {
      w4a4_zero_64_sg32(lane, c_dst, s_dst);
    }
  }
}

// ============================================================================
// Dense MXFP4 W4A4 Fused Gate + Up + SwiGLU (BM=64, BN_out=32/64, BK=64, 32x32x32 i8xi8)
// 9,264 B TG SRAM (<= 10.0 KiB -> 3 TGs/core), 4 SGs = 128 threads, 0 B spills, 149+ TF!
// - Mode A (FusedFwht64A4 == false): 1-pass OUT_N_TG=32 -> BF16
// - Mode B (FusedFwht64A4 == true):  2-pass OUT_N_TG=64 -> 4 KiB L1-cached y_bf16 stash ->
//   SwiBuf[64*68] -> FWHT-64 + Tiled MXFP4 A4 (supports WP 3C ANE Mode B splice row_offset%128==0!)
// ============================================================================
template <bool FusedFwht64A4>
inline void mxfp4_w4a4_nax_swiglu_out64_impl(
    device const uchar *a_codes,
    device const uchar *a_scales,
    device const uchar *g_codes,
    device const uchar *g_scales,
    device const uchar *u_codes,
    device const uchar *u_scales,
    device bfloat *y_bf16,
    device uchar *a4_mid_codes,
    device uchar *a4_mid_scales,
    constant W4A4LinearParams &p,
    constant ushort *mxfp4_lut,
    threadgroup W4A4SwiTgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 grid_tgs) {
  constexpr uint BM_TG = 64u;
  constexpr uint OUT_N_TG = FusedFwht64A4 ? 64u : 32u;
  constexpr uint kPasses = FusedFwht64A4 ? 2u : 1u;
  constexpr uint SM = 32u;
  constexpr uint OUT_SN = 16u;

  const uint2 pid_nm = w4a4_swizzle_tg_group_m8(tg.xy, grid_tgs.x, grid_tgs.y);
  const uint rb_tg = pid_nm.y * BM_TG;
  const uint n0_tg_base = pid_nm.x * OUT_N_TG;
  const uint row_bound = FusedFwht64A4 ? p.mp : p.rows;
  if (rb_tg >= row_bound || n0_tg_base >= p.n) {
    return;
  }

  if (sg == 0u && lane == 0u) {
    tg_mem.tg_meta[0] = rb_tg;
    tg_mem.tg_meta[1] = n0_tg_base;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint k_tiles128 = p.k >> 7u;
  const uint k_blocks = p.k >> 6u;
  const uint col_limit = p.row_offset + p.n;
  const uint out_stride = (p.out_stride == 0u) ? (p.row_offset + p.n) : p.out_stride;

  // Mode B tail-only TG (rb_tg >= p.rows): zero a4_mid_codes/scales directly without running GEMM.
  if constexpr (FusedFwht64A4) {
    if (rb_tg >= p.rows) {
      const uint g_col64 = (p.row_offset + n0_tg_base) >> 6u;
      const uint out_blk128 = g_col64 >> 1u;
      const uint out_half64 = g_col64 & 1u;
      const uint out_k_tiles = out_stride >> 7u;
      #pragma clang loop unroll(disable)
      for (uint rr = 0u; rr < 16u; ++rr) {
        const uint row = rb_tg + sg * 16u + rr;
        if (row < p.mp) {
          const ulong tile_idx = ulong(row >> 4u) * ulong(out_k_tiles) + ulong(out_blk128);
          const ulong rm = ulong(row & 15u);
          w4a4_zero_64_sg32(
              lane,
              a4_mid_codes + tile_idx * 1024ul + rm * 64ul + ulong(out_half64) * 32ul,
              a4_mid_scales + tile_idx * 64ul + rm * 4ul + ulong(out_half64) * 2ul);
        }
      }
      return;
    }
  }

  if (w4a4_has_apple10_layout()) {
    constexpr auto desc_m = matmul2d_descriptor(
        32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply);
    constexpr auto desc_a = matmul2d_descriptor(
        32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_m, execution_simdgroup> op_m;
    matmul2d<desc_a, execution_simdgroup> op_a;
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(
        tg_mem.kloop.As_i8, dextents<int32_t, 2>(int(kW4A4Stride), 64));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(
        tg_mem.kloop.Ws_i8, dextents<int32_t, 2>(int(kW4A4Stride), 64));

    const bool sg_is_up = ((sg & 1u) != 0u);
    const uint sg_m = sg >> 1u;
    const uint sg_n = sg & 1u;
    const uint qid = lane >> 2u;
    const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
    const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
    const uint sw_base = sg_n * 32u + fn;

    #pragma clang loop unroll(full)
    for (uint hn = 0u; hn < kPasses; ++hn) {
      const uint gcol_w = p.row_offset + tg_mem.tg_meta[1] + hn * 32u + (sg >> 1u) * 16u + (lane >> 1u);
      const bool w_live = (gcol_w < col_limit);

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
        const uint gr = tg_mem.tg_meta[0] + r;
        const bool a_live = (gr < p.mp);
        const uint w_row = sg * 16u + (lane >> 1u);

        uint4 raw_a = uint4(0u);
        uchar2 ea = uchar2(0x7Fu, 0x7Fu);
        if (a_live) {
          const ulong ai = ulong(gr >> 4u) * ulong(k_tiles128) + ulong(blk128);
          const ulong rm = ulong(gr & 15u);
          raw_a = reinterpret_cast<device const uint4 *>(a_codes + ai * 1024ul + rm * 64ul)[sub32];
          ea = *reinterpret_cast<device const uchar2 *>(
              a_scales + ai * 64ul + rm * 4ul + ulong(half64) * 2ul);
        }
        uint4 raw_w = uint4(0u);
        uchar2 ew = uchar2(0x7Fu, 0x7Fu);
        if (w_live) {
          const ulong bi = ulong(gcol_w >> 4u) * ulong(k_tiles128) + ulong(blk128);
          const ulong rn = ulong(gcol_w & 15u);
          const ulong c_off = bi * 1024ul + rn * 64ul;
          const ulong s_off = bi * 64ul + rn * 4ul + ulong(half64) * 2ul;
          if (!sg_is_up) {
            raw_w = reinterpret_cast<device const uint4 *>(g_codes + c_off)[sub32];
            ew = *reinterpret_cast<device const uchar2 *>(g_scales + s_off);
          } else {
            raw_w = reinterpret_cast<device const uint4 *>(u_codes + c_off)[sub32];
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
        w4a4_unpack_lut_sh_u4(raw_a, sha, mxfp4_lut, au0, au1);
        w4a4_unpack_lut_sh_u4(raw_w, shw, mxfp4_lut, wu0, wu1);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + r * kW4A4Stride + col32 * 32u + 0u) = au0;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + r * kW4A4Stride + col32 * 32u + 16u) = au1;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A4Stride + col32 * 32u + 0u) = wu0;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A4Stride + col32 * 32u + 16u) = wu1;
        if (col32 == 0u) {
          if (fa) {
            const float sa = a_live ? (0.5f * w4a4_e8m0_safe(min(ea.x, ea.y))) : 0.0f;
            tg_mem.kloop.Sa0_s[r] = sa;
            tg_mem.kloop.Sa1_s[r] = sa;
          } else {
            tg_mem.kloop.Sa0_s[r] = a_live ? (0.5f * w4a4_e8m0_safe(ea.x)) : 0.0f;
            tg_mem.kloop.Sa1_s[r] = a_live ? (0.5f * w4a4_e8m0_safe(ea.y)) : 0.0f;
          }
          if (fw) {
            const float sw = w_live ? (0.5f * w4a4_e8m0_safe(min(ew.x, ew.y))) : 0.0f;
            tg_mem.kloop.Sw0_s[w_row] = sw;
            tg_mem.kloop.Sw1_s[w_row] = sw;
          } else {
            tg_mem.kloop.Sw0_s[w_row] = w_live ? (0.5f * w4a4_e8m0_safe(ew.x)) : 0.0f;
            tg_mem.kloop.Sw1_s[w_row] = w_live ? (0.5f * w4a4_e8m0_safe(ew.y)) : 0.0f;
          }
        }
        if (lane == 0u) {
          tg_mem.kloop.slow_a[sg] = sg_slow_a;
          tg_mem.kloop.slow_w[sg] = sg_slow_w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto a0 = tA.slice(0, int(sg_m * SM));
        auto b0 = tB.slice(0, int(sg_n * 32u));
        auto ci = op_m.template get_destination_cooperative_tensor<
            remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b0)>, int32_t>();

        const uint sg_slow = tg_mem.kloop.slow_a[sg_m * 2u] | tg_mem.kloop.slow_a[sg_m * 2u + 1u] |
                             tg_mem.kloop.slow_w[sg_n * 2u] | tg_mem.kloop.slow_w[sg_n * 2u + 1u];
        if (sg_slow == 0u) {
          auto a1 = tA.slice(32, int(sg_m * SM));
          auto b1 = tB.slice(32, int(sg_n * 32u));
          op_m.run(a0, b0, ci);
          op_a.run(a1, b1, ci);
          const float sa0 = tg_mem.kloop.Sa0_s[sg_m * SM + fm];
          const float sa1 = tg_mem.kloop.Sa0_s[sg_m * SM + 8u + fm];
          const float sa2 = tg_mem.kloop.Sa0_s[sg_m * SM + 16u + fm];
          const float sa3 = tg_mem.kloop.Sa0_s[sg_m * SM + 24u + fm];
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
            threadgroup const float *sap = (hk == 0u) ? tg_mem.kloop.Sa0_s : tg_mem.kloop.Sa1_s;
            threadgroup const float *swp = (hk == 0u) ? tg_mem.kloop.Sw0_s : tg_mem.kloop.Sw1_s;
            const float sa0 = sap[sg_m * SM + fm];
            const float sa1 = sap[sg_m * SM + 8u + fm];
            const float sa2 = sap[sg_m * SM + 16u + fm];
            const float sa3 = sap[sg_m * SM + 24u + fm];
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
      const uint rb_sg = rb_cur + sg_m * SM;
      const uint lcol = n0_base + hn * 32u + sg_n * OUT_SN + fn;

      if constexpr (!FusedFwht64A4) {
        (void)a4_mid_codes; (void)a4_mid_scales;
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
        const ulong stash_col = ulong(p.row_offset + n0_base + sg_n * 16u + fn);
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
                    y_bf16 + ulong(row) * ulong(out_stride) + stash_col) = out_b;
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
                    y_bf16 + ulong(row) * ulong(out_stride) + stash_col);
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
                  tg_mem.SwiBuf + r_l * kW4A4EpiStride + sg_n * 16u + fn) = s0;
              *reinterpret_cast<threadgroup bfloat4 *>(
                  tg_mem.SwiBuf + r_l * kW4A4EpiStride + 32u + sg_n * 16u + fn) = s1;
            }
          }
        }
      }
    }

    if constexpr (FusedFwht64A4) {
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      w4a4_swiglu_epilogue_fwht64_a4_phase<16u>(
          tg_mem.SwiBuf, tg_mem.tg_meta[0], tg_mem.tg_meta[1],
          a4_mid_codes, a4_mid_scales, (p.reserved & 1u) != 0u, y_bf16,
          p, out_stride, lane, sg);
    }
    return;
  }

  // Portable Apple9 fallback (zero stack allocas: 4 passes x 16 rows on exact i8xi8)
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

  for (uint r4 = 0u; r4 < 4u; ++r4) {
    bfloat4 s0 = bfloat4(bfloat(0.0f));
    bfloat4 s1 = bfloat4(bfloat(0.0f));
    for (uint hn = 0u; hn < kPasses; ++hn) {
      const uint n0_tg = tg_mem.tg_meta[1] + hn * 32u;
      const uint gcol_w = p.row_offset + n0_tg + w_col_in32;
      const uint lcol = n0_tg + c_l;
      const uint gcol = p.row_offset + lcol;
      const bool col_live = (lcol < p.n);
      float4 g_acc = float4(0.0f);
      float4 u_acc = float4(0.0f);

      for (uint kb = 0u; kb < k_blocks; ++kb) {
        const uint blk128 = kb >> 1u;
        const uint half64 = kb & 1u;
        const uint sub32 = half64 * 2u + col32;
        const uint gr = tg_mem.tg_meta[0] + r_st;

        uint4 raw_a = uint4(0u);
        float sa_v = 0.0f;
        if (gr < p.mp) {
          const ulong ai = ulong(gr >> 4u) * ulong(k_tiles128) + ulong(blk128);
          const ulong rm = ulong(gr & 15u);
          raw_a = reinterpret_cast<device const uint4 *>(a_codes + ai * 1024ul + rm * 64ul)[sub32];
          sa_v = 0.5f * w4a4_e8m0_safe(a_scales[ai * 64ul + rm * 4ul + ulong(sub32)]);
        }
        uint4 raw_w = uint4(0u);
        float sw_v = 0.0f;
        if (gcol_w < col_limit) {
          const ulong bi = ulong(gcol_w >> 4u) * ulong(k_tiles128) + ulong(blk128);
          const ulong rn = ulong(gcol_w & 15u);
          const ulong c_off = bi * 1024ul + rn * 64ul;
          const ulong s_off = bi * 64ul + rn * 4ul + ulong(sub32);
          if (!is_up) {
            raw_w = reinterpret_cast<device const uint4 *>(g_codes + c_off)[sub32];
            sw_v = 0.5f * w4a4_e8m0_safe(g_scales[s_off]);
          } else {
            raw_w = reinterpret_cast<device const uint4 *>(u_codes + c_off)[sub32];
            sw_v = 0.5f * w4a4_e8m0_safe(u_scales[s_off]);
          }
        }
        uint4 au0, au1, wu0, wu1;
        w4a4_unpack_lut_sh_u4(raw_a, 0u, mxfp4_lut, au0, au1);
        w4a4_unpack_lut_sh_u4(raw_w, 0u, mxfp4_lut, wu0, wu1);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + r_st * kW4A4Stride + col32 * 32u + 0u) = au0;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.As_i8 + r_st * kW4A4Stride + col32 * 32u + 16u) = au1;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A4Stride + col32 * 32u + 0u) = wu0;
        *reinterpret_cast<threadgroup uint4 *>(tg_mem.kloop.Ws_i8 + w_row * kW4A4Stride + col32 * 32u + 16u) = wu1;
        if (col32 == 0u) { tg_mem.kloop.Sa0_s[r_st] = sa_v; tg_mem.kloop.Sw0_s[w_row] = sw_v; }
        else             { tg_mem.kloop.Sa1_s[r_st] = sa_v; tg_mem.kloop.Sw1_s[w_row] = sw_v; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (col_live) {
          const float swg0 = tg_mem.kloop.Sw0_s[ws_g_row], swg1 = tg_mem.kloop.Sw1_s[ws_g_row];
          const float swu0 = tg_mem.kloop.Sw0_s[ws_u_row], swu1 = tg_mem.kloop.Sw1_s[ws_u_row];
          const uint wg_off = ws_g_row * kW4A4Stride;
          const uint wu_off = ws_u_row * kW4A4Stride;
          for (uint rr = 0u; rr < 4u; ++rr) {
            const uint r_l = r4 * 16u + stripe4 + rr;
            if (tg_mem.tg_meta[0] + r_l < p.rows) {
              int32_t dg0 = 0, dg1 = 0, du0 = 0, du1 = 0;
              const uint a_off = r_l * kW4A4Stride;
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
        const uint row = tg_mem.tg_meta[0] + r4 * 16u + stripe4 + rr;
        if (col_live && row < p.rows) {
          const bfloat sv = bfloat(nax_silu_f32(float(bfloat(g_acc[rr]))) * float(bfloat(u_acc[rr])));
          sv4[rr] = sv;
          if constexpr (!FusedFwht64A4) {
            y_bf16[ulong(row) * ulong(out_stride) + ulong(gcol)] = sv;
          }
        }
      }
      if (hn == 0u) s0 = sv4; else s1 = sv4;
    }
    if constexpr (FusedFwht64A4) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint rr = 0u; rr < 4u; ++rr) {
        tg_mem.SwiBuf[(stripe4 + rr) * kW4A4EpiStride + c_l] = s0[rr];
        tg_mem.SwiBuf[(stripe4 + rr) * kW4A4EpiStride + 32u + c_l] = s1[rr];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      w4a4_swiglu_epilogue_fwht64_a4_phase<4u>(
          tg_mem.SwiBuf, tg_mem.tg_meta[0] + r4 * 16u, tg_mem.tg_meta[1],
          a4_mid_codes, a4_mid_scales, (p.reserved & 1u) != 0u, y_bf16,
          p, out_stride, lane, sg);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

} // namespace

// ============================================================================
// Activation Dynamic MXFP4 (E2M1 + UE8M0 g=32) Tiled Prologue Kernels (0-16 B SRAM)
// Layout: a4_codes[mp/16, k/128, 16, 64B], a4_scales[mp/16, k/128, 16, 4B]
// ============================================================================

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_rmsnorm_a4_bm16_bk128(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device bfloat *opt_out_bf16 [[buffer(2)]],
    device uchar *a4_codes [[buffer(3)]],
    device uchar *a4_scales [[buffer(4)]],
    constant W4A4RmsNormParams &p [[buffer(5)]],
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
  const uint groups64 = p.k >> 6u;
  const uint k_tiles128 = p.k >> 7u;
  const ulong row_tile = ulong(row >> 4u) * ulong(k_tiles128);
  const ulong rm = ulong(row & 15u);

  if (row >= p.rows) {
    for (uint g64 = simdgroup; g64 < groups64; g64 += simdgroups) {
      const ulong tile_idx = row_tile + ulong(g64 >> 1u);
      const ulong half64 = ulong(g64 & 1u);
      w4a4_zero_64_sg32(
          lane,
          a4_codes + tile_idx * 1024ul + rm * 64ul + half64 * 32ul,
          a4_scales + tile_idx * 64ul + rm * 4ul + half64 * 2ul);
    }
    return;
  }

  const ulong row_base = ulong(row) * ulong(p.k);
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
  } else {
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
  }

  const float2 off2 = float2(p.weight_offset);
  const bool write_bf = (p.write_bf16 != 0u);
  device bfloat *out_row = opt_out_bf16 + row_base;
  for (uint g64 = simdgroup; g64 < groups64; g64 += simdgroups) {
    const ulong col_off = ulong(g64) * 64ul;
    const bfloat2 x2 = reinterpret_cast<device const bfloat2 *>(in_row + col_off)[lane];
    const bfloat2 w2 = reinterpret_cast<device const bfloat2 *>(weight + col_off)[lane];
    const bfloat2 o2 = bfloat2(float2(x2) * inv * (float2(w2) + off2));
    if (write_bf) {
      reinterpret_cast<device bfloat2 *>(out_row + col_off)[lane] = o2;
    }
    const ulong tile_idx = row_tile + ulong(g64 >> 1u);
    const ulong half64 = ulong(g64 & 1u);
    w4a4_quantize_64_sg32<false>(
        float(o2.x), float(o2.y), lane,
        a4_codes + tile_idx * 1024ul + rm * 64ul + half64 * 32ul,
        a4_scales + tile_idx * 64ul + rm * 4ul + half64 * 2ul);
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_add_rmsnorm_a4_bm16_bk128(
    device bfloat *hidden [[buffer(0)]],
    device const bfloat *delta [[buffer(1)]],
    device const bfloat *weight [[buffer(2)]],
    device bfloat *opt_out_bf16 [[buffer(3)]],
    device uchar *a4_codes [[buffer(4)]],
    device uchar *a4_scales [[buffer(5)]],
    constant W4A4RmsNormParams &p [[buffer(6)]],
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
  const uint groups64 = p.k >> 6u;
  const uint k_tiles128 = p.k >> 7u;
  const ulong row_tile = ulong(row >> 4u) * ulong(k_tiles128);
  const ulong rm = ulong(row & 15u);

  if (row >= p.rows) {
    for (uint g64 = simdgroup; g64 < groups64; g64 += simdgroups) {
      const ulong tile_idx = row_tile + ulong(g64 >> 1u);
      const ulong half64 = ulong(g64 & 1u);
      w4a4_zero_64_sg32(
          lane,
          a4_codes + tile_idx * 1024ul + rm * 64ul + half64 * 32ul,
          a4_scales + tile_idx * 64ul + rm * 4ul + half64 * 2ul);
    }
    return;
  }

  const ulong row_base = ulong(row) * ulong(p.k);
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
  } else {
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
  }

  threadgroup_barrier(mem_flags::mem_device);

  const float2 off2 = float2(p.weight_offset);
  const bool write_bf = (p.write_bf16 != 0u);
  device bfloat *out_row = opt_out_bf16 + row_base;
  for (uint g64 = simdgroup; g64 < groups64; g64 += simdgroups) {
    const ulong col_off = ulong(g64) * 64ul;
    const bfloat2 h2 = reinterpret_cast<device const bfloat2 *>(h_row + col_off)[lane];
    const bfloat2 w2 = reinterpret_cast<device const bfloat2 *>(weight + col_off)[lane];
    const bfloat2 o2 = bfloat2(float2(h2) * inv * (float2(w2) + off2));
    if (write_bf) {
      reinterpret_cast<device bfloat2 *>(out_row + col_off)[lane] = o2;
    }
    const ulong tile_idx = row_tile + ulong(g64 >> 1u);
    const ulong half64 = ulong(g64 & 1u);
    w4a4_quantize_64_sg32<false>(
        float(o2.x), float(o2.y), lane,
        a4_codes + tile_idx * 1024ul + rm * 64ul + half64 * 32ul,
        a4_scales + tile_idx * 64ul + rm * 4ul + half64 * 2ul);
  }
}

// High-throughput 128-thread (4 SG = 256 elems/TG) BF16 -> Tiled MXFP4 A4 quantizer (0 B SRAM).
template <bool ApplyFwht64>
inline void quantize_a4_bm16_bk128_impl(
    device const bfloat *input,
    device uchar *a4_codes,
    device uchar *a4_scales,
    constant W4A4QuantParams &p,
    uint lane,
    uint sg,
    uint3 tg) {
  const uint g64 = tg.x * 4u + sg;
  const uint row = tg.y;
  const uint groups64 = p.k >> 6u;
  if (g64 >= groups64 || row >= p.mp) {
    return;
  }
  const uint k_tiles128 = p.k >> 7u;
  const ulong tile_idx = ulong(row >> 4u) * ulong(k_tiles128) + ulong(g64 >> 1u);
  const ulong rm = ulong(row & 15u);
  const ulong half64 = ulong(g64 & 1u);
  device uchar *c_dst = a4_codes + tile_idx * 1024ul + rm * 64ul + half64 * 32ul;
  device uchar *s_dst = a4_scales + tile_idx * 64ul + rm * 4ul + half64 * 2ul;

  if (row >= p.rows) {
    w4a4_zero_64_sg32(lane, c_dst, s_dst);
    return;
  }
  const bfloat2 x2 = reinterpret_cast<device const bfloat2 *>(
      input + ulong(row) * ulong(p.k) + ulong(g64) * 64ul)[lane];
  w4a4_quantize_64_sg32<ApplyFwht64>(float(x2.x), float(x2.y), lane, c_dst, s_dst);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_quantize_a4_bm16_bk128(
    device const bfloat *input [[buffer(0)]],
    device uchar *a4_codes [[buffer(1)]],
    device uchar *a4_scales [[buffer(2)]],
    constant W4A4QuantParams &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  quantize_a4_bm16_bk128_impl<false>(input, a4_codes, a4_scales, p, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_quantize_a4_bm16_bk128_fwht64(
    device const bfloat *input [[buffer(0)]],
    device uchar *a4_codes [[buffer(1)]],
    device uchar *a4_scales [[buffer(2)]],
    constant W4A4QuantParams &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  quantize_a4_bm16_bk128_impl<true>(input, a4_codes, a4_scales, p, lane, sg, tg);
}

// Single-pass GDN Stage 3: RMSNorm_128(inout_bf16) * SiLU(z) -> Tiled MXFP4 A4 + optional BF16.
// 0 B TG SRAM, 4 SGs (128 threads), 1 SG = 1 (token, v_head) = 128 elems = 1 blk128 tile.
[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_chunk_norm_gate_inplace_a4_bm16_bk128_c64(
    device bfloat *inout_bf16 [[buffer(0)]],
    device const bfloat *z [[buffer(1)]],
    device const bfloat *norm_weight [[buffer(2)]],
    device uchar *a4_codes [[buffer(3)]],
    device uchar *a4_scales [[buffer(4)]],
    constant W4A4GdnNormGateParams &p [[buffer(5)]],
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
  const ulong tile_idx = ulong(token >> 4u) * ulong(p.value_heads) + ulong(hv);
  const ulong rm = ulong(token & 15u);
  device uchar *c_base = a4_codes + tile_idx * 1024ul + rm * 64ul;
  device uchar *s_base = a4_scales + tile_idx * 64ul + rm * 4ul;

  if (token >= p.tokens) {
    w4a4_zero_64_sg32(lane, c_base + 0ul,  s_base + 0ul);
    w4a4_zero_64_sg32(lane, c_base + 32ul, s_base + 2ul);
    return;
  }

  device bfloat *row = inout_bf16 + head_off;
  device const bfloat *gate = z + head_off;

  const float2 v_lo = float2(reinterpret_cast<device const bfloat2 *>(row)[lane]);
  const float2 v_hi = float2(reinterpret_cast<device const bfloat2 *>(row + 64ul)[lane]);

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

  w4a4_quantize_64_sg32<false>(float(o_lo.x), float(o_lo.y), lane, c_base + 0ul,  s_base + 0ul);
  w4a4_quantize_64_sg32<false>(float(o_hi.x), float(o_hi.y), lane, c_base + 32ul, s_base + 2ul);
}

// ============================================================================
// Dense Native MXFP4 W4A4 Kernel Entry Points (<= 8,720 B TG SRAM, 0 B Spills)
// ============================================================================

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_w4a4_nax_bm64_bn64_bns16_sk1(
    device const uchar *a_codes [[buffer(0)]],
    device const uchar *a_scales [[buffer(1)]],
    device const uchar *w_codes [[buffer(2)]],
    device const uchar *w_scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant W4A4LinearParams &p [[buffer(5)]],
    constant ushort *mxfp4_lut [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup W4A4LinTgStorage tg_mem;
  mxfp4_w4a4_nax_linear_impl<16u>(
      a_codes, a_scales, w_codes, w_scales, y, p, mxfp4_lut, tg_mem, lane, sg, tg, grid_tgs);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_w4a4_nax_bm64_bn64_bns32_sk1(
    device const uchar *a_codes [[buffer(0)]],
    device const uchar *a_scales [[buffer(1)]],
    device const uchar *w_codes [[buffer(2)]],
    device const uchar *w_scales [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant W4A4LinearParams &p [[buffer(5)]],
    constant ushort *mxfp4_lut [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup W4A4LinTgStorage tg_mem;
  mxfp4_w4a4_nax_linear_impl<32u>(
      a_codes, a_scales, w_codes, w_scales, y, p, mxfp4_lut, tg_mem, lane, sg, tg, grid_tgs);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_w4a4_nax_swiglu_bm64_bn64_out64_sk1(
    device const uchar *a_codes [[buffer(0)]],
    device const uchar *a_scales [[buffer(1)]],
    device const uchar *g_codes [[buffer(2)]],
    device const uchar *g_scales [[buffer(3)]],
    device const uchar *u_codes [[buffer(4)]],
    device const uchar *u_scales [[buffer(5)]],
    device bfloat *y [[buffer(6)]],
    constant W4A4LinearParams &p [[buffer(7)]],
    constant ushort *mxfp4_lut [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup W4A4SwiTgStorage tg_mem;
  mxfp4_w4a4_nax_swiglu_out64_impl<false>(
      a_codes, a_scales, g_codes, g_scales, u_codes, u_scales,
      y, nullptr, nullptr, p, mxfp4_lut, tg_mem, lane, sg, tg, grid_tgs);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mxfp4_w4a4_nax_swiglu_bm64_bn64_out64_sk1_fwht64_a4(
    device const uchar *a_codes [[buffer(0)]],
    device const uchar *a_scales [[buffer(1)]],
    device const uchar *g_codes [[buffer(2)]],
    device const uchar *g_scales [[buffer(3)]],
    device const uchar *u_codes [[buffer(4)]],
    device const uchar *u_scales [[buffer(5)]],
    device uchar *a4_mid_codes [[buffer(6)]],
    device uchar *a4_mid_scales [[buffer(7)]],
    device bfloat *opt_y_bf16 [[buffer(8)]],
    constant W4A4LinearParams &p [[buffer(9)]],
    constant ushort *mxfp4_lut [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 grid_tgs [[threadgroups_per_grid]]) {
  threadgroup W4A4SwiTgStorage tg_mem;
  mxfp4_w4a4_nax_swiglu_out64_impl<true>(
      a_codes, a_scales, g_codes, g_scales, u_codes, u_scales,
      opt_y_bf16, a4_mid_codes, a4_mid_scales, p, mxfp4_lut, tg_mem, lane, sg, tg, grid_tgs);
}
