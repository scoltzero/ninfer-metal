// Tier-1 Pillar 2 (T1-B) + WP3.3 Adaptive Full-Occupancy Zero-Spill GDN Prefill
// Stage 1 (`omlx_m5_gdn_chunk_prepare_{c64,conv_c64}`): parallel over (key_head, chunk),
//   fuses bfloat4 register-sliding-window causal Conv1D + SiLU on Q & K,
//   normalizes Q & K in FP32 ONCE per (chunk, key_head), computes raw QK^T and KK^T
//   via zero-spill register-fragment 16x32x16 BF16 NAX MMA (4 SGs x 16 rows x 64 cols),
//   solves exact 2x2 block IEEE FP32 (I_64 + L)^-1 * diag(beta) via in-place SRAM
//   forward substitution + exact FP32 simdgroup_float8x8 off-diagonal multiply.
//   Static TG SRAM: 16,384 B (16.00 KiB -> 2 TGs/core, 0 B spills!).
// Stage 2 WP3.3 Adaptive Scheduler (`dv16_flex_c64` + `dv32_flex_c64`):
//   - `dv16_flex_c64` (<= 16 Dv cols/task, supports K_dv=8 [16] and K_dv=10 [12..13]):
//     Static TG SRAM = 9,216 B (9.00 KiB <= 10.00 KiB -> 3 TGs/core!), 0 B spills,
//     persistent strided task loop over `total_tasks = Hv * K_dv`.
//   - `dv32_flex_c64` (<= 32 Dv cols/task, supports K_dv=4 [32] and K_dv=5 [25..26]):
//     Static TG SRAM = 16,384 B (16.00 KiB -> 2 TGs/core!), 0 B spills,
//     persistent strided task loop over `total_tasks = Hv * K_dv`.
//   - Legacy `dv32`, `dv64`, and `mega_{bf16,f16}` ABIs preserved with 0 B spills!

#include "common_math.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define GDN_NAX_ALWAYS_INLINE __attribute__((always_inline))

struct GdnChunkLayerParams {
  uint key_heads;
  uint value_heads;
  uint key_dim;
  uint value_dim;
  uint conv_width;
  uint conv_kernel;
  uint tokens;
  float epsilon;
};

struct GdnChunkMegaParams {
  uint key_heads;
  uint value_heads;
  uint key_dim;
  uint value_dim;
  uint conv_width;
  uint conv_kernel;
  uint tokens;
  float epsilon;
  uint mp;
};

// WP3.3 Flex Stage-2 params (48 B; first 9 words layout-compatible with GdnChunkMegaParams).
struct GdnChunkFlexParams {
  uint key_heads;
  uint value_heads;
  uint key_dim;
  uint value_dim;
  uint conv_width;
  uint conv_kernel;
  uint tokens;
  float epsilon;
  uint mp;
  uint slices_per_head; // K_dv (4, 5, 8, 10)
  uint total_tasks;     // value_heads * K_dv
  uint _pad0;
};

namespace {
constant constexpr uint kChunkSize = 64u;
constant constexpr uint kHalfChunk = 32u;
constant constexpr uint kKeyDim = 128u;
constant constexpr uint kValDim = 128u;
constant constexpr uint kValSgDim = 16u;

constant constexpr uint kDvSlice16 = 16u;
constant constexpr uint kDvSlice32 = 32u;
constant constexpr uint kRhsPad = 72u;   // 64 + 8 for dv16 Apple9 fallback
constant constexpr uint kRhsPad32 = 64u; // exact 4.0 KiB per [32x64] tile in dv32 Apple9 fallback

constant constexpr ulong kQkPairBfloats =
    2ul * ulong(kChunkSize) * ulong(kKeyDim);

constant constexpr ulong kCtrlStrideBytes = 17152ul;
constant constexpr ulong kCtrlMqkByteOff = 8192ul;
constant constexpr ulong kCtrlPrefByteOff = 16384ul;
constant constexpr ulong kCtrlSuffByteOff = 16640ul;
constant constexpr ulong kCtrlMetaByteOff = 16896ul;

struct GdnPrepareTgStorage {
  bfloat buf[kChunkSize * kKeyDim]; // 16,384 B (16.00 KiB)
};
static_assert(sizeof(GdnPrepareTgStorage) <= kApple10MaxPrefillTgSramBytes,
              "GdnPrepareTgStorage exceeds 16,384 B!");

struct GdnDv16TgStorage {
  union {
    float  f32[2560u];  // 10,240 B = 10.00 KiB (<= 10,240 B -> 3 TGs/core!)
    bfloat bf16[5120u]; // 10,240 B
  };
};
static_assert(sizeof(GdnDv16TgStorage) == 10240u,
              "GdnDv16TgStorage must be exact 10.00 KiB!");
static_assert(sizeof(GdnDv16TgStorage) <= 10240u,
              "WP3.3 GdnDv16TgStorage must be <= 10.00 KiB for 3 TGs/core!");

struct GdnDv32TgStorage {
  union {
    float  f32[4096u];  // 16,384 B
    bfloat bf16[8192u]; // 16,384 B
  };
};
static_assert(sizeof(GdnDv32TgStorage) <= kApple10MaxPrefillTgSramBytes,
              "GdnDv32TgStorage exceeds 16,384 B!");

struct GdnMegaTgStorage {
  bfloat work[8u * kChunkSize * kValSgDim]; // 16,384 B
};
static_assert(sizeof(GdnMegaTgStorage) <= kApple10MaxPrefillTgSramBytes,
              "GdnMegaTgStorage exceeds 16,384 B!");

GDN_NAX_ALWAYS_INLINE
inline bool gdn_has_apple10_nax_layout() {
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  const auto c0 = ct_a.get_multidimensional_index(ushort(0u));
  const auto c2 = ct_a.get_multidimensional_index(ushort(2u));
  return (int(c2[0]) - int(c0[0]) == 2) && (int(c2[1]) == int(c0[1]));
}

GDN_NAX_ALWAYS_INLINE
inline vec<bfloat, 8> gdn_load_frag_dev_bf16(
    device const bfloat *base,
    uint stride,
    uint fm,
    uint fn) {
  const bfloat4 v0 =
      *reinterpret_cast<device const bfloat4 *>(base + ulong(fm) * stride + fn);
  const bfloat4 v1 =
      *reinterpret_cast<device const bfloat4 *>(base + ulong(fm + 8u) * stride + fn);
  return vec<bfloat, 8>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
}

template <uint Stride>
GDN_NAX_ALWAYS_INLINE
inline vec<bfloat, 8> gdn_load_frag_tg_bf16(
    const threadgroup bfloat *base,
    uint fm,
    uint fn) {
  const bfloat4 v0 =
      *reinterpret_cast<const threadgroup bfloat4 *>(base + fm * Stride + fn);
  const bfloat4 v1 =
      *reinterpret_cast<const threadgroup bfloat4 *>(base + (fm + 8u) * Stride + fn);
  return vec<bfloat, 8>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
}

// Conflict-free XOR-swizzled [rows x 64] bfloat4 helpers (stride = 64 bfloats = 16 bfloat4s).
GDN_NAX_ALWAYS_INLINE
inline uint gdn_swiz64_idx(uint row, uint col) {
  const uint col4 = col >> 2u;
  const uint phys4 = col4 ^ ((row & 7u) << 1u);
  return row * 64u + (phys4 << 2u) + (col & 3u);
}

GDN_NAX_ALWAYS_INLINE
inline vec<bfloat, 8> gdn_load_frag_tg_swiz64(
    const threadgroup bfloat *base,
    uint row0_base,
    uint col_off,
    uint fm,
    uint fn) {
  const uint col4 = (col_off + fn) >> 2u;
  const uint phys4 = col4 ^ ((fm & 7u) << 1u);
  const uint r0 = row0_base + fm;
  const uint r1 = r0 + 8u;
  const bfloat4 v0 =
      reinterpret_cast<const threadgroup bfloat4 *>(base + r0 * 64u)[phys4];
  const bfloat4 v1 =
      reinterpret_cast<const threadgroup bfloat4 *>(base + r1 * 64u)[phys4];
  return vec<bfloat, 8>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
}

GDN_NAX_ALWAYS_INLINE
inline bfloat gdn_eval_conv_silu_scalar(
    device const bfloat *qkv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    uint token,
    uint channel,
    uint conv_width,
    uint conv_kernel) {
  const uint history_tokens = conv_kernel - 1u;
  float sum = 0.0f;
  for (uint tap = 0u; tap < conv_kernel; ++tap) {
    const int src_tok = int(token) + int(tap) - int(history_tokens);
    const float inp =
        src_tok < 0
            ? float(history[ulong(uint(src_tok + int(history_tokens))) *
                                ulong(conv_width) +
                            ulong(channel)])
            : float(qkv[ulong(uint(src_tok)) * ulong(conv_width) +
                        ulong(channel)]);
    sum = fma(
        inp,
        float(conv_weight[ulong(channel) * ulong(conv_kernel) + ulong(tap)]),
        sum);
  }
  return bfloat(sum / (1.0f + exp(-sum)));
}

GDN_NAX_ALWAYS_INLINE
inline void gdn_load_conv4_weights_vec4(
    device const bfloat *conv_weight,
    uint ch0,
    thread float4 &w0,
    thread float4 &w1,
    thread float4 &w2,
    thread float4 &w3) {
  device const bfloat4 *wp =
      reinterpret_cast<device const bfloat4 *>(conv_weight + ulong(ch0) * 4ul);
  const float4 c0 = float4(wp[0]);
  const float4 c1 = float4(wp[1]);
  const float4 c2 = float4(wp[2]);
  const float4 c3 = float4(wp[3]);
  w0 = float4(c0.x, c1.x, c2.x, c3.x);
  w1 = float4(c0.y, c1.y, c2.y, c3.y);
  w2 = float4(c0.z, c1.z, c2.z, c3.z);
  w3 = float4(c0.w, c1.w, c2.w, c3.w);
}

GDN_NAX_ALWAYS_INLINE
inline float4 gdn_read_qkv_or_hist_vec4(
    device const bfloat *qkv,
    device const bfloat *history,
    int src_tok,
    uint ch0,
    uint conv_width) {
  device const bfloat *ptr =
      src_tok < 0
          ? (history + ulong(uint(src_tok + 3)) * ulong(conv_width) + ulong(ch0))
          : (qkv + ulong(uint(src_tok)) * ulong(conv_width) + ulong(ch0));
  return float4(*reinterpret_cast<device const bfloat4 *>(ptr));
}

GDN_NAX_ALWAYS_INLINE
inline float gdn_read_qkv_or_hist_scalar(
    device const bfloat *qkv,
    device const bfloat *history,
    int src_tok,
    uint ch,
    uint conv_width) {
  return src_tok < 0
             ? float(history[ulong(uint(src_tok + 3)) * ulong(conv_width) +
                             ulong(ch)])
             : float(qkv[ulong(uint(src_tok)) * ulong(conv_width) + ulong(ch)]);
}

template <uint NumSg, uint Rows>
GDN_NAX_ALWAYS_INLINE
inline void gdn_stage_conv_silu_128(
    device const bfloat *qkv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    uint t_base,
    uint valid_rows,
    uint head_ch_off,
    uint conv_width,
    uint conv_kernel,
    threadgroup bfloat *tg_dst,
    uint sg,
    uint lane) {
  constexpr uint kRowsPerSg = Rows / NumSg;
  const uint r_start = sg * kRowsPerSg;
  const uint ch_local = lane * 4u;
  const uint ch0 = head_ch_off + ch_local;

  if (conv_kernel == 4u && (conv_width & 3u) == 0u) {
    float4 w0, w1, w2, w3;
    gdn_load_conv4_weights_vec4(conv_weight, ch0, w0, w1, w2, w3);
    float4 xm3 = float4(0.0f), xm2 = float4(0.0f), xm1 = float4(0.0f);
    if (r_start < valid_rows) {
      const int base_tok = int(t_base + r_start);
      xm3 = gdn_read_qkv_or_hist_vec4(qkv, history, base_tok - 3, ch0, conv_width);
      xm2 = gdn_read_qkv_or_hist_vec4(qkv, history, base_tok - 2, ch0, conv_width);
      xm1 = gdn_read_qkv_or_hist_vec4(qkv, history, base_tok - 1, ch0, conv_width);
    }
    for (uint s = 0u; s < kRowsPerSg; ++s) {
      const uint r = r_start + s;
      bfloat4 out_v = bfloat4(bfloat(0.0f));
      if (r < valid_rows) {
        const float4 xcur = float4(*reinterpret_cast<device const bfloat4 *>(
            qkv + ulong(t_base + r) * ulong(conv_width) + ulong(ch0)));
        float4 sum = fma(xm3, w0, float4(0.0f));
        sum = fma(xm2, w1, sum);
        sum = fma(xm1, w2, sum);
        sum = fma(xcur, w3, sum);
        out_v = bfloat4(sum / (1.0f + exp(-sum)));
        xm3 = xm2;
        xm2 = xm1;
        xm1 = xcur;
      }
      *reinterpret_cast<threadgroup bfloat4 *>(tg_dst + r * 128u + ch_local) = out_v;
    }
  } else {
    for (uint s = 0u; s < kRowsPerSg; ++s) {
      const uint r = r_start + s;
      for (uint j = 0u; j < 4u; ++j) {
        tg_dst[r * 128u + ch_local + j] =
            (r < valid_rows)
                ? gdn_eval_conv_silu_scalar(qkv, conv_weight, history, t_base + r,
                                            ch0 + j, conv_width, conv_kernel)
                : bfloat(0.0f);
      }
    }
  }
}

template <uint TileDv>
GDN_NAX_ALWAYS_INLINE
inline void gdn_stage_conv_silu_v_slice_fast(
    device const bfloat *qkv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    uint t0,
    uint clen,
    uint v_tg_col_base,
    uint dv_valid,
    uint conv_width,
    uint conv_kernel,
    threadgroup bfloat *tg_V,
    uint tid) {
  // Low-GPR vectorized 2-row (dv16) / 4-row (dv32) Conv1D+SiLU using bfloat4
  // weights & sliding window (ONLY 14 GPRs vs 28 GPRs for float4 -> 0 B spills!).
  // Handles ANY dv_valid <= TileDv (including K_dv=10 dv_valid=12..13 & K_dv=5 dv_valid=25..26!).
  constexpr uint kThrPerRow = TileDv / 4u;                // 4 for dv16, 8 for dv32
  constexpr uint kRowsPerThr = 64u / (128u / kThrPerRow); // 2 for dv16, 4 for dv32
  const uint dv_l4 = (tid & (kThrPerRow - 1u)) * 4u;
  const uint r_start = (tid / kThrPerRow) * kRowsPerThr;
  const uint ch0 = v_tg_col_base + dv_l4;

  if (conv_kernel == 4u) {
    const bool full_vec4 =
        ((conv_width & 3u) == 0u) && ((ch0 & 3u) == 0u) && (dv_l4 + 4u <= dv_valid);
    if (full_vec4) {
      // Fast low-GPR sliding-window bfloat4 Conv1D+SiLU: load conv weights ONCE
      // per chunk (4 GPRs, zero overlap with NAX MMA) + branchless fast path for
      // t0 + r_start >= 3 (99.9% of chunks -> 0 B spills, minimal conv latency!).
      device const bfloat4 *wp =
          reinterpret_cast<device const bfloat4 *>(conv_weight + ulong(ch0) * 4ul);
      const bfloat4 c0 = wp[0], c1 = wp[1], c2 = wp[2], c3 = wp[3];
      bfloat4 xm3 = bfloat4(bfloat(0.0f));
      bfloat4 xm2 = bfloat4(bfloat(0.0f));
      bfloat4 xm1 = bfloat4(bfloat(0.0f));
      if (r_start < clen) {
        const uint abs_r = t0 + r_start;
        if (abs_r >= 3u) {
          device const bfloat *qp =
              qkv + ulong(abs_r - 3u) * ulong(conv_width) + ulong(ch0);
          xm3 = *reinterpret_cast<device const bfloat4 *>(qp);
          xm2 = *reinterpret_cast<device const bfloat4 *>(qp + ulong(conv_width));
          xm1 = *reinterpret_cast<device const bfloat4 *>(qp + 2ul * ulong(conv_width));
        } else {
          const int base_tok = int(abs_r);
          xm3 = *reinterpret_cast<device const bfloat4 *>(
              (base_tok - 3 < 0)
                  ? (history + ulong(uint(base_tok)) * ulong(conv_width) + ulong(ch0))
                  : (qkv + ulong(uint(base_tok - 3)) * ulong(conv_width) + ulong(ch0)));
          xm2 = *reinterpret_cast<device const bfloat4 *>(
              (base_tok - 2 < 0)
                  ? (history + ulong(uint(base_tok + 1)) * ulong(conv_width) + ulong(ch0))
                  : (qkv + ulong(uint(base_tok - 2)) * ulong(conv_width) + ulong(ch0)));
          xm1 = *reinterpret_cast<device const bfloat4 *>(
              (base_tok - 1 < 0)
                  ? (history + ulong(uint(base_tok + 2)) * ulong(conv_width) + ulong(ch0))
                  : (qkv + ulong(uint(base_tok - 1)) * ulong(conv_width) + ulong(ch0)));
        }
      }
      #pragma clang loop unroll(disable)
      for (uint s = 0u; s < kRowsPerThr; ++s) {
        const uint r = r_start + s;
        bfloat4 out_v = bfloat4(bfloat(0.0f));
        if (r < clen) {
          const bfloat4 xcur = *reinterpret_cast<device const bfloat4 *>(
              qkv + ulong(t0 + r) * ulong(conv_width) + ulong(ch0));
          float4 sum = fma(float4(xm3), float4(c0.x, c1.x, c2.x, c3.x), float4(0.0f));
          sum = fma(float4(xm2), float4(c0.y, c1.y, c2.y, c3.y), sum);
          sum = fma(float4(xm1), float4(c0.z, c1.z, c2.z, c3.z), sum);
          sum = fma(float4(xcur), float4(c0.w, c1.w, c2.w, c3.w), sum);
          out_v = bfloat4(sum / (1.0f + exp(-sum)));
          xm3 = xm2;
          xm2 = xm1;
          xm1 = xcur;
        }
        *reinterpret_cast<threadgroup bfloat4 *>(tg_V + r * TileDv + dv_l4) = out_v;
      }
      return;
    }

    // Fast 2-row/4-row partial/unaligned tail (handles K_dv=10 & K_dv=5 at full speed!).
    #pragma clang loop unroll(disable)
    for (uint s = 0u; s < kRowsPerThr; ++s) {
      const uint r = r_start + s;
      bfloat4 out_v = bfloat4(bfloat(0.0f));
      if (r < clen && dv_l4 < dv_valid) {
        const int tok = int(t0 + r);
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
          if (dv_l4 + j < dv_valid) {
            const uint ch = ch0 + j;
            const bfloat4 w = *reinterpret_cast<device const bfloat4 *>(
                conv_weight + ulong(ch) * 4ul);
            float sum = fma(gdn_read_qkv_or_hist_scalar(qkv, history, tok - 3, ch, conv_width),
                            float(w.x), 0.0f);
            sum = fma(gdn_read_qkv_or_hist_scalar(qkv, history, tok - 2, ch, conv_width),
                      float(w.y), sum);
            sum = fma(gdn_read_qkv_or_hist_scalar(qkv, history, tok - 1, ch, conv_width),
                      float(w.z), sum);
            sum = fma(float(qkv[ulong(uint(tok)) * ulong(conv_width) + ulong(ch)]),
                      float(w.w), sum);
            out_v[j] = bfloat(sum / (1.0f + exp(-sum)));
          }
        }
      }
      *reinterpret_cast<threadgroup bfloat4 *>(tg_V + r * TileDv + dv_l4) = out_v;
    }
    return;
  }

  for (uint s = 0u; s < kRowsPerThr; ++s) {
    const uint r = r_start + s;
    for (uint j = 0u; j < 4u; ++j) {
      tg_V[r * TileDv + dv_l4 + j] =
          (r < clen && dv_l4 + j < dv_valid)
              ? gdn_eval_conv_silu_scalar(qkv, conv_weight, history, t0 + r,
                                          ch0 + j, conv_width, conv_kernel)
              : bfloat(0.0f);
    }
  }
}

template <uint TileDv>
GDN_NAX_ALWAYS_INLINE
inline void gdn_stage_conv_silu_v_slice(
    device const bfloat *qkv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    uint t0,
    uint clen,
    uint v_tg_col_base,
    uint dv_valid,
    uint conv_width,
    uint conv_kernel,
    threadgroup bfloat *tg_V,
    uint tid) {
  gdn_stage_conv_silu_v_slice_fast<TileDv>(
      qkv, conv_weight, history, t0, clen, v_tg_col_base, dv_valid,
      conv_width, conv_kernel, tg_V, tid);
}

GDN_NAX_ALWAYS_INLINE
inline void gdn_compute_head_decay_beta_sg0(
    device const bfloat *a,
    device const bfloat *b,
    device const bfloat *a_log,
    device const bfloat *dt_bias,
    uint t0,
    uint clen,
    uint hv,
    uint value_heads,
    threadgroup float *tg_G,
    threadgroup float *tg_beta,
    device float *ctrl_pref,
    device float *ctrl_suff,
    device float *ctrl_meta,
    bool emit_ctrl,
    uint lane) {
  const float dt_bias_val = float(dt_bias[hv]);
  const float neg_exp_a = -exp(float(a_log[hv]));

  float g0 = 0.0f, beta0 = 0.0f;
  if (lane < clen) {
    const ulong h_idx0 = ulong(t0 + lane) * ulong(value_heads) + ulong(hv);
    const float act0 = float(a[h_idx0]) + dt_bias_val;
    const float sp0 = act0 > 20.0f ? act0 : log(1.0f + exp(act0));
    g0 = min(0.0f, neg_exp_a * sp0);
    beta0 = 1.0f / (1.0f + exp(-float(b[h_idx0])));
  }
  float g_cum0 = g0;
  #pragma clang loop unroll(full)
  for (ushort off = 1u; off < 32u; off <<= 1u) {
    const float prev = simd_shuffle_up(g_cum0, off);
    if (lane >= off) {
      g_cum0 += prev;
    }
  }
  g_cum0 = min(0.0f, g_cum0);
  const float g_mid = simd_broadcast(g_cum0, 31u);

  const uint r1 = 32u + lane;
  float g1 = 0.0f, beta1 = 0.0f;
  if (r1 < clen) {
    const ulong h_idx1 = ulong(t0 + r1) * ulong(value_heads) + ulong(hv);
    const float act1 = float(a[h_idx1]) + dt_bias_val;
    const float sp1 = act1 > 20.0f ? act1 : log(1.0f + exp(act1));
    g1 = min(0.0f, neg_exp_a * sp1);
    beta1 = 1.0f / (1.0f + exp(-float(b[h_idx1])));
  }
  float g_cum1 = g1;
  #pragma clang loop unroll(full)
  for (ushort off = 1u; off < 32u; off <<= 1u) {
    const float prev = simd_shuffle_up(g_cum1, off);
    if (lane >= off) {
      g_cum1 += prev;
    }
  }
  g_cum1 = min(0.0f, g_mid + g_cum1);
  const float g_last = simd_broadcast(g_cum1, 31u);

  tg_G[lane] = g_cum0;
  tg_G[r1] = g_cum1;
  tg_beta[lane] = beta0;
  tg_beta[r1] = beta1;
  if (emit_ctrl) {
    ctrl_pref[lane] = exp(g_cum0);
    ctrl_pref[r1] = exp(g_cum1);
    ctrl_suff[lane] = exp(min(0.0f, g_last - g_cum0));
    ctrl_suff[r1] = exp(min(0.0f, g_last - g_cum1));
    if (lane == 0u) {
      ctrl_meta[0] = exp(g_last);
    }
  }
}

// ============================================================================
// Zero-Spill Stage 1 (`gdn_chunk_prepare_{c64,conv_c64}`):
//   - 0 `execution_simdgroups<4>` cooperative_tensor allocas!
//   - Apple10: register-fragment `nax_mma_16x32x16_nt<bfloat>` for Q@K^T & K@K^T
//     (4 SGs x [16x64] = 32 floats/lane, non-overlapping Pass A & Pass B -> 0 spills!)
//     + exact IEEE FP32 `simdgroup_float8x8` for off-diagonal L10@X00 & X11@T!
//   - Apple9: pure-FP32 SIMD dot product + `simdgroup_float8x8` (also 0 spills!).
//   - In-place forward substitution directly in tg_slot0/tg_slot1 (0 `x_col` array!).
// ============================================================================
template <bool FuseConv, typename ParamsT>
GDN_NAX_ALWAYS_INLINE
inline void gdn_chunk_prepare_c64_impl(
    device const bfloat *qkv_or_conv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    device const bfloat *a,
    device const bfloat *b,
    device const bfloat *a_log,
    device const bfloat *dt_bias,
    device bfloat *qk_ws,
    device uchar *ctrl_ws,
    constant ParamsT &params,
    threadgroup GdnPrepareTgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg) {
  const uint hk = tg.x;
  const uint chunk_idx = tg.y;
  const uint t0 = chunk_idx * kChunkSize;
  if (hk >= params.key_heads || t0 >= params.tokens || params.key_dim != kKeyDim) {
    return;
  }

  threadgroup bfloat *tg_buf = tg_mem.buf;
  const uint clen = min(kChunkSize, params.tokens - t0);
  const uint repeat = params.value_heads / params.key_heads;
  const uint key_width = params.key_heads * kKeyDim;
  const uint q_head_off = hk * kKeyDim;
  const uint k_head_off = key_width + q_head_off;
  const float inv_scale = rsqrt(128.0f);
  const float inv_scale_sq = 1.0f / 128.0f;

  device bfloat *ws_nk =
      qk_ws + (ulong(chunk_idx) * ulong(params.key_heads) + ulong(hk)) * kQkPairBfloats;
  device bfloat *ws_nq = ws_nk + ulong(kChunkSize) * ulong(kKeyDim);

  // 1. Stage & FP32 L2-normalize K[0..63, 0..127] -> ws_nk.
  if constexpr (FuseConv) {
    gdn_stage_conv_silu_128<4u, kChunkSize>(
        qkv_or_conv, conv_weight, history, t0, clen, k_head_off,
        params.conv_width, params.conv_kernel, tg_buf, sg, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint r = sg; r < kChunkSize; r += 4u) {
    bfloat nk0 = bfloat(0.0f), nk1 = bfloat(0.0f),
           nk2 = bfloat(0.0f), nk3 = bfloat(0.0f);
    const uint row_off = r * kKeyDim;
    if (r < clen) {
      float k0, k1, k2, k3;
      if constexpr (FuseConv) {
        k0 = float(tg_buf[row_off + lane]);
        k1 = float(tg_buf[row_off + lane + 32u]);
        k2 = float(tg_buf[row_off + lane + 64u]);
        k3 = float(tg_buf[row_off + lane + 96u]);
      } else {
        device const bfloat *k_ptr =
            qkv_or_conv + ulong(t0 + r) * ulong(params.conv_width) + ulong(k_head_off);
        k0 = float(k_ptr[lane]);
        k1 = float(k_ptr[lane + 32u]);
        k2 = float(k_ptr[lane + 64u]);
        k3 = float(k_ptr[lane + 96u]);
      }
      float k_sq = fma(k0, k0, 0.0f);
      k_sq = fma(k1, k1, k_sq);
      k_sq = fma(k2, k2, k_sq);
      k_sq = fma(k3, k3, k_sq);
      const float k_inv = rsqrt(simd_sum(k_sq) * inv_scale_sq + 1.0e-6f) * inv_scale;
      nk0 = bfloat(k0 * k_inv);
      nk1 = bfloat(k1 * k_inv);
      nk2 = bfloat(k2 * k_inv);
      nk3 = bfloat(k3 * k_inv);
    }
    ws_nk[row_off + lane] = nk0;
    ws_nk[row_off + lane + 32u] = nk1;
    ws_nk[row_off + lane + 64u] = nk2;
    ws_nk[row_off + lane + 96u] = nk3;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  // 2. Stage & FP32 L2-normalize Q[0..63, 0..127] -> tg_buf & ws_nq.
  if constexpr (FuseConv) {
    gdn_stage_conv_silu_128<4u, kChunkSize>(
        qkv_or_conv, conv_weight, history, t0, clen, q_head_off,
        params.conv_width, params.conv_kernel, tg_buf, sg, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint r = sg; r < kChunkSize; r += 4u) {
    const uint row_off = r * kKeyDim;
    bfloat nq0 = bfloat(0.0f), nq1 = bfloat(0.0f),
           nq2 = bfloat(0.0f), nq3 = bfloat(0.0f);
    if (r < clen) {
      float q0, q1, q2, q3;
      if constexpr (FuseConv) {
        q0 = float(tg_buf[row_off + lane]);
        q1 = float(tg_buf[row_off + lane + 32u]);
        q2 = float(tg_buf[row_off + lane + 64u]);
        q3 = float(tg_buf[row_off + lane + 96u]);
      } else {
        device const bfloat *q_ptr =
            qkv_or_conv + ulong(t0 + r) * ulong(params.conv_width) + ulong(q_head_off);
        q0 = float(q_ptr[lane]);
        q1 = float(q_ptr[lane + 32u]);
        q2 = float(q_ptr[lane + 64u]);
        q3 = float(q_ptr[lane + 96u]);
      }
      float q_sq = fma(q0, q0, 0.0f);
      q_sq = fma(q1, q1, q_sq);
      q_sq = fma(q2, q2, q_sq);
      q_sq = fma(q3, q3, q_sq);
      const float q_inv =
          rsqrt(simd_sum(q_sq) * inv_scale_sq + 1.0e-6f) * inv_scale_sq;
      nq0 = bfloat(q0 * q_inv);
      nq1 = bfloat(q1 * q_inv);
      nq2 = bfloat(q2 * q_inv);
      nq3 = bfloat(q3 * q_inv);
    }
    tg_buf[row_off + lane] = nq0;
    tg_buf[row_off + lane + 32u] = nq1;
    tg_buf[row_off + lane + 64u] = nq2;
    tg_buf[row_off + lane + 96u] = nq3;
    ws_nq[row_off + lane] = nq0;
    ws_nq[row_off + lane + 32u] = nq1;
    ws_nq[row_off + lane + 64u] = nq2;
    ws_nq[row_off + lane + 96u] = nq3;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  threadgroup float *tg_f32 = reinterpret_cast<threadgroup float *>(tg_buf);
  threadgroup float *tg_slot0 = tg_f32;
  threadgroup float *tg_slot1 = tg_f32 + 1024u;
  threadgroup float *tg_slot2 = tg_f32 + 2048u;
  threadgroup float *tg_G = tg_f32 + 3072u;
  threadgroup float *tg_beta = tg_f32 + 3136u;

  const bool is_apple10 = gdn_has_apple10_nax_layout();
  const ushort2 lc = nax_apple10_lane_coord(ushort(lane));
  const uint fm = uint(lc.x);
  const uint fn = uint(lc.y);
  const uint r_sg_base = sg * 16u;

  // 3. Pass A: Compute raw QK^T [16x64]/SG -> emit ctrl_pref, ctrl_suff, ctrl_meta, ctrl_mqk.
  {
    vec<float, 8> qkt0(0.0f), qkt1(0.0f), qkt2(0.0f), qkt3(0.0f);
    if (is_apple10) {
      for (uint k0 = 0u; k0 < kKeyDim; k0 += 16u) {
        const vec<bfloat, 8> a = gdn_load_frag_tg_bf16<kKeyDim>(
            tg_buf + r_sg_base * kKeyDim + k0, fm, fn);
        const vec<bfloat, 8> b0 =
            gdn_load_frag_dev_bf16(ws_nk + 0u * kKeyDim + k0, kKeyDim, fm, fn);
        const vec<bfloat, 8> b1 =
            gdn_load_frag_dev_bf16(ws_nk + 16u * kKeyDim + k0, kKeyDim, fm, fn);
        nax_mma_16x32x16_nt<bfloat>(qkt0, qkt1, a, b0, b1);
        const vec<bfloat, 8> b2 =
            gdn_load_frag_dev_bf16(ws_nk + 32u * kKeyDim + k0, kKeyDim, fm, fn);
        const vec<bfloat, 8> b3 =
            gdn_load_frag_dev_bf16(ws_nk + 48u * kKeyDim + k0, kKeyDim, fm, fn);
        nax_mma_16x32x16_nt<bfloat>(qkt2, qkt3, a, b2, b3);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint rep = 0u; rep < repeat; ++rep) {
      const uint hv = hk * repeat + rep;
      if (hv >= params.value_heads) {
        break;
      }
      device uchar *ctrl_raw =
          ctrl_ws + (ulong(chunk_idx) * ulong(params.value_heads) + ulong(hv)) * kCtrlStrideBytes;
      device bfloat *ctrl_mqk =
          reinterpret_cast<device bfloat *>(ctrl_raw + kCtrlMqkByteOff);
      device float *ctrl_pref =
          reinterpret_cast<device float *>(ctrl_raw + kCtrlPrefByteOff);
      device float *ctrl_suff =
          reinterpret_cast<device float *>(ctrl_raw + kCtrlSuffByteOff);
      device float *ctrl_meta =
          reinterpret_cast<device float *>(ctrl_raw + kCtrlMetaByteOff);

      if (sg == 0u) {
        gdn_compute_head_decay_beta_sg0(
            a, b, a_log, dt_bias, t0, clen, hv, params.value_heads, tg_G,
            tg_beta, ctrl_pref, ctrl_suff, ctrl_meta, /*emit_ctrl=*/true, lane);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (is_apple10) {
        #define PREP_WRITE_MQK_TILE(COL_BASE, VEC)                             \
          do {                                                                 \
            for (uint h = 0u; h < 2u; ++h) {                                   \
              const uint row = r_sg_base + fm + h * 8u;                        \
              const float g_row = (row < clen) ? tg_G[row] : 0.0f;             \
              bfloat4 out4;                                                    \
              for (uint j = 0u; j < 4u; ++j) {                                 \
                const uint col = (COL_BASE) + fn + j;                          \
                float m_val = 0.0f;                                            \
                if (row >= col && row < clen && col < clen) {                  \
                  m_val = exp(min(0.0f, g_row - tg_G[col])) *                  \
                          (VEC)[h * 4u + j];                                   \
                }                                                              \
                out4[j] = bfloat(m_val);                                       \
              }                                                                \
              *reinterpret_cast<device bfloat4 *>(                             \
                  ctrl_mqk + row * kChunkSize + (COL_BASE) + fn) = out4;       \
            }                                                                  \
          } while (false)
        PREP_WRITE_MQK_TILE(0u, qkt0);
        PREP_WRITE_MQK_TILE(16u, qkt1);
        PREP_WRITE_MQK_TILE(32u, qkt2);
        PREP_WRITE_MQK_TILE(48u, qkt3);
        #undef PREP_WRITE_MQK_TILE
      } else {
        for (uint row = r_sg_base; row < r_sg_base + 16u; ++row) {
          for (uint col = 0u; col < kChunkSize; ++col) {
            float m_val = 0.0f;
            if (row >= col && row < clen && col < clen) {
              const float4 q4 = float4(*reinterpret_cast<device const bfloat4 *>(
                  ws_nq + row * kKeyDim + lane * 4u));
              const float4 k4 = float4(*reinterpret_cast<device const bfloat4 *>(
                  ws_nk + col * kKeyDim + lane * 4u));
              const float dp = simd_sum(dot(q4, k4));
              m_val = exp(min(0.0f, tg_G[row] - tg_G[col])) * dp;
            }
            if (lane == 0u) {
              ctrl_mqk[row * kChunkSize + col] = bfloat(m_val);
            }
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // 4. Pass B: Compute raw KK^T [16x64]/SG, then solve (I_64 + L)^-1 * diag(beta).
  vec<float, 8> kkt0(0.0f), kkt1(0.0f), kkt2(0.0f), kkt3(0.0f);
  if (is_apple10) {
    for (uint r = sg; r < kChunkSize; r += 4u) {
      const uint row_off = r * kKeyDim;
      *reinterpret_cast<threadgroup bfloat4 *>(tg_buf + row_off + lane * 4u) =
          *reinterpret_cast<device const bfloat4 *>(ws_nk + row_off + lane * 4u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k0 = 0u; k0 < kKeyDim; k0 += 16u) {
      const vec<bfloat, 8> a = gdn_load_frag_tg_bf16<kKeyDim>(
          tg_buf + r_sg_base * kKeyDim + k0, fm, fn);
      const vec<bfloat, 8> b0 =
          gdn_load_frag_tg_bf16<kKeyDim>(tg_buf + 0u * kKeyDim + k0, fm, fn);
      const vec<bfloat, 8> b1 =
          gdn_load_frag_tg_bf16<kKeyDim>(tg_buf + 16u * kKeyDim + k0, fm, fn);
      nax_mma_16x32x16_nt<bfloat>(kkt0, kkt1, a, b0, b1);
      const vec<bfloat, 8> b2 =
          gdn_load_frag_tg_bf16<kKeyDim>(tg_buf + 32u * kKeyDim + k0, fm, fn);
      const vec<bfloat, 8> b3 =
          gdn_load_frag_tg_bf16<kKeyDim>(tg_buf + 48u * kKeyDim + k0, fm, fn);
      nax_mma_16x32x16_nt<bfloat>(kkt2, kkt3, a, b2, b3);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint rep = 0u; rep < repeat; ++rep) {
    const uint hv = hk * repeat + rep;
    if (hv >= params.value_heads) {
      break;
    }
    device uchar *ctrl_raw =
        ctrl_ws + (ulong(chunk_idx) * ulong(params.value_heads) + ulong(hv)) * kCtrlStrideBytes;
    device bfloat *ctrl_solve = reinterpret_cast<device bfloat *>(ctrl_raw);

    if (sg == 0u) {
      gdn_compute_head_decay_beta_sg0(
          a, b, a_log, dt_bias, t0, clen, hv, params.value_heads, tg_G,
          tg_beta, nullptr, nullptr, nullptr, /*emit_ctrl=*/false, lane);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_apple10) {
      #define PREP_STORE_L_TILE(COL_BASE, VEC)                                 \
        do {                                                                   \
          for (uint h = 0u; h < 2u; ++h) {                                     \
            const uint row = r_sg_base + fm + h * 8u;                          \
            const float g_row = (row < clen) ? tg_G[row] : 0.0f;               \
            const float b_row = (row < clen) ? tg_beta[row] : 0.0f;            \
            for (uint j = 0u; j < 4u; ++j) {                                   \
              const uint col = (COL_BASE) + fn + j;                            \
              float l_val = 0.0f;                                              \
              if (row > col && row < clen) {                                   \
                l_val = b_row * exp(min(0.0f, g_row - tg_G[col])) *            \
                        (VEC)[h * 4u + j];                                     \
              }                                                                \
              if (row < kHalfChunk && col < kHalfChunk) {                      \
                tg_slot0[row * kHalfChunk + col] = l_val;                      \
              } else if (row >= kHalfChunk && col >= kHalfChunk) {             \
                tg_slot1[(row - kHalfChunk) * kHalfChunk + (col - kHalfChunk)] =\
                    l_val;                                                     \
              } else if (row >= kHalfChunk && col < kHalfChunk) {              \
                tg_slot2[(row - kHalfChunk) * kHalfChunk + col] = l_val;       \
              }                                                                \
            }                                                                  \
          }                                                                    \
        } while (false)
      PREP_STORE_L_TILE(0u, kkt0);
      PREP_STORE_L_TILE(16u, kkt1);
      PREP_STORE_L_TILE(32u, kkt2);
      PREP_STORE_L_TILE(48u, kkt3);
      #undef PREP_STORE_L_TILE
    } else {
      for (uint row = r_sg_base; row < r_sg_base + 16u; ++row) {
        for (uint col = 0u; col < kChunkSize; ++col) {
          float l_val = 0.0f;
          if (row > col && row < clen) {
            const float4 k_r = float4(*reinterpret_cast<device const bfloat4 *>(
                ws_nk + row * kKeyDim + lane * 4u));
            const float4 k_c = float4(*reinterpret_cast<device const bfloat4 *>(
                ws_nk + col * kKeyDim + lane * 4u));
            l_val = tg_beta[row] * exp(min(0.0f, tg_G[row] - tg_G[col])) *
                    simd_sum(dot(k_r, k_c));
          }
          if (lane == 0u) {
            if (row < kHalfChunk && col < kHalfChunk) {
              tg_slot0[row * kHalfChunk + col] = l_val;
            } else if (row >= kHalfChunk && col >= kHalfChunk) {
              tg_slot1[(row - kHalfChunk) * kHalfChunk + (col - kHalfChunk)] =
                  l_val;
            } else if (row >= kHalfChunk && col < kHalfChunk) {
              tg_slot2[(row - kHalfChunk) * kHalfChunk + col] = l_val;
            }
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // In-place exact 2x2 block FP32 inversion of (I_64 + L) directly in tg_slot0/tg_slot1.
    if (sg == 0u) {
      const float beta_col = (lane < clen) ? tg_beta[lane] : 0.0f;
      for (uint row = 0u; row < kHalfChunk; ++row) {
        float x_val = 0.0f;
        if (row < clen) {
          if (lane == row) {
            x_val = 1.0f;
          } else if (lane < row) {
            float sum = 0.0f;
            for (uint k = lane; k < row; ++k) {
              sum = fma(tg_slot0[row * kHalfChunk + k],
                        tg_slot0[k * kHalfChunk + lane], sum);
            }
            x_val = -sum;
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tg_slot0[row * kHalfChunk + lane] = x_val;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        ctrl_solve[row * kChunkSize + lane] = bfloat(x_val * beta_col);
        ctrl_solve[row * kChunkSize + (kHalfChunk + lane)] = bfloat(0.0f);
      }
    } else if (sg == 1u) {
      if (clen > kHalfChunk) {
        const uint clen1 = clen - kHalfChunk;
        const float beta_col =
            (kHalfChunk + lane < clen) ? tg_beta[kHalfChunk + lane] : 0.0f;
        for (uint row = 0u; row < kHalfChunk; ++row) {
          float x_val = 0.0f;
          if (row < clen1) {
            if (lane == row) {
              x_val = 1.0f;
            } else if (lane < row) {
              float sum = 0.0f;
              for (uint k = lane; k < row; ++k) {
                sum = fma(tg_slot1[row * kHalfChunk + k],
                          tg_slot1[k * kHalfChunk + lane], sum);
              }
              x_val = -sum;
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          tg_slot1[row * kHalfChunk + lane] = x_val;
          simdgroup_barrier(mem_flags::mem_threadgroup);
          ctrl_solve[(kHalfChunk + row) * kChunkSize + (kHalfChunk + lane)] =
              bfloat(x_val * beta_col);
        }
      } else {
        for (uint row = 0u; row < kHalfChunk; ++row) {
          ctrl_solve[(kHalfChunk + row) * kChunkSize + lane] = bfloat(0.0f);
          ctrl_solve[(kHalfChunk + row) * kChunkSize + (kHalfChunk + lane)] =
              bfloat(0.0f);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Off-diagonal 32x32 block X10 = -X11 @ (L10 @ X00) via exact IEEE FP32
    // simdgroup_float8x8 (4 SGs x [8x32] = 8 floats/lane -> 0 B spills!).
    if (clen > kHalfChunk) {
      const uint sg_r = sg * 8u;
      simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      simdgroup_float8x8 c1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      simdgroup_float8x8 c2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      simdgroup_float8x8 c3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      #pragma clang loop unroll(full)
      for (uint ks = 0u; ks < 4u; ++ks) {
        simdgroup_float8x8 ma, mb0, mb1, mb2, mb3;
        simdgroup_load(ma, tg_slot2 + sg_r * kHalfChunk + ks * 8u, kHalfChunk);
        simdgroup_load(mb0, tg_slot0 + (ks * 8u) * kHalfChunk + 0u, kHalfChunk);
        simdgroup_load(mb1, tg_slot0 + (ks * 8u) * kHalfChunk + 8u, kHalfChunk);
        simdgroup_load(mb2, tg_slot0 + (ks * 8u) * kHalfChunk + 16u, kHalfChunk);
        simdgroup_load(mb3, tg_slot0 + (ks * 8u) * kHalfChunk + 24u, kHalfChunk);
        simdgroup_multiply_accumulate(c0, ma, mb0, c0);
        simdgroup_multiply_accumulate(c1, ma, mb1, c1);
        simdgroup_multiply_accumulate(c2, ma, mb2, c2);
        simdgroup_multiply_accumulate(c3, ma, mb3, c3);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_store(c0, tg_slot2 + sg_r * kHalfChunk + 0u, kHalfChunk);
      simdgroup_store(c1, tg_slot2 + sg_r * kHalfChunk + 8u, kHalfChunk);
      simdgroup_store(c2, tg_slot2 + sg_r * kHalfChunk + 16u, kHalfChunk);
      simdgroup_store(c3, tg_slot2 + sg_r * kHalfChunk + 24u, kHalfChunk);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      c1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      c2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      c3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      #pragma clang loop unroll(full)
      for (uint ks = 0u; ks < 4u; ++ks) {
        simdgroup_float8x8 ma, mb0, mb1, mb2, mb3;
        simdgroup_load(ma, tg_slot1 + sg_r * kHalfChunk + ks * 8u, kHalfChunk);
        simdgroup_load(mb0, tg_slot2 + (ks * 8u) * kHalfChunk + 0u, kHalfChunk);
        simdgroup_load(mb1, tg_slot2 + (ks * 8u) * kHalfChunk + 8u, kHalfChunk);
        simdgroup_load(mb2, tg_slot2 + (ks * 8u) * kHalfChunk + 16u, kHalfChunk);
        simdgroup_load(mb3, tg_slot2 + (ks * 8u) * kHalfChunk + 24u, kHalfChunk);
        simdgroup_multiply_accumulate(c0, ma, mb0, c0);
        simdgroup_multiply_accumulate(c1, ma, mb1, c1);
        simdgroup_multiply_accumulate(c2, ma, mb2, c2);
        simdgroup_multiply_accumulate(c3, ma, mb3, c3);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Reuse tg_slot0 (X00 is dead!) to stage X10 for clean coalesced ctrl_solve store.
      simdgroup_store(c0, tg_slot0 + sg_r * kHalfChunk + 0u, kHalfChunk);
      simdgroup_store(c1, tg_slot0 + sg_r * kHalfChunk + 8u, kHalfChunk);
      simdgroup_store(c2, tg_slot0 + sg_r * kHalfChunk + 16u, kHalfChunk);
      simdgroup_store(c3, tg_slot0 + sg_r * kHalfChunk + 24u, kHalfChunk);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const float beta_col = (lane < clen) ? tg_beta[lane] : 0.0f;
      for (uint r = 0u; r < 8u; ++r) {
        const uint row = sg_r + r;
        const float s_val = (kHalfChunk + row < clen)
                                ? (-tg_slot0[row * kHalfChunk + lane] * beta_col)
                                : 0.0f;
        ctrl_solve[(kHalfChunk + row) * kChunkSize + lane] = bfloat(s_val);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ============================================================================
// Zero-Spill Legacy Mega Recurrence (EpilogueMode: 1 = BF16, 2 = FP16 + zero pad)
// Uses register-fragment `nax_mma_16x16x32` / `nax_mma_16x32x16_nn` (0 spills!).
// ============================================================================
template <bool FuseConv, uint EpilogueMode, typename ParamsT>
GDN_NAX_ALWAYS_INLINE
inline void gdn_chunk_recurrence_c64_impl(
    device const bfloat *qkv_or_conv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    device const bfloat *qk_ws,
    device const uchar *ctrl_ws,
    device float *state,
    device const bfloat *z,
    device const bfloat *norm_weight,
    device bfloat *out_bf16,
    device half *out_f16,
    constant ParamsT &params,
    threadgroup GdnMegaTgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg) {
  const uint hv = tg.x;
  if (hv >= params.value_heads || sg >= 8u || params.key_dim != kKeyDim ||
      params.value_dim != kValDim) {
    return;
  }

  const uint value_width = params.value_heads * kValDim;
  if (params.tokens == 0u) {
    if constexpr (EpilogueMode == 2u) {
      const uint pad_limit = min(params.mp, kChunkSize);
      for (uint pad_t = sg; pad_t < pad_limit; pad_t += 8u) {
        device half *dst =
            out_f16 + ulong(pad_t) * ulong(value_width) + ulong(hv * kValDim);
        dst[lane] = half(0.0h);
        dst[lane + 32u] = half(0.0h);
        dst[lane + 64u] = half(0.0h);
        dst[lane + 96u] = half(0.0h);
      }
    }
    return;
  }

  threadgroup bfloat *tg_work = tg_mem.work; // 8 * 1024 bfloats = 16,384 B
  threadgroup bfloat *my_work = tg_work + sg * (kChunkSize * kValSgDim);
  const uint repeat = params.value_heads / params.key_heads;
  const uint hk = hv / repeat;
  const uint key_width = params.key_heads * kKeyDim;
  const uint dv_sg_base = sg * kValSgDim;
  const uint out_sg_col_base = hv * kValDim + dv_sg_base;
  const uint v_head_col_base = 2u * key_width + hv * kValDim;
  device float *state_sg = state + ulong(out_sg_col_base) * ulong(kKeyDim);

  const bool is_apple10 = gdn_has_apple10_nax_layout();
  const ushort2 lc = nax_apple10_lane_coord(ushort(lane));
  const uint fm = uint(lc.x);
  const uint fn = uint(lc.y);
  const vec<bfloat, 8> kZeroBf8(bfloat(0.0f));

  for (uint c = 0u; c * kChunkSize < params.tokens; ++c) {
    const uint t0 = c * kChunkSize;
    const uint clen = min(kChunkSize, params.tokens - t0);
    const uint n_rblks = (clen + 15u) / 16u;

    device const bfloat *ws_nk =
        qk_ws + (ulong(c) * ulong(params.key_heads) + ulong(hk)) * kQkPairBfloats;
    device const bfloat *ws_nq = ws_nk + ulong(kChunkSize) * ulong(kKeyDim);
    device const uchar *ctrl_raw =
        ctrl_ws + (ulong(c) * ulong(params.value_heads) + ulong(hv)) * kCtrlStrideBytes;
    device const bfloat *ctrl_solve =
        reinterpret_cast<device const bfloat *>(ctrl_raw);
    device const bfloat *ctrl_mqk =
        reinterpret_cast<device const bfloat *>(ctrl_raw + kCtrlMqkByteOff);
    device const float *ctrl_pref =
        reinterpret_cast<device const float *>(ctrl_raw + kCtrlPrefByteOff);
    device const float *ctrl_suff =
        reinterpret_cast<device const float *>(ctrl_raw + kCtrlSuffByteOff);
    const float decay_last =
        reinterpret_cast<device const float *>(ctrl_raw + kCtrlMetaByteOff)[0];

    // Stage V into my_work[dv][r] ([16 x 64] transposed per SG!) via Conv1D+SiLU.
    if constexpr (FuseConv) {
      const uint r_start = sg * 8u;
      const uint ch_local = lane * 4u;
      const uint ch0 = v_head_col_base + ch_local;
      const uint dst_sg = ch_local >> 4u;
      const uint dst_dv = ch_local & 15u;
      if (params.conv_kernel == 4u && (params.conv_width & 3u) == 0u) {
        float4 vw0, vw1, vw2, vw3;
        gdn_load_conv4_weights_vec4(conv_weight, ch0, vw0, vw1, vw2, vw3);
        float4 xm3 = float4(0.0f), xm2 = float4(0.0f), xm1 = float4(0.0f);
        if (r_start < clen) {
          const int base_tok = int(t0 + r_start);
          xm3 = gdn_read_qkv_or_hist_vec4(qkv_or_conv, history, base_tok - 3,
                                          ch0, params.conv_width);
          xm2 = gdn_read_qkv_or_hist_vec4(qkv_or_conv, history, base_tok - 2,
                                          ch0, params.conv_width);
          xm1 = gdn_read_qkv_or_hist_vec4(qkv_or_conv, history, base_tok - 1,
                                          ch0, params.conv_width);
        }
        for (uint s = 0u; s < 8u; ++s) {
          const uint r = r_start + s;
          bfloat4 out_v = bfloat4(bfloat(0.0f));
          if (r < clen) {
            const float4 xcur = float4(*reinterpret_cast<device const bfloat4 *>(
                qkv_or_conv + ulong(t0 + r) * ulong(params.conv_width) + ulong(ch0)));
            float4 sum = fma(xm3, vw0, float4(0.0f));
            sum = fma(xm2, vw1, sum);
            sum = fma(xm1, vw2, sum);
            sum = fma(xcur, vw3, sum);
            out_v = bfloat4(sum / (1.0f + exp(-sum)));
            xm3 = xm2;
            xm2 = xm1;
            xm1 = xcur;
          }
          threadgroup bfloat *dst_sg_buf = tg_work + dst_sg * (kChunkSize * kValSgDim);
          for (uint j = 0u; j < 4u; ++j) {
            dst_sg_buf[(dst_dv + j) * kChunkSize + r] = out_v[j];
          }
        }
      } else {
        for (uint s = 0u; s < 8u; ++s) {
          const uint r = r_start + s;
          threadgroup bfloat *dst_sg_buf = tg_work + dst_sg * (kChunkSize * kValSgDim);
          for (uint j = 0u; j < 4u; ++j) {
            dst_sg_buf[(dst_dv + j) * kChunkSize + r] =
                (r < clen)
                    ? gdn_eval_conv_silu_scalar(qkv_or_conv, conv_weight, history,
                                                t0 + r, ch0 + j,
                                                params.conv_width,
                                                params.conv_kernel)
                    : bfloat(0.0f);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (is_apple10) {
      // 1. rhs = V - pref * (nk @ S_sg^T), overwrite my_work[dv][r] in-place.
      for (uint rb = 0u; rb < 4u; ++rb) {
        const uint r_base = rb * 16u;
        vec<float, 8> pred(0.0f);
        if (rb < n_rblks) {
          for (uint k0 = 0u; k0 < kKeyDim; k0 += 32u) {
            const vec<bfloat, 8> a0 =
                gdn_load_frag_dev_bf16(ws_nk + r_base * kKeyDim + k0, kKeyDim, fm, fn);
            const vec<bfloat, 8> a1 =
                gdn_load_frag_dev_bf16(ws_nk + r_base * kKeyDim + k0 + 16u, kKeyDim, fm, fn);
            const bfloat4 s0_0 = bfloat4(*reinterpret_cast<device const float4 *>(
                state_sg + ulong(fm) * kKeyDim + k0 + fn));
            const bfloat4 s0_1 = bfloat4(*reinterpret_cast<device const float4 *>(
                state_sg + ulong(fm + 8u) * kKeyDim + k0 + fn));
            const bfloat4 s1_0 = bfloat4(*reinterpret_cast<device const float4 *>(
                state_sg + ulong(fm) * kKeyDim + k0 + 16u + fn));
            const bfloat4 s1_1 = bfloat4(*reinterpret_cast<device const float4 *>(
                state_sg + ulong(fm + 8u) * kKeyDim + k0 + 16u + fn));
            const vec<bfloat, 8> b0(s0_0.x, s0_0.y, s0_0.z, s0_0.w,
                                    s0_1.x, s0_1.y, s0_1.z, s0_1.w);
            const vec<bfloat, 8> b1(s1_0.x, s1_0.y, s1_0.z, s1_0.w,
                                    s1_1.x, s1_1.y, s1_1.z, s1_1.w);
            nax_mma_16x16x32<bfloat, true>(pred, a0, a1, b0, b1);
          }
        }
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint r = r_base + fm + h * 8u;
          const float pref = (r < clen) ? ctrl_pref[r] : 0.0f;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            const uint dv = fn + j;
            if (r < clen) {
              const float v_val = float(my_work[dv * kChunkSize + r]);
              my_work[dv * kChunkSize + r] =
                  bfloat(fma(-pref, pred[h * 4u + j], v_val));
            } else {
              my_work[dv * kChunkSize + r] = bfloat(0.0f);
            }
          }
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);

      // 2. delta = solve @ rhs in REVERSE rb=3..0 in-place into my_work[dv][r]!
      for (int rb_i = 3; rb_i >= 0; --rb_i) {
        const uint rb = uint(rb_i);
        const uint r_base = rb * 16u;
        vec<float, 8> delta(0.0f);
        if (rb < n_rblks) {
          for (uint kt = 0u; kt <= rb; kt += 2u) {
            const uint k0 = kt * 16u;
            const vec<bfloat, 8> a0 = gdn_load_frag_dev_bf16(
                ctrl_solve + r_base * kChunkSize + k0, kChunkSize, fm, fn);
            const vec<bfloat, 8> b0 =
                gdn_load_frag_tg_bf16<kChunkSize>(my_work + k0, fm, fn);
            const bool has_k1 = (kt + 1u <= rb);
            const vec<bfloat, 8> a1 =
                has_k1 ? gdn_load_frag_dev_bf16(
                             ctrl_solve + r_base * kChunkSize + k0 + 16u,
                             kChunkSize, fm, fn)
                       : kZeroBf8;
            const vec<bfloat, 8> b1 =
                has_k1 ? gdn_load_frag_tg_bf16<kChunkSize>(my_work + k0 + 16u, fm, fn)
                       : kZeroBf8;
            nax_mma_16x16x32<bfloat, true>(delta, a0, a1, b0, b1);
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint r = r_base + fm + h * 8u;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            my_work[(fn + j) * kChunkSize + r] =
                (r < clen) ? bfloat(delta[h * 4u + j]) : bfloat(0.0f);
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }

      // 3. c_out = pref * (nq @ S_sg^T) + mqk @ delta (in reverse rb=3..0? NO:
      //    we still need delta for S update -> update S FIRST while delta is in my_work,
      //    AFTER computing hist = nq @ S_old^T? Wait: S_old is needed for hist!).
      //    Compute c_out[rb] into out_bf16/f16 staging? Wait: compute S_new?
      //    Even cleaner: compute hist + mqk@delta in REVERSE rb=3..0? Wait: mqk is
      //    lower-triangular! So rb=3..0 reads delta[0..rb] and writes c_out[rb] ->
      //    wait then delta is overwritten before S update!
      //    Instead: write S_new delta_suff^T @ nk BEFORE overwriting delta? NO, S_old is
      //    needed for hist = nq @ S_old^T!
      //    Easy 0-extra-SRAM solution: write hist+mqk@delta directly to `out_bf16` (or
      //    `out_f16` raw bfloat bits) as temporary buffer BEFORE norm_gate!
      //    Check: `out_bf16` / `out_f16` has >= tokens * value_width * 2 bytes!
      device bfloat *tmp_out =
          (EpilogueMode == 1u)
              ? out_bf16
              : reinterpret_cast<device bfloat *>(out_f16);
      for (uint rb = 0u; rb < n_rblks; ++rb) {
        const uint r_base = rb * 16u;
        vec<float, 8> hist(0.0f);
        for (uint k0 = 0u; k0 < kKeyDim; k0 += 32u) {
          const vec<bfloat, 8> a0 =
              gdn_load_frag_dev_bf16(ws_nq + r_base * kKeyDim + k0, kKeyDim, fm, fn);
          const vec<bfloat, 8> a1 =
              gdn_load_frag_dev_bf16(ws_nq + r_base * kKeyDim + k0 + 16u, kKeyDim, fm, fn);
          const bfloat4 s0_0 = bfloat4(*reinterpret_cast<device const float4 *>(
              state_sg + ulong(fm) * kKeyDim + k0 + fn));
          const bfloat4 s0_1 = bfloat4(*reinterpret_cast<device const float4 *>(
              state_sg + ulong(fm + 8u) * kKeyDim + k0 + fn));
          const bfloat4 s1_0 = bfloat4(*reinterpret_cast<device const float4 *>(
              state_sg + ulong(fm) * kKeyDim + k0 + 16u + fn));
          const bfloat4 s1_1 = bfloat4(*reinterpret_cast<device const float4 *>(
              state_sg + ulong(fm + 8u) * kKeyDim + k0 + 16u + fn));
          const vec<bfloat, 8> b0(s0_0.x, s0_0.y, s0_0.z, s0_0.w,
                                  s0_1.x, s0_1.y, s0_1.z, s0_1.w);
          const vec<bfloat, 8> b1(s1_0.x, s1_0.y, s1_0.z, s1_0.w,
                                  s1_1.x, s1_1.y, s1_1.z, s1_1.w);
          nax_mma_16x16x32<bfloat, true>(hist, a0, a1, b0, b1);
        }
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint r = r_base + fm + h * 8u;
          const float pref = (r < clen) ? ctrl_pref[r] : 0.0f;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            hist[h * 4u + j] *= pref;
          }
        }
        for (uint kt = 0u; kt <= rb; kt += 2u) {
          const uint k0 = kt * 16u;
          const vec<bfloat, 8> a0 = gdn_load_frag_dev_bf16(
              ctrl_mqk + r_base * kChunkSize + k0, kChunkSize, fm, fn);
          const vec<bfloat, 8> b0 =
              gdn_load_frag_tg_bf16<kChunkSize>(my_work + k0, fm, fn);
          const bool has_k1 = (kt + 1u <= rb);
          const vec<bfloat, 8> a1 =
              has_k1 ? gdn_load_frag_dev_bf16(
                           ctrl_mqk + r_base * kChunkSize + k0 + 16u,
                           kChunkSize, fm, fn)
                     : kZeroBf8;
          const vec<bfloat, 8> b1 =
              has_k1 ? gdn_load_frag_tg_bf16<kChunkSize>(my_work + k0 + 16u, fm, fn)
                     : kZeroBf8;
          nax_mma_16x16x32<bfloat, true>(hist, a0, a1, b0, b1);
        }
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint r = r_base + fm + h * 8u;
          if (r < clen) {
            *reinterpret_cast<device bfloat4 *>(
                tmp_out + ulong(t0 + r) * ulong(value_width) +
                ulong(out_sg_col_base + fn)) = bfloat4(
                bfloat(hist[h * 4u + 0u]), bfloat(hist[h * 4u + 1u]),
                bfloat(hist[h * 4u + 2u]), bfloat(hist[h * 4u + 3u]));
          }
        }
      }

      // 4. Scale my_work (delta) in-place by ctrl_suff[r] and update state_sg.
      for (uint dv = 0u; dv < kValSgDim; ++dv) {
        const float s0 = (lane < clen) ? ctrl_suff[lane] : 0.0f;
        const float s1 = (32u + lane < clen) ? ctrl_suff[32u + lane] : 0.0f;
        my_work[dv * kChunkSize + lane] =
            bfloat(float(my_work[dv * kChunkSize + lane]) * s0);
        my_work[dv * kChunkSize + 32u + lane] =
            bfloat(float(my_work[dv * kChunkSize + 32u + lane]) * s1);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);

      for (uint k0 = 0u; k0 < kKeyDim; k0 += 32u) {
        vec<float, 8> ms0, ms1;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const float4 f0 = *reinterpret_cast<device const float4 *>(
              state_sg + ulong(fm + h * 8u) * kKeyDim + k0 + fn) * decay_last;
          const float4 f1 = *reinterpret_cast<device const float4 *>(
              state_sg + ulong(fm + h * 8u) * kKeyDim + k0 + 16u + fn) * decay_last;
          ms0[h * 4u + 0u] = f0[0]; ms0[h * 4u + 1u] = f0[1];
          ms0[h * 4u + 2u] = f0[2]; ms0[h * 4u + 3u] = f0[3];
          ms1[h * 4u + 0u] = f1[0]; ms1[h * 4u + 1u] = f1[1];
          ms1[h * 4u + 2u] = f1[2]; ms1[h * 4u + 3u] = f1[3];
        }
        for (uint rb = 0u; rb < n_rblks; ++rb) {
          const uint r_base = rb * 16u;
          const vec<bfloat, 8> a_ds =
              gdn_load_frag_tg_bf16<kChunkSize>(my_work + r_base, fm, fn);
          const vec<bfloat, 8> b0 =
              gdn_load_frag_dev_bf16(ws_nk + r_base * kKeyDim + k0, kKeyDim, fm, fn);
          const vec<bfloat, 8> b1 =
              gdn_load_frag_dev_bf16(ws_nk + r_base * kKeyDim + k0 + 16u, kKeyDim, fm, fn);
          nax_mma_16x32x16_nn<bfloat>(ms0, ms1, a_ds, b0, b1);
        }
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          *reinterpret_cast<device float4 *>(
              state_sg + ulong(fm + h * 8u) * kKeyDim + k0 + fn) =
              float4(ms0[h * 4u + 0u], ms0[h * 4u + 1u],
                     ms0[h * 4u + 2u], ms0[h * 4u + 3u]);
          *reinterpret_cast<device float4 *>(
              state_sg + ulong(fm + h * 8u) * kKeyDim + k0 + 16u + fn) =
              float4(ms1[h * 4u + 0u], ms1[h * 4u + 1u],
                     ms1[h * 4u + 2u], ms1[h * 4u + 3u]);
        }
      }
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

      // 5. In-place RMSNorm_128 * SiLU(z) -> BF16 / FP16 across all 8 SGs.
      const uint f0 = lane, f1 = lane + 32u, f2 = lane + 64u, f3 = lane + 96u;
      const float nw0 = float(norm_weight[f0]), nw1 = float(norm_weight[f1]);
      const float nw2 = float(norm_weight[f2]), nw3 = float(norm_weight[f3]);
      for (uint r = sg; r < clen; r += 8u) {
        const ulong head_row_off =
            ulong(t0 + r) * ulong(value_width) + ulong(hv * kValDim);
        device const bfloat *src_r = tmp_out + head_row_off;
        const float v0 = float(src_r[f0]), v1 = float(src_r[f1]);
        const float v2 = float(src_r[f2]), v3 = float(src_r[f3]);
        float sq = fma(v0, v0, 0.0f);
        sq = fma(v1, v1, sq);
        sq = fma(v2, v2, sq);
        sq = fma(v3, v3, sq);
        const float inv = rsqrt(simd_sum(sq) / 128.0f + params.epsilon);
        device const bfloat *z_row = z + head_row_off;
        const float g0 = float(z_row[f0]), g1 = float(z_row[f1]);
        const float g2 = float(z_row[f2]), g3 = float(z_row[f3]);
        const bfloat o0 = bfloat((v0 * inv * nw0) * g0 / (1.0f + exp(-g0)));
        const bfloat o1 = bfloat((v1 * inv * nw1) * g1 / (1.0f + exp(-g1)));
        const bfloat o2 = bfloat((v2 * inv * nw2) * g2 / (1.0f + exp(-g2)));
        const bfloat o3 = bfloat((v3 * inv * nw3) * g3 / (1.0f + exp(-g3)));
        if constexpr (EpilogueMode == 1u) {
          device bfloat *dst = out_bf16 + head_row_off;
          dst[f0] = o0; dst[f1] = o1; dst[f2] = o2; dst[f3] = o3;
        } else {
          device half *dst = out_f16 + head_row_off;
          dst[f0] = half(clamp(float(o0), -65504.0f, 65504.0f));
          dst[f1] = half(clamp(float(o1), -65504.0f, 65504.0f));
          dst[f2] = half(clamp(float(o2), -65504.0f, 65504.0f));
          dst[f3] = half(clamp(float(o3), -65504.0f, 65504.0f));
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      // Apple9 portable fallback (0 B spills).
      device bfloat *tmp_out =
          (EpilogueMode == 1u) ? out_bf16 : reinterpret_cast<device bfloat *>(out_f16);
      for (uint dv = 0u; dv < kValSgDim; ++dv) {
        device float *s_row = state_sg + ulong(dv) * kKeyDim;
        for (uint r = 0u; r < clen; ++r) {
          float acc = 0.0f;
          for (uint k = lane; k < kKeyDim; k += 32u) {
            acc = fma(float(ws_nk[r * kKeyDim + k]), float(bfloat(s_row[k])), acc);
          }
          const float pred = simd_sum(acc);
          if (lane == 0u) {
            const float v_val = float(my_work[dv * kChunkSize + r]);
            my_work[dv * kChunkSize + r] = bfloat(fma(-ctrl_pref[r], pred, v_val));
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int r_i = int(clen) - 1; r_i >= 0; --r_i) {
          const uint r = uint(r_i);
          float acc = 0.0f;
          for (uint j = lane; j <= r; j += 32u) {
            acc = fma(float(ctrl_solve[r * kChunkSize + j]),
                      float(my_work[dv * kChunkSize + j]), acc);
          }
          const float delta = simd_sum(acc);
          simdgroup_barrier(mem_flags::mem_threadgroup);
          if (lane == 0u) {
            my_work[dv * kChunkSize + r] = bfloat(delta);
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint r = 0u; r < clen; ++r) {
          float acc_h = 0.0f;
          for (uint k = lane; k < kKeyDim; k += 32u) {
            acc_h = fma(float(ws_nq[r * kKeyDim + k]), float(bfloat(s_row[k])), acc_h);
          }
          float h_val = simd_sum(acc_h) * ctrl_pref[r];
          float acc_m = 0.0f;
          for (uint j = lane; j <= r; j += 32u) {
            acc_m = fma(float(ctrl_mqk[r * kChunkSize + j]),
                        float(my_work[dv * kChunkSize + j]), acc_m);
          }
          h_val += simd_sum(acc_m);
          if (lane == 0u) {
            tmp_out[ulong(t0 + r) * ulong(value_width) + ulong(out_sg_col_base + dv)] =
                bfloat(h_val);
          }
        }
        for (uint k = lane; k < kKeyDim; k += 32u) {
          float val = s_row[k] * decay_last;
          for (uint r = 0u; r < clen; ++r) {
            val = fma(float(bfloat(ctrl_suff[r] * float(my_work[dv * kChunkSize + r]))),
                      float(ws_nk[r * kKeyDim + k]), val);
          }
          s_row[k] = val;
        }
      }
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
      const uint f0 = lane, f1 = lane + 32u, f2 = lane + 64u, f3 = lane + 96u;
      const float nw0 = float(norm_weight[f0]), nw1 = float(norm_weight[f1]);
      const float nw2 = float(norm_weight[f2]), nw3 = float(norm_weight[f3]);
      for (uint r = sg; r < clen; r += 8u) {
        const ulong head_row_off =
            ulong(t0 + r) * ulong(value_width) + ulong(hv * kValDim);
        device const bfloat *src_r = tmp_out + head_row_off;
        const float v0 = float(src_r[f0]), v1 = float(src_r[f1]);
        const float v2 = float(src_r[f2]), v3 = float(src_r[f3]);
        float sq = fma(v0, v0, 0.0f);
        sq = fma(v1, v1, sq);
        sq = fma(v2, v2, sq);
        sq = fma(v3, v3, sq);
        const float inv = rsqrt(simd_sum(sq) / 128.0f + params.epsilon);
        device const bfloat *z_row = z + head_row_off;
        const float g0 = float(z_row[f0]), g1 = float(z_row[f1]);
        const float g2 = float(z_row[f2]), g3 = float(z_row[f3]);
        const bfloat o0 = bfloat((v0 * inv * nw0) * g0 / (1.0f + exp(-g0)));
        const bfloat o1 = bfloat((v1 * inv * nw1) * g1 / (1.0f + exp(-g1)));
        const bfloat o2 = bfloat((v2 * inv * nw2) * g2 / (1.0f + exp(-g2)));
        const bfloat o3 = bfloat((v3 * inv * nw3) * g3 / (1.0f + exp(-g3)));
        if constexpr (EpilogueMode == 1u) {
          device bfloat *dst = out_bf16 + head_row_off;
          dst[f0] = o0; dst[f1] = o1; dst[f2] = o2; dst[f3] = o3;
        } else {
          device half *dst = out_f16 + head_row_off;
          dst[f0] = half(clamp(float(o0), -65504.0f, 65504.0f));
          dst[f1] = half(clamp(float(o1), -65504.0f, 65504.0f));
          dst[f2] = half(clamp(float(o2), -65504.0f, 65504.0f));
          dst[f3] = half(clamp(float(o3), -65504.0f, 65504.0f));
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  if constexpr (EpilogueMode == 2u) {
    const uint pad_end =
        min(params.mp, ((params.tokens + kChunkSize - 1u) / kChunkSize) * kChunkSize);
    for (uint pad_t = params.tokens + sg; pad_t < pad_end; pad_t += 8u) {
      device half *dst =
          out_f16 + ulong(pad_t) * ulong(value_width) + ulong(hv * kValDim);
      dst[lane] = half(0.0h);
      dst[lane + 32u] = half(0.0h);
      dst[lane + 64u] = half(0.0h);
      dst[lane + 96u] = half(0.0h);
    }
  }
}

// ============================================================================
// WP3.3 Stage 2 `dv16_flex` Core (`9.00 KiB` SRAM, 0 B spills -> 3 TGs/core!)
// Handles ANY (hv, dv0, dv_valid <= 16): exact K_dv=8 (16) AND K_dv=10 (12..13)!
// ============================================================================
template <bool FuseConv, typename ParamsT>
GDN_NAX_ALWAYS_INLINE
inline void gdn_chunk_recurrence_dv16_task_apple9(
    device const bfloat *qkv_or_conv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    device const bfloat *qk_ws,
    device const uchar *ctrl_ws,
    device float *state,
    device bfloat *out_bf16,
    constant ParamsT &params,
    threadgroup GdnDv16TgStorage &tg_mem,
    uint lane,
    uint sg,
    uint hv,
    uint dv0,
    uint dv_valid) {
  const uint tid = sg * 32u + lane;
  const uint repeat = params.value_heads / params.key_heads;
  const uint hk = hv / repeat;
  const uint key_width = params.key_heads * kKeyDim;
  const uint value_width = params.value_heads * kValDim;
  const uint out_tg_col_base = hv * kValDim + dv0;
  const uint v_tg_col_base = 2u * key_width + out_tg_col_base;
  threadgroup bfloat *tg_bf16 = tg_mem.bf16;
  threadgroup bfloat *tg_rhsT = tg_bf16;
  threadgroup bfloat *tg_V = tg_bf16 + 2304u;
  threadgroup bfloat *tg_deltaT = tg_bf16 + 1152u;

  for (uint c = 0u; c * kChunkSize < params.tokens; ++c) {
    const uint t0 = c * kChunkSize;
    const uint clen = min(kChunkSize, params.tokens - t0);
    device const bfloat *ws_nk =
        qk_ws + (ulong(c) * ulong(params.key_heads) + ulong(hk)) * kQkPairBfloats;
    device const bfloat *ws_nq = ws_nk + ulong(kChunkSize) * ulong(kKeyDim);
    device const uchar *ctrl_raw =
        ctrl_ws + (ulong(c) * ulong(params.value_heads) + ulong(hv)) * kCtrlStrideBytes;
    device const bfloat *ctrl_solve =
        reinterpret_cast<device const bfloat *>(ctrl_raw);
    device const bfloat *ctrl_mqk =
        reinterpret_cast<device const bfloat *>(ctrl_raw + kCtrlMqkByteOff);
    device const float *ctrl_pref =
        reinterpret_cast<device const float *>(ctrl_raw + kCtrlPrefByteOff);
    device const float *ctrl_suff =
        reinterpret_cast<device const float *>(ctrl_raw + kCtrlSuffByteOff);
    const float decay_last =
        reinterpret_cast<device const float *>(ctrl_raw + kCtrlMetaByteOff)[0];

    if constexpr (FuseConv) {
      gdn_stage_conv_silu_v_slice<kDvSlice16>(
          qkv_or_conv, conv_weight, history, t0, clen, v_tg_col_base, dv_valid,
          params.conv_width, params.conv_kernel, tg_V, tid);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint dv_l = sg; dv_l < dv_valid; dv_l += 4u) {
      device float *s_row = state + ulong(out_tg_col_base + dv_l) * kKeyDim;
      for (uint r = 0u; r < clen; ++r) {
        float acc = 0.0f;
        for (uint k = lane; k < kKeyDim; k += 32u) {
          acc = fma(float(ws_nk[r * kKeyDim + k]), float(bfloat(s_row[k])), acc);
        }
        const float pred = simd_sum(acc);
        if (lane == 0u) {
          const float v_val =
              FuseConv
                  ? float(tg_V[r * 16u + dv_l])
                  : float(qkv_or_conv[ulong(t0 + r) * ulong(params.conv_width) +
                                      ulong(v_tg_col_base + dv_l)]);
          tg_rhsT[dv_l * kRhsPad + r] = bfloat(fma(-ctrl_pref[r], pred, v_val));
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint r = 0u; r < clen; ++r) {
        float acc = 0.0f;
        for (uint j = lane; j <= r; j += 32u) {
          acc = fma(float(ctrl_solve[r * kChunkSize + j]),
                    float(tg_rhsT[dv_l * kRhsPad + j]), acc);
        }
        const float d = simd_sum(acc);
        if (lane == 0u) {
          tg_deltaT[dv_l * kRhsPad + r] = bfloat(d);
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint r = 0u; r < clen; ++r) {
        float acc_h = 0.0f;
        for (uint k = lane; k < kKeyDim; k += 32u) {
          acc_h = fma(float(ws_nq[r * kKeyDim + k]), float(bfloat(s_row[k])), acc_h);
        }
        float h_val = simd_sum(acc_h) * ctrl_pref[r];
        float acc_m = 0.0f;
        for (uint j = lane; j <= r; j += 32u) {
          acc_m = fma(float(ctrl_mqk[r * kChunkSize + j]),
                      float(tg_deltaT[dv_l * kRhsPad + j]), acc_m);
        }
        h_val += simd_sum(acc_m);
        if (lane == 0u) {
          out_bf16[ulong(t0 + r) * ulong(value_width) + ulong(out_tg_col_base + dv_l)] =
              bfloat(h_val);
        }
      }
      for (uint k = lane; k < kKeyDim; k += 32u) {
        float val = s_row[k] * decay_last;
        for (uint r = 0u; r < clen; ++r) {
          val = fma(float(bfloat(ctrl_suff[r] * float(tg_deltaT[dv_l * kRhsPad + r]))),
                    float(ws_nk[r * kKeyDim + k]), val);
        }
        s_row[k] = val;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <bool FuseConv, typename ParamsT>
GDN_NAX_ALWAYS_INLINE
inline void gdn_chunk_recurrence_dv16_task(
    device const bfloat *qkv_or_conv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    device const bfloat *qk_ws,
    device const uchar *ctrl_ws,
    device float *state,
    device bfloat *out_bf16,
    constant ParamsT &params,
    threadgroup GdnDv16TgStorage &tg_mem,
    uint lane,
    uint sg,
    uint hv,
    uint dv0,
    uint dv_valid) {
  if (!gdn_has_apple10_nax_layout()) {
    gdn_chunk_recurrence_dv16_task_apple9<FuseConv>(
        qkv_or_conv, conv_weight, history, qk_ws, ctrl_ws, state, out_bf16,
        params, tg_mem, lane, sg, hv, dv0, dv_valid);
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint out_tg_col_base = hv * kValDim + dv0;

  const ushort2 lc = nax_apple10_lane_coord(ushort(lane));
  const uint fm = uint(lc.x);
  const uint fn = uint(lc.y);

  // Persistent register-resident IEEE FP32 state S0, S1 (16 floats/lane total ->
  // ZERO per-chunk device state reads/writes across all chunks!).
  vec<float, 8> S0(0.0f), S1(0.0f);
  #pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    const uint dv_l = fm + h * 8u;
    if (dv_l < dv_valid) {
      const float4 f0 = *reinterpret_cast<device const float4 *>(
          state + ulong(out_tg_col_base + dv_l) * kKeyDim + sg * 32u + fn);
      const float4 f1 = *reinterpret_cast<device const float4 *>(
          state + ulong(out_tg_col_base + dv_l) * kKeyDim + sg * 32u + 16u + fn);
      S0[h * 4u + 0u] = f0[0]; S0[h * 4u + 1u] = f0[1];
      S0[h * 4u + 2u] = f0[2]; S0[h * 4u + 3u] = f0[3];
      S1[h * 4u + 0u] = f1[0]; S1[h * 4u + 1u] = f1[1];
      S1[h * 4u + 2u] = f1[2]; S1[h * 4u + 3u] = f1[3];
    }
  }

  threadgroup float *tg_f32 = tg_mem.f32;
  threadgroup bfloat *tg_bf16 = tg_mem.bf16;
  // Pack loop-invariant scalar metadata (-6 GPRs -> keeps gpr <= 122 for 5 TGs/core = 400 TGs!).
  const uint hv_dv_packed =
      (hv << 8u) | (dv_valid << 1u) | ((((dv0 & 3u) == 0u) && (dv_valid == kDvSlice16)) ? 1u : 0u);

  for (uint c = 0u; c * kChunkSize < params.tokens; ++c) {
    const uint l_hv = hv_dv_packed >> 8u;
    const uint l_dv_valid = (hv_dv_packed >> 1u) & 0x7Fu;
    const bool l_fast16 = (hv_dv_packed & 1u) != 0u;
    const uint l_hk = l_hv / (params.value_heads / params.key_heads);
    const uint l_v_col = 2u * params.key_heads * kKeyDim + out_tg_col_base;
    const uint t0 = c * kChunkSize;
    const uint clen = min(kChunkSize, params.tokens - t0);
    const bool has_hi = (clen > kHalfChunk);

    device const bfloat *ws_nk =
        qk_ws + (ulong(c) * ulong(params.key_heads) + ulong(l_hk)) * kQkPairBfloats;
    device const uchar *ctrl_raw =
        ctrl_ws + (ulong(c) * ulong(params.value_heads) + ulong(l_hv)) * kCtrlStrideBytes;
    device const float *ctrl_pref =
        reinterpret_cast<device const float *>(ctrl_raw + kCtrlPrefByteOff);

    // Stage V into bytes [8192..10239] (disjoint from tg_red [0..8191]!) BEFORE
    // Step 1a — NO barrier needed here because the barrier after nk rh=0 MMAs
    // synchronizes tg_V before sg 0,1 read it (-1 barrier/chunk!).
    threadgroup bfloat *tg_V = tg_bf16 + 4096u;
    if constexpr (FuseConv) {
      gdn_stage_conv_silu_v_slice_fast<kDvSlice16>(
          qkv_or_conv, conv_weight, history, t0, clen, l_v_col, l_dv_valid,
          params.conv_width, params.conv_kernel, tg_V, tid);
      simdgroup_barrier(mem_flags::mem_none);
    }

    // Step 1: Convert register-resident S0/S1 -> BF16 ONCE per chunk, keep peak
    // live regs <= 5-TG/core threshold (400 TGs on 80c -> ZERO cliff on 27B Hv=48!).
    // tg_rhs [64, 16] is ROW-MAJOR and overwrites tg_V [64, 16] IN PLACE per SG
    // (2 coalesced bfloat4 stores, ZERO rhs_bf register vector, ZERO transpose!).
    {
      vec<bfloat, 8> sb0, sb1;
      #pragma clang loop unroll(full)
      for (ushort e = 0u; e < 8u; ++e) {
        sb0[e] = bfloat(S0[e]);
        sb1[e] = bfloat(S1[e]);
      }

      #pragma clang loop unroll(disable)
      for (uint rh = 0u; rh < (has_hi ? 2u : 1u); ++rh) {
        const uint r_off = rh * 32u;
        threadgroup float *my_f32 = tg_f32 + sg * 512u;
        #pragma clang loop unroll(full)
        for (uint rm = 0u; rm < 2u; ++rm) {
          if (rm > 0u) {
            simdgroup_barrier(mem_flags::mem_none);
          }
          vec<float, 8> part(0.0f);
          const vec<bfloat, 8> a0 = gdn_load_frag_dev_bf16(
              ws_nk + (r_off + rm * 16u) * kKeyDim + sg * 32u + 0u, kKeyDim, fm, fn);
          const vec<bfloat, 8> a1 = gdn_load_frag_dev_bf16(
              ws_nk + (r_off + rm * 16u) * kKeyDim + sg * 32u + 16u, kKeyDim, fm, fn);
          nax_mma_16x16x32<bfloat, true>(part, a0, a1, sb0, sb1);
          #pragma clang loop unroll(full)
          for (uint h = 0u; h < 2u; ++h) {
            *reinterpret_cast<threadgroup float4 *>(
                my_f32 + (rm * 16u + fm + h * 8u) * 16u + fn) =
                float4(part[h * 4u + 0u], part[h * 4u + 1u],
                       part[h * 4u + 2u], part[h * 4u + 3u]);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((sg >> 1u) == rh) {
          const uint sub_m = (sg & 1u) * 16u;
          #pragma clang loop unroll(disable)
          for (uint h = 0u; h < 2u; ++h) {
            const uint r = sg * 16u + fm + h * 8u;
            bfloat4 out_r = bfloat4(bfloat(0.0f));
            if (r < clen) {
              const uint idx_flat = (sub_m + fm + h * 8u) * 16u + fn;
              const float4 s0 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + idx_flat);
              const float4 s1 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + 512u + idx_flat);
              const float4 s2 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + 1024u + idx_flat);
              const float4 s3 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + 1536u + idx_flat);
              const float4 tot = (((s0 + s1) + s2) + s3);
              const float pref = ctrl_pref[r];
              bfloat4 v4 = bfloat4(bfloat(0.0f));
              if constexpr (FuseConv) {
                v4 = *reinterpret_cast<const threadgroup bfloat4 *>(tg_V + r * 16u + fn);
              } else if (l_fast16) {
                v4 = *reinterpret_cast<device const bfloat4 *>(
                    qkv_or_conv + ulong(t0 + r) * ulong(params.conv_width) +
                    ulong(l_v_col + fn));
              } else {
                device const bfloat *vp =
                    qkv_or_conv + ulong(t0 + r) * ulong(params.conv_width) +
                    ulong(l_v_col);
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                  v4[j] = (fn + j < l_dv_valid) ? vp[fn + j] : bfloat(0.0f);
                }
              }
              #pragma clang loop unroll(full)
              for (uint j = 0u; j < 4u; ++j) {
                out_r[j] = (fn + j < l_dv_valid)
                               ? bfloat(fma(-pref, tot[j], float(v4[j])))
                               : bfloat(0.0f);
              }
            }
            // Overwrite tg_V [8192..10239] IN PLACE as row-major tg_rhs [64, 16]!
            *reinterpret_cast<threadgroup bfloat4 *>(tg_V + r * 16u + fn) = out_r;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      // Zero rows [32..63] of tg_rhs when !has_hi (clen <= 32) so Step 2 reads 0.
      if (!has_hi && sg >= 2u) {
        *reinterpret_cast<threadgroup bfloat4 *>(tg_V + (sg * 16u + fm) * 16u + fn) =
            bfloat4(bfloat(0.0f));
        *reinterpret_cast<threadgroup bfloat4 *>(tg_V + (sg * 16u + fm + 8u) * 16u + fn) =
            bfloat4(bfloat(0.0f));
      }

      vec<float, 8> H_sg(0.0f);
      device const bfloat *ws_nq = ws_nk + ulong(kChunkSize) * ulong(kKeyDim);
      #pragma clang loop unroll(disable)
      for (uint rh = 0u; rh < (has_hi ? 2u : 1u); ++rh) {
        const uint r_off = rh * 32u;
        threadgroup float *my_f32 = tg_f32 + sg * 512u;
        #pragma clang loop unroll(full)
        for (uint rm = 0u; rm < 2u; ++rm) {
          if (rm > 0u) {
            simdgroup_barrier(mem_flags::mem_none);
          }
          vec<float, 8> part(0.0f);
          const vec<bfloat, 8> a0 = gdn_load_frag_dev_bf16(
              ws_nq + (r_off + rm * 16u) * kKeyDim + sg * 32u + 0u, kKeyDim, fm, fn);
          const vec<bfloat, 8> a1 = gdn_load_frag_dev_bf16(
              ws_nq + (r_off + rm * 16u) * kKeyDim + sg * 32u + 16u, kKeyDim, fm, fn);
          nax_mma_16x16x32<bfloat, true>(part, a0, a1, sb0, sb1);
          #pragma clang loop unroll(full)
          for (uint h = 0u; h < 2u; ++h) {
            *reinterpret_cast<threadgroup float4 *>(
                my_f32 + (rm * 16u + fm + h * 8u) * 16u + fn) =
                float4(part[h * 4u + 0u], part[h * 4u + 1u],
                       part[h * 4u + 2u], part[h * 4u + 3u]);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((sg >> 1u) == rh) {
          const uint sub_m = (sg & 1u) * 16u;
          #pragma clang loop unroll(disable)
          for (uint h = 0u; h < 2u; ++h) {
            const uint r = sg * 16u + fm + h * 8u;
            const float pref = (r < clen) ? ctrl_pref[r] : 0.0f;
            const uint idx_flat = (sub_m + fm + h * 8u) * 16u + fn;
            const float4 s0 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + idx_flat);
            const float4 s1 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + 512u + idx_flat);
            const float4 s2 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + 1024u + idx_flat);
            const float4 s3 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + 1536u + idx_flat);
            const float4 tot = (((s0 + s1) + s2) + s3) * pref;
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) {
              H_sg[h * 4u + j] = (r < clen && fn + j < l_dv_valid) ? tot[j] : 0.0f;
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      // Stash H_sg into bytes [4096..8191] (disjoint from tg_delta [0..2047],
      // tg_deltaSuffT [2048..4095], and tg_rhs [8192..10239]!) so H_sg is 0 GPRs
      // across Step 2 AND Step 2 needs ZERO pre-write barrier (-1 barrier/chunk!).
      threadgroup float4 *tg_H = reinterpret_cast<threadgroup float4 *>(tg_f32 + 1024u);
      tg_H[tid * 2u + 0u] = float4(H_sg[0], H_sg[1], H_sg[2], H_sg[3]);
      tg_H[tid * 2u + 1u] = float4(H_sg[4], H_sg[5], H_sg[6], H_sg[7]);
    }

    // Step 2: Row-major tg_rhs [64, 16] at [8192..10239] via TransposeB=false MMA!
    // Writes row-major tg_delta [64, 16] (2 bfloat4 stores!) + tg_deltaSuffT [16, 64] -> 1 barrier!
    {
      threadgroup bfloat *tg_rhs = tg_bf16 + 4096u;        // bytes [8192 .. 10239], row-major [64, 16]
      threadgroup bfloat *tg_delta = tg_bf16;              // bytes [0 .. 2047],    row-major [64, 16]
      threadgroup bfloat *tg_deltaSuffT = tg_bf16 + 1024u; // bytes [2048 .. 4095], swiz [16, 64]
      device const bfloat *ctrl_solve =
          reinterpret_cast<device const bfloat *>(ctrl_raw);
      device const float *ctrl_suff =
          reinterpret_cast<device const float *>(ctrl_raw + kCtrlSuffByteOff);
      vec<float, 8> D_sg(0.0f);
      if (sg * 16u < clen) {
        {
          const vec<bfloat, 8> a0 = gdn_load_frag_dev_bf16(
              ctrl_solve + (sg * 16u) * kChunkSize + 0u, kChunkSize, fm, fn);
          const vec<bfloat, 8> b0 =
              gdn_load_frag_tg_bf16<16u>(tg_rhs + 0u * 16u, fm, fn);
          const bool has_k1 = (sg >= 1u) && (16u < clen);
          const vec<bfloat, 8> a1 =
              has_k1 ? gdn_load_frag_dev_bf16(
                           ctrl_solve + (sg * 16u) * kChunkSize + 16u,
                           kChunkSize, fm, fn)
                     : vec<bfloat, 8>(bfloat(0.0f));
          const vec<bfloat, 8> b1 =
              has_k1 ? gdn_load_frag_tg_bf16<16u>(tg_rhs + 16u * 16u, fm, fn)
                     : vec<bfloat, 8>(bfloat(0.0f));
          nax_mma_16x16x32<bfloat, false>(D_sg, a0, a1, b0, b1);
        }
        if (sg >= 2u && 32u < clen) {
          simdgroup_barrier(mem_flags::mem_none);
          const vec<bfloat, 8> a0 = gdn_load_frag_dev_bf16(
              ctrl_solve + (sg * 16u) * kChunkSize + 32u, kChunkSize, fm, fn);
          const vec<bfloat, 8> b0 =
              gdn_load_frag_tg_bf16<16u>(tg_rhs + 32u * 16u, fm, fn);
          const bool has_k1 = (sg >= 3u) && (48u < clen);
          const vec<bfloat, 8> a1 =
              has_k1 ? gdn_load_frag_dev_bf16(
                           ctrl_solve + (sg * 16u) * kChunkSize + 48u,
                           kChunkSize, fm, fn)
                     : vec<bfloat, 8>(bfloat(0.0f));
          const vec<bfloat, 8> b1 =
              has_k1 ? gdn_load_frag_tg_bf16<16u>(tg_rhs + 48u * 16u, fm, fn)
                     : vec<bfloat, 8>(bfloat(0.0f));
          nax_mma_16x16x32<bfloat, false>(D_sg, a0, a1, b0, b1);
        }
      }
      #pragma clang loop unroll(disable)
      for (uint h = 0u; h < 2u; ++h) {
        const uint r = sg * 16u + fm + h * 8u;
        const float suff = (r < clen) ? ctrl_suff[r] : 0.0f;
        bfloat4 d_row = bfloat4(bfloat(0.0f));
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
          const uint e = h * 4u + j;
          const uint dv_l = fn + j;
          const bool active = (r < clen) && (dv_l < l_dv_valid);
          d_row[j] = active ? bfloat(D_sg[e]) : bfloat(0.0f);
          tg_deltaSuffT[gdn_swiz64_idx(dv_l, r)] =
              active ? bfloat(suff * D_sg[e]) : bfloat(0.0f);
        }
        *reinterpret_cast<threadgroup bfloat4 *>(tg_delta + r * 16u + fn) = d_row;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Reload H_sg from tg_H [4096..8191], accumulate mqk @ tg_delta (row-major) -> out_bf16!
    if (sg * 16u < clen) {
      const threadgroup float4 *tg_H =
          reinterpret_cast<const threadgroup float4 *>(tg_f32 + 1024u);
      const float4 h0 = tg_H[tid * 2u + 0u];
      const float4 h1 = tg_H[tid * 2u + 1u];
      vec<float, 8> H_sg(h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w);
      device const bfloat *ctrl_mqk =
          reinterpret_cast<device const bfloat *>(ctrl_raw + kCtrlMqkByteOff);
      threadgroup bfloat *tg_delta = tg_bf16;
      {
        const vec<bfloat, 8> a0 = gdn_load_frag_dev_bf16(
            ctrl_mqk + (sg * 16u) * kChunkSize + 0u, kChunkSize, fm, fn);
        const vec<bfloat, 8> b0 =
            gdn_load_frag_tg_bf16<16u>(tg_delta + 0u * 16u, fm, fn);
        const bool has_k1 = (sg >= 1u) && (16u < clen);
        const vec<bfloat, 8> a1 =
            has_k1 ? gdn_load_frag_dev_bf16(
                         ctrl_mqk + (sg * 16u) * kChunkSize + 16u,
                         kChunkSize, fm, fn)
                   : vec<bfloat, 8>(bfloat(0.0f));
        const vec<bfloat, 8> b1 =
            has_k1 ? gdn_load_frag_tg_bf16<16u>(tg_delta + 16u * 16u, fm, fn)
                   : vec<bfloat, 8>(bfloat(0.0f));
        nax_mma_16x16x32<bfloat, false>(H_sg, a0, a1, b0, b1);
      }
      if (sg >= 2u && 32u < clen) {
        simdgroup_barrier(mem_flags::mem_none);
        const vec<bfloat, 8> a0 = gdn_load_frag_dev_bf16(
            ctrl_mqk + (sg * 16u) * kChunkSize + 32u, kChunkSize, fm, fn);
        const vec<bfloat, 8> b0 =
            gdn_load_frag_tg_bf16<16u>(tg_delta + 32u * 16u, fm, fn);
        const bool has_k1 = (sg >= 3u) && (48u < clen);
        const vec<bfloat, 8> a1 =
            has_k1 ? gdn_load_frag_dev_bf16(
                         ctrl_mqk + (sg * 16u) * kChunkSize + 48u,
                         kChunkSize, fm, fn)
                   : vec<bfloat, 8>(bfloat(0.0f));
        const vec<bfloat, 8> b1 =
            has_k1 ? gdn_load_frag_tg_bf16<16u>(tg_delta + 48u * 16u, fm, fn)
                   : vec<bfloat, 8>(bfloat(0.0f));
        nax_mma_16x16x32<bfloat, false>(H_sg, a0, a1, b0, b1);
      }

      const uint value_width = params.value_heads * kValDim;
      #pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const uint r = sg * 16u + fm + h * 8u;
        if (r < clen) {
          if (l_fast16) {
            *reinterpret_cast<device bfloat4 *>(
                out_bf16 + ulong(t0 + r) * ulong(value_width) +
                ulong(out_tg_col_base + fn)) = bfloat4(
                bfloat(H_sg[h * 4u + 0u]), bfloat(H_sg[h * 4u + 1u]),
                bfloat(H_sg[h * 4u + 2u]), bfloat(H_sg[h * 4u + 3u]));
          } else {
            device bfloat *dst =
                out_bf16 + ulong(t0 + r) * ulong(value_width) +
                ulong(out_tg_col_base);
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) {
              if (fn + j < l_dv_valid) {
                dst[fn + j] = bfloat(H_sg[h * 4u + j]);
              }
            }
          }
        }
      }
    }

    // Step 4: Register-resident S = decay_last * S + tg_deltaSuffT @ ws_nk DIRECT
    // via nax_mma_16x32x16_nn (compact non-unrolled loop preserves 5 TGs/core = 400 TGs!).
    const float decay_last =
        reinterpret_cast<device const float *>(ctrl_raw + kCtrlMetaByteOff)[0];
    S0 *= decay_last;
    S1 *= decay_last;
    {
      threadgroup bfloat *tg_deltaSuffT = tg_bf16 + 1024u; // bytes [2048 .. 4095]
      const uint n_rblks = (clen + 15u) >> 4u;
      for (uint rb = 0u; rb < n_rblks; ++rb) {
        const uint r_base = rb * 16u;
        const vec<bfloat, 8> a_ds =
            gdn_load_frag_tg_swiz64(tg_deltaSuffT, 0u, r_base, fm, fn);
        const vec<bfloat, 8> b_nk0 = gdn_load_frag_dev_bf16(
            ws_nk + r_base * kKeyDim + sg * 32u + 0u, kKeyDim, fm, fn);
        const vec<bfloat, 8> b_nk1 = gdn_load_frag_dev_bf16(
            ws_nk + r_base * kKeyDim + sg * 32u + 16u, kKeyDim, fm, fn);
        nax_mma_16x32x16_nn<bfloat>(S0, S1, a_ds, b_nk0, b_nk1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Coalesced float4 writeback of persistent register-resident state S0, S1 ONCE!
  #pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    const uint dv_l = fm + h * 8u;
    if (dv_l < dv_valid) {
      *reinterpret_cast<device float4 *>(
          state + ulong(out_tg_col_base + dv_l) * kKeyDim + sg * 32u + fn) =
          float4(S0[h * 4u + 0u], S0[h * 4u + 1u], S0[h * 4u + 2u], S0[h * 4u + 3u]);
      *reinterpret_cast<device float4 *>(
          state + ulong(out_tg_col_base + dv_l) * kKeyDim + sg * 32u + 16u + fn) =
          float4(S1[h * 4u + 0u], S1[h * 4u + 1u], S1[h * 4u + 2u], S1[h * 4u + 3u]);
    }
  }
}

// ============================================================================
// WP3.3 Stage 2 `dv32_flex` Core (`16.00 KiB` SRAM, 0 B spills -> 2 TGs/core!)
// Handles ANY (hv, dv0, dv_valid <= 32): exact K_dv=4 (32) AND K_dv=5 (25..26)!
// ============================================================================
template <bool FuseConv, typename ParamsT>
GDN_NAX_ALWAYS_INLINE
inline void gdn_chunk_recurrence_dv32_task_apple9(
    device const bfloat *qkv_or_conv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    device const bfloat *qk_ws,
    device const uchar *ctrl_ws,
    device float *state,
    device bfloat *out_bf16,
    constant ParamsT &params,
    threadgroup GdnDv32TgStorage &tg_mem,
    uint lane,
    uint sg,
    uint hv,
    uint dv0,
    uint dv_valid) {
  const uint tid = sg * 32u + lane;
  const uint repeat = params.value_heads / params.key_heads;
  const uint hk = hv / repeat;
  const uint key_width = params.key_heads * kKeyDim;
  const uint value_width = params.value_heads * kValDim;
  const uint out_tg_col_base = hv * kValDim + dv0;
  const uint v_tg_col_base = 2u * key_width + out_tg_col_base;
  threadgroup bfloat *tg_bf16 = tg_mem.bf16;
  threadgroup bfloat *tg_rhsT = tg_bf16;
  threadgroup bfloat *tg_deltaT = tg_bf16 + 2048u;
  threadgroup bfloat *tg_V = tg_bf16 + 4096u;

  for (uint c = 0u; c * kChunkSize < params.tokens; ++c) {
    const uint t0 = c * kChunkSize;
    const uint clen = min(kChunkSize, params.tokens - t0);
    device const bfloat *ws_nk =
        qk_ws + (ulong(c) * ulong(params.key_heads) + ulong(hk)) * kQkPairBfloats;
    device const bfloat *ws_nq = ws_nk + ulong(kChunkSize) * ulong(kKeyDim);
    device const uchar *ctrl_raw =
        ctrl_ws + (ulong(c) * ulong(params.value_heads) + ulong(hv)) * kCtrlStrideBytes;
    device const bfloat *ctrl_solve =
        reinterpret_cast<device const bfloat *>(ctrl_raw);
    device const bfloat *ctrl_mqk =
        reinterpret_cast<device const bfloat *>(ctrl_raw + kCtrlMqkByteOff);
    device const float *ctrl_pref =
        reinterpret_cast<device const float *>(ctrl_raw + kCtrlPrefByteOff);
    device const float *ctrl_suff =
        reinterpret_cast<device const float *>(ctrl_raw + kCtrlSuffByteOff);
    const float decay_last =
        reinterpret_cast<device const float *>(ctrl_raw + kCtrlMetaByteOff)[0];

    if constexpr (FuseConv) {
      gdn_stage_conv_silu_v_slice<kDvSlice32>(
          qkv_or_conv, conv_weight, history, t0, clen, v_tg_col_base, dv_valid,
          params.conv_width, params.conv_kernel, tg_V, tid);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint dv_l = sg; dv_l < dv_valid; dv_l += 4u) {
      device float *s_row = state + ulong(out_tg_col_base + dv_l) * kKeyDim;
      for (uint r = 0u; r < clen; ++r) {
        float acc = 0.0f;
        for (uint k = lane; k < kKeyDim; k += 32u) {
          acc = fma(float(ws_nk[r * kKeyDim + k]), float(bfloat(s_row[k])), acc);
        }
        const float pred = simd_sum(acc);
        if (lane == 0u) {
          const float v_val =
              FuseConv
                  ? float(tg_V[r * 32u + dv_l])
                  : float(qkv_or_conv[ulong(t0 + r) * ulong(params.conv_width) +
                                      ulong(v_tg_col_base + dv_l)]);
          tg_rhsT[dv_l * kRhsPad32 + r] = bfloat(fma(-ctrl_pref[r], pred, v_val));
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint r = 0u; r < clen; ++r) {
        float acc = 0.0f;
        for (uint j = lane; j <= r; j += 32u) {
          acc = fma(float(ctrl_solve[r * kChunkSize + j]),
                    float(tg_rhsT[dv_l * kRhsPad32 + j]), acc);
        }
        const float d = simd_sum(acc);
        if (lane == 0u) {
          tg_deltaT[dv_l * kRhsPad32 + r] = bfloat(d);
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint r = 0u; r < clen; ++r) {
        float acc_h = 0.0f;
        for (uint k = lane; k < kKeyDim; k += 32u) {
          acc_h = fma(float(ws_nq[r * kKeyDim + k]), float(bfloat(s_row[k])), acc_h);
        }
        float h_val = simd_sum(acc_h) * ctrl_pref[r];
        float acc_m = 0.0f;
        for (uint j = lane; j <= r; j += 32u) {
          acc_m = fma(float(ctrl_mqk[r * kChunkSize + j]),
                      float(tg_deltaT[dv_l * kRhsPad32 + j]), acc_m);
        }
        h_val += simd_sum(acc_m);
        if (lane == 0u) {
          out_bf16[ulong(t0 + r) * ulong(value_width) + ulong(out_tg_col_base + dv_l)] =
              bfloat(h_val);
        }
      }
      for (uint k = lane; k < kKeyDim; k += 32u) {
        float val = s_row[k] * decay_last;
        for (uint r = 0u; r < clen; ++r) {
          val = fma(float(bfloat(ctrl_suff[r] * float(tg_deltaT[dv_l * kRhsPad32 + r]))),
                    float(ws_nk[r * kKeyDim + k]), val);
        }
        s_row[k] = val;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <bool FuseConv, typename ParamsT>
GDN_NAX_ALWAYS_INLINE
inline void gdn_chunk_recurrence_dv32_task(
    device const bfloat *qkv_or_conv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    device const bfloat *qk_ws,
    device const uchar *ctrl_ws,
    device float *state,
    device bfloat *out_bf16,
    constant ParamsT &params,
    threadgroup GdnDv32TgStorage &tg_mem,
    uint lane,
    uint sg,
    uint hv,
    uint dv0,
    uint dv_valid) {
  if (!gdn_has_apple10_nax_layout()) {
    gdn_chunk_recurrence_dv32_task_apple9<FuseConv>(
        qkv_or_conv, conv_weight, history, qk_ws, ctrl_ws, state, out_bf16,
        params, tg_mem, lane, sg, hv, dv0, dv_valid);
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint hk = hv / (params.value_heads / params.key_heads);
  const uint out_tg_col_base = hv * kValDim + dv0;
  const uint v_col = 2u * params.key_heads * kKeyDim + out_tg_col_base;

  const ushort2 lc = nax_apple10_lane_coord(ushort(lane));
  const uint fm = uint(lc.x);
  const uint fn = uint(lc.y);

  // Hybrid Register+L1 IEEE FP32 State (S00/S01 in regs, S10/S11 in L1 -> 0 B spills
  // across ALL 6 dv32/dv64 kernels!) + direct ws_nk Step 4 (-4 barriers/chunk!).
  vec<float, 8> S00(0.0f), S01(0.0f);
  #pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    const uint dv_l0 = fm + h * 8u;
    if (dv_l0 < dv_valid) {
      const float4 f0 = *reinterpret_cast<device const float4 *>(
          state + ulong(out_tg_col_base + dv_l0) * kKeyDim + sg * 32u + fn);
      const float4 f1 = *reinterpret_cast<device const float4 *>(
          state + ulong(out_tg_col_base + dv_l0) * kKeyDim + sg * 32u + 16u + fn);
      S00[h * 4u + 0u] = f0[0]; S00[h * 4u + 1u] = f0[1];
      S00[h * 4u + 2u] = f0[2]; S00[h * 4u + 3u] = f0[3];
      S01[h * 4u + 0u] = f1[0]; S01[h * 4u + 1u] = f1[1];
      S01[h * 4u + 2u] = f1[2]; S01[h * 4u + 3u] = f1[3];
    }
  }

  threadgroup float *tg_f32 = tg_mem.f32;
  threadgroup bfloat *tg_bf16 = tg_mem.bf16;
  const vec<bfloat, 8> kZeroBf8(bfloat(0.0f));
  const bool fast_aligned32 = ((dv0 & 3u) == 0u) && (dv_valid == kDvSlice32);

  for (uint c = 0u; c * kChunkSize < params.tokens; ++c) {
    const uint t0 = c * kChunkSize;
    const uint clen = min(kChunkSize, params.tokens - t0);
    const uint n_rblks = (clen + 15u) >> 4u;
    const bool has_hi = (clen > kHalfChunk);

    device const bfloat *ws_nk =
        qk_ws + (ulong(c) * ulong(params.key_heads) + ulong(hk)) * kQkPairBfloats;
    device const uchar *ctrl_raw =
        ctrl_ws + (ulong(c) * ulong(params.value_heads) + ulong(hv)) * kCtrlStrideBytes;
    device const float *ctrl_pref =
        reinterpret_cast<device const float *>(ctrl_raw + kCtrlPrefByteOff);

    {
      threadgroup bfloat *tg_rhsT = tg_bf16;          // bytes [0 .. 4095]
      threadgroup bfloat *tg_V = tg_bf16 + 2048u;     // bytes [4096 .. 8191]
      threadgroup float *tg_red = tg_f32 + 2048u;     // bytes [8192 .. 16383]
      if constexpr (FuseConv) {
        gdn_stage_conv_silu_v_slice_fast<kDvSlice32>(
            qkv_or_conv, conv_weight, history, t0, clen, v_col, dv_valid,
            params.conv_width, params.conv_kernel, tg_V, tid);
      }

      #pragma clang loop unroll(disable)
      for (uint half_v = 0u; half_v < 2u; ++half_v) {
        const uint dv_off = half_v * 16u;
        vec<bfloat, 8> sb0(bfloat(0.0f)), sb1(bfloat(0.0f));
        if (half_v == 0u) {
          #pragma clang loop unroll(full)
          for (ushort e = 0u; e < 8u; ++e) {
            sb0[e] = bfloat(S00[e]);
            sb1[e] = bfloat(S01[e]);
          }
        } else {
          #pragma clang loop unroll(full)
          for (uint h = 0u; h < 2u; ++h) {
            const uint dv_l = 16u + fm + h * 8u;
            if (dv_l < dv_valid) {
              const bfloat4 f0 = bfloat4(*reinterpret_cast<device const float4 *>(
                  state + ulong(out_tg_col_base + dv_l) * kKeyDim + sg * 32u + fn));
              const bfloat4 f1 = bfloat4(*reinterpret_cast<device const float4 *>(
                  state + ulong(out_tg_col_base + dv_l) * kKeyDim + sg * 32u + 16u + fn));
              sb0[h * 4u + 0u] = f0[0]; sb0[h * 4u + 1u] = f0[1];
              sb0[h * 4u + 2u] = f0[2]; sb0[h * 4u + 3u] = f0[3];
              sb1[h * 4u + 0u] = f1[0]; sb1[h * 4u + 1u] = f1[1];
              sb1[h * 4u + 2u] = f1[2]; sb1[h * 4u + 3u] = f1[3];
            }
          }
        }
        #pragma clang loop unroll(disable)
        for (uint rh = 0u; rh < (has_hi ? 2u : 1u); ++rh) {
          const uint r_off = rh * 32u;
          threadgroup float *my_f32 = tg_red + sg * 512u;
          #pragma clang loop unroll(disable)
          for (uint rm = 0u; rm < 2u; ++rm) {
            vec<float, 8> part(0.0f);
            const vec<bfloat, 8> a0 = gdn_load_frag_dev_bf16(
                ws_nk + (r_off + rm * 16u) * kKeyDim + sg * 32u + 0u, kKeyDim, fm, fn);
            const vec<bfloat, 8> a1 = gdn_load_frag_dev_bf16(
                ws_nk + (r_off + rm * 16u) * kKeyDim + sg * 32u + 16u, kKeyDim, fm, fn);
            nax_mma_16x16x32<bfloat, true>(part, a0, a1, sb0, sb1);
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              *reinterpret_cast<threadgroup float4 *>(
                  my_f32 + (rm * 16u + fm + h * 8u) * 16u + fn) =
                  float4(part[h * 4u + 0u], part[h * 4u + 1u],
                         part[h * 4u + 2u], part[h * 4u + 3u]);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if ((sg >> 1u) == rh) {
            const uint sub_m = (sg & 1u) * 16u;
            #pragma clang loop unroll(disable)
            for (uint h = 0u; h < 2u; ++h) {
              const uint r = sg * 16u + fm + h * 8u;
              const uint idx_flat = (sub_m + fm + h * 8u) * 16u + fn;
              const float4 s0 = *reinterpret_cast<const threadgroup float4 *>(tg_red + idx_flat);
              const float4 s1 = *reinterpret_cast<const threadgroup float4 *>(tg_red + 512u + idx_flat);
              const float4 s2 = *reinterpret_cast<const threadgroup float4 *>(tg_red + 1024u + idx_flat);
              const float4 s3 = *reinterpret_cast<const threadgroup float4 *>(tg_red + 1536u + idx_flat);
              const float4 tot = (((s0 + s1) + s2) + s3);
              const float pref = (r < clen) ? ctrl_pref[r] : 0.0f;
              bfloat4 v4 = bfloat4(bfloat(0.0f));
              if (r < clen) {
                if constexpr (FuseConv) {
                  v4 = *reinterpret_cast<const threadgroup bfloat4 *>(tg_V + r * 32u + dv_off + fn);
                } else if (fast_aligned32) {
                  v4 = *reinterpret_cast<device const bfloat4 *>(
                      qkv_or_conv + ulong(t0 + r) * ulong(params.conv_width) +
                      ulong(v_col + dv_off + fn));
                } else {
                  device const bfloat *vp =
                      qkv_or_conv + ulong(t0 + r) * ulong(params.conv_width) +
                      ulong(v_col);
                  #pragma clang loop unroll(full)
                  for (uint j = 0u; j < 4u; ++j) {
                    const uint dv_l = dv_off + fn + j;
                    v4[j] = (dv_l < dv_valid) ? vp[dv_l] : bfloat(0.0f);
                  }
                }
              }
              #pragma clang loop unroll(full)
              for (uint j = 0u; j < 4u; ++j) {
                const uint dv_l = dv_off + fn + j;
                tg_rhsT[gdn_swiz64_idx(dv_l, r)] =
                    (r < clen && dv_l < dv_valid)
                        ? bfloat(fma(-pref, tot[j], float(v4[j])))
                        : bfloat(0.0f);
              }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (!has_hi && (sg >> 1u) == 1u) {
          for (uint h = 0u; h < 2u; ++h) {
            const uint r = sg * 16u + fm + h * 8u;
            for (uint j = 0u; j < 4u; ++j) {
              tg_rhsT[gdn_swiz64_idx(dv_off + fn + j, r)] = bfloat(0.0f);
            }
          }
        }
      }

      device const bfloat *ws_nq = ws_nk + ulong(kChunkSize) * ulong(kKeyDim);
      const uint stash_idx = tid * 2u;
      #pragma clang loop unroll(disable)
      for (uint half_v = 0u; half_v < 2u; ++half_v) {
        const uint dv_off = half_v * 16u;
        vec<bfloat, 8> sb0(bfloat(0.0f)), sb1(bfloat(0.0f));
        if (half_v == 0u) {
          #pragma clang loop unroll(full)
          for (ushort e = 0u; e < 8u; ++e) {
            sb0[e] = bfloat(S00[e]);
            sb1[e] = bfloat(S01[e]);
          }
        } else {
          #pragma clang loop unroll(full)
          for (uint h = 0u; h < 2u; ++h) {
            const uint dv_l = 16u + fm + h * 8u;
            if (dv_l < dv_valid) {
              const bfloat4 f0 = bfloat4(*reinterpret_cast<device const float4 *>(
                  state + ulong(out_tg_col_base + dv_l) * kKeyDim + sg * 32u + fn));
              const bfloat4 f1 = bfloat4(*reinterpret_cast<device const float4 *>(
                  state + ulong(out_tg_col_base + dv_l) * kKeyDim + sg * 32u + 16u + fn));
              sb0[h * 4u + 0u] = f0[0]; sb0[h * 4u + 1u] = f0[1];
              sb0[h * 4u + 2u] = f0[2]; sb0[h * 4u + 3u] = f0[3];
              sb1[h * 4u + 0u] = f1[0]; sb1[h * 4u + 1u] = f1[1];
              sb1[h * 4u + 2u] = f1[2]; sb1[h * 4u + 3u] = f1[3];
            }
          }
        }
        threadgroup float4 *tg_H =
            reinterpret_cast<threadgroup float4 *>(tg_f32 + 1024u + half_v * 1024u);
        if (half_v == 0u) {
          if (!has_hi && (sg >> 1u) == 1u) {
            tg_H[stash_idx + 0u] = float4(0.0f);
            tg_H[stash_idx + 1u] = float4(0.0f);
          }
          #pragma clang loop unroll(disable)
          for (uint rh = 0u; rh < (has_hi ? 2u : 1u); ++rh) {
            const uint r_off = rh * 32u;
            threadgroup float *my_f32 = tg_red + sg * 512u;
            #pragma clang loop unroll(disable)
            for (uint rm = 0u; rm < 2u; ++rm) {
              vec<float, 8> part(0.0f);
              const vec<bfloat, 8> a0 = gdn_load_frag_dev_bf16(
                  ws_nq + (r_off + rm * 16u) * kKeyDim + sg * 32u + 0u, kKeyDim, fm, fn);
              const vec<bfloat, 8> a1 = gdn_load_frag_dev_bf16(
                  ws_nq + (r_off + rm * 16u) * kKeyDim + sg * 32u + 16u, kKeyDim, fm, fn);
              nax_mma_16x16x32<bfloat, true>(part, a0, a1, sb0, sb1);
              for (uint h = 0u; h < 2u; ++h) {
                *reinterpret_cast<threadgroup float4 *>(
                    my_f32 + (rm * 16u + fm + h * 8u) * 16u + fn) =
                    float4(part[h * 4u + 0u], part[h * 4u + 1u],
                           part[h * 4u + 2u], part[h * 4u + 3u]);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if ((sg >> 1u) == rh) {
              const uint sub_m = (sg & 1u) * 16u;
              for (uint h = 0u; h < 2u; ++h) {
                const uint r = sg * 16u + fm + h * 8u;
                const float pref = (r < clen) ? ctrl_pref[r] : 0.0f;
                const uint idx_flat = (sub_m + fm + h * 8u) * 16u + fn;
                const float4 s0 = *reinterpret_cast<const threadgroup float4 *>(tg_red + idx_flat);
                const float4 s1 = *reinterpret_cast<const threadgroup float4 *>(tg_red + 512u + idx_flat);
                const float4 s2 = *reinterpret_cast<const threadgroup float4 *>(tg_red + 1024u + idx_flat);
                const float4 s3 = *reinterpret_cast<const threadgroup float4 *>(tg_red + 1536u + idx_flat);
                const float4 tot = (((s0 + s1) + s2) + s3) * pref;
                float4 h4;
                for (uint j = 0u; j < 4u; ++j) {
                  h4[j] = (r < clen && (fn + j) < dv_valid) ? tot[j] : 0.0f;
                }
                tg_H[stash_idx + h] = h4;
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
        } else {
          vec<float, 8> H_h(0.0f);
          #pragma clang loop unroll(disable)
          for (uint rh = 0u; rh < (has_hi ? 2u : 1u); ++rh) {
            const uint r_off = rh * 32u;
            threadgroup float *my_f32 = tg_red + sg * 512u;
            #pragma clang loop unroll(disable)
            for (uint rm = 0u; rm < 2u; ++rm) {
              vec<float, 8> part(0.0f);
              const vec<bfloat, 8> a0 = gdn_load_frag_dev_bf16(
                  ws_nq + (r_off + rm * 16u) * kKeyDim + sg * 32u + 0u, kKeyDim, fm, fn);
              const vec<bfloat, 8> a1 = gdn_load_frag_dev_bf16(
                  ws_nq + (r_off + rm * 16u) * kKeyDim + sg * 32u + 16u, kKeyDim, fm, fn);
              nax_mma_16x16x32<bfloat, true>(part, a0, a1, sb0, sb1);
              for (uint h = 0u; h < 2u; ++h) {
                *reinterpret_cast<threadgroup float4 *>(
                    my_f32 + (rm * 16u + fm + h * 8u) * 16u + fn) =
                    float4(part[h * 4u + 0u], part[h * 4u + 1u],
                           part[h * 4u + 2u], part[h * 4u + 3u]);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if ((sg >> 1u) == rh) {
              const uint sub_m = (sg & 1u) * 16u;
              for (uint h = 0u; h < 2u; ++h) {
                const uint r = sg * 16u + fm + h * 8u;
                const float pref = (r < clen) ? ctrl_pref[r] : 0.0f;
                const uint idx_flat = (sub_m + fm + h * 8u) * 16u + fn;
                const float4 s0 = *reinterpret_cast<const threadgroup float4 *>(tg_red + idx_flat);
                const float4 s1 = *reinterpret_cast<const threadgroup float4 *>(tg_red + 512u + idx_flat);
                const float4 s2 = *reinterpret_cast<const threadgroup float4 *>(tg_red + 1024u + idx_flat);
                const float4 s3 = *reinterpret_cast<const threadgroup float4 *>(tg_red + 1536u + idx_flat);
                const float4 tot = (((s0 + s1) + s2) + s3) * pref;
                for (uint j = 0u; j < 4u; ++j) {
                  H_h[h * 4u + j] = (r < clen && (dv_off + fn + j) < dv_valid) ? tot[j] : 0.0f;
                }
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
          tg_H[stash_idx + 0u] = float4(H_h[0], H_h[1], H_h[2], H_h[3]);
          tg_H[stash_idx + 1u] = float4(H_h[4], H_h[5], H_h[6], H_h[7]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
      device const bfloat *ctrl_solve =
          reinterpret_cast<device const bfloat *>(ctrl_raw);
      device const float *ctrl_suff =
          reinterpret_cast<device const float *>(ctrl_raw + kCtrlSuffByteOff);
      threadgroup bfloat *tg_rhs_delta = tg_bf16;          // bytes [0 .. 4095]
      threadgroup bfloat *tg_deltaSuffT = tg_bf16 + 6144u; // bytes [12288 .. 16383]
      #pragma clang loop unroll(disable)
      for (uint half_v = 0u; half_v < 2u; ++half_v) {
        const uint dv_off = half_v * 16u;
        vec<float, 8> D_h(0.0f);
        if (sg * 16u < clen) {
          for (uint kt = 0u; kt <= sg; kt += 2u) {
            const uint k0 = kt * 16u;
            if (k0 >= clen) {
              break;
            }
            const vec<bfloat, 8> a0 = gdn_load_frag_dev_bf16(
                ctrl_solve + (sg * 16u) * kChunkSize + k0, kChunkSize, fm, fn);
            const vec<bfloat, 8> b0 =
                gdn_load_frag_tg_swiz64(tg_rhs_delta, dv_off, k0, fm, fn);
            const bool has_k1 = (kt + 1u <= sg) && ((k0 + 16u) < clen);
            const vec<bfloat, 8> a1 =
                has_k1 ? gdn_load_frag_dev_bf16(
                             ctrl_solve + (sg * 16u) * kChunkSize + k0 + 16u,
                             kChunkSize, fm, fn)
                       : kZeroBf8;
            const vec<bfloat, 8> b1 =
                has_k1 ? gdn_load_frag_tg_swiz64(tg_rhs_delta, dv_off, k0 + 16u, fm, fn)
                       : kZeroBf8;
            nax_mma_16x16x32<bfloat, true>(D_h, a0, a1, b0, b1);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(disable)
        for (uint h = 0u; h < 2u; ++h) {
          const uint r = sg * 16u + fm + h * 8u;
          const float suff = (r < clen) ? ctrl_suff[r] : 0.0f;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            const uint e = h * 4u + j;
            const uint dv_l = dv_off + fn + j;
            const bool act = (r < clen) && (dv_l < dv_valid);
            const uint idx = gdn_swiz64_idx(dv_l, r);
            tg_rhs_delta[idx] = act ? bfloat(D_h[e]) : bfloat(0.0f);
            tg_deltaSuffT[idx] = act ? bfloat(suff * D_h[e]) : bfloat(0.0f);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    if (sg * 16u < clen) {
      device const bfloat *ctrl_mqk =
          reinterpret_cast<device const bfloat *>(ctrl_raw + kCtrlMqkByteOff);
      threadgroup bfloat *tg_deltaT = tg_bf16;
      const uint stash_idx = tid * 2u;
      const uint value_width = params.value_heads * kValDim;

      #pragma clang loop unroll(disable)
      for (uint half_v = 0u; half_v < 2u; ++half_v) {
        const uint dv_off = half_v * 16u;
        const threadgroup float4 *tg_H =
            reinterpret_cast<const threadgroup float4 *>(
                tg_f32 + 1024u + half_v * 1024u);
        const float4 h0 = tg_H[stash_idx + 0u];
        const float4 h1 = tg_H[stash_idx + 1u];
        vec<float, 8> H_h(h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w);

        for (uint kt = 0u; kt <= sg; kt += 2u) {
          const uint k0 = kt * 16u;
          if (k0 >= clen) {
            break;
          }
          const vec<bfloat, 8> a0 = gdn_load_frag_dev_bf16(
              ctrl_mqk + (sg * 16u) * kChunkSize + k0, kChunkSize, fm, fn);
          const vec<bfloat, 8> b0 =
              gdn_load_frag_tg_swiz64(tg_deltaT, dv_off, k0, fm, fn);
          const bool has_k1 = (kt + 1u <= sg) && ((k0 + 16u) < clen);
          const vec<bfloat, 8> a1 =
              has_k1 ? gdn_load_frag_dev_bf16(
                           ctrl_mqk + (sg * 16u) * kChunkSize + k0 + 16u,
                           kChunkSize, fm, fn)
                     : kZeroBf8;
          const vec<bfloat, 8> b1 =
              has_k1 ? gdn_load_frag_tg_swiz64(tg_deltaT, dv_off, k0 + 16u, fm, fn)
                     : kZeroBf8;
          nax_mma_16x16x32<bfloat, true>(H_h, a0, a1, b0, b1);
        }

        #pragma clang loop unroll(disable)
        for (uint h = 0u; h < 2u; ++h) {
          const uint r = sg * 16u + fm + h * 8u;
          if (r < clen) {
            if (fast_aligned32) {
              *reinterpret_cast<device bfloat4 *>(
                  out_bf16 + ulong(t0 + r) * ulong(value_width) +
                  ulong(out_tg_col_base + dv_off + fn)) = bfloat4(
                  bfloat(H_h[h * 4u + 0u]), bfloat(H_h[h * 4u + 1u]),
                  bfloat(H_h[h * 4u + 2u]), bfloat(H_h[h * 4u + 3u]));
            } else {
              device bfloat *dst =
                  out_bf16 + ulong(t0 + r) * ulong(value_width) +
                  ulong(out_tg_col_base);
              #pragma clang loop unroll(full)
              for (uint j = 0u; j < 4u; ++j) {
                const uint dv_l = dv_off + fn + j;
                if (dv_l < dv_valid) {
                  dst[dv_l] = bfloat(H_h[h * 4u + j]);
                }
              }
            }
          }
        }
      }
    }

    // Step 4: Update register-resident S00/S01 AND L1-resident S10/S11 via direct ws_nk!
    const float decay_last =
        reinterpret_cast<device const float *>(ctrl_raw + kCtrlMetaByteOff)[0];
    S00 *= decay_last;
    S01 *= decay_last;
    vec<float, 8> S10(0.0f), S11(0.0f);
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint dv_l1 = 16u + fm + h * 8u;
      if (dv_l1 < dv_valid) {
        const float4 f0 = *reinterpret_cast<device const float4 *>(
            state + ulong(out_tg_col_base + dv_l1) * kKeyDim + sg * 32u + fn) *
            decay_last;
        const float4 f1 = *reinterpret_cast<device const float4 *>(
            state + ulong(out_tg_col_base + dv_l1) * kKeyDim + sg * 32u + 16u + fn) *
            decay_last;
        S10[h * 4u + 0u] = f0[0]; S10[h * 4u + 1u] = f0[1];
        S10[h * 4u + 2u] = f0[2]; S10[h * 4u + 3u] = f0[3];
        S11[h * 4u + 0u] = f1[0]; S11[h * 4u + 1u] = f1[1];
        S11[h * 4u + 2u] = f1[2]; S11[h * 4u + 3u] = f1[3];
      }
    }
    {
      threadgroup bfloat *tg_deltaSuffT = tg_bf16 + 6144u; // bytes [12288 .. 16383]
      for (uint rb = 0u; rb < n_rblks; ++rb) {
        const uint r_base = rb * 16u;
        const vec<bfloat, 8> b_nk0 = gdn_load_frag_dev_bf16(
            ws_nk + r_base * kKeyDim + sg * 32u + 0u, kKeyDim, fm, fn);
        const vec<bfloat, 8> b_nk1 = gdn_load_frag_dev_bf16(
            ws_nk + r_base * kKeyDim + sg * 32u + 16u, kKeyDim, fm, fn);
        const vec<bfloat, 8> a_ds0 =
            gdn_load_frag_tg_swiz64(tg_deltaSuffT, 0u, r_base, fm, fn);
        nax_mma_16x32x16_nn<bfloat>(S00, S01, a_ds0, b_nk0, b_nk1);
        const vec<bfloat, 8> a_ds1 =
            gdn_load_frag_tg_swiz64(tg_deltaSuffT, 16u, r_base, fm, fn);
        nax_mma_16x32x16_nn<bfloat>(S10, S11, a_ds1, b_nk0, b_nk1);
      }
    }
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint dv_l1 = 16u + fm + h * 8u;
      if (dv_l1 < dv_valid) {
        *reinterpret_cast<device float4 *>(
            state + ulong(out_tg_col_base + dv_l1) * kKeyDim + sg * 32u + fn) =
            float4(S10[h * 4u + 0u], S10[h * 4u + 1u], S10[h * 4u + 2u], S10[h * 4u + 3u]);
        *reinterpret_cast<device float4 *>(
            state + ulong(out_tg_col_base + dv_l1) * kKeyDim + sg * 32u + 16u + fn) =
            float4(S11[h * 4u + 0u], S11[h * 4u + 1u], S11[h * 4u + 2u], S11[h * 4u + 3u]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Coalesced float4 writeback of persistent register-resident state S00/S01 ONCE!
  #pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    const uint dv_l0 = fm + h * 8u;
    if (dv_l0 < dv_valid) {
      *reinterpret_cast<device float4 *>(
          state + ulong(out_tg_col_base + dv_l0) * kKeyDim + sg * 32u + fn) =
          float4(S00[h * 4u + 0u], S00[h * 4u + 1u], S00[h * 4u + 2u], S00[h * 4u + 3u]);
      *reinterpret_cast<device float4 *>(
          state + ulong(out_tg_col_base + dv_l0) * kKeyDim + sg * 32u + 16u + fn) =
          float4(S01[h * 4u + 0u], S01[h * 4u + 1u], S01[h * 4u + 2u], S01[h * 4u + 3u]);
    }
  }
}

template <bool FuseConv>
GDN_NAX_ALWAYS_INLINE
inline void gdn_chunk_recurrence_dv16_flex_impl(
    device const bfloat *qkv_or_conv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    device const bfloat *qk_ws,
    device const uchar *ctrl_ws,
    device float *state,
    device bfloat *out_bf16,
    constant GdnChunkFlexParams &params,
    threadgroup GdnDv16TgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 tgs_per_grid) {
  const uint k_dv = params.slices_per_head;
  if (sg >= 4u || params.key_dim != kKeyDim || params.value_dim != kValDim ||
      params.tokens == 0u || k_dv < 8u || k_dv > 16u ||
      params.total_tasks == 0u) {
    return;
  }
  const uint stride = max(1u, tgs_per_grid.x);
  for (uint task = tg.x; task < params.total_tasks; task += stride) {
    const uint hv = task / k_dv;
    if (hv >= params.value_heads) {
      break;
    }
    const uint s = task - hv * k_dv;
    const uint dv0 = (s * kValDim) / k_dv;
    const uint dv1 = ((s + 1u) * kValDim) / k_dv;
    const uint dv_valid = dv1 - dv0;
    if (task != tg.x) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    gdn_chunk_recurrence_dv16_task<FuseConv>(
        qkv_or_conv, conv_weight, history, qk_ws, ctrl_ws, state, out_bf16,
        params, tg_mem, lane, sg, hv, dv0, dv_valid);
  }
}

template <bool FuseConv>
GDN_NAX_ALWAYS_INLINE
inline void gdn_chunk_recurrence_dv32_flex_impl(
    device const bfloat *qkv_or_conv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    device const bfloat *qk_ws,
    device const uchar *ctrl_ws,
    device float *state,
    device bfloat *out_bf16,
    constant GdnChunkFlexParams &params,
    threadgroup GdnDv32TgStorage &tg_mem,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 tgs_per_grid) {
  const uint k_dv = params.slices_per_head;
  if (sg >= 4u || params.key_dim != kKeyDim || params.value_dim != kValDim ||
      params.tokens == 0u || k_dv < 4u || k_dv > 16u ||
      params.total_tasks == 0u) {
    return;
  }
  const uint stride = max(1u, tgs_per_grid.x);
  for (uint task = tg.x; task < params.total_tasks; task += stride) {
    const uint hv = task / k_dv;
    if (hv >= params.value_heads) {
      break;
    }
    const uint s = task - hv * k_dv;
    const uint dv0 = (s * kValDim) / k_dv;
    const uint dv1 = ((s + 1u) * kValDim) / k_dv;
    const uint dv_valid = dv1 - dv0;
    if (task != tg.x) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    gdn_chunk_recurrence_dv32_task<FuseConv>(
        qkv_or_conv, conv_weight, history, qk_ws, ctrl_ws, state, out_bf16,
        params, tg_mem, lane, sg, hv, dv0, dv_valid);
  }
}

} // namespace

// ============================================================================
// Exported Kernels
// ============================================================================

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_chunk_prepare_c64(
    device const bfloat *convolved [[buffer(0)]],
    device const bfloat *a [[buffer(1)]],
    device const bfloat *b [[buffer(2)]],
    device const bfloat *a_log [[buffer(3)]],
    device const bfloat *dt_bias [[buffer(4)]],
    device bfloat *qk_ws [[buffer(5)]],
    device uchar *ctrl_ws [[buffer(6)]],
    constant GdnChunkLayerParams &params [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup GdnPrepareTgStorage tg_mem;
  gdn_chunk_prepare_c64_impl<false>(
      convolved, nullptr, nullptr, a, b, a_log, dt_bias, qk_ws, ctrl_ws, params,
      tg_mem, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_chunk_prepare_conv_c64(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device const bfloat *history [[buffer(2)]],
    device const bfloat *a [[buffer(3)]],
    device const bfloat *b [[buffer(4)]],
    device const bfloat *a_log [[buffer(5)]],
    device const bfloat *dt_bias [[buffer(6)]],
    device bfloat *qk_ws [[buffer(7)]],
    device uchar *ctrl_ws [[buffer(8)]],
    constant GdnChunkMegaParams &params [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup GdnPrepareTgStorage tg_mem;
  gdn_chunk_prepare_c64_impl<true>(
      qkv, conv_weight, history, a, b, a_log, dt_bias, qk_ws, ctrl_ws, params,
      tg_mem, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_chunk_recurrence_dv16_flex_c64(
    device const bfloat *convolved [[buffer(0)]],
    device const bfloat *qk_ws [[buffer(1)]],
    device const uchar *ctrl_ws [[buffer(2)]],
    device float *state [[buffer(3)]],
    device bfloat *recurrent [[buffer(4)]],
    constant GdnChunkFlexParams &params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tgs_per_grid [[threadgroups_per_grid]]) {
  alignas(16) threadgroup GdnDv16TgStorage tg_mem;
  gdn_chunk_recurrence_dv16_flex_impl<false>(
      convolved, nullptr, nullptr, qk_ws, ctrl_ws, state, recurrent, params,
      tg_mem, lane, sg, tg, tgs_per_grid);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_chunk_recurrence_conv_dv16_flex_c64(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device const bfloat *history [[buffer(2)]],
    device const bfloat *qk_ws [[buffer(3)]],
    device const uchar *ctrl_ws [[buffer(4)]],
    device float *state [[buffer(5)]],
    device bfloat *recurrent_out [[buffer(6)]],
    constant GdnChunkFlexParams &params [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tgs_per_grid [[threadgroups_per_grid]]) {
  alignas(16) threadgroup GdnDv16TgStorage tg_mem;
  gdn_chunk_recurrence_dv16_flex_impl<true>(
      qkv, conv_weight, history, qk_ws, ctrl_ws, state, recurrent_out, params,
      tg_mem, lane, sg, tg, tgs_per_grid);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_chunk_recurrence_dv32_flex_c64(
    device const bfloat *convolved [[buffer(0)]],
    device const bfloat *qk_ws [[buffer(1)]],
    device const uchar *ctrl_ws [[buffer(2)]],
    device float *state [[buffer(3)]],
    device bfloat *recurrent [[buffer(4)]],
    constant GdnChunkFlexParams &params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tgs_per_grid [[threadgroups_per_grid]]) {
  alignas(16) threadgroup GdnDv32TgStorage tg_mem;
  gdn_chunk_recurrence_dv32_flex_impl<false>(
      convolved, nullptr, nullptr, qk_ws, ctrl_ws, state, recurrent, params,
      tg_mem, lane, sg, tg, tgs_per_grid);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_chunk_recurrence_conv_dv32_flex_c64(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device const bfloat *history [[buffer(2)]],
    device const bfloat *qk_ws [[buffer(3)]],
    device const uchar *ctrl_ws [[buffer(4)]],
    device float *state [[buffer(5)]],
    device bfloat *recurrent_out [[buffer(6)]],
    constant GdnChunkFlexParams &params [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tgs_per_grid [[threadgroups_per_grid]]) {
  alignas(16) threadgroup GdnDv32TgStorage tg_mem;
  gdn_chunk_recurrence_dv32_flex_impl<true>(
      qkv, conv_weight, history, qk_ws, ctrl_ws, state, recurrent_out, params,
      tg_mem, lane, sg, tg, tgs_per_grid);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_chunk_recurrence_c64(
    device const bfloat *convolved [[buffer(0)]],
    device const bfloat *qk_ws [[buffer(1)]],
    device const uchar *ctrl_ws [[buffer(2)]],
    device float *state [[buffer(3)]],
    device bfloat *recurrent [[buffer(4)]],
    constant GdnChunkLayerParams &params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if (tg.x >= params.value_heads || tg.y >= 4u || sg >= 4u ||
      params.key_dim != kKeyDim || params.value_dim != kValDim ||
      params.tokens == 0u) {
    return;
  }
  alignas(16) threadgroup GdnDv32TgStorage tg_mem;
  gdn_chunk_recurrence_dv32_task<false>(
      convolved, nullptr, nullptr, qk_ws, ctrl_ws, state, recurrent, params,
      tg_mem, lane, sg, tg.x, tg.y * kDvSlice32, kDvSlice32);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_chunk_recurrence_conv_dv32_c64(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device const bfloat *history [[buffer(2)]],
    device const bfloat *qk_ws [[buffer(3)]],
    device const uchar *ctrl_ws [[buffer(4)]],
    device float *state [[buffer(5)]],
    device bfloat *recurrent_out [[buffer(6)]],
    constant GdnChunkMegaParams &params [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if (tg.x >= params.value_heads || tg.y >= 4u || sg >= 4u ||
      params.key_dim != kKeyDim || params.value_dim != kValDim ||
      params.tokens == 0u) {
    return;
  }
  alignas(16) threadgroup GdnDv32TgStorage tg_mem;
  gdn_chunk_recurrence_dv32_task<true>(
      qkv, conv_weight, history, qk_ws, ctrl_ws, state, recurrent_out, params,
      tg_mem, lane, sg, tg.x, tg.y * kDvSlice32, kDvSlice32);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_chunk_recurrence_dv64_c64(
    device const bfloat *convolved [[buffer(0)]],
    device const bfloat *qk_ws [[buffer(1)]],
    device const uchar *ctrl_ws [[buffer(2)]],
    device float *state [[buffer(3)]],
    device bfloat *recurrent [[buffer(4)]],
    constant GdnChunkLayerParams &params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if (tg.x >= params.value_heads || tg.y >= 2u || sg >= 4u ||
      params.key_dim != kKeyDim || params.value_dim != kValDim ||
      params.tokens == 0u) {
    return;
  }
  alignas(16) threadgroup GdnDv32TgStorage tg_mem;
  #pragma clang loop unroll(disable)
  for (uint half_s = 0u; half_s < 2u; ++half_s) {
    if (half_s > 0u) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    gdn_chunk_recurrence_dv32_task<false>(
        convolved, nullptr, nullptr, qk_ws, ctrl_ws, state, recurrent, params,
        tg_mem, lane, sg, tg.x, (tg.y * 2u + half_s) * kDvSlice32, kDvSlice32);
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_chunk_recurrence_conv_dv64_c64(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device const bfloat *history [[buffer(2)]],
    device const bfloat *qk_ws [[buffer(3)]],
    device const uchar *ctrl_ws [[buffer(4)]],
    device float *state [[buffer(5)]],
    device bfloat *recurrent_out [[buffer(6)]],
    constant GdnChunkMegaParams &params [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if (tg.x >= params.value_heads || tg.y >= 2u || sg >= 4u ||
      params.key_dim != kKeyDim || params.value_dim != kValDim ||
      params.tokens == 0u) {
    return;
  }
  alignas(16) threadgroup GdnDv32TgStorage tg_mem;
  #pragma clang loop unroll(disable)
  for (uint half_s = 0u; half_s < 2u; ++half_s) {
    if (half_s > 0u) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    gdn_chunk_recurrence_dv32_task<true>(
        qkv, conv_weight, history, qk_ws, ctrl_ws, state, recurrent_out, params,
        tg_mem, lane, sg, tg.x, (tg.y * 2u + half_s) * kDvSlice32, kDvSlice32);
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_chunk_norm_gate_inplace_bf16_c64(
    device bfloat *inout_bf16 [[buffer(0)]],
    device const bfloat *z [[buffer(1)]],
    device const bfloat *norm_weight [[buffer(2)]],
    constant GdnChunkMegaParams &params [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if (sg >= 4u || params.value_dim != kValDim) {
    return;
  }
  const uint logical_head = tg.y * 4u + sg;
  const uint token = logical_head / params.value_heads;
  if (token >= params.tokens) {
    return;
  }
  const uint hv = logical_head - token * params.value_heads;
  const ulong head_off =
      (ulong(token) * ulong(params.value_heads) + ulong(hv)) * ulong(kValDim);
  device bfloat *row = inout_bf16 + head_off;
  device const bfloat *gate = z + head_off;

  const uint f0 = lane, f1 = lane + 32u, f2 = lane + 64u, f3 = lane + 96u;
  const float v0 = float(row[f0]), v1 = float(row[f1]);
  const float v2 = float(row[f2]), v3 = float(row[f3]);
  float sq = fma(v0, v0, 0.0f);
  sq = fma(v1, v1, sq);
  sq = fma(v2, v2, sq);
  sq = fma(v3, v3, sq);
  const float inv = rsqrt(simd_sum(sq) / 128.0f + params.epsilon);

  const float g0 = float(gate[f0]), g1 = float(gate[f1]);
  const float g2 = float(gate[f2]), g3 = float(gate[f3]);
  const float nw0 = float(norm_weight[f0]), nw1 = float(norm_weight[f1]);
  const float nw2 = float(norm_weight[f2]), nw3 = float(norm_weight[f3]);

  row[f0] = bfloat((v0 * inv * nw0) * g0 / (1.0f + exp(-g0)));
  row[f1] = bfloat((v1 * inv * nw1) * g1 / (1.0f + exp(-g1)));
  row[f2] = bfloat((v2 * inv * nw2) * g2 / (1.0f + exp(-g2)));
  row[f3] = bfloat((v3 * inv * nw3) * g3 / (1.0f + exp(-g3)));
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_gdn_chunk_norm_gate_inplace_f16_c64(
    device ushort *inout_u16 [[buffer(0)]],
    device const bfloat *z [[buffer(1)]],
    device const bfloat *norm_weight [[buffer(2)]],
    constant GdnChunkMegaParams &params [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if (sg >= 4u || params.value_dim != kValDim) {
    return;
  }
  const uint pad_end =
      (params.tokens == 0u)
          ? min(params.mp, kChunkSize)
          : min(params.mp, ((params.tokens + kChunkSize - 1u) / kChunkSize) * kChunkSize);
  const uint logical_head = tg.y * 4u + sg;
  const uint token = logical_head / params.value_heads;
  if (token >= pad_end) {
    return;
  }
  const uint hv = logical_head - token * params.value_heads;
  const ulong head_off =
      (ulong(token) * ulong(params.value_heads) + ulong(hv)) * ulong(kValDim);
  device half *dst_f16 = reinterpret_cast<device half *>(inout_u16) + head_off;

  const uint f0 = lane, f1 = lane + 32u, f2 = lane + 64u, f3 = lane + 96u;
  if (token >= params.tokens) {
    dst_f16[f0] = half(0.0h);
    dst_f16[f1] = half(0.0h);
    dst_f16[f2] = half(0.0h);
    dst_f16[f3] = half(0.0h);
    return;
  }

  device const bfloat *row_bf16 =
      reinterpret_cast<device const bfloat *>(inout_u16) + head_off;
  device const bfloat *gate = z + head_off;

  const float v0 = float(row_bf16[f0]), v1 = float(row_bf16[f1]);
  const float v2 = float(row_bf16[f2]), v3 = float(row_bf16[f3]);
  float sq = fma(v0, v0, 0.0f);
  sq = fma(v1, v1, sq);
  sq = fma(v2, v2, sq);
  sq = fma(v3, v3, sq);
  const float inv = rsqrt(simd_sum(sq) / 128.0f + params.epsilon);

  const float g0 = float(gate[f0]), g1 = float(gate[f1]);
  const float g2 = float(gate[f2]), g3 = float(gate[f3]);
  const float nw0 = float(norm_weight[f0]), nw1 = float(norm_weight[f1]);
  const float nw2 = float(norm_weight[f2]), nw3 = float(norm_weight[f3]);

  const bfloat o0 = bfloat((v0 * inv * nw0) * g0 / (1.0f + exp(-g0)));
  const bfloat o1 = bfloat((v1 * inv * nw1) * g1 / (1.0f + exp(-g1)));
  const bfloat o2 = bfloat((v2 * inv * nw2) * g2 / (1.0f + exp(-g2)));
  const bfloat o3 = bfloat((v3 * inv * nw3) * g3 / (1.0f + exp(-g3)));

  dst_f16[f0] = half(clamp(float(o0), -65504.0f, 65504.0f));
  dst_f16[f1] = half(clamp(float(o1), -65504.0f, 65504.0f));
  dst_f16[f2] = half(clamp(float(o2), -65504.0f, 65504.0f));
  dst_f16[f3] = half(clamp(float(o3), -65504.0f, 65504.0f));
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_gdn_chunk_recurrence_mega_bf16_c64(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device const bfloat *history [[buffer(2)]],
    device const bfloat *qk_ws [[buffer(3)]],
    device const uchar *ctrl_ws [[buffer(4)]],
    device float *state [[buffer(5)]],
    device const bfloat *z [[buffer(6)]],
    device const bfloat *norm_weight [[buffer(7)]],
    device bfloat *output_bf16 [[buffer(8)]],
    constant GdnChunkMegaParams &params [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup GdnMegaTgStorage tg_mem;
  gdn_chunk_recurrence_c64_impl<true, 1u>(
      qkv, conv_weight, history, qk_ws, ctrl_ws, state, z, norm_weight,
      output_bf16, nullptr, params, tg_mem, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_gdn_chunk_recurrence_mega_f16_c64(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *conv_weight [[buffer(1)]],
    device const bfloat *history [[buffer(2)]],
    device const bfloat *qk_ws [[buffer(3)]],
    device const uchar *ctrl_ws [[buffer(4)]],
    device float *state [[buffer(5)]],
    device const bfloat *z [[buffer(6)]],
    device const bfloat *norm_weight [[buffer(7)]],
    device half *output_f16 [[buffer(8)]],
    constant GdnChunkMegaParams &params [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  alignas(16) threadgroup GdnMegaTgStorage tg_mem;
  gdn_chunk_recurrence_c64_impl<true, 2u>(
      qkv, conv_weight, history, qk_ws, ctrl_ws, state, z, norm_weight,
      nullptr, output_f16, params, tg_mem, lane, sg, tg);
}
