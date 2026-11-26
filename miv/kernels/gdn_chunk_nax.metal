// Tier-1 Pillar 2 (T1-B): Monolithic Two-Stage C=64 BF16-NAX + FP32-State GDN Prefill
// Stage 1 (`omlx_m5_gdn_chunk_prepare_{c64,conv_c64}`): parallel over (key_head, chunk),
//   fuses bfloat4 register-sliding-window causal Conv1D + SiLU on Q & K,
//   normalizes Q & K in FP32 ONCE per (chunk, key_head) (eliminating repeat=Hv/Hk
//   duplication), computes raw KK^T [64, 64] and QK^T [64, 64] ONCE via 4-SIMDgroup
//   relaxed_precision=true bfloat x bfloat -> float NAX matmul2d, then for each shared
//   value_head solves NaN-proof FP32 cumulative log-decay G[0..63] and exact 2x2 block
//   IEEE FP32 (I_64 + L)^-1 * diag(beta) (parallel SG0/SG1 32x32 + FP32 NAX).
//   Static TG SRAM: 16,384 B (16.00 KiB <= 32,768 B Apple10 limit; 2 TGs/core!).
// Stage 2 (`omlx_m5_gdn_chunk_recurrence_{c64,mega_bf16_c64,mega_f16_c64}`):
//   1 threadgroup (8 SIMDgroups = 256 threads) per value_head (full value_dim=128),
//   keeps persistent [128 x 128] IEEE FP32 recurrent state in NAX cooperative_tensor
//   registers across all C=64 chunks, runs relaxed_precision=true bfloat x bfloat -> float
//   NAX matmul2d with causal 2x2 block pruning, fuses bfloat4 sliding-window Conv1D + SiLU
//   on V, and fuses in-SRAM 0-ULP RMSNorm_128(c_out, norm_weight) * SiLU(z) -> BF16 / FP16.
//   Static TG SRAM: 16,384 B (16.00 KiB <= 32,768 B Apple10 limit).

#include "common_math.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

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

namespace {
constant constexpr uint kChunkSize = 64u;
constant constexpr uint kHalfChunk = 32u;
constant constexpr uint kKeyDim = 128u;
constant constexpr uint kValDim = 128u;
constant constexpr uint kValSgDim = 16u;

// Per-(chunk, key_head) Q/K record in BF16 words (32,768 B = 32 KiB, 256B-aligned):
//   [0 .. 8191]     nk[64][128] (bfloat)
//   [8192 .. 16383] nq[64][128] (bfloat)
constant constexpr ulong kQkPairBfloats =
    2ul * ulong(kChunkSize) * ulong(kKeyDim);

// Per-(chunk, value_head) control record in bytes (256B-aligned = 17,152 B):
//   byte [0 .. 8191]       solve[64][64] (bfloat) = (I + L)^-1 * diag(beta)
//   byte [8192 .. 16383]   mqk[64][64]   (bfloat) = causal exp(G[i]-G[j]) * (nq_i.nk_j)
//   byte [16384 .. 16639]  prefix[64]    (float)  = exp(G[i])
//   byte [16640 .. 16895]  suffix[64]    (float)  = exp(G[63] - G[i])
//   byte [16896 .. 17151]  decay_last    (float)  = exp(G[63]) (+ pad)
constant constexpr ulong kCtrlStrideBytes = 17152ul;
constant constexpr ulong kCtrlMqkByteOff = 8192ul;
constant constexpr ulong kCtrlPrefByteOff = 16384ul;
constant constexpr ulong kCtrlSuffByteOff = 16640ul;
constant constexpr ulong kCtrlMetaByteOff = 16896ul;

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

// Coalesced bfloat4 sliding-window Conv1D(K=4) + SiLU into row-major [Rows x 128] tg_dst.
template <uint NumSg, uint Rows>
inline void gdn_stage_conv_silu_128(
    device const bfloat *qkv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    uint t_base,
    uint valid_rows,
    uint head_ch_off,
    uint conv_width,
    uint conv_kernel,
    float4 w0,
    float4 w1,
    float4 w2,
    float4 w3,
    threadgroup bfloat *tg_dst,
    uint sg,
    uint lane) {
  constexpr uint kRowsPerSg = Rows / NumSg;
  const uint r_start = sg * kRowsPerSg;
  const uint ch_local = lane * 4u;
  const uint ch0 = head_ch_off + ch_local;

  if (conv_kernel == 4u && (conv_width & 3u) == 0u) {
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

template <bool FuseConv, typename ParamsT>
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
    threadgroup bfloat *tg_buf,
    uint lane,
    uint sg,
    uint3 tg) {
  const uint hk = tg.x;
  const uint chunk_idx = tg.y;
  const uint t0 = chunk_idx * kChunkSize;
  if (hk >= params.key_heads || t0 >= params.tokens || params.key_dim != kKeyDim) {
    return;
  }

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

  float4 kw0 = float4(0.0f), kw1 = float4(0.0f), kw2 = float4(0.0f), kw3 = float4(0.0f);
  float4 qw0 = float4(0.0f), qw1 = float4(0.0f), qw2 = float4(0.0f), qw3 = float4(0.0f);
  if constexpr (FuseConv) {
    if (params.conv_kernel == 4u && (params.conv_width & 3u) == 0u) {
      gdn_load_conv4_weights_vec4(conv_weight, k_head_off + lane * 4u,
                                  kw0, kw1, kw2, kw3);
      gdn_load_conv4_weights_vec4(conv_weight, q_head_off + lane * 4u,
                                  qw0, qw1, qw2, qw3);
    }
    gdn_stage_conv_silu_128<4u, kChunkSize>(
        qkv_or_conv, conv_weight, history, t0, clen, k_head_off,
        params.conv_width, params.conv_kernel, kw0, kw1, kw2, kw3, tg_buf, sg, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // 1. Cooperatively FP32 L2-normalize K[0..63, 0..127] ONCE per (chunk, key_head).
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
    tg_buf[row_off + lane] = nk0;
    tg_buf[row_off + lane + 32u] = nk1;
    tg_buf[row_off + lane + 64u] = nk2;
    tg_buf[row_off + lane + 96u] = nk3;
    ws_nk[row_off + lane] = nk0;
    ws_nk[row_off + lane + 32u] = nk1;
    ws_nk[row_off + lane + 64u] = nk2;
    ws_nk[row_off + lane + 96u] = nk3;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2. Compute raw KK^T [64, 64] = tg_buf(K) @ tg_buf(K)^T ONCE per key_head.
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tBuf(
      tg_buf, dextents<int32_t, 2>(int(kKeyDim), int(kChunkSize)));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tK_dev(
      ws_nk, dextents<int32_t, 2>(int(kKeyDim), int(kChunkSize)));

  constexpr auto desc_64_64_128 = matmul2d_descriptor(
      int(kChunkSize), int(kChunkSize), int(kKeyDim), false, true, true,
      matmul2d_descriptor::mode::multiply);
  matmul2d<desc_64_64_128, execution_simdgroups<4>> op_64_nt;
  auto c_kkt = op_64_nt.get_destination_cooperative_tensor<
      decltype(tBuf), decltype(tBuf), float>();
  #pragma clang loop unroll(full)
  for (uint16_t e = 0; e < c_kkt.get_capacity(); ++e) {
    c_kkt[e] = 0.0f;
  }
  op_64_nt.run(tBuf, tBuf, c_kkt);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3. Stage & FP32 L2-normalize Q[0..63, 0..127] into tg_buf (16 KiB) & compute
  //    raw QK^T [64, 64] = tg_buf(Q) @ ws_nk(K)^T ONCE per key_head in ONE 64x64x128 op!
  if constexpr (FuseConv) {
    gdn_stage_conv_silu_128<4u, kChunkSize>(
        qkv_or_conv, conv_weight, history, t0, clen, q_head_off,
        params.conv_width, params.conv_kernel, qw0, qw1, qw2, qw3, tg_buf, sg, lane);
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
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto c_qkt = op_64_nt.get_destination_cooperative_tensor<
      decltype(tBuf), decltype(tK_dev), float>();
  #pragma clang loop unroll(full)
  for (uint16_t e = 0; e < c_qkt.get_capacity(); ++e) {
    c_qkt[e] = 0.0f;
  }
  op_64_nt.run(tBuf, tK_dev, c_qkt);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Partition the single 16 KiB tg_buf (4,096 floats) into four 4 KiB slots:
  //   slot0 [0..1023]:    L00[32x32] -> X00[32x32]
  //   slot1 [1024..2047]: L11[32x32] -> X11[32x32]
  //   slot2 [2048..3071]: L10[32x32] -> T[32x32]
  //   slot3 [3072..4095]: tg_G[64] + tg_beta[64]
  threadgroup float *tg_f32 = reinterpret_cast<threadgroup float *>(tg_buf);
  threadgroup float *tg_slot0 = tg_f32;
  threadgroup float *tg_slot1 = tg_f32 + 1024u;
  threadgroup float *tg_slot2 = tg_f32 + 2048u;
  threadgroup float *tg_G = tg_f32 + 3072u;
  threadgroup float *tg_beta = tg_f32 + 3136u;

  // Strict IEEE FP32 (relaxed_precision = false) for exact (I + L)^-1 off-diagonal solve!
  constexpr auto desc_f32_32 = matmul2d_descriptor(
      int(kHalfChunk), int(kHalfChunk), int(kHalfChunk), false, false, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<desc_f32_32, execution_simdgroups<4>> op_f32_32;

  // 4. Solve per-value-head decay & exact 2x2 block FP32 (I_64 + L)^-1 * diag(beta).
  for (uint rep = 0u; rep < repeat; ++rep) {
    const uint hv = hk * repeat + rep;
    if (hv >= params.value_heads) {
      break;
    }
    device uchar *ctrl_raw =
        ctrl_ws + (ulong(chunk_idx) * ulong(params.value_heads) + ulong(hv)) * kCtrlStrideBytes;
    device bfloat *ctrl_solve = reinterpret_cast<device bfloat *>(ctrl_raw);
    device bfloat *ctrl_mqk =
        reinterpret_cast<device bfloat *>(ctrl_raw + kCtrlMqkByteOff);
    device float *ctrl_pref =
        reinterpret_cast<device float *>(ctrl_raw + kCtrlPrefByteOff);
    device float *ctrl_suff =
        reinterpret_cast<device float *>(ctrl_raw + kCtrlSuffByteOff);
    device float *ctrl_meta =
        reinterpret_cast<device float *>(ctrl_raw + kCtrlMetaByteOff);

    // 4a. NaN-proof cumulative FP32 log-decay G[0..63] <= 0 and beta[0..63] in SG 0.
    if (sg == 0u) {
      const float dt_bias_val = float(dt_bias[hv]);
      const float neg_exp_a = -exp(float(a_log[hv]));

      float g0 = 0.0f, beta0 = 0.0f;
      if (lane < clen) {
        const ulong h_idx0 = ulong(t0 + lane) * ulong(params.value_heads) + ulong(hv);
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
        const ulong h_idx1 = ulong(t0 + r1) * ulong(params.value_heads) + ulong(hv);
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
      ctrl_pref[lane] = exp(g_cum0);
      ctrl_pref[r1] = exp(g_cum1);
      ctrl_suff[lane] = exp(min(0.0f, g_last - g_cum0));
      ctrl_suff[r1] = exp(min(0.0f, g_last - g_cum1));
      if (lane == 0u) {
        ctrl_meta[0] = exp(g_last);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4b. Emit causal BF16 mqk[64, 64] & build compact 3-block FP32 L00, L11, L10.
    #pragma clang loop unroll(full)
    for (uint16_t e = 0; e < c_qkt.get_capacity(); ++e) {
      if (c_qkt.is_valid_element(e)) {
        const auto idx = c_qkt.get_multidimensional_index(e);
        const uint col = uint(idx[0]);
        const uint row = uint(idx[1]);
        float m_val = 0.0f;
        if (row >= col && row < clen && col < clen) {
          m_val = exp(min(0.0f, tg_G[row] - tg_G[col])) * c_qkt[e];
        }
        ctrl_mqk[row * kChunkSize + col] = bfloat(m_val);
      }
    }
    #pragma clang loop unroll(full)
    for (uint16_t e = 0; e < c_kkt.get_capacity(); ++e) {
      if (c_kkt.is_valid_element(e)) {
        const auto idx = c_kkt.get_multidimensional_index(e);
        const uint col = uint(idx[0]);
        const uint row = uint(idx[1]);
        float l_val = 0.0f;
        if (row > col && row < clen) {
          l_val = tg_beta[row] * exp(min(0.0f, tg_G[row] - tg_G[col])) * c_kkt[e];
        }
        if (row < kHalfChunk && col < kHalfChunk) {
          tg_slot0[row * kHalfChunk + col] = l_val;
        } else if (row >= kHalfChunk && col >= kHalfChunk) {
          tg_slot1[(row - kHalfChunk) * kHalfChunk + (col - kHalfChunk)] = l_val;
        } else if (row >= kHalfChunk && col < kHalfChunk) {
          tg_slot2[(row - kHalfChunk) * kHalfChunk + col] = l_val;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4c. Exact 2x2 block FP32 inversion of (I_64 + L):
    //     SG 0 inverts X00 in registers then writes X00 into slot0;
    //     SG 1 simultaneously inverts X11 in registers then writes X11 into slot1.
    if (sg == 0u) {
      const float beta_col = (lane < clen) ? tg_beta[lane] : 0.0f;
      float x_col[kHalfChunk];
      for (uint row = 0u; row < kHalfChunk; ++row) {
        float x_val = 0.0f;
        if (row < clen) {
          if (lane == row) {
            x_val = 1.0f;
          } else if (lane < row) {
            float sum = 0.0f;
            for (uint k = lane; k < row; ++k) {
              sum = fma(tg_slot0[row * kHalfChunk + k], x_col[k], sum);
            }
            x_val = -sum;
          }
        }
        x_col[row] = x_val;
        ctrl_solve[row * kChunkSize + lane] = bfloat(x_val * beta_col);
        ctrl_solve[row * kChunkSize + (kHalfChunk + lane)] = bfloat(0.0f);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint row = 0u; row < kHalfChunk; ++row) {
        tg_slot0[row * kHalfChunk + lane] = x_col[row];
      }
    } else if (sg == 1u) {
      if (clen > kHalfChunk) {
        const uint clen1 = clen - kHalfChunk;
        const float beta_col =
            (kHalfChunk + lane < clen) ? tg_beta[kHalfChunk + lane] : 0.0f;
        float x_col[kHalfChunk];
        for (uint row = 0u; row < kHalfChunk; ++row) {
          float x_val = 0.0f;
          if (row < clen1) {
            if (lane == row) {
              x_val = 1.0f;
            } else if (lane < row) {
              float sum = 0.0f;
              for (uint k = lane; k < row; ++k) {
                sum = fma(tg_slot1[row * kHalfChunk + k], x_col[k], sum);
              }
              x_val = -sum;
            }
          }
          x_col[row] = x_val;
          ctrl_solve[(kHalfChunk + row) * kChunkSize + (kHalfChunk + lane)] =
              bfloat(x_val * beta_col);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = 0u; row < kHalfChunk; ++row) {
          tg_slot1[row * kHalfChunk + lane] = x_col[row];
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

    // Off-diagonal 32x32 block X10 = -X11 @ (L10 @ X00) via pure-FP32 4-SG NAX matmul2d.
    if (clen > kHalfChunk) {
      tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tX00(
          tg_slot0, dextents<int32_t, 2>(int(kHalfChunk), int(kHalfChunk)));
      tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tX11(
          tg_slot1, dextents<int32_t, 2>(int(kHalfChunk), int(kHalfChunk)));
      tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tSlot2(
          tg_slot2, dextents<int32_t, 2>(int(kHalfChunk), int(kHalfChunk)));

      auto c_T = op_f32_32.get_destination_cooperative_tensor<
          decltype(tSlot2), decltype(tX00), float>();
      #pragma clang loop unroll(full)
      for (uint16_t e = 0; e < c_T.get_capacity(); ++e) {
        c_T[e] = 0.0f;
      }
      op_f32_32.run(tSlot2, tX00, c_T);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      c_T.store(tSlot2);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      auto c_X10 = op_f32_32.get_destination_cooperative_tensor<
          decltype(tX11), decltype(tSlot2), float>();
      #pragma clang loop unroll(full)
      for (uint16_t e = 0; e < c_X10.get_capacity(); ++e) {
        c_X10[e] = 0.0f;
      }
      op_f32_32.run(tX11, tSlot2, c_X10);
      #pragma clang loop unroll(full)
      for (uint16_t e = 0; e < c_X10.get_capacity(); ++e) {
        if (c_X10.is_valid_element(e)) {
          const auto idx = c_X10.get_multidimensional_index(e);
          const uint col = uint(idx[0]);
          const uint row = uint(idx[1]);
          const float s_val =
              (kHalfChunk + row < clen) ? (-c_X10[e] * tg_beta[col]) : 0.0f;
          ctrl_solve[(kHalfChunk + row) * kChunkSize + col] = bfloat(s_val);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// EpilogueMode: 0 = raw recurrent BF16, 1 = fused RMSNorm_128 * SiLU(z) -> BF16,
//               2 = fused RMSNorm_128 * SiLU(z) -> FP16 + tail zero-padding.
template <bool FuseConv, uint EpilogueMode, typename ParamsT>
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
    threadgroup bfloat *tg_work,
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

  const uint repeat = params.value_heads / params.key_heads;
  const uint hk = hv / repeat;
  const uint key_width = params.key_heads * kKeyDim;
  const uint dv_sg_base = sg * kValSgDim;
  const uint out_sg_col_base = hv * kValDim + dv_sg_base;
  const uint v_head_col_base = 2u * key_width + hv * kValDim;
  const uint v_sg_col_base = v_head_col_base + dv_sg_base;

  float nw0 = 0.0f, nw1 = 0.0f, nw2 = 0.0f, nw3 = 0.0f;
  if constexpr (EpilogueMode != 0u) {
    nw0 = float(norm_weight[lane]);
    nw1 = float(norm_weight[lane + 32u]);
    nw2 = float(norm_weight[lane + 64u]);
    nw3 = float(norm_weight[lane + 96u]);
  }

  float4 vw0 = float4(0.0f), vw1 = float4(0.0f), vw2 = float4(0.0f), vw3 = float4(0.0f);
  if constexpr (FuseConv) {
    if (params.conv_kernel == 4u && (params.conv_width & 3u) == 0u) {
      gdn_load_conv4_weights_vec4(conv_weight, v_head_col_base + lane * 4u,
                                  vw0, vw1, vw2, vw3);
    }
  }

  // SIMDgroup-private 2 KiB BF16 tile [64 x 16] (also holds [16 x 32] S_sub during Step 1).
  threadgroup bfloat *my_work = tg_work + sg * (kChunkSize * kValSgDim);
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tMyS(
      my_work, dextents<int32_t, 2>(int(kHalfChunk), int(kValSgDim)));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tMyWork(
      my_work, dextents<int32_t, 2>(int(kValSgDim), int(kChunkSize)));
  auto tMyWork0 = tMyWork.slice(0, 0);
  auto tMyWork1 = tMyWork.slice(0, int(kHalfChunk));

  // Apple10 full-speed BF16 NAX tensor core rate (`relaxed_precision = true` with IEEE FP32 CType!).
  constexpr auto desc_nt_mac = matmul2d_descriptor(
      int(kHalfChunk), int(kValSgDim), int(kHalfChunk), false, true, true,
      matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_nn_mul = matmul2d_descriptor(
      int(kHalfChunk), int(kValSgDim), int(kHalfChunk), false, false, true,
      matmul2d_descriptor::mode::multiply);
  constexpr auto desc_nn_mac = matmul2d_descriptor(
      int(kHalfChunk), int(kValSgDim), int(kHalfChunk), false, false, true,
      matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_tn_mac = matmul2d_descriptor(
      int(kValSgDim), int(kHalfChunk), int(kHalfChunk), true, false, true,
      matmul2d_descriptor::mode::multiply_accumulate);

  matmul2d<desc_nt_mac, execution_simdgroup> op_nt_mac;
  matmul2d<desc_nn_mul, execution_simdgroup> op_nn_mul;
  matmul2d<desc_nn_mac, execution_simdgroup> op_nn_mac;
  matmul2d<desc_tn_mac, execution_simdgroup> op_tn_mac;

  // Load persistent IEEE FP32 state S[16, 128] into four [16 x 32] cooperative_tensor tiles.
  device float *state_sg = state + ulong(out_sg_col_base) * ulong(kKeyDim);
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tState_sg(
      state_sg, dextents<int32_t, 2>(int(kKeyDim), int(kValSgDim)));
  auto tState0 = tState_sg.slice(0, 0);
  auto tState1 = tState_sg.slice(32, 0);
  auto tState2 = tState_sg.slice(64, 0);
  auto tState3 = tState_sg.slice(96, 0);

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tDummyNk(
      (device bfloat *)qk_ws, dextents<int32_t, 2>(int(kKeyDim), int(kChunkSize)));
  auto tDummyNkSub = tDummyNk.slice(0, 0);

  auto c_s0 = op_tn_mac.get_destination_cooperative_tensor<
      decltype(tMyWork0), decltype(tDummyNkSub), float>();
  auto c_s1 = op_tn_mac.get_destination_cooperative_tensor<
      decltype(tMyWork0), decltype(tDummyNkSub), float>();
  auto c_s2 = op_tn_mac.get_destination_cooperative_tensor<
      decltype(tMyWork0), decltype(tDummyNkSub), float>();
  auto c_s3 = op_tn_mac.get_destination_cooperative_tensor<
      decltype(tMyWork0), decltype(tDummyNkSub), float>();

  c_s0.load(tState0);
  c_s1.load(tState1);
  c_s2.load(tState2);
  c_s3.load(tState3);

  for (uint c = 0u; c * kChunkSize < params.tokens; ++c) {
    const uint t0 = c * kChunkSize;
    const uint clen = min(kChunkSize, params.tokens - t0);
    const bool has_hi = (clen > kHalfChunk);

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

    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tNk(
        (device bfloat *)ws_nk, dextents<int32_t, 2>(int(kKeyDim), int(kChunkSize)));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tNq(
        (device bfloat *)ws_nq, dextents<int32_t, 2>(int(kKeyDim), int(kChunkSize)));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tSolve(
        (device bfloat *)ctrl_solve,
        dextents<int32_t, 2>(int(kChunkSize), int(kChunkSize)));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tMqk(
        (device bfloat *)ctrl_mqk,
        dextents<int32_t, 2>(int(kChunkSize), int(kChunkSize)));

    // 1. Jointly accumulate pred = nk @ S^T and hist = nq @ S^T across four 32-col slices
    //    while converting each S_sub to BF16 ONCE per chunk.
    auto c_pred0 = op_nt_mac.get_destination_cooperative_tensor<
        decltype(tDummyNkSub), decltype(tMyS), float>();
    auto c_pred1 = op_nt_mac.get_destination_cooperative_tensor<
        decltype(tDummyNkSub), decltype(tMyS), float>();
    auto c_hist0 = op_nt_mac.get_destination_cooperative_tensor<
        decltype(tDummyNkSub), decltype(tMyS), float>();
    auto c_hist1 = op_nt_mac.get_destination_cooperative_tensor<
        decltype(tDummyNkSub), decltype(tMyS), float>();
    #pragma clang loop unroll(full)
    for (uint16_t e = 0; e < c_pred0.get_capacity(); ++e) {
      c_pred0[e] = 0.0f;
      c_pred1[e] = 0.0f;
      c_hist0[e] = 0.0f;
      c_hist1[e] = 0.0f;
    }

    auto accum_k_slice = [&](const thread auto &c_sb, int k_off) {
      #pragma clang loop unroll(full)
      for (uint16_t e = 0; e < c_sb.get_capacity(); ++e) {
        if (c_sb.is_valid_element(e)) {
          const auto idx = c_sb.get_multidimensional_index(e);
          my_work[uint(idx[1]) * kHalfChunk + uint(idx[0])] = bfloat(c_sb[e]);
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      auto tNk0 = tNk.slice(k_off, 0);
      auto tNq0 = tNq.slice(k_off, 0);
      op_nt_mac.run(tNk0, tMyS, c_pred0);
      op_nt_mac.run(tNq0, tMyS, c_hist0);
      if (has_hi) {
        auto tNk1 = tNk.slice(k_off, int(kHalfChunk));
        auto tNq1 = tNq.slice(k_off, int(kHalfChunk));
        op_nt_mac.run(tNk1, tMyS, c_pred1);
        op_nt_mac.run(tNq1, tMyS, c_hist1);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    };
    accum_k_slice(c_s0, 0);
    accum_k_slice(c_s1, 32);
    accum_k_slice(c_s2, 64);
    accum_k_slice(c_s3, 96);

    // Stage V[0..63, 0..127] via coalesced bfloat4 8-row sliding-window Conv1D+SiLU into
    // [sg=0..7][r=0..63][dv=0..15] layout in tg_work so each SG reads its own my_work!
    if constexpr (FuseConv) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint r_start = sg * 8u;
      const uint ch_local = lane * 4u;
      const uint ch0 = v_head_col_base + ch_local;
      const uint dst_sg_off =
          (ch_local >> 4u) * (kChunkSize * kValSgDim) + (ch_local & 15u);
      if (params.conv_kernel == 4u && (params.conv_width & 3u) == 0u) {
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
          *reinterpret_cast<threadgroup bfloat4 *>(
              tg_work + dst_sg_off + r * kValSgDim) = out_v;
        }
      } else {
        for (uint s = 0u; s < 8u; ++s) {
          const uint r = r_start + s;
          for (uint j = 0u; j < 4u; ++j) {
            tg_work[dst_sg_off + r * kValSgDim + j] =
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

    // Scale hist by prefix[r] & write rhs = bfloat(V - prefix[r] * pred) into my_work.
    #pragma clang loop unroll(full)
    for (uint16_t e = 0; e < c_pred0.get_capacity(); ++e) {
      if (c_pred0.is_valid_element(e)) {
        const auto idx = c_pred0.get_multidimensional_index(e);
        const uint dv = uint(idx[0]);
        const uint r = uint(idx[1]);
        bfloat rhs_bf = bfloat(0.0f);
        if (r < clen) {
          const float v_val =
              FuseConv
                  ? float(my_work[r * kValSgDim + dv])
                  : float(qkv_or_conv[ulong(t0 + r) * ulong(params.conv_width) +
                                      ulong(v_sg_col_base + dv)]);
          rhs_bf = bfloat(fma(-ctrl_pref[r], c_pred0[e], v_val));
          c_hist0[e] *= ctrl_pref[r];
        } else {
          c_hist0[e] = 0.0f;
        }
        my_work[r * kValSgDim + dv] = rhs_bf;
      }
    }
    if (has_hi) {
      #pragma clang loop unroll(full)
      for (uint16_t e = 0; e < c_pred1.get_capacity(); ++e) {
        if (c_pred1.is_valid_element(e)) {
          const auto idx = c_pred1.get_multidimensional_index(e);
          const uint dv = uint(idx[0]);
          const uint r = kHalfChunk + uint(idx[1]);
          bfloat rhs_bf = bfloat(0.0f);
          if (r < clen) {
            const float v_val =
                FuseConv
                    ? float(my_work[r * kValSgDim + dv])
                    : float(qkv_or_conv[ulong(t0 + r) * ulong(params.conv_width) +
                                        ulong(v_sg_col_base + dv)]);
            rhs_bf = bfloat(fma(-ctrl_pref[r], c_pred1[e], v_val));
            c_hist1[e] *= ctrl_pref[r];
          } else {
            c_hist1[e] = 0.0f;
          }
          my_work[r * kValSgDim + dv] = rhs_bf;
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Causal 2x2 block delta = solve @ rhs (upper-right 32x32 block is zero!).
    auto tSolve00 = tSolve.slice(0, 0);
    auto c_delta0 = op_nn_mul.get_destination_cooperative_tensor<
        decltype(tSolve00), decltype(tMyWork0), float>();
    auto c_delta1 = op_nn_mul.get_destination_cooperative_tensor<
        decltype(tSolve00), decltype(tMyWork0), float>();
    #pragma clang loop unroll(full)
    for (uint16_t e = 0; e < c_delta0.get_capacity(); ++e) {
      c_delta0[e] = 0.0f;
      c_delta1[e] = 0.0f;
    }
    op_nn_mul.run(tSolve00, tMyWork0, c_delta0);
    if (has_hi) {
      auto tSolve10 = tSolve.slice(0, int(kHalfChunk));
      auto tSolve11 = tSolve.slice(int(kHalfChunk), int(kHalfChunk));
      op_nn_mul.run(tSolve10, tMyWork0, c_delta1);
      op_nn_mac.run(tSolve11, tMyWork1, c_delta1);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Causal 2x2 block c_out = c_hist + mqk @ bfloat(delta).
    #pragma clang loop unroll(full)
    for (uint16_t e = 0; e < c_delta0.get_capacity(); ++e) {
      if (c_delta0.is_valid_element(e)) {
        const auto idx = c_delta0.get_multidimensional_index(e);
        const uint dv = uint(idx[0]);
        const uint r = uint(idx[1]);
        my_work[r * kValSgDim + dv] =
            (r < clen) ? bfloat(c_delta0[e]) : bfloat(0.0f);
      }
    }
    if (has_hi) {
      #pragma clang loop unroll(full)
      for (uint16_t e = 0; e < c_delta1.get_capacity(); ++e) {
        if (c_delta1.is_valid_element(e)) {
          const auto idx = c_delta1.get_multidimensional_index(e);
          const uint dv = uint(idx[0]);
          const uint r = kHalfChunk + uint(idx[1]);
          my_work[r * kValSgDim + dv] =
              (r < clen) ? bfloat(c_delta1[e]) : bfloat(0.0f);
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    auto tMqk00 = tMqk.slice(0, 0);
    op_nn_mac.run(tMqk00, tMyWork0, c_hist0);
    if (has_hi) {
      auto tMqk10 = tMqk.slice(0, int(kHalfChunk));
      auto tMqk11 = tMqk.slice(int(kHalfChunk), int(kHalfChunk));
      op_nn_mac.run(tMqk10, tMyWork0, c_hist1);
      op_nn_mac.run(tMqk11, tMyWork1, c_hist1);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Update persistent IEEE FP32 state S = decay_last * S + delta_suff^T @ nk.
    #pragma clang loop unroll(full)
    for (uint16_t e = 0; e < c_delta0.get_capacity(); ++e) {
      if (c_delta0.is_valid_element(e)) {
        const auto idx = c_delta0.get_multidimensional_index(e);
        const uint dv = uint(idx[0]);
        const uint r = uint(idx[1]);
        my_work[r * kValSgDim + dv] =
            (r < clen) ? bfloat(ctrl_suff[r] * c_delta0[e]) : bfloat(0.0f);
      }
    }
    if (has_hi) {
      #pragma clang loop unroll(full)
      for (uint16_t e = 0; e < c_delta1.get_capacity(); ++e) {
        if (c_delta1.is_valid_element(e)) {
          const auto idx = c_delta1.get_multidimensional_index(e);
          const uint dv = uint(idx[0]);
          const uint r = kHalfChunk + uint(idx[1]);
          my_work[r * kValSgDim + dv] =
              (r < clen) ? bfloat(ctrl_suff[r] * c_delta1[e]) : bfloat(0.0f);
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    auto update_s_slice = [&](thread auto &c_sb, int k_off) {
      #pragma clang loop unroll(full)
      for (uint16_t e = 0; e < c_sb.get_capacity(); ++e) {
        if (c_sb.is_valid_element(e)) {
          c_sb[e] *= decay_last;
        }
      }
      auto tNk0 = tNk.slice(k_off, 0);
      op_tn_mac.run(tMyWork0, tNk0, c_sb);
      if (has_hi) {
        auto tNk1 = tNk.slice(k_off, int(kHalfChunk));
        op_tn_mac.run(tMyWork1, tNk1, c_sb);
      }
    };
    update_s_slice(c_s0, 0);
    update_s_slice(c_s1, 32);
    update_s_slice(c_s2, 64);
    update_s_slice(c_s3, 96);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // 5. Epilogue: raw recurrent BF16 OR fused 0-ULP RMSNorm_128 * SiLU(z) -> BF16 / FP16.
    if constexpr (EpilogueMode == 0u) {
      #pragma clang loop unroll(full)
      for (uint16_t e = 0; e < c_hist0.get_capacity(); ++e) {
        if (c_hist0.is_valid_element(e)) {
          const auto idx = c_hist0.get_multidimensional_index(e);
          const uint dv = uint(idx[0]);
          const uint r = uint(idx[1]);
          if (r < clen) {
            out_bf16[ulong(t0 + r) * ulong(value_width) +
                     ulong(out_sg_col_base + dv)] = bfloat(c_hist0[e]);
          }
        }
      }
      if (has_hi) {
        #pragma clang loop unroll(full)
        for (uint16_t e = 0; e < c_hist1.get_capacity(); ++e) {
          if (c_hist1.is_valid_element(e)) {
            const auto idx = c_hist1.get_multidimensional_index(e);
            const uint dv = uint(idx[0]);
            const uint r = kHalfChunk + uint(idx[1]);
            if (r < clen) {
              out_bf16[ulong(t0 + r) * ulong(value_width) +
                       ulong(out_sg_col_base + dv)] = bfloat(c_hist1[e]);
            }
          }
        }
      }
    } else {
      #pragma clang loop unroll(full)
      for (uint16_t e = 0; e < c_hist0.get_capacity(); ++e) {
        if (c_hist0.is_valid_element(e)) {
          const auto idx = c_hist0.get_multidimensional_index(e);
          my_work[uint(idx[1]) * kValSgDim + uint(idx[0])] = bfloat(c_hist0[e]);
        }
      }
      if (has_hi) {
        #pragma clang loop unroll(full)
        for (uint16_t e = 0; e < c_hist1.get_capacity(); ++e) {
          if (c_hist1.is_valid_element(e)) {
            const auto idx = c_hist1.get_multidimensional_index(e);
            my_work[(kHalfChunk + uint(idx[1])) * kValSgDim + uint(idx[0])] =
                bfloat(c_hist1[e]);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const uint f0 = lane;
      const uint f1 = lane + 32u;
      const uint f2 = lane + 64u;
      const uint f3 = lane + 96u;
      const uint sg0_off = (f0 >> 4u) * (kChunkSize * kValSgDim) + (f0 & 15u);
      const uint sg1_off = (f1 >> 4u) * (kChunkSize * kValSgDim) + (f1 & 15u);
      const uint sg2_off = (f2 >> 4u) * (kChunkSize * kValSgDim) + (f2 & 15u);
      const uint sg3_off = (f3 >> 4u) * (kChunkSize * kValSgDim) + (f3 & 15u);

      for (uint r = sg; r < clen; r += 8u) {
        const uint r_off = r * kValSgDim;
        const float v0 = float(tg_work[sg0_off + r_off]);
        const float v1 = float(tg_work[sg1_off + r_off]);
        const float v2 = float(tg_work[sg2_off + r_off]);
        const float v3 = float(tg_work[sg3_off + r_off]);
        float sq = fma(v0, v0, 0.0f);
        sq = fma(v1, v1, sq);
        sq = fma(v2, v2, sq);
        sq = fma(v3, v3, sq);
        const float inv = rsqrt(simd_sum(sq) / 128.0f + params.epsilon);

        const ulong head_row_off =
            ulong(t0 + r) * ulong(value_width) + ulong(hv * kValDim);
        device const bfloat *z_row = z + head_row_off;
        const float g0 = float(z_row[f0]);
        const float g1 = float(z_row[f1]);
        const float g2 = float(z_row[f2]);
        const float g3 = float(z_row[f3]);

        const bfloat o0 = bfloat((v0 * inv * nw0) * g0 / (1.0f + exp(-g0)));
        const bfloat o1 = bfloat((v1 * inv * nw1) * g1 / (1.0f + exp(-g1)));
        const bfloat o2 = bfloat((v2 * inv * nw2) * g2 / (1.0f + exp(-g2)));
        const bfloat o3 = bfloat((v3 * inv * nw3) * g3 / (1.0f + exp(-g3)));

        if constexpr (EpilogueMode == 1u) {
          device bfloat *dst = out_bf16 + head_row_off;
          dst[f0] = o0;
          dst[f1] = o1;
          dst[f2] = o2;
          dst[f3] = o3;
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

  // Zero-pad tail rows [params.tokens, min(params.mp, ceil64(params.tokens))) for MXFP4 NAX gdn_out.
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

  c_s0.store(tState0);
  c_s1.store(tState1);
  c_s2.store(tState2);
  c_s3.store(tState3);
}

// ============================================================================
// Stage 2 (`v0.8.0-m5rt.9` Full-Chip Split-Dv=32 + Register-Fragment 16x32x16 NAX):
//   Splits value_dim=128 into 4 independent kDvSlice=32 threadgroups per value_head:
//     - 35B-A3B (Hv=32): 32 * 4 = 128 TGs -> 2.00 exact waves across all 64 M5 Ultra cores!
//     - 27B     (Hv=48): 48 * 4 = 192 TGs -> 3.00 exact waves across all 64 M5 Ultra cores!
//   Inside each 4-SG (128-thread) TG:
//     - Persistent IEEE FP32 state S[dv=32, dk_sg=32] is ONLY 32 floats/lane (6.25x lower
//       VGPR footprint per TG vs 256-thread full-Dv=128, eliminating 100% of VGPR spilling!).
//     - Step 1 (`pred` & `hist`), Step 2 (`delta`, 4x4 16x16 causal pruned), Step 3 (`c_out`,
//       4x4 16x16 causal pruned), and Step 4 (`S` update) all run Apple10 register-fragment
//       16x32x16 `bfloat x bfloat -> float` (`relaxed_precision=true`) NAX MMA.
//     - Static TG SRAM: 4,096 floats = 16,384 B (16.00 KiB -> 2 TGs/core, <= 32 KiB even
//       under 2x MTL_SHADER_VALIDATION=1 shadow).
// ============================================================================

constant constexpr uint kDvSlice = 32u;
constant constexpr uint kRhsPad = 72u;   // 64 + 8 (8-byte bfloat4 aligned, bank-conflict-free)
constant constexpr uint kNkTPad = 40u;   // 32 + 8 (8-byte bfloat4 aligned, bank-conflict-free)

inline void gdn_mma_16x32x16_bf16(
    thread vec<float, 8> &Cn0,
    thread vec<float, 8> &Cn1,
    thread const vec<bfloat, 8> &A,
    thread const vec<bfloat, 8> &Bn0,
    thread const vec<bfloat, 8> &Bn1) {
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b =
      gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>,
      remove_addrspace_t<decltype(ct_b)>,
      float>();
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 8u; ++i) {
    ct_a[i] = A[i];
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 8u; ++i) {
    ct_b[i] = Bn0[i];
    ct_b[8u + i] = Bn1[i];
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 8u; ++i) {
    ct_c[i] = Cn0[i];
    ct_c[8u + i] = Cn1[i];
  }
  gemm_op.run(ct_a, ct_b, ct_c);
  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 8u; ++i) {
    Cn0[i] = ct_c[i];
    Cn1[i] = ct_c[8u + i];
  }
}

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

inline vec<bfloat, 8> gdn_load_frag_dev_bf16(
    device const bfloat *base,
    uint stride,
    uint fm,
    uint fn) {
  vec<bfloat, 8> out;
  const bfloat4 v0 =
      *reinterpret_cast<device const bfloat4 *>(base + ulong(fm) * stride + fn);
  const bfloat4 v1 =
      *reinterpret_cast<device const bfloat4 *>(base + ulong(fm + 8u) * stride + fn);
  #pragma clang loop unroll(full)
  for (ushort j = 0u; j < 4u; ++j) {
    out[j] = v0[j];
    out[4u + j] = v1[j];
  }
  return out;
}

template <uint Stride>
inline vec<bfloat, 8> gdn_load_frag_tg_bf16(
    const threadgroup bfloat *base,
    uint fm,
    uint fn) {
  vec<bfloat, 8> out;
  const bfloat4 v0 =
      *reinterpret_cast<const threadgroup bfloat4 *>(base + fm * Stride + fn);
  const bfloat4 v1 =
      *reinterpret_cast<const threadgroup bfloat4 *>(base + (fm + 8u) * Stride + fn);
  #pragma clang loop unroll(full)
  for (ushort j = 0u; j < 4u; ++j) {
    out[j] = v0[j];
    out[4u + j] = v1[j];
  }
  return out;
}

// Apple9 / portable cooperative_tensor fallback for the exact same 4-SG kDvSlice=32 algorithm.
template <bool FuseConv, typename ParamsT>
inline void gdn_chunk_recurrence_dv32_c64_apple9(
    device const bfloat *qkv_or_conv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    device const bfloat *qk_ws,
    device const uchar *ctrl_ws,
    device float *state,
    device bfloat *out_bf16,
    constant ParamsT &params,
    threadgroup float *tg_f32,
    uint lane,
    uint sg,
    uint hv,
    uint dv_slice) {
  const uint tid = sg * 32u + lane;
  const uint repeat = params.value_heads / params.key_heads;
  const uint hk = hv / repeat;
  const uint key_width = params.key_heads * kKeyDim;
  const uint value_width = params.value_heads * kValDim;
  const uint dv_tg_base = dv_slice * kDvSlice;
  const uint out_tg_col_base = hv * kValDim + dv_tg_base;
  const uint v_tg_col_base = 2u * key_width + out_tg_col_base;
  const uint dk_sg_base = sg * 32u;
  const uint r_sg_base = sg * 16u;

  float4 vw0 = float4(0.0f), vw1 = float4(0.0f), vw2 = float4(0.0f), vw3 = float4(0.0f);
  if constexpr (FuseConv) {
    if (params.conv_kernel == 4u && (params.conv_width & 3u) == 0u) {
      gdn_load_conv4_weights_vec4(
          conv_weight, v_tg_col_base + (tid & 7u) * 4u, vw0, vw1, vw2, vw3);
    }
  }

  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, /*relaxed_precision=*/true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> gemm_op;
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b =
      gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  using CtC = decltype(gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>,
      remove_addrspace_t<decltype(ct_b)>,
      float>());

  // Persistent FP32 state S[v_sub=0..1][k_sub=0..1]:
  //   c_s[v_sub] holds [m = dv_in_16 (0..15), n = dk_in_32 (0..31)] = 16 floats/lane * 2 = 32 floats/lane!
  CtC c_s0 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
  CtC c_s1 = gemm_op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();

  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 16u; ++i) {
    if (c_s0.is_valid_element(i)) {
      const auto idx = c_s0.get_multidimensional_index(i);
      const uint dk_l = uint(idx[0]);
      const uint dv_l = uint(idx[1]);
      c_s0[i] = state[ulong(out_tg_col_base + dv_l) * kKeyDim + dk_sg_base + dk_l];
      c_s1[i] = state[ulong(out_tg_col_base + 16u + dv_l) * kKeyDim + dk_sg_base + dk_l];
    } else {
      c_s0[i] = 0.0f;
      c_s1[i] = 0.0f;
    }
  }

  threadgroup bfloat *tg_bf16 = reinterpret_cast<threadgroup bfloat *>(tg_f32);

  for (uint c = 0u; c * kChunkSize < params.tokens; ++c) {
    const uint t0 = c * kChunkSize;
    const uint clen = min(kChunkSize, params.tokens - t0);
    const bool has_hi = (clen > kHalfChunk);

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

    // Stage this SG's S[dv=32, dk=32] as bfloat into tg_bf16[sg * 1024] (4 * 1024 = 4,096 bfloats = 8 KiB)
    // so ct_b can read [dv=0..31, dk=0..15/16..31] cleanly.
    threadgroup bfloat *my_s_bf = tg_bf16 + sg * 1024u;
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      if (c_s0.is_valid_element(i)) {
        const auto idx = c_s0.get_multidimensional_index(i);
        const uint dk_l = uint(idx[0]);
        const uint dv_l = uint(idx[1]);
        my_s_bf[dv_l * 32u + dk_l] = bfloat(c_s0[i]);
        my_s_bf[(16u + dv_l) * 32u + dk_l] = bfloat(c_s1[i]);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    decltype(ct_b) ct_b_s0 =
        gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    decltype(ct_b) ct_b_s1 =
        gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      const auto idx_b = ct_b_s0.get_multidimensional_index(i);
      ct_b_s0[i] = my_s_bf[uint(idx_b[1]) * 32u + uint(idx_b[0])];
      ct_b_s1[i] = my_s_bf[uint(idx_b[1]) * 32u + 16u + uint(idx_b[0])];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto reduce_proj_16x32 = [&](device const bfloat *ws_mat, thread CtC &out_sg) {
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) {
        out_sg[i] = 0.0f;
      }
      for (uint rh = 0u; rh < (has_hi ? 2u : 1u); ++rh) {
        const uint r_off = rh * 32u;
        CtC p0 = gemm_op.template get_destination_cooperative_tensor<
            remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
        CtC p1 = gemm_op.template get_destination_cooperative_tensor<
            remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          p0[i] = 0.0f;
          p1[i] = 0.0f;
        }
        #pragma clang loop unroll(full)
        for (uint ks = 0u; ks < 2u; ++ks) {
          const uint k_col = dk_sg_base + ks * 16u;
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 8u; ++i) {
            const auto ia = ct_a.get_multidimensional_index(i);
            ct_a[i] = ws_mat[(r_off + uint(ia[1])) * kKeyDim + k_col + uint(ia[0])];
          }
          gemm_op.run(ct_a, ks == 0u ? ct_b_s0 : ct_b_s1, p0);
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 8u; ++i) {
            const auto ia = ct_a.get_multidimensional_index(i);
            ct_a[i] = ws_mat[(r_off + 16u + uint(ia[1])) * kKeyDim + k_col + uint(ia[0])];
          }
          gemm_op.run(ct_a, ks == 0u ? ct_b_s0 : ct_b_s1, p1);
        }
        threadgroup float *my_f32 = tg_f32 + sg * 1024u;
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          if (p0.is_valid_element(i)) {
            const auto ic = p0.get_multidimensional_index(i);
            my_f32[uint(ic[1]) * 32u + uint(ic[0])] = p0[i];
            my_f32[(16u + uint(ic[1])) * 32u + uint(ic[0])] = p1[i];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((sg >> 1u) == rh) {
          const uint sub_m = (sg & 1u) * 16u;
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            if (out_sg.is_valid_element(i)) {
              const auto ic = out_sg.get_multidimensional_index(i);
              const uint idx_flat = (sub_m + uint(ic[1])) * 32u + uint(ic[0]);
              out_sg[i] = (((tg_f32[idx_flat] + tg_f32[1024u + idx_flat]) +
                            tg_f32[2048u + idx_flat]) +
                           tg_f32[3072u + idx_flat]);
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    };

    CtC c_pred = gemm_op.template get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
    CtC c_hist = gemm_op.template get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
    reduce_proj_16x32(ws_nk, c_pred);
    reduce_proj_16x32(ws_nq, c_hist);

    // Stage Conv1D+SiLU(V) [64 x 32] into tg_bf16[4096 .. 6143] if FuseConv.
    threadgroup bfloat *tg_rhsT = tg_bf16;               // [0 .. 2303]
    threadgroup bfloat *tg_V = tg_bf16 + 4096u;           // [4096 .. 6143]
    if constexpr (FuseConv) {
      const uint dv_l4 = (tid & 7u) * 4u;
      const uint ch0 = v_tg_col_base + dv_l4;
      const uint r_start = (tid >> 3u) * 4u;
      if (params.conv_kernel == 4u && (params.conv_width & 3u) == 0u) {
        float4 xm3 = float4(0.0f), xm2 = float4(0.0f), xm1 = float4(0.0f);
        if (r_start < clen) {
          const int base_tok = int(t0 + r_start);
          xm3 = gdn_read_qkv_or_hist_vec4(qkv_or_conv, history, base_tok - 3, ch0, params.conv_width);
          xm2 = gdn_read_qkv_or_hist_vec4(qkv_or_conv, history, base_tok - 2, ch0, params.conv_width);
          xm1 = gdn_read_qkv_or_hist_vec4(qkv_or_conv, history, base_tok - 1, ch0, params.conv_width);
        }
        #pragma clang loop unroll(full)
        for (uint s = 0u; s < 4u; ++s) {
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
          *reinterpret_cast<threadgroup bfloat4 *>(tg_V + r * 32u + dv_l4) = out_v;
        }
      } else {
        for (uint s = 0u; s < 4u; ++s) {
          const uint r = r_start + s;
          for (uint j = 0u; j < 4u; ++j) {
            tg_V[r * 32u + dv_l4 + j] =
                (r < clen)
                    ? gdn_eval_conv_silu_scalar(qkv_or_conv, conv_weight, history,
                                                t0 + r, ch0 + j, params.conv_width,
                                                params.conv_kernel)
                    : bfloat(0.0f);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      if (c_pred.is_valid_element(i)) {
        const auto ic = c_pred.get_multidimensional_index(i);
        const uint dv_l = uint(ic[0]);
        const uint r = r_sg_base + uint(ic[1]);
        bfloat rhs_bf = bfloat(0.0f);
        if (r < clen) {
          const float v_val =
              FuseConv
                  ? float(tg_V[r * 32u + dv_l])
                  : float(qkv_or_conv[ulong(t0 + r) * ulong(params.conv_width) +
                                      ulong(v_tg_col_base + dv_l)]);
          const float pref = ctrl_pref[r];
          rhs_bf = bfloat(fma(-pref, c_pred[i], v_val));
          c_hist[i] *= pref;
        } else {
          c_hist[i] = 0.0f;
        }
        tg_rhsT[dv_l * kRhsPad + r] = rhs_bf;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: delta = solve @ rhs (4x4 16x16 causal block pruned)
    CtC c_delta = gemm_op.template get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(ct_a)>, remove_addrspace_t<decltype(ct_b)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      c_delta[i] = 0.0f;
    }
    if (r_sg_base < clen) {
      for (uint kt = 0u; kt <= sg; ++kt) {
        if (kt * 16u < clen) {
          const uint k_off = kt * 16u;
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 8u; ++i) {
            const auto ia = ct_a.get_multidimensional_index(i);
            ct_a[i] = ctrl_solve[(r_sg_base + uint(ia[1])) * kChunkSize + k_off + uint(ia[0])];
          }
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            const auto ib = ct_b.get_multidimensional_index(i);
            ct_b[i] = tg_rhsT[uint(ib[1]) * kRhsPad + k_off + uint(ib[0])];
          }
          gemm_op.run(ct_a, ct_b, c_delta);
        }
      }
    }

    threadgroup bfloat *tg_deltaT = tg_bf16 + 2304u;       // [2304 .. 4607]
    threadgroup bfloat *tg_deltaSuffT = tg_bf16 + 5888u;   // [5888 .. 8191]
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      if (c_delta.is_valid_element(i)) {
        const auto ic = c_delta.get_multidimensional_index(i);
        const uint dv_l = uint(ic[0]);
        const uint r = r_sg_base + uint(ic[1]);
        if (r < clen) {
          tg_deltaT[dv_l * kRhsPad + r] = bfloat(c_delta[i]);
          tg_deltaSuffT[dv_l * kRhsPad + r] = bfloat(ctrl_suff[r] * c_delta[i]);
        } else {
          tg_deltaT[dv_l * kRhsPad + r] = bfloat(0.0f);
          tg_deltaSuffT[dv_l * kRhsPad + r] = bfloat(0.0f);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: c_out = c_hist + mqk @ tg_deltaT -> out_bf16
    if (r_sg_base < clen) {
      for (uint kt = 0u; kt <= sg; ++kt) {
        if (kt * 16u < clen) {
          const uint k_off = kt * 16u;
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 8u; ++i) {
            const auto ia = ct_a.get_multidimensional_index(i);
            ct_a[i] = ctrl_mqk[(r_sg_base + uint(ia[1])) * kChunkSize + k_off + uint(ia[0])];
          }
          #pragma clang loop unroll(full)
          for (ushort i = 0u; i < 16u; ++i) {
            const auto ib = ct_b.get_multidimensional_index(i);
            ct_b[i] = tg_deltaT[uint(ib[1]) * kRhsPad + k_off + uint(ib[0])];
          }
          gemm_op.run(ct_a, ct_b, c_hist);
        }
      }
      #pragma clang loop unroll(full)
      for (ushort i = 0u; i < 16u; ++i) {
        if (c_hist.is_valid_element(i)) {
          const auto ic = c_hist.get_multidimensional_index(i);
          const uint dv_l = uint(ic[0]);
          const uint r = r_sg_base + uint(ic[1]);
          if (r < clen) {
            out_bf16[ulong(t0 + r) * ulong(value_width) + ulong(out_tg_col_base + dv_l)] =
                bfloat(c_hist[i]);
          }
        }
      }
    }

    // Step 4: S = decay_last * S + tg_deltaSuffT @ nk
    #pragma clang loop unroll(full)
    for (ushort i = 0u; i < 16u; ++i) {
      c_s0[i] *= decay_last;
      c_s1[i] *= decay_last;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup bfloat *tg_nkT = tg_bf16;                  // [0 .. 5119]
    const uint t_r = tid >> 2u;
    const uint t_dk0 = (tid & 3u) * 32u;
    for (uint rh = 0u; rh < (has_hi ? 2u : 1u); ++rh) {
      if (rh > 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      const uint r_src = rh * 32u + t_r;
      device const bfloat4 *nk_src4 =
          reinterpret_cast<device const bfloat4 *>(ws_nk + r_src * kKeyDim + t_dk0);
      #pragma clang loop unroll(full)
      for (uint v4 = 0u; v4 < 8u; ++v4) {
        const bfloat4 w4 = nk_src4[v4];
        const uint dk_b = t_dk0 + v4 * 4u;
        tg_nkT[(dk_b + 0u) * kNkTPad + t_r] = w4[0];
        tg_nkT[(dk_b + 1u) * kNkTPad + t_r] = w4[1];
        tg_nkT[(dk_b + 2u) * kNkTPad + t_r] = w4[2];
        tg_nkT[(dk_b + 3u) * kNkTPad + t_r] = w4[3];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      #pragma clang loop unroll(full)
      for (uint rs = 0u; rs < 2u; ++rs) {
        const uint r_sub_nk = rs * 16u;
        const uint r_sub_ds = rh * 32u + r_sub_nk;
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 16u; ++i) {
          const auto ib = ct_b.get_multidimensional_index(i);
          ct_b[i] = tg_nkT[(dk_sg_base + uint(ib[1])) * kNkTPad + r_sub_nk + uint(ib[0])];
        }
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 8u; ++i) {
          const auto ia = ct_a.get_multidimensional_index(i);
          ct_a[i] = tg_deltaSuffT[uint(ia[1]) * kRhsPad + r_sub_ds + uint(ia[0])];
        }
        gemm_op.run(ct_a, ct_b, c_s0);
        #pragma clang loop unroll(full)
        for (ushort i = 0u; i < 8u; ++i) {
          const auto ia = ct_a.get_multidimensional_index(i);
          ct_a[i] = tg_deltaSuffT[(16u + uint(ia[1])) * kRhsPad + r_sub_ds + uint(ia[0])];
        }
        gemm_op.run(ct_a, ct_b, c_s1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  #pragma clang loop unroll(full)
  for (ushort i = 0u; i < 16u; ++i) {
    if (c_s0.is_valid_element(i)) {
      const auto idx = c_s0.get_multidimensional_index(i);
      const uint dk_l = uint(idx[0]);
      const uint dv_l = uint(idx[1]);
      state[ulong(out_tg_col_base + dv_l) * kKeyDim + dk_sg_base + dk_l] = c_s0[i];
      state[ulong(out_tg_col_base + 16u + dv_l) * kKeyDim + dk_sg_base + dk_l] = c_s1[i];
    }
  }
}

template <bool FuseConv, typename ParamsT>
inline void gdn_chunk_recurrence_dv32_c64_impl(
    device const bfloat *qkv_or_conv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    device const bfloat *qk_ws,
    device const uchar *ctrl_ws,
    device float *state,
    device bfloat *out_bf16,
    constant ParamsT &params,
    threadgroup float *tg_f32,
    uint lane,
    uint sg,
    uint3 tg) {
  const uint hv = tg.x;
  const uint dv_slice = tg.y;
  if (hv >= params.value_heads || dv_slice >= 4u || sg >= 4u ||
      params.key_dim != kKeyDim || params.value_dim != kValDim ||
      params.tokens == 0u) {
    return;
  }

  if (!gdn_has_apple10_nax_layout()) {
    gdn_chunk_recurrence_dv32_c64_apple9<FuseConv>(
        qkv_or_conv, conv_weight, history, qk_ws, ctrl_ws, state, out_bf16,
        params, tg_f32, lane, sg, hv, dv_slice);
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint repeat = params.value_heads / params.key_heads;
  const uint hk = hv / repeat;
  const uint key_width = params.key_heads * kKeyDim;
  const uint value_width = params.value_heads * kValDim;
  const uint dv_tg_base = dv_slice * kDvSlice;
  const uint out_tg_col_base = hv * kValDim + dv_tg_base;
  const uint v_tg_col_base = 2u * key_width + out_tg_col_base;
  const uint dk_sg_base = sg * 32u;
  const uint r_sg_base = sg * 16u;

  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

  float4 vw0 = float4(0.0f), vw1 = float4(0.0f), vw2 = float4(0.0f), vw3 = float4(0.0f);
  if constexpr (FuseConv) {
    if (params.conv_kernel == 4u && (params.conv_width & 3u) == 0u) {
      gdn_load_conv4_weights_vec4(
          conv_weight, v_tg_col_base + (tid & 7u) * 4u, vw0, vw1, vw2, vw3);
    }
  }

  // Persistent IEEE FP32 state S[v_sub=0..1][k_sub=0..1] (32 floats/lane total!).
  vec<float, 8> S[2][2];
  #pragma clang loop unroll(full)
  for (uint vs = 0u; vs < 2u; ++vs) {
    #pragma clang loop unroll(full)
    for (uint ks = 0u; ks < 2u; ++ks) {
      const uint dk_col = dk_sg_base + ks * 16u + fn;
      #pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const uint dv_l = vs * 16u + fm + h * 8u;
        const float4 f4 = *reinterpret_cast<device const float4 *>(
            state + ulong(out_tg_col_base + dv_l) * kKeyDim + dk_col);
        S[vs][ks][h * 4u + 0u] = f4[0];
        S[vs][ks][h * 4u + 1u] = f4[1];
        S[vs][ks][h * 4u + 2u] = f4[2];
        S[vs][ks][h * 4u + 3u] = f4[3];
      }
    }
  }

  threadgroup bfloat *tg_bf16 = reinterpret_cast<threadgroup bfloat *>(tg_f32);

  for (uint c = 0u; c * kChunkSize < params.tokens; ++c) {
    const uint t0 = c * kChunkSize;
    const uint clen = min(kChunkSize, params.tokens - t0);
    const bool has_hi = (clen > kHalfChunk);

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

    // Convert S to bfloat ONCE per chunk directly in registers (0 SRAM, 0 barriers!).
    vec<bfloat, 8> Sb[2][2];
    #pragma clang loop unroll(full)
    for (uint vs = 0u; vs < 2u; ++vs) {
      #pragma clang loop unroll(full)
      for (uint ks = 0u; ks < 2u; ++ks) {
        #pragma clang loop unroll(full)
        for (ushort e = 0u; e < 8u; ++e) {
          Sb[vs][ks][e] = bfloat(S[vs][ks][e]);
        }
      }
    }

    auto reduce_proj_apple10 = [&](device const bfloat *ws_mat,
                                   thread vec<float, 8> (&out_sg)[2]) {
      out_sg[0] = vec<float, 8>(0.0f);
      out_sg[1] = vec<float, 8>(0.0f);
      for (uint rh = 0u; rh < (has_hi ? 2u : 1u); ++rh) {
        const uint r_off = rh * 32u;
        vec<float, 8> part[2][2];
        part[0][0] = vec<float, 8>(0.0f);
        part[0][1] = vec<float, 8>(0.0f);
        part[1][0] = vec<float, 8>(0.0f);
        part[1][1] = vec<float, 8>(0.0f);

        volatile int cbar_p;
        const vec<bfloat, 8> a00 =
            gdn_load_frag_dev_bf16(ws_mat + r_off * kKeyDim + dk_sg_base + 0u, kKeyDim, fm, fn);
        const vec<bfloat, 8> a01 =
            gdn_load_frag_dev_bf16(ws_mat + r_off * kKeyDim + dk_sg_base + 16u, kKeyDim, fm, fn);
        const vec<bfloat, 8> a10 =
            gdn_load_frag_dev_bf16(ws_mat + (r_off + 16u) * kKeyDim + dk_sg_base + 0u, kKeyDim, fm, fn);
        const vec<bfloat, 8> a11 =
            gdn_load_frag_dev_bf16(ws_mat + (r_off + 16u) * kKeyDim + dk_sg_base + 16u, kKeyDim, fm, fn);
        gdn_mma_16x32x16_bf16(part[0][0], part[0][1], a00, Sb[0][0], Sb[1][0]);
        gdn_mma_16x32x16_bf16(part[0][0], part[0][1], a01, Sb[0][1], Sb[1][1]);
        gdn_mma_16x32x16_bf16(part[1][0], part[1][1], a10, Sb[0][0], Sb[1][0]);
        gdn_mma_16x32x16_bf16(part[1][0], part[1][1], a11, Sb[0][1], Sb[1][1]);
        (void)cbar_p;

        threadgroup float *my_f32 = tg_f32 + sg * 1024u;
        #pragma clang loop unroll(full)
        for (uint mm = 0u; mm < 2u; ++mm) {
          #pragma clang loop unroll(full)
          for (uint nn = 0u; nn < 2u; ++nn) {
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint r_l = mm * 16u + fm + h * 8u;
              const uint dv_l = nn * 16u + fn;
              *reinterpret_cast<threadgroup float4 *>(my_f32 + r_l * 32u + dv_l) =
                  float4(part[mm][nn][h * 4u + 0u], part[mm][nn][h * 4u + 1u],
                         part[mm][nn][h * 4u + 2u], part[mm][nn][h * 4u + 3u]);
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if ((sg >> 1u) == rh) {
          const uint sub_m = (sg & 1u) * 16u;
          #pragma clang loop unroll(full)
          for (uint nn = 0u; nn < 2u; ++nn) {
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 2u; ++h) {
              const uint idx_flat = (sub_m + fm + h * 8u) * 32u + nn * 16u + fn;
              const float4 s0 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + idx_flat);
              const float4 s1 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + 1024u + idx_flat);
              const float4 s2 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + 2048u + idx_flat);
              const float4 s3 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + 3072u + idx_flat);
              const float4 tot = (((s0 + s1) + s2) + s3);
              out_sg[nn][h * 4u + 0u] = tot[0];
              out_sg[nn][h * 4u + 1u] = tot[1];
              out_sg[nn][h * 4u + 2u] = tot[2];
              out_sg[nn][h * 4u + 3u] = tot[3];
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    };

    vec<float, 8> P_sg[2];
    vec<float, 8> H_sg[2];
    reduce_proj_apple10(ws_nk, P_sg);
    reduce_proj_apple10(ws_nq, H_sg);

    // Stage Conv1D+SiLU(V) [64 x 32] into tg_bf16[4096 .. 6143] if FuseConv.
    threadgroup bfloat *tg_rhsT = tg_bf16;               // [0 .. 2303]
    threadgroup bfloat *tg_V = tg_bf16 + 4096u;           // [4096 .. 6143]
    if constexpr (FuseConv) {
      const uint dv_l4 = (tid & 7u) * 4u;
      const uint ch0 = v_tg_col_base + dv_l4;
      const uint r_start = (tid >> 3u) * 4u;
      if (params.conv_kernel == 4u && (params.conv_width & 3u) == 0u) {
        float4 xm3 = float4(0.0f), xm2 = float4(0.0f), xm1 = float4(0.0f);
        if (r_start < clen) {
          const int base_tok = int(t0 + r_start);
          xm3 = gdn_read_qkv_or_hist_vec4(qkv_or_conv, history, base_tok - 3, ch0, params.conv_width);
          xm2 = gdn_read_qkv_or_hist_vec4(qkv_or_conv, history, base_tok - 2, ch0, params.conv_width);
          xm1 = gdn_read_qkv_or_hist_vec4(qkv_or_conv, history, base_tok - 1, ch0, params.conv_width);
        }
        #pragma clang loop unroll(full)
        for (uint s = 0u; s < 4u; ++s) {
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
          *reinterpret_cast<threadgroup bfloat4 *>(tg_V + r * 32u + dv_l4) = out_v;
        }
      } else {
        for (uint s = 0u; s < 4u; ++s) {
          const uint r = r_start + s;
          for (uint j = 0u; j < 4u; ++j) {
            tg_V[r * 32u + dv_l4 + j] =
                (r < clen)
                    ? gdn_eval_conv_silu_scalar(qkv_or_conv, conv_weight, history,
                                                t0 + r, ch0 + j, params.conv_width,
                                                params.conv_kernel)
                    : bfloat(0.0f);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write transposed tg_rhsT[dv_l * 72 + r] & scale H_sg by prefix[r].
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint r = r_sg_base + fm + h * 8u;
      if (r < clen) {
        const float pref = ctrl_pref[r];
        #pragma clang loop unroll(full)
        for (uint nn = 0u; nn < 2u; ++nn) {
          const uint dv_base = nn * 16u + fn;
          const bfloat4 v4 = FuseConv
              ? *reinterpret_cast<const threadgroup bfloat4 *>(tg_V + r * 32u + dv_base)
              : *reinterpret_cast<device const bfloat4 *>(
                    qkv_or_conv + ulong(t0 + r) * ulong(params.conv_width) +
                    ulong(v_tg_col_base + dv_base));
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            const uint e = h * 4u + j;
            tg_rhsT[(dv_base + j) * kRhsPad + r] =
                bfloat(fma(-pref, P_sg[nn][e], float(v4[j])));
            H_sg[nn][e] *= pref;
          }
        }
      } else {
        #pragma clang loop unroll(full)
        for (uint nn = 0u; nn < 2u; ++nn) {
          const uint dv_base = nn * 16u + fn;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            tg_rhsT[(dv_base + j) * kRhsPad + r] = bfloat(0.0f);
            H_sg[nn][h * 4u + j] = 0.0f;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: delta = solve @ rhs (4x4 16x16 causal block pruned!)
    vec<float, 8> D_sg[2] = {vec<float, 8>(0.0f), vec<float, 8>(0.0f)};
    if (r_sg_base < clen) {
      for (uint kt = 0u; kt <= sg; ++kt) {
        if (kt * 16u < clen) {
          const uint k_off = kt * 16u;
          volatile int cbar_d;
          const vec<bfloat, 8> a_sol =
              gdn_load_frag_dev_bf16(ctrl_solve + r_sg_base * kChunkSize + k_off, kChunkSize, fm, fn);
          const vec<bfloat, 8> b_r0 =
              gdn_load_frag_tg_bf16<kRhsPad>(tg_rhsT + k_off, fm, fn);
          const vec<bfloat, 8> b_r1 =
              gdn_load_frag_tg_bf16<kRhsPad>(tg_rhsT + 16u * kRhsPad + k_off, fm, fn);
          gdn_mma_16x32x16_bf16(D_sg[0], D_sg[1], a_sol, b_r0, b_r1);
          (void)cbar_d;
        }
      }
    }

    // Write BOTH tg_deltaT [2304 .. 4607] and tg_deltaSuffT [5888 .. 8191] before 1 barrier!
    threadgroup bfloat *tg_deltaT = tg_bf16 + 2304u;
    threadgroup bfloat *tg_deltaSuffT = tg_bf16 + 5888u;
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint r = r_sg_base + fm + h * 8u;
      const float suff = (r < clen) ? ctrl_suff[r] : 0.0f;
      #pragma clang loop unroll(full)
      for (uint nn = 0u; nn < 2u; ++nn) {
        const uint dv_base = nn * 16u + fn;
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
          const uint e = h * 4u + j;
          const uint dv_l = dv_base + j;
          tg_deltaT[dv_l * kRhsPad + r] = (r < clen) ? bfloat(D_sg[nn][e]) : bfloat(0.0f);
          tg_deltaSuffT[dv_l * kRhsPad + r] = (r < clen) ? bfloat(suff * D_sg[nn][e]) : bfloat(0.0f);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: c_out = H_sg + mqk @ tg_deltaT (4x4 16x16 causal block pruned) -> coalesced bfloat4 store!
    if (r_sg_base < clen) {
      for (uint kt = 0u; kt <= sg; ++kt) {
        if (kt * 16u < clen) {
          const uint k_off = kt * 16u;
          volatile int cbar_c;
          const vec<bfloat, 8> a_mqk =
              gdn_load_frag_dev_bf16(ctrl_mqk + r_sg_base * kChunkSize + k_off, kChunkSize, fm, fn);
          const vec<bfloat, 8> b_d0 =
              gdn_load_frag_tg_bf16<kRhsPad>(tg_deltaT + k_off, fm, fn);
          const vec<bfloat, 8> b_d1 =
              gdn_load_frag_tg_bf16<kRhsPad>(tg_deltaT + 16u * kRhsPad + k_off, fm, fn);
          gdn_mma_16x32x16_bf16(H_sg[0], H_sg[1], a_mqk, b_d0, b_d1);
          (void)cbar_c;
        }
      }
      #pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const uint r = r_sg_base + fm + h * 8u;
        if (r < clen) {
          #pragma clang loop unroll(full)
          for (uint nn = 0u; nn < 2u; ++nn) {
            const bfloat4 out4 = bfloat4(
                bfloat(H_sg[nn][h * 4u + 0u]),
                bfloat(H_sg[nn][h * 4u + 1u]),
                bfloat(H_sg[nn][h * 4u + 2u]),
                bfloat(H_sg[nn][h * 4u + 3u]));
            *reinterpret_cast<device bfloat4 *>(
                out_bf16 + ulong(t0 + r) * ulong(value_width) +
                ulong(out_tg_col_base + nn * 16u + fn)) = out4;
          }
        }
      }
    }

    // Step 4: S = decay_last * S + tg_deltaSuffT @ nk
    S[0][0] *= decay_last;
    S[0][1] *= decay_last;
    S[1][0] *= decay_last;
    S[1][1] *= decay_last;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup bfloat *tg_nkT = tg_bf16;                  // [0 .. 5119]
    const uint t_r = tid >> 2u;
    const uint t_dk0 = (tid & 3u) * 32u;
    for (uint rh = 0u; rh < (has_hi ? 2u : 1u); ++rh) {
      if (rh > 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      const uint r_src = rh * 32u + t_r;
      device const bfloat4 *nk_src4 =
          reinterpret_cast<device const bfloat4 *>(ws_nk + r_src * kKeyDim + t_dk0);
      #pragma clang loop unroll(full)
      for (uint v4 = 0u; v4 < 8u; ++v4) {
        const bfloat4 w4 = nk_src4[v4];
        const uint dk_b = t_dk0 + v4 * 4u;
        tg_nkT[(dk_b + 0u) * kNkTPad + t_r] = w4[0];
        tg_nkT[(dk_b + 1u) * kNkTPad + t_r] = w4[1];
        tg_nkT[(dk_b + 2u) * kNkTPad + t_r] = w4[2];
        tg_nkT[(dk_b + 3u) * kNkTPad + t_r] = w4[3];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      #pragma clang loop unroll(full)
      for (uint rs = 0u; rs < 2u; ++rs) {
        const uint r_sub_nk = rs * 16u;
        const uint r_sub_ds = rh * 32u + r_sub_nk;
        volatile int cbar_s;
        const vec<bfloat, 8> b_nk0 =
            gdn_load_frag_tg_bf16<kNkTPad>(tg_nkT + dk_sg_base * kNkTPad + r_sub_nk, fm, fn);
        const vec<bfloat, 8> b_nk1 =
            gdn_load_frag_tg_bf16<kNkTPad>(tg_nkT + (dk_sg_base + 16u) * kNkTPad + r_sub_nk, fm, fn);
        const vec<bfloat, 8> a_ds0 =
            gdn_load_frag_tg_bf16<kRhsPad>(tg_deltaSuffT + r_sub_ds, fm, fn);
        const vec<bfloat, 8> a_ds1 =
            gdn_load_frag_tg_bf16<kRhsPad>(tg_deltaSuffT + 16u * kRhsPad + r_sub_ds, fm, fn);
        gdn_mma_16x32x16_bf16(S[0][0], S[0][1], a_ds0, b_nk0, b_nk1);
        gdn_mma_16x32x16_bf16(S[1][0], S[1][1], a_ds1, b_nk0, b_nk1);
        (void)cbar_s;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Coalesced float4 writeback of persistent state S[2][2].
  #pragma clang loop unroll(full)
  for (uint vs = 0u; vs < 2u; ++vs) {
    #pragma clang loop unroll(full)
    for (uint ks = 0u; ks < 2u; ++ks) {
      const uint dk_col = dk_sg_base + ks * 16u + fn;
      #pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const uint dv_l = vs * 16u + fm + h * 8u;
        *reinterpret_cast<device float4 *>(
            state + ulong(out_tg_col_base + dv_l) * kKeyDim + dk_col) =
            float4(S[vs][ks][h * 4u + 0u], S[vs][ks][h * 4u + 1u],
                   S[vs][ks][h * 4u + 2u], S[vs][ks][h * 4u + 3u]);
      }
    }
  }
}

// ============================================================================
// Stage 2 Lossless Base 1.1 (`dv64`: 2 x [128 x 64] Split for 80-Core M5 Ultra):
//   Splits value_dim=128 into 2 independent kDvSlice64=64 TGs per value_head:
//     - 27B     (Hv=48): 48 * 2 = 96 TGs <= 160 resident slots -> 1 SINGLE WAVE!
//     - 35B-A3B (Hv=32): 32 * 2 = 64 TGs -> 1 exact wave on 64-core & lowest SLC traffic!
//   Cuts ws_nk / ws_nq / ctrl_solve / ctrl_mqk / tg_nkT traffic by 2.0x vs dv32,
//   keeps static TG SRAM at EXACT 16,384 B (16.00 KiB -> 2 TGs/core = 160 slots/80c),
//   uses causal block pruning (kt <= sg, 10/16 blocks), coalesced bfloat4 stores,
//   and preserves EXACT 0-ULP bit-for-bit IEEE-754 parity with dv32!
// ============================================================================

constant constexpr uint kDvSlice64 = 64u;
constant constexpr uint kNkTPad16 = 24u; // 16 + 8 (16-byte bfloat4 aligned, 128*24 = 3,072 bfloats)

template <bool FuseConv, typename ParamsT>
inline void gdn_chunk_recurrence_dv64_c64_apple9(
    device const bfloat *qkv_or_conv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    device const bfloat *qk_ws,
    device const uchar *ctrl_ws,
    device float *state,
    device bfloat *out_bf16,
    constant ParamsT &params,
    threadgroup float *tg_f32,
    uint lane,
    uint sg,
    uint hv,
    uint dv_slice) {
  // Portable Apple9 (M4) fallback: execute two sequential dv32 halves so live
  // cooperative_tensor pressure matches dv32 (0 spills under MTL_SHADER_VALIDATION=1,
  // guaranteed 0-ULP parity). Apple10 (M5 Ultra) compiles this branch out completely.
  gdn_chunk_recurrence_dv32_c64_apple9<FuseConv>(
      qkv_or_conv, conv_weight, history, qk_ws, ctrl_ws, state, out_bf16,
      params, tg_f32, lane, sg, hv, dv_slice * 2u + 0u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  gdn_chunk_recurrence_dv32_c64_apple9<FuseConv>(
      qkv_or_conv, conv_weight, history, qk_ws, ctrl_ws, state, out_bf16,
      params, tg_f32, lane, sg, hv, dv_slice * 2u + 1u);
}

template <bool FuseConv, typename ParamsT>
inline void gdn_chunk_recurrence_dv64_c64_impl(
    device const bfloat *qkv_or_conv,
    device const bfloat *conv_weight,
    device const bfloat *history,
    device const bfloat *qk_ws,
    device const uchar *ctrl_ws,
    device float *state,
    device bfloat *out_bf16,
    constant ParamsT &params,
    threadgroup float *tg_f32,
    uint lane,
    uint sg,
    uint3 tg) {
  const uint hv = tg.x;
  const uint dv_slice = tg.y;
  if (hv >= params.value_heads || dv_slice >= 2u || sg >= 4u ||
      params.key_dim != kKeyDim || params.value_dim != kValDim ||
      params.tokens == 0u) {
    return;
  }

  if (!gdn_has_apple10_nax_layout()) {
    gdn_chunk_recurrence_dv64_c64_apple9<FuseConv>(
        qkv_or_conv, conv_weight, history, qk_ws, ctrl_ws, state, out_bf16,
        params, tg_f32, lane, sg, hv, dv_slice);
    return;
  }

  const uint tid = sg * 32u + lane;
  const uint repeat = params.value_heads / params.key_heads;
  const uint hk = hv / repeat;
  const uint key_width = params.key_heads * kKeyDim;
  const uint value_width = params.value_heads * kValDim;
  const uint dv_tg_base = dv_slice * kDvSlice64;
  const uint out_tg_col_base = hv * kValDim + dv_tg_base;
  const uint v_tg_col_base = 2u * key_width + out_tg_col_base;
  const uint dk_sg_base = sg * 32u;
  const uint r_sg_base = sg * 16u;

  const uint qid = lane >> 2u;
  const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
  const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;

  float4 vw0 = float4(0.0f), vw1 = float4(0.0f), vw2 = float4(0.0f), vw3 = float4(0.0f);
  if constexpr (FuseConv) {
    if (params.conv_kernel == 4u && (params.conv_width & 3u) == 0u) {
      gdn_load_conv4_weights_vec4(
          conv_weight, v_tg_col_base + (tid & 15u) * 4u, vw0, vw1, vw2, vw3);
    }
  }

  // Persistent IEEE FP32 state S[vh=0..1][vs=0..1][ks=0..1] (64 floats/lane total).
  vec<float, 8> S[2][2][2];
  #pragma clang loop unroll(full)
  for (uint vh = 0u; vh < 2u; ++vh) {
    #pragma clang loop unroll(full)
    for (uint vs = 0u; vs < 2u; ++vs) {
      #pragma clang loop unroll(full)
      for (uint ks = 0u; ks < 2u; ++ks) {
        const uint dk_col = dk_sg_base + ks * 16u + fn;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint dv_l = vh * 32u + vs * 16u + fm + h * 8u;
          const float4 f4 = *reinterpret_cast<device const float4 *>(
              state + ulong(out_tg_col_base + dv_l) * kKeyDim + dk_col);
          S[vh][vs][ks][h * 4u + 0u] = f4[0];
          S[vh][vs][ks][h * 4u + 1u] = f4[1];
          S[vh][vs][ks][h * 4u + 2u] = f4[2];
          S[vh][vs][ks][h * 4u + 3u] = f4[3];
        }
      }
    }
  }

  threadgroup bfloat *tg_bf16 = reinterpret_cast<threadgroup bfloat *>(tg_f32);

  for (uint c = 0u; c * kChunkSize < params.tokens; ++c) {
    const uint t0 = c * kChunkSize;
    const uint clen = min(kChunkSize, params.tokens - t0);
    const bool has_hi = (clen > kHalfChunk);

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

    // Load ws_mat ONCE per rh half and project against BOTH vh=0 and vh=1 (2x less Q/K traffic!).
    auto reduce_proj_apple10_dv64 = [&](device const bfloat *ws_mat,
                                        thread vec<float, 8> (&out_sg)[2][2]) {
      #pragma clang loop unroll(full)
      for (uint vh = 0u; vh < 2u; ++vh) {
        out_sg[vh][0] = vec<float, 8>(0.0f);
        out_sg[vh][1] = vec<float, 8>(0.0f);
      }
      for (uint rh = 0u; rh < (has_hi ? 2u : 1u); ++rh) {
        const uint r_off = rh * 32u;
        const vec<bfloat, 8> a00 =
            gdn_load_frag_dev_bf16(ws_mat + r_off * kKeyDim + dk_sg_base + 0u, kKeyDim, fm, fn);
        const vec<bfloat, 8> a01 =
            gdn_load_frag_dev_bf16(ws_mat + r_off * kKeyDim + dk_sg_base + 16u, kKeyDim, fm, fn);
        const vec<bfloat, 8> a10 =
            gdn_load_frag_dev_bf16(ws_mat + (r_off + 16u) * kKeyDim + dk_sg_base + 0u, kKeyDim, fm, fn);
        const vec<bfloat, 8> a11 =
            gdn_load_frag_dev_bf16(ws_mat + (r_off + 16u) * kKeyDim + dk_sg_base + 16u, kKeyDim, fm, fn);

        #pragma clang loop unroll(full)
        for (uint vh = 0u; vh < 2u; ++vh) {
          vec<bfloat, 8> Sb00, Sb01, Sb10, Sb11;
          #pragma clang loop unroll(full)
          for (ushort e = 0u; e < 8u; ++e) {
            Sb00[e] = bfloat(S[vh][0][0][e]);
            Sb01[e] = bfloat(S[vh][0][1][e]);
            Sb10[e] = bfloat(S[vh][1][0][e]);
            Sb11[e] = bfloat(S[vh][1][1][e]);
          }
          vec<float, 8> part[2][2];
          part[0][0] = vec<float, 8>(0.0f);
          part[0][1] = vec<float, 8>(0.0f);
          part[1][0] = vec<float, 8>(0.0f);
          part[1][1] = vec<float, 8>(0.0f);

          volatile int cbar_p;
          gdn_mma_16x32x16_bf16(part[0][0], part[0][1], a00, Sb00, Sb10);
          gdn_mma_16x32x16_bf16(part[0][0], part[0][1], a01, Sb01, Sb11);
          gdn_mma_16x32x16_bf16(part[1][0], part[1][1], a10, Sb00, Sb10);
          gdn_mma_16x32x16_bf16(part[1][0], part[1][1], a11, Sb01, Sb11);
          (void)cbar_p;

          threadgroup float *my_f32 = tg_f32 + sg * 1024u;
          #pragma clang loop unroll(full)
          for (uint mm = 0u; mm < 2u; ++mm) {
            #pragma clang loop unroll(full)
            for (uint nn = 0u; nn < 2u; ++nn) {
              #pragma clang loop unroll(full)
              for (uint h = 0u; h < 2u; ++h) {
                const uint r_l = mm * 16u + fm + h * 8u;
                const uint dv_l = nn * 16u + fn;
                *reinterpret_cast<threadgroup float4 *>(my_f32 + r_l * 32u + dv_l) =
                    float4(part[mm][nn][h * 4u + 0u], part[mm][nn][h * 4u + 1u],
                           part[mm][nn][h * 4u + 2u], part[mm][nn][h * 4u + 3u]);
              }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if ((sg >> 1u) == rh) {
            const uint sub_m = (sg & 1u) * 16u;
            #pragma clang loop unroll(full)
            for (uint nn = 0u; nn < 2u; ++nn) {
              #pragma clang loop unroll(full)
              for (uint h = 0u; h < 2u; ++h) {
                const uint idx_flat = (sub_m + fm + h * 8u) * 32u + nn * 16u + fn;
                const float4 s0 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + idx_flat);
                const float4 s1 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + 1024u + idx_flat);
                const float4 s2 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + 2048u + idx_flat);
                const float4 s3 = *reinterpret_cast<const threadgroup float4 *>(tg_f32 + 3072u + idx_flat);
                const float4 tot = (((s0 + s1) + s2) + s3);
                out_sg[vh][nn][h * 4u + 0u] = tot[0];
                out_sg[vh][nn][h * 4u + 1u] = tot[1];
                out_sg[vh][nn][h * 4u + 2u] = tot[2];
                out_sg[vh][nn][h * 4u + 3u] = tot[3];
              }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    };

    vec<float, 8> P_sg[2][2];
    reduce_proj_apple10_dv64(ws_nk, P_sg);

    // Stage Conv1D+SiLU(V) [64 x 64] into tg_bf16[0 .. 4095] if FuseConv.
    threadgroup bfloat *tg_V = tg_bf16;
    if constexpr (FuseConv) {
      const uint dv_l4 = (tid & 15u) * 4u;
      const uint ch0 = v_tg_col_base + dv_l4;
      const uint r_start = (tid >> 4u) * 8u;
      if (params.conv_kernel == 4u && (params.conv_width & 3u) == 0u) {
        float4 xm3 = float4(0.0f), xm2 = float4(0.0f), xm1 = float4(0.0f);
        if (r_start < clen) {
          const int base_tok = int(t0 + r_start);
          xm3 = gdn_read_qkv_or_hist_vec4(qkv_or_conv, history, base_tok - 3, ch0, params.conv_width);
          xm2 = gdn_read_qkv_or_hist_vec4(qkv_or_conv, history, base_tok - 2, ch0, params.conv_width);
          xm1 = gdn_read_qkv_or_hist_vec4(qkv_or_conv, history, base_tok - 1, ch0, params.conv_width);
        }
        #pragma clang loop unroll(full)
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
          *reinterpret_cast<threadgroup bfloat4 *>(tg_V + r * kDvSlice64 + dv_l4) = out_v;
        }
      } else {
        for (uint s = 0u; s < 8u; ++s) {
          const uint r = r_start + s;
          for (uint j = 0u; j < 4u; ++j) {
            tg_V[r * kDvSlice64 + dv_l4 + j] =
                (r < clen)
                    ? gdn_eval_conv_silu_scalar(qkv_or_conv, conv_weight, history,
                                                t0 + r, ch0 + j, params.conv_width,
                                                params.conv_kernel)
                    : bfloat(0.0f);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pack rhs = bfloat(v - pref * P_sg) into 16 uint32 regs (freeing P_sg & tg_V before H_sg!).
    vec<bfloat, 8> rhs_bf[2][2];
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint r = r_sg_base + fm + h * 8u;
      const float pref = (r < clen) ? ctrl_pref[r] : 0.0f;
      #pragma clang loop unroll(full)
      for (uint vh = 0u; vh < 2u; ++vh) {
        #pragma clang loop unroll(full)
        for (uint nn = 0u; nn < 2u; ++nn) {
          const uint dv_base = vh * 32u + nn * 16u + fn;
          if (r < clen) {
            const bfloat4 v4 = FuseConv
                ? *reinterpret_cast<const threadgroup bfloat4 *>(tg_V + r * kDvSlice64 + dv_base)
                : *reinterpret_cast<device const bfloat4 *>(
                      qkv_or_conv + ulong(t0 + r) * ulong(params.conv_width) +
                      ulong(v_tg_col_base + dv_base));
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) {
              const uint e = h * 4u + j;
              rhs_bf[vh][nn][e] = bfloat(fma(-pref, P_sg[vh][nn][e], float(v4[j])));
            }
          } else {
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) {
              rhs_bf[vh][nn][h * 4u + j] = bfloat(0.0f);
            }
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    vec<float, 8> H_sg[2][2];
    reduce_proj_apple10_dv64(ws_nq, H_sg);

    // Stage tg_rhsT [64 x 72] into tg_bf16[0 .. 4607] and scale H_sg by prefix[r].
    threadgroup bfloat *tg_rhsT = tg_bf16;
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint r = r_sg_base + fm + h * 8u;
      const float pref = (r < clen) ? ctrl_pref[r] : 0.0f;
      #pragma clang loop unroll(full)
      for (uint vh = 0u; vh < 2u; ++vh) {
        #pragma clang loop unroll(full)
        for (uint nn = 0u; nn < 2u; ++nn) {
          const uint dv_base = vh * 32u + nn * 16u + fn;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            const uint e = h * 4u + j;
            tg_rhsT[(dv_base + j) * kRhsPad + r] = rhs_bf[vh][nn][e];
            H_sg[vh][nn][e] = (r < clen) ? (H_sg[vh][nn][e] * pref) : 0.0f;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: delta = solve @ rhs (causal kt <= sg, 10/16 blocks, ctrl_solve loaded ONCE!)
    vec<float, 8> D_sg[2][2] = {
        {vec<float, 8>(0.0f), vec<float, 8>(0.0f)},
        {vec<float, 8>(0.0f), vec<float, 8>(0.0f)}};
    if (r_sg_base < clen) {
      for (uint kt = 0u; kt <= sg; ++kt) {
        if (kt * 16u < clen) {
          const uint k_off = kt * 16u;
          volatile int cbar_d;
          const vec<bfloat, 8> a_sol =
              gdn_load_frag_dev_bf16(ctrl_solve + r_sg_base * kChunkSize + k_off, kChunkSize, fm, fn);
          #pragma clang loop unroll(full)
          for (uint vh = 0u; vh < 2u; ++vh) {
            const vec<bfloat, 8> b_r0 =
                gdn_load_frag_tg_bf16<kRhsPad>(tg_rhsT + (vh * 32u) * kRhsPad + k_off, fm, fn);
            const vec<bfloat, 8> b_r1 =
                gdn_load_frag_tg_bf16<kRhsPad>(tg_rhsT + (vh * 32u + 16u) * kRhsPad + k_off, fm, fn);
            gdn_mma_16x32x16_bf16(D_sg[vh][0], D_sg[vh][1], a_sol, b_r0, b_r1);
          }
          (void)cbar_d;
        }
      }
    }

    vec<bfloat, 8> dsuff_bf[2][2];
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint r = r_sg_base + fm + h * 8u;
      const float suff = (r < clen) ? ctrl_suff[r] : 0.0f;
      #pragma clang loop unroll(full)
      for (uint vh = 0u; vh < 2u; ++vh) {
        #pragma clang loop unroll(full)
        for (uint nn = 0u; nn < 2u; ++nn) {
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            const uint e = h * 4u + j;
            dsuff_bf[vh][nn][e] = (r < clen) ? bfloat(suff * D_sg[vh][nn][e]) : bfloat(0.0f);
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reuse tg_bf16[0 .. 4607] for tg_deltaT [64 x 72].
    threadgroup bfloat *tg_deltaT = tg_bf16;
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint r = r_sg_base + fm + h * 8u;
      #pragma clang loop unroll(full)
      for (uint vh = 0u; vh < 2u; ++vh) {
        #pragma clang loop unroll(full)
        for (uint nn = 0u; nn < 2u; ++nn) {
          const uint dv_base = vh * 32u + nn * 16u + fn;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            const uint e = h * 4u + j;
            tg_deltaT[(dv_base + j) * kRhsPad + r] =
                (r < clen) ? bfloat(D_sg[vh][nn][e]) : bfloat(0.0f);
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: c_out = H_sg + mqk @ tg_deltaT (causal kt <= sg, ctrl_mqk loaded ONCE!) -> coalesced bfloat4 store!
    if (r_sg_base < clen) {
      for (uint kt = 0u; kt <= sg; ++kt) {
        if (kt * 16u < clen) {
          const uint k_off = kt * 16u;
          volatile int cbar_c;
          const vec<bfloat, 8> a_mqk =
              gdn_load_frag_dev_bf16(ctrl_mqk + r_sg_base * kChunkSize + k_off, kChunkSize, fm, fn);
          #pragma clang loop unroll(full)
          for (uint vh = 0u; vh < 2u; ++vh) {
            const vec<bfloat, 8> b_d0 =
                gdn_load_frag_tg_bf16<kRhsPad>(tg_deltaT + (vh * 32u) * kRhsPad + k_off, fm, fn);
            const vec<bfloat, 8> b_d1 =
                gdn_load_frag_tg_bf16<kRhsPad>(tg_deltaT + (vh * 32u + 16u) * kRhsPad + k_off, fm, fn);
            gdn_mma_16x32x16_bf16(H_sg[vh][0], H_sg[vh][1], a_mqk, b_d0, b_d1);
          }
          (void)cbar_c;
        }
      }
      #pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const uint r = r_sg_base + fm + h * 8u;
        if (r < clen) {
          #pragma clang loop unroll(full)
          for (uint vh = 0u; vh < 2u; ++vh) {
            #pragma clang loop unroll(full)
            for (uint nn = 0u; nn < 2u; ++nn) {
              const bfloat4 out4 = bfloat4(
                  bfloat(H_sg[vh][nn][h * 4u + 0u]),
                  bfloat(H_sg[vh][nn][h * 4u + 1u]),
                  bfloat(H_sg[vh][nn][h * 4u + 2u]),
                  bfloat(H_sg[vh][nn][h * 4u + 3u]));
              *reinterpret_cast<device bfloat4 *>(
                  out_bf16 + ulong(t0 + r) * ulong(value_width) +
                  ulong(out_tg_col_base + vh * 32u + nn * 16u + fn)) = out4;
            }
          }
        }
      }
    }

    // Step 4: S = decay_last * S + tg_deltaSuffT @ nk (exact 15.0 KiB <= 16.0 KiB!)
    #pragma clang loop unroll(full)
    for (uint vh = 0u; vh < 2u; ++vh) {
      S[vh][0][0] *= decay_last;
      S[vh][0][1] *= decay_last;
      S[vh][1][0] *= decay_last;
      S[vh][1][1] *= decay_last;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup bfloat *tg_deltaSuffT = tg_bf16;          // [0 .. 4607] (64 * 72)
    threadgroup bfloat *tg_nkT = tg_bf16 + 4608u;         // [4608 .. 7679] (128 * 24)
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint r = r_sg_base + fm + h * 8u;
      #pragma clang loop unroll(full)
      for (uint vh = 0u; vh < 2u; ++vh) {
        #pragma clang loop unroll(full)
        for (uint nn = 0u; nn < 2u; ++nn) {
          const uint dv_base = vh * 32u + nn * 16u + fn;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 4u; ++j) {
            tg_deltaSuffT[(dv_base + j) * kRhsPad + r] = dsuff_bf[vh][nn][h * 4u + j];
          }
        }
      }
    }

    const uint t_r = tid >> 3u;
    const uint t_dk0 = (tid & 7u) * 16u;
    const uint n_rs = has_hi ? 4u : 2u;
    for (uint rs = 0u; rs < n_rs; ++rs) {
      if (rs > 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      const uint r_src = rs * 16u + t_r;
      device const bfloat4 *nk_src4 =
          reinterpret_cast<device const bfloat4 *>(ws_nk + r_src * kKeyDim + t_dk0);
      #pragma clang loop unroll(full)
      for (uint v4 = 0u; v4 < 4u; ++v4) {
        const bfloat4 w4 = nk_src4[v4];
        const uint dk_b = t_dk0 + v4 * 4u;
        tg_nkT[(dk_b + 0u) * kNkTPad16 + t_r] = w4[0];
        tg_nkT[(dk_b + 1u) * kNkTPad16 + t_r] = w4[1];
        tg_nkT[(dk_b + 2u) * kNkTPad16 + t_r] = w4[2];
        tg_nkT[(dk_b + 3u) * kNkTPad16 + t_r] = w4[3];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const uint r_sub_ds = rs * 16u;
      volatile int cbar_s;
      const vec<bfloat, 8> b_nk0 =
          gdn_load_frag_tg_bf16<kNkTPad16>(tg_nkT + dk_sg_base * kNkTPad16, fm, fn);
      const vec<bfloat, 8> b_nk1 =
          gdn_load_frag_tg_bf16<kNkTPad16>(tg_nkT + (dk_sg_base + 16u) * kNkTPad16, fm, fn);
      #pragma clang loop unroll(full)
      for (uint vh = 0u; vh < 2u; ++vh) {
        const vec<bfloat, 8> a_ds0 =
            gdn_load_frag_tg_bf16<kRhsPad>(tg_deltaSuffT + (vh * 32u) * kRhsPad + r_sub_ds, fm, fn);
        const vec<bfloat, 8> a_ds1 =
            gdn_load_frag_tg_bf16<kRhsPad>(tg_deltaSuffT + (vh * 32u + 16u) * kRhsPad + r_sub_ds, fm, fn);
        gdn_mma_16x32x16_bf16(S[vh][0][0], S[vh][0][1], a_ds0, b_nk0, b_nk1);
        gdn_mma_16x32x16_bf16(S[vh][1][0], S[vh][1][1], a_ds1, b_nk0, b_nk1);
      }
      (void)cbar_s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Coalesced float4 writeback of persistent state S[2][2][2].
  #pragma clang loop unroll(full)
  for (uint vh = 0u; vh < 2u; ++vh) {
    #pragma clang loop unroll(full)
    for (uint vs = 0u; vs < 2u; ++vs) {
      #pragma clang loop unroll(full)
      for (uint ks = 0u; ks < 2u; ++ks) {
        const uint dk_col = dk_sg_base + ks * 16u + fn;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint dv_l = vh * 32u + vs * 16u + fm + h * 8u;
          *reinterpret_cast<device float4 *>(
              state + ulong(out_tg_col_base + dv_l) * kKeyDim + dk_col) =
              float4(S[vh][vs][ks][h * 4u + 0u], S[vh][vs][ks][h * 4u + 1u],
                     S[vh][vs][ks][h * 4u + 2u], S[vh][vs][ks][h * 4u + 3u]);
        }
      }
    }
  }
}

} // namespace

// Static TG SRAM = 64 * 128 * 2 B = 16,384 B (16.00 KiB <= 32,768 B)
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
  threadgroup bfloat tg_buf[kChunkSize * kKeyDim];
  gdn_chunk_prepare_c64_impl<false>(
      convolved, nullptr, nullptr, a, b, a_log, dt_bias, qk_ws, ctrl_ws, params,
      tg_buf, lane, sg, tg);
}

// Static TG SRAM = 16,384 B (16.00 KiB <= 32,768 B)
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
  threadgroup bfloat tg_buf[kChunkSize * kKeyDim];
  gdn_chunk_prepare_c64_impl<true>(
      qkv, conv_weight, history, a, b, a_log, dt_bias, qk_ws, ctrl_ws, params,
      tg_buf, lane, sg, tg);
}

// Static TG SRAM = 4096 * 4 B = 16,384 B (16.00 KiB <= 32,768 B; 128 threads/TG, grid=(Hv, 4))
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
  alignas(16) threadgroup float tg_f32[4096u];
  gdn_chunk_recurrence_dv32_c64_impl<false>(
      convolved, nullptr, nullptr, qk_ws, ctrl_ws, state, recurrent, params,
      tg_f32, lane, sg, tg);
}

// Static TG SRAM = 4096 * 4 B = 16,384 B (16.00 KiB <= 32,768 B; 128 threads/TG, grid=(Hv, 4))
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
  alignas(16) threadgroup float tg_f32[4096u];
  gdn_chunk_recurrence_dv32_c64_impl<true>(
      qkv, conv_weight, history, qk_ws, ctrl_ws, state, recurrent_out, params,
      tg_f32, lane, sg, tg);
}

// Static TG SRAM = 4096 * 4 B = 16,384 B (16.00 KiB <= 32,768 B; 128 threads/TG, grid=(Hv, 2))
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
  alignas(16) threadgroup float tg_f32[4096u];
  gdn_chunk_recurrence_dv64_c64_impl<false>(
      convolved, nullptr, nullptr, qk_ws, ctrl_ws, state, recurrent, params,
      tg_f32, lane, sg, tg);
}

// Static TG SRAM = 4096 * 4 B = 16,384 B (16.00 KiB <= 32,768 B; 128 threads/TG, grid=(Hv, 2))
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
  alignas(16) threadgroup float tg_f32[4096u];
  gdn_chunk_recurrence_dv64_c64_impl<true>(
      qkv, conv_weight, history, qk_ws, ctrl_ws, state, recurrent_out, params,
      tg_f32, lane, sg, tg);
}

// 0 B TG SRAM, 4-SG (128-thread) SLC-resident in-place 0-ULP RMSNorm_128 * SiLU(z) -> BF16
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

  const uint f0 = lane;
  const uint f1 = lane + 32u;
  const uint f2 = lane + 64u;
  const uint f3 = lane + 96u;

  const float v0 = float(row[f0]);
  const float v1 = float(row[f1]);
  const float v2 = float(row[f2]);
  const float v3 = float(row[f3]);
  float sq = fma(v0, v0, 0.0f);
  sq = fma(v1, v1, sq);
  sq = fma(v2, v2, sq);
  sq = fma(v3, v3, sq);
  const float inv = rsqrt(simd_sum(sq) / 128.0f + params.epsilon);

  const float g0 = float(gate[f0]);
  const float g1 = float(gate[f1]);
  const float g2 = float(gate[f2]);
  const float g3 = float(gate[f3]);

  const float nw0 = float(norm_weight[f0]);
  const float nw1 = float(norm_weight[f1]);
  const float nw2 = float(norm_weight[f2]);
  const float nw3 = float(norm_weight[f3]);

  row[f0] = bfloat((v0 * inv * nw0) * g0 / (1.0f + exp(-g0)));
  row[f1] = bfloat((v1 * inv * nw1) * g1 / (1.0f + exp(-g1)));
  row[f2] = bfloat((v2 * inv * nw2) * g2 / (1.0f + exp(-g2)));
  row[f3] = bfloat((v3 * inv * nw3) * g3 / (1.0f + exp(-g3)));
}

// 0 B TG SRAM, 4-SG (128-thread) SLC-resident in-place 0-ULP RMSNorm_128 * SiLU(z) -> FP16
// with exact [params.tokens .. min(params.mp, ceil64(params.tokens))) zero-padding.
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

  const uint f0 = lane;
  const uint f1 = lane + 32u;
  const uint f2 = lane + 64u;
  const uint f3 = lane + 96u;

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

  const float v0 = float(row_bf16[f0]);
  const float v1 = float(row_bf16[f1]);
  const float v2 = float(row_bf16[f2]);
  const float v3 = float(row_bf16[f3]);
  float sq = fma(v0, v0, 0.0f);
  sq = fma(v1, v1, sq);
  sq = fma(v2, v2, sq);
  sq = fma(v3, v3, sq);
  const float inv = rsqrt(simd_sum(sq) / 128.0f + params.epsilon);

  const float g0 = float(gate[f0]);
  const float g1 = float(gate[f1]);
  const float g2 = float(gate[f2]);
  const float g3 = float(gate[f3]);

  const float nw0 = float(norm_weight[f0]);
  const float nw1 = float(norm_weight[f1]);
  const float nw2 = float(norm_weight[f2]);
  const float nw3 = float(norm_weight[f3]);

  const bfloat o0 = bfloat((v0 * inv * nw0) * g0 / (1.0f + exp(-g0)));
  const bfloat o1 = bfloat((v1 * inv * nw1) * g1 / (1.0f + exp(-g1)));
  const bfloat o2 = bfloat((v2 * inv * nw2) * g2 / (1.0f + exp(-g2)));
  const bfloat o3 = bfloat((v3 * inv * nw3) * g3 / (1.0f + exp(-g3)));

  dst_f16[f0] = half(clamp(float(o0), -65504.0f, 65504.0f));
  dst_f16[f1] = half(clamp(float(o1), -65504.0f, 65504.0f));
  dst_f16[f2] = half(clamp(float(o2), -65504.0f, 65504.0f));
  dst_f16[f3] = half(clamp(float(o3), -65504.0f, 65504.0f));
}

// Static TG SRAM = 16,384 B (16.00 KiB <= 32,768 B)
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
  threadgroup bfloat tg_work[8u * kChunkSize * kValSgDim];
  gdn_chunk_recurrence_c64_impl<true, 1u>(
      qkv, conv_weight, history, qk_ws, ctrl_ws, state, z, norm_weight,
      output_bf16, nullptr, params, tg_work, lane, sg, tg);
}

// Static TG SRAM = 16,384 B (16.00 KiB <= 32,768 B)
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
  threadgroup bfloat tg_work[8u * kChunkSize * kValSgDim];
  gdn_chunk_recurrence_c64_impl<true, 2u>(
      qkv, conv_weight, history, qk_ws, ctrl_ws, state, z, norm_weight,
      nullptr, output_f16, params, tg_work, lane, sg, tg);
}
