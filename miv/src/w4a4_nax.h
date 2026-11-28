#pragma once

#include "mxfp4_nax_cache.h"
#include "nax_common.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace omlx::m5::detail::forward {

inline constexpr U32 kW4A4DefaultMinTokens = 32U;

[[nodiscard]] inline U32 w4a4_nax_mp(U32 t) noexcept {
  return ((t + 63u) / 64u) * 64u;
}

[[nodiscard]] inline U32 w4a4_nax_min_tokens() noexcept {
  if (const char *env = std::getenv("OMLX_M5_W4A4_MIN_TOKENS")) {
    const unsigned long parsed = std::strtoul(env, nullptr, 10);
    if (parsed >= 1UL && parsed <= 4096UL) {
      return static_cast<U32>(parsed);
    }
  }
  return kW4A4DefaultMinTokens;
}

// Default-ON hybrid W4A4 policy (promoted by Phase 3 M5 Ultra precision gate):
// keep full-attention q_proj and k_proj on W4A8 while keeping MLP / GDN /
// o_proj / v_proj on W4A4. Cuts 16K PPL 6.96x (37.73 -> 5.42 Exact, 15.28 ->
// 7.76 Sage2+MInf) while running +0.4% to +0.8% faster at 16K/256K.
// Set OMLX_M5_W4A4_KEEP_QK_A8=0 to explicitly force Pure W4A4 for A/B studies.
[[nodiscard]] inline bool w4a4_keep_qk_a8_enabled() noexcept {
  if (const char *env = std::getenv("OMLX_M5_W4A4_KEEP_QK_A8")) {
    const std::string_view sv(env);
    if (sv == "0" || sv == "false" || sv == "FALSE" || sv == "off" ||
        sv == "OFF") {
      return false;
    }
    return true;
  }
  return true;
}

[[nodiscard]] inline bool w4a4_nax_runtime_eligible(
    const MetalContext &ctx, const std::filesystem::path &libs) {
  return nax_apple10_eligible(ctx, libs, "omlx_m5_w4a4_nax.metallib",
                              "OMLX_M5_DISABLE_W4A4_NAX");
}

struct W4A4WorkspaceLayout {
  U32 mp = 0;
  U32 max_k = 0;
  U32 max_mid_k = 0;
  U64 codes_bytes = 0;
  U64 scales_bytes = 0;
  U64 mid_codes_bytes = 0;
  U64 mid_scales_bytes = 0;
  U64 total_bytes = 0;
};

[[nodiscard]] inline W4A4WorkspaceLayout
w4a4_nax_workspace_layout(U32 max_rows, U32 max_k, U32 max_mid_k = 0u) {
  const U32 mp = w4a4_nax_mp(max_rows);
  const U32 aligned_k = ((max_k + 127u) / 128u) * 128u;
  const U32 aligned_mid_k =
      ((std::max(max_mid_k, max_k) + 127u) / 128u) * 128u;
  const U64 tiles_m = U64(mp / 16u);
  const U64 codes_bytes = mul(mul(tiles_m, U64(aligned_k / 128u)), 1024ULL);
  const U64 scales_bytes = mul(mul(tiles_m, U64(aligned_k / 128u)), 64ULL);
  const U64 mid_codes_bytes =
      mul(mul(tiles_m, U64(aligned_mid_k / 128u)), 1024ULL);
  const U64 mid_scales_bytes =
      mul(mul(tiles_m, U64(aligned_mid_k / 128u)), 64ULL);
  return {.mp = mp,
          .max_k = aligned_k,
          .max_mid_k = aligned_mid_k,
          .codes_bytes = codes_bytes,
          .scales_bytes = scales_bytes,
          .mid_codes_bytes = mid_codes_bytes,
          .mid_scales_bytes = mid_scales_bytes,
          .total_bytes =
              codes_bytes + scales_bytes + mid_codes_bytes + mid_scales_bytes};
}

// Single-pass RMSNorm + Tiled MXFP4 A4 dynamic quantization (zeroes [t, mp) tail rows).
inline void emit_rmsnorm_a4_bm16_bk128(Encoder &e, Ref input, Ref weight,
                                       Ref opt_out_bf16, Ref a4_codes,
                                       Ref a4_scales, U32 k, float eps,
                                       float weight_offset,
                                       bool write_bf16 = false,
                                       std::optional<U32> explicit_t = {},
                                       std::optional<U32> mp_override = {}) {
  if ((k % 128u) != 0u || k == 0u) {
    throw std::invalid_argument(
        "emit_rmsnorm_a4_bm16_bk128 requires k % 128 == 0");
  }
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mp_override.value_or(w4a4_nax_mp(t));
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() || t == 1u ? std::nullopt : std::optional<U32>(1u);
  auto &d = e.emit("w4a4_nax", "omlx_m5_rmsnorm_a4_bm16_bk128",
                   {.width = 128u, .height = mp}, {.width = 128u},
                   {input, weight, opt_out_bf16, a4_codes, a4_scales});
  e.dynamic(d, 5,
            words({k, t, mp, word(eps), word(weight_offset),
                   write_bf16 ? 1u : 0u}),
            tokens_patch);
}

// Single-pass Residual Add + RMSNorm + Tiled MXFP4 A4 dynamic quantization.
inline void emit_add_rmsnorm_a4_bm16_bk128(
    Encoder &e, Ref hidden, Ref delta, Ref weight, Ref opt_out_bf16,
    Ref a4_codes, Ref a4_scales, U32 k, float eps, float weight_offset,
    bool write_bf16 = false, std::optional<U32> explicit_t = {},
    std::optional<U32> mp_override = {}) {
  if ((k % 128u) != 0u || k == 0u) {
    throw std::invalid_argument(
        "emit_add_rmsnorm_a4_bm16_bk128 requires k % 128 == 0");
  }
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mp_override.value_or(w4a4_nax_mp(t));
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() || t == 1u ? std::nullopt : std::optional<U32>(1u);
  auto &d = e.emit("w4a4_nax", "omlx_m5_add_rmsnorm_a4_bm16_bk128",
                   {.width = 128u, .height = mp}, {.width = 128u},
                   {hidden, delta, weight, opt_out_bf16, a4_codes, a4_scales});
  e.dynamic(d, 6,
            words({k, t, mp, word(eps), word(weight_offset),
                   write_bf16 ? 1u : 0u}),
            tokens_patch);
}

// Standalone BF16 -> Tiled MXFP4 A4 quantization (with optional fused FWHT-64).
inline void emit_quantize_a4_bm16_bk128(Encoder &e, Ref input, Ref a4_codes,
                                        Ref a4_scales, U32 k,
                                        std::optional<U32> explicit_t = {},
                                        std::optional<U32> mp_override = {},
                                        bool fwht64 = false) {
  if ((k % 128u) != 0u || k == 0u) {
    throw std::invalid_argument(
        "emit_quantize_a4_bm16_bk128 requires k % 128 == 0");
  }
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mp_override.value_or(w4a4_nax_mp(t));
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  const char *fn = fwht64 ? "omlx_m5_quantize_a4_bm16_bk128_fwht64"
                          : "omlx_m5_quantize_a4_bm16_bk128";
  const U32 groups64 = k / 64u;
  const U32 x_tgs = (groups64 + 3u) / 4u;
  auto &d = e.emit("w4a4_nax", fn,
                   {.width = U64(x_tgs) * 128u, .height = mp}, {.width = 128u},
                   {input, a4_codes, a4_scales});
  e.dynamic(d, 3, words({k, mp, 0u, t}), tokens_patch);
}

// Single-pass GDN Stage 3: RMSNorm_128(inout_bf16) * SiLU(z) -> Tiled MXFP4 A4 + optional BF16.
inline void emit_gdn_norm_gate_inplace_a4_bm16_bk128_c64(
    Encoder &e, U32 capacity, U32 key_heads, U32 value_heads, U32 conv_width,
    Ref inout_bf16, Ref z, Ref norm_weight, Ref a4_codes, Ref a4_scales,
    float eps, bool write_bf16 = false, bool dynamic_tokens = true,
    std::optional<U32> mp_override = {}) {
  const U32 mp = mp_override.value_or(w4a4_nax_mp(capacity));
  const U64 total_heads = U64(mp) * U64(value_heads);
  const std::optional<U32> tok_word =
      dynamic_tokens ? std::optional<U32>{6U} : std::optional<U32>{};
  auto &d = e.emit(
      "w4a4_nax", "omlx_m5_gdn_chunk_norm_gate_inplace_a4_bm16_bk128_c64",
      {.width = 128u, .height = (total_heads + 3u) / 4u}, {.width = 128u},
      {inout_bf16, z, norm_weight, a4_codes, a4_scales});
  e.dynamic(d, 5,
            words({key_heads, value_heads, 128u, 128u, conv_width, 4u, capacity,
                   word(eps), mp, write_bf16 ? 1u : 0u}),
            tok_word);
}

// Dense Native MXFP4 W4A4 Linear (BM=64, BN=64, BK=128, bns16/bns32, Dual-ANE row_offset).
inline void emit_mxfp4_w4a4_nax_linear(Encoder &e, const TiledMxfp4Weight &tw,
                                       Ref a4_codes, Ref a4_scales, Ref y,
                                       std::optional<U32> explicit_t = {},
                                       U32 row_offset = 0u, U32 out_stride = 0u,
                                       U32 slice_n = 0u,
                                       std::optional<U32> mp_override = {}) {
  if ((tw.k % 128u) != 0u || (row_offset % 64u) != 0u || row_offset >= tw.n) {
    throw std::invalid_argument(
        "emit_mxfp4_w4a4_nax_linear requires k % 128 == 0, row_offset % 64 == 0, row_offset < n");
  }
  const U32 eff_n = (slice_n > 0u) ? slice_n : (tw.n - row_offset);
  if (row_offset + eff_n > tw.n) {
    throw std::invalid_argument("emit_mxfp4_w4a4_nax_linear slice exceeds weight n");
  }
  const U32 eff_stride = (out_stride > 0u) ? out_stride : tw.n;
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mp_override.value_or(w4a4_nax_mp(t));
  const std::string fn = "omlx_m5_mxfp4_w4a4_nax_bm64_bn64_bns" +
                         std::to_string(tw.bn) + "_sk1";
  const U32 x_tiles = (eff_n + 63u) / 64u;
  const U32 y_tiles = (t + 63u) / 64u;
  const MetalDispatchSize grid{.width = U64(x_tiles) * 128u, .height = y_tiles};
  const MetalDispatchSize threads{.width = 128u};
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  auto &d = e.emit("w4a4_nax", fn, grid, threads,
                   {a4_codes, a4_scales, tw.codes, tw.scales, y});
  e.dynamic(d, 5,
            words({eff_n, tw.k, mp, t, tw.padded_n, row_offset, eff_stride, 0u}),
            tokens_patch);
  e.dynamic(d, 6, nax_mxfp4_byte_lut_sh_bytes());
}

// Mode A Dense Native MXFP4 W4A4 Fused Gate + Up + SwiGLU -> BF16.
inline void emit_mxfp4_w4a4_nax_swiglu(Encoder &e,
                                       const TiledMxfp4Weight &gw,
                                       const TiledMxfp4Weight &uw,
                                       Ref a4_codes, Ref a4_scales, Ref y,
                                       std::optional<U32> explicit_t = {},
                                       U32 row_offset = 0u,
                                       U32 out_stride = 0u,
                                       U32 slice_n = 0u,
                                       std::optional<U32> mp_override = {}) {
  if (gw.n != uw.n || gw.k != uw.k || gw.bn != 16u || uw.bn != 16u ||
      (gw.k % 128u) != 0u || (row_offset % 64u) != 0u || row_offset >= gw.n) {
    throw std::invalid_argument(
        "emit_mxfp4_w4a4_nax_swiglu requires matching bn=16 weights, k % 128 == 0, row_offset % 64 == 0");
  }
  const U32 eff_n = (slice_n > 0u) ? slice_n : (gw.n - row_offset);
  if (row_offset + eff_n > gw.n) {
    throw std::invalid_argument("emit_mxfp4_w4a4_nax_swiglu slice exceeds weight n");
  }
  const U32 eff_stride = (out_stride > 0u) ? out_stride : gw.n;
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mp_override.value_or(w4a4_nax_mp(t));
  const U32 x_tiles = (eff_n + 31u) / 32u;
  const U32 y_tiles = (t + 63u) / 64u;
  const MetalDispatchSize grid{.width = U64(x_tiles) * 128u, .height = y_tiles};
  const MetalDispatchSize threads{.width = 128u};
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  auto &d = e.emit("w4a4_nax",
                   "omlx_m5_mxfp4_w4a4_nax_swiglu_bm64_bn64_out64_sk1",
                   grid, threads,
                   {a4_codes, a4_scales, gw.codes, gw.scales, uw.codes,
                    uw.scales, y});
  e.dynamic(d, 7,
            words({eff_n, gw.k, mp, t, gw.padded_n, row_offset, eff_stride, 0u}),
            tokens_patch);
  e.dynamic(d, 8, nax_mxfp4_byte_lut_sh_bytes());
}

// Mode B Dense Native MXFP4 W4A4 Fused Gate + Up + SwiGLU + FWHT-64 + Tiled MXFP4 A4.
// Supports WP 3C ANE splice (`row_offset % 128 == 0`, `out_stride % 128 == 0`, `slice_n % 128 == 0`).
inline void emit_mxfp4_w4a4_nax_swiglu_fwht64_a4(
    Encoder &e, const TiledMxfp4Weight &gw, const TiledMxfp4Weight &uw,
    Ref a4_codes, Ref a4_scales, Ref a4_mid_codes, Ref a4_mid_scales,
    Ref opt_y_bf16, bool also_write_bf16, std::optional<U32> explicit_t = {},
    U32 row_offset = 0u, U32 out_stride = 0u, U32 slice_n = 0u,
    std::optional<U32> mp_override = {}) {
  const U32 eff_stride = (out_stride > 0u) ? out_stride : gw.n;
  const U32 eff_n = (slice_n > 0u) ? slice_n : (gw.n - row_offset);
  if (gw.n != uw.n || gw.k != uw.k || gw.bn != 16u || uw.bn != 16u ||
      (gw.k % 128u) != 0u || (eff_stride % 128u) != 0u ||
      (row_offset % 128u) != 0u || row_offset >= gw.n ||
      row_offset + eff_n > gw.n || (eff_n % 64u) != 0u) {
    throw std::invalid_argument(
        "emit_mxfp4_w4a4_nax_swiglu_fwht64_a4 requires bn=16, k%128==0, out_stride%128==0, row_offset%128==0, slice_n%64==0");
  }
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mp_override.value_or(w4a4_nax_mp(t));
  const U32 x_tiles = eff_n / 64u;
  const U32 y_tiles = mp / 64u;
  const MetalDispatchSize grid{.width = U64(x_tiles) * 128u, .height = y_tiles};
  const MetalDispatchSize threads{.width = 128u};
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  const U32 flags = also_write_bf16 ? 1u : 0u;
  auto &d = e.emit(
      "w4a4_nax",
      "omlx_m5_mxfp4_w4a4_nax_swiglu_bm64_bn64_out64_sk1_fwht64_a4", grid,
      threads,
      {a4_codes, a4_scales, gw.codes, gw.scales, uw.codes, uw.scales,
       a4_mid_codes, a4_mid_scales, opt_y_bf16});
  e.dynamic(d, 9,
            words({eff_n, gw.k, mp, t, gw.padded_n, row_offset, eff_stride,
                   flags}),
            tokens_patch);
  e.dynamic(d, 10, nax_mxfp4_byte_lut_sh_bytes());
}

} // namespace omlx::m5::detail::forward
