#pragma once

#include "affine_nax_cache.h"
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

inline constexpr U32 kW4A8DefaultMinTokens = 32U;

[[nodiscard]] inline U32 w4a8_nax_mp(U32 t) noexcept {
  return ((t + 63u) / 64u) * 64u;
}

[[nodiscard]] inline U32 w4a8_nax_min_tokens() noexcept {
  if (const char *env = std::getenv("OMLX_M5_W4A8_MIN_TOKENS")) {
    const unsigned long parsed = std::strtoul(env, nullptr, 10);
    if (parsed >= 1UL && parsed <= 4096UL) {
      return static_cast<U32>(parsed);
    }
  }
  return kW4A8DefaultMinTokens;
}

[[nodiscard]] inline bool w4a8_down_proj_enabled() noexcept {
  if (const char *env = std::getenv("OMLX_M5_W4A8_DOWN_PROJ");
      env != nullptr && std::string_view(env) == "1") {
    return true;
  }
  return false;
}

[[nodiscard]] inline bool w4a8_nax_runtime_eligible(
    const MetalContext &ctx, const std::filesystem::path &libs) {
  return nax_apple10_eligible(ctx, libs, "omlx_m5_w4a8_nax.metallib",
                              "OMLX_M5_DISABLE_W4A8_NAX");
}

struct W4A8WorkspaceLayout {
  U32 mp = 0;
  U32 max_k = 0;
  U64 qa_bytes = 0;
  U64 sa_bytes = 0;
  U64 ra_bytes = 0;
  U64 total_bytes = 0;
};

[[nodiscard]] inline W4A8WorkspaceLayout
w4a8_nax_workspace_layout(U32 max_rows, U32 max_k) {
  const U32 mp = w4a8_nax_mp(max_rows);
  const U32 aligned_k = ((max_k + 63u) / 64u) * 64u;
  const U32 groups = aligned_k / 64u;
  const U64 qa_bytes = mul(U64(mp), U64(aligned_k));
  const U64 sa_bytes = mul(mul(U64(groups), U64(mp)), 4ULL);
  const U64 ra_bytes = sa_bytes;
  return {.mp = mp,
          .max_k = aligned_k,
          .qa_bytes = qa_bytes,
          .sa_bytes = sa_bytes,
          .ra_bytes = ra_bytes,
          .total_bytes = qa_bytes + sa_bytes + ra_bytes};
}

// Single-pass RMSNorm + GS64 INT8 dynamic quantization + Ra.
inline void emit_rmsnorm_a8_g64(Encoder &e, Ref input, Ref weight,
                                Ref opt_out_bf16, Ref qa_out, Ref sa_out,
                                Ref ra_out, U32 k, float eps,
                                float weight_offset, bool write_bf16 = false,
                                std::optional<U32> explicit_t = {},
                                std::optional<U32> mp_override = {}) {
  if ((k % 64u) != 0u || k == 0u) {
    throw std::invalid_argument("emit_rmsnorm_a8_g64 requires k % 64 == 0");
  }
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mp_override.value_or(w4a8_nax_mp(t));
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() || t == 1u ? std::nullopt : std::optional<U32>(1u);
  auto &d = e.emit("w4a8_nax", "omlx_m5_rmsnorm_a8_g64",
                   {.width = 128u, .height = mp}, {.width = 128u},
                   {input, weight, opt_out_bf16, qa_out, sa_out, ra_out});
  e.dynamic(d, 6,
            words({k, t, mp, word(eps), word(weight_offset),
                   write_bf16 ? 1u : 0u}),
            tokens_patch);
}

// Single-pass Residual Add + RMSNorm + GS64 INT8 dynamic quantization + Ra.
inline void emit_add_rmsnorm_a8_g64(Encoder &e, Ref hidden, Ref delta,
                                    Ref weight, Ref opt_out_bf16, Ref qa_out,
                                    Ref sa_out, Ref ra_out, U32 k, float eps,
                                    float weight_offset,
                                    bool write_bf16 = false,
                                    std::optional<U32> explicit_t = {},
                                    std::optional<U32> mp_override = {}) {
  if ((k % 64u) != 0u || k == 0u) {
    throw std::invalid_argument("emit_add_rmsnorm_a8_g64 requires k % 64 == 0");
  }
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mp_override.value_or(w4a8_nax_mp(t));
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() || t == 1u ? std::nullopt : std::optional<U32>(1u);
  auto &d = e.emit("w4a8_nax", "omlx_m5_add_rmsnorm_a8_g64",
                   {.width = 128u, .height = mp}, {.width = 128u},
                   {hidden, delta, weight, opt_out_bf16, qa_out, sa_out,
                    ra_out});
  e.dynamic(d, 7,
            words({k, t, mp, word(eps), word(weight_offset),
                   write_bf16 ? 1u : 0u}),
            tokens_patch);
}

// Standalone BF16 -> GS64 (Qa, Sa, Ra) quantization (e.g. Full-Attn mix -> o_proj).
inline void emit_quantize_a8_g64(Encoder &e, Ref input, Ref qa_out, Ref sa_out,
                                 Ref ra_out, U32 k,
                                 std::optional<U32> explicit_t = {},
                                 std::optional<U32> mp_override = {}) {
  if ((k % 64u) != 0u || k == 0u) {
    throw std::invalid_argument("emit_quantize_a8_g64 requires k % 64 == 0");
  }
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mp_override.value_or(w4a8_nax_mp(t));
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  auto &d = e.emit("w4a8_nax", "omlx_m5_quantize_a8_g64",
                   {.width = U64(k / 64u) * 32u, .height = mp}, {.width = 32u},
                   {input, qa_out, sa_out, ra_out});
  e.dynamic(d, 4, words({k, mp, 0u, t}), tokens_patch);
}

// Single-pass GDN Stage 3: RMSNorm_128(inout_bf16) * SiLU(z) -> (Qa, Sa, Ra) + optional BF16.
inline void emit_gdn_norm_gate_inplace_a8_g64(
    Encoder &e, U32 capacity, U32 key_heads, U32 value_heads, U32 conv_width,
    Ref inout_bf16, Ref z, Ref norm_weight, Ref qa_out, Ref sa_out, Ref ra_out,
    float eps, bool write_bf16 = false, bool dynamic_tokens = true,
    std::optional<U32> mp_override = {}) {
  const U32 mp = mp_override.value_or(w4a8_nax_mp(capacity));
  const U64 total_heads = U64(mp) * U64(value_heads);
  const std::optional<U32> tok_word =
      dynamic_tokens ? std::optional<U32>{6U} : std::optional<U32>{};
  auto &d = e.emit(
      "w4a8_nax", "omlx_m5_gdn_chunk_norm_gate_inplace_a8_g64_c64",
      {.width = 128u, .height = (total_heads + 3u) / 4u}, {.width = 128u},
      {inout_bf16, z, norm_weight, qa_out, sa_out, ra_out});
  e.dynamic(d, 6,
            words({key_heads, value_heads, 128u, 128u, conv_width, 4u, capacity,
                   word(eps), mp, write_bf16 ? 1u : 0u}),
            tok_word);
}

// Dense Affine W4A8 Linear (oQ4e q3/q4/q56/q8 x g64/g128, bns16/bns32, Dual-ANE row_offset).
inline void linear_affine_w4a8_nax(Encoder &e, const TiledAffineWeight &tw,
                                   Ref qa, Ref sa, Ref ra, Ref y,
                                   std::optional<U32> explicit_t = {},
                                   U32 row_offset = 0u, U32 out_stride = 0u,
                                   U32 slice_n = 0u,
                                   std::optional<U32> mp_override = {}) {
  if ((tw.k % 64u) != 0u || (row_offset % 64u) != 0u || row_offset >= tw.n) {
    throw std::invalid_argument(
        "linear_affine_w4a8_nax requires k % 64 == 0, row_offset % 64 == 0, row_offset < n");
  }
  const U32 eff_n = (slice_n > 0u) ? slice_n : (tw.n - row_offset);
  if (row_offset + eff_n > tw.n) {
    throw std::invalid_argument("linear_affine_w4a8_nax slice exceeds weight n");
  }
  const U32 eff_stride = (out_stride > 0u) ? out_stride : tw.n;
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mp_override.value_or(w4a8_nax_mp(t));
  const U32 eff_gs =
      (tw.group_size == 128u && (tw.k % 128u) == 0u) ? 128u : 64u;
  const std::string tag = affine_nax_mode_tag(tw.bits);
  const std::string fn = "omlx_m5_affine_w4a8_nax_" + tag + "_g" +
                         std::to_string(eff_gs) + "_bm64_bn64_bns" +
                         std::to_string(tw.bn) + "_sk1";
  const U32 x_tiles = (eff_n + 63u) / 64u;
  const U32 y_tiles = (t + 63u) / 64u;
  const MetalDispatchSize grid{.width = U64(x_tiles) * 128u, .height = y_tiles};
  const MetalDispatchSize threads{.width = 128u};
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  if (!affine_nax_has_high_plane(tw.bits)) {
    auto &d = e.emit("w4a8_nax", fn, grid, threads,
                     {qa, sa, ra, tw.codes, tw.sb, y});
    e.dynamic(d, 6,
              words({eff_n, tw.k, mp, t, tw.padded_n, row_offset, eff_stride, 0u}),
              tokens_patch);
  } else {
    auto &d = e.emit("w4a8_nax", fn, grid, threads,
                     {qa, sa, ra, tw.codes, tw.high2, tw.sb, y});
    e.dynamic(d, 7,
              words({eff_n, tw.k, mp, t, tw.padded_n, row_offset, eff_stride, 0u}),
              tokens_patch);
  }
}

// Dense Affine W4A8 Fused Gate + Up + SwiGLU (BM=64, BN_out=64, BK=64, Dual-ANE row_offset).
inline void linear_affine_w4a8_nax_swiglu(Encoder &e,
                                          const TiledAffineWeight &gw,
                                          const TiledAffineWeight &uw,
                                          Ref qa, Ref sa, Ref ra, Ref y,
                                          std::optional<U32> explicit_t = {},
                                          U32 row_offset = 0u,
                                          U32 out_stride = 0u,
                                          U32 slice_n = 0u,
                                          std::optional<U32> mp_override = {}) {
  if (gw.n != uw.n || gw.k != uw.k || gw.bn != 16u || uw.bn != 16u ||
      (gw.k % 64u) != 0u || (row_offset % 64u) != 0u || row_offset >= gw.n) {
    throw std::invalid_argument(
        "linear_affine_w4a8_nax_swiglu requires matching bn=16 weights, k % 64 == 0, row_offset % 64 == 0");
  }
  const U32 eff_n = (slice_n > 0u) ? slice_n : (gw.n - row_offset);
  if (row_offset + eff_n > gw.n) {
    throw std::invalid_argument("linear_affine_w4a8_nax_swiglu slice exceeds weight n");
  }
  const U32 eff_stride = (out_stride > 0u) ? out_stride : gw.n;
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mp_override.value_or(w4a8_nax_mp(t));
  const U32 eff_gs = (gw.group_size == 128u && uw.group_size == 128u &&
                      (gw.k % 128u) == 0u)
                         ? 128u
                         : 64u;
  const std::string tag_g = affine_nax_mode_tag(gw.bits);
  const std::string tag_u = affine_nax_mode_tag(uw.bits);
  const std::string mode_tag =
      (tag_g == tag_u) ? tag_g : (tag_g + "_" + tag_u);
  const std::string fn = "omlx_m5_affine_w4a8_nax_" + mode_tag + "_g" +
                         std::to_string(eff_gs) +
                         "_swiglu_bm64_bn64_out64_sk1";
  const U32 x_tiles = (eff_n + 63u) / 64u;
  const U32 y_tiles = (t + 63u) / 64u;
  const MetalDispatchSize grid{.width = U64(x_tiles) * 256u, .height = y_tiles};
  const MetalDispatchSize threads{.width = 256u};
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  if (!affine_nax_has_high_plane(gw.bits) &&
      !affine_nax_has_high_plane(uw.bits)) {
    auto &d = e.emit("w4a8_nax", fn, grid, threads,
                     {qa, sa, ra, gw.codes, gw.sb, uw.codes, uw.sb, y});
    e.dynamic(d, 8,
              words({eff_n, gw.k, mp, t, gw.padded_n, row_offset, eff_stride, 0u}),
              tokens_patch);
  } else {
    const Ref g_hi = gw.high2.buffer != nullptr ? gw.high2 : gw.codes;
    const Ref u_hi = uw.high2.buffer != nullptr ? uw.high2 : uw.codes;
    auto &d = e.emit("w4a8_nax", fn, grid, threads,
                     {qa, sa, ra, gw.codes, g_hi, gw.sb, uw.codes, u_hi,
                      uw.sb, y});
    e.dynamic(d, 10,
              words({eff_n, gw.k, mp, t, gw.padded_n, row_offset, eff_stride, 0u}),
              tokens_patch);
  }
}

// Dense MXFP4 W4A8 Linear (k_sub=32, bns16/bns32, Dual-ANE row_offset).
inline void linear_mxfp4_w4a8_nax(Encoder &e, const TiledMxfp4Weight &tw,
                                  Ref qa, Ref sa, Ref y,
                                  std::optional<U32> explicit_t = {},
                                  U32 row_offset = 0u, U32 out_stride = 0u,
                                  U32 slice_n = 0u,
                                  std::optional<U32> mp_override = {}) {
  if ((tw.k % 128u) != 0u || (row_offset % 64u) != 0u || row_offset >= tw.n) {
    throw std::invalid_argument(
        "linear_mxfp4_w4a8_nax requires k % 128 == 0, row_offset % 64 == 0, row_offset < n");
  }
  const U32 eff_n = (slice_n > 0u) ? slice_n : (tw.n - row_offset);
  if (row_offset + eff_n > tw.n) {
    throw std::invalid_argument("linear_mxfp4_w4a8_nax slice exceeds weight n");
  }
  const U32 eff_stride = (out_stride > 0u) ? out_stride : tw.n;
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mp_override.value_or(w4a8_nax_mp(t));
  const std::string fn = "omlx_m5_mxfp4_w4a8_nax_bm64_bn64_bns" +
                         std::to_string(tw.bn) + "_sk1";
  const U32 x_tiles = (eff_n + 63u) / 64u;
  const U32 y_tiles = (t + 63u) / 64u;
  const MetalDispatchSize grid{.width = U64(x_tiles) * 128u, .height = y_tiles};
  const MetalDispatchSize threads{.width = 128u};
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  auto &d = e.emit("w4a8_nax", fn, grid, threads,
                   {qa, sa, tw.codes, tw.scales, y});
  e.dynamic(d, 5,
            words({eff_n, tw.k, mp, t, tw.padded_n, row_offset, eff_stride, 0u}),
            tokens_patch);
}

// Dense MXFP4 W4A8 Fused Gate + Up + SwiGLU (BM=64, BN_out=64, k_sub=32, Dual-ANE row_offset).
inline void linear_mxfp4_w4a8_nax_swiglu(Encoder &e,
                                         const TiledMxfp4Weight &gw,
                                         const TiledMxfp4Weight &uw,
                                         Ref qa, Ref sa, Ref y,
                                         std::optional<U32> explicit_t = {},
                                         U32 row_offset = 0u,
                                         U32 out_stride = 0u,
                                         U32 slice_n = 0u,
                                         std::optional<U32> mp_override = {}) {
  if (gw.n != uw.n || gw.k != uw.k || gw.bn != 16u || uw.bn != 16u ||
      (gw.k % 128u) != 0u || (row_offset % 64u) != 0u || row_offset >= gw.n) {
    throw std::invalid_argument(
        "linear_mxfp4_w4a8_nax_swiglu requires matching bn=16 weights, k % 128 == 0, row_offset % 64 == 0");
  }
  const U32 eff_n = (slice_n > 0u) ? slice_n : (gw.n - row_offset);
  if (row_offset + eff_n > gw.n) {
    throw std::invalid_argument("linear_mxfp4_w4a8_nax_swiglu slice exceeds weight n");
  }
  const U32 eff_stride = (out_stride > 0u) ? out_stride : gw.n;
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mp_override.value_or(w4a8_nax_mp(t));
  const U32 x_tiles = (eff_n + 63u) / 64u;
  const U32 y_tiles = (t + 63u) / 64u;
  const MetalDispatchSize grid{.width = U64(x_tiles) * 256u, .height = y_tiles};
  const MetalDispatchSize threads{.width = 256u};
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  auto &d = e.emit("w4a8_nax",
                   "omlx_m5_mxfp4_w4a8_nax_swiglu_bm64_bn64_out64_sk1",
                   grid, threads,
                   {qa, sa, gw.codes, gw.scales, uw.codes, uw.scales, y});
  e.dynamic(d, 7,
            words({eff_n, gw.k, mp, t, gw.padded_n, row_offset, eff_stride, 0u}),
            tokens_patch);
}

} // namespace omlx::m5::detail::forward
