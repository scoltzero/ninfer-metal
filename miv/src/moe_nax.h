#pragma once

#include "nax_common.h"
#include "affine_nax_cache.h"
#include "mxfp4_nax_cache.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace omlx::m5::detail::forward {

inline bool moe_nax_runtime_eligible(const MetalContext &ctx,
                                     const std::filesystem::path &libs) {
  return nax_apple10_eligible(ctx, libs, "omlx_m5_moe_nax.metallib",
                              "OMLX_M5_DISABLE_MOE_NAX");
}

inline bool moe_w4a8_nax_runtime_eligible(const MetalContext &ctx,
                                          const std::filesystem::path &libs) {
  return nax_apple10_eligible(ctx, libs, "omlx_m5_w4a8_moe_nax.metallib",
                              "OMLX_M5_DISABLE_W4A8_NAX");
}

inline bool moe_nax_swiglu_out64_env_enabled() {
  // WP3.1 Apple10 M5 Ultra verified: 1-pass BN_out=32 MoE out64 (9,472 B SRAM, 0 spills,
  // 128 threads, 3 TGs/core) beats bm64 by up to 1.74x and HEAD out64 (27,904 B) by 1.04x
  // at T=1024 (default ON).
  if (const char *env = std::getenv("OMLX_M5_DISABLE_MOE_SWIGLU_OUT64");
      env != nullptr && std::string_view(env) == "1") {
    return false;
  }
  if (const char *env = std::getenv("OMLX_M5_MOE_SWIGLU_OUT64");
      env != nullptr) {
    if (std::string_view(env) == "0") {
      return false;
    }
    if (std::string_view(env) == "1") {
      return true;
    }
  }
  return true;
}

// Unlocked down to T=1: tight moe_nax_max_tiles(t, top_k, num_experts) bounds
// non-empty BM=8 tiles to at most t * top_k (8 tiles at T=1, 64 at T=8) while
// preserving 0 ULP AR == DFlash2 == Batch == Prefill parity across T=1..1024.
inline U32 moe_nax_min_tokens() {
  if (const char *env = std::getenv("OMLX_M5_MOE_NAX_MIN_T")) {
    const U32 v = U32(std::strtoul(env, nullptr, 10));
    if (v >= 1u) {
      return v;
    }
  }
  return 1u;
}

inline U32 moe_nax_wide_min_tokens() {
  if (const char *env = std::getenv("OMLX_M5_MOE_WIDE_MIN_T")) {
    const U32 v = U32(std::strtoul(env, nullptr, 10));
    if (v >= 1u) {
      return v;
    }
  }
  return 128u;
}

inline U32 moe_nax_bm64_min_tokens() {
  if (const char *env = std::getenv("OMLX_M5_MOE_BM64_MIN_T")) {
    const U32 v = U32(std::strtoul(env, nullptr, 10));
    if (v >= 1u) {
      return v;
    }
  }
  return 256u;
}

inline U32 moe_nax_max_tiles(U32 tokens, U32 top_k = 8u,
                             U32 num_experts = 256u) {
  if (num_experts == 0u || num_experts > 256u || top_k == 0u) {
    throw std::invalid_argument(
        "MoeNaxWorkspace requires 1 <= num_experts <= 256 and top_k >= 1");
  }
  const U64 total_assign = U64(tokens) * U64(top_k);
  const U64 bound = (total_assign + 7ULL) / 8ULL + U64(num_experts);
  const U64 tight = std::min<U64>(std::max<U64>(1ULL, total_assign), bound);
  return U32(tight);
}

inline bool moe_nax_affine_parse(std::string_view format, U32 &bits_out,
                                 U32 &group_out) {
  if (!format.starts_with("oq_affine_q") || !format.ends_with("_bf16_v1")) {
    return false;
  }
  const auto g_pos = format.find("_g");
  if (g_pos == std::string_view::npos) {
    return false;
  }
  bits_out = U32(std::stoul(std::string(format.substr(11, g_pos - 11))));
  const auto v_pos = format.find("_bf16_v1");
  group_out = U32(std::stoul(std::string(format.substr(g_pos + 2, v_pos - (g_pos + 2)))));
  return (bits_out == 3u || bits_out == 4u || bits_out == 5u ||
          bits_out == 6u || bits_out == 8u) &&
         (group_out == 64u || group_out == 128u);
}

inline bool moe_nax_weight_eligible(const MaterializedWeightView &gw,
                                    const MaterializedWeightView &uw,
                                    const MaterializedWeightView &dw, U32 h,
                                    U32 f) {
  if (gw.format != uw.format || gw.padded_k != uw.padded_k) {
    return false;
  }
  if (gw.format == "mxfp4_e2m1_g32_e8m0_v1" &&
      dw.format == "mxfp4_e2m1_g32_e8m0_v1") {
    return (h % 128u) == 0u && (f % 128u) == 0u && (f % 16u) == 0u &&
           (h % 16u) == 0u && h >= 256u && f >= 256u;
  }
  U32 ub = 0, ug = 0, db = 0, dg = 0;
  if (moe_nax_affine_parse(gw.format, ub, ug) &&
      moe_nax_affine_parse(dw.format, db, dg)) {
    return (h % ug) == 0u && (f % dg) == 0u && (h % 64u) == 0u &&
           (f % 64u) == 0u && (f % 16u) == 0u && (h % 16u) == 0u &&
           h >= 256u && f >= 128u;
  }
  return false;
}

struct MoeNaxWorkspace {
  Ref tile_expert;
  Ref tile_valid_rows;
  Ref tile_assign;
  Ref total_tiles;
  Ref x_f16;
  Ref row_scale;
  Ref mid_f16;
  Ref xsum;
  Ref mid_xsum;
  Ref mid_qa;
  Ref mid_sa;
  Ref mid_ra;
  U32 max_t = 0;
  U32 max_rows = 0;
  U32 max_experts = 256;
  U32 max_top_k = 8;
  bool active = false;

  void allocate(const MetalContext &ctx, U32 max_tokens, U32 h, U32 f,
                std::vector<std::unique_ptr<MetalBuffer>> &buffers,
                U32 num_experts = 256u, U32 top_k = 8u) {
    if (num_experts == 0u || num_experts > 256u || top_k == 0u) {
      throw std::invalid_argument(
          "MoeNaxWorkspace requires 1 <= num_experts <= 256 and top_k >= 1");
    }
    max_t = max_tokens;
    max_experts = num_experts;
    max_top_k = top_k;
    const U32 max_tiles =
        U32((U64(max_tokens) * U64(top_k) + 7ULL) / 8ULL + U64(num_experts));
    max_rows = max_tiles * 8u;
    const U32 h_groups = std::max(1u, (h + 63u) / 64u);
    const U32 f_groups = std::max(1u, (f + 63u) / 64u);
    auto mk = [&](U64 bytes) -> Ref {
      auto buf = std::make_unique<MetalBuffer>(ctx.make_private_buffer(bytes));
      Ref r{buf.get(), 0, bytes};
      buffers.push_back(std::move(buf));
      return r;
    };
    tile_expert = mk(U64(max_tiles) * 4u);
    tile_valid_rows = mk(U64(max_tiles) * 4u);
    tile_assign = mk(U64(max_rows) * 4u);
    total_tiles = mk(16u);
    x_f16 = mk(U64(max_rows) * h * 2u);
    row_scale = mk(U64(max_rows) * 4u);
    mid_f16 = mk(U64(max_rows) * f * 2u);
    xsum = mk(U64(max_rows) * h_groups * 4u);
    mid_xsum = mk(U64(max_rows) * f_groups * 4u);
    mid_qa = mk(U64(max_rows) * f);
    mid_sa = mk(U64(max_rows) * f_groups * 4u);
    mid_ra = mk(U64(max_rows) * f_groups * 4u);
    active = true;
  }
};

// Unified MoE NAX FFN lowering helper supporting both MXFP4 and zero-repack
// oq_affine (q3/q4/q5/q6/q8 x g64/g128) across T=1..2048 and packed B=2..8.
inline bool emit_moe_nax_ffn(Encoder &e, const MoeNaxWorkspace &ws,
                             const MaterializedWeightView &gw,
                             const MaterializedWeightView &uw,
                             const MaterializedWeightView &dw, Ref norm,
                             Ref route_ids, Ref route_scores, Ref expert_output,
                             Ref routed, U32 t, U32 h, U32 f,
                             U32 top_k = 8u, U32 num_experts = 256u,
                             U32 gpu_cores = 64u,
                             std::optional<U32> explicit_t = {},
                             Ref router_logits = {},
                             bool emit_reduce = true,
                             bool use_out64_swiglu = false) {
  if (!ws.active || t > ws.max_t || t < moe_nax_min_tokens() ||
      !moe_nax_weight_eligible(gw, uw, dw, h, f)) {
    return false;
  }
  const U32 max_tiles = moe_nax_max_tiles(t, top_k, num_experts);
  const std::optional<U32> tok_w0 =
      explicit_t.has_value() ? std::optional<U32>{} : std::optional<U32>{0u};
  const U32 wide_min_t = std::max<U32>(65u, moe_nax_wide_min_tokens());
  const U32 bm64_min_t = std::max<U32>(wide_min_t, moe_nax_bm64_min_tokens());
  const bool use_wide =
      (!explicit_t.has_value() || e.delegated_prefill) &&
      e.profile.capacity >= wide_min_t && top_k == 8u && num_experts == 256u;
  const bool use_wide_bm64 = use_wide && e.profile.capacity >= bm64_min_t;
  const bool eff_out64 =
      use_wide_bm64 && (use_out64_swiglu || moe_nax_swiglu_out64_env_enabled());
  const U32 wbm = use_wide_bm64 ? 64u : 32u;
  const U32 max_wtiles =
      use_wide
          ? std::min<U32>(
                max_tiles,
                U32((U64(t) * U64(top_k) + U64(wbm - 1u)) / U64(wbm) +
                    U64(num_experts)))
          : max_tiles;
  const bool use_gx32 = use_wide_bm64 && e.profile.capacity >= 1152u;
  const U32 gemm_max_tiles_word = max_tiles | (use_gx32 ? 0x80000000u : 0u);
  auto make_wide_grid = [&](U32 n_col_tiles, U32 tptg = 128u) -> MetalDispatchSize {
    if (use_gx32) {
      return {.width = 32ULL * U64(tptg),
              .height = U64((max_wtiles + 31u) / 32u) * U64(n_col_tiles)};
    }
    return {.width = U64(n_col_tiles) * U64(tptg), .height = max_wtiles};
  };

  if (router_logits.buffer != nullptr && top_k == 8u && num_experts == 256u) {
    const char *sched_fn =
        use_wide ? (use_wide_bm64
                        ? "omlx_m5_moe_nax_router_top8_bucket_schedule_bm64"
                        : "omlx_m5_moe_nax_router_top8_bucket_schedule_bm32")
                 : "omlx_m5_moe_nax_router_top8_bucket_schedule";
    auto &sched =
        e.emit("moe_nax", sched_fn, {.width = 256u}, {.width = 256u},
               {router_logits, route_ids, route_scores, ws.tile_expert,
                ws.tile_valid_rows, ws.tile_assign, ws.total_tiles});
    e.dynamic(sched, 7, words({t, top_k, h, max_tiles}), tok_w0);
  } else {
    const char *sched_fn =
        use_wide ? (use_wide_bm64 ? "omlx_m5_moe_nax_bucket_schedule_bm64"
                                  : "omlx_m5_moe_nax_bucket_schedule_bm32")
                 : "omlx_m5_moe_nax_bucket_schedule";
    auto &sched = e.emit("moe_nax", sched_fn, {.width = 256u}, {.width = 256u},
                         {route_ids, ws.tile_expert, ws.tile_valid_rows,
                          ws.tile_assign, ws.total_tiles});
    e.dynamic(sched, 5, words({t, top_k, h, max_tiles}), tok_w0);
  }

  if (gw.format == "mxfp4_e2m1_g32_e8m0_v1" &&
      dw.format == "mxfp4_e2m1_g32_e8m0_v1") {
    if (use_wide) {
      const char *up_fn = eff_out64
                              ? "omlx_m5_moe_mxfp4_nax_swiglu_bm64_out64_sk1"
                              : (use_wide_bm64
                                     ? "omlx_m5_moe_mxfp4_nax_swiglu_bm64_bn64_sk1"
                                     : "omlx_m5_moe_mxfp4_nax_swiglu_bm32_bn64_sk1");
      const char *up_tag = eff_out64
                               ? "moe_nax_swiglu_bm64_out64_sk1_v1"
                               : (use_wide_bm64 ? "moe_nax_swiglu_bm64_sk1_v1"
                                                : "moe_nax_swiglu_wide_sk1_v1");
      const U32 up_tptg = 128u;
      const U32 up_x_tiles = (f + 31u) / 32u;
      auto &up_nax = e.emit(
          "moe_nax", up_fn, make_wide_grid(up_x_tiles, up_tptg), {.width = up_tptg},
          {norm, ws.tile_assign, ws.tile_expert, ws.tile_valid_rows,
           ws.total_tiles, plane(gw, "data"), plane(gw, "scale"),
           plane(uw, "data"), plane(uw, "scale"), ws.mid_f16});
      e.dynamic(up_nax, 10,
                words({f, h, U32(gw.padded_k), gemm_max_tiles_word}));
      e.record_weight(gw, f, h, "moe_nax_m5_v1", up_tag, {}, "legacy_route_v1",
                      {}, false, explicit_t);
      e.record_weight(uw, f, h, "moe_nax_m5_v1", up_tag, {}, "legacy_route_v1",
                      {}, false, explicit_t);

      if (dw.has_fwht64()) {
        auto &mid_rot = e.emit(
            "moe_nax", "omlx_m5_moe_nax_mid_fwht64_bf16",
            {.width = U64(f / 64u) * 32u, .height = U64(max_tiles) * 8u},
            {.width = 32u}, {ws.mid_f16, ws.total_tiles});
        e.dynamic(mid_rot, 2, words({t, top_k, f, max_tiles}), tok_w0);
      }

      const char *dn_fn =
          use_wide_bm64 ? "omlx_m5_moe_mxfp4_nax_down_scatter_bm64_bn64_sk1"
                        : "omlx_m5_moe_mxfp4_nax_down_scatter_bm32_bn64_sk1";
      const char *dn_tag = use_wide_bm64 ? "moe_nax_down_bm64_sk1_v1"
                                         : "moe_nax_down_wide_sk1_v1";
      const U32 dn_x_tiles = (h + 63u) / 64u;
      auto &dn_nax = e.emit(
          "moe_nax", dn_fn, make_wide_grid(dn_x_tiles), {.width = 128u},
          {ws.mid_f16, ws.tile_expert, ws.tile_valid_rows, ws.tile_assign,
           ws.total_tiles, route_scores, plane(dw, "data"), plane(dw, "scale"),
           expert_output});
      e.dynamic(dn_nax, 9,
                words({h, f, U32(dw.padded_k), gemm_max_tiles_word}));
      e.record_weight(dw, h, f, "moe_nax_m5_v1", dn_tag, {}, "legacy_route_v1",
                      {}, false, explicit_t);
    } else {
      auto &gather =
          e.emit("moe_nax", "omlx_m5_moe_nax_gather_f16",
                 {.width = 128u, .height = U64(max_tiles) * 8u},
                 {.width = 128u},
                 {norm, ws.tile_assign, ws.total_tiles, ws.x_f16, ws.row_scale});
      e.dynamic(gather, 5, words({t, top_k, h, max_tiles}), tok_w0);

      const U32 up_nsg =
          nax_select_nsg_bn16(f, max_tiles, 4u, gpu_cores, true);
      const std::string up_fn =
          up_nsg == 2u ? "omlx_m5_moe_mxfp4_nax_swiglu_n16_sk4_nsg2"
                       : "omlx_m5_moe_mxfp4_nax_swiglu_n16_sk4";
      const U32 up_x_tiles = ((f / 16u) + up_nsg - 1u) / up_nsg;
      auto &up_nax = e.emit(
          "moe_nax", up_fn,
          {.width = U64(up_x_tiles) * 128u * up_nsg, .height = max_tiles},
          {.width = 128u * up_nsg},
          {ws.x_f16, ws.row_scale, ws.tile_expert, ws.tile_valid_rows,
           ws.total_tiles, plane(gw, "data"), plane(gw, "scale"),
           plane(uw, "data"), plane(uw, "scale"), ws.mid_f16});
      e.dynamic(up_nax, 10, words({f, h, U32(gw.padded_k), max_tiles}));
      e.record_weight(gw, f, h, "moe_nax_m5_v1", "moe_nax_swiglu_v1", {},
                      "legacy_route_v1", {}, false, explicit_t);
      e.record_weight(uw, f, h, "moe_nax_m5_v1", "moe_nax_swiglu_v1", {},
                      "legacy_route_v1", {}, false, explicit_t);

      if (dw.has_fwht64()) {
        auto &mid_rot = e.emit(
            "moe_nax", "omlx_m5_moe_nax_mid_fwht64_f16",
            {.width = U64(f / 64u) * 32u, .height = U64(max_tiles) * 8u},
            {.width = 32u}, {ws.mid_f16, ws.total_tiles});
        e.dynamic(mid_rot, 2, words({t, top_k, f, max_tiles}), tok_w0);
      }

      const U32 dn_nsg =
          nax_select_nsg_bn16(h, max_tiles, 2u, gpu_cores, true);
      const std::string dn_fn =
          dn_nsg == 2u ? "omlx_m5_moe_mxfp4_nax_down_scatter_n16_sk2_nsg2"
                       : "omlx_m5_moe_mxfp4_nax_down_scatter_n16_sk2";
      const U32 dn_x_tiles = ((h / 16u) + dn_nsg - 1u) / dn_nsg;
      auto &dn_nax = e.emit(
          "moe_nax", dn_fn,
          {.width = U64(dn_x_tiles) * 64u * dn_nsg, .height = max_tiles},
          {.width = 64u * dn_nsg},
          {ws.mid_f16, ws.tile_expert, ws.tile_valid_rows, ws.tile_assign,
           ws.total_tiles, route_scores, plane(dw, "data"), plane(dw, "scale"),
           expert_output});
      e.dynamic(dn_nax, 9, words({h, f, U32(dw.padded_k), max_tiles}));
      e.record_weight(dw, h, f, "moe_nax_m5_v1", "moe_nax_down_v1", {},
                      "legacy_route_v1", {}, false, explicit_t);
    }
  } else {
    U32 ub = 0, ug = 0, db = 0, dg = 0;
    (void)moe_nax_affine_parse(gw.format, ub, ug);
    (void)moe_nax_affine_parse(dw.format, db, dg);

    auto code_plane = [](const MaterializedWeightView &w, U32 bits) -> Ref {
      if (bits == 3u) return plane(w, "low_2bit");
      if (bits == 8u) return plane(w, "code");
      return plane(w, "base_code");
    };
    auto high_plane = [](const MaterializedWeightView &w, U32 bits) -> Ref {
      if (bits == 3u || bits == 5u) return plane(w, "high_bit");
      return plane(w, "high_2bit");
    };

    if (use_wide) {
      const char *wide_sfx =
          use_wide_bm64 ? "_bm64_bn64_sk1" : "_bm32_bn64_sk1";
      const char *up_sfx = eff_out64 ? "_bm64_out64_sk1" : wide_sfx;
      const char *up_tag =
          eff_out64 ? "moe_affine_nax_swiglu_bm64_out64_sk1_v1"
                    : (use_wide_bm64 ? "moe_affine_nax_swiglu_bm64_sk1_v1"
                                     : "moe_affine_nax_swiglu_wide_sk1_v1");
      const char *dn_tag = use_wide_bm64 ? "moe_affine_nax_down_bm64_sk1_v1"
                                         : "moe_affine_nax_down_wide_sk1_v1";
      const std::string up_fn = "omlx_m5_moe_affine_nax_swiglu_q" +
                                std::to_string(ub) + "_g" + std::to_string(ug) +
                                up_sfx;
      const U32 up_tptg = 128u;
      const U32 up_x_tiles = (f + 31u) / 32u;
      if (ub == 4u || ub == 8u) {
        auto &up_nax = e.emit(
            "moe_nax", up_fn, make_wide_grid(up_x_tiles, up_tptg),
            {.width = up_tptg},
            {norm, ws.tile_assign, ws.tile_expert, ws.tile_valid_rows,
             ws.total_tiles, code_plane(gw, ub), plane(gw, "scale"),
             plane(gw, "bias"), code_plane(uw, ub), plane(uw, "scale"),
             plane(uw, "bias"), ws.mid_f16});
        e.dynamic(up_nax, 12,
                  words({f, h, U32(gw.padded_k), gemm_max_tiles_word}));
      } else {
        auto &up_nax = e.emit(
            "moe_nax", up_fn, make_wide_grid(up_x_tiles, up_tptg),
            {.width = up_tptg},
            {norm, ws.tile_assign, ws.tile_expert, ws.tile_valid_rows,
             ws.total_tiles, code_plane(gw, ub), high_plane(gw, ub),
             plane(gw, "scale"), plane(gw, "bias"), code_plane(uw, ub),
             high_plane(uw, ub), plane(uw, "scale"), plane(uw, "bias"),
             ws.mid_f16});
        e.dynamic(up_nax, 14,
                  words({f, h, U32(gw.padded_k), gemm_max_tiles_word}));
      }
      e.record_weight(gw, f, h, "moe_nax_m5_v1", up_tag, {}, "legacy_route_v1",
                      {}, false, explicit_t);
      e.record_weight(uw, f, h, "moe_nax_m5_v1", up_tag, {}, "legacy_route_v1",
                      {}, false, explicit_t);

      if (dw.has_fwht64()) {
        auto &mid_rot = e.emit(
            "moe_nax", "omlx_m5_moe_nax_mid_fwht64_bf16",
            {.width = U64(f / 64u) * 32u, .height = U64(max_tiles) * 8u},
            {.width = 32u}, {ws.mid_f16, ws.total_tiles});
        e.dynamic(mid_rot, 2, words({t, top_k, f, max_tiles}), tok_w0);
      }

      const std::string dn_fn = "omlx_m5_moe_affine_nax_down_scatter_q" +
                                std::to_string(db) + "_g" + std::to_string(dg) +
                                wide_sfx;
      const U32 dn_x_tiles = (h + 63u) / 64u;
      if (db == 4u || db == 8u) {
        auto &dn_nax = e.emit(
            "moe_nax", dn_fn, make_wide_grid(dn_x_tiles), {.width = 128u},
            {ws.mid_f16, ws.tile_expert, ws.tile_valid_rows, ws.tile_assign,
             ws.total_tiles, route_scores, code_plane(dw, db),
             plane(dw, "scale"), plane(dw, "bias"), expert_output});
        e.dynamic(dn_nax, 10,
                  words({h, f, U32(dw.padded_k), gemm_max_tiles_word}));
      } else {
        auto &dn_nax = e.emit(
            "moe_nax", dn_fn, make_wide_grid(dn_x_tiles), {.width = 128u},
            {ws.mid_f16, ws.tile_expert, ws.tile_valid_rows, ws.tile_assign,
             ws.total_tiles, route_scores, code_plane(dw, db),
             high_plane(dw, db), plane(dw, "scale"), plane(dw, "bias"),
             expert_output});
        e.dynamic(dn_nax, 11,
                  words({h, f, U32(dw.padded_k), gemm_max_tiles_word}));
      }
      e.record_weight(dw, h, f, "moe_nax_m5_v1", dn_tag, {}, "legacy_route_v1",
                      {}, false, explicit_t);
    } else {
      auto &gather = e.emit(
          "moe_nax", "omlx_m5_moe_nax_gather_bf16_xsum_g64",
          {.width = U64(h / 64u) * 32u, .height = U64(max_tiles) * 8u},
          {.width = 32u},
          {norm, ws.tile_assign, ws.total_tiles, ws.x_f16, ws.xsum});
      e.dynamic(gather, 5, words({t, top_k, h, max_tiles}), tok_w0);

      const U32 up_nsg =
          nax_select_nsg_bn16(f, max_tiles, 4u, gpu_cores, true);
      const std::string up_fn = "omlx_m5_moe_affine_nax_swiglu_q" +
                                std::to_string(ub) + "_g" + std::to_string(ug) +
                                (up_nsg == 2u ? "_n16_sk4_nsg2" : "_n16_sk4");
      const U32 up_x_tiles = ((f / 16u) + up_nsg - 1u) / up_nsg;
      if (ub == 4u || ub == 8u) {
        auto &up_nax = e.emit(
            "moe_nax", up_fn,
            {.width = U64(up_x_tiles) * 128u * up_nsg, .height = max_tiles},
            {.width = 128u * up_nsg},
            {ws.x_f16, ws.xsum, ws.tile_expert, ws.tile_valid_rows,
             ws.total_tiles, code_plane(gw, ub), plane(gw, "scale"),
             plane(gw, "bias"), code_plane(uw, ub), plane(uw, "scale"),
             plane(uw, "bias"), ws.mid_f16});
        e.dynamic(up_nax, 12, words({f, h, U32(gw.padded_k), max_tiles}));
      } else {
        auto &up_nax = e.emit(
            "moe_nax", up_fn,
            {.width = U64(up_x_tiles) * 128u * up_nsg, .height = max_tiles},
            {.width = 128u * up_nsg},
            {ws.x_f16, ws.xsum, ws.tile_expert, ws.tile_valid_rows,
             ws.total_tiles, code_plane(gw, ub), high_plane(gw, ub),
             plane(gw, "scale"), plane(gw, "bias"), code_plane(uw, ub),
             high_plane(uw, ub), plane(uw, "scale"), plane(uw, "bias"),
             ws.mid_f16});
        e.dynamic(up_nax, 14, words({f, h, U32(gw.padded_k), max_tiles}));
      }
      e.record_weight(gw, f, h, "moe_nax_m5_v1", "moe_affine_nax_swiglu_v1", {},
                      "legacy_route_v1", {}, false, explicit_t);
      e.record_weight(uw, f, h, "moe_nax_m5_v1", "moe_affine_nax_swiglu_v1", {},
                      "legacy_route_v1", {}, false, explicit_t);

      if (dw.has_fwht64()) {
        auto &mid_rot = e.emit(
            "moe_nax", "omlx_m5_moe_nax_mid_fwht64_bf16",
            {.width = U64(f / 64u) * 32u, .height = U64(max_tiles) * 8u},
            {.width = 32u}, {ws.mid_f16, ws.total_tiles});
        e.dynamic(mid_rot, 2, words({t, top_k, f, max_tiles}), tok_w0);
      }

      auto &mid_xs = e.emit(
          "moe_nax", "omlx_m5_moe_nax_mid_xsum_g64",
          {.width = U64(f / 64u) * 32u, .height = U64(max_tiles) * 8u},
          {.width = 32u}, {ws.mid_f16, ws.total_tiles, ws.mid_xsum});
      e.dynamic(mid_xs, 3, words({t, top_k, f, max_tiles}), tok_w0);

      const U32 dn_nsg =
          nax_select_nsg_bn16(h, max_tiles, 2u, gpu_cores, true);
      const std::string dn_fn = "omlx_m5_moe_affine_nax_down_scatter_q" +
                                std::to_string(db) + "_g" + std::to_string(dg) +
                                (dn_nsg == 2u ? "_n16_sk2_nsg2" : "_n16_sk2");
      const U32 dn_x_tiles = ((h / 16u) + dn_nsg - 1u) / dn_nsg;
      if (db == 4u || db == 8u) {
        auto &dn_nax = e.emit(
            "moe_nax", dn_fn,
            {.width = U64(dn_x_tiles) * 64u * dn_nsg, .height = max_tiles},
            {.width = 64u * dn_nsg},
            {ws.mid_f16, ws.mid_xsum, ws.tile_expert, ws.tile_valid_rows,
             ws.tile_assign, ws.total_tiles, route_scores, code_plane(dw, db),
             plane(dw, "scale"), plane(dw, "bias"), expert_output});
        e.dynamic(dn_nax, 11, words({h, f, U32(dw.padded_k), max_tiles}));
      } else {
        auto &dn_nax = e.emit(
            "moe_nax", dn_fn,
            {.width = U64(dn_x_tiles) * 64u * dn_nsg, .height = max_tiles},
            {.width = 64u * dn_nsg},
            {ws.mid_f16, ws.mid_xsum, ws.tile_expert, ws.tile_valid_rows,
             ws.tile_assign, ws.total_tiles, route_scores, code_plane(dw, db),
             high_plane(dw, db), plane(dw, "scale"), plane(dw, "bias"),
             expert_output});
        e.dynamic(dn_nax, 12, words({h, f, U32(dw.padded_k), max_tiles}));
      }
      e.record_weight(dw, h, f, "moe_nax_m5_v1", "moe_affine_nax_down_v1", {},
                      "legacy_route_v1", {}, false, explicit_t);
    }
  }

  if (emit_reduce) {
    auto &reduce =
        e.emit("mlp", "omlx_m5_moe_topk_reduce", {.width = U64(t) * h},
               {.width = 128u}, {expert_output, routed});
    e.dynamic(reduce, 2, words({t, top_k, h, 0u}), tok_w0);
  }
  return true;
}

// Route A (WP1 Part B): MoE W4A8 Prefill FFN (Zero-Ws W4A8 out64 Gate+Up+SwiGLU
// + W4A8 down_scatter when dw.has_fwht64() || w4a8_down_proj, else W4A16 down_scatter).
// Consumes shared GS64 INT8 (a8_qa, a8_sa, a8_ra, a8_mp) produced by post_attention_rmsnorm_a8.
inline bool emit_moe_w4a8_nax_ffn(
    Encoder &e, const MoeNaxWorkspace &ws,
    const MaterializedWeightView &gw, const MaterializedWeightView &uw,
    const MaterializedWeightView &dw, Ref a8_qa, Ref a8_sa, Ref a8_ra,
    U32 a8_mp, Ref route_ids, Ref route_scores, Ref expert_output, Ref routed,
    U32 t, U32 h, U32 f, U32 top_k = 8u, U32 num_experts = 256u,
    std::optional<U32> explicit_t = {}, Ref router_logits = {},
    bool emit_reduce = true, bool w4a8_down_proj = false) {
  if (!ws.active || t == 0u || t > ws.max_t || top_k != 8u ||
      num_experts != 256u || !moe_nax_weight_eligible(gw, uw, dw, h, f) ||
      a8_qa.buffer == nullptr || a8_sa.buffer == nullptr || a8_mp < t) {
    return false;
  }
  const bool is_mxfp4 = (gw.format == "mxfp4_e2m1_g32_e8m0_v1" &&
                         dw.format == "mxfp4_e2m1_g32_e8m0_v1");
  if (!is_mxfp4 && a8_ra.buffer == nullptr) {
    return false;
  }

  const U32 max_tiles = moe_nax_max_tiles(t, top_k, num_experts);
  const std::optional<U32> tok_w0 =
      explicit_t.has_value() ? std::optional<U32>{} : std::optional<U32>{0u};
  const U32 max_wtiles = std::min<U32>(
      max_tiles,
      U32((U64(t) * U64(top_k) + 63ULL) / 64ULL + U64(num_experts)));
  const bool use_gx32 = e.profile.capacity >= 1152u;
  const U32 gemm_max_tiles_word = max_tiles | (use_gx32 ? 0x80000000u : 0u);
  const bool use_w4a8_down =
      (dw.has_fwht64() || w4a8_down_proj) && ((f % 64u) == 0u) &&
      (ws.mid_qa.buffer != nullptr) && (ws.mid_sa.buffer != nullptr) &&
      (ws.mid_ra.buffer != nullptr) && (ws.max_rows >= max_tiles * 8u);
  const U32 mid_reserved =
      ws.max_rows | (dw.has_fwht64() ? 0u : 0x80000000u);
  auto make_wide_grid = [&](U32 n_col_tiles, U32 tptg) -> MetalDispatchSize {
    if (use_gx32) {
      return {.width = 32ULL * U64(tptg),
              .height = U64((max_wtiles + 31u) / 32u) * U64(n_col_tiles)};
    }
    return {.width = U64(n_col_tiles) * U64(tptg), .height = max_wtiles};
  };

  if (router_logits.buffer != nullptr) {
    auto &sched = e.emit(
        "moe_nax", "omlx_m5_moe_nax_router_top8_bucket_schedule_bm64",
        {.width = 256u}, {.width = 256u},
        {router_logits, route_ids, route_scores, ws.tile_expert,
         ws.tile_valid_rows, ws.tile_assign, ws.total_tiles});
    e.dynamic(sched, 7, words({t, top_k, h, max_tiles}), tok_w0);
  } else {
    auto &sched = e.emit(
        "moe_nax", "omlx_m5_moe_nax_bucket_schedule_bm64", {.width = 256u},
        {.width = 256u},
        {route_ids, ws.tile_expert, ws.tile_valid_rows, ws.tile_assign,
         ws.total_tiles});
    e.dynamic(sched, 5, words({t, top_k, h, max_tiles}), tok_w0);
  }

  const U32 up_x_tiles_64 = (f + 63u) / 64u;
  const U32 up_x_tiles_32 = (f + 31u) / 32u;
  const U32 dn_x_tiles = (h + 63u) / 64u;
  if (is_mxfp4) {
    if (use_w4a8_down) {
      auto &up_nax = e.emit(
          "w4a8_moe_nax",
          "omlx_m5_moe_mxfp4_w4a8_nax_swiglu_bm64_out64_sk1_fwht64_a8",
          make_wide_grid(up_x_tiles_64, 128u), {.width = 128u},
          {a8_qa, a8_sa, ws.tile_assign, ws.tile_expert, ws.tile_valid_rows,
           ws.total_tiles, plane(gw, "data"), plane(gw, "scale"),
           plane(uw, "data"), plane(uw, "scale"), ws.mid_qa, ws.mid_sa,
           ws.mid_ra});
      e.dynamic(up_nax, 13,
                words({f, h, U32(gw.padded_k), gemm_max_tiles_word, a8_mp,
                       mid_reserved}));
      e.record_weight(gw, f, h, "moe_w4a8_nax_m5_v2",
                      "moe_w4a8_swiglu_bm64_out64_sk1_fwht64_a8_v2", {},
                      "legacy_route_v1", {}, false, explicit_t);
      e.record_weight(uw, f, h, "moe_w4a8_nax_m5_v2",
                      "moe_w4a8_swiglu_bm64_out64_sk1_fwht64_a8_v2", {},
                      "legacy_route_v1", {}, false, explicit_t);

      auto &dn_nax = e.emit(
          "w4a8_moe_nax",
          "omlx_m5_moe_mxfp4_w4a8_nax_down_scatter_bm64_bn64_sk1",
          make_wide_grid(dn_x_tiles, 128u), {.width = 128u},
          {ws.mid_qa, ws.mid_sa, ws.tile_expert, ws.tile_valid_rows,
           ws.tile_assign, ws.total_tiles, route_scores, plane(dw, "data"),
           plane(dw, "scale"), expert_output});
      e.dynamic(dn_nax, 10,
                words({h, f, U32(dw.padded_k), gemm_max_tiles_word, ws.max_rows,
                       0u}));
      e.record_weight(dw, h, f, "moe_w4a8_nax_m5_v2",
                      "moe_w4a8_down_bm64_bn64_sk1_v2", {}, "legacy_route_v1",
                      {}, false, explicit_t);
    } else {
      auto &up_nax = e.emit(
          "w4a8_moe_nax", "omlx_m5_moe_mxfp4_w4a8_nax_swiglu_bm64_out64_sk1",
          make_wide_grid(up_x_tiles_32, 128u), {.width = 128u},
          {a8_qa, a8_sa, ws.tile_assign, ws.tile_expert, ws.tile_valid_rows,
           ws.total_tiles, plane(gw, "data"), plane(gw, "scale"),
           plane(uw, "data"), plane(uw, "scale"), ws.mid_f16});
      e.dynamic(up_nax, 11,
                words({f, h, U32(gw.padded_k), gemm_max_tiles_word, a8_mp,
                       0u}));
      e.record_weight(gw, f, h, "moe_w4a8_nax_m5_v1",
                      "moe_w4a8_swiglu_bm64_out64_sk1_v1", {},
                      "legacy_route_v1", {}, false, explicit_t);
      e.record_weight(uw, f, h, "moe_w4a8_nax_m5_v1",
                      "moe_w4a8_swiglu_bm64_out64_sk1_v1", {},
                      "legacy_route_v1", {}, false, explicit_t);

      if (dw.has_fwht64()) {
        auto &mid_rot = e.emit(
            "moe_nax", "omlx_m5_moe_nax_mid_fwht64_bf16",
            {.width = U64(f / 64u) * 32u, .height = U64(max_tiles) * 8u},
            {.width = 32u}, {ws.mid_f16, ws.total_tiles});
        e.dynamic(mid_rot, 2, words({t, top_k, f, max_tiles}), tok_w0);
      }

      auto &dn_nax = e.emit(
          "moe_nax", "omlx_m5_moe_mxfp4_nax_down_scatter_bm64_bn64_sk1",
          make_wide_grid(dn_x_tiles, 128u), {.width = 128u},
          {ws.mid_f16, ws.tile_expert, ws.tile_valid_rows, ws.tile_assign,
           ws.total_tiles, route_scores, plane(dw, "data"), plane(dw, "scale"),
           expert_output});
      e.dynamic(dn_nax, 9,
                words({h, f, U32(dw.padded_k), gemm_max_tiles_word}));
      e.record_weight(dw, h, f, "moe_nax_m5_v1", "moe_nax_down_bm64_sk1_v1", {},
                      "legacy_route_v1", {}, false, explicit_t);
    }
  } else {
    U32 ub = 0, ug = 0, db = 0, dg = 0;
    (void)moe_nax_affine_parse(gw.format, ub, ug);
    (void)moe_nax_affine_parse(dw.format, db, dg);

    auto code_plane = [](const MaterializedWeightView &w, U32 bits) -> Ref {
      if (bits == 3u) return plane(w, "low_2bit");
      if (bits == 8u) return plane(w, "code");
      return plane(w, "base_code");
    };
    auto high_plane = [](const MaterializedWeightView &w, U32 bits) -> Ref {
      if (bits == 3u || bits == 5u) return plane(w, "high_bit");
      return plane(w, "high_2bit");
    };

    if (use_w4a8_down) {
      const std::string up_fn = "omlx_m5_moe_affine_w4a8_nax_swiglu_q" +
                                std::to_string(ub) + "_g" + std::to_string(ug) +
                                "_bm64_out64_sk1_fwht64_a8";
      if (ub == 4u || ub == 8u) {
        auto &up_nax = e.emit(
            "w4a8_moe_nax", up_fn, make_wide_grid(up_x_tiles_64, 128u),
            {.width = 128u},
            {a8_qa, a8_sa, a8_ra, ws.tile_assign, ws.tile_expert,
             ws.tile_valid_rows, ws.total_tiles, code_plane(gw, ub),
             plane(gw, "scale"), plane(gw, "bias"), code_plane(uw, ub),
             plane(uw, "scale"), plane(uw, "bias"), ws.mid_qa, ws.mid_sa,
             ws.mid_ra});
        e.dynamic(up_nax, 16,
                  words({f, h, U32(gw.padded_k), gemm_max_tiles_word, a8_mp,
                         mid_reserved}));
      } else {
        auto &up_nax = e.emit(
            "w4a8_moe_nax", up_fn, make_wide_grid(up_x_tiles_64, 128u),
            {.width = 128u},
            {a8_qa, a8_sa, a8_ra, ws.tile_assign, ws.tile_expert,
             ws.tile_valid_rows, ws.total_tiles, code_plane(gw, ub),
             high_plane(gw, ub), plane(gw, "scale"), plane(gw, "bias"),
             code_plane(uw, ub), high_plane(uw, ub), plane(uw, "scale"),
             plane(uw, "bias"), ws.mid_qa, ws.mid_sa, ws.mid_ra});
        e.dynamic(up_nax, 18,
                  words({f, h, U32(gw.padded_k), gemm_max_tiles_word, a8_mp,
                         mid_reserved}));
      }
      e.record_weight(gw, f, h, "moe_w4a8_nax_m5_v2",
                      "moe_affine_w4a8_swiglu_bm64_out64_sk1_fwht64_a8_v2", {},
                      "legacy_route_v1", {}, false, explicit_t);
      e.record_weight(uw, f, h, "moe_w4a8_nax_m5_v2",
                      "moe_affine_w4a8_swiglu_bm64_out64_sk1_fwht64_a8_v2", {},
                      "legacy_route_v1", {}, false, explicit_t);

      const std::string dn_fn = "omlx_m5_moe_affine_w4a8_nax_down_scatter_q" +
                                std::to_string(db) + "_g" + std::to_string(dg) +
                                "_bm64_bn64_sk1";
      if (db == 4u || db == 8u) {
        auto &dn_nax = e.emit(
            "w4a8_moe_nax", dn_fn, make_wide_grid(dn_x_tiles, 128u),
            {.width = 128u},
            {ws.mid_qa, ws.mid_sa, ws.mid_ra, ws.tile_expert,
             ws.tile_valid_rows, ws.tile_assign, ws.total_tiles, route_scores,
             code_plane(dw, db), plane(dw, "scale"), plane(dw, "bias"),
             expert_output});
        e.dynamic(dn_nax, 12,
                  words({h, f, U32(dw.padded_k), gemm_max_tiles_word,
                         ws.max_rows, 0u}));
      } else {
        auto &dn_nax = e.emit(
            "w4a8_moe_nax", dn_fn, make_wide_grid(dn_x_tiles, 128u),
            {.width = 128u},
            {ws.mid_qa, ws.mid_sa, ws.mid_ra, ws.tile_expert,
             ws.tile_valid_rows, ws.tile_assign, ws.total_tiles, route_scores,
             code_plane(dw, db), high_plane(dw, db), plane(dw, "scale"),
             plane(dw, "bias"), expert_output});
        e.dynamic(dn_nax, 13,
                  words({h, f, U32(dw.padded_k), gemm_max_tiles_word,
                         ws.max_rows, 0u}));
      }
      e.record_weight(dw, h, f, "moe_w4a8_nax_m5_v2",
                      "moe_affine_w4a8_down_bm64_bn64_sk1_v2", {},
                      "legacy_route_v1", {}, false, explicit_t);
    } else {
      const std::string up_fn = "omlx_m5_moe_affine_w4a8_nax_swiglu_q" +
                                std::to_string(ub) + "_g" + std::to_string(ug) +
                                "_bm64_out64_sk1";
      if (ub == 4u || ub == 8u) {
        auto &up_nax = e.emit(
            "w4a8_moe_nax", up_fn, make_wide_grid(up_x_tiles_32, 128u),
            {.width = 128u},
            {a8_qa, a8_sa, a8_ra, ws.tile_assign, ws.tile_expert,
             ws.tile_valid_rows, ws.total_tiles, code_plane(gw, ub),
             plane(gw, "scale"), plane(gw, "bias"), code_plane(uw, ub),
             plane(uw, "scale"), plane(uw, "bias"), ws.mid_f16});
        e.dynamic(up_nax, 14,
                  words({f, h, U32(gw.padded_k), gemm_max_tiles_word, a8_mp,
                         0u}));
      } else {
        auto &up_nax = e.emit(
            "w4a8_moe_nax", up_fn, make_wide_grid(up_x_tiles_32, 128u),
            {.width = 128u},
            {a8_qa, a8_sa, a8_ra, ws.tile_assign, ws.tile_expert,
             ws.tile_valid_rows, ws.total_tiles, code_plane(gw, ub),
             high_plane(gw, ub), plane(gw, "scale"), plane(gw, "bias"),
             code_plane(uw, ub), high_plane(uw, ub), plane(uw, "scale"),
             plane(uw, "bias"), ws.mid_f16});
        e.dynamic(up_nax, 16,
                  words({f, h, U32(gw.padded_k), gemm_max_tiles_word, a8_mp,
                         0u}));
      }
      e.record_weight(gw, f, h, "moe_w4a8_nax_m5_v1",
                      "moe_affine_w4a8_swiglu_bm64_out64_sk1_v1", {},
                      "legacy_route_v1", {}, false, explicit_t);
      e.record_weight(uw, f, h, "moe_w4a8_nax_m5_v1",
                      "moe_affine_w4a8_swiglu_bm64_out64_sk1_v1", {},
                      "legacy_route_v1", {}, false, explicit_t);

      if (dw.has_fwht64()) {
        auto &mid_rot = e.emit(
            "moe_nax", "omlx_m5_moe_nax_mid_fwht64_bf16",
            {.width = U64(f / 64u) * 32u, .height = U64(max_tiles) * 8u},
            {.width = 32u}, {ws.mid_f16, ws.total_tiles});
        e.dynamic(mid_rot, 2, words({t, top_k, f, max_tiles}), tok_w0);
      }

      const std::string dn_fn = "omlx_m5_moe_affine_nax_down_scatter_q" +
                                std::to_string(db) + "_g" + std::to_string(dg) +
                                "_bm64_bn64_sk1";
      if (db == 4u || db == 8u) {
        auto &dn_nax = e.emit(
            "moe_nax", dn_fn, make_wide_grid(dn_x_tiles, 128u), {.width = 128u},
            {ws.mid_f16, ws.tile_expert, ws.tile_valid_rows, ws.tile_assign,
             ws.total_tiles, route_scores, code_plane(dw, db),
             plane(dw, "scale"), plane(dw, "bias"), expert_output});
        e.dynamic(dn_nax, 10,
                  words({h, f, U32(dw.padded_k), gemm_max_tiles_word}));
      } else {
        auto &dn_nax = e.emit(
            "moe_nax", dn_fn, make_wide_grid(dn_x_tiles, 128u), {.width = 128u},
            {ws.mid_f16, ws.tile_expert, ws.tile_valid_rows, ws.tile_assign,
             ws.total_tiles, route_scores, code_plane(dw, db),
             high_plane(dw, db), plane(dw, "scale"), plane(dw, "bias"),
             expert_output});
        e.dynamic(dn_nax, 11,
                  words({h, f, U32(dw.padded_k), gemm_max_tiles_word}));
      }
      e.record_weight(dw, h, f, "moe_nax_m5_v1",
                      "moe_affine_nax_down_bm64_sk1_v1", {}, "legacy_route_v1",
                      {}, false, explicit_t);
    }
  }

  if (emit_reduce) {
    auto &reduce =
        e.emit("mlp", "omlx_m5_moe_topk_reduce", {.width = U64(t) * h},
               {.width = 128u}, {expert_output, routed});
    e.dynamic(reduce, 2, words({t, top_k, h, 0u}), tok_w0);
  }
  return true;
}

// ===========================================================================
// Dim B T=1 Pure-AR Fused 9-Expert (Top-8 Routed + 1 Shared) Lowering Helpers
// ===========================================================================

inline bool moe9_t1_enabled() {
  if (const char *env = std::getenv("OMLX_M5_DISABLE_MOE9_T1");
      env != nullptr && std::string_view(env) == "1") {
    return false;
  }
  return true;
}

inline bool can_moe9_router_fused_t1(const MaterializedWeightView &router_w,
                                     const AffineNaxCache *affine_cache, U32 h,
                                     U32 num_experts = 256u) {
  if (!moe9_t1_enabled() || num_experts != 256u || router_w.has_fwht64() ||
      router_w.parent_shape.size() != 2u || router_w.row_count != 256u ||
      router_w.in_features != h || h < 64u || (h % 64u) != 0u) {
    return false;
  }
  if (router_w.format == "direct_bf16_v1") {
    return true;
  }
  if (affine_nax_bits(router_w.format) == 8u) {
    if (affine_cache != nullptr) {
      if (const auto *tw = affine_cache->find(router_w.object_id);
          tw != nullptr && tw->bits == 8u && tw->bn == 16u && tw->n == 256u &&
          tw->k == h) {
        return true;
      }
    }
    if (is_oq_nax_pretiled_layout(router_w.layout)) {
      return true;
    }
    const U32 gs = affine_nax_group_size(router_w.format);
    if (router_w.layout == "oq_rowsplit_k128_v1" && (gs == 64u || gs == 128u) &&
        router_w.padded_k >= h && (router_w.padded_k % gs) == 0u) {
      return true;
    }
  }
  return false;
}

inline bool can_moe9_shared_gate_fused_t1(const MaterializedWeightView &sgate_w,
                                          const AffineNaxCache *affine_cache,
                                          U32 h) {
  if (sgate_w.has_fwht64() || sgate_w.parent_shape.size() != 2u ||
      sgate_w.row_count != 1u || sgate_w.in_features != h || h < 64u ||
      (h % 64u) != 0u) {
    return false;
  }
  if (sgate_w.format == "direct_bf16_v1") {
    return true;
  }
  if (affine_nax_bits(sgate_w.format) == 8u) {
    if (affine_cache != nullptr) {
      if (const auto *tw = affine_cache->find(sgate_w.object_id);
          tw != nullptr && tw->bits == 8u && tw->bn == 16u && tw->n == 1u &&
          tw->k == h) {
        return true;
      }
    }
    if (is_oq_nax_pretiled_layout(sgate_w.layout)) {
      return true;
    }
    const U32 gs = affine_nax_group_size(sgate_w.format);
    if (sgate_w.layout == "oq_rowsplit_k128_v1" && (gs == 64u || gs == 128u) &&
        sgate_w.padded_k >= h && (sgate_w.padded_k % gs) == 0u) {
      return true;
    }
  }
  return false;
}

inline bool moe9_resolve_shared_mxfp4(const MaterializedWeightView &w,
                                      const Mxfp4NaxCache *mxfp4_cache, U32 n,
                                      U32 k, Ref *out_codes, Ref *out_scales,
                                      bool *out_tiled) {
  if (w.format != "mxfp4_e2m1_g32_e8m0_v1" || w.parent_shape.size() != 2u ||
      w.row_count != n || w.in_features != k || n >= 65536u || k < 128u ||
      (k % 128u) != 0u) {
    return false;
  }
  if (mxfp4_cache != nullptr) {
    if (const auto *tw = mxfp4_cache->find(w.object_id);
        tw != nullptr && tw->bn == 16u && tw->n == n && tw->k == k) {
      if (out_codes) *out_codes = tw->codes;
      if (out_scales) *out_scales = tw->scales;
      if (out_tiled) *out_tiled = true;
      return true;
    }
  }
  if (w.layout == "mx_nax_tiled_bk128_v1") {
    if (out_codes) *out_codes = plane(w, "nax_data");
    if (out_scales) *out_scales = plane(w, "nax_scale");
    if (out_tiled) *out_tiled = true;
    return true;
  }
  if (w.layout == "mx_rowsplit_k128_v1" && w.padded_k >= k &&
      (w.padded_k % 128u) == 0u) {
    if (out_codes) *out_codes = plane(w, "data");
    if (out_scales) *out_scales = plane(w, "scale");
    if (out_tiled) *out_tiled = false;
    return true;
  }
  return false;
}

inline bool can_moe9_mxfp4_t1(
    const MoeNaxWorkspace &ws, const MaterializedWeightView &gw,
    const MaterializedWeightView &uw, const MaterializedWeightView &dw,
    const MaterializedWeightView &shared_gw,
    const MaterializedWeightView &shared_uw,
    const MaterializedWeightView &shared_dw, const Mxfp4NaxCache *mxfp4_cache,
    U32 t, U32 h, U32 f, U32 shared_f, U32 top_k = 8u, U32 num_experts = 256u) {
  if (!moe9_t1_enabled() || !ws.active || t != 1u || top_k != 8u ||
      num_experts != 256u || shared_f != f || gw.has_fwht64() ||
      uw.has_fwht64() || shared_gw.has_fwht64() || shared_uw.has_fwht64()) {
    return false;
  }
  if (h < 128u || (h % 128u) != 0u || (h % 8u) != 0u || f < 128u ||
      (f % 128u) != 0u || (f % 16u) != 0u) {
    return false;
  }
  const auto is_expert_rowsplit = [](std::string_view layout) {
    return layout == "mx_expert_rowsplit_k128_v1" ||
           layout == "mx_rowsplit_k128_v1";
  };
  if (gw.format != "mxfp4_e2m1_g32_e8m0_v1" ||
      uw.format != "mxfp4_e2m1_g32_e8m0_v1" ||
      dw.format != "mxfp4_e2m1_g32_e8m0_v1" ||
      !is_expert_rowsplit(gw.layout) || !is_expert_rowsplit(uw.layout) ||
      !is_expert_rowsplit(dw.layout) || gw.padded_k != uw.padded_k ||
      gw.padded_k < h || (gw.padded_k % 128u) != 0u || dw.padded_k < f ||
      (dw.padded_k % 128u) != 0u) {
    return false;
  }
  bool tg = false, tu = false, td = false;
  if (!moe9_resolve_shared_mxfp4(shared_gw, mxfp4_cache, f, h, nullptr, nullptr,
                                 &tg) ||
      !moe9_resolve_shared_mxfp4(shared_uw, mxfp4_cache, f, h, nullptr, nullptr,
                                 &tu) ||
      !moe9_resolve_shared_mxfp4(shared_dw, mxfp4_cache, h, f, nullptr, nullptr,
                                 &td)) {
    return false;
  }
  if (tg != tu || (!tg && shared_gw.padded_k != shared_uw.padded_k)) {
    return false;
  }
  return ws.mid_f16.bytes >= U64(top_k + 1u) * U64(f) * 2ULL;
}

// Emits 1-TG omlx_m5_moe9_router_top8_shared_gate_t1 (computing router_logits,
// route_ids[8], route_scores[8], and optionally shared_gate_out[0] in one dispatch).
// Sets *fused_shared_gate_out = true when sgate_w was fused into this dispatch.
inline bool emit_moe9_router_top8_t1(
    Encoder &e, const MaterializedWeightView &router_w,
    const MaterializedWeightView *sgate_w, const AffineNaxCache *affine_cache,
    Ref norm, Ref zero, Ref router_logits_out, Ref route_ids_out,
    Ref route_scores_out, Ref shared_gate_out, U32 h,
    std::optional<U32> explicit_t = {}, bool *fused_shared_gate_out = nullptr) {
  if (fused_shared_gate_out) *fused_shared_gate_out = false;
  if (!can_moe9_router_fused_t1(router_w, affine_cache, h, 256u)) {
    return false;
  }

  U32 r_mode = 0u, r_pad_k = h, r_gs = 64u;
  Ref r_c = zero, r_a0 = zero, r_a1 = zero;
  if (router_w.format == "direct_bf16_v1") {
    r_mode = 0u;
    r_c = plane(router_w, "data");
  } else {
    r_gs = affine_nax_group_size(router_w.format);
    const TiledAffineWeight *tw =
        affine_cache ? affine_cache->find(router_w.object_id) : nullptr;
    if (tw && tw->bits == 8u && tw->bn == 16u && tw->n == 256u && tw->k == h) {
      r_mode = 1u;
      r_c = tw->codes;
      r_a0 = tw->sb;
      r_gs |= 0x80000000u;
    } else if (is_oq_nax_pretiled_layout(router_w.layout)) {
      r_mode = 1u;
      r_c = plane(router_w, "nax_codes");
      r_a0 = plane(router_w, "nax_sb");
      if (is_oq_nax_lane_v2_layout(router_w.layout)) {
        r_gs |= 0x80000000u;
      }
    } else {
      r_mode = 2u;
      r_pad_k = U32(router_w.padded_k);
      r_c = plane(router_w, "code");
      r_a0 = plane(router_w, "scale");
      r_a1 = plane(router_w, "bias");
    }
  }

  U32 sg_mode = 0u, sg_pad_k = h, sg_gs = 64u, sg_pad_n = 16u;
  Ref sg_c = zero, sg_a0 = zero, sg_a1 = zero;
  if (sgate_w != nullptr && shared_gate_out.buffer != nullptr &&
      can_moe9_shared_gate_fused_t1(*sgate_w, affine_cache, h)) {
    if (sgate_w->format == "direct_bf16_v1") {
      sg_mode = 1u;
      sg_c = plane(*sgate_w, "data");
    } else {
      sg_gs = affine_nax_group_size(sgate_w->format);
      const TiledAffineWeight *tw =
          affine_cache ? affine_cache->find(sgate_w->object_id) : nullptr;
      if (tw && tw->bits == 8u && tw->bn == 16u && tw->n == 1u && tw->k == h) {
        sg_mode = 2u;
        sg_c = tw->codes;
        sg_a0 = tw->sb;
        sg_pad_n = tw->padded_n;
        sg_gs |= 0x80000000u;
      } else if (is_oq_nax_pretiled_layout(sgate_w->layout)) {
        sg_mode = 2u;
        sg_c = plane(*sgate_w, "nax_codes");
        sg_a0 = plane(*sgate_w, "nax_sb");
        sg_pad_n = 16u;
        if (is_oq_nax_lane_v2_layout(sgate_w->layout)) {
          sg_gs |= 0x80000000u;
        }
      } else {
        sg_mode = 3u;
        sg_pad_k = U32(sgate_w->padded_k);
        sg_c = plane(*sgate_w, "code");
        sg_a0 = plane(*sgate_w, "scale");
        sg_a1 = plane(*sgate_w, "bias");
      }
    }
    if (fused_shared_gate_out) *fused_shared_gate_out = true;
  }

  Ref out_sg = shared_gate_out.buffer != nullptr ? shared_gate_out : router_logits_out;
  const U32 num_router_tgs = (sg_mode != 0u) ? 65u : 64u;
  const char *router_fn = (r_mode == 1u && (sg_mode == 0u || sg_mode == 2u))
                              ? "omlx_m5_moe9_router_q8_pretiled_65tg_t1"
                              : "omlx_m5_moe9_router_qmv_64tg_t1";
  auto &d = e.emit(
      "moe_nax", router_fn, {.width = U64(num_router_tgs) * 64u},
      {.width = 64u},
      {norm, r_c, r_a0, r_a1, sg_c, sg_a0, sg_a1, router_logits_out, out_sg});
  e.dynamic(d, 9,
            words({h, r_pad_k, r_gs, r_mode, sg_pad_k, sg_gs, sg_pad_n,
                   sg_mode}));
  auto &top8 = e.emit(
      "qwen_forward", "omlx_m5_forward_router_top8_logits",
      {.width = 32u, .height = 1u}, {.width = 32u},
      {router_logits_out, route_ids_out, route_scores_out});
  e.dynamic(top8, 3, words({1u, 256u}),
            explicit_t.has_value() ? std::optional<U32>{}
                                   : std::optional<U32>{0u});
  e.record_weight(router_w, 256u, h, "moe9_t1_v1", "moe9_router_64tg_t1_v2", {},
                  "legacy_route_v1", {}, false, explicit_t);
  if (sg_mode != 0u && sgate_w != nullptr) {
    e.record_weight(*sgate_w, 1u, h, "moe9_t1_v1", "moe9_router_sgate_t1_v2",
                    {}, "legacy_route_v1", {}, false, explicit_t);
  }
  return true;
}

// Emits the 2-dispatch T=1 9-expert (top-8 routed + 1 shared) MXFP4 FFN:
//   1. omlx_m5_moe9_mxfp4_kl16_swiglu_t1  ((f/4)*64, 9, 1) -> ws.mid_f16 (as [9, f] BF16)
//   2. omlx_m5_moe9_mxfp4_kl16_down_sk3_r8_t1 ((h/8)*96, 1, 1) -> delta_out ([1, h] BF16)
// Plugs directly into qwen_forward.cpp's add_normalize(hidden, delta, norm) with 0 extra add dispatch!
inline bool emit_moe9_mxfp4_t1(
    Encoder &e, const MoeNaxWorkspace &ws, const MaterializedWeightView &gw,
    const MaterializedWeightView &uw, const MaterializedWeightView &dw,
    const MaterializedWeightView &shared_gw,
    const MaterializedWeightView &shared_uw,
    const MaterializedWeightView &shared_dw, const Mxfp4NaxCache *mxfp4_cache,
    Ref norm, Ref route_ids, Ref route_scores, Ref shared_gate_logit,
    Ref delta_out, U32 h, U32 f, std::optional<U32> explicit_t = {},
    Ref residual_add_in = {}) {
  if (!can_moe9_mxfp4_t1(ws, gw, uw, dw, shared_gw, shared_uw, shared_dw,
                         mxfp4_cache, 1u, h, f, f, 8u, 256u)) {
    return false;
  }

  Ref sgc{}, sgs{}, suc{}, sus{}, sdc{}, sds{};
  bool up_tiled = false, u_tiled = false, dn_tiled = false;
  (void)moe9_resolve_shared_mxfp4(shared_gw, mxfp4_cache, f, h, &sgc, &sgs,
                                  &up_tiled);
  (void)moe9_resolve_shared_mxfp4(shared_uw, mxfp4_cache, f, h, &suc, &sus,
                                  &u_tiled);
  (void)moe9_resolve_shared_mxfp4(shared_dw, mxfp4_cache, h, f, &sdc, &sds,
                                  &dn_tiled);

  const U32 up_shared_pad_k = up_tiled ? h : U32(shared_gw.padded_k);
  auto &up_d = e.emit(
      "moe_nax", "omlx_m5_moe9_mxfp4_kl16_swiglu_t1",
      {.width = U64((f + 3u) / 4u) * 64u, .height = 9u}, {.width = 64u},
      {norm, route_ids, plane(gw, "data"), plane(gw, "scale"),
       plane(uw, "data"), plane(uw, "scale"), sgc, sgs, suc, sus, ws.mid_f16});
  e.dynamic(up_d, 11,
            words({f, h, U32(gw.padded_k), up_shared_pad_k,
                   up_tiled ? 1u : 0u, 256u, 8u, 0u}));
  e.record_weight(gw, f, h, "moe9_t1_v1", "moe9_mxfp4_swiglu_t1_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(uw, f, h, "moe9_t1_v1", "moe9_mxfp4_swiglu_t1_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(shared_gw, f, h, "moe9_t1_v1", "moe9_mxfp4_swiglu_t1_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(shared_uw, f, h, "moe9_t1_v1", "moe9_mxfp4_swiglu_t1_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);

  const bool add_res = (residual_add_in.buffer != nullptr);
  const Ref norm_or_res = add_res ? residual_add_in : norm;
  const U32 dn_flags = (add_res ? 2u : 0u) | (dw.has_fwht64() ? 4u : 0u) |
                       (shared_dw.has_fwht64() ? 8u : 0u);
  const U32 dn_shared_pad_k = dn_tiled ? f : U32(shared_dw.padded_k);
  auto &dn_d = e.emit(
      "moe_nax", "omlx_m5_moe9_mxfp4_kl16_down_sk3_r8_t1",
      {.width = U64((h + 7u) / 8u) * 96u, .height = 1u}, {.width = 96u},
      {ws.mid_f16, route_ids, route_scores, shared_gate_logit, norm_or_res,
       plane(dw, "data"), plane(dw, "scale"), sdc, sds, delta_out});
  e.dynamic(dn_d, 10,
            words({f, h, U32(dw.padded_k), dn_shared_pad_k,
                   dn_tiled ? 1u : 0u, 256u, 8u, dn_flags}));
  e.record_weight(dw, h, f, "moe9_t1_v1", "moe9_mxfp4_down_t1_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(shared_dw, h, f, "moe9_t1_v1", "moe9_mxfp4_down_t1_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  return true;
}

// Emits 1-dispatch 64/65-TG router + atomic-tail top8 + concurrent shared_gate
// (+ optional inline prologue RMSNorm + cooperative norm_out write for moe9_swiglu_t1).
inline bool emit_moe9_router_fused_top8_t1(
    Encoder &e, const MaterializedWeightView &router_w,
    const MaterializedWeightView *sgate_w, const AffineNaxCache *affine_cache,
    bool prologue_norm, Ref x_or_hidden, Ref rms_w, Ref rms_sq_buf,
    Ref norm_out, Ref zero, Ref router_logits_out, Ref route_ids_out,
    Ref route_scores_out, Ref shared_gate_out, Ref moe_tg_done_buf, U32 h,
    U32 sq_slot, U32 moe_slot, float rms_eps, float w_off,
    std::optional<U32> explicit_t = {}, bool *fused_shared_gate_out = nullptr) {
  if (fused_shared_gate_out) *fused_shared_gate_out = false;
  if (!moe_tg_done_buf.buffer ||
      !can_moe9_router_fused_t1(router_w, affine_cache, h, 256u)) {
    return false;
  }
  if (prologue_norm &&
      (h > 16384u || !rms_w.buffer || !rms_sq_buf.buffer || !norm_out.buffer)) {
    return false;
  }

  U32 r_mode = 0u, r_pad_k = h, r_gs = 64u;
  Ref r_c = zero, r_a0 = zero, r_a1 = zero;
  if (router_w.format == "direct_bf16_v1") {
    r_mode = 0u;
    r_c = plane(router_w, "data");
  } else {
    r_gs = affine_nax_group_size(router_w.format);
    const TiledAffineWeight *tw =
        affine_cache ? affine_cache->find(router_w.object_id) : nullptr;
    if (tw && tw->bits == 8u && tw->bn == 16u && tw->n == 256u && tw->k == h) {
      r_mode = 1u;
      r_c = tw->codes;
      r_a0 = tw->sb;
      r_gs |= 0x80000000u;
    } else if (is_oq_nax_pretiled_layout(router_w.layout)) {
      r_mode = 1u;
      r_c = plane(router_w, "nax_codes");
      r_a0 = plane(router_w, "nax_sb");
      if (is_oq_nax_lane_v2_layout(router_w.layout)) {
        r_gs |= 0x80000000u;
      }
    } else {
      r_mode = 2u;
      r_pad_k = U32(router_w.padded_k);
      r_c = plane(router_w, "code");
      r_a0 = plane(router_w, "scale");
      r_a1 = plane(router_w, "bias");
    }
  }

  U32 sg_mode = 0u, sg_pad_k = h, sg_gs = 64u, sg_pad_n = 16u;
  Ref sg_c = zero, sg_a0 = zero, sg_a1 = zero;
  if (sgate_w != nullptr && shared_gate_out.buffer != nullptr &&
      can_moe9_shared_gate_fused_t1(*sgate_w, affine_cache, h)) {
    if (sgate_w->format == "direct_bf16_v1") {
      sg_mode = 1u;
      sg_c = plane(*sgate_w, "data");
    } else {
      sg_gs = affine_nax_group_size(sgate_w->format);
      const TiledAffineWeight *tw =
          affine_cache ? affine_cache->find(sgate_w->object_id) : nullptr;
      if (tw && tw->bits == 8u && tw->bn == 16u && tw->n == 1u && tw->k == h) {
        sg_mode = 2u;
        sg_c = tw->codes;
        sg_a0 = tw->sb;
        sg_pad_n = tw->padded_n;
        sg_gs |= 0x80000000u;
      } else if (is_oq_nax_pretiled_layout(sgate_w->layout)) {
        sg_mode = 2u;
        sg_c = plane(*sgate_w, "nax_codes");
        sg_a0 = plane(*sgate_w, "nax_sb");
        sg_pad_n = 16u;
        if (is_oq_nax_lane_v2_layout(sgate_w->layout)) {
          sg_gs |= 0x80000000u;
        }
      } else {
        sg_mode = 3u;
        sg_pad_k = U32(sgate_w->padded_k);
        sg_c = plane(*sgate_w, "code");
        sg_a0 = plane(*sgate_w, "scale");
        sg_a1 = plane(*sgate_w, "bias");
      }
    }
    if (fused_shared_gate_out) *fused_shared_gate_out = true;
  }

  Ref out_sg =
      shared_gate_out.buffer != nullptr ? shared_gate_out : router_logits_out;
  const U32 num_router_tgs = (sg_mode != 0u) ? 65u : 64u;
  const bool fast_q8 = (r_mode == 1u && (sg_mode == 0u || sg_mode == 2u));
  if (prologue_norm) {
    const char *router_fn =
        fast_q8 ? "omlx_m5_moe9_router_q8_pretiled_norm_fused_top8_t1"
                : "omlx_m5_moe9_router_qmv_norm_fused_top8_t1";
    auto &d = e.emit(
        "moe_nax", router_fn, {.width = U64(num_router_tgs) * 64u},
        {.width = 64u},
        {x_or_hidden, rms_w, rms_sq_buf, norm_out, r_c, r_a0, r_a1, sg_c, sg_a0,
         sg_a1, router_logits_out, route_ids_out, route_scores_out, out_sg,
         moe_tg_done_buf});
    e.dynamic(d, 15,
              words({h, r_pad_k, r_gs, r_mode, sg_pad_k, sg_gs, sg_pad_n,
                     sg_mode, sq_slot, moe_slot, word(1.0f / float(h)),
                     word(rms_eps), word(w_off)}));
  } else {
    const char *router_fn =
        fast_q8 ? "omlx_m5_moe9_router_q8_pretiled_fused_top8_t1"
                : "omlx_m5_moe9_router_qmv_fused_top8_t1";
    auto &d = e.emit(
        "moe_nax", router_fn, {.width = U64(num_router_tgs) * 64u},
        {.width = 64u},
        {x_or_hidden, r_c, r_a0, r_a1, sg_c, sg_a0, sg_a1, router_logits_out,
         route_ids_out, route_scores_out, out_sg, moe_tg_done_buf});
    e.dynamic(d, 12,
              words({h, r_pad_k, r_gs, r_mode, sg_pad_k, sg_gs, sg_pad_n,
                     sg_mode, 0u, moe_slot, 0u, 0u, 0u}));
  }
  e.record_weight(router_w, 256u, h, "moe9_t1_v1", "moe9_router_fused_top8_v3",
                  {}, "legacy_route_v1", {}, false, explicit_t);
  if (sg_mode != 0u && sgate_w != nullptr) {
    e.record_weight(*sgate_w, 1u, h, "moe9_t1_v1", "moe9_router_sgate_t1_v3",
                    {}, "legacy_route_v1", {}, false, explicit_t);
  }
  return true;
}

// Emits the 2-dispatch T=1 9-expert MXFP4 FFN with fused residual-add + RMS sq-sum epilogue:
//   1. omlx_m5_moe9_mxfp4_kl16_swiglu_t1           -> ws.mid_f16
//   2. omlx_m5_moe9_mxfp4_kl16_down_sk3_r8_add_sq_t1 -> hidden_inout += delta, rms_sq_buf[sq_slot] += sq
inline bool emit_moe9_mxfp4_add_sq_t1(
    Encoder &e, const MoeNaxWorkspace &ws, const MaterializedWeightView &gw,
    const MaterializedWeightView &uw, const MaterializedWeightView &dw,
    const MaterializedWeightView &shared_gw,
    const MaterializedWeightView &shared_uw,
    const MaterializedWeightView &shared_dw, const Mxfp4NaxCache *mxfp4_cache,
    Ref norm, Ref route_ids, Ref route_scores, Ref shared_gate_logit,
    Ref hidden_inout, Ref rms_sq_buf, U32 h, U32 f, U32 sq_slot,
    std::optional<U32> explicit_t = {}) {
  if (!rms_sq_buf.buffer ||
      !can_moe9_mxfp4_t1(ws, gw, uw, dw, shared_gw, shared_uw, shared_dw,
                         mxfp4_cache, 1u, h, f, f, 8u, 256u)) {
    return false;
  }

  Ref sgc{}, sgs{}, suc{}, sus{}, sdc{}, sds{};
  bool up_tiled = false, u_tiled = false, dn_tiled = false;
  (void)moe9_resolve_shared_mxfp4(shared_gw, mxfp4_cache, f, h, &sgc, &sgs,
                                  &up_tiled);
  (void)moe9_resolve_shared_mxfp4(shared_uw, mxfp4_cache, f, h, &suc, &sus,
                                  &u_tiled);
  (void)moe9_resolve_shared_mxfp4(shared_dw, mxfp4_cache, h, f, &sdc, &sds,
                                  &dn_tiled);

  const U32 up_shared_pad_k = up_tiled ? h : U32(shared_gw.padded_k);
  auto &up_d = e.emit(
      "moe_nax", "omlx_m5_moe9_mxfp4_kl16_swiglu_t1",
      {.width = U64((f + 3u) / 4u) * 64u, .height = 9u}, {.width = 64u},
      {norm, route_ids, plane(gw, "data"), plane(gw, "scale"),
       plane(uw, "data"), plane(uw, "scale"), sgc, sgs, suc, sus, ws.mid_f16});
  e.dynamic(up_d, 11,
            words({f, h, U32(gw.padded_k), up_shared_pad_k,
                   up_tiled ? 1u : 0u, 256u, 8u, 0u}));
  e.record_weight(gw, f, h, "moe9_t1_v1", "moe9_mxfp4_swiglu_t1_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(uw, f, h, "moe9_t1_v1", "moe9_mxfp4_swiglu_t1_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(shared_gw, f, h, "moe9_t1_v1", "moe9_mxfp4_swiglu_t1_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(shared_uw, f, h, "moe9_t1_v1", "moe9_mxfp4_swiglu_t1_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);

  const U32 dn_shared_pad_k = dn_tiled ? f : U32(shared_dw.padded_k);
  const U32 dn_flags = (sq_slot << 8u) | (dw.has_fwht64() ? 4u : 0u) |
                       (shared_dw.has_fwht64() ? 8u : 0u);
  auto &dn_d = e.emit(
      "moe_nax", "omlx_m5_moe9_mxfp4_kl16_down_sk3_r8_add_sq_t1",
      {.width = U64((h + 7u) / 8u) * 96u, .height = 1u}, {.width = 96u},
      {ws.mid_f16, route_ids, route_scores, shared_gate_logit, norm,
       plane(dw, "data"), plane(dw, "scale"), sdc, sds, hidden_inout,
       rms_sq_buf});
  e.dynamic(dn_d, 11,
            words({f, h, U32(dw.padded_k), dn_shared_pad_k,
                   dn_tiled ? 1u : 0u, 256u, 8u, dn_flags}));
  e.record_weight(dw, h, f, "moe9_t1_v1", "moe9_mxfp4_down_add_sq_t1_v3", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(shared_dw, h, f, "moe9_t1_v1", "moe9_mxfp4_down_add_sq_t1_v3",
                  {}, "legacy_route_v1", {}, false, explicit_t);
  return true;
}

// ===========================================================================
// Dim D Batched (B in [2, 8]) Zero-Spill 3-Dispatch 9-Expert MoE Helpers
// ===========================================================================

inline bool moe9_bt_enabled() {
  if (!moe9_t1_enabled()) {
    return false;
  }
  if (const char *env = std::getenv("OMLX_M5_DISABLE_MOE9_BT");
      env != nullptr && std::string_view(env) == "1") {
    return false;
  }
  return true;
}

inline bool can_moe9_router_fused_bt(const MaterializedWeightView &router_w,
                                     const AffineNaxCache *affine_cache, U32 t,
                                     U32 h, U32 num_experts = 256u) {
  if (!moe9_bt_enabled() || t < 2u || t > 8u) {
    return false;
  }
  return can_moe9_router_fused_t1(router_w, affine_cache, h, num_experts);
}

inline bool can_moe9_mxfp4_bt(
    const MoeNaxWorkspace &ws, const MaterializedWeightView &gw,
    const MaterializedWeightView &uw, const MaterializedWeightView &dw,
    const MaterializedWeightView &shared_gw,
    const MaterializedWeightView &shared_uw,
    const MaterializedWeightView &shared_dw, const Mxfp4NaxCache *mxfp4_cache,
    U32 t, U32 h, U32 f, U32 shared_f, U32 top_k = 8u, U32 num_experts = 256u) {
  if (!moe9_bt_enabled() || !ws.active || t < 2u || t > 8u || top_k != 8u ||
      num_experts != 256u || shared_f != f || gw.has_fwht64() ||
      uw.has_fwht64() || shared_gw.has_fwht64() || shared_uw.has_fwht64()) {
    return false;
  }
  if (h < 128u || (h % 128u) != 0u || (h % 8u) != 0u || f < 128u ||
      (f % 128u) != 0u || (f % 16u) != 0u) {
    return false;
  }
  const auto is_expert_rowsplit = [](std::string_view layout) {
    return layout == "mx_expert_rowsplit_k128_v1" ||
           layout == "mx_rowsplit_k128_v1";
  };
  if (gw.format != "mxfp4_e2m1_g32_e8m0_v1" ||
      uw.format != "mxfp4_e2m1_g32_e8m0_v1" ||
      dw.format != "mxfp4_e2m1_g32_e8m0_v1" ||
      !is_expert_rowsplit(gw.layout) || !is_expert_rowsplit(uw.layout) ||
      !is_expert_rowsplit(dw.layout) || gw.padded_k != uw.padded_k ||
      gw.padded_k < h || (gw.padded_k % 128u) != 0u || dw.padded_k < f ||
      (dw.padded_k % 128u) != 0u) {
    return false;
  }
  bool tg = false, tu = false, td = false;
  if (!moe9_resolve_shared_mxfp4(shared_gw, mxfp4_cache, f, h, nullptr, nullptr,
                                 &tg) ||
      !moe9_resolve_shared_mxfp4(shared_uw, mxfp4_cache, f, h, nullptr, nullptr,
                                 &tu) ||
      !moe9_resolve_shared_mxfp4(shared_dw, mxfp4_cache, h, f, nullptr, nullptr,
                                 &td)) {
    return false;
  }
  if (tg != tu || (!tg && shared_gw.padded_k != shared_uw.padded_k)) {
    return false;
  }
  return ws.mid_f16.bytes >= U64(t) * U64(top_k + 1u) * U64(f) * 2ULL;
}

// Emits 1-dispatch batched (grid = (64/65, rows, 1)) router + self-resetting
// atomic-tail top8 + concurrent shared_gate (+ optional inline prologue RMSNorm
// reading rms_sq_buf[sq_slot * 8 + b] and cooperatively writing norm_out[b]).
// Note: moe_tg_done_buf[moe_slot * 8 + b] self-resets to 0 inside the elected
// last TG for each row b, so callers may reuse moe_slot = 0 across layers (32 B)
// or pass per-layer moe_slot if moe_tg_done_buf.bytes >= (moe_slot * 8 + rows) * 4.
inline bool emit_moe9_router_fused_top8_bt(
    Encoder &e, const MaterializedWeightView &router_w,
    const MaterializedWeightView *sgate_w, const AffineNaxCache *affine_cache,
    bool prologue_norm, Ref x_or_hidden, Ref rms_w, Ref rms_sq_buf,
    Ref norm_out, Ref zero, Ref router_logits_out, Ref route_ids_out,
    Ref route_scores_out, Ref shared_gate_out, Ref moe_tg_done_buf, U32 rows,
    U32 h, U32 sq_slot, U32 moe_slot, float rms_eps, float w_off,
    std::optional<U32> explicit_t = {}, bool *fused_shared_gate_out = nullptr) {
  if (fused_shared_gate_out) *fused_shared_gate_out = false;
  if (!moe_tg_done_buf.buffer ||
      !can_moe9_router_fused_bt(router_w, affine_cache, rows, h, 256u)) {
    return false;
  }
  if (moe_tg_done_buf.bytes < U64(moe_slot * 8u + rows) * 4ULL) {
    return false;
  }
  if (!x_or_hidden.buffer || x_or_hidden.bytes < U64(rows) * U64(h) * 2ULL ||
      !router_logits_out.buffer ||
      router_logits_out.bytes < U64(rows) * 256ULL * 2ULL ||
      !route_ids_out.buffer || route_ids_out.bytes < U64(rows) * 8ULL * 4ULL ||
      !route_scores_out.buffer ||
      route_scores_out.bytes < U64(rows) * 8ULL * 2ULL) {
    return false;
  }
  if (prologue_norm &&
      (h > 16384u || !rms_w.buffer || !rms_sq_buf.buffer ||
       rms_sq_buf.bytes < U64(sq_slot * 8u + rows) * 4ULL ||
       !norm_out.buffer || norm_out.bytes < U64(rows) * U64(h) * 2ULL)) {
    return false;
  }

  U32 r_mode = 0u, r_pad_k = h, r_gs = 64u;
  Ref r_c = zero, r_a0 = zero, r_a1 = zero;
  if (router_w.format == "direct_bf16_v1") {
    r_mode = 0u;
    r_c = plane(router_w, "data");
  } else {
    r_gs = affine_nax_group_size(router_w.format);
    const TiledAffineWeight *tw =
        affine_cache ? affine_cache->find(router_w.object_id) : nullptr;
    if (tw && tw->bits == 8u && tw->bn == 16u && tw->n == 256u && tw->k == h) {
      r_mode = 1u;
      r_c = tw->codes;
      r_a0 = tw->sb;
      r_gs |= 0x80000000u;
    } else if (is_oq_nax_pretiled_layout(router_w.layout)) {
      r_mode = 1u;
      r_c = plane(router_w, "nax_codes");
      r_a0 = plane(router_w, "nax_sb");
      if (is_oq_nax_lane_v2_layout(router_w.layout)) {
        r_gs |= 0x80000000u;
      }
    } else {
      r_mode = 2u;
      r_pad_k = U32(router_w.padded_k);
      r_c = plane(router_w, "code");
      r_a0 = plane(router_w, "scale");
      r_a1 = plane(router_w, "bias");
    }
  }

  U32 sg_mode = 0u, sg_pad_k = h, sg_gs = 64u, sg_pad_n = 16u;
  Ref sg_c = zero, sg_a0 = zero, sg_a1 = zero;
  if (sgate_w != nullptr && shared_gate_out.buffer != nullptr &&
      shared_gate_out.bytes >= U64(rows) * 2ULL &&
      can_moe9_shared_gate_fused_t1(*sgate_w, affine_cache, h)) {
    if (sgate_w->format == "direct_bf16_v1") {
      sg_mode = 1u;
      sg_c = plane(*sgate_w, "data");
    } else {
      sg_gs = affine_nax_group_size(sgate_w->format);
      const TiledAffineWeight *tw =
          affine_cache ? affine_cache->find(sgate_w->object_id) : nullptr;
      if (tw && tw->bits == 8u && tw->bn == 16u && tw->n == 1u && tw->k == h) {
        sg_mode = 2u;
        sg_c = tw->codes;
        sg_a0 = tw->sb;
        sg_pad_n = tw->padded_n;
        sg_gs |= 0x80000000u;
      } else if (is_oq_nax_pretiled_layout(sgate_w->layout)) {
        sg_mode = 2u;
        sg_c = plane(*sgate_w, "nax_codes");
        sg_a0 = plane(*sgate_w, "nax_sb");
        sg_pad_n = 16u;
        if (is_oq_nax_lane_v2_layout(sgate_w->layout)) {
          sg_gs |= 0x80000000u;
        }
      } else {
        sg_mode = 3u;
        sg_pad_k = U32(sgate_w->padded_k);
        sg_c = plane(*sgate_w, "code");
        sg_a0 = plane(*sgate_w, "scale");
        sg_a1 = plane(*sgate_w, "bias");
      }
    }
    if (fused_shared_gate_out) *fused_shared_gate_out = true;
  }

  Ref out_sg =
      shared_gate_out.buffer != nullptr ? shared_gate_out : router_logits_out;
  const U32 num_router_tgs = (sg_mode != 0u) ? 65u : 64u;
  const bool fast_q8 = (r_mode == 1u && (sg_mode == 0u || sg_mode == 2u));
  if (prologue_norm) {
    const char *router_fn =
        fast_q8 ? "omlx_m5_moe9_router_q8_pretiled_norm_fused_top8_bt"
                : "omlx_m5_moe9_router_qmv_norm_fused_top8_bt";
    auto &d = e.emit(
        "moe_nax", router_fn,
        {.width = U64(num_router_tgs) * 64u, .height = rows}, {.width = 64u},
        {x_or_hidden, rms_w, rms_sq_buf, norm_out, r_c, r_a0, r_a1, sg_c, sg_a0,
         sg_a1, router_logits_out, route_ids_out, route_scores_out, out_sg,
         moe_tg_done_buf});
    e.dynamic(d, 15,
              words({h, r_pad_k, r_gs, r_mode, sg_pad_k, sg_gs, sg_pad_n,
                     sg_mode, sq_slot, moe_slot, word(1.0f / float(h)),
                     word(rms_eps), word(w_off), rows}));
  } else {
    const char *router_fn =
        fast_q8 ? "omlx_m5_moe9_router_q8_pretiled_fused_top8_bt"
                : "omlx_m5_moe9_router_qmv_fused_top8_bt";
    auto &d = e.emit(
        "moe_nax", router_fn,
        {.width = U64(num_router_tgs) * 64u, .height = rows}, {.width = 64u},
        {x_or_hidden, r_c, r_a0, r_a1, sg_c, sg_a0, sg_a1, router_logits_out,
         route_ids_out, route_scores_out, out_sg, moe_tg_done_buf});
    e.dynamic(d, 12,
              words({h, r_pad_k, r_gs, r_mode, sg_pad_k, sg_gs, sg_pad_n,
                     sg_mode, 0u, moe_slot, 0u, 0u, 0u, rows}));
  }
  e.record_weight(router_w, 256u, h, "moe9_bt_v1", "moe9_router_fused_top8_bt_v1",
                  {}, "legacy_route_v1", {}, false, explicit_t);
  if (sg_mode != 0u && sgate_w != nullptr) {
    e.record_weight(*sgate_w, 1u, h, "moe9_bt_v1", "moe9_router_sgate_bt_v1", {},
                    "legacy_route_v1", {}, false, explicit_t);
  }
  return true;
}

// Emits the 2-dispatch batched (B in [2, 8]) 9-expert MXFP4 FFN:
//   1. omlx_m5_moe9_mxfp4_kl16_swiglu_bt      ((f/4)*64, 9*rows, 1) -> ws.mid_f16 ([rows, 9, f] BF16)
//   2. omlx_m5_moe9_mxfp4_kl16_down_sk3_r8_bt ((h/8)*96, rows, 1)   -> delta_out ([rows, h] BF16)
inline bool emit_moe9_mxfp4_bt(
    Encoder &e, const MoeNaxWorkspace &ws, const MaterializedWeightView &gw,
    const MaterializedWeightView &uw, const MaterializedWeightView &dw,
    const MaterializedWeightView &shared_gw,
    const MaterializedWeightView &shared_uw,
    const MaterializedWeightView &shared_dw, const Mxfp4NaxCache *mxfp4_cache,
    Ref norm, Ref route_ids, Ref route_scores, Ref shared_gate_logit,
    Ref delta_out, U32 rows, U32 h, U32 f, std::optional<U32> explicit_t = {},
    Ref residual_add_in = {}) {
  if (!can_moe9_mxfp4_bt(ws, gw, uw, dw, shared_gw, shared_uw, shared_dw,
                         mxfp4_cache, rows, h, f, f, 8u, 256u)) {
    return false;
  }
  if (!norm.buffer || norm.bytes < U64(rows) * U64(h) * 2ULL ||
      !route_ids.buffer || route_ids.bytes < U64(rows) * 8ULL * 4ULL ||
      !route_scores.buffer || route_scores.bytes < U64(rows) * 8ULL * 2ULL ||
      !shared_gate_logit.buffer ||
      shared_gate_logit.bytes < U64(rows) * 2ULL || !delta_out.buffer ||
      delta_out.bytes < U64(rows) * U64(h) * 2ULL) {
    return false;
  }
  const bool add_res = (residual_add_in.buffer != nullptr);
  if (add_res && residual_add_in.bytes < U64(rows) * U64(h) * 2ULL) {
    return false;
  }

  Ref sgc{}, sgs{}, suc{}, sus{}, sdc{}, sds{};
  bool up_tiled = false, u_tiled = false, dn_tiled = false;
  (void)moe9_resolve_shared_mxfp4(shared_gw, mxfp4_cache, f, h, &sgc, &sgs,
                                  &up_tiled);
  (void)moe9_resolve_shared_mxfp4(shared_uw, mxfp4_cache, f, h, &suc, &sus,
                                  &u_tiled);
  (void)moe9_resolve_shared_mxfp4(shared_dw, mxfp4_cache, h, f, &sdc, &sds,
                                  &dn_tiled);

  const U32 up_shared_pad_k = up_tiled ? h : U32(shared_gw.padded_k);
  auto &up_d = e.emit(
      "moe_nax", "omlx_m5_moe9_mxfp4_kl16_swiglu_bt",
      {.width = U64((f + 3u) / 4u) * 64u, .height = U64(rows) * 9ULL},
      {.width = 64u},
      {norm, route_ids, plane(gw, "data"), plane(gw, "scale"),
       plane(uw, "data"), plane(uw, "scale"), sgc, sgs, suc, sus, ws.mid_f16});
  e.dynamic(up_d, 11,
            words({f, h, U32(gw.padded_k), up_shared_pad_k,
                   up_tiled ? 1u : 0u, 256u, 8u, 0u, rows}));
  e.record_weight(gw, f, h, "moe9_bt_v1", "moe9_mxfp4_swiglu_bt_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(uw, f, h, "moe9_bt_v1", "moe9_mxfp4_swiglu_bt_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(shared_gw, f, h, "moe9_bt_v1", "moe9_mxfp4_swiglu_bt_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(shared_uw, f, h, "moe9_bt_v1", "moe9_mxfp4_swiglu_bt_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);

  const Ref norm_or_res = add_res ? residual_add_in : norm;
  const U32 dn_flags = (add_res ? 2u : 0u) | (dw.has_fwht64() ? 4u : 0u) |
                       (shared_dw.has_fwht64() ? 8u : 0u);
  const U32 dn_shared_pad_k = dn_tiled ? f : U32(shared_dw.padded_k);
  auto &dn_d = e.emit(
      "moe_nax", "omlx_m5_moe9_mxfp4_kl16_down_sk3_r8_bt",
      {.width = U64((h + 7u) / 8u) * 96u, .height = rows}, {.width = 96u},
      {ws.mid_f16, route_ids, route_scores, shared_gate_logit, norm_or_res,
       plane(dw, "data"), plane(dw, "scale"), sdc, sds, delta_out});
  e.dynamic(dn_d, 10,
            words({f, h, U32(dw.padded_k), dn_shared_pad_k,
                   dn_tiled ? 1u : 0u, 256u, 8u, dn_flags, rows}));
  e.record_weight(dw, h, f, "moe9_bt_v1", "moe9_mxfp4_down_bt_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(shared_dw, h, f, "moe9_bt_v1", "moe9_mxfp4_down_bt_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  return true;
}

// Emits the 2-dispatch batched (B in [2, 8]) 9-expert MXFP4 FFN with fused
// residual-add + per-row RMS sq-sum epilogue (accumulating rms_sq_buf[sq_slot * 8 + b]):
//   1. omlx_m5_moe9_mxfp4_kl16_swiglu_bt             -> ws.mid_f16
//   2. omlx_m5_moe9_mxfp4_kl16_down_sk3_r8_add_sq_bt -> hidden_inout += delta, rms_sq_buf[sq_slot*8+b] += sq
inline bool emit_moe9_mxfp4_add_sq_bt(
    Encoder &e, const MoeNaxWorkspace &ws, const MaterializedWeightView &gw,
    const MaterializedWeightView &uw, const MaterializedWeightView &dw,
    const MaterializedWeightView &shared_gw,
    const MaterializedWeightView &shared_uw,
    const MaterializedWeightView &shared_dw, const Mxfp4NaxCache *mxfp4_cache,
    Ref norm, Ref route_ids, Ref route_scores, Ref shared_gate_logit,
    Ref hidden_inout, Ref rms_sq_buf, U32 rows, U32 h, U32 f, U32 sq_slot,
    std::optional<U32> explicit_t = {}) {
  if (!rms_sq_buf.buffer ||
      rms_sq_buf.bytes < U64(sq_slot * 8u + rows) * 4ULL ||
      !can_moe9_mxfp4_bt(ws, gw, uw, dw, shared_gw, shared_uw, shared_dw,
                         mxfp4_cache, rows, h, f, f, 8u, 256u)) {
    return false;
  }
  if (!norm.buffer || norm.bytes < U64(rows) * U64(h) * 2ULL ||
      !route_ids.buffer || route_ids.bytes < U64(rows) * 8ULL * 4ULL ||
      !route_scores.buffer || route_scores.bytes < U64(rows) * 8ULL * 2ULL ||
      !shared_gate_logit.buffer ||
      shared_gate_logit.bytes < U64(rows) * 2ULL || !hidden_inout.buffer ||
      hidden_inout.bytes < U64(rows) * U64(h) * 2ULL) {
    return false;
  }

  Ref sgc{}, sgs{}, suc{}, sus{}, sdc{}, sds{};
  bool up_tiled = false, u_tiled = false, dn_tiled = false;
  (void)moe9_resolve_shared_mxfp4(shared_gw, mxfp4_cache, f, h, &sgc, &sgs,
                                  &up_tiled);
  (void)moe9_resolve_shared_mxfp4(shared_uw, mxfp4_cache, f, h, &suc, &sus,
                                  &u_tiled);
  (void)moe9_resolve_shared_mxfp4(shared_dw, mxfp4_cache, h, f, &sdc, &sds,
                                  &dn_tiled);

  const U32 up_shared_pad_k = up_tiled ? h : U32(shared_gw.padded_k);
  auto &up_d = e.emit(
      "moe_nax", "omlx_m5_moe9_mxfp4_kl16_swiglu_bt",
      {.width = U64((f + 3u) / 4u) * 64u, .height = U64(rows) * 9ULL},
      {.width = 64u},
      {norm, route_ids, plane(gw, "data"), plane(gw, "scale"),
       plane(uw, "data"), plane(uw, "scale"), sgc, sgs, suc, sus, ws.mid_f16});
  e.dynamic(up_d, 11,
            words({f, h, U32(gw.padded_k), up_shared_pad_k,
                   up_tiled ? 1u : 0u, 256u, 8u, 0u, rows}));
  e.record_weight(gw, f, h, "moe9_bt_v1", "moe9_mxfp4_swiglu_bt_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(uw, f, h, "moe9_bt_v1", "moe9_mxfp4_swiglu_bt_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(shared_gw, f, h, "moe9_bt_v1", "moe9_mxfp4_swiglu_bt_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(shared_uw, f, h, "moe9_bt_v1", "moe9_mxfp4_swiglu_bt_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);

  const U32 dn_shared_pad_k = dn_tiled ? f : U32(shared_dw.padded_k);
  const U32 dn_flags = (sq_slot << 8u) | (dw.has_fwht64() ? 4u : 0u) |
                       (shared_dw.has_fwht64() ? 8u : 0u);
  auto &dn_d = e.emit(
      "moe_nax", "omlx_m5_moe9_mxfp4_kl16_down_sk3_r8_add_sq_bt",
      {.width = U64((h + 7u) / 8u) * 96u, .height = rows}, {.width = 96u},
      {ws.mid_f16, route_ids, route_scores, shared_gate_logit, norm,
       plane(dw, "data"), plane(dw, "scale"), sdc, sds, hidden_inout,
       rms_sq_buf});
  e.dynamic(dn_d, 11,
            words({f, h, U32(dw.padded_k), dn_shared_pad_k,
                   dn_tiled ? 1u : 0u, 256u, 8u, dn_flags, rows}));
  e.record_weight(dw, h, f, "moe9_bt_v1", "moe9_mxfp4_down_add_sq_bt_v1", {},
                  "legacy_route_v1", {}, false, explicit_t);
  e.record_weight(shared_dw, h, f, "moe9_bt_v1", "moe9_mxfp4_down_add_sq_bt_v1",
                  {}, "legacy_route_v1", {}, false, explicit_t);
  return true;
}

} // namespace omlx::m5::detail::forward
