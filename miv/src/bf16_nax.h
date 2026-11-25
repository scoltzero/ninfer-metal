#pragma once

#include "nax_common.h"

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace omlx::m5::detail::forward {

inline bool bf16_nax_runtime_eligible(const MetalContext &ctx,
                                      const std::filesystem::path &libs) {
  return nax_apple10_eligible(ctx, libs, "omlx_m5_bf16_nax.metallib",
                              "OMLX_M5_DISABLE_BF16_NAX");
}

inline bool bf16_nax_weight_eligible(const MaterializedWeightView &w, U32 n,
                                     U32 k) {
  return w.format == "direct_bf16_v1" && w.parent_shape.size() == 2 &&
         w.row_count == n && w.in_features == k &&
         (w.padded_k == 0u || w.padded_k == k) && k >= 256u &&
         (k % 64u) == 0u && n >= 16u;
}

inline U32 bf16_nax_bk(U32 k) { return (k % 128u) == 0u ? 128u : 64u; }

inline U32 bf16_nax_bn(U32 n, U32 bk) {
  if (const char *env = std::getenv("OMLX_M5_BF16_NAX_BN")) {
    const U32 v = U32(std::strtoul(env, nullptr, 10));
    if (v == 16u || v == 32u || (v == 64u && bk == 128u)) {
      return v;
    }
  }
  return n >= 65536u ? 32u : 16u;
}

inline U32 bf16_nax_split_k(U32 n, U32 k, U32 bk, U32 bn) {
  const U32 blocks = k / bk;
  // Tuned on Apple M5 Ultra 64-GPU: BN=16, SK=4 maximizes occupancy (3 KiB TG
  // mem) and beats SK=8 / BN=32 / BN=64 on DFlash2 5-layer BF16 backbone.
  U32 sk = 4u;
  if (n >= 65536u) {
    sk = 2u;
  }
  if (const char *env = std::getenv("OMLX_M5_BF16_NAX_SK")) {
    const U32 v = U32(std::strtoul(env, nullptr, 10));
    if (v == 1u || v == 2u || v == 4u || (v == 8u && bn <= 32u)) {
      sk = v;
    }
  }
  while (sk > 1u && sk > blocks) {
    sk /= 2u;
  }
  if (bk == 64u && bn == 32u && sk == 1u && blocks >= 2u) {
    sk = 2u;
  }
  if (bk == 64u && bn == 32u && sk > 4u) {
    sk = 4u;
  }
  if (bn == 64u && sk > 4u) {
    sk = 4u;
  }
  return sk;
}

inline U32 bf16_nax_mp(U32 t) { return ((t + 7u) / 8u) * 8u; }

inline void linear_bf16_nax(Encoder &e, const MaterializedWeightView &w, Ref x,
                            Ref y, U32 n, U32 k,
                            std::optional<U32> explicit_t = {},
                            U32 gpu_cores = 64u) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = bf16_nax_mp(t);
  const U32 sel_t = nax_effective_select_t(e, explicit_t);
  const auto sel_exp = nax_effective_select_explicit_t(e, explicit_t);
  const U32 bk = bf16_nax_bk(k);
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);

  const U32 default_padded_n = ((n + 15u) / 16u) * 16u;
  const NaxWidePrefillTile wide_tile =
      (std::getenv("OMLX_M5_BF16_NAX_BN") == nullptr &&
       std::getenv("OMLX_M5_BF16_NAX_SK") == nullptr)
          ? nax_select_wide_prefill_tile(default_padded_n, sel_t, sel_exp,
                                         gpu_cores)
          : NaxWidePrefillTile::none;
  if (wide_tile != NaxWidePrefillTile::none) {
    const bool is_64 = (wide_tile == NaxWidePrefillTile::bm64_bn64);
    const U32 tile_mn = is_64 ? 64u : 32u;
    const U32 tptg = is_64 ? 128u : 32u;
    const U32 x_tiles = (n + tile_mn - 1u) / tile_mn;
    const U32 y_tiles = (t + tile_mn - 1u) / tile_mn;
    const std::string tag = is_64 ? "bm64_bn64" : "bm32_bn32";
    const std::string fn =
        "omlx_m5_bf16_nax_bk" + std::to_string(bk) + "_" + tag + "_sk1";
    const MetalDispatchSize grid{.width = U64(x_tiles) * tptg,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = tptg};
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    auto &d = e.emit("bf16_nax", fn, grid, threads, {x, plane(w, "data"), y});
    e.dynamic(d, 3, words({n, k, mp, t}), tokens_patch);
    e.record_weight(w, n, k, "bf16_nax_m5_v1", "bf16_nax_m5_v1", {},
                    "linear_row_exact_portable_v1",
                    "bf16.nax.bk" + std::to_string(bk) + "." + tag + ".sk1.v1",
                    true, explicit_t);
    return;
  }

  const U32 bn = bf16_nax_bn(n, bk);
  const U32 sk = bf16_nax_split_k(n, k, bk, bn);
  const U32 padded_n = ((n + bn - 1u) / bn) * bn;
  const U32 rf = nax_select_row_factor(sel_t);
  const U32 bm =
      (rf == 1u && sel_t >= 9u &&
       ((bn == 16u && (sk == 2u || sk == 4u || sk == 8u)) ||
        (bn == 32u && sk == 2u)))
          ? ((sel_t >= 17u) ? 32u : 16u)
          : 8u;
  const U32 sel_y_tiles =
      (bm > 8u) ? ((sel_t + bm - 1u) / bm) : nax_y_tiles(bf16_nax_mp(sel_t), rf);
  const U32 y_tiles =
      (bm > 8u) ? ((t + bm - 1u) / bm) : nax_y_tiles(mp, rf);
  const U32 nsg =
      (bm == 8u && bn == 16u)
          ? nax_select_nsg_bn16(padded_n, sel_y_tiles, sk, gpu_cores, true)
          : 1u;

  std::string fn = "omlx_m5_bf16_nax_bk" + std::to_string(bk) +
                   (bm > 8u ? ("_bm" + std::to_string(bm)) : std::string{}) +
                   "_n" + std::to_string(bn) + "_sk" + std::to_string(sk);
  if (bm == 8u && bn == 16u && (rf > 1u || nsg > 1u)) {
    fn += "_rf" + std::to_string(rf) + "_nsg" + std::to_string(nsg);
  } else if (bm == 8u && rf > 1u) {
    fn += "_rf" + std::to_string(rf);
  }
  const U32 x_tiles = (padded_n / bn + nsg - 1u) / nsg;
  const MetalDispatchSize grid{.width = U64(x_tiles) * 32u * sk * nsg,
                               .height = y_tiles};
  const MetalDispatchSize threads{.width = 32u * sk * nsg};
  e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
  auto &d = e.emit("bf16_nax", fn, grid, threads, {x, plane(w, "data"), y});
  e.dynamic(d, 3, words({n, k, mp, t}), tokens_patch);
  e.record_weight(w, n, k, "bf16_nax_m5_v1", "bf16_nax_m5_v1", {},
                  "linear_row_exact_portable_v1",
                  "bf16.nax.bk" + std::to_string(bk) + ".n" +
                      std::to_string(bn) + ".sk" + std::to_string(sk) + ".v1",
                  true, explicit_t);
}

inline void linear_bf16_nax_swiglu(Encoder &e,
                                   const MaterializedWeightView &gw,
                                   const MaterializedWeightView &uw, Ref x,
                                   Ref y, U32 n, U32 k,
                                   U32 gpu_cores = 64u,
                                   std::optional<U32> explicit_t = {}) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = bf16_nax_mp(t);
  const U32 sel_t = nax_effective_select_t(e, explicit_t);
  const auto sel_exp = nax_effective_select_explicit_t(e, explicit_t);
  const U32 bk = bf16_nax_bk(k);
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  const U32 default_padded_n = ((n + 15u) / 16u) * 16u;
  const NaxWidePrefillTile wide_tile =
      (std::getenv("OMLX_M5_BF16_NAX_SWIGLU_BN") == nullptr &&
       std::getenv("OMLX_M5_BF16_NAX_SWIGLU_SK") == nullptr)
          ? nax_select_wide_prefill_tile(default_padded_n, sel_t, sel_exp,
                                         gpu_cores)
          : NaxWidePrefillTile::none;
  if (wide_tile != NaxWidePrefillTile::none) {
    const bool is_64 = (wide_tile == NaxWidePrefillTile::bm64_bn64);
    const U32 tile_m = is_64 ? 64u : 32u;
    const U32 tile_n = is_64 ? 32u : 16u;
    const U32 tptg = is_64 ? 128u : 32u;
    const U32 x_tiles = (n + tile_n - 1u) / tile_n;
    const U32 y_tiles = (t + tile_m - 1u) / tile_m;
    const std::string tag = is_64 ? "bm64_bn64" : "bm32_bn32";
    const std::string fn = "omlx_m5_bf16_nax_swiglu_bk" + std::to_string(bk) +
                           "_" + tag + "_sk1";
    const MetalDispatchSize grid{.width = U64(x_tiles) * tptg,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = tptg};
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    auto &d = e.emit("bf16_nax", fn, grid, threads,
                     {x, plane(gw, "data"), plane(uw, "data"), y});
    e.dynamic(d, 4, words({n, k, mp, t}), tokens_patch);
    const std::string grp =
        "bf16.nax.bk" + std::to_string(bk) + "." + tag + ".sk1.v1";
    e.record_weight(gw, n, k, "bf16_nax_m5_v1", "bf16_nax_m5_v1", {},
                    "linear_row_exact_portable_v1", grp, true, explicit_t);
    e.record_weight(uw, n, k, "bf16_nax_m5_v1", "bf16_nax_m5_v1", {},
                    "linear_row_exact_portable_v1", grp, true, explicit_t);
    return;
  }

  const U32 blocks = k / bk;
  const U32 rf = nax_select_row_factor(sel_t);
  const U32 y_tiles = nax_y_tiles(mp, rf);
  U32 bn = 16u;
  if (const char *env = std::getenv("OMLX_M5_BF16_NAX_SWIGLU_BN")) {
    const U32 v = U32(std::strtoul(env, nullptr, 10));
    if (v == 16u || (v == 32u && bk == 128u)) {
      bn = v;
    }
  }
  U32 sk = 4u;
  if (const char *env = std::getenv("OMLX_M5_BF16_NAX_SWIGLU_SK")) {
    const U32 v = U32(std::strtoul(env, nullptr, 10));
    if (v == 2u || v == 4u || (v == 8u && bn == 16u)) {
      sk = v;
    }
  }
  while (sk > 2u && sk > blocks) {
    sk /= 2u;
  }
  const U32 padded_n = ((n + bn - 1u) / bn) * bn;
  const U32 bm =
      (rf == 1u && sel_t >= 9u && bn == 16u &&
       (sk == 2u || sk == 4u || sk == 8u))
          ? ((sel_t >= 17u) ? 32u : 16u)
          : 8u;
  const U32 sw_y_tiles = (bm > 8u) ? ((t + bm - 1u) / bm) : y_tiles;
  std::string fn = "omlx_m5_bf16_nax_swiglu_bk" + std::to_string(bk) +
                   (bm > 8u ? ("_bm" + std::to_string(bm)) : std::string{}) +
                   "_n" + std::to_string(bn) + "_sk" + std::to_string(sk);
  if (bm == 8u && rf > 1u) {
    fn += "_rf" + std::to_string(rf);
  }
  const MetalDispatchSize grid{.width = U64(padded_n / bn) * 32u * sk,
                               .height = sw_y_tiles};
  const MetalDispatchSize threads{.width = 32u * sk};
  e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
  auto &d = e.emit("bf16_nax", fn, grid, threads,
                   {x, plane(gw, "data"), plane(uw, "data"), y});
  e.dynamic(d, 4, words({n, k, mp, t}), tokens_patch);
  const std::string grp = "bf16.nax.bk" + std::to_string(bk) + ".n" +
                          std::to_string(bn) + ".sk" + std::to_string(sk) +
                          ".v1";
  e.record_weight(gw, n, k, "bf16_nax_m5_v1", "bf16_nax_m5_v1", {},
                  "linear_row_exact_portable_v1", grp, true, explicit_t);
  e.record_weight(uw, n, k, "bf16_nax_m5_v1", "bf16_nax_m5_v1", {},
                  "linear_row_exact_portable_v1", grp, true, explicit_t);
}

} // namespace omlx::m5::detail::forward
