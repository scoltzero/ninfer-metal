#pragma once

#include "nax_common.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace omlx::m5::detail::forward {

struct TiledMxfp4Weight {
  Ref codes;
  Ref scales;
  U32 n = 0;
  U32 k = 0;
  U32 bn = 16;
  U32 padded_n = 0;
  U32 split_k = 1;
};

inline bool mxfp4_nax_runtime_eligible(const MetalContext &ctx,
                                       const std::filesystem::path &libs) {
  return nax_apple10_eligible(ctx, libs, "omlx_m5_mxfp4_nax.metallib",
                              "OMLX_M5_DISABLE_MXFP4_NAX");
}

inline bool mxfp4_nax_weight_eligible(const MaterializedWeightView &w) {
  return w.format == "mxfp4_e2m1_g32_e8m0_v1" &&
         (w.layout == "mx_rowsplit_k128_v1" ||
          w.layout == "mx_nax_tiled_bk128_v1") &&
         w.parent_shape.size() == 2 && w.row_count > 0 &&
         w.in_features >= 512 && (w.in_features % 128u) == 0 &&
         w.padded_k >= w.in_features && (w.padded_k % 128u) == 0;
}

inline U32 mxfp4_nax_bn(U32 n) { return n >= 65536u ? 32u : 16u; }

inline U32 normalize_mxfp4_nax_split_k(U32 target_slices) {
  if (target_slices >= 8u) return 8u;
  if (target_slices >= 4u) return 4u;
  if (target_slices >= 2u) return 2u;
  return 1u;
}

// Deterministic M-independent split-K selection tuned on Apple M5 Ultra:
// BM=8, BN=16, SK=8 wins every layer shape (N=1024..17408) at both M=1 and M=8
// when gpu_cores >= 64 (linear_split_k_target_slices=8), scaling cleanly down
// for smaller SKUs or calibrated overrides; BM=8, BN=32, SK<=2 wins lm_head.
inline U32 mxfp4_nax_split_k(U32 n, U32 k, U32 target_slices = 8u) {
  const U32 blocks = k / 128u;
  const U32 base_sk = normalize_mxfp4_nax_split_k(target_slices);
  U32 sk = (n >= 65536u) ? std::min<U32>(2u, base_sk) : base_sk;
  if (n < 65536u) {
    if (const char *env = std::getenv("OMLX_M5_MXFP4_NAX_SK")) {
      const U32 v = U32(std::strtoul(env, nullptr, 10));
      if (v == 1u || v == 2u || v == 4u || v == 8u) {
        sk = v;
      }
    }
  }
  while (sk > 1u && sk > blocks) {
    sk /= 2u;
  }
  return sk;
}

inline bool t_domain_dispatch_enabled(bool default_on = false) {
  if (const char *env = std::getenv("OMLX_M5_T_DOMAIN_DISPATCH")) {
    const std::string_view sv(env);
    if (sv == "1" || sv == "on" || sv == "ON" || sv == "true" || sv == "TRUE") {
      return true;
    }
    if (sv == "0" || sv == "off" || sv == "OFF" || sv == "false" ||
        sv == "FALSE") {
      return false;
    }
  }
  return default_on;
}

inline bool mxfp4_w4a4_prefill_enabled() {
  if (const char *env = std::getenv("OMLX_M5_ALLOW_W4A4_PREFILL");
      env != nullptr && std::string_view(env) == "1") {
    return true;
  }
  return false;
}

class Mxfp4NaxCache {
public:
  static constexpr bool kIncrementalOwnerCache = true;

  std::recursive_mutex &build_mutex() const noexcept { return build_mutex_; }

  void set_target_split_k(U32 target_slices) {
    std::lock_guard<std::recursive_mutex> lk(build_mutex_);
    target_split_k_ = normalize_mxfp4_nax_split_k(target_slices);
  }

  void register_weight(const MaterializedWeightView &w) {
    std::lock_guard<std::recursive_mutex> lk(build_mutex_);
    if (!mxfp4_nax_weight_eligible(w)) {
      return;
    }
    if (has_entry(w.object_id) || !seen_.insert(w.object_id).second) {
      return;
    }
    if (w.layout == "mx_nax_tiled_bk128_v1") {
      const U32 n = U32(w.row_count);
      const U32 k = U32(w.in_features);
      const U32 bn = mxfp4_nax_bn(n);
      const U32 padded_n = ((n + bn - 1u) / bn) * bn;
      auto entry = std::make_unique<TiledMxfp4Weight>(
          TiledMxfp4Weight{plane(w, "nax_data"), plane(w, "nax_scale"), n, k,
                           bn, padded_n,
                           mxfp4_nax_split_k(n, k, target_split_k_)});
      {
        std::lock_guard<std::mutex> elk(entries_mutex_);
        entries_.try_emplace(w.object_id, std::move(entry));
      }
      seen_.erase(w.object_id);
      return;
    }
    pending_.push_back(w);
  }

  // Optional zero-ULP column concatenation helper for sibling projections
  // sharing input x (e.g. [q, k, v] or [gdn_qkv, z, a, b]) where each non-final
  // slice row_count is a multiple of 16 and total_n < 65536.
  bool register_concat(U32 concat_id,
                       std::span<const MaterializedWeightView> parts) {
    std::lock_guard<std::recursive_mutex> lk(build_mutex_);
    if (parts.size() < 2 || has_entry(concat_id) || seen_.contains(concat_id)) {
      return false;
    }
    const U32 k = U32(parts.front().in_features);
    U32 total_n = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
      const auto &w = parts[i];
      if (!mxfp4_nax_weight_eligible(w) || U32(w.in_features) != k) {
        return false;
      }
      if (i + 1 < parts.size() && (w.row_count % 16u) != 0u) {
        return false;
      }
      total_n += U32(w.row_count);
    }
    if (total_n >= 65536u) {
      return false;
    }
    seen_.insert(concat_id);
    pending_concats_.push_back(
        {concat_id, {parts.begin(), parts.end()}, total_n, k});
    return true;
  }

  void build(const MetalContext &ctx, const std::filesystem::path &libs) {
    std::lock_guard<std::recursive_mutex> lk(build_mutex_);
    struct PendingResetGuard {
      Mxfp4NaxCache *self;
      ~PendingResetGuard() {
        self->pending_.clear();
        self->pending_concats_.clear();
        self->seen_.clear();
      }
    } reset_guard{this};

    if (pending_.empty() && pending_concats_.empty()) {
      return;
    }
    NaxSegmentArena arena;

    struct PendingPlan {
      MaterializedWeightView weight;
      U32 n = 0;
      U32 k = 0;
      U32 bn = 16;
      U32 padded_n = 0;
      U32 sk = 1;
      NaxSegmentArena::Placement codes;
      NaxSegmentArena::Placement scales;
    };
    std::vector<PendingPlan> plans;
    plans.reserve(pending_.size());
    for (const auto &w : pending_) {
      const U32 n = U32(w.row_count);
      const U32 k = U32(w.in_features);
      const U32 bn = mxfp4_nax_bn(n);
      const U32 padded_n = ((n + bn - 1u) / bn) * bn;
      const U64 code_bytes = mul(padded_n, k / 2u);
      const U64 scale_bytes = mul(padded_n, k / 32u);
      plans.push_back(
          {w, n, k, bn, padded_n, mxfp4_nax_split_k(n, k, target_split_k_),
           arena.plan_alloc(code_bytes, "MXFP4 NAX tiled weight"),
           arena.plan_alloc(scale_bytes, "MXFP4 NAX tiled scale")});
    }

    struct ConcatPlan {
      U32 concat_id = 0;
      std::vector<MaterializedWeightView> parts;
      U32 total_n = 0;
      U32 k = 0;
      U32 padded_n = 0;
      U32 sk = 1;
      NaxSegmentArena::Placement codes;
      NaxSegmentArena::Placement scales;
    };
    std::vector<ConcatPlan> cplans;
    cplans.reserve(pending_concats_.size());
    for (auto &cp : pending_concats_) {
      const U32 padded_n = ((cp.total_n + 15u) / 16u) * 16u;
      const U64 code_bytes = mul(padded_n, cp.k / 2u);
      const U64 scale_bytes = mul(padded_n, cp.k / 32u);
      cplans.push_back(
          {cp.concat_id, std::move(cp.parts), cp.total_n, cp.k, padded_n,
           mxfp4_nax_split_k(cp.total_n, cp.k, target_split_k_),
           arena.plan_alloc(code_bytes, "MXFP4 NAX concat weight"),
           arena.plan_alloc(scale_bytes, "MXFP4 NAX concat scale")});
    }

    const std::size_t base_seg = segments_.size();
    arena.materialize(ctx, segments_);

    std::unordered_map<U32, std::unique_ptr<TiledMxfp4Weight>> staged;

    static std::atomic<U64> repack_id{0};
    Encoder enc(ctx, libs, 1,
                "mxfp4.nax.repack." + std::to_string(++repack_id));
    enc.begin_stage("repack_mxfp4_bk128");
    for (const auto &pp : plans) {
      const Ref dst_codes = NaxSegmentArena::resolve(segments_, pp.codes, base_seg);
      const Ref dst_scales = NaxSegmentArena::resolve(segments_, pp.scales, base_seg);
      const U32 k_blocks = pp.k / 128u;
      const std::string fn = pp.bn == 32u ? "omlx_m5_repack_mxfp4_bn32_bk128"
                                          : "omlx_m5_repack_mxfp4_bn16_bk128";
      auto &d = enc.emit("mxfp4_nax", fn,
                         {.width = k_blocks, .height = pp.padded_n},
                         {.width = std::min(k_blocks, 16u), .height = 16u},
                         {plane(pp.weight, "data"), plane(pp.weight, "scale"),
                          dst_codes, dst_scales});
      enc.dynamic(d, 4, words({pp.n, pp.k, U32(pp.weight.padded_k), pp.padded_n}));
      staged.emplace(
          pp.weight.object_id,
          std::make_unique<TiledMxfp4Weight>(TiledMxfp4Weight{
              dst_codes, dst_scales, pp.n, pp.k, pp.bn, pp.padded_n, pp.sk}));
    }
    for (const auto &cp : cplans) {
      const Ref dst_codes = NaxSegmentArena::resolve(segments_, cp.codes, base_seg);
      const Ref dst_scales = NaxSegmentArena::resolve(segments_, cp.scales, base_seg);
      U32 row_off = 0;
      for (const auto &w : cp.parts) {
        const U32 wn = U32(w.row_count);
        const U32 wpadded_n = ((wn + 15u) / 16u) * 16u;
        if (w.layout == "mx_nax_tiled_bk128_v1") {
          const U64 c_off = mul(row_off, cp.k / 2u);
          const U64 c_len = mul(wpadded_n, cp.k / 2u);
          const U64 s_off = mul(row_off, cp.k / 32u);
          const U64 s_len = mul(wpadded_n, cp.k / 32u);
          const U32 c_words = U32(c_len / 4u);
          const U32 s_words = U32(s_len / 4u);
          auto &dc = enc.emit(
              "mxfp4_nax", "omlx_m5_copy_u32_words", {.width = c_words},
              {.width = 256u},
              {plane(w, "nax_data"),
               Ref{dst_codes.buffer, dst_codes.offset + c_off, c_len}});
          enc.dynamic(dc, 2, words({c_words}));
          auto &ds = enc.emit(
              "mxfp4_nax", "omlx_m5_copy_u32_words", {.width = s_words},
              {.width = 256u},
              {plane(w, "nax_scale"),
               Ref{dst_scales.buffer, dst_scales.offset + s_off, s_len}});
          enc.dynamic(ds, 2, words({s_words}));
        } else {
          const U32 k_blocks = cp.k / 128u;
          auto &d = enc.emit(
              "mxfp4_nax", "omlx_m5_repack_mxfp4_slice_bn16_bk128",
              {.width = k_blocks, .height = wpadded_n},
              {.width = std::min(k_blocks, 16u), .height = 16u},
              {plane(w, "data"), plane(w, "scale"), dst_codes, dst_scales});
          enc.dynamic(d, 4,
                      words({wn, cp.k, U32(w.padded_k), row_off, cp.padded_n}));
        }
        row_off += wn;
      }
      staged.emplace(
          cp.concat_id,
          std::make_unique<TiledMxfp4Weight>(TiledMxfp4Weight{
              dst_codes, dst_scales, cp.total_n, cp.k, 16u, cp.padded_n,
              cp.sk}));
    }
    auto profile = enc.finish();
    const std::array calls{MetalCommandInvocation{
        .plan = &profile.plan,
        .bindings = profile.bindings,
        .parameters = &profile.parameters}};
    (void)ctx.execute_batch(calls);
    {
      std::lock_guard<std::mutex> elk(entries_mutex_);
      for (auto &[id, entry] : staged) {
        entries_.try_emplace(id, std::move(entry));
      }
    }
  }

  const TiledMxfp4Weight *find(U32 object_id) const {
    std::lock_guard<std::mutex> elk(entries_mutex_);
    const auto it = entries_.find(object_id);
    return it == entries_.end() ? nullptr : it->second.get();
  }

  bool active() const {
    std::lock_guard<std::mutex> elk(entries_mutex_);
    return !entries_.empty();
  }

  static std::shared_ptr<Mxfp4NaxCache>
  acquire(const void *owner_key, const MetalContext &ctx,
          const std::filesystem::path &libs,
          const std::vector<MaterializedWeightView> &candidates,
          U32 target_split_k = 8u) {
    const U32 norm_sk = normalize_mxfp4_nax_split_k(target_split_k);
    return nax_acquire_owner_cache_variant<Mxfp4NaxCache>(
        owner_key, U64(norm_sk), [&](Mxfp4NaxCache &c) {
          c.set_target_split_k(norm_sk);
          for (const auto &w : candidates) {
            c.register_weight(w);
          }
          c.build(ctx, libs);
        });
  }

private:
  bool has_entry(U32 id) const {
    std::lock_guard<std::mutex> elk(entries_mutex_);
    return entries_.contains(id);
  }

  struct PendingConcat {
    U32 concat_id = 0;
    std::vector<MaterializedWeightView> parts;
    U32 total_n = 0;
    U32 k = 0;
  };
  mutable std::recursive_mutex build_mutex_;
  mutable std::mutex entries_mutex_;
  U32 target_split_k_ = 8u;
  std::unordered_set<U32> seen_;
  std::vector<MaterializedWeightView> pending_;
  std::vector<PendingConcat> pending_concats_;
  std::vector<std::unique_ptr<MetalBuffer>> segments_;
  std::unordered_map<U32, std::unique_ptr<TiledMxfp4Weight>> entries_;
};

inline U32 mxfp4_nax_mp(U32 t) { return ((t + 7u) / 8u) * 8u; }

inline void ensure_f16_cast(Encoder &e, Ref x, Ref f16_scratch,
                            Ref f16_row_scale, U32 k,
                            std::optional<U32> explicit_t = {}) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mxfp4_nax_mp(t);
  const bool dyn_rows = !explicit_t.has_value() && t != 1u;
  const Encoder::F16CastKey key{f16_scratch.buffer, f16_scratch.offset,
                                f16_row_scale.buffer, f16_row_scale.offset,
                                x.buffer, x.offset, k, t, dyn_rows};
  if (e.has_f16_cast(key)) {
    return;
  }
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  auto &d = e.emit("mxfp4_nax", "omlx_m5_bf16_to_f16_rowscale",
                   {.width = 256u, .height = mp}, {.width = 256u},
                   {x, f16_scratch, f16_row_scale});
  e.dynamic(d, 3, words({k, mp, 0u, t}), tokens_patch);
  e.set_f16_cast(key);
}

inline NaxWidePrefillTile
mxfp4_nax_select_wide(const TiledMxfp4Weight &tw, U32 t,
                      std::optional<U32> explicit_t = {}, U32 gpu_cores = 64u,
                      bool is_swiglu = false) {
  if (std::getenv("OMLX_M5_MXFP4_NAX_SK") != nullptr ||
      (is_swiglu && std::getenv("OMLX_M5_MXFP4_NAX_SWIGLU_SK") != nullptr)) {
    return NaxWidePrefillTile::none;
  }
  return nax_select_wide_prefill_tile(tw.padded_n, t, explicit_t, gpu_cores);
}

inline std::string mxfp4_nax_group(const Encoder &e,
                                   const TiledMxfp4Weight &tw,
                                   std::optional<U32> explicit_t = {},
                                   U32 gpu_cores = 64u,
                                   bool is_swiglu = false) {
  const U32 sel_t = nax_effective_select_t(e, explicit_t);
  const auto sel_exp = nax_effective_select_explicit_t(e, explicit_t);
  const auto wide =
      mxfp4_nax_select_wide(tw, sel_t, sel_exp, gpu_cores, is_swiglu);
  if (wide != NaxWidePrefillTile::none) {
    const char *tag =
        (wide == NaxWidePrefillTile::bm64_bn64) ? "bm64_bn64" : "bm32_bn32";
    return "mxfp4.nax.f16_bk128.bns" + std::to_string(tw.bn) + "." + tag +
           ".sk1.v1";
  }
  return "mxfp4.nax.f16_bk128.n" + std::to_string(tw.bn) + ".sk" +
         std::to_string(tw.split_k) + ".v1";
}

inline U32 mxfp4_nax_bm(U32 sel_t, U32 rf, U32 bn, U32 sk) {
  if (rf > 1u || (bn == 32u && sk > 2u)) {
    return 8u;
  }
  if (sel_t >= 17u) {
    return 32u;
  }
  if (sel_t >= 9u) {
    return 16u;
  }
  return 8u;
}

inline void linear_mxfp4_nax_direct_f16(Encoder &e, const TiledMxfp4Weight &tw,
                                        Ref x_f16, Ref row_scale, Ref y,
                                        std::optional<U32> explicit_t = {},
                                        U32 gpu_cores = 64u,
                                        bool unit_scale = false) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mxfp4_nax_mp(t);
  const U32 sel_t = nax_effective_select_t(e, explicit_t);
  const auto sel_exp = nax_effective_select_explicit_t(e, explicit_t);
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);

  const auto wide =
      mxfp4_nax_select_wide(tw, sel_t, sel_exp, gpu_cores, false);
  if (wide != NaxWidePrefillTile::none) {
    const bool is_64 = (wide == NaxWidePrefillTile::bm64_bn64);
    const U32 tile_mn = is_64 ? 64u : 32u;
    const U32 tptg = is_64 ? 128u : 32u;
    const U32 x_tiles = (tw.padded_n + tile_mn - 1u) / tile_mn;
    const U32 y_tiles = (t + tile_mn - 1u) / tile_mn;
    const std::string fn = std::string("omlx_m5_mxfp4_nax_") +
                           (is_64 ? "bm64_bn64" : "bm32_bn32") + "_bns" +
                           std::to_string(tw.bn) + "_sk1";
    const MetalDispatchSize grid{.width = U64(x_tiles) * tptg,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = tptg};
    const U32 mp_word = mp | (unit_scale ? 0x80000000u : 0u);
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    auto &d = e.emit("mxfp4_nax", fn, grid, threads,
                     {x_f16, row_scale, tw.codes, tw.scales, y});
    e.dynamic(d, 5, words({tw.n, tw.k, mp_word, t}), tokens_patch);
    return;
  }

  const U32 rf = nax_select_row_factor(sel_t);
  const U32 bm = mxfp4_nax_bm(sel_t, rf, tw.bn, tw.split_k);
  if (bm > 8u) {
    const U32 y_tiles = (mp + bm - 1u) / bm;
    const U32 x_tiles = tw.padded_n / tw.bn;
    const std::string fn = "omlx_m5_mxfp4_nax_bm" + std::to_string(bm) + "_n" +
                           std::to_string(tw.bn) + "_sk" +
                           std::to_string(tw.split_k);
    const MetalDispatchSize grid{.width = U64(x_tiles) * 32u * tw.split_k,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = 32u * tw.split_k};
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    auto &d = e.emit("mxfp4_nax", fn, grid, threads,
                     {x_f16, row_scale, tw.codes, tw.scales, y});
    e.dynamic(d, 5, words({tw.n, tw.k, mp, t}), tokens_patch);
    return;
  }

  const U32 sel_y_tiles = nax_y_tiles(mxfp4_nax_mp(sel_t), rf);
  const U32 y_tiles = nax_y_tiles(mp, rf);
  const U32 nsg =
      (tw.bn == 16u) ? nax_select_nsg_bn16(tw.padded_n, sel_y_tiles, tw.split_k,
                                           gpu_cores, true)
                     : 1u;

  std::string fn;
  if (rf == 1u && nsg == 1u) {
    fn = "omlx_m5_mxfp4_nax_n" + std::to_string(tw.bn) + "_sk" +
         std::to_string(tw.split_k);
  } else if (tw.bn == 16u) {
    fn = "omlx_m5_mxfp4_nax_n16_sk" + std::to_string(tw.split_k) + "_rf" +
         std::to_string(rf) + "_nsg" + std::to_string(nsg);
  } else {
    fn = "omlx_m5_mxfp4_nax_n32_sk" + std::to_string(tw.split_k) + "_rf" +
         std::to_string(rf);
  }
  const U32 x_tiles = (tw.padded_n / tw.bn + nsg - 1u) / nsg;
  const MetalDispatchSize grid{.width = U64(x_tiles) * 32u * tw.split_k * nsg,
                               .height = y_tiles};
  const MetalDispatchSize threads{.width = 32u * tw.split_k * nsg};
  e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
  auto &d = e.emit("mxfp4_nax", fn, grid, threads,
                   {x_f16, row_scale, tw.codes, tw.scales, y});
  e.dynamic(d, 5, words({tw.n, tw.k, mp, t}), tokens_patch);
}

inline void linear_mxfp4_nax(Encoder &e, const TiledMxfp4Weight &tw, Ref x,
                             Ref y, Ref f16_scratch, Ref f16_row_scale,
                             std::optional<U32> explicit_t = {},
                             U32 gpu_cores = 64u) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mxfp4_nax_mp(t);
  const U32 sel_t = nax_effective_select_t(e, explicit_t);
  const auto sel_exp = nax_effective_select_explicit_t(e, explicit_t);
  const auto wide =
      mxfp4_nax_select_wide(tw, sel_t, sel_exp, gpu_cores, false);
  if (wide != NaxWidePrefillTile::none) {
    const bool is_64 = (wide == NaxWidePrefillTile::bm64_bn64);
    const U32 tile_mn = is_64 ? 64u : 32u;
    const U32 tptg = is_64 ? 128u : 32u;
    const U32 x_tiles = (tw.padded_n + tile_mn - 1u) / tile_mn;
    const U32 y_tiles = (t + tile_mn - 1u) / tile_mn;
    const std::string fn = std::string("omlx_m5_mxfp4_nax_bf16_") +
                           (is_64 ? "bm64_bn64" : "bm32_bn32") + "_bns" +
                           std::to_string(tw.bn) + "_sk1";
    const MetalDispatchSize grid{.width = U64(x_tiles) * tptg,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = tptg};
    const std::optional<U32> tokens_patch =
        explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    auto &d = e.emit("mxfp4_nax", fn, grid, threads,
                     {x, tw.codes, tw.scales, y});
    e.dynamic(d, 4, words({tw.n, tw.k, mp, t}), tokens_patch);
    return;
  }
  ensure_f16_cast(e, x, f16_scratch, f16_row_scale, tw.k, explicit_t);
  linear_mxfp4_nax_direct_f16(e, tw, f16_scratch, f16_row_scale, y, explicit_t,
                              gpu_cores);
}

inline void linear_mxfp4_nax_swiglu_direct_f16(
    Encoder &e, const TiledMxfp4Weight &gw, const TiledMxfp4Weight &uw,
    Ref x_f16, Ref row_scale, Ref y_out, bool out_f16, U32 gpu_cores = 64u,
    std::optional<U32> explicit_t = {}, bool unit_scale = false) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mxfp4_nax_mp(t);
  const U32 sel_t = nax_effective_select_t(e, explicit_t);
  const auto sel_exp = nax_effective_select_explicit_t(e, explicit_t);
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);

  const auto wide =
      mxfp4_nax_select_wide(gw, sel_t, sel_exp, gpu_cores, true);
  if (wide != NaxWidePrefillTile::none) {
    const bool is_64 = (wide == NaxWidePrefillTile::bm64_bn64);
    const U32 tile_m = is_64 ? 64u : 32u;
    const U32 tile_n = is_64 ? 32u : 16u;
    const U32 tptg = is_64 ? 128u : 32u;
    const U32 x_tiles = (gw.padded_n + tile_n - 1u) / tile_n;
    const U32 y_rows = out_f16 ? mp : t;
    const U32 y_tiles = (y_rows + tile_m - 1u) / tile_m;
    const std::string fn =
        std::string(out_f16 ? "omlx_m5_mxfp4_nax_swiglu_f16_"
                            : "omlx_m5_mxfp4_nax_swiglu_") +
        (is_64 ? "bm64_bn64_sk1" : "bm32_bn32_sk1");
    const MetalDispatchSize grid{.width = U64(x_tiles) * tptg,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = tptg};
    const U32 mp_word = mp | (unit_scale ? 0x80000000u : 0u);
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    auto &d = e.emit("mxfp4_nax", fn, grid, threads,
                     {x_f16, row_scale, gw.codes, gw.scales, uw.codes,
                      uw.scales, y_out});
    e.dynamic(d, 7, words({gw.n, gw.k, mp_word, t}), tokens_patch);
    return;
  }

  const U32 rf = nax_select_row_factor(sel_t);
  U32 sk = gw.split_k;
  if (const char *env = std::getenv("OMLX_M5_MXFP4_NAX_SWIGLU_SK")) {
    const U32 v = U32(std::strtoul(env, nullptr, 10));
    if (v == 1u || v == 2u || v == 4u || v == 8u) {
      const U32 blocks = gw.k / 128u;
      sk = v;
      while (sk > 1u && sk > blocks) {
        sk /= 2u;
      }
    }
  }
  const U32 bm = mxfp4_nax_bm(sel_t, rf, 16u, sk);
  if (bm > 8u) {
    const U32 y_tiles = (mp + bm - 1u) / bm;
    const std::string fn =
        std::string(out_f16 ? "omlx_m5_mxfp4_nax_swiglu_f16_bm"
                            : "omlx_m5_mxfp4_nax_swiglu_bm") +
        std::to_string(bm) + "_n16_sk" + std::to_string(sk);
    const MetalDispatchSize grid{.width = U64(gw.padded_n / 16u) * 32u * sk,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = 32u * sk};
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    auto &d = e.emit("mxfp4_nax", fn, grid, threads,
                     {x_f16, row_scale, gw.codes, gw.scales, uw.codes, uw.scales,
                      y_out});
    e.dynamic(d, 7, words({gw.n, gw.k, mp, t}), tokens_patch);
    return;
  }
  const U32 y_tiles = nax_y_tiles(mp, rf);
  std::string fn =
      std::string(out_f16 ? "omlx_m5_mxfp4_nax_swiglu_f16_n16_sk"
                          : "omlx_m5_mxfp4_nax_swiglu_n16_sk") +
      std::to_string(sk);
  if (rf > 1u) {
    fn += "_rf" + std::to_string(rf);
  }
  const MetalDispatchSize grid{.width = U64(gw.padded_n / 16u) * 32u * sk,
                               .height = y_tiles};
  const MetalDispatchSize threads{.width = 32u * sk};
  e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
  auto &d = e.emit("mxfp4_nax", fn, grid, threads,
                   {x_f16, row_scale, gw.codes, gw.scales, uw.codes, uw.scales,
                    y_out});
  e.dynamic(d, 7, words({gw.n, gw.k, mp, t}), tokens_patch);
}

inline void linear_mxfp4_nax_swiglu(
    Encoder &e, const TiledMxfp4Weight &gw, const TiledMxfp4Weight &uw,
    Ref x, Ref f16_scratch, Ref f16_row_scale, Ref y_out, U32 gpu_cores = 64u,
    std::optional<U32> explicit_t = {}) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mxfp4_nax_mp(t);
  const U32 sel_t = nax_effective_select_t(e, explicit_t);
  const auto sel_exp = nax_effective_select_explicit_t(e, explicit_t);
  const auto wide =
      mxfp4_nax_select_wide(gw, sel_t, sel_exp, gpu_cores, true);
  if (wide != NaxWidePrefillTile::none) {
    const bool is_64 = (wide == NaxWidePrefillTile::bm64_bn64);
    const U32 tile_m = is_64 ? 64u : 32u;
    const U32 tile_n = is_64 ? 32u : 16u;
    const U32 tptg = is_64 ? 128u : 32u;
    const U32 x_tiles = (gw.padded_n + tile_n - 1u) / tile_n;
    const U32 y_tiles = (t + tile_m - 1u) / tile_m;
    const std::string fn = std::string("omlx_m5_mxfp4_nax_swiglu_bf16_") +
                           (is_64 ? "bm64_bn64_sk1" : "bm32_bn32_sk1");
    const MetalDispatchSize grid{.width = U64(x_tiles) * tptg,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = tptg};
    const std::optional<U32> tokens_patch =
        explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    auto &d = e.emit("mxfp4_nax", fn, grid, threads,
                     {x, gw.codes, gw.scales, uw.codes, uw.scales, y_out});
    e.dynamic(d, 6, words({gw.n, gw.k, mp, t}), tokens_patch);
    return;
  }
  ensure_f16_cast(e, x, f16_scratch, f16_row_scale, gw.k, explicit_t);
  linear_mxfp4_nax_swiglu_direct_f16(e, gw, uw, f16_scratch, f16_row_scale,
                                     y_out, false, gpu_cores, explicit_t);
}

inline U32 mxfp4_w4a4_mp(U32 t) { return ((t + 15u) / 16u) * 16u; }

inline std::string mxfp4_w4a4_group(const Encoder &e,
                                    const TiledMxfp4Weight &tw,
                                    std::optional<U32> explicit_t = {},
                                    U32 gpu_cores = 64u) {
  const U32 sel_t = nax_effective_select_t(e, explicit_t);
  const auto sel_exp = nax_effective_select_explicit_t(e, explicit_t);
  const auto wide =
      nax_select_wide_prefill_tile(tw.padded_n, sel_t, sel_exp, gpu_cores);
  if (wide != NaxWidePrefillTile::none) {
    const char *tag =
        (wide == NaxWidePrefillTile::bm64_bn64) ? "bm64_bn64" : "bm32_bn32";
    return std::string("mxfp4.w4a4.nax.") + tag + ".bns" +
           std::to_string(tw.bn) + ".sk1.v1";
  }
  return "mxfp4.w4a4.nax.bm16_bk128.n" + std::to_string(tw.bn) + ".v1";
}

inline U32 mxfp4_w4a4_rf(U32 t) {
  U32 max_rf = 1u;
  if (const char *env = std::getenv("OMLX_M5_NAX_MAX_RF")) {
    const U32 v = U32(std::strtoul(env, nullptr, 10));
    if (v >= 4u) max_rf = 4u;
    else if (v >= 2u) max_rf = 2u;
  }
  if (t <= 16u || max_rf == 1u) return 1u;
  if (t <= 32u || max_rf == 2u) return 2u;
  return 4u;
}

inline void ensure_a4_quant(Encoder &e, Ref x, Ref a4_codes, Ref a4_scales,
                            U32 k, std::optional<U32> explicit_t = {}) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mxfp4_w4a4_mp(t);
  const bool dyn_rows = !explicit_t.has_value() && t != 1u;
  const Encoder::F16CastKey key{a4_codes.buffer, a4_codes.offset,
                                a4_scales.buffer, a4_scales.offset,
                                x.buffer, x.offset, k, t, dyn_rows};
  if (e.has_f16_cast(key)) {
    return;
  }
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  auto &d = e.emit("mxfp4_nax", "omlx_m5_bf16_to_mxfp4_a4_bm16_bk128",
                   {.width = U64(k / 32u) * 32u, .height = mp}, {.width = 32u},
                   {x, a4_codes, a4_scales});
  e.dynamic(d, 3, words({k, mp, 0u, t}), tokens_patch);
  e.set_f16_cast(key);
}

inline void linear_mxfp4_w4a4_nax(Encoder &e, const TiledMxfp4Weight &tw, Ref x,
                                  Ref y, Ref a4_codes, Ref a4_scales,
                                  std::optional<U32> explicit_t = {},
                                  U32 gpu_cores = 64u) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mxfp4_w4a4_mp(t);
  const U32 sel_t = nax_effective_select_t(e, explicit_t);
  const auto sel_exp = nax_effective_select_explicit_t(e, explicit_t);
  ensure_a4_quant(e, x, a4_codes, a4_scales, tw.k, explicit_t);
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);

  const auto wide =
      nax_select_wide_prefill_tile(tw.padded_n, sel_t, sel_exp, gpu_cores);
  if (wide != NaxWidePrefillTile::none) {
    const bool is_64 = (wide == NaxWidePrefillTile::bm64_bn64);
    const U32 tile_mn = is_64 ? 64u : 32u;
    const U32 tptg = is_64 ? 128u : 32u;
    const U32 x_tiles = (tw.padded_n + tile_mn - 1u) / tile_mn;
    const U32 y_tiles = (mp + tile_mn - 1u) / tile_mn;
    const std::string fn = std::string("omlx_m5_mxfp4_w4a4_nax_") +
                           (is_64 ? "bm64_bn64" : "bm32_bn32") + "_bns" +
                           std::to_string(tw.bn) + "_sk1";
    const MetalDispatchSize grid{.width = U64(x_tiles) * tptg,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = tptg};
    auto &d = e.emit("mxfp4_nax", fn, grid, threads,
                     {a4_codes, a4_scales, tw.codes, tw.scales, y});
    e.dynamic(d, 5, words({tw.n, tw.k, mp, t}), tokens_patch);
    return;
  }

  const U32 rf = mxfp4_w4a4_rf(sel_t);
  U32 sk = (tw.n >= 65536u) ? 2u : 4u;
  const U32 blocks = tw.k / 128u;
  while (sk > 1u && sk > blocks) {
    sk /= 2u;
  }
  std::string fn = "omlx_m5_mxfp4_w4a4_nax_n" + std::to_string(tw.bn) + "_sk" +
                   std::to_string(sk);
  if (rf > 1u) {
    fn += "_rf" + std::to_string(rf);
  }
  const U32 y_tiles = ((mp / 16u) + rf - 1u) / rf;
  const MetalDispatchSize grid{.width = U64(tw.padded_n / tw.bn) * 32u * sk,
                               .height = y_tiles};
  const MetalDispatchSize threads{.width = 32u * sk};
  auto &d = e.emit("mxfp4_nax", fn, grid, threads,
                   {a4_codes, a4_scales, tw.codes, tw.scales, y});
  e.dynamic(d, 5, words({tw.n, tw.k, mp, t}), tokens_patch);
}

inline void linear_mxfp4_w4a4_nax_swiglu(Encoder &e, const TiledMxfp4Weight &gw,
                                         const TiledMxfp4Weight &uw, Ref x,
                                         Ref y, Ref a4_codes, Ref a4_scales,
                                         U32 gpu_cores = 64u,
                                         std::optional<U32> explicit_t = {}) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = mxfp4_w4a4_mp(t);
  const U32 sel_t = nax_effective_select_t(e, explicit_t);
  const auto sel_exp = nax_effective_select_explicit_t(e, explicit_t);
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  ensure_a4_quant(e, x, a4_codes, a4_scales, gw.k, explicit_t);

  const auto wide =
      nax_select_wide_prefill_tile(gw.padded_n, sel_t, sel_exp, gpu_cores);
  if (wide != NaxWidePrefillTile::none) {
    const bool is_64 = (wide == NaxWidePrefillTile::bm64_bn64);
    const U32 tile_mn = is_64 ? 64u : 32u;
    const U32 tptg = is_64 ? 128u : 32u;
    const U32 x_tiles = (gw.padded_n + tile_mn - 1u) / tile_mn;
    const U32 y_tiles = (mp + tile_mn - 1u) / tile_mn;
    const std::string fn =
        std::string("omlx_m5_mxfp4_w4a4_nax_swiglu_") +
        (is_64 ? "bm64_bn64_sk1" : "bm32_bn32_sk1");
    const MetalDispatchSize grid{.width = U64(x_tiles) * tptg,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = tptg};
    auto &d = e.emit("mxfp4_nax", fn, grid, threads,
                     {a4_codes, a4_scales, gw.codes, gw.scales, uw.codes,
                      uw.scales, y});
    e.dynamic(d, 7, words({gw.n, gw.k, mp, t}), tokens_patch);
    return;
  }

  const U32 rf = mxfp4_w4a4_rf(sel_t);
  U32 sk = 4u;
  const U32 blocks = gw.k / 128u;
  while (sk > 2u && sk > blocks) {
    sk /= 2u;
  }
  std::string fn =
      "omlx_m5_mxfp4_w4a4_nax_swiglu_n16_sk" + std::to_string(sk);
  if (rf > 1u) {
    fn += "_rf" + std::to_string(rf);
  }
  const U32 y_tiles = ((mp / 16u) + rf - 1u) / rf;
  const MetalDispatchSize grid{.width = U64(gw.padded_n / 16u) * 32u * sk,
                               .height = y_tiles};
  const MetalDispatchSize threads{.width = 32u * sk};
  auto &d = e.emit("mxfp4_nax", fn, grid, threads,
                   {a4_codes, a4_scales, gw.codes, gw.scales, uw.codes,
                    uw.scales, y});
  e.dynamic(d, 7, words({gw.n, gw.k, mp, t}), tokens_patch);
}

} // namespace omlx::m5::detail::forward
