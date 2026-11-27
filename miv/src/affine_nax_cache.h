#pragma once

#include "nax_common.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace omlx::m5::detail::forward {

struct TiledAffineWeight {
  Ref codes;
  Ref high2; // valid when bits == 3 ("nax_high1"), 5, or 6 ("nax_high2")
  Ref sb;
  U32 n = 0;
  U32 k = 0;
  U32 bn = 16;
  U32 padded_n = 0;
  U32 split_k = 1;
  U32 bits = 4;       // 3, 4, 5, 6, or 8
  U32 group_size = 64; // 64 or 128
  Ref sb_tiled = {};
  Ref high1 = {};
};

inline bool affine_nax_runtime_eligible(const MetalContext &ctx,
                                        const std::filesystem::path &libs) {
  return nax_apple10_eligible(ctx, libs, "omlx_m5_affine_nax.metallib",
                              "OMLX_M5_DISABLE_AFFINE_NAX");
}

inline U32 affine_nax_bits(std::string_view format) {
  if (format == "oq_affine_q3_g64_bf16_v1" ||
      format == "oq_affine_q3_g128_bf16_v1") {
    return 3u;
  }
  if (format == "oq_affine_q4_g64_bf16_v1" ||
      format == "oq_affine_q4_g128_bf16_v1") {
    return 4u;
  }
  if (format == "oq_affine_q5_g64_bf16_v1" ||
      format == "oq_affine_q5_g128_bf16_v1") {
    return 5u;
  }
  if (format == "oq_affine_q6_g64_bf16_v1" ||
      format == "oq_affine_q6_g128_bf16_v1") {
    return 6u;
  }
  if (format == "oq_affine_q8_g64_bf16_v1" ||
      format == "oq_affine_q8_g128_bf16_v1") {
    return 8u;
  }
  return 0u;
}

inline U32 affine_nax_group_size(std::string_view format) {
  if (format.find("_g64_") != std::string_view::npos) {
    return 64u;
  }
  if (format.find("_g128_") != std::string_view::npos) {
    return 128u;
  }
  return 0u;
}

inline std::string affine_nax_mode_tag(U32 bits) {
  if (bits == 3u) return "q3";
  if (bits == 4u) return "q4";
  if (bits == 8u) return "q8";
  return "q56";
}

inline bool affine_nax_has_high_plane(U32 bits) {
  return bits == 3u || bits == 5u || bits == 6u;
}

inline U64 affine_nax_code_bytes(U32 padded_n, U32 k, U32 bits) {
  if (bits == 3u) return mul(padded_n, k / 4u);
  if (bits == 8u) return mul(padded_n, k);
  return mul(padded_n, k / 2u);
}

inline U64 affine_nax_high_bytes(U32 padded_n, U32 k, U32 bits) {
  if (bits == 3u) return mul(padded_n, k / 8u);
  if (bits == 5u || bits == 6u) return mul(padded_n, k / 4u);
  return 0ULL;
}

inline bool is_oq_nax_tiled_v1_layout(std::string_view layout) noexcept {
  return layout == "oq_nax_tiled_g64_v1" || layout == "oq_nax_tiled_g128_v1";
}

inline bool is_oq_nax_lane_v2_layout(std::string_view layout) noexcept {
  return layout == "oq_nax_lane_g64_v2" || layout == "oq_nax_lane_g128_v2";
}

inline bool is_oq_nax_pretiled_layout(std::string_view layout) noexcept {
  return is_oq_nax_tiled_v1_layout(layout) ||
         is_oq_nax_lane_v2_layout(layout);
}

inline bool affine_nax_weight_eligible(const MaterializedWeightView &w) {
  const U32 bits = affine_nax_bits(w.format);
  const U32 gs = affine_nax_group_size(w.format);
  return bits != 0u && gs != 0u &&
         (w.layout == "oq_rowsplit_k128_v1" ||
          is_oq_nax_pretiled_layout(w.layout)) &&
         w.parent_shape.size() == 2 && w.row_count > 0 &&
         w.in_features >= 512 && (w.in_features % gs) == 0 &&
         (w.in_features % 64u) == 0 && w.padded_k >= w.in_features &&
         (w.padded_k % 128u) == 0;
}

inline U32 affine_nax_bn(U32 n) { return n >= 65536u ? 32u : 16u; }

inline U32 normalize_affine_nax_split_k(U32 target_slices) {
  if (target_slices >= 8u) return 8u;
  if (target_slices >= 4u) return 4u;
  if (target_slices >= 2u) return 2u;
  return 1u;
}

inline U32 affine_nax_split_k(U32 n, U32 k, U32 target_slices = 8u) {
  const U32 groups = k / 64u;
  const U32 base_sk = normalize_affine_nax_split_k(target_slices);
  U32 sk = (n >= 65536u) ? std::min<U32>(2u, base_sk) : base_sk;
  if (n < 65536u) {
    if (const char *env = std::getenv("OMLX_M5_AFFINE_NAX_SK")) {
      const U32 v = U32(std::strtoul(env, nullptr, 10));
      if (v == 1u || v == 2u || v == 4u || v == 8u) {
        sk = v;
      }
    }
  }
  while (sk > 1u && sk > groups) {
    sk /= 2u;
  }
  return sk;
}

inline U32 affine_nax_mp(U32 t) { return ((t + 7u) / 8u) * 8u; }

class AffineNaxCache {
public:
  static constexpr bool kIncrementalOwnerCache = true;

  std::recursive_mutex &build_mutex() const noexcept { return build_mutex_; }

  void set_target_split_k(U32 target_slices) {
    std::lock_guard<std::recursive_mutex> lk(build_mutex_);
    target_split_k_ = normalize_affine_nax_split_k(target_slices);
  }

  void register_weight(const MaterializedWeightView &w, Ref zero) {
    std::lock_guard<std::recursive_mutex> lk(build_mutex_);
    if (!affine_nax_weight_eligible(w)) {
      return;
    }
    if (has_entry(w.object_id) || !seen_.insert(w.object_id).second) {
      return;
    }
    if (is_oq_nax_lane_v2_layout(w.layout)) {
      const U32 n = U32(w.row_count);
      const U32 k = U32(w.in_features);
      const U32 bn = affine_nax_bn(n);
      const U32 padded_n = ((n + bn - 1u) / bn) * bn;
      const U32 bits = affine_nax_bits(w.format);
      const U32 gs = affine_nax_group_size(w.format);
      Ref high = bits == 3u                     ? plane(w, "nax_high1")
                 : (bits == 5u || bits == 6u)   ? plane(w, "nax_high2")
                                                : zero;
      const Ref src_sb = plane(w, "nax_sb");
      Ref pre_sb_tiled{};
      for (const auto &pl : w.planes) {
        if (pl.kind == "nax_sb_tiled" && pl.buffer != nullptr && pl.bytes > 0) {
          pre_sb_tiled = Ref{pl.buffer, pl.offset, pl.bytes};
          break;
        }
      }
      pending_pretiled_.push_back(
          {w.object_id,
           TiledAffineWeight{plane(w, "nax_codes"), high, src_sb, n, k, bn,
                             padded_n,
                             affine_nax_split_k(n, k, target_split_k_), bits,
                             gs, pre_sb_tiled, {}}});
      if (pre_sb_tiled.buffer == nullptr) {
        pending_sb_transposes_.push_back(
            {w.object_id, src_sb, padded_n, k / 64u, bn});
      }
      if (bits == 5u && high.buffer != nullptr) {
        pending_h1_compacts_.push_back(
            {w.object_id, high, mul(padded_n, k / 8u)});
      }
      return;
    }
    if (is_oq_nax_tiled_v1_layout(w.layout)) {
      const U32 n = U32(w.row_count);
      const U32 k = U32(w.in_features);
      const U32 bn = affine_nax_bn(n);
      const U32 padded_n = ((n + bn - 1u) / bn) * bn;
      const U32 bits = affine_nax_bits(w.format);
      const U32 gs = affine_nax_group_size(w.format);
      Ref high = bits == 3u                     ? plane(w, "nax_high1")
                 : (bits == 5u || bits == 6u)   ? plane(w, "nax_high2")
                                                : zero;
      const Ref src_sb = plane(w, "nax_sb");
      Ref pre_sb_tiled{};
      for (const auto &pl : w.planes) {
        if (pl.kind == "nax_sb_tiled" && pl.buffer != nullptr && pl.bytes > 0) {
          pre_sb_tiled = Ref{pl.buffer, pl.offset, pl.bytes};
          break;
        }
      }
      pending_v1_repacks_.push_back(
          {w.object_id, plane(w, "nax_codes"), high, src_sb, pre_sb_tiled, n, k,
           bn, padded_n, affine_nax_split_k(n, k, target_split_k_), bits, gs});
      if (pre_sb_tiled.buffer == nullptr) {
        pending_sb_transposes_.push_back(
            {w.object_id, src_sb, padded_n, k / 64u, bn});
      }
      return;
    }
    pending_.push_back(w);
  }

  // Optional zero-ULP column concatenation helper for homogeneous sibling
  // projections sharing input x (where each non-final slice has row_count % 16 == 0).
  bool register_concat(U32 concat_id,
                       std::span<const MaterializedWeightView> parts) {
    std::lock_guard<std::recursive_mutex> lk(build_mutex_);
    if (parts.size() < 2 || has_entry(concat_id) || seen_.contains(concat_id)) {
      return false;
    }
    const U32 k = U32(parts.front().in_features);
    const U32 bits = affine_nax_bits(parts.front().format);
    U32 gs = affine_nax_group_size(parts.front().format);
    U32 total_n = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
      const auto &w = parts[i];
      if (!affine_nax_weight_eligible(w) || U32(w.in_features) != k ||
          affine_nax_bits(w.format) != bits) {
        return false;
      }
      if (i + 1 < parts.size() && (w.row_count % 16u) != 0u) {
        return false;
      }
      gs = std::min(gs, affine_nax_group_size(w.format));
      total_n += U32(w.row_count);
    }
    if (total_n >= 65536u) {
      return false;
    }
    seen_.insert(concat_id);
    pending_concats_.push_back(
        {concat_id, {parts.begin(), parts.end()}, total_n, k, bits, gs});
    return true;
  }

  void build(const MetalContext &ctx, const std::filesystem::path &libs,
             Ref zero) {
    std::lock_guard<std::recursive_mutex> lk(build_mutex_);
    struct PendingResetGuard {
      AffineNaxCache *self;
      ~PendingResetGuard() {
        self->pending_.clear();
        self->pending_pretiled_.clear();
        self->pending_v1_repacks_.clear();
        self->pending_concats_.clear();
        self->pending_sb_transposes_.clear();
        self->pending_h1_compacts_.clear();
        self->seen_.clear();
      }
    } reset_guard{this};

    const bool has_row_exact =
        std::filesystem::exists(libs / "omlx_m5_row_exact.metallib");
    if (!has_row_exact) {
      pending_sb_transposes_.clear();
      pending_h1_compacts_.clear();
    }
    if (pending_.empty() && pending_v1_repacks_.empty() &&
        pending_concats_.empty() && pending_sb_transposes_.empty() &&
        pending_h1_compacts_.empty()) {
      if (!pending_pretiled_.empty()) {
        std::lock_guard<std::mutex> elk(entries_mutex_);
        for (auto &pp : pending_pretiled_) {
          entries_.try_emplace(
              pp.object_id,
              std::make_unique<TiledAffineWeight>(std::move(pp.weight)));
        }
      }
      return;
    }
    NaxSegmentArena arena;

    struct V1RepackPlan {
      U32 object_id = 0;
      Ref src_codes{};
      Ref src_high{};
      Ref src_sb{};
      Ref pre_sb_tiled{};
      U32 n = 0;
      U32 k = 0;
      U32 bn = 16;
      U32 padded_n = 0;
      U32 sk = 1;
      U32 bits = 4;
      U32 gs = 64;
      NaxSegmentArena::Placement codes;
      NaxSegmentArena::Placement high;
      NaxSegmentArena::Placement high1;
    };
    std::vector<V1RepackPlan> v1_plans;
    v1_plans.reserve(pending_v1_repacks_.size());
    for (const auto &pv : pending_v1_repacks_) {
      const U64 code_bytes = affine_nax_code_bytes(pv.padded_n, pv.k, pv.bits);
      NaxSegmentArena::Placement high_p{};
      if (affine_nax_has_high_plane(pv.bits)) {
        high_p = arena.plan_alloc(affine_nax_high_bytes(pv.padded_n, pv.k, pv.bits),
                                  "Affine NAX v1->lane_v2 high");
      }
      NaxSegmentArena::Placement high1_p{};
      if (has_row_exact && pv.bits == 5u) {
        high1_p = arena.plan_alloc(mul(pv.padded_n, pv.k / 8u),
                                   "Affine NAX v1->lane_v2 q5_high1");
      }
      v1_plans.push_back(
          {pv.object_id, pv.src_codes, pv.src_high, pv.src_sb, pv.pre_sb_tiled,
           pv.n, pv.k, pv.bn, pv.padded_n, pv.sk, pv.bits, pv.gs,
           arena.plan_alloc(code_bytes, "Affine NAX v1->lane_v2 codes"),
           high_p, high1_p});
    }

    struct PendingPlan {
      MaterializedWeightView weight;
      U32 n = 0;
      U32 k = 0;
      U32 bn = 16;
      U32 padded_n = 0;
      U32 sk = 1;
      U32 bits = 4;
      U32 gs = 64;
      NaxSegmentArena::Placement codes;
      NaxSegmentArena::Placement high;
      NaxSegmentArena::Placement sb;
      NaxSegmentArena::Placement sb_tiled;
      NaxSegmentArena::Placement high1;
    };
    std::vector<PendingPlan> plans;
    plans.reserve(pending_.size());
    for (const auto &w : pending_) {
      const U32 n = U32(w.row_count);
      const U32 k = U32(w.in_features);
      const U32 bn = affine_nax_bn(n);
      const U32 padded_n = ((n + bn - 1u) / bn) * bn;
      const U32 bits = affine_nax_bits(w.format);
      const U32 gs = affine_nax_group_size(w.format);
      const U64 code_bytes = affine_nax_code_bytes(padded_n, k, bits);
      const U64 sb_bytes = mul(mul(padded_n, k / 64u), 4u);
      NaxSegmentArena::Placement high_p{};
      if (affine_nax_has_high_plane(bits)) {
        high_p = arena.plan_alloc(affine_nax_high_bytes(padded_n, k, bits),
                                  "Affine NAX tiled high plane");
      }
      NaxSegmentArena::Placement sb_tiled_p{};
      NaxSegmentArena::Placement high1_p{};
      if (has_row_exact) {
        sb_tiled_p = arena.plan_alloc(sb_bytes, "Affine NAX tiled sb_tiled");
        if (bits == 5u) {
          high1_p = arena.plan_alloc(mul(padded_n, k / 8u),
                                     "Affine NAX tiled q5_high1");
        }
      }
      plans.push_back(
          {w, n, k, bn, padded_n, affine_nax_split_k(n, k, target_split_k_),
           bits, gs,
           arena.plan_alloc(code_bytes, "Affine NAX tiled codes"), high_p,
           arena.plan_alloc(sb_bytes, "Affine NAX tiled sb"), sb_tiled_p,
           high1_p});
    }

    struct SbTransposePlan {
      U32 object_id = 0;
      Ref src_sb{};
      U32 padded_n = 0;
      U32 groups = 0;
      U32 bn = 16;
      NaxSegmentArena::Placement sb_tiled;
    };
    std::vector<SbTransposePlan> sb_plans;
    struct H1CompactPlan {
      U32 object_id = 0;
      Ref src_high2{};
      U64 h1_bytes = 0;
      NaxSegmentArena::Placement high1;
    };
    std::vector<H1CompactPlan> h1_plans;
    if (has_row_exact) {
      sb_plans.reserve(pending_sb_transposes_.size());
      for (const auto &pt : pending_sb_transposes_) {
        const U64 sb_bytes = mul(mul(pt.padded_n, pt.groups), 4u);
        sb_plans.push_back(
            {pt.object_id, pt.src_sb, pt.padded_n, pt.groups, pt.bn,
             arena.plan_alloc(sb_bytes, "Affine NAX pretiled sb_tiled")});
      }
      h1_plans.reserve(pending_h1_compacts_.size());
      for (const auto &ph : pending_h1_compacts_) {
        if (ph.h1_bytes > 0ULL && (ph.h1_bytes % 4ULL) == 0ULL &&
            ph.src_high2.bytes >= ph.h1_bytes * 2ULL) {
          h1_plans.push_back(
              {ph.object_id, ph.src_high2, ph.h1_bytes,
               arena.plan_alloc(ph.h1_bytes, "Affine NAX pretiled q5_high1")});
        }
      }
    }

    struct ConcatPlan {
      U32 concat_id = 0;
      std::vector<MaterializedWeightView> parts;
      U32 total_n = 0;
      U32 k = 0;
      U32 padded_n = 0;
      U32 sk = 1;
      U32 bits = 4;
      U32 gs = 64;
      NaxSegmentArena::Placement codes;
      NaxSegmentArena::Placement high;
      NaxSegmentArena::Placement sb;
      NaxSegmentArena::Placement sb_tiled;
      NaxSegmentArena::Placement high1;
    };
    std::vector<ConcatPlan> cplans;
    cplans.reserve(pending_concats_.size());
    for (auto &cp : pending_concats_) {
      const U32 padded_n = ((cp.total_n + 15u) / 16u) * 16u;
      const U64 code_bytes = affine_nax_code_bytes(padded_n, cp.k, cp.bits);
      const U64 sb_bytes = mul(mul(padded_n, cp.k / 64u), 4u);
      NaxSegmentArena::Placement high_p{};
      if (affine_nax_has_high_plane(cp.bits)) {
        high_p = arena.plan_alloc(affine_nax_high_bytes(padded_n, cp.k, cp.bits),
                                  "Affine NAX concat high plane");
      }
      NaxSegmentArena::Placement sb_tiled_p{};
      NaxSegmentArena::Placement high1_p{};
      if (has_row_exact) {
        sb_tiled_p = arena.plan_alloc(sb_bytes, "Affine NAX concat sb_tiled");
        if (cp.bits == 5u) {
          high1_p = arena.plan_alloc(mul(padded_n, cp.k / 8u),
                                     "Affine NAX concat q5_high1");
        }
      }
      cplans.push_back(
          {cp.concat_id, std::move(cp.parts), cp.total_n, cp.k, padded_n,
           affine_nax_split_k(cp.total_n, cp.k, target_split_k_), cp.bits,
           cp.gs,
           arena.plan_alloc(code_bytes, "Affine NAX concat codes"), high_p,
           arena.plan_alloc(sb_bytes, "Affine NAX concat sb"), sb_tiled_p,
           high1_p});
    }

    const std::size_t base_seg = segments_.size();
    arena.materialize(ctx, segments_);

    auto src_base_plane = [&](const MaterializedWeightView &w, U32 bits) -> Ref {
      if (bits == 3u) return plane(w, "low_2bit");
      if (bits == 8u) return plane(w, "code");
      return plane(w, "base_code");
    };
    auto src_high_plane = [&](const MaterializedWeightView &w, U32 bits) -> Ref {
      if (bits == 3u || bits == 5u) return plane(w, "high_bit");
      if (bits == 6u) return plane(w, "high_2bit");
      return zero;
    };

    std::unordered_map<U32, std::unique_ptr<TiledAffineWeight>> staged;
    for (auto &pp : pending_pretiled_) {
      staged.emplace(pp.object_id,
                     std::make_unique<TiledAffineWeight>(std::move(pp.weight)));
    }

    static std::atomic<U64> repack_id{0};
    Encoder enc(ctx, libs, 1,
                "affine.nax.repack." + std::to_string(++repack_id));
    if (!plans.empty() || !v1_plans.empty() || !cplans.empty()) {
      enc.begin_stage("repack_affine_g64");
      for (const auto &vp : v1_plans) {
        const Ref dst_codes =
            NaxSegmentArena::resolve(segments_, vp.codes, base_seg);
        const Ref dst_high =
            affine_nax_has_high_plane(vp.bits)
                ? NaxSegmentArena::resolve(segments_, vp.high, base_seg)
                : zero;
        const Ref dst_high1 =
            (has_row_exact && vp.bits == 5u)
                ? NaxSegmentArena::resolve(segments_, vp.high1, base_seg)
                : Ref{};
        const U32 groups = vp.k / 64u;
        const U64 row_groups = mul(U64(vp.padded_n), U64(groups));
        auto &d = enc.emit(
            "affine_nax", "omlx_m5_repack_oq_v1_to_lane_v2",
            {.width = ((row_groups + 255ULL) / 256ULL) * 256ULL},
            {.width = 256u},
            {vp.src_codes, affine_nax_has_high_plane(vp.bits) ? vp.src_high : zero,
             dst_codes, dst_high});
        enc.dynamic(d, 4, words({U32(row_groups), vp.bits}));
        staged.emplace(
            vp.object_id,
            std::make_unique<TiledAffineWeight>(TiledAffineWeight{
                dst_codes, dst_high, vp.src_sb, vp.n, vp.k, vp.bn, vp.padded_n,
                vp.sk, vp.bits, vp.gs, vp.pre_sb_tiled, dst_high1}));
      }
      for (const auto &pp : plans) {
        const Ref dst_codes =
            NaxSegmentArena::resolve(segments_, pp.codes, base_seg);
        const Ref dst_sb = NaxSegmentArena::resolve(segments_, pp.sb, base_seg);
        const Ref dst_high =
            affine_nax_has_high_plane(pp.bits)
                ? NaxSegmentArena::resolve(segments_, pp.high, base_seg)
                : zero;
        const Ref dst_sb_tiled =
            has_row_exact
                ? NaxSegmentArena::resolve(segments_, pp.sb_tiled, base_seg)
                : Ref{};
        const Ref dst_high1 =
            (has_row_exact && pp.bits == 5u)
                ? NaxSegmentArena::resolve(segments_, pp.high1, base_seg)
                : Ref{};
        const U32 groups = pp.k / 64u;
        const std::string fn = pp.bn == 32u ? "omlx_m5_repack_affine_g64_bn32"
                                            : "omlx_m5_repack_affine_g64_bn16";
        auto &d = enc.emit("affine_nax", fn,
                           {.width = groups, .height = pp.padded_n},
                           {.width = std::min(groups, 16u), .height = 16u},
                           {src_base_plane(pp.weight, pp.bits),
                            src_high_plane(pp.weight, pp.bits),
                            plane(pp.weight, "scale"), plane(pp.weight, "bias"),
                            dst_codes, dst_high, dst_sb});
        enc.dynamic(d, 7,
                    words({pp.n, pp.k, U32(pp.weight.padded_k), pp.padded_n,
                           pp.bits, pp.gs, 0u}));
        staged.emplace(
            pp.weight.object_id,
            std::make_unique<TiledAffineWeight>(TiledAffineWeight{
                dst_codes, dst_high, dst_sb, pp.n, pp.k, pp.bn, pp.padded_n,
                pp.sk, pp.bits, pp.gs, dst_sb_tiled, dst_high1}));
      }
      for (const auto &cp : cplans) {
        const Ref dst_codes =
            NaxSegmentArena::resolve(segments_, cp.codes, base_seg);
        const Ref dst_sb = NaxSegmentArena::resolve(segments_, cp.sb, base_seg);
        const Ref dst_high =
            affine_nax_has_high_plane(cp.bits)
                ? NaxSegmentArena::resolve(segments_, cp.high, base_seg)
                : zero;
        const U32 groups = cp.k / 64u;
        U32 row_off = 0;
        for (const auto &w : cp.parts) {
          const U32 wn = U32(w.row_count);
          const U32 wpadded_n = ((wn + 15u) / 16u) * 16u;
          const U32 wgs = affine_nax_group_size(w.format);
          if (is_oq_nax_lane_v2_layout(w.layout)) {
            const U64 c_off = affine_nax_code_bytes(row_off, cp.k, cp.bits);
            const U64 c_len = affine_nax_code_bytes(wpadded_n, cp.k, cp.bits);
            const U32 c_words = U32(c_len / 4u);
            auto &dc = enc.emit(
                "affine_nax", "omlx_m5_copy_affine_u32_words",
                {.width = c_words}, {.width = 256u},
                {plane(w, "nax_codes"),
                 Ref{dst_codes.buffer, dst_codes.offset + c_off, c_len}});
            enc.dynamic(dc, 2, words({c_words}));
            if (affine_nax_has_high_plane(cp.bits)) {
              const U64 h_off = affine_nax_high_bytes(row_off, cp.k, cp.bits);
              const U64 h_len = affine_nax_high_bytes(wpadded_n, cp.k, cp.bits);
              const U32 h_words = U32(h_len / 4u);
              const Ref wh = cp.bits == 3u ? plane(w, "nax_high1")
                                           : plane(w, "nax_high2");
              auto &dh = enc.emit(
                  "affine_nax", "omlx_m5_copy_affine_u32_words",
                  {.width = h_words}, {.width = 256u},
                  {wh, Ref{dst_high.buffer, dst_high.offset + h_off, h_len}});
              enc.dynamic(dh, 2, words({h_words}));
            }
            auto &dsb = enc.emit(
                "affine_nax", "omlx_m5_copy_affine_sb_slice",
                {.width = groups, .height = wpadded_n},
                {.width = std::min(groups, 16u), .height = 16u},
                {plane(w, "nax_sb"), dst_sb});
            enc.dynamic(dsb, 2,
                        words({groups, wpadded_n, wn, cp.padded_n, row_off}));
          } else if (is_oq_nax_tiled_v1_layout(w.layout)) {
            const U64 c_off = affine_nax_code_bytes(row_off, cp.k, cp.bits);
            const U64 c_len = affine_nax_code_bytes(wpadded_n, cp.k, cp.bits);
            const U64 h_off = affine_nax_high_bytes(row_off, cp.k, cp.bits);
            const U64 h_len = affine_nax_high_bytes(wpadded_n, cp.k, cp.bits);
            const Ref wh = affine_nax_has_high_plane(cp.bits)
                               ? (cp.bits == 3u ? plane(w, "nax_high1")
                                                : plane(w, "nax_high2"))
                               : zero;
            const Ref dst_h_slice =
                affine_nax_has_high_plane(cp.bits)
                    ? Ref{dst_high.buffer, dst_high.offset + h_off, h_len}
                    : zero;
            const U64 row_groups = mul(U64(wpadded_n), U64(groups));
            auto &dr = enc.emit(
                "affine_nax", "omlx_m5_repack_oq_v1_to_lane_v2",
                {.width = ((row_groups + 255ULL) / 256ULL) * 256ULL},
                {.width = 256u},
                {plane(w, "nax_codes"), wh,
                 Ref{dst_codes.buffer, dst_codes.offset + c_off, c_len},
                 dst_h_slice});
            enc.dynamic(dr, 4, words({U32(row_groups), cp.bits}));
            auto &dsb = enc.emit(
                "affine_nax", "omlx_m5_copy_affine_sb_slice",
                {.width = groups, .height = wpadded_n},
                {.width = std::min(groups, 16u), .height = 16u},
                {plane(w, "nax_sb"), dst_sb});
            enc.dynamic(dsb, 2,
                        words({groups, wpadded_n, wn, cp.padded_n, row_off}));
          } else {
            auto &d = enc.emit(
                "affine_nax", "omlx_m5_repack_affine_g64_bn16",
                {.width = groups, .height = wpadded_n},
                {.width = std::min(groups, 16u), .height = 16u},
                {src_base_plane(w, cp.bits), src_high_plane(w, cp.bits),
                 plane(w, "scale"), plane(w, "bias"), dst_codes, dst_high,
                 dst_sb});
            enc.dynamic(d, 7,
                        words({wn, cp.k, U32(w.padded_k), cp.padded_n, cp.bits,
                               wgs, row_off}));
          }
          row_off += wn;
        }
        const Ref dst_sb_tiled =
            has_row_exact
                ? NaxSegmentArena::resolve(segments_, cp.sb_tiled, base_seg)
                : Ref{};
        const Ref dst_high1 =
            (has_row_exact && cp.bits == 5u)
                ? NaxSegmentArena::resolve(segments_, cp.high1, base_seg)
                : Ref{};
        staged.emplace(
            cp.concat_id,
            std::make_unique<TiledAffineWeight>(TiledAffineWeight{
                dst_codes, dst_high, dst_sb, cp.total_n, cp.k, 16u, cp.padded_n,
                cp.sk, cp.bits, cp.gs, dst_sb_tiled, dst_high1}));
      }
    }
    if (has_row_exact &&
        (!plans.empty() || !v1_plans.empty() || !cplans.empty() ||
         !sb_plans.empty() || !h1_plans.empty())) {
      enc.begin_stage("transpose_affine_sb_tiled");
      for (const auto &vp : v1_plans) {
        if (vp.bits == 5u && vp.high1.bytes > 0ULL) {
          const Ref src_h2 = NaxSegmentArena::resolve(segments_, vp.high, base_seg);
          const Ref dst_h1 = NaxSegmentArena::resolve(segments_, vp.high1, base_seg);
          const U32 words_out = U32(vp.high1.bytes / 4ULL);
          auto &dc = enc.emit(
              "row_exact", "omlx_m5_compact_q5_high2_to_high1",
              {.width = ((U64(words_out) + 255ULL) / 256ULL) * 256ULL},
              {.width = 256u}, {src_h2, dst_h1});
          enc.dynamic(dc, 2, words({words_out}));
        }
      }
      for (const auto &pp : plans) {
        const Ref src_sb = NaxSegmentArena::resolve(segments_, pp.sb, base_seg);
        const Ref dst_sb_tiled =
            NaxSegmentArena::resolve(segments_, pp.sb_tiled, base_seg);
        const U32 groups = pp.k / 64u;
        auto &dt = enc.emit(
            "row_exact", "omlx_m5_transpose_affine_sb_to_tiled",
            {.width = pp.padded_n, .height = groups},
            {.width = 16u, .height = std::min(groups, 16u)},
            {src_sb, dst_sb_tiled});
        enc.dynamic(dt, 2, words({groups, pp.padded_n, pp.bn, 0u}));
        if (pp.bits == 5u && pp.high1.bytes > 0ULL) {
          const Ref src_h2 = NaxSegmentArena::resolve(segments_, pp.high, base_seg);
          const Ref dst_h1 = NaxSegmentArena::resolve(segments_, pp.high1, base_seg);
          const U32 words_out = U32(pp.high1.bytes / 4ULL);
          auto &dc = enc.emit(
              "row_exact", "omlx_m5_compact_q5_high2_to_high1",
              {.width = ((U64(words_out) + 255ULL) / 256ULL) * 256ULL},
              {.width = 256u}, {src_h2, dst_h1});
          enc.dynamic(dc, 2, words({words_out}));
        }
      }
      for (const auto &cp : cplans) {
        const Ref src_sb = NaxSegmentArena::resolve(segments_, cp.sb, base_seg);
        const Ref dst_sb_tiled =
            NaxSegmentArena::resolve(segments_, cp.sb_tiled, base_seg);
        const U32 groups = cp.k / 64u;
        auto &dt = enc.emit(
            "row_exact", "omlx_m5_transpose_affine_sb_to_tiled",
            {.width = cp.padded_n, .height = groups},
            {.width = 16u, .height = std::min(groups, 16u)},
            {src_sb, dst_sb_tiled});
        enc.dynamic(dt, 2, words({groups, cp.padded_n, 16u, 0u}));
        if (cp.bits == 5u && cp.high1.bytes > 0ULL) {
          const Ref src_h2 = NaxSegmentArena::resolve(segments_, cp.high, base_seg);
          const Ref dst_h1 = NaxSegmentArena::resolve(segments_, cp.high1, base_seg);
          const U32 words_out = U32(cp.high1.bytes / 4ULL);
          auto &dc = enc.emit(
              "row_exact", "omlx_m5_compact_q5_high2_to_high1",
              {.width = ((U64(words_out) + 255ULL) / 256ULL) * 256ULL},
              {.width = 256u}, {src_h2, dst_h1});
          enc.dynamic(dc, 2, words({words_out}));
        }
      }
      for (const auto &sp : sb_plans) {
        const Ref dst_sb_tiled =
            NaxSegmentArena::resolve(segments_, sp.sb_tiled, base_seg);
        auto &dt = enc.emit(
            "row_exact", "omlx_m5_transpose_affine_sb_to_tiled",
            {.width = sp.padded_n, .height = sp.groups},
            {.width = 16u, .height = std::min(sp.groups, 16u)},
            {sp.src_sb, dst_sb_tiled});
        enc.dynamic(dt, 2, words({sp.groups, sp.padded_n, sp.bn, 0u}));
        if (auto it = staged.find(sp.object_id); it != staged.end()) {
          it->second->sb_tiled = dst_sb_tiled;
        }
      }
      for (const auto &hp : h1_plans) {
        const Ref dst_h1 =
            NaxSegmentArena::resolve(segments_, hp.high1, base_seg);
        const U32 words_out = U32(hp.h1_bytes / 4ULL);
        auto &dc = enc.emit(
            "row_exact", "omlx_m5_compact_q5_high2_to_high1",
            {.width = ((U64(words_out) + 255ULL) / 256ULL) * 256ULL},
            {.width = 256u}, {hp.src_high2, dst_h1});
        enc.dynamic(dc, 2, words({words_out}));
        if (auto it = staged.find(hp.object_id); it != staged.end()) {
          it->second->high1 = dst_h1;
        }
      }
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

  const TiledAffineWeight *find(U32 object_id) const {
    std::lock_guard<std::mutex> elk(entries_mutex_);
    const auto it = entries_.find(object_id);
    return it == entries_.end() ? nullptr : it->second.get();
  }

  bool active() const {
    std::lock_guard<std::mutex> elk(entries_mutex_);
    return !entries_.empty();
  }

  static std::shared_ptr<AffineNaxCache>
  acquire(const void *owner_key, const MetalContext &ctx,
          const std::filesystem::path &libs, Ref zero,
          const std::vector<MaterializedWeightView> &candidates,
          U32 target_split_k = 8u) {
    const U32 norm_sk = normalize_affine_nax_split_k(target_split_k);
    return nax_acquire_owner_cache_variant<AffineNaxCache>(
        owner_key, U64(norm_sk), [&](AffineNaxCache &c) {
          c.set_target_split_k(norm_sk);
          for (const auto &w : candidates) {
            c.register_weight(w, zero);
          }
          c.build(ctx, libs, zero);
        });
  }

private:
  bool has_entry(U32 id) const {
    std::lock_guard<std::mutex> elk(entries_mutex_);
    return entries_.contains(id);
  }

  struct PendingPretiled {
    U32 object_id = 0;
    TiledAffineWeight weight{};
  };
  struct PendingV1Repack {
    U32 object_id = 0;
    Ref src_codes{};
    Ref src_high{};
    Ref src_sb{};
    Ref pre_sb_tiled{};
    U32 n = 0;
    U32 k = 0;
    U32 bn = 16;
    U32 padded_n = 0;
    U32 sk = 1;
    U32 bits = 4;
    U32 gs = 64;
  };
  struct PendingSbTranspose {
    U32 object_id = 0;
    Ref src_sb{};
    U32 padded_n = 0;
    U32 groups = 0;
    U32 bn = 16;
  };
  struct PendingH1Compact {
    U32 object_id = 0;
    Ref src_high2{};
    U64 h1_bytes = 0;
  };
  struct PendingConcat {
    U32 concat_id = 0;
    std::vector<MaterializedWeightView> parts;
    U32 total_n = 0;
    U32 k = 0;
    U32 bits = 4;
    U32 gs = 64;
  };
  mutable std::recursive_mutex build_mutex_;
  mutable std::mutex entries_mutex_;
  U32 target_split_k_ = 8u;
  std::unordered_set<U32> seen_;
  std::vector<MaterializedWeightView> pending_;
  std::vector<PendingPretiled> pending_pretiled_;
  std::vector<PendingV1Repack> pending_v1_repacks_;
  std::vector<PendingSbTranspose> pending_sb_transposes_;
  std::vector<PendingH1Compact> pending_h1_compacts_;
  std::vector<PendingConcat> pending_concats_;
  std::vector<std::unique_ptr<MetalBuffer>> segments_;
  std::unordered_map<U32, std::unique_ptr<TiledAffineWeight>> entries_;
};

inline void emit_affine_rmsnorm_xsum(Encoder &e, Ref input, Ref weight,
                                     Ref output_a16, Ref xs, U32 k,
                                     float eps, float weight_offset,
                                     std::optional<U32> explicit_t = {}) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = affine_nax_mp(t);
  const bool dyn_rows = !explicit_t.has_value() && t != 1u;
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() || t == 1u ? std::nullopt : std::optional<U32>(1u);
  auto &d = e.emit("affine_nax", "omlx_m5_affine_rmsnorm_xsum_g64",
                   {.width = 128u, .height = mp}, {.width = 128u},
                   {input, weight, output_a16, xs});
  e.dynamic(d, 4, words({k, t, mp, word(eps), word(weight_offset)}),
            tokens_patch);
  if (t > 1u) {
    const Encoder::GroupSumKey key{xs.buffer, xs.offset, output_a16.buffer,
                                   output_a16.offset, k, 64u, t, dyn_rows, 2u};
    e.set_group_sum(key);
  }
}

inline void emit_affine_add_rmsnorm_xsum(Encoder &e, Ref hidden, Ref delta,
                                         Ref weight, Ref output_a16, Ref xs,
                                         U32 k, float eps, float weight_offset,
                                         std::optional<U32> explicit_t = {}) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = affine_nax_mp(t);
  const bool dyn_rows = !explicit_t.has_value() && t != 1u;
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() || t == 1u ? std::nullopt : std::optional<U32>(1u);
  auto &d = e.emit("affine_nax", "omlx_m5_affine_add_rmsnorm_xsum_g64",
                   {.width = 128u, .height = mp}, {.width = 128u},
                   {hidden, delta, weight, output_a16, xs});
  e.dynamic(d, 5, words({k, t, mp, word(eps), word(weight_offset)}),
            tokens_patch);
  if (t > 1u) {
    const Encoder::GroupSumKey key{xs.buffer, xs.offset, output_a16.buffer,
                                   output_a16.offset, k, 64u, t, dyn_rows, 2u};
    e.set_group_sum(key);
  }
}

inline U32 affine_nax_bm(U32 sel_t, U32 rf, U32 bn, U32 sk) {
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

inline void ensure_affine_nax_xsum(Encoder &e, Ref x, Ref xs, U32 k,
                                   std::optional<U32> explicit_t = {}) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  if (t == 1u) {
    // At T=1, affine_nax computes per-group64 xsum inline via simd_sum (0 ULP).
    return;
  }
  const U32 mp = affine_nax_mp(t);
  const bool dyn_rows = !explicit_t.has_value() && t != 1u;
  const Encoder::GroupSumKey key{xs.buffer, xs.offset, x.buffer, x.offset,
                                 k,         64u,       t,        dyn_rows, 2u};
  if (e.has_group_sum(key)) {
    return;
  }
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  auto &d = e.emit("affine_nax", "omlx_m5_affine_nax_xsum_g64",
                   {.width = U64(k / 64u) * 32u, .height = mp}, {.width = 32u},
                   {x, xs});
  e.dynamic(d, 2, words({k, mp, 0u, t}), tokens_patch);
  e.set_group_sum(key);
}

// Single-pass FWHT-64 BF16 rotation + GS64 xsum (writes x_rot = x @ H_64 and xs).
inline void ensure_affine_fwht64_xsum(Encoder &e, Ref x, Ref x_rot_dst, Ref xs,
                                      U32 k,
                                      std::optional<U32> explicit_t = {}) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = affine_nax_mp(t);
  const bool dyn_rows = !explicit_t.has_value() && t != 1u;
  const Encoder::GroupSumKey key{xs.buffer, xs.offset, x.buffer, x.offset,
                                 k,         64u,       t,        dyn_rows,
                                 2u,        true};
  const Encoder::GroupSumKey rot_key{xs.buffer,        xs.offset,
                                     x_rot_dst.buffer, x_rot_dst.offset,
                                     k,                64u,
                                     t,                dyn_rows,
                                     2u,               false};
  if (e.has_group_sum(key) && e.has_group_sum(rot_key)) {
    return;
  }
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  auto &d = e.emit("affine_nax", "omlx_m5_affine_fwht64_xsum_g64",
                   {.width = U64(k / 64u) * 32u, .height = mp}, {.width = 32u},
                   {x, x_rot_dst, xs});
  e.dynamic(d, 3, words({k, mp, 0u, t}), tokens_patch);
  e.set_group_sum(key);
  e.set_group_sum(rot_key);
}

inline NaxWidePrefillTile
affine_nax_select_wide(const TiledAffineWeight &tw, U32 t,
                       std::optional<U32> explicit_t = {},
                       U32 gpu_cores = 64u) {
  if (std::getenv("OMLX_M5_AFFINE_NAX_SK") != nullptr) {
    return NaxWidePrefillTile::none;
  }
  return nax_select_wide_prefill_tile(tw.padded_n, t, explicit_t, gpu_cores);
}

inline bool affine_nax_q5_h1_bm16_eligible(const TiledAffineWeight &tw) {
  if (tw.bits != 5u || tw.high1.buffer == nullptr ||
      tw.sb_tiled.buffer == nullptr || (tw.k / 64u) < 2u) {
    return false;
  }
  if (const char *dis = std::getenv("OMLX_M5_DISABLE_Q5_H1_BM16");
      dis != nullptr && std::string_view(dis) == "1") {
    return false;
  }
  if (std::getenv("OMLX_M5_AFFINE_NAX_SK") != nullptr ||
      std::getenv("OMLX_M5_NAX_MAX_RF") != nullptr) {
    return false;
  }
  return true;
}

inline U32 affine_nax_q5_h1_bm16_linear_sk(const TiledAffineWeight &tw) {
  const U32 groups = tw.k / 64u;
  // WP3.1: Cap SK <= 4 (<= 16.0 KiB TG SRAM on Apple10).
  U32 sk = (tw.n >= 65536u) ? 2u : 4u;
  while (sk > 2u && sk > groups) {
    sk /= 2u;
  }
  return sk;
}

inline U32 affine_nax_q5_h1_bm16_swiglu_sk(const TiledAffineWeight &gw) {
  const U32 groups = gw.k / 64u;
  return (groups >= 4u) ? 4u : 2u;
}

inline std::string affine_nax_group(const Encoder &e,
                                    const TiledAffineWeight &tw,
                                    std::optional<U32> explicit_t = {},
                                    U32 gpu_cores = 64u,
                                    U32 peer_group_size = 0u) {
  const U32 sel_t = nax_effective_select_t(e, explicit_t);
  const auto sel_exp = nax_effective_select_explicit_t(e, explicit_t);
  const U32 eff_gs =
      ((peer_group_size == 0u ? tw.group_size
                              : std::min(tw.group_size, peer_group_size)) ==
           128u &&
       (tw.k % 128u) == 0u)
          ? 128u
          : 64u;
  const auto wide = affine_nax_select_wide(tw, sel_t, sel_exp, gpu_cores);
  if (wide != NaxWidePrefillTile::none) {
    const char *tag =
        (wide == NaxWidePrefillTile::bm64_bn64) ? "bm64_bn64" : "bm32_bn32";
    return "oq.nax.q" + std::to_string(tw.bits) + "_g" +
           std::to_string(eff_gs) + ".bns" + std::to_string(tw.bn) + "." + tag +
           ".sk1.v1";
  }
  if (affine_nax_q5_h1_bm16_eligible(tw)) {
    const U32 sk = (peer_group_size != 0u) ? affine_nax_q5_h1_bm16_swiglu_sk(tw)
                                           : affine_nax_q5_h1_bm16_linear_sk(tw);
    return "oq.nax.q5_h1_g" + std::to_string(tw.group_size) + ".bm16_bns" +
           std::to_string(tw.bn) + ".sk" + std::to_string(sk) + ".v1";
  }
  return "oq.nax.q" + std::to_string(tw.bits) + "_g" +
         std::to_string(tw.group_size) + ".n" + std::to_string(tw.bn) + ".sk" +
         std::to_string(tw.split_k) + ".v1";
}

inline bool affine_nax_swiglu_out64_enabled() noexcept {
  // WP3.1 Apple10 M5 Ultra verified: 1-pass BN_out=32 out64 (9,216 B SRAM, 0 spills,
  // 128 threads, 3 TGs/core) beats HEAD out64 (27,648 B) by 1.10-1.16x and beats/matches
  // Split (2x Linear + SwiGLU) & bm64_bn64 across M=512..4096 (default ON).
  if (const char *dis = std::getenv("OMLX_M5_DISABLE_SWIGLU_OUT64");
      dis != nullptr && std::string_view(dis) == "1") {
    return false;
  }
  if (const char *en = std::getenv("OMLX_M5_ENABLE_SWIGLU_OUT64");
      en != nullptr) {
    if (std::string_view(en) == "0") {
      return false;
    }
    if (std::string_view(en) == "1") {
      return true;
    }
  }
  return true;
}

inline void linear_affine_nax(Encoder &e, const TiledAffineWeight &tw, Ref x,
                              Ref xs, Ref y,
                              std::optional<U32> explicit_t = {},
                              U32 gpu_cores = 64u,
                              U32 row_offset = 0u,
                              U32 out_stride = 0u,
                              U32 slice_n = 0u) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = affine_nax_mp(t);
  const U32 sel_t = nax_effective_select_t(e, explicit_t);
  const auto sel_exp = nax_effective_select_explicit_t(e, explicit_t);
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  const std::string tag = affine_nax_mode_tag(tw.bits);
  if (row_offset > 0u || (out_stride > 0u && out_stride != tw.n) ||
      (slice_n > 0u && slice_n != tw.n)) {
    if ((row_offset % 64u) != 0u || row_offset >= tw.n) {
      throw std::invalid_argument(
          "linear_affine_nax slice requires row_offset % 64 == 0 && row_offset < n");
    }
    const U32 eff_n = (slice_n > 0u) ? slice_n : (tw.n - row_offset);
    if (row_offset + eff_n > tw.n) {
      throw std::invalid_argument("linear_affine_nax slice exceeds weight n");
    }
    const U32 eff_stride = (out_stride > 0u) ? out_stride : tw.n;
    const U32 eff_gs =
        (tw.group_size == 128u && (tw.k % 128u) == 0u) ? 128u : 64u;
    const U32 x_tiles = (eff_n + 63u) / 64u;
    const U32 y_tiles = (t + 63u) / 64u;
    const std::string fn = "omlx_m5_affine_nax_" + tag + "_g" +
                           std::to_string(eff_gs) + "_bm64_bn64_bns" +
                           std::to_string(tw.bn) + "_slice_sk1";
    const MetalDispatchSize grid{.width = U64(x_tiles) * 128u,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = 128u};
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    if (!affine_nax_has_high_plane(tw.bits)) {
      auto &d = e.emit("affine_nax", fn, grid, threads,
                       {x, tw.codes, tw.sb, y});
      e.dynamic(d, 4,
                words({eff_n, tw.k, mp, t, tw.padded_n, 1u, row_offset,
                       eff_stride}),
                tokens_patch);
    } else {
      auto &d = e.emit("affine_nax", fn, grid, threads,
                       {x, tw.codes, tw.high2, tw.sb, y});
      e.dynamic(d, 5,
                words({eff_n, tw.k, mp, t, tw.padded_n, 1u, row_offset,
                       eff_stride}),
                tokens_patch);
    }
    return;
  }
  const auto wide = affine_nax_select_wide(tw, sel_t, sel_exp, gpu_cores);
  if (wide != NaxWidePrefillTile::none) {
    const bool is_64 = (wide == NaxWidePrefillTile::bm64_bn64);
    const U32 tile_mn = is_64 ? 64u : 32u;
    const U32 tptg = is_64 ? 128u : 32u;
    const U32 x_tiles = (tw.padded_n + tile_mn - 1u) / tile_mn;
    const U32 y_tiles = (t + tile_mn - 1u) / tile_mn;
    const U32 eff_gs =
        (tw.group_size == 128u && (tw.k % 128u) == 0u) ? 128u : 64u;
    const std::string fn = "omlx_m5_affine_nax_" + tag + "_g" +
                           std::to_string(eff_gs) + "_" +
                           (is_64 ? "bm64_bn64" : "bm32_bn32") + "_bns" +
                           std::to_string(tw.bn) + "_sk1";
    const MetalDispatchSize grid{.width = U64(x_tiles) * tptg,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = tptg};
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    if (!affine_nax_has_high_plane(tw.bits)) {
      auto &d = e.emit("affine_nax", fn, grid, threads,
                       {x, tw.codes, tw.sb, y});
      e.dynamic(d, 4, words({tw.n, tw.k, mp, t, tw.padded_n, 1u}),
                tokens_patch);
    } else {
      auto &d = e.emit("affine_nax", fn, grid, threads,
                       {x, tw.codes, tw.high2, tw.sb, y});
      e.dynamic(d, 5, words({tw.n, tw.k, mp, t, tw.padded_n, 1u}),
                tokens_patch);
    }
    return;
  }

  if (affine_nax_q5_h1_bm16_eligible(tw)) {
    const U32 sk = affine_nax_q5_h1_bm16_linear_sk(tw);
    const U32 bm = (sel_t >= 17u) ? 32u : 16u;
    const U32 x_tiles = (tw.padded_n + 31u) / 32u;
    const U32 y_tiles = (t + bm - 1u) / bm;
    const std::string fn = "omlx_m5_affine_nax_q5_h1_bm" + std::to_string(bm) +
                           "_bns" + std::to_string(tw.bn) + "_sk" +
                           std::to_string(sk);
    const MetalDispatchSize grid{.width = U64(x_tiles) * 32u * sk,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = 32u * sk};
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    auto &d = e.emit("affine_nax", fn, grid, threads,
                     {x, tw.codes, tw.high1, tw.sb_tiled, y});
    e.dynamic(d, 5, words({tw.n, tw.k, mp, t, 0u, sk}), tokens_patch);
    return;
  }

  ensure_affine_nax_xsum(e, x, xs, tw.k, explicit_t);

  const U32 rf = nax_select_row_factor(sel_t);
  const U32 bm = affine_nax_bm(sel_t, rf, tw.bn, tw.split_k);
  const bool use_sb_tiled = (tw.sb_tiled.buffer != nullptr);
  const Ref sb_ref = use_sb_tiled ? tw.sb_tiled : tw.sb;
  const U32 sb_stride_n = use_sb_tiled ? 0u : tw.padded_n;
  if (bm > 8u) {
    const U32 y_tiles = (mp + bm - 1u) / bm;
    const U32 x_tiles = tw.padded_n / tw.bn;
    const std::string fn = "omlx_m5_affine_nax_" + tag + "_bm" +
                           std::to_string(bm) + "_n" + std::to_string(tw.bn) +
                           "_sk" + std::to_string(tw.split_k);
    const MetalDispatchSize grid{.width = U64(x_tiles) * 32u * tw.split_k,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = 32u * tw.split_k};
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    if (!affine_nax_has_high_plane(tw.bits)) {
      auto &d = e.emit("affine_nax", fn, grid, threads,
                       {x, tw.codes, sb_ref, xs, y});
      e.dynamic(d, 5, words({tw.n, tw.k, mp, t, sb_stride_n, tw.split_k}),
                tokens_patch);
    } else {
      auto &d = e.emit("affine_nax", fn, grid, threads,
                       {x, tw.codes, tw.high2, sb_ref, xs, y});
      e.dynamic(d, 6, words({tw.n, tw.k, mp, t, sb_stride_n, tw.split_k}),
                tokens_patch);
    }
    return;
  }

  const U32 sel_y_tiles = nax_y_tiles(affine_nax_mp(sel_t), rf);
  const U32 y_tiles = nax_y_tiles(mp, rf);
  const U32 nsg =
      (tw.bn == 16u) ? nax_select_nsg_bn16(tw.padded_n, sel_y_tiles, tw.split_k,
                                           gpu_cores, true)
                     : 1u;

  std::string fn;
  if (rf == 1u && nsg == 1u) {
    fn = "omlx_m5_affine_nax_" + tag + "_n" + std::to_string(tw.bn) + "_sk" +
         std::to_string(tw.split_k);
  } else if (tw.bn == 16u) {
    fn = "omlx_m5_affine_nax_" + tag + "_n16_sk" +
         std::to_string(tw.split_k) + "_rf" + std::to_string(rf) + "_nsg" +
         std::to_string(nsg);
  } else {
    fn = "omlx_m5_affine_nax_" + tag + "_n32_sk" +
         std::to_string(tw.split_k) + "_rf" + std::to_string(rf);
  }
  const U32 x_tiles = (tw.padded_n / tw.bn + nsg - 1u) / nsg;
  const MetalDispatchSize grid{.width = U64(x_tiles) * 32u * tw.split_k * nsg,
                               .height = y_tiles};
  const MetalDispatchSize threads{.width = 32u * tw.split_k * nsg};
  e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
  if (!affine_nax_has_high_plane(tw.bits)) {
    auto &d = e.emit("affine_nax", fn, grid, threads,
                     {x, tw.codes, sb_ref, xs, y});
    e.dynamic(d, 5, words({tw.n, tw.k, mp, t, sb_stride_n, tw.split_k}),
              tokens_patch);
  } else {
    auto &d = e.emit("affine_nax", fn, grid, threads,
                     {x, tw.codes, tw.high2, sb_ref, xs, y});
    e.dynamic(d, 6, words({tw.n, tw.k, mp, t, sb_stride_n, tw.split_k}),
              tokens_patch);
  }
}

inline void linear_affine_nax_swiglu(Encoder &e, const TiledAffineWeight &gw,
                                     const TiledAffineWeight &uw, Ref x,
                                     Ref xs, Ref y, U32 gpu_cores = 64u,
                                     std::optional<U32> explicit_t = {},
                                     U32 row_offset = 0u,
                                     U32 out_stride = 0u,
                                     U32 slice_n = 0u) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = affine_nax_mp(t);
  const U32 sel_t = nax_effective_select_t(e, explicit_t);
  const auto sel_exp = nax_effective_select_explicit_t(e, explicit_t);
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  const std::string tag_g = affine_nax_mode_tag(gw.bits);
  const std::string tag_u = affine_nax_mode_tag(uw.bits);
  const bool mixed_q4_q56 = (tag_g == "q4" && tag_u == "q56") ||
                            (tag_g == "q56" && tag_u == "q4");
  const bool is_slice = (row_offset > 0u) ||
                        (out_stride > 0u && out_stride != gw.n) ||
                        (slice_n > 0u && slice_n != gw.n);
  const auto wide = affine_nax_select_wide(gw, sel_t, sel_exp, gpu_cores);
  if (is_slice || mixed_q4_q56 ||
      (wide == NaxWidePrefillTile::bm64_bn64 && affine_nax_swiglu_out64_enabled())) {
    if ((row_offset % 64u) != 0u || row_offset >= gw.n) {
      throw std::invalid_argument(
          "linear_affine_nax_swiglu out64 requires row_offset % 64 == 0 && row_offset < n");
    }
    const U32 eff_n = (slice_n > 0u) ? slice_n : (gw.n - row_offset);
    if (row_offset + eff_n > gw.n) {
      throw std::invalid_argument("linear_affine_nax_swiglu slice exceeds weight n");
    }
    const U32 eff_stride = (out_stride > 0u) ? out_stride : gw.n;
    const U32 x_tiles = (eff_n + 31u) / 32u;
    const U32 y_tiles = (t + 63u) / 64u;
    const U32 eff_gs = (gw.group_size == 128u && uw.group_size == 128u &&
                        (gw.k % 128u) == 0u)
                           ? 128u
                           : 64u;
    const std::string mode_tag =
        (tag_g == tag_u) ? tag_g : (tag_g + "_" + tag_u);
    const std::string fn = "omlx_m5_affine_nax_" + mode_tag + "_g" +
                           std::to_string(eff_gs) +
                           "_swiglu_bm64_bn64_out64_sk1";
    const MetalDispatchSize grid{.width = U64(x_tiles) * 128u,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = 128u};
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    if (!affine_nax_has_high_plane(gw.bits) &&
        !affine_nax_has_high_plane(uw.bits)) {
      auto &d = e.emit("affine_nax", fn, grid, threads,
                       {x, gw.codes, gw.sb, uw.codes, uw.sb, y});
      e.dynamic(d, 6,
                words({eff_n, gw.k, mp, t, gw.padded_n, 1u, row_offset,
                       eff_stride}),
                tokens_patch);
    } else {
      const Ref g_hi = gw.high2.buffer != nullptr ? gw.high2 : gw.codes;
      const Ref u_hi = uw.high2.buffer != nullptr ? uw.high2 : uw.codes;
      auto &d = e.emit("affine_nax", fn, grid, threads,
                       {x, gw.codes, g_hi, gw.sb, uw.codes, u_hi, uw.sb, y});
      e.dynamic(d, 8,
                words({eff_n, gw.k, mp, t, gw.padded_n, 1u, row_offset,
                       eff_stride}),
                tokens_patch);
    }
    return;
  }
  const std::string tag = tag_g;
  if (wide != NaxWidePrefillTile::none) {
    const bool is_64 = (wide == NaxWidePrefillTile::bm64_bn64);
    const U32 tile_m = is_64 ? 64u : 32u;
    const U32 tile_n = is_64 ? 32u : 16u;
    const U32 tptg = is_64 ? 128u : 32u;
    const U32 x_tiles = (gw.padded_n + tile_n - 1u) / tile_n;
    const U32 y_tiles = (t + tile_m - 1u) / tile_m;
    const U32 eff_gs = (gw.group_size == 128u && uw.group_size == 128u &&
                        (gw.k % 128u) == 0u)
                           ? 128u
                           : 64u;
    const std::string fn = "omlx_m5_affine_nax_" + tag + "_g" +
                           std::to_string(eff_gs) + "_swiglu_" +
                           (is_64 ? "bm64_bn64_sk1" : "bm32_bn32_sk1");
    const MetalDispatchSize grid{.width = U64(x_tiles) * tptg,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = tptg};
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    if (!affine_nax_has_high_plane(gw.bits)) {
      auto &d = e.emit("affine_nax", fn, grid, threads,
                       {x, gw.codes, gw.sb, uw.codes, uw.sb, y});
      e.dynamic(d, 6, words({gw.n, gw.k, mp, t, gw.padded_n, 1u}),
                tokens_patch);
    } else {
      auto &d = e.emit("affine_nax", fn, grid, threads,
                       {x, gw.codes, gw.high2, gw.sb, uw.codes, uw.high2, uw.sb,
                        y});
      e.dynamic(d, 8, words({gw.n, gw.k, mp, t, gw.padded_n, 1u}),
                tokens_patch);
    }
    return;
  }

  if (affine_nax_q5_h1_bm16_eligible(gw) &&
      affine_nax_q5_h1_bm16_eligible(uw) && gw.bn == 16u && uw.bn == 16u) {
    const U32 sk = affine_nax_q5_h1_bm16_swiglu_sk(gw);
    const U32 bm = (sel_t >= 17u) ? 32u : 16u;
    const std::string fn = "omlx_m5_affine_nax_q5_h1_swiglu_bm" +
                           std::to_string(bm) + "_sk" + std::to_string(sk);
    const MetalDispatchSize grid{.width = U64(gw.padded_n / 16u) * 32u * sk,
                                 .height = (t + bm - 1u) / bm};
    const MetalDispatchSize threads{.width = 32u * sk};
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    auto &d = e.emit("affine_nax", fn, grid, threads,
                     {x, gw.codes, gw.high1, gw.sb_tiled, uw.codes, uw.high1,
                      uw.sb_tiled, y});
    e.dynamic(d, 8, words({gw.n, gw.k, mp, t, 0u, sk}), tokens_patch);
    return;
  }

  ensure_affine_nax_xsum(e, x, xs, gw.k, explicit_t);

  const U32 rf = nax_select_row_factor(sel_t);
  const U32 bm = affine_nax_bm(sel_t, rf, 16u, gw.split_k);
  const bool use_sb_tiled =
      (gw.sb_tiled.buffer != nullptr && uw.sb_tiled.buffer != nullptr);
  const Ref g_sb_ref = use_sb_tiled ? gw.sb_tiled : gw.sb;
  const Ref u_sb_ref = use_sb_tiled ? uw.sb_tiled : uw.sb;
  const U32 sb_stride_n = use_sb_tiled ? 0u : gw.padded_n;
  if (bm > 8u) {
    // WP3.1: Cap SwiGLU bm32_n16 at SK <= 4 (12.0 KiB <= 16.0 KiB TG SRAM).
    const U32 eff_sk = (bm == 32u) ? std::min<U32>(gw.split_k, 4u) : gw.split_k;
    const U32 y_tiles = (mp + bm - 1u) / bm;
    const std::string fn = "omlx_m5_affine_nax_" + tag + "_swiglu_bm" +
                           std::to_string(bm) + "_n16_sk" +
                           std::to_string(eff_sk);
    const MetalDispatchSize grid{.width = U64(gw.padded_n / 16u) * 32u * eff_sk,
                                 .height = y_tiles};
    const MetalDispatchSize threads{.width = 32u * eff_sk};
    e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
    if (!affine_nax_has_high_plane(gw.bits)) {
      auto &d = e.emit("affine_nax", fn, grid, threads,
                       {x, gw.codes, g_sb_ref, uw.codes, u_sb_ref, xs, y});
      e.dynamic(d, 7, words({gw.n, gw.k, mp, t, sb_stride_n, eff_sk}),
                tokens_patch);
    } else {
      auto &d = e.emit("affine_nax", fn, grid, threads,
                       {x, gw.codes, gw.high2, g_sb_ref, uw.codes, uw.high2,
                        u_sb_ref, xs, y});
      e.dynamic(d, 9, words({gw.n, gw.k, mp, t, sb_stride_n, eff_sk}),
                tokens_patch);
    }
    return;
  }

  const U32 y_tiles = nax_y_tiles(mp, rf);
  std::string fn = "omlx_m5_affine_nax_" + tag + "_swiglu_n16_sk" +
                   std::to_string(gw.split_k);
  if (rf > 1u) {
    fn += "_rf" + std::to_string(rf);
  }
  const MetalDispatchSize grid{.width = U64(gw.padded_n / 16u) * 32u * gw.split_k,
                               .height = y_tiles};
  const MetalDispatchSize threads{.width = 32u * gw.split_k};
  e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
  if (!affine_nax_has_high_plane(gw.bits)) {
    auto &d = e.emit("affine_nax", fn, grid, threads,
                     {x, gw.codes, g_sb_ref, uw.codes, u_sb_ref, xs, y});
    e.dynamic(d, 7, words({gw.n, gw.k, mp, t, sb_stride_n, gw.split_k}),
              tokens_patch);
  } else {
    auto &d = e.emit("affine_nax", fn, grid, threads,
                     {x, gw.codes, gw.high2, g_sb_ref, uw.codes, uw.high2,
                      u_sb_ref, xs, y});
    e.dynamic(d, 9, words({gw.n, gw.k, mp, t, sb_stride_n, gw.split_k}),
              tokens_patch);
  }
}

// FWHT-64 wrapper for W4A16 affine_nax linear (rotates x -> x_rot_scratch).
inline void linear_affine_nax_fwht64(Encoder &e, const TiledAffineWeight &tw,
                                     Ref x, Ref x_rot_scratch, Ref xs, Ref y,
                                     std::optional<U32> explicit_t = {},
                                     U32 gpu_cores = 64u,
                                     U32 row_offset = 0u,
                                     U32 out_stride = 0u,
                                     U32 slice_n = 0u) {
  const U32 sel_t = nax_effective_select_t(e, explicit_t);
  const auto sel_exp = nax_effective_select_explicit_t(e, explicit_t);
  const bool is_slice = (row_offset > 0u) ||
                        (out_stride > 0u && out_stride != tw.n) ||
                        (slice_n > 0u && slice_n != tw.n);
  const bool uses_bf16_only =
      is_slice ||
      (affine_nax_select_wide(tw, sel_t, sel_exp, gpu_cores) !=
       NaxWidePrefillTile::none) ||
      affine_nax_q5_h1_bm16_eligible(tw);
  if (uses_bf16_only) {
    emit_fwht64_bf16(e, x, x_rot_scratch, tw.k, explicit_t);
  } else {
    ensure_affine_fwht64_xsum(e, x, x_rot_scratch, xs, tw.k, explicit_t);
  }
  linear_affine_nax(e, tw, x_rot_scratch, xs, y, explicit_t, gpu_cores,
                    row_offset, out_stride, slice_n);
}

// WP3.1 Mode B: Zero-Ws W4A16 out64 Gate + Up + SwiGLU + fused FWHT-64 + GS64 INT8 epilogue.
// Directly produces (qa_out, sa_out, ra_out) for a W4A8 fwht64 down_proj with 0 extra dispatches.
// If also_write_bf16 == true (Bypass LoRA / tensor inspection), also writes unrotated SwiGLU to y_bf16.
inline void linear_affine_nax_swiglu_fwht64_a8(
    Encoder &e, const TiledAffineWeight &gw, const TiledAffineWeight &uw, Ref x,
    Ref qa_out, Ref sa_out, Ref ra_out, Ref opt_y_bf16, bool also_write_bf16,
    U32 a8_mp, std::optional<U32> explicit_t = {}, U32 row_offset = 0u,
    U32 out_stride = 0u, U32 slice_n = 0u) {
  if (gw.n != uw.n || gw.k != uw.k || gw.bn != 16u || uw.bn != 16u ||
      (gw.k % 64u) != 0u || (gw.n % 64u) != 0u || (row_offset % 64u) != 0u ||
      row_offset >= gw.n) {
    throw std::invalid_argument(
        "linear_affine_nax_swiglu_fwht64_a8 requires bn=16, k%64==0, n%64==0, row_offset%64==0");
  }
  const U32 eff_n = (slice_n > 0u) ? slice_n : (gw.n - row_offset);
  if (row_offset + eff_n > gw.n || (eff_n % 64u) != 0u) {
    throw std::invalid_argument(
        "linear_affine_nax_swiglu_fwht64_a8 slice must be 64-aligned within n");
  }
  const U32 eff_stride = (out_stride > 0u) ? out_stride : gw.n;
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 x_tiles = eff_n / 64u;
  const U32 y_tiles = (t + 63u) / 64u;
  const U32 eff_gs = (gw.group_size == 128u && uw.group_size == 128u &&
                      (gw.k % 128u) == 0u)
                         ? 128u
                         : 64u;
  const std::string tag_g = affine_nax_mode_tag(gw.bits);
  const std::string tag_u = affine_nax_mode_tag(uw.bits);
  const std::string mode_tag =
      (tag_g == tag_u) ? tag_g : (tag_g + "_" + tag_u);
  const std::string fn = "omlx_m5_affine_nax_" + mode_tag + "_g" +
                         std::to_string(eff_gs) +
                         "_swiglu_bm64_bn64_out64_sk1_fwht64_a8";
  const MetalDispatchSize grid{.width = U64(x_tiles) * 256u, .height = y_tiles};
  const MetalDispatchSize threads{.width = 256u};
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  const U32 epi_flags = also_write_bf16 ? 1u : 0u;
  e.profile.numeric_contract_id = "linear_row_exact_portable_v1";
  if (!affine_nax_has_high_plane(gw.bits) &&
      !affine_nax_has_high_plane(uw.bits)) {
    auto &d = e.emit("affine_nax", fn, grid, threads,
                     {x, gw.codes, gw.sb, uw.codes, uw.sb, qa_out, sa_out,
                      ra_out, opt_y_bf16});
    e.dynamic(d, 9,
              words({eff_n, gw.k, a8_mp, t, gw.padded_n, epi_flags, row_offset,
                     eff_stride}),
              tokens_patch);
  } else {
    const Ref g_hi = gw.high2.buffer != nullptr ? gw.high2 : gw.codes;
    const Ref u_hi = uw.high2.buffer != nullptr ? uw.high2 : uw.codes;
    auto &d = e.emit("affine_nax", fn, grid, threads,
                     {x, gw.codes, g_hi, gw.sb, uw.codes, u_hi, uw.sb, qa_out,
                      sa_out, ra_out, opt_y_bf16});
    e.dynamic(d, 11,
              words({eff_n, gw.k, a8_mp, t, gw.padded_n, epi_flags, row_offset,
                     eff_stride}),
              tokens_patch);
  }
}

inline void linear_affine_nax_q4_swiglu(Encoder &e, const TiledAffineWeight &gw,
                                        const TiledAffineWeight &uw, Ref x,
                                        Ref xs, Ref y) {
  linear_affine_nax_swiglu(e, gw, uw, x, xs, y);
}

} // namespace omlx::m5::detail::forward
