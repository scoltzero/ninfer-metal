#pragma once

#include "affine_nax_cache.h"
#include "bf16_nax.h"
#include "forward_plan.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/lora.h"
#include "omlx_m5/materialization.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace omlx::m5 {

struct QwenMergedLoraOverrides {
  std::string adapter_path;
  std::string fingerprint;
  std::uint64_t buffer_bytes = 0;
  std::uint32_t adapted_layers = 0;
  std::vector<std::unique_ptr<MetalBuffer>> segments;
  std::unordered_map<std::uint32_t, MaterializedWeightView> views_by_object_id;
};

struct QwenBypassLoraEntry {
  std::uint32_t object_id = 0;
  std::uint32_t n = 0;
  std::uint32_t k = 0;
  std::uint32_t r_valid = 0;
  std::uint32_t r_slot = 0;
  float scale = 1.0F;
  detail::forward::Ref a_ref{}; // [r_slot, k] BF16
  detail::forward::Ref b_ref{}; // [align16(n), r_slot] BF16
};

struct QwenBypassSiblingSlice {
  std::uint32_t object_id = 0;
  std::uint32_t part_index = 0;
  std::uint32_t n = 0;
  std::uint32_t r_valid = 0;
  std::uint32_t r_slot = 0;
  std::uint32_t r_slot_offset = 0;
  float scale = 1.0F;
  detail::forward::Ref b_ref{};
};

struct QwenBypassSiblingGroup {
  std::uint32_t k = 0;
  std::uint32_t total_r_slot = 0;
  detail::forward::Ref concat_a_ref{}; // [total_r_slot, k] BF16 (contiguous slice)
  std::vector<QwenBypassSiblingSlice> slices;
};

enum class QwenBypassGroupKind : std::uint32_t {
  attn_qkv = 0,
  gdn_inputs = 1,
  mlp_gate_up = 2,
  moe_shared_gate_up = 3,
};

struct QwenBypassLoraRuntime {
  std::string adapter_path;
  std::string fingerprint;
  std::uint64_t weight_buffer_bytes = 0;
  std::uint32_t adapted_layers = 0;
  std::uint32_t max_r_slot = 0;
  std::unique_ptr<MetalBuffer> weight_slab;
  std::unordered_map<std::uint32_t, QwenBypassLoraEntry> entries_by_object_id;
  // Key: (layer_index << 4) | uint32_t(QwenBypassGroupKind)
  std::unordered_map<std::uint32_t, QwenBypassSiblingGroup> sibling_groups;

  [[nodiscard]] bool has(std::uint32_t object_id) const noexcept {
    return entries_by_object_id.contains(object_id);
  }

  [[nodiscard]] const QwenBypassLoraEntry *
  find(std::uint32_t object_id) const noexcept {
    const auto it = entries_by_object_id.find(object_id);
    return it == entries_by_object_id.end() ? nullptr : &it->second;
  }

  [[nodiscard]] const QwenBypassSiblingGroup *
  find_group(std::uint32_t layer_index,
             QwenBypassGroupKind kind) const noexcept {
    const std::uint32_t gkey =
        (layer_index << 4U) | static_cast<std::uint32_t>(kind);
    const auto it = sibling_groups.find(gkey);
    return it == sibling_groups.end() ? nullptr : &it->second;
  }
};

namespace detail::lora_rt {

using U32 = std::uint32_t;
using U64 = std::uint64_t;
using detail::forward::Encoder;
using detail::forward::NaxSegmentArena;
using detail::forward::Ref;
using detail::forward::mul;
using detail::forward::plane;
using detail::forward::word;
using detail::forward::words;

inline U64 align_256(U64 v) noexcept { return (v + 255ULL) & ~255ULL; }
inline U32 align_16(U32 v) noexcept { return ((v + 15U) / 16U) * 16U; }

inline std::uint32_t resolve_target_object_id(const QwenLayerCatalog &catalog,
                                              const LoraTargetKey &target) {
  if (target.is_global) {
    for (const auto &st : catalog.global_stages()) {
      if (st.role == target.role && st.object_id != UINT32_MAX) {
        return st.object_id;
      }
    }
    throw std::invalid_argument("LoRA global target role not found: " +
                                target.role);
  }
  const auto &layer = catalog.layer(target.layer_index);
  for (const auto &st : layer.stages) {
    if (st.role == target.role && st.object_id != UINT32_MAX) {
      return st.object_id;
    }
  }
  throw std::invalid_argument("LoRA layer " +
                              std::to_string(target.layer_index) +
                              " role not found: " + target.role);
}

inline U64 estimate_lora_gpu_bytes(const LoadedLoraAdapter &adapter,
                                   std::string_view mode, U32 workspace_rows,
                                   U32 max_concurrency) {
  if (mode == "merge") {
    U64 total = 0;
    for (const auto &u : adapter.updates) {
      const U32 n = u.out_features;
      const U32 k = u.in_features;
      const U32 bn = (n >= 65536U) ? 32U : 16U;
      const U32 padded_n = ((n + bn - 1U) / bn) * bn;
      total += align_256(mul(padded_n, k));
      const U64 sb_bytes = align_256(mul(mul(padded_n, k / 64U), 4U));
      total += 2ULL * sb_bytes;
    }
    return total;
  }
  U64 weights_bytes = 0;
  std::unordered_map<U64, U32> layer_grp_r;
  U32 max_r_slot = 16U;
  for (const auto &u : adapter.updates) {
    const U32 r_slot = std::max<U32>(16U, align_16(u.rank));
    const U32 n_pad = std::max<U32>(16U, align_16(u.out_features));
    weights_bytes += mul(mul(r_slot, u.in_features), 2U);
    weights_bytes += align_256(mul(mul(n_pad, r_slot), 2U));
    max_r_slot = std::max(max_r_slot, r_slot);
    if (!u.target.is_global) {
      U32 gk = 99U;
      if (u.target.role == "attention_q" || u.target.role == "attention_k" ||
          u.target.role == "attention_v") {
        gk = 0U;
      } else if (u.target.role == "gdn_qkv" || u.target.role == "gdn_z" ||
                 u.target.role == "gdn_a" || u.target.role == "gdn_b") {
        gk = 1U;
      } else if (u.target.role == "mlp_gate" || u.target.role == "mlp_up") {
        gk = 2U;
      } else if (u.target.role == "moe_shared_mlp_gate" ||
                 u.target.role == "moe_shared_mlp_up") {
        gk = 3U;
      }
      if (gk < 4U) {
        const U64 key = (U64(u.target.layer_index) << 4U) | gk;
        const U32 sum_r = (layer_grp_r[key] += r_slot);
        max_r_slot = std::max(max_r_slot, sum_r);
      }
    }
  }
  const U32 mp = ((workspace_rows + 7U) / 8U) * 8U;
  const U64 scratch_per_slot = align_256(mul(mul(mp, max_r_slot), 2U));
  return align_256(weights_bytes) + U64(max_concurrency) * scratch_per_slot;
}

inline std::shared_ptr<QwenMergedLoraOverrides> build_merged_lora_overrides(
    const MetalContext &ctx, const std::filesystem::path &libs,
    const MaterializedModel &weights, const QwenLayerCatalog &catalog,
    const LoadedLoraAdapter &adapter) {
  auto out = std::make_shared<QwenMergedLoraOverrides>();
  out->adapter_path = adapter.adapter_dir.string();
  out->fingerprint = adapter.fingerprint;
  out->adapted_layers = adapter.adapted_layers;
  if (adapter.updates.empty()) {
    return out;
  }

  struct MergeJob {
    const LoraProjectionUpdate *upd = nullptr;
    MaterializedWeightView base_w;
    U32 n = 0;
    U32 k = 0;
    U32 bn = 16;
    U32 padded_n = 0;
    bool use_nax_tiled = true;
    U64 a_off = 0;
    U64 a_bytes = 0;
    U64 b_off = 0;
    U64 b_bytes = 0;
    NaxSegmentArena::Placement codes_p{};
    NaxSegmentArena::Placement sb_p{};
    NaxSegmentArena::Placement sb_tiled_p{};
  };

  const bool nax_eligible =
      detail::forward::affine_nax_runtime_eligible(ctx, libs);
  NaxSegmentArena dst_arena;
  std::vector<MergeJob> jobs;
  jobs.reserve(adapter.updates.size());
  U64 staging_bytes = 256ULL; // first 256B zero sentinel

  for (const auto &u : adapter.updates) {
    const U32 obj_id = resolve_target_object_id(catalog, u.target);
    MaterializedWeightView bw = weights.parent_view(obj_id);
    if (bw.parent_shape.size() != 2 || bw.row_count != u.out_features ||
        bw.in_features != u.in_features) {
      throw std::invalid_argument("LoRA target geometry mismatch against "
                                  "materialized weight '" +
                                  bw.name + "'");
    }
    if ((u.in_features % 64U) != 0U) {
      throw std::invalid_argument(
          "LoRA merge mode requires projection in_features divisible by 64 (" +
          bw.name + ")");
    }
    const U32 n = u.out_features;
    const U32 k = u.in_features;
    const U32 bn = (n >= 65536U) ? 32U : 16U;
    const U32 padded_n = ((n + bn - 1U) / bn) * bn;
    const bool use_nax_tiled = nax_eligible && k >= 512U;
    const U32 dst_padded_k = ((k + 127U) / 128U) * 128U;
    const U64 code_bytes =
        use_nax_tiled ? mul(padded_n, k) : mul(n, dst_padded_k);
    const U64 sb_bytes = use_nax_tiled
                             ? mul(mul(padded_n, k / 64U), 4U)
                             : mul(mul(n, dst_padded_k / 64U), 2U);

    const U64 a_off = align_256(staging_bytes);
    const U64 a_bytes = U64(u.a_f32.size()) * sizeof(float);
    staging_bytes = a_off + a_bytes;
    const U64 b_off = align_256(staging_bytes);
    const U64 b_bytes = U64(u.b_f32.size()) * sizeof(float);
    staging_bytes = b_off + b_bytes;

    jobs.push_back(MergeJob{
        .upd = &u,
        .base_w = std::move(bw),
        .n = n,
        .k = k,
        .bn = bn,
        .padded_n = padded_n,
        .use_nax_tiled = use_nax_tiled,
        .a_off = a_off,
        .a_bytes = a_bytes,
        .b_off = b_off,
        .b_bytes = b_bytes,
        .codes_p = dst_arena.plan_alloc(code_bytes, "LoRA merged q8 codes"),
        .sb_p = dst_arena.plan_alloc(sb_bytes, "LoRA merged q8 sb/scale"),
        .sb_tiled_p =
            dst_arena.plan_alloc(sb_bytes, "LoRA merged q8 sb_tiled/bias"),
    });
  }

  const std::size_t base_seg = out->segments.size();
  dst_arena.materialize(ctx, out->segments);
  for (const auto &seg : out->segments) {
    out->buffer_bytes += seg->size();
  }

  MetalBuffer staging = ctx.make_buffer(align_256(staging_bytes));
  staging.zero();
  for (const auto &job : jobs) {
    ctx.upload_into(staging,
                    std::as_bytes(std::span<const float>(job.upd->a_f32)),
                    job.a_off);
    ctx.upload_into(staging,
                    std::as_bytes(std::span<const float>(job.upd->b_f32)),
                    job.b_off);
  }
  const Ref zero_ref{&staging, 0, 256ULL};

  static std::atomic<U64> merge_id{0};
  Encoder enc(ctx, libs, 1U, "lora.merge.q8." + std::to_string(++merge_id));
  enc.begin_stage("lora_merge_q8_g64");

  for (const auto &job : jobs) {
    const auto &bw = job.base_w;
    U32 base_kind = 0U;
    U32 base_bits = 0U;
    U32 base_gs = 64U;
    Ref p0 = zero_ref, p1 = zero_ref, p2 = zero_ref, p3 = zero_ref;

    if (bw.format == "direct_bf16_v1") {
      base_kind = 0U;
      p0 = plane(bw, "data");
    } else if (bw.format == "mxfp4_e2m1_g32_e8m0_v1") {
      base_gs = 32U;
      if (bw.layout == "mx_nax_tiled_bk128_v1") {
        base_kind = 2U;
        p0 = plane(bw, "nax_data");
        p2 = plane(bw, "nax_scale");
      } else {
        base_kind = 1U;
        p0 = plane(bw, "data");
        p2 = plane(bw, "scale");
      }
    } else if (const U32 abits = detail::forward::affine_nax_bits(bw.format);
               abits != 0U) {
      base_bits = abits;
      base_gs = detail::forward::affine_nax_group_size(bw.format);
      if (bw.layout == "oq_nax_tiled_g64_v1" ||
          bw.layout == "oq_nax_tiled_g128_v1") {
        base_kind = 4U;
        p0 = plane(bw, "nax_codes");
        if (abits == 3U) {
          p1 = plane(bw, "nax_high1");
        } else if (abits == 5U || abits == 6U) {
          p1 = plane(bw, "nax_high2");
        }
        p2 = plane(bw, "nax_sb");
      } else {
        base_kind = 3U;
        p0 = plane(bw, abits == 8U   ? "code"
                       : abits == 3U ? "low_2bit"
                                     : "base_code");
        if (abits == 3U || abits == 5U) {
          p1 = plane(bw, "high_bit");
        } else if (abits == 6U) {
          p1 = plane(bw, "high_2bit");
        }
        p2 = plane(bw, "scale");
        p3 = plane(bw, "bias");
      }
    } else {
      throw std::invalid_argument("unsupported base weight format for LoRA "
                                  "merge: " +
                                  bw.format);
    }

    const Ref dst_codes =
        NaxSegmentArena::resolve(out->segments, job.codes_p, base_seg);
    const Ref dst_sb =
        NaxSegmentArena::resolve(out->segments, job.sb_p, base_seg);
    const Ref dst_sb_tiled =
        NaxSegmentArena::resolve(out->segments, job.sb_tiled_p, base_seg);
    const Ref a_ref{&staging, job.a_off, job.a_bytes};
    const Ref b_ref{&staging, job.b_off, job.b_bytes};
    const U32 dst_padded_k = ((job.k + 127U) / 128U) * 128U;

    if (job.use_nax_tiled) {
      const U32 groups = job.k / 64U;
      auto &d = enc.emit(
          "lora", "omlx_m5_lora_merge_to_q8_g64",
          {.width = U64(groups) * 32U, .height = job.padded_n},
          {.width = 32U, .height = 1U},
          {p0, p1, p2, p3, a_ref, b_ref, dst_codes, dst_sb, dst_sb_tiled});
      enc.dynamic(d, 9,
                  words({job.n, job.k, U32(bw.padded_k), job.padded_n, job.bn,
                         base_kind, base_bits, base_gs, job.upd->rank,
                         word(job.upd->scale)}));

      MaterializedWeightView merged_view{
          .object_id = bw.object_id,
          .component = bw.component,
          .name = bw.name,
          .format = "oq_affine_q8_g64_bf16_v1",
          .layout = "oq_nax_tiled_g64_v1",
          .parent_shape = bw.parent_shape,
          .padded_k = dst_padded_k,
          .row_start = 0,
          .row_count = job.n,
          .in_features = job.k,
          .planes =
              {
                  MaterializedPlaneView{
                      .kind = "nax_codes",
                      .buffer = dst_codes.buffer,
                      .segment_index = job.codes_p.seg_index,
                      .offset = dst_codes.offset,
                      .bytes = dst_codes.bytes,
                      .row_stride = dst_codes.bytes % job.n == 0
                                        ? dst_codes.bytes / job.n
                                        : 0,
                  },
                  MaterializedPlaneView{
                      .kind = "nax_sb",
                      .buffer = dst_sb.buffer,
                      .segment_index = job.sb_p.seg_index,
                      .offset = dst_sb.offset,
                      .bytes = dst_sb.bytes,
                      .row_stride =
                          dst_sb.bytes % job.n == 0 ? dst_sb.bytes / job.n : 0,
                  },
                  MaterializedPlaneView{
                      .kind = "nax_sb_tiled",
                      .buffer = dst_sb_tiled.buffer,
                      .segment_index = job.sb_tiled_p.seg_index,
                      .offset = dst_sb_tiled.offset,
                      .bytes = dst_sb_tiled.bytes,
                      .row_stride = 0,
                  },
              },
      };
      out->views_by_object_id.emplace(bw.object_id, std::move(merged_view));
    } else {
      const U32 padded_groups = dst_padded_k / 64U;
      auto &d = enc.emit(
          "lora", "omlx_m5_lora_merge_to_q8_rowsplit_g64",
          {.width = U64(padded_groups) * 32U, .height = job.n},
          {.width = 32U, .height = 1U},
          {p0, p1, p2, p3, a_ref, b_ref, dst_codes, dst_sb, dst_sb_tiled});
      enc.dynamic(d, 9,
                  words({job.n, job.k, U32(bw.padded_k), job.padded_n, job.bn,
                         base_kind, base_bits, base_gs, job.upd->rank,
                         word(job.upd->scale)}));

      MaterializedWeightView merged_view{
          .object_id = bw.object_id,
          .component = bw.component,
          .name = bw.name,
          .format = "oq_affine_q8_g64_bf16_v1",
          .layout = "oq_rowsplit_k128_v1",
          .parent_shape = bw.parent_shape,
          .padded_k = dst_padded_k,
          .row_start = 0,
          .row_count = job.n,
          .in_features = job.k,
          .planes =
              {
                  MaterializedPlaneView{
                      .kind = "code",
                      .buffer = dst_codes.buffer,
                      .segment_index = job.codes_p.seg_index,
                      .offset = dst_codes.offset,
                      .bytes = dst_codes.bytes,
                      .row_stride = dst_padded_k,
                  },
                  MaterializedPlaneView{
                      .kind = "scale",
                      .buffer = dst_sb.buffer,
                      .segment_index = job.sb_p.seg_index,
                      .offset = dst_sb.offset,
                      .bytes = dst_sb.bytes,
                      .row_stride = U64(padded_groups) * 2ULL,
                  },
                  MaterializedPlaneView{
                      .kind = "bias",
                      .buffer = dst_sb_tiled.buffer,
                      .segment_index = job.sb_tiled_p.seg_index,
                      .offset = dst_sb_tiled.offset,
                      .bytes = dst_sb_tiled.bytes,
                      .row_stride = U64(padded_groups) * 2ULL,
                  },
              },
      };
      out->views_by_object_id.emplace(bw.object_id, std::move(merged_view));
    }
  }

  auto profile = enc.finish();
  const std::array calls{MetalCommandInvocation{.plan = &profile.plan,
                                                .bindings = profile.bindings,
                                                .parameters =
                                                    &profile.parameters}};
  (void)ctx.execute_batch(calls);
  return out;
}

inline std::shared_ptr<QwenBypassLoraRuntime> build_bypass_lora_runtime(
    const MetalContext &ctx, const MaterializedModel &weights,
    const QwenLayerCatalog &catalog, const LoadedLoraAdapter &adapter) {
  auto rt = std::make_shared<QwenBypassLoraRuntime>();
  rt->adapter_path = adapter.adapter_dir.string();
  rt->fingerprint = adapter.fingerprint;
  rt->adapted_layers = adapter.adapted_layers;
  if (adapter.updates.empty()) {
    return rt;
  }

  struct ResolvedUpdate {
    const LoraProjectionUpdate *upd = nullptr;
    U32 object_id = 0;
    U32 r_slot = 16;
    U32 n_pad = 16;
    std::optional<QwenBypassGroupKind> group_kind;
    U32 part_index = 0;
    U64 a_off = 0;
    U64 a_bytes = 0;
    U64 b_off = 0;
    U64 b_bytes = 0;
  };

  std::vector<ResolvedUpdate> items;
  items.reserve(adapter.updates.size());
  for (const auto &u : adapter.updates) {
    const U32 obj_id = resolve_target_object_id(catalog, u.target);
    const auto bw = weights.parent_view(obj_id);
    if (bw.parent_shape.size() != 2 || bw.row_count != u.out_features ||
        bw.in_features != u.in_features) {
      throw std::invalid_argument("LoRA bypass target geometry mismatch on '" +
                                  bw.name + "'");
    }
    ResolvedUpdate item{
        .upd = &u,
        .object_id = obj_id,
        .r_slot = std::max<U32>(16U, align_16(u.rank)),
        .n_pad = std::max<U32>(16U, align_16(u.out_features)),
    };
    if (!u.target.is_global) {
      const auto &role = u.target.role;
      if (role == "attention_q") {
        item.group_kind = QwenBypassGroupKind::attn_qkv;
        item.part_index = 0U;
      } else if (role == "attention_k") {
        item.group_kind = QwenBypassGroupKind::attn_qkv;
        item.part_index = 1U;
      } else if (role == "attention_v") {
        item.group_kind = QwenBypassGroupKind::attn_qkv;
        item.part_index = 2U;
      } else if (role == "gdn_qkv") {
        item.group_kind = QwenBypassGroupKind::gdn_inputs;
        item.part_index = 0U;
      } else if (role == "gdn_z") {
        item.group_kind = QwenBypassGroupKind::gdn_inputs;
        item.part_index = 1U;
      } else if (role == "gdn_a") {
        item.group_kind = QwenBypassGroupKind::gdn_inputs;
        item.part_index = 2U;
      } else if (role == "gdn_b") {
        item.group_kind = QwenBypassGroupKind::gdn_inputs;
        item.part_index = 3U;
      } else if (role == "mlp_gate") {
        item.group_kind = QwenBypassGroupKind::mlp_gate_up;
        item.part_index = 0U;
      } else if (role == "mlp_up") {
        item.group_kind = QwenBypassGroupKind::mlp_gate_up;
        item.part_index = 1U;
      } else if (role == "moe_shared_mlp_gate") {
        item.group_kind = QwenBypassGroupKind::moe_shared_gate_up;
        item.part_index = 0U;
      } else if (role == "moe_shared_mlp_up") {
        item.group_kind = QwenBypassGroupKind::moe_shared_gate_up;
        item.part_index = 1U;
      }
    }
    items.push_back(item);
  }

  // Sort so same-layer sibling group items are adjacent in part_index order,
  // enabling zero-duplicate contiguous A_concat slices!
  std::ranges::sort(items, [](const ResolvedUpdate &lhs,
                              const ResolvedUpdate &rhs) {
    if (lhs.upd->target.is_global != rhs.upd->target.is_global) {
      return !lhs.upd->target.is_global;
    }
    if (lhs.upd->target.layer_index != rhs.upd->target.layer_index) {
      return lhs.upd->target.layer_index < rhs.upd->target.layer_index;
    }
    const U32 lg = lhs.group_kind.has_value()
                       ? static_cast<U32>(*lhs.group_kind)
                       : 99U;
    const U32 rg = rhs.group_kind.has_value()
                       ? static_cast<U32>(*rhs.group_kind)
                       : 99U;
    if (lg != rg) {
      return lg < rg;
    }
    if (lhs.part_index != rhs.part_index) {
      return lhs.part_index < rhs.part_index;
    }
    return lhs.upd->target.role < rhs.upd->target.role;
  });

  // First pack A matrices (keeping sibling groups contiguous without 256B gaps inside a group!)
  U64 cursor = 0;
  std::optional<U32> active_gkey;
  for (auto &it : items) {
    std::optional<U32> cur_gkey;
    if (it.group_kind.has_value()) {
      cur_gkey = (it.upd->target.layer_index << 4U) |
                 static_cast<U32>(*it.group_kind);
    }
    if (!cur_gkey.has_value() || cur_gkey != active_gkey) {
      cursor = align_256(cursor);
      active_gkey = cur_gkey;
    }
    it.a_off = cursor;
    it.a_bytes = mul(mul(it.r_slot, it.upd->in_features), 2U);
    cursor += it.a_bytes;
    rt->max_r_slot = std::max(rt->max_r_slot, it.r_slot);
  }

  // Next pack B matrices (256B aligned)
  for (auto &it : items) {
    cursor = align_256(cursor);
    it.b_off = cursor;
    it.b_bytes = mul(mul(it.n_pad, it.r_slot), 2U);
    cursor += it.b_bytes;
  }

  const U64 total_slab_bytes = std::max<U64>(256ULL, align_256(cursor));
  std::vector<std::byte> host_slab(static_cast<std::size_t>(total_slab_bytes),
                                   std::byte{0});

  for (const auto &it : items) {
    const auto &u = *it.upd;
    const U32 r = u.rank;
    const U32 k = u.in_features;
    const U32 n = u.out_features;
    auto *a_dst = reinterpret_cast<std::uint16_t *>(host_slab.data() + it.a_off);
    for (U32 ri = 0; ri < r; ++ri) {
      for (U32 ci = 0; ci < k; ++ci) {
        a_dst[std::size_t(ri) * k + ci] =
            lora_float_to_bf16_rne(u.a_f32[std::size_t(ri) * k + ci]);
      }
    }
    auto *b_dst = reinterpret_cast<std::uint16_t *>(host_slab.data() + it.b_off);
    for (U32 ni = 0; ni < n; ++ni) {
      for (U32 ri = 0; ri < r; ++ri) {
        b_dst[std::size_t(ni) * it.r_slot + ri] =
            lora_float_to_bf16_rne(u.b_f32[std::size_t(ni) * r + ri]);
      }
    }
  }

  rt->weight_slab =
      std::make_unique<MetalBuffer>(ctx.make_buffer(total_slab_bytes));
  rt->weight_slab->write(host_slab);
  rt->weight_buffer_bytes = total_slab_bytes;

  // Populate individual entries & multi-sibling groups
  for (const auto &it : items) {
    const auto &u = *it.upd;
    const Ref a_ref{rt->weight_slab.get(), it.a_off, it.a_bytes};
    const Ref b_ref{rt->weight_slab.get(), it.b_off, it.b_bytes};
    rt->entries_by_object_id.emplace(
        it.object_id,
        QwenBypassLoraEntry{
            .object_id = it.object_id,
            .n = u.out_features,
            .k = u.in_features,
            .r_valid = u.rank,
            .r_slot = it.r_slot,
            .scale = u.scale,
            .a_ref = a_ref,
            .b_ref = b_ref,
        });
    if (it.group_kind.has_value()) {
      const U32 gkey = (u.target.layer_index << 4U) |
                       static_cast<U32>(*it.group_kind);
      auto &grp = rt->sibling_groups[gkey];
      if (grp.slices.empty()) {
        grp.k = u.in_features;
        grp.concat_a_ref = Ref{rt->weight_slab.get(), it.a_off, 0};
      }
      const U32 r_off = grp.total_r_slot;
      grp.total_r_slot += it.r_slot;
      grp.concat_a_ref.bytes += it.a_bytes;
      grp.slices.push_back(QwenBypassSiblingSlice{
          .object_id = it.object_id,
          .part_index = it.part_index,
          .n = u.out_features,
          .r_valid = u.rank,
          .r_slot = it.r_slot,
          .r_slot_offset = r_off,
          .scale = u.scale,
          .b_ref = b_ref,
      });
      rt->max_r_slot = std::max(rt->max_r_slot, grp.total_r_slot);
    }
  }
  return rt;
}

inline void emit_bypass_down(Encoder &e, Ref x, Ref a_ref, Ref tmp_ref,
                             U32 r_out, U32 k, bool bf16_nax_ok, U32 gpu_cores,
                             std::optional<U32> explicit_t = {}) {
  const U32 cap_t = explicit_t.value_or(e.profile.capacity);
  const U32 mp = ((cap_t + 7U) / 8U) * 8U;
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3U);
  if (cap_t < 16U) {
    auto &d = e.emit("lora", "omlx_m5_lora_down_gemv_bf16",
                     {.width = U64(r_out) * 128U, .height = cap_t},
                     {.width = 128U, .height = 1U}, {x, a_ref, tmp_ref});
    e.dynamic(d, 3, words({r_out, k, mp, cap_t}), tokens_patch);
    return;
  }
  if (bf16_nax_ok && r_out >= 16U && (r_out % 16U) == 0U && k >= 256U &&
      (k % 64U) == 0U) {
    (void)detail::forward::emit_bf16_nax_raw(e, a_ref, x, tmp_ref, r_out, k,
                                             explicit_t, gpu_cores);
    return;
  }
  const U32 x_tgs = (r_out + 15U) / 16U;
  const U32 y_tgs = (cap_t + 15U) / 16U;
  auto &d = e.emit("lora", "omlx_m5_lora_down_gemm_bf16",
                   {.width = U64(x_tgs) * 256U, .height = y_tgs},
                   {.width = 256U, .height = 1U}, {x, a_ref, tmp_ref});
  e.dynamic(d, 3, words({r_out, k, mp, cap_t}), tokens_patch);
}

inline void emit_bypass_up_accum(Encoder &e, Ref tmp_ref, Ref b_ref, Ref y_ref,
                                 U32 n, U32 r_valid, U32 r_stride_b,
                                 U32 tmp_stride, U32 tmp_col_offset,
                                 U32 y_stride, U32 y_col_offset, float scale,
                                 bool bf16_nax_ok,
                                 std::optional<U32> explicit_t = {}) {
  const U32 cap_t = explicit_t.value_or(e.profile.capacity);
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3U);
  if (cap_t < 16U) {
    const U32 x_tgs = (n + 127U) / 128U;
    auto &d = e.emit("lora", "omlx_m5_lora_up_accum_small_t_bf16",
                     {.width = U64(x_tgs) * 128U, .height = 1U},
                     {.width = 128U, .height = 1U}, {tmp_ref, b_ref, y_ref});
    e.dynamic(d, 3,
              words({n, r_valid, r_stride_b, cap_t, tmp_stride, tmp_col_offset,
                     y_stride, y_col_offset, word(scale)}),
              tokens_patch);
    return;
  }
  if (bf16_nax_ok && n >= 16U && r_stride_b >= 16U &&
      (r_stride_b % 16U) == 0U && (tmp_col_offset % 16U) == 0U) {
    detail::forward::emit_lora_up_accum_bf16_nax(
        e, tmp_ref, b_ref, y_ref, n, r_valid, r_stride_b, tmp_stride,
        tmp_col_offset, y_stride, y_col_offset, scale, explicit_t);
    return;
  }
  const U32 x_tgs = (n + 15U) / 16U;
  const U32 y_tgs = (cap_t + 15U) / 16U;
  auto &d = e.emit("lora", "omlx_m5_lora_up_accum_tiled_bf16",
                   {.width = U64(x_tgs) * 256U, .height = y_tgs},
                   {.width = 256U, .height = 1U}, {tmp_ref, b_ref, y_ref});
  e.dynamic(d, 3,
            words({n, r_valid, r_stride_b, cap_t, tmp_stride, tmp_col_offset,
                   y_stride, y_col_offset, word(scale)}),
            tokens_patch);
}

inline void emit_bypass_single_entry(Encoder &e,
                                     const QwenBypassLoraEntry &entry, Ref x,
                                     Ref y, Ref tmp_scratch, U32 y_stride,
                                     U32 y_col_offset, bool bf16_nax_ok,
                                     U32 gpu_cores,
                                     std::optional<U32> explicit_t = {}) {
  const U32 cap_t = explicit_t.value_or(e.profile.capacity);
  const U32 r_down = (cap_t < 16U) ? entry.r_valid : entry.r_slot;
  emit_bypass_down(e, x, entry.a_ref, tmp_scratch, r_down, entry.k, bf16_nax_ok,
                   gpu_cores, explicit_t);
  emit_bypass_up_accum(e, tmp_scratch, entry.b_ref, y, entry.n, entry.r_valid,
                       entry.r_slot, r_down, 0U, y_stride, y_col_offset,
                       entry.scale, bf16_nax_ok, explicit_t);
}

struct BypassPartDest {
  Ref y_ref{};
  U32 y_stride = 0;
  U32 y_col_offset = 0;
};

inline void emit_bypass_sibling_group(
    Encoder &e, const QwenBypassSiblingGroup &grp, Ref x,
    std::span<const BypassPartDest> dests, Ref tmp_scratch, bool bf16_nax_ok,
    U32 gpu_cores, std::optional<U32> explicit_t = {}) {
  if (grp.slices.empty()) {
    return;
  }
  if (grp.slices.size() == 1U) {
    const auto &s = grp.slices.front();
    if (s.part_index >= dests.size()) {
      throw std::out_of_range("LoRA bypass sibling part_index out of range");
    }
    const auto &dst = dests[s.part_index];
    const Ref a_single{grp.concat_a_ref.buffer, grp.concat_a_ref.offset,
                       mul(mul(s.r_slot, grp.k), 2U)};
    const QwenBypassLoraEntry single_e{
        .object_id = s.object_id,
        .n = s.n,
        .k = grp.k,
        .r_valid = s.r_valid,
        .r_slot = s.r_slot,
        .scale = s.scale,
        .a_ref = a_single,
        .b_ref = s.b_ref,
    };
    emit_bypass_single_entry(e, single_e, x, dst.y_ref, tmp_scratch,
                             dst.y_stride, dst.y_col_offset, bf16_nax_ok,
                             gpu_cores, explicit_t);
    return;
  }
  emit_bypass_down(e, x, grp.concat_a_ref, tmp_scratch, grp.total_r_slot, grp.k,
                   bf16_nax_ok, gpu_cores, explicit_t);
  for (const auto &s : grp.slices) {
    if (s.part_index >= dests.size()) {
      throw std::out_of_range("LoRA bypass sibling part_index out of range");
    }
    const auto &dst = dests[s.part_index];
    emit_bypass_up_accum(e, tmp_scratch, s.b_ref, dst.y_ref, s.n, s.r_valid,
                         s.r_slot, grp.total_r_slot, s.r_slot_offset,
                         dst.y_stride, dst.y_col_offset, s.scale, bf16_nax_ok,
                         explicit_t);
  }
}

} // namespace detail::lora_rt
} // namespace omlx::m5
