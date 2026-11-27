#include "omlx_m5/dispatch.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace omlx::m5 {
namespace {

constexpr std::array<std::string_view, 13> kFormats = {
    "oq_affine_q3_g64_bf16_v1",
    "oq_affine_q3_g128_bf16_v1",
    "oq_affine_q4_g64_bf16_v1",
    "oq_affine_q4_g128_bf16_v1",
    "oq_affine_q5_g64_bf16_v1",
    "oq_affine_q5_g128_bf16_v1",
    "oq_affine_q6_g64_bf16_v1",
    "oq_affine_q6_g128_bf16_v1",
    "oq_affine_q8_g64_bf16_v1",
    "oq_affine_q8_g128_bf16_v1",
    "mxfp4_e2m1_g32_e8m0_v1",
    "direct_bf16_v1",
    "direct_f16_v1",
};

bool valid_format(std::string_view format) {
  return std::ranges::find(kFormats, format) != kFormats.end();
}

bool mxfp4(std::string_view format) {
  return format == "mxfp4_e2m1_g32_e8m0_v1";
}

bool direct(std::string_view format) {
  return format == "direct_bf16_v1" || format == "direct_f16_v1";
}

bool expert_layout(std::string_view layout) {
  return layout == "oq_expert_rowsplit_k128_v1" ||
         layout == "mx_expert_rowsplit_k128_v1";
}

bool route_eligible(MetalExecutionRoute route,
                    const MetalCapabilityProfile &capabilities) {
  switch (route) {
  case MetalExecutionRoute::classic_msl:
    return capabilities.classic_msl_eligible;
  case MetalExecutionRoute::mlx_bridge:
    return capabilities.mlx_bridge_eligible;
  case MetalExecutionRoute::native_tensorops:
    return capabilities.native_tensorops_eligible;
  case MetalExecutionRoute::unavailable:
    return false;
  }
  return false;
}

void validate_geometry(KernelGeometry geometry, std::uint32_t experts,
                       std::uint32_t top_k) {
  if (geometry == KernelGeometry::dense) {
    if (experts != 0 || top_k != 0) {
      throw std::invalid_argument(
          "dense kernel geometry cannot carry expert metadata");
    }
    return;
  }
  if (experts == 0 || top_k == 0 || top_k > experts) {
    throw std::invalid_argument("MoE kernel geometry is invalid");
  }
}

void validate_layout(std::string_view format, std::string_view layout,
                     KernelGeometry geometry) {
  if (direct(format)) {
    if (layout != "contiguous_le_v1" ||
        geometry == KernelGeometry::moe_expert) {
      throw std::invalid_argument("direct kernel layout is invalid");
    }
    return;
  }
  if (geometry == KernelGeometry::moe_expert) {
    const std::string_view expected = mxfp4(format)
                                          ? "mx_expert_rowsplit_k128_v1"
                                          : "oq_expert_rowsplit_k128_v1";
    if (layout != expected) {
      throw std::invalid_argument("MoE expert kernel requires expert layout");
    }
  } else if (expert_layout(layout)) {
    throw std::invalid_argument("non-expert kernel cannot use expert layout");
  } else if (mxfp4(format)) {
    if (layout != "mx_rowsplit_k128_v1" &&
        layout != "mx_nax_tiled_bk128_v1") {
      throw std::invalid_argument("kernel layout is unsupported");
    }
  } else if (layout != "oq_rowsplit_k128_v1" &&
             layout != "oq_nax_tiled_g64_v1" &&
             layout != "oq_nax_tiled_g128_v1" &&
             layout != "oq_nax_lane_g64_v2" &&
             layout != "oq_nax_lane_g128_v2") {
    throw std::invalid_argument("kernel layout is unsupported");
  }
}

bool same_selector(const KernelCandidate &candidate, const KernelKey &key) {
  return candidate.format_id == key.format_id &&
         candidate.layout_id == key.layout_id &&
         candidate.input_type == key.input_type &&
         candidate.output_type == key.output_type &&
         candidate.accumulator_type == key.accumulator_type &&
         candidate.geometry == key.geometry &&
         candidate.output_features == key.output_features &&
         candidate.input_features == key.input_features &&
         candidate.experts == key.experts && candidate.top_k == key.top_k &&
         candidate.fused_op == key.fused_op;
}

} // namespace

std::string_view to_string(KernelGeometry value) {
  switch (value) {
  case KernelGeometry::dense:
    return "dense";
  case KernelGeometry::moe_expert:
    return "moe_expert";
  case KernelGeometry::moe_router:
    return "moe_router";
  }
  throw std::invalid_argument("unknown kernel geometry");
}

std::string_view to_string(KernelDataType value) {
  switch (value) {
  case KernelDataType::bf16:
    return "bf16";
  case KernelDataType::fp16:
    return "fp16";
  case KernelDataType::fp32:
    return "fp32";
  }
  throw std::invalid_argument("unknown kernel data type");
}

std::string_view to_string(KernelAlignmentClass value) {
  switch (value) {
  case KernelAlignmentClass::full:
    return "full";
  case KernelAlignmentClass::tail:
    return "tail";
  }
  throw std::invalid_argument("unknown kernel alignment class");
}

std::string_view to_string(KernelFusedOp value) {
  switch (value) {
  case KernelFusedOp::linear:
    return "linear";
  case KernelFusedOp::attention_qkv:
    return "attention_qkv";
  case KernelFusedOp::gdn_qkvz:
    return "gdn_qkvz";
  case KernelFusedOp::dense_gate_up:
    return "dense_gate_up";
  case KernelFusedOp::moe_gate_up:
    return "moe_gate_up";
  case KernelFusedOp::moe_down:
    return "moe_down";
  case KernelFusedOp::split_output:
    return "split_output";
  }
  throw std::invalid_argument("unknown fused operation");
}

std::string_view to_string(KernelFamily value) {
  switch (value) {
  case KernelFamily::qmv:
    return "qmv";
  case KernelFamily::row_split:
    return "row_split";
  case KernelFamily::split_k:
    return "split_k";
  case KernelFamily::cooperative:
    return "cooperative";
  case KernelFamily::tensorops:
    return "tensorops";
  case KernelFamily::grouped_moe:
    return "grouped_moe";
  case KernelFamily::per_expert_fallback:
    return "per_expert_fallback";
  case KernelFamily::tail_safe:
    return "tail_safe";
  }
  throw std::invalid_argument("unknown kernel family");
}

KernelKey make_kernel_key(const ArtifactBinding &binding,
                          const ArtifactObject &object, std::uint32_t tokens,
                          ActivationProfile preferred_activation,
                          KernelFusedOp fused_op,
                          KernelAlignmentClass alignment,
                          std::uint32_t moe_top_k) {
  if (binding.object_id != object.id || binding.name != object.name ||
      binding.format != object.format || object.shape.size() < 2 ||
      binding.out_features != object.shape[object.shape.size() - 2] ||
      binding.in_features != object.shape.back()) {
    throw std::invalid_argument(
        "kernel key binding and artifact object disagree");
  }
  KernelGeometry geometry = KernelGeometry::dense;
  std::uint32_t experts = 0;
  std::uint32_t top_k = 0;
  if (!binding.leading_shape.empty()) {
    if (binding.leading_shape.size() != 1 ||
        binding.leading_shape.front() >
            std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("kernel key expert shape is invalid");
    }
    geometry = KernelGeometry::moe_expert;
    experts = static_cast<std::uint32_t>(binding.leading_shape.front());
    top_k = moe_top_k;
  } else if (binding.use == "moe_router") {
    if (binding.out_features > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("MoE router expert count overflows");
    }
    geometry = KernelGeometry::moe_router;
    experts = static_cast<std::uint32_t>(binding.out_features);
    top_k = moe_top_k;
  }
  KernelKey key{
      .format_id = binding.format,
      .layout_id = object.layout,
      .activation_policy = parse_activation_policy(binding.activation_policy),
      .preferred_activation = preferred_activation,
      .input_type = KernelDataType::bf16,
      .output_type = KernelDataType::bf16,
      .accumulator_type = KernelDataType::fp32,
      .geometry = geometry,
      .output_features = binding.out_features,
      .input_features = binding.in_features,
      .tokens = tokens,
      .experts = experts,
      .top_k = top_k,
      .fused_op = fused_op,
      .alignment = alignment,
  };
  validate_kernel_key(key);
  return key;
}

KernelKey make_direct_kernel_key(const ArtifactObject &object,
                                 std::uint32_t tokens, KernelFusedOp fused_op,
                                 KernelAlignmentClass alignment,
                                 KernelGeometry geometry, std::uint32_t experts,
                                 std::uint32_t top_k) {
  if (!direct(object.format) || object.shape.size() != 2) {
    throw std::invalid_argument(
        "direct kernel key requires a rank-2 direct object");
  }
  KernelKey key{
      .format_id = object.format,
      .layout_id = object.layout,
      .activation_policy = ActivationPolicy::a16_only,
      .preferred_activation = ActivationProfile::a16,
      .input_type = object.format == "direct_f16_v1" ? KernelDataType::fp16
                                                     : KernelDataType::bf16,
      .output_type = KernelDataType::bf16,
      .accumulator_type = KernelDataType::fp32,
      .geometry = geometry,
      .output_features = object.shape[0],
      .input_features = object.shape[1],
      .tokens = tokens,
      .experts = experts,
      .top_k = top_k,
      .fused_op = fused_op,
      .alignment = alignment,
  };
  validate_kernel_key(key);
  return key;
}

void validate_kernel_key(const KernelKey &key) {
  if (!valid_format(key.format_id) || key.layout_id.empty() ||
      key.output_features == 0 || key.input_features == 0 || key.tokens == 0) {
    throw std::invalid_argument("kernel key identity or geometry is invalid");
  }
  validate_geometry(key.geometry, key.experts, key.top_k);
  validate_layout(key.format_id, key.layout_id, key.geometry);
  if (key.input_type == KernelDataType::fp32 ||
      key.output_type == KernelDataType::fp32 ||
      key.accumulator_type != KernelDataType::fp32) {
    throw std::invalid_argument("kernel data type contract is unsupported");
  }
  if (key.preferred_activation == ActivationProfile::a4 &&
      (key.activation_policy != ActivationPolicy::allow_a4 ||
       !mxfp4(key.format_id))) {
    throw std::invalid_argument("A4 kernel key requires AllowA4 MXFP4 weights");
  }
  if (!mxfp4(key.format_id) &&
      key.activation_policy != ActivationPolicy::a16_only) {
    throw std::invalid_argument("non-MXFP4 kernel key must use A16Only policy");
  }
}

void validate_kernel_candidate(const KernelCandidate &candidate) {
  if (candidate.id.empty() || candidate.executor_key.empty() ||
      !valid_format(candidate.format_id) || candidate.layout_id.empty() ||
      candidate.output_features == 0 || candidate.input_features == 0 ||
      candidate.min_tokens == 0 ||
      candidate.max_tokens < candidate.min_tokens ||
      candidate.required_route == MetalExecutionRoute::unavailable) {
    throw std::invalid_argument(
        "kernel candidate identity or domain is invalid");
  }
  validate_geometry(candidate.geometry, candidate.experts, candidate.top_k);
  validate_layout(candidate.format_id, candidate.layout_id, candidate.geometry);
  if (candidate.input_type == KernelDataType::fp32 ||
      candidate.output_type == KernelDataType::fp32 ||
      candidate.accumulator_type != KernelDataType::fp32) {
    throw std::invalid_argument("kernel candidate data types are unsupported");
  }
  if (candidate.activation == ActivationProfile::a4 &&
      !mxfp4(candidate.format_id)) {
    throw std::invalid_argument("A4 candidate requires MXFP4 weights");
  }
  if (candidate.tail_safe !=
      (candidate.alignment == KernelAlignmentClass::tail)) {
    throw std::invalid_argument(
        "tail-safe candidate and alignment class disagree");
  }
}

bool candidate_covers(const KernelCandidate &candidate, const KernelKey &key) {
  validate_kernel_candidate(candidate);
  validate_kernel_key(key);
  if (!same_selector(candidate, key) ||
      candidate.activation != key.preferred_activation ||
      key.tokens < candidate.min_tokens || key.tokens > candidate.max_tokens) {
    return false;
  }
  if (key.alignment == KernelAlignmentClass::tail) {
    return candidate.tail_safe;
  }
  return candidate.alignment == KernelAlignmentClass::full ||
         candidate.tail_safe;
}

bool candidate_route_eligible(const KernelCandidate &candidate,
                              const MetalCapabilityProfile &capabilities) {
  validate_kernel_candidate(candidate);
  return route_eligible(candidate.required_route, capabilities);
}

class DispatchTable::Impl {
public:
  std::vector<KernelCandidate> candidates;
  std::unordered_map<std::string, std::size_t> candidate_ids;
  std::unordered_set<std::string> installed_executors;
  bool sealed = false;
};

DispatchTable::DispatchTable() : impl_(std::make_unique<Impl>()) {}
DispatchTable::DispatchTable(DispatchTable &&) noexcept = default;
DispatchTable &DispatchTable::operator=(DispatchTable &&) noexcept = default;
DispatchTable::~DispatchTable() = default;

void DispatchTable::register_candidate(KernelCandidate candidate) {
  if (impl_->sealed) {
    throw std::logic_error("dispatch table is sealed");
  }
  validate_kernel_candidate(candidate);
  if (!impl_->candidate_ids.emplace(candidate.id, impl_->candidates.size())
           .second) {
    throw std::invalid_argument("kernel candidate id is duplicated");
  }
  impl_->candidates.push_back(std::move(candidate));
}

void DispatchTable::seal(std::span<const std::string> installed_executor_keys) {
  if (impl_->sealed) {
    throw std::logic_error("dispatch table is already sealed");
  }
  std::unordered_set<std::string> known_executors;
  for (const auto &candidate : impl_->candidates) {
    known_executors.insert(candidate.executor_key);
  }
  for (const auto &executor : installed_executor_keys) {
    if (executor.empty() || !known_executors.contains(executor) ||
        !impl_->installed_executors.insert(executor).second) {
      throw std::invalid_argument(
          "installed kernel executor is unknown, empty, or duplicated");
    }
  }
  impl_->sealed = true;
}

bool DispatchTable::sealed() const { return impl_->sealed; }

DispatchTableStats DispatchTable::stats() const {
  DispatchTableStats result{
      .candidate_count = impl_->candidates.size(),
      .sealed = impl_->sealed,
  };
  result.installed_candidate_count =
      static_cast<std::uint64_t>(std::ranges::count_if(
          impl_->candidates, [&](const KernelCandidate &candidate) {
            return impl_->installed_executors.contains(candidate.executor_key);
          }));
  result.planned_candidate_count =
      result.candidate_count - result.installed_candidate_count;
  return result;
}

KernelDispatchPlan
DispatchTable::prepare(const KernelKey &key,
                       const MetalCapabilityProfile &capabilities) const {
  if (!impl_->sealed) {
    throw std::logic_error("dispatch table is not sealed");
  }
  validate_kernel_key(key);
  const auto select =
      [&](ActivationProfile profile) -> std::optional<KernelCandidate> {
    auto query = key;
    query.preferred_activation = profile;
    std::optional<KernelCandidate> selected;
    for (const auto &candidate : impl_->candidates) {
      if (!impl_->installed_executors.contains(candidate.executor_key) ||
          !candidate_covers(candidate, query) ||
          !candidate_route_eligible(candidate, capabilities)) {
        continue;
      }
      if (!selected.has_value() || candidate.priority < selected->priority) {
        selected = candidate;
      } else if (candidate.priority == selected->priority) {
        throw std::logic_error(
            "dispatch table has ambiguous installed candidates");
      }
    }
    return selected;
  };

  auto selected = select(key.preferred_activation);
  auto resolved = key.preferred_activation;
  std::vector<std::string> fallback_reasons;
  if (!selected.has_value() &&
      key.preferred_activation == ActivationProfile::a4 &&
      key.activation_policy == ActivationPolicy::allow_a4) {
    selected = select(ActivationProfile::a16);
    resolved = ActivationProfile::a16;
    if (selected.has_value()) {
      fallback_reasons.emplace_back("a4_executor_unavailable");
    }
  }
  if (!selected.has_value()) {
    throw std::out_of_range(
        "no installed kernel candidate covers the requested key");
  }
  if (key.alignment == KernelAlignmentClass::full && selected->tail_safe) {
    fallback_reasons.emplace_back("tail_safe_route");
  }
  if (selected->required_route != capabilities.selected_route) {
    fallback_reasons.emplace_back("capability_route_fallback");
  }
  return {
      .requested = key,
      .candidate = std::move(*selected),
      .resolved_activation = resolved,
      .fallback_reasons = std::move(fallback_reasons),
  };
}

} // namespace omlx::m5
