#include "omlx_m5/activation.h"

#include <algorithm>
#include <limits>
#include <ranges>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace omlx::m5 {
namespace {

bool is_power_of_two(std::uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          std::string_view label) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::overflow_error(std::string(label) + " overflow");
  }
  return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               std::string_view label) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error(std::string(label) + " overflow");
  }
  return left * right;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment,
                       std::string_view label) {
  if (!is_power_of_two(alignment)) {
    throw std::invalid_argument(std::string(label) + " alignment is invalid");
  }
  const auto mask = alignment - 1;
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
    throw std::overflow_error(std::string(label) + " alignment overflow");
  }
  return (value + mask) & ~mask;
}

struct ActivationBytes {
  std::uint64_t a16 = 0;
  std::uint64_t codes = 0;
  std::uint64_t scales = 0;
};

ActivationBytes activation_bytes(std::uint32_t input_features,
                                 std::uint32_t tokens) {
  if (input_features == 0 || tokens == 0 || input_features % 32 != 0) {
    throw std::invalid_argument(
        "activation geometry requires positive K/T and K divisible by 32");
  }
  const auto elements =
      checked_multiply(input_features, tokens, "activation elements");
  return {
      .a16 = checked_multiply(elements, 2, "A16 activation bytes"),
      .codes = elements / 2,
      .scales = elements / 32,
  };
}

ActivationWorkspacePlan validated_plan(ActivationWorkspacePlan plan) {
  const auto expected = plan_activation_workspace(plan.slots, plan.alignment);
  if (plan.capacity_bytes != expected.capacity_bytes ||
      plan.a16_bytes != expected.a16_bytes ||
      plan.a4_code_bytes != expected.a4_code_bytes ||
      plan.a4_scale_bytes != expected.a4_scale_bytes ||
      plan.padding_bytes != expected.padding_bytes ||
      plan.slots != expected.slots) {
    throw std::invalid_argument("activation workspace plan was modified");
  }
  return plan;
}

ActivationPlaneView plane_view(const MutableMetalSlice &slice) {
  return {.buffer = slice.buffer, .offset = slice.offset, .bytes = slice.bytes};
}

struct ActivationReuseKeyHash {
  std::size_t operator()(const ActivationReuseKey &key) const noexcept {
    auto value = std::hash<std::uint64_t>{}(key.source_id);
    const auto mix = [&](std::uint64_t part) {
      value ^= std::hash<std::uint64_t>{}(part) + 0x9E3779B97F4A7C15ULL +
               (value << 6) + (value >> 2);
    };
    mix(key.normalization_id);
    mix(key.input_features);
    mix(key.tokens);
    return value;
  }
};

} // namespace

std::string_view to_string(ActivationPolicy policy) {
  switch (policy) {
  case ActivationPolicy::a16_only:
    return "A16Only";
  case ActivationPolicy::allow_a4:
    return "AllowA4";
  }
  throw std::invalid_argument("unknown activation policy");
}

std::string_view to_string(ActivationProfile profile) {
  switch (profile) {
  case ActivationProfile::a16:
    return "A16";
  case ActivationProfile::a4:
    return "A4";
  }
  throw std::invalid_argument("unknown activation profile");
}

ActivationPolicy parse_activation_policy(std::string_view policy) {
  if (policy == "A16Only") {
    return ActivationPolicy::a16_only;
  }
  if (policy == "AllowA4") {
    return ActivationPolicy::allow_a4;
  }
  throw std::invalid_argument("unsupported activation policy: " +
                              std::string(policy));
}

bool allows_a4(ActivationPolicy policy) {
  return policy == ActivationPolicy::allow_a4;
}

ActivationUse make_activation_use(const ArtifactBinding &binding,
                                  std::optional<std::uint32_t> expert) {
  if (binding.name.empty() || binding.use.empty()) {
    throw std::invalid_argument("activation binding is incomplete");
  }
  return {
      .binding_name = binding.name,
      .role = binding.use,
      .policy = parse_activation_policy(binding.activation_policy),
      .expert = expert,
  };
}

ActivationPolicy common_activation_policy(std::span<const ActivationUse> uses) {
  if (uses.empty()) {
    throw std::invalid_argument("activation use set is empty");
  }
  auto result = ActivationPolicy::allow_a4;
  for (const auto &use : uses) {
    if (use.binding_name.empty() || use.role.empty()) {
      throw std::invalid_argument("activation use is incomplete");
    }
    switch (use.policy) {
    case ActivationPolicy::a16_only:
      result = ActivationPolicy::a16_only;
      break;
    case ActivationPolicy::allow_a4:
      break;
    }
  }
  return result;
}

ActivationProfile resolve_activation_profile(ActivationPolicy policy,
                                             ActivationProfile preferred) {
  if (preferred == ActivationProfile::a4 && !allows_a4(policy)) {
    return ActivationProfile::a16;
  }
  return preferred;
}

ActivationWorkspacePlan
plan_activation_workspace(std::span<const ActivationSlotSpec> slots,
                          std::uint64_t alignment) {
  if (slots.empty()) {
    throw std::invalid_argument("activation workspace has no slots");
  }
  if (alignment < 256 || !is_power_of_two(alignment)) {
    throw std::invalid_argument(
        "activation workspace alignment must be a power of two at least 256");
  }
  ActivationWorkspacePlan plan;
  plan.alignment = alignment;
  plan.slots.assign(slots.begin(), slots.end());
  const auto place = [&](std::uint64_t bytes, std::uint64_t &ledger,
                         std::string_view label) {
    const auto offset = align_up(plan.capacity_bytes, alignment, label);
    plan.padding_bytes = checked_add(
        plan.padding_bytes, offset - plan.capacity_bytes, "activation padding");
    const auto end = checked_add(offset, bytes, "activation workspace");
    const auto aligned_end = align_up(end, alignment, label);
    plan.padding_bytes = checked_add(plan.padding_bytes, aligned_end - end,
                                     "activation padding");
    plan.capacity_bytes = aligned_end;
    ledger = checked_add(ledger, bytes, label);
  };
  for (const auto &slot : slots) {
    const auto bytes =
        activation_bytes(slot.max_input_features, slot.max_tokens);
    place(bytes.a16, plan.a16_bytes, "planned A16 bytes");
    if (!allows_a4(slot.policy)) {
      continue;
    }
    place(bytes.codes, plan.a4_code_bytes, "planned A4 code bytes");
    place(bytes.scales, plan.a4_scale_bytes, "planned A4 scale bytes");
  }
  return plan;
}

class ActivationWorkspace::Impl {
public:
  struct Entry {
    std::size_t slot = 0;
    ActivationReuseKey key;
    std::optional<ActivationPlaneView> a16;
    std::optional<Mxfp4ActivationView> a4;
  };

  Impl(const MetalContext &context, ActivationWorkspacePlan selected_plan)
      : plan(validated_plan(std::move(selected_plan))),
        arena(context, plan.capacity_bytes), claimed(plan.slots.size(), false) {
    counters.capacity_bytes = plan.capacity_bytes;
    counters.slot_count = plan.slots.size();
  }

  [[nodiscard]] std::size_t claim_slot(const ActivationReuseKey &key,
                                       bool need_a4) {
    std::optional<std::size_t> selected;
    std::uint64_t selected_area = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t index = 0; index < plan.slots.size(); ++index) {
      const auto &slot = plan.slots[index];
      if (claimed[index] || slot.max_input_features < key.input_features ||
          slot.max_tokens < key.tokens ||
          (need_a4 && !allows_a4(slot.policy))) {
        continue;
      }
      const auto area = checked_multiply(slot.max_input_features,
                                         slot.max_tokens, "activation slot");
      if (!selected.has_value() || area < selected_area ||
          (area == selected_area && allows_a4(plan.slots[*selected].policy) &&
           !allows_a4(slot.policy))) {
        selected = index;
        selected_area = area;
      }
    }
    if (!selected.has_value()) {
      throw std::bad_alloc();
    }
    claimed[*selected] = true;
    return *selected;
  }

  [[nodiscard]] Mxfp4ActivationView allocate_a4(const ActivationReuseKey &key) {
    const auto bytes = activation_bytes(key.input_features, key.tokens);
    const auto codes = arena.allocate(bytes.codes, plan.alignment);
    const auto scales = arena.allocate(bytes.scales, plan.alignment);
    counters.a4_code_bytes =
        checked_add(counters.a4_code_bytes, bytes.codes, "A4 code ledger");
    counters.a4_scale_bytes =
        checked_add(counters.a4_scale_bytes, bytes.scales, "A4 scale ledger");
    ++counters.a4_materializations;
    return {
        .codes = plane_view(codes),
        .scales = plane_view(scales),
        .group_size = 32,
    };
  }

  ActivationWorkspacePlan plan;
  WorkspaceArena arena;
  std::optional<std::uint64_t> epoch;
  std::vector<bool> claimed;
  std::unordered_map<ActivationReuseKey, Entry, ActivationReuseKeyHash> entries;
  ActivationWorkspaceStats counters;
};

ActivationWorkspace::ActivationWorkspace(const MetalContext &context,
                                         ActivationWorkspacePlan plan)
    : impl_(std::make_unique<Impl>(context, std::move(plan))) {}
ActivationWorkspace::ActivationWorkspace(ActivationWorkspace &&) noexcept =
    default;
ActivationWorkspace &
ActivationWorkspace::operator=(ActivationWorkspace &&) noexcept = default;
ActivationWorkspace::~ActivationWorkspace() = default;

void ActivationWorkspace::begin_epoch(std::uint64_t epoch) {
  if (epoch == 0 || (impl_->epoch.has_value() && epoch <= *impl_->epoch)) {
    throw std::invalid_argument(
        "activation epoch must be positive and strictly increasing");
  }
  impl_->arena.reset();
  impl_->entries.clear();
  std::fill(impl_->claimed.begin(), impl_->claimed.end(), false);
  impl_->epoch = epoch;
  impl_->counters.epoch = epoch;
  ++impl_->counters.epoch_count;
  impl_->counters.current_entries = 0;
}

ActivationLease ActivationWorkspace::acquire(
    const ActivationReuseKey &key, std::span<const ActivationUse> uses,
    ActivationProfile preferred, bool also_materialize_a16) {
  if (!impl_->epoch.has_value()) {
    throw std::logic_error("activation workspace has no active epoch");
  }
  static_cast<void>(activation_bytes(key.input_features, key.tokens));
  if (key.source_id == 0 || key.normalization_id == 0) {
    throw std::invalid_argument(
        "activation reuse key requires nonzero source and normalization ids");
  }
  std::set<std::pair<std::string, std::optional<std::uint32_t>>> unique_uses;
  for (const auto &use : uses) {
    if (!unique_uses.emplace(use.binding_name, use.expert).second) {
      throw std::invalid_argument("activation use is duplicated");
    }
  }
  const auto policy = common_activation_policy(uses);
  const auto profile = resolve_activation_profile(policy, preferred);

  auto found = impl_->entries.find(key);
  const bool cache_hit = found != impl_->entries.end();
  bool produce_a16 = false;
  bool produce_a4 = false;
  if (found == impl_->entries.end()) {
    const auto slot = impl_->claim_slot(key, allows_a4(policy));
    Impl::Entry entry{
        .slot = slot,
        .key = key,
    };
    found = impl_->entries.emplace(key, std::move(entry)).first;
    impl_->counters.current_entries = impl_->entries.size();
  }

  const auto need_a16 =
      profile == ActivationProfile::a16 || also_materialize_a16;
  if (need_a16 && !found->second.a16.has_value()) {
    const auto bytes = activation_bytes(key.input_features, key.tokens);
    const auto a16 = impl_->arena.allocate(bytes.a16, impl_->plan.alignment);
    found->second.a16 = plane_view(a16);
    impl_->counters.a16_bytes = checked_add(impl_->counters.a16_bytes,
                                            bytes.a16, "A16 activation ledger");
    ++impl_->counters.a16_materializations;
    produce_a16 = true;
  }

  if (profile == ActivationProfile::a4 && !found->second.a4.has_value()) {
    if (!allows_a4(impl_->plan.slots.at(found->second.slot).policy)) {
      throw std::logic_error(
          "activation slot cannot be upgraded to an A4 representation");
    }
    found->second.a4 = impl_->allocate_a4(key);
    produce_a4 = true;
  }

  const auto use_count = static_cast<std::uint64_t>(uses.size());
  ++impl_->counters.acquire_calls;
  impl_->counters.consumer_uses = checked_add(
      impl_->counters.consumer_uses, use_count, "activation consumers");
  if (cache_hit) {
    ++impl_->counters.cache_hits;
  }
  if (preferred == ActivationProfile::a4 && profile == ActivationProfile::a16) {
    ++impl_->counters.policy_fallbacks;
  }
  if (need_a16) {
    impl_->counters.avoided_a16_materializations =
        checked_add(impl_->counters.avoided_a16_materializations,
                    produce_a16 ? use_count - 1 : use_count,
                    "avoided A16 materializations");
  }
  if (profile == ActivationProfile::a4) {
    impl_->counters.avoided_a4_quantizations = checked_add(
        impl_->counters.avoided_a4_quantizations,
        produce_a4 ? use_count - 1 : use_count, "avoided A4 quantizations");
  }
  const auto expert_count = static_cast<std::uint64_t>(std::ranges::count_if(
      uses, [](const ActivationUse &use) { return use.expert.has_value(); }));
  if (expert_count > 1) {
    impl_->counters.cross_expert_reuses =
        checked_add(impl_->counters.cross_expert_reuses, expert_count - 1,
                    "cross-expert activation reuses");
  }
  const auto projection_count = use_count - expert_count;
  if (projection_count > 1) {
    impl_->counters.cross_projection_reuses =
        checked_add(impl_->counters.cross_projection_reuses,
                    projection_count - 1, "cross-projection activation reuses");
  }

  ActivationBundle bundle{
      .epoch = *impl_->epoch,
      .key = key,
      .common_policy = policy,
      .profile = profile,
      .normalized_a16 = found->second.a16,
  };
  if (profile == ActivationProfile::a4) {
    bundle.a4 = found->second.a4;
  }
  return {
      .bundle = std::move(bundle),
      .produce_a16 = produce_a16,
      .produce_a4 = produce_a4,
  };
}

const ActivationWorkspacePlan &ActivationWorkspace::plan() const {
  return impl_->plan;
}

ActivationWorkspaceStats ActivationWorkspace::stats() const {
  auto result = impl_->counters;
  const auto arena_stats = impl_->arena.stats();
  result.used_bytes = arena_stats.used_bytes;
  result.peak_bytes = arena_stats.peak_bytes;
  return result;
}

MetalBuffer &ActivationWorkspace::buffer() { return impl_->arena.buffer(); }

} // namespace omlx::m5
