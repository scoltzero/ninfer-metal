#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/activation.h"
#include "omlx_m5/artifact.h"
#include "omlx_m5/metal.h"

namespace {

using Json = nlohmann::json;

std::uint32_t parse_u32(std::string_view value, std::string_view label) {
  std::size_t parsed = 0;
  const auto result = std::stoull(std::string(value), &parsed, 0);
  if (parsed != value.size() ||
      result > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(std::string(label) + " is invalid");
  }
  return static_cast<std::uint32_t>(result);
}

std::string parent_name(std::string_view name) {
  const auto separator = name.rfind('.');
  return separator == std::string_view::npos
             ? std::string(name)
             : std::string(name.substr(0, separator));
}

struct Scenario {
  std::string name;
  std::uint32_t input_features = 0;
  std::vector<omlx::m5::ActivationUse> uses;
};

std::vector<Scenario> build_scenarios(const omlx::m5::ArtifactModel &artifact) {
  std::map<std::string, std::vector<const omlx::m5::ArtifactBinding *>,
           std::less<>>
      groups;
  for (const auto &binding : artifact.bindings()) {
    groups[parent_name(binding.name)].push_back(&binding);
  }
  std::vector<Scenario> result;
  const auto add_projection = [&](std::string_view scenario_name,
                                  std::span<const std::string_view> roles,
                                  bool require_dense) {
    if (std::ranges::any_of(result, [&](const Scenario &scenario) {
          return scenario.name == scenario_name;
        })) {
      return;
    }
    for (const auto &[_, bindings] : groups) {
      std::vector<const omlx::m5::ArtifactBinding *> selected;
      for (const auto role : roles) {
        const auto found = std::ranges::find_if(
            bindings, [&](const omlx::m5::ArtifactBinding *binding) {
              return binding->use == role &&
                     (!require_dense || binding->leading_shape.empty());
            });
        if (found == bindings.end()) {
          selected.clear();
          break;
        }
        selected.push_back(*found);
      }
      if (selected.empty() ||
          std::ranges::any_of(selected, [&](const auto *binding) {
            return binding->in_features != selected.front()->in_features;
          })) {
        continue;
      }
      Scenario scenario{
          .name = std::string(scenario_name),
          .input_features =
              static_cast<std::uint32_t>(selected.front()->in_features),
      };
      for (const auto *binding : selected) {
        scenario.uses.push_back(omlx::m5::make_activation_use(*binding));
      }
      result.push_back(std::move(scenario));
      return;
    }
  };

  static constexpr std::array<std::string_view, 3> attention_roles = {
      "attention_q", "attention_k", "attention_v"};
  static constexpr std::array<std::string_view, 2> dense_roles = {"mlp_gate",
                                                                  "mlp_up"};
  add_projection("attention_qkv", attention_roles, true);
  add_projection("dense_gate_up", dense_roles, true);

  for (const auto &[_, bindings] : groups) {
    const auto gate = std::ranges::find_if(
        bindings, [](const omlx::m5::ArtifactBinding *binding) {
          return binding->use == "moe_expert_gate" &&
                 binding->leading_shape.size() == 1;
        });
    const auto up = std::ranges::find_if(
        bindings, [](const omlx::m5::ArtifactBinding *binding) {
          return binding->use == "moe_expert_up" &&
                 binding->leading_shape.size() == 1;
        });
    if (gate == bindings.end() || up == bindings.end() ||
        (*gate)->in_features != (*up)->in_features ||
        (*gate)->leading_shape != (*up)->leading_shape) {
      continue;
    }
    Scenario scenario{
        .name = "moe_top8_gate_up",
        .input_features = static_cast<std::uint32_t>((*gate)->in_features),
    };
    const auto experts =
        std::min<std::uint64_t>(8, (*gate)->leading_shape.front());
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
      scenario.uses.push_back(omlx::m5::make_activation_use(**gate, expert));
      scenario.uses.push_back(omlx::m5::make_activation_use(**up, expert));
    }
    result.push_back(std::move(scenario));
    break;
  }

  if (result.empty() && !artifact.bindings().empty()) {
    const auto &binding = artifact.bindings().front();
    result.push_back({
        .name = "single_binding",
        .input_features = static_cast<std::uint32_t>(binding.in_features),
        .uses = {omlx::m5::make_activation_use(binding)},
    });
  }
  return result;
}

Json stats_json(const omlx::m5::ActivationWorkspaceStats &stats) {
  return {
      {"capacity_bytes", stats.capacity_bytes},
      {"used_bytes", stats.used_bytes},
      {"peak_bytes", stats.peak_bytes},
      {"epoch", stats.epoch},
      {"epoch_count", stats.epoch_count},
      {"slot_count", stats.slot_count},
      {"current_entries", stats.current_entries},
      {"acquire_calls", stats.acquire_calls},
      {"consumer_uses", stats.consumer_uses},
      {"cache_hits", stats.cache_hits},
      {"policy_fallbacks", stats.policy_fallbacks},
      {"a16_materializations", stats.a16_materializations},
      {"a4_materializations", stats.a4_materializations},
      {"avoided_a16_materializations", stats.avoided_a16_materializations},
      {"avoided_a4_quantizations", stats.avoided_a4_quantizations},
      {"cross_projection_reuses", stats.cross_projection_reuses},
      {"cross_expert_reuses", stats.cross_expert_reuses},
      {"a16_bytes", stats.a16_bytes},
      {"a4_code_bytes", stats.a4_code_bytes},
      {"a4_scale_bytes", stats.a4_scale_bytes},
  };
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_activation ARTIFACT [--max-tokens N]");
    }
    std::uint32_t max_tokens = 8;
    for (int index = 2; index < argc; ++index) {
      const std::string_view argument(argv[index]);
      if (argument == "--max-tokens" && index + 1 < argc) {
        max_tokens = parse_u32(argv[++index], "max tokens");
      } else {
        throw std::invalid_argument("unknown activation argument: " +
                                    std::string(argument));
      }
    }
    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    std::map<std::string, std::uint64_t, std::less<>> policy_counts;
    for (const auto &binding : artifact.bindings()) {
      ++policy_counts[binding.activation_policy];
    }
    const auto scenarios = build_scenarios(artifact);
    if (scenarios.empty()) {
      throw std::runtime_error("artifact has no activation scenarios");
    }

    std::vector<omlx::m5::ActivationSlotSpec> slots;
    slots.reserve(scenarios.size());
    for (const auto &scenario : scenarios) {
      slots.push_back({
          .max_input_features = scenario.input_features,
          .max_tokens = max_tokens,
          .policy = omlx::m5::common_activation_policy(scenario.uses),
      });
    }
    const auto plan = omlx::m5::plan_activation_workspace(slots);
    auto context = omlx::m5::MetalContext::create();
    const auto before = context.registry_stats();
    omlx::m5::ActivationWorkspace workspace(context, plan);
    workspace.begin_epoch(1);
    Json scenario_records = Json::array();
    bool resources_verified = true;
    std::uint64_t source_id = 1;
    for (const auto &scenario : scenarios) {
      const omlx::m5::ActivationReuseKey key{
          .source_id = source_id,
          .normalization_id = 1000 + source_id,
          .input_features = scenario.input_features,
          .tokens = max_tokens,
      };
      const auto first = workspace.acquire(key, scenario.uses,
                                           omlx::m5::ActivationProfile::a4);
      const auto second = workspace.acquire(key, scenario.uses,
                                            omlx::m5::ActivationProfile::a4);
      const auto expected_profile = omlx::m5::resolve_activation_profile(
          omlx::m5::common_activation_policy(scenario.uses),
          omlx::m5::ActivationProfile::a4);
      resources_verified = resources_verified && !second.produce_a16 &&
                           !second.produce_a4 &&
                           first.bundle.profile == expected_profile;
      if (expected_profile == omlx::m5::ActivationProfile::a4) {
        resources_verified =
            resources_verified && !first.produce_a16 && first.produce_a4 &&
            !first.bundle.normalized_a16.has_value() &&
            !second.bundle.normalized_a16.has_value() &&
            first.bundle.a4.has_value() && second.bundle.a4.has_value() &&
            first.bundle.a4->codes.offset == second.bundle.a4->codes.offset &&
            first.bundle.a4->scales.offset == second.bundle.a4->scales.offset;
      } else {
        resources_verified = resources_verified && first.produce_a16 &&
                             !first.produce_a4 &&
                             first.bundle.normalized_a16.has_value() &&
                             second.bundle.normalized_a16.has_value() &&
                             first.bundle.normalized_a16->offset ==
                                 second.bundle.normalized_a16->offset &&
                             !first.bundle.a4.has_value();
      }
      const std::array<std::byte, 1> marker = {std::byte{0x5A}};
      const auto destination =
          expected_profile == omlx::m5::ActivationProfile::a4
              ? first.bundle.a4->codes
              : *first.bundle.normalized_a16;
      destination.buffer->write(marker, destination.offset);
      resources_verified =
          resources_verified &&
          destination.buffer->read(destination.offset, marker.size()) ==
              std::vector<std::byte>(marker.begin(), marker.end());
      scenario_records.push_back({
          {"name", scenario.name},
          {"input_features", scenario.input_features},
          {"tokens", max_tokens},
          {"consumer_count", scenario.uses.size()},
          {"common_policy",
           omlx::m5::to_string(
               omlx::m5::common_activation_policy(scenario.uses))},
          {"resolved_profile", omlx::m5::to_string(expected_profile)},
          {"a16_bytes", first.bundle.normalized_a16.has_value()
                            ? first.bundle.normalized_a16->bytes
                            : 0},
          {"a4_code_bytes",
           first.bundle.a4.has_value() ? first.bundle.a4->codes.bytes : 0},
          {"a4_scale_bytes",
           first.bundle.a4.has_value() ? first.bundle.a4->scales.bytes : 0},
          {"cache_reused", !second.produce_a16 && !second.produce_a4},
      });
      ++source_id;
    }
    const auto after = context.registry_stats();
    const auto stats = workspace.stats();
    const bool ledger_verified =
        after.current_buffer_count == before.current_buffer_count + 1 &&
        after.current_buffer_bytes ==
            before.current_buffer_bytes + plan.capacity_bytes &&
        after.upload_count == before.upload_count &&
        after.uploaded_bytes == before.uploaded_bytes;

    Json policies = Json::object();
    for (const auto &[name, count] : policy_counts) {
      policies[name] = count;
    }
    Json result{
        {"schema", "omlx-m5-b1-3-activation-v1"},
        {"evidence_scope", "M4 resource correctness; not M5 Ultra performance"},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"binding_count", artifact.bindings().size()},
        {"activation_policy_counts", std::move(policies)},
        {"workspace_plan",
         {
             {"alignment", plan.alignment},
             {"capacity_bytes", plan.capacity_bytes},
             {"a16_bytes", plan.a16_bytes},
             {"a4_code_bytes", plan.a4_code_bytes},
             {"a4_scale_bytes", plan.a4_scale_bytes},
             {"padding_bytes", plan.padding_bytes},
             {"slot_count", plan.slots.size()},
         }},
        {"scenarios", std::move(scenario_records)},
        {"stats", stats_json(stats)},
        {"resources_verified", resources_verified},
        {"ledger_verified", ledger_verified},
    };
    std::cout << result.dump(2) << '\n';
    return resources_verified && ledger_verified ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "activation inspection failed: " << error.what() << '\n';
    return 1;
  }
}
