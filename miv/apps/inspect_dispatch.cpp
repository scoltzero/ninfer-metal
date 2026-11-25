#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/dispatch.h"
#include "omlx_m5/materialization.h"

namespace {

using Json = nlohmann::json;

omlx::m5::KernelCandidate
candidate(std::string id, const omlx::m5::KernelKey &key,
          omlx::m5::ActivationProfile activation, omlx::m5::KernelFamily family,
          std::uint32_t priority, std::uint32_t min_tokens,
          std::uint32_t max_tokens, omlx::m5::MetalExecutionRoute route,
          bool tail_safe = false) {
  return {
      .id = id,
      .executor_key = id + ".executor",
      .priority = priority,
      .family = family,
      .format_id = key.format_id,
      .layout_id = key.layout_id,
      .activation = activation,
      .input_type = key.input_type,
      .output_type = key.output_type,
      .accumulator_type = key.accumulator_type,
      .geometry = key.geometry,
      .output_features = key.output_features,
      .input_features = key.input_features,
      .min_tokens = min_tokens,
      .max_tokens = max_tokens,
      .experts = key.experts,
      .top_k = key.top_k,
      .fused_op = key.fused_op,
      .alignment = tail_safe ? omlx::m5::KernelAlignmentClass::tail
                             : omlx::m5::KernelAlignmentClass::full,
      .required_route = route,
      .tail_safe = tail_safe,
  };
}

omlx::m5::KernelFamily interval_family(const omlx::m5::KernelKey &key,
                                       std::uint32_t min_tokens) {
  if (key.geometry == omlx::m5::KernelGeometry::moe_expert) {
    return omlx::m5::KernelFamily::grouped_moe;
  }
  if (min_tokens == 1) {
    return omlx::m5::KernelFamily::qmv;
  }
  if (min_tokens == 9 && (key.format_id.find("_q5_") != std::string::npos ||
                          key.format_id.find("_q6_") != std::string::npos)) {
    return omlx::m5::KernelFamily::split_k;
  }
  if (min_tokens <= 9) {
    return omlx::m5::KernelFamily::row_split;
  }
  return omlx::m5::KernelFamily::cooperative;
}

std::vector<omlx::m5::KernelCandidate>
candidate_contracts(const omlx::m5::KernelKey &key, std::size_t index) {
  const auto prefix = "p" + std::to_string(index);
  std::vector<omlx::m5::KernelCandidate> result;
  const auto add_a16 = [&](std::string suffix, std::uint32_t min_tokens,
                           std::uint32_t max_tokens,
                           omlx::m5::KernelFamily family,
                           std::uint32_t priority) {
    result.push_back(candidate(prefix + ".a16." + std::move(suffix), key,
                               omlx::m5::ActivationProfile::a16, family,
                               priority, min_tokens, max_tokens,
                               omlx::m5::MetalExecutionRoute::classic_msl));
  };
  add_a16("t1", 1, 1, interval_family(key, 1), 10);
  add_a16("t2_8", 2, 8, interval_family(key, 2), 20);
  add_a16("t9_12", 9, 12, interval_family(key, 9), 20);
  add_a16("t13_128", 13, 128, interval_family(key, 13), 30);
  add_a16("t512", 512, 512, interval_family(key, 512), 30);
  add_a16("t1024", 1024, 1024, interval_family(key, 1024), 30);
  result.push_back(
      candidate(prefix + ".a16.tail", key, omlx::m5::ActivationProfile::a16,
                key.geometry == omlx::m5::KernelGeometry::moe_expert
                    ? omlx::m5::KernelFamily::per_expert_fallback
                    : omlx::m5::KernelFamily::tail_safe,
                1000, 1, std::numeric_limits<std::uint32_t>::max(),
                omlx::m5::MetalExecutionRoute::classic_msl, true));

  if (key.activation_policy == omlx::m5::ActivationPolicy::allow_a4) {
    const auto add_a4 = [&](std::string suffix, std::uint32_t min_tokens,
                            std::uint32_t max_tokens, std::uint32_t priority) {
      result.push_back(
          candidate(prefix + ".a4." + std::move(suffix), key,
                    omlx::m5::ActivationProfile::a4,
                    key.geometry == omlx::m5::KernelGeometry::moe_expert
                        ? omlx::m5::KernelFamily::grouped_moe
                        : omlx::m5::KernelFamily::tensorops,
                    priority, min_tokens, max_tokens,
                    omlx::m5::MetalExecutionRoute::native_tensorops));
    };
    add_a4("t2_8", 2, 8, 10);
    add_a4("t9_128", 9, 128, 10);
    add_a4("t512", 512, 512, 10);
    add_a4("t1024", 1024, 1024, 10);
  }
  return result;
}

using Signature =
    std::tuple<std::string, std::string, omlx::m5::ActivationPolicy,
               omlx::m5::KernelDataType, omlx::m5::KernelGeometry,
               std::uint64_t, std::uint64_t, std::uint32_t, std::uint32_t>;

Signature signature(const omlx::m5::KernelKey &key) {
  return {key.format_id,      key.layout_id, key.activation_policy,
          key.input_type,     key.geometry,  key.output_features,
          key.input_features, key.experts,   key.top_k};
}

Json key_json(const omlx::m5::KernelKey &key) {
  return {
      {"format_id", key.format_id},
      {"layout_id", key.layout_id},
      {"activation_policy", omlx::m5::to_string(key.activation_policy)},
      {"preferred_activation", omlx::m5::to_string(key.preferred_activation)},
      {"input_type", omlx::m5::to_string(key.input_type)},
      {"output_type", omlx::m5::to_string(key.output_type)},
      {"accumulator_type", omlx::m5::to_string(key.accumulator_type)},
      {"geometry", omlx::m5::to_string(key.geometry)},
      {"output_features", key.output_features},
      {"input_features", key.input_features},
      {"tokens", key.tokens},
      {"experts", key.experts},
      {"top_k", key.top_k},
      {"fused_op", omlx::m5::to_string(key.fused_op)},
      {"alignment", omlx::m5::to_string(key.alignment)},
  };
}

Json candidate_json(const omlx::m5::KernelCandidate &value) {
  return {
      {"id", value.id},
      {"executor_key", value.executor_key},
      {"priority", value.priority},
      {"family", omlx::m5::to_string(value.family)},
      {"format_id", value.format_id},
      {"layout_id", value.layout_id},
      {"activation", omlx::m5::to_string(value.activation)},
      {"input_type", omlx::m5::to_string(value.input_type)},
      {"output_type", omlx::m5::to_string(value.output_type)},
      {"accumulator_type", omlx::m5::to_string(value.accumulator_type)},
      {"geometry", omlx::m5::to_string(value.geometry)},
      {"output_features", value.output_features},
      {"input_features", value.input_features},
      {"min_tokens", value.min_tokens},
      {"max_tokens", value.max_tokens},
      {"experts", value.experts},
      {"top_k", value.top_k},
      {"fused_op", omlx::m5::to_string(value.fused_op)},
      {"alignment", omlx::m5::to_string(value.alignment)},
      {"required_route", omlx::m5::to_string(value.required_route)},
      {"tail_safe", value.tail_safe},
  };
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument("usage: omlx_m5_inspect_dispatch ARTIFACT");
    }
    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    std::set<Signature> seen;
    std::vector<omlx::m5::KernelKey> problems;
    for (const auto &binding : artifact.bindings()) {
      const auto &object = artifact.object(binding.object_id);
      const auto preferred = binding.activation_policy == "AllowA4"
                                 ? omlx::m5::ActivationProfile::a4
                                 : omlx::m5::ActivationProfile::a16;
      auto key = omlx::m5::make_kernel_key(binding, object, 1, preferred);
      if (seen.insert(signature(key)).second) {
        problems.push_back(std::move(key));
      }
    }
    std::uint64_t direct_problem_count = 0;
    for (const auto &object : artifact.objects()) {
      const auto component = omlx::m5::classify_model_component(object.name);
      if ((component != omlx::m5::ModelComponent::text &&
           component != omlx::m5::ModelComponent::mtp) ||
          object.shape.size() != 2 ||
          (object.format != "direct_bf16_v1" &&
           object.format != "direct_f16_v1")) {
        continue;
      }
      auto key = omlx::m5::make_direct_kernel_key(object, 1);
      if (seen.insert(signature(key)).second) {
        problems.push_back(std::move(key));
        ++direct_problem_count;
      }
    }
    if (problems.empty()) {
      throw std::runtime_error("artifact contains no B2.1 Linear problems");
    }

    std::vector<std::vector<omlx::m5::KernelCandidate>> contracts;
    contracts.reserve(problems.size());
    std::map<std::string, std::uint64_t, std::less<>> format_counts;
    std::map<std::string, std::uint64_t, std::less<>> geometry_counts;
    std::map<std::string, std::uint64_t, std::less<>> family_counts;
    std::map<std::string, std::uint64_t, std::less<>> route_counts;
    std::map<std::string, std::uint64_t, std::less<>> activation_counts;
    for (std::size_t index = 0; index < problems.size(); ++index) {
      ++format_counts[problems[index].format_id];
      ++geometry_counts[std::string(
          omlx::m5::to_string(problems[index].geometry))];
      contracts.push_back(candidate_contracts(problems[index], index));
      for (const auto &entry : contracts.back()) {
        omlx::m5::validate_kernel_candidate(entry);
        ++family_counts[std::string(omlx::m5::to_string(entry.family))];
        ++route_counts[std::string(omlx::m5::to_string(entry.required_route))];
        ++activation_counts[std::string(omlx::m5::to_string(entry.activation))];
      }
    }

    std::vector<std::uint32_t> tokens;
    for (std::uint32_t value = 1; value <= 128; ++value) {
      tokens.push_back(value);
    }
    constexpr std::array anchors = {129U, 255U,  511U,  512U,
                                    513U, 1023U, 1024U, 2048U};
    tokens.insert(tokens.end(), anchors.begin(), anchors.end());

    std::uint64_t key_count = 0;
    std::uint64_t preferred_coverage = 0;
    std::uint64_t activation_fallback_coverage = 0;
    std::uint64_t tail_safe_coverage = 0;
    std::uint64_t uncovered = 0;
    for (std::size_t index = 0; index < problems.size(); ++index) {
      for (const auto token_count : tokens) {
        for (const auto alignment : {omlx::m5::KernelAlignmentClass::full,
                                     omlx::m5::KernelAlignmentClass::tail}) {
          auto key = problems[index];
          key.tokens = token_count;
          key.alignment = alignment;
          if (key.activation_policy == omlx::m5::ActivationPolicy::allow_a4 &&
              token_count == 1) {
            key.preferred_activation = omlx::m5::ActivationProfile::a16;
          }
          omlx::m5::validate_kernel_key(key);
          ++key_count;
          const auto preferred =
              std::ranges::any_of(contracts[index], [&](const auto &entry) {
                return omlx::m5::candidate_covers(entry, key);
              });
          if (preferred) {
            ++preferred_coverage;
            if (alignment == omlx::m5::KernelAlignmentClass::tail ||
                token_count > 128 && token_count != 512 &&
                    token_count != 1024) {
              ++tail_safe_coverage;
            }
            continue;
          }
          bool fallback = false;
          if (key.activation_policy == omlx::m5::ActivationPolicy::allow_a4 &&
              key.preferred_activation == omlx::m5::ActivationProfile::a4) {
            auto a16_key = key;
            a16_key.preferred_activation = omlx::m5::ActivationProfile::a16;
            fallback =
                std::ranges::any_of(contracts[index], [&](const auto &entry) {
                  return omlx::m5::candidate_covers(entry, a16_key);
                });
          }
          if (fallback) {
            ++activation_fallback_coverage;
            ++tail_safe_coverage;
          } else {
            ++uncovered;
          }
        }
      }
    }
    if (uncovered != 0) {
      throw std::runtime_error("B2.1 candidate contract has uncovered keys");
    }

    omlx::m5::DispatchTable planned_table;
    for (const auto &problem_contracts : contracts) {
      for (const auto &entry : problem_contracts) {
        planned_table.register_candidate(entry);
      }
    }
    planned_table.seal({});
    auto sample_key = problems.front();
    sample_key.tokens = 1;
    sample_key.preferred_activation = omlx::m5::ActivationProfile::a16;
    omlx::m5::MetalCapabilityProbe probe;
    probe.classic_command_queue = true;
    probe.shared_event = true;
    bool planned_rejected = false;
    try {
      static_cast<void>(planned_table.prepare(
          sample_key, omlx::m5::select_metal_capability_profile(probe)));
    } catch (const std::out_of_range &) {
      planned_rejected = true;
    }
    if (!planned_rejected) {
      throw std::runtime_error("planned candidates became executable");
    }

    Json sample_candidates = Json::array();
    for (const auto &entry : contracts.front()) {
      sample_candidates.push_back(candidate_json(entry));
    }
    Json result{
        {"schema", "omlx-m5-b2-1-dispatch-contract-v1"},
        {"evidence_scope", "contract coverage; no production kernel installed"},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"problem_count", problems.size()},
        {"direct_problem_count", direct_problem_count},
        {"candidate_count", planned_table.stats().candidate_count},
        {"installed_candidate_count",
         planned_table.stats().installed_candidate_count},
        {"planned_candidate_count",
         planned_table.stats().planned_candidate_count},
        {"format_problem_counts", format_counts},
        {"geometry_problem_counts", geometry_counts},
        {"candidate_family_counts", family_counts},
        {"candidate_route_counts", route_counts},
        {"candidate_activation_counts", activation_counts},
        {"coverage",
         {
             {"key_count", key_count},
             {"preferred", preferred_coverage},
             {"activation_fallback", activation_fallback_coverage},
             {"tail_safe", tail_safe_coverage},
             {"uncovered", uncovered},
         }},
        {"planned_candidates_rejected", planned_rejected},
        {"sample_key", key_json(problems.front())},
        {"sample_candidates", std::move(sample_candidates)},
    };
    std::cout << result.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "dispatch inspection failed: " << error.what() << '\n';
    return 1;
  }
}
