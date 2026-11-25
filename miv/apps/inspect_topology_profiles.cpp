#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/topology_profile.h"

namespace {

nlohmann::json selection_json(const omlx::m5::TopologyProfileCatalog &catalog,
                              omlx::m5::TopologyProfileMode mode,
                              std::uint32_t batch, std::uint32_t tokens) {
  const auto selected =
      catalog.select({.mode = mode,
                      .architecture = catalog.topology().architecture,
                      .quant_profile = catalog.topology().quant_profile,
                      .batch_size = batch,
                      .tokens = tokens,
                      .topology_signature = catalog.topology().signature});
  return {{"mode", omlx::m5::to_string(mode)},
          {"batch", batch},
          {"tokens", tokens},
          {"matched", selected.matched()},
          {"profile_index", selected.matched()
                                ? nlohmann::json(selected.profile_index)
                                : nlohmann::json(nullptr)},
          {"reason", omlx::m5::to_string(selected.reason)},
          {"execution_route", selected.execution_route}};
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_topology_profiles ARTIFACT CONFIG "
          "MAX_CONTEXT");
    }
    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    const auto config = omlx::m5::load_qwen_model_config(argv[2]);
    const auto max_context = static_cast<std::uint32_t>(std::stoul(argv[3]));
    const auto layers =
        omlx::m5::build_qwen_layer_catalog(artifact, config, max_context, 64);
    const auto topology = omlx::m5::describe_qwen_topology(
        artifact, config, layers, max_context, 64);
    const auto catalog = omlx::m5::compile_topology_profiles(topology);

    nlohmann::json profiles = nlohmann::json::array();
    for (const auto &profile : catalog.profiles()) {
      nlohmann::json fields = nlohmann::json::array();
      for (const auto &field : profile.parameters.fields) {
        fields.push_back({{"key", field.key},
                          {"offset", field.offset},
                          {"size", field.size},
                          {"alignment", field.alignment}});
      }
      nlohmann::json ranges = nlohmann::json::array();
      for (const auto &range : profile.ranges) {
        ranges.push_back({{"key", range.key},
                          {"location", range.location},
                          {"length", range.length}});
      }
      profiles.push_back(
          {{"name", profile.name},
           {"signature", profile.structural_signature},
           {"mode", omlx::m5::to_string(profile.key.mode)},
           {"architecture", omlx::m5::to_string(profile.key.architecture)},
           {"quant_profile", omlx::m5::to_string(profile.key.quant_profile)},
           {"batch_size", profile.key.batch_size},
           {"token_capacity", profile.key.token_capacity},
           {"topology_signature", profile.key.topology_signature},
           {"parameter_bytes", profile.parameters.byte_size},
           {"parameter_fields", std::move(fields)},
           {"dispatch_count", profile.dispatches.size()},
           {"ranges", std::move(ranges)}});
    }
    const auto &stats = catalog.stats();
    const nlohmann::json result = {
        {"schema", "omlx-m5-b4-2-topology-profiles-v1"},
        {"evidence_scope",
         "static exact-topology compilation; not M5 Ultra performance"},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"architecture", omlx::m5::to_string(topology.architecture)},
        {"quant_profile", omlx::m5::to_string(topology.quant_profile)},
        {"topology_signature", topology.signature},
        {"max_context", topology.max_context},
        {"kv_page_tokens", topology.kv_page_tokens},
        {"layer_count", topology.layer_count},
        {"top_k", topology.top_k},
        {"stats",
         {{"profile_count", stats.profile_count},
          {"decode_profile_count", stats.decode_profile_count},
          {"prefill_profile_count", stats.prefill_profile_count},
          {"maximum_token_capacity", stats.maximum_token_capacity},
          {"total_parameter_bytes", stats.total_parameter_bytes},
          {"dispatch_records_per_profile", stats.dispatch_records_per_profile},
          {"finite", stats.finite}}},
        {"profiles", std::move(profiles)},
        {"selection_checks",
         {selection_json(catalog, omlx::m5::TopologyProfileMode::decode, 1, 1),
          selection_json(catalog, omlx::m5::TopologyProfileMode::prefill, 1, 5),
          selection_json(catalog, omlx::m5::TopologyProfileMode::prefill, 1,
                         129),
          selection_json(catalog, omlx::m5::TopologyProfileMode::decode, 1, 3),
          selection_json(catalog, omlx::m5::TopologyProfileMode::prefill, 2,
                         16),
          selection_json(catalog, omlx::m5::TopologyProfileMode::prefill, 1,
                         1025)}}};
    std::cout << result.dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "topology profile inspection failed: " << error.what() << '\n';
    return 1;
  }
}
