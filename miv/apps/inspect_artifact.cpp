// Inspect a `.miv` container (supports both `MIV00001` / `miv-artifact-v{1,2}`
// and legacy `OMLXM5A1` / `omlx-m5-artifact-v{1,2}`, including FWHT64/W4A8
// optimized layouts).
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"

namespace {

using Json = nlohmann::json;

struct Options {
  std::filesystem::path artifact;
  bool verify_payload = false;
  bool list_objects = false;
  bool materialization_plan = false;
  std::string sample_object;
  std::uint64_t sample_row = 0;
  std::uint64_t sample_start = 0;
  std::uint64_t sample_count = 0;
};

std::uint64_t parse_u64(std::string_view value, std::string_view label) {
  std::size_t parsed = 0;
  const auto result = std::stoull(std::string(value), &parsed);
  if (parsed != value.size()) {
    throw std::invalid_argument(std::string(label) + " must be an integer");
  }
  return result;
}

Options parse_args(int argc, const char *const *argv) {
  if (argc < 2) {
    throw std::invalid_argument(
        "usage: omlx_m5_inspect_artifact ARTIFACT [options]");
  }
  Options options;
  options.artifact = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    if (argument == "--verify-payload") {
      options.verify_payload = true;
    } else if (argument == "--list-objects") {
      options.list_objects = true;
    } else if (argument == "--materialization-plan") {
      options.materialization_plan = true;
    } else if (argument == "--sample-object" && i + 1 < argc) {
      options.sample_object = argv[++i];
    } else if (argument == "--sample-row" && i + 1 < argc) {
      options.sample_row = parse_u64(argv[++i], "--sample-row");
    } else if (argument == "--sample-start" && i + 1 < argc) {
      options.sample_start = parse_u64(argv[++i], "--sample-start");
    } else if (argument == "--sample-count" && i + 1 < argc) {
      options.sample_count = parse_u64(argv[++i], "--sample-count");
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  std::string(argument));
    }
  }
  return options;
}

} // namespace

int main(int argc, const char *const *argv) {
  try {
    const auto options = parse_args(argc, argv);
    const auto artifact =
        omlx::m5::ArtifactModel::open(options.artifact, options.verify_payload);
    const auto &summary = artifact.summary();
    std::map<std::string, std::size_t> formats;
    for (const auto &object : artifact.objects()) {
      ++formats[object.format];
    }
    Json output = {
        {"schema", "miv-artifact-inspection-v1"},
        {"artifact", artifact.path().string()},
        {"container_magic", summary.container_magic},
        {"manifest_schema", summary.manifest_schema},
        {"container_version", summary.container_version},
        {"container_flags", summary.container_flags},
        {"artifact_bytes", summary.artifact_bytes},
        {"manifest_bytes", summary.manifest_bytes},
        {"payload_offset", summary.payload_offset},
        {"payload_bytes", summary.payload_bytes},
        {"object_count", artifact.objects().size()},
        {"binding_count", artifact.bindings().size()},
        {"source_repo", summary.source_repo},
        {"source_revision", summary.source_revision},
        {"source_profile", summary.source_profile},
        {"payload_verified", options.verify_payload},
        {"formats", formats},
    };
    if (options.list_objects) {
      output["objects"] = Json::array();
      for (const auto &object : artifact.objects()) {
        output["objects"].push_back({{"id", object.id},
                                     {"name", object.name},
                                     {"format", object.format},
                                     {"layout", object.layout},
                                     {"input_transform", object.input_transform},
                                     {"shape", object.shape},
                                     {"padded_k", object.padded_k},
                                     {"object_bytes", object.object_bytes},
                                     {"object_sha256", object.object_sha256}});
      }
    }
    if (options.materialization_plan) {
      const auto plan = artifact.materialization_plan();
      output["materialization_plan"] = {
          {"alignment", plan.alignment},
          {"payload_bytes", plan.payload_bytes},
          {"upload_bytes", plan.upload_bytes},
          {"arena_bytes", plan.arena_bytes},
          {"arena_padding_bytes", plan.arena_padding_bytes},
          {"span_count", plan.spans.size()},
      };
    }
    if (!options.sample_object.empty()) {
      if (options.sample_count == 0) {
        throw std::invalid_argument(
            "--sample-count must be positive with --sample-object");
      }
      const auto &object = artifact.object(options.sample_object);
      output["sample"] = {
          {"object", object.name},
          {"format", object.format},
          {"row", options.sample_row},
          {"start", options.sample_start},
          {"values", artifact.decode_weight_row(object, options.sample_row,
                                                options.sample_start,
                                                options.sample_count)},
      };
    }
    std::cout << output.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "omlx_m5_inspect_artifact: " << error.what() << '\n';
    return 1;
  }
}
