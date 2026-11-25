#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/projection.h"
#include "omlx_m5/qmv.h"
#include "omlx_m5/quantization.h"

namespace {

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

const omlx::m5::ArtifactBinding &
binding(const omlx::m5::ArtifactModel &artifact, std::string_view name) {
  const auto found = std::ranges::find_if(
      artifact.bindings(), [&](const omlx::m5::ArtifactBinding &value) {
        return value.name == name;
      });
  if (found == artifact.bindings().end()) {
    throw std::out_of_range("projection binding is missing: " +
                            std::string(name));
  }
  return *found;
}

std::string remove_suffix(std::string_view value, std::string_view suffix) {
  if (!value.ends_with(suffix)) {
    throw std::invalid_argument("projection binding suffix mismatch");
  }
  return std::string(value.substr(0, value.size() - suffix.size()));
}

omlx::m5::MaterializedWeightView
upload_weight(const omlx::m5::ArtifactModel &artifact,
              const omlx::m5::ArtifactObject &object,
              omlx::m5::MetalContext &context,
              std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> &buffers) {
  omlx::m5::MaterializedWeightView weight{
      .object_id = object.id,
      .component = omlx::m5::classify_model_component(object.name),
      .name = object.name,
      .format = object.format,
      .layout = object.layout,
      .parent_shape = object.shape,
      .padded_k = object.padded_k,
      .row_count = object.shape[0],
      .in_features = object.shape[1],
  };
  for (const auto &selected_plane : object.planes) {
    buffers.push_back(std::make_unique<omlx::m5::MetalBuffer>(
        context.upload(artifact.plane_bytes(object, selected_plane))));
    weight.planes.push_back({
        .kind = selected_plane.kind,
        .buffer = buffers.back().get(),
        .segment_index = static_cast<std::uint32_t>(buffers.size() - 1),
        .bytes = selected_plane.bytes,
        .row_stride = selected_plane.bytes / object.shape[0],
    });
  }
  return weight;
}

nlohmann::json execute_parent(const omlx::m5::ArtifactModel &artifact,
                              const omlx::m5::ArtifactBinding &selected_binding,
                              const omlx::m5::ProjectionBundleSpec &bundle,
                              std::string_view parent_role,
                              const std::filesystem::path &metallib) {
  const auto parent_spec = std::ranges::find_if(
      bundle.parents, [&](const omlx::m5::ProjectionParentSpec &parent) {
        return parent.role == parent_role;
      });
  if (parent_spec == bundle.parents.end()) {
    throw std::logic_error("projection parent role is absent from bundle");
  }
  const auto &object = artifact.object(selected_binding.object_id);
  auto context = omlx::m5::MetalContext::create();
  std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> buffers;
  auto weight = upload_weight(artifact, object, context, buffers);
  omlx::m5::validate_projection_parent(*parent_spec, selected_binding, weight);
  omlx::m5::ProjectionOutputArena output_arena(context, bundle, 1);
  const auto parent_output = output_arena.parent(parent_role);
  auto plan = omlx::m5::prepare_dense_qmv_t1(context, metallib, weight);
  std::vector<std::uint16_t> input_words(object.shape[1]);
  std::vector<float> input(object.shape[1]);
  for (std::uint64_t column = 0; column < object.shape[1]; ++column) {
    const auto raw = static_cast<float>(
                         static_cast<std::int32_t>((column * 17) % 257) - 128) /
                     512.0F;
    input_words[column] = omlx::m5::float_to_bf16(raw);
    input[column] = omlx::m5::bf16_to_float(input_words[column]);
  }
  auto input_buffer = context.upload(bytes(input_words));
  const auto before = context.registry_stats();
  const auto submission = omlx::m5::execute_dense_qmv_t1(
      context, plan, input_buffer, 0, output_arena.buffer(),
      parent_output.offset);
  const auto after_linear = context.registry_stats();
  double max_relative_error = 0.0;
  nlohmann::json outputs = nlohmann::json::array();
  for (const auto &slice : parent_spec->outputs) {
    const auto view = output_arena.output(slice.name);
    const std::vector<std::uint64_t> logical_features = {
        0,
        slice.feature_count / 2,
        slice.feature_count - 1,
    };
    nlohmann::json samples = nlohmann::json::array();
    for (const auto logical_feature : logical_features) {
      const auto block = logical_feature / view.block_features;
      const auto within = logical_feature - block * view.block_features;
      const auto row =
          slice.feature_offset + block * (view.block_stride_bytes / 2) + within;
      const auto weights =
          artifact.decode_weight_row(object, row, 0, object.shape[1]);
      double expected = 0.0;
      for (std::uint64_t column = 0; column < object.shape[1]; ++column) {
        expected += static_cast<double>(input[column]) * weights[column];
      }
      const auto raw = output_arena.buffer().read(
          view.feature_offset(0, logical_feature), 2);
      std::uint16_t word = 0;
      std::memcpy(&word, raw.data(), sizeof(word));
      const auto actual = omlx::m5::bf16_to_float(word);
      max_relative_error = std::max(
          max_relative_error, std::abs(static_cast<double>(actual) - expected) /
                                  std::max(1.0, std::abs(expected)));
      samples.push_back({{"logical_feature", logical_feature},
                         {"parent_row", row},
                         {"actual", actual},
                         {"expected", expected}});
    }
    outputs.push_back({
        {"name", view.name},
        {"feature_offset", slice.feature_offset},
        {"features", view.features},
        {"token_stride_bytes", view.token_stride_bytes},
        {"block_count", view.block_count},
        {"block_features", view.block_features},
        {"block_stride_bytes", view.block_stride_bytes},
        {"logical_bytes", view.logical_bytes},
        {"span_bytes", view.span_bytes},
        {"samples", std::move(samples)},
    });
  }
  const auto after_split = context.registry_stats();
  if (max_relative_error > 0.02 ||
      after_split.dispatches_submitted != after_linear.dispatches_submitted ||
      after_split.upload_count != after_linear.upload_count) {
    throw std::runtime_error("projection execution or zero-copy split failed");
  }
  return {
      {"binding", selected_binding.name},
      {"role", std::string(parent_role)},
      {"format", object.format},
      {"input_features", object.shape[1]},
      {"parent_features", object.shape[0]},
      {"linear_dispatches", submission.dispatch_count},
      {"split_copy_dispatches", output_arena.stats().split_copy_dispatches},
      {"split_dispatch_delta",
       after_split.dispatches_submitted - after_linear.dispatches_submitted},
      {"upload_delta_after_split",
       after_split.upload_count - after_linear.upload_count},
      {"max_relative_error", max_relative_error},
      {"outputs", std::move(outputs)},
      {"registry_dispatch_delta",
       after_split.dispatches_submitted - before.dispatches_submitted},
  };
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_projection ARTIFACT CONFIG QMV_METALLIB");
    }
    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    const auto config = omlx::m5::load_qwen_model_config(argv[2]);
    std::uint64_t bundle_count = 0;
    std::uint64_t parent_count = 0;
    std::uint64_t split_count = 0;
    std::optional<omlx::m5::ArtifactBinding> attention_q;
    std::optional<omlx::m5::ArtifactBinding> gdn_qkv;
    for (const auto &selected_binding : artifact.bindings()) {
      std::optional<omlx::m5::ProjectionBundleSpec> bundle;
      if (selected_binding.use == "attention_q") {
        const auto prefix = remove_suffix(selected_binding.name, ".q_proj");
        bundle =
            omlx::m5::make_full_attention_projection_bundle(config, prefix);
        if (!attention_q.has_value()) {
          attention_q = selected_binding;
        }
      } else if (selected_binding.use == "gdn_qkv") {
        const auto prefix =
            remove_suffix(selected_binding.name, ".in_proj_qkv");
        bundle = omlx::m5::make_gdn_projection_bundle(config, prefix);
        if (!gdn_qkv.has_value()) {
          gdn_qkv = selected_binding;
        }
      } else {
        continue;
      }
      for (const auto &parent : bundle->parents) {
        const auto &selected_parent_binding =
            binding(artifact, parent.binding_name);
        const auto &object = artifact.object(selected_parent_binding.object_id);
        const omlx::m5::MaterializedWeightView metadata{
            .object_id = object.id,
            .name = object.name,
            .format = object.format,
            .layout = object.layout,
            .parent_shape = object.shape,
            .padded_k = object.padded_k,
            .row_count = object.shape[0],
            .in_features = object.shape[1],
        };
        omlx::m5::validate_projection_parent(parent, selected_parent_binding,
                                             metadata);
        ++parent_count;
        split_count += parent.outputs.size();
      }
      ++bundle_count;
    }
    if (!attention_q.has_value() || !gdn_qkv.has_value()) {
      throw std::runtime_error(
          "artifact lacks attention or GDN projection bundles");
    }
    const auto attention_prefix = remove_suffix(attention_q->name, ".q_proj");
    const auto gdn_prefix = remove_suffix(gdn_qkv->name, ".in_proj_qkv");
    nlohmann::json executions = nlohmann::json::array();
    executions.push_back(
        execute_parent(artifact, *attention_q,
                       omlx::m5::make_full_attention_projection_bundle(
                           config, attention_prefix),
                       "attention_q_gate", argv[3]));
    executions.push_back(
        execute_parent(artifact, *gdn_qkv,
                       omlx::m5::make_gdn_projection_bundle(config, gdn_prefix),
                       "gdn_qkv", argv[3]));
    const nlohmann::json result = {
        {"schema", "omlx-m5-b3-1-projection-v1"},
        {"evidence_scope", "M4 real parent projection and zero-copy split"},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"bundle_count", bundle_count},
        {"parent_count", parent_count},
        {"split_count", split_count},
        {"executions", std::move(executions)},
    };
    std::cout << result.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "projection inspection failed: " << error.what() << '\n';
    return 1;
  }
}
