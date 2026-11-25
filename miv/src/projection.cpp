#include "omlx_m5/projection.h"

#include <algorithm>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace omlx::m5 {
namespace {

bool is_power_of_two(std::uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          const char *label) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::overflow_error(std::string(label) + " overflow");
  }
  return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               const char *label) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error(std::string(label) + " overflow");
  }
  return left * right;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
  if (!is_power_of_two(alignment)) {
    throw std::invalid_argument("projection alignment is invalid");
  }
  const auto mask = alignment - 1;
  return checked_add(value, mask, "projection alignment") & ~mask;
}

std::uint64_t positive(std::int32_t value, const char *label) {
  if (value <= 0) {
    throw std::invalid_argument(std::string(label) + " must be positive");
  }
  return static_cast<std::uint64_t>(value);
}

ProjectionParentSpec parent(std::string role, std::string binding_name,
                            std::uint64_t input_features,
                            std::uint64_t output_features,
                            std::vector<ProjectionSliceSpec> outputs) {
  return {
      .role = std::move(role),
      .binding_name = std::move(binding_name),
      .input_features = input_features,
      .output_features = output_features,
      .outputs = std::move(outputs),
  };
}

} // namespace

std::string_view to_string(ProjectionBundleKind kind) {
  switch (kind) {
  case ProjectionBundleKind::full_attention:
    return "full_attention";
  case ProjectionBundleKind::gdn:
    return "gdn";
  case ProjectionBundleKind::dense_mlp:
    return "dense_mlp";
  }
  return "unknown";
}

ProjectionBundleSpec
make_full_attention_projection_bundle(const QwenModelConfig &config,
                                      std::string prefix) {
  const auto hidden = positive(config.hidden_size, "hidden size");
  const auto query = checked_multiply(
      positive(config.num_attention_heads, "attention heads"),
      positive(config.head_dim, "head dimension"), "query features");
  const auto key_value = checked_multiply(
      positive(config.num_key_value_heads, "KV heads"),
      positive(config.head_dim, "head dimension"), "KV features");
  ProjectionBundleSpec result{
      .kind = ProjectionBundleKind::full_attention,
      .prefix = prefix,
      .parents =
          {
              parent("attention_q_gate", prefix + ".q_proj", hidden, query * 2,
                     {{"attention_query", 0, query,
                       positive(config.num_attention_heads, "attention heads"),
                       positive(config.head_dim, "head dimension"),
                       positive(config.head_dim, "head dimension") * 2},
                      {"attention_gate",
                       positive(config.head_dim, "head dimension"), query,
                       positive(config.num_attention_heads, "attention heads"),
                       positive(config.head_dim, "head dimension"),
                       positive(config.head_dim, "head dimension") * 2}}),
              parent("attention_k", prefix + ".k_proj", hidden, key_value,
                     {{"attention_key", 0, key_value}}),
              parent("attention_v", prefix + ".v_proj", hidden, key_value,
                     {{"attention_value", 0, key_value}}),
          },
  };
  validate_projection_bundle(result);
  return result;
}

ProjectionBundleSpec make_gdn_projection_bundle(const QwenModelConfig &config,
                                                std::string prefix) {
  const auto hidden = positive(config.hidden_size, "hidden size");
  const auto key = checked_multiply(
      positive(config.linear_num_key_heads, "linear key heads"),
      positive(config.linear_key_head_dim, "linear key dimension"),
      "GDN key features");
  const auto value = checked_multiply(
      positive(config.linear_num_value_heads, "linear value heads"),
      positive(config.linear_value_head_dim, "linear value dimension"),
      "GDN value features");
  ProjectionBundleSpec result{
      .kind = ProjectionBundleKind::gdn,
      .prefix = prefix,
      .parents =
          {
              parent("gdn_qkv", prefix + ".in_proj_qkv", hidden,
                     key * 2 + value,
                     {{"gdn_query", 0, key},
                      {"gdn_key", key, key},
                      {"gdn_value", key * 2, value}}),
              parent("gdn_z", prefix + ".in_proj_z", hidden, value,
                     {{"gdn_gate_z", 0, value}}),
          },
  };
  validate_projection_bundle(result);
  return result;
}

ProjectionBundleSpec
make_dense_mlp_projection_bundle(const QwenModelConfig &config,
                                 std::string prefix) {
  const auto hidden = positive(config.hidden_size, "hidden size");
  const auto intermediate =
      positive(config.intermediate_size, "MLP intermediate size");
  ProjectionBundleSpec result{
      .kind = ProjectionBundleKind::dense_mlp,
      .prefix = prefix,
      .parents =
          {
              parent("mlp_gate", prefix + ".gate_proj", hidden, intermediate,
                     {{"mlp_gate", 0, intermediate}}),
              parent("mlp_up", prefix + ".up_proj", hidden, intermediate,
                     {{"mlp_up", 0, intermediate}}),
          },
  };
  validate_projection_bundle(result);
  return result;
}

void validate_projection_bundle(const ProjectionBundleSpec &bundle) {
  if (bundle.prefix.empty() || bundle.parents.empty()) {
    throw std::invalid_argument("projection bundle identity is invalid");
  }
  std::unordered_map<std::string, bool> roles;
  std::unordered_map<std::string, bool> outputs;
  std::uint64_t common_input = 0;
  for (const auto &selected_parent : bundle.parents) {
    if (selected_parent.role.empty() || selected_parent.binding_name.empty() ||
        selected_parent.input_features == 0 ||
        selected_parent.output_features == 0 ||
        selected_parent.outputs.empty() ||
        !roles.emplace(selected_parent.role, true).second) {
      throw std::invalid_argument("projection parent identity is invalid");
    }
    if (common_input == 0) {
      common_input = selected_parent.input_features;
    } else if (common_input != selected_parent.input_features) {
      throw std::invalid_argument(
          "projection bundle parents do not share one input");
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> intervals;
    std::uint64_t cursor = 0;
    for (const auto &slice : selected_parent.outputs) {
      const auto block_features = slice.block_features == 0
                                      ? slice.feature_count
                                      : slice.block_features;
      const auto block_stride = slice.block_stride_features == 0
                                    ? block_features
                                    : slice.block_stride_features;
      if (slice.name.empty() || slice.feature_count == 0 ||
          slice.block_count == 0 || block_features == 0 ||
          block_stride < block_features ||
          checked_multiply(slice.block_count, block_features,
                           "projection slice blocks") != slice.feature_count ||
          !outputs.emplace(slice.name, true).second) {
        throw std::invalid_argument(
            "projection output block geometry is invalid");
      }
      for (std::uint64_t block = 0; block < slice.block_count; ++block) {
        const auto begin = checked_add(
            slice.feature_offset,
            checked_multiply(block, block_stride, "projection slice stride"),
            "projection slice begin");
        const auto end =
            checked_add(begin, block_features, "projection slice end");
        if (end > selected_parent.output_features) {
          throw std::invalid_argument(
              "projection output block exceeds its parent");
        }
        intervals.emplace_back(begin, end);
      }
    }
    std::ranges::sort(intervals);
    for (const auto &[begin, end] : intervals) {
      if (begin != cursor) {
        throw std::invalid_argument(
            "projection outputs must uniquely cover the parent");
      }
      cursor = end;
    }
    if (cursor != selected_parent.output_features) {
      throw std::invalid_argument(
          "projection output slices do not cover the parent");
    }
  }
}

void validate_projection_parent(const ProjectionParentSpec &parent,
                                const ArtifactBinding &binding,
                                const MaterializedWeightView &weight) {
  if (binding.name != parent.binding_name ||
      weight.name != parent.binding_name ||
      binding.object_id != weight.object_id ||
      binding.format != weight.format ||
      binding.in_features != parent.input_features ||
      binding.out_features != parent.output_features ||
      weight.parent_shape.size() != 2 || weight.row_start != 0 ||
      weight.row_count != parent.output_features ||
      weight.in_features != parent.input_features) {
    throw std::invalid_argument(
        "projection parent disagrees with artifact binding or weight view");
  }
}

std::uint64_t
ProjectionSplitOutputView::token_offset(std::uint32_t token) const {
  if (token >= tokens) {
    throw std::out_of_range("projection token is outside the output view");
  }
  return checked_add(
      offset,
      checked_multiply(token, token_stride_bytes, "projection token offset"),
      "projection token offset");
}

std::uint64_t
ProjectionSplitOutputView::feature_offset(std::uint32_t token,
                                          std::uint64_t feature) const {
  if (token >= tokens || feature >= features || block_features == 0 ||
      block_count == 0 || features != block_count * block_features) {
    throw std::out_of_range("projection feature is outside the output view");
  }
  const auto block = feature / block_features;
  const auto within = feature - block * block_features;
  return checked_add(
      token_offset(token),
      checked_add(checked_multiply(block, block_stride_bytes,
                                   "projection feature block"),
                  checked_multiply(within, 2, "projection feature byte"),
                  "projection feature offset"),
      "projection feature offset");
}

bool ProjectionSplitOutputView::contiguous_per_token() const {
  return block_count == 1 || block_stride_bytes == block_features * 2;
}

class ProjectionOutputArena::Impl {
public:
  struct ParentPlacement {
    ProjectionParentSpec spec;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
  };

  Impl(const MetalContext &context, ProjectionBundleSpec selected_bundle,
       std::uint32_t selected_tokens, std::uint64_t alignment)
      : bundle(std::move(selected_bundle)), tokens(selected_tokens) {
    validate_projection_bundle(bundle);
    if (tokens == 0 || alignment < 256 || !is_power_of_two(alignment)) {
      throw std::invalid_argument("projection output arena domain is invalid");
    }
    for (const auto &selected_parent : bundle.parents) {
      const auto aligned = align_up(stats.capacity_bytes, alignment);
      stats.alignment_padding_bytes += aligned - stats.capacity_bytes;
      const auto values = checked_multiply(
          tokens, selected_parent.output_features, "projection parent values");
      const auto parent_bytes =
          checked_multiply(values, 2, "projection parent bytes");
      parents.push_back({
          .spec = selected_parent,
          .offset = aligned,
          .bytes = parent_bytes,
      });
      stats.logical_parent_bytes = checked_add(
          stats.logical_parent_bytes, parent_bytes, "projection logical bytes");
      stats.capacity_bytes =
          checked_add(aligned, parent_bytes, "projection arena bytes");
      ++stats.parent_count;
      stats.split_count += selected_parent.outputs.size();
    }
    stats.capacity_bytes = align_up(stats.capacity_bytes, alignment);
    stats.alignment_padding_bytes =
        stats.capacity_bytes - stats.logical_parent_bytes;
    storage = context.make_buffer(stats.capacity_bytes);
  }

  ProjectionBundleSpec bundle;
  std::uint32_t tokens = 0;
  std::vector<ParentPlacement> parents;
  ProjectionOutputArenaStats stats;
  MetalBuffer storage;
};

ProjectionOutputArena::ProjectionOutputArena(const MetalContext &context,
                                             ProjectionBundleSpec bundle,
                                             std::uint32_t tokens,
                                             std::uint64_t alignment)
    : impl_(std::make_unique<Impl>(context, std::move(bundle), tokens,
                                   alignment)) {}
ProjectionOutputArena::ProjectionOutputArena(
    ProjectionOutputArena &&) noexcept = default;
ProjectionOutputArena &
ProjectionOutputArena::operator=(ProjectionOutputArena &&) noexcept = default;
ProjectionOutputArena::~ProjectionOutputArena() = default;

const ProjectionBundleSpec &ProjectionOutputArena::bundle() const {
  return impl_->bundle;
}

std::uint32_t ProjectionOutputArena::tokens() const { return impl_->tokens; }

const ProjectionOutputArenaStats &ProjectionOutputArena::stats() const {
  return impl_->stats;
}

ProjectionParentOutputView
ProjectionOutputArena::parent(std::string_view role) {
  const auto found = std::ranges::find_if(
      impl_->parents, [&](const Impl::ParentPlacement &placement) {
        return placement.spec.role == role;
      });
  if (found == impl_->parents.end()) {
    throw std::out_of_range("unknown projection parent role");
  }
  return {
      .role = found->spec.role,
      .buffer = &impl_->storage,
      .offset = found->offset,
      .bytes = found->bytes,
      .tokens = impl_->tokens,
      .features = found->spec.output_features,
      .token_stride_bytes = found->spec.output_features * 2,
  };
}

ProjectionSplitOutputView ProjectionOutputArena::output(std::string_view name) {
  for (const auto &selected_parent : impl_->parents) {
    const auto found = std::ranges::find_if(
        selected_parent.spec.outputs,
        [&](const ProjectionSliceSpec &slice) { return slice.name == name; });
    if (found == selected_parent.spec.outputs.end()) {
      continue;
    }
    const auto token_stride = selected_parent.spec.output_features * 2;
    const auto block_features = found->block_features == 0
                                    ? found->feature_count
                                    : found->block_features;
    const auto block_stride_features = found->block_stride_features == 0
                                           ? block_features
                                           : found->block_stride_features;
    const auto offset =
        checked_add(selected_parent.offset, found->feature_offset * 2,
                    "projection output offset");
    const auto logical_bytes =
        checked_multiply(checked_multiply(impl_->tokens, found->feature_count,
                                          "projection output values"),
                         2, "projection output bytes");
    const auto final_token = checked_multiply(impl_->tokens - 1, token_stride,
                                              "projection output token span");
    const auto final_block =
        checked_multiply(found->block_count - 1, block_stride_features * 2,
                         "projection output block span");
    const auto span_bytes = checked_add(
        checked_add(final_token, final_block, "projection output span"),
        block_features * 2, "projection output span");
    return {
        .name = found->name,
        .parent_role = selected_parent.spec.role,
        .buffer = &impl_->storage,
        .offset = offset,
        .logical_bytes = logical_bytes,
        .span_bytes = span_bytes,
        .tokens = impl_->tokens,
        .features = found->feature_count,
        .token_stride_bytes = token_stride,
        .block_count = found->block_count,
        .block_features = block_features,
        .block_stride_bytes = block_stride_features * 2,
    };
  }
  throw std::out_of_range("unknown projection split output");
}

MetalBuffer &ProjectionOutputArena::buffer() { return impl_->storage; }

} // namespace omlx::m5
