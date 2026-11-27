#include "omlx_m5/layer_schedule.h"

#include "omlx_m5/materialization.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <unordered_map>

namespace omlx::m5 {
namespace {

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               const char *label) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error(std::string(label) + " overflow");
  }
  return left * right;
}

std::uint64_t linear_dispatches(std::string_view format) {
  return format.starts_with("oq_affine_") ? 2 : 1;
}

struct Resolver {
  const ArtifactModel &artifact;
  std::unordered_map<std::string, const ArtifactBinding *> bindings;
  std::unordered_map<std::string, const ArtifactObject *> objects;
  std::uint64_t lookups = 0;

  explicit Resolver(const ArtifactModel &source) : artifact(source) {
    for (const auto &binding : artifact.bindings()) {
      bindings.emplace(binding.name, &binding);
    }
    for (const auto &object : artifact.objects()) {
      objects.emplace(object.name, &object);
    }
  }

  const ArtifactObject &resolve(std::string_view name) {
    ++lookups;
    const auto binding = bindings.find(std::string(name));
    if (binding != bindings.end()) {
      return artifact.object(binding->second->object_id);
    }
    const auto exact = objects.find(std::string(name));
    if (exact != objects.end()) {
      return *exact->second;
    }
    const auto weight = objects.find(std::string(name) + ".weight");
    if (weight != objects.end()) {
      return *weight->second;
    }
    throw std::out_of_range("layer schedule object is absent: " +
                            std::string(name));
  }

  const ArtifactObject &resolve_suffix(std::string_view suffix) {
    ++lookups;
    const auto found = std::ranges::find_if(
        artifact.objects(), [&](const ArtifactObject &object) {
          return classify_model_component(object.name) ==
                     ModelComponent::text &&
                 object.name.ends_with(suffix);
        });
    if (found == artifact.objects().end()) {
      throw std::out_of_range("global schedule object is absent: " +
                              std::string(suffix));
    }
    return *found;
  }

  bool has(std::string_view name) const {
    return bindings.contains(std::string(name)) ||
           objects.contains(std::string(name)) ||
           objects.contains(std::string(name) + ".weight");
  }

  const ArtifactObject *try_resolve_use_or_suffix(std::string_view use,
                                                  std::string_view suffix) {
    ++lookups;
    const auto binding = std::ranges::find_if(
        artifact.bindings(), [&](const ArtifactBinding &record) {
          return classify_model_component(record.name) ==
                     ModelComponent::text &&
                 record.use == use;
        });
    if (binding != artifact.bindings().end()) {
      return &artifact.object(binding->object_id);
    }
    const auto found = std::ranges::find_if(
        artifact.objects(), [&](const ArtifactObject &object) {
          return classify_model_component(object.name) ==
                     ModelComponent::text &&
                 (object.name == suffix || object.name.ends_with(suffix));
        });
    if (found != artifact.objects().end()) {
      return &(*found);
    }
    return nullptr;
  }

  const ArtifactObject &resolve_use_or_suffix(std::string_view use,
                                              std::string_view suffix) {
    if (const auto *obj = try_resolve_use_or_suffix(use, suffix)) {
      return *obj;
    }
    throw std::out_of_range("global schedule object is absent: " +
                            std::string(suffix));
  }
};

void add_object_stage(QwenLayerSchedule &layer, Resolver &resolver,
                      LayerStageKind kind, std::string role,
                      std::string object_name, std::uint32_t dispatches,
                      std::uint64_t workspace = 0, std::uint64_t state = 0) {
  const auto &object = resolver.resolve(object_name);
  layer.stages.push_back({
      .kind = kind,
      .role = std::move(role),
      .object_name = object.name,
      .object_id = object.id,
      .format = object.format,
      .dispatches = dispatches,
      .workspace_bytes = workspace,
      .state_bytes = state,
  });
  layer.dispatches += dispatches;
  layer.workspace_bytes = std::max(layer.workspace_bytes, workspace);
  layer.state_bytes += state;
}

void add_linear(QwenLayerSchedule &layer, Resolver &resolver, std::string role,
                std::string name) {
  const auto &object = resolver.resolve(name);
  add_object_stage(
      layer, resolver, LayerStageKind::linear, std::move(role), std::move(name),
      static_cast<std::uint32_t>(linear_dispatches(object.format)));
  // add_object_stage resolves the same immutable id to keep all stage creation
  // on one fail-closed path. Both lookups occur only during catalog build.
}

void add_kernel_stage(QwenLayerSchedule &layer, LayerStageKind kind,
                      std::string role, std::uint32_t dispatches,
                      std::uint64_t workspace = 0, std::uint64_t state = 0) {
  layer.stages.push_back({
      .kind = kind,
      .role = std::move(role),
      .dispatches = dispatches,
      .workspace_bytes = workspace,
      .state_bytes = state,
  });
  layer.dispatches += dispatches;
  layer.workspace_bytes = std::max(layer.workspace_bytes, workspace);
  layer.state_bytes += state;
}

std::string layer_prefix(const ArtifactModel &artifact, std::uint32_t layer) {
  const auto dot_marker = ".layers." + std::to_string(layer) + ".";
  const auto root_marker = "layers." + std::to_string(layer) + ".";
  const auto extract = [&](std::string_view name) -> std::optional<std::string> {
    if (classify_model_component(name) != ModelComponent::text) {
      return std::nullopt;
    }
    if (const auto pos = name.find(dot_marker); pos != std::string_view::npos) {
      return std::string(name.substr(0, pos + dot_marker.size() - 1));
    }
    if (name.starts_with(root_marker)) {
      return std::string(name.substr(0, root_marker.size() - 1));
    }
    return std::nullopt;
  };
  for (const auto &binding : artifact.bindings()) {
    if (auto p = extract(binding.name)) {
      return *p;
    }
  }
  for (const auto &object : artifact.objects()) {
    if (auto p = extract(object.name)) {
      return *p;
    }
  }
  throw std::out_of_range("layer prefix is absent");
}

LayerStageRecord global_object_stage(Resolver &resolver, LayerStageKind kind,
                                     std::string role,
                                     const ArtifactObject &object,
                                     std::uint32_t dispatches) {
  ++resolver.lookups;
  return {
      .kind = kind,
      .role = std::move(role),
      .object_name = object.name,
      .object_id = object.id,
      .format = object.format,
      .dispatches = dispatches,
  };
}

} // namespace

std::string_view to_string(LayerStageKind kind) {
  switch (kind) {
  case LayerStageKind::normalization:
    return "normalization";
  case LayerStageKind::linear:
    return "linear";
  case LayerStageKind::attention_postprocess:
    return "attention_postprocess";
  case LayerStageKind::attention:
    return "attention";
  case LayerStageKind::gdn:
    return "gdn";
  case LayerStageKind::router:
    return "router";
  case LayerStageKind::swiglu:
    return "swiglu";
  case LayerStageKind::reduce:
    return "reduce";
  case LayerStageKind::combine:
    return "combine";
  }
  return "unknown";
}

const std::vector<QwenLayerSchedule> &QwenLayerCatalog::layers() const {
  return layers_;
}
const QwenLayerCatalogStats &QwenLayerCatalog::stats() const { return stats_; }
const QwenLayerSchedule &QwenLayerCatalog::layer(std::uint32_t index) const {
  if (index >= layers_.size()) {
    throw std::out_of_range("layer schedule index is out of range");
  }
  return layers_[index];
}
const std::vector<LayerStageRecord> &QwenLayerCatalog::global_stages() const {
  return global_stages_;
}

QwenLayerCatalog build_qwen_layer_catalog(const ArtifactModel &artifact,
                                          const QwenModelConfig &config,
                                          std::uint32_t max_context,
                                          std::uint32_t kv_page_tokens) {
  if (max_context == 0 || kv_page_tokens == 0 ||
      config.num_hidden_layers <= 0 ||
      config.layer_types.size() !=
          static_cast<std::size_t>(config.num_hidden_layers)) {
    throw std::invalid_argument("layer catalog domain is invalid");
  }
  Resolver resolver(artifact);
  QwenLayerCatalog catalog;
  catalog.layers_.reserve(config.num_hidden_layers);
  const auto query_width = checked_multiply(config.num_attention_heads,
                                            config.head_dim, "query width");
  const auto query_out_width =
      config.has_attn_output_gate
          ? checked_multiply(query_width, 2, "gated query width")
          : query_width;
  const auto kv_width =
      checked_multiply(config.num_key_value_heads, config.head_dim, "KV width");
  const bool has_gdn = config.linear_attention_layers() > 0;
  if (has_gdn && config.linear_conv_kernel_dim < 1) {
    throw std::invalid_argument("GDN layers require linear_conv_kernel_dim >= 1");
  }
  const auto gdn_key_width =
      has_gdn ? checked_multiply(config.linear_num_key_heads,
                                 config.linear_key_head_dim, "GDN key width")
              : 0ULL;
  const auto gdn_value_width =
      has_gdn ? checked_multiply(config.linear_num_value_heads,
                                 config.linear_value_head_dim,
                                 "GDN value width")
              : 0ULL;
  const auto gdn_conv_width =
      has_gdn ? (gdn_key_width * 2 + gdn_value_width) : 0ULL;
  const auto kv_pages =
      (static_cast<std::uint64_t>(max_context) + kv_page_tokens - 1) /
      kv_page_tokens;
  const auto attention_state = checked_multiply(
      checked_multiply(kv_pages, kv_page_tokens, "KV capacity"),
      checked_multiply(kv_width, 4, "KV K/V BF16"), "attention state");
  const auto gdn_state =
      has_gdn ? (checked_multiply(gdn_value_width,
                                  config.linear_key_head_dim * 4ULL,
                                  "GDN recurrent state") +
                 checked_multiply(
                     static_cast<std::uint64_t>(config.linear_conv_kernel_dim - 1),
                     gdn_conv_width * 2ULL, "GDN history"))
              : 0ULL;

  for (std::uint32_t index = 0;
       index < static_cast<std::uint32_t>(config.num_hidden_layers); ++index) {
    QwenLayerSchedule layer{
        .layer = index,
        .type = config.layer_types[index],
        .moe = config.num_experts > 0,
        .prefix = layer_prefix(artifact, index),
    };
    add_object_stage(layer, resolver, LayerStageKind::normalization,
                     "input_norm", layer.prefix + ".input_layernorm.weight", 1);
    if (layer.type == LayerType::full_attention) {
      const auto attention = layer.prefix + ".self_attn";
      add_linear(layer, resolver, "attention_q", attention + ".q_proj");
      add_linear(layer, resolver, "attention_k", attention + ".k_proj");
      add_linear(layer, resolver, "attention_v", attention + ".v_proj");
      if (config.has_qk_norm && resolver.has(attention + ".q_norm")) {
        add_object_stage(layer, resolver, LayerStageKind::attention_postprocess,
                         "q_norm", attention + ".q_norm", 0);
        add_object_stage(layer, resolver, LayerStageKind::attention_postprocess,
                         "k_norm", attention + ".k_norm", 1);
      } else {
        add_kernel_stage(layer, LayerStageKind::attention_postprocess,
                         "qk_rope", 1);
      }
      add_kernel_stage(layer, LayerStageKind::attention, "paged_attention", 2,
                       query_out_width * 2, attention_state);
      add_linear(layer, resolver, "attention_o", attention + ".o_proj");
      ++catalog.stats_.full_attention_layers;
    } else {
      const auto gdn = layer.prefix + ".linear_attn";
      add_linear(layer, resolver, "gdn_qkv", gdn + ".in_proj_qkv");
      add_linear(layer, resolver, "gdn_z", gdn + ".in_proj_z");
      add_linear(layer, resolver, "gdn_a", gdn + ".in_proj_a");
      add_linear(layer, resolver, "gdn_b", gdn + ".in_proj_b");
      add_object_stage(layer, resolver, LayerStageKind::gdn, "gdn_conv",
                       gdn + ".conv1d.weight", 0);
      add_object_stage(layer, resolver, LayerStageKind::gdn, "gdn_a_log",
                       gdn + ".A_log", 0);
      add_object_stage(layer, resolver, LayerStageKind::gdn, "gdn_dt_bias",
                       gdn + ".dt_bias", 0);
      add_object_stage(layer, resolver, LayerStageKind::gdn, "gdn_norm",
                       gdn + ".norm.weight", 3,
                       (gdn_conv_width + gdn_value_width) * 2, gdn_state);
      add_linear(layer, resolver, "gdn_out", gdn + ".out_proj");
      ++catalog.stats_.gdn_layers;
    }
    add_object_stage(layer, resolver, LayerStageKind::normalization,
                     "post_attention_norm",
                     layer.prefix + ".post_attention_layernorm.weight", 1);
    if (!layer.moe) {
      const auto mlp = layer.prefix + ".mlp";
      add_linear(layer, resolver, "mlp_gate", mlp + ".gate_proj");
      add_linear(layer, resolver, "mlp_up", mlp + ".up_proj");
      add_kernel_stage(layer, LayerStageKind::swiglu, "mlp_swiglu", 1,
                       config.intermediate_size * 2ULL);
      add_linear(layer, resolver, "mlp_down", mlp + ".down_proj");
      ++catalog.stats_.dense_mlp_layers;
    } else {
      if (config.num_experts < 2 || config.num_experts_per_token < 1) {
        throw std::invalid_argument("invalid MoE schedule expert geometry");
      }
      const auto top_k_u64 =
          static_cast<std::uint64_t>(config.num_experts_per_token);
      const auto mlp = layer.prefix + ".mlp";
      add_linear(layer, resolver, "moe_router", mlp + ".gate");
      add_kernel_stage(layer, LayerStageKind::router, "moe_top8", 1);
      const auto expert = resolver.has(mlp + ".switch_mlp.gate_proj")
                              ? (mlp + ".switch_mlp")
                              : (mlp + ".experts");
      add_linear(layer, resolver, "moe_expert_gate", expert + ".gate_proj");
      add_linear(layer, resolver, "moe_expert_up", expert + ".up_proj");
      add_kernel_stage(layer, LayerStageKind::swiglu, "moe_swiglu", 1,
                       checked_multiply(config.intermediate_size,
                                        top_k_u64 * 2ULL,
                                        "MoE SwiGLU workspace"));
      add_linear(layer, resolver, "moe_expert_down", expert + ".down_proj");
      add_kernel_stage(layer, LayerStageKind::reduce, "moe_top8_reduce", 1);
      if (config.shared_expert_intermediate_size > 0) {
        if (config.has_shared_expert_gate &&
            resolver.has(mlp + ".shared_expert_gate")) {
          add_linear(layer, resolver, "moe_shared_gate",
                     mlp + ".shared_expert_gate");
        }
        const auto shared = resolver.has(mlp + ".shared_expert.gate_proj")
                                ? (mlp + ".shared_expert")
                                : (mlp + ".shared_experts");
        add_linear(layer, resolver, "moe_shared_mlp_gate",
                   shared + ".gate_proj");
        add_linear(layer, resolver, "moe_shared_mlp_up", shared + ".up_proj");
        add_kernel_stage(layer, LayerStageKind::swiglu, "moe_shared_swiglu", 1,
                         config.shared_expert_intermediate_size * 2ULL);
        add_linear(layer, resolver, "moe_shared_mlp_down",
                   shared + ".down_proj");
        add_kernel_stage(layer, LayerStageKind::combine, "moe_shared_combine",
                         1);
      }
      ++catalog.stats_.moe_mlp_layers;
    }
    catalog.stats_.stage_count += layer.stages.size();
    catalog.stats_.dispatches_per_decode_step += layer.dispatches;
    catalog.stats_.unbatched_command_buffers_per_decode_step +=
        std::ranges::count_if(layer.stages, [](const LayerStageRecord &stage) {
          return stage.dispatches > 0;
        });
    catalog.stats_.persistent_state_bytes += layer.state_bytes;
    catalog.stats_.peak_layer_workspace_bytes = std::max(
        catalog.stats_.peak_layer_workspace_bytes, layer.workspace_bytes);
    catalog.layers_.push_back(std::move(layer));
  }

  const ArtifactObject *emb_ptr =
      resolver.try_resolve_use_or_suffix("embedding", "embed_tokens.weight");
  if (emb_ptr == nullptr) {
    emb_ptr = resolver.try_resolve_use_or_suffix("embedding", "embed_tokens");
  }
  if (emb_ptr == nullptr) {
    throw std::out_of_range("global schedule object is absent: embed_tokens");
  }
  const auto &embedding = *emb_ptr;

  const ArtifactObject *final_norm_ptr = nullptr;
  ++resolver.lookups;
  for (const auto &object : artifact.objects()) {
    if (classify_model_component(object.name) == ModelComponent::text &&
        object.name.ends_with("model.norm.weight")) {
      final_norm_ptr = &object;
      break;
    }
  }
  if (final_norm_ptr == nullptr) {
    for (const auto &object : artifact.objects()) {
      if (classify_model_component(object.name) == ModelComponent::text &&
          (object.name == "norm.weight" ||
           (object.name.ends_with(".norm.weight") &&
            object.name.find("layers.") == std::string::npos))) {
        final_norm_ptr = &object;
        break;
      }
    }
  }
  if (final_norm_ptr == nullptr) {
    throw std::out_of_range("global schedule object is absent: model.norm.weight");
  }
  const auto &final_norm = *final_norm_ptr;

  const ArtifactObject *lm_head_ptr =
      resolver.try_resolve_use_or_suffix("lm_head", "lm_head.weight");
  if (lm_head_ptr == nullptr) {
    lm_head_ptr = resolver.try_resolve_use_or_suffix("lm_head", "lm_head");
  }
  const auto &lm_head = (lm_head_ptr != nullptr) ? *lm_head_ptr : embedding;
  catalog.global_stages_.push_back(global_object_stage(
      resolver, LayerStageKind::linear, "embedding", embedding, 1));
  catalog.global_stages_.push_back(global_object_stage(
      resolver, LayerStageKind::normalization, "final_norm", final_norm, 1));
  catalog.global_stages_.push_back(global_object_stage(
      resolver, LayerStageKind::linear, "lm_head", lm_head,
      static_cast<std::uint32_t>(linear_dispatches(lm_head.format))));
  catalog.global_stages_.push_back({
      .kind = LayerStageKind::reduce,
      .role = "sampling",
      .dispatches = 1,
  });
  catalog.stats_.global_stage_count = catalog.global_stages_.size();
  catalog.stats_.stage_count += catalog.global_stages_.size();
  for (const auto &stage : catalog.global_stages_) {
    catalog.stats_.dispatches_per_decode_step += stage.dispatches;
    catalog.stats_.unbatched_command_buffers_per_decode_step +=
        stage.dispatches > 0 ? 1 : 0;
  }
  // B3.9 batches the immutable plans within each layer. The vocabulary
  // boundary is one additional batch; B4 may later batch across layers.
  catalog.stats_.batched_command_buffers_per_decode_step =
      catalog.layers_.size() + 1;
  catalog.stats_.layer_count = catalog.layers_.size();
  catalog.stats_.build_time_artifact_lookups = resolver.lookups;
  catalog.stats_.token_loop_artifact_lookups = 0;
  return catalog;
}

} // namespace omlx::m5
