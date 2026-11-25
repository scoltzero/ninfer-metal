#include "omlx_m5/qwen_program.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace omlx::m5 {
namespace {

struct QwenArConfig {
  std::uint32_t capacity = 0;
  std::uint32_t hidden_size = 0;
  std::uint32_t vocabulary_size = 0;
  std::uint32_t max_context = 0;
  std::uint32_t architecture = 0;
  std::uint32_t quant_profile = 0;
  std::uint32_t expert_top_k = 0;
  std::uint32_t expert_count = 0;
  std::uint32_t full_attention_layers = 0;
  std::uint32_t gdn_layers = 0;
  std::uint32_t layer_count = 0;
  std::uint32_t model_seed = 0;
  std::uint32_t weight_factor_bits = 0;
};

struct QwenArSampling {
  float temperature = 0.0F;
  float top_p = 1.0F;
  std::uint32_t top_k = 0;
  std::uint32_t seed = 0;
};

static_assert(sizeof(QwenArConfig) == 52);
static_assert(sizeof(QwenArSampling) == 16);

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

template <class T> std::vector<std::byte> constant_bytes(const T &value) {
  const auto bytes = std::as_bytes(std::span(&value, 1));
  return {bytes.begin(), bytes.end()};
}

std::uint64_t byte_checksum(std::span<const std::byte> values) {
  std::uint64_t hash = 1'469'598'103'934'665'603ULL;
  for (const auto value : values) {
    hash ^= static_cast<std::uint8_t>(value);
    hash *= 1'099'511'628'211ULL;
  }
  return hash;
}

std::uint32_t ar_mix(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7FEB352DU;
  value ^= value >> 15U;
  value *= 0x846CA68BU;
  value ^= value >> 16U;
  return value;
}

float decode_bf16(std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

float decode_e2m1(std::uint8_t value) {
  constexpr std::array<float, 8> magnitudes = {0.0F, 0.5F, 1.0F, 1.5F,
                                               2.0F, 3.0F, 4.0F, 6.0F};
  const auto magnitude = magnitudes[value & 7U];
  return (value & 8U) == 0 ? magnitude : -magnitude;
}

float decode_e8m0(std::uint8_t value) {
  const auto bits =
      value == 0 ? 0x00400000U : static_cast<std::uint32_t>(value) << 23U;
  return std::bit_cast<float>(bits);
}

std::uint32_t sample_reference(std::span<const float> logits,
                               std::uint32_t position,
                               const QwenArSamplingOptions &sampling,
                               std::uint32_t model_seed) {
  if (logits.empty() || logits.size() > 64 ||
      !std::isfinite(sampling.temperature) || sampling.temperature < 0.0F ||
      !std::isfinite(sampling.top_p) || sampling.top_p < 0.0F ||
      sampling.top_p > 1.0F || sampling.top_k > 64) {
    throw std::invalid_argument("Qwen AR sampling options are invalid");
  }
  std::uint32_t best = 0;
  for (std::uint32_t token = 1; token < logits.size(); ++token) {
    if (logits[token] > logits[best] ||
        (logits[token] == logits[best] && token < best)) {
      best = token;
    }
  }
  if (sampling.temperature <= 0.0F || sampling.top_k == 0) {
    return best;
  }
  const auto count = std::min<std::size_t>(sampling.top_k, logits.size());
  std::vector<std::uint32_t> ids(logits.size());
  for (std::uint32_t index = 0; index < ids.size(); ++index) {
    ids[index] = index;
  }
  std::ranges::sort(ids, [&](std::uint32_t left, std::uint32_t right) {
    return logits[left] != logits[right] ? logits[left] > logits[right]
                                         : left < right;
  });
  ids.resize(count);
  std::vector<float> probabilities(count);
  float sum = 0.0F;
  for (std::size_t index = 0; index < count; ++index) {
    probabilities[index] =
        std::exp((logits[ids[index]] - logits[ids[0]]) / sampling.temperature);
    sum += probabilities[index];
  }
  std::size_t kept = count;
  if (sampling.top_p < 1.0F) {
    float prefix = 0.0F;
    for (std::size_t index = 0; index < count; ++index) {
      prefix += probabilities[index] / sum;
      if (prefix >= sampling.top_p) {
        kept = index + 1;
        break;
      }
    }
  }
  float kept_sum = 0.0F;
  for (std::size_t index = 0; index < kept; ++index) {
    kept_sum += probabilities[index];
  }
  const auto random_bits = ar_mix(sampling.seed ^ position ^ model_seed);
  const auto draw =
      static_cast<float>(random_bits & 0x00FFFFFFU) * (1.0F / 16'777'216.0F);
  float cumulative = 0.0F;
  for (std::size_t index = 0; index < kept; ++index) {
    cumulative += probabilities[index] / kept_sum;
    if (draw < cumulative || index + 1 == kept) {
      return ids[index];
    }
  }
  return ids.back();
}

std::uint64_t default_workspace_bytes(const QwenModelConfig &config,
                                      const TopologyProfileCatalog &profiles,
                                      const QwenLayerCatalog &layers) {
  const auto activation_bytes = checked_multiply(
      checked_multiply(profiles.stats().maximum_token_capacity,
                       static_cast<std::uint64_t>(config.hidden_size),
                       "Program activation rows"),
      4, "Program activation ping-pong");
  return std::max<std::uint64_t>(
      {256, activation_bytes, layers.stats().peak_layer_workspace_bytes});
}

std::vector<std::uint32_t> required_objects(const QwenLayerCatalog &layers) {
  std::vector<std::uint32_t> result;
  const auto append = [&](const LayerStageRecord &stage) {
    if (stage.object_id != UINT32_MAX) {
      result.push_back(stage.object_id);
    }
  };
  for (const auto &stage : layers.global_stages()) {
    append(stage);
  }
  for (const auto &layer : layers.layers()) {
    for (const auto &stage : layer.stages) {
      append(stage);
    }
  }
  std::ranges::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

void validate_plan_identity(const QwenProgramBuildPlan &plan,
                            const QwenLayerCatalog &layers,
                            const DispatchTable &dispatch_table,
                            const TopologyProfileCatalog &profiles) {
  const auto &topology = profiles.topology();
  const auto dispatch_stats = dispatch_table.stats();
  if (!dispatch_table.sealed() || !plan.dispatch_table.sealed ||
      dispatch_stats.candidate_count != plan.dispatch_table.candidate_count ||
      dispatch_stats.installed_candidate_count !=
          plan.dispatch_table.installed_candidate_count ||
      dispatch_stats.planned_candidate_count !=
          plan.dispatch_table.planned_candidate_count ||
      plan.architecture != topology.architecture ||
      plan.quant_profile != topology.quant_profile ||
      plan.topology_signature != topology.signature ||
      plan.layer_count != layers.layers().size() ||
      plan.stage_count != layers.stats().stage_count ||
      plan.dispatch_count != layers.stats().dispatches_per_decode_step ||
      plan.profile_count != profiles.profiles().size() ||
      plan.parameter_bytes != profiles.stats().total_parameter_bytes ||
      plan.required_object_ids != required_objects(layers)) {
    throw std::invalid_argument("Qwen Program build plan identity mismatch");
  }
}

template <class T>
std::vector<std::byte> padded_bytes(std::span<const T> values,
                                    std::uint64_t target_bytes,
                                    std::string_view label) {
  const auto input = std::as_bytes(values);
  if (input.size() > target_bytes) {
    throw std::invalid_argument(std::string(label) +
                                " exceeds profile capacity");
  }
  std::vector<std::byte> result(target_bytes);
  std::ranges::copy(input, result.begin());
  return result;
}

const MetalParameterFieldDescriptor &field(const TopologyProfile &profile,
                                           std::string_view key) {
  const auto found =
      std::ranges::find_if(profile.parameters.fields,
                           [&](const MetalParameterFieldDescriptor &item) {
                             return item.key == key;
                           });
  if (found == profile.parameters.fields.end()) {
    throw std::out_of_range("profile parameter field is absent: " +
                            std::string(key));
  }
  return *found;
}

void append_stage_plan(std::vector<QwenProgramStagePlan> &result,
                       const MaterializedModel &weights, std::uint32_t layer,
                       const LayerStageRecord &stage, QwenProgramStats &stats) {
  QwenProgramStagePlan plan{
      .layer = layer,
      .kind = stage.kind,
      .role = stage.role,
      .object_id = stage.object_id,
      .dispatches = stage.dispatches,
  };
  if (stage.object_id != UINT32_MAX) {
    if (!weights.has_object(stage.object_id)) {
      throw std::invalid_argument(
          "materialized weights do not cover a static stage object");
    }
    const auto view = weights.parent_view(stage.object_id);
    for (const auto &plane : view.planes) {
      if (plane.buffer == nullptr || plane.buffer->gpu_address() == 0) {
        throw std::invalid_argument(
            "materialized stage plane has no GPU address");
      }
      plan.planes.push_back(
          {.kind = plane.kind,
           .segment_index = plane.segment_index,
           .offset = plane.offset,
           .bytes = plane.bytes,
           .gpu_address = plane.buffer->gpu_address() + plane.offset});
    }
    ++stats.construction_artifact_lookups;
    stats.static_plane_binding_count += plan.planes.size();
  }
  result.push_back(std::move(plan));
}

} // namespace

DispatchTable
build_qwen_program_dispatch_table(const ArtifactModel &artifact,
                                  const QwenLayerCatalog &layers,
                                  const TopologyProfileCatalog &profiles) {
  if (layers.layers().empty() || profiles.profiles().empty()) {
    throw std::invalid_argument("Qwen Program dispatch source is empty");
  }
  std::unordered_map<std::uint32_t, const ArtifactBinding *> bindings;
  for (const auto &binding : artifact.bindings()) {
    bindings.emplace(binding.object_id, &binding);
  }
  std::unordered_set<std::string> candidate_ids;
  DispatchTable result;
  const auto append_stage = [&](const LayerStageRecord &stage) {
    if (stage.kind != LayerStageKind::linear || stage.object_id == UINT32_MAX) {
      return;
    }
    const auto &object = artifact.object(stage.object_id);
    for (const auto &profile : profiles.profiles()) {
      KernelKey key;
      const auto binding = bindings.find(object.id);
      if (binding == bindings.end()) {
        if (object.shape.size() != 2 || (object.format != "direct_bf16_v1" &&
                                         object.format != "direct_f16_v1")) {
          throw std::invalid_argument(
              "linear Program stage has no dispatchable artifact binding");
        }
        key = make_direct_kernel_key(object, profile.key.token_capacity);
      } else {
        key =
            make_kernel_key(*binding->second, object,
                            profile.key.token_capacity, ActivationProfile::a16);
      }
      const auto add_candidate = [&](ActivationProfile activation,
                                     MetalExecutionRoute route) {
        auto selected = key;
        selected.preferred_activation = activation;
        const auto activation_name =
            activation == ActivationProfile::a4 ? "a4" : "a16";
        const auto id = "program.object." + std::to_string(object.id) + ".t" +
                        std::to_string(profile.key.token_capacity) + "." +
                        activation_name;
        if (!candidate_ids.insert(id).second) {
          return;
        }
        const auto family =
            profile.key.token_capacity == 1
                ? KernelFamily::qmv
                : (profile.key.token_capacity <= 128 ? KernelFamily::row_split
                                                     : KernelFamily::tail_safe);
        result.register_candidate({
            .id = id,
            .executor_key = "planned." + id,
            .priority = activation == ActivationProfile::a4 ? 10U : 20U,
            .family = family,
            .format_id = selected.format_id,
            .layout_id = selected.layout_id,
            .activation = activation,
            .input_type = selected.input_type,
            .output_type = selected.output_type,
            .accumulator_type = selected.accumulator_type,
            .geometry = selected.geometry,
            .output_features = selected.output_features,
            .input_features = selected.input_features,
            .min_tokens = selected.tokens,
            .max_tokens = selected.tokens,
            .experts = selected.experts,
            .top_k = selected.top_k,
            .fused_op = selected.fused_op,
            .alignment = family == KernelFamily::tail_safe
                             ? KernelAlignmentClass::tail
                             : KernelAlignmentClass::full,
            .required_route = route,
            .tail_safe = family == KernelFamily::tail_safe,
        });
      };
      add_candidate(ActivationProfile::a16, MetalExecutionRoute::classic_msl);
      if (object.format == "mxfp4_e2m1_g32_e8m0_v1") {
        add_candidate(ActivationProfile::a4,
                      MetalExecutionRoute::native_tensorops);
      }
    }
  };
  for (const auto &stage : layers.global_stages()) {
    append_stage(stage);
  }
  for (const auto &layer : layers.layers()) {
    for (const auto &stage : layer.stages) {
      append_stage(stage);
    }
  }
  if (candidate_ids.empty()) {
    throw std::invalid_argument("Qwen Program dispatch catalog is empty");
  }
  result.seal({});
  return result;
}

QwenProgramBuildPlan plan_qwen_program(
    const ArtifactModel &artifact, const QwenModelConfig &config,
    const ModelMaterializationPlan &materialization,
    const QwenLayerCatalog &layers, const TopologyProfileCatalog &profiles,
    DispatchTableStats dispatch_table, QwenProgramBuildOptions options) {
  if (materialization.source != &artifact || !dispatch_table.sealed ||
      options.max_context == 0 || options.kv_page_tokens == 0 ||
      options.max_context != profiles.topology().max_context ||
      options.kv_page_tokens != profiles.topology().kv_page_tokens ||
      layers.stats().token_loop_artifact_lookups != 0) {
    throw std::invalid_argument("Qwen Program planning contract is invalid");
  }
  const auto architecture =
      config.num_experts > 0 ? QwenArchitecture::moe : QwenArchitecture::dense;
  if (architecture != profiles.topology().architecture ||
      layers.layers().size() !=
          static_cast<std::size_t>(config.num_hidden_layers)) {
    throw std::invalid_argument("Qwen Program architecture is inconsistent");
  }
  auto objects = required_objects(layers);
  std::unordered_set<std::uint32_t> selected(materialization.object_ids.begin(),
                                             materialization.object_ids.end());
  for (const auto object_id : objects) {
    static_cast<void>(artifact.object(object_id));
    if (!selected.contains(object_id)) {
      throw std::invalid_argument(
          "materialization plan does not cover a Program weight");
    }
  }
  const auto workspace = options.workspace_bytes == 0
                             ? default_workspace_bytes(config, profiles, layers)
                             : options.workspace_bytes;
  const auto state = options.state_bytes == 0
                         ? layers.stats().persistent_state_bytes
                         : options.state_bytes;
  if (workspace == 0 || state < layers.stats().persistent_state_bytes) {
    throw std::invalid_argument("Qwen Program mutable arena is undersized");
  }
  return {
      .architecture = architecture,
      .quant_profile = profiles.topology().quant_profile,
      .topology_signature = profiles.topology().signature,
      .max_context = options.max_context,
      .kv_page_tokens = options.kv_page_tokens,
      .layer_count = static_cast<std::uint32_t>(layers.layers().size()),
      .full_attention_layers = layers.stats().full_attention_layers,
      .gdn_layers = layers.stats().gdn_layers,
      .hidden_size = static_cast<std::uint32_t>(config.hidden_size),
      .vocabulary_size = static_cast<std::uint32_t>(config.vocab_size),
      .expert_count = static_cast<std::uint32_t>(config.num_experts),
      .expert_top_k = static_cast<std::uint32_t>(config.num_experts_per_token),
      .stage_count = layers.stats().stage_count,
      .dispatch_count = layers.stats().dispatches_per_decode_step,
      .profile_count = profiles.stats().profile_count,
      .weight_segment_count = materialization.segments.size(),
      .weight_bytes = materialization.resident_bytes,
      .workspace_bytes = workspace,
      .state_bytes = state,
      .parameter_bytes = profiles.stats().total_parameter_bytes,
      // Metal 4 commit also installs the context's permanent zero binding.
      .residency_allocation_count =
          materialization.segments.size() + profiles.stats().profile_count + 3,
      .required_object_ids = std::move(objects),
      .dispatch_table = dispatch_table,
  };
}

struct QwenArProfileRuntime {
  MetalCommandPlan plan;
  std::optional<MetalIcbProfile> icb;
  QwenArConfig config;
};

struct QwenArExecutor {
  MetalBuffer hidden;
  MetalBuffer state;
  MetalBuffer logits;
  MetalBuffer sampled;
  std::vector<QwenArProfileRuntime> profiles;
  std::vector<float> oracle_state;
  std::uint32_t next_position = 0;
  std::uint32_t model_seed = 0;
  float weight_factor = 0.0F;
  const MetalBuffer *weight_codes = nullptr;
  std::uint64_t weight_codes_offset = 0;
  const MetalBuffer *weight_scales = nullptr;
  std::uint64_t weight_scales_offset = 0;
  const MetalBuffer *weight_biases = nullptr;
  std::uint64_t weight_biases_offset = 0;
  std::uint32_t last_profile = UINT32_MAX;
  std::uint64_t completed_steps = 0;
  std::optional<std::uint64_t> failure_after;
  std::atomic_bool cancel_requested = false;
  QwenArStats stats;
};

class QwenProgram::Impl {
public:
  Impl(QwenProgramBuildPlan selected_plan, MetalContext &&selected_context,
       MaterializedModel &&selected_weights, QwenLayerCatalog selected_layers,
       DispatchTable &&selected_dispatch,
       TopologyProfileCatalog selected_profiles)
      : plan(std::move(selected_plan)), context(std::move(selected_context)),
        weights(std::move(selected_weights)),
        layers(std::move(selected_layers)),
        dispatch(std::move(selected_dispatch)),
        profiles(std::move(selected_profiles)),
        workspace(context, plan.workspace_bytes),
        state(context, plan.state_bytes) {
    validate_plan_identity(plan, layers, dispatch, profiles);
    if (!weights.valid() ||
        weights.segment_count() != plan.weight_segment_count ||
        weights.stats().resident_bytes != plan.weight_bytes) {
      throw std::invalid_argument(
          "materialized model does not match Qwen Program plan");
    }
    for (const auto object_id : plan.required_object_ids) {
      if (!weights.has_object(object_id)) {
        throw std::invalid_argument(
            "materialized model is missing a Qwen Program object");
      }
    }

    parameters.reserve(profiles.profiles().size());
    for (const auto &profile : profiles.profiles()) {
      parameters.push_back(context.make_parameter_table(profile.parameters));
    }

    residency =
        context.make_residency_set("b4.qwen-program." + plan.topology_signature,
                                   plan.residency_allocation_count);
    for (std::uint32_t index = 0; index < weights.segment_count(); ++index) {
      context.residency_add_buffer(residency, weights.segment(index));
    }
    context.residency_add_buffer(residency, workspace.buffer());
    context.residency_add_buffer(residency, state.buffer());
    for (const auto &table : parameters) {
      context.residency_add_parameters(residency, table);
    }
    context.residency_commit(residency);
    if (residency.allocation_count() != plan.residency_allocation_count) {
      throw std::runtime_error("Qwen Program residency ledger mismatch");
    }

    const auto embedding = std::ranges::find_if(
        layers.global_stages(), [](const LayerStageRecord &stage) {
          return stage.role == "embedding";
        });
    if (embedding == layers.global_stages().end()) {
      throw std::invalid_argument("Qwen Program has no embedding stage");
    }
    append_stage_plan(stage_plans, weights, UINT32_MAX, *embedding, stats);
    for (const auto &layer : layers.layers()) {
      for (const auto &stage : layer.stages) {
        append_stage_plan(stage_plans, weights, layer.layer, stage, stats);
      }
    }
    for (const auto &stage : layers.global_stages()) {
      if (stage.role != "embedding") {
        append_stage_plan(stage_plans, weights, UINT32_MAX, stage, stats);
      }
    }
    if (stage_plans.size() != plan.stage_count) {
      throw std::invalid_argument("Qwen Program stage plan count mismatch");
    }
    stats.static_stage_plan_count = stage_plans.size();
    stats.profile_count = profiles.profiles().size();
    stats.parameter_table_count = parameters.size();
    stats.residency_allocations = residency.allocation_count();
    stats.residency_bytes = residency.allocated_size();
  }

  QwenProgramBuildPlan plan;
  MetalContext context;
  MaterializedModel weights;
  QwenLayerCatalog layers;
  DispatchTable dispatch;
  TopologyProfileCatalog profiles;
  WorkspaceArena workspace;
  StateArena state;
  std::vector<MetalParameterTable> parameters;
  MetalResidencySet residency;
  std::vector<QwenProgramStagePlan> stage_plans;
  QwenSequenceRuntimeState sequence;
  QwenProgramStats stats;
  std::unique_ptr<QwenArExecutor> ar;
};

QwenProgram::QwenProgram(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
QwenProgram::QwenProgram(QwenProgram &&) noexcept = default;
QwenProgram &QwenProgram::operator=(QwenProgram &&) noexcept = default;
QwenProgram::~QwenProgram() = default;

const QwenProgramBuildPlan &QwenProgram::build_plan() const {
  return impl_->plan;
}

const MaterializedModel &QwenProgram::weights() const { return impl_->weights; }
const QwenLayerCatalog &QwenProgram::layers() const { return impl_->layers; }
const DispatchTable &QwenProgram::dispatch_table() const {
  return impl_->dispatch;
}
const TopologyProfileCatalog &QwenProgram::profiles() const {
  return impl_->profiles;
}
const std::vector<QwenProgramStagePlan> &QwenProgram::stage_plans() const {
  return impl_->stage_plans;
}
const QwenProgramStats &QwenProgram::stats() const { return impl_->stats; }
const QwenSequenceRuntimeState &QwenProgram::sequence_state() const {
  return impl_->sequence;
}
const MetalResidencySet &QwenProgram::residency() const {
  return impl_->residency;
}

const MetalParameterTable &
QwenProgram::parameter_table(std::uint32_t profile_index) const {
  if (profile_index >= impl_->parameters.size()) {
    throw std::out_of_range("Qwen Program parameter table index is invalid");
  }
  return impl_->parameters[profile_index];
}

TopologyProfileSelection
QwenProgram::select_profile(TopologyProfileMode mode, std::uint32_t batch_size,
                            std::uint32_t tokens) const {
  return impl_->profiles.select(
      {.mode = mode,
       .architecture = impl_->plan.architecture,
       .quant_profile = impl_->plan.quant_profile,
       .batch_size = batch_size,
       .tokens = tokens,
       .topology_signature = impl_->plan.topology_signature});
}

void QwenProgram::update_parameters(std::uint32_t profile_index,
                                    const QwenDynamicStep &step) {
  if (profile_index >= impl_->parameters.size()) {
    throw std::out_of_range("Qwen Program profile index is invalid");
  }
  const auto &profile = impl_->profiles.profile(profile_index);
  if (step.token_ids.empty() ||
      step.token_ids.size() != step.positions.size() ||
      step.token_ids.size() > profile.key.token_capacity ||
      (profile.key.mode == TopologyProfileMode::decode &&
       step.token_ids.size() != 1)) {
    throw std::invalid_argument("dynamic token rows do not match profile");
  }
  auto &table = impl_->parameters[profile_index];
  const auto write = [&](std::string_view key, auto values) {
    const auto &descriptor = field(profile, key);
    table.write(key, padded_bytes(values, descriptor.size, key));
  };
  write("token_ids", step.token_ids);
  write("positions", step.positions);
  write("kv_frontier", std::span(&step.kv_frontier, 1));
  const auto page_count = static_cast<std::uint32_t>(step.page_table.size());
  write("page_count", std::span(&page_count, 1));
  write("page_table", step.page_table);
  if (profile.key.architecture == QwenArchitecture::moe) {
    if (step.route_indices.size() != step.route_scores.size() ||
        step.route_indices.size() !=
            step.token_ids.size() * impl_->profiles.topology().top_k) {
      throw std::invalid_argument("dynamic MoE route geometry is invalid");
    }
    write("route_indices", step.route_indices);
    write("route_scores", step.route_scores);
  } else if (!step.route_indices.empty() || !step.route_scores.empty()) {
    throw std::invalid_argument("dense profile received MoE route data");
  }
  if (!step.valid_mask.empty() &&
      step.valid_mask.size() != step.token_ids.size() &&
      step.valid_mask.size() != profile.key.token_capacity) {
    throw std::invalid_argument("dynamic valid-mask geometry is invalid");
  }
  if (step.valid_mask.empty()) {
    std::vector<std::uint32_t> valid(step.token_ids.size(), 1);
    write("valid_mask", std::span<const std::uint32_t>(valid));
  } else {
    write("valid_mask", step.valid_mask);
  }
  write("sampling", std::span<const float>(step.sampling));
  write("kernel_params", std::span<const std::uint32_t>(step.kernel_params));
  impl_->sequence.position = step.positions.back();
  impl_->sequence.kv_frontier = step.kv_frontier;
  impl_->sequence.page_count = page_count;
  if (profile.key.mode == TopologyProfileMode::decode) {
    ++impl_->sequence.generated_tokens;
  }
  impl_->sequence.parameter_generation = table.generation();
}

QwenProgramAddressSnapshot QwenProgram::address_snapshot() const {
  QwenProgramAddressSnapshot result;
  result.weight_segments.reserve(impl_->weights.segment_count());
  for (std::uint32_t index = 0; index < impl_->weights.segment_count();
       ++index) {
    result.weight_segments.push_back(
        impl_->weights.segment(index).gpu_address());
  }
  result.workspace = impl_->workspace.buffer().gpu_address();
  result.state = impl_->state.buffer().gpu_address();
  result.parameter_tables.reserve(impl_->parameters.size());
  for (const auto &table : impl_->parameters) {
    result.parameter_tables.push_back(table.gpu_address());
  }
  return result;
}

MetalRegistryStats QwenProgram::registry_stats() const {
  return impl_->context.registry_stats();
}

std::string_view to_string(QwenArExecutionRoute route) {
  switch (route) {
  case QwenArExecutionRoute::execute_batch:
    return "execute_batch";
  case QwenArExecutionRoute::icb_profile:
    return "icb_profile";
  }
  throw std::invalid_argument("unknown Qwen AR execution route");
}

void QwenProgram::install_autoregressive_executor(
    const std::filesystem::path &metallib) {
  if (impl_->ar != nullptr) {
    throw std::logic_error("Qwen AR executor is already installed");
  }
  if (impl_->plan.hidden_size == 0 || impl_->plan.vocabulary_size == 0 ||
      impl_->plan.vocabulary_size > 64 || impl_->plan.max_context == 0 ||
      (impl_->plan.architecture == QwenArchitecture::moe &&
       (impl_->plan.expert_count == 0 || impl_->plan.expert_top_k == 0))) {
    throw std::invalid_argument("Qwen AR fixture geometry is unsupported");
  }
  auto executor = std::make_unique<QwenArExecutor>();
  executor->model_seed = static_cast<std::uint32_t>(
      std::stoull(impl_->plan.topology_signature.substr(0, 8), nullptr, 16));
  const auto maximum_tokens = impl_->profiles.stats().maximum_token_capacity;
  const auto hidden_values = checked_multiply(
      maximum_tokens, impl_->plan.hidden_size, "Qwen AR hidden values");
  const auto state_values =
      checked_add(checked_multiply(impl_->plan.max_context,
                                   impl_->plan.hidden_size, "Qwen AR KV state"),
                  impl_->plan.hidden_size, "Qwen AR recurrent state");
  const auto logit_values = checked_multiply(
      maximum_tokens, impl_->plan.vocabulary_size, "Qwen AR logits");
  executor->hidden = impl_->context.make_buffer(
      checked_multiply(hidden_values, 4, "Qwen AR hidden bytes"));
  executor->state = impl_->context.make_buffer(
      checked_multiply(state_values, 4, "Qwen AR state bytes"));
  executor->logits = impl_->context.make_buffer(
      checked_multiply(logit_values, 4, "Qwen AR logit bytes"));
  executor->sampled = impl_->context.make_buffer(
      checked_multiply(maximum_tokens, 4, "Qwen AR sampled bytes"));
  executor->oracle_state.resize(state_values);

  const auto embedding_stage = std::ranges::find_if(
      impl_->stage_plans, [](const QwenProgramStagePlan &stage) {
        return stage.role == "embedding" && stage.object_id != UINT32_MAX;
      });
  if (embedding_stage == impl_->stage_plans.end()) {
    throw std::invalid_argument("Qwen AR embedding weight is absent");
  }
  const auto embedding_weight =
      impl_->weights.parent_view(embedding_stage->object_id);
  const auto weight_plane =
      [&](std::string_view kind) -> const MaterializedPlaneView & {
    const auto found = std::ranges::find_if(
        embedding_weight.planes,
        [&](const MaterializedPlaneView &plane) { return plane.kind == kind; });
    if (found == embedding_weight.planes.end() || found->buffer == nullptr) {
      throw std::invalid_argument("Qwen AR weight plane is absent: " +
                                  std::string(kind));
    }
    return *found;
  };
  if (impl_->plan.quant_profile == QwenQuantProfile::oq_affine) {
    const auto &codes =
        weight_plane(embedding_weight.format.find("_q8_") != std::string::npos
                         ? "code"
                         : "base_code");
    const auto &scales = weight_plane("scale");
    const auto &biases = weight_plane("bias");
    executor->weight_codes = codes.buffer;
    executor->weight_codes_offset = codes.offset;
    executor->weight_scales = scales.buffer;
    executor->weight_scales_offset = scales.offset;
    executor->weight_biases = biases.buffer;
    executor->weight_biases_offset = biases.offset;
    const auto code = std::to_integer<std::uint8_t>(
                          codes.buffer->read(codes.offset, 1).front()) &
                      0x0FU;
    const auto scale_bytes = scales.buffer->read(scales.offset, 2);
    const auto bias_bytes = biases.buffer->read(biases.offset, 2);
    std::uint16_t scale_bits = 0;
    std::uint16_t bias_bits = 0;
    std::memcpy(&scale_bits, scale_bytes.data(), sizeof(scale_bits));
    std::memcpy(&bias_bits, bias_bytes.data(), sizeof(bias_bits));
    executor->weight_factor =
        (static_cast<float>(code) * decode_bf16(scale_bits) +
         decode_bf16(bias_bits)) *
        (1.0F / 16.0F);
  } else {
    const auto &codes = weight_plane("data");
    const auto &scales = weight_plane("scale");
    executor->weight_codes = codes.buffer;
    executor->weight_codes_offset = codes.offset;
    executor->weight_scales = scales.buffer;
    executor->weight_scales_offset = scales.offset;
    executor->weight_biases = scales.buffer;
    executor->weight_biases_offset = scales.offset;
    const auto code = std::to_integer<std::uint8_t>(
                          codes.buffer->read(codes.offset, 1).front()) &
                      0x0FU;
    const auto scale = std::to_integer<std::uint8_t>(
        scales.buffer->read(scales.offset, 1).front());
    executor->weight_factor = decode_e2m1(code) * decode_e8m0(scale);
  }

  impl_->context.register_library(
      {.key = "b4.qwen_ar.library", .path = metallib});
  const auto warm = [&](std::string key, std::string function) {
    static_cast<void>(impl_->context.warmup_pipeline({
        .key = std::move(key),
        .library_key = "b4.qwen_ar.library",
        .function_name = std::move(function),
        .variant = "icb-ready",
        .support_indirect_command_buffers = true,
    }));
  };
  warm("b4.qwen_ar.embed", "omlx_m5_qwen_ar_embed");
  warm("b4.qwen_ar.dense", "omlx_m5_qwen_ar_dense_layer");
  warm("b4.qwen_ar.moe", "omlx_m5_qwen_ar_moe_layer");
  warm("b4.qwen_ar.logits", "omlx_m5_qwen_ar_logits");
  warm("b4.qwen_ar.sample", "omlx_m5_qwen_ar_sample");

  executor->profiles.reserve(impl_->profiles.profiles().size());
  for (const auto &profile : impl_->profiles.profiles()) {
    const auto dynamic = [&](std::uint32_t argument_index,
                             std::string_view key) {
      const auto &descriptor = field(profile, key);
      return MetalDynamicArgument{.argument_index = argument_index,
                                  .parameter_offset = descriptor.offset,
                                  .minimum_bytes = descriptor.size};
    };
    QwenArConfig config{
        .capacity = profile.key.token_capacity,
        .hidden_size = impl_->plan.hidden_size,
        .vocabulary_size = impl_->plan.vocabulary_size,
        .max_context = impl_->plan.max_context,
        .architecture = impl_->plan.architecture == QwenArchitecture::moe,
        .quant_profile = impl_->plan.quant_profile == QwenQuantProfile::mxfp4,
        .expert_top_k = impl_->plan.expert_top_k,
        .expert_count = impl_->plan.expert_count,
        .full_attention_layers = impl_->plan.full_attention_layers,
        .gdn_layers = impl_->plan.gdn_layers,
        .layer_count = impl_->plan.layer_count,
        .model_seed = executor->model_seed,
        .weight_factor_bits =
            std::bit_cast<std::uint32_t>(executor->weight_factor),
    };
    const auto hidden_bytes =
        checked_multiply(checked_multiply(config.capacity, config.hidden_size,
                                          "Qwen AR profile hidden"),
                         4, "Qwen AR profile hidden bytes");
    const auto logits_bytes = checked_multiply(
        checked_multiply(config.capacity, config.vocabulary_size,
                         "Qwen AR profile logits"),
        4, "Qwen AR profile logit bytes");
    const auto sampled_bytes =
        checked_multiply(config.capacity, 4, "Qwen AR profile sampled bytes");
    const auto state_bytes = executor->state.size();
    MetalDispatchDescriptor layer{
        .pipeline_key = impl_->plan.architecture == QwenArchitecture::moe
                            ? "b4.qwen_ar.moe"
                            : "b4.qwen_ar.dense",
        .grid = {.width = config.hidden_size},
        .threads_per_threadgroup = {.width = 64},
        .resources =
            {
                {.argument_index = 0,
                 .resource_slot = 0,
                 .minimum_bytes = hidden_bytes},
                {.argument_index = 1,
                 .resource_slot = 1,
                 .minimum_bytes = state_bytes},
                {.argument_index = 2, .resource_slot = 4, .minimum_bytes = 1},
                {.argument_index = 3, .resource_slot = 5, .minimum_bytes = 2},
                {.argument_index = 4, .resource_slot = 6, .minimum_bytes = 2},
            },
        .dynamic_arguments =
            {
                dynamic(5, "positions"),
            },
    };
    std::uint32_t layer_config_index = 7;
    if (impl_->plan.architecture == QwenArchitecture::moe) {
      layer.dynamic_arguments.push_back(dynamic(6, "route_indices"));
      layer.dynamic_arguments.push_back(dynamic(7, "route_scores"));
      layer.dynamic_arguments.push_back(dynamic(8, "valid_mask"));
      layer_config_index = 9;
    } else {
      layer.dynamic_arguments.push_back(dynamic(6, "valid_mask"));
    }
    layer.constant_arguments = {{.argument_index = layer_config_index,
                                 .bytes = constant_bytes(config)}};
    std::vector<MetalDispatchDescriptor> dispatches = {
        {
            .pipeline_key = "b4.qwen_ar.embed",
            .barrier_before = false,
            .grid = {.width =
                         checked_multiply(config.capacity, config.hidden_size,
                                          "Qwen AR embed grid")},
            .threads_per_threadgroup = {.width = 128},
            .resources = {{.argument_index = 3,
                           .resource_slot = 0,
                           .minimum_bytes = hidden_bytes}},
            .constant_arguments = {{.argument_index = 4,
                                    .bytes = constant_bytes(config)}},
            .dynamic_arguments = {dynamic(0, "token_ids"),
                                  dynamic(1, "positions"),
                                  dynamic(2, "valid_mask")},
        },
        std::move(layer),
        {
            .pipeline_key = "b4.qwen_ar.logits",
            .grid = {.width = checked_multiply(config.capacity,
                                               config.vocabulary_size,
                                               "Qwen AR logits grid")},
            .threads_per_threadgroup = {.width = 128},
            .resources =
                {
                    {.argument_index = 0,
                     .resource_slot = 0,
                     .minimum_bytes = hidden_bytes},
                    {.argument_index = 1,
                     .resource_slot = 2,
                     .minimum_bytes = logits_bytes},
                },
            .constant_arguments = {{.argument_index = 3,
                                    .bytes = constant_bytes(config)}},
            .dynamic_arguments = {dynamic(2, "valid_mask")},
        },
        {
            .pipeline_key = "b4.qwen_ar.sample",
            .grid = {.width = config.capacity},
            .threads_per_threadgroup = {.width = 64},
            .resources =
                {
                    {.argument_index = 0,
                     .resource_slot = 2,
                     .minimum_bytes = logits_bytes},
                    {.argument_index = 1,
                     .resource_slot = 3,
                     .minimum_bytes = sampled_bytes},
                },
            .constant_arguments = {{.argument_index = 5,
                                    .bytes = constant_bytes(config)}},
            .dynamic_arguments = {dynamic(2, "positions"),
                                  dynamic(3, "valid_mask"),
                                  dynamic(4, "sampling")},
        },
    };
    auto plan = impl_->context.prepare_command_plan({
        .key = "b4.qwen_ar.plan." + profile.name,
        .icb_qualified = true,
        .dispatches = std::move(dispatches),
        .ranges =
            {
                {.key = "full", .location = 0, .length = 4},
                {.key = "embedding", .location = 0, .length = 1},
                {.key = "model", .location = 1, .length = 2},
                {.key = "sampling", .location = 3, .length = 1},
            },
    });
    executor->profiles.push_back({.plan = std::move(plan), .config = config});
  }

  const std::array bindings = {
      MetalBufferBinding{0, &executor->hidden, 0},
      MetalBufferBinding{1, &executor->state, 0},
      MetalBufferBinding{2, &executor->logits, 0},
      MetalBufferBinding{3, &executor->sampled, 0},
      MetalBufferBinding{4, executor->weight_codes,
                         executor->weight_codes_offset},
      MetalBufferBinding{5, executor->weight_scales,
                         executor->weight_scales_offset},
      MetalBufferBinding{6, executor->weight_biases,
                         executor->weight_biases_offset},
  };
  for (std::uint32_t index = 0; index < executor->profiles.size(); ++index) {
    auto &runtime = executor->profiles[index];
    runtime.icb.emplace(impl_->context.prepare_icb_profile(
        "b4.qwen_ar.icb." + impl_->profiles.profile(index).name,
        {.plan = &runtime.plan,
         .bindings = {bindings.begin(), bindings.end()},
         .parameters = &impl_->parameters[index]}));
    executor->stats.icb_recordings += runtime.icb->stats().recording_count;
  }
  impl_->context.seal_warmup();
  executor->stats.installed = true;
  executor->stats.icb_installed = !executor->profiles.empty();
  executor->stats.profile_plan_count = executor->profiles.size();
  impl_->ar = std::move(executor);
  reset_autoregressive();
}

bool QwenProgram::autoregressive_installed() const {
  return impl_->ar != nullptr && impl_->ar->stats.installed;
}

const QwenArStats &QwenProgram::autoregressive_stats() const {
  if (!autoregressive_installed()) {
    throw std::logic_error("Qwen AR executor is not installed");
  }
  return impl_->ar->stats;
}

QwenArAddressSnapshot QwenProgram::autoregressive_address_snapshot() const {
  if (!autoregressive_installed()) {
    throw std::logic_error("Qwen AR executor is not installed");
  }
  return {.hidden = impl_->ar->hidden.gpu_address(),
          .state = impl_->ar->state.gpu_address(),
          .logits = impl_->ar->logits.gpu_address(),
          .sampled = impl_->ar->sampled.gpu_address()};
}

QwenArCheckpoint QwenProgram::checkpoint_autoregressive() const {
  if (!autoregressive_installed()) {
    throw std::logic_error("Qwen AR executor is not installed");
  }
  return {.state = impl_->ar->state.read(),
          .oracle_state = impl_->ar->oracle_state,
          .sequence = impl_->sequence,
          .next_position = impl_->ar->next_position};
}

void QwenProgram::restore_autoregressive(const QwenArCheckpoint &checkpoint) {
  if (!autoregressive_installed() ||
      checkpoint.state.size() != impl_->ar->state.size() ||
      checkpoint.oracle_state.size() != impl_->ar->oracle_state.size() ||
      checkpoint.next_position > impl_->plan.max_context) {
    throw std::invalid_argument("Qwen AR checkpoint is incompatible");
  }
  impl_->ar->state.write(checkpoint.state);
  impl_->ar->oracle_state = checkpoint.oracle_state;
  impl_->ar->next_position = checkpoint.next_position;
  impl_->sequence = checkpoint.sequence;
  impl_->ar->cancel_requested.store(false);
}

QwenArStepResult
QwenProgram::execute_ar(std::span<const std::uint32_t> tokens,
                        TopologyProfileMode mode, QwenArExecutionRoute route,
                        const QwenArSamplingOptions &sampling) {
  if (!autoregressive_installed() || tokens.empty() ||
      tokens.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Qwen AR execution request is invalid");
  }
  auto &executor = *impl_->ar;
  if (executor.cancel_requested.load()) {
    throw std::runtime_error("Qwen AR execution cancelled");
  }
  if (executor.failure_after.has_value() &&
      executor.completed_steps >= *executor.failure_after) {
    throw std::runtime_error("injected Qwen AR execution failure");
  }
  if (executor.next_position + tokens.size() > impl_->plan.max_context) {
    throw std::out_of_range("Qwen AR context capacity exceeded");
  }
  if (sampling.top_k > 64 || !std::isfinite(sampling.temperature) ||
      sampling.temperature < 0.0F || !std::isfinite(sampling.top_p) ||
      sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
    throw std::invalid_argument("Qwen AR sampling options are invalid");
  }
  const auto selected =
      select_profile(mode, 1, static_cast<std::uint32_t>(tokens.size()));
  if (!selected.matched()) {
    throw std::out_of_range("Qwen AR shape has no compiled profile: " +
                            std::string(to_string(selected.reason)));
  }
  const auto &profile = impl_->profiles.profile(selected.profile_index);
  auto &runtime = executor.profiles.at(selected.profile_index);
  std::vector<std::uint32_t> positions(tokens.size());
  for (std::uint32_t index = 0; index < positions.size(); ++index) {
    positions[index] = executor.next_position + index;
  }
  const auto frontier = executor.next_position + tokens.size();
  const auto page_count =
      (frontier + impl_->plan.kv_page_tokens - 1) / impl_->plan.kv_page_tokens;
  std::vector<std::uint32_t> pages(page_count);
  for (std::uint32_t index = 0; index < page_count; ++index) {
    pages[index] = index;
  }
  std::vector<std::uint32_t> routes;
  std::vector<float> route_scores;
  if (impl_->plan.architecture == QwenArchitecture::moe) {
    routes.resize(tokens.size() * impl_->plan.expert_top_k);
    route_scores.resize(routes.size(),
                        1.0F / static_cast<float>(impl_->plan.expert_top_k));
    for (std::size_t row = 0; row < tokens.size(); ++row) {
      for (std::uint32_t expert = 0; expert < impl_->plan.expert_top_k;
           ++expert) {
        routes[row * impl_->plan.expert_top_k + expert] =
            (tokens[row] + expert) % impl_->plan.expert_count;
      }
    }
  }
  std::vector<std::uint32_t> mask(tokens.size(), 1);
  update_parameters(selected.profile_index,
                    {.token_ids = tokens,
                     .positions = positions,
                     .kv_frontier = static_cast<std::uint32_t>(frontier),
                     .page_table = pages,
                     .route_indices = routes,
                     .route_scores = route_scores,
                     .valid_mask = mask});
  const QwenArSampling sampling_parameters{
      .temperature = sampling.temperature,
      .top_p = sampling.top_p,
      .top_k = sampling.top_k,
      .seed = sampling.seed,
  };
  impl_->parameters[selected.profile_index].write(
      "sampling", constant_bytes(sampling_parameters));

  const std::array bindings = {
      MetalBufferBinding{0, &executor.hidden, 0},
      MetalBufferBinding{1, &executor.state, 0},
      MetalBufferBinding{2, &executor.logits, 0},
      MetalBufferBinding{3, &executor.sampled, 0},
      MetalBufferBinding{4, executor.weight_codes,
                         executor.weight_codes_offset},
      MetalBufferBinding{5, executor.weight_scales,
                         executor.weight_scales_offset},
      MetalBufferBinding{6, executor.weight_biases,
                         executor.weight_biases_offset},
  };
  QwenArStepResult result{
      .input_tokens = static_cast<std::uint32_t>(tokens.size()),
      .profile_index = selected.profile_index,
      .profile_capacity = profile.key.token_capacity,
      .route = route,
  };
  if (executor.last_profile != UINT32_MAX &&
      executor.last_profile != selected.profile_index) {
    ++executor.stats.profile_switches;
  }
  executor.last_profile = selected.profile_index;
  const auto registry_before = impl_->context.registry_stats();
  if (route == QwenArExecutionRoute::icb_profile && runtime.icb.has_value()) {
    result.submission = impl_->context.replay_icb(*runtime.icb, "full");
    ++executor.stats.icb_steps;
    ++executor.stats.icb_replays;
  } else {
    if (route == QwenArExecutionRoute::icb_profile) {
      result.fallback = true;
      result.fallback_reason = "icb_profile_unavailable";
      result.route = QwenArExecutionRoute::execute_batch;
      ++executor.stats.fallback_steps;
    }
    const std::array invocation = {MetalCommandInvocation{
        .plan = &runtime.plan,
        .bindings = {bindings.begin(), bindings.end()},
        .parameters = &impl_->parameters[selected.profile_index],
    }};
    result.submission = impl_->context.execute_batch(invocation);
    ++executor.stats.batch_steps;
  }
  const auto registry_after = impl_->context.registry_stats();
  if (registry_after.command_plans_built !=
      registry_before.command_plans_built) {
    ++executor.stats.token_loop_plan_builds;
  }
  executor.stats.token_loop_pipeline_compiles +=
      registry_after.pipelines_compiled - registry_before.pipelines_compiled;
  executor.stats.token_loop_weight_uploads +=
      registry_after.upload_count - registry_before.upload_count;

  const auto &config = runtime.config;
  std::vector<float> expected_hidden(static_cast<std::size_t>(config.capacity) *
                                     config.hidden_size);
  for (std::uint32_t row = 0; row < tokens.size(); ++row) {
    for (std::uint32_t feature = 0; feature < config.hidden_size; ++feature) {
      const auto code = (tokens[row] * 17U + positions[row] * 13U +
                         feature * 7U + config.model_seed) %
                        257U;
      expected_hidden[row * config.hidden_size + feature] =
          static_cast<float>(static_cast<std::int32_t>(code) - 128) *
          (1.0F / 128.0F);
    }
  }
  const auto recurrent_base =
      static_cast<std::size_t>(config.max_context) * config.hidden_size;
  const auto quant_scale = config.quant_profile == 0 ? 0.5F : 0.75F;
  for (std::uint32_t feature = 0; feature < config.hidden_size; ++feature) {
    auto recurrent = config.gdn_layers == 0
                         ? 0.0F
                         : executor.oracle_state[recurrent_base + feature];
    const auto layer_bias =
        static_cast<float>((config.model_seed + feature + config.layer_count) %
                           17U) *
        (1.0F / 256.0F);
    for (std::uint32_t row = 0; row < tokens.size(); ++row) {
      float route_value = 0.0F;
      if (config.architecture != 0) {
        for (std::uint32_t expert = 0; expert < config.expert_top_k; ++expert) {
          const auto route_index = row * config.expert_top_k + expert;
          route_value += static_cast<float>(routes[route_index] + 1U) *
                         route_scores[route_index] * (1.0F / 64.0F);
        }
      }
      const auto hidden_index = row * config.hidden_size + feature;
      const auto value = expected_hidden[hidden_index] * quant_scale +
                         recurrent * 0.25F + route_value * 0.125F + layer_bias;
      expected_hidden[hidden_index] =
          value + executor.weight_factor * (1.0F / 32.0F);
      const auto final_value = expected_hidden[hidden_index];
      if (config.full_attention_layers != 0) {
        const auto state_index =
            (positions[row] % config.max_context) * config.hidden_size +
            feature;
        executor.oracle_state[state_index] = final_value;
      }
      if (config.gdn_layers != 0) {
        recurrent = final_value;
        executor.oracle_state[recurrent_base + feature] = final_value;
      }
    }
  }
  std::vector<float> expected_logits(static_cast<std::size_t>(config.capacity) *
                                     config.vocabulary_size);
  std::vector<std::uint32_t> expected_sampled(config.capacity);
  for (std::uint32_t row = 0; row < tokens.size(); ++row) {
    for (std::uint32_t token = 0; token < config.vocabulary_size; ++token) {
      float value = 0.0F;
      for (std::uint32_t feature = 0; feature < config.hidden_size; ++feature) {
        const auto coefficient =
            static_cast<std::int32_t>(
                (token * 11U + feature * 5U + config.model_seed) % 17U) -
            8;
        value += expected_hidden[row * config.hidden_size + feature] *
                 static_cast<float>(coefficient) * (1.0F / 64.0F);
      }
      expected_logits[row * config.vocabulary_size + token] = value;
    }
    expected_sampled[row] = sample_reference(
        std::span(expected_logits)
            .subspan(row * config.vocabulary_size, config.vocabulary_size),
        positions[row], sampling, config.model_seed);
  }

  const auto decode_floats = [](const std::vector<std::byte> &bytes) {
    if (bytes.size() % sizeof(float) != 0) {
      throw std::runtime_error("Qwen AR float buffer is malformed");
    }
    std::vector<float> values(bytes.size() / sizeof(float));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
  };
  const auto actual_hidden = decode_floats(
      executor.hidden.read(0, expected_hidden.size() * sizeof(float)));
  const auto actual_logits = decode_floats(
      executor.logits.read(0, expected_logits.size() * sizeof(float)));
  const auto actual_state_bytes = executor.state.read();
  const auto actual_state = decode_floats(actual_state_bytes);
  const auto sampled_bytes = executor.sampled.read(
      0, static_cast<std::uint64_t>(config.capacity) * sizeof(std::uint32_t));
  std::vector<std::uint32_t> actual_sampled(config.capacity);
  std::memcpy(actual_sampled.data(), sampled_bytes.data(),
              sampled_bytes.size());
  float maximum_error = 0.0F;
  const auto compare = [&](std::span<const float> actual,
                           std::span<const float> expected) {
    if (actual.size() != expected.size()) {
      throw std::runtime_error("Qwen AR oracle shape mismatch");
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
      maximum_error =
          std::max(maximum_error, std::abs(actual[index] - expected[index]));
    }
  };
  compare(std::span(actual_hidden).first(tokens.size() * config.hidden_size),
          std::span(expected_hidden).first(tokens.size() * config.hidden_size));
  compare(
      std::span(actual_logits).first(tokens.size() * config.vocabulary_size),
      std::span(expected_logits).first(tokens.size() * config.vocabulary_size));
  compare(actual_state, executor.oracle_state);
  const auto samples_match = std::equal(
      expected_sampled.begin(), expected_sampled.begin() + tokens.size(),
      actual_sampled.begin());
  if (maximum_error > 2.0e-5F || !samples_match) {
    throw std::runtime_error(
        "Qwen AR Metal/CPU oracle mismatch (max_error=" +
        std::to_string(maximum_error) + ", expected_token=" +
        std::to_string(expected_sampled[tokens.size() - 1]) +
        ", actual_token=" + std::to_string(actual_sampled[tokens.size() - 1]) +
        ")");
  }
  result.sampled_token = actual_sampled[tokens.size() - 1];
  result.sampled_tokens.assign(actual_sampled.begin(),
                               actual_sampled.begin() + tokens.size());
  result.hidden.assign(actual_hidden.begin(),
                       actual_hidden.begin() +
                           tokens.size() * config.hidden_size);
  result.logits.assign(actual_logits.begin(),
                       actual_logits.begin() +
                           tokens.size() * config.vocabulary_size);
  result.oracle_max_abs_error = maximum_error;
  const auto kv_bytes =
      checked_multiply(checked_multiply(config.max_context, config.hidden_size,
                                        "Qwen AR KV checksum"),
                       4, "Qwen AR KV checksum bytes");
  result.kv_checksum =
      byte_checksum(std::span(actual_state_bytes).first(kv_bytes));
  result.gdn_checksum =
      byte_checksum(std::span(actual_state_bytes).subspan(kv_bytes));
  executor.next_position = static_cast<std::uint32_t>(frontier);
  impl_->sequence.position = executor.next_position;
  impl_->sequence.kv_frontier = executor.next_position;
  impl_->sequence.page_count = page_count;
  ++executor.completed_steps;
  return result;
}

QwenArStepResult QwenProgram::prefill(std::span<const std::uint32_t> prompt,
                                      const QwenArSamplingOptions &sampling) {
  if (prompt.empty()) {
    throw std::invalid_argument("Qwen AR prompt is empty");
  }
  QwenArStepResult result;
  std::size_t offset = 0;
  const auto maximum = impl_->profiles.stats().maximum_token_capacity;
  while (offset < prompt.size()) {
    const auto count = std::min<std::size_t>(maximum, prompt.size() - offset);
    const auto selected_mode =
        count == 1 ? TopologyProfileMode::decode : TopologyProfileMode::prefill;
    result = execute_ar(prompt.subspan(offset, count), selected_mode,
                        QwenArExecutionRoute::execute_batch, sampling);
    offset += count;
  }
  return result;
}

QwenArRunResult QwenProgram::generate(std::span<const std::uint32_t> prompt,
                                      std::uint32_t maximum_new_tokens,
                                      const QwenArSamplingOptions &sampling,
                                      const std::function<bool()> &cancelled) {
  QwenArRunResult result;
  try {
    reset_autoregressive();
    if (maximum_new_tokens == 0) {
      return result;
    }
    auto step = prefill(prompt, sampling);
    auto token = step.sampled_token;
    for (std::uint32_t index = 0; index < maximum_new_tokens; ++index) {
      if (impl_->ar->cancel_requested.load() || (cancelled && cancelled())) {
        result.cancelled = true;
        break;
      }
      result.events.push_back({
          .index = index,
          .token = token,
          .position = static_cast<std::uint32_t>(prompt.size() + index),
          .route = std::string(to_string(step.route)),
      });
      impl_->sequence.generated_tokens = result.events.size();
      if (index + 1 == maximum_new_tokens) {
        break;
      }
      const std::array input = {token};
      step = execute_ar(input, TopologyProfileMode::decode,
                        QwenArExecutionRoute::icb_profile, sampling);
      token = step.sampled_token;
    }
  } catch (const std::exception &error) {
    result.failed = true;
    result.error = error.what();
  }
  return result;
}

void QwenProgram::reset_autoregressive() {
  if (!autoregressive_installed()) {
    throw std::logic_error("Qwen AR executor is not installed");
  }
  auto &executor = *impl_->ar;
  executor.hidden.write(std::vector<std::byte>(executor.hidden.size()));
  executor.state.write(std::vector<std::byte>(executor.state.size()));
  executor.logits.write(std::vector<std::byte>(executor.logits.size()));
  executor.sampled.write(std::vector<std::byte>(executor.sampled.size()));
  std::ranges::fill(executor.oracle_state, 0.0F);
  executor.next_position = 0;
  executor.last_profile = UINT32_MAX;
  executor.completed_steps = 0;
  executor.cancel_requested.store(false);
  impl_->sequence = {};
}

void QwenProgram::request_cancel() {
  if (!autoregressive_installed()) {
    throw std::logic_error("Qwen AR executor is not installed");
  }
  impl_->ar->cancel_requested.store(true);
}

void QwenProgram::inject_failure_after(std::uint64_t completed_steps) {
  if (!autoregressive_installed()) {
    throw std::logic_error("Qwen AR executor is not installed");
  }
  impl_->ar->failure_after = completed_steps;
}

void QwenProgram::clear_failure_injection() {
  if (!autoregressive_installed()) {
    throw std::logic_error("Qwen AR executor is not installed");
  }
  impl_->ar->failure_after.reset();
}

QwenProgram assemble_qwen_program(QwenProgramBuildPlan plan,
                                  MetalContext &&context,
                                  MaterializedModel &&weights,
                                  QwenLayerCatalog layers,
                                  DispatchTable &&dispatch_table,
                                  TopologyProfileCatalog profiles) {
  return QwenProgram(std::make_unique<QwenProgram::Impl>(
      std::move(plan), std::move(context), std::move(weights),
      std::move(layers), std::move(dispatch_table), std::move(profiles)));
}

} // namespace omlx::m5
