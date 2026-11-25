#include "omlx_m5/dflash2.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace omlx::m5 {
namespace {

using Json = nlohmann::json;

struct DFlash2MetalConfig {
  std::uint32_t hidden_size = 0;
  std::uint32_t intermediate_size = 0;
  std::uint32_t vocabulary_size = 0;
  std::uint32_t layer_count = 0;
  std::uint32_t query_heads = 0;
  std::uint32_t key_value_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t target_taps = 0;
  std::uint32_t context_tokens = 0;
  std::uint32_t width = 0;
  std::uint32_t conv_group_size = 0;
  std::uint32_t conv_kernel_size = 0;
  std::uint32_t selector_rank = 0;
  std::uint32_t selector_top_k = 0;
  std::uint32_t sliding_window = 0;
  std::uint32_t mask_token_id = 0;
  float epsilon = 0.0F;
  float rope_theta = 0.0F;
};

static_assert(sizeof(DFlash2MetalConfig) == 72);

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

template <class T> std::vector<std::byte> bytes_of(const T &value) {
  const auto bytes = std::as_bytes(std::span(&value, 1));
  return {bytes.begin(), bytes.end()};
}

template <class T> std::vector<std::byte> bytes_of(std::span<const T> values) {
  const auto bytes = std::as_bytes(values);
  return {bytes.begin(), bytes.end()};
}

std::vector<float> decode_floats(const std::vector<std::byte> &bytes) {
  if (bytes.size() % sizeof(float) != 0) {
    throw std::runtime_error("DFlash2 float buffer is malformed");
  }
  std::vector<float> result(bytes.size() / sizeof(float));
  std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}

float decode_bf16(std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::vector<float> read_bf16(const MaterializedWeightView &view) {
  if (view.format != "direct_bf16_v1" || view.planes.size() != 1 ||
      view.planes.front().kind != "data" ||
      view.planes.front().buffer == nullptr) {
    throw std::invalid_argument("DFlash2 weight is not direct BF16: " +
                                view.name);
  }
  const auto &plane = view.planes.front();
  const auto bytes = plane.buffer->read(plane.offset, plane.bytes);
  if (bytes.size() % sizeof(std::uint16_t) != 0) {
    throw std::runtime_error("DFlash2 BF16 payload is malformed");
  }
  std::vector<float> result(bytes.size() / sizeof(std::uint16_t));
  for (std::size_t index = 0; index < result.size(); ++index) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, bytes.data() + index * sizeof(bits), sizeof(bits));
    result[index] = decode_bf16(bits);
  }
  return result;
}

const ArtifactObject &require_object(const ArtifactModel &artifact,
                                     std::string_view name,
                                     std::span<const std::uint64_t> shape) {
  const auto &object = artifact.object(name);
  if (object.format != "direct_bf16_v1" ||
      object.shape.size() != shape.size() ||
      !std::equal(object.shape.begin(), object.shape.end(), shape.begin())) {
    throw std::invalid_argument("DFlash2 artifact object mismatch: " +
                                std::string(name));
  }
  return object;
}

std::uint64_t workspace_values(const DFlash2ModelConfig &config,
                               std::uint32_t context_tokens,
                               std::uint32_t width) {
  const auto h = config.hidden_size;
  const auto q = std::uint64_t(config.query_heads) * config.head_dim;
  const auto kv = std::uint64_t(config.key_value_heads) * config.head_dim;
  const auto groups = h / config.conv_group_size;
  std::uint64_t total = 0;
  const auto add = [&](std::uint64_t values) {
    total = checked_add(total, values, "DFlash2 workspace");
  };
  add(std::uint64_t(context_tokens) * h);
  add(std::uint64_t(width) * h);
  add(std::uint64_t(width) * 4 * groups);
  add(std::uint64_t(width) * h);
  add(std::uint64_t(width) * q);
  add(std::uint64_t(width) * kv * 2);
  add(std::uint64_t(context_tokens) * kv * 2);
  add(std::uint64_t(width) * q);
  add(std::uint64_t(width) * h);
  add(std::uint64_t(width) * config.intermediate_size * 3);
  return total;
}

std::uint32_t require_dimension(const Json &value, std::string_view name) {
  const auto result = value.at(name).get<std::uint64_t>();
  if (result == 0 || result > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("DFlash2 config dimension is invalid: " +
                                std::string(name));
  }
  return static_cast<std::uint32_t>(result);
}

std::vector<std::int32_t> stable_top_k(std::span<const float> logits,
                                       std::uint32_t count) {
  if (count == 0 || count > logits.size()) {
    throw std::invalid_argument("DFlash2 selector top-k is invalid");
  }
  std::vector<std::int32_t> ids(logits.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(ids.begin(), ids.begin() + count, ids.end(),
                    [&](std::int32_t left, std::int32_t right) {
                      const auto lhs = logits[static_cast<std::size_t>(left)];
                      const auto rhs = logits[static_cast<std::size_t>(right)];
                      return lhs > rhs || (lhs == rhs && left < right);
                    });
  ids.resize(count);
  return ids;
}

} // namespace

DFlash2ModelConfig
load_dflash2_model_config(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::invalid_argument("unable to open DFlash2 config: " +
                                path.string());
  }
  Json value;
  input >> value;
  if (value.at("architectures") != Json::array({"DFlash2DraftModel"}) ||
      value.at("model_type") != "qwen3" || value.value("is_causal", true)) {
    throw std::invalid_argument("DFlash2 config architecture is unsupported");
  }
  const auto &extra = value.at("dflash_config");
  DFlash2ModelConfig config{
      .hidden_size = require_dimension(value, "hidden_size"),
      .intermediate_size = require_dimension(value, "intermediate_size"),
      .vocabulary_size = require_dimension(value, "vocab_size"),
      .layer_count = require_dimension(value, "num_hidden_layers"),
      .query_heads = require_dimension(value, "num_attention_heads"),
      .key_value_heads = require_dimension(value, "num_key_value_heads"),
      .head_dim = require_dimension(value, "head_dim"),
      .max_position_embeddings =
          require_dimension(value, "max_position_embeddings"),
      .sliding_window = require_dimension(value, "sliding_window"),
      .block_size = require_dimension(extra, "block_size"),
      .conv_group_size = require_dimension(extra, "conv_group_size"),
      .conv_kernel_size = require_dimension(extra, "conv_kernel_size"),
      .selector_rank = require_dimension(extra, "selector_rank"),
      .selector_top_k = require_dimension(extra, "selector_top_k"),
      .mask_token_id = extra.at("mask_token_id").get<std::uint32_t>(),
      .epsilon = value.at("rms_norm_eps").get<float>(),
      .rope_theta = value.at("rope_parameters").at("rope_theta").get<float>(),
  };
  config.target_layer_ids =
      extra.at("target_layer_ids").get<std::vector<std::uint32_t>>();
  const auto layer_types =
      value.at("layer_types").get<std::vector<std::string>>();
  if (config.layer_count == 0 || config.layer_count > 5 ||
      layer_types.size() != config.layer_count ||
      !std::ranges::all_of(
          layer_types,
          [](const auto &type) { return type == "sliding_attention"; }) ||
      config.conv_kernel_size != 2 ||
      config.hidden_size % config.conv_group_size ||
      config.query_heads % config.key_value_heads ||
      std::uint64_t(config.query_heads) * config.head_dim >
          config.hidden_size ||
      config.mask_token_id >= config.vocabulary_size ||
      config.selector_top_k > config.vocabulary_size ||
      config.target_layer_ids.empty() || !(config.epsilon > 0.0F) ||
      !(config.rope_theta > 0.0F) || config.block_size < 2 ||
      config.block_size > 16) {
    throw std::invalid_argument("DFlash2 config geometry is unsupported");
  }
  return config;
}

DFlash2ArtifactValidation
validate_dflash2_artifact(const ArtifactModel &artifact,
                          const DFlash2ModelConfig &config,
                          bool require_shared_weights) {
  const auto h = std::uint64_t(config.hidden_size);
  const auto i = std::uint64_t(config.intermediate_size);
  const auto q = std::uint64_t(config.query_heads) * config.head_dim;
  const auto kv = std::uint64_t(config.key_value_heads) * config.head_dim;
  const auto taps = std::uint64_t(config.target_layer_ids.size());
  const auto groups = h / config.conv_group_size;
  require_object(artifact, "dflash2.fc.weight", std::array{h, taps * h});
  require_object(artifact, "dflash2.hidden_norm.weight", std::array{h});
  require_object(artifact, "dflash2.norm.weight", std::array{h});
  require_object(artifact,
                 "dflash2.candidate_selector.hidden_projection.weight",
                 std::array{std::uint64_t(config.selector_rank), h});
  require_object(artifact, "dflash2.candidate_selector.predecessor_codebook",
                 std::array{std::uint64_t(config.vocabulary_size),
                            std::uint64_t(config.selector_rank)});
  require_object(artifact, "dflash2.candidate_selector.successor_codebook",
                 std::array{std::uint64_t(config.vocabulary_size),
                            std::uint64_t(config.selector_rank)});
  for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
    const auto prefix = "dflash2.layers." + std::to_string(layer) + ".";
    require_object(artifact, prefix + "input_layernorm.weight", std::array{h});
    require_object(artifact, prefix + "post_attention_layernorm.weight",
                   std::array{h});
    require_object(artifact, prefix + "self_attn.q_proj.weight",
                   std::array{q, h});
    require_object(artifact, prefix + "self_attn.k_proj.weight",
                   std::array{kv, h});
    require_object(artifact, prefix + "self_attn.v_proj.weight",
                   std::array{kv, h});
    require_object(artifact, prefix + "self_attn.q_norm.weight",
                   std::array{std::uint64_t(config.head_dim)});
    require_object(artifact, prefix + "self_attn.k_norm.weight",
                   std::array{std::uint64_t(config.head_dim)});
    require_object(artifact, prefix + "self_attn.o_proj.weight",
                   std::array{h, q});
    require_object(artifact, prefix + "mlp.gate_proj.weight", std::array{i, h});
    require_object(artifact, prefix + "mlp.up_proj.weight", std::array{i, h});
    require_object(artifact, prefix + "mlp.down_proj.weight", std::array{h, i});
    require_object(artifact, prefix + "attention_conv.base_kernel",
                   std::array{2ULL, 2ULL, h});
    require_object(artifact, prefix + "attention_conv.kernel_projection.weight",
                   std::array{4ULL * groups, h});
    require_object(artifact, prefix + "mlp_conv.base_kernel",
                   std::array{2ULL, 2ULL, h});
    require_object(artifact, prefix + "mlp_conv.kernel_projection.weight",
                   std::array{4ULL * groups, h});
  }
  const bool embedding =
      std::ranges::any_of(artifact.objects(), [](const auto &object) {
        return object.name == "dflash2.shared_embedding.weight";
      });
  const bool head =
      std::ranges::any_of(artifact.objects(), [](const auto &object) {
        return object.name == "dflash2.proposal_head.weight";
      });
  if (embedding) {
    require_object(artifact, "dflash2.shared_embedding.weight",
                   std::array{std::uint64_t(config.vocabulary_size), h});
  }
  if (head) {
    require_object(artifact, "dflash2.proposal_head.weight",
                   std::array{std::uint64_t(config.vocabulary_size), h});
  }
  if (require_shared_weights && (!embedding || !head)) {
    throw std::invalid_argument("executable DFlash2 fixture requires shared "
                                "embedding and proposal head");
  }
  std::uint64_t tensor_bytes = 0;
  for (const auto &object : artifact.objects()) {
    tensor_bytes = checked_add(tensor_bytes, object.object_bytes,
                               "DFlash2 artifact bytes");
  }
  return {.object_count = artifact.objects().size(),
          .tensor_bytes = tensor_bytes,
          .companion_complete = true,
          .shared_embedding_present = embedding,
          .proposal_head_present = head};
}

class DFlash2DraftProgram::Impl {
public:
  explicit Impl(MetalContext selected_context)
      : context(std::move(selected_context)) {}

  DFlash2ModelConfig config;
  DFlash2DraftBuildOptions options;
  MetalContext context;
  MaterializedModel weights;
  std::vector<std::uint64_t> offsets;
  MetalBuffer offsets_buffer;
  MetalBuffer noise_buffer;
  MetalBuffer target_buffer;
  MetalBuffer workspace_buffer;
  MetalBuffer hidden_buffer;
  MetalBuffer logits_buffer;
  MetalBuffer config_buffer;
  MetalCommandPlan plan;
  std::vector<float> embedding;
  std::vector<float> selector_hidden;
  std::vector<float> predecessor_codebook;
  std::vector<float> successor_codebook;
  DFlash2DraftStats stats;
};

DFlash2DraftProgram::DFlash2DraftProgram(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DFlash2DraftProgram::DFlash2DraftProgram(DFlash2DraftProgram &&) noexcept =
    default;
DFlash2DraftProgram &
DFlash2DraftProgram::operator=(DFlash2DraftProgram &&) noexcept = default;
DFlash2DraftProgram::~DFlash2DraftProgram() = default;

DFlash2DraftProgram
DFlash2DraftProgram::create(const std::filesystem::path &artifact_path,
                            const std::filesystem::path &config_path,
                            const std::filesystem::path &metallib_path,
                            DFlash2DraftBuildOptions options) {
  auto impl = std::make_unique<Impl>(MetalContext::create());
  impl->config = load_dflash2_model_config(config_path);
  if (options.max_context == 0 ||
      options.max_context > impl->config.max_position_embeddings ||
      options.max_draft_tokens == 0 ||
      options.max_draft_tokens + 1 > impl->config.block_size) {
    throw std::invalid_argument("DFlash2 draft build options are invalid");
  }
  impl->options = options;
  auto artifact = ArtifactModel::open(artifact_path, true);
  if (artifact.summary().source_profile != "dflash2_bf16") {
    throw std::invalid_argument("DFlash2 artifact profile is unsupported");
  }
  static_cast<void>(validate_dflash2_artifact(artifact, impl->config, true));
  auto materialization =
      plan_model_materialization(artifact, impl->context,
                                 {.components = {ModelComponent::speculative},
                                  .require_verified_payload = true});
  impl->weights =
      materialize_model(artifact, std::move(materialization), impl->context);
  if (impl->weights.segment_count() != 1) {
    throw std::invalid_argument(
        "current DFlash2 executable requires one companion weight segment");
  }

  const auto add_offset = [&](const std::string &name) {
    const auto view = impl->weights.parent_view(name);
    if (view.planes.size() != 1 || view.planes.front().segment_index != 0) {
      throw std::invalid_argument("DFlash2 weight placement is unsupported: " +
                                  name);
    }
    impl->offsets.push_back(view.planes.front().offset);
  };
  add_offset("dflash2.fc.weight");
  add_offset("dflash2.hidden_norm.weight");
  add_offset("dflash2.norm.weight");
  add_offset("dflash2.shared_embedding.weight");
  add_offset("dflash2.proposal_head.weight");
  add_offset("dflash2.candidate_selector.hidden_projection.weight");
  add_offset("dflash2.candidate_selector.predecessor_codebook");
  add_offset("dflash2.candidate_selector.successor_codebook");
  for (std::uint32_t layer = 0; layer < impl->config.layer_count; ++layer) {
    const auto prefix = "dflash2.layers." + std::to_string(layer) + ".";
    for (const auto &suffix :
         {"input_layernorm.weight", "post_attention_layernorm.weight",
          "self_attn.q_proj.weight", "self_attn.k_proj.weight",
          "self_attn.v_proj.weight", "self_attn.q_norm.weight",
          "self_attn.k_norm.weight", "self_attn.o_proj.weight",
          "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
          "attention_conv.base_kernel",
          "attention_conv.kernel_projection.weight", "mlp_conv.base_kernel",
          "mlp_conv.kernel_projection.weight"}) {
      add_offset(prefix + suffix);
    }
  }
  impl->embedding =
      read_bf16(impl->weights.parent_view("dflash2.shared_embedding.weight"));
  impl->selector_hidden = read_bf16(impl->weights.parent_view(
      "dflash2.candidate_selector.hidden_projection.weight"));
  impl->predecessor_codebook = read_bf16(impl->weights.parent_view(
      "dflash2.candidate_selector.predecessor_codebook"));
  impl->successor_codebook = read_bf16(impl->weights.parent_view(
      "dflash2.candidate_selector.successor_codebook"));

  const auto maximum_width = options.max_draft_tokens + 1;
  const auto h = impl->config.hidden_size;
  const auto taps = impl->config.target_layer_ids.size();
  const auto workspace_bytes = checked_multiply(
      workspace_values(impl->config, options.max_context, maximum_width),
      sizeof(float), "DFlash2 workspace bytes");
  impl->offsets_buffer = impl->context.make_buffer(checked_multiply(
      impl->offsets.size(), sizeof(std::uint64_t), "DFlash2 offset bytes"));
  impl->offsets_buffer.write(bytes_of<std::uint64_t>(impl->offsets));
  impl->noise_buffer = impl->context.make_buffer(checked_multiply(
      std::uint64_t(maximum_width) * h, sizeof(float), "DFlash2 noise bytes"));
  impl->target_buffer = impl->context.make_buffer(
      checked_multiply(std::uint64_t(options.max_context) * taps * h,
                       sizeof(float), "DFlash2 target feature bytes"));
  impl->workspace_buffer = impl->context.make_buffer(workspace_bytes);
  impl->hidden_buffer = impl->context.make_buffer(checked_multiply(
      std::uint64_t(maximum_width) * h, sizeof(float), "DFlash2 hidden bytes"));
  impl->logits_buffer = impl->context.make_buffer(checked_multiply(
      std::uint64_t(maximum_width) * impl->config.vocabulary_size,
      sizeof(float), "DFlash2 logits bytes"));
  impl->config_buffer = impl->context.make_buffer(sizeof(DFlash2MetalConfig));

  impl->context.register_library(
      {.key = "b5.dflash2.library", .path = metallib_path});
  static_cast<void>(impl->context.warmup_pipeline({
      .key = "b5.dflash2.program",
      .library_key = "b5.dflash2.library",
      .function_name = "omlx_m5_dflash2_program",
      .variant = "single-pass-masked-block",
  }));
  impl->plan = impl->context.prepare_command_plan({
      .key = "b5.dflash2.plan",
      .dispatches = {{
          .pipeline_key = "b5.dflash2.program",
          .barrier_before = false,
          .grid = {.width = 32},
          .threads_per_threadgroup = {.width = 32},
          .resources = {{.argument_index = 0,
                         .resource_slot = 0,
                         .minimum_bytes = impl->noise_buffer.size()},
                        {.argument_index = 1,
                         .resource_slot = 1,
                         .minimum_bytes = impl->target_buffer.size()},
                        {.argument_index = 2,
                         .resource_slot = 2,
                         .minimum_bytes = impl->weights.segment(0).size()},
                        {.argument_index = 3,
                         .resource_slot = 3,
                         .minimum_bytes = impl->offsets_buffer.size()},
                        {.argument_index = 4,
                         .resource_slot = 4,
                         .minimum_bytes = impl->workspace_buffer.size()},
                        {.argument_index = 5,
                         .resource_slot = 5,
                         .minimum_bytes = impl->hidden_buffer.size()},
                        {.argument_index = 6,
                         .resource_slot = 6,
                         .minimum_bytes = impl->logits_buffer.size()},
                        {.argument_index = 7,
                         .resource_slot = 7,
                         .minimum_bytes = impl->config_buffer.size()}},
      }},
  });
  return DFlash2DraftProgram(std::move(impl));
}

const DFlash2ModelConfig &DFlash2DraftProgram::config() const {
  return impl_->config;
}

const DFlash2DraftStats &DFlash2DraftProgram::stats() const {
  return impl_->stats;
}

DFlash2DraftResult DFlash2DraftProgram::propose(
    std::int32_t anchor_token, std::span<const float> target_features,
    std::uint32_t context_tokens, std::uint32_t draft_tokens, float temperature,
    std::uint64_t seed, std::uint64_t absolute_position) {
  const auto &config = impl_->config;
  const auto taps = config.target_layer_ids.size();
  if (anchor_token < 0 ||
      std::uint32_t(anchor_token) >= config.vocabulary_size ||
      context_tokens == 0 || context_tokens > impl_->options.max_context ||
      draft_tokens == 0 || draft_tokens > impl_->options.max_draft_tokens ||
      target_features.size() !=
          std::uint64_t(context_tokens) * taps * config.hidden_size ||
      !std::isfinite(temperature)) {
    throw std::invalid_argument("DFlash2 proposal request is invalid");
  }
  const auto width = draft_tokens + 1;
  std::vector<float> noise(std::uint64_t(width) * config.hidden_size);
  for (std::uint32_t token = 0; token < width; ++token) {
    const auto id =
        token == 0 ? std::uint32_t(anchor_token) : config.mask_token_id;
    std::copy_n(impl_->embedding.begin() +
                    std::uint64_t(id) * config.hidden_size,
                config.hidden_size,
                noise.begin() + std::uint64_t(token) * config.hidden_size);
  }
  impl_->noise_buffer.write(bytes_of<float>(noise));
  impl_->target_buffer.write(bytes_of<float>(target_features));
  const DFlash2MetalConfig metal{
      .hidden_size = config.hidden_size,
      .intermediate_size = config.intermediate_size,
      .vocabulary_size = config.vocabulary_size,
      .layer_count = config.layer_count,
      .query_heads = config.query_heads,
      .key_value_heads = config.key_value_heads,
      .head_dim = config.head_dim,
      .target_taps = static_cast<std::uint32_t>(taps),
      .context_tokens = context_tokens,
      .width = width,
      .conv_group_size = config.conv_group_size,
      .conv_kernel_size = config.conv_kernel_size,
      .selector_rank = config.selector_rank,
      .selector_top_k = config.selector_top_k,
      .sliding_window = config.sliding_window,
      .mask_token_id = config.mask_token_id,
      .epsilon = config.epsilon,
      .rope_theta = config.rope_theta,
  };
  impl_->config_buffer.write(bytes_of(metal));
  const std::array bindings = {
      MetalBufferBinding{0, &impl_->noise_buffer, 0},
      MetalBufferBinding{1, &impl_->target_buffer, 0},
      MetalBufferBinding{2, &impl_->weights.segment(0), 0},
      MetalBufferBinding{3, &impl_->offsets_buffer, 0},
      MetalBufferBinding{4, &impl_->workspace_buffer, 0},
      MetalBufferBinding{5, &impl_->hidden_buffer, 0},
      MetalBufferBinding{6, &impl_->logits_buffer, 0},
      MetalBufferBinding{7, &impl_->config_buffer, 0},
  };
  const std::array invocation = {MetalCommandInvocation{
      .plan = &impl_->plan, .bindings = {bindings.begin(), bindings.end()}}};
  auto submission = impl_->context.execute_batch(invocation);
  const auto hidden = decode_floats(impl_->hidden_buffer.read(
      0, std::uint64_t(width) * config.hidden_size * sizeof(float)));
  const auto logits = decode_floats(impl_->logits_buffer.read(
      0, std::uint64_t(width) * config.vocabulary_size * sizeof(float)));

  std::vector<CandidateRow> rows;
  rows.reserve(draft_tokens);
  for (std::uint32_t position = 0; position < draft_tokens; ++position) {
    const auto hidden_row = std::span(hidden).subspan(
        std::uint64_t(position + 1) * config.hidden_size, config.hidden_size);
    const auto logit_row = std::span(logits).subspan(
        std::uint64_t(position + 1) * config.vocabulary_size,
        config.vocabulary_size);
    auto ids = stable_top_k(logit_row, config.selector_top_k);
    CandidateRow row;
    row.token_ids = ids;
    row.unary_scores.reserve(ids.size());
    for (const auto id : ids) {
      row.unary_scores.push_back(logit_row[static_cast<std::size_t>(id)]);
    }
    row.projected_hidden.resize(config.selector_rank);
    for (std::uint32_t rank = 0; rank < config.selector_rank; ++rank) {
      float value = 0.0F;
      for (std::uint32_t feature = 0; feature < config.hidden_size; ++feature) {
        value = std::fma(
            impl_->selector_hidden[std::uint64_t(rank) * config.hidden_size +
                                   feature],
            hidden_row[feature], value);
      }
      row.projected_hidden[rank] = value;
    }
    rows.push_back(std::move(row));
  }
  auto selection = select_candidate_path(
      anchor_token, rows, impl_->predecessor_codebook,
      impl_->successor_codebook, config.vocabulary_size, config.selector_rank,
      temperature, seed, absolute_position);
  ++impl_->stats.rounds;
  ++impl_->stats.metal_dispatches;
  impl_->stats.context_tokens += context_tokens;
  impl_->stats.proposal_tokens += draft_tokens;
  return {.selection = std::move(selection),
          .hidden = hidden,
          .logits = logits,
          .candidate_rows = std::move(rows),
          .width = width,
          .draft_dispatches = 1,
          .position_parallel_threads = 32,
          .parallel_backbone = true,
          .dynamic_convolution = true,
          .conditional_selector = true,
          .submission = std::move(submission)};
}

} // namespace omlx::m5
