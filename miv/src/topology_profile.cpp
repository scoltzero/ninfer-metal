#include "omlx_m5/topology_profile.h"

#include "omlx_m5/materialization.h"

#include <algorithm>
#include <array>
#include <bit>
#include <iomanip>
#include <limits>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace omlx::m5 {
namespace {

constexpr std::uint32_t kMaximumProfileCount = 9;

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
  if (alignment == 0 || !std::has_single_bit(alignment)) {
    throw std::invalid_argument("profile alignment is invalid");
  }
  return checked_add(value, alignment - 1, "profile alignment") &
         ~(alignment - 1);
}

class StableHash final {
public:
  void add(std::string_view value) {
    for (const auto byte : value) {
      state_ ^= static_cast<std::uint8_t>(byte);
      state_ *= 1099511628211ULL;
    }
    state_ ^= 0xFFU;
    state_ *= 1099511628211ULL;
  }

  template <class T> void add_integer(T value) {
    static_assert(std::is_integral_v<T> || std::is_enum_v<T>);
    using Value = decltype([] {
      if constexpr (std::is_enum_v<T>) {
        return std::underlying_type_t<T>{};
      } else {
        return T{};
      }
    }());
    using Unsigned = std::make_unsigned_t<Value>;
    auto selected = static_cast<std::uint64_t>(static_cast<Unsigned>(value));
    for (std::size_t index = 0; index < sizeof(Unsigned); ++index) {
      state_ ^= static_cast<std::uint8_t>(selected & 0xFFU);
      state_ *= 1099511628211ULL;
      selected >>= 8U;
    }
  }

  [[nodiscard]] std::string finish() const {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << state_;
    return output.str();
  }

private:
  std::uint64_t state_ = 14695981039346656037ULL;
};

bool direct_format(std::string_view format) {
  return format == "direct_bf16_v1" || format == "direct_f16_v1";
}

bool oq_format(std::string_view format) {
  return format.starts_with("oq_affine_q");
}

QwenQuantProfile classify_quant_profile(const ArtifactModel &artifact) {
  std::uint64_t mxfp4_count = 0;
  std::uint64_t oq_count = 0;
  std::uint64_t direct_count = 0;
  for (const auto &object : artifact.objects()) {
    if (classify_model_component(object.name) != ModelComponent::text) {
      continue;
    }
    if (object.format == "mxfp4_e2m1_g32_e8m0_v1") {
      ++mxfp4_count;
    } else if (oq_format(object.format)) {
      ++oq_count;
    } else if (direct_format(object.format)) {
      ++direct_count;
    }
  }
  if (mxfp4_count > 0 && mxfp4_count >= oq_count) {
    return QwenQuantProfile::mxfp4;
  }
  if (oq_count > 0 || mxfp4_count > 0 || direct_count > 0) {
    return QwenQuantProfile::oq_affine;
  }
  throw std::invalid_argument(
      "topology contains no text weights");
}

void validate_profile_format(QwenQuantProfile /*profile*/,
                             std::string_view format) {
  if (direct_format(format) || oq_format(format) ||
      format == "mxfp4_e2m1_g32_e8m0_v1") {
    return;
  }
  throw std::invalid_argument("artifact format does not belong to profile");
}

std::string topology_signature(const QwenTopologyDescriptor &topology) {
  StableHash hash;
  hash.add("omlx-m5-qwen-topology-v1");
  hash.add_integer(topology.architecture);
  hash.add_integer(topology.quant_profile);
  hash.add_integer(topology.layer_count);
  hash.add_integer(topology.top_k);
  hash.add_integer(topology.max_context);
  hash.add_integer(topology.kv_page_tokens);
  for (const auto &dispatch : topology.dispatches) {
    hash.add_integer(dispatch.index);
    hash.add_integer(dispatch.layer);
    hash.add_integer(dispatch.kind);
    hash.add(dispatch.role);
    hash.add_integer(dispatch.stage_dispatch);
    hash.add_integer(dispatch.object_id == UINT32_MAX ? 0U : 1U);
    hash.add(dispatch.format);
    hash.add(dispatch.layout);
    for (const auto dimension : dispatch.shape) {
      hash.add_integer(dimension);
    }
  }
  for (const auto &range : topology.ranges) {
    hash.add(range.key);
    hash.add_integer(range.location);
    hash.add_integer(range.length);
  }
  return hash.finish();
}

void add_range(std::vector<MetalCommandRangeDescriptor> &ranges,
               std::string key, std::uint32_t location, std::uint32_t length) {
  if (key.empty() || length == 0) {
    throw std::invalid_argument("topology range is empty");
  }
  ranges.push_back(
      {.key = std::move(key), .location = location, .length = length});
}

void append_stage(QwenTopologyDescriptor &result, const ArtifactModel &artifact,
                  std::uint32_t layer, const LayerStageRecord &stage) {
  for (std::uint32_t ordinal = 0; ordinal < stage.dispatches; ++ordinal) {
    TopologyDispatchRecord record{
        .index = static_cast<std::uint32_t>(result.dispatches.size()),
        .layer = layer,
        .kind = stage.kind,
        .role = stage.role,
        .stage_dispatch = ordinal,
        .object_id = stage.object_id,
        .format = stage.format,
    };
    if (stage.object_id != UINT32_MAX) {
      const auto &object = artifact.object(stage.object_id);
      if (object.name != stage.object_name || object.format != stage.format) {
        throw std::invalid_argument(
            "layer catalog and artifact object disagree");
      }
      validate_profile_format(result.quant_profile, object.format);
      record.layout = object.layout;
      record.shape = object.shape;
    }
    result.dispatches.push_back(std::move(record));
  }
}

MetalParameterTableDescriptor
parameter_descriptor(const TopologyProfileKey &key, std::uint32_t max_context,
                     std::uint32_t page_tokens, std::uint32_t top_k) {
  MetalParameterTableDescriptor result;
  result.key = topology_profile_key(key) + ".parameters";
  std::uint64_t cursor = 0;
  auto add = [&](std::string field, std::uint64_t bytes,
                 std::uint64_t alignment) {
    cursor = align_up(cursor, alignment);
    result.fields.push_back({.key = std::move(field),
                             .offset = cursor,
                             .size = bytes,
                             .alignment = alignment});
    cursor = checked_add(cursor, bytes, "profile parameter table");
  };
  const auto token_bytes =
      checked_multiply(key.token_capacity, 4, "profile token parameters");
  const auto page_count =
      (static_cast<std::uint64_t>(max_context) + page_tokens - 1) / page_tokens;
  add("token_ids", token_bytes, 16);
  add("positions", token_bytes, 16);
  add("kv_frontier", 4, 4);
  add("page_count", 4, 4);
  add("page_table", checked_multiply(page_count, 4, "profile page table"), 16);
  if (key.architecture == QwenArchitecture::moe) {
    const auto route_values =
        checked_multiply(key.token_capacity, top_k, "profile route values");
    add("route_indices", checked_multiply(route_values, 4, "route indices"),
        16);
    add("route_scores", checked_multiply(route_values, 4, "route scores"), 16);
  }
  add("valid_mask", token_bytes, 16);
  add("sampling", 16, 16);
  add("kernel_params", 64, 16);
  result.byte_size = align_up(cursor, 256);
  return result;
}

std::string profile_signature(const TopologyProfile &profile) {
  StableHash hash;
  hash.add("omlx-m5-exact-topology-profile-v1");
  hash.add(profile.name);
  for (const auto &field : profile.parameters.fields) {
    hash.add(field.key);
    hash.add_integer(field.offset);
    hash.add_integer(field.size);
    hash.add_integer(field.alignment);
  }
  hash.add_integer(profile.parameters.byte_size);
  for (const auto &dispatch : profile.dispatches) {
    hash.add_integer(dispatch.index);
    hash.add_integer(dispatch.layer);
    hash.add_integer(dispatch.kind);
    hash.add(dispatch.role);
    hash.add_integer(dispatch.stage_dispatch);
    hash.add(dispatch.format);
    hash.add(dispatch.layout);
    for (const auto dimension : dispatch.shape) {
      hash.add_integer(dimension);
    }
  }
  for (const auto &range : profile.ranges) {
    hash.add(range.key);
    hash.add_integer(range.location);
    hash.add_integer(range.length);
  }
  return hash.finish();
}

void validate_capacity(std::uint32_t capacity) {
  if (capacity < 2 || (capacity > 128 && capacity != 512 && capacity != 1024)) {
    throw std::invalid_argument(
        "prefill capacity is outside the frozen B2 support domain");
  }
}

} // namespace

std::string_view to_string(QwenArchitecture value) {
  switch (value) {
  case QwenArchitecture::dense:
    return "dense";
  case QwenArchitecture::moe:
    return "moe";
  }
  throw std::invalid_argument("unknown Qwen architecture");
}

std::string_view to_string(QwenQuantProfile value) {
  switch (value) {
  case QwenQuantProfile::oq_affine:
    return "oq";
  case QwenQuantProfile::mxfp4:
    return "mxfp4";
  }
  throw std::invalid_argument("unknown Qwen quantization profile");
}

std::string_view to_string(TopologyProfileMode value) {
  switch (value) {
  case TopologyProfileMode::decode:
    return "decode";
  case TopologyProfileMode::prefill:
    return "prefill";
  }
  throw std::invalid_argument("unknown topology profile mode");
}

std::string_view to_string(TopologyFallbackReason value) {
  switch (value) {
  case TopologyFallbackReason::none:
    return "none";
  case TopologyFallbackReason::architecture_mismatch:
    return "architecture_mismatch";
  case TopologyFallbackReason::quant_profile_mismatch:
    return "quant_profile_mismatch";
  case TopologyFallbackReason::topology_mismatch:
    return "topology_mismatch";
  case TopologyFallbackReason::unsupported_batch:
    return "unsupported_batch";
  case TopologyFallbackReason::decode_requires_one_token:
    return "decode_requires_one_token";
  case TopologyFallbackReason::prefill_requires_multiple_tokens:
    return "prefill_requires_multiple_tokens";
  case TopologyFallbackReason::unsupported_token_capacity:
    return "unsupported_token_capacity";
  }
  throw std::invalid_argument("unknown topology fallback reason");
}

QwenTopologyDescriptor describe_qwen_topology(const ArtifactModel &artifact,
                                              const QwenModelConfig &config,
                                              const QwenLayerCatalog &layers,
                                              std::uint32_t max_context,
                                              std::uint32_t kv_page_tokens) {
  if (max_context == 0 || kv_page_tokens == 0 ||
      config.max_position_embeddings <= 0 ||
      max_context >
          static_cast<std::uint32_t>(config.max_position_embeddings) ||
      layers.layers().empty() ||
      layers.layers().size() !=
          static_cast<std::size_t>(config.num_hidden_layers) ||
      layers.stats().token_loop_artifact_lookups != 0) {
    throw std::invalid_argument("Qwen topology source is invalid");
  }
  QwenTopologyDescriptor result{
      .architecture = config.num_experts > 0 ? QwenArchitecture::moe
                                             : QwenArchitecture::dense,
      .quant_profile = classify_quant_profile(artifact),
      .layer_count = static_cast<std::uint32_t>(layers.layers().size()),
      .top_k = static_cast<std::uint32_t>(config.num_experts_per_token),
      .max_context = max_context,
      .kv_page_tokens = kv_page_tokens,
  };
  if ((result.architecture == QwenArchitecture::moe && result.top_k == 0) ||
      (result.architecture == QwenArchitecture::dense && result.top_k != 0)) {
    throw std::invalid_argument("Qwen topology expert geometry is invalid");
  }
  const auto embedding = std::ranges::find_if(
      layers.global_stages(),
      [](const LayerStageRecord &stage) { return stage.role == "embedding"; });
  if (embedding == layers.global_stages().end()) {
    throw std::invalid_argument("Qwen topology has no embedding stage");
  }
  const auto embedding_begin =
      static_cast<std::uint32_t>(result.dispatches.size());
  append_stage(result, artifact, UINT32_MAX, *embedding);
  add_range(result.ranges, "embedding", embedding_begin,
            static_cast<std::uint32_t>(result.dispatches.size()) -
                embedding_begin);

  for (const auto &layer : layers.layers()) {
    if (layer.layer >= result.layer_count ||
        layer.moe != (result.architecture == QwenArchitecture::moe)) {
      throw std::invalid_argument("Qwen layer topology is inconsistent");
    }
    const auto begin = static_cast<std::uint32_t>(result.dispatches.size());
    for (const auto &stage : layer.stages) {
      append_stage(result, artifact, layer.layer, stage);
    }
    add_range(result.ranges, "layer." + std::to_string(layer.layer), begin,
              static_cast<std::uint32_t>(result.dispatches.size()) - begin);
  }

  const auto vocabulary_begin =
      static_cast<std::uint32_t>(result.dispatches.size());
  for (const auto &stage : layers.global_stages()) {
    if (stage.role != "embedding") {
      append_stage(result, artifact, UINT32_MAX, stage);
    }
  }
  add_range(result.ranges, "vocabulary", vocabulary_begin,
            static_cast<std::uint32_t>(result.dispatches.size()) -
                vocabulary_begin);
  result.ranges.insert(
      result.ranges.begin(),
      {.key = "full",
       .location = 0,
       .length = static_cast<std::uint32_t>(result.dispatches.size())});
  if (result.dispatches.empty() ||
      result.dispatches.size() != layers.stats().dispatches_per_decode_step) {
    throw std::invalid_argument("Qwen topology dispatch accounting mismatch");
  }
  result.signature = topology_signature(result);
  return result;
}

std::string topology_profile_key(const TopologyProfileKey &key) {
  if (key.batch_size == 0 || key.token_capacity == 0 ||
      key.topology_signature.empty()) {
    throw std::invalid_argument("topology profile key is incomplete");
  }
  return "b4." + std::string(to_string(key.mode)) + "." +
         std::string(to_string(key.architecture)) + "." +
         std::string(to_string(key.quant_profile)) + ".b" +
         std::to_string(key.batch_size) + ".t" +
         std::to_string(key.token_capacity) + "." + key.topology_signature;
}

const QwenTopologyDescriptor &TopologyProfileCatalog::topology() const {
  return topology_;
}

const std::vector<TopologyProfile> &TopologyProfileCatalog::profiles() const {
  return profiles_;
}

const TopologyProfile &
TopologyProfileCatalog::profile(std::uint32_t index) const {
  if (index >= profiles_.size()) {
    throw std::out_of_range("topology profile index is out of range");
  }
  return profiles_[index];
}

const TopologyProfileCatalogStats &TopologyProfileCatalog::stats() const {
  return stats_;
}

TopologyProfileSelection
TopologyProfileCatalog::select(const TopologyProfileRequest &request) const {
  const auto fallback = [](TopologyFallbackReason reason) {
    return TopologyProfileSelection{.reason = reason,
                                    .execution_route = "execute_batch"};
  };
  if (request.architecture != topology_.architecture) {
    return fallback(TopologyFallbackReason::architecture_mismatch);
  }
  if (request.quant_profile != topology_.quant_profile) {
    return fallback(TopologyFallbackReason::quant_profile_mismatch);
  }
  if (request.topology_signature != topology_.signature) {
    return fallback(TopologyFallbackReason::topology_mismatch);
  }
  if (request.batch_size != 1) {
    return fallback(TopologyFallbackReason::unsupported_batch);
  }
  if (request.mode == TopologyProfileMode::decode && request.tokens != 1) {
    return fallback(TopologyFallbackReason::decode_requires_one_token);
  }
  if (request.mode == TopologyProfileMode::prefill && request.tokens < 2) {
    return fallback(TopologyFallbackReason::prefill_requires_multiple_tokens);
  }
  for (std::uint32_t index = 0; index < profiles_.size(); ++index) {
    const auto &key = profiles_[index].key;
    if (key.mode == request.mode && key.batch_size == request.batch_size &&
        key.token_capacity >= request.tokens) {
      return {.profile_index = index};
    }
  }
  return fallback(TopologyFallbackReason::unsupported_token_capacity);
}

TopologyProfileCatalog
compile_topology_profiles(QwenTopologyDescriptor topology,
                          TopologyProfileCompilerOptions options) {
  if (topology.layer_count == 0 || topology.max_context == 0 ||
      topology.kv_page_tokens == 0 || topology.dispatches.empty() ||
      topology.ranges.empty() || options.batch_size != 1) {
    throw std::invalid_argument("topology compiler input is invalid");
  }
  if ((topology.architecture == QwenArchitecture::dense &&
       topology.top_k != 0) ||
      (topology.architecture == QwenArchitecture::moe && topology.top_k == 0)) {
    throw std::invalid_argument("topology compiler expert geometry is invalid");
  }
  for (std::size_t index = 0; index < topology.dispatches.size(); ++index) {
    const auto &dispatch = topology.dispatches[index];
    if (dispatch.index != index) {
      throw std::invalid_argument("topology dispatch indices are not dense");
    }
    if (!dispatch.format.empty()) {
      validate_profile_format(topology.quant_profile, dispatch.format);
    }
  }
  std::unordered_set<std::string> range_keys;
  for (const auto &range : topology.ranges) {
    const auto end =
        checked_add(range.location, range.length, "topology range");
    if (range.key.empty() || range.length == 0 ||
        end > topology.dispatches.size() ||
        !range_keys.insert(range.key).second) {
      throw std::invalid_argument("topology range directory is invalid");
    }
  }
  if (topology.ranges.front().key != "full" ||
      topology.ranges.front().location != 0 ||
      topology.ranges.front().length != topology.dispatches.size()) {
    throw std::invalid_argument("topology full range is invalid");
  }
  const auto expected_signature = topology_signature(topology);
  if (topology.signature.empty()) {
    topology.signature = expected_signature;
  } else if (topology.signature != expected_signature) {
    throw std::invalid_argument("topology signature does not match descriptor");
  }
  if (options.prefill_capacities.empty() ||
      options.prefill_capacities.size() + 1 > kMaximumProfileCount) {
    throw std::invalid_argument("topology profile catalog would be unbounded");
  }
  std::ranges::sort(options.prefill_capacities);
  if (std::ranges::adjacent_find(options.prefill_capacities) !=
      options.prefill_capacities.end()) {
    throw std::invalid_argument("prefill capacities are duplicated");
  }
  for (const auto capacity : options.prefill_capacities) {
    validate_capacity(capacity);
    if (capacity > topology.max_context) {
      throw std::invalid_argument(
          "prefill capacity exceeds the Program context capacity");
    }
  }

  TopologyProfileCatalog result;
  result.topology_ = std::move(topology);
  auto append = [&](TopologyProfileMode mode, std::uint32_t capacity) {
    TopologyProfile profile{
        .key = {.mode = mode,
                .architecture = result.topology_.architecture,
                .quant_profile = result.topology_.quant_profile,
                .batch_size = 1,
                .token_capacity = capacity,
                .topology_signature = result.topology_.signature},
        .dispatches = result.topology_.dispatches,
        .ranges = result.topology_.ranges,
    };
    profile.name = topology_profile_key(profile.key);
    profile.parameters = parameter_descriptor(
        profile.key, result.topology_.max_context,
        result.topology_.kv_page_tokens, result.topology_.top_k);
    profile.structural_signature = profile_signature(profile);
    result.stats_.total_parameter_bytes =
        checked_add(result.stats_.total_parameter_bytes,
                    profile.parameters.byte_size, "profile parameter bytes");
    result.profiles_.push_back(std::move(profile));
  };
  append(TopologyProfileMode::decode, 1);
  for (const auto capacity : options.prefill_capacities) {
    append(TopologyProfileMode::prefill, capacity);
  }
  result.stats_.profile_count = result.profiles_.size();
  result.stats_.decode_profile_count = 1;
  result.stats_.prefill_profile_count = options.prefill_capacities.size();
  result.stats_.maximum_token_capacity = options.prefill_capacities.back();
  result.stats_.dispatch_records_per_profile =
      result.topology_.dispatches.size();
  result.stats_.finite = result.stats_.profile_count <= kMaximumProfileCount;
  return result;
}

} // namespace omlx::m5
