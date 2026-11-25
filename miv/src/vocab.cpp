#include "omlx_m5/vocab.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

namespace omlx::m5 {
namespace {

struct EmbeddingParams {
  std::uint32_t vocabulary = 0;
  std::uint32_t row_start = 0;
  std::uint32_t row_count = 0;
  std::uint32_t hidden_size = 0;
  std::uint32_t padded_hidden_size = 0;
  std::uint32_t tokens = 0;
  std::uint32_t reserved0 = 0;
  std::uint32_t reserved1 = 0;
};

struct SamplingParams {
  std::uint32_t vocabulary = 0;
  std::uint32_t top_k = 0;
  float temperature = 0.0F;
  float top_p = 1.0F;
  std::uint64_t seed = 0;
  std::uint64_t stream = 0;
  std::uint64_t absolute_position = 0;
};

static_assert(sizeof(EmbeddingParams) == 32);
static_assert(sizeof(SamplingParams) == 40);

std::uint32_t checked_u32(std::uint64_t value, const char *label) {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error(std::string(label) + " exceeds uint32");
  }
  return static_cast<std::uint32_t>(value);
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               const char *label) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error(std::string(label) + " overflow");
  }
  return left * right;
}

template <class Params>
std::vector<std::byte> constant_bytes(const Params &value) {
  std::vector<std::byte> result(sizeof(value));
  std::memcpy(result.data(), &value, sizeof(value));
  return result;
}

const MaterializedPlaneView &plane(const MaterializedWeightView &weight,
                                   std::string_view kind) {
  const auto found = std::ranges::find_if(
      weight.planes,
      [&](const MaterializedPlaneView &value) { return value.kind == kind; });
  if (found == weight.planes.end() || found->buffer == nullptr ||
      found->bytes == 0) {
    throw std::invalid_argument("embedding plane is missing: " +
                                std::string(kind));
  }
  return *found;
}

struct EmbeddingFormatSpec {
  std::string function;
  std::string expected_layout;
  std::uint32_t bits = 0;
  std::uint32_t group_size = 1;
};

EmbeddingFormatSpec parse_embedding_format(std::string_view format) {
  if (format == "direct_bf16_v1") {
    return {.function = "omlx_m5_embedding_bf16",
            .expected_layout = "contiguous_le_v1",
            .bits = 16,
            .group_size = 1};
  }
  if (format == "mxfp4_e2m1_g32_e8m0_v1") {
    return {.function = "omlx_m5_embedding_mxfp4",
            .expected_layout = "mx_rowsplit_k128_v1",
            .bits = 4,
            .group_size = 32};
  }
  constexpr std::string_view prefix = "oq_affine_q";
  constexpr std::string_view middle = "_g";
  constexpr std::string_view suffix = "_bf16_v1";
  if (!format.starts_with(prefix) || !format.ends_with(suffix)) {
    throw std::invalid_argument("unsupported embedding format");
  }
  const auto group_position = format.find(middle, prefix.size());
  if (group_position == std::string_view::npos) {
    throw std::invalid_argument("malformed embedding affine format");
  }
  const auto bits =
      format.substr(prefix.size(), group_position - prefix.size());
  const auto group = format.substr(group_position + middle.size(),
                                   format.size() - suffix.size() -
                                       group_position - middle.size());
  if ((bits != "4" && bits != "5" && bits != "6" && bits != "8") ||
      (group != "64" && group != "128")) {
    throw std::invalid_argument("unsupported embedding affine geometry");
  }
  const auto bits_value = static_cast<std::uint32_t>(std::stoul(std::string(bits)));
  const auto group_value =
      static_cast<std::uint32_t>(std::stoul(std::string(group)));
  return {.function =
              "omlx_m5_embedding_q" + std::string(bits) + "_g" + std::string(group),
          .expected_layout = "oq_rowsplit_k128_v1",
          .bits = bits_value,
          .group_size = group_value};
}

void require_plane_bytes(const MaterializedPlaneView &selected,
                         std::uint64_t minimum_bytes) {
  if (selected.bytes < minimum_bytes) {
    throw std::invalid_argument("embedding plane is smaller than required: " +
                                selected.kind);
  }
}

} // namespace

class EmbeddingPlan::Impl {
public:
  MaterializedWeightView weight;
  EmbeddingInfo info;
  MetalCommandPlan command_plan;
  bool uses_high = false;
};

EmbeddingPlan::EmbeddingPlan() = default;
EmbeddingPlan::EmbeddingPlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
EmbeddingPlan::EmbeddingPlan(EmbeddingPlan &&) noexcept = default;
EmbeddingPlan &EmbeddingPlan::operator=(EmbeddingPlan &&) noexcept = default;
EmbeddingPlan::~EmbeddingPlan() = default;

bool EmbeddingPlan::valid() const { return impl_ != nullptr; }

const EmbeddingInfo &EmbeddingPlan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("embedding plan is empty");
  }
  return impl_->info;
}

EmbeddingPlan prepare_embedding(const MetalContext &context,
                                const std::filesystem::path &metallib,
                                MaterializedWeightView weight,
                                std::uint32_t tokens) {
  if (weight.format == "direct_bf16_v1" && weight.padded_k == 0) {
    weight.padded_k = weight.in_features;
  }
  if (tokens == 0 || weight.parent_shape.size() != 2 || weight.row_count == 0 ||
      weight.row_start > weight.parent_shape[0] ||
      weight.row_count > weight.parent_shape[0] - weight.row_start ||
      weight.in_features == 0 || weight.padded_k < weight.in_features ||
      weight.planes.empty()) {
    throw std::invalid_argument("embedding weight view is invalid");
  }
  const auto spec = parse_embedding_format(weight.format);
  if (weight.layout != spec.expected_layout) {
    throw std::invalid_argument("embedding weight layout is unsupported: " +
                                weight.layout);
  }
  if (spec.group_size > 1 && (weight.padded_k % spec.group_size) != 0) {
    throw std::invalid_argument(
        "embedding padded_k must be divisible by group size");
  }
  auto impl = std::make_unique<EmbeddingPlan::Impl>();
  impl->weight = std::move(weight);
  const auto output_values =
      checked_multiply(tokens, impl->weight.in_features, "embedding output");
  impl->info = {
      .executor_key = "embedding." + impl->weight.format + ".r" +
                      std::to_string(impl->weight.row_start) + ".n" +
                      std::to_string(impl->weight.row_count) + ".t" +
                      std::to_string(tokens),
      .format_id = impl->weight.format,
      .tokens = tokens,
      .vocabulary = impl->weight.parent_shape[0],
      .row_start = impl->weight.row_start,
      .row_count = impl->weight.row_count,
      .hidden_size = impl->weight.in_features,
      .token_bytes = checked_multiply(tokens, 4, "embedding token bytes"),
      .output_bytes = checked_multiply(output_values, 2, "embedding bytes"),
  };
  for (const auto &selected : impl->weight.planes) {
    impl->info.weight_plane_bytes += selected.bytes;
  }
  const EmbeddingParams params{
      .vocabulary = checked_u32(impl->weight.parent_shape[0], "vocabulary"),
      .row_start = checked_u32(impl->weight.row_start, "embedding row start"),
      .row_count = checked_u32(impl->weight.row_count, "embedding rows"),
      .hidden_size = checked_u32(impl->weight.in_features, "embedding hidden"),
      .padded_hidden_size =
          checked_u32(impl->weight.padded_k, "embedding padded hidden"),
      .tokens = tokens,
  };
  context.register_library({.key = "b3.vocab.library", .path = metallib});
  const auto pipeline_key = "b3.vocab." + impl->weight.format;
  static_cast<void>(context.warmup_pipeline({
      .key = pipeline_key,
      .library_key = "b3.vocab.library",
      .function_name = spec.function,
      .variant = "lookup",
  }));
  MetalDispatchDescriptor dispatch{
      .pipeline_key = pipeline_key,
      .grid = {.width = output_values},
      .threads_per_threadgroup = {.width = 256},
  };
  const auto rows = impl->weight.row_count;
  const auto padded_k = impl->weight.padded_k;
  if (impl->weight.format == "direct_bf16_v1") {
    const auto &data = plane(impl->weight, "data");
    const auto min_data = checked_multiply(
        checked_multiply(rows, padded_k, "embedding bf16 elements"), 2,
        "embedding bf16 bytes");
    require_plane_bytes(data, min_data);
    dispatch.resources = {
        {.argument_index = 0,
         .resource_slot = 0,
         .minimum_bytes = impl->info.token_bytes},
        {.argument_index = 1, .resource_slot = 1, .minimum_bytes = min_data},
        {.argument_index = 2,
         .resource_slot = 5,
         .minimum_bytes = impl->info.output_bytes},
    };
    dispatch.constant_arguments = {
        {.argument_index = 3, .bytes = constant_bytes(params)}};
  } else if (impl->weight.format == "mxfp4_e2m1_g32_e8m0_v1") {
    const auto &data = plane(impl->weight, "data");
    const auto &scales = plane(impl->weight, "scale");
    const auto min_data =
        checked_multiply(rows, padded_k / 2, "embedding mxfp4 data bytes");
    const auto min_scales =
        checked_multiply(rows, padded_k / 32, "embedding mxfp4 scale bytes");
    require_plane_bytes(data, min_data);
    require_plane_bytes(scales, min_scales);
    dispatch.resources = {
        {.argument_index = 0,
         .resource_slot = 0,
         .minimum_bytes = impl->info.token_bytes},
        {.argument_index = 1, .resource_slot = 1, .minimum_bytes = min_data},
        {.argument_index = 2,
         .resource_slot = 3,
         .minimum_bytes = min_scales},
        {.argument_index = 3,
         .resource_slot = 5,
         .minimum_bytes = impl->info.output_bytes},
    };
    dispatch.constant_arguments = {
        {.argument_index = 4, .bytes = constant_bytes(params)}};
  } else {
    const auto code_kind = spec.bits == 8 ? "code" : "base_code";
    const auto &codes = plane(impl->weight, code_kind);
    const MaterializedPlaneView *high = &codes;
    std::uint64_t min_high = 0;
    if (spec.bits == 5) {
      high = &plane(impl->weight, "high_bit");
      min_high =
          checked_multiply(rows, padded_k / 8, "embedding affine high bytes");
    } else if (spec.bits == 6) {
      high = &plane(impl->weight, "high_2bit");
      min_high =
          checked_multiply(rows, padded_k / 4, "embedding affine high bytes");
    }
    const auto &scales = plane(impl->weight, "scale");
    const auto &biases = plane(impl->weight, "bias");
    const auto min_codes = checked_multiply(
        rows, spec.bits == 8 ? padded_k : padded_k / 2,
        "embedding affine code bytes");
    const auto min_sb = checked_multiply(
        checked_multiply(rows, padded_k / spec.group_size,
                         "embedding affine groups"),
        2, "embedding affine scale/bias bytes");
    require_plane_bytes(codes, min_codes);
    require_plane_bytes(scales, min_sb);
    require_plane_bytes(biases, min_sb);
    impl->uses_high = high != &codes;
    if (impl->uses_high) {
      require_plane_bytes(*high, min_high);
    }
    dispatch.resources = {
        {.argument_index = 0,
         .resource_slot = 0,
         .minimum_bytes = impl->info.token_bytes},
        {.argument_index = 1, .resource_slot = 1, .minimum_bytes = min_codes},
        {.argument_index = 3, .resource_slot = 3, .minimum_bytes = min_sb},
        {.argument_index = 4, .resource_slot = 4, .minimum_bytes = min_sb},
        {.argument_index = 5,
         .resource_slot = 5,
         .minimum_bytes = impl->info.output_bytes},
    };
    if (impl->uses_high) {
      dispatch.resources.insert(dispatch.resources.begin() + 2,
                                {.argument_index = 2,
                                 .resource_slot = 2,
                                 .minimum_bytes = min_high});
    }
    dispatch.constant_arguments = {
        {.argument_index = 6, .bytes = constant_bytes(params)}};
  }
  impl->command_plan = context.prepare_command_plan({
      .key = impl->info.executor_key,
      .dispatches = {std::move(dispatch)},
  });
  return EmbeddingPlan(std::move(impl));
}

MetalSubmission
execute_embedding(const MetalContext &context, const EmbeddingPlan &plan,
                  const MetalBuffer &token_ids, std::uint64_t token_offset,
                  MetalBuffer &output, std::uint64_t output_offset) {
  if (plan.impl_ == nullptr || token_offset % 4 != 0 ||
      output_offset % 2 != 0) {
    throw std::invalid_argument("embedding invocation is invalid");
  }
  std::vector<MetalBufferBinding> bindings = {
      {0, &token_ids, token_offset},
  };
  if (plan.impl_->weight.format == "direct_bf16_v1") {
    const auto &data = plane(plan.impl_->weight, "data");
    bindings.push_back({1, data.buffer, data.offset});
  } else if (plan.impl_->weight.format == "mxfp4_e2m1_g32_e8m0_v1") {
    const auto &data = plane(plan.impl_->weight, "data");
    const auto &scales = plane(plan.impl_->weight, "scale");
    bindings.push_back({1, data.buffer, data.offset});
    bindings.push_back({3, scales.buffer, scales.offset});
  } else {
    const auto code_kind =
        plan.impl_->weight.format.find("_q8_") != std::string::npos
            ? "code"
            : "base_code";
    const auto &codes = plane(plan.impl_->weight, code_kind);
    bindings.push_back({1, codes.buffer, codes.offset});
    if (plan.impl_->uses_high) {
      const auto high_kind =
          plan.impl_->weight.format.find("_q5_") != std::string::npos
              ? "high_bit"
              : "high_2bit";
      const auto &high = plane(plan.impl_->weight, high_kind);
      bindings.push_back({2, high.buffer, high.offset});
    }
    const auto &scales = plane(plan.impl_->weight, "scale");
    const auto &biases = plane(plan.impl_->weight, "bias");
    bindings.push_back({3, scales.buffer, scales.offset});
    bindings.push_back({4, biases.buffer, biases.offset});
  }
  bindings.push_back({5, &output, output_offset});
  return context.execute(plan.impl_->command_plan, bindings);
}

class SamplingPlan::Impl {
public:
  SamplingInfo info;
  MetalCommandPlan command_plan;
  MetalBuffer params_buffer;
};

SamplingPlan::SamplingPlan() = default;
SamplingPlan::SamplingPlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
SamplingPlan::SamplingPlan(SamplingPlan &&) noexcept = default;
SamplingPlan &SamplingPlan::operator=(SamplingPlan &&) noexcept = default;
SamplingPlan::~SamplingPlan() = default;

bool SamplingPlan::valid() const { return impl_ != nullptr; }

const SamplingInfo &SamplingPlan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("sampling plan is empty");
  }
  return impl_->info;
}

SamplingPlan prepare_sampling(const MetalContext &context,
                              const std::filesystem::path &metallib,
                              std::uint64_t vocabulary) {
  if (vocabulary == 0 ||
      vocabulary > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("sampling vocabulary is invalid");
  }
  context.register_library({.key = "b3.vocab.library", .path = metallib});
  static_cast<void>(context.warmup_pipeline({
      .key = "b3.vocab.sampling",
      .library_key = "b3.vocab.library",
      .function_name = "omlx_m5_sample_logits",
      .variant = "greedy_topk64",
  }));
  auto impl = std::make_unique<SamplingPlan::Impl>();
  impl->info = {
      .executor_key = "sampling.v" + std::to_string(vocabulary),
      .vocabulary = vocabulary,
      .logits_bytes = checked_multiply(vocabulary, 2, "sampling logits"),
  };
  impl->params_buffer = context.make_buffer(sizeof(SamplingParams));
  impl->command_plan = context.prepare_command_plan({
      .key = impl->info.executor_key,
      .dispatches = {{
          .pipeline_key = "b3.vocab.sampling",
          .grid = {.width = 256},
          .threads_per_threadgroup = {.width = 256},
          .resources =
              {
                  {.argument_index = 0,
                   .resource_slot = 0,
                   .minimum_bytes = impl->info.logits_bytes},
                  {.argument_index = 1,
                   .resource_slot = 1,
                   .minimum_bytes = impl->info.output_bytes},
                  {.argument_index = 2,
                   .resource_slot = 2,
                   .minimum_bytes = sizeof(SamplingParams)},
              },
      }},
  });
  return SamplingPlan(std::move(impl));
}

MetalSubmission
execute_sampling(const MetalContext &context, const SamplingPlan &plan,
                 const MetalBuffer &logits, std::uint64_t logits_offset,
                 const GpuSamplingOptions &options, MetalBuffer &token,
                 std::uint64_t token_offset) {
  if (plan.impl_ == nullptr || logits_offset % 2 != 0 ||
      options.contract != SamplingContract::LegacyRejectionV1 ||
      token_offset % 4 != 0 || !std::isfinite(options.temperature) ||
      options.temperature < 0.0F || !std::isfinite(options.top_p) ||
      options.top_p <= 0.0F || options.top_p > 1.0F ||
      (options.temperature > 0.0F &&
       (options.top_k == 0 || options.top_k > plan.impl_->info.maximum_top_k ||
        options.top_k > plan.impl_->info.vocabulary))) {
    throw std::invalid_argument("sampling invocation is invalid");
  }
  const SamplingParams params{
      .vocabulary = checked_u32(plan.impl_->info.vocabulary, "vocabulary"),
      .top_k = options.temperature > 0.0F ? options.top_k : 1U,
      .temperature = options.temperature,
      .top_p = options.top_p,
      .seed = options.seed,
      .stream = options.stream,
      .absolute_position = options.absolute_position,
  };
  std::vector<std::byte> params_bytes(sizeof(params));
  std::memcpy(params_bytes.data(), &params, sizeof(params));
  plan.impl_->params_buffer.write(params_bytes);
  const std::vector<MetalBufferBinding> bindings = {
      {0, &logits, logits_offset},
      {1, &token, token_offset},
      {2, &plan.impl_->params_buffer, 0},
  };
  return context.execute(plan.impl_->command_plan, bindings);
}

} // namespace omlx::m5
