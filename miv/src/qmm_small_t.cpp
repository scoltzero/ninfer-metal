#include "omlx_m5/qmm_small_t.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace omlx::m5 {
namespace {

struct QmmSmallTParams {
  std::uint32_t output_features = 0;
  std::uint32_t input_features = 0;
  std::uint32_t padded_input_features = 0;
  std::uint32_t tokens = 0;
};

static_assert(sizeof(QmmSmallTParams) == 16);

struct AffineGeometry {
  std::string bits;
  std::uint32_t group_size = 0;
  bool uses_high_plane = false;
};

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

const MaterializedPlaneView &plane(const MaterializedWeightView &weight,
                                   std::string_view kind) {
  const auto found = std::ranges::find_if(
      weight.planes,
      [&](const MaterializedPlaneView &value) { return value.kind == kind; });
  if (found == weight.planes.end() || found->buffer == nullptr ||
      found->bytes == 0) {
    throw std::invalid_argument("small-T weight plane is missing: " +
                                std::string(kind));
  }
  return *found;
}

AffineGeometry parse_geometry(std::string_view format) {
  constexpr std::string_view prefix = "oq_affine_q";
  constexpr std::string_view middle = "_g";
  constexpr std::string_view suffix = "_bf16_v1";
  if (!format.starts_with(prefix) || !format.ends_with(suffix)) {
    throw std::invalid_argument("Dense oQ small-T format is unsupported: " +
                                std::string(format));
  }
  const auto group_position = format.find(middle, prefix.size());
  if (group_position == std::string_view::npos) {
    throw std::invalid_argument("Dense oQ small-T format is malformed");
  }
  const auto bits =
      format.substr(prefix.size(), group_position - prefix.size());
  const auto group = format.substr(group_position + middle.size(),
                                   format.size() - suffix.size() -
                                       group_position - middle.size());
  if ((bits != "4" && bits != "5" && bits != "6" && bits != "8") ||
      (group != "64" && group != "128")) {
    throw std::invalid_argument(
        "Dense oQ small-T format geometry is unsupported");
  }
  return {
      .bits = std::string(bits),
      .group_size = group == "64" ? 64U : 128U,
      .uses_high_plane = bits == "5" || bits == "6",
  };
}

std::uint32_t capacity_for(std::uint32_t tokens) {
  if (tokens == 2) {
    return 2;
  }
  if (tokens <= 4) {
    return 4;
  }
  if (tokens <= 8) {
    return 8;
  }
  return 12;
}

std::vector<std::byte> constant_bytes(const QmmSmallTParams &params) {
  std::vector<std::byte> result(sizeof(params));
  std::memcpy(result.data(), &params, sizeof(params));
  return result;
}

std::string function_prefix(const AffineGeometry &geometry) {
  return "omlx_m5_oq_q" + geometry.bits + "_g" +
         std::to_string(geometry.group_size);
}

std::uint32_t family_priority(DenseOqSmallTFamily family) {
  switch (family) {
  case DenseOqSmallTFamily::split_k:
    return 10;
  case DenseOqSmallTFamily::cooperative:
    return 10;
  case DenseOqSmallTFamily::row_split:
    return 100;
  }
  throw std::invalid_argument("unknown Dense oQ small-T family");
}

KernelFamily dispatch_family(DenseOqSmallTFamily family) {
  switch (family) {
  case DenseOqSmallTFamily::row_split:
    return KernelFamily::row_split;
  case DenseOqSmallTFamily::split_k:
    return KernelFamily::split_k;
  case DenseOqSmallTFamily::cooperative:
    return KernelFamily::cooperative;
  }
  throw std::invalid_argument("unknown Dense oQ small-T family");
}

} // namespace

std::string_view to_string(DenseOqSmallTFamily value) {
  switch (value) {
  case DenseOqSmallTFamily::row_split:
    return "row_split";
  case DenseOqSmallTFamily::split_k:
    return "split_k";
  case DenseOqSmallTFamily::cooperative:
    return "cooperative";
  }
  return "unknown";
}

class DenseOqSmallTPlan::Impl {
public:
  struct ExecutorState {
    MetalCommandPlan command_plan;
    MetalBuffer partials;
  };

  MaterializedWeightView weight;
  DenseOqSmallTInfo info;
  MetalBuffer group_sums;
  bool uses_high_plane = false;
  std::vector<ExecutorState> executors;
};

DenseOqSmallTPlan::DenseOqSmallTPlan() = default;
DenseOqSmallTPlan::DenseOqSmallTPlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DenseOqSmallTPlan::DenseOqSmallTPlan(DenseOqSmallTPlan &&) noexcept = default;
DenseOqSmallTPlan &
DenseOqSmallTPlan::operator=(DenseOqSmallTPlan &&) noexcept = default;
DenseOqSmallTPlan::~DenseOqSmallTPlan() = default;

bool DenseOqSmallTPlan::valid() const { return impl_ != nullptr; }

const DenseOqSmallTInfo &DenseOqSmallTPlan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("Dense oQ small-T plan is empty");
  }
  return impl_->info;
}

std::vector<std::string> DenseOqSmallTPlan::installed_executor_keys() const {
  if (impl_ == nullptr) {
    throw std::logic_error("Dense oQ small-T plan is empty");
  }
  std::vector<std::string> result;
  result.reserve(impl_->info.executors.size());
  for (const auto &executor_info : impl_->info.executors) {
    result.push_back(executor_info.executor_key);
  }
  return result;
}

const DenseOqSmallTExecutorInfo &
DenseOqSmallTPlan::executor(std::string_view executor_key) const {
  if (impl_ == nullptr) {
    throw std::logic_error("Dense oQ small-T plan is empty");
  }
  const auto found = std::ranges::find_if(
      impl_->info.executors, [&](const DenseOqSmallTExecutorInfo &value) {
        return value.executor_key == executor_key;
      });
  if (found == impl_->info.executors.end()) {
    throw std::out_of_range("Dense oQ small-T executor is not installed");
  }
  return *found;
}

std::vector<KernelCandidate> DenseOqSmallTPlan::dispatch_candidates() const {
  if (impl_ == nullptr) {
    throw std::logic_error("Dense oQ small-T plan is empty");
  }
  std::vector<KernelCandidate> result;
  result.reserve(impl_->info.executors.size() * 2);
  for (const auto &executor_info : impl_->info.executors) {
    const auto make = [&](KernelAlignmentClass alignment, bool tail_safe,
                          std::uint32_t priority) {
      return KernelCandidate{
          .id = executor_info.executor_key + (tail_safe ? ".tail" : ".full"),
          .executor_key = executor_info.executor_key,
          .priority = priority,
          .family = tail_safe ? KernelFamily::tail_safe
                              : dispatch_family(executor_info.family),
          .format_id = impl_->info.format_id,
          .layout_id = impl_->info.layout_id,
          .activation = ActivationProfile::a16,
          .input_type = KernelDataType::bf16,
          .output_type = KernelDataType::bf16,
          .accumulator_type = KernelDataType::fp32,
          .geometry = KernelGeometry::dense,
          .output_features = impl_->info.output_features,
          .input_features = impl_->info.input_features,
          .min_tokens = impl_->info.tokens,
          .max_tokens = impl_->info.tokens,
          .fused_op = KernelFusedOp::linear,
          .alignment = alignment,
          .required_route = MetalExecutionRoute::classic_msl,
          .tail_safe = tail_safe,
      };
    };
    const auto priority = family_priority(executor_info.family);
    auto full = make(KernelAlignmentClass::full, false, priority);
    auto tail = make(KernelAlignmentClass::tail, true, priority + 1000);
    validate_kernel_candidate(full);
    validate_kernel_candidate(tail);
    result.push_back(std::move(full));
    result.push_back(std::move(tail));
  }
  return result;
}

DenseOqSmallTPlan
prepare_dense_oq_small_t(const MetalContext &context,
                         const std::filesystem::path &metallib,
                         MaterializedWeightView weight, std::uint32_t tokens) {
  if (tokens < 2 || tokens > 12) {
    throw std::invalid_argument("Dense oQ small-T requires T=2..12");
  }
  if (weight.parent_shape.size() != 2 || weight.row_count == 0 ||
      weight.row_start + weight.row_count > weight.parent_shape[0] ||
      weight.in_features == 0 || weight.padded_k < weight.in_features ||
      weight.padded_k % 128 != 0 || weight.planes.empty()) {
    throw std::invalid_argument("Dense oQ small-T weight view is invalid");
  }
  const auto geometry = parse_geometry(weight.format);
  if (weight.in_features % geometry.group_size != 0) {
    throw std::invalid_argument(
        "Dense oQ small-T K must be divisible by the affine group size");
  }
  const auto capacity = capacity_for(tokens);
  const auto logical_groups = weight.in_features / geometry.group_size;
  const auto input_values =
      checked_multiply(tokens, weight.in_features, "small-T input values");
  const auto output_values =
      checked_multiply(tokens, weight.row_count, "small-T output values");
  const auto group_sum_values =
      checked_multiply(tokens, logical_groups, "small-T group sums");
  const QmmSmallTParams params{
      .output_features = checked_u32(weight.row_count, "small-T N"),
      .input_features = checked_u32(weight.in_features, "small-T K"),
      .padded_input_features = checked_u32(weight.padded_k, "small-T Kp"),
      .tokens = tokens,
  };

  context.register_library({
      .key = "b2.qmm_small_t.library",
      .path = metallib,
  });
  const auto group_sum_pipeline =
      "b2.qmm_small_t.group_sum.g" + std::to_string(geometry.group_size);
  static_cast<void>(context.warmup_pipeline({
      .key = group_sum_pipeline,
      .library_key = "b2.qmm_small_t.library",
      .function_name = "omlx_m5_activation_group_sum_small_t_g" +
                       std::to_string(geometry.group_size),
      .variant = "t2_12",
  }));

  auto impl = std::make_unique<DenseOqSmallTPlan::Impl>();
  impl->weight = std::move(weight);
  impl->uses_high_plane = geometry.uses_high_plane;
  impl->info = {
      .format_id = impl->weight.format,
      .layout_id = impl->weight.layout,
      .output_features = impl->weight.row_count,
      .input_features = impl->weight.in_features,
      .tokens = tokens,
      .capacity = capacity,
      .input_bytes = checked_multiply(input_values, 2, "small-T input"),
      .output_bytes = checked_multiply(output_values, 2, "small-T output"),
      .group_sum_workspace_bytes =
          checked_multiply(group_sum_values, 4, "small-T group sums"),
  };
  for (const auto &selected_plane : impl->weight.planes) {
    impl->info.weight_plane_bytes += selected_plane.bytes;
  }
  impl->group_sums = context.make_buffer(impl->info.group_sum_workspace_bytes);

  const auto code_kind = geometry.bits == "8" ? "code" : "base_code";
  const auto &codes = plane(impl->weight, code_kind);
  const MaterializedPlaneView *high = &codes;
  if (geometry.bits == "5") {
    high = &plane(impl->weight, "high_bit");
  } else if (geometry.bits == "6") {
    high = &plane(impl->weight, "high_2bit");
  }
  const auto &scales = plane(impl->weight, "scale");
  const auto &biases = plane(impl->weight, "bias");

  const auto common_resources = [&](bool output_is_partial,
                                    std::uint64_t output_bytes) {
    std::vector<MetalResourceArgument> resources = {
        {.argument_index = 0,
         .resource_slot = 0,
         .minimum_bytes = impl->info.input_bytes},
        {.argument_index = 1, .resource_slot = 1, .minimum_bytes = codes.bytes},
        {.argument_index = 3,
         .resource_slot = 3,
         .minimum_bytes = scales.bytes},
        {.argument_index = 4,
         .resource_slot = 4,
         .minimum_bytes = biases.bytes},
        {.argument_index = 5,
         .resource_slot = 5,
         .minimum_bytes = impl->info.group_sum_workspace_bytes},
        {.argument_index = 6,
         .resource_slot = output_is_partial ? 7U : 6U,
         .minimum_bytes = output_bytes},
    };
    if (impl->uses_high_plane) {
      resources.insert(resources.begin() + 2, {.argument_index = 2,
                                               .resource_slot = 2,
                                               .minimum_bytes = high->bytes});
    }
    return resources;
  };
  const MetalDispatchDescriptor group_sum_dispatch{
      .pipeline_key = group_sum_pipeline,
      .grid = {.width = group_sum_values * 32},
      .threads_per_threadgroup = {.width = 32},
      .resources =
          {
              {.argument_index = 0,
               .resource_slot = 0,
               .minimum_bytes = impl->info.input_bytes},
              {.argument_index = 1,
               .resource_slot = 5,
               .minimum_bytes = impl->info.group_sum_workspace_bytes},
          },
      .constant_arguments = {{.argument_index = 2,
                              .bytes = constant_bytes(params)}},
  };
  const auto prefix = function_prefix(geometry);

  const auto add_row_or_cooperative = [&](DenseOqSmallTFamily family) {
    const auto suffix = family == DenseOqSmallTFamily::row_split
                            ? "row_c" + std::to_string(capacity)
                            : "coop_c12";
    const auto function = prefix + "_" + suffix;
    const auto executor_key =
        "dense_oq_small_t." + std::string(to_string(family)) + "." +
        impl->weight.format + ".n" + std::to_string(impl->weight.row_count) +
        ".k" + std::to_string(impl->weight.in_features) + ".t" +
        std::to_string(tokens);
    const auto pipeline_key =
        "b2.qmm_small_t." + impl->weight.format + "." + suffix;
    const auto pipeline = context.warmup_pipeline({
        .key = pipeline_key,
        .library_key = "b2.qmm_small_t.library",
        .function_name = function,
        .variant = "c" + std::to_string(capacity),
    });
    MetalDispatchDescriptor kernel_dispatch{
        .pipeline_key = pipeline_key,
        .grid = {.width = ((impl->weight.row_count + 3) / 4) * 128},
        .threads_per_threadgroup = {.width = 128},
        .resources = common_resources(false, impl->info.output_bytes),
        .constant_arguments = {{.argument_index = 7,
                                .bytes = constant_bytes(params)}},
    };
    auto command_plan = context.prepare_command_plan({
        .key = executor_key,
        .dispatches = {group_sum_dispatch, std::move(kernel_dispatch)},
    });
    impl->info.executors.push_back({
        .family = family,
        .executor_key = executor_key,
        .function_name = function,
        .capacity = capacity,
        .dispatch_count = 2,
        .static_threadgroup_memory_bytes =
            pipeline.static_threadgroup_memory_bytes,
    });
    impl->executors.push_back({.command_plan = std::move(command_plan)});
  };

  if (tokens <= 8) {
    add_row_or_cooperative(DenseOqSmallTFamily::row_split);
  }

  if ((geometry.bits == "5" || geometry.bits == "6") && tokens <= 6) {
    constexpr std::uint64_t split_count = 4;
    const auto partial_values =
        checked_multiply(checked_multiply(impl->weight.row_count, split_count,
                                          "small-T split rows"),
                         capacity, "small-T split capacity");
    const auto partial_bytes =
        checked_multiply(partial_values, 4, "small-T split workspace");
    const auto suffix = "split_c" + std::to_string(capacity);
    const auto function = prefix + "_" + suffix;
    const auto executor_key = "dense_oq_small_t.split_k." +
                              impl->weight.format + ".n" +
                              std::to_string(impl->weight.row_count) + ".k" +
                              std::to_string(impl->weight.in_features) + ".t" +
                              std::to_string(tokens);
    const auto pipeline_key =
        "b2.qmm_small_t." + impl->weight.format + "." + suffix;
    static_cast<void>(context.warmup_pipeline({
        .key = pipeline_key,
        .library_key = "b2.qmm_small_t.library",
        .function_name = function,
        .variant = "c" + std::to_string(capacity),
    }));
    const auto reduce_function =
        "omlx_m5_oq_split_reduce_c" + std::to_string(capacity);
    const auto reduce_pipeline =
        "b2.qmm_small_t.split_reduce.c" + std::to_string(capacity);
    static_cast<void>(context.warmup_pipeline({
        .key = reduce_pipeline,
        .library_key = "b2.qmm_small_t.library",
        .function_name = reduce_function,
        .variant = "c" + std::to_string(capacity),
    }));
    MetalDispatchDescriptor split_dispatch{
        .pipeline_key = pipeline_key,
        .grid = {.width = impl->weight.row_count * 128},
        .threads_per_threadgroup = {.width = 128},
        .resources = common_resources(true, partial_bytes),
        .constant_arguments = {{.argument_index = 7,
                                .bytes = constant_bytes(params)}},
    };
    MetalDispatchDescriptor reduce_dispatch{
        .pipeline_key = reduce_pipeline,
        .grid = {.width = output_values},
        .threads_per_threadgroup = {.width = 128},
        .resources =
            {
                {.argument_index = 0,
                 .resource_slot = 7,
                 .minimum_bytes = partial_bytes},
                {.argument_index = 1,
                 .resource_slot = 6,
                 .minimum_bytes = impl->info.output_bytes},
            },
        .constant_arguments = {{.argument_index = 2,
                                .bytes = constant_bytes(params)}},
    };
    auto command_plan = context.prepare_command_plan({
        .key = executor_key,
        .dispatches = {group_sum_dispatch, std::move(split_dispatch),
                       std::move(reduce_dispatch)},
    });
    impl->info.executors.push_back({
        .family = DenseOqSmallTFamily::split_k,
        .executor_key = executor_key,
        .function_name = function,
        .capacity = capacity,
        .partial_workspace_bytes = partial_bytes,
        .dispatch_count = 3,
    });
    impl->executors.push_back({
        .command_plan = std::move(command_plan),
        .partials = context.make_buffer(partial_bytes),
    });
  }

  if (tokens >= 9) {
    add_row_or_cooperative(DenseOqSmallTFamily::cooperative);
  }

  if (impl->info.executors.empty() ||
      impl->info.executors.size() != impl->executors.size()) {
    throw std::logic_error("Dense oQ small-T executor catalog is empty");
  }
  return DenseOqSmallTPlan(std::move(impl));
}

MetalSubmission execute_dense_oq_small_t(const MetalContext &context,
                                         const DenseOqSmallTPlan &plan,
                                         std::string_view executor_key,
                                         const MetalBuffer &input,
                                         std::uint64_t input_offset,
                                         MetalBuffer &output,
                                         std::uint64_t output_offset) {
  if (plan.impl_ == nullptr) {
    throw std::invalid_argument("Dense oQ small-T plan is empty");
  }
  if (input_offset % 2 != 0 || output_offset % 2 != 0) {
    throw std::invalid_argument(
        "Dense oQ small-T BF16 offsets must be two-byte aligned");
  }
  const auto found = std::ranges::find_if(
      plan.impl_->info.executors, [&](const DenseOqSmallTExecutorInfo &value) {
        return value.executor_key == executor_key;
      });
  if (found == plan.impl_->info.executors.end()) {
    throw std::out_of_range("Dense oQ small-T executor is not installed");
  }
  const auto index =
      static_cast<std::size_t>(found - plan.impl_->info.executors.begin());
  auto &state = plan.impl_->executors[index];
  const auto geometry = parse_geometry(plan.impl_->weight.format);
  const auto code_kind = geometry.bits == "8" ? "code" : "base_code";
  const auto &codes = plane(plan.impl_->weight, code_kind);
  const MaterializedPlaneView *high = &codes;
  if (geometry.bits == "5") {
    high = &plane(plan.impl_->weight, "high_bit");
  } else if (geometry.bits == "6") {
    high = &plane(plan.impl_->weight, "high_2bit");
  }
  const auto &scales = plane(plan.impl_->weight, "scale");
  const auto &biases = plane(plan.impl_->weight, "bias");
  std::vector<MetalBufferBinding> bindings = {
      {.resource_slot = 0, .buffer = &input, .offset = input_offset},
      {.resource_slot = 1, .buffer = codes.buffer, .offset = codes.offset},
  };
  if (plan.impl_->uses_high_plane) {
    bindings.push_back({2, high->buffer, high->offset});
  }
  bindings.push_back({3, scales.buffer, scales.offset});
  bindings.push_back({4, biases.buffer, biases.offset});
  bindings.push_back({5, &plan.impl_->group_sums, 0});
  bindings.push_back({6, &output, output_offset});
  if (found->family == DenseOqSmallTFamily::split_k) {
    bindings.push_back({7, &state.partials, 0});
  }
  return context.execute(state.command_plan, bindings);
}

} // namespace omlx::m5
