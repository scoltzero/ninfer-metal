#include "omlx_m5/qmv.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace omlx::m5 {
namespace {

struct QmvT1Params {
  std::uint32_t output_features = 0;
  std::uint32_t input_features = 0;
  std::uint32_t padded_input_features = 0;
  std::uint32_t reserved = 0;
};

static_assert(sizeof(QmvT1Params) == 16);

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
    throw std::invalid_argument("QMV weight plane is missing: " +
                                std::string(kind));
  }
  return *found;
}

std::string function_name(std::string_view format) {
  if (format == "direct_bf16_v1") {
    return "omlx_m5_bf16_qmv_t1";
  }
  if (format == "mxfp4_e2m1_g32_e8m0_v1") {
    return "omlx_m5_mxfp4_qmv_t1";
  }
  constexpr std::string_view prefix = "oq_affine_q";
  constexpr std::string_view middle = "_g";
  constexpr std::string_view suffix = "_bf16_v1";
  if (!format.starts_with(prefix) || !format.ends_with(suffix)) {
    throw std::invalid_argument("unsupported Dense T=1 QMV format: " +
                                std::string(format));
  }
  const auto group_position = format.find(middle, prefix.size());
  if (group_position == std::string_view::npos) {
    throw std::invalid_argument("malformed affine QMV format");
  }
  const auto bits =
      format.substr(prefix.size(), group_position - prefix.size());
  const auto group = format.substr(group_position + middle.size(),
                                   format.size() - suffix.size() -
                                       group_position - middle.size());
  if ((bits != "4" && bits != "5" && bits != "6" && bits != "8") ||
      (group != "64" && group != "128")) {
    throw std::invalid_argument("unsupported affine QMV format geometry");
  }
  return "omlx_m5_oq_q" + std::string(bits) + "_g" + std::string(group) +
         "_qmv_t1";
}

std::uint32_t affine_group_size(std::string_view format) {
  if (format.find("_g64_") != std::string_view::npos) {
    return 64;
  }
  if (format.find("_g128_") != std::string_view::npos) {
    return 128;
  }
  throw std::invalid_argument("affine QMV group size is unsupported");
}

std::vector<std::byte> constant_bytes(const QmvT1Params &params) {
  std::vector<std::byte> result(sizeof(params));
  std::memcpy(result.data(), &params, sizeof(params));
  return result;
}

} // namespace

class DenseQmvT1Plan::Impl {
public:
  MaterializedWeightView weight;
  DenseQmvT1Info info;
  MetalCommandPlan command_plan;
  MetalBuffer group_sums;
  bool uses_high_plane = false;
};

DenseQmvT1Plan::DenseQmvT1Plan() = default;
DenseQmvT1Plan::DenseQmvT1Plan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DenseQmvT1Plan::DenseQmvT1Plan(DenseQmvT1Plan &&) noexcept = default;
DenseQmvT1Plan &DenseQmvT1Plan::operator=(DenseQmvT1Plan &&) noexcept = default;
DenseQmvT1Plan::~DenseQmvT1Plan() = default;

bool DenseQmvT1Plan::valid() const { return impl_ != nullptr; }

const DenseQmvT1Info &DenseQmvT1Plan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("Dense QMV T=1 plan is empty");
  }
  return impl_->info;
}

std::vector<KernelCandidate>
DenseQmvT1Plan::dispatch_candidates(std::uint32_t full_priority,
                                    std::uint32_t tail_priority) const {
  if (impl_ == nullptr) {
    throw std::logic_error("Dense QMV T=1 plan is empty");
  }
  const auto make = [&](std::string id, std::uint32_t priority,
                        KernelAlignmentClass alignment, bool tail_safe) {
    return KernelCandidate{
        .id = std::move(id),
        .executor_key = impl_->info.executor_key,
        .priority = priority,
        .family = tail_safe ? KernelFamily::tail_safe : KernelFamily::qmv,
        .format_id = impl_->info.format_id,
        .layout_id = impl_->info.layout_id,
        .activation = ActivationProfile::a16,
        .input_type = KernelDataType::bf16,
        .output_type = KernelDataType::bf16,
        .accumulator_type = KernelDataType::fp32,
        .geometry = KernelGeometry::dense,
        .output_features = impl_->info.output_features,
        .input_features = impl_->info.input_features,
        .min_tokens = 1,
        .max_tokens = 1,
        .fused_op = KernelFusedOp::linear,
        .alignment = alignment,
        .required_route = MetalExecutionRoute::classic_msl,
        .tail_safe = tail_safe,
    };
  };
  auto full = make(impl_->info.executor_key + ".full", full_priority,
                   KernelAlignmentClass::full, false);
  auto tail = make(impl_->info.executor_key + ".tail", tail_priority,
                   KernelAlignmentClass::tail, true);
  validate_kernel_candidate(full);
  validate_kernel_candidate(tail);
  return {std::move(full), std::move(tail)};
}

DenseQmvT1Plan prepare_dense_qmv_t1(const MetalContext &context,
                                    const std::filesystem::path &metallib,
                                    MaterializedWeightView weight) {
  if (weight.format == "direct_bf16_v1" && weight.padded_k == 0) {
    weight.padded_k = weight.in_features;
  }
  if (weight.parent_shape.size() != 2 || weight.row_count == 0 ||
      weight.row_start + weight.row_count > weight.parent_shape[0] ||
      weight.in_features == 0 || weight.padded_k < weight.in_features ||
      weight.padded_k % 128 != 0 || weight.planes.empty()) {
    throw std::invalid_argument("Dense T=1 QMV weight view is invalid");
  }
  const auto function = function_name(weight.format);
  if (weight.format == "mxfp4_e2m1_g32_e8m0_v1") {
    if (weight.in_features % 32 != 0) {
      throw std::invalid_argument(
          "Dense T=1 MXFP4 QMV requires K divisible by group size 32");
    }
  } else if (weight.format != "direct_bf16_v1") {
    if (weight.in_features % affine_group_size(weight.format) != 0) {
      throw std::invalid_argument(
          "Dense T=1 affine QMV requires K divisible by group size");
    }
  }
  const auto executor_key = "dense_qmv_t1." + weight.format + ".n" +
                            std::to_string(weight.row_count) + ".k" +
                            std::to_string(weight.in_features);
  const auto pipeline_key = "b2.qmv_t1." + weight.format;
  context.register_library({
      .key = "b2.qmv_t1.library",
      .path = metallib,
  });
  static_cast<void>(context.warmup_pipeline({
      .key = pipeline_key,
      .library_key = "b2.qmv_t1.library",
      .function_name = function,
      .variant = "t1",
  }));

  auto impl = std::make_unique<DenseQmvT1Plan::Impl>();
  impl->weight = std::move(weight);
  impl->info = {
      .executor_key = executor_key,
      .pipeline_key = pipeline_key,
      .function_name = function,
      .format_id = impl->weight.format,
      .layout_id = impl->weight.layout,
      .output_features = impl->weight.row_count,
      .input_features = impl->weight.in_features,
      .input_bytes = checked_multiply(impl->weight.in_features, 2, "QMV input"),
      .output_bytes = checked_multiply(impl->weight.row_count, 2, "QMV output"),
  };
  for (const auto &selected_plane : impl->weight.planes) {
    impl->info.weight_plane_bytes += selected_plane.bytes;
  }

  const QmvT1Params params{
      .output_features = checked_u32(impl->weight.row_count, "QMV N"),
      .input_features = checked_u32(impl->weight.in_features, "QMV K"),
      .padded_input_features = checked_u32(impl->weight.padded_k, "QMV Kp"),
  };
  MetalDispatchDescriptor dispatch{
      .pipeline_key = pipeline_key,
      .grid = {.width = ((impl->weight.row_count + 3) / 4) * 128},
      .threads_per_threadgroup = {.width = 128},
  };
  if (impl->weight.format == "direct_bf16_v1") {
    const auto &data = plane(impl->weight, "data");
    dispatch.resources = {
        {.argument_index = 0,
         .resource_slot = 0,
         .minimum_bytes = impl->info.input_bytes},
        {.argument_index = 1, .resource_slot = 1, .minimum_bytes = data.bytes},
        {.argument_index = 2,
         .resource_slot = 3,
         .minimum_bytes = impl->info.output_bytes},
    };
    dispatch.constant_arguments = {
        {.argument_index = 3, .bytes = constant_bytes(params)}};
    impl->info.dispatch_count = 1;
    impl->command_plan = context.prepare_command_plan({
        .key = executor_key,
        .dispatches = {std::move(dispatch)},
    });
  } else if (impl->weight.format == "mxfp4_e2m1_g32_e8m0_v1") {
    const auto &data = plane(impl->weight, "data");
    const auto &scale = plane(impl->weight, "scale");
    dispatch.resources = {
        {.argument_index = 0,
         .resource_slot = 0,
         .minimum_bytes = impl->info.input_bytes},
        {.argument_index = 1, .resource_slot = 1, .minimum_bytes = data.bytes},
        {.argument_index = 2, .resource_slot = 2, .minimum_bytes = scale.bytes},
        {.argument_index = 3,
         .resource_slot = 3,
         .minimum_bytes = impl->info.output_bytes},
    };
    dispatch.constant_arguments = {
        {.argument_index = 4, .bytes = constant_bytes(params)}};
    impl->info.dispatch_count = 1;
    impl->command_plan = context.prepare_command_plan({
        .key = executor_key,
        .dispatches = {std::move(dispatch)},
    });
  } else {
    const auto group_size = affine_group_size(impl->weight.format);
    const auto logical_groups = impl->weight.in_features / group_size;
    impl->info.workspace_bytes =
        checked_multiply(logical_groups, 4, "QMV group sums");
    impl->group_sums = context.make_buffer(impl->info.workspace_bytes);
    const auto group_sum_pipeline =
        "b2.qmv_t1.group_sum.g" + std::to_string(group_size);
    static_cast<void>(context.warmup_pipeline({
        .key = group_sum_pipeline,
        .library_key = "b2.qmv_t1.library",
        .function_name =
            "omlx_m5_activation_group_sum_g" + std::to_string(group_size),
        .variant = "t1",
    }));
    const auto code_kind = impl->weight.format.find("_q8_") != std::string::npos
                               ? "code"
                               : "base_code";
    const auto &codes = plane(impl->weight, code_kind);
    const MaterializedPlaneView *high = &codes;
    if (impl->weight.format.find("_q5_") != std::string::npos) {
      high = &plane(impl->weight, "high_bit");
    } else if (impl->weight.format.find("_q6_") != std::string::npos) {
      high = &plane(impl->weight, "high_2bit");
    }
    const auto &scale = plane(impl->weight, "scale");
    const auto &bias = plane(impl->weight, "bias");
    dispatch.resources = {
        {.argument_index = 0,
         .resource_slot = 0,
         .minimum_bytes = impl->info.input_bytes},
        {.argument_index = 1, .resource_slot = 1, .minimum_bytes = codes.bytes},
        {.argument_index = 3, .resource_slot = 3, .minimum_bytes = scale.bytes},
        {.argument_index = 4, .resource_slot = 4, .minimum_bytes = bias.bytes},
        {.argument_index = 5,
         .resource_slot = 5,
         .minimum_bytes = impl->info.workspace_bytes},
        {.argument_index = 6,
         .resource_slot = 6,
         .minimum_bytes = impl->info.output_bytes},
    };
    impl->uses_high_plane = high != &codes;
    if (impl->uses_high_plane) {
      dispatch.resources.insert(dispatch.resources.begin() + 2,
                                {.argument_index = 2,
                                 .resource_slot = 2,
                                 .minimum_bytes = high->bytes});
    }
    dispatch.constant_arguments = {
        {.argument_index = 7, .bytes = constant_bytes(params)}};
    MetalDispatchDescriptor group_sum_dispatch{
        .pipeline_key = group_sum_pipeline,
        .grid = {.width = logical_groups * 32},
        .threads_per_threadgroup = {.width = 32},
        .resources =
            {
                {.argument_index = 0,
                 .resource_slot = 0,
                 .minimum_bytes = impl->info.input_bytes},
                {.argument_index = 1,
                 .resource_slot = 5,
                 .minimum_bytes = impl->info.workspace_bytes},
            },
    };
    impl->info.dispatch_count = 2;
    impl->command_plan = context.prepare_command_plan({
        .key = executor_key,
        .dispatches = {std::move(group_sum_dispatch), std::move(dispatch)},
    });
  }
  return DenseQmvT1Plan(std::move(impl));
}

MetalSubmission
execute_dense_qmv_t1(const MetalContext &context, const DenseQmvT1Plan &plan,
                     const MetalBuffer &input, std::uint64_t input_offset,
                     MetalBuffer &output, std::uint64_t output_offset) {
  if (plan.impl_ == nullptr) {
    throw std::invalid_argument("Dense QMV T=1 plan is empty");
  }
  std::vector<MetalBufferBinding> bindings = {
      {.resource_slot = 0, .buffer = &input, .offset = input_offset}};
  if (plan.impl_->weight.format == "direct_bf16_v1") {
    const auto &data = plane(plan.impl_->weight, "data");
    bindings.push_back({1, data.buffer, data.offset});
    bindings.push_back({3, &output, output_offset});
  } else if (plan.impl_->weight.format == "mxfp4_e2m1_g32_e8m0_v1") {
    const auto &data = plane(plan.impl_->weight, "data");
    const auto &scale = plane(plan.impl_->weight, "scale");
    bindings.push_back({1, data.buffer, data.offset});
    bindings.push_back({2, scale.buffer, scale.offset});
    bindings.push_back({3, &output, output_offset});
  } else {
    const auto code_kind =
        plan.impl_->weight.format.find("_q8_") != std::string::npos
            ? "code"
            : "base_code";
    const auto &codes = plane(plan.impl_->weight, code_kind);
    const MaterializedPlaneView *high = &codes;
    if (plan.impl_->weight.format.find("_q5_") != std::string::npos) {
      high = &plane(plan.impl_->weight, "high_bit");
    } else if (plan.impl_->weight.format.find("_q6_") != std::string::npos) {
      high = &plane(plan.impl_->weight, "high_2bit");
    }
    const auto &scale = plane(plan.impl_->weight, "scale");
    const auto &bias = plane(plan.impl_->weight, "bias");
    bindings.push_back({1, codes.buffer, codes.offset});
    if (plan.impl_->uses_high_plane) {
      bindings.push_back({2, high->buffer, high->offset});
    }
    bindings.push_back({3, scale.buffer, scale.offset});
    bindings.push_back({4, bias.buffer, bias.offset});
    bindings.push_back({5, &plan.impl_->group_sums, 0});
    bindings.push_back({6, &output, output_offset});
  }
  return context.execute(plan.impl_->command_plan, bindings);
}

} // namespace omlx::m5
