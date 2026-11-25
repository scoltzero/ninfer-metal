#include "omlx_m5/mxfp4_small_t.h"

#include <cstddef>
#include <cstring>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace omlx::m5 {
namespace {

struct Mxfp4SmallTParams {
  std::uint32_t output_features = 0;
  std::uint32_t input_features = 0;
  std::uint32_t padded_input_features = 0;
  std::uint32_t tokens = 0;
};

static_assert(sizeof(Mxfp4SmallTParams) == 16);

std::uint32_t checked_u32(std::uint64_t value, const char *label) {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error(std::string(label) + " exceeds uint32");
  }
  return static_cast<std::uint32_t>(value);
}

std::int32_t checked_i32(std::uint64_t value, const char *label) {
  if (value >
      static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::overflow_error(std::string(label) + " exceeds int32");
  }
  return static_cast<std::int32_t>(value);
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               const char *label) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error(std::string(label) + " overflow");
  }
  return left * right;
}

template <typename T> std::vector<std::byte> constant_bytes(const T &value) {
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
    throw std::invalid_argument("MXFP4 small-T plane is missing: " +
                                std::string(kind));
  }
  return *found;
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

ActivationProfile activation_for(DenseMxfp4SmallTFamily family) {
  switch (family) {
  case DenseMxfp4SmallTFamily::w4a16_simd:
  case DenseMxfp4SmallTFamily::w4a16_tensorops:
    return ActivationProfile::a16;
  case DenseMxfp4SmallTFamily::w4a4_simd:
  case DenseMxfp4SmallTFamily::w4a4_tensorops:
    return ActivationProfile::a4;
  }
  throw std::invalid_argument("unknown MXFP4 small-T family");
}

MetalExecutionRoute route_for(DenseMxfp4SmallTFamily family) {
  switch (family) {
  case DenseMxfp4SmallTFamily::w4a16_simd:
  case DenseMxfp4SmallTFamily::w4a4_simd:
    return MetalExecutionRoute::classic_msl;
  case DenseMxfp4SmallTFamily::w4a16_tensorops:
  case DenseMxfp4SmallTFamily::w4a4_tensorops:
    return MetalExecutionRoute::native_tensorops;
  }
  throw std::invalid_argument("unknown MXFP4 small-T family");
}

KernelFamily kernel_family(DenseMxfp4SmallTFamily family) {
  return route_for(family) == MetalExecutionRoute::native_tensorops
             ? KernelFamily::tensorops
             : KernelFamily::row_split;
}

std::uint32_t priority_for(DenseMxfp4SmallTFamily family) {
  switch (family) {
  case DenseMxfp4SmallTFamily::w4a4_tensorops:
    return 10;
  case DenseMxfp4SmallTFamily::w4a16_tensorops:
    return 20;
  case DenseMxfp4SmallTFamily::w4a16_simd:
  case DenseMxfp4SmallTFamily::w4a4_simd:
    return 100;
  }
  throw std::invalid_argument("unknown MXFP4 small-T family");
}

void validate_a4_view(const Mxfp4ActivationView &view, std::uint64_t code_bytes,
                      std::uint64_t scale_bytes) {
  if (view.group_size != 32 || view.codes.buffer == nullptr ||
      view.scales.buffer == nullptr || view.codes.bytes < code_bytes ||
      view.scales.bytes < scale_bytes) {
    throw std::invalid_argument("MXFP4 A4 activation view is invalid");
  }
}

} // namespace

class Mxfp4A4QuantizePlan::Impl {
public:
  Mxfp4A4QuantizeInfo info;
  MetalCommandPlan command_plan;
};

Mxfp4A4QuantizePlan::Mxfp4A4QuantizePlan() = default;
Mxfp4A4QuantizePlan::Mxfp4A4QuantizePlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
Mxfp4A4QuantizePlan::Mxfp4A4QuantizePlan(Mxfp4A4QuantizePlan &&) noexcept =
    default;
Mxfp4A4QuantizePlan &
Mxfp4A4QuantizePlan::operator=(Mxfp4A4QuantizePlan &&) noexcept = default;
Mxfp4A4QuantizePlan::~Mxfp4A4QuantizePlan() = default;

bool Mxfp4A4QuantizePlan::valid() const { return impl_ != nullptr; }

const Mxfp4A4QuantizeInfo &Mxfp4A4QuantizePlan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("MXFP4 A4 quantize plan is empty");
  }
  return impl_->info;
}

Mxfp4A4QuantizePlan
prepare_mxfp4_a4_quantizer(const MetalContext &context,
                           const std::filesystem::path &classic_metallib,
                           std::uint64_t input_features, std::uint32_t tokens) {
  if (tokens == 0 || input_features == 0 || input_features % 32 != 0) {
    throw std::invalid_argument(
        "MXFP4 A4 quantizer row/K domain is unsupported");
  }
  const auto values =
      checked_multiply(tokens, input_features, "A4 activation values");
  const Mxfp4SmallTParams params{
      .input_features = checked_u32(input_features, "A4 K"),
      .padded_input_features = checked_u32(input_features, "A4 Kp"),
      .tokens = tokens,
  };
  context.register_library({
      .key = "b2.mxfp4_small_t.classic.library",
      .path = classic_metallib,
  });
  static_cast<void>(context.warmup_pipeline({
      .key = "b2.mxfp4_small_t.quantize_a4",
      .library_key = "b2.mxfp4_small_t.classic.library",
      .function_name = "omlx_m5_mxfp4_quantize_a4",
      .variant = "g32",
  }));
  auto impl = std::make_unique<Mxfp4A4QuantizePlan::Impl>();
  impl->info = {
      .executor_key = "mxfp4_quantize_a4.k" + std::to_string(input_features) +
                      ".t" + std::to_string(tokens),
      .tokens = tokens,
      .input_features = input_features,
      .input_bytes = checked_multiply(values, 2, "A4 input"),
      .code_bytes = values / 2,
      .scale_bytes = values / 32,
      .dispatch_count = 1,
  };
  impl->command_plan = context.prepare_command_plan({
      .key = impl->info.executor_key,
      .dispatches = {{
          .pipeline_key = "b2.mxfp4_small_t.quantize_a4",
          .grid = {.width = impl->info.scale_bytes * 32},
          .threads_per_threadgroup = {.width = 32},
          .resources =
              {
                  {.argument_index = 0,
                   .resource_slot = 0,
                   .minimum_bytes = impl->info.input_bytes},
                  {.argument_index = 1,
                   .resource_slot = 1,
                   .minimum_bytes = impl->info.code_bytes},
                  {.argument_index = 2,
                   .resource_slot = 2,
                   .minimum_bytes = impl->info.scale_bytes},
              },
          .constant_arguments = {{.argument_index = 3,
                                  .bytes = constant_bytes(params)}},
      }},
  });
  return Mxfp4A4QuantizePlan(std::move(impl));
}

MetalSubmission quantize_mxfp4_a4(const MetalContext &context,
                                  const Mxfp4A4QuantizePlan &plan,
                                  const MetalBuffer &input,
                                  std::uint64_t input_offset,
                                  const Mxfp4ActivationView &output) {
  if (plan.impl_ == nullptr) {
    throw std::invalid_argument("MXFP4 A4 quantize plan is empty");
  }
  if (input_offset % 2 != 0) {
    throw std::invalid_argument("MXFP4 A16 input offset is misaligned");
  }
  validate_a4_view(output, plan.impl_->info.code_bytes,
                   plan.impl_->info.scale_bytes);
  const std::vector<MetalBufferBinding> bindings = {
      {0, &input, input_offset},
      {1, output.codes.buffer, output.codes.offset},
      {2, output.scales.buffer, output.scales.offset},
  };
  return context.execute(plan.impl_->command_plan, bindings);
}

std::string_view to_string(DenseMxfp4SmallTFamily value) {
  switch (value) {
  case DenseMxfp4SmallTFamily::w4a16_simd:
    return "w4a16_simd";
  case DenseMxfp4SmallTFamily::w4a4_simd:
    return "w4a4_simd";
  case DenseMxfp4SmallTFamily::w4a16_tensorops:
    return "w4a16_tensorops";
  case DenseMxfp4SmallTFamily::w4a4_tensorops:
    return "w4a4_tensorops";
  }
  return "unknown";
}

class DenseMxfp4SmallTPlan::Impl {
public:
  struct ExecutorState {
    MetalCommandPlan command_plan;
  };

  MaterializedWeightView weight;
  DenseMxfp4SmallTInfo info;
  std::vector<ExecutorState> executors;
};

DenseMxfp4SmallTPlan::DenseMxfp4SmallTPlan() = default;
DenseMxfp4SmallTPlan::DenseMxfp4SmallTPlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DenseMxfp4SmallTPlan::DenseMxfp4SmallTPlan(DenseMxfp4SmallTPlan &&) noexcept =
    default;
DenseMxfp4SmallTPlan &
DenseMxfp4SmallTPlan::operator=(DenseMxfp4SmallTPlan &&) noexcept = default;
DenseMxfp4SmallTPlan::~DenseMxfp4SmallTPlan() = default;

bool DenseMxfp4SmallTPlan::valid() const { return impl_ != nullptr; }

const DenseMxfp4SmallTInfo &DenseMxfp4SmallTPlan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("Dense MXFP4 small-T plan is empty");
  }
  return impl_->info;
}

std::vector<std::string> DenseMxfp4SmallTPlan::installed_executor_keys() const {
  if (impl_ == nullptr) {
    throw std::logic_error("Dense MXFP4 small-T plan is empty");
  }
  std::vector<std::string> result;
  for (const auto &executor_info : impl_->info.executors) {
    if (executor_info.installed) {
      result.push_back(executor_info.executor_key);
    }
  }
  return result;
}

const DenseMxfp4SmallTExecutorInfo &
DenseMxfp4SmallTPlan::executor(std::string_view executor_key) const {
  if (impl_ == nullptr) {
    throw std::logic_error("Dense MXFP4 small-T plan is empty");
  }
  const auto found = std::ranges::find_if(
      impl_->info.executors, [&](const DenseMxfp4SmallTExecutorInfo &value) {
        return value.executor_key == executor_key;
      });
  if (found == impl_->info.executors.end()) {
    throw std::out_of_range("Dense MXFP4 small-T executor is unknown");
  }
  return *found;
}

std::vector<KernelCandidate> DenseMxfp4SmallTPlan::dispatch_candidates() const {
  if (impl_ == nullptr) {
    throw std::logic_error("Dense MXFP4 small-T plan is empty");
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
                              : kernel_family(executor_info.family),
          .format_id = impl_->info.format_id,
          .layout_id = impl_->info.layout_id,
          .activation = executor_info.activation,
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
          .required_route = executor_info.required_route,
          .tail_safe = tail_safe,
      };
    };
    const auto priority = priority_for(executor_info.family);
    auto full = make(KernelAlignmentClass::full, false, priority);
    auto tail = make(KernelAlignmentClass::tail, true, priority + 1000);
    validate_kernel_candidate(full);
    validate_kernel_candidate(tail);
    result.push_back(std::move(full));
    result.push_back(std::move(tail));
  }
  return result;
}

DenseMxfp4SmallTPlan prepare_dense_mxfp4_small_t(
    const MetalContext &context, const std::filesystem::path &classic_metallib,
    const std::filesystem::path &tensorops_metallib,
    MaterializedWeightView weight, std::uint32_t tokens) {
  if (tokens < 2 || tokens > 12) {
    throw std::invalid_argument("Dense MXFP4 small-T requires T=2..12");
  }
  if (weight.format != "mxfp4_e2m1_g32_e8m0_v1" ||
      weight.layout != "mx_rowsplit_k128_v1" ||
      weight.parent_shape.size() != 2 || weight.row_count == 0 ||
      weight.row_start + weight.row_count > weight.parent_shape[0] ||
      weight.in_features == 0 || weight.in_features % 32 != 0 ||
      weight.padded_k < weight.in_features || weight.padded_k % 128 != 0) {
    throw std::invalid_argument("Dense MXFP4 small-T weight view is invalid");
  }
  const auto capacity = capacity_for(tokens);
  const auto input_values =
      checked_multiply(tokens, weight.in_features, "MXFP4 input values");
  const auto output_values =
      checked_multiply(tokens, weight.row_count, "MXFP4 output values");
  const Mxfp4SmallTParams params{
      .output_features = checked_u32(weight.row_count, "MXFP4 N"),
      .input_features = checked_u32(weight.in_features, "MXFP4 K"),
      .padded_input_features = checked_u32(weight.padded_k, "MXFP4 Kp"),
      .tokens = tokens,
  };
  context.register_library({
      .key = "b2.mxfp4_small_t.classic.library",
      .path = classic_metallib,
  });

  auto impl = std::make_unique<DenseMxfp4SmallTPlan::Impl>();
  impl->weight = std::move(weight);
  const auto &data = plane(impl->weight, "data");
  const auto &scales = plane(impl->weight, "scale");
  impl->info = {
      .format_id = impl->weight.format,
      .layout_id = impl->weight.layout,
      .output_features = impl->weight.row_count,
      .input_features = impl->weight.in_features,
      .tokens = tokens,
      .capacity = capacity,
      .input_a16_bytes = checked_multiply(input_values, 2, "MXFP4 A16 input"),
      .input_a4_code_bytes = input_values / 2,
      .input_a4_scale_bytes = input_values / 32,
      .output_bytes = checked_multiply(output_values, 2, "MXFP4 output"),
      .weight_plane_bytes = data.bytes + scales.bytes,
      .native_tensorops_runtime_eligible =
          context.capabilities().profile.native_tensorops_eligible,
  };

  const auto add_classic = [&](DenseMxfp4SmallTFamily family) {
    const auto activation = activation_for(family);
    const auto suffix =
        std::string(to_string(family)) + ".c" + std::to_string(capacity);
    const auto function =
        "omlx_m5_mxfp4_" +
        std::string(activation == ActivationProfile::a16 ? "w4a16_c"
                                                         : "w4a4_c") +
        std::to_string(capacity);
    const auto pipeline_key = "b2.mxfp4_small_t." + suffix;
    static_cast<void>(context.warmup_pipeline({
        .key = pipeline_key,
        .library_key = "b2.mxfp4_small_t.classic.library",
        .function_name = function,
        .variant = "c" + std::to_string(capacity),
    }));
    const auto executor_key = "dense_mxfp4_small_t." +
                              std::string(to_string(family)) + ".n" +
                              std::to_string(impl->weight.row_count) + ".k" +
                              std::to_string(impl->weight.in_features) + ".t" +
                              std::to_string(tokens);
    MetalDispatchDescriptor dispatch{
        .pipeline_key = pipeline_key,
        .grid = {.width = ((impl->weight.row_count + 3) / 4) * 128},
        .threads_per_threadgroup = {.width = 128},
    };
    if (activation == ActivationProfile::a16) {
      dispatch.resources = {
          {.argument_index = 0,
           .resource_slot = 0,
           .minimum_bytes = impl->info.input_a16_bytes},
          {.argument_index = 1,
           .resource_slot = 3,
           .minimum_bytes = data.bytes},
          {.argument_index = 2,
           .resource_slot = 4,
           .minimum_bytes = scales.bytes},
          {.argument_index = 3,
           .resource_slot = 5,
           .minimum_bytes = impl->info.output_bytes},
      };
      dispatch.constant_arguments = {
          {.argument_index = 4, .bytes = constant_bytes(params)}};
    } else {
      dispatch.resources = {
          {.argument_index = 0,
           .resource_slot = 1,
           .minimum_bytes = impl->info.input_a4_code_bytes},
          {.argument_index = 1,
           .resource_slot = 2,
           .minimum_bytes = impl->info.input_a4_scale_bytes},
          {.argument_index = 2,
           .resource_slot = 3,
           .minimum_bytes = data.bytes},
          {.argument_index = 3,
           .resource_slot = 4,
           .minimum_bytes = scales.bytes},
          {.argument_index = 4,
           .resource_slot = 5,
           .minimum_bytes = impl->info.output_bytes},
      };
      dispatch.constant_arguments = {
          {.argument_index = 5, .bytes = constant_bytes(params)}};
    }
    auto command_plan = context.prepare_command_plan({
        .key = executor_key,
        .dispatches = {std::move(dispatch)},
    });
    impl->info.executors.push_back({
        .family = family,
        .executor_key = executor_key,
        .function_name = function,
        .activation = activation,
        .required_route = MetalExecutionRoute::classic_msl,
        .capacity = capacity,
        .dispatch_count = 1,
        .installed = true,
    });
    impl->executors.push_back({.command_plan = std::move(command_plan)});
  };

  add_classic(DenseMxfp4SmallTFamily::w4a16_simd);
  add_classic(DenseMxfp4SmallTFamily::w4a4_simd);

  const auto add_tensorops = [&](DenseMxfp4SmallTFamily family) {
    const auto activation = activation_for(family);
    const auto function = activation == ActivationProfile::a16
                              ? "omlx_m5_mxfp4_w4a16_tensorops"
                              : "omlx_m5_mxfp4_w4a4_tensorops";
    const auto executor_key = "dense_mxfp4_small_t." +
                              std::string(to_string(family)) + ".n" +
                              std::to_string(impl->weight.row_count) + ".k" +
                              std::to_string(impl->weight.in_features) + ".t" +
                              std::to_string(tokens);
    DenseMxfp4SmallTExecutorInfo executor_info{
        .family = family,
        .executor_key = executor_key,
        .function_name = function,
        .activation = activation,
        .required_route = MetalExecutionRoute::native_tensorops,
        .capacity = 16,
        .dispatch_count = 1,
        .installed = false,
    };
    MetalCommandPlan command_plan;
    if (impl->info.native_tensorops_runtime_eligible) {
      if (tensorops_metallib.empty()) {
        throw std::invalid_argument(
            "TensorOps runtime is eligible but its metallib is missing");
      }
      context.register_library({
          .key = "b2.mxfp4_small_t.tensorops.library",
          .path = tensorops_metallib,
      });
      const auto pipeline_key =
          "b2.mxfp4_small_t." + std::string(to_string(family));
      static_cast<void>(context.warmup_pipeline({
          .key = pipeline_key,
          .library_key = "b2.mxfp4_small_t.tensorops.library",
          .function_name = function,
          .variant = "m16n32",
      }));
      MetalDispatchDescriptor dispatch{
          .pipeline_key = pipeline_key,
          .grid = {.width = ((impl->weight.row_count + 31) / 32) * 32},
          .threads_per_threadgroup = {.width = 32},
      };
      if (activation == ActivationProfile::a16) {
        const auto k = checked_i32(impl->weight.padded_k, "TensorOps K");
        const auto n = checked_i32(impl->weight.row_count, "TensorOps N");
        const auto m = static_cast<std::int32_t>(tokens);
        const auto logical_k =
            checked_i32(impl->weight.in_features, "TensorOps logical K");
        dispatch.resources = {
            {.argument_index = 0,
             .resource_slot = 3,
             .minimum_bytes = data.bytes},
            {.argument_index = 1,
             .resource_slot = 4,
             .minimum_bytes = scales.bytes},
            {.argument_index = 2,
             .resource_slot = 0,
             .minimum_bytes = impl->info.input_a16_bytes},
            {.argument_index = 3,
             .resource_slot = 5,
             .minimum_bytes = impl->info.output_bytes},
        };
        dispatch.constant_arguments = {
            {.argument_index = 4, .bytes = constant_bytes(k)},
            {.argument_index = 5, .bytes = constant_bytes(n)},
            {.argument_index = 6, .bytes = constant_bytes(m)},
            {.argument_index = 7, .bytes = constant_bytes(logical_k)},
        };
      } else {
        dispatch.resources = {
            {.argument_index = 0,
             .resource_slot = 3,
             .minimum_bytes = data.bytes},
            {.argument_index = 1,
             .resource_slot = 4,
             .minimum_bytes = scales.bytes},
            {.argument_index = 2,
             .resource_slot = 1,
             .minimum_bytes = impl->info.input_a4_code_bytes},
            {.argument_index = 3,
             .resource_slot = 2,
             .minimum_bytes = impl->info.input_a4_scale_bytes},
            {.argument_index = 4,
             .resource_slot = 5,
             .minimum_bytes = impl->info.output_bytes},
        };
        dispatch.constant_arguments = {
            {.argument_index = 5, .bytes = constant_bytes(params)}};
      }
      command_plan = context.prepare_command_plan({
          .key = executor_key,
          .dispatches = {std::move(dispatch)},
      });
      executor_info.installed = true;
    }
    impl->info.executors.push_back(std::move(executor_info));
    impl->executors.push_back({.command_plan = std::move(command_plan)});
  };

  add_tensorops(DenseMxfp4SmallTFamily::w4a16_tensorops);
  add_tensorops(DenseMxfp4SmallTFamily::w4a4_tensorops);
  return DenseMxfp4SmallTPlan(std::move(impl));
}

MetalSubmission execute_dense_mxfp4_small_t_a16(
    const MetalContext &context, const DenseMxfp4SmallTPlan &plan,
    std::string_view executor_key, const MetalBuffer &input,
    std::uint64_t input_offset, MetalBuffer &output,
    std::uint64_t output_offset) {
  if (plan.impl_ == nullptr || input_offset % 2 != 0 ||
      output_offset % 2 != 0) {
    throw std::invalid_argument("Dense MXFP4 W4A16 invocation is invalid");
  }
  const auto &executor_info = plan.executor(executor_key);
  if (!executor_info.installed ||
      executor_info.activation != ActivationProfile::a16) {
    throw std::invalid_argument("Dense MXFP4 W4A16 executor is unavailable");
  }
  const auto index = static_cast<std::size_t>(
      &executor_info - plan.impl_->info.executors.data());
  const auto &data = plane(plan.impl_->weight, "data");
  const auto &scales = plane(plan.impl_->weight, "scale");
  const std::vector<MetalBufferBinding> bindings = {
      {0, &input, input_offset},
      {3, data.buffer, data.offset},
      {4, scales.buffer, scales.offset},
      {5, &output, output_offset},
  };
  return context.execute(plan.impl_->executors[index].command_plan, bindings);
}

MetalSubmission execute_dense_mxfp4_small_t_a4(const MetalContext &context,
                                               const DenseMxfp4SmallTPlan &plan,
                                               std::string_view executor_key,
                                               const Mxfp4ActivationView &input,
                                               MetalBuffer &output,
                                               std::uint64_t output_offset) {
  if (plan.impl_ == nullptr || output_offset % 2 != 0) {
    throw std::invalid_argument("Dense MXFP4 W4A4 invocation is invalid");
  }
  validate_a4_view(input, plan.impl_->info.input_a4_code_bytes,
                   plan.impl_->info.input_a4_scale_bytes);
  const auto &executor_info = plan.executor(executor_key);
  if (!executor_info.installed ||
      executor_info.activation != ActivationProfile::a4) {
    throw std::invalid_argument("Dense MXFP4 W4A4 executor is unavailable");
  }
  const auto index = static_cast<std::size_t>(
      &executor_info - plan.impl_->info.executors.data());
  const auto &data = plane(plan.impl_->weight, "data");
  const auto &scales = plane(plan.impl_->weight, "scale");
  const std::vector<MetalBufferBinding> bindings = {
      {1, input.codes.buffer, input.codes.offset},
      {2, input.scales.buffer, input.scales.offset},
      {3, data.buffer, data.offset},
      {4, scales.buffer, scales.offset},
      {5, &output, output_offset},
  };
  return context.execute(plan.impl_->executors[index].command_plan, bindings);
}

} // namespace omlx::m5
