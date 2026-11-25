#include "omlx_m5/qmm_broad.h"

#include <cstddef>
#include <cstring>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace omlx::m5 {
namespace {

struct QmmBroadParams {
  std::uint32_t output_features = 0;
  std::uint32_t input_features = 0;
  std::uint32_t padded_input_features = 0;
  std::uint32_t tokens = 0;
};

static_assert(sizeof(QmmBroadParams) == 16);

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
    throw std::invalid_argument("broad QMM plane is missing: " +
                                std::string(kind));
  }
  return *found;
}

AffineGeometry parse_affine(std::string_view format) {
  constexpr std::string_view prefix = "oq_affine_q";
  constexpr std::string_view middle = "_g";
  constexpr std::string_view suffix = "_bf16_v1";
  if (!format.starts_with(prefix) || !format.ends_with(suffix)) {
    throw std::invalid_argument("broad affine format is unsupported");
  }
  const auto group_position = format.find(middle, prefix.size());
  if (group_position == std::string_view::npos) {
    throw std::invalid_argument("broad affine format is malformed");
  }
  const auto bits =
      format.substr(prefix.size(), group_position - prefix.size());
  const auto group = format.substr(group_position + middle.size(),
                                   format.size() - suffix.size() -
                                       group_position - middle.size());
  if ((bits != "4" && bits != "5" && bits != "6" && bits != "8") ||
      (group != "64" && group != "128")) {
    throw std::invalid_argument("broad affine geometry is unsupported");
  }
  return {
      .bits = std::string(bits),
      .group_size = group == "64" ? 64U : 128U,
      .uses_high_plane = bits == "5" || bits == "6",
  };
}

ActivationProfile activation_for(DenseQmmBroadFamily family) {
  switch (family) {
  case DenseQmmBroadFamily::oq_cooperative:
  case DenseQmmBroadFamily::mxfp4_w4a16_simd:
  case DenseQmmBroadFamily::mxfp4_w4a16_tensorops:
    return ActivationProfile::a16;
  case DenseQmmBroadFamily::mxfp4_w4a4_simd:
  case DenseQmmBroadFamily::mxfp4_w4a4_tensorops:
    return ActivationProfile::a4;
  }
  throw std::invalid_argument("unknown broad QMM family");
}

MetalExecutionRoute route_for(DenseQmmBroadFamily family) {
  switch (family) {
  case DenseQmmBroadFamily::oq_cooperative:
  case DenseQmmBroadFamily::mxfp4_w4a16_simd:
  case DenseQmmBroadFamily::mxfp4_w4a4_simd:
    return MetalExecutionRoute::classic_msl;
  case DenseQmmBroadFamily::mxfp4_w4a16_tensorops:
  case DenseQmmBroadFamily::mxfp4_w4a4_tensorops:
    return MetalExecutionRoute::native_tensorops;
  }
  throw std::invalid_argument("unknown broad QMM family");
}

KernelFamily kernel_family(DenseQmmBroadFamily family) {
  return route_for(family) == MetalExecutionRoute::native_tensorops
             ? KernelFamily::tensorops
             : KernelFamily::cooperative;
}

std::uint32_t priority_for(DenseQmmBroadFamily family) {
  switch (family) {
  case DenseQmmBroadFamily::mxfp4_w4a4_tensorops:
    return 10;
  case DenseQmmBroadFamily::mxfp4_w4a16_tensorops:
    return 20;
  case DenseQmmBroadFamily::oq_cooperative:
  case DenseQmmBroadFamily::mxfp4_w4a16_simd:
  case DenseQmmBroadFamily::mxfp4_w4a4_simd:
    return 100;
  }
  throw std::invalid_argument("unknown broad QMM family");
}

bool is_mxfp4(std::string_view format) {
  return format == "mxfp4_e2m1_g32_e8m0_v1";
}

void validate_a4_view(const Mxfp4ActivationView &view, std::uint64_t code_bytes,
                      std::uint64_t scale_bytes) {
  if (view.group_size != 32 || view.codes.buffer == nullptr ||
      view.scales.buffer == nullptr || view.codes.bytes < code_bytes ||
      view.scales.bytes < scale_bytes) {
    throw std::invalid_argument("broad A4 activation view is invalid");
  }
}

} // namespace

bool dense_broad_tokens_supported(std::uint32_t tokens) {
  return (tokens >= 13 && tokens <= 128) || tokens == 512 || tokens == 1024;
}

std::string_view to_string(DenseQmmBroadFamily value) {
  switch (value) {
  case DenseQmmBroadFamily::oq_cooperative:
    return "oq_cooperative";
  case DenseQmmBroadFamily::mxfp4_w4a16_simd:
    return "mxfp4_w4a16_simd";
  case DenseQmmBroadFamily::mxfp4_w4a4_simd:
    return "mxfp4_w4a4_simd";
  case DenseQmmBroadFamily::mxfp4_w4a16_tensorops:
    return "mxfp4_w4a16_tensorops";
  case DenseQmmBroadFamily::mxfp4_w4a4_tensorops:
    return "mxfp4_w4a4_tensorops";
  }
  return "unknown";
}

class DenseQmmBroadPlan::Impl {
public:
  struct ExecutorState {
    MetalCommandPlan command_plan;
  };

  MaterializedWeightView weight;
  DenseQmmBroadInfo info;
  MetalBuffer group_sums;
  bool mxfp4 = false;
  bool uses_high_plane = false;
  std::vector<ExecutorState> executors;
};

DenseQmmBroadPlan::DenseQmmBroadPlan() = default;
DenseQmmBroadPlan::DenseQmmBroadPlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DenseQmmBroadPlan::DenseQmmBroadPlan(DenseQmmBroadPlan &&) noexcept = default;
DenseQmmBroadPlan &
DenseQmmBroadPlan::operator=(DenseQmmBroadPlan &&) noexcept = default;
DenseQmmBroadPlan::~DenseQmmBroadPlan() = default;

bool DenseQmmBroadPlan::valid() const { return impl_ != nullptr; }

const DenseQmmBroadInfo &DenseQmmBroadPlan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("broad QMM plan is empty");
  }
  return impl_->info;
}

std::vector<std::string> DenseQmmBroadPlan::installed_executor_keys() const {
  if (impl_ == nullptr) {
    throw std::logic_error("broad QMM plan is empty");
  }
  std::vector<std::string> result;
  for (const auto &executor_info : impl_->info.executors) {
    if (executor_info.installed) {
      result.push_back(executor_info.executor_key);
    }
  }
  return result;
}

const DenseQmmBroadExecutorInfo &
DenseQmmBroadPlan::executor(std::string_view executor_key) const {
  if (impl_ == nullptr) {
    throw std::logic_error("broad QMM plan is empty");
  }
  const auto found = std::ranges::find_if(
      impl_->info.executors, [&](const DenseQmmBroadExecutorInfo &value) {
        return value.executor_key == executor_key;
      });
  if (found == impl_->info.executors.end()) {
    throw std::out_of_range("broad QMM executor is unknown");
  }
  return *found;
}

std::vector<KernelCandidate> DenseQmmBroadPlan::dispatch_candidates() const {
  if (impl_ == nullptr) {
    throw std::logic_error("broad QMM plan is empty");
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

DenseQmmBroadPlan
prepare_dense_qmm_broad(const MetalContext &context,
                        const std::filesystem::path &broad_metallib,
                        const std::filesystem::path &tensorops_metallib,
                        MaterializedWeightView weight, std::uint32_t tokens) {
  if (!dense_broad_tokens_supported(tokens) ||
      weight.parent_shape.size() != 2 || weight.row_count == 0 ||
      weight.row_start + weight.row_count > weight.parent_shape[0] ||
      weight.in_features == 0 || weight.padded_k < weight.in_features ||
      weight.padded_k % 128 != 0 || weight.planes.empty()) {
    throw std::invalid_argument("broad QMM domain is invalid");
  }
  const bool mxfp4 = is_mxfp4(weight.format);
  std::optional<AffineGeometry> affine;
  if (mxfp4) {
    if (weight.layout != "mx_rowsplit_k128_v1" ||
        weight.in_features % 32 != 0) {
      throw std::invalid_argument("broad MXFP4 geometry is invalid");
    }
  } else {
    affine = parse_affine(weight.format);
    if (weight.layout != "oq_rowsplit_k128_v1" ||
        weight.in_features % affine->group_size != 0) {
      throw std::invalid_argument("broad affine geometry is invalid");
    }
  }
  const auto input_values =
      checked_multiply(tokens, weight.in_features, "broad input values");
  const auto output_values =
      checked_multiply(tokens, weight.row_count, "broad output values");
  const QmmBroadParams params{
      .output_features = checked_u32(weight.row_count, "broad N"),
      .input_features = checked_u32(weight.in_features, "broad K"),
      .padded_input_features = checked_u32(weight.padded_k, "broad Kp"),
      .tokens = tokens,
  };
  context.register_library({
      .key = "b2.qmm_broad.classic.library",
      .path = broad_metallib,
  });
  auto impl = std::make_unique<DenseQmmBroadPlan::Impl>();
  impl->weight = std::move(weight);
  impl->mxfp4 = mxfp4;
  impl->uses_high_plane = affine.has_value() && affine->uses_high_plane;
  impl->info = {
      .format_id = impl->weight.format,
      .layout_id = impl->weight.layout,
      .domain = tokens <= 128 ? "interval" : "bulk",
      .output_features = impl->weight.row_count,
      .input_features = impl->weight.in_features,
      .tokens = tokens,
      .token_tile = 16,
      .token_tiles = (tokens + 15) / 16,
      .input_a16_bytes = checked_multiply(input_values, 2, "broad A16 input"),
      .input_a4_code_bytes = mxfp4 ? input_values / 2 : 0,
      .input_a4_scale_bytes = mxfp4 ? input_values / 32 : 0,
      .output_bytes = checked_multiply(output_values, 2, "broad output"),
      .native_tensorops_runtime_eligible =
          context.capabilities().profile.native_tensorops_eligible,
  };
  for (const auto &selected_plane : impl->weight.planes) {
    impl->info.weight_plane_bytes += selected_plane.bytes;
  }

  if (!mxfp4) {
    const auto logical_groups = impl->weight.in_features / affine->group_size;
    impl->info.workspace_bytes = checked_multiply(
        checked_multiply(tokens, logical_groups, "broad group sums"), 4,
        "broad group sum bytes");
    impl->group_sums = context.make_buffer(impl->info.workspace_bytes);
    const auto group_pipeline =
        "b2.qmm_broad.group_sum.g" + std::to_string(affine->group_size);
    static_cast<void>(context.warmup_pipeline({
        .key = group_pipeline,
        .library_key = "b2.qmm_broad.classic.library",
        .function_name =
            "omlx_m5_broad_group_sum_g" + std::to_string(affine->group_size),
        .variant = "all_t",
    }));
    const auto function = "omlx_m5_oq_q" + affine->bits + "_g" +
                          std::to_string(affine->group_size) + "_broad_t16";
    const auto pipeline_key = "b2.qmm_broad." + impl->weight.format;
    static_cast<void>(context.warmup_pipeline({
        .key = pipeline_key,
        .library_key = "b2.qmm_broad.classic.library",
        .function_name = function,
        .variant = "t16",
    }));
    const auto code_kind = affine->bits == "8" ? "code" : "base_code";
    const auto &codes = plane(impl->weight, code_kind);
    const MaterializedPlaneView *high = &codes;
    if (affine->bits == "5") {
      high = &plane(impl->weight, "high_bit");
    } else if (affine->bits == "6") {
      high = &plane(impl->weight, "high_2bit");
    }
    const auto &scales = plane(impl->weight, "scale");
    const auto &biases = plane(impl->weight, "bias");
    std::vector<MetalResourceArgument> resources = {
        {.argument_index = 0,
         .resource_slot = 0,
         .minimum_bytes = impl->info.input_a16_bytes},
        {.argument_index = 1, .resource_slot = 3, .minimum_bytes = codes.bytes},
        {.argument_index = 3,
         .resource_slot = 5,
         .minimum_bytes = scales.bytes},
        {.argument_index = 4,
         .resource_slot = 6,
         .minimum_bytes = biases.bytes},
        {.argument_index = 5,
         .resource_slot = 7,
         .minimum_bytes = impl->info.workspace_bytes},
        {.argument_index = 6,
         .resource_slot = 8,
         .minimum_bytes = impl->info.output_bytes},
    };
    if (impl->uses_high_plane) {
      resources.insert(resources.begin() + 2, {.argument_index = 2,
                                               .resource_slot = 4,
                                               .minimum_bytes = high->bytes});
    }
    const auto executor_key = "dense_qmm_broad.oq." + impl->weight.format +
                              ".n" + std::to_string(impl->weight.row_count) +
                              ".k" + std::to_string(impl->weight.in_features) +
                              ".t" + std::to_string(tokens);
    auto command_plan = context.prepare_command_plan({
        .key = executor_key,
        .dispatches =
            {{
                 .pipeline_key = group_pipeline,
                 .grid = {.width = tokens * logical_groups * 32},
                 .threads_per_threadgroup = {.width = 32},
                 .resources =
                     {
                         {.argument_index = 0,
                          .resource_slot = 0,
                          .minimum_bytes = impl->info.input_a16_bytes},
                         {.argument_index = 1,
                          .resource_slot = 7,
                          .minimum_bytes = impl->info.workspace_bytes},
                     },
                 .constant_arguments = {{.argument_index = 2,
                                         .bytes = constant_bytes(params)}},
             },
             {
                 .pipeline_key = pipeline_key,
                 .grid = {.width = ((impl->weight.row_count + 3) / 4) * 128,
                          .height = impl->info.token_tiles},
                 .threads_per_threadgroup = {.width = 128},
                 .resources = std::move(resources),
                 .constant_arguments = {{.argument_index = 7,
                                         .bytes = constant_bytes(params)}},
             }},
    });
    impl->info.executors.push_back({
        .family = DenseQmmBroadFamily::oq_cooperative,
        .executor_key = executor_key,
        .function_name = function,
        .activation = ActivationProfile::a16,
        .required_route = MetalExecutionRoute::classic_msl,
        .dispatch_count = 2,
        .installed = true,
    });
    impl->executors.push_back({.command_plan = std::move(command_plan)});
  } else {
    const auto &data = plane(impl->weight, "data");
    const auto &scales = plane(impl->weight, "scale");
    const auto add_classic = [&](DenseQmmBroadFamily family) {
      const auto activation = activation_for(family);
      const auto function = activation == ActivationProfile::a16
                                ? "omlx_m5_mxfp4_w4a16_broad_t16"
                                : "omlx_m5_mxfp4_w4a4_broad_t16";
      const auto pipeline_key =
          "b2.qmm_broad." + std::string(to_string(family));
      static_cast<void>(context.warmup_pipeline({
          .key = pipeline_key,
          .library_key = "b2.qmm_broad.classic.library",
          .function_name = function,
          .variant = "t16",
      }));
      const auto executor_key = "dense_qmm_broad." +
                                std::string(to_string(family)) + ".n" +
                                std::to_string(impl->weight.row_count) + ".k" +
                                std::to_string(impl->weight.in_features) +
                                ".t" + std::to_string(tokens);
      MetalDispatchDescriptor dispatch{
          .pipeline_key = pipeline_key,
          .grid = {.width = ((impl->weight.row_count + 3) / 4) * 128,
                   .height = impl->info.token_tiles},
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
             .resource_slot = 5,
             .minimum_bytes = scales.bytes},
            {.argument_index = 3,
             .resource_slot = 8,
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
             .resource_slot = 5,
             .minimum_bytes = scales.bytes},
            {.argument_index = 4,
             .resource_slot = 8,
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
          .dispatch_count = 1,
          .installed = true,
      });
      impl->executors.push_back({.command_plan = std::move(command_plan)});
    };
    add_classic(DenseQmmBroadFamily::mxfp4_w4a16_simd);
    add_classic(DenseQmmBroadFamily::mxfp4_w4a4_simd);

    const auto add_tensorops = [&](DenseQmmBroadFamily family) {
      const auto activation = activation_for(family);
      const auto function = activation == ActivationProfile::a16
                                ? "omlx_m5_mxfp4_w4a16_tensorops"
                                : "omlx_m5_mxfp4_w4a4_tensorops";
      const auto executor_key = "dense_qmm_broad." +
                                std::string(to_string(family)) + ".n" +
                                std::to_string(impl->weight.row_count) + ".k" +
                                std::to_string(impl->weight.in_features) +
                                ".t" + std::to_string(tokens);
      DenseQmmBroadExecutorInfo executor_info{
          .family = family,
          .executor_key = executor_key,
          .function_name = function,
          .activation = activation,
          .required_route = MetalExecutionRoute::native_tensorops,
          .dispatch_count = 1,
          .installed = false,
      };
      MetalCommandPlan command_plan;
      if (impl->info.native_tensorops_runtime_eligible) {
        if (tensorops_metallib.empty()) {
          throw std::invalid_argument(
              "broad TensorOps runtime is eligible but metallib is missing");
        }
        context.register_library({
            .key = "b2.qmm_broad.tensorops.library",
            .path = tensorops_metallib,
        });
        const auto pipeline_key =
            "b2.qmm_broad." + std::string(to_string(family));
        static_cast<void>(context.warmup_pipeline({
            .key = pipeline_key,
            .library_key = "b2.qmm_broad.tensorops.library",
            .function_name = function,
            .variant = "m16n32",
        }));
        MetalDispatchDescriptor dispatch{
            .pipeline_key = pipeline_key,
            .grid = {.width = ((impl->weight.row_count + 31) / 32) * 32,
                     .height = impl->info.token_tiles},
            .threads_per_threadgroup = {.width = 32},
        };
        if (activation == ActivationProfile::a16) {
          const auto k = checked_i32(impl->weight.padded_k, "broad K");
          const auto n = checked_i32(impl->weight.row_count, "broad N");
          const auto m = static_cast<std::int32_t>(tokens);
          const auto logical_k =
              checked_i32(impl->weight.in_features, "broad logical K");
          dispatch.resources = {
              {.argument_index = 0,
               .resource_slot = 3,
               .minimum_bytes = data.bytes},
              {.argument_index = 1,
               .resource_slot = 5,
               .minimum_bytes = scales.bytes},
              {.argument_index = 2,
               .resource_slot = 0,
               .minimum_bytes = impl->info.input_a16_bytes},
              {.argument_index = 3,
               .resource_slot = 8,
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
               .resource_slot = 5,
               .minimum_bytes = scales.bytes},
              {.argument_index = 2,
               .resource_slot = 1,
               .minimum_bytes = impl->info.input_a4_code_bytes},
              {.argument_index = 3,
               .resource_slot = 2,
               .minimum_bytes = impl->info.input_a4_scale_bytes},
              {.argument_index = 4,
               .resource_slot = 8,
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
    add_tensorops(DenseQmmBroadFamily::mxfp4_w4a16_tensorops);
    add_tensorops(DenseQmmBroadFamily::mxfp4_w4a4_tensorops);
  }
  return DenseQmmBroadPlan(std::move(impl));
}

MetalSubmission execute_dense_qmm_broad_a16(const MetalContext &context,
                                            const DenseQmmBroadPlan &plan,
                                            std::string_view executor_key,
                                            const MetalBuffer &input,
                                            std::uint64_t input_offset,
                                            MetalBuffer &output,
                                            std::uint64_t output_offset) {
  if (plan.impl_ == nullptr || input_offset % 2 != 0 ||
      output_offset % 2 != 0) {
    throw std::invalid_argument("broad A16 invocation is invalid");
  }
  const auto &executor_info = plan.executor(executor_key);
  if (!executor_info.installed ||
      executor_info.activation != ActivationProfile::a16) {
    throw std::invalid_argument("broad A16 executor is unavailable");
  }
  const auto index = static_cast<std::size_t>(
      &executor_info - plan.impl_->info.executors.data());
  std::vector<MetalBufferBinding> bindings = {
      {0, &input, input_offset},
      {8, &output, output_offset},
  };
  if (plan.impl_->mxfp4) {
    const auto &data = plane(plan.impl_->weight, "data");
    const auto &scales = plane(plan.impl_->weight, "scale");
    bindings.push_back({3, data.buffer, data.offset});
    bindings.push_back({5, scales.buffer, scales.offset});
  } else {
    const auto format = parse_affine(plan.impl_->weight.format);
    const auto code_kind = format.bits == "8" ? "code" : "base_code";
    const auto &codes = plane(plan.impl_->weight, code_kind);
    bindings.push_back({3, codes.buffer, codes.offset});
    if (format.bits == "5") {
      const auto &high = plane(plan.impl_->weight, "high_bit");
      bindings.push_back({4, high.buffer, high.offset});
    } else if (format.bits == "6") {
      const auto &high = plane(plan.impl_->weight, "high_2bit");
      bindings.push_back({4, high.buffer, high.offset});
    }
    const auto &scales = plane(plan.impl_->weight, "scale");
    const auto &biases = plane(plan.impl_->weight, "bias");
    bindings.push_back({5, scales.buffer, scales.offset});
    bindings.push_back({6, biases.buffer, biases.offset});
    bindings.push_back({7, &plan.impl_->group_sums, 0});
  }
  return context.execute(plan.impl_->executors[index].command_plan, bindings);
}

MetalSubmission execute_dense_qmm_broad_a4(const MetalContext &context,
                                           const DenseQmmBroadPlan &plan,
                                           std::string_view executor_key,
                                           const Mxfp4ActivationView &input,
                                           MetalBuffer &output,
                                           std::uint64_t output_offset) {
  if (plan.impl_ == nullptr || !plan.impl_->mxfp4 || output_offset % 2 != 0) {
    throw std::invalid_argument("broad A4 invocation is invalid");
  }
  validate_a4_view(input, plan.impl_->info.input_a4_code_bytes,
                   plan.impl_->info.input_a4_scale_bytes);
  const auto &executor_info = plan.executor(executor_key);
  if (!executor_info.installed ||
      executor_info.activation != ActivationProfile::a4) {
    throw std::invalid_argument("broad A4 executor is unavailable");
  }
  const auto index = static_cast<std::size_t>(
      &executor_info - plan.impl_->info.executors.data());
  const auto &data = plane(plan.impl_->weight, "data");
  const auto &scales = plane(plan.impl_->weight, "scale");
  const std::vector<MetalBufferBinding> bindings = {
      {1, input.codes.buffer, input.codes.offset},
      {2, input.scales.buffer, input.scales.offset},
      {3, data.buffer, data.offset},
      {5, scales.buffer, scales.offset},
      {8, &output, output_offset},
  };
  return context.execute(plan.impl_->executors[index].command_plan, bindings);
}

} // namespace omlx::m5
