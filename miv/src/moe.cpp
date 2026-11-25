#include "omlx_m5/moe.h"

#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace omlx::m5 {
namespace {

struct MoeParams {
  std::uint32_t experts = 0;
  std::uint32_t top_k = 0;
  std::uint32_t output_features = 0;
  std::uint32_t input_features = 0;
  std::uint32_t padded_input_features = 0;
  std::uint32_t tokens = 0;
  std::uint32_t apply_scores = 0;
  std::uint32_t assignment_input = 0;
};

static_assert(sizeof(MoeParams) == 32);

struct AffineGeometry {
  std::string bits;
  std::uint32_t group_size = 0;
  bool uses_high_plane = false;
};

bool token_domain(std::uint32_t tokens) {
  return tokens >= 1 && (tokens <= 128 || tokens == 512 || tokens == 1024);
}

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
    throw std::invalid_argument("MoE plane is missing: " + std::string(kind));
  }
  return *found;
}

AffineGeometry parse_affine(std::string_view format) {
  constexpr std::string_view prefix = "oq_affine_q";
  constexpr std::string_view middle = "_g";
  constexpr std::string_view suffix = "_bf16_v1";
  if (!format.starts_with(prefix) || !format.ends_with(suffix)) {
    throw std::invalid_argument("MoE affine format is unsupported");
  }
  const auto group_position = format.find(middle, prefix.size());
  if (group_position == std::string_view::npos) {
    throw std::invalid_argument("MoE affine format is malformed");
  }
  const auto bits =
      format.substr(prefix.size(), group_position - prefix.size());
  const auto group = format.substr(group_position + middle.size(),
                                   format.size() - suffix.size() -
                                       group_position - middle.size());
  if ((bits != "4" && bits != "5" && bits != "6" && bits != "8") ||
      (group != "64" && group != "128")) {
    throw std::invalid_argument("MoE affine geometry is unsupported");
  }
  return {
      .bits = std::string(bits),
      .group_size = group == "64" ? 64U : 128U,
      .uses_high_plane = bits == "5" || bits == "6",
  };
}

bool is_mxfp4(std::string_view format) {
  return format == "mxfp4_e2m1_g32_e8m0_v1";
}

ActivationProfile activation_for(MoeExpertFamily family) {
  switch (family) {
  case MoeExpertFamily::grouped_a16:
  case MoeExpertFamily::fallback_a16:
    return ActivationProfile::a16;
  case MoeExpertFamily::grouped_a4:
  case MoeExpertFamily::fallback_a4:
    return ActivationProfile::a4;
  }
  throw std::invalid_argument("unknown MoE expert family");
}

bool grouped_family(MoeExpertFamily family) {
  return family == MoeExpertFamily::grouped_a16 ||
         family == MoeExpertFamily::grouped_a4;
}

void validate_routes(const MoeRouteView &routes, std::uint32_t tokens,
                     std::uint64_t index_bytes, std::uint64_t score_bytes) {
  if (routes.tokens != tokens || routes.experts != 256 || routes.top_k != 8 ||
      routes.indices.buffer == nullptr || routes.scores.buffer == nullptr ||
      routes.indices.bytes < index_bytes || routes.scores.bytes < score_bytes) {
    throw std::invalid_argument("MoE route view is invalid");
  }
}

void validate_a4(const Mxfp4ActivationView &view, std::uint64_t code_bytes,
                 std::uint64_t scale_bytes) {
  if (view.group_size != 32 || view.codes.buffer == nullptr ||
      view.scales.buffer == nullptr || view.codes.bytes < code_bytes ||
      view.scales.bytes < scale_bytes) {
    throw std::invalid_argument("MoE A4 activation view is invalid");
  }
}

} // namespace

class MoeRouterTop8Plan::Impl {
public:
  MoeRouterTop8Info info;
  MetalCommandPlan command_plan;
};

MoeRouterTop8Plan::MoeRouterTop8Plan() = default;
MoeRouterTop8Plan::MoeRouterTop8Plan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
MoeRouterTop8Plan::MoeRouterTop8Plan(MoeRouterTop8Plan &&) noexcept = default;
MoeRouterTop8Plan &
MoeRouterTop8Plan::operator=(MoeRouterTop8Plan &&) noexcept = default;
MoeRouterTop8Plan::~MoeRouterTop8Plan() = default;

bool MoeRouterTop8Plan::valid() const { return impl_ != nullptr; }

const MoeRouterTop8Info &MoeRouterTop8Plan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("MoE router plan is empty");
  }
  return impl_->info;
}

std::vector<KernelCandidate> MoeRouterTop8Plan::dispatch_candidates() const {
  if (impl_ == nullptr) {
    throw std::logic_error("MoE router plan is empty");
  }
  KernelCandidate candidate{
      .id = impl_->info.executor_key + ".full",
      .executor_key = impl_->info.executor_key,
      .priority = 10,
      .family = KernelFamily::grouped_moe,
      .format_id = "direct_bf16_v1",
      .layout_id = "contiguous_le_v1",
      .activation = ActivationProfile::a16,
      .input_type = KernelDataType::bf16,
      .output_type = KernelDataType::bf16,
      .accumulator_type = KernelDataType::fp32,
      .geometry = KernelGeometry::moe_router,
      .output_features = impl_->info.experts,
      .input_features = impl_->info.experts,
      .min_tokens = impl_->info.tokens,
      .max_tokens = impl_->info.tokens,
      .experts = impl_->info.experts,
      .top_k = impl_->info.top_k,
      .fused_op = KernelFusedOp::linear,
      .alignment = KernelAlignmentClass::full,
      .required_route = MetalExecutionRoute::classic_msl,
      .tail_safe = false,
  };
  validate_kernel_candidate(candidate);
  return {std::move(candidate)};
}

MoeRouterTop8Plan prepare_moe_router_top8(const MetalContext &context,
                                          const std::filesystem::path &metallib,
                                          std::uint32_t tokens) {
  if (!token_domain(tokens)) {
    throw std::invalid_argument("MoE router token domain is unsupported");
  }
  constexpr std::uint32_t experts = 256;
  constexpr std::uint32_t top_k = 8;
  const MoeParams params{
      .experts = experts,
      .top_k = top_k,
      .tokens = tokens,
  };
  context.register_library({.key = "b2.moe.library", .path = metallib});
  static_cast<void>(context.warmup_pipeline({
      .key = "b2.moe.router.top8",
      .library_key = "b2.moe.library",
      .function_name = "omlx_m5_moe_router_top8",
      .variant = "e256k8",
  }));
  auto impl = std::make_unique<MoeRouterTop8Plan::Impl>();
  impl->info = {
      .executor_key = "moe_router_top8.t" + std::to_string(tokens),
      .tokens = tokens,
      .experts = experts,
      .top_k = top_k,
      .probability_bytes =
          checked_multiply(tokens, experts * 2, "router input"),
      .index_bytes = checked_multiply(tokens, top_k * 4, "router indices"),
      .score_bytes = checked_multiply(tokens, top_k * 2, "router scores"),
      .dispatch_count = 1,
  };
  impl->command_plan = context.prepare_command_plan({
      .key = impl->info.executor_key,
      .dispatches = {{
          .pipeline_key = "b2.moe.router.top8",
          .grid = {.width = 32, .height = tokens},
          .threads_per_threadgroup = {.width = 32},
          .resources =
              {
                  {.argument_index = 0,
                   .resource_slot = 0,
                   .minimum_bytes = impl->info.probability_bytes},
                  {.argument_index = 1,
                   .resource_slot = 1,
                   .minimum_bytes = impl->info.index_bytes},
                  {.argument_index = 2,
                   .resource_slot = 2,
                   .minimum_bytes = impl->info.score_bytes},
              },
          .constant_arguments = {{.argument_index = 3,
                                  .bytes = constant_bytes(params)}},
      }},
  });
  return MoeRouterTop8Plan(std::move(impl));
}

MetalSubmission execute_moe_router_top8(const MetalContext &context,
                                        const MoeRouterTop8Plan &plan,
                                        const MetalBuffer &probabilities,
                                        std::uint64_t probability_offset,
                                        const MoeRouteView &output) {
  if (plan.impl_ == nullptr || probability_offset % 2 != 0) {
    throw std::invalid_argument("MoE router invocation is invalid");
  }
  validate_routes(output, plan.impl_->info.tokens, plan.impl_->info.index_bytes,
                  plan.impl_->info.score_bytes);
  const std::vector<MetalBufferBinding> bindings = {
      {0, &probabilities, probability_offset},
      {1, output.indices.buffer, output.indices.offset},
      {2, output.scores.buffer, output.scores.offset},
  };
  return context.execute(plan.impl_->command_plan, bindings);
}

std::string_view to_string(MoeExpertFamily family) {
  switch (family) {
  case MoeExpertFamily::grouped_a16:
    return "grouped_a16";
  case MoeExpertFamily::fallback_a16:
    return "fallback_a16";
  case MoeExpertFamily::grouped_a4:
    return "grouped_a4";
  case MoeExpertFamily::fallback_a4:
    return "fallback_a4";
  }
  return "unknown";
}

class MoeExpertPlan::Impl {
public:
  struct ExecutorState {
    MetalCommandPlan command_plan;
  };

  MaterializedWeightView weight;
  MoeExpertInfo info;
  MetalBuffer group_sums;
  bool mxfp4 = false;
  bool uses_high_plane = false;
  std::vector<ExecutorState> executors;
};

MoeExpertPlan::MoeExpertPlan() = default;
MoeExpertPlan::MoeExpertPlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
MoeExpertPlan::MoeExpertPlan(MoeExpertPlan &&) noexcept = default;
MoeExpertPlan &MoeExpertPlan::operator=(MoeExpertPlan &&) noexcept = default;
MoeExpertPlan::~MoeExpertPlan() = default;

bool MoeExpertPlan::valid() const { return impl_ != nullptr; }

const MoeExpertInfo &MoeExpertPlan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("MoE expert plan is empty");
  }
  return impl_->info;
}

std::vector<std::string> MoeExpertPlan::installed_executor_keys() const {
  if (impl_ == nullptr) {
    throw std::logic_error("MoE expert plan is empty");
  }
  std::vector<std::string> result;
  for (const auto &executor : impl_->info.executors) {
    result.push_back(executor.executor_key);
  }
  return result;
}

const MoeExpertExecutorInfo &
MoeExpertPlan::executor(std::string_view executor_key) const {
  if (impl_ == nullptr) {
    throw std::logic_error("MoE expert plan is empty");
  }
  const auto found = std::ranges::find_if(
      impl_->info.executors, [&](const MoeExpertExecutorInfo &value) {
        return value.executor_key == executor_key;
      });
  if (found == impl_->info.executors.end()) {
    throw std::out_of_range("MoE expert executor is unknown");
  }
  return *found;
}

std::vector<KernelCandidate> MoeExpertPlan::dispatch_candidates() const {
  if (impl_ == nullptr) {
    throw std::logic_error("MoE expert plan is empty");
  }
  std::vector<KernelCandidate> result;
  for (const auto &executor_info : impl_->info.executors) {
    const bool fallback = !grouped_family(executor_info.family);
    KernelCandidate candidate{
        .id = executor_info.executor_key + (fallback ? ".tail" : ".full"),
        .executor_key = executor_info.executor_key,
        .priority = fallback ? 100U : 10U,
        .family = executor_info.kernel_family,
        .format_id = impl_->info.format_id,
        .layout_id = impl_->info.layout_id,
        .activation = executor_info.activation,
        .input_type = KernelDataType::bf16,
        .output_type = KernelDataType::bf16,
        .accumulator_type = KernelDataType::fp32,
        .geometry = KernelGeometry::moe_expert,
        .output_features = impl_->info.output_features,
        .input_features = impl_->info.input_features,
        .min_tokens = impl_->info.tokens,
        .max_tokens = impl_->info.tokens,
        .experts = impl_->info.experts,
        .top_k = impl_->info.top_k,
        .fused_op = impl_->info.fused_op,
        .alignment = executor_info.alignment,
        .required_route = MetalExecutionRoute::classic_msl,
        .tail_safe = fallback,
    };
    validate_kernel_candidate(candidate);
    result.push_back(std::move(candidate));
  }
  return result;
}

MoeExpertPlan prepare_moe_expert(const MetalContext &context,
                                 const std::filesystem::path &metallib,
                                 MaterializedWeightView weight,
                                 std::uint32_t tokens, KernelFusedOp fused_op,
                                 bool apply_route_scores) {
  constexpr std::uint32_t experts = 256;
  constexpr std::uint32_t top_k = 8;
  if (!token_domain(tokens) || weight.parent_shape.size() != 3 ||
      weight.parent_shape[0] != experts || weight.row_start != 0 ||
      weight.row_count != weight.parent_shape[0] * weight.parent_shape[1] ||
      weight.in_features != weight.parent_shape[2] ||
      weight.padded_k < weight.in_features || weight.padded_k % 128 != 0 ||
      (fused_op != KernelFusedOp::moe_gate_up &&
       fused_op != KernelFusedOp::moe_down) ||
      (apply_route_scores && fused_op != KernelFusedOp::moe_down)) {
    throw std::invalid_argument("MoE expert plan domain is invalid");
  }
  const bool mxfp4 = is_mxfp4(weight.format);
  std::optional<AffineGeometry> affine;
  if (mxfp4) {
    if (weight.layout != "mx_expert_rowsplit_k128_v1" ||
        weight.in_features % 32 != 0) {
      throw std::invalid_argument("MoE MXFP4 geometry is invalid");
    }
  } else {
    affine = parse_affine(weight.format);
    if (weight.layout != "oq_expert_rowsplit_k128_v1" ||
        weight.in_features % affine->group_size != 0) {
      throw std::invalid_argument("MoE affine geometry is invalid");
    }
  }
  const auto assignments = checked_multiply(tokens, top_k, "MoE assignments");
  const bool assignment_input = fused_op == KernelFusedOp::moe_down;
  const auto input_rows = assignment_input ? assignments : tokens;
  const auto input_values =
      checked_multiply(input_rows, weight.in_features, "MoE input values");
  const auto output_values = checked_multiply(
      assignments, weight.parent_shape[1], "MoE output values");
  const MoeParams params{
      .experts = experts,
      .top_k = top_k,
      .output_features = checked_u32(weight.parent_shape[1], "MoE N"),
      .input_features = checked_u32(weight.in_features, "MoE K"),
      .padded_input_features = checked_u32(weight.padded_k, "MoE Kp"),
      .tokens = tokens,
      .apply_scores = apply_route_scores ? 1U : 0U,
      .assignment_input = assignment_input ? 1U : 0U,
  };
  context.register_library({.key = "b2.moe.library", .path = metallib});
  auto impl = std::make_unique<MoeExpertPlan::Impl>();
  impl->weight = std::move(weight);
  impl->mxfp4 = mxfp4;
  impl->uses_high_plane = affine.has_value() && affine->uses_high_plane;
  impl->info = {
      .format_id = impl->weight.format,
      .layout_id = impl->weight.layout,
      .fused_op = fused_op,
      .experts = experts,
      .top_k = top_k,
      .tokens = tokens,
      .assignments = assignments,
      .input_rows = input_rows,
      .output_features = impl->weight.parent_shape[1],
      .input_features = impl->weight.in_features,
      .input_a16_bytes = checked_multiply(input_values, 2, "MoE A16 input"),
      .input_a4_code_bytes = mxfp4 ? input_values / 2 : 0,
      .input_a4_scale_bytes = mxfp4 ? input_values / 32 : 0,
      .route_index_bytes = checked_multiply(assignments, 4, "MoE route index"),
      .route_score_bytes = checked_multiply(assignments, 2, "MoE route score"),
      .output_bytes = checked_multiply(output_values, 2, "MoE output"),
      .apply_route_scores = apply_route_scores,
      .assignment_input = assignment_input,
  };
  for (const auto &selected_plane : impl->weight.planes) {
    impl->info.weight_plane_bytes += selected_plane.bytes;
  }
  if (!mxfp4) {
    const auto groups = impl->weight.in_features / affine->group_size;
    impl->info.workspace_bytes =
        checked_multiply(checked_multiply(input_rows, groups, "MoE group sums"),
                         4, "MoE group sum bytes");
    impl->group_sums = context.make_buffer(impl->info.workspace_bytes);
    const auto group_pipeline =
        "b2.moe.group_sum.g" + std::to_string(affine->group_size);
    static_cast<void>(context.warmup_pipeline({
        .key = group_pipeline,
        .library_key = "b2.moe.library",
        .function_name =
            "omlx_m5_moe_group_sum_g" + std::to_string(affine->group_size),
        .variant = "e256k8",
    }));
  }

  const auto add_executor = [&](MoeExpertFamily family) {
    const bool grouped = grouped_family(family);
    const auto activation = activation_for(family);
    std::string function;
    if (mxfp4) {
      function = "omlx_m5_mxfp4_moe_" +
                 std::string(activation == ActivationProfile::a16 ? "w4a16_"
                                                                  : "w4a4_") +
                 (grouped ? "grouped" : "fallback");
    } else {
      function = "omlx_m5_oq_q" + affine->bits + "_g" +
                 std::to_string(affine->group_size) + "_moe_" +
                 (grouped ? "grouped" : "fallback");
    }
    const auto pipeline_key =
        "b2.moe." + impl->weight.format + "." + std::string(to_string(family));
    static_cast<void>(context.warmup_pipeline({
        .key = pipeline_key,
        .library_key = "b2.moe.library",
        .function_name = function,
        .variant = grouped ? "grouped4" : "fallback1",
    }));
    const auto executor_key =
        "moe_expert." + std::string(to_string(family)) + "." +
        impl->weight.format + ".n" +
        std::to_string(impl->weight.parent_shape[1]) + ".k" +
        std::to_string(impl->weight.in_features) + ".t" +
        std::to_string(tokens) +
        (apply_route_scores ? ".weighted" : ".unweighted");
    MetalDispatchDescriptor expert_dispatch{
        .pipeline_key = pipeline_key,
        .grid = {.width = grouped ? ((output_values + 3) / 4) * 128
                                  : output_values * 32},
        .threads_per_threadgroup = {.width = grouped ? 128U : 32U},
    };
    if (!mxfp4) {
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
      expert_dispatch.resources = {
          {.argument_index = 0,
           .resource_slot = 0,
           .minimum_bytes = impl->info.input_a16_bytes},
          {.argument_index = 1,
           .resource_slot = 1,
           .minimum_bytes = impl->info.route_index_bytes},
          {.argument_index = 2,
           .resource_slot = 2,
           .minimum_bytes = impl->info.route_score_bytes},
          {.argument_index = 3,
           .resource_slot = 3,
           .minimum_bytes = codes.bytes},
          {.argument_index = 5,
           .resource_slot = 5,
           .minimum_bytes = scales.bytes},
          {.argument_index = 6,
           .resource_slot = 6,
           .minimum_bytes = biases.bytes},
          {.argument_index = 7,
           .resource_slot = 7,
           .minimum_bytes = impl->info.workspace_bytes},
          {.argument_index = 8,
           .resource_slot = 8,
           .minimum_bytes = impl->info.output_bytes},
      };
      if (impl->uses_high_plane) {
        expert_dispatch.resources.insert(expert_dispatch.resources.begin() + 4,
                                         {.argument_index = 4,
                                          .resource_slot = 4,
                                          .minimum_bytes = high->bytes});
      }
      expert_dispatch.constant_arguments = {
          {.argument_index = 9, .bytes = constant_bytes(params)}};
    } else {
      const auto &data = plane(impl->weight, "data");
      const auto &scales = plane(impl->weight, "scale");
      if (activation == ActivationProfile::a16) {
        expert_dispatch.resources = {
            {.argument_index = 0,
             .resource_slot = 0,
             .minimum_bytes = impl->info.input_a16_bytes},
            {.argument_index = 1,
             .resource_slot = 1,
             .minimum_bytes = impl->info.route_index_bytes},
            {.argument_index = 2,
             .resource_slot = 2,
             .minimum_bytes = impl->info.route_score_bytes},
            {.argument_index = 3,
             .resource_slot = 3,
             .minimum_bytes = data.bytes},
            {.argument_index = 4,
             .resource_slot = 5,
             .minimum_bytes = scales.bytes},
            {.argument_index = 5,
             .resource_slot = 8,
             .minimum_bytes = impl->info.output_bytes},
        };
        expert_dispatch.constant_arguments = {
            {.argument_index = 6, .bytes = constant_bytes(params)}};
      } else {
        expert_dispatch.resources = {
            {.argument_index = 0,
             .resource_slot = 9,
             .minimum_bytes = impl->info.input_a4_code_bytes},
            {.argument_index = 1,
             .resource_slot = 10,
             .minimum_bytes = impl->info.input_a4_scale_bytes},
            {.argument_index = 2,
             .resource_slot = 1,
             .minimum_bytes = impl->info.route_index_bytes},
            {.argument_index = 3,
             .resource_slot = 2,
             .minimum_bytes = impl->info.route_score_bytes},
            {.argument_index = 4,
             .resource_slot = 3,
             .minimum_bytes = data.bytes},
            {.argument_index = 5,
             .resource_slot = 5,
             .minimum_bytes = scales.bytes},
            {.argument_index = 6,
             .resource_slot = 8,
             .minimum_bytes = impl->info.output_bytes},
        };
        expert_dispatch.constant_arguments = {
            {.argument_index = 7, .bytes = constant_bytes(params)}};
      }
    }
    std::vector<MetalDispatchDescriptor> dispatches;
    if (!mxfp4) {
      const auto groups = impl->weight.in_features / affine->group_size;
      dispatches.push_back({
          .pipeline_key =
              "b2.moe.group_sum.g" + std::to_string(affine->group_size),
          .grid = {.width = input_rows * groups * 32},
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
      });
    }
    dispatches.push_back(std::move(expert_dispatch));
    auto command_plan = context.prepare_command_plan({
        .key = executor_key,
        .dispatches = std::move(dispatches),
    });
    impl->info.executors.push_back({
        .family = family,
        .executor_key = executor_key,
        .function_name = function,
        .activation = activation,
        .kernel_family = grouped ? KernelFamily::grouped_moe
                                 : KernelFamily::per_expert_fallback,
        .alignment =
            grouped ? KernelAlignmentClass::full : KernelAlignmentClass::tail,
        .dispatch_count = mxfp4 ? 1U : 2U,
    });
    impl->executors.push_back({.command_plan = std::move(command_plan)});
  };

  add_executor(MoeExpertFamily::grouped_a16);
  add_executor(MoeExpertFamily::fallback_a16);
  if (mxfp4) {
    add_executor(MoeExpertFamily::grouped_a4);
    add_executor(MoeExpertFamily::fallback_a4);
  }
  return MoeExpertPlan(std::move(impl));
}

MetalSubmission
execute_moe_expert_a16(const MetalContext &context, const MoeExpertPlan &plan,
                       std::string_view executor_key, const MetalBuffer &input,
                       std::uint64_t input_offset, const MoeRouteView &routes,
                       MetalBuffer &output, std::uint64_t output_offset) {
  if (plan.impl_ == nullptr || input_offset % 2 != 0 ||
      output_offset % 2 != 0) {
    throw std::invalid_argument("MoE A16 invocation is invalid");
  }
  validate_routes(routes, plan.impl_->info.tokens,
                  plan.impl_->info.route_index_bytes,
                  plan.impl_->info.route_score_bytes);
  const auto &executor_info = plan.executor(executor_key);
  if (executor_info.activation != ActivationProfile::a16) {
    throw std::invalid_argument("MoE executor does not accept A16");
  }
  const auto index = static_cast<std::size_t>(
      &executor_info - plan.impl_->info.executors.data());
  std::vector<MetalBufferBinding> bindings = {
      {0, &input, input_offset},
      {1, routes.indices.buffer, routes.indices.offset},
      {2, routes.scores.buffer, routes.scores.offset},
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

MetalSubmission execute_moe_expert_a4(const MetalContext &context,
                                      const MoeExpertPlan &plan,
                                      std::string_view executor_key,
                                      const Mxfp4ActivationView &input,
                                      const MoeRouteView &routes,
                                      MetalBuffer &output,
                                      std::uint64_t output_offset) {
  if (plan.impl_ == nullptr || !plan.impl_->mxfp4 || output_offset % 2 != 0) {
    throw std::invalid_argument("MoE A4 invocation is invalid");
  }
  validate_routes(routes, plan.impl_->info.tokens,
                  plan.impl_->info.route_index_bytes,
                  plan.impl_->info.route_score_bytes);
  validate_a4(input, plan.impl_->info.input_a4_code_bytes,
              plan.impl_->info.input_a4_scale_bytes);
  const auto &executor_info = plan.executor(executor_key);
  if (executor_info.activation != ActivationProfile::a4) {
    throw std::invalid_argument("MoE executor does not accept A4");
  }
  const auto index = static_cast<std::size_t>(
      &executor_info - plan.impl_->info.executors.data());
  const auto &data = plane(plan.impl_->weight, "data");
  const auto &scales = plane(plan.impl_->weight, "scale");
  const std::vector<MetalBufferBinding> bindings = {
      {9, input.codes.buffer, input.codes.offset},
      {10, input.scales.buffer, input.scales.offset},
      {1, routes.indices.buffer, routes.indices.offset},
      {2, routes.scores.buffer, routes.scores.offset},
      {3, data.buffer, data.offset},
      {5, scales.buffer, scales.offset},
      {8, &output, output_offset},
  };
  return context.execute(plan.impl_->executors[index].command_plan, bindings);
}

} // namespace omlx::m5
