#include "omlx_m5/mlp.h"

#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace omlx::m5 {
namespace {

struct SwiGluParams {
  std::uint32_t rows = 0;
  std::uint32_t features = 0;
  std::uint32_t groups = 0;
  std::uint32_t reserved = 0;
};

struct MoeReduceParams {
  std::uint32_t tokens = 0;
  std::uint32_t top_k = 0;
  std::uint32_t features = 0;
  std::uint32_t reserved = 0;
};

static_assert(sizeof(SwiGluParams) == 16);
static_assert(sizeof(MoeReduceParams) == 16);

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

bool token_domain(std::uint32_t tokens) {
  return tokens >= 1 && (tokens <= 128 || tokens == 512 || tokens == 1024);
}

} // namespace

std::string_view to_string(SwiGluProducerMode mode) {
  switch (mode) {
  case SwiGluProducerMode::a16:
    return "a16";
  case SwiGluProducerMode::a4:
    return "a4";
  case SwiGluProducerMode::a16_a4:
    return "a16_a4";
  }
  return "unknown";
}

class SwiGluPlan::Impl {
public:
  SwiGluInfo info;
  MetalCommandPlan a16;
  MetalCommandPlan a4;
  MetalCommandPlan a16_a4;
};

SwiGluPlan::SwiGluPlan() = default;
SwiGluPlan::SwiGluPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SwiGluPlan::SwiGluPlan(SwiGluPlan &&) noexcept = default;
SwiGluPlan &SwiGluPlan::operator=(SwiGluPlan &&) noexcept = default;
SwiGluPlan::~SwiGluPlan() = default;

bool SwiGluPlan::valid() const { return impl_ != nullptr; }

const SwiGluInfo &SwiGluPlan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("SwiGLU plan is empty");
  }
  return impl_->info;
}

SwiGluPlan prepare_swiglu(const MetalContext &context,
                          const std::filesystem::path &metallib,
                          std::uint32_t rows, std::uint64_t features) {
  if (rows == 0 || features == 0 || features % 32 != 0) {
    throw std::invalid_argument("SwiGLU domain is invalid");
  }
  const auto values = checked_multiply(rows, features, "SwiGLU values");
  const SwiGluParams params{
      .rows = rows,
      .features = checked_u32(features, "SwiGLU features"),
      .groups = checked_u32(features / 32, "SwiGLU groups"),
  };
  context.register_library({.key = "b3.mlp.library", .path = metallib});
  const auto warm = [&](std::string key, std::string function) {
    static_cast<void>(context.warmup_pipeline({
        .key = std::move(key),
        .library_key = "b3.mlp.library",
        .function_name = std::move(function),
        .variant = "bf16_g32",
    }));
  };
  warm("b3.mlp.swiglu.a16", "omlx_m5_swiglu_a16");
  warm("b3.mlp.swiglu.a4", "omlx_m5_swiglu_a4");
  warm("b3.mlp.swiglu.a16_a4", "omlx_m5_swiglu_a16_a4");

  auto impl = std::make_unique<SwiGluPlan::Impl>();
  impl->info = {
      .executor_key =
          "swiglu.r" + std::to_string(rows) + ".n" + std::to_string(features),
      .rows = rows,
      .features = features,
      .input_bytes = checked_multiply(values, 2, "SwiGLU input"),
      .a16_bytes = checked_multiply(values, 2, "SwiGLU A16"),
      .a4_code_bytes = values / 2,
      .a4_scale_bytes = values / 32,
  };
  const auto command = [&](std::string suffix, std::string pipeline,
                           std::vector<MetalResourceArgument> resources,
                           std::uint32_t params_index) {
    return context.prepare_command_plan({
        .key = impl->info.executor_key + "." + suffix,
        .dispatches = {{
            .pipeline_key = std::move(pipeline),
            .grid = {.width = 32, .height = rows * params.groups},
            .threads_per_threadgroup = {.width = 32},
            .resources = std::move(resources),
            .constant_arguments = {{.argument_index = params_index,
                                    .bytes = constant_bytes(params)}},
        }},
    });
  };
  const std::vector<MetalResourceArgument> common = {
      {.argument_index = 0,
       .resource_slot = 0,
       .minimum_bytes = impl->info.input_bytes},
      {.argument_index = 1,
       .resource_slot = 1,
       .minimum_bytes = impl->info.input_bytes},
  };
  auto a16 = common;
  a16.push_back({.argument_index = 2,
                 .resource_slot = 2,
                 .minimum_bytes = impl->info.a16_bytes});
  impl->a16 = command("a16", "b3.mlp.swiglu.a16", std::move(a16), 3);
  auto a4 = common;
  a4.push_back({.argument_index = 2,
                .resource_slot = 3,
                .minimum_bytes = impl->info.a4_code_bytes});
  a4.push_back({.argument_index = 3,
                .resource_slot = 4,
                .minimum_bytes = impl->info.a4_scale_bytes});
  impl->a4 = command("a4", "b3.mlp.swiglu.a4", std::move(a4), 4);
  auto dual = common;
  dual.push_back({.argument_index = 2,
                  .resource_slot = 2,
                  .minimum_bytes = impl->info.a16_bytes});
  dual.push_back({.argument_index = 3,
                  .resource_slot = 3,
                  .minimum_bytes = impl->info.a4_code_bytes});
  dual.push_back({.argument_index = 4,
                  .resource_slot = 4,
                  .minimum_bytes = impl->info.a4_scale_bytes});
  impl->a16_a4 = command("a16_a4", "b3.mlp.swiglu.a16_a4", std::move(dual), 5);
  return SwiGluPlan(std::move(impl));
}

MetalSubmission
produce_swiglu_activation(const MetalContext &context, const SwiGluPlan &plan,
                          const ProjectionParentOutputView &gate,
                          const ProjectionParentOutputView &up,
                          const SwiGluActivationView &output) {
  if (plan.impl_ == nullptr || gate.buffer == nullptr || up.buffer == nullptr ||
      gate.tokens != plan.impl_->info.rows ||
      up.tokens != plan.impl_->info.rows ||
      gate.features != plan.impl_->info.features ||
      up.features != plan.impl_->info.features ||
      gate.token_stride_bytes != plan.impl_->info.features * 2 ||
      up.token_stride_bytes != plan.impl_->info.features * 2 ||
      gate.bytes < plan.impl_->info.input_bytes ||
      up.bytes < plan.impl_->info.input_bytes ||
      output.rows != plan.impl_->info.rows ||
      output.features != plan.impl_->info.features ||
      (!output.a16.has_value() && !output.a4.has_value())) {
    throw std::invalid_argument("SwiGLU invocation is invalid");
  }
  std::vector<MetalBufferBinding> bindings = {
      {0, gate.buffer, gate.offset},
      {1, up.buffer, up.offset},
  };
  const MetalCommandPlan *command = nullptr;
  if (output.a16.has_value()) {
    if (output.a16->buffer == nullptr ||
        output.a16->bytes < plan.impl_->info.a16_bytes) {
      throw std::invalid_argument("SwiGLU A16 output is invalid");
    }
    bindings.push_back({2, output.a16->buffer, output.a16->offset});
  }
  if (output.a4.has_value()) {
    if (output.a4->group_size != 32 || output.a4->codes.buffer == nullptr ||
        output.a4->scales.buffer == nullptr ||
        output.a4->codes.bytes < plan.impl_->info.a4_code_bytes ||
        output.a4->scales.bytes < plan.impl_->info.a4_scale_bytes) {
      throw std::invalid_argument("SwiGLU A4 output is invalid");
    }
    bindings.push_back({3, output.a4->codes.buffer, output.a4->codes.offset});
    bindings.push_back({4, output.a4->scales.buffer, output.a4->scales.offset});
  }
  if (output.a16.has_value() && output.a4.has_value()) {
    command = &plan.impl_->a16_a4;
  } else if (output.a16.has_value()) {
    command = &plan.impl_->a16;
  } else {
    command = &plan.impl_->a4;
  }
  return context.execute(*command, bindings);
}

class MoeTopKReducePlan::Impl {
public:
  MoeTopKReduceInfo info;
  MetalCommandPlan command_plan;
};

MoeTopKReducePlan::MoeTopKReducePlan() = default;
MoeTopKReducePlan::MoeTopKReducePlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
MoeTopKReducePlan::MoeTopKReducePlan(MoeTopKReducePlan &&) noexcept = default;
MoeTopKReducePlan &
MoeTopKReducePlan::operator=(MoeTopKReducePlan &&) noexcept = default;
MoeTopKReducePlan::~MoeTopKReducePlan() = default;

bool MoeTopKReducePlan::valid() const { return impl_ != nullptr; }

const MoeTopKReduceInfo &MoeTopKReducePlan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("MoE reduce plan is empty");
  }
  return impl_->info;
}

MoeTopKReducePlan prepare_moe_topk_reduce(const MetalContext &context,
                                          const std::filesystem::path &metallib,
                                          std::uint32_t tokens,
                                          std::uint64_t features,
                                          std::uint32_t top_k) {
  if (!token_domain(tokens) || features == 0 || top_k == 0 || top_k > 32) {
    throw std::invalid_argument("MoE reduce domain is invalid");
  }
  const auto output_values =
      checked_multiply(tokens, features, "MoE reduce output");
  const auto input_values =
      checked_multiply(output_values, top_k, "MoE reduce input");
  const MoeReduceParams params{
      .tokens = tokens,
      .top_k = top_k,
      .features = checked_u32(features, "MoE reduce features"),
  };
  context.register_library({.key = "b3.mlp.library", .path = metallib});
  static_cast<void>(context.warmup_pipeline({
      .key = "b3.mlp.moe_topk_reduce",
      .library_key = "b3.mlp.library",
      .function_name = "omlx_m5_moe_topk_reduce",
      .variant = "bf16_fp32",
  }));
  auto impl = std::make_unique<MoeTopKReducePlan::Impl>();
  impl->info = {
      .executor_key = "moe_topk_reduce.t" + std::to_string(tokens) + ".k" +
                      std::to_string(top_k) + ".n" + std::to_string(features),
      .tokens = tokens,
      .top_k = top_k,
      .features = features,
      .input_bytes =
          checked_multiply(input_values, 2, "MoE reduce input bytes"),
      .output_bytes =
          checked_multiply(output_values, 2, "MoE reduce output bytes"),
  };
  impl->command_plan = context.prepare_command_plan({
      .key = impl->info.executor_key,
      .dispatches = {{
          .pipeline_key = "b3.mlp.moe_topk_reduce",
          .grid = {.width = output_values},
          .threads_per_threadgroup = {.width = 128},
          .resources =
              {
                  {.argument_index = 0,
                   .resource_slot = 0,
                   .minimum_bytes = impl->info.input_bytes},
                  {.argument_index = 1,
                   .resource_slot = 1,
                   .minimum_bytes = impl->info.output_bytes},
              },
          .constant_arguments = {{.argument_index = 2,
                                  .bytes = constant_bytes(params)}},
      }},
  });
  return MoeTopKReducePlan(std::move(impl));
}

MetalSubmission execute_moe_topk_reduce(const MetalContext &context,
                                        const MoeTopKReducePlan &plan,
                                        const MetalBuffer &assignment_output,
                                        std::uint64_t input_offset,
                                        MetalBuffer &token_output,
                                        std::uint64_t output_offset) {
  if (plan.impl_ == nullptr || input_offset % 2 != 0 ||
      output_offset % 2 != 0) {
    throw std::invalid_argument("MoE reduce invocation is invalid");
  }
  const std::vector<MetalBufferBinding> bindings = {
      {0, &assignment_output, input_offset},
      {1, &token_output, output_offset},
  };
  return context.execute(plan.impl_->command_plan, bindings);
}

class MoeSharedCombinePlan::Impl {
public:
  MoeSharedCombineInfo info;
  MetalCommandPlan command_plan;
};

MoeSharedCombinePlan::MoeSharedCombinePlan() = default;
MoeSharedCombinePlan::MoeSharedCombinePlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
MoeSharedCombinePlan::MoeSharedCombinePlan(MoeSharedCombinePlan &&) noexcept =
    default;
MoeSharedCombinePlan &
MoeSharedCombinePlan::operator=(MoeSharedCombinePlan &&) noexcept = default;
MoeSharedCombinePlan::~MoeSharedCombinePlan() = default;

bool MoeSharedCombinePlan::valid() const { return impl_ != nullptr; }

const MoeSharedCombineInfo &MoeSharedCombinePlan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("MoE shared combine plan is empty");
  }
  return impl_->info;
}

MoeSharedCombinePlan
prepare_moe_shared_combine(const MetalContext &context,
                           const std::filesystem::path &metallib,
                           std::uint32_t tokens, std::uint64_t features) {
  if (!token_domain(tokens) || features == 0) {
    throw std::invalid_argument("MoE shared combine domain is invalid");
  }
  const auto values = checked_multiply(tokens, features, "MoE shared values");
  const MoeReduceParams params{
      .tokens = tokens,
      .features = checked_u32(features, "MoE shared features"),
  };
  context.register_library({.key = "b3.mlp.library", .path = metallib});
  static_cast<void>(context.warmup_pipeline({
      .key = "b3.mlp.moe_shared_combine",
      .library_key = "b3.mlp.library",
      .function_name = "omlx_m5_moe_shared_combine",
      .variant = "bf16_fp32",
  }));
  auto impl = std::make_unique<MoeSharedCombinePlan::Impl>();
  impl->info = {
      .executor_key = "moe_shared_combine.t" + std::to_string(tokens) + ".n" +
                      std::to_string(features),
      .tokens = tokens,
      .features = features,
      .tensor_bytes = checked_multiply(values, 2, "MoE shared tensor"),
      .gate_bytes = checked_multiply(tokens, 2, "MoE shared gate"),
  };
  impl->command_plan = context.prepare_command_plan({
      .key = impl->info.executor_key,
      .dispatches = {{
          .pipeline_key = "b3.mlp.moe_shared_combine",
          .grid = {.width = values},
          .threads_per_threadgroup = {.width = 128},
          .resources =
              {
                  {.argument_index = 0,
                   .resource_slot = 0,
                   .minimum_bytes = impl->info.tensor_bytes},
                  {.argument_index = 1,
                   .resource_slot = 1,
                   .minimum_bytes = impl->info.tensor_bytes},
                  {.argument_index = 2,
                   .resource_slot = 2,
                   .minimum_bytes = impl->info.gate_bytes},
                  {.argument_index = 3,
                   .resource_slot = 3,
                   .minimum_bytes = impl->info.tensor_bytes},
              },
          .constant_arguments = {{.argument_index = 4,
                                  .bytes = constant_bytes(params)}},
      }},
  });
  return MoeSharedCombinePlan(std::move(impl));
}

MetalSubmission execute_moe_shared_combine(
    const MetalContext &context, const MoeSharedCombinePlan &plan,
    const MetalBuffer &routed, std::uint64_t routed_offset,
    const MetalBuffer &shared, std::uint64_t shared_offset,
    const MetalBuffer &shared_gate, std::uint64_t gate_offset,
    MetalBuffer &output, std::uint64_t output_offset) {
  if (plan.impl_ == nullptr || routed_offset % 2 != 0 ||
      shared_offset % 2 != 0 || gate_offset % 2 != 0 ||
      output_offset % 2 != 0) {
    throw std::invalid_argument("MoE shared combine invocation is invalid");
  }
  const std::vector<MetalBufferBinding> bindings = {
      {0, &routed, routed_offset},
      {1, &shared, shared_offset},
      {2, &shared_gate, gate_offset},
      {3, &output, output_offset},
  };
  return context.execute(plan.impl_->command_plan, bindings);
}

} // namespace omlx::m5
