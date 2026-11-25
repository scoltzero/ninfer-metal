#include "omlx_m5/normalization.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace omlx::m5 {
namespace {

struct RmsNormProjectionParams {
  std::uint32_t input_features = 0;
  std::uint32_t tokens = 0;
  float epsilon = 0.0F;
  float weight_offset = 0.0F;
};

static_assert(sizeof(RmsNormProjectionParams) == 16);

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

std::vector<std::byte> constant_bytes(const RmsNormProjectionParams &params) {
  std::vector<std::byte> result(sizeof(params));
  std::memcpy(result.data(), &params, sizeof(params));
  return result;
}

bool token_domain(std::uint32_t tokens) {
  return tokens >= 1 && (tokens <= 128 || tokens == 512 || tokens == 1024);
}

} // namespace

std::string_view to_string(RmsNormProducerMode mode) {
  switch (mode) {
  case RmsNormProducerMode::none:
    return "none";
  case RmsNormProducerMode::a16:
    return "a16";
  case RmsNormProducerMode::a4:
    return "a4";
  case RmsNormProducerMode::a16_a4:
    return "a16_a4";
  }
  return "unknown";
}

class RmsNormProjectionPlan::Impl {
public:
  RmsNormProjectionInfo info;
  MetalCommandPlan a16;
  MetalCommandPlan a4;
  MetalCommandPlan a16_a4;
};

RmsNormProjectionPlan::RmsNormProjectionPlan() = default;
RmsNormProjectionPlan::RmsNormProjectionPlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
RmsNormProjectionPlan::RmsNormProjectionPlan(
    RmsNormProjectionPlan &&) noexcept = default;
RmsNormProjectionPlan &
RmsNormProjectionPlan::operator=(RmsNormProjectionPlan &&) noexcept = default;
RmsNormProjectionPlan::~RmsNormProjectionPlan() = default;

bool RmsNormProjectionPlan::valid() const { return impl_ != nullptr; }

const RmsNormProjectionInfo &RmsNormProjectionPlan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("RMSNorm projection plan is empty");
  }
  return impl_->info;
}

RmsNormProjectionPlan prepare_rmsnorm_projection(
    const MetalContext &context, const std::filesystem::path &metallib,
    std::uint64_t input_features, std::uint32_t tokens, float epsilon) {
  if (input_features == 0 || input_features % 32 != 0 ||
      !token_domain(tokens) || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    throw std::invalid_argument("RMSNorm projection domain is invalid");
  }
  const auto values =
      checked_multiply(input_features, tokens, "RMSNorm values");
  const RmsNormProjectionParams params{
      .input_features = checked_u32(input_features, "RMSNorm K"),
      .tokens = tokens,
      .epsilon = epsilon,
      .weight_offset = 0.0F,
  };
  context.register_library({
      .key = "b3.rmsnorm_projection.library",
      .path = metallib,
  });
  const auto warm = [&](std::string key, std::string function) {
    static_cast<void>(context.warmup_pipeline({
        .key = std::move(key),
        .library_key = "b3.rmsnorm_projection.library",
        .function_name = std::move(function),
        .variant = "bf16_g32",
    }));
  };
  warm("b3.rmsnorm_projection.a16", "omlx_m5_rmsnorm_a16");
  warm("b3.rmsnorm_projection.a4", "omlx_m5_rmsnorm_a4");
  warm("b3.rmsnorm_projection.a16_a4", "omlx_m5_rmsnorm_a16_a4");

  auto impl = std::make_unique<RmsNormProjectionPlan::Impl>();
  impl->info = {
      .executor_key = "rmsnorm_projection.k" + std::to_string(input_features) +
                      ".t" + std::to_string(tokens),
      .input_features = input_features,
      .tokens = tokens,
      .epsilon = epsilon,
      .input_bytes = checked_multiply(values, 2, "RMSNorm input"),
      .weight_bytes = checked_multiply(input_features, 2, "RMSNorm weight"),
      .a16_bytes = checked_multiply(values, 2, "RMSNorm A16"),
      .a4_code_bytes = values / 2,
      .a4_scale_bytes = values / 32,
  };
  const auto descriptor = [&](std::string suffix, std::string pipeline,
                              std::vector<MetalResourceArgument> resources,
                              std::uint32_t params_index) {
    return context.prepare_command_plan({
        .key = impl->info.executor_key + "." + suffix,
        .dispatches = {{
            .pipeline_key = std::move(pipeline),
            .grid = {.width = 128, .height = tokens},
            .threads_per_threadgroup = {.width = 128},
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
       .minimum_bytes = impl->info.weight_bytes},
  };
  auto a16_resources = common;
  a16_resources.push_back({.argument_index = 2,
                           .resource_slot = 2,
                           .minimum_bytes = impl->info.a16_bytes});
  impl->a16 = descriptor("a16", "b3.rmsnorm_projection.a16",
                         std::move(a16_resources), 3);
  auto a4_resources = common;
  a4_resources.push_back({.argument_index = 2,
                          .resource_slot = 3,
                          .minimum_bytes = impl->info.a4_code_bytes});
  a4_resources.push_back({.argument_index = 3,
                          .resource_slot = 4,
                          .minimum_bytes = impl->info.a4_scale_bytes});
  impl->a4 =
      descriptor("a4", "b3.rmsnorm_projection.a4", std::move(a4_resources), 4);
  auto dual_resources = common;
  dual_resources.push_back({.argument_index = 2,
                            .resource_slot = 2,
                            .minimum_bytes = impl->info.a16_bytes});
  dual_resources.push_back({.argument_index = 3,
                            .resource_slot = 3,
                            .minimum_bytes = impl->info.a4_code_bytes});
  dual_resources.push_back({.argument_index = 4,
                            .resource_slot = 4,
                            .minimum_bytes = impl->info.a4_scale_bytes});
  impl->a16_a4 = descriptor("a16_a4", "b3.rmsnorm_projection.a16_a4",
                            std::move(dual_resources), 5);
  return RmsNormProjectionPlan(std::move(impl));
}

RmsNormProduction produce_rmsnorm_activation(const MetalContext &context,
                                             const RmsNormProjectionPlan &plan,
                                             const MetalBuffer &input,
                                             std::uint64_t input_offset,
                                             const MetalBuffer &weight,
                                             std::uint64_t weight_offset,
                                             const ActivationLease &lease) {
  if (plan.impl_ == nullptr || input_offset % 2 != 0 ||
      weight_offset % 2 != 0 || lease.bundle.epoch == 0 ||
      lease.bundle.key.input_features != plan.impl_->info.input_features ||
      lease.bundle.key.tokens != plan.impl_->info.tokens) {
    throw std::invalid_argument("RMSNorm activation production is invalid");
  }
  const auto mode = lease.produce_a16 && lease.produce_a4
                        ? RmsNormProducerMode::a16_a4
                    : lease.produce_a16 ? RmsNormProducerMode::a16
                    : lease.produce_a4  ? RmsNormProducerMode::a4
                                        : RmsNormProducerMode::none;
  if (mode == RmsNormProducerMode::none) {
    return {.mode = mode};
  }
  std::vector<MetalBufferBinding> bindings = {
      {0, &input, input_offset},
      {1, &weight, weight_offset},
  };
  const MetalCommandPlan *command = nullptr;
  if (lease.produce_a16) {
    if (!lease.bundle.normalized_a16.has_value() ||
        lease.bundle.normalized_a16->buffer == nullptr ||
        lease.bundle.normalized_a16->bytes < plan.impl_->info.a16_bytes) {
      throw std::invalid_argument("RMSNorm A16 destination is invalid");
    }
    bindings.push_back({2, lease.bundle.normalized_a16->buffer,
                        lease.bundle.normalized_a16->offset});
  }
  if (lease.produce_a4) {
    if (!lease.bundle.a4.has_value() || lease.bundle.a4->group_size != 32 ||
        lease.bundle.a4->codes.buffer == nullptr ||
        lease.bundle.a4->scales.buffer == nullptr ||
        lease.bundle.a4->codes.bytes < plan.impl_->info.a4_code_bytes ||
        lease.bundle.a4->scales.bytes < plan.impl_->info.a4_scale_bytes) {
      throw std::invalid_argument("RMSNorm A4 destination is invalid");
    }
    bindings.push_back(
        {3, lease.bundle.a4->codes.buffer, lease.bundle.a4->codes.offset});
    bindings.push_back(
        {4, lease.bundle.a4->scales.buffer, lease.bundle.a4->scales.offset});
  }
  switch (mode) {
  case RmsNormProducerMode::a16:
    command = &plan.impl_->a16;
    break;
  case RmsNormProducerMode::a4:
    command = &plan.impl_->a4;
    break;
  case RmsNormProducerMode::a16_a4:
    command = &plan.impl_->a16_a4;
    break;
  case RmsNormProducerMode::none:
    break;
  }
  return {.mode = mode, .submission = context.execute(*command, bindings)};
}

} // namespace omlx::m5
