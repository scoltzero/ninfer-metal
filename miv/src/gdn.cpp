#include "omlx_m5/gdn.h"

#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace omlx::m5 {
namespace {

struct GdnLayerParams {
  std::uint32_t key_heads = 0;
  std::uint32_t value_heads = 0;
  std::uint32_t key_dim = 0;
  std::uint32_t value_dim = 0;
  std::uint32_t conv_width = 0;
  std::uint32_t conv_kernel = 0;
  std::uint32_t tokens = 0;
  float epsilon = 0.0F;
};

static_assert(sizeof(GdnLayerParams) == 32);

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

std::vector<std::byte> constant_bytes(const GdnLayerParams &params) {
  std::vector<std::byte> result(sizeof(params));
  std::memcpy(result.data(), &params, sizeof(params));
  return result;
}

bool token_domain(std::uint32_t tokens) {
  return tokens >= 1 && (tokens <= 128 || tokens == 512 || tokens == 1024);
}

void zero_buffer(MetalBuffer &buffer) {
  std::vector<std::byte> zeros(buffer.size());
  buffer.write(zeros);
}

} // namespace

class GdnLayerState::Impl {
public:
  GdnLayerStateInfo info;
  MetalBuffer recurrent;
  MetalBuffer history;
};

GdnLayerState::GdnLayerState(const MetalContext &context,
                             const QwenModelConfig &config)
    : impl_(std::make_unique<Impl>()) {
  if (config.linear_num_key_heads <= 0 || config.linear_num_value_heads <= 0 ||
      config.linear_key_head_dim <= 0 || config.linear_value_head_dim <= 0 ||
      config.linear_conv_kernel_dim <= 0 ||
      config.linear_num_value_heads % config.linear_num_key_heads != 0) {
    throw std::invalid_argument("GDN state geometry is invalid");
  }
  const auto key_heads = checked_u32(config.linear_num_key_heads, "GDN Hk");
  const auto value_heads = checked_u32(config.linear_num_value_heads, "GDN Hv");
  const auto key_dim = checked_u32(config.linear_key_head_dim, "GDN Dk");
  const auto value_dim = checked_u32(config.linear_value_head_dim, "GDN Dv");
  const auto conv_kernel =
      checked_u32(config.linear_conv_kernel_dim, "GDN conv kernel");
  const auto key_width = checked_multiply(key_heads, key_dim, "GDN key width");
  const auto value_width =
      checked_multiply(value_heads, value_dim, "GDN value width");
  const auto conv_width = key_width * 2 + value_width;
  impl_->info = {
      .key_heads = key_heads,
      .value_heads = value_heads,
      .key_dim = key_dim,
      .value_dim = value_dim,
      .conv_width = checked_u32(conv_width, "GDN conv width"),
      .conv_kernel = conv_kernel,
      .recurrent_state_bytes = checked_multiply(
          checked_multiply(value_width, key_dim, "GDN state values"), 4,
          "GDN state bytes"),
      .conv_history_bytes = checked_multiply(
          checked_multiply(conv_kernel - 1, conv_width, "GDN history values"),
          2, "GDN history bytes"),
  };
  impl_->recurrent = context.make_buffer(impl_->info.recurrent_state_bytes);
  impl_->history = context.make_buffer(impl_->info.conv_history_bytes);
  reset();
}

GdnLayerState::GdnLayerState(GdnLayerState &&) noexcept = default;
GdnLayerState &GdnLayerState::operator=(GdnLayerState &&) noexcept = default;
GdnLayerState::~GdnLayerState() = default;

const GdnLayerStateInfo &GdnLayerState::info() const { return impl_->info; }
void GdnLayerState::reset() {
  zero_buffer(impl_->recurrent);
  zero_buffer(impl_->history);
}
const MetalBuffer &GdnLayerState::recurrent() const { return impl_->recurrent; }
const MetalBuffer &GdnLayerState::history() const { return impl_->history; }

class GdnLayerPlan::Impl {
public:
  GdnLayerInfo info;
  MetalCommandPlan command_plan;
  MetalBuffer convolved;
  MetalBuffer recurrent;
};

GdnLayerPlan::GdnLayerPlan() = default;
GdnLayerPlan::GdnLayerPlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
GdnLayerPlan::GdnLayerPlan(GdnLayerPlan &&) noexcept = default;
GdnLayerPlan &GdnLayerPlan::operator=(GdnLayerPlan &&) noexcept = default;
GdnLayerPlan::~GdnLayerPlan() = default;

bool GdnLayerPlan::valid() const { return impl_ != nullptr; }

const GdnLayerInfo &GdnLayerPlan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("GDN layer plan is empty");
  }
  return impl_->info;
}

GdnLayerPlan prepare_gdn_layer(const MetalContext &context,
                               const std::filesystem::path &metallib,
                               const QwenModelConfig &config,
                               std::uint32_t tokens) {
  if (!token_domain(tokens) || config.rms_norm_eps <= 0.0 ||
      config.linear_num_key_heads <= 0 || config.linear_num_value_heads <= 0 ||
      config.linear_key_head_dim <= 0 || config.linear_value_head_dim <= 0 ||
      config.linear_key_head_dim > 256 || config.linear_value_head_dim > 256 ||
      config.linear_conv_kernel_dim <= 0 ||
      config.linear_num_value_heads % config.linear_num_key_heads != 0) {
    throw std::invalid_argument("GDN layer domain is invalid");
  }
  const auto key_heads = checked_u32(config.linear_num_key_heads, "GDN Hk");
  const auto value_heads = checked_u32(config.linear_num_value_heads, "GDN Hv");
  const auto key_dim = checked_u32(config.linear_key_head_dim, "GDN Dk");
  const auto value_dim = checked_u32(config.linear_value_head_dim, "GDN Dv");
  const auto conv_kernel =
      checked_u32(config.linear_conv_kernel_dim, "GDN conv kernel");
  const auto key_width = checked_multiply(key_heads, key_dim, "GDN key width");
  const auto value_width =
      checked_multiply(value_heads, value_dim, "GDN value width");
  const auto conv_width = key_width * 2 + value_width;
  const auto qkv_values = checked_multiply(tokens, conv_width, "GDN qkv");
  const auto value_values = checked_multiply(tokens, value_width, "GDN value");
  const auto ab_values = checked_multiply(tokens, value_heads, "GDN a/b");
  const GdnLayerParams params{
      .key_heads = key_heads,
      .value_heads = value_heads,
      .key_dim = key_dim,
      .value_dim = value_dim,
      .conv_width = checked_u32(conv_width, "GDN conv width"),
      .conv_kernel = conv_kernel,
      .tokens = tokens,
      .epsilon = static_cast<float>(config.rms_norm_eps),
  };
  auto impl = std::make_unique<GdnLayerPlan::Impl>();
  impl->info = {
      .executor_key = "gdn.layer.hk" + std::to_string(key_heads) + ".hv" +
                      std::to_string(value_heads) + ".dk" +
                      std::to_string(key_dim) + ".dv" +
                      std::to_string(value_dim) + ".t" + std::to_string(tokens),
      .tokens = tokens,
      .key_heads = key_heads,
      .value_heads = value_heads,
      .key_dim = key_dim,
      .value_dim = value_dim,
      .conv_width = checked_u32(conv_width, "GDN conv width"),
      .conv_kernel = conv_kernel,
      .epsilon = static_cast<float>(config.rms_norm_eps),
      .qkv_bytes = checked_multiply(qkv_values, 2, "GDN qkv bytes"),
      .z_bytes = checked_multiply(value_values, 2, "GDN z bytes"),
      .ab_bytes = checked_multiply(ab_values, 2, "GDN a/b bytes"),
      .output_bytes = checked_multiply(value_values, 2, "GDN output bytes"),
      .workspace_bytes =
          checked_multiply(qkv_values + value_values, 2, "GDN workspace bytes"),
  };
  impl->convolved = context.make_buffer(impl->info.qkv_bytes);
  impl->recurrent = context.make_buffer(impl->info.output_bytes);
  context.register_library({.key = "b3.gdn.library", .path = metallib});
  const auto warm = [&](std::string key, std::string function) {
    static_cast<void>(context.warmup_pipeline({
        .key = std::move(key),
        .library_key = "b3.gdn.library",
        .function_name = std::move(function),
        .variant = "bf16_fp32_state",
    }));
  };
  warm("b3.gdn.conv_silu", "omlx_m5_gdn_conv_silu");
  warm("b3.gdn.recurrent", "omlx_m5_gdn_recurrent");
  warm("b3.gdn.norm_gate", "omlx_m5_gdn_norm_gate");
  const auto params_bytes = constant_bytes(params);
  const auto conv_bytes = checked_multiply(
      checked_multiply(conv_width, conv_kernel, "GDN conv weight"), 2,
      "GDN conv weight bytes");
  const auto head_bytes = checked_multiply(value_heads, 2, "GDN head bytes");
  const auto norm_bytes = checked_multiply(value_dim, 2, "GDN norm bytes");
  const auto state_bytes =
      checked_multiply(checked_multiply(value_width, key_dim, "GDN state"), 4,
                       "GDN state bytes");
  const auto history_bytes = checked_multiply(
      checked_multiply(conv_kernel - 1, conv_width, "GDN history"), 2,
      "GDN history bytes");
  impl->command_plan = context.prepare_command_plan({
      .key = impl->info.executor_key,
      .dispatches =
          {
              {
                  .pipeline_key = "b3.gdn.conv_silu",
                  .grid = {.width = conv_width},
                  .threads_per_threadgroup = {.width = 256},
                  .resources =
                      {
                          {.argument_index = 0,
                           .resource_slot = 0,
                           .minimum_bytes = impl->info.qkv_bytes},
                          {.argument_index = 1,
                           .resource_slot = 4,
                           .minimum_bytes = conv_bytes},
                          {.argument_index = 2,
                           .resource_slot = 8,
                           .minimum_bytes = history_bytes},
                          {.argument_index = 3,
                           .resource_slot = 10,
                           .minimum_bytes = impl->info.qkv_bytes},
                      },
                  .constant_arguments = {{.argument_index = 4,
                                          .bytes = params_bytes}},
              },
              {
                  .pipeline_key = "b3.gdn.recurrent",
                  .grid = {.width = 32, .height = value_width},
                  .threads_per_threadgroup = {.width = 32},
                  .resources =
                      {
                          {.argument_index = 0,
                           .resource_slot = 10,
                           .minimum_bytes = impl->info.qkv_bytes},
                          {.argument_index = 1,
                           .resource_slot = 2,
                           .minimum_bytes = impl->info.ab_bytes},
                          {.argument_index = 2,
                           .resource_slot = 3,
                           .minimum_bytes = impl->info.ab_bytes},
                          {.argument_index = 3,
                           .resource_slot = 5,
                           .minimum_bytes = head_bytes},
                          {.argument_index = 4,
                           .resource_slot = 6,
                           .minimum_bytes = head_bytes},
                          {.argument_index = 5,
                           .resource_slot = 9,
                           .minimum_bytes = state_bytes},
                          {.argument_index = 6,
                           .resource_slot = 11,
                           .minimum_bytes = impl->info.output_bytes},
                      },
                  .constant_arguments = {{.argument_index = 7,
                                          .bytes = params_bytes}},
              },
              {
                  .pipeline_key = "b3.gdn.norm_gate",
                  .grid = {.width = 32, .height = tokens * value_heads},
                  .threads_per_threadgroup = {.width = 32},
                  .resources =
                      {
                          {.argument_index = 0,
                           .resource_slot = 11,
                           .minimum_bytes = impl->info.output_bytes},
                          {.argument_index = 1,
                           .resource_slot = 1,
                           .minimum_bytes = impl->info.z_bytes},
                          {.argument_index = 2,
                           .resource_slot = 7,
                           .minimum_bytes = norm_bytes},
                          {.argument_index = 3,
                           .resource_slot = 12,
                           .minimum_bytes = impl->info.output_bytes},
                      },
                  .constant_arguments = {{.argument_index = 4,
                                          .bytes = params_bytes}},
              },
          },
  });
  return GdnLayerPlan(std::move(impl));
}

MetalSubmission execute_gdn_layer(
    const MetalContext &context, const GdnLayerPlan &plan,
    const ProjectionParentOutputView &qkv, const ProjectionParentOutputView &z,
    const MetalBuffer &a, std::uint64_t a_offset, const MetalBuffer &b,
    std::uint64_t b_offset, const GdnDirectWeights &weights,
    GdnLayerState &state, MetalBuffer &output, std::uint64_t output_offset) {
  if (plan.impl_ == nullptr || state.impl_ == nullptr ||
      qkv.buffer == nullptr || z.buffer == nullptr || weights.conv == nullptr ||
      weights.a_log == nullptr || weights.dt_bias == nullptr ||
      weights.norm == nullptr || qkv.tokens != plan.impl_->info.tokens ||
      z.tokens != plan.impl_->info.tokens ||
      qkv.features != plan.impl_->info.conv_width ||
      z.features != plan.impl_->info.value_heads * plan.impl_->info.value_dim ||
      qkv.token_stride_bytes != plan.impl_->info.conv_width * 2ULL ||
      z.token_stride_bytes != z.features * 2 ||
      qkv.bytes < plan.impl_->info.qkv_bytes ||
      z.bytes < plan.impl_->info.z_bytes || a_offset % 2 != 0 ||
      b_offset % 2 != 0 || output_offset % 2 != 0 ||
      state.impl_->info.key_heads != plan.impl_->info.key_heads ||
      state.impl_->info.value_heads != plan.impl_->info.value_heads ||
      state.impl_->info.key_dim != plan.impl_->info.key_dim ||
      state.impl_->info.value_dim != plan.impl_->info.value_dim ||
      weights.conv_bytes <
          plan.impl_->info.conv_width * plan.impl_->info.conv_kernel * 2ULL ||
      weights.a_log_bytes < plan.impl_->info.value_heads * 2ULL ||
      weights.dt_bias_bytes < plan.impl_->info.value_heads * 2ULL ||
      weights.norm_bytes < plan.impl_->info.value_dim * 2ULL) {
    throw std::invalid_argument("GDN layer invocation is invalid");
  }
  const std::vector<MetalBufferBinding> bindings = {
      {0, qkv.buffer, qkv.offset},
      {1, z.buffer, z.offset},
      {2, &a, a_offset},
      {3, &b, b_offset},
      {4, weights.conv, weights.conv_offset},
      {5, weights.a_log, weights.a_log_offset},
      {6, weights.dt_bias, weights.dt_bias_offset},
      {7, weights.norm, weights.norm_offset},
      {8, &state.impl_->history, 0},
      {9, &state.impl_->recurrent, 0},
      {10, &plan.impl_->convolved, 0},
      {11, &plan.impl_->recurrent, 0},
      {12, &output, output_offset},
  };
  return context.execute(plan.impl_->command_plan, bindings);
}

} // namespace omlx::m5
