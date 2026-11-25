#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "omlx_m5/metal.h"

#ifndef OMLX_M5_D3_METALLIB_PATH
#define OMLX_M5_D3_METALLIB_PATH ""
#endif

namespace {

template <typename T>
std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

template <typename T> std::vector<std::byte> constant_bytes(const T &value) {
  std::vector<std::byte> result(sizeof(value));
  std::memcpy(result.data(), &value, sizeof(value));
  return result;
}

void dispatch(omlx::m5::MetalContext &context,
              const omlx::m5::MetalPipelineInfo &pipeline,
              omlx::m5::MetalBuffer &state, const omlx::m5::MetalBuffer &keys,
              const omlx::m5::MetalBuffer &values,
              const omlx::m5::MetalBuffer &log_decay,
              const omlx::m5::MetalBuffer &beta, std::uint32_t key_dim,
              std::uint32_t value_dim, std::uint32_t start, std::uint32_t count,
              float epsilon) {
  if (count == 0) {
    return;
  }
  const auto plan = context
                        .prepare_command_plan({
                            .key = "d3.gdn." + std::to_string(start) + "." +
                                   std::to_string(count),
                            .dispatches = {{
                                .pipeline_key = pipeline.descriptor.key,
                                .grid = {.width = value_dim},
                                .threads_per_threadgroup =
                                    {
                                        .width = std::min<std::uint64_t>(
                                            value_dim,
                                            pipeline
                                                .max_threads_per_threadgroup),
                                    },
                                .resources =
                                    {
                                        {.argument_index = 0,
                                         .resource_slot = 0,
                                         .minimum_bytes = state.size()},
                                        {.argument_index = 1,
                                         .resource_slot = 1,
                                         .minimum_bytes = keys.size()},
                                        {.argument_index = 2,
                                         .resource_slot = 2,
                                         .minimum_bytes = values.size()},
                                        {.argument_index = 3,
                                         .resource_slot = 3,
                                         .minimum_bytes = log_decay.size()},
                                        {.argument_index = 4,
                                         .resource_slot = 4,
                                         .minimum_bytes = beta.size()},
                                    },
                                .constant_arguments =
                                    {
                                        {.argument_index = 5,
                                         .bytes = constant_bytes(key_dim)},
                                        {.argument_index = 6,
                                         .bytes = constant_bytes(value_dim)},
                                        {.argument_index = 7,
                                         .bytes = constant_bytes(start)},
                                        {.argument_index = 8,
                                         .bytes = constant_bytes(count)},
                                        {.argument_index = 9,
                                         .bytes = constant_bytes(epsilon)},
                                    },
                            }},
                        });
  const std::array bindings = {
      omlx::m5::MetalBufferBinding{0, &state, 0},
      omlx::m5::MetalBufferBinding{1, &keys, 0},
      omlx::m5::MetalBufferBinding{2, &values, 0},
      omlx::m5::MetalBufferBinding{3, &log_decay, 0},
      omlx::m5::MetalBufferBinding{4, &beta, 0},
  };
  static_cast<void>(context.execute(plan, bindings));
}

} // namespace

int main() {
  try {
    constexpr std::uint32_t key_dim = 8;
    constexpr std::uint32_t value_dim = 4;
    constexpr std::uint32_t steps = 8;
    constexpr float epsilon = 1.0e-6F;
    auto context = omlx::m5::MetalContext::create();
    context.register_library({
        .key = "d3.gdn.library",
        .path = OMLX_M5_D3_METALLIB_PATH,
    });
    const auto pipeline = context.warmup_pipeline({
        .key = "d3.gdn.pipeline",
        .library_key = "d3.gdn.library",
        .function_name = "omlx_m5_gdn_replay",
        .variant = "metal4-direct",
    });

    std::mt19937 generator(20260919);
    std::uniform_real_distribution<float> distribution(-0.25F, 0.25F);
    std::vector<float> initial(key_dim * value_dim);
    std::vector<float> keys(steps * key_dim);
    std::vector<float> values(steps * value_dim);
    std::vector<float> log_decay(steps);
    std::vector<float> beta(steps);
    std::ranges::generate(initial, [&] { return distribution(generator); });
    std::ranges::generate(keys, [&] { return distribution(generator); });
    std::ranges::generate(values, [&] { return distribution(generator); });
    for (std::uint32_t step = 0; step < steps; ++step) {
      log_decay[step] = -0.01F * static_cast<float>(step + 1);
      beta[step] = 0.25F + 0.01F * static_cast<float>(step);
    }
    auto key_buffer = context.upload(bytes(keys));
    auto value_buffer = context.upload(bytes(values));
    auto decay_buffer = context.upload(bytes(log_decay));
    auto beta_buffer = context.upload(bytes(beta));

    std::uint32_t passed = 0;
    for (std::uint32_t prefix = 0; prefix <= 7; ++prefix) {
      auto combined = context.upload(bytes(initial));
      auto stepped = context.upload(bytes(initial));
      dispatch(context, pipeline, combined, key_buffer, value_buffer,
               decay_buffer, beta_buffer, key_dim, value_dim, 0, prefix,
               epsilon);
      for (std::uint32_t step = 0; step < prefix; ++step) {
        dispatch(context, pipeline, stepped, key_buffer, value_buffer,
                 decay_buffer, beta_buffer, key_dim, value_dim, step, 1,
                 epsilon);
      }
      if (combined.read() != stepped.read()) {
        throw std::runtime_error(
            "Metal 4 ReplaySSM combined/stepped state mismatch");
      }
      ++passed;
    }
    const auto stats = context.registry_stats();
    std::cout << "{\n"
              << "  \"schema_version\": 2,\n"
              << "  \"probe\": \"d3_gdn_replay_metal4\",\n"
              << "  \"device_name\": \"" << context.capabilities().device_name
              << "\",\n"
              << "  \"prefixes_tested\": 8,\n"
              << "  \"bitwise_equal_prefixes\": " << passed << ",\n"
              << "  \"metal4_command_buffers\": "
              << stats.metal4_command_buffers_submitted << ",\n"
              << "  \"classic_command_buffers\": "
              << stats.classic_command_buffers_submitted << "\n"
              << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "omlx_m5_d3_gdn_probe: " << error.what() << '\n';
    return 1;
  }
}
