#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/metal.h"

namespace {

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

std::vector<std::byte> constant_u32(std::uint32_t value) {
  std::vector<std::byte> result(sizeof(value));
  std::memcpy(result.data(), &value, sizeof(value));
  return result;
}

} // namespace

int main() {
  try {
    constexpr std::uint32_t count = 4097;
    auto context = omlx::m5::MetalContext::create();
    context.register_library({
        .key = "b3.batch.smoke.library",
        .path = OMLX_M5_CLASSIC_METALLIB_PATH,
    });
    static_cast<void>(context.warmup_pipeline({
        .key = "b3.batch.copy.pipeline",
        .library_key = "b3.batch.smoke.library",
        .function_name = "omlx_m5_copy_u32",
        .variant = "u32",
    }));
    auto plan = context.prepare_command_plan({
        .key = "b3.batch.copy.plan",
        .dispatches = {{
            .pipeline_key = "b3.batch.copy.pipeline",
            .grid = {.width = count},
            .threads_per_threadgroup = {.width = 128},
            .resources =
                {
                    {.argument_index = 0,
                     .resource_slot = 0,
                     .minimum_bytes = count * 4ULL},
                    {.argument_index = 1,
                     .resource_slot = 1,
                     .minimum_bytes = count * 4ULL},
                },
            .constant_arguments = {{.argument_index = 2,
                                    .bytes = constant_u32(count)}},
        }},
    });
    std::vector<std::uint32_t> values(count);
    for (std::uint32_t index = 0; index < count; ++index) {
      values[index] = index * 31U + 11U;
    }
    auto source = context.upload(bytes(values));
    auto sequential_middle = context.make_buffer(count * 4ULL);
    auto sequential_output = context.make_buffer(count * 4ULL);
    const auto before_sequential = context.registry_stats();
    static_cast<void>(context.execute(
        plan,
        std::array{omlx::m5::MetalBufferBinding{0, &source, 0},
                   omlx::m5::MetalBufferBinding{1, &sequential_middle, 0}}));
    static_cast<void>(context.execute(
        plan,
        std::array{omlx::m5::MetalBufferBinding{0, &sequential_middle, 0},
                   omlx::m5::MetalBufferBinding{1, &sequential_output, 0}}));
    const auto after_sequential = context.registry_stats();

    auto batch_middle = context.make_buffer(count * 4ULL);
    auto batch_output = context.make_buffer(count * 4ULL);
    const std::array invocations = {
        omlx::m5::MetalCommandInvocation{
            .plan = &plan,
            .bindings = {{0, &source, 0}, {1, &batch_middle, 0}},
        },
        omlx::m5::MetalCommandInvocation{
            .plan = &plan,
            .bindings = {{0, &batch_middle, 0}, {1, &batch_output, 0}},
        },
    };
    const auto before_batch = context.registry_stats();
    const auto submission = context.execute_batch(invocations);
    const auto after_batch = context.registry_stats();
    const bool bit_exact = sequential_output.read() == batch_output.read() &&
                           batch_output.read() == bytes(values);
    const nlohmann::json result = {
        {"schema", "omlx-m5-b3-9-command-batch-v1"},
        {"evidence_scope", "M4 command-buffer batching correctness"},
        {"invocation_count", invocations.size()},
        {"dispatch_count", submission.dispatch_count},
        {"sequential_command_buffers",
         after_sequential.command_buffers_submitted -
             before_sequential.command_buffers_submitted},
        {"batched_command_buffers", after_batch.command_buffers_submitted -
                                        before_batch.command_buffers_submitted},
        {"batched_dispatches",
         after_batch.dispatches_submitted - before_batch.dispatches_submitted},
        {"bit_exact", bit_exact},
        {"token_loop_uploads",
         after_batch.upload_count - before_batch.upload_count},
        {"token_loop_pipeline_compiles",
         after_batch.pipelines_compiled - before_batch.pipelines_compiled},
    };
    if (!bit_exact) {
      throw std::runtime_error("command batch output mismatch");
    }
    std::cout << result.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "command batch inspection failed: " << error.what() << '\n';
    return 1;
  }
}
