#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "omlx_m5/hw_counters.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/metal.h"

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <class Exception = std::exception, class Function>
void expect_throws(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

std::vector<std::byte> constant_u32(std::uint32_t value) {
  std::vector<std::byte> result(sizeof(value));
  std::memcpy(result.data(), &value, sizeof(value));
  return result;
}

omlx::m5::MetalCommandPlan make_copy_plan(omlx::m5::MetalContext &context,
                                          std::string key,
                                          std::uint32_t count) {
  context.register_library({
      .key = "b3.schedule.smoke.library",
      .path = OMLX_M5_CLASSIC_METALLIB_PATH,
  });
  static_cast<void>(context.warmup_pipeline({
      .key = "b3.schedule.copy.pipeline",
      .library_key = "b3.schedule.smoke.library",
      .function_name = "omlx_m5_copy_u32",
      .variant = "u32",
  }));
  return context.prepare_command_plan({
      .key = std::move(key),
      .dispatches = {{
          .pipeline_key = "b3.schedule.copy.pipeline",
          .grid = {.width = count},
          .threads_per_threadgroup = {.width = 64},
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
}

void test_command_batch() {
  constexpr std::uint32_t count = 257;
  auto context = omlx::m5::MetalContext::create();
  auto plan = make_copy_plan(context, "b3.schedule.copy", count);
  std::vector<std::uint32_t> values(count);
  for (std::uint32_t index = 0; index < count; ++index) {
    values[index] = index * 17U + 3U;
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
  expect(sequential_output.read() == batch_output.read() &&
             batch_output.read() == bytes(values),
         "command batch changed dependency-ordered output");
  expect(after_sequential.command_buffers_submitted -
                     before_sequential.command_buffers_submitted ==
                 2 &&
             after_batch.command_buffers_submitted -
                     before_batch.command_buffers_submitted ==
                 1 &&
             submission.dispatch_count == 2 &&
             after_batch.dispatches_submitted -
                     before_batch.dispatches_submitted ==
                 2,
         "command batch did not reduce host submissions");
  expect_throws(
      [&] {
        static_cast<void>(context.execute_batch(
            std::span<const omlx::m5::MetalCommandInvocation>{}));
      },
      "empty command batch was accepted");

  auto foreign_context = omlx::m5::MetalContext::create();
  auto foreign_plan =
      make_copy_plan(foreign_context, "b3.schedule.foreign", count);
  const std::array foreign = {omlx::m5::MetalCommandInvocation{
      .plan = &foreign_plan,
      .bindings = {{0, &source, 0}, {1, &batch_output, 0}},
  }};
  expect_throws([&] { static_cast<void>(context.execute_batch(foreign)); },
                "foreign command plan was accepted by batch executor");
}

void test_command_batch_cross_invocation_barrier_with_barrier_before_false() {
  constexpr std::uint32_t count = 65536;
  auto context = omlx::m5::MetalContext::create();
  context.register_library({
      .key = "b3.schedule.smoke.library",
      .path = OMLX_M5_CLASSIC_METALLIB_PATH,
  });
  static_cast<void>(context.warmup_pipeline({
      .key = "b3.schedule.copy.pipeline",
      .library_key = "b3.schedule.smoke.library",
      .function_name = "omlx_m5_copy_u32",
      .variant = "u32",
  }));
  // Matches forward::Encoder::finish() which sets dispatches[0].barrier_before = false.
  auto plan = context.prepare_command_plan({
      .key = "b3.schedule.copy.no_leading_barrier",
      .dispatches = {{
          .pipeline_key = "b3.schedule.copy.pipeline",
          .barrier_before = false,
          .grid = {.width = count},
          .threads_per_threadgroup = {.width = 64},
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
  expect(plan.barrier_count() == 0,
         "single-dispatch plan with barrier_before=false should report 0 intra-plan barriers");

  auto source = context.make_buffer(count * 4ULL);
  auto hop1 = context.make_buffer(count * 4ULL);
  auto hop2 = context.make_buffer(count * 4ULL);
  auto output = context.make_buffer(count * 4ULL);
  const std::array invocations = {
      omlx::m5::MetalCommandInvocation{
          .plan = &plan,
          .bindings = {{0, &source, 0}, {1, &hop1, 0}},
      },
      omlx::m5::MetalCommandInvocation{
          .plan = &plan,
          .bindings = {{0, &hop1, 0}, {1, &hop2, 0}},
      },
      omlx::m5::MetalCommandInvocation{
          .plan = &plan,
          .bindings = {{0, &hop2, 0}, {1, &output, 0}},
      },
  };

  std::vector<std::uint32_t> values(count);
  for (std::uint32_t iter = 0; iter < 8; ++iter) {
    for (std::uint32_t i = 0; i < count; ++i) {
      values[i] = (i * 2654435761U) ^ (iter * 1013904223U + 17U);
    }
    const auto expected = bytes(values);
    source.write(expected);
    hop1.zero();
    hop2.zero();
    output.zero();
    const auto sub = context.execute_batch(invocations);
    expect(sub.dispatch_count == 3 && output.read() == expected,
           "MIVR-3 execute_batch violated cross-invocation ordering when barrier_before=false");
  }
}

void test_private_buffer_host_access_rejected() {
  auto context = omlx::m5::MetalContext::create();
  const std::array<std::byte, 8> payload{};
  const auto before_uploads = context.registry_stats().upload_count;

  // Test both small suballocated private buffers (64 B) and large private
  // buffers (64 KiB, where [MTLBuffer contents] returns nullptr).
  for (const std::uint64_t size : std::array<std::uint64_t, 2>{64ULL, 65536ULL}) {
    auto priv = context.make_private_buffer(size);
    expect_throws<std::logic_error>(
        [&] { static_cast<void>(priv.view()); },
        "MIVR-20 private buffer view() did not throw logic_error");
    expect_throws<std::logic_error>(
        [&] { static_cast<void>(priv.read()); },
        "MIVR-20 private buffer read() did not throw logic_error");
    expect_throws<std::logic_error>(
        [&] { priv.write(payload); },
        "MIVR-20 private buffer write() did not throw logic_error");
    expect_throws<std::logic_error>(
        [&] { priv.write(std::span<const std::byte>{}); },
        "MIVR-20 private buffer empty write() did not throw logic_error");
    expect_throws<std::logic_error>(
        [&] { context.upload_into(priv, payload); },
        "MIVR-20 upload_into(private buffer) did not throw logic_error");
    expect_throws<std::logic_error>(
        [&] { priv.zero(); },
        "MIVR-20 private buffer zero() did not throw logic_error");
  }
  expect(context.registry_stats().upload_count == before_uploads,
         "MIVR-20 failed upload_into mutated upload_count");
}

void test_hw_counter_compute_copy_bounds_and_smoke() {
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(omlx::m5::profile_compute_copy_counters(
            OMLX_M5_CLASSIC_METALLIB_PATH, 16ULL << 30, 0, 1, true, 1));
      },
      "MIVE-19 profile_compute_copy_counters accepted >= 16 GiB byte count");

  const auto copy_stats = omlx::m5::profile_compute_copy_counters(
      OMLX_M5_CLASSIC_METALLIB_PATH, 4096, 1, 2, true, 2);
  expect(copy_stats.dispatch_count_per_iter == 2 && copy_stats.gpu_ns.count == 2,
         "MIVE-3 profile_compute_copy_counters returned unexpected sample counts");

  const auto gap_report = omlx::m5::profile_dispatch_gap_counters(
      OMLX_M5_CLASSIC_METALLIB_PATH, 2, 1, 2);
  expect(gap_report.chain_length == 2 &&
             gap_report.single_encoder_chain.gpu_ns.count == 2,
         "MIVE-3 profile_dispatch_gap_counters returned unexpected sample counts");
}

} // namespace

int main() {
  try {
    test_command_batch();
    test_command_batch_cross_invocation_barrier_with_barrier_before_false();
    test_private_buffer_host_access_rejected();
    test_hw_counter_compute_copy_bounds_and_smoke();
    std::cout << "B3.9 command batch tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B3.9 schedule failure: " << error.what() << '\n';
    return 1;
  }
}
