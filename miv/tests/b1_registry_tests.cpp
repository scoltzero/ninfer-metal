#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "omlx_m5/capability.h"
#include "omlx_m5/metal.h"

#ifndef OMLX_M5_CLASSIC_METALLIB_PATH
#error "OMLX_M5_CLASSIC_METALLIB_PATH is required"
#endif

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <class Exception, class Function>
void expect_throws(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

std::vector<std::byte> bytes_of(std::span<const std::uint32_t> values) {
  const auto bytes = std::as_bytes(values);
  return {bytes.begin(), bytes.end()};
}

std::vector<std::byte> constant_u32(std::uint32_t value) {
  std::vector<std::byte> result(sizeof(value));
  std::memcpy(result.data(), &value, sizeof(value));
  return result;
}

omlx::m5::MetalPipelineDescriptor copy_pipeline() {
  return {
      .key = "classic.copy_u32.default",
      .library_key = "classic.smoke",
      .function_name = "omlx_m5_copy_u32",
      .variant = "default",
  };
}

omlx::m5::MetalCommandPlanDescriptor copy_plan(std::uint32_t element_count,
                                               std::uint64_t thread_width) {
  const auto bytes =
      static_cast<std::uint64_t>(element_count) * sizeof(std::uint32_t);
  return {
      .key = "copy_u32." + std::to_string(element_count),
      .dispatches = {{
          .pipeline_key = "classic.copy_u32.default",
          .grid = {.width = element_count, .height = 1, .depth = 1},
          .threads_per_threadgroup = {.width = thread_width,
                                      .height = 1,
                                      .depth = 1},
          .resources =
              {
                  {.argument_index = 0,
                   .resource_slot = 0,
                   .minimum_bytes = bytes},
                  {.argument_index = 1,
                   .resource_slot = 1,
                   .minimum_bytes = bytes},
              },
          .constant_arguments = {{.argument_index = 2,
                                  .bytes = constant_u32(element_count)}},
      }},
  };
}

void test_capability_adapter() {
  using omlx::m5::MetalCapabilityProbe;
  using omlx::m5::MetalExecutionRoute;

  auto target = omlx::m5::select_metal_capability_profile(MetalCapabilityProbe{
      .runtime_os_major = 27,
      .sdk_max_allowed = 270000,
      .apple_gpu_family = 9,
      .classic_command_queue = true,
      .shared_event = true,
      .metal4_headers = true,
      .metal4_command_queue = true,
      .metal4_command_allocator = true,
      .metal4_command_buffer = true,
      .metal4_compute_encoder = true,
      .metal4_compiler = true,
      .metal4_argument_table = true,
      .compute_icb_api = true,
      .compute_icb_barrier = true,
      .residency_sets = true,
      .tensor_api = true,
      .native_tensorops_runtime = true,
      .native_tensorops_source = "fixture",
  });
  expect(target.selected_route == MetalExecutionRoute::native_tensorops,
         "validated macOS 27 fixture did not select native TensorOps");
  expect(target.native_tensorops_eligible,
         "native TensorOps fixture was not marked eligible");
  expect(target.compute_icb_eligible && target.metal4_icb_eligible,
         "macOS 27 fixture was not accepted for Metal 4 compute ICB");
  expect(target.metal4_direct_eligible,
         "macOS 27 fixture was not accepted for direct Metal 4 execution");

  auto unvalidated = target.probe;
  unvalidated.native_tensorops_runtime = false;
  auto classic = omlx::m5::select_metal_capability_profile(unvalidated);
  expect(classic.selected_route == MetalExecutionRoute::classic_msl,
         "unvalidated TensorOps did not fall back to classic Metal");
  expect(std::ranges::find(classic.fallback_reasons,
                           "tensorops_not_runtime_validated") !=
             classic.fallback_reasons.end(),
         "TensorOps fallback reason is missing");
  expect(classic.compute_icb_eligible && classic.metal4_icb_eligible,
         "ICB eligibility incorrectly depended on TensorOps validation");

  auto classic_icb_probe = target.probe;
  classic_icb_probe.metal4_command_queue = false;
  classic_icb_probe.metal4_command_allocator = false;
  classic_icb_probe.metal4_command_buffer = false;
  classic_icb_probe.metal4_compute_encoder = false;
  const auto classic_icb =
      omlx::m5::select_metal_capability_profile(classic_icb_probe);
  expect(classic_icb.compute_icb_eligible && !classic_icb.metal4_icb_eligible,
         "classic compute ICB capability was not separated from Metal 4");

  MetalCapabilityProbe bridge_probe;
  bridge_probe.shared_event = true;
  bridge_probe.mlx_bridge_available = true;
  const auto bridge = omlx::m5::select_metal_capability_profile(bridge_probe);
  expect(bridge.selected_route == MetalExecutionRoute::mlx_bridge,
         "MLX bridge fixture did not select the bridge route");

  const auto unavailable =
      omlx::m5::select_metal_capability_profile(MetalCapabilityProbe{});
  expect(unavailable.selected_route == MetalExecutionRoute::unavailable,
         "empty capability fixture was accepted");
}

void test_registry_and_command_plan() {
  constexpr std::uint32_t element_count = 257;
  std::vector<std::uint32_t> values(element_count);
  for (std::uint32_t index = 0; index < element_count; ++index) {
    values[index] = index * 17U + 3U;
  }
  const auto source_bytes = bytes_of(values);

  {
    auto default_context = omlx::m5::MetalContext::create();
    const auto &default_caps = default_context.capabilities();
    const auto &default_probe = default_caps.profile.probe;
    const bool expected_auto_tensorops =
        default_probe.apple_gpu_family >= 10 && default_probe.metal4_headers &&
        default_probe.metal4_command_queue &&
        default_probe.metal4_command_allocator &&
        default_probe.metal4_command_buffer &&
        default_probe.metal4_compute_encoder &&
        default_probe.metal4_compiler && default_probe.metal4_argument_table &&
        default_probe.tensor_api && default_probe.shared_event;
    expect(default_probe.apple_gpu_family >= 1 &&
               (!default_caps.device_name.starts_with("Apple M5") ||
                default_probe.apple_gpu_family >= 10),
           "Apple GPU family probe missed Apple10+ support");
    expect(default_probe.native_tensorops_runtime == expected_auto_tensorops,
           "default host TensorOps auto-detection disagreed with Apple10+ Metal 4 probe");
    expect(default_caps.profile.selected_route ==
               (default_probe.native_tensorops_runtime
                    ? omlx::m5::MetalExecutionRoute::native_tensorops
                    : omlx::m5::MetalExecutionRoute::classic_msl),
           "default host selected route disagreed with probe.native_tensorops_runtime");
    auto injected_true = omlx::m5::MetalContext::create(true);
    expect(injected_true.capabilities().profile.probe.native_tensorops_runtime &&
               injected_true.capabilities().nax_capability_source == "injected",
           "explicit TensorOps true override was not honored");
  }

  auto context = omlx::m5::MetalContext::create(false);
  const auto &capabilities = context.capabilities();
  expect(!capabilities.device_name.empty(), "Metal device name is empty");
  expect(capabilities.profile.probe.classic_command_queue,
         "classic command queue probe failed");
  expect(capabilities.profile.probe.shared_event, "shared event probe failed");
  expect(capabilities.profile.probe.metal4_headers ==
             (capabilities.profile.probe.sdk_max_allowed >= 260000),
         "Metal 4 header capability disagrees with the build SDK");
  expect(!capabilities.profile.probe.native_tensorops_runtime &&
             capabilities.profile.selected_route ==
                 omlx::m5::MetalExecutionRoute::classic_msl,
         "explicit false override must select classic Metal");

  const omlx::m5::MetalLibraryDescriptor library{
      .key = "classic.smoke",
      .path = OMLX_M5_CLASSIC_METALLIB_PATH,
  };
  {
    auto checksum_context = omlx::m5::MetalContext::create(false);
    auto wrong_hash = library;
    wrong_hash.key = "classic.wrong_hash";
    wrong_hash.expected_sha256 = std::string(64, '0');
    expect_throws<std::runtime_error>(
        [&] { checksum_context.register_library(wrong_hash); },
        "wrong library SHA-256 was accepted");
    expect(checksum_context.registry_stats().library_load_failures == 1,
           "library checksum failure was not accounted");
  }
  context.register_library(library);
  context.register_library(library);
  auto conflict = library;
  conflict.expected_sha256 = std::string(64, '0');
  expect_throws<std::invalid_argument>(
      [&] { context.register_library(conflict); },
      "conflicting library identity was accepted");

  auto missing = copy_pipeline();
  missing.key = "classic.missing";
  missing.function_name = "omlx_m5_missing";
  expect_throws<std::runtime_error>(
      [&] { static_cast<void>(context.warmup_pipeline(missing)); },
      "missing Metal function compiled");

  const auto pipeline = context.warmup_pipeline(copy_pipeline());
  const auto cached_pipeline = context.warmup_pipeline(copy_pipeline());
  expect(pipeline.thread_execution_width > 0,
         "pipeline thread execution width is empty");
  expect(pipeline.max_threads_per_threadgroup >=
             pipeline.thread_execution_width,
         "pipeline thread limits are inconsistent");
  expect(pipeline.bindings.size() == 3,
         "copy pipeline reflection binding count mismatch");
  expect(cached_pipeline.bindings.size() == pipeline.bindings.size(),
         "cached pipeline reflection changed");

  const auto descriptor =
      copy_plan(element_count, pipeline.thread_execution_width);
  const auto plan = context.prepare_command_plan(descriptor);
  const auto cached_plan = context.prepare_command_plan(descriptor);
  expect(plan.valid() && cached_plan.valid(), "command plan is invalid");
  expect(plan.dispatch_count() == 1 && plan.resource_slot_count() == 2,
         "command plan geometry mismatch");
  auto conflicting_plan = descriptor;
  ++conflicting_plan.dispatches.front().grid.width;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(context.prepare_command_plan(conflicting_plan));
      },
      "conflicting command plan key was accepted");

  auto incomplete = descriptor;
  incomplete.key = "copy_u32.incomplete";
  incomplete.dispatches.front().constant_arguments.clear();
  expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(context.prepare_command_plan(incomplete)); },
      "reflection-incomplete command plan was accepted");

  auto source = context.upload(source_bytes);
  auto destination = context.make_buffer(source_bytes.size());
  const std::array bindings = {
      omlx::m5::MetalBufferBinding{.resource_slot = 0, .buffer = &source},
      omlx::m5::MetalBufferBinding{.resource_slot = 1, .buffer = &destination},
  };

  context.seal_warmup();
  expect(context.warmup_sealed(), "Metal warmup did not seal");
  expect_throws<std::logic_error>(
      [&] { static_cast<void>(context.warmup_pipeline(copy_pipeline())); },
      "pipeline compilation remained available after warmup seal");
  expect_throws<std::logic_error>(
      [&] { context.register_library(library); },
      "library loading remained available after warmup seal");

  const auto first = context.execute(plan, bindings);
  const auto second = context.execute(plan, bindings);
  expect(first.shared_event_value > 0 &&
             second.shared_event_value > first.shared_event_value,
         "shared event values are not monotonic");
  expect(destination.read() == source_bytes,
         "prepared command plan copy mismatch");
  const std::array missing_binding = {
      omlx::m5::MetalBufferBinding{.resource_slot = 0, .buffer = &source},
  };
  expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(context.execute(plan, missing_binding)); },
      "incomplete invocation binding was accepted");

  auto other_context = omlx::m5::MetalContext::create(false);
  expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(other_context.execute(plan, bindings)); },
      "foreign context accepted a command plan");
  auto foreign_destination = other_context.make_buffer(source_bytes.size());
  const std::array foreign_buffer_bindings = {
      omlx::m5::MetalBufferBinding{.resource_slot = 0, .buffer = &source},
      omlx::m5::MetalBufferBinding{.resource_slot = 1,
                                   .buffer = &foreign_destination},
  };
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(context.execute(plan, foreign_buffer_bindings));
      },
      "Metal 4 executor accepted a buffer from another context");

  constexpr std::size_t thread_count = 4;
  std::vector<omlx::m5::MetalBuffer> destinations;
  destinations.reserve(thread_count);
  for (std::size_t index = 0; index < thread_count; ++index) {
    destinations.push_back(context.make_buffer(source_bytes.size()));
  }
  std::vector<std::exception_ptr> errors(thread_count);
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (std::size_t index = 0; index < thread_count; ++index) {
    threads.emplace_back([&, index] {
      try {
        const std::array invocation = {
            omlx::m5::MetalBufferBinding{.resource_slot = 0, .buffer = &source},
            omlx::m5::MetalBufferBinding{.resource_slot = 1,
                                         .buffer = &destinations[index]},
        };
        static_cast<void>(context.execute(plan, invocation));
      } catch (...) {
        errors[index] = std::current_exception();
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  for (std::size_t index = 0; index < thread_count; ++index) {
    if (errors[index] != nullptr) {
      std::rethrow_exception(errors[index]);
    }
    expect(destinations[index].read() == source_bytes,
           "concurrent command plan copy mismatch");
  }

  const auto before_heap = context.registry_stats();
  {
    auto heap = context.make_heap(1ULL << 20);
    auto heap_buffer = heap.make_buffer(4096);
    const auto during_heap = context.registry_stats();
    expect(during_heap.current_heap_count == 1 &&
               during_heap.current_heap_bytes >= (1ULL << 20),
           "live heap accounting mismatch");
    expect(during_heap.current_buffer_count ==
               before_heap.current_buffer_count + 1,
           "heap buffer accounting mismatch");
    expect(heap_buffer.size() == 4096, "heap buffer size mismatch");
  }
  const auto after_heap = context.registry_stats();
  expect(after_heap.current_heap_count == 0 &&
             after_heap.current_buffer_count ==
                 before_heap.current_buffer_count,
         "resource destruction did not update the ledger");

  const auto barrier = context.submit_barrier();
  expect(barrier.wall_ms >= 0.0, "barrier timing is invalid");
  const auto stats = context.registry_stats();
  expect(stats.libraries_loaded == 1 && stats.library_cache_hits == 1,
         "library cache accounting mismatch");
  expect(stats.pipelines_compiled == 1 && stats.pipeline_cache_hits == 1 &&
             stats.pipeline_compile_failures == 1,
         "pipeline cache accounting mismatch");
  expect(stats.command_plans_built == 1 && stats.command_plan_cache_hits == 1,
         "command plan cache accounting mismatch");
  expect(stats.command_buffers_submitted == 7 &&
             stats.dispatches_submitted == 6,
         "submission accounting mismatch");
  const auto expected_metal4_submissions =
      capabilities.profile.metal4_direct_eligible ? 7ULL : 0ULL;
  expect(stats.metal4_command_buffers_submitted ==
                 expected_metal4_submissions &&
             stats.classic_command_buffers_submitted ==
                 7ULL - expected_metal4_submissions,
         "command plans did not use the qualified submission path");
  expect(stats.metal4_argument_tables_created ==
                 (capabilities.profile.metal4_direct_eligible ? 6ULL : 0ULL) &&
             stats.stable_parameter_buffers_created == 1 &&
             stats.stable_parameter_bytes == sizeof(std::uint32_t),
         "Metal 4 argument-table or stable-parameter accounting mismatch");
  expect(stats.last_shared_event_value == 7,
         "shared event accounting mismatch");
  expect(stats.upload_count == 1 && stats.uploaded_bytes == source_bytes.size(),
         "upload accounting mismatch");
  expect(stats.current_buffer_count == 3 + thread_count,
         "live buffer count mismatch");
  expect(stats.peak_buffer_bytes >= stats.current_buffer_bytes,
         "buffer peak accounting mismatch");
  expect(stats.peak_heap_bytes >= (1ULL << 20),
         "heap peak accounting mismatch");
  expect(stats.warmup_sealed, "sealed state missing from registry stats");
}

void test_registry_lifetime() {
  std::weak_ptr<void> lifetime;
  omlx::m5::MetalCommandPlan retained;
  {
    auto context=omlx::m5::MetalContext::create();
    lifetime=context.diagnostic_owner_lifetime();
    context.register_library({.key="classic.smoke",.path=OMLX_M5_CLASSIC_METALLIB_PATH});
    const auto pipeline=context.warmup_pipeline(copy_pipeline());
    retained=context.prepare_command_plan(copy_plan(4,pipeline.thread_execution_width));
  }
  expect(!lifetime.expired(),"external command plan lost its owning registry");
  retained={};
  expect(lifetime.expired(),"command plan cache retained a registry ownership cycle");
  auto context=omlx::m5::MetalContext::create();
  context.register_library({.key="classic.smoke",.path=OMLX_M5_CLASSIC_METALLIB_PATH});
  const auto pipeline=context.warmup_pipeline(copy_pipeline());
  const auto descriptor=copy_plan(4,pipeline.thread_execution_width);
  auto plan=context.prepare_command_plan(descriptor);
  expect(context.registry_stats().current_buffer_count==1,"constant buffer missing");
  plan={};
  expect(context.registry_stats().current_buffer_count==0,"expired plan retained constant buffers");
  auto restored=context.prepare_command_plan(descriptor);
  expect(restored.valid() && context.registry_stats().command_plans_built==2,"expired same-key plan was not rebuilt");
  auto conflict=descriptor;++conflict.dispatches.front().grid.width;
  expect_throws<std::invalid_argument>([&]{static_cast<void>(context.prepare_command_plan(conflict));},
    "live cached descriptor allowed a conflicting key");
  restored={};
  auto replaced=context.prepare_command_plan(conflict);
  expect(replaced.valid() && context.registry_stats().command_plans_built==3,
    "expired cached descriptor poisoned command plan key");
  replaced={};
  if (context.capabilities().profile.compute_icb_eligible) {
    auto icb_pipe_desc = copy_pipeline();
    icb_pipe_desc.key = "classic.copy_u32.icb";
    icb_pipe_desc.variant = "icb";
    icb_pipe_desc.support_indirect_command_buffers = true;
    const auto icb_pipe = context.warmup_pipeline(icb_pipe_desc);
    const std::array<std::uint32_t, 4> input_vals = {11U, 22U, 33U, 44U};
    const auto input_bytes = bytes_of(input_vals);
    auto src = context.upload(input_bytes);
    auto dst = context.make_buffer(input_bytes.size());
    omlx::m5::MetalCommandPlanDescriptor dyn_desc{
        .key = "copy_u32.icb.dyn",
        .icb_qualified = true,
        .dispatches = {{
            .pipeline_key = icb_pipe_desc.key,
            .grid = {.width = 4, .height = 1, .depth = 1},
            .threads_per_threadgroup =
                {.width = icb_pipe.thread_execution_width, .height = 1, .depth = 1},
            .resources =
                {
                    {.argument_index = 0,
                     .resource_slot = 0,
                     .minimum_bytes = input_bytes.size()},
                    {.argument_index = 1,
                     .resource_slot = 1,
                     .minimum_bytes = input_bytes.size()},
                },
            .dynamic_arguments =
                {{.argument_index = 2, .parameter_offset = 0, .minimum_bytes = 4}},
        }},
    };
    auto dyn_plan = context.prepare_command_plan(dyn_desc);
    omlx::m5::MetalIcbProfile icb_profile;
    {
      auto params = context.make_parameter_table({
          .key = "copy_u32.icb.params",
          .byte_size = 256,
          .fields = {{.key = "count", .offset = 0, .size = 4, .alignment = 4}},
      });
      params.write("count", constant_u32(4));
      icb_profile = context.prepare_icb_profile(
          "copy_u32.icb.profile",
          {.plan = &dyn_plan,
           .bindings = {{0, &src, 0}, {1, &dst, 0}},
           .parameters = &params});
    }
    const auto replay = context.replay_icb(icb_profile);
    expect(replay.dispatch_count == 1 && dst.read() == input_bytes,
           "ICB replay after parameter table destruction failed");
  }
}

} // namespace

int main() {
  try {
    test_capability_adapter();
    test_registry_and_command_plan();
    test_registry_lifetime();
    std::cout << "B1.1 Metal registry tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B1.1 Metal registry test failure: " << error.what() << '\n';
    return 1;
  }
}
