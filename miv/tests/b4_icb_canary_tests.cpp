#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "omlx_m5/metal.h"

#ifndef OMLX_M5_ICB_CANARY_METALLIB_PATH
#error "OMLX_M5_ICB_CANARY_METALLIB_PATH is required"
#endif

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

void install_canary(omlx::m5::MetalContext &context) {
  context.register_library({
      .key = "b4.0.icb.canary.library",
      .path = OMLX_M5_ICB_CANARY_METALLIB_PATH,
  });
  const auto stage_a = context.warmup_pipeline({
      .key = "b4.0.icb.stage_a.pipeline",
      .library_key = "b4.0.icb.canary.library",
      .function_name = "omlx_m5_icb_stage_a",
      .variant = "icb",
      .support_indirect_command_buffers = true,
  });
  const auto stage_b = context.warmup_pipeline({
      .key = "b4.0.icb.stage_b.pipeline",
      .library_key = "b4.0.icb.canary.library",
      .function_name = "omlx_m5_icb_stage_b",
      .variant = "icb",
      .support_indirect_command_buffers = true,
  });
  expect(stage_a.indirect_command_buffers_enabled &&
             stage_b.indirect_command_buffers_enabled,
         "pipeline ICB support was not enabled");
}

void test_replay_and_fail_closed_fallbacks() {
  auto context = omlx::m5::MetalContext::create();
  install_canary(context);
  const auto &profile = context.capabilities().profile;
  expect(profile.probe.compute_icb_api,
         "current M4 host does not expose compute ICB API");
  expect(profile.probe.compute_icb_barrier,
         "current M4 host does not expose compute ICB barriers");
  expect(profile.probe.residency_sets,
         "current M4 host does not expose residency sets");
  expect(profile.compute_icb_eligible,
         "current M4 host was not accepted for classic compute ICB");

  const auto replay = context.run_icb_canary(
      "b4.0.icb.stage_a.pipeline", "b4.0.icb.stage_b.pipeline",
      {.iterations = 128, .element_count = 257});
  const auto expected_route = profile.metal4_icb_eligible
                                  ? std::string("metal4_compute_icb")
                                  : std::string("classic_compute_icb");
  expect(replay.execution_route == expected_route,
         "eligible canary did not execute through an ICB");
  expect(!profile.metal4_icb_eligible || replay.metal4_pipeline_icb_enabled,
         "Metal 4 route did not use Metal 4 ICB-qualified pipelines");
  expect(replay.pipeline_icb_enabled && replay.dependency_barrier_encoded &&
             replay.bit_exact && replay.changing_parameters_observed &&
             replay.resources_released,
         "ICB replay correctness contract failed");
  expect(replay.icb_recordings == 1 && replay.icb_resets == 1 &&
             replay.icb_execute_calls == 128 &&
             replay.token_loop_icb_recordings == 0 &&
             replay.replay_command_buffer_commits == 128 &&
             replay.oracle_execute_batch_commits == 128 &&
             replay.compared_iterations == 128,
         "ICB recording/replay accounting changed");

  const auto missing_barrier = context.run_icb_canary(
      "b4.0.icb.stage_a.pipeline", "b4.0.icb.stage_b.pipeline",
      {.iterations = 4, .element_count = 257, .omit_dependency_barrier = true});
  expect(missing_barrier.execution_route == "execute_batch_fallback" &&
             missing_barrier.fallback_reason == "dependency_barrier_missing" &&
             !missing_barrier.dependency_barrier_encoded &&
             missing_barrier.icb_recordings == 0 &&
             missing_barrier.icb_execute_calls == 0 &&
             missing_barrier.bit_exact,
         "missing dependency barrier did not fail closed");

  const auto forced_fallback = context.run_icb_canary(
      "b4.0.icb.stage_a.pipeline", "b4.0.icb.stage_b.pipeline",
      {.iterations = 4,
       .element_count = 257,
       .force_execute_batch_fallback = true});
  expect(forced_fallback.execution_route == "execute_batch_fallback" &&
             forced_fallback.fallback_reason ==
                 "forced_execute_batch_fallback" &&
             forced_fallback.icb_recordings == 0 &&
             forced_fallback.icb_execute_calls == 0 &&
             forced_fallback.bit_exact && forced_fallback.resources_released,
         "forced ICB fallback did not preserve correctness");
}

void test_replay_icb_batch_barriers_and_stats_locking() {
  using namespace omlx::m5;
  auto context = MetalContext::create();
  install_canary(context);

  struct CanaryParams {
    std::uint32_t addend = 0;
    std::uint32_t multiplier = 1;
    std::uint32_t xor_mask = 0;
    std::uint32_t element_count = 0;
  };
  static_assert(sizeof(CanaryParams) == 16);

  constexpr std::uint32_t kCount = 257;
  constexpr std::uint64_t kBytes = std::uint64_t(kCount) * sizeof(std::uint32_t);
  std::vector<std::uint32_t> source_vals(kCount);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    source_vals[i] = i * 2654435761U + 17U;
  }
  auto source = context.upload(std::as_bytes(std::span(source_vals)));
  auto middle = context.make_buffer(kBytes);
  auto output = context.make_buffer(kBytes);
  auto oracle_middle = context.make_buffer(kBytes);
  auto oracle_output = context.make_buffer(kBytes);

  auto params = context.make_parameter_table({
      .key = "mivr11.icb_batch.params",
      .byte_size = sizeof(CanaryParams),
      .fields = {{.key = "step",
                  .offset = 0,
                  .size = sizeof(CanaryParams),
                  .alignment = 4}},
  });

  const auto make_stage_plan = [&](std::string key, std::string pipeline_key) {
    const auto threads = std::max<std::uint64_t>(
        1, context.pipeline_info(pipeline_key).thread_execution_width);
    return context.prepare_command_plan({
        .key = std::move(key),
        .icb_qualified = true,
        .dispatches = {{
            .pipeline_key = std::move(pipeline_key),
            .kind = MetalDispatchKind::threads,
            .barrier_before = true,
            .grid = {.width = kCount, .height = 1, .depth = 1},
            .threads_per_threadgroup = {.width = threads, .height = 1, .depth = 1},
            .resources = {{.argument_index = 0,
                           .resource_slot = 0,
                           .minimum_bytes = kBytes},
                          {.argument_index = 1,
                           .resource_slot = 1,
                           .minimum_bytes = kBytes}},
            .dynamic_arguments = {{.argument_index = 2,
                                   .parameter_offset = 0,
                                   .minimum_bytes = sizeof(CanaryParams)}},
        }},
        .ranges = {{.key = "full", .location = 0, .length = 1}},
    });
  };

  const auto plan_a =
      make_stage_plan("mivr11.plan_a", "b4.0.icb.stage_a.pipeline");
  const auto plan_b =
      make_stage_plan("mivr11.plan_b", "b4.0.icb.stage_b.pipeline");

  auto prof_a = context.prepare_icb_profile(
      "mivr11.prof_a",
      {.plan = &plan_a,
       .bindings = {{0, &source, 0}, {1, &middle, 0}},
       .parameters = &params});
  auto prof_b = context.prepare_icb_profile(
      "mivr11.prof_b",
      {.plan = &plan_b,
       .bindings = {{0, &middle, 0}, {1, &output, 0}},
       .parameters = &params});

  const std::array<MetalCommandInvocation, 2> oracle_calls{{
      {.plan = &plan_a,
       .bindings = {{0, &source, 0}, {1, &oracle_middle, 0}},
       .parameters = &params},
      {.plan = &plan_b,
       .bindings = {{0, &oracle_middle, 0}, {1, &oracle_output, 0}},
       .parameters = &params},
  }};
  const std::array<MetalIcbProfile *const, 2> batch_profiles{&prof_a, &prof_b};

  // Concurrent stats() reader verifies replay_icb_batch locks profile mutexes.
  std::atomic<bool> stop_reader{false};
  std::atomic<std::uint64_t> stat_reads{0};
  std::atomic<bool> stats_bad{false};
  std::thread reader([&] {
    while (!stop_reader.load(std::memory_order_acquire)) {
      const auto sa = prof_a.stats();
      const auto sb = prof_b.stats();
      if (sa.replay_count > 64 || sb.replay_count > 64) {
        stats_bad.store(true, std::memory_order_relaxed);
      }
      stat_reads.fetch_add(1, std::memory_order_relaxed);
    }
  });
  struct ReaderGuard {
    std::atomic<bool> &stop;
    std::thread &t;
    ~ReaderGuard() {
      stop.store(true, std::memory_order_release);
      if (t.joinable()) t.join();
    }
  } reader_guard{stop_reader, reader};

  constexpr std::uint32_t kIters = 64;
  for (std::uint32_t iter = 0; iter < kIters; ++iter) {
    const CanaryParams p{
        .addend = 13U + iter * 7U,
        .multiplier = 3U + (iter % 5U) * 2U,
        .xor_mask = 0xA5A50000U ^ (iter * 257U),
        .element_count = kCount,
    };
    params.write("step", std::as_bytes(std::span(&p, 1)));
    static_cast<void>(context.replay_icb_batch(batch_profiles));
    static_cast<void>(context.execute_batch(oracle_calls));
    const auto actual = output.read(0, kBytes);
    const auto expected = oracle_output.read(0, kBytes);
    expect(actual == expected,
           "MIVR-11 replay_icb_batch producer->consumer mismatch at iter " +
               std::to_string(iter));
  }

  stop_reader.store(true, std::memory_order_release);
  reader.join();

  expect(!stats_bad.load() && stat_reads.load() > 0 &&
             prof_a.stats().replay_count == kIters &&
             prof_b.stats().replay_count == kIters,
         "MIVR-11 replay_icb_batch stats accounting/locking failed");
}

void test_icb_rejects_threadgroups_indirect() {
  using namespace omlx::m5;
  auto context = MetalContext::create();
  install_canary(context);

  const auto threads = std::max<std::uint64_t>(
      1, context.pipeline_info("b4.0.icb.stage_a.pipeline")
             .thread_execution_width);
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(context.prepare_command_plan({
            .key = "mivr19.indirect_icb_rejected",
            .icb_qualified = true,
            .dispatches = {{
                .pipeline_key = "b4.0.icb.stage_a.pipeline",
                .kind = MetalDispatchKind::threadgroups_indirect,
                .grid = {.width = 1, .height = 1, .depth = 1},
                .threads_per_threadgroup = {.width = threads,
                                            .height = 1,
                                            .depth = 1},
                .resources = {{.argument_index = 0,
                               .resource_slot = 0,
                               .minimum_bytes = 64},
                              {.argument_index = 1,
                               .resource_slot = 1,
                               .minimum_bytes = 64}},
                .dynamic_arguments = {{.argument_index = 2,
                                       .parameter_offset = 0,
                                       .minimum_bytes = 16}},
                .indirect_grid_resource_slot = 2,
                .indirect_grid_offset = 0,
            }},
            .ranges = {{.key = "full", .location = 0, .length = 1}},
        }));
      },
      "MIVR-19 ICB-qualified plan accepted threadgroups_indirect dispatch");
}

void test_async_icb_barrier_and_profile_lifetime() {
  using namespace omlx::m5;
  auto context = MetalContext::create();
  install_canary(context);

  struct CanaryParams {
    std::uint32_t addend = 0;
    std::uint32_t multiplier = 1;
    std::uint32_t xor_mask = 0;
    std::uint32_t element_count = 0;
  };
  static_assert(sizeof(CanaryParams) == 16);

  constexpr std::uint32_t kCount = 65536;
  constexpr std::uint64_t kBytes = std::uint64_t(kCount) * sizeof(std::uint32_t);
  std::vector<std::uint32_t> source_vals(kCount);
  for (std::uint32_t i = 0; i < kCount; ++i) {
    source_vals[i] = i * 2654435761U + 29U;
  }
  auto source = context.upload(std::as_bytes(std::span(source_vals)));
  auto middle = context.make_buffer(kBytes);
  auto output = context.make_buffer(kBytes);
  auto oracle_middle = context.make_buffer(kBytes);
  auto oracle_output = context.make_buffer(kBytes);

  auto params = context.make_parameter_table({
      .key = "mivr18_21.params",
      .byte_size = sizeof(CanaryParams),
      .fields = {{.key = "step",
                  .offset = 0,
                  .size = sizeof(CanaryParams),
                  .alignment = 4}},
  });

  const auto threads_a = std::max<std::uint64_t>(
      1, context.pipeline_info("b4.0.icb.stage_a.pipeline")
             .thread_execution_width);
  const auto threads_b = std::max<std::uint64_t>(
      1, context.pipeline_info("b4.0.icb.stage_b.pipeline")
             .thread_execution_width);
  const auto plan = context.prepare_command_plan({
      .key = "mivr18_21.two_stage",
      .icb_qualified = true,
      .dispatches =
          {
              {
                  .pipeline_key = "b4.0.icb.stage_a.pipeline",
                  .kind = MetalDispatchKind::threads,
                  .barrier_before = false,
                  .grid = {.width = kCount, .height = 1, .depth = 1},
                  .threads_per_threadgroup = {.width = threads_a,
                                              .height = 1,
                                              .depth = 1},
                  .resources = {{.argument_index = 0,
                                 .resource_slot = 0,
                                 .minimum_bytes = kBytes},
                                {.argument_index = 1,
                                 .resource_slot = 1,
                                 .minimum_bytes = kBytes}},
                  .dynamic_arguments = {{.argument_index = 2,
                                         .parameter_offset = 0,
                                         .minimum_bytes = sizeof(CanaryParams)}},
              },
              {
                  .pipeline_key = "b4.0.icb.stage_b.pipeline",
                  .kind = MetalDispatchKind::threads,
                  .barrier_before = true,
                  .grid = {.width = kCount, .height = 1, .depth = 1},
                  .threads_per_threadgroup = {.width = threads_b,
                                              .height = 1,
                                              .depth = 1},
                  .resources = {{.argument_index = 0,
                                 .resource_slot = 1,
                                 .minimum_bytes = kBytes},
                                {.argument_index = 1,
                                 .resource_slot = 2,
                                 .minimum_bytes = kBytes}},
                  .dynamic_arguments = {{.argument_index = 2,
                                         .parameter_offset = 0,
                                         .minimum_bytes = sizeof(CanaryParams)}},
              },
          },
      .ranges = {{.key = "full", .location = 0, .length = 2}},
  });

  const auto run_oracle = [&](const CanaryParams &p) {
    params.write("step", std::as_bytes(std::span(&p, 1)));
    static_cast<void>(context.execute(
        plan,
        std::array{MetalBufferBinding{0, &source, 0},
                   MetalBufferBinding{1, &oracle_middle, 0},
                   MetalBufferBinding{2, &oracle_output, 0}},
        &params));
    return oracle_output.read(0, kBytes);
  };

  // MIVR-18: submit_barrier() must synchronize in-flight replay_icb_async work.
  {
    auto prof = context.prepare_icb_profile(
        "mivr18.prof",
        {.plan = &plan,
         .bindings = {{0, &source, 0}, {1, &middle, 0}, {2, &output, 0}},
         .parameters = &params});
    const CanaryParams p{.addend = 19U,
                         .multiplier = 5U,
                         .xor_mask = 0x13579BDFU,
                         .element_count = kCount};
    const auto expected = run_oracle(p);
    middle.zero();
    output.zero();
    auto pending = context.replay_icb_async(prof);
    static_cast<void>(context.submit_barrier());
    expect(output.read(0, kBytes) == expected,
           "MIVR-18 submit_barrier() returned before async ICB work completed");
    static_cast<void>(context.wait_submission(std::move(pending)));
    expect(prof.stats().replay_count == 1,
           "MIVR-18 replay_count mismatch after wait_submission");
  }

  // MIVR-21 (move): moving MetalIcbProfile while replay_icb_async is in flight
  // must keep submission valid and update the moved profile's stats safely.
  {
    auto prof = context.prepare_icb_profile(
        "mivr21.move.prof",
        {.plan = &plan,
         .bindings = {{0, &source, 0}, {1, &middle, 0}, {2, &output, 0}},
         .parameters = &params});
    const CanaryParams p{.addend = 37U,
                         .multiplier = 7U,
                         .xor_mask = 0x2468ACE0U,
                         .element_count = kCount};
    const auto expected = run_oracle(p);
    middle.zero();
    output.zero();
    auto pending = context.replay_icb_async(prof);
    MetalIcbProfile moved_prof = std::move(prof);
    expect(!prof.valid() && moved_prof.valid(),
           "MIVR-21 MetalIcbProfile move state invalid");
    static_cast<void>(context.wait_submission(std::move(pending)));
    expect(output.read(0, kBytes) == expected &&
               moved_prof.stats().replay_count == 1,
           "MIVR-21 wait_submission failed after moving MetalIcbProfile");
  }

  // MIVR-21 (destroy): destroying MetalIcbProfile before wait_submission must
  // keep underlying ICB/residency resources alive without UAF.
  {
    const CanaryParams p{.addend = 53U,
                         .multiplier = 11U,
                         .xor_mask = 0xCAFEBABEU,
                         .element_count = kCount};
    const auto expected = run_oracle(p);
    middle.zero();
    output.zero();
    MetalPendingSubmission pending;
    {
      auto ephemeral_prof = context.prepare_icb_profile(
          "mivr21.destroy.prof",
          {.plan = &plan,
           .bindings = {{0, &source, 0}, {1, &middle, 0}, {2, &output, 0}},
           .parameters = &params});
      pending = context.replay_icb_async(ephemeral_prof);
    }
    expect(pending.valid(), "MIVR-21 pending submission invalidated on profile destroy");
    static_cast<void>(context.wait_submission(std::move(pending)));
    expect(output.read(0, kBytes) == expected,
           "MIVR-21 wait_submission output mismatch after profile destruction");
  }
}

} // namespace

int main() {
  try {
    test_replay_and_fail_closed_fallbacks();
    test_replay_icb_batch_barriers_and_stats_locking();
    test_icb_rejects_threadgroups_indirect();
    test_async_icb_barrier_and_profile_lifetime();
    std::cout << "B4.0 compute ICB canary tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B4.0 compute ICB canary failure: " << error.what() << '\n';
    return 1;
  }
}
