#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "omlx_m5/metal.h"

#ifndef OMLX_M5_ICB_CANARY_METALLIB_PATH
#error "OMLX_M5_ICB_CANARY_METALLIB_PATH is required"
#endif

namespace {

struct KernelParameters {
  std::uint32_t addend = 0;
  std::uint32_t multiplier = 1;
  std::uint32_t xor_mask = 0;
  std::uint32_t element_count = 0;
};

static_assert(sizeof(KernelParameters) == 16);

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Exception, typename Function>
void expect_throws(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

template <typename T> std::span<const std::byte> byte_span(const T &value) {
  return std::as_bytes(std::span(&value, 1));
}

template <typename T>
std::span<const std::byte> byte_span(const std::vector<T> &values) {
  return std::as_bytes(std::span(values));
}

omlx::m5::MetalParameterTableDescriptor parameter_descriptor() {
  return {
      .key = "b4.1.decode.parameters",
      .byte_size = 256,
      .fields =
          {
              {.key = "token_id", .offset = 0, .size = 4, .alignment = 4},
              {.key = "position", .offset = 4, .size = 4, .alignment = 4},
              {.key = "kv_frontier", .offset = 8, .size = 4, .alignment = 4},
              {.key = "page_count", .offset = 12, .size = 4, .alignment = 4},
              {.key = "page_table", .offset = 64, .size = 32, .alignment = 16},
              {.key = "route_indices",
               .offset = 96,
               .size = 32,
               .alignment = 16},
              {.key = "route_scores",
               .offset = 128,
               .size = 32,
               .alignment = 16},
              {.key = "valid_mask", .offset = 160, .size = 16, .alignment = 16},
              {.key = "sampling", .offset = 176, .size = 16, .alignment = 16},
              {.key = "kernel_params",
               .offset = 192,
               .size = 16,
               .alignment = 16},
          },
  };
}

void initialize_table(omlx::m5::MetalParameterTable &table,
                      std::uint32_t element_count) {
  const std::uint32_t token = 17;
  const std::uint32_t position = 31;
  const std::uint32_t frontier = 64;
  const std::uint32_t page_count = 8;
  const std::array<std::uint32_t, 8> pages = {0, 1, 2, 3, 4, 5, 6, 7};
  const std::array<std::uint32_t, 8> routes = {7, 6, 5, 4, 3, 2, 1, 0};
  const std::array<float, 8> scores = {0.3F,  0.2F,  0.15F, 0.1F,
                                       0.08F, 0.07F, 0.06F, 0.04F};
  const std::array<std::uint32_t, 4> mask = {1, 0, 0, 0};
  const std::array<float, 4> sampling = {0.9F, 0.8F, 1.0F, 64.0F};
  const KernelParameters params{.element_count = element_count};
  table.write("token_id", byte_span(token));
  table.write("position", byte_span(position));
  table.write("kv_frontier", byte_span(frontier));
  table.write("page_count", byte_span(page_count));
  table.write("page_table", std::as_bytes(std::span(pages)));
  table.write("route_indices", std::as_bytes(std::span(routes)));
  table.write("route_scores", std::as_bytes(std::span(scores)));
  table.write("valid_mask", std::as_bytes(std::span(mask)));
  table.write("sampling", std::as_bytes(std::span(sampling)));
  table.write("kernel_params", byte_span(params));
}

void test_parameter_and_range_abi() {
  constexpr std::uint32_t element_count = 257;
  auto context = omlx::m5::MetalContext::create();
  context.register_library({
      .key = "b4.1.parameter.library",
      .path = OMLX_M5_ICB_CANARY_METALLIB_PATH,
  });
  const auto stage_a = context.warmup_pipeline({
      .key = "b4.1.parameter.stage_a",
      .library_key = "b4.1.parameter.library",
      .function_name = "omlx_m5_icb_stage_a",
      .variant = "icb-ready",
      .support_indirect_command_buffers = true,
  });
  const auto stage_b = context.warmup_pipeline({
      .key = "b4.1.parameter.stage_b",
      .library_key = "b4.1.parameter.library",
      .function_name = "omlx_m5_icb_stage_b",
      .variant = "icb-ready",
      .support_indirect_command_buffers = true,
  });
  const auto non_icb_pipeline = context.warmup_pipeline({
      .key = "b4.1.parameter.non_icb",
      .library_key = "b4.1.parameter.library",
      .function_name = "omlx_m5_icb_stage_a",
      .variant = "direct-only",
  });
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(context.prepare_command_plan({
            .key = "b4.1.invalid.icb.pipeline",
            .icb_qualified = true,
            .dispatches = {{
                .pipeline_key = non_icb_pipeline.descriptor.key,
                .grid = {.width = element_count},
                .threads_per_threadgroup =
                    {.width = non_icb_pipeline.thread_execution_width},
                .resources =
                    {
                        {.argument_index = 0,
                         .resource_slot = 0,
                         .minimum_bytes = element_count * 4ULL},
                        {.argument_index = 1,
                         .resource_slot = 1,
                         .minimum_bytes = element_count * 4ULL},
                    },
                .dynamic_arguments = {{.argument_index = 2,
                                       .parameter_offset = 192,
                                       .minimum_bytes = 16}},
            }},
        }));
      },
      "ICB-qualified plan accepted a direct-only pipeline");
  const auto plan =
      context.prepare_command_plan({
          .key = "b4.1.parameter.plan",
          .icb_qualified = true,
          .dispatches = {{
                             .pipeline_key = stage_a.descriptor.key,
                             .barrier_before = false,
                             .grid = {.width = element_count},
                             .threads_per_threadgroup =
                                 {.width = stage_a.thread_execution_width},
                             .resources =
                                 {
                                     {.argument_index = 0,
                                      .resource_slot = 0,
                                      .minimum_bytes = element_count * 4ULL},
                                     {.argument_index = 1,
                                      .resource_slot = 1,
                                      .minimum_bytes = element_count * 4ULL},
                                 },
                             .dynamic_arguments = {{.argument_index = 2,
                                                    .parameter_offset = 192,
                                                    .minimum_bytes = 16}},
                         },
                         {
                             .pipeline_key = stage_b.descriptor.key,
                             .barrier_before = true,
                             .grid = {.width = element_count},
                             .threads_per_threadgroup =
                                 {.width = stage_b.thread_execution_width},
                             .resources =
                                 {
                                     {.argument_index = 0,
                                      .resource_slot = 1,
                                      .minimum_bytes = element_count * 4ULL},
                                     {.argument_index = 1,
                                      .resource_slot = 2,
                                      .minimum_bytes = element_count * 4ULL},
                                 },
                             .dynamic_arguments = {{.argument_index = 2,
                                                    .parameter_offset = 192,
                                                    .minimum_bytes = 16}},
                         }},
          .ranges =
              {
                  {.key = "full", .location = 0, .length = 2},
                  {.key = "stage_a", .location = 0, .length = 1},
                  {.key = "stage_b", .location = 1, .length = 1},
              },
      });
  expect(plan.icb_qualified() && plan.ranges().size() == 3 &&
             plan.ranges()[0].key == "full" && plan.ranges()[0].length == 2,
         "ICB qualification or command ranges were not frozen");

  auto table = context.make_parameter_table(parameter_descriptor());
  initialize_table(table, element_count);
  expect(table.valid() && table.size() == 256 && table.field_count() == 10,
         "parameter table geometry mismatch");
  const auto stable_address = table.gpu_address();
  const auto buffers_before_updates = context.registry_stats();
  std::vector<std::uint32_t> source_values(element_count);
  for (std::uint32_t index = 0; index < element_count; ++index) {
    source_values[index] = index * 31U + 11U;
  }
  auto source = context.upload(byte_span(source_values));
  auto middle = context.make_buffer(element_count * 4ULL);
  auto output = context.make_buffer(element_count * 4ULL);
  const std::array bindings = {
      omlx::m5::MetalBufferBinding{0, &source, 0},
      omlx::m5::MetalBufferBinding{1, &middle, 0},
      omlx::m5::MetalBufferBinding{2, &output, 0},
  };
  auto residency = context.make_residency_set("b4.1.decode.residency", 8);
  context.residency_add_plan(residency, plan);
  context.residency_add_parameters(residency, table);
  context.residency_add_buffer(residency, source);
  context.residency_add_buffer(residency, middle);
  context.residency_add_buffer(residency, output);
  context.residency_commit(residency);
  expect(residency.valid() && residency.committed() &&
             residency.allocation_count() == 7 &&
             residency.allocated_size() > 0,
         "persistent residency ABI did not freeze all allocations");
  expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(context.execute(plan, bindings)); },
      "dynamic plan accepted a missing parameter table");

  constexpr std::uint32_t iterations = 128;
  for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
    const KernelParameters params{
        .addend = iteration * 17U + 3U,
        .multiplier = (iteration % 31U) * 2U + 1U,
        .xor_mask = 0x9E3779B9U ^ iteration,
        .element_count = element_count,
    };
    table.write("kernel_params", byte_span(params));
    static_cast<void>(context.execute(plan, bindings, &table));
    std::vector<std::uint32_t> expected(element_count);
    for (std::uint32_t index = 0; index < element_count; ++index) {
      expected[index] =
          ((source_values[index] + params.addend) * params.multiplier) ^
          params.xor_mask;
    }
    expect(output.read() == std::vector<std::byte>(byte_span(expected).begin(),
                                                   byte_span(expected).end()),
           "dynamic parameter output mismatch");
  }
  const auto buffers_after_updates = context.registry_stats();
  expect(table.gpu_address() == stable_address &&
             buffers_after_updates.current_buffer_count ==
                 buffers_before_updates.current_buffer_count + 3 &&
             buffers_after_updates.upload_count ==
                 buffers_before_updates.upload_count + 1 &&
             table.generation() == 10 + iterations,
         "dynamic updates changed address, allocated buffers, or uploaded");
  expect(residency.committed() && residency.allocation_count() == 7,
         "dynamic updates changed the persistent residency set");
  expect(table.read("kernel_params").size() == sizeof(KernelParameters),
         "parameter field readback failed");

  auto other_context = omlx::m5::MetalContext::create();
  auto foreign_table =
      other_context.make_parameter_table(parameter_descriptor());
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(context.execute(plan, bindings, &foreign_table));
      },
      "dynamic plan accepted a foreign parameter table");
  auto foreign_buffer = other_context.make_buffer(element_count * 4ULL);
  expect_throws<std::invalid_argument>(
      [&] { context.residency_add_buffer(residency, foreign_buffer); },
      "persistent residency accepted a foreign buffer");

  auto overlapping = parameter_descriptor();
  overlapping.key = "b4.1.overlap";
  overlapping.fields[1].offset = 0;
  expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(context.make_parameter_table(overlapping)); },
      "overlapping parameter fields were accepted");

  auto invalid_range = omlx::m5::MetalCommandPlanDescriptor{
      .key = "b4.1.invalid.range",
      .dispatches = {{
          .pipeline_key = stage_a.descriptor.key,
          .grid = {.width = element_count},
          .threads_per_threadgroup = {.width = stage_a.thread_execution_width},
          .resources =
              {
                  {.argument_index = 0,
                   .resource_slot = 0,
                   .minimum_bytes = element_count * 4ULL},
                  {.argument_index = 1,
                   .resource_slot = 1,
                   .minimum_bytes = element_count * 4ULL},
              },
          .dynamic_arguments = {{.argument_index = 2,
                                 .parameter_offset = 192,
                                 .minimum_bytes = 16}},
      }},
      .ranges = {{.key = "past_end", .location = 1, .length = 1}},
  };
  expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(context.prepare_command_plan(invalid_range)); },
      "out-of-range command range was accepted");
}

} // namespace

int main() {
  try {
    test_parameter_and_range_abi();
    std::cout << "B4.1 dynamic parameter/range ABI tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B4.1 parameter ABI failure: " << error.what() << '\n';
    return 1;
  }
}
