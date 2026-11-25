#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/metal.h"

#ifndef OMLX_M5_ICB_CANARY_METALLIB_PATH
#error "OMLX_M5_ICB_CANARY_METALLIB_PATH is required"
#endif

namespace {

using Json = nlohmann::json;

struct KernelParameters {
  std::uint32_t addend = 0;
  std::uint32_t multiplier = 1;
  std::uint32_t xor_mask = 0;
  std::uint32_t element_count = 0;
};

static_assert(sizeof(KernelParameters) == 16);

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

} // namespace

int main(int argc, const char *const *argv) {
  try {
    const std::uint32_t iterations =
        argc == 2 ? static_cast<std::uint32_t>(std::stoul(argv[1])) : 10'000;
    if (argc > 2 || iterations == 0) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_parameter_abi [ITERATIONS]");
    }
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
    const auto descriptor = parameter_descriptor();
    auto table = context.make_parameter_table(descriptor);
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

    const auto address_before = table.gpu_address();
    const auto registry_before = context.registry_stats();
    bool bit_exact = true;
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
      const auto expected_bytes = byte_span(expected);
      bit_exact = bit_exact && output.read() == std::vector<std::byte>(
                                                    expected_bytes.begin(),
                                                    expected_bytes.end());
    }
    const auto registry_after = context.registry_stats();

    Json fields = Json::array();
    for (const auto &field : descriptor.fields) {
      fields.push_back({
          {"key", field.key},
          {"offset", field.offset},
          {"size", field.size},
          {"alignment", field.alignment},
      });
    }
    Json ranges = Json::array();
    for (const auto &range : plan.ranges()) {
      ranges.push_back({
          {"key", range.key},
          {"location", range.location},
          {"length", range.length},
      });
    }
    const Json result = {
        {"schema", "omlx-m5-b4-1-parameter-abi-v1"},
        {"evidence_scope",
         "M4 macOS 27 Metal 4 dynamic-parameter correctness; not M5 "
         "performance"},
        {"iterations", iterations},
        {"bit_exact", bit_exact},
        {"table",
         {
             {"key", table.key()},
             {"byte_size", table.size()},
             {"field_count", table.field_count()},
             {"fields", fields},
             {"generation", table.generation()},
             {"address_stable", table.gpu_address() == address_before},
         }},
        {"plan",
         {
             {"key", plan.key()},
             {"icb_qualified", plan.icb_qualified()},
             {"dispatch_count", plan.dispatch_count()},
             {"ranges", ranges},
         }},
        {"residency",
         {
             {"key", residency.key()},
             {"committed", residency.committed()},
             {"allocation_count", residency.allocation_count()},
             {"allocated_size", residency.allocated_size()},
         }},
        {"registry",
         {
             {"dynamic_parameter_tables_created",
              registry_after.dynamic_parameter_tables_created},
             {"dynamic_parameter_bytes",
              registry_after.dynamic_parameter_bytes},
             {"persistent_residency_sets_created",
              registry_after.persistent_residency_sets_created},
             {"persistent_residency_commits",
              registry_after.persistent_residency_commits},
             {"persistent_residency_allocations",
              registry_after.persistent_residency_allocations},
         }},
        {"registry_delta",
         {
             {"buffer_count", registry_after.current_buffer_count -
                                  registry_before.current_buffer_count},
             {"upload_count",
              registry_after.upload_count - registry_before.upload_count},
             {"dynamic_updates", registry_after.dynamic_parameter_updates -
                                     registry_before.dynamic_parameter_updates},
             {"command_buffers", registry_after.command_buffers_submitted -
                                     registry_before.command_buffers_submitted},
             {"metal4_command_buffers",
              registry_after.metal4_command_buffers_submitted -
                  registry_before.metal4_command_buffers_submitted},
             {"classic_command_buffers",
              registry_after.classic_command_buffers_submitted -
                  registry_before.classic_command_buffers_submitted},
             {"argument_tables",
              registry_after.metal4_argument_tables_created -
                  registry_before.metal4_argument_tables_created},
         }},
    };
    if (!bit_exact) {
      throw std::runtime_error("dynamic parameter output mismatch");
    }
    std::cout << result.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "parameter ABI inspection failed: " << error.what() << '\n';
    return 1;
  }
}
