#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/capability.h"
#include "omlx_m5/metal.h"

namespace {

using Json = nlohmann::json;

std::vector<std::byte> constant_u32(std::uint32_t value) {
  std::vector<std::byte> result(sizeof(value));
  std::memcpy(result.data(), &value, sizeof(value));
  return result;
}

Json stats_json(const omlx::m5::MetalRegistryStats &stats) {
  return {
      {"libraries_loaded", stats.libraries_loaded},
      {"library_cache_hits", stats.library_cache_hits},
      {"library_load_failures", stats.library_load_failures},
      {"pipelines_compiled", stats.pipelines_compiled},
      {"pipeline_cache_hits", stats.pipeline_cache_hits},
      {"pipeline_compile_failures", stats.pipeline_compile_failures},
      {"command_plans_built", stats.command_plans_built},
      {"command_plan_cache_hits", stats.command_plan_cache_hits},
      {"command_buffers_submitted", stats.command_buffers_submitted},
      {"dispatches_submitted", stats.dispatches_submitted},
      {"metal4_command_buffers_submitted",
       stats.metal4_command_buffers_submitted},
      {"classic_command_buffers_submitted",
       stats.classic_command_buffers_submitted},
      {"metal4_argument_tables_created", stats.metal4_argument_tables_created},
      {"stable_parameter_buffers_created",
       stats.stable_parameter_buffers_created},
      {"stable_parameter_bytes", stats.stable_parameter_bytes},
      {"dynamic_parameter_tables_created",
       stats.dynamic_parameter_tables_created},
      {"dynamic_parameter_bytes", stats.dynamic_parameter_bytes},
      {"dynamic_parameter_updates", stats.dynamic_parameter_updates},
      {"persistent_residency_sets_created",
       stats.persistent_residency_sets_created},
      {"persistent_residency_commits", stats.persistent_residency_commits},
      {"persistent_residency_allocations",
       stats.persistent_residency_allocations},
      {"submission_failures", stats.submission_failures},
      {"last_shared_event_value", stats.last_shared_event_value},
      {"current_buffer_count", stats.current_buffer_count},
      {"current_buffer_bytes", stats.current_buffer_bytes},
      {"peak_buffer_bytes", stats.peak_buffer_bytes},
      {"current_heap_count", stats.current_heap_count},
      {"current_heap_bytes", stats.current_heap_bytes},
      {"peak_heap_bytes", stats.peak_heap_bytes},
      {"upload_count", stats.upload_count},
      {"uploaded_bytes", stats.uploaded_bytes},
      {"warmup_sealed", stats.warmup_sealed},
  };
}

} // namespace

int main(int argc, const char *const *argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_runtime CLASSIC_METALLIB");
    }
    constexpr std::uint32_t element_count = 1024;
    constexpr std::uint64_t payload_bytes =
        element_count * sizeof(std::uint32_t);
    std::vector<std::uint32_t> values(element_count);
    for (std::uint32_t index = 0; index < element_count; ++index) {
      values[index] = index ^ 0xA5A55A5AU;
    }
    const auto source_bytes = std::as_bytes(std::span(values));

    auto context = omlx::m5::MetalContext::create();
    context.register_library({
        .key = "classic.smoke",
        .path = std::filesystem::path(argv[1]),
    });
    context.register_library({
        .key = "classic.smoke",
        .path = std::filesystem::path(argv[1]),
    });
    const auto pipeline = context.warmup_pipeline({
        .key = "classic.copy_u32.default",
        .library_key = "classic.smoke",
        .function_name = "omlx_m5_copy_u32",
        .variant = "default",
    });
    static_cast<void>(context.warmup_pipeline({
        .key = "classic.copy_u32.default",
        .library_key = "classic.smoke",
        .function_name = "omlx_m5_copy_u32",
        .variant = "default",
    }));
    const omlx::m5::MetalCommandPlanDescriptor plan_descriptor{
        .key = "b1_1.copy_u32.1024",
        .dispatches = {{
            .pipeline_key = "classic.copy_u32.default",
            .grid = {.width = element_count, .height = 1, .depth = 1},
            .threads_per_threadgroup = {.width =
                                            pipeline.thread_execution_width,
                                        .height = 1,
                                        .depth = 1},
            .resources =
                {
                    {.argument_index = 0,
                     .resource_slot = 0,
                     .minimum_bytes = payload_bytes},
                    {.argument_index = 1,
                     .resource_slot = 1,
                     .minimum_bytes = payload_bytes},
                },
            .constant_arguments = {{.argument_index = 2,
                                    .bytes = constant_u32(element_count)}},
        }},
    };
    const auto plan = context.prepare_command_plan(plan_descriptor);
    static_cast<void>(context.prepare_command_plan(plan_descriptor));
    auto source = context.upload(source_bytes);
    auto destination = context.make_buffer(payload_bytes);
    const std::array bindings = {
        omlx::m5::MetalBufferBinding{.resource_slot = 0, .buffer = &source},
        omlx::m5::MetalBufferBinding{.resource_slot = 1,
                                     .buffer = &destination},
    };
    context.seal_warmup();
    const auto submission = context.execute(plan, bindings);
    const auto copied =
        destination.read() ==
        std::vector<std::byte>(source_bytes.begin(), source_bytes.end());
    if (!copied) {
      throw std::runtime_error("runtime inspection copy verification failed");
    }

    const auto &capabilities = context.capabilities();
    const auto &probe = capabilities.profile.probe;
    Json binding_records = Json::array();
    for (const auto &binding : pipeline.bindings) {
      binding_records.push_back({
          {"index", binding.index},
          {"name", binding.name},
          {"kind", binding.kind},
          {"access", binding.access},
          {"alignment", binding.alignment},
          {"data_size", binding.data_size},
      });
    }
    const Json output = {
        {"schema", "omlx-m5-b1-1-runtime-v1"},
        {"evidence_scope", "M4 runtime correctness; not M5 Ultra performance"},
        {"hardware",
         {
             {"device_name", capabilities.device_name},
             {"architecture_name", capabilities.architecture_name},
             {"apple_gpu_family", probe.apple_gpu_family},
             {"max_buffer_length", capabilities.max_buffer_length},
             {"recommended_working_set_bytes",
              capabilities.recommended_working_set_bytes},
         }},
        {"software",
         {
             {"runtime_os", capabilities.runtime_os},
             {"sdk_max_allowed", probe.sdk_max_allowed},
         }},
        {"capability",
         {
             {"selected_route",
              omlx::m5::to_string(capabilities.profile.selected_route)},
             {"native_tensorops_eligible",
              capabilities.profile.native_tensorops_eligible},
             {"classic_msl_eligible",
              capabilities.profile.classic_msl_eligible},
             {"mlx_bridge_eligible", capabilities.profile.mlx_bridge_eligible},
             {"classic_command_queue", probe.classic_command_queue},
             {"shared_event", probe.shared_event},
             {"metal4_headers", probe.metal4_headers},
             {"metal4_command_queue", probe.metal4_command_queue},
             {"metal4_command_allocator", probe.metal4_command_allocator},
             {"metal4_command_buffer", probe.metal4_command_buffer},
             {"metal4_compute_encoder", probe.metal4_compute_encoder},
             {"metal4_compiler", probe.metal4_compiler},
             {"metal4_argument_table", probe.metal4_argument_table},
             {"compute_icb_api", probe.compute_icb_api},
             {"compute_icb_barrier", probe.compute_icb_barrier},
             {"residency_sets", probe.residency_sets},
             {"compute_icb_eligible",
              capabilities.profile.compute_icb_eligible},
             {"metal4_icb_eligible", capabilities.profile.metal4_icb_eligible},
             {"metal4_direct_eligible",
              capabilities.profile.metal4_direct_eligible},
             {"metal4_fallback_reasons",
              capabilities.profile.metal4_fallback_reasons},
             {"icb_fallback_reasons",
              capabilities.profile.icb_fallback_reasons},
             {"tensor_api", probe.tensor_api},
             {"native_tensorops_runtime", probe.native_tensorops_runtime},
             {"native_tensorops_source", probe.native_tensorops_source},
             {"fallback_reasons", capabilities.profile.fallback_reasons},
         }},
        {"pipeline",
         {
             {"key", pipeline.descriptor.key},
             {"function", pipeline.descriptor.function_name},
             {"thread_execution_width", pipeline.thread_execution_width},
             {"max_threads_per_threadgroup",
              pipeline.max_threads_per_threadgroup},
             {"static_threadgroup_memory_bytes",
              pipeline.static_threadgroup_memory_bytes},
             {"metal4_pipeline_enabled", pipeline.metal4_pipeline_enabled},
             {"metal4_icb_enabled",
              pipeline.metal4_indirect_command_buffers_enabled},
             {"bindings", binding_records},
         }},
        {"command_plan",
         {
             {"key", plan.key()},
             {"dispatch_count", plan.dispatch_count()},
             {"resource_slot_count", plan.resource_slot_count()},
             {"copy_verified", copied},
             {"shared_event_value", submission.shared_event_value},
         }},
        {"registry", stats_json(context.registry_stats())},
    };
    std::cout << output.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "omlx_m5_inspect_runtime: " << error.what() << '\n';
    return 1;
  }
}
