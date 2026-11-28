#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/capability.h"

namespace omlx::m5 {

struct MetalCapabilities {
  std::string device_name;
  std::string architecture_name;
  std::string runtime_os;
  std::uint64_t max_buffer_length = 0;
  std::uint64_t recommended_working_set_bytes = 0;
  bool metal4_nax_runtime = false;
  std::string nax_capability_source = "unverified";
  MetalCapabilityProfile profile;
};

struct GpuTiming {
  double wall_ms = 0.0;
  double gpu_ms = 0.0;
};

struct MetalContextOptions {
  std::optional<bool> native_tensorops_runtime_override;
  bool mlx_bridge_available = false;
};

struct MetalLibraryDescriptor {
  std::string key;
  std::filesystem::path path;
  std::string expected_sha256;
};

struct MetalPipelineDescriptor {
  std::string key;
  std::string library_key;
  std::string function_name;
  std::string variant = "default";
  bool support_indirect_command_buffers = false;
};

struct MetalPipelineBindingInfo {
  std::uint32_t index = 0;
  std::string name;
  std::string kind;
  std::string access;
  std::uint64_t alignment = 0;
  std::uint64_t data_size = 0;
};

struct MetalPipelineInfo {
  MetalPipelineDescriptor descriptor;
  std::uint64_t thread_execution_width = 0;
  std::uint64_t max_threads_per_threadgroup = 0;
  std::uint64_t static_threadgroup_memory_bytes = 0;
  bool indirect_command_buffers_enabled = false;
  bool metal4_pipeline_enabled = false;
  bool metal4_indirect_command_buffers_enabled = false;
  std::vector<MetalPipelineBindingInfo> bindings;
};

struct MetalDispatchSize {
  std::uint64_t width = 1;
  std::uint64_t height = 1;
  std::uint64_t depth = 1;
};

struct MetalResourceArgument {
  std::uint32_t argument_index = 0;
  std::uint32_t resource_slot = 0;
  std::uint64_t minimum_bytes = 0;
};

struct MetalConstantArgument {
  std::uint32_t argument_index = 0;
  std::vector<std::byte> bytes;
};

struct MetalDynamicArgument {
  std::uint32_t argument_index = 0;
  std::uint64_t parameter_offset = 0;
  std::uint64_t minimum_bytes = 0;
};

enum class MetalDispatchKind : std::uint8_t {
  threads,
  threadgroups,
  threadgroups_indirect,
};

struct MetalDispatchDescriptor {
  std::string pipeline_key;
  MetalDispatchKind kind = MetalDispatchKind::threads;
  bool barrier_before = true;
  MetalDispatchSize grid;
  MetalDispatchSize threads_per_threadgroup;
  std::vector<MetalResourceArgument> resources;
  std::vector<MetalConstantArgument> constant_arguments;
  std::vector<MetalDynamicArgument> dynamic_arguments;
  std::uint32_t indirect_grid_resource_slot = 0;
  std::uint64_t indirect_grid_offset = 0;
};

struct MetalCommandRangeDescriptor {
  std::string key;
  std::uint32_t location = 0;
  std::uint32_t length = 0;
};

struct MetalCommandPlanDescriptor {
  std::string key;
  bool icb_qualified = false;
  std::vector<MetalDispatchDescriptor> dispatches;
  std::vector<MetalCommandRangeDescriptor> ranges;
};

struct MetalParameterFieldDescriptor {
  std::string key;
  std::uint64_t offset = 0;
  std::uint64_t size = 0;
  std::uint64_t alignment = 1;
};

struct MetalParameterTableDescriptor {
  std::string key;
  std::uint64_t byte_size = 0;
  std::vector<MetalParameterFieldDescriptor> fields;
};

struct MetalRegistryStats {
  std::uint64_t libraries_loaded = 0;
  std::uint64_t library_cache_hits = 0;
  std::uint64_t library_load_failures = 0;
  std::uint64_t pipelines_compiled = 0;
  std::uint64_t pipeline_cache_hits = 0;
  std::uint64_t pipeline_compile_failures = 0;
  std::uint64_t command_plans_built = 0;
  std::uint64_t command_plan_cache_hits = 0;
  std::uint64_t command_buffers_submitted = 0;
  std::uint64_t dispatches_submitted = 0;
  std::uint64_t metal4_command_buffers_submitted = 0;
  std::uint64_t classic_command_buffers_submitted = 0;
  std::uint64_t metal4_argument_tables_created = 0;
  std::uint64_t metal4_argument_table_allocations = 0;
  std::uint64_t metal4_argument_table_pool_hits = 0;
  std::uint64_t stable_parameter_buffers_created = 0;
  std::uint64_t stable_parameter_bytes = 0;
  std::uint64_t dynamic_parameter_tables_created = 0;
  std::uint64_t dynamic_parameter_bytes = 0;
  std::uint64_t dynamic_parameter_updates = 0;
  std::uint64_t persistent_residency_sets_created = 0;
  std::uint64_t persistent_residency_commits = 0;
  std::uint64_t persistent_residency_allocations = 0;
  std::uint64_t icb_profiles_created = 0;
  std::uint64_t icb_profile_recordings = 0;
  std::uint64_t icb_profile_replays = 0;
  std::uint64_t submission_failures = 0;
  std::uint64_t last_shared_event_value = 0;
  std::uint64_t current_buffer_count = 0;
  std::uint64_t current_buffer_bytes = 0;
  std::uint64_t peak_buffer_bytes = 0;
  std::uint64_t current_heap_count = 0;
  std::uint64_t current_heap_bytes = 0;
  std::uint64_t peak_heap_bytes = 0;
  std::uint64_t upload_count = 0;
  std::uint64_t uploaded_bytes = 0;
  bool warmup_sealed = false;
};

struct MetalSubmission {
  GpuTiming timing;
  std::uint64_t shared_event_value = 0;
  std::uint64_t dispatch_count = 0;
};

// In-flight asynchronous command-buffer submission handle for 2-deep pure-AR
// command queue pipelining. Destructor safely drains any unawaited GPU work.
class MetalPendingSubmission final {
public:
  MetalPendingSubmission();
  MetalPendingSubmission(MetalPendingSubmission &&) noexcept;
  MetalPendingSubmission &operator=(MetalPendingSubmission &&) noexcept;
  MetalPendingSubmission(const MetalPendingSubmission &) = delete;
  MetalPendingSubmission &operator=(const MetalPendingSubmission &) = delete;
  ~MetalPendingSubmission();

  [[nodiscard]] bool valid() const noexcept;

private:
  struct Impl;
  friend class MetalContext;

  explicit MetalPendingSubmission(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

struct MetalIcbCanaryOptions {
  std::uint64_t iterations = 10'000;
  std::uint32_t element_count = 4096;
  bool prefer_metal4_outer = true;
  bool force_execute_batch_fallback = false;
  bool omit_dependency_barrier = false;
};

struct MetalIcbCanaryReport {
  std::string execution_route;
  std::string fallback_reason;
  bool compute_icb_eligible = false;
  bool metal4_icb_eligible = false;
  bool pipeline_icb_enabled = false;
  bool metal4_pipeline_icb_enabled = false;
  bool dependency_barrier_encoded = false;
  bool bit_exact = false;
  bool changing_parameters_observed = false;
  bool resources_released = false;
  std::uint64_t iterations = 0;
  std::uint64_t parameter_updates = 0;
  std::uint64_t compared_iterations = 0;
  std::uint64_t icb_recordings = 0;
  std::uint64_t icb_resets = 0;
  std::uint64_t icb_execute_calls = 0;
  std::uint64_t token_loop_icb_recordings = 0;
  std::uint64_t replay_command_buffer_commits = 0;
  std::uint64_t replay_dispatches = 0;
  std::uint64_t oracle_execute_batch_commits = 0;
  std::uint64_t oracle_dispatches = 0;
  std::uint64_t output_checksum = 0;
  double icb_record_cpu_ms = 0.0;
  double replay_cpu_encode_ms = 0.0;
  double replay_wall_ms = 0.0;
  double replay_gpu_ms = 0.0;
  double oracle_cpu_encode_ms = 0.0;
  double oracle_wall_ms = 0.0;
  double oracle_gpu_ms = 0.0;
};

class MetalBuffer final {
public:
  MetalBuffer();
  MetalBuffer(MetalBuffer &&) noexcept;
  MetalBuffer &operator=(MetalBuffer &&) noexcept;
  MetalBuffer(const MetalBuffer &) = delete;
  MetalBuffer &operator=(const MetalBuffer &) = delete;
  ~MetalBuffer();

  [[nodiscard]] std::uint64_t size() const;
  [[nodiscard]] std::uint64_t gpu_address() const;
  [[nodiscard]] void *native_mtl_buffer() const noexcept;
  [[nodiscard]] std::vector<std::byte> read(std::uint64_t offset = 0,
                                            std::uint64_t length = 0) const;
  [[nodiscard]] std::span<const std::byte> view(std::uint64_t offset = 0,
                                                std::uint64_t length = 0) const;
  void write(std::span<const std::byte> data, std::uint64_t offset = 0);
  void zero();

private:
  class Impl;
  friend class MetalContext;
  friend class MetalHeap;
  friend class MetalParameterTable;

  explicit MetalBuffer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

struct MetalBufferBinding {
  std::uint32_t resource_slot = 0;
  const MetalBuffer *buffer = nullptr;
  std::uint64_t offset = 0;
};

class MetalParameterTable final {
public:
  MetalParameterTable();
  MetalParameterTable(MetalParameterTable &&) noexcept;
  MetalParameterTable &operator=(MetalParameterTable &&) noexcept;
  MetalParameterTable(const MetalParameterTable &) = delete;
  MetalParameterTable &operator=(const MetalParameterTable &) = delete;
  ~MetalParameterTable();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const std::string &key() const;
  [[nodiscard]] std::uint64_t size() const;
  [[nodiscard]] std::uint64_t gpu_address() const;
  [[nodiscard]] std::uint64_t generation() const;
  [[nodiscard]] std::uint64_t field_count() const;
  void write(std::string_view field, std::span<const std::byte> data);
  [[nodiscard]] std::vector<std::byte> read(std::string_view field) const;

private:
  class Impl;
  friend class MetalContext;
  friend class MetalIcbProfile;

  explicit MetalParameterTable(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

class MetalResidencySet final {
public:
  MetalResidencySet();
  MetalResidencySet(MetalResidencySet &&) noexcept;
  MetalResidencySet &operator=(MetalResidencySet &&) noexcept;
  MetalResidencySet(const MetalResidencySet &) = delete;
  MetalResidencySet &operator=(const MetalResidencySet &) = delete;
  ~MetalResidencySet();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const std::string &key() const;
  [[nodiscard]] std::uint64_t allocation_count() const;
  [[nodiscard]] std::uint64_t allocated_size() const;
  [[nodiscard]] bool committed() const;

private:
  class Impl;
  friend class MetalContext;

  explicit MetalResidencySet(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

class MetalHeap final {
public:
  MetalHeap();
  MetalHeap(MetalHeap &&) noexcept;
  MetalHeap &operator=(MetalHeap &&) noexcept;
  MetalHeap(const MetalHeap &) = delete;
  MetalHeap &operator=(const MetalHeap &) = delete;
  ~MetalHeap();

  [[nodiscard]] std::uint64_t size() const;
  [[nodiscard]] MetalBuffer make_buffer(std::uint64_t length) const;

private:
  class Impl;
  friend class MetalContext;

  explicit MetalHeap(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

class MetalCommandPlan final {
public:
  class Impl;

  MetalCommandPlan();
  MetalCommandPlan(const MetalCommandPlan &) noexcept;
  MetalCommandPlan &operator=(const MetalCommandPlan &) noexcept;
  MetalCommandPlan(MetalCommandPlan &&) noexcept;
  MetalCommandPlan &operator=(MetalCommandPlan &&) noexcept;
  ~MetalCommandPlan();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const std::string &key() const;
  [[nodiscard]] std::uint64_t dispatch_count() const;
  [[nodiscard]] std::uint64_t barrier_count() const;
  [[nodiscard]] std::uint64_t resource_slot_count() const;
  [[nodiscard]] bool icb_qualified() const;
  [[nodiscard]] const std::vector<MetalCommandRangeDescriptor> &ranges() const;

private:
  friend class MetalContext;

  explicit MetalCommandPlan(std::shared_ptr<const Impl> impl);
  std::shared_ptr<const Impl> impl_;
};

struct MetalCommandInvocation {
  const MetalCommandPlan *plan = nullptr;
  std::vector<MetalBufferBinding> bindings;
  const MetalParameterTable *parameters = nullptr;
};

struct MetalIcbProfileStats {
  std::string execution_route;
  std::uint64_t command_count = 0;
  std::uint64_t recording_count = 0;
  std::uint64_t replay_count = 0;
  std::uint64_t residency_allocation_count = 0;
};

class MetalIcbProfile final {
public:
  MetalIcbProfile();
  MetalIcbProfile(MetalIcbProfile &&) noexcept;
  MetalIcbProfile &operator=(MetalIcbProfile &&) noexcept;
  MetalIcbProfile(const MetalIcbProfile &) = delete;
  MetalIcbProfile &operator=(const MetalIcbProfile &) = delete;
  ~MetalIcbProfile();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const std::string &key() const;
  [[nodiscard]] const std::vector<MetalCommandRangeDescriptor> &ranges() const;
  [[nodiscard]] MetalIcbProfileStats stats() const;

private:
  class Impl;
  friend class MetalContext;
  friend class MetalPendingSubmission;

  explicit MetalIcbProfile(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

class MetalContext final {
public:
  static MetalContext create();
  static MetalContext create(std::optional<bool> nax_runtime_override);
  static MetalContext create(MetalContextOptions options);

  MetalContext(MetalContext &&) noexcept;
  MetalContext &operator=(MetalContext &&) noexcept;
  MetalContext(const MetalContext &) = delete;
  MetalContext &operator=(const MetalContext &) = delete;
  ~MetalContext();

  [[nodiscard]] const MetalCapabilities &capabilities() const;
  [[nodiscard]] std::uint64_t current_allocated_bytes() const;
  // Test-only ownership probe; does not retain the registry or expose data.
  [[nodiscard]] std::weak_ptr<void> diagnostic_owner_lifetime() const;
  [[nodiscard]] MetalBuffer make_buffer(std::uint64_t length) const;
  [[nodiscard]] MetalBuffer make_private_buffer(std::uint64_t length) const;
  [[nodiscard]] MetalBuffer upload(std::span<const std::byte> data) const;
  void upload_into(MetalBuffer &buffer, std::span<const std::byte> data,
                   std::uint64_t offset = 0) const;
  [[nodiscard]] MetalHeap make_heap(std::uint64_t size) const;
  [[nodiscard]] MetalParameterTable
  make_parameter_table(MetalParameterTableDescriptor descriptor) const;
  [[nodiscard]] MetalResidencySet
  make_residency_set(std::string key, std::uint64_t initial_capacity = 0) const;
  void residency_add_buffer(MetalResidencySet &residency,
                            const MetalBuffer &buffer) const;
  void residency_add_parameters(MetalResidencySet &residency,
                                const MetalParameterTable &parameters) const;
  void residency_add_plan(MetalResidencySet &residency,
                          const MetalCommandPlan &plan) const;
  void residency_commit(MetalResidencySet &residency) const;
  void register_library(MetalLibraryDescriptor descriptor) const;
  [[nodiscard]] MetalPipelineInfo
  warmup_pipeline(MetalPipelineDescriptor descriptor) const;
  [[nodiscard]] MetalPipelineInfo
  pipeline_info(const std::string &pipeline_key) const;
  [[nodiscard]] MetalCommandPlan
  prepare_command_plan(MetalCommandPlanDescriptor descriptor) const;
  void seal_warmup() const;
  [[nodiscard]] bool warmup_sealed() const;
  [[nodiscard]] MetalSubmission
  execute(const MetalCommandPlan &plan,
          std::span<const MetalBufferBinding> bindings,
          const MetalParameterTable *parameters = nullptr) const;
  // Encodes every invocation, in order, into one command buffer. Resource
  // dependencies between adjacent plans are preserved by encoder order.
  [[nodiscard]] MetalSubmission
  execute_batch(std::span<const MetalCommandInvocation> invocations) const;
  // Records one ICB-qualified immutable command plan against stable resource
  // and parameter-buffer addresses. Replay changes buffer contents only; it
  // never rebuilds the command list or rebinds resources.
  [[nodiscard]] MetalIcbProfile
  prepare_icb_profile(std::string key,
                      const MetalCommandInvocation &invocation) const;
  [[nodiscard]] MetalSubmission
  replay_icb(MetalIcbProfile &profile,
             std::string_view range_key = "full") const;
  [[nodiscard]] MetalSubmission
  replay_icb_range(MetalIcbProfile &profile, std::uint32_t location,
                   std::uint32_t length) const;
  [[nodiscard]] MetalPendingSubmission
  replay_icb_async(MetalIcbProfile &profile,
                   std::string_view range_key = "full") const;
  [[nodiscard]] MetalSubmission
  wait_submission(MetalPendingSubmission &&pending) const;
  [[nodiscard]] MetalSubmission
  replay_icb_batch(std::span<MetalIcbProfile *const> profiles) const;
  // B4.0 eligibility canary. The two supplied pipelines must opt into ICBs and
  // implement the fixed source->middle->output contract from icb_canary.metal.
  // A missing capability or dependency barrier fails closed to execute_batch.
  [[nodiscard]] MetalIcbCanaryReport
  run_icb_canary(const std::string &stage_a_pipeline_key,
                 const std::string &stage_b_pipeline_key,
                 MetalIcbCanaryOptions options = {}) const;
  [[nodiscard]] MetalRegistryStats registry_stats() const;
  [[nodiscard]] GpuTiming submit_barrier() const;
  [[nodiscard]] GpuTiming submit_classic_barrier() const;
  [[nodiscard]] void *native_mtl_device() const noexcept;
  [[nodiscard]] void *native_mtl_command_queue() const noexcept;
  // Encodes invocations into a caller-owned classic id<MTLCommandBuffer>
  // without committing or waiting (used for Dual-ANE GPU-suffix overlap).
  std::uint64_t encode_into_command_buffer(
      void *mtl_command_buffer,
      std::span<const MetalCommandInvocation> invocations) const;
  // Allocates a command buffer on the classic MTLCommandQueue, encodes all
  // invocations, and commits asynchronously without waiting on the CPU.
  std::uint64_t submit_classic_async(
      std::span<const MetalCommandInvocation> invocations) const;

private:
  class Impl;

  explicit MetalContext(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5
