#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "omlx_m5/metal.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <dispatch/dispatch.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <ranges>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifndef __MAC_OS_X_VERSION_MAX_ALLOWED
#define __MAC_OS_X_VERSION_MAX_ALLOWED 0
#endif

namespace omlx::m5 {

namespace {

std::string ns_string(NSString *value) {
  return value == nil ? std::string{} : std::string([value UTF8String]);
}

NSString *to_ns_string(const std::string &value) {
  NSString *result = [NSString stringWithUTF8String:value.c_str()];
  if (result == nil) {
    throw std::invalid_argument("string is not valid UTF-8");
  }
  return result;
}

void validate_range(std::uint64_t size, std::uint64_t offset,
                    std::uint64_t length) {
  if (offset > size || length > size - offset) {
    throw std::out_of_range("Metal buffer range is out of bounds");
  }
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               const char *label) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error(std::string(label) + " overflow");
  }
  return left * right;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          const char *label) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::overflow_error(std::string(label) + " overflow");
  }
  return left + right;
}

NSUInteger checked_ns_integer(std::uint64_t value, const char *label) {
  if (value > std::numeric_limits<NSUInteger>::max()) {
    throw std::overflow_error(std::string(label) + " exceeds NSUInteger");
  }
  return static_cast<NSUInteger>(value);
}

MTLSize make_mtl_size(const MetalDispatchSize &size, const char *label) {
  if (size.width == 0 || size.height == 0 || size.depth == 0) {
    throw std::invalid_argument(std::string(label) + " must be positive");
  }
  return MTLSizeMake(checked_ns_integer(size.width, label),
                     checked_ns_integer(size.height, label),
                     checked_ns_integer(size.depth, label));
}

bool lower_hex(std::string_view value, std::size_t digits) {
  return value.size() == digits &&
         std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::string sha256_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("unable to open Metal library: " + path.string());
  }
  CC_SHA256_CTX context{};
  if (CC_SHA256_Init(&context) != 1) {
    throw std::runtime_error("unable to initialize Metal library SHA-256");
  }
  std::vector<char> buffer(1024 * 1024);
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0 && CC_SHA256_Update(&context, buffer.data(),
                                      static_cast<CC_LONG>(count)) != 1) {
      throw std::runtime_error("unable to update Metal library SHA-256");
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("unable to read Metal library: " + path.string());
  }
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  if (CC_SHA256_Final(digest.data(), &context) != 1) {
    throw std::runtime_error("unable to finalize Metal library SHA-256");
  }
  constexpr std::string_view hex = "0123456789abcdef";
  std::string output;
  output.reserve(digest.size() * 2);
  for (const auto byte : digest) {
    output.push_back(hex[byte >> 4]);
    output.push_back(hex[byte & 0x0FU]);
  }
  return output;
}

std::string binding_kind(MTLBindingType type) {
  switch (type) {
  case MTLBindingTypeBuffer:
    return "buffer";
  case MTLBindingTypeThreadgroupMemory:
    return "threadgroup_memory";
  case MTLBindingTypeTexture:
    return "texture";
  case MTLBindingTypeSampler:
    return "sampler";
  case MTLBindingTypeImageblockData:
    return "imageblock_data";
  case MTLBindingTypeImageblock:
    return "imageblock";
  case MTLBindingTypeVisibleFunctionTable:
    return "visible_function_table";
  case MTLBindingTypePrimitiveAccelerationStructure:
    return "primitive_acceleration_structure";
  case MTLBindingTypeInstanceAccelerationStructure:
    return "instance_acceleration_structure";
  case MTLBindingTypeIntersectionFunctionTable:
    return "intersection_function_table";
  case MTLBindingTypeObjectPayload:
    return "object_payload";
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
  case MTLBindingTypeTensor:
    return "tensor";
#endif
  }
  return "unknown";
}

std::string binding_access(MTLBindingAccess access) {
  switch (access) {
  case MTLBindingAccessReadOnly:
    return "read";
  case MTLBindingAccessReadWrite:
    return "read_write";
  case MTLBindingAccessWriteOnly:
    return "write";
  }
  return "unknown";
}

bool same_pipeline(const MetalPipelineDescriptor &left,
                   const MetalPipelineDescriptor &right) {
  return left.key == right.key && left.library_key == right.library_key &&
         left.function_name == right.function_name &&
         left.variant == right.variant &&
         left.support_indirect_command_buffers ==
             right.support_indirect_command_buffers;
}

bool same_size(const MetalDispatchSize &left, const MetalDispatchSize &right) {
  return left.width == right.width && left.height == right.height &&
         left.depth == right.depth;
}

bool same_resource(const MetalResourceArgument &left,
                   const MetalResourceArgument &right) {
  return left.argument_index == right.argument_index &&
         left.resource_slot == right.resource_slot &&
         left.minimum_bytes == right.minimum_bytes;
}

bool same_constant(const MetalConstantArgument &left,
                   const MetalConstantArgument &right) {
  return left.argument_index == right.argument_index &&
         left.bytes == right.bytes;
}

bool same_dynamic(const MetalDynamicArgument &left,
                  const MetalDynamicArgument &right) {
  return left.argument_index == right.argument_index &&
         left.parameter_offset == right.parameter_offset &&
         left.minimum_bytes == right.minimum_bytes;
}

bool same_range(const MetalCommandRangeDescriptor &left,
                const MetalCommandRangeDescriptor &right) {
  return left.key == right.key && left.location == right.location &&
         left.length == right.length;
}

bool same_dispatch(const MetalDispatchDescriptor &left,
                   const MetalDispatchDescriptor &right) {
  return left.pipeline_key == right.pipeline_key && left.kind == right.kind &&
         left.barrier_before == right.barrier_before &&
         same_size(left.grid, right.grid) &&
         same_size(left.threads_per_threadgroup,
                   right.threads_per_threadgroup) &&
         (left.kind != MetalDispatchKind::threadgroups_indirect ||
          (left.indirect_grid_resource_slot ==
               right.indirect_grid_resource_slot &&
           left.indirect_grid_offset == right.indirect_grid_offset)) &&
         left.resources.size() == right.resources.size() &&
         left.constant_arguments.size() == right.constant_arguments.size() &&
         left.dynamic_arguments.size() == right.dynamic_arguments.size() &&
         std::ranges::equal(left.resources, right.resources, same_resource) &&
         std::ranges::equal(left.constant_arguments, right.constant_arguments,
                            same_constant) &&
         std::ranges::equal(left.dynamic_arguments, right.dynamic_arguments,
                            same_dynamic);
}

bool same_plan(const MetalCommandPlanDescriptor &left,
               const MetalCommandPlanDescriptor &right) {
  return left.key == right.key && left.icb_qualified == right.icb_qualified &&
         left.dispatches.size() == right.dispatches.size() &&
         left.ranges.size() == right.ranges.size() &&
         std::ranges::equal(left.dispatches, right.dispatches, same_dispatch) &&
         std::ranges::equal(left.ranges, right.ranges, same_range);
}

class ResourceLedger final {
public:
  void add_buffer(std::uint64_t bytes) {
    std::scoped_lock lock(mutex_);
    ++buffer_count_;
    buffer_bytes_ += bytes;
    peak_buffer_bytes_ = std::max(peak_buffer_bytes_, buffer_bytes_);
  }

  void remove_buffer(std::uint64_t bytes) {
    std::scoped_lock lock(mutex_);
    if (buffer_count_ == 0 || bytes > buffer_bytes_) {
      std::terminate();
    }
    --buffer_count_;
    buffer_bytes_ -= bytes;
  }

  void add_heap(std::uint64_t bytes) {
    std::scoped_lock lock(mutex_);
    ++heap_count_;
    heap_bytes_ += bytes;
    peak_heap_bytes_ = std::max(peak_heap_bytes_, heap_bytes_);
  }

  void remove_heap(std::uint64_t bytes) {
    std::scoped_lock lock(mutex_);
    if (heap_count_ == 0 || bytes > heap_bytes_) {
      std::terminate();
    }
    --heap_count_;
    heap_bytes_ -= bytes;
  }

  void record_upload(std::uint64_t bytes) {
    std::scoped_lock lock(mutex_);
    ++upload_count_;
    uploaded_bytes_ += bytes;
  }

  void fill(MetalRegistryStats &stats) const {
    std::scoped_lock lock(mutex_);
    stats.current_buffer_count = buffer_count_;
    stats.current_buffer_bytes = buffer_bytes_;
    stats.peak_buffer_bytes = peak_buffer_bytes_;
    stats.current_heap_count = heap_count_;
    stats.current_heap_bytes = heap_bytes_;
    stats.peak_heap_bytes = peak_heap_bytes_;
    stats.upload_count = upload_count_;
    stats.uploaded_bytes = uploaded_bytes_;
  }

private:
  mutable std::mutex mutex_;
  std::uint64_t buffer_count_ = 0;
  std::uint64_t buffer_bytes_ = 0;
  std::uint64_t peak_buffer_bytes_ = 0;
  std::uint64_t heap_count_ = 0;
  std::uint64_t heap_bytes_ = 0;
  std::uint64_t peak_heap_bytes_ = 0;
  std::uint64_t upload_count_ = 0;
  std::uint64_t uploaded_bytes_ = 0;
};

struct LibraryEntry {
  MetalLibraryDescriptor descriptor;
  std::string sha256;
  id<MTLLibrary> library;
};

struct PipelineEntry {
  MetalPipelineDescriptor descriptor;
  MetalPipelineInfo info;
  NSUInteger max_buffer_bind_count = 0;
  id<MTLComputePipelineState> pipeline;
  id<MTLComputePipelineState> metal4_pipeline;
};

struct IcbCanaryParameters {
  std::uint32_t addend = 0;
  std::uint32_t multiplier = 1;
  std::uint32_t xor_mask = 0;
  std::uint32_t element_count = 0;
};

static_assert(sizeof(IcbCanaryParameters) == 16);

class RegistryState;

} // namespace

class MetalBuffer::Impl {
public:
  Impl(id<MTLBuffer> value, std::shared_ptr<ResourceLedger> selected_ledger)
      : buffer(value), ledger(std::move(selected_ledger)),
        length(buffer == nil ? 0
                             : static_cast<std::uint64_t>([buffer length])) {
    if (buffer == nil || ledger == nullptr) {
      throw std::runtime_error("Metal buffer allocation failed");
    }
    ledger->add_buffer(length);
  }

  ~Impl() { ledger->remove_buffer(length); }

  id<MTLBuffer> buffer;
  std::shared_ptr<ResourceLedger> ledger;
  std::uint64_t length = 0;
};

class MetalHeap::Impl {
public:
  Impl(id<MTLHeap> value, std::shared_ptr<ResourceLedger> selected_ledger)
      : heap(value), ledger(std::move(selected_ledger)),
        length(heap == nil ? 0 : static_cast<std::uint64_t>([heap size])) {
    if (heap == nil || ledger == nullptr) {
      throw std::runtime_error("Metal heap allocation failed");
    }
    ledger->add_heap(length);
  }

  ~Impl() { ledger->remove_heap(length); }

  id<MTLHeap> heap;
  std::shared_ptr<ResourceLedger> ledger;
  std::uint64_t length = 0;
};

namespace {

struct CommandPlanCacheEntry {
  MetalCommandPlanDescriptor descriptor;
  std::weak_ptr<const MetalCommandPlan::Impl> plan;
};

class RegistryState final : public std::enable_shared_from_this<RegistryState> {
public:
  RegistryState(id<MTLDevice> selected_device, MetalContextOptions options)
      : device(selected_device), ledger(std::make_shared<ResourceLedger>()) {
    if (device == nil) {
      throw std::runtime_error("Metal device is unavailable");
    }
    @autoreleasepool {
      queue = [device newCommandQueue];
      shared_event = [device newSharedEvent];

      capabilities.device_name = ns_string([device name]);
      capabilities.max_buffer_length =
          static_cast<std::uint64_t>([device maxBufferLength]);
      capabilities.recommended_working_set_bytes =
          static_cast<std::uint64_t>([device recommendedMaxWorkingSetSize]);
      if (@available(macOS 14.0, *)) {
        capabilities.architecture_name = ns_string([device architecture].name);
      }

      const auto version = [NSProcessInfo processInfo].operatingSystemVersion;
      capabilities.runtime_os = std::to_string(version.majorVersion) + "." +
                                std::to_string(version.minorVersion) + "." +
                                std::to_string(version.patchVersion);

      MetalCapabilityProbe probe;
      probe.runtime_os_major = static_cast<std::uint32_t>(version.majorVersion);
      probe.runtime_os_minor = static_cast<std::uint32_t>(version.minorVersion);
      probe.runtime_os_patch = static_cast<std::uint32_t>(version.patchVersion);
      probe.sdk_max_allowed = __MAC_OS_X_VERSION_MAX_ALLOWED;
      probe.classic_command_queue = queue != nil;
      probe.shared_event = shared_event != nil;
      probe.metal4_headers = __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000;
      probe.compute_icb_api = [device
          respondsToSelector:@selector
          (newIndirectCommandBufferWithDescriptor:maxCommandCount:options:)];
      probe.compute_icb_barrier = version.majorVersion >= 11;
      probe.residency_sets = [device
          respondsToSelector:@selector(newResidencySetWithDescriptor:error:)];
      probe.mlx_bridge_available = options.mlx_bridge_available;
      for (std::uint32_t family = 12; family >= 1; --family) {
        const auto candidate = static_cast<MTLGPUFamily>(1000 + family);
        if ([device supportsFamily:candidate]) {
          probe.apple_gpu_family = family;
          break;
        }
      }

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
      if (@available(macOS 26.0, *)) {
        metal4_queue = [device newMTL4CommandQueue];
        id a0 = [device newCommandAllocator];
        id a1 = [device newCommandAllocator];
        if (a0 != nil)
          free_metal4_allocators.push_back(a0);
        if (a1 != nil)
          free_metal4_allocators.push_back(a1);
        probe.metal4_command_queue = metal4_queue != nil;
        probe.metal4_command_allocator = (a0 != nil && a1 != nil);
        id<MTL4CommandBuffer> command_buffer = [device newCommandBuffer];
        probe.metal4_command_buffer = command_buffer != nil;
        probe.metal4_compute_encoder =
            command_buffer != nil &&
            [command_buffer
                respondsToSelector:@selector(computeCommandEncoder)];
        MTL4CompilerDescriptor *compiler_descriptor =
            [[MTL4CompilerDescriptor alloc] init];
        compiler_descriptor.label = @"omlx_m5.runtime.compiler";
        NSError *compiler_error = nil;
        metal4_compiler = [device newCompilerWithDescriptor:compiler_descriptor
                                                      error:&compiler_error];
        probe.metal4_compiler = metal4_compiler != nil;
        MTL4ArgumentTableDescriptor *argument_table_descriptor =
            [[MTL4ArgumentTableDescriptor alloc] init];
        argument_table_descriptor.maxBufferBindCount = 1;
        argument_table_descriptor.initializeBindings = YES;
        NSError *argument_table_error = nil;
        id<MTL4ArgumentTable> argument_table =
            [device newArgumentTableWithDescriptor:argument_table_descriptor
                                             error:&argument_table_error];
        probe.metal4_argument_table = argument_table != nil;
        metal4_zero_binding =
            [device newBufferWithLength:4096
                                options:MTLResourceStorageModeShared |
                                        MTLResourceHazardTrackingModeTracked];
        if (metal4_zero_binding != nil) {
          std::memset([metal4_zero_binding contents], 0,
                      [metal4_zero_binding length]);
        }
        probe.metal4_argument_table =
            probe.metal4_argument_table && metal4_zero_binding != nil;
        probe.tensor_api =
            [device respondsToSelector:@selector
                    (tensorSizeAndAlignWithDescriptor:)] &&
            [device
                respondsToSelector:@selector(newTensorWithDescriptor:error:)];
      }
#endif

      const bool metal4_control_plane =
          probe.metal4_headers && probe.metal4_command_queue &&
          probe.metal4_command_allocator && probe.metal4_command_buffer &&
          probe.metal4_compute_encoder && probe.metal4_compiler &&
          probe.metal4_argument_table && probe.tensor_api && probe.shared_event;
      if (options.native_tensorops_runtime_override.has_value()) {
        probe.native_tensorops_runtime =
            *options.native_tensorops_runtime_override;
        probe.native_tensorops_source = "injected";
      } else if (probe.apple_gpu_family >= 10 && metal4_control_plane) {
        probe.native_tensorops_runtime = true;
        probe.native_tensorops_source = "apple10_metal4_probe";
      }
      capabilities.metal4_nax_runtime = probe.native_tensorops_runtime;
      capabilities.nax_capability_source = probe.native_tensorops_source;
      capabilities.profile = select_metal_capability_profile(std::move(probe));
    }

    if (queue == nil || shared_event == nil) {
      throw std::runtime_error(
          "classic Metal command queue or shared event is unavailable");
    }
  }

  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLSharedEvent> shared_event;
  // Keep these untyped so a deployment target older than macOS 26 can carry
  // the adapter while all typed Metal 4 use remains inside @available guards.
  id metal4_queue;
  std::vector<id> free_metal4_allocators;
  id metal4_compiler;
  id<MTLBuffer> metal4_zero_binding;
  std::array<id, 32> metal4_argument_table_pool{};
  std::shared_ptr<ResourceLedger> ledger;
  MetalCapabilities capabilities;
  mutable std::mutex mutex;
  mutable std::mutex submission_mutex;
  std::unordered_map<std::string, std::shared_ptr<LibraryEntry>> libraries;
  std::unordered_map<std::string, std::shared_ptr<PipelineEntry>> pipelines;
  std::unordered_map<std::string, CommandPlanCacheEntry> command_plans;
  MetalRegistryStats stats;
  std::string last_classic_async_error;
  std::atomic<std::uint64_t> next_event_value{0};
};

} // namespace

class MetalParameterTable::Impl {
public:
  Impl(std::shared_ptr<RegistryState> selected_owner,
       MetalParameterTableDescriptor selected_descriptor,
       MetalBuffer selected_buffer)
      : owner(std::move(selected_owner)),
        descriptor(std::move(selected_descriptor)),
        buffer(std::move(selected_buffer)) {
    for (std::size_t index = 0; index < descriptor.fields.size(); ++index) {
      fields.emplace(descriptor.fields[index].key, index);
    }
  }

  std::shared_ptr<RegistryState> owner;
  MetalParameterTableDescriptor descriptor;
  MetalBuffer buffer;
  std::unordered_map<std::string, std::size_t> fields;
  mutable std::mutex mutex;
  std::uint64_t generation = 0;
};

class MetalResidencySet::Impl {
public:
  Impl(std::shared_ptr<RegistryState> selected_owner, std::string selected_key,
       id<MTLResidencySet> selected_set)
      : owner(std::move(selected_owner)), key(std::move(selected_key)),
        residency_set(selected_set) {}

  std::shared_ptr<RegistryState> owner;
  std::string key;
  id<MTLResidencySet> residency_set;
  std::set<const void *> allocation_ids;
  mutable std::mutex mutex;
  bool is_committed = false;
};

class MetalCommandPlan::Impl {
public:
  struct StableArgument {
    std::uint32_t argument_index = 0;
    MetalBuffer buffer;
  };

  std::shared_ptr<RegistryState> owner;
  MetalCommandPlanDescriptor descriptor;
  std::vector<std::shared_ptr<PipelineEntry>> pipelines;
  std::vector<std::uint32_t> resource_slots;
  std::vector<std::vector<StableArgument>> stable_arguments;
};

class MetalIcbProfile::Impl {
public:
  std::shared_ptr<RegistryState> owner;
  std::string key;
  std::shared_ptr<const MetalCommandPlan::Impl> plan;
  std::shared_ptr<MetalParameterTable::Impl> parameters;
  id<MTLIndirectCommandBuffer> commands;
  id<MTLResidencySet> residency;
  std::vector<id<MTLBuffer>> retained_buffers;
  bool metal4_outer = false;
  mutable std::mutex mutex;
  MetalIcbProfileStats stats;
};

class MetalContext::Impl {
public:
  using ResourceBindingMap =
      std::unordered_map<std::uint32_t, const MetalBufferBinding *>;

  explicit Impl(std::shared_ptr<RegistryState> selected_state)
      : state(std::move(selected_state)) {}

  static ResourceBindingMap validate_dispatch_bindings(
      const ResourceLedger &ledger, const MetalCommandPlan::Impl &plan,
      std::span<const MetalBufferBinding> bindings,
      const MetalBuffer *parameter_buffer, std::string_view context_prefix,
      bool check_parameter_alignment);

  std::shared_ptr<RegistryState> state;
};

MetalBuffer::MetalBuffer() = default;
MetalBuffer::MetalBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MetalBuffer::MetalBuffer(MetalBuffer &&) noexcept = default;
MetalBuffer &MetalBuffer::operator=(MetalBuffer &&) noexcept = default;
MetalBuffer::~MetalBuffer() = default;

std::uint64_t MetalBuffer::size() const {
  return impl_ == nullptr ? 0 : impl_->length;
}

std::uint64_t MetalBuffer::gpu_address() const {
  return impl_ == nullptr ? 0 : impl_->buffer.gpuAddress;
}

void *MetalBuffer::native_mtl_buffer() const noexcept {
  return impl_ == nullptr ? nullptr : (__bridge void *)impl_->buffer;
}

std::vector<std::byte> MetalBuffer::read(std::uint64_t offset,
                                         std::uint64_t length) const {
  const auto span = view(offset, length);
  return std::vector<std::byte>(span.begin(), span.end());
}

std::span<const std::byte> MetalBuffer::view(std::uint64_t offset,
                                             std::uint64_t length) const {
  if (impl_ == nullptr) {
    throw std::logic_error("Metal buffer is empty");
  }
  if (length == 0) {
    length = size() - offset;
  }
  validate_range(size(), offset, length);
  if (impl_->buffer.storageMode == MTLStorageModePrivate) {
    throw std::logic_error("Metal private buffer is not host-accessible");
  }
  const void *contents = [impl_->buffer contents];
  if (contents == nullptr) {
    throw std::logic_error("Metal private buffer is not host-accessible");
  }
  const auto *source = static_cast<const std::byte *>(contents) + offset;
  return std::span<const std::byte>(source, length);
}

void MetalBuffer::write(std::span<const std::byte> data, std::uint64_t offset) {
  if (impl_ == nullptr) {
    throw std::logic_error("Metal buffer is empty");
  }
  validate_range(size(), offset, data.size());
  if (impl_->buffer.storageMode == MTLStorageModePrivate) {
    throw std::logic_error("Metal private buffer is not host-accessible");
  }
  void *contents = [impl_->buffer contents];
  if (contents == nullptr) {
    throw std::logic_error("Metal private buffer is not host-accessible");
  }
  if (data.empty()) {
    return;
  }
  auto *destination = static_cast<std::byte *>(contents) + offset;
  std::memcpy(destination, data.data(), data.size());
}

void MetalBuffer::zero() {
  if (impl_ == nullptr) {
    throw std::logic_error("Metal buffer is empty");
  }
  if (impl_->length == 0) {
    return;
  }
  if (impl_->buffer.storageMode == MTLStorageModePrivate) {
    throw std::logic_error("Metal private buffer cannot be zeroed on host");
  }
  void *destination = [impl_->buffer contents];
  if (destination == nullptr) {
    throw std::logic_error("Metal private buffer cannot be zeroed on host");
  }
  std::memset(destination, 0, static_cast<std::size_t>(impl_->length));
}

MetalParameterTable::MetalParameterTable() = default;
MetalParameterTable::MetalParameterTable(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
MetalParameterTable::MetalParameterTable(MetalParameterTable &&) noexcept =
    default;
MetalParameterTable &
MetalParameterTable::operator=(MetalParameterTable &&) noexcept = default;
MetalParameterTable::~MetalParameterTable() = default;

bool MetalParameterTable::valid() const { return impl_ != nullptr; }

const std::string &MetalParameterTable::key() const {
  if (impl_ == nullptr) {
    throw std::logic_error("Metal parameter table is empty");
  }
  return impl_->descriptor.key;
}

std::uint64_t MetalParameterTable::size() const {
  return impl_ == nullptr ? 0 : impl_->descriptor.byte_size;
}

std::uint64_t MetalParameterTable::gpu_address() const {
  if (impl_ == nullptr) {
    throw std::logic_error("Metal parameter table is empty");
  }
  return impl_->buffer.impl_->buffer.gpuAddress;
}

std::uint64_t MetalParameterTable::generation() const {
  if (impl_ == nullptr) {
    return 0;
  }
  std::scoped_lock lock(impl_->mutex);
  return impl_->generation;
}

std::uint64_t MetalParameterTable::field_count() const {
  return impl_ == nullptr ? 0 : impl_->descriptor.fields.size();
}

void MetalParameterTable::write(std::string_view field,
                                std::span<const std::byte> data) {
  if (impl_ == nullptr) {
    throw std::logic_error("Metal parameter table is empty");
  }
  std::scoped_lock lock(impl_->mutex);
  const auto found = impl_->fields.find(std::string(field));
  if (found == impl_->fields.end()) {
    throw std::out_of_range("Metal parameter field is absent: " +
                            std::string(field));
  }
  const auto &descriptor = impl_->descriptor.fields[found->second];
  if (data.size() != descriptor.size) {
    throw std::invalid_argument("Metal parameter field size mismatch");
  }
  impl_->buffer.write(data, descriptor.offset);
  ++impl_->generation;
  auto &state = *impl_->owner;
  std::scoped_lock state_lock(state.mutex);
  ++state.stats.dynamic_parameter_updates;
}

std::vector<std::byte> MetalParameterTable::read(std::string_view field) const {
  if (impl_ == nullptr) {
    throw std::logic_error("Metal parameter table is empty");
  }
  std::scoped_lock lock(impl_->mutex);
  const auto found = impl_->fields.find(std::string(field));
  if (found == impl_->fields.end()) {
    throw std::out_of_range("Metal parameter field is absent: " +
                            std::string(field));
  }
  const auto &descriptor = impl_->descriptor.fields[found->second];
  return impl_->buffer.read(descriptor.offset, descriptor.size);
}

MetalResidencySet::MetalResidencySet() = default;
MetalResidencySet::MetalResidencySet(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
MetalResidencySet::MetalResidencySet(MetalResidencySet &&) noexcept = default;
MetalResidencySet &
MetalResidencySet::operator=(MetalResidencySet &&) noexcept = default;
MetalResidencySet::~MetalResidencySet() {
  if (impl_ != nullptr && impl_->residency_set != nil) {
    [impl_->residency_set endResidency];
  }
}

bool MetalResidencySet::valid() const { return impl_ != nullptr; }

const std::string &MetalResidencySet::key() const {
  if (impl_ == nullptr) {
    throw std::logic_error("Metal residency set is empty");
  }
  return impl_->key;
}

std::uint64_t MetalResidencySet::allocation_count() const {
  if (impl_ == nullptr) {
    return 0;
  }
  std::scoped_lock lock(impl_->mutex);
  return impl_->allocation_ids.size();
}

std::uint64_t MetalResidencySet::allocated_size() const {
  if (impl_ == nullptr || impl_->residency_set == nil) {
    return 0;
  }
  std::scoped_lock lock(impl_->mutex);
  return impl_->residency_set.allocatedSize;
}

bool MetalResidencySet::committed() const {
  if (impl_ == nullptr) {
    return false;
  }
  std::scoped_lock lock(impl_->mutex);
  return impl_->is_committed;
}

MetalHeap::MetalHeap() = default;
MetalHeap::MetalHeap(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MetalHeap::MetalHeap(MetalHeap &&) noexcept = default;
MetalHeap &MetalHeap::operator=(MetalHeap &&) noexcept = default;
MetalHeap::~MetalHeap() = default;

std::uint64_t MetalHeap::size() const {
  return impl_ == nullptr ? 0 : impl_->length;
}

MetalBuffer MetalHeap::make_buffer(std::uint64_t length) const {
  if (impl_ == nullptr || length == 0) {
    throw std::invalid_argument("Metal heap buffer length must be positive");
  }
  id<MTLBuffer> buffer =
      [impl_->heap newBufferWithLength:checked_ns_integer(length, "heap buffer")
                               options:MTLResourceStorageModeShared |
                                       MTLResourceHazardTrackingModeTracked];
  return MetalBuffer(
      std::make_unique<MetalBuffer::Impl>(buffer, impl_->ledger));
}

MetalCommandPlan::MetalCommandPlan() = default;
MetalCommandPlan::MetalCommandPlan(std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {}
MetalCommandPlan::MetalCommandPlan(const MetalCommandPlan &) noexcept = default;
MetalCommandPlan &
MetalCommandPlan::operator=(const MetalCommandPlan &) noexcept = default;
MetalCommandPlan::MetalCommandPlan(MetalCommandPlan &&) noexcept = default;
MetalCommandPlan &
MetalCommandPlan::operator=(MetalCommandPlan &&) noexcept = default;
MetalCommandPlan::~MetalCommandPlan() = default;

bool MetalCommandPlan::valid() const { return impl_ != nullptr; }

const std::string &MetalCommandPlan::key() const {
  if (impl_ == nullptr) {
    throw std::logic_error("Metal command plan is empty");
  }
  return impl_->descriptor.key;
}

std::uint64_t MetalCommandPlan::dispatch_count() const {
  return impl_ == nullptr ? 0 : impl_->descriptor.dispatches.size();
}

std::uint64_t MetalCommandPlan::barrier_count() const {
  if (impl_ == nullptr) {
    return 0;
  }
  std::uint64_t count = 0;
  for (std::size_t index = 1; index < impl_->descriptor.dispatches.size();
       ++index) {
    if (impl_->descriptor.dispatches[index].barrier_before) {
      ++count;
    }
  }
  return count;
}

std::uint64_t MetalCommandPlan::resource_slot_count() const {
  return impl_ == nullptr ? 0 : impl_->resource_slots.size();
}

bool MetalCommandPlan::icb_qualified() const {
  return impl_ != nullptr && impl_->descriptor.icb_qualified;
}

const std::vector<MetalCommandRangeDescriptor> &
MetalCommandPlan::ranges() const {
  if (impl_ == nullptr) {
    throw std::logic_error("Metal command plan is empty");
  }
  return impl_->descriptor.ranges;
}

MetalIcbProfile::MetalIcbProfile() = default;
MetalIcbProfile::MetalIcbProfile(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
MetalIcbProfile::MetalIcbProfile(MetalIcbProfile &&) noexcept = default;
MetalIcbProfile &
MetalIcbProfile::operator=(MetalIcbProfile &&) noexcept = default;
MetalIcbProfile::~MetalIcbProfile() = default;

bool MetalIcbProfile::valid() const { return impl_ != nullptr; }

const std::string &MetalIcbProfile::key() const {
  if (impl_ == nullptr) {
    throw std::logic_error("Metal ICB profile is empty");
  }
  return impl_->key;
}

const std::vector<MetalCommandRangeDescriptor> &
MetalIcbProfile::ranges() const {
  if (impl_ == nullptr) {
    throw std::logic_error("Metal ICB profile is empty");
  }
  return impl_->plan->descriptor.ranges;
}

MetalIcbProfileStats MetalIcbProfile::stats() const {
  if (impl_ == nullptr) {
    throw std::logic_error("Metal ICB profile is empty");
  }
  std::scoped_lock lock(impl_->mutex);
  return impl_->stats;
}

MetalContext MetalContext::create() { return create(MetalContextOptions{}); }

MetalContext MetalContext::create(std::optional<bool> nax_runtime_override) {
  MetalContextOptions options;
  options.native_tensorops_runtime_override = nax_runtime_override;
  return create(options);
}

MetalContext MetalContext::create(MetalContextOptions options) {
  return MetalContext(std::make_unique<Impl>(std::make_shared<RegistryState>(
      MTLCreateSystemDefaultDevice(), std::move(options))));
}

MetalContext::MetalContext(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
MetalContext::MetalContext(MetalContext &&) noexcept = default;
MetalContext &MetalContext::operator=(MetalContext &&) noexcept = default;
MetalContext::~MetalContext() = default;

const MetalCapabilities &MetalContext::capabilities() const {
  return impl_->state->capabilities;
}

std::uint64_t MetalContext::current_allocated_bytes() const {
  return static_cast<std::uint64_t>(impl_->state->device.currentAllocatedSize);
}
std::weak_ptr<void> MetalContext::diagnostic_owner_lifetime() const {
  return impl_->state;
}

MetalBuffer MetalContext::make_buffer(std::uint64_t length) const {
  const auto &state = *impl_->state;
  if (length == 0 || length > state.capabilities.max_buffer_length) {
    throw std::invalid_argument("Metal buffer length is unsupported");
  }
  id<MTLBuffer> buffer =
      [state.device newBufferWithLength:checked_ns_integer(length, "buffer")
                                options:MTLResourceStorageModeShared |
                                        MTLResourceHazardTrackingModeTracked];
  return MetalBuffer(std::make_unique<MetalBuffer::Impl>(buffer, state.ledger));
}

MetalBuffer MetalContext::make_private_buffer(std::uint64_t length) const {
  const auto &state = *impl_->state;
  if (length == 0 || length > state.capabilities.max_buffer_length) {
    throw std::invalid_argument("Metal private buffer length is unsupported");
  }
  id<MTLBuffer> buffer =
      [state.device newBufferWithLength:checked_ns_integer(length, "buffer")
                                options:MTLResourceStorageModePrivate |
                                        MTLResourceHazardTrackingModeTracked];
  return MetalBuffer(std::make_unique<MetalBuffer::Impl>(buffer, state.ledger));
}

MetalBuffer MetalContext::upload(std::span<const std::byte> data) const {
  if (data.empty()) {
    throw std::invalid_argument("cannot upload an empty Metal buffer");
  }
  auto buffer = make_buffer(data.size());
  upload_into(buffer, data);
  return buffer;
}

void MetalContext::upload_into(MetalBuffer &buffer,
                               std::span<const std::byte> data,
                               std::uint64_t offset) const {
  if (data.empty() || buffer.impl_ == nullptr ||
      buffer.impl_->ledger.get() != impl_->state->ledger.get()) {
    throw std::invalid_argument(
        "Metal upload destination belongs to another context or is empty");
  }
  buffer.write(data, offset);
  impl_->state->ledger->record_upload(data.size());
}

MetalHeap MetalContext::make_heap(std::uint64_t size) const {
  const auto &state = *impl_->state;
  if (size == 0 || size > state.capabilities.recommended_working_set_bytes) {
    throw std::invalid_argument("Metal heap size is unsupported");
  }
  id<MTLHeap> heap = nil;
  @autoreleasepool {
    MTLHeapDescriptor *descriptor = [[MTLHeapDescriptor alloc] init];
    descriptor.size = checked_ns_integer(size, "heap");
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
    heap = [state.device newHeapWithDescriptor:descriptor];
  }
  return MetalHeap(std::make_unique<MetalHeap::Impl>(heap, state.ledger));
}

MetalParameterTable MetalContext::make_parameter_table(
    MetalParameterTableDescriptor descriptor) const {
  if (descriptor.key.empty() || descriptor.byte_size == 0 ||
      descriptor.byte_size > impl_->state->capabilities.max_buffer_length ||
      descriptor.fields.empty()) {
    throw std::invalid_argument("Metal parameter table descriptor is invalid");
  }
  std::set<std::string> keys;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> intervals;
  intervals.reserve(descriptor.fields.size());
  for (const auto &field : descriptor.fields) {
    if (field.key.empty() || field.size == 0 || field.alignment == 0 ||
        !std::has_single_bit(field.alignment) ||
        field.offset % field.alignment != 0 || !keys.insert(field.key).second) {
      throw std::invalid_argument("Metal parameter field is invalid");
    }
    const auto end = checked_add(field.offset, field.size, "parameter field");
    if (end > descriptor.byte_size) {
      throw std::invalid_argument("Metal parameter field is out of bounds");
    }
    intervals.emplace_back(field.offset, end);
  }
  std::ranges::sort(intervals);
  for (std::size_t index = 1; index < intervals.size(); ++index) {
    if (intervals[index].first < intervals[index - 1].second) {
      throw std::invalid_argument("Metal parameter fields overlap");
    }
  }
  auto buffer = make_buffer(descriptor.byte_size);
  buffer.write(std::vector<std::byte>(descriptor.byte_size));
  {
    auto &state = *impl_->state;
    std::scoped_lock lock(state.mutex);
    ++state.stats.dynamic_parameter_tables_created;
    state.stats.dynamic_parameter_bytes += descriptor.byte_size;
  }
  return MetalParameterTable(std::make_shared<MetalParameterTable::Impl>(
      impl_->state, std::move(descriptor), std::move(buffer)));
}

MetalResidencySet
MetalContext::make_residency_set(std::string key,
                                 std::uint64_t initial_capacity) const {
  if (key.empty() ||
      initial_capacity > std::numeric_limits<NSUInteger>::max() ||
      !impl_->state->capabilities.profile.probe.residency_sets) {
    throw std::invalid_argument("Metal residency set descriptor is invalid");
  }
  id<MTLResidencySet> residency_set = nil;
  @autoreleasepool {
    MTLResidencySetDescriptor *descriptor =
        [[MTLResidencySetDescriptor alloc] init];
    descriptor.label = to_ns_string(key);
    descriptor.initialCapacity = static_cast<NSUInteger>(initial_capacity);
    NSError *error = nil;
    residency_set =
        [impl_->state->device newResidencySetWithDescriptor:descriptor
                                                      error:&error];
    if (residency_set == nil) {
      throw std::runtime_error("Metal residency set creation failed: " +
                               ns_string([error localizedDescription]));
    }
  }
  {
    std::scoped_lock lock(impl_->state->mutex);
    ++impl_->state->stats.persistent_residency_sets_created;
  }
  auto result = MetalResidencySet(std::make_unique<MetalResidencySet::Impl>(
      impl_->state, std::move(key), residency_set));
  if (impl_->state->capabilities.profile.metal4_direct_eligible) {
    const void *identity =
        (__bridge const void *)impl_->state->metal4_zero_binding;
    result.impl_->allocation_ids.insert(identity);
    [result.impl_->residency_set
        addAllocation:impl_->state->metal4_zero_binding];
    std::scoped_lock lock(impl_->state->mutex);
    ++impl_->state->stats.persistent_residency_allocations;
  }
  return result;
}

void MetalContext::residency_add_buffer(MetalResidencySet &residency,
                                        const MetalBuffer &buffer) const {
  if (residency.impl_ == nullptr || buffer.impl_ == nullptr ||
      residency.impl_->owner.get() != impl_->state.get() ||
      buffer.impl_->ledger.get() != impl_->state->ledger.get()) {
    throw std::invalid_argument("Metal residency buffer is empty or foreign");
  }
  std::scoped_lock lock(residency.impl_->mutex);
  const void *identity = (__bridge const void *)buffer.impl_->buffer;
  if (residency.impl_->allocation_ids.insert(identity).second) {
    [residency.impl_->residency_set addAllocation:buffer.impl_->buffer];
    residency.impl_->is_committed = false;
    std::scoped_lock state_lock(impl_->state->mutex);
    ++impl_->state->stats.persistent_residency_allocations;
  }
}

void MetalContext::residency_add_parameters(
    MetalResidencySet &residency, const MetalParameterTable &parameters) const {
  if (parameters.impl_ == nullptr ||
      parameters.impl_->owner.get() != impl_->state.get()) {
    throw std::invalid_argument(
        "Metal residency parameter table is empty or foreign");
  }
  residency_add_buffer(residency, parameters.impl_->buffer);
}

void MetalContext::residency_add_plan(MetalResidencySet &residency,
                                      const MetalCommandPlan &plan) const {
  if (residency.impl_ == nullptr || plan.impl_ == nullptr ||
      residency.impl_->owner.get() != impl_->state.get() ||
      plan.impl_->owner.get() != impl_->state.get()) {
    throw std::invalid_argument("Metal residency plan is empty or foreign");
  }
  std::scoped_lock lock(residency.impl_->mutex);
  const auto add_allocation = [&](id<MTLAllocation> allocation) {
    const void *identity = (__bridge const void *)allocation;
    if (residency.impl_->allocation_ids.insert(identity).second) {
      [residency.impl_->residency_set addAllocation:allocation];
      ++impl_->state->stats.persistent_residency_allocations;
    }
  };
  {
    std::scoped_lock state_lock(impl_->state->mutex);
    for (std::size_t index = 0; index < plan.impl_->pipelines.size(); ++index) {
      const auto &pipeline = plan.impl_->pipelines[index];
      add_allocation(impl_->state->capabilities.profile.metal4_direct_eligible
                         ? pipeline->metal4_pipeline
                         : pipeline->pipeline);
      for (const auto &argument : plan.impl_->stable_arguments[index]) {
        add_allocation(argument.buffer.impl_->buffer);
      }
    }
  }
  residency.impl_->is_committed = false;
}

void MetalContext::residency_commit(MetalResidencySet &residency) const {
  if (residency.impl_ == nullptr ||
      residency.impl_->owner.get() != impl_->state.get()) {
    throw std::invalid_argument("Metal residency set is empty or foreign");
  }
  std::scoped_lock lock(residency.impl_->mutex);
  [residency.impl_->residency_set commit];
  residency.impl_->is_committed = true;
  std::scoped_lock state_lock(impl_->state->mutex);
  ++impl_->state->stats.persistent_residency_commits;
}

void MetalContext::register_library(MetalLibraryDescriptor descriptor) const {
  if (descriptor.key.empty() || descriptor.path.empty()) {
    throw std::invalid_argument("Metal library key and path are required");
  }
  if (!descriptor.expected_sha256.empty() &&
      !lower_hex(descriptor.expected_sha256, 64)) {
    throw std::invalid_argument("Metal library SHA-256 is invalid");
  }
  descriptor.path = std::filesystem::weakly_canonical(descriptor.path);
  if (!std::filesystem::is_regular_file(descriptor.path)) {
    if (const char *reloc_dir = std::getenv("OMLX_M5_RELOCATE_METALLIB_DIR");
        reloc_dir != nullptr && *reloc_dir != '\0' &&
        descriptor.path.extension() == ".metallib") {
      const auto candidate = std::filesystem::weakly_canonical(
          std::filesystem::path(reloc_dir) / descriptor.path.filename());
      if (std::filesystem::is_regular_file(candidate)) {
        descriptor.path = candidate;
      }
    }
  }
  if (!std::filesystem::is_regular_file(descriptor.path)) {
    throw std::invalid_argument("Metal library path is not a regular file");
  }

  auto &state = *impl_->state;
  std::scoped_lock lock(state.mutex);
  if (state.stats.warmup_sealed) {
    throw std::logic_error("Metal registry warmup is sealed");
  }
  const auto existing = state.libraries.find(descriptor.key);
  if (existing != state.libraries.end()) {
    const bool expected_matches =
        descriptor.expected_sha256.empty() ||
        descriptor.expected_sha256 == existing->second->sha256;
    if (descriptor.path != existing->second->descriptor.path ||
        !expected_matches) {
      throw std::invalid_argument("Metal library key is already bound");
    }
    ++state.stats.library_cache_hits;
    return;
  }

  try {
    const auto digest = sha256_file(descriptor.path);
    if (!descriptor.expected_sha256.empty() &&
        descriptor.expected_sha256 != digest) {
      throw std::runtime_error("Metal library SHA-256 mismatch");
    }
    id<MTLLibrary> library = nil;
    @autoreleasepool {
      NSURL *url =
          [NSURL fileURLWithPath:to_ns_string(descriptor.path.string())];
      NSError *error = nil;
      library = [state.device newLibraryWithURL:url error:&error];
      if (library == nil) {
        throw std::runtime_error("Metal library load failed: " +
                                 ns_string([error localizedDescription]));
      }
    }
    auto entry = std::make_shared<LibraryEntry>();
    entry->descriptor = std::move(descriptor);
    entry->sha256 = digest;
    entry->library = library;
    state.libraries.emplace(entry->descriptor.key, std::move(entry));
    ++state.stats.libraries_loaded;
  } catch (...) {
    ++state.stats.library_load_failures;
    throw;
  }
}

MetalPipelineInfo
MetalContext::warmup_pipeline(MetalPipelineDescriptor descriptor) const {
  if (descriptor.key.empty() || descriptor.library_key.empty() ||
      descriptor.function_name.empty() || descriptor.variant.empty()) {
    throw std::invalid_argument("Metal pipeline descriptor is incomplete");
  }
  auto &state = *impl_->state;
  std::scoped_lock lock(state.mutex);
  if (state.stats.warmup_sealed) {
    throw std::logic_error("Metal registry warmup is sealed");
  }
  const auto cached = state.pipelines.find(descriptor.key);
  if (cached != state.pipelines.end()) {
    if (!same_pipeline(descriptor, cached->second->descriptor)) {
      throw std::invalid_argument("Metal pipeline key is already bound");
    }
    ++state.stats.pipeline_cache_hits;
    return cached->second->info;
  }
  const auto library = state.libraries.find(descriptor.library_key);
  if (library == state.libraries.end()) {
    throw std::invalid_argument("Metal pipeline library is not registered");
  }

  try {
    @autoreleasepool {
      id<MTLFunction> function = [library->second->library
          newFunctionWithName:to_ns_string(descriptor.function_name)];
      if (function == nil) {
        throw std::runtime_error("Metal function is absent: " +
                                 descriptor.function_name);
      }
      NSError *error = nil;
      MTLAutoreleasedComputePipelineReflection reflection = nil;
      id<MTLComputePipelineState> pipeline = nil;
      if (descriptor.support_indirect_command_buffers) {
        MTLComputePipelineDescriptor *pipeline_descriptor =
            [[MTLComputePipelineDescriptor alloc] init];
        pipeline_descriptor.label = to_ns_string(descriptor.key);
        pipeline_descriptor.computeFunction = function;
        pipeline_descriptor.supportIndirectCommandBuffers = YES;
        pipeline = [state.device
            newComputePipelineStateWithDescriptor:pipeline_descriptor
                                          options:MTLPipelineOptionBindingInfo
                                       reflection:&reflection
                                            error:&error];
      } else {
        pipeline = [state.device
            newComputePipelineStateWithFunction:function
                                        options:MTLPipelineOptionBindingInfo
                                     reflection:&reflection
                                          error:&error];
      }
      if (pipeline == nil) {
        throw std::runtime_error("Metal pipeline compilation failed: " +
                                 ns_string([error localizedDescription]));
      }
      MetalPipelineInfo info;
      info.descriptor = descriptor;
      info.thread_execution_width =
          static_cast<std::uint64_t>(pipeline.threadExecutionWidth);
      info.max_threads_per_threadgroup =
          static_cast<std::uint64_t>(pipeline.maxTotalThreadsPerThreadgroup);
      info.static_threadgroup_memory_bytes =
          static_cast<std::uint64_t>(pipeline.staticThreadgroupMemoryLength);
      info.indirect_command_buffers_enabled =
          pipeline.supportIndirectCommandBuffers;
      if (descriptor.support_indirect_command_buffers &&
          !info.indirect_command_buffers_enabled) {
        throw std::runtime_error(
            "Metal pipeline did not enable indirect command buffers");
      }
      id<MTLComputePipelineState> metal4_pipeline = nil;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
      if (state.capabilities.profile.probe.metal4_compiler) {
        if (@available(macOS 26.0, *)) {
          MTL4LibraryFunctionDescriptor *function_descriptor =
              [[MTL4LibraryFunctionDescriptor alloc] init];
          function_descriptor.name = to_ns_string(descriptor.function_name);
          function_descriptor.library = library->second->library;
          MTL4ComputePipelineDescriptor *pipeline_descriptor =
              [[MTL4ComputePipelineDescriptor alloc] init];
          pipeline_descriptor.label = to_ns_string(descriptor.key + ".metal4");
          pipeline_descriptor.computeFunctionDescriptor = function_descriptor;
          pipeline_descriptor.supportIndirectCommandBuffers =
              descriptor.support_indirect_command_buffers
                  ? MTL4IndirectCommandBufferSupportStateEnabled
                  : MTL4IndirectCommandBufferSupportStateDisabled;
          NSError *metal4_error = nil;
          metal4_pipeline = [(id<MTL4Compiler>)state.metal4_compiler
              newComputePipelineStateWithDescriptor:pipeline_descriptor
                                compilerTaskOptions:nil
                                              error:&metal4_error];
          if (metal4_pipeline == nil ||
              (descriptor.support_indirect_command_buffers &&
               !metal4_pipeline.supportIndirectCommandBuffers)) {
            throw std::runtime_error(
                "Metal 4 ICB pipeline compilation failed: " +
                ns_string([metal4_error localizedDescription]));
          }
          info.metal4_pipeline_enabled = true;
          info.metal4_indirect_command_buffers_enabled =
              metal4_pipeline.supportIndirectCommandBuffers;
        }
      }
#endif
      for (id<MTLBinding> binding in reflection.bindings) {
        if (!binding.used) {
          continue;
        }
        MetalPipelineBindingInfo binding_info{
            .index = static_cast<std::uint32_t>(binding.index),
            .name = ns_string(binding.name),
            .kind = binding_kind(binding.type),
            .access = binding_access(binding.access),
        };
        if (binding.type == MTLBindingTypeBuffer) {
          id<MTLBufferBinding> buffer_binding = (id<MTLBufferBinding>)binding;
          binding_info.alignment =
              static_cast<std::uint64_t>(buffer_binding.bufferAlignment);
          binding_info.data_size =
              static_cast<std::uint64_t>(buffer_binding.bufferDataSize);
        }
        info.bindings.push_back(std::move(binding_info));
      }
      std::ranges::sort(info.bindings, {}, &MetalPipelineBindingInfo::index);

      auto entry = std::make_shared<PipelineEntry>();
      entry->descriptor = std::move(descriptor);
      entry->info = info;
      for (const auto &binding : info.bindings) {
        if (binding.kind == "buffer") {
          entry->max_buffer_bind_count =
              std::max(entry->max_buffer_bind_count,
                       static_cast<NSUInteger>(binding.index) + 1);
        }
      }
      entry->pipeline = pipeline;
      entry->metal4_pipeline = metal4_pipeline;
      state.pipelines.emplace(entry->descriptor.key, std::move(entry));
      ++state.stats.pipelines_compiled;
      return info;
    }
  } catch (...) {
    ++state.stats.pipeline_compile_failures;
    throw;
  }
}

MetalPipelineInfo
MetalContext::pipeline_info(const std::string &pipeline_key) const {
  auto &state = *impl_->state;
  std::scoped_lock lock(state.mutex);
  const auto pipeline = state.pipelines.find(pipeline_key);
  if (pipeline == state.pipelines.end()) {
    throw std::out_of_range("Metal pipeline is not warmed: " + pipeline_key);
  }
  return pipeline->second->info;
}

MetalCommandPlan MetalContext::prepare_command_plan(
    MetalCommandPlanDescriptor descriptor) const {
  if (descriptor.key.empty() || descriptor.dispatches.empty()) {
    throw std::invalid_argument("Metal command plan is empty");
  }
  if (descriptor.dispatches.size() > UINT32_MAX) {
    throw std::invalid_argument("Metal command plan is too large");
  }
  if (descriptor.ranges.empty()) {
    descriptor.ranges.push_back({
        .key = descriptor.key + ".full",
        .location = 0,
        .length = static_cast<std::uint32_t>(descriptor.dispatches.size()),
    });
  }
  std::set<std::string> range_keys;
  for (const auto &range : descriptor.ranges) {
    if (range.key.empty() || range.length == 0 ||
        !range_keys.insert(range.key).second ||
        range.location > descriptor.dispatches.size() ||
        range.length > descriptor.dispatches.size() - range.location) {
      throw std::invalid_argument("Metal command range is invalid");
    }
  }
  auto &state = *impl_->state;
  std::scoped_lock lock(state.mutex);
  if (state.stats.warmup_sealed) {
    throw std::logic_error("Metal registry warmup is sealed");
  }
  const auto cached = state.command_plans.find(descriptor.key);
  if (cached != state.command_plans.end()) {
    if (auto plan = cached->second.plan.lock()) {
      if (!same_plan(descriptor, cached->second.descriptor)) {
        throw std::invalid_argument("Metal command plan key is already bound");
      }
      ++state.stats.command_plan_cache_hits;
      return MetalCommandPlan(std::move(plan));
    }
    state.command_plans.erase(cached);
  }

  auto plan = std::make_shared<MetalCommandPlan::Impl>();
  plan->owner = impl_->state;
  plan->descriptor = std::move(descriptor);
  std::set<std::uint32_t> resource_slots;
  for (const auto &dispatch : plan->descriptor.dispatches) {
    const auto pipeline = state.pipelines.find(dispatch.pipeline_key);
    if (pipeline == state.pipelines.end()) {
      throw std::invalid_argument(
          "command plan references an unwarmed pipeline");
    }
    if (plan->descriptor.icb_qualified &&
        dispatch.kind == MetalDispatchKind::threadgroups_indirect) {
      throw std::invalid_argument(
          "ICB-qualified plan does not support threadgroups_indirect dispatch");
    }
    if (plan->descriptor.icb_qualified &&
        (!pipeline->second->info.indirect_command_buffers_enabled ||
         (state.capabilities.profile.metal4_icb_eligible &&
          !pipeline->second->info.metal4_indirect_command_buffers_enabled))) {
      throw std::invalid_argument(
          "ICB-qualified plan references a non-ICB pipeline");
    }
    const auto thread_count = checked_multiply(
        checked_multiply(dispatch.threads_per_threadgroup.width,
                         dispatch.threads_per_threadgroup.height,
                         "threadgroup"),
        dispatch.threads_per_threadgroup.depth, "threadgroup");
    static_cast<void>(make_mtl_size(dispatch.grid, "dispatch grid"));
    static_cast<void>(make_mtl_size(dispatch.threads_per_threadgroup,
                                    "threads per threadgroup"));
    if (thread_count > pipeline->second->info.max_threads_per_threadgroup) {
      throw std::invalid_argument("command plan threadgroup is too large");
    }

    std::set<std::uint32_t> argument_indices;
    for (const auto &resource : dispatch.resources) {
      if (resource.minimum_bytes == 0 ||
          !argument_indices.insert(resource.argument_index).second) {
        throw std::invalid_argument("command plan resource binding is invalid");
      }
      const auto reflected = std::ranges::find_if(
          pipeline->second->info.bindings,
          [&](const MetalPipelineBindingInfo &binding) {
            return binding.index == resource.argument_index;
          });
      if (reflected == pipeline->second->info.bindings.end() ||
          reflected->kind != "buffer" ||
          resource.minimum_bytes < reflected->data_size) {
        throw std::invalid_argument(
            "command plan resource disagrees with pipeline reflection at " +
            std::to_string(resource.argument_index) + " (minimum=" +
            std::to_string(resource.minimum_bytes) + ", reflected=" +
            (reflected == pipeline->second->info.bindings.end()
                 ? std::string("missing")
                 : std::to_string(reflected->data_size)) +
            ")");
      }
      resource_slots.insert(resource.resource_slot);
    }
    for (const auto &constant_argument : dispatch.constant_arguments) {
      if (constant_argument.bytes.empty() ||
          constant_argument.bytes.size() > 4096 ||
          !argument_indices.insert(constant_argument.argument_index).second) {
        throw std::invalid_argument("command plan constant binding is invalid");
      }
      const auto reflected = std::ranges::find_if(
          pipeline->second->info.bindings,
          [&](const MetalPipelineBindingInfo &binding) {
            return binding.index == constant_argument.argument_index;
          });
      if (reflected == pipeline->second->info.bindings.end() ||
          reflected->kind != "buffer" ||
          constant_argument.bytes.size() < reflected->data_size) {
        throw std::invalid_argument(
            "command plan constant data disagrees with pipeline reflection");
      }
    }
    for (const auto &dynamic_argument : dispatch.dynamic_arguments) {
      if (dynamic_argument.minimum_bytes == 0 ||
          !argument_indices.insert(dynamic_argument.argument_index).second) {
        throw std::invalid_argument("command plan dynamic binding is invalid");
      }
      const auto reflected = std::ranges::find_if(
          pipeline->second->info.bindings,
          [&](const MetalPipelineBindingInfo &binding) {
            return binding.index == dynamic_argument.argument_index;
          });
      if (reflected == pipeline->second->info.bindings.end() ||
          reflected->kind != "buffer" ||
          dynamic_argument.minimum_bytes < reflected->data_size ||
          (reflected->alignment != 0 &&
           dynamic_argument.parameter_offset % reflected->alignment != 0)) {
        throw std::invalid_argument(
            "command plan dynamic data disagrees with pipeline reflection");
      }
    }
    for (const auto &binding : pipeline->second->info.bindings) {
      if (binding.kind != "buffer" ||
          !argument_indices.contains(binding.index)) {
        throw std::invalid_argument(
            "command plan does not cover pipeline reflection bindings");
      }
    }
    if (argument_indices.size() != pipeline->second->info.bindings.size()) {
      throw std::invalid_argument(
          "command plan contains an unknown pipeline binding");
    }
    if (dispatch.kind == MetalDispatchKind::threadgroups_indirect) {
      if ((dispatch.indirect_grid_offset % 4ULL) != 0ULL) {
        throw std::invalid_argument(
            "command plan indirect grid offset must be 4-byte aligned");
      }
      resource_slots.insert(dispatch.indirect_grid_resource_slot);
    }
    std::vector<MetalCommandPlan::Impl::StableArgument> stable_arguments;
    stable_arguments.reserve(dispatch.constant_arguments.size());
    for (const auto &constant_argument : dispatch.constant_arguments) {
      auto buffer = make_buffer(constant_argument.bytes.size());
      buffer.write(constant_argument.bytes);
      stable_arguments.push_back({
          .argument_index = constant_argument.argument_index,
          .buffer = std::move(buffer),
      });
      ++state.stats.stable_parameter_buffers_created;
      state.stats.stable_parameter_bytes += constant_argument.bytes.size();
    }
    plan->pipelines.push_back(pipeline->second);
    plan->stable_arguments.push_back(std::move(stable_arguments));
  }
  plan->resource_slots.assign(resource_slots.begin(), resource_slots.end());
  state.command_plans.insert_or_assign(
      plan->descriptor.key,
      CommandPlanCacheEntry{plan->descriptor, plan});
  ++state.stats.command_plans_built;
  return MetalCommandPlan(std::move(plan));
}

void MetalContext::seal_warmup() const {
  auto &state = *impl_->state;
  std::scoped_lock lock(state.mutex);
  state.stats.warmup_sealed = true;
}

bool MetalContext::warmup_sealed() const {
  auto &state = *impl_->state;
  std::scoped_lock lock(state.mutex);
  return state.stats.warmup_sealed;
}

MetalContext::Impl::ResourceBindingMap
MetalContext::Impl::validate_dispatch_bindings(
    const ResourceLedger &ledger, const MetalCommandPlan::Impl &plan,
    std::span<const MetalBufferBinding> bindings,
    const MetalBuffer *parameter_buffer, std::string_view context_prefix,
    bool check_parameter_alignment) {
  ResourceBindingMap resources;
  for (const auto &binding : bindings) {
    if (binding.buffer == nullptr || binding.buffer->impl_ == nullptr ||
        binding.buffer->impl_->ledger.get() != &ledger ||
        !resources.emplace(binding.resource_slot, &binding).second) {
      throw std::invalid_argument(std::string(context_prefix) +
                                  " binding is invalid");
    }
  }
  if (resources.size() != plan.resource_slots.size()) {
    throw std::invalid_argument(std::string(context_prefix) +
                                " resource slot count mismatch");
  }
  for (const auto slot : plan.resource_slots) {
    if (!resources.contains(slot)) {
      throw std::invalid_argument(std::string(context_prefix) +
                                  " resource slot is missing");
    }
  }
  for (std::size_t index = 0; index < plan.descriptor.dispatches.size();
       ++index) {
    const auto &dispatch = plan.descriptor.dispatches[index];
    const auto &pipeline = plan.pipelines[index]->info;
    for (const auto &resource : dispatch.resources) {
      const auto *selected = resources.at(resource.resource_slot);
      validate_range(selected->buffer->size(), selected->offset,
                     resource.minimum_bytes);
      const auto reflected = std::ranges::find_if(
          pipeline.bindings, [&](const MetalPipelineBindingInfo &binding) {
            return binding.index == resource.argument_index;
          });
      if (reflected == pipeline.bindings.end() ||
          (reflected->alignment != 0 &&
           selected->offset % reflected->alignment != 0)) {
        throw std::invalid_argument(std::string(context_prefix) +
                                    " buffer alignment is invalid");
      }
    }
    if (dispatch.kind == MetalDispatchKind::threadgroups_indirect) {
      const auto *ind = resources.at(dispatch.indirect_grid_resource_slot);
      const auto ind_off = checked_add(ind->offset, dispatch.indirect_grid_offset,
                                       "indirect grid offset");
      validate_range(ind->buffer->size(), ind_off, 12ULL);
      if ((ind_off % 4ULL) != 0ULL) {
        throw std::invalid_argument(std::string(context_prefix) +
                                    " indirect grid alignment is invalid");
      }
    }
    for (const auto &dynamic_argument : dispatch.dynamic_arguments) {
      validate_range(parameter_buffer->size(),
                     dynamic_argument.parameter_offset,
                     dynamic_argument.minimum_bytes);
      if (check_parameter_alignment) {
        const auto reflected = std::ranges::find_if(
            pipeline.bindings, [&](const MetalPipelineBindingInfo &binding) {
              return binding.index == dynamic_argument.argument_index;
            });
        if (reflected == pipeline.bindings.end() ||
            (reflected->alignment != 0 &&
             dynamic_argument.parameter_offset % reflected->alignment != 0)) {
          throw std::invalid_argument(std::string(context_prefix) +
                                      " parameter alignment is invalid");
        }
      }
    }
  }
  return resources;
}

namespace {

struct SubmittedWorkResult {
  GpuTiming timing;
  std::uint64_t event_value = 0;
};

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
struct Metal4AsyncFeedbackState {
  dispatch_semaphore_t completed = dispatch_semaphore_create(0);
  NSError *error = nil;
  CFTimeInterval gpu_start = 0.0;
  CFTimeInterval gpu_end = 0.0;
};

API_AVAILABLE(macos(26.0))
id<MTL4CommandAllocator> acquire_metal4_allocator(RegistryState &state) {
  id<MTL4CommandAllocator> allocator = nil;
  {
    std::scoped_lock lock(state.mutex);
    if (!state.free_metal4_allocators.empty()) {
      allocator = state.free_metal4_allocators.back();
      state.free_metal4_allocators.pop_back();
    }
  }
  if (allocator == nil) {
    allocator = [state.device newCommandAllocator];
  }
  if (allocator != nil) {
    [allocator reset];
  }
  return allocator;
}

API_AVAILABLE(macos(26.0))
void release_metal4_allocator(RegistryState &state,
                              id<MTL4CommandAllocator> allocator) noexcept {
  if (allocator == nil)
    return;
  [allocator reset];
  std::scoped_lock lock(state.mutex);
  state.free_metal4_allocators.push_back(allocator);
}

template <typename EncodeFn>
API_AVAILABLE(macos(26.0))
SubmittedWorkResult submit_metal4_work(
    RegistryState &state, NSString *label,
    std::span<const id<MTLResidencySet>> residency_sets,
    bool signal_shared_event, bool record_failure_on_alloc_or_timeout,
    bool record_failure_on_error, std::string_view alloc_error_msg,
    std::string_view timeout_error_msg, std::string_view exec_error_prefix,
    EncodeFn &&encode_fn) {
  id<MTL4CommandAllocator> allocator = acquire_metal4_allocator(state);
  if (allocator == nil) {
    if (record_failure_on_alloc_or_timeout) {
      std::scoped_lock lock(state.mutex);
      ++state.stats.submission_failures;
    }
    throw std::runtime_error(std::string(alloc_error_msg));
  }
  struct AllocatorGuard {
    RegistryState *st;
    id<MTL4CommandAllocator> alloc;
    bool active = true;
    ~AllocatorGuard() {
      if (active && st != nullptr && alloc != nil)
        release_metal4_allocator(*st, alloc);
    }
  } alloc_guard{&state, allocator, true};

  id<MTL4CommandBuffer> command_buffer = [state.device newCommandBuffer];
  if (command_buffer == nil) {
    if (record_failure_on_alloc_or_timeout) {
      std::scoped_lock lock(state.mutex);
      ++state.stats.submission_failures;
    }
    throw std::runtime_error(std::string(alloc_error_msg));
  }
  [command_buffer beginCommandBufferWithAllocator:allocator];
  command_buffer.label = label;
  if (residency_sets.size() == 1) {
    [command_buffer useResidencySet:residency_sets.front()];
  } else if (residency_sets.size() > 1) {
    [command_buffer useResidencySets:residency_sets.data()
                               count:residency_sets.size()];
  }
  encode_fn(command_buffer);
  [command_buffer endCommandBuffer];

  auto fb_state = std::make_shared<Metal4AsyncFeedbackState>();
  MTL4CommitOptions *commit_options = [[MTL4CommitOptions alloc] init];
  [commit_options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
    fb_state->error = feedback.error;
    fb_state->gpu_start = feedback.GPUStartTime;
    fb_state->gpu_end = feedback.GPUEndTime;
    dispatch_semaphore_signal(fb_state->completed);
  }];
  const std::uint64_t event_value =
      signal_shared_event ? (state.next_event_value.fetch_add(1) + 1) : 0;
  const auto start = std::chrono::steady_clock::now();
  const id<MTL4CommandBuffer> command_buffers[] = {command_buffer};
  [state.metal4_queue commit:command_buffers count:1 options:commit_options];
  alloc_guard.active = false;
  if (signal_shared_event) {
    [state.metal4_queue signalEvent:state.shared_event value:event_value];
  }
  if (dispatch_semaphore_wait(
          fb_state->completed,
          dispatch_time(DISPATCH_TIME_NOW, 30LL * NSEC_PER_SEC)) != 0) {
    // Do not reset/release an uncompleted GPU allocator on timeout!
    if (record_failure_on_alloc_or_timeout) {
      std::scoped_lock lock(state.mutex);
      ++state.stats.submission_failures;
    }
    throw std::runtime_error(std::string(timeout_error_msg));
  }
  const auto wall_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - start)
                           .count();
  release_metal4_allocator(state, allocator);
  if (fb_state->error != nil) {
    if (record_failure_on_error) {
      std::scoped_lock lock(state.mutex);
      ++state.stats.submission_failures;
    }
    throw std::runtime_error(std::string(exec_error_prefix) +
                             ns_string([fb_state->error localizedDescription]));
  }
  const double gpu_ms = fb_state->gpu_end > fb_state->gpu_start
                            ? (fb_state->gpu_end - fb_state->gpu_start) * 1000.0
                            : 0.0;
  return {.timing = {.wall_ms = wall_ms, .gpu_ms = gpu_ms},
          .event_value = event_value};
}
#endif

template <typename EncodeFn>
SubmittedWorkResult submit_classic_work(
    RegistryState &state, NSString *label, bool record_failure_stat,
    std::string_view alloc_error_msg, std::string_view exec_error_prefix,
    EncodeFn &&encode_fn) {
  id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
  if (command_buffer == nil) {
    if (record_failure_stat) {
      std::scoped_lock lock(state.mutex);
      ++state.stats.submission_failures;
    }
    throw std::runtime_error(std::string(alloc_error_msg));
  }
  if (label != nil) {
    command_buffer.label = label;
  }
  encode_fn(command_buffer);

  std::uint64_t event_value = 0;
  std::chrono::steady_clock::time_point start;
  {
    std::scoped_lock submission_lock(state.submission_mutex);
    event_value = state.next_event_value.fetch_add(1) + 1;
    [command_buffer encodeSignalEvent:state.shared_event value:event_value];
    start = std::chrono::steady_clock::now();
    [command_buffer commit];
  }
  [command_buffer waitUntilCompleted];
  const auto wall_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - start)
                           .count();
  if (command_buffer.status == MTLCommandBufferStatusError) {
    if (record_failure_stat) {
      std::scoped_lock lock(state.mutex);
      ++state.stats.submission_failures;
    }
    throw std::runtime_error(std::string(exec_error_prefix) +
                             ns_string([command_buffer.error localizedDescription]));
  }
  const double gpu_ms =
      command_buffer.GPUEndTime > command_buffer.GPUStartTime
          ? (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0
          : 0.0;
  return {.timing = {.wall_ms = wall_ms, .gpu_ms = gpu_ms},
          .event_value = event_value};
}

} // namespace

MetalSubmission
MetalContext::execute(const MetalCommandPlan &plan,
                      std::span<const MetalBufferBinding> bindings,
                      const MetalParameterTable *parameters) const {
  MetalCommandInvocation invocation{
      .plan = &plan,
      .bindings = {bindings.begin(), bindings.end()},
      .parameters = parameters,
  };
  return execute_batch(std::span<const MetalCommandInvocation>(&invocation, 1));
}

MetalSubmission MetalContext::execute_batch(
    std::span<const MetalCommandInvocation> invocations) const {
  if (invocations.empty()) {
    throw std::invalid_argument("Metal command batch is empty");
  }
  std::vector<Impl::ResourceBindingMap> invocation_resources;
  invocation_resources.reserve(invocations.size());
  std::set<const MetalParameterTable::Impl *> locked_parameter_tables;
  std::vector<std::unique_lock<std::mutex>> parameter_locks;
  parameter_locks.reserve(invocations.size());
  std::uint64_t total_dispatches = 0;
  for (const auto &invocation : invocations) {
    if (invocation.plan == nullptr || invocation.plan->impl_ == nullptr ||
        invocation.plan->impl_->owner.get() != impl_->state.get()) {
      throw std::invalid_argument(
          "Metal command batch plan belongs to another context");
    }
    const auto requires_parameters =
        std::ranges::any_of(invocation.plan->impl_->descriptor.dispatches,
                            [](const MetalDispatchDescriptor &dispatch) {
                              return !dispatch.dynamic_arguments.empty();
                            });
    if (requires_parameters &&
        (invocation.parameters == nullptr ||
         invocation.parameters->impl_ == nullptr ||
         invocation.parameters->impl_->owner.get() != impl_->state.get())) {
      throw std::invalid_argument(
          "Metal command batch parameter table is missing or foreign");
    }
    if (requires_parameters &&
        locked_parameter_tables.insert(invocation.parameters->impl_.get())
            .second) {
      parameter_locks.emplace_back(invocation.parameters->impl_->mutex);
    }
    auto resources = Impl::validate_dispatch_bindings(
        *impl_->state->ledger, *invocation.plan->impl_, invocation.bindings,
        requires_parameters ? &invocation.parameters->impl_->buffer : nullptr,
        "Metal batch", true);
    total_dispatches += invocation.plan->impl_->descriptor.dispatches.size();
    invocation_resources.push_back(std::move(resources));
  }

  auto &state = *impl_->state;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
  if (state.capabilities.profile.metal4_direct_eligible) {
    if (@available(macOS 26.0, *)) {
      std::scoped_lock submission_lock(state.submission_mutex);
      @autoreleasepool {
        std::unordered_set<const void *> seen_allocations;
        std::vector<id<MTLAllocation>> unique_allocations;
        seen_allocations.reserve(invocations.size() * 64 + 64);
        unique_allocations.reserve(invocations.size() * 64 + 64);
        const auto add_unique = [&](id<MTLAllocation> allocation) {
          if (allocation != nil &&
              seen_allocations.insert((__bridge const void *)allocation)
                  .second) {
            unique_allocations.push_back(allocation);
          }
        };
        add_unique(state.metal4_zero_binding);
        for (std::size_t invocation_index = 0;
             invocation_index < invocations.size(); ++invocation_index) {
          const auto &invocation = invocations[invocation_index];
          const auto &resources = invocation_resources[invocation_index];
          for (std::size_t dispatch_index = 0;
               dispatch_index <
               invocation.plan->impl_->descriptor.dispatches.size();
               ++dispatch_index) {
            const auto &dispatch =
                invocation.plan->impl_->descriptor.dispatches[dispatch_index];
            const auto &pipeline =
                invocation.plan->impl_->pipelines[dispatch_index];
            if (pipeline->metal4_pipeline == nil ||
                !pipeline->info.metal4_pipeline_enabled) {
              throw std::runtime_error(
                  "Metal 4 batch references a non-Metal-4 pipeline");
            }
            add_unique(pipeline->metal4_pipeline);
            for (const auto &resource : dispatch.resources) {
              add_unique(resources.at(resource.resource_slot)
                             ->buffer->impl_->buffer);
            }
            if (dispatch.kind == MetalDispatchKind::threadgroups_indirect) {
              add_unique(resources.at(dispatch.indirect_grid_resource_slot)
                             ->buffer->impl_->buffer);
            }
            for (const auto &argument :
                 invocation.plan->impl_->stable_arguments[dispatch_index]) {
              add_unique(argument.buffer.impl_->buffer);
            }
            if (!dispatch.dynamic_arguments.empty()) {
              add_unique(
                  invocation.parameters->impl_->buffer.impl_->buffer);
            }
          }
        }
        MTLResidencySetDescriptor *residency_descriptor =
            [[MTLResidencySetDescriptor alloc] init];
        residency_descriptor.label = @"omlx_m5.execute_batch.residency";
        residency_descriptor.initialCapacity = unique_allocations.size();
        NSError *residency_error = nil;
        id<MTLResidencySet> residency_set =
            [state.device newResidencySetWithDescriptor:residency_descriptor
                                                  error:&residency_error];
        if (residency_set == nil) {
          std::scoped_lock lock(state.mutex);
          ++state.stats.submission_failures;
          throw std::runtime_error(
              "Metal 4 batch residency creation failed: " +
              ns_string([residency_error localizedDescription]));
        }
        if (!unique_allocations.empty()) {
          [residency_set addAllocations:unique_allocations.data()
                                  count:unique_allocations.size()];
        }
        [residency_set commit];

        std::uint64_t argument_tables_created = 0;
        std::uint64_t argument_table_allocations = 0;
        std::uint64_t argument_table_pool_hits = 0;
        NSString *cb_label = to_ns_string(
            invocations.size() == 1
                ? invocations.front().plan->impl_->descriptor.key + ".metal4"
                : "omlx_m5.metal4.command_batch." +
                      std::to_string(invocations.size()));
        const id<MTLResidencySet> residency_arr[] = {residency_set};
        const auto submitted = submit_metal4_work(
            state, cb_label, std::span(residency_arr, 1), true, true, true,
            "Metal 4 batch command buffer allocation failed",
            "Metal 4 command batch timed out", "Metal 4 command batch failed: ",
            [&](id<MTL4CommandBuffer> command_buffer) {
              id<MTL4ComputeCommandEncoder> encoder =
                  [command_buffer computeCommandEncoder];
              if (encoder == nil) {
                std::scoped_lock lock(state.mutex);
                ++state.stats.submission_failures;
                throw std::runtime_error(
                    "Metal 4 batch compute encoder allocation failed");
              }
              const MTLStages queue_stages = MTLStageDispatch | MTLStageBlit;
              [encoder barrierAfterQueueStages:queue_stages
                                  beforeStages:queue_stages
                             visibilityOptions:MTL4VisibilityOptionDevice];
              bool first_dispatch = true;
              for (std::size_t invocation_index = 0;
                   invocation_index < invocations.size(); ++invocation_index) {
                const auto &invocation = invocations[invocation_index];
                const auto &resources = invocation_resources[invocation_index];
                for (std::size_t dispatch_index = 0;
                     dispatch_index <
                     invocation.plan->impl_->descriptor.dispatches.size();
                     ++dispatch_index) {
                  const auto &dispatch =
                      invocation.plan->impl_->descriptor
                          .dispatches[dispatch_index];
                  const auto &pipeline =
                      invocation.plan->impl_->pipelines[dispatch_index];
                  if (!first_dispatch &&
                      (dispatch_index == 0 || dispatch.barrier_before)) {
                    [encoder barrierAfterEncoderStages:MTLStageDispatch
                                   beforeEncoderStages:MTLStageDispatch
                                     visibilityOptions:
                                         MTL4VisibilityOptionDevice];
                  }
                  first_dispatch = false;

                  const NSUInteger max_buffer_bind_count =
                      pipeline->max_buffer_bind_count;
                  const NSUInteger pool_slot =
                      std::max<NSUInteger>(1, max_buffer_bind_count);
                  if (pool_slot >= state.metal4_argument_table_pool.size()) {
                    throw std::invalid_argument(
                        "Metal 4 argument table bind count exceeds hardware limit");
                  }
                  id<MTL4ArgumentTable> argument_table =
                      (id<MTL4ArgumentTable>)
                          state.metal4_argument_table_pool[pool_slot];
                  if (argument_table == nil) {
                    MTL4ArgumentTableDescriptor *argument_descriptor =
                        [[MTL4ArgumentTableDescriptor alloc] init];
                    argument_descriptor.label =
                        to_ns_string("omlx_m5.argument_table.pool.b" +
                                     std::to_string(pool_slot));
                    argument_descriptor.maxBufferBindCount = pool_slot;
                    argument_descriptor.initializeBindings = YES;
                    NSError *argument_error = nil;
                    argument_table = [state.device
                        newArgumentTableWithDescriptor:argument_descriptor
                                                 error:&argument_error];
                    if (argument_table == nil) {
                      throw std::runtime_error(
                          "Metal 4 argument table creation failed: " +
                          ns_string([argument_error localizedDescription]));
                    }
                    state.metal4_argument_table_pool[pool_slot] =
                        argument_table;
                    ++argument_table_allocations;
                  } else {
                    ++argument_table_pool_hits;
                  }
                  ++argument_tables_created;
                  for (NSUInteger argument_index = 0;
                       argument_index < max_buffer_bind_count;
                       ++argument_index) {
                    [argument_table
                        setAddress:state.metal4_zero_binding.gpuAddress
                           atIndex:argument_index];
                  }
                  for (const auto &resource : dispatch.resources) {
                    const auto *selected = resources.at(resource.resource_slot);
                    const auto address = checked_add(
                        selected->buffer->impl_->buffer.gpuAddress,
                        selected->offset, "Metal 4 buffer GPU address");
                    [argument_table setAddress:address
                                       atIndex:resource.argument_index];
                  }
                  for (const auto &argument :
                       invocation.plan->impl_->stable_arguments[dispatch_index]) {
                    [argument_table
                        setAddress:argument.buffer.impl_->buffer.gpuAddress
                           atIndex:argument.argument_index];
                  }
                  for (const auto &argument : dispatch.dynamic_arguments) {
                    const auto address = checked_add(
                        invocation.parameters->impl_->buffer.impl_->buffer
                            .gpuAddress,
                        argument.parameter_offset,
                        "Metal 4 parameter GPU address");
                    [argument_table setAddress:address
                                       atIndex:argument.argument_index];
                  }
                  [encoder setComputePipelineState:pipeline->metal4_pipeline];
                  [encoder setArgumentTable:argument_table];
                  if (dispatch.kind == MetalDispatchKind::threads) {
                    [encoder dispatchThreads:make_mtl_size(dispatch.grid,
                                                           "dispatch grid")
                        threadsPerThreadgroup:make_mtl_size(
                                                  dispatch.threads_per_threadgroup,
                                                  "threads per threadgroup")];
                  } else if (dispatch.kind ==
                             MetalDispatchKind::threadgroups_indirect) {
                    const auto *ind =
                        resources.at(dispatch.indirect_grid_resource_slot);
                    const auto ind_addr = checked_add(
                        checked_add(ind->buffer->impl_->buffer.gpuAddress,
                                    ind->offset,
                                    "Metal 4 indirect buffer GPU address"),
                        dispatch.indirect_grid_offset,
                        "Metal 4 indirect grid offset");
                    [encoder dispatchThreadgroupsWithIndirectBuffer:ind_addr
                        threadsPerThreadgroup:make_mtl_size(
                                                  dispatch.threads_per_threadgroup,
                                                  "threads per threadgroup")];
                  } else {
                    [encoder
                         dispatchThreadgroups:make_mtl_size(dispatch.grid,
                                                            "dispatch grid")
                        threadsPerThreadgroup:make_mtl_size(
                                                  dispatch.threads_per_threadgroup,
                                                  "threads per threadgroup")];
                  }
                }
              }
              [encoder barrierAfterStages:queue_stages
                        beforeQueueStages:queue_stages
                        visibilityOptions:MTL4VisibilityOptionDevice];
              [encoder endEncoding];
            });
        {
          std::scoped_lock lock(state.mutex);
          ++state.stats.command_buffers_submitted;
          ++state.stats.metal4_command_buffers_submitted;
          state.stats.dispatches_submitted += total_dispatches;
          state.stats.metal4_argument_tables_created += argument_tables_created;
          state.stats.metal4_argument_table_allocations +=
              argument_table_allocations;
          state.stats.metal4_argument_table_pool_hits +=
              argument_table_pool_hits;
          state.stats.last_shared_event_value =
              std::max(state.stats.last_shared_event_value,
                       submitted.event_value);
        }
        return {
            .timing = submitted.timing,
            .shared_event_value = submitted.event_value,
            .dispatch_count = total_dispatches,
        };
      }
    }
  }
#endif
  @autoreleasepool {
    NSString *cb_label = to_ns_string(
        invocations.size() == 1
            ? invocations.front().plan->impl_->descriptor.key
            : "omlx_m5.command_batch." + std::to_string(invocations.size()));
    const auto submitted = submit_classic_work(
        state, cb_label, true, "Metal command buffer allocation failed",
        "Metal command batch failed: ",
        [&](id<MTLCommandBuffer> command_buffer) {
          for (std::size_t invocation_index = 0;
               invocation_index < invocations.size(); ++invocation_index) {
            const auto &invocation = invocations[invocation_index];
            const auto &resources = invocation_resources[invocation_index];
            for (std::size_t index = 0;
                 index < invocation.plan->impl_->descriptor.dispatches.size();
                 ++index) {
              const auto &dispatch =
                  invocation.plan->impl_->descriptor.dispatches[index];
              id<MTLComputeCommandEncoder> encoder =
                  [command_buffer computeCommandEncoder];
              if (encoder == nil) {
                std::scoped_lock lock(state.mutex);
                ++state.stats.submission_failures;
                throw std::runtime_error(
                    "Metal compute encoder allocation failed");
              }
              [encoder setComputePipelineState:invocation.plan->impl_
                                                   ->pipelines[index]
                                                   ->pipeline];
              for (const auto &resource : dispatch.resources) {
                const auto *selected = resources.at(resource.resource_slot);
                [encoder setBuffer:selected->buffer->impl_->buffer
                            offset:checked_ns_integer(selected->offset,
                                                      "buffer offset")
                           atIndex:resource.argument_index];
              }
              for (const auto &argument :
                   invocation.plan->impl_->stable_arguments[index]) {
                [encoder setBuffer:argument.buffer.impl_->buffer
                            offset:0
                           atIndex:argument.argument_index];
              }
              for (const auto &argument : dispatch.dynamic_arguments) {
                [encoder
                    setBuffer:invocation.parameters->impl_->buffer.impl_->buffer
                       offset:checked_ns_integer(argument.parameter_offset,
                                                 "parameter offset")
                      atIndex:argument.argument_index];
              }
              if (dispatch.kind == MetalDispatchKind::threads) {
                [encoder
                      dispatchThreads:make_mtl_size(dispatch.grid,
                                                    "dispatch grid")
                    threadsPerThreadgroup:make_mtl_size(
                                              dispatch.threads_per_threadgroup,
                                              "threads per threadgroup")];
              } else if (dispatch.kind ==
                         MetalDispatchKind::threadgroups_indirect) {
                const auto *ind =
                    resources.at(dispatch.indirect_grid_resource_slot);
                const auto ind_off = checked_ns_integer(
                    checked_add(ind->offset, dispatch.indirect_grid_offset,
                                "indirect grid offset"),
                    "indirect grid offset");
                [encoder
                    dispatchThreadgroupsWithIndirectBuffer:ind->buffer->impl_
                                                               ->buffer
                                      indirectBufferOffset:ind_off
                                     threadsPerThreadgroup:
                                         make_mtl_size(
                                             dispatch.threads_per_threadgroup,
                                             "threads per threadgroup")];
              } else {
                [encoder
                     dispatchThreadgroups:make_mtl_size(dispatch.grid,
                                                        "dispatch grid")
                    threadsPerThreadgroup:make_mtl_size(
                                              dispatch.threads_per_threadgroup,
                                              "threads per threadgroup")];
              }
              [encoder endEncoding];
            }
          }
        });
    {
      std::scoped_lock lock(state.mutex);
      ++state.stats.command_buffers_submitted;
      ++state.stats.classic_command_buffers_submitted;
      state.stats.dispatches_submitted += total_dispatches;
      state.stats.last_shared_event_value =
          std::max(state.stats.last_shared_event_value, submitted.event_value);
    }
    return {
        .timing = submitted.timing,
        .shared_event_value = submitted.event_value,
        .dispatch_count = total_dispatches,
    };
  }
}

MetalIcbProfile MetalContext::prepare_icb_profile(
    std::string key, const MetalCommandInvocation &invocation) const {
  if (key.empty() || invocation.plan == nullptr ||
      invocation.plan->impl_ == nullptr ||
      invocation.plan->impl_->owner.get() != impl_->state.get() ||
      !invocation.plan->impl_->descriptor.icb_qualified) {
    throw std::invalid_argument("Metal ICB profile plan is invalid");
  }
  auto &state = *impl_->state;
  if (!state.capabilities.profile.compute_icb_eligible) {
    throw std::runtime_error("compute ICB is unavailable");
  }
  const auto requires_parameters =
      std::ranges::any_of(invocation.plan->impl_->descriptor.dispatches,
                          [](const MetalDispatchDescriptor &dispatch) {
                            return !dispatch.dynamic_arguments.empty();
                          });
  if (requires_parameters &&
      (invocation.parameters == nullptr ||
       invocation.parameters->impl_ == nullptr ||
       invocation.parameters->impl_->owner.get() != impl_->state.get())) {
    throw std::invalid_argument(
        "Metal ICB profile parameter table is missing or foreign");
  }
  const auto resources = Impl::validate_dispatch_bindings(
      *state.ledger, *invocation.plan->impl_, invocation.bindings,
      requires_parameters ? &invocation.parameters->impl_->buffer : nullptr,
      "Metal ICB profile", false);

  auto profile = std::make_shared<MetalIcbProfile::Impl>();
  profile->owner = impl_->state;
  profile->key = std::move(key);
  profile->plan = invocation.plan->impl_;
  profile->parameters =
      requires_parameters ? invocation.parameters->impl_ : nullptr;
  profile->metal4_outer =
      state.capabilities.profile.metal4_icb_eligible &&
      std::ranges::all_of(
          profile->plan->pipelines,
          [](const std::shared_ptr<PipelineEntry> &pipeline) {
            return pipeline->info.metal4_indirect_command_buffers_enabled;
          });

  @autoreleasepool {
    NSUInteger maximum_binding_count = 0;
    for (const auto &pipeline : profile->plan->pipelines) {
      for (const auto &binding : pipeline->info.bindings) {
        if (binding.kind == "buffer") {
          maximum_binding_count =
              std::max(maximum_binding_count,
                       static_cast<NSUInteger>(binding.index) + 1);
        }
      }
    }
    MTLIndirectCommandBufferDescriptor *descriptor =
        [[MTLIndirectCommandBufferDescriptor alloc] init];
    descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatchThreads |
                              MTLIndirectCommandTypeConcurrentDispatch;
    descriptor.inheritPipelineState = NO;
    descriptor.inheritBuffers = NO;
    descriptor.maxKernelBufferBindCount = maximum_binding_count;
    profile->commands = [state.device
        newIndirectCommandBufferWithDescriptor:descriptor
                               maxCommandCount:profile->plan->descriptor
                                                   .dispatches.size()
                                       options:
                                           MTLResourceStorageModeShared |
                                           MTLResourceHazardTrackingModeTracked];
    if (profile->commands == nil) {
      throw std::runtime_error("Metal ICB profile allocation failed");
    }

    std::set<const void *> retained_ids;
    const auto retain_buffer = [&](id<MTLBuffer> buffer) {
      if (buffer != nil &&
          retained_ids.insert((__bridge const void *)buffer).second) {
        profile->retained_buffers.push_back(buffer);
      }
    };
    for (const auto &binding : invocation.bindings) {
      retain_buffer(binding.buffer->impl_->buffer);
    }
    if (requires_parameters) {
      retain_buffer(invocation.parameters->impl_->buffer.impl_->buffer);
    }

    for (std::size_t index = 0;
         index < profile->plan->descriptor.dispatches.size(); ++index) {
      const auto &dispatch = profile->plan->descriptor.dispatches[index];
      const auto &pipeline = profile->plan->pipelines[index];
      id<MTLIndirectComputeCommand> command =
          [profile->commands indirectComputeCommandAtIndex:index];
      if (command == nil) {
        throw std::runtime_error("Metal ICB command allocation failed");
      }
      id<MTLComputePipelineState> pipeline_state =
          profile->metal4_outer ? pipeline->metal4_pipeline
                                : pipeline->pipeline;
      if (pipeline_state == nil) {
        throw std::runtime_error("Metal ICB pipeline is unavailable");
      }
      [command setComputePipelineState:pipeline_state];
      // Apple Silicon MTLIndirectComputeCommand buffer offsets silently wrap at
      // 4 GiB in hardware ICB execution without triggering API validation.
      constexpr std::uint64_t kMaxIcbBufferOffsetBytes =
          (1ULL << 32) - 4096ULL;
      for (const auto &resource : dispatch.resources) {
        const auto *selected = resources.at(resource.resource_slot);
        if (selected->offset > kMaxIcbBufferOffsetBytes) {
          throw std::invalid_argument(
              "Metal ICB resource offset exceeds 4 GiB hardware ICB limit: " +
              std::to_string(selected->offset));
        }
        [command setKernelBuffer:selected->buffer->impl_->buffer
                          offset:checked_ns_integer(selected->offset,
                                                    "ICB resource offset")
                         atIndex:resource.argument_index];
      }
      for (const auto &argument : profile->plan->stable_arguments[index]) {
        retain_buffer(argument.buffer.impl_->buffer);
        [command setKernelBuffer:argument.buffer.impl_->buffer
                          offset:0
                         atIndex:argument.argument_index];
      }
      for (const auto &argument : dispatch.dynamic_arguments) {
        if (argument.parameter_offset > kMaxIcbBufferOffsetBytes) {
          throw std::invalid_argument(
              "Metal ICB parameter offset exceeds 4 GiB hardware ICB limit: " +
              std::to_string(argument.parameter_offset));
        }
        [command
            setKernelBuffer:invocation.parameters->impl_->buffer.impl_->buffer
                     offset:checked_ns_integer(argument.parameter_offset,
                                               "ICB parameter offset")
                    atIndex:argument.argument_index];
      }
      if (index != 0 && dispatch.barrier_before) {
        if (![command respondsToSelector:@selector(setBarrier)]) {
          throw std::runtime_error("Metal ICB command barrier is unavailable");
        }
        [command setBarrier];
      }
      if (dispatch.kind == MetalDispatchKind::threads) {
        [command concurrentDispatchThreads:make_mtl_size(dispatch.grid,
                                                         "ICB dispatch grid")
                     threadsPerThreadgroup:make_mtl_size(
                                               dispatch.threads_per_threadgroup,
                                               "ICB threadgroup")];
      } else if (dispatch.kind == MetalDispatchKind::threadgroups) {
        [command
            concurrentDispatchThreadgroups:make_mtl_size(dispatch.grid,
                                                         "ICB threadgroup grid")
                     threadsPerThreadgroup:make_mtl_size(
                                               dispatch.threads_per_threadgroup,
                                               "ICB threadgroup")];
      } else {
        throw std::invalid_argument(
            "Metal ICB profile does not support threadgroups_indirect dispatch");
      }
    }

    MTLResidencySetDescriptor *residency_descriptor =
        [[MTLResidencySetDescriptor alloc] init];
    residency_descriptor.label = to_ns_string(profile->key + ".residency");
    NSError *residency_error = nil;
    profile->residency =
        [state.device newResidencySetWithDescriptor:residency_descriptor
                                              error:&residency_error];
    if (profile->residency == nil) {
      throw std::runtime_error(
          "Metal ICB profile residency allocation failed: " +
          ns_string([residency_error localizedDescription]));
    }
    std::set<const void *> allocation_ids;
    const auto add_allocation = [&](id<MTLAllocation> allocation) {
      if (allocation != nil &&
          allocation_ids.insert((__bridge const void *)allocation).second) {
        [profile->residency addAllocation:allocation];
      }
    };
    add_allocation(state.metal4_zero_binding);
    add_allocation(profile->commands);
    for (const auto &buffer : profile->retained_buffers) {
      add_allocation(buffer);
    }
    for (const auto &pipeline : profile->plan->pipelines) {
      add_allocation(profile->metal4_outer ? pipeline->metal4_pipeline
                                           : pipeline->pipeline);
    }
    [profile->residency commit];
    profile->stats = {
        .execution_route = profile->metal4_outer ? "metal4_compute_icb"
                                                 : "classic_compute_icb",
        .command_count = profile->plan->descriptor.dispatches.size(),
        .recording_count = 1,
        .residency_allocation_count = allocation_ids.size(),
    };
  }
  {
    std::scoped_lock lock(state.mutex);
    ++state.stats.icb_profiles_created;
    ++state.stats.icb_profile_recordings;
  }
  return MetalIcbProfile(std::move(profile));
}

struct MetalPendingSubmission::Impl {
  std::shared_ptr<RegistryState> owner;
  std::shared_ptr<MetalIcbProfile::Impl> profile;
  bool metal4_outer = false;
  id metal4_allocator = nil;
  std::uint64_t event_value = 0;
  std::uint64_t dispatch_count = 0;
  std::chrono::steady_clock::time_point start{};
  id<MTLCommandBuffer> classic_cb = nil;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
  std::shared_ptr<Metal4AsyncFeedbackState> metal4_fb;
#endif
  bool consumed = false;

  ~Impl() {
    if (consumed || !owner)
      return;
    consumed = true;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (metal4_outer && metal4_fb) {
      if (@available(macOS 26.0, *)) {
        if (dispatch_semaphore_wait(
                metal4_fb->completed,
                dispatch_time(DISPATCH_TIME_NOW, 30LL * NSEC_PER_SEC)) == 0) {
          release_metal4_allocator(*owner, metal4_allocator);
        }
      }
      return;
    }
#endif
    if (classic_cb != nil) {
      [classic_cb waitUntilCompleted];
    }
  }
};

MetalPendingSubmission::MetalPendingSubmission() = default;
MetalPendingSubmission::MetalPendingSubmission(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
MetalPendingSubmission::MetalPendingSubmission(
    MetalPendingSubmission &&) noexcept = default;
MetalPendingSubmission &MetalPendingSubmission::operator=(
    MetalPendingSubmission &&) noexcept = default;
MetalPendingSubmission::~MetalPendingSubmission() = default;
bool MetalPendingSubmission::valid() const noexcept {
  return impl_ != nullptr && !impl_->consumed;
}

MetalPendingSubmission
MetalContext::replay_icb_async(MetalIcbProfile &profile,
                               std::string_view range_key) const {
  if (profile.impl_ == nullptr ||
      profile.impl_->owner.get() != impl_->state.get()) {
    throw std::invalid_argument("Metal ICB profile is empty or foreign");
  }
  const auto range = std::ranges::find_if(
      profile.impl_->plan->descriptor.ranges,
      [&](const MetalCommandRangeDescriptor &candidate) {
        return candidate.key == range_key ||
               (range_key == "full" && candidate.location == 0 &&
                candidate.length ==
                    profile.impl_->plan->descriptor.dispatches.size());
      });
  if (range == profile.impl_->plan->descriptor.ranges.end()) {
    throw std::out_of_range("Metal ICB profile range is absent");
  }
  std::unique_lock parameter_lock =
      profile.impl_->parameters == nullptr
          ? std::unique_lock<std::mutex>{}
          : std::unique_lock(profile.impl_->parameters->mutex);
  std::scoped_lock profile_lock(profile.impl_->mutex);
  auto &state = *impl_->state;

  auto pending = std::make_unique<MetalPendingSubmission::Impl>();
  pending->owner = impl_->state;
  pending->profile = profile.impl_;
  pending->metal4_outer = profile.impl_->metal4_outer;
  pending->dispatch_count = range->length;

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
  if (profile.impl_->metal4_outer) {
    if (@available(macOS 26.0, *)) {
      id<MTL4CommandAllocator> allocator = acquire_metal4_allocator(state);
      if (allocator == nil) {
        throw std::runtime_error("Metal 4 async ICB allocator creation failed");
      }
      struct AllocatorGuard {
        RegistryState *st;
        id<MTL4CommandAllocator> alloc;
        bool active = true;
        ~AllocatorGuard() {
          if (active && st != nullptr && alloc != nil)
            release_metal4_allocator(*st, alloc);
        }
      } alloc_guard{&state, allocator, true};

      @autoreleasepool {
        std::scoped_lock submission_lock(state.submission_mutex);
        id<MTL4CommandBuffer> command_buffer = [state.device newCommandBuffer];
        if (command_buffer == nil) {
          throw std::runtime_error("Metal 4 async ICB command buffer failed");
        }
        [command_buffer beginCommandBufferWithAllocator:allocator];
        command_buffer.label = to_ns_string(profile.impl_->key + ".async");
        [command_buffer useResidencySet:profile.impl_->residency];
        id<MTL4ComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];
        if (encoder == nil) {
          throw std::runtime_error("Metal 4 async ICB encoder failed");
        }
        const MTLStages queue_stages = MTLStageDispatch | MTLStageBlit;
        [encoder barrierAfterQueueStages:queue_stages
                            beforeStages:queue_stages
                       visibilityOptions:MTL4VisibilityOptionDevice];
        [encoder executeCommandsInBuffer:profile.impl_->commands
                               withRange:NSMakeRange(range->location,
                                                     range->length)];
        [encoder barrierAfterStages:queue_stages
                  beforeQueueStages:queue_stages
                  visibilityOptions:MTL4VisibilityOptionDevice];
        [encoder endEncoding];
        [command_buffer endCommandBuffer];

        auto fb_state = std::make_shared<Metal4AsyncFeedbackState>();
        MTL4CommitOptions *commit_options = [[MTL4CommitOptions alloc] init];
        [commit_options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
          fb_state->error = feedback.error;
          fb_state->gpu_start = feedback.GPUStartTime;
          fb_state->gpu_end = feedback.GPUEndTime;
          dispatch_semaphore_signal(fb_state->completed);
        }];
        pending->metal4_allocator = allocator;
        pending->metal4_fb = fb_state;
        pending->start = std::chrono::steady_clock::now();
        const id<MTL4CommandBuffer> command_buffers[] = {command_buffer};
        [state.metal4_queue commit:command_buffers
                             count:1
                           options:commit_options];
        alloc_guard.active = false;
      }
      return MetalPendingSubmission(std::move(pending));
    }
    throw std::runtime_error("Metal 4 ICB profile requires macOS 26+");
  }
#endif

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
      throw std::runtime_error("classic async ICB command buffer failed");
    }
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      throw std::runtime_error("classic async ICB profile encoder failed");
    }
    for (const auto &buffer : profile.impl_->retained_buffers) {
      [encoder useResource:buffer
                     usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    }
    [encoder executeCommandsInBuffer:profile.impl_->commands
                           withRange:NSMakeRange(range->location,
                                                 range->length)];
    [encoder endEncoding];
    {
      std::scoped_lock submission_lock(state.submission_mutex);
      pending->event_value = state.next_event_value.fetch_add(1) + 1;
      [command_buffer encodeSignalEvent:state.shared_event
                                  value:pending->event_value];
      pending->start = std::chrono::steady_clock::now();
      [command_buffer commit];
    }
    pending->classic_cb = command_buffer;
  }
  return MetalPendingSubmission(std::move(pending));
}

MetalSubmission
MetalContext::wait_submission(MetalPendingSubmission &&pending) const {
  if (!pending.impl_ || pending.impl_->consumed ||
      pending.impl_->owner.get() != impl_->state.get()) {
    throw std::invalid_argument(
        "Metal pending submission is empty, consumed, or foreign");
  }
  auto p = std::move(pending.impl_);
  p->consumed = true;
  auto &state = *impl_->state;
  GpuTiming timing;

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
  if (p->metal4_outer) {
    if (@available(macOS 26.0, *)) {
      if (dispatch_semaphore_wait(
              p->metal4_fb->completed,
              dispatch_time(DISPATCH_TIME_NOW, 30LL * NSEC_PER_SEC)) != 0) {
        throw std::runtime_error("Metal 4 async ICB profile replay timed out");
      }
      const auto wall_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - p->start)
                               .count();
      release_metal4_allocator(state, p->metal4_allocator);
      if (p->metal4_fb->error != nil) {
        throw std::runtime_error(
            "Metal 4 async ICB profile replay failed: " +
            ns_string([p->metal4_fb->error localizedDescription]));
      }
      const double gpu_ms =
          p->metal4_fb->gpu_end > p->metal4_fb->gpu_start
              ? (p->metal4_fb->gpu_end - p->metal4_fb->gpu_start) * 1000.0
              : 0.0;
      timing = {.wall_ms = wall_ms, .gpu_ms = gpu_ms};
    } else {
      throw std::runtime_error("Metal 4 ICB profile requires macOS 26+");
    }
  } else
#endif
  {
    [p->classic_cb waitUntilCompleted];
    const auto wall_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - p->start)
                             .count();
    if (p->classic_cb.status == MTLCommandBufferStatusError) {
      throw std::runtime_error(
          "classic async ICB profile replay failed: " +
          ns_string([p->classic_cb.error localizedDescription]));
    }
    const double gpu_ms =
        p->classic_cb.GPUEndTime > p->classic_cb.GPUStartTime
            ? (p->classic_cb.GPUEndTime - p->classic_cb.GPUStartTime) * 1000.0
            : 0.0;
    timing = {.wall_ms = wall_ms, .gpu_ms = gpu_ms};
  }

  if (p->profile != nullptr) {
    std::scoped_lock profile_lock(p->profile->mutex);
    ++p->profile->stats.replay_count;
  }
  {
    std::scoped_lock lock(state.mutex);
    ++state.stats.command_buffers_submitted;
    if (p->metal4_outer) {
      ++state.stats.metal4_command_buffers_submitted;
    } else {
      ++state.stats.classic_command_buffers_submitted;
      state.stats.last_shared_event_value =
          std::max(state.stats.last_shared_event_value, p->event_value);
    }
    state.stats.dispatches_submitted += p->dispatch_count;
    ++state.stats.icb_profile_replays;
  }
  return {.timing = timing,
          .shared_event_value = p->event_value,
          .dispatch_count = p->dispatch_count};
}

MetalSubmission MetalContext::replay_icb(MetalIcbProfile &profile,
                                         std::string_view range_key) const {
  if (profile.impl_ == nullptr ||
      profile.impl_->owner.get() != impl_->state.get()) {
    throw std::invalid_argument("Metal ICB profile is empty or foreign");
  }
  const auto range = std::ranges::find_if(
      profile.impl_->plan->descriptor.ranges,
      [&](const MetalCommandRangeDescriptor &candidate) {
        return candidate.key == range_key ||
               (range_key == "full" && candidate.location == 0 &&
                candidate.length ==
                    profile.impl_->plan->descriptor.dispatches.size());
      });
  if (range == profile.impl_->plan->descriptor.ranges.end()) {
    throw std::out_of_range("Metal ICB profile range is absent");
  }
  return replay_icb_range(profile, range->location, range->length);
}

MetalSubmission MetalContext::replay_icb_range(MetalIcbProfile &profile,
                                               std::uint32_t location,
                                               std::uint32_t length) const {
  if (profile.impl_ == nullptr ||
      profile.impl_->owner.get() != impl_->state.get()) {
    throw std::invalid_argument("Metal ICB profile is empty or foreign");
  }
  const std::uint64_t total_dispatches =
      profile.impl_->plan->descriptor.dispatches.size();
  if (length == 0 ||
      static_cast<std::uint64_t>(location) +
              static_cast<std::uint64_t>(length) >
          total_dispatches) {
    throw std::out_of_range("Metal ICB profile subrange is out of bounds");
  }
  std::unique_lock parameter_lock =
      profile.impl_->parameters == nullptr
          ? std::unique_lock<std::mutex>{}
          : std::unique_lock(profile.impl_->parameters->mutex);
  std::scoped_lock profile_lock(profile.impl_->mutex);
  auto &state = *impl_->state;
  GpuTiming timing;
  std::uint64_t event_value = 0;

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
  if (profile.impl_->metal4_outer) {
    if (@available(macOS 26.0, *)) {
      std::scoped_lock submission_lock(state.submission_mutex);
      @autoreleasepool {
        const id<MTLResidencySet> residency_arr[] = {profile.impl_->residency};
        const auto submitted = submit_metal4_work(
            state, to_ns_string(profile.impl_->key + ".replay"),
            std::span(residency_arr, 1), false, false, false,
            "Metal 4 ICB command buffer failed",
            "Metal 4 ICB profile replay timed out",
            "Metal 4 ICB profile replay failed: ",
            [&](id<MTL4CommandBuffer> command_buffer) {
              id<MTL4ComputeCommandEncoder> encoder =
                  [command_buffer computeCommandEncoder];
              if (encoder == nil) {
                throw std::runtime_error("Metal 4 ICB encoder failed");
              }
              const MTLStages queue_stages = MTLStageDispatch | MTLStageBlit;
              [encoder barrierAfterQueueStages:queue_stages
                                  beforeStages:queue_stages
                             visibilityOptions:MTL4VisibilityOptionDevice];
              [encoder executeCommandsInBuffer:profile.impl_->commands
                                     withRange:NSMakeRange(location, length)];
              [encoder barrierAfterStages:queue_stages
                        beforeQueueStages:queue_stages
                        visibilityOptions:MTL4VisibilityOptionDevice];
              [encoder endEncoding];
            });
        timing = submitted.timing;
      }
    } else {
      throw std::runtime_error("Metal 4 ICB profile requires macOS 26+");
    }
  } else
#endif
  {
    @autoreleasepool {
      const auto submitted = submit_classic_work(
          state, nil, false, "classic ICB profile encoder failed",
          "classic ICB profile replay failed: ",
          [&](id<MTLCommandBuffer> command_buffer) {
            id<MTLComputeCommandEncoder> encoder =
                [command_buffer computeCommandEncoder];
            if (encoder == nil) {
              throw std::runtime_error("classic ICB profile encoder failed");
            }
            for (const auto &buffer : profile.impl_->retained_buffers) {
              [encoder useResource:buffer
                             usage:MTLResourceUsageRead |
                                   MTLResourceUsageWrite];
            }
            [encoder executeCommandsInBuffer:profile.impl_->commands
                                   withRange:NSMakeRange(location, length)];
            [encoder endEncoding];
          });
      timing = submitted.timing;
      event_value = submitted.event_value;
    }
  }
  ++profile.impl_->stats.replay_count;
  {
    std::scoped_lock lock(state.mutex);
    ++state.stats.command_buffers_submitted;
    if (profile.impl_->metal4_outer) {
      ++state.stats.metal4_command_buffers_submitted;
    } else {
      ++state.stats.classic_command_buffers_submitted;
      state.stats.last_shared_event_value =
          std::max(state.stats.last_shared_event_value, event_value);
    }
    state.stats.dispatches_submitted += length;
    ++state.stats.icb_profile_replays;
  }
  return {.timing = timing,
          .shared_event_value = event_value,
          .dispatch_count = length};
}

MetalSubmission
MetalContext::replay_icb_batch(std::span<MetalIcbProfile *const> profiles) const {
  if (profiles.empty()) {
    return {};
  }
  if (profiles.size() == 1) {
    return replay_icb(*profiles[0]);
  }
  std::uint64_t total_dispatches = 0;
  bool all_metal4 = true;
  bool any_metal4 = false;
  for (auto *prof : profiles) {
    if (prof == nullptr || prof->impl_ == nullptr ||
        prof->impl_->owner.get() != impl_->state.get()) {
      throw std::invalid_argument("Metal ICB batch contains empty or foreign profile");
    }
    total_dispatches += prof->impl_->plan->descriptor.dispatches.size();
    all_metal4 = all_metal4 && prof->impl_->metal4_outer;
    any_metal4 = any_metal4 || prof->impl_->metal4_outer;
  }
  if (any_metal4 != all_metal4) {
    throw std::invalid_argument("Metal ICB batch mixes Metal 4 and classic profiles");
  }

  // Acquire deduplicated, address-ordered parameter & profile mutexes matching
  // replay_icb()'s lock hierarchy (parameters->mutex before profile->mutex).
  std::vector<std::mutex *> param_mutexes;
  std::vector<std::mutex *> profile_mutexes;
  param_mutexes.reserve(profiles.size());
  profile_mutexes.reserve(profiles.size());
  for (auto *prof : profiles) {
    if (prof->impl_->parameters != nullptr) {
      param_mutexes.push_back(&prof->impl_->parameters->mutex);
    }
    profile_mutexes.push_back(&prof->impl_->mutex);
  }
  std::sort(param_mutexes.begin(), param_mutexes.end());
  param_mutexes.erase(std::unique(param_mutexes.begin(), param_mutexes.end()),
                      param_mutexes.end());
  std::sort(profile_mutexes.begin(), profile_mutexes.end());
  profile_mutexes.erase(
      std::unique(profile_mutexes.begin(), profile_mutexes.end()),
      profile_mutexes.end());
  std::vector<std::unique_lock<std::mutex>> parameter_locks;
  std::vector<std::unique_lock<std::mutex>> profile_locks;
  parameter_locks.reserve(param_mutexes.size());
  profile_locks.reserve(profile_mutexes.size());
  for (auto *m : param_mutexes) {
    parameter_locks.emplace_back(*m);
  }
  for (auto *m : profile_mutexes) {
    profile_locks.emplace_back(*m);
  }

  auto &state = *impl_->state;
  GpuTiming timing;
  std::uint64_t event_value = 0;

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
  if (all_metal4) {
    if (@available(macOS 26.0, *)) {
      std::scoped_lock submission_lock(state.submission_mutex);
      @autoreleasepool {
        std::vector<id<MTLResidencySet>> residency_sets;
        residency_sets.reserve(profiles.size());
        for (auto *prof : profiles) {
          residency_sets.push_back(prof->impl_->residency);
        }
        const auto submitted = submit_metal4_work(
            state, @"omlx_m5.replay_icb_batch", residency_sets, false, false,
            false, "Metal 4 ICB batch command buffer failed",
            "Metal 4 ICB batch replay timed out",
            "Metal 4 ICB batch replay failed: ",
            [&](id<MTL4CommandBuffer> command_buffer) {
              const MTLStages queue_stages = MTLStageDispatch | MTLStageBlit;
              for (auto *prof : profiles) {
                const auto len = prof->impl_->plan->descriptor.dispatches.size();
                if (len == 0) continue;
                id<MTL4ComputeCommandEncoder> encoder =
                    [command_buffer computeCommandEncoder];
                if (encoder == nil) {
                  throw std::runtime_error("Metal 4 ICB batch encoder failed");
                }
                [encoder barrierAfterQueueStages:queue_stages
                                    beforeStages:queue_stages
                               visibilityOptions:MTL4VisibilityOptionDevice];
                [encoder executeCommandsInBuffer:prof->impl_->commands
                                       withRange:NSMakeRange(0, len)];
                [encoder barrierAfterStages:queue_stages
                          beforeQueueStages:queue_stages
                          visibilityOptions:MTL4VisibilityOptionDevice];
                [encoder endEncoding];
              }
            });
        timing = submitted.timing;
      }
    } else {
      throw std::runtime_error("Metal 4 ICB batch requires macOS 26+");
    }
  } else
#endif
  {
    @autoreleasepool {
      const auto submitted = submit_classic_work(
          state, nil, false, "classic ICB batch command buffer failed",
          "classic ICB batch replay failed: ",
          [&](id<MTLCommandBuffer> command_buffer) {
            for (auto *prof : profiles) {
              const auto len = prof->impl_->plan->descriptor.dispatches.size();
              if (len == 0) continue;
              id<MTLComputeCommandEncoder> encoder =
                  [command_buffer computeCommandEncoder];
              if (encoder == nil) {
                throw std::runtime_error("classic ICB batch encoder failed");
              }
              for (const auto &buffer : prof->impl_->retained_buffers) {
                [encoder useResource:buffer
                               usage:MTLResourceUsageRead |
                                     MTLResourceUsageWrite];
              }
              [encoder executeCommandsInBuffer:prof->impl_->commands
                                     withRange:NSMakeRange(0, len)];
              [encoder endEncoding];
            }
          });
      timing = submitted.timing;
      event_value = submitted.event_value;
    }
  }
  for (auto *prof : profiles) {
    ++prof->impl_->stats.replay_count;
  }
  {
    std::scoped_lock lock(state.mutex);
    ++state.stats.command_buffers_submitted;
    if (all_metal4) {
      ++state.stats.metal4_command_buffers_submitted;
    } else {
      ++state.stats.classic_command_buffers_submitted;
      state.stats.last_shared_event_value =
          std::max(state.stats.last_shared_event_value, event_value);
    }
    state.stats.dispatches_submitted += total_dispatches;
    state.stats.icb_profile_replays += profiles.size();
  }
  return {.timing = timing,
          .shared_event_value = event_value,
          .dispatch_count = total_dispatches};
}

MetalIcbCanaryReport
MetalContext::run_icb_canary(const std::string &stage_a_pipeline_key,
                             const std::string &stage_b_pipeline_key,
                             MetalIcbCanaryOptions options) const {
  if (options.iterations == 0 || options.element_count == 0) {
    throw std::invalid_argument("ICB canary dimensions must be positive");
  }

  auto &state = *impl_->state;
  std::shared_ptr<PipelineEntry> stage_a_pipeline;
  std::shared_ptr<PipelineEntry> stage_b_pipeline;
  {
    std::scoped_lock lock(state.mutex);
    const auto stage_a = state.pipelines.find(stage_a_pipeline_key);
    const auto stage_b = state.pipelines.find(stage_b_pipeline_key);
    if (stage_a == state.pipelines.end() || stage_b == state.pipelines.end()) {
      throw std::invalid_argument("ICB canary pipeline is not warmed");
    }
    stage_a_pipeline = stage_a->second;
    stage_b_pipeline = stage_b->second;
  }

  const auto valid_canary_reflection = [](const PipelineEntry &pipeline) {
    if (pipeline.info.bindings.size() != 3) {
      return false;
    }
    for (std::uint32_t index = 0; index < 3; ++index) {
      const auto &binding = pipeline.info.bindings[index];
      if (binding.index != index || binding.kind != "buffer") {
        return false;
      }
    }
    return true;
  };
  if (!valid_canary_reflection(*stage_a_pipeline) ||
      !valid_canary_reflection(*stage_b_pipeline)) {
    throw std::invalid_argument(
        "ICB canary pipeline reflection does not match the fixed contract");
  }

  const auto payload_bytes = checked_multiply(
      options.element_count, sizeof(std::uint32_t), "ICB canary payload");
  const auto stage_a_threads =
      std::max<std::uint64_t>(1, stage_a_pipeline->info.thread_execution_width);
  const auto stage_b_threads =
      std::max<std::uint64_t>(1, stage_b_pipeline->info.thread_execution_width);
  const auto stage_a_plan = prepare_command_plan({
      .key = "b4.0.icb.stage_a." + std::to_string(options.element_count),
      .dispatches = {{
          .pipeline_key = stage_a_pipeline_key,
          .grid = {.width = options.element_count},
          .threads_per_threadgroup = {.width = stage_a_threads},
          .resources =
              {
                  {.argument_index = 0,
                   .resource_slot = 0,
                   .minimum_bytes = payload_bytes},
                  {.argument_index = 1,
                   .resource_slot = 1,
                   .minimum_bytes = payload_bytes},
                  {.argument_index = 2,
                   .resource_slot = 2,
                   .minimum_bytes = sizeof(IcbCanaryParameters)},
              },
      }},
  });
  const auto stage_b_plan = prepare_command_plan({
      .key = "b4.0.icb.stage_b." + std::to_string(options.element_count),
      .dispatches = {{
          .pipeline_key = stage_b_pipeline_key,
          .grid = {.width = options.element_count},
          .threads_per_threadgroup = {.width = stage_b_threads},
          .resources =
              {
                  {.argument_index = 0,
                   .resource_slot = 0,
                   .minimum_bytes = payload_bytes},
                  {.argument_index = 1,
                   .resource_slot = 1,
                   .minimum_bytes = payload_bytes},
                  {.argument_index = 2,
                   .resource_slot = 2,
                   .minimum_bytes = sizeof(IcbCanaryParameters)},
              },
      }},
  });

  MetalIcbCanaryReport report{
      .compute_icb_eligible = state.capabilities.profile.compute_icb_eligible,
      .metal4_icb_eligible = state.capabilities.profile.metal4_icb_eligible,
      .pipeline_icb_enabled =
          stage_a_pipeline->info.indirect_command_buffers_enabled &&
          stage_b_pipeline->info.indirect_command_buffers_enabled,
      .metal4_pipeline_icb_enabled =
          stage_a_pipeline->info.metal4_indirect_command_buffers_enabled &&
          stage_b_pipeline->info.metal4_indirect_command_buffers_enabled,
      .dependency_barrier_encoded = !options.omit_dependency_barrier,
      .bit_exact = true,
      .iterations = options.iterations,
  };
  bool use_icb = report.compute_icb_eligible && report.pipeline_icb_enabled;
  if (options.force_execute_batch_fallback) {
    use_icb = false;
    report.fallback_reason = "forced_execute_batch_fallback";
  } else if (options.omit_dependency_barrier) {
    use_icb = false;
    report.fallback_reason = "dependency_barrier_missing";
  } else if (!report.compute_icb_eligible) {
    use_icb = false;
    report.fallback_reason =
        state.capabilities.profile.icb_fallback_reasons.empty()
            ? "compute_icb_unavailable"
            : state.capabilities.profile.icb_fallback_reasons.front();
  } else if (!report.pipeline_icb_enabled) {
    use_icb = false;
    report.fallback_reason = "pipeline_icb_support_disabled";
  }

  const auto before_resources = registry_stats();
  {
    std::vector<std::uint32_t> source_values(options.element_count);
    for (std::uint32_t index = 0; index < options.element_count; ++index) {
      source_values[index] = index * 2'654'435'761U + 17U;
    }
    auto source = upload(std::as_bytes(std::span(source_values)));
    auto replay_middle = make_buffer(payload_bytes);
    auto replay_output = make_buffer(payload_bytes);
    auto oracle_middle = make_buffer(payload_bytes);
    auto oracle_output = make_buffer(payload_bytes);
    auto parameters = make_buffer(sizeof(IcbCanaryParameters));

    const std::array replay_invocations = {
        MetalCommandInvocation{
            .plan = &stage_a_plan,
            .bindings = {{0, &source, 0},
                         {1, &replay_middle, 0},
                         {2, &parameters, 0}},
        },
        MetalCommandInvocation{
            .plan = &stage_b_plan,
            .bindings = {{0, &replay_middle, 0},
                         {1, &replay_output, 0},
                         {2, &parameters, 0}},
        },
    };
    const std::array oracle_invocations = {
        MetalCommandInvocation{
            .plan = &stage_a_plan,
            .bindings = {{0, &source, 0},
                         {1, &oracle_middle, 0},
                         {2, &parameters, 0}},
        },
        MetalCommandInvocation{
            .plan = &stage_b_plan,
            .bindings = {{0, &oracle_middle, 0},
                         {1, &oracle_output, 0},
                         {2, &parameters, 0}},
        },
    };

    id<MTLIndirectCommandBuffer> indirect_commands = nil;
    id<MTLResidencySet> residency_set = nil;
    bool use_metal4_outer =
        use_icb && options.prefer_metal4_outer && report.metal4_icb_eligible;
    if (use_metal4_outer && !report.metal4_pipeline_icb_enabled) {
      use_icb = false;
      use_metal4_outer = false;
      report.fallback_reason = "metal4_pipeline_icb_support_disabled";
    }
    id<MTLComputePipelineState> selected_stage_a_pipeline =
        use_metal4_outer ? stage_a_pipeline->metal4_pipeline
                         : stage_a_pipeline->pipeline;
    id<MTLComputePipelineState> selected_stage_b_pipeline =
        use_metal4_outer ? stage_b_pipeline->metal4_pipeline
                         : stage_b_pipeline->pipeline;
    if (use_icb) {
      @autoreleasepool {
        const auto record_start = std::chrono::steady_clock::now();
        MTLIndirectCommandBufferDescriptor *descriptor =
            [[MTLIndirectCommandBufferDescriptor alloc] init];
        descriptor.commandTypes =
            MTLIndirectCommandTypeConcurrentDispatchThreads;
        descriptor.inheritPipelineState = NO;
        descriptor.inheritBuffers = NO;
        descriptor.maxKernelBufferBindCount = 3;
        indirect_commands = [state.device
            newIndirectCommandBufferWithDescriptor:descriptor
                                   maxCommandCount:2
                                           options:
                                               MTLResourceStorageModeShared |
                                               MTLResourceHazardTrackingModeTracked];
        if (indirect_commands == nil) {
          use_icb = false;
          use_metal4_outer = false;
          report.fallback_reason = "compute_icb_creation_failed";
        } else {
          id<MTLIndirectComputeCommand> stage_a =
              [indirect_commands indirectComputeCommandAtIndex:0];
          id<MTLIndirectComputeCommand> stage_b =
              [indirect_commands indirectComputeCommandAtIndex:1];
          if (stage_a == nil || stage_b == nil ||
              ![stage_b respondsToSelector:@selector(setBarrier)]) {
            use_icb = false;
            use_metal4_outer = false;
            report.fallback_reason =
                "compute_icb_command_or_barrier_unavailable";
          } else {
            [stage_a setComputePipelineState:selected_stage_a_pipeline];
            [stage_a setKernelBuffer:source.impl_->buffer offset:0 atIndex:0];
            [stage_a setKernelBuffer:replay_middle.impl_->buffer
                              offset:0
                             atIndex:1];
            [stage_a setKernelBuffer:parameters.impl_->buffer
                              offset:0
                             atIndex:2];
            [stage_a
                concurrentDispatchThreads:make_mtl_size(
                                              {.width = options.element_count},
                                              "ICB stage A grid")
                    threadsPerThreadgroup:make_mtl_size(
                                              {.width = stage_a_threads},
                                              "ICB stage A threadgroup")];

            [stage_b setComputePipelineState:selected_stage_b_pipeline];
            [stage_b setKernelBuffer:replay_middle.impl_->buffer
                              offset:0
                             atIndex:0];
            [stage_b setKernelBuffer:replay_output.impl_->buffer
                              offset:0
                             atIndex:1];
            [stage_b setKernelBuffer:parameters.impl_->buffer
                              offset:0
                             atIndex:2];
            [stage_b setBarrier];
            [stage_b
                concurrentDispatchThreads:make_mtl_size(
                                              {.width = options.element_count},
                                              "ICB stage B grid")
                    threadsPerThreadgroup:make_mtl_size(
                                              {.width = stage_b_threads},
                                              "ICB stage B threadgroup")];
            report.icb_recordings = 1;
          }
        }
        report.icb_record_cpu_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - record_start)
                .count();
      }
    }

    if (use_icb) {
      @autoreleasepool {
        MTLResidencySetDescriptor *descriptor =
            [[MTLResidencySetDescriptor alloc] init];
        descriptor.label = @"omlx_m5.b4_0.icb_canary";
        descriptor.initialCapacity = 9;
        NSError *error = nil;
        residency_set = [state.device newResidencySetWithDescriptor:descriptor
                                                              error:&error];
        if (residency_set == nil) {
          use_icb = false;
          use_metal4_outer = false;
          report.fallback_reason = "residency_set_creation_failed:" +
                                   ns_string([error localizedDescription]);
        } else {
          const id<MTLAllocation> allocations[] = {
              source.impl_->buffer,        replay_middle.impl_->buffer,
              replay_output.impl_->buffer, oracle_middle.impl_->buffer,
              oracle_output.impl_->buffer, parameters.impl_->buffer,
              indirect_commands,           selected_stage_a_pipeline,
              selected_stage_b_pipeline,
          };
          [residency_set addAllocations:allocations count:9];
          [residency_set commit];
        }
      }
    }

    report.execution_route =
        use_icb
            ? (use_metal4_outer ? "metal4_compute_icb" : "classic_compute_icb")
            : "execute_batch_fallback";

    auto execute_classic_icb = [&]() -> GpuTiming {
      @autoreleasepool {
        const auto submitted = submit_classic_work(
            state, @"omlx_m5.b4_0.classic_icb_replay", true,
            "classic ICB outer encoder allocation failed",
            "classic compute ICB replay failed: ",
            [&](id<MTLCommandBuffer> command_buffer) {
              const auto encode_start = std::chrono::steady_clock::now();
              id<MTLComputeCommandEncoder> encoder =
                  [command_buffer computeCommandEncoder];
              if (encoder == nil) {
                std::scoped_lock lock(state.mutex);
                ++state.stats.submission_failures;
                throw std::runtime_error(
                    "classic ICB outer encoder allocation failed");
              }
              [encoder useResource:source.impl_->buffer
                             usage:MTLResourceUsageRead];
              [encoder useResource:replay_middle.impl_->buffer
                             usage:MTLResourceUsageRead |
                                   MTLResourceUsageWrite];
              [encoder useResource:replay_output.impl_->buffer
                             usage:MTLResourceUsageWrite];
              [encoder useResource:parameters.impl_->buffer
                             usage:MTLResourceUsageRead];
              [encoder executeCommandsInBuffer:indirect_commands
                                     withRange:NSMakeRange(0, 2)];
              [encoder endEncoding];
              report.replay_cpu_encode_ms +=
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - encode_start)
                      .count();
            });
        {
          std::scoped_lock lock(state.mutex);
          ++state.stats.command_buffers_submitted;
          ++state.stats.classic_command_buffers_submitted;
          state.stats.dispatches_submitted += 2;
          state.stats.last_shared_event_value =
              std::max(state.stats.last_shared_event_value,
                       submitted.event_value);
        }
        return submitted.timing;
      }
    };

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    auto execute_metal4_icb = [&]() -> GpuTiming {
      if (@available(macOS 26.0, *)) {
        std::scoped_lock submission_lock(state.submission_mutex);
        @autoreleasepool {
          const id<MTLResidencySet> residency_arr[] = {residency_set};
          const auto submitted = submit_metal4_work(
              state, @"omlx_m5.b4_0.metal4_icb_replay",
              std::span(residency_arr, 1), false, false, true,
              "Metal 4 command buffer allocation failed",
              "Metal 4 ICB replay timed out",
              "Metal 4 compute ICB replay failed: ",
              [&](id<MTL4CommandBuffer> command_buffer) {
                const auto encode_start = std::chrono::steady_clock::now();
                id<MTL4ComputeCommandEncoder> encoder =
                    [command_buffer computeCommandEncoder];
                if (encoder == nil) {
                  throw std::runtime_error(
                      "Metal 4 compute encoder allocation failed");
                }
                [encoder executeCommandsInBuffer:indirect_commands
                                       withRange:NSMakeRange(0, 2)];
                [encoder endEncoding];
                report.replay_cpu_encode_ms +=
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - encode_start)
                        .count();
              });
          {
            std::scoped_lock lock(state.mutex);
            ++state.stats.command_buffers_submitted;
            ++state.stats.metal4_command_buffers_submitted;
            state.stats.dispatches_submitted += 2;
          }
          return submitted.timing;
        }
      }
      throw std::runtime_error(
          "Metal 4 ICB route selected on an older runtime");
    };
#endif

    std::vector<std::byte> previous_output;
    std::uint64_t changed_outputs = 0;
    constexpr std::uint64_t fnv_offset = 1'469'598'103'934'665'603ULL;
    constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
    report.output_checksum = fnv_offset;
    for (std::uint64_t iteration = 0; iteration < options.iterations;
         ++iteration) {
      const IcbCanaryParameters current{
          .addend = static_cast<std::uint32_t>(iteration * 17U + 3U),
          .multiplier = static_cast<std::uint32_t>((iteration % 31U) * 2U + 1U),
          .xor_mask = static_cast<std::uint32_t>(0x9E3779B9U ^ iteration),
          .element_count = options.element_count,
      };
      parameters.write(std::as_bytes(std::span(&current, 1)));
      ++report.parameter_updates;

      if (use_icb) {
        GpuTiming replay_timing;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
        if (use_metal4_outer) {
          replay_timing = execute_metal4_icb();
        } else
#endif
        {
          replay_timing = execute_classic_icb();
        }
        ++report.icb_execute_calls;
        ++report.replay_command_buffer_commits;
        report.replay_dispatches += 2;
        report.replay_wall_ms += replay_timing.wall_ms;
        report.replay_gpu_ms += replay_timing.gpu_ms;
      } else {
        const auto call_start = std::chrono::steady_clock::now();
        const auto replay_submission = execute_batch(replay_invocations);
        const auto call_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - call_start)
                                 .count();
        report.replay_cpu_encode_ms +=
            std::max(0.0, call_ms - replay_submission.timing.wall_ms);
        report.replay_wall_ms += replay_submission.timing.wall_ms;
        report.replay_gpu_ms += replay_submission.timing.gpu_ms;
        ++report.replay_command_buffer_commits;
        report.replay_dispatches += replay_submission.dispatch_count;
      }

      const auto oracle_call_start = std::chrono::steady_clock::now();
      const auto oracle_submission = execute_batch(oracle_invocations);
      const auto oracle_call_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - oracle_call_start)
              .count();
      report.oracle_cpu_encode_ms +=
          std::max(0.0, oracle_call_ms - oracle_submission.timing.wall_ms);
      report.oracle_wall_ms += oracle_submission.timing.wall_ms;
      report.oracle_gpu_ms += oracle_submission.timing.gpu_ms;
      ++report.oracle_execute_batch_commits;
      report.oracle_dispatches += oracle_submission.dispatch_count;

      std::vector<std::uint32_t> expected(options.element_count);
      for (std::uint32_t index = 0; index < options.element_count; ++index) {
        expected[index] =
            ((source_values[index] + current.addend) * current.multiplier) ^
            current.xor_mask;
      }
      const auto expected_span = std::as_bytes(std::span(expected));
      const std::vector<std::byte> expected_bytes(expected_span.begin(),
                                                  expected_span.end());
      const auto replay_bytes = replay_output.read();
      const auto oracle_bytes = oracle_output.read();
      report.bit_exact = report.bit_exact && replay_bytes == oracle_bytes &&
                         replay_bytes == expected_bytes;
      ++report.compared_iterations;
      if (!previous_output.empty() && previous_output != replay_bytes) {
        ++changed_outputs;
      }
      previous_output = replay_bytes;
      for (const auto value : replay_bytes) {
        report.output_checksum ^=
            static_cast<std::uint64_t>(std::to_integer<unsigned char>(value));
        report.output_checksum *= fnv_prime;
      }
    }
    report.changing_parameters_observed =
        options.iterations == 1 || changed_outputs == options.iterations - 1;

    if (indirect_commands != nil) {
      [indirect_commands resetWithRange:NSMakeRange(0, 2)];
      report.icb_resets = 1;
    }
    if (residency_set != nil) {
      [residency_set endResidency];
    }
  }
  const auto after_resources = registry_stats();
  report.resources_released = after_resources.current_buffer_count ==
                                  before_resources.current_buffer_count &&
                              after_resources.current_buffer_bytes ==
                                  before_resources.current_buffer_bytes;
  return report;
}

MetalRegistryStats MetalContext::registry_stats() const {
  auto &state = *impl_->state;
  MetalRegistryStats result;
  {
    std::scoped_lock lock(state.mutex);
    result = state.stats;
  }
  state.ledger->fill(result);
  return result;
}

GpuTiming MetalContext::submit_barrier() const {
  auto &state = *impl_->state;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
  if (state.capabilities.profile.metal4_direct_eligible) {
    if (@available(macOS 26.0, *)) {
      std::scoped_lock submission_lock(state.submission_mutex);
      @autoreleasepool {
        const auto start = std::chrono::steady_clock::now();
        const auto submitted = submit_metal4_work(
            state, @"omlx_m5.metal4.barrier", {}, true, true, true,
            "Metal 4 barrier command buffer allocation failed",
            "Metal 4 barrier timed out", "Metal 4 barrier failed: ",
            [&](id<MTL4CommandBuffer> command_buffer) {
              id<MTL4ComputeCommandEncoder> encoder =
                  [command_buffer computeCommandEncoder];
              if (encoder == nil) {
                std::scoped_lock lock(state.mutex);
                ++state.stats.submission_failures;
                throw std::runtime_error(
                    "Metal 4 barrier compute encoder allocation failed");
              }
              const MTLStages queue_stages = MTLStageDispatch | MTLStageBlit;
              [encoder barrierAfterQueueStages:queue_stages
                                  beforeStages:queue_stages
                             visibilityOptions:MTL4VisibilityOptionDevice];
              [encoder endEncoding];
            });
        if (![state.shared_event waitUntilSignaledValue:submitted.event_value
                                              timeoutMS:30000]) {
          std::scoped_lock lock(state.mutex);
          ++state.stats.submission_failures;
          throw std::runtime_error("Metal 4 barrier timed out");
        }
        auto timing = submitted.timing;
        timing.wall_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - start)
                             .count();
        {
          std::scoped_lock lock(state.mutex);
          ++state.stats.command_buffers_submitted;
          ++state.stats.metal4_command_buffers_submitted;
          state.stats.last_shared_event_value =
              std::max(state.stats.last_shared_event_value,
                       submitted.event_value);
        }
        return timing;
      }
    }
  }
#endif
  @autoreleasepool {
    const auto submitted = submit_classic_work(
        state, nil, true, "Metal barrier allocation failed",
        "Metal barrier failed: ", [&](id<MTLCommandBuffer> command_buffer) {
          id<MTLBlitCommandEncoder> encoder =
              [command_buffer blitCommandEncoder];
          if (encoder == nil) {
            std::scoped_lock lock(state.mutex);
            ++state.stats.submission_failures;
            throw std::runtime_error("Metal barrier allocation failed");
          }
          [encoder endEncoding];
        });
    std::string async_err;
    {
      std::scoped_lock lock(state.mutex);
      ++state.stats.command_buffers_submitted;
      ++state.stats.classic_command_buffers_submitted;
      state.stats.last_shared_event_value =
          std::max(state.stats.last_shared_event_value, submitted.event_value);
      if (!state.last_classic_async_error.empty()) {
        async_err = std::move(state.last_classic_async_error);
        state.last_classic_async_error.clear();
      }
    }
    if (!async_err.empty()) {
      throw std::runtime_error("Metal classic async submission failed: " +
                               async_err);
    }
    return submitted.timing;
  }
}

GpuTiming MetalContext::submit_classic_barrier() const {
  auto &state = *impl_->state;
  @autoreleasepool {
    const auto submitted = submit_classic_work(
        state, @"omlx_m5.classic.barrier", true,
        "Metal classic barrier allocation failed",
        "Metal classic barrier failed: ",
        [&](id<MTLCommandBuffer> command_buffer) {
          id<MTLBlitCommandEncoder> encoder =
              [command_buffer blitCommandEncoder];
          if (encoder == nil) {
            std::scoped_lock lock(state.mutex);
            ++state.stats.submission_failures;
            throw std::runtime_error("Metal classic barrier allocation failed");
          }
          [encoder endEncoding];
        });
    std::string async_err;
    {
      std::scoped_lock lock(state.mutex);
      ++state.stats.command_buffers_submitted;
      ++state.stats.classic_command_buffers_submitted;
      state.stats.last_shared_event_value =
          std::max(state.stats.last_shared_event_value, submitted.event_value);
      if (!state.last_classic_async_error.empty()) {
        async_err = std::move(state.last_classic_async_error);
        state.last_classic_async_error.clear();
      }
    }
    if (!async_err.empty()) {
      throw std::runtime_error("Metal classic async submission failed: " +
                               async_err);
    }
    return submitted.timing;
  }
}

void *MetalContext::native_mtl_device() const noexcept {
  return (impl_ == nullptr || impl_->state == nullptr)
             ? nullptr
             : (__bridge void *)impl_->state->device;
}

void *MetalContext::native_mtl_command_queue() const noexcept {
  return (impl_ == nullptr || impl_->state == nullptr)
             ? nullptr
             : (__bridge void *)impl_->state->queue;
}

std::uint64_t MetalContext::encode_into_command_buffer(
    void *mtl_command_buffer,
    std::span<const MetalCommandInvocation> invocations) const {
  if (mtl_command_buffer == nullptr) {
    throw std::invalid_argument("Metal command buffer pointer is null");
  }
  if (invocations.empty()) {
    return 0;
  }
  std::vector<Impl::ResourceBindingMap> invocation_resources;
  invocation_resources.reserve(invocations.size());
  std::set<const MetalParameterTable::Impl *> locked_parameter_tables;
  std::vector<std::unique_lock<std::mutex>> parameter_locks;
  parameter_locks.reserve(invocations.size());
  std::uint64_t total_dispatches = 0;
  for (const auto &invocation : invocations) {
    if (invocation.plan == nullptr || invocation.plan->impl_ == nullptr ||
        invocation.plan->impl_->owner.get() != impl_->state.get()) {
      throw std::invalid_argument(
          "Metal command plan belongs to another context");
    }
    const auto requires_parameters =
        std::ranges::any_of(invocation.plan->impl_->descriptor.dispatches,
                            [](const MetalDispatchDescriptor &dispatch) {
                              return !dispatch.dynamic_arguments.empty();
                            });
    if (requires_parameters &&
        (invocation.parameters == nullptr ||
         invocation.parameters->impl_ == nullptr ||
         invocation.parameters->impl_->owner.get() != impl_->state.get())) {
      throw std::invalid_argument(
          "Metal parameter table is missing or foreign");
    }
    if (requires_parameters &&
        locked_parameter_tables.insert(invocation.parameters->impl_.get())
            .second) {
      parameter_locks.emplace_back(invocation.parameters->impl_->mutex);
    }
    auto resources = Impl::validate_dispatch_bindings(
        *impl_->state->ledger, *invocation.plan->impl_, invocation.bindings,
        requires_parameters ? &invocation.parameters->impl_->buffer : nullptr,
        "Metal encode", true);
    total_dispatches += invocation.plan->impl_->descriptor.dispatches.size();
    invocation_resources.push_back(std::move(resources));
  }

  auto &state = *impl_->state;
  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer =
        (__bridge id<MTLCommandBuffer>)mtl_command_buffer;
    for (std::size_t invocation_index = 0;
         invocation_index < invocations.size(); ++invocation_index) {
      const auto &invocation = invocations[invocation_index];
      const auto &resources = invocation_resources[invocation_index];
      for (std::size_t index = 0;
           index < invocation.plan->impl_->descriptor.dispatches.size();
           ++index) {
        const auto &dispatch =
            invocation.plan->impl_->descriptor.dispatches[index];
        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];
        if (encoder == nil) {
          std::scoped_lock lock(state.mutex);
          ++state.stats.submission_failures;
          throw std::runtime_error("Metal compute encoder allocation failed");
        }
        [encoder setComputePipelineState:invocation.plan->impl_
                                             ->pipelines[index]
                                             ->pipeline];
        for (const auto &resource : dispatch.resources) {
          const auto *selected = resources.at(resource.resource_slot);
          [encoder setBuffer:selected->buffer->impl_->buffer
                      offset:checked_ns_integer(selected->offset,
                                                "buffer offset")
                     atIndex:resource.argument_index];
        }
        for (const auto &argument :
             invocation.plan->impl_->stable_arguments[index]) {
          [encoder setBuffer:argument.buffer.impl_->buffer
                      offset:0
                     atIndex:argument.argument_index];
        }
        for (const auto &argument : dispatch.dynamic_arguments) {
          [encoder setBuffer:invocation.parameters->impl_->buffer.impl_->buffer
                      offset:checked_ns_integer(argument.parameter_offset,
                                                "parameter offset")
                     atIndex:argument.argument_index];
        }
        if (dispatch.kind == MetalDispatchKind::threads) {
          [encoder dispatchThreads:make_mtl_size(dispatch.grid, "dispatch grid")
              threadsPerThreadgroup:make_mtl_size(
                                        dispatch.threads_per_threadgroup,
                                        "threads per threadgroup")];
        } else if (dispatch.kind == MetalDispatchKind::threadgroups_indirect) {
          const auto *ind =
              resources.at(dispatch.indirect_grid_resource_slot);
          const auto ind_off = checked_ns_integer(
              checked_add(ind->offset, dispatch.indirect_grid_offset,
                          "indirect grid offset"),
              "indirect grid offset");
          [encoder dispatchThreadgroupsWithIndirectBuffer:ind->buffer->impl_
                                                              ->buffer
                                     indirectBufferOffset:ind_off
                                    threadsPerThreadgroup:
                                        make_mtl_size(
                                            dispatch.threads_per_threadgroup,
                                            "threads per threadgroup")];
        } else {
          [encoder
               dispatchThreadgroups:make_mtl_size(dispatch.grid,
                                                  "dispatch grid")
              threadsPerThreadgroup:make_mtl_size(
                                        dispatch.threads_per_threadgroup,
                                        "threads per threadgroup")];
        }
        [encoder endEncoding];
      }
    }
  }
  {
    std::scoped_lock lock(state.mutex);
    state.stats.dispatches_submitted += total_dispatches;
  }
  return total_dispatches;
}

std::uint64_t MetalContext::submit_classic_async(
    std::span<const MetalCommandInvocation> invocations) const {
  if (invocations.empty()) {
    return 0;
  }
  auto &state = *impl_->state;
  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
      std::scoped_lock lock(state.mutex);
      ++state.stats.submission_failures;
      throw std::runtime_error("Metal async command buffer allocation failed");
    }
    command_buffer.label = to_ns_string(
        invocations.size() == 1
            ? invocations.front().plan->impl_->descriptor.key + ".async"
            : "omlx_m5.classic_async." + std::to_string(invocations.size()));
    const auto dispatches =
        encode_into_command_buffer((__bridge void *)command_buffer, invocations);
    auto state_shared = impl_->state;
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
      if (completed.status == MTLCommandBufferStatusError) {
        std::scoped_lock lock(state_shared->mutex);
        ++state_shared->stats.submission_failures;
        if (state_shared->last_classic_async_error.empty()) {
          state_shared->last_classic_async_error =
              ns_string([completed.error localizedDescription]);
        }
      }
    }];
    {
      std::scoped_lock submission_lock(state.submission_mutex);
      [command_buffer commit];
    }
    {
      std::scoped_lock lock(state.mutex);
      ++state.stats.command_buffers_submitted;
      ++state.stats.classic_command_buffers_submitted;
    }
    return dispatches;
  }
}

} // namespace omlx::m5
