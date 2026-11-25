#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "omlx_m5/hw_counters.h"
#include "omlx_m5/quantization.h"

#include <mach/mach_time.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dispatch/dispatch.h>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifndef __MAC_OS_X_VERSION_MAX_ALLOWED
#define __MAC_OS_X_VERSION_MAX_ALLOWED 0
#endif

namespace omlx::m5 {

namespace {

std::string ns_str(NSString *value) {
  return value == nil ? std::string{} : std::string([value UTF8String]);
}

NSString *to_ns_str(std::string_view value) {
  std::string owned(value);
  NSString *result = [NSString stringWithUTF8String:owned.c_str()];
  if (result == nil) {
    throw std::invalid_argument("invalid UTF-8 string");
  }
  return result;
}

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
struct Metal4CounterFeedbackState {
  dispatch_semaphore_t completed = dispatch_semaphore_create(0);
  NSError *error = nil;
  CFTimeInterval gpu_start = 0.0;
  CFTimeInterval gpu_end = 0.0;
};

struct Metal4CommitTiming {
  double wall_ns = 0.0;
  double cb_gpu_ns = 0.0;
};

API_AVAILABLE(macos(26.0))
Metal4CommitTiming commit_and_wait_metal4_counter_cb(
    id<MTL4CommandQueue> queue, id<MTL4CommandAllocator> allocator,
    id<MTL4CommandBuffer> cb, id<MTLSharedEvent> shared_ev,
    std::uint64_t ev_val, std::string_view label) {
  auto fb_state = std::make_shared<Metal4CounterFeedbackState>();
  MTL4CommitOptions *commit_opts = [[MTL4CommitOptions alloc] init];
  [commit_opts addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
    fb_state->error = fb.error;
    fb_state->gpu_start = fb.GPUStartTime;
    fb_state->gpu_end = fb.GPUEndTime;
    dispatch_semaphore_signal(fb_state->completed);
  }];
  const auto t0 = std::chrono::steady_clock::now();
  const id<MTL4CommandBuffer> cbs[] = {cb};
  [queue commit:cbs count:1 options:commit_opts];
  if (shared_ev != nil) {
    [queue signalEvent:shared_ev value:ev_val];
  }
  if (dispatch_semaphore_wait(
          fb_state->completed,
          dispatch_time(DISPATCH_TIME_NOW, 30LL * NSEC_PER_SEC)) != 0) {
    // Do not reset an in-flight MTL4CommandAllocator after a wait timeout.
    throw std::runtime_error(std::string(label) + " timed out");
  }
  const double wall_ns = std::chrono::duration<double, std::nano>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
  [allocator reset];
  if (fb_state->error != nil) {
    throw std::runtime_error(std::string(label) + " failed: " +
                             ns_str([fb_state->error localizedDescription]));
  }
  const double cb_gpu_ns =
      fb_state->gpu_end > fb_state->gpu_start
          ? (fb_state->gpu_end - fb_state->gpu_start) * 1.0e9
          : 0.0;
  return {.wall_ns = wall_ns, .cb_gpu_ns = cb_gpu_ns};
}
#endif

double mach_ticks_to_ns(std::uint64_t ticks) {
  static const mach_timebase_info_data_t info = [] {
    mach_timebase_info_data_t result{};
    mach_timebase_info(&result);
    return result;
  }();
  if (info.denom == 0) {
    return static_cast<double>(ticks);
  }
  return static_cast<double>(ticks) * static_cast<double>(info.numer) /
         static_cast<double>(info.denom);
}

double percentile_sorted(std::span<const double> sorted, double pct) {
  if (sorted.empty()) {
    return 0.0;
  }
  if (sorted.size() == 1) {
    return sorted.front();
  }
  const double rank =
      std::clamp(pct, 0.0, 100.0) / 100.0 * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(rank));
  const auto hi = static_cast<std::size_t>(std::ceil(rank));
  const double frac = rank - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

std::string lower_ascii(std::string_view input) {
  std::string out(input);
  for (char &ch : out) {
    if (ch >= 'A' && ch <= 'Z') {
      ch = static_cast<char>(ch - 'A' + 'a');
    }
  }
  return out;
}

std::string classify_counter_set_name(NSString *raw_name) {
  if (raw_name == nil) {
    return "unknown";
  }
  if ([raw_name isEqualToString:MTLCommonCounterSetTimestamp]) {
    return "timestamp";
  }
  if ([raw_name isEqualToString:MTLCommonCounterSetStageUtilization]) {
    return "stage_utilization";
  }
  if ([raw_name isEqualToString:MTLCommonCounterSetStatistic]) {
    return "statistic";
  }
  const auto lowered = lower_ascii(ns_str(raw_name));
  if (lowered.find("timestamp") != std::string::npos) {
    return "timestamp";
  }
  if (lowered.find("utilization") != std::string::npos) {
    return "stage_utilization";
  }
  if (lowered.find("statistic") != std::string::npos) {
    return "statistic";
  }
  return "vendor";
}

id<MTLCounterSet> find_counter_set(id<MTLDevice> device,
                                   std::string_view canonical_kind) {
  if (device == nil || device.counterSets == nil) {
    return nil;
  }
  for (id<MTLCounterSet> cs in device.counterSets) {
    if (classify_counter_set_name(cs.name) == canonical_kind) {
      return cs;
    }
  }
  return nil;
}

struct CounterBufferAllocation {
  id<MTLCounterSampleBuffer> sample_buffer = nil;
  id<MTLBuffer> resolve_buffer = nil;
  bool shared_mode = false;
  NSUInteger sample_count = 0;
};

CounterBufferAllocation
allocate_sample_buffer(id<MTLDevice> device, id<MTLCounterSet> counter_set,
                       NSUInteger sample_count, std::string_view label,
                       std::size_t resolved_stride_bytes) {
  CounterBufferAllocation out{};
  if (device == nil || counter_set == nil || sample_count == 0) {
    return out;
  }
  out.sample_count = sample_count;
  MTLCounterSampleBufferDescriptor *desc =
      [[MTLCounterSampleBufferDescriptor alloc] init];
  desc.counterSet = counter_set;
  desc.sampleCount = sample_count;
  desc.label = to_ns_str(label);
  desc.storageMode = MTLStorageModeShared;
  NSError *error = nil;
  out.sample_buffer = [device newCounterSampleBufferWithDescriptor:desc
                                                             error:&error];
  if (out.sample_buffer != nil) {
    out.shared_mode = true;
    return out;
  }
  error = nil;
  desc.storageMode = MTLStorageModePrivate;
  out.sample_buffer = [device newCounterSampleBufferWithDescriptor:desc
                                                             error:&error];
  if (out.sample_buffer != nil) {
    out.shared_mode = false;
    const NSUInteger total_bytes =
        std::max<NSUInteger>(256U, sample_count * resolved_stride_bytes);
    out.resolve_buffer =
        [device newBufferWithLength:total_bytes
                            options:MTLResourceStorageModeShared |
                                    MTLResourceHazardTrackingModeTracked];
  }
  return out;
}

std::vector<std::byte>
resolve_sample_buffer(id<MTLCommandQueue> queue,
                      const CounterBufferAllocation &alloc, NSRange range,
                      std::size_t expected_entry_bytes) {
  if (alloc.sample_buffer == nil || range.length == 0) {
    return {};
  }
  if (alloc.shared_mode) {
    NSData *data = [alloc.sample_buffer resolveCounterRange:range];
    if (data != nil && data.length >= range.length * expected_entry_bytes) {
      const auto *ptr = static_cast<const std::byte *>(data.bytes);
      return std::vector<std::byte>(ptr, ptr + data.length);
    }
  }
  if (queue == nil) {
    return {};
  }
  id<MTLDevice> device = queue.device;
  id<MTLBuffer> dst = alloc.resolve_buffer;
  const NSUInteger needed = range.length * expected_entry_bytes;
  if (dst == nil || dst.length < needed) {
    dst = [device newBufferWithLength:std::max<NSUInteger>(256U, needed)
                              options:MTLResourceStorageModeShared |
                                      MTLResourceHazardTrackingModeTracked];
  }
  if (dst == nil) {
    return {};
  }
  std::memset(dst.contents, 0xFF, dst.length);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  if (cb == nil || blit == nil) {
    return {};
  }
  [blit resolveCounters:alloc.sample_buffer
                inRange:range
      destinationBuffer:dst
      destinationOffset:0];
  [blit endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.status == MTLCommandBufferStatusError) {
    return {};
  }
  const auto *ptr = static_cast<const std::byte *>(dst.contents);
  return std::vector<std::byte>(ptr, ptr + needed);
}

std::optional<std::uint64_t> valid_counter_u64(std::uint64_t raw) {
  if (raw == MTLCounterErrorValue) {
    return std::nullopt;
  }
  return raw;
}

double choose_resolved_timestamp_scale(std::uint64_t raw_delta,
                                       double cb_gpu_ns,
                                       double empirical_tick_to_ns,
                                       std::uint64_t metal4_freq_hz) {
  if (raw_delta == 0) {
    return 1.0;
  }
  const double as_ns = static_cast<double>(raw_delta);
  if (cb_gpu_ns > 100.0) {
    const double ratio_1 = as_ns / cb_gpu_ns;
    if (ratio_1 >= 0.05 && ratio_1 <= 3.0) {
      return 1.0;
    }
    if (metal4_freq_hz > 0) {
      const double freq_scale = 1.0e9 / static_cast<double>(metal4_freq_hz);
      const double ratio_f = (as_ns * freq_scale) / cb_gpu_ns;
      if (ratio_f >= 0.05 && ratio_f <= 3.0) {
        return freq_scale;
      }
    }
    if (empirical_tick_to_ns > 0.0) {
      const double ratio_e = (as_ns * empirical_tick_to_ns) / cb_gpu_ns;
      if (ratio_e >= 0.05 && ratio_e <= 3.0) {
        return empirical_tick_to_ns;
      }
    }
  }
  return 1.0;
}

} // namespace

DistributionStats compute_distribution_stats(std::span<const double> samples) {
  DistributionStats stats{};
  if (samples.empty()) {
    return stats;
  }
  std::vector<double> sorted(samples.begin(), samples.end());
  std::ranges::sort(sorted);
  stats.count = sorted.size();
  stats.min = sorted.front();
  stats.max = sorted.back();
  stats.p50 = percentile_sorted(sorted, 50.0);
  stats.p90 = percentile_sorted(sorted, 90.0);
  stats.p99 = percentile_sorted(sorted, 99.0);
  const double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
  stats.mean = sum / static_cast<double>(sorted.size());
  if (sorted.size() > 1) {
    double sq_diff = 0.0;
    for (const double val : sorted) {
      const double d = val - stats.mean;
      sq_diff += d * d;
    }
    stats.stddev =
        std::sqrt(sq_diff / static_cast<double>(sorted.size() - 1));
  }
  if (std::abs(stats.mean) > 1e-12) {
    stats.cv = stats.stddev / std::abs(stats.mean);
  }
  return stats;
}

HwCounterConfig HwCounterConfig::from_environment() {
  HwCounterConfig cfg{};
  const char *env = std::getenv("OMLX_M5_HW_COUNTERS");
  if (env != nullptr) {
    const std::string_view val(env);
    cfg.enabled = (val == "1" || val == "true" || val == "on" || val == "ON");
  }
  return cfg;
}

class HwCounterCollector::Impl {
public:
  mutable std::mutex mutex;
  std::vector<HwPassCounterSample> records;
};

HwCounterCollector::HwCounterCollector(HwCounterConfig config)
    : config_(config), enabled_(config.enabled) {
  if (enabled_) {
    impl_ = std::make_unique<Impl>();
    impl_->records.reserve(std::min<std::uint32_t>(config_.max_samples, 1024U));
  }
}

HwCounterCollector::HwCounterCollector(HwCounterCollector &&) noexcept = default;
HwCounterCollector &
HwCounterCollector::operator=(HwCounterCollector &&) noexcept = default;
HwCounterCollector::~HwCounterCollector() = default;

void HwCounterCollector::record_slow(const HwPassCounterSample &sample) {
  if (!impl_) {
    return;
  }
  std::scoped_lock lock(impl_->mutex);
  if (impl_->records.size() < config_.max_samples) {
    impl_->records.push_back(sample);
  }
}

void HwCounterCollector::clear() {
  if (!impl_) {
    return;
  }
  std::scoped_lock lock(impl_->mutex);
  impl_->records.clear();
}

std::size_t HwCounterCollector::sample_count() const {
  if (!enabled_ || !impl_) {
    return 0;
  }
  std::scoped_lock lock(impl_->mutex);
  return impl_->records.size();
}

std::vector<HwPassCounterSample> HwCounterCollector::samples() const {
  if (!enabled_ || !impl_) {
    return {};
  }
  std::scoped_lock lock(impl_->mutex);
  return impl_->records;
}

std::string classify_kernel_family(std::string_view stage,
                                   std::string_view library,
                                   std::string_view function,
                                   std::string_view route_family) {
  const auto st = lower_ascii(stage);
  const auto lib = lower_ascii(library);
  const auto fn = lower_ascii(function);
  const auto rf = lower_ascii(route_family);

  if (lib.find("moe") != std::string::npos ||
      fn.find("moe") != std::string::npos ||
      fn.find("expert") != std::string::npos ||
      fn.find("router") != std::string::npos) {
    return "moe";
  }
  if (lib.find("attention") != std::string::npos ||
      fn.find("attention") != std::string::npos ||
      fn.find("rope") != std::string::npos ||
      fn.find("kv_page") != std::string::npos ||
      st.find(".attention") != std::string::npos) {
    return "attention";
  }
  if (lib.find("gdn") != std::string::npos ||
      fn.find("gdn") != std::string::npos ||
      st.find(".gdn") != std::string::npos) {
    return "gdn";
  }
  if (lib.find("rmsnorm") != std::string::npos ||
      fn.find("rmsnorm") != std::string::npos ||
      fn.find("normalize") != std::string::npos) {
    return "rmsnorm_residual";
  }
  if (lib.find("vocab") != std::string::npos ||
      lib.find("position_sampling") != std::string::npos ||
      fn.find("sample_logits") != std::string::npos ||
      st == "embedding" || st == "sample") {
    return "vocab_sampling";
  }
  if (lib.find("nax") != std::string::npos ||
      lib.find("tensorops") != std::string::npos ||
      fn.find("nax") != std::string::npos ||
      rf.find("nax") != std::string::npos ||
      rf.find("tensorops") != std::string::npos ||
      rf.find("w4a4") != std::string::npos) {
    return "nax_gemm";
  }
  if (lib.find("qmv") != std::string::npos ||
      lib.find("qmm") != std::string::npos ||
      lib.find("mxfp4_small_t") != std::string::npos ||
      lib.find("row_exact") != std::string::npos ||
      lib.find("affine_lane") != std::string::npos ||
      lib.find("mlp") != std::string::npos ||
      st.find(".mlp") != std::string::npos || st == "head") {
    return "simd_gemv_gemm";
  }
  if (fn.find("copy") != std::string::npos ||
      fn.find("blit") != std::string::npos ||
      fn.find("slice") != std::string::npos ||
      fn.find("last_row") != std::string::npos) {
    return "blit_copy_sync";
  }
  return "other_compute";
}

std::string classify_bottleneck(double arithmetic_intensity,
                                double mean_gpu_ns,
                                double effective_gbs,
                                double nominal_bandwidth_gbs) {
  if (mean_gpu_ns > 0.0 && mean_gpu_ns < 8'000.0 &&
      effective_gbs < 0.20 * nominal_bandwidth_gbs) {
    return "launch_latency_bound";
  }
  // On M5 Ultra 64-core (~1.2 TB/s DRAM, ~60+ TFLOPS BF16/FP4 NAX), ridge point
  // is roughly ~40-60 FLOPs/byte.
  if (arithmetic_intensity >= 40.0 &&
      effective_gbs < 0.55 * nominal_bandwidth_gbs) {
    return "compute_bound";
  }
  if (effective_gbs >= 0.35 * nominal_bandwidth_gbs ||
      arithmetic_intensity < 24.0) {
    return "memory_bandwidth_bound";
  }
  return "mixed_compute_memory";
}

HwPassAggregateStats
summarize_pass_samples(std::string_view label,
                       std::span<const HwPassCounterSample> samples) {
  HwPassAggregateStats out{};
  out.label = std::string(label);
  if (samples.empty()) {
    return out;
  }
  out.family = samples.front().family;
  out.pass_kind = samples.front().pass_kind;
  out.counter_source = samples.front().counter_source;
  out.dispatch_count_per_iter = samples.front().dispatch_count;
  out.bytes_per_iter = samples.front().bytes_transferred;
  out.flops_per_iter = samples.front().flops;
  if (out.bytes_per_iter > 0 && out.flops_per_iter > 0.0) {
    out.arithmetic_intensity_flops_per_byte =
        out.flops_per_iter / static_cast<double>(out.bytes_per_iter);
  }

  std::vector<double> gpu_vals;
  std::vector<double> gap_vals;
  std::vector<double> wall_vals;
  std::vector<double> cb_gpu_vals;
  std::vector<double> cycles_vals;
  std::vector<double> inv_vals;
  gpu_vals.reserve(samples.size());
  gap_vals.reserve(samples.size());
  wall_vals.reserve(samples.size());
  cb_gpu_vals.reserve(samples.size());

  for (const auto &s : samples) {
    gpu_vals.push_back(s.gpu_ns);
    gap_vals.push_back(s.dispatch_gap_ns);
    wall_vals.push_back(s.wall_ns);
    cb_gpu_vals.push_back(s.command_buffer_gpu_ns);
    if (s.total_cycles.has_value()) {
      cycles_vals.push_back(static_cast<double>(*s.total_cycles));
    }
    if (s.compute_kernel_invocations.has_value()) {
      inv_vals.push_back(static_cast<double>(*s.compute_kernel_invocations));
    }
  }

  out.gpu_ns = compute_distribution_stats(gpu_vals);
  out.dispatch_gap_ns = compute_distribution_stats(gap_vals);
  out.wall_ns = compute_distribution_stats(wall_vals);
  out.command_buffer_gpu_ns = compute_distribution_stats(cb_gpu_vals);
  if (!cycles_vals.empty()) {
    out.total_cycles = compute_distribution_stats(cycles_vals);
  }
  if (!inv_vals.empty()) {
    out.compute_kernel_invocations = compute_distribution_stats(inv_vals);
  }

  if (out.gpu_ns.mean > 0.0 && out.bytes_per_iter > 0) {
    out.mean_effective_gbs =
        static_cast<double>(out.bytes_per_iter) / out.gpu_ns.mean;
  }
  if (out.gpu_ns.p50 > 0.0 && out.bytes_per_iter > 0) {
    out.p50_effective_gbs =
        static_cast<double>(out.bytes_per_iter) / out.gpu_ns.p50;
  }
  if (out.gpu_ns.mean > 0.0 && out.flops_per_iter > 0.0) {
    out.mean_effective_tflops = (out.flops_per_iter / out.gpu_ns.mean) * 1.0e-3;
  }
  if (out.gpu_ns.p50 > 0.0 && out.flops_per_iter > 0.0) {
    out.p50_effective_tflops = (out.flops_per_iter / out.gpu_ns.p50) * 1.0e-3;
  }
  out.mean_host_overhead_ns =
      std::max(0.0, out.wall_ns.mean - out.gpu_ns.mean);
  out.bottleneck_class = classify_bottleneck(
      out.arithmetic_intensity_flops_per_byte, out.gpu_ns.mean,
      out.mean_effective_gbs);
  return out;
}

std::vector<HwFamilyBreakdownEntry>
summarize_family_breakdown(std::span<const HwPassCounterSample> samples) {
  std::map<std::string, HwFamilyBreakdownEntry> grouped;
  double grand_total_gpu_ns = 0.0;
  for (const auto &s : samples) {
    const std::string key = s.family.empty() ? "other_compute" : s.family;
    auto &entry = grouped[key];
    entry.family = key;
    ++entry.sample_count;
    entry.total_dispatches += s.dispatch_count;
    entry.total_gpu_ns += s.gpu_ns;
    entry.total_bytes += static_cast<double>(s.bytes_transferred);
    entry.total_flops += s.flops;
    grand_total_gpu_ns += s.gpu_ns;
  }
  std::vector<HwFamilyBreakdownEntry> out;
  out.reserve(grouped.size());
  for (auto &[_, entry] : grouped) {
    if (grand_total_gpu_ns > 0.0) {
      entry.share_of_gpu_pct = 100.0 * entry.total_gpu_ns / grand_total_gpu_ns;
    }
    if (entry.total_gpu_ns > 0.0 && entry.total_bytes > 0.0) {
      entry.mean_effective_gbs = entry.total_bytes / entry.total_gpu_ns;
    }
    if (entry.total_gpu_ns > 0.0 && entry.total_flops > 0.0) {
      entry.mean_effective_tflops =
          (entry.total_flops / entry.total_gpu_ns) * 1.0e-3;
    }
    out.push_back(std::move(entry));
  }
  std::ranges::sort(out, [](const auto &a, const auto &b) {
    return a.total_gpu_ns > b.total_gpu_ns;
  });
  return out;
}

std::vector<HwPassAggregateStats>
HwCounterCollector::summarize_by_label() const {
  const auto all = samples();
  std::map<std::string, std::vector<HwPassCounterSample>> by_label;
  std::vector<std::string> order;
  for (const auto &s : all) {
    if (!by_label.contains(s.label)) {
      order.push_back(s.label);
    }
    by_label[s.label].push_back(s);
  }
  std::vector<HwPassAggregateStats> result;
  result.reserve(order.size());
  for (const auto &key : order) {
    result.push_back(summarize_pass_samples(key, by_label[key]));
  }
  return result;
}

std::vector<HwFamilyBreakdownEntry>
HwCounterCollector::summarize_by_family() const {
  const auto all = samples();
  return summarize_family_breakdown(all);
}

HwOverheadCheck measure_hw_counter_cpu_overhead(std::uint64_t iterations) {
  HwOverheadCheck check{};
  check.iterations = std::max<std::uint64_t>(1000ULL, iterations);

  HwCounterCollector disabled_collector(HwCounterConfig::disabled());
  volatile std::uint64_t sink = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (std::uint64_t i = 0; i < check.iterations; ++i) {
    disabled_collector.record_lazy([&] {
      sink = sink + 1;
      return HwPassCounterSample{.label = "noop", .gpu_ns = 1.0};
    });
  }
  const auto t1 = std::chrono::steady_clock::now();
  check.disabled_ns_per_call =
      std::chrono::duration<double, std::nano>(t1 - t0).count() /
      static_cast<double>(check.iterations);
  check.zero_allocation_when_disabled =
      (disabled_collector.sample_count() == 0 && sink == 0);

  HwCounterConfig enabled_cfg{};
  enabled_cfg.enabled = true;
  enabled_cfg.max_samples = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(check.iterations, 16'384ULL));
  HwCounterCollector enabled_collector(enabled_cfg);
  HwPassCounterSample prototype{
      .label = "overhead_probe",
      .family = "nax_gemm",
      .pass_kind = "metal4_compute",
      .counter_source = "metal4_counter_heap",
      .gpu_ns = 1000.0,
  };
  const auto e0 = std::chrono::steady_clock::now();
  for (std::uint64_t i = 0; i < enabled_cfg.max_samples; ++i) {
    enabled_collector.record(prototype);
  }
  const auto e1 = std::chrono::steady_clock::now();
  check.enabled_record_ns_per_call =
      std::chrono::duration<double, std::nano>(e1 - e0).count() /
      static_cast<double>(enabled_cfg.max_samples);
  return check;
}

HwCounterDeviceCapabilities enumerate_hw_counter_capabilities() {
  HwCounterDeviceCapabilities caps{};
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      throw std::runtime_error("Metal default device unavailable");
    }
    caps.device_name = ns_str([device name]);
    caps.recommended_working_set_bytes =
        static_cast<std::uint64_t>([device recommendedMaxWorkingSetSize]);
    caps.max_buffer_length = static_cast<std::uint64_t>([device maxBufferLength]);
    if (@available(macOS 14.0, *)) {
      caps.architecture_name = ns_str([device architecture].name);
    }
    const auto ver = [NSProcessInfo processInfo].operatingSystemVersion;
    caps.runtime_os = std::to_string(ver.majorVersion) + "." +
                      std::to_string(ver.minorVersion) + "." +
                      std::to_string(ver.patchVersion);

    for (std::uint32_t family = 12; family >= 1; --family) {
      if ([device supportsFamily:static_cast<MTLGPUFamily>(1000 + family)]) {
        caps.apple_gpu_family = family;
        break;
      }
    }

    caps.sampling_points.at_stage_boundary =
        [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
    caps.sampling_points.at_draw_boundary =
        [device supportsCounterSampling:MTLCounterSamplingPointAtDrawBoundary];
    caps.sampling_points.at_dispatch_boundary = [device
        supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
    caps.sampling_points.at_tile_dispatch_boundary = [device
        supportsCounterSampling:MTLCounterSamplingPointAtTileDispatchBoundary];
    caps.sampling_points.at_blit_boundary =
        [device supportsCounterSampling:MTLCounterSamplingPointAtBlitBoundary];

    NSArray<id<MTLCounterSet>> *sets = device.counterSets;
    caps.metal3_counter_sets_available = (sets != nil && sets.count > 0);
    if (sets != nil) {
      for (id<MTLCounterSet> cs in sets) {
        HwCounterSetInfo info{};
        info.name = ns_str(cs.name);
        info.canonical_kind = classify_counter_set_name(cs.name);
        for (id<MTLCounter> counter in cs.counters) {
          info.counters.push_back(ns_str(counter.name));
        }
        MTLCounterSampleBufferDescriptor *desc =
            [[MTLCounterSampleBufferDescriptor alloc] init];
        desc.counterSet = cs;
        desc.sampleCount = 4;
        desc.label = @"omlx_m5.hw_counter.cap_probe";
        desc.storageMode = MTLStorageModeShared;
        NSError *err = nil;
        id<MTLCounterSampleBuffer> shared_buf =
            [device newCounterSampleBufferWithDescriptor:desc error:&err];
        info.shared_sample_buffer_supported = (shared_buf != nil);
        err = nil;
        desc.storageMode = MTLStorageModePrivate;
        id<MTLCounterSampleBuffer> priv_buf =
            [device newCounterSampleBufferWithDescriptor:desc error:&err];
        info.private_sample_buffer_supported = (priv_buf != nil);

        const bool usable = info.shared_sample_buffer_supported ||
                            info.private_sample_buffer_supported;
        if (usable && info.canonical_kind == "timestamp") {
          caps.timestamp_counter_supported = true;
        } else if (usable && info.canonical_kind == "stage_utilization") {
          caps.stage_utilization_supported = true;
        } else if (usable && info.canonical_kind == "statistic") {
          caps.statistic_counter_supported = true;
        }
        caps.counter_sets.push_back(std::move(info));
      }
    }

    MTLTimestamp cpu0 = 0;
    MTLTimestamp gpu0 = 0;
    const std::uint64_t mach0 = mach_absolute_time();
    [device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    MTLTimestamp cpu1 = 0;
    MTLTimestamp gpu1 = 0;
    const std::uint64_t mach1 = mach_absolute_time();
    [device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
    caps.sampled_cpu_timestamp_ns = static_cast<std::uint64_t>(cpu1);
    caps.sampled_gpu_timestamp = static_cast<std::uint64_t>(gpu1);
    const double host_elapsed_ns = mach_ticks_to_ns(mach1 - mach0);
    double empirical_gpu_tick_ns = 1.0;
    if (gpu1 > gpu0 && host_elapsed_ns > 100'000.0) {
      empirical_gpu_tick_ns =
          host_elapsed_ns / static_cast<double>(gpu1 - gpu0);
    }

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) {
      if ([device respondsToSelector:@selector(queryTimestampFrequency)] &&
          [device respondsToSelector:@selector(newCounterHeapWithDescriptor:error:)]) {
        caps.metal4_timestamp_frequency_hz = [device queryTimestampFrequency];
        caps.metal4_timestamp_entry_bytes =
            [device sizeOfCounterHeapEntry:MTL4CounterHeapTypeTimestamp];
        MTL4CounterHeapDescriptor *heap_desc =
            [[MTL4CounterHeapDescriptor alloc] init];
        heap_desc.type = MTL4CounterHeapTypeTimestamp;
        heap_desc.count = 8;
        NSError *heap_err = nil;
        id<MTL4CounterHeap> heap =
            [device newCounterHeapWithDescriptor:heap_desc error:&heap_err];
        caps.metal4_counter_heap_supported =
            (heap != nil && caps.metal4_timestamp_frequency_hz > 0);
        if (caps.metal4_timestamp_frequency_hz > 0) {
          caps.metal4_heap_tick_to_ns_scale =
              1.0e9 / static_cast<double>(caps.metal4_timestamp_frequency_hz);
          caps.timestamp_calibration_source = "metal4_query_timestamp_frequency";
        }
      }
    }
#endif
    if (caps.timestamp_calibration_source == "uncalibrated") {
      caps.metal4_heap_tick_to_ns_scale = empirical_gpu_tick_ns;
      caps.timestamp_calibration_source = "sample_timestamps_mach_correlation";
    }
    // Resolved MTLCounterResultTimestamp values from resolveCounterRange: are
    // already normalized to nanoseconds by Metal on Apple Silicon; verified
    // dynamically on first sampled pass.
    caps.resolved_timestamp_to_ns_scale = 1.0;
  }
  return caps;
}

HwPassAggregateStats
profile_blit_copy_counters(std::uint64_t bytes, std::uint32_t warmups,
                           std::uint32_t iterations, bool private_storage,
                           std::uint32_t copies_per_pass) {
  if (bytes == 0 || iterations == 0 || copies_per_pass == 0) {
    throw std::invalid_argument("invalid blit copy counter probe arguments");
  }
  const std::string label =
      std::string("blit_copy_") + (private_storage ? "private_" : "shared_") +
      std::to_string(bytes / (1024ULL * 1024ULL)) + "mb_x" +
      std::to_string(copies_per_pass);

  std::vector<HwPassCounterSample> samples;
  samples.reserve(iterations);

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (device == nil || queue == nil) {
      throw std::runtime_error("Metal device/queue unavailable for blit probe");
    }
    const auto caps = enumerate_hw_counter_capabilities();
    const MTLResourceOptions opts =
        (private_storage ? MTLResourceStorageModePrivate
                         : MTLResourceStorageModeShared) |
        MTLResourceHazardTrackingModeTracked;
    id<MTLBuffer> src = [device newBufferWithLength:bytes options:opts];
    id<MTLBuffer> dst = [device newBufferWithLength:bytes options:opts];
    if (src == nil || dst == nil) {
      throw std::runtime_error("Metal blit buffer allocation failed");
    }

    // Initialize source buffer so pages are faulted and non-zero.
    {
      id<MTLCommandBuffer> init_cb = [queue commandBuffer];
      id<MTLBlitCommandEncoder> init_blit = [init_cb blitCommandEncoder];
      [init_blit fillBuffer:src range:NSMakeRange(0, bytes) value:0x5AU];
      [init_blit fillBuffer:dst range:NSMakeRange(0, bytes) value:0x00U];
      [init_blit endEncoding];
      [init_cb commit];
      [init_cb waitUntilCompleted];
    }

    const std::uint32_t total_runs = warmups + iterations;
    const bool can_sample_blit = caps.sampling_points.at_blit_boundary ||
                                 caps.sampling_points.at_stage_boundary;
    id<MTLCounterSet> ts_set =
        can_sample_blit ? find_counter_set(device, "timestamp") : nil;
    CounterBufferAllocation ts_buf = allocate_sample_buffer(
        device, ts_set, total_runs * 2U, "omlx_m5.blit.ts",
        sizeof(MTLCounterResultTimestamp));

    double ts_scale = 1.0;
    bool ts_scale_calibrated = false;

    for (std::uint32_t run = 0; run < total_runs; ++run) {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = nil;
      bool attached_ts = false;
      if (ts_buf.sample_buffer != nil) {
        MTLBlitPassDescriptor *pass = [MTLBlitPassDescriptor blitPassDescriptor];
        pass.sampleBufferAttachments[0].sampleBuffer = ts_buf.sample_buffer;
        pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = run * 2U;
        pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = run * 2U + 1U;
        blit = [cb blitCommandEncoderWithDescriptor:pass];
        attached_ts = (blit != nil);
      }
      if (blit == nil) {
        blit = [cb blitCommandEncoder];
      }
      for (std::uint32_t c = 0; c < copies_per_pass; ++c) {
        [blit copyFromBuffer:src
                sourceOffset:0
                    toBuffer:dst
           destinationOffset:0
                        size:bytes];
      }
      [blit endEncoding];
      const auto t0 = std::chrono::steady_clock::now();
      [cb commit];
      [cb waitUntilCompleted];
      const double wall_ns =
          std::chrono::duration<double, std::nano>(
              std::chrono::steady_clock::now() - t0)
              .count();
      if (cb.status == MTLCommandBufferStatusError) {
        throw std::runtime_error("Metal blit copy pass failed");
      }
      const double cb_gpu_ns =
          cb.GPUEndTime > cb.GPUStartTime
              ? (cb.GPUEndTime - cb.GPUStartTime) * 1.0e9
              : 0.0;

      double pass_gpu_ns = cb_gpu_ns;
      std::string source = "command_buffer_feedback";
      if (attached_ts) {
        const auto raw = resolve_sample_buffer(
            queue, ts_buf, NSMakeRange(run * 2U, 2U),
            sizeof(MTLCounterResultTimestamp));
        if (raw.size() >= 2U * sizeof(MTLCounterResultTimestamp)) {
          MTLCounterResultTimestamp s0{}, s1{};
          std::memcpy(&s0, raw.data(), sizeof(s0));
          std::memcpy(&s1, raw.data() + sizeof(s0), sizeof(s1));
          if (s0.timestamp != MTLCounterErrorValue &&
              s1.timestamp != MTLCounterErrorValue &&
              s1.timestamp >= s0.timestamp) {
            const std::uint64_t delta = s1.timestamp - s0.timestamp;
            if (!ts_scale_calibrated && delta > 0) {
              ts_scale = choose_resolved_timestamp_scale(
                  delta, cb_gpu_ns, caps.metal4_heap_tick_to_ns_scale,
                  caps.metal4_timestamp_frequency_hz);
              ts_scale_calibrated = true;
            }
            pass_gpu_ns = static_cast<double>(delta) * ts_scale;
            source = "mtl_counter_sample_buffer";
          }
        }
      }

      if (run >= warmups) {
        samples.push_back(HwPassCounterSample{
            .label = label,
            .family = "blit_copy_sync",
            .pass_kind = "blit",
            .counter_source = source,
            .gpu_ns = pass_gpu_ns,
            .dispatch_gap_ns = std::max(0.0, cb_gpu_ns - pass_gpu_ns),
            .wall_ns = wall_ns,
            .command_buffer_gpu_ns = cb_gpu_ns,
            .bytes_transferred = bytes * 2ULL * copies_per_pass,
            .flops = 0.0,
            .dispatch_count = copies_per_pass,
        });
      }
    }
  }
  return summarize_pass_samples(label, samples);
}

HwPassAggregateStats
profile_compute_copy_counters(const std::filesystem::path &smoke_metallib,
                              std::uint64_t bytes, std::uint32_t warmups,
                              std::uint32_t iterations, bool use_metal4,
                              std::uint32_t dispatches_per_encoder) {
  if (bytes == 0 || bytes % sizeof(std::uint32_t) != 0 ||
      bytes / sizeof(std::uint32_t) > std::numeric_limits<std::uint32_t>::max() ||
      iterations == 0 || dispatches_per_encoder == 0) {
    throw std::invalid_argument("invalid compute copy counter probe arguments");
  }
  const auto elements = static_cast<std::uint32_t>(bytes / sizeof(std::uint32_t));
  const std::string label =
      std::string(use_metal4 ? "metal4_compute_copy_" : "classic_compute_copy_") +
      std::to_string(bytes / (1024ULL * 1024ULL)) + "mb_x" +
      std::to_string(dispatches_per_encoder);

  std::vector<HwPassCounterSample> samples;
  samples.reserve(iterations);

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      throw std::runtime_error("Metal device unavailable");
    }
    const auto caps = enumerate_hw_counter_capabilities();
    NSError *lib_err = nil;
    NSURL *url = [NSURL fileURLWithPath:to_ns_str(smoke_metallib.string())];
    id<MTLLibrary> lib = [device newLibraryWithURL:url error:&lib_err];
    if (lib == nil) {
      throw std::runtime_error("failed to load smoke metallib: " +
                               ns_str([lib_err localizedDescription]));
    }

    id<MTLBuffer> src =
        [device newBufferWithLength:bytes
                            options:MTLResourceStorageModeShared |
                                    MTLResourceHazardTrackingModeTracked];
    id<MTLBuffer> dst =
        [device newBufferWithLength:bytes
                            options:MTLResourceStorageModeShared |
                                    MTLResourceHazardTrackingModeTracked];
    id<MTLBuffer> count_buf =
        [device newBufferWithLength:sizeof(std::uint32_t)
                            options:MTLResourceStorageModeShared |
                                    MTLResourceHazardTrackingModeTracked];
    if (src == nil || dst == nil || count_buf == nil) {
      throw std::runtime_error("compute copy buffer allocation failed");
    }
    std::memset(src.contents, 0x3CU, bytes);
    std::memset(dst.contents, 0x00U, bytes);
    std::memcpy(count_buf.contents, &elements, sizeof(elements));

    const std::uint32_t total_runs = warmups + iterations;

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (use_metal4 && caps.metal4_counter_heap_supported) {
      if (@available(macOS 26.0, *)) {
        id<MTL4CommandQueue> q4 = [device newMTL4CommandQueue];
        id<MTL4CommandAllocator> alloc4 = [device newCommandAllocator];
        MTL4CompilerDescriptor *comp_desc = [[MTL4CompilerDescriptor alloc] init];
        NSError *err = nil;
        id<MTL4Compiler> compiler =
            [device newCompilerWithDescriptor:comp_desc error:&err];
        MTL4LibraryFunctionDescriptor *fn_desc =
            [[MTL4LibraryFunctionDescriptor alloc] init];
        fn_desc.name = @"omlx_m5_copy_u32";
        fn_desc.library = lib;
        MTL4ComputePipelineDescriptor *pipe_desc =
            [[MTL4ComputePipelineDescriptor alloc] init];
        pipe_desc.computeFunctionDescriptor = fn_desc;
        id<MTLComputePipelineState> pso =
            [compiler newComputePipelineStateWithDescriptor:pipe_desc
                                        compilerTaskOptions:nil
                                                      error:&err];
        MTL4ArgumentTableDescriptor *at_desc =
            [[MTL4ArgumentTableDescriptor alloc] init];
        at_desc.maxBufferBindCount = 3;
        at_desc.initializeBindings = YES;
        id<MTL4ArgumentTable> arg_table =
            [device newArgumentTableWithDescriptor:at_desc error:&err];
        MTLResidencySetDescriptor *res_desc =
            [[MTLResidencySetDescriptor alloc] init];
        id<MTLResidencySet> residency =
            [device newResidencySetWithDescriptor:res_desc error:&err];
        MTL4CounterHeapDescriptor *heap_desc =
            [[MTL4CounterHeapDescriptor alloc] init];
        heap_desc.type = MTL4CounterHeapTypeTimestamp;
        const NSUInteger timestamps_per_run = dispatches_per_encoder * 2U;
        heap_desc.count = total_runs * timestamps_per_run;
        id<MTL4CounterHeap> heap =
            [device newCounterHeapWithDescriptor:heap_desc error:&err];
        id<MTLSharedEvent> shared_ev = [device newSharedEvent];

        if (q4 != nil && alloc4 != nil && pso != nil && arg_table != nil &&
            residency != nil && heap != nil && shared_ev != nil) {
          [residency addAllocation:src];
          [residency addAllocation:dst];
          [residency addAllocation:count_buf];
          [residency addAllocation:pso];
          [residency commit];
          [q4 addResidencySet:residency];

          const NSUInteger tg_w = std::min<NSUInteger>(256U, pso.maxTotalThreadsPerThreadgroup);
          std::uint64_t ev_val = 0;

          for (std::uint32_t run = 0; run < total_runs; ++run) {
            id<MTL4CommandBuffer> cb = [device newCommandBuffer];
            [cb beginCommandBufferWithAllocator:alloc4];
            id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [arg_table setAddress:src.gpuAddress atIndex:0];
            [arg_table setAddress:dst.gpuAddress atIndex:1];
            [arg_table setAddress:count_buf.gpuAddress atIndex:2];
            [enc setArgumentTable:arg_table];

            const NSUInteger base = run * timestamps_per_run;
            for (std::uint32_t d = 0; d < dispatches_per_encoder; ++d) {
              if (d > 0) {
                [enc barrierAfterEncoderStages:MTLStageDispatch
                           beforeEncoderStages:MTLStageDispatch
                             visibilityOptions:MTL4VisibilityOptionDevice];
              }
              [enc writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                        intoHeap:heap
                                         atIndex:base + d * 2U];
              [enc barrierAfterEncoderStages:(MTLStageDispatch | MTLStageBlit)
                         beforeEncoderStages:MTLStageDispatch
                           visibilityOptions:MTL4VisibilityOptionNone];
              [enc dispatchThreads:MTLSizeMake(elements, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg_w, 1, 1)];
              [enc barrierAfterEncoderStages:MTLStageDispatch
                         beforeEncoderStages:(MTLStageDispatch | MTLStageBlit)
                           visibilityOptions:MTL4VisibilityOptionNone];
              [enc writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                        intoHeap:heap
                                         atIndex:base + d * 2U + 1U];
            }
            [enc endEncoding];
            [cb endCommandBuffer];

            ++ev_val;
            const auto timing = commit_and_wait_metal4_counter_cb(
                q4, alloc4, cb, shared_ev, ev_val, "Metal 4 compute copy");
            const double wall_ns = timing.wall_ns;
            const double cb_gpu_ns = timing.cb_gpu_ns;

            double kernel_gpu_ns = cb_gpu_ns;
            double gap_ns = 0.0;
            NSData *resolved = [heap
                resolveCounterRange:NSMakeRange(base, timestamps_per_run)];
            if (resolved != nil &&
                resolved.length >=
                    timestamps_per_run * sizeof(MTL4TimestampHeapEntry)) {
              const auto *entries =
                  static_cast<const MTL4TimestampHeapEntry *>(resolved.bytes);
              double sum_kernel_ns = 0.0;
              double sum_gap_ns = 0.0;
              for (std::uint32_t d = 0; d < dispatches_per_encoder; ++d) {
                const auto ts0 = entries[d * 2U].timestamp;
                const auto ts1 = entries[d * 2U + 1U].timestamp;
                if (ts1 >= ts0) {
                  sum_kernel_ns += static_cast<double>(ts1 - ts0) *
                                   caps.metal4_heap_tick_to_ns_scale;
                }
                if (d > 0) {
                  const auto prev_end = entries[(d - 1U) * 2U + 1U].timestamp;
                  if (ts0 >= prev_end) {
                    sum_gap_ns += static_cast<double>(ts0 - prev_end) *
                                  caps.metal4_heap_tick_to_ns_scale;
                  }
                }
              }
              if (sum_kernel_ns > 0.0) {
                kernel_gpu_ns = sum_kernel_ns;
                gap_ns = sum_gap_ns;
              }
            }

            if (run >= warmups) {
              samples.push_back(HwPassCounterSample{
                  .label = label,
                  .family = "blit_copy_sync",
                  .pass_kind = "metal4_compute",
                  .counter_source = "metal4_counter_heap",
                  .gpu_ns = kernel_gpu_ns,
                  .dispatch_gap_ns = gap_ns,
                  .wall_ns = wall_ns,
                  .command_buffer_gpu_ns = cb_gpu_ns,
                  .bytes_transferred = bytes * 2ULL * dispatches_per_encoder,
                  .flops = 0.0,
                  .dispatch_count = dispatches_per_encoder,
              });
            }
          }
          return summarize_pass_samples(label, samples);
        }
      }
    }
#endif

    // Classic compute pass with MTLCounterSampleBuffer attachments
    // (Timestamp + StageUtilization + Statistic where supported).
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLFunction> fn = [lib newFunctionWithName:@"omlx_m5_copy_u32"];
    NSError *pso_err = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&pso_err];
    if (queue == nil || pso == nil) {
      throw std::runtime_error("classic compute copy pipeline failed");
    }

    const bool can_stage = caps.sampling_points.at_stage_boundary;
    id<MTLCounterSet> ts_set =
        can_stage ? find_counter_set(device, "timestamp") : nil;
    id<MTLCounterSet> util_set =
        can_stage ? find_counter_set(device, "stage_utilization") : nil;
    id<MTLCounterSet> stat_set =
        can_stage ? find_counter_set(device, "statistic") : nil;

    CounterBufferAllocation ts_buf = allocate_sample_buffer(
        device, ts_set, total_runs * 2U, "omlx_m5.compute.ts",
        sizeof(MTLCounterResultTimestamp));
    CounterBufferAllocation util_buf = allocate_sample_buffer(
        device, util_set, total_runs * 2U, "omlx_m5.compute.util",
        sizeof(MTLCounterResultStageUtilization));
    CounterBufferAllocation stat_buf = allocate_sample_buffer(
        device, stat_set, total_runs * 2U, "omlx_m5.compute.stat",
        sizeof(MTLCounterResultStatistic));

    const NSUInteger tg_w =
        std::min<NSUInteger>(256U, pso.maxTotalThreadsPerThreadgroup);
    double ts_scale = 1.0;
    bool ts_calibrated = false;

    for (std::uint32_t run = 0; run < total_runs; ++run) {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = nil;
      bool has_ts = false;
      bool has_util = false;
      bool has_stat = false;

      if (ts_buf.sample_buffer != nil) {
        MTLComputePassDescriptor *pass =
            [MTLComputePassDescriptor computePassDescriptor];
        pass.sampleBufferAttachments[0].sampleBuffer = ts_buf.sample_buffer;
        pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = run * 2U;
        pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = run * 2U + 1U;
        has_ts = true;
        if (util_buf.sample_buffer != nil) {
          pass.sampleBufferAttachments[1].sampleBuffer = util_buf.sample_buffer;
          pass.sampleBufferAttachments[1].startOfEncoderSampleIndex = run * 2U;
          pass.sampleBufferAttachments[1].endOfEncoderSampleIndex =
              run * 2U + 1U;
          has_util = true;
        }
        if (stat_buf.sample_buffer != nil) {
          pass.sampleBufferAttachments[2].sampleBuffer = stat_buf.sample_buffer;
          pass.sampleBufferAttachments[2].startOfEncoderSampleIndex = run * 2U;
          pass.sampleBufferAttachments[2].endOfEncoderSampleIndex =
              run * 2U + 1U;
          has_stat = true;
        }
        enc = [cb computeCommandEncoderWithDescriptor:pass];
        if (enc == nil && (has_util || has_stat)) {
          // Fallback to timestamp-only attachment if simultaneous counter sets
          // cannot be sampled in one pass on this GPU.
          has_util = false;
          has_stat = false;
          MTLComputePassDescriptor *ts_only =
              [MTLComputePassDescriptor computePassDescriptor];
          ts_only.sampleBufferAttachments[0].sampleBuffer =
              ts_buf.sample_buffer;
          ts_only.sampleBufferAttachments[0].startOfEncoderSampleIndex =
              run * 2U;
          ts_only.sampleBufferAttachments[0].endOfEncoderSampleIndex =
              run * 2U + 1U;
          enc = [cb computeCommandEncoderWithDescriptor:ts_only];
        }
        if (enc == nil) {
          has_ts = false;
        }
      }
      if (enc == nil) {
        enc = [cb computeCommandEncoder];
      }

      [enc setComputePipelineState:pso];
      [enc setBuffer:src offset:0 atIndex:0];
      [enc setBuffer:dst offset:0 atIndex:1];
      [enc setBuffer:count_buf offset:0 atIndex:2];
      for (std::uint32_t d = 0; d < dispatches_per_encoder; ++d) {
        if (d > 0) {
          [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [enc dispatchThreads:MTLSizeMake(elements, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_w, 1, 1)];
      }
      [enc endEncoding];

      const auto t0 = std::chrono::steady_clock::now();
      [cb commit];
      [cb waitUntilCompleted];
      const double wall_ns =
          std::chrono::duration<double, std::nano>(
              std::chrono::steady_clock::now() - t0)
              .count();
      if (cb.status == MTLCommandBufferStatusError) {
        throw std::runtime_error("classic compute copy failed");
      }
      const double cb_gpu_ns =
          cb.GPUEndTime > cb.GPUStartTime
              ? (cb.GPUEndTime - cb.GPUStartTime) * 1.0e9
              : 0.0;

      double pass_gpu_ns = cb_gpu_ns;
      std::string source = "command_buffer_feedback";
      if (has_ts) {
        const auto raw = resolve_sample_buffer(
            queue, ts_buf, NSMakeRange(run * 2U, 2U),
            sizeof(MTLCounterResultTimestamp));
        if (raw.size() >= 2U * sizeof(MTLCounterResultTimestamp)) {
          MTLCounterResultTimestamp s0{}, s1{};
          std::memcpy(&s0, raw.data(), sizeof(s0));
          std::memcpy(&s1, raw.data() + sizeof(s0), sizeof(s1));
          if (s0.timestamp != MTLCounterErrorValue &&
              s1.timestamp != MTLCounterErrorValue &&
              s1.timestamp >= s0.timestamp) {
            const std::uint64_t delta = s1.timestamp - s0.timestamp;
            if (!ts_calibrated && delta > 0) {
              ts_scale = choose_resolved_timestamp_scale(
                  delta, cb_gpu_ns, caps.metal4_heap_tick_to_ns_scale,
                  caps.metal4_timestamp_frequency_hz);
              ts_calibrated = true;
            }
            pass_gpu_ns = static_cast<double>(delta) * ts_scale;
            source = "mtl_counter_sample_buffer";
          }
        }
      }

      std::optional<std::uint64_t> total_cycles;
      if (has_util) {
        const auto raw = resolve_sample_buffer(
            queue, util_buf, NSMakeRange(run * 2U, 2U),
            sizeof(MTLCounterResultStageUtilization));
        if (raw.size() >= 2U * sizeof(MTLCounterResultStageUtilization)) {
          MTLCounterResultStageUtilization u0{}, u1{};
          std::memcpy(&u0, raw.data(), sizeof(u0));
          std::memcpy(&u1, raw.data() + sizeof(u0), sizeof(u1));
          if (u0.totalCycles != MTLCounterErrorValue &&
              u1.totalCycles != MTLCounterErrorValue &&
              u1.totalCycles >= u0.totalCycles) {
            total_cycles = u1.totalCycles - u0.totalCycles;
          } else {
            total_cycles = valid_counter_u64(u1.totalCycles);
          }
        }
      }

      std::optional<std::uint64_t> kernel_invocations;
      if (has_stat) {
        const auto raw = resolve_sample_buffer(
            queue, stat_buf, NSMakeRange(run * 2U, 2U),
            sizeof(MTLCounterResultStatistic));
        if (raw.size() >= 2U * sizeof(MTLCounterResultStatistic)) {
          MTLCounterResultStatistic st0{}, st1{};
          std::memcpy(&st0, raw.data(), sizeof(st0));
          std::memcpy(&st1, raw.data() + sizeof(st0), sizeof(st1));
          if (st0.computeKernelInvocations != MTLCounterErrorValue &&
              st1.computeKernelInvocations != MTLCounterErrorValue &&
              st1.computeKernelInvocations >= st0.computeKernelInvocations) {
            kernel_invocations =
                st1.computeKernelInvocations - st0.computeKernelInvocations;
          } else {
            kernel_invocations =
                valid_counter_u64(st1.computeKernelInvocations);
          }
        }
      }

      if (run >= warmups) {
        samples.push_back(HwPassCounterSample{
            .label = label,
            .family = "blit_copy_sync",
            .pass_kind = "classic_compute",
            .counter_source = source,
            .gpu_ns = pass_gpu_ns,
            .dispatch_gap_ns = std::max(0.0, cb_gpu_ns - pass_gpu_ns),
            .wall_ns = wall_ns,
            .command_buffer_gpu_ns = cb_gpu_ns,
            .bytes_transferred = bytes * 2ULL * dispatches_per_encoder,
            .flops = 0.0,
            .dispatch_count = dispatches_per_encoder,
            .total_cycles = total_cycles,
            .compute_kernel_invocations = kernel_invocations,
        });
      }
    }
  }
  return summarize_pass_samples(label, samples);
}

HwDispatchGapReport
profile_dispatch_gap_counters(const std::filesystem::path &smoke_metallib,
                              std::uint32_t chain_length,
                              std::uint32_t warmups,
                              std::uint32_t iterations) {
  if (chain_length < 2 || iterations == 0) {
    throw std::invalid_argument("chain_length >= 2 and iterations > 0 required");
  }
  HwDispatchGapReport report{};
  report.chain_length = chain_length;

  // Use a small 64 KiB payload so kernel body is tiny (~sub-microsecond) and
  // hardware timestamp intervals directly expose dispatch & encoder gaps.
  constexpr std::uint64_t kSmallBytes = 64ULL * 1024ULL;
  const auto elements =
      static_cast<std::uint32_t>(kSmallBytes / sizeof(std::uint32_t));
  const std::uint32_t total_runs = warmups + iterations;

  std::vector<HwPassCounterSample> single_enc_samples;
  std::vector<HwPassCounterSample> multi_enc_samples;
  std::vector<double> intra_gaps_ns;
  std::vector<double> inter_gaps_ns;

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (device == nil || queue == nil) {
      throw std::runtime_error("Metal device/queue unavailable");
    }
    const auto caps = enumerate_hw_counter_capabilities();
    NSError *err = nil;
    NSURL *url = [NSURL fileURLWithPath:to_ns_str(smoke_metallib.string())];
    id<MTLLibrary> lib = [device newLibraryWithURL:url error:&err];
    id<MTLFunction> fn = [lib newFunctionWithName:@"omlx_m5_copy_u32"];
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (pso == nil) {
      throw std::runtime_error("dispatch gap pipeline compilation failed");
    }

    id<MTLBuffer> buf_a =
        [device newBufferWithLength:kSmallBytes
                            options:MTLResourceStorageModeShared |
                                    MTLResourceHazardTrackingModeTracked];
    id<MTLBuffer> buf_b =
        [device newBufferWithLength:kSmallBytes
                            options:MTLResourceStorageModeShared |
                                    MTLResourceHazardTrackingModeTracked];
    id<MTLBuffer> cnt_buf =
        [device newBufferWithLength:sizeof(std::uint32_t)
                            options:MTLResourceStorageModeShared |
                                    MTLResourceHazardTrackingModeTracked];
    std::memset(buf_a.contents, 0x11U, kSmallBytes);
    std::memset(buf_b.contents, 0x00U, kSmallBytes);
    std::memcpy(cnt_buf.contents, &elements, sizeof(elements));
    const NSUInteger tg_w =
        std::min<NSUInteger>(256U, pso.maxTotalThreadsPerThreadgroup);

    // 1) Separate encoders in one command buffer (measures inter-encoder gap
    // via MTLComputePassDescriptor start/end timestamps on each encoder).
    id<MTLCounterSet> ts_set = caps.sampling_points.at_stage_boundary
                                   ? find_counter_set(device, "timestamp")
                                   : nil;
    CounterBufferAllocation multi_ts = allocate_sample_buffer(
        device, ts_set, total_runs * chain_length * 2U,
        "omlx_m5.gap.multi_encoder", sizeof(MTLCounterResultTimestamp));
    double ts_scale = 1.0;
    bool ts_calibrated = false;

    for (std::uint32_t run = 0; run < total_runs; ++run) {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      const NSUInteger base = run * chain_length * 2U;
      for (std::uint32_t d = 0; d < chain_length; ++d) {
        id<MTLComputeCommandEncoder> enc = nil;
        if (multi_ts.sample_buffer != nil) {
          MTLComputePassDescriptor *pass =
              [MTLComputePassDescriptor computePassDescriptor];
          pass.sampleBufferAttachments[0].sampleBuffer = multi_ts.sample_buffer;
          pass.sampleBufferAttachments[0].startOfEncoderSampleIndex =
              base + d * 2U;
          pass.sampleBufferAttachments[0].endOfEncoderSampleIndex =
              base + d * 2U + 1U;
          enc = [cb computeCommandEncoderWithDescriptor:pass];
        }
        if (enc == nil) {
          enc = [cb computeCommandEncoder];
        }
        [enc setComputePipelineState:pso];
        [enc setBuffer:(d % 2 == 0 ? buf_a : buf_b) offset:0 atIndex:0];
        [enc setBuffer:(d % 2 == 0 ? buf_b : buf_a) offset:0 atIndex:1];
        [enc setBuffer:cnt_buf offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(elements, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_w, 1, 1)];
        [enc endEncoding];
      }
      const auto t0 = std::chrono::steady_clock::now();
      [cb commit];
      [cb waitUntilCompleted];
      const double wall_ns =
          std::chrono::duration<double, std::nano>(
              std::chrono::steady_clock::now() - t0)
              .count();
      const double cb_gpu_ns =
          cb.GPUEndTime > cb.GPUStartTime
              ? (cb.GPUEndTime - cb.GPUStartTime) * 1.0e9
              : 0.0;

      double active_gpu_ns = cb_gpu_ns;
      double sum_inter_gap_ns = 0.0;
      std::string src_kind = "command_buffer_feedback";
      if (multi_ts.sample_buffer != nil) {
        const auto raw = resolve_sample_buffer(
            queue, multi_ts, NSMakeRange(base, chain_length * 2U),
            sizeof(MTLCounterResultTimestamp));
        if (raw.size() >=
            chain_length * 2U * sizeof(MTLCounterResultTimestamp)) {
          const auto *pts =
              reinterpret_cast<const MTLCounterResultTimestamp *>(raw.data());
          const auto first_ts = pts[0].timestamp;
          const auto last_ts = pts[chain_length * 2U - 1U].timestamp;
          if (!ts_calibrated && first_ts != MTLCounterErrorValue &&
              last_ts != MTLCounterErrorValue && last_ts > first_ts) {
            ts_scale = choose_resolved_timestamp_scale(
                last_ts - first_ts, cb_gpu_ns,
                caps.metal4_heap_tick_to_ns_scale,
                caps.metal4_timestamp_frequency_hz);
            ts_calibrated = true;
          }
          double sum_enc_ns = 0.0;
          for (std::uint32_t d = 0; d < chain_length; ++d) {
            const auto s0 = pts[d * 2U].timestamp;
            const auto s1 = pts[d * 2U + 1U].timestamp;
            if (s0 != MTLCounterErrorValue && s1 != MTLCounterErrorValue &&
                s1 >= s0) {
              sum_enc_ns += static_cast<double>(s1 - s0) * ts_scale;
            }
            if (d > 0) {
              const auto prev_e = pts[(d - 1U) * 2U + 1U].timestamp;
              if (prev_e != MTLCounterErrorValue &&
                  s0 != MTLCounterErrorValue && s0 >= prev_e) {
                const double g = static_cast<double>(s0 - prev_e) * ts_scale;
                sum_inter_gap_ns += g;
                if (run >= warmups) {
                  inter_gaps_ns.push_back(g);
                }
              }
            }
          }
          if (sum_enc_ns > 0.0) {
            active_gpu_ns = sum_enc_ns + sum_inter_gap_ns;
            src_kind = "mtl_counter_sample_buffer";
          }
        }
      }
      if (run >= warmups) {
        multi_enc_samples.push_back(HwPassCounterSample{
            .label = "dispatch_chain_separate_encoders",
            .family = "blit_copy_sync",
            .pass_kind = "classic_compute",
            .counter_source = src_kind,
            .gpu_ns = active_gpu_ns,
            .dispatch_gap_ns = sum_inter_gap_ns,
            .wall_ns = wall_ns,
            .command_buffer_gpu_ns = cb_gpu_ns,
            .bytes_transferred = kSmallBytes * 2ULL * chain_length,
            .dispatch_count = chain_length,
        });
      }
    }

    // 2) Single encoder chain (uses Metal 4 MTL4CounterHeap precise timestamps
    // when available, otherwise single MTLComputePassDescriptor).
    bool ran_metal4 = false;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (caps.metal4_counter_heap_supported) {
      if (@available(macOS 26.0, *)) {
        id<MTL4CommandQueue> q4 = [device newMTL4CommandQueue];
        id<MTL4CommandAllocator> alloc4 = [device newCommandAllocator];
        MTL4CompilerDescriptor *cd = [[MTL4CompilerDescriptor alloc] init];
        id<MTL4Compiler> compiler =
            [device newCompilerWithDescriptor:cd error:&err];
        MTL4LibraryFunctionDescriptor *fd =
            [[MTL4LibraryFunctionDescriptor alloc] init];
        fd.name = @"omlx_m5_copy_u32";
        fd.library = lib;
        MTL4ComputePipelineDescriptor *pd =
            [[MTL4ComputePipelineDescriptor alloc] init];
        pd.computeFunctionDescriptor = fd;
        id<MTLComputePipelineState> pso4 =
            [compiler newComputePipelineStateWithDescriptor:pd
                                        compilerTaskOptions:nil
                                                      error:&err];
        MTL4ArgumentTableDescriptor *atd =
            [[MTL4ArgumentTableDescriptor alloc] init];
        atd.maxBufferBindCount = 3;
        atd.initializeBindings = YES;
        id<MTL4ArgumentTable> at =
            [device newArgumentTableWithDescriptor:atd error:&err];
        MTLResidencySetDescriptor *rsd =
            [[MTLResidencySetDescriptor alloc] init];
        id<MTLResidencySet> res =
            [device newResidencySetWithDescriptor:rsd error:&err];
        MTL4CounterHeapDescriptor *hd =
            [[MTL4CounterHeapDescriptor alloc] init];
        hd.type = MTL4CounterHeapTypeTimestamp;
        hd.count = total_runs * chain_length * 2U;
        id<MTL4CounterHeap> heap =
            [device newCounterHeapWithDescriptor:hd error:&err];

        if (q4 != nil && alloc4 != nil && pso4 != nil && at != nil &&
            res != nil && heap != nil) {
          ran_metal4 = true;
          [res addAllocation:buf_a];
          [res addAllocation:buf_b];
          [res addAllocation:cnt_buf];
          [res addAllocation:pso4];
          [res commit];
          [q4 addResidencySet:res];

          for (std::uint32_t run = 0; run < total_runs; ++run) {
            id<MTL4CommandBuffer> cb = [device newCommandBuffer];
            [cb beginCommandBufferWithAllocator:alloc4];
            id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso4];
            const NSUInteger base = run * chain_length * 2U;
            for (std::uint32_t d = 0; d < chain_length; ++d) {
              if (d > 0) {
                [enc barrierAfterEncoderStages:MTLStageDispatch
                           beforeEncoderStages:MTLStageDispatch
                             visibilityOptions:MTL4VisibilityOptionDevice];
              }
              [at setAddress:(d % 2 == 0 ? buf_a.gpuAddress : buf_b.gpuAddress)
                     atIndex:0];
              [at setAddress:(d % 2 == 0 ? buf_b.gpuAddress : buf_a.gpuAddress)
                     atIndex:1];
              [at setAddress:cnt_buf.gpuAddress atIndex:2];
              [enc setArgumentTable:at];
              [enc writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                        intoHeap:heap
                                         atIndex:base + d * 2U];
              [enc barrierAfterEncoderStages:(MTLStageDispatch | MTLStageBlit)
                         beforeEncoderStages:MTLStageDispatch
                           visibilityOptions:MTL4VisibilityOptionNone];
              [enc dispatchThreads:MTLSizeMake(elements, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg_w, 1, 1)];
              [enc barrierAfterEncoderStages:MTLStageDispatch
                         beforeEncoderStages:(MTLStageDispatch | MTLStageBlit)
                           visibilityOptions:MTL4VisibilityOptionNone];
              [enc writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                        intoHeap:heap
                                         atIndex:base + d * 2U + 1U];
            }
            [enc endEncoding];
            [cb endCommandBuffer];

            const auto timing = commit_and_wait_metal4_counter_cb(
                q4, alloc4, cb, nil, 0, "Metal 4 dispatch gap");
            const double wall_ns = timing.wall_ns;
            const double cb_gpu_ns = timing.cb_gpu_ns;

            double total_chain_ns = cb_gpu_ns;
            double sum_intra_gap = 0.0;
            NSData *data = [heap resolveCounterRange:NSMakeRange(
                                                         base, chain_length * 2U)];
            if (data != nil &&
                data.length >=
                    chain_length * 2U * sizeof(MTL4TimestampHeapEntry)) {
              const auto *pts =
                  static_cast<const MTL4TimestampHeapEntry *>(data.bytes);
              const auto first_ts = pts[0].timestamp;
              const auto last_ts = pts[chain_length * 2U - 1U].timestamp;
              if (last_ts >= first_ts) {
                total_chain_ns = static_cast<double>(last_ts - first_ts) *
                                 caps.metal4_heap_tick_to_ns_scale;
              }
              for (std::uint32_t d = 1; d < chain_length; ++d) {
                const auto prev_e = pts[(d - 1U) * 2U + 1U].timestamp;
                const auto cur_s = pts[d * 2U].timestamp;
                if (cur_s >= prev_e) {
                  const double g = static_cast<double>(cur_s - prev_e) *
                                   caps.metal4_heap_tick_to_ns_scale;
                  sum_intra_gap += g;
                  if (run >= warmups) {
                    intra_gaps_ns.push_back(g);
                  }
                }
              }
            }
            if (run >= warmups) {
              single_enc_samples.push_back(HwPassCounterSample{
                  .label = "dispatch_chain_single_encoder",
                  .family = "blit_copy_sync",
                  .pass_kind = "metal4_compute",
                  .counter_source = "metal4_counter_heap",
                  .gpu_ns = total_chain_ns,
                  .dispatch_gap_ns = sum_intra_gap,
                  .wall_ns = wall_ns,
                  .command_buffer_gpu_ns = cb_gpu_ns,
                  .bytes_transferred = kSmallBytes * 2ULL * chain_length,
                  .dispatch_count = chain_length,
              });
            }
          }
        }
      }
    }
#endif
    if (!ran_metal4) {
      const auto fallback_single = profile_compute_copy_counters(
          smoke_metallib, kSmallBytes, warmups, iterations, false,
          chain_length);
      report.single_encoder_chain = fallback_single;
    } else {
      report.single_encoder_chain = summarize_pass_samples(
          "dispatch_chain_single_encoder", single_enc_samples);
    }
    report.separate_encoder_chain = summarize_pass_samples(
        "dispatch_chain_separate_encoders", multi_enc_samples);
    report.intra_encoder_dispatch_gap_ns =
        compute_distribution_stats(intra_gaps_ns);
    report.inter_encoder_boundary_gap_ns =
        compute_distribution_stats(inter_gaps_ns);
    if (report.single_encoder_chain.gpu_ns.mean > 0.0) {
      report.encoder_coalescing_speedup =
          report.separate_encoder_chain.gpu_ns.mean /
          report.single_encoder_chain.gpu_ns.mean;
    }
  }
  return report;
}

HwPassAggregateStats
profile_nax_qmm_counters(const std::filesystem::path &nax_metallib,
                         std::uint32_t m, std::uint32_t n, std::uint32_t k,
                         std::uint32_t bits, std::uint32_t warmups,
                         std::uint32_t iterations) {
  if ((bits != 4 && bits != 5) || k == 0 || k % 64 != 0 || n == 0 ||
      n % 64 != 0 || m == 0 || iterations == 0) {
    throw std::invalid_argument("invalid NAX QMM counter probe parameters");
  }
  const std::string label = "nax_q" + std::to_string(bits) + "_m" +
                            std::to_string(m) + "_n" + std::to_string(n) +
                            "_k" + std::to_string(k);

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
  if (@available(macOS 26.0, *)) {
    @autoreleasepool {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) {
        throw std::runtime_error("Metal device unavailable");
      }
      const auto caps = enumerate_hw_counter_capabilities();
      NSError *err = nil;
      NSURL *url = [NSURL fileURLWithPath:to_ns_str(nax_metallib.string())];
      id<MTLLibrary> lib = [device newLibraryWithURL:url error:&err];
      if (lib == nil) {
        throw std::runtime_error("NAX metallib load failed: " +
                                 ns_str([err localizedDescription]));
      }
      id<MTL4CommandQueue> q4 = [device newMTL4CommandQueue];
      id<MTL4CommandAllocator> alloc4 = [device newCommandAllocator];
      MTL4CompilerDescriptor *cd = [[MTL4CompilerDescriptor alloc] init];
      id<MTL4Compiler> compiler =
          [device newCompilerWithDescriptor:cd error:&err];
      const std::string fn_name =
          "omlx_m5_q" + std::to_string(bits) + "_affine_g64_nax_bfloat16_t";
      MTL4LibraryFunctionDescriptor *fd =
          [[MTL4LibraryFunctionDescriptor alloc] init];
      fd.name = to_ns_str(fn_name);
      fd.library = lib;
      MTL4ComputePipelineDescriptor *pd =
          [[MTL4ComputePipelineDescriptor alloc] init];
      pd.computeFunctionDescriptor = fd;
      id<MTLComputePipelineState> pso =
          [compiler newComputePipelineStateWithDescriptor:pd
                                      compilerTaskOptions:nil
                                                    error:&err];
      if (q4 == nil || alloc4 == nil || pso == nil) {
        throw std::runtime_error("Metal 4 NAX pipeline compilation failed: " +
                                 ns_str([err localizedDescription]));
      }

      const std::uint64_t row_weight_bytes =
          (static_cast<std::uint64_t>(k) * bits / 32ULL) * sizeof(std::uint32_t);
      const std::uint64_t groups_per_row = k / 64U;
      const std::uint64_t weights_bytes =
          static_cast<std::uint64_t>(n) * row_weight_bytes;
      const std::uint64_t params_bytes = static_cast<std::uint64_t>(n) *
                                         groups_per_row * sizeof(std::uint16_t);
      const std::uint64_t input_bytes =
          static_cast<std::uint64_t>(m) * k * sizeof(std::uint16_t);
      const std::uint64_t output_bytes =
          static_cast<std::uint64_t>(m) * n * sizeof(std::uint16_t);

      const MTLResourceOptions shared_opts =
          MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked;
      id<MTLBuffer> w_buf =
          [device newBufferWithLength:weights_bytes options:shared_opts];
      id<MTLBuffer> s_buf =
          [device newBufferWithLength:params_bytes options:shared_opts];
      id<MTLBuffer> b_buf =
          [device newBufferWithLength:params_bytes options:shared_opts];
      id<MTLBuffer> x_buf =
          [device newBufferWithLength:input_bytes options:shared_opts];
      id<MTLBuffer> y_buf =
          [device newBufferWithLength:output_bytes options:shared_opts];
      id<MTLBuffer> const_buf = [device newBufferWithLength:3 * sizeof(std::int32_t)
                                                    options:shared_opts];
      if (w_buf == nil || s_buf == nil || b_buf == nil || x_buf == nil ||
          y_buf == nil || const_buf == nil) {
        throw std::runtime_error("NAX counter probe buffer allocation failed");
      }

      std::memset(w_buf.contents, 0x55U, weights_bytes);
      const std::uint16_t bf16_scale = float_to_bf16(0.015625F);
      const std::uint16_t bf16_bias = float_to_bf16(-0.03125F);
      const std::uint16_t bf16_x = float_to_bf16(0.125F);
      auto *s_ptr = static_cast<std::uint16_t *>(s_buf.contents);
      auto *b_ptr = static_cast<std::uint16_t *>(b_buf.contents);
      for (std::size_t i = 0; i < params_bytes / sizeof(std::uint16_t); ++i) {
        s_ptr[i] = bf16_scale;
        b_ptr[i] = bf16_bias;
      }
      auto *x_ptr = static_cast<std::uint16_t *>(x_buf.contents);
      for (std::size_t i = 0; i < input_bytes / sizeof(std::uint16_t); ++i) {
        x_ptr[i] = bf16_x;
      }
      std::memset(y_buf.contents, 0, output_bytes);
      const std::array<std::int32_t, 3> dims{static_cast<std::int32_t>(k),
                                             static_cast<std::int32_t>(n),
                                             static_cast<std::int32_t>(m)};
      std::memcpy(const_buf.contents, dims.data(), sizeof(dims));

      MTL4ArgumentTableDescriptor *atd =
          [[MTL4ArgumentTableDescriptor alloc] init];
      atd.maxBufferBindCount = 8;
      atd.initializeBindings = YES;
      id<MTL4ArgumentTable> at =
          [device newArgumentTableWithDescriptor:atd error:&err];
      MTLResidencySetDescriptor *rsd = [[MTLResidencySetDescriptor alloc] init];
      id<MTLResidencySet> res =
          [device newResidencySetWithDescriptor:rsd error:&err];
      [res addAllocation:w_buf];
      [res addAllocation:s_buf];
      [res addAllocation:b_buf];
      [res addAllocation:x_buf];
      [res addAllocation:y_buf];
      [res addAllocation:const_buf];
      [res addAllocation:pso];
      [res commit];
      [q4 addResidencySet:res];

      const std::uint32_t total_runs = warmups + iterations;
      MTL4CounterHeapDescriptor *hd = [[MTL4CounterHeapDescriptor alloc] init];
      hd.type = MTL4CounterHeapTypeTimestamp;
      hd.count = total_runs * 2U;
      id<MTL4CounterHeap> heap =
          caps.metal4_counter_heap_supported
              ? [device newCounterHeapWithDescriptor:hd error:&err]
              : nil;
      id<MTLSharedEvent> shared_ev = [device newSharedEvent];
      std::uint64_t ev_val = 0;

      std::vector<HwPassCounterSample> samples;
      samples.reserve(iterations);

      const std::uint64_t total_bytes =
          weights_bytes + 2ULL * params_bytes + input_bytes + output_bytes;
      const double total_flops = 2.0 * static_cast<double>(m) *
                                 static_cast<double>(n) *
                                 static_cast<double>(k);

      for (std::uint32_t run = 0; run < total_runs; ++run) {
        id<MTL4CommandBuffer> cb = [device newCommandBuffer];
        [cb beginCommandBufferWithAllocator:alloc4];
        id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [at setAddress:w_buf.gpuAddress atIndex:0];
        [at setAddress:s_buf.gpuAddress atIndex:1];
        [at setAddress:b_buf.gpuAddress atIndex:2];
        [at setAddress:x_buf.gpuAddress atIndex:3];
        [at setAddress:y_buf.gpuAddress atIndex:4];
        [at setAddress:const_buf.gpuAddress atIndex:5];
        [at setAddress:const_buf.gpuAddress + 4ULL atIndex:6];
        [at setAddress:const_buf.gpuAddress + 8ULL atIndex:7];
        [enc setArgumentTable:at];
        if (heap != nil) {
          [enc writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                    intoHeap:heap
                                     atIndex:run * 2U];
          [enc barrierAfterEncoderStages:(MTLStageDispatch | MTLStageBlit)
                     beforeEncoderStages:MTLStageDispatch
                       visibilityOptions:MTL4VisibilityOptionNone];
        }
        [enc dispatchThreadgroups:MTLSizeMake((n + 63U) / 64U, (m + 63U) / 64U, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
        if (heap != nil) {
          [enc barrierAfterEncoderStages:MTLStageDispatch
                     beforeEncoderStages:(MTLStageDispatch | MTLStageBlit)
                       visibilityOptions:MTL4VisibilityOptionNone];
          [enc writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                    intoHeap:heap
                                     atIndex:run * 2U + 1U];
        }
        [enc endEncoding];
        [cb endCommandBuffer];

        ++ev_val;
        const auto timing = commit_and_wait_metal4_counter_cb(
            q4, alloc4, cb, shared_ev, ev_val, "Metal 4 NAX kernel execution");
        const double wall_ns = timing.wall_ns;
        const double cb_gpu_ns = timing.cb_gpu_ns;
        double kernel_gpu_ns = cb_gpu_ns;
        std::string source = "command_buffer_feedback";

        if (heap != nil) {
          NSData *data =
              [heap resolveCounterRange:NSMakeRange(run * 2U, 2U)];
          if (data != nil &&
              data.length >= 2U * sizeof(MTL4TimestampHeapEntry)) {
            const auto *pts =
                static_cast<const MTL4TimestampHeapEntry *>(data.bytes);
            if (pts[1].timestamp >= pts[0].timestamp &&
                pts[1].timestamp > 0) {
              kernel_gpu_ns =
                  static_cast<double>(pts[1].timestamp - pts[0].timestamp) *
                  caps.metal4_heap_tick_to_ns_scale;
              source = "metal4_counter_heap";
            }
          }
        }

        if (run >= warmups) {
          samples.push_back(HwPassCounterSample{
              .label = label,
              .family = "nax_gemm",
              .pass_kind = "metal4_compute",
              .counter_source = source,
              .gpu_ns = kernel_gpu_ns,
              .dispatch_gap_ns = std::max(0.0, cb_gpu_ns - kernel_gpu_ns),
              .wall_ns = wall_ns,
              .command_buffer_gpu_ns = cb_gpu_ns,
              .bytes_transferred = total_bytes,
              .flops = total_flops,
              .dispatch_count = 1,
          });
        }
      }
      return summarize_pass_samples(label, samples);
    }
  }
#endif
  throw std::runtime_error("Metal 4 NAX counter profiling requires macOS 26+");
}

} // namespace omlx::m5
