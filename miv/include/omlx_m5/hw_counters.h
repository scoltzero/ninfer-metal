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

namespace omlx::m5 {

struct DistributionStats {
  std::uint64_t count = 0;
  double min = 0.0;
  double p50 = 0.0;
  double p90 = 0.0;
  double p99 = 0.0;
  double max = 0.0;
  double mean = 0.0;
  double stddev = 0.0;
  double cv = 0.0;
};

[[nodiscard]] DistributionStats
compute_distribution_stats(std::span<const double> samples);

struct HwCounterSetInfo {
  std::string name;
  std::string canonical_kind;
  std::vector<std::string> counters;
  bool shared_sample_buffer_supported = false;
  bool private_sample_buffer_supported = false;
};

struct HwCounterSamplingPoints {
  bool at_stage_boundary = false;
  bool at_draw_boundary = false;
  bool at_dispatch_boundary = false;
  bool at_tile_dispatch_boundary = false;
  bool at_blit_boundary = false;
};

struct HwCounterDeviceCapabilities {
  std::string device_name;
  std::string architecture_name;
  std::string runtime_os;
  std::uint32_t apple_gpu_family = 0;
  std::uint64_t recommended_working_set_bytes = 0;
  std::uint64_t max_buffer_length = 0;
  bool metal3_counter_sets_available = false;
  bool timestamp_counter_supported = false;
  bool stage_utilization_supported = false;
  bool statistic_counter_supported = false;
  bool metal4_counter_heap_supported = false;
  std::uint64_t metal4_timestamp_frequency_hz = 0;
  std::uint64_t metal4_timestamp_entry_bytes = 0;
  std::uint64_t sampled_cpu_timestamp_ns = 0;
  std::uint64_t sampled_gpu_timestamp = 0;
  double resolved_timestamp_to_ns_scale = 1.0;
  double metal4_heap_tick_to_ns_scale = 1.0;
  std::string timestamp_calibration_source = "uncalibrated";
  HwCounterSamplingPoints sampling_points;
  std::vector<HwCounterSetInfo> counter_sets;
};

struct HwCounterConfig {
  bool enabled = false;
  bool capture_timestamps = true;
  bool capture_stage_utilization = true;
  bool capture_statistics = true;
  bool prefer_metal4_counter_heap = true;
  std::uint32_t max_samples = 8192;

  [[nodiscard]] static constexpr HwCounterConfig disabled() noexcept {
    return HwCounterConfig{};
  }
  [[nodiscard]] static HwCounterConfig from_environment();
};

struct HwPassCounterSample {
  std::string label;
  std::string family;
  std::string pass_kind;
  std::string counter_source;
  double gpu_ns = 0.0;
  double dispatch_gap_ns = 0.0;
  double wall_ns = 0.0;
  double command_buffer_gpu_ns = 0.0;
  std::uint64_t bytes_transferred = 0;
  double flops = 0.0;
  std::uint64_t dispatch_count = 1;
  std::optional<std::uint64_t> total_cycles;
  std::optional<std::uint64_t> vertex_cycles;
  std::optional<std::uint64_t> fragment_cycles;
  std::optional<std::uint64_t> render_target_cycles;
  std::optional<std::uint64_t> compute_kernel_invocations;
};

struct HwPassAggregateStats {
  std::string label;
  std::string family;
  std::string pass_kind;
  std::string counter_source;
  std::uint64_t dispatch_count_per_iter = 1;
  std::uint64_t bytes_per_iter = 0;
  double flops_per_iter = 0.0;
  double arithmetic_intensity_flops_per_byte = 0.0;
  std::string bottleneck_class = "unknown";
  DistributionStats gpu_ns;
  DistributionStats dispatch_gap_ns;
  DistributionStats wall_ns;
  DistributionStats command_buffer_gpu_ns;
  double mean_effective_gbs = 0.0;
  double p50_effective_gbs = 0.0;
  double mean_effective_tflops = 0.0;
  double p50_effective_tflops = 0.0;
  double mean_host_overhead_ns = 0.0;
  std::optional<DistributionStats> total_cycles;
  std::optional<DistributionStats> compute_kernel_invocations;
};

struct HwFamilyBreakdownEntry {
  std::string family;
  std::uint64_t sample_count = 0;
  std::uint64_t total_dispatches = 0;
  double total_gpu_ns = 0.0;
  double share_of_gpu_pct = 0.0;
  double total_bytes = 0.0;
  double total_flops = 0.0;
  double mean_effective_gbs = 0.0;
  double mean_effective_tflops = 0.0;
};

struct HwOverheadCheck {
  std::uint64_t iterations = 0;
  double disabled_ns_per_call = 0.0;
  double enabled_record_ns_per_call = 0.0;
  bool zero_allocation_when_disabled = true;
};

// Lightweight statistical hardware counter collector.
// When disabled (`config.enabled == false`), `enabled_` is false inline and
// `record(...)` returns immediately without touching mutexes, heap memory, or
// Objective-C/Metal objects.
class HwCounterCollector final {
public:
  explicit HwCounterCollector(HwCounterConfig config = HwCounterConfig::disabled());
  HwCounterCollector(HwCounterCollector &&) noexcept;
  HwCounterCollector &operator=(HwCounterCollector &&) noexcept;
  HwCounterCollector(const HwCounterCollector &) = delete;
  HwCounterCollector &operator=(const HwCounterCollector &) = delete;
  ~HwCounterCollector();

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }
  [[nodiscard]] const HwCounterConfig &config() const noexcept {
    return config_;
  }

  void record(const HwPassCounterSample &sample) {
    if (!enabled_) {
      return;
    }
    record_slow(sample);
  }

  template <typename SampleFactory>
  void record_lazy(SampleFactory &&factory) {
    if (!enabled_) {
      return;
    }
    record_slow(factory());
  }

  void clear();
  [[nodiscard]] std::size_t sample_count() const;
  [[nodiscard]] std::vector<HwPassCounterSample> samples() const;
  [[nodiscard]] std::vector<HwPassAggregateStats> summarize_by_label() const;
  [[nodiscard]] std::vector<HwFamilyBreakdownEntry> summarize_by_family() const;

private:
  void record_slow(const HwPassCounterSample &sample);

  HwCounterConfig config_;
  bool enabled_ = false;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

struct HwDispatchGapReport {
  std::uint32_t chain_length = 0;
  HwPassAggregateStats single_encoder_chain;
  HwPassAggregateStats separate_encoder_chain;
  DistributionStats intra_encoder_dispatch_gap_ns;
  DistributionStats inter_encoder_boundary_gap_ns;
  double encoder_coalescing_speedup = 1.0;
};

[[nodiscard]] HwCounterDeviceCapabilities enumerate_hw_counter_capabilities();

[[nodiscard]] HwOverheadCheck
measure_hw_counter_cpu_overhead(std::uint64_t iterations = 100'000);

[[nodiscard]] HwPassAggregateStats
summarize_pass_samples(std::string_view label,
                       std::span<const HwPassCounterSample> samples);

[[nodiscard]] std::vector<HwFamilyBreakdownEntry>
summarize_family_breakdown(std::span<const HwPassCounterSample> samples);

[[nodiscard]] std::string classify_kernel_family(std::string_view stage,
                                                 std::string_view library,
                                                 std::string_view function,
                                                 std::string_view route_family);

[[nodiscard]] std::string classify_bottleneck(double arithmetic_intensity,
                                              double mean_gpu_ns,
                                              double effective_gbs,
                                              double nominal_bandwidth_gbs = 1200.0);

// Hardware MTLCounterSampleBuffer Blit pass profiler (attaches Timestamp +
// StageUtilization/Statistic sample buffers where supported on MTLBlitPassDescriptor).
[[nodiscard]] HwPassAggregateStats
profile_blit_copy_counters(std::uint64_t bytes, std::uint32_t warmups,
                           std::uint32_t iterations,
                           bool private_storage = true,
                           std::uint32_t copies_per_pass = 1);

// Hardware Compute copy profiler (attaches MTL4CounterHeap on Metal 4 compute
// encoder or MTLCounterSampleBuffer on MTLComputePassDescriptor).
[[nodiscard]] HwPassAggregateStats
profile_compute_copy_counters(const std::filesystem::path &smoke_metallib,
                              std::uint64_t bytes, std::uint32_t warmups,
                              std::uint32_t iterations, bool use_metal4 = true,
                              std::uint32_t dispatches_per_encoder = 1);

// Hardware intra-encoder vs inter-encoder dispatch gap profiler using precise
// GPU timestamps across back-to-back dispatches.
[[nodiscard]] HwDispatchGapReport
profile_dispatch_gap_counters(const std::filesystem::path &smoke_metallib,
                              std::uint32_t chain_length,
                              std::uint32_t warmups,
                              std::uint32_t iterations);

// Hardware Metal 4 NAX quantized GEMM counter profiler using MTL4CounterHeap
// precise timestamps around NAX Q4/Q5 affine kernel dispatches.
[[nodiscard]] HwPassAggregateStats
profile_nax_qmm_counters(const std::filesystem::path &nax_metallib,
                         std::uint32_t m, std::uint32_t n, std::uint32_t k,
                         std::uint32_t bits, std::uint32_t warmups,
                         std::uint32_t iterations);

} // namespace omlx::m5
