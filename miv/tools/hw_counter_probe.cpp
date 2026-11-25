#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/hw_counters.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/qwen_forward.h"

#ifndef OMLX_M5_CLASSIC_METALLIB_PATH
#define OMLX_M5_CLASSIC_METALLIB_PATH ""
#endif

#ifndef OMLX_M5_NAX_METALLIB_PATH
#define OMLX_M5_NAX_METALLIB_PATH ""
#endif

namespace {

using Json = nlohmann::json;
using namespace omlx::m5;

struct Options {
  std::string mode;
  std::uint32_t warmups = 5;
  std::uint32_t iterations = 30;
  std::filesystem::path kernel_dir;
  std::filesystem::path model_artifact;
  std::filesystem::path model_config;
  std::uint32_t max_context = 4096;
  std::uint32_t prefill_tokens = 128;
  std::uint32_t verify_tokens = 8;
  std::uint32_t decode_steps = 20;
  std::uint32_t stage_repeats = 5;
  std::filesystem::path output;
};

std::uint32_t parse_u32(const std::string &text, std::string_view flag) {
  std::size_t used = 0;
  if (text.empty() || text.front() == '-') {
    throw std::invalid_argument(std::string(flag) + " must be non-negative");
  }
  const auto val = std::stoull(text, &used);
  if (used != text.size() || val > UINT32_MAX) {
    throw std::invalid_argument("invalid integer for " + std::string(flag));
  }
  return static_cast<std::uint32_t>(val);
}

Options parse_args(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto next = [&]() -> std::string {
      if (++i >= argc) {
        throw std::invalid_argument("missing value for " + arg);
      }
      return argv[i];
    };
    if (arg == "--mode") {
      o.mode = next();
    } else if (arg == "--warmups") {
      o.warmups = parse_u32(next(), arg);
    } else if (arg == "--iterations") {
      o.iterations = parse_u32(next(), arg);
    } else if (arg == "--kernel-directory") {
      o.kernel_dir = next();
    } else if (arg == "--model-artifact") {
      o.model_artifact = next();
    } else if (arg == "--model-config") {
      o.model_config = next();
    } else if (arg == "--max-context") {
      o.max_context = parse_u32(next(), arg);
    } else if (arg == "--prefill-tokens") {
      o.prefill_tokens = parse_u32(next(), arg);
    } else if (arg == "--verify-tokens") {
      o.verify_tokens = parse_u32(next(), arg);
    } else if (arg == "--decode-steps") {
      o.decode_steps = parse_u32(next(), arg);
    } else if (arg == "--stage-repeats") {
      o.stage_repeats = parse_u32(next(), arg);
    } else if (arg == "--output") {
      o.output = next();
    } else if (arg == "-h" || arg == "--help") {
      std::cout
          << "usage: omlx_m5_hw_counter_probe [--mode enumerate|h0|model|all]\n"
          << "       [--warmups N] [--iterations N] [--kernel-directory DIR]\n"
          << "       [--model-artifact FILE --model-config FILE]\n"
          << "       [--max-context N] [--prefill-tokens N] [--verify-tokens N]\n"
          << "       [--decode-steps N] [--stage-repeats N] [--output FILE]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (o.iterations == 0) {
    throw std::invalid_argument("--iterations must be >= 1");
  }
  if (o.mode.empty()) {
    o.mode = o.model_artifact.empty() ? "h0" : "model";
  }
  if (o.mode != "enumerate" && o.mode != "h0" && o.mode != "model" &&
      o.mode != "all") {
    throw std::invalid_argument(
        "--mode must be enumerate, h0, model, or all");
  }
  if ((o.mode == "model" || (o.mode == "all" && !o.model_artifact.empty())) &&
      (o.model_artifact.empty() || o.model_config.empty())) {
    throw std::invalid_argument(
        "model mode requires both --model-artifact and --model-config");
  }
  return o;
}

std::filesystem::path resolve_kernel_dir(const Options &o, const char *argv0) {
  if (!o.kernel_dir.empty() && std::filesystem::is_directory(o.kernel_dir)) {
    return std::filesystem::weakly_canonical(o.kernel_dir);
  }
  if (std::string_view(OMLX_M5_CLASSIC_METALLIB_PATH).size() > 0) {
    const auto candidate =
        std::filesystem::path(OMLX_M5_CLASSIC_METALLIB_PATH).parent_path();
    if (std::filesystem::is_directory(candidate)) {
      return std::filesystem::weakly_canonical(candidate);
    }
  }
  if (argv0 != nullptr && *argv0 != '\0') {
    const auto exe_dir =
        std::filesystem::weakly_canonical(argv0).parent_path();
    for (const auto &cand :
         {exe_dir / "metallib", exe_dir.parent_path() / "metallib"}) {
      if (std::filesystem::is_directory(cand)) {
        return cand;
      }
    }
  }
  return o.kernel_dir;
}

Json dist_json(const DistributionStats &d) {
  return {
      {"count", d.count},
      {"min", d.min},
      {"p50", d.p50},
      {"p90", d.p90},
      {"p99", d.p99},
      {"max", d.max},
      {"mean", d.mean},
      {"stddev", d.stddev},
      {"cv", d.cv},
  };
}

Json pass_stats_json(const HwPassAggregateStats &s) {
  Json j = {
      {"label", s.label},
      {"family", s.family},
      {"pass_kind", s.pass_kind},
      {"counter_source", s.counter_source},
      {"dispatch_count_per_iter", s.dispatch_count_per_iter},
      {"bytes_per_iter", s.bytes_per_iter},
      {"flops_per_iter", s.flops_per_iter},
      {"arithmetic_intensity_flops_per_byte",
       s.arithmetic_intensity_flops_per_byte},
      {"bottleneck_class", s.bottleneck_class},
      {"gpu_ns", dist_json(s.gpu_ns)},
      {"gpu_ms_p50", s.gpu_ns.p50 * 1.0e-6},
      {"gpu_ms_mean", s.gpu_ns.mean * 1.0e-6},
      {"dispatch_gap_ns", dist_json(s.dispatch_gap_ns)},
      {"wall_ns", dist_json(s.wall_ns)},
      {"command_buffer_gpu_ns", dist_json(s.command_buffer_gpu_ns)},
      {"mean_effective_gbs", s.mean_effective_gbs},
      {"p50_effective_gbs", s.p50_effective_gbs},
      {"mean_effective_tflops", s.mean_effective_tflops},
      {"p50_effective_tflops", s.p50_effective_tflops},
      {"mean_host_overhead_ns", s.mean_host_overhead_ns},
  };
  if (s.total_cycles.has_value()) {
    j["total_cycles"] = dist_json(*s.total_cycles);
  } else {
    j["total_cycles"] = nullptr;
  }
  if (s.compute_kernel_invocations.has_value()) {
    j["compute_kernel_invocations"] = dist_json(*s.compute_kernel_invocations);
  } else {
    j["compute_kernel_invocations"] = nullptr;
  }
  return j;
}

Json family_breakdown_json(const std::vector<HwFamilyBreakdownEntry> &entries) {
  Json arr = Json::array();
  for (const auto &e : entries) {
    arr.push_back({
        {"family", e.family},
        {"sample_count", e.sample_count},
        {"total_dispatches", e.total_dispatches},
        {"total_gpu_ns", e.total_gpu_ns},
        {"total_gpu_ms", e.total_gpu_ns * 1.0e-6},
        {"share_of_gpu_pct", e.share_of_gpu_pct},
        {"mean_effective_gbs", e.mean_effective_gbs},
        {"mean_effective_tflops", e.mean_effective_tflops},
    });
  }
  return arr;
}

Json capabilities_json(const HwCounterDeviceCapabilities &c) {
  Json sets = Json::array();
  for (const auto &cs : c.counter_sets) {
    sets.push_back({
        {"name", cs.name},
        {"canonical_kind", cs.canonical_kind},
        {"counters", cs.counters},
        {"shared_sample_buffer_supported", cs.shared_sample_buffer_supported},
        {"private_sample_buffer_supported", cs.private_sample_buffer_supported},
    });
  }
  return {
      {"device_name", c.device_name},
      {"architecture_name", c.architecture_name},
      {"runtime_os", c.runtime_os},
      {"apple_gpu_family", c.apple_gpu_family},
      {"recommended_working_set_bytes", c.recommended_working_set_bytes},
      {"max_buffer_length", c.max_buffer_length},
      {"metal3_counter_sets_available", c.metal3_counter_sets_available},
      {"timestamp_counter_supported", c.timestamp_counter_supported},
      {"stage_utilization_supported", c.stage_utilization_supported},
      {"statistic_counter_supported", c.statistic_counter_supported},
      {"metal4_counter_heap_supported", c.metal4_counter_heap_supported},
      {"metal4_timestamp_frequency_hz", c.metal4_timestamp_frequency_hz},
      {"metal4_timestamp_entry_bytes", c.metal4_timestamp_entry_bytes},
      {"sampled_cpu_timestamp_ns", c.sampled_cpu_timestamp_ns},
      {"sampled_gpu_timestamp", c.sampled_gpu_timestamp},
      {"resolved_timestamp_to_ns_scale", c.resolved_timestamp_to_ns_scale},
      {"metal4_heap_tick_to_ns_scale", c.metal4_heap_tick_to_ns_scale},
      {"timestamp_calibration_source", c.timestamp_calibration_source},
      {"sampling_points",
       {
           {"at_stage_boundary", c.sampling_points.at_stage_boundary},
           {"at_draw_boundary", c.sampling_points.at_draw_boundary},
           {"at_dispatch_boundary", c.sampling_points.at_dispatch_boundary},
           {"at_tile_dispatch_boundary",
            c.sampling_points.at_tile_dispatch_boundary},
           {"at_blit_boundary", c.sampling_points.at_blit_boundary},
       }},
      {"counter_sets", sets},
  };
}

double bits_per_weight_format(std::string_view fmt) {
  if (fmt.find("mxfp4") != std::string_view::npos) {
    return 4.25; // 4-bit E2M1 + 8-bit E8M0 per 32 elements
  }
  if (fmt.find("q4_g64") != std::string_view::npos) {
    return 4.5; // 4-bit + BF16 scale/bias per 64
  }
  if (fmt.find("q5_g64") != std::string_view::npos) {
    return 5.5;
  }
  if (fmt.find("q6_g64") != std::string_view::npos) {
    return 6.5;
  }
  if (fmt.find("q8_g64") != std::string_view::npos) {
    return 8.5;
  }
  if (fmt.find("bf16") != std::string_view::npos) {
    return 16.0;
  }
  return 4.5;
}

Json run_h0_suite(const Options &o, const std::filesystem::path &kernel_dir,
                  const HwCounterDeviceCapabilities &caps) {
  const auto smoke_metallib = kernel_dir / "omlx_m5_smoke.metallib";
  const auto nax_metallib = kernel_dir / "omlx_m5_nax_smoke.metallib";

  HwCounterConfig collector_cfg{};
  collector_cfg.enabled = true;
  HwCounterCollector collector(collector_cfg);
  auto record_agg = [&](const HwPassAggregateStats &s) {
    collector.record(HwPassCounterSample{
        .label = s.label,
        .family = s.family,
        .pass_kind = s.pass_kind,
        .counter_source = s.counter_source,
        .gpu_ns = s.gpu_ns.mean,
        .dispatch_gap_ns = s.dispatch_gap_ns.mean,
        .wall_ns = s.wall_ns.mean,
        .command_buffer_gpu_ns = s.command_buffer_gpu_ns.mean,
        .bytes_transferred = s.bytes_per_iter,
        .flops = s.flops_per_iter,
        .dispatch_count = s.dispatch_count_per_iter,
    });
  };

  Json benchmarks = Json::array();
  const std::uint64_t mb64 = 64ULL * 1024ULL * 1024ULL;
  const std::uint64_t mb256 = 256ULL * 1024ULL * 1024ULL;
  const std::uint32_t large_iters = std::max<std::uint32_t>(4U, o.iterations / 2U);

  const auto blit_64_priv =
      profile_blit_copy_counters(mb64, o.warmups, o.iterations, true, 1);
  benchmarks.push_back(pass_stats_json(blit_64_priv));
  record_agg(blit_64_priv);

  const auto blit_256_priv =
      profile_blit_copy_counters(mb256, std::min(o.warmups, 3U), large_iters,
                                 true, 1);
  benchmarks.push_back(pass_stats_json(blit_256_priv));
  record_agg(blit_256_priv);

  const auto blit_256_shared =
      profile_blit_copy_counters(mb256, std::min(o.warmups, 3U), large_iters,
                                 false, 1);
  benchmarks.push_back(pass_stats_json(blit_256_shared));
  record_agg(blit_256_shared);

  Json dispatch_gap_json = nullptr;
  if (std::filesystem::is_regular_file(smoke_metallib)) {
    const auto comp_64_m4 = profile_compute_copy_counters(
        smoke_metallib, mb64, o.warmups, o.iterations, true, 1);
    benchmarks.push_back(pass_stats_json(comp_64_m4));

    const auto comp_256_m4 = profile_compute_copy_counters(
        smoke_metallib, mb256, std::min(o.warmups, 3U), large_iters, true, 1);
    benchmarks.push_back(pass_stats_json(comp_256_m4));

    const auto comp_64_classic = profile_compute_copy_counters(
        smoke_metallib, mb64, o.warmups, o.iterations, false, 1);
    benchmarks.push_back(pass_stats_json(comp_64_classic));

    const auto gap_report =
        profile_dispatch_gap_counters(smoke_metallib, 16U, o.warmups, o.iterations);
    dispatch_gap_json = {
        {"chain_length", gap_report.chain_length},
        {"single_encoder_chain", pass_stats_json(gap_report.single_encoder_chain)},
        {"separate_encoder_chain",
         pass_stats_json(gap_report.separate_encoder_chain)},
        {"intra_encoder_dispatch_gap_ns",
         dist_json(gap_report.intra_encoder_dispatch_gap_ns)},
        {"inter_encoder_boundary_gap_ns",
         dist_json(gap_report.inter_encoder_boundary_gap_ns)},
        {"encoder_coalescing_speedup", gap_report.encoder_coalescing_speedup},
    };
  }

  Json nax_sweep = Json::array();
  const bool legacy_nax_smoke_present =
      std::filesystem::is_regular_file(nax_metallib);
  if (caps.apple_gpu_family >= 9 && legacy_nax_smoke_present) {
    struct NaxShape {
      std::uint32_t m;
      std::uint32_t n;
      std::uint32_t k;
      std::uint32_t bits;
    };
    const std::array<NaxShape, 8> shapes{{
        {1, 5120, 5120, 4},
        {8, 5120, 5120, 4},
        {16, 5120, 5120, 4},
        {64, 5120, 5120, 4},
        {256, 5120, 5120, 4},
        {512, 5120, 5120, 4},
        {1, 5120, 5120, 5},
        {8, 5120, 5120, 5},
    }};
    for (const auto &sh : shapes) {
      const std::uint32_t iters =
          sh.m >= 256 ? std::max<std::uint32_t>(6U, o.iterations / 3U)
                      : o.iterations;
      try {
        const auto stats = profile_nax_qmm_counters(
            nax_metallib, sh.m, sh.n, sh.k, sh.bits, o.warmups, iters);
        nax_sweep.push_back(pass_stats_json(stats));
        record_agg(stats);
      } catch (const std::exception &e) {
        nax_sweep.push_back({
            {"label", "nax_q" + std::to_string(sh.bits) + "_m" +
                          std::to_string(sh.m)},
            {"error", e.what()},
        });
      }
    }
  }

  return {
      {"copy_and_compute_passes", benchmarks},
      {"dispatch_gap_analysis", dispatch_gap_json},
      {"nax_gemm_counter_sweep", nax_sweep},
      {"collector_family_summary",
       family_breakdown_json(collector.summarize_by_family())},
      {"key_findings",
       {
           {"peak_blit_private_256mb_p50_gbs", blit_256_priv.p50_effective_gbs},
           {"peak_blit_private_256mb_mean_gbs", blit_256_priv.mean_effective_gbs},
           {"peak_blit_shared_256mb_p50_gbs", blit_256_shared.p50_effective_gbs},
           {"blit_64mb_private_p50_gbs", blit_64_priv.p50_effective_gbs},
           {"legacy_mlx_nax_smoke_present", legacy_nax_smoke_present},
           {"legacy_mlx_nax_smoke_skipped", !legacy_nax_smoke_present},
       }},
  };
}

struct StageWeightAccounting {
  std::string primary_family = "other_compute";
  std::map<std::string, double> family_weight_bytes;
  std::map<std::string, double> family_flops;
  std::set<std::string> kernels;
  double total_weight_bytes = 0.0;
  double total_flops = 0.0;
};

std::map<std::string, StageWeightAccounting>
analyze_profile_stages(const QwenProfileDescription &desc,
                       const QwenModelConfig &cfg, std::uint32_t actual_tokens) {
  std::map<std::string, StageWeightAccounting> out;

  // Map every weight route in this profile to its stage & kernel family.
  for (const auto &w : desc.weights) {
    auto &acc = out[w.stage];
    const std::string fam =
        classify_kernel_family("", w.library, w.function, w.family);
    acc.kernels.insert(w.library + ":" + w.function);

    // Check if this weight belongs to routed 256-expert MoE vs dense/shared.
    const bool is_routed_moe =
        (cfg.num_experts > 0 &&
         (w.object.find(".experts.") != std::string::npos ||
          w.object.find("switch_mlp") != std::string::npos));
    const double bits = bits_per_weight_format(w.format);
    double active_experts = 1.0;
    double flops_mul = static_cast<double>(actual_tokens);
    if (is_routed_moe) {
      // Each token activates top-k (8) experts; across T tokens unique active
      // experts follow 256 * (1 - ((256-8)/256)^T).
      const double single_miss =
          static_cast<double>(cfg.num_experts - cfg.num_experts_per_token) /
          static_cast<double>(cfg.num_experts);
      active_experts =
          static_cast<double>(cfg.num_experts) *
          (1.0 - std::pow(single_miss, static_cast<double>(actual_tokens)));
      flops_mul = static_cast<double>(actual_tokens) *
                  static_cast<double>(cfg.num_experts_per_token);
    }
    const std::uint32_t eff_t = (w.stage == "head" && w.capacity == 1U)
                                    ? 1U
                                    : actual_tokens;
    if (w.stage == "head" && w.capacity == 1U) {
      flops_mul = 1.0;
    }
    (void)eff_t;
    const double bytes = active_experts * static_cast<double>(w.n) *
                         static_cast<double>(w.k) * (bits / 8.0);
    const double flops =
        2.0 * flops_mul * static_cast<double>(w.n) * static_cast<double>(w.k);
    const std::string weight_fam = is_routed_moe ? "moe" : fam;
    acc.family_weight_bytes[weight_fam] += bytes;
    acc.family_flops[weight_fam] += flops;
    acc.total_weight_bytes += bytes;
    acc.total_flops += flops;
  }

  for (const auto &st : desc.stages) {
    auto &acc = out[st.name];
    if (st.name == "embedding" || st.name == "sample") {
      acc.primary_family = "vocab_sampling";
    } else if (st.name == "head") {
      if (!acc.family_weight_bytes.empty()) {
        acc.primary_family = acc.family_weight_bytes.begin()->first;
      } else {
        acc.primary_family = "vocab_sampling";
      }
    } else if (st.name.find(".attention") != std::string::npos) {
      acc.primary_family = "attention";
    } else if (st.name.find(".gdn") != std::string::npos) {
      acc.primary_family = "gdn";
    } else if (st.name.find(".mlp") != std::string::npos) {
      acc.primary_family = (cfg.num_experts > 0) ? "moe" : "nax_gemm";
      if (cfg.num_experts == 0 && !acc.family_weight_bytes.empty()) {
        // Distinguish NAX GEMM vs SIMD GEMV/GEMM on T=1 vs T>=2!
        double best_b = -1.0;
        for (const auto &[k_fam, b_val] : acc.family_weight_bytes) {
          if (b_val > best_b) {
            best_b = b_val;
            acc.primary_family = k_fam;
          }
        }
      }
    }
  }
  return out;
}

Json profile_single_width_stages(
    QwenForward &forward, const QwenModelConfig &cfg,
    const std::vector<std::uint32_t> &tokens, std::uint32_t repeats,
    const GpuSamplingOptions &sampling, std::string_view phase_name) {
  const auto actual_t = static_cast<std::uint32_t>(tokens.size());
  const auto desc = forward.inspect_profile(actual_t, false);
  const auto accounting = analyze_profile_stages(desc, cfg, actual_t);

  // Measure monolithic single-commit ICB execution first.
  std::vector<double> mono_gpu_ns;
  std::vector<double> mono_wall_ns;
  mono_gpu_ns.reserve(repeats);
  mono_wall_ns.reserve(repeats);
  const auto vocab_bound =
      static_cast<std::uint32_t>(std::max(1, cfg.vocab_size));
  for (std::uint32_t r = 0; r < repeats + 1U; ++r) {
    forward.reset();
    // Seed KV cache with a short prefix when measuring decode/verify so
    // attention/GDN execute realistic non-zero state!
    if (actual_t <= 8U) {
      std::vector<std::uint32_t> prefix(64U);
      for (std::uint32_t i = 0; i < 64U; ++i) {
        prefix[i] = (i * 37U + 11U) % vocab_bound;
      }
      static_cast<void>(forward.run(prefix, sampling, false, true, false));
    }
    const auto res = forward.run(tokens, sampling, false, true, false);
    if (r >= 1U) {
      mono_gpu_ns.push_back(res.submission.timing.gpu_ms * 1.0e6);
      mono_wall_ns.push_back(res.submission.timing.wall_ms * 1.0e6);
    }
  }

  // Measure per-stage ICB range execution across repeats.
  std::map<std::string, std::vector<double>> stage_gpu_samples_ns;
  std::map<std::string, std::vector<double>> stage_wall_samples_ns;
  std::map<std::string, std::uint64_t> stage_dispatches;
  std::vector<std::string> stage_order;

  for (std::uint32_t r = 0; r < repeats + 1U; ++r) {
    forward.reset();
    if (actual_t <= 8U) {
      std::vector<std::uint32_t> prefix(64U);
      for (std::uint32_t i = 0; i < 64U; ++i) {
        prefix[i] = (i * 37U + 11U) % vocab_bound;
      }
      static_cast<void>(forward.run(prefix, sampling, false, true, false));
    }
    const auto res = forward.run(tokens, sampling, false, true, true);
    if (r == 0U) {
      for (const auto &st : res.stage_timings) {
        stage_order.push_back(st.stage);
        stage_dispatches[st.stage] = st.dispatch_count;
      }
    } else {
      for (const auto &st : res.stage_timings) {
        stage_gpu_samples_ns[st.stage].push_back(st.gpu_ms * 1.0e6);
        stage_wall_samples_ns[st.stage].push_back(st.wall_ms * 1.0e6);
      }
    }
  }

  // Aggregate both by Architectural Stage Category (attention_block, gdn_block,
  // mlp_or_moe_block, head, embedding, sample) AND by Kernel/Compute Family!
  struct BucketAgg {
    std::uint64_t stage_instances = 0;
    std::uint64_t total_dispatches = 0;
    double mean_gpu_ns = 0.0;
    double p50_gpu_ns = 0.0;
    double total_bytes = 0.0;
    double total_flops = 0.0;
    std::set<std::string> kernels;
  };
  std::map<std::string, BucketAgg> arch_buckets;
  std::map<std::string, BucketAgg> kernel_family_buckets;

  double sum_stage_mean_gpu_ns = 0.0;
  double sum_stage_p50_gpu_ns = 0.0;
  double total_active_weight_bytes = 0.0;
  double total_model_flops = 0.0;

  for (const auto &st_name : stage_order) {
    const auto g_stats = compute_distribution_stats(stage_gpu_samples_ns[st_name]);
    sum_stage_mean_gpu_ns += g_stats.mean;
    sum_stage_p50_gpu_ns += g_stats.p50;

    std::string arch_cat = "other";
    if (st_name == "embedding") {
      arch_cat = "embedding";
    } else if (st_name == "head") {
      arch_cat = "lm_head";
    } else if (st_name == "sample") {
      arch_cat = "sampling";
    } else if (st_name.find(".attention") != std::string::npos) {
      arch_cat = "full_attention_sublayer";
    } else if (st_name.find(".gdn") != std::string::npos) {
      arch_cat = "gdn_linear_attention_sublayer";
    } else if (st_name.find(".mlp") != std::string::npos) {
      arch_cat = (cfg.num_experts > 0) ? "moe_sublayer" : "dense_mlp_sublayer";
    }

    const auto acc_it = accounting.find(st_name);
    const double st_bytes =
        acc_it != accounting.end() ? acc_it->second.total_weight_bytes : 0.0;
    const double st_flops =
        acc_it != accounting.end() ? acc_it->second.total_flops : 0.0;
    const std::string k_fam = acc_it != accounting.end()
                                  ? acc_it->second.primary_family
                                  : "other_compute";

    total_active_weight_bytes += st_bytes;
    total_model_flops += st_flops;

    auto &ab = arch_buckets[arch_cat];
    ++ab.stage_instances;
    ab.total_dispatches += stage_dispatches[st_name];
    ab.mean_gpu_ns += g_stats.mean;
    ab.p50_gpu_ns += g_stats.p50;
    ab.total_bytes += st_bytes;
    ab.total_flops += st_flops;
    if (acc_it != accounting.end()) {
      ab.kernels.insert(acc_it->second.kernels.begin(),
                        acc_it->second.kernels.end());
    }

    auto &kb = kernel_family_buckets[k_fam];
    ++kb.stage_instances;
    kb.total_dispatches += stage_dispatches[st_name];
    kb.mean_gpu_ns += g_stats.mean;
    kb.p50_gpu_ns += g_stats.p50;
    kb.total_bytes += st_bytes;
    kb.total_flops += st_flops;
    if (acc_it != accounting.end()) {
      kb.kernels.insert(acc_it->second.kernels.begin(),
                        acc_it->second.kernels.end());
    }
  }

  const auto mono_gpu = compute_distribution_stats(mono_gpu_ns);
  const auto mono_wall = compute_distribution_stats(mono_wall_ns);

  auto format_buckets = [&](const std::map<std::string, BucketAgg> &m) {
    struct Row {
      std::string key;
      BucketAgg val;
    };
    std::vector<Row> rows;
    for (const auto &[k, v] : m) {
      rows.push_back({k, v});
    }
    std::ranges::sort(rows, [](const Row &a, const Row &b) {
      return a.val.mean_gpu_ns > b.val.mean_gpu_ns;
    });
    Json arr = Json::array();
    for (const auto &r : rows) {
      const double share = sum_stage_mean_gpu_ns > 0.0
                               ? 100.0 * r.val.mean_gpu_ns / sum_stage_mean_gpu_ns
                               : 0.0;
      // Scale to monolithic single-commit GPU time so estimated single-commit
      // ms sums cleanly to real production step latency.
      const double est_mono_gpu_ms =
          (mono_gpu.mean * 1.0e-6) * (share / 100.0);
      const double eff_gbs =
          r.val.mean_gpu_ns > 0.0 ? r.val.total_bytes / r.val.mean_gpu_ns : 0.0;
      const double eff_tflops =
          r.val.mean_gpu_ns > 0.0
              ? (r.val.total_flops / r.val.mean_gpu_ns) * 1.0e-3
              : 0.0;
      arr.push_back({
          {"category", r.key},
          {"stage_instances", r.val.stage_instances},
          {"dispatches", r.val.total_dispatches},
          {"isolated_stage_gpu_ms_mean", r.val.mean_gpu_ns * 1.0e-6},
          {"isolated_stage_gpu_ms_p50", r.val.p50_gpu_ns * 1.0e-6},
          {"estimated_monolithic_gpu_ms", est_mono_gpu_ms},
          {"share_of_gpu_pct", share},
          {"active_weight_bytes", r.val.total_bytes},
          {"flops", r.val.total_flops},
          {"isolated_effective_gbs", eff_gbs},
          {"isolated_effective_tflops", eff_tflops},
          {"kernels", r.val.kernels},
      });
    }
    return arr;
  };

  const double mono_effective_gbs =
      mono_gpu.mean > 0.0 ? total_active_weight_bytes / mono_gpu.mean : 0.0;
  const double mono_effective_tflops =
      mono_gpu.mean > 0.0 ? (total_model_flops / mono_gpu.mean) * 1.0e-3 : 0.0;
  const double tok_per_s_gpu =
      mono_gpu.mean > 0.0
          ? static_cast<double>(actual_t) / (mono_gpu.mean * 1.0e-9)
          : 0.0;
  const double tok_per_s_wall =
      mono_wall.mean > 0.0
          ? static_cast<double>(actual_t) / (mono_wall.mean * 1.0e-9)
          : 0.0;

  // Summarize exact weight projection routes across the entire model at this T.
  struct RouteGroup {
    std::uint64_t weight_count = 0;
    double active_bytes = 0.0;
    double flops = 0.0;
    std::set<std::string> kernels;
  };
  std::map<std::string, RouteGroup> route_groups;
  for (const auto &w : desc.weights) {
    const bool is_routed_moe =
        (cfg.num_experts > 0 &&
         (w.object.find(".experts.") != std::string::npos ||
          w.object.find("switch_mlp") != std::string::npos));
    const std::string fam =
        is_routed_moe
            ? (w.library.find("nax") != std::string::npos ||
                       w.function.find("nax") != std::string::npos
                   ? "moe_nax"
                   : "moe_grouped")
            : classify_kernel_family("", w.library, w.function, w.family);
    const double bits = bits_per_weight_format(w.format);
    double active_experts = 1.0;
    double flops_mul = (w.stage == "head" && w.capacity == 1U)
                           ? 1.0
                           : static_cast<double>(actual_t);
    if (is_routed_moe) {
      const double single_miss =
          static_cast<double>(cfg.num_experts - cfg.num_experts_per_token) /
          static_cast<double>(cfg.num_experts);
      active_experts =
          static_cast<double>(cfg.num_experts) *
          (1.0 - std::pow(single_miss, static_cast<double>(actual_t)));
      flops_mul = static_cast<double>(actual_t) *
                  static_cast<double>(cfg.num_experts_per_token);
    }
    auto &rg = route_groups[fam];
    ++rg.weight_count;
    rg.active_bytes += active_experts * static_cast<double>(w.n) *
                       static_cast<double>(w.k) * (bits / 8.0);
    rg.flops +=
        2.0 * flops_mul * static_cast<double>(w.n) * static_cast<double>(w.k);
    rg.kernels.insert(w.library + ":" + w.function);
  }
  Json operator_routes_json = Json::array();
  for (const auto &[fam, rg] : route_groups) {
    operator_routes_json.push_back({
        {"route_family", fam},
        {"weight_tensors", rg.weight_count},
        {"active_weight_bytes", rg.active_bytes},
        {"share_of_active_bytes_pct",
         total_active_weight_bytes > 0.0
             ? 100.0 * rg.active_bytes / total_active_weight_bytes
             : 0.0},
        {"flops", rg.flops},
        {"share_of_flops_pct",
         total_model_flops > 0.0 ? 100.0 * rg.flops / total_model_flops : 0.0},
        {"kernels", rg.kernels},
    });
  }

  return {
      {"phase", phase_name},
      {"tokens", actual_t},
      {"profile_capacity", desc.capacity},
      {"total_dispatches", desc.dispatch_count},
      {"total_barriers", desc.barrier_count},
      {"monolithic_icb_gpu_ns", dist_json(mono_gpu)},
      {"monolithic_icb_wall_ns", dist_json(mono_wall)},
      {"monolithic_icb_gpu_ms_p50", mono_gpu.p50 * 1.0e-6},
      {"monolithic_icb_wall_ms_p50", mono_wall.p50 * 1.0e-6},
      {"host_commit_and_readback_overhead_ms",
       std::max(0.0, (mono_wall.mean - mono_gpu.mean) * 1.0e-6)},
      {"sum_isolated_stages_gpu_ms_mean", sum_stage_mean_gpu_ns * 1.0e-6},
      {"sum_isolated_stages_gpu_ms_p50", sum_stage_p50_gpu_ns * 1.0e-6},
      {"icb_single_commit_coalescing_saved_gpu_ms",
       (sum_stage_mean_gpu_ns - mono_gpu.mean) * 1.0e-6},
      {"icb_single_commit_speedup_vs_stage_commits",
       mono_gpu.mean > 0.0 ? sum_stage_mean_gpu_ns / mono_gpu.mean : 1.0},
      {"active_weight_bytes_per_step", total_active_weight_bytes},
      {"flops_per_step", total_model_flops},
      {"arithmetic_intensity_flops_per_byte",
       total_active_weight_bytes > 0.0
           ? total_model_flops / total_active_weight_bytes
           : 0.0},
      {"monolithic_effective_weight_gbs", mono_effective_gbs},
      {"nominal_1200gbs_utilization_pct",
       (mono_effective_gbs / 1200.0) * 100.0},
      {"monolithic_effective_tflops", mono_effective_tflops},
      {"throughput_gpu_tok_per_s", tok_per_s_gpu},
      {"throughput_wall_tok_per_s", tok_per_s_wall},
      {"bottleneck_class",
       classify_bottleneck(total_active_weight_bytes > 0.0
                               ? total_model_flops / total_active_weight_bytes
                               : 0.0,
                           mono_gpu.mean, mono_effective_gbs)},
      {"sublayer_breakdown", format_buckets(arch_buckets)},
      {"kernel_family_breakdown", format_buckets(kernel_family_buckets)},
      {"projection_route_summary", operator_routes_json},
  };
}

Json run_model_suite(const Options &o, const std::filesystem::path &kernel_dir) {
  auto context = MetalContext::create();
  const auto artifact = ArtifactModel::open(o.model_artifact, true);
  const auto config = load_qwen_model_config(o.model_config);
  auto mat_plan = plan_model_materialization(
      artifact, context,
      {.components = {ModelComponent::text}, .require_verified_payload = true});
  const std::uint64_t resident_bytes = mat_plan.resident_bytes;
  const auto weights = materialize_model(artifact, std::move(mat_plan), context);
  const auto layers =
      build_qwen_layer_catalog(artifact, config, o.max_context, 64);

  const bool split_kv = config.full_attention_layers() > 0 &&
                        context.capabilities().profile.probe.apple_gpu_family >= 9;
  QwenForward forward(context, weights, layers, config, o.max_context,
                      kernel_dir, {}, /*inspect_routes=*/false,
                      /*speculative=*/false, /*enable_m4_short_routes=*/false,
                      /*diagnostic_override=*/{},
                      /*enable_attention_split=*/split_kv,
                      /*enable_batch=*/false, /*moe_override=*/{},
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1,
                      AttentionNumericPolicy::RowExactV1);

  GpuSamplingOptions sampling{};
  sampling.contract = SamplingContract::PositionCoupledV1;
  sampling.temperature = 0.0F;
  sampling.top_k = 1;
  sampling.top_p = 1.0F;
  sampling.seed = 42ULL;

  // 1) Continuous realistic prefill + decode loop timing distribution.
  const auto vocab_bound =
      static_cast<std::uint32_t>(std::max(1, config.vocab_size));
  std::vector<std::uint32_t> prompt(o.prefill_tokens);
  for (std::uint32_t i = 0; i < o.prefill_tokens; ++i) {
    prompt[i] = (i * 41U + 7U) % vocab_bound;
  }
  forward.reset();
  const auto warm_prefill = forward.run(prompt, sampling, false, true, false);
  (void)warm_prefill;

  forward.reset();
  const auto prefill_res = forward.run(prompt, sampling, false, true, false);
  std::uint32_t cur_tok = prefill_res.token;
  std::vector<double> decode_step_gpu_ns;
  std::vector<double> decode_step_wall_ns;
  std::vector<double> decode_host_gap_ns;
  decode_step_gpu_ns.reserve(o.decode_steps);
  decode_step_wall_ns.reserve(o.decode_steps);
  decode_host_gap_ns.reserve(o.decode_steps);

  for (std::uint32_t step = 0; step < o.decode_steps; ++step) {
    const std::array<std::uint32_t, 1> single{cur_tok};
    const auto r = forward.run(single, sampling, false, true, false);
    cur_tok = r.token;
    const double g_ns = r.submission.timing.gpu_ms * 1.0e6;
    const double w_ns = r.submission.timing.wall_ms * 1.0e6;
    decode_step_gpu_ns.push_back(g_ns);
    decode_step_wall_ns.push_back(w_ns);
    decode_host_gap_ns.push_back(std::max(0.0, w_ns - g_ns));
  }

  const auto decode_gpu_dist = compute_distribution_stats(decode_step_gpu_ns);
  const auto decode_wall_dist = compute_distribution_stats(decode_step_wall_ns);
  const auto decode_gap_dist = compute_distribution_stats(decode_host_gap_ns);

  // 2) Detailed stage & kernel-family breakdown across Decode (T=1),
  // SpecVerify/Batch (T=8), and Prefill (T=prefill_tokens).
  const std::vector<std::uint32_t> t1_toks{19U % vocab_bound};
  std::vector<std::uint32_t> t8_toks(o.verify_tokens);
  for (std::uint32_t i = 0; i < o.verify_tokens; ++i) {
    t8_toks[i] = (i * 17U + 3U) % vocab_bound;
  }

  const Json decode_t1_breakdown = profile_single_width_stages(
      forward, config, t1_toks, o.stage_repeats, sampling, "decode_t1");
  const Json verify_t8_breakdown = profile_single_width_stages(
      forward, config, t8_toks, o.stage_repeats, sampling,
      "verify_or_batch_t" + std::to_string(o.verify_tokens));
  const Json prefill_breakdown = profile_single_width_stages(
      forward, config, prompt, std::max<std::uint32_t>(2U, o.stage_repeats / 2U),
      sampling, "prefill_t" + std::to_string(o.prefill_tokens));

  return {
      {"artifact", o.model_artifact.filename().string()},
      {"source_repo", artifact.summary().source_repo},
      {"source_revision", artifact.summary().source_revision},
      {"resident_weight_bytes", resident_bytes},
      {"allocated_metal_bytes", context.current_allocated_bytes()},
      {"model_architecture",
       {
           {"num_hidden_layers", config.num_hidden_layers},
           {"full_attention_layers", config.full_attention_layers()},
           {"gdn_layers",
            config.num_hidden_layers - config.full_attention_layers()},
           {"hidden_size", config.hidden_size},
           {"vocab_size", config.vocab_size},
           {"num_experts", config.num_experts},
           {"num_experts_per_token", config.num_experts_per_token},
       }},
      {"continuous_decode_loop",
       {
           {"prefill_tokens", o.prefill_tokens},
           {"prefill_gpu_ms", prefill_res.submission.timing.gpu_ms},
           {"prefill_wall_ms", prefill_res.submission.timing.wall_ms},
           {"prefill_tok_per_s",
            prefill_res.submission.timing.wall_ms > 0.0
                ? static_cast<double>(o.prefill_tokens) /
                      (prefill_res.submission.timing.wall_ms * 1.0e-3)
                : 0.0},
           {"decode_steps", o.decode_steps},
           {"decode_step_gpu_ns", dist_json(decode_gpu_dist)},
           {"decode_step_wall_ns", dist_json(decode_wall_dist)},
           {"decode_host_gap_ns", dist_json(decode_gap_dist)},
           {"decode_gpu_ms_p50", decode_gpu_dist.p50 * 1.0e-6},
           {"decode_wall_ms_p50", decode_wall_dist.p50 * 1.0e-6},
           {"decode_tok_per_s_p50_wall",
            decode_wall_dist.p50 > 0.0
                ? 1.0e9 / decode_wall_dist.p50
                : 0.0},
           {"decode_tok_per_s_p50_gpu",
            decode_gpu_dist.p50 > 0.0 ? 1.0e9 / decode_gpu_dist.p50 : 0.0},
       }},
      {"profiles",
       {
           {"decode_t1", decode_t1_breakdown},
           {"verify_t8", verify_t8_breakdown},
           {"prefill", prefill_breakdown},
       }},
  };
}

void write_atomic_json(const std::filesystem::path &path, const Json &payload) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  const auto tmp = path.string() + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("unable to open output file: " + tmp);
    }
    out << payload.dump(2) << '\n';
    out.flush();
    if (!out) {
      throw std::runtime_error("failed writing JSON to: " + tmp);
    }
  }
  std::filesystem::rename(tmp, path);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto opts = parse_args(argc, argv);
    const auto kernel_dir = resolve_kernel_dir(opts, argv[0]);
    const auto caps = enumerate_hw_counter_capabilities();
    const auto overhead = measure_hw_counter_cpu_overhead(100'000);

    Json root = {
        {"schema", "omlx-m5-hw-counters-v1"},
        {"mode", opts.mode},
        {"device_counter_capabilities", capabilities_json(caps)},
        {"zero_overhead_disabled_check",
         {
             {"iterations", overhead.iterations},
             {"disabled_ns_per_call", overhead.disabled_ns_per_call},
             {"enabled_record_ns_per_call", overhead.enabled_record_ns_per_call},
             {"zero_allocation_when_disabled",
              overhead.zero_allocation_when_disabled},
         }},
    };

    if (opts.mode == "h0" || opts.mode == "all") {
      root["h0_microbenchmarks"] = run_h0_suite(opts, kernel_dir, caps);
    }
    if (opts.mode == "model" ||
        (opts.mode == "all" && !opts.model_artifact.empty())) {
      root["model_hardware_breakdown"] = run_model_suite(opts, kernel_dir);
    }

    if (opts.output.empty()) {
      std::cout << root.dump(2) << '\n';
    } else {
      write_atomic_json(opts.output, root);
    }
    return 0;
  } catch (const std::exception &err) {
    std::cerr << "omlx_m5_hw_counter_probe: " << err.what() << '\n';
    return 1;
  }
}
