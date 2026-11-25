#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/dflash_forward.h"
#include "omlx_m5/quantization.h"

namespace {

std::uint64_t digest(const std::vector<float> &values) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const float value : values) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    for (unsigned byte = 0; byte < 4; ++byte) {
      hash ^= (bits >> (byte * 8)) & 0xffU;
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

struct OwnedWeight {
  omlx::m5::MaterializedWeightView view;
  std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> buffers;
};

OwnedWeight upload_object(const omlx::m5::ArtifactModel &artifact,
                          const omlx::m5::ArtifactObject &object,
                          const omlx::m5::MetalContext &context) {
  OwnedWeight result;
  result.view = {
      .object_id = object.id,
      .component = omlx::m5::classify_model_component(object.name),
      .name = object.name,
      .format = object.format,
      .layout = object.layout,
      .parent_shape = object.shape,
      .padded_k = object.padded_k,
      .row_count = object.shape.at(0),
      .in_features = object.shape.at(1),
  };
  for (const auto &plane : object.planes) {
    result.buffers.push_back(std::make_unique<omlx::m5::MetalBuffer>(
        context.upload(artifact.plane_bytes(object, plane))));
    result.view.planes.push_back({
        .kind = plane.kind,
        .buffer = result.buffers.back().get(),
        .segment_index = static_cast<std::uint32_t>(result.buffers.size() - 1),
        .offset = 0,
        .bytes = plane.bytes,
        .row_stride = plane.bytes / object.shape.at(0),
    });
  }
  return result;
}

std::uint32_t parse_u32(const char *value, const char *label) {
  const auto parsed = std::stoull(value);
  if (!parsed || parsed > UINT32_MAX)
    throw std::invalid_argument(std::string(label) + " is invalid");
  return static_cast<std::uint32_t>(parsed);
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 9 || argc > 12)
      throw std::invalid_argument(
          "usage: omlx_m5_benchmark_dflash_slice DRAFT_ARTIFACT DRAFT_CONFIG "
          "TARGET_ARTIFACT LIBRARY_DIRECTORY MAX_CONTEXT WARMUPS REPEATS K "
          "[on] [gpu|paired|paired-random]");
    bool enable_short_routes = false, gpu_selector = false, paired = false;
    float paired_temperature = 0.0F;
    for (int arg = 9; arg < argc; ++arg) {
      const std::string_view mode = argv[arg];
      if (mode == "on")
        enable_short_routes = true;
      else if (mode == "gpu")
        gpu_selector = true;
      else if (mode == "paired")
        paired = true;
      else if (mode == "paired-random") {
        paired = true;
        paired_temperature = 0.7F;
      } else
        throw std::invalid_argument("unknown draft route/selector mode");
    }
    const auto max_context = parse_u32(argv[5], "max context");
    const auto warmups = parse_u32(argv[6], "warmups");
    const auto repeats = parse_u32(argv[7], "repeats");
    const auto draft_tokens = parse_u32(argv[8], "draft tokens");

    auto context = omlx::m5::MetalContext::create();
    auto draft_artifact = omlx::m5::ArtifactModel::open(argv[1], true);
    auto draft_config = omlx::m5::load_dflash2_model_config(argv[2]);
    auto draft_weights = omlx::m5::materialize_model(
        draft_artifact,
        omlx::m5::plan_model_materialization(
            draft_artifact, context,
            {.components = {omlx::m5::ModelComponent::speculative},
             .require_verified_payload = true}),
        context);

    // B0 verifies the full target payload; this benchmark only materializes
    // its two shared slices. Avoid rehashing the full 27B artifact per run.
    auto target_artifact = omlx::m5::ArtifactModel::open(argv[3], false);
    auto embedding = upload_object(
        target_artifact,
        target_artifact.object("language_model.model.embed_tokens"), context);
    auto head = upload_object(target_artifact,
                              target_artifact.object("language_model.lm_head"),
                              context);

    const auto feature_width =
        std::uint64_t(draft_config.target_layer_ids.size()) *
        draft_config.hidden_size;
    auto features = context.make_buffer(std::uint64_t(max_context) *
                                        feature_width * sizeof(std::uint16_t));
    constexpr std::uint64_t chunk_values = 1U << 20;
    std::vector<std::uint16_t> chunk(chunk_values);
    for (std::uint64_t offset = 0;
         offset < std::uint64_t(max_context) * feature_width;) {
      const auto count = std::min<std::uint64_t>(
          chunk_values, std::uint64_t(max_context) * feature_width - offset);
      for (std::uint64_t i = 0; i < count; ++i) {
        const float value =
            float(std::int32_t(((offset + i) * 17U) % 257U) - 128) / 1024.0F;
        chunk[i] = omlx::m5::float_to_bf16(value);
      }
      features.write(std::as_bytes(std::span(chunk.data(), std::size_t(count))),
                     offset * sizeof(std::uint16_t));
      offset += count;
    }

    omlx::m5::DFlashForward program(context, draft_weights, draft_config,
                                    max_context, embedding.view, head.view,
                                    features, argv[4], enable_short_routes);
    std::uint64_t shared_weight_bytes = 0;
    for (const auto &plane : embedding.view.planes)
      shared_weight_bytes += plane.bytes;
    for (const auto &plane : head.view.planes)
      shared_weight_bytes += plane.bytes;
    if (paired) {
      nlohmann::json comparisons = nlohmann::json::array();
      for (const auto context_tokens :
           {32U, std::min(512U, max_context), max_context}) {
        program.reset();
        static_cast<void>(program.propose(7, context_tokens, draft_tokens,
                                          paired_temperature, 17, true, true,
                                          true));
        for (std::uint32_t i = 0; i < warmups; ++i) {
          static_cast<void>(program.propose(7, context_tokens, draft_tokens,
                                            paired_temperature, 123 + i, true,
                                            true, true));
          static_cast<void>(program.propose(7, context_tokens, draft_tokens,
                                            paired_temperature, 123 + i));
        }
        std::vector<double> cpu_gpu, gpu_gpu, cpu_wall, gpu_wall;
        std::uint64_t cpu_read = 0, gpu_read = 0;
        unsigned fallbacks = 0;
        float max_probability_error = 0.0F;
        for (std::uint32_t iteration = 0; iteration < repeats; ++iteration) {
          const auto seed = 1000U + iteration;
          const auto run = [&](bool use_gpu) {
            const auto started = std::chrono::steady_clock::now();
            auto result = program.propose(7, context_tokens, draft_tokens,
                                          paired_temperature, seed, true,
                                          !use_gpu, !use_gpu);
            const auto elapsed = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - started)
                                     .count();
            return std::pair{std::move(result), elapsed};
          };
          auto [first, first_wall] = run(iteration % 2 == 1);
          auto [second, second_wall] = run(iteration % 2 == 0);
          auto &cpu = iteration % 2 ? second : first;
          auto &gpu = iteration % 2 ? first : second;
          const auto old_wall = iteration % 2 ? second_wall : first_wall;
          const auto new_wall = iteration % 2 ? first_wall : second_wall;
          if (cpu.selection.tokens != gpu.selection.tokens ||
              cpu.selection.steps.size() != gpu.selection.steps.size())
            throw std::runtime_error("paired selector token mismatch");
          for (std::size_t row = 0; row < cpu.selection.steps.size(); ++row) {
            const auto &a = cpu.selection.steps[row];
            const auto &b = gpu.selection.steps[row];
            if (a.candidate_ids != b.candidate_ids ||
                a.probabilities.size() != b.probabilities.size())
              throw std::runtime_error("paired selector top-k mismatch");
            for (std::size_t index = 0; index < a.probabilities.size(); ++index)
              if ((max_probability_error =
                       std::max(max_probability_error,
                                std::abs(a.probabilities[index] -
                                         b.probabilities[index]))) > 2e-6F)
                throw std::runtime_error("paired selector q mismatch");
          }
          cpu_gpu.push_back(cpu.submission.timing.gpu_ms);
          gpu_gpu.push_back(gpu.submission.timing.gpu_ms);
          cpu_wall.push_back(old_wall);
          gpu_wall.push_back(new_wall);
          cpu_read = cpu.host_read_bytes;
          gpu_read = gpu.host_read_bytes;
          fallbacks += gpu.compact_cpu_fallback;
        }
        comparisons.push_back(
            {{"context", context_tokens},
             {"cpu_gpu_ms", cpu_gpu},
             {"gpu_gpu_ms", gpu_gpu},
             {"cpu_wall_ms", cpu_wall},
             {"gpu_wall_ms", gpu_wall},
             {"cpu_host_read_bytes", cpu_read},
             {"gpu_host_read_bytes", gpu_read},
             {"gpu_compact_fallbacks", fallbacks},
             {"tokens_exact", true},
             {"max_abs_q_difference", max_probability_error}});
      }
      std::cout
          << nlohmann::json(
                 {{"schema", "omlx-m5-p03-paired-companion-v1"},
                  {"scope", "real companion and shared target slices; "
                            "synthetic features; not full target throughput"},
                  {"source_revision", draft_artifact.summary().source_revision},
                  {"resident_draft_bytes",
                   draft_weights.stats().resident_bytes},
                  {"target_embedding_head_bytes", shared_weight_bytes},
                  {"temperature", paired_temperature},
                  {"draft_tokens", draft_tokens},
                  {"warmups", warmups},
                  {"repeats", repeats},
                  {"comparisons", comparisons},
                  {"m5_ultra_qualified", false}})
                 .dump(2)
          << '\n';
      return 0;
    }
    nlohmann::json records = nlohmann::json::array();
    for (const auto context_tokens :
         {32U, std::min(512U, max_context), max_context}) {
      for (const bool use_icb : {false, true}) {
        if (!use_icb && context_tokens != 32)
          continue;
        program.reset();
        const auto cold =
            program.propose(7, context_tokens, draft_tokens, 0, 17, use_icb,
                            !gpu_selector, !gpu_selector);
        for (std::uint32_t i = 0; i < warmups; ++i)
          static_cast<void>(program.propose(7, context_tokens, draft_tokens, 0,
                                            123 + i, use_icb, !gpu_selector,
                                            !gpu_selector));
        std::vector<double> gpu_ms, wall_ms;
        gpu_ms.reserve(repeats);
        wall_ms.reserve(repeats);
        std::uint64_t dispatches = 0;
        std::uint32_t hot_projected_rows = 0;
        std::uint32_t short_matrix_calls = 0;
        std::uint64_t host_read_bytes = 0;
        nlohmann::json hot_outputs;
        std::vector<float> first_hot_logits;
        for (std::uint32_t i = 0; i < repeats; ++i) {
          const auto started = std::chrono::steady_clock::now();
          const auto result =
              program.propose(7, context_tokens, draft_tokens, 0, 1000 + i,
                              use_icb, !gpu_selector, !gpu_selector);
          const auto elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();
          gpu_ms.push_back(result.submission.timing.gpu_ms);
          wall_ms.push_back(elapsed);
          dispatches = result.draft_dispatches;
          hot_projected_rows += result.context_rows_projected;
          short_matrix_calls += result.short_matrix_calls;
          host_read_bytes = result.host_read_bytes;
          if (i == 0) {
            hot_outputs = {{"hidden_digest", digest(result.hidden)},
                           {"logits_digest", digest(result.logits)},
                           {"proposal_tokens", result.selection.tokens}};
            first_hot_logits = result.logits;
          }
        }
        if (hot_projected_rows != 0)
          throw std::runtime_error("steady draft proposal reprojected context");
        nlohmann::json stage_trace = nullptr;
        if (gpu_selector && use_icb && context_tokens == 32) {
          stage_trace = nlohmann::json::array();
          for (const auto &stage : program.profile_stages())
            stage_trace.push_back({{"stage", stage.stage},
                                   {"gpu_ms", stage.gpu_ms},
                                   {"wall_ms", stage.wall_ms},
                                   {"dispatches", stage.dispatch_count}});
        }
        if (const char *trace_dir = std::getenv("M5_SLICE_TRACE_DIR");
            trace_dir != nullptr && use_icb &&
            (context_tokens == 32 || context_tokens == max_context) &&
            !gpu_selector) {
          std::filesystem::create_directories(trace_dir);
          const std::string filename =
              std::string(enable_short_routes ? "on-" : "off-") +
              std::to_string(context_tokens) + "-logits.f32";
          std::ofstream trace(std::filesystem::path(trace_dir) / filename,
                              std::ios::binary);
          trace.write(reinterpret_cast<const char *>(first_hot_logits.data()),
                      std::streamsize(first_hot_logits.size() * sizeof(float)));
          if (!trace)
            throw std::runtime_error("DFlash slice trace write failed");
        }
        records.push_back({
            {"route", use_icb ? "icb" : "execute_batch"},
            {"context_tokens", context_tokens},
            {"draft_tokens", draft_tokens},
            {"verify_width", draft_tokens + 1},
            {"cold_gpu_ms", cold.submission.timing.gpu_ms},
            {"cold_context_rows", cold.context_rows_projected},
            {"cold_launched_rows", cold.context_rows_launched},
            {"cold_hidden_digest", digest(cold.hidden)},
            {"cold_logits_digest", digest(cold.logits)},
            {"cold_proposal_tokens", cold.selection.tokens},
            {"hot_outputs", std::move(hot_outputs)},
            {"hot_projected_rows", hot_projected_rows},
            {"cached_context_rows", program.cached_context()},
            {"dispatches", dispatches},
            {"short_matrix_calls", short_matrix_calls},
            {"gpu_ms", gpu_ms},
            {"wall_ms", wall_ms},
            {"host_read_bytes", host_read_bytes},
            {"diagnostic_stage_trace", stage_trace},
        });
      }
    }
    const auto registry = context.registry_stats();
    std::cout << nlohmann::json(
                     {
                         {"schema", "omlx-m5-real-dflash-slice-v1"},
                         {"scope", "M4 real 3.6GB companion plus target "
                                   "embedding/head slices; "
                                   "synthetic target-layer features; not "
                                   "acceptance or full target "
                                   "throughput"},
                         {"source_repo", draft_artifact.summary().source_repo},
                         {"source_revision",
                          draft_artifact.summary().source_revision},
                         {"resident_draft_bytes",
                          draft_weights.stats().resident_bytes},
                         {"target_embedding_head_bytes", shared_weight_bytes},
                         {"max_context", max_context},
                         {"warmups", warmups},
                         {"repeats", repeats},
                         {"m4_short_routes", enable_short_routes},
                         {"selector_mode",
                          gpu_selector ? "gpu_compact" : "cpu_diagnostic"},
                         {"records", records},
                         {"pipelines_compiled", registry.pipelines_compiled},
                         {"plans_built", registry.command_plans_built},
                     })
                     .dump(2)
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "DFlash slice benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
