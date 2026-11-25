#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "omlx_m5/engine.h"
#include "omlx_m5/metal.h"

namespace {

using namespace std::chrono_literals;
using omlx::m5::Engine;
using omlx::m5::EventKind;
using omlx::m5::LoadOptions;
using omlx::m5::SpeculativeBackend;
using omlx::m5::SubmitRequest;
using omlx::m5::TokenEvent;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<TokenEvent> collect(Engine &engine, std::uint64_t request_id) {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  std::vector<TokenEvent> result;
  bool terminal = false;
  while (!terminal && std::chrono::steady_clock::now() < deadline) {
    for (auto &event : engine.poll_events(20ms)) {
      if (event.request_id != request_id) {
        throw std::runtime_error("native backend returned a foreign event");
      }
      terminal = event.kind != EventKind::tokens;
      result.push_back(std::move(event));
    }
  }
  expect(terminal, "native backend request did not terminate");
  return result;
}

std::vector<std::int32_t> emitted(const std::vector<TokenEvent> &events) {
  std::vector<std::int32_t> result;
  for (const auto &event : events) {
    if (event.kind == EventKind::tokens) {
      result.insert(result.end(), event.tokens.begin(), event.tokens.end());
    }
  }
  return result;
}

LoadOptions ar_options(const std::filesystem::path &fixtures,
                       const std::filesystem::path &qwen_metallib,
                       bool control_fixture = false) {
  LoadOptions options;
  const auto real = fixtures.parent_path() / "qwen-execution/affine";
  options.model_dir =
      (control_fixture ? fixtures / "ar-dense-oq" : real / "source").string();
  options.max_context = 1024;
  options.path_overrides["artifact"] =
      (control_fixture ? fixtures / "ar-dense-oq.omlxm5"
                       : real / "model.omlxm5")
          .string();
  options.path_overrides["qwen_ar_metallib"] = qwen_metallib.string();
  return options;
}

void test_ar_native(const std::filesystem::path &fixtures,
                    const std::filesystem::path &qwen_metallib) {
  auto engine = Engine::create(ar_options(fixtures, qwen_metallib));
  SubmitRequest request;
  request.prompt_tokens = {1, 7, 3, 9, 2};
  request.max_new_tokens = 8;
  const auto first = collect(*engine, engine->submit(request));
  const auto first_tokens = emitted(first);
  expect(first.back().kind == EventKind::finished &&
             first.back().finish_reason == "length" && first_tokens.size() == 8,
         "native AR request failed");
  const auto stats = engine->stats();
  expect(stats.backend_kind == "native" && !stats.artifact_id.empty() &&
             !stats.route_table_id.empty() && stats.actual_t == 1 &&
             stats.actual_m == 1 && stats.actual_k == 0 &&
             stats.kernel_family == "qwen_forward_a16_metal4" &&
             stats.model_execution_kind == "qwen_dense",
         "native AR statistics are incomplete");
  expect(stats.ar_steps == 8 && stats.ar_layer_calls == 16 &&
             stats.ar_host_read_bytes == 32 && stats.short_matrix_calls == 0 &&
             stats.short_lane_calls == 0,
         "native AR did not execute every layer or read only selected tokens");

  auto legacy_options = ar_options(fixtures, qwen_metallib);
  legacy_options.path_overrides["m4_short_routes"] = "off";
  auto legacy = Engine::create(legacy_options);
  const auto legacy_tokens = emitted(collect(*legacy, legacy->submit(request)));
  expect(legacy_tokens == first_tokens &&
             legacy->stats().route_table_id.find(":m4-short-off") !=
                 std::string::npos,
         "forced legacy short route changed model output or identity");
  legacy->shutdown();

  if (omlx::m5::MetalContext::create()
          .capabilities()
          .profile.probe.apple_gpu_family >= 9) {
    auto split_options = ar_options(fixtures, qwen_metallib);
    split_options.path_overrides["attention_split_kv"] = "on";
    auto split = Engine::create(split_options);
    const auto split_tokens = emitted(collect(*split, split->submit(request)));
    expect(split_tokens == first_tokens &&
               split->stats().route_table_id.find(
                   "attention-split-diagnostic-") != std::string::npos,
           "P06 actual Engine split profile changed control output/route");
    split->shutdown();

    auto short_on_options = ar_options(fixtures, qwen_metallib);
    short_on_options.path_overrides["m4_short_routes"] = "on";
    auto short_on = Engine::create(short_on_options);
    expect(emitted(collect(*short_on, short_on->submit(request))) ==
                   first_tokens &&
               short_on->stats().route_table_id.find(":m4-short-on") !=
                   std::string::npos,
           "explicit m4_short_routes=on failed on Apple9+ GPU");
    short_on->shutdown();
  }

  request.stop_token_seqs = {{first_tokens.front()}};
  const auto stopped = collect(*engine, engine->submit(request));
  expect(emitted(stopped).empty() && stopped.back().finish_reason == "stop",
         "native token-stop boundary failed");

  request.stop_token_seqs.clear();
  request.sampling.min_p = 0.1F;
  const auto unsupported = collect(*engine, engine->submit(request));
  expect(unsupported.size() == 1 &&
             unsupported.front().kind == EventKind::failed &&
             !unsupported.front().error.empty(),
         "unsupported native request did not fail before first token");
  expect(engine->stats().fallback_reason == "unsupported_sampling",
         "native fallback reason was not exposed");
  engine->shutdown();
}

void test_dflash_native(const std::filesystem::path &fixtures,
                        const std::filesystem::path &qwen_metallib,
                        const std::filesystem::path &draft_artifact,
                        const std::filesystem::path &draft_config,
                        const std::filesystem::path &dflash_metallib) {
  auto options = ar_options(fixtures, qwen_metallib, true);
  options.draft_dir = draft_config.parent_path().string();
  options.speculative.backend = SpeculativeBackend::dflash2;
  options.speculative.draft_tokens = 7;
  options.path_overrides["draft_artifact"] = draft_artifact.string();
  options.path_overrides["dflash2_metallib"] = dflash_metallib.string();
  bool rejected = false;
  try {
    auto engine = Engine::create(options);
  } catch (const std::invalid_argument &error) {
    rejected =
        std::string(error.what()).find("companion/target geometry mismatch") !=
        std::string::npos;
  }
  expect(rejected, "production factory accepted the DFlash control fixture");
}

void test_real_dflash(const std::filesystem::path &fixtures,
                      const std::filesystem::path &library) {
  auto options = ar_options(fixtures, library);
  auto ar = Engine::create(options);
  SubmitRequest request;
  for (int i = 0; i < 33; ++i)
    request.prompt_tokens.push_back((71 * i + 256) % 257);
  request.max_new_tokens = 24;
  auto baseline = emitted(collect(*ar, ar->submit(request)));
  ar->shutdown();
  auto draft = fixtures.parent_path() / "dflash-execution";
  options.draft_dir = (draft / "source").string();
  options.path_overrides["draft_artifact"] = (draft / "model.omlxm5").string();
  options.speculative = {.backend = SpeculativeBackend::dflash2,
                         .draft_tokens = 7};
  auto engine = Engine::create(options);
  auto first = collect(*engine, engine->submit(request));
  expect(first.back().kind == EventKind::finished,
         "real DFlash failed: " + first.back().error);
  expect(emitted(first) == baseline,
         "real DFlash greedy continuation differs from AR");
  const bool gpu_capable = omlx::m5::MetalContext::create()
                               .capabilities()
                               .profile.probe.apple_gpu_family >= 9;
  if (gpu_capable) {
    auto split_options = options;
    split_options.path_overrides["attention_split_kv"] = "on";
    auto split_engine = Engine::create(split_options);
    const auto split_tokens =
        emitted(collect(*split_engine, split_engine->submit(request)));
    expect(split_tokens == baseline && split_engine->stats().spec_rounds > 0 &&
               split_engine->stats().route_table_id.find(
                   "attention-split-diagnostic-") != std::string::npos,
           "P06 DFlash target split profile changed greedy output");
    split_engine->shutdown();
  }
  auto stats = engine->stats();
  expect(stats.model_execution_kind == "qwen_dense" && stats.spec_rounds > 0 &&
             stats.spec_rounds == stats.spec_target_verify_calls &&
             stats.spec_rounds == stats.spec_replay_calls &&
             stats.ar_steps == 1 && stats.spec_context_rows_projected > 0 &&
             stats.spec_context_rows_reused > 0 &&
             stats.spec_context_rows_launched >=
                 stats.spec_context_rows_projected,
         "real DFlash repeated target forward or missing replay");
  for (int budget = 2; budget <= 9; ++budget) {
    request.max_new_tokens = budget;
    const auto bounded = emitted(collect(*engine, engine->submit(request)));
    expect(bounded == std::vector<std::int32_t>(baseline.begin(),
                                                baseline.begin() + budget),
           "P04 K=0..7 request budget changed greedy continuation");
  }
  request.max_new_tokens = 24;
  expect(stats.route_table_id.find(gpu_capable ? "selector-gpu-"
                                               : "selector-cpu-pending-target") !=
                 std::string::npos &&
             (!gpu_capable ||
              (stats.spec_host_read_bytes > 0 &&
               stats.spec_host_read_bytes < stats.spec_rounds * 6000)),
         "P03 production selector route/readback mismatch");
  request.stop_token_seqs = {{baseline[4], baseline[5], baseline[6]}};
  auto stopped = collect(*engine, engine->submit(request));
  expect(stopped.back().finish_reason == "stop" &&
             emitted(stopped) == std::vector<std::int32_t>(
                                     baseline.begin(), baseline.begin() + 4),
         "DFlash stop sequence across rounds failed");
  request.stop_token_seqs.clear();
  request.max_new_tokens = 512;
  auto id = engine->submit(request);
  expect(engine->cancel(id), "cancel not accepted");
  expect(collect(*engine, id).back().kind == EventKind::aborted,
         "DFlash cancel failed");
  request.max_new_tokens = 24;
  expect(emitted(collect(*engine, engine->submit(request))) == baseline,
         "DFlash reset after cancel failed");
  request.sampling = {
      .temperature = 0.7F, .top_p = 0.9F, .top_k = 16, .seed = 123};
  auto sampled = emitted(collect(*engine, engine->submit(request)));
  expect(sampled.size() == 24 &&
             emitted(collect(*engine, engine->submit(request))) == sampled,
         "DFlash seeded sampling failed");
  if (gpu_capable) {
    auto gpu_fresh = Engine::create(options);
    const auto new_tokens =
        emitted(collect(*gpu_fresh, gpu_fresh->submit(request)));
    const auto gpu_stats = gpu_fresh->stats();
    auto cpu_options = options;
    cpu_options.path_overrides["target_verify_selector"] = "cpu";
    auto cpu_engine = Engine::create(cpu_options);
    const auto old = emitted(collect(*cpu_engine, cpu_engine->submit(request)));
    const auto cpu_stats = cpu_engine->stats();
    expect(old == sampled && new_tokens == sampled &&
               cpu_stats.route_table_id.find("verify-cpu-diagnostic") !=
                   std::string::npos &&
               gpu_stats.route_table_id.find("verify-gpu-") !=
                   std::string::npos &&
               gpu_stats.spec_verify_selector_gpu_ms > 0 &&
               cpu_stats.spec_verify_selector_gpu_ms == 0 &&
               cpu_stats.spec_host_read_bytes > gpu_stats.spec_host_read_bytes,
           "P04 actual request CPU/GPU verify A/B differs");
    gpu_fresh->shutdown();
    cpu_engine->shutdown();
  }
  engine->shutdown();
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 6) {
      throw std::invalid_argument(
          "usage: omlx_m5_b6_native_backend_tests TARGET_FIXTURES "
          "QWEN_METALLIB DRAFT_ARTIFACT DRAFT_CONFIG DFLASH_METALLIB");
    }
    test_ar_native(argv[1], argv[2]);
    test_real_dflash(argv[1], argv[2]);
    test_dflash_native(argv[1], argv[2], argv[3], argv[4], argv[5]);
    std::cout << "B6 native backend tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B6 native backend failure: " << error.what() << '\n';
    return 1;
  }
}
