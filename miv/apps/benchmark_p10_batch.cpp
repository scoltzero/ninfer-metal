#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "omlx_m5/engine.h"
#include "omlx_m5/metal.h"

using namespace omlx::m5;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace {
LoadOptions options(const std::filesystem::path &fixture,
                    const std::filesystem::path &libraries,
                    const std::filesystem::path &draft, int b,
                    bool batch, bool speculative) {
  LoadOptions o;
  o.model_dir = (fixture / "affine/source").string();
  o.max_context = 1024;
  o.max_concurrency = b;
  o.path_overrides["artifact"] = (fixture / "affine/model.omlxm5").string();
  o.path_overrides["qwen_ar_metallib"] =
      (libraries / "omlx_m5_qwen_ar.metallib").string();
  if (batch) o.path_overrides["batch_mode"] = "on";
  if (speculative) {
    o.draft_dir = (draft / "source").string();
    o.path_overrides["draft_artifact"] = (draft / "model.omlxm5").string();
    o.path_overrides["dflash2_metallib"] =
        (libraries / "omlx_m5_dflash2_program.metallib").string();
    o.speculative = {.backend = SpeculativeBackend::dflash2,
                     .draft_tokens = 7};
  }
  return o;
}
std::vector<std::vector<std::int32_t>> execute(Engine &engine, int active,
                                               bool speculative) {
  std::vector<std::uint64_t> ids;
  ids.reserve(active);
  for (int row = 0; row < active; ++row) {
    SubmitRequest request;
    for (int i = 0; i < (speculative ? 33 : 8 + row % 4); ++i)
      request.prompt_tokens.push_back((i * 71 + row * 13 + 29) % 257);
    request.max_new_tokens = speculative ? 9 : 12;
    request.sampling.seed = 101 + row;
    ids.push_back(engine.submit(std::move(request)));
  }
  std::vector<std::vector<std::int32_t>> tokens(active);
  std::vector<bool> done(active);
  const auto deadline = Clock::now() + std::chrono::seconds(60);
  while (std::count(done.begin(), done.end(), true) < active &&
         Clock::now() < deadline)
    for (const auto &event : engine.poll_events(std::chrono::milliseconds(20))) {
      const auto found = std::find(ids.begin(), ids.end(), event.request_id);
      if (found == ids.end()) throw std::runtime_error("P10 foreign event");
      const auto row = found - ids.begin();
      if (event.kind == EventKind::tokens)
        tokens[row].insert(tokens[row].end(), event.tokens.begin(),
                           event.tokens.end());
      else if (event.kind == EventKind::finished)
        done[row] = true;
      else
        throw std::runtime_error("P10 benchmark request failed: " + event.error);
    }
  if (std::count(done.begin(), done.end(), true) != active)
    throw std::runtime_error("P10 benchmark timeout");
  return tokens;
}
std::vector<std::uint64_t> difference(const std::vector<std::uint64_t> &after,
                                      const std::vector<std::uint64_t> &before) {
  if (after.size() != before.size()) throw std::logic_error("P10 stats shape");
  auto result = after;
  for (std::size_t i = 0; i < result.size(); ++i) result[i] -= before[i];
  return result;
}
double median(std::vector<double> values) {
  std::ranges::sort(values);
  return (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2;
}
Json measure(const std::filesystem::path &fixture,
             const std::filesystem::path &libraries,
             const std::filesystem::path &draft, int b, int active,
             bool batch, bool speculative, int repeats) {
  auto engine = Engine::create(options(fixture, libraries, draft,
                                       b, batch, speculative));
  for (int i = 0; i < 3; ++i) static_cast<void>(execute(*engine, active, speculative));
  std::vector<double> wall;
  std::vector<std::vector<std::int32_t>> oracle;
  const auto before = engine->stats();
  for (int i = 0; i < repeats; ++i) {
    const auto start = Clock::now();
    auto output = execute(*engine, active, speculative);
    wall.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    if (oracle.empty()) oracle = output;
    if (oracle != output) throw std::runtime_error("P10 output changed across repeats");
  }
  const auto after = engine->stats();
  engine->shutdown();
  return {{"configured_b", b}, {"active_b", active},
          {"speculative", speculative}, {"batch_mode", batch},
          {"wall_ms", wall}, {"wall_median_ms", median(wall)},
          {"batch_submissions", after.batch_submissions - before.batch_submissions},
          {"batched_rows", after.batch_rows - before.batch_rows},
          {"batch_gpu_ms", after.batch_gpu_ms - before.batch_gpu_ms},
          {"batch_sampler_icb_replays",
           after.batch_sampler_icb_replays - before.batch_sampler_icb_replays},
          {"batch_fallback_steps",
           after.batch_fallback_steps - before.batch_fallback_steps},
          {"batch_size_counts", difference(after.batch_size_counts,
                                             before.batch_size_counts)},
          {"batch_width_counts", difference(after.batch_width_counts,
                                              before.batch_width_counts)},
          {"tokens", oracle}};
}
std::array<Json, 2> measure_pair(const std::filesystem::path &fixture,
                                 const std::filesystem::path &libraries,
                                 const std::filesystem::path &draft,
                                 int b, int active, bool speculative,
                                 int repeats) {
  std::array<std::unique_ptr<Engine>, 2> engines{
      Engine::create(options(fixture, libraries, draft, b, false, speculative)),
      Engine::create(options(fixture, libraries, draft, b, true, speculative))};
  for (int warm = 0; warm < 3; ++warm)
    for (int offset = 0; offset < 2; ++offset)
      static_cast<void>(execute(*engines[(warm + offset) % 2], active,
                                speculative));
  std::array<StatsReport, 2> before{engines[0]->stats(), engines[1]->stats()};
  std::array<std::vector<double>, 2> wall;
  std::vector<std::vector<std::int32_t>> oracle;
  for (int iteration = 0; iteration < repeats; ++iteration)
    for (int offset = 0; offset < 2; ++offset) {
      const int selected = (iteration + offset) % 2;
      const auto start = Clock::now();
      auto output = execute(*engines[selected], active, speculative);
      wall[selected].push_back(
          std::chrono::duration<double, std::milli>(Clock::now() - start).count());
      if (oracle.empty()) oracle = output;
      if (oracle != output)
        throw std::runtime_error("P10 paired A/B changed output");
    }
  std::array<Json, 2> records;
  for (int selected = 0; selected < 2; ++selected) {
    const auto after = engines[selected]->stats();
    records[selected] = {
        {"configured_b", b}, {"active_b", active},
        {"speculative", speculative}, {"batch_mode", selected == 1},
        {"wall_ms", wall[selected]},
        {"wall_median_ms", median(wall[selected])},
        {"batch_submissions",
         after.batch_submissions - before[selected].batch_submissions},
        {"batched_rows", after.batch_rows - before[selected].batch_rows},
        {"batch_gpu_ms", after.batch_gpu_ms - before[selected].batch_gpu_ms},
        {"batch_sampler_icb_replays",
         after.batch_sampler_icb_replays -
             before[selected].batch_sampler_icb_replays},
        {"batch_fallback_steps",
         after.batch_fallback_steps - before[selected].batch_fallback_steps},
        {"batch_size_counts", difference(after.batch_size_counts,
                                           before[selected].batch_size_counts)},
        {"batch_width_counts", difference(after.batch_width_counts,
                                            before[selected].batch_width_counts)},
        {"tokens", oracle}};
    engines[selected]->shutdown();
  }
  return records;
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 6)
      throw std::invalid_argument(
          "benchmark_p10_batch QWEN_FIXTURES LIBRARIES DFLASH_FIXTURE REPEATS OUTPUT");
    const int repeats = std::stoi(argv[4]);
    if (repeats < 5) throw std::invalid_argument("P10 repeats >=5 required");
    Json records = Json::array();
    for (bool spec : {false, true}) {
      records.push_back(measure(argv[1], argv[2], argv[3], 1, 1, false,
                                spec, repeats));
      for (int b : {2, 4, 8}) {
        const auto paired = measure_pair(argv[1], argv[2], argv[3], b, b,
                                         spec, repeats);
        records.push_back(paired[0]);
        records.push_back(paired[1]);
      }
      // B1 traffic on an installed B2 service is separately paired.
      const auto single = measure_pair(argv[1], argv[2], argv[3], 2, 1,
                                       spec, repeats);
      records.push_back(single[0]);
      records.push_back(single[1]);
    }
    Json result = {{"schema", "omlx-m5-p10-batch-slice-v1"},
                   {"device", MetalContext::create().capabilities().device_name},
                   {"scope", "small real mixed Qwen/DFlash fixture; same prompts and seeds; wall includes submit/poll; no target-model or M5 Ultra inference"},
                   {"warmups", 3}, {"repeats", repeats},
                   {"pairing", "off/on per iteration with alternating order; dedicated B1 shown separately"},
                   {"records", records}};
    std::ofstream output(argv[5]);
    output << result.dump(2) << '\n';
    if (!output) throw std::runtime_error("P10 benchmark output failed");
    std::cout << argv[5] << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "P10 batch benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
