#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "omlx_m5/engine.h"
#include "omlx_m5/metal.h"
#include "../src/native_backend.h"

using namespace omlx::m5;
using namespace std::chrono_literals;
namespace {
void require(bool value, const char *why) {
  if (!value) throw std::runtime_error(why);
}
LoadOptions options(const std::filesystem::path &fixtures,
                    const std::filesystem::path &libraries, int concurrency) {
  LoadOptions o;
  const auto root = fixtures / "affine";
  o.model_dir = (root / "source").string();
  o.max_context = 1024;
  o.max_concurrency = concurrency;
  o.path_overrides["artifact"] = (root / "model.omlxm5").string();
  o.path_overrides["qwen_ar_metallib"] =
      (libraries / "omlx_m5_qwen_ar.metallib").string();
  return o;
}
SubmitRequest request(int seed, int length, int generated) {
  SubmitRequest r;
  for (int i = 0; i < length; ++i)
    r.prompt_tokens.push_back((71 * i + seed) % 257);
  r.max_new_tokens = generated;
  r.sampling.seed = seed;
  return r;
}
struct Result {
  std::vector<std::int32_t> tokens;
  EventKind terminal = EventKind::tokens;
  std::string reason;
};
std::map<std::uint64_t, Result> collect(
    Engine &engine, const std::vector<std::uint64_t> &ids,
    std::vector<std::uint64_t> *order = nullptr) {
  std::map<std::uint64_t, Result> result;
  const std::set<std::uint64_t> expected(ids.begin(), ids.end());
  const auto deadline = std::chrono::steady_clock::now() + 60s;
  while (std::chrono::steady_clock::now() < deadline) {
    for (const auto &event : engine.poll_events(20ms)) {
      require(expected.contains(event.request_id), "P09 foreign event");
      auto &row = result[event.request_id];
      if (event.kind == EventKind::tokens) {
        if (order) order->push_back(event.request_id);
        row.tokens.insert(row.tokens.end(), event.tokens.begin(),
                          event.tokens.end());
      } else {
        require(row.terminal == EventKind::tokens, "P09 duplicate terminal");
        row.terminal = event.kind;
        row.reason = event.finish_reason;
      }
    }
    if (result.size() == ids.size()) {
      bool done = true;
      for (auto id : ids) done &= result.at(id).terminal != EventKind::tokens;
      if (done) return result;
    }
  }
  throw std::runtime_error("P09 request starvation or timeout");
}
void check_native(const std::filesystem::path &fixtures,
                  const std::filesystem::path &libraries, int concurrency,
                  bool speculative, const std::filesystem::path &draft_fixture,
                  bool sampled = false, bool batched = false,
                  int sample_top_k = 16) {
  auto o = options(fixtures, libraries, concurrency);
  if (batched)
    o.path_overrides["batch_mode"] = "on";
  else if (concurrency >= 2)
    o.path_overrides["batch_mode"] = "off";
  if (speculative) {
    o.draft_dir = (draft_fixture / "source").string();
    o.path_overrides["draft_artifact"] =
        (draft_fixture / "model.omlxm5").string();
    o.path_overrides["dflash2_metallib"] =
        (libraries / "omlx_m5_dflash2_program.metallib").string();
    o.speculative = {.backend = SpeculativeBackend::dflash2, .draft_tokens = 7};
  }
  if (concurrency == 2 && !speculative) {
    auto with_prefix = o;
    with_prefix.path_overrides["prefix_cache_bytes"] = "4096";
    auto bounded = Engine::create(with_prefix);
    require(bounded->capacity().max_concurrency == 2,
            "P10 per-slot prefix cache failed C2 admission");
    bounded->shutdown();
  }
  auto serial_options = o;
  serial_options.max_concurrency = 1;
  serial_options.path_overrides.erase("batch_mode");
  auto serial = Engine::create(serial_options);
  auto parallel = Engine::create(o);
  std::vector<SubmitRequest> inputs;
  for (int i = 0; i < concurrency; ++i) {
    auto input = request(11 + i, speculative ? 33 + i : 5 + i * 3,
                         batched && speculative ? 2 + i : 3 + i % 4);
    if (sampled)
      input.sampling = {.temperature = 0.7F, .top_p = 0.9F,
                        .top_k = sample_top_k,
                        .seed = std::uint64_t(100 + i)};
    inputs.push_back(std::move(input));
  }
  std::vector<std::vector<std::int32_t>> oracle;
  for (const auto &input : inputs) {
    const auto id = serial->submit(input);
    auto result = collect(*serial, {id}).at(id);
    require(result.terminal == EventKind::finished, "P09 serial oracle failed");
    oracle.push_back(std::move(result.tokens));
  }
  std::vector<std::uint64_t> ids;
  for (const auto &input : inputs) ids.push_back(parallel->submit(input));
  std::vector<std::uint64_t> order;
  const auto results = collect(*parallel, ids, &order);
  for (int i = 0; i < concurrency; ++i) {
    const auto &row = results.at(ids[i]);
    require(row.terminal == EventKind::finished && row.tokens == oracle[i],
            "P09 concurrent slot output differs from serial oracle");
  }
  if (speculative) {
    const auto stats = parallel->stats();
    require(stats.spec_rounds > 0 &&
                stats.spec_rounds == stats.spec_target_verify_calls &&
                stats.spec_rounds == stats.spec_replay_calls,
            "P09 DFlash slot lost a verify/replay transaction");
  }
  if (batched) {
    const auto stats = parallel->stats();
    require(stats.batch_submissions > 0 &&
                (speculative ? stats.batched_verify_rounds
                             : stats.batched_ar_steps) > 0,
            "P10 selected batch route did not execute");
    if (!speculative &&
        MetalContext::create().capabilities().profile.compute_icb_eligible)
      require(stats.batch_sampler_icb_replays > 0,
              "P10 installed batch sampler ICB was not replayed");
  }
  require(parallel->capacity().active_requests == 0 &&
              parallel->capacity().queued_requests == 0 &&
              parallel->capacity().max_concurrency == concurrency,
          "P09 slot cleanup/capacity incorrect");
  if (concurrency >= 2)
    require(order.size() >= 2 && order[0] != order[1],
            "P09 requests did not interleave before first outputs");
  auto reuse = request(99, speculative ? 33 : 8, 4);
  const auto reuse_id = parallel->submit(reuse);
  const auto oracle_id = serial->submit(reuse);
  const auto reuse_expected = collect(*serial, {oracle_id}).at(oracle_id);
  require(collect(*parallel, {reuse_id}).at(reuse_id).tokens ==
              reuse_expected.tokens && reuse_expected.reason == "length",
          "P09 slot reuse leaked previous state");
  if (concurrency == 2) {
    auto long_request = request(77, speculative ? 33 : 8, 400);
    const auto cancelled = parallel->submit(long_request);
    bool saw_active_token = false;
    const auto cancel_deadline = std::chrono::steady_clock::now() + 5s;
    while (!saw_active_token &&
           std::chrono::steady_clock::now() < cancel_deadline)
      for (const auto &event : parallel->poll_events(20ms)) {
        require(event.request_id == cancelled &&
                    event.kind == EventKind::tokens,
                "P09 active request terminated before cancellation");
        saw_active_token = true;
      }
    require(saw_active_token && parallel->cancel(cancelled),
            "P09 active cancel rejected");
    const auto aborted = collect(*parallel, {cancelled}).at(cancelled);
    require(aborted.terminal == EventKind::aborted,
            "P09 cancelled active slot did not terminate");
    const auto after_id = parallel->submit(reuse);
    const auto serial_after_id = serial->submit(reuse);
    require(collect(*parallel, {after_id}).at(after_id).tokens ==
                collect(*serial, {serial_after_id}).at(serial_after_id).tokens,
            "P09 cancellation poisoned reused slot");
    auto stop = reuse;
    stop.stop_token_seqs = {{reuse_expected.tokens.front()}};
    const auto stop_id = parallel->submit(stop);
    const auto serial_stop_id = serial->submit(stop);
    const auto stopped = collect(*parallel, {stop_id}).at(stop_id);
    const auto stopped_oracle = collect(*serial, {serial_stop_id}).at(serial_stop_id);
    require(stopped.terminal == EventKind::finished &&
                stopped.reason == "stop" && stopped.tokens.empty() &&
                stopped.tokens == stopped_oracle.tokens &&
                stopped_oracle.reason == "stop",
            "P09 stop boundary differs from serial");
    auto invalid = reuse;
    invalid.sampling.top_k = 65;
    const auto invalid_id = parallel->submit(invalid);
    require(collect(*parallel, {invalid_id}).at(invalid_id).terminal ==
                EventKind::failed &&
                parallel->stats().fallback_reason == "unsupported_sampling",
            "P09 failed request was not rejected/recorded");
    const auto recovered_id = parallel->submit(reuse);
    require(collect(*parallel, {recovered_id}).at(recovered_id).tokens ==
                reuse_expected.tokens,
            "P09 failed request poisoned slot reuse");
  }
  parallel->shutdown();
  serial->shutdown();
}
void check_prefill_fairness(const std::filesystem::path &fixtures,
                            const std::filesystem::path &libraries,
                            bool batched = false) {
  auto o = options(fixtures, libraries, 2);
  o.max_context = 2048;
  o.path_overrides["batch_mode"] = batched ? "on" : "off";
  auto serial_o = o;
  serial_o.max_concurrency = 1;
  serial_o.path_overrides.erase("batch_mode");
  auto serial = Engine::create(serial_o);
  auto concurrent = Engine::create(o);
  const auto long_prompt = request(41, 1030, 4);
  const auto short_prompt = request(42, 5, 4);
  const auto long_oracle_id = serial->submit(long_prompt);
  const auto long_oracle = collect(*serial, {long_oracle_id}).at(long_oracle_id);
  const auto short_oracle_id = serial->submit(short_prompt);
  const auto short_oracle = collect(*serial, {short_oracle_id}).at(short_oracle_id);
  const auto long_id = concurrent->submit(long_prompt);
  const auto short_id = concurrent->submit(short_prompt);
  std::vector<std::uint64_t> order;
  const auto rows = collect(*concurrent, {long_id, short_id}, &order);
  require(rows.at(long_id).tokens == long_oracle.tokens &&
              rows.at(short_id).tokens == short_oracle.tokens &&
              !order.empty() && order.front() == short_id,
          "P09 decode-ready request starved behind chunked prefill");
  if (batched)
    require(concurrent->stats().batch_fallback_steps > 0,
            "P10 large prefill did not use whole-step fallback");
  concurrent->shutdown();
  serial->shutdown();
}
void check_mivr13_decode_protect_and_short_prefill(
    const std::filesystem::path &fixtures,
    const std::filesystem::path &libraries,
    const std::filesystem::path &draft_fixture) {
  // (a) DFlash2 decode stream (first decode step produced_==0 is non-batchable!)
  // concurrent with a long 1030-token prefill must still be recognized as an
  // active decode stream (decode_slots) and flush in_flight_commit cleanly.
  {
    auto o = options(fixtures, libraries, 2);
    o.max_context = 2048;
    o.path_overrides["batch_mode"] = "on";
    o.draft_dir = (draft_fixture / "source").string();
    o.path_overrides["draft_artifact"] =
        (draft_fixture / "model.omlxm5").string();
    o.path_overrides["dflash2_metallib"] =
        (libraries / "omlx_m5_dflash2_program.metallib").string();
    o.speculative = {.backend = SpeculativeBackend::dflash2, .draft_tokens = 7};
    auto serial_o = o;
    serial_o.max_concurrency = 1;
    serial_o.path_overrides.erase("batch_mode");
    auto serial = Engine::create(serial_o);
    auto concurrent = Engine::create(o);

    const auto short_decode = request(51, 33, 12);
    const auto long_prefill = request(52, 1030, 4);
    const auto sd_oid = serial->submit(short_decode);
    const auto sd_oracle = collect(*serial, {sd_oid}).at(sd_oid);
    const auto lp_oid = serial->submit(long_prefill);
    const auto lp_oracle = collect(*serial, {lp_oid}).at(lp_oid);

    // Admit short_decode first so it reaches decode (produced_ == 0 -> non-batchable
    // DFlash first step or in-flight commit), then submit long_prefill concurrently.
    const auto sd_id = concurrent->submit(short_decode);
    const auto lp_id = concurrent->submit(long_prefill);
    std::vector<std::uint64_t> order;
    const auto rows = collect(*concurrent, {sd_id, lp_id}, &order);
    require(rows.at(sd_id).terminal == EventKind::finished &&
                rows.at(lp_id).terminal == EventKind::finished &&
                rows.at(sd_id).tokens == sd_oracle.tokens &&
                rows.at(lp_id).tokens == lp_oracle.tokens,
            "MIVR-13 DFlash decode + long prefill diverged from serial oracle");
    require(!order.empty() && order.front() == sd_id,
            "MIVR-13 DFlash active decode starved behind long prefill");
    concurrent->shutdown();
    serial->shutdown();
  }

  // (b) Pure prefill (short <=8 token prefill with max_new_tokens=1 + long
  // prefill, zero active decodes): short prefill must NOT be misclassified as
  // an active decode stream.
  {
    auto o = options(fixtures, libraries, 2);
    o.max_context = 2048;
    o.path_overrides["batch_mode"] = "on";
    auto serial_o = o;
    serial_o.max_concurrency = 1;
    serial_o.path_overrides.erase("batch_mode");
    auto serial = Engine::create(serial_o);
    auto concurrent = Engine::create(o);

    const auto short_pf = request(61, 6, 1);
    const auto long_pf = request(62, 1030, 2);
    const auto sp_oid = serial->submit(short_pf);
    const auto sp_oracle = collect(*serial, {sp_oid}).at(sp_oid);
    const auto lp_oid = serial->submit(long_pf);
    const auto lp_oracle = collect(*serial, {lp_oid}).at(lp_oid);

    const auto sp_id = concurrent->submit(short_pf);
    const auto lp_id = concurrent->submit(long_pf);
    const auto rows = collect(*concurrent, {sp_id, lp_id});
    require(rows.at(sp_id).tokens == sp_oracle.tokens &&
                rows.at(lp_id).tokens == lp_oracle.tokens,
            "MIVR-13 short prefill + long prefill diverged from serial oracle");
    concurrent->shutdown();
    serial->shutdown();
  }
}
void check_queue_bound() {
  LoadOptions o;
  o.max_concurrency = 2;
  auto engine = Engine::create_fake(o, {.event_batch_size = 1,
                                        .step_delay_us = 10000});
  const auto input = request(1, 2, 100);
  std::vector<std::uint64_t> ids{engine->submit(input), engine->submit(input)};
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (engine->capacity().active_requests < 2 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(1ms);
  require(engine->capacity().active_requests == 2,
          "P09 two active slots were not admitted");
  for (int i = 0; i < 8; ++i) ids.push_back(engine->submit(input));
  require(engine->capacity().queued_requests == 8 &&
              !engine->capacity().accepting,
          "P09 bounded FIFO capacity did not close");
  bool rejected = false;
  try { static_cast<void>(engine->submit(input)); }
  catch (const std::runtime_error &) { rejected = true; }
  require(rejected, "P09 queue overflow was admitted");
  for (auto id : ids) require(engine->cancel(id), "P09 cancel queued/active failed");
  const auto rows = collect(*engine, ids);
  for (auto id : ids)
    require(rows.at(id).terminal == EventKind::aborted,
            "P09 cancellation did not drain bounded queue");
  require(engine->capacity().active_requests == 0 &&
              engine->capacity().queued_requests == 0,
          "P09 bounded queue leaked slots after cancellation");
  engine->shutdown();
}
void check_batch_compaction(const std::filesystem::path &fixtures,
                            const std::filesystem::path &libraries) {
  auto o = options(fixtures, libraries, 4);
  o.path_overrides["batch_mode"] = "on";
  auto serial_o = o;
  serial_o.max_concurrency = 1;
  serial_o.path_overrides.erase("batch_mode");
  auto serial = Engine::create(serial_o);
  auto batch = Engine::create(o);
  std::vector<SubmitRequest> inputs;
  std::vector<std::uint64_t> ids;
  for (int i = 0; i < 4; ++i) {
    inputs.push_back(request(81 + i, 33 + i, 8));
    ids.push_back(batch->submit(inputs.back()));
  }
  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (batch->capacity().active_requests < 4 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(1ms);
  require(batch->capacity().active_requests == 4 && batch->cancel(ids[1]),
          "P10 could not cancel a physical middle slot");
  const auto rows = collect(*batch, ids);
  require(rows.at(ids[1]).terminal == EventKind::aborted,
          "P10 canceled row emitted a terminal result incorrectly");
  for (int i : {0, 2, 3}) {
    const auto oracle_id = serial->submit(inputs[i]);
    const auto oracle = collect(*serial, {oracle_id}).at(oracle_id);
    require(rows.at(ids[i]).tokens == oracle.tokens &&
                rows.at(ids[i]).terminal == EventKind::finished,
            "P10 compact physical slot mapping crossed row state");
  }
  const auto stats = batch->stats();
  require(stats.batch_size_counts[3] > 0 && stats.batch_rows >= 3 &&
              stats.batch_width_counts[1] > 0,
          "P10 compact valid-mask plan did not execute");
  batch->shutdown();
  serial->shutdown();
}
void check_batch_prefix(const std::filesystem::path &fixtures,
                        const std::filesystem::path &libraries) {
  auto o = options(fixtures, libraries, 2);
  o.path_overrides["batch_mode"] = "on";
  o.path_overrides["prefix_cache_bytes"] = "67108864";
  auto batch = Engine::create(o);
  o.max_concurrency = 1;
  o.path_overrides.erase("batch_mode");
  o.path_overrides.erase("prefix_cache_bytes");
  auto serial = Engine::create(o);
  std::array<SubmitRequest, 2> requests{request(71, 5, 5),
                                         request(72, 7, 5)};
  for (int round = 0; round < 3; ++round) {
    if (round == 2)
      for (auto &row : requests) row.prompt_tokens.push_back(37);
    std::vector<std::uint64_t> ids;
    for (const auto &row : requests) ids.push_back(batch->submit(row));
    const auto actual = collect(*batch, ids);
    for (std::size_t i = 0; i < requests.size(); ++i) {
      const auto oracle_id = serial->submit(requests[i]);
      require(actual.at(ids[i]).tokens ==
                  collect(*serial, {oracle_id}).at(oracle_id).tokens,
              "P10 per-slot prefix restore changed batched output");
    }
  }
  const auto stats = batch->stats();
  require(stats.prefix_hits >= 2 && stats.prefix_partial_hits >= 2 &&
              stats.batch_submissions > 0,
          "P10 per-slot complete/partial prefix was not restored");
  batch->shutdown();
  serial->shutdown();
}
void check_mive14_solo_ar_to_batch_transition(
    const std::filesystem::path &fixtures,
    const std::filesystem::path &libraries) {
  auto o = options(fixtures, libraries, 4);
  o.max_context = 2048;
  o.path_overrides["batch_mode"] = "on";
  auto serial_o = o;
  serial_o.max_concurrency = 1;
  serial_o.path_overrides.erase("batch_mode");
  auto serial = Engine::create(serial_o);

  auto oracle_tokens = [&](const SubmitRequest &req) {
    const auto id = serial->submit(req);
    const auto row = collect(*serial, {id}).at(id);
    require(row.terminal == EventKind::finished,
            "MIVE-14 serial oracle request failed");
    return row.tokens;
  };

  detail::NativeBackend backend(o);
  backend.bind_owner(std::this_thread::get_id());

  auto run_all_in_batch = [&](std::vector<std::unique_ptr<detail::BackendTask>> &tasks,
                              std::vector<std::optional<detail::ExecutionResult>> &done) {
    for (int iter = 0; iter < 128; ++iter) {
      std::vector<std::size_t> active_idx;
      std::vector<detail::BackendTask *> active_ptrs;
      for (std::size_t i = 0; i < tasks.size(); ++i) {
        if (tasks[i]) {
          active_idx.push_back(i);
          active_ptrs.push_back(tasks[i].get());
        }
      }
      if (active_ptrs.empty())
        return;
      if (active_ptrs.size() == 1U) {
        const std::size_t idx = active_idx[0];
        if (auto out = tasks[idx]->advance()) {
          done[idx] = std::move(out);
          tasks[idx].reset();
        }
      } else {
        auto outcomes = backend.advance_batch(active_ptrs);
        require(outcomes.size() == active_ptrs.size(),
                "MIVE-14 advance_batch outcome count mismatch");
        for (std::size_t k = 0; k < outcomes.size(); ++k) {
          if (outcomes[k].has_value()) {
            const std::size_t idx = active_idx[k];
            done[idx] = std::move(outcomes[k]);
            tasks[idx].reset();
          }
        }
      }
    }
    throw std::runtime_error("MIVE-14 batch loop exceeded iteration bound");
  };

  // (a) Solo slot 0 builds in_flight_ar_ (ar_in_flight_count_ == 1), then two
  // short <=8-token requests arrive and enter Case 2 packed batch.
  {
    const auto r0 = request(91, 6, 8);
    const auto r1 = request(92, 5, 4);
    const auto r2 = request(93, 6, 4);
    const auto exp0 = oracle_tokens(r0);
    const auto exp1 = oracle_tokens(r1);
    const auto exp2 = oracle_tokens(r2);

    std::vector<std::int32_t> out0, out1, out2;
    auto c0 = std::make_shared<std::atomic<bool>>(false);
    auto c1 = std::make_shared<std::atomic<bool>>(false);
    auto c2 = std::make_shared<std::atomic<bool>>(false);

    std::vector<std::unique_ptr<detail::BackendTask>> tasks(3);
    std::vector<std::optional<detail::ExecutionResult>> done(3);
    tasks[0] = backend.start(r0, c0, [&](std::vector<std::int32_t> toks) {
      out0.insert(out0.end(), toks.begin(), toks.end());
    });
    // Step 1: solo prefill; Step 2: solo AR step leaving second_ar in_flight_ar_.
    require(!tasks[0]->advance().has_value() && !tasks[0]->advance().has_value(),
            "MIVE-14 solo slot 0 finished before pipelining in_flight_ar_");

    tasks[1] = backend.start(r1, c1, [&](std::vector<std::int32_t> toks) {
      out1.insert(out1.end(), toks.begin(), toks.end());
    });
    tasks[2] = backend.start(r2, c2, [&](std::vector<std::int32_t> toks) {
      out2.insert(out2.end(), toks.begin(), toks.end());
    });

    run_all_in_batch(tasks, done);
    require(done[0] && done[0]->kind == EventKind::finished && out0 == exp0 &&
                done[1] && done[1]->kind == EventKind::finished && out1 == exp1 &&
                done[2] && done[2]->kind == EventKind::finished && out2 == exp2,
            "MIVE-14 solo in_flight_ar_ -> Case 2 packed batch failed or diverged");
  }

  // (b) Cancelled solo slot 0 with active in_flight_ar_ + concurrent >16-token
  // delegated prefill on slot 1 (Case 1).
  {
    const auto r0 = request(94, 6, 8);
    const auto r1 = request(95, 24, 4);
    const auto exp1 = oracle_tokens(r1);

    std::vector<std::int32_t> out0, out1;
    auto c0 = std::make_shared<std::atomic<bool>>(false);
    auto c1 = std::make_shared<std::atomic<bool>>(false);
    std::vector<std::unique_ptr<detail::BackendTask>> tasks(2);
    std::vector<std::optional<detail::ExecutionResult>> done(2);

    tasks[0] = backend.start(r0, c0, [&](std::vector<std::int32_t> toks) {
      out0.insert(out0.end(), toks.begin(), toks.end());
    });
    require(!tasks[0]->advance().has_value() && !tasks[0]->advance().has_value(),
            "MIVE-14 solo slot 0 (b) finished before pipelining in_flight_ar_");
    c0->store(true, std::memory_order_release);

    tasks[1] = backend.start(r1, c1, [&](std::vector<std::int32_t> toks) {
      out1.insert(out1.end(), toks.begin(), toks.end());
    });
    run_all_in_batch(tasks, done);
    require(done[0] && done[0]->kind == EventKind::aborted && done[1] &&
                done[1]->kind == EventKind::finished && out1 == exp1,
            "MIVE-14 cancelled slot 0 in_flight_ar_ broke delegated prefill");
  }

  // (c) Cancelled solo slot 0 with active in_flight_ar_ + two 9..512-token
  // prefills entering Packed Multi-Slot TTFT Prefill.
  {
    const auto r0 = request(96, 6, 8);
    const auto r1 = request(97, 24, 4);
    const auto r2 = request(98, 20, 4);
    const auto exp1 = oracle_tokens(r1);
    const auto exp2 = oracle_tokens(r2);

    std::vector<std::int32_t> out0, out1, out2;
    auto c0 = std::make_shared<std::atomic<bool>>(false);
    auto c1 = std::make_shared<std::atomic<bool>>(false);
    auto c2 = std::make_shared<std::atomic<bool>>(false);
    std::vector<std::unique_ptr<detail::BackendTask>> tasks(3);
    std::vector<std::optional<detail::ExecutionResult>> done(3);

    tasks[0] = backend.start(r0, c0, [&](std::vector<std::int32_t> toks) {
      out0.insert(out0.end(), toks.begin(), toks.end());
    });
    require(!tasks[0]->advance().has_value() && !tasks[0]->advance().has_value(),
            "MIVE-14 solo slot 0 (c) finished before pipelining in_flight_ar_");
    c0->store(true, std::memory_order_release);

    tasks[1] = backend.start(r1, c1, [&](std::vector<std::int32_t> toks) {
      out1.insert(out1.end(), toks.begin(), toks.end());
    });
    tasks[2] = backend.start(r2, c2, [&](std::vector<std::int32_t> toks) {
      out2.insert(out2.end(), toks.begin(), toks.end());
    });
    run_all_in_batch(tasks, done);
    require(done[0] && done[0]->kind == EventKind::aborted && done[1] &&
                done[1]->kind == EventKind::finished && out1 == exp1 &&
                done[2] && done[2]->kind == EventKind::finished && out2 == exp2,
            "MIVE-14 cancelled slot 0 in_flight_ar_ broke packed TTFT prefill");
  }

  serial->shutdown();
}
void check_fake() {
  LoadOptions o;
  o.max_concurrency = 2;
  auto engine = Engine::create_fake(o, {.event_batch_size = 1,
                                        .step_delay_us = 1000});
  const auto first = engine->submit(request(1, 2, 6));
  const auto second = engine->submit(request(2, 2, 6));
  std::vector<std::uint64_t> order;
  const auto rows = collect(*engine, {first, second}, &order);
  require(rows.at(first).tokens.size() == 6 &&
              rows.at(second).tokens.size() == 6 && order.size() == 12 &&
              order[0] == first && order[1] == second,
          "P09 FIFO ready rotation failed");
  engine->shutdown();
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 4 && argc != 5)
      throw std::invalid_argument(
          "concurrency_tests QWEN_FIXTURES LIBRARIES DFLASH_FIXTURE [--p10]");
    if (argc == 5) {
      if (std::string_view(argv[4]) != "--p10")
        throw std::invalid_argument("unknown concurrency test mode");
      for (int b = 2; b <= 8; ++b) {
        check_native(argv[1], argv[2], b, false, argv[3], false, true);
        check_native(argv[1], argv[2], b, true, argv[3], false, true);
      }
      check_prefill_fairness(argv[1], argv[2], true);
      check_mivr13_decode_protect_and_short_prefill(argv[1], argv[2], argv[3]);
      for (int top_k : {1, 16, 64}) {
        check_native(argv[1], argv[2], 2, false, argv[3], true, true, top_k);
        check_native(argv[1], argv[2], 2, true, argv[3], true, true, top_k);
      }
      check_batch_compaction(argv[1], argv[2]);
      check_batch_prefix(argv[1], argv[2]);
      check_mive14_solo_ar_to_batch_transition(argv[1], argv[2]);
      std::cout << "P10 AR/DFlash batch B2/B4/B8 W1..8 passed\n";
      return 0;
    }
    check_fake();
    check_queue_bound();
    check_prefill_fairness(argv[1], argv[2]);
    for (int b : {2, 4, 8})
      check_native(argv[1], argv[2], b, false, argv[3]);
    for (int b : {2, 4, 8})
      check_native(argv[1], argv[2], b, true, argv[3]);
    check_native(argv[1], argv[2], 2, false, argv[3], true);
    check_native(argv[1], argv[2], 2, true, argv[3], true);
    std::cout << "P09 FIFO, native and DFlash B2/B4/B8 passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "P09 failure: " << error.what() << '\n';
    return 1;
  }
}
