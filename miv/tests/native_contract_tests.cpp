#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "omlx_m5/engine.h"

namespace {

using namespace std::chrono_literals;
using omlx::m5::Engine;
using omlx::m5::EventKind;
using omlx::m5::FakeBackendOptions;
using omlx::m5::LoadOptions;
using omlx::m5::SpeculativeBackend;
using omlx::m5::SubmitRequest;
using omlx::m5::TokenEvent;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<TokenEvent>
collect_terminal(Engine &engine, std::uint64_t request_id,
                 std::chrono::milliseconds timeout = 2000ms) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::vector<TokenEvent> collected;
  bool terminal = false;
  while (!terminal && std::chrono::steady_clock::now() < deadline) {
    for (auto &event : engine.poll_events(20ms)) {
      if (event.request_id != request_id) {
        throw std::runtime_error("received an event for another request");
      }
      terminal = event.kind == EventKind::finished ||
                 event.kind == EventKind::aborted ||
                 event.kind == EventKind::failed;
      collected.push_back(std::move(event));
    }
  }
  expect(terminal, "request did not reach a terminal state");
  return collected;
}

void test_normal_generation() {
  FakeBackendOptions fake;
  fake.event_batch_size = 2;
  auto engine = Engine::create_fake({}, fake);

  SubmitRequest request;
  request.prompt_tokens = {7, 10};
  request.max_new_tokens = 5;
  const auto id = engine->submit(std::move(request));
  const auto events = collect_terminal(*engine, id);

  std::vector<std::int32_t> tokens;
  expect(events.back().kind == EventKind::finished, "expected finished event");
  expect(events.back().finish_reason == "length", "expected length finish");
  for (const auto &event : events) {
    if (event.kind == EventKind::tokens) {
      tokens.insert(tokens.end(), event.tokens.begin(), event.tokens.end());
    }
  }
  expect(tokens == std::vector<std::int32_t>({11, 12, 13, 14, 15}),
         "fake backend token sequence is not deterministic");

  const auto stats = engine->stats();
  expect(stats.submitted == 1, "submitted count mismatch");
  expect(stats.completed == 1, "completed count mismatch");
  expect(stats.emitted_tokens == 5, "emitted token count mismatch");
  expect(!engine->cancel(id), "terminal request must not remain cancellable");
}

void test_active_cancellation() {
  FakeBackendOptions fake;
  fake.step_delay_us = 2000;
  auto engine = Engine::create_fake({}, fake);

  SubmitRequest request;
  request.prompt_tokens = {1};
  request.max_new_tokens = 100;
  const auto id = engine->submit(std::move(request));

  bool saw_token = false;
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (!saw_token && std::chrono::steady_clock::now() < deadline) {
    for (const auto &event : engine->poll_events(20ms)) {
      if (event.kind == EventKind::tokens) {
        saw_token = true;
      }
    }
  }
  expect(saw_token, "cancellation test never observed a token");
  expect(engine->cancel(id), "active request was not cancellable");
  const auto events = collect_terminal(*engine, id);
  expect(events.back().kind == EventKind::aborted, "expected aborted event");
  expect(engine->stats().aborted == 1, "aborted count mismatch");
}

void test_injected_failure() {
  FakeBackendOptions fake;
  fake.fail_before_first = true;
  auto engine = Engine::create_fake({}, fake);

  SubmitRequest request;
  request.max_new_tokens = 2;
  const auto id = engine->submit(std::move(request));
  const auto events = collect_terminal(*engine, id);
  expect(events.size() == 1, "failure-before-first emitted extra events");
  expect(events.front().kind == EventKind::failed, "expected failed event");
  expect(!events.front().error.empty(), "failure event needs an error");
  expect(engine->stats().failed == 1, "failed count mismatch");
  expect(engine->stats().emitted_tokens == 0, "failure emitted tokens");
}

void test_failure_after_published_tokens() {
  FakeBackendOptions fake;
  fake.fail_after_tokens = 2;
  auto engine = Engine::create_fake({}, fake);

  SubmitRequest request;
  request.prompt_tokens = {20};
  request.max_new_tokens = 5;
  const auto id = engine->submit(std::move(request));
  const auto events = collect_terminal(*engine, id);

  std::vector<std::int32_t> tokens;
  for (const auto &event : events) {
    if (event.kind == EventKind::tokens) {
      tokens.insert(tokens.end(), event.tokens.begin(), event.tokens.end());
    }
  }
  expect(tokens == std::vector<std::int32_t>({21, 22}),
         "post-publication failure boundary emitted unexpected tokens");
  expect(events.back().kind == EventKind::failed, "expected terminal failure");
  expect(engine->stats().failed == 1, "post-publication failure not counted");
  expect(engine->stats().emitted_tokens == 2, "published token count mismatch");
}

void test_queued_cancellation() {
  FakeBackendOptions fake;
  fake.step_delay_us = 2000;
  auto engine = Engine::create_fake({}, fake);

  SubmitRequest first;
  first.max_new_tokens = 100;
  SubmitRequest second;
  second.max_new_tokens = 3;
  const auto first_id = engine->submit(std::move(first));
  const auto second_id = engine->submit(std::move(second));
  expect(engine->cancel(second_id), "queued request was not cancellable");
  expect(engine->cancel(first_id), "active request was not cancellable");

  bool first_terminal = false;
  bool second_terminal = false;
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while ((!first_terminal || !second_terminal) &&
         std::chrono::steady_clock::now() < deadline) {
    for (const auto &event : engine->poll_events(20ms)) {
      if (event.kind != EventKind::aborted) {
        continue;
      }
      first_terminal = first_terminal || event.request_id == first_id;
      second_terminal = second_terminal || event.request_id == second_id;
    }
  }
  expect(first_terminal, "first request did not abort");
  expect(second_terminal, "queued request did not abort");
  expect(engine->stats().aborted == 2, "queued cancellation count mismatch");
  const auto capacity = engine->capacity();
  expect(capacity.active_requests == 0, "active request leaked");
  expect(capacity.queued_requests == 0, "queued request leaked");
}

void test_shutdown_contract() {
  auto engine = Engine::create_fake();
  engine->shutdown();
  engine->shutdown();
  expect(!engine->capacity().accepting, "shutdown engine is still accepting");

  SubmitRequest request;
  bool threw = false;
  try {
    static_cast<void>(engine->submit(std::move(request)));
  } catch (const std::runtime_error &) {
    threw = true;
  }
  expect(threw, "submit after shutdown must fail");
}

void test_load_validation() {
  LoadOptions load;
  load.max_concurrency = 9;
  bool threw = false;
  try {
    static_cast<void>(Engine::create_fake(load));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  expect(threw, "P09 accepted concurrency beyond eight slots");
  load.max_concurrency = 2;
  auto two = Engine::create_fake(load);
  expect(two->capacity().max_concurrency == 2,
         "P09 rejected bounded C2 admission");
  two->shutdown();

  load = {};
  load.speculative.draft_tokens = 1;
  threw = false;
  try {
    static_cast<void>(Engine::create_fake(load));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  expect(threw, "disabled speculation accepted a nonzero draft window");

  load = {};
  load.speculative.backend = SpeculativeBackend::dflash2;
  load.speculative.draft_tokens = 0;
  static_cast<void>(Engine::create_fake(load));
  load.speculative.draft_tokens = 1;
  static_cast<void>(Engine::create_fake(load));
  load.speculative.draft_tokens = 7;
  static_cast<void>(Engine::create_fake(load));

  load.speculative.draft_tokens = 8;
  threw = false;
  try {
    static_cast<void>(Engine::create_fake(load));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  expect(threw, "DFlash2 accepted draft_tokens outside [0, 7]");

  load = {};
  load.speculative.backend = SpeculativeBackend::mtp;
  load.speculative.draft_tokens = 0;
  static_cast<void>(Engine::create_fake(load));
  load.speculative.draft_tokens = 4;
  static_cast<void>(Engine::create_fake(load));

  load.speculative.draft_tokens = 5;
  threw = false;
  try {
    static_cast<void>(Engine::create_fake(load));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  expect(threw, "MTP accepted draft_tokens outside [0, 4]");

  load = {};
  load.speculative.backend = SpeculativeBackend::dflash;
  load.speculative.draft_tokens = 1;
  threw = false;
  try {
    static_cast<void>(Engine::create_fake(load));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  expect(threw, "R1 accepted an unimplemented speculative backend");
}

void test_request_validation() {
  LoadOptions load;
  load.max_context = 4;
  auto engine = Engine::create_fake(load);

  SubmitRequest request;
  request.prompt_tokens = {1, 2, 3};
  request.max_new_tokens = 2;
  bool threw = false;
  try {
    static_cast<void>(engine->submit(request));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  expect(threw, "context budget overflow was accepted");

  request.prompt_tokens = {1};
  request.max_new_tokens = 1;
  request.sampling.top_p = 0.0F;
  threw = false;
  try {
    static_cast<void>(engine->submit(request));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  expect(threw, "invalid top_p was accepted");

  request.sampling.top_p = 1.0F;
  request.stop_token_seqs = {{}};
  threw = false;
  try {
    static_cast<void>(engine->submit(request));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  expect(threw, "empty stop sequence was accepted");
}

void test_mive20_fake_backend_flush_and_nonneg_tokens() {
  // 1) event_batch_size > 1 with fail_after_tokens must flush buffered tokens
  //    before returning EventKind::failed on both B=1 and B=2 paths.
  for (std::int32_t concurrency : {1, 2}) {
    LoadOptions load;
    load.max_concurrency = concurrency;
    FakeBackendOptions fake;
    fake.event_batch_size = 4;
    fake.fail_after_tokens = 2;
    auto engine = Engine::create_fake(load, fake);

    SubmitRequest request;
    request.prompt_tokens = {20};
    request.max_new_tokens = 6;
    const auto id = engine->submit(std::move(request));
    const auto events = collect_terminal(*engine, id);

    std::vector<std::int32_t> tokens;
    for (const auto &ev : events) {
      if (ev.kind == EventKind::tokens) {
        tokens.insert(tokens.end(), ev.tokens.begin(), ev.tokens.end());
      }
    }
    expect(tokens == std::vector<std::int32_t>({21, 22}),
           "MIVE-20 mid-batch failure lost buffered tokens (B=" +
               std::to_string(concurrency) + ")");
    expect(events.back().kind == EventKind::failed,
           "MIVE-20 expected failed terminal event");
    expect(engine->stats().emitted_tokens == 2 && engine->stats().failed == 1,
           "MIVE-20 mid-batch failure stats mismatch");
    engine->shutdown();
  }

  // 2) Negative prompt base token must wrap into [0, vocab_size) without
  //    emitting negative token IDs on both B=1 and B=2 paths.
  for (std::int32_t concurrency : {1, 2}) {
    LoadOptions load;
    load.max_concurrency = concurrency;
    FakeBackendOptions fake;
    fake.event_batch_size = 3;
    fake.vocab_size = 100;
    auto engine = Engine::create_fake(load, fake);

    SubmitRequest request;
    request.prompt_tokens = {-5};
    request.max_new_tokens = 6;
    const auto id = engine->submit(std::move(request));
    const auto events = collect_terminal(*engine, id);

    std::vector<std::int32_t> tokens;
    for (const auto &ev : events) {
      if (ev.kind == EventKind::tokens) {
        tokens.insert(tokens.end(), ev.tokens.begin(), ev.tokens.end());
      }
    }
    expect(tokens == std::vector<std::int32_t>({96, 97, 98, 99, 0, 1}),
           "MIVE-20 negative base produced negative or wrong token IDs (B=" +
               std::to_string(concurrency) + ")");
    expect(events.back().kind == EventKind::finished,
           "MIVE-20 expected finished terminal event");
    engine->shutdown();
  }

  // 3) Mid-batch cancellation with event_batch_size > 1 must flush already
  //    generated tokens before EventKind::aborted on both B=1 and B=2.
  for (std::int32_t concurrency : {1, 2}) {
    LoadOptions load;
    load.max_concurrency = concurrency;
    FakeBackendOptions fake;
    fake.event_batch_size = 32;
    fake.step_delay_us = 10'000;
    auto engine = Engine::create_fake(load, fake);

    SubmitRequest request;
    request.prompt_tokens = {10};
    request.max_new_tokens = 40;
    const auto id = engine->submit(std::move(request));
    std::this_thread::sleep_for(35ms);
    expect(engine->cancel(id), "MIVE-20 active request should be cancellable");
    const auto events = collect_terminal(*engine, id);

    std::vector<std::int32_t> tokens;
    for (const auto &ev : events) {
      if (ev.kind == EventKind::tokens) {
        tokens.insert(tokens.end(), ev.tokens.begin(), ev.tokens.end());
      }
    }
    expect(events.back().kind == EventKind::aborted,
           "MIVE-20 expected aborted terminal event");
    expect(!tokens.empty() && tokens.size() < 32,
           "MIVE-20 mid-batch cancellation did not flush partial batch (B=" +
               std::to_string(concurrency) + ", count=" +
               std::to_string(tokens.size()) + ")");
    engine->shutdown();
  }
}

} // namespace

int main() {
  try {
    test_normal_generation();
    test_active_cancellation();
    test_injected_failure();
    test_failure_after_published_tokens();
    test_queued_cancellation();
    test_shutdown_contract();
    test_load_validation();
    test_request_validation();
    test_mive20_fake_backend_flush_and_nonneg_tokens();
  } catch (const std::exception &error) {
    std::cerr << "native contract test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "native contract tests passed\n";
  return 0;
}
