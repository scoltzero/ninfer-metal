#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "omlx_m5/artifact.h"
#include "omlx_m5/engine.h"
#include "omlx_m5/kv_page_pool.h"
#include "omlx_m5/qwen_forward.h"
#include "../src/native_backend.h"
#include "../src/shared_prefix_cache.h"

using namespace omlx::m5;
using namespace std::chrono_literals;
namespace {
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
std::vector<std::uint32_t> prompt(std::size_t count, std::size_t salt = 256) {
  std::vector<std::uint32_t> tokens(count);
  for (std::size_t i = 0; i < count; ++i)
    tokens[i] = std::uint32_t((71 * i + salt) % 257);
  return tokens;
}
std::vector<std::int32_t> run(Engine &engine, const SubmitRequest &request) {
  const auto id = engine.submit(request);
  const auto deadline = std::chrono::steady_clock::now() + 30s;
  std::vector<std::int32_t> tokens;
  while (std::chrono::steady_clock::now() < deadline) {
    for (const auto &event : engine.poll_events(20ms)) {
      require(event.request_id == id, "P08 foreign request event");
      if (event.kind == EventKind::tokens)
        tokens.insert(tokens.end(), event.tokens.begin(), event.tokens.end());
      else {
        require(event.kind == EventKind::finished,
                "P08 request failed or cancelled unexpectedly");
        return tokens;
      }
    }
  }
  throw std::runtime_error("P08 request timeout");
}
LoadOptions options(const std::filesystem::path &fixtures,
                    const std::filesystem::path &libraries, bool cache) {
  LoadOptions value;
  const auto root = fixtures / "affine";
  value.model_dir = (root / "source").string();
  value.max_context = 1024;
  value.path_overrides["artifact"] = (root / "model.omlxm5").string();
  value.path_overrides["qwen_ar_metallib"] =
      (libraries / "omlx_m5_qwen_ar.metallib").string();
  if (cache) value.path_overrides["prefix_cache_bytes"] = "67108864";
  return value;
}
void page_restore(const MetalContext &context) {
  KvPagePool pool(context, 12, 4, 4);
  static_cast<void>(pool.begin_append(3));
  pool.commit(3);
  const auto lease = pool.retain_prefix();
  pool.reset();
  require(pool.tokens() == 0 && pool.free_pages() == 3,
          "P08 reset released retained page");
  pool.restore_prefix(lease);
  require(pool.tokens() == 3 && pool.physical_slot(0) == 0,
          "P08 restored mapping/frontier incorrect");
  const auto copies = pool.begin_append(2);
  require(copies.size() == 1 && copies[0].source == 0 &&
              copies[0].destination != copies[0].source,
          "P08 restored partial tail was not COW");
  pool.abort();
  require(pool.tokens() == 3 && pool.physical_slot(0) == 0,
          "P08 abort changed retained prefix");
  pool.reset();
  pool.release_prefix(lease);
  require(pool.free_pages() == 4, "P08 eviction leaked pages");
  bool rejected = false;
  try { pool.restore_prefix(lease); }
  catch (const std::out_of_range &) { rejected = true; }
  require(rejected, "P08 evicted lease remained restorable");
}
void qwen_restore(const std::filesystem::path &fixtures,
                  const std::filesystem::path &libraries) {
  const auto root = fixtures / "affine";
  const auto config = load_qwen_model_config(root / "source/config.json");
  const auto artifact = ArtifactModel::open(root / "model.omlxm5", true);
  auto context = MetalContext::create();
  page_restore(context);
  auto weights = materialize_model(
      artifact, plan_model_materialization(artifact, context), context);
  auto layers = build_qwen_layer_catalog(artifact, config, 1024, 64);
  const std::array<std::uint32_t, 2> taps{0, 1};
  QwenForward program(context, weights, layers, config, 1024, libraries, taps,
                      false, true);
  auto prefix = prompt(63);
  static_cast<void>(program.run(prefix));
  auto checkpoint = program.checkpoint();
  require(checkpoint.position == 63 && checkpoint.kv_lease &&
              checkpoint.accounted_bytes == program.checkpoint_bytes() &&
              !checkpoint.feature_history.empty(),
          "P08 checkpoint omitted target/GDN/feature state");
  const auto saved_state = program.inspect_recurrent_state();
  const std::array<std::uint32_t, 2> a{37, 43}, b{47, 53};
  const auto expected_a = program.run(a, {}, true);
  const auto state_a = program.inspect_recurrent_state();
  const auto features_a = program.feature_history().read(
      0, std::uint64_t(program.position()) * taps.size() * config.hidden_size * 2);
  program.reset();
  auto foreign = checkpoint;
  ++foreign.owner_id;
  bool wrong_owner = false;
  try { program.restore(foreign); }
  catch (const std::invalid_argument &) { wrong_owner = true; }
  require(wrong_owner, "P08 foreign checkpoint identity accepted");
  program.restore(checkpoint);
  require(program.position() == 63 &&
              program.inspect_recurrent_state() == saved_state,
          "P08 restore changed GDN/conv state");
  const auto restored_a = program.run(a, {}, true);
  require(restored_a.token == expected_a.token &&
              restored_a.logits == expected_a.logits &&
              program.inspect_recurrent_state() == state_a &&
              program.feature_history().read(0, features_a.size()) == features_a,
          "P08 restored partial-page continuation differs from cold run");
  program.reset();
  program.restore(checkpoint);
  const auto fork_b = program.run(b, {}, true);
  const auto state_b = program.inspect_recurrent_state();
  program.reset();
  static_cast<void>(program.run(prefix));
  const auto cold_b = program.run(b, {}, true);
  require(fork_b.token == cold_b.token && fork_b.logits == cold_b.logits &&
              state_b == program.inspect_recurrent_state(),
          "P08 forked suffix differs from cold run");
  program.reset();
  program.restore(checkpoint);
  static_cast<void>(program.verify(a));
  bool rejected = false;
  try { static_cast<void>(program.checkpoint()); }
  catch (const std::logic_error &) { rejected = true; }
  require(rejected, "P08 accepted pending verify checkpoint");
  program.abort_verify();
  program.release(checkpoint);
  rejected = false;
  try { program.restore(checkpoint); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "P08 released checkpoint remained valid");

  // Zero-move UMA continuation fastpath: checkpoint_uma / restore_uma /
  // restore_shared_uma / release_uma + slab recycling + cross-slot CoW parity.
  {
    program.reset();
    static_cast<void>(program.run(prefix));
    auto uma_ckpt = program.checkpoint_uma();
    require(uma_ckpt.position == 63 && uma_ckpt.kv_lease != 0 &&
                uma_ckpt.uma_epoch_id != 0 && uma_ckpt.uma_slab != nullptr &&
                uma_ckpt.kv_pages.size() == 1 &&
                program.non_kv_checkpoint_bytes() < program.checkpoint_bytes(),
            "P08 UMA checkpoint missing slab/epoch/KV lease");
    const MetalBuffer *first_slab_ptr = uma_ckpt.uma_slab.get();

    // Tampered epoch / owner / identity must be rejected without mutating state.
    program.reset();
    for (int mut = 0; mut < 4; ++mut) {
      auto bad = uma_ckpt;
      if (mut == 0) ++bad.uma_epoch_id;
      if (mut == 1) ++bad.owner_id;
      if (mut == 2) bad.execution_identity[0] ^= 1;
      if (mut == 3) ++bad.position;
      bool denied = false;
      try { program.restore_uma(bad); }
      catch (const std::invalid_argument &) { denied = true; }
      require(denied && program.position() == 0,
              "P08 tampered UMA continuation accepted or mutated state");
    }

    // Same-slot UMA restore must match cold run bit-for-bit.
    program.restore_uma(uma_ckpt);
    require(program.position() == 63 &&
                program.inspect_recurrent_state() == saved_state,
            "P08 UMA restore changed recurrent state");
    const auto uma_a = program.run(a, {}, true);
    require(uma_a.token == expected_a.token &&
                uma_a.logits == expected_a.logits &&
                program.inspect_recurrent_state() == state_a &&
                program.feature_history().read(0, features_a.size()) ==
                    features_a,
            "P08 UMA restore diverged from cold continuation");

    // Cross-slot UMA restore onto secondary slot sharing KvPagePool (zero KV dup).
    QwenForward peer(context, weights, layers, config, 1024, libraries, taps,
                     false, true, false, std::nullopt, false, true,
                     std::nullopt, LinearNumericPolicy::Legacy,
                     SamplingContract::LegacyRejectionV1,
                     AttentionNumericPolicy::Legacy, false, "bf16", &program);
    const auto free_before_peer = program.kv_page_pool()->free_pages();
    peer.restore_shared_uma(uma_ckpt, &program);
    require(program.kv_page_pool()->free_pages() == free_before_peer,
            "P08 UMA cross-slot restore allocated duplicate KV pages");
    require(peer.position() == 63 &&
                peer.inspect_recurrent_state() == saved_state,
            "P08 UMA cross-slot restore changed recurrent state");
    const auto peer_b = peer.run(b, {}, true);
    require(peer_b.token == cold_b.token && peer_b.logits == cold_b.logits &&
                peer.inspect_recurrent_state() == state_b,
            "P08 UMA cross-slot forked continuation diverged from cold run");
    peer.reset();

    // Release recycles slab into pool and invalidates epoch.
    program.reset();
    program.release_uma(uma_ckpt);
    require(uma_ckpt.uma_slab == nullptr && uma_ckpt.uma_epoch_id == 0,
            "P08 release_uma did not reset handle");
    bool stale_denied = false;
    try { program.restore_uma(uma_ckpt); }
    catch (const std::invalid_argument &) { stale_denied = true; }
    require(stale_denied, "P08 released UMA continuation remained restorable");

    static_cast<void>(program.run(prefix));
    auto uma_ckpt2 = program.checkpoint_uma();
    require(uma_ckpt2.uma_slab.get() == first_slab_ptr,
            "P08 UMA checkpoint pool did not recycle slab");
    program.reset();
    program.release_uma(uma_ckpt2);
    program.trim_uma_checkpoint_pool(0);
    require(program.kv_page_pool()->free_pages() ==
                program.kv_page_pool()->physical_pages(),
            "P08 UMA checkpoint/release leaked KV physical pages");
  }
}
void engine_restore(const std::filesystem::path &fixtures,
                    const std::filesystem::path &libraries,
                    const std::filesystem::path &draft_fixture) {
  auto cached = Engine::create(options(fixtures, libraries, true));
  auto cold = Engine::create(options(fixtures, libraries, false));
  SubmitRequest request;
  for (auto token : prompt(63)) request.prompt_tokens.push_back(token);
  request.max_new_tokens = 5;
  const auto expected = run(*cold, request);
  require(run(*cached, request) == expected, "P08 initial miss differs");
  require(run(*cached, request) == expected, "P08 exact hit differs");
  auto stats = cached->stats();
  require(stats.prefix_hits == 1 && stats.prefix_partial_hits == 0 &&
              stats.prefix_restored_tokens == 63 && stats.prefix_misses == 1,
          "P08 exact hit stats incomplete");
  request.prompt_tokens.push_back(37);
  request.prompt_tokens.push_back(43);
  const auto branch_a_prompt = request.prompt_tokens;
  const auto branch_a_expected = run(*cold, request);
  require(run(*cached, request) == branch_a_expected,
          "P08 partial hit differs");
  stats = cached->stats();
  require(stats.prefix_partial_hits == 1 &&
              stats.prefix_restored_tokens == 126 && stats.prefix_evictions == 0,
          "P08 partial hit/replace stats incomplete");
  request.prompt_tokens.back() = 53;
  require(run(*cached, request) == run(*cold, request),
          "P08 divergent prompt differs");
  require(cached->stats().prefix_partial_hits == 2 &&
              cached->stats().prefix_restored_tokens == 189 &&
              cached->stats().prefix_evictions == 0,
          "P08 multi-entry LRU evicted non-selected prefix unnecessarily");

  // Re-running branch A ([...37, 43]) must still be an exact hit (multi-entry retained)!
  auto branch_a_req = request;
  branch_a_req.prompt_tokens = branch_a_prompt;
  const auto before_branch_a = cached->stats();
  require(run(*cached, branch_a_req) == branch_a_expected &&
              cached->stats().prefix_hits == before_branch_a.prefix_hits + 1 &&
              cached->stats().prefix_partial_hits ==
                  before_branch_a.prefix_partial_hits &&
              cached->stats().prefix_evictions == 0,
          "P08 non-selected branch A checkpoint was wiped instead of retained");

  // Cross-seed / cross-temperature / cross-top_k exact hit & partial hit parity!
  for (const auto &samp : std::array<SamplingSpec, 4>{{
           {.temperature = 0.7F, .top_p = 0.9F, .top_k = 16, .seed = 17},
           {.temperature = 0.7F, .top_p = 0.9F, .top_k = 16, .seed = 99991},
           {.temperature = 1.2F, .top_p = 0.75F, .top_k = 32, .seed = 42},
           {.temperature = 0.0F, .top_p = 1.0F, .top_k = 0, .seed = 555},
       }}) {
    request.sampling = samp;
    const auto before = cached->stats();
    require(run(*cached, request) == run(*cold, request) &&
                cached->stats().prefix_hits == before.prefix_hits + 1 &&
                cached->stats().prefix_misses == before.prefix_misses,
            "P08 cross-seed/cross-temperature exact hit failed or missed");
  }
  // Cross-seed partial prefix hit extending cached prompt
  auto partial_cross = request;
  partial_cross.prompt_tokens.push_back(61);
  partial_cross.sampling = {
      .temperature = 0.85F, .top_p = 0.88F, .top_k = 24, .seed = 7777};
  {
    const auto before = cached->stats();
    require(run(*cached, partial_cross) == run(*cold, partial_cross) &&
                cached->stats().prefix_partial_hits ==
                    before.prefix_partial_hits + 1 &&
                cached->stats().prefix_misses == before.prefix_misses,
            "P08 cross-seed partial hit failed or missed");
  }

  // Multi-entry retention across 6 completely unrelated prefixes (> old cap of 4)
  std::vector<SubmitRequest> distinct_reqs;
  std::vector<std::vector<std::int32_t>> distinct_oracles;
  for (std::size_t idx = 0; idx < 6; ++idx) {
    SubmitRequest distinct;
    for (auto token : prompt(24 + idx * 3, 10 + idx * 29))
      distinct.prompt_tokens.push_back(token);
    distinct.max_new_tokens = 4;
    distinct.sampling = {.temperature = 0.6F,
                         .top_p = 0.92F,
                         .top_k = 16,
                         .seed = 1000 + idx};
    distinct_oracles.push_back(run(*cold, distinct));
    require(run(*cached, distinct) == distinct_oracles.back(),
            "P08 distinct initial miss differs");
    distinct_reqs.push_back(std::move(distinct));
  }
  {
    const auto before_multi = cached->stats();
    for (std::size_t idx = 0; idx < distinct_reqs.size(); ++idx) {
      // Flip seed on replay to test simultaneous multi-entry + cross-seed hits
      auto replay = distinct_reqs[idx];
      replay.sampling.seed = 5000 + idx;
      require(run(*cached, replay) == run(*cold, replay),
              "P08 multi-entry cross-seed replay differs from teacher");
    }
    const auto after_multi = cached->stats();
    require(after_multi.prefix_hits ==
                    before_multi.prefix_hits + distinct_reqs.size() &&
                after_multi.prefix_misses == before_multi.prefix_misses &&
                after_multi.prefix_evictions == before_multi.prefix_evictions,
            "P08 multi-entry cache wiped unrelated prefixes within budget");
  }

  // Explicit head-row regime sweep across t == 1, 2 <= t <= 8, and t > 8
  for (std::size_t regime_len : std::array<std::size_t, 5>{1, 5, 8, 9, 25}) {
    SubmitRequest regime_req;
    for (auto token : prompt(regime_len, 180 + regime_len * 13))
      regime_req.prompt_tokens.push_back(token);
    regime_req.max_new_tokens = 3;
    regime_req.sampling = {.temperature = 0.7F,
                           .top_p = 0.9F,
                           .top_k = 16,
                           .seed = 2000 + regime_len};
    require(run(*cached, regime_req) == run(*cold, regime_req),
            "P08 head-row regime initial miss differs");
    auto regime_replay = regime_req;
    regime_replay.sampling = {.temperature = 1.1F,
                              .top_p = 0.8F,
                              .top_k = 32,
                              .seed = 9000 + regime_len};
    const auto before_regime = cached->stats();
    require(run(*cached, regime_replay) == run(*cold, regime_replay) &&
                cached->stats().prefix_hits == before_regime.prefix_hits + 1 &&
                cached->stats().prefix_misses == before_regime.prefix_misses,
            "P08 cross-seed last_logits resample failed across t==1 / t<=8 / t>8");
  }

  request.max_new_tokens = 512;
  const auto cancelled_id = cached->submit(request);
  require(cached->cancel(cancelled_id), "P08 cancellation was not accepted");
  bool aborted = false;
  const auto deadline = std::chrono::steady_clock::now() + 30s;
  while (!aborted && std::chrono::steady_clock::now() < deadline)
    for (const auto &event : cached->poll_events(20ms))
      if (event.request_id == cancelled_id && event.kind == EventKind::aborted)
        aborted = true;
  require(aborted, "P08 cancellation did not terminate");
  request.max_new_tokens = 5;
  require(run(*cached, request) == run(*cold, request),
          "P08 request after cancellation differs");
  cached->shutdown();
  cold->shutdown();

  // Budget-driven LRU order test: budget fits 2 entries, 3rd evicts LRU (not MRU)
  {
    const auto root = fixtures / "affine";
    const auto cfg = load_qwen_model_config(root / "source/config.json");
    const auto art = ArtifactModel::open(root / "model.omlxm5", true);
    auto ctx = MetalContext::create();
    auto w = materialize_model(art, plan_model_materialization(art, ctx), ctx);
    auto l = build_qwen_layer_catalog(art, cfg, 1024, 64);
    QwenForward probe_prog(ctx, w, l, cfg, 1024, libraries);
    static_cast<void>(probe_prog.run(prompt(16, 3)));
    const auto one_entry_approx =
        probe_prog.checkpoint_bytes() + 16 * 4 + std::uint64_t(cfg.vocab_size) * 2 + 512;
    auto lru_opts = options(fixtures, libraries, true);
    lru_opts.path_overrides["prefix_cache_bytes"] =
        std::to_string(one_entry_approx * 2 + 256);
    auto lru_engine = Engine::create(lru_opts);
    auto lru_cold = Engine::create(options(fixtures, libraries, false));
    SubmitRequest req_a, req_b, req_c;
    for (auto t : prompt(16, 3)) req_a.prompt_tokens.push_back(t);
    for (auto t : prompt(16, 41)) req_b.prompt_tokens.push_back(t);
    for (auto t : prompt(16, 89)) req_c.prompt_tokens.push_back(t);
    req_a.max_new_tokens = req_b.max_new_tokens = req_c.max_new_tokens = 2;
    static_cast<void>(run(*lru_engine, req_a)); // [A]
    static_cast<void>(run(*lru_engine, req_b)); // [A, B]
    // Touch A so B becomes LRU -> [B, A]
    require(run(*lru_engine, req_a) == run(*lru_cold, req_a) &&
                lru_engine->stats().prefix_hits == 1 &&
                lru_engine->stats().prefix_evictions == 0,
            "P08 2-entry budget did not retain A and B");
    // Insert C -> evicts B (LRU), retains [A, C]
    require(run(*lru_engine, req_c) == run(*lru_cold, req_c) &&
                lru_engine->stats().prefix_evictions == 1,
            "P08 3rd entry did not trigger LRU eviction");
    const auto hits_before_a = lru_engine->stats().prefix_hits;
    require(run(*lru_engine, req_a) == run(*lru_cold, req_a) &&
                lru_engine->stats().prefix_hits == hits_before_a + 1,
            "P08 MRU entry A was evicted instead of LRU entry B");
    const auto misses_before_b = lru_engine->stats().prefix_misses;
    require(run(*lru_engine, req_b) == run(*lru_cold, req_b) &&
                lru_engine->stats().prefix_misses == misses_before_b + 1,
            "P08 evicted LRU entry B unexpectedly hit");
    lru_engine->shutdown();
    lru_cold->shutdown();
  }

  // Budget ceiling tests: 128 GiB accepted, >128 GiB and negative rejected
  {
    auto large_budget_opts = options(fixtures, libraries, true);
    large_budget_opts.path_overrides["prefix_cache_bytes"] =
        std::to_string(128ULL << 30);
    auto large_engine = Engine::create(large_budget_opts);
    large_engine->shutdown();

    large_budget_opts.path_overrides["prefix_cache_bytes"] =
        std::to_string((128ULL << 30) + 1ULL);
    bool rejected_over_128g = false;
    try { static_cast<void>(Engine::create(large_budget_opts)); }
    catch (const std::invalid_argument &) { rejected_over_128g = true; }
    require(rejected_over_128g, "P08 >128GiB prefix_cache_bytes was accepted");
  }

  auto tiny_options = options(fixtures, libraries, true);
  tiny_options.path_overrides["prefix_cache_bytes"] = "1";
  auto tiny = Engine::create(tiny_options);
  request = {};
  for (auto token : prompt(8)) request.prompt_tokens.push_back(token);
  request.max_new_tokens = 1;
  static_cast<void>(run(*tiny, request));
  static_cast<void>(run(*tiny, request));
  require(tiny->stats().prefix_hits == 0 && tiny->stats().prefix_misses == 2,
          "P08 insufficient budget retained an entry");
  tiny->shutdown();
  tiny_options.path_overrides["prefix_cache_bytes"] = "-1";
  bool rejected_budget = false;
  try { static_cast<void>(Engine::create(tiny_options)); }
  catch (const std::invalid_argument &) { rejected_budget = true; }
  require(rejected_budget, "P08 invalid budget was accepted");

  auto pressure_opts = options(fixtures, libraries, true);
  pressure_opts.path_overrides["kv_pool_tokens"] = "1088";
  auto pressure = Engine::create(pressure_opts);
  auto pressure_cold = Engine::create(options(fixtures, libraries, false));
  request = {};
  for (auto token : prompt(1021)) request.prompt_tokens.push_back(token);
  request.max_new_tokens = 1;
  static_cast<void>(run(*pressure, request));
  request.prompt_tokens.push_back(37);
  request.max_new_tokens = 2;
  require(run(*pressure, request) == run(*pressure_cold, request) &&
              pressure->stats().prefix_partial_hits == 1 &&
              pressure->stats().prefix_evictions >= 1,
          "P08 bounded page pool did not evict under decode pressure");
  pressure->shutdown();
  pressure_cold->shutdown();

  // Dim E B=1 multi-prefix residency test: 4 distinct pinned prefixes totaling
  // 26 KV pages (> old 17-page B=1 pool limit for max_context=1024) must all
  // stay resident simultaneously with zero evictions and 100% warm tail hits.
  {
    auto b1_eng = Engine::create(options(fixtures, libraries, true));
    auto b1_cold = Engine::create(options(fixtures, libraries, false));
    const std::array<std::size_t, 4> depths{128, 256, 512, 768};
    std::vector<std::vector<std::int32_t>> prefixes;
    for (std::size_t i = 0; i < depths.size(); ++i) {
      const auto p = prompt(depths[i], 300 + i * 73);
      std::vector<std::int32_t> pt(p.begin(), p.end());
      SubmitRequest seed;
      seed.prompt_tokens = pt;
      seed.max_new_tokens = 2;
      seed.pin_prefix = true;
      require(run(*b1_eng, seed) == run(*b1_cold, seed),
              "P08 Dim E B=1 pinned seed diverged");
      prefixes.push_back(std::move(pt));
    }
    const auto after_seed = b1_eng->stats();
    require(after_seed.prefix_pinned_entries == 4 &&
                after_seed.prefix_evictions == 0,
            "P08 Dim E B=1 failed to keep all 4 pinned prefixes (>17 pages) resident");

    for (std::size_t i = 0; i < prefixes.size(); ++i) {
      SubmitRequest warm;
      warm.prompt_tokens = prefixes[i];
      for (auto t : prompt(16, 900 + i * 19))
        warm.prompt_tokens.push_back(std::int32_t(t));
      warm.max_new_tokens = 4;
      warm.pin_prefix = false;
      const auto before = b1_eng->stats();
      require(run(*b1_eng, warm) == run(*b1_cold, warm),
              "P08 Dim E B=1 warm tail hit diverged from cold oracle");
      const auto after = b1_eng->stats();
      require(after.prefix_partial_hits == before.prefix_partial_hits + 1 &&
                  after.prefix_restored_tokens ==
                      before.prefix_restored_tokens + depths[i] &&
                  after.prefix_evictions == 0 &&
                  after.prefix_pinned_entries == 4,
              "P08 Dim E B=1 warm tail missed pinned prefix or evicted");
    }
    b1_eng->shutdown();
    b1_cold->shutdown();
  }

  // Dim E B=4 CoW test: 1 pinned prefix + 4 concurrent warm unique tails must
  // share physical KV pages via cross-slot UMA CoW, run packed batch, match
  // cold oracle, and skip redundant warm tail saves before token 1.
  {
    auto b4_opts = options(fixtures, libraries, true);
    b4_opts.max_concurrency = 4;
    b4_opts.path_overrides["batch_mode"] = "on";
    auto b4_eng = Engine::create(b4_opts);
    auto b4_cold = Engine::create(options(fixtures, libraries, false));

    const auto shared = prompt(512, 141);
    SubmitRequest seed;
    seed.prompt_tokens.assign(shared.begin(), shared.end());
    seed.max_new_tokens = 2;
    seed.pin_prefix = true;
    require(run(*b4_eng, seed) == run(*b4_cold, seed),
            "P08 Dim E B=4 CoW cold seed diverged");

    std::array<SubmitRequest, 4> reqs;
    std::array<std::vector<std::int32_t>, 4> oracles;
    std::array<std::uint64_t, 4> ids{};
    std::array<std::vector<std::int32_t>, 4> actuals;
    for (std::size_t s = 0; s < 4; ++s) {
      reqs[s].prompt_tokens.assign(shared.begin(), shared.end());
      for (auto t : prompt(16, 1200 + s * 31))
        reqs[s].prompt_tokens.push_back(std::int32_t(t));
      reqs[s].max_new_tokens = 6;
      reqs[s].pin_prefix = false;
      oracles[s] = run(*b4_cold, reqs[s]);
    }
    for (std::size_t s = 0; s < 4; ++s)
      ids[s] = b4_eng->submit(reqs[s]);
    std::size_t finished = 0;
    const auto b4_deadline = std::chrono::steady_clock::now() + 30s;
    while (finished < 4 && std::chrono::steady_clock::now() < b4_deadline) {
      for (const auto &ev : b4_eng->poll_events(20ms)) {
        for (std::size_t s = 0; s < 4; ++s) {
          if (ev.request_id != ids[s])
            continue;
          if (ev.kind == EventKind::tokens) {
            actuals[s].insert(actuals[s].end(), ev.tokens.begin(),
                              ev.tokens.end());
          } else {
            require(ev.kind == EventKind::finished,
                    "P08 Dim E B=4 CoW request failed");
            ++finished;
          }
        }
      }
    }
    require(finished == 4, "P08 Dim E B=4 CoW timed out");
    for (std::size_t s = 0; s < 4; ++s)
      require(actuals[s] == oracles[s],
              "P08 Dim E B=4 CoW output diverged from cold oracle");
    const auto b4_st = b4_eng->stats();
    require(b4_st.prefix_partial_hits == 4 &&
                b4_st.prefix_shared_hits >= 3 &&
                b4_st.prefix_evictions == 0 &&
                b4_st.prefix_pinned_entries == 1 &&
                b4_st.packed_batch_calls >= 1,
            "P08 Dim E B=4 CoW stats mismatch");

    // Re-running reqs[0] must hit the pinned 512-tok prefix (partial hit), NOT
    // an exact 528-tok hit, proving warm unique tails skipped redundant save_prefix().
    const auto before_re = b4_eng->stats();
    require(run(*b4_eng, reqs[0]) == oracles[0],
            "P08 Dim E B=4 CoW replay diverged");
    const auto after_re = b4_eng->stats();
    require(after_re.prefix_partial_hits == before_re.prefix_partial_hits + 1 &&
                after_re.prefix_restored_tokens ==
                    before_re.prefix_restored_tokens + 512,
            "P08 Dim E B=4 CoW unexpectedly saved unpinned warm unique tail");
    b4_eng->shutdown();
    b4_cold->shutdown();
  }

  auto draft_options = options(fixtures, libraries, true);
  draft_options.draft_dir = (draft_fixture / "source").string();
  draft_options.path_overrides["draft_artifact"] =
      (draft_fixture / "model.omlxm5").string();
  draft_options.path_overrides["dflash2_metallib"] =
      (libraries / "omlx_m5_dflash2_program.metallib").string();
  draft_options.speculative = {.backend = SpeculativeBackend::dflash2,
                               .draft_tokens = 7};
  auto dflash = Engine::create(draft_options);
  draft_options.path_overrides.erase("prefix_cache_bytes");
  auto dflash_cold = Engine::create(draft_options);
  request = {};
  for (auto token : prompt(33)) request.prompt_tokens.push_back(token);
  request.max_new_tokens = 9;
  const auto expected_draft = run(*dflash_cold, request);
  require(run(*dflash, request) == expected_draft &&
              run(*dflash, request) == expected_draft &&
              dflash->stats().prefix_hits == 1 &&
              dflash->stats().spec_rounds > 0,
          "P08 DFlash cached target features/anchor differ");
  // Cross-seed / cross-temperature hit with DFlash speculative decoding
  request.sampling = {.temperature = 0.7F, .top_p = 0.9F, .top_k = 16, .seed = 31337};
  require(run(*dflash, request) == run(*dflash_cold, request) &&
              dflash->stats().prefix_hits == 2 &&
              dflash->stats().prefix_misses == 1,
          "P08 DFlash cross-seed/temperature exact hit differs");
  request.prompt_tokens.push_back(37);
  request.sampling.seed = 424242;
  require(run(*dflash, request) == run(*dflash_cold, request) &&
              dflash->stats().prefix_partial_hits == 1,
          "P08 DFlash partial target feature continuation differs");
  dflash->shutdown();
  dflash_cold->shutdown();
}

void multiturn_cache_and_slab_invariants(
    const std::filesystem::path &fixtures,
    const std::filesystem::path &libraries) {
  struct DummySamplingKey {
    bool operator==(const DummySamplingKey &) const = default;
  };
  const auto root = fixtures / "affine";
  const auto config = load_qwen_model_config(root / "source/config.json");
  const auto artifact = ArtifactModel::open(root / "model.omlxm5", true);
  auto context = MetalContext::create();
  auto weights = materialize_model(
      artifact, plan_model_materialization(artifact, context), context);
  constexpr std::uint32_t kMaxCtx = 2048;
  auto layers = build_qwen_layer_catalog(artifact, config, kMaxCtx, 64);
  const std::array<std::uint32_t, 2> taps{0, 1};
  const std::string exec_id = "p08.multiturn.rolling.uma.test";

  // 4-slot shared KV + UMA checkpoint pool with feature_history taps enabled
  // so growing DFlash2 feature_history slab quanta + orphan-free resize are
  // directly verified via MetalContext::current_allocated_bytes().
  std::vector<std::unique_ptr<QwenForward>> slots;
  slots.reserve(4);
  slots.push_back(std::make_unique<QwenForward>(
      context, weights, layers, config, kMaxCtx, libraries, taps, false, true,
      false, std::nullopt, false, true, std::nullopt,
      LinearNumericPolicy::RowExactPortableV1,
      SamplingContract::PositionCoupledV1, AttentionNumericPolicy::RowExactV1,
      false, "bf16", nullptr, false, 256U));
  slots[0]->bind_execution_identity(exec_id);
  for (std::size_t s = 1; s < 4; ++s) {
    slots.push_back(std::make_unique<QwenForward>(
        context, weights, layers, config, kMaxCtx, libraries, taps, false, true,
        false, std::nullopt, false, true, std::nullopt,
        LinearNumericPolicy::RowExactPortableV1,
        SamplingContract::PositionCoupledV1, AttentionNumericPolicy::RowExactV1,
        false, "bf16", slots[0].get()));
    slots[s]->bind_execution_identity(exec_id);
  }
  auto program_at = [&](std::size_t s) -> QwenForward & { return *slots.at(s); };
  const GpuSamplingOptions samp{
      .temperature = 0.0F,
      .top_k = 1,
      .top_p = 1.0F,
      .seed = 1,
      .contract = SamplingContract::PositionCoupledV1};

  // (1) B=1 10-turn growing history with 1 pinned System Prompt:
  //     size() == 2 (1 pinned + 1 unpinned), 0 evictions, 0 orphan slabs!
  {
    detail::SharedPrefixCache<DummySamplingKey> cache(64ULL << 20);
    cache.set_max_unpinned_entries(8);
    const auto sys_toks = prompt(256, 701);
    slots[0]->reset();
    static_cast<void>(slots[0]->run(sys_toks, samp));
    bool inserted = false;
    const auto ev0 = cache.insert_checkpoint(
        0, exec_id, sys_toks, DummySamplingKey{}, 0U, true, true, program_at,
        &inserted);
    require(inserted && ev0 == 0 && cache.size() == 1 &&
                cache.pinned_count() == 1 && cache.unpinned_count() == 0,
            "P08 multiturn white-box system pin failed");

    std::vector<std::uint32_t> history(sys_toks.begin(), sys_toks.end());
    for (std::size_t turn = 1; turn <= 10; ++turn) {
      const auto delta = prompt(92, 1000 + turn * 53);
      std::vector<std::uint32_t> next_hist = history;
      next_hist.insert(next_hist.end(), delta.begin(), delta.end());

      const auto match = cache.find_best(
          exec_id, next_hist, SamplingContract::LegacyRejectionV1);
      require(match.has_value() && match->matched_tokens == history.size(),
              "P08 multiturn white-box B=1 turn did not match previous frontier");
      bool cross_slot = false;
      auto &hit =
          cache.restore_into_slot(match->entry_index, 0, program_at, cross_slot);
      const std::uint64_t matched_id = hit.id;
      require(!cross_slot && slots[0]->position() == history.size(),
              "P08 multiturn white-box B=1 restore mismatch");

      static_cast<void>(slots[0]->run(delta, samp));
      require(slots[0]->position() == next_hist.size(),
              "P08 multiturn white-box B=1 advance mismatch");

      bool turn_inserted = false;
      const auto ev = cache.insert_checkpoint(
          0, exec_id, next_hist, DummySamplingKey{}, 0U, false, false,
          program_at, &turn_inserted, matched_id);
      require(turn_inserted && ev == 0 && cache.size() == 2 &&
                  cache.pinned_count() == 1 && cache.unpinned_count() == 1,
              "P08 multiturn white-box B=1 failed 1x in-place supersede");

      // Zero orphan slabs in UmaCheckpointPool: trimming free_slabs to 0 must
      // release 0 bytes of Metal allocations after every growing turn!
      const auto bytes_before_trim = context.current_allocated_bytes();
      slots[0]->trim_uma_checkpoint_pool(0);
      const auto bytes_after_trim = context.current_allocated_bytes();
      require(bytes_after_trim == bytes_before_trim,
              "P08 multiturn UmaCheckpointPool left undersized orphan slab");

      history = std::move(next_hist);
    }
    require(history.size() > 1024,
            "P08 multiturn white-box B=1 did not cross 1024 tokens");
    slots[0]->reset();
    static_cast<void>(cache.clear_all(program_at));
    require(slots[0]->kv_page_pool()->free_pages() ==
                slots[0]->kv_page_pool()->physical_pages(),
            "P08 multiturn white-box B=1 leaked KV pages");
  }

  // (2) B=4 4-session x 4-turn sharing 1 pinned System Prompt:
  //     Wave 1 has 3 cross-slot CoW hits; Waves 2..4 have 4 rolling hits/wave;
  //     size() == 5 (1 pinned + 4 rolling frontiers), 0 evictions, 0 orphans!
  {
    detail::SharedPrefixCache<DummySamplingKey> cache(64ULL << 20);
    cache.set_max_unpinned_entries(8);
    const auto sys_toks = prompt(512, 811);
    slots[0]->reset();
    static_cast<void>(slots[0]->run(sys_toks, samp));
    bool inserted = false;
    const auto ev0 = cache.insert_checkpoint(
        0, exec_id, sys_toks, DummySamplingKey{}, 0U, true, true, program_at,
        &inserted);
    require(inserted && ev0 == 0 && cache.size() == 1,
            "P08 multiturn white-box B=4 system pin failed");

    std::array<std::vector<std::uint32_t>, 4> histories;
    for (std::size_t s = 0; s < 4; ++s)
      histories[s].assign(sys_toks.begin(), sys_toks.end());

    for (std::size_t wave = 1; wave <= 4; ++wave) {
      std::size_t cross_hits = 0;
      std::array<std::uint64_t, 4> matched_ids{};
      for (std::size_t s = 0; s < 4; ++s) {
        const auto delta = prompt(160, 2000 + wave * 211 + s * 47);
        histories[s].insert(histories[s].end(), delta.begin(), delta.end());
        const auto match = cache.find_best(
            exec_id, histories[s], SamplingContract::LegacyRejectionV1);
        require(match.has_value() &&
                    match->matched_tokens == histories[s].size() - delta.size(),
                "P08 multiturn white-box B=4 wave match mismatch");
        bool cross_slot = false;
        auto &hit = cache.restore_into_slot(match->entry_index, s, program_at,
                                            cross_slot);
        if (cross_slot)
          ++cross_hits;
        matched_ids[s] = hit.id;
        static_cast<void>(slots[s]->run(delta, samp));
      }
      if (wave == 1) {
        require(cross_hits == 3,
                "P08 multiturn white-box B=4 Wave 1 expected 3 CoW hits");
      } else {
        require(cross_hits == 0,
                "P08 multiturn white-box B=4 Wave 2..4 expected same-slot hits");
      }
      for (std::size_t s = 0; s < 4; ++s) {
        bool ok = false;
        const auto ev = cache.insert_checkpoint(
            s, exec_id, histories[s], DummySamplingKey{}, 0U, false, false,
            program_at, &ok, matched_ids[s]);
        require(ok && ev == 0,
                "P08 multiturn white-box B=4 supersede insert failed");
      }
      require(cache.size() == 5 && cache.pinned_count() == 1 &&
                  cache.unpinned_count() == 4,
              "P08 multiturn white-box B=4 cache size != 5 (1 pinned + 4 rolling)");
      const auto bytes_before_trim = context.current_allocated_bytes();
      slots[0]->trim_uma_checkpoint_pool(0);
      require(context.current_allocated_bytes() == bytes_before_trim,
              "P08 multiturn white-box B=4 left orphan slabs");
    }
    require(histories[0].size() > 1024,
            "P08 multiturn white-box B=4 did not cross 1024 tokens");
    for (std::size_t s = 0; s < 4; ++s)
      slots[s]->reset();
    static_cast<void>(cache.clear_all(program_at));
    require(slots[0]->kv_page_pool()->free_pages() ==
                slots[0]->kv_page_pool()->physical_pages(),
            "P08 multiturn white-box B=4 leaked KV pages");
  }
}

void multiturn_engine_tests(const std::filesystem::path &fixtures,
                            const std::filesystem::path &libraries,
                            const std::filesystem::path &draft_fixture) {
  auto make_opts = [&](bool cache, bool rolling, bool dflash_mode,
                       std::int32_t max_conc) {
    LoadOptions o;
    const auto root = fixtures / "affine";
    o.model_dir = (root / "source").string();
    o.max_context = 2048;
    o.max_concurrency = max_conc;
    o.sampling_contract = "position_coupled_v1";
    o.path_overrides["artifact"] = (root / "model.omlxm5").string();
    o.path_overrides["kernel_directory"] = libraries.string();
    if (cache)
      o.path_overrides["prefix_cache_bytes"] = std::to_string(64ULL << 20);
    if (rolling)
      o.path_overrides["rolling_prefix_cache"] = "on";
    if (max_conc > 1)
      o.path_overrides["batch_mode"] = "on";
    if (dflash_mode) {
      o.draft_dir = (draft_fixture / "source").string();
      o.path_overrides["draft_artifact"] =
          (draft_fixture / "model.omlxm5").string();
      o.path_overrides["dflash2_metallib"] =
          (libraries / "omlx_m5_dflash2_program.metallib").string();
      o.speculative = {.backend = SpeculativeBackend::dflash2,
                       .draft_tokens = 5};
    }
    return o;
  };

  // Verify rolling_prefix_cache execution identity isolation + env var fallback.
  {
    auto non_rolling = Engine::create(make_opts(true, false, false, 1));
    auto explicit_rolling = Engine::create(make_opts(true, true, false, 1));
    require(non_rolling->stats().execution_identity_id !=
                explicit_rolling->stats().execution_identity_id,
            "P08 rolling_prefix_cache did not isolate execution_identity_id");

    const char *prev_env = std::getenv("OMLX_M5_ROLLING_PREFIX_CACHE");
    const std::optional<std::string> saved_env =
        prev_env ? std::optional<std::string>(prev_env) : std::nullopt;
    setenv("OMLX_M5_ROLLING_PREFIX_CACHE", "1", 1);
    auto env_rolling = Engine::create(make_opts(true, false, false, 1));
    auto explicit_off_opts = make_opts(true, false, false, 1);
    explicit_off_opts.path_overrides["rolling_prefix_cache"] = "off";
    auto explicit_off = Engine::create(explicit_off_opts);
    if (saved_env.has_value())
      setenv("OMLX_M5_ROLLING_PREFIX_CACHE", saved_env->c_str(), 1);
    else
      unsetenv("OMLX_M5_ROLLING_PREFIX_CACHE");

    require(env_rolling->stats().execution_identity_id ==
                explicit_rolling->stats().execution_identity_id,
            "P08 OMLX_M5_ROLLING_PREFIX_CACHE=1 env fallback did not activate");
    require(explicit_off->stats().execution_identity_id ==
                non_rolling->stats().execution_identity_id,
            "P08 explicit rolling_prefix_cache=off did not override env var");
    non_rolling->shutdown();
    explicit_rolling->shutdown();
    env_rolling->shutdown();
    explicit_off->shutdown();
  }

  // (a) 10-turn B=1 short chat (PositionCoupledV1 + rolling_prefix_cache="on",
  //     crossing < 1024 and > 1024, BOTH AR and DFlash2).
  for (bool dflash_mode : {false, true}) {
    auto rolling = Engine::create(make_opts(true, true, dflash_mode, 1));
    auto cold = Engine::create(make_opts(false, false, dflash_mode, 1));

    const auto sys = prompt(256, dflash_mode ? 521 : 419);
    {
      SubmitRequest seed;
      seed.prompt_tokens.assign(sys.begin(), sys.end());
      seed.max_new_tokens = 2;
      seed.pin_prefix = true;
      require(run(*rolling, seed) == run(*cold, seed),
              "P08 multiturn B=1 pinned system seed diverged");
      const auto st_seed = rolling->stats();
      require(st_seed.prefix_pinned_entries == 1 && st_seed.prefix_evictions == 0,
              "P08 multiturn B=1 system seed did not pin cleanly");
    }

    std::vector<std::int32_t> history(sys.begin(), sys.end());
    std::vector<std::int32_t> turn1_probe;
    std::size_t turn2_10_partial_hits = 0;
    std::size_t expected_prev_committed = sys.size();

    for (std::size_t turn = 1; turn <= 10; ++turn) {
      const auto user_q = prompt(64, 1500 + turn * 37 + (dflash_mode ? 7 : 0));
      history.insert(history.end(), user_q.begin(), user_q.end());

      SubmitRequest req;
      req.prompt_tokens = history;
      req.max_new_tokens = 32;
      req.sampling = {.temperature = 0.0F, .top_p = 1.0F, .top_k = 1, .seed = 7};
      req.pin_prefix = false;

      const auto before = rolling->stats();
      const auto actual = run(*rolling, req);
      const auto expected = run(*cold, req);
      require(actual == expected && actual.size() == 32,
              "P08 multiturn B=1 greedy output diverged from cold reference");

      const auto after = rolling->stats();
      require(after.prefix_partial_hits == before.prefix_partial_hits + 1,
              "P08 multiturn B=1 turn missed rolling partial hit");
      require(after.prefix_restored_tokens ==
                  before.prefix_restored_tokens + expected_prev_committed,
              "P08 multiturn B=1 restored token count did not equal "
              "Prompt_{k-1} + committed(Reply_{k-1})");
      require(after.prefix_evictions == 0 && after.prefix_pinned_entries == 1,
              "P08 multiturn B=1 triggered unexpected eviction or unpin");
      if (turn >= 2)
        ++turn2_10_partial_hits;

      // Append actual generated tokens to causal history for next turn.
      history.insert(history.end(), actual.begin(), actual.end());
      // Exact committed prefix: prompt + first (max_new_tokens - 1) tokens
      // (final emitted token was sampled, not fed into KV/GDN).
      expected_prev_committed = history.size() - 1;
      if (turn == 1) {
        turn1_probe.assign(history.begin(), history.end() - 1);
        turn1_probe.push_back(251);
        turn1_probe.push_back(252);
      }
    }
    require(turn2_10_partial_hits == 9 && history.size() > 1024,
            "P08 multiturn B=1 did not achieve 9 rolling hits crossing 1024");
    if (dflash_mode) {
      require(rolling->stats().spec_rounds > 0,
              "P08 multiturn B=1 DFlash2 did not execute speculative rounds");
    }

    // Prove Turn 1's unpinned entry was superseded (only pinned System + Turn 10
    // remain): probing Turn 1's old committed prefix + divergent suffix must fall
    // back to System(256), NOT Turn 1 (~351)!
    {
      SubmitRequest probe_req;
      probe_req.prompt_tokens = turn1_probe;
      probe_req.max_new_tokens = 2;
      probe_req.sampling = {
          .temperature = 0.0F, .top_p = 1.0F, .top_k = 1, .seed = 7};
      const auto before = rolling->stats();
      require(run(*rolling, probe_req) == run(*cold, probe_req),
              "P08 multiturn B=1 supersede probe diverged");
      const auto after = rolling->stats();
      require(after.prefix_partial_hits == before.prefix_partial_hits + 1 &&
                  after.prefix_restored_tokens ==
                      before.prefix_restored_tokens + sys.size(),
              "P08 multiturn B=1 stale Turn 1 entry was not superseded");
    }

    // Stop-sequence Pitfall #1 check (async AR drain_in_flight_ar / DFlash2 stop):
    {
      const auto base_u = prompt(40, dflash_mode ? 3101 : 3001);
      std::vector<std::int32_t> p1(sys.begin(), sys.end());
      p1.insert(p1.end(), base_u.begin(), base_u.end());

      SubmitRequest probe_full;
      probe_full.prompt_tokens = p1;
      probe_full.max_new_tokens = 12;
      probe_full.sampling = {
          .temperature = 0.0F, .top_p = 1.0F, .top_k = 1, .seed = 11};
      const auto full_reply = run(*cold, probe_full);
      require(full_reply.size() == 12, "P08 stop probe full reply too short");
      const std::int32_t stop_tok = full_reply[5];

      SubmitRequest t1 = probe_full;
      t1.stop_token_seqs = {{stop_tok}};
      const auto t1_out = run(*rolling, t1);
      const auto t1_cold = run(*cold, t1);
      require(t1_out == t1_cold, "P08 multiturn stop-sequence Turn 1 diverged");

      // Turn 2 appends t1_out + stop_tok (like ChatML <|im_end|>) + user_2.
      std::vector<std::int32_t> p2 = p1;
      p2.insert(p2.end(), t1_out.begin(), t1_out.end());
      p2.push_back(stop_tok);
      const auto u2 = prompt(24, 3201);
      p2.insert(p2.end(), u2.begin(), u2.end());

      SubmitRequest t2;
      t2.prompt_tokens = p2;
      t2.max_new_tokens = 8;
      t2.sampling = {.temperature = 0.0F, .top_p = 1.0F, .top_k = 1, .seed = 11};
      const auto before_t2 = rolling->stats();
      const auto t2_actual = run(*rolling, t2);
      const auto t2_expected = run(*cold, t2);
      const auto after_t2 = rolling->stats();
      const std::size_t restored =
          after_t2.prefix_restored_tokens - before_t2.prefix_restored_tokens;
      require(t2_actual == t2_expected,
              "P08 multiturn stop-sequence Turn 2 diverged from cold oracle");
      require(after_t2.prefix_partial_hits == before_t2.prefix_partial_hits + 1 &&
                  (restored == p1.size() + t1_out.size() ||
                   restored == p1.size() + t1_out.size() + 1),
              "P08 multiturn stop-sequence Turn 2 failed exact rolling hit");
    }

    rolling->shutdown();
    cold->shutdown();
  }

  // (a2) Thinking-mode multi-turn chat where prior assistant <think>...</think>
  //      reasoning tokens are stripped by ChatML on subsequent user turns,
  //      plus same-turn tool-call continuation keeping <think> (AR & DFlash2).
  for (bool dflash_mode : {false, true}) {
    auto think_opts = make_opts(true, true, dflash_mode, 1);
    think_opts.path_overrides["think_open_token_id"] = "248";
    think_opts.path_overrides["think_close_token_id"] = "249";
    auto rolling = Engine::create(think_opts);
    auto cold = Engine::create(make_opts(false, false, dflash_mode, 1));

    const auto sys = prompt(128, dflash_mode ? 613 : 607);
    {
      SubmitRequest seed;
      seed.prompt_tokens.assign(sys.begin(), sys.end());
      seed.max_new_tokens = 2;
      seed.pin_prefix = true;
      require(run(*rolling, seed) == run(*cold, seed),
              "P08 thinking multiturn system seed diverged");
    }

    std::vector<std::int32_t> stripped_history(sys.begin(), sys.end());
    std::vector<std::int32_t> turn1_think_probe;
    std::size_t expected_restored = sys.size();

    for (std::size_t turn = 1; turn <= 4; ++turn) {
      auto u_q = prompt(48, 2100 + turn * 43);
      for (auto &t : u_q)
        if (t == 248 || t == 249)
          t = 240;
      stripped_history.insert(stripped_history.end(), u_q.begin(), u_q.end());
      const std::size_t pre_think_len = stripped_history.size();
      // Open <think>\n header at end of prompt (e.g. <|im_start|>assistant\n<think>\n)
      stripped_history.push_back(248);
      stripped_history.push_back(10);

      SubmitRequest req;
      req.prompt_tokens = stripped_history;
      req.max_new_tokens = 24;
      req.sampling = {.temperature = 0.0F, .top_p = 1.0F, .top_k = 1, .seed = 19};
      req.pin_prefix = false;

      const auto before = rolling->stats();
      const auto actual = run(*rolling, req);
      const auto expected = run(*cold, req);
      require(actual == expected && actual.size() == 24,
              "P08 thinking multiturn greedy output diverged");
      const auto after = rolling->stats();
      require(after.prefix_partial_hits == before.prefix_partial_hits + 1,
              "P08 thinking multiturn missed partial hit");
      require(after.prefix_restored_tokens ==
                  before.prefix_restored_tokens + expected_restored,
              "P08 thinking multiturn restored token count mismatch");
      require(after.prefix_evictions == 0,
              "P08 thinking multiturn triggered unexpected eviction");

      if (turn == 1) {
        turn1_think_probe.assign(stripped_history.begin(),
                                 stripped_history.begin() +
                                     std::ptrdiff_t(pre_think_len));
        turn1_think_probe.push_back(245);
        turn1_think_probe.push_back(246);
      }

      // Simulate exact Qwen3.8 ChatML BPE merge when <think>...</think> is
      // stripped: trailing [248, 10] (`<think>\n`) becomes [248, 250, 249, 250]
      // (`<think>\n\n</think>\n\n`, where `\n\n` merges into token 250 != 10)!
      stripped_history.resize(pre_think_len);
      expected_restored = pre_think_len;
      stripped_history.push_back(248);
      stripped_history.push_back(250);
      stripped_history.push_back(249);
      stripped_history.push_back(250);
      stripped_history.insert(stripped_history.end(), actual.begin() + 16,
                              actual.end());
    }

    // Prove Turn 1's unpinned <think> prompt anchor was superseded by Turn 4:
    // probing Turn 1's prompt + divergent suffix must fall back to System(128).
    {
      SubmitRequest probe_req;
      probe_req.prompt_tokens = turn1_think_probe;
      probe_req.max_new_tokens = 2;
      probe_req.sampling = {
          .temperature = 0.0F, .top_p = 1.0F, .top_k = 1, .seed = 19};
      const auto before = rolling->stats();
      require(run(*rolling, probe_req) == run(*cold, probe_req),
              "P08 thinking multiturn supersede probe diverged");
      const auto after = rolling->stats();
      require(after.prefix_restored_tokens ==
                  before.prefix_restored_tokens + sys.size(),
              "P08 thinking multiturn stale Turn 1 anchor was not superseded");
    }

    rolling->shutdown();
    cold->shutdown();
  }

  // (b) B=4 concurrent 4-session x 4-turn multi-turn chat sharing 1 pinned
  //     System Prompt (crossing 1024 on Wave 4 so packed prefill boundary fix
  //     is exercised end-to-end for BOTH AR and DFlash2!).
  for (bool b4_dflash : {false, true}) {
    auto b4 = Engine::create(make_opts(true, true, b4_dflash, 4));
    auto b4_cold = Engine::create(make_opts(false, false, b4_dflash, 1));

    const auto shared = prompt(512, b4_dflash ? 431 : 411);
    {
      SubmitRequest seed;
      seed.prompt_tokens.assign(shared.begin(), shared.end());
      seed.max_new_tokens = 2;
      seed.pin_prefix = true;
      require(run(*b4, seed) == run(*b4_cold, seed),
              "P08 multiturn B=4 system seed diverged");
    }

    std::array<std::vector<std::int32_t>, 4> histories;
    std::array<std::size_t, 4> expected_committed{};
    std::vector<std::int32_t> s0_wave1_probe;
    for (std::size_t s = 0; s < 4; ++s) {
      histories[s].assign(shared.begin(), shared.end());
      expected_committed[s] = shared.size();
    }

    for (std::size_t wave = 1; wave <= 4; ++wave) {
      const std::size_t u_len = (wave == 4) ? 160 : 80;
      const std::int32_t g_len = 48;
      std::array<SubmitRequest, 4> reqs;
      std::array<std::vector<std::int32_t>, 4> oracles;
      std::array<std::uint64_t, 4> ids{};
      std::array<std::vector<std::int32_t>, 4> actuals;
      std::size_t expected_wave_restored = 0;

      for (std::size_t s = 0; s < 4; ++s) {
        const auto u = prompt(u_len, 4000 + wave * 313 + s * 61);
        histories[s].insert(histories[s].end(), u.begin(), u.end());
        reqs[s].prompt_tokens = histories[s];
        reqs[s].max_new_tokens = g_len;
        reqs[s].sampling = {
            .temperature = 0.0F, .top_p = 1.0F, .top_k = 1, .seed = 100 + s};
        reqs[s].pin_prefix = false;
        oracles[s] = run(*b4_cold, reqs[s]);
        expected_wave_restored += expected_committed[s];
      }

      const auto before = b4->stats();
      for (std::size_t s = 0; s < 4; ++s)
        ids[s] = b4->submit(reqs[s]);
      std::size_t finished = 0;
      const auto deadline = std::chrono::steady_clock::now() + 60s;
      while (finished < 4 && std::chrono::steady_clock::now() < deadline) {
        for (const auto &ev : b4->poll_events(20ms)) {
          for (std::size_t s = 0; s < 4; ++s) {
            if (ev.request_id != ids[s])
              continue;
            if (ev.kind == EventKind::tokens) {
              actuals[s].insert(actuals[s].end(), ev.tokens.begin(),
                                ev.tokens.end());
            } else {
              require(ev.kind == EventKind::finished,
                      "P08 multiturn B=4 wave request failed");
              ++finished;
            }
          }
        }
      }
      require(finished == 4, "P08 multiturn B=4 wave timed out");
      for (std::size_t s = 0; s < 4; ++s) {
        require(actuals[s] == oracles[s] && actuals[s].size() == std::size_t(g_len),
                "P08 multiturn B=4 wave output diverged from cold oracle");
        histories[s].insert(histories[s].end(), actuals[s].begin(),
                            actuals[s].end());
        expected_committed[s] = histories[s].size() - 1;
      }
      if (wave == 1) {
        s0_wave1_probe.assign(histories[0].begin(), histories[0].end() - 1);
        s0_wave1_probe.push_back(241);
        s0_wave1_probe.push_back(242);
      }

      const auto after = b4->stats();
      require(after.prefix_partial_hits == before.prefix_partial_hits + 4,
              "P08 multiturn B=4 wave did not get 4 partial hits");
      require(after.prefix_restored_tokens ==
                  before.prefix_restored_tokens + expected_wave_restored,
              "P08 multiturn B=4 wave restored tokens mismatch");
      require(after.prefix_evictions == 0 && after.prefix_pinned_entries == 1,
              "P08 multiturn B=4 wave evicted or lost pinned system prompt");
      require(after.packed_batch_calls > before.packed_batch_calls,
              "P08 multiturn B=4 wave did not execute packed batch");
      if (wave == 1) {
        require(after.prefix_shared_hits == 3,
                "P08 multiturn B=4 Wave 1 expected 3 cross-slot CoW hits");
      }
    }
    require(histories[0].size() > 1024,
            "P08 multiturn B=4 did not cross 1024 tokens");

    // Verify stale Wave 1 frontier for Session 0 was superseded (falls back to
    // pinned System 512, not Wave 1's 639).
    {
      SubmitRequest probe;
      probe.prompt_tokens = s0_wave1_probe;
      probe.max_new_tokens = 2;
      probe.sampling = {.temperature = 0.0F, .top_p = 1.0F, .top_k = 1, .seed = 1};
      const auto before = b4->stats();
      require(run(*b4, probe) == run(*b4_cold, probe),
              "P08 multiturn B=4 supersede probe diverged");
      const auto after = b4->stats();
      require(after.prefix_partial_hits == before.prefix_partial_hits + 1 &&
                  after.prefix_restored_tokens ==
                      before.prefix_restored_tokens + shared.size(),
              "P08 multiturn B=4 stale Wave 1 entry was not superseded");
    }

    b4->shutdown();
    b4_cold->shutdown();
  }
}

void regression_mivr1_and_mivr2(const std::filesystem::path &fixtures,
                                const std::filesystem::path &libraries) {
  // MIVR-1: Packed multi-slot TTFT prefill (rem > 8) with prefix cache enabled
  // and LegacyRejectionV1 sampling must pass width=1 to NativeBatchSampler and
  // save the exact sampled token without poisoning SharedPrefixCache.
  {
    auto b4_opts = options(fixtures, libraries, true);
    b4_opts.max_concurrency = 4;
    b4_opts.path_overrides["batch_mode"] = "on";
    auto b4_eng = Engine::create(b4_opts);
    auto cold = Engine::create(options(fixtures, libraries, false));

    const std::array<std::size_t, 4> lens{16, 20, 24, 28};
    const std::array<std::size_t, 4> salts{51, 97, 143, 189};
    std::array<SubmitRequest, 4> reqs;
    std::array<std::vector<std::int32_t>, 4> oracles;
    std::array<std::uint64_t, 4> ids{};
    std::array<std::vector<std::int32_t>, 4> actuals;
    for (std::size_t s = 0; s < 4; ++s) {
      for (auto t : prompt(lens[s], salts[s]))
        reqs[s].prompt_tokens.push_back(std::int32_t(t));
      reqs[s].max_new_tokens = 4;
      reqs[s].pin_prefix = (s == 0);
      oracles[s] = run(*cold, reqs[s]);
    }
    for (std::size_t s = 0; s < 4; ++s)
      ids[s] = b4_eng->submit(reqs[s]);
    std::size_t finished = 0;
    const auto deadline = std::chrono::steady_clock::now() + 30s;
    while (finished < 4 && std::chrono::steady_clock::now() < deadline) {
      for (const auto &ev : b4_eng->poll_events(20ms)) {
        for (std::size_t s = 0; s < 4; ++s) {
          if (ev.request_id != ids[s])
            continue;
          if (ev.kind == EventKind::tokens) {
            actuals[s].insert(actuals[s].end(), ev.tokens.begin(),
                              ev.tokens.end());
          } else {
            require(ev.kind == EventKind::finished,
                    "MIVR-1 packed prefill request failed");
            ++finished;
          }
        }
      }
    }
    require(finished == 4, "MIVR-1 packed prefill timed out");
    const auto st = b4_eng->stats();
    require(st.packed_batch_calls >= 1 && st.prefix_pinned_entries == 1,
            "MIVR-1 did not exercise packed prefill or pinned count mismatch");
    for (std::size_t s = 0; s < 4; ++s)
      require(actuals[s] == oracles[s],
              "MIVR-1 packed prefill legacy sample diverged from cold oracle");

    // Exact prefix replays must also match oracle (proving prefix cache was not
    // poisoned with a stale/zero sampled_token).
    const auto before_hits = b4_eng->stats().prefix_hits;
    for (std::size_t s = 0; s < 4; ++s)
      require(run(*b4_eng, reqs[s]) == oracles[s],
              "MIVR-1 cached prefix replay poisoned by wrong sampled token");
    require(b4_eng->stats().prefix_hits == before_hits + 4,
            "MIVR-1 cached prefix replays did not hit");

    b4_eng->shutdown();
    cold->shutdown();
  }

  // MIVR-2 (a): Hard page-pressure eviction of a pinned prefix must update
  // stats.prefix_pinned_entries under stats_mutex_.
  {
    auto pressure_opts = options(fixtures, libraries, true);
    pressure_opts.path_overrides["kv_pool_tokens"] = "1088";
    auto pressure = Engine::create(pressure_opts);
    auto pressure_cold = Engine::create(options(fixtures, libraries, false));
    SubmitRequest req;
    for (auto token : prompt(1021, 313))
      req.prompt_tokens.push_back(std::int32_t(token));
    req.max_new_tokens = 1;
    req.pin_prefix = true;
    require(run(*pressure, req) == run(*pressure_cold, req),
            "MIVR-2 pinned seed diverged");
    require(pressure->stats().prefix_pinned_entries == 1,
            "MIVR-2 pinned count did not reflect pinned seed");
    SubmitRequest evict_req;
    for (auto token : prompt(192, 777))
      evict_req.prompt_tokens.push_back(std::int32_t(token));
    evict_req.max_new_tokens = 2;
    evict_req.pin_prefix = false;
    require(run(*pressure, evict_req) == run(*pressure_cold, evict_req),
            "MIVR-2 pressure eviction continuation diverged");
    const auto st = pressure->stats();
    require(st.prefix_evictions >= 1 && st.prefix_pinned_entries == 0,
            "MIVR-2 pinned count stale after page-pressure eviction");
    pressure->shutdown();
    pressure_cold->shutdown();
  }

  // MIVR-2 (b): Concurrent Engine::stats() polling from another thread while
  // worker inserts/pins/trims SharedPrefixCache entries.
  {
    auto eng = Engine::create(options(fixtures, libraries, true));
    std::atomic<bool> stop_reader{false};
    std::atomic<std::uint64_t> reads{0};
    std::atomic<bool> bad_count{false};
    std::thread reader([&] {
      while (!stop_reader.load(std::memory_order_acquire)) {
        const auto st = eng->stats();
        if (st.prefix_pinned_entries > 3)
          bad_count.store(true, std::memory_order_relaxed);
        reads.fetch_add(1, std::memory_order_relaxed);
      }
    });
    for (std::size_t i = 0; i < 12; ++i) {
      SubmitRequest req;
      for (auto t : prompt(16, 600 + i * 37))
        req.prompt_tokens.push_back(std::int32_t(t));
      req.max_new_tokens = 2;
      req.pin_prefix = (i < 3);
      static_cast<void>(run(*eng, req));
      if (i == 6)
        static_cast<void>(eng->trim_cache(0));
    }
    static_cast<void>(eng->trim_cache(0));
    stop_reader.store(true, std::memory_order_release);
    reader.join();
    require(reads.load() > 0 && !bad_count.load() &&
                eng->stats().prefix_pinned_entries == 3,
            "MIVR-2 concurrent stats() reader saw invalid pinned count");
    eng->shutdown();
  }

  // MIVR-12: Prefix-cache extra physical KV page sizing must deduct planned
  // weights, secondary slot buffers, and runtime_reserve_bytes from the
  // working-set budget instead of oversizing shared_kv_pages and failing load.
  {
    const auto root = fixtures / "affine";
    const auto cfg = load_qwen_model_config(root / "source/config.json");
    const auto art = ArtifactModel::open(root / "model.omlxm5", true);
    auto ctx = MetalContext::create();
    const auto planned_w = plan_model_materialization(
        art, ctx, {.components = {ModelComponent::text},
                   .require_verified_payload = false});
    const std::uint32_t kCtx = std::uint32_t(cfg.max_position_embeddings);
    const auto ws_rows =
        resolve_qwen_primary_workspace_rows(kCtx, false, cfg.num_experts > 1);
    const auto base_mem = plan_qwen_forward_memory(
        cfg, kCtx, 0U, false, false, false, false, false, false, 0U, ws_rows,
        false);
    const auto base_b = qwen_load_budget(
        planned_w.resident_bytes, base_mem,
        ctx.capabilities().recommended_working_set_bytes, 0ULL);

    // Tight budget: fits base load + ~1 MiB slack, while requesting 4 GiB
    // prefix cache (would previously blow qwen_load_budget by ignoring weights
    // and the 64 MiB runtime_reserve_bytes).
    auto tight_opts = options(fixtures, libraries, true);
    tight_opts.max_context = kCtx;
    tight_opts.memory_budget_bytes = base_b.required_bytes + (1ULL << 20);
    tight_opts.path_overrides["prefix_cache_bytes"] =
        std::to_string(4ULL << 30);

    auto cold_opts = options(fixtures, libraries, false);
    cold_opts.max_context = kCtx;
    auto tight_eng = Engine::create(tight_opts);
    auto tight_cold = Engine::create(cold_opts);
    SubmitRequest req;
    for (auto t : prompt(32, 911))
      req.prompt_tokens.push_back(std::int32_t(t));
    req.max_new_tokens = 3;
    require(run(*tight_eng, req) == run(*tight_cold, req),
            "MIVR-12 tight working-set prefix-cached engine diverged");
    tight_eng->shutdown();
    tight_cold->shutdown();
  }
}

void regression_phase3_runtime(const std::filesystem::path &fixtures,
                               const std::filesystem::path &libraries) {
  const auto root = fixtures / "affine";
  const auto cfg = load_qwen_model_config(root / "source/config.json");
  const auto art = ArtifactModel::open(root / "model.omlxm5", true);
  auto ctx = MetalContext::create();
  const auto planned_w = plan_model_materialization(
      art, ctx,
      {.components = {ModelComponent::text},
       .require_verified_payload = false});
  const std::uint32_t kCtx = std::uint32_t(cfg.max_position_embeddings);
  const auto ws_rows =
      resolve_qwen_primary_workspace_rows(kCtx, false, cfg.num_experts > 1);

  // MIVK-VF-4: NativeQwenState kv_dtype aliases ("affine4" == "affine4_g128",
  // "" == "bfloat16" == "bf16") must share identical memory accounting and
  // execution_identity.
  {
    auto a4_mem = plan_qwen_forward_memory(
        cfg, kCtx, 0U, false, false, false, false, false, false, 0U, ws_rows,
        /*affine4_kv=*/true);
    const auto sec_a4 = plan_qwen_forward_memory(
        cfg, kCtx, 0U, false, false, false, false, false, false, 0U, 16U,
        /*affine4_kv=*/true);
    const std::uint32_t logical_pages = (kCtx + 63U) / 64U;
    const std::uint64_t a4_page_b =
        kv_page_bytes(KvStorageFormat::Affine4Group128, 64U,
                      std::uint32_t(cfg.num_key_value_heads),
                      std::uint32_t(cfg.head_dim));
    const std::uint64_t a4_kv_slabs =
        std::uint64_t(logical_pages + 1U) * a4_page_b * 2ULL *
        std::uint64_t(cfg.full_attention_layers());
    a4_mem.workspace_bytes += sec_a4.workspace_bytes;
    a4_mem.state_bytes +=
        (sec_a4.state_bytes >= a4_kv_slabs ? (sec_a4.state_bytes - a4_kv_slabs)
                                           : 0ULL) +
        a4_kv_slabs; // B=2 shared_kv_pages = 2 * (logical_pages + 1)
    const auto a4_budget = qwen_load_budget(
        planned_w.resident_bytes, a4_mem,
        ctx.capabilities().recommended_working_set_bytes, 0ULL);

    auto make_kv_opts = [&](std::string kv_dtype, std::int32_t conc,
                            std::uint64_t budget) {
      auto o = options(fixtures, libraries, false);
      o.max_context = kCtx;
      o.max_concurrency = conc;
      o.kv_dtype = std::move(kv_dtype);
      o.memory_budget_bytes = budget;
      return o;
    };
    auto backend_exec_id = [](const detail::NativeBackend &be) {
      StatsReport st;
      be.populate_stats(st);
      return st.execution_identity_id;
    };

    // Tight budget (256 KiB slack < ~2.96 MiB BF16-Affine4 delta across B=2)
    // must reject BF16 yet succeed for BOTH "affine4" and "affine4_g128".
    auto make_tight_a4_budget = [&]() {
      return a4_budget.required_bytes + a4_budget.runtime_reserve_bytes +
             ctx.current_allocated_bytes() + (256ULL << 10);
    };
    bool bf16_rejected_on_tight_a4_budget = false;
    try {
      detail::NativeBackend be_bf16_b2(
          make_kv_opts("bf16", 2, make_tight_a4_budget()));
      (void)be_bf16_b2;
    } catch (const std::runtime_error &) {
      bf16_rejected_on_tight_a4_budget = true;
    }
    require(bf16_rejected_on_tight_a4_budget,
            "MIVK-VF-4 tight affine4 budget did not bound BF16 KV");
    std::string a4_alias_id;
    {
      detail::NativeBackend be_a4_alias(
          make_kv_opts("affine4", 2, make_tight_a4_budget()));
      a4_alias_id = backend_exec_id(be_a4_alias);
    }
    std::string a4_canon_id;
    {
      detail::NativeBackend be_a4_canon(
          make_kv_opts("affine4_g128", 2, make_tight_a4_budget()));
      a4_canon_id = backend_exec_id(be_a4_canon);
    }
    require(!a4_alias_id.empty() && a4_alias_id == a4_canon_id,
            "MIVK-VF-4 affine4 alias and affine4_g128 execution_identity differ");

    detail::NativeBackend be_bf16(make_kv_opts("bf16", 1, 0ULL));
    detail::NativeBackend be_bfloat16(make_kv_opts("bfloat16", 1, 0ULL));
    detail::NativeBackend be_empty(make_kv_opts("", 1, 0ULL));
    const auto bf16_id = backend_exec_id(be_bf16);
    require(bf16_id == backend_exec_id(be_bfloat16) &&
                bf16_id == backend_exec_id(be_empty) &&
                bf16_id != a4_canon_id,
            "MIVK-VF-4 bf16/bfloat16/empty execution_identity differ");

    // MIVK-VF-2 alignment: PositionCoupledV1 must ignore flash_nax_prefill in
    // execution_identity as well.
    auto coupled_off = make_kv_opts("bf16", 1, 0ULL);
    coupled_off.sampling_contract = "position_coupled_v1";
    auto coupled_on = coupled_off;
    coupled_on.path_overrides["flash_nax_prefill"] = "1";
    auto eng_c_off = Engine::create(coupled_off);
    auto eng_c_on = Engine::create(coupled_on);
    require(eng_c_off->stats().execution_identity_id ==
                eng_c_on->stats().execution_identity_id,
            "MIVK-VF-2 PositionCoupledV1 stamped flash_nax_prefill identity");
    eng_c_off->shutdown();
    eng_c_on->shutdown();
  }

  // MIVR-22: NativeBackend::execute must reject an already-occupied slot 0 and
  // keep slot 0 reusable after normal/exceptional paths.
  {
    auto be_opts = options(fixtures, libraries, false);
    be_opts.max_concurrency = 2;
    detail::NativeBackend backend(be_opts);
    backend.bind_owner(std::this_thread::get_id());

    SubmitRequest req;
    for (auto t : prompt(8, 123))
      req.prompt_tokens.push_back(std::int32_t(t));
    req.max_new_tokens = 2;
    auto cancel_ptr = std::make_shared<std::atomic<bool>>(false);
    const std::atomic<bool> cancel_flag{false};
    auto noop_emit = [](std::vector<std::int32_t>) {};

    auto task0 = backend.start(req, cancel_ptr, noop_emit);
    bool rejected_occupied = false;
    try {
      static_cast<void>(backend.execute(2U, req, cancel_flag, noop_emit));
    } catch (const std::logic_error &) {
      rejected_occupied = true;
    }
    require(rejected_occupied,
            "MIVR-22 NativeBackend::execute did not reject occupied slot 0");

    task0.reset();
    const auto res = backend.execute(3U, req, cancel_flag, noop_emit);
    require(res.kind == EventKind::finished,
            "MIVR-22 NativeBackend::execute failed after slot 0 release");
  }
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 4)
      throw std::invalid_argument("p08_prefix_tests QWEN_FIXTURES LIBRARIES DFLASH_FIXTURE");
    qwen_restore(argv[1], argv[2]);
    engine_restore(argv[1], argv[2], argv[3]);
    multiturn_cache_and_slab_invariants(argv[1], argv[2]);
    multiturn_engine_tests(argv[1], argv[2], argv[3]);
    regression_mivr1_and_mivr2(argv[1], argv[2]);
    regression_phase3_runtime(argv[1], argv[2]);
    std::cout << "P08 complete checkpoint, multi-entry LRU, cross-seed hits, DFlash and rolling multi-turn passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "P08 prefix failure: " << error.what() << '\n';
    return 1;
  }
}
