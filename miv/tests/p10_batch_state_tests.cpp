#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "omlx_m5/artifact.h"
#include "omlx_m5/qwen_forward.h"
#include "../src/affine_nax_cache.h"
#include "../src/attention_nax.h"
#include "../src/mxfp4_nax_cache.h"

using namespace omlx::m5;
namespace {
void require(bool value, const char *why) {
  if (!value) throw std::runtime_error(why);
}
std::vector<std::uint32_t> tokens(std::size_t count, std::uint32_t salt) {
  std::vector<std::uint32_t> result(count);
  for (std::size_t i = 0; i < count; ++i)
    result[i] = (salt + std::uint32_t(i * 71)) % 257;
  return result;
}
void check(const std::filesystem::path &root,
           const std::filesystem::path &libs, bool verify,
           bool split_attention = false) {
  auto config = load_qwen_model_config(root / "source/config.json");
  auto artifact = ArtifactModel::open(root / "model.omlxm5", true);
  auto context = MetalContext::create();
  auto weights = materialize_model(
      artifact, plan_model_materialization(artifact, context), context);
  auto layers = build_qwen_layer_catalog(artifact, config, 1024, 64);
  const std::array<std::uint32_t, 2> taps{0, 1};
  std::vector<std::unique_ptr<QwenForward>> batch, serial;
  for (int i = 0; i < 8; ++i) {
    batch.push_back(std::make_unique<QwenForward>(
        context, weights, layers, config, 1024, libs, taps, false, true,
        false, std::nullopt, split_attention, true));
    serial.push_back(std::make_unique<QwenForward>(
        context, weights, layers, config, 1024, libs, taps, false, true,
        false, std::nullopt, split_attention));
  }
  const auto installed = context.registry_stats();
  std::vector<std::vector<std::uint32_t>> inputs;
  std::vector<QwenPreparedStep> prepared;
  std::vector<MetalCommandInvocation> calls;
  std::vector<QwenForwardResult> reference;
  for (int i = 0; i < 8; ++i) {
    const auto prefix = tokens(63 + i, std::uint32_t(i * 11));
    static_cast<void>(batch[i]->run(prefix));
    static_cast<void>(serial[i]->run(prefix));
    inputs.push_back(tokens(i + 1, std::uint32_t(29 + i * 7)));
    prepared.push_back(batch[i]->prepare_batch(inputs.back(), {}, verify));
    calls.push_back(prepared.back().invocation());
    reference.push_back(verify ? serial[i]->verify(inputs.back(), {}, true)
                               : serial[i]->run(inputs.back(), {}, true));
  }
  const auto submission = context.execute_batch(calls);
  require(submission.dispatch_count > 0, "P10 multi-row submit was empty");
  for (int i = 0; i < 8; ++i) {
    auto step = batch[i]->finish_batch(std::move(prepared[i]));
    require(step.tokens == inputs[i].size() &&
                batch[i]->read_logits(step.tokens) == reference[i].logits,
            "P10 per-row logits differ from serial Qwen");
    if (verify) {
      const std::uint32_t accepted = i % 3 == 0 ? 0 :
                                     i % 3 == 1 ? step.tokens / 2 : step.tokens;
      static_cast<void>(batch[i]->commit_prefix(accepted));
      static_cast<void>(serial[i]->commit_prefix(accepted));
    }
    require(batch[i]->position() == serial[i]->position() &&
                batch[i]->inspect_recurrent_state() ==
                    serial[i]->inspect_recurrent_state(),
            "P10 GDN/conv or position crossed rows");
    const auto feature_bytes =
        std::uint64_t(batch[i]->position()) * taps.size() * config.hidden_size * 2;
    require(batch[i]->feature_history().read(0, feature_bytes) ==
                serial[i]->feature_history().read(0, feature_bytes),
            "P10 target feature history crossed rows");
    const std::array<std::uint32_t, 1> continuation{std::uint32_t(31 + i)};
    auto actual = batch[i]->run(continuation, {}, true);
    auto expected = serial[i]->run(continuation, {}, true);
    require(actual.token == expected.token && actual.logits == expected.logits,
            "P10 continued output differs after batch/Replay");
  }
  const auto finished = context.registry_stats();
  require(finished.pipelines_compiled == installed.pipelines_compiled &&
              finished.command_plans_built == installed.command_plans_built &&
              finished.current_buffer_count == installed.current_buffer_count &&
              finished.icb_profiles_created == installed.icb_profiles_created,
          "P10 token loop recompiled a pipeline/plan or allocated a GPU buffer");
}

void check_mive4_mive5_and_mivk_vf1(const std::filesystem::path &root,
                                    const std::filesystem::path &libs) {
  auto config = load_qwen_model_config(root / "source/config.json");
  auto artifact = ArtifactModel::open(root / "model.omlxm5", true);
  auto context = MetalContext::create();
  auto weights = materialize_model(
      artifact, plan_model_materialization(artifact, context), context);
  auto layers = build_qwen_layer_catalog(artifact, config, 1024, 64);
  const std::array<std::uint32_t, 2> taps{0, 1};

  // MIVK-VF-1: Legacy + affine4_g128 and Legacy + flash_nax_prefill must plan
  // and allocate attention_positions so affine4 NAX/split-KV kernels activate.
  {
    auto has_buf = [](const QwenForwardMemoryPlan &plan,
                      std::string_view name) {
      return std::any_of(
          plan.buffers.begin(), plan.buffers.end(),
          [&](const QwenBufferPlan &b) { return b.name == name; });
    };
    const auto a4_plan = plan_qwen_forward_memory(
        config, 1024, 0, false, false, true, false, false, false, 0, 1024,
        /*affine4_kv=*/true);
    require(has_buf(a4_plan, "attention_positions"),
            "MIVK-VF-1 Legacy+affine4_g128 omitted attention_positions");
    const auto fnax_plan = plan_qwen_forward_memory(
        config, 1024, 0, false, false, false, false, false, false, 0, 1024,
        /*affine4_kv=*/false, 0, /*flash_nax_prefill=*/true);
    require(has_buf(fnax_plan, "attention_positions"),
            "MIVK-VF-1 Legacy+flash_nax_prefill omitted attention_positions");

    QwenForward legacy_a4(
        context, weights, layers, config, 1024, libs, {},
        /*inspect_routes=*/false, /*speculative=*/false, false, std::nullopt,
        /*enable_attention_split=*/true, false, std::nullopt,
        LinearNumericPolicy::Legacy, SamplingContract::LegacyRejectionV1,
        AttentionNumericPolicy::Legacy, false, "affine4_g128");
    require(has_buf(legacy_a4.memory_plan(), "attention_positions"),
            "MIVK-VF-1 Legacy+affine4_g128 QwenForward memory_plan missing attention_positions");
    const std::array<std::uint32_t, 8> a4_prompt{3, 7, 11, 15, 19, 23, 27, 31};
    const auto a4_r0 = legacy_a4.run(a4_prompt);
    const std::array<std::uint32_t, 1> a4_next{a4_r0.token};
    const auto a4_r1 = legacy_a4.run(a4_next);
    require(a4_r0.token < std::uint32_t(config.vocab_size) &&
                a4_r1.token < std::uint32_t(config.vocab_size),
            "MIVK-VF-1 Legacy+affine4_g128 generated out-of-range token");
  }

  // MIVE-4 & MIVE-5:
  // - MIVE-4: non-verify packed AR batch with speculative=true writes back
  //   pre-final_norm trunk hidden rows to secondary sp.hidden[0] and to
  //   primary hidden[0] when primary is placed at r_off > 0.
  // - MIVE-5: secondary delegated prefill (>16 tokens) and secondary-only
  //   packed batch invalidate primary has_last_logits.
  {
    const GpuSamplingOptions samp{
        .temperature = 0.0F,
        .top_k = 1,
        .top_p = 1.0F,
        .seed = 1,
        .contract = SamplingContract::PositionCoupledV1};
    QwenForward slot0(context, weights, layers, config, 1024, libs, taps, false,
                      /*speculative=*/true, false, std::nullopt, false,
                      /*enable_batch=*/true, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1,
                      AttentionNumericPolicy::RowExactV1);
    QwenForward slot1(context, weights, layers, config, 1024, libs, taps, false,
                      /*speculative=*/true, false, std::nullopt, false,
                      /*enable_batch=*/true, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1,
                      AttentionNumericPolicy::RowExactV1, false, "bf16",
                      &slot0);
    QwenForward solo0(context, weights, layers, config, 1024, libs, taps, false,
                      /*speculative=*/true, false, std::nullopt, false,
                      /*enable_batch=*/true, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1,
                      AttentionNumericPolicy::RowExactV1);
    QwenForward solo1(context, weights, layers, config, 1024, libs, taps, false,
                      /*speculative=*/true, false, std::nullopt, false,
                      /*enable_batch=*/true, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1,
                      AttentionNumericPolicy::RowExactV1);

    const auto p0 = tokens(8, 3);
    const auto p1 = tokens(24, 11);
    static_cast<void>(slot0.run(p0, samp));
    static_cast<void>(solo0.run(p0, samp));
    require(slot0.read_last_logits_bf16() == solo0.read_last_logits_bf16(),
            "MIVE-5 initial slot0 logits mismatch");

    // Secondary >16-token delegated prefill overwrites primary logits[0].
    static_cast<void>(slot1.run(p1, samp));
    static_cast<void>(solo1.run(p1, samp));
    require(slot1.read_last_logits_bf16() == solo1.read_last_logits_bf16(),
            "MIVE-5 secondary delegated prefill logits mismatch");
    bool stale_primary_rejected = false;
    try {
      static_cast<void>(slot0.read_last_logits_bf16());
    } catch (const std::logic_error &) {
      stale_primary_rejected = true;
    }
    require(stale_primary_rejected,
            "MIVE-5 primary read_last_logits_bf16 returned secondary logits");

    // MIVE-4 normal order {slot0, slot1}:
    const std::array<std::uint32_t, 1> d0{5}, d1{9};
    const std::array<PackedSlotSlice, 2> fwd_slices{{
        {&slot0, 0U, d0, samp, false},
        {&slot1, 1U, d1, samp, false},
    }};
    static_cast<void>(slot0.run_packed_batch(fwd_slices, true));
    static_cast<void>(solo0.run(d0, samp));
    static_cast<void>(solo1.run(d1, samp));
    const std::uint64_t row_bytes = std::uint64_t(config.hidden_size) * 2ULL;
    require(slot0.trunk_hidden_buffer().read(0, row_bytes) ==
                    solo0.trunk_hidden_buffer().read(0, row_bytes) &&
                slot1.trunk_hidden_buffer().read(0, row_bytes) ==
                    solo1.trunk_hidden_buffer().read(0, row_bytes),
            "MIVE-4 packed non-verify AR did not scatter trunk_hidden to slot1");

    // MIVE-4 reverse order {slot1, slot0} (primary at r_off == 1):
    const std::array<std::uint32_t, 1> d0b{13}, d1b{21};
    const std::array<PackedSlotSlice, 2> rev_slices{{
        {&slot1, 1U, d1b, samp, false},
        {&slot0, 0U, d0b, samp, false},
    }};
    static_cast<void>(slot0.run_packed_batch(rev_slices, true));
    static_cast<void>(solo0.run(d0b, samp));
    static_cast<void>(solo1.run(d1b, samp));
    require(slot0.trunk_hidden_buffer().read(0, row_bytes) ==
                    solo0.trunk_hidden_buffer().read(0, row_bytes) &&
                slot1.trunk_hidden_buffer().read(0, row_bytes) ==
                    solo1.trunk_hidden_buffer().read(0, row_bytes),
            "MIVE-4 reversed packed non-verify AR did not update primary hidden[0]");

    // MIVE-5 secondary-only packed batch:
    require(slot0.read_last_logits_bf16() == solo0.read_last_logits_bf16(),
            "MIVE-5 slot0 logits mismatch before secondary-only packed batch");
    const std::array<std::uint32_t, 1> d1c{25};
    const std::array<PackedSlotSlice, 1> sec_only{{
        {&slot1, 1U, d1c, samp, false},
    }};
    static_cast<void>(slot0.run_packed_batch(sec_only, true));
    bool stale_after_packed_rejected = false;
    try {
      static_cast<void>(slot0.read_last_logits_bf16());
    } catch (const std::logic_error &) {
      stale_after_packed_rejected = true;
    }
    require(stale_after_packed_rejected,
            "MIVE-5 primary read_last_logits_bf16 not invalidated after secondary-only packed batch");
  }
}

void check_phase3_engine_regressions(const std::filesystem::path &fixtures,
                                     const std::filesystem::path &libs) {
  const auto root = fixtures / "affine";
  const auto config = load_qwen_model_config(root / "source/config.json");
  const auto artifact = ArtifactModel::open(root / "model.omlxm5", true);
  auto context = MetalContext::create();
  auto weights = materialize_model(
      artifact, plan_model_materialization(artifact, context), context);
  auto layers = build_qwen_layer_catalog(artifact, config, 1024, 64);
  const std::array<std::uint32_t, 2> taps{0, 1};
  const GpuSamplingOptions samp{
      .temperature = 0.0F,
      .top_k = 1,
      .top_p = 1.0F,
      .seed = 1,
      .contract = SamplingContract::PositionCoupledV1};

  // MIVE-1: Packed verify with primary at b_idx=1 (r_off=2 < w_b=5, total_m=7<=8)
  // must scatter both logits and target features without overlapping in-place race.
  {
    QwenForward slot0(context, weights, layers, config, 1024, libs, taps, false,
                      /*speculative=*/true, false, std::nullopt, false,
                      /*enable_batch=*/true, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1,
                      AttentionNumericPolicy::RowExactV1);
    QwenForward slot1(context, weights, layers, config, 1024, libs, taps, false,
                      /*speculative=*/true, false, std::nullopt, false,
                      /*enable_batch=*/true, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1,
                      AttentionNumericPolicy::RowExactV1, false, "bf16",
                      &slot0);
    QwenForward solo0(context, weights, layers, config, 1024, libs, taps, false,
                      /*speculative=*/true, false, std::nullopt, false,
                      /*enable_batch=*/true, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1,
                      AttentionNumericPolicy::RowExactV1);
    QwenForward solo1(context, weights, layers, config, 1024, libs, taps, false,
                      /*speculative=*/true, false, std::nullopt, false,
                      /*enable_batch=*/true, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1,
                      AttentionNumericPolicy::RowExactV1);

    const auto p0 = tokens(12, 5);
    const auto p1 = tokens(10, 19);
    static_cast<void>(slot0.run(p0, samp));
    static_cast<void>(solo0.run(p0, samp));
    static_cast<void>(slot1.run(p1, samp));
    static_cast<void>(solo1.run(p1, samp));

    const auto v1 = tokens(2, 31);
    const auto v0 = tokens(5, 43);
    const std::array<PackedSlotSlice, 2> verify_slices{{
        {&slot1, 1U, v1, samp, true},
        {&slot0, 0U, v0, samp, true},
    }};
    static_cast<void>(slot0.run_packed_batch(verify_slices, true, true));
    const auto ref0 = solo0.verify(v0, samp, true);
    const auto ref1 = solo1.verify(v1, samp, true);
    require(slot0.read_logits(5) == ref0.logits,
            "MIVE-1 primary verify logits corrupted by in-place overlap");
    require(slot1.read_logits(2) == ref1.logits,
            "MIVE-1 secondary verify logits diverged");

    static_cast<void>(slot0.commit_prefix(5));
    static_cast<void>(solo0.commit_prefix(5));
    static_cast<void>(slot1.commit_prefix(2));
    static_cast<void>(solo1.commit_prefix(2));
    const std::uint64_t feat0_bytes =
        std::uint64_t(slot0.position()) * taps.size() * config.hidden_size * 2;
    const std::uint64_t feat1_bytes =
        std::uint64_t(slot1.position()) * taps.size() * config.hidden_size * 2;
    require(slot0.feature_history().read(0, feat0_bytes) ==
                solo0.feature_history().read(0, feat0_bytes),
            "MIVE-1 primary verify features corrupted by in-place overlap");
    require(slot1.feature_history().read(0, feat1_bytes) ==
                solo1.feature_history().read(0, feat1_bytes),
            "MIVE-1 secondary verify features diverged");
  }

  // MIVE-12: Incremental owner-cache acquisition, target_split_k disambiguation,
  // pointer stability across insertions, and concurrent reader safety.
  {
    using detail::forward::AffineNaxCache;
    using detail::forward::Mxfp4NaxCache;
    using detail::forward::Ref;

    auto zero_buf = context.make_buffer(256);
    zero_buf.zero();
    const Ref zero_ref{&zero_buf, 0, 256};

    auto make_aff_w = [&](std::uint32_t id, const MetalBuffer &codes,
                          const MetalBuffer &sb) {
      MaterializedWeightView w;
      w.object_id = id;
      w.name = "syn_aff_" + std::to_string(id);
      w.format = "oq_affine_q4_g64_bf16_v1";
      w.layout = "oq_nax_tiled_g64_v1";
      w.parent_shape = {32ULL, 512ULL};
      w.padded_k = 512ULL;
      w.row_count = 32ULL;
      w.in_features = 512ULL;
      w.planes = {
          {.kind = "nax_codes", .buffer = &codes, .bytes = codes.size()},
          {.kind = "nax_sb", .buffer = &sb, .bytes = sb.size()},
      };
      return w;
    };
    auto aff_codes = context.make_buffer(32ULL * 512ULL / 2ULL);
    auto aff_sb = context.make_buffer(32ULL * (512ULL / 64ULL) * 4ULL);
    aff_codes.zero();
    aff_sb.zero();
    const auto aw0 = make_aff_w(101U, aff_codes, aff_sb);
    const auto aw1 = make_aff_w(102U, aff_codes, aff_sb);
    const auto aw2 = make_aff_w(103U, aff_codes, aff_sb);

    int aff_owner_tag = 0;
    auto ac1 = AffineNaxCache::acquire(&aff_owner_tag, context, libs, zero_ref,
                                       {aw0}, 8U);
    const auto *ap0 = ac1->find(101U);
    require(ap0 != nullptr && ac1->find(102U) == nullptr,
            "MIVE-12 AffineNaxCache initial subset unexpected");
    auto ac2 = AffineNaxCache::acquire(&aff_owner_tag, context, libs, zero_ref,
                                       {aw0, aw1}, 8U);
    require(ac2.get() == ac1.get() && ac2->find(101U) == ap0 &&
                ac2->find(102U) != nullptr,
            "MIVE-12 AffineNaxCache skipped incremental build on cache hit");
    auto ac_sk4 = AffineNaxCache::acquire(&aff_owner_tag, context, libs,
                                          zero_ref, {aw0}, 4U);
    require(ac_sk4.get() != ac1.get() && ac_sk4->find(101U) != nullptr,
            "MIVE-12 AffineNaxCache failed to disambiguate target_split_k");

    std::atomic<bool> stop_reader{false};
    std::atomic<bool> reader_bad{false};
    std::thread reader([&] {
      while (!stop_reader.load(std::memory_order_acquire)) {
        if (!ac1->active() || ac1->find(101U) != ap0)
          reader_bad.store(true, std::memory_order_relaxed);
      }
    });
    auto ac3 = AffineNaxCache::acquire(&aff_owner_tag, context, libs, zero_ref,
                                       {aw0, aw1, aw2}, 8U);
    stop_reader.store(true, std::memory_order_release);
    reader.join();
    require(!reader_bad.load() && ac3.get() == ac1.get() &&
                ac3->find(103U) != nullptr && ac3->find(101U) == ap0,
            "MIVE-12 AffineNaxCache concurrent reader/incremental build failed");

    auto make_mx_w = [&](std::uint32_t id, const MetalBuffer &data,
                         const MetalBuffer &scale) {
      MaterializedWeightView w;
      w.object_id = id;
      w.name = "syn_mx_" + std::to_string(id);
      w.format = "mxfp4_e2m1_g32_e8m0_v1";
      w.layout = "mx_nax_tiled_bk128_v1";
      w.parent_shape = {32ULL, 512ULL};
      w.padded_k = 512ULL;
      w.row_count = 32ULL;
      w.in_features = 512ULL;
      w.planes = {
          {.kind = "nax_data", .buffer = &data, .bytes = data.size()},
          {.kind = "nax_scale", .buffer = &scale, .bytes = scale.size()},
      };
      return w;
    };
    auto mx_data = context.make_buffer(32ULL * 512ULL / 2ULL);
    auto mx_scale = context.make_buffer(32ULL * (512ULL / 32ULL));
    mx_data.zero();
    mx_scale.zero();
    const auto mw0 = make_mx_w(201U, mx_data, mx_scale);
    const auto mw1 = make_mx_w(202U, mx_data, mx_scale);
    int mx_owner_tag = 0;
    auto mc1 =
        Mxfp4NaxCache::acquire(&mx_owner_tag, context, libs, {mw0}, 8U);
    const auto *mp0 = mc1->find(201U);
    require(mp0 != nullptr && mc1->find(202U) == nullptr,
            "MIVE-12 Mxfp4NaxCache initial subset unexpected");
    auto mc2 =
        Mxfp4NaxCache::acquire(&mx_owner_tag, context, libs, {mw0, mw1}, 8U);
    require(mc2.get() == mc1.get() && mc2->find(201U) == mp0 &&
                mc2->find(202U) != nullptr,
            "MIVE-12 Mxfp4NaxCache skipped incremental build on cache hit");
    auto mc_sk4 =
        Mxfp4NaxCache::acquire(&mx_owner_tag, context, libs, {mw0}, 4U);
    require(mc_sk4.get() != mc1.get() && mc_sk4->find(201U) != nullptr,
            "MIVE-12 Mxfp4NaxCache failed to disambiguate target_split_k");
  }

  // MIVE-14: Reject execute/prepare_batch/checkpoint*/run_packed_batch while
  // async steps are in flight; drain & invalidate stale handles on reset/restore.
  {
    QwenForward q(context, weights, layers, config, 1024, libs, taps, false,
                  /*speculative=*/true, false, std::nullopt, true,
                  /*enable_batch=*/true, std::nullopt,
                  LinearNumericPolicy::RowExactPortableV1,
                  SamplingContract::PositionCoupledV1,
                  AttentionNumericPolicy::RowExactV1);
    const auto p = tokens(16, 41);
    const auto init = q.run(p, samp);
    auto ckpt = q.checkpoint_uma();
    if (q.can_pipeline_ar(samp)) {
      auto ar1 = q.enqueue_ar_step(samp, true, init.token);
      auto expect_reject = [&](auto &&fn, const char *msg) {
        bool threw = false;
        try {
          fn();
        } catch (const std::logic_error &) {
          threw = true;
        }
        require(threw, msg);
      };
      const std::array<std::uint32_t, 1> one{init.token};
      expect_reject([&] { static_cast<void>(q.run(one, samp)); },
                    "MIVE-14 execute allowed with async AR in flight");
      expect_reject([&] { static_cast<void>(q.prepare_batch(one, samp)); },
                    "MIVE-14 prepare_batch allowed with async AR in flight");
      expect_reject([&] { static_cast<void>(q.checkpoint()); },
                    "MIVE-14 checkpoint allowed with async AR in flight");
      expect_reject([&] { static_cast<void>(q.checkpoint_uma()); },
                    "MIVE-14 checkpoint_uma allowed with async AR in flight");
      const std::array<PackedSlotSlice, 1> ps{{{&q, 0U, one, samp, false}}};
      expect_reject([&] { static_cast<void>(q.run_packed_batch(ps, true)); },
                    "MIVE-14 run_packed_batch allowed with async AR in flight");

      q.restore_uma(ckpt);
      require(q.position() == 16U,
              "MIVE-14 restore_uma did not restore position");
      expect_reject([&] { static_cast<void>(q.finish_ar_step(std::move(ar1))); },
                    "MIVE-14 stale async AR handle accepted after restore_uma");

      auto ar2 = q.enqueue_ar_step(samp, true, init.token);
      q.reset();
      require(q.position() == 0U, "MIVE-14 reset did not zero position");
      const auto init2 = q.run(p, samp);
      auto ar3 = q.enqueue_ar_step(samp, true, init2.token);
      {
        auto drop_stale = std::move(ar2);
        (void)drop_stale;
      }
      const auto fin3 = q.finish_ar_step(std::move(ar3));
      require(fin3.token < std::uint32_t(config.vocab_size) &&
                  q.position() == 17U,
              "MIVE-14 stale handle dtor corrupted live async AR step");
    }
    q.release_uma(ckpt);

    q.reset();
    if (q.can_pipeline_prefill(64)) {
      auto pf = q.enqueue_prefill_step(tokens(64, 77));
      bool run_rejected = false;
      try {
        static_cast<void>(q.run(tokens(1, 3), samp));
      } catch (const std::logic_error &) {
        run_rejected = true;
      }
      require(run_rejected,
              "MIVE-14 execute allowed with async prefill in flight");
      q.reset();
      bool stale_pf_rejected = false;
      try {
        static_cast<void>(q.finish_prefill_step(std::move(pf)));
      } catch (const std::logic_error &) {
        stale_pf_rejected = true;
      }
      require(stale_pf_rejected && q.position() == 0U,
              "MIVE-14 stale async prefill handle accepted after reset");
    }
  }

  // MIVK-VF-2: RowExactV1 must block non-row-exact flash_nax_prefill / frag48.
  {
    QwenForward exact_off(context, weights, layers, config, 1024, libs, {},
                          false, true, false, std::nullopt, true, false,
                          std::nullopt,
                          LinearNumericPolicy::RowExactPortableV1,
                          SamplingContract::PositionCoupledV1,
                          AttentionNumericPolicy::RowExactV1, false, "bf16",
                          nullptr, /*enable_flash_nax_prefill=*/false);
    QwenForward exact_on(context, weights, layers, config, 1024, libs, {},
                         false, true, false, std::nullopt, true, false,
                         std::nullopt, LinearNumericPolicy::RowExactPortableV1,
                         SamplingContract::PositionCoupledV1,
                         AttentionNumericPolicy::RowExactV1, false, "bf16",
                         nullptr, /*enable_flash_nax_prefill=*/true);
    require(exact_on.inspect_profile(32).attention_row_exact_eligible &&
                exact_on.inspect_profile(4, true).attention_row_exact_eligible,
            "MIVK-VF-2 RowExactV1 selected non-row-exact flash_nax kernel");
    const auto p40 = tokens(40, 13);
    const auto r_off = exact_off.run(p40, samp, true);
    const auto r_on = exact_on.run(p40, samp, true);
    require(r_off.logits == r_on.logits && r_off.token == r_on.token,
            "MIVK-VF-2 RowExactV1 prefill diverged with flash_nax_prefill=true");
    const auto v4 = tokens(4, 99);
    const auto v_off = exact_off.verify(v4, samp, true);
    const auto v_on = exact_on.verify(v4, samp, true);
    require(v_off.logits == v_on.logits,
            "MIVK-VF-2 RowExactV1 verify diverged with flash_nax_prefill=true");
  }

  // MIVK-VF-3: RowExactV1 ar_pipe_ must ignore split_kv_partition_cap_override_
  // on <12-partition chips and stay 0-ULP consistent with sync run().
  {
    require(detail::forward::exact_attention_split_chunk_tokens(8192U, 2U) ==
                    1024U &&
                detail::forward::exact_attention_split_chunk_tokens(8192U, 0U) ==
                    512U,
            "MIVK-VF-3 precondition chunk_tokens check failed");
    const char *prev_cores = std::getenv("OMLX_M5_GPU_CORES");
    const std::optional<std::string> saved_cores =
        prev_cores ? std::optional<std::string>(prev_cores) : std::nullopt;
    setenv("OMLX_M5_GPU_CORES", "10", 1);
    struct EnvRestore {
      std::optional<std::string> prev;
      ~EnvRestore() {
        if (prev.has_value())
          setenv("OMLX_M5_GPU_CORES", prev->c_str(), 1);
        else
          unsetenv("OMLX_M5_GPU_CORES");
      }
    } restore_cores{saved_cores};

    auto long_cfg = config;
    long_cfg.max_position_embeddings = 8192;
    auto long_layers = build_qwen_layer_catalog(artifact, long_cfg, 8192, 64);
    QwenForward sync_q(context, weights, long_layers, long_cfg, 8192, libs, {},
                       false, false, false, std::nullopt, true, false,
                       std::nullopt, LinearNumericPolicy::RowExactPortableV1,
                       SamplingContract::PositionCoupledV1,
                       AttentionNumericPolicy::RowExactV1);
    QwenForward async_q(context, weights, long_layers, long_cfg, 8192, libs, {},
                        false, false, false, std::nullopt, true, false,
                        std::nullopt, LinearNumericPolicy::RowExactPortableV1,
                        SamplingContract::PositionCoupledV1,
                        AttentionNumericPolicy::RowExactV1);
    const auto p600 = tokens(600, 17);
    const auto s0 = sync_q.run(p600, samp);
    const auto a0 = async_q.run(p600, samp);
    require(s0.token == a0.token, "MIVK-VF-3 prefill token diverged");
    if (async_q.can_pipeline_ar(samp)) {
      auto st1 = async_q.enqueue_ar_step(samp, true, a0.token);
      auto st2 = async_q.enqueue_ar_step(samp, false, 0U);
      const auto af1 = async_q.finish_ar_step(std::move(st1));
      const auto af2 = async_q.finish_ar_step(std::move(st2));
      const std::array<std::uint32_t, 1> t1{s0.token};
      const auto sf1 = sync_q.run(t1, samp);
      const std::array<std::uint32_t, 1> t2{sf1.token};
      const auto sf2 = sync_q.run(t2, samp);
      require(af1.token == sf1.token && af2.token == sf2.token &&
                  async_q.read_last_logits_bf16() ==
                      sync_q.read_last_logits_bf16(),
              "MIVK-VF-3 ar_pipe_ diverged from sync RowExactV1 on <12-partition chip");
    }
  }

  // MIVE-15: emit_paged_attention_affine4_g128 must preserve caller query/gate
  // strides in split_params (hd==128 and hd==256), forward tok_word/pos_word,
  // and record pos_patches.
  if (std::filesystem::is_regular_file(libs /
                                       "omlx_m5_attention_affine4.metallib") &&
      detail::forward::affine4_kv_runtime_eligible(context, libs, 8U, 2U,
                                                   128U)) {
    using detail::forward::Encoder;
    using detail::forward::Ref;
    using detail::forward::U32;
    using detail::forward::U64;

    auto scratch = context.make_buffer(4ULL << 20);
    const Ref r{&scratch, 0, scratch.size()};
    for (const U32 hd : {128U, 256U}) {
      QwenModelConfig sc = config;
      sc.num_attention_heads = 8;
      sc.num_key_value_heads = 2;
      sc.head_dim = int(hd);
      Encoder enc(context, libs, 2U,
                  "mive15.test.hd" + std::to_string(hd));
      enc.begin_stage("attn");
      const U32 custom_q_tok_stride = 4096U;
      const U32 custom_q_head_stride = hd * 4U;
      const U32 custom_g_tok_stride = 4096U;
      const U32 custom_g_head_stride = hd * 4U;
      const U32 pos_ph = 0xFF000001U;
      auto in_params = detail::forward::words(
          {8U, 2U, hd, 2U, pos_ph, 2048U, 64U, 33U, custom_q_tok_stride,
           custom_q_head_stride, 2U * hd, 2U * hd, custom_g_tok_stride,
           custom_g_head_stride});
      std::vector<std::pair<U64, U32>> patches;
      detail::forward::emit_paged_attention_affine4_g128(
          enc, sc, 2U, /*split=*/true, /*context_envelope=*/1024U,
          /*max_context=*/2048U, /*physical_pages=*/33U, r, r, r, r, r, r, r, r,
          r, r, std::move(in_params), /*tok_word=*/std::nullopt,
          /*pos_word=*/std::nullopt, &patches, /*q_prepacked=*/false);
      const std::size_t expected_patches = (hd == 256U) ? 3U : 2U;
      require(patches.size() == expected_patches && enc.profile.patches.empty(),
              "MIVE-15 affine4 split_attention patch wiring incorrect");
      for (const auto &[off, word_idx] : patches) {
        require(word_idx == 4U &&
                    off + 16U * sizeof(U32) <=
                        enc.profile.parameter_data.size(),
                "MIVE-15 affine4 split_attention pos_patch offset invalid");
        std::array<U32, 16> got{};
        std::memcpy(got.data(), enc.profile.parameter_data.data() + off,
                    sizeof(got));
        require(got[4] == pos_ph && got[8] == custom_q_tok_stride &&
                    got[9] == custom_q_head_stride &&
                    got[12] == custom_g_tok_stride &&
                    got[13] == custom_g_head_stride && got[14] == 2U &&
                    got[15] == 512U,
                "MIVE-15 affine4 split_params dropped caller strides or pos");
      }
    }
  }
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument("p10_batch_state_tests FIXTURES LIBRARIES");
    for (const auto *codec : {"affine", "mxfp4"}) {
      const auto root = std::filesystem::path(argv[1]) / codec;
      check(root, argv[2], false);
      check(root, argv[2], true);
    }
    check_mive4_mive5_and_mivk_vf1(
        std::filesystem::path(argv[1]) / "affine", argv[2]);
    check_phase3_engine_regressions(argv[1], argv[2]);
    if (MetalContext::create()
            .capabilities()
            .profile.probe.apple_gpu_family >= 9) {
      check(std::filesystem::path(argv[1]) / "affine", argv[2], false, true);
      check(std::filesystem::path(argv[1]) / "affine", argv[2], true, true);
    }
    std::cout << "P10 B8 W1..8 logits/state/Replay matched serial\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "P10 batch state failure: " << error.what() << '\n';
    return 1;
  }
}
