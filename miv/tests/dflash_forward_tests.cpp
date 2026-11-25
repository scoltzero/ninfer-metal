#include "omlx_m5/artifact.h"
#include "omlx_m5/dflash_forward.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using namespace omlx::m5;
using Json = nlohmann::json;
std::vector<float> flatten(const Json &j) {
  std::vector<float> out;
  const auto visit = [&](auto &self, const Json &v) -> void {
    if (v.is_array())
      for (const auto &x : v)
        self(self, x);
    else
      out.push_back(v.get<float>());
  };
  visit(visit, j);
  return out;
}
Json compare(const std::vector<float> &a, const std::vector<float> &b) {
  if (a.size() != b.size())
    throw std::runtime_error("shape mismatch");
  double maximum = 0, error = 0, scale = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      throw std::runtime_error("nonfinite");
    double d = a[i] - b[i];
    maximum = std::max(maximum, std::abs(d));
    error += d * d;
    scale += double(b[i]) * b[i];
  }
  double relative = std::sqrt(error / std::max(scale, 1e-12));
  return {{"max_abs", maximum},
          {"relative_l2", relative},
          {"passed", std::isfinite(maximum) && std::isfinite(relative) &&
                         maximum <= 0.08 && relative <= 0.02}};
}
int main(int argc, char **argv) {
  try {
    if (argc < 4 || argc > 6)
      throw std::invalid_argument(
          "dflash_forward_tests TARGET_FIXTURES DRAFT_FIXTURE LIBRARIES [--row-exact]");
    bool row_exact=false,position=false;
    for(int i=4;i<argc;++i) {
      if(std::string_view(argv[i])=="--row-exact")row_exact=true;
      else if(std::string_view(argv[i])=="--position-coupled")position=true;
      else throw std::invalid_argument("unknown draft numeric/sampling policy");
    }
    Json traces = Json::array();
    Json report = {{"cases", Json::array()}, {"replay_cases", 0}};
    bool passed = true;
    auto context = MetalContext::create();
    const auto root = std::filesystem::path(argv[1]) / "affine";
    auto artifact = ArtifactModel::open(root / "model.omlxm5", true);
    auto weights = materialize_model(
        artifact, plan_model_materialization(artifact, context), context);
    auto config = load_qwen_model_config(root / "source/config.json");
    auto layers = build_qwen_layer_catalog(artifact, config, 2048, 64);
    const std::array<std::uint32_t, 2> taps{0, 1};
    QwenForward target(context, weights, layers, config, 2048, argv[3], taps,
                       false, true);
    auto draft_artifact = ArtifactModel::open(
        std::filesystem::path(argv[2]) / "model.omlxm5", true);
    auto draft_config = load_dflash2_model_config(
        std::filesystem::path(argv[2]) / "source/config.json");
    auto valid = validate_dflash2_artifact(draft_artifact, draft_config, false);
    if (valid.shared_embedding_present || valid.proposal_head_present)
      throw std::runtime_error("fixture contains copied target weights");
    auto draft_weights = materialize_model(
        draft_artifact,
        plan_model_materialization(draft_artifact, context,
                                   {.components = {ModelComponent::speculative},
                                    .require_verified_payload = true}),
        context);
    auto test_features = context.make_buffer(4096 * 256 * 2);
    const auto before_draft = context.registry_stats();
    DFlashForward draft(context, draft_weights, draft_config, 4096,
                        target.embedding_weight(), target.head_weight(),
                        test_features, argv[3], false,
                        row_exact ? LinearNumericPolicy::RowExactPortableV1
                                  : LinearNumericPolicy::Legacy,
                        position?SamplingContract::PositionCoupledV1:SamplingContract::LegacyRejectionV1);
    const auto installed = context.registry_stats();
    const auto draft_memory = plan_dflash_forward_memory(draft_config, 4096);
    const auto numeric_profile = draft.inspect_profile();
    if (row_exact) {
      if (numeric_profile.numeric_contract_id != "linear_row_exact_portable_v1" ||
          numeric_profile.weights.empty() || numeric_profile.whole_program_row_exact_qualified)
        throw std::runtime_error("draft row-exact metadata scope invalid");
      for (const auto &r:numeric_profile.weights) {
        const bool valid_source =
            r.source == "row_exact_portable_v1" ||
            r.source == "bf16_nax_m5_v1" ||
            r.source == "mxfp4_nax_m5_v1" ||
            r.source == "affine_nax_m5_v1";
        const bool valid_lib =
            r.library == "row_exact" ||
            r.library == "bf16_nax" ||
            r.library == "mxfp4_nax" ||
            r.library == "affine_nax";
        if (!r.row_exact_eligible || r.row_exact_group.empty() ||
            !valid_source || !valid_lib)
          throw std::runtime_error("draft real installer bypassed row-exact selector");
      }
    }
    report["linear_numeric_contract_id"] = numeric_profile.numeric_contract_id;
    report["row_exact_installer_checked"] = row_exact;
    report["whole_program_row_exact_qualified"] = false;
    report["sampling_contract_id"]=numeric_profile.sampling_contract_id;
    if (installed.current_buffer_bytes - before_draft.current_buffer_bytes !=
        draft_memory.buffer_bytes() + installed.dynamic_parameter_bytes -
            before_draft.dynamic_parameter_bytes)
      throw std::runtime_error("draft buffer plan differs from allocations");
    Json oracle;
    std::ifstream(std::filesystem::path(argv[2]) / "oracle.json") >> oracle;
    for (const auto &test : oracle.at("cases")) {
      // Each oracle case owns a different target feature history. Incremental
      // reuse is valid only within one continuous request.
      draft.reset();
      auto features = flatten(test.at("features"));
      std::vector<std::uint16_t> raw(features.size());
      for (size_t i = 0; i < raw.size(); ++i)
        raw[i] = std::bit_cast<std::uint32_t>(features[i]) >> 16;
      test_features.write(std::as_bytes(std::span(raw)));
      auto cpu = draft.propose(7, test.at("context"), test.at("k"), 0, 123,
                               false, true, true);
      auto direct = draft.propose(7, test.at("context"), test.at("k"), 0, 123,
                                  false, true);
      auto icb = draft.propose(7, test.at("context"), test.at("k"), 0, 123,
                               true, true);
      if (direct.hidden != icb.hidden || direct.logits != icb.logits ||
          direct.selection.tokens != icb.selection.tokens ||
          direct.selection.tokens != cpu.selection.tokens ||
          direct.selection.steps.size() != cpu.selection.steps.size() ||
          !direct.gpu_selector)
        throw std::runtime_error("draft ICB/direct mismatch");
      for (std::size_t i = 0; i < direct.selection.steps.size(); ++i) {
        if (direct.selection.steps[i].candidate_ids !=
                cpu.selection.steps[i].candidate_ids ||
            direct.selection.steps[i].probabilities.size() !=
                cpu.selection.steps[i].probabilities.size())
          throw std::runtime_error("GPU greedy selector differs from CPU");
        for (std::size_t j = 0;
             j < direct.selection.steps[i].probabilities.size(); ++j) {
          const float lhs = direct.selection.steps[i].probabilities[j];
          const float rhs = cpu.selection.steps[i].probabilities[j];
          if (!std::isfinite(lhs) || !std::isfinite(rhs) ||
              !(std::abs(lhs - rhs) <= 2e-6f))
            throw std::runtime_error("GPU greedy selector probability drift");
        }
      }
      auto hidden = compare(direct.hidden, flatten(test.at("hidden"))),
           logits = compare(direct.logits, flatten(test.at("logits")));
      auto expected = test.at("proposals").get<std::vector<std::int32_t>>();
      passed &=
          hidden.at("passed").get<bool>() && logits.at("passed").get<bool>();
      for (float temperature : {0.0F, 0.7F}) {
        auto selected = temperature == 0
                            ? direct
                            : draft.propose(7, test.at("context"), test.at("k"),
                                            temperature, 123, true, true);
        auto reference =
            temperature == 0
                ? cpu
                : draft.propose(7, test.at("context"), test.at("k"),
                                temperature, 123, true, true, true);
        if (selected.selection.tokens != reference.selection.tokens ||
            selected.selection.steps.size() != reference.selection.steps.size())
          throw std::runtime_error("GPU conditional selector token mismatch");
        for (std::size_t i = 0; i < selected.selection.steps.size(); ++i) {
          const auto &a = selected.selection.steps[i];
          const auto &b = reference.selection.steps[i];
          if (a.candidate_ids != b.candidate_ids ||
              a.probabilities.size() != b.probabilities.size())
            throw std::runtime_error("GPU conditional selector top-k mismatch");
          for (std::size_t j = 0; j < a.probabilities.size(); ++j)
            if (!std::isfinite(a.probabilities[j]) ||
                !std::isfinite(b.probabilities[j]) ||
                !(std::abs(a.probabilities[j] - b.probabilities[j]) <= 2e-6f))
              throw std::runtime_error("GPU conditional q differs from CPU");
        }
        Json steps = Json::array();
        for (const auto &step : selected.selection.steps)
          steps.push_back({{"token", step.token},
                           {"ids", step.candidate_ids},
                           {"probabilities", step.probabilities}});
        traces.push_back({{"context", test.at("context")},
                          {"k", test.at("k")},
                          {"temperature", temperature},
                          {"hidden", selected.hidden},
                          {"logits", selected.logits},
                          {"steps", steps}});
      }
      report["cases"].push_back(
          {{"context", test.at("context")},
           {"k", test.at("k")},
           {"hidden", hidden},
           {"logits", logits},
           {"proposal_parity", direct.selection.tokens == expected},
           {"actual_proposals", direct.selection.tokens},
           {"expected_proposals", expected},
           {"dispatches", direct.draft_dispatches}});
    }
    // One continuous feature history: growing prefixes must reuse previously
    // committed rows, while a fresh materialization must be bit-exact.
    std::vector<std::uint16_t> stable_features(4096 * 256);
    for (std::size_t i = 0; i < stable_features.size(); ++i) {
      const float value = float(int((i * 17U) % 257U) - 128) / 1024.0F;
      stable_features[i] = std::bit_cast<std::uint32_t>(value) >> 16;
    }
    test_features.write(std::as_bytes(std::span(stable_features)));
    draft.reset();
    report["p03_compact_cases"] = Json::array();
    for (unsigned k = 1; k <= 7; ++k)
      for (float temperature : {0.0F, 0.7F}) {
        const auto seed = 123U + 17U * k;
        const auto cpu =
            draft.propose(7, 33, k, temperature, seed, true, true, true);
        const auto gpu = draft.propose(7, 33, k, temperature, seed);
        if (!gpu.hidden.empty() || !gpu.logits.empty() || !gpu.gpu_selector ||
            gpu.compact_cpu_fallback ||
            gpu.selection.tokens != cpu.selection.tokens ||
            gpu.selection.steps.size() != cpu.selection.steps.size() ||
            gpu.host_read_bytes >= cpu.host_read_bytes)
          throw std::runtime_error("P03 compact GPU selector boundary");
        for (std::size_t row = 0; row < gpu.selection.steps.size(); ++row) {
          const auto &a = gpu.selection.steps[row];
          const auto &b = cpu.selection.steps[row];
          if (a.candidate_ids != b.candidate_ids)
            throw std::runtime_error("P03 compact top-k IDs changed");
          for (std::size_t index = 0; index < a.probabilities.size(); ++index)
            if (!std::isfinite(a.probabilities[index]) ||
                !std::isfinite(b.probabilities[index]) ||
                !(std::abs(a.probabilities[index] - b.probabilities[index]) <=
                  2e-6F))
              throw std::runtime_error("P03 compact q probabilities changed");
        }
        report["p03_compact_cases"].push_back(
            {{"k", k},
             {"temperature", temperature},
             {"gpu_host_read_bytes", gpu.host_read_bytes},
             {"cpu_host_read_bytes", cpu.host_read_bytes},
             {"tokens", gpu.selection.tokens}});
      }
    const auto edge_reference =
        draft.propose(7, 33, 1, 2.0F, 0, true, true, true);
    const auto &edge_q = edge_reference.selection.steps.front().probabilities;
    std::uint64_t edge_seed = 0;
    for (std::uint64_t seed = 1; seed < 200000 && !edge_seed; ++seed) {
      if(position) {
        double best=-INFINITY,second=-INFINITY;
        const auto &ids=edge_reference.selection.steps.front().candidate_ids;
        for(std::size_t i=0;i<edge_q.size();++i) {
          const double u=position_sampling_uniform(seed,34,std::uint32_t(ids[i]),draft_sampling_purpose);
          const double score=std::log(double(edge_q[i]))-std::log(-std::log(u));
          if(score>best){second=best;best=score;}else second=std::max(second,score);
        }
        if(best-second<.00005)edge_seed=seed;
        continue;
      }
      double cumulative = 0;
      const double draw = candidate_selector_uniform(seed, 33);
      for (std::size_t i = 0; i + 1 < edge_q.size(); ++i) {
        cumulative += edge_q[i];
        if (std::abs(draw - cumulative) < 0.00005) {
          edge_seed = seed;
          break;
        }
      }
    }
    if (!edge_seed)
      throw std::runtime_error("P03 could not exercise random boundary");
    const auto edge_cpu =
        draft.propose(7, 33, 1, 2.0F, edge_seed, true, true, true);
    const auto edge_gpu = draft.propose(7, 33, 1, 2.0F, edge_seed);
    if (!edge_gpu.compact_cpu_fallback ||
        edge_gpu.selection.tokens != edge_cpu.selection.tokens ||
        edge_gpu.selection.steps.front().candidate_ids !=
            edge_cpu.selection.steps.front().candidate_ids ||
        edge_gpu.selection.steps.front().probabilities !=
            edge_cpu.selection.steps.front().probabilities ||
        !edge_gpu.hidden.empty() || !edge_gpu.logits.empty())
      throw std::runtime_error(
          "P03 compact random-boundary fallback changed q");
    report["p03_boundary_fallback"] = {
        {"seed", edge_seed},
        {"gpu_host_read_bytes", edge_gpu.host_read_bytes},
        {"cpu_host_read_bytes", edge_cpu.host_read_bytes},
        {"compact", true}};
    draft.reset();
    unsigned previous = 0;
    report["incremental_cache"] = Json::array();
    for (unsigned length : {1U, 3U, 7U, 12U, 20U, 32U, 33U, 63U, 64U, 65U, 127U,
                            128U, 129U, 1023U, 1024U, 1025U, 2048U, 2049U,
                            4096U}) {
      auto incremental = draft.propose(7, length, 7, 0, 123, true, true);
      const unsigned first = length >= draft_config.sliding_window
                                 ? length - draft_config.sliding_window + 1
                                 : 0;
      const unsigned expected_new = length - std::max(first, previous);
      if (incremental.context_rows_projected != expected_new ||
          incremental.context_rows_launched > 2 * std::max(1U, expected_new))
        throw std::runtime_error(
            "draft incremental context work disagrees with frontier");
      auto hot = draft.propose(7, length, 7, 0, 123, true, true);
      if (hot.context_rows_projected || hot.hidden != incremental.hidden ||
          hot.logits != incremental.logits ||
          hot.selection.tokens != incremental.selection.tokens)
        throw std::runtime_error(
            "hot draft repeated context or changed outputs");
      draft.reset();
      auto cold = draft.propose(7, length, 7, 0, 123, true, true);
      if (incremental.hidden != cold.hidden ||
          incremental.logits != cold.logits ||
          incremental.selection.tokens != cold.selection.tokens)
        throw std::runtime_error(
            "incremental draft differs from cold materialization");
      report["incremental_cache"].push_back(
          {{"context", length},
           {"new_rows", incremental.context_rows_projected},
           {"launched_rows", incremental.context_rows_launched},
           {"reused_rows", incremental.context_rows_reused},
           {"hot_rows", hot.context_rows_projected},
           {"hot_gpu_ms", hot.submission.timing.gpu_ms},
           {"cold_gpu_ms", cold.submission.timing.gpu_ms}});
      previous = length;
    }
    bool rewind_rejected = false;
    try {
      static_cast<void>(draft.propose(7, 1, 7, 0, 123));
    } catch (const std::invalid_argument &) {
      rewind_rejected = true;
    }
    if (!rewind_rejected)
      throw std::runtime_error("draft reused a rewound sequence");
    draft.reset();
    static_cast<void>(draft.propose(7, 1, 7, 0, 123));
    const auto after = context.registry_stats();
    if (installed.pipelines_compiled != after.pipelines_compiled ||
        installed.command_plans_built != after.command_plans_built ||
        installed.current_buffer_count != after.current_buffer_count ||
        installed.upload_count != after.upload_count)
      throw std::runtime_error("draft loop created installation resources");
    // Real GDN replay must match teacher forcing for every accepted prefix, not
    // a synthetic SequenceState hash. Verify writes no committed recurrent
    // state.
    for (const char *mode : {"affine", "mxfp4", "moe-affine", "moe-mxfp4",
                             "mixed-oq", "moe-mixed-oq"}) {
      auto path = std::filesystem::path(argv[1]) / mode;
      auto art = ArtifactModel::open(path / "model.omlxm5", true);
      auto ctx = MetalContext::create();
      auto cfg = load_qwen_model_config(path / "source/config.json");
      auto ws =
          materialize_model(art, plan_model_materialization(art, ctx), ctx);
      auto ls = build_qwen_layer_catalog(art, cfg, 1024, 64);
      QwenForward transaction(ctx, ws, ls, cfg, 1024, argv[3], taps, false,
                              true);
      // A separate context avoids plan-key aliasing between two live mutable
      // executors.
      auto tc = MetalContext::create();
      auto tw = materialize_model(art, plan_model_materialization(art, tc), tc);
      QwenForward teacher(tc, tw, ls, cfg, 1024, argv[3], taps, false, true);
      std::vector<std::uint32_t> prompt(33);
      for (unsigned i = 0; i < prompt.size(); ++i)
        prompt[i] = (71 * i + 256) % 257;
      const std::array<std::uint32_t, 8> tokens{7,   3,  100, 200,
                                                256, 44, 23,  87};
      for (unsigned n = 0; n <= 8; ++n) {
        transaction.reset();
        teacher.reset();
        static_cast<void>(transaction.run(prompt));
        static_cast<void>(teacher.run(prompt));
        auto before = transaction.inspect_recurrent_state();
        static_cast<void>(transaction.verify(tokens));
        if (before != transaction.inspect_recurrent_state() ||
            transaction.position() != 33)
          throw std::runtime_error("verification mutated committed state");
        static_cast<void>(transaction.commit_prefix(n));
        if (n)
          static_cast<void>(teacher.run(std::span(tokens).first(n)));
        if (n == 8 && transaction.inspect_recurrent_state() !=
                          teacher.inspect_recurrent_state())
          throw std::runtime_error(
              "full-prefix replay not bit-exact with teacher state");
        if (transaction.position() != teacher.position())
          throw std::runtime_error("committed frontier mismatch");
        const std::array<std::uint32_t, 1> continuation{37};
        auto a = transaction.run(continuation, {}, true),
             b = teacher.run(continuation, {}, true);
        auto error = compare(a.logits, b.logits);
        if (!error.at("passed").get<bool>())
          throw std::runtime_error(std::string(mode) +
                                   " replay continuation: " + error.dump());
        report["replay_cases"] = report["replay_cases"].get<int>() + 1;
      }
      if (std::string(mode) == "affine") {
        for (unsigned accepted = 0; accepted <= 7; ++accepted) {
          transaction.reset();
          teacher.reset();
          auto anchor = transaction.run(prompt).token;
          static_cast<void>(teacher.run(prompt));
          std::vector<std::uint32_t> input{anchor};
          for (unsigned i = 0; i < 7; ++i) {
            auto token = teacher.run(std::span(&input.back(), 1)).token;
            input.push_back(token);
          }
          if (accepted < 7)
            input[accepted + 1] = (input[accepted + 1] + 1) % 257;
          auto verification = transaction.verify(input, {}, true);
          std::vector<SelectorStep> proposals;
          std::vector<TargetDistribution> distributions;
          for (unsigned row = 0; row < 8; ++row) {
            auto begin = verification.logits.begin() + row * 257;
            auto id =
                std::uint32_t(std::max_element(begin, begin + 257) - begin);
            TargetDistribution distribution{std::vector<float>(257, 0)};
            distribution.probabilities[id] = 1;
            distributions.push_back(std::move(distribution));
            if (row < 7)
              proposals.push_back(
                  {.token = std::int32_t(input[row + 1]),
                   .candidate_ids = {std::int32_t(input[row + 1])},
                   .probabilities = {1}});
          }
          auto verified =
              verify_speculative(proposals, distributions, 0, 123, 33);
          if (verified.accepted_proposals != accepted ||
              verified.used_correction != (accepted < 7))
            throw std::runtime_error(
                "real target acceptance/correction mismatch");
          auto count = verified.licensed_outputs.size();
          static_cast<void>(transaction.commit_prefix(std::uint32_t(count)));
          teacher.reset();
          static_cast<void>(teacher.run(prompt));
          static_cast<void>(teacher.run(std::span(input).first(count)));
          auto continuation = std::uint32_t(verified.licensed_outputs.back());
          auto a = transaction.run(std::span(&continuation, 1), {}, true);
          auto b = teacher.run(std::span(&continuation, 1), {}, true);
          if (!compare(a.logits, b.logits).at("passed").get<bool>())
            throw std::runtime_error(
                "licensed real target continuation mismatch");
        }
        report["real_acceptance_cases"] = 8;
      }
      transaction.reset();
      static_cast<void>(transaction.run(prompt));
      auto before = transaction.inspect_recurrent_state();
      static_cast<void>(transaction.verify(tokens));
      transaction.abort_verify();
      if (before != transaction.inspect_recurrent_state() ||
          transaction.position() != 33)
        throw std::runtime_error("abort changed committed state");
    }
    // Regression tests for MIVE-7:
    // (1) plan_dflash_forward_memory must include o_proj input width q in "sums".
    {
      auto wide_q_cfg = draft_config;
      wide_q_cfg.hidden_size = 64;
      wide_q_cfg.intermediate_size = 64;
      wide_q_cfg.query_heads = 16;
      wide_q_cfg.head_dim = 16;
      const auto wide_plan = plan_dflash_forward_memory(wide_q_cfg, 256);
      const auto it = std::find_if(
          wide_plan.buffers.begin(), wide_plan.buffers.end(),
          [](const auto &buf) { return buf.name == "sums"; });
      if (it == wide_plan.buffers.end() ||
          it->bytes != 8ULL * ((256ULL + 31ULL) / 32ULL) * 4ULL)
        throw std::runtime_error(
            "MIVE-7: plan_dflash_forward_memory sums buffer omitted q width");
    }
    // (2) propose_packed_batch must reject duplicate slot_draft pointers and
    // secondary slots with mismatched max_context or sliding_window.
    {
      const auto expect_packed_invalid =
          [&](std::span<const DFlashPackedSlotSlice> slices,
              const char *label) {
            bool threw = false;
            try {
              static_cast<void>(draft.propose_packed_batch(slices, false));
            } catch (const std::invalid_argument &) {
              threw = true;
            }
            if (!threw)
              throw std::runtime_error(std::string("MIVE-7: ") + label);
          };
      const std::array<DFlashPackedSlotSlice, 2> dup_slices{
          DFlashPackedSlotSlice{.slot_draft = &draft,
                                .anchor = 7,
                                .context_tokens = 1,
                                .draft_tokens = 1},
          DFlashPackedSlotSlice{.slot_draft = &draft,
                                .anchor = 7,
                                .context_tokens = 1,
                                .draft_tokens = 1}};
      expect_packed_invalid(dup_slices, "duplicate slot_draft was not rejected");

      DFlashForward short_ctx_draft(
          context, draft_weights, draft_config, 2048, target.embedding_weight(),
          target.head_weight(), test_features, argv[3], false,
          row_exact ? LinearNumericPolicy::RowExactPortableV1
                    : LinearNumericPolicy::Legacy,
          position ? SamplingContract::PositionCoupledV1
                   : SamplingContract::LegacyRejectionV1);
      const std::array<DFlashPackedSlotSlice, 2> ctx_mismatch_slices{
          DFlashPackedSlotSlice{.slot_draft = &draft,
                                .anchor = 7,
                                .context_tokens = 1,
                                .draft_tokens = 1},
          DFlashPackedSlotSlice{.slot_draft = &short_ctx_draft,
                                .anchor = 7,
                                .context_tokens = 1,
                                .draft_tokens = 1}};
      expect_packed_invalid(ctx_mismatch_slices,
                            "mismatched max_context slot was not rejected");

      auto alt_win_cfg = draft_config;
      alt_win_cfg.sliding_window =
          draft_config.sliding_window > 1 ? draft_config.sliding_window - 1
                                          : draft_config.sliding_window + 1;
      DFlashForward short_win_draft(
          context, draft_weights, alt_win_cfg, 4096, target.embedding_weight(),
          target.head_weight(), test_features, argv[3], false,
          row_exact ? LinearNumericPolicy::RowExactPortableV1
                    : LinearNumericPolicy::Legacy,
          position ? SamplingContract::PositionCoupledV1
                   : SamplingContract::LegacyRejectionV1);
      const std::array<DFlashPackedSlotSlice, 2> win_mismatch_slices{
          DFlashPackedSlotSlice{.slot_draft = &draft,
                                .anchor = 7,
                                .context_tokens = 1,
                                .draft_tokens = 1},
          DFlashPackedSlotSlice{.slot_draft = &short_win_draft,
                                .anchor = 7,
                                .context_tokens = 1,
                                .draft_tokens = 1}};
      expect_packed_invalid(win_mismatch_slices,
                            "mismatched sliding_window slot was not rejected");
    }
    std::ofstream(std::filesystem::path(argv[2]) / "native-traces.json")
        << traces.dump();
    report["passed"] = passed;
    std::cout << report.dump(2) << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
