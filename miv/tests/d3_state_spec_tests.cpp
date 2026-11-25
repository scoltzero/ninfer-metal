#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "omlx_m5/dflash2.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/state.h"

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool bitwise_equal(const std::vector<float> &left,
                   const std::vector<float> &right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) ==
             0;
}

std::vector<omlx::m5::GdnTransitionRecord>
make_records(const omlx::m5::GdnGeometry &geometry, std::size_t count) {
  std::mt19937 generator(20260919);
  std::uniform_real_distribution<float> distribution(-0.25F, 0.25F);
  std::vector<omlx::m5::GdnTransitionRecord> records;
  for (std::size_t step = 0; step < count; ++step) {
    std::vector<float> key(geometry.key_dim);
    std::vector<float> value(geometry.value_dim);
    std::ranges::generate(key, [&] { return distribution(generator); });
    std::ranges::generate(value, [&] { return distribution(generator); });
    records.push_back(omlx::m5::make_represented_record(
        key, value, -0.01F * static_cast<float>(step + 1),
        0.25F + 0.01F * static_cast<float>(step)));
  }
  return records;
}

void test_replay_closed_loop() {
  const omlx::m5::GdnGeometry geometry{
      .key_dim = 8,
      .value_dim = 4,
      .epsilon = 1.0e-6F,
  };
  std::vector<float> initial_values(geometry.key_dim * geometry.value_dim);
  for (std::size_t index = 0; index < initial_values.size(); ++index) {
    initial_values[index] = static_cast<float>(index) * 0.001F;
  }
  const omlx::m5::GdnState checkpoint(geometry, initial_values);
  const auto records = make_records(geometry, 8);
  for (std::size_t accepted = 0; accepted <= 7; ++accepted) {
    auto direct = checkpoint;
    for (std::size_t step = 0; step < accepted; ++step) {
      direct.apply(records[step]);
    }
    const auto replayed =
        omlx::m5::replay_prefix(checkpoint, records, accepted);
    expect(bitwise_equal(direct.values(), replayed.values()),
           "ReplaySSM prefix is not a bitwise closed-loop clone");
  }
}

void test_round_transaction() {
  const omlx::m5::GdnGeometry geometry{.key_dim = 8, .value_dim = 4};
  const auto records = make_records(geometry, 8);
  omlx::m5::SequenceState sequence(64, omlx::m5::GdnState(geometry));
  sequence.target_frontier = 63;
  sequence.context_frontier = 63;
  sequence.kv_page_count = 1;

  {
    omlx::m5::RoundTransaction transaction(sequence, 8);
    transaction.set_verify_records(records);
    transaction.set_licensed_outputs({10, 11, 12, 13, 14});
    const auto published = transaction.commit(3);
    expect(published == std::vector<std::int32_t>({10, 11, 12}),
           "transaction published the wrong prefix");
    expect(transaction.stage() == omlx::m5::RoundStage::committed,
           "transaction did not enter committed state");
  }
  expect(sequence.target_frontier == 66, "target frontier did not advance");
  expect(sequence.context_frontier == 63,
         "context advanced before materialize");
  expect(sequence.kv_page_count == 2,
         "KV page frontier did not cross boundary");
  expect(sequence.generated_tokens == 3, "generated counter mismatch");
  expect(sequence.pending_anchor == 12, "pending anchor mismatch");
  sequence.materialize_pending_context();
  expect(sequence.context_frontier == 66, "context did not catch up");

  const auto committed_state = sequence.gdn.values();
  {
    omlx::m5::RoundTransaction transaction(sequence, 8);
    transaction.cancel();
  }
  expect(bitwise_equal(committed_state, sequence.gdn.values()),
         "cancelled transaction mutated committed state");
  {
    omlx::m5::RoundTransaction transaction(sequence, 8);
    transaction.set_verify_records(records);
    transaction.abort();
  }
  expect(bitwise_equal(committed_state, sequence.gdn.values()),
         "aborted verified transaction mutated committed state");

  for (std::uint32_t width = 2; width <= 8; ++width) {
    omlx::m5::SequenceState width_sequence(64, omlx::m5::GdnState(geometry));
    omlx::m5::RoundTransaction transaction(width_sequence, width);
    transaction.set_verify_records(std::vector<omlx::m5::GdnTransitionRecord>(
        records.begin(), records.begin() + width));
    transaction.set_licensed_outputs({42});
    expect(transaction.commit(1) == std::vector<std::int32_t>({42}),
           "M=2..8 transaction width coverage failed");
  }
}

void test_dynamic_convolution_and_context() {
  const omlx::m5::DynamicConvSpec spec{.width = 4, .group_width = 2};
  const std::vector<float> input = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
  };
  const std::vector<float> base = {
      1, 1, 1, 1, 0.5F, 0.5F, 0.5F, 0.5F,
  };
  const std::vector<float> delta(3 * 2 * 2, 0.0F);
  const auto output =
      omlx::m5::dynamic_grouped_conv_two_tap(input, 3, spec, base, delta);
  expect(output == std::vector<float>({
                       1,
                       2,
                       3,
                       4,
                       5.5F,
                       7,
                       8.5F,
                       10,
                       11.5F,
                       13,
                       14.5F,
                       16,
                   }),
         "two-tap dynamic convolution mismatch");

  const std::vector<float> features = {3, 4, 0, 5};
  const std::vector<float> identity = {1, 0, 0, 1};
  const std::vector<float> norm = {1, 1};
  const auto context = omlx::m5::materialize_target_context(
      features, 2, 2, identity, 2, norm, 1.0e-6F);
  expect(context.size() == 4, "context materializer output size mismatch");
  for (std::size_t row = 0; row < 2; ++row) {
    const auto mean_square = (context[row * 2] * context[row * 2] +
                              context[row * 2 + 1] * context[row * 2 + 1]) /
                             2.0F;
    expect(std::abs(mean_square - 1.0F) < 1.0e-5F,
           "context materializer RMS normalization mismatch");
  }
}

void test_selector_and_verification() {
  constexpr std::uint32_t vocab = 4;
  constexpr std::uint32_t rank = 1;
  const std::vector<float> predecessor = {0, 1, 2, 3};
  const std::vector<float> successor = {0, 1, 2, 3};
  const std::vector<omlx::m5::CandidateRow> rows = {
      {
          .token_ids = {1, 2},
          .unary_scores = {0.0F, 0.0F},
          .projected_hidden = {1.0F},
      },
      {
          .token_ids = {2, 3},
          .unary_scores = {0.0F, 0.0F},
          .projected_hidden = {1.0F},
      },
  };
  const auto selected = omlx::m5::select_candidate_path(
      1, rows, predecessor, successor, vocab, rank, 0.0F, 7, 10);
  expect(selected.tokens == std::vector<std::int32_t>({2, 3}),
         "greedy conditional candidate path mismatch");
  expect(selected.steps[0].probabilities == std::vector<float>({0.0F, 1.0F}),
         "greedy proposal q is not one-hot");

  const auto sampled_a = omlx::m5::select_candidate_path(
      1, rows, predecessor, successor, vocab, rank, 0.7F, 99, 10);
  const auto sampled_b = omlx::m5::select_candidate_path(
      1, rows, predecessor, successor, vocab, rank, 0.7F, 99, 10);
  expect(sampled_a.tokens == sampled_b.tokens,
         "selector RNG is not batch-independent and deterministic");

  const std::vector<omlx::m5::TargetDistribution> mismatch_target = {
      {.probabilities = {0.0F, 0.0F, 1.0F, 0.0F}},
      {.probabilities = {0.0F, 1.0F, 0.0F, 0.0F}},
      {.probabilities = {0.0F, 0.0F, 0.0F, 1.0F}},
  };
  const auto mismatch = omlx::m5::verify_speculative(
      selected.steps, mismatch_target, 0.0F, 0, 10);
  expect(mismatch.accepted_proposals == 1, "accepted prefix mismatch");
  expect(mismatch.used_correction, "target mismatch did not use correction");
  expect(mismatch.licensed_outputs == std::vector<std::int32_t>({2, 1}),
         "greedy correction token mismatch");

  const std::vector<omlx::m5::TargetDistribution> all_accept_target = {
      {.probabilities = {0.0F, 0.0F, 1.0F, 0.0F}},
      {.probabilities = {0.0F, 0.0F, 0.0F, 1.0F}},
      {.probabilities = {1.0F, 0.0F, 0.0F, 0.0F}},
  };
  const auto all_accept = omlx::m5::verify_speculative(
      selected.steps, all_accept_target, 0.0F, 0, 10);
  expect(all_accept.accepted_proposals == 2, "all-accept prefix mismatch");
  expect(!all_accept.used_correction, "all-accept used correction");
  expect(all_accept.licensed_outputs == std::vector<std::int32_t>({2, 3, 0}),
         "bonus token mismatch");

  bool observed_correction = false;
  const omlx::m5::SelectorStep stochastic_proposal{
      .token = 1,
      .candidate_ids = {1, 2},
      .probabilities = {0.95F, 0.05F},
  };
  const std::vector<omlx::m5::TargetDistribution> stochastic_target = {
      {.probabilities = {0.8F, 0.05F, 0.15F, 0.0F}},
      {.probabilities = {0.0F, 0.0F, 0.0F, 1.0F}},
  };
  for (std::uint64_t seed = 0; seed < 128 && !observed_correction; ++seed) {
    const auto result = omlx::m5::verify_speculative(
        std::span<const omlx::m5::SelectorStep>(&stochastic_proposal, 1),
        stochastic_target, 1.0F, seed, 0);
    observed_correction = result.used_correction;
  }
  expect(observed_correction, "sampling correction path was never exercised");

  // Regression: q_token == 0 with p_token == 0 MUST reject (never accept a zero-p token).
  const omlx::m5::SelectorStep zero_q_zero_p_proposal{
      .token = 3,
      .candidate_ids = {0, 1},
      .probabilities = {0.4F, 0.6F},
  };
  const std::vector<omlx::m5::TargetDistribution> zero_q_target = {
      {.probabilities = {0.4F, 0.6F, 0.0F, 0.0F}},
      {.probabilities = {0.0F, 0.0F, 1.0F, 0.0F}},
  };
  for (std::uint64_t seed = 0; seed < 32; ++seed) {
    const auto rejected_zero = omlx::m5::verify_speculative(
        std::span<const omlx::m5::SelectorStep>(&zero_q_zero_p_proposal, 1),
        zero_q_target, 1.0F, seed, 0);
    expect(rejected_zero.accepted_proposals == 0 &&
               rejected_zero.used_correction &&
               (rejected_zero.licensed_outputs.front() == 0 ||
                rejected_zero.licensed_outputs.front() == 1),
           "q_token==0 with p_token==0 must reject and fall back to p on zero-mass residual");
  }

  // Regression: q_token == 0 with p_token > 0 MUST accept with probability 1.
  const omlx::m5::SelectorStep zero_q_pos_p_proposal{
      .token = 1,
      .candidate_ids = {0, 2},
      .probabilities = {0.5F, 0.5F},
  };
  for (std::uint64_t seed = 0; seed < 16; ++seed) {
    const auto accepted_zero_q = omlx::m5::verify_speculative(
        std::span<const omlx::m5::SelectorStep>(&zero_q_pos_p_proposal, 1),
        zero_q_target, 1.0F, seed, 0);
    expect(accepted_zero_q.accepted_proposals == 1 &&
               !accepted_zero_q.used_correction &&
               accepted_zero_q.licensed_outputs.front() == 1,
           "q_token==0 with p_token>0 must accept");
  }

  // Regression: float_to_bf16 preserves NaN as canonical quiet BF16 NaN (0x7FC0).
  for (const std::uint32_t raw_nan :
       {0x7FC00000U, 0x7FFFFFFFU, 0xFFC00001U, 0xFFFFFFFFU, 0x7F800001U}) {
    const float nan_val = std::bit_cast<float>(raw_nan);
    const auto bf16_nan = omlx::m5::float_to_bf16(nan_val);
    expect(bf16_nan == 0x7FC0U && std::isnan(omlx::m5::bf16_to_float(bf16_nan)),
           "float_to_bf16 failed to preserve NaN as 0x7FC0");
  }
}

void test_frontend_preview() {
  const std::vector<std::int32_t> licensed = {1, 7, 8, 9};
  const auto stopped =
      omlx::m5::preview_licensed_outputs(licensed, 8, {}, {{7, 8}}, false);
  expect(stopped.tokens == std::vector<std::int32_t>({1}),
         "stop sequence was not removed from published output");
  expect(stopped.commit_length == 1, "stop commit length mismatch");
  expect(stopped.finish_reason == omlx::m5::PreviewFinishReason::stop,
         "stop finish reason mismatch");

  const auto eos = omlx::m5::preview_licensed_outputs(
      licensed, 8, std::vector<std::int32_t>({8}), {}, false);
  expect(eos.tokens == std::vector<std::int32_t>({1, 7}),
         "EOS preview prefix mismatch");
  expect(eos.finish_reason == omlx::m5::PreviewFinishReason::eos,
         "EOS finish reason mismatch");

  const auto length =
      omlx::m5::preview_licensed_outputs(licensed, 2, {}, {}, false);
  expect(length.tokens.size() == 2, "budget was not enforced");
  expect(length.finish_reason == omlx::m5::PreviewFinishReason::length,
         "length finish reason mismatch");

  const std::vector<std::int32_t> exact_budget = {1, 7};
  const auto exact_length = omlx::m5::preview_licensed_outputs(
      exact_budget, exact_budget.size(), {}, {}, false);
  expect(exact_length.tokens == exact_budget,
         "exact budget discarded licensed output");
  expect(exact_length.finish_reason == omlx::m5::PreviewFinishReason::length,
         "exact budget did not terminate with length");

  const auto cancelled =
      omlx::m5::preview_licensed_outputs(licensed, 8, {}, {}, true);
  expect(cancelled.commit_length == 0, "cancelled preview committed tokens");
  expect(cancelled.finish_reason == omlx::m5::PreviewFinishReason::cancelled,
         "cancel finish reason mismatch");
}

} // namespace

int main() {
  try {
    test_replay_closed_loop();
    test_round_transaction();
    test_dynamic_convolution_and_context();
    test_selector_and_verification();
    test_frontend_preview();
  } catch (const std::exception &error) {
    std::cerr << "D3 state/spec test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "D3 state/spec tests passed\n";
  return 0;
}
