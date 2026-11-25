#include "../src/forward_plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "omlx_m5/quantization.h"

using namespace omlx::m5;
using namespace omlx::m5::detail::forward;

namespace {
struct SelectorParams {
  U32 vocab, hidden, rank, top_k, blocks, groups, width, anchor, position;
  float temperature;
  float uniforms[7];
};
static_assert(sizeof(SelectorParams) == 68);

void check_ties(const std::filesystem::path &libraries, U32 vocabulary) {
  constexpr U32 rows = 8, top_k = 8;
  const U32 blocks = (vocabulary + 255) / 256;
  const U32 groups = (blocks + 31) / 32;
  auto context = MetalContext::create();
  std::vector<std::uint16_t> logits(U64(rows) * vocabulary);
  for (U32 row = 0; row < rows; ++row) {
    for (U32 token = 0; token < vocabulary; ++token) {
      float value = 0.0F;
      if (row == 0) {
        // Anchor row (intentionally skipped by omlx_m5_selector_topk_*).
        value = 9.0F;
      } else if (row == 1) {
        // All-equal ties: lowest token IDs must win deterministically.
        value = 0.0F;
      } else if (row == 2 && token + 1 == vocabulary) {
        // Partial-tail block wins rank 0 with a distinct score.
        value = 4.0F;
      } else if (row == 2 && token + 2 == vocabulary) {
        value = 3.5F;
      } else if ((token + row * 7U) % 29U == 0U) {
        // Distinct tied peak positions and BF16-exact scores per row.
        value = 2.0F + 0.25F * float(row);
      } else {
        value = float(int((token + row * 5U) % 17U) - 8) / 8.0F;
      }
      logits[U64(row) * vocabulary + token] = float_to_bf16(value);
    }
  }
  auto input = context.upload(std::as_bytes(std::span(logits)));
  auto partial_ids = context.make_buffer(U64(rows) * blocks * top_k * 4);
  auto partial_scores = context.make_buffer(partial_ids.size());
  auto group_ids = context.make_buffer(U64(rows) * groups * top_k * 4);
  auto group_scores = context.make_buffer(group_ids.size());
  constexpr U32 kSentinelId = 0xDEADBEEFU;
  constexpr float kSentinelScore = -1234.5F;
  std::vector<U32> init_ids(U64(rows) * top_k, kSentinelId);
  std::vector<float> init_scores(U64(rows) * top_k, kSentinelScore);
  auto ids = context.upload(std::as_bytes(std::span(init_ids)));
  auto scores = context.upload(std::as_bytes(std::span(init_scores)));
  const SelectorParams p{.vocab = vocabulary,
                         .top_k = top_k,
                         .blocks = blocks,
                         .groups = groups,
                         .width = rows};
  const auto raw = std::as_bytes(std::span(&p, 1));
  const std::vector<std::byte> params(raw.begin(), raw.end());
  auto ref = [](const MetalBuffer &b) -> Ref { return {&b, 0, b.size()}; };
  Encoder encoder(context, libraries, 8,
                  "p03.topk.tie." + std::to_string(vocabulary));
  auto &chunks =
      encoder.emit("dflash_forward", "omlx_m5_selector_topk_chunks",
                   {.width = blocks, .height = rows}, {.width = 128},
                   {ref(input), ref(partial_ids), ref(partial_scores)});
  encoder.dynamic(chunks, 3, params);
  auto &reduce =
      encoder.emit("dflash_forward", "omlx_m5_selector_topk_reduce32",
                   {.width = groups, .height = rows}, {.width = 128},
                   {ref(partial_ids), ref(partial_scores), ref(group_ids),
                    ref(group_scores)});
  encoder.dynamic(reduce, 4, params);
  auto &merge =
      encoder.emit("dflash_forward", "omlx_m5_selector_topk_merge",
                   {.width = rows}, {.width = 32},
                   {ref(group_ids), ref(group_scores), ref(ids), ref(scores)});
  encoder.dynamic(merge, 4, params);
  auto profile = encoder.finish();
  const std::array call = {
      MetalCommandInvocation{.plan = &profile.plan,
                             .bindings = profile.bindings,
                             .parameters = &profile.parameters}};
  const auto submission = context.execute_batch(call);
  if (submission.dispatch_count != 3)
    throw std::runtime_error("P03 top-k dispatch count changed");
  // Verify row 0 (anchor slot) is intentionally untouched by the selector top-k kernels.
  {
    const auto row0_id_bytes = ids.read(0, top_k * 4);
    const auto row0_score_bytes = scores.read(0, top_k * 4);
    std::array<U32, top_k> row0_ids{};
    std::array<float, top_k> row0_scores{};
    std::memcpy(row0_ids.data(), row0_id_bytes.data(), row0_id_bytes.size());
    std::memcpy(row0_scores.data(), row0_score_bytes.data(),
                row0_score_bytes.size());
    for (U32 i = 0; i < top_k; ++i) {
      if (row0_ids[i] != kSentinelId || row0_scores[i] != kSentinelScore) {
        throw std::runtime_error(
            "P03 top-k unexpectedly modified anchor row 0");
      }
    }
  }
  for (U32 row = 1; row < rows; ++row) {
    auto id_bytes = ids.read(U64(row) * top_k * 4, top_k * 4);
    auto score_bytes = scores.read(U64(row) * top_k * 4, top_k * 4);
    std::array<U32, top_k> observed_ids{};
    std::array<float, top_k> observed_scores{};
    std::memcpy(observed_ids.data(), id_bytes.data(), id_bytes.size());
    std::memcpy(observed_scores.data(), score_bytes.data(), score_bytes.size());
    std::vector<U32> expected(vocabulary);
    std::iota(expected.begin(), expected.end(), 0);
    const auto score = [&](U32 token) {
      const U32 bits = U32(logits[U64(row) * vocabulary + token]) << 16;
      return std::bit_cast<float>(bits);
    };
    std::partial_sort(expected.begin(), expected.begin() + top_k,
                      expected.end(), [&](U32 a, U32 b) {
                        return score(a) > score(b) ||
                               (score(a) == score(b) && a < b);
                      });
    if (!std::equal(observed_ids.begin(), observed_ids.end(), expected.begin()))
      throw std::runtime_error("P03 tied/tail top-k IDs differ from CPU");
    for (U32 i = 0; i < top_k; ++i) {
      const float expected_score = score(expected[i]);
      if (!std::isfinite(observed_scores[i]) ||
          observed_scores[i] != expected_score) {
        throw std::runtime_error("P03 tied/tail top-k scores differ from CPU");
      }
    }
  }
}
struct SelectorPositionKey {
  U64 seed, position, purpose;
};
static_assert(sizeof(SelectorPositionKey) == 24);

void check_topk16_unaligned(const std::filesystem::path &libraries,
                            U32 vocabulary) {
  constexpr U32 rows = 8, top_k = 16, batch = 2;
  auto context = MetalContext::create();
  std::vector<std::uint16_t> logits(U64(batch) * rows * vocabulary);
  for (U32 b = 0; b < batch; ++b) {
    for (U32 row = 0; row < rows; ++row) {
      for (U32 token = 0; token < vocabulary; ++token) {
        float value = float(int((token * 13U + row * 7U + b * 11U) % 41U) - 20) / 8.0F;
        if (token + 1 == vocabulary) value = 5.0F + float(row) * 0.25F;
        if (token == (row + b * 3U) % vocabulary) value = 6.0F;
        logits[(U64(b) * rows + row) * vocabulary + token] = float_to_bf16(value);
      }
    }
  }
  auto input = context.upload(std::as_bytes(std::span(logits)));
  auto partial_ids = context.make_buffer(U64(batch) * rows * top_k * 64ULL * 4ULL);
  auto partial_scores = context.make_buffer(partial_ids.size());
  auto out_ids = context.make_buffer(U64(batch) * rows * top_k * 4ULL);
  auto out_scores = context.make_buffer(out_ids.size());

  const SelectorParams p{.vocab = vocabulary, .top_k = top_k, .width = rows};
  const auto p_raw = std::as_bytes(std::span(&p, 1));
  const std::vector<std::byte> p_bytes(p_raw.begin(), p_raw.end());
  auto ref = [](const MetalBuffer &b) -> Ref { return {&b, 0, b.size()}; };

  // Single-slot topk16 stage1 + stage2 (vocab % 4 != 0 -> unaligned rows 1..7)
  {
    Encoder enc(context, libraries, 8, "p03.topk16.unaligned." + std::to_string(vocabulary));
    auto &s1 = enc.emit("dflash_forward", "omlx_m5_selector_topk16_stage1",
                        {.width = 64ULL * 256ULL, .height = rows - 1U}, {.width = 256ULL},
                        {ref(input), ref(partial_ids), ref(partial_scores)});
    enc.dynamic(s1, 3, p_bytes);
    auto &s2 = enc.emit("dflash_forward", "omlx_m5_selector_topk16_stage2",
                        {.width = 64ULL, .height = rows - 1U}, {.width = 64ULL},
                        {ref(partial_ids), ref(partial_scores), ref(out_ids), ref(out_scores)});
    enc.dynamic(s2, 4, p_bytes);
    auto prof = enc.finish();
    const std::array call{MetalCommandInvocation{
        .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
    static_cast<void>(context.execute_batch(call));
    for (U32 row = 1; row < rows; ++row) {
      const auto id_bytes = out_ids.read(U64(row) * top_k * 4ULL, top_k * 4ULL);
      std::array<U32, top_k> got_ids{};
      std::memcpy(got_ids.data(), id_bytes.data(), id_bytes.size());
      std::vector<U32> expected(vocabulary);
      std::iota(expected.begin(), expected.end(), 0U);
      const auto score = [&](U32 tok) {
        return std::bit_cast<float>(U32(logits[U64(row) * vocabulary + tok]) << 16U);
      };
      std::partial_sort(expected.begin(), expected.begin() + top_k, expected.end(),
                        [&](U32 a, U32 b_tok) {
                          return score(a) > score(b_tok) ||
                                 (score(a) == score(b_tok) && a < b_tok);
                        });
      if (!std::equal(got_ids.begin(), got_ids.end(), expected.begin())) {
        throw std::runtime_error("P03 topk16 unaligned vocab IDs mismatch");
      }
    }
  }

  // Batched topk16 stage1 + stage2 across 2 slots (rows 1..7 and 9..15)
  {
    const std::array<SelectorParams, batch> ps{p, p};
    const auto ps_raw = std::as_bytes(std::span(ps));
    const std::vector<std::byte> ps_bytes(ps_raw.begin(), ps_raw.end());
    Encoder enc(context, libraries, 8, "p03.topk16.batched.unaligned." + std::to_string(vocabulary));
    auto &s1 = enc.emit("dflash_forward", "omlx_m5_selector_topk16_stage1_batched",
                        {.width = 64ULL * 256ULL, .height = U64(batch) * 7ULL}, {.width = 256ULL},
                        {ref(input), ref(partial_ids), ref(partial_scores)});
    enc.dynamic(s1, 3, ps_bytes);
    auto &s2 = enc.emit("dflash_forward", "omlx_m5_selector_topk16_stage2_batched",
                        {.width = 64ULL, .height = U64(batch) * 7ULL}, {.width = 64ULL},
                        {ref(partial_ids), ref(partial_scores), ref(out_ids), ref(out_scores)});
    enc.dynamic(s2, 4, ps_bytes);
    auto prof = enc.finish();
    const std::array call{MetalCommandInvocation{
        .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
    static_cast<void>(context.execute_batch(call));
    for (U32 b = 0; b < batch; ++b) {
      for (U32 row = 1; row < rows; ++row) {
        const U32 global_row = b * rows + row;
        const auto id_bytes = out_ids.read(U64(global_row) * top_k * 4ULL, top_k * 4ULL);
        std::array<U32, top_k> got_ids{};
        std::memcpy(got_ids.data(), id_bytes.data(), id_bytes.size());
        std::vector<U32> expected(vocabulary);
        std::iota(expected.begin(), expected.end(), 0U);
        const auto score = [&](U32 tok) {
          return std::bit_cast<float>(
              U32(logits[U64(global_row) * vocabulary + tok]) << 16U);
        };
        std::partial_sort(expected.begin(), expected.begin() + top_k, expected.end(),
                          [&](U32 a, U32 b_tok) {
                            return score(a) > score(b_tok) ||
                                   (score(a) == score(b_tok) && a < b_tok);
                          });
        if (!std::equal(got_ids.begin(), got_ids.end(), expected.begin())) {
          throw std::runtime_error("P03 topk16 batched unaligned vocab IDs mismatch");
        }
      }
    }
  }
}

void check_finish_sentinels(const std::filesystem::path &libraries) {
  // MIVK-2: vocab < top_k (sentinel 0xffffffffu candidates) and invalid anchor
  // must never fault or read out of bounds in selector_finish* kernels.
  constexpr U32 vocab = 4, top_k = 16, rank = 128, width = 4, slots = 2;
  auto context = MetalContext::create();
  std::vector<std::uint16_t> pred_data(U64(vocab) * rank, float_to_bf16(0.25F));
  std::vector<std::uint16_t> succ_data(U64(vocab) * rank, float_to_bf16(0.5F));
  std::vector<float> projected_data(U64(slots) * 8ULL * rank, 0.125F);
  std::vector<U32> ids_data(U64(slots) * 8ULL * top_k, 0xffffffffU);
  std::vector<float> unary_data(U64(slots) * 8ULL * top_k,
                                -std::numeric_limits<float>::infinity());
  // Slot 0: partial valid candidates (vocab=4 < top_k=16); Slot 1: all sentinels.
  for (U32 r = 1; r < width; ++r) {
    for (U32 c = 0; c < vocab; ++c) {
      ids_data[U64(r) * top_k + c] = c;
      unary_data[U64(r) * top_k + c] = float(c) * 0.5F;
    }
  }
  auto pred = context.upload(std::as_bytes(std::span(pred_data)));
  auto succ = context.upload(std::as_bytes(std::span(succ_data)));
  auto proj = context.upload(std::as_bytes(std::span(projected_data)));
  auto ids = context.upload(std::as_bytes(std::span(ids_data)));
  auto unary = context.upload(std::as_bytes(std::span(unary_data)));
  auto selected = context.make_buffer(U64(slots) * 8ULL * 4ULL);
  auto probs = context.make_buffer(U64(slots) * 8ULL * top_k * 4ULL);
  auto fallback = context.make_buffer(U64(slots) * 4ULL);
  auto ref = [](const MetalBuffer &b) -> Ref { return {&b, 0, b.size()}; };

  const auto run_variant = [&](bool position, bool batched, float temperature,
                               U32 anchor_slot0) {
    SelectorParams p0{.vocab = vocab,
                      .hidden = 0,
                      .rank = rank,
                      .top_k = top_k,
                      .width = width,
                      .anchor = anchor_slot0,
                      .temperature = temperature};
    for (auto &u : p0.uniforms) u = 0.999999F;
    SelectorParams p1 = p0;
    p1.anchor = 0xffffffffU;
    std::vector<SelectorParams> ps;
    ps.push_back(p0);
    if (batched) ps.push_back(p1);
    const auto p_raw = std::as_bytes(std::span(ps));
    const std::vector<std::byte> p_bytes(p_raw.begin(), p_raw.end());

    const std::array<SelectorPositionKey, slots> keys{
        SelectorPositionKey{1234ULL, 10ULL, 1ULL},
        SelectorPositionKey{5678ULL, 20ULL, 1ULL}};
    const auto k_raw = std::as_bytes(std::span(keys.data(), batched ? slots : 1U));
    const std::vector<std::byte> k_bytes(k_raw.begin(), k_raw.end());

    const std::string fn =
        position ? (batched ? "omlx_m5_selector_finish_position_batched"
                            : "omlx_m5_selector_finish_position")
                 : (batched ? "omlx_m5_selector_finish_batched"
                            : "omlx_m5_selector_finish");
    Encoder enc(context, libraries, width,
                "p03.sentinel." + fn + "." + std::to_string(int(temperature * 10.0F)) +
                    "." + std::to_string(anchor_slot0));
    auto &d = enc.emit("dflash_forward", fn,
                       {.width = 1024ULL, .height = batched ? slots : 1ULL},
                       {.width = 1024ULL},
                       {ref(ids), ref(unary), ref(proj), ref(pred), ref(succ),
                        ref(selected), ref(probs), ref(fallback)});
    enc.dynamic(d, 8, p_bytes);
    if (position) {
      enc.dynamic(d, 9, k_bytes);
    }
    auto prof = enc.finish();
    const std::array call{MetalCommandInvocation{
        .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
    static_cast<void>(context.execute_batch(call));

    const auto sel_raw = selected.read(0, U64(batched ? slots : 1U) * 8ULL * 4ULL);
    const auto fb_raw = fallback.read(0, U64(batched ? slots : 1U) * 4ULL);
    std::vector<U32> sel(sel_raw.size() / 4ULL);
    std::vector<U32> fb(fb_raw.size() / 4ULL);
    std::memcpy(sel.data(), sel_raw.data(), sel_raw.size());
    std::memcpy(fb.data(), fb_raw.data(), fb_raw.size());
    for (U32 r = 1; r < width; ++r) {
      if (sel[r] >= vocab) {
        throw std::runtime_error(fn + " produced out-of-vocab token on slot 0");
      }
    }
    if (anchor_slot0 >= vocab && fb[0] != 1U) {
      throw std::runtime_error(fn + " failed to set fallback on invalid anchor");
    }
    if (batched) {
      if (fb[1] != 1U) {
        throw std::runtime_error(fn + " failed to set fallback on all-sentinel slot");
      }
      for (U32 r = 1; r < width; ++r) {
        if (sel[8U + r] >= vocab) {
          throw std::runtime_error(fn + " produced out-of-vocab token on slot 1");
        }
      }
    }
  };

  for (bool position : {false, true}) {
    for (bool batched : {false, true}) {
      for (float temp : {0.0F, 0.8F}) {
        run_variant(position, batched, temp, 0U);
        run_variant(position, batched, temp, 0xffffffffU);
      }
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("p03_selector_tests LIBRARIES");
    check_ties(argv[1], 257);
    check_ties(argv[1], 8193);
    check_topk16_unaligned(argv[1], 257);
    check_finish_sentinels(argv[1]);
    std::cout << "P03 tied, partial-tail, unaligned topk16, and sentinel finish tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
