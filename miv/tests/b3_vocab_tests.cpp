#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/forward_plan.h"
#include "omlx_m5/qmv.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/vocab.h"

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <class Function>
void expect_throws(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error(message);
}

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

std::vector<std::uint16_t> read_words(const omlx::m5::MetalBuffer &buffer,
                                      std::uint64_t count) {
  const auto raw = buffer.read(0, count * 2);
  std::vector<std::uint16_t> result(count);
  std::memcpy(result.data(), raw.data(), raw.size());
  return result;
}

struct Fixture {
  std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> buffers;
  omlx::m5::MaterializedWeightView view;
  std::vector<float> values;
};

void add_plane(Fixture &fixture, omlx::m5::MetalContext &context,
               std::string kind, std::vector<std::byte> data,
               std::uint64_t stride) {
  fixture.buffers.push_back(
      std::make_unique<omlx::m5::MetalBuffer>(context.upload(data)));
  fixture.view.planes.push_back({
      .kind = std::move(kind),
      .buffer = fixture.buffers.back().get(),
      .bytes = data.size(),
      .row_stride = stride,
  });
}

Fixture direct_fixture(omlx::m5::MetalContext &context) {
  constexpr std::uint32_t rows = 3;
  constexpr std::uint32_t width = 128;
  Fixture fixture;
  fixture.view = {
      .name = "embedding.direct",
      .format = "direct_bf16_v1",
      .layout = "contiguous_le_v1",
      .parent_shape = {100, width},
      .row_start = 10,
      .row_count = rows,
      .in_features = width,
  };
  std::vector<std::uint16_t> words(rows * width);
  fixture.values.resize(words.size());
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t column = 0; column < width; ++column) {
      words[row * width + column] = omlx::m5::float_to_bf16(
          float(static_cast<std::int32_t>((row * 17 + column * 5) % 97) - 48) /
          128.0F);
      fixture.values[row * width + column] =
          omlx::m5::bf16_to_float(words[row * width + column]);
    }
  }
  add_plane(fixture, context, "data", bytes(words), width * 2);
  return fixture;
}

Fixture q4_fixture(omlx::m5::MetalContext &context) {
  constexpr std::uint32_t rows = 3;
  constexpr std::uint32_t width = 128;
  Fixture fixture;
  fixture.view = {
      .name = "embedding.q4",
      .format = "oq_affine_q4_g64_bf16_v1",
      .layout = "oq_rowsplit_k128_v1",
      .parent_shape = {100, width},
      .padded_k = width,
      .row_start = 10,
      .row_count = rows,
      .in_features = width,
  };
  std::vector<std::uint8_t> codes(rows * width / 2);
  std::vector<std::uint16_t> scales(rows * 2);
  std::vector<std::uint16_t> biases(rows * 2);
  fixture.values.resize(rows * width);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t group = 0; group < 2; ++group) {
      scales[row * 2 + group] =
          omlx::m5::float_to_bf16(0.015625F * float(row + group + 1));
      biases[row * 2 + group] =
          omlx::m5::float_to_bf16(-0.03125F * float(row + 1));
    }
    for (std::uint32_t column = 0; column < width; ++column) {
      const auto code = (row * 7U + column * 5U) & 15U;
      codes[row * (width / 2) + column / 2] |=
          static_cast<std::uint8_t>(code << ((column & 1U) * 4U));
      fixture.values[row * width + column] = std::fma(
          omlx::m5::bf16_to_float(scales[row * 2 + column / 64]), float(code),
          omlx::m5::bf16_to_float(biases[row * 2 + column / 64]));
    }
  }
  add_plane(fixture, context, "base_code", bytes(codes), width / 2);
  add_plane(fixture, context, "scale", bytes(scales), 4);
  add_plane(fixture, context, "bias", bytes(biases), 4);
  return fixture;
}

void check_embedding(omlx::m5::MetalContext &context, Fixture &fixture) {
  const std::vector<std::uint32_t> ids = {10, 12};
  auto token_ids = context.upload(bytes(ids));
  auto plan = omlx::m5::prepare_embedding(context, OMLX_M5_VOCAB_METALLIB_PATH,
                                          fixture.view, ids.size());
  auto output = context.make_buffer(plan.info().output_bytes);
  const auto submission =
      omlx::m5::execute_embedding(context, plan, token_ids, 0, output, 0);
  expect(submission.dispatch_count == 1, "embedding dispatch count is wrong");
  const auto words = read_words(output, ids.size() * 128);
  for (std::size_t token = 0; token < ids.size(); ++token) {
    const auto local_row = ids[token] - 10;
    for (std::uint32_t feature = 0; feature < 128; ++feature) {
      const auto actual = omlx::m5::bf16_to_float(words[token * 128 + feature]);
      const auto expected = fixture.values[local_row * 128 + feature];
      expect(std::abs(actual - expected) <=
                 0.005F * std::max(1.0F, std::abs(expected)),
             "embedding lookup differs from reference");
    }
  }
}

void test_direct_qmv(omlx::m5::MetalContext &context, Fixture &fixture) {
  std::vector<std::uint16_t> input_words(128);
  for (std::uint32_t index = 0; index < 128; ++index) {
    input_words[index] = omlx::m5::float_to_bf16(float(index % 17) / 64.0F);
  }
  auto input = context.upload(bytes(input_words));
  auto plan = omlx::m5::prepare_dense_qmv_t1(
      context, OMLX_M5_QMV_T1_METALLIB_PATH, fixture.view);
  auto output = context.make_buffer(plan.info().output_bytes);
  static_cast<void>(
      omlx::m5::execute_dense_qmv_t1(context, plan, input, 0, output, 0));
  const auto words = read_words(output, 3);
  for (std::uint32_t row = 0; row < 3; ++row) {
    double expected = 0.0;
    for (std::uint32_t column = 0; column < 128; ++column) {
      expected += omlx::m5::bf16_to_float(input_words[column]) *
                  fixture.values[row * 128 + column];
    }
    const auto actual = omlx::m5::bf16_to_float(words[row]);
    expect(std::abs(actual - expected) <=
               0.02 * std::max(1.0, std::abs(expected)),
           "direct BF16 QMV differs from reference");
  }
}

std::uint32_t read_token(const omlx::m5::MetalBuffer &buffer) {
  const auto raw = buffer.read(0, 4);
  std::uint32_t token = 0;
  std::memcpy(&token, raw.data(), sizeof(token));
  return token;
}

void test_sampling(omlx::m5::MetalContext &context) {
  constexpr std::uint32_t vocabulary = 1024;
  std::vector<std::uint16_t> words(vocabulary);
  for (std::uint32_t token = 0; token < vocabulary; ++token) {
    words[token] = omlx::m5::float_to_bf16(
        float(static_cast<std::int32_t>((token * 37U) % 211U) - 105) / 16.0F);
  }
  words[777] = omlx::m5::float_to_bf16(20.0F);
  auto logits = context.upload(bytes(words));
  auto output = context.make_buffer(4);
  auto plan = omlx::m5::prepare_sampling(context, OMLX_M5_VOCAB_METALLIB_PATH,
                                         vocabulary);
  static_cast<void>(
      omlx::m5::execute_sampling(context, plan, logits, 0, {}, output, 0));
  expect(read_token(output) == 777, "GPU greedy sampling selected wrong token");
  const omlx::m5::GpuSamplingOptions sampled{
      .temperature = 0.8F,
      .top_k = 8,
      .top_p = 0.9F,
      .seed = 12345,
      .stream = 7,
      .absolute_position = 19,
  };
  static_cast<void>(
      omlx::m5::execute_sampling(context, plan, logits, 0, sampled, output, 0));
  const auto first = read_token(output);
  static_cast<void>(
      omlx::m5::execute_sampling(context, plan, logits, 0, sampled, output, 0));
  expect(read_token(output) == first,
         "GPU sampled token is not counter deterministic");
  auto invalid = sampled;
  invalid.top_k = 65;
  expect_throws(
      [&] {
        static_cast<void>(omlx::m5::execute_sampling(context, plan, logits, 0,
                                                     invalid, output, 0));
      },
      "sampling accepted top_k above its fixed bound");

  // MIVK-7: NaN and all -Inf logits must yield out-of-range sentinel tokens.
  auto nan_words = words;
  nan_words[13] = 0x7FC0U;
  auto nan_logits = context.upload(bytes(nan_words));
  static_cast<void>(
      omlx::m5::execute_sampling(context, plan, nan_logits, 0, {}, output, 0));
  expect(read_token(output) == 0xffffffffU,
         "greedy sampling ignored NaN logit");
  static_cast<void>(omlx::m5::execute_sampling(context, plan, nan_logits, 0,
                                               sampled, output, 0));
  expect(read_token(output) == 0xffffffffU,
         "stochastic sampling ignored NaN logit");

  const std::vector<std::uint16_t> neg_inf_words(vocabulary, 0xFF80U);
  auto neg_inf_logits = context.upload(bytes(neg_inf_words));
  static_cast<void>(omlx::m5::execute_sampling(context, plan, neg_inf_logits, 0,
                                               {}, output, 0));
  expect(read_token(output) >= vocabulary,
         "greedy sampling failed to flag all -Inf logits");
  static_cast<void>(omlx::m5::execute_sampling(context, plan, neg_inf_logits, 0,
                                               sampled, output, 0));
  expect(read_token(output) >= vocabulary,
         "stochastic sampling failed to flag all -Inf logits");

  auto masked_words = neg_inf_words;
  masked_words[5] = omlx::m5::float_to_bf16(2.0F);
  masked_words[99] = omlx::m5::float_to_bf16(1.0F);
  auto masked_logits = context.upload(bytes(masked_words));
  auto masked_opts = sampled;
  masked_opts.top_p = 1.0F;
  static_cast<void>(omlx::m5::execute_sampling(context, plan, masked_logits, 0,
                                               masked_opts, output, 0));
  const auto masked_tok = read_token(output);
  expect(masked_tok == 5U || masked_tok == 99U,
         "stochastic sampling selected -Inf sentinel over valid finite logits");

  // MIVK-7: 2-stage 64-TG greedy argmax + feed_stage2 NaN & all -Inf guard.
  const auto libraries =
      std::filesystem::path(OMLX_M5_VOCAB_METALLIB_PATH).parent_path();
  constexpr std::uint32_t stage_vocab = 257U;
  std::vector<std::uint16_t> s_words(stage_vocab,
                                     omlx::m5::float_to_bf16(-1.0F));
  s_words[200] = omlx::m5::float_to_bf16(5.0F);
  auto s_logits = context.upload(bytes(s_words));
  auto s_parts = context.make_buffer(64ULL * 8ULL);
  auto s_out = context.make_buffer(4);
  auto s_next = context.make_buffer(4);
  struct StageSamplingParams {
    std::uint32_t vocabulary;
    std::uint32_t top_k;
    float temperature;
    float top_p;
    std::uint64_t seed;
    std::uint64_t stream;
    std::uint64_t absolute_position;
  };
  static_assert(sizeof(StageSamplingParams) == 40);
  const StageSamplingParams sp{
      .vocabulary = stage_vocab,
      .top_k = 1U,
      .temperature = 0.0F,
      .top_p = 1.0F,
      .seed = 0,
      .stream = 0,
      .absolute_position = 0,
  };
  const auto sp_span = std::as_bytes(std::span(&sp, 1));
  const std::vector<std::byte> sp_bytes(sp_span.begin(), sp_span.end());
  omlx::m5::detail::forward::Encoder enc(context, libraries, 1U,
                                         "b3.vocab.stage2.nan");
  auto &st1 = enc.emit("vocab", "omlx_m5_vocab_argmax_stage1_64tg",
                       {.width = 64ULL * 256ULL}, {.width = 256ULL},
                       {{&s_logits, 0, s_logits.size()},
                        {&s_parts, 0, s_parts.size()}});
  enc.dynamic(st1, 2, omlx::m5::detail::forward::words({stage_vocab}));
  auto &st2 = enc.emit("vocab", "omlx_m5_sample_logits_feed_stage2",
                       {.width = 256ULL}, {.width = 256ULL},
                       {{&s_logits, 0, s_logits.size()},
                        {&s_parts, 0, s_parts.size()},
                        {&s_out, 0, s_out.size()},
                        {&s_next, 0, s_next.size()}});
  enc.dynamic(st2, 4, sp_bytes);
  auto prof = enc.finish();
  const std::array call{omlx::m5::MetalCommandInvocation{
      .plan = &prof.plan,
      .bindings = prof.bindings,
      .parameters = &prof.parameters}};
  static_cast<void>(context.execute_batch(call));
  expect(read_token(s_out) == 200U && read_token(s_next) == 200U,
         "2-stage vocab argmax selected wrong clean token");

  s_words[17] = 0x7FC0U;
  s_logits.write(bytes(s_words));
  static_cast<void>(context.execute_batch(call));
  expect(read_token(s_out) == 0xffffffffU && read_token(s_next) == 0xffffffffU,
         "2-stage vocab argmax ignored NaN logit");

  std::fill(s_words.begin(), s_words.end(), static_cast<std::uint16_t>(0xFF80U));
  s_logits.write(bytes(s_words));
  static_cast<void>(context.execute_batch(call));
  expect(read_token(s_out) >= stage_vocab && read_token(s_next) >= stage_vocab,
         "2-stage vocab argmax failed to flag all -Inf logits");
}

void test_embedding_validation_and_padded_bf16(omlx::m5::MetalContext &context) {
  // MIVE-17: validate layout, padded_k % group_size == 0, and plane minimum bytes.
  {
    auto bad_layout = direct_fixture(context);
    bad_layout.view.layout = "oq_rowsplit_k128_v1";
    expect_throws(
        [&] {
          static_cast<void>(omlx::m5::prepare_embedding(
              context, OMLX_M5_VOCAB_METALLIB_PATH, bad_layout.view, 1));
        },
        "prepare_embedding accepted wrong direct BF16 layout");

    auto bad_q4_layout = q4_fixture(context);
    bad_q4_layout.view.layout = "oq_nax_tiled_g64_v1";
    expect_throws(
        [&] {
          static_cast<void>(omlx::m5::prepare_embedding(
              context, OMLX_M5_VOCAB_METALLIB_PATH, bad_q4_layout.view, 1));
        },
        "prepare_embedding accepted wrong affine layout");

    auto bad_group = q4_fixture(context);
    bad_group.view.in_features = 96;
    bad_group.view.padded_k = 100;
    expect_throws(
        [&] {
          static_cast<void>(omlx::m5::prepare_embedding(
              context, OMLX_M5_VOCAB_METALLIB_PATH, bad_group.view, 1));
        },
        "prepare_embedding accepted padded_k not divisible by group_size");

    auto truncated_direct = direct_fixture(context);
    truncated_direct.view.planes[0].bytes -= 2;
    expect_throws(
        [&] {
          static_cast<void>(omlx::m5::prepare_embedding(
              context, OMLX_M5_VOCAB_METALLIB_PATH, truncated_direct.view, 1));
        },
        "prepare_embedding accepted truncated direct BF16 plane");

    auto truncated_q4 = q4_fixture(context);
    truncated_q4.view.planes[1].bytes -= 2;
    expect_throws(
        [&] {
          static_cast<void>(omlx::m5::prepare_embedding(
              context, OMLX_M5_VOCAB_METALLIB_PATH, truncated_q4.view, 1));
        },
        "prepare_embedding accepted truncated affine scale plane");
  }

  // MIVK-4: direct_bf16_v1 embedding must honor padded_hidden_size > hidden_size.
  {
    constexpr std::uint32_t rows = 3;
    constexpr std::uint32_t logical_k = 96;
    constexpr std::uint32_t padded_k = 128;
    Fixture padded;
    padded.view = {
        .name = "embedding.direct.padded",
        .format = "direct_bf16_v1",
        .layout = "contiguous_le_v1",
        .parent_shape = {50, logical_k},
        .padded_k = padded_k,
        .row_start = 10,
        .row_count = rows,
        .in_features = logical_k,
    };
    std::vector<std::uint16_t> words(rows * padded_k,
                                     omlx::m5::float_to_bf16(77.0F));
    padded.values.resize(rows * logical_k);
    for (std::uint32_t row = 0; row < rows; ++row) {
      for (std::uint32_t col = 0; col < logical_k; ++col) {
        const float val =
            float(static_cast<std::int32_t>((row * 23 + col * 7) % 89) - 44) /
            64.0F;
        words[row * padded_k + col] = omlx::m5::float_to_bf16(val);
        padded.values[row * logical_k + col] =
            omlx::m5::bf16_to_float(words[row * padded_k + col]);
      }
    }
    add_plane(padded, context, "data", bytes(words), padded_k * 2);
    const std::vector<std::uint32_t> ids = {10, 11, 12};
    auto token_ids = context.upload(bytes(ids));
    auto plan = omlx::m5::prepare_embedding(
        context, OMLX_M5_VOCAB_METALLIB_PATH, padded.view, ids.size());
    auto output = context.make_buffer(plan.info().output_bytes);
    static_cast<void>(
        omlx::m5::execute_embedding(context, plan, token_ids, 0, output, 0));
    const auto out_words = read_words(output, ids.size() * logical_k);
    for (std::size_t token = 0; token < ids.size(); ++token) {
      const auto local_row = ids[token] - 10;
      for (std::uint32_t col = 0; col < logical_k; ++col) {
        const auto actual =
            omlx::m5::bf16_to_float(out_words[token * logical_k + col]);
        const auto expected = padded.values[local_row * logical_k + col];
        expect(actual == expected,
               "padded BF16 embedding lookup ignored padded_hidden_size");
      }
    }
  }
}

} // namespace

int main() {
  try {
    auto context = omlx::m5::MetalContext::create();
    auto direct = direct_fixture(context);
    auto q4 = q4_fixture(context);
    check_embedding(context, direct);
    check_embedding(context, q4);
    test_direct_qmv(context, direct);
    test_sampling(context);
    test_embedding_validation_and_padded_bf16(context);
    std::cout << "B3.8 embedding/lm-head/sampling tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B3.8 vocabulary failure: " << error.what() << '\n';
    return 1;
  }
}
