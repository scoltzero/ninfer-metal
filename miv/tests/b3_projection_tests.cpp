#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "omlx_m5/projection.h"
#include "omlx_m5/qmv.h"
#include "omlx_m5/quantization.h"

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

omlx::m5::QwenModelConfig config() {
  omlx::m5::QwenModelConfig value;
  value.hidden_size = 128;
  value.num_attention_heads = 2;
  value.num_key_value_heads = 1;
  value.head_dim = 4;
  value.linear_num_key_heads = 1;
  value.linear_num_value_heads = 2;
  value.linear_key_head_dim = 4;
  value.linear_value_head_dim = 4;
  return value;
}

void test_fixed_bundle_geometry() {
  const auto selected_config = config();
  const auto attention = omlx::m5::make_full_attention_projection_bundle(
      selected_config, "layer.self_attn");
  expect(attention.parents.size() == 3 &&
             attention.parents[0].output_features == 16 &&
             attention.parents[0].outputs[0].feature_count == 8 &&
             attention.parents[0].outputs[0].block_count == 2 &&
             attention.parents[0].outputs[0].block_features == 4 &&
             attention.parents[0].outputs[0].block_stride_features == 8 &&
             attention.parents[0].outputs[1].feature_offset == 4 &&
             attention.parents[1].output_features == 4 &&
             attention.parents[2].output_features == 4,
         "full-attention projection geometry is wrong");
  const auto gdn = omlx::m5::make_gdn_projection_bundle(selected_config,
                                                        "layer.linear_attn");
  expect(gdn.parents.size() == 2 && gdn.parents[0].output_features == 16 &&
             gdn.parents[0].outputs[0].feature_count == 4 &&
             gdn.parents[0].outputs[1].feature_offset == 4 &&
             gdn.parents[0].outputs[2].feature_offset == 8 &&
             gdn.parents[0].outputs[2].feature_count == 8 &&
             gdn.parents[1].output_features == 8,
         "GDN projection geometry is wrong");

  auto invalid = attention;
  invalid.parents[0].outputs[1].feature_offset = 3;
  expect_throws([&] { omlx::m5::validate_projection_bundle(invalid); },
                "overlapping split outputs were accepted");
}

void test_zero_copy_strided_views() {
  auto context = omlx::m5::MetalContext::create();
  const auto attention = omlx::m5::make_full_attention_projection_bundle(
      config(), "layer.self_attn");
  omlx::m5::ProjectionOutputArena arena(context, attention, 3);
  const auto parent = arena.parent("attention_q_gate");
  const auto query = arena.output("attention_query");
  const auto gate = arena.output("attention_gate");
  expect(parent.buffer == query.buffer && query.buffer == gate.buffer &&
             query.token_stride_bytes == 32 && gate.token_stride_bytes == 32 &&
             query.block_count == 2 && query.block_features == 4 &&
             query.block_stride_bytes == 16 &&
             gate.offset == query.offset + 8 &&
             arena.stats().split_copy_dispatches == 0,
         "projection split views are not zero-copy strided aliases");

  std::vector<std::uint16_t> values(3 * 16);
  for (std::uint32_t token = 0; token < 3; ++token) {
    for (std::uint32_t feature = 0; feature < 16; ++feature) {
      values[token * 16 + feature] =
          omlx::m5::float_to_bf16(static_cast<float>(token * 100 + feature));
    }
  }
  arena.buffer().write(bytes(values), parent.offset);
  for (std::uint32_t token = 0; token < 3; ++token) {
    for (std::uint32_t feature = 0; feature < 8; ++feature) {
      std::uint16_t query_word = 0;
      std::uint16_t gate_word = 0;
      const auto query_bytes =
          arena.buffer().read(query.feature_offset(token, feature), 2);
      const auto gate_bytes =
          arena.buffer().read(gate.feature_offset(token, feature), 2);
      std::memcpy(&query_word, query_bytes.data(), sizeof(query_word));
      std::memcpy(&gate_word, gate_bytes.data(), sizeof(gate_word));
      const auto head = feature / 4;
      const auto within = feature % 4;
      expect(omlx::m5::bf16_to_float(query_word) ==
                     static_cast<float>(token * 100 + head * 8 + within) &&
                 omlx::m5::bf16_to_float(gate_word) ==
                     static_cast<float>(token * 100 + head * 8 + 4 + within),
             "strided split output points at the wrong feature range");
    }
  }
  expect_throws([&] { static_cast<void>(query.token_offset(3)); },
                "split output accepted an out-of-range token");
}

struct Fixture {
  std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> buffers;
  omlx::m5::MaterializedWeightView view;
  std::vector<float> expected;
};

void add_plane(Fixture &fixture, omlx::m5::MetalContext &context,
               std::string kind, std::vector<std::byte> data,
               std::uint64_t row_stride) {
  fixture.buffers.push_back(
      std::make_unique<omlx::m5::MetalBuffer>(context.upload(data)));
  fixture.view.planes.push_back({
      .kind = std::move(kind),
      .buffer = fixture.buffers.back().get(),
      .segment_index = static_cast<std::uint32_t>(fixture.buffers.size() - 1),
      .bytes = data.size(),
      .row_stride = row_stride,
  });
}

Fixture q4_parent_fixture(omlx::m5::MetalContext &context,
                          std::span<const float> input) {
  constexpr std::uint32_t n = 16;
  constexpr std::uint32_t k = 128;
  constexpr std::uint32_t group_size = 64;
  Fixture fixture;
  fixture.view = {
      .object_id = 7,
      .name = "layer.self_attn.q_proj",
      .format = "oq_affine_q4_g64_bf16_v1",
      .layout = "oq_rowsplit_k128_v1",
      .parent_shape = {n, k},
      .padded_k = k,
      .row_count = n,
      .in_features = k,
  };
  std::vector<std::uint8_t> codes(n * k / 2);
  std::vector<std::uint16_t> scales(n * k / group_size);
  std::vector<std::uint16_t> biases(n * k / group_size);
  fixture.expected.resize(n);
  for (std::uint32_t row = 0; row < n; ++row) {
    for (std::uint32_t group = 0; group < k / group_size; ++group) {
      scales[row * (k / group_size) + group] =
          omlx::m5::float_to_bf16(0.015625F * float(row + group + 1));
      biases[row * (k / group_size) + group] =
          omlx::m5::float_to_bf16(-0.03125F * float(row + 1));
    }
    double sum = 0.0;
    for (std::uint32_t column = 0; column < k; ++column) {
      const auto code = (row * 7U + column * 5U) & 0x0FU;
      codes[row * (k / 2) + column / 2] |=
          static_cast<std::uint8_t>(code << ((column & 1U) * 4U));
      const auto group = column / group_size;
      const auto weight = std::fma(
          omlx::m5::bf16_to_float(scales[row * (k / group_size) + group]),
          static_cast<float>(code),
          omlx::m5::bf16_to_float(biases[row * (k / group_size) + group]));
      sum += static_cast<double>(input[column]) * weight;
    }
    fixture.expected[row] = static_cast<float>(sum);
  }
  add_plane(fixture, context, "base_code", bytes(codes), k / 2);
  add_plane(fixture, context, "scale", bytes(scales), k / group_size * 2);
  add_plane(fixture, context, "bias", bytes(biases), k / group_size * 2);
  return fixture;
}

void test_real_linear_to_split_views() {
  auto context = omlx::m5::MetalContext::create();
  std::vector<std::uint16_t> input_words(128);
  std::vector<float> input(128);
  for (std::uint32_t column = 0; column < 128; ++column) {
    const auto value =
        static_cast<float>(static_cast<std::int32_t>(column) - 64) / 256.0F;
    input_words[column] = omlx::m5::float_to_bf16(value);
    input[column] = omlx::m5::bf16_to_float(input_words[column]);
  }
  auto fixture = q4_parent_fixture(context, input);
  const omlx::m5::ArtifactBinding binding{
      .name = fixture.view.name,
      .object_id = fixture.view.object_id,
      .format = fixture.view.format,
      .use = "attention_q",
      .activation_policy = "A16Only",
      .out_features = 16,
      .in_features = 128,
  };
  auto bundle = omlx::m5::make_full_attention_projection_bundle(
      config(), "layer.self_attn");
  omlx::m5::validate_projection_parent(bundle.parents[0], binding,
                                       fixture.view);
  omlx::m5::ProjectionOutputArena arena(context, bundle, 1);
  const auto parent = arena.parent("attention_q_gate");
  auto plan = omlx::m5::prepare_dense_qmv_t1(
      context, OMLX_M5_QMV_T1_METALLIB_PATH, fixture.view);
  auto input_buffer = context.upload(bytes(input_words));
  const auto before = context.registry_stats();
  static_cast<void>(omlx::m5::execute_dense_qmv_t1(
      context, plan, input_buffer, 0, arena.buffer(), parent.offset));
  const auto after_linear = context.registry_stats();
  const auto query = arena.output("attention_query");
  const auto gate = arena.output("attention_gate");
  const auto after_split = context.registry_stats();
  expect(after_split.dispatches_submitted ==
                 after_linear.dispatches_submitted &&
             after_split.upload_count == after_linear.upload_count &&
             after_linear.dispatches_submitted - before.dispatches_submitted ==
                 plan.info().dispatch_count,
         "split-output introduced a dispatch or upload");
  for (std::uint32_t feature = 0; feature < 8; ++feature) {
    std::uint16_t query_word = 0;
    std::uint16_t gate_word = 0;
    const auto query_bytes =
        arena.buffer().read(query.feature_offset(0, feature), 2);
    const auto gate_bytes =
        arena.buffer().read(gate.feature_offset(0, feature), 2);
    std::memcpy(&query_word, query_bytes.data(), sizeof(query_word));
    std::memcpy(&gate_word, gate_bytes.data(), sizeof(gate_word));
    const auto query_actual = omlx::m5::bf16_to_float(query_word);
    const auto gate_actual = omlx::m5::bf16_to_float(gate_word);
    const auto head = feature / 4;
    const auto within = feature % 4;
    const auto query_row = head * 8 + within;
    const auto gate_row = query_row + 4;
    expect(std::abs(query_actual - fixture.expected[query_row]) <=
                   0.02F *
                       std::max(1.0F, std::abs(fixture.expected[query_row])) &&
               std::abs(gate_actual - fixture.expected[gate_row]) <=
                   0.02F * std::max(1.0F, std::abs(fixture.expected[gate_row])),
           "fused parent output split differs from the Linear oracle");
  }
}

} // namespace

int main() {
  try {
    test_fixed_bundle_geometry();
    test_zero_copy_strided_views();
    test_real_linear_to_split_views();
    std::cout << "B3.1 projection bundle tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B3.1 projection bundle failure: " << error.what() << '\n';
    return 1;
  }
}
