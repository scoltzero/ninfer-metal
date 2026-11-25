#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/activation.h"
#include "omlx_m5/artifact.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/normalization.h"
#include "omlx_m5/qmv.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/vocab.h"

namespace {

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

const omlx::m5::ArtifactObject &
object_by_use_or_suffix(const omlx::m5::ArtifactModel &artifact,
                        std::string_view use, std::string_view suffix) {
  const auto binding = std::ranges::find_if(
      artifact.bindings(), [&](const omlx::m5::ArtifactBinding &record) {
        return record.use == use;
      });
  if (binding != artifact.bindings().end()) {
    return artifact.object(binding->object_id);
  }
  const auto object = std::ranges::find_if(
      artifact.objects(), [&](const omlx::m5::ArtifactObject &record) {
        return record.name.ends_with(suffix);
      });
  if (object == artifact.objects().end()) {
    throw std::out_of_range("vocabulary object is absent: " +
                            std::string(suffix));
  }
  return *object;
}

const omlx::m5::ArtifactObject &
object_by_suffix(const omlx::m5::ArtifactModel &artifact,
                 std::string_view suffix) {
  const auto object = std::ranges::find_if(
      artifact.objects(), [&](const omlx::m5::ArtifactObject &record) {
        return record.name.ends_with(suffix);
      });
  if (object == artifact.objects().end()) {
    throw std::out_of_range("direct object is absent: " + std::string(suffix));
  }
  return *object;
}

struct UploadedView {
  std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> buffers;
  omlx::m5::MaterializedWeightView view;
};

UploadedView upload_rows(const omlx::m5::ArtifactModel &artifact,
                         const omlx::m5::ArtifactObject &object,
                         omlx::m5::MetalContext &context,
                         std::uint64_t row_start, std::uint64_t row_count) {
  if (object.shape.size() != 2 || row_count == 0 ||
      row_start + row_count > object.shape[0]) {
    throw std::invalid_argument("vocabulary row slice is invalid");
  }
  UploadedView result;
  result.view = {
      .object_id = object.id,
      .component = omlx::m5::classify_model_component(object.name),
      .name = object.name,
      .format = object.format,
      .layout = object.layout,
      .parent_shape = object.shape,
      .padded_k = object.padded_k,
      .row_start = row_start,
      .row_count = row_count,
      .in_features = object.shape[1],
  };
  for (const auto &selected_plane : object.planes) {
    if (selected_plane.bytes % object.shape[0] != 0) {
      throw std::runtime_error("vocabulary plane has no row stride");
    }
    const auto stride = selected_plane.bytes / object.shape[0];
    const auto plane = artifact.plane_bytes(object, selected_plane);
    const auto selected = plane.subspan(row_start * stride, row_count * stride);
    result.buffers.push_back(
        std::make_unique<omlx::m5::MetalBuffer>(context.upload(selected)));
    result.view.planes.push_back({
        .kind = selected_plane.kind,
        .buffer = result.buffers.back().get(),
        .bytes = selected.size(),
        .row_stride = stride,
    });
  }
  return result;
}

std::vector<std::uint16_t>
direct_words(const omlx::m5::ArtifactModel &artifact,
             const omlx::m5::ArtifactObject &object) {
  if (object.format != "direct_bf16_v1" || object.planes.size() != 1) {
    throw std::invalid_argument("final norm is not direct BF16");
  }
  const auto raw = artifact.plane_bytes(object, object.planes.front());
  std::vector<std::uint16_t> result(raw.size() / 2);
  std::memcpy(result.data(), raw.data(), raw.size());
  return result;
}

std::vector<std::uint16_t> read_words(const omlx::m5::MetalBuffer &buffer,
                                      std::uint64_t offset,
                                      std::uint64_t count) {
  const auto raw = buffer.read(offset, count * 2);
  std::vector<std::uint16_t> result(count);
  std::memcpy(result.data(), raw.data(), raw.size());
  return result;
}

std::vector<float> floats(std::span<const std::uint16_t> words) {
  std::vector<float> result(words.size());
  std::ranges::transform(words, result.begin(), omlx::m5::bf16_to_float);
  return result;
}

std::uint32_t read_token(const omlx::m5::MetalBuffer &buffer) {
  const auto raw = buffer.read(0, 4);
  std::uint32_t result = 0;
  std::memcpy(&result, raw.data(), sizeof(result));
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 6) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_vocab ARTIFACT CONFIG VOCAB_METALLIB "
          "QMV_METALLIB RMS_METALLIB");
    }
    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    const auto config = omlx::m5::load_qwen_model_config(argv[2]);
    const auto &embedding_object =
        object_by_use_or_suffix(artifact, "embedding", "embed_tokens.weight");
    const auto &lm_head_object =
        object_by_use_or_suffix(artifact, "lm_head", "lm_head.weight");
    const auto &norm_object = object_by_suffix(artifact, "model.norm.weight");
    constexpr std::uint32_t embedding_start = 100;
    constexpr std::uint32_t embedding_rows = 3;
    constexpr std::uint32_t lm_head_start = 200;
    constexpr std::uint32_t lm_head_rows = 64;
    auto context = omlx::m5::MetalContext::create();
    auto embedding = upload_rows(artifact, embedding_object, context,
                                 embedding_start, embedding_rows);
    const std::vector<std::uint32_t> token_values = {
        embedding_start, embedding_start + 1, embedding_start + 2};
    auto token_ids = context.upload(bytes(token_values));
    auto embedding_plan = omlx::m5::prepare_embedding(
        context, argv[3], embedding.view, token_values.size());
    auto embedding_output =
        context.make_buffer(embedding_plan.info().output_bytes);
    const auto embedding_submission = omlx::m5::execute_embedding(
        context, embedding_plan, token_ids, 0, embedding_output, 0);
    const auto embedding_words = read_words(
        embedding_output, 0,
        static_cast<std::uint64_t>(token_values.size()) * config.hidden_size);

    const auto norm_words = direct_words(artifact, norm_object);
    auto norm_weight = context.upload(bytes(norm_words));
    auto normalized = context.make_buffer(config.hidden_size * 2ULL);
    const omlx::m5::ActivationLease norm_lease{
        .bundle =
            {
                .epoch = 1,
                .key = {.source_id = embedding_object.id,
                        .normalization_id = norm_object.id,
                        .input_features =
                            static_cast<std::uint32_t>(config.hidden_size),
                        .tokens = 1},
                .normalized_a16 =
                    omlx::m5::ActivationPlaneView{
                        .buffer = &normalized,
                        .bytes =
                            static_cast<std::uint64_t>(config.hidden_size) * 2},
            },
        .produce_a16 = true,
    };
    auto norm_plan = omlx::m5::prepare_rmsnorm_projection(
        context, argv[5], config.hidden_size, 1,
        static_cast<float>(config.rms_norm_eps));
    const auto norm_submission = omlx::m5::produce_rmsnorm_activation(
        context, norm_plan, embedding_output, 0, norm_weight, 0, norm_lease);
    const auto normalized_words = read_words(normalized, 0, config.hidden_size);

    auto lm_head = upload_rows(artifact, lm_head_object, context, lm_head_start,
                               lm_head_rows);
    auto lm_head_plan =
        omlx::m5::prepare_dense_qmv_t1(context, argv[4], lm_head.view);
    auto logits = context.make_buffer(lm_head_plan.info().output_bytes);
    const auto lm_head_submission = omlx::m5::execute_dense_qmv_t1(
        context, lm_head_plan, normalized, 0, logits, 0);
    const auto logit_words = read_words(logits, 0, lm_head_rows);

    std::vector<std::uint16_t> synthetic_logits(config.vocab_size);
    for (std::int32_t token = 0; token < config.vocab_size; ++token) {
      synthetic_logits[token] = omlx::m5::float_to_bf16(
          float((static_cast<std::uint32_t>(token) * 37U) % 211U) / 32.0F);
    }
    constexpr std::uint32_t forced_greedy = 777;
    synthetic_logits[forced_greedy] = omlx::m5::float_to_bf16(8.0F);
    auto synthetic = context.upload(bytes(synthetic_logits));
    auto sampled_token = context.make_buffer(4);
    auto sampling_plan =
        omlx::m5::prepare_sampling(context, argv[3], config.vocab_size);
    const auto greedy_submission = omlx::m5::execute_sampling(
        context, sampling_plan, synthetic, 0, {}, sampled_token, 0);
    const auto greedy_token = read_token(sampled_token);
    const omlx::m5::GpuSamplingOptions sampled_options{
        .temperature = 2.0F,
        .top_k = 16,
        .top_p = 0.95F,
        .seed = 123456789,
        .stream = 7,
        .absolute_position = 29,
    };
    const auto sampled_submission =
        omlx::m5::execute_sampling(context, sampling_plan, synthetic, 0,
                                   sampled_options, sampled_token, 0);
    const auto sampled_first = read_token(sampled_token);
    static_cast<void>(omlx::m5::execute_sampling(context, sampling_plan,
                                                 synthetic, 0, sampled_options,
                                                 sampled_token, 0));
    const auto sampled_second = read_token(sampled_token);
    if (greedy_token != forced_greedy || sampled_first != sampled_second) {
      throw std::runtime_error("GPU sampling contract failed");
    }
    const nlohmann::json result = {
        {"schema", "omlx-m5-b3-8-vocab-v1"},
        {"evidence_scope",
         "M4 embedding/final norm/lm-head/sampling correctness"},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"embedding",
         {{"object", embedding_object.name},
          {"format", embedding_object.format},
          {"row_start", embedding_start},
          {"row_count", embedding_rows},
          {"tokens", token_values},
          {"output", floats(embedding_words)},
          {"dispatches", embedding_submission.dispatch_count}}},
        {"final_norm",
         {{"object", norm_object.name},
          {"input",
           floats(std::span(embedding_words).first(config.hidden_size))},
          {"output", floats(normalized_words)},
          {"dispatches", norm_submission.submission->dispatch_count}}},
        {"lm_head",
         {{"object", lm_head_object.name},
          {"format", lm_head_object.format},
          {"row_start", lm_head_start},
          {"row_count", lm_head_rows},
          {"logits", floats(logit_words)},
          {"dispatches", lm_head_submission.dispatch_count}}},
        {"sampling",
         {{"vocabulary", config.vocab_size},
          {"greedy_token", greedy_token},
          {"sampled_token", sampled_first},
          {"repeat_token", sampled_second},
          {"greedy_dispatches", greedy_submission.dispatch_count},
          {"sampled_dispatches", sampled_submission.dispatch_count},
          {"temperature", sampled_options.temperature},
          {"top_k", sampled_options.top_k},
          {"top_p", sampled_options.top_p},
          {"seed", sampled_options.seed},
          {"stream", sampled_options.stream},
          {"absolute_position", sampled_options.absolute_position}}},
    };
    std::cout << result.dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "vocabulary inspection failed: " << error.what() << '\n';
    return 1;
  }
}
