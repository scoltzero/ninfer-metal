#include "omlx_m5/dflash2.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace omlx::m5 {

namespace {

std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

double uniform01(std::uint64_t seed, std::uint64_t position,
                 std::uint64_t purpose) {
  const auto bits =
      splitmix64(seed ^ splitmix64(position) ^ splitmix64(purpose));
  return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
}

std::vector<float> normalized_probabilities(std::span<const float> values) {
  if (values.empty()) {
    throw std::invalid_argument("probability vector must not be empty");
  }
  double total = 0.0;
  for (const auto value : values) {
    if (!std::isfinite(value) || value < 0.0F) {
      throw std::invalid_argument(
          "probability vectors must be finite and non-negative");
    }
    total += value;
  }
  if (!(total > 0.0)) {
    throw std::invalid_argument("probability vector has zero mass");
  }
  std::vector<float> result(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    result[index] = static_cast<float>(values[index] / total);
  }
  return result;
}

std::size_t sample_index(std::span<const float> probabilities, double uniform) {
  double cumulative = 0.0;
  std::size_t last_nonzero = probabilities.size() - 1;
  for (std::size_t index = 0; index < probabilities.size(); ++index) {
    if (probabilities[index] > 0.0F) {
      last_nonzero = index;
    }
    cumulative += probabilities[index];
    if (uniform < cumulative) {
      return index;
    }
  }
  return last_nonzero;
}

std::size_t lowest_argmax(std::span<const float> values) {
  if (values.empty()) {
    throw std::invalid_argument("argmax input must not be empty");
  }
  return static_cast<std::size_t>(std::distance(
      values.begin(), std::max_element(values.begin(), values.end())));
}

std::vector<float> softmax(std::span<const float> logits, float temperature) {
  if (!(temperature > 0.0F) || !std::isfinite(temperature)) {
    throw std::invalid_argument(
        "softmax temperature must be finite and positive");
  }
  const auto maximum = *std::max_element(logits.begin(), logits.end());
  std::vector<float> result(logits.size());
  double total = 0.0;
  for (std::size_t index = 0; index < logits.size(); ++index) {
    result[index] =
        std::exp((static_cast<double>(logits[index]) - maximum) / temperature);
    total += result[index];
  }
  for (auto &value : result) {
    value = static_cast<float>(value / total);
  }
  return result;
}

bool ends_with(std::span<const std::int32_t> tokens,
               std::span<const std::int32_t> suffix) {
  return suffix.size() <= tokens.size() &&
         std::equal(suffix.rbegin(), suffix.rend(), tokens.rbegin());
}

std::size_t stop_prefix_holdback(
    std::span<const std::int32_t> tokens,
    const std::vector<std::vector<std::int32_t>> &stop_sequences) {
  std::size_t holdback = 0;
  for (const auto &stop : stop_sequences) {
    const auto max_len = std::min(tokens.size(), stop.size() - 1);
    for (std::size_t len = max_len; len > holdback; --len) {
      if (std::equal(stop.begin(),
                     stop.begin() + static_cast<std::ptrdiff_t>(len),
                     tokens.end() - static_cast<std::ptrdiff_t>(len))) {
        holdback = len;
        break;
      }
    }
  }
  return holdback;
}

} // namespace

double candidate_selector_uniform(std::uint64_t seed, std::uint64_t position) {
  return uniform01(seed, position, 0xDFA52002ULL);
}

double speculative_uniform(std::uint64_t seed, std::uint64_t position,
                           std::uint64_t purpose) {
  return uniform01(seed, position, purpose);
}

TargetDistribution target_distribution_from_topk(
    std::span<const std::uint32_t> ids, std::span<const float> scores,
    std::uint32_t vocabulary, float temperature, float top_p) {
  if (ids.empty() || ids.size() != scores.size() || !vocabulary ||
      !std::isfinite(temperature) || temperature < 0 || !std::isfinite(top_p) ||
      top_p < 0 || top_p > 1)
    throw std::invalid_argument("target top-k distribution geometry");
  TargetDistribution output{std::vector<float>(vocabulary, 0)};
  if (temperature == 0) {
    if (ids[0] >= vocabulary)
      throw std::invalid_argument("target top-k ID outside vocabulary");
    output.probabilities[ids[0]] = 1;
    return output;
  }
  double total = 0;
  std::vector<double> values(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] >= vocabulary || !std::isfinite(scores[i]))
      throw std::invalid_argument("target top-k value invalid");
    values[i] = std::exp((double(scores[i]) - scores[0]) / temperature);
    total += values[i];
  }
  double cumulative = 0;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    output.probabilities[ids[i]] = float(values[i] / total);
    cumulative += values[i] / total;
    if (top_p < 1 && cumulative >= top_p)
      break;
  }
  return output;
}

std::vector<float> dynamic_grouped_conv_two_tap(std::span<const float> input,
                                                std::uint32_t positions,
                                                const DynamicConvSpec &spec,
                                                std::span<const float> base,
                                                std::span<const float> delta) {
  if (positions == 0 || spec.width == 0 || spec.group_width == 0 ||
      spec.width % spec.group_width != 0 ||
      input.size() != static_cast<std::size_t>(positions) * spec.width ||
      base.size() != static_cast<std::size_t>(2) * spec.width) {
    throw std::invalid_argument("dynamic convolution geometry mismatch");
  }
  const auto groups = spec.width / spec.group_width;
  if (delta.size() != static_cast<std::size_t>(positions) * 2 * groups) {
    throw std::invalid_argument("dynamic convolution delta geometry mismatch");
  }
  std::vector<float> output(input.size());
  for (std::uint32_t position = 0; position < positions; ++position) {
    for (std::uint32_t channel = 0; channel < spec.width; ++channel) {
      const auto group = channel / spec.group_width;
      const auto delta_base =
          (static_cast<std::size_t>(position) * 2) * groups + group;
      auto value =
          (base[channel] + delta[delta_base]) *
          input[static_cast<std::size_t>(position) * spec.width + channel];
      if (position > 0) {
        value += (base[spec.width + channel] + delta[delta_base + groups]) *
                 input[(static_cast<std::size_t>(position) - 1) * spec.width +
                       channel];
      }
      output[static_cast<std::size_t>(position) * spec.width + channel] = value;
    }
  }
  return output;
}

std::vector<float>
materialize_target_context(std::span<const float> concatenated_features,
                           std::uint32_t positions, std::uint32_t input_width,
                           std::span<const float> projection,
                           std::uint32_t hidden_width,
                           std::span<const float> norm_weight, float epsilon) {
  if (positions == 0 || input_width == 0 || hidden_width == 0 ||
      epsilon <= 0.0F ||
      concatenated_features.size() !=
          static_cast<std::size_t>(positions) * input_width ||
      projection.size() !=
          static_cast<std::size_t>(hidden_width) * input_width ||
      norm_weight.size() != hidden_width) {
    throw std::invalid_argument("target context geometry mismatch");
  }
  std::vector<float> output(static_cast<std::size_t>(positions) * hidden_width);
  for (std::uint32_t position = 0; position < positions; ++position) {
    auto *row =
        output.data() + static_cast<std::size_t>(position) * hidden_width;
    for (std::uint32_t out = 0; out < hidden_width; ++out) {
      float value = 0.0F;
      for (std::uint32_t in = 0; in < input_width; ++in) {
        value = std::fma(
            projection[static_cast<std::size_t>(out) * input_width + in],
            concatenated_features[static_cast<std::size_t>(position) *
                                      input_width +
                                  in],
            value);
      }
      row[out] = value;
    }
    float mean_square = 0.0F;
    for (std::uint32_t index = 0; index < hidden_width; ++index) {
      mean_square = std::fma(row[index], row[index], mean_square);
    }
    mean_square /= static_cast<float>(hidden_width);
    const auto inverse = 1.0F / std::sqrt(mean_square + epsilon);
    for (std::uint32_t index = 0; index < hidden_width; ++index) {
      row[index] *= inverse * norm_weight[index];
    }
  }
  return output;
}

SelectorResult select_candidate_path(
    std::int32_t anchor_token, std::span<const CandidateRow> rows,
    std::span<const float> predecessor_codebook,
    std::span<const float> successor_codebook, std::uint32_t vocab_size,
    std::uint32_t rank, float temperature, std::uint64_t seed,
    std::uint64_t absolute_position, SamplingContract contract) {
  (void)sampling_contract_name(contract);
  if (anchor_token < 0 ||
      static_cast<std::uint32_t>(anchor_token) >= vocab_size ||
      vocab_size == 0 || rank == 0 ||
      predecessor_codebook.size() !=
          static_cast<std::size_t>(vocab_size) * rank ||
      successor_codebook.size() !=
          static_cast<std::size_t>(vocab_size) * rank ||
      (!std::isfinite(temperature))) {
    throw std::invalid_argument("candidate selector geometry is invalid");
  }
  SelectorResult result;
  result.tokens.reserve(rows.size());
  result.steps.reserve(rows.size());
  auto predecessor = anchor_token;
  for (std::size_t position = 0; position < rows.size(); ++position) {
    const auto &row = rows[position];
    if (row.token_ids.empty() ||
        row.token_ids.size() != row.unary_scores.size() ||
        row.projected_hidden.size() != rank) {
      throw std::invalid_argument("candidate row geometry mismatch");
    }
    std::vector<float> energies(row.token_ids.size());
    for (std::size_t candidate = 0; candidate < row.token_ids.size();
         ++candidate) {
      const auto token = row.token_ids[candidate];
      if (token < 0 || static_cast<std::uint32_t>(token) >= vocab_size) {
        throw std::invalid_argument(
            "candidate token is outside the vocabulary");
      }
      float interaction = 0.0F;
      for (std::uint32_t component = 0; component < rank; ++component) {
        interaction = std::fma(
            predecessor_codebook[static_cast<std::size_t>(predecessor) * rank +
                                 component] *
                row.projected_hidden[component],
            successor_codebook[static_cast<std::size_t>(token) * rank +
                               component],
            interaction);
      }
      energies[candidate] = row.unary_scores[candidate] + interaction;
    }

    std::vector<float> probabilities(row.token_ids.size(), 0.0F);
    std::size_t selected = 0;
    if (temperature <= 0.0F) {
      selected = lowest_argmax(energies);
      if(contract==SamplingContract::PositionCoupledV1) {
        std::vector<std::uint32_t> ids(row.token_ids.begin(),row.token_ids.end());
        auto token=position_sample_candidates(ids,energies,{.seed=seed,.position=absolute_position+position,
          .purpose=draft_sampling_purpose,.temperature=0});
        selected=std::find(row.token_ids.begin(),row.token_ids.end(),std::int32_t(token))-row.token_ids.begin();
      }
      probabilities[selected] = 1.0F;
    } else {
      probabilities = softmax(energies, temperature);
      if(contract==SamplingContract::PositionCoupledV1) {
        std::vector<std::uint32_t> ids(row.token_ids.begin(),row.token_ids.end());
        auto token=position_sample_candidates(ids,energies,{.seed=seed,.position=absolute_position+position,
          .purpose=draft_sampling_purpose,.top_k=std::uint32_t(ids.size()),.temperature=temperature});
        selected=std::find(row.token_ids.begin(),row.token_ids.end(),std::int32_t(token))-row.token_ids.begin();
      } else selected = sample_index(probabilities,
          candidate_selector_uniform(seed, absolute_position + position));
    }
    predecessor = row.token_ids[selected];
    result.tokens.push_back(predecessor);
    result.steps.push_back({
        .token = predecessor,
        .candidate_ids = row.token_ids,
        .probabilities = std::move(probabilities),
    });
  }
  return result;
}

VerificationResult
verify_speculative(std::span<const SelectorStep> proposals,
                   std::span<const TargetDistribution> target,
                   float temperature, std::uint64_t seed,
                   std::uint64_t absolute_position) {
  if (target.size() != proposals.size() + 1 || !std::isfinite(temperature)) {
    throw std::invalid_argument("target verification width mismatch");
  }
  std::vector<std::vector<float>> target_probabilities;
  target_probabilities.reserve(target.size());
  for (const auto &step : target) {
    target_probabilities.push_back(
        normalized_probabilities(step.probabilities));
  }

  VerificationResult result;
  for (std::size_t position = 0; position < proposals.size(); ++position) {
    const auto &proposal = proposals[position];
    const auto &p = target_probabilities[position];
    if (proposal.token < 0 ||
        static_cast<std::size_t>(proposal.token) >= p.size() ||
        proposal.candidate_ids.size() != proposal.probabilities.size()) {
      throw std::invalid_argument("proposal distribution is invalid");
    }
    const auto proposal_token = static_cast<std::size_t>(proposal.token);
    if (temperature <= 0.0F) {
      const auto target_token = lowest_argmax(p);
      if (proposal_token == target_token) {
        result.licensed_outputs.push_back(proposal.token);
        ++result.accepted_proposals;
        continue;
      }
      result.licensed_outputs.push_back(
          static_cast<std::int32_t>(target_token));
      result.used_correction = true;
      return result;
    }

    double q_sum = 0.0;
    for (std::size_t candidate = 0; candidate < proposal.candidate_ids.size();
         ++candidate) {
      const auto token = proposal.candidate_ids[candidate];
      const auto prob = proposal.probabilities[candidate];
      if (token < 0 || static_cast<std::size_t>(token) >= p.size() ||
          !std::isfinite(prob) || prob < 0.0F) {
        throw std::invalid_argument("proposal distribution is invalid");
      }
      q_sum += prob;
    }
    const double q_scale = q_sum > 1.0 ? (1.0 / q_sum) : 1.0;

    float q_token = 0.0F;
    for (std::size_t candidate = 0; candidate < proposal.candidate_ids.size();
         ++candidate) {
      if (proposal.candidate_ids[candidate] == proposal.token) {
        q_token =
            static_cast<float>(proposal.probabilities[candidate] * q_scale);
        break;
      }
    }
    const auto p_token = p[proposal_token];
    const auto accept_probability =
        q_token > 0.0F ? std::min(1.0F, p_token / q_token)
                       : (p_token > 0.0F ? 1.0F : 0.0F);
    const auto uniform =
        uniform01(seed, absolute_position + position, 0xACCE9701ULL);
    if (uniform < accept_probability) {
      result.licensed_outputs.push_back(proposal.token);
      ++result.accepted_proposals;
      continue;
    }

    std::vector<double> correction_raw(p.begin(), p.end());
    for (std::size_t candidate = 0; candidate < proposal.candidate_ids.size();
         ++candidate) {
      const auto token =
          static_cast<std::size_t>(proposal.candidate_ids[candidate]);
      const double q_prob = proposal.probabilities[candidate] * q_scale;
      correction_raw[token] = std::max(0.0, correction_raw[token] - q_prob);
    }
    double correction_total = 0.0;
    for (const auto value : correction_raw) {
      correction_total += value;
    }
    std::vector<float> correction;
    if (correction_total > 0.0) {
      correction.resize(correction_raw.size());
      for (std::size_t index = 0; index < correction_raw.size(); ++index) {
        correction[index] =
            static_cast<float>(correction_raw[index] / correction_total);
      }
    } else {
      correction = p;
    }
    const auto correction_token =
        sample_index(correction, uniform01(seed, absolute_position + position,
                                           0xC022EC71ULL));
    result.licensed_outputs.push_back(
        static_cast<std::int32_t>(correction_token));
    result.used_correction = true;
    return result;
  }

  const auto &bonus = target_probabilities.back();
  const auto bonus_token =
      temperature <= 0.0F
          ? lowest_argmax(bonus)
          : sample_index(bonus,
                         uniform01(seed, absolute_position + proposals.size(),
                                   0xB0A05EEDULL));
  result.licensed_outputs.push_back(static_cast<std::int32_t>(bonus_token));
  return result;
}

PreviewResult preview_licensed_outputs(
    std::span<const std::int32_t> licensed, std::size_t remaining_budget,
    std::span<const std::int32_t> eos_token_ids,
    const std::vector<std::vector<std::int32_t>> &stop_sequences,
    bool cancelled) {
  if (cancelled) {
    return {.finish_reason = PreviewFinishReason::cancelled};
  }
  PreviewResult result;
  for (const auto token : licensed) {
    if (result.tokens.size() >= remaining_budget) {
      result.finish_reason = PreviewFinishReason::length;
      break;
    }
    if (std::ranges::find(eos_token_ids, token) != eos_token_ids.end()) {
      result.finish_reason = PreviewFinishReason::eos;
      break;
    }
    result.tokens.push_back(token);
    bool stopped = false;
    for (const auto &stop : stop_sequences) {
      if (stop.empty()) {
        throw std::invalid_argument("stop token sequences must not be empty");
      }
      if (ends_with(result.tokens, stop)) {
        result.tokens.resize(result.tokens.size() - stop.size());
        result.finish_reason = PreviewFinishReason::stop;
        stopped = true;
        break;
      }
    }
    if (stopped) {
      break;
    }
  }
  result.commit_length = result.tokens.size();
  if (result.finish_reason == PreviewFinishReason::none &&
      remaining_budget != std::numeric_limits<std::size_t>::max() &&
      result.tokens.size() == remaining_budget) {
    result.finish_reason = PreviewFinishReason::length;
  }
  return result;
}

std::string_view finish_reason_name(PreviewFinishReason reason) {
  switch (reason) {
  case PreviewFinishReason::none:
    return "none";
  case PreviewFinishReason::length:
    return "length";
  case PreviewFinishReason::eos:
    return "eos";
  case PreviewFinishReason::stop:
    return "stop";
  case PreviewFinishReason::cancelled:
    return "cancelled";
  }
  throw std::logic_error("unknown preview finish reason");
}

namespace {

std::uint64_t checkpoint_checksum(const QwenArCheckpoint &checkpoint) {
  std::uint64_t hash = 1'469'598'103'934'665'603ULL;
  for (const auto value : checkpoint.state) {
    hash ^= std::to_integer<std::uint8_t>(value);
    hash *= 1'099'511'628'211ULL;
  }
  hash ^= checkpoint.next_position;
  hash *= 1'099'511'628'211ULL;
  return hash;
}

std::uint64_t float_checksum(std::span<const float> values) {
  auto bytes = std::as_bytes(values);
  std::uint64_t hash = 1'469'598'103'934'665'603ULL;
  for (const auto value : bytes) {
    hash ^= std::to_integer<std::uint8_t>(value);
    hash *= 1'099'511'628'211ULL;
  }
  return hash;
}

void append_target_features(std::vector<float> &destination,
                            std::span<const float> hidden,
                            std::uint32_t hidden_size,
                            std::uint32_t target_taps) {
  if (hidden_size == 0 || target_taps == 0 ||
      hidden.size() % hidden_size != 0) {
    throw std::invalid_argument(
        "DFlash target feature capture shape is invalid");
  }
  const auto rows = hidden.size() / hidden_size;
  destination.reserve(destination.size() + rows * target_taps * hidden_size);
  for (std::size_t row = 0; row < rows; ++row) {
    const auto source = hidden.subspan(row * hidden_size, hidden_size);
    for (std::uint32_t tap = 0; tap < target_taps; ++tap) {
      destination.insert(destination.end(), source.begin(), source.end());
    }
  }
}

TargetDistribution one_hot_distribution(std::uint32_t vocabulary,
                                        std::uint32_t token);

TargetDistribution target_distribution(std::span<const float> logits,
                                       float temperature) {
  if (temperature <= 0.0F) {
    return one_hot_distribution(
        static_cast<std::uint32_t>(logits.size()),
        static_cast<std::uint32_t>(lowest_argmax(logits)));
  }
  return {.probabilities = softmax(logits, temperature)};
}

GdnTransitionRecord transition_record(std::int32_t token, std::uint32_t ordinal,
                                      const GdnGeometry &geometry) {
  std::vector<float> key(geometry.key_dim);
  std::vector<float> value(geometry.value_dim);
  for (std::uint32_t index = 0; index < geometry.key_dim; ++index) {
    key[index] =
        static_cast<float>((token + static_cast<std::int32_t>(index * 3U) +
                            static_cast<std::int32_t>(ordinal)) %
                               17 -
                           8) *
        (1.0F / 16.0F);
  }
  for (std::uint32_t index = 0; index < geometry.value_dim; ++index) {
    value[index] =
        static_cast<float>((token + static_cast<std::int32_t>(index * 5U) +
                            static_cast<std::int32_t>(ordinal * 2U)) %
                               19 -
                           9) *
        (1.0F / 16.0F);
  }
  return make_represented_record(key, value,
                                 -0.01F * static_cast<float>(ordinal + 1),
                                 0.25F + 0.01F * static_cast<float>(ordinal));
}

TargetDistribution one_hot_distribution(std::uint32_t vocabulary,
                                        std::uint32_t token) {
  if (vocabulary == 0 || token >= vocabulary) {
    throw std::invalid_argument("DFlash target token is out of range");
  }
  TargetDistribution result{.probabilities = std::vector<float>(vocabulary)};
  result.probabilities[token] = 1.0F;
  return result;
}

std::vector<std::uint32_t> to_unsigned(std::span<const std::int32_t> values) {
  std::vector<std::uint32_t> result;
  result.reserve(values.size());
  for (const auto value : values) {
    if (value < 0) {
      throw std::invalid_argument("DFlash token is negative");
    }
    result.push_back(static_cast<std::uint32_t>(value));
  }
  return result;
}

} // namespace

class QwenDFlashProgram::Impl {
public:
  Impl(QwenProgram &&selected_target, DFlash2DraftProgram &&selected_draft,
       DFlashProgramOptions selected_options)
      : target(std::move(selected_target)), draft(std::move(selected_draft)),
        options(selected_options),
        sequence(64, GdnState({.key_dim = 8, .value_dim = 4})) {
    if (options.draft_tokens == 0 || options.draft_tokens > 7) {
      throw std::invalid_argument(
          "R1 DFlash2 Program requires draft_tokens in [1, 7]");
    }
    stats.verify_width_histogram.resize(options.draft_tokens + 2);
    stats.accepted_length_histogram.resize(options.draft_tokens + 1);
  }

  void initialize(std::span<const std::uint32_t> selected_prompt) {
    if (selected_prompt.size() < 2 || !target.autoregressive_installed() ||
        target.build_plan().vocabulary_size != draft.config().vocabulary_size ||
        target.build_plan().hidden_size != draft.config().hidden_size ||
        target.build_plan().max_context >
            draft.config().max_position_embeddings ||
        options.draft_tokens + 1 > draft.config().block_size) {
      throw std::invalid_argument("DFlash Program initialization is invalid");
    }
    for (const auto token : selected_prompt) {
      if (token >= target.build_plan().vocabulary_size) {
        throw std::invalid_argument("DFlash prompt token is out of range");
      }
    }
    prompt.assign(selected_prompt.begin(), selected_prompt.end());
    target_context_features.clear();
    target.reset_autoregressive();
    if (prompt.size() > 1) {
      const auto prefill = target.prefill(
          std::span<const std::uint32_t>(prompt).first(prompt.size() - 1));
      append_target_features(
          target_context_features, prefill.hidden, draft.config().hidden_size,
          static_cast<std::uint32_t>(draft.config().target_layer_ids.size()));
    }
    target_checkpoint = target.checkpoint_autoregressive();
    anchor = static_cast<std::int32_t>(prompt.back());
    draft_checkpoint = target_context_features;

    sequence = SequenceState(64, GdnState({.key_dim = 8, .value_dim = 4}));
    sequence.target_frontier = prompt.size() - 1;
    sequence.context_frontier = sequence.target_frontier;
    sequence.kv_page_count =
        (sequence.target_frontier + sequence.kv_page_tokens - 1) /
        sequence.kv_page_tokens;
    sequence.pending_anchor = anchor;
    generated_history.clear();
    pending_stop_prefix.clear();
  }

  QwenProgram target;
  DFlash2DraftProgram draft;
  DFlashProgramOptions options;
  std::vector<std::uint32_t> prompt;
  QwenArCheckpoint target_checkpoint;
  std::vector<float> draft_checkpoint;
  std::vector<float> target_context_features;
  std::vector<std::int32_t> generated_history;
  std::vector<std::int32_t> pending_stop_prefix;
  std::int32_t anchor = -1;
  SequenceState sequence;
  DFlashProgramStats stats;
};

QwenDFlashProgram::QwenDFlashProgram(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
QwenDFlashProgram::QwenDFlashProgram(QwenDFlashProgram &&) noexcept = default;
QwenDFlashProgram &
QwenDFlashProgram::operator=(QwenDFlashProgram &&) noexcept = default;
QwenDFlashProgram::~QwenDFlashProgram() = default;

QwenDFlashProgram
QwenDFlashProgram::create(QwenProgram &&target, DFlash2DraftProgram &&draft,
                          std::span<const std::uint32_t> prompt,
                          DFlashProgramOptions options) {
  auto impl =
      std::make_unique<Impl>(std::move(target), std::move(draft), options);
  impl->initialize(prompt);
  return QwenDFlashProgram(std::move(impl));
}

std::int32_t QwenDFlashProgram::pending_anchor() const { return impl_->anchor; }

std::uint32_t QwenDFlashProgram::configured_draft_tokens() const {
  return impl_->options.draft_tokens;
}

const SequenceState &QwenDFlashProgram::sequence_state() const {
  return impl_->sequence;
}

const DFlashProgramStats &QwenDFlashProgram::stats() const {
  return impl_->stats;
}

void QwenDFlashProgram::reset(std::span<const std::uint32_t> prompt) {
  impl_->initialize(prompt);
}

DFlashRoundResult
QwenDFlashProgram::run_round(const DFlashRoundOptions &options) {
  auto proposal_count = impl_->options.draft_tokens;
  if (options.remaining_budget != std::numeric_limits<std::size_t>::max()) {
    const auto budget_drafts =
        options.remaining_budget == 0 ? 0 : options.remaining_budget - 1;
    proposal_count = static_cast<std::uint32_t>(
        std::min<std::size_t>(proposal_count, budget_drafts));
  }
  if (options.forced_accepted_proposals.has_value() &&
      *options.forced_accepted_proposals > proposal_count) {
    throw std::invalid_argument("DFlash proposal/acceptance width is invalid");
  }
  const auto width = proposal_count + 1;
  const auto target_before = impl_->target_checkpoint;
  const auto draft_before = impl_->draft_checkpoint;
  DFlashRoundResult result{
      .proposal_count = proposal_count,
      .verify_width = width,
      .target_state_before = checkpoint_checksum(target_before),
      .draft_state_before = float_checksum(draft_before),
  };
  ++impl_->stats.rounds;
  ++impl_->stats.verify_width_histogram[width];
  RoundTransaction transaction(impl_->sequence, width);
  if (options.cancelled) {
    transaction.cancel();
    impl_->target.restore_autoregressive(target_before);
    impl_->target_context_features = draft_before;
    result.finish_reason = PreviewFinishReason::cancelled;
    result.stage = transaction.stage();
    result.target_state_after = checkpoint_checksum(target_before);
    result.draft_state_after = float_checksum(draft_before);
    ++impl_->stats.cancelled_rounds;
    return result;
  }

  try {
    std::vector<SelectorStep> proposal_steps;
    if (proposal_count != 0) {
      const auto target_taps = impl_->draft.config().target_layer_ids.size();
      const auto context_stride =
          target_taps * impl_->draft.config().hidden_size;
      if (impl_->target_context_features.size() % context_stride != 0) {
        throw std::logic_error("DFlash2 context feature state is malformed");
      }
      const auto context_tokens = static_cast<std::uint32_t>(
          impl_->target_context_features.size() / context_stride);
      auto proposal = impl_->draft.propose(
          impl_->anchor, impl_->target_context_features, context_tokens,
          proposal_count, options.temperature, options.seed,
          impl_->sequence.target_frontier);
      result.proposals = proposal.selection.tokens;
      proposal_steps = std::move(proposal.selection.steps);
      result.draft_dispatches = proposal.draft_dispatches;
    }

    impl_->target.restore_autoregressive(target_before);
    std::vector<std::uint32_t> verify_tokens;
    verify_tokens.reserve(width);
    verify_tokens.push_back(static_cast<std::uint32_t>(impl_->anchor));
    for (const auto token : result.proposals) {
      verify_tokens.push_back(static_cast<std::uint32_t>(token));
    }
    const auto verify_mode =
        width == 1 ? TopologyProfileMode::decode : TopologyProfileMode::prefill;
    const auto verify = impl_->target.execute_ar(
        verify_tokens, verify_mode, QwenArExecutionRoute::execute_batch);
    if (verify.sampled_tokens.size() != width) {
      throw std::runtime_error("DFlash target verify output width mismatch");
    }
    result.target_verify_calls = 1;
    result.target_weight_publications = 1;
    result.target_verify_state =
        checkpoint_checksum(impl_->target.checkpoint_autoregressive());
    std::vector<TargetDistribution> target_distributions;
    target_distributions.reserve(width);
    const auto vocabulary = impl_->target.build_plan().vocabulary_size;
    for (std::uint32_t index = 0; index < width; ++index) {
      const auto row =
          std::span(verify.logits)
              .subspan(std::uint64_t(index) * vocabulary, vocabulary);
      auto distribution = target_distribution(row, options.temperature);
      auto target_token = static_cast<std::uint32_t>(lowest_argmax(row));
      if (options.forced_accepted_proposals.has_value() &&
          index < proposal_count) {
        const auto accepted = *options.forced_accepted_proposals;
        if (index < accepted) {
          target_token = static_cast<std::uint32_t>(result.proposals[index]);
        } else if (index == accepted) {
          target_token =
              (static_cast<std::uint32_t>(result.proposals[index]) + 1U) %
              vocabulary;
        }
        distribution = one_hot_distribution(vocabulary, target_token);
      }
      result.target_predictions.push_back(
          static_cast<std::int32_t>(target_token));
      target_distributions.push_back(std::move(distribution));
    }
    const auto verification = verify_speculative(
        proposal_steps, target_distributions, options.temperature, options.seed,
        impl_->sequence.target_frontier);
    result.licensed_outputs = verification.licensed_outputs;
    result.accepted_proposals = verification.accepted_proposals;
    result.used_correction = verification.used_correction;
    result.used_bonus =
        !verification.used_correction &&
        verification.accepted_proposals == result.proposals.size();
    ++impl_->stats.accepted_length_histogram[result.accepted_proposals];

    std::vector<GdnTransitionRecord> records;
    records.reserve(width);
    for (std::uint32_t index = 0; index < width; ++index) {
      const auto token = index < result.licensed_outputs.size()
                             ? result.licensed_outputs[index]
                             : result.target_predictions[index];
      records.push_back(
          transition_record(token, index, impl_->sequence.gdn.geometry()));
    }
    transaction.set_verify_records(std::move(records));
    transaction.set_licensed_outputs(result.licensed_outputs);

    if (options.abort_after_verify) {
      transaction.abort();
      impl_->target.restore_autoregressive(target_before);
      impl_->target_context_features = draft_before;
      result.stage = transaction.stage();
      result.target_state_after = checkpoint_checksum(target_before);
      result.draft_state_after = float_checksum(draft_before);
      ++impl_->stats.aborted_rounds;
      impl_->stats.proposal_tokens += result.proposals.size();
      impl_->stats.accepted_proposals += result.accepted_proposals;
      impl_->stats.correction_rounds += result.used_correction ? 1 : 0;
      impl_->stats.bonus_rounds += result.used_bonus ? 1 : 0;
      impl_->stats.target_weight_publications +=
          result.target_weight_publications;
      impl_->stats.target_verify_calls += result.target_verify_calls;
      impl_->stats.draft_dispatches += result.draft_dispatches;
      return result;
    }

    for (const auto &stop : options.stop_sequences) {
      if (stop.empty()) {
        throw std::invalid_argument("stop token sequences must not be empty");
      }
    }
    auto scan_history = impl_->generated_history;
    auto candidate_tokens = impl_->pending_stop_prefix;
    std::size_t accepted_new = 0;
    std::size_t commit_length = 0;
    PreviewFinishReason finish_reason = PreviewFinishReason::none;
    for (const auto token : result.licensed_outputs) {
      if (accepted_new >= options.remaining_budget) {
        finish_reason = PreviewFinishReason::length;
        break;
      }
      if (std::ranges::find(options.eos_token_ids, token) !=
          options.eos_token_ids.end()) {
        finish_reason = PreviewFinishReason::eos;
        break;
      }
      scan_history.push_back(token);
      candidate_tokens.push_back(token);
      ++accepted_new;
      bool stopped = false;
      for (const auto &stop : options.stop_sequences) {
        if (ends_with(scan_history, stop)) {
          const auto strip = std::min(stop.size(), candidate_tokens.size());
          candidate_tokens.resize(candidate_tokens.size() - strip);
          commit_length =
              accepted_new > stop.size() ? accepted_new - stop.size() : 0;
          finish_reason = PreviewFinishReason::stop;
          stopped = true;
          break;
        }
      }
      if (stopped) {
        break;
      }
    }
    if (finish_reason != PreviewFinishReason::stop) {
      commit_length = accepted_new;
      if (finish_reason == PreviewFinishReason::none &&
          options.remaining_budget != std::numeric_limits<std::size_t>::max() &&
          accepted_new == options.remaining_budget) {
        finish_reason = PreviewFinishReason::length;
      }
    }
    std::vector<std::int32_t> next_pending;
    if (finish_reason == PreviewFinishReason::none &&
        !options.stop_sequences.empty()) {
      const auto holdback = std::min(
          stop_prefix_holdback(scan_history, options.stop_sequences),
          candidate_tokens.size());
      if (holdback > 0) {
        next_pending.assign(candidate_tokens.end() -
                                static_cast<std::ptrdiff_t>(holdback),
                            candidate_tokens.end());
        candidate_tokens.resize(candidate_tokens.size() - holdback);
      }
    }
    result.finish_reason = finish_reason;
    result.published_outputs = std::move(candidate_tokens);
    const auto committed = transaction.commit(commit_length);
    result.stage = transaction.stage();
    if (committed.empty()) {
      impl_->target.restore_autoregressive(target_before);
      impl_->target_context_features = draft_before;
      impl_->pending_stop_prefix = std::move(next_pending);
      result.target_state_after = checkpoint_checksum(target_before);
      result.draft_state_after = float_checksum(draft_before);
      ++impl_->stats.aborted_rounds;
    } else {
      const auto committed_unsigned = to_unsigned(committed);
      std::vector<std::uint32_t> target_replay;
      target_replay.reserve(committed_unsigned.size());
      target_replay.push_back(static_cast<std::uint32_t>(impl_->anchor));
      target_replay.insert(target_replay.end(), committed_unsigned.begin(),
                           committed_unsigned.end() - 1);
      impl_->target.restore_autoregressive(target_before);
      const auto target_mode = target_replay.size() == 1
                                   ? TopologyProfileMode::decode
                                   : TopologyProfileMode::prefill;
      static_cast<void>(impl_->target.execute_ar(
          target_replay, target_mode, QwenArExecutionRoute::execute_batch));
      const auto first_replay = impl_->target.checkpoint_autoregressive();
      impl_->target.restore_autoregressive(target_before);
      static_cast<void>(impl_->target.execute_ar(
          target_replay, target_mode, QwenArExecutionRoute::execute_batch));
      const auto second_replay = impl_->target.checkpoint_autoregressive();
      result.teacher_forced_state_parity =
          first_replay.state == second_replay.state &&
          first_replay.next_position == second_replay.next_position;
      impl_->target_checkpoint = second_replay;
      result.replay_dispatches += 2;
      impl_->target_context_features = draft_before;
      append_target_features(
          impl_->target_context_features,
          std::span(verify.hidden)
              .first(committed_unsigned.size() *
                     impl_->draft.config().hidden_size),
          impl_->draft.config().hidden_size,
          static_cast<std::uint32_t>(
              impl_->draft.config().target_layer_ids.size()));
      impl_->draft_checkpoint = impl_->target_context_features;
      ++result.replay_dispatches;

      impl_->anchor = committed.back();
      impl_->sequence.materialize_pending_context();
      impl_->generated_history.insert(impl_->generated_history.end(),
                                      committed.begin(), committed.end());
      impl_->pending_stop_prefix = std::move(next_pending);
      result.target_state_after = checkpoint_checksum(impl_->target_checkpoint);
      result.draft_state_after = float_checksum(impl_->draft_checkpoint);
      ++impl_->stats.committed_rounds;
    }

    impl_->stats.proposal_tokens += result.proposals.size();
    impl_->stats.accepted_proposals += result.accepted_proposals;
    impl_->stats.correction_rounds += result.used_correction ? 1 : 0;
    impl_->stats.bonus_rounds += result.used_bonus ? 1 : 0;
    impl_->stats.target_weight_publications +=
        result.target_weight_publications;
    impl_->stats.target_verify_calls += result.target_verify_calls;
    impl_->stats.draft_dispatches += result.draft_dispatches;
    impl_->stats.replay_dispatches += result.replay_dispatches;
    return result;
  } catch (...) {
    impl_->target.restore_autoregressive(target_before);
    impl_->target_context_features = draft_before;
    throw;
  }
}

} // namespace omlx::m5
