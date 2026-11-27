#include "chunk_blend_cache.h"
#include "native_backend.h"
#include "round_cost.h"
#include "shared_prefix_cache.h"
#include "silicon_tuning.h"
#include "spec_controller.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "forward_plan.h"
#include "lora_runtime.h"
#include "omlx_m5/artifact.h"
#include "omlx_m5/bounded_batch.h"
#include "omlx_m5/dflash_forward.h"
#include "omlx_m5/dispatch.h"
#include "omlx_m5/execution_contract.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/lora.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/mtp_forward.h"
#include "omlx_m5/qwen_forward.h"
#include "omlx_m5/target_verify_gpu.h"
#include "native_request_controls.h"
#include "omlx_m5/topology_profile.h"

namespace omlx::m5::detail {
using U32 = std::uint32_t;

namespace {

template <class Drafter>
inline void sync_draft_target_ring(Drafter &draft, std::uint32_t ring_rows) {
  if constexpr (requires { draft.set_target_feature_ring(ring_rows); }) {
    draft.set_target_feature_ring(ring_rows);
  } else {
    (void)draft;
    (void)ring_rows;
  }
}

template <class Drafter>
inline void sync_draft_target_epoch(Drafter &draft, std::uint64_t epoch) {
  if constexpr (requires { draft.set_target_feature_epoch(epoch); }) {
    draft.set_target_feature_epoch(epoch);
  } else {
    (void)draft;
    (void)epoch;
  }
}

template <class Drafter>
inline DFlash2DraftResult
invoke_draft_propose(Drafter &draft, std::uint32_t anchor,
                     std::uint32_t context_tokens, std::uint32_t draft_tokens,
                     float temperature, std::uint64_t seed, bool use_icb,
                     bool inspect_tensors, bool cpu_selector,
                     bool tokens_only) {
  if constexpr (requires {
                  draft.propose(anchor, context_tokens, draft_tokens,
                                temperature, seed, use_icb, inspect_tensors,
                                cpu_selector, tokens_only);
                }) {
    return draft.propose(anchor, context_tokens, draft_tokens, temperature,
                         seed, use_icb, inspect_tensors, cpu_selector,
                         tokens_only);
  } else {
    (void)tokens_only;
    return draft.propose(anchor, context_tokens, draft_tokens, temperature,
                         seed, use_icb, inspect_tensors, cpu_selector);
  }
}

template <class Drafter>
inline DFlashPreparedProposal invoke_draft_prepare_propose(
    Drafter &draft, std::uint32_t anchor, std::uint32_t context_tokens,
    std::uint32_t draft_tokens, float temperature, std::uint64_t seed,
    bool use_icb, bool inspect_tensors, bool cpu_selector, bool tokens_only) {
  if constexpr (requires {
                  draft.prepare_propose(anchor, context_tokens, draft_tokens,
                                        temperature, seed, use_icb,
                                        inspect_tensors, cpu_selector,
                                        tokens_only);
                }) {
    return draft.prepare_propose(anchor, context_tokens, draft_tokens,
                                 temperature, seed, use_icb, inspect_tensors,
                                 cpu_selector, tokens_only);
  } else {
    (void)tokens_only;
    return draft.prepare_propose(anchor, context_tokens, draft_tokens,
                                 temperature, seed, use_icb, inspect_tensors,
                                 cpu_selector);
  }
}

template <class Prog>
concept HasPackedBatchAndSample =
    requires(Prog &prog, std::span<const PackedSlotSlice> slices, bool icb) {
      {
        prog.run_packed_batch_and_sample(slices, icb)
      } -> std::same_as<PackedBatchResult>;
    };

template <class Prog>
concept HasPackedBatchScatterFlag =
    requires(Prog &prog, std::span<const PackedSlotSlice> slices, bool icb,
             bool scatter) {
      {
        prog.run_packed_batch(slices, icb, scatter)
      } -> std::same_as<PackedBatchResult>;
    };

template <class Res>
inline std::uint64_t packed_sampling_host_read_bytes(const Res &res) {
  if constexpr (requires { res.sampling_host_read_bytes; }) {
    return std::uint64_t(res.sampling_host_read_bytes);
  } else {
    return 0ULL;
  }
}

template <class Res>
inline std::uint64_t packed_compact_fallback_rows(const Res &res) {
  if constexpr (requires { res.compact_fallback_rows; }) {
    return std::uint64_t(res.compact_fallback_rows);
  } else {
    return 0ULL;
  }
}

template <class Prog>
inline PackedBatchResult invoke_packed_batch_maybe_sample(
    Prog &prog, std::span<const PackedSlotSlice> slices, bool use_icb,
    bool fuse_sample, bool scatter_slot_logits, bool &fused_sampled_out) {
  if constexpr (HasPackedBatchAndSample<Prog>) {
    if (fuse_sample) {
      fused_sampled_out = true;
      return prog.run_packed_batch_and_sample(slices, use_icb);
    }
  }
  fused_sampled_out = false;
  if constexpr (HasPackedBatchScatterFlag<Prog>) {
    return prog.run_packed_batch(slices, use_icb, scatter_slot_logits);
  } else {
    (void)scatter_slot_logits;
    return prog.run_packed_batch(slices, use_icb);
  }
}

std::optional<QwenLinearRouteOverride>
parse_linear_route_ab(const LoadOptions &options,
                      const MetalCapabilities &capabilities) {
  const auto entry = options.path_overrides.find("linear_route_ab");
  if (entry == options.path_overrides.end())
    return {};
  if (capabilities.profile.probe.apple_gpu_family < 9)
    throw std::invalid_argument("linear_route_ab requires Apple9+ GPU");
  std::array<std::string_view, 6> fields;
  std::string_view text = entry->second;
  for (std::size_t i = 0; i < fields.size(); ++i) {
    const auto separator = text.find('|');
    if (i + 1 == fields.size()) {
      if (separator != std::string_view::npos || text.empty())
        throw std::invalid_argument(
            "linear_route_ab requires six exact fields");
      fields[i] = text;
    } else {
      if (separator == std::string_view::npos || separator == 0)
        throw std::invalid_argument(
            "linear_route_ab requires six exact fields");
      fields[i] = text.substr(0, separator);
      text.remove_prefix(separator + 1);
    }
  }
  const auto number = [](std::string_view field) {
    std::uint32_t value = 0;
    const auto [end, error] =
        std::from_chars(field.data(), field.data() + field.size(), value);
    if (error != std::errc{} || end != field.data() + field.size() || !value)
      throw std::invalid_argument(
          "linear_route_ab geometry must be positive U32");
    return value;
  };
  if (fields[5] != "baseline" && fields[5] != "lane" && fields[5] != "matrix")
    throw std::invalid_argument("linear_route_ab choice unsupported");
  return QwenLinearRouteOverride{std::string(fields[0]), std::string(fields[1]),
                                 std::string(fields[5]), number(fields[2]),
                                 number(fields[3]),      number(fields[4])};
}

std::optional<QwenMoeRouteOverride>
parse_moe_route_ab(const LoadOptions &options,
                   const MetalCapabilities &capabilities) {
  const auto entry = options.path_overrides.find("moe_route_ab");
  if (entry == options.path_overrides.end())
    return {};
  if (capabilities.profile.probe.apple_gpu_family < 9)
    throw std::invalid_argument("moe_route_ab requires Apple9+ GPU");
  std::array<std::string_view, 7> fields;
  std::string_view text = entry->second;
  for (std::size_t i = 0; i < fields.size(); ++i) {
    const auto separator = text.find('|');
    if (i + 1 == fields.size()) {
      if (separator != std::string_view::npos || text.empty())
        throw std::invalid_argument("moe_route_ab requires seven fields");
      fields[i] = text;
    } else {
      if (!separator || separator == std::string_view::npos)
        throw std::invalid_argument("moe_route_ab requires seven fields");
      fields[i] = text.substr(0, separator);
      text.remove_prefix(separator + 1);
    }
  }
  const auto number = [](std::string_view field) {
    std::uint32_t value = 0;
    const auto [end, error] =
        std::from_chars(field.data(), field.data() + field.size(), value);
    if (error != std::errc{} || end != field.data() + field.size() || !value)
      throw std::invalid_argument("moe_route_ab geometry invalid");
    return value;
  };
  if (fields[6] != "grouped" && fields[6] != "fallback")
    throw std::invalid_argument("moe_route_ab choice invalid");
  return QwenMoeRouteOverride{std::string(fields[0]), std::string(fields[1]),
                              std::string(fields[2]), std::string(fields[6]),
                              number(fields[3]),      number(fields[4]),
                              number(fields[5])};
}

std::size_t parse_pin_prefix_min_tokens(const LoadOptions &options) {
  const auto it = options.path_overrides.find("pin_prefix_min_tokens");
  if (it == options.path_overrides.end())
    return 0;
  std::size_t tokens = 0;
  const auto &text = it->second;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), tokens);
  if (error != std::errc{} || end != text.data() + text.size())
    throw std::invalid_argument("pin_prefix_min_tokens must be unsigned int");
  return tokens;
}

bool is_false_flag(std::string_view text) noexcept {
  return text == "0" || text == "off" || text == "OFF" || text == "false" ||
         text == "FALSE";
}

bool is_true_flag(std::string_view text) noexcept {
  return text == "1" || text == "on" || text == "ON" || text == "true" ||
         text == "TRUE";
}

bool resolve_rolling_prefix_cache(const LoadOptions &options) {
  if (const auto it = options.path_overrides.find("rolling_prefix_cache");
      it != options.path_overrides.end()) {
    const std::string_view v = it->second;
    if (is_true_flag(v))
      return true;
    if (is_false_flag(v))
      return false;
    throw std::invalid_argument(
        "rolling_prefix_cache must be on/1/true or off/0/false");
  }
  if (const char *env = std::getenv("OMLX_M5_ROLLING_PREFIX_CACHE");
      env && *env) {
    const std::string_view v(env);
    if (is_true_flag(v))
      return true;
    if (is_false_flag(v))
      return false;
    throw std::invalid_argument(
        "OMLX_M5_ROLLING_PREFIX_CACHE must be on/1/true or off/0/false");
  }
  return false;
}

std::optional<std::filesystem::path>
resolve_external_mtp_artifact_path(const LoadOptions &options) {
  if (const auto it = options.path_overrides.find("mtp_artifact");
      it != options.path_overrides.end() && !it->second.empty()) {
    return std::filesystem::path(it->second);
  }
  if (const auto it = options.path_overrides.find("mtp_artifact_path");
      it != options.path_overrides.end() && !it->second.empty()) {
    return std::filesystem::path(it->second);
  }
  return std::nullopt;
}

bool resolve_mtp_enabled(const LoadOptions &options, bool has_embedded_mtp,
                         bool has_external_mtp, bool dflash_speculative) {
  if (dflash_speculative) {
    return false;
  }
  const bool available = has_embedded_mtp || has_external_mtp;
  if (options.speculative.backend == SpeculativeBackend::mtp) {
    if (!available) {
      throw std::invalid_argument(
          "speculative.backend=mtp requested, but checkpoint/artifact has no MTP weights");
    }
    return true;
  }
  if (const auto it = options.path_overrides.find("enable_mtp");
      it != options.path_overrides.end()) {
    const std::string_view v = it->second;
    if (is_false_flag(v)) {
      return false;
    }
    if (is_true_flag(v)) {
      if (!available) {
        throw std::invalid_argument(
            "enable_mtp=1 requested, but checkpoint/artifact has no MTP weights");
      }
      return true;
    }
    if (v != "auto" && v != "AUTO") {
      throw std::invalid_argument(
          "enable_mtp must be auto, on/1/true, or off/0/false");
    }
  }
  if (const char *dis = std::getenv("OMLX_M5_DISABLE_MTP");
      dis && *dis && is_true_flag(dis)) {
    return false;
  }
  if (const char *en = std::getenv("OMLX_M5_ENABLE_MTP"); en && *en) {
    const std::string_view v(en);
    if (is_false_flag(v)) {
      return false;
    }
    if (is_true_flag(v)) {
      if (!available) {
        throw std::invalid_argument(
            "OMLX_M5_ENABLE_MTP=1 set, but checkpoint/artifact has no MTP weights");
      }
      return true;
    }
  }
  return available;
}

} // namespace

std::uint64_t prefix_budget(const LoadOptions &options) {
  const auto found = options.path_overrides.find("prefix_cache_bytes");
  if (found == options.path_overrides.end())
    return 0; // P08 opt-in until target-machine cost qualification.
  std::uint64_t bytes = 0;
  const auto &text = found->second;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), bytes);
  if (error != std::errc{} || end != text.data() + text.size() ||
      bytes > (128ULL << 30))
    throw std::invalid_argument("prefix_cache_bytes must be 0..128GiB");
  return bytes;
}

struct BatchSampleParams {
  std::uint32_t count = 0, context_envelope = 0;
  std::array<std::uint32_t, 8> valid{}, slot{}, width{};
  std::array<detail::forward::SampleParams, 8> sampling{};
};
static_assert(sizeof(BatchSampleParams) == 424);

class NativeBatchSampler final {
public:
  NativeBatchSampler(const MetalContext &context,
                     std::span<QwenForward *const> rows,
                     const std::filesystem::path &libraries,
                     std::uint32_t max_context)
      : context_(context),
        output_(context.make_buffer(8 * sizeof(std::uint32_t))),
        max_context_(max_context) {
    if (rows.size() < 2 || rows.size() > 8)
      throw std::invalid_argument("P10 batch sampler requires B=2..8");
    detail::forward::Encoder e(context, libraries, std::uint32_t(rows.size()),
                               "p10.batch.sample");
    e.begin_stage("sample");
    const auto &get = [&](std::size_t i) -> const MetalBuffer & {
      return rows[std::min(i, rows.size() - 1)]->batch_logits_buffer();
    };
    const auto ref = [&](const MetalBuffer &buffer) {
      return detail::forward::Ref{&buffer, 0, buffer.size()};
    };
    auto &d = e.emit("vocab", "omlx_m5_sample_logits_batched",
                     {.width = 256, .height = rows.size()}, {.width = 256},
                     {ref(get(0)), ref(get(1)), ref(get(2)), ref(get(3)),
                      ref(get(4)), ref(get(5)), ref(get(6)), ref(get(7)),
                      ref(output_)});
    const BatchSampleParams initial{};
    const auto data = std::as_bytes(std::span(&initial, 1));
    e.dynamic(d, 9, std::vector<std::byte>(data.begin(), data.end()));
    profile_ = e.finish();
    if (context_.capabilities().profile.compute_icb_eligible)
      profile_.icb.emplace(context_.prepare_icb_profile(
          profile_.plan.key(), {.plan = &profile_.plan,
                                .bindings = profile_.bindings,
                                .parameters = &profile_.parameters}));
  }
  [[nodiscard]] bool uses_icb() const { return profile_.icb.has_value(); }
  std::pair<std::vector<std::uint32_t>, MetalSubmission>
  sample(const BatchSampleParams &params) {
    if (params.count > 8 || params.count == 0 ||
        params.context_envelope != max_context_)
      throw std::invalid_argument("P10 sampler envelope invalid");
    const auto bytes = std::as_bytes(std::span(&params, 1));
    profile_.parameters.write("step", bytes);
    const std::array calls{
        MetalCommandInvocation{.plan = &profile_.plan,
                               .bindings = profile_.bindings,
                               .parameters = &profile_.parameters}};
    const auto submission = profile_.icb ? context_.replay_icb(*profile_.icb)
                                         : context_.execute_batch(calls);
    const auto data = output_.read(0, params.count * sizeof(std::uint32_t));
    std::vector<std::uint32_t> tokens(params.count);
    std::memcpy(tokens.data(), data.data(), data.size());
    for (std::uint32_t row = 0; row < params.count; ++row) {
      if (params.valid[row] && tokens[row] >= params.sampling[row].vocabulary) {
        throw std::runtime_error("nonfinite or invalid legacy sampled token");
      }
    }
    return {tokens, submission};
  }

private:
  const MetalContext &context_;
  MetalBuffer output_;
  std::uint32_t max_context_;
  detail::forward::Profile profile_;
};

struct SamplingKey {
  std::uint32_t temperature_bits = 0;
  std::uint32_t top_p_bits = 0;
  std::uint32_t min_p_bits = 0;
  std::int32_t top_k = 0;
  std::uint64_t seed = 0;
  bool operator==(const SamplingKey &) const = default;
};

SamplingKey make_sampling_key(const SamplingSpec &sampling) {
  const float temperature =
      sampling.temperature == 0.0F ? 0.0F : sampling.temperature;
  const float top_p = sampling.top_p == 0.0F ? 0.0F : sampling.top_p;
  const float min_p = sampling.min_p == 0.0F ? 0.0F : sampling.min_p;
  return {
      .temperature_bits = std::bit_cast<std::uint32_t>(temperature),
      .top_p_bits = std::bit_cast<std::uint32_t>(top_p),
      .min_p_bits = std::bit_cast<std::uint32_t>(min_p),
      .top_k = sampling.top_k,
      .seed = sampling.seed,
  };
}

class LegacyAnchorSampler final {
public:
  LegacyAnchorSampler(const MetalContext &context, const MetalBuffer &logits,
                      std::uint32_t vocab,
                      const std::filesystem::path &libraries)
      : context_(context), output_(context.make_buffer(sizeof(std::uint32_t))),
        vocab_(vocab) {
    static std::atomic<std::uint64_t> next_id{0};
    detail::forward::Encoder e(context, libraries, 1,
                               "c1.anchor.sample." + std::to_string(++next_id));
    e.begin_stage("sample");
    const detail::forward::Ref logits_ref{&logits, 0, logits.size()};
    const detail::forward::Ref output_ref{&output_, 0, output_.size()};
    auto &d = e.emit("vocab", "omlx_m5_sample_logits", {.width = 256},
                     {.width = 256}, {logits_ref, output_ref});
    offset_ = e.dynamic(
        d, 2, std::vector<std::byte>(sizeof(detail::forward::SampleParams)));
    profile_ = e.finish();
    if (context_.capabilities().profile.compute_icb_eligible)
      profile_.icb.emplace(context_.prepare_icb_profile(
          profile_.plan.key(), {.plan = &profile_.plan,
                                .bindings = profile_.bindings,
                                .parameters = &profile_.parameters}));
  }

  std::uint32_t sample(std::size_t prompt_tokens,
                       const GpuSamplingOptions &sampling) {
    if (!prompt_tokens || sampling.top_k > 64 ||
        (sampling.temperature > 0 && sampling.top_k == 0) ||
        sampling.temperature < 0 || !std::isfinite(sampling.temperature) ||
        !std::isfinite(sampling.top_p) || sampling.top_p < 0 ||
        sampling.top_p > 1)
      throw std::invalid_argument(
          "Qwen forward token/context/sampling request invalid");
    const detail::forward::SampleParams params{
        vocab_,
        sampling.top_k,
        sampling.temperature,
        sampling.top_p,
        sampling.seed,
        sampling.stream,
        std::uint64_t(prompt_tokens - 1)};
    std::memcpy(profile_.parameter_data.data() + offset_, &params,
                sizeof(params));
    profile_.parameters.write("step", profile_.parameter_data);
    if (profile_.icb)
      static_cast<void>(context_.replay_icb(*profile_.icb));
    else {
      const std::array calls{
          MetalCommandInvocation{.plan = &profile_.plan,
                                 .bindings = profile_.bindings,
                                 .parameters = &profile_.parameters}};
      static_cast<void>(context_.execute_batch(calls));
    }
    const auto raw = output_.read(0, sizeof(std::uint32_t));
    std::uint32_t token = 0;
    std::memcpy(&token, raw.data(), sizeof(token));
    if (token >= vocab_) {
      throw std::runtime_error("nonfinite or invalid legacy sampled token");
    }
    return token;
  }

private:
  const MetalContext &context_;
  MetalBuffer output_;
  std::uint32_t vocab_ = 0;
  std::uint64_t offset_ = 0;
  detail::forward::Profile profile_;
};

// #5164 Decode-Protect Anti-Stall Scheduler state:
// Calibrates contended prefill chunks targeting <= 45ms pause, enforces a
// 10 tok/s/stream decode duty-cycle floor with max_bypass_skips=8 starvation
// bound, and designates a progress_owner slot under KV pool pressure.
struct ContendedPrefillScheduler {
  static constexpr double kTargetPrefillPauseMs = 45.0;
  static constexpr double kDecodeFloorTokPerMs = 0.010; // 10 tok/s/stream
  static constexpr std::uint32_t kMaxBypassSkips = 8;

  double ema_prefill_ms_per_tok = 0.38; // ~390ms / 1024 tok initial prior
  double ema_decode_tokens_per_step = 1.0;
  std::uint32_t decode_debt_rounds = 0;
  std::vector<std::uint32_t> bypass_skips;
  std::optional<std::size_t> progress_owner_slot;

  explicit ContendedPrefillScheduler(std::size_t max_slots = 1,
                                     std::uint32_t hidden_size = 5120)
      : bypass_skips(std::max<std::size_t>(1, max_slots), 0u) {
    if (hidden_size < 1024)
      ema_prefill_ms_per_tok = 0.01;
  }

  [[nodiscard]] std::size_t calibrated_contended_budget() const noexcept {
    const double raw = kTargetPrefillPauseMs / std::max(0.005, ema_prefill_ms_per_tok);
    if (raw >= 896.0)
      return 1024;
    if (raw >= 448.0)
      return 512;
    if (raw >= 224.0)
      return 256;
    return 128;
  }

  void observe_prefill_chunk(std::size_t tokens, double ms,
                             bool had_active_decodes) noexcept {
    if (tokens > 0 && ms > 0.01 && std::isfinite(ms)) {
      const double per_tok = ms / double(tokens);
      ema_prefill_ms_per_tok = 0.75 * ema_prefill_ms_per_tok + 0.25 * per_tok;
    }
    if (had_active_decodes && ms > kTargetPrefillPauseMs) {
      const double needed_toks = ms * kDecodeFloorTokPerMs;
      const double per_round = std::max(1.0, ema_decode_tokens_per_step);
      const auto debt = std::uint32_t(std::ceil(needed_toks / per_round));
      decode_debt_rounds = std::clamp<std::uint32_t>(debt, 1u, 6u);
    }
  }

  void observe_decode_round(double avg_tokens_per_stream) noexcept {
    if (avg_tokens_per_step_valid(avg_tokens_per_stream)) {
      ema_decode_tokens_per_step =
          0.80 * ema_decode_tokens_per_step + 0.20 * avg_tokens_per_stream;
    }
    if (decode_debt_rounds > 0)
      --decode_debt_rounds;
  }

  [[nodiscard]] bool should_defer_prefill(std::size_t slot) noexcept {
    if (slot < bypass_skips.size() && bypass_skips[slot] >= kMaxBypassSkips) {
      bypass_skips[slot] = 0;
      return false; // Starvation cap reached: service prefill now.
    }
    if (progress_owner_slot.has_value() && *progress_owner_slot != slot) {
      if (slot < bypass_skips.size())
        ++bypass_skips[slot];
      return true;
    }
    if (decode_debt_rounds > 0) {
      if (slot < bypass_skips.size())
        ++bypass_skips[slot];
      return true;
    }
    if (slot < bypass_skips.size())
      bypass_skips[slot] = 0;
    return false;
  }

private:
  [[nodiscard]] static bool avg_tokens_per_step_valid(double v) noexcept {
    return v >= 0.5 && std::isfinite(v);
  }
};

std::optional<std::pair<std::uint32_t, std::uint32_t>>
resolve_think_token_ids(const std::filesystem::path &model_dir,
                        const QwenModelConfig &config,
                        const LoadOptions &options) {
  auto parse_u32 = [](std::string_view sv) -> std::optional<std::uint32_t> {
    std::uint32_t val = 0;
    const auto *beg = sv.data();
    const auto *end = beg + sv.size();
    const auto [ptr, ec] = std::from_chars(beg, end, val);
    if (ec == std::errc{} && ptr == end)
      return val;
    return std::nullopt;
  };
  const auto it_open = options.path_overrides.find("think_open_token_id");
  const auto it_close = options.path_overrides.find("think_close_token_id");
  if (it_open != options.path_overrides.end() &&
      it_close != options.path_overrides.end()) {
    const auto o = parse_u32(it_open->second);
    const auto c = parse_u32(it_close->second);
    if (o.has_value() && c.has_value() && *o != *c)
      return std::make_pair(*o, *c);
  }
  const auto tok_cfg_path = model_dir / "tokenizer_config.json";
  if (std::filesystem::is_regular_file(tok_cfg_path)) {
    try {
      std::ifstream in(tok_cfg_path);
      if (in.good()) {
        const auto j = nlohmann::json::parse(in, nullptr, false);
        if (j.is_object() && j.contains("added_tokens_decoder") &&
            j["added_tokens_decoder"].is_object()) {
          std::optional<std::uint32_t> open_id, close_id;
          for (const auto &[k, v] : j["added_tokens_decoder"].items()) {
            if (!v.is_object() || !v.contains("content") ||
                !v["content"].is_string())
              continue;
            const auto &content = v["content"].get_ref<const std::string &>();
            if (content == "<think>")
              open_id = parse_u32(k);
            else if (content == "</think>")
              close_id = parse_u32(k);
          }
          if (open_id.has_value() && close_id.has_value() &&
              *open_id != *close_id &&
              *open_id < std::uint32_t(config.vocab_size) &&
              *close_id < std::uint32_t(config.vocab_size)) {
            return std::make_pair(*open_id, *close_id);
          }
        }
      }
    } catch (...) {
    }
  }
  if (config.vocab_size >= 248070)
    return std::make_pair(248068U, 248069U);
  if (config.vocab_size >= 151669 && config.vocab_size < 160000)
    return std::make_pair(151667U, 151668U);
  return std::nullopt;
}

class NativeQwenState {
public:
  NativeQwenState(const std::filesystem::path &model_dir,
                  const std::filesystem::path &artifact_path,
                  const std::filesystem::path &libraries,
                  std::uint32_t max_context, const LoadOptions &options)
      : cache_budget_bytes(prefix_budget(options)),
        pin_prefix_min_tokens(parse_pin_prefix_min_tokens(options)),
        prefix_cache(prefix_budget(options)),
        context(MetalContext::create()),
        config(load_qwen_model_config(model_dir / "config.json")) {
    sampling_contract = parse_sampling_contract(options.sampling_contract);
    const std::string effective_lora_mode =
        options.lora_mode.empty() ? std::string("merge") : options.lora_mode;
    if (effective_lora_mode != "merge" && effective_lora_mode != "bypass") {
      throw std::invalid_argument("lora_mode must be merge or bypass");
    }
    rolling_prefix_cache = resolve_rolling_prefix_cache(options);
    think_token_ids = resolve_think_token_ids(model_dir, config, options);
    if (options.max_concurrency < 1 || options.max_concurrency > 8)
      throw std::invalid_argument("max_concurrency must be in [1, 8]");
    if (rolling_prefix_cache) {
      const std::size_t per_slot_mult = think_token_ids.has_value() ? 4U : 2U;
      const std::size_t min_cap = think_token_ids.has_value() ? 16U : 8U;
      prefix_cache.set_max_unpinned_entries(std::max<std::size_t>(
          min_cap, per_slot_mult * std::size_t(options.max_concurrency)));
    }
    occupied.assign(std::size_t(options.max_concurrency), false);
    slot_dflash_prefix.assign(std::size_t(options.max_concurrency), {});
    slot_mtp_think_frontier.assign(std::size_t(options.max_concurrency), {});
    slot_mtp_post_frontier.assign(std::size_t(options.max_concurrency), {});
    slot_rolling_think_anchor_ids.assign(std::size_t(options.max_concurrency),
                                         0ULL);
    slot_rolling_post_ids.assign(std::size_t(options.max_concurrency), 0ULL);
    slot_request_samplers.resize(std::size_t(options.max_concurrency));
    slot_batch_enabled = (options.max_concurrency >= 2);
    if (const auto it = options.path_overrides.find("batch_mode");
        it != options.path_overrides.end()) {
      if (it->second != "on" && it->second != "off")
        throw std::invalid_argument("batch_mode must be on or off");
      if (it->second == "on" && options.max_concurrency == 1)
        throw std::invalid_argument("batch_mode=on requires B>=2");
      slot_batch_enabled = it->second == "on";
    }
    if (const auto entry =
            options.path_overrides.find("batch_projection_prototype");
        entry != options.path_overrides.end()) {
      if (entry->second != "on" && entry->second != "off")
        throw std::invalid_argument(
            "batch_projection_prototype must be on or off");
      shared_head_enabled = entry->second == "on";
      if (shared_head_enabled &&
          (!slot_batch_enabled ||
           sampling_contract != SamplingContract::PositionCoupledV1))
        throw std::invalid_argument(
            "shared head prototype requires position-coupled batch mode");
    }
    const auto &capabilities = context.capabilities();
    const bool apple9_or_newer =
        capabilities.profile.probe.apple_gpu_family >= 9;
    bool enable_m4_short_routes = capabilities.device_name == "Apple M4";
    if (const auto it = options.path_overrides.find("m4_short_routes");
        it != options.path_overrides.end()) {
      if (it->second != "auto" && it->second != "on" && it->second != "off")
        throw std::invalid_argument(
            "m4_short_routes must be auto, on, or off");
      if (it->second == "on" && !apple9_or_newer)
        throw std::invalid_argument("m4_short_routes=on requires Apple9+ GPU");
      if (it->second != "auto")
        enable_m4_short_routes = it->second == "on";
    }
    const auto diagnostic_route = parse_linear_route_ab(options, capabilities);
    const auto moe_route = parse_moe_route_ab(options, capabilities);
    bool enable_attention_split =
        apple9_or_newer && config.hidden_size >= 1024 &&
        config.full_attention_layers() > 0;
    if (const auto it = options.path_overrides.find("attention_split_kv");
        it != options.path_overrides.end()) {
      if (it->second != "on" && it->second != "off")
        throw std::invalid_argument("attention_split_kv must be on or off");
      if (it->second == "on" && !apple9_or_newer)
        throw std::invalid_argument(
            "attention_split_kv=on requires Apple9+ GPU");
      enable_attention_split = it->second == "on";
    }
    bool enable_gpu_verify = apple9_or_newer;
    if (const auto it = options.path_overrides.find("target_verify_selector");
        it != options.path_overrides.end()) {
      if (it->second != "cpu" && it->second != "gpu")
        throw std::invalid_argument(
            "target_verify_selector must be cpu or gpu");
      if (it->second == "gpu" && !apple9_or_newer)
        throw std::invalid_argument(
            "target_verify_selector=gpu requires Apple9+ GPU");
      enable_gpu_verify = it->second == "gpu";
    }
    if (sampling_contract == SamplingContract::PositionCoupledV1) {
      if (options.path_overrides.contains("target_verify_selector") &&
          !enable_gpu_verify)
        throw std::invalid_argument(
            "position-coupled verification uses the same GPU/compact-fallback "
            "sampler as AR; legacy CPU selector is a separate contract");
      enable_gpu_verify = true;
    }

    std::optional<ArtifactModel> artifact(
        ArtifactModel::open(artifact_path, false));
    const bool speculative =
        options.speculative.backend == SpeculativeBackend::dflash2;
    const auto ext_mtp_path = resolve_external_mtp_artifact_path(options);
    const bool embedded_mtp_present = has_mtp_weights(*artifact);
    const bool mtp_enabled = resolve_mtp_enabled(
        options, embedded_mtp_present, ext_mtp_path.has_value(), speculative);
    const auto mtp_depth_policy =
        mtp_enabled
            ? resolve_mtp_depth_policy(
                  options, std::uint32_t(config.hidden_size),
                  MtpForward::kMaxDraftTokens,
                  MtpForward::kDefaultMaxDraftTokens)
            : MtpDepthPolicy{.max_draft_tokens = 0U, .fixed_depth = true};
    const std::uint32_t mtp_draft_tokens = mtp_depth_policy.max_draft_tokens;
    const bool any_speculative = speculative || mtp_enabled;

    std::optional<ArtifactModel> ext_mtp_artifact;
    std::optional<ModelMaterializationPlan> ext_mtp_materialization;
    if (mtp_enabled && ext_mtp_path.has_value()) {
      ext_mtp_artifact.emplace(ArtifactModel::open(*ext_mtp_path, false));
      if (!has_mtp_weights(*ext_mtp_artifact)) {
        throw std::invalid_argument("mtp_artifact contains no ModelComponent::mtp weights");
      }
      ext_mtp_materialization.emplace(plan_model_materialization(
          *ext_mtp_artifact, context,
          {.components = {ModelComponent::mtp},
           .require_verified_payload = false}));
    }

    const auto silicon_tuning = resolve_silicon_tuning(context);
    DFlash2ModelConfig draft_config;
    std::filesystem::path draft_path;
    std::optional<ArtifactModel> draft_artifact;
    std::optional<ModelMaterializationPlan> draft_materialization;
    std::uint32_t effective_dflash_draft_tokens =
        options.speculative.draft_tokens;
    if (speculative) {
      draft_config = load_dflash2_model_config(
          std::filesystem::path(options.draft_dir) / "config.json");
      if (effective_dflash_draft_tokens == 0u) {
        const std::uint32_t block_cap =
            draft_config.block_size > 1u ? (draft_config.block_size - 1u) : 1u;
        const std::uint32_t upper = std::min<std::uint32_t>(7u, block_cap);
        effective_dflash_draft_tokens =
            std::clamp<std::uint32_t>(silicon_tuning.dflash_draft_tokens, 1u, upper);
      }
      if (draft_config.hidden_size != std::uint32_t(config.hidden_size) ||
          draft_config.vocabulary_size != std::uint32_t(config.vocab_size) ||
          max_context > draft_config.max_position_embeddings ||
          effective_dflash_draft_tokens + 1 > draft_config.block_size ||
          std::ranges::any_of(draft_config.target_layer_ids, [&](auto id) {
            return id >= std::uint32_t(config.num_hidden_layers);
          }))
        throw std::invalid_argument(
            "DFlash companion/target geometry mismatch");
      const auto found = options.path_overrides.find("draft_artifact");
      if (found != options.path_overrides.end()) {
        draft_path = std::filesystem::path(found->second);
      } else {
        const std::filesystem::path draft_base(options.draft_dir);
        std::error_code draft_ec;
        draft_path =
            std::filesystem::is_regular_file(draft_base / "model.miv", draft_ec) ||
                    !std::filesystem::is_regular_file(draft_base / "model.omlxm5",
                                                      draft_ec)
                ? draft_base / "model.miv"
                : draft_base / "model.omlxm5";
      }
      draft_artifact.emplace(ArtifactModel::open(draft_path, false));
      static_cast<void>(
          validate_dflash2_artifact(*draft_artifact, draft_config, false));
      draft_materialization.emplace(plan_model_materialization(
          *draft_artifact, context,
          {.components = {ModelComponent::speculative},
           .require_verified_payload = false}));
    }
    slot_draft_config = draft_config;
    slot_short_routes = enable_m4_short_routes;
    slot_speculative = speculative;
    slot_dflash_draft_tokens = effective_dflash_draft_tokens;
    slot_mtp_enabled = mtp_enabled;
    slot_mtp_draft_tokens = mtp_draft_tokens;
    slot_gpu_verify = enable_gpu_verify;
    slot_attention_split = enable_attention_split;
    slot_route_override = diagnostic_route;
    slot_moe_override = moe_route;
    slot_libraries = libraries;
    const KvStorageFormat kv_format = parse_kv_storage_format(options.kv_dtype);
    slot_kv_dtype = std::string(kv_storage_format_name(kv_format));

    const bool affine4_kv = (kv_format == KvStorageFormat::Affine4Group128);
    const bool coupled =
        sampling_contract == SamplingContract::PositionCoupledV1;
    const auto flash_nax_override =
        options.path_overrides.find("flash_nax_prefill");
    const bool apple10_nax =
        (capabilities.device_name.starts_with("Apple M5") ||
         capabilities.profile.probe.apple_gpu_family >= 10) &&
        capabilities.profile.probe.runtime_os_major >= 27 &&
        capabilities.profile.probe.metal4_compiler &&
        capabilities.profile.probe.tensor_api;
    const bool real_model_prefill_eligible =
        config.hidden_size >= 1024 && config.full_attention_layers() > 0 &&
        (affine4_kv
             ? (config.head_dim == 256 &&
                std::filesystem::is_regular_file(
                    libraries / "omlx_m5_attention_affine4.metallib"))
             : ((config.head_dim == 256 || config.head_dim == 128 ||
                 config.head_dim == 64) &&
                std::filesystem::is_regular_file(
                    libraries / "omlx_m5_attention_nax_prefill.metallib")));
    const bool flash_nax_env_off = [&] {
      const char *v = std::getenv("OMLX_M5_DISABLE_FLASH_NAX_PREFILL");
      if (!v)
        return false;
      const std::string_view sv(v);
      return sv == "1" || sv == "true" || sv == "TRUE" || sv == "on" ||
             sv == "ON";
    }();
    const bool flash_nax_explicit_off =
        flash_nax_env_off ||
        (flash_nax_override != options.path_overrides.end() &&
         (flash_nax_override->second == "0" ||
          flash_nax_override->second == "off" ||
          flash_nax_override->second == "false"));
    const bool flash_nax_explicit_on =
        options.prefill_engine == "flash_nax" ||
        (flash_nax_override != options.path_overrides.end() &&
         !flash_nax_explicit_off);
    const bool enable_flash_nax_prefill =
        !flash_nax_explicit_off &&
        (config.hidden_size >= 1024
             ? (real_model_prefill_eligible &&
                (apple10_nax || flash_nax_explicit_on))
             : (!coupled && flash_nax_explicit_on));
    const auto primary_ws_rows = resolve_qwen_primary_workspace_rows(
        max_context, false, config.num_experts > 1);
    this->primary_ws_rows = primary_ws_rows;
    layers = build_qwen_layer_catalog(*artifact, config, max_context, 64);
    {
      std::uint32_t emb_obj = UINT32_MAX;
      std::uint32_t head_obj = UINT32_MAX;
      for (const auto &st : layers.global_stages()) {
        if (st.role == "embedding")
          emb_obj = st.object_id;
        if (st.role == "lm_head")
          head_obj = st.object_id;
      }
      lm_head_tied = (emb_obj != UINT32_MAX && emb_obj == head_obj);
    }
    std::optional<LoadedLoraAdapter> loaded_lora;
    std::uint64_t est_lora_bytes = 0;
    if (!options.lora_adapter.empty()) {
      loaded_lora = load_lora_adapter_for_qwen(options.lora_adapter, config,
                                               lm_head_tied);
      if (shared_head_enabled) {
        for (const auto &u : loaded_lora->updates) {
          if (u.target.is_global && u.target.role == "lm_head") {
            throw std::invalid_argument(
                "batch_projection_prototype=on does not support lm_head LoRA "
                "adaptation");
          }
        }
      }
      est_lora_bytes = detail::lora_rt::estimate_lora_gpu_bytes(
          *loaded_lora, effective_lora_mode, primary_ws_rows,
          std::uint32_t(options.max_concurrency));
    }
    {
      bool saw_mxfp4 = false;
      bool saw_affine = false;
      bool saw_bf16 = false;
      bool saw_down_fwht64 =
          ((artifact->summary().container_flags & kArtifactFlagFwht64) != 0U);
      for (const auto &ls : layers.layers()) {
        for (const auto &st : ls.stages) {
          if (st.kind != LayerStageKind::linear || st.object_id == UINT32_MAX ||
              st.role == "moe_router" || st.role == "moe_shared_gate") {
            continue;
          }
          if (st.format == "mxfp4_e2m1_g32_e8m0_v1") {
            saw_mxfp4 = true;
          } else if (st.format.starts_with("oq_affine_q")) {
            saw_affine = true;
          } else if (st.format == "direct_bf16_v1") {
            saw_bf16 = true;
          }
          if (st.role == "mlp_down" || st.role == "moe_expert_down") {
            if (artifact->object(st.object_id).has_fwht64()) {
              saw_down_fwht64 = true;
            }
          }
        }
      }
      if (saw_mxfp4 && !saw_affine && !saw_bf16) {
        model_quant_format = "mxfp4";
      } else if (saw_affine && !saw_mxfp4 && !saw_bf16) {
        model_quant_format = "oq_affine";
      } else if (saw_bf16 && !saw_mxfp4 && !saw_affine) {
        model_quant_format = "bf16";
      } else if (saw_mxfp4 || saw_affine || saw_bf16) {
        model_quant_format = "mixed";
      }
      effective_linear_mode = options.prefill_linear_mode;
      if (options.prefill_linear_mode == PrefillLinearMode::W4A4) {
        const bool allow_no_fwht64 = []() {
          const char *v = std::getenv("OMLX_M5_W4A4_ALLOW_NO_FWHT64");
          return v != nullptr && std::string_view(v) == "1";
        }();
        const bool w4a4_lib_ready =
            apple10_nax &&
            std::filesystem::is_regular_file(libraries /
                                             "omlx_m5_w4a4_nax.metallib");
        const bool w4a8_lib_ready =
            apple10_nax &&
            std::filesystem::is_regular_file(libraries /
                                             "omlx_m5_w4a8_nax.metallib");
        if (model_quant_format != "mxfp4") {
          prefill_linear_fallback = true;
          prefill_linear_fallback_reason = "requires_mxfp4_model";
          effective_linear_mode =
              (w4a8_lib_ready && (model_quant_format == "oq_affine" ||
                                  model_quant_format == "mixed"))
                  ? PrefillLinearMode::W4A8
                  : PrefillLinearMode::BFloat16;
        } else if (!saw_down_fwht64 && !allow_no_fwht64) {
          prefill_linear_fallback = true;
          prefill_linear_fallback_reason = "requires_mxfp4_v2_fwht64";
          effective_linear_mode = w4a8_lib_ready ? PrefillLinearMode::W4A8
                                                 : PrefillLinearMode::BFloat16;
        } else if (!w4a4_lib_ready) {
          prefill_linear_fallback = true;
          prefill_linear_fallback_reason = "nax_unavailable";
          effective_linear_mode = w4a8_lib_ready ? PrefillLinearMode::W4A8
                                                 : PrefillLinearMode::BFloat16;
        }
      }
    }
    const bool w4a4_keep_qk_a8 = []() {
      const char *v = std::getenv("OMLX_M5_W4A4_KEEP_QK_A8");
      if (!v)
        return false;
      const std::string_view sv(v);
      return sv == "1" || sv == "true" || sv == "TRUE" || sv == "on" ||
             sv == "ON";
    }();
    const bool enable_w4a4_ws =
        (effective_linear_mode == PrefillLinearMode::W4A4);
    const bool enable_w4a8_ws =
        (effective_linear_mode == PrefillLinearMode::W4A8 ||
         (enable_w4a4_ws && w4a4_keep_qk_a8));
    const bool enable_minf_ws =
        (options.prefill_attention_mode == PrefillAttentionMode::Sage2 &&
         options.prefill_sparse_minference &&
         config.full_attention_layers() > 0 && config.head_dim == 256);
    auto memory = plan_qwen_forward_memory(
        config, max_context,
        std::uint32_t(draft_config.target_layer_ids.size()), false,
        any_speculative, enable_attention_split, coupled, coupled, false, 0u,
        primary_ws_rows, affine4_kv, 0u, enable_flash_nax_prefill,
        enable_w4a8_ws, enable_minf_ws, false, enable_w4a4_ws);
    std::uint64_t per_slot_extra_ws = 0;
    std::uint64_t per_slot_extra_state = 0;
    std::uint64_t companion_weights = 0;
    if (speculative) {
      auto draft_memory = plan_dflash_forward_memory(draft_config, max_context);
      per_slot_extra_ws += draft_memory.workspace_bytes;
      memory.largest_buffer_bytes = std::max(memory.largest_buffer_bytes,
                                             draft_memory.largest_buffer_bytes);
      companion_weights = draft_materialization->resident_bytes;
      if (enable_gpu_verify) {
        const auto verify_bytes = target_gpu_verify_workspace_bytes(
            std::uint32_t(config.vocab_size), sampling_contract);
        per_slot_extra_ws += verify_bytes;
        memory.largest_buffer_bytes =
            std::max(memory.largest_buffer_bytes, verify_bytes);
      }
      per_slot_extra_ws +=
          (2ULL * draft_config.vocabulary_size + draft_config.hidden_size) *
          draft_config.selector_rank * 4;
    } else if (mtp_enabled) {
      auto mtp_memory =
          plan_mtp_forward_memory(config, max_context, affine4_kv);
      per_slot_extra_ws += mtp_memory.workspace_bytes;
      per_slot_extra_state += mtp_memory.state_bytes;
      memory.largest_buffer_bytes = std::max(memory.largest_buffer_bytes,
                                             mtp_memory.largest_buffer_bytes);
      if (ext_mtp_materialization.has_value()) {
        companion_weights = ext_mtp_materialization->resident_bytes;
      }
      if (enable_gpu_verify) {
        const auto verify_bytes = target_gpu_verify_workspace_bytes(
            std::uint32_t(config.vocab_size), sampling_contract);
        per_slot_extra_ws += verify_bytes;
        memory.largest_buffer_bytes =
            std::max(memory.largest_buffer_bytes, verify_bytes);
      }
    }
    std::vector<ModelComponent> main_components{ModelComponent::text};
    if (mtp_enabled && !ext_mtp_path.has_value()) {
      main_components.push_back(ModelComponent::mtp);
    }
    const auto planned_weights =
        plan_model_materialization(*artifact, context,
                                   {.components = main_components,
                                    .require_verified_payload = false});

    const std::uint32_t logical_kv_pages = (max_context + 63u) / 64u;
    const std::uint64_t kv_page_b =
        config.full_attention_layers() > 0
            ? kv_page_bytes(kv_format, 64u,
                            std::uint32_t(config.num_key_value_heads),
                            std::uint32_t(config.head_dim))
            : 0ULL;
    const std::uint64_t kv_projs =
        2ULL * std::uint64_t(config.full_attention_layers());
    const std::uint64_t per_page_all_layers = kv_page_b * kv_projs;

    if (options.max_concurrency > 1) {
      // Secondary slots share slot[0]'s 1024-row prefill workspace AND slot[0]'s
      // physical KV slabs (allocating only 16-row decode/verify scratch + GDN state).
      const auto secondary = plan_qwen_forward_memory(
          config, max_context,
          std::uint32_t(draft_config.target_layer_ids.size()), false,
          any_speculative, enable_attention_split, coupled, coupled, false, 0u,
          16u, affine4_kv, 0u, enable_flash_nax_prefill);
      const auto extra_count = std::uint64_t(options.max_concurrency - 1);
      memory.workspace_bytes +=
          per_slot_extra_ws +
          extra_count * (secondary.workspace_bytes + per_slot_extra_ws);
      const std::uint64_t secondary_kv_slabs =
          std::uint64_t(logical_kv_pages + 1u) * per_page_all_layers;
      const std::uint64_t secondary_state_only =
          secondary.state_bytes >= secondary_kv_slabs
              ? (secondary.state_bytes - secondary_kv_slabs)
              : 0ULL;
      memory.state_bytes +=
          per_slot_extra_state +
          extra_count * (secondary_state_only + per_slot_extra_state);
    } else {
      memory.workspace_bytes += per_slot_extra_ws;
      memory.state_bytes += per_slot_extra_state;
    }
    if (slot_batch_enabled && coupled) {
      const auto slots = std::uint64_t(options.max_concurrency),
                 vocab = std::uint64_t(config.vocab_size),
                 hidden = std::uint64_t(config.hidden_size);
      memory.workspace_bytes +=
          slots * vocab * 2 + slots * (8 + 64 * 8 + 64 * 16);
      if (shared_head_enabled) {
        const auto rows = slots * 8;
        memory.workspace_bytes +=
            rows * (hidden + vocab) * 2 + rows * (hidden / 32) * 4 + 4;
        memory.largest_buffer_bytes =
            std::max(memory.largest_buffer_bytes, rows * vocab * 2);
      }
    }

    const std::uint64_t rec_ws_bytes =
        context.capabilities().recommended_working_set_bytes;
    const std::uint64_t effective_ws_limit =
        (options.memory_budget_bytes > 0)
            ? std::min<std::uint64_t>(rec_ws_bytes, options.memory_budget_bytes)
            : rec_ws_bytes;
    const auto base_budget = qwen_load_budget(
        planned_weights.resident_bytes + companion_weights + est_lora_bytes,
        memory, effective_ws_limit, context.current_allocated_bytes());
    const std::uint64_t base_required_bytes =
        base_budget.required_bytes +
        std::uint64_t(options.max_concurrency - 1) *
            base_budget.runtime_reserve_bytes;

    std::uint32_t shared_kv_pages = 0u;
    {
      const std::uint64_t cache_budget = prefix_budget(options);
      std::uint64_t override_kv_tok = 0;
      if (const auto kv_it = options.path_overrides.find("kv_pool_tokens");
          kv_it != options.path_overrides.end()) {
        override_kv_tok = std::strtoull(kv_it->second.c_str(), nullptr, 10);
      } else if (const char *env_kv = std::getenv("OMLX_M5_KV_POOL_TOKENS")) {
        override_kv_tok = std::strtoull(env_kv, nullptr, 10);
      }
      if (override_kv_tok > 0) {
        shared_kv_pages = std::max<std::uint32_t>(
            logical_kv_pages + 1u,
            std::uint32_t((override_kv_tok + 63ULL) / 64ULL));
      } else {
        if (options.max_concurrency > 1) {
          const auto b_slots = std::uint32_t(options.max_concurrency);
          shared_kv_pages = std::min<std::uint32_t>(
              logical_kv_pages * b_slots + b_slots,
              std::max<std::uint32_t>(logical_kv_pages + 8u * b_slots, 1088u));
        }
        // Size physical KvPagePool arena for UMA resident prefix caching even
        // when max_concurrency == 1 (zero extra overhead when prefix_cache_bytes == 0).
        if (cache_budget > 0 && per_page_all_layers > 0ULL) {
          const std::uint64_t budget_pages = cache_budget / per_page_all_layers;
          const std::uint64_t avail_ws =
              (base_budget.available_bytes > base_required_bytes)
                  ? (base_budget.available_bytes - base_required_bytes)
                  : 0ULL;
          const std::uint64_t ws_cap_pages =
              (avail_ws / 2ULL) / per_page_all_layers;
          const std::uint64_t extra_cap = std::min<std::uint64_t>(
              std::max<std::uint64_t>(4ULL * std::uint64_t(logical_kv_pages),
                                      64ULL),
              ws_cap_pages);
          const std::uint64_t extra_pages =
              std::min<std::uint64_t>(budget_pages, extra_cap);
          if (extra_pages > 0) {
            const std::uint64_t desired =
                std::uint64_t(logical_kv_pages) + 1ULL + extra_pages;
            const std::uint32_t desired_u32 = std::uint32_t(
                std::min<std::uint64_t>(desired, UINT32_MAX - 1));
            shared_kv_pages =
                std::max<std::uint32_t>(shared_kv_pages, desired_u32);
          }
        }
      }
      if (shared_kv_pages > logical_kv_pages + 1u && kv_page_b > 0ULL) {
        const auto max_safe =
            std::min<std::uint64_t>(context.capabilities().max_buffer_length,
                                    8ULL << 30) /
            kv_page_b;
        if (max_safe >= logical_kv_pages + 1u) {
          shared_kv_pages = std::min<std::uint32_t>(
              shared_kv_pages, std::uint32_t(max_safe));
        }
      }
    }
    if (shared_kv_pages > logical_kv_pages + 1u && per_page_all_layers > 0ULL) {
      memory.state_bytes +=
          std::uint64_t(shared_kv_pages - (logical_kv_pages + 1u)) *
          per_page_all_layers;
      memory.largest_buffer_bytes = std::max(
          memory.largest_buffer_bytes,
          std::uint64_t(shared_kv_pages) * kv_page_b);
    }
    if (mtp_enabled && prefix_cache.enabled()) {
      // Enable CoW shared MTP KV arena for BOTH B=1 (multi-conversation /
      // pinned-prefix retention) and B>=2 (cross-slot sharing), sized to cover
      // all active slots plus target prefix-cache headroom so MTP pages do not
      // starve before target KV pages.
      const auto slots = std::uint64_t(options.max_concurrency);
      const std::uint64_t logical_pages = std::uint64_t(logical_kv_pages);
      const std::uint64_t mtp_page_bytes = kv_page_bytes(
          kv_format, 64U, std::uint32_t(config.num_key_value_heads),
          std::uint32_t(config.head_dim));
      const std::uint64_t min_shared_pages = slots * (logical_pages + 1ULL);
      const std::uint64_t target_extra_pages =
          (shared_kv_pages > logical_kv_pages + 1u)
              ? std::uint64_t(shared_kv_pages - (logical_kv_pages + 1u))
              : 0ULL;
      const auto buffer_limit = std::min<std::uint64_t>(
          context.capabilities().max_buffer_length, 8ULL << 30);
      if (mtp_page_bytes > 0ULL && min_shared_pages <= buffer_limit / mtp_page_bytes) {
        const std::uint64_t max_buf_pages = buffer_limit / mtp_page_bytes;
        const std::uint64_t target_added_bytes =
            (shared_kv_pages > logical_kv_pages + 1u && per_page_all_layers > 0ULL)
                ? std::uint64_t(shared_kv_pages - (logical_kv_pages + 1u)) *
                      per_page_all_layers
                : 0ULL;
        const std::uint64_t used_before_mtp_extra =
            base_required_bytes + target_added_bytes;
        const std::uint64_t avail_for_mtp =
            base_budget.available_bytes > used_before_mtp_extra
                ? (base_budget.available_bytes - used_before_mtp_extra)
                : 0ULL;
        const std::uint64_t baseline_mtp_pages = slots * logical_pages;
        const std::uint64_t max_ws_pages =
            baseline_mtp_pages + avail_for_mtp / (2ULL * mtp_page_bytes);
        const std::uint64_t desired_pages = std::min<std::uint64_t>(
            {min_shared_pages + target_extra_pages, max_buf_pages,
             std::max(min_shared_pages, max_ws_pages)});
        if (desired_pages >= min_shared_pages && desired_pages <= UINT32_MAX - 1) {
          mtp_shared_kv_pages = std::uint32_t(desired_pages);
          memory.state_bytes +=
              2ULL * (desired_pages - baseline_mtp_pages) * mtp_page_bytes;
          memory.largest_buffer_bytes = std::max(
              memory.largest_buffer_bytes, desired_pages * mtp_page_bytes);
        }
      }
    }
    if (memory.largest_buffer_bytes > context.capabilities().max_buffer_length)
      throw std::invalid_argument("Qwen planned buffer exceeds device limit");
    auto budget = qwen_load_budget(
        planned_weights.resident_bytes + companion_weights + est_lora_bytes,
        memory, effective_ws_limit, context.current_allocated_bytes());
    budget.required_bytes +=
        std::uint64_t(options.max_concurrency - 1) *
        budget.runtime_reserve_bytes; // per-slot ICB/pipeline parameter room
    budget.fits = budget.required_bytes <= budget.available_bytes;
    if (!budget.fits)
      throw std::runtime_error(
          "native Qwen load exceeds Metal working-set budget: required=" +
          std::to_string(budget.required_bytes) +
          " available=" + std::to_string(budget.available_bytes));

    artifact->verify_payload();
    auto materialization =
        plan_model_materialization(*artifact, context,
                                   {.components = main_components,
                                    .require_verified_payload = true});
    const auto topology =
        describe_qwen_topology(*artifact, config, layers, max_context, 64);
    artifact_id = artifact->summary().source_repo + "@" +
                  artifact->summary().source_revision;
    route_table_id =
        "qwen-forward-a16-v1:" + topology.signature +
        (enable_m4_short_routes ? ":m4-short-on" : ":m4-short-off");
    if (diagnostic_route)
      route_table_id += ":linear-ab-v1:" + diagnostic_route->object + ":w" +
                        std::to_string(diagnostic_route->capacity) + ":" +
                        diagnostic_route->choice;
    const std::string gpu_family_tag =
        capabilities.device_name == "Apple M4"
            ? "m4"
            : "apple" +
                  std::to_string(capabilities.profile.probe.apple_gpu_family);
    if (enable_attention_split)
      route_table_id += ":attention-split-diagnostic-" + gpu_family_tag;
    if (coupled)
      route_table_id +=
          ":numeric-linear-row-exact-v1:sampling-position_coupled_v1";
    if (moe_route)
      route_table_id += ":moe-ab-v1:" + moe_route->object + ":" +
                        moe_route->format + ":" + moe_route->layout + ":n" +
                        std::to_string(moe_route->n) + ":k" +
                        std::to_string(moe_route->k) + ":t" +
                        std::to_string(moe_route->capacity) + ":" +
                        moe_route->choice;
    weights =
        materialize_model(*artifact, std::move(materialization), context);
    const auto model_digest = execution_artifact_digest(*artifact);
    artifact.reset();

    if (loaded_lora.has_value()) {
      lora_adapter_path = options.lora_adapter;
      lora_mode_active = effective_lora_mode;
      if (effective_lora_mode == "merge") {
        merged_lora = detail::lora_rt::build_merged_lora_overrides(
            context, libraries, weights, layers, *loaded_lora);
        lora_buffer_bytes = merged_lora->buffer_bytes;
        lora_adapted_layers = merged_lora->adapted_layers;
      } else {
        bypass_lora = detail::lora_rt::build_bypass_lora_runtime(
            context, weights, layers, *loaded_lora);
        lora_adapted_layers = bypass_lora->adapted_layers;
      }
    }

    const auto lin_policy = coupled ? LinearNumericPolicy::RowExactPortableV1
                                    : LinearNumericPolicy::Legacy;
    const auto attn_policy =
        coupled ? AttentionNumericPolicy::RowExactV1
                : AttentionNumericPolicy::Legacy;
    QwenPrefillOptions primary_prefill_opts{
        .linear_mode = effective_linear_mode,
        .attention_mode = options.prefill_attention_mode,
        .sparse_minference = options.prefill_sparse_minference,
        .cache_blend = options.prefill_cache_blend,
        .ane_offload = options.prefill_ane_offload,
        .ane_mode = options.prefill_ane_mode,
        .artifact_digest = model_digest,
    };
    QwenPrefillOptions secondary_prefill_opts = primary_prefill_opts;
    secondary_prefill_opts.cache_blend = false;
    secondary_prefill_opts.ane_offload = false;
    secondary_prefill_opts.ane_mode = PrefillAneMode::Off;
    program = std::make_unique<QwenForward>(
        context, weights, layers, config, max_context, libraries,
        draft_config.target_layer_ids, false, any_speculative,
        enable_m4_short_routes, diagnostic_route, enable_attention_split,
        slot_batch_enabled, moe_route, lin_policy, sampling_contract,
        attn_policy, false, slot_kv_dtype, nullptr, enable_flash_nax_prefill,
        shared_kv_pages, merged_lora, bypass_lora, primary_prefill_opts);

    std::string draft_digest = "none";
    std::string ext_mtp_digest;
    if (speculative) {
      if (enable_gpu_verify)
        gpu_verifier = std::make_unique<TargetGpuVerifier>(
            context, program->verification_logits_buffer(),
            std::uint32_t(config.vocab_size), libraries, sampling_contract);
      draft_artifact->verify_payload();
      companion_id = draft_artifact->summary().source_repo + "@" +
                     draft_artifact->summary().source_revision + ":" +
                     std::to_string(slot_dflash_draft_tokens);
      draft_weights =
          materialize_model(*draft_artifact,
                            plan_model_materialization(
                                *draft_artifact, context,
                                {.components = {ModelComponent::speculative},
                                 .require_verified_payload = true}),
                            context);
      draft_digest = execution_artifact_digest(*draft_artifact);
      draft_artifact.reset();
      auto draft_libraries = libraries;
      if (const auto it = options.path_overrides.find("dflash2_metallib");
          it != options.path_overrides.end())
        draft_libraries = std::filesystem::path(it->second).parent_path();
      slot_draft_libraries = draft_libraries;
      draft = std::make_unique<DFlashForward>(
          context, draft_weights, draft_config, max_context,
          program->embedding_weight(), program->head_weight(),
          program->feature_history(), draft_libraries, enable_m4_short_routes,
          lin_policy, sampling_contract);
      sync_draft_target_ring(*draft, program->feature_ring_rows());
      route_table_id += apple9_or_newer
                            ? "+dflash-forward-v1:selector-gpu-" +
                                  gpu_family_tag
                            : "+dflash-forward-v1:selector-cpu-pending-target";
      route_table_id += gpu_verifier
                            ? ":verify-gpu-" + gpu_family_tag
                            : apple9_or_newer
                                  ? ":verify-cpu-diagnostic"
                                  : ":verify-cpu-pending-target";
    } else if (mtp_enabled) {
      if (enable_gpu_verify) {
        gpu_verifier = std::make_unique<TargetGpuVerifier>(
            context, program->verification_logits_buffer(),
            std::uint32_t(config.vocab_size), libraries, sampling_contract);
      }
      if (ext_mtp_artifact.has_value()) {
        ext_mtp_artifact->verify_payload();
        companion_id = ext_mtp_artifact->summary().source_repo + "@" +
                       ext_mtp_artifact->summary().source_revision + ":mtp:" +
                       std::to_string(mtp_draft_tokens);
        mtp_external_weights = materialize_model(
            *ext_mtp_artifact,
            plan_model_materialization(
                *ext_mtp_artifact, context,
                {.components = {ModelComponent::mtp},
                 .require_verified_payload = true}),
            context);
        ext_mtp_digest = execution_artifact_digest(*ext_mtp_artifact);
        ext_mtp_artifact.reset();
      } else {
        companion_id = artifact_id + ":mtp:" + std::to_string(mtp_draft_tokens);
      }
      const MaterializedModel &active_mtp_weights =
          mtp_external_weights.valid() ? mtp_external_weights : weights;
      mtp = std::make_unique<MtpForward>(
          context, active_mtp_weights, config, max_context,
          program->embedding_weight(), program->head_weight(),
          program->final_norm_weight(), program->trunk_hidden_buffer(),
          program->normalization_weight_offset(), libraries,
          enable_m4_short_routes, lin_policy, sampling_contract, slot_kv_dtype,
          &layers, &weights, nullptr, nullptr, mtp_shared_kv_pages);
      route_table_id += "+mtp-forward-v1:k" + std::to_string(mtp_draft_tokens);
      route_table_id += gpu_verifier
                            ? ":verify-gpu-" + gpu_family_tag
                            : apple9_or_newer
                                  ? ":verify-cpu-diagnostic"
                                  : ":verify-cpu-pending-target";
    }
    if (mtp_enabled) {
      mtp_controllers.reserve(std::size_t(options.max_concurrency));
      for (int s_idx = 0; s_idx < options.max_concurrency; ++s_idx) {
        mtp_controllers.emplace_back(mtp_draft_tokens,
                                     mtp_depth_policy.fixed_depth);
      }
    }

    for (int index = 1; index < options.max_concurrency; ++index) {
      auto &slot = extra_slots.emplace_back();
      slot.program = std::make_unique<QwenForward>(
          context, weights, layers, config, max_context, slot_libraries,
          slot_draft_config.target_layer_ids, false, any_speculative,
          slot_short_routes, slot_route_override, slot_attention_split,
          slot_batch_enabled, slot_moe_override, lin_policy, sampling_contract,
          attn_policy, false, slot_kv_dtype, program.get(),
          enable_flash_nax_prefill, 0U, merged_lora, bypass_lora,
          secondary_prefill_opts);
      if (slot_speculative) {
        if (slot_gpu_verify)
          slot.verifier = std::make_unique<TargetGpuVerifier>(
              context, slot.program->verification_logits_buffer(),
              std::uint32_t(config.vocab_size), slot_libraries,
              sampling_contract);
        slot.draft = std::make_unique<DFlashForward>(
            context, draft_weights, slot_draft_config, max_context,
            slot.program->embedding_weight(), slot.program->head_weight(),
            slot.program->feature_history(), slot_draft_libraries,
            slot_short_routes, lin_policy, sampling_contract);
        sync_draft_target_ring(*slot.draft, slot.program->feature_ring_rows());
      } else if (slot_mtp_enabled) {
        if (slot_gpu_verify) {
          slot.verifier = std::make_unique<TargetGpuVerifier>(
              context, slot.program->verification_logits_buffer(),
              std::uint32_t(config.vocab_size), slot_libraries,
              sampling_contract);
        }
        const MaterializedModel &active_mtp_weights =
            mtp_external_weights.valid() ? mtp_external_weights : weights;
        slot.mtp = std::make_unique<MtpForward>(
            context, active_mtp_weights, config, max_context,
            slot.program->embedding_weight(), slot.program->head_weight(),
            slot.program->final_norm_weight(),
            slot.program->trunk_hidden_buffer(),
            slot.program->normalization_weight_offset(), slot_libraries,
            slot_short_routes, lin_policy, sampling_contract, slot_kv_dtype,
            &layers, &weights, &program->trunk_hidden_buffer(),
            mtp_shared_kv_pages != 0U ? mtp.get() : nullptr);
      }
    }

    if (bypass_lora != nullptr) {
      std::uint64_t scratch_sum = program->lora_scratch_bytes();
      for (const auto &es : extra_slots) {
        if (es.program)
          scratch_sum += es.program->lora_scratch_bytes();
      }
      lora_buffer_bytes = bypass_lora->weight_buffer_bytes + scratch_sum;
    }

    // Preserve exact canonical execution_fields when default bf16 + standard
    // prefill_engine are used; isolate non-bf16 KV or flash_nax prefill in digest.
    execution_fields = {
        {"model", model_digest},
        {"model_config", execution_file_digest(model_dir / "config.json")},
        {"draft", draft_digest},
        {"draft_config",
         speculative
             ? execution_file_digest(std::filesystem::path(options.draft_dir) /
                                     "config.json")
             : "none"},
        {"numeric",
         coupled ? "linear_row_exact_portable_v1+attention.row_exact.v1+gdn."
                   "sequential.simd32.f32_state.conv_bf16.v1"
                 : "legacy_route_v1"},
        {"sampling", std::string(sampling_contract_name(sampling_contract))},
        {"chunk", std::string(prefill_chunk_policy_id(sampling_contract))},
        {"kernel", execution_kernel_bundle_digest(libraries)},
        {"draft_kernel",
         speculative ? execution_kernel_bundle_digest(slot_draft_libraries)
                     : "none"},
        {"source", std::string(execution_source_id())},
        {"route", route_table_id},
        {"device", context.capabilities().device_name},
        {"row_layout", "slot-local.W1-8.page64.v1"},
        {"batch_mode", slot_batch_enabled ? "grouped" : "off"},
        {"shared_projection", shared_head_enabled ? "head_rerun_v1" : "off"},
        {"slot_capacity", std::to_string(options.max_concurrency)},
        {"max_context", std::to_string(max_context)},
        {"draft_tokens", std::to_string(slot_dflash_draft_tokens)}};
    if (kv_format != KvStorageFormat::BFloat16) {
      execution_fields.emplace("kv_dtype", slot_kv_dtype);
    }
    if (enable_flash_nax_prefill) {
      execution_fields.emplace("prefill_attn", "flash_nax_v1");
    }
    if (rolling_prefix_cache) {
      execution_fields.emplace("prefix_mode", "rolling_uma_v1");
    }
    if (mtp_enabled) {
      execution_fields.emplace(
          "mtp", ext_mtp_digest.empty() ? "embedded" : ext_mtp_digest);
      execution_fields.emplace("mtp_draft_tokens",
                               std::to_string(mtp_draft_tokens));
    }
    if (loaded_lora.has_value()) {
      execution_fields.emplace("lora", loaded_lora->fingerprint);
      execution_fields.emplace("lora_mode", effective_lora_mode);
    }
    LoadOptions effective_id_options = options;
    effective_id_options.prefill_linear_mode = effective_linear_mode;
    append_prefill_execution_identity_fields(effective_id_options,
                                             execution_fields);
    execution_identity = execution_identity_digest(execution_fields);
    for (std::size_t index = 0; index < occupied.size(); ++index)
      program_at(index).bind_execution_identity(execution_identity);

    if (options.prefill_cache_blend) {
      const std::uint64_t blend_budget = std::min<std::uint64_t>(
          cache_budget_bytes > 0ULL ? cache_budget_bytes : (2ULL << 30),
          64ULL << 30);
      blend_cache = std::make_unique<detail::ChunkBlendCache>(blend_budget);
      blend_model_exec_hash = detail::compute_blend_model_exec_hash(
          model_digest, slot_kv_dtype,
          prefill_linear_mode_to_string(effective_linear_mode),
          prefill_attention_mode_to_string(options.prefill_attention_mode),
          options.prefill_sparse_minference, options.prefill_ane_offload);
    }

    if (cache_budget_bytes) {
      anchor_logits.emplace(
          context.make_buffer(std::uint64_t(config.vocab_size) * 2));
      if (coupled) {
        coupled_anchor_sampler = std::make_unique<PositionKeyedSampler>(
            context, *anchor_logits, std::uint32_t(config.vocab_size), 1,
            slot_libraries);
      } else {
        legacy_anchor_sampler = std::make_unique<LegacyAnchorSampler>(
            context, *anchor_logits, std::uint32_t(config.vocab_size),
            slot_libraries);
      }
    }
    if (slot_batch_enabled) {
      std::vector<QwenForward *> rows;
      rows.reserve(occupied.size());
      for (std::size_t i = 0; i < occupied.size(); ++i)
        rows.push_back(&program_at(i));
      if (coupled) {
        coupled_batch_sampler = std::make_unique<CoupledBatchSampler>(
            context, rows, slot_libraries);
        if (shared_head_enabled)
          shared_head = std::make_unique<SharedHeadProjection>(context, rows,
                                                               slot_libraries);
      } else {
        batch_sampler = std::make_unique<NativeBatchSampler>(
            context, rows, slot_libraries, max_context);
      }
    }

    spec_controller = AdaptiveSpecController(SpecControllerConfig{
        .mode = resolve_spec_search_mode(options),
        .max_draft_tokens = slot_dflash_draft_tokens,
        .max_slots = std::uint32_t(options.max_concurrency),
        .hidden_size = std::uint32_t(config.hidden_size),
        .num_layers = std::uint32_t(config.num_hidden_layers),
        .full_attn_layers = std::uint32_t(config.full_attention_layers()),
        .num_experts = std::uint32_t(config.num_experts),
        .experts_per_token = std::uint32_t(config.num_experts_per_token),
    });
    scheduler = ContendedPrefillScheduler(
        std::size_t(options.max_concurrency),
        std::uint32_t(config.hidden_size));
  }

  struct ExtraSlot {
    std::unique_ptr<QwenForward> program;
    std::unique_ptr<DFlashForward> draft;
    std::unique_ptr<MtpForward> mtp;
    std::unique_ptr<TargetGpuVerifier> verifier;
  };
  QwenForward &program_at(std::size_t index) {
    return index ? *extra_slots.at(index - 1).program : *program;
  }
  DFlashForward *draft_at(std::size_t index) {
    return index ? extra_slots.at(index - 1).draft.get() : draft.get();
  }
  MtpForward *mtp_at(std::size_t index) {
    return index ? extra_slots.at(index - 1).mtp.get() : mtp.get();
  }
  MtpDepthController *mtp_controller_at(std::size_t index) {
    return index < mtp_controllers.size() ? &mtp_controllers[index] : nullptr;
  }
  const MtpDepthController *mtp_controller_at(std::size_t index) const {
    return index < mtp_controllers.size() ? &mtp_controllers[index] : nullptr;
  }
  TargetGpuVerifier *verifier_at(std::size_t index) {
    return index ? extra_slots.at(index - 1).verifier.get()
                 : gpu_verifier.get();
  }
  RequestGpuSampler &request_sampler_at(std::size_t index, bool need_penalties) {
    auto &entry = slot_request_samplers.at(index);
    const auto vocabulary = std::uint32_t(config.vocab_size);
    const auto cap =
        need_penalties
            ? std::min<std::size_t>(kMaxRepetitionContextSize, vocabulary)
            : 0U;
    if (!entry) {
      entry = std::make_unique<RequestGpuSampler>(
          context, slot_libraries,
          program_at(index).verification_logits_buffer(), vocabulary, cap);
    } else if (need_penalties) {
      entry->ensure_penalty_capacity(cap);
    }
    return *entry;
  }
  [[nodiscard]] std::size_t active_occupied_count() const noexcept {
    std::size_t n = 0;
    for (bool occ : occupied)
      if (occ)
        ++n;
    return n;
  }

  ~NativeQwenState() {
    try {
      if (program) {
        program->abort_verify();
        for (std::size_t i = 1; i < occupied.size(); ++i)
          program_at(i).abort_verify();
        prefix_cache.clear_all(
            [this](std::size_t s) -> QwenForward & { return program_at(s); });
      }
    } catch (...) {
    }
  }

  std::uint32_t sample_anchor(std::span<const std::byte> logits_bytes,
                              std::size_t prompt_tokens,
                              const GpuSamplingOptions &sampling) {
    if (!anchor_logits ||
        logits_bytes.size() != std::uint64_t(config.vocab_size) * 2)
      throw std::logic_error("prefix anchor logits unavailable");
    anchor_logits->write(logits_bytes);
    if (sampling_contract == SamplingContract::PositionCoupledV1) {
      const std::array rows{PositionSamplingRow{
          .seed = sampling.seed,
          .position = std::uint64_t(prompt_tokens),
          .top_k = sampling.top_k,
          .temperature = sampling.temperature,
          .top_p = sampling.top_p,
      }};
      return coupled_anchor_sampler->sample(rows).tokens.at(0);
    }
    return legacy_anchor_sampler->sample(prompt_tokens, sampling);
  }

  void refresh_pressure_owner(std::span<const std::size_t> decode_slots) {
    scheduler.progress_owner_slot.reset();
    bool under_severe_pressure = false;
    for (std::size_t s = 0; s < occupied.size(); ++s) {
      if (!occupied[s])
        continue;
      if (const auto *pool = program_at(s).kv_page_pool()) {
        if (pool->physical_pages() > 0 &&
            pool->free_pages() * 100u < pool->physical_pages() * 8u) {
          under_severe_pressure = true;
          break;
        }
      }
    }
    if (!under_severe_pressure || decode_slots.empty())
      return;
    std::size_t best_slot = decode_slots.front();
    std::uint32_t max_pos = program_at(best_slot).position();
    for (std::size_t s : decode_slots) {
      const auto pos = program_at(s).position();
      if (pos > max_pos) {
        max_pos = pos;
        best_slot = s;
      }
    }
    scheduler.progress_owner_slot = best_slot;
  }

  [[nodiscard]] SamplingContract effective_prefill_contract() const noexcept {
    return rolling_prefix_cache ? SamplingContract::LegacyRejectionV1
                                : sampling_contract;
  }

  SamplingContract sampling_contract = SamplingContract::LegacyRejectionV1;
  bool rolling_prefix_cache = false;
  ExecutionIdentityFields execution_fields;
  std::string execution_identity;
  std::uint64_t cache_budget_bytes = 0;
  std::size_t pin_prefix_min_tokens = 0;
  SharedPrefixCache<SamplingKey> prefix_cache;
  std::unique_ptr<detail::ChunkBlendCache> blend_cache;
  std::uint64_t blend_model_exec_hash = 0;
  std::string companion_id;
  MetalContext context;
  QwenModelConfig config;
  MaterializedModel weights, draft_weights, mtp_external_weights;
  QwenLayerCatalog layers;
  std::unique_ptr<QwenForward> program;
  std::unique_ptr<DFlashForward> draft;
  std::unique_ptr<MtpForward> mtp;
  std::vector<MtpDepthController> mtp_controllers;
  std::unique_ptr<TargetGpuVerifier> gpu_verifier;
  std::string artifact_id, route_table_id;
  DFlash2ModelConfig slot_draft_config;
  std::filesystem::path slot_libraries, slot_draft_libraries;
  std::optional<QwenLinearRouteOverride> slot_route_override;
  std::optional<QwenMoeRouteOverride> slot_moe_override;
  std::string slot_kv_dtype = "bf16";
  std::uint32_t slot_dflash_draft_tokens = 0;
  std::uint32_t slot_mtp_draft_tokens = 0;
  std::uint32_t mtp_shared_kv_pages = 0;
  bool slot_short_routes = false, slot_speculative = false,
       slot_mtp_enabled = false, slot_gpu_verify = false,
       slot_attention_split = false, slot_batch_enabled = false;
  std::atomic<bool> external_gpu_contended{false};
  std::vector<ExtraSlot> extra_slots;
  struct SlotMtpFrontierSnapshot {
    std::vector<std::uint32_t> tokens;
    std::vector<std::byte> pending_trunk;
    void clear() noexcept {
      tokens.clear();
      pending_trunk.clear();
    }
  };
  std::vector<bool> occupied;
  std::vector<std::vector<std::uint32_t>> slot_dflash_prefix;
  std::vector<SlotMtpFrontierSnapshot> slot_mtp_think_frontier;
  std::vector<SlotMtpFrontierSnapshot> slot_mtp_post_frontier;
  std::optional<std::pair<std::uint32_t, std::uint32_t>> think_token_ids;
  std::vector<std::uint64_t> slot_rolling_think_anchor_ids;
  std::vector<std::uint64_t> slot_rolling_post_ids;
  std::unique_ptr<NativeBatchSampler> batch_sampler;
  std::unique_ptr<CoupledBatchSampler> coupled_batch_sampler;
  std::unique_ptr<SharedHeadProjection> shared_head;
  std::optional<MetalBuffer> anchor_logits;
  std::unique_ptr<PositionKeyedSampler> coupled_anchor_sampler;
  std::unique_ptr<LegacyAnchorSampler> legacy_anchor_sampler;
  std::vector<std::unique_ptr<RequestGpuSampler>> slot_request_samplers;
  bool shared_head_enabled = false;
  bool lm_head_tied = false;
  std::uint32_t primary_ws_rows = 1024;
  std::string lora_adapter_path;
  std::string lora_mode_active;
  std::uint64_t lora_buffer_bytes = 0;
  std::uint32_t lora_adapted_layers = 0;
  std::string model_quant_format;
  PrefillLinearMode effective_linear_mode = PrefillLinearMode::BFloat16;
  bool prefill_linear_fallback = false;
  std::string prefill_linear_fallback_reason;
  std::shared_ptr<QwenMergedLoraOverrides> merged_lora;
  std::shared_ptr<QwenBypassLoraRuntime> bypass_lora;
  AdaptiveSpecController spec_controller;
  ContendedPrefillScheduler scheduler;
};

namespace {

std::filesystem::path
override_or_default(const LoadOptions &options, std::string_view key,
                    const std::filesystem::path &fallback) {
  const auto found = options.path_overrides.find(std::string(key));
  return found == options.path_overrides.end()
             ? fallback
             : std::filesystem::path(found->second);
}

class TokenPublisher final {
public:
  TokenPublisher(const std::vector<std::vector<std::int32_t>> &stop_sequences,
                 const EmitTokens &emit)
      : stop_sequences_(stop_sequences), emit_(emit) {
    for (const auto &sequence : stop_sequences_) {
      maximum_stop_ = std::max(maximum_stop_, sequence.size());
    }
  }

  bool push(std::int32_t token) {
    pending_.push_back(token);
    for (const auto &stop : stop_sequences_) {
      if (stop.size() <= pending_.size() &&
          std::equal(stop.rbegin(), stop.rend(), pending_.rbegin())) {
        pending_.resize(pending_.size() - stop.size());
        flush_all();
        stopped_ = true;
        return false;
      }
    }
    const auto holdback = maximum_stop_ == 0 ? 0 : maximum_stop_ - 1;
    if (pending_.size() > holdback) {
      const auto count = pending_.size() - holdback;
      emit_({pending_.begin(),
             pending_.begin() + static_cast<std::ptrdiff_t>(count)});
      pending_.erase(pending_.begin(),
                     pending_.begin() + static_cast<std::ptrdiff_t>(count));
    }
    return true;
  }

  std::size_t preview_count(std::span<const std::int32_t> tokens) const {
    auto tail = pending_;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      tail.push_back(tokens[i]);
      for (const auto &stop : stop_sequences_)
        if (stop.size() <= tail.size() &&
            std::equal(stop.rbegin(), stop.rend(), tail.rbegin()))
          return i + 1;
    }
    return tokens.size();
  }
  void finish() { flush_all(); }
  [[nodiscard]] bool stopped() const { return stopped_; }

private:
  void flush_all() {
    if (!pending_.empty()) {
      emit_(std::move(pending_));
      pending_.clear();
    }
  }

  const std::vector<std::vector<std::int32_t>> &stop_sequences_;
  const EmitTokens &emit_;
  std::vector<std::int32_t> pending_;
  std::size_t maximum_stop_ = 0;
  bool stopped_ = false;
};

GpuSamplingOptions sampling_options(const SamplingSpec &sampling) {
  if (sampling.min_p != 0.0F || sampling.top_k > 64) {
    throw std::invalid_argument(
        "native backend supports min_p=0 and top_k<=64 only");
  }
  return {.temperature = sampling.temperature,
          .top_k = static_cast<std::uint32_t>(sampling.top_k),
          .top_p = sampling.top_p,
          .seed = sampling.seed};
}

TargetDistribution target_distribution(std::span<const float> logits,
                                       const GpuSamplingOptions &sampling) {
  std::vector<std::uint32_t> ids(logits.size());
  std::iota(ids.begin(), ids.end(), 0);
  for (auto value : logits)
    if (!std::isfinite(value))
      throw std::runtime_error("nonfinite target logits");
  const auto count = sampling.temperature > 0
                         ? std::min<std::size_t>(sampling.top_k, ids.size())
                         : 1;
  if (!count)
    throw std::invalid_argument("positive temperature requires target top-k");
  std::partial_sort(
      ids.begin(), ids.begin() + count, ids.end(), [&](auto a, auto b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
      });
  ids.resize(count);
  std::vector<float> scores;
  scores.reserve(count);
  for (auto id : ids)
    scores.push_back(logits[id]);
  return target_distribution_from_topk(ids, scores,
                                       std::uint32_t(logits.size()),
                                       sampling.temperature, sampling.top_p);
}

} // namespace

// Unified single-slot & multi-slot request state machine (Ponytail P1-1).
// Both NativeBackend::execute (B=1) and stepped/batched concurrency route
// through NativeTask so telemetry, prefix caching, SpecSearch, and ReplaySSM
// stay in a single canonical implementation.
class NativeTask final : public BackendTask {
  friend class NativeBackend;

public:
  NativeTask(NativeBackend &backend, NativeQwenState &owner, std::size_t slot,
             SubmitRequest request,
             std::shared_ptr<std::atomic<bool>> cancelled, EmitTokens emit,
             GpuSamplingOptions sampling)
      : backend_(backend), owner_(owner), slot_(slot),
        request_(std::move(request)), cancelled_(std::move(cancelled)),
        emit_(std::move(emit)), publisher_(request_.stop_token_seqs, emit_),
        sampling_(sampling) {
    if (request_.sampling.has_penalties() || request_.thinking_budget)
      controls_.emplace(request_, std::uint32_t(owner_.config.vocab_size));
    owner_.spec_controller.reset_slot(slot_);
    if (slot_ < owner_.scheduler.bypass_skips.size())
      owner_.scheduler.bypass_skips[slot_] = 0;
    program().abort_verify();

    prompt_.reserve(request_.prompt_tokens.size());
    for (auto token : request_.prompt_tokens)
      prompt_.push_back(std::uint32_t(token));

    bool prefix_same_slot = false;
    bool restored_mtp_prefix = false;
    if (owner_.prefix_cache.enabled()) {
      const auto match = owner_.prefix_cache.find_best(
          owner_.execution_identity, prompt_, owner_.effective_prefill_contract(),
          owner_.mtp_shared_kv_pages != 0U);
      if (match.has_value()) {
        bool cross_slot = false;
        auto &hit = owner_.prefix_cache.restore_into_slot(
            match->entry_index, slot_,
            [&](std::size_t s) -> QwenForward & { return owner_.program_at(s); },
            cross_slot);
        prefix_same_slot = !cross_slot;
        offset_ = hit.tokens.size();
        if (hit.mtp && owner_.mtp_shared_kv_pages != 0U) {
          mtp()->restore_prefix(*hit.mtp, std::uint32_t(offset_ - 1U));
          restored_mtp_prefix = true;
        }
        matched_tokens_ = offset_;
        matched_entry_id_ = hit.id;
        matched_pinned_prefix_ = hit.pinned && (offset_ < prompt_.size());
        committed_tokens_.assign(hit.tokens.begin(), hit.tokens.end());
        if (offset_ == prompt_.size()) {
          if (!request_.sampling.has_penalties() &&
              hit.sampled_key == make_sampling_key(request_.sampling)) {
            step_.token = hit.sampled_token;
          } else {
            pending_anchor_logits_ = hit.last_logits;
            needs_anchor_resample_ = true;
          }
        }
        std::scoped_lock lock(backend_.stats_mutex_);
        ++backend_.backend_stats_.prefix_hits;
        if (cross_slot)
          ++backend_.backend_stats_.prefix_shared_hits;
        if (offset_ < prompt_.size())
          ++backend_.backend_stats_.prefix_partial_hits;
        backend_.backend_stats_.prefix_restored_tokens += offset_;
      } else {
        program().reset();
        matched_entry_id_ = 0;
        committed_tokens_.clear();
        owner_.prefix_cache.record_miss(prompt_);
        std::scoped_lock lock(backend_.stats_mutex_);
        ++backend_.backend_stats_.prefix_misses;
      }
    } else {
      program().reset();
      matched_entry_id_ = 0;
      committed_tokens_.clear();
    }
    if (draft()) {
      sync_draft_target_ring(*draft(), program().feature_ring_rows());
      bool reused_draft_kv = false;
      if (prefix_same_slot && offset_ > 0U &&
          slot_ < owner_.slot_dflash_prefix.size()) {
        const auto &cached_df = owner_.slot_dflash_prefix[slot_];
        if (!cached_df.empty() && cached_df.size() <= offset_ &&
            std::equal(cached_df.begin(), cached_df.end(), prompt_.begin())) {
          draft()->retain_frontier(std::uint32_t(cached_df.size()),
                                   program().feature_epoch());
          reused_draft_kv = true;
        }
      }
      if (!reused_draft_kv) {
        if (slot_ < owner_.slot_dflash_prefix.size())
          owner_.slot_dflash_prefix[slot_].clear();
        draft()->reset();
      }
    }
    if (mtp()) {
      if (auto *mc = mtp_controller())
        mc->on_new_request(std::uint32_t(prompt_.size()));
      bool reused_mtp_kv = restored_mtp_prefix;
      if (owner_.mtp_shared_kv_pages == 0U && prefix_same_slot && offset_ > 0U &&
          slot_ < owner_.slot_mtp_post_frontier.size()) {
        auto try_restore_mtp =
            [&](const NativeQwenState::SlotMtpFrontierSnapshot &snap) {
              if (snap.tokens.size() == offset_ &&
                  !snap.pending_trunk.empty() &&
                  std::equal(snap.tokens.begin(), snap.tokens.end(),
                             prompt_.begin())) {
                try {
                  mtp()->restore_frontier(std::uint32_t(offset_ - 1U),
                                          snap.pending_trunk);
                  return true;
                } catch (...) {
                  return false;
                }
              }
              return false;
            };
        reused_mtp_kv =
            try_restore_mtp(owner_.slot_mtp_post_frontier[slot_]) ||
            try_restore_mtp(owner_.slot_mtp_think_frontier[slot_]);
        if (reused_mtp_kv) {
          if (owner_.slot_mtp_post_frontier[slot_].tokens.size() > offset_)
            owner_.slot_mtp_post_frontier[slot_].clear();
          if (owner_.slot_mtp_think_frontier[slot_].tokens.size() > offset_)
            owner_.slot_mtp_think_frontier[slot_].clear();
        }
      }
      if (!reused_mtp_kv) {
        if (slot_ < owner_.slot_mtp_post_frontier.size()) {
          owner_.slot_mtp_think_frontier[slot_].clear();
          owner_.slot_mtp_post_frontier[slot_].clear();
        }
        mtp()->reset();
      }
      mtp_primed_ = (offset_ == 0U) || reused_mtp_kv;
      mtp_exited_ = false;
      mtp_committed_tokens_.clear();
      last_mtp_round_committed_.clear();
    }
  }

  ~NativeTask() override {
    const bool unwinding = std::uncaught_exceptions() > 0;
    try {
      drain_in_flight_prefill();
      drain_in_flight_ar();
      drain_in_flight_commit();
      program().abort_verify();
      if (draft()) {
        const bool keep_draft =
            !unwinding && !cancelled_->load(std::memory_order_acquire) &&
            owner_.prefix_cache.enabled() &&
            slot_ < owner_.slot_dflash_prefix.size();
        const std::size_t df_front =
            keep_draft ? std::size_t(draft()->cached_context()) : 0U;
        if (keep_draft && df_front > 0U &&
            df_front <= committed_tokens_.size()) {
          owner_.slot_dflash_prefix[slot_].assign(
              committed_tokens_.begin(),
              committed_tokens_.begin() + std::ptrdiff_t(df_front));
        } else {
          if (slot_ < owner_.slot_dflash_prefix.size())
            owner_.slot_dflash_prefix[slot_].clear();
          draft()->reset();
        }
      }
      program().reset();
      if (mtp()) {
        const bool keep_mtp =
            owner_.mtp_shared_kv_pages == 0U &&
            !unwinding && !cancelled_->load(std::memory_order_acquire) &&
            owner_.prefix_cache.enabled() &&
            slot_ < owner_.slot_mtp_post_frontier.size() &&
            (!owner_.slot_mtp_post_frontier[slot_].tokens.empty() ||
             !owner_.slot_mtp_think_frontier[slot_].tokens.empty());
        if (!keep_mtp) {
          if (slot_ < owner_.slot_mtp_post_frontier.size()) {
            owner_.slot_mtp_think_frontier[slot_].clear();
            owner_.slot_mtp_post_frontier[slot_].clear();
          }
          mtp()->reset();
        }
      }
    } catch (...) {
    }
    if (owner_.scheduler.progress_owner_slot == slot_)
      owner_.scheduler.progress_owner_slot.reset();
    owner_.occupied.at(slot_) = false;
  }

  template <class Operation> auto with_page_pressure(Operation operation) {
    for (;;) {
      try {
        return operation();
      } catch (const std::overflow_error &error) {
        if (std::string_view(error.what()) !=
                "KV physical page pool exhausted" ||
            !owner_.prefix_cache.evict_for_page_pressure(
                slot_, [&](std::size_t s) -> QwenForward & {
                  return owner_.program_at(s);
                })) {
          throw;
        }
        std::scoped_lock lock(backend_.stats_mutex_);
        ++backend_.backend_stats_.prefix_evictions;
        backend_.backend_stats_.prefix_pinned_entries =
            owner_.prefix_cache.pinned_count();
      }
    }
  }

  [[nodiscard]] PositionSamplingRow controlled_row(
      std::uint64_t position, bool valid = true) const {
    return {.seed = sampling_.seed,
            .position = position,
            .top_k = sampling_.top_k,
            .temperature = sampling_.temperature,
            .top_p = sampling_.top_p,
            .valid = static_cast<std::uint32_t>(valid)};
  }

  RequestGpuSampler &controlled_sampler() {
    return owner_.request_sampler_at(slot_, controls_->max_penalty_tokens() > 0);
  }

  void apply_controlled_sample(QwenForwardResult &result, bool cached = false) {
    if (!controls_)
      return;
    if (const auto forced = controls_->forced_token()) {
      result.token = *forced;
      return;
    }
    if (!request_.sampling.has_penalties())
      return;
    std::vector<RequestPenaltyToken> penalties;
    controls_->append_penalties(penalties);
    const auto row = controlled_row(program().position());
    const std::array<std::uint32_t, 2> range{0U, std::uint32_t(penalties.size())};
    auto selected =
        cached ? controlled_sampler().sample_cached(
                     row, pending_anchor_logits_, penalties)
               : controlled_sampler().sample(std::span(&row, 1),
                                             std::span(&range, 1), penalties,
                                             program().last_logits_row());
    result.token = selected.tokens.front();
    result.host_read_bytes += selected.host_read_bytes;
    result.submission.dispatch_count += selected.submission.dispatch_count;
    result.submission.timing.gpu_ms += selected.submission.timing.gpu_ms;
    result.submission.timing.wall_ms += selected.submission.timing.wall_ms;
  }

  void resolve_anchor_if_needed() {
    const bool controlled_sample = controls_ &&
        (request_.sampling.has_penalties() || controls_->forced_token().has_value());
    if (controlled_sample) {
      apply_controlled_sample(step_, needs_anchor_resample_);
      pending_anchor_logits_.clear();
      needs_anchor_resample_ = false;
      return;
    }
    if (!needs_anchor_resample_)
      return;
    step_.token = owner_.sample_anchor(pending_anchor_logits_, prompt_.size(),
                                       sampling_);
    const auto skey = make_sampling_key(request_.sampling);
    owner_.prefix_cache.update_anchor_sample(owner_.execution_identity, prompt_,
                                             skey, step_.token);
    pending_anchor_logits_.clear();
    needs_anchor_resample_ = false;
  }

  bool publish_token(std::int32_t token) {
    const bool continued = publisher_.push(token);
    if (controls_ && continued)
      controls_->accept(token);
    return continued;
  }

  [[nodiscard]] bool should_pin_prefix(std::size_t token_len) const noexcept {
    return request_.pin_prefix ||
           (owner_.pin_prefix_min_tokens > 0 &&
            token_len >= owner_.pin_prefix_min_tokens);
  }

  [[nodiscard]] std::size_t pre_tail_anchor_offset() const noexcept {
    if (prompt_.size() < 2048)
      return 0;
    const std::size_t pre_tail = (prompt_.size() / 1024) * 1024;
    return (pre_tail < prompt_.size()) ? pre_tail : 0;
  }

  [[nodiscard]] bool should_save_prefix() const noexcept {
    if (!owner_.prefix_cache.enabled() ||
        cancelled_->load(std::memory_order_acquire))
      return false;
    // Skip redundant full-prompt save + read_last_logits_bf16 on unpinned warm
    // tail hits extending an already-pinned shared prefix (B=1 / B=4 CoW and
    // 2K..64K warm decode scaling), re-enabling 1-CB fused sampling!
    if (!request_.pin_prefix && matched_pinned_prefix_)
      return false;
    // In rolling UMA mode, unpinned multi-token turns defer checkpointing to
    // decode-end (save_post_decode_prefix), keeping 1-CB fused sampling active
    // and avoiding double checkpoints / read_last_logits_bf16() on every turn.
    if (owner_.rolling_prefix_cache && !request_.pin_prefix &&
        request_.max_new_tokens > 1)
      return false;
    return true;
  }

  [[nodiscard]] bool will_save_sparse_anchor(std::size_t next_offset) const {
    if (!owner_.prefix_cache.enabled() ||
        cancelled_->load(std::memory_order_acquire))
      return false;
    const std::size_t pre_tail = pre_tail_anchor_offset();
    if (pre_tail == 0 || next_offset != pre_tail || matched_tokens_ >= pre_tail)
      return false;
    if (!request_.pin_prefix && matched_pinned_prefix_)
      return false;
    if (owner_.rolling_prefix_cache && !request_.pin_prefix &&
        request_.max_new_tokens > 1)
      return false;
    const std::uint64_t est_kv =
        ((std::uint64_t(pre_tail) + 63ULL) / 64ULL) *
        program().kv_page_all_layers_bytes();
    return owner_.prefix_cache.budget_bytes() >=
           program().non_kv_checkpoint_bytes() + est_kv;
  }

  void maybe_save_sparse_anchor() {
    const std::size_t think_anchor = open_thinking_anchor_offset();
    if (think_anchor > 0U && offset_ == think_anchor &&
        saved_think_anchor_id_ == 0ULL &&
        !cancelled_->load(std::memory_order_acquire)) {
      save_rolling_thinking_anchor(think_anchor);
      return;
    }
    // Snapshot ONLY the single pre-tail 1024-aligned boundary so 0 -> pre_tail
    // runs at full 2048/4096-chunk async pipelined speed with zero intermediate
    // stalls or pin explosion.
    if (!will_save_sparse_anchor(offset_))
      return;
    const auto sub = std::span<const std::uint32_t>(prompt_).first(offset_);
    const bool pin_entry = should_pin_prefix(offset_);
    bool inserted = false;
    const auto evicted = owner_.prefix_cache.insert_checkpoint(
        slot_, owner_.execution_identity, sub, SamplingKey{}, 0u, false,
        pin_entry,
        [&](std::size_t s) -> QwenForward & { return owner_.program_at(s); },
        &inserted, 0, checkpoint_mtp(offset_));
    if (pin_entry && inserted)
      pre_tail_pinned_ = true;
    {
      std::scoped_lock lock(backend_.stats_mutex_);
      backend_.backend_stats_.prefix_evictions += evicted;
      backend_.backend_stats_.prefix_pinned_entries =
          owner_.prefix_cache.pinned_count();
    }
  }

  // Returns the exact prompt index immediately BEFORE an unclosed trailing
  // `<think>` tag (e.g. `<|im_start|>assistant\n` before `<think>\n`).
  // Saving at `open_pos` rather than `prompt_.size()` guarantees an exact match
  // on the next turn whether the chat template strips `<think>...</think>`
  // completely OR renders `<think>\n\n</think>\n\n` (where BPE merges `\n\n`
  // into token 271 instead of single `\n` 198).
  [[nodiscard]] std::size_t open_thinking_anchor_offset() const noexcept {
    if (!owner_.rolling_prefix_cache || !owner_.prefix_cache.enabled() ||
        request_.pin_prefix || request_.max_new_tokens <= 1 ||
        !owner_.think_token_ids.has_value() || prompt_.size() < 2U) {
      return 0U;
    }
    const auto [open_id, close_id] = *owner_.think_token_ids;
    const std::size_t tail = std::min<std::size_t>(8U, prompt_.size());
    std::size_t open_pos = 0U;
    bool saw_open = false;
    for (std::size_t i = prompt_.size() - tail; i < prompt_.size(); ++i) {
      if (prompt_[i] == open_id) {
        open_pos = i;
        saw_open = true;
      } else if (prompt_[i] == close_id) {
        saw_open = false;
      }
    }
    return (saw_open && open_pos > matched_tokens_ && open_pos < prompt_.size())
               ? open_pos
               : 0U;
  }

  void save_rolling_thinking_anchor(std::size_t prefix_len) {
    release_prior_slot_rolling_pair_if_matched();
    const auto sub = std::span<const std::uint32_t>(prompt_).first(prefix_len);
    bool inserted = false;
    const auto evicted = owner_.prefix_cache.insert_checkpoint(
        slot_, owner_.execution_identity, sub, SamplingKey{}, 0u,
        /*include_anchor_logits=*/false, /*pin_entry=*/false,
        [&](std::size_t s) -> QwenForward & { return owner_.program_at(s); },
        &inserted, /*supersede_entry_id=*/matched_entry_id_,
        checkpoint_mtp(prefix_len));
    if (inserted && slot_ < owner_.slot_rolling_think_anchor_ids.size()) {
      saved_think_anchor_id_ = owner_.prefix_cache.last_inserted_id();
      owner_.slot_rolling_think_anchor_ids[slot_] = saved_think_anchor_id_;
      if (owner_.mtp_shared_kv_pages == 0U &&
          slot_ < owner_.slot_mtp_think_frontier.size()) {
        owner_.slot_mtp_think_frontier[slot_].clear();
        if (mtp_active() && prefix_len == offset_ && offset_ > 0U &&
            mtp()->committed_tokens() + 1U == std::uint32_t(offset_) &&
            mtp()->has_pending_trunk()) {
          auto snap = mtp()->snapshot_pending_trunk();
          if (!snap.empty()) {
            owner_.slot_mtp_think_frontier[slot_].tokens.assign(
                prompt_.begin(), prompt_.begin() + std::ptrdiff_t(prefix_len));
            owner_.slot_mtp_think_frontier[slot_].pending_trunk =
                std::move(snap);
            if (owner_.slot_mtp_post_frontier[slot_].tokens.size() >
                prefix_len) {
              owner_.slot_mtp_post_frontier[slot_].clear();
            }
          }
        }
      }
    }
    std::scoped_lock lock(backend_.stats_mutex_);
    backend_.backend_stats_.prefix_evictions += evicted;
    backend_.backend_stats_.prefix_pinned_entries =
        owner_.prefix_cache.pinned_count();
  }

  [[nodiscard]] bool has_open_thinking_turn() const noexcept {
    if (!owner_.think_token_ids.has_value())
      return false;
    const auto [open_id, close_id] = *owner_.think_token_ids;
    if (step_.token == open_id)
      return true;
    const std::size_t tail = std::min<std::size_t>(8U, prompt_.size());
    bool saw_open = false;
    for (std::size_t i = prompt_.size() - tail; i < prompt_.size(); ++i) {
      if (prompt_[i] == open_id)
        saw_open = true;
      else if (prompt_[i] == close_id)
        saw_open = false;
    }
    return saw_open;
  }

  void release_prior_slot_rolling_pair_if_matched() {
    if (matched_entry_id_ == 0ULL ||
        slot_ >= owner_.slot_rolling_think_anchor_ids.size())
      return;
    const std::uint64_t prev_think =
        owner_.slot_rolling_think_anchor_ids[slot_];
    const std::uint64_t prev_post = owner_.slot_rolling_post_ids[slot_];
    if (matched_entry_id_ != prev_think && matched_entry_id_ != prev_post)
      return;
    auto prog_fn = [&](std::size_t s) -> QwenForward & {
      return owner_.program_at(s);
    };
    (void)owner_.prefix_cache.release_unpinned_slot_entry_by_id(
        prev_think, slot_, owner_.execution_identity, prog_fn);
    if (prev_post != prev_think) {
      (void)owner_.prefix_cache.release_unpinned_slot_entry_by_id(
          prev_post, slot_, owner_.execution_identity, prog_fn);
    }
    owner_.slot_rolling_think_anchor_ids[slot_] = 0ULL;
    owner_.slot_rolling_post_ids[slot_] = 0ULL;
  }

  void save_prefix() {
    if (offset_ != prompt_.size())
      return;
    if (!should_save_prefix()) {
      if (saved_think_anchor_id_ == 0ULL && owner_.rolling_prefix_cache &&
          owner_.prefix_cache.enabled() &&
          !cancelled_->load(std::memory_order_acquire) &&
          !request_.pin_prefix && request_.max_new_tokens > 1 &&
          has_open_thinking_turn()) {
        save_rolling_thinking_anchor(prompt_.size());
      }
      return;
    }
    // If pre_tail was already pinned as the shared prefix target on this seed
    // request, save the full prompt (+tail) unpinned so only the shared prefix
    // consumes pinned budget.
    const bool pin_entry =
        !pre_tail_pinned_ && should_pin_prefix(prompt_.size());
    const auto evicted = owner_.prefix_cache.insert_checkpoint(
        slot_, owner_.execution_identity, prompt_,
        make_sampling_key(request_.sampling), step_.token, true, pin_entry,
        [&](std::size_t s) -> QwenForward & { return owner_.program_at(s); },
        nullptr, 0, checkpoint_mtp(offset_));
    {
      std::scoped_lock lock(backend_.stats_mutex_);
      backend_.backend_stats_.prefix_evictions += evicted;
      backend_.backend_stats_.prefix_pinned_entries =
          owner_.prefix_cache.pinned_count();
    }
  }

  std::optional<MtpPrefixCheckpoint> checkpoint_mtp(std::size_t target_pos) {
    if (owner_.mtp_shared_kv_pages == 0U || !mtp_active() || target_pos == 0U ||
        mtp()->committed_tokens() + 1U != target_pos ||
        !mtp()->has_pending_trunk()) {
      return std::nullopt;
    }
    return mtp()->checkpoint_prefix();
  }

  void seal_mtp_after_decode(std::size_t pos) {
    if (mtp_active() && pos > 0U &&
        !(mtp()->committed_tokens() + 1U == pos && mtp()->has_pending_trunk()) &&
        !last_mtp_round_committed_.empty() &&
        mtp()->committed_tokens() + last_mtp_round_committed_.size() == pos) {
      try {
        (void)mtp()->seal_post_verify_frontier(last_mtp_round_committed_, true);
      } catch (...) {
        // A failed optional cache save does not invalidate the result.
      }
    }
  }

  void save_post_decode_prefix() {
    if (!owner_.rolling_prefix_cache || !owner_.prefix_cache.enabled() ||
        cancelled_->load(std::memory_order_acquire) || request_.pin_prefix ||
        post_decode_saved_)
      return;
    post_decode_saved_ = true;
    drain_in_flight_commit();
    drain_in_flight_ar();
    drain_in_flight_prefill();
    const auto pos = std::size_t(program().position());
    if (pos <= matched_tokens_ || committed_tokens_.size() != pos)
      return;
    if (saved_think_anchor_id_ == 0ULL) {
      release_prior_slot_rolling_pair_if_matched();
    }
    const std::uint64_t supersede_id =
        (saved_think_anchor_id_ != 0ULL) ? 0ULL : matched_entry_id_;
    if (owner_.mtp_shared_kv_pages != 0U)
      seal_mtp_after_decode(pos);
    bool inserted = false;
    const auto evicted = owner_.prefix_cache.insert_checkpoint(
        slot_, owner_.execution_identity, committed_tokens_, SamplingKey{}, 0u,
        /*include_anchor_logits=*/false, /*pin_entry=*/false,
        [&](std::size_t s) -> QwenForward & { return owner_.program_at(s); },
        &inserted, /*supersede_entry_id=*/supersede_id, checkpoint_mtp(pos));
    if (inserted && slot_ < owner_.slot_rolling_post_ids.size()) {
      owner_.slot_rolling_post_ids[slot_] =
          owner_.prefix_cache.last_inserted_id();
      if (saved_think_anchor_id_ == 0ULL) {
        owner_.slot_rolling_think_anchor_ids[slot_] = 0ULL;
        if (slot_ < owner_.slot_mtp_think_frontier.size()) {
          owner_.slot_mtp_think_frontier[slot_].clear();
        }
      }
      if (owner_.mtp_shared_kv_pages == 0U &&
          slot_ < owner_.slot_mtp_post_frontier.size()) {
        owner_.slot_mtp_post_frontier[slot_].clear();
        if (mtp_active() && pos > 0U) {
          seal_mtp_after_decode(pos);
          if (mtp()->committed_tokens() + 1U == std::uint32_t(pos) &&
              mtp()->has_pending_trunk()) {
            auto snap = mtp()->snapshot_pending_trunk();
            if (!snap.empty()) {
              owner_.slot_mtp_post_frontier[slot_].tokens = committed_tokens_;
              owner_.slot_mtp_post_frontier[slot_].pending_trunk =
                  std::move(snap);
            }
          }
        }
      }
    }
    {
      std::scoped_lock lock(backend_.stats_mutex_);
      backend_.backend_stats_.prefix_evictions += evicted;
      backend_.backend_stats_.prefix_pinned_entries =
          owner_.prefix_cache.pinned_count();
    }
  }

  ExecutionResult finish_task(ExecutionResult res) {
    if (res.kind == EventKind::finished) {
      save_post_decode_prefix();
    }
    return res;
  }

  [[nodiscard]] bool batchable() const {
    if (cancelled_->load(std::memory_order_acquire) ||
        in_flight_ar_.has_value() || in_flight_prefill_.has_value() ||
        in_flight_commit_.has_value())
      return false;
    if (offset_ < prompt_.size()) {
      if (controls_ && controls_->needs_intervention_within(1U))
        return false;
      if (const std::size_t ta = open_thinking_anchor_offset(); ta > offset_)
        return false;
      return prompt_.size() - offset_ <= 8;
    }
    if (mtp_active()) {
      if (controls_ && controls_->needs_intervention_within(5U))
        return false;
      if (const auto *mc = owner_.mtp_controller_at(slot_);
          mc && mc->should_exit())
        return false;
      return produced_ > 0;
    }
    if (draft()) {
      if (controls_ && controls_->needs_intervention_within(8U))
        return false;
      return produced_ > 0;
    }
    if (controls_ && controls_->needs_intervention_within(2U))
      return false;
    return true;
  }

  [[nodiscard]] bool is_spec_verify_ready() const {
    return !cancelled_->load(std::memory_order_acquire) &&
           offset_ >= prompt_.size() && draft() != nullptr && produced_ > 0;
  }

  [[nodiscard]] SpecSlotQuery make_spec_query() const {
    const auto remaining = std::uint32_t(request_.max_new_tokens) - produced_;
    const auto available =
        std::uint32_t(backend_.options_.max_context) - program().position();
    if (!available)
      throw std::out_of_range("DFlash target context exhausted");
    const auto max_k =
        std::min({backend_.options_.speculative.draft_tokens, remaining - 1,
                  available - 1});
    return SpecSlotQuery{
        .slot = slot_,
        .max_k = max_k,
        .context_tokens = program().position(),
    };
  }

  std::optional<ExecutionResult>
  prepare_batch_row(bool *out_wants_dflash_draft = nullptr,
                    bool *out_wants_mtp_draft = nullptr,
                    std::optional<SpecSlotDecision> assigned_spec = std::nullopt,
                    std::uint32_t batch_slots_hint = 1U) {
    if (out_wants_dflash_draft)
      *out_wants_dflash_draft = false;
    if (out_wants_mtp_draft)
      *out_wants_mtp_draft = false;
    if (cancelled_->load(std::memory_order_acquire))
      return ExecutionResult{.kind = EventKind::aborted,
                             .finish_reason = "cancelled"};
    pending_input_.clear();
    pending_mtp_probe_ = false;
    pending_prefill_ = offset_ < prompt_.size();
    pending_was_first_chunk_ = pending_prefill_ && (offset_ == 0);
    if (!pending_prefill_)
      resolve_anchor_if_needed();
    pending_mtp_verify_ = !pending_prefill_ && mtp_active();
    if (pending_mtp_verify_) {
      auto *ctrl = mtp_controller();
      if (ctrl && ctrl->should_exit()) {
        mtp_exited_ = true;
        last_mtp_round_committed_.clear();
        note_fallback("mtp_adaptive_exit");
        pending_mtp_verify_ = false;
        pending_verify_ = (draft() != nullptr);
        step_.token = anchor_;
        if (!pending_verify_) {
          pending_input_.push_back(anchor_);
          return std::nullopt;
        }
      }
    }
    pending_verify_ = !pending_prefill_ && (draft() != nullptr || mtp_active());
    if (pending_prefill_) {
      const auto count = prompt_.size() - offset_;
      if (count > 8)
        throw std::logic_error("P10 large prefill not batchable");
      pending_input_.insert(pending_input_.end(), prompt_.begin() + offset_,
                            prompt_.end());
    } else if (!pending_verify_) {
      const auto token = std::int32_t(step_.token);
      if (!publish_token(token))
        return finish_task(ExecutionResult{.finish_reason = "stop"});
      ++produced_;
      if (produced_ == std::uint32_t(request_.max_new_tokens)) {
        publisher_.finish();
        return finish_task(ExecutionResult{});
      }
      pending_input_.push_back(std::uint32_t(token));
    }
    if (pending_mtp_verify_) {
      const auto remaining = std::uint32_t(request_.max_new_tokens) - produced_;
      const auto available =
          std::uint32_t(backend_.options_.max_context) - program().position();
      if (!available)
        throw std::out_of_range("MTP target context exhausted");
      const std::uint32_t max_allowed_k = std::min(
          {owner_.slot_mtp_draft_tokens, remaining - 1U, available - 1U});
      auto *ctrl = mtp_controller();
      const bool was_probing = ctrl && ctrl->is_probing(batch_slots_hint);
      pending_max_k_ = max_allowed_k;
      pending_k_ = ctrl ? ctrl->select_depth(max_allowed_k, batch_slots_hint)
                        : max_allowed_k;
      pending_mtp_probe_ =
          (pending_k_ == 0U && max_allowed_k > 0U && was_probing);
      if (pending_k_ == 0U && max_allowed_k > 0U) {
        std::scoped_lock lock(backend_.stats_mutex_);
        if (pending_mtp_probe_) {
          ++backend_.backend_stats_.spec_search_probe_rounds;
        } else {
          note_fallback("adaptive_gate");
          ++backend_.backend_stats_.spec_search_gated_rounds;
        }
      }
      pending_cost_ = std::make_unique<RoundCostRecorder>(
          backend_.backend_stats_, backend_.stats_mutex_, pending_k_);
      pending_proposal_ = {};
      if (pending_k_ > 0U) {
        if (out_wants_mtp_draft) {
          *out_wants_mtp_draft = true;
          return std::nullopt;
        }
        const bool tokens_only =
            owner_.sampling_contract == SamplingContract::PositionCoupledV1 &&
            verifier() != nullptr;
        pending_proposal_ =
            pending_cost_->measure(RoundCostRecorder::draft, [&] {
              return mtp()->propose(mtp_committed_tokens_, program().position(),
                                    pending_k_, sampling_, true, tokens_only);
            });
      } else if (available > 1U &&
                 (remaining > 1U ||
                  (owner_.mtp_shared_kv_pages != 0U && owner_.rolling_prefix_cache))) {
        pending_proposal_.submission =
            pending_cost_->measure(RoundCostRecorder::draft, [&] {
              return mtp()->fold_committed_only(mtp_committed_tokens_, true);
            });
      }
      pending_input_.push_back(anchor_);
      pending_input_.insert(pending_input_.end(),
                            pending_proposal_.selection.tokens.begin(),
                            pending_proposal_.selection.tokens.end());
    } else if (pending_verify_) {
      const auto query = make_spec_query();
      const SpecSlotDecision decision =
          assigned_spec.has_value()
              ? *assigned_spec
              : owner_.spec_controller.select_single_depth(query);
      pending_max_k_ = query.max_k;
      pending_k_ = decision.k;
      if (decision.gated && pending_k_ == 0)
        note_fallback("adaptive_gate");
      if (decision.gated || decision.probe) {
        std::scoped_lock lock(backend_.stats_mutex_);
        if (decision.gated)
          ++backend_.backend_stats_.spec_search_gated_rounds;
        if (decision.probe)
          ++backend_.backend_stats_.spec_search_probe_rounds;
      }
      pending_cost_ = std::make_unique<RoundCostRecorder>(
          backend_.backend_stats_, backend_.stats_mutex_, pending_k_);
      pending_proposal_ = {};
      if (pending_k_) {
        sync_draft_target_epoch(*draft(), program().feature_epoch());
        if (out_wants_dflash_draft) {
          *out_wants_dflash_draft = true;
          return std::nullopt;
        }
        const bool cpu_sel =
            owner_.context.capabilities().profile.probe.apple_gpu_family < 9;
        const bool tokens_only =
            owner_.sampling_contract == SamplingContract::PositionCoupledV1 &&
            verifier() != nullptr;
        pending_proposal_ =
            pending_cost_->measure(RoundCostRecorder::draft, [&] {
              return invoke_draft_propose(
                  *draft(), anchor_, program().position(), pending_k_,
                  request_.sampling.temperature, request_.sampling.seed, true,
                  false, cpu_sel, tokens_only);
            });
      }
      pending_input_.push_back(anchor_);
      pending_input_.insert(pending_input_.end(),
                            pending_proposal_.selection.tokens.begin(),
                            pending_proposal_.selection.tokens.end());
    }
    return std::nullopt;
  }

  void apply_draft_result(DFlash2DraftResult &&proposal) {
    pending_proposal_ = std::move(proposal);
    pending_input_.push_back(anchor_);
    pending_input_.insert(pending_input_.end(),
                          pending_proposal_.selection.tokens.begin(),
                          pending_proposal_.selection.tokens.end());
  }

  void finish_deferred_draft(DFlashPreparedProposal &&prep, bool batched,
                             MetalSubmission batched_sub) {
    auto res = pending_cost_->measure(RoundCostRecorder::draft, [&] {
      return draft()->finish_propose(std::move(prep), batched, batched_sub);
    });
    apply_draft_result(std::move(res));
  }

  void finish_deferred_mtp_draft(MtpPreparedProposal &&prep, bool batched,
                                 MetalSubmission batched_sub) {
    auto res = pending_cost_->measure(RoundCostRecorder::draft, [&] {
      return mtp()->finish_propose(std::move(prep), batched, batched_sub);
    });
    apply_draft_result(std::move(res));
  }

  QwenPreparedStep prepare_gpu_batch() {
    return with_page_pressure([&] {
      return program().prepare_batch(pending_input_, sampling_, pending_verify_);
    });
  }

  void complete_batch_ar_result(QwenForwardResult &&res) {
    step_ = std::move(res);
    if (pending_prefill_)
      offset_ += pending_input_.size();
    else
      record_ar_step();
    committed_tokens_.insert(committed_tokens_.end(), pending_input_.begin(),
                             pending_input_.end());
    std::scoped_lock lock(backend_.stats_mutex_);
    auto &stats = backend_.backend_stats_;
    ++stats.ar_steps;
    stats.actual_t = stats.actual_m = step_.tokens;
    stats.actual_k = 0;
    stats.ar_layer_calls += step_.executed_layers;
    stats.short_matrix_calls += step_.short_matrix_calls;
    stats.short_lane_calls += step_.short_lane_calls;
    stats.ar_host_read_bytes += step_.host_read_bytes;
    ++stats.batched_ar_steps;
  }

  void complete_batch_ar(QwenPreparedStep &&prepared, std::uint32_t sampled) {
    complete_batch_ar_result(program().finish_batch(std::move(prepared), sampled));
  }

  struct PreparedCommit {
    VerificationResult verified;
    std::size_t count = 0;
    MetalIcbProfile *icb = nullptr;
    double verify_gpu_ms = 0.0;
    double selector_gpu_ms = 0.0;
    bool selector_timed = false;
  };

  std::optional<ExecutionResult>
  record_batch_verify_result(QwenForwardResult &&res,
                             double verify_gpu_share_ms) {
    step_ = std::move(res);
    record_spec_step(pending_k_, pending_mtp_probe_ ? 0U : pending_max_k_);
    {
      std::scoped_lock lock(backend_.stats_mutex_);
      auto &stats = backend_.backend_stats_;
      ++stats.spec_rounds;
      ++stats.spec_target_verify_calls;
      ++stats.batched_verify_rounds;
      stats.spec_proposal_tokens += pending_k_;
      stats.short_matrix_calls +=
          pending_proposal_.short_matrix_calls + step_.short_matrix_calls;
      stats.short_lane_calls +=
          pending_proposal_.short_lane_calls + step_.short_lane_calls;
      stats.spec_context_rows_projected +=
          pending_proposal_.context_rows_projected;
      stats.spec_context_rows_reused += pending_proposal_.context_rows_reused;
      stats.spec_context_rows_launched +=
          pending_proposal_.context_rows_launched;
      stats.spec_host_read_bytes +=
          step_.host_read_bytes + pending_proposal_.host_read_bytes;
      stats.spec_draft_gpu_ms += pending_proposal_.submission.timing.gpu_ms;
      stats.spec_verify_gpu_ms += verify_gpu_share_ms;
      stats.actual_t = stats.actual_m = std::uint32_t(pending_input_.size());
      stats.actual_k = pending_k_;
    }
    if (cancelled_->load(std::memory_order_acquire)) {
      program().abort_verify();
      return ExecutionResult{.kind = EventKind::aborted,
                             .finish_reason = "cancelled"};
    }
    return std::nullopt;
  }

  std::optional<ExecutionResult>
  finish_batch_select_and_prepare_commit(
      std::optional<TargetGpuPreparedSelect> &&sel_prep, bool sel_batched,
      MetalSubmission sel_batched_sub, double verify_gpu_share_ms,
      PreparedCommit &out_commit) {
    out_commit.verify_gpu_ms = verify_gpu_share_ms;
    const auto sample_begin = RoundCostRecorder::Clock::now();
    VerificationResult verified;
    bool selector_timed = (verifier() == nullptr);
    double selector_gpu_ms = 0.0;
    try {
      if (verifier()) {
        TargetGpuVerifyDecision selected =
            sel_prep.has_value()
                ? verifier()->finish_prepared_select(
                      std::move(*sel_prep), sel_batched,
                      std::move(sel_batched_sub))
                : verifier()->select(pending_proposal_.selection.steps,
                                     sampling_, program().position());
        verified = std::move(selected.verification);
        selector_gpu_ms = selected.submission.timing.gpu_ms;
        selector_timed = selector_gpu_ms > 0.0;
        std::scoped_lock lock(backend_.stats_mutex_);
        backend_.backend_stats_.spec_host_read_bytes += selected.host_read_bytes;
        backend_.backend_stats_.spec_verify_selector_gpu_ms += selector_gpu_ms;
      } else {
        auto logits = program().read_logits(std::uint32_t(pending_input_.size()));
        {
          std::scoped_lock lock(backend_.stats_mutex_);
          backend_.backend_stats_.spec_host_read_bytes += logits.size() * 2;
        }
        std::vector<TargetDistribution> distributions;
        for (std::size_t row = 0; row < pending_input_.size(); ++row)
          distributions.push_back(target_distribution(
              std::span(logits).subspan(row * owner_.config.vocab_size,
                                        owner_.config.vocab_size),
              sampling_));
        verified = verify_speculative(
            pending_proposal_.selection.steps, distributions,
            request_.sampling.temperature, request_.sampling.seed,
            program().position());
      }
    } catch (...) {
      program().abort_verify();
      throw;
    }
    if (!sel_batched) {
      pending_cost_->add_select(RoundCostRecorder::milliseconds(sample_begin));
    }
    const auto count = publisher_.preview_count(verified.licensed_outputs);
    if (cancelled_->load(std::memory_order_acquire)) {
      program().abort_verify();
      return ExecutionResult{.kind = EventKind::aborted,
                             .finish_reason = "cancelled"};
    }
    out_commit.icb = pending_cost_->measure(RoundCostRecorder::replay, [&] {
      return program().prepare_commit_prefix(std::uint32_t(count));
    });
    out_commit.count = count;
    out_commit.selector_gpu_ms = selector_gpu_ms;
    out_commit.selector_timed = selector_timed;
    out_commit.verified = std::move(verified);
    return std::nullopt;
  }

  std::optional<ExecutionResult>
  finish_batch_verify_commit(PreparedCommit &&pc, bool batched_executed,
                             double batched_gpu_ms,
                             std::uint32_t batch_total_m,
                             std::uint32_t batch_slots = 1U) {
    const auto count = pc.count;
    const double slot_scale = double(std::max<std::uint32_t>(1U, batch_slots));
    auto commit = pending_cost_->measure(RoundCostRecorder::replay, [&] {
      return program().finish_commit_prefix(std::uint32_t(count),
                                            batched_executed);
    });
    if (count > 0) {
      committed_tokens_.insert(committed_tokens_.end(), pending_input_.begin(),
                               pending_input_.begin() + std::ptrdiff_t(count));
    }
    if (pending_mtp_verify_) {
      if (count > 0) {
        // Packed verification leaves each slot's trunk rows in its own buffer.
        // Replaying the commit ICBs together does not change those rows.
        last_mtp_round_committed_.assign(
            pending_input_.begin(),
            pending_input_.begin() + std::ptrdiff_t(count));
      } else {
        last_mtp_round_committed_.clear();
      }
    }
    if (batched_executed) {
      commit.timing.gpu_ms = batched_gpu_ms;
    }
    pending_cost_->mark_committed(std::uint32_t(count));
    {
      std::scoped_lock lock(backend_.stats_mutex_);
      auto &stats = backend_.backend_stats_;
      ++stats.spec_replay_calls;
      if (pc.verify_gpu_ms > 0 && (count == 0 || commit.timing.gpu_ms > 0) &&
          pc.selector_timed &&
          (!pending_k_ || pending_proposal_.submission.timing.gpu_ms > 0))
        ++stats.spec_gpu_timed_rounds;
      stats.spec_accepted_tokens +=
          std::min(count, std::size_t(pc.verified.accepted_proposals));
      stats.spec_commit_gpu_ms += commit.timing.gpu_ms;
    }
    const bool unclipped =
        (count == pc.verified.licensed_outputs.size()) &&
        (produced_ + count < std::uint32_t(request_.max_new_tokens));
    if (pending_mtp_verify_) {
      if (auto *ctrl = mtp_controller()) {
        const double cycle_gpu_ms = pending_proposal_.submission.timing.gpu_ms +
                                    pc.verify_gpu_ms + pc.selector_gpu_ms +
                                    commit.timing.gpu_ms;
        const double cycle_wall_ms =
            pending_cost_->phase_ms(RoundCostRecorder::draft) +
            pending_cost_->phase_ms(RoundCostRecorder::verify) +
            pending_cost_->phase_ms(RoundCostRecorder::sampling) +
            pending_cost_->phase_ms(RoundCostRecorder::replay);
        ctrl->observe(pending_k_, std::uint32_t(pc.verified.accepted_proposals),
                      cycle_gpu_ms > 0.0 ? cycle_gpu_ms : cycle_wall_ms,
                      unclipped, batch_slots);
      }
    } else {
      const double draft_scale =
          slot_scale / (1.0 + 0.35 * (slot_scale - 1.0));
      const double commit_scale =
          slot_scale / (1.0 + 0.15 * (slot_scale - 1.0));
      owner_.spec_controller.observe_round(
          slot_, pending_k_, std::uint32_t(pc.verified.accepted_proposals),
          unclipped,
          pending_cost_->phase_ms(RoundCostRecorder::draft) * draft_scale,
          pending_cost_->phase_ms(RoundCostRecorder::verify) * slot_scale,
          (pending_cost_->phase_ms(RoundCostRecorder::sampling) +
           pending_cost_->phase_ms(RoundCostRecorder::replay)) *
              commit_scale,
          batch_total_m, pending_proposal_.context_rows_projected > 16U);
    }

    if (cancelled_->load(std::memory_order_acquire))
      return ExecutionResult{.kind = EventKind::aborted,
                             .finish_reason = "cancelled"};
    const bool stopped = pending_cost_->measure(RoundCostRecorder::publish, [&] {
      for (std::size_t i = 0; i < count; ++i)
        if (!publish_token(pc.verified.licensed_outputs[i]))
          return true;
      if (produced_ + count == std::uint32_t(request_.max_new_tokens))
        publisher_.finish();
      return false;
    });
    pending_cost_->finish(std::uint32_t(count));
    pending_cost_.reset();
    if (stopped)
      return finish_task(ExecutionResult{.finish_reason = "stop"});
    produced_ += std::uint32_t(count);
    anchor_ = std::uint32_t(pc.verified.licensed_outputs.at(count - 1));
    if (pending_mtp_verify_) {
      mtp_committed_tokens_.clear();
      mtp_committed_tokens_.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        mtp_committed_tokens_.push_back(
            std::uint32_t(pc.verified.licensed_outputs[i]));
      }
    }
    if (produced_ == std::uint32_t(request_.max_new_tokens)) {
      publisher_.finish();
      return finish_task(ExecutionResult{});
    }
    return std::nullopt;
  }

  std::optional<ExecutionResult> advance_contended(std::size_t contended_cap,
                                                   bool had_active_decodes) {
    if (cancelled_->load(std::memory_order_acquire)) {
      drain_in_flight_prefill();
      drain_in_flight_ar();
      drain_in_flight_commit();
      return ExecutionResult{.kind = EventKind::aborted,
                             .finish_reason = "cancelled"};
    }
    if (in_flight_commit_.has_value() &&
        (had_active_decodes || owner_.active_occupied_count() > 1U)) {
      flush_in_flight_commit();
    }
    if (offset_ < prompt_.size()) {
      if (!blend_attempted_ && owner_.blend_cache &&
          owner_.blend_cache->enabled() && program().cache_blend_eligible() &&
          (prompt_.size() - offset_) >= detail::kBlendDefaultMinSuffixTokens &&
          !had_active_decodes && owner_.active_occupied_count() == 1U &&
          !mtp_active() && !in_flight_prefill_.has_value()) {
        blend_attempted_ = true;
        const auto pf_begin = RoundCostRecorder::Clock::now();
        step_ = with_page_pressure([&] {
          return program().run_blend_prefill(
              prompt_, static_cast<std::uint32_t>(offset_), *owner_.blend_cache,
              owner_.blend_model_exec_hash, sampling_, false);
        });
        const std::size_t consumed = prompt_.size() - offset_;
        const double pf_wall = RoundCostRecorder::milliseconds(pf_begin);
        const double pf_ms =
            step_.submission.timing.gpu_ms > 0.0 ? step_.submission.timing.gpu_ms
                                                 : pf_wall;
        owner_.scheduler.observe_prefill_chunk(consumed, pf_ms,
                                               had_active_decodes);
        committed_tokens_.insert(
            committed_tokens_.end(),
            prompt_.begin() + static_cast<std::ptrdiff_t>(offset_),
            prompt_.end());
        offset_ = prompt_.size();
        save_prefix();
        std::scoped_lock lock(backend_.stats_mutex_);
        auto &stats = backend_.backend_stats_;
        ++stats.ar_steps;
        stats.actual_t = stats.actual_m = step_.tokens;
        stats.actual_k = 0;
        stats.ar_layer_calls += step_.executed_layers;
        stats.short_matrix_calls += step_.short_matrix_calls;
        stats.short_lane_calls += step_.short_lane_calls;
        stats.ar_host_read_bytes += step_.host_read_bytes;
        accumulate_prefill_telemetry_locked(stats, step_.prefill_delta);
        if (step_.submission.timing.gpu_ms > 0) {
          ++stats.ar_gpu_timed_steps;
          stats.ar_prefill_gpu_ms += step_.submission.timing.gpu_ms;
        }
        return std::nullopt;
      }
      const bool was_first_chunk = (offset_ == 0);
      const auto pf_contract = owner_.effective_prefill_contract();
      const auto canonical =
          prefill_next_chunk(offset_, prompt_.size(), pf_contract);
      const std::uint32_t primary_ws_rows =
          owner_.program_at(0).memory_plan().workspace_rows;
      const std::uint32_t uncontended_cap =
          (program().ane_offload_active() && program().ane_chunk_tokens() > 0U)
              ? std::min<std::uint32_t>(primary_ws_rows,
                                        program().ane_chunk_tokens())
              : primary_ws_rows;
      auto count =
          (!had_active_decodes && contended_cap == 0 &&
           owner_.active_occupied_count() == 1U)
              ? prefill_next_uncontended_chunk(offset_, prompt_.size(),
                                               pf_contract, uncontended_cap)
              : prefill_next_contended_chunk(offset_, prompt_.size(),
                                             pf_contract, contended_cap);
      const std::size_t pre_tail = pre_tail_anchor_offset();
      if (pre_tail > offset_ && offset_ + count > pre_tail &&
          will_save_sparse_anchor(pre_tail)) {
        count = pre_tail - offset_;
      }
      const std::size_t think_anchor = open_thinking_anchor_offset();
      if (think_anchor > offset_ && offset_ + count > think_anchor) {
        count = think_anchor - offset_;
      }
      const auto chunk_span = std::span(prompt_).subspan(offset_, count);
      const bool is_last_chunk = (offset_ + count == prompt_.size());
      const bool need_sync_anchor =
          will_save_sparse_anchor(offset_ + count) ||
          (think_anchor > 0U && offset_ + count == think_anchor);

      if (!is_last_chunk && !had_active_decodes &&
          owner_.active_occupied_count() == 1U && !mtp_active() &&
          !need_sync_anchor && program().can_pipeline_prefill(count)) {
        auto next_pf = with_page_pressure(
            [&] { return program().enqueue_prefill_step(chunk_span); });
        if (in_flight_prefill_.has_value()) {
          flush_in_flight_prefill(false);
        }
        in_flight_prefill_ = std::move(next_pf);
        offset_ += count;
        committed_tokens_.insert(committed_tokens_.end(), chunk_span.begin(),
                                 chunk_span.end());
        std::scoped_lock lock(backend_.stats_mutex_);
        auto &stats = backend_.backend_stats_;
        ++stats.ar_steps;
        stats.actual_t = stats.actual_m = std::uint32_t(count);
        stats.actual_k = 0;
        stats.ar_layer_calls += std::uint32_t(owner_.config.num_hidden_layers);
        return std::nullopt;
      }

      if (in_flight_prefill_.has_value()) {
        flush_in_flight_prefill(had_active_decodes);
      }

      const auto pf_begin = RoundCostRecorder::Clock::now();
      step_ = with_page_pressure([&] {
        return is_last_chunk ? program().run(chunk_span, sampling_, false, true)
                             : program().run_trunk(chunk_span, true);
      });
      MetalSubmission mtp_prime_sub{};
      if (mtp_active()) {
        mtp_prime_sub =
            mtp()->prime_prefill_chunk(chunk_span, was_first_chunk, true);
      }
      const double pf_wall = RoundCostRecorder::milliseconds(pf_begin);
      const double pf_ms =
          step_.submission.timing.gpu_ms > 0.0
              ? (step_.submission.timing.gpu_ms + mtp_prime_sub.timing.gpu_ms)
              : pf_wall;
      owner_.scheduler.observe_prefill_chunk(count, pf_ms, had_active_decodes);
      offset_ += count;
      committed_tokens_.insert(committed_tokens_.end(), chunk_span.begin(),
                               chunk_span.end());
      if (offset_ == prompt_.size())
        save_prefix();
      else
        maybe_save_sparse_anchor();

      std::scoped_lock lock(backend_.stats_mutex_);
      auto &stats = backend_.backend_stats_;
      ++stats.ar_steps;
      if (count < canonical)
        ++stats.contended_prefill_chunks;
      stats.actual_t = stats.actual_m = step_.tokens;
      stats.actual_k = 0;
      stats.ar_layer_calls += step_.executed_layers;
      stats.short_matrix_calls += step_.short_matrix_calls;
      stats.short_lane_calls += step_.short_lane_calls;
      stats.ar_host_read_bytes += step_.host_read_bytes;
      accumulate_prefill_telemetry_locked(stats, step_.prefill_delta);
      if (step_.submission.timing.gpu_ms > 0) {
        ++stats.ar_gpu_timed_steps;
        stats.ar_prefill_gpu_ms +=
            step_.submission.timing.gpu_ms + mtp_prime_sub.timing.gpu_ms;
      }
      return std::nullopt;
    }
    if (in_flight_prefill_.has_value()) {
      flush_in_flight_prefill(had_active_decodes);
    }
    if (produced_ == 0)
      resolve_anchor_if_needed();
    auto out = draft() ? advance_dflash()
                       : (mtp_active() ? advance_mtp() : advance_ar());
    if (out.has_value() && out->kind == EventKind::finished) {
      save_post_decode_prefix();
    }
    return out;
  }

  std::optional<ExecutionResult> advance() override {
    const bool ext_contended =
        owner_.external_gpu_contended.load(std::memory_order_acquire);
    const std::size_t cap =
        ext_contended ? owner_.scheduler.calibrated_contended_budget() : 0;
    return advance_contended(cap, ext_contended);
  }

  std::optional<RequestStats> request_stats() const override {
    RequestStats result = request_stats_;
    result.prefix_restored_tokens = matched_tokens_;
    if (result.spec_rounds == 0)
      result.decode_mode = "ar";
    else if (result.ar_steps > 0)
      result.decode_mode = "mixed";
    else
      result.decode_mode = draft() ? "dflash2" : "mtp";
    return result;
  }

private:
  void note_fallback(const char *reason) {
    if (request_stats_.fallback_reason.empty())
      request_stats_.fallback_reason = reason;
  }

  void record_ar_step() {
    ++request_stats_.ar_steps;
    if (mtp() != nullptr && !mtp_primed_ && matched_tokens_ > 0)
      note_fallback("mtp_prefix_not_restored");
  }

  void record_spec_step(std::uint32_t depth, std::uint32_t max_k = 1U) {
    if (depth > 0U)
      ++request_stats_.spec_rounds;
    else if (max_k > 0U)
      record_ar_step();
  }

  QwenForward &program() { return owner_.program_at(slot_); }
  const QwenForward &program() const { return owner_.program_at(slot_); }
  DFlashForward *draft() { return owner_.draft_at(slot_); }
  const DFlashForward *draft() const { return owner_.draft_at(slot_); }
  MtpForward *mtp() { return owner_.mtp_at(slot_); }
  const MtpForward *mtp() const { return owner_.mtp_at(slot_); }
  MtpDepthController *mtp_controller() {
    return owner_.mtp_controller_at(slot_);
  }
  [[nodiscard]] bool mtp_active() const noexcept {
    return mtp() != nullptr && mtp_primed_ && !mtp_exited_;
  }
  TargetGpuVerifier *verifier() { return owner_.verifier_at(slot_); }

  void flush_in_flight_prefill(bool had_active_decodes) {
    if (!in_flight_prefill_.has_value())
      return;
    auto done = program().finish_prefill_step(std::move(*in_flight_prefill_));
    in_flight_prefill_.reset();
    const double pf_ms = done.submission.timing.gpu_ms > 0.0
                             ? done.submission.timing.gpu_ms
                             : done.submission.timing.wall_ms;
    owner_.scheduler.observe_prefill_chunk(done.tokens, pf_ms,
                                           had_active_decodes);
    std::scoped_lock lock(backend_.stats_mutex_);
    auto &stats = backend_.backend_stats_;
    stats.short_matrix_calls += done.short_matrix_calls;
    stats.short_lane_calls += done.short_lane_calls;
    stats.ar_host_read_bytes += done.host_read_bytes;
    accumulate_prefill_telemetry_locked(stats, done.prefill_delta);
    if (done.submission.timing.gpu_ms > 0.0) {
      ++stats.ar_gpu_timed_steps;
      stats.ar_prefill_gpu_ms += done.submission.timing.gpu_ms;
    }
  }

  void drain_in_flight_prefill() noexcept {
    if (in_flight_prefill_.has_value()) {
      try {
        program().drain_prefill_step(std::move(*in_flight_prefill_));
      } catch (...) {
      }
      in_flight_prefill_.reset();
    }
  }

  struct InFlightCommit {
    MetalPendingSubmission pending;
    bool other_phases_gpu_timed = false;
  };

  void flush_in_flight_commit() {
    if (!in_flight_commit_.has_value())
      return;
    auto ic = std::move(*in_flight_commit_);
    in_flight_commit_.reset();
    auto done = owner_.context.wait_submission(std::move(ic.pending));
    if (done.timing.gpu_ms > 0.0) {
      last_commit_gpu_ms_ = done.timing.gpu_ms;
    }
    std::scoped_lock lock(backend_.stats_mutex_);
    auto &stats = backend_.backend_stats_;
    stats.spec_commit_gpu_ms += done.timing.gpu_ms;
    if (ic.other_phases_gpu_timed && done.timing.gpu_ms > 0.0) {
      ++stats.spec_gpu_timed_rounds;
    }
  }

  void drain_in_flight_commit() noexcept {
    if (in_flight_commit_.has_value()) {
      try {
        flush_in_flight_commit();
      } catch (...) {
        in_flight_commit_.reset();
      }
    }
  }

  template <class ProgT>
  static constexpr bool kProgramHasAsyncVerify = requires(
      ProgT &prog, std::span<const std::uint32_t> toks,
      const GpuSamplingOptions &samp) {
    static_cast<bool>(prog.can_enqueue_verify(toks, samp));
    prog.finish_verify(prog.enqueue_verify(toks, samp));
  };

  template <class ProgT>
  bool try_pipelined_verify_and_select(
      ProgT &prog, std::span<const std::uint32_t> input,
      std::span<const SelectorStep> steps, RoundCostRecorder &cost,
      TargetGpuVerifyDecision &out_selected) {
    if (controls_) {
      auto trial = *controls_;
      std::array<PositionSamplingRow, 8> rows{};
      std::array<std::optional<std::uint32_t>, 8> forced{};
      std::array<std::array<std::uint32_t, 2>, 8> ranges{};
      std::vector<RequestPenaltyToken> penalties;
      const auto width = steps.size() + 1;
      const auto pos = prog.position();
      bool has_controls = request_.sampling.has_penalties();
      bool needs_sample = false;
      for (std::size_t i = 0; i < width; ++i) {
        forced.at(i) = trial.forced_token();
        has_controls |= forced[i].has_value();
        needs_sample |= !forced[i];
        rows[i] = controlled_row(pos + i + 1, !forced[i]);
        ranges[i][0] = std::uint32_t(penalties.size());
        if (!forced[i])
          trial.append_penalties(penalties);
        ranges[i][1] = std::uint32_t(penalties.size()) - ranges[i][0];
        if (i < steps.size())
          trial.accept(steps[i].token);
      }
      if (has_controls) {
        step_ = cost.measure(RoundCostRecorder::verify, [&] {
          return with_page_pressure([&] { return prog.verify(input, sampling_); });
        });
        if (cancelled_->load(std::memory_order_acquire))
          return true;
        const auto sample_begin = RoundCostRecorder::Clock::now();
        PositionSamplingResult selected;
        try {
          if (needs_sample)
            selected = controlled_sampler().sample(std::span(rows).first(width),
                std::span(ranges).first(width), penalties, 0U);
        } catch (...) {
          prog.abort_verify();
          throw;
        }
        out_selected.host_read_bytes = selected.host_read_bytes;
        out_selected.submission = std::move(selected.submission);
        auto &verified = out_selected.verification;
        for (std::size_t i = 0; i < width; ++i) {
          const auto token = forced[i] ? *forced[i] : selected.tokens[i];
          verified.licensed_outputs.push_back(std::int32_t(token));
          if (i == steps.size())
            break;
          if (std::int32_t(token) != steps[i].token) {
            verified.used_correction = true;
            break;
          }
          ++verified.accepted_proposals;
        }
        cost.add_select(RoundCostRecorder::milliseconds(sample_begin));
        return true;
      }
    }
    if constexpr (kProgramHasAsyncVerify<ProgT>) {
      if (owner_.active_occupied_count() == 1U && verifier() != nullptr &&
          verifier()->can_enqueue_select(true) &&
          prog.can_enqueue_verify(input, sampling_)) {
        const auto verify_pos = prog.position();
        std::optional<TargetGpuPendingSelect> pend_sel;
        step_ = cost.measure(RoundCostRecorder::verify, [&] {
          auto pend_ver = with_page_pressure(
              [&] { return prog.enqueue_verify(input, sampling_); });
          try {
            pend_sel.emplace(
                verifier()->enqueue_select(steps, sampling_, verify_pos, true));
          } catch (...) {
            try {
              (void)prog.finish_verify(std::move(pend_ver));
            } catch (...) {
            }
            prog.abort_verify();
            throw;
          }
          return prog.finish_verify(std::move(pend_ver));
        });
        if (cancelled_->load(std::memory_order_acquire)) {
          if (pend_sel->async_icb) {
            (void)owner_.context.wait_submission(std::move(pend_sel->pending));
          }
          return true;
        }
        const auto sample_begin = RoundCostRecorder::Clock::now();
        try {
          out_selected = verifier()->finish_select(std::move(*pend_sel));
        } catch (...) {
          prog.abort_verify();
          throw;
        }
        cost.add_select(RoundCostRecorder::milliseconds(sample_begin));
        return true;
      }
    }
    (void)prog;
    (void)input;
    (void)steps;
    (void)cost;
    (void)out_selected;
    return false;
  }

  void drain_in_flight_ar() noexcept {
    if (in_flight_ar_.has_value()) {
      const bool was_valid = in_flight_ar_->valid();
      try {
        program().drain_ar_step(std::move(*in_flight_ar_));
        if (was_valid) {
          committed_tokens_.push_back(in_flight_ar_token_);
        }
      } catch (...) {
      }
      in_flight_ar_.reset();
      in_flight_ar_token_ = 0;
    }
  }

  std::optional<ExecutionResult> advance_ar() {
    const auto token = std::int32_t(step_.token);
    if (!publish_token(token)) {
      drain_in_flight_ar();
      return ExecutionResult{.finish_reason = "stop"};
    }
    ++produced_;
    if (produced_ == std::uint32_t(request_.max_new_tokens)) {
      drain_in_flight_ar();
      publisher_.finish();
      return ExecutionResult{};
    }
    const auto ar_begin = RoundCostRecorder::Clock::now();
    const bool solo_async =
        owner_.active_occupied_count() == 1U &&
        (!controls_ || !controls_->needs_intervention_within(2U)) &&
        program().can_pipeline_ar(sampling_);
    const auto max_ctx = std::uint32_t(backend_.options_.max_context);
    if (in_flight_ar_.has_value()) {
      std::optional<QwenAsyncArStep> next_ar;
      if (solo_async &&
          produced_ + 1U < std::uint32_t(request_.max_new_tokens) &&
          program().ar_virtual_position() < max_ctx) {
        next_ar = with_page_pressure(
            [&] { return program().enqueue_ar_step(sampling_, false, 0U); });
      }
      step_ = program().finish_ar_step(std::move(*in_flight_ar_));
      committed_tokens_.push_back(in_flight_ar_token_);
      in_flight_ar_ = std::move(next_ar);
      in_flight_ar_token_ = in_flight_ar_.has_value() ? step_.token : 0U;
    } else if (solo_async && program().ar_virtual_position() < max_ctx) {
      auto first_ar = with_page_pressure([&] {
        return program().enqueue_ar_step(sampling_, true, std::uint32_t(token));
      });
      std::optional<QwenAsyncArStep> second_ar;
      if (produced_ + 1U < std::uint32_t(request_.max_new_tokens) &&
          program().ar_virtual_position() < max_ctx) {
        second_ar = with_page_pressure(
            [&] { return program().enqueue_ar_step(sampling_, false, 0U); });
      }
      step_ = program().finish_ar_step(std::move(first_ar));
      committed_tokens_.push_back(std::uint32_t(token));
      in_flight_ar_ = std::move(second_ar);
      in_flight_ar_token_ = in_flight_ar_.has_value() ? step_.token : 0U;
    } else {
      const std::array<std::uint32_t, 1> input{std::uint32_t(token)};
      step_ = with_page_pressure([&] { return program().run(input, sampling_); });
      apply_controlled_sample(step_);
      committed_tokens_.push_back(std::uint32_t(token));
    }
    record_ar_step();
    const double ar_wall = RoundCostRecorder::milliseconds(ar_begin);
    owner_.spec_controller.observe_ar_step(
        step_.submission.timing.gpu_ms > 0 ? step_.submission.timing.gpu_ms
                                           : ar_wall);
    std::scoped_lock lock(backend_.stats_mutex_);
    auto &stats = backend_.backend_stats_;
    ++stats.ar_steps;
    stats.actual_t = stats.actual_m = step_.tokens;
    stats.actual_k = 0;
    stats.ar_layer_calls += step_.executed_layers;
    stats.short_matrix_calls += step_.short_matrix_calls;
    stats.short_lane_calls += step_.short_lane_calls;
    stats.ar_host_read_bytes += step_.host_read_bytes;
    if (step_.submission.timing.gpu_ms > 0) {
      ++stats.ar_gpu_timed_steps;
      stats.ar_decode_gpu_ms += step_.submission.timing.gpu_ms;
    }
    return std::nullopt;
  }

  std::optional<ExecutionResult> advance_dflash() {
    if (produced_ == 0) {
      anchor_ = step_.token;
      if (!publish_token(std::int32_t(anchor_)))
        return ExecutionResult{.finish_reason = "stop"};
      produced_ = 1;
      if (produced_ == std::uint32_t(request_.max_new_tokens)) {
        publisher_.finish();
        return ExecutionResult{};
      }
      return std::nullopt;
    }
    const auto spec_query = make_spec_query();
    const auto decision =
        owner_.spec_controller.select_single_depth(spec_query);
    const auto k = decision.k;
    if (decision.gated && k == 0)
      note_fallback("adaptive_gate");
    if (decision.gated || decision.probe) {
      std::scoped_lock lock(backend_.stats_mutex_);
      if (decision.gated)
        ++backend_.backend_stats_.spec_search_gated_rounds;
      if (decision.probe)
        ++backend_.backend_stats_.spec_search_probe_rounds;
    }

    RoundCostRecorder cost(backend_.backend_stats_, backend_.stats_mutex_, k);
    DFlash2DraftResult proposal;
    if (k) {
      const bool cpu_sel =
          owner_.context.capabilities().profile.probe.apple_gpu_family < 9;
      const bool tokens_only =
          owner_.sampling_contract == SamplingContract::PositionCoupledV1 &&
          verifier() != nullptr;
      sync_draft_target_epoch(*draft(), program().feature_epoch());
      proposal = cost.measure(RoundCostRecorder::draft, [&] {
        return invoke_draft_propose(
            *draft(), anchor_, program().position(), k,
            request_.sampling.temperature, request_.sampling.seed, true, false,
            cpu_sel, tokens_only);
      });
      flush_in_flight_commit();
    }
    std::vector<std::uint32_t> input{anchor_};
    input.insert(input.end(), proposal.selection.tokens.begin(),
                 proposal.selection.tokens.end());
    TargetGpuVerifyDecision preselected{};
    const bool pipelined_vs = try_pipelined_verify_and_select(
        program(), input, proposal.selection.steps, cost, preselected);
    record_spec_step(k, spec_query.max_k);
    if (!pipelined_vs) {
      step_ = cost.measure(RoundCostRecorder::verify, [&] {
        return with_page_pressure(
            [&] { return program().verify(input, sampling_); });
      });
    }
    flush_in_flight_commit();
    {
      std::scoped_lock lock(backend_.stats_mutex_);
      auto &stats = backend_.backend_stats_;
      ++stats.spec_rounds;
      ++stats.spec_target_verify_calls;
      stats.spec_proposal_tokens += k;
      stats.short_matrix_calls +=
          proposal.short_matrix_calls + step_.short_matrix_calls;
      stats.short_lane_calls +=
          proposal.short_lane_calls + step_.short_lane_calls;
      stats.spec_context_rows_projected += proposal.context_rows_projected;
      stats.spec_context_rows_reused += proposal.context_rows_reused;
      stats.spec_context_rows_launched += proposal.context_rows_launched;
      stats.spec_host_read_bytes +=
          step_.host_read_bytes + proposal.host_read_bytes;
      stats.spec_draft_gpu_ms += proposal.submission.timing.gpu_ms;
      stats.spec_verify_gpu_ms += step_.submission.timing.gpu_ms;
      stats.actual_t = stats.actual_m = std::uint32_t(input.size());
      stats.actual_k = k;
    }
    if (cancelled_->load(std::memory_order_acquire)) {
      program().abort_verify();
      return ExecutionResult{.kind = EventKind::aborted,
                             .finish_reason = "cancelled"};
    }
    VerificationResult verified;
    bool selector_timed = !verifier();
    if (pipelined_vs) {
      verified = std::move(preselected.verification);
      selector_timed = preselected.submission.timing.gpu_ms > 0;
      std::scoped_lock lock(backend_.stats_mutex_);
      backend_.backend_stats_.spec_host_read_bytes +=
          preselected.host_read_bytes;
      backend_.backend_stats_.spec_verify_selector_gpu_ms +=
          preselected.submission.timing.gpu_ms;
    } else {
      const auto sample_begin = RoundCostRecorder::Clock::now();
      try {
        if (verifier()) {
          auto selected = verifier()->select(proposal.selection.steps, sampling_,
                                             program().position());
          verified = std::move(selected.verification);
          std::scoped_lock lock(backend_.stats_mutex_);
          backend_.backend_stats_.spec_host_read_bytes +=
              selected.host_read_bytes;
          backend_.backend_stats_.spec_verify_selector_gpu_ms +=
              selected.submission.timing.gpu_ms;
          selector_timed = selected.submission.timing.gpu_ms > 0;
        } else {
          auto logits = program().read_logits(std::uint32_t(input.size()));
          {
            std::scoped_lock lock(backend_.stats_mutex_);
            backend_.backend_stats_.spec_host_read_bytes += logits.size() * 2;
          }
          std::vector<TargetDistribution> distributions;
          for (std::size_t row = 0; row < input.size(); ++row)
            distributions.push_back(target_distribution(
                std::span(logits).subspan(row * owner_.config.vocab_size,
                                          owner_.config.vocab_size),
                sampling_));
          verified = verify_speculative(
              proposal.selection.steps, distributions,
              request_.sampling.temperature, request_.sampling.seed,
              program().position());
        }
      } catch (...) {
        program().abort_verify();
        throw;
      }
      cost.add_select(RoundCostRecorder::milliseconds(sample_begin));
    }
    const auto count = publisher_.preview_count(verified.licensed_outputs);
    if (cancelled_->load(std::memory_order_acquire)) {
      program().abort_verify();
      return ExecutionResult{.kind = EventKind::aborted,
                             .finish_reason = "cancelled"};
    }
    const bool terminal_round =
        (count < verified.licensed_outputs.size()) ||
        (produced_ + count >= std::uint32_t(request_.max_new_tokens));
    const bool other_timed =
        step_.submission.timing.gpu_ms > 0 && selector_timed &&
        (!k || proposal.submission.timing.gpu_ms > 0);
    MetalIcbProfile *async_commit_icb = nullptr;
    if (!terminal_round && count > 0 && owner_.active_occupied_count() == 1U) {
      async_commit_icb = cost.measure(RoundCostRecorder::replay, [&] {
        return program().prepare_commit_prefix(std::uint32_t(count));
      });
    }
    if (async_commit_icb != nullptr) {
      auto pend_c = cost.measure(RoundCostRecorder::replay, [&] {
        return owner_.context.replay_icb_async(*async_commit_icb);
      });
      (void)cost.measure(RoundCostRecorder::replay, [&] {
        return program().finish_commit_prefix(std::uint32_t(count), true);
      });
      if (count > 0) {
        committed_tokens_.insert(committed_tokens_.end(), input.begin(),
                                 input.begin() + std::ptrdiff_t(count));
      }
      in_flight_commit_ = InFlightCommit{
          .pending = std::move(pend_c), .other_phases_gpu_timed = other_timed};
      cost.mark_committed(std::uint32_t(count));
      std::scoped_lock lock(backend_.stats_mutex_);
      auto &stats = backend_.backend_stats_;
      ++stats.spec_replay_calls;
      stats.spec_accepted_tokens +=
          std::min(count, std::size_t(verified.accepted_proposals));
    } else {
      auto commit = cost.measure(RoundCostRecorder::replay, [&] {
        return program().commit_prefix(std::uint32_t(count));
      });
      if (count > 0) {
        committed_tokens_.insert(committed_tokens_.end(), input.begin(),
                                 input.begin() + std::ptrdiff_t(count));
      }
      if (commit.timing.gpu_ms > 0.0) {
        last_commit_gpu_ms_ = commit.timing.gpu_ms;
      }
      cost.mark_committed(std::uint32_t(count));
      std::scoped_lock lock(backend_.stats_mutex_);
      auto &stats = backend_.backend_stats_;
      ++stats.spec_replay_calls;
      if (other_timed && (count == 0 || commit.timing.gpu_ms > 0))
        ++stats.spec_gpu_timed_rounds;
      stats.spec_accepted_tokens +=
          std::min(count, std::size_t(verified.accepted_proposals));
      stats.spec_commit_gpu_ms += commit.timing.gpu_ms;
    }
    const bool unclipped =
        (count == verified.licensed_outputs.size()) &&
        (produced_ + count < std::uint32_t(request_.max_new_tokens));
    owner_.spec_controller.observe_round(
        slot_, k, std::uint32_t(verified.accepted_proposals), unclipped,
        cost.phase_ms(RoundCostRecorder::draft),
        cost.phase_ms(RoundCostRecorder::verify),
        cost.phase_ms(RoundCostRecorder::sampling) +
            cost.phase_ms(RoundCostRecorder::replay),
        std::uint32_t(input.size()), proposal.context_rows_projected > 16U);

    if (cancelled_->load(std::memory_order_acquire)) {
      drain_in_flight_commit();
      return ExecutionResult{.kind = EventKind::aborted,
                             .finish_reason = "cancelled"};
    }
    const bool stopped = cost.measure(RoundCostRecorder::publish, [&] {
      for (std::size_t i = 0; i < count; ++i)
        if (!publish_token(verified.licensed_outputs[i]))
          return true;
      if (produced_ + count == std::uint32_t(request_.max_new_tokens))
        publisher_.finish();
      return false;
    });
    cost.finish(std::uint32_t(count));
    if (stopped) {
      flush_in_flight_commit();
      return ExecutionResult{.finish_reason = "stop"};
    }
    produced_ += std::uint32_t(count);
    anchor_ = std::uint32_t(verified.licensed_outputs.at(count - 1));
    if (produced_ == std::uint32_t(request_.max_new_tokens)) {
      flush_in_flight_commit();
      publisher_.finish();
      return ExecutionResult{};
    }
    return std::nullopt;
  }

  std::optional<ExecutionResult> advance_mtp() {
    if (produced_ == 0) {
      anchor_ = step_.token;
      if (!publish_token(std::int32_t(anchor_)))
        return ExecutionResult{.finish_reason = "stop"};
      produced_ = 1;
      mtp_committed_tokens_.assign(1U, anchor_);
      if (produced_ == std::uint32_t(request_.max_new_tokens)) {
        publisher_.finish();
        return ExecutionResult{};
      }
      return std::nullopt;
    }

    const auto remaining = std::uint32_t(request_.max_new_tokens) - produced_;
    const auto available =
        std::uint32_t(backend_.options_.max_context) - program().position();
    if (!available)
      throw std::out_of_range("MTP target context exhausted");

    auto *ctrl = mtp_controller();
    if (ctrl && ctrl->should_exit()) {
      mtp_exited_ = true;
      last_mtp_round_committed_.clear();
      note_fallback("mtp_adaptive_exit");
      record_ar_step();
      step_.token = anchor_;
      // Step anchor_ in AR and resume standard AR stream cleanly.
      const std::array<std::uint32_t, 1> input{anchor_};
      const auto ar_begin = RoundCostRecorder::Clock::now();
      step_ = with_page_pressure([&] { return program().run(input, sampling_); });
      apply_controlled_sample(step_);
      committed_tokens_.push_back(anchor_);
      flush_in_flight_commit();
      const double ar_wall = RoundCostRecorder::milliseconds(ar_begin);
      owner_.spec_controller.observe_ar_step(
          step_.submission.timing.gpu_ms > 0 ? step_.submission.timing.gpu_ms
                                             : ar_wall);
      std::scoped_lock lock(backend_.stats_mutex_);
      auto &stats = backend_.backend_stats_;
      ++stats.ar_steps;
      stats.actual_t = stats.actual_m = step_.tokens;
      stats.actual_k = 0;
      stats.ar_layer_calls += step_.executed_layers;
      stats.short_matrix_calls += step_.short_matrix_calls;
      stats.short_lane_calls += step_.short_lane_calls;
      stats.ar_host_read_bytes += step_.host_read_bytes;
      accumulate_prefill_telemetry_locked(stats, step_.prefill_delta);
      if (step_.submission.timing.gpu_ms > 0) {
        ++stats.ar_gpu_timed_steps;
        stats.ar_decode_gpu_ms += step_.submission.timing.gpu_ms;
      }
      return std::nullopt;
    }

    const std::uint32_t max_allowed_k =
        std::min({owner_.slot_mtp_draft_tokens, remaining - 1U, available - 1U});
    const bool was_probing = ctrl && ctrl->is_probing(1U);
    const std::uint32_t k =
        ctrl ? ctrl->select_depth(max_allowed_k, 1U) : max_allowed_k;

    if (k == 0U) {
      if (max_allowed_k > 0U) {
        if (was_probing) {
          std::scoped_lock lock(backend_.stats_mutex_);
          ++backend_.backend_stats_.spec_search_probe_rounds;
        } else {
          record_ar_step();
          note_fallback("adaptive_gate");
          std::scoped_lock lock(backend_.stats_mutex_);
          ++backend_.backend_stats_.spec_search_gated_rounds;
        }
      }
      const auto k0_begin = RoundCostRecorder::Clock::now();
      MetalSubmission fold_sub{};
      const bool keep_rolling_mtp =
          owner_.rolling_prefix_cache && owner_.prefix_cache.enabled();
      if ((remaining > 1U || keep_rolling_mtp) && available > 1U) {
        fold_sub = mtp()->fold_committed_only(mtp_committed_tokens_, true);
      }
      const std::array<std::uint32_t, 1> input{anchor_};
      step_ = with_page_pressure([&] { return program().run(input, sampling_); });
      apply_controlled_sample(step_);
      committed_tokens_.push_back(anchor_);
      if ((remaining > 1U || keep_rolling_mtp) && available > 1U) {
        last_mtp_round_committed_.assign(1U, anchor_);
      } else {
        last_mtp_round_committed_.clear();
      }
      flush_in_flight_commit();
      const double k0_wall = RoundCostRecorder::milliseconds(k0_begin);
      const double k0_ms =
          step_.submission.timing.gpu_ms > 0.0
              ? (fold_sub.timing.gpu_ms + step_.submission.timing.gpu_ms)
              : k0_wall;
      if (ctrl && max_allowed_k > 0U) {
        ctrl->observe(0U, 0U, k0_ms, true);
      }
      {
        std::scoped_lock lock(backend_.stats_mutex_);
        auto &stats = backend_.backend_stats_;
        ++stats.ar_steps;
        stats.actual_t = stats.actual_m = step_.tokens;
        stats.actual_k = 0;
        stats.ar_layer_calls += step_.executed_layers;
        stats.short_matrix_calls += step_.short_matrix_calls;
        stats.short_lane_calls += step_.short_lane_calls;
        stats.ar_host_read_bytes += step_.host_read_bytes;
        if (step_.submission.timing.gpu_ms > 0) {
          ++stats.ar_gpu_timed_steps;
          stats.ar_decode_gpu_ms +=
              fold_sub.timing.gpu_ms + step_.submission.timing.gpu_ms;
        }
      }
      const auto next_tok = std::int32_t(step_.token);
      if (!publish_token(next_tok))
        return ExecutionResult{.finish_reason = "stop"};
      ++produced_;
      anchor_ = step_.token;
      mtp_committed_tokens_.assign(1U, anchor_);
      if (produced_ == std::uint32_t(request_.max_new_tokens)) {
        publisher_.finish();
        return ExecutionResult{};
      }
      return std::nullopt;
    }

    RoundCostRecorder cost(backend_.backend_stats_, backend_.stats_mutex_, k);
    const bool tokens_only =
        owner_.sampling_contract == SamplingContract::PositionCoupledV1 &&
        verifier() != nullptr;

    // ICB #1: MtpForward single-CB history fold (c = 1..5) + chained draft (d = 1..K-1)
    auto proposal = cost.measure(RoundCostRecorder::draft, [&] {
      return mtp()->propose(mtp_committed_tokens_, program().position(), k,
                            sampling_, true, tokens_only);
    });
    // Any previous round's async commit_prefix preceded propose on the serialized
    // Metal 4 queue, so harvesting its GPU timing here completes in 0 ns!
    flush_in_flight_commit();

    // ICB #2: Target QwenForward::verify([anchor, d_1..d_k]) + fused single-ICB
    // TargetGpuVerifier (64xT-TG sample + verify_equality) + pipelined commit_prefix
    std::vector<std::uint32_t> input{anchor_};
    input.insert(input.end(), proposal.selection.tokens.begin(),
                 proposal.selection.tokens.end());
    TargetGpuVerifyDecision preselected{};
    const bool pipelined_vs = try_pipelined_verify_and_select(
        program(), input, proposal.selection.steps, cost, preselected);
    record_spec_step(k, max_allowed_k);
    if (!pipelined_vs) {
      step_ = cost.measure(RoundCostRecorder::verify, [&] {
        return with_page_pressure(
            [&] { return program().verify(input, sampling_); });
      });
    }
    {
      std::scoped_lock lock(backend_.stats_mutex_);
      auto &stats = backend_.backend_stats_;
      ++stats.spec_rounds;
      ++stats.spec_target_verify_calls;
      stats.spec_proposal_tokens += k;
      stats.short_matrix_calls +=
          proposal.short_matrix_calls + step_.short_matrix_calls;
      stats.short_lane_calls +=
          proposal.short_lane_calls + step_.short_lane_calls;
      stats.spec_host_read_bytes +=
          step_.host_read_bytes + proposal.host_read_bytes;
      stats.spec_draft_gpu_ms += proposal.submission.timing.gpu_ms;
      stats.spec_verify_gpu_ms += step_.submission.timing.gpu_ms;
      stats.actual_t = stats.actual_m = std::uint32_t(input.size());
      stats.actual_k = k;
    }
    if (cancelled_->load(std::memory_order_acquire)) {
      program().abort_verify();
      return ExecutionResult{.kind = EventKind::aborted,
                             .finish_reason = "cancelled"};
    }

    VerificationResult verified;
    bool selector_timed = !verifier();
    double sel_gpu_ms = 0.0;
    if (pipelined_vs) {
      verified = std::move(preselected.verification);
      sel_gpu_ms = preselected.submission.timing.gpu_ms;
      selector_timed = sel_gpu_ms > 0.0;
      std::scoped_lock lock(backend_.stats_mutex_);
      backend_.backend_stats_.spec_host_read_bytes +=
          preselected.host_read_bytes;
      backend_.backend_stats_.spec_verify_selector_gpu_ms += sel_gpu_ms;
    } else {
      const auto sample_begin = RoundCostRecorder::Clock::now();
      try {
        if (verifier()) {
          auto selected = verifier()->select(proposal.selection.steps, sampling_,
                                             program().position());
          verified = std::move(selected.verification);
          sel_gpu_ms = selected.submission.timing.gpu_ms;
          selector_timed = sel_gpu_ms > 0.0;
          std::scoped_lock lock(backend_.stats_mutex_);
          backend_.backend_stats_.spec_host_read_bytes +=
              selected.host_read_bytes;
          backend_.backend_stats_.spec_verify_selector_gpu_ms += sel_gpu_ms;
        } else {
          auto logits = program().read_logits(std::uint32_t(input.size()));
          {
            std::scoped_lock lock(backend_.stats_mutex_);
            backend_.backend_stats_.spec_host_read_bytes += logits.size() * 2;
          }
          std::vector<TargetDistribution> distributions;
          for (std::size_t row = 0; row < input.size(); ++row)
            distributions.push_back(target_distribution(
                std::span(logits).subspan(row * owner_.config.vocab_size,
                                          owner_.config.vocab_size),
                sampling_));
          verified = verify_speculative(
              proposal.selection.steps, distributions,
              request_.sampling.temperature, request_.sampling.seed,
              program().position());
        }
      } catch (...) {
        program().abort_verify();
        throw;
      }
      cost.add_select(RoundCostRecorder::milliseconds(sample_begin));
    }

    const auto count = publisher_.preview_count(verified.licensed_outputs);
    if (cancelled_->load(std::memory_order_acquire)) {
      program().abort_verify();
      return ExecutionResult{.kind = EventKind::aborted,
                             .finish_reason = "cancelled"};
    }
    const bool terminal_round =
        (count < verified.licensed_outputs.size()) ||
        (produced_ + count >= std::uint32_t(request_.max_new_tokens));
    const bool other_timed = step_.submission.timing.gpu_ms > 0 &&
                             selector_timed &&
                             proposal.submission.timing.gpu_ms > 0;
    MetalIcbProfile *async_commit_icb = nullptr;
    if (!terminal_round && count > 0 && owner_.active_occupied_count() == 1U) {
      async_commit_icb = cost.measure(RoundCostRecorder::replay, [&] {
        return program().prepare_commit_prefix(std::uint32_t(count));
      });
    }
    double commit_est_gpu_ms = 0.0;
    if (async_commit_icb != nullptr) {
      auto pend_c = cost.measure(RoundCostRecorder::replay, [&] {
        return owner_.context.replay_icb_async(*async_commit_icb);
      });
      (void)cost.measure(RoundCostRecorder::replay, [&] {
        return program().finish_commit_prefix(std::uint32_t(count), true);
      });
      if (count > 0) {
        committed_tokens_.insert(committed_tokens_.end(), input.begin(),
                                 input.begin() + std::ptrdiff_t(count));
      }
      in_flight_commit_ = InFlightCommit{
          .pending = std::move(pend_c), .other_phases_gpu_timed = other_timed};
      commit_est_gpu_ms = last_commit_gpu_ms_;
      cost.mark_committed(std::uint32_t(count));
      std::scoped_lock lock(backend_.stats_mutex_);
      auto &stats = backend_.backend_stats_;
      ++stats.spec_replay_calls;
      stats.spec_accepted_tokens +=
          std::min(count, std::size_t(verified.accepted_proposals));
    } else {
      auto commit = cost.measure(RoundCostRecorder::replay, [&] {
        return program().commit_prefix(std::uint32_t(count));
      });
      if (count > 0) {
        committed_tokens_.insert(committed_tokens_.end(), input.begin(),
                                 input.begin() + std::ptrdiff_t(count));
      }
      if (commit.timing.gpu_ms > 0.0) {
        last_commit_gpu_ms_ = commit.timing.gpu_ms;
      }
      commit_est_gpu_ms = commit.timing.gpu_ms;
      cost.mark_committed(std::uint32_t(count));
      std::scoped_lock lock(backend_.stats_mutex_);
      auto &stats = backend_.backend_stats_;
      ++stats.spec_replay_calls;
      if (other_timed && (count == 0 || commit.timing.gpu_ms > 0))
        ++stats.spec_gpu_timed_rounds;
      stats.spec_accepted_tokens +=
          std::min(count, std::size_t(verified.accepted_proposals));
      stats.spec_commit_gpu_ms += commit.timing.gpu_ms;
    }
    if (count > 0) {
      last_mtp_round_committed_.assign(input.begin(),
                                       input.begin() + std::ptrdiff_t(count));
    } else {
      last_mtp_round_committed_.clear();
    }
    const bool unclipped =
        (count == verified.licensed_outputs.size()) &&
        (produced_ + count < std::uint32_t(request_.max_new_tokens));
    if (ctrl) {
      const double cycle_gpu_ms = proposal.submission.timing.gpu_ms +
                                  step_.submission.timing.gpu_ms + sel_gpu_ms +
                                  commit_est_gpu_ms;
      const double cycle_wall_ms =
          cost.phase_ms(RoundCostRecorder::draft) +
          cost.phase_ms(RoundCostRecorder::verify) +
          cost.phase_ms(RoundCostRecorder::sampling) +
          cost.phase_ms(RoundCostRecorder::replay);
      ctrl->observe(k, std::uint32_t(verified.accepted_proposals),
                    cycle_gpu_ms > 0.0 ? cycle_gpu_ms : cycle_wall_ms,
                    unclipped);
    }

    if (cancelled_->load(std::memory_order_acquire)) {
      drain_in_flight_commit();
      return ExecutionResult{.kind = EventKind::aborted,
                             .finish_reason = "cancelled"};
    }
    const bool stopped = cost.measure(RoundCostRecorder::publish, [&] {
      for (std::size_t i = 0; i < count; ++i)
        if (!publish_token(verified.licensed_outputs[i]))
          return true;
      if (produced_ + count == std::uint32_t(request_.max_new_tokens))
        publisher_.finish();
      return false;
    });
    cost.finish(std::uint32_t(count));
    if (stopped) {
      flush_in_flight_commit();
      return ExecutionResult{.finish_reason = "stop"};
    }

    produced_ += std::uint32_t(count);
    anchor_ = std::uint32_t(verified.licensed_outputs.at(count - 1));
    mtp_committed_tokens_.clear();
    mtp_committed_tokens_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      mtp_committed_tokens_.push_back(
          std::uint32_t(verified.licensed_outputs[i]));
    }
    if (produced_ == std::uint32_t(request_.max_new_tokens)) {
      flush_in_flight_commit();
      publisher_.finish();
      return ExecutionResult{};
    }
    return std::nullopt;
  }

  static void accumulate_prefill_telemetry_locked(
      StatsReport &stats, const QwenPrefillTelemetry &d) noexcept {
    stats.prefill_w4a8_tokens += d.prefill_w4a8_tokens;
    stats.prefill_w4a4_tokens += d.prefill_w4a4_tokens;
    stats.prefill_sage2_tokens += d.prefill_sage2_tokens;
    stats.prefill_minference_blocks_skipped +=
        d.prefill_minference_blocks_skipped;
    stats.prefill_minference_blocks_total += d.prefill_minference_blocks_total;
    stats.prefill_cache_blend_hit_tokens += d.prefill_cache_blend_hit_tokens;
    stats.prefill_cache_blend_recomputed_tokens +=
        d.prefill_cache_blend_recomputed_tokens;
    stats.prefill_ane_offload_tokens += d.prefill_ane_offload_tokens;
    stats.prefill_ane_fallback_tokens += d.prefill_ane_fallback_tokens;
  }

  NativeBackend &backend_;
  NativeQwenState &owner_;
  std::size_t slot_;
  SubmitRequest request_;
  std::shared_ptr<std::atomic<bool>> cancelled_;
  EmitTokens emit_;
  TokenPublisher publisher_;
  GpuSamplingOptions sampling_;
  std::optional<RequestControlState> controls_;
  QwenForwardResult step_;
  std::vector<std::uint32_t> prompt_, pending_input_, mtp_committed_tokens_,
      last_mtp_round_committed_, committed_tokens_;
  std::vector<std::byte> pending_anchor_logits_;
  bool pending_prefill_ = false, pending_was_first_chunk_ = false,
       pending_verify_ = false, pending_mtp_verify_ = false,
       pending_mtp_probe_ = false, needs_anchor_resample_ = false,
       mtp_primed_ = false, mtp_exited_ = false, matched_pinned_prefix_ = false,
       pre_tail_pinned_ = false, post_decode_saved_ = false,
       blend_attempted_ = false;
  std::uint32_t pending_k_ = 0, pending_max_k_ = 0;
  DFlash2DraftResult pending_proposal_;
  std::unique_ptr<RoundCostRecorder> pending_cost_;
  RequestStats request_stats_;
  std::size_t offset_ = 0, matched_tokens_ = 0;
  std::uint64_t matched_entry_id_ = 0, saved_think_anchor_id_ = 0;
  std::uint32_t produced_ = 0, anchor_ = 0, in_flight_ar_token_ = 0;
  double last_commit_gpu_ms_ = 0.25;
  std::optional<QwenAsyncPrefillStep> in_flight_prefill_;
  std::optional<QwenAsyncArStep> in_flight_ar_;
  std::optional<InFlightCommit> in_flight_commit_;
};

NativeBackend::NativeBackend(const LoadOptions &options) : options_(options) {
  static_cast<void>(parse_sampling_contract(options.sampling_contract));
  if (options.max_context < 1024)
    throw std::invalid_argument("native backend requires max_context >= 1024");
  const std::filesystem::path model_dir(options.model_dir);
  std::error_code art_ec;
  const auto default_artifact =
      std::filesystem::is_regular_file(model_dir / "model.miv", art_ec) ||
              !std::filesystem::is_regular_file(model_dir / "model.omlxm5", art_ec)
          ? (model_dir / "model.miv")
          : (model_dir / "model.omlxm5");
  const auto artifact =
      override_or_default(options, "artifact", default_artifact);
  const auto metallib = override_or_default(
      options, "qwen_ar_metallib", model_dir / "omlx_m5_qwen_ar.metallib");
  const auto libraries =
      override_or_default(options, "kernel_directory", metallib.parent_path());
  ar_program_ = std::make_unique<NativeQwenState>(
      model_dir, artifact, libraries,
      static_cast<std::uint32_t>(options.max_context), options);
  if (ar_program_->mtp_shared_kv_pages != 0U) {
    for (std::size_t slot = 0; slot < ar_program_->occupied.size(); ++slot) {
      ar_program_->mtp_at(slot)->set_kv_reclaimer([this] {
        if (!ar_program_->prefix_cache.evict_for_mtp_page_pressure(
                [this](std::size_t s) -> QwenForward & {
                  return ar_program_->program_at(s);
                })) {
          return false;
        }
        std::scoped_lock lock(stats_mutex_);
        ++backend_stats_.prefix_evictions;
        backend_stats_.prefix_pinned_entries =
            ar_program_->prefix_cache.pinned_count();
        return true;
      });
    }
  }
  if (options_.speculative.backend == SpeculativeBackend::dflash2) {
    options_.speculative.draft_tokens = ar_program_->slot_dflash_draft_tokens;
  }
  backend_stats_.backend_kind = "native";
  backend_stats_.model_execution_kind =
      ar_program_->config.num_experts ? "qwen_moe" : "qwen_dense";
  backend_stats_.artifact_id = ar_program_->artifact_id;
  backend_stats_.route_table_id = ar_program_->route_table_id;
  backend_stats_.sampling_contract_id =
      ar_program_->execution_fields.at("sampling");
  backend_stats_.numeric_contract_id =
      ar_program_->execution_fields.at("numeric");
  backend_stats_.chunk_policy_id = ar_program_->execution_fields.at("chunk");
  backend_stats_.execution_identity_id =
      execution_identity_digest(ar_program_->execution_fields);
  backend_stats_.compiled_execution_source_id =
      ar_program_->execution_fields.at("source");
  backend_stats_.kernel_bundle_id = ar_program_->execution_fields.at("kernel");
  backend_stats_.batch_execution_kind =
      ar_program_->shared_head_enabled
          ? "grouped_plus_shared_head_rerun_v1"
          : ar_program_->slot_batch_enabled ? "packed_shared_weights_v2"
                                            : "independent_slots_v1";
  if (options_.lora_mode.empty()) {
    options_.lora_mode = "merge";
  }
  backend_stats_.kernel_family = ar_program_->draft
                                     ? "qwen_dflash2_forward_a16_metal4"
                                     : ar_program_->mtp
                                           ? "qwen_mtp_forward_a16_metal4"
                                           : "qwen_forward_a16_metal4";
  backend_stats_.lora_adapter = ar_program_->lora_adapter_path;
  backend_stats_.lora_mode = ar_program_->lora_adapter_path.empty()
                                 ? std::string{}
                                 : ar_program_->lora_mode_active;
  backend_stats_.lora_buffer_bytes = ar_program_->lora_buffer_bytes;
  backend_stats_.lora_adapted_layers = ar_program_->lora_adapted_layers;
  populate_prefill_config_stats(options_, backend_stats_);
  backend_stats_.model_quant_format = ar_program_->model_quant_format;
  backend_stats_.prefill_linear_mode_effective =
      prefill_linear_mode_to_string(ar_program_->effective_linear_mode);
  backend_stats_.prefill_linear_fallback = ar_program_->prefill_linear_fallback;
  backend_stats_.prefill_linear_fallback_reason =
      ar_program_->prefill_linear_fallback_reason;
  backend_stats_.prefill_ane_active =
      ar_program_->program->ane_offload_active();
  backend_stats_.prefill_ane_split_alpha =
      ar_program_->program->ane_split_alpha();
}

NativeBackend::~NativeBackend() = default;

void NativeBackend::bind_owner(std::thread::id owner) {
  if (owner_.has_value() && *owner_ != owner) {
    throw std::logic_error("native backend owner may only be bound once");
  }
  owner_ = owner;
}

void NativeBackend::assert_owner() const {
  if (!owner_.has_value() || *owner_ != std::this_thread::get_id()) {
    throw std::logic_error(
        "native backend mutation attempted outside the worker thread");
  }
}

ExecutionResult NativeBackend::execute(std::uint64_t /*request_id*/,
                                       const SubmitRequest &request,
                                       const std::atomic<bool> &cancelled,
                                       const EmitTokens &emit_tokens) {
  assert_owner();
  if (request.prompt_tokens.empty()) {
    return {.kind = EventKind::failed,
            .finish_reason = "error",
            .error = "native backend requires a non-empty prompt"};
  }
  if (std::ranges::any_of(request.prompt_tokens,
                          [](std::int32_t token) { return token < 0; })) {
    return {.kind = EventKind::failed,
            .finish_reason = "error",
            .error = "native backend prompt token is negative"};
  }
  GpuSamplingOptions sampling;
  try {
    sampling = sampling_options(request.sampling);
    sampling.contract = parse_sampling_contract(options_.sampling_contract);
  } catch (...) {
    std::scoped_lock lock(stats_mutex_);
    backend_stats_.fallback_reason = "unsupported_sampling";
    throw;
  }
  {
    std::scoped_lock lock(stats_mutex_);
    backend_stats_.fallback_reason.clear();
  }
  if (cancelled.load(std::memory_order_acquire))
    return {.kind = EventKind::aborted, .finish_reason = "cancelled"};

  auto &owner = *ar_program_;
  if (owner.occupied.at(0))
    throw std::logic_error("P09 active slot 0 already occupied");
  owner.occupied.at(0) = true;
  struct Slot0Guard {
    std::vector<bool> &occupied;
    bool armed = true;
    ~Slot0Guard() {
      if (armed)
        occupied.at(0) = false;
    }
  } slot0_guard{owner.occupied};
  // Non-owning aliasing shared_ptr: task lives strictly on this stack frame.
  std::shared_ptr<std::atomic<bool>> cancel_alias(
      std::shared_ptr<std::atomic<bool>>{},
      const_cast<std::atomic<bool> *>(&cancelled));
  NativeTask task(*this, owner, 0, request, std::move(cancel_alias),
                  emit_tokens, sampling);
  slot0_guard.armed = false;
  for (;;) {
    if (auto out = task.advance()) {
      out->request_stats = task.request_stats();
      return *out;
    }
  }
}

std::unique_ptr<BackendTask>
NativeBackend::start(SubmitRequest request,
                     std::shared_ptr<std::atomic<bool>> cancelled,
                     EmitTokens emit) {
  assert_owner();
  if (request.prompt_tokens.empty() ||
      std::ranges::any_of(request.prompt_tokens,
                          [](std::int32_t token) { return token < 0; }))
    throw std::invalid_argument("native backend requires nonnegative prompt");
  GpuSamplingOptions sampling;
  try {
    sampling = sampling_options(request.sampling);
    sampling.contract = parse_sampling_contract(options_.sampling_contract);
  } catch (...) {
    std::scoped_lock lock(stats_mutex_);
    backend_stats_.fallback_reason = "unsupported_sampling";
    throw;
  }
  {
    std::scoped_lock lock(stats_mutex_);
    backend_stats_.fallback_reason.clear();
  }
  auto &owner = *ar_program_;
  const auto found = std::ranges::find(owner.occupied, false);
  if (found == owner.occupied.end())
    throw std::logic_error("P09 active slot admission exceeded");
  std::size_t slot = std::size_t(found - owner.occupied.begin());
  if (owner.prefix_cache.enabled()) {
    std::vector<std::uint32_t> prompt_u32;
    prompt_u32.reserve(request.prompt_tokens.size());
    for (auto token : request.prompt_tokens)
      prompt_u32.push_back(std::uint32_t(token));
    if (const auto match = owner.prefix_cache.find_best(
            owner.execution_identity, prompt_u32,
            owner.effective_prefill_contract(), owner.mtp_shared_kv_pages != 0U)) {
      if (match->preferred_slot < owner.occupied.size() &&
          !owner.occupied[match->preferred_slot]) {
        slot = match->preferred_slot; // Local zero-copy affinity when free.
      }
    }
  }
  owner.occupied[slot] = true;
  try {
    return std::make_unique<NativeTask>(*this, owner, slot, std::move(request),
                                        std::move(cancelled), std::move(emit),
                                        sampling);
  } catch (...) {
    owner.occupied[slot] = false;
    throw;
  }
}

bool NativeBackend::batch_enabled() const {
  return ar_program_->slot_batch_enabled;
}

std::vector<std::optional<ExecutionResult>>
NativeBackend::advance_batch(std::span<BackendTask *const> tasks) {
  assert_owner();
  auto &owner = *ar_program_;
  std::vector<std::optional<ExecutionResult>> outcomes(tasks.size());
  std::vector<std::size_t> eligible;
  std::vector<std::size_t> decode_slots;
  std::vector<std::size_t> non_batchable;

  for (std::size_t i = 0; i < tasks.size(); ++i) {
    auto *task = dynamic_cast<NativeTask *>(tasks[i]);
    if (!task)
      throw std::logic_error("P10 nonnative batch task");
    if (tasks.size() >= 2) {
      if (task->in_flight_prefill_.has_value()) {
        task->flush_in_flight_prefill(false);
      }
      if (task->in_flight_commit_.has_value()) {
        task->flush_in_flight_commit();
      }
      if (task->in_flight_ar_.has_value()) {
        const bool was_cancelled =
            task->cancelled_->load(std::memory_order_acquire);
        try {
          outcomes[i] = task->advance_contended(0, true);
          if (!was_cancelled) {
            owner.scheduler.observe_decode_round(1.0);
            std::scoped_lock lock(stats_mutex_);
            ++backend_stats_.batch_fallback_steps;
          }
          if (!outcomes[i].has_value()) {
            decode_slots.push_back(task->slot_);
          }
        } catch (const std::exception &error) {
          task->drain_in_flight_ar();
          outcomes[i] = ExecutionResult{.kind = EventKind::failed,
                                        .finish_reason = "error",
                                        .error = error.what()};
        }
        continue;
      }
    }
    if (!task->cancelled_->load(std::memory_order_acquire) &&
        task->offset_ >= task->prompt_.size()) {
      decode_slots.push_back(task->slot_);
    }
    if (task->batchable()) {
      eligible.push_back(i);
    } else {
      non_batchable.push_back(i);
    }
  }

  struct MtpPrimeSpec {
    NativeTask *task = nullptr;
    std::span<const std::uint32_t> chunk{};
    bool was_first_chunk = false;
    U32 row_off = 0U;
  };
  auto prime_mtp_tasks = [&](std::span<const MtpPrimeSpec> specs,
                             bool from_packed_primary) -> std::vector<double> {
    std::vector<double> prime_gpu_ms(specs.size(), 0.0);
    std::vector<std::size_t> active;
    active.reserve(specs.size());
    for (std::size_t k = 0; k < specs.size(); ++k) {
      if (specs[k].task && specs[k].task->mtp_active() &&
          !specs[k].chunk.empty()) {
        active.push_back(k);
      }
    }
    if (active.empty()) {
      return prime_gpu_ms;
    }
    if (active.size() == 1U) {
      const std::size_t k = active[0];
      const auto &sp = specs[k];
      const bool force_delegated = from_packed_primary && (sp.task->slot_ != 0U);
      const U32 row_off = from_packed_primary ? sp.row_off : 0U;
      const auto sub = sp.task->mtp()->prime_prefill_chunk(
          sp.chunk, sp.was_first_chunk, true, row_off, force_delegated);
      prime_gpu_ms[k] = sub.timing.gpu_ms;
      return prime_gpu_ms;
    }

    std::vector<std::size_t> prepared_idx;
    std::vector<MtpPreparedPrime> prepared;
    std::vector<std::size_t> fallback_idx;
    bool all_first_chunk = true;
    for (std::size_t k : active) {
      const auto &sp = specs[k];
      const bool force_delegated = from_packed_primary && (sp.task->slot_ != 0U);
      const U32 row_off = from_packed_primary ? sp.row_off : 0U;
      auto prep = sp.task->mtp()->prepare_prime_prefill_chunk(
          sp.chunk, sp.was_first_chunk, true, row_off, force_delegated);
      if (prep.has_value()) {
        all_first_chunk = all_first_chunk && sp.was_first_chunk;
        prepared_idx.push_back(k);
        prepared.push_back(*prep);
      } else {
        fallback_idx.push_back(k);
      }
    }
    if (!prepared.empty()) {
      if (all_first_chunk) {
        std::vector<MetalIcbProfile *> icbs;
        icbs.reserve(prepared.size() * 2U);
        for (const auto &pr : prepared) {
          if (pr.prime_icb)
            icbs.push_back(pr.prime_icb);
          if (pr.save_icb)
            icbs.push_back(pr.save_icb);
        }
        const auto sub = owner.context.replay_icb_batch(icbs);
        const double share = sub.timing.gpu_ms / double(prepared.size());
        for (std::size_t k : prepared_idx)
          prime_gpu_ms[k] = share;
      } else {
        std::vector<MetalIcbProfile *> prime_icbs;
        std::vector<MetalIcbProfile *> save_icbs;
        prime_icbs.reserve(prepared.size());
        save_icbs.reserve(prepared.size());
        for (const auto &pr : prepared) {
          if (pr.prime_icb)
            prime_icbs.push_back(pr.prime_icb);
          if (pr.save_icb)
            save_icbs.push_back(pr.save_icb);
        }
        double total_ms = 0.0;
        if (!prime_icbs.empty())
          total_ms += owner.context.replay_icb_batch(prime_icbs).timing.gpu_ms;
        if (!save_icbs.empty())
          total_ms += owner.context.replay_icb_batch(save_icbs).timing.gpu_ms;
        const double share = total_ms / double(prepared.size());
        for (std::size_t k : prepared_idx)
          prime_gpu_ms[k] = share;
      }
    }
    for (std::size_t k : fallback_idx) {
      const auto &sp = specs[k];
      const bool force_delegated = from_packed_primary && (sp.task->slot_ != 0U);
      const U32 row_off = from_packed_primary ? sp.row_off : 0U;
      const auto sub = sp.task->mtp()->prime_prefill_chunk(
          sp.chunk, sp.was_first_chunk, true, row_off, force_delegated);
      prime_gpu_ms[k] = sub.timing.gpu_ms;
    }
    return prime_gpu_ms;
  };

  owner.refresh_pressure_owner(decode_slots);

  // Packed Multi-Slot TTFT Prefill: when no decodes are active and >= 2 fresh
  // single-chunk prefills (rem <= 512, at least one > 8) fit inside the shared
  // workspace_rows_, execute them together in 1 packed CB instead of 2B - 1
  // serial prefill CBs! (Long > 512 prompts stay un-packed for fairness.)
  if (decode_slots.empty() && !owner.shared_head && tasks.size() >= 2U) {
    struct PrefillCand {
      std::size_t index = 0;
      NativeTask *task = nullptr;
      std::size_t old_offset = 0;
      std::size_t rem = 0;
    };
    std::vector<PrefillCand> cands;
    cands.reserve(tasks.size());
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      auto *task = static_cast<NativeTask *>(tasks[i]);
      if (outcomes[i].has_value() ||
          task->cancelled_->load(std::memory_order_acquire) ||
          task->in_flight_ar_.has_value() ||
          task->in_flight_prefill_.has_value() ||
          task->in_flight_commit_.has_value() ||
          task->offset_ >= task->prompt_.size()) {
        continue;
      }
      const std::size_t rem = task->prompt_.size() - task->offset_;
      if (rem == 0 || rem > 512U ||
          rem != prefill_next_chunk(task->offset_, task->prompt_.size(),
                                    owner.effective_prefill_contract()) ||
          task->will_save_sparse_anchor(task->prompt_.size()) ||
          task->open_thinking_anchor_offset() > task->offset_) {
        continue;
      }
      cands.push_back({i, task, task->offset_, rem});
    }
    std::sort(cands.begin(), cands.end(),
              [](const PrefillCand &a, const PrefillCand &b) {
                return a.task->slot_ < b.task->slot_;
              });
    const std::size_t ws_cap = owner.program->memory_plan().workspace_rows;
    std::vector<PrefillCand> prefill_group;
    std::size_t packed_pf_tokens = 0;
    bool any_rem_gt_8 = false;
    for (const auto &c : cands) {
      if (prefill_group.size() >= 8U || packed_pf_tokens + c.rem > ws_cap)
        continue;
      prefill_group.push_back(c);
      packed_pf_tokens += c.rem;
      if (c.rem > 8U)
        any_rem_gt_8 = true;
    }
    if (prefill_group.size() >= 2U && any_rem_gt_8) {
      std::vector<bool> consumed(tasks.size(), false);
      for (const auto &g : prefill_group)
        consumed[g.index] = true;
      try {
        const auto pf_begin = RoundCostRecorder::Clock::now();
        std::vector<PackedSlotSlice> slices;
        std::vector<MtpPrimeSpec> mtp_specs;
        slices.reserve(prefill_group.size());
        mtp_specs.reserve(prefill_group.size());
        U32 row_off = 0U;
        for (const auto &g : prefill_group) {
          const auto span =
              std::span(g.task->prompt_).subspan(g.old_offset, g.rem);
          slices.push_back(PackedSlotSlice{
              .slot_program = &g.task->program(),
              .slot_index = U32(g.task->slot_),
              .tokens = span,
              .sampling = g.task->sampling_,
              .verify = false,
          });
          mtp_specs.push_back(MtpPrimeSpec{
              .task = g.task,
              .chunk = span,
              .was_first_chunk = (g.old_offset == 0U),
              .row_off = row_off,
          });
          row_off += U32(g.rem);
        }

        const bool any_wants_prefix_save = std::any_of(
            prefill_group.begin(), prefill_group.end(),
            [](const PrefillCand &g) { return g.task->should_save_prefix(); });
        const bool can_fuse_sample = !any_wants_prefix_save;
        const bool need_scatter_logits =
            any_wants_prefix_save ||
            !(owner.coupled_batch_sampler &&
              owner.coupled_batch_sampler->can_sample_contiguous());
        bool fused_sampled = false;
        auto packed_out = [&] {
          for (;;) {
            try {
              return invoke_packed_batch_maybe_sample(
                  *owner.program, slices, true, can_fuse_sample,
                  need_scatter_logits, fused_sampled);
            } catch (const std::overflow_error &error) {
              if (std::string_view(error.what()) !=
                      "KV physical page pool exhausted" ||
                  !owner.prefix_cache.evict_for_page_pressure(
                      std::nullopt, [&](std::size_t s) -> QwenForward & {
                        return owner.program_at(s);
                      })) {
                throw;
              }
              std::scoped_lock lock(stats_mutex_);
              ++backend_stats_.prefix_evictions;
              backend_stats_.prefix_pinned_entries =
                  owner.prefix_cache.pinned_count();
            }
          }
        }();
        if (packed_out.slot_results.size() != prefill_group.size())
          throw std::logic_error("packed prefill result count mismatch");

        std::vector<U32> sampled_tokens(prefill_group.size(), 0U);
        double sample_gpu_ms = 0.0;
        std::uint64_t sample_read_bytes = 0U;
        std::uint64_t sample_fb_rows = 0U;
        bool sample_icb_used = false;

        if (fused_sampled) {
          for (std::size_t k = 0; k < prefill_group.size(); ++k)
            sampled_tokens[k] = packed_out.slot_results[k].token;
          sample_read_bytes = packed_sampling_host_read_bytes(packed_out);
          sample_fb_rows = packed_compact_fallback_rows(packed_out);
          sample_icb_used =
              owner.context.capabilities().profile.compute_icb_eligible;
        } else if (owner.coupled_batch_sampler) {
          std::vector<PositionSamplingRow> keys;
          std::vector<BatchRow> rows;
          keys.reserve(prefill_group.size());
          rows.reserve(prefill_group.size());
          for (const auto &g : prefill_group) {
            const U32 valid =
                g.task->cancelled_->load(std::memory_order_acquire) ? 0U : 1U;
            rows.push_back({U32(g.task->slot_), 1U, valid});
            keys.push_back(PositionSamplingRow{
                .seed = g.task->sampling_.seed,
                .position = std::uint64_t(g.task->program().position()),
                .top_k = g.task->sampling_.top_k,
                .temperature = g.task->sampling_.temperature,
                .top_p = g.task->sampling_.top_p,
                .valid = valid,
            });
          }
          auto selected =
              owner.coupled_batch_sampler->can_sample_contiguous()
                  ? owner.coupled_batch_sampler->sample_contiguous(keys, true)
                  : owner.coupled_batch_sampler->sample(rows, keys);
          sampled_tokens = std::move(selected.tokens);
          sample_gpu_ms = selected.submission.timing.gpu_ms;
          sample_read_bytes = selected.host_read_bytes;
          sample_fb_rows = selected.compact_fallback_rows;
          sample_icb_used = owner.coupled_batch_sampler->uses_icb();
        } else {
          BatchSampleParams params{};
          params.context_envelope = U32(options_.max_context);
          params.count = U32(prefill_group.size());
          for (std::size_t k = 0; k < prefill_group.size(); ++k) {
            const auto &g = prefill_group[k];
            params.valid[k] = 1U;
            params.slot[k] = U32(g.task->slot_);
            params.width[k] = 1U;
            params.sampling[k] = {
                g.task->program().vocabulary_size(),
                g.task->sampling_.top_k,
                g.task->sampling_.temperature,
                g.task->sampling_.top_p,
                g.task->sampling_.seed,
                g.task->sampling_.stream,
                g.task->program().position() -
                    (owner.sampling_contract ==
                             SamplingContract::PositionCoupledV1
                         ? 0U
                         : 1U)};
          }
          auto selected = owner.batch_sampler->sample(params);
          sampled_tokens = std::move(selected.first);
          sample_gpu_ms = selected.second.timing.gpu_ms;
          sample_icb_used = owner.batch_sampler->uses_icb();
        }

        // Prime any MTP slots directly from primary.hidden while untouched!
        const auto mtp_prime_ms = prime_mtp_tasks(mtp_specs, true);
        const double pf_wall_share = RoundCostRecorder::milliseconds(pf_begin) /
                                     double(prefill_group.size());
        const double fwd_gpu_share =
            (packed_out.submission.timing.gpu_ms + sample_gpu_ms) /
            double(prefill_group.size());

        std::uint64_t slot_read_sum = 0U;
        for (std::size_t k = 0; k < prefill_group.size(); ++k) {
          auto &g = prefill_group[k];
          g.task->step_ = std::move(packed_out.slot_results[k]);
          g.task->step_.token = sampled_tokens[k];
          g.task->offset_ += g.rem;
          const auto span =
              std::span(g.task->prompt_).subspan(g.old_offset, g.rem);
          g.task->committed_tokens_.insert(g.task->committed_tokens_.end(),
                                           span.begin(), span.end());
          g.task->save_prefix();
          slot_read_sum += g.task->step_.host_read_bytes;
          const double pf_ms =
              fwd_gpu_share > 0.0 ? (fwd_gpu_share + mtp_prime_ms[k])
                                  : pf_wall_share;
          owner.scheduler.observe_prefill_chunk(g.rem, pf_ms, false);
        }

        {
          std::scoped_lock lock(stats_mutex_);
          auto &stats = backend_stats_;
          ++stats.packed_batch_calls;
          stats.packed_batch_tokens += packed_pf_tokens;
          ++stats.batch_submissions;
          stats.batch_rows += prefill_group.size();
          if (prefill_group.size() < stats.batch_size_counts.size())
            ++stats.batch_size_counts[prefill_group.size()];
          stats.batch_sampling_host_read_bytes += sample_read_bytes;
          stats.batch_compact_fallback_rows += sample_fb_rows;
          if (!fused_sampled || (slot_read_sum == 0U && sample_read_bytes > 0U))
            stats.ar_host_read_bytes += sample_read_bytes;
          stats.actual_b = U32(prefill_group.size());
          stats.actual_w = 1U;
          for (std::size_t k = 0; k < prefill_group.size(); ++k) {
            const auto &g = prefill_group[k];
            ++stats.ar_steps;
            ++stats.batched_ar_steps;
            stats.actual_t = stats.actual_m = g.task->step_.tokens;
            stats.actual_k = 0;
            stats.ar_layer_calls += g.task->step_.executed_layers;
            stats.short_matrix_calls += g.task->step_.short_matrix_calls;
            stats.short_lane_calls += g.task->step_.short_lane_calls;
            stats.ar_host_read_bytes += g.task->step_.host_read_bytes;
            if (g.rem < stats.batch_width_counts.size())
              ++stats.batch_width_counts[g.rem];
            stats.actual_w = std::max(stats.actual_w, U32(g.rem));
            if (fwd_gpu_share > 0.0) {
              ++stats.ar_gpu_timed_steps;
              stats.ar_prefill_gpu_ms += fwd_gpu_share + mtp_prime_ms[k];
            }
          }
          stats.batch_gpu_ms +=
              packed_out.submission.timing.gpu_ms + sample_gpu_ms;
          if (sample_icb_used)
            ++stats.batch_sampler_icb_replays;
        }
      } catch (const std::exception &error) {
        for (const auto &g : prefill_group) {
          outcomes[g.index] = ExecutionResult{.kind = EventKind::failed,
                                              .finish_reason = "error",
                                              .error = error.what()};
        }
      }
      std::erase_if(eligible, [&](std::size_t idx) { return consumed[idx]; });
      std::erase_if(non_batchable,
                    [&](std::size_t idx) { return consumed[idx]; });
      if (eligible.empty() && non_batchable.empty()) {
        return outcomes;
      }
    }
  }

  const bool has_active_decodes = !decode_slots.empty();

  // Case 1: Fewer than 2 batchable tasks -> advance ready decode/short tasks
  // FIRST so concurrent decodes never block behind a 1024-token prefill!
  if (eligible.size() < 2) {
    for (std::size_t i : eligible) {
      auto *task = static_cast<NativeTask *>(tasks[i]);
      const bool was_decode = task->offset_ >= task->prompt_.size();
      try {
        outcomes[i] = task->advance();
        if (was_decode)
          owner.scheduler.observe_decode_round(1.0);
        std::scoped_lock lock(stats_mutex_);
        ++backend_stats_.batch_fallback_steps;
      } catch (const std::exception &error) {
        outcomes[i] = ExecutionResult{.kind = EventKind::failed,
                                      .finish_reason = "error",
                                      .error = error.what()};
      }
    }
    const bool ext_contended =
        owner.external_gpu_contended.load(std::memory_order_acquire);
    const bool prefill_contended = has_active_decodes || ext_contended;
    const std::size_t pf_cap =
        prefill_contended ? owner.scheduler.calibrated_contended_budget() : 0;
    for (bool decode_pass : {true, false}) {
      for (std::size_t i : non_batchable) {
        auto *task = static_cast<NativeTask *>(tasks[i]);
        const bool cancelled = task->cancelled_->load(std::memory_order_acquire);
        const bool was_decode =
            !cancelled && task->offset_ >= task->prompt_.size();
        if (was_decode != decode_pass)
          continue;
        if (has_active_decodes && !was_decode && !cancelled &&
            owner.scheduler.should_defer_prefill(task->slot_)) {
          continue;
        }
        try {
          outcomes[i] = task->advance_contended(pf_cap, prefill_contended);
          if (was_decode)
            owner.scheduler.observe_decode_round(1.0);
          std::scoped_lock lock(stats_mutex_);
          ++backend_stats_.batch_fallback_steps;
        } catch (const std::exception &error) {
          outcomes[i] = ExecutionResult{.kind = EventKind::failed,
                                        .finish_reason = "error",
                                        .error = error.what()};
        }
      }
    }
    return outcomes;
  }

  // Case 2: >= 2 batchable tasks -> run the cross-slot packed batch FIRST,
  // then service non-batchable prefills under the #5164 decode-protect budget!
  std::vector<SpecSlotQuery> spec_queries;
  std::vector<std::size_t> spec_eligible_pos;
  for (std::size_t pos = 0; pos < eligible.size(); ++pos) {
    auto *task = static_cast<NativeTask *>(tasks[eligible[pos]]);
    if (task->is_spec_verify_ready()) {
      try {
        spec_queries.push_back(task->make_spec_query());
        spec_eligible_pos.push_back(pos);
      } catch (const std::exception &error) {
        outcomes[eligible[pos]] = ExecutionResult{.kind = EventKind::failed,
                                                  .finish_reason = "error",
                                                  .error = error.what()};
      }
    }
  }
  const auto spec_decisions =
      spec_queries.empty()
          ? std::vector<SpecSlotDecision>{}
          : owner.spec_controller.select_batch_depths(spec_queries);

  struct DeferredRow {
    std::size_t index = 0;
    NativeTask *task = nullptr;
    bool wants_dflash_draft = false;
    bool wants_mtp_draft = false;
  };
  std::vector<DeferredRow> staged;
  staged.reserve(eligible.size());
  std::size_t spec_cursor = 0;
  for (std::size_t pos = 0; pos < eligible.size(); ++pos) {
    const std::size_t i = eligible[pos];
    if (outcomes[i])
      continue;
    auto *task = static_cast<NativeTask *>(tasks[i]);
    std::optional<SpecSlotDecision> assigned;
    if (spec_cursor < spec_eligible_pos.size() &&
        spec_eligible_pos[spec_cursor] == pos) {
      assigned = spec_decisions[spec_cursor++];
    }
    try {
      bool wants_df = false;
      bool wants_mtp = false;
      outcomes[i] = task->prepare_batch_row(&wants_df, &wants_mtp, assigned,
                                            std::uint32_t(eligible.size()));
      if (outcomes[i])
        continue;
      staged.push_back({i, task, wants_df, wants_mtp});
    } catch (const std::exception &error) {
      outcomes[i] = ExecutionResult{.kind = EventKind::failed,
                                    .finish_reason = "error",
                                    .error = error.what()};
    }
  }

  // Canonical slot-ascending order guarantees slot 0 always starts at row_off 0
  // (0-copy in-place trunk hidden) and avoids B! packed ICB cache permutations.
  std::sort(staged.begin(), staged.end(),
            [](const DeferredRow &a, const DeferredRow &b) {
              return a.task->slot_ < b.task->slot_;
            });

  // 1. Batched DFlash2 draft proposal: packed [8*B, H] single-CB drafter +
  // shared target lm_head when >= 2 DFlash slots draft together!
  {
    std::vector<std::size_t> df_idxs;
    for (std::size_t s = 0; s < staged.size(); ++s) {
      if (staged[s].wants_dflash_draft)
        df_idxs.push_back(s);
    }
    const bool cpu_sel =
        owner.context.capabilities().profile.probe.apple_gpu_family < 9;
    const U32 df_ctx_cap = std::min<U32>(U32(options_.max_context), 1024U);
    if (df_idxs.size() >= 2U && owner.draft != nullptr &&
        8U * df_idxs.size() <= df_ctx_cap) {
      try {
        std::vector<DFlashPackedSlotSlice> df_slices;
        df_slices.reserve(df_idxs.size());
        for (std::size_t s : df_idxs) {
          auto *task = staged[s].task;
          const bool tokens_only =
              owner.sampling_contract == SamplingContract::PositionCoupledV1 &&
              task->verifier() != nullptr;
          df_slices.push_back(DFlashPackedSlotSlice{
              .slot_draft = task->draft(),
              .anchor = task->anchor_,
              .context_tokens = task->program().position(),
              .draft_tokens = task->pending_k_,
              .temperature = task->request_.sampling.temperature,
              .seed = task->request_.sampling.seed,
              .tokens_only = tokens_only,
          });
        }
        const auto draft_begin = RoundCostRecorder::Clock::now();
        auto packed_df =
            owner.draft->propose_packed_batch(df_slices, true, cpu_sel);
        const double draft_wall = RoundCostRecorder::milliseconds(draft_begin) /
                                  double(df_idxs.size());
        if (packed_df.slot_results.size() != df_idxs.size())
          throw std::logic_error("packed DFlash draft result count mismatch");
        for (std::size_t k = 0; k < df_idxs.size(); ++k) {
          auto *task = staged[df_idxs[k]].task;
          if (task->pending_cost_)
            task->pending_cost_->add_draft(draft_wall);
          task->apply_draft_result(std::move(packed_df.slot_results[k]));
        }
      } catch (const std::exception &error) {
        for (std::size_t s : df_idxs) {
          outcomes[staged[s].index] =
              ExecutionResult{.kind = EventKind::failed,
                              .finish_reason = "error",
                              .error = error.what()};
        }
      }
    } else if (df_idxs.size() >= 2U) {
      std::vector<std::optional<DFlashPreparedProposal>> preps(df_idxs.size());
      std::vector<MetalIcbProfile *> draft_icbs;
      std::vector<std::size_t> draft_icb_pos;
      for (std::size_t k = 0; k < df_idxs.size(); ++k) {
        auto *task = staged[df_idxs[k]].task;
        const bool tokens_only =
            owner.sampling_contract == SamplingContract::PositionCoupledV1 &&
            task->verifier() != nullptr;
        try {
          preps[k] = task->pending_cost_->measure(RoundCostRecorder::draft, [&] {
            return invoke_draft_prepare_propose(
                *task->draft(), task->anchor_, task->program().position(),
                task->pending_k_, task->request_.sampling.temperature,
                task->request_.sampling.seed, true, false, cpu_sel,
                tokens_only);
          });
          if (preps[k]->icb) {
            draft_icbs.push_back(preps[k]->icb);
            draft_icb_pos.push_back(k);
          }
        } catch (const std::exception &error) {
          outcomes[staged[df_idxs[k]].index] =
              ExecutionResult{.kind = EventKind::failed,
                              .finish_reason = "error",
                              .error = error.what()};
        }
      }
      MetalSubmission batched_draft_sub;
      bool did_batch_draft = false;
      if (draft_icbs.size() >= 2U) {
        const auto draft_begin = RoundCostRecorder::Clock::now();
        batched_draft_sub = owner.context.replay_icb_batch(draft_icbs);
        const double draft_wall = RoundCostRecorder::milliseconds(draft_begin) /
                                  double(draft_icbs.size());
        for (std::size_t k : draft_icb_pos) {
          auto *task = staged[df_idxs[k]].task;
          if (task->pending_cost_)
            task->pending_cost_->add_draft(draft_wall);
        }
        did_batch_draft = true;
      }
      for (std::size_t k = 0; k < df_idxs.size(); ++k) {
        if (!preps[k])
          continue;
        try {
          const bool was_batched = did_batch_draft && (preps[k]->icb != nullptr);
          MetalSubmission per_sub = batched_draft_sub;
          if (was_batched) {
            per_sub.timing.gpu_ms /= double(draft_icbs.size());
            per_sub.dispatch_count /= draft_icbs.size();
          }
          staged[df_idxs[k]].task->finish_deferred_draft(
              std::move(*preps[k]), was_batched, per_sub);
        } catch (const std::exception &error) {
          outcomes[staged[df_idxs[k]].index] =
              ExecutionResult{.kind = EventKind::failed,
                              .finish_reason = "error",
                              .error = error.what()};
        }
      }
    } else if (df_idxs.size() == 1U) {
      auto *task = staged[df_idxs[0]].task;
      const bool tokens_only =
          owner.sampling_contract == SamplingContract::PositionCoupledV1 &&
          task->verifier() != nullptr;
      try {
        auto res = task->pending_cost_->measure(RoundCostRecorder::draft, [&] {
          return invoke_draft_propose(
              *task->draft(), task->anchor_, task->program().position(),
              task->pending_k_, task->request_.sampling.temperature,
              task->request_.sampling.seed, true, false, cpu_sel, tokens_only);
        });
        task->apply_draft_result(std::move(res));
      } catch (const std::exception &error) {
        outcomes[staged[df_idxs[0]].index] =
            ExecutionResult{.kind = EventKind::failed,
                            .finish_reason = "error",
                            .error = error.what()};
      }
    }
  }

  // 2. Batched Native MTP draft proposal: packed multi-slot MTP Layer 0 +
  // shared [B, 248320] lm_head in 1 CB across all B slots!
  {
    std::vector<std::size_t> mtp_idxs;
    for (std::size_t s = 0; s < staged.size(); ++s) {
      if (staged[s].wants_mtp_draft && !outcomes[staged[s].index])
        mtp_idxs.push_back(s);
    }
    if (mtp_idxs.size() >= 2U && owner.mtp != nullptr &&
        owner.mtp->can_propose_packed_batch(U32(mtp_idxs.size()))) {
      try {
        std::vector<MtpPackedSlotSlice> mtp_slices;
        mtp_slices.reserve(mtp_idxs.size());
        for (std::size_t s : mtp_idxs) {
          auto *task = staged[s].task;
          const bool tokens_only =
              owner.sampling_contract == SamplingContract::PositionCoupledV1 &&
              task->verifier() != nullptr;
          mtp_slices.push_back(MtpPackedSlotSlice{
              .slot_mtp = task->mtp(),
              .committed_tokens = task->mtp_committed_tokens_,
              .target_position = task->program().position(),
              .draft_tokens = task->pending_k_,
              .sampling = task->sampling_,
              .tokens_only = tokens_only,
          });
        }
        const auto draft_begin = RoundCostRecorder::Clock::now();
        auto packed_mtp = owner.mtp->propose_packed_batch(mtp_slices, true);
        const double draft_wall = RoundCostRecorder::milliseconds(draft_begin) /
                                  double(mtp_idxs.size());
        if (packed_mtp.slot_results.size() != mtp_idxs.size())
          throw std::logic_error("packed MTP draft result count mismatch");
        for (std::size_t k = 0; k < mtp_idxs.size(); ++k) {
          auto *task = staged[mtp_idxs[k]].task;
          if (task->pending_cost_)
            task->pending_cost_->add_draft(draft_wall);
          task->apply_draft_result(std::move(packed_mtp.slot_results[k]));
        }
      } catch (const std::exception &error) {
        for (std::size_t s : mtp_idxs) {
          outcomes[staged[s].index] =
              ExecutionResult{.kind = EventKind::failed,
                              .finish_reason = "error",
                              .error = error.what()};
        }
      }
    } else if (mtp_idxs.size() >= 2U) {
      std::vector<std::optional<MtpPreparedProposal>> preps(mtp_idxs.size());
      std::vector<MetalIcbProfile *> mtp_icbs;
      std::vector<std::size_t> mtp_icb_pos;
      for (std::size_t k = 0; k < mtp_idxs.size(); ++k) {
        auto *task = staged[mtp_idxs[k]].task;
        const bool tokens_only =
            owner.sampling_contract == SamplingContract::PositionCoupledV1 &&
            task->verifier() != nullptr;
        try {
          preps[k] = task->pending_cost_->measure(RoundCostRecorder::draft, [&] {
            return task->mtp()->prepare_propose(
                task->mtp_committed_tokens_, task->program().position(),
                task->pending_k_, task->sampling_, true, tokens_only);
          });
          if (preps[k]->icb) {
            mtp_icbs.push_back(preps[k]->icb);
            mtp_icb_pos.push_back(k);
          }
        } catch (const std::exception &error) {
          outcomes[staged[mtp_idxs[k]].index] =
              ExecutionResult{.kind = EventKind::failed,
                              .finish_reason = "error",
                              .error = error.what()};
        }
      }
      MetalSubmission batched_mtp_sub;
      bool did_batch_mtp = false;
      if (mtp_icbs.size() >= 2U) {
        const auto draft_begin = RoundCostRecorder::Clock::now();
        batched_mtp_sub = owner.context.replay_icb_batch(mtp_icbs);
        const double draft_wall = RoundCostRecorder::milliseconds(draft_begin) /
                                  double(mtp_icbs.size());
        for (std::size_t k : mtp_icb_pos) {
          auto *task = staged[mtp_idxs[k]].task;
          if (task->pending_cost_)
            task->pending_cost_->add_draft(draft_wall);
        }
        did_batch_mtp = true;
      }
      for (std::size_t k = 0; k < mtp_idxs.size(); ++k) {
        if (!preps[k])
          continue;
        try {
          const bool was_batched = did_batch_mtp && (preps[k]->icb != nullptr);
          MetalSubmission per_sub = batched_mtp_sub;
          if (was_batched) {
            per_sub.timing.gpu_ms /= double(mtp_icbs.size());
            per_sub.dispatch_count /= mtp_icbs.size();
          }
          staged[mtp_idxs[k]].task->finish_deferred_mtp_draft(
              std::move(*preps[k]), was_batched, per_sub);
        } catch (const std::exception &error) {
          outcomes[staged[mtp_idxs[k]].index] =
              ExecutionResult{.kind = EventKind::failed,
                              .finish_reason = "error",
                              .error = error.what()};
        }
      }
    } else if (mtp_idxs.size() == 1U) {
      auto *task = staged[mtp_idxs[0]].task;
      const bool tokens_only =
          owner.sampling_contract == SamplingContract::PositionCoupledV1 &&
          task->verifier() != nullptr;
      try {
        auto res = task->pending_cost_->measure(RoundCostRecorder::draft, [&] {
          return task->mtp()->propose(task->mtp_committed_tokens_,
                                      task->program().position(),
                                      task->pending_k_, task->sampling_, true,
                                      tokens_only);
        });
        task->apply_draft_result(std::move(res));
      } catch (const std::exception &error) {
        outcomes[staged[mtp_idxs[0]].index] =
            ExecutionResult{.kind = EventKind::failed,
                            .finish_reason = "error",
                            .error = error.what()};
      }
    }
  }

  struct PendingItem {
    std::size_t index = 0;
    NativeTask *task = nullptr;
    std::optional<QwenPreparedStep> prepared;
    QwenForwardResult packed_result{};
    U32 row_off = 0U;
  };
  std::vector<PendingItem> pending;
  pending.reserve(staged.size());
  bool all_verify = true;
  bool none_verify = true;
  bool any_cancelled = false;
  std::uint32_t total_batch_tokens = 0;
  for (auto &row : staged) {
    if (outcomes[row.index])
      continue;
    if (row.task->pending_verify_)
      none_verify = false;
    else
      all_verify = false;
    if (row.task->cancelled_->load(std::memory_order_acquire))
      any_cancelled = true;
    pending.push_back(
        {row.index, row.task, std::nullopt, {}, total_batch_tokens});
    total_batch_tokens += std::uint32_t(row.task->pending_input_.size());
  }

  if (!pending.empty()) {
    try {
      const bool use_packed_batch =
          pending.size() >= 2 && !owner.shared_head &&
          (all_verify || none_verify);
      MetalSubmission forward;
      MetalSubmission projection;
      bool fused_sampled = false;
      std::uint64_t fused_sample_read_bytes = 0U;
      std::uint64_t fused_compact_fallback_rows = 0U;
      const auto shared_verify_begin = RoundCostRecorder::Clock::now();

      if (use_packed_batch) {
        std::vector<PackedSlotSlice> slices;
        slices.reserve(pending.size());
        for (auto &item : pending) {
          slices.push_back(PackedSlotSlice{
              .slot_program = &item.task->program(),
              .slot_index = U32(item.task->slot_),
              .tokens = item.task->pending_input_,
              .sampling = item.task->sampling_,
              .verify = item.task->pending_verify_,
          });
        }
        const bool any_prefill_wants_save = std::any_of(
            pending.begin(), pending.end(), [](const PendingItem &item) {
              return item.task->pending_prefill_ &&
                     item.task->should_save_prefix();
            });
        const bool can_fuse_sample =
            none_verify && !any_cancelled && !any_prefill_wants_save;
        const bool scatter_slot_logits =
            !none_verify || any_prefill_wants_save ||
            !(owner.coupled_batch_sampler &&
              owner.coupled_batch_sampler->can_sample_contiguous());
        auto packed_out = [&] {
          for (;;) {
            try {
              return invoke_packed_batch_maybe_sample(
                  *owner.program, slices, true, can_fuse_sample,
                  scatter_slot_logits, fused_sampled);
            } catch (const std::overflow_error &error) {
              if (std::string_view(error.what()) !=
                      "KV physical page pool exhausted" ||
                  !owner.prefix_cache.evict_for_page_pressure(
                      std::nullopt, [&](std::size_t s) -> QwenForward & {
                        return owner.program_at(s);
                      })) {
                throw;
              }
              std::scoped_lock lock(stats_mutex_);
              ++backend_stats_.prefix_evictions;
              backend_stats_.prefix_pinned_entries =
                  owner.prefix_cache.pinned_count();
            }
          }
        }();
        forward = std::move(packed_out.submission);
        if (packed_out.slot_results.size() != pending.size())
          throw std::logic_error("packed batch result count mismatch");
        if (fused_sampled) {
          fused_sample_read_bytes = packed_sampling_host_read_bytes(packed_out);
          fused_compact_fallback_rows = packed_compact_fallback_rows(packed_out);
        }
        for (std::size_t p_i = 0; p_i < pending.size(); ++p_i)
          pending[p_i].packed_result = std::move(packed_out.slot_results[p_i]);
        {
          std::scoped_lock lock(stats_mutex_);
          ++backend_stats_.packed_batch_calls;
          backend_stats_.packed_batch_tokens += total_batch_tokens;
        }
      } else {
        std::vector<MetalCommandInvocation> calls;
        std::vector<MetalIcbProfile *> forward_icbs;
        bool all_forward_icb = true;
        std::vector<PendingItem> active_pending;
        active_pending.reserve(pending.size());
        for (auto &item : pending) {
          try {
            auto prep = item.task->prepare_gpu_batch();
            calls.push_back(prep.invocation());
            if (auto *icb = prep.icb_profile())
              forward_icbs.push_back(icb);
            else
              all_forward_icb = false;
            item.prepared.emplace(std::move(prep));
            active_pending.push_back(std::move(item));
          } catch (const std::exception &error) {
            outcomes[item.index] =
                ExecutionResult{.kind = EventKind::failed,
                                .finish_reason = "error",
                                .error = error.what()};
          }
        }
        pending = std::move(active_pending);
        if (!pending.empty()) {
          forward = (all_forward_icb && forward_icbs.size() == pending.size())
                        ? owner.context.replay_icb_batch(forward_icbs)
                        : owner.context.execute_batch(calls);
          if (owner.shared_head) {
            std::vector<BatchRow> projection_rows;
            projection_rows.reserve(pending.size());
            for (const auto &item : pending) {
              projection_rows.push_back(
                  {U32(item.task->slot_),
                   U32(item.task->pending_input_.size()),
                   item.task->cancelled_->load(std::memory_order_acquire)
                       ? 0U
                       : 1U});
            }
            projection = owner.shared_head->project(projection_rows);
          }
        }
      }

      if (!pending.empty()) {
        // Divide shared verify wall & GPU ms across active batch rows.
        const double shared_verify_wall =
            RoundCostRecorder::milliseconds(shared_verify_begin) /
            double(pending.size());
        const double verify_gpu_share_ms =
            (forward.timing.gpu_ms + projection.timing.gpu_ms) /
            double(pending.size());
        for (auto &item : pending) {
          if (item.task->pending_cost_)
            item.task->pending_cost_->add_verify(shared_verify_wall);
        }

        BatchSampleParams params{};
        params.context_envelope = std::uint32_t(options_.max_context);
        std::vector<std::size_t> ar_rows;
        std::vector<MtpPrimeSpec> ar_mtp_primes;
        std::uint64_t ar_slot_read_sum = 0U;
        for (auto &item : pending) {
          if (item.task->pending_verify_)
            continue;
          if (use_packed_batch) {
            item.task->complete_batch_ar_result(std::move(item.packed_result));
          } else {
            item.task->complete_batch_ar(std::move(*item.prepared), 0);
          }
          ar_slot_read_sum += item.task->step_.host_read_bytes;
          if (item.task->pending_prefill_ && item.task->mtp_active()) {
            ar_mtp_primes.push_back(MtpPrimeSpec{
                .task = item.task,
                .chunk = item.task->pending_input_,
                .was_first_chunk = item.task->pending_was_first_chunk_,
                .row_off = item.row_off,
            });
          }
          const auto row = ar_rows.size();
          params.valid[row] = 1;
          params.slot[row] = std::uint32_t(item.task->slot_);
          params.width[row] = item.task->step_.tokens;
          params.sampling[row] = {
              item.task->program().vocabulary_size(),
              item.task->sampling_.top_k,
              item.task->sampling_.temperature,
              item.task->sampling_.top_p,
              item.task->sampling_.seed,
              item.task->sampling_.stream,
              item.task->program().position() -
                  (owner.sampling_contract ==
                           SamplingContract::PositionCoupledV1
                       ? 0
                       : 1)};
          ar_rows.push_back(item.index);
        }
        double sample_gpu_ms = 0;
        if (!ar_rows.empty()) {
          if (fused_sampled) {
            std::scoped_lock lock(stats_mutex_);
            backend_stats_.batch_sampling_host_read_bytes +=
                fused_sample_read_bytes;
            backend_stats_.batch_compact_fallback_rows +=
                fused_compact_fallback_rows;
            if (ar_slot_read_sum == 0U && fused_sample_read_bytes > 0U)
              backend_stats_.ar_host_read_bytes += fused_sample_read_bytes;
          } else {
            params.count = std::uint32_t(ar_rows.size());
            std::vector<U32> tokens;
            MetalSubmission sample;
            if (owner.coupled_batch_sampler) {
              std::vector<BatchRow> rows;
              std::vector<PositionSamplingRow> keys;
              rows.reserve(ar_rows.size());
              keys.reserve(ar_rows.size());
              for (U32 row = 0; row < ar_rows.size(); ++row) {
                auto *task = static_cast<NativeTask *>(tasks[ar_rows[row]]);
                const U32 valid =
                    task->cancelled_->load(std::memory_order_acquire) ? 0U : 1U;
                rows.push_back({params.slot[row], params.width[row], valid});
                const auto &key = params.sampling[row];
                keys.push_back({.seed = key.seed,
                                .position = key.absolute_position,
                                .top_k = key.top_k,
                                .temperature = key.temperature,
                                .top_p = key.top_p,
                                .valid = valid});
              }
              auto selected =
                  (use_packed_batch && none_verify &&
                   owner.coupled_batch_sampler->can_sample_contiguous())
                      ? owner.coupled_batch_sampler->sample_contiguous(keys,
                                                                       true)
                      : owner.coupled_batch_sampler->sample(rows, keys);
              tokens = std::move(selected.tokens);
              sample = std::move(selected.submission);
              std::scoped_lock lock(stats_mutex_);
              backend_stats_.batch_sampling_host_read_bytes +=
                  selected.host_read_bytes;
              backend_stats_.batch_compact_fallback_rows +=
                  selected.compact_fallback_rows;
              backend_stats_.ar_host_read_bytes += selected.host_read_bytes;
            } else {
              auto selected = owner.batch_sampler->sample(params);
              tokens = std::move(selected.first);
              sample = std::move(selected.second);
            }
            sample_gpu_ms = sample.timing.gpu_ms;
            for (std::size_t row = 0; row < ar_rows.size(); ++row) {
              auto *task = static_cast<NativeTask *>(tasks[ar_rows[row]]);
              task->step_.token = tokens[row];
            }
          }
          if (!ar_mtp_primes.empty()) {
            (void)prime_mtp_tasks(ar_mtp_primes, use_packed_batch);
          }
          for (std::size_t row = 0; row < ar_rows.size(); ++row) {
            auto *task = static_cast<NativeTask *>(tasks[ar_rows[row]]);
            if (task->pending_prefill_)
              task->save_prefix();
          }
        }

        // Single-CB Multi-Slot TargetGpuVerifier (Phase A: prepare, Phase B:
        // 1 replay_icb_batch across all B slots, Phase C: finish + prepare commit).
        struct StagedSelect {
          std::size_t pending_idx = 0;
          std::optional<TargetGpuPreparedSelect> sel_prep;
        };
        std::vector<StagedSelect> staged_selects;
        std::vector<MetalIcbProfile *> select_icbs;
        for (std::size_t p_idx = 0; p_idx < pending.size(); ++p_idx) {
          auto &item = pending[p_idx];
          if (!item.task->pending_verify_)
            continue;
          try {
            QwenForwardResult vres =
                use_packed_batch
                    ? std::move(item.packed_result)
                    : item.task->program().finish_batch(
                          std::move(*item.prepared));
            outcomes[item.index] = item.task->record_batch_verify_result(
                std::move(vres), verify_gpu_share_ms);
            if (outcomes[item.index])
              continue;
            std::optional<TargetGpuPreparedSelect> sp;
            if (auto *ver = item.task->verifier()) {
              sp = ver->prepare_select(
                  item.task->pending_proposal_.selection.steps,
                  item.task->sampling_, item.task->program().position(), true);
              if (sp->icb)
                select_icbs.push_back(sp->icb);
            }
            staged_selects.push_back({p_idx, std::move(sp)});
          } catch (const std::exception &error) {
            try {
              item.task->program().abort_verify();
            } catch (...) {
            }
            outcomes[item.index] =
                ExecutionResult{.kind = EventKind::failed,
                                .finish_reason = "error",
                                .error = error.what()};
          }
        }

        MetalSubmission batched_select_sub;
        const bool did_batch_select = select_icbs.size() >= 2U;
        if (did_batch_select) {
          try {
            const auto sel_begin = RoundCostRecorder::Clock::now();
            batched_select_sub = owner.context.replay_icb_batch(select_icbs);
            const double sel_wall = RoundCostRecorder::milliseconds(sel_begin) /
                                    double(select_icbs.size());
            for (auto &ss : staged_selects) {
              auto &item = pending[ss.pending_idx];
              if (ss.sel_prep && ss.sel_prep->icb && item.task->pending_cost_)
                item.task->pending_cost_->add_select(sel_wall);
            }
          } catch (const std::exception &error) {
            for (auto &ss : staged_selects) {
              auto &item = pending[ss.pending_idx];
              try {
                item.task->program().abort_verify();
              } catch (...) {
              }
              outcomes[item.index] =
                  ExecutionResult{.kind = EventKind::failed,
                                  .finish_reason = "error",
                                  .error = error.what()};
            }
            staged_selects.clear();
          }
        }

        struct StagedCommit {
          std::size_t pending_idx;
          NativeTask::PreparedCommit pc;
        };
        std::vector<StagedCommit> staged_commits;
        std::vector<MetalIcbProfile *> commit_icbs;
        for (auto &ss : staged_selects) {
          auto &item = pending[ss.pending_idx];
          try {
            const bool sel_batched =
                did_batch_select && ss.sel_prep && (ss.sel_prep->icb != nullptr);
            MetalSubmission per_sel_sub = batched_select_sub;
            if (sel_batched) {
              per_sel_sub.timing.gpu_ms /= double(select_icbs.size());
              per_sel_sub.timing.wall_ms /= double(select_icbs.size());
              per_sel_sub.dispatch_count /= select_icbs.size();
            }
            NativeTask::PreparedCommit pc;
            outcomes[item.index] =
                item.task->finish_batch_select_and_prepare_commit(
                    std::move(ss.sel_prep), sel_batched, std::move(per_sel_sub),
                    verify_gpu_share_ms, pc);
            if (outcomes[item.index])
              continue;
            if (pc.icb)
              commit_icbs.push_back(pc.icb);
            staged_commits.push_back({ss.pending_idx, std::move(pc)});
          } catch (const std::exception &error) {
            outcomes[item.index] =
                ExecutionResult{.kind = EventKind::failed,
                                .finish_reason = "error",
                                .error = error.what()};
          }
        }

        MetalSubmission batched_commit_sub;
        const bool did_batch_commit = commit_icbs.size() >= 2;
        if (did_batch_commit) {
          try {
            const auto commit_begin = RoundCostRecorder::Clock::now();
            batched_commit_sub = owner.context.replay_icb_batch(commit_icbs);
            const double commit_wall =
                RoundCostRecorder::milliseconds(commit_begin) /
                double(commit_icbs.size());
            for (auto &sc : staged_commits) {
              auto &item = pending[sc.pending_idx];
              if (sc.pc.icb && item.task->pending_cost_)
                item.task->pending_cost_->add_replay(commit_wall);
            }
          } catch (const std::exception &error) {
            for (auto &sc : staged_commits) {
              auto &item = pending[sc.pending_idx];
              try {
                item.task->program().abort_verify();
              } catch (...) {
              }
              outcomes[item.index] =
                  ExecutionResult{.kind = EventKind::failed,
                                  .finish_reason = "error",
                                  .error = error.what()};
            }
            staged_commits.clear();
          }
        }
        std::size_t total_committed_spec = 0;
        for (auto &sc : staged_commits) {
          auto &item = pending[sc.pending_idx];
          total_committed_spec += sc.pc.count;
          try {
            const bool was_batched =
                did_batch_commit && (sc.pc.icb != nullptr);
            const double share_ms =
                was_batched ? (batched_commit_sub.timing.gpu_ms /
                               double(commit_icbs.size()))
                            : 0.0;
            outcomes[item.index] = item.task->finish_batch_verify_commit(
                std::move(sc.pc), was_batched, share_ms, total_batch_tokens,
                std::uint32_t(pending.size()));
          } catch (const std::exception &error) {
            outcomes[item.index] =
                ExecutionResult{.kind = EventKind::failed,
                                .finish_reason = "error",
                                .error = error.what()};
          }
        }

        std::size_t decode_streams_in_batch = 0;
        std::size_t decode_ar_tokens = 0;
        for (const auto &item : pending) {
          if (!item.task->pending_prefill_) {
            ++decode_streams_in_batch;
            if (!item.task->pending_verify_)
              ++decode_ar_tokens;
          }
        }
        if (decode_streams_in_batch > 0) {
          const double avg_tok_per_stream =
              double(decode_ar_tokens + total_committed_spec) /
              double(decode_streams_in_batch);
          owner.scheduler.observe_decode_round(avg_tok_per_stream);
        }

        std::scoped_lock lock(stats_mutex_);
        ++backend_stats_.batch_submissions;
        backend_stats_.batch_rows += pending.size();
        if (pending.size() < backend_stats_.batch_size_counts.size())
          ++backend_stats_.batch_size_counts[pending.size()];
        for (const auto &item : pending) {
          const auto w = item.task->pending_input_.size();
          if (w < backend_stats_.batch_width_counts.size())
            ++backend_stats_.batch_width_counts[w];
        }
        if (owner.shared_head && projection.dispatch_count) {
          ++backend_stats_.shared_head_projection_calls;
          if (owner.shared_head->last_rows() > 1)
            ++backend_stats_.shared_head_multirow_calls;
          backend_stats_.shared_head_projection_rows +=
              owner.shared_head->last_rows();
          backend_stats_.shared_head_projection_gpu_ms +=
              projection.timing.gpu_ms;
          backend_stats_.actual_m = owner.shared_head->last_rows();
        }
        backend_stats_.actual_b = std::uint32_t(pending.size());
        backend_stats_.actual_w = 1;
        for (const auto &item : pending)
          backend_stats_.actual_w =
              std::max(backend_stats_.actual_w,
                       std::uint32_t(item.task->pending_input_.size()));
        backend_stats_.batch_gpu_ms +=
            forward.timing.gpu_ms + projection.timing.gpu_ms + sample_gpu_ms;
        if (!ar_rows.empty() &&
            ((fused_sampled &&
              owner.context.capabilities().profile.compute_icb_eligible) ||
             (owner.batch_sampler && owner.batch_sampler->uses_icb()) ||
             (owner.coupled_batch_sampler &&
              owner.coupled_batch_sampler->uses_icb())))
          ++backend_stats_.batch_sampler_icb_replays;
      }
    } catch (const std::exception &error) {
      for (auto &item : pending) {
        if (item.task->pending_verify_) {
          try {
            item.task->program().abort_verify();
          } catch (...) {
          }
        }
        outcomes[item.index] = ExecutionResult{.kind = EventKind::failed,
                                               .finish_reason = "error",
                                               .error = error.what()};
      }
    }
  }

  // Service non-batchable long prefill tasks AFTER the ready decode batch,
  // bounded by the #5164 contended chunk budget + decode-floor duty cycle.
  const bool ext_contended =
      owner.external_gpu_contended.load(std::memory_order_acquire);
  const bool prefill_contended = has_active_decodes || ext_contended;
  const std::size_t pf_cap =
      prefill_contended ? owner.scheduler.calibrated_contended_budget() : 0;
  for (std::size_t i : non_batchable) {
    auto *task = static_cast<NativeTask *>(tasks[i]);
    const bool cancelled = task->cancelled_->load(std::memory_order_acquire);
    const bool was_decode = !cancelled && task->offset_ >= task->prompt_.size();
    if (has_active_decodes && !was_decode && !cancelled &&
        owner.scheduler.should_defer_prefill(task->slot_)) {
      continue;
    }
    try {
      outcomes[i] = task->advance_contended(pf_cap, prefill_contended);
      if (was_decode)
        owner.scheduler.observe_decode_round(1.0);
      std::scoped_lock lock(stats_mutex_);
      ++backend_stats_.batch_fallback_steps;
    } catch (const std::exception &error) {
      outcomes[i] = ExecutionResult{.kind = EventKind::failed,
                                    .finish_reason = "error",
                                    .error = error.what()};
    }
  }
  return outcomes;
}

void NativeBackend::populate_stats(StatsReport &stats) const {
  std::scoped_lock lock(stats_mutex_);
  stats.backend_kind = backend_stats_.backend_kind;
  if (ar_program_->slot_speculative) {
    stats.speculative_backend = "dflash2";
    stats.max_draft_tokens = ar_program_->slot_dflash_draft_tokens;
  } else if (ar_program_->slot_mtp_enabled) {
    stats.speculative_backend = "mtp";
    stats.max_draft_tokens = ar_program_->slot_mtp_draft_tokens;
  } else {
    stats.speculative_backend = "none";
  }
  stats.model_execution_kind = backend_stats_.model_execution_kind;
  stats.batch_submissions = backend_stats_.batch_submissions;
  stats.batch_rows = backend_stats_.batch_rows;
  stats.batch_fallback_steps = backend_stats_.batch_fallback_steps;
  stats.batch_size_counts = backend_stats_.batch_size_counts;
  stats.batch_width_counts = backend_stats_.batch_width_counts;
  stats.batched_ar_steps = backend_stats_.batched_ar_steps;
  stats.batched_verify_rounds = backend_stats_.batched_verify_rounds;
  stats.batch_gpu_ms = backend_stats_.batch_gpu_ms;
  stats.batch_sampler_icb_replays = backend_stats_.batch_sampler_icb_replays;
  stats.batch_sampling_host_read_bytes =
      backend_stats_.batch_sampling_host_read_bytes;
  stats.batch_compact_fallback_rows =
      backend_stats_.batch_compact_fallback_rows;
  stats.shared_head_projection_calls =
      backend_stats_.shared_head_projection_calls;
  stats.shared_head_multirow_calls = backend_stats_.shared_head_multirow_calls;
  stats.shared_head_projection_rows =
      backend_stats_.shared_head_projection_rows;
  stats.shared_head_projection_gpu_ms =
      backend_stats_.shared_head_projection_gpu_ms;
  stats.actual_b = backend_stats_.actual_b;
  stats.actual_w = backend_stats_.actual_w;
  stats.prefix_hits = backend_stats_.prefix_hits;
  stats.prefix_partial_hits = backend_stats_.prefix_partial_hits;
  stats.prefix_misses = backend_stats_.prefix_misses;
  stats.prefix_evictions = backend_stats_.prefix_evictions;
  stats.prefix_restored_tokens = backend_stats_.prefix_restored_tokens;
  stats.ar_steps = backend_stats_.ar_steps;
  stats.ar_gpu_timed_steps = backend_stats_.ar_gpu_timed_steps;
  stats.ar_layer_calls = backend_stats_.ar_layer_calls;
  stats.ar_host_read_bytes = backend_stats_.ar_host_read_bytes;
  stats.ar_prefill_gpu_ms = backend_stats_.ar_prefill_gpu_ms;
  stats.ar_decode_gpu_ms = backend_stats_.ar_decode_gpu_ms;
  stats.spec_rounds = backend_stats_.spec_rounds;
  stats.spec_cost_completed_rounds = backend_stats_.spec_cost_completed_rounds;
  stats.spec_cost_aborted_rounds = backend_stats_.spec_cost_aborted_rounds;
  stats.spec_cost_committed_tokens = backend_stats_.spec_cost_committed_tokens;
  stats.spec_cost_round_wall_ms = backend_stats_.spec_cost_round_wall_ms;
  stats.spec_cost_phase_wall_ms = backend_stats_.spec_cost_phase_wall_ms;
  stats.spec_cost_depth_rounds = backend_stats_.spec_cost_depth_rounds;
  stats.spec_cost_depth_committed_tokens =
      backend_stats_.spec_cost_depth_committed_tokens;
  stats.spec_cost_depth_wall_ms = backend_stats_.spec_cost_depth_wall_ms;
  stats.spec_gpu_timed_rounds = backend_stats_.spec_gpu_timed_rounds;
  stats.spec_target_verify_calls = backend_stats_.spec_target_verify_calls;
  stats.spec_replay_calls = backend_stats_.spec_replay_calls;
  stats.spec_proposal_tokens = backend_stats_.spec_proposal_tokens;
  stats.spec_accepted_tokens = backend_stats_.spec_accepted_tokens;
  stats.spec_host_read_bytes = backend_stats_.spec_host_read_bytes;
  stats.spec_context_rows_projected =
      backend_stats_.spec_context_rows_projected;
  stats.spec_context_rows_reused = backend_stats_.spec_context_rows_reused;
  stats.spec_context_rows_launched = backend_stats_.spec_context_rows_launched;
  stats.short_matrix_calls = backend_stats_.short_matrix_calls;
  stats.short_lane_calls = backend_stats_.short_lane_calls;
  stats.spec_draft_gpu_ms = backend_stats_.spec_draft_gpu_ms;
  stats.spec_verify_gpu_ms = backend_stats_.spec_verify_gpu_ms;
  stats.spec_verify_selector_gpu_ms =
      backend_stats_.spec_verify_selector_gpu_ms;
  stats.spec_commit_gpu_ms = backend_stats_.spec_commit_gpu_ms;
  stats.sampling_contract_id = backend_stats_.sampling_contract_id;
  stats.numeric_contract_id = backend_stats_.numeric_contract_id;
  stats.chunk_policy_id = backend_stats_.chunk_policy_id;
  stats.execution_identity_id = backend_stats_.execution_identity_id;
  stats.compiled_execution_source_id =
      backend_stats_.compiled_execution_source_id;
  stats.kernel_bundle_id = backend_stats_.kernel_bundle_id;
  stats.batch_execution_kind = backend_stats_.batch_execution_kind;
  stats.artifact_id = backend_stats_.artifact_id;
  stats.route_table_id = backend_stats_.route_table_id;
  stats.actual_t = backend_stats_.actual_t;
  stats.actual_m = backend_stats_.actual_m;
  stats.actual_k = backend_stats_.actual_k;
  stats.kernel_family = backend_stats_.kernel_family;
  stats.fallback_reason = backend_stats_.fallback_reason;
  stats.spec_search_gated_rounds = backend_stats_.spec_search_gated_rounds;
  stats.spec_search_probe_rounds = backend_stats_.spec_search_probe_rounds;
  stats.prefix_shared_hits = backend_stats_.prefix_shared_hits;
  stats.prefix_pinned_entries = backend_stats_.prefix_pinned_entries;
  stats.contended_prefill_chunks = backend_stats_.contended_prefill_chunks;
  stats.packed_batch_calls = backend_stats_.packed_batch_calls;
  stats.packed_batch_tokens = backend_stats_.packed_batch_tokens;
  stats.lora_adapter = backend_stats_.lora_adapter;
  stats.lora_mode = backend_stats_.lora_mode;
  stats.lora_buffer_bytes = backend_stats_.lora_buffer_bytes;
  stats.lora_adapted_layers = backend_stats_.lora_adapted_layers;
  stats.prefill_linear_mode = backend_stats_.prefill_linear_mode;
  stats.prefill_linear_mode_effective =
      backend_stats_.prefill_linear_mode_effective;
  stats.prefill_linear_fallback = backend_stats_.prefill_linear_fallback;
  stats.prefill_linear_fallback_reason =
      backend_stats_.prefill_linear_fallback_reason;
  stats.model_quant_format = backend_stats_.model_quant_format;
  stats.prefill_attention_mode = backend_stats_.prefill_attention_mode;
  stats.prefill_sparse_minference = backend_stats_.prefill_sparse_minference;
  stats.prefill_sparse_minference_effective =
      backend_stats_.prefill_sparse_minference_effective;
  stats.prefill_sparse_minference_inactive_reason =
      backend_stats_.prefill_sparse_minference_inactive_reason;
  stats.prefill_cache_blend = backend_stats_.prefill_cache_blend;
  stats.prefill_ane_mode = backend_stats_.prefill_ane_mode;
  stats.prefill_ane_mode_effective = backend_stats_.prefill_ane_mode_effective;
  stats.prefill_ane_inactive_reason =
      backend_stats_.prefill_ane_inactive_reason;
  stats.prefill_ane_offload = backend_stats_.prefill_ane_offload;
  stats.prefill_ane_active = backend_stats_.prefill_ane_active;
  stats.prefill_ane_split_alpha = backend_stats_.prefill_ane_split_alpha;
  stats.prefill_w4a8_tokens = backend_stats_.prefill_w4a8_tokens;
  stats.prefill_w4a4_tokens = backend_stats_.prefill_w4a4_tokens;
  stats.prefill_sage2_tokens = backend_stats_.prefill_sage2_tokens;
  stats.prefill_minference_blocks_skipped =
      backend_stats_.prefill_minference_blocks_skipped;
  stats.prefill_minference_blocks_total =
      backend_stats_.prefill_minference_blocks_total;
  stats.prefill_cache_blend_hit_tokens =
      backend_stats_.prefill_cache_blend_hit_tokens;
  stats.prefill_cache_blend_recomputed_tokens =
      backend_stats_.prefill_cache_blend_recomputed_tokens;
  stats.prefill_ane_offload_tokens = backend_stats_.prefill_ane_offload_tokens;
  stats.prefill_ane_fallback_tokens =
      backend_stats_.prefill_ane_fallback_tokens;
}

TrimCacheReport NativeBackend::trim_cache(std::uint64_t target_free_bytes) {
  assert_owner();
  if (!ar_program_ || !ar_program_->program) {
    return {};
  }
  const std::uint64_t before_alloc =
      ar_program_->context.current_allocated_bytes();
  auto [evicted, freed_cached] = ar_program_->prefix_cache.trim_unpinned(
      target_free_bytes, [this](std::size_t s) -> QwenForward & {
        return ar_program_->program_at(s);
      });
  if (ar_program_->blend_cache && ar_program_->blend_cache->enabled() &&
      freed_cached < target_free_bytes) {
    const std::uint64_t blend_now = ar_program_->blend_cache->cached_bytes();
    const std::uint64_t need = target_free_bytes - freed_cached;
    const std::uint64_t target_blend =
        blend_now > need ? (blend_now - need) : 0ULL;
    const std::uint64_t blend_freed =
        ar_program_->blend_cache->trim_to_bytes(target_blend);
    freed_cached += blend_freed;
  }
  const std::uint64_t after_alloc =
      ar_program_->context.current_allocated_bytes();
  const std::uint64_t freed_alloc =
      (before_alloc > after_alloc) ? (before_alloc - after_alloc) : 0ULL;
  if (evicted > 0) {
    for (std::size_t s = 0; s < ar_program_->slot_dflash_prefix.size(); ++s) {
      if (s < ar_program_->occupied.size() && !ar_program_->occupied[s]) {
        ar_program_->slot_dflash_prefix[s].clear();
        if (s < ar_program_->slot_mtp_post_frontier.size()) {
          ar_program_->slot_mtp_think_frontier[s].clear();
          ar_program_->slot_mtp_post_frontier[s].clear();
        }
        if (s < ar_program_->slot_rolling_think_anchor_ids.size()) {
          ar_program_->slot_rolling_think_anchor_ids[s] = 0ULL;
          ar_program_->slot_rolling_post_ids[s] = 0ULL;
        }
        if (auto *df = ar_program_->draft_at(s))
          df->reset();
        if (auto *mt = ar_program_->mtp_at(s))
          mt->reset();
      }
    }
    std::scoped_lock lock(stats_mutex_);
    backend_stats_.prefix_evictions += evicted;
    backend_stats_.prefix_pinned_entries =
        ar_program_->prefix_cache.pinned_count();
  }
  return TrimCacheReport{
      .freed_bytes = std::max<std::uint64_t>(freed_alloc, freed_cached),
      .resident_bytes = after_alloc,
      .evicted_entries = evicted,
  };
}

void NativeBackend::set_scheduler_hint(const SchedulerHint &hint) {
  if (ar_program_) {
    ar_program_->external_gpu_contended.store(hint.gpu_contended,
                                              std::memory_order_release);
  }
}

bool NativeBackend::reload_lora(const std::string &lora_adapter,
                                const std::string &lora_mode,
                                std::string *error) {
  assert_owner();
  const std::string effective_mode =
      lora_mode.empty() ? std::string("merge") : lora_mode;
  if (effective_mode != "merge" && effective_mode != "bypass") {
    if (error != nullptr) {
      *error = "lora_mode must be merge or bypass";
    }
    return false;
  }
  if (!ar_program_ || !ar_program_->program) {
    if (error != nullptr) {
      *error = "native backend is not loaded";
    }
    return false;
  }
  if ((!ar_program_->lora_adapter_path.empty() &&
       ar_program_->lora_mode_active == "merge") ||
      (!lora_adapter.empty() && effective_mode == "merge")) {
    if (error != nullptr) {
      *error = "merge mode requires full backend reload";
    }
    return false;
  }
  if (ar_program_->active_occupied_count() != 0U) {
    if (error != nullptr) {
      *error = "cannot hot-swap LoRA while inference slots are active";
    }
    return false;
  }

  try {
    auto &owner = *ar_program_;
    std::shared_ptr<QwenBypassLoraRuntime> new_bypass;
    if (!lora_adapter.empty()) {
      LoadedLoraAdapter loaded = load_lora_adapter_for_qwen(
          lora_adapter, owner.config, owner.lm_head_tied);
      if (owner.shared_head_enabled) {
        for (const auto &u : loaded.updates) {
          if (u.target.is_global && u.target.role == "lm_head") {
            throw std::invalid_argument(
                "batch_projection_prototype=on does not support lm_head LoRA "
                "adaptation");
          }
        }
      }
      const std::uint64_t new_lora_est =
          detail::lora_rt::estimate_lora_gpu_bytes(
              loaded, "bypass", owner.primary_ws_rows,
              std::uint32_t(owner.occupied.size()));
      const std::uint64_t rec_ws =
          owner.context.capabilities().recommended_working_set_bytes;
      const std::uint64_t ws_limit =
          options_.memory_budget_bytes > 0
              ? std::min<std::uint64_t>(rec_ws, options_.memory_budget_bytes)
              : rec_ws;
      const std::uint64_t cur_alloc = owner.context.current_allocated_bytes();
      const std::uint64_t base_alloc =
          cur_alloc > owner.lora_buffer_bytes
              ? (cur_alloc - owner.lora_buffer_bytes)
              : 0ULL;
      if (base_alloc + new_lora_est > ws_limit) {
        throw std::runtime_error(
            "LoRA bypass hot-swap exceeds Metal memory budget: required=" +
            std::to_string(base_alloc + new_lora_est) +
            " available=" + std::to_string(ws_limit));
      }
      new_bypass = detail::lora_rt::build_bypass_lora_runtime(
          owner.context, owner.weights, owner.layers, loaded);
    }

    owner.prefix_cache.clear_all([&](std::size_t s) -> QwenForward & {
      return owner.program_at(s);
    });
    owner.execution_fields.erase("lora");
    owner.execution_fields.erase("lora_mode");
    if (new_bypass != nullptr) {
      owner.execution_fields.emplace("lora", new_bypass->fingerprint);
      owner.execution_fields.emplace("lora_mode", "bypass");
    }
    owner.execution_identity =
        execution_identity_digest(owner.execution_fields);

    for (std::size_t s = 0; s < owner.occupied.size(); ++s) {
      owner.program_at(s).set_bypass_lora(new_bypass, owner.execution_identity);
      owner.spec_controller.reset_slot(s);
      if (auto *dr = owner.draft_at(s)) {
        dr->reset();
        sync_draft_target_ring(*dr, owner.program_at(s).feature_ring_rows());
      }
      if (auto *mt = owner.mtp_at(s)) {
        mt->reset();
      }
      if (s < owner.slot_mtp_post_frontier.size()) {
        owner.slot_mtp_think_frontier[s].clear();
        owner.slot_mtp_post_frontier[s].clear();
      }
      if (auto *mc = owner.mtp_controller_at(s)) {
        mc->on_new_request();
      }
    }

    std::uint64_t scratch_sum = 0;
    if (new_bypass != nullptr) {
      for (std::size_t s = 0; s < owner.occupied.size(); ++s) {
        scratch_sum += owner.program_at(s).lora_scratch_bytes();
      }
    }
    owner.bypass_lora = new_bypass;
    owner.lora_adapter_path = lora_adapter;
    owner.lora_mode_active =
        lora_adapter.empty() ? std::string{} : std::string("bypass");
    owner.lora_buffer_bytes =
        new_bypass != nullptr ? (new_bypass->weight_buffer_bytes + scratch_sum)
                              : 0ULL;
    owner.lora_adapted_layers =
        new_bypass != nullptr ? new_bypass->adapted_layers : 0U;

    options_.lora_adapter = lora_adapter;
    options_.lora_mode =
        lora_adapter.empty() ? std::string("merge") : std::string("bypass");
    {
      std::scoped_lock lock(stats_mutex_);
      backend_stats_.execution_identity_id = owner.execution_identity;
      backend_stats_.prefix_pinned_entries = 0;
      backend_stats_.lora_adapter = owner.lora_adapter_path;
      backend_stats_.lora_mode = owner.lora_mode_active;
      backend_stats_.lora_buffer_bytes = owner.lora_buffer_bytes;
      backend_stats_.lora_adapted_layers = owner.lora_adapted_layers;
    }
    return true;
  } catch (const std::exception &ex) {
    if (error != nullptr) {
      *error = ex.what();
    }
    return false;
  }
}

} // namespace omlx::m5::detail
