#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "omlx_m5/types.h"

namespace omlx::m5::detail {

enum class SpecSearchMode {
  Off,
  Auto,
  On,
};

[[nodiscard]] inline SpecSearchMode parse_spec_search_mode(std::string_view text) {
  if (text.empty() || text == "auto" || text == "AUTO")
    return SpecSearchMode::Auto;
  if (text == "on" || text == "ON" || text == "1" || text == "true")
    return SpecSearchMode::On;
  if (text == "off" || text == "OFF" || text == "0" || text == "false")
    return SpecSearchMode::Off;
  throw std::invalid_argument("spec_search must be auto, on, or off");
}

[[nodiscard]] inline SpecSearchMode resolve_spec_search_mode(const LoadOptions &options) {
  if (const auto it = options.path_overrides.find("spec_search");
      it != options.path_overrides.end()) {
    return parse_spec_search_mode(it->second);
  }
  if (const char *env = std::getenv("OMLX_M5_SPEC_SEARCH"); env && *env) {
    return parse_spec_search_mode(env);
  }
  return parse_spec_search_mode(options.spec_search);
}

struct MtpDepthPolicy {
  std::uint32_t max_draft_tokens = 4;
  bool fixed_depth = false;
};

[[nodiscard]] inline MtpDepthPolicy
resolve_mtp_depth_policy(const LoadOptions &options, std::uint32_t hidden_size,
                         std::uint32_t max_supported = 4,
                         std::uint32_t default_max = 4) {
  std::string_view raw;
  if (const auto it = options.path_overrides.find("mtp_draft_tokens");
      it != options.path_overrides.end()) {
    raw = it->second;
  } else if (const char *env = std::getenv("OMLX_M5_MTP_DRAFT_TOKENS");
             env && *env) {
    raw = env;
  }
  bool explicit_k = false;
  std::uint32_t k = default_max;
  if (!raw.empty()) {
    std::uint32_t parsed = 0;
    const auto [end, err] =
        std::from_chars(raw.data(), raw.data() + raw.size(), parsed);
    if (err != std::errc{} || end != raw.data() + raw.size() || parsed == 0U ||
        parsed > max_supported) {
      throw std::invalid_argument("mtp_draft_tokens must be in [1, 4]");
    }
    k = parsed;
    explicit_k = true;
  } else if (options.speculative.backend == SpeculativeBackend::mtp &&
             options.speculative.draft_tokens > 0U) {
    k = std::min(options.speculative.draft_tokens, max_supported);
  }

  auto is_off_flag = [](std::string_view s) noexcept {
    return s == "0" || s == "off" || s == "OFF" || s == "false" || s == "FALSE";
  };
  auto is_on_flag = [](std::string_view s) noexcept {
    return s == "1" || s == "on" || s == "ON" || s == "true" || s == "TRUE";
  };

  std::string_view fixed_raw;
  if (const auto it = options.path_overrides.find("mtp_fixed_depth");
      it != options.path_overrides.end()) {
    fixed_raw = it->second;
  } else if (const char *env = std::getenv("OMLX_M5_MTP_FIXED_DEPTH");
             env && *env) {
    fixed_raw = env;
  }
  if (!fixed_raw.empty()) {
    if (is_on_flag(fixed_raw)) {
      return {.max_draft_tokens = k, .fixed_depth = true};
    }
    if (is_off_flag(fixed_raw)) {
      return {.max_draft_tokens = k, .fixed_depth = false};
    }
    throw std::invalid_argument(
        "mtp_fixed_depth must be on/1/true or off/0/false");
  }

  const auto spec_mode = resolve_spec_search_mode(options);
  // Tiny unit-test fixtures (hidden_size < 512) stay deterministic unless
  // spec_search="on" is explicitly set. Real models (hidden_size >= 512) enable
  // MtpDepthController by default even when DFlash2's spec_search is "off",
  // locking fixed depth only when mtp_draft_tokens / OMLX_M5_MTP_DRAFT_TOKENS
  // is explicitly overridden without spec_search="on".
  const bool fixed =
      (hidden_size < 512U && spec_mode != SpecSearchMode::On) ||
      (explicit_k && spec_mode != SpecSearchMode::On);
  return {.max_draft_tokens = k, .fixed_depth = fixed};
}

struct SpecControllerConfig {
  SpecSearchMode mode = SpecSearchMode::Auto;
  std::uint32_t max_draft_tokens = 7;
  std::uint32_t max_slots = 1;
  std::uint32_t hidden_size = 5120;
  std::uint32_t num_layers = 64;
  std::uint32_t full_attn_layers = 16;
  std::uint32_t num_experts = 0;
  std::uint32_t experts_per_token = 0;
  double plain_margin = 1.05;
  std::uint32_t warmup_rounds = 4;
  std::uint32_t max_batch_verify_tokens = 48;
};

struct SpecSlotQuery {
  std::size_t slot = 0;
  std::uint32_t max_k = 0;
  std::uint32_t context_tokens = 0;
};

struct SpecSlotDecision {
  std::uint32_t k = 0;
  bool gated = false;
  bool probe = false;
};

// Adaptive speculative depth & MoE bandwidth controller combining:
// 1. SpecSearch online position-conditional survival S(1..K) EMA.
// 2. TensorFold DraftDepth plain_guard (plain_margin = 1.05) + exponential probe backoff.
// 3. P12 / H1-M calibrated MoE unique-expert coupon-collector cost U(M).
// 4. Non-concave lookahead batch goodput allocator across concurrent slots (M <= 48).
class AdaptiveSpecController final {
public:
  static constexpr std::size_t kMaxDepth = 7;
  // Calibrated to M5 Ultra H1-M Qwen3.6-35B-A3B route trace:
  // M=1 -> U(1)=8.00, M=8 -> U(8)=45.00, M=32 -> U(32)=134.8.
  static constexpr double kDefaultTokenCorrelation = 0.727;

  explicit AdaptiveSpecController(const SpecControllerConfig &cfg = {})
      : cfg_(cfg), slots_(std::max<std::size_t>(1, cfg.max_slots)) {
    const bool is_moe = cfg_.num_experts > 0 && cfg_.experts_per_token > 0;
    if (is_moe) {
      // Priors calibrated from M5 Ultra 64c 35B-A3B (AR ~7.7ms, DFlash2 ~6.5ms).
      ema_ar_base_ms_ = 7.7;
      ema_draft_base_ms_ = 5.4;
      ema_draft_tok_ms_ = 0.25;
      ema_commit_ms_ = 0.75;
    } else {
      // Priors calibrated from M5 Ultra 64c 27B + DFlash2 parallel block-diffusion:
      // DFlash2 proposes all K<=7 tokens in ONE BM=8 NAX forward (~7.6ms base)
      // plus ~0.03ms per-token GPU selector step.
      ema_ar_base_ms_ = 24.4;
      ema_draft_base_ms_ = 7.4;
      ema_draft_tok_ms_ = 0.03;
      ema_commit_ms_ = 0.80;
    }
    if (cfg_.hidden_size < 1024) {
      const double scale =
          std::max(0.02, double(cfg_.hidden_size) / 5120.0) *
          std::max(0.05, double(cfg_.num_layers) / 64.0);
      ema_ar_base_ms_ *= scale;
      ema_draft_base_ms_ *= scale;
      ema_draft_tok_ms_ *= scale;
      ema_commit_ms_ *= scale;
    }
    for (auto &slot : slots_)
      reset_slot_state(slot);
  }

  [[nodiscard]] SpecSearchMode mode() const noexcept { return cfg_.mode; }

  void reset_slot(std::size_t slot_idx) {
    if (slot_idx < slots_.size())
      reset_slot_state(slots_[slot_idx]);
  }

  [[nodiscard]] static double
  expected_unique_experts(std::uint32_t num_experts, std::uint32_t top_k,
                          std::uint32_t total_tokens,
                          double token_corr = kDefaultTokenCorrelation) noexcept {
    if (num_experts == 0 || top_k == 0 || total_tokens == 0)
      return 0.0;
    if (top_k >= num_experts)
      return double(num_experts);
    if (total_tokens == 1)
      return double(top_k);
    const double m_eff = 1.0 + token_corr * double(total_tokens - 1);
    const double miss_one = 1.0 - double(top_k) / double(num_experts);
    const double unique = double(num_experts) * (1.0 - std::pow(miss_one, m_eff));
    return std::clamp(unique, double(top_k), double(num_experts));
  }

  [[nodiscard]] static double nax_band_factor(std::uint32_t total_tokens) noexcept {
    if (total_tokens <= 1)
      return 1.00;
    if (total_tokens <= 8)
      return 1.07; // BM=8 single y-tile
    if (total_tokens <= 16)
      return 1.24; // #5309 Skinny NAX RF=2
    if (total_tokens <= 32)
      return 1.52; // #5309 Skinny NAX RF=4
    if (total_tokens <= 48)
      return 1.85; // #5309 Skinny NAX RF=6
    return 2.35 + 0.015 * double(total_tokens - 48);
  }

  [[nodiscard]] SpecSlotDecision select_single_depth(const SpecSlotQuery &query) {
    const std::array<SpecSlotQuery, 1> batch{query};
    return select_batch_depths(batch).front();
  }

  [[nodiscard]] std::vector<SpecSlotDecision>
  select_batch_depths(std::span<const SpecSlotQuery> queries) {
    std::vector<SpecSlotDecision> out(queries.size());
    if (queries.empty())
      return out;

    // Off mode, tiny unit-test fixtures, or single-slot dense DFlash2 in Auto
    // mode: deterministic fixed max_k. On dense B=1 NAX (nax_band_factor(8) ==
    // 1.07), M=8 target verification streams weights once, whereas gating to
    // K=0 drops into non-T-domain verify(M=1)+commit and hurts goodput.
    const bool small_fixture = cfg_.hidden_size < 512;
    const bool is_moe = cfg_.num_experts > 0 && cfg_.experts_per_token > 0;
    const bool auto_dense_solo =
        (cfg_.mode == SpecSearchMode::Auto && !is_moe && queries.size() == 1);
    if (cfg_.mode == SpecSearchMode::Off ||
        (cfg_.mode == SpecSearchMode::Auto && small_fixture) ||
        auto_dense_solo) {
      for (std::size_t i = 0; i < queries.size(); ++i)
        out[i].k = std::min<std::uint32_t>(queries[i].max_k, kMaxDepth);
      return out;
    }

    const std::uint32_t B = std::uint32_t(queries.size());

    // Check if all active slots with max_k > 0 are still in initial warmup.
    // Exception: under MoE concurrency (B >= 3), apply bandwidth-aware allocation
    // immediately even during warmup to avoid a 32-token expert-read storm.
    bool any_warmup = false;
    for (const auto &q : queries) {
      if (q.max_k > 0 && q.slot < slots_.size() &&
          slots_[q.slot].observed_rounds < cfg_.warmup_rounds) {
        any_warmup = true;
        break;
      }
    }
    if (cfg_.mode == SpecSearchMode::Auto && any_warmup && !(is_moe && B >= 3)) {
      std::uint32_t total_m = B;
      for (std::size_t i = 0; i < queries.size(); ++i) {
        const std::uint32_t cap =
            total_m < cfg_.max_batch_verify_tokens
                ? std::min<std::uint32_t>(
                      {queries[i].max_k, std::uint32_t(kMaxDepth),
                       cfg_.max_batch_verify_tokens - total_m})
                : 0u;
        out[i].k = cap;
        total_m += cap;
      }
      return out;
    }

    std::vector<std::uint32_t> k_alloc(queries.size(), 0u);
    const double r0 = evaluate_batch_goodput(queries, k_alloc);

    // Coordinate-lookahead greedy search over K_b in [0..max_k_b] subject to
    // sum(K_b + 1) <= max_batch_verify_tokens. Lookahead across delta=1..max_k
    // jumps cleanly across the fixed draft_base_ms startup cost at K=0 -> K>0.
    std::uint32_t total_m = B;
    double current_goodput = r0;
    for (;;) {
      double best_goodput = current_goodput;
      std::size_t best_idx = queries.size();
      std::uint32_t best_next_k = 0;

      for (std::size_t i = 0; i < queries.size(); ++i) {
        const std::uint32_t limit =
            std::min<std::uint32_t>(queries[i].max_k, std::uint32_t(kMaxDepth));
        if (k_alloc[i] >= limit)
          continue;
        for (std::uint32_t cand_k = k_alloc[i] + 1; cand_k <= limit; ++cand_k) {
          const std::uint32_t delta = cand_k - k_alloc[i];
          if (total_m + delta > cfg_.max_batch_verify_tokens)
            break;
          const std::uint32_t prev = k_alloc[i];
          k_alloc[i] = cand_k;
          const double g = evaluate_batch_goodput(queries, k_alloc);
          k_alloc[i] = prev;
          if (g > best_goodput + 1e-9) {
            best_goodput = g;
            best_idx = i;
            best_next_k = cand_k;
          }
        }
      }
      if (best_idx == queries.size())
        break;
      total_m += (best_next_k - k_alloc[best_idx]);
      k_alloc[best_idx] = best_next_k;
      current_goodput = best_goodput;
    }

    const bool wins_over_plain =
        current_goodput >= cfg_.plain_margin * r0 && total_m > B;

    bool probe_scheduled = false;
    for (std::size_t i = 0; i < queries.size(); ++i) {
      const std::uint32_t limit =
          std::min<std::uint32_t>(queries[i].max_k, std::uint32_t(kMaxDepth));
      auto &st = slots_[std::min(queries[i].slot, slots_.size() - 1)];
      if (limit == 0) {
        out[i].k = 0;
        continue;
      }
      if (wins_over_plain && k_alloc[i] > 0) {
        out[i].k = k_alloc[i];
        st.in_gated_backoff = false;
        st.backoff_interval = 8;
        st.backoff_remaining = 0;
      } else {
        // Plain guard triggered for this slot (negative/marginal speculation).
        if (!st.in_gated_backoff) {
          st.in_gated_backoff = true;
          st.backoff_interval = 8;
          st.backoff_remaining = st.backoff_interval;
        }
        if (st.backoff_remaining > 0) {
          --st.backoff_remaining;
          out[i].k = 0;
          out[i].gated = true;
        } else if (!probe_scheduled &&
                   total_m + std::min<std::uint32_t>(3u, limit) <=
                       cfg_.max_batch_verify_tokens) {
          // Periodic lightweight K<=3 probe to detect reasoning -> code transition.
          const std::uint32_t probe_k = std::min<std::uint32_t>(3u, limit);
          out[i].k = probe_k;
          out[i].probe = true;
          total_m += probe_k;
          probe_scheduled = true;
          st.backoff_interval = std::min<std::uint32_t>(128u, st.backoff_interval * 2u);
          st.backoff_remaining = st.backoff_interval;
        } else {
          out[i].k = 0;
          out[i].gated = true;
        }
      }
    }
    return out;
  }

  void observe_ar_step(double decode_ms) noexcept {
    if (decode_ms > 0.01 && std::isfinite(decode_ms)) {
      ema_ar_base_ms_ = 0.80 * ema_ar_base_ms_ + 0.20 * decode_ms;
    }
  }

  void observe_round(std::size_t slot_idx, std::uint32_t depth_k,
                     std::uint32_t accepted_proposals, bool unclipped_by_budget,
                     double draft_ms, double verify_ms, double commit_ms,
                     std::uint32_t batch_total_m = 0,
                     bool cold_context_projection = false) noexcept {
    auto &st = slots_[std::min(slot_idx, slots_.size() - 1)];
    if (depth_k > 0) {
      ++st.observed_rounds;
      const std::size_t observed_positions =
          unclipped_by_budget
              ? std::min<std::size_t>(depth_k, kMaxDepth)
              : std::min<std::size_t>(accepted_proposals, kMaxDepth);
      for (std::size_t j = 0; j < observed_positions; ++j) {
        const double hit = (accepted_proposals >= j + 1) ? 1.0 : 0.0;
        st.survival[j] = 0.82 * st.survival[j] + 0.18 * hit;
        if (j > 0 && st.survival[j] > st.survival[j - 1])
          st.survival[j] = st.survival[j - 1];
      }
      // Propagate decay to unobserved deeper positions when early rejection occurs
      // so low-acceptance streams quickly pull down tail optimism.
      if (unclipped_by_budget && accepted_proposals < depth_k) {
        for (std::size_t j = depth_k; j < kMaxDepth; ++j) {
          const double prev = st.survival[j - 1];
          st.survival[j] = std::min(st.survival[j], prev * 0.85);
        }
      }
      if (!cold_context_projection && draft_ms > 0.01 &&
          std::isfinite(draft_ms)) {
        const double est_base = std::max(0.05, draft_ms - ema_draft_tok_ms_ * double(depth_k));
        ema_draft_base_ms_ = 0.80 * ema_draft_base_ms_ + 0.20 * est_base;
      }
    }
    const std::uint32_t eff_m =
        batch_total_m > 0 ? batch_total_m : (depth_k + 1u);
    if (verify_ms > 0.01 && std::isfinite(verify_ms)) {
      const double cost_mult = verify_cost_multiplier(eff_m);
      const double implied_ar = verify_ms / std::max(1.0, cost_mult);
      ema_ar_base_ms_ = 0.80 * ema_ar_base_ms_ + 0.20 * implied_ar;
    }
    if (commit_ms > 0.001 && std::isfinite(commit_ms) && depth_k > 0) {
      ema_commit_ms_ = 0.80 * ema_commit_ms_ + 0.20 * commit_ms;
    }
  }

  [[nodiscard]] double expected_tokens(std::size_t slot_idx,
                                       std::uint32_t k) const noexcept {
    const auto &st = slots_[std::min(slot_idx, slots_.size() - 1)];
    double total = 1.0;
    const std::size_t limit = std::min<std::size_t>(k, kMaxDepth);
    for (std::size_t j = 0; j < limit; ++j)
      total += st.survival[j];
    return total;
  }

  [[nodiscard]] double predict_batch_ms(std::span<const SpecSlotQuery> queries,
                                        std::span<const std::uint32_t> k_alloc) const noexcept {
    std::uint32_t total_m = 0;
    std::uint32_t max_k = 0;
    std::uint32_t active_drafts = 0;
    double kv_ms = 0.0;
    const double kv_unit_ms =
        (double(cfg_.full_attn_layers) / 16.0) * 0.085; // ~0.085 ms / 1K ctx / 16-row tile

    for (std::size_t i = 0; i < queries.size(); ++i) {
      const std::uint32_t k = k_alloc[i];
      const std::uint32_t w = k + 1u;
      total_m += w;
      max_k = std::max(max_k, k);
      if (k > 0)
        ++active_drafts;
      const double row_tiles = double((w + 15u) / 16u);
      kv_ms += (double(queries[i].context_tokens) / 1024.0) * kv_unit_ms * row_tiles;
    }

    const double draft_ms =
        active_drafts == 0
            ? 0.0
            : (ema_draft_base_ms_ + ema_draft_tok_ms_ * double(max_k)) *
                  (1.0 + 0.35 * double(active_drafts - 1));
    const double verify_ms =
        ema_ar_base_ms_ * verify_cost_multiplier(total_m) + kv_ms;
    const double commit_ms =
        active_drafts == 0
            ? 0.0
            : ema_commit_ms_ * (1.0 + 0.15 * double(active_drafts - 1));
    return std::max(0.05, draft_ms + verify_ms + commit_ms);
  }

private:
  struct SlotState {
    std::array<double, kMaxDepth> survival{};
    std::uint32_t observed_rounds = 0;
    bool in_gated_backoff = false;
    std::uint32_t backoff_interval = 8;
    std::uint32_t backoff_remaining = 0;
  };

  static void reset_slot_state(SlotState &st) noexcept {
    double p = 1.0;
    for (std::size_t j = 0; j < kMaxDepth; ++j) {
      p *= 0.72;
      st.survival[j] = p;
    }
    st.observed_rounds = 0;
    st.in_gated_backoff = false;
    st.backoff_interval = 8;
    st.backoff_remaining = 0;
  }

  [[nodiscard]] double verify_cost_multiplier(std::uint32_t total_m) const noexcept {
    if (total_m <= 1)
      return 1.0;
    const bool is_moe = cfg_.num_experts > 0 && cfg_.experts_per_token > 0;
    if (!is_moe)
      return nax_band_factor(total_m);
    // MoE: ~40% shared/attention/router + ~60% routed expert weights at M=1.
    // Expert weight bytes scale directly with unique routed experts U(M) / top_k.
    const double u_m = expected_unique_experts(
        cfg_.num_experts, cfg_.experts_per_token, total_m);
    const double expert_ratio = u_m / double(cfg_.experts_per_token);
    return 0.40 * nax_band_factor(total_m) + 0.60 * expert_ratio;
  }

  [[nodiscard]] double evaluate_batch_goodput(
      std::span<const SpecSlotQuery> queries,
      std::span<const std::uint32_t> k_alloc) const noexcept {
    double expected_toks = 0.0;
    for (std::size_t i = 0; i < queries.size(); ++i)
      expected_toks += expected_tokens(queries[i].slot, k_alloc[i]);
    const double ms = predict_batch_ms(queries, k_alloc);
    return expected_toks / ms; // tokens per ms
  }

  SpecControllerConfig cfg_;
  std::vector<SlotState> slots_;
  double ema_ar_base_ms_ = 24.0;
  double ema_draft_base_ms_ = 6.2;
  double ema_draft_tok_ms_ = 0.35;
  double ema_commit_ms_ = 0.85;
};

} // namespace omlx::m5::detail
