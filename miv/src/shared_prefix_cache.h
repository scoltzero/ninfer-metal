#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "omlx_m5/execution_contract.h"
#include "omlx_m5/kv_page_pool.h"
#include "omlx_m5/mtp_forward.h"
#include "omlx_m5/qwen_forward.h"
#include "omlx_m5/types.h"

namespace omlx::m5::detail {

namespace prefix_detail {

[[nodiscard]] inline std::uint64_t fnv1a_tokens(std::span<const std::uint32_t> tokens,
                                                std::uint64_t seed = 14695981039346656037ULL) noexcept {
  std::uint64_t h = seed;
  for (std::uint32_t tok : tokens) {
    h ^= std::uint64_t(tok);
    h *= 1099511628211ULL;
  }
  return h;
}

[[nodiscard]] inline std::vector<std::uint64_t>
compute_page_hashes(std::span<const std::uint32_t> tokens,
                    std::size_t page_tokens = 64) {
  std::vector<std::uint64_t> hashes;
  if (page_tokens == 0)
    return hashes;
  const std::size_t full_pages = tokens.size() / page_tokens;
  hashes.reserve(full_pages);
  std::uint64_t running = 14695981039346656037ULL;
  for (std::size_t p = 0; p < full_pages; ++p) {
    running = fnv1a_tokens(tokens.subspan(p * page_tokens, page_tokens), running);
    hashes.push_back(running);
  }
  return hashes;
}

} // namespace prefix_detail

// Engine-wide Cross-Slot COW Zero-Move UMA Prefix Cache combining:
// 1. Cross-request / cross-slot prefix sharing (ninfer ebdc67e5) with 64-tok page-hash indexing.
// 2. Pre-tail 1024-token absolute-boundary GDN/KV anchor checkpoints + explicit block pinning.
// 3. Zero-move GPU-resident UMA continuation slabs + exact unique physical KV page budget accounting.
// 4. WAKMan 3-expert shadow ghost tournament (LRU vs LFU vs CostGDSF).
// 5. KVSET online Mattson stack-distance MRC estimator for dynamic page-pressure sizing.
template <class SamplingKeyType>
class SharedPrefixCache final {
public:
  struct Entry {
    std::uint64_t id = 0;
    std::string identity;
    std::vector<std::uint32_t> tokens;
    std::vector<std::uint64_t> page_hashes;
    std::uint64_t prompt_hash = 0;
    std::size_t source_slot = 0;
    QwenUmaContinuation continuation;
    std::optional<MtpPrefixCheckpoint> mtp;
    std::vector<std::byte> last_logits;
    SamplingKeyType sampled_key{};
    std::uint32_t sampled_token = 0;
    bool has_anchor_logits = true;
    bool pinned = false;
    std::uint64_t non_kv_bytes = 0;
    std::uint64_t kv_page_bytes_all_layers = 0;
    const void *kv_allocator_id = nullptr;
    std::uint64_t accounted_bytes = 0;
    std::uint64_t frequency = 1;
    std::uint64_t last_access_tick = 0;
    double recovery_cost = 0.0;
  };

  struct MatchResult {
    std::size_t entry_index = 0;
    std::size_t matched_tokens = 0;
    std::size_t preferred_slot = 0;
    bool exact_prompt = false;
    bool cross_slot_shared = false;
  };

  explicit SharedPrefixCache(std::uint64_t budget_bytes = 0)
      : budget_bytes_(budget_bytes) {}

  [[nodiscard]] bool enabled() const noexcept { return budget_bytes_ > 0; }
  [[nodiscard]] std::uint64_t budget_bytes() const noexcept { return budget_bytes_; }
  [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
  [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }

  [[nodiscard]] std::uint64_t cached_bytes() const noexcept {
    return total_non_kv_bytes_ + total_unique_kv_bytes_;
  }

  [[nodiscard]] std::uint64_t cached_kv_bytes() const noexcept {
    return total_unique_kv_bytes_;
  }

  [[nodiscard]] std::size_t unique_retained_kv_pages() const noexcept {
    return retained_kv_pages_.size();
  }

  [[nodiscard]] std::uint64_t pinned_count() const noexcept {
    std::uint64_t count = 0;
    for (const auto &e : entries_)
      if (e.pinned)
        ++count;
    return count;
  }

  [[nodiscard]] std::size_t unpinned_count() const noexcept {
    std::size_t count = 0;
    for (const auto &e : entries_)
      if (!e.pinned)
        ++count;
    return count;
  }

  void set_max_unpinned_entries(std::size_t cap) noexcept {
    max_unpinned_entries_ = cap;
  }

  [[nodiscard]] std::size_t max_unpinned_entries() const noexcept {
    return max_unpinned_entries_;
  }

  [[nodiscard]] std::uint64_t last_inserted_id() const noexcept {
    return last_inserted_id_;
  }

  // ninfer 3b76b98c recovery cost: linear MLP/GDN O(L) + quadratic attention O(L^2).
  [[nodiscard]] static double compute_recovery_cost(std::size_t token_len) noexcept {
    const double l = double(token_len);
    return l + (l * l) / 4096.0;
  }

  // KVSET online Mattson reuse locality in [0, 1].
  [[nodiscard]] double mrc_reuse_locality() const noexcept {
    if (mrc_total_lookups_ == 0)
      return 0.5;
    const std::uint64_t near_hits =
        mrc_hist_[0] + mrc_hist_[1] + mrc_hist_[2]; // stack depth <= 64
    return std::clamp(double(near_hits) / double(mrc_total_lookups_), 0.0, 1.0);
  }

  [[nodiscard]] std::uint64_t
  effective_budget_bytes(std::uint32_t free_pages = 0,
                         std::uint32_t physical_pages = 0) const noexcept {
    if (budget_bytes_ == 0 || physical_pages == 0)
      return budget_bytes_;
    // Retained cache pages ARE the intended UMA resident prefix cache.
    // Only apply gentle unpinned throttling when physical headroom for active
    // decode is genuinely scarce (< 16 free pages = < 1024 tokens AND < 6.25%).
    const std::uint32_t low_watermark =
        std::min<std::uint32_t>(16U, std::max<std::uint32_t>(1U, physical_pages / 16U));
    if (free_pages >= low_watermark)
      return budget_bytes_;
    const double locality = mrc_reuse_locality();
    const double pressure_scale = 0.70 + 0.25 * locality;
    return std::uint64_t(double(budget_bytes_) * pressure_scale);
  }

  [[nodiscard]] std::optional<MatchResult>
  find_best(std::string_view execution_identity,
            std::span<const std::uint32_t> prompt, SamplingContract contract,
            bool prefer_mtp = false) const {
    if (budget_bytes_ == 0 || entries_.empty() || prompt.empty())
      return std::nullopt;
    const auto prompt_page_hashes = prefix_detail::compute_page_hashes(prompt);
    std::size_t best_any_idx = entries_.size();
    std::size_t best_any_len = 0;
    std::size_t best_mtp_idx = entries_.size();
    std::size_t best_mtp_len = 0;

    for (std::size_t i = 0; i < entries_.size(); ++i) {
      const auto &e = entries_[i];
      const std::size_t len = e.tokens.size();
      if (len > prompt.size() || e.identity != execution_identity)
        continue;
      if (len == prompt.size() && !e.has_anchor_logits)
        continue;
      if (!prefill_checkpoint_compatible(len, prompt.size(), contract))
        continue;
      const bool has_mtp = e.mtp.has_value();
      if (prefer_mtp && !has_mtp && best_mtp_idx != entries_.size())
        continue;
      const std::size_t cur_best_idx =
          (prefer_mtp && has_mtp) ? best_mtp_idx : best_any_idx;
      const std::size_t cur_best_len =
          (prefer_mtp && has_mtp) ? best_mtp_len : best_any_len;
      if (cur_best_idx != entries_.size() && len < cur_best_len)
        continue;
      // Fast O(L/64) page-hash filter before exact token equality check.
      const std::size_t pages = e.page_hashes.size();
      if (pages <= prompt_page_hashes.size() && pages > 0 &&
          e.page_hashes[pages - 1] != prompt_page_hashes[pages - 1]) {
        continue;
      }
      if (!std::equal(e.tokens.begin(), e.tokens.end(), prompt.begin()))
        continue;
      // Tie-break equal-length matches toward more recently accessed entry.
      if (best_any_idx == entries_.size() || len > best_any_len ||
          (len == best_any_len &&
           e.last_access_tick >= entries_[best_any_idx].last_access_tick)) {
        best_any_idx = i;
        best_any_len = len;
      }
      if (has_mtp &&
          (best_mtp_idx == entries_.size() || len > best_mtp_len ||
           (len == best_mtp_len &&
            e.last_access_tick >= entries_[best_mtp_idx].last_access_tick))) {
        best_mtp_idx = i;
        best_mtp_len = len;
      }
    }
    const std::size_t chosen_idx =
        (prefer_mtp && best_mtp_idx != entries_.size()) ? best_mtp_idx
                                                        : best_any_idx;
    if (chosen_idx == entries_.size())
      return std::nullopt;
    const std::size_t chosen_len =
        (prefer_mtp && best_mtp_idx != entries_.size()) ? best_mtp_len
                                                        : best_any_len;
    return MatchResult{
        .entry_index = chosen_idx,
        .matched_tokens = chosen_len,
        .preferred_slot = entries_[chosen_idx].source_slot,
        .exact_prompt = (chosen_len == prompt.size()),
        .cross_slot_shared = false,
    };
  }

  void record_miss(std::span<const std::uint32_t> prompt) {
    if (budget_bytes_ == 0 || prompt.empty())
      return;
    const std::uint64_t h = prefix_detail::fnv1a_tokens(prompt);
    observe_mattson_stack(h);
    observe_wakman_ghost_miss(h);
  }

  template <class SlotProgramFn>
  Entry &restore_into_slot(std::size_t entry_index, std::size_t dst_slot,
                           SlotProgramFn &&program_at, bool &out_cross_slot) {
    auto &entry = entries_.at(entry_index);
    ++entry.frequency;
    entry.last_access_tick = ++clock_tick_;
    observe_mattson_stack(entry.prompt_hash);

    QwenForward &dst_prog = program_at(dst_slot);
    QwenForward &src_prog = program_at(entry.source_slot);
    // Zero-move UMA restore overwrites 100% of GDN recurrent states, active
    // feature_history, and KV page table mappings directly — skip dst_prog.reset()
    // to eliminate 146.8 MiB of redundant single-core CPU memset per warm hit!
    if (dst_slot == entry.source_slot) {
      dst_prog.restore_uma(entry.continuation);
      out_cross_slot = false;
    } else {
      dst_prog.restore_shared_uma(entry.continuation, &src_prog);
      out_cross_slot = true;
    }
    return entry;
  }

  void update_anchor_sample(std::string_view execution_identity,
                            std::span<const std::uint32_t> prompt,
                            const SamplingKeyType &skey,
                            std::uint32_t sampled_token) {
    const std::uint64_t ph = prefix_detail::fnv1a_tokens(prompt);
    for (auto &e : entries_) {
      if (e.has_anchor_logits && e.prompt_hash == ph &&
          e.tokens.size() == prompt.size() && e.identity == execution_identity &&
          std::equal(e.tokens.begin(), e.tokens.end(), prompt.begin())) {
        e.sampled_key = skey;
        e.sampled_token = sampled_token;
      }
    }
  }

  template <class SlotProgramFn>
  std::uint64_t insert_checkpoint(std::size_t source_slot,
                                  std::string_view execution_identity,
                                  std::span<const std::uint32_t> tokens,
                                  const SamplingKeyType &skey,
                                  std::uint32_t sampled_token,
                                  bool include_anchor_logits,
                                  bool pin_entry,
                                  SlotProgramFn &&program_at,
                                  bool *out_inserted = nullptr,
                                  std::uint64_t supersede_entry_id = 0,
                                  std::optional<MtpPrefixCheckpoint> mtp = std::nullopt) {
    last_inserted_id_ = 0;
    if (out_inserted)
      *out_inserted = false;
    if (budget_bytes_ == 0 || tokens.empty())
      return 0;
    if (mtp && (std::size_t(mtp->tokens) + 1U != tokens.size() ||
                mtp->kv_allocator_id() == nullptr)) {
      throw std::invalid_argument("MTP checkpoint does not match target prefix");
    }
    QwenForward &prog = program_at(source_slot);
    const std::uint64_t logits_bytes =
        include_anchor_logits ? std::uint64_t(prog.vocabulary_size()) * 2ULL : 0ULL;
    const auto mtp_non_kv = mtp ? mtp->non_kv_bytes() : 0ULL;
    const auto mtp_kv_bytes =
        mtp ? std::uint64_t(mtp->kv_pages.size()) * mtp->kv_page_bytes : 0ULL;
    const std::uint64_t non_kv_req =
        prog.non_kv_checkpoint_bytes() +
        std::uint64_t(tokens.size()) * sizeof(std::uint32_t) + logits_bytes +
        execution_identity.size() + sizeof(Entry) + mtp_non_kv;
    const std::uint64_t kv_page_bytes_all = prog.kv_page_all_layers_bytes();
    const auto *pool = prog.kv_page_pool();
    const void *alloc_id =
        pool ? static_cast<const void *>(pool->shared_allocator().get())
             : nullptr;
    const std::vector<std::uint32_t> current_slots =
        pool ? pool->mapped_slots() : std::vector<std::uint32_t>{};
    const std::uint64_t full_entry_bytes =
        non_kv_req + std::uint64_t(current_slots.size()) * kv_page_bytes_all +
        mtp_kv_bytes;
    if (full_entry_bytes > budget_bytes_)
      return 0;

    const std::uint64_t ph = prefix_detail::fnv1a_tokens(tokens);
    // Replace existing identical checkpoint without counting as budget eviction.
    for (std::size_t i = entries_.size(); i-- > 0;) {
      if (entries_[i].prompt_hash == ph &&
          entries_[i].tokens.size() == tokens.size() &&
          entries_[i].identity == execution_identity &&
          std::equal(entries_[i].tokens.begin(), entries_[i].tokens.end(),
                     tokens.begin())) {
        if (!include_anchor_logits && entries_[i].has_anchor_logits) {
          // Preserve the richer target anchor. The MTP lease owns its
          // own source and can upgrade a matching target-only entry.
          auto &existing = entries_[i];
          if (mtp && !existing.mtp && cached_bytes() <= budget_bytes_ &&
              mtp_non_kv + marginal_kv_bytes(mtp->kv_allocator_id(),
                                            mtp->kv_pages, mtp->kv_page_bytes)
                  <= budget_bytes_ - cached_bytes()) {
            existing.mtp = std::move(mtp);
            existing.non_kv_bytes += mtp_non_kv;
            existing.accounted_bytes += mtp_non_kv + mtp_kv_bytes;
            existing.continuation.accounted_bytes = existing.accounted_bytes;
            total_non_kv_bytes_ += mtp_non_kv;
            retain_page_accounting(existing.mtp->kv_allocator_id(),
                                   existing.mtp->kv_pages,
                                   existing.mtp->kv_page_bytes);
          }
          existing.pinned = existing.pinned || pin_entry;
          last_inserted_id_ = existing.id;
          if (out_inserted)
            *out_inserted = true;
          return 0;
        }
        pin_entry = pin_entry || entries_[i].pinned;
        release_entry(i, program_at, false);
      }
    }

    // In-place rolling supersede: release the immediately extended unpinned
    // ancestor BEFORE budget accounting and checkpoint_uma() so its UMA slab
    // returns to UmaCheckpointPool::free_slabs for zero-alloc 1x reuse and any
    // CoW-diverged partial tail KV page is freed while live pages keep ref >= 1.
    // Pinned ancestors (e.g. shared System Prompt) are NEVER superseded.
    if (supersede_entry_id != 0ULL) {
      for (std::size_t i = entries_.size(); i-- > 0;) {
        const auto &prev = entries_[i];
        if (prev.id != supersede_entry_id)
          continue;
        if (!prev.pinned && prev.identity == execution_identity &&
            prev.tokens.size() < tokens.size() &&
            std::equal(prev.tokens.begin(), prev.tokens.end(), tokens.begin())) {
          release_entry(i, program_at, false);
        }
        break;
      }
    }

    std::uint64_t evicted = 0;
    if (!pin_entry && max_unpinned_entries_ > 0U) {
      while (unpinned_count() >= max_unpinned_entries_) {
        if (!evict_best_victim(program_at, false))
          break;
        ++evicted;
      }
    }

    auto marginal_required = [&]() -> std::uint64_t {
      return non_kv_req +
             marginal_kv_bytes(alloc_id, current_slots, kv_page_bytes_all) +
             (mtp ? marginal_kv_bytes(mtp->kv_allocator_id(), mtp->kv_pages,
                                      mtp->kv_page_bytes) : 0ULL);
    };

    const std::uint32_t free_p = pool ? pool->free_pages() : 0U;
    const std::uint32_t phys_p = pool ? pool->physical_pages() : 0U;
    const std::uint64_t initial_marginal = marginal_required();
    const std::uint64_t eff_budget =
        std::max<std::uint64_t>(
            pin_entry ? budget_bytes_ : effective_budget_bytes(free_p, phys_p),
            std::min<std::uint64_t>(budget_bytes_, initial_marginal));

    while (!entries_.empty()) {
      const std::uint64_t req = marginal_required();
      if (cached_bytes() <= eff_budget &&
          req <= eff_budget - cached_bytes()) {
        break;
      }
      if (!evict_best_victim(program_at, false))
        break;
      ++evicted;
    }
    while (!entries_.empty()) {
      const std::uint64_t req = marginal_required();
      if (cached_bytes() <= budget_bytes_ &&
          req <= budget_bytes_ - cached_bytes()) {
        break;
      }
      if (!evict_best_victim(program_at, false))
        break;
      ++evicted;
    }
    if (cached_bytes() > budget_bytes_ ||
        marginal_required() > budget_bytes_ - cached_bytes()) {
      return evicted;
    }

    Entry entry;
    entry.id = ++next_id_;
    entry.identity = std::string(execution_identity);
    entry.tokens.assign(tokens.begin(), tokens.end());
    entry.page_hashes = prefix_detail::compute_page_hashes(tokens);
    entry.prompt_hash = ph;
    entry.source_slot = source_slot;
    entry.continuation = prog.checkpoint_uma();
    entry.continuation.accounted_bytes = full_entry_bytes;
    entry.mtp = std::move(mtp);
    if (include_anchor_logits) {
      entry.last_logits = prog.read_last_logits_bf16();
      entry.sampled_key = skey;
      entry.sampled_token = sampled_token;
      entry.has_anchor_logits = true;
    } else {
      entry.has_anchor_logits = false;
    }
    entry.pinned = pin_entry;
    entry.non_kv_bytes = non_kv_req;
    entry.kv_page_bytes_all_layers = kv_page_bytes_all;
    entry.kv_allocator_id = alloc_id;
    entry.accounted_bytes = full_entry_bytes;
    entry.frequency = 1;
    entry.last_access_tick = ++clock_tick_;
    entry.recovery_cost = compute_recovery_cost(tokens.size());

    total_non_kv_bytes_ += entry.non_kv_bytes;
    retain_page_accounting(entry.kv_allocator_id, entry.continuation.kv_pages,
                           entry.kv_page_bytes_all_layers);
    if (entry.mtp) {
      retain_page_accounting(entry.mtp->kv_allocator_id(), entry.mtp->kv_pages,
                             entry.mtp->kv_page_bytes);
    }

    observe_mattson_stack(ph);
    last_inserted_id_ = entry.id;
    entries_.push_back(std::move(entry));
    if (out_inserted)
      *out_inserted = true;
    return evicted;
  }

  template <class SlotProgramFn>
  bool release_unpinned_slot_entry_by_id(std::uint64_t entry_id,
                                         std::size_t source_slot,
                                         std::string_view execution_identity,
                                         SlotProgramFn &&program_at) {
    if (entry_id == 0ULL)
      return false;
    for (std::size_t i = entries_.size(); i-- > 0;) {
      const auto &prev = entries_[i];
      if (prev.id != entry_id)
        continue;
      if (!prev.pinned && prev.source_slot == source_slot &&
          prev.identity == execution_identity) {
        release_entry(i, program_at, false);
        return true;
      }
      break;
    }
    return false;
  }

  template <class SlotProgramFn>
  bool evict_for_page_pressure(std::optional<std::size_t> prefer_slot,
                               SlotProgramFn &&program_at) {
    if (entries_.empty())
      return false;
    // If KV pools are per-slot, prioritize freeing a retained lease on prefer_slot
    // first if one exists; under shared KV arena any victim frees physical pages.
    if (prefer_slot.has_value()) {
      const auto idx = select_victim_index(false, prefer_slot);
      if (idx < entries_.size()) {
        record_wakman_eviction(idx);
        release_entry(idx, program_at, true);
        return true;
      }
    }
    if (evict_best_victim(program_at, false))
      return true;
    // Last resort under hard physical page exhaustion: allow evicting a pinned entry
    // rather than throwing an unrecoverable overflow_error mid-decode.
    return evict_best_victim(program_at, true);
  }

  template <class SlotProgramFn>
  bool evict_for_mtp_page_pressure(SlotProgramFn &&program_at) {
    // MTP and target have different physical arenas.
    auto index = select_victim_index(false, std::nullopt, true);
    if (index == entries_.size())
      index = select_victim_index(true, std::nullopt, true);
    if (index == entries_.size())
      return false;
    record_wakman_eviction(index);
    release_entry(index, program_at, true);
    return true;
  }

  template <class SlotProgramFn>
  std::pair<std::uint64_t, std::uint64_t>
  trim_unpinned(std::uint64_t target_free_bytes,
                SlotProgramFn &&program_at) noexcept {
    const std::uint64_t before_bytes = cached_bytes();
    std::uint64_t evicted = 0;
    while (unpinned_count() > 0) {
      const std::uint64_t freed_so_far =
          (before_bytes > cached_bytes()) ? (before_bytes - cached_bytes())
                                          : 0ULL;
      if (target_free_bytes > 0 && freed_so_far >= target_free_bytes) {
        break;
      }
      try {
        if (!evict_best_victim(program_at, false)) {
          break;
        }
        ++evicted;
      } catch (...) {
        break;
      }
    }
    try {
      program_at(0).trim_uma_checkpoint_pool(0);
    } catch (...) {
    }
    const std::uint64_t freed_bytes =
        (before_bytes > cached_bytes()) ? (before_bytes - cached_bytes())
                                        : 0ULL;
    return {evicted, freed_bytes};
  }

  template <class SlotProgramFn>
  std::uint64_t clear_all(SlotProgramFn &&program_at) noexcept {
    const std::uint64_t count = entries_.size();
    while (!entries_.empty()) {
      try {
        release_entry(entries_.size() - 1, program_at, false);
      } catch (...) {
        entries_.pop_back();
      }
    }
    retained_kv_pages_.clear();
    total_non_kv_bytes_ = 0;
    total_unique_kv_bytes_ = 0;
    if (count > 0) {
      try {
        program_at(0).trim_uma_checkpoint_pool(0);
      } catch (...) {
      }
    }
    return count;
  }

  [[nodiscard]] std::array<double, 3> expert_weights() const noexcept {
    return expert_weights_;
  }

private:
  struct GhostRecord {
    std::uint64_t prompt_hash = 0;
    std::uint8_t wanted_keep_mask = 0; // bit i = expert i did NOT pick this victim
  };

  void observe_mattson_stack(std::uint64_t hash) {
    ++mrc_total_lookups_;
    auto it = std::find(mattson_stack_.begin(), mattson_stack_.end(), hash);
    if (it != mattson_stack_.end()) {
      const std::size_t dist = std::size_t(it - mattson_stack_.begin()) + 1;
      if (dist <= 4)
        ++mrc_hist_[0];
      else if (dist <= 16)
        ++mrc_hist_[1];
      else if (dist <= 64)
        ++mrc_hist_[2];
      else
        ++mrc_hist_[3];
      mattson_stack_.erase(it);
    }
    mattson_stack_.push_front(hash);
    if (mattson_stack_.size() > 256)
      mattson_stack_.pop_back();
  }

  void observe_wakman_ghost_miss(std::uint64_t hash) noexcept {
    for (auto it = ghosts_.begin(); it != ghosts_.end(); ++it) {
      if (it->prompt_hash != hash)
        continue;
      const std::uint8_t mask = it->wanted_keep_mask;
      ghosts_.erase(it);
      if (mask == 0 || mask == 0x7)
        return;
      double sum = 0.0;
      for (std::size_t k = 0; k < 3; ++k) {
        if (mask & (1u << k))
          expert_weights_[k] *= 1.25; // MWU reward for expert that wanted to keep
        expert_weights_[k] = std::max(0.05, expert_weights_[k]);
        sum += expert_weights_[k];
      }
      for (double &w : expert_weights_)
        w /= sum;
      return;
    }
  }

  [[nodiscard]] double raw_gdsf_priority(const Entry &e) const noexcept {
    const double bytes = std::max<double>(1.0, double(e.accounted_bytes));
    return clock_aging_ + (double(e.frequency) * e.recovery_cost) / bytes;
  }

  [[nodiscard]] std::size_t
  select_victim_index(bool allow_pinned,
                      std::optional<std::size_t> restrict_slot = std::nullopt,
                      bool require_mtp = false) const {
    std::vector<std::size_t> cands;
    cands.reserve(entries_.size());
    for (std::size_t i = 0; i < entries_.size(); ++i) {
      if (!allow_pinned && entries_[i].pinned)
        continue;
      if (restrict_slot.has_value() && entries_[i].source_slot != *restrict_slot)
        continue;
      if (require_mtp && !entries_[i].mtp)
        continue;
      cands.push_back(i);
    }
    if (cands.empty())
      return entries_.size();
    if (cands.size() == 1)
      return cands.front();

    // Window-MRU nursery protection: keep the single most-recently-accessed
    // entry protected while >=1 older unpinned candidate can be reclaimed.
    const auto mru_it = std::max_element(
        cands.begin(), cands.end(), [&](std::size_t a, std::size_t b) {
          return entries_[a].last_access_tick < entries_[b].last_access_tick;
        });
    cands.erase(mru_it);
    if (cands.size() == 1)
      return cands.front();

    const std::size_t m = cands.size();
    const double denom = double(m - 1);

    // Rank candidates by each of the 3 WAKMan experts (0 = worst/evict, 1 = best/keep).
    auto compute_ranks = [&](auto &&less_fn) {
      std::vector<std::size_t> order(m);
      for (std::size_t r = 0; r < m; ++r)
        order[r] = r;
      std::sort(order.begin(), order.end(),
                [&](std::size_t a, std::size_t b) {
                  return less_fn(entries_[cands[a]], entries_[cands[b]]);
                });
      std::vector<double> ranks(m, 0.0);
      for (std::size_t pos = 0; pos < m; ++pos)
        ranks[order[pos]] = double(pos) / denom;
      return std::pair{std::move(ranks), cands[order.front()]};
    };

    auto [r_lru, v_lru] = compute_ranks([](const Entry &a, const Entry &b) {
      return a.last_access_tick < b.last_access_tick;
    });
    auto [r_lfu, v_lfu] = compute_ranks([](const Entry &a, const Entry &b) {
      if (a.frequency != b.frequency)
        return a.frequency < b.frequency;
      return a.last_access_tick < b.last_access_tick;
    });
    auto [r_gdsf, v_gdsf] = compute_ranks([this](const Entry &a, const Entry &b) {
      const double pa = raw_gdsf_priority(a);
      const double pb = raw_gdsf_priority(b);
      if (std::abs(pa - pb) > 1e-12)
        return pa < pb;
      return a.last_access_tick < b.last_access_tick;
    });
    (void)v_lru;
    (void)v_lfu;
    (void)v_gdsf;

    std::size_t best_local = 0;
    double lowest_blended = 1e30;
    for (std::size_t r = 0; r < m; ++r) {
      const double blended = expert_weights_[0] * r_lru[r] +
                             expert_weights_[1] * r_lfu[r] +
                             expert_weights_[2] * r_gdsf[r];
      if (blended < lowest_blended - 1e-12 ||
          (std::abs(blended - lowest_blended) <= 1e-12 &&
           entries_[cands[r]].last_access_tick <
               entries_[cands[best_local]].last_access_tick)) {
        lowest_blended = blended;
        best_local = r;
      }
    }
    return cands[best_local];
  }

  void record_wakman_eviction(std::size_t victim_idx) {
    if (entries_.size() <= 1 || victim_idx >= entries_.size())
      return;
    // Identify which single expert would NOT have picked victim_idx as its #1 eviction.
    std::size_t min_lru = victim_idx, min_lfu = victim_idx, min_gdsf = victim_idx;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
      if (entries_[i].pinned)
        continue;
      if (entries_[i].last_access_tick < entries_[min_lru].last_access_tick)
        min_lru = i;
      if (entries_[i].frequency < entries_[min_lfu].frequency ||
          (entries_[i].frequency == entries_[min_lfu].frequency &&
           entries_[i].last_access_tick < entries_[min_lfu].last_access_tick))
        min_lfu = i;
      if (raw_gdsf_priority(entries_[i]) < raw_gdsf_priority(entries_[min_gdsf]))
        min_gdsf = i;
    }
    std::uint8_t keep_mask = 0;
    if (min_lru != victim_idx)
      keep_mask |= 0x1;
    if (min_lfu != victim_idx)
      keep_mask |= 0x2;
    if (min_gdsf != victim_idx)
      keep_mask |= 0x4;
    ghosts_.push_back({entries_[victim_idx].prompt_hash, keep_mask});
    if (ghosts_.size() > 128)
      ghosts_.pop_front();
  }

  [[nodiscard]] std::uint64_t marginal_kv_bytes(
      const void *allocator, std::span<const std::uint32_t> pages,
      std::uint64_t page_bytes) const {
    if (page_bytes == 0ULL)
      return 0ULL;
    std::uint64_t bytes = 0;
    for (auto phys : pages) {
      if (!retained_kv_pages_.contains({allocator, phys}))
        bytes += page_bytes;
    }
    return bytes;
  }

  void retain_page_accounting(const void *allocator,
                              std::span<const std::uint32_t> pages,
                              std::uint64_t page_bytes) {
    if (page_bytes == 0ULL)
      return;
    for (auto phys : pages) {
      if (retained_kv_pages_[{allocator, phys}]++ == 0U)
        total_unique_kv_bytes_ += page_bytes;
    }
  }

  void release_page_accounting(const void *allocator,
                               std::span<const std::uint32_t> pages,
                               std::uint64_t page_bytes) {
    if (page_bytes == 0ULL)
      return;
    for (auto phys : pages) {
      const auto it = retained_kv_pages_.find({allocator, phys});
      if (it != retained_kv_pages_.end() && --it->second == 0U) {
        retained_kv_pages_.erase(it);
        total_unique_kv_bytes_ -= std::min(total_unique_kv_bytes_, page_bytes);
      }
    }
  }

  template <class SlotProgramFn>
  void release_entry(std::size_t index, SlotProgramFn &&program_at, bool update_aging) {
    auto &entry = entries_.at(index);
    if (update_aging) {
      clock_aging_ = std::max(clock_aging_, raw_gdsf_priority(entry));
    }
    if (total_non_kv_bytes_ >= entry.non_kv_bytes)
      total_non_kv_bytes_ -= entry.non_kv_bytes;
    else
      total_non_kv_bytes_ = 0;
    release_page_accounting(entry.kv_allocator_id, entry.continuation.kv_pages,
                            entry.kv_page_bytes_all_layers);
    if (entry.mtp) {
      release_page_accounting(entry.mtp->kv_allocator_id(),
                              entry.mtp->kv_pages, entry.mtp->kv_page_bytes);
    }
    QwenForward &src_prog = program_at(entry.source_slot);
    src_prog.release_uma(entry.continuation);
    entries_.erase(entries_.begin() + std::ptrdiff_t(index));
  }

  template <class SlotProgramFn>
  bool evict_best_victim(SlotProgramFn &&program_at, bool allow_pinned) {
    const std::size_t idx = select_victim_index(allow_pinned);
    if (idx >= entries_.size())
      return false;
    record_wakman_eviction(idx);
    release_entry(idx, program_at, true);
    return true;
  }

  std::uint64_t budget_bytes_ = 0;
  std::size_t max_unpinned_entries_ = 0;
  std::uint64_t next_id_ = 0;
  std::uint64_t last_inserted_id_ = 0;
  std::uint64_t clock_tick_ = 0;
  double clock_aging_ = 0.0;
  std::uint64_t total_non_kv_bytes_ = 0;
  std::uint64_t total_unique_kv_bytes_ = 0;
  std::map<std::pair<const void *, std::uint32_t>, std::uint32_t>
      retained_kv_pages_;
  // WAKMan initial policy weights: balanced CostGDSF + LRU + LFU.
  std::array<double, 3> expert_weights_{0.35, 0.25, 0.40};
  std::deque<GhostRecord> ghosts_;
  // KVSET Mattson stack-distance histogram.
  std::deque<std::uint64_t> mattson_stack_;
  std::array<std::uint64_t, 4> mrc_hist_{};
  std::uint64_t mrc_total_lookups_ = 0;
  std::vector<Entry> entries_;
};

} // namespace omlx::m5::detail
