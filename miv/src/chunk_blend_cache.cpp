#include "chunk_blend_cache.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace omlx::m5::detail {
namespace {

using U32 = std::uint32_t;
using U64 = std::uint64_t;

// Deterministic 256-entry SplitMix64 Gear table for content-defined chunking
// of unbroken spans exceeding `l_max`.
constexpr std::array<U64, 256> make_gear_table() noexcept {
  std::array<U64, 256> tbl{};
  U64 state = 0x243f6a8885a308d3ULL; // pi fractional hex constant
  for (std::size_t i = 0; i < 256; ++i) {
    state += 0x9e3779b97f4a7c15ULL;
    U64 z = state;
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    tbl[i] = z ^ (z >> 31U);
  }
  return tbl;
}

inline constexpr std::array<U64, 256> kGearTable = make_gear_table();

inline U32 gear_token_bucket(U32 tok) noexcept {
  U64 x = U64(tok) + 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
  return U32((x ^ (x >> 31U)) & 0xFFULL);
}

bool match_seq_at(std::span<const U32> tokens, U32 pos, U32 limit,
                  const std::vector<U32> &seq) noexcept {
  if (seq.empty() || pos + seq.size() > limit) {
    return false;
  }
  for (std::size_t i = 0; i < seq.size(); ++i) {
    if (tokens[pos + i] != seq[i]) {
      return false;
    }
  }
  return true;
}

bool match_seq_ending_at(std::span<const U32> tokens, U32 pos_end, U32 min_start,
                         const std::vector<U32> &seq) noexcept {
  if (seq.empty() || pos_end < min_start + seq.size()) {
    return false;
  }
  const U32 start = pos_end - U32(seq.size());
  for (std::size_t i = 0; i < seq.size(); ++i) {
    if (tokens[start + i] != seq[i]) {
      return false;
    }
  }
  return true;
}

U64 int8_g128_scale_count(U32 width) noexcept {
  return (U64(width) + 127ULL) / 128ULL;
}

} // namespace

U64 fnv1a_64_blend_bytes(const void *data, std::size_t bytes,
                         U64 seed) noexcept {
  const auto *p = static_cast<const unsigned char *>(data);
  U64 h = seed;
  for (std::size_t i = 0; i < bytes; ++i) {
    h ^= U64(p[i]);
    h *= 1099511628211ULL;
  }
  return h;
}

U64 fnv1a_64_blend_tokens(std::span<const U32> tokens, U64 seed) noexcept {
  U64 h = seed;
  for (const U32 tok : tokens) {
    h ^= U64(tok);
    h *= 1099511628211ULL;
  }
  return h;
}

U64 compute_blend_model_exec_hash(std::string_view artifact_digest,
                                  std::string_view kv_dtype,
                                  std::string_view prefill_linear_mode,
                                  std::string_view prefill_attention_mode,
                                  bool prefill_sparse_minference,
                                  std::string_view prefill_ane_mode) noexcept {
  U64 h = 14695981039346656037ULL;
  auto feed = [&](std::string_view s) {
    h = fnv1a_64_blend_bytes(s.data(), s.size(), h);
    const unsigned char sep = 0x1FU;
    h = fnv1a_64_blend_bytes(&sep, 1, h);
  };
  feed("blend_v1");
  feed(artifact_digest);
  feed(kv_dtype);
  feed(prefill_linear_mode);
  feed(prefill_attention_mode);
  feed(prefill_sparse_minference ? "minf=1" : "minf=0");
  if (prefill_ane_mode.empty() || prefill_ane_mode == "off") {
    feed("ane=off");
  } else {
    std::string ane_tag("ane=");
    ane_tag.append(prefill_ane_mode);
    feed(ane_tag);
  }
  return h;
}

std::size_t BlendChunkKeyHash::operator()(const BlendChunkKey &k) const noexcept {
  U64 h = k.model_exec_hash;
  h ^= k.token_content_hash + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
  const U64 packed = (U64(k.token_length) << 32U) ^ U64(k.first_token) ^
                     (U64(k.last_token) << 16U);
  h ^= packed + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
  return static_cast<std::size_t>(h);
}

BlendChunkKey make_blend_chunk_key(U64 model_exec_hash,
                                   std::span<const U32> tokens) noexcept {
  BlendChunkKey key;
  key.model_exec_hash = model_exec_hash;
  key.token_content_hash = fnv1a_64_blend_tokens(tokens);
  key.token_length = U32(tokens.size());
  if (!tokens.empty()) {
    key.first_token = tokens.front();
    key.last_token = tokens.back();
  }
  return key;
}

std::vector<BlendChunkSegment>
segment_blend_chunks(std::span<const U32> tokens, U32 prefix_tokens,
                     const BlendSegmentConfig &cfg) {
  std::vector<BlendChunkSegment> result;
  const U32 total = U32(tokens.size());
  if (prefix_tokens >= total) {
    return result;
  }

  const U32 l_min = std::max<U32>(1U, cfg.l_min);
  const U32 l_max = std::max<U32>(l_min, cfg.l_max);
  const U32 suffix_len = total - prefix_tokens;

  auto make_seg = [&](U32 start, U32 len, BlendSegmentKind kind) {
    BlendChunkSegment seg;
    seg.start = start;
    seg.length = len;
    seg.kind = kind;
    seg.boundary_window =
        (kind == BlendSegmentKind::CacheableChunk)
            ? std::min<U32>(cfg.boundary_window, len)
            : 0U;
    seg.content_hash = fnv1a_64_blend_tokens(tokens.subspan(start, len));
    return seg;
  };

  // Mandatory dense fallback if suffix is below blend activation threshold.
  const U32 dense_tail = std::min<U32>(cfg.dense_tail_tokens, suffix_len);
  const U32 cacheable_end = total - dense_tail;
  if (suffix_len < cfg.min_suffix_for_blend || cacheable_end <= prefix_tokens ||
      (cacheable_end - prefix_tokens) < l_min) {
    result.push_back(
        make_seg(prefix_tokens, suffix_len, BlendSegmentKind::UncachedDense));
    return result;
  }

  U32 cur = prefix_tokens;
  while (cur < cacheable_end) {
    const U32 rem = cacheable_end - cur;
    if (rem < l_min) {
      // Fold sub-L_min remainder into the trailing dense frontier so preceding
      // document chunk hashes stay 100% invariant to query/tail edits.
      break;
    }

    const U32 search_lo = cur + l_min;
    const U32 search_hi = std::min<U32>(cacheable_end, cur + l_max);
    U32 cut = 0U;

    // 1. Structural delimiter-anchored cut (prefix or suffix delimiter).
    if (!cfg.prefix_delimiters.empty() || !cfg.suffix_delimiters.empty()) {
      for (U32 pos = search_lo; pos <= search_hi; ++pos) {
        bool matched = false;
        if (pos < cacheable_end) {
          for (const auto &seq : cfg.prefix_delimiters) {
            if (match_seq_at(tokens, pos, cacheable_end, seq)) {
              cut = pos;
              matched = true;
              break;
            }
          }
        }
        if (matched) {
          break;
        }
        for (const auto &seq : cfg.suffix_delimiters) {
          if (match_seq_ending_at(tokens, pos, cur, seq)) {
            cut = pos;
            matched = true;
            break;
          }
        }
        if (matched) {
          break;
        }
      }
    }

    // 2. If no delimiter found: keep entire span if <= l_max, otherwise apply
    //    deterministic Gear rolling hash cut clamped to [gear_min, l_max].
    if (cut == 0U) {
      if (rem <= l_max) {
        cut = cacheable_end;
      } else {
        const U32 gear_lo =
            cur + std::clamp<U32>(cfg.gear_min_len, l_min, l_max);
        const U32 gear_hi = cur + l_max;
        const U32 min_win = std::max<U32>(1U, cfg.gear_window);
        U64 fp = 0ULL;
        const U64 mask = U64(cfg.gear_mask);
        for (U32 pos = cur; pos < gear_hi; ++pos) {
          fp = (fp << 1U) + kGearTable[gear_token_bucket(tokens[pos])];
          const U32 next_pos = pos + 1U;
          if (next_pos >= gear_lo && (next_pos - cur) >= min_win &&
              (fp & mask) == 0ULL) {
            cut = next_pos;
            break;
          }
        }
        if (cut == 0U) {
          cut = gear_hi;
        }
      }
    }

    result.push_back(
        make_seg(cur, cut - cur, BlendSegmentKind::CacheableChunk));
    cur = cut;
  }

  // Mandatory trailing `UNCACHED_DENSE` frontier (>= min(128, tail_len)).
  if (cur < total) {
    result.push_back(
        make_seg(cur, total - cur, BlendSegmentKind::UncachedDense));
  }
  return result;
}

U64 BlendChunkGeometry::attn_vec_bytes() const noexcept {
  return U64(head_dim / 2U) + sizeof(KvAffine4VecMeta);
}

U64 BlendChunkGeometry::attn_k_pre_bytes() const noexcept {
  return U64(num_full_attn_layers) * U64(token_length) * U64(kv_heads) *
         attn_vec_bytes();
}

U64 BlendChunkGeometry::attn_k_rotary_bf16_bytes() const noexcept {
  return U64(num_full_attn_layers) * U64(token_length) * U64(kv_heads) *
         U64(rotary_dim) * 2ULL;
}

U64 BlendChunkGeometry::attn_v_bytes() const noexcept {
  return attn_k_pre_bytes();
}

U64 BlendChunkGeometry::h3_res_i8_bytes() const noexcept {
  return U64(token_length) * U64(hidden_size);
}

U64 BlendChunkGeometry::h3_res_scale_bytes() const noexcept {
  return U64(token_length) * int8_g128_scale_count(hidden_size) * 2ULL;
}

U64 BlendChunkGeometry::q3_pre_i8_bytes() const noexcept {
  return U64(token_length) * U64(q_heads) * U64(head_dim);
}

U64 BlendChunkGeometry::q3_pre_scale_bytes() const noexcept {
  return U64(token_length) * int8_g128_scale_count(q_heads * head_dim) * 2ULL;
}

U64 BlendChunkGeometry::mix3_cached_i8_bytes() const noexcept {
  return q3_pre_i8_bytes();
}

U64 BlendChunkGeometry::mix3_cached_scale_bytes() const noexcept {
  return q3_pre_scale_bytes();
}

U64 BlendChunkGeometry::gdn_tail_qkv_bytes() const noexcept {
  const U64 hist_rows = (gdn_conv_kernel > 1U) ? U64(gdn_conv_kernel - 1U) : 0ULL;
  return U64(num_gdn_layers) * hist_rows * U64(gdn_conv_width) * 2ULL;
}

U64 BlendChunkGeometry::gdn_op_data_bytes() const noexcept {
  const U64 elem_bytes =
      (gdn_op_format == BlendGdnOperatorFormat::Fp8E4M3RowScale) ? 1ULL : 2ULL;
  return U64(num_gdn_layers) * U64(gdn_value_heads) * U64(gdn_value_dim) *
         U64(gdn_key_dim) * elem_bytes;
}

U64 BlendChunkGeometry::gdn_op_scale_bytes() const noexcept {
  return U64(num_gdn_layers) * U64(gdn_value_heads) * U64(gdn_value_dim) * 2ULL;
}

U64 BlendChunkGeometry::gdn_cum_decay_bytes() const noexcept {
  return U64(num_gdn_layers) * U64(token_length) * U64(gdn_value_heads) * 2ULL;
}

U64 BlendChunkGeometry::total_payload_bytes() const noexcept {
  return attn_k_pre_bytes() + attn_k_rotary_bf16_bytes() + attn_v_bytes() +
         h3_res_i8_bytes() + h3_res_scale_bytes() + q3_pre_i8_bytes() +
         q3_pre_scale_bytes() + mix3_cached_i8_bytes() +
         mix3_cached_scale_bytes() + gdn_tail_qkv_bytes() +
         2ULL * gdn_op_data_bytes() + 2ULL * gdn_op_scale_bytes() +
         gdn_cum_decay_bytes();
}

BlendChunkPayload allocate_blend_chunk_payload(const MetalContext &context,
                                               const BlendChunkGeometry &geom) {
  auto alloc_opt = [&](U64 bytes) -> MetalBuffer {
    if (bytes == 0ULL) {
      return {};
    }
    return context.make_buffer(bytes);
  };
  BlendChunkPayload p;
  p.attn_k_pre = alloc_opt(geom.attn_k_pre_bytes());
  p.attn_k_rotary_bf16 = alloc_opt(geom.attn_k_rotary_bf16_bytes());
  p.attn_v = alloc_opt(geom.attn_v_bytes());
  p.h3_res_i8 = alloc_opt(geom.h3_res_i8_bytes());
  p.h3_res_scales = alloc_opt(geom.h3_res_scale_bytes());
  p.q3_pre_i8 = alloc_opt(geom.q3_pre_i8_bytes());
  p.q3_pre_scales = alloc_opt(geom.q3_pre_scale_bytes());
  p.mix3_cached_i8 = alloc_opt(geom.mix3_cached_i8_bytes());
  p.mix3_cached_scales = alloc_opt(geom.mix3_cached_scale_bytes());
  p.gdn_tail_qkv = alloc_opt(geom.gdn_tail_qkv_bytes());
  p.gdn_m_data = alloc_opt(geom.gdn_op_data_bytes());
  p.gdn_m_row_scales = alloc_opt(geom.gdn_op_scale_bytes());
  p.gdn_b_data = alloc_opt(geom.gdn_op_data_bytes());
  p.gdn_b_row_scales = alloc_opt(geom.gdn_op_scale_bytes());
  p.gdn_cum_decay_f16 = alloc_opt(geom.gdn_cum_decay_bytes());
  return p;
}

ChunkBlendCache::ChunkBlendCache(U64 budget_bytes)
    : budget_bytes_(budget_bytes) {}

bool ChunkBlendCache::enabled() const noexcept {
  std::scoped_lock lock(mutex_);
  return budget_bytes_ > 0ULL;
}

U64 ChunkBlendCache::budget_bytes() const noexcept {
  std::scoped_lock lock(mutex_);
  return budget_bytes_;
}

void ChunkBlendCache::set_budget_bytes(U64 budget_bytes) {
  std::scoped_lock lock(mutex_);
  budget_bytes_ = budget_bytes;
  evict_to_locked(budget_bytes_);
}

U64 ChunkBlendCache::cached_bytes() const {
  std::scoped_lock lock(mutex_);
  return cached_bytes_;
}

std::size_t ChunkBlendCache::size() const {
  std::scoped_lock lock(mutex_);
  return lru_.size();
}

bool ChunkBlendCache::empty() const {
  std::scoped_lock lock(mutex_);
  return lru_.empty();
}

BlendCacheStats ChunkBlendCache::stats() const {
  std::scoped_lock lock(mutex_);
  return BlendCacheStats{
      .hits = hits_,
      .misses = misses_,
      .insertions = insertions_,
      .evictions = evictions_,
      .cached_bytes = cached_bytes_,
      .budget_bytes = budget_bytes_,
      .entry_count = lru_.size(),
  };
}

std::shared_ptr<const BlendChunkEntry>
ChunkBlendCache::lookup(U64 model_exec_hash, std::span<const U32> tokens) {
  if (tokens.empty()) {
    return nullptr;
  }
  const BlendChunkKey key = make_blend_chunk_key(model_exec_hash, tokens);
  std::scoped_lock lock(mutex_);
  if (budget_bytes_ == 0ULL || lru_.empty()) {
    ++misses_;
    return nullptr;
  }
  const auto [begin, end] = index_.equal_range(key);
  for (auto it = begin; it != end; ++it) {
    const auto list_it = it->second;
    const auto &entry = *list_it;
    // Exact token-span equality guard eliminates all hash-collision risk.
    if (entry->tokens.size() == tokens.size() &&
        std::equal(entry->tokens.begin(), entry->tokens.end(), tokens.begin())) {
      ++entry->hit_count;
      ++hits_;
      if (list_it != lru_.begin()) {
        lru_.splice(lru_.begin(), lru_, list_it);
      }
      return entry;
    }
  }
  ++misses_;
  return nullptr;
}

bool ChunkBlendCache::insert(std::shared_ptr<BlendChunkEntry> entry) {
  if (!entry || entry->tokens.empty()) {
    return false;
  }
  const BlendChunkKey expected_key =
      make_blend_chunk_key(entry->key.model_exec_hash, entry->tokens);
  entry->key = expected_key;
  if (entry->accounted_bytes == 0ULL) {
    const U64 payload_b = entry->geometry.total_payload_bytes();
    entry->accounted_bytes =
        payload_b + U64(entry->tokens.size()) * sizeof(U32) +
        sizeof(BlendChunkEntry);
  }

  std::scoped_lock lock(mutex_);
  if (budget_bytes_ == 0ULL || entry->accounted_bytes > budget_bytes_) {
    return false;
  }

  // Replace existing exact match in-place (not counted as budget eviction).
  const auto [begin, end] = index_.equal_range(entry->key);
  for (auto it = begin; it != end; ++it) {
    const auto &existing = *(it->second);
    if (existing->tokens.size() == entry->tokens.size() &&
        std::equal(existing->tokens.begin(), existing->tokens.end(),
                   entry->tokens.begin())) {
      erase_locked(it);
      break;
    }
  }

  const U64 allowed_before = budget_bytes_ - entry->accounted_bytes;
  if (cached_bytes_ > allowed_before) {
    evict_to_locked(allowed_before);
  }
  if (cached_bytes_ + entry->accounted_bytes > budget_bytes_) {
    return false;
  }

  cached_bytes_ += entry->accounted_bytes;
  lru_.push_front(std::move(entry));
  index_.emplace(lru_.front()->key, lru_.begin());
  ++insertions_;
  return true;
}

U64 ChunkBlendCache::trim_to_bytes(U64 target_bytes) {
  std::scoped_lock lock(mutex_);
  const U64 before = evictions_;
  evict_to_locked(target_bytes);
  return evictions_ - before;
}

U64 ChunkBlendCache::clear() {
  std::scoped_lock lock(mutex_);
  const U64 count = U64(lru_.size());
  index_.clear();
  lru_.clear();
  cached_bytes_ = 0ULL;
  return count;
}

BlendPromptPlan ChunkBlendCache::plan_prompt(U64 model_exec_hash,
                                             std::span<const U32> prompt,
                                             U32 prefix_tokens,
                                             const BlendSegmentConfig &cfg) {
  BlendPromptPlan plan;
  plan.prefix_tokens = std::min<U32>(prefix_tokens, U32(prompt.size()));
  plan.total_tokens = U32(prompt.size());
  const auto raw_segs = segment_blend_chunks(prompt, plan.prefix_tokens, cfg);
  plan.segments.reserve(raw_segs.size());
  const U32 suffix_len = plan.total_tokens - plan.prefix_tokens;
  plan.suffix_token_flags.assign(suffix_len, kBlendTokenMandatory);
  plan.suffix_prompt_positions.resize(suffix_len);
  for (U32 i = 0; i < suffix_len; ++i) {
    plan.suffix_prompt_positions[i] = plan.prefix_tokens + i;
  }

  for (const auto &seg : raw_segs) {
    BlendPlannedSegment planned;
    planned.segment = seg;
    if (seg.kind == BlendSegmentKind::CacheableChunk) {
      planned.hit_entry =
          lookup(model_exec_hash, prompt.subspan(seg.start, seg.length));
    }
    const U32 rel_start = seg.start - plan.prefix_tokens;
    if (planned.is_cache_hit()) {
      plan.hit_tokens += seg.length;
      const U32 w_bnd = std::min<U32>(seg.boundary_window, seg.length);
      plan.boundary_tokens += w_bnd;
      for (U32 t = 0; t < w_bnd; ++t) {
        plan.suffix_token_flags[rel_start + t] = kBlendTokenMandatory;
      }
      for (U32 t = w_bnd; t < seg.length; ++t) {
        plan.suffix_token_flags[rel_start + t] = kBlendTokenInteriorCandidate;
      }
    } else {
      plan.miss_tokens += seg.length;
      if (seg.kind == BlendSegmentKind::UncachedDense) {
        plan.dense_tail_tokens += seg.length;
      }
      for (U32 t = 0; t < seg.length; ++t) {
        plan.suffix_token_flags[rel_start + t] = kBlendTokenMandatory;
      }
    }
    plan.segments.push_back(std::move(planned));
  }
  return plan;
}

void ChunkBlendCache::erase_locked(IndexMap::iterator map_it) noexcept {
  const auto list_it = map_it->second;
  const U64 bytes = (*list_it)->accounted_bytes;
  cached_bytes_ = (cached_bytes_ >= bytes) ? (cached_bytes_ - bytes) : 0ULL;
  lru_.erase(list_it);
  index_.erase(map_it);
}

void ChunkBlendCache::evict_to_locked(U64 target_bytes) noexcept {
  while (cached_bytes_ > target_bytes && !lru_.empty()) {
    const auto victim_list_it = std::prev(lru_.end());
    const auto &victim = *victim_list_it;
    const auto [begin, end] = index_.equal_range(victim->key);
    for (auto it = begin; it != end; ++it) {
      if (it->second == victim_list_it) {
        erase_locked(it);
        ++evictions_;
        break;
      }
    }
  }
}

} // namespace omlx::m5::detail
