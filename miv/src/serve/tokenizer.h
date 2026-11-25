// Copyright 2026 The NInfer Authors / omlx-backend-m5ultra-ninfer contributors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Adapted from ninfer/src/text/tokenizer.h (Apache-2.0).
// Modifications:
// - Moved into namespace omlx::m5::serve and updated include path.
// - Added BpeSplit::LetterDigit1 for Qwen3.8-27B-4bit compatibility.
// - Added optional lookup_token_id() helper for special/EOS token resolution.

#pragma once

#include "serve/byte_span.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace omlx::m5::serve {

struct EncodeOptions {
  bool parse_added_tokens = true;
  std::size_t max_tokens = std::numeric_limits<std::size_t>::max();
};

struct DecodeOptions {
  bool skip_special_tokens = false;
  std::vector<int> stop_token_ids;
};

struct AddedToken {
  int id = -1;
  std::string content;
  bool single_word = false;
  bool lstrip = false;
  bool rstrip = false;
  bool normalized = false;
  bool special = false;
};

struct DecodedTokenView {
  std::string_view bytes;
  bool special = false;
};

struct TokenizerResources {
  std::string_view tokenizer_json;
  // Optional: may be empty if only tokenizer.json is available.
  std::string_view tokenizer_config_json;
};

struct BpeMergeRule {
  int rank = 0;
  int result = -1;
};

struct BpeMergeEntry {
  std::uint64_t key = 0;
  int rank = 0;
  int result = -1;
};

enum class BpeSplit : std::uint8_t {
  LetterMarkDigit1,
  LetterDigit3,
  LetterDigit1,
};

struct BpePipeline {
  bool normalize_nfc = false;
  BpeSplit split = BpeSplit::LetterMarkDigit1;
  bool ignore_merges = false;
};

struct TokenBytesHash {
  using is_transparent = void;

  std::size_t operator()(std::string_view bytes) const noexcept {
    return std::hash<std::string_view>{}(bytes);
  }
};

using TokenBytesMap =
    std::unordered_map<std::string, int, TokenBytesHash, std::equal_to<>>;

class BpeMergeTable {
public:
  void reset(std::size_t rule_count) {
    std::size_t capacity = 1;
    while (capacity < rule_count * 2U) {
      capacity <<= 1U;
    }
    slots_.assign(capacity, BpeMergeEntry{});
    mask_ = capacity - 1U;
  }

  bool insert(std::uint64_t key, BpeMergeRule rule) {
    std::uint64_t index = mix(key) & mask_;
    while (slots_[index].result >= 0) {
      if (slots_[index].key == key) {
        return false;
      }
      index = (index + 1U) & mask_;
    }
    slots_[index] =
        BpeMergeEntry{.key = key, .rank = rule.rank, .result = rule.result};
    return true;
  }

  [[nodiscard]] const BpeMergeEntry *find(std::uint64_t key) const noexcept {
    std::uint64_t index = mix(key) & mask_;
    while (true) {
      const BpeMergeEntry &entry = slots_[index];
      if (entry.result < 0) {
        return nullptr;
      }
      if (entry.key == key) {
        return &entry;
      }
      index = (index + 1U) & mask_;
    }
  }

private:
  static std::uint64_t mix(std::uint64_t key) noexcept {
    key += 0x9e3779b97f4a7c15ULL;
    key = (key ^ (key >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    key = (key ^ (key >> 27U)) * 0x94d049bb133111ebULL;
    return key ^ (key >> 31U);
  }

  std::vector<BpeMergeEntry> slots_ = std::vector<BpeMergeEntry>(1);
  std::uint64_t mask_ = 0;
};

struct TokenBoundaryResult {
  std::optional<std::size_t> exact_frontier;
  std::size_t stable_frontier = 0;
};

struct BoundaryEncodedText {
  std::vector<int> input_ids;
  std::vector<TokenBoundaryResult> boundaries;
};

class Tokenizer {
public:
  explicit Tokenizer(TokenizerResources resources);

  std::vector<int> encode(std::string_view text,
                          EncodeOptions options = {}) const;
  BoundaryEncodedText
  encode_with_boundaries(std::string_view text,
                         std::span<const std::size_t> byte_boundaries,
                         EncodeOptions options = {},
                         std::span<const ByteSpan> literal_spans = {}) const;
  std::string decode(std::span<const int> ids,
                     DecodeOptions options = {}) const;
  [[nodiscard]] DecodedTokenView decoded_token(int id) const;
  [[nodiscard]] std::string_view
  decode_token_bytes(int id, bool skip_special_tokens = false) const;

  [[nodiscard]] std::optional<int>
  lookup_token_id(std::string_view token_text) const noexcept;

  [[nodiscard]] std::size_t vocab_size() const noexcept {
    return decoded_token_bytes_.size();
  }

  [[nodiscard]] bool is_special_token(int id) const noexcept;
  [[nodiscard]] bool is_valid_token(int id) const noexcept;
  [[nodiscard]] bool has_exact_token_domain(std::size_t size) const noexcept;

private:
  std::vector<std::string> decoded_token_bytes_;
  std::vector<bool> valid_token_ids_;
  std::vector<bool> special_token_ids_;
  BpePipeline pipeline_;
  TokenBytesMap whole_token_ids_;
  TokenBytesMap added_token_by_content_;
  BpeMergeTable bpe_merge_rules_;
  std::array<int, 256> byte_token_ids_{};
  std::vector<AddedToken> added_tokens_;
  std::array<std::vector<std::size_t>, 256> added_token_candidates_;
};

} // namespace omlx::m5::serve
