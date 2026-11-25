// Copyright 2026 The NInfer Authors / omlx-backend-m5ultra-ninfer contributors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Adapted from ninfer/src/text/byte_span.h (Apache-2.0).
// Modifications: moved into namespace omlx::m5::serve.

#pragma once

#include <algorithm>
#include <cstddef>
#include <span>

namespace omlx::m5::serve {

struct ByteSpan {
  std::size_t begin = 0;
  std::size_t end = 0;
};

// Spans are nonempty, ordered and disjoint. A partial overlap also counts.
inline bool overlaps(std::span<const ByteSpan> spans, std::size_t begin,
                     std::size_t end) {
  const auto it = std::lower_bound(
      spans.begin(), spans.end(), begin,
      [](ByteSpan span, std::size_t pos) { return span.end <= pos; });
  return begin < end && it != spans.end() && it->begin < end;
}

} // namespace omlx::m5::serve
