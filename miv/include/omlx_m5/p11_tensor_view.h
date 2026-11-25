#pragma once

#include <cstdint>

#include "omlx_m5/artifact.h"

namespace omlx::m5 {

// SDK27 FP4+E8M0 parent-buffer view. Extents are innermost-K first; a
// row-sliced consumer keeps the parent attachments and slices in the shader.
// This is a diagnostic contract, not an installed Qwen production route.
struct NativeMxfp4TensorView {
  std::uint32_t rows = 0;
  std::uint32_t input_features = 0;
  std::uint32_t padded_k = 0;
  std::uint32_t row_start = 0;
  std::uint32_t row_count = 0;
  std::uint32_t block_k = 32;
  std::uint64_t data_row_bytes = 0;
  std::uint64_t scale_row_bytes = 0;
  std::uint64_t data_bytes = 0;
  std::uint64_t scale_bytes = 0;
};

[[nodiscard]] NativeMxfp4TensorView
native_mxfp4_weight_view(const ArtifactObject &, std::uint32_t row_start,
                         std::uint32_t row_count);

[[nodiscard]] NativeMxfp4TensorView
native_mxfp4_activation_view(std::uint32_t tokens, std::uint32_t padded_k);

} // namespace omlx::m5
