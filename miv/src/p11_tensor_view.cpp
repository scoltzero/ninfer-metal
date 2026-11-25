#include "omlx_m5/p11_tensor_view.h"

#include <limits>
#include <stdexcept>
#include <string_view>

namespace omlx::m5 {
namespace {
std::uint32_t checked_u32(std::uint64_t value) {
  if (!value || value > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
    throw std::invalid_argument("P11 tensor extent exceeds 32-bit index");
  return std::uint32_t(value);
}
const ArtifactPlane &plane(const ArtifactObject &object, std::string_view kind) {
  for (const auto &candidate : object.planes)
    if (candidate.kind == kind) return candidate;
  throw std::invalid_argument("P11 MXFP4 data/scale plane missing");
}
NativeMxfp4TensorView geometry(std::uint32_t rows, std::uint32_t k,
                                std::uint32_t padded_k,
                                std::uint32_t row_start,
                                std::uint32_t row_count) {
  if (!rows || !k || k % 32 || padded_k < k || padded_k % 256 ||
      !row_count || row_start >= rows || row_count > rows - row_start)
    throw std::invalid_argument(
        "P11 FP4 view needs K32, padded K256 and an in-range parent slice");
  return {.rows = rows,
          .input_features = k,
          .padded_k = padded_k,
          .row_start = row_start,
          .row_count = row_count,
          .block_k = 32,
          .data_row_bytes = padded_k / 2,
          .scale_row_bytes = padded_k / 32,
          .data_bytes = std::uint64_t(rows) * (padded_k / 2),
          .scale_bytes = std::uint64_t(rows) * (padded_k / 32)};
}
} // namespace
NativeMxfp4TensorView native_mxfp4_weight_view(
    const ArtifactObject &object, std::uint32_t row_start,
    std::uint32_t row_count) {
  if (object.format != "mxfp4_e2m1_g32_e8m0_v1" ||
      object.layout != "mx_rowsplit_k128_v1" || object.shape.size() != 2)
    throw std::invalid_argument(
        "P11 native FP4 accepts MXFP4 E2M1/E8M0 only; affine bias/high-bit "
        "planes require a different route");
  const auto result = geometry(checked_u32(object.shape[0]),
                               checked_u32(object.shape[1]),
                               checked_u32(object.padded_k), row_start,
                               row_count);
  if (object.planes.size() != 2)
    throw std::invalid_argument("P11 native FP4 has unexpected planes");
  const auto &data = plane(object, "data"), &scales = plane(object, "scale");
  if (data.bytes != result.data_bytes || scales.bytes != result.scale_bytes ||
      data.alignment < 128 || scales.alignment < 128 ||
      data.offset % 128 || scales.offset % 128)
    throw std::invalid_argument("P11 native FP4 plane size/alignment mismatch");
  return result;
}
NativeMxfp4TensorView native_mxfp4_activation_view(
    std::uint32_t tokens, std::uint32_t padded_k) {
  return geometry(tokens, padded_k, padded_k, 0, tokens);
}
} // namespace omlx::m5
