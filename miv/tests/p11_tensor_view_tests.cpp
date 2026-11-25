#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "omlx_m5/artifact.h"
#include "omlx_m5/p11_tensor_view.h"

using namespace omlx::m5;
namespace {
void require(bool value, const char *why) {
  if (!value) throw std::runtime_error(why);
}
template <class F> void rejected(F &&action, const char *why) {
  bool failed = false;
  try { action(); } catch (const std::invalid_argument &) { failed = true; }
  require(failed, why);
}
float decode(std::uint8_t code, std::uint8_t scale) {
  constexpr std::array<float, 8> magnitudes{0, .5f, 1, 1.5f, 2, 3, 4, 6};
  const float value = (code & 8) ? -magnitudes[code & 7] : magnitudes[code & 7];
  const float factor = std::bit_cast<float>(
      scale == 0 ? 0x00400000u : std::uint32_t(scale) << 23);
  return value * factor;
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 2 && argc != 3 && argc != 4)
      throw std::invalid_argument(
          "p11_tensor_view_tests ARTIFACT [OBJECT [--reject]]");
    auto artifact = ArtifactModel::open(argv[1], argc == 2);
    if (argc == 4) {
      if (std::string_view(argv[3]) != "--reject")
        throw std::invalid_argument("P11 unknown view test option");
      const auto &rejected_object = artifact.object(argv[2]);
      rejected([&] {
        static_cast<void>(native_mxfp4_weight_view(rejected_object, 0, 1));
      }, "P11 accepted actual non-MXFP4 artifact");
      std::cout << "P11 actual affine artifact rejected as native FP4\n";
      return 0;
    }
    const auto &object = artifact.object(
        argc == 3 ? argv[2] :
        "language_model.model.layers.0.linear_attn.out_proj");
    const auto view = native_mxfp4_weight_view(object, 1, 32);
    require(view.rows == object.shape[0] &&
                view.input_features == object.shape[1] &&
                view.padded_k == object.padded_k && view.row_start == 1 &&
                view.data_row_bytes == object.padded_k / 2 &&
                view.scale_row_bytes == object.padded_k / 32 &&
                view.data_bytes == object.shape[0] * view.data_row_bytes &&
                view.scale_bytes == object.shape[0] * view.scale_row_bytes,
            "P11 parent-backed FP4 view/seam geometry changed");
    // An isolated row-1 scale pointer is only 8-byte offset; keep the aligned
    // whole parent tensor and slice in the shader instead of hot repacking.
    require(view.scale_row_bytes % 128 != 0 && view.row_start == 1,
            "P11 row slice unexpectedly carries an aligned scale offset");
    const auto &data = artifact.plane(object, "data");
    const auto &scale = artifact.plane(object, "scale");
    auto codes = artifact.plane_bytes(object, data);
    auto scales = artifact.plane_bytes(object, scale);
    for (std::uint64_t row : {0ULL, object.shape[0] - 1}) {
      const auto reference = artifact.decode_weight_row(object, row, 0, 32);
      for (std::uint64_t col = 0; col < 32; ++col) {
        const auto packed = std::uint8_t(codes[row * view.data_row_bytes + col / 2]);
        const auto code = std::uint8_t((packed >> ((col % 2) * 4)) & 15);
        const auto scale_byte =
            std::uint8_t(scales[row * view.scale_row_bytes + col / 32]);
        require(std::abs(decode(code, scale_byte) - reference[col]) < 1e-6f,
                "P11 native plane interpretation differs from artifact math");
      }
    }
    const auto activation = native_mxfp4_activation_view(16, 256);
    require(activation.rows == 16 && activation.data_bytes == 2048 &&
                activation.scale_bytes == 128 && activation.block_k == 32,
            "P11 activation A4 plane geometry changed");
    rejected([&] { static_cast<void>(native_mxfp4_activation_view(1, 128)); },
             "P11 accepted unaligned FP4 outer stride");
    rejected([&] { static_cast<void>(native_mxfp4_weight_view(
                        object, std::uint32_t(object.shape[0] - 1), 2)); },
             "P11 accepted out-of-range parent slice");
    auto bad = object;
    bad.format = "oq_affine_q4_g64_bf16_v1";
    rejected([&] { static_cast<void>(native_mxfp4_weight_view(bad, 0, 1)); },
             "P11 mapped affine bias/scale to E8M0");
    bad = object;
    bad.planes[1].bytes--;
    rejected([&] { static_cast<void>(native_mxfp4_weight_view(bad, 0, 1)); },
             "P11 accepted short auxiliary plane");
    bad = object;
    bad.planes[1].offset++;
    rejected([&] { static_cast<void>(native_mxfp4_weight_view(bad, 0, 1)); },
             "P11 accepted unaligned auxiliary plane");
    bad = object;
    bad.planes.push_back(bad.planes[1]);
    bad.planes.back().kind = "bias";
    rejected([&] { static_cast<void>(native_mxfp4_weight_view(bad, 0, 1)); },
             "P11 silently dropped an extra bias plane");
    if (argc == 2) {
      const auto &short_row = artifact.object(
          "language_model.model.layers.0.linear_attn.in_proj_a");
      rejected([&] {
        static_cast<void>(native_mxfp4_weight_view(short_row, 0, 1));
      }, "P11 accepted K128 with 64-byte FP4 outer stride");
    }
    std::cout << "P11 FP4 parent view, math and negative boundaries passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "P11 tensor view failure: " << error.what() << '\n';
    return 1;
  }
}
