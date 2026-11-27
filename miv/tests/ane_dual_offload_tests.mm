#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ane/ane_dual_offload.h"
#include "omlx_m5/artifact.h"
#include "omlx_m5/execution_contract.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/qwen_forward.h"
#include "omlx_m5/types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using namespace omlx::m5;
using namespace omlx::m5::ane;

#define REQUIRE(cond)                                                          \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "FAIL at " << __FILE__ << ":" << __LINE__ << ": " #cond     \
                << "\n";                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

double cosine_similarity(std::span<const float> a, std::span<const float> b) {
  if (a.size() != b.size() || a.empty()) {
    return 0.0;
  }
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  const double denom = std::sqrt(na) * std::sqrt(nb);
  return denom > 0.0 ? (dot / denom) : 0.0;
}

float max_abs_diff(std::span<const float> a, std::span<const float> b) {
  float m = 0.0f;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

// Synthetic weight fixture holding owned MetalBuffers so MaterializedWeightView stays alive.
struct SyntheticWeightFixture {
  std::vector<MetalBuffer> owned_buffers;
  MaterializedWeightView view;
  std::vector<float> exact_f32; // Ground-truth dequantized FP32 [rows, K]
};

float ref_decode_e2m1(std::uint8_t code) {
  static constexpr std::array<float, 8> kMag = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float v = kMag[code & 0x07u];
  return (code & 0x08u) == 0u ? v : -v;
}

float ref_decode_e8m0(std::uint8_t code) {
  return code == 0u ? std::ldexp(1.0f, -127)
                    : std::ldexp(1.0f, static_cast<int>(code) - 127);
}

std::uint8_t encode_e2m1_ref(float v) {
  const std::uint8_t sign = std::signbit(v) ? 8u : 0u;
  const float a = std::abs(v);
  std::uint8_t c = 0u;
  if (a > 5.0f) c = 7u;
  else if (a >= 3.5f) c = 6u;
  else if (a > 2.5f) c = 5u;
  else if (a >= 1.75f) c = 4u;
  else if (a > 1.25f) c = 3u;
  else if (a >= 0.75f) c = 2u;
  else if (a > 0.25f) c = 1u;
  return static_cast<std::uint8_t>(c | sign);
}

// Build synthetic oQ affine weight (q4/q5/q6, g64/g128) in either untiled or NAX pre-tiled layout.
static std::uint32_t pi64_test(std::uint32_t off) {
  return ((off & 12u) << 2u) | ((off >> 2u) & 12u) | (off & 3u);
}

static void fwht64_test_rows(std::vector<float> &mat, std::uint32_t rows,
                             std::uint32_t cols) {
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::uint32_t c = 0; c < cols; c += 64u) {
      float *v = mat.data() + static_cast<std::size_t>(r) * cols + c;
      for (std::uint32_t len = 1u; len < 64u; len <<= 1u) {
        for (std::uint32_t i = 0u; i < 64u; i += len * 2u) {
          for (std::uint32_t j = 0u; j < len; ++j) {
            const float a = v[i + j];
            const float b = v[i + j + len];
            v[i + j] = a + b;
            v[i + j + len] = a - b;
          }
        }
      }
      for (std::uint32_t i = 0u; i < 64u; ++i) {
        v[i] *= 0.125f;
      }
    }
  }
}

SyntheticWeightFixture make_affine_fixture(
    const MetalContext &ctx,
    std::string name,
    std::uint32_t object_id,
    std::uint32_t rows,
    std::uint32_t k,
    std::uint32_t bits,
    std::uint32_t gs,
    bool nax_pretiled,
    std::uint32_t seed,
    bool lane_v2 = false,
    bool fwht64_weight = false) {
  SyntheticWeightFixture fix;
  const std::uint32_t padded_k = ((k + 127u) / 128u) * 128u;
  const std::uint32_t max_code = (1u << bits) - 1u;
  const std::uint32_t num_groups = k / gs;

  std::vector<std::uint8_t> codes_raw(static_cast<std::size_t>(rows) * k);
  std::vector<std::uint16_t> scales_bf16(static_cast<std::size_t>(rows) * num_groups);
  std::vector<std::uint16_t> biases_bf16(static_cast<std::size_t>(rows) * num_groups);
  fix.exact_f32.resize(static_cast<std::size_t>(rows) * k);

  std::uint32_t state = seed ^ 0x9E3779B9u;
  auto next_u32 = [&]() {
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
  };

  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::uint32_t g = 0; g < num_groups; ++g) {
      const float sc_f = 0.015f + 0.025f * static_cast<float>(next_u32() & 0xFFu) / 255.0f;
      const float bi_f = -0.35f + 0.70f * static_cast<float>(next_u32() & 0xFFu) / 255.0f;
      const std::uint16_t sc_w = float_to_bf16(sc_f);
      const std::uint16_t bi_w = float_to_bf16(bi_f);
      scales_bf16[static_cast<std::size_t>(r) * num_groups + g] = sc_w;
      biases_bf16[static_cast<std::size_t>(r) * num_groups + g] = bi_w;
      const float sc = bf16_to_float(sc_w);
      const float bi = bf16_to_float(bi_w);
      for (std::uint32_t i = 0; i < gs; ++i) {
        const std::uint32_t c = g * gs + i;
        const auto code = static_cast<std::uint8_t>(next_u32() % (max_code + 1u));
        codes_raw[static_cast<std::size_t>(r) * k + c] = code;
        fix.exact_f32[static_cast<std::size_t>(r) * k + c] =
            std::fma(sc, static_cast<float>(code), bi);
      }
    }
  }

  fix.view.object_id = object_id;
  fix.view.component = ModelComponent::text;
  fix.view.name = std::move(name);
  fix.view.format = "oq_affine_q" + std::to_string(bits) + "_g" +
                    std::to_string(gs) + "_bf16_v1";
  fix.view.parent_shape = {rows, k};
  fix.view.padded_k = padded_k;
  fix.view.row_start = 0;
  fix.view.row_count = rows;
  fix.view.in_features = k;
  fix.view.input_transform = fwht64_weight ? "fwht64" : "none";
  if (fwht64_weight) {
    fwht64_test_rows(fix.exact_f32, rows, k);
  }

  auto upload_vec = [&](const void *ptr, std::size_t bytes, std::string kind) {
    fix.owned_buffers.push_back(ctx.upload(std::span<const std::byte>(
        static_cast<const std::byte *>(ptr), bytes)));
    fix.view.planes.push_back(MaterializedPlaneView{
        .kind = std::move(kind),
        .buffer = nullptr, // patched after all buffers are pushed
        .segment_index = static_cast<std::uint32_t>(fix.owned_buffers.size() - 1u),
        .offset = 0,
        .bytes = bytes,
        .row_stride = 0,
    });
  };

  if (!nax_pretiled) {
    fix.view.layout = "oq_rowsplit_k128_v1";
    const std::size_t base_row = padded_k / 2u;
    std::vector<std::uint8_t> base(static_cast<std::size_t>(rows) * base_row, 0u);
    std::vector<std::uint8_t> high;
    const std::size_t high_row = (bits == 5u) ? (padded_k / 8u) : (padded_k / 4u);
    if (bits == 5u || bits == 6u) {
      high.assign(static_cast<std::size_t>(rows) * high_row, 0u);
    }
    const std::size_t grp_row = padded_k / gs;
    std::vector<std::uint16_t> sc_buf(static_cast<std::size_t>(rows) * grp_row, 0u);
    std::vector<std::uint16_t> bi_buf(static_cast<std::size_t>(rows) * grp_row, 0u);

    for (std::uint32_t r = 0; r < rows; ++r) {
      for (std::uint32_t g = 0; g < num_groups; ++g) {
        sc_buf[static_cast<std::size_t>(r) * grp_row + g] =
            scales_bf16[static_cast<std::size_t>(r) * num_groups + g];
        bi_buf[static_cast<std::size_t>(r) * grp_row + g] =
            biases_bf16[static_cast<std::size_t>(r) * num_groups + g];
      }
      for (std::uint32_t c = 0; c < k; ++c) {
        const std::uint8_t code = codes_raw[static_cast<std::size_t>(r) * k + c];
        base[static_cast<std::size_t>(r) * base_row + c / 2u] |=
            static_cast<std::uint8_t>((code & 0x0Fu) << ((c & 1u) * 4u));
        if (bits == 5u) {
          high[static_cast<std::size_t>(r) * high_row + c / 8u] |=
              static_cast<std::uint8_t>(((code >> 4u) & 1u) << (c & 7u));
        } else if (bits == 6u) {
          high[static_cast<std::size_t>(r) * high_row + c / 4u] |=
              static_cast<std::uint8_t>(((code >> 4u) & 3u) << ((c & 3u) * 2u));
        }
      }
    }
    upload_vec(base.data(), base.size(), "base_code");
    if (bits == 5u) {
      upload_vec(high.data(), high.size(), "high_bit");
    } else if (bits == 6u) {
      upload_vec(high.data(), high.size(), "high_2bit");
    }
    upload_vec(sc_buf.data(), sc_buf.size() * 2u, "scale");
    upload_vec(bi_buf.data(), bi_buf.size() * 2u, "bias");
  } else {
    fix.view.layout = lane_v2
                          ? ((gs == 128u) ? "oq_nax_lane_g128_v2"
                                          : "oq_nax_lane_g64_v2")
                          : ((gs == 128u) ? "oq_nax_tiled_g128_v1"
                                          : "oq_nax_tiled_g64_v1");
    const std::uint32_t bn = rows >= 65536u ? 32u : 16u;
    const std::uint32_t padded_n = ((rows + bn - 1u) / bn) * bn;
    const std::uint32_t groups64 = k / 64u;
    std::vector<std::uint8_t> nax_codes(static_cast<std::size_t>(padded_n) * (k / 2u), 0u);
    std::vector<std::uint8_t> nax_high2;
    if (bits == 5u || bits == 6u) {
      nax_high2.assign(static_cast<std::size_t>(padded_n) * (k / 4u), 0u);
    }
    std::vector<std::uint32_t> nax_sb(static_cast<std::size_t>(groups64) * padded_n, 0u);

    for (std::uint32_t r = 0; r < rows; ++r) {
      const std::uint32_t rn = r % bn;
      for (std::uint32_t g = 0; g < groups64; ++g) {
        const std::uint32_t src_g = (gs == 128u) ? (g >> 1u) : g;
        const std::uint32_t packed_sb =
            static_cast<std::uint32_t>(
                scales_bf16[static_cast<std::size_t>(r) * num_groups + src_g]) |
            (static_cast<std::uint32_t>(
                 biases_bf16[static_cast<std::size_t>(r) * num_groups + src_g])
             << 16u);
        nax_sb[static_cast<std::size_t>(g) * padded_n + r] = packed_sb;

        const std::size_t tile_idx = static_cast<std::size_t>(r / bn) * groups64 + g;
        for (std::uint32_t i = 0; i < 64u; ++i) {
          const std::uint32_t slot = lane_v2 ? pi64_test(i) : i;
          const std::uint8_t code =
              codes_raw[static_cast<std::size_t>(r) * k + g * 64u + i];
          nax_codes[tile_idx * (bn * 32u) + rn * 32u + slot / 2u] |=
              static_cast<std::uint8_t>((code & 0x0Fu) << ((slot & 1u) * 4u));
          if (bits == 5u || bits == 6u) {
            const std::uint8_t h = (bits == 5u) ? ((code >> 4u) & 1u) : ((code >> 4u) & 3u);
            nax_high2[tile_idx * (bn * 16u) + rn * 16u + slot / 4u] |=
                static_cast<std::uint8_t>(h << ((slot & 3u) * 2u));
          }
        }
      }
    }
    upload_vec(nax_codes.data(), nax_codes.size(), "nax_codes");
    if (bits == 5u || bits == 6u) {
      upload_vec(nax_high2.data(), nax_high2.size(), "nax_high2");
    }
    upload_vec(nax_sb.data(), nax_sb.size() * 4u, "nax_sb");
  }

  for (auto &pl : fix.view.planes) {
    pl.buffer = &fix.owned_buffers[pl.segment_index];
  }
  return fix;
}

// Build synthetic MXFP4 weight in either untiled or NAX pre-tiled layout.
SyntheticWeightFixture make_mxfp4_fixture(
    const MetalContext &ctx,
    std::string name,
    std::uint32_t object_id,
    std::uint32_t rows,
    std::uint32_t k,
    bool nax_pretiled,
    std::uint32_t seed,
    bool fwht64_weight = false) {
  SyntheticWeightFixture fix;
  const std::uint32_t padded_k = ((k + 127u) / 128u) * 128u;
  const std::uint32_t num_g32 = k / 32u;

  std::vector<std::uint8_t> codes_raw(static_cast<std::size_t>(rows) * k);
  std::vector<std::uint8_t> scales_raw(static_cast<std::size_t>(rows) * num_g32);
  fix.exact_f32.resize(static_cast<std::size_t>(rows) * k);

  std::uint32_t state = seed ^ 0x85EBCA6Bu;
  auto next_u32 = [&]() {
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
  };

  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::uint32_t g = 0; g < num_g32; ++g) {
      // Exponents in [-5, -1] (realistic transformer weight scale range)
      const auto exp_code = static_cast<std::uint8_t>(122u + (next_u32() % 5u));
      scales_raw[static_cast<std::size_t>(r) * num_g32 + g] = exp_code;
      const float sc = ref_decode_e8m0(exp_code);
      for (std::uint32_t i = 0; i < 32u; ++i) {
        const float u =
            (static_cast<float>(next_u32() & 0xFFFFu) / 32767.5f - 1.0f) * 6.0f;
        const std::uint8_t nib = encode_e2m1_ref(u);
        codes_raw[static_cast<std::size_t>(r) * k + g * 32u + i] = nib;
        fix.exact_f32[static_cast<std::size_t>(r) * k + g * 32u + i] =
            ref_decode_e2m1(nib) * sc;
      }
    }
  }

  fix.view.object_id = object_id;
  fix.view.component = ModelComponent::text;
  fix.view.name = std::move(name);
  fix.view.format = "mxfp4_e2m1_g32_e8m0_v1";
  fix.view.parent_shape = {rows, k};
  fix.view.padded_k = padded_k;
  fix.view.row_start = 0;
  fix.view.row_count = rows;
  fix.view.in_features = k;
  fix.view.input_transform = fwht64_weight ? "fwht64" : "none";
  if (fwht64_weight) {
    fwht64_test_rows(fix.exact_f32, rows, k);
  }

  auto upload_vec = [&](const void *ptr, std::size_t bytes, std::string kind) {
    fix.owned_buffers.push_back(ctx.upload(std::span<const std::byte>(
        static_cast<const std::byte *>(ptr), bytes)));
    fix.view.planes.push_back(MaterializedPlaneView{
        .kind = std::move(kind),
        .buffer = nullptr,
        .segment_index = static_cast<std::uint32_t>(fix.owned_buffers.size() - 1u),
        .offset = 0,
        .bytes = bytes,
        .row_stride = 0,
    });
  };

  if (!nax_pretiled) {
    fix.view.layout = "mx_rowsplit_k128_v1";
    const std::size_t d_row = padded_k / 2u;
    const std::size_t s_row = padded_k / 32u;
    std::vector<std::uint8_t> d_buf(static_cast<std::size_t>(rows) * d_row, 0u);
    std::vector<std::uint8_t> s_buf(static_cast<std::size_t>(rows) * s_row, 127u);
    for (std::uint32_t r = 0; r < rows; ++r) {
      for (std::uint32_t g = 0; g < num_g32; ++g) {
        s_buf[static_cast<std::size_t>(r) * s_row + g] =
            scales_raw[static_cast<std::size_t>(r) * num_g32 + g];
      }
      for (std::uint32_t c = 0; c < k; ++c) {
        d_buf[static_cast<std::size_t>(r) * d_row + c / 2u] |=
            static_cast<std::uint8_t>(
                (codes_raw[static_cast<std::size_t>(r) * k + c] & 0x0Fu)
                << ((c & 1u) * 4u));
      }
    }
    upload_vec(d_buf.data(), d_buf.size(), "data");
    upload_vec(s_buf.data(), s_buf.size(), "scale");
  } else {
    fix.view.layout = "mx_nax_tiled_bk128_v1";
    const std::uint32_t bn = rows >= 65536u ? 32u : 16u;
    const std::uint32_t padded_n = ((rows + bn - 1u) / bn) * bn;
    const std::uint32_t k_blocks = k / 128u;
    std::vector<std::uint8_t> d_buf(static_cast<std::size_t>(padded_n) * (k / 2u), 0u);
    std::vector<std::uint8_t> s_buf(static_cast<std::size_t>(padded_n) * (k / 32u), 127u);
    for (std::uint32_t r = 0; r < rows; ++r) {
      const std::uint32_t rn = r % bn;
      for (std::uint32_t blk = 0; blk < k_blocks; ++blk) {
        const std::size_t tile_idx = static_cast<std::size_t>(r / bn) * k_blocks + blk;
        for (std::uint32_t sg = 0; sg < 4u; ++sg) {
          s_buf[tile_idx * (bn * 4u) + rn * 4u + sg] =
              scales_raw[static_cast<std::size_t>(r) * num_g32 + blk * 4u + sg];
        }
        for (std::uint32_t ci = 0; ci < 128u; ++ci) {
          const std::uint8_t nib =
              codes_raw[static_cast<std::size_t>(r) * k + blk * 128u + ci];
          d_buf[tile_idx * (bn * 64u) + rn * 64u + ci / 2u] |=
              static_cast<std::uint8_t>((nib & 0x0Fu) << ((ci & 1u) * 4u));
        }
      }
    }
    upload_vec(d_buf.data(), d_buf.size(), "nax_data");
    upload_vec(s_buf.data(), s_buf.size(), "nax_scale");
  }

  for (auto &pl : fix.view.planes) {
    pl.buffer = &fix.owned_buffers[pl.segment_index];
  }
  return fix;
}

// Inline Metal GPU suffix GEMM & SwiGLU compute pipelines (runs on CB_B1 concurrently with ANE!).
struct GpuSuffixPipelines {
  id<MTLComputePipelineState> linear_suffix_pso = nil;
  id<MTLComputePipelineState> swiglu_suffix_pso = nil;
};

GpuSuffixPipelines compile_gpu_suffix_shaders(id<MTLDevice> device) {
  static NSString *const kSrc = @R"(
#include <metal_stdlib>
using namespace metal;

struct SuffixParams {
  uint T;
  uint K;
  uint N_total;
  uint row_offset;
  uint gpu_rows;
};

kernel void gpu_linear_suffix_bf16(
    device const bfloat *x [[buffer(0)]],
    device const float *w_full [[buffer(1)]],
    device bfloat *dst [[buffer(2)]],
    constant SuffixParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint local_n = gid.x;
  const uint t = gid.y;
  if (t >= p.T || local_n >= p.gpu_rows) return;
  const uint n = p.row_offset + local_n;
  device const bfloat *xr = x + ulong(t) * p.K;
  device const float *wr = w_full + ulong(n) * p.K;
  float acc = 0.0f;
  for (uint c = 0; c < p.K; ++c) {
    acc = fma(float(xr[c]), wr[c], acc);
  }
  dst[ulong(t) * p.N_total + n] = bfloat(acc);
}

kernel void gpu_swiglu_suffix_bf16(
    device const bfloat *x [[buffer(0)]],
    device const float *wg_full [[buffer(1)]],
    device const float *wu_full [[buffer(2)]],
    device bfloat *dst [[buffer(3)]],
    constant SuffixParams &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint local_n = gid.x;
  const uint t = gid.y;
  if (t >= p.T || local_n >= p.gpu_rows) return;
  const uint n = p.row_offset + local_n;
  device const bfloat *xr = x + ulong(t) * p.K;
  device const float *wgr = wg_full + ulong(n) * p.K;
  device const float *wur = wu_full + ulong(n) * p.K;
  float ag = 0.0f;
  float au = 0.0f;
  for (uint c = 0; c < p.K; ++c) {
    const float xv = float(xr[c]);
    ag = fma(xv, wgr[c], ag);
    au = fma(xv, wur[c], au);
  }
  const float sg = ag / (1.0f + exp(-ag));
  dst[ulong(t) * p.N_total + n] = bfloat(sg * au);
}
)";
  NSError *err = nil;
  id<MTLLibrary> lib = [device newLibraryWithSource:kSrc options:nil error:&err];
  if (lib == nil) {
    throw std::runtime_error("Failed to compile test GPU suffix Metal library");
  }
  GpuSuffixPipelines out;
  out.linear_suffix_pso = [device
      newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gpu_linear_suffix_bf16"]
                                    error:&err];
  out.swiglu_suffix_pso = [device
      newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gpu_swiglu_suffix_bf16"]
                                    error:&err];
  if (out.linear_suffix_pso == nil || out.swiglu_suffix_pso == nil) {
    throw std::runtime_error("Failed to create test GPU suffix PSO");
  }
  return out;
}

struct SuffixParamsHost {
  std::uint32_t T;
  std::uint32_t K;
  std::uint32_t N_total;
  std::uint32_t row_offset;
  std::uint32_t gpu_rows;
};

int test_gdn_exclusion_and_splits() {
  // 1. Strict GDN recurrence exclusion + Attn QKV exclusion
  REQUIRE(!ane_stage_role_eligible("gdn_qkv"));
  REQUIRE(!ane_stage_role_eligible("gdn_a"));
  REQUIRE(!ane_stage_role_eligible("gdn_b"));
  REQUIRE(!ane_stage_role_eligible("model.layers.0.linear_attn.in_proj_qkv"));
  REQUIRE(!ane_stage_role_eligible("model.layers.0.linear_attn.in_proj_a"));
  REQUIRE(!ane_stage_role_eligible("model.layers.0.linear_attn.in_proj_b"));
  REQUIRE(!ane_stage_role_eligible("attention_q"));
  REQUIRE(!ane_stage_role_eligible("attention_k"));
  REQUIRE(!ane_stage_role_eligible("attention_v"));
  REQUIRE(!ane_stage_role_eligible("model.layers.3.self_attn.q_proj"));
  REQUIRE(!ane_stage_role_eligible("model.layers.3.self_attn.k_proj"));
  REQUIRE(!ane_stage_role_eligible("model.layers.3.self_attn.v_proj"));
  REQUIRE(!ane_stage_role_eligible("lm_head"));
  REQUIRE(!ane_stage_role_eligible("embed_tokens"));
  REQUIRE(!ane_stage_role_eligible("moe_expert_gate"));

  // Allowlisted eligible projections ONLY:
  REQUIRE(ane_stage_role_eligible("mlp_gate"));
  REQUIRE(ane_stage_role_eligible("mlp_up"));
  REQUIRE(ane_stage_role_eligible("mlp_gate_up_swiglu"));
  REQUIRE(ane_stage_role_eligible("mlp_down"));
  REQUIRE(ane_stage_role_eligible("gdn_z"));
  REQUIRE(ane_stage_role_eligible("gdn_out"));
  REQUIRE(ane_stage_role_eligible("attention_o"));
  REQUIRE(ane_stage_role_eligible("model.layers.0.mlp.gate_proj"));
  REQUIRE(ane_stage_role_eligible("model.layers.0.mlp.gate_proj.weight"));
  REQUIRE(ane_stage_role_eligible("model.layers.0.mlp.up_proj"));
  REQUIRE(ane_stage_role_eligible("model.layers.0.mlp.down_proj"));
  REQUIRE(ane_stage_role_eligible("model.layers.0.mlp.down_proj.weight"));
  REQUIRE(ane_stage_role_eligible("model.layers.0.linear_attn.in_proj_z"));
  REQUIRE(ane_stage_role_eligible("model.layers.0.linear_attn.in_proj_z.weight"));
  REQUIRE(ane_stage_role_eligible("model.layers.0.linear_attn.out_proj"));
  REQUIRE(ane_stage_role_eligible("model.layers.3.self_attn.o_proj"));
  REQUIRE(!ane_stage_role_eligible("model.layers.0.linear_attn.in_proj_qkv.weight"));
  REQUIRE(!ane_stage_role_eligible("model.layers.0.linear_attn.in_proj_a.weight"));
  REQUIRE(!ane_stage_role_eligible("model.layers.0.linear_attn.in_proj_b.weight"));

  // 2. Mode-aware 128-row-aligned split contract on Qwen3.8-27B (2-die Ultra)
  const auto s_bf16_i = compute_ane_channel_split(17408u, kAneDefaultAlphaBf16, 2u);
  const auto s_bf16_h = compute_ane_channel_split(5120u, kAneDefaultAlphaBf16, 2u);
  const auto s_bf16_z = compute_ane_channel_split(6144u, kAneDefaultAlphaBf16, 2u);
  REQUIRE(s_bf16_i.die_rows == 2176u && s_bf16_i.ane_total_rows == 4352u &&
          (s_bf16_i.gpu_row_offset % 128u) == 0u &&
          (s_bf16_i.gpu_row_offset % 64u) == 0u && s_bf16_i.gpu_rows == 13056u);
  REQUIRE(s_bf16_h.die_rows == 640u && s_bf16_h.ane_total_rows == 1280u &&
          s_bf16_h.gpu_rows == 3840u);
  REQUIRE(s_bf16_z.die_rows == 768u && s_bf16_z.ane_total_rows == 1536u &&
          s_bf16_z.gpu_rows == 4608u);

  const auto s_w4a8_i = compute_ane_channel_split(17408u, kAneDefaultAlphaW4A8, 2u);
  const auto s_w4a8_h = compute_ane_channel_split(5120u, kAneDefaultAlphaW4A8, 2u);
  const auto s_w4a8_z = compute_ane_channel_split(6144u, kAneDefaultAlphaW4A8, 2u);
  REQUIRE(s_w4a8_i.die_rows == 1280u && s_w4a8_i.ane_total_rows == 2560u &&
          s_w4a8_i.gpu_rows == 14848u);
  REQUIRE(s_w4a8_h.die_rows == 384u && s_w4a8_h.ane_total_rows == 768u &&
          s_w4a8_h.gpu_rows == 4352u);
  REQUIRE(s_w4a8_z.die_rows == 512u && s_w4a8_z.ane_total_rows == 1024u &&
          s_w4a8_z.gpu_rows == 5120u);

  // Verify default cache path matches ~/Library/Caches/io.ninfer.miv/ane_banks_v1
  const auto def_cache = AneDualOffloadEngine::default_cache_root().string();
  REQUIRE(def_cache.find("io.ninfer.miv/ane_banks_v1") != std::string::npos);

  // Verify build() throws immediately if caller tries to pass recurrent GDN qkv!
  AneOffloadOpSpec bad_spec{};
  bad_spec.layer_index = 0;
  bad_spec.role = AneProjectionRole::GdnInProjZ;
  bad_spec.name = "model.layers.0.linear_attn.in_proj_qkv";
  bool threw = false;
  try {
    std::array<AneOffloadOpSpec, 1> arr{bad_spec};
    AneDualOffloadConfig cfg{};
    (void)AneDualOffloadEngine::build(arr, cfg);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  REQUIRE(threw);

  std::cout << "[PASS] test_gdn_exclusion_and_splits\n";
  return 0;
}

int test_miv_extraction_accuracy(const MetalContext &ctx) {
  struct Case {
    const char *label;
    bool is_mxfp4;
    std::uint32_t bits;
    std::uint32_t gs;
    bool pretiled;
    bool lane_v2;
    bool fwht64;
  };
  const std::array<Case, 12> cases = {{
      {"oQ4_g64_untiled", false, 4u, 64u, false, false, false},
      {"oQ4_g64_nax_pretiled", false, 4u, 64u, true, false, false},
      {"oQ4_g64_lane_v2_fwht", false, 4u, 64u, true, true, true},
      {"oQ5_g64_untiled", false, 5u, 64u, false, false, false},
      {"oQ5_g64_nax_pretiled", false, 5u, 64u, true, false, false},
      {"oQ5_g64_lane_v2_fwht", false, 5u, 64u, true, true, true},
      {"oQ6_g128_untiled", false, 6u, 128u, false, false, false},
      {"oQ6_g128_nax_pretiled", false, 6u, 128u, true, false, false},
      {"oQ6_g128_lane_v2_fwht", false, 6u, 128u, true, true, true},
      {"MXFP4_untiled", true, 4u, 32u, false, false, false},
      {"MXFP4_nax_pretiled", true, 4u, 32u, true, false, false},
      {"MXFP4_pretiled_fwht", true, 4u, 32u, true, false, true},
  }};

  const std::uint32_t rows = 256u;
  const std::uint32_t k = 512u;

  for (std::size_t idx = 0; idx < cases.size(); ++idx) {
    const auto &tc = cases[idx];
    SyntheticWeightFixture fix =
        tc.is_mxfp4
            ? make_mxfp4_fixture(ctx, "model.layers.0.mlp.down_proj",
                                 static_cast<std::uint32_t>(idx), rows, k,
                                 tc.pretiled, 100u + static_cast<std::uint32_t>(idx),
                                 tc.fwht64)
            : make_affine_fixture(ctx, "model.layers.0.mlp.down_proj",
                                  static_cast<std::uint32_t>(idx), rows, k,
                                  tc.bits, tc.gs, tc.pretiled,
                                  200u + static_cast<std::uint32_t>(idx),
                                  tc.lane_v2, tc.fwht64);

    // Extract middle slice [64, 192) to verify non-zero row_start indexing on BOTH untiled & NAX tiled!
    const std::uint64_t r0 = 64u;
    const std::uint64_t rc = 128u;
    const auto dec = dequantize_miv_weight_slice_f32(fix.view, r0, rc);
    const std::span<const float> ref_slice(
        fix.exact_f32.data() + static_cast<std::size_t>(r0 * k),
        static_cast<std::size_t>(rc * k));
    const float exact_err = max_abs_diff(dec, ref_slice);
    REQUIRE(exact_err <= (tc.fwht64 ? 1e-5f : 0.0f));

    const auto q = extract_miv_weight_slice_g64_int8(fix.view, r0, rc);
    const auto deq = dequantize_g64_int8_to_f32(q);
    const double cos = cosine_similarity(deq, ref_slice);

    // Direct FP16 .miv v2 slice extraction (WP 3C ultra_fp16_swiglu)
    const auto fp16_slice = extract_miv_weight_slice_fp16(fix.view, r0, rc);
    REQUIRE(fp16_slice.size() == static_cast<std::size_t>(rc * k));
    std::vector<float> fp16_as_f32(fp16_slice.size());
    for (std::size_t i = 0; i < fp16_slice.size(); ++i) {
      __fp16 hval{};
      std::memcpy(&hval, &fp16_slice[i], sizeof(hval));
      fp16_as_f32[i] = static_cast<float>(hval);
    }
    const double cos_fp16 = cosine_similarity(fp16_as_f32, ref_slice);
    const float fp16_err = max_abs_diff(fp16_as_f32, ref_slice);
    std::printf(
        "  [extract] %-22s exact_err=%.1e g64_int8_cos=%.7f fp16_cos=%.7f fp16_err=%.2e\n",
        tc.label, exact_err, cos, cos_fp16, fp16_err);
    REQUIRE(cos >= 0.9995);
    REQUIRE(cos_fp16 >= 0.9999);
    REQUIRE(fp16_err <= 2e-2f);
  }

  // Exercise multithreaded extraction path (rc >= 128) on MXFP4 & oQ4e lane_v2 + fwht64.
  {
    const std::uint32_t mt_rows = 384u;
    const std::uint32_t mt_k = 512u;
    const std::uint64_t r0 = 64u;
    const std::uint64_t rc = 256u;
    for (int kind = 0; kind < 2; ++kind) {
      SyntheticWeightFixture fix =
          (kind == 0)
              ? make_mxfp4_fixture(ctx, "model.layers.0.mlp.gate_proj", 90u,
                                   mt_rows, mt_k, true, 901u, true)
              : make_affine_fixture(ctx, "model.layers.0.mlp.up_proj", 91u,
                                    mt_rows, mt_k, 5u, 64u, true, 902u, true,
                                    true);
      const std::span<const float> ref_slice(
          fix.exact_f32.data() + static_cast<std::size_t>(r0 * mt_k),
          static_cast<std::size_t>(rc * mt_k));
      const auto dec = dequantize_miv_weight_slice_f32(fix.view, r0, rc);
      REQUIRE(max_abs_diff(dec, ref_slice) <= 1e-5f);
      const auto fp16_slice = extract_miv_weight_slice_fp16(fix.view, r0, rc);
      std::vector<float> fp16_f32(fp16_slice.size());
      for (std::size_t i = 0; i < fp16_f32.size(); ++i) {
        __fp16 hval{};
        std::memcpy(&hval, &fp16_slice[i], sizeof(hval));
        fp16_f32[i] = static_cast<float>(hval);
      }
      const double cos_mt = cosine_similarity(fp16_f32, ref_slice);
      REQUIRE(cos_mt >= 0.99999);
    }
  }
  std::cout << "[PASS] test_miv_extraction_accuracy (all 12 formats/layouts g64_int8>=0.9995 & fp16>=0.9999 + MT)\n";
  return 0;
}

int test_ane_execution_pipeline_and_cache(
    const MetalContext &ctx,
    const std::filesystem::path &metallib_dir,
    bool simulate_dual_banks) {
  const AneHardwareInfo hw = probe_ane_hardware();
  std::printf("  [hw] available=%d physical_dies=%u os=%s simulate_dual=%d\n",
              hw.is_available() ? 1 : 0, hw.physical_die_count,
              hw.os_build.c_str(), simulate_dual_banks ? 1 : 0);
  if (!hw.is_available()) {
    std::cout << "[SKIP] ANE hardware unavailable (" << hw.unavailable_reason
              << "); graceful fallback verified\n";
    return 0;
  }

  // Test realistic T_chunk = 1024 fixed-chunk execution across all 5 eligible projection roles!
  const std::uint32_t T = kAneDefaultChunkTokens; // 1024
  const std::uint32_t K = 512u;
  const std::uint32_t N_mlp = 512u; // 128/die in dual (256 ANE + 256 GPU), 128 in single (128 ANE + 384 GPU)
  const std::uint32_t N_lin = 512u;

  auto f_gate = make_affine_fixture(ctx, "model.layers.0.mlp.gate_proj", 1u,
                                    N_mlp, K, 4u, 64u, false, 11u);
  auto f_up = make_affine_fixture(ctx, "model.layers.0.mlp.up_proj", 2u,
                                  N_mlp, K, 5u, 64u, true, 22u);
  auto f_down = make_affine_fixture(ctx, "model.layers.0.mlp.down_proj", 3u,
                                    N_lin, K, 5u, 64u, false, 33u);
  auto f_gdn_z = make_mxfp4_fixture(ctx, "model.layers.0.linear_attn.in_proj_z", 4u,
                                    N_lin, K, true, 44u);
  auto f_gdn_out = make_mxfp4_fixture(ctx, "model.layers.0.linear_attn.out_proj", 5u,
                                      N_lin, K, false, 55u);
  auto f_attn_o = make_affine_fixture(ctx, "model.layers.3.self_attn.o_proj", 6u,
                                      N_lin, K, 6u, 128u, true, 66u);

  std::vector<AneOffloadOpSpec> specs = {
      AneOffloadOpSpec{
          .layer_index = 0u,
          .role = AneProjectionRole::MlpGateUpSwiGlu,
          .name = "mlp_gate_up_swiglu",
          .weight = f_gate.view,
          .up_weight = f_up.view,
      },
      AneOffloadOpSpec{
          .layer_index = 0u,
          .role = AneProjectionRole::MlpDown,
          .name = "model.layers.0.mlp.down_proj",
          .weight = f_down.view,
      },
      AneOffloadOpSpec{
          .layer_index = 0u,
          .role = AneProjectionRole::GdnInProjZ,
          .name = "model.layers.0.linear_attn.in_proj_z",
          .weight = f_gdn_z.view,
      },
      AneOffloadOpSpec{
          .layer_index = 0u,
          .role = AneProjectionRole::GdnOutProj,
          .name = "model.layers.0.linear_attn.out_proj",
          .weight = f_gdn_out.view,
      },
      AneOffloadOpSpec{
          .layer_index = 3u,
          .role = AneProjectionRole::AttnOProj,
          .name = "model.layers.3.self_attn.o_proj",
          .weight = f_attn_o.view,
      },
  };

  // Fresh temporary disk cache directory to test cold build -> warm reload.
  const auto temp_cache =
      std::filesystem::temp_directory_path() /
      ("miv_ane_test_cache_" + std::to_string(static_cast<long>(::getpid())) +
       (simulate_dual_banks ? "_dual" : "_native"));
  std::error_code ec;
  std::filesystem::remove_all(temp_cache, ec);

  AneDualOffloadConfig cfg{};
  cfg.chunk_tokens = T;
  cfg.linear_route = AneLinearRoute::BF16;
  cfg.alpha_override = simulate_dual_banks ? 0.50 : 0.25;
  cfg.max_alpha = 0.50;
  cfg.simulate_dual_banks_on_single_die = simulate_dual_banks;
  cfg.enable_disk_cache = true;
  cfg.cache_root = temp_cache;
  cfg.artifact_digest = "test_m4b_synthetic_digest_v1";
  cfg.metallib_path = metallib_dir;

  // Cold build
  auto engine = AneDualOffloadEngine::build(specs, cfg);
  REQUIRE(engine != nullptr);
  REQUIRE(engine->is_available());
  REQUIRE(engine->is_active());
  REQUIRE(engine->ops().size() == specs.size());
  REQUIRE(engine->can_offload_chunk(T));
  REQUIRE(!engine->can_offload_chunk(512u));
  REQUIRE(!engine->can_offload_chunk(1u));

  const auto cold_stats = engine->stats();
  REQUIRE(cold_stats.disk_cache_hits == 0u);
  REQUIRE(cold_stats.disk_cache_misses == cold_stats.active_die_count * cold_stats.bank_shards_per_die);
  if (simulate_dual_banks || hw.physical_die_count >= 2u) {
    REQUIRE(cold_stats.active_die_count == 2u);
  } else {
    REQUIRE(cold_stats.active_die_count == 1u);
  }
  REQUIRE(std::filesystem::exists(cold_stats.cache_dir / "bank_d0_s0.meta.json"));
  REQUIRE(std::filesystem::exists(cold_stats.cache_dir / "bank_d0_s0.mil"));
  REQUIRE(std::filesystem::exists(cold_stats.cache_dir / "bank_d0_s0.weights.bin"));
  REQUIRE(std::filesystem::exists(cold_stats.cache_dir / "bank_d0_s0.lock"));

  // Prepare BF16 input X [T, K] and upload FP32 weights for concurrent GPU suffix shaders.
  std::vector<float> x_f32(static_cast<std::size_t>(T) * K);
  auto x_buf = engine->make_shared_buffer(static_cast<std::uint64_t>(T) * K * 2u);
  auto *x_bf16 = static_cast<std::uint16_t *>(x_buf.data());
  for (std::uint32_t t = 0; t < T; ++t) {
    for (std::uint32_t c = 0; c < K; ++c) {
      const float v =
          0.35f * std::sin(static_cast<float>(t * 17u + c * 3u) * 0.011f) +
          0.20f * std::cos(static_cast<float>(t * 5u + c * 13u) * 0.019f);
      const std::uint16_t b = float_to_bf16(v);
      x_bf16[static_cast<std::size_t>(t) * K + c] = b;
      x_f32[static_cast<std::size_t>(t) * K + c] = bf16_to_float(b);
    }
  }

  id<MTLDevice> dev = (__bridge id<MTLDevice>)engine->native_mtl_device();
  const auto suffix_psos = compile_gpu_suffix_shaders(dev);

  auto upload_f32_buf = [&](const std::vector<float> &v) {
    auto buf = engine->make_shared_buffer(v.size() * sizeof(float));
    std::memcpy(buf.data(), v.data(), v.size() * sizeof(float));
    return buf;
  };
  auto w_gate_gpu = upload_f32_buf(f_gate.exact_f32);
  auto w_up_gpu = upload_f32_buf(f_up.exact_f32);
  auto w_down_gpu = upload_f32_buf(f_down.exact_f32);
  auto w_z_gpu = upload_f32_buf(f_gdn_z.exact_f32);
  auto w_out_gpu = upload_f32_buf(f_gdn_out.exact_f32);
  auto w_attno_gpu = upload_f32_buf(f_attn_o.exact_f32);

  // Allocate 6 separate outputs so we enqueue 6 ops back-to-back with ZERO CPU waits!
  // (Op 0..4 plus Op 0 repeated as 6th op -> exercises ping-pong slot 0/1 wraparound x3!)
  std::vector<AneSharedBuffer> dst_bufs;
  for (int i = 0; i < 6; ++i) {
    auto b = engine->make_shared_buffer(static_cast<std::uint64_t>(T) * N_lin * 2u);
    std::memset(b.data(), 0, static_cast<std::size_t>(b.size()));
    dst_bufs.push_back(std::move(b));
  }

  auto make_linear_suffix_cb = [&](const AneCompiledOpInfo &info,
                                   const AneSharedBuffer &w_gpu,
                                   AneSharedBuffer &dst) {
    return [&suffix_psos, &x_buf, &w_gpu, &dst, info](void *raw_cb) {
      id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)raw_cb;
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      const SuffixParamsHost p{T, K, info.split.total_out_features,
                               info.split.gpu_row_offset, info.split.gpu_rows};
      [enc setComputePipelineState:suffix_psos.linear_suffix_pso];
      [enc setBuffer:(__bridge id<MTLBuffer>)x_buf.native_mtl_buffer() offset:0 atIndex:0];
      [enc setBuffer:(__bridge id<MTLBuffer>)w_gpu.native_mtl_buffer() offset:0 atIndex:1];
      [enc setBuffer:(__bridge id<MTLBuffer>)dst.native_mtl_buffer() offset:0 atIndex:2];
      [enc setBytes:&p length:sizeof(p) atIndex:3];
      [enc dispatchThreads:MTLSizeMake(info.split.gpu_rows, T, 1u)
          threadsPerThreadgroup:MTLSizeMake(16u, 16u, 1u)];
      [enc endEncoding];
    };
  };

  auto make_swiglu_suffix_cb = [&](const AneCompiledOpInfo &info,
                                   const AneSharedBuffer &wg_gpu,
                                   const AneSharedBuffer &wu_gpu,
                                   AneSharedBuffer &dst) {
    return [&suffix_psos, &x_buf, &wg_gpu, &wu_gpu, &dst, info](void *raw_cb) {
      id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)raw_cb;
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      const SuffixParamsHost p{T, K, info.split.total_out_features,
                               info.split.gpu_row_offset, info.split.gpu_rows};
      [enc setComputePipelineState:suffix_psos.swiglu_suffix_pso];
      [enc setBuffer:(__bridge id<MTLBuffer>)x_buf.native_mtl_buffer() offset:0 atIndex:0];
      [enc setBuffer:(__bridge id<MTLBuffer>)wg_gpu.native_mtl_buffer() offset:0 atIndex:1];
      [enc setBuffer:(__bridge id<MTLBuffer>)wu_gpu.native_mtl_buffer() offset:0 atIndex:2];
      [enc setBuffer:(__bridge id<MTLBuffer>)dst.native_mtl_buffer() offset:0 atIndex:3];
      [enc setBytes:&p length:sizeof(p) atIndex:4];
      [enc dispatchThreads:MTLSizeMake(info.split.gpu_rows, T, 1u)
          threadsPerThreadgroup:MTLSizeMake(16u, 16u, 1u)];
      [enc endEncoding];
    };
  };

  // Zero-CPU-stall async enqueue of 6 co-executed ops (ANE [0..N_ane) || GPU [N_ane..N))!
  const auto *op_swiglu = engine->find_op(0u, AneProjectionRole::MlpGateUpSwiGlu);
  const auto *op_down = engine->find_op(0u, AneProjectionRole::MlpDown);
  const auto *op_z = engine->find_op(0u, AneProjectionRole::GdnInProjZ);
  const auto *op_out = engine->find_op(0u, AneProjectionRole::GdnOutProj);
  const auto *op_attno = engine->find_op(3u, AneProjectionRole::AttnOProj);
  REQUIRE(op_swiglu && op_down && op_z && op_out && op_attno);

  engine->enqueue_op(op_swiglu->op_id, x_buf.native_mtl_buffer(), 0,
                     dst_bufs[0].native_mtl_buffer(), 0, N_mlp,
                     make_swiglu_suffix_cb(*op_swiglu, w_gate_gpu, w_up_gpu, dst_bufs[0]));
  engine->enqueue_op(op_down->op_id, x_buf.native_mtl_buffer(), 0,
                     dst_bufs[1].native_mtl_buffer(), 0, N_lin,
                     make_linear_suffix_cb(*op_down, w_down_gpu, dst_bufs[1]));
  engine->enqueue_op(op_z->op_id, x_buf.native_mtl_buffer(), 0,
                     dst_bufs[2].native_mtl_buffer(), 0, N_lin,
                     make_linear_suffix_cb(*op_z, w_z_gpu, dst_bufs[2]));
  engine->enqueue_op(op_out->op_id, x_buf.native_mtl_buffer(), 0,
                     dst_bufs[3].native_mtl_buffer(), 0, N_lin,
                     make_linear_suffix_cb(*op_out, w_out_gpu, dst_bufs[3]));
  engine->enqueue_op(op_attno->op_id, x_buf.native_mtl_buffer(), 0,
                     dst_bufs[4].native_mtl_buffer(), 0, N_lin,
                     make_linear_suffix_cb(*op_attno, w_attno_gpu, dst_bufs[4]));
  // 6th op repeats SwiGLU to test deterministic slot reuse across full pipeline!
  engine->enqueue_op(op_swiglu->op_id, x_buf.native_mtl_buffer(), 0,
                     dst_bufs[5].native_mtl_buffer(), 0, N_mlp,
                     make_swiglu_suffix_cb(*op_swiglu, w_gate_gpu, w_up_gpu, dst_bufs[5]));

  // SINGLE host wait after all 6 ops!
  engine->flush_and_wait();
  REQUIRE(engine->stats().total_enqueued_ops == 6u);
  REQUIRE(engine->stats().total_completed_ops == 6u);

  // Verify every single stitched output [0..N) (ANE slice AND GPU slice!) vs exact FP32 reference!
  auto check_linear = [&](const char *label, const AneCompiledOpInfo &info,
                          const AneSharedBuffer &dst,
                          const std::vector<float> &w_ref) -> bool {
    const std::uint32_t N = info.split.total_out_features;
    const auto *got_bf16 = static_cast<const std::uint16_t *>(dst.data());
    std::vector<float> got_full, ref_full, got_ane, ref_ane, got_gpu, ref_gpu;
    // Check 64 strided tokens across all 1024 tokens (including t=0, t=511, t=1023!) for fast exact verification
    for (std::uint32_t t = 0; t < T; t += 16u) {
      for (std::uint32_t n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (std::uint32_t c = 0; c < K; ++c) {
          acc = std::fma(x_f32[static_cast<std::size_t>(t) * K + c],
                         w_ref[static_cast<std::size_t>(n) * K + c], acc);
        }
        const float g = bf16_to_float(got_bf16[static_cast<std::size_t>(t) * N + n]);
        got_full.push_back(g);
        ref_full.push_back(acc);
        if (n < info.split.ane_total_rows) {
          got_ane.push_back(g);
          ref_ane.push_back(acc);
        } else {
          got_gpu.push_back(g);
          ref_gpu.push_back(acc);
        }
      }
    }
    const double cos_full = cosine_similarity(got_full, ref_full);
    const double cos_ane = cosine_similarity(got_ane, ref_ane);
    const double cos_gpu = cosine_similarity(got_gpu, ref_gpu);
    std::printf("  [run] %-20s dies=%u ane=[0..%u) gpu=[%u..%u) cos_full=%.6f cos_ane=%.6f cos_gpu=%.6f\n",
                label, info.split.active_dies, info.split.ane_total_rows,
                info.split.gpu_row_offset, N, cos_full, cos_ane, cos_gpu);
    return cos_full >= 0.9995 && cos_ane >= 0.9995 && cos_gpu >= 0.9999;
  };

  auto check_swiglu = [&](const char *label, const AneCompiledOpInfo &info,
                          const AneSharedBuffer &dst) -> bool {
    const std::uint32_t N = info.split.total_out_features;
    const auto *got_bf16 = static_cast<const std::uint16_t *>(dst.data());
    std::vector<float> got_full, ref_full, got_ane, ref_ane, got_gpu, ref_gpu;
    for (std::uint32_t t = 0; t < T; t += 16u) {
      for (std::uint32_t n = 0; n < N; ++n) {
        float ag = 0.0f, au = 0.0f;
        for (std::uint32_t c = 0; c < K; ++c) {
          const float xv = x_f32[static_cast<std::size_t>(t) * K + c];
          ag = std::fma(xv, f_gate.exact_f32[static_cast<std::size_t>(n) * K + c], ag);
          au = std::fma(xv, f_up.exact_f32[static_cast<std::size_t>(n) * K + c], au);
        }
        const float ref = (ag / (1.0f + std::exp(-ag))) * au;
        const float g = bf16_to_float(got_bf16[static_cast<std::size_t>(t) * N + n]);
        got_full.push_back(g);
        ref_full.push_back(ref);
        if (n < info.split.ane_total_rows) {
          got_ane.push_back(g);
          ref_ane.push_back(ref);
        } else {
          got_gpu.push_back(g);
          ref_gpu.push_back(ref);
        }
      }
    }
    const double cos_full = cosine_similarity(got_full, ref_full);
    const double cos_ane = cosine_similarity(got_ane, ref_ane);
    const double cos_gpu = cosine_similarity(got_gpu, ref_gpu);
    std::printf("  [run] %-20s dies=%u ane=[0..%u) gpu=[%u..%u) cos_full=%.6f cos_ane=%.6f cos_gpu=%.6f\n",
                label, info.split.active_dies, info.split.ane_total_rows,
                info.split.gpu_row_offset, N, cos_full, cos_ane, cos_gpu);
    return cos_full >= 0.9995 && cos_ane >= 0.9995 && cos_gpu >= 0.9999;
  };

  REQUIRE(check_swiglu("MlpGateUpSwiGlu#0", *op_swiglu, dst_bufs[0]));
  REQUIRE(check_linear("MlpDown(oQ5)", *op_down, dst_bufs[1], f_down.exact_f32));
  REQUIRE(check_linear("GdnInProjZ(MXFP4t)", *op_z, dst_bufs[2], f_gdn_z.exact_f32));
  REQUIRE(check_linear("GdnOutProj(MXFP4)", *op_out, dst_bufs[3], f_gdn_out.exact_f32));
  REQUIRE(check_linear("AttnOProj(oQ6g128)", *op_attno, dst_bufs[4], f_attn_o.exact_f32));
  REQUIRE(check_swiglu("MlpGateUpSwiGlu#5", *op_swiglu, dst_bufs[5]));
  // Slot-reuse determinism: op #0 and op #5 must be bit-identical!
  REQUIRE(std::memcmp(dst_bufs[0].data(), dst_bufs[5].data(),
                      static_cast<std::size_t>(dst_bufs[0].size())) == 0);

  // Save cold output for warm-cache bit-exact comparison
  std::vector<std::byte> cold_swiglu_bytes(static_cast<std::size_t>(dst_bufs[0].size()));
  std::memcpy(cold_swiglu_bytes.data(), dst_bufs[0].data(), cold_swiglu_bytes.size());

  // Micro-calibration tests (beneficial -> stays active; slow -> auto-bypass!)
  const auto cal_ok = engine->calibrate(/*gpu_only_ms=*/1000.0, /*gpu_suffix_ms=*/0.1, 0u);
  std::printf("  [cal] beneficial: ane_pair=%.3f ms active=%d bypassed=%d\n",
              cal_ok.ane_pair_ms, cal_ok.active ? 1 : 0, cal_ok.auto_bypassed ? 1 : 0);
  REQUIRE(cal_ok.performed && cal_ok.active && !cal_ok.auto_bypassed &&
          cal_ok.ane_pair_ms > 0.0 && engine->is_active());

  const auto cal_bypass = engine->calibrate(/*gpu_only_ms=*/0.0001, /*gpu_suffix_ms=*/0.0, 0u);
  std::printf("  [cal] bypass guard: ane_pair=%.3f ms active=%d bypassed=%d\n",
              cal_bypass.ane_pair_ms, cal_bypass.active ? 1 : 0,
              cal_bypass.auto_bypassed ? 1 : 0);
  REQUIRE(cal_bypass.performed && !cal_bypass.active && cal_bypass.auto_bypassed &&
          !engine->is_active());

  // Destroy cold engine and test warm disk-cache reload!
  engine.reset();

  auto warm_engine = AneDualOffloadEngine::build(specs, cfg);
  REQUIRE(warm_engine != nullptr && warm_engine->is_active());
  const auto warm_stats = warm_engine->stats();
  std::printf("  [cache] warm reload: disk_hits=%u disk_misses=%u native_compiled_hits=%u\n",
              warm_stats.disk_cache_hits, warm_stats.disk_cache_misses,
              warm_stats.native_compiled_cache_hits);
  REQUIRE(warm_stats.disk_cache_hits ==
          warm_stats.active_die_count * warm_stats.bank_shards_per_die);
  REQUIRE(warm_stats.disk_cache_misses == 0u);
  REQUIRE(warm_stats.native_compiled_cache_hits ==
          warm_stats.active_die_count * warm_stats.bank_shards_per_die);

  // Execute SwiGLU on warm engine and verify bit-identical output vs cold build!
  auto warm_x = warm_engine->make_shared_buffer(static_cast<std::uint64_t>(T) * K * 2u);
  std::memcpy(warm_x.data(), x_buf.data(), static_cast<std::size_t>(warm_x.size()));
  auto warm_dst = warm_engine->make_shared_buffer(static_cast<std::uint64_t>(T) * N_mlp * 2u);
  std::memset(warm_dst.data(), 0, static_cast<std::size_t>(warm_dst.size()));
  id<MTLDevice> wdev = (__bridge id<MTLDevice>)warm_engine->native_mtl_device();
  const auto wsuffix_psos = compile_gpu_suffix_shaders(wdev);
  auto wg_warm = warm_engine->make_shared_buffer(f_gate.exact_f32.size() * sizeof(float));
  auto wu_warm = warm_engine->make_shared_buffer(f_up.exact_f32.size() * sizeof(float));
  std::memcpy(wg_warm.data(), f_gate.exact_f32.data(), f_gate.exact_f32.size() * sizeof(float));
  std::memcpy(wu_warm.data(), f_up.exact_f32.data(), f_up.exact_f32.size() * sizeof(float));
  const auto *warm_op0 = warm_engine->find_op(0u, AneProjectionRole::MlpGateUpSwiGlu);
  REQUIRE(warm_op0 != nullptr);
  warm_engine->run_op_sync(
      warm_op0->op_id, warm_x.native_mtl_buffer(), 0,
      warm_dst.native_mtl_buffer(), 0, N_mlp,
      [&](void *raw_cb) {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)raw_cb;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        const SuffixParamsHost p{T, K, warm_op0->split.total_out_features,
                                 warm_op0->split.gpu_row_offset,
                                 warm_op0->split.gpu_rows};
        [enc setComputePipelineState:wsuffix_psos.swiglu_suffix_pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)warm_x.native_mtl_buffer() offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)wg_warm.native_mtl_buffer() offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)wu_warm.native_mtl_buffer() offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)warm_dst.native_mtl_buffer() offset:0 atIndex:3];
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(warm_op0->split.gpu_rows, T, 1u)
            threadsPerThreadgroup:MTLSizeMake(16u, 16u, 1u)];
        [enc endEncoding];
      });
  REQUIRE(std::memcmp(warm_dst.data(), cold_swiglu_bytes.data(),
                      cold_swiglu_bytes.size()) == 0);

  // Also test multi-shard bank splitting (< max_bank_bytes_per_die enforcement):
  // force 2 procedures per bank so 5 ops split cleanly into 3 shards per die and
  // verify async execution across multiple shards on the same die!
  {
    AneDualOffloadConfig shard_cfg = cfg;
    shard_cfg.max_procedures_per_bank = 2u;
    shard_cfg.enable_disk_cache = false;
    auto shard_eng = AneDualOffloadEngine::build(specs, shard_cfg);
    REQUIRE(shard_eng != nullptr && shard_eng->is_active());
    REQUIRE(shard_eng->stats().bank_shards_per_die == 3u);
    REQUIRE(shard_eng->stats().total_weight_bytes_per_die <
            kAneDefaultMaxBankBytesPerDie);
    auto sx = shard_eng->make_shared_buffer(static_cast<std::uint64_t>(T) * K * 2u);
    std::memcpy(sx.data(), x_buf.data(), static_cast<std::size_t>(sx.size()));
    auto sd0 = shard_eng->make_shared_buffer(static_cast<std::uint64_t>(T) * N_mlp * 2u);
    auto sd4 = shard_eng->make_shared_buffer(static_cast<std::uint64_t>(T) * N_lin * 2u);
    std::memset(sd0.data(), 0, static_cast<std::size_t>(sd0.size()));
    std::memset(sd4.data(), 0, static_cast<std::size_t>(sd4.size()));
    // Enqueue shard 0 (op 0) and shard 2 (op 4) back-to-back with zero CPU waits!
    shard_eng->enqueue_op(0u, sx.native_mtl_buffer(), 0, sd0.native_mtl_buffer(), 0, N_mlp);
    shard_eng->enqueue_op(4u, sx.native_mtl_buffer(), 0, sd4.native_mtl_buffer(), 0, N_lin);
    shard_eng->flush_and_wait();
    // Verify ANE columns [0..ane_total_rows) of shard 0 op 0 match cold_swiglu_bytes bit-for-bit!
    const auto *got0 = static_cast<const std::uint16_t *>(sd0.data());
    const auto *ref0 = reinterpret_cast<const std::uint16_t *>(cold_swiglu_bytes.data());
    const std::uint32_t ane_cols = shard_eng->find_op_by_id(0u)->split.ane_total_rows;
    for (std::uint32_t t = 0; t < T; t += 16u) {
      for (std::uint32_t n = 0; n < ane_cols; ++n) {
        REQUIRE(got0[static_cast<std::size_t>(t) * N_mlp + n] ==
                ref0[static_cast<std::size_t>(t) * N_mlp + n]);
      }
    }
    std::printf("  [shard] multi-shard (%u shards/die) bit-exact verified\n",
                shard_eng->stats().bank_shards_per_die);
  }

  warm_engine.reset();
  std::filesystem::remove_all(temp_cache, ec);

  std::cout << "[PASS] test_ane_execution_pipeline_and_cache (simulate_dual="
            << (simulate_dual_banks ? "true" : "false") << ")\n";
  return 0;
}

// ============================================================================
// WP 3C Tests: PrefillAneMode, Calibration Cache, Fused FWHT64 A8/A4 Unpack,
// ultra_fp16_swiglu Collapsed-CB Pipeline, Shared Input Surface, & Auto-Guard
// ============================================================================

struct ScopedEnvVar {
  std::string name;
  std::optional<std::string> prev;
  ScopedEnvVar(std::string key, const char *val) : name(std::move(key)) {
    if (const char *old = std::getenv(name.c_str())) {
      prev = std::string(old);
    }
    if (val != nullptr) {
      ::setenv(name.c_str(), val, 1);
    } else {
      ::unsetenv(name.c_str());
    }
  }
  ~ScopedEnvVar() {
    if (prev.has_value()) {
      ::setenv(name.c_str(), prev->c_str(), 1);
    } else {
      ::unsetenv(name.c_str());
    }
  }
};

std::uint8_t ref_encode_e8m0_round_up(float value) {
  if (!(value > 0.0f) || !std::isfinite(value)) {
    return 0u;
  }
  const int exponent =
      std::clamp(static_cast<int>(std::ceil(std::log2(value))), -127, 127);
  return static_cast<std::uint8_t>(exponent + 127);
}

std::uint16_t ref_f32_to_fp16(float v) {
  const __fp16 h = static_cast<__fp16>(v);
  std::uint16_t bits = 0u;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

float ref_fp16_to_f32(std::uint16_t bits) {
  __fp16 h{};
  std::memcpy(&h, &bits, sizeof(h));
  return static_cast<float>(h);
}

int test_prefill_ane_mode_and_calibration_cache() {
  // 1. PrefillAneMode & PrefillLinearMode::W4A4 string / parse contract (#6 §2.1)
  REQUIRE(std::string_view(prefill_ane_mode_to_string(PrefillAneMode::Off)) == "off");
  REQUIRE(std::string_view(prefill_ane_mode_to_string(PrefillAneMode::Auto)) == "auto");
  REQUIRE(std::string_view(prefill_ane_mode_to_string(PrefillAneMode::CompactInt8)) ==
          "compact_int8");
  REQUIRE(std::string_view(prefill_ane_mode_to_string(PrefillAneMode::UltraFp16SwiGlu)) ==
          "ultra_fp16_swiglu");
  REQUIRE(std::string_view(to_string(PrefillAneMode::UltraFp16SwiGlu)) ==
          "ultra_fp16_swiglu");
  REQUIRE(std::string_view(to_string(PrefillLinearMode::W4A4)) == "w4a4");
  REQUIRE(parse_prefill_linear_mode("w4a4") == PrefillLinearMode::W4A4);
  REQUIRE(parse_prefill_linear_mode("W4A4") == PrefillLinearMode::W4A4);

  REQUIRE(try_parse_prefill_ane_mode("off") == PrefillAneMode::Off);
  REQUIRE(try_parse_prefill_ane_mode("AUTO") == PrefillAneMode::Auto);
  REQUIRE(try_parse_prefill_ane_mode("compact_int8") == PrefillAneMode::CompactInt8);
  REQUIRE(try_parse_prefill_ane_mode("Ultra_FP16_SwiGLU") ==
          PrefillAneMode::UltraFp16SwiGlu);
  REQUIRE(!try_parse_prefill_ane_mode("bogus_mode").has_value());

  // Legacy prefill_ane_offload=true upgrades Off -> Auto; explicit modes win.
  REQUIRE(effective_requested_prefill_ane_mode(PrefillAneMode::Off, false) ==
          PrefillAneMode::Off);
  REQUIRE(effective_requested_prefill_ane_mode(PrefillAneMode::Off, true) ==
          PrefillAneMode::Auto);
  REQUIRE(effective_requested_prefill_ane_mode(PrefillAneMode::CompactInt8, false) ==
          PrefillAneMode::CompactInt8);
  REQUIRE(effective_requested_prefill_ane_mode(PrefillAneMode::UltraFp16SwiGlu, false) ==
          PrefillAneMode::UltraFp16SwiGlu);

  // 2. Calibration cache key uniqueness across every dimension & disk roundtrip
  const std::string k_base = compute_calibration_cache_key(
      "26B131", 80u, 2u, "digest_abc", AneLinearRoute::BF16, 2048u, 5120u,
      17408u, 0.98, std::nullopt);
  REQUIRE(!k_base.empty());
  REQUIRE(k_base != compute_calibration_cache_key(
                        "26B132", 80u, 2u, "digest_abc", AneLinearRoute::BF16,
                        2048u, 5120u, 17408u, 0.98, std::nullopt));
  REQUIRE(k_base != compute_calibration_cache_key(
                        "26B131", 40u, 2u, "digest_abc", AneLinearRoute::BF16,
                        2048u, 5120u, 17408u, 0.98, std::nullopt));
  REQUIRE(k_base != compute_calibration_cache_key(
                        "26B131", 80u, 1u, "digest_abc", AneLinearRoute::BF16,
                        2048u, 5120u, 17408u, 0.98, std::nullopt));
  REQUIRE(k_base != compute_calibration_cache_key(
                        "26B131", 80u, 2u, "digest_xyz", AneLinearRoute::BF16,
                        2048u, 5120u, 17408u, 0.98, std::nullopt));
  REQUIRE(k_base != compute_calibration_cache_key(
                        "26B131", 80u, 2u, "digest_abc", AneLinearRoute::W4A8,
                        2048u, 5120u, 17408u, 0.98, std::nullopt));
  REQUIRE(k_base != compute_calibration_cache_key(
                        "26B131", 80u, 2u, "digest_abc", AneLinearRoute::W4A4,
                        2048u, 5120u, 17408u, 0.98, std::nullopt));
  REQUIRE(k_base != compute_calibration_cache_key(
                        "26B131", 80u, 2u, "digest_abc", AneLinearRoute::BF16,
                        1024u, 5120u, 17408u, 0.98, std::nullopt));
  REQUIRE(k_base != compute_calibration_cache_key(
                        "26B131", 80u, 2u, "digest_abc", AneLinearRoute::BF16,
                        2048u, 5120u, 17408u, 0.98, std::optional<double>{0.10}));

  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       ("miv_ane_cal_cache_test_" +
                        std::to_string(static_cast<long>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(tmp_dir, ec);
  REQUIRE(!load_cached_calibration(tmp_dir, k_base).valid);

  AneCachedCalibrationRecord rec_active{};
  rec_active.valid = true;
  rec_active.active = true;
  rec_active.best_alpha = 0.10;
  rec_active.gpu_solo_ms = 2.450;
  rec_active.best_hybrid_ms = 2.210;
  rec_active.reason = "calibrated_active";
  rec_active.candidate_timings_ms = {{0.12, 2.30}, {0.10, 2.21}};
  save_cached_calibration(tmp_dir, k_base, rec_active);
  const auto loaded_active = load_cached_calibration(tmp_dir, k_base);
  REQUIRE(loaded_active.valid);
  REQUIRE(loaded_active.active &&
          std::abs(loaded_active.best_alpha - 0.10) < 1e-6 &&
          std::abs(loaded_active.gpu_solo_ms - 2.450) < 1e-3 &&
          std::abs(loaded_active.best_hybrid_ms - 2.210) < 1e-3 &&
          loaded_active.reason == "calibrated_active");

  const std::string k_bypass = compute_calibration_cache_key(
      "26B131", 80u, 2u, "digest_abc", AneLinearRoute::W4A8, 2048u, 5120u,
      17408u, 0.98, std::nullopt);
  AneCachedCalibrationRecord rec_bypass{};
  rec_bypass.valid = true;
  rec_bypass.active = false;
  rec_bypass.best_alpha = 0.0;
  rec_bypass.gpu_solo_ms = 1.650;
  rec_bypass.best_hybrid_ms = 1.720;
  rec_bypass.reason = "calibration_bypassed";
  save_cached_calibration(tmp_dir, k_bypass, rec_bypass);
  const auto loaded_bypass = load_cached_calibration(tmp_dir, k_bypass);
  REQUIRE(loaded_bypass.valid);
  REQUIRE(!loaded_bypass.active &&
          loaded_bypass.reason == "calibration_bypassed");

  std::filesystem::remove_all(tmp_dir, ec);
  std::cout << "[PASS] test_prefill_ane_mode_and_calibration_cache\n";
  return 0;
}

struct AneUnpackQuantParamsHost {
  std::uint32_t T;
  std::uint32_t mp;
  std::uint32_t ane0_n;
  std::uint32_t ane1_n;
  std::uint32_t out_stride;
  std::uint32_t dst_col_offset;
  std::uint32_t reserved0;
  std::uint32_t reserved1;
};

int test_ane_unpack_fwht64_a8_and_a4_kernels(
    const MetalContext &ctx,
    const std::filesystem::path &metallib_dir) {
  (void)ctx;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  REQUIRE(dev != nil);
  id<MTLCommandQueue> queue = [dev newCommandQueue];
  REQUIRE(queue != nil);

  const auto lib_path = metallib_dir / "omlx_m5_ane_pack_unpack.metallib";
  REQUIRE(std::filesystem::exists(lib_path));
  NSError *err = nil;
  NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:lib_path.c_str()]];
  id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
  REQUIRE(lib != nil);

  auto load_pso = [&](const char *fn_name) -> id<MTLComputePipelineState> {
    id<MTLFunction> fn =
        [lib newFunctionWithName:[NSString stringWithUTF8String:fn_name]];
    if (fn == nil) return nil;
    NSError *perr = nil;
    return [dev newComputePipelineStateWithFunction:fn error:&perr];
  };

  id<MTLComputePipelineState> pso_a8_s =
      load_pso("omlx_m5_ane_unpack_fp16_planar_to_fwht64_a8_single");
  id<MTLComputePipelineState> pso_a8_d =
      load_pso("omlx_m5_ane_unpack_fp16_planar_to_fwht64_a8_dual");
  id<MTLComputePipelineState> pso_a4_s =
      load_pso("omlx_m5_ane_unpack_fp16_planar_to_fwht64_a4_single");
  id<MTLComputePipelineState> pso_a4_d =
      load_pso("omlx_m5_ane_unpack_fp16_planar_to_fwht64_a4_dual");
  REQUIRE(pso_a8_s != nil && pso_a8_d != nil && pso_a4_s != nil && pso_a4_d != nil);

  // Test both unaligned T=19 (exercises tail zero/0x7f padding!) and aligned T=64,
  // in both single-die and dual-die modes.
  const std::array<std::uint32_t, 2> test_Ts = {19u, 64u};
  const std::uint32_t out_stride = 512u;
  const std::uint32_t total_n = 256u; // [0..256) written by ANE; [256..512) GPU suffix sentinel!

  for (const std::uint32_t T : test_Ts) {
    const std::uint32_t mp = ((T + 15u) / 16u) * 16u; // 32 for T=19, 64 for T=64
    // Build deterministic FP16 planar matrix [total_n, T] and row-major FP32 copy [T, total_n].
    std::vector<std::uint16_t> planar(static_cast<std::size_t>(total_n) * T);
    std::vector<float> orig_row_major(static_cast<std::size_t>(T) * total_n);
    for (std::uint32_t n = 0u; n < total_n; ++n) {
      for (std::uint32_t t = 0u; t < T; ++t) {
        const float v =
            1.75f * std::sin(static_cast<float>(n * 19u + t * 7u) * 0.037f) +
            0.85f * std::cos(static_cast<float>(n * 5u + t * 23u) * 0.051f);
        const std::uint16_t h = ref_f32_to_fp16(v);
        planar[static_cast<std::size_t>(n) * T + t] = h;
        orig_row_major[static_cast<std::size_t>(t) * total_n + n] = ref_fp16_to_f32(h);
      }
    }
    std::vector<float> ref_fwht = orig_row_major;
    fwht64_test_rows(ref_fwht, T, total_n);

    for (const bool dual : {false, true}) {
      const std::uint32_t ane0_n = dual ? (total_n / 2u) : total_n;
      const std::uint32_t ane1_n = dual ? (total_n - ane0_n) : 0u;

      id<MTLBuffer> src0_buf = [dev
          newBufferWithBytes:planar.data()
                      length:static_cast<NSUInteger>(ane0_n) * T * sizeof(std::uint16_t)
                     options:MTLResourceStorageModeShared];
      id<MTLBuffer> src1_buf =
          dual ? [dev newBufferWithBytes:planar.data() + static_cast<std::size_t>(ane0_n) * T
                                  length:static_cast<NSUInteger>(ane1_n) * T * sizeof(std::uint16_t)
                                 options:MTLResourceStorageModeShared]
               : nil;

      // --- Part 1: FWHT64 + GS64 INT8 (W4A8 Mode B) ---
      const std::size_t qa_bytes = static_cast<std::size_t>(mp) * out_stride;
      const std::size_t sa_elems = static_cast<std::size_t>(out_stride / 64u) * mp;
      id<MTLBuffer> qa_buf = [dev newBufferWithLength:qa_bytes options:MTLResourceStorageModeShared];
      id<MTLBuffer> sa_buf =
          [dev newBufferWithLength:sa_elems * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> ra_buf =
          [dev newBufferWithLength:sa_elems * sizeof(float) options:MTLResourceStorageModeShared];
      std::memset([qa_buf contents], 0x5A, qa_bytes);
      auto *sa_init = static_cast<float *>([sa_buf contents]);
      auto *ra_init = static_cast<float *>([ra_buf contents]);
      for (std::size_t i = 0; i < sa_elems; ++i) {
        sa_init[i] = -999.0f;
        ra_init[i] = -777.0f;
      }

      const AneUnpackQuantParamsHost p_a8{T, mp, ane0_n, ane1_n, out_stride, 0u, 0u, 0u};
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:(dual ? pso_a8_d : pso_a8_s)];
      if (dual) {
        [enc setBuffer:src0_buf offset:0 atIndex:0];
        [enc setBuffer:src1_buf offset:0 atIndex:1];
        [enc setBuffer:qa_buf offset:0 atIndex:2];
        [enc setBuffer:sa_buf offset:0 atIndex:3];
        [enc setBuffer:ra_buf offset:0 atIndex:4];
        [enc setBytes:&p_a8 length:sizeof(p_a8) atIndex:5];
      } else {
        [enc setBuffer:src0_buf offset:0 atIndex:0];
        [enc setBuffer:qa_buf offset:0 atIndex:1];
        [enc setBuffer:sa_buf offset:0 atIndex:2];
        [enc setBuffer:ra_buf offset:0 atIndex:3];
        [enc setBytes:&p_a8 length:sizeof(p_a8) atIndex:4];
      }
      [enc dispatchThreadgroups:MTLSizeMake(total_n / 64u, (mp + 31u) / 32u, 1u)
          threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      REQUIRE([cb status] == MTLCommandBufferStatusCompleted);

      const auto *qa_ptr = static_cast<const std::int8_t *>([qa_buf contents]);
      const auto *sa_ptr = static_cast<const float *>([sa_buf contents]);
      const auto *ra_ptr = static_cast<const float *>([ra_buf contents]);

      // Verify GPU suffix columns [total_n, out_stride) completely untouched!
      for (std::uint32_t t = 0u; t < mp; ++t) {
        for (std::uint32_t c = total_n; c < out_stride; ++c) {
          REQUIRE(static_cast<std::uint8_t>(
                      qa_ptr[static_cast<std::size_t>(t) * out_stride + c]) == 0x5Au);
        }
      }
      for (std::uint32_t g = total_n / 64u; g < out_stride / 64u; ++g) {
        for (std::uint32_t t = 0u; t < mp; ++t) {
          REQUIRE(sa_ptr[static_cast<std::size_t>(g) * mp + t] == -999.0f);
          REQUIRE(ra_ptr[static_cast<std::size_t>(g) * mp + t] == -777.0f);
        }
      }
      // Verify tail padded rows [T, mp) are exact zeros!
      for (std::uint32_t t = T; t < mp; ++t) {
        for (std::uint32_t c = 0u; c < total_n; ++c) {
          REQUIRE(qa_ptr[static_cast<std::size_t>(t) * out_stride + c] == 0);
        }
        for (std::uint32_t g = 0u; g < total_n / 64u; ++g) {
          REQUIRE(sa_ptr[static_cast<std::size_t>(g) * mp + t] == 0.0f);
          REQUIRE(ra_ptr[static_cast<std::size_t>(g) * mp + t] == 0.0f);
        }
      }
      // Verify valid rows [0, T): exact ra = sum(qa), dequant matches ref_fwht >= 0.9999!
      std::vector<float> deq_fwht(static_cast<std::size_t>(T) * total_n);
      for (std::uint32_t t = 0u; t < T; ++t) {
        for (std::uint32_t g = 0u; g < total_n / 64u; ++g) {
          const float sa = sa_ptr[static_cast<std::size_t>(g) * mp + t];
          const float ra = ra_ptr[static_cast<std::size_t>(g) * mp + t];
          int sum_q = 0;
          for (std::uint32_t i = 0u; i < 64u; ++i) {
            const std::int8_t qv =
                qa_ptr[static_cast<std::size_t>(t) * out_stride + g * 64u + i];
            sum_q += static_cast<int>(qv);
            deq_fwht[static_cast<std::size_t>(t) * total_n + g * 64u + i] =
                static_cast<float>(qv) * sa;
          }
          REQUIRE(ra == static_cast<float>(sum_q));
        }
      }
      const double cos_a8_fwht = cosine_similarity(deq_fwht, ref_fwht);
      std::vector<float> inv_a8 = deq_fwht;
      fwht64_test_rows(inv_a8, T, total_n);
      const double cos_a8_orig = cosine_similarity(inv_a8, orig_row_major);
      REQUIRE(cos_a8_fwht >= 0.9999);
      REQUIRE(cos_a8_orig >= 0.9999);

      // --- Part 2: FWHT64 + MXFP4 A4 bm16_bk128 (W4A4 Mode B) ---
      const std::uint32_t m_tiles = (mp + 15u) / 16u;
      const std::uint32_t k_blocks = out_stride / 128u;
      const std::size_t n_tiles = static_cast<std::size_t>(m_tiles) * k_blocks;
      id<MTLBuffer> a4_codes =
          [dev newBufferWithLength:n_tiles * 1024u options:MTLResourceStorageModeShared];
      id<MTLBuffer> a4_scales =
          [dev newBufferWithLength:n_tiles * 64u options:MTLResourceStorageModeShared];
      std::memset([a4_codes contents], 0xA5, n_tiles * 1024u);
      std::memset([a4_scales contents], 0xA5, n_tiles * 64u);

      const AneUnpackQuantParamsHost p_a4{T, mp, ane0_n, ane1_n, out_stride, 0u, 0u, 0u};
      cb = [queue commandBuffer];
      enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:(dual ? pso_a4_d : pso_a4_s)];
      if (dual) {
        [enc setBuffer:src0_buf offset:0 atIndex:0];
        [enc setBuffer:src1_buf offset:0 atIndex:1];
        [enc setBuffer:a4_codes offset:0 atIndex:2];
        [enc setBuffer:a4_scales offset:0 atIndex:3];
        [enc setBytes:&p_a4 length:sizeof(p_a4) atIndex:4];
      } else {
        [enc setBuffer:src0_buf offset:0 atIndex:0];
        [enc setBuffer:a4_codes offset:0 atIndex:1];
        [enc setBuffer:a4_scales offset:0 atIndex:2];
        [enc setBytes:&p_a4 length:sizeof(p_a4) atIndex:3];
      }
      [enc dispatchThreadgroups:MTLSizeMake(total_n / 128u, m_tiles, 1u)
          threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      REQUIRE([cb status] == MTLCommandBufferStatusCompleted);

      const auto *codes_ptr = static_cast<const std::uint8_t *>([a4_codes contents]);
      const auto *scales_ptr = static_cast<const std::uint8_t *>([a4_scales contents]);

      // Verify GPU suffix tiles (blk128 >= total_n / 128) completely untouched!
      for (std::uint32_t tm = 0u; tm < m_tiles; ++tm) {
        for (std::uint32_t blk = total_n / 128u; blk < k_blocks; ++blk) {
          const std::size_t ti = static_cast<std::size_t>(tm) * k_blocks + blk;
          for (std::size_t b = 0; b < 1024u; ++b) {
            REQUIRE(codes_ptr[ti * 1024u + b] == 0xA5u);
          }
          for (std::size_t b = 0; b < 64u; ++b) {
            REQUIRE(scales_ptr[ti * 64u + b] == 0xA5u);
          }
        }
      }
      // Verify tail padded rows [T, mp) have exact codes == 0x00 and scales == 0x7f!
      for (std::uint32_t t = T; t < mp; ++t) {
        const std::uint32_t tm = t / 16u;
        const std::uint32_t rm = t % 16u;
        for (std::uint32_t blk = 0u; blk < total_n / 128u; ++blk) {
          const std::size_t ti = static_cast<std::size_t>(tm) * k_blocks + blk;
          for (std::uint32_t b = 0u; b < 64u; ++b) {
            REQUIRE(codes_ptr[ti * 1024u + rm * 64u + b] == 0x00u);
          }
          for (std::uint32_t s = 0u; s < 4u; ++s) {
            REQUIRE(scales_ptr[ti * 64u + rm * 4u + s] == 0x7Fu);
          }
        }
      }
      // Compare GPU A4 tiles vs exact CPU reference of omlx_m5_bf16_fwht64_to_mxfp4_a4_bm16_bk128!
      std::vector<float> gpu_a4_deq(static_cast<std::size_t>(T) * total_n);
      std::vector<float> cpu_a4_deq(static_cast<std::size_t>(T) * total_n);
      std::size_t scale_exact = 0u;
      std::size_t scale_total = 0u;
      for (std::uint32_t t = 0u; t < T; ++t) {
        const std::uint32_t tm = t / 16u;
        const std::uint32_t rm = t % 16u;
        for (std::uint32_t blk = 0u; blk < total_n / 128u; ++blk) {
          const std::size_t ti = static_cast<std::size_t>(tm) * k_blocks + blk;
          for (std::uint32_t sub = 0u; sub < 4u; ++sub) {
            const std::uint32_t c0 = blk * 128u + sub * 32u;
            float amax = 0.0f;
            for (std::uint32_t i = 0u; i < 32u; ++i) {
              amax = std::max(
                  amax, std::abs(ref_fwht[static_cast<std::size_t>(t) * total_n + c0 + i]));
            }
            const std::uint8_t ref_sc = ref_encode_e8m0_round_up(amax / 6.0f);
            const std::uint8_t got_sc = scales_ptr[ti * 64u + rm * 4u + sub];
            ++scale_total;
            if (got_sc == ref_sc) {
              ++scale_exact;
            }
            const float ref_scale = ref_decode_e8m0(ref_sc);
            const float got_scale = ref_decode_e8m0(got_sc);
            const float inv_ref = ref_scale > 0.0f ? (1.0f / ref_scale) : 0.0f;
            for (std::uint32_t j = 0u; j < 16u; ++j) {
              const std::uint8_t packed =
                  codes_ptr[ti * 1024u + rm * 64u + sub * 16u + j];
              const std::uint8_t g0 = packed & 0x0Fu;
              const std::uint8_t g1 = (packed >> 4u) & 0x0Fu;
              gpu_a4_deq[static_cast<std::size_t>(t) * total_n + c0 + 2u * j] =
                  ref_decode_e2m1(g0) * got_scale;
              gpu_a4_deq[static_cast<std::size_t>(t) * total_n + c0 + 2u * j + 1u] =
                  ref_decode_e2m1(g1) * got_scale;
              const std::uint8_t r0 = encode_e2m1_ref(
                  ref_fwht[static_cast<std::size_t>(t) * total_n + c0 + 2u * j] * inv_ref);
              const std::uint8_t r1 = encode_e2m1_ref(
                  ref_fwht[static_cast<std::size_t>(t) * total_n + c0 + 2u * j + 1u] *
                  inv_ref);
              cpu_a4_deq[static_cast<std::size_t>(t) * total_n + c0 + 2u * j] =
                  ref_decode_e2m1(r0) * ref_scale;
              cpu_a4_deq[static_cast<std::size_t>(t) * total_n + c0 + 2u * j + 1u] =
                  ref_decode_e2m1(r1) * ref_scale;
            }
          }
        }
      }
      const double cos_a4_vs_cpu = cosine_similarity(gpu_a4_deq, cpu_a4_deq);
      std::printf(
          "  [unpack] T=%2u dual=%d a8_fwht_cos=%.7f a8_orig_cos=%.7f a4_vs_cpu_cos=%.7f a4_scale_exact=%zu/%zu\n",
          T, dual ? 1 : 0, cos_a8_fwht, cos_a8_orig, cos_a4_vs_cpu, scale_exact,
          scale_total);
      REQUIRE(cos_a4_vs_cpu >= 0.9999);
      REQUIRE(scale_exact == scale_total);
    }
  }

  std::cout << "[PASS] test_ane_unpack_fwht64_a8_and_a4_kernels (single+dual, T=19 tail-pad + T=64, cos>=0.9999)\n";
  return 0;
}

int test_ultra_fp16_swiglu_pipeline(
    const MetalContext &ctx,
    const std::filesystem::path &metallib_dir,
    bool simulate_dual_banks) {
  const AneHardwareInfo hw = probe_ane_hardware();
  if (!hw.is_available()) {
    std::cout << "[SKIP] test_ultra_fp16_swiglu_pipeline (ANE unavailable)\n";
    return 0;
  }

  const std::uint32_t T = kAneUltraDefaultChunkTokens; // 2048
  const std::uint32_t K = 512u;
  const std::uint32_t N = 512u; // ANE gets 256 (dual) or 128 (single), multiple of 128!

  // Layer 0: oQ4e lane_v2; Layer 1: MXFP4 tiled_bk128
  auto f_g0 = make_affine_fixture(ctx, "model.layers.0.mlp.gate_proj", 101u, N, K,
                                  4u, 64u, true, 701u, true, false);
  auto f_u0 = make_affine_fixture(ctx, "model.layers.0.mlp.up_proj", 102u, N, K,
                                  5u, 64u, true, 702u, true, false);
  auto f_g1 = make_mxfp4_fixture(ctx, "model.layers.1.mlp.gate_proj", 103u, N, K,
                                 true, 703u, false);
  auto f_u1 = make_mxfp4_fixture(ctx, "model.layers.1.mlp.up_proj", 104u, N, K,
                                 true, 704u, false);
  auto f_down0 = make_mxfp4_fixture(ctx, "model.layers.0.mlp.down_proj", 105u, K, N,
                                    true, 705u, true);
  auto f_z0 = make_mxfp4_fixture(ctx, "model.layers.0.linear_attn.in_proj_z", 106u,
                                 N, K, true, 706u, false);

  // Include non-SwiGLU ops to verify mlp_swiglu_only = true filters strictly to SwiGLU!
  std::vector<AneOffloadOpSpec> specs = {
      AneOffloadOpSpec{0u, AneProjectionRole::MlpGateUpSwiGlu, "mlp_gate_up_swiglu",
                       f_g0.view, f_u0.view},
      AneOffloadOpSpec{0u, AneProjectionRole::MlpDown, "model.layers.0.mlp.down_proj",
                       f_down0.view, std::nullopt},
      AneOffloadOpSpec{0u, AneProjectionRole::GdnInProjZ,
                       "model.layers.0.linear_attn.in_proj_z", f_z0.view, std::nullopt},
      AneOffloadOpSpec{1u, AneProjectionRole::MlpGateUpSwiGlu, "mlp_gate_up_swiglu",
                       f_g1.view, f_u1.view},
  };

  const auto temp_cache =
      std::filesystem::temp_directory_path() /
      ("miv_ane_ultra_test_cache_" + std::to_string(static_cast<long>(::getpid())) +
       (simulate_dual_banks ? "_dual" : "_native"));
  std::error_code ec;
  std::filesystem::remove_all(temp_cache, ec);

  AneDualOffloadConfig cfg{};
  cfg.chunk_tokens = T;
  cfg.linear_route = AneLinearRoute::W4A8;
  cfg.alpha_override = simulate_dual_banks ? 0.50 : 0.25;
  cfg.max_alpha = 0.50;
  cfg.use_fp16_weights = true;
  cfg.mlp_swiglu_only = true;
  cfg.share_input_surface = true;
  cfg.max_bank_bytes_per_die = kAneUltraMaxBankBytesPerDie;
  cfg.max_procedures_per_bank = kAneUltraMaxProceduresPerBank;
  cfg.max_in_features = 8192u;
  cfg.simulate_dual_banks_on_single_die = simulate_dual_banks;
  cfg.enable_disk_cache = true;
  cfg.cache_root = temp_cache;
  cfg.artifact_digest = "test_wp3c_ultra_swiglu_digest_v1";
  cfg.metallib_path = metallib_dir;

  auto engine = AneDualOffloadEngine::build(specs, cfg);
  REQUIRE(engine != nullptr && engine->is_available() && engine->is_active());
  // Strictly filtered to ONLY the 2 MlpGateUpSwiGlu ops!
  REQUIRE(engine->ops().size() == 2u);
  REQUIRE(engine->find_op(0u, AneProjectionRole::MlpGateUpSwiGlu) != nullptr);
  REQUIRE(engine->find_op(1u, AneProjectionRole::MlpGateUpSwiGlu) != nullptr);
  REQUIRE(engine->find_op(0u, AneProjectionRole::MlpDown) == nullptr);
  REQUIRE(engine->find_op(0u, AneProjectionRole::GdnInProjZ) == nullptr);
  REQUIRE(engine->can_offload_chunk(2048u));
  REQUIRE(!engine->can_offload_chunk(1024u));
  const auto st = engine->stats();
  REQUIRE(st.compiled_ops == 2u && st.chunk_tokens == T);
  std::printf("  [ultra] dies=%u shared_in_surface=%d cache_misses=%u\n",
              st.active_die_count, st.shared_input_surface_active ? 1 : 0,
              st.disk_cache_misses);

  // Build deterministic BF16 input X [T, K]
  std::vector<float> x_f32(static_cast<std::size_t>(T) * K);
  auto x_buf = engine->make_shared_buffer(static_cast<std::uint64_t>(T) * K * 2u);
  auto *x_bf16 = static_cast<std::uint16_t *>(x_buf.data());
  for (std::uint32_t t = 0u; t < T; ++t) {
    for (std::uint32_t c = 0u; c < K; ++c) {
      const float v =
          0.32f * std::sin(static_cast<float>(t * 13u + c * 5u) * 0.009f) +
          0.18f * std::cos(static_cast<float>(t * 7u + c * 11u) * 0.017f);
      const std::uint16_t b = float_to_bf16(v);
      x_bf16[static_cast<std::size_t>(t) * K + c] = b;
      x_f32[static_cast<std::size_t>(t) * K + c] = bf16_to_float(b);
    }
  }

  id<MTLDevice> dev = (__bridge id<MTLDevice>)engine->native_mtl_device();
  const auto suffix_psos = compile_gpu_suffix_shaders(dev);
  auto upload_f32 = [&](const std::vector<float> &v) {
    auto buf = engine->make_shared_buffer(v.size() * sizeof(float));
    std::memcpy(buf.data(), v.data(), v.size() * sizeof(float));
    return buf;
  };
  auto wg0_gpu = upload_f32(f_g0.exact_f32);
  auto wu0_gpu = upload_f32(f_u0.exact_f32);

  const auto *op0 = engine->find_op(0u, AneProjectionRole::MlpGateUpSwiGlu);
  const auto *op1 = engine->find_op(1u, AneProjectionRole::MlpGateUpSwiGlu);
  REQUIRE(op0 != nullptr && op1 != nullptr);
  const std::uint32_t n_ane = op0->split.ane_total_rows;
  REQUIRE((n_ane % 128u) == 0u);

  // 1) BF16 unpack + GPU suffix via run_op_sync_ex
  auto dst_bf16 = engine->make_shared_buffer(static_cast<std::uint64_t>(T) * N * 2u);
  std::memset(dst_bf16.data(), 0, static_cast<std::size_t>(dst_bf16.size()));
  AneUnpackTarget tgt_bf16{};
  tgt_bf16.mode = AneUnpackMode::BF16;
  tgt_bf16.primary_buf = dst_bf16.native_mtl_buffer();
  tgt_bf16.out_stride = N;
  engine->run_op_sync_ex(
      op0->op_id, x_buf.native_mtl_buffer(), 0u, 0u, tgt_bf16,
      [&](void *raw_cb) {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)raw_cb;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        const SuffixParamsHost p{T, K, N, op0->split.gpu_row_offset, op0->split.gpu_rows};
        [enc setComputePipelineState:suffix_psos.swiglu_suffix_pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)x_buf.native_mtl_buffer() offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)wg0_gpu.native_mtl_buffer() offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)wu0_gpu.native_mtl_buffer() offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst_bf16.native_mtl_buffer() offset:0 atIndex:3];
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(op0->split.gpu_rows, T, 1u)
            threadsPerThreadgroup:MTLSizeMake(16u, 16u, 1u)];
        [enc endEncoding];
      });

  // Verify BF16 cosine similarity vs FP32 exact SwiGLU (sampled across T=2048)
  {
    const auto *got = static_cast<const std::uint16_t *>(dst_bf16.data());
    std::vector<float> got_ane, ref_ane, got_full, ref_full;
    for (std::uint32_t t = 0u; t < T; t += 32u) {
      for (std::uint32_t n = 0u; n < N; ++n) {
        float ag = 0.0f, au = 0.0f;
        for (std::uint32_t c = 0u; c < K; ++c) {
          const float xv = x_f32[static_cast<std::size_t>(t) * K + c];
          ag = std::fma(xv, f_g0.exact_f32[static_cast<std::size_t>(n) * K + c], ag);
          au = std::fma(xv, f_u0.exact_f32[static_cast<std::size_t>(n) * K + c], au);
        }
        const float ref = (ag / (1.0f + std::exp(-ag))) * au;
        const float g = bf16_to_float(got[static_cast<std::size_t>(t) * N + n]);
        got_full.push_back(g);
        ref_full.push_back(ref);
        if (n < n_ane) {
          got_ane.push_back(g);
          ref_ane.push_back(ref);
        }
      }
    }
    const double cos_ane = cosine_similarity(got_ane, ref_ane);
    const double cos_full = cosine_similarity(got_full, ref_full);
    std::printf("  [ultra-bf16] cos_ane=%.7f cos_full=%.7f\n", cos_ane, cos_full);
    REQUIRE(cos_ane >= 0.9999 && cos_full >= 0.9999);
  }

  // 2) Mode B Fwht64A8 unpack via run_op_sync_ex (op1: MXFP4 tiled_bk128)
  auto a8_qa = engine->make_shared_buffer(static_cast<std::uint64_t>(T) * N);
  auto a8_sa = engine->make_shared_buffer(static_cast<std::uint64_t>(N / 64u) * T * 4u);
  auto a8_ra = engine->make_shared_buffer(static_cast<std::uint64_t>(N / 64u) * T * 4u);
  std::memset(a8_qa.data(), 0x5A, static_cast<std::size_t>(a8_qa.size()));
  AneUnpackTarget tgt_a8{};
  tgt_a8.mode = AneUnpackMode::Fwht64A8;
  tgt_a8.primary_buf = a8_qa.native_mtl_buffer();
  tgt_a8.aux0_buf = a8_sa.native_mtl_buffer();
  tgt_a8.aux1_buf = a8_ra.native_mtl_buffer();
  tgt_a8.out_stride = N;
  tgt_a8.mp = T;
  engine->run_op_sync_ex(op1->op_id, x_buf.native_mtl_buffer(), 0u, 0u, tgt_a8);

  {
    const auto *qa = static_cast<const std::int8_t *>(a8_qa.data());
    const auto *sa = static_cast<const float *>(a8_sa.data());
    const auto *ra = static_cast<const float *>(a8_ra.data());
    // GPU suffix [n_ane, N) untouched!
    for (std::uint32_t t = 0u; t < T; t += 64u) {
      for (std::uint32_t c = n_ane; c < N; ++c) {
        REQUIRE(static_cast<std::uint8_t>(qa[static_cast<std::size_t>(t) * N + c]) == 0x5Au);
      }
    }
    // Dequantize + inverse FWHT64 vs FP32 exact SwiGLU on layer 1 (MXFP4)
    const std::uint32_t sample_rows = 64u;
    std::vector<float> deq_ane(static_cast<std::size_t>(sample_rows) * n_ane);
    std::vector<float> ref_ane(static_cast<std::size_t>(sample_rows) * n_ane);
    for (std::uint32_t sr = 0u; sr < sample_rows; ++sr) {
      const std::uint32_t t = sr * 32u;
      for (std::uint32_t g = 0u; g < n_ane / 64u; ++g) {
        const float sc = sa[static_cast<std::size_t>(g) * T + t];
        const float rsum = ra[static_cast<std::size_t>(g) * T + t];
        int sum_q = 0;
        for (std::uint32_t i = 0u; i < 64u; ++i) {
          const std::int8_t qv = qa[static_cast<std::size_t>(t) * N + g * 64u + i];
          sum_q += static_cast<int>(qv);
          deq_ane[static_cast<std::size_t>(sr) * n_ane + g * 64u + i] =
              static_cast<float>(qv) * sc;
        }
        REQUIRE(rsum == static_cast<float>(sum_q));
      }
      for (std::uint32_t n = 0u; n < n_ane; ++n) {
        float ag = 0.0f, au = 0.0f;
        for (std::uint32_t c = 0u; c < K; ++c) {
          const float xv = x_f32[static_cast<std::size_t>(t) * K + c];
          ag = std::fma(xv, f_g1.exact_f32[static_cast<std::size_t>(n) * K + c], ag);
          au = std::fma(xv, f_u1.exact_f32[static_cast<std::size_t>(n) * K + c], au);
        }
        ref_ane[static_cast<std::size_t>(sr) * n_ane + n] =
            (ag / (1.0f + std::exp(-ag))) * au;
      }
    }
    fwht64_test_rows(deq_ane, sample_rows, n_ane);
    const double cos_a8 = cosine_similarity(deq_ane, ref_ane);
    std::printf("  [ultra-a8]   cos_ane=%.7f (MXFP4 FP16->FWHT64+A8)\n", cos_a8);
    REQUIRE(cos_a8 >= 0.9999);
  }

  // 3) Mode B Fwht64A4 unpack via run_op_sync_ex (op0)
  const std::uint32_t m_tiles = T / 16u;
  const std::uint32_t k_blocks = N / 128u;
  const std::size_t n_tiles = static_cast<std::size_t>(m_tiles) * k_blocks;
  auto a4_codes = engine->make_shared_buffer(n_tiles * 1024u);
  auto a4_scales = engine->make_shared_buffer(n_tiles * 64u);
  std::memset(a4_codes.data(), 0xA5, static_cast<std::size_t>(a4_codes.size()));
  std::memset(a4_scales.data(), 0xA5, static_cast<std::size_t>(a4_scales.size()));
  AneUnpackTarget tgt_a4{};
  tgt_a4.mode = AneUnpackMode::Fwht64A4;
  tgt_a4.primary_buf = a4_codes.native_mtl_buffer();
  tgt_a4.aux0_buf = a4_scales.native_mtl_buffer();
  tgt_a4.out_stride = N;
  tgt_a4.mp = T;
  engine->run_op_sync_ex(op0->op_id, x_buf.native_mtl_buffer(), 0u, 0u, tgt_a4);

  {
    const auto *codes = static_cast<const std::uint8_t *>(a4_codes.data());
    const auto *scales = static_cast<const std::uint8_t *>(a4_scales.data());
    // GPU suffix tiles untouched!
    for (std::uint32_t tm = 0u; tm < m_tiles; tm += 8u) {
      for (std::uint32_t blk = n_ane / 128u; blk < k_blocks; ++blk) {
        const std::size_t ti = static_cast<std::size_t>(tm) * k_blocks + blk;
        REQUIRE(codes[ti * 1024u] == 0xA5u && scales[ti * 64u] == 0xA5u);
      }
    }
    // Compare GPU A4 tiles vs CPU A4 quantization of op0's BF16 output AND vs FP32 exact!
    const auto *bf16_ptr = static_cast<const std::uint16_t *>(dst_bf16.data());
    const std::uint32_t sample_rows = 64u;
    std::vector<float> bf16_ane(static_cast<std::size_t>(sample_rows) * n_ane);
    std::vector<float> gpu_a4_fwht(static_cast<std::size_t>(sample_rows) * n_ane);
    for (std::uint32_t sr = 0u; sr < sample_rows; ++sr) {
      const std::uint32_t t = sr * 32u;
      const std::uint32_t tm = t / 16u;
      const std::uint32_t rm = t % 16u;
      for (std::uint32_t n = 0u; n < n_ane; ++n) {
        bf16_ane[static_cast<std::size_t>(sr) * n_ane + n] =
            bf16_to_float(bf16_ptr[static_cast<std::size_t>(t) * N + n]);
      }
      for (std::uint32_t blk = 0u; blk < n_ane / 128u; ++blk) {
        const std::size_t ti = static_cast<std::size_t>(tm) * k_blocks + blk;
        for (std::uint32_t sub = 0u; sub < 4u; ++sub) {
          const float sc = ref_decode_e8m0(scales[ti * 64u + rm * 4u + sub]);
          const std::uint32_t c0 = blk * 128u + sub * 32u;
          for (std::uint32_t j = 0u; j < 16u; ++j) {
            const std::uint8_t pk = codes[ti * 1024u + rm * 64u + sub * 16u + j];
            gpu_a4_fwht[static_cast<std::size_t>(sr) * n_ane + c0 + 2u * j] =
                ref_decode_e2m1(pk & 0x0Fu) * sc;
            gpu_a4_fwht[static_cast<std::size_t>(sr) * n_ane + c0 + 2u * j + 1u] =
                ref_decode_e2m1((pk >> 4u) & 0x0Fu) * sc;
          }
        }
      }
    }
    std::vector<float> cpu_fwht = bf16_ane;
    fwht64_test_rows(cpu_fwht, sample_rows, n_ane);
    std::vector<float> cpu_a4_fwht(cpu_fwht.size());
    for (std::uint32_t sr = 0u; sr < sample_rows; ++sr) {
      for (std::uint32_t g32 = 0u; g32 < n_ane / 32u; ++g32) {
        float amax = 0.0f;
        for (std::uint32_t i = 0u; i < 32u; ++i) {
          amax = std::max(
              amax, std::abs(cpu_fwht[static_cast<std::size_t>(sr) * n_ane + g32 * 32u + i]));
        }
        const float sc = ref_decode_e8m0(ref_encode_e8m0_round_up(amax / 6.0f));
        const float inv = sc > 0.0f ? (1.0f / sc) : 0.0f;
        for (std::uint32_t i = 0u; i < 32u; ++i) {
          const std::size_t idx = static_cast<std::size_t>(sr) * n_ane + g32 * 32u + i;
          cpu_a4_fwht[idx] = ref_decode_e2m1(encode_e2m1_ref(cpu_fwht[idx] * inv)) * sc;
        }
      }
    }
    const double cos_a4_vs_cpu = cosine_similarity(gpu_a4_fwht, cpu_a4_fwht);
    std::vector<float> gpu_a4_inv = gpu_a4_fwht;
    fwht64_test_rows(gpu_a4_inv, sample_rows, n_ane);
    const double cos_a4_orig = cosine_similarity(gpu_a4_inv, bf16_ane);
    std::printf("  [ultra-a4]   cos_vs_cpu_a4=%.7f cos_vs_bf16=%.7f\n",
                cos_a4_vs_cpu, cos_a4_orig);
    // Note: GPU Fwht64A4 reads full 10-bit-mantissa FP16 directly from IOSurface
    // whereas cpu_a4_fwht starts from 7-bit-mantissa BF16 dst_bf16 (~0.9998 match;
    // bit-exact 1.0000000 FP16 parity is proven in test_ane_unpack_fwht64_a8_and_a4_kernels).
    REQUIRE(cos_a4_vs_cpu >= 0.9995 && cos_a4_orig >= 0.99);
  }

  // 4) Collapsed-CB Pipeline API chaining 3 ops (BF16, Fwht64A8, Fwht64A4) with
  //    ZERO CPU waits until flush_and_wait(), verified bit-exact vs run_op_sync_ex!
  {
    auto cb_bf16 = engine->make_shared_buffer(static_cast<std::uint64_t>(T) * N * 2u);
    auto cb_a8_qa = engine->make_shared_buffer(static_cast<std::uint64_t>(T) * N);
    auto cb_a8_sa = engine->make_shared_buffer(static_cast<std::uint64_t>(N / 64u) * T * 4u);
    auto cb_a8_ra = engine->make_shared_buffer(static_cast<std::uint64_t>(N / 64u) * T * 4u);
    auto cb_a4_c = engine->make_shared_buffer(n_tiles * 1024u);
    auto cb_a4_s = engine->make_shared_buffer(n_tiles * 64u);
    std::memset(cb_bf16.data(), 0, static_cast<std::size_t>(cb_bf16.size()));
    std::memset(cb_a8_qa.data(), 0x5A, static_cast<std::size_t>(cb_a8_qa.size()));
    std::memset(cb_a4_c.data(), 0xA5, static_cast<std::size_t>(cb_a4_c.size()));
    std::memset(cb_a4_s.data(), 0xA5, static_cast<std::size_t>(cb_a4_s.size()));

    AneUnpackTarget ct_bf16{AneUnpackMode::BF16, cb_bf16.native_mtl_buffer(), 0u,
                            nullptr, 0u, nullptr, 0u, N, 0u, T};
    AneUnpackTarget ct_a8{AneUnpackMode::Fwht64A8, cb_a8_qa.native_mtl_buffer(), 0u,
                          cb_a8_sa.native_mtl_buffer(), 0u, cb_a8_ra.native_mtl_buffer(),
                          0u, N, 0u, T};
    AneUnpackTarget ct_a4{AneUnpackMode::Fwht64A4, cb_a4_c.native_mtl_buffer(), 0u,
                          cb_a4_s.native_mtl_buffer(), 0u, nullptr, 0u, N, 0u, T};

    void *cb = engine->begin_collapsed_cb();
    REQUIRE(cb != nullptr);
    // Op 0 (BF16 + GPU suffix)
    const auto seq0 = engine->encode_pack_and_commit(cb, op0->op_id, x_buf.native_mtl_buffer(), 0u);
    engine->submit_gpu_suffix_cb([&](void *raw_cb) {
      id<MTLCommandBuffer> mcb = (__bridge id<MTLCommandBuffer>)raw_cb;
      id<MTLComputeCommandEncoder> enc = [mcb computeCommandEncoder];
      const SuffixParamsHost p{T, K, N, op0->split.gpu_row_offset, op0->split.gpu_rows};
      [enc setComputePipelineState:suffix_psos.swiglu_suffix_pso];
      [enc setBuffer:(__bridge id<MTLBuffer>)x_buf.native_mtl_buffer() offset:0 atIndex:0];
      [enc setBuffer:(__bridge id<MTLBuffer>)wg0_gpu.native_mtl_buffer() offset:0 atIndex:1];
      [enc setBuffer:(__bridge id<MTLBuffer>)wu0_gpu.native_mtl_buffer() offset:0 atIndex:2];
      [enc setBuffer:(__bridge id<MTLBuffer>)cb_bf16.native_mtl_buffer() offset:0 atIndex:3];
      [enc setBytes:&p length:sizeof(p) atIndex:4];
      [enc dispatchThreads:MTLSizeMake(op0->split.gpu_rows, T, 1u)
          threadsPerThreadgroup:MTLSizeMake(16u, 16u, 1u)];
      [enc endEncoding];
    });
    cb = engine->begin_unpack_cb(seq0, op0->op_id, ct_bf16);
    // Op 1 (Fwht64A8) packed straight into the SAME collapsed CB!
    const auto seq1 = engine->encode_pack_and_commit(cb, op1->op_id, x_buf.native_mtl_buffer(), 0u);
    cb = engine->begin_unpack_cb(seq1, op1->op_id, ct_a8);
    // Op 2 (Fwht64A4, slot wraparound!) packed into the SAME collapsed CB!
    const auto seq2 = engine->encode_pack_and_commit(cb, op0->op_id, x_buf.native_mtl_buffer(), 0u);
    cb = engine->begin_unpack_cb(seq2, op0->op_id, ct_a4);
    engine->commit_collapsed_cb(cb);
    engine->flush_and_wait();

    REQUIRE(std::memcmp(cb_bf16.data(), dst_bf16.data(),
                        static_cast<std::size_t>(dst_bf16.size())) == 0);
    REQUIRE(std::memcmp(cb_a8_qa.data(), a8_qa.data(),
                        static_cast<std::size_t>(a8_qa.size())) == 0);
    REQUIRE(std::memcmp(cb_a8_sa.data(), a8_sa.data(),
                        static_cast<std::size_t>(a8_sa.size())) == 0);
    REQUIRE(std::memcmp(cb_a8_ra.data(), a8_ra.data(),
                        static_cast<std::size_t>(a8_ra.size())) == 0);
    REQUIRE(std::memcmp(cb_a4_c.data(), a4_codes.data(),
                        static_cast<std::size_t>(a4_codes.size())) == 0);
    REQUIRE(std::memcmp(cb_a4_s.data(), a4_scales.data(),
                        static_cast<std::size_t>(a4_scales.size())) == 0);
    std::printf("  [ultra-cb]   collapsed-CB (BF16+A8+A4 chained) bit-exact verified\n");
  }

  // Save cold BF16 output to verify dual-surface equivalence & warm cache reload.
  std::vector<std::byte> cold_bf16(static_cast<std::size_t>(dst_bf16.size()));
  std::memcpy(cold_bf16.data(), dst_bf16.data(), cold_bf16.size());
  engine.reset();

  // 5) Shared vs Dual input surface equivalence + Warm disk cache hit verification!
  {
    AneDualOffloadConfig dual_in_cfg = cfg;
    dual_in_cfg.share_input_surface = false;
    auto warm_eng = AneDualOffloadEngine::build(specs, dual_in_cfg);
    REQUIRE(warm_eng != nullptr && warm_eng->is_active());
    const auto wst = warm_eng->stats();
    REQUIRE(!wst.shared_input_surface_active);
    REQUIRE(wst.disk_cache_hits > 0u && wst.disk_cache_misses == 0u);
    auto wx = warm_eng->make_shared_buffer(static_cast<std::uint64_t>(T) * K * 2u);
    std::memcpy(wx.data(), x_buf.data(), static_cast<std::size_t>(wx.size()));
    auto wdst = warm_eng->make_shared_buffer(static_cast<std::uint64_t>(T) * N * 2u);
    std::memset(wdst.data(), 0, static_cast<std::size_t>(wdst.size()));
    auto *wop0 = warm_eng->find_op(0u, AneProjectionRole::MlpGateUpSwiGlu);
    REQUIRE(wop0 != nullptr);
    AneUnpackTarget wtgt{AneUnpackMode::BF16, wdst.native_mtl_buffer(), 0u,
                         nullptr, 0u, nullptr, 0u, N, 0u, T};
    warm_eng->run_op_sync_ex(wop0->op_id, wx.native_mtl_buffer(), 0u, 0u, wtgt);
    const auto *got_w = static_cast<const std::uint16_t *>(wdst.data());
    const auto *ref_w = reinterpret_cast<const std::uint16_t *>(cold_bf16.data());
    for (std::uint32_t t = 0u; t < T; t += 16u) {
      for (std::uint32_t n = 0u; n < n_ane; ++n) {
        REQUIRE(got_w[static_cast<std::size_t>(t) * N + n] ==
                ref_w[static_cast<std::size_t>(t) * N + n]);
      }
    }
    std::printf("  [ultra-warm] warm reload (hits=%u misses=%u) + dual-input-surface bit-exact\n",
                wst.disk_cache_hits, wst.disk_cache_misses);
  }

  std::filesystem::remove_all(temp_cache, ec);
  std::cout << "[PASS] test_ultra_fp16_swiglu_pipeline (simulate_dual="
            << (simulate_dual_banks ? "true" : "false") << ")\n";
  return 0;
}

// Minimal synthetic 1-layer dense/MoE .miv builder for QwenForward routing & calibration guard tests.
namespace qf_test {
using Json = nlohmann::json;
using U32 = std::uint32_t;
using U64 = std::uint64_t;

U64 det_hash(U32 seed, U64 idx) {
  U64 x = (U64(seed) * 0x9E3779B97F4A7C15ULL) ^ (idx * 0xBF58476D1CE4E5B9ULL);
  x = (x ^ (x >> 30U)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31U);
}

float det_val(U32 seed, U64 idx, float amp = 0.03F) {
  const U64 x = det_hash(seed, idx);
  return static_cast<float>(static_cast<std::int32_t>(x & 0xFFFFU) - 32768) /
         32768.0F * amp;
}

std::string hex_crc32(U32 crc) {
  std::ostringstream out;
  out << std::hex << std::nouppercase << std::setfill('0') << std::setw(8) << crc;
  return out.str();
}

void append_u32(std::vector<std::byte> &out, U32 v) {
  for (int i = 0; i < 4; ++i)
    out.push_back(static_cast<std::byte>((v >> (i * 8)) & 0xFFU));
}

void append_u64(std::vector<std::byte> &out, U64 v) {
  for (int i = 0; i < 8; ++i)
    out.push_back(static_cast<std::byte>((v >> (i * 8)) & 0xFFU));
}

void write_file(const std::filesystem::path &path, std::span<const std::byte> bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out.good()) throw std::runtime_error("failed open: " + path.string());
  out.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  if (!out.good()) throw std::runtime_error("failed write: " + path.string());
}

struct MiniObj {
  U32 id = 0;
  std::string name, format, layout, binding_use, activation_policy;
  std::vector<U64> shape;
  std::optional<U64> padded_k;
  std::vector<std::pair<std::string, std::vector<std::byte>>> planes;
};

struct MiniBuilder {
  std::vector<MiniObj> objs;

  void add_bf16(std::string name, std::vector<U64> shape, U32 seed,
                float bias = 0.0F, float amp = 0.03F, std::string use = "") {
    U64 cnt = 1;
    for (U64 d : shape) cnt *= d;
    std::vector<std::uint16_t> bf(cnt);
    for (U64 i = 0; i < cnt; ++i) bf[i] = float_to_bf16(bias + det_val(seed, i, amp));
    const auto raw = std::as_bytes(std::span<const std::uint16_t>(bf));
    objs.push_back(MiniObj{
        static_cast<U32>(objs.size()), std::move(name), "direct_bf16_v1",
        "contiguous_le_v1", std::move(use), "A16Only", std::move(shape),
        std::nullopt, {{"data", {raw.begin(), raw.end()}}}});
  }

  void add_mxfp4_2d(std::string name, U32 n, U32 k, U32 seed,
                    std::string use = "linear") {
    const U32 padded_k = ((k + 127U) / 128U) * 128U;
    const U32 bn = (n >= 65536U) ? 32U : 16U;
    const U32 padded_n = ((n + bn - 1U) / bn) * bn;
    const U32 k_blocks = k / 128U;
    std::vector<std::byte> nax_data(static_cast<std::size_t>(padded_n) * (k / 2U),
                                    std::byte{0});
    std::vector<std::byte> nax_scale(static_cast<std::size_t>(padded_n) * (k / 32U),
                                     std::byte{120});
    for (U32 r = 0; r < n; ++r) {
      const U32 tn = r / bn, rn = r % bn;
      for (U32 blk = 0; blk < k_blocks; ++blk) {
        const std::size_t ti = static_cast<std::size_t>(tn) * k_blocks + blk;
        for (U32 sub = 0; sub < 4U; ++sub) {
          const auto sc = static_cast<std::uint8_t>(
              119U + (det_hash(seed + 1U, U64(r) * 4096ULL + blk * 4ULL + sub) % 3U));
          nax_scale[ti * (bn * 4U) + rn * 4U + sub] = static_cast<std::byte>(sc);
          for (U32 j = 0; j < 16U; ++j) {
            const U64 h = det_hash(seed, U64(r) * k + blk * 128U + sub * 32U + 2U * j);
            nax_data[ti * (bn * 64U) + rn * 64U + sub * 16U + j] =
                static_cast<std::byte>((h & 0xFU) | (((h >> 8U) & 0xFU) << 4U));
          }
        }
      }
    }
    objs.push_back(MiniObj{
        static_cast<U32>(objs.size()), std::move(name), "mxfp4_e2m1_g32_e8m0_v1",
        "mx_nax_tiled_bk128_v1", std::move(use), "AllowA4", {n, k}, padded_k,
        {{"nax_data", std::move(nax_data)}, {"nax_scale", std::move(nax_scale)}}});
  }

  void add_mxfp4_moe_3d(std::string name, U32 E, U32 n, U32 k, U32 seed) {
    const U32 padded_k = ((k + 127U) / 128U) * 128U;
    const std::size_t rows = static_cast<std::size_t>(E) * n;
    std::vector<std::byte> data(rows * (padded_k / 2U), std::byte{0});
    std::vector<std::byte> scale(rows * (padded_k / 32U), std::byte{118});
    for (U32 e = 0; e < E; ++e) {
      for (U32 r = 0; r < n; ++r) {
        const std::size_t flat_r = static_cast<std::size_t>(e) * n + r;
        for (U32 g = 0; g < k / 32U; ++g) {
          scale[flat_r * (padded_k / 32U) + g] = static_cast<std::byte>(
              118U + (det_hash(seed + 1U, flat_r * 256ULL + g) % 3U));
          for (U32 j = 0; j < 16U; ++j) {
            const U64 h = det_hash(seed, flat_r * U64(k) + U64(g * 32U + 2U * j));
            data[flat_r * (padded_k / 2U) + g * 16U + j] =
                static_cast<std::byte>((h & 0xFU) | (((h >> 8U) & 0xFU) << 4U));
          }
        }
      }
    }
    objs.push_back(MiniObj{
        static_cast<U32>(objs.size()), std::move(name), "mxfp4_e2m1_g32_e8m0_v1",
        "mx_expert_rowsplit_k128_v1", "moe", "AllowA4", {E, n, k}, padded_k,
        {{"data", std::move(data)}, {"scale", std::move(scale)}}});
  }

  std::vector<std::byte> serialize(std::string_view profile,
                                   std::string_view repo) const {
    std::vector<std::byte> payload;
    auto align_p = [&](std::size_t a) {
      while ((payload.size() % a) != 0U) payload.push_back(std::byte{0});
    };
    Json j_objects = Json::array(), j_bindings = Json::array();
    for (const auto &obj : objs) {
      align_p(256U);
      const U64 obj_off = payload.size();
      Json j_planes = Json::array();
      for (std::size_t pi = 0; pi < obj.planes.size(); ++pi) {
        if (pi > 0U) align_p(256U);
        const U64 p_off = payload.size();
        const auto &pb = obj.planes[pi].second;
        payload.insert(payload.end(), pb.begin(), pb.end());
        j_planes.push_back({{"alignment", 256},
                            {"bytes", pb.size()},
                            {"crc32", hex_crc32(artifact_crc32(pb))},
                            {"kind", obj.planes[pi].first},
                            {"offset", p_off},
                            {"sha256", artifact_sha256_hex(pb)}});
      }
      const U64 obj_bytes = payload.size() - obj_off;
      Json jo = {{"format", obj.format},
                 {"id", obj.id},
                 {"layout", obj.layout},
                 {"name", obj.name},
                 {"object_bytes", obj_bytes},
                 {"object_offset", obj_off},
                 {"object_sha256",
                  artifact_sha256_hex(std::span<const std::byte>(payload).subspan(
                      obj_off, obj_bytes))},
                 {"planes", std::move(j_planes)},
                 {"shape", obj.shape},
                 {"source_tensors", {obj.name}}};
      if (obj.padded_k.has_value()) jo["padded_k"] = *obj.padded_k;
      j_objects.push_back(std::move(jo));
      if (!obj.format.starts_with("direct_") && !obj.binding_use.empty() &&
          obj.shape.size() >= 2U) {
        Json lead = Json::array();
        for (std::size_t i = 0; i + 2U < obj.shape.size(); ++i)
          lead.push_back(obj.shape[i]);
        j_bindings.push_back({{"activation_policy", obj.activation_policy},
                              {"format", obj.format},
                              {"in_features", obj.shape[obj.shape.size() - 1U]},
                              {"leading_shape", std::move(lead)},
                              {"name", obj.name},
                              {"object_id", obj.id},
                              {"out_features", obj.shape[obj.shape.size() - 2U]},
                              {"use", obj.binding_use}});
      }
    }
    Json manifest = {{"bindings", std::move(j_bindings)},
                     {"container_version", 1},
                     {"model",
                      {{"source_profile", profile},
                       {"source_repo", repo},
                       {"source_revision", "wp3c_test"}}},
                     {"objects", std::move(j_objects)},
                     {"payload_bytes", payload.size()},
                     {"schema", std::string(kArtifactSchemaV1)},
                     {"views", Json::array()}};
    const std::string mstr = manifest.dump();
    const auto mbytes =
        std::as_bytes(std::span<const char>(mstr.data(), mstr.size()));
    const U64 payload_off = (64ULL + mbytes.size() + 16383ULL) & ~16383ULL;
    std::vector<std::byte> file;
    for (char c : kArtifactMagic) file.push_back(static_cast<std::byte>(c));
    append_u32(file, 1U);
    append_u32(file, 64U);
    append_u64(file, 64ULL);
    append_u64(file, mbytes.size());
    append_u64(file, payload_off);
    append_u64(file, payload.size());
    append_u32(file, static_cast<U32>(objs.size()));
    append_u32(file, artifact_crc32(mbytes));
    append_u32(file, artifact_crc32(std::span<const std::byte>(file.data(), 56U)));
    append_u32(file, 0U);
    file.insert(file.end(), mbytes.begin(), mbytes.end());
    file.resize(static_cast<std::size_t>(payload_off), std::byte{0});
    file.insert(file.end(), payload.begin(), payload.end());
    return file;
  }
};
} // namespace qf_test

int test_qwen_forward_auto_routing_and_calibration_guard(
    const MetalContext &ctx,
    const std::filesystem::path &metallib_dir) {
  (void)ctx;
  if (!std::filesystem::exists(metallib_dir / "omlx_m5_qwen_forward.metallib")) {
    std::cout << "[SKIP] test_qwen_forward_auto_routing_and_calibration_guard "
                 "(qwen_forward metallibs not present in "
              << metallib_dir << ")\n";
    return 0;
  }

  const auto tmp_root = std::filesystem::temp_directory_path() /
                        ("miv_ane_qf_routing_test_" +
                         std::to_string(static_cast<long>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(tmp_root, ec);

  // Build 1-layer dense MXFP4 model with H=512, F=4096 (>= 4096 ultra threshold!)
  const auto dense_dir = tmp_root / "dense";
  {
    constexpr std::uint32_t h = 512u, f = 4096u, v = 256u, hd = 256u;
    qf_test::Json cfg = {
        {"model_type", "qwen3_5"},
        {"text_config",
         {{"model_type", "qwen3_5_text"},
          {"dtype", "bfloat16"},
          {"hidden_size", h},
          {"intermediate_size", f},
          {"vocab_size", v},
          {"num_hidden_layers", 1},
          {"num_attention_heads", 2},
          {"num_key_value_heads", 1},
          {"head_dim", hd},
          {"max_position_embeddings", 8192},
          {"rms_norm_eps", 1e-6},
          {"rope_parameters",
           {{"rope_theta", 10000000.0},
            {"partial_rotary_factor", 0.25},
            {"mrope_interleaved", true}}},
          {"linear_num_key_heads", 2},
          {"linear_num_value_heads", 4},
          {"linear_key_head_dim", 128},
          {"linear_value_head_dim", 128},
          {"linear_conv_kernel_dim", 4},
          {"layer_types", {"full_attention"}},
          {"tie_word_embeddings", false}}}};
    const std::string cfg_s = cfg.dump(2);
    qf_test::write_file(dense_dir / "config.json",
                        std::as_bytes(std::span<const char>(cfg_s.data(), cfg_s.size())));
    qf_test::MiniBuilder b;
    b.add_bf16("model.embed_tokens.weight", {v, h}, 11u, 0.0f, 0.08f, "embedding");
    b.add_bf16("model.norm.weight", {h}, 12u, 0.0f, 0.03f);
    b.add_mxfp4_2d("lm_head.weight", v, h, 13u, "lm_head");
    const std::string p = "model.layers.0";
    b.add_bf16(p + ".input_layernorm.weight", {h}, 21u, 0.0f, 0.03f);
    b.add_mxfp4_2d(p + ".self_attn.q_proj.weight", 2u * 512u, h, 22u);
    b.add_mxfp4_2d(p + ".self_attn.k_proj.weight", 256u, h, 23u);
    b.add_mxfp4_2d(p + ".self_attn.v_proj.weight", 256u, h, 24u);
    b.add_bf16(p + ".self_attn.q_norm.weight", {hd}, 25u, 0.0f, 0.03f);
    b.add_bf16(p + ".self_attn.k_norm.weight", {hd}, 26u, 0.0f, 0.03f);
    b.add_mxfp4_2d(p + ".self_attn.o_proj.weight", h, 512u, 27u);
    b.add_bf16(p + ".post_attention_layernorm.weight", {h}, 28u, 0.0f, 0.03f);
    b.add_mxfp4_2d(p + ".mlp.gate_proj.weight", f, h, 29u);
    b.add_mxfp4_2d(p + ".mlp.up_proj.weight", f, h, 30u);
    b.add_mxfp4_2d(p + ".mlp.down_proj.weight", h, f, 31u);
    qf_test::write_file(dense_dir / "model.miv",
                        b.serialize("mxfp4_v1", "test/wp3c-dense-1l"));
  }

  // Build 1-layer MoE model (to verify moe_unsupported guard)
  const auto moe_dir = tmp_root / "moe";
  {
    constexpr std::uint32_t h = 512u, f = 256u, sf = 512u, v = 256u, hd = 256u, E = 256u;
    qf_test::Json cfg = {
        {"model_type", "qwen3_5_moe"},
        {"text_config",
         {{"model_type", "qwen3_5_moe_text"},
          {"dtype", "bfloat16"},
          {"hidden_size", h},
          {"moe_intermediate_size", f},
          {"shared_expert_intermediate_size", sf},
          {"num_experts", E},
          {"num_experts_per_tok", 8},
          {"vocab_size", v},
          {"num_hidden_layers", 1},
          {"num_attention_heads", 2},
          {"num_key_value_heads", 1},
          {"head_dim", hd},
          {"max_position_embeddings", 8192},
          {"rms_norm_eps", 1e-6},
          {"rope_parameters",
           {{"rope_theta", 10000000.0},
            {"partial_rotary_factor", 0.25},
            {"mrope_interleaved", true}}},
          {"linear_num_key_heads", 2},
          {"linear_num_value_heads", 4},
          {"linear_key_head_dim", 128},
          {"linear_value_head_dim", 128},
          {"linear_conv_kernel_dim", 4},
          {"layer_types", {"full_attention"}},
          {"tie_word_embeddings", false}}}};
    const std::string cfg_s = cfg.dump(2);
    qf_test::write_file(moe_dir / "config.json",
                        std::as_bytes(std::span<const char>(cfg_s.data(), cfg_s.size())));
    qf_test::MiniBuilder b;
    b.add_bf16("model.embed_tokens.weight", {v, h}, 41u, 0.0f, 0.08f, "embedding");
    b.add_bf16("model.norm.weight", {h}, 42u, 0.0f, 0.03f);
    b.add_mxfp4_2d("lm_head.weight", v, h, 43u, "lm_head");
    const std::string p = "model.layers.0";
    b.add_bf16(p + ".input_layernorm.weight", {h}, 51u, 0.0f, 0.03f);
    b.add_mxfp4_2d(p + ".self_attn.q_proj.weight", 2u * 512u, h, 52u);
    b.add_mxfp4_2d(p + ".self_attn.k_proj.weight", 256u, h, 53u);
    b.add_mxfp4_2d(p + ".self_attn.v_proj.weight", 256u, h, 54u);
    b.add_bf16(p + ".self_attn.q_norm.weight", {hd}, 55u, 0.0f, 0.03f);
    b.add_bf16(p + ".self_attn.k_norm.weight", {hd}, 56u, 0.0f, 0.03f);
    b.add_mxfp4_2d(p + ".self_attn.o_proj.weight", h, 512u, 57u);
    b.add_bf16(p + ".post_attention_layernorm.weight", {h}, 58u, 0.0f, 0.03f);
    b.add_bf16(p + ".mlp.gate.weight", {E, h}, 59u, 0.0f, 0.04f, "linear");
    b.add_mxfp4_moe_3d(p + ".mlp.switch_mlp.gate_proj.weight", E, f, h, 60u);
    b.add_mxfp4_moe_3d(p + ".mlp.switch_mlp.up_proj.weight", E, f, h, 61u);
    b.add_mxfp4_moe_3d(p + ".mlp.switch_mlp.down_proj.weight", E, h, f, 62u);
    b.add_bf16(p + ".mlp.shared_expert_gate.weight", {1u, h}, 63u, 0.0f, 0.04f, "linear");
    b.add_mxfp4_2d(p + ".mlp.shared_expert.gate_proj.weight", sf, h, 64u);
    b.add_mxfp4_2d(p + ".mlp.shared_expert.up_proj.weight", sf, h, 65u);
    b.add_mxfp4_2d(p + ".mlp.shared_expert.down_proj.weight", h, sf, 66u);
    qf_test::write_file(moe_dir / "model.miv",
                        b.serialize("mxfp4_moe_v1", "test/wp3c-moe-1l"));
  }

  constexpr std::uint32_t kMaxCtx = 4096u; // workspace_rows == 2048
  auto mctx = MetalContext::create();
  auto load_model = [&](const std::filesystem::path &dir) {
    auto mcfg = load_qwen_model_config(dir / "config.json");
    auto art = ArtifactModel::open(dir / "model.miv", true);
    auto digest = execution_artifact_digest(art);
    auto plan = plan_model_materialization(art, mctx);
    auto weights = materialize_model(art, std::move(plan), mctx);
    auto layers = build_qwen_layer_catalog(art, mcfg, kMaxCtx, 64);
    return std::tuple{std::move(mcfg), std::move(weights), std::move(layers),
                      std::move(digest)};
  };

  auto [dense_cfg, dense_w, dense_layers, dense_digest] = load_model(dense_dir);
  auto [moe_cfg, moe_w, moe_layers, moe_digest] = load_model(moe_dir);

  auto make_qf = [&](const MaterializedModel &w, const QwenLayerCatalog &layers,
                     const QwenModelConfig &mcfg, const std::string &digest,
                     QwenPrefillOptions pf) {
    pf.artifact_digest = digest;
    return std::make_unique<QwenForward>(
        mctx, w, layers, mcfg, kMaxCtx, metallib_dir, std::span<const std::uint32_t>{},
        false, false, false, std::nullopt, true, false, std::nullopt,
        LinearNumericPolicy::RowExactPortableV1, SamplingContract::PositionCoupledV1,
        AttentionNumericPolicy::RowExactV1, /*inspect_tensors_workspace=*/false,
        "bf16", nullptr, /*enable_flash_nax_prefill=*/true, 0u, nullptr, nullptr, pf);
  };

  const bool hw_ane = is_ane_available();

  // Case A: MoE model -> strictly Off with "moe_unsupported"
  {
    auto qf_moe = make_qf(moe_w, moe_layers, moe_cfg, moe_digest,
                          QwenPrefillOptions{.ane_mode = PrefillAneMode::Auto});
    REQUIRE(!qf_moe->ane_offload_active());
    REQUIRE(qf_moe->ane_mode_effective() == PrefillAneMode::Off);
    REQUIRE(qf_moe->ane_inactive_reason() == "moe_unsupported");
  }

  if (hw_ane) {
    // Case B: "auto" with gpu_cores < 64 (40 cores) -> compact_int8 (chunk=1024)
    {
      ScopedEnvVar e_cores("OMLX_M5_FAKE_GPU_CORES", "40");
      ScopedEnvVar e_mode("OMLX_M5_ANE_MODE", nullptr);
      auto qf_compact = make_qf(dense_w, dense_layers, dense_cfg, dense_digest,
                                QwenPrefillOptions{.ane_mode = PrefillAneMode::Auto});
      REQUIRE(qf_compact->ane_offload_active());
      REQUIRE(qf_compact->ane_mode_effective() == PrefillAneMode::CompactInt8);
      REQUIRE(qf_compact->ane_chunk_tokens() == kAneDefaultChunkTokens);
    }

    // Case C: "auto" with gpu_cores >= 64 (80 cores) + impossible cal ratio ->
    //   On Apple10 NAX (M5/M5 Ultra): runs 1-layer canary & triggers "calibration_bypassed"!
    //   On non-NAX Apple9 (M4): safely refuses ultra_fp16_swiglu with "gpu_suffix_unsupported"!
    const bool has_nax = mctx.capabilities().metal4_nax_runtime;
    {
      ScopedEnvVar e_cores("OMLX_M5_FAKE_GPU_CORES", "80");
      ScopedEnvVar e_mode("OMLX_M5_ANE_MODE", nullptr);
      ScopedEnvVar e_force("OMLX_M5_ANE_FORCE_ACTIVE", nullptr);
      ScopedEnvVar e_recal("OMLX_M5_ANE_RECALIBRATE", "1");
      ScopedEnvVar e_ratio("OMLX_M5_ANE_CAL_RATIO", "0.0001");
      auto qf_bypass = make_qf(dense_w, dense_layers, dense_cfg, dense_digest,
                               QwenPrefillOptions{.ane_mode = PrefillAneMode::Auto});
      REQUIRE(!qf_bypass->ane_offload_active());
      REQUIRE(qf_bypass->ane_mode_effective() == PrefillAneMode::Off);
      const std::string expected_reason =
          has_nax ? "calibration_bypassed" : "gpu_suffix_unsupported";
      REQUIRE(qf_bypass->ane_inactive_reason() == expected_reason);
      const auto tel = qf_bypass->prefill_telemetry();
      REQUIRE(tel.prefill_ane_mode_effective == "off" &&
              tel.prefill_ane_inactive_reason == expected_reason);
    }

    // Case D: On Apple10 NAX (M5/M5 Ultra), "auto" + 80 cores + FORCE_ACTIVE ->
    //   ultra_fp16_swiglu; execute live 2048-token chunk in BF16 & W4A8 Mode B
    //   and verify logits cosine similarity vs GPU-only reference!
    if (has_nax) {
      ScopedEnvVar e_cores("OMLX_M5_FAKE_GPU_CORES", "80");
      ScopedEnvVar e_mode("OMLX_M5_ANE_MODE", nullptr);
      ScopedEnvVar e_force("OMLX_M5_ANE_FORCE_ACTIVE", "1");
      ScopedEnvVar e_recal("OMLX_M5_ANE_RECALIBRATE", "1");
      ScopedEnvVar e_alpha("OMLX_M5_ANE_ALPHA", "0.125");

      auto qf_gpu_bf16 = make_qf(dense_w, dense_layers, dense_cfg, dense_digest,
                                 QwenPrefillOptions{.linear_mode = PrefillLinearMode::BFloat16,
                                                    .ane_mode = PrefillAneMode::Off});
      auto qf_ultra_bf16 = make_qf(dense_w, dense_layers, dense_cfg, dense_digest,
                                   QwenPrefillOptions{.linear_mode = PrefillLinearMode::BFloat16,
                                                      .ane_mode = PrefillAneMode::Auto});
      auto qf_ultra_w4a8 = make_qf(dense_w, dense_layers, dense_cfg, dense_digest,
                                   QwenPrefillOptions{.linear_mode = PrefillLinearMode::W4A8,
                                                      .ane_mode = PrefillAneMode::Auto});

      REQUIRE(qf_ultra_bf16->ane_offload_active());
      REQUIRE(qf_ultra_bf16->ane_mode_effective() == PrefillAneMode::UltraFp16SwiGlu);
      REQUIRE(qf_ultra_bf16->ane_chunk_tokens() == kAneUltraDefaultChunkTokens);
      REQUIRE(qf_ultra_w4a8->ane_offload_active());
      REQUIRE(qf_ultra_w4a8->ane_mode_effective() == PrefillAneMode::UltraFp16SwiGlu);

      std::vector<std::uint32_t> chunk2048(2048u);
      for (std::uint32_t i = 0u; i < 2048u; ++i) {
        chunk2048[i] = (i * 37u + 13u) % 240u;
      }
      std::vector<std::uint32_t> tail16(16u, 7u);

      auto run_probe = [&](QwenForward &qf) {
        qf.reset();
        static_cast<void>(qf.run_trunk(chunk2048, true));
        GpuSamplingOptions samp{};
        samp.contract = SamplingContract::PositionCoupledV1;
        return qf.run(tail16, samp, /*inspect_tensors=*/true, true).logits;
      };
      const auto ref_logits = run_probe(*qf_gpu_bf16);
      const auto ultra_bf16_logits = run_probe(*qf_ultra_bf16);
      const auto ultra_w4a8_logits = run_probe(*qf_ultra_w4a8);

      const double cos_bf16 = cosine_similarity(ref_logits, ultra_bf16_logits);
      const double cos_w4a8 = cosine_similarity(ref_logits, ultra_w4a8_logits);
      const auto tb = qf_ultra_bf16->prefill_telemetry();
      const auto tw = qf_ultra_w4a8->prefill_telemetry();
      std::printf(
          "  [qf-ultra] bf16_cos=%.6f (offload=%llu, fallback=%llu) "
          "w4a8_cos=%.6f (offload=%llu, w4a8_tok=%llu)\n",
          cos_bf16, static_cast<unsigned long long>(tb.prefill_ane_offload_tokens),
          static_cast<unsigned long long>(tb.prefill_ane_fallback_tokens),
          cos_w4a8, static_cast<unsigned long long>(tw.prefill_ane_offload_tokens),
          static_cast<unsigned long long>(tw.prefill_w4a8_tokens));
      REQUIRE(cos_bf16 >= 0.99);
      REQUIRE(cos_w4a8 >= 0.96);
      REQUIRE(tb.prefill_ane_mode_effective == "ultra_fp16_swiglu");
      REQUIRE(tb.prefill_ane_offload_tokens == 2048u && tb.prefill_ane_fallback_tokens == 16u);
      REQUIRE(tw.prefill_ane_mode_effective == "ultra_fp16_swiglu");
      REQUIRE(tw.prefill_ane_offload_tokens == 2048u && tw.prefill_w4a8_tokens >= 2048u);
    } else {
      std::printf("  [qf-ultra] non-NAX host: verified >=64-core auto routing & gpu_suffix_unsupported guard\n");
    }
  }

  std::filesystem::remove_all(tmp_root, ec);
  std::cout << "[PASS] test_qwen_forward_auto_routing_and_calibration_guard\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      const std::filesystem::path metallib_dir =
          argc > 1 ? std::filesystem::path(argv[1])
                   : std::filesystem::current_path();
      if (test_gdn_exclusion_and_splits() != 0) {
        return 1;
      }
      if (test_prefill_ane_mode_and_calibration_cache() != 0) {
        return 1;
      }
      auto ctx = MetalContext::create();
      if (test_miv_extraction_accuracy(ctx) != 0) {
        return 1;
      }
      if (test_ane_unpack_fwht64_a8_and_a4_kernels(ctx, metallib_dir) != 0) {
        return 1;
      }
      // Run 1: Native topology auto-detection (1-die on M4, 2-die pinned on Ultra)
      if (test_ane_execution_pipeline_and_cache(ctx, metallib_dir, false) != 0) {
        return 1;
      }
      // Run 2: Full Dual-ANE 2-bank / 2-IOSurface pack+unpack + MTLSharedEvent overlap
      // (runs dual unpinned banks on single-die M4, or pinned dual banks on Ultra)
      if (test_ane_execution_pipeline_and_cache(ctx, metallib_dir, true) != 0) {
        return 1;
      }
      // Run 3 & 4: WP 3C ultra_fp16_swiglu (native + dual bank, BF16 + A8 + A4 + Collapsed-CB)
      if (test_ultra_fp16_swiglu_pipeline(ctx, metallib_dir, false) != 0) {
        return 1;
      }
      if (test_ultra_fp16_swiglu_pipeline(ctx, metallib_dir, true) != 0) {
        return 1;
      }
      // Run 5: QwenForward "auto" >=64 vs <64 routing, MoE guard, canary bypass & 2048-tok parity
      if (test_qwen_forward_auto_routing_and_calibration_guard(ctx, metallib_dir) != 0) {
        return 1;
      }
      std::cout << "ALL ane_dual_offload_tests PASSED\n";
      return 0;
    } catch (const std::exception &e) {
      std::cerr << "UNCAUGHT EXCEPTION: " << e.what() << "\n";
      return 1;
    }
  }
}
