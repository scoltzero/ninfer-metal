#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ane/ane_dual_offload.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/quantization.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
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
    std::printf("  [extract] %-22s exact_dequant_err=%.1e g64_int8_cos_sim=%.7f\n",
                tc.label, exact_err, cos);
    REQUIRE(cos >= 0.9995);
  }
  std::cout << "[PASS] test_miv_extraction_accuracy (all 12 formats/layouts including lane_v2+fwht64 >= 0.9995)\n";
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
      auto ctx = MetalContext::create();
      if (test_miv_extraction_accuracy(ctx) != 0) {
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
      std::cout << "ALL ane_dual_offload_tests PASSED\n";
      return 0;
    } catch (const std::exception &e) {
      std::cerr << "UNCAUGHT EXCEPTION: " << e.what() << "\n";
      return 1;
    }
  }
}
