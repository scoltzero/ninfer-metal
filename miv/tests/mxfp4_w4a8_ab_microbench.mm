// WP 3A M5 Ultra Live Microbenchmark & Accuracy/Spill Audit:
// Compares Baseline Phase-2 128t Ws_i8 vs Candidate alpha (256t/512t Zero-Ws Shift-Merge)
// vs Candidate beta (256t Native MXFP8 x MXFP8 -> float) vs Native W4A4 across Qwen3.8-27B shapes.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

struct BenchGemmParams {
  std::uint32_t n;
  std::uint32_t k;
  std::uint32_t mp;
  std::uint32_t rows;
  std::uint32_t padded_n;
  std::uint32_t row_offset;
  std::uint32_t out_stride;
  std::uint32_t reserved;
};

struct BenchQuantParams {
  std::uint32_t k;
  std::uint32_t mp;
  std::uint32_t reserved;
  std::uint32_t rows;
};

namespace {

struct AgxSpillInfo {
  bool available = false;
  std::uint32_t gpr_count = 0;
  std::uint32_t spill_bytes = 0;
};

AgxSpillInfo inspect_agx(id<MTLComputePipelineState> pso) {
  AgxSpillInfo info{};
  Class cls = object_getClass(pso);
  while (cls != Nil) {
    Ivar iv = class_getInstanceVariable(cls, "_impl");
    if (iv != nullptr) {
      const char *enc = ivar_getTypeEncoding(iv);
      if (enc != nullptr && std::strstr(enc, "RuntimeState") != nullptr) {
        const auto *base = reinterpret_cast<const std::uint8_t *>((__bridge const void *)pso);
        const auto *u32 = reinterpret_cast<const std::uint32_t *>(base + ivar_getOffset(iv));
        info.available = true;
        info.gpr_count = u32[42];
        info.spill_bytes = u32[45];
      }
      break;
    }
    cls = class_getSuperclass(cls);
  }
  return info;
}

inline std::uint16_t f32_to_bf16(float x) {
  std::uint32_t u = std::bit_cast<std::uint32_t>(x);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return std::uint16_t(u >> 16);
}

inline float bf16_to_f32(std::uint16_t b) {
  return std::bit_cast<float>(std::uint32_t(b) << 16);
}

inline float decode_e2m1(std::uint32_t c) {
  static constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float v = kMag[c & 7u];
  return (c & 8u) ? -v : v;
}

inline float decode_e8m0(std::uint8_t s) {
  if (s == 255u) return 0.0f;
  return std::ldexp(1.0f, s == 0 ? -127 : int(s) - 127);
}

struct AccuracyResult {
  double cos_sim = 0.0;
  float max_abs_err = 0.0f;
  float max_rel_rmse = 0.0f;
};

AccuracyResult compare_bf16_vs_ref(const std::uint16_t *got, const std::vector<float> &ref) {
  double dot = 0.0, ng = 0.0, nr = 0.0, sse = 0.0;
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float g = bf16_to_f32(got[i]);
    const float r = ref[i];
    const float diff = std::fabs(g - r);
    max_abs = std::max(max_abs, diff);
    dot += double(g) * double(r);
    ng += double(g) * double(g);
    nr += double(r) * double(r);
    sse += double(diff) * double(diff);
  }
  AccuracyResult res;
  res.cos_sim = (ng > 0.0 && nr > 0.0) ? (dot / (std::sqrt(ng) * std::sqrt(nr))) : 1.0;
  res.max_abs_err = max_abs;
  res.max_rel_rmse = (nr > 0.0) ? float(std::sqrt(sse / nr)) : 0.0f;
  return res;
}

id<MTLComputePipelineState> build_pso(id<MTLDevice> dev, id<MTLLibrary> lib, const char *name, bool icb) {
  id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
  if (!fn) {
    std::fprintf(stderr, "Missing kernel: %s\n", name);
    std::exit(1);
  }
  MTLComputePipelineDescriptor *d = [MTLComputePipelineDescriptor new];
  d.computeFunction = fn;
  d.supportIndirectCommandBuffers = icb ? YES : NO;
  NSError *err = nil;
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone reflection:nil error:&err];
  if (!pso) {
    std::fprintf(stderr, "PSO fail %s (icb=%d): %s\n", name, int(icb), err.localizedDescription.UTF8String);
    std::exit(1);
  }
  return pso;
}

// Fill MXFP4 pretiled weights [padded_n/16, k/128, 16, 64B] + scales [padded_n/16, k/128, 16, 4B].
// If stress_large_de is true, intentionally inject |de| in {4, 5, 6, 8, 12} on ~30% of rows!
void fill_mxfp4_tiled(std::uint8_t *codes, std::uint8_t *scales,
                      std::uint32_t n, std::uint32_t k, std::uint32_t seed,
                      bool stress_large_de, std::vector<float> *ref_w_out = nullptr) {
  const std::uint32_t blocks128 = k / 128u;
  const std::uint32_t tiles_n = (n + 15u) / 16u;
  std::memset(codes, 0, std::size_t(tiles_n) * blocks128 * 1024u);
  std::memset(scales, 127, std::size_t(tiles_n) * blocks128 * 64u);
  if (ref_w_out) ref_w_out->assign(std::size_t(n) * k, 0.0f);

  std::uint32_t st = seed * 2654435761u + 17u;
  auto next_u32 = [&]() {
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    return st;
  };
  for (std::uint32_t r = 0; r < n; ++r) {
    const std::uint32_t tn = r / 16u;
    const std::uint32_t rn = r % 16u;
    for (std::uint32_t blk = 0; blk < blocks128; ++blk) {
      const std::size_t idx = std::size_t(tn) * blocks128 + blk;
      std::uint8_t *c_ptr = codes + idx * 1024u + rn * 64u;
      std::uint8_t *s_ptr = scales + idx * 64u + rn * 4u;
      for (std::uint32_t half = 0; half < 2u; ++half) {
        std::uint8_t e0 = std::uint8_t(123u + (next_u32() % 4u));
        int delta = int(next_u32() % 5u) - 2; // normal |de| <= 2
        if (stress_large_de && ((r + blk + half) % 3u == 0u)) {
          const int big[6] = {-12, -6, -4, 4, 5, 9};
          delta = big[next_u32() % 6u];
        }
        std::uint8_t e1 = std::uint8_t(std::clamp(int(e0) + delta, 0, 250));
        if (stress_large_de && r == 3u && blk == 0u && half == 0u) {
          e0 = 0u; e1 = 2u; // boundary c=0 fast path check!
        }
        s_ptr[half * 2u + 0u] = e0;
        s_ptr[half * 2u + 1u] = e1;
      }
      for (std::uint32_t byte_i = 0; byte_i < 64u; ++byte_i) {
        const std::uint32_t u = next_u32();
        const std::uint8_t lo = u & 0xFu;
        const std::uint8_t hi = (u >> 8) & 0xFu;
        c_ptr[byte_i] = std::uint8_t(lo | (hi << 4));
        if (ref_w_out) {
          const std::uint32_t c0 = blk * 128u + byte_i * 2u;
          const float sc = decode_e8m0(s_ptr[byte_i / 16u]);
          (*ref_w_out)[std::size_t(r) * k + c0 + 0u] = decode_e2m1(lo) * sc;
          (*ref_w_out)[std::size_t(r) * k + c0 + 1u] = decode_e2m1(hi) * sc;
        }
      }
    }
  }
}

void fill_bf16_acts(std::uint16_t *dst, std::uint32_t m, std::uint32_t k, std::uint32_t seed,
                    std::vector<float> *ref_x_out = nullptr) {
  if (ref_x_out) ref_x_out->resize(std::size_t(m) * k);
  std::uint32_t st = seed * 1664525u + 1013904223u;
  auto next_f = [&]() {
    // Approximate unit Gaussian via sum of 4 uniforms
    float s = 0.0f;
    for (int i = 0; i < 4; ++i) {
      st = st * 1664525u + 1013904223u;
      s += float((st >> 8) & 0xFFFFu) * (1.0f / 65536.0f) - 0.5f;
    }
    return s * 1.4f;
  };
  for (std::size_t i = 0; i < std::size_t(m) * k; ++i) {
    // Add occasional realistic activation outliers (~1%)
    float v = next_f();
    if ((i % 131u) == 0u) v *= 4.5f;
    const std::uint16_t b = f32_to_bf16(v);
    dst[i] = b;
    if (ref_x_out) (*ref_x_out)[i] = bf16_to_f32(b);
  }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2) {
      std::fprintf(stderr, "Usage: %s <mxfp4_w4a8_ab_microbench.metallib> [--quick]\n", argv[0]);
      return 2;
    }
    const bool quick = (argc >= 3 && std::string(argv[2]) == "--quick");
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    const bool apple10 = [dev supportsFamily:(MTLGPUFamily)1010];
    std::printf("Device: %s (Apple10=%d)\n", dev.name.UTF8String, int(apple10));

    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&err];
    if (!lib) {
      std::fprintf(stderr, "Failed to load metallib: %s\n", err.localizedDescription.UTF8String);
      return 1;
    }

    struct Variant {
      const char *name;
      const char *fn;
      std::uint32_t bm;
      std::uint32_t bn;
      std::uint32_t tptg;
      int kind; // 0=A8, 1=MXFP8, 2=W4A4
      id<MTLComputePipelineState> pso0;
      id<MTLComputePipelineState> pso1;
      AgxSpillInfo spill0;
      AgxSpillInfo spill1;
    };

    std::vector<Variant> variants = {
        {"baseline_ws_128t",      "bench_baseline_ws_128t",        64,  64, 128, 0, nil, nil, {}, {}},
        {"alpha_win_sm_lut2k",    "bench_alpha_win_sm_lut_32x32",  64,  64, 128, 0, nil, nil, {}, {}},
        {"alpha_win_sm_lut512",   "bench_alpha_win_sm_lut512_32x32",64, 64, 128, 0, nil, nil, {}, {}},
        {"alpha_256t_cta_s64",    "bench_alpha_256t_cta_s64",      64,  64, 256, 0, nil, nil, {}, {}},
        {"alpha_256t_ta_s64",     "bench_alpha_256t_ta_s64",       64,  64, 256, 0, nil, nil, {}, {}},
        {"alpha_256t_cta_s72",    "bench_alpha_256t_cta_s72",      64,  64, 256, 0, nil, nil, {}, {}},
        {"alpha_512t_n128_cta",   "bench_alpha_512t_n128_cta_s64", 64, 128, 512, 0, nil, nil, {}, {}},
        {"alpha_512t_m128_cta",   "bench_alpha_512t_m128_cta_s64",128,  64, 512, 0, nil, nil, {}, {}},
        {"beta_mxfp8_tg_256t",    "bench_beta_mxfp8_tg_256t",      64,  64, 256, 1, nil, nil, {}, {}},
        {"beta_mxfp8_dva_256t",   "bench_beta_mxfp8_dva_256t",     64,  64, 256, 1, nil, nil, {}, {}},
        {"ref_w4a4_128t",         "bench_w4a4_zero_spill_128t",    64,  64, 128, 2, nil, nil, {}, {}},
        {"w4a4_win_tg_128p",      "bench_w4a4_win_tg_128p",        64,  64, 128, 2, nil, nil, {}, {}},
    };

    id<MTLComputePipelineState> pso_qa8 = build_pso(dev, lib, "bench_quantize_a8_g64", false);
    id<MTLComputePipelineState> pso_qfp8 = build_pso(dev, lib, "bench_quantize_mxfp8_g32", false);
    id<MTLComputePipelineState> pso_qa4 = build_pso(dev, lib, "bench_quantize_a4_bm16_bk128", false);

    std::printf("\n=== 1. Apple10 SRAM & Spill Audit (icb=0 / icb=1) ===\n");
    std::printf("%-24s | %8s | %8s | %9s | %11s\n", "Variant", "TG SRAM", "MaxThr", "GPR(0/1)", "Spill(0/1)");
    bool any_spill = false;
    for (auto &v : variants) {
      v.pso0 = build_pso(dev, lib, v.fn, false);
      v.pso1 = build_pso(dev, lib, v.fn, true);
      v.spill0 = inspect_agx(v.pso0);
      v.spill1 = inspect_agx(v.pso1);
      std::printf("%-24s | %6lu B | %8lu | %4u/%-4u | %4u B/%-4u B\n",
                  v.name,
                  (unsigned long)v.pso0.staticThreadgroupMemoryLength,
                  (unsigned long)v.pso0.maxTotalThreadsPerThreadgroup,
                  v.spill0.gpr_count, v.spill1.gpr_count,
                  v.spill0.spill_bytes, v.spill1.spill_bytes);
      if (apple10 && (v.spill0.spill_bytes > 0 || v.spill1.spill_bytes > 0)) {
        any_spill = true;
      }
    }
    if (any_spill) {
      std::printf("WARNING: One or more candidate variants spilled on Apple10!\n");
    }

    // ========================================================================
    // 2. Accuracy Audit: Normal (|de| <= 2) AND Stress (|de| > 3 + c=0)
    // ========================================================================
    std::printf("\n=== 2. Numerical Accuracy Audit (M=128, N=256, K=512 vs FP32 CPU Ref) ===\n");
    for (int stress = 0; stress <= 1; ++stress) {
      const std::uint32_t M = 128u, N = 256u, K = 512u, MP = 128u;
      std::vector<float> ref_x, ref_w;
      id<MTLBuffer> x_buf = [dev newBufferWithLength:M * K * 2 options:MTLResourceStorageModeShared];
      id<MTLBuffer> wc_buf = [dev newBufferWithLength:(N / 16u) * (K / 128u) * 1024u options:MTLResourceStorageModeShared];
      id<MTLBuffer> ws_buf = [dev newBufferWithLength:(N / 16u) * (K / 128u) * 64u options:MTLResourceStorageModeShared];
      fill_bf16_acts((std::uint16_t *)x_buf.contents, M, K, 42u + stress, &ref_x);
      fill_mxfp4_tiled((std::uint8_t *)wc_buf.contents, (std::uint8_t *)ws_buf.contents,
                       N, K, 99u + stress, stress != 0, &ref_w);

      std::vector<float> ref_y(M * N, 0.0f);
      for (std::uint32_t r = 0; r < M; ++r)
        for (std::uint32_t c = 0; c < N; ++c) {
          double acc = 0.0;
          for (std::uint32_t d = 0; d < K; ++d) acc += double(ref_x[r * K + d]) * double(ref_w[c * K + d]);
          ref_y[r * N + c] = float(acc);
        }

      id<MTLBuffer> qa8_buf = [dev newBufferWithLength:MP * K options:MTLResourceStorageModeShared];
      id<MTLBuffer> sa8_buf = [dev newBufferWithLength:(K / 64u) * MP * 4 options:MTLResourceStorageModeShared];
      id<MTLBuffer> qfp8_buf = [dev newBufferWithLength:MP * K options:MTLResourceStorageModeShared];
      id<MTLBuffer> sfp8_buf = [dev newBufferWithLength:MP * (K / 32u) options:MTLResourceStorageModeShared];
      id<MTLBuffer> qa4_buf = [dev newBufferWithLength:(MP / 16u) * (K / 128u) * 1024u options:MTLResourceStorageModeShared];
      id<MTLBuffer> sa4_buf = [dev newBufferWithLength:(MP / 16u) * (K / 128u) * 64u options:MTLResourceStorageModeShared];
      id<MTLBuffer> y_buf = [dev newBufferWithLength:M * N * 2 options:MTLResourceStorageModeShared];
      id<MTLBuffer> y_base_buf = [dev newBufferWithLength:M * N * 2 options:MTLResourceStorageModeShared];

      BenchQuantParams qp{K, MP, 0, M};
      {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso_qa8];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:qa8_buf offset:0 atIndex:1];
        [enc setBuffer:sa8_buf offset:0 atIndex:2];
        [enc setBytes:&qp length:sizeof(qp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(K / 64u, MP, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        [enc setComputePipelineState:pso_qfp8];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:qfp8_buf offset:0 atIndex:1];
        [enc setBuffer:sfp8_buf offset:0 atIndex:2];
        [enc setBytes:&qp length:sizeof(qp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((K / 128u + 3u) / 4u, MP, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:pso_qa4];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:qa4_buf offset:0 atIndex:1];
        [enc setBuffer:sa4_buf offset:0 atIndex:2];
        [enc setBytes:&qp length:sizeof(qp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(K / 32u, MP, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
      }

      std::printf("--- Dataset: %s ---\n", stress ? "STRESS (|de| in 4..12 + c=0)" : "NORMAL (|de| <= 2)");
      BenchGemmParams gp{N, K, MP, M, N, 0, N, 0};
      for (std::size_t vi = 0; vi < variants.size(); ++vi) {
        const auto &v = variants[vi];
        if (!apple10 && v.kind == 0 && vi > 0) continue; // Apple10 lane layout kernels
        if (!apple10 && v.kind >= 1) continue;
        std::memset(y_buf.contents, 0, M * N * 2);
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:v.pso0];
        if (v.kind == 0) {
          [enc setBuffer:qa8_buf offset:0 atIndex:0];
          [enc setBuffer:sa8_buf offset:0 atIndex:1];
        } else if (v.kind == 1) {
          [enc setBuffer:qfp8_buf offset:0 atIndex:0];
          [enc setBuffer:sfp8_buf offset:0 atIndex:1];
        } else {
          [enc setBuffer:qa4_buf offset:0 atIndex:0];
          [enc setBuffer:sa4_buf offset:0 atIndex:1];
        }
        [enc setBuffer:wc_buf offset:0 atIndex:2];
        [enc setBuffer:ws_buf offset:0 atIndex:3];
        [enc setBuffer:y_buf offset:0 atIndex:4];
        [enc setBytes:&gp length:sizeof(gp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((N + v.bn - 1u) / v.bn, (M + v.bm - 1u) / v.bm, 1)
            threadsPerThreadgroup:MTLSizeMake(v.tptg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
          std::printf("  %-24s GPU ERROR\n", v.name);
          return 1;
        }
        if (vi == 0) {
          std::memcpy(y_base_buf.contents, y_buf.contents, M * N * 2);
        }
        const auto acc = compare_bf16_vs_ref((const std::uint16_t *)y_buf.contents, ref_y);
        // Also measure exact diff vs baseline_ws_128t for Candidate alpha!
        float max_vs_base = 0.0f;
        const auto *gb = (const std::uint16_t *)y_base_buf.contents;
        const auto *gv = (const std::uint16_t *)y_buf.contents;
        for (std::size_t i = 0; i < std::size_t(M) * N; ++i) {
          max_vs_base = std::max(max_vs_base, std::fabs(bf16_to_f32(gv[i]) - bf16_to_f32(gb[i])));
        }
        std::printf("  %-24s cos_sim=%.7f  max_abs=%.5f  rel_rmse=%.5f  max_vs_base=%.5f\n",
                    v.name, acc.cos_sim, acc.max_abs_err, acc.max_rel_rmse, max_vs_base);
      }
    }

    if (!apple10) {
      std::printf("\nNon-Apple10 host; skipping M5 Ultra TFLOPS timing.\n");
      return 0;
    }

    // ========================================================================
    // 3. M5 Ultra Live Performance Microbenchmark (Qwen3.8-27B Shapes)
    // ========================================================================
    struct Shape {
      const char *label;
      std::uint32_t m, n, k;
    };
    const std::vector<Shape> shapes = {
        {"Gate/Up [2048,5120]x[17408,5120]",   2048, 17408,  5120},
        {"Down    [2048,17408]x[5120,17408]",  2048,  5120, 17408},
        {"QKVZ    [2048,5120]x[10240,5120]",   2048, 10240,  5120},
    };
    const int kWarmup = quick ? 3 : 6;
    const int kIters = quick ? 10 : 20;

    std::printf("\n=== 3. M5 Ultra 80-Core Performance Benchmark (warmup=%d, iters=%d) ===\n", kWarmup, kIters);
    for (const auto &sh : shapes) {
      const std::uint32_t M = sh.m, N = sh.n, K = sh.k, MP = ((M + 63u) / 64u) * 64u;
      const double flops = 2.0 * double(M) * double(N) * double(K);

      id<MTLBuffer> x_buf = [dev newBufferWithLength:std::size_t(M) * K * 2 options:MTLResourceStorageModeShared];
      id<MTLBuffer> wc_buf = [dev newBufferWithLength:std::size_t(N / 16u) * (K / 128u) * 1024u options:MTLResourceStorageModeShared];
      id<MTLBuffer> ws_buf = [dev newBufferWithLength:std::size_t(N / 16u) * (K / 128u) * 64u options:MTLResourceStorageModeShared];
      fill_bf16_acts((std::uint16_t *)x_buf.contents, M, K, 123u);
      fill_mxfp4_tiled((std::uint8_t *)wc_buf.contents, (std::uint8_t *)ws_buf.contents, N, K, 456u, false);

      id<MTLBuffer> qa8_buf = [dev newBufferWithLength:std::size_t(MP) * K options:MTLResourceStorageModePrivate];
      id<MTLBuffer> sa8_buf = [dev newBufferWithLength:std::size_t(K / 64u) * MP * 4 options:MTLResourceStorageModePrivate];
      id<MTLBuffer> qfp8_buf = [dev newBufferWithLength:std::size_t(MP) * K options:MTLResourceStorageModePrivate];
      id<MTLBuffer> sfp8_buf = [dev newBufferWithLength:std::size_t(MP) * (K / 32u) options:MTLResourceStorageModePrivate];
      id<MTLBuffer> qa4_buf = [dev newBufferWithLength:std::size_t(MP / 16u) * (K / 128u) * 1024u options:MTLResourceStorageModePrivate];
      id<MTLBuffer> sa4_buf = [dev newBufferWithLength:std::size_t(MP / 16u) * (K / 128u) * 64u options:MTLResourceStorageModePrivate];
      id<MTLBuffer> y_buf = [dev newBufferWithLength:std::size_t(M) * N * 2 options:MTLResourceStorageModePrivate];

      // Measure quantizer overhead first
      BenchQuantParams qp{K, MP, 0, M};
      auto time_quant = [&](id<MTLComputePipelineState> pso, id<MTLBuffer> d0, id<MTLBuffer> d1, MTLSize grid, MTLSize tg) -> double {
        for (int i = 0; i < kWarmup; ++i) {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:pso];
          [enc setBuffer:x_buf offset:0 atIndex:0];
          [enc setBuffer:d0 offset:0 atIndex:1];
          [enc setBuffer:d1 offset:0 atIndex:2];
          [enc setBytes:&qp length:sizeof(qp) atIndex:3];
          [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
          [enc endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
        }
        std::vector<double> ms;
        for (int i = 0; i < kIters; ++i) {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:pso];
          [enc setBuffer:x_buf offset:0 atIndex:0];
          [enc setBuffer:d0 offset:0 atIndex:1];
          [enc setBuffer:d1 offset:0 atIndex:2];
          [enc setBytes:&qp length:sizeof(qp) atIndex:3];
          [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
          [enc endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
          ms.push_back((cb.GPUEndTime - cb.GPUStartTime) * 1e3);
        }
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2];
      };

      const double q_a8_ms = time_quant(pso_qa8, qa8_buf, sa8_buf, MTLSizeMake(K / 64u, MP, 1), MTLSizeMake(32, 1, 1));
      const double q_fp8_ms = time_quant(pso_qfp8, qfp8_buf, sfp8_buf, MTLSizeMake((K / 128u + 3u) / 4u, MP, 1), MTLSizeMake(128, 1, 1));
      const double q_a4_ms = time_quant(pso_qa4, qa4_buf, sa4_buf, MTLSizeMake(K / 32u, MP, 1), MTLSizeMake(32, 1, 1));

      std::printf("\nShape: %s (%.2f GFLOP) | Quant ms: A8=%.3f, MXFP8=%.3f, A4=%.3f\n",
                  sh.label, flops * 1e-9, q_a8_ms, q_fp8_ms, q_a4_ms);
      std::printf("  %-24s | %8s | %10s | %10s | %8s\n",
                  "Variant", "GEMM ms", "GEMM TF", "Q+GEMM TF", "vs Base");

      BenchGemmParams gp{N, K, MP, M, N, 0, N, 0};
      double base_ms = 0.0;
      for (std::size_t vi = 0; vi < variants.size(); ++vi) {
        const auto &v = variants[vi];
        id<MTLBuffer> a0 = (v.kind == 0) ? qa8_buf : (v.kind == 1) ? qfp8_buf : qa4_buf;
        id<MTLBuffer> a1 = (v.kind == 0) ? sa8_buf : (v.kind == 1) ? sfp8_buf : sa4_buf;
        const double q_ms = (v.kind == 0) ? q_a8_ms : (v.kind == 1) ? q_fp8_ms : q_a4_ms;
        const MTLSize grid = MTLSizeMake((N + v.bn - 1u) / v.bn, (M + v.bm - 1u) / v.bm, 1);
        const MTLSize tg = MTLSizeMake(v.tptg, 1, 1);

        for (int i = 0; i < kWarmup; ++i) {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:v.pso0];
          [enc setBuffer:a0 offset:0 atIndex:0];
          [enc setBuffer:a1 offset:0 atIndex:1];
          [enc setBuffer:wc_buf offset:0 atIndex:2];
          [enc setBuffer:ws_buf offset:0 atIndex:3];
          [enc setBuffer:y_buf offset:0 atIndex:4];
          [enc setBytes:&gp length:sizeof(gp) atIndex:5];
          [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
          [enc endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
        }
        std::vector<double> ms;
        for (int i = 0; i < kIters; ++i) {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:v.pso0];
          [enc setBuffer:a0 offset:0 atIndex:0];
          [enc setBuffer:a1 offset:0 atIndex:1];
          [enc setBuffer:wc_buf offset:0 atIndex:2];
          [enc setBuffer:ws_buf offset:0 atIndex:3];
          [enc setBuffer:y_buf offset:0 atIndex:4];
          [enc setBytes:&gp length:sizeof(gp) atIndex:5];
          [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
          [enc endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
          ms.push_back((cb.GPUEndTime - cb.GPUStartTime) * 1e3);
        }
        std::sort(ms.begin(), ms.end());
        const double med_ms = ms[ms.size() / 2];
        if (vi == 0) base_ms = med_ms;
        const double tf = (flops * 1e-9) / med_ms;
        const double e2e_tf = (flops * 1e-9) / (med_ms + q_ms);
        std::printf("  %-24s | %8.3f | %10.2f | %10.2f | %7.2fx\n",
                    v.name, med_ms, tf, e2e_tf, base_ms / med_ms);
      }
    }
    return 0;
  }
}
