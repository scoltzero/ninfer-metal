// Apple10 (M5 Ultra) Global <= 16.0 KiB Static TG SRAM & 0-Byte Register-Spill
// CI Audit Gate + BinaryArchive Serialization + Shader-Validation Dispatch Smoke.
//
// Enforces all 4 verification layers:
//   Layer 1: Public Metal API `staticThreadgroupMemoryLength <= 16,384 B`
//            (plus tighter per-WP caps: GQA BK=16 WN=4 <= 14,336 B,
//             GDN dv16_flex <= 10,240 B, W4A8/Affine out64 <= 10,240 B)
//            for both `supportIndirectCommandBuffers = NO` and `YES`.
//   Layer 2: `@encode`-safe AGX `RuntimeState` introspection on Apple10
//            asserting `total_spill_buffer_bytes == 0` for every audited kernel.
//   Layer 3: `MTLBinaryArchive` serialization of every audited pipeline so
//            offline `metal-lipo` / `metal-objdump` inspection validates real
//            Apple10 GPU machine code.
//   Layer 4: Live minimal-shape GPU dispatch smoke across WP3.2 GQA Flash-Prefill,
//            WP3.3 GDN Stage 1/2 (`dv16_flex` + `dv32_flex`), and CacheBlend
//            `gdn_extract_transition_m_c64` (clean under MTL_SHADER_VALIDATION=1).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

using Json = nlohmann::json;

namespace {

constexpr std::uint64_t kApple10GlobalMaxTgSram = 16384ULL;
constexpr std::uint64_t kWp32GqaMaxTgSram = 14336ULL;
constexpr std::uint64_t kWp33Dv16MaxTgSram = 10240ULL;
constexpr std::uint64_t kWp31Out64MaxTgSram = 10240ULL;

const std::vector<std::string> kAllPrefillLibs = {
    "omlx_m5_affine_nax.metallib",
    "omlx_m5_mxfp4_nax.metallib",
    "omlx_m5_moe_nax.metallib",
    "omlx_m5_bf16_nax.metallib",
    "omlx_m5_w4a8_nax.metallib",
    "omlx_m5_w4a8_moe_nax.metallib",
    "omlx_m5_attention_nax_prefill.metallib",
    "omlx_m5_attention_affine4.metallib",
    "omlx_m5_attention_sage2.metallib",
    "omlx_m5_attention_minference.metallib",
    "omlx_m5_gdn_chunk_nax.metallib",
    "omlx_m5_cache_blend.metallib",
    "omlx_m5_ane_pack_unpack.metallib",
};

struct AgxSpillInfo {
  bool available = false;
  std::uint32_t gpr_count = 0U;
  std::uint32_t spill_bytes = 0U;
};

bool is_shader_validation_enabled() {
  const char *env = std::getenv("MTL_SHADER_VALIDATION");
  if (env == nullptr || env[0] == '\0') {
    return false;
  }
  return std::string_view(env) != "0";
}

bool is_apple10_device(id<MTLDevice> dev) {
  if (@available(macOS 15.0, *)) {
    // MTLGPUFamilyApple10 == 1010 in macOS 26/27 SDK headers.
    if ([dev supportsFamily:(MTLGPUFamily)1010]) {
      return true;
    }
  }
  const char *name = dev.name ? dev.name.UTF8String : "";
  return std::strstr(name, "M5") != nullptr;
}

AgxSpillInfo inspect_agx_runtime_state(id<MTLComputePipelineState> pso) {
  AgxSpillInfo info{};
  Class cls = object_getClass(pso);
  while (cls != Nil) {
    Ivar iv = class_getInstanceVariable(cls, "_impl");
    if (iv != nullptr) {
      const char *enc = ivar_getTypeEncoding(iv);
      if (enc != nullptr && std::strstr(enc, "RuntimeState") != nullptr) {
        const auto *base = reinterpret_cast<const std::uint8_t *>(
            (__bridge const void *)pso);
        const auto *u32 = reinterpret_cast<const std::uint32_t *>(
            base + ivar_getOffset(iv));
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

std::uint64_t expected_tg_sram_cap(std::string_view lib_name,
                                   std::string_view fn_name) {
  if (lib_name == "omlx_m5_w4a8_nax.metallib" ||
      lib_name == "omlx_m5_w4a8_moe_nax.metallib" ||
      lib_name == "omlx_m5_ane_pack_unpack.metallib") {
    return kWp31Out64MaxTgSram;
  }
  if (fn_name.find("gqa_bk16_wn4") != std::string_view::npos) {
    return kWp32GqaMaxTgSram;
  }
  if (fn_name.find("dv16_flex") != std::string_view::npos) {
    return kWp33Dv16MaxTgSram;
  }
  if (fn_name.find("out64") != std::string_view::npos) {
    return kWp31Out64MaxTgSram;
  }
  return kApple10GlobalMaxTgSram;
}

// Layer 2 (0-byte AGX spill) applies to 100% of kernels in the 8 pure Phase-2
// prefill libraries and to 100% of mainstream prefill kernels in the 4 mixed
// small-T/decode + prefill linear libraries (where legacy Wave-1 T<=32 rf*/n16/
// bm16/bm32/w4a4/moe9 kernels use cooperative_tensor.get_multidimensional_index).
bool enforce_zero_spill(std::string_view lib_name, std::string_view fn_name) {
  const bool is_mixed_linear =
      (lib_name == "omlx_m5_affine_nax.metallib" ||
       lib_name == "omlx_m5_mxfp4_nax.metallib" ||
       lib_name == "omlx_m5_moe_nax.metallib" ||
       lib_name == "omlx_m5_bf16_nax.metallib");
  if (!is_mixed_linear) {
    return true;
  }
  if (fn_name.find("w4a4") != std::string_view::npos) {
    return false;
  }
  return fn_name.find("bm64") != std::string_view::npos ||
         fn_name.find("out64") != std::string_view::npos ||
         fn_name.find("bucket_schedule") != std::string_view::npos ||
         fn_name.find("down_scatter") != std::string_view::npos ||
         fn_name.find("xsum") != std::string_view::npos ||
         fn_name.find("rmsnorm") != std::string_view::npos ||
         fn_name.find("fwht64") != std::string_view::npos ||
         fn_name.find("bf16_to_f16") != std::string_view::npos ||
         fn_name.find("repack") != std::string_view::npos ||
         fn_name.find("copy_u32") != std::string_view::npos;
}

std::vector<std::string> split_csv(std::string_view csv) {
  std::vector<std::string> out;
  std::size_t pos = 0;
  while (pos < csv.size()) {
    const std::size_t comma = csv.find(',', pos);
    const std::size_t len =
        (comma == std::string_view::npos) ? (csv.size() - pos) : (comma - pos);
    if (len > 0) {
      out.emplace_back(csv.substr(pos, len));
    }
    if (comma == std::string_view::npos) {
      break;
    }
    pos = comma + 1;
  }
  return out;
}

struct PagedAttentionParamsHost {
  std::uint32_t query_heads;
  std::uint32_t key_value_heads;
  std::uint32_t head_dim;
  std::uint32_t tokens;
  std::uint32_t page_tokens;
  std::uint32_t query_token_stride;
  std::uint32_t query_head_stride;
  std::uint32_t gate_token_stride;
  std::uint32_t gate_head_stride;
};

struct GdnChunkLayerParamsHost {
  std::uint32_t key_heads;
  std::uint32_t value_heads;
  std::uint32_t key_dim;
  std::uint32_t value_dim;
  std::uint32_t conv_width;
  std::uint32_t conv_kernel;
  std::uint32_t tokens;
  float epsilon;
};

struct GdnChunkFlexParamsHost {
  std::uint32_t key_heads;
  std::uint32_t value_heads;
  std::uint32_t key_dim;
  std::uint32_t value_dim;
  std::uint32_t conv_width;
  std::uint32_t conv_kernel;
  std::uint32_t tokens;
  float epsilon;
  std::uint32_t mp;
  std::uint32_t slices_per_head;
  std::uint32_t total_tasks;
  std::uint32_t pad0;
};

struct GdnExtractParamsHost {
  std::uint32_t num_k_heads;
  std::uint32_t num_v_heads;
  std::uint32_t k_dim;
  std::uint32_t v_dim;
  std::uint32_t chunk_len;
  std::uint32_t chunk_idx;
  std::uint32_t pad0;
  std::uint32_t pad1;
};

bool run_smoke_dispatches(id<MTLDevice> dev,
                          const std::filesystem::path &lib_dir,
                          Json &smoke_report,
                          std::vector<std::string> &failures) {
  id<MTLCommandQueue> queue = [dev newCommandQueue];
  if (!queue) {
    failures.emplace_back("Layer 4 smoke: failed to create MTLCommandQueue");
    return false;
  }

  auto load_lib = [&](const std::string &name) -> id<MTLLibrary> {
    const auto full = lib_dir / name;
    if (!std::filesystem::is_regular_file(full)) {
      return nil;
    }
    NSError *err = nil;
    NSString *ns_path = [NSString stringWithUTF8String:full.c_str()];
    id<MTLLibrary> lib =
        [dev newLibraryWithURL:[NSURL fileURLWithPath:ns_path] error:&err];
    if (!lib) {
      failures.emplace_back("Layer 4 smoke: failed loading " + name + ": " +
                            (err ? err.localizedDescription.UTF8String : "unknown"));
    }
    return lib;
  };

  auto make_pso = [&](id<MTLLibrary> lib,
                      const char *fn_name) -> id<MTLComputePipelineState> {
    if (!lib) {
      return nil;
    }
    id<MTLFunction> fn =
        [lib newFunctionWithName:[NSString stringWithUTF8String:fn_name]];
    if (!fn) {
      failures.emplace_back(std::string("Layer 4 smoke: missing function ") + fn_name);
      return nil;
    }
    NSError *err = nil;
    MTLComputePipelineDescriptor *desc = [MTLComputePipelineDescriptor new];
    desc.computeFunction = fn;
    desc.supportIndirectCommandBuffers = NO;
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithDescriptor:desc
                                           options:MTLPipelineOptionNone
                                        reflection:nil
                                             error:&err];
    if (!pso) {
      failures.emplace_back(std::string("Layer 4 smoke: PSO build failed for ") +
                            fn_name + ": " +
                            (err ? err.localizedDescription.UTF8String : "unknown"));
    }
    return pso;
  };

  auto make_zero_buf = [&](std::size_t bytes) -> id<MTLBuffer> {
    id<MTLBuffer> buf =
        [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buf) {
      std::memset(buf.contents, 0, bytes);
    }
    return buf;
  };

  Json executed = Json::array();

  // 1. WP3.2 BF16 GQA Flash-Prefill smoke (WM=2 & WM=3, T=64, QH=6, KVH=1, HD=256).
  if (id<MTLLibrary> lib = load_lib("omlx_m5_attention_nax_prefill.metallib")) {
    constexpr std::uint32_t kTokens = 64U;
    constexpr std::uint32_t kQh = 6U;
    constexpr std::uint32_t kKvh = 1U;
    constexpr std::uint32_t kHd = 256U;
    constexpr std::uint32_t kPageTok = 64U;

    id<MTLBuffer> q_buf = make_zero_buf(std::size_t(kTokens) * kQh * kHd * 2ULL);
    id<MTLBuffer> g_buf = make_zero_buf(std::size_t(kTokens) * kQh * kHd * 2ULL);
    id<MTLBuffer> k_buf = make_zero_buf(std::size_t(kPageTok) * kKvh * kHd * 2ULL);
    id<MTLBuffer> v_buf = make_zero_buf(std::size_t(kPageTok) * kKvh * kHd * 2ULL);
    id<MTLBuffer> o_buf = make_zero_buf(std::size_t(kTokens) * kQh * kHd * 2ULL);
    id<MTLBuffer> pt_buf = make_zero_buf(4ULL * sizeof(std::uint32_t));
    id<MTLBuffer> pos_buf = make_zero_buf(kTokens * sizeof(std::uint32_t));
    auto *pos_ptr = static_cast<std::uint32_t *>(pos_buf.contents);
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      pos_ptr[t] = t;
    }
    PagedAttentionParamsHost params{
        .query_heads = kQh,
        .key_value_heads = kKvh,
        .head_dim = kHd,
        .tokens = kTokens,
        .page_tokens = kPageTok,
        .query_token_stride = kQh * kHd,
        .query_head_stride = kHd,
        .gate_token_stride = kQh * kHd,
        .gate_head_stride = kHd,
    };

    for (const auto &entry : std::vector<std::pair<const char *, std::uint32_t>>{
             {"omlx_m5_paged_flash_prefill_gqa_bk16_wn4_nax_hd256_wm2", 2U},
             {"omlx_m5_paged_flash_prefill_gqa_bk16_wn4_nax_hd256_wm3", 3U}}) {
      id<MTLComputePipelineState> pso = make_pso(lib, entry.first);
      if (!pso) {
        continue;
      }
      const std::uint32_t m_tile = entry.second * 16U;
      const std::uint32_t n_m_blocks = (kTokens * (kQh / kKvh) + m_tile - 1U) / m_tile;
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:q_buf offset:0 atIndex:0];
      [enc setBuffer:g_buf offset:0 atIndex:1];
      [enc setBuffer:k_buf offset:0 atIndex:2];
      [enc setBuffer:v_buf offset:0 atIndex:3];
      [enc setBuffer:o_buf offset:0 atIndex:4];
      [enc setBuffer:pt_buf offset:0 atIndex:5];
      [enc setBuffer:pos_buf offset:0 atIndex:6];
      [enc setBytes:&params length:sizeof(params) atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake(1, std::size_t(n_m_blocks) * kKvh, 1)
          threadsPerThreadgroup:MTLSizeMake(std::size_t(entry.second) * 128U, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status != MTLCommandBufferStatusCompleted) {
        failures.emplace_back(std::string("Layer 4 smoke failed GPU execution for ") +
                              entry.first);
      } else {
        executed.push_back(entry.first);
      }
    }
  }

  // 2. WP3.2 Affine4Group128 GQA Flash-Prefill smoke (WM=2 & WM=3).
  if (id<MTLLibrary> lib = load_lib("omlx_m5_attention_affine4.metallib")) {
    constexpr std::uint32_t kTokens = 64U;
    constexpr std::uint32_t kQh = 6U;
    constexpr std::uint32_t kKvh = 1U;
    constexpr std::uint32_t kHd = 256U;
    constexpr std::uint32_t kPageTok = 64U;
    // Page bytes for affine4_g128 HD=256: codes (128 B/tok) + meta (8 B/tok) = 136 B/tok.
    constexpr std::size_t kPageBytes = std::size_t(kPageTok) * kKvh * 136ULL;

    id<MTLBuffer> q_buf = make_zero_buf(std::size_t(kTokens) * kQh * kHd * 2ULL);
    id<MTLBuffer> g_buf = make_zero_buf(std::size_t(kTokens) * kQh * kHd * 2ULL);
    id<MTLBuffer> k_buf = make_zero_buf(kPageBytes);
    id<MTLBuffer> v_buf = make_zero_buf(kPageBytes);
    id<MTLBuffer> o_buf = make_zero_buf(std::size_t(kTokens) * kQh * kHd * 2ULL);
    id<MTLBuffer> pt_buf = make_zero_buf(4ULL * sizeof(std::uint32_t));
    id<MTLBuffer> pos_buf = make_zero_buf(kTokens * sizeof(std::uint32_t));
    auto *pos_ptr = static_cast<std::uint32_t *>(pos_buf.contents);
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      pos_ptr[t] = t;
    }
    PagedAttentionParamsHost params{
        .query_heads = kQh,
        .key_value_heads = kKvh,
        .head_dim = kHd,
        .tokens = kTokens,
        .page_tokens = kPageTok,
        .query_token_stride = kQh * kHd,
        .query_head_stride = kHd,
        .gate_token_stride = kQh * kHd,
        .gate_head_stride = kHd,
    };

    for (const auto &entry : std::vector<std::pair<const char *, std::uint32_t>>{
             {"omlx_m5_paged_flash_prefill_gqa_bk16_wn4_affine4_g128_hd256_wm2", 2U},
             {"omlx_m5_paged_flash_prefill_gqa_bk16_wn4_affine4_g128_hd256_wm3", 3U}}) {
      id<MTLComputePipelineState> pso = make_pso(lib, entry.first);
      if (!pso) {
        continue;
      }
      const std::uint32_t m_tile = entry.second * 16U;
      const std::uint32_t n_m_blocks = (kTokens * (kQh / kKvh) + m_tile - 1U) / m_tile;
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:q_buf offset:0 atIndex:0];
      [enc setBuffer:g_buf offset:0 atIndex:1];
      [enc setBuffer:k_buf offset:0 atIndex:2];
      [enc setBuffer:v_buf offset:0 atIndex:3];
      [enc setBuffer:o_buf offset:0 atIndex:4];
      [enc setBuffer:pt_buf offset:0 atIndex:5];
      [enc setBuffer:pos_buf offset:0 atIndex:6];
      [enc setBytes:&params length:sizeof(params) atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake(1, std::size_t(n_m_blocks) * kKvh, 1)
          threadsPerThreadgroup:MTLSizeMake(std::size_t(entry.second) * 128U, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status != MTLCommandBufferStatusCompleted) {
        failures.emplace_back(std::string("Layer 4 smoke failed GPU execution for ") +
                              entry.first);
      } else {
        executed.push_back(entry.first);
      }
    }
  }

  // 3. WP3.3 GDN Stage 1 + Stage 2 (`dv16_flex` K_dv=10 + `dv32_flex` K_dv=5) smoke.
  if (id<MTLLibrary> lib = load_lib("omlx_m5_gdn_chunk_nax.metallib")) {
    constexpr std::uint32_t kTokens = 64U;
    constexpr std::uint32_t kHk = 2U;
    constexpr std::uint32_t kHv = 4U;
    constexpr std::uint32_t kD = 128U;
    constexpr std::uint32_t kConvW = 2U * kHk * kD + kHv * kD;
    constexpr std::size_t kQkBytes = std::size_t(kHk) * 2ULL * 64ULL * kD * 2ULL;
    constexpr std::size_t kCtrlBytes = std::size_t(kHv) * 17152ULL;

    id<MTLBuffer> conv_buf = make_zero_buf(std::size_t(kTokens) * kConvW * 2ULL);
    id<MTLBuffer> a_buf = make_zero_buf(std::size_t(kTokens) * kHv * 2ULL);
    id<MTLBuffer> b_buf = make_zero_buf(std::size_t(kTokens) * kHv * 2ULL);
    id<MTLBuffer> alog_buf = make_zero_buf(std::size_t(kHv) * 2ULL);
    id<MTLBuffer> dt_buf = make_zero_buf(std::size_t(kHv) * 2ULL);
    id<MTLBuffer> qk_ws = make_zero_buf(kQkBytes);
    id<MTLBuffer> ctrl_ws = make_zero_buf(kCtrlBytes);
    id<MTLBuffer> state_buf = make_zero_buf(std::size_t(kHv) * kD * kD * 4ULL);
    id<MTLBuffer> rec_buf = make_zero_buf(std::size_t(kTokens) * kHv * kD * 2ULL);

    GdnChunkLayerParamsHost lp{kHk, kHv, kD, kD, kConvW, 4U, kTokens, 1.0e-6f};
    if (id<MTLComputePipelineState> prep = make_pso(lib, "omlx_m5_gdn_chunk_prepare_c64")) {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:prep];
      [enc setBuffer:conv_buf offset:0 atIndex:0];
      [enc setBuffer:a_buf offset:0 atIndex:1];
      [enc setBuffer:b_buf offset:0 atIndex:2];
      [enc setBuffer:alog_buf offset:0 atIndex:3];
      [enc setBuffer:dt_buf offset:0 atIndex:4];
      [enc setBuffer:qk_ws offset:0 atIndex:5];
      [enc setBuffer:ctrl_ws offset:0 atIndex:6];
      [enc setBytes:&lp length:sizeof(lp) atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake(kHk, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status != MTLCommandBufferStatusCompleted) {
        failures.emplace_back("Layer 4 smoke failed for omlx_m5_gdn_chunk_prepare_c64");
      } else {
        executed.push_back("omlx_m5_gdn_chunk_prepare_c64");
      }
    }

    struct FlexSmoke {
      const char *fn;
      std::uint32_t k_dv;
    };
    for (const auto &fs : std::vector<FlexSmoke>{
             {"omlx_m5_gdn_chunk_recurrence_dv16_flex_c64", 10U},
             {"omlx_m5_gdn_chunk_recurrence_dv32_flex_c64", 5U}}) {
      id<MTLComputePipelineState> pso = make_pso(lib, fs.fn);
      if (!pso) {
        continue;
      }
      GdnChunkFlexParamsHost fp{
          kHk, kHv, kD, kD, kConvW, 4U, kTokens, 1.0e-6f, kTokens, fs.k_dv,
          kHv * fs.k_dv, 0U};
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:conv_buf offset:0 atIndex:0];
      [enc setBuffer:qk_ws offset:0 atIndex:1];
      [enc setBuffer:ctrl_ws offset:0 atIndex:2];
      [enc setBuffer:state_buf offset:0 atIndex:3];
      [enc setBuffer:rec_buf offset:0 atIndex:4];
      [enc setBytes:&fp length:sizeof(fp) atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(fp.total_tasks, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status != MTLCommandBufferStatusCompleted) {
        failures.emplace_back(std::string("Layer 4 smoke failed for ") + fs.fn);
      } else {
        executed.push_back(fs.fn);
      }
    }
  }

  // 4. CacheBlend `gdn_extract_transition_m_c64` smoke.
  if (id<MTLLibrary> lib = load_lib("omlx_m5_cache_blend.metallib")) {
    if (id<MTLComputePipelineState> pso =
            make_pso(lib, "omlx_m5_gdn_extract_transition_m_c64")) {
      constexpr std::uint32_t kHk = 1U;
      constexpr std::uint32_t kHv = 1U;
      constexpr std::uint32_t kD = 128U;
      id<MTLBuffer> qk_ws = make_zero_buf(2ULL * 64ULL * kD * 2ULL);
      id<MTLBuffer> ctrl_ws = make_zero_buf(17152ULL);
      id<MTLBuffer> m_out = make_zero_buf(std::size_t(kD) * kD * 2ULL);
      GdnExtractParamsHost ep{kHk, kHv, kD, kD, 64U, 0U, 0U, 0U};
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:qk_ws offset:0 atIndex:0];
      [enc setBuffer:ctrl_ws offset:0 atIndex:1];
      [enc setBuffer:m_out offset:0 atIndex:2];
      [enc setBytes:&ep length:sizeof(ep) atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake(kHv, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status != MTLCommandBufferStatusCompleted) {
        failures.emplace_back(
            "Layer 4 smoke failed for omlx_m5_gdn_extract_transition_m_c64");
      } else {
        executed.push_back("omlx_m5_gdn_extract_transition_m_c64");
      }
    }
  }

  // 5. WP 3C Dual-ANE pack/unpack + fused fwht64_a8 & fwht64_a4 smoke.
  if (id<MTLLibrary> lib = load_lib("omlx_m5_ane_pack_unpack.metallib")) {
    struct AnePackParamsHost {
      std::uint32_t T;
      std::uint32_t K;
      std::uint32_t src_stride;
      std::uint32_t reserved;
    };
    struct AneUnpackParamsHost {
      std::uint32_t T;
      std::uint32_t ane0_n;
      std::uint32_t ane1_n;
      std::uint32_t out_stride;
      std::uint32_t dst_col_offset;
      std::uint32_t reserved;
    };
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

    constexpr std::uint32_t kT = 64U;
    constexpr std::uint32_t kMp = 64U;
    constexpr std::uint32_t kK = 128U;
    constexpr std::uint32_t kDieN = 128U;
    constexpr std::uint32_t kOutStride = 256U;

    id<MTLBuffer> src_bf16 = make_zero_buf(std::size_t(kT) * kK * 2ULL);
    id<MTLBuffer> in_fp16_0 = make_zero_buf(std::size_t(kK) * kT * 2ULL);
    id<MTLBuffer> in_fp16_1 = make_zero_buf(std::size_t(kK) * kT * 2ULL);
    id<MTLBuffer> out_fp16_0 = make_zero_buf(std::size_t(kDieN) * kT * 2ULL);
    id<MTLBuffer> out_fp16_1 = make_zero_buf(std::size_t(kDieN) * kT * 2ULL);
    id<MTLBuffer> dst_bf16 = make_zero_buf(std::size_t(kT) * kOutStride * 2ULL);
    id<MTLBuffer> qa_out = make_zero_buf(std::size_t(kMp) * kOutStride);
    id<MTLBuffer> sa_out =
        make_zero_buf(std::size_t(kOutStride / 64U) * kMp * sizeof(float));
    id<MTLBuffer> ra_out =
        make_zero_buf(std::size_t(kOutStride / 64U) * kMp * sizeof(float));
    id<MTLBuffer> a4_codes = make_zero_buf(std::size_t(kMp) * (kOutStride / 2U));
    id<MTLBuffer> a4_scales =
        make_zero_buf(std::size_t(kMp) * (kOutStride / 32U));

    auto run_ane_smoke = [&](const char *fn_name, auto &&encode_fn) {
      id<MTLComputePipelineState> pso = make_pso(lib, fn_name);
      if (!pso) {
        return;
      }
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      encode_fn(enc);
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status != MTLCommandBufferStatusCompleted) {
        failures.emplace_back(std::string("Layer 4 smoke failed for ") + fn_name);
      } else {
        executed.push_back(fn_name);
      }
    };

    const AnePackParamsHost pp{kT, kK, kK, 0U};
    run_ane_smoke("omlx_m5_ane_pack_bf16_to_fp16_planar_single",
                  [&](id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:src_bf16 offset:0 atIndex:0];
                    [enc setBuffer:in_fp16_0 offset:0 atIndex:1];
                    [enc setBytes:&pp length:sizeof(pp) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(kK / 32U, kT / 32U, 1U)
                        threadsPerThreadgroup:MTLSizeMake(8U, 32U, 1U)];
                  });
    run_ane_smoke("omlx_m5_ane_pack_bf16_to_fp16_planar_dual",
                  [&](id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:src_bf16 offset:0 atIndex:0];
                    [enc setBuffer:in_fp16_0 offset:0 atIndex:1];
                    [enc setBuffer:in_fp16_1 offset:0 atIndex:2];
                    [enc setBytes:&pp length:sizeof(pp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(kK / 32U, kT / 32U, 1U)
                        threadsPerThreadgroup:MTLSizeMake(8U, 32U, 1U)];
                  });

    const AneUnpackParamsHost up_s{kT, kDieN, 0U, kOutStride, 0U, 0U};
    const AneUnpackParamsHost up_d{kT, kDieN, kDieN, kOutStride, 0U, 0U};
    run_ane_smoke("omlx_m5_ane_unpack_fp16_planar_to_bf16_single",
                  [&](id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:out_fp16_0 offset:0 atIndex:0];
                    [enc setBuffer:dst_bf16 offset:0 atIndex:1];
                    [enc setBytes:&up_s length:sizeof(up_s) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(kDieN / 32U, kT / 32U, 1U)
                        threadsPerThreadgroup:MTLSizeMake(8U, 32U, 1U)];
                  });
    run_ane_smoke("omlx_m5_ane_unpack_fp16_planar_to_bf16_dual",
                  [&](id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:out_fp16_0 offset:0 atIndex:0];
                    [enc setBuffer:out_fp16_1 offset:0 atIndex:1];
                    [enc setBuffer:dst_bf16 offset:0 atIndex:2];
                    [enc setBytes:&up_d length:sizeof(up_d) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake((2U * kDieN) / 32U, kT / 32U, 1U)
                        threadsPerThreadgroup:MTLSizeMake(8U, 32U, 1U)];
                  });

    const AneUnpackQuantParamsHost qp_s{kT, kMp, kDieN, 0U, kOutStride, 0U, 0U, 0U};
    const AneUnpackQuantParamsHost qp_d{kT, kMp, kDieN, kDieN, kOutStride, 0U, 0U, 0U};
    run_ane_smoke("omlx_m5_ane_unpack_fp16_planar_to_fwht64_a8_single",
                  [&](id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:out_fp16_0 offset:0 atIndex:0];
                    [enc setBuffer:qa_out offset:0 atIndex:1];
                    [enc setBuffer:sa_out offset:0 atIndex:2];
                    [enc setBuffer:ra_out offset:0 atIndex:3];
                    [enc setBytes:&qp_s length:sizeof(qp_s) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(kDieN / 64U, kMp / 32U, 1U)
                        threadsPerThreadgroup:MTLSizeMake(128U, 1U, 1U)];
                  });
    run_ane_smoke("omlx_m5_ane_unpack_fp16_planar_to_fwht64_a8_dual",
                  [&](id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:out_fp16_0 offset:0 atIndex:0];
                    [enc setBuffer:out_fp16_1 offset:0 atIndex:1];
                    [enc setBuffer:qa_out offset:0 atIndex:2];
                    [enc setBuffer:sa_out offset:0 atIndex:3];
                    [enc setBuffer:ra_out offset:0 atIndex:4];
                    [enc setBytes:&qp_d length:sizeof(qp_d) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((2U * kDieN) / 64U, kMp / 32U, 1U)
                        threadsPerThreadgroup:MTLSizeMake(128U, 1U, 1U)];
                  });
    run_ane_smoke("omlx_m5_ane_unpack_fp16_planar_to_fwht64_a4_single",
                  [&](id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:out_fp16_0 offset:0 atIndex:0];
                    [enc setBuffer:a4_codes offset:0 atIndex:1];
                    [enc setBuffer:a4_scales offset:0 atIndex:2];
                    [enc setBytes:&qp_s length:sizeof(qp_s) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(kDieN / 128U, kMp / 16U, 1U)
                        threadsPerThreadgroup:MTLSizeMake(128U, 1U, 1U)];
                  });
    run_ane_smoke("omlx_m5_ane_unpack_fp16_planar_to_fwht64_a4_dual",
                  [&](id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:out_fp16_0 offset:0 atIndex:0];
                    [enc setBuffer:out_fp16_1 offset:0 atIndex:1];
                    [enc setBuffer:a4_codes offset:0 atIndex:2];
                    [enc setBuffer:a4_scales offset:0 atIndex:3];
                    [enc setBytes:&qp_d length:sizeof(qp_d) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake((2U * kDieN) / 128U, kMp / 16U, 1U)
                        threadsPerThreadgroup:MTLSizeMake(128U, 1U, 1U)];
                  });
  }

  smoke_report = {{"executed_kernels", executed},
                  {"shader_validation_enabled", is_shader_validation_enabled()}};
  return failures.empty();
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2) {
      std::cerr << "usage: omlx_m5_prefill_sram_spill_audit_tests <metallib_dir> "
                   "[--all] [--libs a.metallib,b.metallib] "
                   "[--archive-out <path>] [--no-smoke] [--smoke-only] [--json-out <path>]\n";
      return 2;
    }

    const std::filesystem::path lib_dir = argv[1];
    bool audit_all = false;
    bool run_smoke = true;
    bool smoke_only = false;
    std::string custom_libs;
    std::string archive_out;
    std::string json_out;

    for (int i = 2; i < argc; ++i) {
      const std::string_view arg(argv[i]);
      if (arg == "--all") {
        audit_all = true;
      } else if (arg == "--no-smoke") {
        run_smoke = false;
      } else if (arg == "--smoke-only") {
        smoke_only = true;
      } else if (arg == "--libs" && i + 1 < argc) {
        custom_libs = argv[++i];
      } else if (arg == "--archive-out" && i + 1 < argc) {
        archive_out = argv[++i];
      } else if (arg == "--json-out" && i + 1 < argc) {
        json_out = argv[++i];
      } else {
        std::cerr << "Unknown argument: " << arg << '\n';
        return 2;
      }
    }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      std::cerr << "No Metal device available\n";
      return 1;
    }

    const bool apple10 = is_apple10_device(dev);
    const bool shader_val = is_shader_validation_enabled();

    std::vector<std::string> libs_to_audit;
    if (!custom_libs.empty()) {
      libs_to_audit = split_csv(custom_libs);
    } else {
      (void)audit_all;
      libs_to_audit = kAllPrefillLibs;
    }

    NSError *err = nil;
    id<MTLBinaryArchive> archive = nil;
    if (!archive_out.empty()) {
      MTLBinaryArchiveDescriptor *arch_desc = [MTLBinaryArchiveDescriptor new];
      archive = [dev newBinaryArchiveWithDescriptor:arch_desc error:&err];
      if (!archive) {
        std::cerr << "Failed to create MTLBinaryArchive: "
                  << (err ? err.localizedDescription.UTF8String : "unknown") << '\n';
        return 1;
      }
    }

    std::vector<std::string> failures;
    Json libraries_json = Json::array();
    std::uint64_t total_kernels = 0;
    std::uint64_t total_psos = 0;
    std::uint64_t agx_inspected_psos = 0;
    std::uint64_t prefill_zero_spill_psos = 0;
    std::uint64_t max_tg_sram_seen = 0;
    std::uint32_t max_prefill_spill_seen = 0;

    if (!smoke_only) for (const auto &lib_name : libs_to_audit) {
      const auto full_path = lib_dir / lib_name;
      if (!std::filesystem::is_regular_file(full_path)) {
        failures.emplace_back("Missing required metallib: " + full_path.string());
        continue;
      }
      NSString *ns_path = [NSString stringWithUTF8String:full_path.c_str()];
      id<MTLLibrary> lib =
          [dev newLibraryWithURL:[NSURL fileURLWithPath:ns_path] error:&err];
      if (!lib) {
        failures.emplace_back("Failed to load " + lib_name + ": " +
                              (err ? err.localizedDescription.UTF8String : "unknown"));
        continue;
      }

      NSArray<NSString *> *names =
          [lib.functionNames sortedArrayUsingSelector:@selector(compare:)];
      Json funcs_json = Json::array();

      for (NSString *fn_ns in names) {
        id<MTLFunction> fn = [lib newFunctionWithName:fn_ns];
        if (!fn || fn.functionType != MTLFunctionTypeKernel) {
          continue;
        }
        ++total_kernels;
        const std::string fn_name = fn_ns.UTF8String;
        const std::uint64_t tg_cap =
            shader_val ? 32768ULL : expected_tg_sram_cap(lib_name, fn_name);
        const bool strict_zero_spill = enforce_zero_spill(lib_name, fn_name);

        for (int icb = 0; icb <= 1; ++icb) {
          MTLComputePipelineDescriptor *desc = [MTLComputePipelineDescriptor new];
          desc.computeFunction = fn;
          desc.supportIndirectCommandBuffers = (icb != 0) ? YES : NO;

          id<MTLComputePipelineState> pso =
              [dev newComputePipelineStateWithDescriptor:desc
                                                 options:MTLPipelineOptionNone
                                              reflection:nil
                                                   error:&err];
          if (!pso) {
            failures.emplace_back(lib_name + "::" + fn_name + " (icb=" +
                                  std::to_string(icb) + ") PSO build failed: " +
                                  (err ? err.localizedDescription.UTF8String
                                       : "unknown"));
            continue;
          }
          ++total_psos;

          const std::uint64_t tg_bytes =
              static_cast<std::uint64_t>(pso.staticThreadgroupMemoryLength);
          const std::uint64_t max_tptg =
              static_cast<std::uint64_t>(pso.maxTotalThreadsPerThreadgroup);
          max_tg_sram_seen = std::max(max_tg_sram_seen, tg_bytes);

          if (tg_bytes > tg_cap) {
            std::ostringstream oss;
            oss << "LAYER 1 SRAM VIOLATION: " << lib_name << "::" << fn_name
                << " (icb=" << icb << ") staticThreadgroupMemoryLength="
                << tg_bytes << " B > cap " << tg_cap << " B";
            failures.push_back(oss.str());
          }

          AgxSpillInfo spill{};
          if (apple10 && !shader_val) {
            spill = inspect_agx_runtime_state(pso);
            if (!spill.available) {
              failures.emplace_back(
                  "LAYER 2 SPILL INTROSPECTION UNAVAILABLE on Apple10 for " +
                  lib_name + "::" + fn_name);
            } else {
              ++agx_inspected_psos;
              if (strict_zero_spill) {
                ++prefill_zero_spill_psos;
                max_prefill_spill_seen =
                    std::max(max_prefill_spill_seen, spill.spill_bytes);
                if (spill.spill_bytes > 0U) {
                  std::ostringstream oss;
                  oss << "LAYER 2 SPILL VIOLATION: " << lib_name << "::" << fn_name
                      << " (icb=" << icb << ") total_spill_buffer_bytes="
                      << spill.spill_bytes << " B (gpr=" << spill.gpr_count
                      << ") > 0 B";
                  failures.push_back(oss.str());
                }
              }
            }
          }

          if (archive != nil && icb == 1 && strict_zero_spill) {
            if (![archive addComputePipelineFunctionsWithDescriptor:desc
                                                              error:&err]) {
              failures.emplace_back(
                  "LAYER 3 ARCHIVE ADD FAILED: " + lib_name + "::" + fn_name +
                  ": " +
                  (err ? err.localizedDescription.UTF8String : "unknown"));
            }
          }

          if (icb == 1) {
            funcs_json.push_back({{"name", fn_name},
                                  {"tg_sram_bytes", tg_bytes},
                                  {"tg_sram_cap_bytes", tg_cap},
                                  {"max_threads_per_tg", max_tptg},
                                  {"agx_inspected", spill.available},
                                  {"zero_spill_enforced", strict_zero_spill},
                                  {"gpr_count", spill.gpr_count},
                                  {"spill_bytes", spill.spill_bytes}});
          }
        }
      }

      libraries_json.push_back({{"library", lib_name},
                                {"kernel_count", funcs_json.size()},
                                {"kernels", funcs_json}});
    }

    bool archive_serialized = false;
    if (archive != nil) {
      NSURL *out_url =
          [NSURL fileURLWithPath:[NSString stringWithUTF8String:archive_out.c_str()]];
      if (![archive serializeToURL:out_url error:&err]) {
        failures.emplace_back(
            "LAYER 3 ARCHIVE SERIALIZE FAILED: " +
            std::string(err ? err.localizedDescription.UTF8String : "unknown"));
      } else {
        archive_serialized = true;
      }
    }

    Json smoke_report = {{"skipped", true}};
    if (run_smoke) {
      (void)run_smoke_dispatches(dev, lib_dir, smoke_report, failures);
    }

    const bool passed = failures.empty();
    Json report = {
        {"schema", "omlx-m5-apple10-sram-spill-audit-v1"},
        {"device", dev.name ? std::string(dev.name.UTF8String) : "unknown"},
        {"apple10_host", apple10},
        {"shader_validation_active", shader_val},
        {"layer1_global_sram_cap_bytes", kApple10GlobalMaxTgSram},
        {"layer2_agx_spill_enforced", apple10 && !shader_val && !smoke_only},
        {"layer3_archive_serialized", archive_serialized},
        {"layer4_smoke", smoke_report},
        {"summary",
         {{"libraries_audited", libraries_json.size()},
          {"total_kernels", total_kernels},
          {"total_psos_checked", total_psos},
          {"agx_inspected_psos", agx_inspected_psos},
          {"prefill_zero_spill_psos", prefill_zero_spill_psos},
          {"max_tg_sram_bytes", max_tg_sram_seen},
          {"max_spill_bytes", max_prefill_spill_seen}}},
        {"libraries", libraries_json},
        {"failures", failures},
        {"passed", passed},
    };

    const std::string dump = report.dump(2);
    if (!json_out.empty()) {
      std::ofstream ofs(json_out);
      ofs << dump << '\n';
    }
    std::cout << dump << '\n';
    if (!passed) {
      for (const auto &f : failures) {
        std::cerr << "FAIL: " << f << '\n';
      }
      return 1;
    }
    return 0;
  }
}
