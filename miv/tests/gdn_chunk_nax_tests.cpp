#include "../src/gdn_prefill.h"
#include "omlx_m5/artifact.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/qwen_forward.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

using namespace omlx::m5;
using namespace omlx::m5::detail::forward;
using Json = nlohmann::json;

namespace {

void require(bool ok, std::string_view msg) {
  if (!ok) {
    throw std::runtime_error("GDN-CHUNK-NAX TEST FAIL: " + std::string(msg));
  }
}

struct ScopedEnv {
  std::string name;
  std::optional<std::string> prev;
  ScopedEnv(std::string key, const char *val) : name(std::move(key)) {
    if (const char *old = std::getenv(name.c_str())) {
      prev = std::string(old);
    }
    if (val != nullptr) {
      setenv(name.c_str(), val, 1);
    } else {
      unsetenv(name.c_str());
    }
  }
  ~ScopedEnv() {
    if (prev.has_value()) {
      setenv(name.c_str(), prev->c_str(), 1);
    } else {
      unsetenv(name.c_str());
    }
  }
};

struct ErrorStats {
  double max_abs = 0.0;
  double rel_l2 = 0.0;
};

ErrorStats compare_f32(std::span<const float> actual,
                       std::span<const float> expected,
                       std::string_view label) {
  require(actual.size() == expected.size(),
          std::string(label) + ": size mismatch");
  double diff_sq = 0.0;
  double ref_sq = 0.0;
  double max_abs = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    require(std::isfinite(actual[i]),
            std::string(label) + ": non-finite actual value at index " +
                std::to_string(i));
    require(std::isfinite(expected[i]),
            std::string(label) + ": non-finite expected value at index " +
                std::to_string(i));
    const double d = std::abs(double(actual[i]) - double(expected[i]));
    max_abs = std::max(max_abs, d);
    diff_sq += d * d;
    ref_sq += double(expected[i]) * double(expected[i]);
  }
  const double rel_l2 = std::sqrt(diff_sq / std::max(ref_sq, 1e-12));
  require(std::isfinite(max_abs) && std::isfinite(rel_l2),
          std::string(label) + ": non-finite error summary");
  return {.max_abs = max_abs, .rel_l2 = rel_l2};
}

float f16_bits_to_float(std::uint16_t h) {
  const std::uint32_t sign = std::uint32_t(h & 0x8000U) << 16U;
  const std::uint32_t exp = (h >> 10U) & 0x1FU;
  const std::uint32_t mant = h & 0x03FFU;
  if (exp == 0U) {
    if (mant == 0U) {
      return std::bit_cast<float>(sign);
    }
    const float mag = std::ldexp(float(mant), -24);
    return (sign != 0U) ? -mag : mag;
  }
  if (exp == 31U) {
    const std::uint32_t bits = sign | 0x7F800000U | (mant << 13U);
    return std::bit_cast<float>(bits);
  }
  const std::uint32_t bits = sign | ((exp + 112U) << 23U) | (mant << 13U);
  return std::bit_cast<float>(bits);
}

std::vector<float> read_f32_buf(const MetalBuffer &buf, std::size_t count) {
  const auto raw = buf.read(0, count * sizeof(float));
  std::vector<float> out(count);
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

std::vector<std::uint16_t> read_u16_buf(const MetalBuffer &buf,
                                        std::size_t count) {
  const auto raw = buf.read(0, count * sizeof(std::uint16_t));
  std::vector<std::uint16_t> out(count);
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

std::vector<float> u16_bf16_to_f32(std::span<const std::uint16_t> words) {
  std::vector<float> out(words.size());
  for (std::size_t i = 0; i < words.size(); ++i) {
    out[i] = bf16_to_float(words[i]);
  }
  return out;
}

std::vector<float> u16_f16_to_f32(std::span<const std::uint16_t> words) {
  std::vector<float> out(words.size());
  for (std::size_t i = 0; i < words.size(); ++i) {
    out[i] = f16_bits_to_float(words[i]);
  }
  return out;
}

float round_bf16(float x) { return bf16_to_float(float_to_bf16(x)); }

// Exact CPU reference of the Two-Stage C=64 BF16-NAX + IEEE FP32-State algorithm
// (validating the mathematical formulation across all hosts).
void cpu_gdn_chunk_c64(U32 tokens, U32 hk_count, U32 hv_count,
                       std::span<const std::uint16_t> convolved_bf16,
                       std::span<const std::uint16_t> a_bf16,
                       std::span<const std::uint16_t> b_bf16,
                       std::span<const std::uint16_t> a_log_bf16,
                       std::span<const std::uint16_t> dt_bias_bf16,
                       std::vector<float> &state,
                       std::vector<std::uint16_t> &recurrent_out) {
  constexpr U32 C = 64U;
  constexpr U32 D = 128U;
  const U32 repeat = hv_count / hk_count;
  const U32 key_width = hk_count * D;
  const U32 value_width = hv_count * D;
  const U32 conv_width = 2U * key_width + value_width;
  const float inv_scale = 1.0f / std::sqrt(128.0f);
  const float inv_scale_sq = 1.0f / 128.0f;

  recurrent_out.assign(std::size_t(tokens) * value_width, 0U);

  for (U32 hv = 0; hv < hv_count; ++hv) {
    const U32 hk = hv / repeat;
    const float dt_bias_val = bf16_to_float(dt_bias_bf16[hv]);
    const float neg_exp_a = -std::exp(bf16_to_float(a_log_bf16[hv]));

    for (U32 c = 0; c * C < tokens; ++c) {
      const U32 t0 = c * C;
      const U32 clen = std::min<U32>(C, tokens - t0);

      std::array<float, C * D> nk{};
      std::array<float, C * D> nq{};
      std::array<float, C> G{};
      std::array<float, C> beta{};
      std::array<float, C> prefix{};
      std::array<float, C> suffix{};

      float g_cum = 0.0f;
      for (U32 r = 0; r < C; ++r) {
        if (r < clen) {
          const std::size_t row_base = std::size_t(t0 + r) * conv_width;
          float q_sq = 0.0f;
          float k_sq = 0.0f;
          for (U32 d = 0; d < D; ++d) {
            const float qv = bf16_to_float(convolved_bf16[row_base + hk * D + d]);
            const float kv =
                bf16_to_float(convolved_bf16[row_base + key_width + hk * D + d]);
            nq[r * D + d] = qv;
            nk[r * D + d] = kv;
            q_sq += qv * qv;
            k_sq += kv * kv;
          }
          const float q_inv =
              (1.0f / std::sqrt(q_sq * inv_scale_sq + 1.0e-6f)) * inv_scale_sq;
          const float k_inv =
              (1.0f / std::sqrt(k_sq * inv_scale_sq + 1.0e-6f)) * inv_scale;
          for (U32 d = 0; d < D; ++d) {
            nq[r * D + d] = round_bf16(nq[r * D + d] * q_inv);
            nk[r * D + d] = round_bf16(nk[r * D + d] * k_inv);
          }
          const std::size_t h_idx = std::size_t(t0 + r) * hv_count + hv;
          const float act = bf16_to_float(a_bf16[h_idx]) + dt_bias_val;
          const float sp = act > 20.0f ? act : std::log(1.0f + std::exp(act));
          const float g_val = std::min(0.0f, neg_exp_a * sp);
          g_cum = std::min(0.0f, g_cum + g_val);
          beta[r] = 1.0f / (1.0f + std::exp(-bf16_to_float(b_bf16[h_idx])));
        }
        G[r] = g_cum;
        prefix[r] = std::exp(G[r]);
      }
      const float g_last = G[C - 1U];
      const float decay_last = std::exp(g_last);
      for (U32 r = 0; r < C; ++r) {
        suffix[r] = std::exp(std::min(0.0f, g_last - G[r]));
      }

      std::array<float, C * C> L{};
      for (U32 i = 0; i < clen; ++i) {
        for (U32 j = 0; j < i; ++j) {
          float dot = 0.0f;
          for (U32 d = 0; d < D; ++d) {
            dot += nk[i * D + d] * nk[j * D + d];
          }
          L[i * C + j] =
              beta[i] * std::exp(std::min(0.0f, G[i] - G[j])) * dot;
        }
      }

      std::array<float, C * C> X{};
      std::array<float, C * C> solve{};
      for (U32 i = 0; i < clen; ++i) {
        for (U32 j = 0; j <= i; ++j) {
          float x_val = 0.0f;
          if (i == j) {
            x_val = 1.0f;
          } else {
            float s = 0.0f;
            for (U32 k = j; k < i; ++k) {
              s += L[i * C + k] * X[k * C + j];
            }
            x_val = -s;
          }
          X[i * C + j] = x_val;
          solve[i * C + j] = round_bf16(x_val * beta[j]);
        }
      }

      std::array<float, C * C> mqk{};
      for (U32 i = 0; i < clen; ++i) {
        for (U32 j = 0; j <= i; ++j) {
          float dot = 0.0f;
          for (U32 d = 0; d < D; ++d) {
            dot += nq[i * D + d] * nk[j * D + d];
          }
          mqk[i * C + j] = round_bf16(std::exp(std::min(0.0f, G[i] - G[j])) * dot);
        }
      }

      float *state_hv = state.data() + std::size_t(hv * D) * D;
      std::array<float, D * D> s_bf16{};
      for (std::size_t idx = 0; idx < D * D; ++idx) {
        s_bf16[idx] = round_bf16(state_hv[idx]);
      }

      std::array<float, C * D> rhs{};
      for (U32 r = 0; r < clen; ++r) {
        const std::size_t v_base =
            std::size_t(t0 + r) * conv_width + 2U * key_width + hv * D;
        for (U32 dv = 0; dv < D; ++dv) {
          float pred = 0.0f;
          const float *s_row = s_bf16.data() + std::size_t(dv) * D;
          for (U32 d = 0; d < D; ++d) {
            pred += nk[r * D + d] * s_row[d];
          }
          const float vv = bf16_to_float(convolved_bf16[v_base + dv]);
          rhs[r * D + dv] = round_bf16(vv - prefix[r] * pred);
        }
      }

      std::array<float, C * D> delta{};
      std::array<float, C * D> delta_bf{};
      for (U32 r = 0; r < clen; ++r) {
        for (U32 dv = 0; dv < D; ++dv) {
          float acc = 0.0f;
          for (U32 k = 0; k <= r; ++k) {
            acc += solve[r * C + k] * rhs[k * D + dv];
          }
          delta[r * D + dv] = acc;
          delta_bf[r * D + dv] = round_bf16(acc);
        }
      }

      for (U32 r = 0; r < clen; ++r) {
        for (U32 dv = 0; dv < D; ++dv) {
          float hist = 0.0f;
          const float *s_row = s_bf16.data() + std::size_t(dv) * D;
          for (U32 d = 0; d < D; ++d) {
            hist += nq[r * D + d] * s_row[d];
          }
          float intra = 0.0f;
          for (U32 k = 0; k <= r; ++k) {
            intra += mqk[r * C + k] * delta_bf[k * D + dv];
          }
          recurrent_out[std::size_t(t0 + r) * value_width + hv * D + dv] =
              float_to_bf16(prefix[r] * hist + intra);
        }
      }

      for (U32 dv = 0; dv < D; ++dv) {
        float *s_row = state_hv + std::size_t(dv) * D;
        for (U32 d = 0; d < D; ++d) {
          float add_s = 0.0f;
          for (U32 r = 0; r < clen; ++r) {
            const float ds = round_bf16(suffix[r] * delta[r * D + dv]);
            add_s += ds * nk[r * D + d];
          }
          s_row[d] = decay_last * s_row[d] + add_s;
        }
      }
    }
  }
}

Json test_memory_and_pipeline_invariants(const MetalContext &ctx,
                                         const std::filesystem::path &libs,
                                         bool &gpu_pipeline_ready) {
  const auto metallib_path = libs / "omlx_m5_gdn_chunk_nax.metallib";
  require(std::filesystem::is_regular_file(metallib_path) &&
              std::filesystem::file_size(metallib_path) > 0,
          "omlx_m5_gdn_chunk_nax.metallib missing or empty");

  // Qwen3.5-35B-A3B head geometry: Hk=16, Hv=32, Dk=128, Dv=128.
  QwenModelConfig cfg35b;
  cfg35b.model_type = "qwen3_5_text";
  cfg35b.hidden_size = 5120;
  cfg35b.intermediate_size = 17408;
  cfg35b.vocab_size = 248320;
  cfg35b.num_hidden_layers = 4;
  cfg35b.layer_types = {LayerType::linear_attention, LayerType::linear_attention,
                        LayerType::linear_attention, LayerType::full_attention};
  cfg35b.num_attention_heads = 24;
  cfg35b.num_key_value_heads = 4;
  cfg35b.head_dim = 256;
  cfg35b.linear_num_key_heads = 16;
  cfg35b.linear_num_value_heads = 32;
  cfg35b.linear_key_head_dim = 128;
  cfg35b.linear_value_head_dim = 128;
  cfg35b.linear_conv_kernel_dim = 4;

  const auto ws_1k = gdn_chunk_nax_workspace_layout(1024U, 16U, 32U);
  require(ws_1k.max_chunks == 16U && ws_1k.qk_bytes == 8388608ULL &&
              ws_1k.ctrl_bytes == 8781824ULL &&
              ws_1k.total_bytes == 17170432ULL &&
              ws_1k.total_bytes <= (2ULL << 30),
          "C=64 GDN chunk workspace layout disagrees with 16.375 MiB <= 2 GiB spec");

  const auto ws_2k = gdn_chunk_nax_workspace_layout(2048U, 16U, 48U);
  require(ws_2k.max_chunks == 32U && ws_2k.total_bytes == 43122688ULL &&
              ws_2k.total_bytes <= (2ULL << 30),
          "27B 2048-row C=64 GDN chunk workspace layout disagrees with <= 2 GiB spec");

  const auto primary_plan = plan_qwen_forward_memory(
      cfg35b, 65536U, 2U, false, true, false, true, true, false, 0U, 1024U, false);
  const auto secondary_plan = plan_qwen_forward_memory(
      cfg35b, 65536U, 2U, false, true, false, true, true, false, 0U, 16U, true);

  bool saw_primary_ws = false;
  for (const auto &b : primary_plan.buffers) {
    require(b.name.find(".scratch") == std::string::npos,
            "forbidden .scratch substring in primary plan: " + b.name);
    if (b.name == "qwen.workspace.gdn_chunk") {
      require(b.bytes == ws_1k.total_bytes && !b.state,
              "qwen.workspace.gdn_chunk wrong byte size or state flag");
      saw_primary_ws = true;
    }
  }
  require(saw_primary_ws,
          "qwen.workspace.gdn_chunk missing from primary 1024-row memory plan");

  for (const auto &b : secondary_plan.buffers) {
    require(b.name != "qwen.workspace.gdn_chunk",
            "secondary 16-row slot unexpectedly allocated qwen.workspace.gdn_chunk");
  }

  {
    ScopedEnv clear_force("OMLX_M5_FORCE_GDN_CHUNK_NAX", nullptr);
    ScopedEnv clear_nax_force("OMLX_M5_FORCE_NAX_APPLE10", nullptr);
    ScopedEnv clear_dis("OMLX_M5_DISABLE_GDN_CHUNK_NAX", nullptr);
    const bool default_eligible =
        gdn_chunk_nax_runtime_eligible(ctx, libs, 16U, 32U, 128U, 128U);
    const bool is_apple10 =
        ctx.capabilities().device_name.starts_with("Apple M5") ||
        ctx.capabilities().profile.probe.apple_gpu_family >= 10;
    require(default_eligible == is_apple10,
            "default gdn_chunk_nax_runtime_eligible disagrees with Apple10 gate");
    require(!gdn_chunk_nax_runtime_eligible(ctx, libs, 16U, 32U, 64U, 128U) &&
                !gdn_chunk_nax_runtime_eligible(ctx, libs, 16U, 32U, 128U, 64U),
            "non-128 head_dim unexpectedly accepted by gdn_chunk_nax_runtime_eligible");
  }
  const bool is_apple9_plus =
      ctx.capabilities().device_name.starts_with("Apple M5") ||
      ctx.capabilities().profile.probe.apple_gpu_family >= 9;
  const char *sv_env = std::getenv("MTL_SHADER_VALIDATION");
  const bool shader_validation_active =
      sv_env != nullptr && std::string_view(sv_env) != "0" &&
      !std::string_view(sv_env).empty();
  const U64 sram_scale = shader_validation_active ? 2ULL : 1ULL;
  const U64 sram_pad = shader_validation_active ? 256ULL : 0ULL;

  ctx.register_library(
      {.key = "test.gdn_layer", .path = libs / "omlx_m5_gdn_layer.metallib"});
  const auto conv_vec_info = ctx.warmup_pipeline(
      {.key = "test.gdn_conv_silu_vec4_t8",
       .library_key = "test.gdn_layer",
       .function_name = "omlx_m5_gdn_conv_silu_vec4_t8",
       .variant = "test",
       .support_indirect_command_buffers =
           ctx.capabilities().profile.compute_icb_eligible});
  require(conv_vec_info.static_threadgroup_memory_bytes <= sram_pad,
          "omlx_m5_gdn_conv_silu_vec4_t8 expected 0 B static TG SRAM");

  U64 prep_tg_bytes = 0;
  U64 prep_conv_tg_bytes = 0;
  U64 rec_tg_bytes = 0;
  U64 rec_conv_dv32_tg_bytes = 0;
  U64 rec_dv64_tg_bytes = 0;
  U64 rec_conv_dv64_tg_bytes = 0;
  U64 ng_inplace_bf16_tg_bytes = 0;
  U64 ng_inplace_f16_tg_bytes = 0;
  U64 rec_mega_bf16_tg_bytes = 0;
  U64 rec_mega_f16_tg_bytes = 0;
  std::string pipeline_note = "ok";
  try {
    ctx.register_library(
        {.key = "test.gdn_chunk_nax", .path = metallib_path});
    const bool icb_ok = ctx.capabilities().profile.compute_icb_eligible;
    const auto prep_info = ctx.warmup_pipeline(
        {.key = "test.gdn_chunk_prepare_c64",
         .library_key = "test.gdn_chunk_nax",
         .function_name = "omlx_m5_gdn_chunk_prepare_c64",
         .variant = "test",
         .support_indirect_command_buffers = icb_ok});
    const auto prep_conv_info = ctx.warmup_pipeline(
        {.key = "test.gdn_chunk_prepare_conv_c64",
         .library_key = "test.gdn_chunk_nax",
         .function_name = "omlx_m5_gdn_chunk_prepare_conv_c64",
         .variant = "test",
         .support_indirect_command_buffers = icb_ok});
    const auto rec_info = ctx.warmup_pipeline(
        {.key = "test.gdn_chunk_recurrence_c64",
         .library_key = "test.gdn_chunk_nax",
         .function_name = "omlx_m5_gdn_chunk_recurrence_c64",
         .variant = "test",
         .support_indirect_command_buffers = icb_ok});
    const auto rec_conv_dv32_info = ctx.warmup_pipeline(
        {.key = "test.gdn_chunk_recurrence_conv_dv32_c64",
         .library_key = "test.gdn_chunk_nax",
         .function_name = "omlx_m5_gdn_chunk_recurrence_conv_dv32_c64",
         .variant = "test",
         .support_indirect_command_buffers = icb_ok});
    const auto rec_dv64_info = ctx.warmup_pipeline(
        {.key = "test.gdn_chunk_recurrence_dv64_c64",
         .library_key = "test.gdn_chunk_nax",
         .function_name = "omlx_m5_gdn_chunk_recurrence_dv64_c64",
         .variant = "test",
         .support_indirect_command_buffers = icb_ok});
    const auto rec_conv_dv64_info = ctx.warmup_pipeline(
        {.key = "test.gdn_chunk_recurrence_conv_dv64_c64",
         .library_key = "test.gdn_chunk_nax",
         .function_name = "omlx_m5_gdn_chunk_recurrence_conv_dv64_c64",
         .variant = "test",
         .support_indirect_command_buffers = icb_ok});
    const auto ng_bf16_info = ctx.warmup_pipeline(
        {.key = "test.gdn_chunk_norm_gate_inplace_bf16_c64",
         .library_key = "test.gdn_chunk_nax",
         .function_name = "omlx_m5_gdn_chunk_norm_gate_inplace_bf16_c64",
         .variant = "test",
         .support_indirect_command_buffers = icb_ok});
    const auto ng_f16_info = ctx.warmup_pipeline(
        {.key = "test.gdn_chunk_norm_gate_inplace_f16_c64",
         .library_key = "test.gdn_chunk_nax",
         .function_name = "omlx_m5_gdn_chunk_norm_gate_inplace_f16_c64",
         .variant = "test",
         .support_indirect_command_buffers = icb_ok});
    const auto rec_bf16_info = ctx.warmup_pipeline(
        {.key = "test.gdn_chunk_recurrence_mega_bf16_c64",
         .library_key = "test.gdn_chunk_nax",
         .function_name = "omlx_m5_gdn_chunk_recurrence_mega_bf16_c64",
         .variant = "test",
         .support_indirect_command_buffers = icb_ok});
    const auto rec_f16_info = ctx.warmup_pipeline(
        {.key = "test.gdn_chunk_recurrence_mega_f16_c64",
         .library_key = "test.gdn_chunk_nax",
         .function_name = "omlx_m5_gdn_chunk_recurrence_mega_f16_c64",
         .variant = "test",
         .support_indirect_command_buffers = icb_ok});

    prep_tg_bytes = prep_info.static_threadgroup_memory_bytes;
    prep_conv_tg_bytes = prep_conv_info.static_threadgroup_memory_bytes;
    rec_tg_bytes = rec_info.static_threadgroup_memory_bytes;
    rec_conv_dv32_tg_bytes = rec_conv_dv32_info.static_threadgroup_memory_bytes;
    rec_dv64_tg_bytes = rec_dv64_info.static_threadgroup_memory_bytes;
    rec_conv_dv64_tg_bytes = rec_conv_dv64_info.static_threadgroup_memory_bytes;
    ng_inplace_bf16_tg_bytes = ng_bf16_info.static_threadgroup_memory_bytes;
    ng_inplace_f16_tg_bytes = ng_f16_info.static_threadgroup_memory_bytes;
    rec_mega_bf16_tg_bytes = rec_bf16_info.static_threadgroup_memory_bytes;
    rec_mega_f16_tg_bytes = rec_f16_info.static_threadgroup_memory_bytes;

    require(prep_info.max_threads_per_threadgroup >= 128ULL &&
                prep_conv_info.max_threads_per_threadgroup >= 128ULL &&
                rec_info.max_threads_per_threadgroup >= 128ULL &&
                rec_conv_dv32_info.max_threads_per_threadgroup >= 128ULL &&
                rec_dv64_info.max_threads_per_threadgroup >= 128ULL &&
                rec_conv_dv64_info.max_threads_per_threadgroup >= 128ULL &&
                ng_bf16_info.max_threads_per_threadgroup >= 128ULL &&
                ng_f16_info.max_threads_per_threadgroup >= 128ULL &&
                rec_bf16_info.max_threads_per_threadgroup >= 256ULL &&
                rec_f16_info.max_threads_per_threadgroup >= 256ULL,
            "GDN chunk NAX max_threads_per_threadgroup below required width");

    const U64 apple10_cap = 32768ULL;
    const U64 sram_cap = 16384ULL * sram_scale;
    require(prep_tg_bytes <= apple10_cap && prep_conv_tg_bytes <= apple10_cap &&
                rec_tg_bytes <= apple10_cap &&
                rec_conv_dv32_tg_bytes <= apple10_cap &&
                rec_dv64_tg_bytes <= apple10_cap &&
                rec_conv_dv64_tg_bytes <= apple10_cap &&
                ng_inplace_bf16_tg_bytes <= sram_pad &&
                ng_inplace_f16_tg_bytes <= sram_pad &&
                rec_mega_bf16_tg_bytes <= apple10_cap &&
                rec_mega_f16_tg_bytes <= apple10_cap,
            "GDN chunk NAX static TG SRAM exceeded 32,768 B Apple10 limit");
    require(prep_tg_bytes <= sram_cap && prep_conv_tg_bytes <= sram_cap &&
                rec_tg_bytes <= sram_cap &&
                rec_conv_dv32_tg_bytes <= sram_cap &&
                rec_dv64_tg_bytes <= sram_cap &&
                rec_conv_dv64_tg_bytes <= sram_cap &&
                rec_mega_bf16_tg_bytes <= sram_cap &&
                rec_mega_f16_tg_bytes <= sram_cap,
            "GDN chunk NAX static TG SRAM exceeded 16.0 KiB design ceiling");
    gpu_pipeline_ready = true;
  } catch (const std::exception &ex) {
    require(!is_apple9_plus,
            std::string("Apple9+ failed to compile GDN chunk NAX pipeline: ") +
                ex.what());
    gpu_pipeline_ready = false;
    pipeline_note = ex.what();
  }

  return {{"metallib_bytes", std::filesystem::file_size(metallib_path)},
          {"workspace_bytes_35b_1k", ws_1k.total_bytes},
          {"workspace_bytes_27b_2k", ws_2k.total_bytes},
          {"gpu_pipeline_ready", gpu_pipeline_ready},
          {"conv_vec4_t8_tg_sram_bytes",
           conv_vec_info.static_threadgroup_memory_bytes},
          {"prepare_c64_tg_sram_bytes", prep_tg_bytes},
          {"prepare_conv_c64_tg_sram_bytes", prep_conv_tg_bytes},
          {"recurrence_c64_tg_sram_bytes", rec_tg_bytes},
          {"recurrence_conv_dv32_c64_tg_sram_bytes", rec_conv_dv32_tg_bytes},
          {"recurrence_dv64_c64_tg_sram_bytes", rec_dv64_tg_bytes},
          {"recurrence_conv_dv64_c64_tg_sram_bytes", rec_conv_dv64_tg_bytes},
          {"norm_gate_inplace_bf16_tg_sram_bytes", ng_inplace_bf16_tg_bytes},
          {"norm_gate_inplace_f16_tg_sram_bytes", ng_inplace_f16_tg_bytes},
          {"recurrence_mega_bf16_c64_tg_sram_bytes", rec_mega_bf16_tg_bytes},
          {"recurrence_mega_f16_c64_tg_sram_bytes", rec_mega_f16_tg_bytes},
          {"pipeline_note", pipeline_note}};
}

Json test_operator_parity(const MetalContext &ctx,
                          const std::filesystem::path &libs,
                          bool gpu_pipeline_ready) {
  // Exercise across T = {64, 73, 100, 128, 512, 1024} (partial tail chunks at
  // T=73 [clen=9 <= 32] and T=100 [clen=36 > 32]), dynamic capacity=1024 -> live=100
  // ICB patching, 0-ULP conv_silu_vec4_t8, mega_bf16 + mega_f16 epilogue fusion,
  // and extreme-decay NaN guard!
  constexpr U32 hk = 4U;
  constexpr U32 hv = 12U; // repeat = 3 (matches Qwen3.8-27B Hk=16, Hv=48!)
  constexpr U32 kd = 128U;
  constexpr U32 vd = 128U;
  constexpr U32 gv = hv * vd;
  constexpr U32 conv_w = 2U * hk * kd + gv;
  constexpr U32 max_t = 1024U;

  std::vector<std::uint16_t> qkv_words(std::size_t(max_t) * conv_w);
  std::vector<std::uint16_t> conv_wt_words(std::size_t(conv_w) * 4U);
  std::vector<std::uint16_t> init_hist_words(3U * conv_w);
  std::vector<std::uint16_t> z_words(std::size_t(max_t) * gv);
  std::vector<std::uint16_t> norm_wt_words(vd);
  std::vector<std::uint16_t> a_words(std::size_t(max_t) * hv);
  std::vector<std::uint16_t> b_words(std::size_t(max_t) * hv);
  std::vector<std::uint16_t> a_log_words(hv);
  std::vector<std::uint16_t> dt_bias_words(hv);
  std::vector<float> init_state(std::size_t(gv) * kd);

  for (std::size_t i = 0; i < qkv_words.size(); ++i) {
    const float val =
        float(int((i * 37ULL + 19ULL) % 251ULL) - 125) / 64.0f;
    qkv_words[i] = float_to_bf16(val);
  }
  for (std::size_t i = 0; i < conv_wt_words.size(); ++i) {
    const float val =
        float(int((i * 17ULL + 5ULL) % 89ULL) - 44) / 64.0f;
    conv_wt_words[i] = float_to_bf16(val);
  }
  for (std::size_t i = 0; i < init_hist_words.size(); ++i) {
    const float val =
        float(int((i * 23ULL + 11ULL) % 131ULL) - 65) / 64.0f;
    init_hist_words[i] = float_to_bf16(val);
  }
  for (std::size_t i = 0; i < z_words.size(); ++i) {
    const float val =
        float(int((i * 31ULL + 13ULL) % 197ULL) - 98) / 64.0f;
    z_words[i] = float_to_bf16(val);
  }
  for (U32 d = 0; d < vd; ++d) {
    norm_wt_words[d] = float_to_bf16(0.75f + 0.5f * float(d % 17U) / 16.0f);
  }
  for (std::size_t i = 0; i < a_words.size(); ++i) {
    const float av = float(int((i * 13ULL + 7ULL) % 101ULL) - 55) / 16.0f;
    const float bv = float(int((i * 29ULL + 3ULL) % 97ULL) - 48) / 16.0f;
    a_words[i] = float_to_bf16(av);
    b_words[i] = float_to_bf16(bv);
  }
  for (U32 h = 0; h < hv; ++h) {
    a_log_words[h] = float_to_bf16(-1.2f + 0.2f * float(h));
    dt_bias_words[h] = float_to_bf16(-0.5f + 0.08f * float(h));
  }
  for (std::size_t i = 0; i < init_state.size(); ++i) {
    init_state[i] = float(int((i * 43ULL + 11ULL) % 199ULL) - 99) / 256.0f;
  }

  auto qkv_buf = ctx.upload(std::as_bytes(std::span(qkv_words)));
  auto conv_wt_buf = ctx.upload(std::as_bytes(std::span(conv_wt_words)));
  auto hist_ref_buf = ctx.make_buffer(init_hist_words.size() * sizeof(std::uint16_t));
  auto hist_nax_buf = ctx.make_buffer(init_hist_words.size() * sizeof(std::uint16_t));
  auto conv_buf = ctx.make_buffer(std::size_t(max_t) * conv_w * sizeof(std::uint16_t));
  auto conv_vec_buf = ctx.make_buffer(std::size_t(max_t) * conv_w * sizeof(std::uint16_t));
  auto z_buf = ctx.upload(std::as_bytes(std::span(z_words)));
  auto norm_wt_buf = ctx.upload(std::as_bytes(std::span(norm_wt_words)));
  auto a_buf = ctx.upload(std::as_bytes(std::span(a_words)));
  auto b_buf = ctx.upload(std::as_bytes(std::span(b_words)));
  auto a_log_buf = ctx.upload(std::as_bytes(std::span(a_log_words)));
  auto dt_bias_buf = ctx.upload(std::as_bytes(std::span(dt_bias_words)));
  auto state_seq_buf = ctx.make_buffer(init_state.size() * sizeof(float));
  auto rec_seq_buf = ctx.make_buffer(std::size_t(max_t) * gv * sizeof(std::uint16_t));
  auto out_seq_bf16_buf = ctx.make_buffer(std::size_t(max_t) * gv * sizeof(std::uint16_t));
  auto out_seq_f16_buf = ctx.make_buffer(std::size_t(max_t) * gv * sizeof(std::uint16_t));
  auto state_nax_buf = ctx.make_buffer(init_state.size() * sizeof(float));
  auto rec_nax_buf = ctx.make_buffer(std::size_t(max_t) * gv * sizeof(std::uint16_t));
  auto out_nax_bf16_buf = ctx.make_buffer(std::size_t(max_t) * gv * sizeof(std::uint16_t));
  auto out_nax_f16_buf = ctx.make_buffer(std::size_t(max_t) * gv * sizeof(std::uint16_t));
  const auto ws_layout = gdn_chunk_nax_workspace_layout(max_t, hk, hv);
  auto ws_buf = ctx.make_private_buffer(ws_layout.total_bytes);

  // 0. Verify 0-ULP bit-identity of `omlx_m5_gdn_conv_silu_vec4_t8` vs `omlx_m5_gdn_conv_silu_2d`!
  for (U32 ct : {4U, 8U, 13U, 64U, 100U, 1024U}) {
    hist_ref_buf.write(std::as_bytes(std::span(init_hist_words)));
    hist_nax_buf.write(std::as_bytes(std::span(init_hist_words)));
    const auto params =
        words({hk, hv, kd, vd, conv_w, 4U, ct, word(1.0e-6f)});
    {
      Encoder e(ctx, libs, ct, "test.conv2d.ref." + std::to_string(ct));
      auto &d = e.emit("gdn_layer", "omlx_m5_gdn_conv_silu_2d",
                       {.width = conv_w, .height = ct}, {.width = 64, .height = 4},
                       {{&qkv_buf, 0, qkv_buf.size()},
                        {&conv_wt_buf, 0, conv_wt_buf.size()},
                        {&hist_ref_buf, 0, hist_ref_buf.size()},
                        {&conv_buf, 0, conv_buf.size()}});
      e.dynamic(d, 4, params);
      auto &ch = e.emit("gdn_layer", "omlx_m5_gdn_commit_history",
                        {.width = conv_w}, {.width = 256},
                        {{&hist_ref_buf, 0, hist_ref_buf.size()},
                         {&qkv_buf, 0, qkv_buf.size()}});
      e.dynamic(ch, 2, params);
      auto prof = e.finish(false);
      const std::array calls{MetalCommandInvocation{
          .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
      (void)ctx.execute_batch(calls);
    }
    {
      Encoder e(ctx, libs, ct, "test.conv_vec4." + std::to_string(ct));
      emit_gdn_conv_silu(e, ct, conv_w, {&qkv_buf, 0, qkv_buf.size()},
                         {&conv_wt_buf, 0, conv_wt_buf.size()},
                         {&hist_nax_buf, 0, hist_nax_buf.size()},
                         {&conv_vec_buf, 0, conv_vec_buf.size()}, params);
      auto prof = e.finish(false);
      const std::array calls{MetalCommandInvocation{
          .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
      (void)ctx.execute_batch(calls);
    }
    require(read_u16_buf(conv_buf, std::size_t(ct) * conv_w) ==
                    read_u16_buf(conv_vec_buf, std::size_t(ct) * conv_w) &&
                read_u16_buf(hist_ref_buf, init_hist_words.size()) ==
                    read_u16_buf(hist_nax_buf, init_hist_words.size()),
            "omlx_m5_gdn_conv_silu_vec4_t8 0-ULP mismatch at T=" +
                std::to_string(ct));
  }

  // Materialize full max_t convolved words for standalone recurrence tests.
  const auto conv_words = read_u16_buf(conv_buf, std::size_t(max_t) * conv_w);

  auto run_seq_gpu = [&](U32 live_t, U32 mp) {
    hist_ref_buf.write(std::as_bytes(std::span(init_hist_words)));
    state_seq_buf.write(std::as_bytes(std::span(init_state)));
    Encoder e(ctx, libs, live_t, "test.gdn.seq." + std::to_string(live_t));
    const auto params =
        words({hk, hv, kd, vd, conv_w, 4U, live_t, word(1.0e-6f)});
    emit_gdn_conv_silu(e, live_t, conv_w, {&qkv_buf, 0, qkv_buf.size()},
                       {&conv_wt_buf, 0, conv_wt_buf.size()},
                       {&hist_ref_buf, 0, hist_ref_buf.size()},
                       {&conv_buf, 0, conv_buf.size()}, params);
    emit_gdn_recurrent_forward(
        e, live_t, gv, kd, vd, {&conv_buf, 0, conv_buf.size()},
        {&a_buf, 0, a_buf.size()}, {&b_buf, 0, b_buf.size()},
        {&a_log_buf, 0, a_log_buf.size()},
        {&dt_bias_buf, 0, dt_bias_buf.size()},
        {&state_seq_buf, 0, state_seq_buf.size()},
        {&rec_seq_buf, 0, rec_seq_buf.size()}, params, 8U);
    auto &ng_bf16 = e.emit(
        "gdn_layer", "omlx_m5_gdn_norm_gate",
        {.width = 32, .height = U64(live_t) * hv}, {.width = 32},
        {{&rec_seq_buf, 0, rec_seq_buf.size()},
         {&z_buf, 0, z_buf.size()},
         {&norm_wt_buf, 0, norm_wt_buf.size()},
         {&out_seq_bf16_buf, 0, out_seq_bf16_buf.size()}});
    e.dynamic(ng_bf16, 4, params);
    auto &ng_f16 = e.emit(
        "gdn_layer", "omlx_m5_gdn_norm_gate_f16",
        {.width = 32, .height = U64(mp) * hv}, {.width = 32},
        {{&rec_seq_buf, 0, rec_seq_buf.size()},
         {&z_buf, 0, z_buf.size()},
         {&norm_wt_buf, 0, norm_wt_buf.size()},
         {&out_seq_f16_buf, 0, out_seq_f16_buf.size()}});
    e.dynamic(ng_f16, 4, params);
    auto prof = e.finish(true);
    const std::array calls{MetalCommandInvocation{
        .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
    (void)ctx.execute_batch(calls);
  };

  Json width_reports = Json::array();
  for (U32 live_t : {64U, 73U, 100U, 128U, 512U, 1024U}) {
    const U32 cap_t = (live_t == 100U) ? 1024U : ((live_t == 73U) ? 128U : live_t);
    const U32 mp = ((cap_t + 7U) / 8U) * 8U;
    run_seq_gpu(live_t, mp);
    const auto seq_rec =
        u16_bf16_to_f32(read_u16_buf(rec_seq_buf, std::size_t(live_t) * gv));
    const auto seq_out_bf16 =
        u16_bf16_to_f32(read_u16_buf(out_seq_bf16_buf, std::size_t(live_t) * gv));
    const auto seq_out_f16 =
        u16_f16_to_f32(read_u16_buf(out_seq_f16_buf, std::size_t(live_t) * gv));
    const auto seq_state = read_f32_buf(state_seq_buf, init_state.size());
    const auto seq_hist = read_u16_buf(hist_ref_buf, init_hist_words.size());

    std::vector<float> cpu_state = init_state;
    std::vector<std::uint16_t> cpu_rec_u16;
    cpu_gdn_chunk_c64(live_t, hk, hv, conv_words, a_words, b_words, a_log_words,
                      dt_bias_words, cpu_state, cpu_rec_u16);
    const auto cpu_rec = u16_bf16_to_f32(cpu_rec_u16);

    const auto cpu_rec_err =
        compare_f32(cpu_rec, seq_rec, "cpu_chunk_rec_T" + std::to_string(live_t));
    const auto cpu_st_err =
        compare_f32(cpu_state, seq_state, "cpu_chunk_st_T" + std::to_string(live_t));
    require(cpu_rec_err.max_abs <= 2.5e-2 && cpu_rec_err.rel_l2 <= 5.0e-3 &&
                cpu_st_err.max_abs <= 5.0e-3 && cpu_st_err.rel_l2 <= 3.0e-3,
            "CPU C=64 BF16-NAX chunk reference diverged from sequential recurrence at T=" +
                std::to_string(live_t));

    Json entry = {{"tokens", live_t},
                  {"cpu_rec_max_abs", cpu_rec_err.max_abs},
                  {"cpu_state_max_abs", cpu_st_err.max_abs}};

    if (gpu_pipeline_ready) {
      // A) Standalone `emit_gdn_recurrent_forward` (`omlx_m5_gdn_chunk_{prepare,recurrence}_c64`)
      {
        Encoder e(ctx, libs, cap_t,
                  "test.gdn.nax.standalone.cap" + std::to_string(cap_t) + ".live" +
                      std::to_string(live_t));
        const auto params =
            words({hk, hv, kd, vd, conv_w, 4U, cap_t, word(1.0e-6f)});
        emit_gdn_recurrent_forward(
            e, cap_t, gv, kd, vd, {&conv_buf, 0, conv_buf.size()},
            {&a_buf, 0, a_buf.size()}, {&b_buf, 0, b_buf.size()},
            {&a_log_buf, 0, a_log_buf.size()},
            {&dt_bias_buf, 0, dt_bias_buf.size()},
            {&state_nax_buf, 0, state_nax_buf.size()},
            {&rec_nax_buf, 0, rec_nax_buf.size()}, params, 8U,
            {&ws_buf, 0, ws_buf.size()}, true, hk, hv);
        require(e.profile.gdn_numeric_contract_id ==
                    "gdn.chunk64_bf16_nax.f32_state.conv_bf16.v1",
                "standalone chunk64 NAX contract ID not set");
        auto prof = e.finish(true);
        for (const auto &patch : prof.patches) {
          const U32 v = live_t * patch.multiplier;
          std::memcpy(prof.parameter_data.data() + patch.offset + patch.word * 4U,
                      &v, 4U);
        }
        prof.parameters.write("step", prof.parameter_data);

        std::vector<std::uint16_t> sentinel(std::size_t(max_t) * gv, 0xa5a5U);
        for (bool use_icb : {false, true}) {
          if (use_icb && !prof.icb.has_value()) {
            continue;
          }
          state_nax_buf.write(std::as_bytes(std::span(init_state)));
          rec_nax_buf.write(std::as_bytes(std::span(sentinel)));
          if (use_icb) {
            (void)ctx.replay_icb(*prof.icb);
          } else {
            const std::array calls{MetalCommandInvocation{
                .plan = &prof.plan,
                .bindings = prof.bindings,
                .parameters = &prof.parameters}};
            (void)ctx.execute_batch(calls);
          }
          const auto all_u16 = read_u16_buf(rec_nax_buf, std::size_t(max_t) * gv);
          for (std::size_t idx = std::size_t(live_t) * gv; idx < all_u16.size();
               ++idx) {
            require(all_u16[idx] == 0xa5a5U,
                    "GDN chunk64 NAX overwrote sentinel past live_t");
          }
          const auto gpu_rec = u16_bf16_to_f32(std::span<const std::uint16_t>(
              all_u16.data(), std::size_t(live_t) * gv));
          const auto gpu_state = read_f32_buf(state_nax_buf, init_state.size());

          const auto rec_err = compare_f32(
              gpu_rec, seq_rec, "gpu_nax_rec_T" + std::to_string(live_t));
          const auto st_err = compare_f32(
              gpu_state, seq_state, "gpu_nax_st_T" + std::to_string(live_t));
          const auto vs_cpu_st = compare_f32(
              gpu_state, cpu_state, "gpu_vs_cpu_st_T" + std::to_string(live_t));
          require(rec_err.max_abs <= 2.5e-2 && rec_err.rel_l2 <= 5.0e-3 &&
                      st_err.max_abs <= 5.0e-3 && st_err.rel_l2 <= 3.0e-3 &&
                      vs_cpu_st.max_abs <= 2.0e-3,
                  "GPU C=64 NAX kernel diverged at T=" + std::to_string(live_t));
          entry[use_icb ? "gpu_icb_rec_max_abs" : "gpu_rec_max_abs"] =
              rec_err.max_abs;
          entry[use_icb ? "gpu_icb_state_max_abs" : "gpu_state_max_abs"] =
              st_err.max_abs;
        }
      }

      // B) Full Monolithic Mega-Prefill (`try_emit_gdn_mega_prefill`, both BF16 & FP16 epilogues!)
      const auto standalone_rec_u16 =
          read_u16_buf(rec_nax_buf, std::size_t(live_t) * gv);
      const auto standalone_state = read_f32_buf(state_nax_buf, init_state.size());

      // Compute 0-ULP reference of norm_gate(standalone_rec_u16) so we verify the
      // fused epilogue is 0-ULP bit-for-bit identical to separate norm_gate{_f16}!
      rec_seq_buf.write(std::as_bytes(std::span(standalone_rec_u16)));
      {
        Encoder en(ctx, libs, live_t, "test.epilogue.exact." + std::to_string(live_t));
        const auto p8 = words({hk, hv, kd, vd, conv_w, 4U, live_t, word(1.0e-6f)});
        auto &d1 = en.emit("gdn_layer", "omlx_m5_gdn_norm_gate",
                           {.width = 32, .height = U64(live_t) * hv}, {.width = 32},
                           {{&rec_seq_buf, 0, rec_seq_buf.size()},
                            {&z_buf, 0, z_buf.size()},
                            {&norm_wt_buf, 0, norm_wt_buf.size()},
                            {&out_seq_bf16_buf, 0, out_seq_bf16_buf.size()}});
        en.dynamic(d1, 4, p8);
        auto &d2 = en.emit("gdn_layer", "omlx_m5_gdn_norm_gate_f16",
                           {.width = 32, .height = U64(mp) * hv}, {.width = 32},
                           {{&rec_seq_buf, 0, rec_seq_buf.size()},
                            {&z_buf, 0, z_buf.size()},
                            {&norm_wt_buf, 0, norm_wt_buf.size()},
                            {&out_seq_f16_buf, 0, out_seq_f16_buf.size()}});
        en.dynamic(d2, 4, p8);
        auto prof = en.finish(false);
        const std::array calls{MetalCommandInvocation{
            .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
        (void)ctx.execute_batch(calls);
      }
      const auto exact_epilogue_bf16 =
          read_u16_buf(out_seq_bf16_buf, std::size_t(live_t) * gv);
      const auto exact_epilogue_f16 =
          read_u16_buf(out_seq_f16_buf, std::size_t(live_t) * gv);

      for (bool out_f16 : {false, true}) {
        Encoder e(ctx, libs, cap_t,
                  std::string("test.gdn.mega.") + (out_f16 ? "f16." : "bf16.") +
                      std::to_string(live_t));
        const auto p8 = words({hk, hv, kd, vd, conv_w, 4U, cap_t, word(1.0e-6f)});
        const auto p9 =
            words({hk, hv, kd, vd, conv_w, 4U, cap_t, word(1.0e-6f), mp});
        Ref dst_ref = out_f16 ? Ref{&out_nax_f16_buf, 0, out_nax_f16_buf.size()}
                              : Ref{&out_nax_bf16_buf, 0, out_nax_bf16_buf.size()};
        const bool ok = try_emit_gdn_mega_prefill(
            e, cap_t, gv, hk, hv, kd, vd, conv_w,
            {&qkv_buf, 0, qkv_buf.size()},
            {&conv_wt_buf, 0, conv_wt_buf.size()},
            {&hist_nax_buf, 0, hist_nax_buf.size()},
            {&a_buf, 0, a_buf.size()},
            {&b_buf, 0, b_buf.size()},
            {&a_log_buf, 0, a_log_buf.size()},
            {&dt_bias_buf, 0, dt_bias_buf.size()},
            {&state_nax_buf, 0, state_nax_buf.size()},
            {&z_buf, 0, z_buf.size()},
            {&norm_wt_buf, 0, norm_wt_buf.size()},
            dst_ref, out_f16, {&ws_buf, 0, ws_buf.size()}, true, p8, p9, true);
        require(ok && e.profile.gdn_numeric_contract_id ==
                          "gdn.chunk64_bf16_nax.f32_state.mega_v1",
                "try_emit_gdn_mega_prefill failed to emit mega contract");
        auto prof = e.finish(true);
        for (const auto &patch : prof.patches) {
          const U32 v = live_t * patch.multiplier;
          std::memcpy(prof.parameter_data.data() + patch.offset + patch.word * 4U,
                      &v, 4U);
        }
        prof.parameters.write("step", prof.parameter_data);

        std::vector<std::uint16_t> sentinel(std::size_t(max_t) * gv, 0x5a5aU);
        hist_nax_buf.write(std::as_bytes(std::span(init_hist_words)));
        state_nax_buf.write(std::as_bytes(std::span(init_state)));
        if (out_f16) {
          out_nax_f16_buf.write(std::as_bytes(std::span(sentinel)));
        } else {
          out_nax_bf16_buf.write(std::as_bytes(std::span(sentinel)));
        }
        if (prof.icb.has_value()) {
          (void)ctx.replay_icb(*prof.icb);
        } else {
          const std::array calls{MetalCommandInvocation{
              .plan = &prof.plan,
              .bindings = prof.bindings,
              .parameters = &prof.parameters}};
          (void)ctx.execute_batch(calls);
        }

        // Verify 0-ULP bit-exact match of fused conv + fused epilogue + state + history!
        require(read_f32_buf(state_nax_buf, init_state.size()) == standalone_state,
                "mega prefill state 0-ULP mismatch vs standalone chunk64 at T=" +
                    std::to_string(live_t));
        require(read_u16_buf(hist_nax_buf, init_hist_words.size()) == seq_hist,
                "mega prefill committed history 0-ULP mismatch at T=" +
                    std::to_string(live_t));
        if (!out_f16) {
          const auto all_bf16 =
              read_u16_buf(out_nax_bf16_buf, std::size_t(max_t) * gv);
          require(std::equal(exact_epilogue_bf16.begin(),
                             exact_epilogue_bf16.end(), all_bf16.begin()),
                  "mega_bf16 epilogue 0-ULP mismatch vs gdn_norm_gate at T=" +
                      std::to_string(live_t));
          for (std::size_t idx = std::size_t(live_t) * gv; idx < all_bf16.size();
               ++idx) {
            require(all_bf16[idx] == 0x5a5aU,
                    "mega_bf16 overwrote sentinel past live_t");
          }
          const auto end_err =
              compare_f32(u16_bf16_to_f32(exact_epilogue_bf16), seq_out_bf16,
                          "mega_bf16_vs_seq");
          require(end_err.max_abs <= 8.0e-2 && end_err.rel_l2 <= 1.5e-2,
                  "mega_bf16 diverged from sequential output at T=" +
                      std::to_string(live_t) +
                      " (max_abs=" + std::to_string(end_err.max_abs) +
                      ", rel_l2=" + std::to_string(end_err.rel_l2) + ")");
          entry["mega_bf16_vs_seq_max_abs"] = end_err.max_abs;
        } else {
          const auto all_f16 =
              read_u16_buf(out_nax_f16_buf, std::size_t(max_t) * gv);
          require(std::equal(exact_epilogue_f16.begin(),
                             exact_epilogue_f16.end(), all_f16.begin()),
                  "mega_f16 epilogue 0-ULP mismatch vs gdn_norm_gate_f16 at T=" +
                      std::to_string(live_t));
          const U32 pad_end = std::min(mp, ((live_t + 63U) / 64U) * 64U);
          for (std::size_t idx = std::size_t(live_t) * gv;
               idx < std::size_t(pad_end) * gv; ++idx) {
            require(all_f16[idx] == 0x0000U,
                    "mega_f16 failed to zero-pad tail row up to ceil64(live_t)");
          }
          for (std::size_t idx = std::size_t(pad_end) * gv; idx < all_f16.size();
               ++idx) {
            require(all_f16[idx] == 0x5a5aU,
                    "mega_f16 overwrote sentinel past ceil64(live_t)");
          }
          const auto end_err =
              compare_f32(u16_f16_to_f32(exact_epilogue_f16), seq_out_f16,
                          "mega_f16_vs_seq");
          require(end_err.max_abs <= 8.0e-2 && end_err.rel_l2 <= 1.5e-2,
                  "mega_f16 diverged from sequential output at T=" +
                      std::to_string(live_t) +
                      " (max_abs=" + std::to_string(end_err.max_abs) +
                      ", rel_l2=" + std::to_string(end_err.rel_l2) + ")");
          entry["mega_f16_vs_seq_max_abs"] = end_err.max_abs;
        }
      }
    }
    width_reports.push_back(std::move(entry));
  }

  // Extreme negative log-decay NaN-immunity test (G_63 < -1000 where exp(G_63)==0 -> 0/0 in naive ratio!)
  {
    std::vector<std::uint16_t> strong_a(std::size_t(100U) * hv,
                                        float_to_bf16(32.0f));
    std::vector<std::uint16_t> strong_alog(hv, float_to_bf16(1.8f));
    a_buf.write(std::as_bytes(std::span(strong_a)));
    a_log_buf.write(std::as_bytes(std::span(strong_alog)));

    run_seq_gpu(100U, 104U);
    const auto seq_rec =
        u16_bf16_to_f32(read_u16_buf(rec_seq_buf, std::size_t(100U) * gv));
    const auto seq_st = read_f32_buf(state_seq_buf, init_state.size());

    std::vector<float> cpu_st = init_state;
    std::vector<std::uint16_t> cpu_rec_u16;
    cpu_gdn_chunk_c64(100U, hk, hv, conv_words, strong_a, b_words, strong_alog,
                      dt_bias_words, cpu_st, cpu_rec_u16);
    const auto cpu_err =
        compare_f32(u16_bf16_to_f32(cpu_rec_u16), seq_rec, "strong_decay_cpu");
    const auto cpu_st_err = compare_f32(cpu_st, seq_st, "strong_decay_cpu_st");
    require(cpu_err.max_abs <= 2.5e-2 && cpu_err.rel_l2 <= 1.0e-2 &&
                cpu_st_err.max_abs <= 5.0e-3 && cpu_st_err.rel_l2 <= 1.0e-2,
            "strong-decay NaN-proof CPU check failed (rec_max=" +
                std::to_string(cpu_err.max_abs) +
                ", rec_rel=" + std::to_string(cpu_err.rel_l2) +
                ", st_max=" + std::to_string(cpu_st_err.max_abs) +
                ", st_rel=" + std::to_string(cpu_st_err.rel_l2) + ")");

    if (gpu_pipeline_ready) {
      state_nax_buf.write(std::as_bytes(std::span(init_state)));
      Encoder e(ctx, libs, 128U, "test.gdn.nax.strong_decay");
      const auto params =
          words({hk, hv, kd, vd, conv_w, 4U, 128U, word(1.0e-6f)});
      emit_gdn_recurrent_forward(
          e, 128U, gv, kd, vd, {&conv_buf, 0, conv_buf.size()},
          {&a_buf, 0, a_buf.size()}, {&b_buf, 0, b_buf.size()},
          {&a_log_buf, 0, a_log_buf.size()},
          {&dt_bias_buf, 0, dt_bias_buf.size()},
          {&state_nax_buf, 0, state_nax_buf.size()},
          {&rec_nax_buf, 0, rec_nax_buf.size()}, params, 8U,
          {&ws_buf, 0, ws_buf.size()}, true, hk, hv);
      auto prof = e.finish(false);
      for (const auto &patch : prof.patches) {
        const U32 v = 100U * patch.multiplier;
        std::memcpy(prof.parameter_data.data() + patch.offset + patch.word * 4U,
                    &v, 4U);
      }
      prof.parameters.write("step", prof.parameter_data);
      const std::array calls{MetalCommandInvocation{
          .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
      (void)ctx.execute_batch(calls);
      const auto gpu_rec =
          u16_bf16_to_f32(read_u16_buf(rec_nax_buf, std::size_t(100U) * gv));
      const auto gpu_st = read_f32_buf(state_nax_buf, init_state.size());
      const auto g_err = compare_f32(gpu_rec, seq_rec, "strong_decay_gpu");
      const auto gs_err = compare_f32(gpu_st, seq_st, "strong_decay_gpu_st");
      require(g_err.max_abs <= 2.5e-2 && g_err.rel_l2 <= 1.0e-2 &&
                  gs_err.max_abs <= 5.0e-3 && gs_err.rel_l2 <= 1.0e-2,
              "strong-decay NaN-proof GPU check failed (rec_max=" +
                  std::to_string(g_err.max_abs) +
                  ", rec_rel=" + std::to_string(g_err.rel_l2) +
                  ", st_max=" + std::to_string(gs_err.max_abs) +
                  ", st_rel=" + std::to_string(gs_err.rel_l2) + ")");
    }
  }

  return width_reports;
}

Json test_dv64_zero_ulp_parity(const MetalContext &ctx,
                                const std::filesystem::path &libs,
                                bool gpu_pipeline_ready) {
  // Verify heuristic and OMLX_M5_GDN_STAGE2_SLICE override first.
  {
    ScopedEnv clear_slice("OMLX_M5_GDN_STAGE2_SLICE", nullptr);
    require(gdn_stage2_use_dv64(48U, 80U), "Hv=48 on 80-core must select dv64");
    require(gdn_stage2_use_dv64(48U, 64U), "Hv=48 on 64-core must select dv64");
    require(!gdn_stage2_use_dv64(32U, 80U), "Hv=32 on 80-core must select dv32");
    require(gdn_stage2_use_dv64(32U, 64U), "Hv=32 on 64-core must select dv64");
    require(gdn_stage2_use_dv64(48U, 0U), "default gpu_cores=80 must select dv64 for Hv=48");
    require(!gdn_stage2_use_dv64(32U, 0U), "default gpu_cores=80 must select dv32 for Hv=32");
  }
  {
    ScopedEnv force_64("OMLX_M5_GDN_STAGE2_SLICE", "64");
    require(gdn_stage2_use_dv64(32U, 80U), "OMLX_M5_GDN_STAGE2_SLICE=64 override failed");
  }
  {
    ScopedEnv force_32("OMLX_M5_GDN_STAGE2_SLICE", "32");
    require(!gdn_stage2_use_dv64(48U, 80U), "OMLX_M5_GDN_STAGE2_SLICE=32 override failed");
  }

  if (!gpu_pipeline_ready) {
    return {{"skipped", true}};
  }

  Json reports = Json::array();
  constexpr U32 kd = 128U;
  constexpr U32 vd = 128U;
  constexpr U32 max_t = 2048U;

  struct Geom {
    const char *name;
    U32 hk;
    U32 hv;
  };
  for (const auto &g : std::array<Geom, 2>{{{"27b_hv48", 16U, 48U}, {"35b_hv32", 16U, 32U}}}) {
    const U32 hk = g.hk;
    const U32 hv = g.hv;
    const U32 gv = hv * vd;
    const U32 conv_w = 2U * hk * kd + gv;

    std::vector<std::uint16_t> qkv_words(std::size_t(max_t) * conv_w);
    std::vector<std::uint16_t> conv_wt_words(std::size_t(conv_w) * 4U);
    std::vector<std::uint16_t> init_hist_words(3U * conv_w);
    std::vector<std::uint16_t> z_words(std::size_t(max_t) * gv);
    std::vector<std::uint16_t> norm_wt_words(vd);
    std::vector<std::uint16_t> a_words(std::size_t(max_t) * hv);
    std::vector<std::uint16_t> b_words(std::size_t(max_t) * hv);
    std::vector<std::uint16_t> a_log_words(hv);
    std::vector<std::uint16_t> dt_bias_words(hv);
    std::vector<float> init_state(std::size_t(gv) * kd);

    for (std::size_t i = 0; i < qkv_words.size(); ++i) {
      qkv_words[i] = float_to_bf16(float(int((i * 37ULL + 19ULL + hv) % 251ULL) - 125) / 64.0f);
    }
    for (std::size_t i = 0; i < conv_wt_words.size(); ++i) {
      conv_wt_words[i] = float_to_bf16(float(int((i * 17ULL + 5ULL) % 89ULL) - 44) / 64.0f);
    }
    for (std::size_t i = 0; i < init_hist_words.size(); ++i) {
      init_hist_words[i] = float_to_bf16(float(int((i * 23ULL + 11ULL) % 131ULL) - 65) / 64.0f);
    }
    for (std::size_t i = 0; i < z_words.size(); ++i) {
      z_words[i] = float_to_bf16(float(int((i * 31ULL + 13ULL) % 197ULL) - 98) / 64.0f);
    }
    for (U32 d = 0; d < vd; ++d) {
      norm_wt_words[d] = float_to_bf16(0.75f + 0.5f * float(d % 17U) / 16.0f);
    }
    for (std::size_t i = 0; i < a_words.size(); ++i) {
      a_words[i] = float_to_bf16(float(int((i * 13ULL + 7ULL) % 101ULL) - 55) / 16.0f);
      b_words[i] = float_to_bf16(float(int((i * 29ULL + 3ULL) % 97ULL) - 48) / 16.0f);
    }
    for (U32 h = 0; h < hv; ++h) {
      a_log_words[h] = float_to_bf16(-1.2f + 0.05f * float(h));
      dt_bias_words[h] = float_to_bf16(-0.5f + 0.03f * float(h));
    }
    for (std::size_t i = 0; i < init_state.size(); ++i) {
      init_state[i] = float(int((i * 43ULL + 11ULL) % 199ULL) - 99) / 256.0f;
    }

    auto qkv_buf = ctx.upload(std::as_bytes(std::span(qkv_words)));
    auto conv_wt_buf = ctx.upload(std::as_bytes(std::span(conv_wt_words)));
    auto hist_32_buf = ctx.make_buffer(init_hist_words.size() * 2ULL);
    auto hist_64_buf = ctx.make_buffer(init_hist_words.size() * 2ULL);
    auto z_buf = ctx.upload(std::as_bytes(std::span(z_words)));
    auto norm_wt_buf = ctx.upload(std::as_bytes(std::span(norm_wt_words)));
    auto a_buf = ctx.upload(std::as_bytes(std::span(a_words)));
    auto b_buf = ctx.upload(std::as_bytes(std::span(b_words)));
    auto a_log_buf = ctx.upload(std::as_bytes(std::span(a_log_words)));
    auto dt_bias_buf = ctx.upload(std::as_bytes(std::span(dt_bias_words)));
    auto state_32_buf = ctx.make_buffer(init_state.size() * sizeof(float));
    auto state_64_buf = ctx.make_buffer(init_state.size() * sizeof(float));
    auto rec_32_buf = ctx.make_buffer(std::size_t(max_t) * gv * 2ULL);
    auto rec_64_buf = ctx.make_buffer(std::size_t(max_t) * gv * 2ULL);
    const auto ws_layout = gdn_chunk_nax_workspace_layout(max_t, hk, hv);
    auto ws_buf = ctx.make_private_buffer(ws_layout.total_bytes);

    for (U32 live_t : {64U, 128U, 193U, 2048U}) {
      const U32 mp = ((live_t + 7U) / 8U) * 8U;
      const auto p9 = words({hk, hv, kd, vd, conv_w, 4U, live_t, word(1.0e-6f), mp});
      const auto ws_t = gdn_chunk_nax_workspace_layout(live_t, hk, hv);
      const Ref qk_ref{&ws_buf, 0, ws_t.qk_bytes};
      const Ref ctrl_ref{&ws_buf, ws_t.qk_bytes, ws_t.ctrl_bytes};

      // Direct Stage 1 + Stage 2 raw recurrent_out & state0 0-ULP comparison: dv32 vs dv64!
      auto run_stage2 = [&](bool use_dv64, MetalBuffer &st_buf, MetalBuffer &rec_buf) {
        st_buf.write(std::as_bytes(std::span(init_state)));
        std::vector<std::uint16_t> sentinel(std::size_t(live_t) * gv, 0xA5A5U);
        rec_buf.write(std::as_bytes(std::span(sentinel)));
        hist_32_buf.write(std::as_bytes(std::span(init_hist_words)));
        Encoder e(ctx, libs, live_t,
                  std::string("test.gdn.") + g.name + (use_dv64 ? ".dv64.T" : ".dv32.T") +
                      std::to_string(live_t));
        auto &prep = e.emit(
            "gdn_chunk_nax", "omlx_m5_gdn_chunk_prepare_conv_c64",
            {.width = U64(hk) * 128U, .height = ws_t.max_chunks}, {.width = 128U},
            {{&qkv_buf, 0, qkv_buf.size()},
             {&conv_wt_buf, 0, conv_wt_buf.size()},
             {&hist_32_buf, 0, hist_32_buf.size()},
             {&a_buf, 0, a_buf.size()},
             {&b_buf, 0, b_buf.size()},
             {&a_log_buf, 0, a_log_buf.size()},
             {&dt_bias_buf, 0, dt_bias_buf.size()},
             qk_ref,
             ctrl_ref});
        e.dynamic(prep, 9, p9);
        auto &rec = e.emit(
            "gdn_chunk_nax",
            use_dv64 ? "omlx_m5_gdn_chunk_recurrence_conv_dv64_c64"
                     : "omlx_m5_gdn_chunk_recurrence_conv_dv32_c64",
            {.width = U64(hv) * 128U, .height = use_dv64 ? 2U : 4U}, {.width = 128U},
            {{&qkv_buf, 0, qkv_buf.size()},
             {&conv_wt_buf, 0, conv_wt_buf.size()},
             {&hist_32_buf, 0, hist_32_buf.size()},
             qk_ref,
             ctrl_ref,
             {&st_buf, 0, st_buf.size()},
             {&rec_buf, 0, rec_buf.size()}});
        e.dynamic(rec, 7, p9);
        auto prof = e.finish(false);
        const std::array calls{MetalCommandInvocation{
            .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
        (void)ctx.execute_batch(calls);
      };

      run_stage2(false, state_32_buf, rec_32_buf);
      run_stage2(true, state_64_buf, rec_64_buf);

      const auto rec32 = read_u16_buf(rec_32_buf, std::size_t(live_t) * gv);
      const auto rec64 = read_u16_buf(rec_64_buf, std::size_t(live_t) * gv);
      const auto st32 = read_f32_buf(state_32_buf, init_state.size());
      const auto st64 = read_f32_buf(state_64_buf, init_state.size());

      if (rec32 != rec64) {
        std::size_t first_i = 0;
        while (first_i < rec32.size() && rec32[first_i] == rec64[first_i]) ++first_i;
        require(false,
                std::string(g.name) + " dv64 vs dv32 recurrent_out 0-ULP mismatch at T=" +
                    std::to_string(live_t) + " i=" + std::to_string(first_i) +
                    " rec32=" + std::to_string(rec32[first_i]) +
                    " rec64=" + std::to_string(rec64[first_i]));
      }
      require(st32 == st64,
              std::string(g.name) + " dv64 vs dv32 state0 0-ULP mismatch at T=" +
                  std::to_string(live_t));

      // Also verify try_emit_gdn_mega_prefill with OMLX_M5_GDN_STAGE2_SLICE=64 vs =32!
      const auto p8 = words({hk, hv, kd, vd, conv_w, 4U, live_t, word(1.0e-6f)});
      auto run_mega = [&](const char *slice_val, MetalBuffer &st_buf,
                          MetalBuffer &hist_buf, MetalBuffer &out_buf) {
        ScopedEnv env_slice("OMLX_M5_GDN_STAGE2_SLICE", slice_val);
        st_buf.write(std::as_bytes(std::span(init_state)));
        hist_buf.write(std::as_bytes(std::span(init_hist_words)));
        Encoder e(ctx, libs, live_t,
                  std::string("test.gdn.mega.") + g.name + ".s" + slice_val + ".T" +
                      std::to_string(live_t));
        const bool ok = try_emit_gdn_mega_prefill(
            e, live_t, gv, hk, hv, kd, vd, conv_w,
            {&qkv_buf, 0, qkv_buf.size()},
            {&conv_wt_buf, 0, conv_wt_buf.size()},
            {&hist_buf, 0, hist_buf.size()},
            {&a_buf, 0, a_buf.size()},
            {&b_buf, 0, b_buf.size()},
            {&a_log_buf, 0, a_log_buf.size()},
            {&dt_bias_buf, 0, dt_bias_buf.size()},
            {&st_buf, 0, st_buf.size()},
            {&z_buf, 0, z_buf.size()},
            {&norm_wt_buf, 0, norm_wt_buf.size()},
            {&out_buf, 0, out_buf.size()}, false, {&ws_buf, 0, ws_buf.size()},
            true, p8, p9, false, 80U);
        require(ok, "try_emit_gdn_mega_prefill failed in dv64 test");
        auto prof = e.finish(false);
        const std::array calls{MetalCommandInvocation{
            .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
        (void)ctx.execute_batch(calls);
      };
      run_mega("32", state_32_buf, hist_32_buf, rec_32_buf);
      run_mega("64", state_64_buf, hist_64_buf, rec_64_buf);
      require(read_u16_buf(rec_32_buf, std::size_t(live_t) * gv) ==
                  read_u16_buf(rec_64_buf, std::size_t(live_t) * gv),
              std::string(g.name) + " mega dv64 vs dv32 output 0-ULP mismatch at T=" +
                  std::to_string(live_t));
      require(read_f32_buf(state_32_buf, init_state.size()) ==
                  read_f32_buf(state_64_buf, init_state.size()),
              std::string(g.name) + " mega dv64 vs dv32 state0 0-ULP mismatch at T=" +
                  std::to_string(live_t));
      require(read_u16_buf(hist_32_buf, init_hist_words.size()) ==
                  read_u16_buf(hist_64_buf, init_hist_words.size()),
              std::string(g.name) + " mega dv64 vs dv32 history 0-ULP mismatch at T=" +
                  std::to_string(live_t));

      reports.push_back({{"geom", g.name},
                         {"hk", hk},
                         {"hv", hv},
                         {"tokens", live_t},
                         {"zero_ulp_exact", true}});
    }
  }

  return reports;
}

Json run_qwen_fixture_integration(const MetalContext &ctx,
                                  const std::filesystem::path &root,
                                  const std::filesystem::path &libs,
                                  bool gpu_pipeline_ready,
                                  std::string_view label) {
  const auto cfg = load_qwen_model_config(root / "source/config.json");
  const auto artifact = ArtifactModel::open(root / "model.omlxm5", true);
  const auto weights =
      materialize_model(artifact, plan_model_materialization(artifact, ctx), ctx);
  const auto layers = build_qwen_layer_catalog(artifact, cfg, 1024, 64);
  const std::array<U32, 2> taps{0, 1};

  auto make_toks = [&](U32 n, U32 seed) {
    std::vector<U32> out(n);
    for (U32 i = 0; i < n; ++i) {
      out[i] = (73U * (i + seed) + 17U) % U32(cfg.vocab_size);
    }
    return out;
  };

  ScopedEnv env_disable("OMLX_M5_DISABLE_GDN_CHUNK_NAX", "1");
  QwenForward seq_fwd(ctx, weights, layers, cfg, 1024, libs, taps, false, true,
                      false, std::nullopt, false, false, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1,
                      omlx::m5::AttentionNumericPolicy::RowExactV1);

  require(seq_fwd.inspect_profile(64).gdn_numeric_contract_id ==
                  "gdn.sequential.simd32.f32_state.conv_bf16.v1" &&
              seq_fwd.inspect_profile(1024).gdn_numeric_contract_id ==
                  "gdn.sequential.simd32.f32_state.conv_bf16.v1",
          std::string(label) + ": disabled GDN chunk NAX left sequential contract");

  if (!gpu_pipeline_ready) {
    return {{"fixture", label},
            {"qwen_chunk_nax_executed", false},
            {"sequential_fallback_verified", true}};
  }

  ScopedEnv env_enable("OMLX_M5_DISABLE_GDN_CHUNK_NAX", "0");
  ScopedEnv env_force("OMLX_M5_FORCE_GDN_CHUNK_NAX", "1");
  QwenForward chunk_fwd(ctx, weights, layers, cfg, 1024, libs, taps, false,
                        true, false, std::nullopt, false, false, std::nullopt,
                        LinearNumericPolicy::RowExactPortableV1,
                        SamplingContract::PositionCoupledV1,
                        omlx::m5::AttentionNumericPolicy::RowExactV1);
  QwenForward chunk_teacher(ctx, weights, layers, cfg, 1024, libs, taps, false,
                            true, false, std::nullopt, false, false,
                            std::nullopt,
                            LinearNumericPolicy::RowExactPortableV1,
                            SamplingContract::PositionCoupledV1,
                            omlx::m5::AttentionNumericPolicy::RowExactV1);

  // 1. T < 64 and verify MUST stay 0-ULP sequential!
  for (U32 cap : {1U, 8U, 32U}) {
    require(chunk_fwd.inspect_profile(cap).gdn_numeric_contract_id ==
                "gdn.sequential.simd32.f32_state.conv_bf16.v1",
            std::string(label) + ": T < 64 profile switched GDN contract");
  }
  require(chunk_fwd.inspect_profile(8, true).gdn_numeric_contract_id ==
              "gdn.sequential.simd32.f32_state.conv_bf16.v1",
          std::string(label) + ": verify profile switched GDN contract");
  for (U32 cap : {64U, 128U, 512U, 1024U}) {
    require(chunk_fwd.inspect_profile(cap).gdn_numeric_contract_id ==
                "gdn.chunk64_bf16_nax.f32_state.mega_v1",
            std::string(label) + ": T >= 64 profile did not install mega_v1 contract");
  }

  const GpuSamplingOptions coupled_opts{
      .contract = SamplingContract::PositionCoupledV1};

  for (U32 short_t : {1U, 8U, 16U, 24U, 32U}) {
    seq_fwd.reset();
    chunk_fwd.reset();
    const auto toks = make_toks(short_t, short_t * 11U);
    const auto rs = seq_fwd.run(toks, coupled_opts, true);
    const auto rc = chunk_fwd.run(toks, coupled_opts, true);
    require(rs.logits == rc.logits && rs.hidden == rc.hidden &&
                seq_fwd.inspect_recurrent_state() ==
                    chunk_fwd.inspect_recurrent_state(),
            std::string(label) + ": capacity < 64 0-ULP broken at T=" +
                std::to_string(short_t));
  }

  // 2. Prefill parity across T = {63, 64, 73, 100, 128, 512, 1024} + direct == ICB!
  double max_logit_abs = 0.0;
  for (U32 prefill_t : {63U, 64U, 73U, 100U, 128U, 512U, 1024U}) {
    const auto toks = make_toks(prefill_t, prefill_t * 3U);
    seq_fwd.reset();
    chunk_fwd.reset();
    const auto rs = seq_fwd.run(toks, coupled_opts, true, true);
    const auto rc_icb = chunk_fwd.run(toks, coupled_opts, true, true);
    chunk_fwd.reset();
    const auto rc_dir = chunk_fwd.run(toks, coupled_opts, true, false);
    require(rc_icb.logits == rc_dir.logits && rc_icb.hidden == rc_dir.hidden,
            std::string(label) + ": ICB vs direct mismatch at T=" +
                std::to_string(prefill_t));
    const auto logit_err = compare_f32(
        rc_icb.logits, rs.logits,
        std::string(label) + "_logits_T" + std::to_string(prefill_t));
    const auto hid_err = compare_f32(
        rc_icb.hidden, rs.hidden,
        std::string(label) + "_hidden_T" + std::to_string(prefill_t));
    max_logit_abs = std::max(max_logit_abs, logit_err.max_abs);
    require(logit_err.max_abs <= 0.12 && logit_err.rel_l2 <= 0.015 &&
                hid_err.max_abs <= 0.12 && hid_err.rel_l2 <= 0.015,
            std::string(label) + ": prefill accuracy out of tolerance at T=" +
                std::to_string(prefill_t));
  }

  // 3. Coupled AR == DFlash2 verify/commit 0-ULP post-prefill parity after T=100 chunked prefill!
  const auto prompt100 = make_toks(100U, 900U);
  const auto draft8 = make_toks(8U, 999U);
  chunk_fwd.reset();
  chunk_teacher.reset();
  (void)chunk_fwd.run(prompt100, coupled_opts);
  (void)chunk_teacher.run(prompt100, coupled_opts);
  require(chunk_fwd.inspect_recurrent_state() ==
              chunk_teacher.inspect_recurrent_state(),
          std::string(label) + ": coupled prefill recurrent state diverged");
  const auto ver = chunk_fwd.verify(draft8, coupled_opts, true);
  (void)chunk_fwd.commit_prefix(5U);
  for (U32 r = 0; r < 5U; ++r) {
    const auto step = chunk_teacher.run(
        std::span<const U32>(draft8).subspan(r, 1U), coupled_opts, true);
    require(std::equal(step.logits.begin(), step.logits.end(),
                       ver.logits.begin() + std::size_t(r) * cfg.vocab_size),
            std::string(label) + ": post-chunk-prefill verify/AR 0-ULP logits mismatch");
  }
  require(chunk_fwd.inspect_recurrent_state() ==
              chunk_teacher.inspect_recurrent_state(),
          std::string(label) + ": post-chunk-prefill commit/AR 0-ULP state mismatch");

  // 4. Secondary 16-row slot delegated prefill (T=24 -> tb8, T=54 -> tb8, T=73 & T=100 -> chunk64_mega)
  //    MUST match standalone 1024-row solo 0-ULP bit-for-bit!
  QwenForward primary_slot(
      ctx, weights, layers, cfg, 1024, libs, taps, false, true, false,
      std::nullopt, false, true, std::nullopt,
      LinearNumericPolicy::RowExactPortableV1,
      SamplingContract::PositionCoupledV1,
      omlx::m5::AttentionNumericPolicy::RowExactV1, false, "bf16", nullptr);
  QwenForward secondary_slot(ctx, weights, layers, cfg, 1024, libs, taps, false,
                             true, false, std::nullopt, false, true,
                             std::nullopt,
                             LinearNumericPolicy::RowExactPortableV1,
                             SamplingContract::PositionCoupledV1,
                             omlx::m5::AttentionNumericPolicy::RowExactV1,
                             false, "bf16", &primary_slot);
  require(secondary_slot.memory_plan().workspace_rows == 16U,
          "secondary_slot workspace_rows expected 16");
  for (U32 deleg_t : {24U, 54U, 73U, 100U}) {
    const auto p_toks = make_toks(deleg_t, 1200U + deleg_t);
    secondary_slot.reset();
    chunk_teacher.reset();
    const auto r_sec = secondary_slot.run(p_toks, coupled_opts);
    const auto r_solo = chunk_teacher.run(p_toks, coupled_opts);
    require(r_sec.token == r_solo.token &&
                secondary_slot.read_last_logits_bf16() ==
                    chunk_teacher.read_last_logits_bf16() &&
                secondary_slot.inspect_recurrent_state() ==
                    chunk_teacher.inspect_recurrent_state(),
            std::string(label) +
                ": secondary 16-row delegated prefill 0-ULP mismatch vs solo at T=" +
                std::to_string(deleg_t));
  }

  return {{"fixture", label},
          {"qwen_chunk_nax_executed", true},
          {"max_logit_abs_vs_seq", max_logit_abs},
          {"dispatch_count_t1024", chunk_fwd.inspect_profile(1024).dispatch_count},
          {"seq_dispatch_count_t1024", seq_fwd.inspect_profile(1024).dispatch_count},
          {"short_t_0ulp_verified", true},
          {"coupled_verify_0ulp_verified", true},
          {"secondary_delegated_0ulp_verified", true}};
}

Json test_qwen_forward_integration(const MetalContext &ctx,
                                   const std::filesystem::path &fixtures,
                                   const std::filesystem::path &libs,
                                   bool gpu_pipeline_ready) {
  Json reports = Json::array();
  reports.push_back(run_qwen_fixture_integration(
      ctx, fixtures / "affine", libs, gpu_pipeline_ready, "affine"));
  {
    ScopedEnv force_nax("OMLX_M5_FORCE_NAX_APPLE10", "1");
    reports.push_back(run_qwen_fixture_integration(
        ctx, fixtures / "mxfp4", libs, gpu_pipeline_ready, "mxfp4"));
  }
  return reports;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "usage: omlx_m5_gdn_chunk_nax_tests <qwen_fixtures> <metallib_dir>");
    }
    const std::filesystem::path fixtures = argv[1];
    const std::filesystem::path libs = argv[2];
    {
      // Regression check for TEST-3: compare_f32 must reject NaN in expected.
      const std::array<float, 2> finite_vals{0.0f, 1.0f};
      const std::array<float, 2> nan_vals{0.0f, std::numeric_limits<float>::quiet_NaN()};
      bool rejected = false;
      try {
        (void)compare_f32(finite_vals, nan_vals, "nan_expected_guard");
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      require(rejected, "compare_f32 failed to reject NaN in expected");
    }
    auto ctx = MetalContext::create();

    bool gpu_pipeline_ready = false;
    const auto inv =
        test_memory_and_pipeline_invariants(ctx, libs, gpu_pipeline_ready);
    const auto op = test_operator_parity(ctx, libs, gpu_pipeline_ready);
    const auto dv64 = test_dv64_zero_ulp_parity(ctx, libs, gpu_pipeline_ready);
    const auto e2e =
        test_qwen_forward_integration(ctx, fixtures, libs, gpu_pipeline_ready);

    std::cout << Json({{"schema", "omlx-m5-t1b-gdn-mega-c64-v1"},
                       {"device", ctx.capabilities().device_name},
                       {"invariants", inv},
                       {"operator_parity", op},
                       {"dv64_zero_ulp_parity", dv64},
                       {"qwen_integration", e2e},
                       {"passed", true}})
                     .dump(2)
              << '\n';
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
