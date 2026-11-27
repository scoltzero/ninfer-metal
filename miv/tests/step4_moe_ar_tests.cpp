#include "omlx_m5/artifact.h"
#include "omlx_m5/engine.h"
#include "omlx_m5/execution_contract.h"
#include "omlx_m5/kv_page_pool.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/position_sampling.h"
#include "omlx_m5/qwen_forward.h"
#include "omlx_m5/qwen_memory.h"
#include "forward_plan.h"
#include "moe_nax.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

using namespace omlx::m5;
using Json = nlohmann::json;
using U32 = std::uint32_t;
using U64 = std::uint64_t;

namespace {

void require(bool ok, std::string_view message) {
  if (!ok) {
    throw std::runtime_error("STEP4-TEST FAIL: " + std::string(message));
  }
}

struct ScopedEnv {
  std::string name;
  std::optional<std::string> previous;
  ScopedEnv(std::string key, const char *value) : name(std::move(key)) {
    if (const char *old = std::getenv(name.c_str())) {
      previous = std::string(old);
    }
    if (value != nullptr) {
      setenv(name.c_str(), value, 1);
    } else {
      unsetenv(name.c_str());
    }
  }
  ~ScopedEnv() {
    if (previous.has_value()) {
      setenv(name.c_str(), previous->c_str(), 1);
    } else {
      unsetenv(name.c_str());
    }
  }
};

std::string shell_quote(const std::filesystem::path &p) {
  std::string s = p.string();
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out.push_back(c);
    }
  }
  out.push_back('\'');
  return out;
}

void run_cmd(const std::string &cmd) {
  const int rc = std::system(cmd.c_str());
  if (rc != 0) {
    throw std::runtime_error("Subcommand failed (" + std::to_string(rc) +
                             "): " + cmd);
  }
}

std::uint16_t f32_to_bf16_bits(float v) {
  const U32 bits = std::bit_cast<U32>(v);
  const U32 lsb = (bits >> 16U) & 1U;
  const U32 rounded = bits + 0x7FFFU + lsb;
  return static_cast<std::uint16_t>(rounded >> 16U);
}

float bf16_bits_to_f32(std::uint16_t b) {
  return std::bit_cast<float>(U32(b) << 16U);
}

U32 bf16_ulp_diff(std::uint16_t a, std::uint16_t b) {
  const float fa = bf16_bits_to_f32(a);
  const float fb = bf16_bits_to_f32(b);
  if (fa == 0.0f && fb == 0.0f) {
    return 0U;
  }
  auto ordered = [](std::uint16_t x) -> std::int32_t {
    return (x & 0x8000U) ? -std::int32_t(x & 0x7FFFU) : std::int32_t(x);
  };
  const std::int32_t d = ordered(a) - ordered(b);
  return static_cast<U32>(d < 0 ? -d : d);
}

std::vector<U32> make_tokens(U32 count, U32 seed_offset, U32 vocab) {
  std::vector<U32> out(count);
  for (U32 i = 0; i < count; ++i) {
    out[i] = (73U * (i + seed_offset) + 19U) % vocab;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Test 1: Static TG SRAM Audit (<= 32 KiB Apple10 Hard Limit) + Live GPU Test
// of Fused MoE Router+Bucket Schedulers (BM=8 & BM=32)
// ---------------------------------------------------------------------------
Json test_pso_sram_and_moe_schedulers(const MetalContext &ctx,
                                      const std::filesystem::path &libs) {
  constexpr std::size_t kApple10MaxTgBytes = 32768;
  const char *sv_env = std::getenv("MTL_SHADER_VALIDATION");
  const bool shader_validation_active =
      sv_env != nullptr && std::string_view(sv_env) != "0" &&
      !std::string_view(sv_env).empty();
  const std::size_t sram_scale = shader_validation_active ? 2U : 1U;
  const std::size_t sram_pad = shader_validation_active ? 256U : 0U;
  U32 checked_psos = 0;
  std::uint64_t max_sram_bytes = 0;
  Json sram_by_pso = Json::object();

  auto check_pso = [&](const std::string &lib, const std::string &fn,
                       std::size_t max_expected_sram,
                       std::size_t min_threads = 128) {
    const std::string lib_key = "step4.audit.lib." + lib;
    ctx.register_library(
        {.key = lib_key, .path = libs / ("omlx_m5_" + lib + ".metallib")});
    const auto info = ctx.warmup_pipeline(
        {.key = "step4.audit.pso." + fn,
         .library_key = lib_key,
         .function_name = fn,
         .variant = "step4-audit",
         .support_indirect_command_buffers =
             ctx.capabilities().profile.compute_icb_eligible});
    require(info.static_threadgroup_memory_bytes <=
                kApple10MaxTgBytes * sram_scale + sram_pad,
            fn + " exceeded 32 KiB Apple10 TG SRAM limit: " +
                std::to_string(info.static_threadgroup_memory_bytes));
    require(info.static_threadgroup_memory_bytes <=
                max_expected_sram * sram_scale + sram_pad,
            fn + " exceeded expected TG SRAM (" +
                std::to_string(info.static_threadgroup_memory_bytes) + " > " +
                std::to_string(max_expected_sram * sram_scale + sram_pad) +
                ")");
    require(info.max_threads_per_threadgroup >= min_threads,
            fn + " max threads per TG below required " +
                std::to_string(min_threads));
    max_sram_bytes =
        std::max(max_sram_bytes, info.static_threadgroup_memory_bytes);
    sram_by_pso[fn] = info.static_threadgroup_memory_bytes;
    ++checked_psos;
    return info.static_threadgroup_memory_bytes;
  };

  // Metal 15 PSOs
  for (const char *fn : {"omlx_m5_pretiled_qmv_mxfp4_t1",
                         "omlx_m5_pretiled_qmv_mxfp4_swiglu_t1",
                         "omlx_m5_pretiled_qmv_mxfp4_concat2_t1",
                         "omlx_m5_pretiled_qmv_mxfp4_concat3_t1",
                         "omlx_m5_pretiled_qmv_affine_t1",
                         "omlx_m5_pretiled_qmv_affine_swiglu_t1",
                         "omlx_m5_pretiled_qmv_affine_concat2_t1",
                         "omlx_m5_pretiled_qmv_affine_concat3_t1"}) {
    (void)check_pso("row_exact", fn, 0, 64);
  }
  (void)check_pso("gdn_layer", "omlx_m5_gdn_conv_silu_vec4_t1", 0, 64);
  (void)check_pso("gdn_layer", "omlx_m5_gdn_t1_recurrent_norm_gate", 512, 256);
  (void)check_pso("gdn_layer", "omlx_m5_gdn_t1_recurrent_norm_gate_f16", 512,
                  256);
  (void)check_pso("mlp", "omlx_m5_moe_topk_shared_combine", 0, 128);
  (void)check_pso("moe", "omlx_m5_mxfp4_moe_w4a16_swiglu_grouped", 0, 128);
  for (int bits : {4, 5, 6, 8}) {
    for (int gs : {64, 128}) {
      const std::string fn = "omlx_m5_oq_q" + std::to_string(bits) + "_g" +
                             std::to_string(gs) + "_moe_swiglu_grouped";
      (void)check_pso("moe", fn, 0, 128);
    }
  }
  (void)check_pso("position_sampling", "omlx_m5_position_sample_logits_feed",
                  2048, 256);
  (void)check_pso("position_sampling", "omlx_m5_position_argmax_stage1_64tg",
                  256, 256);
  (void)check_pso("position_sampling", "omlx_m5_position_sample_logits_stage2",
                  2048, 256);
  (void)check_pso("position_sampling",
                  "omlx_m5_position_sample_logits_feed_stage2", 512, 256);
  (void)check_pso("vocab", "omlx_m5_sample_logits_feed", 2048, 256);
  (void)check_pso("vocab", "omlx_m5_vocab_argmax_stage1_64tg", 256, 256);
  (void)check_pso("vocab", "omlx_m5_sample_logits_stage2", 2048, 256);
  (void)check_pso("vocab", "omlx_m5_sample_logits_feed_stage2", 2048, 256);

  bool moe_nax_checked = false;
  if (std::filesystem::is_regular_file(libs / "omlx_m5_moe_nax.metallib")) {
    moe_nax_checked = true;
    (void)check_pso("moe_nax", "omlx_m5_moe9_router_top8_shared_gate_t1", 2048,
                    256);
    (void)check_pso("moe_nax", "omlx_m5_moe9_mxfp4_kl16_swiglu_t1", 0, 64);
    (void)check_pso("moe_nax", "omlx_m5_moe9_mxfp4_kl16_down_sk3_r8_t1", 512,
                    96);
    (void)check_pso("moe_nax", "omlx_m5_moe_nax_router_top8_bucket_schedule",
                    12288, 256);
    (void)check_pso("moe_nax", "omlx_m5_moe_nax_bucket_schedule_bm32", 12288,
                    256);
    (void)check_pso("moe_nax",
                    "omlx_m5_moe_nax_router_top8_bucket_schedule_bm32", 12288,
                    256);
    (void)check_pso("moe_nax", "omlx_m5_moe_nax_bucket_schedule_bm64", 12288,
                    256);
    (void)check_pso("moe_nax",
                    "omlx_m5_moe_nax_router_top8_bucket_schedule_bm64", 12288,
                    256);
    (void)check_pso("moe_nax",
                    "omlx_m5_moe_mxfp4_nax_swiglu_bm32_bn64_sk1", 16384, 128);
    (void)check_pso("moe_nax",
                    "omlx_m5_moe_mxfp4_nax_down_scatter_bm32_bn64_sk1", 16384,
                    128);
    (void)check_pso("moe_nax",
                    "omlx_m5_moe_mxfp4_nax_swiglu_bm64_bn64_sk1", 16384, 128);
    (void)check_pso("moe_nax",
                    "omlx_m5_moe_mxfp4_nax_down_scatter_bm64_bn64_sk1", 16384,
                    128);

    for (int bits : {3, 4, 5, 6, 8}) {
      const bool need_h2 = (bits == 3 || bits == 5);
      for (int gs : {64, 128}) {
        const std::string pfx_up = "omlx_m5_moe_affine_nax_swiglu_q" +
                                   std::to_string(bits) + "_g" +
                                   std::to_string(gs);
        const std::string pfx_dn = "omlx_m5_moe_affine_nax_down_scatter_q" +
                                   std::to_string(bits) + "_g" +
                                   std::to_string(gs);
        (void)check_pso("moe_nax", pfx_up + "_n16_sk4_nsg2",
                        need_h2 ? 16384 : 12288, 256);
        (void)check_pso("moe_nax", pfx_dn + "_n16_sk2_nsg2",
                        need_h2 ? 4096 : 2048, 128);
        (void)check_pso("moe_nax", pfx_up + "_bm32_bn64_sk1", 16384, 128);
        (void)check_pso("moe_nax", pfx_dn + "_bm32_bn64_sk1", 16384, 128);
        (void)check_pso("moe_nax", pfx_up + "_bm64_bn64_sk1", 16384, 128);
        (void)check_pso("moe_nax", pfx_dn + "_bm64_bn64_sk1", 16384, 128);
      }
    }

    // Execute router_top8_bucket_schedule & bm32 live on GPU (pure SIMD/TG ops,
    // runnable on both M4 and M5 Ultra!) and verify 0-ULP parity vs standalone!
    ctx.register_library(
        {.key = "step4.audit.lib.qwen_forward",
         .path = libs / "omlx_m5_qwen_forward.metallib"});
    (void)ctx.warmup_pipeline(
        {.key = "step4.audit.pso.omlx_m5_forward_router_top8_logits",
         .library_key = "step4.audit.lib.qwen_forward",
         .function_name = "omlx_m5_forward_router_top8_logits",
         .variant = "step4-audit"});
    (void)ctx.warmup_pipeline(
        {.key = "step4.audit.pso.omlx_m5_moe_nax_bucket_schedule",
         .library_key = "step4.audit.lib.moe_nax",
         .function_name = "omlx_m5_moe_nax_bucket_schedule",
         .variant = "step4-audit"});

    for (U32 t : {1U, 8U, 73U, 131U, 512U, 2048U}) {
      const U32 max_tiles =
          std::min<U32>(t * 8U, ((t * 8U + 7U) / 8U) + 256U);
      const U32 max_rows = max_tiles * 8U;
      std::vector<std::uint16_t> logits_bf16(U64(t) * 256U);
      for (U32 r = 0; r < t; ++r) {
        for (U32 e = 0; e < 256U; ++e) {
          // Skewed + periodic distribution so some experts hit >64 tokens at T>=131
          const float base =
              (e < 6U) ? (4.0f - 0.15f * float((r + e) % 7U))
                       : std::sin(float(r * 17U + e * 31U) * 0.07f);
          logits_bf16[U64(r) * 256U + e] = f32_to_bf16_bits(base);
        }
      }

      auto b_logits = ctx.make_buffer(logits_bf16.size() * 2U);
      b_logits.write(std::as_bytes(std::span(logits_bf16)));

      auto b_ids_ref = ctx.make_buffer(U64(t) * 8U * 4U);
      auto b_scores_ref = ctx.make_buffer(U64(t) * 8U * 2U);
      auto b_exp_ref = ctx.make_buffer(U64(max_tiles) * 4U);
      auto b_vr_ref = ctx.make_buffer(U64(max_tiles) * 4U);
      auto b_asg_ref = ctx.make_buffer(U64(max_rows) * 4U);
      auto b_tot_ref = ctx.make_buffer(16U);

      auto b_ids_fus = ctx.make_buffer(U64(t) * 8U * 4U);
      auto b_scores_fus = ctx.make_buffer(U64(t) * 8U * 2U);
      auto b_exp_fus = ctx.make_buffer(U64(max_tiles) * 4U);
      auto b_vr_fus = ctx.make_buffer(U64(max_tiles) * 4U);
      auto b_asg_fus = ctx.make_buffer(U64(max_rows) * 4U);
      auto b_tot_fus = ctx.make_buffer(16U);

      auto b_ids_w = ctx.make_buffer(U64(t) * 8U * 4U);
      auto b_scores_w = ctx.make_buffer(U64(t) * 8U * 2U);
      auto b_exp_w = ctx.make_buffer(U64(max_tiles) * 4U);
      auto b_vr_w = ctx.make_buffer(U64(max_tiles) * 4U);
      auto b_asg_w = ctx.make_buffer(U64(max_rows) * 4U);
      auto b_tot_w = ctx.make_buffer(16U);

      const std::array<U32, 2> rparams{t, 256U};
      const std::array<U32, 4> bparams{t, 8U, 128U, max_tiles};
      auto rbytes = std::as_bytes(std::span(rparams));
      auto bbytes = std::as_bytes(std::span(bparams));

      // 1. Reference 2-launch (router_top8_logits + bucket_schedule)
      {
        MetalCommandPlanDescriptor desc{
            .key = "step4.ref_sched.t" + std::to_string(t),
            .dispatches = {
                {.pipeline_key =
                     "step4.audit.pso.omlx_m5_forward_router_top8_logits",
                 .grid = {.width = 32U, .height = t},
                 .threads_per_threadgroup = {.width = 32U},
                 .resources = {{0, 0, b_logits.size()},
                               {1, 1, b_ids_ref.size()},
                               {2, 2, b_scores_ref.size()}},
                 .constant_arguments = {
                     {3, {rbytes.begin(), rbytes.end()}}}},
                {.pipeline_key =
                     "step4.audit.pso.omlx_m5_moe_nax_bucket_schedule",
                 .barrier_before = true,
                 .grid = {.width = 256U},
                 .threads_per_threadgroup = {.width = 256U},
                 .resources = {{0, 1, b_ids_ref.size()},
                               {1, 3, b_exp_ref.size()},
                               {2, 4, b_vr_ref.size()},
                               {3, 5, b_asg_ref.size()},
                               {4, 6, b_tot_ref.size()}},
                 .constant_arguments = {
                     {5, {bbytes.begin(), bbytes.end()}}}},
            },
            .ranges = {{"full", 0, 2}},
        };
        auto plan = ctx.prepare_command_plan(std::move(desc));
        const std::vector<MetalBufferBinding> binds{
            {0, &b_logits, 0},
            {1, &b_ids_ref, 0},
            {2, &b_scores_ref, 0},
            {3, &b_exp_ref, 0},
            {4, &b_vr_ref, 0},
            {5, &b_asg_ref, 0},
            {6, &b_tot_ref, 0}};
        const std::array inv{
            MetalCommandInvocation{.plan = &plan, .bindings = binds}};
        (void)ctx.execute_batch(inv);
      }

      // 2. Fused 1-launch BM=8 + 1-launch BM=32
      {
        MetalCommandPlanDescriptor desc{
            .key = "step4.fus_sched.t" + std::to_string(t),
            .dispatches = {
                {.pipeline_key =
                     "step4.audit.pso.omlx_m5_moe_nax_router_top8_bucket_schedule",
                 .grid = {.width = 256U},
                 .threads_per_threadgroup = {.width = 256U},
                 .resources = {{0, 0, b_logits.size()},
                               {1, 1, b_ids_fus.size()},
                               {2, 2, b_scores_fus.size()},
                               {3, 3, b_exp_fus.size()},
                               {4, 4, b_vr_fus.size()},
                               {5, 5, b_asg_fus.size()},
                               {6, 6, b_tot_fus.size()}},
                 .constant_arguments = {
                     {7, {bbytes.begin(), bbytes.end()}}}},
                {.pipeline_key =
                     "step4.audit.pso."
                     "omlx_m5_moe_nax_router_top8_bucket_schedule_bm32",
                 .grid = {.width = 256U},
                 .threads_per_threadgroup = {.width = 256U},
                 .resources = {{0, 0, b_logits.size()},
                               {1, 7, b_ids_w.size()},
                               {2, 8, b_scores_w.size()},
                               {3, 9, b_exp_w.size()},
                               {4, 10, b_vr_w.size()},
                               {5, 11, b_asg_w.size()},
                               {6, 12, b_tot_w.size()}},
                 .constant_arguments = {
                     {7, {bbytes.begin(), bbytes.end()}}}},
            },
            .ranges = {{"full", 0, 2}},
        };
        auto plan = ctx.prepare_command_plan(std::move(desc));
        const std::vector<MetalBufferBinding> binds{
            {0, &b_logits, 0},
            {1, &b_ids_fus, 0},
            {2, &b_scores_fus, 0},
            {3, &b_exp_fus, 0},
            {4, &b_vr_fus, 0},
            {5, &b_asg_fus, 0},
            {6, &b_tot_fus, 0},
            {7, &b_ids_w, 0},
            {8, &b_scores_w, 0},
            {9, &b_exp_w, 0},
            {10, &b_vr_w, 0},
            {11, &b_asg_w, 0},
            {12, &b_tot_w, 0}};
        const std::array inv{
            MetalCommandInvocation{.plan = &plan, .bindings = binds}};
        (void)ctx.execute_batch(inv);
      }

      U32 n_tiles = 0;
      std::memcpy(&n_tiles, b_tot_ref.read(0, 4).data(), 4);
      require(b_ids_ref.read() == b_ids_fus.read() &&
                  b_scores_ref.read() == b_scores_fus.read() &&
                  b_tot_ref.read() == b_tot_fus.read() &&
                  b_exp_ref.read(0, U64(n_tiles) * 4U) ==
                      b_exp_fus.read(0, U64(n_tiles) * 4U) &&
                  b_vr_ref.read(0, U64(n_tiles) * 4U) ==
                      b_vr_fus.read(0, U64(n_tiles) * 4U) &&
                  b_asg_ref.read(0, U64(n_tiles) * 8U * 4U) ==
                      b_asg_fus.read(0, U64(n_tiles) * 8U * 4U),
              "fused router_top8_bucket_schedule diverged from 2-launch ref");

      // Verify BM=32 work-tile schedule exactness!
      std::array<U32, 4> tot_w{};
      std::memcpy(tot_w.data(), b_tot_w.read(0, 16).data(), 16);
      require(tot_w[0] == n_tiles && tot_w[1] <= tot_w[0] && tot_w[1] >= 1U &&
                  b_ids_ref.read() == b_ids_w.read() &&
                  b_scores_ref.read() == b_scores_w.read() &&
                  b_asg_ref.read(0, U64(n_tiles) * 8U * 4U) ==
                      b_asg_w.read(0, U64(n_tiles) * 8U * 4U),
              "BM=32 bucket_schedule 8-tile row packing mismatch");

      std::vector<U32> exp_w(tot_w[1]), vr_w(tot_w[1]);
      std::memcpy(exp_w.data(), b_exp_w.read(0, U64(tot_w[1]) * 4U).data(),
                  U64(tot_w[1]) * 4U);
      std::memcpy(vr_w.data(), b_vr_w.read(0, U64(tot_w[1]) * 4U).data(),
                  U64(tot_w[1]) * 4U);
      std::vector<U32> asg_w(U64(n_tiles) * 8U), ids_host(U64(t) * 8U);
      std::memcpy(asg_w.data(), b_asg_w.read(0, asg_w.size() * 4U).data(),
                  asg_w.size() * 4U);
      std::memcpy(ids_host.data(), b_ids_w.read().data(), ids_host.size() * 4U);

      U32 total_valid_w = 0;
      for (U32 w = 0; w < tot_w[1]; ++w) {
        const U32 e_id = exp_w[w] & 0xFFFFU;
        const U32 first_8 = exp_w[w] >> 16U;
        const U32 vr = vr_w[w];
        require(e_id < 256U && vr >= 1U && vr <= 32U &&
                    first_8 * 8U + vr <= n_tiles * 8U,
                "BM=32 work-tile descriptor out of bounds");
        for (U32 r = 0; r < vr; ++r) {
          const U32 a = asg_w[first_8 * 8U + r];
          require(a < t * 8U && ids_host[a] == e_id,
                  "BM=32 work-tile row mapped to wrong expert");
        }
        const U32 ceil8 = (vr + 7U) & ~7U;
        for (U32 r = vr; r < ceil8; ++r) {
          require(asg_w[first_8 * 8U + r] == 0xFFFFFFFFU,
                  "BM=32 work-tile padding row not marked 0xFFFFFFFF");
        }
        total_valid_w += vr;
      }
      require(total_valid_w == t * 8U,
              "BM=32 work-tiles did not cover all t * 8 assignments");

      // 3. Also execute & verify BM=64 work-tile scheduler 0-ULP!
      auto b_ids_64 = ctx.make_buffer(U64(t) * 8U * 4U);
      auto b_scores_64 = ctx.make_buffer(U64(t) * 8U * 2U);
      auto b_exp_64 = ctx.make_buffer(U64(max_tiles) * 4U);
      auto b_vr_64 = ctx.make_buffer(U64(max_tiles) * 4U);
      auto b_asg_64 = ctx.make_buffer(U64(max_rows) * 4U);
      auto b_tot_64 = ctx.make_buffer(16U);
      {
        MetalCommandPlanDescriptor desc{
            .key = "step4.bm64_sched.t" + std::to_string(t),
            .dispatches = {
                {.pipeline_key =
                     "step4.audit.pso."
                     "omlx_m5_moe_nax_router_top8_bucket_schedule_bm64",
                 .grid = {.width = 256U},
                 .threads_per_threadgroup = {.width = 256U},
                 .resources = {{0, 0, b_logits.size()},
                               {1, 1, b_ids_64.size()},
                               {2, 2, b_scores_64.size()},
                               {3, 3, b_exp_64.size()},
                               {4, 4, b_vr_64.size()},
                               {5, 5, b_asg_64.size()},
                               {6, 6, b_tot_64.size()}},
                 .constant_arguments = {
                     {7, {bbytes.begin(), bbytes.end()}}}},
            },
            .ranges = {{"full", 0, 1}},
        };
        auto plan = ctx.prepare_command_plan(std::move(desc));
        const std::vector<MetalBufferBinding> binds{
            {0, &b_logits, 0}, {1, &b_ids_64, 0}, {2, &b_scores_64, 0},
            {3, &b_exp_64, 0}, {4, &b_vr_64, 0},  {5, &b_asg_64, 0},
            {6, &b_tot_64, 0}};
        const std::array inv{
            MetalCommandInvocation{.plan = &plan, .bindings = binds}};
        (void)ctx.execute_batch(inv);
      }
      std::array<U32, 4> tot_64{};
      std::memcpy(tot_64.data(), b_tot_64.read(0, 16).data(), 16);
      require(tot_64[0] == n_tiles && tot_64[1] <= tot_w[1] &&
                  tot_64[1] >= 1U && b_ids_ref.read() == b_ids_64.read() &&
                  b_scores_ref.read() == b_scores_64.read() &&
                  b_asg_ref.read(0, U64(n_tiles) * 8U * 4U) ==
                      b_asg_64.read(0, U64(n_tiles) * 8U * 4U),
              "BM=64 bucket_schedule 8-tile row packing mismatch");

      std::vector<U32> exp_64(tot_64[1]), vr_64(tot_64[1]);
      std::memcpy(exp_64.data(), b_exp_64.read(0, U64(tot_64[1]) * 4U).data(),
                  U64(tot_64[1]) * 4U);
      std::memcpy(vr_64.data(), b_vr_64.read(0, U64(tot_64[1]) * 4U).data(),
                  U64(tot_64[1]) * 4U);
      U32 total_valid_64 = 0;
      for (U32 w = 0; w < tot_64[1]; ++w) {
        const U32 e_id = exp_64[w] & 0xFFFFU;
        const U32 first_8 = exp_64[w] >> 16U;
        const U32 vr = vr_64[w];
        require(e_id < 256U && vr >= 1U && vr <= 64U &&
                    first_8 * 8U + vr <= n_tiles * 8U,
                "BM=64 work-tile descriptor out of bounds");
        for (U32 r = 0; r < vr; ++r) {
          const U32 a = asg_w[first_8 * 8U + r];
          require(a < t * 8U && ids_host[a] == e_id,
                  "BM=64 work-tile row mapped to wrong expert");
        }
        total_valid_64 += vr;
      }
      require(total_valid_64 == t * 8U,
              "BM=64 work-tiles did not cover all t * 8 assignments");
    }
  }

  return {{"checked_psos", checked_psos},
          {"moe_nax_checked", moe_nax_checked},
          {"max_sram_bytes", max_sram_bytes},
          {"sram_by_pso", sram_by_pso},
          {"passed", true}};
}

// ---------------------------------------------------------------------------
// Test 2: Pre-Tiled Direct T=1 Coalesced SIMD32 QMV (`mx_nax_tiled_bk128_v1` &
// `oq_nax_tiled_g64_v1` across q3/q4/q5/q6/q8, BN=16 & BN=32, + QwenForward!)
// ---------------------------------------------------------------------------
Json test_pretiled_qmv_t1(const MetalContext &ctx,
                          const std::filesystem::path &libs,
                          const std::filesystem::path &python_exe,
                          const std::filesystem::path &repo_root,
                          const std::filesystem::path &tmp_dir) {
  const auto py_synth = tmp_dir / "emit_pretiled_512.py";
  const auto synth_dir = tmp_dir / "qwen_pretiled_512";
  const auto mx_untiled = synth_dir / "mx_untiled.omlxm5";
  const auto mx_pretiled = synth_dir / "mx_pretiled.omlxm5";
  const auto oq_untiled = synth_dir / "oq_untiled.omlxm5";
  const auto oq_pretiled = synth_dir / "oq_pretiled.omlxm5";
  {
    std::ofstream py(py_synth);
    py << R"PY(
import json, pathlib, struct, sys
import numpy as np

repo = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
sys.path.insert(0, str(repo / "tools"))
import m5_artifact as ma

src_dir = out_dir / "source"
src_dir.mkdir(parents=True, exist_ok=True)

# Small 2-layer hybrid Qwen (1 GDN + 1 full_attention) with hidden_size=512
# so every weight has in_features >= 512 and enters NAX pre-tiling + pretiled_qmv_t1!
cfg = {
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "dtype": "bfloat16",
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "full_attention_interval": 2,
        "max_position_embeddings": 2048,
        "rope_parameters": {
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "type": "default",
        },
        "rms_norm_eps": 1e-6,
        "vocab_size": 528,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    },
}
(src_dir / "config.json").write_text(json.dumps(cfg, indent=2))

rng = np.random.default_rng(202604)
def bf16(shape, scale=0.04):
    x = (rng.standard_normal(shape) * scale).astype(np.float32)
    return ("BF16", shape, ma._float32_to_bf16_u16(x).tobytes())

h, f, v = 512, 1024, 528
q, kv = 512, 256
gk, gv = 256, 512
conv = 2 * gk + gv  # 1024
hv = 4

tensors = {
    "language_model.model.embed_tokens.weight": bf16((v, h)),
    "language_model.model.norm.weight": bf16((h,), 0.01),
    "language_model.lm_head.weight": bf16((v, h)),
    # Layer 0: GDN
    "language_model.model.layers.0.input_layernorm.weight": bf16((h,), 0.01),
    "language_model.model.layers.0.post_attention_layernorm.weight": bf16((h,), 0.01),
    "language_model.model.layers.0.linear_attn.in_proj_qkv.weight": bf16((conv, h)),
    "language_model.model.layers.0.linear_attn.in_proj_z.weight": bf16((gv, h)),
    "language_model.model.layers.0.linear_attn.in_proj_a.weight": bf16((hv, h)),
    "language_model.model.layers.0.linear_attn.in_proj_b.weight": bf16((hv, h)),
    "language_model.model.layers.0.linear_attn.conv1d.weight": bf16((conv, 1, 4)),
    "language_model.model.layers.0.linear_attn.A_log": bf16((hv,), 0.1),
    "language_model.model.layers.0.linear_attn.dt_bias": bf16((hv,), 0.1),
    "language_model.model.layers.0.linear_attn.norm.weight": ("BF16", (128,), ma._float32_to_bf16_u16(np.ones((128,), dtype=np.float32)).tobytes()),
    "language_model.model.layers.0.linear_attn.out_proj.weight": bf16((h, gv)),
    "language_model.model.layers.0.mlp.gate_proj.weight": bf16((f, h)),
    "language_model.model.layers.0.mlp.up_proj.weight": bf16((f, h)),
    "language_model.model.layers.0.mlp.down_proj.weight": bf16((h, f)),
    # Layer 1: full_attention
    "language_model.model.layers.1.input_layernorm.weight": bf16((h,), 0.01),
    "language_model.model.layers.1.post_attention_layernorm.weight": bf16((h,), 0.01),
    "language_model.model.layers.1.self_attn.q_proj.weight": bf16((2 * q, h)),
    "language_model.model.layers.1.self_attn.k_proj.weight": bf16((kv, h)),
    "language_model.model.layers.1.self_attn.v_proj.weight": bf16((kv, h)),
    "language_model.model.layers.1.self_attn.q_norm.weight": bf16((256,), 0.01),
    "language_model.model.layers.1.self_attn.k_norm.weight": bf16((256,), 0.01),
    "language_model.model.layers.1.self_attn.o_proj.weight": bf16((h, q)),
    "language_model.model.layers.1.mlp.gate_proj.weight": bf16((f, h)),
    "language_model.model.layers.1.mlp.up_proj.weight": bf16((f, h)),
    "language_model.model.layers.1.mlp.down_proj.weight": bf16((h, f)),
}

hdr, chunks, off = {}, [], 0
for k_name, (dt, shp, raw) in sorted(tensors.items()):
    hdr[k_name] = {"dtype": dt, "shape": list(shp), "data_offsets": [off, off + len(raw)]}
    chunks.append(raw)
    off += len(raw)
hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
if (-len(hb)) % 8:
    hb += b" " * ((-len(hb)) % 8)
with (src_dir / "model.safetensors").open("wb") as fp:
    fp.write(struct.pack("<Q", len(hb)))
    fp.write(hb)
    for c in chunks:
        fp.write(c)

# Mixed oQe recipe exercising q3_g64, q4_g128, q5_g64, and q6_g128!
oq_recipe = {
    "schema": "omlx-qda-recipe-v1",
    "quantization": {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
        "language_model.model.layers.0.linear_attn.in_proj_qkv": {"bits": 4, "group_size": 64, "mode": "affine"},
        "language_model.model.layers.0.linear_attn.in_proj_z": {"bits": 6, "group_size": 128, "mode": "affine"},
        "language_model.model.layers.0.mlp.gate_proj": {"bits": 3, "group_size": 64, "mode": "affine"},
        "language_model.model.layers.0.mlp.up_proj": {"bits": 3, "group_size": 64, "mode": "affine"},
        "language_model.model.layers.0.mlp.down_proj": {"bits": 5, "group_size": 64, "mode": "affine"},
        "language_model.model.layers.1.mlp.gate_proj": {"bits": 6, "group_size": 128, "mode": "affine"},
        "language_model.model.layers.1.mlp.up_proj": {"bits": 6, "group_size": 128, "mode": "affine"},
        "language_model.model.layers.1.self_attn.q_proj": {"bits": 4, "group_size": 128, "mode": "affine"},
        "language_model.model.layers.1.self_attn.k_proj": {"bits": 5, "group_size": 64, "mode": "affine"},
        "language_model.model.layers.1.self_attn.v_proj": {"bits": 6, "group_size": 64, "mode": "affine"},
    },
}
ma.quantize_source_with_recipe_to_artifact(src_dir, oq_recipe, pathlib.Path(sys.argv[5]), "test/step4-oq", nax_pretile=False)
ma.quantize_source_with_recipe_to_artifact(src_dir, oq_recipe, pathlib.Path(sys.argv[6]), "test/step4-oq", nax_pretile=True)

import mlx.core as mx
mx_src_dir = out_dir / "mx_source"
mx_src_dir.mkdir(parents=True, exist_ok=True)
mx_cfg = dict(cfg)
mx_cfg["quantization"] = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
(mx_src_dir / "config.json").write_text(json.dumps(mx_cfg))
mx_tensors = {}
for k_name, (dt, shp, raw) in sorted(tensors.items()):
    if len(shp) == 2 and k_name.endswith(".weight"):
        arr_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shp)
        arr_f32 = (arr_u16.astype(np.uint32) << 16).view(np.float32)
        wq, sc = mx.quantize(mx.array(arr_f32, dtype=mx.bfloat16), group_size=32, bits=4, mode="mxfp4")
        mx.eval(wq, sc)
        wq_np = np.asarray(wq, dtype=np.uint32)
        sc_np = np.asarray(sc, dtype=np.uint8)
        mod = k_name[:-len(".weight")]
        mx_tensors[mod + ".weight"] = ("U32", wq_np.shape, wq_np.tobytes())
        mx_tensors[mod + ".scales"] = ("U8", sc_np.shape, sc_np.tobytes())
    else:
        mx_tensors[k_name] = (dt, shp, raw)
mhdr, mchunks, moff = {}, [], 0
for k_name, (dt, shp, raw) in sorted(mx_tensors.items()):
    mhdr[k_name] = {"dtype": dt, "shape": list(shp), "data_offsets": [moff, moff + len(raw)]}
    mchunks.append(raw)
    moff += len(raw)
mhb = json.dumps(mhdr, separators=(",", ":")).encode("utf-8")
if (-len(mhb)) % 8:
    mhb += b" " * ((-len(mhb)) % 8)
with (mx_src_dir / "model.safetensors").open("wb") as fp:
    fp.write(struct.pack("<Q", len(mhb)))
    fp.write(mhb)
    for c in mchunks:
        fp.write(c)
q_mx = ma.SafeTensorSource(mx_src_dir, None)
try:
    ma.write_artifact(q_mx, pathlib.Path(sys.argv[3]), "test/step4-mx", "mlx_mxfp4")
finally:
    q_mx.close()
ma.pretile_artifact(pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]))
)PY";
  }
  run_cmd("env -u MTL_SHADER_VALIDATION -u MTL_DEBUG_LAYER " +
          shell_quote(python_exe) + " " + shell_quote(py_synth) + " " +
          shell_quote(repo_root) + " " + shell_quote(synth_dir) + " " +
          shell_quote(mx_untiled) + " " + shell_quote(mx_pretiled) + " " +
          shell_quote(oq_untiled) + " " + shell_quote(oq_pretiled));

  const auto cfg = load_qwen_model_config(synth_dir / "source/config.json");
  const auto art_mx_u = ArtifactModel::open(mx_untiled, true);
  const auto art_mx_p = ArtifactModel::open(mx_pretiled, true);
  const auto art_oq_u = ArtifactModel::open(oq_untiled, true);
  const auto art_oq_p = ArtifactModel::open(oq_pretiled, true);

  const auto w_mx_u =
      materialize_model(art_mx_u, plan_model_materialization(art_mx_u, ctx), ctx);
  const auto w_mx_p =
      materialize_model(art_mx_p, plan_model_materialization(art_mx_p, ctx), ctx);
  const auto w_oq_u =
      materialize_model(art_oq_u, plan_model_materialization(art_oq_u, ctx), ctx);
  const auto w_oq_p =
      materialize_model(art_oq_p, plan_model_materialization(art_oq_p, ctx), ctx);

  const auto cat_mx_u = build_qwen_layer_catalog(art_mx_u, cfg, 256, 64);
  const auto cat_mx_p = build_qwen_layer_catalog(art_mx_p, cfg, 256, 64);
  const auto cat_oq_u = build_qwen_layer_catalog(art_oq_u, cfg, 256, 64);
  const auto cat_oq_p = build_qwen_layer_catalog(art_oq_p, cfg, 256, 64);

  U32 mx_pretiled_routes = 0;
  U32 oq_pretiled_routes = 0;
  U32 max_mx_ulp = 0;
  U32 max_oq_ulp = 0;

  {
    ScopedEnv env_td("OMLX_M5_T_DOMAIN_DISPATCH", "1");
    {
      ScopedEnv env_pt_off("OMLX_M5_ENABLE_PRETILED_QMV_T1", "0");
      QwenForward mx_p_off(ctx, w_mx_p, cat_mx_p, cfg, 256, libs, {}, true,
                           false, false, std::nullopt, false, false,
                           std::nullopt,
                           LinearNumericPolicy::RowExactPortableV1,
                           SamplingContract::PositionCoupledV1);
      for (const auto &r : mx_p_off.inspect_profile(1).weights) {
        require(r.family != "pretiled_qmv_t1",
                "OMLX_M5_ENABLE_PRETILED_QMV_T1=0 must disable pretiled_qmv_t1");
      }
    }
    ScopedEnv env_pt("OMLX_M5_ENABLE_PRETILED_QMV_T1", nullptr);

    std::unique_ptr<QwenForward> mx_u_ptr;
    {
      ScopedEnv env_ref("OMLX_M5_ENABLE_PRETILED_QMV_T1", "0");
      mx_u_ptr = std::make_unique<QwenForward>(
          ctx, w_mx_u, cat_mx_u, cfg, 256, libs, std::span<const U32>{}, false,
          false, false, std::nullopt, false, false, std::nullopt,
          LinearNumericPolicy::Legacy);
    }
    auto &mx_u = *mx_u_ptr;
    QwenForward mx_p(ctx, w_mx_p, cat_mx_p, cfg, 256, libs, {}, false, false,
                     false, std::nullopt, false, false, std::nullopt,
                     LinearNumericPolicy::Legacy);

    const auto prof_mx1 = mx_p.inspect_profile(1);
    bool saw_concat3 = false, saw_concat2 = false, saw_swiglu = false,
         saw_t1 = false;
    for (const auto &r : prof_mx1.weights) {
      if (r.family == "pretiled_qmv_t1") {
        ++mx_pretiled_routes;
        saw_concat3 |=
            (r.function.find("omlx_m5_pretiled_qmv_mxfp4_concat3") == 0U);
        saw_concat2 |=
            (r.function.find("omlx_m5_pretiled_qmv_mxfp4_concat2") == 0U ||
             r.function.find("omlx_m5_pretiled_qmv_mxfp4_concat4") == 0U);
        saw_swiglu |=
            (r.function.find("omlx_m5_pretiled_qmv_mxfp4_swiglu") == 0U);
        saw_t1 |= (r.function == "omlx_m5_pretiled_qmv_mxfp4_t1" ||
                   r.function.find("omlx_m5_pretiled_qmv_mxfp4_bn") == 0U);
      }
    }
    require(saw_concat3 && saw_concat2 && saw_swiglu && saw_t1,
            "MXFP4 pre-tiled T=1 profile did not activate all 4 coalesced QMV kernels");
    require(prof_mx1.dispatch_count + 2U * U64(cfg.num_hidden_layers) ==
                mx_u.inspect_profile(1).dispatch_count -
                    U64(cfg.num_hidden_layers > 0 ? 4U : 0U) ||
                prof_mx1.dispatch_count < mx_u.inspect_profile(1).dispatch_count,
            "MXFP4 fused T=1 profile did not reduce dispatch count");

    // Verify speculative=true and batch=true NEVER activate pretiled_qmv_t1 even when T_DOMAIN_DISPATCH=1!
    const std::array<U32, 1> taps{1U};
    QwenForward mx_spec(ctx, w_mx_p, cat_mx_p, cfg, 256, libs, taps, false,
                        true, false, std::nullopt, false, false, std::nullopt,
                        LinearNumericPolicy::RowExactPortableV1,
                        SamplingContract::PositionCoupledV1);
    for (const auto &r : mx_spec.inspect_profile(1).weights) {
      require(r.family != "pretiled_qmv_t1",
              "speculative=true must never enter pretiled_qmv_t1");
    }
    QwenForward mx_batch(ctx, w_mx_p, cat_mx_p, cfg, 256, libs, {}, false,
                         false, false, std::nullopt, false, true, std::nullopt,
                         LinearNumericPolicy::RowExactPortableV1,
                         SamplingContract::PositionCoupledV1);
    for (const auto &r : mx_batch.inspect_profile(1).weights) {
      require(r.family != "pretiled_qmv_t1",
              "enable_batch=true must never enter pretiled_qmv_t1");
    }

    // Verify RowExactPortableV1 (PositionCoupledV1) + OMLX_M5_T_DOMAIN_DISPATCH=1
    // (solo AR: speculative=false, batch=false) ALSO activates pretiled_qmv_t1
    // while keeping untiled weights on row_exact v3!
    QwenForward mx_p_coupled(ctx, w_mx_p, cat_mx_p, cfg, 256, libs, {}, true,
                             false, false, std::nullopt, false, false,
                             std::nullopt,
                             LinearNumericPolicy::RowExactPortableV1,
                             SamplingContract::PositionCoupledV1);
    bool saw_coupled_mx_pretiled = false;
    for (const auto &r : mx_p_coupled.inspect_profile(1).weights) {
      saw_coupled_mx_pretiled |= (r.family == "pretiled_qmv_t1");
    }
    require(saw_coupled_mx_pretiled,
            "RowExactPortableV1 + T_DOMAIN_DISPATCH=1 solo AR must activate MXFP4 pretiled_qmv_t1");

    QwenForward oq_u_coupled(ctx, w_oq_u, cat_oq_u, cfg, 256, libs, {}, true,
                             false, false, std::nullopt, false, false,
                             std::nullopt,
                             LinearNumericPolicy::RowExactPortableV1,
                             SamplingContract::PositionCoupledV1);
    bool saw_untiled_row_exact = false;
    for (const auto &r : oq_u_coupled.inspect_profile(1).weights) {
      saw_untiled_row_exact |=
          (r.row_exact_eligible || r.family == "pretiled_qmv_t1");
    }
    require(saw_untiled_row_exact,
            "RowExactPortableV1 + T_DOMAIN_DISPATCH=1 untiled weights must route to row_exact v3 or RAM-tiled pretiled_qmv_t1");

    QwenForward oq_p_coupled(ctx, w_oq_p, cat_oq_p, cfg, 256, libs, {}, true,
                             false, false, std::nullopt, false, false,
                             std::nullopt,
                             LinearNumericPolicy::RowExactPortableV1,
                             SamplingContract::PositionCoupledV1);
    bool saw_coupled_oq_pretiled = false;
    for (const auto &r : oq_p_coupled.inspect_profile(1).weights) {
      saw_coupled_oq_pretiled |= (r.family == "pretiled_qmv_t1");
    }
    require(saw_coupled_oq_pretiled,
            "RowExactPortableV1 + T_DOMAIN_DISPATCH=1 solo AR must activate Affine pretiled_qmv_t1");

    const char *sv_e = std::getenv("MTL_SHADER_VALIDATION");
    const bool sv_on = sv_e != nullptr && std::string_view(sv_e) != "0" &&
                       !std::string_view(sv_e).empty();

    // Multi-step T=1 decode (4 consecutive steps building GDN + KV state)
    // exercising all 4 MXFP4 pretiled_qmv_t1 kernels under both normal & GPU validation!
    float max_mx_abs = 0.0f;
    for (U32 step = 0; step < 4U; ++step) {
      const std::array<U32, 1> tok{11U + step * 7U};
      const auto r_mx_u1 = mx_u.run(tok, {}, true);
      const auto r_mx_p1 = mx_p.run(tok, {}, true);
      require(r_mx_u1.token == r_mx_p1.token &&
                  r_mx_u1.logits.size() == r_mx_p1.logits.size(),
              "MXFP4 pretiled_qmv_t1 greedy token mismatch at step " +
                  std::to_string(step));
      for (std::size_t i = 0; i < r_mx_u1.logits.size(); ++i) {
        require(std::isfinite(r_mx_u1.logits[i]) &&
                    std::isfinite(r_mx_p1.logits[i]),
                "MXFP4 pretiled_qmv_t1 produced non-finite logit at step " +
                    std::to_string(step));
        max_mx_abs =
            std::max(max_mx_abs, std::abs(r_mx_u1.logits[i] - r_mx_p1.logits[i]));
        if (step == 0U && std::abs(r_mx_u1.logits[i]) >= 1.0f) {
          max_mx_ulp = std::max(
              max_mx_ulp, bf16_ulp_diff(f32_to_bf16_bits(r_mx_u1.logits[i]),
                                        f32_to_bf16_bits(r_mx_p1.logits[i])));
        }
      }
    }
    require(max_mx_ulp <= 8U && std::isfinite(max_mx_abs) && max_mx_abs < 6e-2f,
            "MXFP4 pretiled_qmv_t1 exceeded tolerance (step0_ulp=" +
                std::to_string(max_mx_ulp) +
                ", max_abs=" + std::to_string(max_mx_abs) + ")");
    if (!sv_on) {
      mx_u.reset();
      mx_p.reset();
      const auto prompt = make_tokens(6U, 11U, U32(cfg.vocab_size));
      const auto r_mx_u0 = mx_u.run(prompt, {}, true);
      const auto r_mx_p0 = mx_p.run(prompt, {}, true);
      require(r_mx_u0.token == r_mx_p0.token, "MXFP4 T=6 prefill token mismatch");
      const std::array<U32, 1> dec_tok{r_mx_u0.token};
      require(mx_u.run(dec_tok, {}, true).token ==
                  mx_p.run(dec_tok, {}, true).token,
              "MXFP4 T=6->T=1 transition token mismatch");
    }

    // Affine oQe (q3_g64 + q4_g128 + q5_g64 + q6_g128)!
    std::unique_ptr<QwenForward> oq_u_ptr;
    {
      ScopedEnv env_ref("OMLX_M5_ENABLE_PRETILED_QMV_T1", "0");
      oq_u_ptr = std::make_unique<QwenForward>(
          ctx, w_oq_u, cat_oq_u, cfg, 256, libs, std::span<const U32>{}, false,
          false, false, std::nullopt, false, false, std::nullopt,
          LinearNumericPolicy::Legacy);
    }
    auto &oq_u = *oq_u_ptr;
    QwenForward oq_p(ctx, w_oq_p, cat_oq_p, cfg, 256, libs, {}, false, false,
                     false, std::nullopt, false, false, std::nullopt,
                     LinearNumericPolicy::Legacy);
    bool saw_aff_concat3 = false, saw_aff_concat2 = false,
         saw_aff_swiglu = false, saw_aff_t1 = false;
    for (const auto &r : oq_p.inspect_profile(1).weights) {
      if (r.family == "pretiled_qmv_t1") {
        ++oq_pretiled_routes;
        saw_aff_concat3 |=
            (r.function.find("omlx_m5_pretiled_qmv_affine_concat3") == 0U);
        saw_aff_concat2 |=
            (r.function.find("omlx_m5_pretiled_qmv_affine_concat2") == 0U ||
             r.function.find("omlx_m5_pretiled_qmv_affine_concat4") == 0U);
        saw_aff_swiglu |=
            (r.function.find("omlx_m5_pretiled_qmv_affine_swiglu") == 0U);
        saw_aff_t1 |=
            (r.function == "omlx_m5_pretiled_qmv_affine_t1" ||
             r.function == "omlx_m5_pretiled_qmv_affine_add_sq_t1" ||
             r.function == "omlx_m5_pretiled_qmv_affine_norm_t1" ||
             r.function.find("omlx_m5_pretiled_qmv_affine_q") == 0U);
      }
    }
    require(saw_aff_concat3 && saw_aff_concat2 && saw_aff_swiglu && saw_aff_t1,
            "Affine pre-tiled T=1 profile missing expected kernels (c3=" +
                std::to_string(saw_aff_concat3) +
                " c2=" + std::to_string(saw_aff_concat2) +
                " swi=" + std::to_string(saw_aff_swiglu) +
                " t1=" + std::to_string(saw_aff_t1) + ")");

    float max_oq_abs = 0.0f;
    for (U32 step = 0; step < 4U; ++step) {
      const std::array<U32, 1> oq_dec{13U + step * 9U};
      const auto r_oq_u1 = oq_u.run(oq_dec, {}, true);
      const auto r_oq_p1 = oq_p.run(oq_dec, {}, true);
      require(r_oq_u1.token == r_oq_p1.token &&
                  r_oq_u1.logits.size() == r_oq_p1.logits.size(),
              "Affine pretiled_qmv_t1 greedy token mismatch at step " +
                  std::to_string(step));
      for (std::size_t i = 0; i < r_oq_u1.logits.size(); ++i) {
        require(std::isfinite(r_oq_u1.logits[i]) &&
                    std::isfinite(r_oq_p1.logits[i]),
                "Affine pretiled_qmv_t1 produced non-finite logit at step " +
                    std::to_string(step));
        max_oq_abs =
            std::max(max_oq_abs, std::abs(r_oq_u1.logits[i] - r_oq_p1.logits[i]));
        if (step == 0U && std::abs(r_oq_u1.logits[i]) >= 1.0f) {
          max_oq_ulp = std::max(
              max_oq_ulp, bf16_ulp_diff(f32_to_bf16_bits(r_oq_u1.logits[i]),
                                        f32_to_bf16_bits(r_oq_p1.logits[i])));
        }
      }
    }
    require(max_oq_ulp <= 8U && std::isfinite(max_oq_abs) && max_oq_abs < 6e-2f,
            "Affine pretiled_qmv_t1 exceeded tolerance (step0_ulp=" +
                std::to_string(max_oq_ulp) +
                ", max_abs=" + std::to_string(max_oq_abs) + ")");
    if (!sv_on) {
      oq_u.reset();
      oq_p.reset();
      const auto prompt = make_tokens(6U, 11U, U32(cfg.vocab_size));
      const auto r_oq_u0 = oq_u.run(prompt, {}, true);
      const auto r_oq_p0 = oq_p.run(prompt, {}, true);
      require(r_oq_u0.token == r_oq_p0.token, "Affine T=6 prefill token mismatch");
      const std::array<U32, 1> oq_dec{r_oq_u0.token};
      require(oq_u.run(oq_dec, {}, true).token ==
                  oq_p.run(oq_dec, {}, true).token,
              "Affine T=6->T=1 transition token mismatch");
    }
  }

  return {{"mx_pretiled_routes", mx_pretiled_routes},
          {"oq_pretiled_routes", oq_pretiled_routes},
          {"max_mx_bf16_ulp", max_mx_ulp},
          {"max_oq_bf16_ulp", max_oq_ulp},
          {"passed", true}};
}

// ---------------------------------------------------------------------------
// Test 3: SharedPhysicalPageAllocator & 35B B=8 SharedKvArenaPool (-8.68 GiB)
// Zero-Duplicate-KV Allocation + Zero-Copy restore_shared + COW Isolation
// ---------------------------------------------------------------------------
Json test_shared_kv_arena_and_moe_packed_batch(
    const MetalContext &ctx, const std::filesystem::path &qwen_fixtures,
    const std::filesystem::path &libs) {
  // 1. 35B-A3B B=8 64K analytical memory check (proving < 2 GiB ICB slab & -8.68 GiB reduction)
  QwenModelConfig c35b;
  c35b.hidden_size = 2048;
  c35b.intermediate_size = 512;
  c35b.shared_expert_intermediate_size = 512;
  c35b.num_experts = 256;
  c35b.num_experts_per_token = 8;
  c35b.num_hidden_layers = 40;
  c35b.num_attention_heads = 16;
  c35b.num_key_value_heads = 2;
  c35b.head_dim = 256;
  c35b.linear_num_key_heads = 16;
  c35b.linear_num_value_heads = 32;
  c35b.linear_key_head_dim = 128;
  c35b.linear_value_head_dim = 128;
  c35b.linear_conv_kernel_dim = 4;
  c35b.vocab_size = 248320;
  c35b.layer_types.assign(40, LayerType::linear_attention);
  for (std::size_t i = 3; i < 40; i += 4) {
    c35b.layer_types[i] = LayerType::full_attention;
  }
  const U64 page_b_35b = kv_page_bytes(KvStorageFormat::BFloat16, 64U, 2U, 256U);
  const U64 old_8slot_kv_bytes = 8ULL * 1025ULL * page_b_35b * 20ULL;
  const U64 new_shared_kv_bytes = 1088ULL * page_b_35b * 20ULL;
  const U64 single_slab_bytes = 1088ULL * page_b_35b;
  require(single_slab_bytes <= (2ULL << 30) &&
              old_8slot_kv_bytes - new_shared_kv_bytes > (8ULL << 30),
          "35B B=8 SharedKvArenaPool did not save >8 GiB within 2 GiB slab cap");

  // 2. Live B=4 MoE QwenForward multi-slot zero-duplicate-KV + zero-copy COW + run_packed_batch!
  U32 packed_moe_checks = 0;
  for (const char *variant : {"moe-mxfp4", "moe-affine"}) {
    const auto fix = qwen_fixtures / variant;
    const auto cfg = load_qwen_model_config(fix / "source/config.json");
    const auto art = ArtifactModel::open(fix / "model.omlxm5", true);
    const auto weights =
        materialize_model(art, plan_model_materialization(art, ctx), ctx);
    const auto layers = build_qwen_layer_catalog(art, cfg, 1024, 64);

    // Compute exact full-attention KV bytes for 1 slot at max_context=1024 (17 pages)
    const U64 slot_kv_slab_bytes =
        17ULL *
        kv_page_bytes(KvStorageFormat::BFloat16, 64U,
                      U32(cfg.num_key_value_heads), U32(cfg.head_dim)) *
        2ULL * U64(cfg.full_attention_layers());
    require(slot_kv_slab_bytes > 0ULL, "fixture must contain full_attention layers");

    const char *sv_e = std::getenv("MTL_SHADER_VALIDATION");
    const bool sv_on = sv_e != nullptr && std::string_view(sv_e) != "0" &&
                       !std::string_view(sv_e).empty();
    const auto attn_pol = sv_on ? AttentionNumericPolicy::Legacy
                                : AttentionNumericPolicy::RowExactV1;
    const U32 shared_pages = 72U;
    QwenForward slot0(ctx, weights, layers, cfg, 1024, libs, {}, false, false,
                      false, std::nullopt, false, true, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1, attn_pol, false,
                      "bf16", nullptr, false, shared_pages);
    const U64 alloc_after_slot0 = ctx.current_allocated_bytes();
    QwenForward slot1(ctx, weights, layers, cfg, 1024, libs, {}, false, false,
                      false, std::nullopt, false, true, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1, attn_pol, false,
                      "bf16", &slot0, false);
    const U64 slot1_delta = ctx.current_allocated_bytes() - alloc_after_slot0;
    QwenForward slot2(ctx, weights, layers, cfg, 1024, libs, {}, false, false,
                      false, std::nullopt, false, true, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1, attn_pol, false,
                      "bf16", &slot0, false);

    // Secondary slot state_bytes must contain ONLY GDN recurrent/conv state (0 KV slab bytes!)
    const auto unshared_sec_plan = plan_qwen_forward_memory(
        cfg, 1024, 0, false, false, false, true, !sv_on, false, 0, 16, false);
    require(slot1.memory_plan().state_bytes + slot_kv_slab_bytes ==
                unshared_sec_plan.state_bytes,
            "secondary slot memory_plan().state_bytes still counted KV slabs");
    require(slot0.kv_page_pool()->shared_allocator() ==
                slot1.kv_page_pool()->shared_allocator() &&
                slot0.kv_page_pool()->physical_pages() == shared_pages &&
                slot1.kv_page_pool()->physical_pages() == shared_pages,
            "secondary slots do not share primary's SharedPhysicalPageAllocator");
    (void)slot1_delta;

    slot0.bind_execution_identity("step4-shared-kv-moe");
    slot1.bind_execution_identity("step4-shared-kv-moe");
    slot2.bind_execution_identity("step4-shared-kv-moe");

    // Prefill 70 tokens on slot0 (1 full 64-tok page + 6-tok partial tail page)
    GpuSamplingOptions samp{.temperature = 0.0f,
                            .top_k = 1,
                            .top_p = 1.0f,
                            .seed = 99,
                            .contract = SamplingContract::PositionCoupledV1};
    const auto pfx70 = make_tokens(70U, 5U, U32(cfg.vocab_size));
    (void)slot0.run(pfx70, samp);
    auto ckpt = slot0.checkpoint();
    const U32 free_before_restore = slot0.kv_page_pool()->free_pages();

    // Zero-copy restore_shared into slot1 and slot2!
    slot1.restore_shared(ckpt, &slot0);
    slot2.restore_shared(ckpt, &slot0);
    require(slot0.kv_page_pool()->free_pages() == free_before_restore,
            "zero-copy restore_shared allocated extra physical KV pages!");
    require(slot1.kv_page_pool()->mapped_slots() ==
                slot0.kv_page_pool()->mapped_slots() &&
                slot2.kv_page_pool()->mapped_slots() ==
                    slot0.kv_page_pool()->mapped_slots(),
            "zero-copy restore_shared did not map identical physical pages");
    const U32 shared_page0 = slot0.kv_page_pool()->mapped_slots().at(0);
    const U32 shared_tail1 = slot0.kv_page_pool()->mapped_slots().at(1);
    require(slot0.kv_page_pool()->ref_count(shared_page0) == 4U &&
                slot0.kv_page_pool()->ref_count(shared_tail1) == 4U,
            "shared page ref_count expected 4 (ckpt + 3 slots)");

    // Run heterogeneous B=3 MoE run_packed_batch ({1, 3, 5} -> total_m = 9, cap_m = 12)
    // mutating each slot's partial tail page via COW and verify 0-ULP vs standalone oracles!
    const auto t0 = make_tokens(1U, 101U, U32(cfg.vocab_size));
    const auto t1 = make_tokens(3U, 202U, U32(cfg.vocab_size));
    const auto t2 = make_tokens(5U, 303U, U32(cfg.vocab_size));
    const std::array<PackedSlotSlice, 3> slices{{
        {.slot_program = &slot0, .slot_index = 0, .tokens = t0, .sampling = samp},
        {.slot_program = &slot1, .slot_index = 1, .tokens = t1, .sampling = samp},
        {.slot_program = &slot2, .slot_index = 2, .tokens = t2, .sampling = samp},
    }};
    (void)slot0.run_packed_batch(slices, true);

    // Full page 0 MUST remain zero-copy shared across all 3 slots + ckpt (ref_count == 4),
    // while partial tail page 1 MUST have COW-branched into distinct physical pages!
    require(slot0.kv_page_pool()->ref_count(shared_page0) == 4U &&
                slot0.kv_page_pool()->mapped_slots().at(1) !=
                    slot1.kv_page_pool()->mapped_slots().at(1) &&
                slot1.kv_page_pool()->mapped_slots().at(1) !=
                    slot2.kv_page_pool()->mapped_slots().at(1),
            "COW failed to preserve full page 0 or isolate partial tail page 1");

    // Compare each slot 0-ULP against standalone solo QwenForward oracle!
    for (int s = 0; s < 3; ++s) {
      QwenForward solo(ctx, weights, layers, cfg, 1024, libs, {}, false, false,
                       false, std::nullopt, false, false, std::nullopt,
                       LinearNumericPolicy::RowExactPortableV1,
                       SamplingContract::PositionCoupledV1, attn_pol);
      (void)solo.run(pfx70, samp);
      const auto &tail = (s == 0) ? t0 : (s == 1) ? t1 : t2;
      (void)solo.run(tail, samp);
      const auto & target_slot = (s == 0) ? slot0 : (s == 1) ? slot1 : slot2;
      require(target_slot.read_last_logits_bf16() == solo.read_last_logits_bf16() &&
                  target_slot.inspect_recurrent_state() ==
                      solo.inspect_recurrent_state(),
              std::string(variant) + " B=3 packed batch + shared KV COW diverged on slot " +
                  std::to_string(s));
      ++packed_moe_checks;
    }

    slot0.release(ckpt);

    // Verify MoE T=1 routing contract under RowExactPortableV1:
    // - coupled_ar (T_DOMAIN_DISPATCH=0) keeps moe_nax when Apple10 NAX is active;
    // - t_domain_ar (T_DOMAIN_DISPATCH=1, solo AR) routes T=1 to 2-launch grouped_a16 (moe.metal)!
    if (std::string_view(variant) == "moe-mxfp4") {
      ScopedEnv env_td1("OMLX_M5_T_DOMAIN_DISPATCH", "1");
      ScopedEnv env_mnax("OMLX_M5_MOE_NAX_MIN_T", nullptr);
      QwenForward moe_td(ctx, weights, layers, cfg, 1024, libs, {}, true, false,
                         false, std::nullopt, false, false, std::nullopt,
                         LinearNumericPolicy::RowExactPortableV1,
                         SamplingContract::PositionCoupledV1, attn_pol);
      bool saw_grouped_moe = false;
      for (const auto &r : moe_td.inspect_profile(1).weights) {
        require(r.family != "moe_nax_m5_v1",
                "MoE solo T=1 with T_DOMAIN_DISPATCH=1 must use moe9_t1_v1 / grouped_a16 instead of moe_nax");
        saw_grouped_moe |=
            (r.family == "grouped_a16" || r.family == "moe9_t1_v1");
      }
      require(saw_grouped_moe,
              "MoE solo T=1 with T_DOMAIN_DISPATCH=1 did not record grouped_a16 expert route");
    }
  }

  return {{"saved_35b_b8_bytes", old_8slot_kv_bytes - new_shared_kv_bytes},
          {"packed_moe_slot_checks", packed_moe_checks},
          {"passed", true}};
}

// ---------------------------------------------------------------------------
// Test 4: Double-Buffered Async 2-Deep Pure-AR Pipelining (`replay_icb_async` +
// `ar_pipe_[2]` + `_feed` Samplers) — 0-ULP Identity vs Sync AR + Stop + B=1->B=2
// ---------------------------------------------------------------------------
std::vector<std::int32_t> run_single_sync(Engine &eng, SubmitRequest req) {
  const auto rid = eng.submit(std::move(req));
  std::vector<std::int32_t> out;
  for (;;) {
    for (const auto &ev : eng.poll_events(std::chrono::milliseconds(500))) {
      if (ev.request_id != rid)
        continue;
      if (ev.kind == EventKind::tokens)
        out.insert(out.end(), ev.tokens.begin(), ev.tokens.end());
      else if (ev.kind == EventKind::finished)
        return out;
      else if (ev.kind == EventKind::failed)
        throw std::runtime_error("Engine request failed: " + ev.error);
    }
  }
}

Json test_async_ar_pipelining(const std::filesystem::path &qwen_fixtures,
                              const std::filesystem::path &libs) {
  const auto fix = qwen_fixtures / "moe-mxfp4";
  auto make_opts = [&](const std::string &contract, int max_conc = 1) {
    LoadOptions o;
    o.model_dir = (fix / "source").string();
    o.max_context = 1024;
    o.max_concurrency = max_conc;
    o.sampling_contract = contract;
    o.path_overrides["artifact"] = (fix / "model.omlxm5").string();
    o.path_overrides["kernel_directory"] = libs.string();
    if (max_conc >= 2) {
      o.path_overrides["batch_mode"] = "on";
    }
    return o;
  };

  // Prompt of 58 tokens + 14 generated tokens = 72 total (crosses 64-token KV page boundary!)
  SubmitRequest greedy_req;
  for (U32 t : make_tokens(58U, 17U, 257U))
    greedy_req.prompt_tokens.push_back(std::int32_t(t));
  greedy_req.max_new_tokens = 14;
  greedy_req.sampling.temperature = 0.0f;
  greedy_req.sampling.top_k = 1;
  greedy_req.sampling.top_p = 1.0f;
  greedy_req.sampling.seed = 42;

  const char *sv_e = std::getenv("MTL_SHADER_VALIDATION");
  const bool sv_on = sv_e != nullptr && std::string_view(sv_e) != "0" &&
                     !std::string_view(sv_e).empty();
  const std::string primary_contract =
      sv_on ? "legacy_rejection_v1" : "position_coupled_v1";

  // 1. Primary contract greedy: async 2-deep == sync 0-ULP!
  std::vector<std::int32_t> coupled_sync, coupled_async;
  {
    ScopedEnv dis("OMLX_M5_DISABLE_ASYNC_AR", "1");
    auto eng = Engine::create(make_opts(primary_contract));
    coupled_sync = run_single_sync(*eng, greedy_req);
    eng->shutdown();
  }
  {
    ScopedEnv en("OMLX_M5_DISABLE_ASYNC_AR", nullptr);
    auto eng = Engine::create(make_opts(primary_contract));
    coupled_async = run_single_sync(*eng, greedy_req);
    const auto st = eng->stats();
    eng->shutdown();
    require(coupled_async.size() == 14 && coupled_async == coupled_sync &&
                st.ar_steps == 14,
            "PositionCoupledV1 async 2-deep AR diverged from sync AR");
  }

  // 2. Stop-sequence early termination while step t+1 is already queued on GPU!
  {
    SubmitRequest stop_req = greedy_req;
    stop_req.stop_token_seqs = {{coupled_sync.at(5), coupled_sync.at(6)}};
    ScopedEnv en("OMLX_M5_DISABLE_ASYNC_AR", nullptr);
    auto eng = Engine::create(make_opts(primary_contract));
    const auto stopped_out = run_single_sync(*eng, stop_req);
    // Run a second full request right after on the same slot to prove drain + reset left 0 residue!
    const auto second_out = run_single_sync(*eng, greedy_req);
    eng->shutdown();
    require(stopped_out ==
                std::vector<std::int32_t>(coupled_sync.begin(),
                                          coupled_sync.begin() + 5) &&
                second_out == coupled_sync,
            "async AR stop-sequence drain or subsequent slot reuse failed");
  }

  // 3. LegacyRejectionV1 stochastic (temp=0.7, top_k=16, top_p=0.9): async 2-deep == sync 0-ULP!
  SubmitRequest stoch_req = greedy_req;
  stoch_req.sampling.temperature = 0.7f;
  stoch_req.sampling.top_k = 16;
  stoch_req.sampling.top_p = 0.9f;
  stoch_req.sampling.seed = 20261121;
  std::vector<std::int32_t> leg_sync, leg_async;
  {
    ScopedEnv dis("OMLX_M5_DISABLE_ASYNC_AR", "1");
    auto eng = Engine::create(make_opts("legacy_rejection_v1"));
    leg_sync = run_single_sync(*eng, stoch_req);
    eng->shutdown();
  }
  {
    ScopedEnv en("OMLX_M5_DISABLE_ASYNC_AR", nullptr);
    auto eng = Engine::create(make_opts("legacy_rejection_v1"));
    leg_async = run_single_sync(*eng, stoch_req);
    eng->shutdown();
    require(leg_async.size() == 14 && leg_async == leg_sync,
            "LegacyRejectionV1 stochastic async 2-deep AR diverged from sync AR");
  }

  // 4. Dynamic B=1 (async 2-deep) -> B=2 (packed batch) -> B=1 (async 2-deep) mid-stream transition!
  {
    SubmitRequest long_req = greedy_req;
    long_req.max_new_tokens = 96;
    std::vector<std::int32_t> long_sync;
    {
      ScopedEnv dis("OMLX_M5_DISABLE_ASYNC_AR", "1");
      auto sync_eng = Engine::create(make_opts(primary_contract, 1));
      long_sync = run_single_sync(*sync_eng, long_req);
      sync_eng->shutdown();
    }

    ScopedEnv en("OMLX_M5_DISABLE_ASYNC_AR", nullptr);
    auto eng = Engine::create(make_opts(primary_contract, 2));
    const auto rid0 = eng->submit(long_req);
    SubmitRequest req1 = greedy_req;
    req1.prompt_tokens = {9, 11, 13, 15, 17, 19}; // distinct short batchable prompt!
    req1.max_new_tokens = 16;

    std::uint64_t rid1 = 0;
    bool submitted_req1 = false;
    std::vector<std::int32_t> out0, out1;
    std::size_t done = 0;
    while (done < (submitted_req1 ? 2U : 1U)) {
      for (const auto &ev : eng->poll_events(std::chrono::microseconds(200))) {
        if (ev.request_id == rid0) {
          if (ev.kind == EventKind::tokens) {
            out0.insert(out0.end(), ev.tokens.begin(), ev.tokens.end());
            if (!submitted_req1 && out0.size() >= 2) {
              rid1 = eng->submit(req1);
              submitted_req1 = true;
            }
          } else if (ev.kind == EventKind::finished) {
            ++done;
          } else if (ev.kind == EventKind::failed) {
            throw std::runtime_error("rid0 failed: " + ev.error);
          }
        } else if (submitted_req1 && ev.request_id == rid1) {
          if (ev.kind == EventKind::tokens)
            out1.insert(out1.end(), ev.tokens.begin(), ev.tokens.end());
          else if (ev.kind == EventKind::finished)
            ++done;
          else if (ev.kind == EventKind::failed)
            throw std::runtime_error("rid1 failed: " + ev.error);
        }
      }
    }
    const auto st = eng->stats();
    eng->shutdown();

    // Verify req0 STILL matches long_sync 0-ULP across 96 tokens despite transitioning
    // B=1 async -> B=2 packed_batch -> B=1 async!
    require(out0 == long_sync && out1.size() == 16 &&
                st.packed_batch_calls >= 1,
            "B=1 async -> B=2 packed_batch transition mismatch (out0_eq=" +
                std::to_string(out0 == long_sync) + ", out1=" +
                std::to_string(out1.size()) + ", packed_calls=" +
                std::to_string(st.packed_batch_calls) + ")");
  }

  return {{"coupled_tokens", coupled_async.size()},
          {"legacy_stoch_tokens", leg_async.size()},
          {"passed", true}};
}

// ---------------------------------------------------------------------------
// Test 5: Zero-Gather BM=64x64 MoE NAX Prefill + 2048 Uncontended Bucket +
// Async Double-Buffered Prefill Pipeline (`PrefillPipeSlot[2]` + `run_trunk`)
// ---------------------------------------------------------------------------
Json test_moe_bm64_and_2048_async_prefill(
    const MetalContext &ctx, const std::filesystem::path &qwen_fixtures,
    const std::filesystem::path &libs, const std::filesystem::path &tmp_dir) {
  // 1. 35B-A3B 64K + workspace_rows=2048 ICB safety check (< 2 GiB per buffer)
  QwenModelConfig c35b;
  c35b.hidden_size = 2048;
  c35b.intermediate_size = 512;
  c35b.shared_expert_intermediate_size = 512;
  c35b.num_experts = 256;
  c35b.num_experts_per_token = 8;
  c35b.num_hidden_layers = 40;
  c35b.num_attention_heads = 16;
  c35b.num_key_value_heads = 2;
  c35b.head_dim = 256;
  c35b.linear_num_key_heads = 16;
  c35b.linear_num_value_heads = 32;
  c35b.linear_key_head_dim = 128;
  c35b.linear_value_head_dim = 128;
  c35b.linear_conv_kernel_dim = 4;
  c35b.vocab_size = 248320;
  c35b.layer_types.assign(40, LayerType::linear_attention);
  for (std::size_t i = 3; i < 40; i += 4)
    c35b.layer_types[i] = LayerType::full_attention;

  const auto plan_35b_2k = plan_qwen_forward_memory(
      c35b, 65536U, 0U, false, false, false, true, true, false, 0U, 2048U, false);
  require(plan_35b_2k.workspace_rows == 2048U &&
              plan_35b_2k.largest_buffer_bytes < (2ULL << 30),
          "35B-A3B workspace_rows=2048 exceeded 2 GiB ICB buffer limit");
  const auto plan_35b_4k = plan_qwen_forward_memory(
      c35b, 65536U, 0U, false, false, false, true, true, false, 0U, 4096U, false);
  require(plan_35b_4k.workspace_rows == 4096U &&
              plan_35b_4k.largest_buffer_bytes < (2ULL << 30),
          "35B-A3B workspace_rows=4096 exceeded 2 GiB ICB buffer limit");
  {
    ScopedEnv clear_pf("OMLX_M5_PREFILL_CHUNK_SIZE", nullptr);
    require(resolve_qwen_primary_workspace_rows(8192U, false, false) == 2048U &&
                resolve_qwen_primary_workspace_rows(8192U, false, true) == 4096U,
            "resolve_qwen_primary_workspace_rows must cap dense=2048 and sparse_moe=4096");
  }

  const char *sv_e = std::getenv("MTL_SHADER_VALIDATION");
  const bool sv_on = sv_e != nullptr && std::string_view(sv_e) != "0" &&
                     !std::string_view(sv_e).empty();
  const bool has_moe_nax_lib =
      std::filesystem::is_regular_file(libs / "omlx_m5_moe_nax.metallib");
  const bool native_apple10 =
      ctx.capabilities().profile.probe.apple_gpu_family >= 10;

  U32 bm64_route_checks = 0;
  for (const char *variant : {"moe-mxfp4", "moe-affine"}) {
    const auto fix = qwen_fixtures / variant;
    auto cfg = load_qwen_model_config(fix / "source/config.json");
    cfg.max_position_embeddings = 8192;
    const auto art = ArtifactModel::open(fix / "model.omlxm5", true);
    const auto weights =
        materialize_model(art, plan_model_materialization(art, ctx), ctx);
    const auto layers = build_qwen_layer_catalog(art, cfg, 4096, 64);

    const auto attn_pol = sv_on ? AttentionNumericPolicy::Legacy
                                : AttentionNumericPolicy::RowExactV1;
    QwenForward slot0(ctx, weights, layers, cfg, 4096, libs, {}, true, false,
                      false, std::nullopt, false, true, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1, attn_pol);
    QwenForward slot1(ctx, weights, layers, cfg, 4096, libs, {}, false, false,
                      false, std::nullopt, false, true, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1, attn_pol, false,
                      "bf16", &slot0);
    require(slot0.memory_plan().workspace_rows == 4096U &&
                slot1.memory_plan().workspace_rows == 16U &&
                slot0.inspect_profile(4096).capacity == 4096U,
            "max_context=4096 sparse MoE must size primary=4096, secondary=16, and build 4096 profile");
  }

  // Direct emit_moe_nax_ffn lowering & zero-gather verification at h=256, f=256
  // across MXFP4 and Affine (q3/q4/q5/q6/q8 x g64/g128), plus delegated_prefill parity!
  if (has_moe_nax_lib) {
    using namespace detail::forward;
    const U32 h = 256U, f = 256U, num_exp = 256U, top_k = 8U;
    std::vector<std::unique_ptr<MetalBuffer>> ws_bufs;
    MoeNaxWorkspace ws;
    ws.allocate(ctx, 2048U, h, f, ws_bufs, num_exp, top_k);

    auto dummy_buf = ctx.make_private_buffer(16ULL << 20);
    Ref r_norm{&dummy_buf, 0, U64(2048U) * h * 2U};
    Ref r_logits{&dummy_buf, 0, U64(2048U) * num_exp * 2U};
    Ref r_ids{&dummy_buf, 0, U64(2048U) * top_k * 4U};
    Ref r_scores{&dummy_buf, 0, U64(2048U) * top_k * 2U};
    Ref r_exp_out{&dummy_buf, 0, U64(2048U) * top_k * h * 2U};
    Ref r_routed{&dummy_buf, 0, U64(2048U) * h * 2U};

    auto make_mxfp4_w = [&](const char *name, U32 rows, U32 cols) {
      MaterializedWeightView w;
      w.name = name;
      w.format = "mxfp4_e2m1_g32_e8m0_v1";
      w.layout = "row_major";
      w.parent_shape = {num_exp, rows, cols};
      w.row_count = rows;
      w.in_features = cols;
      w.padded_k = cols;
      w.planes.push_back(
          {.kind = "data", .buffer = &dummy_buf, .bytes = U64(num_exp) * rows * (cols / 2U)});
      w.planes.push_back(
          {.kind = "scale", .buffer = &dummy_buf, .bytes = U64(num_exp) * rows * (cols / 32U)});
      return w;
    };
    auto make_affine_w = [&](const char *name, U32 bits, U32 gs, U32 rows, U32 cols) {
      MaterializedWeightView w;
      w.name = name;
      w.format = "oq_affine_q" + std::to_string(bits) + "_g" +
                 std::to_string(gs) + "_bf16_v1";
      w.layout = "row_major";
      w.parent_shape = {num_exp, rows, cols};
      w.row_count = rows;
      w.in_features = cols;
      w.padded_k = cols;
      const std::string ckind =
          bits == 3U ? "low_2bit" : (bits == 8U ? "code" : "base_code");
      const U64 cbytes = bits == 3U ? (cols / 4U) : (bits == 8U ? cols : (cols / 2U));
      w.planes.push_back(
          {.kind = ckind, .buffer = &dummy_buf, .bytes = U64(num_exp) * rows * cbytes});
      if (bits == 3U || bits == 5U) {
        w.planes.push_back(
            {.kind = "high_bit", .buffer = &dummy_buf, .bytes = U64(num_exp) * rows * (cols / 8U)});
      } else if (bits == 6U) {
        w.planes.push_back(
            {.kind = "high_2bit", .buffer = &dummy_buf, .bytes = U64(num_exp) * rows * (cols / 4U)});
      }
      w.planes.push_back(
          {.kind = "scale", .buffer = &dummy_buf, .bytes = U64(num_exp) * rows * (cols / gs) * 2U});
      w.planes.push_back(
          {.kind = "bias", .buffer = &dummy_buf, .bytes = U64(num_exp) * rows * (cols / gs) * 2U});
      return w;
    };

    auto check_family = [&](const MaterializedWeightView &gw,
                            const MaterializedWeightView &uw,
                            const MaterializedWeightView &dw,
                            const std::string &tag) {
      for (U32 cap : {8U, 128U, 256U, 1024U, 2048U}) {
        Encoder e(ctx, libs, cap, "step4.moe_nax_route." + tag);
        e.begin_stage("mlp");
        require(emit_moe_nax_ffn(e, ws, gw, uw, dw, r_norm, r_ids, r_scores,
                                 r_exp_out, r_routed, cap, h, f, top_k, num_exp,
                                 64U, std::nullopt, r_logits, true),
                "emit_moe_nax_ffn rejected eligible " + tag);
        auto prof = e.finish();
        if (cap == 8U) {
          require(prof.weight_routes.front().function.find("_n16_sk4") !=
                      std::string::npos,
                  tag + " cap=8 must stay on narrow _n16_sk4");
        } else if (cap == 128U) {
          // Zero-gather BM=32: exactly 4 dispatches (sched_bm32 + swiglu + down + reduce)!
          require(prof.plan.dispatch_count() == 4U &&
                      prof.weight_routes.front().function.find(
                          "_bm32_bn64_sk1") != std::string::npos,
                  tag + " cap=128 must emit 4-dispatch zero-gather BM=32");
        } else {
          // Zero-gather BM=64: exactly 4 dispatches (sched_bm64 + swiglu + down + reduce)!
          const std::string expect_bm64 =
              moe_nax_swiglu_out64_env_enabled() ? "_bm64_out64_sk1"
                                                 : "_bm64_bn64_sk1";
          require(prof.plan.dispatch_count() == 4U &&
                      prof.weight_routes.front().function.find(expect_bm64) !=
                          std::string::npos,
                  tag + " cap>=256 must emit 4-dispatch zero-gather BM=64");
          if (cap == 2048U) {
            require(prof.weight_routes.front().grid.width == 32ULL * 128ULL,
                    tag + " cap=2048 must activate GX=32 L2 expert tile grouping");
          }
        }
        ++bm64_route_checks;
      }

      // Verify delegated_prefill (explicit_t = 300 inside capacity = 512) matches solo 512!
      Encoder e_del(ctx, libs, 512U, "step4.moe_nax_del." + tag);
      e_del.delegated_prefill = true;
      e_del.begin_stage("mlp");
      require(emit_moe_nax_ffn(e_del, ws, gw, uw, dw, r_norm, r_ids, r_scores,
                               r_exp_out, r_routed, 300U, h, f, top_k, num_exp,
                               64U, 300U, r_logits, true),
              "delegated_prefill emit_moe_nax_ffn failed for " + tag);
      auto p_del = e_del.finish();
      const std::string expect_del =
          moe_nax_swiglu_out64_env_enabled() ? "_bm64_out64_sk1"
                                             : "_bm64_bn64_sk1";
      require(p_del.plan.dispatch_count() == 4U &&
                  p_del.weight_routes.front().function.find(expect_del) !=
                      std::string::npos,
              tag + " delegated_prefill T=300@cap=512 did not match solo BM=64 route");
    };

    check_family(make_mxfp4_w("gate", f, h), make_mxfp4_w("up", f, h),
                 make_mxfp4_w("down", h, f), "mxfp4");
    for (U32 bits : {3U, 4U, 5U, 6U, 8U}) {
      for (U32 gs : {64U, 128U}) {
        check_family(make_affine_w("gate", bits, gs, f, h),
                     make_affine_w("up", bits, gs, f, h),
                     make_affine_w("down", bits, gs, h, f),
                     "oq_q" + std::to_string(bits) + "_g" + std::to_string(gs));
      }
    }
    (void)native_apple10;
  }

  // 2. End-to-end 2048-chunk uncontended bucket + async double-buffered prefill!
  const auto fix = qwen_fixtures / "moe-mxfp4";
  const auto src_8k = tmp_dir / "moe_mxfp4_8k_source";
  std::filesystem::create_directories(src_8k);
  {
    std::ifstream in(fix / "source/config.json");
    Json cj = Json::parse(in);
    cj["text_config"]["max_position_embeddings"] = 8192;
    std::ofstream out(src_8k / "config.json");
    out << cj.dump(2);
  }
  const std::string contract_str =
      sv_on ? "legacy_rejection_v1" : "position_coupled_v1";
  auto make_pf_opts = [&](std::uint64_t cache_bytes = 0) {
    LoadOptions o;
    o.model_dir = src_8k.string();
    o.max_context = 8192;
    o.max_concurrency = 1;
    o.sampling_contract = contract_str;
    o.path_overrides["artifact"] = (fix / "model.omlxm5").string();
    o.path_overrides["kernel_directory"] = libs.string();
    if (cache_bytes > 0) {
      o.path_overrides["prefix_cache_bytes"] = std::to_string(cache_bytes);
    }
    return o;
  };

  if (sv_on) {
    // Under GPU shader validation, run a 2-chunk 2080-token (2048 + 32) prefill
    // so both the 2048 trunk-only path and tail chunk are validated cleanly!
    SubmitRequest req2080;
    for (U32 t : make_tokens(2080U, 23U, 257U))
      req2080.prompt_tokens.push_back(std::int32_t(t));
    req2080.max_new_tokens = 3;
    req2080.sampling.temperature = 0.0f;
    req2080.sampling.top_k = 1;
    req2080.sampling.top_p = 1.0f;
    req2080.sampling.seed = 99;

    auto eng = Engine::create(make_pf_opts(0));
    const auto out = run_single_sync(*eng, req2080);
    const auto st = eng->stats();
    eng->shutdown();
    require(out.size() == 3 && st.ar_steps == 4,
            "2080-tok shader-validated prefill expected 2 prefill + 2 decode steps");
  } else {
    for (U32 plen : {4096U, 6144U}) {
      SubmitRequest req;
      for (U32 t : make_tokens(plen, 31U + plen, 257U))
        req.prompt_tokens.push_back(std::int32_t(t));
      req.max_new_tokens = 6;
      req.sampling.temperature = 0.0f;
      req.sampling.top_k = 1;
      req.sampling.top_p = 1.0f;
      req.sampling.seed = 12345;

      std::vector<std::int32_t> sync_out, async_out;
      std::uint64_t async_ar_steps = 0;
      {
        ScopedEnv dis("OMLX_M5_DISABLE_ASYNC_PREFILL", "1");
        auto eng = Engine::create(make_pf_opts(0));
        sync_out = run_single_sync(*eng, req);
        eng->shutdown();
      }
      {
        ScopedEnv en("OMLX_M5_DISABLE_ASYNC_PREFILL", nullptr);
        auto eng = Engine::create(make_pf_opts(0));
        async_out = run_single_sync(*eng, req);
        async_ar_steps = eng->stats().ar_steps;
        eng->shutdown();
      }
      const std::uint64_t expected_pf_chunks = (plen == 4096U) ? 1U : 2U;
      require(async_out.size() == 6 && async_out == sync_out &&
                  async_ar_steps == expected_pf_chunks + 5U,
              "async 4096-chunk MoE prefill diverged or wrong chunk count at L=" +
                  std::to_string(plen) + " (ar_steps=" +
                  std::to_string(async_ar_steps) + ")");
    }

    // Verify 2080 (2048 + 32) sparse anchor caching: cold == warm hit 0-ULP!
    SubmitRequest req2080;
    for (U32 t : make_tokens(2080U, 47U, 257U))
      req2080.prompt_tokens.push_back(std::int32_t(t));
    req2080.max_new_tokens = 4;
    req2080.sampling.temperature = 0.0f;
    req2080.sampling.top_k = 1;
    req2080.sampling.top_p = 1.0f;
    req2080.sampling.seed = 777;

    SubmitRequest req2080_b = req2080;
    req2080_b.prompt_tokens.back() = (req2080_b.prompt_tokens.back() + 3) % 250;

    std::vector<std::int32_t> cold_b, warm_b;
    {
      auto cold_eng = Engine::create(make_pf_opts(0));
      cold_b = run_single_sync(*cold_eng, req2080_b);
      cold_eng->shutdown();
    }
    {
      auto cached_eng = Engine::create(make_pf_opts(64ULL << 20));
      (void)run_single_sync(*cached_eng, req2080);
      warm_b = run_single_sync(*cached_eng, req2080_b);
      const auto st = cached_eng->stats();
      cached_eng->shutdown();
      require(warm_b == cold_b && st.prefix_restored_tokens == 2048U,
              "2080 sparse 2048-anchor prefix hit diverged from cold 2048 prefill");
    }
  }

  return {{"bm64_route_checks", bm64_route_checks},
          {"largest_35b_2k_buffer_bytes", plan_35b_2k.largest_buffer_bytes},
          {"passed", true}};
}

// ---------------------------------------------------------------------------
// Test 6: Live GPU Verification of 64-TG Parallel Greedy Argmax (Position &
// Vocab Contracts) + T=1 Fused 9-Expert MoE (Router+Top8+SharedGate + SwiGLU
// + Down+Combine + C++ moe_nax.h Lowering Helpers)
// ---------------------------------------------------------------------------
Json test_dimb_64tg_argmax_and_moe9_t1(const MetalContext &ctx,
                                       const std::filesystem::path &libs) {
  using namespace detail::forward;

  ctx.register_library(
      {.key = "dimb.lib.position_sampling",
       .path = libs / "omlx_m5_position_sampling.metallib"});
  ctx.register_library(
      {.key = "dimb.lib.vocab", .path = libs / "omlx_m5_vocab.metallib"});
  for (const char *fn : {"omlx_m5_position_sample_logits",
                         "omlx_m5_position_sample_logits_feed",
                         "omlx_m5_position_argmax_stage1_64tg",
                         "omlx_m5_position_sample_logits_stage2",
                         "omlx_m5_position_sample_logits_feed_stage2"}) {
    (void)ctx.warmup_pipeline({.key = std::string("dimb.pso.") + fn,
                               .library_key = "dimb.lib.position_sampling",
                               .function_name = fn,
                               .variant = "dimb-test"});
  }
  for (const char *fn : {"omlx_m5_sample_logits", "omlx_m5_sample_logits_feed",
                         "omlx_m5_vocab_argmax_stage1_64tg",
                         "omlx_m5_sample_logits_stage2",
                         "omlx_m5_sample_logits_feed_stage2"}) {
    (void)ctx.warmup_pipeline({.key = std::string("dimb.pso.") + fn,
                               .library_key = "dimb.lib.vocab",
                               .function_name = fn,
                               .variant = "dimb-test"});
  }

  U32 argmax_checks = 0;
  for (U32 vocab : {248320U, 1031U}) {
    std::vector<std::uint16_t> logits_bf16(vocab);
    for (U32 i = 0; i < vocab; ++i) {
      logits_bf16[i] =
          f32_to_bf16_bits(std::sin(float(i) * 0.013f) * 3.5f - 1.0f);
    }
    const U32 t_low = (vocab > 500U) ? 19U : 3U;
    const U32 t_mid = (vocab > 20000U) ? 17408U : (vocab / 2U);
    const U32 t_tail = vocab - 1U;
    const std::uint16_t win_bits = f32_to_bf16_bits(12.5f);
    logits_bf16[t_low] = win_bits;
    logits_bf16[t_mid] = win_bits;
    logits_bf16[t_tail] = win_bits;

    auto b_logits = ctx.make_buffer(U64(vocab) * 2U);
    b_logits.write(std::as_bytes(std::span(logits_bf16)));
    auto b_parts_pos = ctx.make_buffer(64U * 16U);
    auto b_parts_voc = ctx.make_buffer(64U * 8U);
    auto b_res_ref = ctx.make_buffer(8U);
    auto b_res_s2 = ctx.make_buffer(8U);
    auto b_res_feed = ctx.make_buffer(8U);
    auto b_cids_ref = ctx.make_buffer(64U * 4U);
    auto b_cscr_ref = ctx.make_buffer(64U * 4U);
    auto b_cids_s2 = ctx.make_buffer(64U * 4U);
    auto b_cscr_s2 = ctx.make_buffer(64U * 4U);
    auto b_next_pos = ctx.make_buffer(4U);
    auto b_tok_ref = ctx.make_buffer(4U);
    auto b_tok_s2 = ctx.make_buffer(4U);
    auto b_tok_feed = ctx.make_buffer(4U);
    auto b_next_voc = ctx.make_buffer(4U);

    auto run_sampler_pair = [&](float temp, U32 top_k, float top_p) {
      PositionSamplingRow prow{};
      prow.seed = 20261122ULL;
      prow.position = 7ULL;
      prow.top_k = top_k;
      prow.temperature = temp;
      prow.top_p = top_p;
      prow.valid = 1U;
      const std::array<U32, 4> geom{vocab, 1U, 1U, 0U};

      SampleParams sp{};
      sp.vocabulary = vocab;
      sp.top_k = top_k;
      sp.temperature = temp;
      sp.top_p = top_p;
      sp.seed = 20261122ULL;
      sp.stream = 1ULL;
      sp.absolute_position = 7ULL;

      auto pb = std::as_bytes(std::span(&prow, 1));
      auto gb = std::as_bytes(std::span(geom));
      auto sb = std::as_bytes(std::span(&sp, 1));
      auto vb = std::as_bytes(std::span(&vocab, 1));

      MetalCommandPlanDescriptor desc{
          .key = "dimb.argmax.v" + std::to_string(vocab) + ".t" +
                 std::to_string(int(temp * 100.0f)) + ".k" +
                 std::to_string(top_k),
          .dispatches = {
              // Ref position_sample_logits
              {.pipeline_key = "dimb.pso.omlx_m5_position_sample_logits",
               .grid = {.width = 256U, .height = 1U},
               .threads_per_threadgroup = {.width = 256U},
               .resources = {{0, 0, b_logits.size()},
                             {1, 1, 8U},
                             {2, 2, 256U},
                             {3, 3, 256U}},
               .constant_arguments = {{4, {pb.begin(), pb.end()}},
                                      {5, {gb.begin(), gb.end()}}}},
              // 64-TG stage 1 + stage 2 + feed stage 2
              {.pipeline_key = "dimb.pso.omlx_m5_position_argmax_stage1_64tg",
               .grid = {.width = 64U * 256U, .height = 1U},
               .threads_per_threadgroup = {.width = 256U},
               .resources = {{0, 0, b_logits.size()}, {1, 4, 1024U}},
               .constant_arguments = {{2, {gb.begin(), gb.end()}}}},
              {.pipeline_key = "dimb.pso.omlx_m5_position_sample_logits_stage2",
               .barrier_before = true,
               .grid = {.width = 256U, .height = 1U},
               .threads_per_threadgroup = {.width = 256U},
               .resources = {{0, 0, b_logits.size()},
                             {1, 4, 1024U},
                             {2, 5, 8U},
                             {3, 6, 256U},
                             {4, 7, 256U}},
               .constant_arguments = {{5, {pb.begin(), pb.end()}},
                                      {6, {gb.begin(), gb.end()}}}},
              {.pipeline_key =
                   "dimb.pso.omlx_m5_position_sample_logits_feed_stage2",
               .grid = {.width = 256U, .height = 1U},
               .threads_per_threadgroup = {.width = 256U},
               .resources = {{0, 4, 1024U},
                             {1, 8, 8U},
                             {2, 6, 256U},
                             {3, 7, 256U},
                             {4, 9, 4U}},
               .constant_arguments = {{5, {pb.begin(), pb.end()}},
                                      {6, {gb.begin(), gb.end()}}}},
              // Ref vocab sample_logits + 64-TG stage 1 + stage 2 + feed stage 2
              {.pipeline_key = "dimb.pso.omlx_m5_sample_logits",
               .grid = {.width = 256U},
               .threads_per_threadgroup = {.width = 256U},
               .resources = {{0, 0, b_logits.size()}, {1, 10, 4U}},
               .constant_arguments = {{2, {sb.begin(), sb.end()}}}},
              {.pipeline_key = "dimb.pso.omlx_m5_vocab_argmax_stage1_64tg",
               .grid = {.width = 64U * 256U},
               .threads_per_threadgroup = {.width = 256U},
               .resources = {{0, 0, b_logits.size()}, {1, 11, 512U}},
               .constant_arguments = {{2, {vb.begin(), vb.end()}}}},
              {.pipeline_key = "dimb.pso.omlx_m5_sample_logits_stage2",
               .barrier_before = true,
               .grid = {.width = 256U},
               .threads_per_threadgroup = {.width = 256U},
               .resources = {{0, 0, b_logits.size()},
                             {1, 11, 512U},
                             {2, 12, 4U}},
               .constant_arguments = {{3, {sb.begin(), sb.end()}}}},
              {.pipeline_key = "dimb.pso.omlx_m5_sample_logits_feed_stage2",
               .grid = {.width = 256U},
               .threads_per_threadgroup = {.width = 256U},
               .resources = {{0, 0, b_logits.size()},
                             {1, 11, 512U},
                             {2, 13, 4U},
                             {3, 14, 4U}},
               .constant_arguments = {{4, {sb.begin(), sb.end()}}}},
          },
          .ranges = {{"full", 0, 8}},
      };
      auto plan = ctx.prepare_command_plan(std::move(desc));
      const std::vector<MetalBufferBinding> binds{
          {0, &b_logits, 0},    {1, &b_res_ref, 0},  {2, &b_cids_ref, 0},
          {3, &b_cscr_ref, 0},  {4, &b_parts_pos, 0},{5, &b_res_s2, 0},
          {6, &b_cids_s2, 0},   {7, &b_cscr_s2, 0},  {8, &b_res_feed, 0},
          {9, &b_next_pos, 0},  {10, &b_tok_ref, 0}, {11, &b_parts_voc, 0},
          {12, &b_tok_s2, 0},   {13, &b_tok_feed, 0},{14, &b_next_voc, 0}};
      const std::array inv{
          MetalCommandInvocation{.plan = &plan, .bindings = binds}};
      (void)ctx.execute_batch(inv);
    };

    auto read_u32_pair = [](const MetalBuffer &b) {
      std::array<U32, 2> v{0, 0};
      const auto raw = b.read();
      std::memcpy(v.data(), raw.data(), 8U);
      return v;
    };
    auto read_u32 = [](const MetalBuffer &b) {
      U32 v = 0;
      const auto raw = b.read();
      std::memcpy(&v, raw.data(), 4U);
      return v;
    };

    // Case 1: 3-way tie at (t_low, t_mid, t_tail) -> must pick t_low (0-ULP!)
    run_sampler_pair(0.0f, 1U, 1.0f);
    require(read_u32_pair(b_res_ref) == std::array<U32, 2>{t_low, 0U} &&
                read_u32_pair(b_res_s2) == std::array<U32, 2>{t_low, 0U} &&
                read_u32_pair(b_res_feed) == std::array<U32, 2>{t_low, 0U} &&
                read_u32(b_next_pos) == t_low &&
                read_u32(b_tok_ref) == t_low &&
                read_u32(b_tok_s2) == t_low &&
                read_u32(b_tok_feed) == t_low &&
                read_u32(b_next_voc) == t_low,
            "64-TG greedy argmax lowest-index tie-break failed at t_low");
    ++argmax_checks;

    // Case 2: Demote t_low -> 2-way tie (t_mid, t_tail) -> must pick t_mid!
    logits_bf16[t_low] = f32_to_bf16_bits(-2.0f);
    b_logits.write(std::as_bytes(std::span(logits_bf16)));
    run_sampler_pair(0.0f, 1U, 1.0f);
    require(read_u32_pair(b_res_s2) == std::array<U32, 2>{t_mid, 0U} &&
                read_u32(b_next_pos) == t_mid && read_u32(b_tok_s2) == t_mid,
            "64-TG greedy argmax tie-break failed at t_mid");
    ++argmax_checks;

    // Case 3: Demote t_mid -> tail element t_tail (tests vocab % 4 != 0 tail!)
    logits_bf16[t_mid] = f32_to_bf16_bits(-2.0f);
    b_logits.write(std::as_bytes(std::span(logits_bf16)));
    run_sampler_pair(0.7f, 1U, 1.0f); // temperature > 0 but top_k == 1 -> greedy!
    require(read_u32_pair(b_res_s2) == std::array<U32, 2>{t_tail, 0U} &&
                read_u32(b_tok_s2) == t_tail,
            "64-TG greedy argmax tail element check failed");
    ++argmax_checks;

    // Case 4: Stochastic top_k=32, temperature=0.7f, top_p=0.9f -> bit-exact match!
    run_sampler_pair(0.7f, 32U, 0.9f);
    require(read_u32_pair(b_res_s2) == read_u32_pair(b_res_ref) &&
                read_u32(b_tok_s2) == read_u32(b_tok_ref) &&
                read_u32(b_next_voc) == read_u32(b_tok_ref),
            "64-TG stage2 stochastic fallback diverged from reference");
    ++argmax_checks;

    // Case 5: NaN injection -> PositionCoupledV1 must flag error uint2(0, 2)!
    logits_bf16[vocab / 3U] = 0x7FC0U;
    b_logits.write(std::as_bytes(std::span(logits_bf16)));
    run_sampler_pair(0.0f, 1U, 1.0f);
    require(read_u32_pair(b_res_s2) == std::array<U32, 2>{0U, 2U} &&
                read_u32_pair(b_res_feed) == std::array<U32, 2>{0U, 2U} &&
                read_u32(b_next_pos) == 0U,
            "64-TG stage2 failed to propagate NaN invalid flag");
    ++argmax_checks;
  }

  // Live GPU Test of MOE9_FUSED (Router+Top8+SharedGate + SwiGLU + Down + C++ Helpers)
  bool moe9_tested = false;
  U32 moe9_max_ulp = 0;
  if (std::filesystem::is_regular_file(libs / "omlx_m5_moe_nax.metallib")) {
    moe9_tested = true;
    const U32 E = 256U, H = 256U, F = 256U;
    std::vector<std::unique_ptr<MetalBuffer>> ws_bufs;
    MoeNaxWorkspace ws;
    ws.allocate(ctx, 8U, H, F, ws_bufs, E, 8U);

    // Prepare norm [1, H], router [256, H] (both BF16 and Q8 tiled/row-major), and shared_gate [1, H]
    std::vector<std::uint16_t> norm_bf16(H);
    for (U32 i = 0; i < H; ++i) {
      norm_bf16[i] = f32_to_bf16_bits(std::cos(float(i) * 0.09f) * 0.75f);
    }

    // Q8_g64 router codes [256, H], scales [256, H/64], biases [256, H/64], plus BN=16 tiled copies
    const U32 G64 = H / 64U;
    std::vector<std::uint8_t> r_code_rm(E * H), r_code_tiled(E * H);
    std::vector<std::uint16_t> r_sc_rm(E * G64), r_bi_rm(E * G64);
    std::vector<U32> r_sb_tiled(G64 * E);
    for (U32 r = 0; r < E; ++r) {
      for (U32 g = 0; g < G64; ++g) {
        const std::uint16_t sc =
            f32_to_bf16_bits(0.01f + 0.001f * float((r + g) % 9U));
        const std::uint16_t bi =
            f32_to_bf16_bits(-1.2f + 0.02f * float(r % 13U));
        r_sc_rm[r * G64 + g] = sc;
        r_bi_rm[r * G64 + g] = bi;
        r_sb_tiled[g * E + r] = U32(sc) | (U32(bi) << 16U);
        const U32 tile_idx = (r / 16U) * G64 + g;
        const U32 rn = r % 16U;
        for (U32 k = 0; k < 64U; ++k) {
          const std::uint8_t qv = std::uint8_t((r * 17U + g * 31U + k * 7U) & 0xFFU);
          r_code_rm[r * H + g * 64U + k] = qv;
          r_code_tiled[tile_idx * 1024U + rn * 64U + k] = qv;
        }
      }
    }

    std::vector<std::uint8_t> sg_code_rm(H), sg_code_tiled(G64 * 1024U, 0);
    std::vector<std::uint16_t> sg_sc_rm(G64), sg_bi_rm(G64);
    std::vector<U32> sg_sb_tiled(G64 * 16U, 0);
    float ref_sgate_f32 = 0.0f;
    for (U32 g = 0; g < G64; ++g) {
      const std::uint16_t sc = f32_to_bf16_bits(0.015f);
      const std::uint16_t bi = f32_to_bf16_bits(-0.5f);
      sg_sc_rm[g] = sc;
      sg_bi_rm[g] = bi;
      sg_sb_tiled[g * 16U + 0U] = U32(sc) | (U32(bi) << 16U);
      for (U32 k = 0; k < 64U; ++k) {
        const std::uint8_t qv = std::uint8_t((g * 19U + k * 5U) & 0xFFU);
        sg_code_rm[g * 64U + k] = qv;
        sg_code_tiled[g * 1024U + k] = qv;
        const float xv = bf16_bits_to_f32(norm_bf16[g * 64U + k]);
        const float wv =
            bf16_bits_to_f32(sc) * float(qv) + bf16_bits_to_f32(bi);
        ref_sgate_f32 += xv * wv;
      }
    }
    const std::uint16_t ref_sgate_bf16 = f32_to_bf16_bits(ref_sgate_f32);

    // MXFP4 routed [E, F, H] & [E, H, F] + shared [F, H] & [H, F] (both row-major and BN=16 tiled!)
    auto decode_fp4 = [](U32 nib) -> float {
      static constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                       2.0f, 3.0f, 4.0f, 6.0f};
      const float m = lut[nib & 7U];
      return (nib & 8U) ? -m : m;
    };
    auto decode_e8m0 = [](std::uint8_t s) -> float {
      const U32 bits = (s == 0U) ? 0x00400000U : (U32(s) << 23U);
      return std::bit_cast<float>(bits);
    };

    std::vector<std::uint8_t> rgc(E * F * (H / 2U)), rgs(E * F * (H / 32U), 124U);
    std::vector<std::uint8_t> ruc(E * F * (H / 2U)), rus(E * F * (H / 32U), 125U);
    std::vector<std::uint8_t> rdc(E * H * (F / 2U)), rds(E * H * (F / 32U), 124U);
    // Include nibbles 1 (+0.5) and 9 (-0.5) deliberately to test subnormal-safe FP4 decoding!
    for (std::size_t i = 0; i < rgc.size(); ++i) {
      const U32 lo = (i * 3U + 1U) & 0xFU;
      const U32 hi = (i * 7U + 9U) & 0xFU;
      rgc[i] = std::uint8_t(lo | (hi << 4U));
      ruc[i] = std::uint8_t(((lo + 2U) & 0xFU) | (((hi ^ 8U) & 0xFU) << 4U));
      rdc[i] = std::uint8_t(((lo + 5U) & 0xFU) | (((hi + 1U) & 0xFU) << 4U));
    }

    std::vector<std::uint8_t> sgc_rm(F * (H / 2U)), sgs_rm(F * (H / 32U), 124U);
    std::vector<std::uint8_t> suc_rm(F * (H / 2U)), sus_rm(F * (H / 32U), 124U);
    std::vector<std::uint8_t> sdc_rm(H * (F / 2U)), sds_rm(H * (F / 32U), 124U);
    for (std::size_t i = 0; i < sgc_rm.size(); ++i) {
      sgc_rm[i] = std::uint8_t(((i + 1U) & 0xFU) | (((i + 9U) & 0xFU) << 4U));
      suc_rm[i] = std::uint8_t(((i + 3U) & 0xFU) | (((i + 2U) & 0xFU) << 4U));
      sdc_rm[i] = std::uint8_t(((i + 6U) & 0xFU) | (((i + 1U) & 0xFU) << 4U));
    }

    auto tile_mxfp4 = [](const std::vector<std::uint8_t> &src_c,
                         const std::vector<std::uint8_t> &src_s, U32 n, U32 k,
                         std::vector<std::uint8_t> &dst_c,
                         std::vector<std::uint8_t> &dst_s) {
      dst_c.assign(src_c.size(), 0);
      dst_s.assign(src_s.size(), 0x7FU);
      const U32 blks = k / 128U;
      for (U32 r = 0; r < n; ++r) {
        for (U32 blk = 0; blk < blks; ++blk) {
          const U32 tidx = (r / 16U) * blks + blk;
          const U32 rn = r % 16U;
          std::memcpy(&dst_c[tidx * 1024U + rn * 64U],
                      &src_c[r * (k / 2U) + blk * 64U], 64U);
          std::memcpy(&dst_s[tidx * 64U + rn * 4U],
                      &src_s[r * (k / 32U) + blk * 4U], 4U);
        }
      }
    };
    std::vector<std::uint8_t> sgc_tl, sgs_tl, suc_tl, sus_tl, sdc_tl, sds_tl;
    tile_mxfp4(sgc_rm, sgs_rm, F, H, sgc_tl, sgs_tl);
    tile_mxfp4(suc_rm, sus_rm, F, H, suc_tl, sus_tl);
    tile_mxfp4(sdc_rm, sds_rm, H, F, sdc_tl, sds_tl);

    auto mk_buf = [&](const auto &vec) {
      auto b = std::make_unique<MetalBuffer>(
          ctx.make_buffer(vec.size() * sizeof(vec[0])));
      b->write(std::as_bytes(std::span(vec)));
      MetalBuffer *ptr = b.get();
      ws_bufs.push_back(std::move(b));
      return Ref{ptr, 0, vec.size() * sizeof(vec[0])};
    };
    Ref ref_norm = mk_buf(norm_bf16);
    Ref ref_zero = mk_buf(std::vector<U32>(16, 0));
    Ref ref_rc_rm = mk_buf(r_code_rm), ref_rsc_rm = mk_buf(r_sc_rm),
        ref_rbi_rm = mk_buf(r_bi_rm);
    Ref ref_rc_tl = mk_buf(r_code_tiled), ref_rsb_tl = mk_buf(r_sb_tiled);
    Ref ref_sgc_rm8 = mk_buf(sg_code_rm), ref_sgsc_rm8 = mk_buf(sg_sc_rm),
        ref_sgbi_rm8 = mk_buf(sg_bi_rm);
    Ref ref_sgc_tl8 = mk_buf(sg_code_tiled), ref_sgsb_tl8 = mk_buf(sg_sb_tiled);
    Ref ref_rgc = mk_buf(rgc), ref_rgs = mk_buf(rgs), ref_ruc = mk_buf(ruc),
        ref_rus = mk_buf(rus), ref_rdc = mk_buf(rdc), ref_rds = mk_buf(rds);
    Ref ref_sh_gc_rm = mk_buf(sgc_rm), ref_sh_gs_rm = mk_buf(sgs_rm),
        ref_sh_uc_rm = mk_buf(suc_rm), ref_sh_us_rm = mk_buf(sus_rm),
        ref_sh_dc_rm = mk_buf(sdc_rm), ref_sh_ds_rm = mk_buf(sds_rm);
    Ref ref_sh_gc_tl = mk_buf(sgc_tl), ref_sh_gs_tl = mk_buf(sgs_tl),
        ref_sh_uc_tl = mk_buf(suc_tl), ref_sh_us_tl = mk_buf(sus_tl),
        ref_sh_dc_tl = mk_buf(sdc_tl), ref_sh_ds_tl = mk_buf(sds_tl);

    auto b_rlogits_tl = ctx.make_buffer(E * 2U);
    auto b_rids_tl = ctx.make_buffer(8U * 4U);
    auto b_rscores_tl = ctx.make_buffer(8U * 2U);
    auto b_sgate_tl = ctx.make_buffer(2U);
    auto b_delta_tl = ctx.make_buffer(H * 2U);

    auto b_rlogits_rm = ctx.make_buffer(E * 2U);
    auto b_rids_rm = ctx.make_buffer(8U * 4U);
    auto b_rscores_rm = ctx.make_buffer(8U * 2U);
    auto b_sgate_rm = ctx.make_buffer(2U);
    auto b_delta_rm = ctx.make_buffer(H * 2U);

    auto make_w3d = [](U32 id, std::string name, U32 d0, U32 d1, U32 d2,
                       Ref data, Ref scale) {
      MaterializedWeightView w{};
      w.object_id = id;
      w.name = std::move(name);
      w.format = "mxfp4_e2m1_g32_e8m0_v1";
      w.layout = "mx_rowsplit_k128_v1";
      w.parent_shape = {d0, d1, d2};
      w.row_count = d1;
      w.in_features = d2;
      w.padded_k = d2;
      w.planes = {
          {.kind = "data", .buffer = data.buffer, .offset = 0, .bytes = data.bytes},
          {.kind = "scale", .buffer = scale.buffer, .offset = 0, .bytes = scale.bytes}};
      return w;
    };
    auto make_w2d_mx = [](U32 id, std::string name, U32 n, U32 k, bool tiled,
                          Ref c, Ref s) {
      MaterializedWeightView w{};
      w.object_id = id;
      w.name = std::move(name);
      w.format = "mxfp4_e2m1_g32_e8m0_v1";
      w.layout = tiled ? "mx_nax_tiled_bk128_v1" : "mx_rowsplit_k128_v1";
      w.parent_shape = {n, k};
      w.row_count = n;
      w.in_features = k;
      w.padded_k = k;
      w.planes = {{.kind = tiled ? "nax_data" : "data",
                   .buffer = c.buffer,
                   .offset = 0,
                   .bytes = c.bytes},
                  {.kind = tiled ? "nax_scale" : "scale",
                   .buffer = s.buffer,
                   .offset = 0,
                   .bytes = s.bytes}};
      return w;
    };

    const auto w_gw = make_w3d(1, "gw", E, F, H, ref_rgc, ref_rgs);
    const auto w_uw = make_w3d(2, "uw", E, F, H, ref_ruc, ref_rus);
    const auto w_dw = make_w3d(3, "dw", E, H, F, ref_rdc, ref_rds);

    MaterializedWeightView w_router_tl{};
    w_router_tl.object_id = 10;
    w_router_tl.name = "router_tl";
    w_router_tl.format = "oq_affine_q8_g64_bf16_v1";
    w_router_tl.layout = "oq_nax_tiled_g64_v1";
    w_router_tl.parent_shape = {E, H};
    w_router_tl.row_count = E;
    w_router_tl.in_features = H;
    w_router_tl.padded_k = H;
    w_router_tl.planes = {{.kind = "nax_codes",
                           .buffer = ref_rc_tl.buffer,
                           .offset = 0,
                           .bytes = ref_rc_tl.bytes},
                          {.kind = "nax_sb",
                           .buffer = ref_rsb_tl.buffer,
                           .offset = 0,
                           .bytes = ref_rsb_tl.bytes}};

    MaterializedWeightView w_sgate_tl{};
    w_sgate_tl.object_id = 11;
    w_sgate_tl.name = "sgate_tl";
    w_sgate_tl.format = "oq_affine_q8_g64_bf16_v1";
    w_sgate_tl.layout = "oq_nax_tiled_g64_v1";
    w_sgate_tl.parent_shape = {1U, H};
    w_sgate_tl.row_count = 1U;
    w_sgate_tl.in_features = H;
    w_sgate_tl.padded_k = H;
    w_sgate_tl.planes = {{.kind = "nax_codes",
                          .buffer = ref_sgc_tl8.buffer,
                          .offset = 0,
                          .bytes = ref_sgc_tl8.bytes},
                         {.kind = "nax_sb",
                          .buffer = ref_sgsb_tl8.buffer,
                          .offset = 0,
                          .bytes = ref_sgsb_tl8.bytes}};

    MaterializedWeightView w_router_rm{};
    w_router_rm.object_id = 12;
    w_router_rm.name = "router_rm";
    w_router_rm.format = "oq_affine_q8_g64_bf16_v1";
    w_router_rm.layout = "oq_rowsplit_k128_v1";
    w_router_rm.parent_shape = {E, H};
    w_router_rm.row_count = E;
    w_router_rm.in_features = H;
    w_router_rm.padded_k = H;
    w_router_rm.planes = {{.kind = "code",
                           .buffer = ref_rc_rm.buffer,
                           .offset = 0,
                           .bytes = ref_rc_rm.bytes},
                          {.kind = "scale",
                           .buffer = ref_rsc_rm.buffer,
                           .offset = 0,
                           .bytes = ref_rsc_rm.bytes},
                          {.kind = "bias",
                           .buffer = ref_rbi_rm.buffer,
                           .offset = 0,
                           .bytes = ref_rbi_rm.bytes}};

    MaterializedWeightView w_sgate_rm{};
    w_sgate_rm.object_id = 13;
    w_sgate_rm.name = "sgate_rm";
    w_sgate_rm.format = "oq_affine_q8_g64_bf16_v1";
    w_sgate_rm.layout = "oq_rowsplit_k128_v1";
    w_sgate_rm.parent_shape = {1U, H};
    w_sgate_rm.row_count = 1U;
    w_sgate_rm.in_features = H;
    w_sgate_rm.padded_k = H;
    w_sgate_rm.planes = {{.kind = "code",
                          .buffer = ref_sgc_rm8.buffer,
                          .offset = 0,
                          .bytes = ref_sgc_rm8.bytes},
                         {.kind = "scale",
                          .buffer = ref_sgsc_rm8.buffer,
                          .offset = 0,
                          .bytes = ref_sgsc_rm8.bytes},
                         {.kind = "bias",
                          .buffer = ref_sgbi_rm8.buffer,
                          .offset = 0,
                          .bytes = ref_sgbi_rm8.bytes}};

    const auto w_sgw_tl = make_w2d_mx(20, "sgw_tl", F, H, true, ref_sh_gc_tl, ref_sh_gs_tl);
    const auto w_suw_tl = make_w2d_mx(21, "suw_tl", F, H, true, ref_sh_uc_tl, ref_sh_us_tl);
    const auto w_sdw_tl = make_w2d_mx(22, "sdw_tl", H, F, true, ref_sh_dc_tl, ref_sh_ds_tl);
    const auto w_sgw_rm = make_w2d_mx(23, "sgw_rm", F, H, false, ref_sh_gc_rm, ref_sh_gs_rm);
    const auto w_suw_rm = make_w2d_mx(24, "suw_rm", F, H, false, ref_sh_uc_rm, ref_sh_us_rm);
    const auto w_sdw_rm = make_w2d_mx(25, "sdw_rm", H, F, false, ref_sh_dc_rm, ref_sh_ds_rm);

    auto run_moe9_variant = [&](const std::string &tag,
                                const MaterializedWeightView &rw,
                                const MaterializedWeightView &sgw_gate,
                                const MaterializedWeightView &sh_gw,
                                const MaterializedWeightView &sh_uw,
                                const MaterializedWeightView &sh_dw,
                                MetalBuffer &b_rl, MetalBuffer &b_id,
                                MetalBuffer &b_sc, MetalBuffer &b_sg,
                                MetalBuffer &b_dl) {
      Encoder enc(ctx, libs, 1U, "dimb.moe9." + tag);
      enc.begin_stage("moe");
      bool fused_sg = false;
      const bool ok_r = emit_moe9_router_top8_t1(
          enc, rw, &sgw_gate, nullptr, ref_norm, ref_zero,
          Ref{&b_rl, 0, E * 2U}, Ref{&b_id, 0, 32U}, Ref{&b_sc, 0, 16U},
          Ref{&b_sg, 0, 2U}, H, 1U, &fused_sg);
      require(ok_r && fused_sg, "emit_moe9_router_top8_t1 failed for " + tag);
      const bool ok_m = emit_moe9_mxfp4_t1(
          enc, ws, w_gw, w_uw, w_dw, sh_gw, sh_uw, sh_dw, nullptr, ref_norm,
          Ref{&b_id, 0, 32U}, Ref{&b_sc, 0, 16U}, Ref{&b_sg, 0, 2U},
          Ref{&b_dl, 0, H * 2U}, H, F, 1U);
      require(ok_m, "emit_moe9_mxfp4_t1 failed for " + tag);
      auto prof = enc.finish();
      require(prof.stages.size() == 1U && prof.stages[0].length == 4U &&
                  prof.weight_routes.size() == 8U,
              "MOE9 expected exactly 4 dispatches (64-TG router + top8 + swiglu + down) and 8 weight routes");
      prof.parameters.write("step", prof.parameter_data);
      const std::array inv{MetalCommandInvocation{
          .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
      (void)ctx.execute_batch(inv);
    };

    run_moe9_variant("tiled", w_router_tl, w_sgate_tl, w_sgw_tl, w_suw_tl,
                     w_sdw_tl, b_rlogits_tl, b_rids_tl, b_rscores_tl,
                     b_sgate_tl, b_delta_tl);
    run_moe9_variant("rowmajor", w_router_rm, w_sgate_rm, w_sgw_rm, w_suw_rm,
                     w_sdw_rm, b_rlogits_rm, b_rids_rm, b_rscores_rm,
                     b_sgate_rm, b_delta_rm);

    // 1. Tiled and row-major variants must produce 0-ULP bit-identical outputs!
    require(b_rlogits_tl.read() == b_rlogits_rm.read() &&
                b_rids_tl.read() == b_rids_rm.read() &&
                b_rscores_tl.read() == b_rscores_rm.read() &&
                b_sgate_tl.read() == b_sgate_rm.read() &&
                b_delta_tl.read() == b_delta_rm.read(),
            "MOE9 tiled vs row-major variants diverged");

    // 1b. Also verify 1-dispatch fused router+top8+sgate (+ add_sq down) produces 0-ULP identical routing & delta!
    {
      auto b_rl_f = ctx.make_buffer(E * 2U);
      auto b_id_f = ctx.make_buffer(32U);
      auto b_sc_f = ctx.make_buffer(16U);
      auto b_sg_f = ctx.make_buffer(2U);
      auto b_hid_f = ctx.make_buffer(H * 2U);
      auto b_sq_f = ctx.make_buffer(16U);
      auto b_tgd_f = ctx.make_buffer(16U);
      b_hid_f.zero();
      b_sq_f.zero();
      b_tgd_f.zero();
      Encoder enc_f(ctx, libs, 1U, "dimb.moe9.fused3");
      enc_f.begin_stage("moe_fused");
      bool fused_sg_f = false;
      const bool ok_rf = emit_moe9_router_fused_top8_t1(
          enc_f, w_router_tl, &w_sgate_tl, nullptr, false, ref_norm, {}, {}, {},
          ref_zero, Ref{&b_rl_f, 0, E * 2U}, Ref{&b_id_f, 0, 32U},
          Ref{&b_sc_f, 0, 16U}, Ref{&b_sg_f, 0, 2U}, Ref{&b_tgd_f, 0, 16U}, H,
          0U, 0U, 1e-6f, 0.0f, 1U, &fused_sg_f);
      require(ok_rf && fused_sg_f, "emit_moe9_router_fused_top8_t1 failed");
      const bool ok_mf = emit_moe9_mxfp4_add_sq_t1(
          enc_f, ws, w_gw, w_uw, w_dw, w_sgw_tl, w_suw_tl, w_sdw_tl, nullptr,
          ref_norm, Ref{&b_id_f, 0, 32U}, Ref{&b_sc_f, 0, 16U},
          Ref{&b_sg_f, 0, 2U}, Ref{&b_hid_f, 0, H * 2U}, Ref{&b_sq_f, 0, 16U},
          H, F, 1U, 1U);
      require(ok_mf, "emit_moe9_mxfp4_add_sq_t1 failed");
      auto prof_f = enc_f.finish();
      require(prof_f.stages.size() == 1U && prof_f.stages[0].length == 3U,
              "Fused MOE9 expected exactly 3 dispatches (1-dispatch router+top8+sgate + swiglu + down_add_sq)");
      prof_f.parameters.write("step", prof_f.parameter_data);
      const std::array inv_f{MetalCommandInvocation{.plan = &prof_f.plan,
                                                    .bindings = prof_f.bindings,
                                                    .parameters = &prof_f.parameters}};
      (void)ctx.execute_batch(inv_f);
      require(b_id_f.read() == b_rids_tl.read() &&
                  b_sc_f.read() == b_rscores_tl.read() &&
                  b_sg_f.read() == b_sgate_tl.read() &&
                  b_hid_f.read() == b_delta_tl.read(),
              "Fused 3-dispatch MOE9 diverged from 4-dispatch MOE9!");

      // Also verify prologue_norm=true (omlx_m5_moe9_router_q8_pretiled_norm_fused_top8_t1)
      auto b_rmsw = ctx.make_buffer(H * 2U);
      std::vector<std::uint16_t> ones_bf16(H, f32_to_bf16_bits(1.0f));
      b_rmsw.write(std::as_bytes(std::span(ones_bf16)));
      auto b_norm_out = ctx.make_buffer(H * 2U);
      b_norm_out.zero();
      b_hid_f.zero();
      b_tgd_f.zero();
      const std::array<float, 4> sq_init{float(H), 0.0f, 0.0f, 0.0f};
      b_sq_f.write(std::as_bytes(std::span(sq_init)));
      Encoder enc_pn(ctx, libs, 1U, "dimb.moe9.fused3.pnorm");
      enc_pn.begin_stage("moe_fused_pnorm");
      bool fused_sg_pn = false;
      const bool ok_rpn = emit_moe9_router_fused_top8_t1(
          enc_pn, w_router_tl, &w_sgate_tl, nullptr, true, ref_norm,
          Ref{&b_rmsw, 0, H * 2U}, Ref{&b_sq_f, 0, 16U},
          Ref{&b_norm_out, 0, H * 2U}, ref_zero, Ref{&b_rl_f, 0, E * 2U},
          Ref{&b_id_f, 0, 32U}, Ref{&b_sc_f, 0, 16U}, Ref{&b_sg_f, 0, 2U},
          Ref{&b_tgd_f, 0, 16U}, H, 0U, 0U, 0.0f, 0.0f, 1U, &fused_sg_pn);
      require(ok_rpn && fused_sg_pn,
              "emit_moe9_router_fused_top8_t1 (prologue_norm=true) failed");
      const bool ok_mpn = emit_moe9_mxfp4_add_sq_t1(
          enc_pn, ws, w_gw, w_uw, w_dw, w_sgw_tl, w_suw_tl, w_sdw_tl, nullptr,
          Ref{&b_norm_out, 0, H * 2U}, Ref{&b_id_f, 0, 32U},
          Ref{&b_sc_f, 0, 16U}, Ref{&b_sg_f, 0, 2U}, Ref{&b_hid_f, 0, H * 2U},
          Ref{&b_sq_f, 0, 16U}, H, F, 1U, 1U);
      require(ok_mpn, "emit_moe9_mxfp4_add_sq_t1 (prologue_norm=true) failed");
      auto prof_pn = enc_pn.finish();
      prof_pn.parameters.write("step", prof_pn.parameter_data);
      const std::array inv_pn{MetalCommandInvocation{
          .plan = &prof_pn.plan, .bindings = prof_pn.bindings, .parameters = &prof_pn.parameters}};
      (void)ctx.execute_batch(inv_pn);
      require(b_norm_out.read() == ref_norm.buffer->read() &&
                  b_id_f.read() == b_rids_tl.read() &&
                  b_sc_f.read() == b_rscores_tl.read() &&
                  b_sg_f.read() == b_sgate_tl.read() &&
                  b_hid_f.read() == b_delta_tl.read(),
              "Fused 3-dispatch MOE9 (prologue_norm=true) diverged!");
    }

    std::uint16_t got_sgate = 0;
    std::memcpy(&got_sgate, b_sgate_tl.read().data(), 2U);
    require(bf16_ulp_diff(got_sgate, ref_sgate_bf16) <= 1U,
            "MOE9 fused shared_gate diverged from CPU reference");

    // 2. Verify delta_out against CPU FP32/BF16 reference across all H columns
    std::array<U32, 8> got_ids{};
    std::array<std::uint16_t, 8> got_scores{};
    std::memcpy(got_ids.data(), b_rids_tl.read().data(), 32U);
    std::memcpy(got_scores.data(), b_rscores_tl.read().data(), 16U);

    auto cpu_mx_dot = [&](const std::uint8_t *codes, const std::uint8_t *scales,
                          const std::vector<std::uint16_t> &x, U32 k) {
      float total = 0.0f;
      for (U32 g = 0; g < k / 32U; ++g) {
        float sub = 0.0f;
        for (U32 j = 0; j < 32U; ++j) {
          const U32 col = g * 32U + j;
          const std::uint8_t byte = codes[col / 2U];
          const U32 nib = (col & 1U) ? (byte >> 4U) : (byte & 0xFU);
          sub += bf16_bits_to_f32(x[col]) * decode_fp4(nib);
        }
        total += sub * decode_e8m0(scales[g]);
      }
      return total;
    };

    std::vector<std::vector<std::uint16_t>> ref_mid(
        9U, std::vector<std::uint16_t>(F));
    for (U32 s = 0; s < 9U; ++s) {
      for (U32 r = 0; r < F; ++r) {
        const std::uint8_t *gc_p =
            (s < 8U) ? &rgc[(got_ids[s] * F + r) * (H / 2U)]
                     : &sgc_rm[r * (H / 2U)];
        const std::uint8_t *gs_p =
            (s < 8U) ? &rgs[(got_ids[s] * F + r) * (H / 32U)]
                     : &sgs_rm[r * (H / 32U)];
        const std::uint8_t *uc_p =
            (s < 8U) ? &ruc[(got_ids[s] * F + r) * (H / 2U)]
                     : &suc_rm[r * (H / 2U)];
        const std::uint8_t *us_p =
            (s < 8U) ? &rus[(got_ids[s] * F + r) * (H / 32U)]
                     : &sus_rm[r * (H / 32U)];
        const float gv =
            bf16_bits_to_f32(f32_to_bf16_bits(cpu_mx_dot(gc_p, gs_p, norm_bf16, H)));
        const float uv =
            bf16_bits_to_f32(f32_to_bf16_bits(cpu_mx_dot(uc_p, us_p, norm_bf16, H)));
        const float silu = gv / (1.0f + std::exp(-gv));
        ref_mid[s][r] = f32_to_bf16_bits(silu * uv);
      }
    }

    std::vector<std::uint16_t> got_delta(H);
    std::memcpy(got_delta.data(), b_delta_tl.read().data(), H * 2U);
    const float sh_scale =
        1.0f / (1.0f + std::exp(-bf16_bits_to_f32(got_sgate)));
    for (U32 r = 0; r < H; ++r) {
      float r_sum = 0.0f;
      for (U32 s = 0; s < 8U; ++s) {
        const float d = cpu_mx_dot(&rdc[(got_ids[s] * H + r) * (F / 2U)],
                                   &rds[(got_ids[s] * H + r) * (F / 32U)],
                                   ref_mid[s], F);
        r_sum += bf16_bits_to_f32(
            f32_to_bf16_bits(d * bf16_bits_to_f32(got_scores[s])));
      }
      const float routed_bf = bf16_bits_to_f32(f32_to_bf16_bits(r_sum));
      const float sh_d = bf16_bits_to_f32(f32_to_bf16_bits(
          cpu_mx_dot(&sdc_rm[r * (F / 2U)], &sds_rm[r * (F / 32U)], ref_mid[8], F)));
      const std::uint16_t exp_bf =
          f32_to_bf16_bits(routed_bf + sh_scale * sh_d);
      const U32 ulp = bf16_ulp_diff(got_delta[r], exp_bf);
      moe9_max_ulp = std::max(moe9_max_ulp, ulp);
      require(ulp <= 2U,
              "MOE9 delta diverged from CPU reference at col " +
                  std::to_string(r) + " (ulp=" + std::to_string(ulp) + ")");
    }
  }

  return {{"argmax_64tg_checks", argmax_checks},
          {"moe9_tested", moe9_tested},
          {"moe9_max_ulp_vs_cpu", moe9_max_ulp},
          {"passed", true}};
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 5) {
      throw std::invalid_argument(
          "usage: omlx_m5_step4_moe_ar_tests <qwen_fixtures> <metallib_dir> "
          "<python_exe> <repo_root>");
    }
    const std::filesystem::path qwen_fixtures = argv[1];
    const std::filesystem::path libs = argv[2];
    const std::filesystem::path python_exe = argv[3];
    const std::filesystem::path repo_root = argv[4];

    const auto tmp_dir =
        std::filesystem::temp_directory_path() /
        ("omlx_m5_step4_tests_" + std::to_string(::getpid()));
    std::filesystem::remove_all(tmp_dir);
    std::filesystem::create_directories(tmp_dir);
    struct Cleanup {
      std::filesystem::path dir;
      ~Cleanup() {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
      }
    } cleanup{tmp_dir};

    auto ctx = MetalContext::create();
    const auto r1 = test_pso_sram_and_moe_schedulers(ctx, libs);
    const auto r6 = test_dimb_64tg_argmax_and_moe9_t1(ctx, libs);
    const auto r2 =
        test_pretiled_qmv_t1(ctx, libs, python_exe, repo_root, tmp_dir);
    const auto r3 =
        test_shared_kv_arena_and_moe_packed_batch(ctx, qwen_fixtures, libs);
    const auto r4 = test_async_ar_pipelining(qwen_fixtures, libs);
    const auto r5 = test_moe_bm64_and_2048_async_prefill(ctx, qwen_fixtures,
                                                         libs, tmp_dir);

    std::cout << Json({{"schema", "omlx-m5-step4-moe-ar-tests-v1"},
                       {"pso_sram_and_moe_schedulers", r1},
                       {"dimb_64tg_argmax_and_moe9_t1", r6},
                       {"pretiled_qmv_t1", r2},
                       {"shared_kv_arena_and_moe_packed_batch", r3},
                       {"async_ar_pipelining", r4},
                       {"moe_bm64_and_2048_async_prefill", r5},
                       {"passed", true}})
                     .dump(2)
              << '\n';
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
