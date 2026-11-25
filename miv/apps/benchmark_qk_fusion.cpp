#include "../src/forward_plan.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/model_config.h"
#include "omlx_m5/quantization.h"

using namespace omlx::m5;
using namespace omlx::m5::detail::forward;
namespace {
float represented(std::uint16_t code) {
  return std::bit_cast<float>(std::uint32_t(code) << 16);
}
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return (values[values.size() / 2] + values[(values.size() - 1) / 2]) / 2;
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 5)
      throw std::invalid_argument(
          "benchmark_qk_fusion CONFIG LIBRARIES T REPEATS");
    const auto config = load_qwen_model_config(argv[1]);
    const std::filesystem::path libraries = argv[2];
    const U32 tokens = U32(std::stoul(argv[3]));
    const U32 repeats = U32(std::stoul(argv[4]));
    if ((tokens != 1 && tokens != 8 && tokens != 128) || repeats < 5)
      throw std::invalid_argument("q/k fusion diagnostic T or repeats invalid");
    const U32 qh = U32(config.num_attention_heads),
              kvh = U32(config.num_key_value_heads), hd = U32(config.head_dim);
    const U32 query_stride = 2 * qh * hd, key_stride = kvh * hd;
    auto context = MetalContext::create();
    std::vector<std::uint16_t> query(U64(tokens) * query_stride);
    std::vector<std::uint16_t> key(U64(tokens) * key_stride);
    for (U32 row = 0; row < tokens; ++row) {
      for (U32 feature = 0; feature < query_stride; ++feature)
        query[U64(row) * query_stride + feature] = float_to_bf16(
            float(std::int32_t((feature * 17 + row * 23) % 257) - 128) / 512);
      for (U32 feature = 0; feature < key_stride; ++feature)
        key[U64(row) * key_stride + feature] = float_to_bf16(
            float(std::int32_t((feature * 13 + row * 31) % 257) - 128) / 512);
    }
    std::vector<std::uint16_t> q_norm(hd), k_norm(hd);
    for (U32 feature = 0; feature < hd; ++feature) {
      q_norm[feature] =
          float_to_bf16(1.0f + float(int(feature % 31) - 15) / 256);
      k_norm[feature] =
          float_to_bf16(1.0f + float(int(feature % 29) - 14) / 256);
    }
    auto fused_q = context.upload(std::as_bytes(std::span(query)));
    auto fused_k = context.upload(std::as_bytes(std::span(key)));
    auto unfused_q = context.upload(std::as_bytes(std::span(query)));
    auto unfused_k = context.upload(std::as_bytes(std::span(key)));
    auto q_weight = context.upload(std::as_bytes(std::span(q_norm)));
    auto k_weight = context.upload(std::as_bytes(std::span(k_norm)));
    const U32 rotary = U32(std::llround(hd * config.partial_rotary_factor));
    const auto params =
        words({qh, kvh, hd, rotary, tokens, query_stride, 2 * hd, key_stride,
               hd, 1024, word(float(config.rms_norm_eps)),
               word(float(config.rope_theta)), 0});
    const auto ref = [](const MetalBuffer &b) -> Ref {
      return {&b, 0, b.size()};
    };
    auto build = [&](bool fused) {
      Encoder e(context, libraries, tokens,
                fused ? "p06.qk.fused" : "p06.qk.unfused");
      const MetalDispatchSize grid{.width = 32,
                                   .height = U64(tokens) * (qh + kvh)};
      if (fused) {
        auto &d =
            e.emit("attention_postprocess", "omlx_m5_attention_qk_norm_rope",
                   grid, {.width = 32},
                   {ref(fused_q), ref(fused_k), ref(q_weight), ref(k_weight)});
        e.dynamic(d, 4, params);
      } else {
        auto &norm = e.emit(
            "attention_postprocess", "omlx_m5_attention_qk_norm_only", grid,
            {.width = 32},
            {ref(unfused_q), ref(unfused_k), ref(q_weight), ref(k_weight)});
        e.dynamic(norm, 4, params);
        auto &rope =
            e.emit("attention_postprocess", "omlx_m5_attention_qk_rope_only",
                   grid, {.width = 32}, {ref(unfused_q), ref(unfused_k)});
        e.dynamic(rope, 2, params);
      }
      return e.finish();
    };
    auto fused_profile = build(true), unfused_profile = build(false);
    auto run = [&](Profile &p) {
      const std::array calls = {
          MetalCommandInvocation{.plan = &p.plan,
                                 .bindings = p.bindings,
                                 .parameters = &p.parameters}};
      return context.execute_batch(calls);
    };
    const auto reset = [&](bool fused) {
      (fused ? fused_q : unfused_q).write(std::as_bytes(std::span(query)));
      (fused ? fused_k : unfused_k).write(std::as_bytes(std::span(key)));
    };
    std::vector<double> fused_gpu, unfused_gpu, fused_wall, unfused_wall;
    for (U32 iteration = 0; iteration < repeats + 3; ++iteration) {
      MetalSubmission a, b;
      if (iteration % 2) {
        reset(false);
        b = run(unfused_profile);
        reset(true);
        a = run(fused_profile);
      } else {
        reset(true);
        a = run(fused_profile);
        reset(false);
        b = run(unfused_profile);
      }
      if (iteration >= 3) {
        fused_gpu.push_back(a.timing.gpu_ms);
        unfused_gpu.push_back(b.timing.gpu_ms);
        fused_wall.push_back(a.timing.wall_ms);
        unfused_wall.push_back(b.timing.wall_ms);
      }
    }
    const auto q_a = fused_q.read(), q_b = unfused_q.read();
    const auto k_a = fused_k.read(), k_b = unfused_k.read();
    double maximum = 0, squared = 0, scale = 0;
    bool gate_unchanged = true;
    for (U32 row = 0; row < tokens; ++row)
      for (U32 head = 0; head < qh; ++head)
        for (U32 feature = 0; feature < hd; ++feature) {
          const U64 index = U64(row) * query_stride + head * 2 * hd + feature;
          std::uint16_t old_bits, next_bits;
          std::memcpy(&old_bits, q_a.data() + index * 2, 2);
          std::memcpy(&next_bits, q_b.data() + index * 2, 2);
          const auto old = represented(old_bits), next = represented(next_bits);
          maximum = std::max(maximum, std::abs(double(old) - next));
          squared += (double(old) - next) * (double(old) - next);
          scale += double(old) * old;
          const U64 gate = index + hd;
          gate_unchanged &=
              std::equal(q_a.begin() + gate * 2, q_a.begin() + gate * 2 + 2,
                         q_b.begin() + gate * 2) &&
              std::equal(q_a.begin() + gate * 2, q_a.begin() + gate * 2 + 2,
                         std::as_bytes(std::span(query)).begin() + gate * 2);
        }
    for (U64 index = 0; index < key.size(); ++index) {
      std::uint16_t old_bits, next_bits;
      std::memcpy(&old_bits, k_a.data() + index * 2, 2);
      std::memcpy(&next_bits, k_b.data() + index * 2, 2);
      const auto old = represented(old_bits), next = represented(next_bits);
      maximum = std::max(maximum, std::abs(double(old) - next));
      squared += (double(old) - next) * (double(old) - next);
      scale += double(old) * old;
    }
    const double relative = std::sqrt(squared / std::max(scale, 1e-12));
    if (!gate_unchanged || maximum > 0.03125 || relative > 0.003)
      throw std::runtime_error(
          "P06 q/k two-pass numeric or gate mismatch: max=" +
          std::to_string(maximum) + " relative=" + std::to_string(relative) +
          " gate=" + std::to_string(gate_unchanged));
    std::cout
        << nlohmann::json(
               {{"schema", "omlx-m5-p06-qk-fusion-v1"},
                {"scope", "real target head geometry, synthetic q/k/norm"},
                {"device", context.capabilities().device_name},
                {"tokens", tokens},
                {"query_heads", qh},
                {"kv_heads", kvh},
                {"head_dim", hd},
                {"fused_gpu_ms", fused_gpu},
                {"unfused_gpu_ms", unfused_gpu},
                {"fused_wall_ms", fused_wall},
                {"unfused_wall_ms", unfused_wall},
                {"fused_gpu_median_ms", median(fused_gpu)},
                {"unfused_gpu_median_ms", median(unfused_gpu)},
                {"max_abs", maximum},
                {"relative_l2", relative},
                {"gate_unchanged", gate_unchanged},
                {"fused_dispatches", fused_profile.plan.dispatch_count()},
                {"unfused_dispatches", unfused_profile.plan.dispatch_count()}})
               .dump(2)
        << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "P06 q/k fusion failed: " << error.what() << '\n';
    return 1;
  }
}
