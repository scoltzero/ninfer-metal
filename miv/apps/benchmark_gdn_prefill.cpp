#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../src/forward_plan.h"
#include "omlx_m5/artifact.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/quantization.h"

namespace {
using namespace omlx::m5;
using namespace omlx::m5::detail::forward;
using Json = nlohmann::json;

Ref ref(const MetalBuffer &buffer) { return {&buffer, 0, buffer.size()}; }

const ArtifactObject &module_object(const ArtifactModel &artifact,
                                    const std::string &name) {
  try {
    return artifact.object(name);
  } catch (const std::out_of_range &) {
    return artifact.object(name + ".weight");
  }
}

MaterializedWeightView
upload_weight(const ArtifactModel &artifact, const ArtifactObject &object,
              const MetalContext &context,
              std::vector<std::unique_ptr<MetalBuffer>> &buffers) {
  if (object.shape.size() != 2)
    throw std::invalid_argument("GDN projection is not a matrix");
  MaterializedWeightView weight{.object_id = object.id,
                                .component =
                                    classify_model_component(object.name),
                                .name = object.name,
                                .format = object.format,
                                .layout = object.layout,
                                .parent_shape = object.shape,
                                .padded_k = object.padded_k,
                                .row_count = object.shape[0],
                                .in_features = object.shape[1]};
  for (const auto &plane : object.planes) {
    buffers.push_back(std::make_unique<MetalBuffer>(
        context.upload(artifact.plane_bytes(object, plane))));
    weight.planes.push_back({.kind = plane.kind,
                             .buffer = buffers.back().get(),
                             .segment_index = U32(buffers.size() - 1),
                             .offset = 0,
                             .bytes = plane.bytes,
                             .row_stride = plane.bytes / object.shape[0]});
  }
  return weight;
}

MetalBuffer direct(const ArtifactModel &artifact, std::string name,
                   const MetalContext &context) {
  const auto &object = artifact.object(name);
  if (object.format != "direct_bf16_v1" || object.planes.size() != 1)
    throw std::invalid_argument("GDN direct weight format mismatch: " + name);
  return context.upload(artifact.plane_bytes(object, object.planes.front()));
}

std::vector<std::uint16_t> words_for(U32 rows, U32 width, U32 a, U32 b) {
  std::vector<std::uint16_t> result(U64(rows) * width);
  for (U32 row = 0; row < rows; ++row)
    for (U32 col = 0; col < width; ++col)
      result[U64(row) * width + col] = float_to_bf16(
          float(std::int32_t((U64(col) * a + U64(row) * b) % 257) - 128) /
          256.0f);
  return result;
}

template <class T> std::vector<std::byte> as_bytes(const std::vector<T> &data) {
  auto view = std::as_bytes(std::span(data));
  return {view.begin(), view.end()};
}

double median(std::vector<double> samples) {
  std::ranges::sort(samples);
  return (samples[(samples.size() - 1) / 2] + samples[samples.size() / 2]) / 2;
}

Json difference(const std::vector<std::byte> &baseline,
                const std::vector<std::byte> &candidate, bool bf16) {
  if (baseline.size() != candidate.size() || baseline.size() % (bf16 ? 2 : 4))
    throw std::runtime_error("GDN comparison size mismatch");
  double error2 = 0, reference2 = 0, maximum = 0;
  const U64 count = baseline.size() / (bf16 ? 2 : 4);
  for (U64 i = 0; i < count; ++i) {
    float lhs = 0, rhs = 0;
    if (bf16) {
      std::uint16_t a = 0, b = 0;
      std::memcpy(&a, baseline.data() + i * 2, 2);
      std::memcpy(&b, candidate.data() + i * 2, 2);
      lhs = bf16_to_float(a);
      rhs = bf16_to_float(b);
    } else {
      std::memcpy(&lhs, baseline.data() + i * 4, 4);
      std::memcpy(&rhs, candidate.data() + i * 4, 4);
    }
    if (!std::isfinite(lhs) || !std::isfinite(rhs))
      throw std::runtime_error("GDN nonfinite output or state");
    const double delta = double(lhs) - rhs;
    error2 += delta * delta;
    reference2 += double(lhs) * lhs;
    maximum = std::max(maximum, std::abs(delta));
  }
  return {{"relative_l2", std::sqrt(error2 / std::max(reference2, 1e-12))},
          {"max_abs", maximum},
          {"bit_exact", baseline == candidate},
          {"elements", count}};
}

struct Variant {
  std::string name;
  MetalBuffer history, convolved, state_in, state_out, recurrent, output;
  MetalBuffer q, k, v, g, beta, y, qk_inverse, gate_coeff;
  Profile profile, full_profile;
  MetalBuffer full_qkv, full_z, full_a, full_b, full_sums, full_delta;
};

Json measure(const MetalContext &context, const ArtifactModel &artifact,
             const QwenModelConfig &config, const std::filesystem::path &libs,
             std::string_view prefix, U32 tokens, U32 repeats,
             bool include_full_slice, bool p07) {
  const U32 hk = U32(config.linear_num_key_heads);
  const U32 hv = U32(config.linear_num_value_heads);
  const U32 kd = U32(config.linear_key_head_dim);
  const U32 vd = U32(config.linear_value_head_dim);
  const U32 taps = U32(config.linear_conv_kernel_dim);
  if (kd != 128 || vd != 128 || taps < 2 || hv % hk)
    throw std::invalid_argument(
        "GDN candidate requires 128-dim target geometry");
  const U32 conv_width = 2 * hk * kd + hv * vd;
  const U64 state_values = U64(hv) * vd * kd;
  const U64 output_values = U64(tokens) * hv * vd;
  const auto qkv =
      context.upload(as_bytes(words_for(tokens, conv_width, 11, 17)));
  const auto z = context.upload(as_bytes(words_for(tokens, hv * vd, 13, 19)));
  const auto a = context.upload(as_bytes(words_for(tokens, hv, 3, 5)));
  const auto b = context.upload(as_bytes(words_for(tokens, hv, 7, 11)));
  const auto conv_weight =
      direct(artifact, std::string(prefix) + ".conv1d.weight", context);
  const auto a_log = direct(artifact, std::string(prefix) + ".A_log", context);
  const auto dt_bias =
      direct(artifact, std::string(prefix) + ".dt_bias", context);
  const auto norm_weight =
      direct(artifact, std::string(prefix) + ".norm.weight", context);
  std::vector<float> initial_state(state_values);
  for (U64 i = 0; i < state_values; ++i)
    initial_state[i] = float(std::int32_t((i * 13) % 47) - 23) / 16384.0f;
  const auto initial_state_bytes = as_bytes(initial_state);
  const auto initial_history_bytes =
      as_bytes(words_for(taps - 1, conv_width, 23, 31));
  const auto params = words({hk, hv, kd, vd, conv_width, taps, tokens,
                             std::bit_cast<U32>(float(config.rms_norm_eps))});
  const auto candidate_params = words({tokens, hk, hv, conv_width});
  std::vector<Variant> variants;
  variants.reserve(5);
  for (const char *name : {"recurrent", "share_qk", "share_gate",
                           "share_both", "blocked8"}) {
    auto &run = variants.emplace_back();
    run.name = name;
    run.history = context.make_buffer(initial_history_bytes.size());
    run.convolved = context.make_buffer(U64(tokens) * conv_width * 2);
    run.state_in = context.make_buffer(initial_state_bytes.size());
    run.recurrent = context.make_buffer(output_values * 2);
    run.output = context.make_buffer(output_values * 2);
    if (run.name != "recurrent" && run.name != "blocked8") {
      run.qk_inverse = context.make_buffer(U64(tokens) * hk * 2 * 4);
      run.gate_coeff = context.make_buffer(U64(tokens) * hv * 2 * 4);
    }
    if (run.name == "blocked8") {
      run.state_out = context.make_buffer(initial_state_bytes.size());
      run.q = context.make_buffer(U64(tokens) * hk * kd * 4);
      run.k = context.make_buffer(U64(tokens) * hk * kd * 4);
      run.v = context.make_buffer(output_values * 4);
      run.g = context.make_buffer(U64(tokens) * hv * 4);
      run.beta = context.make_buffer(U64(tokens) * hv * 4);
      run.y = context.make_buffer(output_values * 4);
    }
    Encoder encoder(context, libs, tokens,
                    "gdn.prefill." + run.name + ".t" + std::to_string(tokens));
    encoder.begin_stage("conv");
    auto &conv = encoder.emit(
        "gdn_layer", "omlx_m5_gdn_conv_silu", {.width = conv_width},
        {.width = 256},
        {ref(qkv), ref(conv_weight), ref(run.history), ref(run.convolved)});
    encoder.dynamic(conv, 4, params);
    if (run.name == "recurrent") {
      encoder.begin_stage("recurrent");
      auto &rec =
          encoder.emit("gdn_layer", "omlx_m5_gdn_recurrent",
                       {.width = 32, .height = U64(hv) * vd}, {.width = 32},
                       {ref(run.convolved), ref(a), ref(b), ref(a_log),
                        ref(dt_bias), ref(run.state_in), ref(run.recurrent)});
      encoder.dynamic(rec, 7, params);
    } else if (run.name != "blocked8") {
      if (run.name != "share_gate") {
        encoder.begin_stage("prepare_qk");
        auto &prepare = encoder.emit(
            "gdn_layer", "omlx_m5_gdn_prepare_qk_inverse",
            {.width = 32, .height = U64(tokens) * hk}, {.width = 32},
            {ref(run.convolved), ref(run.qk_inverse)});
        encoder.dynamic(prepare, 2, params);
      }
      if (run.name != "share_qk") {
        encoder.begin_stage("prepare_gate");
        auto &prepare = encoder.emit(
            "gdn_layer", "omlx_m5_gdn_prepare_gate",
            {.width = U64(tokens) * hv}, {.width = 256},
            {ref(a), ref(b), ref(a_log), ref(dt_bias), ref(run.gate_coeff)});
        encoder.dynamic(prepare, 5, params);
      }
      encoder.begin_stage("recurrent");
      const std::string kernel = "omlx_m5_gdn_recurrent_" + run.name;
      auto &rec = encoder.emit(
          "gdn_layer", kernel, {.width = 32, .height = U64(hv) * vd},
          {.width = 32},
          {ref(run.convolved), ref(a), ref(b), ref(a_log), ref(dt_bias),
           ref(run.state_in), ref(run.recurrent), ref(run.qk_inverse),
           ref(run.gate_coeff)});
      encoder.dynamic(rec, 9, params);
    } else {
      encoder.begin_stage("prepare");
      auto &prepare = encoder.emit(
          "gdn_candidate_io", "omlx_m5_gdn_prepare_candidates",
          {.width = 32, .height = U64(tokens) * hv}, {.width = 32},
          {ref(run.convolved), ref(a), ref(b), ref(a_log), ref(dt_bias),
           ref(run.q), ref(run.k), ref(run.v), ref(run.g), ref(run.beta)});
      encoder.dynamic(prepare, 10, candidate_params);
      encoder.begin_stage("blocked8");
      auto &block = encoder.emit(
          "gdn_prefill_candidates", "omlx_m5_gdn_blocked_prefill",
          {.width = U64(4) * 256, .height = hv}, {.width = 256},
          {ref(run.q), ref(run.k), ref(run.v), ref(run.g), ref(run.beta),
           ref(run.state_in), ref(run.y), ref(run.state_out)});
      encoder.dynamic(block, 8, candidate_params);
      encoder.begin_stage("cast");
      auto &cast =
          encoder.emit("gdn_candidate_io", "omlx_m5_gdn_candidate_output",
                       {.width = output_values}, {.width = 256},
                       {ref(run.y), ref(run.recurrent)});
      encoder.dynamic(cast, 2, candidate_params);
    }
    encoder.begin_stage("norm_gate");
    auto &norm = encoder.emit(
        "gdn_layer", "omlx_m5_gdn_norm_gate",
        {.width = 32, .height = U64(tokens) * hv}, {.width = 32},
        {ref(run.recurrent), ref(z), ref(norm_weight), ref(run.output)});
    encoder.dynamic(norm, 4, params);
    run.profile = encoder.finish();
  }
  const auto reset = [&](Variant &run) {
    run.state_in.write(initial_state_bytes);
    run.history.write(initial_history_bytes);
  };
  const auto execute = [&](Variant &run) {
    const std::array calls{
        MetalCommandInvocation{.plan = &run.profile.plan,
                               .bindings = run.profile.bindings,
                               .parameters = &run.profile.parameters}};
    return context.execute_batch(calls);
  };
  for (auto &run : variants) {
    reset(run);
    static_cast<void>(execute(run));
  }
  const auto baseline_output = variants[0].output.read();
  const auto baseline_state = variants[0].state_in.read();
  Json checks = Json::object();
  for (auto &run : variants) {
    if (run.name == "recurrent")
      continue;
    checks[run.name] = {
        {"output", difference(baseline_output, run.output.read(), true)},
        {"state", difference(baseline_state,
                             (run.name == "blocked8" ? run.state_out
                                                     : run.state_in).read(),
                             false)},
        {"conv_history_bit_exact",
         variants[0].history.read() == run.history.read()},
        {"convolved_bit_exact",
         variants[0].convolved.read() == run.convolved.read()}};
  }
  Json timings = Json::object();
  for (auto &run : variants) {
    std::vector<double> gpu, wall;
    for (U32 iteration = 0; iteration < repeats + 3; ++iteration) {
      reset(run);
      const auto submission = execute(run);
      if (iteration < 3)
        continue;
      gpu.push_back(submission.timing.gpu_ms);
      wall.push_back(submission.timing.wall_ms);
    }
    timings[run.name] = {{"gpu_ms", gpu},
                         {"wall_ms", wall},
                         {"median_gpu_ms", median(gpu)},
                         {"median_wall_ms", median(wall)},
                         {"dispatches", run.profile.plan.dispatch_count()}};
  }
  Json staging = Json::object();
  for (const std::string name : {"prepare_qk", "prepare_gate"}) {
    Encoder encoder(context, libs, tokens,
                    "gdn.staging." + name + ".t" + std::to_string(tokens));
    encoder.begin_stage(name);
    if (name == "prepare_qk") {
      auto &prepare = encoder.emit(
          "gdn_layer", "omlx_m5_gdn_prepare_qk_inverse",
          {.width = 32, .height = U64(tokens) * hk}, {.width = 32},
          {ref(variants[0].convolved), ref(variants[1].qk_inverse)});
      encoder.dynamic(prepare, 2, params);
    } else {
      auto &prepare = encoder.emit(
          "gdn_layer", "omlx_m5_gdn_prepare_gate",
          {.width = U64(tokens) * hv}, {.width = 256},
          {ref(a), ref(b), ref(a_log), ref(dt_bias),
           ref(variants[2].gate_coeff)});
      encoder.dynamic(prepare, 5, params);
    }
    auto profile = encoder.finish();
    std::vector<double> gpu, wall;
    for (U32 iteration = 0; iteration < repeats + 3; ++iteration) {
      const std::array calls{
          MetalCommandInvocation{.plan = &profile.plan,
                                 .bindings = profile.bindings,
                                 .parameters = &profile.parameters}};
      const auto submission = context.execute_batch(calls);
      if (iteration >= 3) {
        gpu.push_back(submission.timing.gpu_ms);
        wall.push_back(submission.timing.wall_ms);
      }
    }
    staging[name] = {{"gpu_ms", gpu},
                     {"wall_ms", wall},
                     {"median_gpu_ms", median(gpu)},
                     {"median_wall_ms", median(wall)},
                     {"scope", "one isolated prepare dispatch after conv; "
                               "command submission included; not additive to "
                               "full-chain time"}};
  }
  Json full_slice = nullptr;
  if (include_full_slice && (p07 || tokens != 129)) {
    const auto binding =
        [&](const std::string &name) -> const ArtifactObject & {
      const auto found = std::ranges::find_if(
          artifact.bindings(),
          [&](const ArtifactBinding &entry) { return entry.name == name; });
      if (found == artifact.bindings().end())
        throw std::out_of_range("GDN projection binding missing: " + name);
      return artifact.object(found->object_id);
    };
    std::vector<std::unique_ptr<MetalBuffer>> weight_buffers;
    const auto qkv_weight =
        upload_weight(artifact, binding(std::string(prefix) + ".in_proj_qkv"),
                      context, weight_buffers);
    const auto z_weight =
        upload_weight(artifact, binding(std::string(prefix) + ".in_proj_z"),
                      context, weight_buffers);
    const auto a_weight = upload_weight(
        artifact, module_object(artifact, std::string(prefix) + ".in_proj_a"),
        context, weight_buffers);
    const auto b_weight = upload_weight(
        artifact, module_object(artifact, std::string(prefix) + ".in_proj_b"),
        context, weight_buffers);
    const auto out_weight =
        upload_weight(artifact, binding(std::string(prefix) + ".out_proj"),
                      context, weight_buffers);
    const U32 hidden = U32(config.hidden_size);
    const U32 value_width = hv * vd;
    const auto input =
        context.upload(as_bytes(words_for(tokens, hidden, 19, 23)));
    auto zero = context.make_buffer(256);
    zero.write(std::vector<std::byte>(256));
    for (auto &run : variants) {
      run.full_qkv = context.make_buffer(U64(tokens) * conv_width * 2);
      run.full_z = context.make_buffer(U64(tokens) * value_width * 2);
      run.full_a = context.make_buffer(U64(tokens) * hv * 2);
      run.full_b = context.make_buffer(U64(tokens) * hv * 2);
      run.full_sums =
          context.make_buffer(U64(tokens) * std::max(hidden, value_width) * 4);
      run.full_delta = context.make_buffer(U64(tokens) * hidden * 2);
      Encoder encoder(context, libs, tokens,
                      "gdn.full." + run.name + ".t" + std::to_string(tokens));
      encoder.begin_stage("input_projections");
      linear(encoder, qkv_weight, ref(input), ref(run.full_qkv), conv_width,
             hidden, ref(run.full_sums), ref(zero));
      linear(encoder, z_weight, ref(input), ref(run.full_z), value_width,
             hidden, ref(run.full_sums), ref(zero));
      linear(encoder, a_weight, ref(input), ref(run.full_a), hv, hidden,
             ref(run.full_sums), ref(zero));
      linear(encoder, b_weight, ref(input), ref(run.full_b), hv, hidden,
             ref(run.full_sums), ref(zero));
      encoder.begin_stage("conv");
      auto &conv = encoder.emit("gdn_layer", "omlx_m5_gdn_conv_silu",
                                {.width = conv_width}, {.width = 256},
                                {ref(run.full_qkv), ref(conv_weight),
                                 ref(run.history), ref(run.convolved)});
      encoder.dynamic(conv, 4, params);
      if (run.name == "recurrent") {
        encoder.begin_stage("recurrent");
        auto &rec = encoder.emit(
            "gdn_layer", "omlx_m5_gdn_recurrent",
            {.width = 32, .height = U64(hv) * vd}, {.width = 32},
            {ref(run.convolved), ref(run.full_a), ref(run.full_b), ref(a_log),
             ref(dt_bias), ref(run.state_in), ref(run.recurrent)});
        encoder.dynamic(rec, 7, params);
    } else if (run.name != "blocked8") {
      if (run.name != "share_gate") {
        encoder.begin_stage("prepare_qk");
        auto &prepare = encoder.emit(
            "gdn_layer", "omlx_m5_gdn_prepare_qk_inverse",
            {.width = 32, .height = U64(tokens) * hk}, {.width = 32},
            {ref(run.convolved), ref(run.qk_inverse)});
        encoder.dynamic(prepare, 2, params);
      }
      if (run.name != "share_qk") {
        encoder.begin_stage("prepare_gate");
        auto &prepare = encoder.emit(
            "gdn_layer", "omlx_m5_gdn_prepare_gate",
            {.width = U64(tokens) * hv}, {.width = 256},
            {ref(run.full_a), ref(run.full_b), ref(a_log), ref(dt_bias), ref(run.gate_coeff)});
        encoder.dynamic(prepare, 5, params);
      }
      encoder.begin_stage("recurrent");
      const std::string kernel = "omlx_m5_gdn_recurrent_" + run.name;
      auto &rec = encoder.emit(
          "gdn_layer", kernel, {.width = 32, .height = U64(hv) * vd},
          {.width = 32},
          {ref(run.convolved), ref(run.full_a), ref(run.full_b), ref(a_log), ref(dt_bias),
           ref(run.state_in), ref(run.recurrent), ref(run.qk_inverse),
           ref(run.gate_coeff)});
      encoder.dynamic(rec, 9, params);
    } else {
      encoder.begin_stage("prepare");
        auto &prepare = encoder.emit(
            "gdn_candidate_io", "omlx_m5_gdn_prepare_candidates",
            {.width = 32, .height = U64(tokens) * hv}, {.width = 32},
            {ref(run.convolved), ref(run.full_a), ref(run.full_b), ref(a_log),
             ref(dt_bias), ref(run.q), ref(run.k), ref(run.v), ref(run.g),
             ref(run.beta)});
        encoder.dynamic(prepare, 10, candidate_params);
        encoder.begin_stage("blocked8");
        auto &block = encoder.emit(
            "gdn_prefill_candidates", "omlx_m5_gdn_blocked_prefill",
            {.width = U64(4) * 256, .height = hv}, {.width = 256},
            {ref(run.q), ref(run.k), ref(run.v), ref(run.g), ref(run.beta),
             ref(run.state_in), ref(run.y), ref(run.state_out)});
        encoder.dynamic(block, 8, candidate_params);
        encoder.begin_stage("cast");
        auto &cast =
            encoder.emit("gdn_candidate_io", "omlx_m5_gdn_candidate_output",
                         {.width = output_values}, {.width = 256},
                         {ref(run.y), ref(run.recurrent)});
        encoder.dynamic(cast, 2, candidate_params);
      }
      encoder.begin_stage("norm_gate");
      auto &norm =
          encoder.emit("gdn_layer", "omlx_m5_gdn_norm_gate",
                       {.width = 32, .height = U64(tokens) * hv}, {.width = 32},
                       {ref(run.recurrent), ref(run.full_z), ref(norm_weight),
                        ref(run.output)});
      encoder.dynamic(norm, 4, params);
      encoder.begin_stage("output_projection");
      linear(encoder, out_weight, ref(run.output), ref(run.full_delta), hidden,
             value_width, ref(run.full_sums), ref(zero));
      run.full_profile = encoder.finish();
    }
    const auto execute_full = [&](Variant &run) {
      const std::array calls{
          MetalCommandInvocation{.plan = &run.full_profile.plan,
                                 .bindings = run.full_profile.bindings,
                                 .parameters = &run.full_profile.parameters}};
      return context.execute_batch(calls);
    };
    for (auto &run : variants) {
      reset(run);
      static_cast<void>(execute_full(run));
    }
    const auto baseline_delta = variants[0].full_delta.read();
    const auto baseline_state = variants[0].state_in.read();
    Json candidates = Json::object();
    for (auto &run : variants) {
      if (run.name == "recurrent") continue;
      candidates[run.name] = {
          {"output", difference(baseline_delta, run.full_delta.read(), true)},
          {"state", difference(baseline_state,
                               (run.name == "blocked8" ? run.state_out
                                                       : run.state_in).read(),
                               false)},
          {"conv_history_bit_exact", variants[0].history.read() == run.history.read()},
          {"dispatches", run.full_profile.plan.dispatch_count()}};
    }
    std::vector<std::vector<double>> full_gpu(variants.size()),
                                     full_wall(variants.size());
    for (U32 iteration = 0; iteration < repeats + 3; ++iteration) {
      // Rotate execution order to prevent one candidate always following the
      // same previous profile or always seeing the warmest device state.
      for (U32 offset = 0; offset < variants.size(); ++offset) {
        const U32 selected = (iteration + offset) % U32(variants.size());
        auto &run = variants[selected];
        reset(run);
        const auto submission = execute_full(run);
        if (iteration >= 3) {
          full_gpu[selected].push_back(submission.timing.gpu_ms);
          full_wall[selected].push_back(submission.timing.wall_ms);
        }
      }
    }
    for (U32 i = 1; i < variants.size(); ++i) {
      auto &entry = candidates[variants[i].name];
      entry["gpu_ms"] = full_gpu[i];
      entry["wall_ms"] = full_wall[i];
      entry["median_gpu_ms"] = median(full_gpu[i]);
      entry["median_wall_ms"] = median(full_wall[i]);
    }
    full_slice = {
        {"scope", "four real quantized input projections + conv/recurrent/norm "
                  "gate + real output projection; synthetic BF16 normalized "
                  "hidden; no input RMSNorm or residual"},
        {"baseline_gpu_ms", full_gpu[0]},
        {"candidate_gpu_ms", full_gpu.back()},
        {"baseline_wall_ms", full_wall[0]},
        {"candidate_wall_ms", full_wall.back()},
        {"baseline_gpu_median_ms", median(full_gpu[0])},
        {"candidate_gpu_median_ms", median(full_gpu.back())},
        {"output", candidates["blocked8"]["output"]},
        {"state", candidates["blocked8"]["state"]},
        {"baseline_dispatches", variants[0].full_profile.plan.dispatch_count()},
        {"candidate_dispatches", variants.back().full_profile.plan.dispatch_count()},
        {"candidates", candidates}};
  }
  return {{"tokens", tokens},
          {"key_heads", hk},
          {"value_heads", hv},
          {"head_dim", kd},
          {"checks", checks},
          {"timings", timings},
          {"staging", staging},
          {"full_slice", full_slice},
          {"measured_scope",
           "GDN mixer conv+recurrent+norm gate only; quantized "
           "input and output projections excluded"}};
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 6 || argc > 8)
      throw std::invalid_argument(
          "usage: benchmark_gdn_prefill ARTIFACT CONFIG LIBRARIES REPEATS "
          "OUTPUT [--mixer-only] [--p07]");
    bool include_full_slice = true, p07 = false;
    for (int i = 6; i < argc; ++i) {
      const std::string_view option(argv[i]);
      if (option == "--mixer-only") include_full_slice = false;
      else if (option == "--p07") p07 = true;
      else throw std::invalid_argument("unknown GDN benchmark option");
    }
    const auto artifact = ArtifactModel::open(argv[1], false);
    const auto config = load_qwen_model_config(argv[2]);
    auto context = MetalContext::create();
    const U32 repeats = U32(std::stoul(argv[4]));
    if (repeats < 5)
      throw std::invalid_argument(
          "GDN benchmark requires at least five repeats");
    const auto first = std::ranges::find_if(artifact.bindings(),
                                            [](const ArtifactBinding &binding) {
                                              return binding.use == "gdn_qkv";
                                            });
    if (first == artifact.bindings().end() ||
        !first->name.ends_with(".in_proj_qkv"))
      throw std::invalid_argument("GDN projection binding absent");
    const auto prefix = first->name.substr(
        0, first->name.size() - std::string_view(".in_proj_qkv").size());
    Json records = Json::array();
    const std::vector<U32> shapes = p07
        ? std::vector<U32>{1, 8, 128, 129, 512, 1024}
        : std::vector<U32>{128, 129, 512, 1024};
    for (U32 tokens : shapes)
      records.push_back(measure(context, artifact, config, argv[3], prefix,
                                tokens, repeats, include_full_slice, p07));
    Json result = {{"schema", p07 ? "omlx-m5-gdn-prefill-candidates-v2"
                                    : "omlx-m5-gdn-prefill-candidates-v1"},
                   {"device", context.capabilities().device_name},
                   {"source_revision", artifact.summary().source_revision},
                   {"artifact", argv[1]},
                   {"config", argv[2]},
                   {"warmups", 3},
                   {"repeats", repeats},
                   {"records", records},
                   {"production_route_changed", false}};
    std::ofstream output(argv[5]);
    output << result.dump(2) << '\n';
    if (!output)
      throw std::runtime_error("unable to write GDN result");
    std::cout << argv[5] << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "GDN prefill benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
