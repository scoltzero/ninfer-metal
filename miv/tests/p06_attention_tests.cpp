#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/qwen_forward.h"

using namespace omlx::m5;
using Json = nlohmann::json;
bool exact_numeric=false;
namespace {
bool strict_two_pass_env() {
  const char *val = std::getenv("OMLX_M5_STRICT_TWO_PASS_ATTN");
  if (!val)
    return false;
  const std::string_view sv(val);
  return sv == "1" || sv == "true" || sv == "TRUE" || sv == "on" || sv == "ON";
}
struct ScopedEnv {
  std::string key;
  std::optional<std::string> prev;
  ScopedEnv(std::string k, const char *v) : key(std::move(k)) {
    if (const char *old = std::getenv(key.c_str()))
      prev = std::string(old);
    if (v)
      setenv(key.c_str(), v, 1);
    else
      unsetenv(key.c_str());
  }
  ~ScopedEnv() {
    if (prev)
      setenv(key.c_str(), prev->c_str(), 1);
    else
      unsetenv(key.c_str());
  }
};
void require(bool condition, const char *reason) {
  if (!condition)
    throw std::runtime_error(reason);
}
std::vector<std::uint32_t> tokens(std::size_t count, std::uint32_t offset = 0) {
  std::vector<std::uint32_t> result(count);
  for (std::size_t i = 0; i < count; ++i)
    result[i] = std::uint32_t((71 * (i + offset) + 256) % 257);
  return result;
}
Json compare(const QwenForwardResult &a, const QwenForwardResult &b) {
  if(exact_numeric && (a.logits!=b.logits || a.hidden!=b.hidden))
    throw std::runtime_error("P06-TF exact installed profiles changed output bits");
  if (a.logits.size() != b.logits.size() || a.hidden.size() != b.hidden.size())
    throw std::runtime_error("P06 full forward output shape differs");
  double squared = 0, scale = 0, maximum = 0;
  for (std::size_t i = 0; i < a.logits.size(); ++i) {
    if (!std::isfinite(a.logits[i]) || !std::isfinite(b.logits[i]))
      throw std::runtime_error("P06 nonfinite logits");
    const auto delta = double(a.logits[i]) - b.logits[i];
    maximum = std::max(maximum, std::abs(delta));
    squared += delta * delta;
    scale += double(a.logits[i]) * a.logits[i];
  }
  const double relative = std::sqrt(squared / std::max(scale, 1e-12));
  require(maximum <= 0.08 && relative <= 0.02,
          "P06 split full forward exceeds model numeric tolerance");
  return {{"max_abs", maximum},
          {"relative_l2", relative},
          {"same_token", a.token == b.token}};
}
void run_mode(const std::filesystem::path &fixtures,
              const std::filesystem::path &libs, const char *mode, Json &out) {
  const auto root = fixtures / mode;
  auto config = load_qwen_model_config(root / "source/config.json");
  auto artifact = ArtifactModel::open(root / "model.omlxm5", true);
  auto baseline_context = MetalContext::create();
  auto split_context = MetalContext::create();
  auto baseline_weights = materialize_model(
      artifact, plan_model_materialization(artifact, baseline_context),
      baseline_context);
  auto split_weights = materialize_model(
      artifact, plan_model_materialization(artifact, split_context),
      split_context);
  auto layers = build_qwen_layer_catalog(artifact, config, 2048, 64);
  const std::array<std::uint32_t, 2> taps{0, 1};
  QwenForward baseline(baseline_context, baseline_weights, layers, config, 2048,
                       libs,taps,false,true,false,{},false,false,{},
                       exact_numeric?LinearNumericPolicy::RowExactPortableV1:LinearNumericPolicy::Legacy,
                       SamplingContract::LegacyRejectionV1,
                       exact_numeric?AttentionNumericPolicy::RowExactV1:AttentionNumericPolicy::Legacy);
  QwenForward split(split_context, split_weights, layers, config, 2048, libs,
                    taps,false,true,false,{},true,false,{},
                    exact_numeric?LinearNumericPolicy::RowExactPortableV1:LinearNumericPolicy::Legacy,
                    SamplingContract::LegacyRejectionV1,
                    exact_numeric?AttentionNumericPolicy::RowExactV1:AttentionNumericPolicy::Legacy);
  require(split.memory_plan().workspace_bytes >
              baseline.memory_plan().workspace_bytes,
          "P06 split workspace not accounted");
  const auto before = split_context.registry_stats();
  Json cases = Json::array();
  for (const auto history : {0U, 63U, 129U, 512U, 1024U}) {
    for (const auto width : {1U, 2U, 4U, 8U}) {
      baseline.reset();
      split.reset();
      if (history) {
        const auto prefix = tokens(history);
        static_cast<void>(baseline.run(prefix));
        static_cast<void>(split.run(prefix));
      }
      const auto request = tokens(width, history);
      const auto desc = split.inspect_profile(width);
      const bool strict_2p = strict_two_pass_env();
      const char *expected_partial =
          exact_numeric ? (strict_2p ? "omlx_m5_paged_attention_exact_max"
                                     : "omlx_m5_paged_attention_exact_1pass_partial")
                        : "omlx_m5_paged_attention_split_partial";
      const char *expected_merge =
          exact_numeric ? (strict_2p ? "omlx_m5_paged_attention_exact_merge"
                                     : "omlx_m5_paged_attention_exact_1pass_merge")
                        : "omlx_m5_paged_attention_split_merge";
      const std::uint64_t expected_exact_ws =
          std::uint64_t(desc.capacity) * config.num_attention_heads * 4ULL *
          (strict_2p ? 4ULL
                     : (8ULL + std::uint64_t(config.head_dim) * 4ULL));
      require(
          desc.split_attention && desc.split_count == 4 &&
              desc.context_envelope == 2048 && desc.split_workspace_bytes > 0 &&
              desc.attention_kernel == expected_partial &&
              desc.attention_merge_kernel == expected_merge &&
              desc.attention_grid.width ==
                  std::uint64_t(desc.split_count) * 32 &&
              desc.attention_grid.height ==
                  std::uint64_t(desc.capacity) * config.num_attention_heads &&
              desc.dispatch_count ==
                  baseline.inspect_profile(width).dispatch_count + 1,
          "P06 actual candidate profile did not install split/merge");
      if(exact_numeric)require(desc.attention_row_exact_eligible &&
          desc.attention_numeric_contract_id==baseline.inspect_profile(width).attention_numeric_contract_id &&
          desc.split_workspace_bytes==expected_exact_ws &&
          !desc.whole_program_row_exact_qualified,"P06-TF metadata/ordered scratch qualification");
      const auto a = baseline.run(request, {}, true);
      const auto b = split.run(request, {}, true);
      auto metric = compare(a, b);
      require(baseline.inspect_recurrent_state() ==
                  split.inspect_recurrent_state(),
              "P06 split modified unrelated recurrent state");
      metric["history"] = history;
      metric["width"] = width;
      metric["scratch_bytes"] = desc.split_workspace_bytes;
      cases.push_back(metric);
    }
  }
  // Verify and ReplaySSM keep the same committed frontier for every selected
  // prefix. The split route uses the same nonmonotonic P05 page table.
  const std::array<std::uint32_t, 8> verification{7,   3,  100, 200,
                                                  256, 44, 23,  87};
  for (const auto accepted : {0U, 4U, 8U}) {
    baseline.reset();
    split.reset();
    const auto prefix = tokens(65);
    static_cast<void>(baseline.run(prefix));
    static_cast<void>(split.run(prefix));
    const auto first = baseline.verify(verification, {}, true);
    const auto second = split.verify(verification, {}, true);
    auto metric = compare(first, second);
    require(split.inspect_profile(8, true).split_attention &&
                baseline.position() == split.position() &&
                baseline.inspect_recurrent_state() ==
                    split.inspect_recurrent_state(),
            "P06 verify mutated committed state");
    static_cast<void>(baseline.commit_prefix(accepted));
    static_cast<void>(split.commit_prefix(accepted));
    require(baseline.position() == split.position() &&
                baseline.inspect_recurrent_state() ==
                    split.inspect_recurrent_state(),
            "P06 split ReplaySSM state differs");
    const auto continuation = tokens(1, 65 + accepted);
    metric["continuation"] = compare(baseline.run(continuation, {}, true),
                                     split.run(continuation, {}, true));
    metric["accepted"] = accepted;
    cases.push_back(metric);
  }
  // Releasing several P05 pages and rerunning the same request produces a
  // nonmonotonic physical mapping; split must use the table, not logical IDs.
  baseline.reset();
  split.reset();
  const auto page_prompt = tokens(129);
  for (unsigned attempt = 0; attempt < 3; ++attempt) {
    static_cast<void>(baseline.run(page_prompt));
    static_cast<void>(split.run(page_prompt));
    const auto mapping = split.inspect_kv_page_table();
    if (!(mapping[0] < mapping[1] && mapping[1] < mapping[2]))
      break;
    baseline.reset();
    split.reset();
  }
  const auto mapping = split.inspect_kv_page_table();
  require(!(mapping[0] < mapping[1] && mapping[1] < mapping[2]),
          "P06 nonmonotonic page mapping was not exercised");
  auto mapped_metric = compare(baseline.run(tokens(8, 129), {}, true),
                               split.run(tokens(8, 129), {}, true));
  mapped_metric["nonmonotonic_page_table"] = true;
  cases.push_back(mapped_metric);
  split.reset();
  static_cast<void>(split.run(tokens(63)));
  const auto direct_split = split.run(tokens(8, 63), {}, true, false);
  const auto direct_state = split.inspect_recurrent_state();
  split.reset();
  static_cast<void>(split.run(tokens(63)));
  const auto icb_split = split.run(tokens(8, 63), {}, true, true);
  require(direct_split.hidden == icb_split.hidden &&
              direct_split.logits == icb_split.logits &&
              direct_split.token == icb_split.token &&
              direct_state == split.inspect_recurrent_state(),
          "P06 split direct/ICB changed model output or state");
  if (std::string_view(mode) == "affine") {
    auto long_context = MetalContext::create();
    auto long_weights = materialize_model(
        artifact, plan_model_materialization(artifact, long_context),
        long_context);
    auto long_layers = build_qwen_layer_catalog(artifact, config, 32768, 64);
    QwenForward installed(long_context, long_weights, long_layers, config,
                          32768,libs,taps,false,true,false,{},true,false,{},
                          exact_numeric?LinearNumericPolicy::RowExactPortableV1:LinearNumericPolicy::Legacy,
                          SamplingContract::LegacyRejectionV1,
                          exact_numeric?AttentionNumericPolicy::RowExactV1:AttentionNumericPolicy::Legacy);
    require(installed.inspect_split_profile(1, 2048).split_count == 4 &&
                installed.inspect_split_profile(1, 2049).split_count == 16 &&
                installed.inspect_split_profile(1, 8192).split_count == 16 &&
                installed.inspect_split_profile(1, 8193).split_count == 64 &&
                installed.inspect_split_profile(8, 32768).split_count == 64 &&
                installed.inspect_split_profile(8, 32768).capacity == 8,
            "P06 bounded 2K/8K/32K profiles not installed");
    bool rejected = false;
    try {
      static_cast<void>(installed.inspect_split_profile(8, 32769));
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "P06 context outside installed envelope accepted");
    if (!exact_numeric) {
      auto layers_256k = build_qwen_layer_catalog(artifact, config, 262144, 64);
      QwenForward legacy_fnax_256k(
          long_context, long_weights, layers_256k, config, 262144, libs, taps,
          false, true, false, {}, true, false, {}, LinearNumericPolicy::Legacy,
          SamplingContract::LegacyRejectionV1, AttentionNumericPolicy::Legacy,
          false, "bf16", nullptr, /*enable_flash_nax_prefill=*/true);
      require(legacy_fnax_256k.inspect_split_profile(8, 262144).split_count ==
                  256,
              "P06 Legacy + flash_nax_prefill 256K cap failed");
      static_cast<void>(legacy_fnax_256k.run(tokens(65)));
      static_cast<void>(legacy_fnax_256k.run(tokens(4, 65)));

      ScopedEnv gate_env("OMLX_M5_STRICT_TWO_PASS_ATTN", "1");
      QwenForward legacy_fnax_strict(
          long_context, long_weights, layers, config, 2048, libs, taps, false,
          true, false, {}, true, false, {}, LinearNumericPolicy::Legacy,
          SamplingContract::LegacyRejectionV1, AttentionNumericPolicy::Legacy,
          false, "bf16", nullptr, /*enable_flash_nax_prefill=*/true);
      require(legacy_fnax_strict.inspect_profile(8).attention_kernel ==
                  "omlx_m5_paged_attention_split_partial",
              "P06 Legacy + flash_nax_prefill + strict_2pass failed");
      static_cast<void>(legacy_fnax_strict.run(tokens(65)));
      static_cast<void>(legacy_fnax_strict.run(tokens(4, 65)));
    }
    if (exact_numeric && !strict_two_pass_env()) {
      auto layers_64k = build_qwen_layer_catalog(artifact, config, 65536, 64);
      QwenForward installed_64k(long_context, long_weights, layers_64k, config,
                                65536, libs, taps, false, true, false, {}, true,
                                false, {},
                                LinearNumericPolicy::RowExactPortableV1,
                                SamplingContract::LegacyRejectionV1,
                                AttentionNumericPolicy::RowExactV1);
      require(installed_64k.inspect_split_profile(1, 2048).split_count == 4 &&
                  installed_64k.inspect_split_profile(1, 8192).split_count == 16 &&
                  installed_64k.inspect_split_profile(1, 32768).split_count == 64 &&
                  installed_64k.inspect_split_profile(1, 32769).split_count == 128 &&
                  installed_64k.inspect_split_profile(8, 65536).split_count == 128,
              "P06 64K dynamic 512-token envelopes not installed");
      auto layers_256k = build_qwen_layer_catalog(artifact, config, 262144, 64);
      QwenForward installed_256k(long_context, long_weights, layers_256k,
                                 config, 262144, libs, taps, false, true, false,
                                 {}, true, false, {},
                                 LinearNumericPolicy::RowExactPortableV1,
                                 SamplingContract::LegacyRejectionV1,
                                 AttentionNumericPolicy::RowExactV1);
      require(installed_256k.inspect_split_profile(8, 262144).split_count == 256,
              "P06 max_partitions=256 cap failed at 256K");

      ScopedEnv gate_env("OMLX_M5_STRICT_TWO_PASS_ATTN", "1");
      QwenForward strict_base(long_context, long_weights, layers, config, 2048,
                              libs, taps, false, true, false, {}, false, false,
                              {}, LinearNumericPolicy::RowExactPortableV1,
                              SamplingContract::LegacyRejectionV1,
                              AttentionNumericPolicy::RowExactV1);
      QwenForward strict_split(long_context, long_weights, layers, config, 2048,
                               libs, taps, false, true, false, {}, true, false,
                               {}, LinearNumericPolicy::RowExactPortableV1,
                               SamplingContract::LegacyRejectionV1,
                               AttentionNumericPolicy::RowExactV1);
      const auto sdesc = strict_split.inspect_profile(8);
      require(sdesc.attention_kernel == "omlx_m5_paged_attention_exact_max" &&
                  sdesc.attention_merge_kernel ==
                      "omlx_m5_paged_attention_exact_merge" &&
                  sdesc.split_workspace_bytes ==
                      8ULL * config.num_attention_heads * 4ULL * 4ULL,
              "OMLX_M5_STRICT_TWO_PASS_ATTN=1 did not select strict 2-pass");
      static_cast<void>(strict_base.run(tokens(513)));
      static_cast<void>(strict_split.run(tokens(513)));
      const auto sb_out = strict_base.run(tokens(8, 513), {}, true);
      const auto ss_out = strict_split.run(tokens(8, 513), {}, true);
      require(sb_out.logits == ss_out.logits && sb_out.hidden == ss_out.hidden,
              "OMLX_M5_STRICT_TWO_PASS_ATTN=1 split vs base 0-ULP mismatch");
    }
  }
  const auto after = split_context.registry_stats();
  require(before.current_buffer_count == after.current_buffer_count &&
              before.pipelines_compiled == after.pipelines_compiled &&
              before.command_plans_built == after.command_plans_built,
          "P06 run created dynamic installation resources");
  out.push_back({{"mode", mode}, {"cases", cases}});
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 3 && argc != 4)
      throw std::invalid_argument("p06_attention_tests FIXTURES LIBRARIES [--row-exact]");
    if(argc==4){if(std::string_view(argv[3])!="--row-exact")throw std::invalid_argument("unknown P06 policy");exact_numeric=true;}
    Json modes = Json::array();
    for (const auto *mode : {"affine", "mxfp4"})
      run_mode(argv[1], argv[2], mode, modes);
    std::cout << Json({{"schema", "omlx-m5-p06-forward-v1"},
                       {"scope", "small two-layer target, not full model"},
                       {"exact_numeric_policy",exact_numeric},
                       {"modes", modes}})
                     .dump(2)
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "P06 attention failed: " << error.what() << '\n';
    return 1;
  }
}
