#include "omlx_m5/engine.h"
#include "omlx_m5/execution_contract.h"
#include "omlx_m5/kv_page_pool.h"
#include "omlx_m5/types.h"
#include "serve/config.h"

#include <nlohmann/json.hpp>

#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace omlx::m5;
using namespace omlx::m5::serve;

void check(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error("CHECK FAILED: " + message);
  }
}

class ScopedEnvVar final {
public:
  ScopedEnvVar(std::string key, std::optional<std::string> value)
      : key_(std::move(key)) {
    if (const char *prev = std::getenv(key_.c_str())) {
      had_prev_ = true;
      prev_ = prev;
    }
    if (value.has_value()) {
      ::setenv(key_.c_str(), value->c_str(), 1);
    } else {
      ::unsetenv(key_.c_str());
    }
  }
  ~ScopedEnvVar() {
    if (had_prev_) {
      ::setenv(key_.c_str(), prev_.c_str(), 1);
    } else {
      ::unsetenv(key_.c_str());
    }
  }
  ScopedEnvVar(const ScopedEnvVar &) = delete;
  ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

private:
  std::string key_;
  bool had_prev_ = false;
  std::string prev_;
};

ServeConfig parse_args(std::vector<std::string> args) {
  std::vector<char *> argv;
  argv.reserve(args.size());
  for (auto &a : args) {
    argv.push_back(a.data());
  }
  return parse_serve_config(static_cast<int>(argv.size()), argv.data());
}

void test_enum_helpers_and_runtime_stats_fields() {
  check(std::string(to_string(PrefillLinearMode::BFloat16)) == "bf16",
        "BFloat16 string mismatch");
  check(std::string(to_string(PrefillLinearMode::W4A8)) == "w4a8",
        "W4A8 string mismatch");
  check(std::string(to_string(PrefillLinearMode::W4A4)) == "w4a4",
        "W4A4 string mismatch");
  check(std::string(to_string(PrefillAttentionMode::ExactNax)) == "exact_nax",
        "ExactNax string mismatch");
  check(std::string(to_string(PrefillAttentionMode::Sage2)) == "sage2",
        "Sage2 string mismatch");
  check(std::string(to_string(PrefillAneMode::Off)) == "off",
        "Ane Off string mismatch");
  check(std::string(to_string(PrefillAneMode::Auto)) == "auto",
        "Ane Auto string mismatch");
  check(std::string(to_string(PrefillAneMode::CompactInt8)) == "compact_int8",
        "Ane CompactInt8 string mismatch");
  check(std::string(to_string(PrefillAneMode::UltraFp16SwiGlu)) ==
            "ultra_fp16_swiglu",
        "Ane UltraFp16SwiGlu string mismatch");

  check(parse_prefill_linear_mode("bf16") == PrefillLinearMode::BFloat16,
        "parse bf16 failed");
  check(parse_prefill_linear_mode("W4A8") == PrefillLinearMode::W4A8,
        "parse W4A8 failed");
  check(parse_prefill_linear_mode("W4A4") == PrefillLinearMode::W4A4,
        "parse W4A4 failed");
  check(parse_prefill_attention_mode("exact_nax") ==
            PrefillAttentionMode::ExactNax,
        "parse exact_nax failed");
  check(parse_prefill_attention_mode("SAGE2") == PrefillAttentionMode::Sage2,
        "parse SAGE2 failed");
  check(parse_prefill_ane_mode("off") == PrefillAneMode::Off,
        "parse ane off failed");
  check(parse_prefill_ane_mode("AUTO") == PrefillAneMode::Auto,
        "parse ane auto failed");
  check(parse_prefill_ane_mode("compact_int8") == PrefillAneMode::CompactInt8,
        "parse ane compact_int8 failed");
  check(parse_prefill_ane_mode("ULTRA_FP16_SWIGLU") ==
            PrefillAneMode::UltraFp16SwiGlu,
        "parse ane ultra_fp16_swiglu failed");

  bool threw_linear = false;
  try {
    (void)parse_prefill_linear_mode("fp8");
  } catch (const std::invalid_argument &) {
    threw_linear = true;
  }
  check(threw_linear, "invalid prefill_linear_mode must throw");

  bool threw_attn = false;
  try {
    (void)parse_prefill_attention_mode("flash2");
  } catch (const std::invalid_argument &) {
    threw_attn = true;
  }
  check(threw_attn, "invalid prefill_attention_mode must throw");

  bool threw_ane = false;
  try {
    (void)parse_prefill_ane_mode("turbo_ane");
  } catch (const std::invalid_argument &) {
    threw_ane = true;
  }
  check(threw_ane, "invalid prefill_ane_mode must throw");

  RuntimeStats stats;
  check(stats.prefill_linear_mode == "bf16" &&
            stats.prefill_linear_mode_effective == "bf16" &&
            !stats.prefill_linear_fallback &&
            stats.prefill_linear_fallback_reason.empty() &&
            stats.model_quant_format.empty() &&
            stats.prefill_attention_mode == "exact_nax" &&
            !stats.prefill_sparse_minference &&
            !stats.prefill_sparse_minference_effective &&
            !stats.prefill_cache_blend && stats.prefill_ane_mode == "off" &&
            stats.prefill_ane_mode_effective == "off" &&
            stats.prefill_ane_inactive_reason.empty() &&
            !stats.prefill_ane_offload && !stats.prefill_ane_active &&
            stats.prefill_ane_split_alpha == 0.0 &&
            stats.prefill_w4a8_tokens == 0 && stats.prefill_w4a4_tokens == 0 &&
            stats.prefill_sage2_tokens == 0 &&
            stats.prefill_minference_blocks_skipped == 0 &&
            stats.prefill_minference_blocks_total == 0 &&
            stats.prefill_cache_blend_hit_tokens == 0 &&
            stats.prefill_cache_blend_recomputed_tokens == 0 &&
            stats.prefill_ane_offload_tokens == 0 &&
            stats.prefill_ane_fallback_tokens == 0,
        "RuntimeStats defaults mismatch");
  std::cout << "[spi] PASS: enum helpers & RuntimeStats defaults\n";
}

void test_all_96_orthogonal_combinations_and_hash_isolation() {
  const ExecutionIdentityFields baseline_fields = {
      {"model", "deadbeef"},
      {"numeric", "linear_row_exact_portable_v1+attention.row_exact.v1"},
      {"sampling", "position_coupled_v1"},
      {"max_context", "262144"},
  };
  const std::string baseline_digest =
      execution_identity_digest(baseline_fields);

  const std::vector<PrefillLinearMode> linear_modes = {
      PrefillLinearMode::BFloat16,
      PrefillLinearMode::W4A8,
      PrefillLinearMode::W4A4,
  };
  const std::vector<PrefillAneMode> ane_modes = {
      PrefillAneMode::Off,
      PrefillAneMode::Auto,
      PrefillAneMode::CompactInt8,
      PrefillAneMode::UltraFp16SwiGlu,
  };

  std::set<std::string> distinct_digests;
  int count = 0;

  for (const auto linear_mode : linear_modes) {
    for (int am = 0; am < 2; ++am) {
      for (int sp = 0; sp < 2; ++sp) {
        for (int cb = 0; cb < 2; ++cb) {
          for (const auto ane_mode : ane_modes) {
            ++count;
            LoadOptions opts;
            opts.max_context = 262144;
            opts.max_concurrency = 4;
            opts.kv_dtype = "affine4_g128";
            opts.sampling_contract = "legacy_rejection_v1";
            opts.prefill_linear_mode = linear_mode;
            opts.prefill_attention_mode = am == 0
                                              ? PrefillAttentionMode::ExactNax
                                              : PrefillAttentionMode::Sage2;
            opts.prefill_sparse_minference = (sp != 0);
            opts.prefill_cache_blend = (cb != 0);
            opts.prefill_ane_mode = ane_mode;
            opts.prefill_ane_offload = (ane_mode != PrefillAneMode::Off);

            auto engine = Engine::create_fake(opts, FakeBackendOptions{});
            const StatsReport st = engine->stats();
            engine->shutdown();

            check(st.max_context == 262144,
                  "max_context 262144 must not be clamped");
            check(st.prefill_linear_mode ==
                      prefill_linear_mode_to_string(opts.prefill_linear_mode),
                  "prefill_linear_mode mutated or cleared");
            check(st.prefill_linear_mode_effective == st.prefill_linear_mode,
                  "prefill_linear_mode_effective default mismatch");
            check(st.prefill_attention_mode ==
                      prefill_attention_mode_to_string(
                          opts.prefill_attention_mode),
                  "prefill_attention_mode mutated or cleared");
            check(st.prefill_sparse_minference ==
                      opts.prefill_sparse_minference,
                  "prefill_sparse_minference mutated");
            check(st.prefill_cache_blend == opts.prefill_cache_blend,
                  "prefill_cache_blend mutated");
            check(st.prefill_ane_mode ==
                      prefill_ane_mode_to_string(opts.prefill_ane_mode),
                  "prefill_ane_mode mutated or cleared");
            check(st.prefill_ane_offload == (ane_mode != PrefillAneMode::Off),
                  "prefill_ane_offload compat flag mismatch");
            const bool expected_eff =
                (opts.prefill_attention_mode == PrefillAttentionMode::Sage2) &&
                opts.prefill_sparse_minference;
            check(st.prefill_sparse_minference_effective == expected_eff,
                  "prefill_sparse_minference_effective mismatch");
            if (opts.prefill_sparse_minference && !expected_eff) {
              check(st.prefill_sparse_minference_inactive_reason ==
                        "requires_sage2",
                    "missing requires_sage2 inactive reason");
            } else {
              check(st.prefill_sparse_minference_inactive_reason.empty(),
                    "unexpected inactive reason");
            }

            ExecutionIdentityFields fields = baseline_fields;
            append_prefill_execution_identity_fields(opts, fields);
            const std::string digest = execution_identity_digest(fields);
            distinct_digests.insert(digest);

            if (linear_mode == PrefillLinearMode::BFloat16 && am == 0 &&
                ane_mode == PrefillAneMode::Off) {
              // Default + inactive sparse_minference + cache_blend preserve
              // exact Tier-1 SharedPrefixCache hash!
              check(digest == baseline_digest,
                    "default/tier1-compatible execution_fields hash drifted!");
            } else {
              check(digest != baseline_digest,
                    "non-default active switch must isolate execution_fields!");
            }
          }
        }
      }
    }
  }
  check(count == 96, "expected 96 combinations");
  // 3 (bf16|w4a8|w4a4) x 3 (exact_nax|sage2|sage2+minference) x 4 (ane modes) = 36
  check(distinct_digests.size() == 36,
        "expected 36 distinct execution_fields digests across 96 raw modes, "
        "got: " +
            std::to_string(distinct_digests.size()));

  // Verify effective_linear_mode fallback override aligns execution digest with executed mode.
  {
    LoadOptions req_w4a4;
    req_w4a4.prefill_linear_mode = PrefillLinearMode::W4A4;
    ExecutionIdentityFields fallback_fields = baseline_fields;
    append_prefill_execution_identity_fields(req_w4a4, fallback_fields,
                                             PrefillLinearMode::W4A8);
    LoadOptions req_w4a8;
    req_w4a8.prefill_linear_mode = PrefillLinearMode::W4A8;
    ExecutionIdentityFields w4a8_fields = baseline_fields;
    append_prefill_execution_identity_fields(req_w4a8, w4a8_fields);
    check(execution_identity_digest(fallback_fields) ==
              execution_identity_digest(w4a8_fields),
          "w4a4->w4a8 fallback execution digest must match w4a8 digest");
  }

  std::cout << "[spi] PASS: all 96 orthogonal combinations & 36-way hash "
               "isolation\n";
}

void test_serve_config_toml_env_cli_and_legacy_shim() {
  const ScopedEnvVar clear_cfg("NINFER_METAL_CONFIG", std::nullopt);
  const ScopedEnvVar clear_lm("NINFER_PREFILL_LINEAR_MODE", std::nullopt);
  const ScopedEnvVar clear_am("NINFER_PREFILL_ATTENTION_MODE", std::nullopt);
  const ScopedEnvVar clear_sp("NINFER_PREFILL_SPARSE_MINFERENCE", std::nullopt);
  const ScopedEnvVar clear_cb("NINFER_PREFILL_CACHE_BLEND", std::nullopt);
  const ScopedEnvVar clear_ane("NINFER_PREFILL_ANE_OFFLOAD", std::nullopt);
  const ScopedEnvVar clear_ane_mode("NINFER_PREFILL_ANE_MODE", std::nullopt);

  // 1) Repo default ninfer-metal.toml has max_context=262144 + 5 defaults.
  const auto repo_toml =
      std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
      "omlx-extend" / "config" / "ninfer-metal.toml";
  {
    const auto cfg = parse_args(
        {"ninfer-metal-serve", "--config", repo_toml.string()});
    check(cfg.max_context == 262144, "repo toml max_context must be 262144");
    check(cfg.max_concurrency == 4, "repo toml max_concurrency must be 4");
    check(cfg.kv_dtype == "affine4_g128",
          "repo toml kv_dtype must be affine4_g128");
    check(cfg.prefix_cache_gb.has_value() && *cfg.prefix_cache_gb == 4.0,
          "repo toml prefix_cache_gb must be 4.0");
    check(cfg.prefill_linear_mode == PrefillLinearMode::BFloat16 &&
              cfg.prefill_attention_mode == PrefillAttentionMode::ExactNax &&
              !cfg.prefill_sparse_minference && !cfg.prefill_cache_blend &&
              cfg.prefill_ane_mode == PrefillAneMode::Off &&
              !cfg.prefill_ane_offload,
          "repo toml prefill switch defaults mismatch");
  }

  // 2) Custom TOML (w4a4 + ultra_fp16_swiglu) + NINFER_PREFILL_* env + CLI overrides.
  const auto tmp_dir =
      std::filesystem::temp_directory_path() /
      ("omlx_m5_spi_cfg_" + std::to_string(::getpid()));
  std::filesystem::create_directories(tmp_dir);
  const auto custom_toml = tmp_dir / "custom.toml";
  {
    std::ofstream out(custom_toml);
    out << "[engine]\n"
        << "max_context = 262144\n"
        << "max_concurrency = 4\n"
        << "kv_dtype = \"affine4_g128\"\n"
        << "prefill_linear_mode = \"w4a4\"\n"
        << "prefill_attention_mode = \"sage2\"\n"
        << "prefill_sparse_minference = true\n"
        << "prefill_cache_blend = true\n"
        << "prefill_ane_mode = \"ultra_fp16_swiglu\"\n";
  }
  {
    const auto cfg = parse_args(
        {"ninfer-metal-serve", "--config", custom_toml.string()});
    const auto lo = to_load_options(cfg);
    check(lo.max_context == 262144 &&
              lo.prefill_linear_mode == PrefillLinearMode::W4A4 &&
              lo.prefill_attention_mode == PrefillAttentionMode::Sage2 &&
              lo.prefill_sparse_minference && lo.prefill_cache_blend &&
              lo.prefill_ane_mode == PrefillAneMode::UltraFp16SwiGlu &&
              lo.prefill_ane_offload,
          "custom all-on TOML failed to map to LoadOptions");
  }

  // Env overrides TOML; CLI overrides env!
  {
    const ScopedEnvVar e_lm("NINFER_PREFILL_LINEAR_MODE", "w4a4");
    const ScopedEnvVar e_am("NINFER_PREFILL_ATTENTION_MODE", "sage2");
    const ScopedEnvVar e_sp("NINFER_PREFILL_SPARSE_MINFERENCE", "true");
    const ScopedEnvVar e_cb("NINFER_PREFILL_CACHE_BLEND", "1");
    const ScopedEnvVar e_ane_mode("NINFER_PREFILL_ANE_MODE", "compact_int8");

    const auto env_cfg = parse_args(
        {"ninfer-metal-serve", "--config", repo_toml.string()});
    check(env_cfg.prefill_linear_mode == PrefillLinearMode::W4A4 &&
              env_cfg.prefill_attention_mode == PrefillAttentionMode::Sage2 &&
              env_cfg.prefill_sparse_minference &&
              env_cfg.prefill_cache_blend &&
              env_cfg.prefill_ane_mode == PrefillAneMode::CompactInt8 &&
              env_cfg.prefill_ane_offload,
          "NINFER_PREFILL_* env vars did not override TOML");

    const auto cli_cfg = parse_args({
        "ninfer-metal-serve",
        "--config",
        repo_toml.string(),
        "--prefill-linear-mode",
        "bf16",
        "--prefill-attention-mode=exact_nax",
        "--no-prefill-sparse-minference",
        "--no-prefill-cache-blend",
        "--prefill-ane-mode",
        "ultra_fp16_swiglu",
    });
    check(cli_cfg.prefill_linear_mode == PrefillLinearMode::BFloat16 &&
              cli_cfg.prefill_attention_mode ==
                  PrefillAttentionMode::ExactNax &&
              !cli_cfg.prefill_sparse_minference &&
              !cli_cfg.prefill_cache_blend &&
              cli_cfg.prefill_ane_mode == PrefillAneMode::UltraFp16SwiGlu &&
              cli_cfg.prefill_ane_offload,
          "CLI flags did not override NINFER_PREFILL_* env vars");

    const auto cli_off = parse_args({
        "ninfer-metal-serve",
        "--config",
        repo_toml.string(),
        "--prefill-ane-mode=off",
    });
    check(cli_off.prefill_ane_mode == PrefillAneMode::Off &&
              !cli_off.prefill_ane_offload,
          "CLI --prefill-ane-mode=off did not override env");
  }

  // 3) Legacy prefill_engine = "nax_ane" shim maps prefill_ane_mode=auto & prefill_ane_offload=true
  //    WITHOUT touching w4a8 / sage2 / minference / cache_blend!
  const auto legacy_toml = tmp_dir / "legacy_ane.toml";
  {
    std::ofstream out(legacy_toml);
    out << "[engine]\n"
        << "max_context = 262144\n"
        << "prefill_engine = \"nax_ane\"\n"
        << "prefill_linear_mode = \"w4a8\"\n"
        << "prefill_attention_mode = \"sage2\"\n"
        << "prefill_sparse_minference = true\n"
        << "prefill_cache_blend = true\n";
  }
  {
    const auto cfg = parse_args(
        {"ninfer-metal-serve", "--config", legacy_toml.string()});
    const auto lo = to_load_options(cfg);
    check(lo.prefill_ane_mode == PrefillAneMode::Auto && lo.prefill_ane_offload,
          "legacy prefill_engine=nax_ane must set prefill_ane_mode=auto & offload=true");
    check(lo.prefill_linear_mode == PrefillLinearMode::W4A8 &&
              lo.prefill_attention_mode == PrefillAttentionMode::Sage2 &&
              lo.prefill_sparse_minference && lo.prefill_cache_blend,
          "legacy prefill_engine=nax_ane shim must not touch other switches");

    // Explicit CLI --no-prefill-ane-offload overrides legacy prefill_engine!
    const auto overridden =
        parse_args({"ninfer-metal-serve", "--config", legacy_toml.string(),
                    "--no-prefill-ane-offload"});
    const auto lo_off = to_load_options(overridden);
    check(lo_off.prefill_ane_mode == PrefillAneMode::Off &&
              !lo_off.prefill_ane_offload,
          "explicit --no-prefill-ane-offload must override legacy nax_ane");
  }

  // 4) Legacy prefill_ane_offload = true -> Auto; explicit prefill_ane_mode wins in same layer
  const auto legacy_bool_toml = tmp_dir / "legacy_bool_ane.toml";
  {
    std::ofstream out(legacy_bool_toml);
    out << "[engine]\n"
        << "prefill_ane_offload = true\n";
  }
  {
    const auto cfg = parse_args(
        {"ninfer-metal-serve", "--config", legacy_bool_toml.string()});
    check(cfg.prefill_ane_mode == PrefillAneMode::Auto &&
              cfg.prefill_ane_offload,
          "legacy prefill_ane_offload=true must map to PrefillAneMode::Auto");
    const auto overridden = parse_args(
        {"ninfer-metal-serve", "--config", legacy_bool_toml.string(),
         "--prefill-ane-mode", "compact_int8"});
    check(overridden.prefill_ane_mode == PrefillAneMode::CompactInt8 &&
              overridden.prefill_ane_offload,
          "explicit --prefill-ane-mode must override legacy prefill_ane_offload");
  }

  std::error_code ec;
  std::filesystem::remove_all(tmp_dir, ec);
  std::cout << "[spi] PASS: TOML, NINFER_PREFILL_* env, CLI & legacy shim\n";
}

void test_256k_kv_page_sizing_c3() {
  // Qwen3.8-27B geometry: 16 full-attn layers, 4 KV heads, head_dim=256, page=64
  const std::uint64_t kv_page_b =
      kv_page_bytes(KvStorageFormat::Affine4Group128, 64U, 4U, 256U);
  check(kv_page_b == 36992ULL,
        "unexpected affine4_g128 page bytes: " + std::to_string(kv_page_b));
  const std::uint64_t per_page_all_layers = 16ULL * 2ULL * kv_page_b;
  check(per_page_all_layers == 1183744ULL,
        "unexpected per_page_all_layers bytes");
  const std::uint32_t logical_pages = (262144U + 63U) / 64U;
  const std::uint32_t single_slot_pages = logical_pages + 1U;
  check(single_slot_pages == 4097U, "256K single-slot pages should be 4097");
  const std::uint64_t max_safe_pages = (8ULL << 30) / kv_page_b;
  check(single_slot_pages <= max_safe_pages,
        "256K pages exceed 8 GiB slab safe limit");
  check(single_slot_pages * per_page_all_layers == 4849799168ULL,
        "256K affine4_g128 KV bytes mismatch");
  std::cout << "[spi] PASS: 256K (262,144) KvPagePool C3 sizing verified\n";
}

Json uds_rpc(int fd, std::uint64_t id, const std::string &method,
             const Json &params = Json::object()) {
  const std::string line =
      Json{{"id", id}, {"method", method}, {"params", params}}.dump() + "\n";
  std::size_t off = 0;
  while (off < line.size()) {
    const auto n = ::send(fd, line.data() + off, line.size() - off, 0);
    if (n <= 0) {
      throw std::runtime_error("send to mseries_backendd failed");
    }
    off += static_cast<std::size_t>(n);
  }
  std::string resp;
  char ch = 0;
  while (true) {
    const auto n = ::recv(fd, &ch, 1, 0);
    if (n <= 0) {
      throw std::runtime_error("recv from mseries_backendd failed");
    }
    if (ch == '\n') {
      break;
    }
    resp.push_back(ch);
  }
  const Json doc = Json::parse(resp);
  if (!doc.value("ok", false)) {
    throw std::runtime_error("RPC error for " + method + ": " + doc.dump());
  }
  return doc.at("result");
}

void test_sidecar_daemon_describe_and_96_modes(const std::string &daemon_path) {
  // 1) --describe check
  const std::string cmd = "\"" + daemon_path + "\" --describe";
  FILE *pipe = ::popen(cmd.c_str(), "r");
  check(pipe != nullptr, "popen --describe failed");
  std::string out;
  char buf[4096];
  while (const std::size_t n = std::fread(buf, 1, sizeof(buf), pipe)) {
    out.append(buf, n);
  }
  const int rc = ::pclose(pipe);
  check(rc == 0, "--describe exited non-zero");
  const Json desc = Json::parse(out);
  check(desc.at("load_defaults").at("prefill_ane_mode") == "off",
        "load_defaults.prefill_ane_mode mismatch");
  std::map<std::string, Json> schema;
  for (const auto &item : desc.at("options_schema")) {
    schema[item.at("key").get<std::string>()] = item;
  }
  check(schema.contains("max_context") &&
            schema.at("max_context").at("max").get<int>() >= 262144,
        "options_schema.max_context ceiling must be >= 262144");
  check(schema.contains("prefill_linear_mode") &&
            schema.at("prefill_linear_mode").at("default") == "bf16" &&
            schema.at("prefill_linear_mode").at("options") ==
                Json::array({"bf16", "w4a8"}),
        "options_schema.prefill_linear_mode mismatch");
  check(schema.contains("prefill_attention_mode") &&
            schema.at("prefill_attention_mode").at("default") == "exact_nax" &&
            schema.at("prefill_attention_mode").at("options") ==
                Json::array({"exact_nax", "sage2"}),
        "options_schema.prefill_attention_mode mismatch");
  for (const char *bk : {"prefill_sparse_minference", "prefill_cache_blend"}) {
    check(schema.contains(bk) && schema.at(bk).at("type") == "bool" &&
              schema.at(bk).at("default") == false,
          std::string("options_schema.") + bk + " mismatch");
  }
  check(schema.contains("prefill_ane_mode") &&
            schema.at("prefill_ane_mode").at("type") == "enum" &&
            schema.at("prefill_ane_mode").at("default") == "off" &&
            schema.at("prefill_ane_mode").at("options") ==
                Json::array(
                    {"off", "auto", "compact_int8", "ultra_fp16_swiglu"}),
        "options_schema.prefill_ane_mode mismatch");

  // 2) Live UDS daemon: hello + all 96 combinations + legacy shim
  const auto sock_path =
      std::filesystem::temp_directory_path() /
      ("m5spi_" + std::to_string(::getpid()) + ".sock");
  std::error_code ec;
  std::filesystem::remove(sock_path, ec);

  const pid_t pid = ::fork();
  check(pid >= 0, "fork failed");
  if (pid == 0) {
    ::execl(daemon_path.c_str(), daemon_path.c_str(), "--socket",
            sock_path.c_str(), static_cast<char *>(nullptr));
    ::_exit(127);
  }

  int fd = -1;
  try {
    for (int attempt = 0; attempt < 200; ++attempt) {
      fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
      sockaddr_un addr{};
      addr.sun_family = AF_UNIX;
      std::strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);
      if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) ==
          0) {
        break;
      }
      ::close(fd);
      fd = -1;
      std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    check(fd >= 0, "failed to connect to mseries_backendd socket");

    std::uint64_t next_id = 1;
    const Json hello = uds_rpc(fd, next_id++, "hello");
    check(hello.at("options_schema") == desc.at("options_schema"),
          "hello and --describe options_schema diverged");

    int rpc_combos = 0;
    for (const std::string &lm : {"bf16", "w4a8", "w4a4"}) {
      for (const std::string &am : {"exact_nax", "sage2"}) {
        for (bool sp : {false, true}) {
          for (bool cb : {false, true}) {
            for (const std::string &ane_mode :
                 {"off", "auto", "compact_int8", "ultra_fp16_swiglu"}) {
              ++rpc_combos;
              const Json st = uds_rpc(
                  fd, next_id++, "load",
                  {
                      {"fake", true},
                      {"max_context", 262144},
                      {"prefill_linear_mode", lm},
                      {"prefill_attention_mode", am},
                      {"prefill_sparse_minference", sp},
                      {"prefill_cache_blend", cb},
                      {"prefill_ane_mode", ane_mode},
                  });
              check(st.at("max_context").get<int>() == 262144,
                    "daemon load max_context mismatch");
              check(st.at("prefill_linear_mode").get<std::string>() == lm,
                    "daemon prefill_linear_mode mismatch");
              check(st.at("prefill_linear_mode_effective").get<std::string>() ==
                        lm,
                    "daemon prefill_linear_mode_effective mismatch");
              check(st.at("prefill_attention_mode").get<std::string>() == am,
                    "daemon prefill_attention_mode mismatch");
              check(st.at("prefill_sparse_minference").get<bool>() == sp,
                    "daemon prefill_sparse_minference mismatch");
              check(st.at("prefill_sparse_minference_effective").get<bool>() ==
                        (am == "sage2" && sp),
                    "daemon prefill_sparse_minference_effective mismatch");
              check(st.at("prefill_cache_blend").get<bool>() == cb,
                    "daemon prefill_cache_blend mismatch");
              check(st.at("prefill_ane_mode").get<std::string>() == ane_mode,
                    "daemon prefill_ane_mode mismatch");
              check(st.at("prefill_ane_offload").get<bool>() ==
                        (ane_mode != "off"),
                    "daemon prefill_ane_offload compat mismatch");
              check(st.contains("prefill_linear_fallback") &&
                        st.contains("prefill_linear_fallback_reason") &&
                        st.contains("model_quant_format") &&
                        st.contains("prefill_ane_mode_effective") &&
                        st.contains("prefill_ane_inactive_reason") &&
                        st.contains("prefill_ane_active") &&
                        st.contains("prefill_ane_split_alpha") &&
                        st.contains("prefill_w4a8_tokens") &&
                        st.contains("prefill_w4a4_tokens") &&
                        st.contains("prefill_sage2_tokens") &&
                        st.contains("prefill_minference_blocks_skipped") &&
                        st.contains("prefill_minference_blocks_total") &&
                        st.contains("prefill_cache_blend_hit_tokens") &&
                        st.contains("prefill_cache_blend_recomputed_tokens") &&
                        st.contains("prefill_ane_offload_tokens") &&
                        st.contains("prefill_ane_fallback_tokens"),
                    "daemon stats missing prefill telemetry keys");
            }
          }
        }
      }
    }
    check(rpc_combos == 96, "expected 96 daemon RPC combinations");

    // Legacy prefill_engine = "nax_ane" shim over RPC
    {
      const Json st = uds_rpc(fd, next_id++, "load",
                              {
                                  {"fake", true},
                                  {"max_context", 262144},
                                  {"prefill_engine", "nax_ane"},
                                  {"prefill_linear_mode", "w4a4"},
                                  {"prefill_attention_mode", "sage2"},
                                  {"prefill_sparse_minference", true},
                                  {"prefill_cache_blend", true},
                              });
      check(st.at("prefill_ane_mode").get<std::string>() == "auto" &&
                st.at("prefill_ane_offload").get<bool>() == true &&
                st.at("prefill_linear_mode") == "w4a4" &&
                st.at("prefill_attention_mode") == "sage2" &&
                st.at("prefill_sparse_minference").get<bool>() == true &&
                st.at("prefill_sparse_minference_effective").get<bool>() ==
                    true &&
                st.at("prefill_cache_blend").get<bool>() == true,
            "daemon legacy nax_ane shim must not clobber other switches");
    }
    // Legacy prefill_ane_offload=true maps to prefill_ane_mode="auto"
    {
      const Json st = uds_rpc(fd, next_id++, "load",
                              {
                                  {"fake", true},
                                  {"prefill_ane_offload", true},
                              });
      check(st.at("prefill_ane_mode").get<std::string>() == "auto" &&
                st.at("prefill_ane_offload").get<bool>() == true,
            "legacy prefill_ane_offload=true must map to prefill_ane_mode=auto");
    }
    // Explicit prefill_ane_mode="off" beats legacy prefill_ane_offload=true & nax_ane
    {
      const Json st = uds_rpc(fd, next_id++, "load",
                              {
                                  {"fake", true},
                                  {"prefill_engine", "nax_ane"},
                                  {"prefill_ane_offload", true},
                                  {"prefill_ane_mode", "off"},
                              });
      check(st.at("prefill_ane_mode").get<std::string>() == "off" &&
                st.at("prefill_ane_offload").get<bool>() == false,
            "explicit prefill_ane_mode=off must override legacy flags");
    }

    (void)uds_rpc(fd, next_id++, "shutdown");
    ::close(fd);
    fd = -1;
  } catch (...) {
    if (fd >= 0) {
      ::close(fd);
    }
    ::kill(pid, SIGTERM);
    int st = 0;
    ::waitpid(pid, &st, 0);
    std::filesystem::remove(sock_path, ec);
    throw;
  }

  int status = 0;
  ::waitpid(pid, &status, 0);
  std::filesystem::remove(sock_path, ec);
  std::cout << "[spi] PASS: mseries_backendd --describe, 96-mode RPC & legacy "
               "shim\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    test_enum_helpers_and_runtime_stats_fields();
    test_all_96_orthogonal_combinations_and_hash_isolation();
    test_serve_config_toml_env_cli_and_legacy_shim();
    test_256k_kv_page_sizing_c3();
    if (argc >= 2 && argv[1] && argv[1][0] != '\0') {
      test_sidecar_daemon_describe_and_96_modes(argv[1]);
    }
    std::cout << "[prefill_switches_spi_tests] ALL TESTS PASSED!\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "[prefill_switches_spi_tests] FAIL: " << ex.what() << "\n";
    return 1;
  }
}
