#pragma once

#include "omlx_m5/types.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace omlx::m5::serve {

struct ModelServeConfig {
  std::string id;
  std::string model_dir;
  std::string provider = "auto";
  std::string artifact;
  std::string draft_dir;
  std::string draft_artifact;
  std::string mtp_artifact;
  std::string kernel_dir;
  std::string chat_template;
  std::string lora_adapter;
  std::string lora_mode = "merge";

  std::int32_t max_concurrency = 1;
  std::int32_t max_context = 32768;
  std::string kv_dtype = "bf16";
  std::string prefill_engine = "auto";
  PrefillLinearMode prefill_linear_mode = PrefillLinearMode::BFloat16;
  PrefillAttentionMode prefill_attention_mode = PrefillAttentionMode::ExactNax;
  bool prefill_sparse_minference = false;
  bool prefill_cache_blend = false;
  bool prefill_ane_offload = false;
  bool prefill_ane_offload_explicit = false;
  std::string spec_search = "auto";
  std::string sampling_contract = kDefaultServingSamplingContract;
  std::optional<bool> rolling_prefix_cache;
  std::optional<double> prefix_cache_gb;
  std::optional<std::string> batch_mode;
  std::optional<double> memory_budget_gb;
  std::uint64_t memory_budget_bytes = 0;

  SpeculativeBackend spec_backend = SpeculativeBackend::none;
  bool spec_backend_auto = false;
  std::uint32_t draft_tokens = 0;
};

struct SidecarModelOverride {
  std::optional<std::string> artifact;
  std::optional<std::string> draft_dir;
  std::optional<std::string> draft_artifact;
  std::optional<std::string> mtp_artifact;
  std::optional<std::string> kernel_dir;
  std::optional<std::string> lora_adapter;
  std::optional<std::string> lora_mode;

  std::optional<std::int32_t> max_concurrency;
  std::optional<std::int32_t> max_context;
  std::optional<std::string> kv_dtype;
  std::optional<std::string> prefill_engine;
  std::optional<PrefillLinearMode> prefill_linear_mode;
  std::optional<PrefillAttentionMode> prefill_attention_mode;
  std::optional<bool> prefill_sparse_minference;
  std::optional<bool> prefill_cache_blend;
  std::optional<bool> prefill_ane_offload;
  std::optional<std::string> spec_search;
  std::optional<std::string> sampling_contract;
  std::optional<bool> rolling_prefix_cache;
  std::optional<double> prefix_cache_gb;
  std::optional<std::string> batch_mode;
  std::optional<double> memory_budget_gb;
  std::optional<std::uint64_t> memory_budget_bytes;

  std::optional<SpeculativeBackend> spec_backend;
  std::optional<bool> spec_backend_auto;
  std::optional<std::uint32_t> draft_tokens;
};

struct ServeConfig {
  // Meta flags
  bool show_version = false;
  bool show_help = false;
  bool no_default_config = false;
  bool config_auto_discovered = false;
  std::string config_path;

  // [server]
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string api_key;
  bool cors = true;
  std::vector<std::string> providers_dirs;
  std::optional<double> memory_budget_gb;
  std::uint64_t server_memory_budget_bytes = 0;

  // [model] (single-model / default template)
  std::string model_dir;
  std::string provider = "auto";
  std::string artifact;
  std::string draft_dir;
  std::string draft_artifact;
  std::string mtp_artifact;
  std::string kernel_dir;
  std::string chat_template;
  std::string model_id;
  std::string lora_adapter;
  std::string lora_mode = "merge";

  // Tracks explicit CLI overrides so CLI > per-model ([sidecar.models] / [[models]]) > top-level TOML
  bool cli_lora_adapter_override = false;
  bool cli_lora_mode_override = false;
  bool cli_max_concurrency_override = false;
  bool cli_max_context_override = false;
  bool cli_kv_dtype_override = false;
  bool cli_spec_backend_override = false;
  bool cli_draft_tokens_override = false;
  bool cli_rolling_prefix_cache_override = false;
  bool cli_prefix_cache_gb_override = false;
  bool cli_sampling_contract_override = false;
  bool cli_artifact_override = false;
  bool cli_draft_dir_override = false;
  bool cli_draft_artifact_override = false;
  bool cli_mtp_artifact_override = false;
  bool cli_kernel_dir_override = false;
  bool cli_prefill_linear_mode_override = false;
  bool cli_prefill_attention_mode_override = false;
  bool cli_prefill_sparse_minference_override = false;
  bool cli_prefill_cache_blend_override = false;
  bool cli_prefill_ane_offload_override = false;

  // [engine]
  std::int32_t max_concurrency = 1;
  std::int32_t max_context = 32768;
  std::string kv_dtype = "bf16";
  std::string prefill_engine = "auto";
  PrefillLinearMode prefill_linear_mode = PrefillLinearMode::BFloat16;
  PrefillAttentionMode prefill_attention_mode = PrefillAttentionMode::ExactNax;
  bool prefill_sparse_minference = false;
  bool prefill_cache_blend = false;
  bool prefill_ane_offload = false;
  bool prefill_ane_offload_explicit = false;
  std::string spec_search = "auto";
  std::string sampling_contract = kDefaultServingSamplingContract;
  std::optional<bool> rolling_prefix_cache;
  std::optional<double> prefix_cache_gb;
  std::optional<std::string> batch_mode;
  std::optional<double> engine_memory_budget_gb;
  std::uint64_t memory_budget_bytes = 0;

  // [speculative]
  SpeculativeBackend spec_backend = SpeculativeBackend::none;
  bool spec_backend_auto = false;
  std::uint32_t draft_tokens = 0;

  // [sidecar.models."<id>"] per-model overrides
  std::map<std::string, SidecarModelOverride> sidecar_models;

  // [[models]] multi-model array (or repeatable --model-dir CLI entries)
  std::vector<ModelServeConfig> models;
};

// Parses TOML config file (explicit --config or auto-discovered default unless
// --no-default-config is passed) and overlays CLI arguments.
ServeConfig parse_serve_config(int argc, char **argv);

// Discovers the first existing default ninfer-metal.toml in precedence order:
// 1) $NINFER_METAL_CONFIG
// 2) ./ninfer-metal.toml
// 3) ~/.config/ninfer-metal/ninfer-metal.toml
// 4) ~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/config/ninfer-metal.toml
// 5) <exe_dir>/../config/ninfer-metal.toml
// 6) <repo_root>/omlx-extend/config/ninfer-metal.toml
std::optional<std::string> discover_default_serve_config_path();

// Loads a ninfer-metal.toml file directly into an existing config.
void load_serve_toml_file(const std::string &path, ServeConfig &config);

// Expands a leading "~/" using $HOME when present.
std::string expand_user_path(const std::string &path);

// Resolves `spec_backend_auto` (`speculative.backend = "auto"`) against
// companion artifacts (`dflash2/` -> dflash2, `mtp.miv` / embedded MTP ->
// mtp, else none).
void resolve_auto_speculative(ModelServeConfig &model_cfg);

// Human-readable name for a SpeculativeBackend (or "auto" when unresolved).
const char *spec_backend_name(SpeculativeBackend backend,
                              bool is_auto = false) noexcept;

// Discovers the metallib directory using:
// 1) explicit kernel_dir, 2) OMLX_M5_METALLIB_DIR env,
// 3) <exe_dir>/../metallib via _NSGetExecutablePath,
// 4) <exe_dir>/metallib, 5) build-dir compile-time fallback, 6) model_dir.
std::string resolve_kernel_dir(const std::string &explicit_kernel_dir,
                               const std::string &model_dir = "");

// Resolves default model_id from model_dir basename when unset.
std::string resolve_model_id(const ModelServeConfig &model_cfg);
std::string resolve_model_id(const ServeConfig &config);

// Expands ServeConfig into the concrete list of models to host.
// Returns `config.models` if populated, otherwise synthesizes a 1-element list
// from the top-level [model] / CLI fields when `model_dir` is non-empty.
std::vector<ModelServeConfig> resolve_model_configs(const ServeConfig &config);

// Resolves total server-wide UMA memory budget cap in bytes (0 = auto).
std::uint64_t resolve_total_memory_budget_bytes(const ServeConfig &config);

// Converts ModelServeConfig or ServeConfig into omlx::m5::LoadOptions.
LoadOptions to_load_options(const ModelServeConfig &model_cfg);
LoadOptions to_load_options(const ServeConfig &config);

// Formats --version output string.
std::string format_version_string();

// Formats --help usage string.
std::string format_help_string();

} // namespace omlx::m5::serve
