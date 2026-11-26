#include "serve/config.h"

#include "omlx_m5/artifact.h"
#include "omlx_m5/mtp_forward.h"

#include <nlohmann/json.hpp>
#include <tomlplusplus/toml.hpp>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#ifndef NINFER_METAL_DISPLAY_VERSION
#define NINFER_METAL_DISPLAY_VERSION "miv-v0.02"
#endif
#ifndef NINFER_METAL_CODENAME
#define NINFER_METAL_CODENAME "miv"
#endif
#ifndef NINFER_METAL_SEMVER
#define NINFER_METAL_SEMVER "0.0.2"
#endif
#ifndef NINFER_METAL_CORE_ABI
#define NINFER_METAL_CORE_ABI 1
#endif

namespace omlx::m5::serve {
namespace {

bool parse_on_off(std::string_view value, std::string_view flag_name) {
  if (value == "on" || value == "ON" || value == "1" || value == "true" ||
      value == "TRUE") {
    return true;
  }
  if (value == "off" || value == "OFF" || value == "0" || value == "false" ||
      value == "FALSE") {
    return false;
  }
  throw std::invalid_argument(std::string(flag_name) +
                              " expects on/off (or true/false/1/0)");
}

void parse_spec_backend_setting(std::string_view value,
                                SpeculativeBackend &backend, bool &is_auto) {
  if (value == "auto") {
    backend = SpeculativeBackend::none;
    is_auto = true;
    return;
  }
  is_auto = false;
  if (value == "none" || value.empty()) {
    backend = SpeculativeBackend::none;
    return;
  }
  if (value == "mtp") {
    backend = SpeculativeBackend::mtp;
    return;
  }
  if (value == "dflash") {
    backend = SpeculativeBackend::dflash;
    return;
  }
  if (value == "dflash2") {
    backend = SpeculativeBackend::dflash2;
    return;
  }
  throw std::invalid_argument(
      "unsupported --spec-backend (expected auto, none, mtp, or dflash2): " +
      std::string(value));
}

std::string parse_lora_mode(std::string_view value, std::string_view source) {
  if (value == "merge" || value == "bypass") {
    return std::string(value);
  }
  throw std::invalid_argument("unsupported " + std::string(source) +
                              " (expected 'merge' or 'bypass'): " +
                              std::string(value));
}

int parse_int_arg(std::string_view value, std::string_view flag_name) {
  int parsed = 0;
  const auto [end, ec] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (ec != std::errc{} || end != value.data() + value.size()) {
    throw std::invalid_argument(std::string(flag_name) +
                                " expects an integer, got: " +
                                std::string(value));
  }
  return parsed;
}

std::uint32_t parse_u32_arg(std::string_view value,
                            std::string_view flag_name) {
  std::uint32_t parsed = 0;
  const auto [end, ec] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (ec != std::errc{} || end != value.data() + value.size()) {
    throw std::invalid_argument(std::string(flag_name) +
                                " expects an unsigned integer, got: " +
                                std::string(value));
  }
  return parsed;
}

double parse_double_arg(std::string_view value, std::string_view flag_name) {
  std::string copy(value);
  std::size_t consumed = 0;
  double parsed = 0.0;
  try {
    parsed = std::stod(copy, &consumed);
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string(flag_name) +
                                " expects a number, got: " + copy);
  }
  if (consumed != copy.size() || !std::isfinite(parsed)) {
    throw std::invalid_argument(std::string(flag_name) +
                                " expects a finite number, got: " + copy);
  }
  return parsed;
}

bool dir_has_metallib(const std::filesystem::path &dir) {
  std::error_code ec;
  if (!std::filesystem::is_directory(dir, ec)) {
    return false;
  }
  if (std::filesystem::is_regular_file(dir / "omlx_m5_qwen_forward.metallib",
                                       ec)) {
    return true;
  }
  for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
    if (entry.is_regular_file(ec) && entry.path().extension() == ".metallib") {
      return true;
    }
  }
  return false;
}

std::optional<std::filesystem::path> current_executable_dir() {
#ifdef __APPLE__
  std::uint32_t size = 0;
  (void)_NSGetExecutablePath(nullptr, &size);
  if (size == 0) {
    return std::nullopt;
  }
  std::string buf(size, '\0');
  if (_NSGetExecutablePath(buf.data(), &size) != 0) {
    return std::nullopt;
  }
  const std::filesystem::path raw(buf.c_str());
  std::error_code ec;
  const auto canonical = std::filesystem::weakly_canonical(raw, ec);
  const auto &resolved = ec ? raw : canonical;
  if (resolved.has_parent_path()) {
    return resolved.parent_path();
  }
#endif
  return std::nullopt;
}

void read_toml_bool_or_on_off(const toml::table &tbl, std::string_view key,
                              bool &target) {
  if (const auto *node = tbl.get(key)) {
    if (const auto val = node->value<bool>()) {
      target = *val;
    } else if (const auto str = node->value<std::string>()) {
      target = parse_on_off(*str, key);
    } else {
      throw std::invalid_argument("TOML key '" + std::string(key) +
                                  "' must be boolean or on/off string");
    }
  }
}

void read_toml_optional_bool_or_on_off(const toml::table &tbl,
                                       std::string_view key,
                                       std::optional<bool> &target) {
  if (const auto *node = tbl.get(key)) {
    if (const auto val = node->value<bool>()) {
      target = *val;
    } else if (const auto str = node->value<std::string>()) {
      target = parse_on_off(*str, key);
    } else {
      throw std::invalid_argument("TOML key '" + std::string(key) +
                                  "' must be boolean or on/off string");
    }
  }
}

void read_toml_optional_double(const toml::table &tbl, std::string_view key,
                               std::optional<double> &target) {
  if (const auto *node = tbl.get(key)) {
    if (const auto d = node->value<double>()) {
      if (!std::isfinite(*d)) {
        throw std::invalid_argument("TOML key '" + std::string(key) +
                                    "' must be finite");
      }
      target = *d;
    } else if (const auto i = node->value<std::int64_t>()) {
      target = static_cast<double>(*i);
    } else {
      throw std::invalid_argument("TOML key '" + std::string(key) +
                                  "' must be a number");
    }
  }
}

void read_toml_string_or_array(const toml::table &tbl, std::string_view key,
                               std::vector<std::string> &target) {
  if (const auto *node = tbl.get(key)) {
    if (const auto s = node->value<std::string>()) {
      if (!s->empty()) {
        target.push_back(*s);
      }
    } else if (const auto *arr = node->as_array()) {
      for (const auto &elem : *arr) {
        if (const auto s = elem.value<std::string>()) {
          if (!s->empty()) {
            target.push_back(*s);
          }
        } else {
          throw std::invalid_argument("TOML key '" + std::string(key) +
                                      "' array elements must be strings");
        }
      }
    } else {
      throw std::invalid_argument("TOML key '" + std::string(key) +
                                  "' must be a string or array of strings");
    }
  }
}

std::optional<std::string> read_toml_optional_string(const toml::table &tbl,
                                                     std::string_view key) {
  if (const auto *node = tbl.get(key)) {
    if (const auto s = node->value<std::string>()) {
      return *s;
    }
    throw std::invalid_argument("TOML key '" + std::string(key) +
                                "' must be a string");
  }
  return std::nullopt;
}

std::uint64_t gb_to_bytes(double gb) {
  if (!(gb > 0.0) || !std::isfinite(gb)) {
    return 0;
  }
  return static_cast<std::uint64_t>(gb * static_cast<double>(1ULL << 30));
}

std::string derive_model_id_from_dir(const std::string &model_dir) {
  if (!model_dir.empty()) {
    std::filesystem::path p(model_dir);
    const auto normalized = p.lexically_normal();
    std::string name = normalized.filename().string();
    if (name.empty() || name == ".") {
      name = normalized.parent_path().filename().string();
    }
    if (!name.empty()) {
      return name;
    }
  }
  return "ninfer-metal-model";
}

void validate_model_config(const ModelServeConfig &model_cfg) {
  if (model_cfg.max_concurrency < 1 || model_cfg.max_concurrency > 8) {
    throw std::invalid_argument("--max-concurrency must be in [1, 8]");
  }
  if (model_cfg.max_context <= 0) {
    throw std::invalid_argument("--max-context must be positive");
  }
  if (model_cfg.kv_dtype != "bf16" && model_cfg.kv_dtype != "affine4_g128") {
    throw std::invalid_argument("--kv-dtype must be bf16 or affine4_g128");
  }
  if (model_cfg.lora_mode != "merge" && model_cfg.lora_mode != "bypass") {
    throw std::invalid_argument(
        "--lora-mode must be 'merge' or 'bypass', got: " +
        model_cfg.lora_mode);
  }
  if (model_cfg.prefix_cache_gb.has_value() &&
      (*model_cfg.prefix_cache_gb < 0.0 || *model_cfg.prefix_cache_gb > 128.0)) {
    throw std::invalid_argument("--prefix-cache-gb must be in [0, 128]");
  }
  if (model_cfg.memory_budget_gb.has_value() &&
      (*model_cfg.memory_budget_gb < 0.0 ||
       !std::isfinite(*model_cfg.memory_budget_gb))) {
    throw std::invalid_argument("memory_budget_gb must be non-negative");
  }
}

void validate_serve_config(const ServeConfig &config) {
  if (config.port < 0 || config.port > 65535) {
    throw std::invalid_argument("--port must be in [0, 65535]");
  }
  if (config.max_concurrency < 1 || config.max_concurrency > 8) {
    throw std::invalid_argument("--max-concurrency must be in [1, 8]");
  }
  if (config.max_context <= 0) {
    throw std::invalid_argument("--max-context must be positive");
  }
  if (config.kv_dtype != "bf16" && config.kv_dtype != "affine4_g128") {
    throw std::invalid_argument("--kv-dtype must be bf16 or affine4_g128");
  }
  if (config.lora_mode != "merge" && config.lora_mode != "bypass") {
    throw std::invalid_argument(
        "--lora-mode must be 'merge' or 'bypass', got: " + config.lora_mode);
  }
  if (config.prefix_cache_gb.has_value() &&
      (*config.prefix_cache_gb < 0.0 || *config.prefix_cache_gb > 128.0)) {
    throw std::invalid_argument("--prefix-cache-gb must be in [0, 128]");
  }
  if (config.memory_budget_gb.has_value() &&
      (*config.memory_budget_gb < 0.0 ||
       !std::isfinite(*config.memory_budget_gb))) {
    throw std::invalid_argument("--memory-budget-gb must be non-negative");
  }
  if (config.engine_memory_budget_gb.has_value() &&
      (*config.engine_memory_budget_gb < 0.0 ||
       !std::isfinite(*config.engine_memory_budget_gb))) {
    throw std::invalid_argument("engine.memory_budget_gb must be non-negative");
  }
  for (const auto &[id, ov] : config.sidecar_models) {
    (void)id;
    if (ov.lora_mode.has_value() && *ov.lora_mode != "merge" &&
        *ov.lora_mode != "bypass") {
      throw std::invalid_argument(
          "sidecar.models lora_mode must be 'merge' or 'bypass', got: " +
          *ov.lora_mode);
    }
    if (ov.max_concurrency.has_value() &&
        (*ov.max_concurrency < 1 || *ov.max_concurrency > 8)) {
      throw std::invalid_argument("--max-concurrency must be in [1, 8]");
    }
    if (ov.max_context.has_value() && *ov.max_context <= 0) {
      throw std::invalid_argument("--max-context must be positive");
    }
    if (ov.kv_dtype.has_value() && *ov.kv_dtype != "bf16" &&
        *ov.kv_dtype != "affine4_g128") {
      throw std::invalid_argument("--kv-dtype must be bf16 or affine4_g128");
    }
  }
  for (const auto &m : config.models) {
    validate_model_config(m);
  }
}

const SidecarModelOverride *
find_sidecar_model_override(const ServeConfig &config,
                            const ModelServeConfig &m) {
  if (config.sidecar_models.empty()) {
    return nullptr;
  }
  if (!m.id.empty()) {
    if (const auto it = config.sidecar_models.find(m.id);
        it != config.sidecar_models.end()) {
      return &it->second;
    }
  }
  if (!m.model_dir.empty()) {
    const std::string derived =
        derive_model_id_from_dir(expand_user_path(m.model_dir));
    if (const auto it = config.sidecar_models.find(derived);
        it != config.sidecar_models.end()) {
      return &it->second;
    }
  }
  return nullptr;
}

void apply_sidecar_model_override(ModelServeConfig &m,
                                  const ServeConfig &config,
                                  bool respect_cli_overrides) {
  const SidecarModelOverride *ov = find_sidecar_model_override(config, m);
  if (ov == nullptr) {
    return;
  }
  if (ov->artifact.has_value() &&
      !(respect_cli_overrides && config.cli_artifact_override)) {
    m.artifact = *ov->artifact;
  }
  if (ov->draft_dir.has_value() &&
      !(respect_cli_overrides && config.cli_draft_dir_override)) {
    m.draft_dir = *ov->draft_dir;
  }
  if (ov->draft_artifact.has_value() &&
      !(respect_cli_overrides && config.cli_draft_artifact_override)) {
    m.draft_artifact = *ov->draft_artifact;
  }
  if (ov->mtp_artifact.has_value() &&
      !(respect_cli_overrides && config.cli_mtp_artifact_override)) {
    m.mtp_artifact = *ov->mtp_artifact;
  }
  if (ov->kernel_dir.has_value() &&
      !(respect_cli_overrides && config.cli_kernel_dir_override)) {
    m.kernel_dir = *ov->kernel_dir;
  }
  if (ov->lora_adapter.has_value() &&
      !(respect_cli_overrides && config.cli_lora_adapter_override)) {
    m.lora_adapter = *ov->lora_adapter;
  }
  if (ov->lora_mode.has_value() &&
      !(respect_cli_overrides && config.cli_lora_mode_override)) {
    m.lora_mode = *ov->lora_mode;
  }
  if (ov->max_concurrency.has_value() &&
      !(respect_cli_overrides && config.cli_max_concurrency_override)) {
    m.max_concurrency = *ov->max_concurrency;
  }
  if (ov->max_context.has_value() &&
      !(respect_cli_overrides && config.cli_max_context_override)) {
    m.max_context = *ov->max_context;
  }
  if (ov->kv_dtype.has_value() &&
      !(respect_cli_overrides && config.cli_kv_dtype_override)) {
    m.kv_dtype = *ov->kv_dtype;
  }
  if (ov->prefill_engine.has_value()) {
    m.prefill_engine = *ov->prefill_engine;
  }
  if (ov->spec_search.has_value()) {
    m.spec_search = *ov->spec_search;
  }
  if (ov->sampling_contract.has_value() &&
      !(respect_cli_overrides && config.cli_sampling_contract_override)) {
    m.sampling_contract = *ov->sampling_contract;
  }
  if (ov->rolling_prefix_cache.has_value() &&
      !(respect_cli_overrides && config.cli_rolling_prefix_cache_override)) {
    m.rolling_prefix_cache = *ov->rolling_prefix_cache;
  }
  if (ov->prefix_cache_gb.has_value() &&
      !(respect_cli_overrides && config.cli_prefix_cache_gb_override)) {
    m.prefix_cache_gb = *ov->prefix_cache_gb;
  }
  if (ov->batch_mode.has_value()) {
    m.batch_mode = *ov->batch_mode;
  }
  if (ov->memory_budget_gb.has_value()) {
    m.memory_budget_gb = *ov->memory_budget_gb;
    m.memory_budget_bytes = 0;
  }
  if (ov->memory_budget_bytes.has_value()) {
    m.memory_budget_bytes = *ov->memory_budget_bytes;
  }
  if (ov->spec_backend.has_value() &&
      !(respect_cli_overrides && config.cli_spec_backend_override)) {
    m.spec_backend = *ov->spec_backend;
    m.spec_backend_auto = ov->spec_backend_auto.value_or(false);
  }
  if (ov->draft_tokens.has_value() &&
      !(respect_cli_overrides && config.cli_draft_tokens_override)) {
    m.draft_tokens = *ov->draft_tokens;
  }
}

ModelServeConfig default_model_from_serve_config(const ServeConfig &config,
                                                 bool apply_sidecar_override = true) {
  ModelServeConfig m;
  m.id = config.model_id;
  m.model_dir = config.model_dir;
  m.provider = config.provider;
  m.artifact = config.artifact;
  m.draft_dir = config.draft_dir;
  m.draft_artifact = config.draft_artifact;
  m.mtp_artifact = config.mtp_artifact;
  m.kernel_dir = config.kernel_dir;
  m.chat_template = config.chat_template;
  m.lora_adapter = config.lora_adapter;
  m.lora_mode = config.lora_mode;
  m.max_concurrency = config.max_concurrency;
  m.max_context = config.max_context;
  m.kv_dtype = config.kv_dtype;
  m.prefill_engine = config.prefill_engine;
  m.spec_search = config.spec_search;
  m.sampling_contract = config.sampling_contract;
  m.rolling_prefix_cache = config.rolling_prefix_cache;
  m.prefix_cache_gb = config.prefix_cache_gb;
  m.batch_mode = config.batch_mode;
  m.memory_budget_gb = config.engine_memory_budget_gb;
  m.memory_budget_bytes = config.memory_budget_bytes;
  m.spec_backend = config.spec_backend;
  m.spec_backend_auto = config.spec_backend_auto;
  m.draft_tokens = config.draft_tokens;
  if (apply_sidecar_override) {
    apply_sidecar_model_override(m, config, /*respect_cli_overrides=*/true);
  }
  return m;
}

void expand_model_paths(ModelServeConfig &m) {
  m.model_dir = expand_user_path(m.model_dir);
  m.artifact = expand_user_path(m.artifact);
  m.draft_dir = expand_user_path(m.draft_dir);
  m.draft_artifact = expand_user_path(m.draft_artifact);
  m.mtp_artifact = expand_user_path(m.mtp_artifact);
  m.kernel_dir = expand_user_path(m.kernel_dir);
  m.chat_template = expand_user_path(m.chat_template);
  m.lora_adapter = expand_user_path(m.lora_adapter);
}

void expand_sidecar_override_paths(SidecarModelOverride &ov) {
  if (ov.artifact.has_value()) {
    ov.artifact = expand_user_path(*ov.artifact);
  }
  if (ov.draft_dir.has_value()) {
    ov.draft_dir = expand_user_path(*ov.draft_dir);
  }
  if (ov.draft_artifact.has_value()) {
    ov.draft_artifact = expand_user_path(*ov.draft_artifact);
  }
  if (ov.mtp_artifact.has_value()) {
    ov.mtp_artifact = expand_user_path(*ov.mtp_artifact);
  }
  if (ov.kernel_dir.has_value()) {
    ov.kernel_dir = expand_user_path(*ov.kernel_dir);
  }
  if (ov.lora_adapter.has_value()) {
    ov.lora_adapter = expand_user_path(*ov.lora_adapter);
  }
}

void expand_serve_config_paths(ServeConfig &config) {
  config.config_path = expand_user_path(config.config_path);
  for (auto &dir : config.providers_dirs) {
    dir = expand_user_path(dir);
  }
  config.model_dir = expand_user_path(config.model_dir);
  config.artifact = expand_user_path(config.artifact);
  config.draft_dir = expand_user_path(config.draft_dir);
  config.draft_artifact = expand_user_path(config.draft_artifact);
  config.mtp_artifact = expand_user_path(config.mtp_artifact);
  config.kernel_dir = expand_user_path(config.kernel_dir);
  config.chat_template = expand_user_path(config.chat_template);
  config.lora_adapter = expand_user_path(config.lora_adapter);
  for (auto &[_, ov] : config.sidecar_models) {
    expand_sidecar_override_paths(ov);
  }
  for (auto &m : config.models) {
    expand_model_paths(m);
  }
}

std::optional<std::filesystem::path>
find_omlxm5_in_dir(const std::filesystem::path &dir) {
  std::error_code ec;
  if (!std::filesystem::is_directory(dir, ec)) {
    return std::nullopt;
  }
  for (const char *preferred :
       {"model.miv", "draft.miv", "model.omlxm5", "draft.omlxm5"}) {
    const auto cand = dir / preferred;
    if (std::filesystem::is_regular_file(cand, ec)) {
      return cand;
    }
  }
  for (const char *ext : {".miv", ".omlxm5"}) {
    std::vector<std::filesystem::path> matches;
    for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
      if (!entry.is_regular_file(ec) || entry.path().extension() != ext) {
        continue;
      }
      const auto fname = entry.path().filename().string();
      if (!fname.empty() && fname.front() == '.') {
        continue;
      }
      if (fname == "mtp.miv" || fname == "mtp.omlxm5") {
        continue;
      }
      matches.push_back(entry.path());
    }
    if (!matches.empty()) {
      std::sort(matches.begin(), matches.end());
      return matches.front();
    }
  }
  return std::nullopt;
}

std::filesystem::path
resolve_draft_config_dir(const std::filesystem::path &base_dir) {
  std::error_code ec;
  if (std::filesystem::is_regular_file(base_dir / "config.json", ec)) {
    return base_dir;
  }
  const auto source_dir = base_dir / "source";
  if (std::filesystem::is_regular_file(source_dir / "config.json", ec)) {
    return source_dir;
  }
  return base_dir;
}

bool config_json_has_mtp(const std::filesystem::path &config_path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(config_path, ec)) {
    return false;
  }
  std::ifstream in(config_path, std::ios::in | std::ios::binary);
  if (!in.is_open()) {
    return false;
  }
  try {
    const auto doc = nlohmann::json::parse(in);
    if (!doc.is_object()) {
      return false;
    }
    auto check_obj = [](const nlohmann::json &obj) -> bool {
      if (obj.contains("mtp_num_hidden_layers") &&
          obj.at("mtp_num_hidden_layers").is_number_integer()) {
        return obj.at("mtp_num_hidden_layers").get<std::int64_t>() > 0;
      }
      if (obj.contains("num_nextn_predict_layers") &&
          obj.at("num_nextn_predict_layers").is_number_integer()) {
        return obj.at("num_nextn_predict_layers").get<std::int64_t>() > 0;
      }
      return false;
    };
    if (check_obj(doc)) {
      return true;
    }
    if (doc.contains("text_config") && doc.at("text_config").is_object()) {
      return check_obj(doc.at("text_config"));
    }
  } catch (...) {
    return false;
  }
  return false;
}

std::optional<std::string>
find_existing_config_file(const std::filesystem::path &candidate) {
  if (candidate.empty()) {
    return std::nullopt;
  }
  std::error_code ec;
  if (!std::filesystem::is_regular_file(candidate, ec)) {
    return std::nullopt;
  }
  const auto canonical = std::filesystem::weakly_canonical(candidate, ec);
  return (ec ? candidate : canonical).string();
}

SidecarModelOverride parse_sidecar_model_override(const toml::table &entry,
                                                  std::string_view section) {
  SidecarModelOverride ov;
  if (const auto v = read_toml_optional_string(entry, "artifact")) {
    ov.artifact = *v;
  } else if (const auto v2 = read_toml_optional_string(entry, "artifact_path")) {
    ov.artifact = *v2;
  }
  if (const auto v = read_toml_optional_string(entry, "draft_dir")) {
    ov.draft_dir = *v;
  }
  if (const auto v = read_toml_optional_string(entry, "draft_artifact")) {
    ov.draft_artifact = *v;
  } else if (const auto v2 =
                 read_toml_optional_string(entry, "draft_artifact_path")) {
    ov.draft_artifact = *v2;
  }
  if (const auto v = read_toml_optional_string(entry, "mtp_artifact")) {
    ov.mtp_artifact = *v;
  } else if (const auto v2 =
                 read_toml_optional_string(entry, "mtp_artifact_path")) {
    ov.mtp_artifact = *v2;
  }
  if (const auto v = read_toml_optional_string(entry, "metallib_dir")) {
    ov.kernel_dir = *v;
  } else if (const auto v2 = read_toml_optional_string(entry, "kernel_dir")) {
    ov.kernel_dir = *v2;
  }
  if (const auto v = read_toml_optional_string(entry, "lora_adapter")) {
    ov.lora_adapter = *v;
  } else if (const auto v2 = read_toml_optional_string(entry, "lora")) {
    ov.lora_adapter = *v2;
  }
  if (const auto v = read_toml_optional_string(entry, "lora_mode")) {
    ov.lora_mode = parse_lora_mode(*v, std::string(section) + ".lora_mode");
  }
  if (const auto v = entry["max_concurrency"].value<std::int64_t>()) {
    ov.max_concurrency = static_cast<std::int32_t>(*v);
  }
  if (const auto v = entry["max_context"].value<std::int64_t>()) {
    ov.max_context = static_cast<std::int32_t>(*v);
  }
  if (const auto v = read_toml_optional_string(entry, "kv_dtype")) {
    ov.kv_dtype = *v;
  }
  if (const auto v = read_toml_optional_string(entry, "prefill_engine")) {
    ov.prefill_engine = *v;
  }
  if (const auto v = read_toml_optional_string(entry, "spec_search")) {
    ov.spec_search = *v;
  }
  if (const auto v = read_toml_optional_string(entry, "sampling_contract")) {
    ov.sampling_contract = *v;
  }
  read_toml_optional_bool_or_on_off(entry, "rolling_prefix_cache",
                                    ov.rolling_prefix_cache);
  read_toml_optional_double(entry, "prefix_cache_gb", ov.prefix_cache_gb);
  if (const auto v = read_toml_optional_string(entry, "batch_mode")) {
    ov.batch_mode = *v;
  }
  read_toml_optional_double(entry, "memory_budget_gb", ov.memory_budget_gb);
  if (const auto v = entry["memory_budget_bytes"].value<std::int64_t>()) {
    if (*v < 0) {
      throw std::invalid_argument(std::string(section) +
                                  ".memory_budget_bytes must be non-negative");
    }
    ov.memory_budget_bytes = static_cast<std::uint64_t>(*v);
  }
  if (const auto v = read_toml_optional_string(entry, "spec_backend")) {
    SpeculativeBackend sb = SpeculativeBackend::none;
    bool sb_auto = false;
    parse_spec_backend_setting(*v, sb, sb_auto);
    ov.spec_backend = sb;
    ov.spec_backend_auto = sb_auto;
  } else if (const auto v2 =
                 read_toml_optional_string(entry, "speculative_backend")) {
    SpeculativeBackend sb = SpeculativeBackend::none;
    bool sb_auto = false;
    parse_spec_backend_setting(*v2, sb, sb_auto);
    ov.spec_backend = sb;
    ov.spec_backend_auto = sb_auto;
  }
  if (const auto v = entry["draft_tokens"].value<std::int64_t>()) {
    if (*v < 0) {
      throw std::invalid_argument(std::string(section) +
                                  ".draft_tokens must be non-negative");
    }
    ov.draft_tokens = static_cast<std::uint32_t>(*v);
  }
  return ov;
}

} // namespace

std::string expand_user_path(const std::string &path) {
  if (path.empty() || path.front() != '~') {
    return path;
  }
  if (path.size() == 1 || path[1] == '/') {
    if (const char *home = std::getenv("HOME"); home && *home) {
      std::string base(home);
      if (path.size() > 1 && base.size() > 1 && base.back() == '/') {
        base.pop_back();
      }
      return base + path.substr(1);
    }
  }
  return path;
}

std::optional<std::string> discover_default_serve_config_path() {
  // 1) $NINFER_METAL_CONFIG
  if (const char *env = std::getenv("NINFER_METAL_CONFIG"); env && *env) {
    if (auto found = find_existing_config_file(
            std::filesystem::path(expand_user_path(env)));
        found.has_value()) {
      return found;
    }
  }

  // 2) ./ninfer-metal.toml
  if (auto found =
          find_existing_config_file(std::filesystem::path("ninfer-metal.toml"));
      found.has_value()) {
    return found;
  }

  // 3) ~/.config/ninfer-metal/ninfer-metal.toml
  if (auto found = find_existing_config_file(std::filesystem::path(
          expand_user_path("~/.config/ninfer-metal/ninfer-metal.toml")));
      found.has_value()) {
    return found;
  }

  // 4) ~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/config/ninfer-metal.toml
  if (auto found = find_existing_config_file(
          std::filesystem::path(expand_user_path(
              "~/Library/Application Support/oMLX/Providers/io.ninfer.metal/"
              "current/config/ninfer-metal.toml")));
      found.has_value()) {
    return found;
  }

  // 5) <exe_dir>/../config/ninfer-metal.toml (relocatable release tarball)
  const auto exe_dir = current_executable_dir();
  if (exe_dir.has_value()) {
    if (auto found = find_existing_config_file(exe_dir->parent_path() /
                                               "config" / "ninfer-metal.toml");
        found.has_value()) {
      return found;
    }

    // 6) <repo_root>/omlx-extend/config/ninfer-metal.toml (build dir inside checkout)
    std::filesystem::path cursor = *exe_dir;
    for (int depth = 0; depth < 6 && !cursor.empty(); ++depth) {
      if (auto found = find_existing_config_file(cursor / "omlx-extend" /
                                                 "config" / "ninfer-metal.toml");
          found.has_value()) {
        return found;
      }
      const auto parent = cursor.parent_path();
      if (parent == cursor) {
        break;
      }
      cursor = parent;
    }
  }
#ifdef OMLX_M5_DEFAULT_METALLIB_DIR
  else {
    std::filesystem::path cursor(OMLX_M5_DEFAULT_METALLIB_DIR);
    for (int depth = 0; depth < 6 && !cursor.empty(); ++depth) {
      if (auto found = find_existing_config_file(cursor / "omlx-extend" /
                                                 "config" / "ninfer-metal.toml");
          found.has_value()) {
        return found;
      }
      const auto parent = cursor.parent_path();
      if (parent == cursor) {
        break;
      }
      cursor = parent;
    }
  }
#endif

  return std::nullopt;
}

void resolve_auto_speculative(ModelServeConfig &model_cfg) {
  if (!model_cfg.spec_backend_auto) {
    return;
  }
  expand_model_paths(model_cfg);
  std::error_code ec;

  // 1) DFlash2 companion artifact takes priority when present.
  if (!model_cfg.draft_artifact.empty() &&
      std::filesystem::is_regular_file(model_cfg.draft_artifact, ec)) {
    model_cfg.spec_backend = SpeculativeBackend::dflash2;
    if (model_cfg.draft_dir.empty()) {
      model_cfg.draft_dir =
          resolve_draft_config_dir(
              std::filesystem::path(model_cfg.draft_artifact).parent_path())
              .string();
    }
    model_cfg.spec_backend_auto = false;
    return;
  }
  if (!model_cfg.draft_dir.empty()) {
    if (const auto art = find_omlxm5_in_dir(model_cfg.draft_dir);
        art.has_value()) {
      model_cfg.spec_backend = SpeculativeBackend::dflash2;
      model_cfg.draft_artifact = art->string();
      model_cfg.spec_backend_auto = false;
      return;
    }
  }
  if (!model_cfg.model_dir.empty()) {
    const std::filesystem::path model_dir(model_cfg.model_dir);
    const auto dflash_dir = model_dir / "dflash2";
    if (std::filesystem::is_directory(dflash_dir, ec)) {
      if (const auto art = find_omlxm5_in_dir(dflash_dir); art.has_value()) {
        model_cfg.spec_backend = SpeculativeBackend::dflash2;
        model_cfg.draft_artifact = art->string();
        if (model_cfg.draft_dir.empty()) {
          model_cfg.draft_dir = resolve_draft_config_dir(dflash_dir).string();
        }
        model_cfg.spec_backend_auto = false;
        return;
      }
    }
  }

  // 2) MTP external artifact (`mtp.miv` / `mtp.omlxm5`) or embedded MTP weights.
  if (!model_cfg.mtp_artifact.empty() &&
      std::filesystem::is_regular_file(model_cfg.mtp_artifact, ec)) {
    model_cfg.spec_backend = SpeculativeBackend::mtp;
    model_cfg.spec_backend_auto = false;
    return;
  }
  if (!model_cfg.model_dir.empty()) {
    const std::filesystem::path model_dir(model_cfg.model_dir);
    for (const char *mtp_name : {"mtp.miv", "mtp.omlxm5"}) {
      const auto mtp_file = model_dir / mtp_name;
      if (std::filesystem::is_regular_file(mtp_file, ec)) {
        model_cfg.spec_backend = SpeculativeBackend::mtp;
        model_cfg.mtp_artifact = mtp_file.string();
        model_cfg.spec_backend_auto = false;
        return;
      }
    }

    const std::filesystem::path default_art =
        std::filesystem::is_regular_file(model_dir / "model.miv", ec) ||
                !std::filesystem::is_regular_file(model_dir / "model.omlxm5", ec)
            ? (model_dir / "model.miv")
            : (model_dir / "model.omlxm5");
    const std::filesystem::path target_art =
        !model_cfg.artifact.empty() ? std::filesystem::path(model_cfg.artifact)
                                    : default_art;
    bool inspected_real_artifact = false;
    if (std::filesystem::is_regular_file(target_art, ec)) {
      try {
        const auto art = ArtifactModel::open(target_art, false);
        inspected_real_artifact = true;
        if (has_mtp_weights(art)) {
          model_cfg.spec_backend = SpeculativeBackend::mtp;
          model_cfg.spec_backend_auto = false;
          return;
        }
      } catch (...) {
        // Synthetic placeholder artifact; fall back to config.json inspection.
      }
    }
    if (!inspected_real_artifact &&
        config_json_has_mtp(model_dir / "config.json")) {
      model_cfg.spec_backend = SpeculativeBackend::mtp;
      model_cfg.spec_backend_auto = false;
      return;
    }
  }

  // 3) No speculative companion found -> standard AR (`none`).
  model_cfg.spec_backend = SpeculativeBackend::none;
  model_cfg.spec_backend_auto = false;
}

const char *spec_backend_name(SpeculativeBackend backend,
                              bool is_auto) noexcept {
  if (is_auto) {
    return "auto";
  }
  switch (backend) {
  case SpeculativeBackend::none:
    return "none";
  case SpeculativeBackend::mtp:
    return "mtp";
  case SpeculativeBackend::dflash:
    return "dflash";
  case SpeculativeBackend::dflash2:
    return "dflash2";
  }
  return "none";
}

void load_serve_toml_file(const std::string &path, ServeConfig &config) {
  const std::string expanded_path = expand_user_path(path);
  toml::table tbl;
  try {
    tbl = toml::parse_file(expanded_path);
  } catch (const toml::parse_error &err) {
    throw std::runtime_error("failed to parse TOML config '" + expanded_path +
                             "': " + std::string(err.description()));
  }

  if (const auto *server = tbl["server"].as_table()) {
    if (const auto v = (*server)["host"].value<std::string>()) {
      config.host = *v;
    }
    if (const auto v = (*server)["port"].value<std::int64_t>()) {
      config.port = static_cast<int>(*v);
    }
    if (const auto v = (*server)["api_key"].value<std::string>()) {
      config.api_key = *v;
    }
    read_toml_bool_or_on_off(*server, "cors", config.cors);
    read_toml_string_or_array(*server, "providers_dir", config.providers_dirs);
    read_toml_string_or_array(*server, "providers_dirs", config.providers_dirs);
    read_toml_optional_double(*server, "memory_budget_gb",
                              config.memory_budget_gb);
    if (const auto v = (*server)["memory_budget_bytes"].value<std::int64_t>()) {
      if (*v < 0) {
        throw std::invalid_argument(
            "TOML server.memory_budget_bytes must be non-negative");
      }
      config.server_memory_budget_bytes = static_cast<std::uint64_t>(*v);
    }
  }

  if (const auto *model = tbl["model"].as_table()) {
    if (const auto v = (*model)["model_dir"].value<std::string>()) {
      config.model_dir = *v;
    } else if (const auto v2 = (*model)["dir"].value<std::string>()) {
      config.model_dir = *v2;
    }
    if (const auto v = (*model)["provider"].value<std::string>()) {
      config.provider = *v;
    }
    if (const auto v = (*model)["artifact"].value<std::string>()) {
      config.artifact = *v;
    } else if (const auto v2 = (*model)["artifact_path"].value<std::string>()) {
      config.artifact = *v2;
    }
    if (const auto v = (*model)["draft_dir"].value<std::string>()) {
      config.draft_dir = *v;
    }
    if (const auto v = (*model)["draft_artifact"].value<std::string>()) {
      config.draft_artifact = *v;
    } else if (const auto v2 =
                   (*model)["draft_artifact_path"].value<std::string>()) {
      config.draft_artifact = *v2;
    }
    if (const auto v = (*model)["mtp_artifact"].value<std::string>()) {
      config.mtp_artifact = *v;
    } else if (const auto v2 =
                   (*model)["mtp_artifact_path"].value<std::string>()) {
      config.mtp_artifact = *v2;
    }
    if (const auto v = (*model)["metallib_dir"].value<std::string>()) {
      config.kernel_dir = *v;
    } else if (const auto v2 = (*model)["kernel_dir"].value<std::string>()) {
      config.kernel_dir = *v2;
    }
    if (const auto v = (*model)["chat_template"].value<std::string>()) {
      config.chat_template = *v;
    }
    if (const auto v = (*model)["model_id"].value<std::string>()) {
      config.model_id = *v;
    } else if (const auto v2 = (*model)["id"].value<std::string>()) {
      config.model_id = *v2;
    }
    if (const auto v = read_toml_optional_string(*model, "lora_adapter")) {
      config.lora_adapter = *v;
    } else if (const auto v2 = read_toml_optional_string(*model, "lora")) {
      config.lora_adapter = *v2;
    }
    if (const auto v = read_toml_optional_string(*model, "lora_mode")) {
      config.lora_mode = parse_lora_mode(*v, "TOML model.lora_mode");
    }
  }

  if (const auto *engine = tbl["engine"].as_table()) {
    if (const auto v = (*engine)["max_concurrency"].value<std::int64_t>()) {
      config.max_concurrency = static_cast<std::int32_t>(*v);
    }
    if (const auto v = (*engine)["max_context"].value<std::int64_t>()) {
      config.max_context = static_cast<std::int32_t>(*v);
    }
    if (const auto v = (*engine)["kv_dtype"].value<std::string>()) {
      config.kv_dtype = *v;
    }
    if (const auto v = (*engine)["prefill_engine"].value<std::string>()) {
      config.prefill_engine = *v;
    }
    if (const auto v = (*engine)["spec_search"].value<std::string>()) {
      config.spec_search = *v;
    }
    if (const auto v = (*engine)["sampling_contract"].value<std::string>()) {
      config.sampling_contract = *v;
    }
    read_toml_optional_bool_or_on_off(*engine, "rolling_prefix_cache",
                                      config.rolling_prefix_cache);
    read_toml_optional_double(*engine, "prefix_cache_gb",
                              config.prefix_cache_gb);
    read_toml_optional_double(*engine, "memory_budget_gb",
                              config.engine_memory_budget_gb);
    if (const auto v = (*engine)["batch_mode"].value<std::string>()) {
      config.batch_mode = *v;
    }
    if (const auto v = (*engine)["memory_budget_bytes"].value<std::int64_t>()) {
      if (*v < 0) {
        throw std::invalid_argument(
            "TOML engine.memory_budget_bytes must be non-negative");
      }
      config.memory_budget_bytes = static_cast<std::uint64_t>(*v);
    }
    if (const auto v = (*engine)["spec_backend"].value<std::string>()) {
      parse_spec_backend_setting(*v, config.spec_backend,
                                 config.spec_backend_auto);
    }
    if (const auto v = (*engine)["metallib_dir"].value<std::string>()) {
      config.kernel_dir = *v;
    } else if (const auto v2 = (*engine)["kernel_dir"].value<std::string>()) {
      config.kernel_dir = *v2;
    }
    if (const auto v = read_toml_optional_string(*engine, "lora_adapter")) {
      if (!v->empty() || config.lora_adapter.empty()) {
        config.lora_adapter = *v;
      }
    } else if (const auto v2 = read_toml_optional_string(*engine, "lora")) {
      if (!v2->empty() || config.lora_adapter.empty()) {
        config.lora_adapter = *v2;
      }
    }
    if (const auto v = read_toml_optional_string(*engine, "lora_mode")) {
      config.lora_mode = parse_lora_mode(*v, "TOML engine.lora_mode");
    }
  }

  if (const auto *spec = tbl["speculative"].as_table()) {
    if (const auto v = (*spec)["backend"].value<std::string>()) {
      parse_spec_backend_setting(*v, config.spec_backend,
                                 config.spec_backend_auto);
    }
    if (const auto v = (*spec)["draft_tokens"].value<std::int64_t>()) {
      if (*v < 0) {
        throw std::invalid_argument(
            "TOML speculative.draft_tokens must be non-negative");
      }
      config.draft_tokens = static_cast<std::uint32_t>(*v);
    }
    if (const auto v = (*spec)["draft_dir"].value<std::string>()) {
      config.draft_dir = *v;
    }
    if (const auto v = (*spec)["draft_artifact"].value<std::string>()) {
      config.draft_artifact = *v;
    } else if (const auto v2 =
                   (*spec)["draft_artifact_path"].value<std::string>()) {
      config.draft_artifact = *v2;
    }
  }

  if (const auto *sidecar = tbl["sidecar"].as_table()) {
    if (const auto v = read_toml_optional_string(*sidecar, "lora_adapter")) {
      if (!v->empty() || config.lora_adapter.empty()) {
        config.lora_adapter = *v;
      }
    } else if (const auto v2 = read_toml_optional_string(*sidecar, "lora")) {
      if (!v2->empty() || config.lora_adapter.empty()) {
        config.lora_adapter = *v2;
      }
    }
    if (const auto v = read_toml_optional_string(*sidecar, "lora_mode")) {
      config.lora_mode = parse_lora_mode(*v, "TOML sidecar.lora_mode");
    }
    if (const auto v = (*sidecar)["max_concurrency"].value<std::int64_t>()) {
      config.max_concurrency = static_cast<std::int32_t>(*v);
    }
    if (const auto v = (*sidecar)["max_context"].value<std::int64_t>()) {
      config.max_context = static_cast<std::int32_t>(*v);
    }
    if (const auto v = (*sidecar)["kv_dtype"].value<std::string>()) {
      config.kv_dtype = *v;
    }
    if (const auto v = (*sidecar)["prefill_engine"].value<std::string>()) {
      config.prefill_engine = *v;
    }
    if (const auto v = (*sidecar)["spec_search"].value<std::string>()) {
      config.spec_search = *v;
    }
    if (const auto v = (*sidecar)["sampling_contract"].value<std::string>()) {
      config.sampling_contract = *v;
    }
    read_toml_optional_bool_or_on_off(*sidecar, "rolling_prefix_cache",
                                      config.rolling_prefix_cache);
    read_toml_optional_double(*sidecar, "prefix_cache_gb",
                              config.prefix_cache_gb);
    if (const auto v = (*sidecar)["batch_mode"].value<std::string>()) {
      config.batch_mode = *v;
    }
    if (const auto v = (*sidecar)["memory_budget_bytes"].value<std::int64_t>()) {
      if (*v < 0) {
        throw std::invalid_argument(
            "TOML sidecar.memory_budget_bytes must be non-negative");
      }
      config.memory_budget_bytes = static_cast<std::uint64_t>(*v);
    }
    if (const auto *sidecar_models = (*sidecar)["models"].as_table()) {
      for (const auto &[k, node] : *sidecar_models) {
        const auto *entry = node.as_table();
        if (entry == nullptr) {
          throw std::invalid_argument("TOML [sidecar.models.\"" +
                                      std::string(k.str()) +
                                      "\"] must be a table");
        }
        SidecarModelOverride ov = parse_sidecar_model_override(
            *entry, "TOML sidecar.models." + std::string(k.str()));
        config.sidecar_models[std::string(k.str())] = std::move(ov);
      }
    }
  }

  if (const auto *models_arr = tbl["models"].as_array()) {
    for (const auto &item : *models_arr) {
      const auto *entry = item.as_table();
      if (entry == nullptr) {
        throw std::invalid_argument("TOML [[models]] elements must be tables");
      }
      ModelServeConfig m = default_model_from_serve_config(
          config, /*apply_sidecar_override=*/false);
      m.id.clear();
      m.model_dir.clear();
      m.artifact.clear();
      m.draft_dir.clear();
      m.draft_artifact.clear();
      m.mtp_artifact.clear();
      m.chat_template.clear();

      if (const auto v = (*entry)["id"].value<std::string>()) {
        m.id = *v;
      } else if (const auto v2 = (*entry)["model_id"].value<std::string>()) {
        m.id = *v2;
      }
      if (const auto v = (*entry)["model_dir"].value<std::string>()) {
        m.model_dir = *v;
      } else if (const auto v2 = (*entry)["dir"].value<std::string>()) {
        m.model_dir = *v2;
      }
      apply_sidecar_model_override(m, config, /*respect_cli_overrides=*/false);

      if (const auto v = (*entry)["provider"].value<std::string>()) {
        m.provider = *v;
      }
      if (const auto v = (*entry)["artifact"].value<std::string>()) {
        m.artifact = *v;
      } else if (const auto v2 =
                     (*entry)["artifact_path"].value<std::string>()) {
        m.artifact = *v2;
      }
      if (const auto v = (*entry)["draft_dir"].value<std::string>()) {
        m.draft_dir = *v;
      }
      if (const auto v = (*entry)["draft_artifact"].value<std::string>()) {
        m.draft_artifact = *v;
      } else if (const auto v2 =
                     (*entry)["draft_artifact_path"].value<std::string>()) {
        m.draft_artifact = *v2;
      }
      if (const auto v = (*entry)["mtp_artifact"].value<std::string>()) {
        m.mtp_artifact = *v;
      } else if (const auto v2 =
                     (*entry)["mtp_artifact_path"].value<std::string>()) {
        m.mtp_artifact = *v2;
      }
      if (const auto v = (*entry)["metallib_dir"].value<std::string>()) {
        m.kernel_dir = *v;
      } else if (const auto v2 = (*entry)["kernel_dir"].value<std::string>()) {
        m.kernel_dir = *v2;
      }
      if (const auto v = (*entry)["chat_template"].value<std::string>()) {
        m.chat_template = *v;
      }
      if (const auto v = read_toml_optional_string(*entry, "lora_adapter")) {
        m.lora_adapter = *v;
      } else if (const auto v2 = read_toml_optional_string(*entry, "lora")) {
        m.lora_adapter = *v2;
      }
      if (const auto v = read_toml_optional_string(*entry, "lora_mode")) {
        m.lora_mode = parse_lora_mode(*v, "TOML [[models]].lora_mode");
      }
      if (const auto v = (*entry)["max_concurrency"].value<std::int64_t>()) {
        m.max_concurrency = static_cast<std::int32_t>(*v);
      }
      if (const auto v = (*entry)["max_context"].value<std::int64_t>()) {
        m.max_context = static_cast<std::int32_t>(*v);
      }
      if (const auto v = (*entry)["kv_dtype"].value<std::string>()) {
        m.kv_dtype = *v;
      }
      if (const auto v = (*entry)["prefill_engine"].value<std::string>()) {
        m.prefill_engine = *v;
      }
      if (const auto v = (*entry)["spec_search"].value<std::string>()) {
        m.spec_search = *v;
      }
      if (const auto v = (*entry)["sampling_contract"].value<std::string>()) {
        m.sampling_contract = *v;
      }
      if (const auto v = (*entry)["spec_backend"].value<std::string>()) {
        parse_spec_backend_setting(*v, m.spec_backend, m.spec_backend_auto);
      } else if (const auto v2 =
                     (*entry)["speculative_backend"].value<std::string>()) {
        parse_spec_backend_setting(*v2, m.spec_backend, m.spec_backend_auto);
      }
      if (const auto v = (*entry)["draft_tokens"].value<std::int64_t>()) {
        if (*v < 0) {
          throw std::invalid_argument(
              "TOML [[models]].draft_tokens must be non-negative");
        }
        m.draft_tokens = static_cast<std::uint32_t>(*v);
      }
      read_toml_optional_bool_or_on_off(*entry, "rolling_prefix_cache",
                                        m.rolling_prefix_cache);
      read_toml_optional_double(*entry, "prefix_cache_gb", m.prefix_cache_gb);
      if (const auto v = (*entry)["batch_mode"].value<std::string>()) {
        m.batch_mode = *v;
      }
      if (entry->get("memory_budget_gb") != nullptr) {
        read_toml_optional_double(*entry, "memory_budget_gb",
                                  m.memory_budget_gb);
        m.memory_budget_bytes = 0;
      }
      if (const auto v =
              (*entry)["memory_budget_bytes"].value<std::int64_t>()) {
        if (*v < 0) {
          throw std::invalid_argument(
              "TOML [[models]].memory_budget_bytes must be non-negative");
        }
        m.memory_budget_bytes = static_cast<std::uint64_t>(*v);
      }
      if (m.model_dir.empty()) {
        throw std::invalid_argument(
            "TOML [[models]] entry is missing required 'model_dir'");
      }
      config.models.push_back(std::move(m));
    }
  }

  expand_serve_config_paths(config);
}

ServeConfig parse_serve_config(int argc, char **argv) {
  ServeConfig config;

  // Pass 1: discover --config / --no-default-config / --version / --help first
  // so CLI flags always override TOML regardless of argument ordering.
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--version") {
      config.show_version = true;
      return config;
    }
    if (arg == "--help" || arg == "-h") {
      config.show_help = true;
      return config;
    }
    if (arg == "--no-default-config") {
      config.no_default_config = true;
    } else if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::invalid_argument("--config requires a path");
      }
      config.config_path = argv[++i];
    } else if (arg.starts_with("--config=")) {
      config.config_path = std::string(arg.substr(9));
    }
  }

  if (!config.config_path.empty()) {
    config.config_path = expand_user_path(config.config_path);
    load_serve_toml_file(config.config_path, config);
  } else if (!config.no_default_config) {
    if (const auto auto_cfg = discover_default_serve_config_path();
        auto_cfg.has_value()) {
      config.config_path = *auto_cfg;
      config.config_auto_discovered = true;
      load_serve_toml_file(config.config_path, config);
    }
  }

  std::vector<std::string> cli_model_dirs;
  std::vector<std::string> cli_providers_dirs;
  struct CliModelOverrides {
    bool provider = false;
    bool artifact = false;
    bool draft_dir = false;
    bool draft_artifact = false;
    bool mtp_artifact = false;
    bool kernel_dir = false;
    bool chat_template = false;
    bool model_id = false;
    bool lora_adapter = false;
    bool lora_mode = false;
    bool max_concurrency = false;
    bool max_context = false;
    bool kv_dtype = false;
    bool spec_backend = false;
    bool draft_tokens = false;
    bool rolling_prefix_cache = false;
    bool prefix_cache_gb = false;
    bool sampling_contract = false;
  } cli_overrides;

  // Pass 2: apply all CLI flags on top of TOML values.
  for (int i = 1; i < argc; ++i) {
    std::string_view raw(argv[i]);
    std::string_view flag = raw;
    std::optional<std::string_view> inline_value;
    if (raw.starts_with("--")) {
      const std::size_t eq = raw.find('=');
      if (eq != std::string_view::npos) {
        flag = raw.substr(0, eq);
        inline_value = raw.substr(eq + 1);
      }
    }

    const auto next_val = [&]() -> std::string_view {
      if (inline_value.has_value()) {
        return *inline_value;
      }
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(flag) + " requires a value");
      }
      return std::string_view(argv[++i]);
    };

    if (flag == "--config") {
      (void)next_val();
    } else if (flag == "--no-default-config") {
      config.no_default_config = true;
    } else if (flag == "--model-dir") {
      cli_model_dirs.emplace_back(next_val());
    } else if (flag == "--providers-dir") {
      cli_providers_dirs.emplace_back(next_val());
    } else if (flag == "--provider") {
      config.provider = std::string(next_val());
      cli_overrides.provider = true;
    } else if (flag == "--artifact") {
      config.artifact = std::string(next_val());
      cli_overrides.artifact = true;
      config.cli_artifact_override = true;
    } else if (flag == "--draft-dir") {
      config.draft_dir = std::string(next_val());
      cli_overrides.draft_dir = true;
      config.cli_draft_dir_override = true;
    } else if (flag == "--draft-artifact") {
      config.draft_artifact = std::string(next_val());
      cli_overrides.draft_artifact = true;
      config.cli_draft_artifact_override = true;
    } else if (flag == "--mtp-artifact") {
      config.mtp_artifact = std::string(next_val());
      cli_overrides.mtp_artifact = true;
      config.cli_mtp_artifact_override = true;
    } else if (flag == "--metallib-dir" || flag == "--kernel-dir") {
      config.kernel_dir = std::string(next_val());
      cli_overrides.kernel_dir = true;
      config.cli_kernel_dir_override = true;
    } else if (flag == "--chat-template") {
      config.chat_template = std::string(next_val());
      cli_overrides.chat_template = true;
    } else if (flag == "--model-id") {
      config.model_id = std::string(next_val());
      cli_overrides.model_id = true;
    } else if (flag == "--lora" || flag == "--lora-adapter") {
      config.lora_adapter = std::string(next_val());
      cli_overrides.lora_adapter = true;
      config.cli_lora_adapter_override = true;
    } else if (flag == "--lora-mode") {
      config.lora_mode = parse_lora_mode(next_val(), flag);
      cli_overrides.lora_mode = true;
      config.cli_lora_mode_override = true;
    } else if (flag == "--host") {
      config.host = std::string(next_val());
    } else if (flag == "--port") {
      config.port = parse_int_arg(next_val(), flag);
    } else if (flag == "--api-key") {
      config.api_key = std::string(next_val());
    } else if (flag == "--memory-budget-gb") {
      config.memory_budget_gb = parse_double_arg(next_val(), flag);
    } else if (flag == "--cors") {
      if (inline_value.has_value()) {
        config.cors = parse_on_off(*inline_value, flag);
      } else if (i + 1 < argc) {
        const std::string_view peek(argv[i + 1]);
        if (peek == "on" || peek == "off" || peek == "true" ||
            peek == "false" || peek == "1" || peek == "0") {
          config.cors = parse_on_off(peek, flag);
          ++i;
        } else {
          config.cors = true;
        }
      } else {
        config.cors = true;
      }
    } else if (flag == "--no-cors") {
      config.cors = false;
    } else if (flag == "--max-concurrency" || flag == "-c" || flag == "-b") {
      config.max_concurrency = parse_int_arg(next_val(), flag);
      cli_overrides.max_concurrency = true;
      config.cli_max_concurrency_override = true;
    } else if (flag == "--max-context") {
      config.max_context = parse_int_arg(next_val(), flag);
      cli_overrides.max_context = true;
      config.cli_max_context_override = true;
    } else if (flag == "--kv-dtype") {
      config.kv_dtype = std::string(next_val());
      cli_overrides.kv_dtype = true;
      config.cli_kv_dtype_override = true;
    } else if (flag == "--spec-backend") {
      parse_spec_backend_setting(next_val(), config.spec_backend,
                                 config.spec_backend_auto);
      cli_overrides.spec_backend = true;
      config.cli_spec_backend_override = true;
    } else if (flag == "--draft-tokens") {
      config.draft_tokens = parse_u32_arg(next_val(), flag);
      cli_overrides.draft_tokens = true;
      config.cli_draft_tokens_override = true;
    } else if (flag == "--rolling-prefix-cache") {
      config.rolling_prefix_cache = parse_on_off(next_val(), flag);
      cli_overrides.rolling_prefix_cache = true;
      config.cli_rolling_prefix_cache_override = true;
    } else if (flag == "--prefix-cache-gb") {
      config.prefix_cache_gb = parse_double_arg(next_val(), flag);
      cli_overrides.prefix_cache_gb = true;
      config.cli_prefix_cache_gb_override = true;
    } else if (flag == "--sampling-contract") {
      config.sampling_contract = std::string(next_val());
      cli_overrides.sampling_contract = true;
      config.cli_sampling_contract_override = true;
    } else if (flag == "--version") {
      config.show_version = true;
    } else if (flag == "--help" || flag == "-h") {
      config.show_help = true;
    } else {
      throw std::invalid_argument("unknown argument: " + std::string(raw));
    }
  }

  // CLI --providers-dir entries take highest precedence (prepend before TOML).
  if (!cli_providers_dirs.empty()) {
    cli_providers_dirs.insert(cli_providers_dirs.end(),
                              config.providers_dirs.begin(),
                              config.providers_dirs.end());
    config.providers_dirs = std::move(cli_providers_dirs);
  }

  // Propagate all explicitly provided CLI overrides to TOML [[models]]
  // entries when no CLI --model-dir replaced them.
  if (cli_model_dirs.empty()) {
    for (auto &m : config.models) {
      if (cli_overrides.provider) {
        m.provider = config.provider;
      }
      if (cli_overrides.artifact) {
        m.artifact = config.artifact;
      }
      if (cli_overrides.draft_dir) {
        m.draft_dir = config.draft_dir;
      }
      if (cli_overrides.draft_artifact) {
        m.draft_artifact = config.draft_artifact;
      }
      if (cli_overrides.mtp_artifact) {
        m.mtp_artifact = config.mtp_artifact;
      }
      if (cli_overrides.kernel_dir) {
        m.kernel_dir = config.kernel_dir;
      }
      if (cli_overrides.chat_template) {
        m.chat_template = config.chat_template;
      }
      if (cli_overrides.model_id && config.models.size() == 1) {
        m.id = config.model_id;
      }
      if (cli_overrides.lora_adapter) {
        m.lora_adapter = config.lora_adapter;
      }
      if (cli_overrides.lora_mode) {
        m.lora_mode = config.lora_mode;
      }
      if (cli_overrides.max_concurrency) {
        m.max_concurrency = config.max_concurrency;
      }
      if (cli_overrides.max_context) {
        m.max_context = config.max_context;
      }
      if (cli_overrides.kv_dtype) {
        m.kv_dtype = config.kv_dtype;
      }
      if (cli_overrides.spec_backend) {
        m.spec_backend = config.spec_backend;
        m.spec_backend_auto = config.spec_backend_auto;
      }
      if (cli_overrides.draft_tokens) {
        m.draft_tokens = config.draft_tokens;
      }
      if (cli_overrides.rolling_prefix_cache) {
        m.rolling_prefix_cache = config.rolling_prefix_cache;
      }
      if (cli_overrides.prefix_cache_gb) {
        m.prefix_cache_gb = config.prefix_cache_gb;
      }
      if (cli_overrides.sampling_contract) {
        m.sampling_contract = config.sampling_contract;
      }
    }
  } else if (cli_model_dirs.size() == 1) {
    config.model_dir = cli_model_dirs.front();
    config.models.clear();
  } else {
    config.model_dir = cli_model_dirs.front();
    config.models.clear();
    for (const auto &dir : cli_model_dirs) {
      ModelServeConfig m = default_model_from_serve_config(
          config, /*apply_sidecar_override=*/false);
      m.model_dir = dir;
      m.id.clear();
      apply_sidecar_model_override(m, config, /*respect_cli_overrides=*/true);
      config.models.push_back(std::move(m));
    }
  }

  expand_serve_config_paths(config);
  validate_serve_config(config);
  return config;
}

std::string resolve_kernel_dir(const std::string &explicit_kernel_dir,
                               const std::string &model_dir) {
  if (!explicit_kernel_dir.empty()) {
    return expand_user_path(explicit_kernel_dir);
  }
  if (const char *env = std::getenv("OMLX_M5_METALLIB_DIR"); env && *env) {
    return expand_user_path(std::string(env));
  }
  if (const auto exe_dir = current_executable_dir(); exe_dir.has_value()) {
    const auto parent_metallib = exe_dir->parent_path() / "metallib";
    if (dir_has_metallib(parent_metallib)) {
      return parent_metallib.string();
    }
    const auto sibling_metallib = *exe_dir / "metallib";
    if (dir_has_metallib(sibling_metallib)) {
      return sibling_metallib.string();
    }
  }
#ifdef OMLX_M5_DEFAULT_METALLIB_DIR
  {
    const std::filesystem::path build_dir(OMLX_M5_DEFAULT_METALLIB_DIR);
    if (dir_has_metallib(build_dir)) {
      return build_dir.string();
    }
  }
#endif
  if (!model_dir.empty()) {
    const std::filesystem::path md(expand_user_path(model_dir));
    if (dir_has_metallib(md)) {
      return md.string();
    }
  }
  return {};
}

std::string resolve_model_id(const ModelServeConfig &model_cfg) {
  if (!model_cfg.id.empty()) {
    return model_cfg.id;
  }
  return derive_model_id_from_dir(model_cfg.model_dir);
}

std::string resolve_model_id(const ServeConfig &config) {
  if (!config.model_id.empty()) {
    return config.model_id;
  }
  if (!config.models.empty()) {
    return resolve_model_id(config.models.front());
  }
  return derive_model_id_from_dir(config.model_dir);
}

std::vector<ModelServeConfig> resolve_model_configs(const ServeConfig &config) {
  std::vector<ModelServeConfig> result;
  if (!config.models.empty()) {
    result = config.models;
  } else if (!config.model_dir.empty()) {
    result.push_back(default_model_from_serve_config(config));
  }

  std::unordered_set<std::string> seen_ids;
  for (auto &m : result) {
    expand_model_paths(m);
    resolve_auto_speculative(m);
    if (m.provider.empty()) {
      m.provider = "auto";
    }
    if (m.memory_budget_bytes == 0 && m.memory_budget_gb.has_value() &&
        *m.memory_budget_gb > 0.0) {
      m.memory_budget_bytes = gb_to_bytes(*m.memory_budget_gb);
    }

    std::string base_id = resolve_model_id(m);
    std::string unique_id = base_id;
    int suffix = 2;
    while (!seen_ids.insert(unique_id).second) {
      unique_id = base_id + "-" + std::to_string(suffix++);
    }
    m.id = std::move(unique_id);
  }
  return result;
}

std::uint64_t resolve_total_memory_budget_bytes(const ServeConfig &config) {
  if (config.memory_budget_gb.has_value() && *config.memory_budget_gb > 0.0) {
    return gb_to_bytes(*config.memory_budget_gb);
  }
  if (config.server_memory_budget_bytes > 0) {
    return config.server_memory_budget_bytes;
  }
  return 0;
}

LoadOptions to_load_options(const ModelServeConfig &model_cfg) {
  ModelServeConfig resolved = model_cfg;
  expand_model_paths(resolved);
  resolve_auto_speculative(resolved);

  LoadOptions opts;
  opts.model_dir = resolved.model_dir;
  opts.draft_dir = resolved.draft_dir;
  opts.max_concurrency = resolved.max_concurrency;
  opts.max_context = resolved.max_context;
  opts.kv_dtype = resolved.kv_dtype;
  opts.prefill_engine = resolved.prefill_engine;
  opts.spec_search = resolved.spec_search;
  opts.sampling_contract = resolved.sampling_contract;
  opts.memory_budget_bytes = resolved.memory_budget_bytes;
  if (opts.memory_budget_bytes == 0 && resolved.memory_budget_gb.has_value() &&
      *resolved.memory_budget_gb > 0.0) {
    opts.memory_budget_bytes = gb_to_bytes(*resolved.memory_budget_gb);
  }
  opts.speculative.backend = resolved.spec_backend;
  opts.speculative.draft_tokens = resolved.draft_tokens;
  opts.lora_adapter = resolved.lora_adapter;
  opts.lora_mode = resolved.lora_mode.empty()
                       ? std::string("merge")
                       : parse_lora_mode(resolved.lora_mode, "--lora-mode");

  const std::string kernel_dir =
      resolve_kernel_dir(resolved.kernel_dir, opts.model_dir);
  if (!kernel_dir.empty()) {
    opts.path_overrides["kernel_directory"] = kernel_dir;
  }
  if (!resolved.artifact.empty()) {
    opts.path_overrides["artifact"] = resolved.artifact;
  }
  if (resolved.draft_artifact.empty() &&
      resolved.spec_backend == SpeculativeBackend::dflash2 &&
      !resolved.draft_dir.empty()) {
    if (const auto art = find_omlxm5_in_dir(resolved.draft_dir);
        art.has_value()) {
      resolved.draft_artifact = art->string();
    }
  }
  if (!resolved.draft_artifact.empty()) {
    opts.path_overrides["draft_artifact"] = resolved.draft_artifact;
  }
  if (!resolved.mtp_artifact.empty()) {
    opts.path_overrides["mtp_artifact"] = resolved.mtp_artifact;
  }
  if (resolved.rolling_prefix_cache.has_value()) {
    opts.path_overrides["rolling_prefix_cache"] =
        *resolved.rolling_prefix_cache ? "on" : "off";
  }
  if (resolved.prefix_cache_gb.has_value()) {
    const auto bytes = static_cast<std::uint64_t>(
        *resolved.prefix_cache_gb * static_cast<double>(1ULL << 30));
    opts.path_overrides["prefix_cache_bytes"] = std::to_string(bytes);
  }
  if (resolved.batch_mode.has_value() && !resolved.batch_mode->empty()) {
    opts.path_overrides["batch_mode"] = *resolved.batch_mode;
  } else if (resolved.max_concurrency >= 2) {
    opts.path_overrides["batch_mode"] = "on";
  }
  return opts;
}

LoadOptions to_load_options(const ServeConfig &config) {
  ModelServeConfig m = default_model_from_serve_config(config);
  if (m.memory_budget_bytes == 0 && !m.memory_budget_gb.has_value()) {
    m.memory_budget_gb = config.memory_budget_gb;
    m.memory_budget_bytes = config.server_memory_budget_bytes;
  }
  return to_load_options(m);
}

std::string format_version_string() {
  std::ostringstream oss;
  oss << "ninfer-metal-serve " << NINFER_METAL_DISPLAY_VERSION
      << " (codename=" << NINFER_METAL_CODENAME
      << " semver=" << NINFER_METAL_SEMVER
      << " core_abi=" << NINFER_METAL_CORE_ABI << ")";
  return oss.str();
}

std::string format_help_string() {
  std::ostringstream oss;
  oss << format_version_string() << "\n"
      << "Standalone native Apple Silicon Metal 4 / NAX multi-microkernel "
         "OpenAI-compatible HTTP server.\n\n"
      << "Usage:\n"
      << "  ninfer-metal-serve --model-dir PATH [options]\n"
      << "  ninfer-metal-serve --model-dir MODEL_A --model-dir MODEL_B "
         "[--providers-dir DIR] [options]\n"
      << "  ninfer-metal-serve --config ninfer-metal.toml [overrides]\n\n"
      << "Configuration:\n"
      << "  --config PATH                    Load TOML config ([server], "
         "[model], [[models]], [engine], [speculative], [sidecar])\n"
      << "                                   Auto-discovers ninfer-metal.toml "
         "when omitted; CLI flags override TOML\n"
      << "  --no-default-config              Disable automatic ninfer-metal.toml "
         "discovery (use compiled defaults)\n\n"
      << "Models, Shaders & Parallel Microkernel Providers:\n"
      << "  --model-dir PATH                 Model directory (repeatable for "
         "multi-model parallel serving)\n"
      << "  --provider MODE                  Backend selection: auto | inproc "
         "| <provider_id> (default: auto)\n"
      << "  --providers-dir PATH             Directory of Path B SPI v2 "
         "microkernel sidecars (repeatable;\n"
      << "                                   scans "
         "<dir>/*/current/provider.json; also checks OMLX_PROVIDERS_DIR\n"
      << "                                   and ~/Library/Application "
         "Support/oMLX/Providers)\n"
      << "  --metallib-dir PATH              Metal 4 / NAX .metallib shader "
         "directory (alias: --kernel-dir;\n"
      << "                                   defaults to OMLX_M5_METALLIB_DIR, "
         "<exe>/../metallib, build fallback)\n"
      << "  --kernel-dir PATH                Alias for --metallib-dir (Metal "
         ".metallib shaders, not SPI providers)\n"
      << "  --artifact PATH                  Explicit model .miv (or "
         "provider) artifact override\n"
      << "  --draft-dir PATH                 Speculative draft model "
         "directory\n"
      << "  --draft-artifact PATH            Explicit draft .miv artifact "
         "override\n"
      << "  --mtp-artifact PATH              External MTP head .miv "
         "artifact override\n"
      << "  --chat-template PATH             Explicit Jinja chat template file "
         "override\n"
      << "  --model-id NAME                  Exposed OpenAI model identifier "
         "(default: model-dir basename)\n"
      << "  --lora, --lora-adapter PATH      LoRA adapter directory (HF PEFT "
         "or mlx-lm safetensors)\n"
      << "  --lora-mode merge|bypass         LoRA execution mode: merge "
         "(load-time fusion, default) | bypass\n\n"
      << "HTTP Server & UMA Memory Coordination:\n"
      << "  --host ADDR                      Bind address (default: "
         "127.0.0.1)\n"
      << "  --port PORT                      Listen port (default: 8080)\n"
      << "  --api-key KEY                    Optional Bearer API key for "
         "/v1/* endpoints\n"
      << "  --memory-budget-gb FLOAT         Total UMA memory budget cap in "
         "GiB (default: 81% physical RAM)\n"
      << "  --cors [on|off]                  Enable/disable CORS headers "
         "(default: on)\n"
      << "  --no-cors                        Disable CORS headers\n\n"
      << "Engine & Speculative Decoding:\n"
      << "  --max-concurrency, -c, -b N      Max concurrent slots 1..8 "
         "(default: 1)\n"
      << "  --max-context N                  Max sequence context length "
         "(default: 32768)\n"
      << "  --kv-dtype bf16|affine4_g128     KV cache storage format (default: "
         "bf16)\n"
      << "  --spec-backend auto|none|mtp|dflash2\n"
      << "                                   Speculative decoding backend "
         "(auto detects dflash2/ -> mtp -> none)\n"
      << "  --draft-tokens N                 Speculative draft token depth "
         "(default: 0 = auto)\n"
      << "  --rolling-prefix-cache on|off    Enable/disable rolling UMA prefix "
         "cache\n"
      << "  --prefix-cache-gb FLOAT          Prefix cache memory budget in GiB "
         "(0..128)\n"
      << "  --sampling-contract NAME         Sampling contract "
         "(legacy_rejection_v1|position_coupled_v1)\n\n"
      << "Info:\n"
      << "  --version                        Print version identity and exit\n"
      << "  --help, -h                       Show this help message and exit\n";
  return oss.str();
}

} // namespace omlx::m5::serve
