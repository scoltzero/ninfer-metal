#include "serve/provider_registry.h"

#include <nlohmann/json.hpp>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace omlx::m5::serve {
namespace {

using Json = nlohmann::json;

std::string to_lower(std::string_view input) {
  std::string out(input);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

std::string trim(std::string_view input) {
  std::size_t start = 0;
  while (start < input.size() &&
         std::isspace(static_cast<unsigned char>(input[start]))) {
    ++start;
  }
  std::size_t end = input.size();
  while (end > start &&
         std::isspace(static_cast<unsigned char>(input[end - 1]))) {
    --end;
  }
  return std::string(input.substr(start, end - start));
}

bool is_builtin_provider_token(std::string_view token) {
  const std::string low = to_lower(trim(token));
  return low == "inproc" || low == "miv" || low == "m5rt" ||
         low == "io.ninfer.metal" || low == "com.omlx.mseries" ||
         low == "ninfer-metal" || low == "ninfer_metal";
}

std::string normalize_extension(std::string_view raw) {
  std::string ext = to_lower(trim(raw));
  if (!ext.empty() && ext.front() != '.') {
    ext.insert(ext.begin(), '.');
  }
  return ext;
}

std::vector<std::string> parse_string_list(const Json &obj, const char *key,
                                           bool lowercase = false,
                                           bool as_extension = false) {
  std::vector<std::string> out;
  if (!obj.contains(key) || !obj.at(key).is_array()) {
    return out;
  }
  for (const auto &item : obj.at(key)) {
    if (!item.is_string()) {
      continue;
    }
    std::string val = trim(item.get<std::string>());
    if (val.empty()) {
      continue;
    }
    if (as_extension) {
      val = normalize_extension(val);
    } else if (lowercase) {
      val = to_lower(val);
    }
    if (std::find(out.begin(), out.end(), val) == out.end()) {
      out.push_back(std::move(val));
    }
  }
  return out;
}

std::optional<Json> read_json_file(const std::filesystem::path &path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) {
    return std::nullopt;
  }
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in.is_open()) {
    return std::nullopt;
  }
  try {
    Json parsed = Json::parse(in);
    if (!parsed.is_object()) {
      return std::nullopt;
    }
    return parsed;
  } catch (...) {
    return std::nullopt;
  }
}

bool is_executable_file(const std::filesystem::path &path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) {
    return false;
  }
  return ::access(path.c_str(), X_OK) == 0;
}

std::optional<ProviderManifest>
parse_provider_dir(const std::filesystem::path &entry_dir) {
  std::filesystem::path provider_root;
  std::optional<Json> manifest_json;

  const std::array<std::filesystem::path, 2> candidates = {
      entry_dir / "current",
      entry_dir,
  };
  for (const auto &root : candidates) {
    std::error_code ec;
    if (!std::filesystem::is_directory(root, ec)) {
      continue;
    }
    auto parsed = read_json_file(root / "provider.json");
    if (parsed.has_value()) {
      provider_root = root;
      manifest_json = std::move(parsed);
      break;
    }
  }
  if (!manifest_json.has_value()) {
    return std::nullopt;
  }

  const Json &doc = *manifest_json;
  std::string provider_id =
      trim(doc.value("provider_id", entry_dir.filename().string()));
  if (provider_id.empty() || is_builtin_provider_token(provider_id)) {
    return std::nullopt;
  }

  if (!doc.contains("protocol_version") ||
      !doc.at("protocol_version").is_number_integer()) {
    return std::nullopt;
  }
  const auto proto = doc.at("protocol_version").get<std::int64_t>();
  if (proto != 1 && proto != 2) {
    return std::nullopt;
  }

  std::filesystem::path binary_path;
  for (const char *key : {"entrypoint", "binary", "daemon_path"}) {
    if (doc.contains(key) && doc.at(key).is_string()) {
      const std::string raw = trim(doc.at(key).get<std::string>());
      if (raw.empty()) {
        continue;
      }
      std::filesystem::path cand(expand_user_path(raw));
      if (cand.is_relative()) {
        cand = provider_root / cand;
      }
      if (is_executable_file(cand)) {
        std::error_code ec;
        const auto canon = std::filesystem::weakly_canonical(cand, ec);
        binary_path = ec ? cand : canon;
        break;
      }
    }
  }
  if (binary_path.empty()) {
    return std::nullopt;
  }

  ProviderManifest m;
  m.provider_id = std::move(provider_id);
  m.legacy_provider_ids = parse_string_list(doc, "legacy_provider_ids");
  m.engine_id = trim(doc.value("engine_id", m.provider_id));
  if (m.engine_id.empty()) {
    m.engine_id = m.provider_id;
  }
  m.display_name = trim(doc.value("display_name", m.engine_id));
  m.display_version = trim(doc.value("display_version", std::string{}));
  m.protocol_version = static_cast<std::uint32_t>(proto);
  if (doc.contains("core_abi") && doc.at("core_abi").is_number_integer() &&
      doc.at("core_abi").get<std::int64_t>() >= 0) {
    m.core_abi = static_cast<std::uint32_t>(doc.at("core_abi").get<std::int64_t>());
  }
  m.provider_root = provider_root;
  m.binary_path = std::move(binary_path);
  m.supported_topologies =
      parse_string_list(doc, "supported_topologies", /*lowercase=*/true);
  m.supported_model_types =
      parse_string_list(doc, "supported_model_types", /*lowercase=*/true);
  m.artifact_extensions = parse_string_list(
      doc, "artifact_extensions", /*lowercase=*/true, /*as_extension=*/true);
  m.kv_formats = parse_string_list(doc, "kv_formats");
  m.capabilities = parse_string_list(doc, "capabilities");
  return m;
}

bool manifest_matches_token(const ProviderManifest &m,
                            std::string_view token) {
  const std::string low = to_lower(trim(token));
  if (low.empty()) {
    return false;
  }
  if (to_lower(m.provider_id) == low || to_lower(m.engine_id) == low) {
    return true;
  }
  for (const auto &legacy : m.legacy_provider_ids) {
    if (to_lower(legacy) == low) {
      return true;
    }
  }
  return false;
}

bool contains_ci(const std::vector<std::string> &haystack,
                 std::string_view needle) {
  const std::string low = to_lower(needle);
  return std::any_of(haystack.begin(), haystack.end(),
                     [&](const std::string &item) {
                       return to_lower(item) == low;
                     });
}

struct ModelDirMetadata {
  std::string pack_provider;
  std::string pack_engine;
  std::string pack_topology;
  std::string pack_artifact;
  std::vector<std::string> model_types;
  std::vector<std::filesystem::path> top_level_files;
};

ModelDirMetadata inspect_model_dir(const std::filesystem::path &model_dir) {
  ModelDirMetadata meta;
  std::error_code ec;
  if (model_dir.empty() || !std::filesystem::is_directory(model_dir, ec)) {
    return meta;
  }

  // 1) Read model_pack.json (fallback omlxm5-manifest.json)
  std::optional<Json> pack_meta;
  for (const char *name : {"model_pack.json", "omlxm5-manifest.json"}) {
    pack_meta = read_json_file(model_dir / name);
    if (pack_meta.has_value()) {
      break;
    }
  }
  if (pack_meta.has_value()) {
    const Json &pm = *pack_meta;
    for (const char *k : {"provider_id", "provider"}) {
      if (pm.contains(k) && pm.at(k).is_string()) {
        meta.pack_provider = trim(pm.at(k).get<std::string>());
        if (!meta.pack_provider.empty()) {
          break;
        }
      }
    }
    for (const char *k : {"engine_id", "engine"}) {
      if (pm.contains(k) && pm.at(k).is_string()) {
        meta.pack_engine = trim(pm.at(k).get<std::string>());
        if (!meta.pack_engine.empty()) {
          break;
        }
      }
    }
    for (const char *k : {"topology_id", "topology"}) {
      if (pm.contains(k) && pm.at(k).is_string()) {
        meta.pack_topology = to_lower(trim(pm.at(k).get<std::string>()));
        if (!meta.pack_topology.empty()) {
          break;
        }
      }
    }
    for (const char *k : {"artifact_path", "artifact"}) {
      if (pm.contains(k) && pm.at(k).is_string()) {
        const std::string raw = trim(pm.at(k).get<std::string>());
        if (!raw.empty()) {
          std::filesystem::path p(expand_user_path(raw));
          if (p.is_relative()) {
            p = model_dir / p;
          }
          meta.pack_artifact = p.string();
          break;
        }
      }
    }
    if (pm.contains("model_type") && pm.at("model_type").is_string()) {
      const std::string mt = to_lower(trim(pm.at("model_type").get<std::string>()));
      if (!mt.empty()) {
        meta.model_types.push_back(mt);
      }
    }
  }

  // 2) Read config.json (model_type, text_config.model_type, topology_id)
  if (const auto cfg = read_json_file(model_dir / "config.json");
      cfg.has_value()) {
    const Json &c = *cfg;
    if (meta.pack_topology.empty() && c.contains("topology_id") &&
        c.at("topology_id").is_string()) {
      meta.pack_topology =
          to_lower(trim(c.at("topology_id").get<std::string>()));
    }
    auto add_mt = [&](const std::string &raw) {
      const std::string mt = to_lower(trim(raw));
      if (!mt.empty() && std::find(meta.model_types.begin(),
                                   meta.model_types.end(),
                                   mt) == meta.model_types.end()) {
        meta.model_types.push_back(mt);
      }
    };
    if (c.contains("model_type") && c.at("model_type").is_string()) {
      add_mt(c.at("model_type").get<std::string>());
    }
    if (c.contains("text_config") && c.at("text_config").is_object()) {
      const auto &tc = c.at("text_config");
      if (tc.contains("model_type") && tc.at("model_type").is_string()) {
        add_mt(tc.at("model_type").get<std::string>());
      }
    }
  }

  // 3) Top-level non-recursive file listing for O(1) artifact extension probes
  for (const auto &entry : std::filesystem::directory_iterator(model_dir, ec)) {
    if (entry.is_regular_file(ec)) {
      meta.top_level_files.push_back(entry.path());
    }
  }
  std::sort(meta.top_level_files.begin(), meta.top_level_files.end());

  return meta;
}

std::string find_matching_artifact(const ProviderManifest &provider,
                                   const std::string &explicit_artifact,
                                   const ModelDirMetadata &meta) {
  if (!explicit_artifact.empty()) {
    const std::string ext =
        normalize_extension(std::filesystem::path(explicit_artifact).extension().string());
    if (provider.artifact_extensions.empty() ||
        contains_ci(provider.artifact_extensions, ext)) {
      return explicit_artifact;
    }
    return {};
  }
  if (!meta.pack_artifact.empty()) {
    const std::string ext =
        normalize_extension(std::filesystem::path(meta.pack_artifact).extension().string());
    if (provider.artifact_extensions.empty() ||
        contains_ci(provider.artifact_extensions, ext)) {
      return meta.pack_artifact;
    }
    return {};
  }
  for (const auto &ext : provider.artifact_extensions) {
    for (const auto &file : meta.top_level_files) {
      if (normalize_extension(file.extension().string()) == ext) {
        return file.string();
      }
    }
  }
  return {};
}

} // namespace

ProviderRegistry
ProviderRegistry::discover(const std::vector<std::string> &explicit_dirs,
                           bool include_env_and_default) {
  ProviderRegistry registry;
  std::vector<std::filesystem::path> search_dirs;
  std::unordered_set<std::string> seen_dirs;

  auto add_search_dir = [&](const std::string &raw) {
    const std::string expanded = expand_user_path(trim(raw));
    if (expanded.empty()) {
      return;
    }
    std::filesystem::path p(expanded);
    std::error_code ec;
    if (!std::filesystem::is_directory(p, ec)) {
      return;
    }
    const auto canon = std::filesystem::weakly_canonical(p, ec);
    const std::string key = (ec ? p : canon).string();
    if (seen_dirs.insert(key).second) {
      search_dirs.push_back(ec ? p : canon);
    }
  };

  for (const auto &dir : explicit_dirs) {
    add_search_dir(dir);
  }

  if (include_env_and_default) {
    if (const char *env = std::getenv("OMLX_PROVIDERS_DIR"); env && *env) {
      std::stringstream ss(env);
      std::string part;
      while (std::getline(ss, part, ':')) {
        add_search_dir(part);
      }
    }
    add_search_dir("~/Library/Application Support/oMLX/Providers");
  }

  for (const auto &root : search_dirs) {
    std::error_code ec;
    std::vector<std::filesystem::path> subdirs;
    for (const auto &entry : std::filesystem::directory_iterator(root, ec)) {
      if (entry.is_directory(ec)) {
        subdirs.push_back(entry.path());
      }
    }
    std::sort(subdirs.begin(), subdirs.end());

    for (const auto &sub : subdirs) {
      if (is_builtin_provider_token(sub.filename().string())) {
        continue;
      }
      auto parsed = parse_provider_dir(sub);
      if (parsed.has_value()) {
        registry.register_provider(std::move(*parsed));
      }
    }
  }

  return registry;
}

const ProviderManifest *
ProviderRegistry::find(std::string_view id_or_engine) const noexcept {
  for (const auto &p : providers_) {
    if (manifest_matches_token(p, id_or_engine)) {
      return &p;
    }
  }
  return nullptr;
}

void ProviderRegistry::register_provider(ProviderManifest manifest) {
  if (manifest.provider_id.empty() ||
      is_builtin_provider_token(manifest.provider_id)) {
    return;
  }
  if (find(manifest.provider_id) != nullptr ||
      find(manifest.engine_id) != nullptr) {
    return;
  }
  providers_.push_back(std::move(manifest));
}

ProviderResolution resolve_provider(const ModelServeConfig &model_entry,
                                    const ProviderRegistry &registry) {
  const std::string requested = to_lower(trim(model_entry.provider));

  // 1) Explicit in-process selection
  if (is_builtin_provider_token(requested)) {
    return ProviderResolution{
        .use_sidecar = false,
        .provider_id = "io.ninfer.metal",
        .backend_mode = "inproc",
        .resolved_artifact_path = model_entry.artifact,
        .reason = "explicit inproc",
    };
  }

  // 2) Explicit external Path B SPI provider selection
  if (!requested.empty() && requested != "auto") {
    const ProviderManifest *found = registry.find(requested);
    if (found == nullptr) {
      throw std::invalid_argument(
          "requested provider '" + model_entry.provider +
          "' was not discovered in any --providers-dir");
    }
    const auto meta = inspect_model_dir(model_entry.model_dir);
    return ProviderResolution{
        .use_sidecar = true,
        .provider_id = found->provider_id,
        .backend_mode = "sidecar",
        .sidecar_provider = *found,
        .resolved_artifact_path =
            find_matching_artifact(*found, model_entry.artifact, meta),
        .reason = "explicit provider",
    };
  }

  // 3) O(1) "auto" routing
  const auto meta = inspect_model_dir(model_entry.model_dir);

  // 3a) Explicit provider_id / engine_id in model_pack.json
  if (!meta.pack_provider.empty() || !meta.pack_engine.empty()) {
    const std::string &token =
        !meta.pack_provider.empty() ? meta.pack_provider : meta.pack_engine;
    if (is_builtin_provider_token(token)) {
      return ProviderResolution{
          .use_sidecar = false,
          .provider_id = "io.ninfer.metal",
          .backend_mode = "inproc",
          .resolved_artifact_path =
              !model_entry.artifact.empty() ? model_entry.artifact
                                            : meta.pack_artifact,
          .reason = "model_pack explicit inproc",
      };
    }
    if (const ProviderManifest *found = registry.find(token);
        found != nullptr) {
      return ProviderResolution{
          .use_sidecar = true,
          .provider_id = found->provider_id,
          .backend_mode = "sidecar",
          .sidecar_provider = *found,
          .resolved_artifact_path =
              find_matching_artifact(*found, model_entry.artifact, meta),
          .reason = "model_pack explicit provider_id",
      };
    }
    if (!meta.pack_engine.empty() && meta.pack_engine != token) {
      if (const ProviderManifest *found = registry.find(meta.pack_engine);
          found != nullptr) {
        return ProviderResolution{
            .use_sidecar = true,
            .provider_id = found->provider_id,
            .backend_mode = "sidecar",
            .sidecar_provider = *found,
            .resolved_artifact_path =
                find_matching_artifact(*found, model_entry.artifact, meta),
            .reason = "model_pack explicit engine_id",
        };
      }
    }
  }

  // 3b) Path B match by topology, model_type, and artifact extension (with
  //     strict no-hijack rules for built-in .miv / .omlxm5 Qwen models).
  const auto is_builtin_miv_ext = [](const std::string &ext) {
    return ext == ".miv" || ext == ".omlxm5";
  };
  const std::string explicit_ext =
      !model_entry.artifact.empty()
          ? normalize_extension(
                std::filesystem::path(model_entry.artifact).extension().string())
          : (!meta.pack_artifact.empty()
                 ? normalize_extension(std::filesystem::path(meta.pack_artifact)
                                           .extension()
                                           .string())
                 : std::string{});

  const bool dir_has_builtin_art = std::any_of(
      meta.top_level_files.begin(), meta.top_level_files.end(),
      [&](const std::filesystem::path &p) {
        return is_builtin_miv_ext(
            normalize_extension(p.extension().string()));
      });
  if (is_builtin_miv_ext(explicit_ext) ||
      (explicit_ext.empty() && dir_has_builtin_art)) {
    return ProviderResolution{
        .use_sidecar = false,
        .provider_id = "io.ninfer.metal",
        .backend_mode = "inproc",
        .resolved_artifact_path =
            !model_entry.artifact.empty() ? model_entry.artifact
                                          : meta.pack_artifact,
        .reason = "built-in .miv/.omlxm5 inproc miv",
    };
  }

  const ProviderManifest *best_provider = nullptr;
  int best_score = 0;
  std::string best_artifact;
  std::string best_reason;

  for (const auto &candidate : registry.providers()) {
    // Reject if explicit artifact extension is incompatible with candidate.
    if (!explicit_ext.empty() && !candidate.artifact_extensions.empty() &&
        !contains_ci(candidate.artifact_extensions, explicit_ext)) {
      continue;
    }

    // Topology check
    bool topology_matched = false;
    if (!meta.pack_topology.empty() && !candidate.supported_topologies.empty()) {
      if (!contains_ci(candidate.supported_topologies, meta.pack_topology)) {
        continue;
      }
      topology_matched = true;
    }

    // model_type check (config.json model_type / text_config.model_type)
    bool model_type_matched = false;
    if (!meta.model_types.empty() && !candidate.supported_model_types.empty()) {
      for (const auto &mt : meta.model_types) {
        if (contains_ci(candidate.supported_model_types, mt)) {
          model_type_matched = true;
          break;
        }
      }
      if (!model_type_matched && !topology_matched) {
        continue;
      }
    }

    // Artifact extension probe: if a candidate declares required artifact
    // extensions, a matching artifact must actually exist.
    const std::string matched_art =
        find_matching_artifact(candidate, model_entry.artifact, meta);
    const bool ext_matched = !matched_art.empty();
    if (!candidate.artifact_extensions.empty() && !ext_matched) {
      continue;
    }
    const bool custom_ext_matched =
        ext_matched &&
        !is_builtin_miv_ext(normalize_extension(
            std::filesystem::path(matched_art).extension().string()));

    int score = 0;
    std::string reason;
    if (topology_matched && ext_matched) {
      score = 4;
      reason = "topology+extension match";
    } else if (topology_matched) {
      score = 3;
      reason = "topology match";
    } else if (model_type_matched && ext_matched) {
      score = 3;
      reason = "model_type+extension match";
    } else if (model_type_matched) {
      score = 2;
      reason = "model_type match";
    } else if (custom_ext_matched) {
      score = 1;
      reason = "custom artifact extension match";
    }

    if (score > best_score) {
      best_score = score;
      best_provider = &candidate;
      best_artifact = matched_art;
      best_reason = std::move(reason);
    }
  }

  if (best_provider != nullptr) {
    return ProviderResolution{
        .use_sidecar = true,
        .provider_id = best_provider->provider_id,
        .backend_mode = "sidecar",
        .sidecar_provider = *best_provider,
        .resolved_artifact_path = std::move(best_artifact),
        .reason = std::move(best_reason),
    };
  }

  // 3c) Default fallback: in-process miv mainline
  return ProviderResolution{
      .use_sidecar = false,
      .provider_id = "io.ninfer.metal",
      .backend_mode = "inproc",
      .resolved_artifact_path =
          !model_entry.artifact.empty() ? model_entry.artifact
                                        : meta.pack_artifact,
      .reason = "default inproc miv",
  };
}

} // namespace omlx::m5::serve
