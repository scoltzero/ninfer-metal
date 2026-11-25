#pragma once

#include "serve/config.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace omlx::m5::serve {

struct ProviderManifest {
  std::string provider_id;
  std::vector<std::string> legacy_provider_ids;
  std::string engine_id;
  std::string display_name;
  std::string display_version;
  std::uint32_t protocol_version = 0;
  std::uint32_t core_abi = 1;
  std::filesystem::path provider_root;
  std::filesystem::path binary_path;
  std::vector<std::string> supported_topologies;
  std::vector<std::string> supported_model_types;
  std::vector<std::string> artifact_extensions;
  std::vector<std::string> kv_formats;
  std::vector<std::string> capabilities;
};

struct ProviderResolution {
  bool use_sidecar = false;
  std::string provider_id = "io.ninfer.metal";
  std::string backend_mode = "inproc";
  std::optional<ProviderManifest> sidecar_provider;
  std::string resolved_artifact_path;
  std::string reason;
};

class ProviderRegistry final {
public:
  // Scans explicit_dirs, OMLX_PROVIDERS_DIR (colon-separated), and
  // ~/Library/Application Support/oMLX/Providers (when include_env_and_default
  // is true) for `<dir>/*/current/provider.json` (fallback `<dir>/*/provider.json`).
  static ProviderRegistry
  discover(const std::vector<std::string> &explicit_dirs = {},
           bool include_env_and_default = true);

  [[nodiscard]] const std::vector<ProviderManifest> &providers() const noexcept {
    return providers_;
  }

  // Looks up a discovered external Path B provider by provider_id, engine_id,
  // or legacy_provider_ids (case-insensitive).
  [[nodiscard]] const ProviderManifest *
  find(std::string_view id_or_engine) const noexcept;

  // Manually register a provider manifest (useful for tests or overrides).
  void register_provider(ProviderManifest manifest);

private:
  std::vector<ProviderManifest> providers_;
};

// O(1) metadata-only provider resolver:
// - Explicit `model_entry.provider`:
//     "inproc" | "miv" | "io.ninfer.metal" | "m5rt" -> in-process miv
//     explicit provider_id / engine_id              -> matching SPI provider
// - "auto": reads `model_pack.json` / `omlxm5-manifest.json` (`provider_id`,
//   `engine_id`, `topology_id`, `artifact_path`), `config.json` (`model_type`,
//   `text_config.model_type`), and artifact extension:
//     explicit provider_id match > Path B topology+extension/model_type match >
//     in-process miv (`io.ninfer.metal`).
ProviderResolution resolve_provider(const ModelServeConfig &model_entry,
                                    const ProviderRegistry &registry);

} // namespace omlx::m5::serve
