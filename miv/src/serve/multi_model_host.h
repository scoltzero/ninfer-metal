#pragma once

#include "serve/chat_template.h"
#include "serve/config.h"
#include "serve/engine_router.h"
#include "serve/provider_registry.h"
#include "serve/tokenizer.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace omlx::m5::serve {

struct ModelInstance {
  std::string id;
  std::string api_alias;
  nlohmann::json generation_defaults = nlohmann::json::object();
  std::string provider_id = "io.ninfer.metal";
  std::string backend_mode = "inproc"; // "inproc" | "sidecar"
  std::shared_ptr<EngineRouter> router;
  std::shared_ptr<Tokenizer> tokenizer;
  std::shared_ptr<ChatTemplate> chat_template;
  std::int32_t max_context = 32768;
  std::int32_t max_concurrency = 1;
  std::uint64_t allocated_budget_bytes = 0;
  std::string lora_adapter;
  std::string lora_mode = "merge";
  std::optional<PrefillAneMode> configured_ane_mode;
};

// Queries total physical RAM via sysctl hw.memsize on Apple Silicon.
std::uint64_t detect_physical_memory_bytes() noexcept;

// Computes default safe UMA memory cap (0.81 * physical RAM).
std::uint64_t default_uma_memory_budget_bytes() noexcept;

class MultiModelHost final {
public:
  // If total_memory_budget_bytes == 0, defaults to 0.81 * physical RAM.
  explicit MultiModelHost(std::uint64_t total_memory_budget_bytes = 0);
  ~MultiModelHost();

  MultiModelHost(const MultiModelHost &) = delete;
  MultiModelHost &operator=(const MultiModelHost &) = delete;

  // Loads a model using O(1) SPI v2 provider resolution (either in-process
  // miv Engine or spawned Path B UdsSidecarBackend) while applying UMA budget
  // coordination and idle cache trimming if needed.
  std::shared_ptr<ModelInstance>
  load_model(const ModelServeConfig &model_cfg,
             const ProviderRegistry &registry);

  // Registers a pre-built ModelInstance (used by single-model compatibility
  // wrappers and tests) and wires contention coordination onto its router.
  std::shared_ptr<ModelInstance> add_model(ModelInstance instance);

  [[nodiscard]] std::shared_ptr<ModelInstance>
  find_model(std::string_view model_id) const;

  [[nodiscard]] std::vector<std::shared_ptr<ModelInstance>> models() const;
  [[nodiscard]] std::size_t model_count() const;
  [[nodiscard]] std::string primary_model_id() const;
  [[nodiscard]] bool all_running() const;

  // Trims caches on currently idle backends until target_free_bytes are
  // reclaimed (or all idle backends have been trimmed). Returns total bytes
  // reported freed.
  std::uint64_t trim_idle_caches(std::uint64_t target_free_bytes);

  // Trims cache on a specific loaded model.
  TrimCacheReport trim_model_cache(std::string_view model_id,
                                   std::uint64_t target_free_bytes = 0);

  [[nodiscard]] std::uint64_t total_memory_budget_bytes() const noexcept {
    return total_memory_budget_bytes_;
  }
  [[nodiscard]] std::uint64_t total_resident_memory_bytes() const;
  [[nodiscard]] std::uint64_t remaining_memory_budget_bytes() const;

  [[nodiscard]] bool is_gpu_contended() const noexcept {
    return gpu_contended_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::uint64_t contention_transitions() const noexcept {
    return contention_transitions_.load(std::memory_order_relaxed);
  }

  // Recomputes active model count and dispatches set_scheduler_hint(true/false)
  // to backends as contention begins/ends.
  void update_contention_hints();

  // Cleanly stops all routers and shuts down all inproc and sidecar backends.
  void stop_all();

private:
  std::uint64_t compute_model_load_budget(const ModelServeConfig &model_cfg);

  std::uint64_t total_memory_budget_bytes_ = 0;

  mutable std::mutex models_mutex_;
  std::unordered_map<std::string, std::shared_ptr<ModelInstance>> models_by_id_;
  std::vector<std::shared_ptr<ModelInstance>> model_order_;

  std::mutex coord_mutex_;
  std::unordered_map<std::string, bool> last_contended_hint_;
  std::atomic<bool> gpu_contended_{false};
  std::atomic<std::uint64_t> contention_transitions_{0};
  std::atomic<bool> stopping_{false};
};

} // namespace omlx::m5::serve
