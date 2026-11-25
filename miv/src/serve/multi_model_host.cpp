#include "serve/multi_model_host.h"

#include "omlx_m5/engine.h"
#include "serve/serve_backend.h"
#include "serve/uds_sidecar_backend.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace omlx::m5::serve {
namespace {

std::string read_file_contents(const std::filesystem::path &path,
                               bool required) {
  std::ifstream input(path, std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    if (required) {
      throw std::runtime_error("failed to open required file: " +
                               path.string());
    }
    return {};
  }
  std::ostringstream out;
  out << input.rdbuf();
  return out.str();
}

} // namespace

std::uint64_t detect_physical_memory_bytes() noexcept {
#ifdef __APPLE__
  std::uint64_t memsize = 0;
  std::size_t len = sizeof(memsize);
  if (::sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) == 0 &&
      memsize > 0) {
    return memsize;
  }
#endif
  return 0;
}

std::uint64_t default_uma_memory_budget_bytes() noexcept {
  const std::uint64_t phys = detect_physical_memory_bytes();
  if (phys == 0) {
    return 0;
  }
  return static_cast<std::uint64_t>(static_cast<double>(phys) * 0.81);
}

MultiModelHost::MultiModelHost(std::uint64_t total_memory_budget_bytes)
    : total_memory_budget_bytes_(total_memory_budget_bytes > 0
                                     ? total_memory_budget_bytes
                                     : default_uma_memory_budget_bytes()) {}

MultiModelHost::~MultiModelHost() { stop_all(); }

std::uint64_t MultiModelHost::total_resident_memory_bytes() const {
  std::vector<std::shared_ptr<ModelInstance>> snapshot = models();
  std::uint64_t sum = 0;
  for (const auto &m : snapshot) {
    if (m != nullptr && m->router != nullptr) {
      sum += m->router->resident_memory_bytes();
    }
  }
  return sum;
}

std::uint64_t MultiModelHost::remaining_memory_budget_bytes() const {
  if (total_memory_budget_bytes_ == 0) {
    return 0;
  }
  const std::uint64_t used = total_resident_memory_bytes();
  return total_memory_budget_bytes_ > used ? (total_memory_budget_bytes_ - used)
                                           : 0ULL;
}

std::uint64_t MultiModelHost::trim_idle_caches(std::uint64_t target_free_bytes) {
  const auto snapshot = models();
  std::uint64_t total_freed = 0;
  for (const auto &m : snapshot) {
    if (m == nullptr || m->router == nullptr || !m->router->is_running()) {
      continue;
    }
    if (m->router->active_requests() > 0) {
      continue;
    }
    const std::uint64_t need =
        (target_free_bytes > total_freed) ? (target_free_bytes - total_freed) : 0;
    try {
      const auto report = m->router->trim_cache(need);
      total_freed += report.freed_bytes;
    } catch (...) {
      // Best-effort trim across heterogeneous providers.
    }
    if (target_free_bytes > 0 && total_freed >= target_free_bytes) {
      break;
    }
  }
  return total_freed;
}

TrimCacheReport
MultiModelHost::trim_model_cache(std::string_view model_id,
                                 std::uint64_t target_free_bytes) {
  auto model = find_model(model_id);
  if (model == nullptr || model->router == nullptr) {
    throw std::invalid_argument("unknown model id for trim_cache: " +
                                std::string(model_id));
  }
  return model->router->trim_cache(target_free_bytes);
}

std::uint64_t
MultiModelHost::compute_model_load_budget(const ModelServeConfig &model_cfg) {
  const std::uint64_t explicit_budget = model_cfg.memory_budget_bytes;
  if (total_memory_budget_bytes_ == 0) {
    return explicit_budget;
  }

  std::uint64_t used = total_resident_memory_bytes();
  std::uint64_t available =
      total_memory_budget_bytes_ > used ? (total_memory_budget_bytes_ - used) : 0;

  // Under memory pressure (available < explicit_budget, or >85% budget already
  // resident), trim idle caches from previously loaded models first.
  const std::uint64_t pressure_threshold =
      total_memory_budget_bytes_ / 100ULL * 15ULL;
  if (explicit_budget > 0 && available < explicit_budget) {
    (void)trim_idle_caches(explicit_budget - available);
    used = total_resident_memory_bytes();
    available = total_memory_budget_bytes_ > used
                    ? (total_memory_budget_bytes_ - used)
                    : 0;
  } else if (explicit_budget == 0 && available < pressure_threshold) {
    (void)trim_idle_caches(pressure_threshold - available);
    used = total_resident_memory_bytes();
    available = total_memory_budget_bytes_ > used
                    ? (total_memory_budget_bytes_ - used)
                    : 0;
  }

  if (explicit_budget > 0) {
    return available > 0 ? std::min(explicit_budget, available)
                         : explicit_budget;
  }
  return available;
}

std::shared_ptr<ModelInstance>
MultiModelHost::load_model(const ModelServeConfig &model_cfg,
                           const ProviderRegistry &registry) {
  const std::filesystem::path model_dir(expand_user_path(model_cfg.model_dir));
  if (model_dir.empty() || !std::filesystem::is_directory(model_dir)) {
    throw std::runtime_error("model directory does not exist: " +
                             model_dir.string());
  }

  const std::string model_id = resolve_model_id(model_cfg);
  if (find_model(model_id) != nullptr) {
    throw std::invalid_argument("duplicate model id: " + model_id);
  }

  const ProviderResolution resolution = resolve_provider(model_cfg, registry);

  const std::string tokenizer_json =
      read_file_contents(model_dir / "tokenizer.json", true);
  const std::string tokenizer_config_json =
      read_file_contents(model_dir / "tokenizer_config.json", false);

  auto tokenizer = std::make_shared<Tokenizer>(TokenizerResources{
      .tokenizer_json = tokenizer_json,
      .tokenizer_config_json = tokenizer_config_json,
  });

  auto chat_template = std::make_shared<ChatTemplate>(ChatTemplate::load(
      model_dir, *tokenizer, expand_user_path(model_cfg.chat_template)));

  LoadOptions load_options = to_load_options(model_cfg);
  if (!resolution.resolved_artifact_path.empty() &&
      !load_options.path_overrides.contains("artifact")) {
    load_options.path_overrides["artifact"] = resolution.resolved_artifact_path;
  }

  const std::uint64_t budget_bytes = compute_model_load_budget(model_cfg);
  load_options.memory_budget_bytes = budget_bytes;

  std::shared_ptr<ServeBackend> backend;
  if (resolution.use_sidecar) {
    if (!resolution.sidecar_provider.has_value()) {
      throw std::runtime_error(
          "internal error: sidecar resolution missing manifest");
    }
    UdsSidecarOptions sidecar_opts;
    sidecar_opts.provider_id = resolution.provider_id;
    sidecar_opts.binary_path = resolution.sidecar_provider->binary_path;
    sidecar_opts.load_options = load_options;
    sidecar_opts.rolling_prefix_cache = model_cfg.rolling_prefix_cache;
    sidecar_opts.prefix_cache_gb = model_cfg.prefix_cache_gb;
    backend = std::make_shared<UdsSidecarBackend>(std::move(sidecar_opts));
  } else {
    const std::uint64_t before_fp = current_process_phys_footprint_bytes();
    std::shared_ptr<Engine> engine(Engine::create(load_options));
    const std::uint64_t after_fp = current_process_phys_footprint_bytes();
    const std::uint64_t delta_fp =
        after_fp > before_fp ? (after_fp - before_fp) : 0ULL;
    backend = std::make_shared<LocalEngineBackend>(
        std::move(engine), delta_fp, resolution.provider_id);
  }

  auto router = std::make_shared<EngineRouter>(
      backend, chat_template->metadata().eos_token_ids);

  ModelInstance inst;
  inst.id = model_id;
  inst.provider_id = backend->provider_id();
  inst.backend_mode = backend->backend_mode();
  inst.router = std::move(router);
  inst.tokenizer = std::move(tokenizer);
  inst.chat_template = std::move(chat_template);
  inst.max_context = load_options.max_context;
  inst.max_concurrency = load_options.max_concurrency;
  inst.allocated_budget_bytes = budget_bytes;
  inst.lora_adapter = load_options.lora_adapter;
  inst.lora_mode = load_options.lora_mode;
  return add_model(std::move(inst));
}

std::shared_ptr<ModelInstance>
MultiModelHost::add_model(ModelInstance instance) {
  if (instance.id.empty()) {
    throw std::invalid_argument("ModelInstance requires a non-empty id");
  }
  if (instance.router == nullptr || instance.tokenizer == nullptr ||
      instance.chat_template == nullptr) {
    throw std::invalid_argument(
        "ModelInstance requires non-null router, tokenizer, and chat_template");
  }
  if (instance.provider_id.empty()) {
    instance.provider_id = instance.router->provider_id();
  }
  if (instance.backend_mode.empty()) {
    instance.backend_mode = instance.router->backend_mode();
  }

  auto shared = std::make_shared<ModelInstance>(std::move(instance));
  {
    std::lock_guard lock(models_mutex_);
    if (models_by_id_.contains(shared->id)) {
      throw std::invalid_argument("duplicate model id: " + shared->id);
    }
    models_by_id_.emplace(shared->id, shared);
    model_order_.push_back(shared);
  }

  shared->router->set_activity_callback([this] { update_contention_hints(); });
  return shared;
}

std::shared_ptr<ModelInstance>
MultiModelHost::find_model(std::string_view model_id) const {
  std::lock_guard lock(models_mutex_);
  const auto it = models_by_id_.find(std::string(model_id));
  if (it == models_by_id_.end()) {
    return nullptr;
  }
  return it->second;
}

std::vector<std::shared_ptr<ModelInstance>> MultiModelHost::models() const {
  std::lock_guard lock(models_mutex_);
  return model_order_;
}

std::size_t MultiModelHost::model_count() const {
  std::lock_guard lock(models_mutex_);
  return model_order_.size();
}

std::string MultiModelHost::primary_model_id() const {
  std::lock_guard lock(models_mutex_);
  if (model_order_.empty()) {
    return {};
  }
  return model_order_.front()->id;
}

bool MultiModelHost::all_running() const {
  const auto snapshot = models();
  if (snapshot.empty()) {
    return false;
  }
  for (const auto &m : snapshot) {
    if (m == nullptr || m->router == nullptr || !m->router->is_running()) {
      return false;
    }
  }
  return true;
}

void MultiModelHost::update_contention_hints() {
  if (stopping_.load(std::memory_order_acquire)) {
    return;
  }

  std::lock_guard coord_lock(coord_mutex_);
  if (stopping_.load(std::memory_order_acquire)) {
    return;
  }

  const auto snapshot = models();
  std::size_t active_models = 0;
  for (const auto &m : snapshot) {
    if (m != nullptr && m->router != nullptr && m->router->is_running() &&
        m->router->active_requests() > 0) {
      ++active_models;
    }
  }

  const bool contended = (active_models >= 2);
  const bool prev_contended =
      gpu_contended_.exchange(contended, std::memory_order_acq_rel);
  if (prev_contended != contended) {
    contention_transitions_.fetch_add(1, std::memory_order_relaxed);
  }

  for (const auto &m : snapshot) {
    if (m == nullptr || m->router == nullptr || !m->router->is_running()) {
      continue;
    }
    const bool model_active = m->router->active_requests() > 0;
    const bool desired = contended && model_active;
    const bool prev = last_contended_hint_[m->id];
    if (desired != prev) {
      last_contended_hint_[m->id] = desired;
      try {
        m->router->set_scheduler_hint(SchedulerHint{.gpu_contended = desired});
      } catch (...) {
        // Ignore transient hint errors if backend is closing.
      }
    }
  }
}

void MultiModelHost::stop_all() {
  if (stopping_.exchange(true, std::memory_order_acq_rel)) {
    return;
  }

  const auto snapshot = models();
  for (const auto &m : snapshot) {
    if (m != nullptr && m->router != nullptr) {
      m->router->set_activity_callback({});
    }
  }
  for (const auto &m : snapshot) {
    if (m != nullptr && m->router != nullptr) {
      m->router->stop();
    }
  }
}

} // namespace omlx::m5::serve
