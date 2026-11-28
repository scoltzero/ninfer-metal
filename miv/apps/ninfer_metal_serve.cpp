#include "serve/config.h"
#include "serve/http_server.h"
#include "serve/multi_model_host.h"
#include "serve/provider_registry.h"

#include <atomic>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

namespace {

std::atomic<int> g_signal_pipe_write_fd{-1};

extern "C" void handle_shutdown_signal(int /*signum*/) {
  const int fd = g_signal_pipe_write_fd.load(std::memory_order_relaxed);
  if (fd >= 0) {
    const char byte = 1;
    const auto written = ::write(fd, &byte, 1);
    (void)written;
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto config = omlx::m5::serve::parse_serve_config(argc, argv);
    if (config.show_version) {
      std::cout << omlx::m5::serve::format_version_string() << '\n';
      return 0;
    }
    if (config.show_help) {
      std::cout << omlx::m5::serve::format_help_string();
      return 0;
    }

    const auto model_configs = omlx::m5::serve::resolve_model_configs(config);
    if (config.describe_models) {
      nlohmann::json list = nlohmann::json::array();
      for (const auto &m : model_configs) {
        nlohmann::json engine = {
            {"max_context", m.max_context}, {"max_concurrency", m.max_concurrency},
            {"kv_dtype", m.kv_dtype}, {"draft_tokens", m.draft_tokens},
            {"spec_backend", omlx::m5::serve::spec_backend_name(m.spec_backend, m.spec_backend_auto)},
            {"artifact", m.artifact}, {"draft_artifact", m.draft_artifact}, {"mtp_artifact", m.mtp_artifact},
            {"draft_dir", m.draft_dir}, {"metallib_dir", m.kernel_dir}, {"chat_template", m.chat_template},
            {"lora_adapter", m.lora_adapter}, {"lora_mode", m.lora_mode}, {"sampling_contract", m.sampling_contract},
            {"prefill_linear_mode", omlx::m5::prefill_linear_mode_to_string(m.prefill_linear_mode)},
            {"prefill_attention_mode", omlx::m5::prefill_attention_mode_to_string(m.prefill_attention_mode)},
            {"prefill_sparse_minference", m.prefill_sparse_minference}, {"prefill_cache_blend", m.prefill_cache_blend},
            {"prefill_ane_offload", m.prefill_ane_offload},
        };
        if (m.rolling_prefix_cache) engine["rolling_prefix_cache"] = *m.rolling_prefix_cache;
        if (m.prefix_cache_gb) engine["prefix_cache_gb"] = *m.prefix_cache_gb;
        if (m.memory_budget_gb) engine["memory_budget_gb"] = *m.memory_budget_gb;
        if (m.batch_mode) engine["batch_mode"] = *m.batch_mode;
        for (auto &[key, value] : engine.items()) {
          if (!value.is_string()) value = value.dump();
        }
        list.push_back({{"engine", engine}, {"id", m.id}, {"directory", m.model_dir},
                        {"max_context", m.max_context}, {"max_concurrency", m.max_concurrency},
                        {"kv_dtype", m.kv_dtype}, {"spec_backend",
                            omlx::m5::serve::spec_backend_name(m.spec_backend, m.spec_backend_auto)}});
      }
      std::cout << list.dump() << '\n';
      return 0;
    }
    if (model_configs.empty()) {
      std::cerr << "ninfer-metal-serve: --model-dir is required (or set "
                   "[model].model_dir / [[models]] in --config)\n"
                << "Run 'ninfer-metal-serve --help' for usage.\n";
      return 2;
    }

    std::cerr << "[ninfer-metal-serve] "
              << omlx::m5::serve::format_version_string() << '\n';
    if (!config.config_path.empty()) {
      std::cerr << "[ninfer-metal-serve] Config: " << config.config_path
                << (config.config_auto_discovered ? " (auto-discovered)" : "")
                << '\n';
    }

    const auto registry =
        omlx::m5::serve::ProviderRegistry::discover(config.providers_dirs, !config.explicit_providers_only);
    if (!registry.providers().empty()) {
      std::cerr << "[ninfer-metal-serve] Discovered "
                << registry.providers().size()
                << " Path B SPI v2 provider(s):\n";
      for (const auto &p : registry.providers()) {
        std::cerr << "  - " << p.provider_id << " (engine=" << p.engine_id
                  << ", proto=v" << p.protocol_version << ", bin="
                  << p.binary_path.string() << ")\n";
      }
    }

    const std::uint64_t total_budget =
        omlx::m5::serve::resolve_total_memory_budget_bytes(config);
    auto host = std::make_shared<omlx::m5::serve::MultiModelHost>(total_budget);
    const double budget_gib =
        static_cast<double>(host->total_memory_budget_bytes()) /
        static_cast<double>(1ULL << 30);
    std::cerr << "[ninfer-metal-serve] UMA memory budget cap: " << std::fixed
              << std::setprecision(2) << budget_gib << " GiB\n";

    for (const auto &mcfg : model_configs) {
      const auto resolved =
          omlx::m5::serve::resolve_provider(mcfg, registry);
      std::cerr << "[ninfer-metal-serve] Loading model '" << mcfg.id
                << "' from " << mcfg.model_dir << " (provider="
                << resolved.provider_id << ", mode=" << resolved.backend_mode
                << ", reason=" << resolved.reason
                << ", max_concurrency=" << mcfg.max_concurrency
                << ", max_context=" << mcfg.max_context
                << ", kv_dtype=" << mcfg.kv_dtype << ", spec="
                << omlx::m5::serve::spec_backend_name(mcfg.spec_backend,
                                                      mcfg.spec_backend_auto);
      if (!mcfg.lora_adapter.empty()) {
        std::cerr << ", lora=" << mcfg.lora_adapter << " (" << mcfg.lora_mode
                  << ")";
      }
      std::cerr << ")...\n";
      auto inst = host->load_model(mcfg, registry);
      std::cerr << "[ninfer-metal-serve] Ready: '" << inst->id
                << "' (provider=" << inst->provider_id
                << ", mode=" << inst->backend_mode
                << ", template=" << inst->chat_template->metadata().source_name
                << ", eos_ids="
                << inst->chat_template->metadata().eos_token_ids.size()
                << ")\n";
    }

    omlx::m5::serve::ServeHttpServer http_server(config, host);
    const int bound_port = http_server.bind();

    // Signal self-pipe so SIGINT / SIGTERM cleanly stops HTTP + all backends.
    int sig_pipe[2] = {-1, -1};
    if (::pipe(sig_pipe) != 0) {
      throw std::runtime_error("failed to create signal pipe");
    }
    g_signal_pipe_write_fd.store(sig_pipe[1], std::memory_order_release);
    std::signal(SIGPIPE, SIG_IGN);

    struct sigaction action {};
    action.sa_handler = handle_shutdown_signal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    (void)::sigaction(SIGINT, &action, nullptr);
    (void)::sigaction(SIGTERM, &action, nullptr);

    std::atomic<bool> exiting{false};
    std::thread signal_watcher([&] {
      char byte = 0;
      const auto n = ::read(sig_pipe[0], &byte, 1);
      (void)n;
      if (!exiting.load(std::memory_order_acquire)) {
        std::cerr
            << "\n[ninfer-metal-serve] Shutdown signal received, draining...\n";
        http_server.stop();
      }
    });

    std::cerr << "[ninfer-metal-serve] Listening on http://" << config.host
              << ':' << bound_port << " (" << host->model_count()
              << " model(s), auth="
              << (std::getenv("MIV_ACCESS_FILE") ? "app-policy" : (config.api_key.empty() ? "none" : "bearer"))
              << ", cors=" << (config.cors ? "on" : "off") << ")\n";

    http_server.listen_after_bind();

    exiting.store(true, std::memory_order_release);
    {
      const char wake = 1;
      const auto written = ::write(sig_pipe[1], &wake, 1);
      (void)written;
    }
    if (signal_watcher.joinable()) {
      signal_watcher.join();
    }
    g_signal_pipe_write_fd.store(-1, std::memory_order_release);
    ::close(sig_pipe[0]);
    ::close(sig_pipe[1]);

    host->stop_all();
    std::cerr << "[ninfer-metal-serve] Clean shutdown complete.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ninfer-metal-serve: " << error.what() << '\n';
    return 1;
  }
}
