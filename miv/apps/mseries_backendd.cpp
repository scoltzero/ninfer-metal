#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task_info.h>
#endif

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/engine.h"

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

namespace {

using Json = nlohmann::json;
constexpr std::uint32_t kProtocolVersion = 2;
constexpr std::size_t kMaximumMessageBytes = 16U * 1024U * 1024U;
constexpr int kListenBacklog = 16;

std::atomic<int> g_wakeup_write_fd{-1};

extern "C" void handle_termination_signal(int /*signum*/) {
  const int fd = g_wakeup_write_fd.load(std::memory_order_relaxed);
  if (fd >= 0) {
    const char byte = 1;
    const auto written = ::write(fd, &byte, 1);
    (void)written;
  }
}

class FileDescriptor final {
public:
  explicit FileDescriptor(int value = -1) : value_(value) {}
  ~FileDescriptor() {
    if (value_ >= 0) {
      ::close(value_);
    }
  }
  FileDescriptor(const FileDescriptor &) = delete;
  FileDescriptor &operator=(const FileDescriptor &) = delete;
  FileDescriptor(FileDescriptor &&other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  FileDescriptor &operator=(FileDescriptor &&other) noexcept {
    if (this != &other) {
      if (value_ >= 0) {
        ::close(value_);
      }
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  [[nodiscard]] int get() const { return value_; }

private:
  int value_;
};

class SocketPathGuard final {
public:
  explicit SocketPathGuard(std::filesystem::path path)
      : path_(std::move(path)) {}
  ~SocketPathGuard() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

private:
  std::filesystem::path path_;
};

class SignalGuard final {
public:
  explicit SignalGuard(int wakeup_write_fd) {
    g_wakeup_write_fd.store(wakeup_write_fd, std::memory_order_release);
    std::signal(SIGPIPE, SIG_IGN);

    struct sigaction action {};
    action.sa_handler = handle_termination_signal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    (void)::sigaction(SIGTERM, &action, &old_sigterm_);
    (void)::sigaction(SIGINT, &action, &old_sigint_);
  }

  ~SignalGuard() {
    g_wakeup_write_fd.store(-1, std::memory_order_release);
    (void)::sigaction(SIGTERM, &old_sigterm_, nullptr);
    (void)::sigaction(SIGINT, &old_sigint_, nullptr);
  }

  SignalGuard(const SignalGuard &) = delete;
  SignalGuard &operator=(const SignalGuard &) = delete;

private:
  struct sigaction old_sigterm_ {};
  struct sigaction old_sigint_ {};
};

class MessageTooLargeError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

[[noreturn]] void system_error(std::string_view operation) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::strerror(errno));
}

void set_cloexec(int fd) {
  const int flags = ::fcntl(fd, F_GETFD, 0);
  if (flags >= 0) {
    (void)::fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
  }
}

void set_nonblocking(int fd, bool enabled) {
  const int flags = ::fcntl(fd, F_GETFL, 0);
  if (flags < 0) {
    system_error("fcntl(F_GETFL)");
  }
  const int updated = enabled ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
  if (::fcntl(fd, F_SETFL, updated) != 0) {
    system_error("fcntl(F_SETFL)");
  }
}

void configure_socket_nosigpipe(int fd) {
#ifdef SO_NOSIGPIPE
  const int one = 1;
  (void)::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#else
  (void)fd;
#endif
}

void trigger_wakeup_pipe(int wakeup_write_fd) noexcept {
  if (wakeup_write_fd >= 0) {
    const char byte = 1;
    const auto written = ::write(wakeup_write_fd, &byte, 1);
    (void)written;
  }
}

bool is_stop_signaled(int wakeup_read_fd) noexcept {
  if (wakeup_read_fd < 0) {
    return false;
  }
  pollfd pfd{.fd = wakeup_read_fd, .events = POLLIN, .revents = 0};
  const int ready = ::poll(&pfd, 1, 0);
  return ready > 0 && (pfd.revents & (POLLIN | POLLHUP | POLLERR)) != 0;
}

std::uint64_t current_phys_footprint_bytes() noexcept {
#ifdef __APPLE__
  task_vm_info_data_t vm_info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  const kern_return_t kr =
      task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&vm_info), &count);
  if (kr == KERN_SUCCESS) {
    return static_cast<std::uint64_t>(vm_info.phys_footprint);
  }
#endif
  return 0;
}

class LineReader final {
public:
  explicit LineReader(int socket, int stop_fd)
      : socket_(socket), stop_fd_(stop_fd) {
    buffer_.reserve(4096);
  }

  std::optional<std::string> read_line() {
    for (;;) {
      const auto newline = buffer_.find('\n', scan_offset_);
      if (newline != std::string::npos) {
        if (newline > kMaximumMessageBytes) {
          throw MessageTooLargeError("sidecar message exceeds the size limit");
        }
        std::string line = buffer_.substr(0, newline);
        if (!line.empty() && line.back() == '\r') {
          line.pop_back();
        }
        buffer_.erase(0, newline + 1);
        scan_offset_ = 0;
        return line;
      }
      scan_offset_ = buffer_.size();
      if (buffer_.size() > kMaximumMessageBytes) {
        throw MessageTooLargeError("sidecar message exceeds the size limit");
      }

      std::array<pollfd, 2> pfds{{
          {.fd = socket_, .events = POLLIN, .revents = 0},
          {.fd = stop_fd_, .events = POLLIN, .revents = 0},
      }};
      const pollfd *pfd_ptr = pfds.data();
      const nfds_t nfds = stop_fd_ >= 0 ? 2 : 1;
      const int ready = ::poll(const_cast<pollfd *>(pfd_ptr), nfds, -1);
      if (ready < 0) {
        if (errno == EINTR) {
          continue;
        }
        system_error("poll");
      }
      if (stop_fd_ >= 0 &&
          (pfds[1].revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
        return std::nullopt;
      }

      std::array<char, 8192> chunk{};
      const auto received = ::recv(socket_, chunk.data(), chunk.size(), 0);
      if (received == 0) {
        if (buffer_.empty()) {
          return std::nullopt;
        }
        if (buffer_.size() > kMaximumMessageBytes) {
          throw MessageTooLargeError("sidecar message exceeds the size limit");
        }
        std::string tail = std::move(buffer_);
        buffer_.clear();
        scan_offset_ = 0;
        if (!tail.empty() && tail.back() == '\r') {
          tail.pop_back();
        }
        return tail;
      }
      if (received < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
          continue;
        }
        if (errno == ECONNRESET || errno == EPIPE || errno == ENOTCONN ||
            errno == EBADF) {
          return std::nullopt;
        }
        system_error("recv");
      }
      buffer_.append(chunk.data(), static_cast<std::size_t>(received));
    }
  }

  [[nodiscard]] bool has_buffered_data() const noexcept {
    return !buffer_.empty();
  }

private:
  int socket_;
  int stop_fd_;
  std::string buffer_;
  std::size_t scan_offset_ = 0;
};

[[nodiscard]] bool write_all(int socket, std::string value) {
  value.push_back('\n');
  std::size_t offset = 0;
  while (offset < value.size()) {
    const auto written =
        ::send(socket, value.data() + offset, value.size() - offset, 0);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        pollfd pfd{.fd = socket, .events = POLLOUT, .revents = 0};
        (void)::poll(&pfd, 1, 100);
        continue;
      }
      if (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN ||
          errno == EBADF) {
        return false;
      }
      system_error("send");
    }
    offset += static_cast<std::size_t>(written);
  }
  return true;
}

std::string_view to_string(omlx::m5::EventKind value) {
  switch (value) {
  case omlx::m5::EventKind::tokens:
    return "tokens";
  case omlx::m5::EventKind::finished:
    return "finished";
  case omlx::m5::EventKind::aborted:
    return "aborted";
  case omlx::m5::EventKind::failed:
    return "failed";
  }
  throw std::invalid_argument("unknown event kind");
}

Json event_json(const omlx::m5::TokenEvent &event) {
  return {
      {"request_id", event.request_id},
      {"tokens", event.tokens},
      {"kind", to_string(event.kind)},
      {"finish_reason", event.finish_reason},
      {"error", event.error},
  };
}

omlx::m5::SpeculativeBackend speculative_backend(std::string_view value) {
  if (value == "none") {
    return omlx::m5::SpeculativeBackend::none;
  }
  if (value == "mtp") {
    return omlx::m5::SpeculativeBackend::mtp;
  }
  if (value == "dflash") {
    return omlx::m5::SpeculativeBackend::dflash;
  }
  if (value == "dflash2") {
    return omlx::m5::SpeculativeBackend::dflash2;
  }
  throw std::invalid_argument("unsupported speculative backend");
}

std::string override_value_to_string(const Json &value,
                                     std::string_view key,
                                     bool bool_as_on_off = true) {
  if (value.is_string()) {
    return value.get<std::string>();
  }
  if (value.is_boolean() && bool_as_on_off) {
    return value.get<bool>() ? "on" : "off";
  }
  if (value.is_number_unsigned()) {
    return std::to_string(value.get<std::uint64_t>());
  }
  if (value.is_number_integer()) {
    return std::to_string(value.get<std::int64_t>());
  }
  throw std::invalid_argument(std::string(key) +
                              " has an unsupported override value type");
}

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

std::string parse_lora_mode_strict(std::string_view value,
                                   std::string_view source) {
  if (value == "merge" || value == "bypass") {
    return std::string(value);
  }
  throw std::invalid_argument("unsupported " + std::string(source) +
                              " (expected 'merge' or 'bypass'): " +
                              std::string(value));
}

struct CliOptions {
  bool show_version = false;
  bool show_help = false;
  bool exit_on_disconnect = false;
  std::filesystem::path socket_path;
  std::optional<std::string> default_lora_adapter;
  std::optional<std::string> default_lora_mode;
};

omlx::m5::LoadOptions load_options(const Json &value,
                                   const CliOptions &cli_defaults = {}) {
  omlx::m5::LoadOptions result;
  result.model_dir = expand_user_path(value.value("model_dir", ""));
  result.draft_dir = expand_user_path(value.value("draft_dir", ""));
  result.max_concurrency = value.value("max_concurrency", 1);
  result.max_context = value.value("max_context", 32768);
  result.kv_dtype = value.value("kv_dtype", "bf16");
  result.prefill_engine = value.value("prefill_engine", "auto");
  result.spec_search = value.value("spec_search", "auto");
  result.sampling_contract =
      value.value("sampling_contract", "legacy_rejection_v1");
  if (value.contains("memory_budget_bytes") &&
      !value.at("memory_budget_bytes").is_null()) {
    const auto &mb = value.at("memory_budget_bytes");
    if (mb.is_number_unsigned()) {
      result.memory_budget_bytes = mb.get<std::uint64_t>();
    } else if (mb.is_number_integer() && mb.get<std::int64_t>() >= 0) {
      result.memory_budget_bytes =
          static_cast<std::uint64_t>(mb.get<std::int64_t>());
    } else {
      throw std::invalid_argument(
          "memory_budget_bytes must be a non-negative integer");
    }
  }

  if (value.contains("speculative")) {
    const auto &speculative = value.at("speculative");
    if (!speculative.is_object()) {
      throw std::invalid_argument("speculative must be an object");
    }
    result.speculative.backend =
        speculative_backend(speculative.value("backend", "none"));
    result.speculative.draft_tokens = speculative.value("draft_tokens", 0U);
  } else {
    if (value.contains("speculative_backend")) {
      result.speculative.backend = speculative_backend(
          value.at("speculative_backend").get<std::string>());
    }
    if (value.contains("draft_tokens")) {
      result.speculative.draft_tokens =
          value.at("draft_tokens").get<std::uint32_t>();
    }
  }

  const auto overrides = value.value("path_overrides", Json::object());
  if (!overrides.is_object()) {
    throw std::invalid_argument("path_overrides must be an object");
  }
  for (const auto &[key, selected] : overrides.items()) {
    result.path_overrides.emplace(key, override_value_to_string(selected, key));
  }

  // Expose all runtime/backend load tunables cleanly at top-level as well as
  // through path_overrides so sidecar clients can pass structured JSON fields.
  constexpr std::array<std::pair<std::string_view, std::string_view>, 23>
      kTopLevelOverrideAliases{{
          {"prefix_cache_bytes", "prefix_cache_bytes"},
          {"pin_prefix_min_tokens", "pin_prefix_min_tokens"},
          {"rolling_prefix_cache", "rolling_prefix_cache"},
          {"flash_nax_prefill", "flash_nax_prefill"},
          {"batch_mode", "batch_mode"},
          {"batch_projection_prototype", "batch_projection_prototype"},
          {"m4_short_routes", "m4_short_routes"},
          {"attention_split_kv", "attention_split_kv"},
          {"target_verify_selector", "target_verify_selector"},
          {"artifact", "artifact"},
          {"artifact_path", "artifact"},
          {"draft_artifact", "draft_artifact"},
          {"draft_artifact_path", "draft_artifact"},
          {"kernel_directory", "kernel_directory"},
          {"qwen_ar_metallib", "qwen_ar_metallib"},
          {"dflash2_metallib", "dflash2_metallib"},
          {"linear_route_ab", "linear_route_ab"},
          {"moe_route_ab", "moe_route_ab"},
          {"enable_mtp", "enable_mtp"},
          {"mtp_draft_tokens", "mtp_draft_tokens"},
          {"mtp_fixed_depth", "mtp_fixed_depth"},
          {"mtp_artifact", "mtp_artifact"},
          {"mtp_artifact_path", "mtp_artifact_path"},
      }};
  for (const auto &[json_key, override_key] : kTopLevelOverrideAliases) {
    const std::string key_str(json_key);
    const std::string target_key(override_key);
    if (value.contains(key_str) && !value.at(key_str).is_null() &&
        !result.path_overrides.contains(target_key)) {
      const auto serialized =
          override_value_to_string(value.at(key_str), json_key);
      if (!serialized.empty()) {
        result.path_overrides.emplace(target_key, serialized);
      }
    }
  }
  if (!result.path_overrides.contains("prefix_cache_bytes") &&
      value.contains("prefix_cache_gb") &&
      !value.at("prefix_cache_gb").is_null()) {
    const double gb = value.at("prefix_cache_gb").get<double>();
    if (!(gb >= 0.0) || gb > 128.0) {
      throw std::invalid_argument("prefix_cache_gb must be in [0, 128]");
    }
    const auto bytes =
        static_cast<std::uint64_t>(gb * static_cast<double>(1ULL << 30));
    result.path_overrides.emplace("prefix_cache_bytes", std::to_string(bytes));
  }

  auto read_lora_adapter_key = [&](const char *key) -> std::optional<std::string> {
    if (!value.contains(key)) {
      return std::nullopt;
    }
    const auto &node = value.at(key);
    if (node.is_null()) {
      return std::string{};
    }
    if (node.is_string()) {
      return expand_user_path(node.get<std::string>());
    }
    throw std::invalid_argument(std::string(key) + " must be a string or null");
  };

  if (const auto explicit_lora = read_lora_adapter_key("lora_adapter");
      explicit_lora.has_value()) {
    result.lora_adapter = *explicit_lora;
  } else if (const auto alias_lora = read_lora_adapter_key("lora");
             alias_lora.has_value()) {
    result.lora_adapter = *alias_lora;
  } else if (const auto it = result.path_overrides.find("lora_adapter");
             it != result.path_overrides.end()) {
    result.lora_adapter = expand_user_path(it->second);
  } else if (cli_defaults.default_lora_adapter.has_value()) {
    result.lora_adapter = *cli_defaults.default_lora_adapter;
  }

  const std::string fallback_mode =
      cli_defaults.default_lora_mode.value_or("merge");
  if (value.contains("lora_mode") && !value.at("lora_mode").is_null()) {
    if (!value.at("lora_mode").is_string()) {
      throw std::invalid_argument("lora_mode must be a string");
    }
    const std::string raw_mode = value.at("lora_mode").get<std::string>();
    result.lora_mode = raw_mode.empty()
                           ? fallback_mode
                           : parse_lora_mode_strict(raw_mode, "lora_mode");
  } else if (const auto it = result.path_overrides.find("lora_mode");
             it != result.path_overrides.end() && !it->second.empty()) {
    result.lora_mode = parse_lora_mode_strict(it->second, "lora_mode");
  } else {
    result.lora_mode = fallback_mode;
  }

  return result;
}

omlx::m5::FakeBackendOptions fake_options(const Json &value) {
  if (!value.is_object()) {
    throw std::invalid_argument("fake_options must be an object");
  }
  return {
      .event_batch_size = value.value("event_batch_size", 1),
      .step_delay_us = value.value("step_delay_us", 0),
      .fail_after_tokens = value.value("fail_after_tokens", -1),
      .fail_before_first = value.value("fail_before_first", false),
      .vocab_size = value.value("vocab_size", 248320),
  };
}

omlx::m5::SubmitRequest submit_request(const Json &value) {
  omlx::m5::SubmitRequest result;
  result.prompt_tokens =
      value.at("prompt_tokens").get<std::vector<std::int32_t>>();
  result.max_new_tokens = value.value("max_new_tokens", 1);
  const auto sampling = value.value("sampling", Json::object());
  if (!sampling.is_object()) {
    throw std::invalid_argument("sampling must be an object");
  }
  result.sampling.temperature = sampling.value("temperature", 0.0F);
  result.sampling.top_p = sampling.value("top_p", 1.0F);
  result.sampling.min_p = sampling.value("min_p", 0.0F);
  result.sampling.top_k = sampling.value("top_k", 0);
  result.sampling.seed = sampling.value("seed", 0ULL);
  result.stop_token_seqs =
      value.value("stop_token_seqs", std::vector<std::vector<std::int32_t>>{});
  result.pin_prefix = value.value("pin_prefix", false);
  return result;
}

Json capacity_json(const omlx::m5::CapacityReport &value) {
  return {
      {"max_concurrency", value.max_concurrency},
      {"active_requests", value.active_requests},
      {"queued_requests", value.queued_requests},
      {"accepting", value.accepting},
      {"pid", static_cast<std::int64_t>(::getpid())},
      {"phys_footprint_bytes", current_phys_footprint_bytes()},
  };
}

Json stats_json(const omlx::m5::StatsReport &value) {
  return {
      {"pid", static_cast<std::int64_t>(::getpid())},
      {"phys_footprint_bytes", current_phys_footprint_bytes()},
      {"submitted", value.submitted},
      {"completed", value.completed},
      {"aborted", value.aborted},
      {"failed", value.failed},
      {"emitted_tokens", value.emitted_tokens},
      {"backend_kind", value.backend_kind},
      {"model_execution_kind", value.model_execution_kind},
      {"sampling_contract_id", value.sampling_contract_id},
      {"numeric_contract_id", value.numeric_contract_id},
      {"chunk_policy_id", value.chunk_policy_id},
      {"execution_identity_id", value.execution_identity_id},
      {"compiled_execution_source_id", value.compiled_execution_source_id},
      {"kernel_bundle_id", value.kernel_bundle_id},
      {"batch_execution_kind", value.batch_execution_kind},
      {"batch_sampling_host_read_bytes", value.batch_sampling_host_read_bytes},
      {"batch_compact_fallback_rows", value.batch_compact_fallback_rows},
      {"shared_head_projection_calls", value.shared_head_projection_calls},
      {"shared_head_multirow_calls", value.shared_head_multirow_calls},
      {"shared_head_projection_rows", value.shared_head_projection_rows},
      {"shared_head_projection_gpu_ms", value.shared_head_projection_gpu_ms},

      {"spec_gpu_timed_rounds", value.spec_gpu_timed_rounds},
      {"spec_rounds", value.spec_rounds},
      {"spec_cost_completed_rounds", value.spec_cost_completed_rounds},
      {"spec_cost_aborted_rounds", value.spec_cost_aborted_rounds},
      {"spec_cost_committed_tokens", value.spec_cost_committed_tokens},
      {"spec_cost_round_wall_ms", value.spec_cost_round_wall_ms},
      {"spec_cost_phase_wall_ms", value.spec_cost_phase_wall_ms},
      {"spec_cost_depth_rounds", value.spec_cost_depth_rounds},
      {"spec_cost_depth_committed_tokens", value.spec_cost_depth_committed_tokens},
      {"spec_cost_depth_wall_ms", value.spec_cost_depth_wall_ms},

      {"spec_target_verify_calls", value.spec_target_verify_calls},
      {"spec_replay_calls", value.spec_replay_calls},
      {"spec_proposal_tokens", value.spec_proposal_tokens},
      {"spec_accepted_tokens", value.spec_accepted_tokens},
      {"spec_host_read_bytes", value.spec_host_read_bytes},
      {"spec_context_rows_projected", value.spec_context_rows_projected},
      {"spec_context_rows_reused", value.spec_context_rows_reused},
      {"spec_context_rows_launched", value.spec_context_rows_launched},
      {"short_matrix_calls", value.short_matrix_calls},
      {"short_lane_calls", value.short_lane_calls},
      {"spec_draft_gpu_ms", value.spec_draft_gpu_ms},
      {"spec_verify_gpu_ms", value.spec_verify_gpu_ms},
      {"spec_verify_selector_gpu_ms", value.spec_verify_selector_gpu_ms},
      {"spec_commit_gpu_ms", value.spec_commit_gpu_ms},
      {"batch_submissions", value.batch_submissions},
      {"batch_rows", value.batch_rows},
      {"batch_fallback_steps", value.batch_fallback_steps},
      {"batch_size_counts", value.batch_size_counts},
      {"batch_width_counts", value.batch_width_counts},
      {"batched_ar_steps", value.batched_ar_steps},
      {"batched_verify_rounds", value.batched_verify_rounds},
      {"batch_gpu_ms", value.batch_gpu_ms},
      {"batch_sampler_icb_replays", value.batch_sampler_icb_replays},
      {"actual_b", value.actual_b},
      {"actual_w", value.actual_w},
      {"prefix_hits", value.prefix_hits},
      {"prefix_partial_hits", value.prefix_partial_hits},
      {"prefix_misses", value.prefix_misses},
      {"prefix_evictions", value.prefix_evictions},
      {"prefix_restored_tokens", value.prefix_restored_tokens},
      {"ar_steps", value.ar_steps},
      {"ar_gpu_timed_steps", value.ar_gpu_timed_steps},
      {"ar_layer_calls", value.ar_layer_calls},
      {"ar_host_read_bytes", value.ar_host_read_bytes},
      {"ar_prefill_gpu_ms", value.ar_prefill_gpu_ms},
      {"ar_decode_gpu_ms", value.ar_decode_gpu_ms},
      {"artifact_id", value.artifact_id},
      {"route_table_id", value.route_table_id},
      {"actual_t", value.actual_t},
      {"actual_m", value.actual_m},
      {"actual_k", value.actual_k},
      {"kernel_family", value.kernel_family},
      {"fallback_reason", value.fallback_reason},
      {"spec_search_gated_rounds", value.spec_search_gated_rounds},
      {"spec_search_probe_rounds", value.spec_search_probe_rounds},
      {"prefix_shared_hits", value.prefix_shared_hits},
      {"prefix_pinned_entries", value.prefix_pinned_entries},
      {"contended_prefill_chunks", value.contended_prefill_chunks},
      {"packed_batch_calls", value.packed_batch_calls},
      {"packed_batch_tokens", value.packed_batch_tokens},
      {"lora_adapter", value.lora_adapter},
      {"lora_mode", value.lora_mode},
      {"lora_buffer_bytes", value.lora_buffer_bytes},
      {"lora_adapted_layers", value.lora_adapted_layers},
  };
}

bool base_load_options_equal(const omlx::m5::LoadOptions &a,
                             const omlx::m5::LoadOptions &b, bool fake_a,
                             bool fake_b,
                             const omlx::m5::FakeBackendOptions &fa,
                             const omlx::m5::FakeBackendOptions &fb) {
  if (fake_a != fake_b) {
    return false;
  }
  if (fake_a) {
    if (fa.event_batch_size != fb.event_batch_size ||
        fa.step_delay_us != fb.step_delay_us ||
        fa.fail_after_tokens != fb.fail_after_tokens ||
        fa.fail_before_first != fb.fail_before_first ||
        fa.vocab_size != fb.vocab_size) {
      return false;
    }
  }
  return a.model_dir == b.model_dir && a.draft_dir == b.draft_dir &&
         a.max_concurrency == b.max_concurrency &&
         a.max_context == b.max_context &&
         a.speculative.backend == b.speculative.backend &&
         a.speculative.draft_tokens == b.speculative.draft_tokens &&
         a.kv_dtype == b.kv_dtype && a.prefill_engine == b.prefill_engine &&
         a.path_overrides == b.path_overrides &&
         a.sampling_contract == b.sampling_contract &&
         a.spec_search == b.spec_search &&
         a.memory_budget_bytes == b.memory_budget_bytes;
}

class Session final {
public:
  explicit Session(CliOptions cli_options = {})
      : cli_options_(std::move(cli_options)) {}

  Json execute(std::string_view method, const Json &params) {
    if (method == "hello") {
      return {
          {"provider_id", "io.ninfer.metal"},
          {"provider_version", NINFER_METAL_DISPLAY_VERSION},
          {"codename", NINFER_METAL_CODENAME},
          {"semver", NINFER_METAL_SEMVER},
          {"core_abi", NINFER_METAL_CORE_ABI},
          {"protocol_version", kProtocolVersion},
          {"engine_contract_version", 1},
          {"pid", static_cast<std::int64_t>(::getpid())},
          {"phys_footprint_bytes", current_phys_footprint_bytes()},
          {"capabilities",
           {"ar", "mtp", "dflash2", "lora", "reload_lora", "kv_affine4_g128",
            "rolling_prefix_cache", "trim_cache", "memory_budget_bytes",
            "set_scheduler_hint", "cancel", "stats", "capacity",
            "subscribe_events"}},
          {"kv_formats", {"bf16", "affine4_g128"}},
          {"supported_standard_settings",
           {"max_context_window", "lora_adapter", "mtp_enabled",
            "mtp_adaptive_max_depth", "mtp_fixed_depth", "dflash_enabled",
            "dflash_draft_model", "dflash_block_size", "turboquant_kv_enabled",
            "turboquant_kv_scheme"}},
          {"options_schema",
           Json::array({
               {{"key", "max_concurrency"},
                {"label", "Concurrency (B/C)"},
                {"type", "int"},
                {"min", 1},
                {"max", 8},
                {"default", 1},
                {"reload_required", true}},
               {{"key", "rolling_prefix_cache"},
                {"label", "Rolling Prefix Cache"},
                {"type", "bool"},
                {"default", true},
                {"reload_required", true}},
               {{"key", "prefix_cache_gb"},
                {"label", "Prefix Cache Budget (GiB)"},
                {"type", "bytes_gb"},
                {"min", 0},
                {"max", 128},
                {"default", 4},
                {"reload_required", true}},
               {{"key", "sampling_contract"},
                {"label", "Sampling Contract"},
                {"type", "enum"},
                {"options",
                 Json::array({"position_coupled_v1", "legacy_rejection_v1"})},
                {"default", "legacy_rejection_v1"},
                {"reload_required", true}},
               {{"key", "lora_mode"},
                {"label", "LoRA Mode"},
                {"type", "enum"},
                {"options", Json::array({"merge", "bypass"})},
                {"default", "merge"},
                {"reload_required", true}},
               {{"key", "artifact_path"},
                {"label", "Artifact Path"},
                {"type", "string"},
                {"default", ""},
                {"reload_required", true}},
               {{"key", "draft_artifact_path"},
                {"label", "Draft Artifact Path"},
                {"type", "string"},
                {"default", ""},
                {"reload_required", true}},
           })},
      };
    }
    if (method == "load") {
      // Validate parameters before acquiring or replacing any loaded engine.
      const auto options = load_options(params, cli_options_);
      const bool is_fake = params.value("fake", false);
      const auto fopts =
          fake_options(params.value("fake_options", Json::object()));

      std::shared_ptr<omlx::m5::Engine> existing_engine;
      omlx::m5::LoadOptions prev_options;
      bool prev_fake = false;
      omlx::m5::FakeBackendOptions prev_fopts;
      {
        std::lock_guard lock(mutex_);
        if (stopping_ || shutdown_) {
          throw std::runtime_error("sidecar is shutting down");
        }
        if (loading_) {
          throw std::logic_error("an engine load is already in progress");
        }
        loading_ = true;
        existing_engine = engine_;
        prev_options = loaded_options_;
        prev_fake = loaded_fake_;
        prev_fopts = loaded_fake_options_;
      }

      if (existing_engine != nullptr) {
        const bool same_base = base_load_options_equal(
            prev_options, options, prev_fake, is_fake, prev_fopts, fopts);
        if (same_base && prev_options.lora_adapter == options.lora_adapter &&
            prev_options.lora_mode == options.lora_mode) {
          {
            std::lock_guard lock(mutex_);
            loading_ = false;
          }
          cv_.notify_all();
          Json res = stats_json(existing_engine->stats());
          res["hot_reloaded"] = false;
          return res;
        }
        if (same_base &&
            (options.lora_adapter.empty() || options.lora_mode == "bypass") &&
            (prev_options.lora_adapter.empty() ||
             prev_options.lora_mode == "bypass")) {
          std::string reload_err;
          if (existing_engine->reload_lora(options.lora_adapter,
                                           options.lora_mode, &reload_err)) {
            {
              std::lock_guard lock(mutex_);
              loaded_options_ = options;
              loading_ = false;
            }
            cv_.notify_all();
            Json res = stats_json(existing_engine->stats());
            res["hot_reloaded"] = true;
            return res;
          }
        }
        // Full reload fallback (e.g. merge mode or changed base options):
        // tear down old engine before allocating new UMA Metal buffers.
        {
          std::lock_guard lock(mutex_);
          engine_.reset();
        }
        cv_.notify_all();
        existing_engine->shutdown();
        existing_engine.reset();
      }

      std::shared_ptr<omlx::m5::Engine> created;
      try {
        if (is_fake) {
          created = std::shared_ptr<omlx::m5::Engine>(
              omlx::m5::Engine::create_fake(options, fopts));
        } else {
          created = std::shared_ptr<omlx::m5::Engine>(
              omlx::m5::Engine::create(options));
        }
      } catch (...) {
        {
          std::lock_guard lock(mutex_);
          loading_ = false;
        }
        cv_.notify_all();
        throw;
      }
      bool stop_immediately = false;
      {
        std::lock_guard lock(mutex_);
        loading_ = false;
        if (stopping_ || shutdown_) {
          stop_immediately = true;
        } else {
          engine_ = created;
          loaded_options_ = options;
          loaded_fake_ = is_fake;
          loaded_fake_options_ = fopts;
        }
      }
      cv_.notify_all();
      if (stop_immediately) {
        created->shutdown();
        throw std::runtime_error("sidecar is shutting down");
      }
      Json res = stats_json(created->stats());
      res["hot_reloaded"] = false;
      return res;
    }
    if (method == "reload_lora") {
      std::string new_adapter;
      if (params.contains("lora_adapter") && !params.at("lora_adapter").is_null()) {
        if (!params.at("lora_adapter").is_string()) {
          throw std::invalid_argument("lora_adapter must be a string or null");
        }
        new_adapter = expand_user_path(params.at("lora_adapter").get<std::string>());
      } else if (params.contains("lora") && !params.at("lora").is_null()) {
        if (!params.at("lora").is_string()) {
          throw std::invalid_argument("lora must be a string or null");
        }
        new_adapter = expand_user_path(params.at("lora").get<std::string>());
      }

      std::optional<std::string> explicit_mode;
      if (params.contains("lora_mode") && !params.at("lora_mode").is_null()) {
        if (!params.at("lora_mode").is_string()) {
          throw std::invalid_argument("lora_mode must be a string");
        }
        const std::string raw = params.at("lora_mode").get<std::string>();
        if (!raw.empty()) {
          explicit_mode = parse_lora_mode_strict(raw, "lora_mode");
        }
      }

      std::shared_ptr<omlx::m5::Engine> existing_engine;
      omlx::m5::LoadOptions prev_options;
      bool prev_fake = false;
      omlx::m5::FakeBackendOptions prev_fopts;
      {
        std::lock_guard lock(mutex_);
        if (stopping_ || shutdown_) {
          throw std::runtime_error("sidecar is shutting down");
        }
        if (loading_) {
          throw std::logic_error("an engine load is already in progress");
        }
        if (engine_ == nullptr) {
          throw std::logic_error("no engine is loaded");
        }
        loading_ = true;
        existing_engine = engine_;
        prev_options = loaded_options_;
        prev_fake = loaded_fake_;
        prev_fopts = loaded_fake_options_;
      }

      const std::string target_mode = explicit_mode.value_or(
          (!prev_options.lora_mode.empty() && prev_options.lora_mode == "bypass")
              ? std::string("bypass")
              : std::string("merge"));

      const bool can_try_hot =
          (new_adapter.empty() || target_mode == "bypass") &&
          (prev_options.lora_adapter.empty() ||
           prev_options.lora_mode == "bypass");
      std::string reload_err;
      if (can_try_hot &&
          existing_engine->reload_lora(new_adapter, target_mode, &reload_err)) {
        {
          std::lock_guard lock(mutex_);
          loaded_options_.lora_adapter = new_adapter;
          loaded_options_.lora_mode = target_mode;
          loading_ = false;
        }
        cv_.notify_all();
        Json res = stats_json(existing_engine->stats());
        res["hot_reloaded"] = true;
        return res;
      }

      // If hot-swap failed for a bypass adapter validation reason (rather than
      // merge mode requiring full reload), preserve existing_engine and surface
      // the error cleanly.
      if (can_try_hot && target_mode == "bypass" && !new_adapter.empty() &&
          reload_err.find("full") == std::string::npos &&
          reload_err.find("reload") == std::string::npos) {
        {
          std::lock_guard lock(mutex_);
          loading_ = false;
        }
        cv_.notify_all();
        throw std::runtime_error(reload_err.empty() ? "reload_lora failed"
                                                    : reload_err);
      }

      omlx::m5::LoadOptions updated_options = prev_options;
      updated_options.lora_adapter = new_adapter;
      updated_options.lora_mode = target_mode;
      {
        std::lock_guard lock(mutex_);
        engine_.reset();
      }
      cv_.notify_all();
      existing_engine->shutdown();
      existing_engine.reset();

      std::shared_ptr<omlx::m5::Engine> created;
      try {
        if (prev_fake) {
          created = std::shared_ptr<omlx::m5::Engine>(
              omlx::m5::Engine::create_fake(updated_options, prev_fopts));
        } else {
          created = std::shared_ptr<omlx::m5::Engine>(
              omlx::m5::Engine::create(updated_options));
        }
      } catch (...) {
        {
          std::lock_guard lock(mutex_);
          loading_ = false;
        }
        cv_.notify_all();
        throw;
      }
      bool stop_immediately = false;
      {
        std::lock_guard lock(mutex_);
        loading_ = false;
        if (stopping_ || shutdown_) {
          stop_immediately = true;
        } else {
          engine_ = created;
          loaded_options_ = updated_options;
        }
      }
      cv_.notify_all();
      if (stop_immediately) {
        created->shutdown();
        throw std::runtime_error("sidecar is shutting down");
      }
      Json res = stats_json(created->stats());
      res["hot_reloaded"] = false;
      return res;
    }
    if (method == "shutdown") {
      shutdown();
      return {{"shutdown", true}};
    }

    const auto engine = require_engine();
    if (method == "submit") {
      return {{"request_id", engine->submit(submit_request(params))}};
    }
    if (method == "cancel") {
      return {{"cancelled",
               engine->cancel(params.at("request_id").get<std::uint64_t>())}};
    }
    if (method == "poll") {
      const auto max_wait_us = params.value("max_wait_us", 20'000LL);
      if (max_wait_us < 0 ||
          max_wait_us > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("max_wait_us is invalid");
      }
      Json events = Json::array();
      for (const auto &event :
           engine->poll_events(std::chrono::microseconds(max_wait_us))) {
        events.push_back(event_json(event));
      }
      return {{"events", std::move(events)}};
    }
    if (method == "capacity") {
      return capacity_json(engine->capacity());
    }
    if (method == "stats") {
      return stats_json(engine->stats());
    }
    if (method == "trim_cache") {
      std::uint64_t target_free_bytes = 0;
      if (params.contains("target_free_bytes") &&
          !params.at("target_free_bytes").is_null()) {
        const auto &val = params.at("target_free_bytes");
        if (val.is_number_unsigned()) {
          target_free_bytes = val.get<std::uint64_t>();
        } else if (val.is_number_integer() && val.get<std::int64_t>() >= 0) {
          target_free_bytes =
              static_cast<std::uint64_t>(val.get<std::int64_t>());
        } else {
          throw std::invalid_argument(
              "target_free_bytes must be a non-negative integer");
        }
      }
      const auto report = engine->trim_cache(target_free_bytes);
      const auto phys = current_phys_footprint_bytes();
      return {
          {"freed_bytes", report.freed_bytes},
          {"evicted_entries", report.evicted_entries},
          {"resident_bytes", report.resident_bytes > 0 ? report.resident_bytes : phys},
          {"metal_resident_bytes", report.resident_bytes},
          {"phys_footprint_bytes", phys},
      };
    }
    if (method == "set_scheduler_hint") {
      bool gpu_contended = false;
      if (params.contains("gpu_contended") &&
          !params.at("gpu_contended").is_null()) {
        if (!params.at("gpu_contended").is_boolean()) {
          throw std::invalid_argument("gpu_contended must be a boolean");
        }
        gpu_contended = params.at("gpu_contended").get<bool>();
      }
      engine->set_scheduler_hint({.gpu_contended = gpu_contended});
      return {{"gpu_contended", gpu_contended}};
    }
    throw std::invalid_argument("unknown sidecar method");
  }

  void shutdown() {
    std::lock_guard call_lock(shutdown_call_mutex_);
    std::shared_ptr<omlx::m5::Engine> engine_to_stop;
    {
      std::lock_guard lock(mutex_);
      if (shutdown_) {
        return;
      }
      stopping_ = true;
      engine_to_stop = std::move(engine_);
    }
    cv_.notify_all();
    if (engine_to_stop != nullptr) {
      engine_to_stop->shutdown();
    }
    {
      std::lock_guard lock(mutex_);
      shutdown_ = true;
    }
    cv_.notify_all();
  }

  [[nodiscard]] std::shared_ptr<omlx::m5::Engine>
  wait_for_engine(std::chrono::microseconds max_wait) {
    std::unique_lock lock(mutex_);
    if (engine_ != nullptr) {
      return engine_;
    }
    if (stopping_ || shutdown_) {
      return nullptr;
    }
    cv_.wait_for(lock, max_wait,
                 [this] { return engine_ != nullptr || stopping_ || shutdown_; });
    return engine_;
  }

  [[nodiscard]] bool shutdown_requested() const {
    std::lock_guard lock(mutex_);
    return stopping_ || shutdown_;
  }

private:
  [[nodiscard]] std::shared_ptr<omlx::m5::Engine> require_engine() const {
    std::lock_guard lock(mutex_);
    if (stopping_ || shutdown_) {
      throw std::runtime_error("sidecar is shutting down");
    }
    if (engine_ == nullptr) {
      throw std::logic_error("no engine is loaded");
    }
    return engine_;
  }

  CliOptions cli_options_;
  mutable std::mutex mutex_;
  std::mutex shutdown_call_mutex_;
  std::condition_variable cv_;
  std::shared_ptr<omlx::m5::Engine> engine_;
  omlx::m5::LoadOptions loaded_options_;
  bool loaded_fake_ = false;
  omlx::m5::FakeBackendOptions loaded_fake_options_;
  bool loading_ = false;
  bool stopping_ = false;
  bool shutdown_ = false;
};

void stream_subscribed_events(int client_fd, int stop_read_fd, Session &session,
                              std::chrono::microseconds poll_interval) {
  std::shared_ptr<omlx::m5::Engine> last_engine;
  for (;;) {
    if (is_stop_signaled(stop_read_fd)) {
      session.shutdown();
    }
    auto engine = session.wait_for_engine(poll_interval);
    if (engine != nullptr) {
      last_engine = engine;
      const auto events = engine->poll_events(poll_interval);
      for (const auto &event : events) {
        const Json frame = {{"event", event_json(event)}};
        if (!write_all(client_fd, frame.dump())) {
          return;
        }
      }
    }

    if (session.shutdown_requested() || is_stop_signaled(stop_read_fd)) {
      session.shutdown();
      if (last_engine != nullptr) {
        for (const auto &event :
             last_engine->poll_events(std::chrono::microseconds(0))) {
          const Json frame = {{"event", event_json(event)}};
          if (!write_all(client_fd, frame.dump())) {
            return;
          }
        }
      }
      const Json eof_frame = {{"event", {{"kind", "eof"}}}};
      (void)write_all(client_fd, eof_frame.dump());
      return;
    }

    char peek_byte = 0;
    const auto peeked =
        ::recv(client_fd, &peek_byte, 1, MSG_PEEK | MSG_DONTWAIT);
    if (peeked == 0 || (peeked < 0 && errno != EAGAIN &&
                        errno != EWOULDBLOCK && errno != EINTR)) {
      return;
    }
  }
}

void handle_client(FileDescriptor client, int stop_read_fd, int stop_write_fd,
                   bool exit_on_disconnect, Session &session,
                   bool &served_any_request) {
  LineReader reader(client.get(), stop_read_fd);
  try {
    while (const auto line = reader.read_line()) {
      served_any_request = true;
      Json response;
      Json id;
      bool enter_subscription = false;
      std::chrono::microseconds poll_interval{50'000};
      try {
        const auto request = Json::parse(*line);
        id = request.at("id");
        if (!id.is_number_unsigned() && !id.is_number_integer() &&
            !id.is_string()) {
          throw std::invalid_argument("request id has an unsupported type");
        }
        const auto method = request.at("method").get<std::string>();
        const auto params = request.value("params", Json::object());
        if (!params.is_object()) {
          throw std::invalid_argument("request params must be an object");
        }
        if (method == "subscribe_events") {
          const auto interval_us = params.value("poll_interval_us", 50'000LL);
          if (interval_us <= 0 ||
              interval_us > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("poll_interval_us is invalid");
          }
          poll_interval = std::chrono::microseconds(interval_us);
          enter_subscription = true;
          response = {{"id", id}, {"ok", true}, {"result", {{"subscribed", true}}}};
        } else {
          response = {{"id", id},
                      {"ok", true},
                      {"result", session.execute(method, params)}};
        }
      } catch (const std::exception &error) {
        enter_subscription = false;
        response = {
            {"id", id},
            {"ok", false},
            {"error", {{"code", "request_failed"}, {"message", error.what()}}}};
      }

      const bool sent = write_all(client.get(), response.dump());
      if (session.shutdown_requested() && !enter_subscription) {
        trigger_wakeup_pipe(stop_write_fd);
        break;
      }
      if (!sent) {
        break;
      }
      if (enter_subscription) {
        stream_subscribed_events(client.get(), stop_read_fd, session,
                                 poll_interval);
        break;
      }
    }
  } catch (const MessageTooLargeError &error) {
    const Json err_response = {
        {"id", nullptr},
        {"ok", false},
        {"error", {{"code", "message_too_large"}, {"message", error.what()}}}};
    (void)write_all(client.get(), err_response.dump());
  } catch (...) {
    // Isolate per-connection transport failures from the daemon process.
  }
  (void)exit_on_disconnect;
}

FileDescriptor listen_socket(const std::filesystem::path &path) {
  if (path.native().size() >= sizeof(sockaddr_un::sun_path)) {
    throw std::invalid_argument("Unix socket path is too long");
  }
  std::error_code status_ec;
  const auto existing_status = std::filesystem::symlink_status(path, status_ec);
  if (existing_status.type() != std::filesystem::file_type::not_found) {
    if (status_ec || !std::filesystem::is_socket(existing_status)) {
      throw std::invalid_argument("Unix socket path already exists");
    }
    FileDescriptor probe(::socket(AF_UNIX, SOCK_STREAM, 0));
    if (probe.get() < 0) {
      system_error("socket");
    }
    set_cloexec(probe.get());
    configure_socket_nosigpipe(probe.get());
    sockaddr_un probe_addr{};
    probe_addr.sun_family = AF_UNIX;
    std::memcpy(probe_addr.sun_path, path.c_str(), path.native().size() + 1);
    if (::connect(probe.get(), reinterpret_cast<sockaddr *>(&probe_addr),
                  sizeof(probe_addr)) == 0) {
      throw std::invalid_argument("Unix socket path already exists");
    }
    if (errno != ECONNREFUSED && errno != ENOENT) {
      throw std::invalid_argument("Unix socket path already exists");
    }
    std::error_code remove_ec;
    std::filesystem::remove(path, remove_ec);
    if (remove_ec) {
      throw std::runtime_error("failed to remove stale Unix socket: " +
                               remove_ec.message());
    }
  }

  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  FileDescriptor socket(::socket(AF_UNIX, SOCK_STREAM, 0));
  if (socket.get() < 0) {
    system_error("socket");
  }
  set_cloexec(socket.get());
  configure_socket_nosigpipe(socket.get());
  set_nonblocking(socket.get(), true);

  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, path.c_str(), path.native().size() + 1);
  if (::bind(socket.get(), reinterpret_cast<sockaddr *>(&address),
             sizeof(address)) != 0) {
    system_error("bind");
  }
  if (::chmod(path.c_str(), S_IRUSR | S_IWUSR) != 0) {
    system_error("chmod");
  }
  if (::listen(socket.get(), kListenBacklog) != 0) {
    system_error("listen");
  }
  return socket;
}

std::string format_daemon_help() {
  return "Usage: mseries-backendd [--exit-on-disconnect] "
         "[--lora|--lora-adapter PATH] [--lora-mode merge|bypass] "
         "--socket PATH\n"
         "       mseries-backendd --version | --help\n\n"
         "Options:\n"
         "  --socket PATH                AF_UNIX socket path (or positional PATH)\n"
         "  --exit-on-disconnect         Exit when all active clients disconnect\n"
         "  --lora, --lora-adapter PATH  Default LoRA adapter directory for load RPC\n"
         "  --lora-mode merge|bypass     Default LoRA execution mode (default: merge)\n"
         "  --version                    Print daemon & protocol version and exit\n"
         "  --help, -h                   Show this help message and exit\n";
}

CliOptions parse_cli(int argc, char **argv) {
  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--version") {
      options.show_version = true;
      return options;
    }
    if (arg == "--help" || arg == "-h") {
      options.show_help = true;
      return options;
    }
  }

  bool have_socket = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view raw(argv[i]);
    std::string_view flag = raw;
    std::optional<std::string_view> inline_val;
    if (raw.starts_with("--")) {
      const auto eq = raw.find('=');
      if (eq != std::string_view::npos) {
        flag = raw.substr(0, eq);
        inline_val = raw.substr(eq + 1);
      }
    }
    const auto next_val = [&]() -> std::string {
      if (inline_val.has_value()) {
        return std::string(*inline_val);
      }
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(flag) + " requires a value");
      }
      return std::string(argv[++i]);
    };

    if (flag == "--exit-on-disconnect") {
      options.exit_on_disconnect = true;
      continue;
    }
    if (flag == "--socket") {
      if (have_socket) {
        throw std::invalid_argument(
            "usage: mseries-backendd [--exit-on-disconnect] "
            "[--lora|--lora-adapter PATH] [--lora-mode merge|bypass] "
            "--socket PATH | --version");
      }
      options.socket_path = expand_user_path(next_val());
      have_socket = true;
      continue;
    }
    if (flag == "--lora" || flag == "--lora-adapter") {
      options.default_lora_adapter = expand_user_path(next_val());
      continue;
    }
    if (flag == "--lora-mode") {
      options.default_lora_mode = parse_lora_mode_strict(next_val(), flag);
      continue;
    }
    if (!raw.empty() && raw.front() != '-' && !have_socket) {
      options.socket_path = expand_user_path(std::string(raw));
      have_socket = true;
      continue;
    }
    throw std::invalid_argument(
        "usage: mseries-backendd [--exit-on-disconnect] "
        "[--lora|--lora-adapter PATH] [--lora-mode merge|bypass] "
        "--socket PATH | --version");
  }
  if (!have_socket || options.socket_path.empty()) {
    throw std::invalid_argument(
        "usage: mseries-backendd [--exit-on-disconnect] "
        "[--lora|--lora-adapter PATH] [--lora-mode merge|bypass] "
        "--socket PATH | --version");
  }
  return options;
}

int run(const CliOptions &options) {
  int pipe_fds[2] = {-1, -1};
  if (::pipe(pipe_fds) != 0) {
    system_error("pipe");
  }
  FileDescriptor stop_read(pipe_fds[0]);
  FileDescriptor stop_write(pipe_fds[1]);
  set_cloexec(stop_read.get());
  set_cloexec(stop_write.get());
  set_nonblocking(stop_read.get(), true);
  set_nonblocking(stop_write.get(), true);

  SignalGuard signal_guard(stop_write.get());
  auto listener = listen_socket(options.socket_path);
  SocketPathGuard socket_guard(options.socket_path);

  Session session(options);
  std::mutex workers_mutex;
  std::condition_variable workers_cv;
  std::unordered_map<std::uint64_t, std::thread> active_workers;
  std::vector<std::thread> finished_workers;
  std::uint64_t next_conn_id = 1;

  const auto reap_finished_workers = [&]() {
    std::vector<std::thread> to_join;
    {
      std::lock_guard lock(workers_mutex);
      to_join.swap(finished_workers);
    }
    for (auto &worker : to_join) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  };

  for (;;) {
    reap_finished_workers();
    std::array<pollfd, 2> pfds{{
        {.fd = listener.get(), .events = POLLIN, .revents = 0},
        {.fd = stop_read.get(), .events = POLLIN, .revents = 0},
    }};
    const int ready = ::poll(pfds.data(), pfds.size(), -1);
    if (ready < 0) {
      if (errno == EINTR) {
        if (is_stop_signaled(stop_read.get())) {
          break;
        }
        continue;
      }
      system_error("poll");
    }
    if ((pfds[1].revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
      break;
    }
    if ((pfds[0].revents & POLLIN) == 0) {
      continue;
    }

    for (;;) {
      const int accepted = ::accept(listener.get(), nullptr, nullptr);
      if (accepted < 0) {
        if (errno == EINTR) {
          continue;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          break;
        }
        system_error("accept");
      }

      FileDescriptor client_fd(accepted);
      set_cloexec(client_fd.get());
      configure_socket_nosigpipe(client_fd.get());
      // Clear O_NONBLOCK inherited from the listening socket on BSD/macOS.
      set_nonblocking(client_fd.get(), false);

      std::lock_guard lock(workers_mutex);
      const std::uint64_t conn_id = next_conn_id++;
      active_workers.emplace(
          conn_id,
          std::thread([&, conn_id, fd = std::move(client_fd)]() mutable {
            bool served_any_request = false;
            handle_client(std::move(fd), stop_read.get(), stop_write.get(),
                          options.exit_on_disconnect, session,
                          served_any_request);
            {
              std::lock_guard done_lock(workers_mutex);
              const auto it = active_workers.find(conn_id);
              if (it != active_workers.end()) {
                finished_workers.push_back(std::move(it->second));
                active_workers.erase(it);
              }
              if (options.exit_on_disconnect && served_any_request &&
                  active_workers.empty()) {
                trigger_wakeup_pipe(stop_write.get());
              }
              workers_cv.notify_all();
            }
          }));
    }
  }

  // Ensure stop_read is readable so all idle LineReaders wake immediately,
  // and shut down the Engine so all poll_events / subscribe_events loops wake.
  trigger_wakeup_pipe(stop_write.get());
  session.shutdown();

  {
    std::unique_lock lock(workers_mutex);
    workers_cv.wait(lock, [&] { return active_workers.empty(); });
  }
  reap_finished_workers();
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = parse_cli(argc, argv);
    if (options.show_version) {
      std::cout << "mseries-backendd " << NINFER_METAL_DISPLAY_VERSION
                << " (codename=" << NINFER_METAL_CODENAME
                << " semver=" << NINFER_METAL_SEMVER
                << " core_abi=" << NINFER_METAL_CORE_ABI
                << ") protocol=" << kProtocolVersion << '\n';
      return 0;
    }
    if (options.show_help) {
      std::cout << format_daemon_help();
      return 0;
    }
    return run(options);
  } catch (const std::exception &error) {
    std::cerr << "mseries-backendd: " << error.what() << '\n';
    return 1;
  }
}
