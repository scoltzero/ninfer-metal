#include "serve/uds_sidecar_backend.h"

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef __APPLE__
#include <crt_externs.h>
#else
extern char **environ;
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace omlx::m5::serve {
namespace {

using Json = nlohmann::json;

constexpr std::size_t kMaxSidecarLineBytes = 16U * 1024U * 1024U;

char **current_environ() noexcept {
#ifdef __APPLE__
  char ***envp = _NSGetEnviron();
  return envp != nullptr ? *envp : nullptr;
#else
  return environ;
#endif
}

void set_cloexec(int fd) noexcept {
  if (fd < 0) {
    return;
  }
  const int flags = ::fcntl(fd, F_GETFD, 0);
  if (flags >= 0) {
    (void)::fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
  }
}

void configure_nosigpipe(int fd) noexcept {
#ifdef SO_NOSIGPIPE
  if (fd >= 0) {
    const int one = 1;
    (void)::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
  }
#else
  (void)fd;
#endif
}

struct EphemeralSocketLocation {
  std::filesystem::path dir;
  std::filesystem::path socket_path;
};

EphemeralSocketLocation make_ephemeral_socket_location() {
  std::error_code ec;
  std::filesystem::path base("/tmp");
  if (!std::filesystem::is_directory(base, ec)) {
    base = std::filesystem::temp_directory_path(ec);
  }
  const std::string tmpl_str = (base / "nf-spi-XXXXXX").string();
  std::vector<char> tmpl(tmpl_str.begin(), tmpl_str.end());
  tmpl.push_back('\0');
  char *created = ::mkdtemp(tmpl.data());
  if (created == nullptr) {
    throw std::runtime_error(
        std::string("mkdtemp failed for sidecar socket directory: ") +
        std::strerror(errno));
  }
  std::filesystem::path dir(created);
  if (::chmod(dir.c_str(), 0700) != 0) {
    const int err = errno;
    std::error_code rm_ec;
    std::filesystem::remove_all(dir, rm_ec);
    throw std::runtime_error("chmod(0700) failed on sidecar socket directory " +
                             dir.string() + ": " + std::strerror(err));
  }

  std::filesystem::path candidate = dir / "spi.sock";
  if (candidate.native().size() >= sizeof(sockaddr_un::sun_path)) {
    std::error_code rm_ec;
    std::filesystem::remove_all(dir, rm_ec);
    throw std::runtime_error("ephemeral Unix socket path exceeds sun_path: " +
                             candidate.string());
  }

  const auto st = std::filesystem::symlink_status(candidate, ec);
  if ((ec && ec != std::errc::no_such_file_or_directory) ||
      st.type() != std::filesystem::file_type::not_found) {
    std::error_code rm_file_ec;
    (void)std::filesystem::remove(candidate, rm_file_ec);
    std::error_code after_ec;
    const auto after_st = std::filesystem::symlink_status(candidate, after_ec);
    if (rm_file_ec || after_st.type() != std::filesystem::file_type::not_found) {
      std::error_code rm_dir_ec;
      std::filesystem::remove_all(dir, rm_dir_ec);
      throw std::runtime_error("failed to remove pre-existing sidecar socket " +
                               candidate.string() + ": " +
                               rm_file_ec.message());
    }
  }
  return EphemeralSocketLocation{.dir = std::move(dir),
                                 .socket_path = std::move(candidate)};
}

bool verify_peer_uid(int fd, std::string &error_out) noexcept {
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) ||      \
    defined(__NetBSD__)
  uid_t peer_uid = static_cast<uid_t>(-1);
  gid_t peer_gid = static_cast<gid_t>(-1);
  if (::getpeereid(fd, &peer_uid, &peer_gid) != 0) {
    error_out = std::string("getpeereid failed on sidecar socket: ") +
                std::strerror(errno);
    return false;
  }
  const uid_t self_uid = ::geteuid();
  if (peer_uid != self_uid) {
    error_out = "sidecar socket peer UID (" + std::to_string(peer_uid) +
                ") does not match current EUID (" + std::to_string(self_uid) +
                ")";
    return false;
  }
  return true;
#elif defined(SO_PEERCRED)
  struct ucred cred {};
  socklen_t len = sizeof(cred);
  if (::getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &len) != 0) {
    error_out = std::string("getsockopt(SO_PEERCRED) failed: ") +
                std::strerror(errno);
    return false;
  }
  const uid_t self_uid = ::geteuid();
  if (cred.uid != self_uid) {
    error_out = "sidecar socket peer UID (" + std::to_string(cred.uid) +
                ") does not match current EUID (" + std::to_string(self_uid) +
                ")";
    return false;
  }
  return true;
#else
  (void)fd;
  (void)error_out;
  return true;
#endif
}

std::string_view speculative_backend_to_string(SpeculativeBackend backend) {
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

EventKind parse_event_kind(std::string_view kind) {
  if (kind == "tokens") {
    return EventKind::tokens;
  }
  if (kind == "finished") {
    return EventKind::finished;
  }
  if (kind == "aborted") {
    return EventKind::aborted;
  }
  if (kind == "failed") {
    return EventKind::failed;
  }
  throw std::runtime_error("unknown sidecar event kind: " + std::string(kind));
}

TokenEvent parse_token_event(const Json &ev) {
  TokenEvent out;
  out.request_id = ev.value("request_id", 0ULL);
  if (ev.contains("tokens") && ev.at("tokens").is_array()) {
    out.tokens = ev.at("tokens").get<std::vector<std::int32_t>>();
  }
  out.kind = parse_event_kind(ev.at("kind").get<std::string>());
  if (ev.contains("finish_reason") && ev.at("finish_reason").is_string()) {
    out.finish_reason = ev.at("finish_reason").get<std::string>();
  }
  if (ev.contains("error") && ev.at("error").is_string()) {
    out.error = ev.at("error").get<std::string>();
  }
  const auto stats = ev.find("request_stats");
  if (stats != ev.end() && stats->is_object()) {
    out.request_stats = RequestStats{
        .prefix_restored_tokens = stats->value("prefix_restored_tokens", 0ULL),
        .spec_rounds = stats->value("spec_rounds", 0ULL),
        .ar_steps = stats->value("ar_steps", 0ULL),
        .decode_mode = stats->value("decode_mode", ""),
        .fallback_reason = stats->value("fallback_reason", ""),
    };
  }
  return out;
}

bool write_line(int fd, std::string line, std::chrono::milliseconds timeout) {
  line.push_back('\n');
  std::size_t offset = 0;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (offset < line.size()) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      return false;
    }
    const auto remaining_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
            .count();
    pollfd pfd{.fd = fd, .events = POLLOUT, .revents = 0};
    const int ready =
        ::poll(&pfd, 1, static_cast<int>(std::max<std::int64_t>(1, remaining_ms)));
    if (ready < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (ready == 0) {
      return false;
    }
    if ((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      return false;
    }
    const auto written =
        ::send(fd, line.data() + offset, line.size() - offset, 0);
    if (written < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      return false;
    }
    if (written == 0) {
      return false;
    }
    offset += static_cast<std::size_t>(written);
  }
  return true;
}

// Reads a single newline-delimited frame from fd into buffer.
// Returns std::nullopt on clean EOF or stop_fd signal.
// Throws std::runtime_error on timeout, oversized frame, or socket error.
std::optional<std::string>
read_line(int fd, std::string &buffer, std::chrono::milliseconds timeout,
          int stop_fd = -1) {
  const bool infinite = timeout.count() < 0;
  const auto deadline =
      infinite ? std::chrono::steady_clock::time_point::max()
               : (std::chrono::steady_clock::now() + timeout);

  for (;;) {
    const auto nl = buffer.find('\n');
    if (nl != std::string::npos) {
      if (nl > kMaxSidecarLineBytes) {
        throw std::runtime_error("sidecar response exceeds max line size");
      }
      std::string line = buffer.substr(0, nl);
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      buffer.erase(0, nl + 1);
      return line;
    }
    if (buffer.size() > kMaxSidecarLineBytes) {
      throw std::runtime_error("sidecar response exceeds max line size");
    }

    int poll_timeout_ms = -1;
    if (!infinite) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        throw std::runtime_error("timed out waiting for sidecar response");
      }
      poll_timeout_ms = static_cast<int>(
          std::max<std::int64_t>(
              1, std::chrono::duration_cast<std::chrono::milliseconds>(
                     deadline - now)
                     .count()));
    }

    std::array<pollfd, 2> pfds{{
        {.fd = fd, .events = POLLIN, .revents = 0},
        {.fd = stop_fd, .events = POLLIN, .revents = 0},
    }};
    const nfds_t nfds = stop_fd >= 0 ? 2 : 1;
    const int ready = ::poll(pfds.data(), nfds, poll_timeout_ms);
    if (ready < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(std::string("sidecar poll failed: ") +
                               std::strerror(errno));
    }
    if (ready == 0) {
      throw std::runtime_error("timed out waiting for sidecar response");
    }
    if (stop_fd >= 0 && (pfds[1].revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
      return std::nullopt;
    }
    if ((pfds[0].revents & POLLNVAL) != 0) {
      return std::nullopt;
    }

    std::array<char, 8192> chunk{};
    const auto n = ::recv(fd, chunk.data(), chunk.size(), 0);
    if (n == 0) {
      if (buffer.empty()) {
        return std::nullopt;
      }
      std::string tail = std::move(buffer);
      buffer.clear();
      if (!tail.empty() && tail.back() == '\r') {
        tail.pop_back();
      }
      return tail;
    }
    if (n < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      if (errno == ECONNRESET || errno == EPIPE || errno == ENOTCONN ||
          errno == EBADF) {
        return std::nullopt;
      }
      throw std::runtime_error(std::string("sidecar recv failed: ") +
                               std::strerror(errno));
    }
    buffer.append(chunk.data(), static_cast<std::size_t>(n));
  }
}

void wake_pipe(int write_fd) noexcept {
  if (write_fd >= 0) {
    const char byte = 1;
    const auto ignored = ::write(write_fd, &byte, 1);
    (void)ignored;
  }
}

std::uint64_t read_u64_field(const Json &obj, const char *key,
                             std::uint64_t fallback = 0) {
  if (!obj.contains(key) || obj.at(key).is_null()) {
    return fallback;
  }
  const auto &v = obj.at(key);
  if (v.is_number_unsigned()) {
    return v.get<std::uint64_t>();
  }
  if (v.is_number_integer() && v.get<std::int64_t>() >= 0) {
    return static_cast<std::uint64_t>(v.get<std::int64_t>());
  }
  return fallback;
}

} // namespace

struct UdsSidecarBackend::Impl {
  UdsSidecarOptions options;
  std::string provider_id;
  std::uint32_t protocol_version = 2;
  bool supports_subscribe_events = false;
  bool supports_thinking_budget = false;
  bool supports_sampling_penalties = false;
  std::filesystem::path socket_dir;
  std::filesystem::path socket_path;

  std::atomic<pid_t> child_pid{-1};
  std::atomic<bool> shutdown_requested{false};
  std::atomic<std::uint64_t> resident_bytes{0};
  std::atomic<std::uint64_t> next_rpc_id{1};

  std::mutex control_mutex;
  std::mutex shutdown_mutex;
  int control_fd = -1;
  std::string control_buffer;

  int event_fd = -1;
  int event_stop_pipe[2] = {-1, -1};
  std::thread event_thread;

  mutable std::mutex event_mutex;
  std::condition_variable event_cv;
  std::deque<TokenEvent> event_queue;
  std::unordered_set<std::uint64_t> active_request_ids;
  std::unordered_set<std::uint64_t> early_terminal_request_ids;
  bool event_stream_closed = false;
  std::string event_stream_error;

  explicit Impl(UdsSidecarOptions opts)
      : options(std::move(opts)), provider_id(options.provider_id) {
    if (options.binary_path.empty()) {
      throw std::invalid_argument(
          "UdsSidecarBackend requires a non-empty binary_path");
    }
    if (::access(options.binary_path.c_str(), X_OK) != 0) {
      throw std::runtime_error("sidecar binary is not executable: " +
                               options.binary_path.string());
    }
    std::signal(SIGPIPE, SIG_IGN);
  }

  ~Impl() { shutdown(); }

  int connect_socket_once() const {
    if (socket_path.native().size() >= sizeof(sockaddr_un::sun_path)) {
      throw std::runtime_error("sidecar socket path exceeds sun_path: " +
                               socket_path.string());
    }
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
      throw std::runtime_error(std::string("socket(AF_UNIX) failed: ") +
                               std::strerror(errno));
    }
    set_cloexec(fd);
    configure_nosigpipe(fd);

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::memcpy(addr.sun_path, socket_path.c_str(),
                socket_path.native().size() + 1);
    if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
      std::string peer_err;
      if (!verify_peer_uid(fd, peer_err)) {
        ::close(fd);
        throw std::runtime_error(peer_err);
      }
      return fd;
    }
    const int err = errno;
    ::close(fd);
    errno = err;
    return -1;
  }

  void check_child_alive_before_ready() {
    const pid_t pid = child_pid.load(std::memory_order_acquire);
    if (pid <= 0) {
      return;
    }
    int status = 0;
    const pid_t reaped = ::waitpid(pid, &status, WNOHANG);
    if (reaped == pid) {
      child_pid.store(-1, std::memory_order_release);
      if (WIFEXITED(status)) {
        throw std::runtime_error(
            "sidecar process exited before socket ready (code=" +
            std::to_string(WEXITSTATUS(status)) + "): " +
            options.binary_path.string());
      }
      if (WIFSIGNALED(status)) {
        throw std::runtime_error(
            "sidecar process terminated by signal " +
            std::to_string(WTERMSIG(status)) + " before socket ready: " +
            options.binary_path.string());
      }
      throw std::runtime_error(
          "sidecar process terminated before socket ready: " +
          options.binary_path.string());
    }
  }

  int connect_with_retry(std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
      check_child_alive_before_ready();
      const int fd = connect_socket_once();
      if (fd >= 0) {
        return fd;
      }
      const int err = errno;
      if (err != ENOENT && err != ECONNREFUSED && err != EAGAIN &&
          err != EINTR) {
        throw std::runtime_error("failed to connect to sidecar socket " +
                                 socket_path.string() + ": " +
                                 std::strerror(err));
      }
      check_child_alive_before_ready();
      if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error("timed out waiting for sidecar socket " +
                                 socket_path.string());
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  void spawn_child() {
    auto loc = make_ephemeral_socket_location();
    socket_dir = std::move(loc.dir);
    socket_path = std::move(loc.socket_path);

    std::string bin_str = options.binary_path.string();
    std::string flag_socket = "--socket";
    std::string sock_str = socket_path.string();
    std::string flag_exit = "--exit-on-disconnect";
    std::array<char *, 5> argv{{
        bin_str.data(),
        flag_socket.data(),
        sock_str.data(),
        flag_exit.data(),
        nullptr,
    }};

    posix_spawnattr_t attr;
    if (::posix_spawnattr_init(&attr) != 0) {
      throw std::runtime_error("posix_spawnattr_init failed");
    }
    struct AttrGuard {
      posix_spawnattr_t *a;
      ~AttrGuard() { (void)::posix_spawnattr_destroy(a); }
    } attr_guard{&attr};

    short flags = POSIX_SPAWN_SETPGROUP;
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
    flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#endif
    (void)::posix_spawnattr_setflags(&attr, flags);
    (void)::posix_spawnattr_setpgroup(&attr, 0);

    posix_spawn_file_actions_t actions;
    if (::posix_spawn_file_actions_init(&actions) != 0) {
      throw std::runtime_error("posix_spawn_file_actions_init failed");
    }
    struct ActionsGuard {
      posix_spawn_file_actions_t *a;
      ~ActionsGuard() { (void)::posix_spawn_file_actions_destroy(a); }
    } actions_guard{&actions};

    (void)::posix_spawn_file_actions_addopen(&actions, STDIN_FILENO,
                                             "/dev/null", O_RDONLY, 0);
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
    (void)::posix_spawn_file_actions_addinherit_np(&actions, STDOUT_FILENO);
    (void)::posix_spawn_file_actions_addinherit_np(&actions, STDERR_FILENO);
#endif

    pid_t spawned = -1;
    const int rc = ::posix_spawn(&spawned, bin_str.c_str(), &actions, &attr,
                                 argv.data(), current_environ());
    if (rc != 0) {
      throw std::runtime_error("failed to spawn sidecar '" + bin_str +
                               "': " + std::strerror(rc));
    }
    child_pid.store(spawned, std::memory_order_release);
  }

  void invalidate_control_connection_locked() noexcept {
    if (control_fd >= 0) {
      ::close(control_fd);
      control_fd = -1;
    }
    control_buffer.clear();
  }

  void ensure_event_stream_open() const {
    std::lock_guard lock(event_mutex);
    if (event_stream_closed ||
        shutdown_requested.load(std::memory_order_acquire)) {
      throw std::runtime_error(event_stream_error.empty()
                                   ? "sidecar event stream is closed"
                                   : event_stream_error);
    }
  }

  Json control_call_locked(std::string_view method, const Json &params,
                           std::chrono::milliseconds timeout) {
    if (control_fd < 0) {
      throw std::runtime_error("sidecar control connection is closed");
    }
    const std::uint64_t req_id =
        next_rpc_id.fetch_add(1, std::memory_order_relaxed);
    const Json payload = {
        {"id", req_id},
        {"method", std::string(method)},
        {"params", params.is_object() ? params : Json::object()},
    };
    if (!write_line(control_fd, payload.dump(), timeout)) {
      invalidate_control_connection_locked();
      throw std::runtime_error("failed to write sidecar RPC '" +
                               std::string(method) + "'");
    }
    std::optional<std::string> line;
    try {
      line = read_line(control_fd, control_buffer, timeout);
    } catch (...) {
      invalidate_control_connection_locked();
      throw;
    }
    if (!line.has_value()) {
      invalidate_control_connection_locked();
      throw std::runtime_error("unexpected EOF reading sidecar RPC '" +
                               std::string(method) + "'");
    }
    Json resp;
    try {
      resp = Json::parse(*line);
    } catch (...) {
      invalidate_control_connection_locked();
      throw;
    }
    if (!resp.is_object() || !resp.contains("id") ||
        !(resp.at("id").is_number_unsigned() ||
          (resp.at("id").is_number_integer() &&
           resp.at("id").get<std::int64_t>() >= 0))) {
      invalidate_control_connection_locked();
      throw std::runtime_error("sidecar RPC response for '" +
                               std::string(method) + "' missing valid id");
    }
    const std::uint64_t resp_id = resp.at("id").get<std::uint64_t>();
    if (resp_id != req_id) {
      invalidate_control_connection_locked();
      throw std::runtime_error("sidecar RPC response id mismatch for '" +
                               std::string(method) + "': expected " +
                               std::to_string(req_id) + ", got " +
                               std::to_string(resp_id));
    }
    if (!resp.contains("ok") || !resp.at("ok").is_boolean()) {
      invalidate_control_connection_locked();
      throw std::runtime_error("malformed sidecar RPC response for '" +
                               std::string(method) + "'");
    }
    if (!resp.at("ok").get<bool>()) {
      std::string err_msg = "sidecar RPC failed";
      if (resp.contains("error") && resp.at("error").is_object()) {
        err_msg = resp.at("error").value("message", err_msg);
      }
      throw std::runtime_error(err_msg);
    }
    return resp.value("result", Json::object());
  }

  Json control_call(std::string_view method, const Json &params = Json::object(),
                    std::optional<std::chrono::milliseconds> timeout_override =
                        std::nullopt) {
    std::lock_guard lock(control_mutex);
    return control_call_locked(method, params,
                               timeout_override.value_or(options.rpc_timeout));
  }

  Json build_load_params() const {
    const auto &lo = options.load_options;
    Json params = {
        {"model_dir", lo.model_dir},
        {"max_concurrency", lo.max_concurrency},
        {"max_context", lo.max_context},
        {"kv_dtype", lo.kv_dtype},
        {"prefill_engine", lo.prefill_engine},
        {"prefill_linear_mode",
         std::string(prefill_linear_mode_to_string(lo.prefill_linear_mode))},
        {"prefill_attention_mode",
         std::string(
             prefill_attention_mode_to_string(lo.prefill_attention_mode))},
        {"prefill_sparse_minference", lo.prefill_sparse_minference},
        {"prefill_cache_blend", lo.prefill_cache_blend},
        {"prefill_ane_mode",
         std::string(prefill_ane_mode_to_string(
             effective_load_prefill_ane_mode(lo)))},
        {"prefill_ane_offload",
         effective_load_prefill_ane_mode(lo) != PrefillAneMode::Off},
        {"spec_search", lo.spec_search},
        {"sampling_contract", lo.sampling_contract},
        {"memory_budget_bytes", lo.memory_budget_bytes},
        {"lora_adapter", lo.lora_adapter},
        {"lora_mode", lo.lora_mode.empty() ? std::string("merge") : lo.lora_mode},
        {"speculative",
         {
             {"backend",
              std::string(speculative_backend_to_string(lo.speculative.backend))},
             {"draft_tokens", lo.speculative.draft_tokens},
         }},
    };
    if (!lo.draft_dir.empty()) {
      params["draft_dir"] = lo.draft_dir;
    }

    Json overrides = Json::object();
    for (const auto &[k, v] : lo.path_overrides) {
      overrides[k] = v;
    }
    if (!overrides.empty()) {
      params["path_overrides"] = overrides;
    }

    // Also mirror key fields at top-level for third-party SPI v2 providers.
    if (const auto it = lo.path_overrides.find("artifact");
        it != lo.path_overrides.end() && !it->second.empty()) {
      params["artifact"] = it->second;
      params["artifact_path"] = it->second;
    }
    if (const auto it = lo.path_overrides.find("draft_artifact");
        it != lo.path_overrides.end() && !it->second.empty()) {
      params["draft_artifact"] = it->second;
      params["draft_artifact_path"] = it->second;
    }
    if (const auto it = lo.path_overrides.find("mtp_artifact");
        it != lo.path_overrides.end() && !it->second.empty()) {
      params["mtp_artifact"] = it->second;
    }
    if (const auto it = lo.path_overrides.find("kernel_directory");
        it != lo.path_overrides.end() && !it->second.empty()) {
      params["kernel_directory"] = it->second;
    }
    if (options.rolling_prefix_cache.has_value()) {
      params["rolling_prefix_cache"] = *options.rolling_prefix_cache;
    } else if (const auto it = lo.path_overrides.find("rolling_prefix_cache");
               it != lo.path_overrides.end()) {
      params["rolling_prefix_cache"] =
          (it->second == "on" || it->second == "true" || it->second == "1");
    }
    if (options.prefix_cache_gb.has_value()) {
      params["prefix_cache_gb"] = *options.prefix_cache_gb;
    }
    return params;
  }

  void start() {
    if (::pipe(event_stop_pipe) != 0) {
      throw std::runtime_error("failed to create sidecar stop pipe");
    }
    set_cloexec(event_stop_pipe[0]);
    set_cloexec(event_stop_pipe[1]);

    spawn_child();
    control_fd = connect_with_retry(options.startup_timeout);

    // 1) SPI v2 hello handshake
    const Json hello =
        control_call("hello", Json::object(), options.startup_timeout);
    if (!hello.contains("protocol_version") ||
        !hello.at("protocol_version").is_number_integer()) {
      throw std::runtime_error(
          "sidecar hello missing integer protocol_version");
    }
    const auto proto = hello.at("protocol_version").get<std::int64_t>();
    if (proto != 1 && proto != 2) {
      throw std::runtime_error("unsupported sidecar protocol_version: " +
                               std::to_string(proto));
    }
    protocol_version = static_cast<std::uint32_t>(proto);
    if (provider_id.empty()) {
      provider_id = hello.value("provider_id", std::string("unknown.sidecar"));
    }
    if (const auto phys = read_u64_field(hello, "phys_footprint_bytes");
        phys > 0) {
      resident_bytes.store(phys, std::memory_order_relaxed);
    }

    supports_subscribe_events = false;
    if (protocol_version >= 2) {
      if (!hello.contains("capabilities") ||
          !hello.at("capabilities").is_array()) {
        supports_subscribe_events = true;
      } else {
        for (const auto &cap : hello.at("capabilities")) {
          if (!cap.is_string()) {
            continue;
          }
          const auto &name = cap.get_ref<const std::string &>();
          supports_subscribe_events |= name == "subscribe_events";
          supports_thinking_budget |= name == "thinking_budget";
          supports_sampling_penalties |= name == "sampling_penalties";
        }
      }
    }

    // 2) Load model into the sidecar process
    const Json load_res =
        control_call("load", build_load_params(), options.rpc_timeout);
    if (const auto phys = read_u64_field(load_res, "phys_footprint_bytes");
        phys > 0) {
      resident_bytes.store(phys, std::memory_order_relaxed);
    }

    // 3) Connect dedicated event channel (subscribe_events on v2, poll on v1)
    event_fd = connect_with_retry(options.startup_timeout);
    std::string initial_event_buffer;
    if (supports_subscribe_events) {
      const Json sub_req = {
          {"id", 1},
          {"method", "subscribe_events"},
          {"params", {{"poll_interval_us", options.event_poll_interval_us}}},
      };
      if (!write_line(event_fd, sub_req.dump(), options.startup_timeout)) {
        throw std::runtime_error(
            "failed to send subscribe_events to sidecar");
      }
      const auto ack_line = read_line(event_fd, initial_event_buffer,
                                      options.startup_timeout);
      if (!ack_line.has_value()) {
        throw std::runtime_error("unexpected EOF waiting for subscribe_events");
      }
      const Json ack = Json::parse(*ack_line);
      if (!ack.value("ok", false)) {
        throw std::runtime_error("sidecar subscribe_events rejected: " +
                                 ack.dump());
      }
    }

    event_thread = std::thread(
        [this, buf = std::move(initial_event_buffer)]() mutable {
          event_reader_loop(std::move(buf));
        });
  }

  void record_event_locked(TokenEvent ev) {
    if (ev.kind != EventKind::tokens) {
      if (active_request_ids.erase(ev.request_id) == 0) {
        early_terminal_request_ids.insert(ev.request_id);
      }
    }
    event_queue.push_back(std::move(ev));
  }

  void event_reader_loop(std::string buffer) {
    std::string close_reason = "sidecar process disconnected unexpectedly";
    try {
      if (supports_subscribe_events) {
        while (!shutdown_requested.load(std::memory_order_acquire)) {
          const auto line = read_line(event_fd, buffer,
                                      std::chrono::milliseconds(-1),
                                      event_stop_pipe[0]);
          if (!line.has_value()) {
            break;
          }
          if (line->empty()) {
            continue;
          }
          const Json frame = Json::parse(*line);
          if (!frame.contains("event") || !frame.at("event").is_object()) {
            continue;
          }
          const auto &ev = frame.at("event");
          if (ev.value("kind", "") == "eof") {
            break;
          }
          TokenEvent parsed = parse_token_event(ev);
          {
            std::lock_guard lock(event_mutex);
            record_event_locked(std::move(parsed));
          }
          event_cv.notify_all();
        }
      } else {
        // Protocol v1 polling fallback on dedicated event socket.
        std::uint64_t poll_id = 1;
        while (!shutdown_requested.load(std::memory_order_acquire)) {
          const Json req = {
              {"id", poll_id++},
              {"method", "poll"},
              {"params", {{"max_wait_us", 20'000}}},
          };
          if (!write_line(event_fd, req.dump(), std::chrono::milliseconds(5000))) {
            break;
          }
          const auto line = read_line(event_fd, buffer,
                                      std::chrono::milliseconds(5000),
                                      event_stop_pipe[0]);
          if (!line.has_value()) {
            break;
          }
          const Json resp = Json::parse(*line);
          if (!resp.value("ok", false)) {
            break;
          }
          const auto &res = resp.value("result", Json::object());
          if (res.contains("events") && res.at("events").is_array() &&
              !res.at("events").empty()) {
            {
              std::lock_guard lock(event_mutex);
              for (const auto &ev : res.at("events")) {
                record_event_locked(parse_token_event(ev));
              }
            }
            event_cv.notify_all();
          }
        }
      }
    } catch (const std::exception &ex) {
      close_reason = std::string("sidecar event stream error: ") + ex.what();
    } catch (...) {
      close_reason = "sidecar event stream terminated unexpectedly";
    }
    {
      std::lock_guard lock(event_mutex);
      const bool shutting_down =
          shutdown_requested.load(std::memory_order_acquire);
      event_stream_closed = true;
      if (!shutting_down) {
        event_stream_error = close_reason;
      }
      for (const std::uint64_t req_id : active_request_ids) {
        event_queue.push_back(TokenEvent{
            .request_id = req_id,
            .kind = shutting_down ? EventKind::aborted : EventKind::failed,
            .finish_reason = shutting_down ? "cancelled" : "error",
            .error = shutting_down ? "sidecar shutting down" : close_reason,
        });
      }
      active_request_ids.clear();
    }
    event_cv.notify_all();
  }

  void reap_child() noexcept {
    const pid_t pid = child_pid.load(std::memory_order_acquire);
    if (pid <= 0) {
      return;
    }

    auto wait_until = [pid](std::chrono::milliseconds max_wait) -> bool {
      const auto deadline = std::chrono::steady_clock::now() + max_wait;
      for (;;) {
        int status = 0;
        const pid_t r = ::waitpid(pid, &status, WNOHANG);
        if (r == pid || (r < 0 && errno == ECHILD)) {
          return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    };

    // Give graceful shutdown RPC / EOF a chance to exit cleanly first.
    if (!wait_until(std::chrono::milliseconds(1500))) {
      (void)::kill(-pid, SIGTERM);
      (void)::kill(pid, SIGTERM);
      if (!wait_until(std::chrono::milliseconds(2000))) {
        (void)::kill(-pid, SIGKILL);
        (void)::kill(pid, SIGKILL);
        int status = 0;
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
        }
      }
    }
    child_pid.store(-1, std::memory_order_release);
  }

  void shutdown() noexcept {
    std::lock_guard shutdown_lock(shutdown_mutex);
    shutdown_requested.store(true, std::memory_order_release);

    // Best-effort graceful shutdown RPC before closing sockets.
    {
      std::lock_guard lock(control_mutex);
      if (control_fd >= 0) {
        try {
          (void)control_call_locked("shutdown", Json::object(),
                                    std::chrono::milliseconds(1500));
        } catch (...) {
        }
        invalidate_control_connection_locked();
      }
    }

    wake_pipe(event_stop_pipe[1]);
    if (event_fd >= 0) {
      (void)::shutdown(event_fd, SHUT_RDWR);
    }
    if (event_thread.joinable()) {
      event_thread.join();
    }
    if (event_fd >= 0) {
      ::close(event_fd);
      event_fd = -1;
    }
    if (event_stop_pipe[0] >= 0) {
      ::close(event_stop_pipe[0]);
      event_stop_pipe[0] = -1;
    }
    if (event_stop_pipe[1] >= 0) {
      ::close(event_stop_pipe[1]);
      event_stop_pipe[1] = -1;
    }

    {
      std::lock_guard lock(event_mutex);
      event_stream_closed = true;
    }
    event_cv.notify_all();

    reap_child();

    if (!socket_path.empty()) {
      std::error_code ec;
      std::filesystem::remove(socket_path, ec);
    }
    if (!socket_dir.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(socket_dir, ec);
      socket_dir.clear();
    }
  }
};

UdsSidecarBackend::UdsSidecarBackend(UdsSidecarOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {
  try {
    impl_->start();
  } catch (...) {
    impl_->shutdown();
    throw;
  }
}

UdsSidecarBackend::~UdsSidecarBackend() { shutdown(); }

std::uint64_t UdsSidecarBackend::submit(SubmitRequest request) {
  impl_->ensure_event_stream_open();
  const bool coupled =
      impl_->options.load_options.sampling_contract == "position_coupled_v1";
  if (request.thinking_budget && (!impl_->supports_thinking_budget || !coupled)) {
    throw std::invalid_argument(
        "thinking_budget is not supported by this sidecar and sampling contract");
  }
  if (request.sampling.has_penalties() &&
      (!impl_->supports_sampling_penalties || !coupled)) {
    throw std::invalid_argument(
        "sampling penalties are not supported by this sidecar and sampling contract");
  }
  Json params = {
      {"prompt_tokens", request.prompt_tokens},
      {"max_new_tokens", request.max_new_tokens},
      {"sampling",
       {
           {"temperature", request.sampling.temperature},
           {"top_p", request.sampling.top_p},
           {"min_p", request.sampling.min_p},
           {"top_k", request.sampling.top_k},
           {"seed", request.sampling.seed},
       }},
      {"stop_token_seqs", request.stop_token_seqs},
      {"pin_prefix", request.pin_prefix},
  };
  if (request.sampling.has_penalties()) {
    auto &sampling = params.at("sampling");
    sampling["repetition_penalty"] = request.sampling.repetition_penalty;
    sampling["repetition_context_size"] = request.sampling.repetition_context_size;
    sampling["presence_penalty"] = request.sampling.presence_penalty;
    sampling["frequency_penalty"] = request.sampling.frequency_penalty;
  }
  if (const auto &budget = request.thinking_budget) {
    params["thinking_budget"] = {
        {"budget", budget->budget},
        {"start_in_thinking", budget->start_in_thinking},
        {"start_token_ids", budget->start_token_ids},
        {"end_token_ids", budget->end_token_ids},
        {"force_token_ids", budget->force_token_ids},
    };
  }
  const Json result = impl_->control_call("submit", params);
  const auto req_id = result.at("request_id").get<std::uint64_t>();
  {
    std::lock_guard lock(impl_->event_mutex);
    if (impl_->early_terminal_request_ids.erase(req_id) == 0) {
      if (impl_->event_stream_closed) {
        const bool shutting_down =
            impl_->shutdown_requested.load(std::memory_order_acquire);
        impl_->event_queue.push_back(TokenEvent{
            .request_id = req_id,
            .kind = shutting_down ? EventKind::aborted : EventKind::failed,
            .finish_reason = shutting_down ? "cancelled" : "error",
            .error = shutting_down
                         ? "sidecar shutting down"
                         : (impl_->event_stream_error.empty()
                                ? "sidecar process disconnected unexpectedly"
                                : impl_->event_stream_error),
        });
        impl_->event_cv.notify_all();
      } else {
        impl_->active_request_ids.insert(req_id);
      }
    }
  }
  return req_id;
}

bool UdsSidecarBackend::cancel(std::uint64_t request_id) {
  if (impl_->shutdown_requested.load(std::memory_order_acquire)) {
    return false;
  }
  try {
    const Json result =
        impl_->control_call("cancel", {{"request_id", request_id}});
    return result.value("cancelled", false);
  } catch (...) {
    return false;
  }
}

std::vector<TokenEvent>
UdsSidecarBackend::poll_events(std::chrono::microseconds max_wait) {
  std::unique_lock lock(impl_->event_mutex);
  if (impl_->event_queue.empty() && !impl_->event_stream_closed &&
      !impl_->shutdown_requested.load(std::memory_order_acquire)) {
    impl_->event_cv.wait_for(lock, max_wait, [&] {
      return !impl_->event_queue.empty() || impl_->event_stream_closed ||
             impl_->shutdown_requested.load(std::memory_order_acquire);
    });
  }
  std::vector<TokenEvent> drained;
  drained.reserve(impl_->event_queue.size());
  while (!impl_->event_queue.empty()) {
    drained.push_back(std::move(impl_->event_queue.front()));
    impl_->event_queue.pop_front();
  }
  if (!drained.empty()) {
    return drained;
  }
  if (impl_->event_stream_closed &&
      !impl_->shutdown_requested.load(std::memory_order_acquire)) {
    throw std::runtime_error(
        impl_->event_stream_error.empty()
            ? "sidecar event stream closed unexpectedly"
            : impl_->event_stream_error);
  }
  return drained;
}

CapacityReport UdsSidecarBackend::capacity() const {
  impl_->ensure_event_stream_open();
  const Json res = impl_->control_call("capacity");
  if (const auto phys = read_u64_field(res, "phys_footprint_bytes"); phys > 0) {
    impl_->resident_bytes.store(phys, std::memory_order_relaxed);
  }
  CapacityReport cap;
  cap.max_concurrency = res.value("max_concurrency", 1);
  cap.active_requests = res.value("active_requests", 0);
  cap.queued_requests = res.value("queued_requests", 0);
  cap.accepting = res.value("accepting", true);
  return cap;
}

StatsReport UdsSidecarBackend::stats() const {
  impl_->ensure_event_stream_open();
  const Json res = impl_->control_call("stats");
  if (const auto phys = read_u64_field(res, "phys_footprint_bytes"); phys > 0) {
    impl_->resident_bytes.store(phys, std::memory_order_relaxed);
  }
  StatsReport st;
  st.submitted = read_u64_field(res, "submitted");
  st.completed = read_u64_field(res, "completed");
  st.aborted = read_u64_field(res, "aborted");
  st.failed = read_u64_field(res, "failed");
  st.emitted_tokens = read_u64_field(res, "emitted_tokens");
  st.backend_kind = res.value("backend_kind", std::string("sidecar"));
  st.model_execution_kind =
      res.value("model_execution_kind", std::string("sidecar"));
  st.prefix_hits = read_u64_field(res, "prefix_hits");
  st.prefix_partial_hits = read_u64_field(res, "prefix_partial_hits");
  st.prefix_misses = read_u64_field(res, "prefix_misses");
  st.prefix_evictions = read_u64_field(res, "prefix_evictions");
  st.prefix_restored_tokens = read_u64_field(res, "prefix_restored_tokens");
  st.spec_rounds = read_u64_field(res, "spec_rounds");
  st.spec_proposal_tokens = read_u64_field(res, "spec_proposal_tokens");
  st.spec_accepted_tokens = read_u64_field(res, "spec_accepted_tokens");
  st.contended_prefill_chunks = read_u64_field(res, "contended_prefill_chunks");
  st.artifact_id = res.value("artifact_id", std::string{});
  st.kernel_family = res.value("kernel_family", std::string{});
  st.lora_adapter = res.value("lora_adapter", std::string{});
  st.lora_mode = res.value("lora_mode", std::string{});
  st.lora_buffer_bytes = read_u64_field(res, "lora_buffer_bytes");
  st.lora_adapted_layers =
      static_cast<std::uint32_t>(read_u64_field(res, "lora_adapted_layers"));
  st.prefill_linear_mode =
      res.value("prefill_linear_mode", std::string("bf16"));
  st.prefill_linear_mode_effective =
      res.value("prefill_linear_mode_effective", st.prefill_linear_mode);
  st.prefill_linear_fallback = res.value("prefill_linear_fallback", false);
  st.prefill_linear_fallback_reason =
      res.value("prefill_linear_fallback_reason", std::string{});
  st.model_quant_format = res.value("model_quant_format", std::string{});
  st.prefill_attention_mode =
      res.value("prefill_attention_mode", std::string("exact_nax"));
  st.prefill_sparse_minference =
      res.value("prefill_sparse_minference", false);
  st.prefill_sparse_minference_effective =
      res.value("prefill_sparse_minference_effective", false);
  st.prefill_sparse_minference_inactive_reason =
      res.value("prefill_sparse_minference_inactive_reason", std::string{});
  st.prefill_cache_blend = res.value("prefill_cache_blend", false);
  st.prefill_ane_offload = res.value("prefill_ane_offload", false);
  st.prefill_ane_mode = res.value(
      "prefill_ane_mode",
      st.prefill_ane_offload ? std::string("auto") : std::string("off"));
  st.prefill_ane_mode_effective =
      res.value("prefill_ane_mode_effective", std::string("off"));
  st.prefill_ane_inactive_reason =
      res.value("prefill_ane_inactive_reason", std::string{});
  st.prefill_ane_active = res.value("prefill_ane_active", false);
  st.prefill_ane_split_alpha = res.value("prefill_ane_split_alpha", 0.0);
  st.prefill_w4a8_tokens = read_u64_field(res, "prefill_w4a8_tokens");
  st.prefill_w4a4_tokens = read_u64_field(res, "prefill_w4a4_tokens");
  st.prefill_sage2_tokens = read_u64_field(res, "prefill_sage2_tokens");
  st.prefill_minference_blocks_skipped =
      read_u64_field(res, "prefill_minference_blocks_skipped");
  st.prefill_minference_blocks_total =
      read_u64_field(res, "prefill_minference_blocks_total");
  st.prefill_cache_blend_hit_tokens =
      read_u64_field(res, "prefill_cache_blend_hit_tokens");
  st.prefill_cache_blend_recomputed_tokens =
      read_u64_field(res, "prefill_cache_blend_recomputed_tokens");
  st.prefill_ane_offload_tokens =
      read_u64_field(res, "prefill_ane_offload_tokens");
  st.prefill_ane_fallback_tokens =
      read_u64_field(res, "prefill_ane_fallback_tokens");
  return st;
}

TrimCacheReport
UdsSidecarBackend::trim_cache(std::uint64_t target_free_bytes) {
  const Json res = impl_->control_call(
      "trim_cache", {{"target_free_bytes", target_free_bytes}});
  TrimCacheReport report;
  report.freed_bytes = read_u64_field(res, "freed_bytes");
  report.evicted_entries = read_u64_field(res, "evicted_entries");
  report.resident_bytes = read_u64_field(
      res, "resident_bytes", read_u64_field(res, "phys_footprint_bytes"));
  if (report.resident_bytes > 0) {
    impl_->resident_bytes.store(report.resident_bytes,
                                std::memory_order_relaxed);
  }
  return report;
}

void UdsSidecarBackend::set_scheduler_hint(const SchedulerHint &hint) {
  if (impl_->shutdown_requested.load(std::memory_order_acquire)) {
    return;
  }
  (void)impl_->control_call("set_scheduler_hint",
                            {{"gpu_contended", hint.gpu_contended}});
}

bool UdsSidecarBackend::reload_lora(const std::string &lora_adapter,
                                    const std::string &lora_mode,
                                    std::string *error) {
  if (impl_->shutdown_requested.load(std::memory_order_acquire)) {
    if (error != nullptr) {
      *error = "sidecar is shutting down";
    }
    return false;
  }
  try {
    const std::string mode =
        lora_mode.empty() ? std::string("bypass") : lora_mode;
    const Json res = impl_->control_call(
        "reload_lora", {{"lora_adapter", lora_adapter}, {"lora_mode", mode}});
    if (const auto phys = read_u64_field(res, "phys_footprint_bytes");
        phys > 0) {
      impl_->resident_bytes.store(phys, std::memory_order_relaxed);
    }
    impl_->options.load_options.lora_adapter = lora_adapter;
    impl_->options.load_options.lora_mode = mode;
    return true;
  } catch (const std::exception &ex) {
    if (error != nullptr) {
      *error = ex.what();
    }
    return false;
  }
}

std::uint64_t UdsSidecarBackend::resident_memory_bytes() const {
  return impl_->resident_bytes.load(std::memory_order_relaxed);
}

void UdsSidecarBackend::shutdown() {
  if (impl_ != nullptr) {
    impl_->shutdown();
  }
}

std::string UdsSidecarBackend::provider_id() const {
  return impl_->provider_id;
}

pid_t UdsSidecarBackend::child_pid() const noexcept {
  return impl_->child_pid.load(std::memory_order_acquire);
}

const std::filesystem::path &UdsSidecarBackend::socket_path() const noexcept {
  return impl_->socket_path;
}

std::uint32_t UdsSidecarBackend::protocol_version() const noexcept {
  return impl_->protocol_version;
}

} // namespace omlx::m5::serve
