// Test-only SPI v2 UDS microkernel sidecar daemon (omlx_m5_mock_spi_provider).
// NOT part of REQUIRED_BINARIES or release packages.
// Speaks exact SPI v2 NDJSON wire protocol for multi-microkernel E2E tests.

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

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
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

namespace {

using Json = nlohmann::json;

std::atomic<int> g_stop_write_fd{-1};

extern "C" void handle_signal(int /*signum*/) {
  const int fd = g_stop_write_fd.load(std::memory_order_relaxed);
  if (fd >= 0) {
    const char b = 1;
    const auto r = ::write(fd, &b, 1);
    (void)r;
  }
}

void trigger_pipe(int fd) noexcept {
  if (fd >= 0) {
    const char b = 1;
    const auto r = ::write(fd, &b, 1);
    (void)r;
  }
}

bool pipe_signaled(int fd) noexcept {
  if (fd < 0) {
    return false;
  }
  pollfd pfd{.fd = fd, .events = POLLIN, .revents = 0};
  return ::poll(&pfd, 1, 0) > 0 &&
         (pfd.revents & (POLLIN | POLLHUP | POLLERR)) != 0;
}

void set_cloexec(int fd) noexcept {
  const int f = ::fcntl(fd, F_GETFD, 0);
  if (f >= 0) {
    (void)::fcntl(fd, F_SETFD, f | FD_CLOEXEC);
  }
}

void set_nonblocking(int fd, bool on) {
  const int f = ::fcntl(fd, F_GETFL, 0);
  if (f >= 0) {
    (void)::fcntl(fd, F_SETFL, on ? (f | O_NONBLOCK) : (f & ~O_NONBLOCK));
  }
}

void configure_nosigpipe(int fd) noexcept {
#ifdef SO_NOSIGPIPE
  const int one = 1;
  (void)::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#else
  (void)fd;
#endif
}

std::uint64_t current_phys_footprint() noexcept {
#ifdef __APPLE__
  task_vm_info_data_t vm{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&vm), &count) == KERN_SUCCESS &&
      vm.phys_footprint > 0) {
    return static_cast<std::uint64_t>(vm.phys_footprint);
  }
#endif
  return 16ULL * 1024ULL * 1024ULL;
}

bool write_line(int fd, std::string line) {
  line.push_back('\n');
  std::size_t off = 0;
  while (off < line.size()) {
    const auto n = ::send(fd, line.data() + off, line.size() - off, 0);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        pollfd pfd{.fd = fd, .events = POLLOUT, .revents = 0};
        (void)::poll(&pfd, 1, 50);
        continue;
      }
      return false;
    }
    if (n == 0) {
      return false;
    }
    off += static_cast<std::size_t>(n);
  }
  return true;
}

std::optional<std::string> read_line(int fd, int stop_fd, std::string &buf) {
  for (;;) {
    const auto nl = buf.find('\n');
    if (nl != std::string::npos) {
      std::string out = buf.substr(0, nl);
      if (!out.empty() && out.back() == '\r') {
        out.pop_back();
      }
      buf.erase(0, nl + 1);
      return out;
    }
    std::array<pollfd, 2> pfds{{
        {.fd = fd, .events = POLLIN, .revents = 0},
        {.fd = stop_fd, .events = POLLIN, .revents = 0},
    }};
    const int r = ::poll(pfds.data(), stop_fd >= 0 ? 2 : 1, -1);
    if (r < 0) {
      if (errno == EINTR) {
        continue;
      }
      return std::nullopt;
    }
    if (stop_fd >= 0 && (pfds[1].revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
      return std::nullopt;
    }
    std::array<char, 4096> chunk{};
    const auto n = ::recv(fd, chunk.data(), chunk.size(), 0);
    if (n == 0) {
      if (buf.empty()) {
        return std::nullopt;
      }
      std::string tail = std::move(buf);
      buf.clear();
      return tail;
    }
    if (n < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      return std::nullopt;
    }
    buf.append(chunk.data(), static_cast<std::size_t>(n));
  }
}

bool parse_bool_or_toggle(const Json &value, std::string_view key) {
  if (value.is_boolean()) {
    return value.get<bool>();
  }
  if (value.is_string()) {
    const std::string s = value.get<std::string>();
    if (s == "on" || s == "ON" || s == "true" || s == "TRUE" || s == "1") {
      return true;
    }
    if (s == "off" || s == "OFF" || s == "false" || s == "FALSE" || s == "0") {
      return false;
    }
    throw std::invalid_argument(std::string(key) +
                                " expects boolean or on/off string");
  }
  if (value.is_number_integer()) {
    const auto n = value.get<std::int64_t>();
    if (n == 0 || n == 1) {
      return n == 1;
    }
  }
  throw std::invalid_argument(std::string(key) +
                              " has an unsupported boolean/toggle value type");
}

struct MockJob {
  std::uint64_t request_id = 0;
  std::vector<std::int32_t> prompt_tokens;
  std::int32_t max_new_tokens = 1;
  std::vector<std::vector<std::int32_t>> stop_token_seqs;
  std::size_t max_stop_len = 0;
  std::vector<std::int32_t> pending_tokens;
  std::int32_t step_index = 0;
  std::atomic<bool> cancelled{false};
};

class MockSession final {
public:
  MockSession() {
    if (const char *env = std::getenv("MOCK_SPI_STEP_DELAY_US"); env && *env) {
      const long parsed = std::strtol(env, nullptr, 10);
      if (parsed >= 0) {
        step_delay_us_ = static_cast<std::int32_t>(parsed);
      }
    }
    if (const char *env = std::getenv("MOCK_SPI_PROVIDER_ID"); env && *env) {
      provider_id_ = env;
    }
  }

  ~MockSession() { shutdown(); }

  Json execute(std::string_view method, const Json &params) {
    if (method == "hello") {
      return {
          {"provider_id", provider_id_},
          {"provider_version", "mock-v0.01"},
          {"core_abi", 1},
          {"protocol_version", 2},
          {"engine_contract_version", 1},
          {"pid", static_cast<std::int64_t>(::getpid())},
          {"phys_footprint_bytes", current_phys_footprint()},
          {"capabilities",
           {"ar", "trim_cache", "memory_budget_bytes", "set_scheduler_hint",
            "cancel", "stats", "capacity", "subscribe_events"}},
          {"kv_formats", {"bf16"}},
      };
    }
    if (method == "load") {
      std::lock_guard lock(mutex_);
      if (loaded_) {
        throw std::logic_error("already loaded");
      }
      if (stopping_) {
        throw std::runtime_error("mock engine is shutting down");
      }
      const Json *overrides = nullptr;
      if (params.contains("path_overrides") &&
          !params.at("path_overrides").is_null()) {
        if (!params.at("path_overrides").is_object()) {
          throw std::invalid_argument("path_overrides must be an object");
        }
        overrides = &params.at("path_overrides");
      }

      model_dir_ = params.value("model_dir", "");
      std::string artifact;
      if (params.contains("artifact") && params.at("artifact").is_string()) {
        artifact = params.at("artifact").get<std::string>();
      } else if (params.contains("artifact_path") &&
                 params.at("artifact_path").is_string()) {
        artifact = params.at("artifact_path").get<std::string>();
      }
      if (artifact.empty() && overrides != nullptr) {
        if (overrides->contains("artifact") &&
            overrides->at("artifact").is_string()) {
          artifact = overrides->at("artifact").get<std::string>();
        } else if (overrides->contains("artifact_path") &&
                   overrides->at("artifact_path").is_string()) {
          artifact = overrides->at("artifact_path").get<std::string>();
        }
      }
      artifact_ = std::move(artifact);

      max_concurrency_ = std::max(1, params.value("max_concurrency", 1));
      max_context_ = params.value("max_context", 4096);
      kv_dtype_ = params.value("kv_dtype", "bf16");
      memory_budget_bytes_ = params.value("memory_budget_bytes", 0ULL);

      rolling_prefix_cache_ = false;
      if (params.contains("rolling_prefix_cache") &&
          !params.at("rolling_prefix_cache").is_null()) {
        rolling_prefix_cache_ = parse_bool_or_toggle(
            params.at("rolling_prefix_cache"), "rolling_prefix_cache");
      } else if (overrides != nullptr &&
                 overrides->contains("rolling_prefix_cache") &&
                 !overrides->at("rolling_prefix_cache").is_null()) {
        rolling_prefix_cache_ =
            parse_bool_or_toggle(overrides->at("rolling_prefix_cache"),
                                 "path_overrides.rolling_prefix_cache");
      }

      prefix_cache_gb_ = params.value("prefix_cache_gb", 0.0);
      loaded_ = true;
      worker_ = std::thread([this] { worker_loop(); });
      write_audit_locked();
      return stats_locked();
    }
    if (method == "shutdown") {
      shutdown();
      return {{"shutdown", true}};
    }
    if (method == "submit") {
      auto job = std::make_shared<MockJob>();
      job->prompt_tokens =
          params.at("prompt_tokens").get<std::vector<std::int32_t>>();
      job->max_new_tokens = params.value("max_new_tokens", 1);
      job->stop_token_seqs = params.value(
          "stop_token_seqs", std::vector<std::vector<std::int32_t>>{});
      for (const auto &seq : job->stop_token_seqs) {
        job->max_stop_len = std::max(job->max_stop_len, seq.size());
      }
      std::uint64_t req_id = 0;
      {
        std::lock_guard lock(mutex_);
        if (!loaded_ || stopping_) {
          throw std::runtime_error("mock engine not accepting requests");
        }
        req_id = next_request_id_++;
        job->request_id = req_id;
        jobs_by_id_.emplace(req_id, job);
        job_queue_.push_back(job);
        ++submitted_;
      }
      job_cv_.notify_all();
      return {{"request_id", req_id}};
    }
    if (method == "cancel") {
      const auto req_id = params.at("request_id").get<std::uint64_t>();
      bool cancelled = false;
      {
        std::lock_guard lock(mutex_);
        const auto it = jobs_by_id_.find(req_id);
        if (it != jobs_by_id_.end()) {
          it->second->cancelled.store(true, std::memory_order_release);
          cancelled = true;
        }
      }
      if (cancelled) {
        job_cv_.notify_all();
      }
      return {{"cancelled", cancelled}};
    }
    if (method == "poll") {
      const auto wait_us = params.value("max_wait_us", 20'000LL);
      const auto evs =
          poll_events(std::chrono::microseconds(std::max<std::int64_t>(0, wait_us)));
      Json arr = Json::array();
      for (const auto &e : evs) {
        arr.push_back(e);
      }
      return {{"events", std::move(arr)}};
    }
    if (method == "capacity") {
      std::lock_guard lock(mutex_);
      return {
          {"max_concurrency", max_concurrency_},
          {"active_requests", active_requests_},
          {"queued_requests", static_cast<std::int32_t>(job_queue_.size())},
          {"accepting", loaded_ && !stopping_},
          {"pid", static_cast<std::int64_t>(::getpid())},
          {"phys_footprint_bytes", current_phys_footprint()},
      };
    }
    if (method == "stats") {
      std::lock_guard lock(mutex_);
      return stats_locked();
    }
    if (method == "trim_cache") {
      const auto target = params.value("target_free_bytes", 0ULL);
      std::lock_guard lock(mutex_);
      ++trim_cache_calls_;
      last_trim_target_bytes_ = target;
      const std::uint64_t freed = target > 0 ? std::min<std::uint64_t>(target, 4ULL * 1024 * 1024) : 1024ULL * 1024ULL;
      ++prefix_evictions_;
      write_audit_locked();
      const auto phys = current_phys_footprint();
      return {
          {"freed_bytes", freed},
          {"evicted_entries", 1},
          {"resident_bytes", phys},
          {"phys_footprint_bytes", phys},
      };
    }
    if (method == "set_scheduler_hint") {
      const bool contended = params.value("gpu_contended", false);
      std::lock_guard lock(mutex_);
      gpu_contended_ = contended;
      scheduler_hint_history_.push_back(contended);
      if (contended) {
        ++contended_hint_trues_;
      }
      write_audit_locked();
      return {{"gpu_contended", contended}};
    }
    throw std::invalid_argument("unknown method: " + std::string(method));
  }

  std::vector<Json> poll_events(std::chrono::microseconds max_wait) {
    std::unique_lock lock(event_mutex_);
    if (events_.empty() && !events_eof_) {
      event_cv_.wait_for(lock, max_wait, [&] {
        return !events_.empty() || events_eof_;
      });
    }
    std::vector<Json> out;
    out.reserve(events_.size());
    while (!events_.empty()) {
      out.push_back(std::move(events_.front()));
      events_.pop_front();
    }
    return out;
  }

  bool is_stopping() const noexcept {
    return stopping_atomic_.load(std::memory_order_acquire);
  }

  void shutdown() {
    std::lock_guard shutdown_lock(shutdown_mutex_);
    {
      std::lock_guard lock(mutex_);
      if (!stopping_) {
        stopping_ = true;
        for (auto &[id, job] : jobs_by_id_) {
          job->cancelled.store(true, std::memory_order_release);
        }
        shutdown_received_ = true;
        write_audit_locked();
      }
    }
    job_cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    {
      std::lock_guard lock(event_mutex_);
      events_eof_ = true;
    }
    stopping_atomic_.store(true, std::memory_order_release);
    event_cv_.notify_all();
  }

private:
  Json stats_locked() const {
    return {
        {"pid", static_cast<std::int64_t>(::getpid())},
        {"phys_footprint_bytes", current_phys_footprint()},
        {"submitted", submitted_},
        {"completed", completed_},
        {"aborted", aborted_},
        {"failed", failed_},
        {"emitted_tokens", emitted_tokens_},
        {"backend_kind", "mock_spi_v2"},
        {"model_execution_kind", "mock_microkernel"},
        {"prefix_evictions", prefix_evictions_},
        {"contended_prefill_chunks", contended_hint_trues_},
        {"memory_budget_bytes", memory_budget_bytes_},
    };
  }

  void write_audit_locked() const {
    if (model_dir_.empty()) {
      return;
    }
    try {
      const std::filesystem::path audit_path =
          std::filesystem::path(model_dir_) / ".mock_spi_audit.json";
      const std::filesystem::path tmp_path =
          std::filesystem::path(model_dir_) / ".mock_spi_audit.json.tmp";
      Json doc = {
          {"provider_id", provider_id_},
          {"pid", static_cast<std::int64_t>(::getpid())},
          {"model_dir", model_dir_},
          {"artifact", artifact_},
          {"max_concurrency", max_concurrency_},
          {"max_context", max_context_},
          {"kv_dtype", kv_dtype_},
          {"memory_budget_bytes", memory_budget_bytes_},
          {"rolling_prefix_cache", rolling_prefix_cache_},
          {"prefix_cache_gb", prefix_cache_gb_},
          {"scheduler_hint_history", scheduler_hint_history_},
          {"trim_cache_calls", trim_cache_calls_},
          {"last_trim_target_bytes", last_trim_target_bytes_},
          {"shutdown_received", shutdown_received_},
      };
      std::ofstream out(tmp_path, std::ios::out | std::ios::trunc);
      if (out.is_open()) {
        out << doc.dump(2) << '\n';
        out.close();
        std::error_code ec;
        std::filesystem::rename(tmp_path, audit_path, ec);
      }
    } catch (...) {
    }
  }

  void push_event(Json ev) {
    {
      std::lock_guard lock(event_mutex_);
      events_.push_back(std::move(ev));
    }
    event_cv_.notify_all();
  }

  void worker_loop() {
    std::vector<std::shared_ptr<MockJob>> active_jobs;
    for (;;) {
      {
        std::unique_lock lock(mutex_);
        job_cv_.wait(lock, [&] {
          return stopping_ || !active_jobs.empty() || !job_queue_.empty();
        });
        if (stopping_) {
          std::vector<std::shared_ptr<MockJob>> to_abort;
          to_abort.reserve(active_jobs.size() + job_queue_.size());
          for (auto &j : active_jobs) {
            to_abort.push_back(std::move(j));
          }
          active_jobs.clear();
          while (!job_queue_.empty()) {
            to_abort.push_back(std::move(job_queue_.front()));
            job_queue_.pop_front();
          }
          active_requests_ = 0;
          for (const auto &j : to_abort) {
            ++aborted_;
            jobs_by_id_.erase(j->request_id);
          }
          lock.unlock();
          for (const auto &j : to_abort) {
            push_event({
                {"request_id", j->request_id},
                {"tokens", Json::array()},
                {"kind", "aborted"},
                {"finish_reason", "cancelled"},
                {"error", ""},
            });
          }
          return;
        }
        const std::size_t cap =
            static_cast<std::size_t>(std::max(1, max_concurrency_));
        while (active_jobs.size() < cap && !job_queue_.empty()) {
          active_jobs.push_back(std::move(job_queue_.front()));
          job_queue_.pop_front();
        }
        active_requests_ = static_cast<std::int32_t>(active_jobs.size());
      }

      if (active_jobs.empty()) {
        continue;
      }

      if (step_delay_us_ > 0) {
        std::unique_lock lock(mutex_);
        job_cv_.wait_for(lock, std::chrono::microseconds(step_delay_us_),
                         [&] { return stopping_; });
        if (stopping_) {
          continue;
        }
      }

      std::vector<std::shared_ptr<MockJob>> remaining;
      remaining.reserve(active_jobs.size());
      for (auto &job : active_jobs) {
        if (job->cancelled.load(std::memory_order_acquire)) {
          push_event({
              {"request_id", job->request_id},
              {"tokens", Json::array()},
              {"kind", "aborted"},
              {"finish_reason", "cancelled"},
              {"error", ""},
          });
          std::lock_guard lock(mutex_);
          ++aborted_;
          jobs_by_id_.erase(job->request_id);
          continue;
        }

        if (job->step_index >= job->max_new_tokens) {
          if (!job->pending_tokens.empty()) {
            const auto flush_count = job->pending_tokens.size();
            push_event({
                {"request_id", job->request_id},
                {"tokens", job->pending_tokens},
                {"kind", "tokens"},
                {"finish_reason", ""},
                {"error", ""},
            });
            job->pending_tokens.clear();
            std::lock_guard lock(mutex_);
            emitted_tokens_ += flush_count;
          }
          push_event({
              {"request_id", job->request_id},
              {"tokens", Json::array()},
              {"kind", "finished"},
              {"finish_reason", "length"},
              {"error", ""},
          });
          std::lock_guard lock(mutex_);
          ++completed_;
          jobs_by_id_.erase(job->request_id);
          continue;
        }

        const std::int32_t base =
            job->prompt_tokens.empty() ? 0 : job->prompt_tokens.back();
        const std::int32_t tok = (base + job->step_index + 1) % 360;
        ++job->step_index;
        job->pending_tokens.push_back(tok);

        std::size_t matched_len = 0;
        for (const auto &seq : job->stop_token_seqs) {
          if (!seq.empty() && seq.size() <= job->pending_tokens.size() &&
              std::equal(seq.rbegin(), seq.rend(),
                         job->pending_tokens.rbegin())) {
            matched_len = std::max(matched_len, seq.size());
          }
        }
        if (matched_len > 0) {
          job->pending_tokens.resize(job->pending_tokens.size() - matched_len);
          if (!job->pending_tokens.empty()) {
            const auto flush_count = job->pending_tokens.size();
            push_event({
                {"request_id", job->request_id},
                {"tokens", job->pending_tokens},
                {"kind", "tokens"},
                {"finish_reason", ""},
                {"error", ""},
            });
            job->pending_tokens.clear();
            std::lock_guard lock(mutex_);
            emitted_tokens_ += flush_count;
          }
          push_event({
              {"request_id", job->request_id},
              {"tokens", Json::array()},
              {"kind", "finished"},
              {"finish_reason", "stop"},
              {"error", ""},
          });
          std::lock_guard lock(mutex_);
          ++completed_;
          jobs_by_id_.erase(job->request_id);
          continue;
        }

        const std::size_t hold =
            job->max_stop_len > 0 ? (job->max_stop_len - 1) : 0;
        if (job->pending_tokens.size() > hold) {
          const std::size_t flush_n = job->pending_tokens.size() - hold;
          std::vector<std::int32_t> ready(
              job->pending_tokens.begin(),
              job->pending_tokens.begin() +
                  static_cast<std::ptrdiff_t>(flush_n));
          job->pending_tokens.erase(
              job->pending_tokens.begin(),
              job->pending_tokens.begin() +
                  static_cast<std::ptrdiff_t>(flush_n));
          push_event({
              {"request_id", job->request_id},
              {"tokens", ready},
              {"kind", "tokens"},
              {"finish_reason", ""},
              {"error", ""},
          });
          std::lock_guard lock(mutex_);
          emitted_tokens_ += ready.size();
        }

        if (job->step_index >= job->max_new_tokens) {
          if (!job->pending_tokens.empty()) {
            const auto flush_count = job->pending_tokens.size();
            push_event({
                {"request_id", job->request_id},
                {"tokens", job->pending_tokens},
                {"kind", "tokens"},
                {"finish_reason", ""},
                {"error", ""},
            });
            job->pending_tokens.clear();
            std::lock_guard lock(mutex_);
            emitted_tokens_ += flush_count;
          }
          push_event({
              {"request_id", job->request_id},
              {"tokens", Json::array()},
              {"kind", "finished"},
              {"finish_reason", "length"},
              {"error", ""},
          });
          std::lock_guard lock(mutex_);
          ++completed_;
          jobs_by_id_.erase(job->request_id);
          continue;
        }

        remaining.push_back(std::move(job));
      }

      active_jobs = std::move(remaining);
      {
        std::lock_guard lock(mutex_);
        active_requests_ = static_cast<std::int32_t>(active_jobs.size());
      }
    }
  }

  std::string provider_id_ = "io.test.mock-microkernel";
  std::int32_t step_delay_us_ = 20'000;

  std::mutex shutdown_mutex_;
  mutable std::mutex mutex_;
  std::condition_variable job_cv_;
  bool loaded_ = false;
  bool stopping_ = false;
  std::atomic<bool> stopping_atomic_{false};
  std::string model_dir_;
  std::string artifact_;
  std::int32_t max_concurrency_ = 1;
  std::int32_t max_context_ = 4096;
  std::string kv_dtype_ = "bf16";
  std::uint64_t memory_budget_bytes_ = 0;
  bool rolling_prefix_cache_ = false;
  double prefix_cache_gb_ = 0.0;
  bool gpu_contended_ = false;
  std::vector<bool> scheduler_hint_history_;
  std::uint64_t contended_hint_trues_ = 0;
  std::uint64_t trim_cache_calls_ = 0;
  std::uint64_t last_trim_target_bytes_ = 0;
  std::uint64_t prefix_evictions_ = 0;
  bool shutdown_received_ = false;

  std::uint64_t next_request_id_ = 1;
  std::uint64_t submitted_ = 0;
  std::uint64_t completed_ = 0;
  std::uint64_t aborted_ = 0;
  std::uint64_t failed_ = 0;
  std::uint64_t emitted_tokens_ = 0;
  std::int32_t active_requests_ = 0;
  std::unordered_map<std::uint64_t, std::shared_ptr<MockJob>> jobs_by_id_;
  std::deque<std::shared_ptr<MockJob>> job_queue_;
  std::thread worker_;

  std::mutex event_mutex_;
  std::condition_variable event_cv_;
  std::deque<Json> events_;
  bool events_eof_ = false;
};

void handle_client(int client_fd, int stop_read_fd, int stop_write_fd,
                   MockSession &session, bool &served_any) {
  std::string buf;
  while (const auto line = read_line(client_fd, stop_read_fd, buf)) {
    served_any = true;
    Json resp;
    Json id;
    bool enter_sub = false;
    std::chrono::microseconds poll_interval{20'000};
    try {
      const Json req = Json::parse(*line);
      id = req.at("id");
      const std::string method = req.at("method").get<std::string>();
      const Json params = req.value("params", Json::object());
      if (method == "subscribe_events") {
        const auto us = params.value("poll_interval_us", 20'000LL);
        poll_interval =
            std::chrono::microseconds(std::max<std::int64_t>(1000, us));
        enter_sub = true;
        resp = {{"id", id}, {"ok", true}, {"result", {{"subscribed", true}}}};
      } else {
        resp = {{"id", id}, {"ok", true}, {"result", session.execute(method, params)}};
      }
    } catch (const std::exception &ex) {
      enter_sub = false;
      resp = {{"id", id},
              {"ok", false},
              {"error", {{"code", "request_failed"}, {"message", ex.what()}}}};
    }

    if (!write_line(client_fd, resp.dump())) {
      break;
    }
    if (session.is_stopping() && !enter_sub) {
      trigger_pipe(stop_write_fd);
      break;
    }
    if (enter_sub) {
      for (;;) {
        const auto events = session.poll_events(poll_interval);
        for (const auto &ev : events) {
          if (!write_line(client_fd, Json({{"event", ev}}).dump())) {
            return;
          }
        }
        if (session.is_stopping() || pipe_signaled(stop_read_fd)) {
          session.shutdown();
          const auto final_events =
              session.poll_events(std::chrono::microseconds(0));
          for (const auto &ev : final_events) {
            if (!write_line(client_fd, Json({{"event", ev}}).dump())) {
              return;
            }
          }
          (void)write_line(client_fd,
                           Json({{"event", {{"kind", "eof"}}}}).dump());
          return;
        }
        char peek = 0;
        const auto pr = ::recv(client_fd, &peek, 1, MSG_PEEK | MSG_DONTWAIT);
        if (pr == 0 ||
            (pr < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) {
          return;
        }
      }
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  bool exit_on_disconnect = false;
  std::filesystem::path socket_path;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--version") {
      std::cout << "mock-spi-provider mock-v0.01 protocol=2\n";
      return 0;
    }
    if (arg == "--exit-on-disconnect") {
      exit_on_disconnect = true;
    } else if (arg == "--socket" && i + 1 < argc) {
      socket_path = argv[++i];
    }
  }
  if (socket_path.empty()) {
    std::cerr << "usage: omlx_m5_mock_spi_provider --socket PATH "
                 "[--exit-on-disconnect]\n";
    return 2;
  }
  if (socket_path.native().size() >= sizeof(sockaddr_un::sun_path)) {
    std::cerr << "mock-spi-provider: Unix socket path is too long\n";
    return 1;
  }

  std::error_code status_ec;
  const auto existing_status =
      std::filesystem::symlink_status(socket_path, status_ec);
  if (existing_status.type() != std::filesystem::file_type::not_found) {
    if ((status_ec && status_ec != std::errc::no_such_file_or_directory) ||
        !std::filesystem::is_socket(existing_status)) {
      std::cerr << "mock-spi-provider: Unix socket path already exists\n";
      return 1;
    }
    const int probe_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (probe_fd < 0) {
      return 1;
    }
    set_cloexec(probe_fd);
    configure_nosigpipe(probe_fd);
    sockaddr_un probe_addr{};
    probe_addr.sun_family = AF_UNIX;
    std::memcpy(probe_addr.sun_path, socket_path.c_str(),
                socket_path.native().size() + 1);
    const int probe_rc =
        ::connect(probe_fd, reinterpret_cast<sockaddr *>(&probe_addr),
                  sizeof(probe_addr));
    const int probe_err = errno;
    ::close(probe_fd);
    if (probe_rc == 0) {
      std::cerr << "mock-spi-provider: Unix socket path already in use\n";
      return 1;
    }
    if (probe_err != ECONNREFUSED && probe_err != ENOENT) {
      std::cerr << "mock-spi-provider: Unix socket path already exists\n";
      return 1;
    }
    std::error_code remove_ec;
    std::filesystem::remove(socket_path, remove_ec);
    if (remove_ec) {
      std::cerr << "mock-spi-provider: failed to remove stale Unix socket\n";
      return 1;
    }
  }

  int stop_pipe[2] = {-1, -1};
  if (::pipe(stop_pipe) != 0) {
    return 1;
  }
  set_cloexec(stop_pipe[0]);
  set_cloexec(stop_pipe[1]);
  set_nonblocking(stop_pipe[0], true);
  set_nonblocking(stop_pipe[1], true);
  g_stop_write_fd.store(stop_pipe[1], std::memory_order_release);
  std::signal(SIGPIPE, SIG_IGN);
  struct sigaction act {};
  act.sa_handler = handle_signal;
  sigemptyset(&act.sa_mask);
  (void)::sigaction(SIGTERM, &act, nullptr);
  (void)::sigaction(SIGINT, &act, nullptr);

  std::error_code ec;
  if (!socket_path.parent_path().empty()) {
    std::filesystem::create_directories(socket_path.parent_path(), ec);
  }

  const int listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    return 1;
  }
  set_cloexec(listen_fd);
  configure_nosigpipe(listen_fd);
  set_nonblocking(listen_fd, true);

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::memcpy(addr.sun_path, socket_path.c_str(),
              socket_path.native().size() + 1);
  if (::bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
      ::listen(listen_fd, 16) != 0) {
    ::close(listen_fd);
    return 1;
  }

  MockSession session;
  std::mutex workers_mutex;
  std::condition_variable workers_cv;
  std::unordered_map<std::uint64_t, std::thread> active;
  std::vector<std::thread> finished;
  std::uint64_t next_cid = 1;

  const auto reap_finished = [&] {
    std::vector<std::thread> to_join;
    {
      std::lock_guard lock(workers_mutex);
      to_join.swap(finished);
    }
    for (auto &t : to_join) {
      if (t.joinable()) {
        t.join();
      }
    }
  };

  for (;;) {
    reap_finished();
    std::array<pollfd, 2> pfds{{
        {.fd = listen_fd, .events = POLLIN, .revents = 0},
        {.fd = stop_pipe[0], .events = POLLIN, .revents = 0},
    }};
    const int ready = ::poll(pfds.data(), pfds.size(), -1);
    if (ready < 0) {
      if (errno == EINTR) {
        if (pipe_signaled(stop_pipe[0])) {
          break;
        }
        continue;
      }
      break;
    }
    if ((pfds[1].revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
      break;
    }
    if ((pfds[0].revents & POLLIN) == 0) {
      continue;
    }
    for (;;) {
      const int cfd = ::accept(listen_fd, nullptr, nullptr);
      if (cfd < 0) {
        if (errno == EINTR) {
          continue;
        }
        break;
      }
      set_cloexec(cfd);
      configure_nosigpipe(cfd);
      set_nonblocking(cfd, false);

      std::lock_guard lock(workers_mutex);
      const std::uint64_t cid = next_cid++;
      active.emplace(cid, std::thread([&, cid, cfd] {
                       bool served = false;
                       handle_client(cfd, stop_pipe[0], stop_pipe[1], session,
                                     served);
                       ::close(cfd);
                       std::lock_guard done_lock(workers_mutex);
                       const auto it = active.find(cid);
                       if (it != active.end()) {
                         finished.push_back(std::move(it->second));
                         active.erase(it);
                       }
                       if (exit_on_disconnect && served && active.empty()) {
                         trigger_pipe(stop_pipe[1]);
                       }
                       workers_cv.notify_all();
                     }));
    }
  }

  trigger_pipe(stop_pipe[1]);
  session.shutdown();
  {
    std::unique_lock lock(workers_mutex);
    workers_cv.wait(lock, [&] { return active.empty(); });
  }
  reap_finished();
  ::close(listen_fd);
  ::close(stop_pipe[0]);
  ::close(stop_pipe[1]);
  std::filesystem::remove(socket_path, ec);
  return 0;
}
