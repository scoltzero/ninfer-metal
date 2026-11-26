#include "serve/http_server.h"

#include "serve/output_parser.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task_info.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef NINFER_METAL_DISPLAY_VERSION
#define NINFER_METAL_DISPLAY_VERSION "miv-v0.02"
#endif

namespace omlx::m5::serve {
namespace {

using Json = nlohmann::json;

std::int64_t unix_timestamp_now() {
  return static_cast<std::int64_t>(std::time(nullptr));
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

void write_json_response(httplib::Response &res, int status,
                         const Json &payload) {
  res.status = status;
  res.set_content(payload.dump(), "application/json; charset=utf-8");
}

Json make_openai_error_json(std::string_view message, std::string_view type,
                            std::string_view code = "") {
  return {
      {"error",
       {
           {"message", std::string(message)},
           {"type", std::string(type)},
           {"param", nullptr},
           {"code", code.empty() ? Json(nullptr) : Json(std::string(code))},
       }},
  };
}

void write_openai_error(httplib::Response &res, int status,
                        std::string_view message, std::string_view type,
                        std::string_view code = "") {
  write_json_response(res, status, make_openai_error_json(message, type, code));
}

std::vector<std::string> parse_stop_strings(const Json &body) {
  std::vector<std::string> stops;
  if (!body.contains("stop") || body.at("stop").is_null()) {
    return stops;
  }
  const auto &val = body.at("stop");
  if (val.is_string()) {
    const auto s = val.get<std::string>();
    if (!s.empty()) {
      stops.push_back(s);
    }
    return stops;
  }
  if (val.is_array()) {
    for (const auto &item : val) {
      if (!item.is_string()) {
        throw std::invalid_argument("stop array elements must be strings");
      }
      const auto s = item.get<std::string>();
      if (!s.empty()) {
        stops.push_back(s);
      }
    }
    return stops;
  }
  throw std::invalid_argument("stop must be a string or an array of strings");
}

SamplingSpec parse_sampling_spec(const Json &body) {
  SamplingSpec spec;
  if (body.contains("temperature") && !body.at("temperature").is_null()) {
    if (!body.at("temperature").is_number()) {
      throw std::invalid_argument("temperature must be a number");
    }
    spec.temperature = body.at("temperature").get<float>();
    if (!std::isfinite(spec.temperature) || spec.temperature < 0.0F) {
      throw std::invalid_argument("temperature must be finite and >= 0");
    }
  }
  if (body.contains("top_p") && !body.at("top_p").is_null()) {
    if (!body.at("top_p").is_number()) {
      throw std::invalid_argument("top_p must be a number");
    }
    spec.top_p = body.at("top_p").get<float>();
    if (!std::isfinite(spec.top_p) || spec.top_p <= 0.0F ||
        spec.top_p > 1.0F) {
      throw std::invalid_argument("top_p must be finite and in (0, 1]");
    }
  }
  if (body.contains("min_p") && !body.at("min_p").is_null()) {
    if (!body.at("min_p").is_number()) {
      throw std::invalid_argument("min_p must be a number");
    }
    spec.min_p = body.at("min_p").get<float>();
    if (!std::isfinite(spec.min_p) || spec.min_p < 0.0F ||
        spec.min_p > 1.0F) {
      throw std::invalid_argument("min_p must be finite and in [0, 1]");
    }
    if (spec.min_p != 0.0F) {
      throw std::invalid_argument(
          "min_p > 0 is not supported (only min_p=0 is supported)");
    }
  }
  if (body.contains("top_k") && !body.at("top_k").is_null()) {
    if (!body.at("top_k").is_number_integer()) {
      throw std::invalid_argument("top_k must be an integer");
    }
    const auto raw_top_k = body.at("top_k").get<std::int64_t>();
    if (raw_top_k < 0 || raw_top_k > 64) {
      throw std::invalid_argument("top_k must be in [0, 64]");
    }
    spec.top_k = static_cast<std::int32_t>(raw_top_k);
  }
  if (spec.temperature > 0.0F && spec.top_k == 0) {
    spec.top_k = 64;
  }
  if (body.contains("seed") && !body.at("seed").is_null()) {
    if (!body.at("seed").is_number_integer()) {
      throw std::invalid_argument("seed must be an integer");
    }
    if (body.at("seed").is_number_unsigned()) {
      spec.seed = body.at("seed").get<std::uint64_t>();
    } else {
      const auto s = body.at("seed").get<std::int64_t>();
      spec.seed = static_cast<std::uint64_t>(s);
    }
  }
  return spec;
}

bool parse_stream_flag(const Json &body) {
  if (!body.contains("stream") || body.at("stream").is_null()) {
    return false;
  }
  if (!body.at("stream").is_boolean()) {
    throw std::invalid_argument("stream must be a boolean");
  }
  return body.at("stream").get<bool>();
}

bool parse_include_usage(const Json &body) {
  if (!body.contains("stream_options") || body.at("stream_options").is_null()) {
    return false;
  }
  const auto &opts = body.at("stream_options");
  if (!opts.is_object()) {
    throw std::invalid_argument("stream_options must be an object");
  }
  if (opts.contains("include_usage") && !opts.at("include_usage").is_null()) {
    if (!opts.at("include_usage").is_boolean()) {
      throw std::invalid_argument("stream_options.include_usage must be bool");
    }
    return opts.at("include_usage").get<bool>();
  }
  return false;
}

std::int32_t parse_max_new_tokens(const Json &body, std::size_t prompt_len,
                                  std::int32_t max_context,
                                  std::int32_t default_tokens) {
  if (prompt_len >= static_cast<std::size_t>(max_context)) {
    throw std::invalid_argument("prompt length meets or exceeds max_context");
  }
  const std::int32_t remaining =
      max_context - static_cast<std::int32_t>(prompt_len);

  std::optional<std::int64_t> requested;
  if (body.contains("max_completion_tokens") &&
      !body.at("max_completion_tokens").is_null()) {
    if (!body.at("max_completion_tokens").is_number_integer()) {
      throw std::invalid_argument("max_completion_tokens must be an integer");
    }
    requested = body.at("max_completion_tokens").get<std::int64_t>();
  } else if (body.contains("max_tokens") && !body.at("max_tokens").is_null()) {
    if (!body.at("max_tokens").is_number_integer()) {
      throw std::invalid_argument("max_tokens must be an integer");
    }
    requested = body.at("max_tokens").get<std::int64_t>();
  }

  if (!requested.has_value()) {
    return std::min(default_tokens, remaining);
  }
  if (*requested <= 0) {
    throw std::invalid_argument("max_tokens must be positive");
  }
  if (*requested > static_cast<std::int64_t>(remaining)) {
    throw std::invalid_argument(
        "prompt plus max_tokens exceeds max_context (" +
        std::to_string(max_context) + ")");
  }
  return static_cast<std::int32_t>(*requested);
}

std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
extract_tool_param_types(const Json &tools) {
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      result;
  if (!tools.is_array()) {
    return result;
  }
  for (const auto &tool : tools) {
    if (!tool.is_object()) {
      continue;
    }
    const Json *fn = &tool;
    if (tool.contains("function") && tool.at("function").is_object()) {
      fn = &tool.at("function");
    }
    if (!fn->contains("name") || !fn->at("name").is_string()) {
      continue;
    }
    const std::string name = fn->at("name").get<std::string>();
    if (!fn->contains("parameters") || !fn->at("parameters").is_object()) {
      continue;
    }
    const auto &params = fn->at("parameters");
    if (!params.contains("properties") || !params.at("properties").is_object()) {
      continue;
    }
    auto &param_map = result[name];
    for (const auto &prop : params.at("properties").items()) {
      if (prop.value().is_object() && prop.value().contains("type") &&
          prop.value().at("type").is_string()) {
        param_map.emplace(prop.key(),
                          prop.value().at("type").get<std::string>());
      }
    }
  }
  return result;
}

Json format_tool_call_object(const ParsedToolCall &tc) {
  return {
      {"id", tc.id},
      {"type", "function"},
      {"function",
       {
           {"name", tc.name},
           {"arguments", tc.arguments},
       }},
  };
}

Json format_tool_call_delta(const ParsedToolCall &tc) {
  return {
      {"index", tc.index},
      {"id", tc.id},
      {"type", "function"},
      {"function",
       {
           {"name", tc.name},
           {"arguments", tc.arguments},
       }},
  };
}

bool send_sse_json(httplib::DataSink &sink, const Json &payload) {
  if (!sink.is_writable()) {
    return false;
  }
  const std::string chunk = "data: " + payload.dump() + "\n\n";
  return sink.write(chunk.data(), chunk.size());
}

bool send_sse_done(httplib::DataSink &sink) {
  if (!sink.is_writable()) {
    return false;
  }
  static constexpr std::string_view kDone = "data: [DONE]\n\n";
  return sink.write(kDone.data(), kDone.size());
}

std::string_view safe_decode_token_bytes(const Tokenizer &tokenizer,
                                         std::int32_t tok) noexcept {
  const int id = static_cast<int>(tok);
  if (!tokenizer.is_valid_token(id)) {
    return {};
  }
  try {
    return tokenizer.decode_token_bytes(id, true);
  } catch (...) {
    return {};
  }
}

std::string resolve_finish_reason(const std::string &backend_reason,
                                  bool stopped_by_string,
                                  std::size_t tool_call_count) {
  std::string reason = backend_reason;
  if (stopped_by_string) {
    reason = "stop";
  } else if (reason.empty()) {
    reason = "stop";
  }
  if (tool_call_count > 0 && reason == "stop") {
    return "tool_calls";
  }
  return reason;
}

std::string escape_prometheus_label(std::string_view input) {
  std::string out;
  out.reserve(input.size());
  for (const char ch : input) {
    if (ch == '\\') {
      out += "\\\\";
    } else if (ch == '"') {
      out += "\\\"";
    } else if (ch == '\n') {
      out += "\\n";
    } else {
      out.push_back(ch);
    }
  }
  return out;
}

} // namespace

struct ServeHttpServer::Impl {
  ServeConfig config;
  std::shared_ptr<MultiModelHost> host_;
  std::string primary_model_id;
  std::int64_t created_ts = unix_timestamp_now();
  httplib::Server server;
  std::atomic<int> bound_port{-1};
  std::atomic<std::uint64_t> next_completion_seq{1};

  Impl(ServeConfig cfg, std::shared_ptr<MultiModelHost> host)
      : config(std::move(cfg)), host_(std::move(host)) {
    if (host_ == nullptr || host_->model_count() == 0) {
      throw std::invalid_argument(
          "ServeHttpServer requires a MultiModelHost with at least one model");
    }
    primary_model_id = host_->primary_model_id();

    std::size_t total_concurrency = 0;
    for (const auto &m : host_->models()) {
      total_concurrency += static_cast<std::size_t>(
          std::max<std::int32_t>(1, m->max_concurrency));
    }
    const std::size_t pool_threads =
        std::max<std::size_t>(16U, total_concurrency * 4U + 4U);
    server.new_task_queue = [pool_threads] {
      return new httplib::ThreadPool(pool_threads);
    };
    register_routes();
  }

  Json model_object(const ModelInstance &m) const {
    std::string lora_adapter = m.lora_adapter;
    std::string lora_mode =
        m.lora_adapter.empty() ? std::string{} : m.lora_mode;
    std::uint64_t lora_buffer_bytes = 0;
    std::uint32_t lora_adapted_layers = 0;
    if (m.router != nullptr && m.router->is_running()) {
      try {
        const auto st = m.router->stats();
        lora_adapter = st.lora_adapter;
        lora_mode = st.lora_mode;
        lora_buffer_bytes = st.lora_buffer_bytes;
        lora_adapted_layers = st.lora_adapted_layers;
      } catch (...) {
      }
    }
    return {
        {"id", m.id},
        {"object", "model"},
        {"created", created_ts},
        {"owned_by", "ninfer-metal"},
        {"max_model_len", m.max_context},
        {"provider_id", m.provider_id},
        {"backend_mode", m.backend_mode},
        {"lora_adapter", std::move(lora_adapter)},
        {"lora_mode", std::move(lora_mode)},
        {"lora_buffer_bytes", lora_buffer_bytes},
        {"lora_adapted_layers", lora_adapted_layers},
    };
  }

  std::shared_ptr<ModelInstance>
  resolve_target_model(const Json &body, httplib::Response &res) const {
    std::string requested;
    if (body.contains("model") && !body.at("model").is_null()) {
      if (!body.at("model").is_string()) {
        write_openai_error(res, 400, "model must be a string",
                           "invalid_request_error", "invalid_model");
        return nullptr;
      }
      requested = body.at("model").get<std::string>();
    }
    if (requested.empty()) {
      const auto all = host_->models();
      if (all.size() == 1) {
        return all.front();
      }
      write_openai_error(
          res, 400, "model field is required when multiple models are loaded",
          "invalid_request_error", "missing_model");
      return nullptr;
    }
    auto found = host_->find_model(requested);
    if (found == nullptr) {
      write_openai_error(res, 404, "Model '" + requested + "' not found",
                         "invalid_request_error", "model_not_found");
      return nullptr;
    }
    return found;
  }

  void apply_cors_headers(httplib::Response &res) const {
    if (!config.cors) {
      return;
    }
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers",
                   "Authorization, Content-Type");
  }

  bool authorize_request(const httplib::Request &req,
                         httplib::Response &res) const {
    if (config.api_key.empty()) {
      return true;
    }
    const std::string expected = "Bearer " + config.api_key;
    const auto auth = req.get_header_value("Authorization");
    if (auth != expected) {
      write_openai_error(res, 401, "Invalid or missing API key",
                         "authentication_error", "invalid_api_key");
      return false;
    }
    return true;
  }

  void register_routes() {
    server.set_pre_routing_handler(
        [this](const httplib::Request &req, httplib::Response &res) {
          apply_cors_headers(res);
          if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
          }
          if (req.path.starts_with("/v1/") && !authorize_request(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
          }
          return httplib::Server::HandlerResponse::Unhandled;
        });

    server.set_exception_handler(
        [this](const httplib::Request & /*req*/, httplib::Response &res,
               std::exception_ptr ep) {
          apply_cors_headers(res);
          std::string msg = "internal server error";
          if (ep != nullptr) {
            try {
              std::rethrow_exception(ep);
            } catch (const std::exception &err) {
              if (err.what() != nullptr && *err.what() != '\0') {
                msg = err.what();
              }
            } catch (...) {
            }
          }
          write_openai_error(res, 500, msg, "server_error");
        });

    server.set_error_handler(
        [this](const httplib::Request & /*req*/, httplib::Response &res) {
          apply_cors_headers(res);
          if (res.body.empty()) {
            if (res.status == 404) {
              write_openai_error(res, 404, "Endpoint not found",
                                 "invalid_request_error", "not_found");
            } else if (res.status == 405) {
              write_openai_error(res, 405, "Method not allowed",
                                 "invalid_request_error", "method_not_allowed");
            }
          }
        });

    server.Get("/health",
               [this](const httplib::Request &, httplib::Response &res) {
                 handle_health(res);
               });

    server.Get("/metrics",
               [this](const httplib::Request &, httplib::Response &res) {
                 handle_metrics(res);
               });

    server.Get("/v1/models",
               [this](const httplib::Request &, httplib::Response &res) {
                 Json data = Json::array();
                 for (const auto &m : host_->models()) {
                   data.push_back(model_object(*m));
                 }
                 write_json_response(
                     res, 200,
                     {{"object", "list"}, {"data", std::move(data)}});
               });

    server.Get(R"(/v1/models/(.+))",
               [this](const httplib::Request &req, httplib::Response &res) {
                 const std::string requested_id = req.matches[1].str();
                 const auto found = host_->find_model(requested_id);
                 if (found == nullptr) {
                   write_openai_error(res, 404,
                                      "Model '" + requested_id + "' not found",
                                      "invalid_request_error",
                                      "model_not_found");
                   return;
                 }
                 write_json_response(res, 200, model_object(*found));
               });

    server.Post("/v1/chat/completions",
                [this](const httplib::Request &req, httplib::Response &res) {
                  handle_chat_completions(req, res);
                });

    server.Post("/v1/completions",
                [this](const httplib::Request &req, httplib::Response &res) {
                  handle_completions(req, res);
                });
  }

  void handle_health(httplib::Response &res) const {
    const auto all_models = host_->models();
    Json models_json = Json::array();
    if (!host_->all_running()) {
      for (const auto &m : all_models) {
        models_json.push_back({
            {"id", m->id},
            {"provider_id", m->provider_id},
            {"backend_mode", m->backend_mode},
        });
      }
      write_json_response(
          res, 503,
          {{"status", "shutting_down"},
           {"model", primary_model_id},
           {"version", NINFER_METAL_DISPLAY_VERSION},
           {"models", std::move(models_json)}});
      return;
    }

    std::int32_t total_max_concurrency = 0;
    std::int32_t total_active = 0;
    std::string primary_backend;
    std::string primary_lora_adapter;
    std::string primary_lora_mode;
    std::uint64_t total_lora_buffer_bytes = 0;
    std::uint32_t total_lora_adapted_layers = 0;
    bool captured_primary_lora = false;
    std::size_t healthy_count = 0;
    std::size_t unhealthy_count = 0;
    for (const auto &m : all_models) {
      try {
        const auto cap = m->router->capacity();
        const auto st = m->router->stats();
        const auto resident = m->router->resident_memory_bytes();
        if (primary_backend.empty()) {
          primary_backend = st.backend_kind;
        }
        if (!captured_primary_lora) {
          primary_lora_adapter = st.lora_adapter;
          primary_lora_mode = st.lora_mode;
          captured_primary_lora = true;
        }
        total_lora_buffer_bytes += st.lora_buffer_bytes;
        total_lora_adapted_layers += st.lora_adapted_layers;
        total_max_concurrency += cap.max_concurrency;
        total_active += cap.active_requests;
        ++healthy_count;
        models_json.push_back({
            {"id", m->id},
            {"provider_id", m->provider_id},
            {"backend_mode", m->backend_mode},
            {"status", "ok"},
            {"backend", st.backend_kind},
            {"max_concurrency", cap.max_concurrency},
            {"active_requests", cap.active_requests},
            {"resident_memory_bytes", resident},
            {"lora_adapter", st.lora_adapter},
            {"lora_mode", st.lora_mode},
            {"lora_buffer_bytes", st.lora_buffer_bytes},
            {"lora_adapted_layers", st.lora_adapted_layers},
        });
      } catch (const std::exception &err) {
        ++unhealthy_count;
        std::uint64_t resident = 0;
        try {
          resident = m->router->resident_memory_bytes();
        } catch (...) {
        }
        models_json.push_back({
            {"id", m->id},
            {"provider_id", m->provider_id},
            {"backend_mode", m->backend_mode},
            {"status", "error"},
            {"error", err.what()},
            {"max_concurrency", m->max_concurrency},
            {"active_requests",
             static_cast<std::int32_t>(m->router->active_requests())},
            {"resident_memory_bytes", resident},
            {"lora_adapter", m->lora_adapter},
            {"lora_mode",
             m->lora_adapter.empty() ? std::string{} : m->lora_mode},
            {"lora_buffer_bytes", 0ULL},
            {"lora_adapted_layers", 0U},
        });
      } catch (...) {
        ++unhealthy_count;
        models_json.push_back({
            {"id", m->id},
            {"provider_id", m->provider_id},
            {"backend_mode", m->backend_mode},
            {"status", "error"},
            {"error", "backend health check failed"},
            {"max_concurrency", m->max_concurrency},
            {"active_requests",
             static_cast<std::int32_t>(m->router->active_requests())},
            {"resident_memory_bytes", 0},
            {"lora_adapter", m->lora_adapter},
            {"lora_mode",
             m->lora_adapter.empty() ? std::string{} : m->lora_mode},
            {"lora_buffer_bytes", 0ULL},
            {"lora_adapted_layers", 0U},
        });
      }
    }
    const int http_status = unhealthy_count == 0 ? 200 : 503;
    const char *health_status =
        unhealthy_count == 0 ? "ok"
                             : (healthy_count > 0 ? "degraded" : "unhealthy");
    write_json_response(
        res, http_status,
        {{"status", health_status},
         {"model", primary_model_id},
         {"version", NINFER_METAL_DISPLAY_VERSION},
         {"backend", primary_backend},
         {"max_concurrency", total_max_concurrency},
         {"active_requests", total_active},
         {"gpu_contended", host_->is_gpu_contended()},
         {"memory_budget_bytes", host_->total_memory_budget_bytes()},
         {"lora_adapter", primary_lora_adapter},
         {"lora_mode", primary_lora_mode},
         {"lora_buffer_bytes", total_lora_buffer_bytes},
         {"lora_adapted_layers", total_lora_adapted_layers},
         {"models", std::move(models_json)}});
  }

  void handle_metrics(httplib::Response &res) const {
    const auto all_models = host_->models();
    const auto phys_bytes = current_phys_footprint_bytes();

    struct PerModelSnapshot {
      std::shared_ptr<ModelInstance> model;
      RouterMetricsSnapshot snap;
      std::uint64_t resident_bytes = 0;
    };
    std::vector<PerModelSnapshot> per_model;
    per_model.reserve(all_models.size());

    RouterMetricsSnapshot agg{};
    agg.capacity.max_concurrency = 0;
    for (const auto &m : all_models) {
      RouterMetricsSnapshot snap;
      std::uint64_t res_bytes = 0;
      try {
        snap = m->router->metrics_snapshot();
      } catch (...) {
        snap.backend_healthy = false;
        snap.capacity.accepting = false;
      }
      try {
        res_bytes = m->router->resident_memory_bytes();
      } catch (...) {
      }
      agg.requests_total += snap.requests_total;
      agg.requests_succeeded += snap.requests_succeeded;
      agg.requests_failed += snap.requests_failed;
      agg.requests_cancelled += snap.requests_cancelled;
      agg.active_requests += snap.active_requests;
      agg.generated_tokens_total += snap.generated_tokens_total;
      agg.prompt_tokens_total += snap.prompt_tokens_total;
      agg.capacity.queued_requests += snap.capacity.queued_requests;
      agg.capacity.max_concurrency += snap.backend_healthy
                                          ? snap.capacity.max_concurrency
                                          : m->max_concurrency;
      agg.engine_stats.emitted_tokens += snap.engine_stats.emitted_tokens;
      agg.engine_stats.prefix_hits += snap.engine_stats.prefix_hits;
      agg.engine_stats.prefix_partial_hits +=
          snap.engine_stats.prefix_partial_hits;
      agg.engine_stats.prefix_misses += snap.engine_stats.prefix_misses;
      agg.engine_stats.prefix_evictions += snap.engine_stats.prefix_evictions;
      agg.engine_stats.prefix_restored_tokens +=
          snap.engine_stats.prefix_restored_tokens;
      agg.engine_stats.spec_rounds += snap.engine_stats.spec_rounds;
      agg.engine_stats.spec_proposal_tokens +=
          snap.engine_stats.spec_proposal_tokens;
      agg.engine_stats.spec_accepted_tokens +=
          snap.engine_stats.spec_accepted_tokens;
      per_model.push_back(PerModelSnapshot{
          .model = m,
          .snap = std::move(snap),
          .resident_bytes = res_bytes,
      });
    }

    std::ostringstream oss;
    oss << "# HELP ninfer_metal_requests_total Total inference requests "
           "submitted.\n"
        << "# TYPE ninfer_metal_requests_total counter\n"
        << "ninfer_metal_requests_total " << agg.requests_total << "\n"
        << "# HELP ninfer_metal_requests_succeeded_total Total completed "
           "requests.\n"
        << "# TYPE ninfer_metal_requests_succeeded_total counter\n"
        << "ninfer_metal_requests_succeeded_total " << agg.requests_succeeded
        << "\n"
        << "# HELP ninfer_metal_requests_failed_total Total failed requests.\n"
        << "# TYPE ninfer_metal_requests_failed_total counter\n"
        << "ninfer_metal_requests_failed_total " << agg.requests_failed << "\n"
        << "# HELP ninfer_metal_requests_cancelled_total Total cancelled "
           "requests.\n"
        << "# TYPE ninfer_metal_requests_cancelled_total counter\n"
        << "ninfer_metal_requests_cancelled_total " << agg.requests_cancelled
        << "\n"
        << "# HELP ninfer_metal_active_requests Current in-flight requests.\n"
        << "# TYPE ninfer_metal_active_requests gauge\n"
        << "ninfer_metal_active_requests " << agg.active_requests << "\n"
        << "# HELP ninfer_metal_queued_requests Current queued requests in "
           "engine.\n"
        << "# TYPE ninfer_metal_queued_requests gauge\n"
        << "ninfer_metal_queued_requests " << agg.capacity.queued_requests
        << "\n"
        << "# HELP ninfer_metal_max_concurrency Configured max concurrent "
           "slots.\n"
        << "# TYPE ninfer_metal_max_concurrency gauge\n"
        << "ninfer_metal_max_concurrency " << agg.capacity.max_concurrency
        << "\n"
        << "# HELP ninfer_metal_generated_tokens_total Total generated tokens "
           "delivered.\n"
        << "# TYPE ninfer_metal_generated_tokens_total counter\n"
        << "ninfer_metal_generated_tokens_total " << agg.generated_tokens_total
        << "\n"
        << "# HELP ninfer_metal_prompt_tokens_total Total prompt tokens "
           "processed.\n"
        << "# TYPE ninfer_metal_prompt_tokens_total counter\n"
        << "ninfer_metal_prompt_tokens_total " << agg.prompt_tokens_total
        << "\n"
        << "# HELP ninfer_metal_engine_emitted_tokens_total Total tokens "
           "emitted by engine.\n"
        << "# TYPE ninfer_metal_engine_emitted_tokens_total counter\n"
        << "ninfer_metal_engine_emitted_tokens_total "
        << agg.engine_stats.emitted_tokens << "\n"
        << "# HELP ninfer_metal_prefix_cache_hits_total Prefix cache full "
           "hits.\n"
        << "# TYPE ninfer_metal_prefix_cache_hits_total counter\n"
        << "ninfer_metal_prefix_cache_hits_total "
        << agg.engine_stats.prefix_hits << "\n"
        << "# HELP ninfer_metal_prefix_cache_partial_hits_total Prefix cache "
           "partial hits.\n"
        << "# TYPE ninfer_metal_prefix_cache_partial_hits_total counter\n"
        << "ninfer_metal_prefix_cache_partial_hits_total "
        << agg.engine_stats.prefix_partial_hits << "\n"
        << "# HELP ninfer_metal_prefix_cache_misses_total Prefix cache "
           "misses.\n"
        << "# TYPE ninfer_metal_prefix_cache_misses_total counter\n"
        << "ninfer_metal_prefix_cache_misses_total "
        << agg.engine_stats.prefix_misses << "\n"
        << "# HELP ninfer_metal_prefix_cache_evictions_total Prefix cache "
           "evictions.\n"
        << "# TYPE ninfer_metal_prefix_cache_evictions_total counter\n"
        << "ninfer_metal_prefix_cache_evictions_total "
        << agg.engine_stats.prefix_evictions << "\n"
        << "# HELP ninfer_metal_prefix_cache_restored_tokens_total Total "
           "tokens restored from prefix cache.\n"
        << "# TYPE ninfer_metal_prefix_cache_restored_tokens_total counter\n"
        << "ninfer_metal_prefix_cache_restored_tokens_total "
        << agg.engine_stats.prefix_restored_tokens << "\n"
        << "# HELP ninfer_metal_spec_rounds_total Speculative decoding "
           "rounds.\n"
        << "# TYPE ninfer_metal_spec_rounds_total counter\n"
        << "ninfer_metal_spec_rounds_total " << agg.engine_stats.spec_rounds
        << "\n"
        << "# HELP ninfer_metal_spec_proposal_tokens_total Speculative "
           "proposed tokens.\n"
        << "# TYPE ninfer_metal_spec_proposal_tokens_total counter\n"
        << "ninfer_metal_spec_proposal_tokens_total "
        << agg.engine_stats.spec_proposal_tokens << "\n"
        << "# HELP ninfer_metal_spec_accepted_tokens_total Speculative "
           "accepted tokens.\n"
        << "# TYPE ninfer_metal_spec_accepted_tokens_total counter\n"
        << "ninfer_metal_spec_accepted_tokens_total "
        << agg.engine_stats.spec_accepted_tokens << "\n"
        << "# HELP ninfer_metal_phys_footprint_bytes Process physical memory "
           "footprint in bytes.\n"
        << "# TYPE ninfer_metal_phys_footprint_bytes gauge\n"
        << "ninfer_metal_phys_footprint_bytes " << phys_bytes << "\n"
        << "# HELP ninfer_metal_loaded_models Number of loaded models hosted "
           "on this server.\n"
        << "# TYPE ninfer_metal_loaded_models gauge\n"
        << "ninfer_metal_loaded_models " << per_model.size() << "\n"
        << "# HELP ninfer_metal_gpu_contended Whether 2+ models are actively "
           "contending for GPU.\n"
        << "# TYPE ninfer_metal_gpu_contended gauge\n"
        << "ninfer_metal_gpu_contended " << (host_->is_gpu_contended() ? 1 : 0)
        << "\n"
        << "# HELP ninfer_metal_memory_budget_bytes Configured server UMA "
           "memory budget in bytes.\n"
        << "# TYPE ninfer_metal_memory_budget_bytes gauge\n"
        << "ninfer_metal_memory_budget_bytes "
        << host_->total_memory_budget_bytes() << "\n";

    // Per-model labeled Prometheus metrics
    oss << "# HELP ninfer_metal_model_requests_total Total inference requests "
           "per model.\n"
        << "# TYPE ninfer_metal_model_requests_total counter\n";
    for (const auto &entry : per_model) {
      const std::string labels =
          "model=\"" + escape_prometheus_label(entry.model->id) +
          "\",provider=\"" +
          escape_prometheus_label(entry.model->provider_id) +
          "\",backend_mode=\"" +
          escape_prometheus_label(entry.model->backend_mode) + "\"";
      oss << "ninfer_metal_model_requests_total{" << labels << "} "
          << entry.snap.requests_total << "\n";
    }
    oss << "# HELP ninfer_metal_model_active_requests Current in-flight "
           "requests per model.\n"
        << "# TYPE ninfer_metal_model_active_requests gauge\n";
    for (const auto &entry : per_model) {
      const std::string labels =
          "model=\"" + escape_prometheus_label(entry.model->id) +
          "\",provider=\"" +
          escape_prometheus_label(entry.model->provider_id) +
          "\",backend_mode=\"" +
          escape_prometheus_label(entry.model->backend_mode) + "\"";
      oss << "ninfer_metal_model_active_requests{" << labels << "} "
          << entry.snap.active_requests << "\n";
    }
    oss << "# HELP ninfer_metal_model_generated_tokens_total Total generated "
           "tokens per model.\n"
        << "# TYPE ninfer_metal_model_generated_tokens_total counter\n";
    for (const auto &entry : per_model) {
      const std::string labels =
          "model=\"" + escape_prometheus_label(entry.model->id) +
          "\",provider=\"" +
          escape_prometheus_label(entry.model->provider_id) +
          "\",backend_mode=\"" +
          escape_prometheus_label(entry.model->backend_mode) + "\"";
      oss << "ninfer_metal_model_generated_tokens_total{" << labels << "} "
          << entry.snap.generated_tokens_total << "\n";
    }
    oss << "# HELP ninfer_metal_model_resident_memory_bytes Resident UMA "
           "footprint per model.\n"
        << "# TYPE ninfer_metal_model_resident_memory_bytes gauge\n";
    for (const auto &entry : per_model) {
      const std::string labels =
          "model=\"" + escape_prometheus_label(entry.model->id) +
          "\",provider=\"" +
          escape_prometheus_label(entry.model->provider_id) +
          "\",backend_mode=\"" +
          escape_prometheus_label(entry.model->backend_mode) + "\"";
      oss << "ninfer_metal_model_resident_memory_bytes{" << labels << "} "
          << entry.resident_bytes << "\n";
    }
    oss << "# HELP ninfer_metal_model_healthy Whether the model backend is "
           "healthy (1) or faulted (0).\n"
        << "# TYPE ninfer_metal_model_healthy gauge\n";
    for (const auto &entry : per_model) {
      const std::string labels =
          "model=\"" + escape_prometheus_label(entry.model->id) +
          "\",provider=\"" +
          escape_prometheus_label(entry.model->provider_id) +
          "\",backend_mode=\"" +
          escape_prometheus_label(entry.model->backend_mode) + "\"";
      oss << "ninfer_metal_model_healthy{" << labels << "} "
          << (entry.snap.backend_healthy ? 1 : 0) << "\n";
    }

    res.status = 200;
    res.set_content(oss.str(), "text/plain; version=0.0.4; charset=utf-8");
  }

  void handle_chat_completions(const httplib::Request &req,
                               httplib::Response &res) {
    Json body;
    try {
      body = Json::parse(req.body);
    } catch (const Json::exception &err) {
      write_openai_error(res, 400,
                         std::string("Invalid JSON body: ") + err.what(),
                         "invalid_request_error", "invalid_json");
      return;
    }
    if (!body.is_object()) {
      write_openai_error(res, 400, "Request body must be a JSON object",
                         "invalid_request_error");
      return;
    }
    const auto target = resolve_target_model(body, res);
    if (target == nullptr) {
      return;
    }
    if (!body.contains("messages") || !body.at("messages").is_array() ||
        body.at("messages").empty()) {
      write_openai_error(res, 400, "messages must be a non-empty JSON array",
                         "invalid_request_error", "missing_messages");
      return;
    }
    if (body.contains("n") && !body.at("n").is_null()) {
      if (!body.at("n").is_number_integer() ||
          body.at("n").get<std::int64_t>() != 1) {
        write_openai_error(res, 400, "Only n=1 is supported",
                           "invalid_request_error", "unsupported_n");
        return;
      }
    }

    bool stream = false;
    bool include_usage = false;
    SamplingSpec sampling;
    std::vector<std::string> stop_strings;
    ChatRenderResult rendered;
    std::vector<std::int32_t> prompt_tokens;
    std::int32_t max_new_tokens = 1;
    OutputParserOptions parser_opts;

    try {
      stream = parse_stream_flag(body);
      include_usage = parse_include_usage(body);
      sampling = parse_sampling_spec(body);
      stop_strings = parse_stop_strings(body);

      Json tools = Json();
      if (body.contains("tools") && !body.at("tools").is_null()) {
        if (!body.at("tools").is_array()) {
          throw std::invalid_argument("tools must be an array");
        }
        tools = body.at("tools");
      }

      Json extra_kwargs = Json::object();
      if (body.contains("chat_template_kwargs") &&
          !body.at("chat_template_kwargs").is_null()) {
        if (!body.at("chat_template_kwargs").is_object()) {
          throw std::invalid_argument("chat_template_kwargs must be an object");
        }
        extra_kwargs = body.at("chat_template_kwargs");
      }
      if (body.contains("enable_thinking") &&
          !body.at("enable_thinking").is_null()) {
        extra_kwargs["enable_thinking"] = body.at("enable_thinking");
      }
      if (body.contains("reasoning_effort") &&
          !body.at("reasoning_effort").is_null()) {
        extra_kwargs["reasoning_effort"] = body.at("reasoning_effort");
      }

      rendered = target->chat_template->render(body.at("messages"), tools,
                                               extra_kwargs, true);

      auto encoded = target->tokenizer->encode_with_boundaries(
          rendered.prompt_text, {}, {}, rendered.literal_spans);
      if (encoded.input_ids.empty()) {
        throw std::invalid_argument("rendered prompt produced zero tokens");
      }
      prompt_tokens.reserve(encoded.input_ids.size());
      for (const int id : encoded.input_ids) {
        prompt_tokens.push_back(static_cast<std::int32_t>(id));
      }

      max_new_tokens = parse_max_new_tokens(body, prompt_tokens.size(),
                                            target->max_context, 2048);

      parser_opts.enable_thinking_split = true;
      parser_opts.initial_in_thinking = rendered.prompt_opens_think;
      parser_opts.enable_tool_parsing = true;
      if (body.contains("tool_choice") && body.at("tool_choice").is_string() &&
          body.at("tool_choice").get<std::string>() == "none") {
        parser_opts.enable_tool_parsing = false;
      }
      parser_opts.stop_strings = std::move(stop_strings);
      parser_opts.tool_param_types = extract_tool_param_types(tools);
    } catch (const std::invalid_argument &err) {
      write_openai_error(res, 400, err.what(), "invalid_request_error");
      return;
    } catch (const std::exception &err) {
      write_openai_error(res, 400, err.what(), "invalid_request_error");
      return;
    }

    const std::uint64_t seq =
        next_completion_seq.fetch_add(1, std::memory_order_relaxed);
    const std::string completion_id = "chatcmpl-" + std::to_string(seq);
    const std::int64_t created = unix_timestamp_now();
    const std::size_t prompt_token_count = prompt_tokens.size();
    parser_opts.tool_call_id_prefix = "call_" + std::to_string(seq) + "_";

    SubmitRequest submit_req;
    submit_req.prompt_tokens = std::move(prompt_tokens);
    submit_req.max_new_tokens = max_new_tokens;
    submit_req.sampling = sampling;

    std::shared_ptr<RequestStream> req_stream;
    try {
      req_stream = std::make_shared<RequestStream>(
          target->router->submit(std::move(submit_req)));
    } catch (const std::invalid_argument &err) {
      write_openai_error(res, 400, err.what(), "invalid_request_error");
      return;
    } catch (const std::exception &err) {
      write_openai_error(res, 503, err.what(), "server_error", "overloaded");
      return;
    }

    const auto start_tp = std::chrono::steady_clock::now();

    if (!stream) {
      OutputParser parser(std::move(parser_opts));
      std::string reasoning_acc;
      std::string content_acc;
      std::vector<ParsedToolCall> tool_calls_acc;
      std::size_t completion_tokens = 0;
      std::string backend_finish_reason;
      std::string error_msg;
      bool failed = false;
      bool saw_first_token = false;
      auto first_token_tp = start_tp;

      while (auto ev = req_stream->next_event()) {
        if (ev->kind == EventKind::tokens) {
          if (!saw_first_token && !ev->tokens.empty()) {
            saw_first_token = true;
            first_token_tp = std::chrono::steady_clock::now();
          }
          for (const std::int32_t tok : ev->tokens) {
            ++completion_tokens;
            const auto bytes =
                safe_decode_token_bytes(*target->tokenizer, tok);
            if (bytes.empty()) {
              continue;
            }
            auto delta = parser.feed_bytes(bytes);
            reasoning_acc.append(delta.reasoning);
            content_acc.append(delta.content);
            for (auto &tc : delta.tool_calls) {
              tool_calls_acc.push_back(std::move(tc));
            }
            if (delta.stopped_by_string) {
              req_stream->cancel();
              break;
            }
          }
          if (parser.stopped_by_string()) {
            break;
          }
        } else {
          backend_finish_reason = ev->finish_reason;
          if (ev->kind == EventKind::failed) {
            failed = true;
            error_msg = ev->error.empty() ? "engine generation failed"
                                          : ev->error;
          }
          break;
        }
      }

      if (failed) {
        write_openai_error(res, 500, error_msg, "server_error");
        return;
      }

      {
        auto tail = parser.finish();
        reasoning_acc.append(tail.reasoning);
        content_acc.append(tail.content);
        for (auto &tc : tail.tool_calls) {
          tool_calls_acc.push_back(std::move(tc));
        }
      }

      const auto end_tp = std::chrono::steady_clock::now();
      const double prompt_ms =
          saw_first_token
              ? std::chrono::duration<double, std::milli>(first_token_tp -
                                                          start_tp)
                    .count()
              : std::chrono::duration<double, std::milli>(end_tp - start_tp)
                    .count();
      const double predicted_ms =
          saw_first_token
              ? std::chrono::duration<double, std::milli>(end_tp -
                                                          first_token_tp)
                    .count()
              : 0.0;
      const double predicted_tps =
          predicted_ms > 0.0
              ? (static_cast<double>(completion_tokens) * 1000.0 / predicted_ms)
              : 0.0;

      const std::string finish_reason = resolve_finish_reason(
          backend_finish_reason, parser.stopped_by_string(),
          tool_calls_acc.size());

      Json message = {
          {"role", "assistant"},
          {"content", (!content_acc.empty() || tool_calls_acc.empty())
                          ? Json(content_acc)
                          : Json(nullptr)},
      };
      if (!reasoning_acc.empty()) {
        message["reasoning_content"] = reasoning_acc;
      }
      if (!tool_calls_acc.empty()) {
        Json tc_arr = Json::array();
        for (const auto &tc : tool_calls_acc) {
          tc_arr.push_back(format_tool_call_object(tc));
        }
        message["tool_calls"] = std::move(tc_arr);
      }

      Json response = {
          {"id", completion_id},
          {"object", "chat.completion"},
          {"created", created},
          {"model", target->id},
          {"choices",
           Json::array({{
               {"index", 0},
               {"message", std::move(message)},
               {"finish_reason", finish_reason},
           }})},
          {"usage",
           {
               {"prompt_tokens", prompt_token_count},
               {"completion_tokens", completion_tokens},
               {"total_tokens", prompt_token_count + completion_tokens},
           }},
          {"timings",
           {
               {"prompt_ms", prompt_ms},
               {"predicted_ms", predicted_ms},
               {"predicted_per_second", predicted_tps},
           }},
      };
      write_json_response(res, 200, response);
      return;
    }

    // Streaming SSE response.
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_header("X-Accel-Buffering", "no");

    auto tokenizer = target->tokenizer;
    const std::string model_name = target->id;

    res.set_chunked_content_provider(
        "text/event-stream; charset=utf-8",
        [req_stream = std::move(req_stream),
         tokenizer = std::move(tokenizer),
         parser_opts = std::move(parser_opts), completion_id, created,
         model_name, prompt_token_count,
         include_usage](std::size_t /*offset*/, httplib::DataSink &sink) mutable {
          OutputParser parser(parser_opts);
          std::size_t completion_tokens = 0;
          std::string backend_finish_reason;
          std::string error_msg;
          bool failed = false;
          bool client_disconnected = false;

          const auto emit_delta = [&](const OutputDelta &delta) -> bool {
            if (!delta.reasoning.empty()) {
              Json chunk = {
                  {"id", completion_id},
                  {"object", "chat.completion.chunk"},
                  {"created", created},
                  {"model", model_name},
                  {"choices",
                   Json::array({{
                       {"index", 0},
                       {"delta", {{"reasoning_content", delta.reasoning}}},
                       {"finish_reason", nullptr},
                   }})},
              };
              if (!send_sse_json(sink, chunk)) {
                return false;
              }
            }
            if (!delta.content.empty()) {
              Json chunk = {
                  {"id", completion_id},
                  {"object", "chat.completion.chunk"},
                  {"created", created},
                  {"model", model_name},
                  {"choices",
                   Json::array({{
                       {"index", 0},
                       {"delta", {{"content", delta.content}}},
                       {"finish_reason", nullptr},
                   }})},
              };
              if (!send_sse_json(sink, chunk)) {
                return false;
              }
            }
            if (!delta.tool_calls.empty()) {
              Json tc_deltas = Json::array();
              for (const auto &tc : delta.tool_calls) {
                tc_deltas.push_back(format_tool_call_delta(tc));
              }
              Json chunk = {
                  {"id", completion_id},
                  {"object", "chat.completion.chunk"},
                  {"created", created},
                  {"model", model_name},
                  {"choices",
                   Json::array({{
                       {"index", 0},
                       {"delta", {{"tool_calls", std::move(tc_deltas)}}},
                       {"finish_reason", nullptr},
                   }})},
              };
              if (!send_sse_json(sink, chunk)) {
                return false;
              }
            }
            return true;
          };

          // Initial role delta per OpenAI SSE protocol.
          {
            Json init_chunk = {
                {"id", completion_id},
                {"object", "chat.completion.chunk"},
                {"created", created},
                {"model", model_name},
                {"choices",
                 Json::array({{
                     {"index", 0},
                     {"delta", {{"role", "assistant"}, {"content", ""}}},
                     {"finish_reason", nullptr},
                 }})},
            };
            if (!send_sse_json(sink, init_chunk)) {
              req_stream->cancel();
              return false;
            }
          }

          while (auto ev = req_stream->next_event()) {
            if (!sink.is_writable()) {
              client_disconnected = true;
              req_stream->cancel();
              break;
            }
            if (ev->kind == EventKind::tokens) {
              for (const std::int32_t tok : ev->tokens) {
                ++completion_tokens;
                const auto bytes = safe_decode_token_bytes(*tokenizer, tok);
                if (bytes.empty()) {
                  continue;
                }
                auto delta = parser.feed_bytes(bytes);
                if (!delta.empty() && !emit_delta(delta)) {
                  client_disconnected = true;
                  req_stream->cancel();
                  break;
                }
                if (delta.stopped_by_string) {
                  req_stream->cancel();
                  break;
                }
              }
              if (client_disconnected || parser.stopped_by_string()) {
                break;
              }
            } else {
              backend_finish_reason = ev->finish_reason;
              if (ev->kind == EventKind::failed) {
                failed = true;
                error_msg = ev->error.empty() ? "engine generation failed"
                                              : ev->error;
              }
              break;
            }
          }

          if (!client_disconnected && failed) {
            (void)send_sse_json(
                sink, make_openai_error_json(error_msg, "server_error"));
          } else if (!client_disconnected) {
            auto tail = parser.finish();
            if (!tail.empty()) {
              (void)emit_delta(tail);
            }
            const std::string finish_reason = resolve_finish_reason(
                backend_finish_reason, parser.stopped_by_string(),
                parser.tool_call_count());

            Json final_chunk = {
                {"id", completion_id},
                {"object", "chat.completion.chunk"},
                {"created", created},
                {"model", model_name},
                {"choices",
                 Json::array({{
                     {"index", 0},
                     {"delta", Json::object()},
                     {"finish_reason", finish_reason},
                 }})},
            };
            (void)send_sse_json(sink, final_chunk);

            if (include_usage) {
              Json usage_chunk = {
                  {"id", completion_id},
                  {"object", "chat.completion.chunk"},
                  {"created", created},
                  {"model", model_name},
                  {"choices", Json::array()},
                  {"usage",
                   {
                       {"prompt_tokens", prompt_token_count},
                       {"completion_tokens", completion_tokens},
                       {"total_tokens", prompt_token_count + completion_tokens},
                   }},
              };
              (void)send_sse_json(sink, usage_chunk);
            }
            (void)send_sse_done(sink);
          }

          if (client_disconnected) {
            return false;
          }
          sink.done();
          return true;
        });
  }

  void handle_completions(const httplib::Request &req, httplib::Response &res) {
    Json body;
    try {
      body = Json::parse(req.body);
    } catch (const Json::exception &err) {
      write_openai_error(res, 400,
                         std::string("Invalid JSON body: ") + err.what(),
                         "invalid_request_error", "invalid_json");
      return;
    }
    if (!body.is_object()) {
      write_openai_error(res, 400, "Request body must be a JSON object",
                         "invalid_request_error");
      return;
    }
    const auto target = resolve_target_model(body, res);
    if (target == nullptr) {
      return;
    }
    if (!body.contains("prompt") || body.at("prompt").is_null()) {
      write_openai_error(res, 400, "prompt is required",
                         "invalid_request_error", "missing_prompt");
      return;
    }

    bool stream = false;
    bool include_usage = false;
    bool echo = false;
    SamplingSpec sampling;
    std::vector<std::string> stop_strings;
    std::string prompt_text;
    std::vector<std::int32_t> prompt_tokens;
    std::int32_t max_new_tokens = 1;

    try {
      stream = parse_stream_flag(body);
      include_usage = parse_include_usage(body);
      if (body.contains("echo") && !body.at("echo").is_null()) {
        if (!body.at("echo").is_boolean()) {
          throw std::invalid_argument("echo must be a boolean");
        }
        echo = body.at("echo").get<bool>();
      }
      sampling = parse_sampling_spec(body);
      stop_strings = parse_stop_strings(body);

      const auto &prompt_val = body.at("prompt");
      if (prompt_val.is_string()) {
        prompt_text = prompt_val.get<std::string>();
        const auto encoded = target->tokenizer->encode(prompt_text);
        prompt_tokens.assign(encoded.begin(), encoded.end());
      } else if (prompt_val.is_array() && !prompt_val.empty()) {
        if (prompt_val[0].is_string()) {
          if (prompt_val.size() != 1) {
            throw std::invalid_argument(
                "only a single string prompt is supported");
          }
          prompt_text = prompt_val[0].get<std::string>();
          const auto encoded = target->tokenizer->encode(prompt_text);
          prompt_tokens.assign(encoded.begin(), encoded.end());
        } else if (prompt_val[0].is_number_integer()) {
          prompt_tokens.reserve(prompt_val.size());
          for (const auto &el : prompt_val) {
            if (!el.is_number_integer()) {
              throw std::invalid_argument(
                  "prompt token array must contain integers");
            }
            const auto id = el.get<std::int32_t>();
            if (!target->tokenizer->is_valid_token(static_cast<int>(id))) {
              throw std::invalid_argument("prompt token id out of range");
            }
            prompt_tokens.push_back(id);
          }
        } else {
          throw std::invalid_argument(
              "prompt must be a string, [string], or [int]");
        }
      } else {
        throw std::invalid_argument("prompt must be non-empty");
      }

      if (prompt_tokens.empty()) {
        throw std::invalid_argument("prompt produced zero tokens");
      }

      max_new_tokens = parse_max_new_tokens(body, prompt_tokens.size(),
                                            target->max_context, 256);
    } catch (const std::invalid_argument &err) {
      write_openai_error(res, 400, err.what(), "invalid_request_error");
      return;
    } catch (const std::exception &err) {
      write_openai_error(res, 400, err.what(), "invalid_request_error");
      return;
    }

    const std::uint64_t seq =
        next_completion_seq.fetch_add(1, std::memory_order_relaxed);
    const std::string completion_id = "cmpl-" + std::to_string(seq);
    const std::int64_t created = unix_timestamp_now();
    const std::size_t prompt_token_count = prompt_tokens.size();

    SubmitRequest submit_req;
    submit_req.prompt_tokens = std::move(prompt_tokens);
    submit_req.max_new_tokens = max_new_tokens;
    submit_req.sampling = sampling;

    std::shared_ptr<RequestStream> req_stream;
    try {
      req_stream = std::make_shared<RequestStream>(
          target->router->submit(std::move(submit_req)));
    } catch (const std::invalid_argument &err) {
      write_openai_error(res, 400, err.what(), "invalid_request_error");
      return;
    } catch (const std::exception &err) {
      write_openai_error(res, 503, err.what(), "server_error", "overloaded");
      return;
    }

    OutputParserOptions parser_opts;
    parser_opts.enable_thinking_split = false;
    parser_opts.enable_tool_parsing = false;
    parser_opts.stop_strings = std::move(stop_strings);

    if (!stream) {
      OutputParser parser(std::move(parser_opts));
      std::string text_acc = echo ? prompt_text : std::string{};
      std::size_t completion_tokens = 0;
      std::string backend_finish_reason;
      bool failed = false;
      std::string error_msg;

      while (auto ev = req_stream->next_event()) {
        if (ev->kind == EventKind::tokens) {
          for (const std::int32_t tok : ev->tokens) {
            ++completion_tokens;
            const auto bytes =
                safe_decode_token_bytes(*target->tokenizer, tok);
            if (bytes.empty()) {
              continue;
            }
            auto delta = parser.feed_bytes(bytes);
            text_acc.append(delta.content);
            if (delta.stopped_by_string) {
              req_stream->cancel();
              break;
            }
          }
          if (parser.stopped_by_string()) {
            break;
          }
        } else {
          backend_finish_reason = ev->finish_reason;
          if (ev->kind == EventKind::failed) {
            failed = true;
            error_msg = ev->error.empty() ? "engine generation failed"
                                          : ev->error;
          }
          break;
        }
      }

      if (failed) {
        write_openai_error(res, 500, error_msg, "server_error");
        return;
      }

      text_acc.append(parser.finish().content);
      const std::string finish_reason = resolve_finish_reason(
          backend_finish_reason, parser.stopped_by_string(), 0);

      Json response = {
          {"id", completion_id},
          {"object", "text_completion"},
          {"created", created},
          {"model", target->id},
          {"choices",
           Json::array({{
               {"index", 0},
               {"text", std::move(text_acc)},
               {"logprobs", nullptr},
               {"finish_reason", finish_reason},
           }})},
          {"usage",
           {
               {"prompt_tokens", prompt_token_count},
               {"completion_tokens", completion_tokens},
               {"total_tokens", prompt_token_count + completion_tokens},
           }},
      };
      write_json_response(res, 200, response);
      return;
    }

    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_header("X-Accel-Buffering", "no");

    auto tokenizer = target->tokenizer;
    const std::string model_name = target->id;

    res.set_chunked_content_provider(
        "text/event-stream; charset=utf-8",
        [req_stream = std::move(req_stream),
         tokenizer = std::move(tokenizer),
         parser_opts = std::move(parser_opts), completion_id, created,
         model_name, prompt_token_count, include_usage, echo,
         prompt_text](std::size_t /*offset*/, httplib::DataSink &sink) mutable {
          OutputParser parser(parser_opts);
          std::size_t completion_tokens = 0;
          std::string backend_finish_reason;
          std::string error_msg;
          bool failed = false;
          bool client_disconnected = false;

          const auto emit_text = [&](std::string_view text) -> bool {
            if (text.empty()) {
              return true;
            }
            Json chunk = {
                {"id", completion_id},
                {"object", "text_completion"},
                {"created", created},
                {"model", model_name},
                {"choices",
                 Json::array({{
                     {"index", 0},
                     {"text", std::string(text)},
                     {"logprobs", nullptr},
                     {"finish_reason", nullptr},
                 }})},
            };
            return send_sse_json(sink, chunk);
          };

          if (echo && !prompt_text.empty() && !emit_text(prompt_text)) {
            req_stream->cancel();
            return false;
          }

          while (auto ev = req_stream->next_event()) {
            if (!sink.is_writable()) {
              client_disconnected = true;
              req_stream->cancel();
              break;
            }
            if (ev->kind == EventKind::tokens) {
              for (const std::int32_t tok : ev->tokens) {
                ++completion_tokens;
                const auto bytes = safe_decode_token_bytes(*tokenizer, tok);
                if (bytes.empty()) {
                  continue;
                }
                auto delta = parser.feed_bytes(bytes);
                if (!emit_text(delta.content)) {
                  client_disconnected = true;
                  req_stream->cancel();
                  break;
                }
                if (delta.stopped_by_string) {
                  req_stream->cancel();
                  break;
                }
              }
              if (client_disconnected || parser.stopped_by_string()) {
                break;
              }
            } else {
              backend_finish_reason = ev->finish_reason;
              if (ev->kind == EventKind::failed) {
                failed = true;
                error_msg = ev->error.empty() ? "engine generation failed"
                                              : ev->error;
              }
              break;
            }
          }

          if (!client_disconnected && failed) {
            (void)send_sse_json(
                sink, make_openai_error_json(error_msg, "server_error"));
          } else if (!client_disconnected) {
            auto tail = parser.finish();
            (void)emit_text(tail.content);
            const std::string finish_reason = resolve_finish_reason(
                backend_finish_reason, parser.stopped_by_string(), 0);

            Json final_chunk = {
                {"id", completion_id},
                {"object", "text_completion"},
                {"created", created},
                {"model", model_name},
                {"choices",
                 Json::array({{
                     {"index", 0},
                     {"text", ""},
                     {"logprobs", nullptr},
                     {"finish_reason", finish_reason},
                 }})},
            };
            (void)send_sse_json(sink, final_chunk);

            if (include_usage) {
              Json usage_chunk = {
                  {"id", completion_id},
                  {"object", "text_completion"},
                  {"created", created},
                  {"model", model_name},
                  {"choices", Json::array()},
                  {"usage",
                   {
                       {"prompt_tokens", prompt_token_count},
                       {"completion_tokens", completion_tokens},
                       {"total_tokens", prompt_token_count + completion_tokens},
                   }},
              };
              (void)send_sse_json(sink, usage_chunk);
            }
            (void)send_sse_done(sink);
          }

          if (client_disconnected) {
            return false;
          }
          sink.done();
          return true;
        });
  }
};

ServeHttpServer::ServeHttpServer(ServeConfig config,
                                 std::shared_ptr<EngineRouter> router,
                                 std::shared_ptr<Tokenizer> tokenizer,
                                 std::shared_ptr<ChatTemplate> chat_template) {
  auto host = std::make_shared<MultiModelHost>(
      resolve_total_memory_budget_bytes(config));
  ModelInstance inst;
  inst.id = resolve_model_id(config);
  inst.provider_id = router != nullptr ? router->provider_id() : "io.ninfer.metal";
  inst.backend_mode = router != nullptr ? router->backend_mode() : "inproc";
  inst.router = std::move(router);
  inst.tokenizer = std::move(tokenizer);
  inst.chat_template = std::move(chat_template);
  inst.max_context = config.max_context;
  inst.max_concurrency = config.max_concurrency;
  inst.lora_adapter = config.lora_adapter;
  inst.lora_mode = config.lora_mode;
  host->add_model(std::move(inst));
  impl_ = std::make_unique<Impl>(std::move(config), std::move(host));
}

ServeHttpServer::ServeHttpServer(ServeConfig config,
                                 std::shared_ptr<MultiModelHost> host)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(host))) {}

ServeHttpServer::~ServeHttpServer() { stop(); }

int ServeHttpServer::bind() {
  if (impl_->config.port == 0) {
    const int port = impl_->server.bind_to_any_port(impl_->config.host);
    if (port <= 0) {
      throw std::runtime_error("failed to bind to ephemeral port on " +
                               impl_->config.host);
    }
    impl_->bound_port.store(port, std::memory_order_release);
    return port;
  }
  if (!impl_->server.bind_to_port(impl_->config.host, impl_->config.port)) {
    throw std::runtime_error("failed to bind HTTP server to " +
                             impl_->config.host + ":" +
                             std::to_string(impl_->config.port));
  }
  impl_->bound_port.store(impl_->config.port, std::memory_order_release);
  return impl_->config.port;
}

void ServeHttpServer::listen_after_bind() { impl_->server.listen_after_bind(); }

void ServeHttpServer::stop() {
  if (impl_ != nullptr) {
    impl_->server.stop();
  }
}

int ServeHttpServer::bound_port() const noexcept {
  return impl_->bound_port.load(std::memory_order_acquire);
}

const std::string &ServeHttpServer::model_id() const noexcept {
  return impl_->primary_model_id;
}

} // namespace omlx::m5::serve
