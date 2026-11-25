#include "omlx_m5/engine.h"
#include "serve/chat_template.h"
#include "serve/config.h"
#include "serve/engine_router.h"
#include "serve/http_server.h"
#include "serve/multi_model_host.h"
#include "serve/output_parser.h"
#include "serve/provider_registry.h"
#include "serve/serve_backend.h"
#include "serve/tokenizer.h"
#include "serve/uds_sidecar_backend.h"
#include "serve/unicode.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <netinet/in.h>
#include <poll.h>
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
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace omlx::m5;
using namespace omlx::m5::serve;

void check(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error("TEST FAILURE: " + message);
  }
}

std::string byte_to_gpt2_utf8(int byte) {
  static const auto table = [] {
    std::array<std::string, 256> out;
    std::uint32_t next = 256;
    for (int b = 0; b < 256; ++b) {
      const bool visible =
          (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
      const std::uint32_t cp = visible ? static_cast<std::uint32_t>(b) : next++;
      out[static_cast<std::size_t>(b)] =
          unicode_internal::codepoint_to_utf8(static_cast<std::int32_t>(cp));
    }
    return out;
  }();
  return table[static_cast<std::size_t>(byte)];
}

std::string encode_gpt2_bytes(std::string_view raw) {
  std::string out;
  for (unsigned char c : raw) {
    out += byte_to_gpt2_utf8(static_cast<int>(c));
  }
  return out;
}

std::string read_file_text(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("cannot open file: " + path.string());
  }
  std::ostringstream oss;
  oss << in.rdbuf();
  return oss.str();
}

struct SyntheticTestAssets {
  std::shared_ptr<Tokenizer> tokenizer;
  std::shared_ptr<ChatTemplate> chat_template;
  std::vector<std::int32_t> eos_ids;
  int vocab_size = 0;
  std::string tokenizer_json;
  std::string tokenizer_config_json;
  std::string template_source;
};

SyntheticTestAssets build_synthetic_assets() {
  Json vocab = Json::object();
  for (int b = 0; b < 256; ++b) {
    vocab[byte_to_gpt2_utf8(b)] = b;
  }

  // Crafted sequential token blocks that FakeBackend emits deterministically
  // starting from (prompt_tokens.back() + 1).
  auto add_vocab_piece = [&](int id, std::string_view raw_bytes) {
    vocab[encode_gpt2_bytes(raw_bytes)] = id;
  };

  // Scenario 1: plain text (trigger id = 300)
  add_vocab_piece(301, "Hello");
  add_vocab_piece(302, " ");
  add_vocab_piece(303, "world");
  add_vocab_piece(304, "!");

  // Scenario 2: <think> split across token boundaries (trigger id = 310)
  add_vocab_piece(311, "<th");
  add_vocab_piece(312, "ink>\nplan ");
  add_vocab_piece(313, "step 1</thi");
  add_vocab_piece(314, "nk>\n\nFinal ");
  add_vocab_piece(315, "answer.");

  // Scenario 3: <think> + Qwen XML <tool_call> (trigger id = 320)
  add_vocab_piece(321, "<think>check weather</think>\n\n<tool_");
  add_vocab_piece(322, "call>\n<function=get_weather>\n");
  add_vocab_piece(323, "<parameter=city>\nTokyo\n</parameter>\n");
  add_vocab_piece(324, "<parameter=days>\n3\n</parameter>\n");
  add_vocab_piece(325, "</function>\n</tool_call>");

  // Scenario 4: Qwen JSON <tool_call> (trigger id = 330)
  add_vocab_piece(331, "<tool_call>\n{\"name\": \"lookup_stock\", ");
  add_vocab_piece(
      332, "\"arguments\": {\"ticker\": \"AAPL\", \"limit\": 5}}\n</tool_call>");

  // Scenario 5: Stop string split across tokens (trigger id = 340)
  add_vocab_piece(341, "alpha ");
  add_vocab_piece(342, "beta ST");
  add_vocab_piece(343, "OP_HERE gamma");

  auto mk_added = [](int id, const char *content, bool special) {
    return Json{
        {"id", id},
        {"content", content},
        {"single_word", false},
        {"lstrip", false},
        {"rstrip", false},
        {"normalized", false},
        {"special", special},
    };
  };

  Json added = Json::array({
      mk_added(290, "<|im_start|>", true),
      mk_added(291, "<|im_end|>", true),
      mk_added(300, "<|trig_plain|>", true),
      mk_added(305, "<|eos_plain|>", true),
      mk_added(310, "<|trig_think|>", true),
      mk_added(316, "<|eos_think|>", true),
      mk_added(320, "<|trig_xml|>", true),
      mk_added(326, "<|eos_xml|>", true),
      mk_added(330, "<|trig_json|>", true),
      mk_added(333, "<|eos_json|>", true),
      mk_added(340, "<|trig_stop|>", true),
      mk_added(344, "<|eos_stop|>", true),
  });

  Json tok_json = {
      {"version", "1.0"},
      {"normalizer", {{"type", "NFC"}}},
      {"pre_tokenizer",
       {{"type", "Sequence"},
        {"pretokenizers",
         Json::array(
             {{{"type", "Split"},
               {"pattern",
                {{"Regex",
                  R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
               {"behavior", "Isolated"},
               {"invert", false}},
              {{"type", "ByteLevel"},
               {"add_prefix_space", false},
               {"trim_offsets", true},
               {"use_regex", false}}})}}},
      {"decoder", {{"type", "ByteLevel"}}},
      {"post_processor", nullptr},
      {"added_tokens", std::move(added)},
      {"model",
       {{"type", "BPE"},
        {"dropout", nullptr},
        {"unk_token", nullptr},
        {"continuing_subword_prefix", ""},
        {"end_of_word_suffix", ""},
        {"byte_fallback", false},
        {"ignore_merges", false},
        {"vocab", std::move(vocab)},
        {"merges", Json::array()}}},
  };

  const std::string tok_str = tok_json.dump();
  auto tokenizer = std::make_shared<Tokenizer>(TokenizerResources{
      .tokenizer_json = tok_str,
      .tokenizer_config_json = "{}",
  });

  const std::string tmpl_source =
      "{%- for m in messages -%}"
      "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
      "{%- endfor -%}"
      "{%- if add_generation_prompt -%}"
      "<|im_start|>assistant\n"
      "{%- if scenario is defined and scenario == 'think' -%}<|trig_think|>"
      "{%- elif scenario is defined and scenario == 'xml_tool' -%}<|trig_xml|>"
      "{%- elif scenario is defined and scenario == 'json_tool' -%}<|trig_json|>"
      "{%- elif scenario is defined and scenario == 'stop' -%}<|trig_stop|>"
      "{%- else -%}<|trig_plain|>"
      "{%- endif -%}"
      "{%- endif -%}";

  ChatTemplateMetadata meta;
  meta.source_name = "synthetic_test_template";
  meta.eos_token = "<|im_end|>";
  meta.eos_token_ids = {291, 305, 316, 326, 333, 344};
  meta.max_position_embeddings = 4096;

  auto chat_template = std::make_shared<ChatTemplate>(
      ChatTemplate::from_source(tmpl_source, meta.source_name, meta));

  return SyntheticTestAssets{
      .tokenizer = std::move(tokenizer),
      .chat_template = std::move(chat_template),
      .eos_ids = meta.eos_token_ids,
      .vocab_size = 360,
      .tokenizer_json = tok_str,
      .tokenizer_config_json = Json({{"eos_token", "<|im_end|>"}}).dump(),
      .template_source = tmpl_source,
  };
}

struct ParsedSse {
  std::vector<Json> chunks;
  bool saw_done = false;
};

ParsedSse parse_sse_body(const std::string &body) {
  ParsedSse result;
  std::istringstream iss(body);
  std::string line;
  while (std::getline(iss, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.starts_with("data: ")) {
      continue;
    }
    const std::string payload = line.substr(6);
    if (payload == "[DONE]") {
      result.saw_done = true;
      continue;
    }
    result.chunks.push_back(Json::parse(payload));
  }
  return result;
}

void test_output_parser_utf8_and_formats() {
  // 1) Byte-by-byte UTF-8 multibyte reconstruction (CJK + 4-byte emoji)
  {
    OutputParser parser({.enable_thinking_split = true,
                         .initial_in_thinking = false,
                         .enable_tool_parsing = true});
    const std::string sample = "<think>思考中</think>\n\n你好🚀";
    std::string reasoning;
    std::string content;
    for (char ch : sample) {
      auto d = parser.feed_bytes(std::string_view(&ch, 1));
      reasoning += d.reasoning;
      content += d.content;
    }
    auto tail = parser.finish();
    reasoning += tail.reasoning;
    content += tail.content;
    check(reasoning == "思考中",
          "UTF-8 byte-by-byte reasoning mismatch: " + reasoning);
    check(content == "你好🚀",
          "UTF-8 byte-by-byte content mismatch: " + content);
  }

  // 2) Initial in-thinking mode (Qwen3.5/3.8 template pre-appends <think>\n)
  {
    OutputParser parser({.enable_thinking_split = true,
                         .initial_in_thinking = true,
                         .enable_tool_parsing = true});
    auto d1 = parser.feed_bytes("inner thought</think>\n\nanswer");
    auto d2 = parser.finish();
    check(d1.reasoning + d2.reasoning == "inner thought",
          "initial_in_thinking reasoning mismatch");
    check(d1.content + d2.content == "answer",
          "initial_in_thinking content mismatch");
  }
}

void test_toml_and_cli_config() {
  const auto tmp_path = std::filesystem::temp_directory_path() /
                        "ninfer_metal_serve_test_config.toml";
  {
    std::ofstream out(tmp_path);
    out << "[server]\n"
        << "host = \"0.0.0.0\"\n"
        << "port = 9191\n"
        << "api_key = \"toml-secret\"\n"
        << "cors = false\n\n"
        << "[model]\n"
        << "model_dir = \"/tmp/model-a\"\n"
        << "model_id = \"custom-qwen\"\n\n"
        << "[engine]\n"
        << "max_concurrency = 4\n"
        << "max_context = 16384\n"
        << "kv_dtype = \"affine4_g128\"\n"
        << "rolling_prefix_cache = \"on\"\n"
        << "prefix_cache_gb = 2.5\n\n"
        << "[speculative]\n"
        << "backend = \"mtp\"\n"
        << "draft_tokens = 3\n";
  }

  const std::string cfg_str = tmp_path.string();
  std::vector<std::string> args = {
      "ninfer-metal-serve",
      "--config",
      cfg_str,
      "--port",
      "9292",
      "-b",
      "2",
      "--cors",
      "on",
      "--kv-dtype",
      "bf16",
      "--spec-backend",
      "dflash2",
      "--draft-tokens",
      "4",
      "--rolling-prefix-cache",
      "off",
      "--prefix-cache-gb",
      "1.5",
  };
  std::vector<char *> argv;
  argv.reserve(args.size());
  for (auto &a : args) {
    argv.push_back(a.data());
  }

  const auto cfg =
      parse_serve_config(static_cast<int>(argv.size()), argv.data());
  std::error_code ec;
  std::filesystem::remove(tmp_path, ec);

  check(cfg.host == "0.0.0.0", "TOML host not loaded");
  check(cfg.port == 9292, "CLI --port failed to override TOML");
  check(cfg.api_key == "toml-secret", "TOML api_key not loaded");
  check(cfg.cors == true, "CLI --cors on failed to override TOML");
  check(cfg.model_dir == "/tmp/model-a", "TOML model_dir not loaded");
  check(cfg.model_id == "custom-qwen", "TOML model_id not loaded");
  check(cfg.max_concurrency == 2, "CLI -b failed to override TOML");
  check(cfg.max_context == 16384, "TOML max_context not loaded");
  check(cfg.kv_dtype == "bf16", "CLI --kv-dtype failed to override TOML");
  check(cfg.spec_backend == SpeculativeBackend::dflash2,
        "CLI --spec-backend failed to override TOML");
  check(cfg.draft_tokens == 4, "CLI --draft-tokens failed to override TOML");
  check(cfg.rolling_prefix_cache.has_value() && !*cfg.rolling_prefix_cache,
        "CLI --rolling-prefix-cache failed to override TOML");
  check(cfg.prefix_cache_gb.has_value() && *cfg.prefix_cache_gb == 1.5,
        "CLI --prefix-cache-gb failed to override TOML");

  const auto load_opts = to_load_options(cfg);
  check(load_opts.path_overrides.at("rolling_prefix_cache") == "off",
        "to_load_options rolling_prefix_cache mismatch");
  check(load_opts.path_overrides.at("batch_mode") == "on",
        "to_load_options batch_mode should be on for B=2");
  check(!load_opts.path_overrides.at("prefix_cache_bytes").empty(),
        "to_load_options prefix_cache_bytes missing");
}

void test_http_server_e2e() {
  auto assets = build_synthetic_assets();

  ServeConfig config;
  config.host = "127.0.0.1";
  config.port = 0; // Ephemeral port
  config.api_key = "test-bearer-secret";
  config.cors = true;
  config.model_id = "qwen-e2e-fake";
  config.max_concurrency = 4;
  config.max_context = 4096;

  LoadOptions load_opts;
  load_opts.max_concurrency = 4;
  load_opts.max_context = 4096;

  FakeBackendOptions fake_opts;
  fake_opts.event_batch_size = 1;
  fake_opts.step_delay_us = 1500; // Interleave B=4 concurrent requests cleanly
  fake_opts.vocab_size = assets.vocab_size;

  std::shared_ptr<Engine> engine(Engine::create_fake(load_opts, fake_opts));
  auto router = std::make_shared<EngineRouter>(engine, assets.eos_ids);
  ServeHttpServer server(config, router, assets.tokenizer, assets.chat_template);
  const int port = server.bind();
  check(port > 0, "ephemeral bind failed");

  std::thread server_thread([&] { server.listen_after_bind(); });

  const auto cleanup = [&] {
    server.stop();
    if (server_thread.joinable()) {
      server_thread.join();
    }
    router->stop();
  };

  try {
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(15, 0);
    const httplib::Headers auth_headers = {
        {"Authorization", "Bearer test-bearer-secret"},
    };

    // 1) GET /health
    {
      auto res = client.Get("/health");
      check(res && res->status == 200, "/health should return 200");
      auto body = Json::parse(res->body);
      check(body.at("status") == "ok", "/health status must be ok");
      check(body.at("model") == "qwen-e2e-fake", "/health model mismatch");
      check(body.at("version") == NINFER_METAL_DISPLAY_VERSION,
            "/health version mismatch");
      check(res->get_header_value("Access-Control-Allow-Origin") == "*",
            "CORS header missing on /health");
    }

    // 2) CORS OPTIONS preflight (unauthenticated)
    {
      auto res = client.Options("/v1/chat/completions");
      check(res && res->status == 204, "OPTIONS preflight must return 204");
      check(res->get_header_value("Access-Control-Allow-Origin") == "*",
            "CORS origin header missing on OPTIONS");
    }

    // 3) Auth 401 checks
    {
      auto res = client.Get("/v1/models");
      check(res && res->status == 401, "Missing Bearer key must return 401");
      auto res2 = client.Get(
          "/v1/models",
          httplib::Headers{{"Authorization", "Bearer wrong-key"}});
      check(res2 && res2->status == 401, "Wrong Bearer key must return 401");
    }

    // 4) GET /v1/models & /v1/models/:id + 404
    {
      auto res = client.Get("/v1/models", auth_headers);
      check(res && res->status == 200, "/v1/models must return 200");
      auto body = Json::parse(res->body);
      check(body.at("object") == "list", "/v1/models object != list");
      check(body.at("data").size() == 1, "/v1/models data size != 1");
      check(body.at("data")[0].at("id") == "qwen-e2e-fake",
            "/v1/models id mismatch");
      check(body.at("data")[0].at("max_model_len") == 4096,
            "/v1/models max_model_len mismatch");

      auto res_id = client.Get("/v1/models/qwen-e2e-fake", auth_headers);
      check(res_id && res_id->status == 200, "/v1/models/:id must return 200");
      check(Json::parse(res_id->body).at("max_model_len") == 4096,
            "/v1/models/:id max_model_len mismatch");

      auto res_404 = client.Get("/v1/models/does-not-exist", auth_headers);
      check(res_404 && res_404->status == 404,
            "unknown model id must return 404");

      auto res_route_404 = client.Get("/v1/unknown_endpoint", auth_headers);
      check(res_route_404 && res_route_404->status == 404,
            "unknown endpoint must return 404");
    }

    // 5) 400 Bad Request validations
    {
      auto bad_json = client.Post("/v1/chat/completions", auth_headers,
                                  "{not-json", "application/json");
      check(bad_json && bad_json->status == 400,
            "invalid JSON must return 400");

      auto no_msgs = client.Post("/v1/chat/completions", auth_headers,
                                 Json({{"model", "qwen-e2e-fake"}}).dump(),
                                 "application/json");
      check(no_msgs && no_msgs->status == 400,
            "missing messages must return 400");

      auto neg_tokens = client.Post(
          "/v1/chat/completions", auth_headers,
          Json({{"model", "qwen-e2e-fake"},
                {"messages", {{{"role", "user"}, {"content", "hi"}}}},
                {"max_tokens", -5}})
              .dump(),
          "application/json");
      check(neg_tokens && neg_tokens->status == 400,
            "negative max_tokens must return 400");
    }

    // 6) POST /v1/completions sync + SSE + usage + [DONE]
    {
      Json req = {
          {"model", "qwen-e2e-fake"},
          {"prompt", Json::array({300})},
          {"max_tokens", 16},
      };
      auto res = client.Post("/v1/completions", auth_headers, req.dump(),
                             "application/json");
      check(res && res->status == 200, "/v1/completions sync must return 200");
      auto body = Json::parse(res->body);
      check(body.at("object") == "text_completion",
            "/v1/completions object mismatch");
      check(body.at("choices")[0].at("text") == "Hello world!",
            "/v1/completions text mismatch: " +
                body.at("choices")[0].at("text").get<std::string>());
      check(body.at("choices")[0].at("finish_reason") == "stop",
            "/v1/completions finish_reason must be stop");
      check(body.at("usage").at("prompt_tokens") == 1 &&
                body.at("usage").at("completion_tokens") == 4,
            "/v1/completions usage mismatch");

      req["stream"] = true;
      req["stream_options"] = {{"include_usage", true}};
      auto sse_res = client.Post("/v1/completions", auth_headers, req.dump(),
                                 "application/json");
      check(sse_res && sse_res->status == 200,
            "/v1/completions SSE must return 200");
      auto sse = parse_sse_body(sse_res->body);
      check(sse.saw_done, "/v1/completions SSE missing [DONE]");
      std::string joined;
      bool saw_usage = false;
      for (const auto &c : sse.chunks) {
        if (!c.at("choices").empty()) {
          joined += c.at("choices")[0].value("text", "");
        }
        if (c.contains("usage") && !c.at("usage").is_null()) {
          saw_usage = true;
          check(c.at("usage").at("completion_tokens") == 4,
                "/v1/completions SSE usage completion_tokens mismatch");
        }
      }
      check(joined == "Hello world!",
            "/v1/completions SSE text mismatch: " + joined);
      check(saw_usage, "/v1/completions SSE missing usage chunk");
    }

    // 7) POST /v1/chat/completions sync & SSE: <think> split
    {
      Json req = {
          {"model", "qwen-e2e-fake"},
          {"messages", {{{"role", "user"}, {"content", "Think carefully"}}}},
          {"max_completion_tokens", 32},
          {"chat_template_kwargs", {{"scenario", "think"}}},
      };
      auto res = client.Post("/v1/chat/completions", auth_headers, req.dump(),
                             "application/json");
      check(res && res->status == 200, "chat sync think must return 200");
      auto body = Json::parse(res->body);
      const auto &msg = body.at("choices")[0].at("message");
      check(msg.at("reasoning_content") == "plan step 1",
            "chat sync reasoning_content mismatch");
      check(msg.at("content") == "Final answer.",
            "chat sync content mismatch");
      check(body.at("choices")[0].at("finish_reason") == "stop",
            "chat sync think finish_reason mismatch");
      check(body.at("usage").at("completion_tokens") == 5,
            "chat sync think completion_tokens mismatch");

      // Streaming SSE with include_usage
      req["stream"] = true;
      req["stream_options"] = {{"include_usage", true}};
      auto sse_res = client.Post("/v1/chat/completions", auth_headers,
                                 req.dump(), "application/json");
      check(sse_res && sse_res->status == 200,
            "chat SSE think must return 200");
      auto sse = parse_sse_body(sse_res->body);
      check(sse.saw_done, "chat SSE missing [DONE]");
      std::string reasoning;
      std::string content;
      std::string finish_reason;
      bool saw_usage = false;
      for (const auto &c : sse.chunks) {
        check(c.at("object") == "chat.completion.chunk",
              "SSE chunk object must be chat.completion.chunk");
        if (!c.at("choices").empty()) {
          const auto &delta = c.at("choices")[0].at("delta");
          if (delta.contains("reasoning_content")) {
            reasoning += delta.at("reasoning_content").get<std::string>();
          }
          if (delta.contains("content")) {
            content += delta.at("content").get<std::string>();
          }
          if (!c.at("choices")[0].at("finish_reason").is_null()) {
            finish_reason =
                c.at("choices")[0].at("finish_reason").get<std::string>();
          }
        }
        if (c.contains("usage") && !c.at("usage").is_null()) {
          saw_usage = true;
          check(c.at("usage").at("completion_tokens") == 5,
                "chat SSE usage completion_tokens mismatch");
        }
      }
      check(reasoning == "plan step 1",
            "chat SSE reasoning mismatch: " + reasoning);
      check(content == "Final answer.",
            "chat SSE content mismatch: " + content);
      check(finish_reason == "stop", "chat SSE finish_reason mismatch");
      check(saw_usage, "chat SSE missing usage chunk");
    }

    // 8) Qwen XML <tool_call> (sync + SSE)
    {
      Json tools = Json::array({{
          {"type", "function"},
          {"function",
           {{"name", "get_weather"},
            {"parameters",
             {{"type", "object"},
              {"properties",
               {{"city", {{"type", "string"}}},
                {"days", {{"type", "integer"}}}}}}}}},
      }});
      Json req = {
          {"model", "qwen-e2e-fake"},
          {"messages", {{{"role", "user"}, {"content", "Weather in Tokyo?"}}}},
          {"tools", tools},
          {"max_tokens", 32},
          {"chat_template_kwargs", {{"scenario", "xml_tool"}}},
      };
      auto res = client.Post("/v1/chat/completions", auth_headers, req.dump(),
                             "application/json");
      check(res && res->status == 200, "XML tool sync must return 200");
      auto body = Json::parse(res->body);
      const auto &choice = body.at("choices")[0];
      check(choice.at("finish_reason") == "tool_calls",
            "XML tool finish_reason must be tool_calls");
      check(choice.at("message").at("reasoning_content") == "check weather",
            "XML tool reasoning_content mismatch");
      const auto &tcs = choice.at("message").at("tool_calls");
      check(tcs.size() == 1, "XML tool_calls size != 1");
      check(tcs[0].at("function").at("name") == "get_weather",
            "XML tool name mismatch");
      auto parsed_args =
          Json::parse(tcs[0].at("function").at("arguments").get<std::string>());
      check(parsed_args.at("city") == "Tokyo" && parsed_args.at("days") == 3,
            "XML tool arguments mismatch: " + parsed_args.dump());

      req["stream"] = true;
      auto sse_res = client.Post("/v1/chat/completions", auth_headers,
                                 req.dump(), "application/json");
      check(sse_res && sse_res->status == 200, "XML tool SSE must return 200");
      auto sse = parse_sse_body(sse_res->body);
      check(sse.saw_done, "XML tool SSE missing [DONE]");
      bool saw_tool_delta = false;
      std::string finish_reason;
      for (const auto &c : sse.chunks) {
        if (c.at("choices").empty()) {
          continue;
        }
        const auto &delta = c.at("choices")[0].at("delta");
        if (delta.contains("tool_calls")) {
          saw_tool_delta = true;
          const auto &tc = delta.at("tool_calls")[0];
          check(tc.at("function").at("name") == "get_weather",
                "XML SSE tool name mismatch");
          auto a = Json::parse(
              tc.at("function").at("arguments").get<std::string>());
          check(a.at("city") == "Tokyo" && a.at("days") == 3,
                "XML SSE tool args mismatch");
        }
        if (!c.at("choices")[0].at("finish_reason").is_null()) {
          finish_reason =
              c.at("choices")[0].at("finish_reason").get<std::string>();
        }
      }
      check(saw_tool_delta, "XML tool SSE missing tool_calls delta");
      check(finish_reason == "tool_calls",
            "XML tool SSE finish_reason must be tool_calls");
    }

    // 9) Qwen JSON <tool_call> (sync + SSE)
    {
      Json req = {
          {"model", "qwen-e2e-fake"},
          {"messages", {{{"role", "user"}, {"content", "Lookup AAPL"}}}},
          {"max_tokens", 32},
          {"chat_template_kwargs", {{"scenario", "json_tool"}}},
      };
      auto res = client.Post("/v1/chat/completions", auth_headers, req.dump(),
                             "application/json");
      check(res && res->status == 200, "JSON tool sync must return 200");
      auto body = Json::parse(res->body);
      const auto &choice = body.at("choices")[0];
      check(choice.at("finish_reason") == "tool_calls",
            "JSON tool finish_reason must be tool_calls");
      const auto &tc = choice.at("message").at("tool_calls")[0];
      check(tc.at("function").at("name") == "lookup_stock",
            "JSON tool name mismatch");
      auto parsed_args =
          Json::parse(tc.at("function").at("arguments").get<std::string>());
      check(parsed_args.at("ticker") == "AAPL" && parsed_args.at("limit") == 5,
            "JSON tool args mismatch");
    }

    // 10) Stop-string holdback across token boundary
    {
      Json req = {
          {"model", "qwen-e2e-fake"},
          {"messages", {{{"role", "user"}, {"content", "Stop test"}}}},
          {"stop", Json::array({"STOP_HERE"})},
          {"max_tokens", 32},
          {"chat_template_kwargs", {{"scenario", "stop"}}},
      };
      auto res = client.Post("/v1/chat/completions", auth_headers, req.dump(),
                             "application/json");
      check(res && res->status == 200, "stop test must return 200");
      auto body = Json::parse(res->body);
      check(body.at("choices")[0].at("message").at("content") == "alpha beta ",
            "stop-string holdback content mismatch: " +
                body.at("choices")[0].at("message").at("content").dump());
      check(body.at("choices")[0].at("finish_reason") == "stop",
            "stop-string finish_reason must be stop");
    }

    // 11) B=4 concurrent demuxing across 4 simultaneous HTTP clients
    {
      const std::array<std::string, 4> scenarios = {"plain", "think",
                                                    "xml_tool", "json_tool"};
      std::array<std::string, 4> errors;
      std::vector<std::thread> workers;
      workers.reserve(4);
      for (std::size_t i = 0; i < 4; ++i) {
        workers.emplace_back([&, i] {
          try {
            httplib::Client c("127.0.0.1", port);
            c.set_read_timeout(15, 0);
            Json req = {
                {"model", "qwen-e2e-fake"},
                {"messages",
                 {{{"role", "user"},
                   {"content", "concurrent req " + std::to_string(i)}}}},
                {"max_tokens", 32},
                {"chat_template_kwargs", {{"scenario", scenarios[i]}}},
            };
            auto r = c.Post("/v1/chat/completions", auth_headers, req.dump(),
                            "application/json");
            check(r && r->status == 200, "B=4 request failed");
            auto b = Json::parse(r->body);
            const auto &msg = b.at("choices")[0].at("message");
            if (scenarios[i] == "plain") {
              check(msg.at("content") == "Hello world!",
                    "B=4 plain content mismatch");
            } else if (scenarios[i] == "think") {
              check(msg.at("reasoning_content") == "plan step 1" &&
                        msg.at("content") == "Final answer.",
                    "B=4 think mismatch");
            } else if (scenarios[i] == "xml_tool") {
              check(msg.at("tool_calls")[0].at("function").at("name") ==
                        "get_weather",
                    "B=4 xml_tool mismatch");
            } else if (scenarios[i] == "json_tool") {
              check(msg.at("tool_calls")[0].at("function").at("name") ==
                        "lookup_stock",
                    "B=4 json_tool mismatch");
            }
          } catch (const std::exception &ex) {
            errors[i] = ex.what();
          }
        });
      }
      for (auto &w : workers) {
        w.join();
      }
      for (std::size_t i = 0; i < 4; ++i) {
        check(errors[i].empty(),
              "B=4 worker " + std::to_string(i) + " failed: " + errors[i]);
      }
    }

    // 12) GET /metrics Prometheus exposition
    {
      auto res = client.Get("/metrics");
      check(res && res->status == 200, "/metrics must return 200");
      const auto &text = res->body;
      check(text.find("ninfer_metal_requests_total ") != std::string::npos,
            "/metrics missing ninfer_metal_requests_total");
      check(text.find("ninfer_metal_active_requests ") != std::string::npos,
            "/metrics missing ninfer_metal_active_requests");
      check(text.find("ninfer_metal_generated_tokens_total ") !=
                std::string::npos,
            "/metrics missing ninfer_metal_generated_tokens_total");
      check(text.find("ninfer_metal_max_concurrency 4") != std::string::npos,
            "/metrics missing ninfer_metal_max_concurrency 4");
      check(text.find("ninfer_metal_prefix_cache_hits_total ") !=
                std::string::npos,
            "/metrics missing prefix cache metrics");
    }
  } catch (...) {
    cleanup();
    throw;
  }
  cleanup();
}

void test_real_local_qwen_models(const std::filesystem::path &models_root) {
  if (!std::filesystem::is_directory(models_root)) {
    std::cout << "[serve_e2e] models/ not present at " << models_root.string()
              << "; skipping real-model tokenizer/template round-trip.\n";
    return;
  }

  const std::array<std::filesystem::path, 2> candidates = {
      models_root / "mlx-community" / "Qwen3.8-27B-4bit",
      models_root / "Jundot" / "Qwen3.8-27B-oQ4e-mtp",
  };

  int tested = 0;
  for (const auto &model_dir : candidates) {
    if (!std::filesystem::is_regular_file(model_dir / "tokenizer.json")) {
      continue;
    }
    const std::string tok_json = read_file_text(model_dir / "tokenizer.json");
    const std::string tok_cfg =
        std::filesystem::is_regular_file(model_dir / "tokenizer_config.json")
            ? read_file_text(model_dir / "tokenizer_config.json")
            : std::string{};

    Tokenizer tok(TokenizerResources{
        .tokenizer_json = tok_json,
        .tokenizer_config_json = tok_cfg,
    });
    check(tok.vocab_size() > 150000,
          "unexpected Qwen vocab size for " + model_dir.string());

    // Round-trip English + CJK + numbers + punctuation
    const std::string sample = "Hello Qwen3.8! 2026年Apple M5 Ultra推理测试: 12345.";
    const auto ids = tok.encode(sample);
    check(!ids.empty(), "real Qwen encode returned empty");
    const std::string decoded = tok.decode(ids);
    check(decoded == sample, "real Qwen round-trip mismatch for " +
                                 model_dir.string() + ": got '" + decoded + "'");

    // Verify sibling chat_template.jinja + EOS extraction + rendering
    auto ct = ChatTemplate::load(model_dir, tok);
    check(!ct.metadata().eos_token_ids.empty(),
          "real Qwen EOS token IDs should be non-empty for " +
              model_dir.string());
    check(ct.metadata().source_name.find("chat_template.jinja") !=
              std::string::npos,
          "expected sibling chat_template.jinja to be resolved for " +
              model_dir.string());

    Json messages = Json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", "What is 2+2?"}},
    });
    Json tools = Json::array({{
        {"type", "function"},
        {"function",
         {{"name", "calc"},
          {"description", "Add numbers"},
          {"parameters",
           {{"type", "object"},
            {"properties", {{"expr", {{"type", "string"}}}}}}}}},
    }});

    auto rendered_think = ct.render(messages, tools, {{"enable_thinking", true}});
    check(rendered_think.prompt_text.find("<|im_start|>assistant\n") !=
              std::string::npos,
          "rendered prompt missing assistant turn header");
    check(rendered_think.prompt_text.find("<tools>") != std::string::npos,
          "rendered prompt missing <tools> section");
    check(rendered_think.prompt_opens_think,
          "expected prompt_opens_think=true when enable_thinking=true");

    auto rendered_nothink =
        ct.render(messages, Json(), {{"enable_thinking", false}});
    check(!rendered_nothink.prompt_opens_think,
          "expected prompt_opens_think=false when enable_thinking=false");

    auto enc = tok.encode_with_boundaries(rendered_think.prompt_text, {}, {},
                                          rendered_think.literal_spans);
    check(enc.input_ids.size() > 20, "real Qwen template encode too short");
    std::cout << "[serve_e2e] Real Qwen model verified: " << model_dir.string()
              << " (vocab=" << tok.vocab_size()
              << ", eos_ids=" << ct.metadata().eos_token_ids.size()
              << ", prompt_tokens=" << enc.input_ids.size() << ")\n";
    ++tested;
  }
  check(tested > 0, "models/ directory exists but no Qwen tokenizer was tested");
}

void test_multi_model_config_and_aliases(
    const std::filesystem::path &models_root) {
  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       ("ninfer_serve_cfg_test_" +
                        std::to_string(static_cast<std::uint64_t>(::getpid())));
  std::filesystem::create_directories(tmp_dir);
  const auto toml_path = tmp_dir / "multi.toml";
  {
    std::ofstream out(toml_path);
    out << "[server]\n"
        << "host = \"127.0.0.1\"\n"
        << "port = 8181\n"
        << "providers_dir = [\"/tmp/prov-a\", \"/tmp/prov-b\"]\n"
        << "memory_budget_gb = 32.0\n\n"
        << "[model]\n"
        << "model_dir = \"/tmp/legacy-model\"\n"
        << "artifact_path = \"/tmp/legacy-model/target.omlxm5\"\n"
        << "draft_artifact_path = \"/tmp/legacy-model/draft.omlxm5\"\n\n"
        << "[engine]\n"
        << "max_concurrency = 2\n"
        << "max_context = 16384\n"
        << "kv_dtype = \"affine4_g128\"\n"
        << "metallib_dir = \"/tmp/shaders-toml\"\n\n"
        << "[[models]]\n"
        << "id = \"qwen-main\"\n"
        << "model_dir = \"/tmp/models/qwen\"\n"
        << "provider = \"inproc\"\n"
        << "artifact_path = \"/tmp/models/qwen/qwen.omlxm5\"\n"
        << "draft_artifact_path = \"/tmp/models/qwen/draft.omlxm5\"\n"
        << "max_concurrency = 4\n"
        << "rolling_prefix_cache = true\n"
        << "prefix_cache_gb = 4.0\n"
        << "memory_budget_gb = 20.0\n\n"
        << "[[models]]\n"
        << "id = \"llama-sidecar\"\n"
        << "model_dir = \"/tmp/models/llama\"\n"
        << "provider = \"auto\"\n"
        << "artifact = \"/tmp/models/llama/model.mockbin\"\n"
        << "max_context = 8192\n"
        << "kv_dtype = \"bf16\"\n"
        << "memory_budget_gb = 10.0\n";
  }

  std::string cfg_str = toml_path.string();
  std::vector<std::string> args = {
      "ninfer-metal-serve",
      "--config",
      cfg_str,
      "--providers-dir",
      "/tmp/prov-cli",
      "--metallib-dir",
      "/tmp/shaders-cli",
      "--memory-budget-gb",
      "40.0",
  };
  std::vector<char *> argv;
  for (auto &a : args) {
    argv.push_back(a.data());
  }
  const auto cfg =
      parse_serve_config(static_cast<int>(argv.size()), argv.data());

  check(cfg.artifact == "/tmp/legacy-model/target.omlxm5",
        "[model].artifact_path alias not parsed");
  check(cfg.draft_artifact == "/tmp/legacy-model/draft.omlxm5",
        "[model].draft_artifact_path alias not parsed");
  check(cfg.kernel_dir == "/tmp/shaders-cli",
        "--metallib-dir failed to override TOML metallib_dir");
  check(cfg.providers_dirs.size() == 3 &&
            cfg.providers_dirs[0] == "/tmp/prov-cli" &&
            cfg.providers_dirs[1] == "/tmp/prov-a" &&
            cfg.providers_dirs[2] == "/tmp/prov-b",
        "providers_dir precedence/merging mismatch");
  check(resolve_total_memory_budget_bytes(cfg) == (40ULL << 30),
        "--memory-budget-gb conversion mismatch");

  const auto resolved = resolve_model_configs(cfg);
  check(resolved.size() == 2, "expected 2 [[models]] entries");
  check(resolved[0].id == "qwen-main" && resolved[0].provider == "inproc" &&
            resolved[0].artifact == "/tmp/models/qwen/qwen.omlxm5" &&
            resolved[0].draft_artifact == "/tmp/models/qwen/draft.omlxm5" &&
            resolved[0].max_concurrency == 4 &&
            resolved[0].kernel_dir == "/tmp/shaders-cli" &&
            resolved[0].memory_budget_bytes == (20ULL << 30),
        "first [[models]] entry mismatch");
  check(resolved[1].id == "llama-sidecar" && resolved[1].provider == "auto" &&
            resolved[1].artifact == "/tmp/models/llama/model.mockbin" &&
            resolved[1].max_context == 8192 && resolved[1].kv_dtype == "bf16" &&
            resolved[1].memory_budget_bytes == (10ULL << 30),
        "second [[models]] entry mismatch");

  // Repeatable --model-dir CLI & --kernel-dir alias
  std::vector<std::string> cli_args = {
      "ninfer-metal-serve",
      "--model-dir",
      "/tmp/m1",
      "--model-dir",
      "/tmp/m2",
      "--kernel-dir",
      "/tmp/legacy-shaders",
      "--provider",
      "auto",
  };
  std::vector<char *> cli_argv;
  for (auto &a : cli_args) {
    cli_argv.push_back(a.data());
  }
  const auto cli_cfg =
      parse_serve_config(static_cast<int>(cli_argv.size()), cli_argv.data());
  check(cli_cfg.kernel_dir == "/tmp/legacy-shaders",
        "--kernel-dir alias not accepted");
  const auto cli_models = resolve_model_configs(cli_cfg);
  check(cli_models.size() == 2 && cli_models[0].id == "m1" &&
            cli_models[1].id == "m2",
        "repeatable --model-dir resolution mismatch");

  const std::string help = format_help_string();
  check(help.find("--metallib-dir") != std::string::npos &&
            help.find("--providers-dir") != std::string::npos &&
            help.find("--kernel-dir") != std::string::npos &&
            help.find("--memory-budget-gb") != std::string::npos,
        "--help text missing required multi-microkernel flags");

  const auto example_toml =
      std::filesystem::path(models_root).parent_path() / "omlx-extend" /
      "config" / "ninfer-metal.toml.example";
  if (std::filesystem::is_regular_file(example_toml)) {
    ServeConfig example_cfg;
    load_serve_toml_file(example_toml.string(), example_cfg);
    check(example_cfg.host == "127.0.0.1" && example_cfg.port == 8080 &&
              example_cfg.kv_dtype == "affine4_g128",
          "ninfer-metal.toml.example failed validation");
  }

  std::error_code ec;
  std::filesystem::remove_all(tmp_dir, ec);
}

void test_multi_microkernel_parallel_e2e(
    const std::filesystem::path &mock_provider_bin) {
  check(!mock_provider_bin.empty() &&
            std::filesystem::is_regular_file(mock_provider_bin),
        "mock SPI provider binary missing: " + mock_provider_bin.string());

  auto assets = build_synthetic_assets();

  const auto tmp_root = std::filesystem::temp_directory_path() /
                        ("ninfer_multi_spi_e2e_" +
                         std::to_string(static_cast<std::uint64_t>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(tmp_root, ec);
  std::filesystem::create_directories(tmp_root);

  struct DirCleanup {
    std::filesystem::path dir;
    ~DirCleanup() {
      std::error_code cleanup_ec;
      std::filesystem::remove_all(dir, cleanup_ec);
    }
  } dir_cleanup{tmp_root};

  // 1) Set up Providers/ directory with:
  //    - io.test.mock-microkernel/current/provider.json (executable Path B SPI v2)
  //    - io.ninfer.metal/current/provider.json (built-in, must stay inproc)
  const auto providers_root = tmp_root / "Providers";
  const auto mock_prov_dir =
      providers_root / "io.test.mock-microkernel" / "current";
  std::filesystem::create_directories(mock_prov_dir / "bin");
  const auto installed_bin = mock_prov_dir / "bin" / "mock_spi_provider";
  std::filesystem::copy_file(mock_provider_bin, installed_bin,
                             std::filesystem::copy_options::overwrite_existing);
  ::chmod(installed_bin.c_str(), 0755);

  {
    std::ofstream out(mock_prov_dir / "provider.json");
    out << Json({
               {"schema", "ninfer-metal-provider-v1"},
               {"provider_id", "io.test.mock-microkernel"},
               {"engine_id", "mock-llama"},
               {"display_name", "Mock Llama Microkernel"},
               {"display_version", "mock-v0.01"},
               {"protocol_version", 2},
               {"core_abi", 1},
               {"entrypoint", "bin/mock_spi_provider"},
               {"supported_topologies", Json::array({"llama_mock_v1"})},
               {"supported_model_types", Json::array({"llama", "llama3"})},
               {"artifact_extensions", Json::array({".mockbin"})},
               {"kv_formats", Json::array({"bf16"})},
           })
               .dump(2);
  }

  // Also stage a built-in io.ninfer.metal manifest to ensure no self-hijack.
  const auto builtin_prov_dir = providers_root / "io.ninfer.metal" / "current";
  std::filesystem::create_directories(builtin_prov_dir / "bin");
  std::filesystem::copy_file(mock_provider_bin,
                             builtin_prov_dir / "bin" / "mseries_backendd",
                             std::filesystem::copy_options::overwrite_existing);
  ::chmod((builtin_prov_dir / "bin" / "mseries_backendd").c_str(), 0755);
  {
    std::ofstream out(builtin_prov_dir / "provider.json");
    out << Json({
               {"provider_id", "io.ninfer.metal"},
               {"engine_id", "m5rt"},
               {"protocol_version", 2},
               {"entrypoint", "bin/mseries_backendd"},
           })
               .dump(2);
  }

  // 2) Stage Model A (qwen3_8 -> inproc miv) and Model B (llama -> sidecar)
  const auto model_a_dir = tmp_root / "models" / "Qwen3.8-Test";
  const auto model_b_dir = tmp_root / "models" / "Llama-Mock-Test";
  std::filesystem::create_directories(model_a_dir);
  std::filesystem::create_directories(model_b_dir);

  {
    std::ofstream out(model_a_dir / "config.json");
    out << Json({{"model_type", "qwen3_8"}, {"max_position_embeddings", 4096}})
               .dump(2);
  }
  {
    std::ofstream out(model_a_dir / "model.omlxm5");
    out << "dummy-qwen-artifact";
  }

  {
    std::ofstream out(model_b_dir / "config.json");
    out << Json({{"model_type", "llama"},
                 {"max_position_embeddings", 4096},
                 {"eos_token_id", assets.eos_ids}})
               .dump(2);
  }
  {
    std::ofstream out(model_b_dir / "model_pack.json");
    out << Json({{"topology_id", "llama_mock_v1"},
                 {"artifact_path", "model.mockbin"}})
               .dump(2);
  }
  {
    std::ofstream out(model_b_dir / "model.mockbin");
    out << "dummy-llama-artifact";
  }
  {
    std::ofstream out(model_b_dir / "tokenizer.json");
    out << assets.tokenizer_json;
  }
  {
    std::ofstream out(model_b_dir / "tokenizer_config.json");
    out << assets.tokenizer_config_json;
  }
  {
    std::ofstream out(model_b_dir / "chat_template.jinja");
    out << assets.template_source;
  }

  // 3) Verify ProviderRegistry discovery & O(1) routing without hijack
  const auto registry =
      ProviderRegistry::discover({providers_root.string()}, false);
  check(registry.providers().size() == 1,
        "expected exactly 1 external Path B provider, got " +
            std::to_string(registry.providers().size()));
  const auto *mock_prov = registry.find("io.test.mock-microkernel");
  check(mock_prov != nullptr, "mock provider not discovered by provider_id");
  check(registry.find("mock-llama") != nullptr,
        "mock provider not found by engine_id");
  check(mock_prov->protocol_version == 2 && mock_prov->core_abi == 1 &&
            mock_prov->supported_topologies.size() == 1 &&
            mock_prov->supported_topologies[0] == "llama_mock_v1" &&
            mock_prov->supported_model_types.size() == 2 &&
            mock_prov->artifact_extensions.size() == 1 &&
            mock_prov->artifact_extensions[0] == ".mockbin",
        "mock provider.json fields parsed incorrectly");

  ModelServeConfig cfg_a;
  cfg_a.id = "qwen-inproc";
  cfg_a.model_dir = model_a_dir.string();
  cfg_a.provider = "auto";
  cfg_a.max_concurrency = 2;
  cfg_a.max_context = 4096;

  ModelServeConfig cfg_b;
  cfg_b.id = "llama-sidecar";
  cfg_b.model_dir = model_b_dir.string();
  cfg_b.provider = "auto";
  cfg_b.max_concurrency = 2;
  cfg_b.max_context = 4096;
  cfg_b.rolling_prefix_cache = true;
  cfg_b.prefix_cache_gb = 1.5;

  const auto res_a = resolve_provider(cfg_a, registry);
  check(!res_a.use_sidecar && res_a.provider_id == "io.ninfer.metal" &&
            res_a.backend_mode == "inproc",
        "Model A (qwen3_8) was hijacked or misrouted: " + res_a.provider_id);

  const auto res_b = resolve_provider(cfg_b, registry);
  check(res_b.use_sidecar && res_b.provider_id == "io.test.mock-microkernel" &&
            res_b.backend_mode == "sidecar" &&
            res_b.resolved_artifact_path.ends_with("model.mockbin"),
        "Model B (llama) failed auto-routing to io.test.mock-microkernel");

  // Explicit override checks
  ModelServeConfig cfg_b_forced_inproc = cfg_b;
  cfg_b_forced_inproc.provider = "inproc";
  check(!resolve_provider(cfg_b_forced_inproc, registry).use_sidecar,
        "explicit provider=inproc must override auto routing");

  ModelServeConfig cfg_a_forced_sidecar = cfg_a;
  cfg_a_forced_sidecar.provider = "mock-llama";
  check(resolve_provider(cfg_a_forced_sidecar, registry).use_sidecar,
        "explicit engine_id must route to sidecar");

  // 4) Host Model A (in-process LocalEngineBackend) AND Model B (live spawned
  //    UdsSidecarBackend) concurrently on ONE HTTP port!
  const std::uint64_t total_uma_budget = 16ULL << 30; // 16 GiB cap
  auto host = std::make_shared<MultiModelHost>(total_uma_budget);

  // Add Model A as in-process FakeBackend with 25ms step delay for overlap
  LoadOptions fake_load_opts;
  fake_load_opts.max_concurrency = 2;
  fake_load_opts.max_context = 4096;
  FakeBackendOptions fake_opts;
  fake_opts.event_batch_size = 1;
  fake_opts.step_delay_us = 25'000;
  fake_opts.vocab_size = assets.vocab_size;

  std::shared_ptr<Engine> engine_a(
      Engine::create_fake(fake_load_opts, fake_opts));
  auto backend_a = std::make_shared<LocalEngineBackend>(
      engine_a, 2ULL << 30, "io.ninfer.metal");
  auto router_a = std::make_shared<EngineRouter>(backend_a, assets.eos_ids);
  host->add_model(ModelInstance{
      .id = "qwen-inproc",
      .provider_id = "io.ninfer.metal",
      .backend_mode = "inproc",
      .router = router_a,
      .tokenizer = assets.tokenizer,
      .chat_template = assets.chat_template,
      .max_context = 4096,
      .max_concurrency = 2,
      .allocated_budget_bytes = 2ULL << 30,
  });

  // Load Model B via MultiModelHost::load_model (spawns live UdsSidecarBackend)
  auto model_b = host->load_model(cfg_b, registry);
  check(model_b->id == "llama-sidecar" &&
            model_b->provider_id == "io.test.mock-microkernel" &&
            model_b->backend_mode == "sidecar",
        "Model B loaded metadata mismatch");
  check(model_b->allocated_budget_bytes == (14ULL << 30),
        "expected Model B UMA budget to subtract Model A's 2 GiB footprint "
        "(expected 14 GiB, got " +
            std::to_string(model_b->allocated_budget_bytes) + ")");

  auto *sidecar_ptr =
      dynamic_cast<UdsSidecarBackend *>(model_b->router->backend().get());
  check(sidecar_ptr != nullptr, "Model B backend must be UdsSidecarBackend");
  const pid_t sidecar_pid = sidecar_ptr->child_pid();
  const std::filesystem::path sidecar_socket = sidecar_ptr->socket_path();
  check(sidecar_pid > 0, "sidecar child_pid must be positive");
  check(std::filesystem::exists(sidecar_socket),
        "sidecar socket should exist while running");
  // MIVR-7: socket must live inside a private 0700 mkdtemp directory, not
  // directly in world-writable /tmp.
  {
    const auto sock_dir = sidecar_socket.parent_path();
    check(sock_dir != std::filesystem::path("/tmp"),
          "sidecar socket must not live directly in /tmp: " +
              sidecar_socket.string());
    struct stat dir_st {};
    check(::stat(sock_dir.c_str(), &dir_st) == 0 &&
              (dir_st.st_mode & 0777) == 0700,
          "sidecar socket directory must have 0700 permissions: " +
              sock_dir.string());
  }

  ServeConfig http_cfg;
  http_cfg.host = "127.0.0.1";
  http_cfg.port = 0;
  http_cfg.cors = true;
  ServeHttpServer server(http_cfg, host);
  const int port = server.bind();
  check(port > 0, "multi-model HTTP ephemeral bind failed");
  std::thread server_thread([&] { server.listen_after_bind(); });

  const auto cleanup = [&] {
    server.stop();
    if (server_thread.joinable()) {
      server_thread.join();
    }
    host->stop_all();
  };

  try {
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(15, 0);

    // 4a) GET /v1/models lists both models with provider_id and backend_mode
    {
      auto res = client.Get("/v1/models");
      check(res && res->status == 200, "multi /v1/models must return 200");
      auto body = Json::parse(res->body);
      check(body.at("data").size() == 2, "multi /v1/models must list 2 models");
      const auto &m0 = body.at("data")[0];
      const auto &m1 = body.at("data")[1];
      check(m0.at("id") == "qwen-inproc" &&
                m0.at("provider_id") == "io.ninfer.metal" &&
                m0.at("backend_mode") == "inproc",
            "Model A /v1/models entry mismatch: " + m0.dump());
      check(m1.at("id") == "llama-sidecar" &&
                m1.at("provider_id") == "io.test.mock-microkernel" &&
                m1.at("backend_mode") == "sidecar",
            "Model B /v1/models entry mismatch: " + m1.dump());

      auto res_b_id = client.Get("/v1/models/llama-sidecar");
      check(res_b_id && res_b_id->status == 200 &&
                Json::parse(res_b_id->body).at("backend_mode") == "sidecar",
            "/v1/models/llama-sidecar lookup failed");
    }

    // 4b) GET /health lists both models
    {
      auto res = client.Get("/health");
      check(res && res->status == 200, "multi /health must return 200");
      auto body = Json::parse(res->body);
      check(body.at("status") == "ok", "multi /health status != ok");
      check(body.at("models").size() == 2,
            "multi /health models array must have 2 entries");
      check(body.at("models")[0].at("backend_mode") == "inproc" &&
                body.at("models")[1].at("backend_mode") == "sidecar",
            "multi /health backend_mode mismatch");
    }

    // 4c) Unknown model -> 404, missing model in multi-model mode -> 400
    {
      auto r404 = client.Post(
          "/v1/chat/completions",
          Json({{"model", "no-such-model"},
                {"messages", {{{"role", "user"}, {"content", "hi"}}}}})
              .dump(),
          "application/json");
      check(r404 && r404->status == 404,
            "unknown model must return 404 in multi-model mode");
      auto err_body = Json::parse(r404->body);
      check(err_body.at("error").at("code") == "model_not_found",
            "unknown model error code mismatch");

      auto r400 = client.Post(
          "/v1/chat/completions",
          Json({{"messages", {{{"role", "user"}, {"content", "hi"}}}}}).dump(),
          "application/json");
      check(r400 && r400->status == 400,
            "omitting model in multi-model mode must return 400");
    }

    // 4d) Simultaneous sync + SSE requests to Model A (inproc) & Model B
    //     (sidecar) on ONE HTTP port, overlapping to trigger gpu_contended!
    {
      std::array<std::string, 4> errors;
      std::vector<std::thread> workers;
      workers.reserve(4);

      // Worker 0: Model A sync (think scenario, 5 tokens * 25ms = ~125ms)
      workers.emplace_back([&] {
        try {
          httplib::Client c("127.0.0.1", port);
          c.set_read_timeout(15, 0);
          Json req = {
              {"model", "qwen-inproc"},
              {"messages", {{{"role", "user"}, {"content", "Think A"}}}},
              {"max_tokens", 16},
              {"chat_template_kwargs", {{"scenario", "think"}}},
          };
          auto r = c.Post("/v1/chat/completions", req.dump(), "application/json");
          check(r && r->status == 200, "Model A sync failed");
          auto b = Json::parse(r->body);
          check(b.at("model") == "qwen-inproc", "Model A sync model mismatch");
          check(b.at("choices")[0].at("message").at("content") ==
                    "Final answer.",
                "Model A sync content mismatch");
        } catch (const std::exception &ex) {
          errors[0] = ex.what();
        }
      });

      // Worker 1: Model B sync (think scenario, 5 tokens * 20ms = ~100ms)
      workers.emplace_back([&] {
        try {
          httplib::Client c("127.0.0.1", port);
          c.set_read_timeout(15, 0);
          Json req = {
              {"model", "llama-sidecar"},
              {"messages", {{{"role", "user"}, {"content", "Think B"}}}},
              {"max_tokens", 16},
              {"chat_template_kwargs", {{"scenario", "think"}}},
          };
          auto r = c.Post("/v1/chat/completions", req.dump(), "application/json");
          check(r && r->status == 200, "Model B sync failed");
          auto b = Json::parse(r->body);
          check(b.at("model") == "llama-sidecar",
                "Model B sync model mismatch");
          check(b.at("choices")[0].at("message").at("reasoning_content") ==
                    "plan step 1" &&
                    b.at("choices")[0].at("message").at("content") ==
                        "Final answer.",
                "Model B sync output mismatch: " + b.dump());
        } catch (const std::exception &ex) {
          errors[1] = ex.what();
        }
      });

      // Worker 2: Model A SSE streaming (plain scenario)
      workers.emplace_back([&] {
        try {
          httplib::Client c("127.0.0.1", port);
          c.set_read_timeout(15, 0);
          Json req = {
              {"model", "qwen-inproc"},
              {"messages", {{{"role", "user"}, {"content", "Stream A"}}}},
              {"max_tokens", 16},
              {"stream", true},
          };
          auto r = c.Post("/v1/chat/completions", req.dump(), "application/json");
          check(r && r->status == 200, "Model A SSE failed");
          auto sse = parse_sse_body(r->body);
          check(sse.saw_done, "Model A SSE missing [DONE]");
          std::string acc;
          for (const auto &chunk : sse.chunks) {
            check(chunk.at("model") == "qwen-inproc",
                  "Model A SSE model field mismatch");
            if (!chunk.at("choices").empty()) {
              acc += chunk.at("choices")[0].at("delta").value("content", "");
            }
          }
          check(acc == "Hello world!", "Model A SSE content mismatch: " + acc);
        } catch (const std::exception &ex) {
          errors[2] = ex.what();
        }
      });

      // Worker 3: Model B SSE streaming (plain scenario)
      workers.emplace_back([&] {
        try {
          httplib::Client c("127.0.0.1", port);
          c.set_read_timeout(15, 0);
          Json req = {
              {"model", "llama-sidecar"},
              {"messages", {{{"role", "user"}, {"content", "Stream B"}}}},
              {"max_tokens", 16},
              {"stream", true},
              {"stream_options", {{"include_usage", true}}},
          };
          auto r = c.Post("/v1/chat/completions", req.dump(), "application/json");
          check(r && r->status == 200, "Model B SSE failed");
          auto sse = parse_sse_body(r->body);
          check(sse.saw_done, "Model B SSE missing [DONE]");
          std::string acc;
          bool saw_usage = false;
          for (const auto &chunk : sse.chunks) {
            check(chunk.at("model") == "llama-sidecar",
                  "Model B SSE model field mismatch");
            if (!chunk.at("choices").empty()) {
              acc += chunk.at("choices")[0].at("delta").value("content", "");
            }
            if (chunk.contains("usage") && !chunk.at("usage").is_null()) {
              saw_usage = true;
            }
          }
          check(acc == "Hello world!", "Model B SSE content mismatch: " + acc);
          check(saw_usage, "Model B SSE missing usage chunk");
        } catch (const std::exception &ex) {
          errors[3] = ex.what();
        }
      });

      for (auto &w : workers) {
        w.join();
      }
      for (std::size_t i = 0; i < errors.size(); ++i) {
        check(errors[i].empty(),
              "parallel multi-model worker " + std::to_string(i) +
                  " failed: " + errors[i]);
      }
    }

    // 4e) POST /v1/completions routed to Model B sidecar
    {
      Json req = {
          {"model", "llama-sidecar"},
          {"prompt", Json::array({300})},
          {"max_tokens", 16},
      };
      auto r = client.Post("/v1/completions", req.dump(), "application/json");
      check(r && r->status == 200, "Model B /v1/completions failed");
      auto b = Json::parse(r->body);
      check(b.at("model") == "llama-sidecar" &&
                b.at("choices")[0].at("text") == "Hello world!",
            "Model B /v1/completions response mismatch: " + b.dump());
    }

    // 4f) GET /metrics exposes aggregate + per-model labeled metrics
    {
      auto r = client.Get("/metrics");
      check(r && r->status == 200, "multi /metrics must return 200");
      check(r->body.find("ninfer_metal_loaded_models 2") != std::string::npos,
            "multi /metrics missing loaded_models=2");
      check(r->body.find(
                "model=\"qwen-inproc\",provider=\"io.ninfer.metal\","
                "backend_mode=\"inproc\"") != std::string::npos,
            "multi /metrics missing Model A labels");
      check(r->body.find(
                "model=\"llama-sidecar\",provider=\"io.test.mock-microkernel\","
                "backend_mode=\"sidecar\"") != std::string::npos,
            "multi /metrics missing Model B labels");
    }

    // 4g) Exercise trim_cache on sidecar backend
    {
      const auto trim_res = host->trim_model_cache("llama-sidecar", 8192);
      check(trim_res.freed_bytes > 0 && trim_res.evicted_entries >= 1,
            "trim_cache on sidecar returned zero freed bytes");
      const auto idle_freed = host->trim_idle_caches(4096);
      check(idle_freed > 0, "trim_idle_caches should reclaim from sidecar");
    }

    // Confirm contention cleared once all requests finished
    check(!host->is_gpu_contended(),
          "gpu_contended should be false after all concurrent requests finish");
    check(host->contention_transitions() >= 2,
          "expected at least 2 contention transitions (false->true->false), "
          "got " +
              std::to_string(host->contention_transitions()));
  } catch (...) {
    cleanup();
    throw;
  }

  cleanup();

  // 5) Verify sidecar audit log: memory_budget_bytes forwarded, gpu_contended
  //    transitioned true -> false, trim_cache called, shutdown received.
  const auto audit_path = model_b_dir / ".mock_spi_audit.json";
  check(std::filesystem::is_regular_file(audit_path),
        "mock SPI provider audit file missing: " + audit_path.string());
  const Json audit = Json::parse(read_file_text(audit_path));
  check(audit.at("memory_budget_bytes").get<std::uint64_t>() == (14ULL << 30),
        "sidecar load did not receive expected memory_budget_bytes: " +
            audit.dump());
  check(audit.at("artifact").get<std::string>().ends_with("model.mockbin"),
        "sidecar load did not receive resolved artifact path");
  check(audit.at("rolling_prefix_cache").get<bool>() == true,
        "sidecar load did not forward rolling_prefix_cache");
  check(audit.at("trim_cache_calls").get<std::uint64_t>() >= 2,
        "sidecar did not record trim_cache calls");
  check(audit.at("shutdown_received").get<bool>() == true,
        "sidecar did not record graceful shutdown RPC");

  const auto &hints = audit.at("scheduler_hint_history");
  check(hints.is_array() && hints.size() >= 2,
        "expected scheduler_hint_history [true, ..., false], got: " +
            hints.dump());
  bool saw_true = false;
  for (const auto &h : hints) {
    if (h.is_boolean() && h.get<bool>()) {
      saw_true = true;
    }
  }
  check(saw_true, "sidecar never received set_scheduler_hint(gpu_contended=true)");
  check(hints.back().is_boolean() && !hints.back().get<bool>(),
        "sidecar final scheduler hint must be gpu_contended=false, got: " +
            hints.dump());

  // 6) Verify zero orphan sidecar process and cleaned-up ephemeral socket!
  const int alive_rc = ::kill(sidecar_pid, 0);
  check(alive_rc != 0 && errno == ESRCH,
        "sidecar child process (pid=" + std::to_string(sidecar_pid) +
            ") still alive after shutdown!");
  check(!std::filesystem::exists(sidecar_socket),
        "ephemeral sidecar UDS socket was not unlinked on shutdown");
  check(!std::filesystem::exists(sidecar_socket.parent_path()),
        "ephemeral sidecar 0700 socket dir was not removed on shutdown");
}

void test_default_config_auto_discovery_tilde_and_auto_spec() {
  const auto tmp_root =
      std::filesystem::temp_directory_path() /
      ("ninfer_default_cfg_test_" +
       std::to_string(static_cast<std::uint64_t>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(tmp_root, ec);
  std::filesystem::create_directories(tmp_root);

  const std::filesystem::path orig_cwd = std::filesystem::current_path();
  const auto save_env = [](const char *name) -> std::optional<std::string> {
    if (const char *val = std::getenv(name)) {
      return std::string(val);
    }
    return std::nullopt;
  };
  const auto orig_home = save_env("HOME");
  const auto orig_cfg_env = save_env("NINFER_METAL_CONFIG");

  struct EnvGuard {
    std::filesystem::path cwd;
    std::filesystem::path tmp;
    std::optional<std::string> home;
    std::optional<std::string> cfg_env;
    ~EnvGuard() {
      std::error_code guard_ec;
      std::filesystem::current_path(cwd, guard_ec);
      if (home.has_value()) {
        ::setenv("HOME", home->c_str(), 1);
      } else {
        ::unsetenv("HOME");
      }
      if (cfg_env.has_value()) {
        ::setenv("NINFER_METAL_CONFIG", cfg_env->c_str(), 1);
      } else {
        ::unsetenv("NINFER_METAL_CONFIG");
      }
      std::filesystem::remove_all(tmp, guard_ec);
    }
  } guard{orig_cwd, tmp_root, orig_home, orig_cfg_env};

  const auto fake_home = tmp_root / "home";
  const auto work_dir = tmp_root / "work";
  std::filesystem::create_directories(fake_home);
  std::filesystem::create_directories(work_dir);
  ::setenv("HOME", fake_home.c_str(), 1);
  ::unsetenv("NINFER_METAL_CONFIG");
  std::filesystem::current_path(work_dir);

  auto parse_args = [](std::vector<std::string> args) {
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (auto &a : args) {
      argv.push_back(a.data());
    }
    return parse_serve_config(static_cast<int>(argv.size()), argv.data());
  };

  // 1) --no-default-config gives bare compiled defaults (no TOML loaded).
  {
    const auto bare =
        parse_args({"ninfer-metal-serve", "--no-default-config"});
    check(bare.no_default_config && !bare.config_auto_discovered &&
              bare.config_path.empty(),
          "--no-default-config should skip default config discovery");
    check(bare.max_concurrency == 1 && bare.max_context == 32768 &&
              bare.kv_dtype == "bf16" &&
              bare.spec_backend == SpeculativeBackend::none &&
              !bare.spec_backend_auto,
          "--no-default-config did not preserve bare compiled defaults");
  }

  // 2) Auto-discovery falls back to <repo_root>/omlx-extend/config/ninfer-metal.toml
  //    when no higher-priority config exists, and expands ~/ providers_dir!
  {
    const auto auto_cfg = parse_args({"ninfer-metal-serve"});
    check(auto_cfg.config_auto_discovered,
          "expected repo default ninfer-metal.toml to be auto-discovered");
    check(auto_cfg.config_path.ends_with("omlx-extend/config/ninfer-metal.toml"),
          "expected repo omlx-extend/config/ninfer-metal.toml, got: " +
              auto_cfg.config_path);
    check(auto_cfg.host == "127.0.0.1" && auto_cfg.port == 8080 &&
              auto_cfg.cors == true && auto_cfg.max_concurrency == 1 &&
              auto_cfg.max_context == 65536 &&
              auto_cfg.kv_dtype == "bf16" &&
              auto_cfg.rolling_prefix_cache.has_value() &&
              *auto_cfg.rolling_prefix_cache == true &&
              auto_cfg.prefix_cache_gb.has_value() &&
              *auto_cfg.prefix_cache_gb == 4.0 &&
              auto_cfg.sampling_contract == "position_coupled_v1" &&
              auto_cfg.lora_adapter.empty() &&
              auto_cfg.lora_mode == "merge" &&
              auto_cfg.spec_backend_auto == true && auto_cfg.draft_tokens == 0,
          "default omlx-extend/config/ninfer-metal.toml values mismatch");
    const std::string expected_providers =
        (fake_home / "Library/Application Support/oMLX/Providers").string();
    check(auto_cfg.providers_dirs.size() == 1 &&
              auto_cfg.providers_dirs[0] == expected_providers,
          "~/ expansion failed for default providers_dir: " +
              (auto_cfg.providers_dirs.empty() ? "<empty>"
                                               : auto_cfg.providers_dirs[0]));

    // CLI flags ALWAYS override auto-discovered TOML values!
    const auto overridden = parse_args({
        "ninfer-metal-serve",
        "--port",
        "9099",
        "--max-concurrency",
        "2",
        "--kv-dtype",
        "affine4_g128",
        "--spec-backend",
        "none",
        "--rolling-prefix-cache",
        "off",
    });
    check(overridden.config_auto_discovered && overridden.port == 9099 &&
              overridden.max_concurrency == 2 &&
              overridden.kv_dtype == "affine4_g128" &&
              overridden.spec_backend == SpeculativeBackend::none &&
              !overridden.spec_backend_auto &&
              overridden.rolling_prefix_cache.has_value() &&
              !*overridden.rolling_prefix_cache,
          "CLI flags failed to override auto-discovered default TOML");
  }

  // 3) Precedence chain: $NINFER_METAL_CONFIG > ./ninfer-metal.toml >
  //    ~/.config/ninfer-metal/ninfer-metal.toml > Providers/.../config/ninfer-metal.toml
  const auto prov_cfg =
      fake_home / "Library/Application Support/oMLX/Providers/io.ninfer.metal/"
                  "current/config/ninfer-metal.toml";
  const auto user_cfg = fake_home / ".config/ninfer-metal/ninfer-metal.toml";
  const auto cwd_cfg = work_dir / "ninfer-metal.toml";
  const auto env_cfg = tmp_root / "env-override.toml";

  std::filesystem::create_directories(prov_cfg.parent_path());
  {
    std::ofstream out(prov_cfg);
    out << "[server]\nport = 8404\n";
  }
  check(parse_args({"ninfer-metal-serve"}).port == 8404,
        "Providers/io.ninfer.metal/current/config/ninfer-metal.toml should beat repo default");

  std::filesystem::create_directories(user_cfg.parent_path());
  {
    std::ofstream out(user_cfg);
    out << "[server]\nport = 8303\n";
  }
  check(parse_args({"ninfer-metal-serve"}).port == 8303,
        "~/.config/ninfer-metal/ninfer-metal.toml should beat Providers config");

  {
    std::ofstream out(cwd_cfg);
    out << "[server]\nport = 8202\n";
  }
  check(parse_args({"ninfer-metal-serve"}).port == 8202,
        "./ninfer-metal.toml should beat ~/.config config");

  {
    std::ofstream out(env_cfg);
    out << "[server]\nport = 8101\n";
  }
  ::setenv("NINFER_METAL_CONFIG", env_cfg.c_str(), 1);
  check(parse_args({"ninfer-metal-serve"}).port == 8101,
        "$NINFER_METAL_CONFIG should beat ./ninfer-metal.toml");
  ::unsetenv("NINFER_METAL_CONFIG");

  // 4) Leading ~/ expansion across all path fields including --config.
  const auto tilde_cfg = fake_home / "tilde-test.toml";
  {
    std::ofstream out(tilde_cfg);
    out << "[server]\n"
        << "providers_dir = [\"~/prov-one\"]\n\n"
        << "[model]\n"
        << "model_dir = \"~/m/base\"\n"
        << "artifact = \"~/m/base/model.omlxm5\"\n"
        << "draft_dir = \"~/m/draft\"\n"
        << "draft_artifact = \"~/m/draft/model.omlxm5\"\n"
        << "mtp_artifact = \"~/m/base/mtp.omlxm5\"\n"
        << "metallib_dir = \"~/shaders\"\n"
        << "chat_template = \"~/templates/qwen.jinja\"\n";
  }
  const auto expanded =
      parse_args({"ninfer-metal-serve", "--config", "~/tilde-test.toml"});
  const std::string h = fake_home.string();
  check(expanded.config_path == h + "/tilde-test.toml" &&
            expanded.providers_dirs.size() == 1 &&
            expanded.providers_dirs[0] == h + "/prov-one" &&
            expanded.model_dir == h + "/m/base" &&
            expanded.artifact == h + "/m/base/model.omlxm5" &&
            expanded.draft_dir == h + "/m/draft" &&
            expanded.draft_artifact == h + "/m/draft/model.omlxm5" &&
            expanded.mtp_artifact == h + "/m/base/mtp.omlxm5" &&
            expanded.kernel_dir == h + "/shaders" &&
            expanded.chat_template == h + "/templates/qwen.jinja",
        "leading ~/ expansion missed one or more config fields");

  // 5) speculative.backend = "auto" resolution:
  //    dflash2/ companion -> dflash2; mtp.omlxm5 -> mtp; embedded MTP -> mtp;
  //    plain model -> none; dflash2/ takes precedence over mtp.omlxm5.
  const auto spec_root = tmp_root / "spec-models";
  const auto model_dflash = spec_root / "with-dflash";
  const auto model_mtp_ext = spec_root / "with-mtp-ext";
  const auto model_mtp_emb = spec_root / "with-mtp-emb";
  const auto model_plain = spec_root / "plain";
  std::filesystem::create_directories(model_dflash / "dflash2");
  std::filesystem::create_directories(model_mtp_ext);
  std::filesystem::create_directories(model_mtp_emb);
  std::filesystem::create_directories(model_plain);

  {
    std::ofstream(model_dflash / "dflash2" / "config.json") << "{}";
    std::ofstream(model_dflash / "dflash2" / "model.omlxm5") << "dummy";
    // Also place mtp.omlxm5 to verify dflash2/ wins over mtp.omlxm5!
    std::ofstream(model_dflash / "mtp.omlxm5") << "dummy";
  }
  {
    std::ofstream(model_mtp_ext / "mtp.omlxm5") << "dummy";
  }
  {
    std::ofstream(model_mtp_emb / "config.json")
        << R"({"text_config": {"mtp_num_hidden_layers": 1}})";
  }
  {
    std::ofstream(model_plain / "config.json")
        << R"({"text_config": {"hidden_size": 128}})";
  }

  auto resolve_for = [&](const std::filesystem::path &dir) {
    const auto cfg = parse_args({"ninfer-metal-serve", "--no-default-config",
                                 "--model-dir", dir.string(), "--spec-backend",
                                 "auto"});
    const auto models = resolve_model_configs(cfg);
    check(models.size() == 1, "expected 1 model config");
    return models.front();
  };

  const auto r_dflash = resolve_for(model_dflash);
  check(r_dflash.spec_backend == SpeculativeBackend::dflash2 &&
            !r_dflash.spec_backend_auto &&
            r_dflash.draft_dir == (model_dflash / "dflash2").string() &&
            r_dflash.draft_artifact ==
                (model_dflash / "dflash2" / "model.omlxm5").string(),
        "auto spec failed to resolve dflash2/ companion artifact");

  const auto r_mtp_ext = resolve_for(model_mtp_ext);
  check(r_mtp_ext.spec_backend == SpeculativeBackend::mtp &&
            !r_mtp_ext.spec_backend_auto &&
            r_mtp_ext.mtp_artifact == (model_mtp_ext / "mtp.omlxm5").string(),
        "auto spec failed to resolve mtp.omlxm5");

  const auto r_mtp_emb = resolve_for(model_mtp_emb);
  check(r_mtp_emb.spec_backend == SpeculativeBackend::mtp &&
            !r_mtp_emb.spec_backend_auto,
        "auto spec failed to resolve embedded MTP from config.json");

  const auto r_plain = resolve_for(model_plain);
  check(r_plain.spec_backend == SpeculativeBackend::none &&
            !r_plain.spec_backend_auto,
        "auto spec should fall back to none for plain model");

  const auto help = format_help_string();
  check(help.find("--no-default-config") != std::string::npos &&
            help.find("auto|none|mtp|dflash2") != std::string::npos,
        "--help text missing --no-default-config or auto spec backend");
}

class FailingServeBackend final : public ServeBackend {
public:
  void configure_next(std::vector<std::int32_t> tokens_before_fail,
                      std::string error_message) {
    std::lock_guard lock(mutex_);
    next_tokens_ = std::move(tokens_before_fail);
    next_error_ = std::move(error_message);
  }

  std::uint64_t submit(SubmitRequest /*request*/) override {
    std::lock_guard lock(mutex_);
    const std::uint64_t id = next_id_++;
    if (!next_tokens_.empty()) {
      events_.push_back(TokenEvent{
          .request_id = id,
          .tokens = next_tokens_,
          .kind = EventKind::tokens,
      });
    }
    events_.push_back(TokenEvent{
        .request_id = id,
        .kind = EventKind::failed,
        .finish_reason = "",
        .error = next_error_,
    });
    cv_.notify_all();
    return id;
  }

  bool cancel(std::uint64_t /*request_id*/) override { return true; }

  std::vector<TokenEvent>
  poll_events(std::chrono::microseconds max_wait) override {
    std::unique_lock lock(mutex_);
    cv_.wait_for(lock, max_wait,
                 [this] { return !events_.empty() || stopped_; });
    std::vector<TokenEvent> out;
    while (!events_.empty()) {
      out.push_back(std::move(events_.front()));
      events_.pop_front();
    }
    return out;
  }

  CapacityReport capacity() const override {
    return CapacityReport{
        .max_concurrency = 4,
        .accepting = true,
    };
  }

  StatsReport stats() const override { return {}; }
  TrimCacheReport trim_cache(std::uint64_t /*target_free_bytes*/) override {
    return {};
  }
  void set_scheduler_hint(const SchedulerHint & /*hint*/) override {}
  std::uint64_t resident_memory_bytes() const override { return 0; }
  void shutdown() override {
    std::lock_guard lock(mutex_);
    stopped_ = true;
    cv_.notify_all();
  }
  std::string provider_id() const override { return "test.failing"; }
  std::string backend_mode() const override { return "inproc"; }

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<TokenEvent> events_;
  std::vector<std::int32_t> next_tokens_;
  std::string next_error_;
  std::uint64_t next_id_ = 1;
  bool stopped_ = false;
};

void test_streaming_backend_failure_sse() {
  auto assets = build_synthetic_assets();
  auto failing_backend = std::make_shared<FailingServeBackend>();
  auto router = std::make_shared<EngineRouter>(failing_backend, assets.eos_ids);

  ServeConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model_id = "qwen-fail-test";
  config.max_concurrency = 4;
  config.max_context = 4096;

  ServeHttpServer server(config, router, assets.tokenizer, assets.chat_template);
  const int port = server.bind();
  check(port > 0, "MIVR-4 failing server bind failed");

  std::thread server_thread([&] { server.listen_after_bind(); });
  const auto cleanup = [&] {
    server.stop();
    if (server_thread.joinable()) {
      server_thread.join();
    }
    router->stop();
  };

  try {
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(15, 0);

    auto assert_sse_error = [](const ParsedSse &sse,
                               std::string_view expected_message,
                               std::string_view label) {
      check(!sse.saw_done,
            std::string(label) + ": failed SSE stream must not emit [DONE]");
      bool saw_error = false;
      for (const auto &c : sse.chunks) {
        if (c.contains("error")) {
          saw_error = true;
          check(c.at("error").at("type") == "server_error",
                std::string(label) + ": error.type != server_error");
          check(c.at("error").at("message") == expected_message,
                std::string(label) + ": error.message mismatch: " +
                    c.at("error").at("message").get<std::string>());
        } else {
          if (c.contains("choices") && !c.at("choices").empty()) {
            check(c.at("choices")[0].at("finish_reason").is_null(),
                  std::string(label) +
                      ": failed SSE stream must not emit finish_reason");
          }
          check(!c.contains("usage") || c.at("usage").is_null(),
                std::string(label) +
                    ": failed SSE stream must not emit usage chunk");
        }
      }
      check(saw_error,
            std::string(label) + ": failed SSE stream missing error event");
    };

    // 1) /v1/chat/completions SSE failure before first token
    {
      failing_backend->configure_next({}, "metal command buffer timeout");
      Json req = {
          {"model", "qwen-fail-test"},
          {"messages", {{{"role", "user"}, {"content", "hi"}}}},
          {"max_tokens", 16},
          {"stream", true},
          {"stream_options", {{"include_usage", true}}},
      };
      auto res =
          client.Post("/v1/chat/completions", req.dump(), "application/json");
      check(res && res->status == 200,
            "chat SSE pre-token failure request failed");
      auto sse = parse_sse_body(res->body);
      assert_sse_error(sse, "metal command buffer timeout",
                       "chat SSE pre-token failure");
    }

    // 2) /v1/chat/completions SSE failure mid-stream after partial tokens
    {
      failing_backend->configure_next({301, 302}, "kv page pool exhausted");
      Json req = {
          {"model", "qwen-fail-test"},
          {"messages", {{{"role", "user"}, {"content", "hi"}}}},
          {"max_tokens", 16},
          {"stream", true},
          {"stream_options", {{"include_usage", true}}},
      };
      auto res =
          client.Post("/v1/chat/completions", req.dump(), "application/json");
      check(res && res->status == 200,
            "chat SSE mid-stream failure request failed");
      auto sse = parse_sse_body(res->body);
      assert_sse_error(sse, "kv page pool exhausted",
                       "chat SSE mid-stream failure");
      std::string content;
      for (const auto &c : sse.chunks) {
        if (c.contains("choices") && !c.at("choices").empty() &&
            c.at("choices")[0].at("delta").contains("content")) {
          content +=
              c.at("choices")[0].at("delta").at("content").get<std::string>();
        }
      }
      check(content == "Hello ",
            "chat SSE mid-stream partial content mismatch: " + content);
    }

    // 3) /v1/completions SSE failure before first token (empty error fallback)
    {
      failing_backend->configure_next({}, "");
      Json req = {
          {"model", "qwen-fail-test"},
          {"prompt", Json::array({300})},
          {"max_tokens", 16},
          {"stream", true},
          {"stream_options", {{"include_usage", true}}},
      };
      auto res = client.Post("/v1/completions", req.dump(), "application/json");
      check(res && res->status == 200,
            "completions SSE pre-token failure request failed");
      auto sse = parse_sse_body(res->body);
      assert_sse_error(sse, "engine generation failed",
                       "completions SSE pre-token failure");
    }

    // 4) /v1/completions SSE failure mid-stream after partial token
    {
      failing_backend->configure_next({301}, "sidecar crashed");
      Json req = {
          {"model", "qwen-fail-test"},
          {"prompt", Json::array({300})},
          {"max_tokens", 16},
          {"stream", true},
          {"stream_options", {{"include_usage", true}}},
      };
      auto res = client.Post("/v1/completions", req.dump(), "application/json");
      check(res && res->status == 200,
            "completions SSE mid-stream failure request failed");
      auto sse = parse_sse_body(res->body);
      assert_sse_error(sse, "sidecar crashed",
                       "completions SSE mid-stream failure");
      std::string text;
      for (const auto &c : sse.chunks) {
        if (c.contains("choices") && !c.at("choices").empty()) {
          text += c.at("choices")[0].value("text", "");
        }
      }
      check(text == "Hello",
            "completions SSE mid-stream partial text mismatch: " + text);
    }
  } catch (...) {
    cleanup();
    throw;
  }
  cleanup();
}

class RecordingServeBackend final : public ServeBackend {
public:
  std::uint64_t submit(SubmitRequest request) override {
    std::lock_guard lock(mutex_);
    const std::uint64_t id = next_id_++;
    recorded_.push_back(request);
    events_.push_back(TokenEvent{
        .request_id = id,
        .tokens = {301},
        .kind = EventKind::tokens,
    });
    events_.push_back(TokenEvent{
        .request_id = id,
        .kind = EventKind::finished,
        .finish_reason = "stop",
    });
    cv_.notify_all();
    return id;
  }

  bool cancel(std::uint64_t /*request_id*/) override { return true; }

  std::vector<TokenEvent>
  poll_events(std::chrono::microseconds max_wait) override {
    std::unique_lock lock(mutex_);
    cv_.wait_for(lock, max_wait,
                 [this] { return !events_.empty() || stopped_; });
    std::vector<TokenEvent> out;
    while (!events_.empty()) {
      out.push_back(std::move(events_.front()));
      events_.pop_front();
    }
    return out;
  }

  CapacityReport capacity() const override {
    return CapacityReport{.max_concurrency = 4, .accepting = true};
  }
  StatsReport stats() const override { return {.backend_kind = "recording"}; }
  TrimCacheReport trim_cache(std::uint64_t /*target_free_bytes*/) override {
    return {};
  }
  void set_scheduler_hint(const SchedulerHint & /*hint*/) override {}
  std::uint64_t resident_memory_bytes() const override { return 0; }
  void shutdown() override {
    std::lock_guard lock(mutex_);
    stopped_ = true;
    cv_.notify_all();
  }
  std::string provider_id() const override { return "io.ninfer.metal"; }
  std::string backend_mode() const override { return "inproc"; }

  std::vector<SubmitRequest> recorded() const {
    std::lock_guard lock(mutex_);
    return recorded_;
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<TokenEvent> events_;
  std::vector<SubmitRequest> recorded_;
  std::uint64_t next_id_ = 1;
  bool stopped_ = false;
};

void test_mivr5_sampling_defaults_and_validation() {
  auto assets = build_synthetic_assets();
  auto backend = std::make_shared<RecordingServeBackend>();
  auto router = std::make_shared<EngineRouter>(backend, assets.eos_ids);

  ServeConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model_id = "qwen-sampling-test";
  config.max_concurrency = 4;
  config.max_context = 4096;

  ServeHttpServer server(config, router, assets.tokenizer, assets.chat_template);
  const int port = server.bind();
  check(port > 0, "MIVR-5 server bind failed");
  std::thread server_thread([&] { server.listen_after_bind(); });
  const auto cleanup = [&] {
    server.stop();
    if (server_thread.joinable()) {
      server_thread.join();
    }
    router->stop();
  };

  try {
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(15, 0);

    // 1) Standard OpenAI chat request with temperature > 0 and no top_k
    //    must default top_k to 64.
    {
      Json req = {
          {"model", "qwen-sampling-test"},
          {"messages", {{{"role", "user"}, {"content", "hi"}}}},
          {"temperature", 0.7},
          {"max_tokens", 8},
      };
      auto res =
          client.Post("/v1/chat/completions", req.dump(), "application/json");
      check(res && res->status == 200,
            "chat temperature=0.7 without top_k must return 200");
      const auto rec = backend->recorded();
      check(rec.size() == 1 && rec.back().sampling.top_k == 64 &&
                std::abs(rec.back().sampling.temperature - 0.7F) < 1e-5F,
            "chat temperature=0.7 should default top_k=64");
    }

    // 2) Standard OpenAI /v1/completions with temperature > 0 and no top_k
    {
      Json req = {
          {"model", "qwen-sampling-test"},
          {"prompt", Json::array({300})},
          {"temperature", 0.5},
          {"max_tokens", 8},
      };
      auto res = client.Post("/v1/completions", req.dump(), "application/json");
      check(res && res->status == 200,
            "completions temperature=0.5 without top_k must return 200");
      const auto rec = backend->recorded();
      check(rec.size() == 2 && rec.back().sampling.top_k == 64,
            "completions temperature=0.5 should default top_k=64");
    }

    // 3) Explicit top_k=0 with temperature > 0 maps to 64; explicit top_k=16
    //    is preserved; greedy temperature=0 keeps top_k=0; min_p=0 accepted.
    {
      Json req_zero = {
          {"model", "qwen-sampling-test"},
          {"messages", {{{"role", "user"}, {"content", "hi"}}}},
          {"temperature", 0.8},
          {"top_k", 0},
          {"min_p", 0.0},
      };
      auto r1 = client.Post("/v1/chat/completions", req_zero.dump(),
                            "application/json");
      check(r1 && r1->status == 200, "temperature=0.8 top_k=0 must succeed");
      check(backend->recorded().back().sampling.top_k == 64,
            "temperature=0.8 top_k=0 must map to 64");

      Json req_16 = {
          {"model", "qwen-sampling-test"},
          {"messages", {{{"role", "user"}, {"content", "hi"}}}},
          {"temperature", 0.8},
          {"top_k", 16},
      };
      auto r2 =
          client.Post("/v1/chat/completions", req_16.dump(), "application/json");
      check(r2 && r2->status == 200, "temperature=0.8 top_k=16 must succeed");
      check(backend->recorded().back().sampling.top_k == 16,
            "explicit top_k=16 must be preserved");

      Json req_greedy = {
          {"model", "qwen-sampling-test"},
          {"messages", {{{"role", "user"}, {"content", "hi"}}}},
          {"temperature", 0.0},
      };
      auto r3 = client.Post("/v1/chat/completions", req_greedy.dump(),
                            "application/json");
      check(r3 && r3->status == 200, "greedy temperature=0 must succeed");
      check(backend->recorded().back().sampling.top_k == 0,
            "greedy temperature=0 should keep top_k=0");
    }

    // 4) Unsupported min_p > 0 and top_k > 64 must be rejected with HTTP 400
    //    at the HTTP layer before backend submission.
    const std::size_t count_before = backend->recorded().size();
    {
      Json bad_min_p = {
          {"model", "qwen-sampling-test"},
          {"messages", {{{"role", "user"}, {"content", "hi"}}}},
          {"min_p", 0.05},
      };
      auto r = client.Post("/v1/chat/completions", bad_min_p.dump(),
                           "application/json");
      check(r && r->status == 400, "chat min_p=0.05 must return HTTP 400");
      check(Json::parse(r->body).at("error").at("type") ==
                "invalid_request_error",
            "chat min_p=0.05 error type mismatch");

      Json bad_min_p_cmpl = {
          {"model", "qwen-sampling-test"},
          {"prompt", Json::array({300})},
          {"min_p", 0.1},
      };
      auto r2 = client.Post("/v1/completions", bad_min_p_cmpl.dump(),
                            "application/json");
      check(r2 && r2->status == 400,
            "completions min_p=0.1 must return HTTP 400");

      Json bad_top_k = {
          {"model", "qwen-sampling-test"},
          {"messages", {{{"role", "user"}, {"content", "hi"}}}},
          {"temperature", 0.7},
          {"top_k", 65},
      };
      auto r3 = client.Post("/v1/chat/completions", bad_top_k.dump(),
                            "application/json");
      check(r3 && r3->status == 400, "chat top_k=65 must return HTTP 400");

      Json bad_top_k_cmpl = {
          {"model", "qwen-sampling-test"},
          {"prompt", Json::array({300})},
          {"top_k", 128},
      };
      auto r4 = client.Post("/v1/completions", bad_top_k_cmpl.dump(),
                            "application/json");
      check(r4 && r4->status == 400,
            "completions top_k=128 must return HTTP 400");
    }
    check(backend->recorded().size() == count_before,
          "invalid sampling requests must not reach backend->submit");
  } catch (...) {
    cleanup();
    throw;
  }
  cleanup();
}

void test_mivr9_config_cli_and_budgets() {
  const auto tmp_dir =
      std::filesystem::temp_directory_path() /
      ("ninfer_mivr9_test_" +
       std::to_string(static_cast<std::uint64_t>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  struct DirCleanup {
    std::filesystem::path dir;
    ~DirCleanup() {
      std::error_code cleanup_ec;
      std::filesystem::remove_all(dir, cleanup_ec);
    }
  } guard{tmp_dir};

  auto parse_args = [](std::vector<std::string> args) {
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (auto &a : args) {
      argv.push_back(a.data());
    }
    return parse_serve_config(static_cast<int>(argv.size()), argv.data());
  };

  // 1) [engine].memory_budget_gb must NOT overwrite [server].memory_budget_gb,
  //    and [[models]] entries without explicit budget must inherit [engine]'s.
  const auto toml1 = tmp_dir / "budgets.toml";
  {
    std::ofstream out(toml1);
    out << "[server]\n"
        << "host = \"127.0.0.1\"\n"
        << "port = 8181\n"
        << "memory_budget_gb = 48.0\n\n"
        << "[engine]\n"
        << "max_concurrency = 2\n"
        << "max_context = 16384\n"
        << "kv_dtype = \"bf16\"\n"
        << "sampling_contract = \"legacy_rejection_v1\"\n"
        << "rolling_prefix_cache = true\n"
        << "prefix_cache_gb = 4.0\n"
        << "memory_budget_gb = 12.0\n\n"
        << "[speculative]\n"
        << "backend = \"mtp\"\n"
        << "draft_tokens = 2\n\n"
        << "[[models]]\n"
        << "id = \"m-inherit\"\n"
        << "model_dir = \"/tmp/m-inherit\"\n\n"
        << "[[models]]\n"
        << "id = \"m-override\"\n"
        << "model_dir = \"/tmp/m-override\"\n"
        << "max_concurrency = 4\n"
        << "max_context = 32768\n"
        << "kv_dtype = \"affine4_g128\"\n"
        << "memory_budget_gb = 20.0\n";
  }

  {
    const auto cfg = parse_args({"ninfer-metal-serve", "--config",
                                 toml1.string()});
    check(resolve_total_memory_budget_bytes(cfg) == (48ULL << 30),
          "[engine].memory_budget_gb must not overwrite [server].memory_budget_gb");
    const auto models = resolve_model_configs(cfg);
    check(models.size() == 2, "expected 2 models");
    check(models[0].memory_budget_bytes == (12ULL << 30),
          "[[models]][0] must inherit [engine].memory_budget_gb (12 GiB)");
    check(models[1].memory_budget_bytes == (20ULL << 30),
          "[[models]][1] explicit memory_budget_gb (20 GiB) mismatch");
  }

  // 2) CLI flags must override TOML [[models]] even when --model-dir is omitted.
  {
    const auto cfg = parse_args({
        "ninfer-metal-serve",
        "--config",
        toml1.string(),
        "--max-concurrency",
        "3",
        "--max-context",
        "8192",
        "--kv-dtype",
        "affine4_g128",
        "--spec-backend",
        "none",
        "--draft-tokens",
        "5",
        "--rolling-prefix-cache",
        "off",
        "--prefix-cache-gb",
        "1.25",
        "--sampling-contract",
        "position_coupled_v1",
        "--chat-template",
        "/tmp/cli-override.jinja",
    });
    const auto models = resolve_model_configs(cfg);
    check(models.size() == 2, "expected 2 models after CLI overrides");
    for (const auto &m : models) {
      check(m.max_concurrency == 3,
            "CLI --max-concurrency did not override [[models]]");
      check(m.max_context == 8192,
            "CLI --max-context did not override [[models]]");
      check(m.kv_dtype == "affine4_g128",
            "CLI --kv-dtype did not override [[models]]");
      check(m.spec_backend == SpeculativeBackend::none && !m.spec_backend_auto,
            "CLI --spec-backend did not override [[models]]");
      check(m.draft_tokens == 5,
            "CLI --draft-tokens did not override [[models]]");
      check(m.rolling_prefix_cache.has_value() && !*m.rolling_prefix_cache,
            "CLI --rolling-prefix-cache did not override [[models]]");
      check(m.prefix_cache_gb.has_value() && *m.prefix_cache_gb == 1.25,
            "CLI --prefix-cache-gb did not override [[models]]");
      check(m.sampling_contract == "position_coupled_v1",
            "CLI --sampling-contract did not override [[models]]");
      check(m.chat_template == "/tmp/cli-override.jinja",
            "CLI --chat-template did not override [[models]]");
    }
  }

  // 3) [engine].memory_budget_bytes must not overwrite server total cap, and
  //    [[models]].memory_budget_gb overrides inherited engine bytes.
  const auto toml2 = tmp_dir / "engine_bytes.toml";
  {
    std::ofstream out(toml2);
    out << "[server]\n"
        << "memory_budget_gb = 64.0\n\n"
        << "[engine]\n"
        << "memory_budget_bytes = " << (8ULL << 30) << "\n\n"
        << "[[models]]\n"
        << "id = \"m-bytes\"\n"
        << "model_dir = \"/tmp/m-bytes\"\n\n"
        << "[[models]]\n"
        << "id = \"m-gb\"\n"
        << "model_dir = \"/tmp/m-gb\"\n"
        << "memory_budget_gb = 16.0\n";
  }
  {
    const auto cfg =
        parse_args({"ninfer-metal-serve", "--config", toml2.string()});
    check(resolve_total_memory_budget_bytes(cfg) == (64ULL << 30),
          "server total budget mismatch when [engine].memory_budget_bytes set");
    const auto models = resolve_model_configs(cfg);
    check(models.size() == 2 &&
              models[0].memory_budget_bytes == (8ULL << 30) &&
              models[1].memory_budget_bytes == (16ULL << 30),
          "per-model engine bytes inheritance / gb override mismatch");
  }
}

void test_mivr10_provider_auto_routing_no_hijack() {
  const auto tmp_dir =
      std::filesystem::temp_directory_path() /
      ("ninfer_mivr10_test_" +
       std::to_string(static_cast<std::uint64_t>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  struct DirCleanup {
    std::filesystem::path dir;
    ~DirCleanup() {
      std::error_code cleanup_ec;
      std::filesystem::remove_all(dir, cleanup_ec);
    }
  } guard{tmp_dir};

  // Register 3 external Path B providers trying to match qwen3_5 / llama
  ProviderRegistry reg;
  reg.register_provider(ProviderManifest{
      .provider_id = "io.test.no-ext",
      .engine_id = "no-ext",
      .protocol_version = 2,
      .binary_path = "/bin/true",
      .supported_model_types = {"qwen3_5", "qwen3_8"},
      .artifact_extensions = {},
  });
  reg.register_provider(ProviderManifest{
      .provider_id = "io.test.qwen-omlxm5",
      .engine_id = "qwen-ext",
      .protocol_version = 2,
      .binary_path = "/bin/true",
      .supported_model_types = {"qwen3_5", "qwen3_8"},
      .artifact_extensions = {".omlxm5"},
  });
  reg.register_provider(ProviderManifest{
      .provider_id = "io.test.custom-ext",
      .engine_id = "custom-ext",
      .protocol_version = 2,
      .binary_path = "/bin/true",
      .supported_topologies = {"llama_custom_v1"},
      .supported_model_types = {"qwen3_5", "llama"},
      .artifact_extensions = {".custom"},
  });

  // Case 1: Built-in Qwen dir with model.omlxm5 (no model_pack.json) must NOT
  // be hijacked by any external provider.
  const auto qwen_builtin = tmp_dir / "qwen-builtin";
  std::filesystem::create_directories(qwen_builtin);
  std::ofstream(qwen_builtin / "config.json") << R"({"model_type": "qwen3_5"})";
  std::ofstream(qwen_builtin / "model.omlxm5") << "dummy";
  {
    ModelServeConfig m;
    m.model_dir = qwen_builtin.string();
    m.provider = "auto";
    const auto res = resolve_provider(m, reg);
    check(!res.use_sidecar && res.provider_id == "io.ninfer.metal",
          "built-in Qwen model.omlxm5 was hijacked by: " + res.provider_id);
  }

  // Case 2: External provider declares artifact_extensions=[".custom"], but
  // model dir has NO .custom file -> must NOT be selected despite matching
  // model_type and topology_id.
  const auto llama_no_ext = tmp_dir / "llama-no-custom";
  std::filesystem::create_directories(llama_no_ext);
  std::ofstream(llama_no_ext / "config.json") << R"({"model_type": "llama"})";
  std::ofstream(llama_no_ext / "model_pack.json")
      << R"({"topology_id": "llama_custom_v1"})";
  {
    ModelServeConfig m;
    m.model_dir = llama_no_ext.string();
    m.provider = "auto";
    const auto res = resolve_provider(m, reg);
    check(!res.use_sidecar && res.provider_id == "io.ninfer.metal",
          "provider with missing required extension must not be selected, got: " +
              res.provider_id);
  }

  // Case 3: When model.custom DOES exist, io.test.custom-ext IS selected.
  const auto llama_with_ext = tmp_dir / "llama-with-custom";
  std::filesystem::create_directories(llama_with_ext);
  std::ofstream(llama_with_ext / "config.json") << R"({"model_type": "llama"})";
  std::ofstream(llama_with_ext / "model.custom") << "dummy-custom";
  {
    ModelServeConfig m;
    m.model_dir = llama_with_ext.string();
    m.provider = "auto";
    const auto res = resolve_provider(m, reg);
    check(res.use_sidecar && res.provider_id == "io.test.custom-ext" &&
              res.resolved_artifact_path.ends_with("model.custom"),
          "provider with matching .custom artifact should be selected");
  }

  // Case 4: Explicit provider_id in model_pack.json still routes to sidecar
  // even when model.omlxm5 exists.
  const auto qwen_explicit = tmp_dir / "qwen-explicit-pack";
  std::filesystem::create_directories(qwen_explicit);
  std::ofstream(qwen_explicit / "config.json")
      << R"({"model_type": "qwen3_5"})";
  std::ofstream(qwen_explicit / "model.omlxm5") << "dummy";
  std::ofstream(qwen_explicit / "model_pack.json")
      << R"({"provider_id": "io.test.qwen-omlxm5"})";
  {
    ModelServeConfig m;
    m.model_dir = qwen_explicit.string();
    m.provider = "auto";
    const auto res = resolve_provider(m, reg);
    check(res.use_sidecar && res.provider_id == "io.test.qwen-omlxm5",
          "explicit model_pack.json provider_id should still route to sidecar");
  }
}

bool raw_uds_write_line(int fd, std::string line) {
  line.push_back('\n');
  std::size_t off = 0;
  while (off < line.size()) {
    const auto n = ::send(fd, line.data() + off, line.size() - off, 0);
    if (n < 0) {
      if (errno == EINTR) {
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

std::optional<std::string> raw_uds_read_line(int fd, std::string &buf,
                                             int timeout_ms) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
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
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      return std::nullopt;
    }
    const int rem_ms = static_cast<int>(
        std::max<std::int64_t>(
            1, std::chrono::duration_cast<std::chrono::milliseconds>(deadline -
                                                                     now)
                   .count()));
    pollfd pfd{.fd = fd, .events = POLLIN, .revents = 0};
    const int r = ::poll(&pfd, 1, rem_ms);
    if (r < 0) {
      if (errno == EINTR) {
        continue;
      }
      return std::nullopt;
    }
    if (r == 0) {
      return std::nullopt;
    }
    std::array<char, 4096> chunk{};
    const auto n = ::recv(fd, chunk.data(), chunk.size(), 0);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) {
        continue;
      }
      if (!buf.empty()) {
        std::string tail = std::move(buf);
        buf.clear();
        return tail;
      }
      return std::nullopt;
    }
    buf.append(chunk.data(), static_cast<std::size_t>(n));
  }
}

int raw_uds_connect(const std::filesystem::path &sock_path) {
  if (sock_path.native().size() >= sizeof(sockaddr_un::sun_path)) {
    return -1;
  }
  const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::memcpy(addr.sun_path, sock_path.c_str(), sock_path.native().size() + 1);
  if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
    return fd;
  }
  ::close(fd);
  return -1;
}

void test_sidecar_crash_and_rpc_resilience(
    const std::filesystem::path &mock_provider_bin) {
  auto assets = build_synthetic_assets();
  const auto tmp_root =
      std::filesystem::temp_directory_path() /
      ("ninfer_mivr68_test_" +
       std::to_string(static_cast<std::uint64_t>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(tmp_root, ec);
  std::filesystem::create_directories(tmp_root);
  struct DirCleanup {
    std::filesystem::path dir;
    ~DirCleanup() {
      std::error_code cleanup_ec;
      std::filesystem::remove_all(dir, cleanup_ec);
    }
  } dir_cleanup{tmp_root};

  // Part A (MIVR-6): sidecar SIGKILL during in-flight sync + SSE requests
  // must fail both HTTP streams promptly, clear active_requests, wake
  // poll_events without sleeping max_wait, and keep /health + /metrics alive.
  {
    const auto model_dir = tmp_root / "crash-model";
    std::filesystem::create_directories(model_dir);
    std::ofstream(model_dir / "tokenizer.json") << assets.tokenizer_json;
    std::ofstream(model_dir / "tokenizer_config.json")
        << assets.tokenizer_config_json;
    std::ofstream(model_dir / "chat_template.jinja") << assets.template_source;
    std::ofstream(model_dir / "config.json")
        << R"({"model_type": "llama", "max_position_embeddings": 4096})";

    auto host = std::make_shared<MultiModelHost>(16ULL << 30);

    LoadOptions fake_lo;
    fake_lo.max_concurrency = 2;
    fake_lo.max_context = 4096;
    FakeBackendOptions fake_opts;
    fake_opts.vocab_size = assets.vocab_size;
    std::shared_ptr<Engine> healthy_eng(Engine::create_fake(fake_lo, fake_opts));
    auto healthy_router = std::make_shared<EngineRouter>(
        std::make_shared<LocalEngineBackend>(healthy_eng, 0, "io.ninfer.metal"),
        assets.eos_ids);
    host->add_model(ModelInstance{
        .id = "healthy-inproc",
        .provider_id = "io.ninfer.metal",
        .backend_mode = "inproc",
        .router = healthy_router,
        .tokenizer = assets.tokenizer,
        .chat_template = assets.chat_template,
        .max_context = 4096,
        .max_concurrency = 2,
    });

    UdsSidecarOptions sc_opts;
    sc_opts.provider_id = "io.test.mock-microkernel";
    sc_opts.binary_path = mock_provider_bin;
    sc_opts.load_options.model_dir = model_dir.string();
    sc_opts.load_options.max_concurrency = 2;
    sc_opts.load_options.max_context = 4096;
    auto sidecar_backend = std::make_shared<UdsSidecarBackend>(sc_opts);
    const pid_t sidecar_pid = sidecar_backend->child_pid();
    auto sidecar_router =
        std::make_shared<EngineRouter>(sidecar_backend, assets.eos_ids);
    auto crash_model = host->add_model(ModelInstance{
        .id = "crash-sidecar",
        .provider_id = "io.test.mock-microkernel",
        .backend_mode = "sidecar",
        .router = sidecar_router,
        .tokenizer = assets.tokenizer,
        .chat_template = assets.chat_template,
        .max_context = 4096,
        .max_concurrency = 2,
    });

    ServeConfig http_cfg;
    http_cfg.host = "127.0.0.1";
    http_cfg.port = 0;
    ServeHttpServer server(http_cfg, host);
    const int port = server.bind();
    check(port > 0, "MIVR-6 server bind failed");
    std::thread server_thread([&] { server.listen_after_bind(); });
    const auto cleanup = [&] {
      server.stop();
      if (server_thread.joinable()) {
        server_thread.join();
      }
      host->stop_all();
    };

    try {
      int sync_status = 0;
      std::string sync_body;
      int sse_status = 0;
      std::string sse_body;

      // Prompt token 1 -> emits 2..201 (none in eos_ids {291, 305, ...}),
      // taking ~4s at 20ms/step so SIGKILL reliably lands mid-flight.
      std::thread t_sync([&] {
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(10, 0);
        Json req = {
            {"model", "crash-sidecar"},
            {"prompt", Json::array({1})},
            {"max_tokens", 200},
        };
        if (auto r = c.Post("/v1/completions", req.dump(), "application/json")) {
          sync_status = r->status;
          sync_body = r->body;
        }
      });
      std::thread t_sse([&] {
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(10, 0);
        Json req = {
            {"model", "crash-sidecar"},
            {"prompt", Json::array({1})},
            {"max_tokens", 200},
            {"stream", true},
        };
        if (auto r = c.Post("/v1/completions", req.dump(), "application/json")) {
          sse_status = r->status;
          sse_body = r->body;
        }
      });

      const auto wait_deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (crash_model->router->active_requests() < 2 &&
             std::chrono::steady_clock::now() < wait_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      check(crash_model->router->active_requests() == 2,
            "expected 2 active sidecar requests before SIGKILL");

      // Kill sidecar abruptly (simulating OOM / SIGSEGV crash)
      check(::kill(sidecar_pid, SIGKILL) == 0, "SIGKILL on sidecar failed");

      t_sync.join();
      t_sse.join();

      check(sync_status == 500,
            "sync request during sidecar crash must return 500, got " +
                std::to_string(sync_status));
      const auto sync_json = Json::parse(sync_body);
      check(sync_json.at("error").at("type") == "server_error",
            "sync crash response missing server_error JSON");

      check(sse_status == 200, "SSE crash response HTTP status mismatch");
      const auto sse = parse_sse_body(sse_body);
      check(!sse.saw_done, "SSE crash stream must not emit [DONE]");
      bool saw_sse_error = false;
      for (const auto &chunk : sse.chunks) {
        if (chunk.contains("error") &&
            chunk.at("error").at("type") == "server_error") {
          saw_sse_error = true;
        }
      }
      check(saw_sse_error, "SSE crash stream missing server_error event");
      check(crash_model->router->active_requests() == 0,
            "active_requests leaked after sidecar crash: " +
                std::to_string(crash_model->router->active_requests()));

      // poll_events after event_stream_closed must wake promptly (not wait 500ms)
      const auto poll_start = std::chrono::steady_clock::now();
      bool poll_threw = false;
      try {
        (void)sidecar_backend->poll_events(std::chrono::milliseconds(500));
      } catch (const std::runtime_error &) {
        poll_threw = true;
      }
      const auto poll_elapsed_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - poll_start)
              .count();
      check(poll_threw, "poll_events on dead sidecar should report closed stream");
      check(poll_elapsed_ms < 200,
            "poll_events blocked for max_wait after event_stream_closed: " +
                std::to_string(poll_elapsed_ms) + "ms");

      // /health must return structured 503 degraded JSON (NOT empty 500)
      httplib::Client client("127.0.0.1", port);
      client.set_read_timeout(5, 0);
      {
        auto h = client.Get("/health");
        check(h && h->status == 503 && !h->body.empty(),
              "/health must return non-empty 503 degraded JSON after crash");
        const auto hj = Json::parse(h->body);
        check(hj.at("status") == "degraded",
              "/health status must be degraded, got: " + hj.dump());
        check(hj.at("models").size() == 2 &&
                  hj.at("models")[0].at("status") == "ok" &&
                  hj.at("models")[1].at("status") == "error",
              "/health per-model status mismatch: " + hj.dump());
      }

      // /metrics must return HTTP 200 Prometheus text with health gauges
      {
        auto m = client.Get("/metrics");
        check(m && m->status == 200 && !m->body.empty(),
              "/metrics must return 200 after sidecar crash");
        check(m->body.find(
                  "ninfer_metal_model_healthy{model=\"healthy-inproc\","
                  "provider=\"io.ninfer.metal\",backend_mode=\"inproc\"} 1") !=
                  std::string::npos,
              "/metrics missing healthy=1 for healthy-inproc");
        check(m->body.find(
                  "ninfer_metal_model_healthy{model=\"crash-sidecar\","
                  "provider=\"io.test.mock-microkernel\","
                  "backend_mode=\"sidecar\"} 0") != std::string::npos,
              "/metrics missing healthy=0 for crash-sidecar");
      }
    } catch (...) {
      cleanup();
      throw;
    }
    cleanup();
  }

  // Part B (MIVR-8): control RPC timeout invalidates control_fd and response
  // id mismatch is rejected + closes control_fd.
  const std::string py_shebang =
      (::access("/usr/bin/python3", X_OK) == 0) ? "#!/usr/bin/python3\n"
                                                : "#!/usr/bin/env python3\n";
  auto write_faulty_sidecar = [&](const std::filesystem::path &path,
                                  std::string_view mode) {
    std::ofstream out(path);
    out << py_shebang
        << "import json, os, socket, sys, threading, time\n"
        << "MODE = '" << mode << "'\n"
        << "sock_path = None\n"
        << "args = sys.argv[1:]\n"
        << "for i, a in enumerate(args):\n"
        << "    if a == '--socket' and i + 1 < len(args):\n"
        << "        sock_path = args[i + 1]\n"
        << "if os.path.exists(sock_path):\n"
        << "    os.unlink(sock_path)\n"
        << "srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        << "srv.bind(sock_path)\n"
        << "srv.listen(8)\n"
        << "stop_ev = threading.Event()\n"
        << "def send_json(conn, obj):\n"
        << "    conn.sendall((json.dumps(obj) + '\\n').encode('utf-8'))\n"
        << "def handle(conn):\n"
        << "    buf = b''\n"
        << "    try:\n"
        << "        while not stop_ev.is_set():\n"
        << "            chunk = conn.recv(4096)\n"
        << "            if not chunk:\n"
        << "                break\n"
        << "            buf += chunk\n"
        << "            while b'\\n' in buf:\n"
        << "                line, buf = buf.split(b'\\n', 1)\n"
        << "                req = json.loads(line.decode('utf-8'))\n"
        << "                rid = req['id']\n"
        << "                m = req['method']\n"
        << "                if m == 'hello':\n"
        << "                    send_json(conn, {'id': rid, 'ok': True, 'result': {'protocol_version': 2, 'provider_id': 'io.test.faulty', 'capabilities': ['subscribe_events']}})\n"
        << "                elif m == 'load':\n"
        << "                    send_json(conn, {'id': rid, 'ok': True, 'result': {}})\n"
        << "                elif m == 'subscribe_events':\n"
        << "                    send_json(conn, {'id': rid, 'ok': True, 'result': {'subscribed': True}})\n"
        << "                    conn.settimeout(0.1)\n"
        << "                    while not stop_ev.is_set():\n"
        << "                        try:\n"
        << "                            data = conn.recv(1)\n"
        << "                            if not data:\n"
        << "                                stop_ev.set()\n"
        << "                                break\n"
        << "                        except socket.timeout:\n"
        << "                            continue\n"
        << "                        except Exception:\n"
        << "                            stop_ev.set()\n"
        << "                            break\n"
        << "                    try:\n"
        << "                        send_json(conn, {'event': {'kind': 'eof'}})\n"
        << "                    except Exception:\n"
        << "                        pass\n"
        << "                    return\n"
        << "                elif m == 'stats':\n"
        << "                    if MODE == 'slow_stats':\n"
        << "                        time.sleep(0.45)\n"
        << "                        send_json(conn, {'id': rid, 'ok': True, 'result': {'backend_kind': 'stale_late'}})\n"
        << "                    elif MODE == 'bad_id':\n"
        << "                        send_json(conn, {'id': rid + 99, 'ok': True, 'result': {'backend_kind': 'wrong_id'}})\n"
        << "                elif m == 'capacity':\n"
        << "                    send_json(conn, {'id': rid, 'ok': True, 'result': {'max_concurrency': 1, 'active_requests': 0, 'queued_requests': 0, 'accepting': True}})\n"
        << "                elif m == 'shutdown':\n"
        << "                    send_json(conn, {'id': rid, 'ok': True, 'result': {'shutdown': True}})\n"
        << "                    stop_ev.set()\n"
        << "                    return\n"
        << "    except Exception:\n"
        << "        pass\n"
        << "    finally:\n"
        << "        try:\n"
        << "            conn.close()\n"
        << "        except Exception:\n"
        << "            pass\n"
        << "srv.settimeout(0.1)\n"
        << "while not stop_ev.is_set():\n"
        << "    try:\n"
        << "        c, _ = srv.accept()\n"
        << "        threading.Thread(target=handle, args=(c,), daemon=True).start()\n"
        << "    except socket.timeout:\n"
        << "        continue\n"
        << "srv.close()\n";
    out.close();
    ::chmod(path.c_str(), 0755);
  };

  // Subtest B1: RPC timeout invalidates control_fd so stale reply is never
  // misread by subsequent capacity() call.
  {
    const auto slow_script = tmp_root / "slow_rpc_sidecar.py";
    write_faulty_sidecar(slow_script, "slow_stats");
    UdsSidecarOptions opts;
    opts.provider_id = "io.test.faulty-slow";
    opts.binary_path = slow_script;
    opts.startup_timeout = std::chrono::milliseconds(5000);
    opts.rpc_timeout = std::chrono::milliseconds(150);
    UdsSidecarBackend sc(opts);

    bool timed_out = false;
    try {
      (void)sc.stats();
    } catch (const std::runtime_error &err) {
      timed_out =
          std::string(err.what()).find("timed out") != std::string::npos;
    }
    check(timed_out, "slow stats() RPC must time out");

    // Allow delayed server reply time to arrive if control_fd had stayed open.
    std::this_thread::sleep_for(std::chrono::milliseconds(400));

    bool closed_err = false;
    try {
      (void)sc.capacity();
    } catch (const std::runtime_error &err) {
      closed_err =
          std::string(err.what()).find("closed") != std::string::npos;
    }
    check(closed_err,
          "subsequent capacity() after timeout must fail with closed "
          "control_fd instead of reading stale response");
    sc.shutdown();
  }

  // Subtest B2: mismatched response id throws and invalidates control_fd.
  {
    const auto bad_id_script = tmp_root / "bad_id_sidecar.py";
    write_faulty_sidecar(bad_id_script, "bad_id");
    UdsSidecarOptions opts;
    opts.provider_id = "io.test.faulty-bad-id";
    opts.binary_path = bad_id_script;
    opts.startup_timeout = std::chrono::milliseconds(5000);
    opts.rpc_timeout = std::chrono::milliseconds(2000);
    UdsSidecarBackend sc(opts);

    bool id_mismatch = false;
    try {
      (void)sc.stats();
    } catch (const std::runtime_error &err) {
      id_mismatch =
          std::string(err.what()).find("id mismatch") != std::string::npos;
    }
    check(id_mismatch, "mismatched RPC response id must be rejected");

    bool closed_err = false;
    try {
      (void)sc.capacity();
    } catch (const std::runtime_error &err) {
      closed_err =
          std::string(err.what()).find("closed") != std::string::npos;
    }
    check(closed_err,
          "control_fd must be invalidated after RPC response id mismatch");
    sc.shutdown();
  }
}

void test_test6_mock_spi_provider_protocol(
    const std::filesystem::path &mock_provider_bin) {
  const auto tmp_root =
      std::filesystem::temp_directory_path() /
      ("ninfer_test6_mock_" +
       std::to_string(static_cast<std::uint64_t>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(tmp_root, ec);
  std::filesystem::create_directories(tmp_root);
  struct DirCleanup {
    std::filesystem::path dir;
    ~DirCleanup() {
      std::error_code cleanup_ec;
      std::filesystem::remove_all(dir, cleanup_ec);
    }
  } dir_cleanup{tmp_root};

  // Subtests 1 & 2: true interleaved concurrency + multi-token stop holdback
  {
    UdsSidecarOptions opts;
    opts.provider_id = "io.test.mock-microkernel";
    opts.binary_path = mock_provider_bin;
    opts.load_options.model_dir = tmp_root.string();
    opts.load_options.max_concurrency = 2;
    opts.load_options.max_context = 4096;
    UdsSidecarBackend sc(opts);

    // 1) Concurrent interleaving across 2 active requests
    SubmitRequest req1;
    req1.prompt_tokens = {1};
    req1.max_new_tokens = 8;
    SubmitRequest req2;
    req2.prompt_tokens = {50};
    req2.max_new_tokens = 8;

    const std::uint64_t id1 = sc.submit(req1);
    const std::uint64_t id2 = sc.submit(req2);

    std::vector<std::pair<std::uint64_t, EventKind>> timeline;
    std::unordered_map<std::uint64_t, std::vector<std::int32_t>> emitted;
    std::unordered_set<std::uint64_t> done;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (done.size() < 2 && std::chrono::steady_clock::now() < deadline) {
      for (const auto &ev : sc.poll_events(std::chrono::milliseconds(50))) {
        timeline.emplace_back(ev.request_id, ev.kind);
        if (ev.kind == EventKind::tokens) {
          emitted[ev.request_id].insert(emitted[ev.request_id].end(),
                                        ev.tokens.begin(), ev.tokens.end());
        } else {
          check(ev.kind == EventKind::finished && ev.finish_reason == "length",
                "unexpected terminal event in concurrency test");
          done.insert(ev.request_id);
        }
      }
    }
    check(done.size() == 2 && emitted[id1].size() == 8 &&
              emitted[id2].size() == 8,
          "both concurrent mock SPI requests must emit 8 tokens and finish");

    bool saw_id2_token = false;
    bool saw_id1_finished = false;
    bool id2_started_before_id1_done = false;
    bool id1_token_after_id2_started = false;
    for (const auto &[rid, kind] : timeline) {
      if (rid == id2 && kind == EventKind::tokens) {
        if (!saw_id1_finished) {
          id2_started_before_id1_done = true;
        }
        saw_id2_token = true;
      }
      if (rid == id1 && kind == EventKind::tokens && saw_id2_token) {
        id1_token_after_id2_started = true;
      }
      if (rid == id1 && kind != EventKind::tokens) {
        saw_id1_finished = true;
      }
    }
    check(id2_started_before_id1_done && id1_token_after_id2_started,
          "mock_spi_provider failed to interleave concurrent request events");

    // 2a) Multi-token stop_token_seqs holdback: stop on {302, 303, 304} must
    //     emit ONLY {301} without leaking {302, 303}.
    {
      SubmitRequest stop_req;
      stop_req.prompt_tokens = {300};
      stop_req.max_new_tokens = 10;
      stop_req.stop_token_seqs = {{302, 303, 304}};
      const std::uint64_t sid = sc.submit(stop_req);
      std::vector<std::int32_t> toks;
      std::string finish_reason;
      bool finished = false;
      const auto dl =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (!finished && std::chrono::steady_clock::now() < dl) {
        for (const auto &ev : sc.poll_events(std::chrono::milliseconds(50))) {
          if (ev.request_id != sid) {
            continue;
          }
          if (ev.kind == EventKind::tokens) {
            toks.insert(toks.end(), ev.tokens.begin(), ev.tokens.end());
          } else {
            finished = true;
            finish_reason = ev.finish_reason;
            check(ev.kind == EventKind::finished,
                  "multi-token stop request did not finish cleanly");
          }
        }
      }
      check(finished && finish_reason == "stop",
            "multi-token stop request must end with finish_reason=stop");
      check(toks == std::vector<std::int32_t>({301}),
            "multi-token stop sequence leaked prefix tokens");
    }

    // 2b) Partial multi-token prefix at max_new_tokens must be flushed cleanly.
    {
      SubmitRequest partial_req;
      partial_req.prompt_tokens = {300};
      partial_req.max_new_tokens = 3; // generates 301, 302, 303
      partial_req.stop_token_seqs = {{302, 303, 304}};
      const std::uint64_t pid = sc.submit(partial_req);
      std::vector<std::int32_t> toks;
      std::string finish_reason;
      bool finished = false;
      const auto dl =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (!finished && std::chrono::steady_clock::now() < dl) {
        for (const auto &ev : sc.poll_events(std::chrono::milliseconds(50))) {
          if (ev.request_id != pid) {
            continue;
          }
          if (ev.kind == EventKind::tokens) {
            toks.insert(toks.end(), ev.tokens.begin(), ev.tokens.end());
          } else {
            finished = true;
            finish_reason = ev.finish_reason;
          }
        }
      }
      check(finished && finish_reason == "length",
            "partial stop prefix request must end with finish_reason=length");
      check(toks == std::vector<std::int32_t>({301, 302, 303}),
            "held-back non-matching prefix tokens were not flushed at length");
    }

    sc.shutdown();
  }

  // Subtest 3: shutdown() must deliver "aborted" for ALL in-flight AND queued
  // requests BEFORE emitting {"kind": "eof"} on subscribe_events.
  {
    const auto sock_path =
        std::filesystem::path("/tmp") /
        ("nf-t6-" + std::to_string(static_cast<std::uint64_t>(::getpid())) +
         ".sock");
    std::filesystem::remove(sock_path, ec);
    std::string bin_str = mock_provider_bin.string();
    std::string flag_sock = "--socket";
    std::string sock_str = sock_path.string();
    std::string flag_exit = "--exit-on-disconnect";
    std::array<char *, 5> argv{{
        bin_str.data(),
        flag_sock.data(),
        sock_str.data(),
        flag_exit.data(),
        nullptr,
    }};
#ifdef __APPLE__
    char **envp = *_NSGetEnviron();
#else
    char **envp = environ;
#endif
    pid_t child = -1;
    check(::posix_spawn(&child, bin_str.c_str(), nullptr, nullptr, argv.data(),
                        envp) == 0 &&
              child > 0,
          "failed to spawn mock_spi_provider for shutdown ordering test");

    int ctrl_fd = -1;
    int ev_fd = -1;
    const auto cleanup_raw = [&] {
      if (ctrl_fd >= 0) {
        ::close(ctrl_fd);
        ctrl_fd = -1;
      }
      if (ev_fd >= 0) {
        ::close(ev_fd);
        ev_fd = -1;
      }
      if (child > 0) {
        int status = 0;
        const auto dl =
            std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (::waitpid(child, &status, WNOHANG) == 0 &&
               std::chrono::steady_clock::now() < dl) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (::waitpid(child, &status, WNOHANG) == 0) {
          (void)::kill(child, SIGKILL);
          (void)::waitpid(child, &status, 0);
        }
        child = -1;
      }
      std::error_code rm_ec;
      std::filesystem::remove(sock_path, rm_ec);
    };

    try {
      const auto conn_dl =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (ctrl_fd < 0 && std::chrono::steady_clock::now() < conn_dl) {
        ctrl_fd = raw_uds_connect(sock_path);
        if (ctrl_fd < 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
      check(ctrl_fd >= 0, "failed to connect control socket to mock provider");
      ev_fd = raw_uds_connect(sock_path);
      check(ev_fd >= 0, "failed to connect event socket to mock provider");

      std::string ctrl_buf;
      std::string ev_buf;
      auto ctrl_rpc = [&](int id, const char *method, const Json &params) {
        Json req = {{"id", id}, {"method", method}, {"params", params}};
        check(raw_uds_write_line(ctrl_fd, req.dump()), "ctrl write failed");
        auto line = raw_uds_read_line(ctrl_fd, ctrl_buf, 5000);
        check(line.has_value(), "ctrl read timeout");
        auto resp = Json::parse(*line);
        check(resp.value("ok", false),
              std::string("ctrl RPC failed: ") + resp.dump());
        return resp.at("result");
      };

      (void)ctrl_rpc(1, "hello", Json::object());
      (void)ctrl_rpc(2, "load",
                     {{"model_dir", tmp_root.string()},
                      {"max_concurrency", 2},
                      {"max_context", 4096}});

      check(raw_uds_write_line(
                ev_fd,
                Json({{"id", 1},
                      {"method", "subscribe_events"},
                      {"params", {{"poll_interval_us", 10'000}}}})
                    .dump()),
            "subscribe_events write failed");
      auto sub_ack = raw_uds_read_line(ev_fd, ev_buf, 5000);
      check(sub_ack.has_value() && Json::parse(*sub_ack).value("ok", false),
            "subscribe_events ack failed");

      // Submit 3 long jobs (2 active + 1 queued)
      std::unordered_set<std::uint64_t> submitted_ids;
      for (int k = 0; k < 3; ++k) {
        auto r = ctrl_rpc(3 + k, "submit",
                          {{"prompt_tokens", Json::array({1 + k})},
                           {"max_new_tokens", 100}});
        submitted_ids.insert(r.at("request_id").get<std::uint64_t>());
      }

      // Wait for at least one token event so jobs are actively running.
      bool saw_token = false;
      std::unordered_set<std::uint64_t> aborted_before_eof;
      bool saw_eof = false;
      while (!saw_token) {
        auto line = raw_uds_read_line(ev_fd, ev_buf, 5000);
        check(line.has_value(), "timed out waiting for first token event");
        const auto ev = Json::parse(*line).at("event");
        if (ev.value("kind", "") == "tokens") {
          saw_token = true;
        }
      }

      // Trigger graceful shutdown while all 3 jobs are still active/queued.
      (void)ctrl_rpc(10, "shutdown", Json::object());

      while (!saw_eof) {
        auto line = raw_uds_read_line(ev_fd, ev_buf, 5000);
        check(line.has_value(), "timed out waiting for eof after shutdown");
        const auto ev = Json::parse(*line).at("event");
        const std::string kind = ev.value("kind", "");
        if (kind == "aborted") {
          aborted_before_eof.insert(ev.at("request_id").get<std::uint64_t>());
        } else if (kind == "eof") {
          saw_eof = true;
        }
      }

      check(saw_eof, "subscribe_events did not emit eof on shutdown");
      check(aborted_before_eof == submitted_ids,
            "shutdown truncated aborted events before eof (expected 3, got " +
                std::to_string(aborted_before_eof.size()) + ")");
    } catch (...) {
      cleanup_raw();
      throw;
    }
    cleanup_raw();
  }
}

class PaddingTokenServeBackend final : public ServeBackend {
public:
  std::uint64_t submit(SubmitRequest /*request*/) override {
    std::lock_guard lock(mutex_);
    const std::uint64_t id = next_id_++;
    // Valid token 301 ("Hello"), out-of-range padding IDs 999999 & -1, valid
    // token 304 ("!").
    events_.push_back(TokenEvent{
        .request_id = id,
        .tokens = {301, 999999, -1, 304},
        .kind = EventKind::tokens,
    });
    events_.push_back(TokenEvent{
        .request_id = id,
        .kind = EventKind::finished,
        .finish_reason = "stop",
    });
    cv_.notify_all();
    return id;
  }

  bool cancel(std::uint64_t /*request_id*/) override { return true; }

  std::vector<TokenEvent>
  poll_events(std::chrono::microseconds max_wait) override {
    std::unique_lock lock(mutex_);
    cv_.wait_for(lock, max_wait,
                 [this] { return !events_.empty() || stopped_; });
    std::vector<TokenEvent> out;
    while (!events_.empty()) {
      out.push_back(std::move(events_.front()));
      events_.pop_front();
    }
    return out;
  }

  CapacityReport capacity() const override {
    return CapacityReport{.max_concurrency = 4, .accepting = true};
  }
  StatsReport stats() const override { return {.backend_kind = "padding_test"}; }
  TrimCacheReport trim_cache(std::uint64_t /*target_free_bytes*/) override {
    return {};
  }
  void set_scheduler_hint(const SchedulerHint & /*hint*/) override {}
  std::uint64_t resident_memory_bytes() const override { return 0; }
  void shutdown() override {
    std::lock_guard lock(mutex_);
    stopped_ = true;
    cv_.notify_all();
  }
  std::string provider_id() const override { return "io.ninfer.metal"; }
  std::string backend_mode() const override { return "inproc"; }

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<TokenEvent> events_;
  std::uint64_t next_id_ = 1;
  bool stopped_ = false;
};

struct RawHttpChunkedResponse {
  std::string status_line;
  std::string body;
};

bool raw_tcp_send_all(int fd, std::string_view data) {
  std::size_t off = 0;
  while (off < data.size()) {
    const auto n = ::send(fd, data.data() + off, data.size() - off, 0);
    if (n < 0) {
      if (errno == EINTR) {
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

bool raw_tcp_read_more(int fd, std::string &buf, int timeout_ms) {
  pollfd pfd{.fd = fd, .events = POLLIN, .revents = 0};
  const int r = ::poll(&pfd, 1, timeout_ms);
  if (r <= 0) {
    return false;
  }
  std::array<char, 4096> chunk{};
  const auto n = ::recv(fd, chunk.data(), chunk.size(), 0);
  if (n <= 0) {
    return false;
  }
  buf.append(chunk.data(), static_cast<std::size_t>(n));
  return true;
}

RawHttpChunkedResponse raw_tcp_read_chunked_response(int fd, std::string &buf,
                                                     int timeout_ms) {
  // Read headers until "\r\n\r\n"
  std::size_t hdr_end = std::string::npos;
  while ((hdr_end = buf.find("\r\n\r\n")) == std::string::npos) {
    check(raw_tcp_read_more(fd, buf, timeout_ms),
          "TCP keep-alive connection closed before response headers");
  }
  const std::string headers = buf.substr(0, hdr_end);
  buf.erase(0, hdr_end + 4);

  RawHttpChunkedResponse resp;
  const std::size_t first_crlf = headers.find("\r\n");
  resp.status_line = first_crlf == std::string::npos
                         ? headers
                         : headers.substr(0, first_crlf);

  // Decode HTTP/1.1 chunked body until 0\r\n\r\n
  for (;;) {
    std::size_t sz_end = std::string::npos;
    while ((sz_end = buf.find("\r\n")) == std::string::npos) {
      check(raw_tcp_read_more(fd, buf, timeout_ms),
            "TCP connection closed while reading chunk size");
    }
    const std::string size_hex = buf.substr(0, sz_end);
    buf.erase(0, sz_end + 2);
    const std::size_t chunk_len = std::stoull(size_hex, nullptr, 16);
    if (chunk_len == 0) {
      while (buf.size() < 2) {
        check(raw_tcp_read_more(fd, buf, timeout_ms),
              "TCP connection closed before final chunk CRLF");
      }
      buf.erase(0, 2);
      break;
    }
    while (buf.size() < chunk_len + 2) {
      check(raw_tcp_read_more(fd, buf, timeout_ms),
            "TCP connection closed mid-chunk payload");
    }
    resp.body.append(buf, 0, chunk_len);
    buf.erase(0, chunk_len + 2);
  }
  return resp;
}

void test_mivr14_mivr15_mivr16_p2_fixes() {
  // 1) MIVR-16: OutputParser must preserve bare trailing "<tool_call>" on
  //    finish() and stop_strings without silently dropping it.
  {
    OutputParser p1({.enable_thinking_split = true,
                     .initial_in_thinking = false,
                     .enable_tool_parsing = true});
    auto d1 = p1.feed_bytes("<tool_call>");
    auto t1 = p1.finish();
    check(d1.content + t1.content == "<tool_call>",
          "MIVR-16 bare <tool_call> was dropped: '" + d1.content + t1.content +
              "'");
    check(p1.tool_call_count() == 0,
          "MIVR-16 bare <tool_call> must not count as a parsed tool call");

    OutputParser p2({.enable_thinking_split = true,
                     .initial_in_thinking = false,
                     .enable_tool_parsing = true});
    std::string acc;
    const std::string sample = "prefix <tool_call>";
    for (char ch : sample) {
      acc += p2.feed_bytes(std::string_view(&ch, 1)).content;
    }
    acc += p2.finish().content;
    check(acc == "prefix <tool_call>",
          "MIVR-16 byte-by-byte trailing <tool_call> mismatch: '" + acc + "'");

    OutputParser p3({.enable_thinking_split = false,
                     .initial_in_thinking = false,
                     .enable_tool_parsing = true,
                     .stop_strings = {"STOP"}});
    auto d3 = p3.feed_bytes("hello <tool_call>STOP");
    auto t3 = p3.finish();
    check(d3.stopped_by_string, "MIVR-16 stop string did not trigger");
    check(d3.content + t3.content == "hello <tool_call>",
          "MIVR-16 stop-string bare <tool_call> mismatch: '" + d3.content +
              t3.content + "'");
  }

  // 2) MIVR-14 (out-of-range/padding output token IDs across all 4 endpoints)
  //    & MIVR-15 (HTTP/1.1 Keep-Alive connection reuse across sequential SSE
  //    requests on a single TCP socket).
  auto assets = build_synthetic_assets();
  auto backend = std::make_shared<PaddingTokenServeBackend>();
  auto router = std::make_shared<EngineRouter>(backend, assets.eos_ids);

  ServeConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model_id = "qwen-p2-test";
  config.max_concurrency = 4;
  config.max_context = 4096;

  ServeHttpServer server(config, router, assets.tokenizer, assets.chat_template);
  const int port = server.bind();
  check(port > 0, "MIVR-14/15 server bind failed");
  std::thread server_thread([&] { server.listen_after_bind(); });
  const auto cleanup = [&] {
    server.stop();
    if (server_thread.joinable()) {
      server_thread.join();
    }
    router->stop();
  };

  try {
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(10, 0);

    // MIVR-14a: non-streaming /v1/chat/completions skips padding/invalid tokens
    {
      Json req = {
          {"model", "qwen-p2-test"},
          {"messages", {{{"role", "user"}, {"content", "hi"}}}},
          {"max_tokens", 16},
      };
      auto res =
          client.Post("/v1/chat/completions", req.dump(), "application/json");
      check(res && res->status == 200,
            "MIVR-14 chat sync failed on out-of-range padding token");
      check(Json::parse(res->body).at("choices")[0].at("message").at(
                "content") == "Hello!",
            "MIVR-14 chat sync content mismatch");
    }

    // MIVR-14b: non-streaming /v1/completions skips padding/invalid tokens
    {
      Json req = {
          {"model", "qwen-p2-test"},
          {"prompt", Json::array({300})},
          {"max_tokens", 16},
      };
      auto res = client.Post("/v1/completions", req.dump(), "application/json");
      check(res && res->status == 200,
            "MIVR-14 completions sync failed on out-of-range padding token");
      check(Json::parse(res->body).at("choices")[0].at("text") == "Hello!",
            "MIVR-14 completions sync text mismatch");
    }

    // MIVR-14c + MIVR-15: 3 sequential streaming SSE requests over ONE raw TCP
    // socket (chat SSE #1, chat SSE #2, completions SSE #3) verify BOTH
    // padding-token resilience in SSE AND Keep-Alive socket reuse after
    // sink.done() returns true.
    {
      const int tcp_fd = ::socket(AF_INET, SOCK_STREAM, 0);
      check(tcp_fd >= 0, "failed to create raw TCP socket for Keep-Alive test");
      struct FdCloser {
        int fd;
        ~FdCloser() {
          if (fd >= 0) {
            ::close(fd);
          }
        }
      } fd_closer{tcp_fd};

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(static_cast<std::uint16_t>(port));
      check(::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1,
            "inet_pton failed");
      check(::connect(tcp_fd, reinterpret_cast<sockaddr *>(&addr),
                      sizeof(addr)) == 0,
            "TCP connect to ServeHttpServer failed");

      std::string tcp_buf;
      auto post_keepalive_sse = [&](std::string_view path,
                                    const Json &payload) {
        const std::string body = payload.dump();
        std::ostringstream req;
        req << "POST " << path << " HTTP/1.1\r\n"
            << "Host: 127.0.0.1:" << port << "\r\n"
            << "Content-Type: application/json\r\n"
            << "Connection: keep-alive\r\n"
            << "Content-Length: " << body.size() << "\r\n\r\n"
            << body;
        check(raw_tcp_send_all(tcp_fd, req.str()),
              "Keep-Alive TCP send failed on " + std::string(path));
        return raw_tcp_read_chunked_response(tcp_fd, tcp_buf, 5000);
      };

      const Json chat_req = {
          {"model", "qwen-p2-test"},
          {"messages", {{{"role", "user"}, {"content", "hi"}}}},
          {"max_tokens", 16},
          {"stream", true},
      };
      for (int iter = 0; iter < 2; ++iter) {
        const auto resp =
            post_keepalive_sse("/v1/chat/completions", chat_req);
        check(resp.status_line.find("200") != std::string::npos,
              "MIVR-15 chat SSE iter " + std::to_string(iter) +
                  " bad status: " + resp.status_line);
        const auto sse = parse_sse_body(resp.body);
        check(sse.saw_done,
              "MIVR-15 chat SSE iter " + std::to_string(iter) +
                  " missing [DONE]");
        std::string content;
        for (const auto &c : sse.chunks) {
          if (!c.at("choices").empty()) {
            content += c.at("choices")[0].at("delta").value("content", "");
          }
        }
        check(content == "Hello!",
              "MIVR-14/15 chat SSE content mismatch: '" + content + "'");
      }

      const Json cmpl_req = {
          {"model", "qwen-p2-test"},
          {"prompt", Json::array({300})},
          {"max_tokens", 16},
          {"stream", true},
      };
      {
        const auto resp = post_keepalive_sse("/v1/completions", cmpl_req);
        check(resp.status_line.find("200") != std::string::npos,
              "MIVR-15 completions SSE on reused Keep-Alive socket failed: " +
                  resp.status_line);
        const auto sse = parse_sse_body(resp.body);
        check(sse.saw_done,
              "MIVR-15 completions SSE missing [DONE] on reused socket");
        std::string text;
        for (const auto &c : sse.chunks) {
          if (!c.at("choices").empty()) {
            text += c.at("choices")[0].value("text", "");
          }
        }
        check(text == "Hello!",
              "MIVR-14/15 completions SSE text mismatch: '" + text + "'");
      }
    }
  } catch (...) {
    cleanup();
    throw;
  }
  cleanup();
}

void test_test7_mock_spi_provider_compat(
    const std::filesystem::path &mock_provider_bin) {
  const auto tmp_root =
      std::filesystem::temp_directory_path() /
      ("ninfer_test7_mock_" +
       std::to_string(static_cast<std::uint64_t>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(tmp_root, ec);
  std::filesystem::create_directories(tmp_root);
  struct DirCleanup {
    std::filesystem::path dir;
    ~DirCleanup() {
      std::error_code cleanup_ec;
      std::filesystem::remove_all(dir, cleanup_ec);
    }
  } dir_cleanup{tmp_root};

#ifdef __APPLE__
  char **envp = *_NSGetEnviron();
#else
  char **envp = environ;
#endif

  auto spawn_mock = [&](const std::filesystem::path &sock_path) -> pid_t {
    std::string bin_str = mock_provider_bin.string();
    std::string flag_sock = "--socket";
    std::string sock_str = sock_path.string();
    std::string flag_exit = "--exit-on-disconnect";
    std::array<char *, 5> argv{{
        bin_str.data(),
        flag_sock.data(),
        sock_str.data(),
        flag_exit.data(),
        nullptr,
    }};
    pid_t child = -1;
    check(::posix_spawn(&child, bin_str.c_str(), nullptr, nullptr, argv.data(),
                        envp) == 0 &&
              child > 0,
          "TEST-7 failed to spawn mock_spi_provider");
    return child;
  };

  auto reap_child = [](pid_t &child) -> int {
    int status = 0;
    const auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (::waitpid(child, &status, WNOHANG) == 0 &&
           std::chrono::steady_clock::now() < dl) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (::waitpid(child, &status, WNOHANG) == 0) {
      (void)::kill(child, SIGKILL);
      (void)::waitpid(child, &status, 0);
    }
    child = -1;
    return status;
  };

  // 1) Overlong socket path (>= sizeof(sockaddr_un::sun_path)) must exit
  //    cleanly with non-zero exit code (no stack buffer overflow / signal).
  {
    const std::string overlong_name(sizeof(sockaddr_un::sun_path) + 24, 'a');
    const std::filesystem::path long_sock =
        std::filesystem::path("/tmp") / (overlong_name + ".sock");
    pid_t child = spawn_mock(long_sock);
    const int status = reap_child(child);
    check(WIFEXITED(status) && WEXITSTATUS(status) != 0,
          "TEST-7 overlong sun_path must exit cleanly with non-zero status");
  }

  // 2) path_overrides.artifact + string rolling_prefix_cache="on" + live
  //    socket protection (second instance cannot unlink live socket).
  {
    const auto model_dir_on = tmp_root / "model-on";
    std::filesystem::create_directories(model_dir_on);
    const auto sock_path =
        std::filesystem::path("/tmp") /
        ("nf-t7-a-" + std::to_string(static_cast<std::uint64_t>(::getpid())) +
         ".sock");
    std::filesystem::remove(sock_path, ec);

    pid_t child1 = spawn_mock(sock_path);
    int fd1 = -1;
    const auto cleanup_live = [&] {
      if (fd1 >= 0) {
        ::close(fd1);
        fd1 = -1;
      }
      if (child1 > 0) {
        (void)reap_child(child1);
      }
      std::error_code rm_ec;
      std::filesystem::remove(sock_path, rm_ec);
    };

    try {
      const auto dl =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (fd1 < 0 && std::chrono::steady_clock::now() < dl) {
        fd1 = raw_uds_connect(sock_path);
        if (fd1 < 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
      check(fd1 >= 0, "TEST-7 failed to connect to live mock_spi_provider");

      std::string buf;
      auto rpc = [&](int id, const char *method, const Json &params) {
        Json req = {{"id", id}, {"method", method}, {"params", params}};
        check(raw_uds_write_line(fd1, req.dump()), "TEST-7 RPC write failed");
        auto line = raw_uds_read_line(fd1, buf, 5000);
        check(line.has_value(), "TEST-7 RPC read timeout");
        auto resp = Json::parse(*line);
        check(resp.value("ok", false),
              std::string("TEST-7 RPC failed: ") + resp.dump());
        return resp.at("result");
      };

      (void)rpc(1, "hello", Json::object());
      // Exact m5_sidecar.py wire format: string rolling_prefix_cache +
      // artifact inside path_overrides.
      (void)rpc(2, "load",
                {{"model_dir", model_dir_on.string()},
                 {"max_concurrency", 2},
                 {"rolling_prefix_cache", "on"},
                 {"path_overrides",
                  {{"artifact", "/tmp/via-override.omlxm5"}}}});

      const Json audit =
          Json::parse(read_file_text(model_dir_on / ".mock_spi_audit.json"));
      check(audit.at("artifact") == "/tmp/via-override.omlxm5",
            "TEST-7 path_overrides.artifact not parsed: " + audit.dump());
      check(audit.at("rolling_prefix_cache").get<bool>() == true,
            "TEST-7 string rolling_prefix_cache='on' not parsed as true");

      // Spawn a second instance pointing at the SAME live socket: must refuse
      // to overwrite and leave instance #1 alive and serving.
      pid_t child2 = spawn_mock(sock_path);
      const int st2 = reap_child(child2);
      check(WIFEXITED(st2) && WEXITSTATUS(st2) != 0,
            "TEST-7 second mock_spi_provider must refuse live socket path");

      // Confirm live instance #1 is still reachable on sock_path.
      const int probe_fd = raw_uds_connect(sock_path);
      check(probe_fd >= 0,
            "TEST-7 live socket was unlinked/broken by second instance");
      ::close(probe_fd);

      (void)rpc(3, "shutdown", Json::object());
    } catch (...) {
      cleanup_live();
      throw;
    }
    cleanup_live();
  }

  // 3) String rolling_prefix_cache="off" parses as false.
  {
    const auto model_dir_off = tmp_root / "model-off";
    std::filesystem::create_directories(model_dir_off);
    const auto sock_path =
        std::filesystem::path("/tmp") /
        ("nf-t7-b-" + std::to_string(static_cast<std::uint64_t>(::getpid())) +
         ".sock");
    std::filesystem::remove(sock_path, ec);

    pid_t child = spawn_mock(sock_path);
    int fd = -1;
    const auto cleanup_off = [&] {
      if (fd >= 0) {
        ::close(fd);
        fd = -1;
      }
      if (child > 0) {
        (void)reap_child(child);
      }
      std::error_code rm_ec;
      std::filesystem::remove(sock_path, rm_ec);
    };
    try {
      const auto dl =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (fd < 0 && std::chrono::steady_clock::now() < dl) {
        fd = raw_uds_connect(sock_path);
        if (fd < 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
      check(fd >= 0, "TEST-7(off) failed to connect to mock_spi_provider");
      std::string buf;
      auto rpc = [&](int id, const char *method, const Json &params) {
        Json req = {{"id", id}, {"method", method}, {"params", params}};
        check(raw_uds_write_line(fd, req.dump()), "TEST-7(off) write failed");
        auto line = raw_uds_read_line(fd, buf, 5000);
        check(line.has_value(), "TEST-7(off) read timeout");
        auto resp = Json::parse(*line);
        check(resp.value("ok", false),
              std::string("TEST-7(off) RPC failed: ") + resp.dump());
        return resp.at("result");
      };
      (void)rpc(1, "load",
                {{"model_dir", model_dir_off.string()},
                 {"rolling_prefix_cache", "off"}});
      const Json audit =
          Json::parse(read_file_text(model_dir_off / ".mock_spi_audit.json"));
      check(audit.at("rolling_prefix_cache").get<bool>() == false,
            "TEST-7 string rolling_prefix_cache='off' not parsed as false");
      (void)rpc(2, "shutdown", Json::object());
    } catch (...) {
      cleanup_off();
      throw;
    }
    cleanup_off();
  }
}

void test_lora_cli_toml_precedence_and_observability(
    const std::filesystem::path &models_root,
    const std::filesystem::path &mock_bin) {
  const auto tmp_root =
      std::filesystem::temp_directory_path() /
      ("ninfer_lora_serve_test_" +
       std::to_string(static_cast<std::uint64_t>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(tmp_root, ec);
  std::filesystem::create_directories(tmp_root);

  const char *orig_home_raw = std::getenv("HOME");
  const std::optional<std::string> orig_home =
      orig_home_raw != nullptr ? std::optional<std::string>(orig_home_raw)
                               : std::nullopt;
  const auto fake_home = tmp_root / "home";
  std::filesystem::create_directories(fake_home);
  ::setenv("HOME", fake_home.c_str(), 1);

  struct EnvAndDirGuard {
    std::filesystem::path dir;
    std::optional<std::string> home;
    ~EnvAndDirGuard() {
      if (home.has_value()) {
        ::setenv("HOME", home->c_str(), 1);
      } else {
        ::unsetenv("HOME");
      }
      std::error_code rm_ec;
      std::filesystem::remove_all(dir, rm_ec);
    }
  } guard{tmp_root, orig_home};

  auto parse_cli_vec = [](std::vector<std::string> args) -> ServeConfig {
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (auto &a : args) {
      argv.push_back(a.data());
    }
    return parse_serve_config(static_cast<int>(argv.size()), argv.data());
  };

  // 1) --help documents --lora, --lora-adapter, --lora-mode, and [sidecar]
  {
    const std::string help = format_help_string();
    check(help.find("--lora") != std::string::npos &&
              help.find("--lora-adapter") != std::string::npos &&
              help.find("--lora-mode") != std::string::npos &&
              help.find("[sidecar]") != std::string::npos,
          "ninfer-metal-serve --help missing LoRA flags or [sidecar]");
  }

  // 2) CLI flags (--lora, --lora-adapter, --lora-mode) + ~/ expansion + default mode
  {
    const auto c1 = parse_cli_vec({
        "ninfer-metal-serve",
        "--no-default-config",
        "--model-dir",
        "~/models/qwen",
        "--lora",
        "~/adapters/peft-a",
    });
    check(c1.lora_adapter == (fake_home / "adapters/peft-a").string(),
          "--lora ~/ expansion mismatch: " + c1.lora_adapter);
    check(c1.lora_mode == "merge", "default --lora-mode must be merge");
    const auto lo1 = to_load_options(c1);
    check(lo1.lora_adapter == (fake_home / "adapters/peft-a").string() &&
              lo1.lora_mode == "merge",
          "to_load_options(ServeConfig) LoRA fields mismatch");

    const auto c2 = parse_cli_vec({
        "ninfer-metal-serve",
        "--no-default-config",
        "--model-dir=/tmp/qwen",
        "--lora-adapter=~/adapters/mlx-b",
        "--lora-mode=bypass",
    });
    check(c2.lora_adapter == (fake_home / "adapters/mlx-b").string() &&
              c2.lora_mode == "bypass",
          "--lora-adapter= / --lora-mode= parsing mismatch");
    const auto resolved2 = resolve_model_configs(c2);
    check(resolved2.size() == 1 &&
              resolved2[0].lora_adapter ==
                  (fake_home / "adapters/mlx-b").string() &&
              resolved2[0].lora_mode == "bypass",
          "resolve_model_configs LoRA fields mismatch");
    const auto lo2 = to_load_options(resolved2[0]);
    check(lo2.lora_adapter == (fake_home / "adapters/mlx-b").string() &&
              lo2.lora_mode == "bypass",
          "to_load_options(ModelServeConfig) LoRA fields mismatch");
  }

  // 3) Invalid --lora-mode rejected across CLI and all TOML sections
  {
    for (const char *bad_mode : {"invalid", "quant", ""}) {
      bool threw = false;
      try {
        (void)parse_cli_vec({
            "ninfer-metal-serve",
            "--no-default-config",
            "--lora-mode",
            bad_mode,
        });
      } catch (const std::invalid_argument &ex) {
        const std::string msg = ex.what();
        threw = msg.find("merge") != std::string::npos &&
                msg.find("bypass") != std::string::npos;
      }
      check(threw, std::string("CLI failed to reject invalid --lora-mode '") +
                       bad_mode + "'");
    }

    const std::array<std::string, 5> bad_tomls = {
        "[model]\nmodel_dir = \"/tmp/m\"\nlora_mode = \"bad\"\n",
        "[engine]\nlora_mode = \"bad\"\n",
        "[sidecar]\nlora_mode = \"bad\"\n",
        "[sidecar.models.\"m1\"]\nlora_mode = \"bad\"\n",
        "[[models]]\nmodel_dir = \"/tmp/m1\"\nlora_mode = \"bad\"\n",
    };
    for (std::size_t idx = 0; idx < bad_tomls.size(); ++idx) {
      const auto p = tmp_root / ("bad_lora_" + std::to_string(idx) + ".toml");
      {
        std::ofstream out(p);
        out << bad_tomls[idx];
      }
      bool threw = false;
      try {
        ServeConfig cfg;
        load_serve_toml_file(p.string(), cfg);
      } catch (const std::invalid_argument &ex) {
        const std::string msg = ex.what();
        threw = msg.find("merge") != std::string::npos &&
                msg.find("bypass") != std::string::npos;
      }
      check(threw, "TOML section index " + std::to_string(idx) +
                       " failed to reject invalid lora_mode");
    }
  }

  // 4) TOML sections ([model], [engine], [sidecar], [sidecar.models."<id>"],
  //    [[models]]) and exact precedence:
  //    CLI > per-model ([[models]] / [sidecar.models."<id>"]) > top-level TOML
  {
    // 4a) [model] sets lora_adapter, [engine] template has lora_adapter = ""
    //     -> [model]'s non-empty lora_adapter is preserved; [sidecar.models."m-one"]
    //     overrides top-level for m-one; CLI overrides [sidecar.models."m-one"]!
    const auto single_toml = tmp_root / "single_precedence.toml";
    {
      std::ofstream out(single_toml);
      out << "[model]\n"
          << "model_dir = \"~/models/m-one\"\n"
          << "lora_adapter = \"~/loras/from-model\"\n"
          << "lora_mode = \"merge\"\n\n"
          << "[engine]\n"
          << "lora_adapter = \"\"\n"
          << "lora_mode = \"merge\"\n\n"
          << "[sidecar]\n"
          << "lora_adapter = \"~/loras/from-sidecar\"\n"
          << "lora_mode = \"merge\"\n\n"
          << "[sidecar.models.\"m-one\"]\n"
          << "lora_adapter = \"~/loras/from-sidecar-model\"\n"
          << "lora_mode = \"bypass\"\n";
    }

    // Without CLI --lora: [sidecar.models."m-one"] wins over [sidecar]/[engine]/[model]
    const auto cfg_no_cli = parse_cli_vec({
        "ninfer-metal-serve",
        "--config",
        single_toml.string(),
    });
    check(cfg_no_cli.lora_adapter ==
              (fake_home / "loras/from-sidecar").string(),
          "top-level [sidecar] lora_adapter not captured");
    const auto res_no_cli = resolve_model_configs(cfg_no_cli);
    check(res_no_cli.size() == 1 &&
              res_no_cli[0].lora_adapter ==
                  (fake_home / "loras/from-sidecar-model").string() &&
              res_no_cli[0].lora_mode == "bypass",
          "[sidecar.models.\"m-one\"] failed to override top-level TOML: " +
              res_no_cli[0].lora_adapter + " mode=" + res_no_cli[0].lora_mode);
    const auto lo_no_cli = to_load_options(cfg_no_cli);
    check(lo_no_cli.lora_adapter ==
              (fake_home / "loras/from-sidecar-model").string() &&
              lo_no_cli.lora_mode == "bypass",
          "to_load_options(ServeConfig) did not apply [sidecar.models] override");

    // With CLI --lora and --lora-mode: CLI beats [sidecar.models."m-one"]!
    const auto cfg_cli_win = parse_cli_vec({
        "ninfer-metal-serve",
        "--config",
        single_toml.string(),
        "--lora",
        "~/loras/from-cli",
        "--lora-mode",
        "merge",
    });
    const auto res_cli_win = resolve_model_configs(cfg_cli_win);
    check(res_cli_win.size() == 1 &&
              res_cli_win[0].lora_adapter ==
                  (fake_home / "loras/from-cli").string() &&
              res_cli_win[0].lora_mode == "merge",
          "CLI --lora/--lora-mode failed to beat [sidecar.models]");

    // 4b) [model] non-empty lora_adapter preserved when [engine] has default ""
    const auto model_only_toml = tmp_root / "model_only_lora.toml";
    {
      std::ofstream out(model_only_toml);
      out << "[model]\n"
          << "model_dir = \"/tmp/plain-m\"\n"
          << "lora_adapter = \"~/loras/model-sec\"\n"
          << "lora_mode = \"bypass\"\n\n"
          << "[engine]\n"
          << "lora_adapter = \"\"\n";
    }
    const auto cfg_model_only = parse_cli_vec({
        "ninfer-metal-serve",
        "--config",
        model_only_toml.string(),
    });
    check(cfg_model_only.lora_adapter ==
              (fake_home / "loras/model-sec").string() &&
              cfg_model_only.lora_mode == "bypass",
          "[model] lora_adapter/lora_mode not preserved when [engine] has empty adapter");

    // 4c) Multi-model [[models]] + [sidecar.models."<id>"] + [engine] + CLI
    const auto multi_toml = tmp_root / "multi_lora.toml";
    {
      std::ofstream out(multi_toml);
      out << "[engine]\n"
          << "lora_adapter = \"~/loras/engine-global\"\n"
          << "lora_mode = \"merge\"\n\n"
          << "[sidecar.models.\"m-sidecar-ov\"]\n"
          << "lora_adapter = \"~/loras/sidecar-per-model\"\n"
          << "lora_mode = \"bypass\"\n\n"
          << "[[models]]\n"
          << "id = \"m-inherit-engine\"\n"
          << "model_dir = \"/tmp/m-inherit-engine\"\n\n"
          << "[[models]]\n"
          << "id = \"m-sidecar-ov\"\n"
          << "model_dir = \"/tmp/m-sidecar-ov\"\n\n"
          << "[[models]]\n"
          << "id = \"m-explicit\"\n"
          << "model_dir = \"/tmp/m-explicit\"\n"
          << "lora_adapter = \"~/loras/models-explicit\"\n"
          << "lora_mode = \"bypass\"\n";
    }
    const auto cfg_multi = parse_cli_vec({
        "ninfer-metal-serve",
        "--config",
        multi_toml.string(),
    });
    const auto m_list = resolve_model_configs(cfg_multi);
    check(m_list.size() == 3, "expected 3 models in multi_lora.toml");
    check(m_list[0].lora_adapter ==
              (fake_home / "loras/engine-global").string() &&
              m_list[0].lora_mode == "merge",
          "[[models]][0] failed to inherit [engine] LoRA defaults");
    check(m_list[1].lora_adapter ==
              (fake_home / "loras/sidecar-per-model").string() &&
              m_list[1].lora_mode == "bypass",
          "[[models]][1] failed to apply [sidecar.models.\"m-sidecar-ov\"]");
    check(m_list[2].lora_adapter ==
              (fake_home / "loras/models-explicit").string() &&
              m_list[2].lora_mode == "bypass",
          "[[models]][2] failed to apply explicit [[models]] LoRA fields");

    // CLI override propagates across all [[models]] entries (including clearing via --lora "")
    const auto cfg_multi_cli = parse_cli_vec({
        "ninfer-metal-serve",
        "--config",
        multi_toml.string(),
        "--lora-adapter",
        "~/loras/cli-all",
        "--lora-mode",
        "bypass",
    });
    for (const auto &m : resolve_model_configs(cfg_multi_cli)) {
      check(m.lora_adapter == (fake_home / "loras/cli-all").string() &&
                m.lora_mode == "bypass",
            "CLI --lora-adapter/--lora-mode did not override [[models]] entry " +
                m.id);
    }

    const auto cfg_multi_clear = parse_cli_vec({
        "ninfer-metal-serve",
        "--config",
        multi_toml.string(),
        "--lora",
        "",
    });
    for (const auto &m : resolve_model_configs(cfg_multi_clear)) {
      check(m.lora_adapter.empty(),
            "CLI --lora '' failed to clear TOML lora_adapter on " + m.id);
    }
  }

  // 5) HTTP /v1/models, /v1/models/:id, and /health observability + live reload_lora
  {
    auto assets = build_synthetic_assets();
    ServeConfig config;
    config.host = "127.0.0.1";
    config.port = 0;
    config.model_id = "qwen-lora-http";
    config.max_concurrency = 2;
    config.max_context = 4096;
    config.lora_adapter = "/tmp/peft-lora-v1";
    config.lora_mode = "bypass";

    LoadOptions load_opts;
    load_opts.max_concurrency = 2;
    load_opts.max_context = 4096;
    load_opts.lora_adapter = "/tmp/peft-lora-v1";
    load_opts.lora_mode = "bypass";

    FakeBackendOptions fake_opts;
    fake_opts.vocab_size = assets.vocab_size;

    std::shared_ptr<Engine> engine(Engine::create_fake(load_opts, fake_opts));
    auto router = std::make_shared<EngineRouter>(engine, assets.eos_ids);
    ServeHttpServer server(config, router, assets.tokenizer,
                           assets.chat_template);
    const int port = server.bind();
    check(port > 0, "LoRA HTTP test bind failed");
    std::thread server_thread([&] { server.listen_after_bind(); });
    const auto cleanup_http = [&] {
      server.stop();
      if (server_thread.joinable()) {
        server_thread.join();
      }
      router->stop();
    };

    try {
      httplib::Client client("127.0.0.1", port);
      client.set_read_timeout(10, 0);

      auto check_lora_endpoints = [&](const std::string &exp_adapter,
                                      const std::string &exp_mode,
                                      std::uint64_t exp_bytes,
                                      std::uint32_t exp_layers) {
        auto r_models = client.Get("/v1/models");
        check(r_models && r_models->status == 200, "/v1/models failed");
        const auto m0 = Json::parse(r_models->body).at("data").at(0);
        check(m0.at("lora_adapter") == exp_adapter &&
                  m0.at("lora_mode") == exp_mode &&
                  m0.at("lora_buffer_bytes").get<std::uint64_t>() == exp_bytes &&
                  m0.at("lora_adapted_layers").get<std::uint32_t>() ==
                      exp_layers,
              "/v1/models LoRA metadata mismatch: " + m0.dump());

        auto r_single = client.Get("/v1/models/qwen-lora-http");
        check(r_single && r_single->status == 200,
              "/v1/models/qwen-lora-http failed");
        const auto ms = Json::parse(r_single->body);
        check(ms.at("lora_adapter") == exp_adapter &&
                  ms.at("lora_mode") == exp_mode &&
                  ms.at("lora_buffer_bytes").get<std::uint64_t>() == exp_bytes &&
                  ms.at("lora_adapted_layers").get<std::uint32_t>() ==
                      exp_layers,
              "/v1/models/:id LoRA metadata mismatch: " + ms.dump());

        auto r_health = client.Get("/health");
        check(r_health && r_health->status == 200, "/health failed");
        const auto hj = Json::parse(r_health->body);
        check(hj.at("lora_adapter") == exp_adapter &&
                  hj.at("lora_mode") == exp_mode &&
                  hj.at("lora_buffer_bytes").get<std::uint64_t>() == exp_bytes &&
                  hj.at("lora_adapted_layers").get<std::uint32_t>() ==
                      exp_layers,
              "/health top-level LoRA metadata mismatch: " + hj.dump());
        const auto &hm0 = hj.at("models").at(0);
        check(hm0.at("lora_adapter") == exp_adapter &&
                  hm0.at("lora_mode") == exp_mode &&
                  hm0.at("lora_buffer_bytes").get<std::uint64_t>() ==
                      exp_bytes &&
                  hm0.at("lora_adapted_layers").get<std::uint32_t>() ==
                      exp_layers,
              "/health models[0] LoRA metadata mismatch: " + hm0.dump());
      };

      check_lora_endpoints("/tmp/peft-lora-v1", "bypass", 65536ULL, 1U);

      // Hot-swap bypass adapter while HTTP server is live!
      std::string err;
      check(router->reload_lora("/tmp/peft-lora-v2", "bypass", &err),
            "EngineRouter::reload_lora(v2) failed: " + err);
      check_lora_endpoints("/tmp/peft-lora-v2", "bypass", 65536ULL, 1U);

      // Unload bypass adapter to empty!
      check(router->reload_lora("", "bypass", &err),
            "EngineRouter::reload_lora('') failed: " + err);
      check_lora_endpoints("", "", 0ULL, 0U);
    } catch (...) {
      cleanup_http();
      throw;
    }
    cleanup_http();
  }

  // 6) UdsSidecarBackend SPI v2 load / stats / reload_lora payload forwarding
  {
    const std::string py_shebang =
        (::access("/usr/bin/python3", X_OK) == 0) ? "#!/usr/bin/python3\n"
                                                  : "#!/usr/bin/env python3\n";
    const auto lora_sc_script = tmp_root / "lora_spi_sidecar.py";
    const auto audit_json = tmp_root / "lora_spi_audit.json";
    {
      std::ofstream out(lora_sc_script);
      out << py_shebang
          << "import json, os, socket, sys, threading\n"
          << "AUDIT = " << Json(audit_json.string()).dump() << "\n"
          << "sock_path = None\n"
          << "args = sys.argv[1:]\n"
          << "for i, a in enumerate(args):\n"
          << "    if a == '--socket' and i + 1 < len(args):\n"
          << "        sock_path = args[i + 1]\n"
          << "if os.path.exists(sock_path):\n"
          << "    os.unlink(sock_path)\n"
          << "srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
          << "srv.bind(sock_path)\n"
          << "srv.listen(8)\n"
          << "state = {'lora_adapter': '', 'lora_mode': '', 'lora_buffer_bytes': 0, 'lora_adapted_layers': 0, 'loads': [], 'reloads': []}\n"
          << "stop_ev = threading.Event()\n"
          << "lock = threading.Lock()\n"
          << "def send_json(c, o):\n"
          << "    c.sendall((json.dumps(o) + '\\n').encode('utf-8'))\n"
          << "def save_audit():\n"
          << "    with open(AUDIT, 'w', encoding='utf-8') as f:\n"
          << "        json.dump(state, f)\n"
          << "def handle(c):\n"
          << "    buf = b''\n"
          << "    try:\n"
          << "        while not stop_ev.is_set():\n"
          << "            ch = c.recv(4096)\n"
          << "            if not ch:\n"
          << "                break\n"
          << "            buf += ch\n"
          << "            while b'\\n' in buf:\n"
          << "                line, buf = buf.split(b'\\n', 1)\n"
          << "                req = json.loads(line.decode('utf-8'))\n"
          << "                rid, m, p = req['id'], req['method'], req.get('params', {})\n"
          << "                if m == 'hello':\n"
          << "                    send_json(c, {'id': rid, 'ok': True, 'result': {'protocol_version': 2, 'provider_id': 'io.test.lora-sc', 'capabilities': ['lora', 'reload_lora', 'subscribe_events']}})\n"
          << "                elif m == 'load':\n"
          << "                    with lock:\n"
          << "                        la = p.get('lora_adapter', '')\n"
          << "                        lm = p.get('lora_mode', 'merge')\n"
          << "                        state['loads'].append({'lora_adapter': la, 'lora_mode': lm})\n"
          << "                        state['lora_adapter'] = la\n"
          << "                        state['lora_mode'] = lm if la else ''\n"
          << "                        state['lora_buffer_bytes'] = 131072 if la else 0\n"
          << "                        state['lora_adapted_layers'] = 28 if la else 0\n"
          << "                        save_audit()\n"
          << "                        send_json(c, {'id': rid, 'ok': True, 'result': dict(state)})\n"
          << "                elif m == 'reload_lora':\n"
          << "                    with lock:\n"
          << "                        la = p.get('lora_adapter', '')\n"
          << "                        lm = p.get('lora_mode', 'bypass')\n"
          << "                        state['reloads'].append({'lora_adapter': la, 'lora_mode': lm})\n"
          << "                        state['lora_adapter'] = la\n"
          << "                        state['lora_mode'] = lm if la else ''\n"
          << "                        state['lora_buffer_bytes'] = 262144 if la else 0\n"
          << "                        state['lora_adapted_layers'] = 14 if la else 0\n"
          << "                        save_audit()\n"
          << "                        send_json(c, {'id': rid, 'ok': True, 'result': dict(state)})\n"
          << "                elif m == 'stats':\n"
          << "                    with lock:\n"
          << "                        send_json(c, {'id': rid, 'ok': True, 'result': dict(state, backend_kind='lora_sc')})\n"
          << "                elif m == 'capacity':\n"
          << "                    send_json(c, {'id': rid, 'ok': True, 'result': {'max_concurrency': 1, 'active_requests': 0, 'queued_requests': 0, 'accepting': True}})\n"
          << "                elif m == 'subscribe_events':\n"
          << "                    send_json(c, {'id': rid, 'ok': True, 'result': {'subscribed': True}})\n"
          << "                    c.settimeout(0.1)\n"
          << "                    while not stop_ev.is_set():\n"
          << "                        try:\n"
          << "                            if not c.recv(1):\n"
          << "                                break\n"
          << "                        except socket.timeout:\n"
          << "                            continue\n"
          << "                        except Exception:\n"
          << "                            break\n"
          << "                    try:\n"
          << "                        send_json(c, {'event': {'kind': 'eof'}})\n"
          << "                    except Exception:\n"
          << "                        pass\n"
          << "                    return\n"
          << "                elif m == 'shutdown':\n"
          << "                    send_json(c, {'id': rid, 'ok': True, 'result': {'shutdown': True}})\n"
          << "                    stop_ev.set()\n"
          << "                    return\n"
          << "    finally:\n"
          << "        c.close()\n"
          << "srv.settimeout(0.1)\n"
          << "while not stop_ev.is_set():\n"
          << "    try:\n"
          << "        conn, _ = srv.accept()\n"
          << "        threading.Thread(target=handle, args=(conn,), daemon=True).start()\n"
          << "    except socket.timeout:\n"
          << "        continue\n"
          << "srv.close()\n";
    }
    ::chmod(lora_sc_script.c_str(), 0755);

    UdsSidecarOptions sc_opts;
    sc_opts.provider_id = "io.test.lora-sc";
    sc_opts.binary_path = lora_sc_script;
    sc_opts.load_options.model_dir = "/tmp/model-sc";
    sc_opts.load_options.lora_adapter = "/tmp/sc-lora-init";
    sc_opts.load_options.lora_mode = "bypass";

    UdsSidecarBackend sc(sc_opts);
    const auto st1 = sc.stats();
    check(st1.lora_adapter == "/tmp/sc-lora-init" &&
              st1.lora_mode == "bypass" && st1.lora_buffer_bytes == 131072ULL &&
              st1.lora_adapted_layers == 28U,
          "UdsSidecarBackend initial LoRA stats mismatch");

    std::string sc_err;
    check(sc.reload_lora("/tmp/sc-lora-hot", "bypass", &sc_err),
          "UdsSidecarBackend::reload_lora failed: " + sc_err);
    const auto st2 = sc.stats();
    check(st2.lora_adapter == "/tmp/sc-lora-hot" &&
              st2.lora_mode == "bypass" && st2.lora_buffer_bytes == 262144ULL &&
              st2.lora_adapted_layers == 14U,
          "UdsSidecarBackend hot-reloaded LoRA stats mismatch");
    sc.shutdown();

    const Json audit = Json::parse(read_file_text(audit_json));
    check(audit.at("loads").size() == 1 &&
              audit.at("loads")[0].at("lora_adapter") == "/tmp/sc-lora-init" &&
              audit.at("loads")[0].at("lora_mode") == "bypass",
          "UdsSidecarBackend load payload did not forward LoRA params: " +
              audit.dump());
    check(audit.at("reloads").size() == 1 &&
              audit.at("reloads")[0].at("lora_adapter") == "/tmp/sc-lora-hot" &&
              audit.at("reloads")[0].at("lora_mode") == "bypass",
          "UdsSidecarBackend reload_lora payload mismatch: " + audit.dump());
  }

  // 7) Launcher script start_ninfer_serve.sh --lora / NINFER_LORA_* & validation
  const auto launcher_script = std::filesystem::path(models_root).parent_path() /
                               "omlx-extend" / "scripts" /
                               "start_ninfer_serve.sh";
  if (std::filesystem::is_regular_file(launcher_script)) {
    const auto fake_serve_bin = tmp_root / "fake-ninfer-serve";
    {
      std::ofstream out(fake_serve_bin);
      out << "#!/bin/sh\nexit 0\n";
    }
    ::chmod(fake_serve_bin.c_str(), 0755);
    const auto dummy_model = tmp_root / "dummy-model";
    std::filesystem::create_directories(dummy_model);

    auto run_bash = [&](const std::string &cmd) -> std::pair<int, std::string> {
      FILE *pipe = ::popen((cmd + " 2>&1").c_str(), "r");
      check(pipe != nullptr, "popen failed");
      std::string output;
      std::array<char, 512> buf{};
      while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) !=
             nullptr) {
        output += buf.data();
      }
      const int rc = ::pclose(pipe);
      const int code = WIFEXITED(rc) ? WEXITSTATUS(rc) : rc;
      return {code, output};
    };

    // Help output
    {
      const auto [code, out] =
          run_bash("bash \"" + launcher_script.string() + "\" --help");
      check(code == 0 && out.find("--lora") != std::string::npos &&
                out.find("--lora-mode") != std::string::npos,
            "start_ninfer_serve.sh --help missing --lora/--lora-mode: " + out);
    }
    // Env vars NINFER_LORA_ADAPTER & NINFER_LORA_MODE with ~/ expansion
    {
      const auto [code, out] = run_bash(
          "NINFER_METAL_SERVE_BIN=\"" + fake_serve_bin.string() +
          "\" NINFER_LORA_ADAPTER=\"~/env-lora\" NINFER_LORA_MODE=bypass "
          "bash \"" +
          launcher_script.string() + "\" --dry-run --no-default-config \"" +
          dummy_model.string() + "\"");
      check(code == 0 &&
                out.find("--lora " + (fake_home / "env-lora").string()) !=
                    std::string::npos &&
                out.find("--lora-mode bypass") != std::string::npos,
            "start_ninfer_serve.sh env NINFER_LORA_* failed: " + out);
    }
    // CLI overrides env vars
    {
      const auto [code, out] = run_bash(
          "NINFER_METAL_SERVE_BIN=\"" + fake_serve_bin.string() +
          "\" NINFER_LORA_ADAPTER=\"~/env-lora\" NINFER_LORA_MODE=bypass "
          "bash \"" +
          launcher_script.string() +
          "\" --dry-run --no-default-config --lora-adapter \"~/cli-lora\" "
          "--lora-mode merge \"" +
          dummy_model.string() + "\"");
      check(code == 0 &&
                out.find("--lora-adapter " +
                         (fake_home / "cli-lora").string()) !=
                    std::string::npos &&
                out.find("--lora-mode merge") != std::string::npos &&
                out.find("env-lora") == std::string::npos,
            "start_ninfer_serve.sh CLI failed to override NINFER_LORA_*: " +
                out);
    }
    // Invalid --lora-mode rejected with clear error
    {
      const auto [code, out] = run_bash(
          "NINFER_METAL_SERVE_BIN=\"" + fake_serve_bin.string() + "\" bash \"" +
          launcher_script.string() + "\" --dry-run --lora-mode invalid \"" +
          dummy_model.string() + "\"");
      check(code != 0 && out.find("merge") != std::string::npos &&
                out.find("bypass") != std::string::npos,
            "start_ninfer_serve.sh failed to reject invalid --lora-mode: " +
                out);
    }
  }

  // 8) Verify sibling CLI binaries (m5rt_runner & mseries_backendd) --help and
  //    invalid --lora-mode rejection if built alongside mock_bin.
  if (!mock_bin.empty()) {
    const auto bin_dir = mock_bin.parent_path();
    auto run_cmd = [](const std::string &cmd) -> std::pair<int, std::string> {
      FILE *pipe = ::popen((cmd + " 2>&1").c_str(), "r");
      check(pipe != nullptr, "popen failed");
      std::string output;
      std::array<char, 512> buf{};
      while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) !=
             nullptr) {
        output += buf.data();
      }
      const int rc = ::pclose(pipe);
      const int code = WIFEXITED(rc) ? WEXITSTATUS(rc) : rc;
      return {code, output};
    };

    const auto runner_bin = bin_dir / "m5rt_runner";
    if (::access(runner_bin.c_str(), X_OK) == 0) {
      const auto [hc, hout] = run_cmd("\"" + runner_bin.string() + "\" --help");
      check(hc == 0 && hout.find("--lora") != std::string::npos &&
                hout.find("--lora-adapter") != std::string::npos &&
                hout.find("--lora-mode") != std::string::npos,
            "m5rt_runner --help missing LoRA flags: " + hout);
      const auto [bc, bout] =
          run_cmd("\"" + runner_bin.string() + "\" --lora-mode invalid");
      check(bc != 0 && bout.find("merge") != std::string::npos &&
                bout.find("bypass") != std::string::npos,
            "m5rt_runner failed to reject invalid --lora-mode: " + bout);
    }

    const auto daemon_bin = bin_dir / "mseries_backendd";
    if (::access(daemon_bin.c_str(), X_OK) == 0) {
      const auto [hc, hout] = run_cmd("\"" + daemon_bin.string() + "\" --help");
      check(hc == 0 && hout.find("--lora") != std::string::npos &&
                hout.find("--lora-adapter") != std::string::npos &&
                hout.find("--lora-mode") != std::string::npos,
            "mseries_backendd --help missing LoRA flags: " + hout);
      const auto [bc, bout] = run_cmd("\"" + daemon_bin.string() +
                                      "\" --lora-mode invalid --socket /tmp/x");
      check(bc != 0 && bout.find("merge") != std::string::npos &&
                bout.find("bypass") != std::string::npos,
            "mseries_backendd failed to reject invalid --lora-mode: " + bout);
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    test_output_parser_utf8_and_formats();
    std::cout << "[serve_e2e] PASS: OutputParser UTF-8 & thinking states\n";

    test_toml_and_cli_config();
    std::cout << "[serve_e2e] PASS: TOML config & CLI overrides\n";

    test_default_config_auto_discovery_tilde_and_auto_spec();
    std::cout << "[serve_e2e] PASS: Default ninfer-metal.toml auto-discovery, "
                 "~/ expansion, speculative=auto & CLI overrides\n";

    const std::filesystem::path models_root =
        argc > 1 ? std::filesystem::path(argv[1])
                 : std::filesystem::path("models");

    test_multi_model_config_and_aliases(models_root);
    std::cout << "[serve_e2e] PASS: Multi-model TOML [[models]], aliases & "
                 "--metallib-dir/--providers-dir CLI\n";

    test_mivr9_config_cli_and_budgets();
    std::cout << "[serve_e2e] PASS: MIVR-9 [[models]] CLI override propagation "
                 "& [server] vs [engine] memory budget isolation\n";

    test_mivr10_provider_auto_routing_no_hijack();
    std::cout << "[serve_e2e] PASS: MIVR-10 resolve_provider(auto) built-in "
                 ".omlxm5 no-hijack & required extension validation\n";

    test_http_server_e2e();
    std::cout << "[serve_e2e] PASS: HTTP E2E (/health, /metrics, /v1/models, "
                 "/v1/completions, /v1/chat/completions sync+SSE+<think>+XML&JSON "
                 "tools+stop, B=4 demux, auth 401/400/404)\n";

    test_mivr5_sampling_defaults_and_validation();
    std::cout << "[serve_e2e] PASS: MIVR-5 OpenAI temperature>0 default "
                 "top_k=64 & min_p/top_k HTTP 400 validation\n";

    test_streaming_backend_failure_sse();
    std::cout << "[serve_e2e] PASS: Streaming SSE backend failure error events "
                 "(/v1/chat/completions & /v1/completions)\n";

    test_mivr14_mivr15_mivr16_p2_fixes();
    std::cout << "[serve_e2e] PASS: MIVR-14/15/16 padding output tokens, "
                 "SSE Keep-Alive connection reuse & bare trailing <tool_call>\n";

    const std::filesystem::path opt_mock_bin =
        (argc > 2 && std::string_view(argv[2]).size() > 0)
            ? std::filesystem::path(argv[2])
            : std::filesystem::path{};
    test_lora_cli_toml_precedence_and_observability(models_root, opt_mock_bin);
    std::cout << "[serve_e2e] PASS: LoRA CLI flags, TOML precedence across "
                 "[model]/[engine]/[sidecar]/[sidecar.models]/[[models]], "
                 "/v1/models + /health observability & SPI v2 forwarding\n";

    if (argc > 2 && std::string_view(argv[2]).size() > 0) {
      const std::filesystem::path mock_bin(argv[2]);
      test_multi_microkernel_parallel_e2e(mock_bin);
      std::cout
          << "[serve_e2e] PASS: Multi-microkernel parallel E2E "
             "(ProviderRegistry discovery + O(1) no-hijack routing + inproc & "
             "live UdsSidecarBackend concurrent sync/SSE on ONE port + UMA "
             "budget + gpu_contended true->false + trim_cache + 0700 socket "
             "dir + zero orphans)\n";

      test_sidecar_crash_and_rpc_resilience(mock_bin);
      std::cout << "[serve_e2e] PASS: MIVR-6/MIVR-8 sidecar crash recovery, "
                   "degraded /health & /metrics, RPC timeout & id validation\n";

      test_test6_mock_spi_provider_protocol(mock_bin);
      std::cout << "[serve_e2e] PASS: TEST-6 mock_spi_provider concurrent "
                   "interleaving, multi-token stop holdback & shutdown aborted "
                   "before eof\n";

      test_test7_mock_spi_provider_compat(mock_bin);
      std::cout << "[serve_e2e] PASS: TEST-7 mock_spi_provider path_overrides "
                   "artifact, string rolling_prefix_cache, live socket guard & "
                   "sun_path bounds\n";
    }

    test_real_local_qwen_models(models_root);

    std::cout << "[serve_e2e] ALL TESTS PASSED!\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "[serve_e2e] FAIL: " << ex.what() << '\n';
    return 1;
  }
}
