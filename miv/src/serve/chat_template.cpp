#include "serve/chat_template.h"

#include <algorithm>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace omlx::m5::serve {
namespace {

using Json = nlohmann::json;
using OrderedJson = nlohmann::ordered_json;

std::optional<std::string> read_optional_file(const std::filesystem::path &p) {
  std::ifstream input(p, std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    return std::nullopt;
  }
  std::ostringstream out;
  out << input.rdbuf();
  return out.str();
}

std::string read_required_file(const std::filesystem::path &p,
                               std::string_view label) {
  auto content = read_optional_file(p);
  if (!content.has_value()) {
    throw std::runtime_error("failed to read " + std::string(label) + ": " +
                             p.string());
  }
  return std::move(*content);
}

std::optional<Json> read_optional_json(const std::filesystem::path &p) {
  auto text = read_optional_file(p);
  if (!text.has_value() || text->empty()) {
    return std::nullopt;
  }
  try {
    return Json::parse(*text);
  } catch (const Json::exception &err) {
    throw std::runtime_error("failed to parse " + p.string() + ": " +
                             err.what());
  }
}

std::string extract_token_string(const Json &obj, const char *key) {
  if (!obj.is_object() || !obj.contains(key) || obj.at(key).is_null()) {
    return {};
  }
  const auto &val = obj.at(key);
  if (val.is_string()) {
    return val.get<std::string>();
  }
  if (val.is_object() && val.contains("content") &&
      val.at("content").is_string()) {
    return val.at("content").get<std::string>();
  }
  return {};
}

void add_unique_eos_id(std::vector<std::int32_t> &ids, std::int32_t id,
                       const Tokenizer &tokenizer) {
  if (id < 0 || !tokenizer.is_valid_token(id)) {
    return;
  }
  if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
    ids.push_back(id);
  }
}

void extract_eos_ids_from_value(const Json &val, std::vector<std::int32_t> &ids,
                                const Tokenizer &tokenizer) {
  if (val.is_number_integer()) {
    const auto v = val.get<std::int64_t>();
    if (v >= 0 && v <= std::numeric_limits<std::int32_t>::max()) {
      add_unique_eos_id(ids, static_cast<std::int32_t>(v), tokenizer);
    }
  } else if (val.is_array()) {
    for (const auto &entry : val) {
      if (entry.is_number_integer()) {
        const auto v = entry.get<std::int64_t>();
        if (v >= 0 && v <= std::numeric_limits<std::int32_t>::max()) {
          add_unique_eos_id(ids, static_cast<std::int32_t>(v), tokenizer);
        }
      }
    }
  }
}

std::optional<std::string>
extract_chat_template_string(const Json &tokenizer_config) {
  if (!tokenizer_config.is_object() ||
      !tokenizer_config.contains("chat_template") ||
      tokenizer_config.at("chat_template").is_null()) {
    return std::nullopt;
  }
  const auto &ct = tokenizer_config.at("chat_template");
  if (ct.is_string()) {
    const auto text = ct.get<std::string>();
    if (!text.empty()) {
      return text;
    }
    return std::nullopt;
  }
  if (ct.is_array()) {
    std::optional<std::string> fallback;
    for (const auto &entry : ct) {
      if (!entry.is_object() || !entry.contains("template") ||
          !entry.at("template").is_string()) {
        continue;
      }
      const auto name = entry.value("name", std::string{});
      const auto tmpl = entry.at("template").get<std::string>();
      if (tmpl.empty()) {
        continue;
      }
      if (name == "default") {
        return tmpl;
      }
      if (!fallback.has_value()) {
        fallback = tmpl;
      }
    }
    return fallback;
  }
  return std::nullopt;
}

OrderedJson to_ordered_json(const Json &input) {
  if (input.is_null()) {
    return nullptr;
  }
  if (input.is_boolean()) {
    return input.get<bool>();
  }
  if (input.is_number_unsigned()) {
    return input.get<std::uint64_t>();
  }
  if (input.is_number_integer()) {
    return input.get<std::int64_t>();
  }
  if (input.is_number_float()) {
    return input.get<double>();
  }
  if (input.is_string()) {
    return input.get<std::string>();
  }
  if (input.is_array()) {
    OrderedJson arr = OrderedJson::array();
    for (const auto &item : input) {
      arr.push_back(to_ordered_json(item));
    }
    return arr;
  }
  if (input.is_object()) {
    OrderedJson obj = OrderedJson::object();
    for (const auto &item : input.items()) {
      obj[item.key()] = to_ordered_json(item.value());
    }
    return obj;
  }
  return nullptr;
}

OrderedJson normalize_messages_for_template(const Json &messages) {
  if (!messages.is_array()) {
    throw std::invalid_argument("messages must be a JSON array");
  }
  OrderedJson out = OrderedJson::array();
  for (const auto &msg : messages) {
    if (!msg.is_object()) {
      throw std::invalid_argument("each message must be a JSON object");
    }
    OrderedJson norm = to_ordered_json(msg);
    if (!norm.contains("content") || norm["content"].is_null()) {
      norm["content"] = "";
    }
    // OpenAI API passes tool_calls[].function.arguments as a JSON-encoded
    // string, while HuggingFace/Qwen Jinja templates iterate `arguments|items`.
    if (norm.contains("tool_calls") && norm["tool_calls"].is_array()) {
      for (auto &tc : norm["tool_calls"]) {
        if (!tc.is_object()) {
          continue;
        }
        if (tc.contains("function") && tc["function"].is_object()) {
          auto &fn = tc["function"];
          if (fn.contains("arguments") && fn["arguments"].is_string()) {
            const std::string raw = fn["arguments"].get<std::string>();
            if (!raw.empty()) {
              try {
                auto parsed = OrderedJson::parse(raw);
                if (parsed.is_object()) {
                  fn["arguments"] = std::move(parsed);
                }
              } catch (...) {
                // Leave as string if not valid JSON object.
              }
            } else {
              fn["arguments"] = OrderedJson::object();
            }
          }
        }
      }
    }
    out.push_back(std::move(norm));
  }
  return out;
}

bool prompt_ends_with_open_think(std::string_view prompt) noexcept {
  // Trim trailing horizontal/vertical newlines after checking the tag.
  std::string_view trimmed = prompt;
  while (!trimmed.empty() &&
         (trimmed.back() == '\n' || trimmed.back() == '\r')) {
    trimmed.remove_suffix(1);
  }
  return trimmed.ends_with("<think>");
}

} // namespace

std::string ChatTemplate::default_chatml_template() {
  return "{%- for message in messages -%}\n"
         "<|im_start|>{{ message.role }}\n"
         "{{ message.content }}<|im_end|>\n"
         "{%- endfor -%}\n"
         "{%- if add_generation_prompt -%}\n"
         "<|im_start|>assistant\n"
         "{%- endif -%}";
}

ChatTemplate::ChatTemplate(std::shared_ptr<JinjaTemplate> compiled,
                           ChatTemplateMetadata metadata)
    : compiled_(std::move(compiled)), metadata_(std::move(metadata)) {}

ChatTemplate ChatTemplate::from_source(std::string source,
                                       std::string source_name,
                                       ChatTemplateMetadata metadata) {
  if (metadata.source_name.empty()) {
    metadata.source_name = source_name;
  }
  auto compiled =
      std::make_shared<JinjaTemplate>(std::move(source), std::move(source_name));
  return ChatTemplate(std::move(compiled), std::move(metadata));
}

ChatTemplate ChatTemplate::load(const std::filesystem::path &model_dir,
                                const Tokenizer &tokenizer,
                                const std::string &override_template_path) {
  ChatTemplateMetadata metadata;

  const auto tokenizer_config =
      read_optional_json(model_dir / "tokenizer_config.json");
  const auto config_json = read_optional_json(model_dir / "config.json");
  const auto generation_config =
      read_optional_json(model_dir / "generation_config.json");

  if (tokenizer_config.has_value()) {
    metadata.bos_token = extract_token_string(*tokenizer_config, "bos_token");
    metadata.eos_token = extract_token_string(*tokenizer_config, "eos_token");
    if (!metadata.eos_token.empty()) {
      if (const auto id = tokenizer.lookup_token_id(metadata.eos_token);
          id.has_value()) {
        add_unique_eos_id(metadata.eos_token_ids, *id, tokenizer);
      }
    }
    if (tokenizer_config->contains("eos_token_id")) {
      extract_eos_ids_from_value(tokenizer_config->at("eos_token_id"),
                                 metadata.eos_token_ids, tokenizer);
    }
  }

  if (config_json.has_value()) {
    if (config_json->contains("eos_token_id")) {
      extract_eos_ids_from_value(config_json->at("eos_token_id"),
                                 metadata.eos_token_ids, tokenizer);
    }
    if (config_json->contains("max_position_embeddings") &&
        config_json->at("max_position_embeddings").is_number_integer()) {
      metadata.max_position_embeddings =
          config_json->at("max_position_embeddings").get<std::int32_t>();
    }
    if (config_json->contains("text_config") &&
        config_json->at("text_config").is_object()) {
      const auto &tc = config_json->at("text_config");
      if (tc.contains("eos_token_id")) {
        extract_eos_ids_from_value(tc.at("eos_token_id"),
                                   metadata.eos_token_ids, tokenizer);
      }
      if (tc.contains("max_position_embeddings") &&
          tc.at("max_position_embeddings").is_number_integer()) {
        metadata.max_position_embeddings =
            tc.at("max_position_embeddings").get<std::int32_t>();
      }
    }
  }

  if (generation_config.has_value() &&
      generation_config->contains("eos_token_id")) {
    extract_eos_ids_from_value(generation_config->at("eos_token_id"),
                               metadata.eos_token_ids, tokenizer);
  }

  // Fallback EOS detection from common special tokens if configs omitted them.
  if (metadata.eos_token_ids.empty()) {
    for (const std::string_view candidate :
         {"<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"}) {
      if (const auto id = tokenizer.lookup_token_id(candidate);
          id.has_value()) {
        add_unique_eos_id(metadata.eos_token_ids, *id, tokenizer);
        if (metadata.eos_token.empty()) {
          metadata.eos_token = std::string(candidate);
        }
      }
    }
  }

  // Chat template priority:
  // 1) --chat-template PATH
  // 2) model_dir/chat_template.jinja
  // 3) tokenizer_config.json["chat_template"]
  // 4) ChatML fallback
  std::string template_source;
  std::string source_name;
  if (!override_template_path.empty()) {
    const std::filesystem::path p(override_template_path);
    template_source = read_required_file(p, "chat template");
    source_name = p.string();
  } else if (auto sibling = read_optional_file(model_dir / "chat_template.jinja");
             sibling.has_value() && !sibling->empty()) {
    template_source = std::move(*sibling);
    source_name = (model_dir / "chat_template.jinja").string();
  } else if (tokenizer_config.has_value()) {
    if (auto embedded = extract_chat_template_string(*tokenizer_config);
        embedded.has_value()) {
      template_source = std::move(*embedded);
      source_name = (model_dir / "tokenizer_config.json").string();
    }
  }

  if (template_source.empty()) {
    template_source = default_chatml_template();
    source_name = "chatml_fallback";
  }

  metadata.source_name = source_name;
  return from_source(std::move(template_source), std::move(source_name),
                     std::move(metadata));
}

ChatRenderResult ChatTemplate::render(const Json &messages, const Json &tools,
                                      const Json &extra_kwargs,
                                      bool add_generation_prompt) const {
  OrderedJson context = OrderedJson::object();
  context["messages"] = normalize_messages_for_template(messages);
  context["add_generation_prompt"] = add_generation_prompt;
  if (!metadata_.bos_token.empty()) {
    context["bos_token"] = metadata_.bos_token;
  }
  if (!metadata_.eos_token.empty()) {
    context["eos_token"] = metadata_.eos_token;
  }
  if (tools.is_array() && !tools.empty()) {
    context["tools"] = to_ordered_json(tools);
  }
  if (extra_kwargs.is_object()) {
    for (const auto &item : extra_kwargs.items()) {
      if (item.key() == "messages" || item.key() == "add_generation_prompt") {
        continue;
      }
      context[item.key()] = to_ordered_json(item.value());
    }
  }

  const std::vector<std::string> control_vars{"bos_token", "eos_token"};
  TemplateRenderOptions opts;
  opts.control_variables = control_vars;

  TemplateOutput rendered = compiled_->render(context, opts);
  ChatRenderResult result;
  result.prompt_opens_think = prompt_ends_with_open_think(rendered.text);
  result.prompt_text = std::move(rendered.text);
  result.literal_spans = std::move(rendered.literal_spans);
  return result;
}

} // namespace omlx::m5::serve
