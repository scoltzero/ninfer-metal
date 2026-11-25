#pragma once

#include "serve/byte_span.h"
#include "serve/jinja.h"
#include "serve/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace omlx::m5::serve {

struct ChatTemplateMetadata {
  std::string source_name;
  std::string bos_token;
  std::string eos_token;
  std::vector<std::int32_t> eos_token_ids;
  std::int32_t max_position_embeddings = 0;
};

struct ChatRenderResult {
  std::string prompt_text;
  std::vector<ByteSpan> literal_spans;
  bool prompt_opens_think = false;
};

class ChatTemplate {
public:
  static ChatTemplate load(const std::filesystem::path &model_dir,
                           const Tokenizer &tokenizer,
                           const std::string &override_template_path = "");

  static ChatTemplate from_source(std::string source, std::string source_name,
                                  ChatTemplateMetadata metadata = {});

  [[nodiscard]] ChatRenderResult
  render(const nlohmann::json &messages,
         const nlohmann::json &tools = nlohmann::json(),
         const nlohmann::json &extra_kwargs = nlohmann::json::object(),
         bool add_generation_prompt = true) const;

  [[nodiscard]] const ChatTemplateMetadata &metadata() const noexcept {
    return metadata_;
  }

  static std::string default_chatml_template();

private:
  ChatTemplate(std::shared_ptr<JinjaTemplate> compiled,
               ChatTemplateMetadata metadata);

  std::shared_ptr<JinjaTemplate> compiled_;
  ChatTemplateMetadata metadata_;
};

} // namespace omlx::m5::serve
