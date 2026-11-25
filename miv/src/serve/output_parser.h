#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace omlx::m5::serve {

struct ParsedToolCall {
  std::size_t index = 0;
  std::string id;
  std::string name;
  std::string arguments;
};

struct OutputDelta {
  std::string reasoning;
  std::string content;
  std::vector<ParsedToolCall> tool_calls;
  bool stopped_by_string = false;

  [[nodiscard]] bool empty() const noexcept {
    return reasoning.empty() && content.empty() && tool_calls.empty();
  }
};

struct OutputParserOptions {
  bool enable_thinking_split = true;
  bool initial_in_thinking = false;
  bool enable_tool_parsing = true;
  std::vector<std::string> stop_strings;
  std::string tool_call_id_prefix = "call_";
  // Optional map: function_name -> (parameter_name -> JSON schema type string)
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      tool_param_types;
};

class OutputParser {
public:
  explicit OutputParser(OutputParserOptions options = {});

  OutputDelta feed_bytes(std::string_view raw_bytes);
  OutputDelta finish();

  [[nodiscard]] bool stopped_by_string() const noexcept {
    return stopped_by_string_;
  }
  [[nodiscard]] std::size_t tool_call_count() const noexcept {
    return next_tool_index_;
  }

private:
  enum class State {
    Content,
    Thinking,
    ToolCall,
  };

  std::string decode_utf8(std::string_view raw_bytes, bool flushing);
  std::string apply_stop_strings(std::string_view valid_utf8, bool flushing);
  void process_channels(std::string_view text, bool flushing, OutputDelta &out);
  void emit_content(std::string_view text, OutputDelta &out);
  void emit_reasoning(std::string_view text, OutputDelta &out);
  bool try_parse_tool_call(std::string_view body, ParsedToolCall &parsed) const;
  bool try_parse_json_tool_call(std::string_view body,
                                ParsedToolCall &parsed) const;
  bool try_parse_xml_tool_call(std::string_view body,
                               ParsedToolCall &parsed) const;

  OutputParserOptions options_;
  State state_ = State::Content;
  bool stopped_by_string_ = false;
  bool finished_ = false;
  bool skip_leading_think_newline_ = false;
  int skip_post_think_newlines_ = 0;

  std::string utf8_pending_;
  std::string stop_buffer_;
  std::string channel_buffer_;
  std::string tool_buffer_;
  std::size_t next_tool_index_ = 0;
};

} // namespace omlx::m5::serve
