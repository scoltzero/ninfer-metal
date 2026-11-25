#include "serve/output_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace omlx::m5::serve {
namespace {

constexpr std::string_view kThinkOpen = "<think>";
constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kToolOpen = "<tool_call>";
constexpr std::string_view kToolClose = "</tool_call>";
constexpr std::string_view kFunctionOpen = "<function=";
constexpr std::string_view kFunctionClose = "</function>";
constexpr std::string_view kParamOpen = "<parameter=";
constexpr std::string_view kParamClose = "</parameter>";
constexpr std::string_view kReplacementChar = "\xEF\xBF\xBD";

std::size_t longest_suffix_matching_prefix(std::string_view text,
                                           std::string_view needle) noexcept {
  if (needle.size() <= 1 || text.empty()) {
    return 0;
  }
  const std::size_t max_len = std::min(text.size(), needle.size() - 1);
  for (std::size_t len = max_len; len > 0; --len) {
    if (text.substr(text.size() - len) == needle.substr(0, len)) {
      return len;
    }
  }
  return 0;
}

std::string_view trim_ascii(std::string_view text) noexcept {
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.front()))) {
    text.remove_prefix(1);
  }
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.back()))) {
    text.remove_suffix(1);
  }
  return text;
}

std::string_view strip_single_edge_newlines(std::string_view text) noexcept {
  if (text.starts_with("\r\n")) {
    text.remove_prefix(2);
  } else if (text.starts_with('\n')) {
    text.remove_prefix(1);
  }
  if (text.ends_with("\r\n")) {
    text.remove_suffix(2);
  } else if (text.ends_with('\n')) {
    text.remove_suffix(1);
  }
  return text;
}

bool is_valid_utf8_sequence(std::string_view bytes, std::size_t &seq_len,
                            bool &incomplete) noexcept {
  incomplete = false;
  seq_len = 1;
  if (bytes.empty()) {
    return false;
  }
  const auto b0 = static_cast<unsigned char>(bytes[0]);
  if (b0 < 0x80U) {
    seq_len = 1;
    return true;
  }
  if (b0 >= 0xC2U && b0 <= 0xDFU) {
    if (bytes.size() < 2) {
      incomplete = true;
      return false;
    }
    const auto b1 = static_cast<unsigned char>(bytes[1]);
    if ((b1 & 0xC0U) != 0x80U) {
      return false;
    }
    seq_len = 2;
    return true;
  }
  if (b0 >= 0xE0U && b0 <= 0xEFU) {
    if (bytes.size() >= 2) {
      const auto b1 = static_cast<unsigned char>(bytes[1]);
      if ((b1 & 0xC0U) != 0x80U) {
        return false;
      }
      if (b0 == 0xE0U && b1 < 0xA0U) {
        return false; // Overlong
      }
      if (b0 == 0xEDU && b1 >= 0xA0U) {
        return false; // UTF-16 surrogate
      }
    }
    if (bytes.size() < 3) {
      incomplete = true;
      return false;
    }
    const auto b2 = static_cast<unsigned char>(bytes[2]);
    if ((b2 & 0xC0U) != 0x80U) {
      return false;
    }
    seq_len = 3;
    return true;
  }
  if (b0 >= 0xF0U && b0 <= 0xF4U) {
    if (bytes.size() >= 2) {
      const auto b1 = static_cast<unsigned char>(bytes[1]);
      if ((b1 & 0xC0U) != 0x80U) {
        return false;
      }
      if (b0 == 0xF0U && b1 < 0x90U) {
        return false; // Overlong
      }
      if (b0 == 0xF4U && b1 > 0x8FU) {
        return false; // > U+10FFFF
      }
    }
    if (bytes.size() >= 3) {
      const auto b2 = static_cast<unsigned char>(bytes[2]);
      if ((b2 & 0xC0U) != 0x80U) {
        return false;
      }
    }
    if (bytes.size() < 4) {
      incomplete = true;
      return false;
    }
    const auto b3 = static_cast<unsigned char>(bytes[3]);
    if ((b3 & 0xC0U) != 0x80U) {
      return false;
    }
    seq_len = 4;
    return true;
  }
  return false;
}

nlohmann::ordered_json parse_xml_param_value(std::string_view raw_value,
                                             std::string_view schema_type) {
  const std::string_view value = strip_single_edge_newlines(raw_value);
  if (schema_type == "string" || value.empty()) {
    return std::string(value);
  }
  const std::string_view trimmed = trim_ascii(value);
  if (trimmed.empty()) {
    return std::string(value);
  }
  const bool looks_json_literal =
      trimmed == "true" || trimmed == "false" || trimmed == "null" ||
      trimmed.front() == '{' || trimmed.front() == '[' ||
      trimmed.front() == '-' ||
      std::isdigit(static_cast<unsigned char>(trimmed.front()));
  if (looks_json_literal || schema_type == "integer" ||
      schema_type == "number" || schema_type == "boolean" ||
      schema_type == "object" || schema_type == "array") {
    try {
      auto parsed = nlohmann::ordered_json::parse(trimmed);
      if (schema_type.empty() ||
          (schema_type == "boolean" && parsed.is_boolean()) ||
          (schema_type == "integer" && parsed.is_number_integer()) ||
          (schema_type == "number" && parsed.is_number()) ||
          (schema_type == "object" && parsed.is_object()) ||
          (schema_type == "array" && parsed.is_array())) {
        return parsed;
      }
    } catch (...) {
      // Fallback to raw string.
    }
  }
  return std::string(value);
}

} // namespace

OutputParser::OutputParser(OutputParserOptions options)
    : options_(std::move(options)) {
  // Drop any empty stop strings defensively.
  std::erase_if(options_.stop_strings,
                [](const std::string &s) { return s.empty(); });
  if (options_.enable_thinking_split && options_.initial_in_thinking) {
    state_ = State::Thinking;
  }
}

std::string OutputParser::decode_utf8(std::string_view raw_bytes,
                                      bool flushing) {
  utf8_pending_.append(raw_bytes);
  std::string out;
  out.reserve(utf8_pending_.size());
  std::size_t pos = 0;
  while (pos < utf8_pending_.size()) {
    std::size_t seq_len = 0;
    bool incomplete = false;
    const std::string_view rem(utf8_pending_.data() + pos,
                               utf8_pending_.size() - pos);
    if (is_valid_utf8_sequence(rem, seq_len, incomplete)) {
      out.append(rem.substr(0, seq_len));
      pos += seq_len;
      continue;
    }
    if (incomplete && !flushing) {
      break;
    }
    out.append(kReplacementChar);
    pos += 1;
  }
  utf8_pending_.erase(0, pos);
  return out;
}

std::string OutputParser::apply_stop_strings(std::string_view valid_utf8,
                                             bool flushing) {
  if (options_.stop_strings.empty()) {
    return std::string(valid_utf8);
  }
  stop_buffer_.append(valid_utf8);

  std::size_t earliest_pos = std::string::npos;
  for (const auto &stop : options_.stop_strings) {
    const std::size_t pos = stop_buffer_.find(stop);
    if (pos != std::string::npos && pos < earliest_pos) {
      earliest_pos = pos;
    }
  }
  if (earliest_pos != std::string::npos) {
    std::string released = stop_buffer_.substr(0, earliest_pos);
    stop_buffer_.clear();
    stopped_by_string_ = true;
    return released;
  }
  if (flushing) {
    std::string released = std::move(stop_buffer_);
    stop_buffer_.clear();
    return released;
  }

  std::size_t holdback = 0;
  for (const auto &stop : options_.stop_strings) {
    holdback =
        std::max(holdback, longest_suffix_matching_prefix(stop_buffer_, stop));
  }
  const std::size_t release_len = stop_buffer_.size() - holdback;
  std::string released = stop_buffer_.substr(0, release_len);
  stop_buffer_.erase(0, release_len);
  return released;
}

void OutputParser::emit_content(std::string_view text, OutputDelta &out) {
  while (!text.empty() && skip_post_think_newlines_ > 0) {
    if (text.starts_with("\r\n")) {
      text.remove_prefix(2);
      --skip_post_think_newlines_;
    } else if (text.front() == '\n') {
      text.remove_prefix(1);
      --skip_post_think_newlines_;
    } else {
      skip_post_think_newlines_ = 0;
      break;
    }
  }
  if (!text.empty()) {
    skip_post_think_newlines_ = 0;
    out.content.append(text);
  }
}

void OutputParser::emit_reasoning(std::string_view text, OutputDelta &out) {
  if (skip_leading_think_newline_ && !text.empty()) {
    if (text.starts_with("\r\n")) {
      text.remove_prefix(2);
    } else if (text.front() == '\n') {
      text.remove_prefix(1);
    }
    skip_leading_think_newline_ = false;
  }
  if (!text.empty()) {
    out.reasoning.append(text);
  }
}

bool OutputParser::try_parse_json_tool_call(std::string_view body,
                                            ParsedToolCall &parsed) const {
  const std::string_view trimmed = trim_ascii(body);
  if (trimmed.empty() || trimmed.front() != '{') {
    return false;
  }
  try {
    const auto obj = nlohmann::ordered_json::parse(trimmed);
    if (!obj.is_object() || !obj.contains("name") ||
        !obj.at("name").is_string()) {
      return false;
    }
    parsed.name = obj.at("name").get<std::string>();
    if (parsed.name.empty()) {
      return false;
    }
    if (!obj.contains("arguments") || obj.at("arguments").is_null()) {
      parsed.arguments = "{}";
    } else if (obj.at("arguments").is_string()) {
      parsed.arguments = obj.at("arguments").get<std::string>();
    } else {
      parsed.arguments = obj.at("arguments").dump();
    }
    return true;
  } catch (...) {
    return false;
  }
}

bool OutputParser::try_parse_xml_tool_call(std::string_view body,
                                           ParsedToolCall &parsed) const {
  const std::size_t fn_pos = body.find(kFunctionOpen);
  if (fn_pos == std::string_view::npos) {
    return false;
  }
  const std::size_t name_begin = fn_pos + kFunctionOpen.size();
  const std::size_t name_end = body.find('>', name_begin);
  if (name_end == std::string_view::npos) {
    return false;
  }
  const std::string_view fn_name =
      trim_ascii(body.substr(name_begin, name_end - name_begin));
  if (fn_name.empty()) {
    return false;
  }

  std::size_t cursor = name_end + 1;
  const std::size_t fn_close = body.find(kFunctionClose, cursor);
  const std::size_t fn_limit =
      fn_close == std::string_view::npos ? body.size() : fn_close;
  const std::string_view fn_body = body.substr(cursor, fn_limit - cursor);

  const auto schema_it = options_.tool_param_types.find(std::string(fn_name));
  nlohmann::ordered_json args = nlohmann::ordered_json::object();

  std::size_t p_cursor = 0;
  while (p_cursor < fn_body.size()) {
    const std::size_t p_open = fn_body.find(kParamOpen, p_cursor);
    if (p_open == std::string_view::npos) {
      break;
    }
    const std::size_t pname_begin = p_open + kParamOpen.size();
    const std::size_t pname_end = fn_body.find('>', pname_begin);
    if (pname_end == std::string_view::npos) {
      return false;
    }
    const std::string_view pname =
        trim_ascii(fn_body.substr(pname_begin, pname_end - pname_begin));
    if (pname.empty()) {
      return false;
    }
    const std::size_t val_begin = pname_end + 1;
    const std::size_t val_end = fn_body.find(kParamClose, val_begin);
    if (val_end == std::string_view::npos) {
      return false;
    }
    std::string_view schema_type;
    if (schema_it != options_.tool_param_types.end()) {
      if (const auto pit = schema_it->second.find(std::string(pname));
          pit != schema_it->second.end()) {
        schema_type = pit->second;
      }
    }
    args[std::string(pname)] = parse_xml_param_value(
        fn_body.substr(val_begin, val_end - val_begin), schema_type);
    p_cursor = val_end + kParamClose.size();
  }

  parsed.name = std::string(fn_name);
  parsed.arguments = args.dump();
  return true;
}

bool OutputParser::try_parse_tool_call(std::string_view body,
                                       ParsedToolCall &parsed) const {
  if (try_parse_json_tool_call(body, parsed)) {
    return true;
  }
  return try_parse_xml_tool_call(body, parsed);
}

void OutputParser::process_channels(std::string_view text, bool flushing,
                                    OutputDelta &out) {
  if (!options_.enable_thinking_split && !options_.enable_tool_parsing) {
    out.content.append(text);
    return;
  }

  channel_buffer_.append(text);

  while (!channel_buffer_.empty()) {
    if (state_ == State::Content) {
      const std::size_t think_pos = options_.enable_thinking_split
                                        ? channel_buffer_.find(kThinkOpen)
                                        : std::string::npos;
      const std::size_t tool_pos = options_.enable_tool_parsing
                                       ? channel_buffer_.find(kToolOpen)
                                       : std::string::npos;

      if (think_pos != std::string::npos &&
          (tool_pos == std::string::npos || think_pos < tool_pos)) {
        if (think_pos > 0) {
          emit_content(std::string_view(channel_buffer_).substr(0, think_pos),
                       out);
        }
        channel_buffer_.erase(0, think_pos + kThinkOpen.size());
        state_ = State::Thinking;
        skip_leading_think_newline_ = true;
        continue;
      }
      if (tool_pos != std::string::npos) {
        if (tool_pos > 0) {
          emit_content(std::string_view(channel_buffer_).substr(0, tool_pos),
                       out);
        }
        channel_buffer_.erase(0, tool_pos + kToolOpen.size());
        state_ = State::ToolCall;
        tool_buffer_.clear();
        continue;
      }

      if (flushing) {
        emit_content(channel_buffer_, out);
        channel_buffer_.clear();
        break;
      }

      std::size_t holdback = 0;
      if (options_.enable_thinking_split) {
        holdback = std::max(
            holdback,
            longest_suffix_matching_prefix(channel_buffer_, kThinkOpen));
      }
      if (options_.enable_tool_parsing) {
        holdback = std::max(
            holdback,
            longest_suffix_matching_prefix(channel_buffer_, kToolOpen));
      }
      const std::size_t emit_len = channel_buffer_.size() - holdback;
      if (emit_len > 0) {
        emit_content(std::string_view(channel_buffer_).substr(0, emit_len),
                     out);
        channel_buffer_.erase(0, emit_len);
      }
      break;
    }

    if (state_ == State::Thinking) {
      const std::size_t close_pos = channel_buffer_.find(kThinkClose);
      if (close_pos != std::string::npos) {
        if (close_pos > 0) {
          emit_reasoning(std::string_view(channel_buffer_).substr(0, close_pos),
                         out);
        }
        channel_buffer_.erase(0, close_pos + kThinkClose.size());
        state_ = State::Content;
        skip_leading_think_newline_ = false;
        skip_post_think_newlines_ = 2;
        continue;
      }
      if (flushing) {
        emit_reasoning(channel_buffer_, out);
        channel_buffer_.clear();
        break;
      }
      const std::size_t holdback =
          longest_suffix_matching_prefix(channel_buffer_, kThinkClose);
      const std::size_t emit_len = channel_buffer_.size() - holdback;
      if (emit_len > 0) {
        emit_reasoning(std::string_view(channel_buffer_).substr(0, emit_len),
                       out);
        channel_buffer_.erase(0, emit_len);
      }
      break;
    }

    // State::ToolCall
    tool_buffer_.append(channel_buffer_);
    channel_buffer_.clear();
    const std::size_t close_pos = tool_buffer_.find(kToolClose);
    if (close_pos != std::string::npos) {
      const std::string_view body(tool_buffer_.data(), close_pos);
      ParsedToolCall parsed;
      if (try_parse_tool_call(body, parsed)) {
        parsed.index = next_tool_index_++;
        parsed.id = options_.tool_call_id_prefix + std::to_string(parsed.index);
        out.tool_calls.push_back(std::move(parsed));
      } else {
        // Malformed tool call block -> preserve raw text as content.
        std::string fallback(kToolOpen);
        fallback.append(body);
        fallback.append(kToolClose);
        emit_content(fallback, out);
      }
      const std::string remainder =
          tool_buffer_.substr(close_pos + kToolClose.size());
      tool_buffer_.clear();
      state_ = State::Content;
      if (!remainder.empty()) {
        channel_buffer_ = remainder;
        continue;
      }
      break;
    }
    break;
  }

  if (flushing && state_ == State::ToolCall) {
    std::string fallback(kToolOpen);
    fallback.append(tool_buffer_);
    tool_buffer_.clear();
    state_ = State::Content;
    emit_content(fallback, out);
  }
}

OutputDelta OutputParser::feed_bytes(std::string_view raw_bytes) {
  OutputDelta out;
  if (finished_ || stopped_by_string_) {
    out.stopped_by_string = stopped_by_string_;
    return out;
  }
  const std::string valid = decode_utf8(raw_bytes, false);
  if (valid.empty()) {
    return out;
  }
  const std::string released = apply_stop_strings(valid, false);
  if (!released.empty()) {
    process_channels(released, false, out);
  }
  if (stopped_by_string_) {
    process_channels({}, true, out);
    out.stopped_by_string = true;
  }
  return out;
}

OutputDelta OutputParser::finish() {
  OutputDelta out;
  if (finished_) {
    out.stopped_by_string = stopped_by_string_;
    return out;
  }
  finished_ = true;
  if (!stopped_by_string_) {
    const std::string tail_utf8 = decode_utf8({}, true);
    const std::string released = apply_stop_strings(tail_utf8, true);
    process_channels(released, true, out);
  } else {
    process_channels({}, true, out);
  }
  out.stopped_by_string = stopped_by_string_;
  return out;
}

} // namespace omlx::m5::serve
