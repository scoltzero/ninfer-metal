#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/program.h"

namespace {

std::size_t parse_size(std::string_view value, const char *name) {
  std::size_t parsed = 0;
  const auto result = std::stoull(std::string(value), &parsed);
  if (parsed != value.size()) {
    throw std::invalid_argument(std::string(name) + " must be an integer");
  }
  return static_cast<std::size_t>(result);
}

omlx::m5::ReferenceProgramConfig config() {
  return {
      .vocab_size = 96,
      .hidden_size = 24,
      .intermediate_size = 48,
      .attention_heads = 3,
      .kv_heads = 1,
      .head_dim = 8,
      .linear_key_heads = 1,
      .linear_value_heads = 3,
      .linear_key_dim = 4,
      .linear_value_dim = 4,
      .linear_conv_kernel = 4,
      .max_context = 4096,
      .rotary_dim = 4,
      .rms_norm_eps = 1.0e-6F,
      .rope_theta = 10000.0,
      .eos_token_id = -1,
      .layer_types = {omlx::m5::LayerType::linear_attention,
                      omlx::m5::LayerType::linear_attention,
                      omlx::m5::LayerType::linear_attention,
                      omlx::m5::LayerType::full_attention},
  };
}

} // namespace

int main(int argc, const char *const *argv) {
  try {
    std::size_t decode_tokens = 256;
    std::size_t chunk_size = 16;
    for (int index = 1; index < argc; ++index) {
      const std::string_view argument(argv[index]);
      if (argument == "--decode-tokens" && index + 1 < argc) {
        decode_tokens = parse_size(argv[++index], "--decode-tokens");
      } else if (argument == "--chunk-size" && index + 1 < argc) {
        chunk_size = parse_size(argv[++index], "--chunk-size");
      } else {
        throw std::invalid_argument("unknown or incomplete argument: " +
                                    std::string(argument));
      }
    }
    auto program =
        omlx::m5::ReferenceProgram::create_deterministic(config(), 20260919);
    const std::vector<std::int32_t> prompt = {4, 8, 15, 16, 23, 42};
    const auto result =
        program.generate(prompt, chunk_size, decode_tokens, {}, 7);
    const nlohmann::json output = {
        {"schema_version", 1},
        {"scope", "M4 correctness synthetic Qwen-like reference"},
        {"prefill_tokens", prompt.size()},
        {"prefill_chunk", chunk_size},
        {"requested_decode_tokens", decode_tokens},
        {"generated_tokens", result.tokens.size()},
        {"cancelled", result.cancelled},
        {"reached_eos", result.reached_eos},
        {"final_frontier", program.position()},
        {"parameter_count", program.parameter_count()},
        {"tokens", result.tokens},
    };
    std::cout << output.dump(2) << '\n';
    return result.tokens.size() == decode_tokens ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "omlx_m5_reference_runner: " << error.what() << '\n';
    return 1;
  }
}
