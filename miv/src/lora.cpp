#include "omlx_m5/lora.h"

#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/safetensors.h"

namespace omlx::m5 {

namespace {

using Json = nlohmann::json;

std::filesystem::path expand_user_dir(const std::filesystem::path &raw) {
  const std::string s = raw.string();
  if (s == "~" || s.starts_with("~/")) {
    if (const char *home = std::getenv("HOME");
        home != nullptr && home[0] != '\0') {
      if (s == "~") {
        return std::filesystem::path(home);
      }
      return std::filesystem::path(home) / s.substr(2);
    }
  }
  return raw;
}

std::string to_lower_ascii(std::string_view input) {
  std::string out(input);
  for (char &ch : out) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return out;
}

float fp16_to_float(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
  const std::uint32_t exp = (h >> 10U) & 0x1FU;
  const std::uint32_t mant = h & 0x03FFU;
  if (exp == 0U) {
    if (mant == 0U) {
      return std::bit_cast<float>(sign);
    }
    const float mag = std::ldexp(static_cast<float>(mant), -24);
    return (sign != 0U) ? -mag : mag;
  }
  if (exp == 31U) {
    return std::bit_cast<float>(sign | 0x7F800000U | (mant << 13U));
  }
  const std::uint32_t f_bits =
      sign | ((exp + 112U) << 23U) | (mant << 13U);
  return std::bit_cast<float>(f_bits);
}

bool json_truthy(const Json &obj, std::string_view key) {
  const std::string k(key);
  if (!obj.contains(k) || obj.at(k).is_null()) {
    return false;
  }
  const auto &v = obj.at(k);
  if (v.is_boolean()) {
    return v.get<bool>();
  }
  if (v.is_number_integer()) {
    return v.get<std::int64_t>() != 0;
  }
  if (v.is_number_float()) {
    return v.get<double>() != 0.0;
  }
  if (v.is_string()) {
    return !v.get<std::string>().empty();
  }
  if (v.is_array() || v.is_object()) {
    return !v.empty();
  }
  return true;
}

bool module_matches_pattern(std::string_view module_path,
                            std::string_view normalized_path,
                            std::string_view pattern) {
  if (pattern.empty()) {
    return false;
  }
  auto suffix_match = [](std::string_view candidate, std::string_view pat) {
    if (candidate == pat) {
      return true;
    }
    if (candidate.size() > pat.size() && candidate.ends_with(pat) &&
        candidate[candidate.size() - pat.size() - 1] == '.') {
      return true;
    }
    return false;
  };
  return suffix_match(module_path, pattern) ||
         suffix_match(normalized_path, pattern);
}

template <typename T>
std::optional<T> match_pattern_map(const std::map<std::string, T> &patterns,
                                   std::string_view raw_module,
                                   std::string_view normalized_module) {
  std::optional<T> best;
  std::size_t best_len = 0;
  for (const auto &[pat, val] : patterns) {
    if (module_matches_pattern(raw_module, normalized_module, pat)) {
      if (!best.has_value() || pat.size() > best_len) {
        best = val;
        best_len = pat.size();
      }
    }
  }
  return best;
}

struct ParsedAdapterSpec {
  LoraFormat format = LoraFormat::peft;
  std::filesystem::path weights_path;
  bool sharded_index = false;
  std::optional<std::uint32_t> default_rank;
  std::optional<double> default_alpha;
  std::optional<double> explicit_scale;
  bool mlx_has_explicit_scale = false;
  bool use_rslora = false;
  bool global_fan_in_fan_out = false;
  std::vector<std::string> fan_in_fan_out_modules;
  std::map<std::string, std::uint32_t> rank_pattern;
  std::map<std::string, double> alpha_pattern;
  std::string config_canonical;
};

void parse_fan_in_fan_out(const Json &val, ParsedAdapterSpec &spec) {
  if (val.is_boolean()) {
    spec.global_fan_in_fan_out = val.get<bool>();
  } else if (val.is_array()) {
    for (const auto &item : val) {
      if (!item.is_string()) {
        throw std::invalid_argument(
            "fan_in_fan_out array entries must be module strings");
      }
      spec.fan_in_fan_out_modules.push_back(item.get<std::string>());
    }
  } else if (!val.is_null()) {
    throw std::invalid_argument(
        "fan_in_fan_out must be a boolean or list of module names");
  }
}

void parse_pattern_tables(const Json &src, ParsedAdapterSpec &spec) {
  if (src.contains("rank_pattern") && !src.at("rank_pattern").is_null()) {
    const auto &rp = src.at("rank_pattern");
    if (!rp.is_object()) {
      throw std::invalid_argument("rank_pattern must be a JSON object");
    }
    for (const auto &[k, v] : rp.items()) {
      if (!v.is_number_integer() || v.get<std::int64_t>() <= 0) {
        throw std::invalid_argument("rank_pattern values must be positive integers");
      }
      spec.rank_pattern[k] = static_cast<std::uint32_t>(v.get<std::int64_t>());
    }
  }
  if (src.contains("alpha_pattern") && !src.at("alpha_pattern").is_null()) {
    const auto &ap = src.at("alpha_pattern");
    if (!ap.is_object()) {
      throw std::invalid_argument("alpha_pattern must be a JSON object");
    }
    for (const auto &[k, v] : ap.items()) {
      if (!v.is_number()) {
        throw std::invalid_argument("alpha_pattern values must be numeric");
      }
      const double d = v.get<double>();
      if (!std::isfinite(d)) {
        throw std::invalid_argument("alpha_pattern contains non-finite value");
      }
      spec.alpha_pattern[k] = d;
    }
  }
}

ParsedAdapterSpec read_adapter_spec(const std::filesystem::path &dir) {
  const auto config_path = dir / "adapter_config.json";
  if (!std::filesystem::is_regular_file(config_path)) {
    throw std::invalid_argument("LoRA adapter_config.json is missing in " +
                                dir.string());
  }
  Json config;
  {
    std::ifstream in(config_path);
    if (!in) {
      throw std::runtime_error("unable to open " + config_path.string());
    }
    try {
      config = Json::parse(in);
    } catch (const Json::exception &ex) {
      throw std::invalid_argument("invalid JSON in " + config_path.string() +
                                  ": " + ex.what());
    }
  }
  if (!config.is_object()) {
    throw std::invalid_argument("adapter_config.json must be a JSON object");
  }

  ParsedAdapterSpec spec;
  spec.config_canonical = config.dump();

  if (config.contains("peft_type")) {
    const std::string peft_type =
        config.at("peft_type").is_string()
            ? to_lower_ascii(config.at("peft_type").get<std::string>())
            : std::string{};
    if (peft_type != "lora") {
      throw std::invalid_argument("PEFT type '" + peft_type +
                                  "' is not supported; only LORA is supported");
    }
    constexpr std::array<std::pair<std::string_view, std::string_view>, 10>
        kUnsupportedPeft{{
            {"use_dora", "DoRA adapters"},
            {"modules_to_save",
             "adapters that replace whole modules (modules_to_save)"},
            {"trainable_token_indices",
             "adapters that train token embeddings"},
            {"target_parameters",
             "adapters that target raw parameters such as MoE experts"},
            {"layer_replication", "adapters with layer replication"},
            {"use_qalora", "QA-LoRA adapters"},
            {"alora_invocation_tokens", "activated LoRA (aLoRA) adapters"},
            {"arrow_config", "Arrow multi-adapter routing"},
            {"lora_bias", "adapters with a LoRA bias"},
            {"loftq_config",
             "LoftQ adapters requiring their own quantized base"},
        }};
    for (const auto &[key, label] : kUnsupportedPeft) {
      if (json_truthy(config, key)) {
        throw std::invalid_argument(std::string(label) + " are not supported");
      }
    }
    if (config.contains("init_lora_weights") &&
        config.at("init_lora_weights").is_string()) {
      const std::string init =
          to_lower_ascii(config.at("init_lora_weights").get<std::string>());
      if (init.starts_with("pissa") || init.starts_with("olora") ||
          init.starts_with("corda")) {
        throw std::invalid_argument(
            init +
            " adapters need the residual base they were trained on; convert "
            "to plain LoRA first");
      }
    }
    if (config.contains("bias") && !config.at("bias").is_null()) {
      if (!config.at("bias").is_string() ||
          config.at("bias").get<std::string>() != "none") {
        throw std::invalid_argument(
            "adapters that train biases are not supported");
      }
    }
    if (config.contains("r") && !config.at("r").is_null()) {
      if (!config.at("r").is_number_integer() ||
          config.at("r").get<std::int64_t>() <= 0) {
        throw std::invalid_argument("PEFT 'r' must be a positive integer");
      }
      spec.default_rank =
          static_cast<std::uint32_t>(config.at("r").get<std::int64_t>());
    } else if (config.contains("rank") && !config.at("rank").is_null()) {
      if (!config.at("rank").is_number_integer() ||
          config.at("rank").get<std::int64_t>() <= 0) {
        throw std::invalid_argument("PEFT 'rank' must be a positive integer");
      }
      spec.default_rank =
          static_cast<std::uint32_t>(config.at("rank").get<std::int64_t>());
    }
    if (config.contains("lora_alpha") && config.at("lora_alpha").is_number()) {
      spec.default_alpha = config.at("lora_alpha").get<double>();
    } else if (config.contains("alpha") && config.at("alpha").is_number()) {
      spec.default_alpha = config.at("alpha").get<double>();
    }
    if (config.contains("scale") && config.at("scale").is_number()) {
      spec.explicit_scale = config.at("scale").get<double>();
    }
    if (!spec.default_alpha.has_value() && !spec.explicit_scale.has_value()) {
      throw std::invalid_argument(
          "lora_alpha is missing from the PEFT adapter config");
    }
    if (config.contains("use_rslora") && config.at("use_rslora").is_boolean()) {
      spec.use_rslora = config.at("use_rslora").get<bool>();
    }
    if (config.contains("fan_in_fan_out")) {
      parse_fan_in_fan_out(config.at("fan_in_fan_out"), spec);
    }
    parse_pattern_tables(config, spec);
    spec.format = LoraFormat::peft;
    const auto peft_single = dir / "adapter_model.safetensors";
    const auto peft_index = dir / "adapter_model.safetensors.index.json";
    if (std::filesystem::is_regular_file(peft_single)) {
      spec.weights_path = peft_single;
    } else if (std::filesystem::is_regular_file(peft_index)) {
      spec.weights_path = peft_index;
      spec.sharded_index = true;
    } else {
      throw std::invalid_argument(
          "adapter_model.safetensors is missing from " + dir.string());
    }
    return spec;
  }

  // mlx-lm / mlx-vlm format
  const std::string ft_type =
      config.contains("fine_tune_type") &&
              config.at("fine_tune_type").is_string()
          ? to_lower_ascii(config.at("fine_tune_type").get<std::string>())
          : std::string("lora");
  if (ft_type == "dora") {
    throw std::invalid_argument("DoRA adapters are not supported");
  }
  if (ft_type != "lora") {
    throw std::invalid_argument("fine_tune_type '" + ft_type +
                                "' is not a LoRA adapter");
  }
  spec.format = LoraFormat::mlx;
  if (config.contains("use_rslora") && config.at("use_rslora").is_boolean()) {
    spec.use_rslora = config.at("use_rslora").get<bool>();
  }
  if (config.contains("fan_in_fan_out")) {
    parse_fan_in_fan_out(config.at("fan_in_fan_out"), spec);
  }
  parse_pattern_tables(config, spec);

  if (config.contains("lora_parameters")) {
    const auto &lp = config.at("lora_parameters");
    if (!lp.is_object()) {
      throw std::invalid_argument("lora_parameters must be a JSON object");
    }
    if (lp.contains("rank") && lp.at("rank").is_number_integer() &&
        lp.at("rank").get<std::int64_t>() > 0) {
      spec.default_rank =
          static_cast<std::uint32_t>(lp.at("rank").get<std::int64_t>());
    } else if (lp.contains("r") && lp.at("r").is_number_integer() &&
               lp.at("r").get<std::int64_t>() > 0) {
      spec.default_rank =
          static_cast<std::uint32_t>(lp.at("r").get<std::int64_t>());
    }
    if (lp.contains("alpha") && lp.at("alpha").is_number()) {
      spec.default_alpha = lp.at("alpha").get<double>();
    } else if (lp.contains("lora_alpha") && lp.at("lora_alpha").is_number()) {
      spec.default_alpha = lp.at("lora_alpha").get<double>();
    }
    if (lp.contains("scale") && lp.at("scale").is_number()) {
      spec.explicit_scale = lp.at("scale").get<double>();
      spec.mlx_has_explicit_scale = true;
    }
    if (lp.contains("use_rslora") && lp.at("use_rslora").is_boolean()) {
      spec.use_rslora = lp.at("use_rslora").get<bool>();
    }
    if (lp.contains("fan_in_fan_out")) {
      parse_fan_in_fan_out(lp.at("fan_in_fan_out"), spec);
    }
    parse_pattern_tables(lp, spec);
  }
  if (!spec.default_rank.has_value() && config.contains("rank") &&
      config.at("rank").is_number_integer() &&
      config.at("rank").get<std::int64_t>() > 0) {
    spec.default_rank =
        static_cast<std::uint32_t>(config.at("rank").get<std::int64_t>());
  }
  if (!spec.default_alpha.has_value() && config.contains("lora_alpha") &&
      config.at("lora_alpha").is_number()) {
    spec.default_alpha = config.at("lora_alpha").get<double>();
  } else if (!spec.default_alpha.has_value() && config.contains("alpha") &&
             config.at("alpha").is_number()) {
    spec.default_alpha = config.at("alpha").get<double>();
  }
  if (!spec.explicit_scale.has_value() && config.contains("scale") &&
      config.at("scale").is_number()) {
    spec.explicit_scale = config.at("scale").get<double>();
  }
  if (!spec.explicit_scale.has_value() && !spec.default_alpha.has_value()) {
    throw std::invalid_argument(
        "lora_parameters.scale (or alpha) is missing from the mlx-lm adapter "
        "config");
  }
  const auto mlx_weights = dir / "adapters.safetensors";
  if (!std::filesystem::is_regular_file(mlx_weights)) {
    throw std::invalid_argument("adapters.safetensors is missing from " +
                                dir.string());
  }
  spec.weights_path = mlx_weights;
  return spec;
}

struct SplitKeyResult {
  std::string module_path;
  char ab = 0; // 'a' or 'b'
};

std::optional<SplitKeyResult> split_lora_tensor_key(LoraFormat fmt,
                                                    std::string_view key) {
  if (fmt == LoraFormat::peft) {
    std::string_view rest = key;
    while (rest.starts_with("base_model.model.")) {
      rest.remove_prefix(std::string_view("base_model.model.").size());
    }
    if (!rest.ends_with(".weight")) {
      return std::nullopt;
    }
    rest.remove_suffix(std::string_view(".weight").size());
    // Match <module>.lora_A or <module>.lora_A.<adapter_name>
    for (const auto &[marker, ch] :
         std::array<std::pair<std::string_view, char>, 2>{
             {{".lora_A", 'a'}, {".lora_B", 'b'}}}) {
      const auto pos = rest.rfind(marker);
      if (pos == std::string_view::npos || pos == 0) {
        continue;
      }
      const auto tail = rest.substr(pos + marker.size());
      if (tail.empty() ||
          (tail.front() == '.' && tail.substr(1).find('.') == std::string_view::npos &&
           tail.size() > 1)) {
        return SplitKeyResult{std::string(rest.substr(0, pos)), ch};
      }
    }
    return std::nullopt;
  }

  // mlx format: <module>.lora_a or <module>.lora_b
  if (key.ends_with(".lora_a") && key.size() > 7) {
    return SplitKeyResult{std::string(key.substr(0, key.size() - 7)), 'a'};
  }
  if (key.ends_with(".lora_b") && key.size() > 7) {
    return SplitKeyResult{std::string(key.substr(0, key.size() - 7)), 'b'};
  }
  return std::nullopt;
}

bool contains_non_decoder_component(std::string_view prefix) {
  const std::string lower = to_lower_ascii(prefix);
  for (std::string_view bad :
       {"vision", "visual", "audio", "image", "encoder", "projector", "mtp"}) {
    if (lower.find(bad) != std::string::npos) {
      return true;
    }
  }
  return false;
}

std::string normalize_module_prefix(std::string_view raw) {
  std::string_view cur = raw;
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::string_view pfx :
         {"base_model.model.", "language_model.model.",
          "model.language_model.", "language_model.", "model."}) {
      if (cur.starts_with(pfx)) {
        cur.remove_prefix(pfx.size());
        changed = true;
      }
    }
  }
  return std::string(cur);
}

std::vector<float> decode_tensor_f32(const TensorView &tv,
                                     std::string_view tensor_name) {
  if (tv.info == nullptr || tv.data == nullptr) {
    throw std::runtime_error("null safetensor view for " +
                             std::string(tensor_name));
  }
  if (tv.info->shape.size() != 2 || tv.info->shape[0] == 0 ||
      tv.info->shape[1] == 0) {
    throw std::invalid_argument("LoRA tensor '" + std::string(tensor_name) +
                                "' must be a non-empty 2D matrix");
  }
  const std::uint64_t count = tv.info->shape[0] * tv.info->shape[1];
  std::vector<float> out(static_cast<std::size_t>(count));
  switch (tv.info->dtype) {
  case TensorDType::f32: {
    std::memcpy(out.data(), tv.data, static_cast<std::size_t>(count) * 4U);
    break;
  }
  case TensorDType::bf16: {
    for (std::uint64_t i = 0; i < count; ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, tv.data + i * 2U, 2U);
      out[static_cast<std::size_t>(i)] = lora_bf16_to_float(bits);
    }
    break;
  }
  case TensorDType::f16: {
    for (std::uint64_t i = 0; i < count; ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, tv.data + i * 2U, 2U);
      out[static_cast<std::size_t>(i)] = fp16_to_float(bits);
    }
    break;
  }
  default:
    throw std::invalid_argument(
        "unsupported LoRA safetensor dtype " +
        std::string(dtype_name(tv.info->dtype)) + " on '" +
        std::string(tensor_name) + "' (expected BF16, F16, or F32)");
  }
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (!std::isfinite(out[i])) {
      throw std::invalid_argument("LoRA tensor '" + std::string(tensor_name) +
                                  "' contains non-finite values");
    }
  }
  return out;
}

std::vector<float> transpose_2d(std::span<const float> src, std::uint32_t rows,
                                std::uint32_t cols) {
  std::vector<float> dst(static_cast<std::size_t>(rows) * cols);
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::uint32_t c = 0; c < cols; ++c) {
      dst[static_cast<std::size_t>(c) * rows + r] =
          src[static_cast<std::size_t>(r) * cols + c];
    }
  }
  return dst;
}

struct CanonicalFactors {
  std::uint32_t rank = 0;
  std::vector<float> a_rk; // [r, K]
  std::vector<float> b_nr; // [N, r]
};

CanonicalFactors canonicalize_factors(
    const ParsedAdapterSpec &spec, std::string_view raw_module,
    std::string_view norm_module, const TensorView &ta, const TensorView &tb,
    std::uint32_t expected_n, std::uint32_t expected_k, bool fan_in_fan_out) {
  const auto a_s0 = static_cast<std::uint32_t>(ta.info->shape[0]);
  const auto a_s1 = static_cast<std::uint32_t>(ta.info->shape[1]);
  const auto b_s0 = static_cast<std::uint32_t>(tb.info->shape[0]);
  const auto b_s1 = static_cast<std::uint32_t>(tb.info->shape[1]);

  auto raw_a = decode_tensor_f32(ta, ta.info->name);
  auto raw_b = decode_tensor_f32(tb, tb.info->name);

  CanonicalFactors res;
  if (spec.format == LoraFormat::mlx) {
    // mlx-lm stores lora_a as [K, r] and lora_b as [r, N] (unless fan_in_fan_out flips)
    if (!fan_in_fan_out) {
      if (a_s1 != b_s0) {
        throw std::invalid_argument(std::string(norm_module) +
                                    " has LoRA a and b of different ranks (" +
                                    std::to_string(a_s1) + " vs " +
                                    std::to_string(b_s0) + ")");
      }
      if (a_s0 != expected_k || b_s1 != expected_n) {
        throw std::invalid_argument(
            std::string(norm_module) + " is " + std::to_string(expected_k) +
            " -> " + std::to_string(expected_n) + "; the adapter expects " +
            std::to_string(a_s0) + " -> " + std::to_string(b_s1));
      }
      res.rank = a_s1;
      res.a_rk = transpose_2d(raw_a, a_s0, a_s1);
      res.b_nr = transpose_2d(raw_b, b_s0, b_s1);
    } else {
      if (a_s0 == b_s1 && a_s1 == expected_k && b_s0 == expected_n) {
        res.rank = a_s0;
        res.a_rk = std::move(raw_a);
        res.b_nr = std::move(raw_b);
      } else if (a_s1 == b_s0 && a_s0 == expected_n && b_s1 == expected_k) {
        res.rank = a_s1;
        res.a_rk = std::move(raw_b);
        res.b_nr = std::move(raw_a);
      } else {
        throw std::invalid_argument(
            std::string(norm_module) + " fan_in_fan_out shape mismatch for " +
            std::to_string(expected_k) + " -> " + std::to_string(expected_n));
      }
    }
  } else {
    // PEFT: standard lora_A is [r, K], lora_B is [N, r]
    if (!fan_in_fan_out) {
      if (a_s0 != b_s1) {
        throw std::invalid_argument(std::string(norm_module) +
                                    " has LoRA A and B of different ranks (" +
                                    std::to_string(a_s0) + " vs " +
                                    std::to_string(b_s1) + ")");
      }
      if (a_s1 != expected_k || b_s0 != expected_n) {
        throw std::invalid_argument(
            std::string(norm_module) + " is " + std::to_string(expected_k) +
            " -> " + std::to_string(expected_n) + "; the adapter expects " +
            std::to_string(a_s1) + " -> " + std::to_string(b_s0));
      }
      res.rank = a_s0;
      res.a_rk = std::move(raw_a);
      res.b_nr = std::move(raw_b);
    } else {
      // fan_in_fan_out=true:
      // Case 1: stored transposed as A=[K, r], B=[r, N] -> transpose each
      // Case 2: PEFT Conv1D delta=(B @ A)^T where B=[K, r], A=[r, N] -> A_canon=B^T [r, K], B_canon=A^T [N, r]
      // Case 3: PEFT Conv1D where miv already transposed base W to [N, K] and A=[r, K], B=[N, r]
      const auto pat_r =
          match_pattern_map(spec.rank_pattern, raw_module, norm_module);
      const std::optional<std::uint32_t> cfg_r =
          pat_r.has_value() ? pat_r : spec.default_rank;
      const auto rank_ok = [&](std::uint32_t r_cand) {
        return !cfg_r.has_value() || r_cand == *cfg_r ||
               (r_cand != expected_k && r_cand != expected_n);
      };
      if (a_s1 == b_s0 && a_s0 == expected_k && b_s1 == expected_n &&
          rank_ok(a_s1)) {
        res.rank = a_s1;
        res.a_rk = transpose_2d(raw_a, a_s0, a_s1);
        res.b_nr = transpose_2d(raw_b, b_s0, b_s1);
      } else if (a_s0 == b_s1 && a_s1 == expected_n && b_s0 == expected_k &&
                 rank_ok(a_s0)) {
        res.rank = a_s0;
        res.a_rk = transpose_2d(raw_b, b_s0, b_s1);
        res.b_nr = transpose_2d(raw_a, a_s0, a_s1);
      } else if (a_s0 == b_s1 && a_s1 == expected_k && b_s0 == expected_n) {
        res.rank = a_s0;
        res.a_rk = std::move(raw_a);
        res.b_nr = std::move(raw_b);
      } else {
        throw std::invalid_argument(
            std::string(norm_module) +
            " fan_in_fan_out LoRA A/B shape mismatch for " +
            std::to_string(expected_k) + " -> " + std::to_string(expected_n));
      }
    }
  }

  if (const auto pat_r =
          match_pattern_map(spec.rank_pattern, raw_module, norm_module);
      pat_r.has_value() && res.rank != *pat_r) {
    throw std::invalid_argument(std::string(norm_module) + " has rank " +
                                std::to_string(res.rank) +
                                " but rank_pattern specifies " +
                                std::to_string(*pat_r));
  }
  return res;
}

float compute_lora_scale(const ParsedAdapterSpec &spec,
                         std::string_view raw_module,
                         std::string_view norm_module, std::uint32_t rank) {
  if (rank == 0) {
    throw std::invalid_argument("LoRA rank must be positive");
  }
  const double denom =
      spec.use_rslora ? std::sqrt(static_cast<double>(rank))
                      : static_cast<double>(rank);
  if (spec.format == LoraFormat::mlx && spec.mlx_has_explicit_scale) {
    const double sc = *spec.explicit_scale;
    if (!std::isfinite(sc)) {
      throw std::invalid_argument("non-finite mlx LoRA scale");
    }
    return static_cast<float>(sc);
  }
  if (const auto alpha_override =
          match_pattern_map(spec.alpha_pattern, raw_module, norm_module);
      alpha_override.has_value()) {
    const double sc = *alpha_override / denom;
    if (!std::isfinite(sc)) {
      throw std::invalid_argument("non-finite LoRA alpha_pattern scale");
    }
    return static_cast<float>(sc);
  }
  if (spec.explicit_scale.has_value() && !spec.default_alpha.has_value()) {
    const double sc = *spec.explicit_scale;
    if (!std::isfinite(sc)) {
      throw std::invalid_argument("non-finite LoRA explicit scale");
    }
    return static_cast<float>(sc);
  }
  const double alpha = spec.default_alpha.value_or(static_cast<double>(rank));
  const double sc = alpha / denom;
  if (!std::isfinite(sc)) {
    throw std::invalid_argument("non-finite LoRA scale");
  }
  return static_cast<float>(sc);
}

bool is_module_fan_in_fan_out(const ParsedAdapterSpec &spec,
                              std::string_view raw_module,
                              std::string_view norm_module) {
  if (spec.global_fan_in_fan_out) {
    return true;
  }
  for (const auto &pat : spec.fan_in_fan_out_modules) {
    if (module_matches_pattern(raw_module, norm_module, pat)) {
      return true;
    }
  }
  return false;
}

} // namespace

std::uint16_t lora_float_to_bf16_rne(float value) noexcept {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t lsb = (bits >> 16U) & 1U;
  bits += 0x7FFFU + lsb;
  return static_cast<std::uint16_t>(bits >> 16U);
}

float lora_bf16_to_float(std::uint16_t bits) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

LoadedLoraAdapter
load_lora_adapter_for_qwen(const std::filesystem::path &adapter_dir,
                           const QwenModelConfig &config,
                           bool lm_head_tied) {
  const auto expanded_dir = expand_user_dir(adapter_dir);
  if (!std::filesystem::is_directory(expanded_dir)) {
    throw std::invalid_argument("LoRA adapter directory not found: " +
                                adapter_dir.string());
  }
  const auto canonical_dir = std::filesystem::weakly_canonical(expanded_dir);
  const ParsedAdapterSpec spec = read_adapter_spec(canonical_dir);
  const SafeTensorModel st =
      spec.sharded_index ? SafeTensorModel::open(canonical_dir)
                         : SafeTensorModel::open_file(spec.weights_path);

  struct RawPair {
    std::optional<std::string> a_key;
    std::optional<std::string> b_key;
  };
  std::map<std::string, RawPair> raw_pairs;
  for (const auto &tname : st.tensor_names()) {
    const auto split = split_lora_tensor_key(spec.format, tname);
    if (!split.has_value()) {
      throw std::invalid_argument(
          "the adapter holds tensors ninfer-metal does not apply: " + tname);
    }
    auto &entry = raw_pairs[split->module_path];
    if (split->ab == 'a') {
      if (entry.a_key.has_value()) {
        throw std::invalid_argument("duplicate LoRA A tensor for " +
                                    split->module_path);
      }
      entry.a_key = tname;
    } else {
      if (entry.b_key.has_value()) {
        throw std::invalid_argument("duplicate LoRA B tensor for " +
                                    split->module_path);
      }
      entry.b_key = tname;
    }
  }
  if (raw_pairs.empty()) {
    throw std::invalid_argument("the adapter holds no LoRA tensors");
  }

  const auto h = static_cast<std::uint32_t>(config.hidden_size);
  const auto f = static_cast<std::uint32_t>(config.intermediate_size);
  const auto shared_f =
      static_cast<std::uint32_t>(config.shared_expert_intermediate_size);
  const auto vocab = static_cast<std::uint32_t>(config.vocab_size);
  const auto q_heads = static_cast<std::uint32_t>(config.num_attention_heads);
  const auto kv_heads = static_cast<std::uint32_t>(config.num_key_value_heads);
  const auto head_dim = static_cast<std::uint32_t>(config.head_dim);
  const auto q_width = q_heads * head_dim;
  const auto kv_width = kv_heads * head_dim;
  const auto nk = static_cast<std::uint32_t>(config.linear_num_key_heads);
  const auto nv = static_cast<std::uint32_t>(config.linear_num_value_heads);
  const auto dk = static_cast<std::uint32_t>(config.linear_key_head_dim);
  const auto dv = static_cast<std::uint32_t>(config.linear_value_head_dim);
  const auto gdn_qkv_width = 2U * nk * dk + nv * dv;
  const auto gdn_v_width = nv * dv;
  const auto num_layers = static_cast<std::uint32_t>(config.num_hidden_layers);

  struct SplitQHalf {
    std::uint32_t rank = 0;
    float scale = 1.0F;
    std::vector<float> a_rk;
    std::vector<float> b_qr; // [q_width, r]
    std::string source;
  };
  struct PendingSplitQ {
    std::optional<SplitQHalf> q_only;
    std::optional<SplitQHalf> gate_only;
  };
  std::map<std::uint32_t, PendingSplitQ> split_q_by_layer;

  std::vector<LoraProjectionUpdate> updates;
  std::set<std::pair<bool, std::pair<std::uint32_t, std::string>>> seen_targets;

  auto register_update = [&](LoraProjectionUpdate &&upd) {
    const auto ukey = std::make_pair(
        upd.target.is_global,
        std::make_pair(upd.target.layer_index, upd.target.role));
    if (!seen_targets.insert(ukey).second) {
      throw std::invalid_argument(
          "conflicting/duplicate LoRA target for role '" + upd.target.role +
          "' on layer " + std::to_string(upd.target.layer_index));
    }
    updates.push_back(std::move(upd));
  };

  for (const auto &[raw_mod, pair] : raw_pairs) {
    if (!pair.a_key.has_value() || !pair.b_key.has_value()) {
      throw std::invalid_argument(raw_mod +
                                  " is missing its LoRA A or B tensor");
    }
    if (contains_non_decoder_component(raw_mod)) {
      throw std::invalid_argument(
          "only decoder projections can be adapted; adapter targets '" +
          raw_mod + "'");
    }
    const std::string norm_mod = normalize_module_prefix(raw_mod);
    const bool fifo = is_module_fan_in_fan_out(spec, raw_mod, norm_mod);
    const TensorView ta = st.tensor(*pair.a_key);
    const TensorView tb = st.tensor(*pair.b_key);

    if (norm_mod == "lm_head") {
      if (lm_head_tied) {
        throw std::invalid_argument(
            "cannot apply LoRA to tied lm_head; only untied lm_head or decoder "
            "layers can be adapted");
      }
      auto cf = canonicalize_factors(spec, raw_mod, norm_mod, ta, tb, vocab, h,
                                     fifo);
      const float sc = compute_lora_scale(spec, raw_mod, norm_mod, cf.rank);
      register_update(LoraProjectionUpdate{
          .target = {.is_global = true, .layer_index = 0, .role = "lm_head"},
          .source_module = norm_mod,
          .rank = cf.rank,
          .out_features = vocab,
          .in_features = h,
          .scale = sc,
          .a_f32 = std::move(cf.a_rk),
          .b_f32 = std::move(cf.b_nr),
      });
      continue;
    }

    if (!norm_mod.starts_with("layers.")) {
      throw std::invalid_argument(
          "only projections inside decoder layers (or untied lm_head) can be "
          "adapted; adapter targets '" +
          raw_mod + "'");
    }

    const auto after_layers = std::string_view(norm_mod).substr(7);
    const auto dot_pos = after_layers.find('.');
    if (dot_pos == std::string_view::npos || dot_pos == 0) {
      throw std::invalid_argument("malformed decoder layer path: " + raw_mod);
    }
    const auto layer_str = after_layers.substr(0, dot_pos);
    for (char ch : layer_str) {
      if (!std::isdigit(static_cast<unsigned char>(ch))) {
        throw std::invalid_argument("invalid layer index in " + raw_mod);
      }
    }
    const auto layer_idx =
        static_cast<std::uint32_t>(std::stoul(std::string(layer_str)));
    if (layer_idx >= num_layers) {
      throw std::invalid_argument(
          "the adapter targets layer " + std::to_string(layer_idx) +
          "; the model has " + std::to_string(num_layers));
    }
    const std::string_view sub = after_layers.substr(dot_pos + 1);
    const LayerType ltype = config.layer_types.at(layer_idx);

    auto emit_direct = [&](std::string role, std::uint32_t exp_n,
                           std::uint32_t exp_k) {
      auto cf = canonicalize_factors(spec, raw_mod, norm_mod, ta, tb, exp_n,
                                     exp_k, fifo);
      const float sc = compute_lora_scale(spec, raw_mod, norm_mod, cf.rank);
      register_update(LoraProjectionUpdate{
          .target =
              {.is_global = false, .layer_index = layer_idx, .role = std::move(role)},
          .source_module = norm_mod,
          .rank = cf.rank,
          .out_features = exp_n,
          .in_features = exp_k,
          .scale = sc,
          .a_f32 = std::move(cf.a_rk),
          .b_f32 = std::move(cf.b_nr),
      });
    };

    if (sub.starts_with("self_attn.")) {
      if (ltype != LayerType::full_attention) {
        throw std::invalid_argument("layer " + std::to_string(layer_idx) +
                                    " is GDN linear_attn, not self_attn (" +
                                    raw_mod + ")");
      }
      const auto proj = sub.substr(10);
      if (proj == "q_proj") {
        if (!config.has_attn_output_gate) {
          emit_direct("attention_q", q_width, h);
        } else {
          // Support both fused [2*q_width, h] (Qwen3.5/3.8 standard) and split [q_width, h]
          const std::uint32_t cand_out = [&]() -> std::uint32_t {
            if (spec.format == LoraFormat::mlx && !fifo) {
              return static_cast<std::uint32_t>(tb.info->shape[1]);
            }
            if (spec.format == LoraFormat::peft && !fifo) {
              return static_cast<std::uint32_t>(tb.info->shape[0]);
            }
            const auto b0 = static_cast<std::uint32_t>(tb.info->shape[0]);
            const auto b1 = static_cast<std::uint32_t>(tb.info->shape[1]);
            if (b0 == 2U * q_width || b1 == 2U * q_width) {
              return 2U * q_width;
            }
            if (b0 == q_width || b1 == q_width) {
              return q_width;
            }
            return 2U * q_width;
          }();
          if (cand_out == q_width && q_width > 0) {
            auto cf = canonicalize_factors(spec, raw_mod, norm_mod, ta, tb,
                                           q_width, h, fifo);
            const float sc =
                compute_lora_scale(spec, raw_mod, norm_mod, cf.rank);
            auto &sq = split_q_by_layer[layer_idx];
            if (sq.q_only.has_value()) {
              throw std::invalid_argument("duplicate split q_proj on layer " +
                                          std::to_string(layer_idx));
            }
            sq.q_only = SplitQHalf{cf.rank, sc, std::move(cf.a_rk),
                                   std::move(cf.b_nr), norm_mod};
          } else {
            emit_direct("attention_q", 2U * q_width, h);
          }
        }
      } else if (proj == "q_gate_proj") {
        if (!config.has_attn_output_gate) {
          throw std::invalid_argument(
              "q_gate_proj is not valid for ungated attention models");
        }
        auto cf = canonicalize_factors(spec, raw_mod, norm_mod, ta, tb, q_width,
                                       h, fifo);
        const float sc = compute_lora_scale(spec, raw_mod, norm_mod, cf.rank);
        auto &sq = split_q_by_layer[layer_idx];
        if (sq.gate_only.has_value()) {
          throw std::invalid_argument("duplicate q_gate_proj on layer " +
                                      std::to_string(layer_idx));
        }
        sq.gate_only = SplitQHalf{cf.rank, sc, std::move(cf.a_rk),
                                  std::move(cf.b_nr), norm_mod};
      } else if (proj == "k_proj") {
        emit_direct("attention_k", kv_width, h);
      } else if (proj == "v_proj") {
        emit_direct("attention_v", kv_width, h);
      } else if (proj == "o_proj") {
        emit_direct("attention_o", h, q_width);
      } else {
        throw std::invalid_argument("the model has no supported self_attn "
                                    "projection '" +
                                    std::string(proj) + "'");
      }
      continue;
    }

    if (sub.starts_with("linear_attn.")) {
      if (ltype != LayerType::linear_attention) {
        throw std::invalid_argument("layer " + std::to_string(layer_idx) +
                                    " is full self_attn, not linear_attn (" +
                                    raw_mod + ")");
      }
      const auto proj = sub.substr(12);
      if (proj == "in_proj_qkv") {
        emit_direct("gdn_qkv", gdn_qkv_width, h);
      } else if (proj == "in_proj_z") {
        emit_direct("gdn_z", gdn_v_width, h);
      } else if (proj == "in_proj_a") {
        emit_direct("gdn_a", nv, h);
      } else if (proj == "in_proj_b") {
        emit_direct("gdn_b", nv, h);
      } else if (proj == "out_proj") {
        emit_direct("gdn_out", h, gdn_v_width);
      } else if (proj == "in_proj_qkvz") {
        // De-interleave Qwen3-Next / HF [nk, 2*dk + 2*(nv/nk)*dv] into gdn_qkv and gdn_z!
        const std::uint32_t total_qkvz = gdn_qkv_width + gdn_v_width;
        auto cf = canonicalize_factors(spec, raw_mod, norm_mod, ta, tb,
                                       total_qkvz, h, fifo);
        const float sc = compute_lora_scale(spec, raw_mod, norm_mod, cf.rank);
        const std::uint32_t r = cf.rank;
        const std::uint32_t v_per_k = (nv / nk) * dv;
        const std::uint32_t group_stride = 2U * dk + 2U * v_per_k;
        std::vector<float> b_qkv(static_cast<std::size_t>(gdn_qkv_width) * r,
                                 0.0F);
        std::vector<float> b_z(static_cast<std::size_t>(gdn_v_width) * r, 0.0F);
        for (std::uint32_t kh = 0; kh < nk; ++kh) {
          const std::uint32_t src_base = kh * group_stride;
          // Q [0..dk) -> gdn_qkv[kh*dk .. (kh+1)*dk)
          std::memcpy(b_qkv.data() + static_cast<std::size_t>(kh * dk) * r,
                      cf.b_nr.data() + static_cast<std::size_t>(src_base) * r,
                      static_cast<std::size_t>(dk) * r * sizeof(float));
          // K [dk..2*dk) -> gdn_qkv[nk*dk + kh*dk .. nk*dk + (kh+1)*dk)
          std::memcpy(
              b_qkv.data() + static_cast<std::size_t>(nk * dk + kh * dk) * r,
              cf.b_nr.data() + static_cast<std::size_t>(src_base + dk) * r,
              static_cast<std::size_t>(dk) * r * sizeof(float));
          // V [2*dk..2*dk+v_per_k) -> gdn_qkv[2*nk*dk + kh*v_per_k ..]
          std::memcpy(
              b_qkv.data() +
                  static_cast<std::size_t>(2U * nk * dk + kh * v_per_k) * r,
              cf.b_nr.data() +
                  static_cast<std::size_t>(src_base + 2U * dk) * r,
              static_cast<std::size_t>(v_per_k) * r * sizeof(float));
          // Z [2*dk+v_per_k..2*dk+2*v_per_k) -> gdn_z[kh*v_per_k ..]
          std::memcpy(b_z.data() + static_cast<std::size_t>(kh * v_per_k) * r,
                      cf.b_nr.data() +
                          static_cast<std::size_t>(src_base + 2U * dk +
                                                   v_per_k) *
                              r,
                      static_cast<std::size_t>(v_per_k) * r * sizeof(float));
        }
        std::vector<float> a_copy = cf.a_rk;
        register_update(LoraProjectionUpdate{
            .target = {.is_global = false,
                       .layer_index = layer_idx,
                       .role = "gdn_qkv"},
            .source_module = norm_mod,
            .rank = r,
            .out_features = gdn_qkv_width,
            .in_features = h,
            .scale = sc,
            .a_f32 = std::move(cf.a_rk),
            .b_f32 = std::move(b_qkv),
        });
        register_update(LoraProjectionUpdate{
            .target =
                {.is_global = false, .layer_index = layer_idx, .role = "gdn_z"},
            .source_module = norm_mod,
            .rank = r,
            .out_features = gdn_v_width,
            .in_features = h,
            .scale = sc,
            .a_f32 = std::move(a_copy),
            .b_f32 = std::move(b_z),
        });
      } else if (proj == "in_proj_ba") {
        // De-interleave Qwen3-Next / HF [nk, 2*(nv/nk)] where b is first half and a is second half!
        auto cf = canonicalize_factors(spec, raw_mod, norm_mod, ta, tb,
                                       2U * nv, h, fifo);
        const float sc = compute_lora_scale(spec, raw_mod, norm_mod, cf.rank);
        const std::uint32_t r = cf.rank;
        const std::uint32_t g_per_k = nv / nk;
        std::vector<float> b_b(static_cast<std::size_t>(nv) * r, 0.0F);
        std::vector<float> b_a(static_cast<std::size_t>(nv) * r, 0.0F);
        for (std::uint32_t kh = 0; kh < nk; ++kh) {
          const std::uint32_t src_base = kh * 2U * g_per_k;
          std::memcpy(b_b.data() + static_cast<std::size_t>(kh * g_per_k) * r,
                      cf.b_nr.data() + static_cast<std::size_t>(src_base) * r,
                      static_cast<std::size_t>(g_per_k) * r * sizeof(float));
          std::memcpy(b_a.data() + static_cast<std::size_t>(kh * g_per_k) * r,
                      cf.b_nr.data() +
                          static_cast<std::size_t>(src_base + g_per_k) * r,
                      static_cast<std::size_t>(g_per_k) * r * sizeof(float));
        }
        std::vector<float> a_copy = cf.a_rk;
        register_update(LoraProjectionUpdate{
            .target =
                {.is_global = false, .layer_index = layer_idx, .role = "gdn_b"},
            .source_module = norm_mod,
            .rank = r,
            .out_features = nv,
            .in_features = h,
            .scale = sc,
            .a_f32 = std::move(cf.a_rk),
            .b_f32 = std::move(b_b),
        });
        register_update(LoraProjectionUpdate{
            .target =
                {.is_global = false, .layer_index = layer_idx, .role = "gdn_a"},
            .source_module = norm_mod,
            .rank = r,
            .out_features = nv,
            .in_features = h,
            .scale = sc,
            .a_f32 = std::move(a_copy),
            .b_f32 = std::move(b_a),
        });
      } else {
        throw std::invalid_argument("the model has no supported linear_attn "
                                    "projection '" +
                                    std::string(proj) + "'");
      }
      continue;
    }

    // MLP / MoE shared expert projections
    std::string_view mlp_sub = sub;
    if (mlp_sub.starts_with("mlp.")) {
      mlp_sub.remove_prefix(4);
    }
    if (mlp_sub == "gate" || mlp_sub == "router" ||
        mlp_sub.starts_with("experts.") ||
        mlp_sub.starts_with("switch_mlp.")) {
      throw std::invalid_argument(
          std::string(sub) +
          " is a MoE router or expert bank, which is not supported");
    }
    if (config.num_experts == 0) {
      if (mlp_sub == "gate_proj") {
        emit_direct("mlp_gate", f, h);
      } else if (mlp_sub == "up_proj") {
        emit_direct("mlp_up", f, h);
      } else if (mlp_sub == "down_proj") {
        emit_direct("mlp_down", h, f);
      } else {
        throw std::invalid_argument("the dense MLP layer has no projection '" +
                                    std::string(sub) + "'");
      }
    } else {
      if (mlp_sub == "shared_expert.gate_proj") {
        emit_direct("moe_shared_mlp_gate", shared_f, h);
      } else if (mlp_sub == "shared_expert.up_proj") {
        emit_direct("moe_shared_mlp_up", shared_f, h);
      } else if (mlp_sub == "shared_expert.down_proj") {
        emit_direct("moe_shared_mlp_down", h, shared_f);
      } else if (mlp_sub == "shared_expert_gate") {
        emit_direct("moe_shared_gate", 1U, h);
      } else {
        throw std::invalid_argument("the MoE layer has no supported LoRA "
                                    "projection '" +
                                    std::string(sub) + "'");
      }
    }
  }

  // Finalize any split q_proj / q_gate_proj into interleaved [q_heads, 2, head_dim] attention_q
  for (auto &[layer_idx, sq] : split_q_by_layer) {
    if (sq.q_only.has_value() && sq.gate_only.has_value()) {
      const auto &qh = *sq.q_only;
      const auto &gh = *sq.gate_only;
      const std::uint32_t r_tot = qh.rank + gh.rank;
      std::vector<float> a_comb(static_cast<std::size_t>(r_tot) * h, 0.0F);
      std::memcpy(a_comb.data(), qh.a_rk.data(),
                  qh.a_rk.size() * sizeof(float));
      std::memcpy(a_comb.data() + qh.a_rk.size(), gh.a_rk.data(),
                  gh.a_rk.size() * sizeof(float));
      std::vector<float> b_comb(static_cast<std::size_t>(2U * q_width) * r_tot,
                                0.0F);
      for (std::uint32_t hd = 0; hd < q_heads; ++hd) {
        for (std::uint32_t d = 0; d < head_dim; ++d) {
          const std::uint32_t src_row = hd * head_dim + d;
          const std::uint32_t dst_q_row = (hd * 2U + 0U) * head_dim + d;
          const std::uint32_t dst_g_row = (hd * 2U + 1U) * head_dim + d;
          for (std::uint32_t ri = 0; ri < qh.rank; ++ri) {
            b_comb[static_cast<std::size_t>(dst_q_row) * r_tot + ri] =
                qh.scale *
                qh.b_qr[static_cast<std::size_t>(src_row) * qh.rank + ri];
          }
          for (std::uint32_t ri = 0; ri < gh.rank; ++ri) {
            b_comb[static_cast<std::size_t>(dst_g_row) * r_tot + qh.rank + ri] =
                gh.scale *
                gh.b_qr[static_cast<std::size_t>(src_row) * gh.rank + ri];
          }
        }
      }
      register_update(LoraProjectionUpdate{
          .target = {.is_global = false,
                     .layer_index = layer_idx,
                     .role = "attention_q"},
          .source_module = qh.source + "+" + gh.source,
          .rank = r_tot,
          .out_features = 2U * q_width,
          .in_features = h,
          .scale = 1.0F,
          .a_f32 = std::move(a_comb),
          .b_f32 = std::move(b_comb),
      });
    } else {
      const bool is_gate = sq.gate_only.has_value();
      auto &half = is_gate ? *sq.gate_only : *sq.q_only;
      const std::uint32_t r = half.rank;
      std::vector<float> b_full(static_cast<std::size_t>(2U * q_width) * r,
                                0.0F);
      for (std::uint32_t hd = 0; hd < q_heads; ++hd) {
        for (std::uint32_t d = 0; d < head_dim; ++d) {
          const std::uint32_t src_row = hd * head_dim + d;
          const std::uint32_t dst_row =
              (hd * 2U + (is_gate ? 1U : 0U)) * head_dim + d;
          std::memcpy(
              b_full.data() + static_cast<std::size_t>(dst_row) * r,
              half.b_qr.data() + static_cast<std::size_t>(src_row) * r,
              static_cast<std::size_t>(r) * sizeof(float));
        }
      }
      register_update(LoraProjectionUpdate{
          .target = {.is_global = false,
                     .layer_index = layer_idx,
                     .role = "attention_q"},
          .source_module = std::move(half.source),
          .rank = r,
          .out_features = 2U * q_width,
          .in_features = h,
          .scale = half.scale,
          .a_f32 = std::move(half.a_rk),
          .b_f32 = std::move(b_full),
      });
    }
  }

  std::ranges::sort(updates, [](const LoraProjectionUpdate &lhs,
                                const LoraProjectionUpdate &rhs) {
    if (lhs.target.is_global != rhs.target.is_global) {
      return !lhs.target.is_global;
    }
    if (lhs.target.layer_index != rhs.target.layer_index) {
      return lhs.target.layer_index < rhs.target.layer_index;
    }
    return lhs.target.role < rhs.target.role;
  });

  std::set<std::uint32_t> adapted_decoder_layers;
  bool adapted_lm_head = false;
  for (const auto &u : updates) {
    if (u.target.is_global) {
      adapted_lm_head = true;
    } else {
      adapted_decoder_layers.insert(u.target.layer_index);
    }
  }

  std::ostringstream fp_stream;
  fp_stream << spec.config_canonical << "|"
            << (spec.format == LoraFormat::peft ? "peft" : "mlx") << "|";
  struct stat w_stat{};
  if (::stat(spec.weights_path.c_str(), &w_stat) == 0) {
    fp_stream << w_stat.st_size << ":" << w_stat.st_mtimespec.tv_sec << ":"
              << w_stat.st_mtimespec.tv_nsec << "|";
  }
  for (const auto &tname : st.tensor_names()) {
    const auto &ti = st.tensor_info(tname);
    fp_stream << tname << ":" << dtype_name(ti.dtype) << ":" << ti.data_offset
              << ":" << ti.nbytes << "|";
  }
  const std::string fp_str = fp_stream.str();
  const std::string fp_hex = artifact_sha256_hex(
      std::as_bytes(std::span(fp_str.data(), fp_str.size())));

  return LoadedLoraAdapter{
      .adapter_dir = canonical_dir,
      .format = spec.format,
      .fingerprint = fp_hex.substr(0, 16),
      .adapted_layers = static_cast<std::uint32_t>(
          adapted_decoder_layers.size() + (adapted_lm_head ? 1U : 0U)),
      .updates = std::move(updates),
  };
}

} // namespace omlx::m5
