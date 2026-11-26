#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task_info.h>
#endif

#include <nlohmann/json.hpp>

#include "chunk_blend_cache.h"
#include "omlx_m5/artifact.h"
#include "omlx_m5/execution_contract.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/qwen_forward.h"
#include "omlx_m5/types.h"
#include "serve/config.h"
#include "serve/tokenizer.h"

namespace {

using namespace omlx::m5;
using Json = nlohmann::json;

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

double bytes_to_gib(std::uint64_t bytes) noexcept {
  return static_cast<double>(bytes) / static_cast<double>(1ULL << 30);
}

std::string read_file_bytes(const std::filesystem::path &path, bool required) {
  std::ifstream input(path, std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    if (required) {
      throw std::runtime_error("failed to open file: " + path.string());
    }
    return {};
  }
  std::ostringstream out;
  out << input.rdbuf();
  return out.str();
}

void write_float_binary(const std::filesystem::path &path,
                        std::span<const float> values) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open logits output file: " +
                             path.string());
  }
  out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!out.good()) {
    throw std::runtime_error("failed writing logits file: " + path.string());
  }
}

std::vector<float> read_float_binary(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::in | std::ios::binary | std::ios::ate);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open logits binary file: " +
                             path.string());
  }
  const auto byte_size = in.tellg();
  if (byte_size <= 0 || (byte_size % static_cast<std::streamoff>(sizeof(float))) != 0) {
    throw std::runtime_error("invalid float32 logits file size: " +
                             path.string());
  }
  in.seekg(0, std::ios::beg);
  std::vector<float> values(static_cast<std::size_t>(byte_size) / sizeof(float));
  in.read(reinterpret_cast<char *>(values.data()), byte_size);
  if (!in.good()) {
    throw std::runtime_error("failed reading logits binary file: " +
                             path.string());
  }
  return values;
}

serve::Tokenizer load_model_tokenizer(const std::filesystem::path &model_dir) {
  const std::string tok_json =
      read_file_bytes(model_dir / "tokenizer.json", true);
  const std::string tok_cfg_json =
      read_file_bytes(model_dir / "tokenizer_config.json", false);
  return serve::Tokenizer({
      .tokenizer_json = tok_json,
      .tokenizer_config_json = tok_cfg_json,
  });
}

struct RowLogSoftmaxStats {
  double logsumexp = 0.0;
  double max_logit = -std::numeric_limits<double>::infinity();
  std::uint32_t argmax_id = 0;
  double l2_norm = 0.0;
};

RowLogSoftmaxStats compute_row_stats(std::span<const float> logits) {
  if (logits.empty()) {
    throw std::invalid_argument("empty logits row");
  }
  double max_val = -std::numeric_limits<double>::infinity();
  std::uint32_t argmax = 0;
  double sq_sum = 0.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const double v = static_cast<double>(logits[i]);
    if (!std::isfinite(v)) {
      throw std::runtime_error("non-finite logit encountered at vocab index " +
                               std::to_string(i));
    }
    sq_sum += v * v;
    if (v > max_val) {
      max_val = v;
      argmax = static_cast<std::uint32_t>(i);
    }
  }
  double exp_sum = 0.0;
  for (const float vf : logits) {
    exp_sum += std::exp(static_cast<double>(vf) - max_val);
  }
  return RowLogSoftmaxStats{
      .logsumexp = max_val + std::log(exp_sum),
      .max_logit = max_val,
      .argmax_id = argmax,
      .l2_norm = std::sqrt(sq_sum),
  };
}

Json top_k_logits_json(std::span<const float> logits, double logsumexp,
                       std::size_t k) {
  const std::size_t keep = std::min(k, logits.size());
  std::vector<std::uint32_t> idx(logits.size());
  std::iota(idx.begin(), idx.end(), 0U);
  std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(keep),
                    idx.end(), [&](std::uint32_t a, std::uint32_t b) {
                      return logits[a] > logits[b];
                    });
  Json arr = Json::array();
  for (std::size_t i = 0; i < keep; ++i) {
    const std::uint32_t tok = idx[i];
    const double val = static_cast<double>(logits[tok]);
    const double lp = val - logsumexp;
    arr.push_back({
        {"token_id", tok},
        {"logit", val},
        {"logp", lp},
        {"prob", std::exp(lp)},
    });
  }
  return arr;
}

Json compare_logit_vectors(std::span<const float> ref,
                           std::span<const float> cand) {
  if (ref.size() != cand.size() || ref.empty()) {
    throw std::invalid_argument("logit vector size mismatch: ref=" +
                                std::to_string(ref.size()) +
                                " cand=" + std::to_string(cand.size()));
  }
  const auto ref_stats = compute_row_stats(ref);
  const auto cand_stats = compute_row_stats(cand);

  double dot = 0.0;
  double norm_ref_sq = 0.0;
  double norm_cand_sq = 0.0;
  double diff_sq = 0.0;
  double max_abs_diff = 0.0;
  double kl_ref_cand = 0.0;
  double kl_cand_ref = 0.0;
  double js_div = 0.0;

  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double a = static_cast<double>(ref[i]);
    const double b = static_cast<double>(cand[i]);
    dot += a * b;
    norm_ref_sq += a * a;
    norm_cand_sq += b * b;
    const double d = a - b;
    diff_sq += d * d;
    max_abs_diff = std::max(max_abs_diff, std::abs(d));

    const double lp_ref = a - ref_stats.logsumexp;
    const double lp_cand = b - cand_stats.logsumexp;
    const double p = std::exp(lp_ref);
    const double q = std::exp(lp_cand);
    if (p > 0.0) {
      kl_ref_cand += p * (lp_ref - lp_cand);
    }
    if (q > 0.0) {
      kl_cand_ref += q * (lp_cand - lp_ref);
    }
    const double m = 0.5 * (p + q);
    if (m > 0.0) {
      if (p > 0.0) {
        js_div += 0.5 * p * std::log(p / m);
      }
      if (q > 0.0) {
        js_div += 0.5 * q * std::log(q / m);
      }
    }
  }

  const double denom = std::sqrt(norm_ref_sq) * std::sqrt(norm_cand_sq);
  const double cos_sim = denom > 0.0 ? (dot / denom) : 0.0;
  const double l2_rel_err =
      norm_ref_sq > 0.0 ? std::sqrt(diff_sq / norm_ref_sq) : 0.0;

  auto top_n_ids = [](std::span<const float> vec, std::size_t n) {
    const std::size_t keep = std::min(n, vec.size());
    std::vector<std::uint32_t> idx(vec.size());
    std::iota(idx.begin(), idx.end(), 0U);
    std::partial_sort(idx.begin(),
                      idx.begin() + static_cast<std::ptrdiff_t>(keep),
                      idx.end(), [&](std::uint32_t x, std::uint32_t y) {
                        return vec[x] > vec[y];
                      });
    idx.resize(keep);
    return idx;
  };
  auto overlap_count = [](const std::vector<std::uint32_t> &a,
                          const std::vector<std::uint32_t> &b) {
    std::size_t count = 0;
    for (const auto x : a) {
      if (std::find(b.begin(), b.end(), x) != b.end()) {
        ++count;
      }
    }
    return count;
  };

  const auto ref_top5 = top_n_ids(ref, 5);
  const auto cand_top5 = top_n_ids(cand, 5);
  const auto ref_top16 = top_n_ids(ref, 16);
  const auto cand_top16 = top_n_ids(cand, 16);

  return {
      {"vocab_size", ref.size()},
      {"cosine_similarity", cos_sim},
      {"kl_ref_to_cand", kl_ref_cand},
      {"kl_cand_to_ref", kl_cand_ref},
      {"js_divergence", js_div},
      {"max_abs_diff", max_abs_diff},
      {"l2_rel_err", l2_rel_err},
      {"top1_ref", ref_stats.argmax_id},
      {"top1_cand", cand_stats.argmax_id},
      {"top1_match", ref_stats.argmax_id == cand_stats.argmax_id},
      {"top5_overlap", overlap_count(ref_top5, cand_top5)},
      {"top16_overlap", overlap_count(ref_top16, cand_top16)},
  };
}

std::filesystem::path resolve_artifact_path(
    const std::filesystem::path &model_dir,
    const std::filesystem::path &explicit_artifact) {
  if (!explicit_artifact.empty()) {
    return explicit_artifact;
  }
  const auto miv = model_dir / "model.miv";
  if (std::filesystem::is_regular_file(miv)) {
    return miv;
  }
  const auto legacy = model_dir / "model.omlxm5";
  if (std::filesystem::is_regular_file(legacy)) {
    return legacy;
  }
  throw std::runtime_error("no model.miv or model.omlxm5 found in " +
                           model_dir.string());
}

struct CliOptions {
  bool show_help = false;
  std::string mode = "ppl"; // tokenize | ppl | prefill-logits | compare-logits
  std::filesystem::path model_dir;
  std::filesystem::path artifact;
  std::filesystem::path metallib_dir;
  std::filesystem::path text_file;
  std::string text_inline;
  std::filesystem::path tokens_file;
  std::vector<std::uint32_t> tokens_inline;
  std::filesystem::path out_file;
  std::filesystem::path logits_out;
  std::filesystem::path compare_logits;
  std::filesystem::path ref_logits;
  std::filesystem::path cand_logits;

  std::string kv_dtype = "affine4_g128";
  std::string sampling_contract = kDefaultServingSamplingContract;
  std::uint32_t max_context = 32768;
  std::uint32_t chunk_size = 512;
  std::uint32_t max_tokens = 0;
  std::uint32_t max_new_tokens = 0;
  std::size_t top_k_report = 16;

  // Orthogonal Wave 2/3 prefill switches (propagated via env + future API hooks).
  std::string prefill_linear_mode = "bf16";
  std::string prefill_attention_mode = "exact_nax";
  bool prefill_sparse_minference = false;
  bool prefill_cache_blend = false;
  bool prefill_ane_offload = false;
};

std::string format_help() {
  return R"(Usage: omlx_m5_prefill_matrix_bench --mode <tokenize|ppl|prefill-logits|compare-logits> [options]

Modes:
  tokenize        Tokenize text using exact ninfer-metal-serve BPE tokenizer
  ppl             Teacher-forced chunked logprob / PPL evaluation + final-token logits
  prefill-logits  Production-chunk prefill + final-token logits + optional greedy decode
  compare-logits  Compare two saved float32 logit vectors (cosine, KL, JS, top-k overlap)

Common options:
  --model-dir PATH                 Model directory containing config.json, tokenizer.json, model.miv
  --artifact PATH                  Optional explicit .miv artifact override
  --metallib-dir PATH              Metal .metallib shader directory
  --kv-dtype affine4_g128|bf16     KV cache format (default: affine4_g128)
  --sampling-contract NAME         Sampling contract (default: position_coupled_v1)
  --max-context N                  Max sequence context (default: 32768)
  --chunk-size N                   Tokens per teacher-forced chunk in ppl mode (default: 512)
  --max-tokens N                   Optional truncation length for input sequence
  --max-new-tokens N               Greedy decode tokens in prefill-logits mode (default: 0)
  --text-file PATH                 UTF-8 text input file
  --text STRING                    Inline UTF-8 text input
  --tokens-file PATH               JSON file containing [int,...] or {"tokens":[int,...]}
  --tokens ID,ID,...               Comma-separated token IDs
  --out PATH                       Optional JSON output file (in addition to stdout)
  --logits-out PATH                Optional binary float32 dump of final-token logits
  --compare-logits PATH            Optional reference float32 logits for cosine/KL check
  --ref PATH --cand PATH           Binary float32 logit files for compare-logits mode

Orthogonal prefill switches (for Wave 2/3 runtime builds):
  --prefill-linear-mode bf16|w4a8
  --prefill-attention-mode exact_nax|sage2
  --prefill-sparse-minference / --no-prefill-sparse-minference
  --prefill-cache-blend / --no-prefill-cache-blend
  --prefill-ane-offload / --no-prefill-ane-offload
)";
}

std::uint32_t parse_u32(std::string_view text, std::string_view flag) {
  if (text.empty() || text.front() == '-') {
    throw std::invalid_argument(std::string(flag) + " requires nonnegative int");
  }
  std::size_t used = 0;
  const unsigned long long val = std::stoull(std::string(text), &used);
  if (used != text.size() || val > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("invalid uint32 for " + std::string(flag) +
                                ": " + std::string(text));
  }
  return static_cast<std::uint32_t>(val);
}

CliOptions parse_args(int argc, char **argv) {
  CliOptions o;
  for (int i = 1; i < argc; ++i) {
    std::string raw(argv[i]);
    if (raw == "-h" || raw == "--help") {
      o.show_help = true;
      return o;
    }
    std::string key = raw;
    std::optional<std::string> inline_val;
    if (raw.starts_with("--")) {
      const auto eq = raw.find('=');
      if (eq != std::string::npos) {
        key = raw.substr(0, eq);
        inline_val = raw.substr(eq + 1);
      }
    }
    auto value = [&]() -> std::string {
      if (inline_val.has_value()) {
        return *inline_val;
      }
      if (++i >= argc) {
        throw std::invalid_argument("missing value for " + key);
      }
      return argv[i];
    };

    if (key == "--mode") {
      o.mode = value();
    } else if (key == "--model-dir") {
      o.model_dir = serve::expand_user_path(value());
    } else if (key == "--artifact") {
      o.artifact = serve::expand_user_path(value());
    } else if (key == "--metallib-dir" || key == "--kernel-dir") {
      o.metallib_dir = serve::expand_user_path(value());
    } else if (key == "--text-file") {
      o.text_file = serve::expand_user_path(value());
    } else if (key == "--text") {
      o.text_inline = value();
    } else if (key == "--tokens-file") {
      o.tokens_file = serve::expand_user_path(value());
    } else if (key == "--tokens") {
      std::stringstream ss(value());
      std::string item;
      while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
          o.tokens_inline.push_back(parse_u32(item, "--tokens"));
        }
      }
    } else if (key == "--out") {
      o.out_file = serve::expand_user_path(value());
    } else if (key == "--logits-out") {
      o.logits_out = serve::expand_user_path(value());
    } else if (key == "--compare-logits") {
      o.compare_logits = serve::expand_user_path(value());
    } else if (key == "--ref") {
      o.ref_logits = serve::expand_user_path(value());
    } else if (key == "--cand") {
      o.cand_logits = serve::expand_user_path(value());
    } else if (key == "--kv-dtype") {
      o.kv_dtype = value();
    } else if (key == "--sampling-contract") {
      o.sampling_contract = value();
    } else if (key == "--max-context") {
      o.max_context = parse_u32(value(), key);
    } else if (key == "--chunk-size") {
      o.chunk_size = parse_u32(value(), key);
    } else if (key == "--max-tokens") {
      o.max_tokens = parse_u32(value(), key);
    } else if (key == "--max-new-tokens") {
      o.max_new_tokens = parse_u32(value(), key);
    } else if (key == "--top-k-report") {
      o.top_k_report = parse_u32(value(), key);
    } else if (key == "--prefill-linear-mode") {
      o.prefill_linear_mode = value();
    } else if (key == "--prefill-attention-mode") {
      o.prefill_attention_mode = value();
    } else if (key == "--prefill-sparse-minference") {
      o.prefill_sparse_minference = true;
    } else if (key == "--no-prefill-sparse-minference") {
      o.prefill_sparse_minference = false;
    } else if (key == "--prefill-cache-blend") {
      o.prefill_cache_blend = true;
    } else if (key == "--no-prefill-cache-blend") {
      o.prefill_cache_blend = false;
    } else if (key == "--prefill-ane-offload") {
      o.prefill_ane_offload = true;
    } else if (key == "--no-prefill-ane-offload") {
      o.prefill_ane_offload = false;
    } else {
      throw std::invalid_argument("unknown argument: " + key);
    }
  }
  return o;
}

void export_prefill_switch_env(const CliOptions &o) {
  if (o.prefill_linear_mode != "bf16") {
    ::setenv("NINFER_PREFILL_LINEAR_MODE", o.prefill_linear_mode.c_str(), 1);
  }
  if (o.prefill_attention_mode != "exact_nax") {
    ::setenv("NINFER_PREFILL_ATTENTION_MODE", o.prefill_attention_mode.c_str(),
             1);
  }
  if (o.prefill_sparse_minference) {
    ::setenv("NINFER_PREFILL_SPARSE_MINFERENCE", "1", 1);
  }
  if (o.prefill_cache_blend) {
    ::setenv("NINFER_PREFILL_CACHE_BLEND", "1", 1);
  }
  if (o.prefill_ane_offload) {
    ::setenv("NINFER_PREFILL_ANE_OFFLOAD", "1", 1);
  }
}

std::vector<std::uint32_t> load_input_tokens(const CliOptions &o) {
  std::vector<std::uint32_t> tokens;
  if (!o.tokens_inline.empty()) {
    tokens = o.tokens_inline;
  } else if (!o.tokens_file.empty()) {
    const Json parsed = Json::parse(read_file_bytes(o.tokens_file, true));
    const Json &arr = parsed.is_object() && parsed.contains("tokens")
                          ? parsed.at("tokens")
                          : parsed;
    if (!arr.is_array()) {
      throw std::invalid_argument("--tokens-file must contain a JSON array or {\"tokens\": [...]}");
    }
    tokens.reserve(arr.size());
    for (const auto &el : arr) {
      if (!el.is_number_integer()) {
        throw std::invalid_argument("token id must be integer");
      }
      const auto id = el.get<std::int64_t>();
      if (id < 0 || id > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("token id out of range");
      }
      tokens.push_back(static_cast<std::uint32_t>(id));
    }
  } else if (!o.text_file.empty() || !o.text_inline.empty()) {
    if (o.model_dir.empty()) {
      throw std::invalid_argument("--model-dir is required to tokenize text");
    }
    const auto tok = load_model_tokenizer(o.model_dir);
    const std::string text = !o.text_file.empty()
                                 ? read_file_bytes(o.text_file, true)
                                 : o.text_inline;
    const auto encoded = tok.encode(text);
    tokens.reserve(encoded.size());
    for (const int id : encoded) {
      tokens.push_back(static_cast<std::uint32_t>(id));
    }
  } else {
    throw std::invalid_argument(
        "specify one of --tokens, --tokens-file, --text-file, or --text");
  }

  if (o.max_tokens > 0 && tokens.size() > o.max_tokens) {
    tokens.resize(o.max_tokens);
  }
  return tokens;
}

void emit_json(const Json &payload, const std::filesystem::path &out_file) {
  const std::string serialized = payload.dump(2);
  if (!out_file.empty()) {
    if (out_file.has_parent_path()) {
      std::filesystem::create_directories(out_file.parent_path());
    }
    std::ofstream out(out_file, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
      throw std::runtime_error("failed to write output file: " +
                               out_file.string());
    }
    out << serialized << '\n';
  }
  std::cout << serialized << '\n' << std::flush;
}

struct LoadedModel {
  LoadedModel() = default;
  LoadedModel(const LoadedModel &) = delete;
  LoadedModel &operator=(const LoadedModel &) = delete;
  LoadedModel(LoadedModel &&) = delete;
  LoadedModel &operator=(LoadedModel &&) = delete;

  std::optional<MetalContext> context;
  QwenModelConfig config;
  std::filesystem::path artifact_path;
  std::filesystem::path metallib_dir;
  SamplingContract contract = SamplingContract::PositionCoupledV1;
  std::optional<MaterializedModel> weights;
  std::optional<QwenLayerCatalog> layers;
  // Declared LAST so it is destroyed FIRST while context/weights/layers live.
  std::unique_ptr<QwenForward> forward;
  double load_wall_s = 0.0;
};

std::unique_ptr<LoadedModel> load_forward_model(const CliOptions &o,
                                                std::uint32_t max_context) {
  if (o.model_dir.empty()) {
    throw std::invalid_argument("--model-dir is required");
  }
  const auto t0 = std::chrono::steady_clock::now();
  auto loaded = std::make_unique<LoadedModel>();
  loaded->artifact_path = resolve_artifact_path(o.model_dir, o.artifact);
  loaded->metallib_dir = std::filesystem::path(
      serve::resolve_kernel_dir(o.metallib_dir.string(), o.model_dir.string()));
  loaded->config = load_qwen_model_config(o.model_dir / "config.json");
  loaded->context.emplace(MetalContext::create());
  auto artifact = ArtifactModel::open(loaded->artifact_path, false);
  artifact.verify_payload();
  auto mat_plan = plan_model_materialization(
      artifact, *loaded->context,
      {.components = {ModelComponent::text}, .require_verified_payload = true});
  loaded->weights.emplace(
      materialize_model(artifact, std::move(mat_plan), *loaded->context));
  loaded->layers.emplace(
      build_qwen_layer_catalog(artifact, loaded->config, max_context, 64));

  loaded->contract = parse_sampling_contract(o.sampling_contract);
  const bool coupled = loaded->contract == SamplingContract::PositionCoupledV1;
  const auto lin_policy =
      coupled ? LinearNumericPolicy::RowExactPortableV1 : LinearNumericPolicy::Legacy;
  const auto attn_policy =
      coupled ? AttentionNumericPolicy::RowExactV1 : AttentionNumericPolicy::Legacy;
  const auto &cap = loaded->context->capabilities();
  const bool apple9_or_newer = cap.profile.probe.apple_gpu_family >= 9;
  const bool apple10_nax =
      (cap.device_name.starts_with("Apple M5") ||
       cap.profile.probe.apple_gpu_family >= 10) &&
      cap.profile.probe.runtime_os_major >= 27 &&
      cap.profile.probe.metal4_compiler && cap.profile.probe.tensor_api;
  const bool affine4_kv = (o.kv_dtype == "affine4_g128");
  const bool flash_nax_eligible =
      loaded->config.hidden_size >= 1024 &&
      loaded->config.full_attention_layers() > 0 &&
      (affine4_kv
           ? (loaded->config.head_dim == 256 &&
              std::filesystem::is_regular_file(
                  loaded->metallib_dir / "omlx_m5_attention_affine4.metallib"))
           : ((loaded->config.head_dim == 256 ||
               loaded->config.head_dim == 128 ||
               loaded->config.head_dim == 64) &&
              std::filesystem::is_regular_file(
                  loaded->metallib_dir /
                  "omlx_m5_attention_nax_prefill.metallib")));
  const bool enable_flash_nax_prefill = flash_nax_eligible && apple10_nax;

  const auto artifact_digest = execution_artifact_digest(artifact);
  QwenPrefillOptions pf_opts{
      .linear_mode = parse_prefill_linear_mode(o.prefill_linear_mode),
      .attention_mode = parse_prefill_attention_mode(o.prefill_attention_mode),
      .sparse_minference = o.prefill_sparse_minference,
      .cache_blend = o.prefill_cache_blend,
      .ane_offload = o.prefill_ane_offload,
      .artifact_digest = artifact_digest,
  };
  loaded->forward = std::make_unique<QwenForward>(
      *loaded->context, *loaded->weights, *loaded->layers, loaded->config,
      max_context, loaded->metallib_dir, std::span<const std::uint32_t>{},
      false, false, cap.device_name == "Apple M4", std::nullopt,
      apple9_or_newer, false, std::nullopt, lin_policy, loaded->contract,
      attn_policy, false, o.kv_dtype, nullptr, enable_flash_nax_prefill, 0U,
      nullptr, nullptr, pf_opts);

  loaded->load_wall_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  return loaded;
}

int run_tokenize(const CliOptions &o) {
  if (o.model_dir.empty()) {
    throw std::invalid_argument("--model-dir is required for tokenize mode");
  }
  const auto tok = load_model_tokenizer(o.model_dir);
  std::string text;
  if (!o.text_file.empty()) {
    text = read_file_bytes(o.text_file, true);
  } else if (!o.text_inline.empty()) {
    text = o.text_inline;
  } else {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    text = ss.str();
  }
  auto ids = tok.encode(text);
  if (o.max_tokens > 0 && ids.size() > o.max_tokens) {
    ids.resize(o.max_tokens);
  }
  emit_json({{"mode", "tokenize"},
             {"model_dir", o.model_dir.string()},
             {"count", ids.size()},
             {"tokens", ids}},
            o.out_file);
  return 0;
}

int run_compare_logits(const CliOptions &o) {
  if (o.ref_logits.empty() || o.cand_logits.empty()) {
    throw std::invalid_argument(
        "compare-logits requires --ref PATH and --cand PATH");
  }
  const auto ref = read_float_binary(o.ref_logits);
  const auto cand = read_float_binary(o.cand_logits);
  Json out = compare_logit_vectors(ref, cand);
  out["mode"] = "compare-logits";
  out["ref"] = o.ref_logits.string();
  out["cand"] = o.cand_logits.string();
  emit_json(out, o.out_file);
  return 0;
}

int run_ppl(const CliOptions &o) {
  export_prefill_switch_env(o);
  const auto tokens = load_input_tokens(o);
  if (tokens.size() < 2) {
    throw std::invalid_argument("ppl mode requires at least 2 input tokens");
  }
  const std::uint32_t effective_ctx =
      std::max<std::uint32_t>(o.max_context,
                              static_cast<std::uint32_t>(tokens.size() + 64U));
  auto loaded = load_forward_model(o, effective_ctx);
  const std::uint32_t ws_rows = loaded->forward->memory_plan().workspace_rows;
  const std::uint32_t chunk_cap =
      std::clamp<std::uint32_t>(o.chunk_size, 16U, ws_rows);
  const std::size_t vocab_size = loaded->forward->vocabulary_size();

  std::uint64_t peak_phys = current_phys_footprint_bytes();
  double total_nll = 0.0;
  std::size_t scored_tokens = 0;
  std::size_t top1_hits = 0;
  Json per_chunk = Json::array();
  std::vector<float> last_logits;
  RowLogSoftmaxStats last_stats{};

  const auto eval_t0 = std::chrono::steady_clock::now();
  std::size_t offset = 0;
  while (offset < tokens.size()) {
    std::size_t remaining = tokens.size() - offset;
    std::size_t step = std::min<std::size_t>(remaining, chunk_cap);
    // Keep chunk size either >8 or ==1 so inspect_tensors_workspace==false
    // always takes the full chunked head-projection or single-row path safely.
    if (step > 1 && step <= 8) {
      step = 1;
    } else if (remaining > step && (remaining - step) > 1 &&
               (remaining - step) <= 8 && step > 16) {
      step -= 8;
    }

    const auto c_t0 = std::chrono::steady_clock::now();
    GpuSamplingOptions sampling{};
    sampling.absolute_position = offset + step - 1;
    sampling.contract = loaded->contract;
    const auto res = loaded->forward->run(
        std::span<const std::uint32_t>(tokens.data() + offset, step),
        sampling, /*inspect_tensors=*/true, /*use_icb=*/true,
        /*profile_stages=*/false);
    const double chunk_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - c_t0)
            .count();

    if (res.logits.size() != step * vocab_size) {
      throw std::runtime_error(
          "unexpected logits count from QwenForward::run: got " +
          std::to_string(res.logits.size()) + " expected " +
          std::to_string(step * vocab_size));
    }

    double chunk_nll = 0.0;
    std::size_t chunk_scored = 0;
    std::size_t chunk_hits = 0;
    for (std::size_t r = 0; r < step; ++r) {
      const std::size_t global_pos = offset + r;
      const std::span<const float> row(res.logits.data() + r * vocab_size,
                                       vocab_size);
      const auto stats = compute_row_stats(row);
      if (global_pos + 1 < tokens.size()) {
        const std::uint32_t target = tokens[global_pos + 1];
        if (target >= vocab_size) {
          throw std::runtime_error("target token out of vocab range");
        }
        const double logp =
            static_cast<double>(row[target]) - stats.logsumexp;
        const double nll = -logp;
        total_nll += nll;
        chunk_nll += nll;
        ++scored_tokens;
        ++chunk_scored;
        if (stats.argmax_id == target) {
          ++top1_hits;
          ++chunk_hits;
        }
      }
      if (global_pos + 1 == tokens.size()) {
        last_logits.assign(row.begin(), row.end());
        last_stats = stats;
      }
    }

    const std::uint64_t phys_now = current_phys_footprint_bytes();
    peak_phys = std::max(peak_phys, phys_now);
    per_chunk.push_back({
        {"offset", offset},
        {"tokens", step},
        {"scored", chunk_scored},
        {"mean_nll", chunk_scored > 0 ? (chunk_nll / double(chunk_scored)) : 0.0},
        {"ppl", chunk_scored > 0 ? std::exp(chunk_nll / double(chunk_scored)) : 0.0},
        {"top1_acc", chunk_scored > 0 ? (double(chunk_hits) / double(chunk_scored)) : 0.0},
        {"wall_s", chunk_s},
        {"phys_footprint_gib", bytes_to_gib(phys_now)},
    });
    offset += step;
  }
  const double eval_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - eval_t0)
          .count();

  if (!o.logits_out.empty() && !last_logits.empty()) {
    write_float_binary(o.logits_out, last_logits);
  }

  const double mean_nll =
      scored_tokens > 0 ? (total_nll / static_cast<double>(scored_tokens)) : 0.0;
  const double ppl = std::exp(mean_nll);
  const double top1_acc =
      scored_tokens > 0 ? (static_cast<double>(top1_hits) /
                           static_cast<double>(scored_tokens))
                        : 0.0;

  const auto pf_tel = loaded->forward->prefill_telemetry();
  Json report = {
      {"mode", "ppl"},
      {"model_dir", o.model_dir.string()},
      {"artifact", loaded->artifact_path.string()},
      {"metallib_dir", loaded->metallib_dir.string()},
      {"device_name", loaded->context->capabilities().device_name},
      {"kv_dtype", o.kv_dtype},
      {"sampling_contract", o.sampling_contract},
      {"max_context", effective_ctx},
      {"workspace_rows", ws_rows},
      {"chunk_size", chunk_cap},
      {"prefill_switches",
       {
           {"prefill_linear_mode", o.prefill_linear_mode},
           {"prefill_attention_mode", o.prefill_attention_mode},
           {"prefill_sparse_minference", o.prefill_sparse_minference},
           {"prefill_cache_blend", o.prefill_cache_blend},
           {"prefill_ane_offload", o.prefill_ane_offload},
       }},
      {"prefill_telemetry",
       {
           {"prefill_w4a8_tokens", pf_tel.prefill_w4a8_tokens},
           {"prefill_sage2_tokens", pf_tel.prefill_sage2_tokens},
           {"prefill_minference_blocks_skipped",
            pf_tel.prefill_minference_blocks_skipped},
           {"prefill_minference_blocks_total",
            pf_tel.prefill_minference_blocks_total},
           {"prefill_ane_active", pf_tel.prefill_ane_active},
           {"prefill_ane_split_alpha", pf_tel.prefill_ane_split_alpha},
           {"prefill_ane_offload_tokens", pf_tel.prefill_ane_offload_tokens},
           {"prefill_ane_fallback_tokens", pf_tel.prefill_ane_fallback_tokens},
           {"prefill_cache_blend_hit_tokens",
            pf_tel.prefill_cache_blend_hit_tokens},
           {"prefill_cache_blend_recomputed_tokens",
            pf_tel.prefill_cache_blend_recomputed_tokens},
       }},
      {"num_tokens", tokens.size()},
      {"scored_tokens", scored_tokens},
      {"mean_nll", mean_nll},
      {"ppl", ppl},
      {"top1_accuracy", top1_acc},
      {"load_wall_s", loaded->load_wall_s},
      {"eval_wall_s", eval_s},
      {"eval_tok_per_s", eval_s > 0.0 ? (double(tokens.size()) / eval_s) : 0.0},
      {"peak_phys_footprint_bytes", peak_phys},
      {"peak_phys_footprint_gib", bytes_to_gib(peak_phys)},
      {"metal_allocated_bytes", loaded->context->current_allocated_bytes()},
      {"last_logits_l2", last_stats.l2_norm},
      {"last_token_argmax", last_stats.argmax_id},
      {"last_token_topk",
       top_k_logits_json(last_logits, last_stats.logsumexp, o.top_k_report)},
      {"per_chunk", std::move(per_chunk)},
  };
  if (!o.logits_out.empty()) {
    report["logits_out"] = o.logits_out.string();
  }
  if (!o.compare_logits.empty() && !last_logits.empty()) {
    const auto ref = read_float_binary(o.compare_logits);
    report["logit_comparison"] = compare_logit_vectors(ref, last_logits);
  }
  emit_json(report, o.out_file);
  return 0;
}

int run_prefill_logits(const CliOptions &o) {
  export_prefill_switch_env(o);
  const auto tokens = load_input_tokens(o);
  if (tokens.empty()) {
    throw std::invalid_argument("prefill-logits requires non-empty input");
  }
  const std::uint32_t needed_ctx = static_cast<std::uint32_t>(
      tokens.size() + std::max<std::uint32_t>(o.max_new_tokens, 1U));
  const std::uint32_t effective_ctx = std::max(o.max_context, needed_ctx);
  auto loaded = load_forward_model(o, effective_ctx);
  const std::uint32_t ws_rows = loaded->forward->memory_plan().workspace_rows;
  std::uint64_t peak_phys = current_phys_footprint_bytes();

  const auto pf_t0 = std::chrono::steady_clock::now();
  // Run trunk on prefix tokens [0 .. L-1) in production ws_rows chunks.
  const std::size_t prefix_len = tokens.size() - 1;
  std::size_t offset = 0;
  while (offset < prefix_len) {
    const std::size_t step = std::min<std::size_t>(ws_rows, prefix_len - offset);
    static_cast<void>(loaded->forward->run_trunk(
        std::span<const std::uint32_t>(tokens.data() + offset, step),
        /*use_icb=*/true));
    peak_phys = std::max(peak_phys, current_phys_footprint_bytes());
    offset += step;
  }
  // Final prompt token produces exact last-token logits + first greedy token.
  const std::uint32_t last_prompt_tok = tokens.back();
  GpuSamplingOptions pf_sampling{};
  pf_sampling.absolute_position = tokens.size() - 1;
  pf_sampling.contract = loaded->contract;
  const auto last_res = loaded->forward->run(
      std::span<const std::uint32_t>(&last_prompt_tok, 1), pf_sampling,
      /*inspect_tensors=*/true, /*use_icb=*/true, /*profile_stages=*/false);
  const double prefill_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - pf_t0)
          .count();
  peak_phys = std::max(peak_phys, current_phys_footprint_bytes());

  const auto last_stats = compute_row_stats(last_res.logits);
  if (!o.logits_out.empty()) {
    write_float_binary(o.logits_out, last_res.logits);
  }

  std::vector<std::uint32_t> generated;
  double decode_s = 0.0;
  if (o.max_new_tokens > 0) {
    generated.reserve(o.max_new_tokens);
    std::uint32_t next_tok = last_res.token;
    generated.push_back(next_tok);
    const auto dec_t0 = std::chrono::steady_clock::now();
    for (std::uint32_t g = 1; g < o.max_new_tokens; ++g) {
      if (loaded->config.eos_token_id >= 0 &&
          next_tok == static_cast<std::uint32_t>(loaded->config.eos_token_id)) {
        break;
      }
      GpuSamplingOptions dec_sampling{};
      dec_sampling.absolute_position = tokens.size() + g - 1;
      dec_sampling.contract = loaded->contract;
      const auto step_res = loaded->forward->run(
          std::span<const std::uint32_t>(&next_tok, 1), dec_sampling,
          /*inspect_tensors=*/false, /*use_icb=*/true, false);
      next_tok = step_res.token;
      generated.push_back(next_tok);
      peak_phys = std::max(peak_phys, current_phys_footprint_bytes());
    }
    decode_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - dec_t0)
            .count();
  }

  const auto pf_tel = loaded->forward->prefill_telemetry();
  Json report = {
      {"mode", "prefill-logits"},
      {"model_dir", o.model_dir.string()},
      {"artifact", loaded->artifact_path.string()},
      {"metallib_dir", loaded->metallib_dir.string()},
      {"device_name", loaded->context->capabilities().device_name},
      {"kv_dtype", o.kv_dtype},
      {"sampling_contract", o.sampling_contract},
      {"max_context", effective_ctx},
      {"workspace_rows", ws_rows},
      {"prefill_switches",
       {
           {"prefill_linear_mode", o.prefill_linear_mode},
           {"prefill_attention_mode", o.prefill_attention_mode},
           {"prefill_sparse_minference", o.prefill_sparse_minference},
           {"prefill_cache_blend", o.prefill_cache_blend},
           {"prefill_ane_offload", o.prefill_ane_offload},
       }},
      {"prefill_telemetry",
       {
           {"prefill_w4a8_tokens", pf_tel.prefill_w4a8_tokens},
           {"prefill_sage2_tokens", pf_tel.prefill_sage2_tokens},
           {"prefill_minference_blocks_skipped",
            pf_tel.prefill_minference_blocks_skipped},
           {"prefill_minference_blocks_total",
            pf_tel.prefill_minference_blocks_total},
           {"prefill_ane_active", pf_tel.prefill_ane_active},
           {"prefill_ane_split_alpha", pf_tel.prefill_ane_split_alpha},
           {"prefill_ane_offload_tokens", pf_tel.prefill_ane_offload_tokens},
           {"prefill_ane_fallback_tokens", pf_tel.prefill_ane_fallback_tokens},
           {"prefill_cache_blend_hit_tokens",
            pf_tel.prefill_cache_blend_hit_tokens},
           {"prefill_cache_blend_recomputed_tokens",
            pf_tel.prefill_cache_blend_recomputed_tokens},
       }},
      {"prompt_tokens", tokens.size()},
      {"load_wall_s", loaded->load_wall_s},
      {"prefill_wall_s", prefill_s},
      {"prefill_tok_per_s",
       prefill_s > 0.0 ? (double(tokens.size()) / prefill_s) : 0.0},
      {"generated_count", generated.size()},
      {"generated_tokens", generated},
      {"decode_wall_s", decode_s},
      {"decode_tok_per_s",
       (generated.size() > 1 && decode_s > 0.0)
           ? (double(generated.size() - 1) / decode_s)
           : 0.0},
      {"peak_phys_footprint_bytes", peak_phys},
      {"peak_phys_footprint_gib", bytes_to_gib(peak_phys)},
      {"metal_allocated_bytes", loaded->context->current_allocated_bytes()},
      {"last_logits_l2", last_stats.l2_norm},
      {"last_token_argmax", last_stats.argmax_id},
      {"last_token_topk",
       top_k_logits_json(last_res.logits, last_stats.logsumexp, o.top_k_report)},
  };
  if (!o.logits_out.empty()) {
    report["logits_out"] = o.logits_out.string();
  }
  if (!o.compare_logits.empty()) {
    const auto ref = read_float_binary(o.compare_logits);
    report["logit_comparison"] = compare_logit_vectors(ref, last_res.logits);
  }
  emit_json(report, o.out_file);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto opts = parse_args(argc, argv);
    if (opts.show_help) {
      std::cout << format_help();
      return 0;
    }
    if (opts.mode == "tokenize") {
      return run_tokenize(opts);
    }
    if (opts.mode == "ppl") {
      return run_ppl(opts);
    }
    if (opts.mode == "prefill-logits") {
      return run_prefill_logits(opts);
    }
    if (opts.mode == "compare-logits") {
      return run_compare_logits(opts);
    }
    throw std::invalid_argument("unsupported --mode: " + opts.mode);
  } catch (const std::exception &err) {
    std::cerr << "omlx_m5_prefill_matrix_bench error: " << err.what() << '\n';
    return 1;
  }
}
