#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/memory_plan.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/safetensors.h"

namespace {

using Json = nlohmann::json;
using omlx::m5::LayerType;
using omlx::m5::QwenModelConfig;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

struct TensorFixture {
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::vector<std::byte> payload;
};

std::vector<std::byte> pack_codes(const std::vector<std::uint32_t> &codes,
                                  std::uint32_t bits) {
  std::vector<std::byte> packed((codes.size() * bits + 7) / 8);
  for (std::size_t index = 0; index < codes.size(); ++index) {
    const auto bit_offset = index * bits;
    auto value = codes[index];
    for (std::uint32_t bit = 0; bit < bits; ++bit) {
      if ((value & (1U << bit)) != 0U) {
        const auto absolute = bit_offset + bit;
        packed[absolute / 8] |= static_cast<std::byte>(1U << (absolute % 8));
      }
    }
  }
  return packed;
}

std::vector<std::byte> bf16_values(std::initializer_list<float> values) {
  std::vector<std::byte> bytes;
  bytes.reserve(values.size() * sizeof(std::uint16_t));
  for (const auto value : values) {
    const auto encoded = omlx::m5::float_to_bf16(value);
    bytes.push_back(static_cast<std::byte>(encoded & 0xFFU));
    bytes.push_back(static_cast<std::byte>(encoded >> 8));
  }
  return bytes;
}

std::uint64_t write_shard(const std::filesystem::path &path,
                          const std::map<std::string, TensorFixture> &tensors) {
  Json header = Json::object();
  std::uint64_t offset = 0;
  for (const auto &[name, tensor] : tensors) {
    header[name] = {
        {"dtype", tensor.dtype},
        {"shape", tensor.shape},
        {"data_offsets", {offset, offset + tensor.payload.size()}},
    };
    offset += tensor.payload.size();
  }
  auto serialized = header.dump();
  while (serialized.size() % 8 != 0) {
    serialized.push_back(' ');
  }
  const auto header_size = static_cast<std::uint64_t>(serialized.size());
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(&header_size),
               sizeof(header_size));
  output.write(serialized.data(),
               static_cast<std::streamsize>(serialized.size()));
  for (const auto &[_, tensor] : tensors) {
    output.write(reinterpret_cast<const char *>(tensor.payload.data()),
                 static_cast<std::streamsize>(tensor.payload.size()));
  }
  output.close();
  return std::filesystem::file_size(path);
}

Json make_config() {
  return {
      {"model_type", "qwen3_5"},
      {"text_config",
       {
           {"model_type", "qwen3_5_text"},
           {"dtype", "bfloat16"},
           {"hidden_size", 64},
           {"num_hidden_layers", 2},
           {"intermediate_size", 128},
           {"vocab_size", 256},
           {"num_attention_heads", 4},
           {"num_key_value_heads", 2},
           {"head_dim", 16},
           {"linear_num_key_heads", 2},
           {"linear_num_value_heads", 4},
           {"linear_key_head_dim", 8},
           {"linear_value_head_dim", 8},
           {"linear_conv_kernel_dim", 4},
           {"max_position_embeddings", 1024},
           {"layer_types", {"linear_attention", "full_attention"}},
       }},
      {"quantization",
       {
           {"bits", 4},
           {"group_size", 64},
           {"mode", "affine"},
           {"q5", {{"bits", 5}, {"group_size", 64}, {"mode", "affine"}}},
       }},
  };
}

std::filesystem::path make_fixture() {
  const auto root =
      std::filesystem::temp_directory_path() / "omlx-m5-d1-host-fixture";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  std::vector<std::uint32_t> q4_codes(64);
  std::vector<std::uint32_t> q5_codes(64);
  for (std::size_t i = 0; i < 64; ++i) {
    q4_codes[i] = static_cast<std::uint32_t>(i % 16);
    q5_codes[i] = static_cast<std::uint32_t>((i * 3) % 32);
  }
  const std::map<std::string, TensorFixture> tensors = {
      {"q4.biases", {"BF16", {1, 1}, bf16_values({-1.0F})}},
      {"q4.scales", {"BF16", {1, 1}, bf16_values({0.5F})}},
      {"q4.weight", {"U32", {1, 8}, pack_codes(q4_codes, 4)}},
      {"q5.biases", {"BF16", {1, 1}, bf16_values({0.5F})}},
      {"q5.scales", {"BF16", {1, 1}, bf16_values({0.25F})}},
      {"q5.weight", {"U32", {1, 10}, pack_codes(q5_codes, 5)}},
  };
  const std::string shard_name = "model-00001-of-00001.safetensors";
  const auto shard_size = write_shard(root / shard_name, tensors);
  Json weight_map = Json::object();
  for (const auto &[name, _] : tensors) {
    weight_map[name] = shard_name;
  }
  std::ofstream(root / "model.safetensors.index.json")
      << Json{
             {"metadata", {{"total_size", shard_size}}},
             {"weight_map", weight_map},
         }
             .dump(2);
  std::ofstream(root / "config.json") << make_config().dump(2);
  std::ofstream(root / "REVISION") << "fixture-revision\n";
  return root;
}

void test_safetensors_and_quantization() {
  const auto root = make_fixture();
  const auto config = omlx::m5::load_qwen_model_config(root / "config.json");
  expect(config.num_hidden_layers == 2, "config parser lost layer count");
  expect(config.full_attention_layers() == 1, "full attention count mismatch");
  expect(config.quantization_for("q5").bits == 5, "q5 override was ignored");

  const auto model = omlx::m5::SafeTensorModel::open(root, "fixture-revision");
  expect(model.shard_count() == 1, "fixture shard count mismatch");
  expect(model.tensor_count() == 6, "fixture tensor count mismatch");
  expect(omlx::m5::quantized_module_names(model) ==
             std::vector<std::string>({"q4", "q5"}),
         "quantized module discovery mismatch");

  const auto q4 = omlx::m5::bind_quantized_module(model, config, "q4");
  const auto q4_values = omlx::m5::dequantize_row(q4, 0, 0, 16);
  for (std::size_t i = 0; i < q4_values.size(); ++i) {
    expect(q4_values[i] == static_cast<float>(i % 16) * 0.5F - 1.0F,
           "q4 independent decoder mismatch");
  }

  const auto q5 = omlx::m5::bind_quantized_module(model, config, "q5");
  const auto q5_values = omlx::m5::dequantize_row(q5, 0, 0, 32);
  const auto q5_values_f64 = omlx::m5::dequantize_row_f64(q5, 0, 0, 32);
  for (std::size_t i = 0; i < q5_values.size(); ++i) {
    const auto code = static_cast<float>((i * 3) % 32);
    expect(q5_values[i] == code * 0.25F + 0.5F,
           "q5 independent decoder mismatch");
    expect(q5_values_f64[i] == static_cast<double>(code) * 0.25 + 0.5,
           "q5 FP64 oracle mismatch");
  }

  bool revision_failed = false;
  try {
    static_cast<void>(omlx::m5::SafeTensorModel::open(root, "wrong-revision"));
  } catch (const std::runtime_error &) {
    revision_failed = true;
  }
  expect(revision_failed, "revision mismatch was accepted");
  std::filesystem::remove_all(root);
}

void test_missing_shard() {
  const auto root =
      std::filesystem::temp_directory_path() / "omlx-m5-d1-missing-shard";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  std::ofstream(root / "REVISION") << "fixture\n";
  std::ofstream(root / "model.safetensors.index.json")
      << Json{
             {"metadata", {{"total_size", 1}}},
             {"weight_map", {{"x", "missing.safetensors"}}},
         }
             .dump();
  bool failed = false;
  try {
    static_cast<void>(omlx::m5::SafeTensorModel::open(root));
  } catch (const std::runtime_error &) {
    failed = true;
  }
  expect(failed, "missing shard was accepted");
  std::filesystem::remove_all(root);
}

void expect_open_failure(const std::filesystem::path &root,
                         const std::string &message) {
  bool failed = false;
  try {
    static_cast<void>(omlx::m5::SafeTensorModel::open(root));
  } catch (const std::runtime_error &) {
    failed = true;
  }
  expect(failed, message);
}

void test_corrupt_and_unsafe_manifests() {
  const auto corrupt_root =
      std::filesystem::temp_directory_path() / "omlx-m5-d1-corrupt-header";
  std::filesystem::remove_all(corrupt_root);
  std::filesystem::create_directories(corrupt_root);
  const std::string shard_name = "model.safetensors";
  {
    std::ofstream shard(corrupt_root / shard_name, std::ios::binary);
    const std::uint64_t impossible_header = 1024;
    shard.write(reinterpret_cast<const char *>(&impossible_header),
                sizeof(impossible_header));
  }
  std::ofstream(corrupt_root / "REVISION") << "fixture\n";
  std::ofstream(corrupt_root / "model.safetensors.index.json")
      << Json{
             {"metadata",
              {{"total_size",
                std::filesystem::file_size(corrupt_root / shard_name)}}},
             {"weight_map", {{"x", shard_name}}},
         }
             .dump();
  expect_open_failure(corrupt_root, "corrupt safetensors header was accepted");
  std::filesystem::remove_all(corrupt_root);

  const auto unsafe_root =
      std::filesystem::temp_directory_path() / "omlx-m5-d1-unsafe-index";
  std::filesystem::remove_all(unsafe_root);
  std::filesystem::create_directories(unsafe_root);
  std::ofstream(unsafe_root / "REVISION") << "fixture\n";
  std::ofstream(unsafe_root / "model.safetensors.index.json")
      << Json{
             {"metadata", {{"total_size", 1}}},
             {"weight_map", {{"x", "../outside.safetensors"}}},
         }
             .dump();
  expect_open_failure(unsafe_root, "unsafe shard traversal was accepted");
  std::filesystem::remove_all(unsafe_root);

  const auto overlap_root =
      std::filesystem::temp_directory_path() / "omlx-m5-d1-overlap";
  std::filesystem::remove_all(overlap_root);
  std::filesystem::create_directories(overlap_root);
  Json overlap_header = {
      {"a", {{"dtype", "U8"}, {"shape", {4}}, {"data_offsets", {0, 4}}}},
      {"b", {{"dtype", "U8"}, {"shape", {4}}, {"data_offsets", {2, 6}}}},
  };
  auto overlap_serialized = overlap_header.dump();
  while (overlap_serialized.size() % 8 != 0) {
    overlap_serialized.push_back(' ');
  }
  {
    std::ofstream shard(overlap_root / shard_name, std::ios::binary);
    const auto header_size =
        static_cast<std::uint64_t>(overlap_serialized.size());
    shard.write(reinterpret_cast<const char *>(&header_size),
                sizeof(header_size));
    shard.write(overlap_serialized.data(),
                static_cast<std::streamsize>(overlap_serialized.size()));
    const std::uint8_t payload[6] = {};
    shard.write(reinterpret_cast<const char *>(payload), sizeof(payload));
  }
  const auto overlap_size =
      std::filesystem::file_size(overlap_root / shard_name);
  std::ofstream(overlap_root / "REVISION") << "fixture\n";
  std::ofstream(overlap_root / "model.safetensors.index.json")
      << Json{
             {"metadata", {{"total_size", overlap_size}}},
             {"weight_map", {{"a", shard_name}, {"b", shard_name}}},
         }
             .dump();
  expect_open_failure(overlap_root, "overlapping tensor ranges were accepted");
  std::filesystem::remove_all(overlap_root);

  const auto overflow_root =
      std::filesystem::temp_directory_path() / "omlx-m5-d1-overflow-shape";
  std::filesystem::remove_all(overflow_root);
  std::filesystem::create_directories(overflow_root);
  const std::map<std::string, TensorFixture> overflow_tensor = {
      {"x",
       {"U8",
        {std::numeric_limits<std::uint64_t>::max(), 2},
        std::vector<std::byte>{}}},
  };
  const auto overflow_size =
      write_shard(overflow_root / shard_name, overflow_tensor);
  std::ofstream(overflow_root / "REVISION") << "fixture\n";
  std::ofstream(overflow_root / "model.safetensors.index.json")
      << Json{
             {"metadata", {{"total_size", overflow_size}}},
             {"weight_map", {{"x", shard_name}}},
         }
             .dump();
  expect_open_failure(overflow_root, "overflowing tensor shape was accepted");
  std::filesystem::remove_all(overflow_root);
}

void test_target_memory_plan() {
  QwenModelConfig config;
  config.num_key_value_heads = 4;
  config.head_dim = 256;
  config.linear_num_key_heads = 16;
  config.linear_num_value_heads = 48;
  config.linear_key_head_dim = 128;
  config.linear_value_head_dim = 128;
  config.linear_conv_kernel_dim = 4;
  config.layer_types.insert(config.layer_types.end(), 48,
                            LayerType::linear_attention);
  config.layer_types.insert(config.layer_types.end(), 16,
                            LayerType::full_attention);
  const auto plan = omlx::m5::plan_sequence_memory(config, 32768);
  expect(plan.kv_bytes_per_token == 65536, "KV bytes/token mismatch");
  expect(plan.kv_page_count == 512, "KV page count mismatch");
  expect(plan.kv_total_bytes == 2147483648ULL, "32K KV capacity mismatch");
  expect(plan.gdn_state_bytes == 150994944, "GDN state must be 144 MiB");
  expect(plan.gdn_conv_history_bytes == 2949120,
         "GDN convolution history mismatch");
}

void test_model_config_validation() {
  const auto root =
      std::filesystem::temp_directory_path() / "omlx-m5-d1-config-validation";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const auto write_cfg = [&](const std::string &name, const Json &cfg) {
    const auto path = root / name;
    std::ofstream(path) << cfg.dump(2);
    return path;
  };
  const auto expect_cfg_runtime_error = [&](const std::string &name,
                                            const Json &cfg,
                                            const std::string &message) {
    const auto path = write_cfg(name, cfg);
    bool caught_runtime = false;
    try {
      static_cast<void>(omlx::m5::load_qwen_model_config(path));
    } catch (const Json::exception &) {
      throw std::runtime_error(message + " (leaked nlohmann::json::exception)");
    } catch (const std::runtime_error &) {
      caught_runtime = true;
    }
    expect(caught_runtime, message);
  };

  // Array eos_token_id, null eos_token_id, empty array eos_token_id, and
  // null intermediate_size falling back to moe_intermediate_size.
  {
    auto cfg = make_config();
    cfg["text_config"]["eos_token_id"] = Json::array({151645, 151643});
    cfg["text_config"]["intermediate_size"] = nullptr;
    cfg["text_config"]["moe_intermediate_size"] = 192;
    const auto parsed =
        omlx::m5::load_qwen_model_config(write_cfg("array_eos.json", cfg));
    expect(parsed.eos_token_id == 151645 && parsed.intermediate_size == 192,
           "failed to parse array eos_token_id and null intermediate_size");
  }
  {
    auto cfg = make_config();
    cfg["text_config"]["eos_token_id"] = nullptr;
    const auto parsed =
        omlx::m5::load_qwen_model_config(write_cfg("null_eos.json", cfg));
    expect(parsed.eos_token_id == -1, "null eos_token_id did not map to -1");
  }
  {
    auto cfg = make_config();
    cfg["text_config"]["eos_token_id"] = Json::array();
    const auto parsed =
        omlx::m5::load_qwen_model_config(write_cfg("empty_eos.json", cfg));
    expect(parsed.eos_token_id == -1, "empty eos_token_id did not map to -1");
  }

  // Malformed types and invalid geometry must throw std::runtime_error.
  {
    auto cfg = make_config();
    cfg["text_config"]["eos_token_id"] = "bad";
    expect_cfg_runtime_error("bad_eos.json", cfg,
                             "string eos_token_id was accepted");
  }
  {
    auto cfg = make_config();
    cfg["text_config"]["eos_token_id"] = Json::array({151645, "bad"});
    expect_cfg_runtime_error("bad_eos_elem.json", cfg,
                             "non-integer eos_token_id array was accepted");
  }
  {
    auto cfg = make_config();
    cfg["text_config"]["intermediate_size"] = "bad";
    expect_cfg_runtime_error("bad_inter.json", cfg,
                             "string intermediate_size was accepted");
  }
  {
    auto cfg = make_config();
    cfg["text_config"]["num_hidden_layers"] = -1;
    expect_cfg_runtime_error("neg_layers.json", cfg,
                             "negative num_hidden_layers was accepted");
  }
  {
    auto cfg = make_config();
    cfg["text_config"]["hidden_size"] = 0;
    expect_cfg_runtime_error("zero_hidden.json", cfg,
                             "zero hidden_size was accepted");
  }
  {
    auto cfg = make_config();
    cfg["text_config"]["vocab_size"] = 0;
    expect_cfg_runtime_error("zero_vocab.json", cfg,
                             "zero vocab_size was accepted");
  }
  {
    auto cfg = make_config();
    cfg["text_config"]["num_attention_heads"] = 3;
    cfg["text_config"]["num_key_value_heads"] = 2;
    expect_cfg_runtime_error("bad_attn_div.json", cfg,
                             "indivisible attention head count was accepted");
  }
  {
    auto cfg = make_config();
    cfg["text_config"]["linear_num_value_heads"] = 3;
    cfg["text_config"]["linear_num_key_heads"] = 2;
    expect_cfg_runtime_error("bad_lin_div.json", cfg,
                             "indivisible linear head count was accepted");
  }
  {
    auto cfg = make_config();
    cfg["text_config"]["num_experts"] = 4;
    cfg["text_config"]["num_experts_per_tok"] = 5;
    cfg["text_config"]["shared_expert_intermediate_size"] = 64;
    expect_cfg_runtime_error("bad_moe_topk.json", cfg,
                             "num_experts_per_tok > num_experts was accepted");
  }
  std::filesystem::remove_all(root);
}

} // namespace

int main() {
  try {
    test_safetensors_and_quantization();
    test_missing_shard();
    test_corrupt_and_unsafe_manifests();
    test_target_memory_plan();
    test_model_config_validation();
  } catch (const std::exception &error) {
    std::cerr << "D1 host test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "D1 host tests passed\n";
  return 0;
}
