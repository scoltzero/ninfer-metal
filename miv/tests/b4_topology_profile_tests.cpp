#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "omlx_m5/artifact.h"
#include "omlx_m5/dispatch.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/qwen_program.h"
#include "omlx_m5/topology_profile.h"

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <class Function>
void expect_throws(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error(message);
}

omlx::m5::QwenTopologyDescriptor topology(bool moe) {
  using namespace omlx::m5;
  QwenTopologyDescriptor result{
      .architecture = moe ? QwenArchitecture::moe : QwenArchitecture::dense,
      .quant_profile =
          moe ? QwenQuantProfile::mxfp4 : QwenQuantProfile::oq_affine,
      .layer_count = 2,
      .top_k = moe ? 8U : 0U,
      .max_context = 4096,
      .kv_page_tokens = 64,
      .dispatches =
          {
              {.index = 0,
               .role = "embedding",
               .format = "direct_bf16_v1",
               .layout = "contiguous_le_v1",
               .shape = {128, 64}},
              {.index = 1,
               .layer = 0,
               .role = "projection",
               .format =
                   moe ? "mxfp4_e2m1_g32_e8m0_v1" : "oq_affine_q4_g64_bf16_v1",
               .layout = moe ? "mx_rowsplit_k128_v1" : "oq_rowsplit_k128_v1",
               .shape = {64, 64}},
              {.index = 2,
               .layer = 1,
               .role = "projection",
               .stage_dispatch = 0,
               .format =
                   moe ? "mxfp4_e2m1_g32_e8m0_v1" : "oq_affine_q5_g64_bf16_v1",
               .layout = moe ? "mx_rowsplit_k128_v1" : "oq_rowsplit_k128_v1",
               .shape = {64, 64}},
              {.index = 3,
               .layer = 1,
               .role = "projection",
               .stage_dispatch = 1,
               .format =
                   moe ? "mxfp4_e2m1_g32_e8m0_v1" : "oq_affine_q5_g64_bf16_v1",
               .layout = moe ? "mx_rowsplit_k128_v1" : "oq_rowsplit_k128_v1",
               .shape = {64, 64}},
              {.index = 4, .kind = LayerStageKind::reduce, .role = "sampling"},
          },
      .ranges =
          {
              {.key = "full", .location = 0, .length = 5},
              {.key = "embedding", .location = 0, .length = 1},
              {.key = "layer.0", .location = 1, .length = 1},
              {.key = "layer.1", .location = 2, .length = 2},
              {.key = "vocabulary", .location = 4, .length = 1},
          },
  };
  return result;
}

const omlx::m5::MetalParameterFieldDescriptor *
find_field(const omlx::m5::TopologyProfile &profile, std::string_view key) {
  const auto found =
      std::ranges::find_if(profile.parameters.fields,
                           [&](const auto &field) { return field.key == key; });
  return found == profile.parameters.fields.end() ? nullptr : &*found;
}

void test_finite_profile_compilation() {
  using namespace omlx::m5;
  const auto first = compile_topology_profiles(topology(false));
  const auto second = compile_topology_profiles(topology(false));
  expect(first.stats().finite && first.stats().profile_count == 7 &&
             first.stats().decode_profile_count == 1 &&
             first.stats().prefill_profile_count == 6 &&
             first.stats().maximum_token_capacity == 1024 &&
             first.stats().dispatch_records_per_profile == 5,
         "profile catalog is not finite or complete");
  expect(first.topology().signature == second.topology().signature &&
             first.profiles().size() == second.profiles().size(),
         "topology compilation is not deterministic");
  const std::vector<std::uint32_t> expected = {1, 16, 32, 64, 128, 512, 1024};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto &left = first.profile(index);
    const auto &right = second.profile(index);
    expect(left.key.token_capacity == expected[index] &&
               left.name == right.name &&
               left.structural_signature == right.structural_signature &&
               left.parameters.byte_size == right.parameters.byte_size &&
               left.dispatches == right.dispatches &&
               left.ranges.size() == right.ranges.size(),
           "compiled profile changed between identical inputs");
    expect(find_field(left, "token_ids") != nullptr &&
               find_field(left, "positions") != nullptr &&
               find_field(left, "page_table") != nullptr &&
               find_field(left, "valid_mask") != nullptr &&
               find_field(left, "sampling") != nullptr &&
               find_field(left, "request_id") == nullptr,
           "dynamic/request fields violate the profile key contract");
  }
}

void test_selection_and_fallback() {
  using namespace omlx::m5;
  const auto catalog = compile_topology_profiles(topology(false));
  const auto request = [&](TopologyProfileMode mode, std::uint32_t batch,
                           std::uint32_t tokens) {
    return TopologyProfileRequest{
        .mode = mode,
        .architecture = QwenArchitecture::dense,
        .quant_profile = QwenQuantProfile::oq_affine,
        .batch_size = batch,
        .tokens = tokens,
        .topology_signature = catalog.topology().signature,
    };
  };
  expect(catalog.select(request(TopologyProfileMode::decode, 1, 1)).matched(),
         "decode profile did not match");
  const auto tail = catalog.select(request(TopologyProfileMode::prefill, 1, 5));
  expect(tail.matched() &&
             catalog.profile(tail.profile_index).key.token_capacity == 16,
         "prefill tail did not select the smallest capacity");
  const auto bulk =
      catalog.select(request(TopologyProfileMode::prefill, 1, 129));
  expect(bulk.matched() &&
             catalog.profile(bulk.profile_index).key.token_capacity == 512,
         "prefill bulk capacity selection failed");
  expect(
      catalog.select(request(TopologyProfileMode::decode, 1, 3)).reason ==
              TopologyFallbackReason::decode_requires_one_token &&
          catalog.select(request(TopologyProfileMode::prefill, 2, 16)).reason ==
              TopologyFallbackReason::unsupported_batch &&
          catalog.select(request(TopologyProfileMode::prefill, 1, 1025))
                  .reason == TopologyFallbackReason::unsupported_token_capacity,
      "unsupported shapes did not fail closed to execute_batch");
  auto wrong = request(TopologyProfileMode::decode, 1, 1);
  wrong.quant_profile = QwenQuantProfile::mxfp4;
  expect(catalog.select(wrong).reason ==
             TopologyFallbackReason::quant_profile_mismatch,
         "quantization profiles were mixed");
  wrong = request(TopologyProfileMode::decode, 1, 1);
  wrong.topology_signature = "foreign";
  expect(catalog.select(wrong).reason ==
             TopologyFallbackReason::topology_mismatch,
         "foreign topology was accepted");
}

void test_moe_and_bounded_input() {
  using namespace omlx::m5;
  const auto catalog = compile_topology_profiles(topology(true));
  expect(find_field(catalog.profile(0), "route_indices") != nullptr &&
             find_field(catalog.profile(0), "route_scores") != nullptr,
         "MoE profile omitted dynamic route buffers");
  auto invalid = topology(false);
  expect_throws(
      [&] {
        static_cast<void>(compile_topology_profiles(
            invalid, {.prefill_capacities = {16, 32, 64, 128, 256}}));
      },
      "unsupported B2 capacity was accepted");
  expect_throws(
      [&] {
        static_cast<void>(
            compile_topology_profiles(topology(false), {.batch_size = 2}));
      },
      "undeclared concurrent batch profile was accepted");
  auto invalid_range = topology(false);
  invalid_range.ranges.front().length = 99;
  expect_throws(
      [&] { static_cast<void>(compile_topology_profiles(invalid_range)); },
      "out-of-bounds full range was accepted");
  auto short_context = topology(false);
  short_context.max_context = 128;
  expect_throws(
      [&] { static_cast<void>(compile_topology_profiles(short_context)); },
      "profile capacity larger than Program context was accepted");
}

struct SyntheticObjectSpec {
  std::string name;
  std::string format;
  std::string layout;
  std::vector<std::uint64_t> shape;
  std::uint64_t padded_k = 0;
  std::string binding_use; // non-empty => emit ArtifactBinding
  std::string activation_policy = "A16Only";
};

std::string hex_crc32(std::uint32_t crc) {
  std::ostringstream out;
  out << std::hex << std::nouppercase << std::setfill('0') << std::setw(8)
      << crc;
  return out.str();
}

void append_u32(std::vector<std::byte> &out, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<std::byte>((v >> (i * 8)) & 0xFFU));
  }
}

void append_u64(std::vector<std::byte> &out, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::byte>((v >> (i * 8)) & 0xFFU));
  }
}

void write_synthetic_artifact(const std::filesystem::path &path,
                              const std::string &source_profile,
                              const std::vector<SyntheticObjectSpec> &specs) {
  struct PlaneSpec {
    std::string kind;
    std::uint64_t offset;
    std::uint64_t bytes;
    std::string crc32;
    std::string sha256;
  };
  struct BuiltObject {
    std::uint32_t id;
    SyntheticObjectSpec spec;
    std::uint64_t object_offset;
    std::uint64_t object_bytes;
    std::string object_sha256;
    std::vector<PlaneSpec> planes;
  };

  std::vector<std::byte> payload;
  const auto align_payload = [&](std::size_t alignment) {
    while (payload.size() % alignment != 0) {
      payload.push_back(std::byte{0});
    }
  };
  const auto append_plane = [&](std::vector<PlaneSpec> &planes,
                                const std::string &kind, std::uint64_t bytes) {
    align_payload(256);
    const std::uint64_t off = payload.size();
    payload.resize(payload.size() + static_cast<std::size_t>(bytes),
                   std::byte{0x11});
    const auto span = std::span<const std::byte>(payload).subspan(
        static_cast<std::size_t>(off), static_cast<std::size_t>(bytes));
    planes.push_back({.kind = kind,
                      .offset = off,
                      .bytes = bytes,
                      .crc32 = hex_crc32(omlx::m5::artifact_crc32(span)),
                      .sha256 = omlx::m5::artifact_sha256_hex(span)});
  };

  std::vector<BuiltObject> built;
  built.reserve(specs.size());
  for (std::uint32_t id = 0; id < specs.size(); ++id) {
    const auto &s = specs[id];
    align_payload(256);
    const std::uint64_t obj_off = payload.size();
    std::vector<PlaneSpec> planes;
    if (s.format == "direct_bf16_v1") {
      std::uint64_t elems = 1;
      for (const auto d : s.shape) {
        elems *= d;
      }
      append_plane(planes, "data", elems * 2ULL);
    } else if (s.format == "mxfp4_e2m1_g32_e8m0_v1" &&
               s.layout == "mx_nax_tiled_bk128_v1") {
      const std::uint64_t rows = s.shape[0];
      const std::uint64_t k = s.shape[1];
      const std::uint64_t bn = rows >= 65536ULL ? 32ULL : 16ULL;
      const std::uint64_t pn = ((rows + bn - 1ULL) / bn) * bn;
      append_plane(planes, "nax_data", pn * (k / 2ULL));
      append_plane(planes, "nax_scale", pn * (k / 32ULL));
    } else if (s.format == "oq_affine_q3_g64_bf16_v1" &&
               s.layout == "oq_nax_tiled_g64_v1") {
      const std::uint64_t rows = s.shape[0];
      const std::uint64_t k = s.shape[1];
      const std::uint64_t bn = rows >= 65536ULL ? 32ULL : 16ULL;
      const std::uint64_t pn = ((rows + bn - 1ULL) / bn) * bn;
      append_plane(planes, "nax_codes", pn * (k / 4ULL));
      append_plane(planes, "nax_high1", pn * (k / 8ULL));
      append_plane(planes, "nax_sb", pn * ((k / 64ULL) * 4ULL));
    } else if (s.format == "oq_affine_q4_g64_bf16_v1" &&
               s.layout == "oq_rowsplit_k128_v1") {
      const std::uint64_t rows = s.shape[0];
      const std::uint64_t pk = s.padded_k;
      append_plane(planes, "base_code", rows * (pk / 2ULL));
      append_plane(planes, "scale", rows * (pk / 64ULL) * 2ULL);
      append_plane(planes, "bias", rows * (pk / 64ULL) * 2ULL);
    } else {
      throw std::invalid_argument("unsupported synthetic object spec");
    }
    const std::uint64_t obj_bytes = payload.size() - obj_off;
    const auto obj_span = std::span<const std::byte>(payload).subspan(
        static_cast<std::size_t>(obj_off),
        static_cast<std::size_t>(obj_bytes));
    built.push_back(
        {.id = id,
         .spec = s,
         .object_offset = obj_off,
         .object_bytes = obj_bytes,
         .object_sha256 = omlx::m5::artifact_sha256_hex(obj_span),
         .planes = std::move(planes)});
  }

  std::ostringstream json;
  json << "{\"bindings\":[";
  bool first_binding = true;
  for (const auto &obj : built) {
    if (obj.spec.binding_use.empty()) {
      continue;
    }
    if (!first_binding) {
      json << ",";
    }
    first_binding = false;
    json << "{\"activation_policy\":\"" << obj.spec.activation_policy
         << "\",\"format\":\"" << obj.spec.format << "\",\"in_features\":"
         << obj.spec.shape.back() << ",\"leading_shape\":[],\"name\":\""
         << obj.spec.name << "\",\"object_id\":" << obj.id
         << ",\"out_features\":" << obj.spec.shape[obj.spec.shape.size() - 2]
         << ",\"use\":\"" << obj.spec.binding_use << "\"}";
  }
  json << "],\"container_version\":1,\"model\":{\"source_profile\":\""
       << source_profile
       << "\",\"source_repo\":\"test/synth\",\"source_revision\":\"r1\"},"
       << "\"objects\":[";
  for (std::size_t i = 0; i < built.size(); ++i) {
    const auto &obj = built[i];
    if (i > 0) {
      json << ",";
    }
    json << "{\"format\":\"" << obj.spec.format << "\",\"id\":" << obj.id
         << ",\"layout\":\"" << obj.spec.layout << "\",\"name\":\""
         << obj.spec.name << "\",\"object_bytes\":" << obj.object_bytes
         << ",\"object_offset\":" << obj.object_offset
         << ",\"object_sha256\":\"" << obj.object_sha256 << "\"";
    if (obj.spec.padded_k > 0) {
      json << ",\"padded_k\":" << obj.spec.padded_k;
    }
    json << ",\"planes\":[";
    for (std::size_t j = 0; j < obj.planes.size(); ++j) {
      const auto &pl = obj.planes[j];
      if (j > 0) {
        json << ",";
      }
      json << "{\"alignment\":256,\"bytes\":" << pl.bytes << ",\"crc32\":\""
           << pl.crc32 << "\",\"kind\":\"" << pl.kind
           << "\",\"offset\":" << pl.offset << ",\"sha256\":\"" << pl.sha256
           << "\"}";
    }
    json << "],\"shape\":[";
    for (std::size_t j = 0; j < obj.spec.shape.size(); ++j) {
      if (j > 0) {
        json << ",";
      }
      json << obj.spec.shape[j];
    }
    json << "],\"source_tensors\":[\"" << obj.spec.name << "\"]}";
  }
  json << "],\"payload_bytes\":" << payload.size()
       << ",\"schema\":\"omlx-m5-artifact-v1\",\"views\":[]}";

  const auto manifest_str = json.str();
  const auto manifest_bytes =
      std::as_bytes(std::span(manifest_str.data(), manifest_str.size()));
  constexpr std::uint64_t kAlign = 16384ULL;
  const std::uint64_t raw_end = 64ULL + manifest_bytes.size();
  const std::uint64_t payload_offset = (raw_end + kAlign - 1ULL) & ~(kAlign - 1ULL);
  const std::uint32_t manifest_crc = omlx::m5::artifact_crc32(manifest_bytes);

  std::vector<std::byte> file;
  constexpr char magic[8] = {'O', 'M', 'L', 'X', 'M', '5', 'A', '1'};
  for (char c : magic) {
    file.push_back(static_cast<std::byte>(c));
  }
  append_u32(file, 1U);
  append_u32(file, 64U);
  append_u64(file, 64ULL);
  append_u64(file, manifest_bytes.size());
  append_u64(file, payload_offset);
  append_u64(file, payload.size());
  append_u32(file, static_cast<std::uint32_t>(built.size()));
  append_u32(file, manifest_crc);
  const std::uint32_t header_crc =
      omlx::m5::artifact_crc32(std::span<const std::byte>(file.data(), 56));
  append_u32(file, header_crc);
  append_u32(file, 0U);
  file.insert(file.end(), manifest_bytes.begin(), manifest_bytes.end());
  file.resize(static_cast<std::size_t>(payload_offset), std::byte{0});
  file.insert(file.end(), payload.begin(), payload.end());

  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(file.data()),
            static_cast<std::streamsize>(file.size()));
}

omlx::m5::QwenModelConfig make_minimal_linear_config() {
  omlx::m5::QwenModelConfig cfg;
  cfg.hidden_size = 512;
  cfg.num_hidden_layers = 1;
  cfg.intermediate_size = 512;
  cfg.vocab_size = 64;
  cfg.num_attention_heads = 4;
  cfg.num_key_value_heads = 2;
  cfg.head_dim = 128;
  cfg.linear_num_key_heads = 4;
  cfg.linear_num_value_heads = 4;
  cfg.linear_key_head_dim = 128;
  cfg.linear_value_head_dim = 128;
  cfg.linear_conv_kernel_dim = 4;
  cfg.max_position_embeddings = 4096;
  cfg.layer_types = {omlx::m5::LayerType::linear_attention};
  return cfg;
}

std::vector<SyntheticObjectSpec>
make_text_layer0_specs(const std::string &format, const std::string &layout,
                       const std::string &policy) {
  const std::string p = "language_model.model.layers.0";
  std::vector<SyntheticObjectSpec> specs;
  const auto add_q = [&](std::string name, std::uint64_t rows,
                         std::string use) {
    specs.push_back({.name = std::move(name),
                     .format = format,
                     .layout = layout,
                     .shape = {rows, 512},
                     .padded_k = 512,
                     .binding_use = std::move(use),
                     .activation_policy = policy});
  };
  const auto add_bf16 = [&](std::string name, std::vector<std::uint64_t> sh) {
    specs.push_back({.name = std::move(name),
                     .format = "direct_bf16_v1",
                     .layout = "contiguous_le_v1",
                     .shape = std::move(sh)});
  };
  add_bf16(p + ".input_layernorm.weight", {512});
  add_q(p + ".linear_attn.in_proj_qkv", 512, "linear");
  add_q(p + ".linear_attn.in_proj_z", 512, "linear");
  add_q(p + ".linear_attn.in_proj_a", 512, "linear");
  add_q(p + ".linear_attn.in_proj_b", 512, "linear");
  add_bf16(p + ".linear_attn.conv1d.weight", {512, 1, 4});
  add_bf16(p + ".linear_attn.A_log", {512});
  add_bf16(p + ".linear_attn.dt_bias", {512});
  add_bf16(p + ".linear_attn.norm.weight", {512});
  add_q(p + ".linear_attn.out_proj", 512, "linear");
  add_bf16(p + ".post_attention_layernorm.weight", {512});
  add_q(p + ".mlp.gate_proj", 512, "linear");
  add_q(p + ".mlp.up_proj", 512, "linear");
  add_q(p + ".mlp.down_proj", 512, "linear");
  add_q("language_model.model.embed_tokens", 64, "embedding");
  add_bf16("language_model.model.norm.weight", {512});
  add_q("language_model.lm_head", 64, "lm_head");
  return specs;
}

void test_bundled_companions_and_q3_pretiled() {
  using namespace omlx::m5;
  const auto dir = std::filesystem::temp_directory_path() /
                   ("omlx-m5-b4-topo-reg-" + std::to_string(::getpid()));
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto cfg = make_minimal_linear_config();

  // 1. MIVE-11 + MIVE-8: MXFP4 NAX pre-tiled text bundled with root-level
  //    dflash2.layers.0.*, mtp.layers.0.* (Q4), mtp.embed_tokens, and
  //    vision_model.norm.weight appearing BEFORE text objects.
  {
    std::vector<SyntheticObjectSpec> specs;
    specs.push_back({.name = "dflash2.layers.0.self_attn.q_proj",
                     .format = "oq_affine_q4_g64_bf16_v1",
                     .layout = "oq_rowsplit_k128_v1",
                     .shape = {64, 512},
                     .padded_k = 512,
                     .binding_use = "linear",
                     .activation_policy = "A16Only"});
    specs.push_back({.name = "mtp.layers.0.mlp.gate_proj",
                     .format = "oq_affine_q4_g64_bf16_v1",
                     .layout = "oq_rowsplit_k128_v1",
                     .shape = {64, 512},
                     .padded_k = 512,
                     .binding_use = "linear",
                     .activation_policy = "A16Only"});
    specs.push_back({.name = "mtp.embed_tokens",
                     .format = "oq_affine_q4_g64_bf16_v1",
                     .layout = "oq_rowsplit_k128_v1",
                     .shape = {64, 512},
                     .padded_k = 512,
                     .binding_use = "embedding",
                     .activation_policy = "A16Only"});
    specs.push_back({.name = "vision_model.norm.weight",
                     .format = "direct_bf16_v1",
                     .layout = "contiguous_le_v1",
                     .shape = {512}});
    auto text_specs = make_text_layer0_specs(
        "mxfp4_e2m1_g32_e8m0_v1", "mx_nax_tiled_bk128_v1", "AllowA4");
    specs.insert(specs.end(), text_specs.begin(), text_specs.end());

    const auto path = dir / "bundled_mxfp4_nax.omlxm5";
    write_synthetic_artifact(path, "mlx_mxfp4", specs);
    auto artifact = ArtifactModel::open(path, true);
    auto catalog = build_qwen_layer_catalog(artifact, cfg, 1024, 64);
    expect(catalog.layer(0).prefix == "language_model.model.layers.0",
           "layer_prefix bound to non-text companion layer");
    for (const auto &stage : catalog.global_stages()) {
      if (stage.role == "embedding") {
        expect(stage.object_name == "language_model.model.embed_tokens",
               "global embedding bound to companion object: " +
                   stage.object_name);
      } else if (stage.role == "final_norm") {
        expect(stage.object_name == "language_model.model.norm.weight",
               "global final_norm bound to vision object: " +
                   stage.object_name);
      }
    }
    auto desc = describe_qwen_topology(artifact, cfg, catalog, 1024, 64);
    expect(desc.quant_profile == QwenQuantProfile::mxfp4,
           "MXFP4 text with Q4 companion misclassified");
    auto profiles = compile_topology_profiles(desc);
    auto table = build_qwen_program_dispatch_table(artifact, catalog, profiles);
    expect(table.stats().candidate_count > 0,
           "dispatch table empty for MXFP4 NAX pre-tiled artifact");
  }

  // 2. MIVE-8: Q3 NAX pre-tiled text artifact builds topology & dispatch table.
  {
    const auto specs = make_text_layer0_specs(
        "oq_affine_q3_g64_bf16_v1", "oq_nax_tiled_g64_v1", "A16Only");
    const auto path = dir / "q3_nax.omlxm5";
    write_synthetic_artifact(path, "oq_affine", specs);
    auto artifact = ArtifactModel::open(path, true);
    auto catalog = build_qwen_layer_catalog(artifact, cfg, 1024, 64);
    auto desc = describe_qwen_topology(artifact, cfg, catalog, 1024, 64);
    expect(desc.quant_profile == QwenQuantProfile::oq_affine,
           "Q3 text artifact did not classify as oq_affine");
    auto profiles = compile_topology_profiles(desc);
    auto table = build_qwen_program_dispatch_table(artifact, catalog, profiles);
    expect(table.stats().candidate_count > 0,
           "dispatch table empty for Q3 NAX pre-tiled artifact");
  }

  // 3. MIVE-8: MXFP4 + Q3 mixed inside text component must be rejected.
  {
    auto specs = make_text_layer0_specs(
        "mxfp4_e2m1_g32_e8m0_v1", "mx_nax_tiled_bk128_v1", "AllowA4");
    for (auto &s : specs) {
      if (s.name == "language_model.model.layers.0.mlp.down_proj") {
        s.format = "oq_affine_q3_g64_bf16_v1";
        s.layout = "oq_nax_tiled_g64_v1";
        s.activation_policy = "A16Only";
      }
    }
    const auto path = dir / "mixed_mxfp4_q3.omlxm5";
    write_synthetic_artifact(path, "mlx_mxfp4", specs);
    auto artifact = ArtifactModel::open(path, true);
    auto catalog = build_qwen_layer_catalog(artifact, cfg, 1024, 64);
    expect_throws(
        [&] {
          static_cast<void>(
              describe_qwen_topology(artifact, cfg, catalog, 1024, 64));
        },
        "MXFP4 + Q3 text artifact was not rejected");
  }

  std::filesystem::remove_all(dir);
}

} // namespace

int main() {
  try {
    test_finite_profile_compilation();
    test_selection_and_fallback();
    test_moe_and_bounded_input();
    test_bundled_companions_and_q3_pretiled();
    std::cout << "B4.2 exact topology profile tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B4.2 topology profile failure: " << error.what() << '\n';
    return 1;
  }
}
