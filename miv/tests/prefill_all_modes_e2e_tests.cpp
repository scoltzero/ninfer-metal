#include "ane/ane_dual_offload.h"
#include "chunk_blend_cache.h"
#include "omlx_m5/artifact.h"
#include "omlx_m5/engine.h"
#include "omlx_m5/execution_contract.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/qwen_forward.h"

#include <algorithm>
#include <array>
#include <bit>
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
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {

using namespace omlx::m5;
using namespace std::chrono_literals;
using Json = nlohmann::json;
using U32 = std::uint32_t;
using U64 = std::uint64_t;

void require(bool cond, std::string_view msg) {
  if (!cond) {
    throw std::runtime_error("PREFILL-ALL-MODES-E2E FAIL: " + std::string(msg));
  }
}

struct ScopedEnv {
  std::string name;
  std::optional<std::string> previous;
  ScopedEnv(std::string key, const char *value) : name(std::move(key)) {
    if (const char *old = std::getenv(name.c_str())) {
      previous = std::string(old);
    }
    if (value != nullptr) {
      setenv(name.c_str(), value, 1);
    } else {
      unsetenv(name.c_str());
    }
  }
  ~ScopedEnv() {
    if (previous.has_value()) {
      setenv(name.c_str(), previous->c_str(), 1);
    } else {
      unsetenv(name.c_str());
    }
  }
};

struct TempDirGuard {
  std::filesystem::path path;
  ~TempDirGuard() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

std::uint16_t f32_to_bf16(float v) {
  const U32 bits = std::bit_cast<U32>(v);
  const U32 lsb = (bits >> 16U) & 1U;
  return static_cast<std::uint16_t>((bits + 0x7FFFU + lsb) >> 16U);
}

U64 det_hash(U32 seed, U64 idx) {
  U64 x = (U64(seed) * 0x9E3779B97F4A7C15ULL) ^ (idx * 0xBF58476D1CE4E5B9ULL);
  x = (x ^ (x >> 30U)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31U);
}

float det_val(U32 seed, U64 idx, float amp = 0.04F) {
  const U64 x = det_hash(seed, idx);
  const float u =
      static_cast<float>(static_cast<std::int32_t>(x & 0xFFFFU) - 32768) /
      32768.0F;
  return u * amp;
}

std::string hex_crc32(U32 crc) {
  std::ostringstream out;
  out << std::hex << std::nouppercase << std::setfill('0') << std::setw(8)
      << crc;
  return out.str();
}

void append_u32(std::vector<std::byte> &out, U32 v) {
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<std::byte>((v >> (i * 8)) & 0xFFU));
  }
}

void append_u64(std::vector<std::byte> &out, U64 v) {
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::byte>((v >> (i * 8)) & 0xFFU));
  }
}

void write_binary_file(const std::filesystem::path &path,
                       std::span<const std::byte> bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  require(out.good(), "failed to open file for write: " + path.string());
  out.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  require(out.good(), "failed writing file: " + path.string());
}

void write_text_file(const std::filesystem::path &path, std::string_view text) {
  write_binary_file(
      path, std::as_bytes(std::span<const char>(text.data(), text.size())));
}

double cosine_similarity(std::span<const float> a, std::span<const float> b) {
  require(a.size() == b.size() && !a.empty(), "cosine_similarity size mismatch");
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    require(std::isfinite(a[i]) && std::isfinite(b[i]),
            "non-finite value in logit vector");
    dot += double(a[i]) * double(b[i]);
    na += double(a[i]) * double(a[i]);
    nb += double(b[i]) * double(b[i]);
  }
  const double denom = std::sqrt(na) * std::sqrt(nb);
  return denom > 0.0 ? (dot / denom) : 0.0;
}

enum class WeightEnc {
  DirectBf16,
  Mxfp4Tiled,
  Mxfp4RowSplit3D,
  AffineQ4G64Tiled,
  AffineQ8G64Tiled,
  AffineQ8G128Untiled,
};

struct BuiltPlane {
  std::string kind;
  std::vector<std::byte> bytes;
};

struct BuiltObject {
  U32 id = 0;
  std::string name;
  std::string format;
  std::string layout;
  std::vector<U64> shape;
  std::optional<U64> padded_k;
  std::vector<BuiltPlane> planes;
  std::string binding_use;
  std::string activation_policy;
};

class SyntheticArtifactBuilder {
public:
  U32 add_bf16(std::string name, std::vector<U64> shape, U32 seed,
               float base_bias = 0.0F, float amp = 0.04F,
               std::string binding_use = "") {
    U64 count = 1;
    for (U64 d : shape) {
      count *= d;
    }
    std::vector<std::uint16_t> bf(count);
    for (U64 i = 0; i < count; ++i) {
      bf[i] = f32_to_bf16(base_bias + det_val(seed, i, amp));
    }
    const auto raw = std::as_bytes(std::span<const std::uint16_t>(bf));
    BuiltObject obj{
        .id = static_cast<U32>(objects_.size()),
        .name = std::move(name),
        .format = "direct_bf16_v1",
        .layout = "contiguous_le_v1",
        .shape = std::move(shape),
        .planes = {{.kind = "data", .bytes = {raw.begin(), raw.end()}}},
        .binding_use = std::move(binding_use),
        .activation_policy = "A16Only",
    };
    objects_.push_back(std::move(obj));
    return objects_.back().id;
  }

  U32 add_linear_2d(std::string name, U32 n, U32 k, WeightEnc enc, U32 seed,
                    std::string use = "linear", float amp = 0.04F) {
    if (enc == WeightEnc::DirectBf16) {
      return add_bf16(std::move(name), {n, k}, seed, 0.0F, amp, std::move(use));
    }
    const U32 id = static_cast<U32>(objects_.size());
    const U32 padded_k = ((k + 127U) / 128U) * 128U;
    BuiltObject obj{
        .id = id,
        .name = std::move(name),
        .shape = {n, k},
        .padded_k = padded_k,
        .binding_use = std::move(use),
    };

    if (enc == WeightEnc::Mxfp4Tiled) {
      obj.format = "mxfp4_e2m1_g32_e8m0_v1";
      obj.layout = "mx_nax_tiled_bk128_v1";
      obj.activation_policy = "AllowA4";
      const U32 bn = (n >= 65536U) ? 32U : 16U;
      const U32 padded_n = ((n + bn - 1U) / bn) * bn;
      const U32 k_blocks = k / 128U;
      std::vector<std::byte> nax_data(std::size_t(padded_n) * (k / 2U),
                                      std::byte{0});
      std::vector<std::byte> nax_scale(std::size_t(padded_n) * (k / 32U),
                                       std::byte{118});
      for (U32 r = 0; r < n; ++r) {
        const U32 tn = r / bn;
        const U32 rn = r % bn;
        for (U32 blk = 0; blk < k_blocks; ++blk) {
          const std::size_t tile_idx = std::size_t(tn) * k_blocks + blk;
          for (U32 sub = 0; sub < 4U; ++sub) {
            const std::uint8_t sc_code = static_cast<std::uint8_t>(
                119U + (det_hash(seed + 1U, U64(r) * 4096ULL + blk * 4ULL + sub) % 3U));
            nax_scale[tile_idx * (bn * 4U) + rn * 4U + sub] =
                static_cast<std::byte>(sc_code);
            for (U32 j = 0; j < 16U; ++j) {
              const U32 c0 = blk * 128U + sub * 32U + 2U * j;
              const U64 h0 = det_hash(seed, U64(r) * U64(k) + U64(c0));
              const U32 q0 = static_cast<U32>(h0 & 0xFU);
              const U32 q1 = static_cast<U32>((h0 >> 8U) & 0xFU);
              nax_data[tile_idx * (bn * 64U) + rn * 64U + sub * 16U + j] =
                  static_cast<std::byte>(q0 | (q1 << 4U));
            }
          }
        }
      }
      obj.planes.push_back({"nax_data", std::move(nax_data)});
      obj.planes.push_back({"nax_scale", std::move(nax_scale)});
    } else if (enc == WeightEnc::AffineQ4G64Tiled ||
               enc == WeightEnc::AffineQ8G64Tiled) {
      const U32 bits = (enc == WeightEnc::AffineQ4G64Tiled) ? 4U : 8U;
      const U32 gs = 64U;
      const U32 max_q = (1U << bits) - 1U;
      obj.format = "oq_affine_q" + std::to_string(bits) + "_g" +
                   std::to_string(gs) + "_bf16_v1";
      obj.layout = "oq_nax_tiled_g64_v1";
      obj.activation_policy = "A16Only";
      const U32 bn = (n >= 65536U) ? 32U : 16U;
      const U32 padded_n = ((n + bn - 1U) / bn) * bn;
      const U32 groups64 = k / 64U;
      std::vector<std::byte> nax_codes(
          bits == 8U ? (std::size_t(padded_n) * k)
                     : (std::size_t(padded_n) * (k / 2U)),
          std::byte{0});
      std::vector<std::byte> nax_sb(std::size_t(groups64) * padded_n * 4U,
                                    std::byte{0});
      for (U32 r = 0; r < n; ++r) {
        for (U32 g64 = 0; g64 < groups64; ++g64) {
          const float span =
              2.0F * amp * (0.6F + 0.1F * float((r + g64 + seed) % 5U));
          const std::uint16_t sc_bf = f32_to_bf16(span / float(max_q));
          const std::uint16_t bi_bf = f32_to_bf16(-0.5F * span);
          const U32 packed_sb =
              static_cast<U32>(sc_bf) | (static_cast<U32>(bi_bf) << 16U);
          std::memcpy(nax_sb.data() + (std::size_t(g64) * padded_n + r) * 4U,
                      &packed_sb, 4U);
        }
        const U32 tn = r / bn;
        const U32 rn = r % bn;
        for (U32 col = 0; col < k; ++col) {
          const U32 q =
              static_cast<U32>(det_hash(seed, U64(r) * U64(k) + U64(col))) &
              max_q;
          const U32 g64 = col / 64U;
          const U32 off = col % 64U;
          const std::size_t tile_idx = std::size_t(tn) * groups64 + g64;
          if (bits == 8U) {
            nax_codes[tile_idx * (bn * 64U) + rn * 64U + off] =
                static_cast<std::byte>(q);
          } else {
            const std::size_t cb_idx =
                tile_idx * (bn * 32U) + rn * 32U + (off >> 1U);
            const std::uint8_t cur =
                std::to_integer<std::uint8_t>(nax_codes[cb_idx]);
            nax_codes[cb_idx] = static_cast<std::byte>(
                cur | ((q & 0xFU) << ((off & 1U) * 4U)));
          }
        }
      }
      obj.planes.push_back({"nax_codes", std::move(nax_codes)});
      obj.planes.push_back({"nax_sb", std::move(nax_sb)});
    } else if (enc == WeightEnc::AffineQ8G128Untiled) {
      obj.format = "oq_affine_q8_g128_bf16_v1";
      obj.layout = "oq_rowsplit_k128_v1";
      obj.activation_policy = "A16Only";
      const U32 gs = 128U;
      const U32 padded_groups = padded_k / gs;
      std::vector<std::byte> code(std::size_t(n) * padded_k, std::byte{0});
      std::vector<std::uint16_t> scale(std::size_t(n) * padded_groups, 0U);
      std::vector<std::uint16_t> bias(std::size_t(n) * padded_groups, 0U);
      for (U32 r = 0; r < n; ++r) {
        for (U32 sg = 0; sg < k / gs; ++sg) {
          const float span =
              2.0F * amp * (0.6F + 0.1F * float((r + sg + seed) % 5U));
          scale[std::size_t(r) * padded_groups + sg] =
              f32_to_bf16(span / 255.0F);
          bias[std::size_t(r) * padded_groups + sg] = f32_to_bf16(-0.5F * span);
        }
        for (U32 col = 0; col < k; ++col) {
          const U32 q = (seed * 31U + r * 19U + col * 23U) & 0xFFU;
          code[std::size_t(r) * padded_k + col] = static_cast<std::byte>(q);
        }
      }
      const auto s_raw = std::as_bytes(std::span<const std::uint16_t>(scale));
      const auto b_raw = std::as_bytes(std::span<const std::uint16_t>(bias));
      obj.planes.push_back({"code", std::move(code)});
      obj.planes.push_back({"scale", {s_raw.begin(), s_raw.end()}});
      obj.planes.push_back({"bias", {b_raw.begin(), b_raw.end()}});
    } else {
      throw std::invalid_argument("unsupported 2D encoding");
    }

    objects_.push_back(std::move(obj));
    return id;
  }

  U32 add_moe_mxfp4_3d(std::string name, U32 num_experts, U32 n, U32 k,
                       U32 seed) {
    const U32 id = static_cast<U32>(objects_.size());
    const U32 padded_k = ((k + 127U) / 128U) * 128U;
    const std::size_t rows = std::size_t(num_experts) * n;
    std::vector<std::byte> data(rows * (padded_k / 2U), std::byte{0});
    std::vector<std::byte> scale(rows * (padded_k / 32U), std::byte{118});
    for (U32 e = 0; e < num_experts; ++e) {
      for (U32 r = 0; r < n; ++r) {
        const std::size_t flat_r = std::size_t(e) * n + r;
        for (U32 g = 0; g < k / 32U; ++g) {
          scale[flat_r * (padded_k / 32U) + g] = static_cast<std::byte>(
              118U + (det_hash(seed + 1U, flat_r * 256ULL + g) % 3U));
          for (U32 j = 0; j < 16U; ++j) {
            const U32 c0 = g * 32U + 2U * j;
            const U64 h0 = det_hash(seed, flat_r * U64(k) + U64(c0));
            const U32 q0 = static_cast<U32>(h0 & 0xFU);
            const U32 q1 = static_cast<U32>((h0 >> 8U) & 0xFU);
            data[flat_r * (padded_k / 2U) + g * 16U + j] =
                static_cast<std::byte>(q0 | (q1 << 4U));
          }
        }
      }
    }
    BuiltObject obj{
        .id = id,
        .name = std::move(name),
        .format = "mxfp4_e2m1_g32_e8m0_v1",
        .layout = "mx_expert_rowsplit_k128_v1",
        .shape = {num_experts, n, k},
        .padded_k = padded_k,
        .planes = {{"data", std::move(data)}, {"scale", std::move(scale)}},
        .binding_use = "moe",
        .activation_policy = "AllowA4",
    };
    objects_.push_back(std::move(obj));
    return id;
  }

  std::vector<std::byte> serialize(std::string_view source_profile,
                                   std::string_view source_repo,
                                   std::string_view source_rev = "rev1") const {
    std::vector<std::byte> payload;
    const auto align_payload = [&](std::size_t align) {
      while ((payload.size() % align) != 0U) {
        payload.push_back(std::byte{0});
      }
    };

    Json j_objects = Json::array();
    Json j_bindings = Json::array();
    for (const auto &obj : objects_) {
      align_payload(256U);
      const U64 obj_offset = payload.size();
      Json j_planes = Json::array();
      for (std::size_t pi = 0; pi < obj.planes.size(); ++pi) {
        if (pi > 0U) {
          align_payload(256U);
        }
        const auto &pl = obj.planes[pi];
        const U64 p_off = payload.size();
        payload.insert(payload.end(), pl.bytes.begin(), pl.bytes.end());
        j_planes.push_back({
            {"alignment", 256},
            {"bytes", pl.bytes.size()},
            {"crc32", hex_crc32(artifact_crc32(pl.bytes))},
            {"kind", pl.kind},
            {"offset", p_off},
            {"sha256", artifact_sha256_hex(pl.bytes)},
        });
      }
      const U64 obj_bytes = payload.size() - obj_offset;
      const auto obj_sha = artifact_sha256_hex(
          std::span<const std::byte>(payload).subspan(obj_offset, obj_bytes));
      Json jo = {
          {"format", obj.format},
          {"id", obj.id},
          {"layout", obj.layout},
          {"name", obj.name},
          {"object_bytes", obj_bytes},
          {"object_offset", obj_offset},
          {"object_sha256", obj_sha},
          {"planes", std::move(j_planes)},
          {"shape", obj.shape},
          {"source_tensors", {obj.name}},
      };
      if (obj.padded_k.has_value()) {
        jo["padded_k"] = *obj.padded_k;
      }
      j_objects.push_back(std::move(jo));

      if (!obj.format.starts_with("direct_") && !obj.binding_use.empty() &&
          obj.shape.size() >= 2U) {
        Json lead = Json::array();
        for (std::size_t i = 0; i + 2U < obj.shape.size(); ++i) {
          lead.push_back(obj.shape[i]);
        }
        j_bindings.push_back({
            {"activation_policy", obj.activation_policy},
            {"format", obj.format},
            {"in_features", obj.shape[obj.shape.size() - 1U]},
            {"leading_shape", std::move(lead)},
            {"name", obj.name},
            {"object_id", obj.id},
            {"out_features", obj.shape[obj.shape.size() - 2U]},
            {"use", obj.binding_use},
        });
      }
    }

    Json manifest = {
        {"bindings", std::move(j_bindings)},
        {"container_version", 1},
        {"model",
         {{"source_profile", source_profile},
          {"source_repo", source_repo},
          {"source_revision", source_rev}}},
        {"objects", std::move(j_objects)},
        {"payload_bytes", payload.size()},
        {"schema", std::string(kArtifactSchemaV1)},
        {"views", Json::array()},
    };
    const std::string manifest_str = manifest.dump();
    const auto manifest_bytes = std::as_bytes(
        std::span<const char>(manifest_str.data(), manifest_str.size()));
    const U64 raw_end = 64ULL + manifest_bytes.size();
    const U64 payload_offset = (raw_end + 16383ULL) & ~16383ULL;
    const U32 manifest_crc = artifact_crc32(manifest_bytes);

    std::vector<std::byte> file;
    for (char c : kArtifactMagic) {
      file.push_back(static_cast<std::byte>(c));
    }
    append_u32(file, 1U);
    append_u32(file, 64U);
    append_u64(file, 64ULL);
    append_u64(file, manifest_bytes.size());
    append_u64(file, payload_offset);
    append_u64(file, payload.size());
    append_u32(file, static_cast<U32>(objects_.size()));
    append_u32(file, manifest_crc);
    const U32 hdr_crc =
        artifact_crc32(std::span<const std::byte>(file.data(), 56U));
    append_u32(file, hdr_crc);
    append_u32(file, 0U);
    file.insert(file.end(), manifest_bytes.begin(), manifest_bytes.end());
    file.resize(static_cast<std::size_t>(payload_offset), std::byte{0});
    file.insert(file.end(), payload.begin(), payload.end());
    return file;
  }

private:
  std::vector<BuiltObject> objects_;
};

struct BuiltModel {
  std::filesystem::path model_dir;
  std::filesystem::path artifact_path;
  QwenModelConfig config;
};

// Build an 8-layer Qwen3.8 Dense Hybrid model ([GDN, GDN, GDN, Full-Attn] x 2)
// with HD=256, rotary_dim=64, H=1024, F=1024, gv=1024 so ALL 5 modules
// (W4A8 + out64 SwiGLU + dv64 GDN + Sage2/MInference + Dual-ANE + GDN-CacheBlend)
// satisfy every shape/alignment gate live.
BuiltModel build_dense_hybrid_model(const std::filesystem::path &dir,
                                    bool use_mxfp4) {
  std::filesystem::create_directories(dir);
  constexpr U32 kLayers = 8U;
  constexpr U32 h = 1024U;
  constexpr U32 f = 1024U;
  constexpr U32 v = 256U;
  constexpr U32 q_heads = 4U;
  constexpr U32 kv_heads = 2U;
  constexpr U32 hd = 256U;
  constexpr U32 q = q_heads * hd;   // 1024
  constexpr U32 kv = kv_heads * hd; // 512
  constexpr U32 gk_heads = 4U;
  constexpr U32 gv_heads = 8U;
  constexpr U32 gdn_kd = 128U;
  constexpr U32 gdn_vd = 128U;
  constexpr U32 gk = gk_heads * gdn_kd; // 512
  constexpr U32 gv = gv_heads * gdn_vd; // 1024
  constexpr U32 conv = 2U * gk + gv;    // 2048

  Json layer_types = Json::array();
  for (U32 li = 0; li < kLayers; ++li) {
    layer_types.push_back((li % 4U == 3U) ? "full_attention"
                                          : "linear_attention");
  }
  Json cfg = {
      {"model_type", "qwen3_5"},
      {"text_config",
       {{"model_type", "qwen3_5_text"},
        {"dtype", "bfloat16"},
        {"hidden_size", h},
        {"intermediate_size", f},
        {"vocab_size", v},
        {"num_hidden_layers", kLayers},
        {"num_attention_heads", q_heads},
        {"num_key_value_heads", kv_heads},
        {"head_dim", hd},
        {"max_position_embeddings", 32768},
        {"rms_norm_eps", 1e-6},
        {"rope_parameters",
         {{"rope_theta", 10000000.0},
          {"partial_rotary_factor", 0.25},
          {"mrope_interleaved", true}}},
        {"linear_num_key_heads", gk_heads},
        {"linear_num_value_heads", gv_heads},
        {"linear_key_head_dim", gdn_kd},
        {"linear_value_head_dim", gdn_vd},
        {"linear_conv_kernel_dim", 4},
        {"layer_types", layer_types},
        {"tie_word_embeddings", false}}},
  };
  write_text_file(dir / "config.json", cfg.dump(2));

  const WeightEnc main_enc =
      use_mxfp4 ? WeightEnc::Mxfp4Tiled : WeightEnc::AffineQ4G64Tiled;
  const WeightEnc small_enc =
      use_mxfp4 ? WeightEnc::Mxfp4Tiled : WeightEnc::AffineQ8G128Untiled;
  const WeightEnc head_enc =
      use_mxfp4 ? WeightEnc::Mxfp4Tiled : WeightEnc::AffineQ8G64Tiled;

  SyntheticArtifactBuilder b;
  b.add_bf16("model.embed_tokens.weight", {v, h}, 101U, 0.0F, 0.10F,
             "embedding");
  b.add_bf16("model.norm.weight", {h}, 102U, 0.0F, 0.03F);
  b.add_linear_2d("lm_head.weight", v, h, head_enc, 103U, "lm_head", 0.04F);

  for (U32 li = 0; li < kLayers; ++li) {
    const std::string p = "model.layers." + std::to_string(li);
    const U32 s = 1000U + li * 100U;
    b.add_bf16(p + ".input_layernorm.weight", {h}, s + 1U, 0.0F, 0.03F);
    if (li % 4U == 3U) {
      b.add_linear_2d(p + ".self_attn.q_proj.weight", 2U * q, h, main_enc,
                      s + 2U);
      b.add_linear_2d(p + ".self_attn.k_proj.weight", kv, h, main_enc, s + 3U);
      b.add_linear_2d(p + ".self_attn.v_proj.weight", kv, h, main_enc, s + 4U);
      b.add_bf16(p + ".self_attn.q_norm.weight", {hd}, s + 5U, 0.0F, 0.03F);
      b.add_bf16(p + ".self_attn.k_norm.weight", {hd}, s + 6U, 0.0F, 0.03F);
      b.add_linear_2d(p + ".self_attn.o_proj.weight", h, q, main_enc, s + 7U);
    } else {
      b.add_linear_2d(p + ".linear_attn.in_proj_qkv.weight", conv, h, main_enc,
                      s + 12U);
      b.add_linear_2d(p + ".linear_attn.in_proj_z.weight", gv, h, main_enc,
                      s + 13U);
      b.add_linear_2d(p + ".linear_attn.in_proj_a.weight", gv_heads, h,
                      small_enc, s + 14U);
      b.add_linear_2d(p + ".linear_attn.in_proj_b.weight", gv_heads, h,
                      small_enc, s + 15U);
      b.add_bf16(p + ".linear_attn.conv1d.weight", {conv, 1U, 4U}, s + 16U,
                 0.0F, 0.05F);
      b.add_bf16(p + ".linear_attn.A_log", {gv_heads}, s + 17U, 0.4F, 0.05F);
      b.add_bf16(p + ".linear_attn.dt_bias", {gv_heads}, s + 18U, -1.2F, 0.05F);
      b.add_bf16(p + ".linear_attn.norm.weight", {gdn_vd}, s + 19U, 1.0F,
                 0.03F);
      b.add_linear_2d(p + ".linear_attn.out_proj.weight", h, gv, main_enc,
                      s + 20U);
    }
    b.add_bf16(p + ".post_attention_layernorm.weight", {h}, s + 30U, 0.0F,
               0.03F);
    b.add_linear_2d(p + ".mlp.gate_proj.weight", f, h, main_enc, s + 31U);
    b.add_linear_2d(p + ".mlp.up_proj.weight", f, h, main_enc, s + 32U);
    b.add_linear_2d(p + ".mlp.down_proj.weight", h, f, main_enc, s + 33U);
  }

  const auto artifact_path = dir / "model.miv";
  write_binary_file(
      artifact_path,
      b.serialize(use_mxfp4 ? "mxfp4_v1" : "oq4e_v1",
                  use_mxfp4 ? "test/qwen38-dense-mxfp4"
                            : "test/qwen38-dense-oq4"));
  return {
      .model_dir = dir,
      .artifact_path = artifact_path,
      .config = load_qwen_model_config(dir / "config.json"),
  };
}

// Build a 2-layer Qwen3.8 MoE Hybrid model ([GDN, Full-Attn], E=256, top_k=8,
// MXFP4) exercising emit_moe_w4a8_nax_ffn + W4A8 shared expert + W4A16 decode.
BuiltModel build_moe_hybrid_model(const std::filesystem::path &dir) {
  std::filesystem::create_directories(dir);
  constexpr U32 kLayers = 2U;
  constexpr U32 h = 512U;
  constexpr U32 f = 256U;
  constexpr U32 shared_f = 512U;
  constexpr U32 num_experts = 256U;
  constexpr U32 top_k = 8U;
  constexpr U32 v = 256U;
  constexpr U32 q_heads = 2U;
  constexpr U32 kv_heads = 1U;
  constexpr U32 hd = 256U;
  constexpr U32 q = q_heads * hd;   // 512
  constexpr U32 kv = kv_heads * hd; // 256
  constexpr U32 gk_heads = 2U;
  constexpr U32 gv_heads = 4U;
  constexpr U32 gdn_kd = 128U;
  constexpr U32 gdn_vd = 128U;
  constexpr U32 gk = gk_heads * gdn_kd; // 256
  constexpr U32 gv = gv_heads * gdn_vd; // 512
  constexpr U32 conv = 2U * gk + gv;    // 1024

  Json cfg = {
      {"model_type", "qwen3_5_moe"},
      {"text_config",
       {{"model_type", "qwen3_5_moe_text"},
        {"dtype", "bfloat16"},
        {"hidden_size", h},
        {"moe_intermediate_size", f},
        {"shared_expert_intermediate_size", shared_f},
        {"num_experts", num_experts},
        {"num_experts_per_tok", top_k},
        {"vocab_size", v},
        {"num_hidden_layers", kLayers},
        {"num_attention_heads", q_heads},
        {"num_key_value_heads", kv_heads},
        {"head_dim", hd},
        {"max_position_embeddings", 8192},
        {"rms_norm_eps", 1e-6},
        {"rope_parameters",
         {{"rope_theta", 10000000.0},
          {"partial_rotary_factor", 0.25},
          {"mrope_interleaved", true}}},
        {"linear_num_key_heads", gk_heads},
        {"linear_num_value_heads", gv_heads},
        {"linear_key_head_dim", gdn_kd},
        {"linear_value_head_dim", gdn_vd},
        {"linear_conv_kernel_dim", 4},
        {"layer_types", {"linear_attention", "full_attention"}},
        {"tie_word_embeddings", false}}},
  };
  write_text_file(dir / "config.json", cfg.dump(2));

  SyntheticArtifactBuilder b;
  b.add_bf16("model.embed_tokens.weight", {v, h}, 501U, 0.0F, 0.10F,
             "embedding");
  b.add_bf16("model.norm.weight", {h}, 502U, 0.0F, 0.03F);
  b.add_linear_2d("lm_head.weight", v, h, WeightEnc::Mxfp4Tiled, 503U,
                  "lm_head", 0.04F);

  for (U32 li = 0; li < kLayers; ++li) {
    const std::string p = "model.layers." + std::to_string(li);
    const U32 s = 2000U + li * 100U;
    b.add_bf16(p + ".input_layernorm.weight", {h}, s + 1U, 0.0F, 0.03F);
    if (li == 1U) {
      b.add_linear_2d(p + ".self_attn.q_proj.weight", 2U * q, h,
                      WeightEnc::Mxfp4Tiled, s + 2U);
      b.add_linear_2d(p + ".self_attn.k_proj.weight", kv, h,
                      WeightEnc::Mxfp4Tiled, s + 3U);
      b.add_linear_2d(p + ".self_attn.v_proj.weight", kv, h,
                      WeightEnc::Mxfp4Tiled, s + 4U);
      b.add_bf16(p + ".self_attn.q_norm.weight", {hd}, s + 5U, 0.0F, 0.03F);
      b.add_bf16(p + ".self_attn.k_norm.weight", {hd}, s + 6U, 0.0F, 0.03F);
      b.add_linear_2d(p + ".self_attn.o_proj.weight", h, q,
                      WeightEnc::Mxfp4Tiled, s + 7U);
    } else {
      b.add_linear_2d(p + ".linear_attn.in_proj_qkv.weight", conv, h,
                      WeightEnc::Mxfp4Tiled, s + 12U);
      b.add_linear_2d(p + ".linear_attn.in_proj_z.weight", gv, h,
                      WeightEnc::Mxfp4Tiled, s + 13U);
      b.add_linear_2d(p + ".linear_attn.in_proj_a.weight", gv_heads, h,
                      WeightEnc::Mxfp4Tiled, s + 14U);
      b.add_linear_2d(p + ".linear_attn.in_proj_b.weight", gv_heads, h,
                      WeightEnc::Mxfp4Tiled, s + 15U);
      b.add_bf16(p + ".linear_attn.conv1d.weight", {conv, 1U, 4U}, s + 16U,
                 0.0F, 0.05F);
      b.add_bf16(p + ".linear_attn.A_log", {gv_heads}, s + 17U, 0.4F, 0.05F);
      b.add_bf16(p + ".linear_attn.dt_bias", {gv_heads}, s + 18U, -1.2F, 0.05F);
      b.add_bf16(p + ".linear_attn.norm.weight", {gdn_vd}, s + 19U, 1.0F,
                 0.03F);
      b.add_linear_2d(p + ".linear_attn.out_proj.weight", h, gv,
                      WeightEnc::Mxfp4Tiled, s + 20U);
    }
    b.add_bf16(p + ".post_attention_layernorm.weight", {h}, s + 30U, 0.0F,
               0.03F);
    b.add_bf16(p + ".mlp.gate.weight", {num_experts, h}, s + 31U, 0.0F, 0.04F,
               "linear");
    b.add_moe_mxfp4_3d(p + ".mlp.switch_mlp.gate_proj.weight", num_experts, f,
                       h, s + 32U);
    b.add_moe_mxfp4_3d(p + ".mlp.switch_mlp.up_proj.weight", num_experts, f, h,
                       s + 33U);
    b.add_moe_mxfp4_3d(p + ".mlp.switch_mlp.down_proj.weight", num_experts, h,
                       f, s + 34U);
    b.add_bf16(p + ".mlp.shared_expert_gate.weight", {1U, h}, s + 35U, 0.0F,
               0.04F, "linear");
    b.add_linear_2d(p + ".mlp.shared_expert.gate_proj.weight", shared_f, h,
                    WeightEnc::Mxfp4Tiled, s + 36U);
    b.add_linear_2d(p + ".mlp.shared_expert.up_proj.weight", shared_f, h,
                    WeightEnc::Mxfp4Tiled, s + 37U);
    b.add_linear_2d(p + ".mlp.shared_expert.down_proj.weight", h, shared_f,
                    WeightEnc::Mxfp4Tiled, s + 38U);
  }

  const auto artifact_path = dir / "model.miv";
  write_binary_file(artifact_path,
                    b.serialize("mxfp4_moe_v1", "test/qwen38-moe-mxfp4"));
  return {
      .model_dir = dir,
      .artifact_path = artifact_path,
      .config = load_qwen_model_config(dir / "config.json"),
  };
}

std::vector<U32> make_tokens(std::size_t n, U32 seed, U32 vocab = 240U) {
  std::vector<U32> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = static_cast<U32>((U64(seed) * 131ULL + U64(i) * 37ULL + 7ULL) %
                              U64(vocab));
  }
  return out;
}

std::vector<std::int32_t> to_i32(std::span<const U32> toks) {
  std::vector<std::int32_t> out(toks.size());
  for (std::size_t i = 0; i < toks.size(); ++i) {
    out[i] = static_cast<std::int32_t>(toks[i]);
  }
  return out;
}

std::vector<std::int32_t> run_engine_request(Engine &engine,
                                             std::span<const U32> prompt,
                                             std::int32_t max_new_tokens) {
  SubmitRequest req;
  req.prompt_tokens = to_i32(prompt);
  req.max_new_tokens = max_new_tokens;
  req.sampling.temperature = 0.0F;
  req.sampling.seed = 42ULL;
  const auto id = engine.submit(req);
  const auto deadline = std::chrono::steady_clock::now() + 180s;
  std::vector<std::int32_t> emitted;
  bool done = false;
  while (!done && std::chrono::steady_clock::now() < deadline) {
    for (auto &ev : engine.poll_events(10ms)) {
      require(ev.request_id == id, "foreign request event");
      if (ev.kind == EventKind::tokens) {
        emitted.insert(emitted.end(), ev.tokens.begin(), ev.tokens.end());
      } else if (ev.kind == EventKind::finished) {
        done = true;
      } else {
        throw std::runtime_error("Engine request failed: " + ev.error);
      }
    }
  }
  require(done, "Engine request timed out");
  require(static_cast<std::int32_t>(emitted.size()) == max_new_tokens,
          "unexpected generated token count");
  return emitted;
}

LoadOptions make_load_options(const BuiltModel &m,
                              const std::filesystem::path &libs,
                              std::int32_t max_context,
                              PrefillLinearMode linear_mode,
                              PrefillAttentionMode attn_mode,
                              bool sparse_minf, bool cache_blend,
                              bool ane_offload, bool disable_prefix_cache) {
  LoadOptions opts;
  opts.model_dir = m.model_dir.string();
  opts.max_context = max_context;
  opts.kv_dtype = "affine4_g128";
  opts.sampling_contract = kDefaultServingSamplingContract;
  opts.prefill_linear_mode = linear_mode;
  opts.prefill_attention_mode = attn_mode;
  opts.prefill_sparse_minference = sparse_minf;
  opts.prefill_cache_blend = cache_blend;
  opts.prefill_ane_offload = ane_offload;
  opts.path_overrides["artifact"] = m.artifact_path.string();
  opts.path_overrides["kernel_directory"] = libs.string();
  if (disable_prefix_cache) {
    opts.path_overrides["prefix_cache_bytes"] = "0";
  }
  return opts;
}

struct LoadedDirect {
  MetalContext ctx;
  MaterializedModel weights;
  QwenLayerCatalog layers;
  std::string digest;

  static LoadedDirect open(const BuiltModel &m, U32 max_context) {
    auto ctx = MetalContext::create();
    auto art = ArtifactModel::open(m.artifact_path, true);
    art.verify_payload();
    auto digest = execution_artifact_digest(art);
    auto plan = plan_model_materialization(art, ctx);
    auto weights = materialize_model(art, std::move(plan), ctx);
    auto layers = build_qwen_layer_catalog(art, m.config, max_context, 64);
    return LoadedDirect{
        .ctx = std::move(ctx),
        .weights = std::move(weights),
        .layers = std::move(layers),
        .digest = std::move(digest),
    };
  }
};

std::unique_ptr<QwenForward>
make_forward(LoadedDirect &ld, const BuiltModel &m,
             const std::filesystem::path &libs, U32 max_context,
             QwenPrefillOptions pf_opts, bool enable_flash_nax = true) {
  pf_opts.artifact_digest = ld.digest;
  return std::make_unique<QwenForward>(
      ld.ctx, ld.weights, ld.layers, m.config, max_context, libs,
      std::span<const U32>{}, false, false, false, std::nullopt, true, false,
      std::nullopt, LinearNumericPolicy::RowExactPortableV1,
      SamplingContract::PositionCoupledV1, AttentionNumericPolicy::RowExactV1,
      false, "affine4_g128", nullptr, enable_flash_nax, 0U, nullptr, nullptr,
      pf_opts);
}

// Run full chunked prefill and return last-token logits directly from the
// final prefill chunk (no artificial T=1 decode split).
std::vector<float> run_prefill_and_last_logits(QwenForward &fwd,
                                               std::span<const U32> tokens,
                                               U32 *out_first_token = nullptr) {
  fwd.reset();
  const U32 ws_rows = fwd.memory_plan().workspace_rows;
  std::size_t off = 0;
  while (tokens.size() - off > ws_rows) {
    static_cast<void>(fwd.run_trunk(tokens.subspan(off, ws_rows), true));
    off += ws_rows;
  }
  const auto last_chunk = tokens.subspan(off);
  GpuSamplingOptions samp{};
  samp.absolute_position = tokens.size() - 1U;
  samp.contract = SamplingContract::PositionCoupledV1;
  const auto res = fwd.run(last_chunk, samp, true, true, false);
  if (out_first_token != nullptr) {
    *out_first_token = res.token;
  }
  const std::size_t vocab = res.logits.size() / last_chunk.size();
  return std::vector<float>(res.logits.end() - static_cast<std::ptrdiff_t>(vocab),
                            res.logits.end());
}

// Execute a single exact chunk of size T (e.g., T = 1024 for ANE) via run_trunk
// followed by a GPU-only tail prefill chunk to read post-chunk logits without
// disabling ANE via inspect_tensors=true.
std::vector<float> run_chunk_then_probe_logits(QwenForward &fwd,
                                               std::span<const U32> chunk_toks,
                                               std::span<const U32> tail_toks) {
  fwd.reset();
  static_cast<void>(fwd.run_trunk(chunk_toks, true));
  GpuSamplingOptions samp{};
  samp.absolute_position = chunk_toks.size() + tail_toks.size() - 1U;
  samp.contract = SamplingContract::PositionCoupledV1;
  const auto res = fwd.run(tail_toks, samp, true, true, false);
  const std::size_t vocab = res.logits.size() / tail_toks.size();
  return std::vector<float>(res.logits.end() - static_cast<std::ptrdiff_t>(vocab),
                            res.logits.end());
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2,
            "Usage: omlx_m5_prefill_all_modes_e2e_tests METALLIB_DIR");
    const std::filesystem::path libs(argv[1]);

    // Unlock Metal 4 NAX kernels on M4 + macOS 27 so ALL 5 modules execute live.
    ScopedEnv force_nax("OMLX_M5_FORCE_NAX_APPLE10", "1");

    const auto tmp_root =
        std::filesystem::temp_directory_path() /
        ("omlx_m5_prefill_all_modes_" + std::to_string(::getpid()));
    TempDirGuard cleanup{tmp_root};

    const auto dense_oq = build_dense_hybrid_model(tmp_root / "dense_oq", false);
    const auto dense_mx = build_dense_hybrid_model(tmp_root / "dense_mx", true);
    const auto moe_mx = build_moe_hybrid_model(tmp_root / "moe_mx");

    Json report = Json::object();

    // =========================================================================
    // 1. bf16 + exact_nax (0-ULP / exact ICB parity + decode continuation)
    // =========================================================================
    {
      const auto prompt = make_tokens(128U, 11U);
      for (const auto &[label, model] :
           std::array<std::pair<std::string, const BuiltModel *>, 2>{
               {{"dense_oq", &dense_oq}, {"dense_mx", &dense_mx}}}) {
        auto ld = LoadedDirect::open(*model, 1024U);
        auto fwd = make_forward(ld, *model, libs, 1024U, {});
        GpuSamplingOptions samp{};
        samp.absolute_position = prompt.size() - 1U;
        samp.contract = SamplingContract::PositionCoupledV1;

        fwd->reset();
        const auto direct = fwd->run(prompt, samp, true, false, false);
        fwd->reset();
        const auto icb = fwd->run(prompt, samp, true, true, false);
        require(direct.logits == icb.logits && direct.token == icb.token,
                label + " bf16+exact_nax direct vs ICB mismatch");

        auto eng = Engine::create(make_load_options(
            *model, libs, 1024, PrefillLinearMode::BFloat16,
            PrefillAttentionMode::ExactNax, false, false, false, true));
        const auto gen1 = run_engine_request(*eng, prompt, 8);
        const auto gen2 = run_engine_request(*eng, prompt, 8);
        require(gen1 == gen2, label + " bf16+exact_nax decode non-deterministic");
        const auto st = eng->stats();
        require(st.prefill_linear_mode == "bf16" &&
                    st.prefill_attention_mode == "exact_nax" &&
                    st.prefill_w4a8_tokens == 0 && st.prefill_sage2_tokens == 0,
                label + " bf16+exact_nax baseline counters polluted");
        eng->shutdown();
      }
      report["case1_bf16_exact_nax"] = "passed";
    }

    // =========================================================================
    // 2. w4a8 + exact_nax (Dense oq_affine, Dense mxfp4, MoE mxfp4; decode W4A16)
    // =========================================================================
    {
      const auto prompt = make_tokens(128U, 22U);
      Json case2 = Json::object();
      for (const auto &[label, model] :
           std::array<std::pair<std::string, const BuiltModel *>, 3>{
               {{"dense_oq", &dense_oq},
                {"dense_mx", &dense_mx},
                {"moe_mx", &moe_mx}}}) {
        auto ld = LoadedDirect::open(*model, 1024U);
        auto fwd_bf16 = make_forward(ld, *model, libs, 1024U, {});
        auto fwd_w4a8 = make_forward(
            ld, *model, libs, 1024U,
            QwenPrefillOptions{.linear_mode = PrefillLinearMode::W4A8});

        const auto ref_logits = run_prefill_and_last_logits(*fwd_bf16, prompt);
        const auto w4a8_logits = run_prefill_and_last_logits(*fwd_w4a8, prompt);
        const auto tel_after_pf = fwd_w4a8->prefill_telemetry();
        require(tel_after_pf.prefill_w4a8_tokens >= 127ULL,
                label + " W4A8 prefill_w4a8_tokens did not increment");

        // Verify decode T=1 stays W4A16 (zero W4A8 tokens added during decode).
        U32 dec_tok = 5U;
        GpuSamplingOptions ds{};
        ds.absolute_position = prompt.size();
        ds.contract = SamplingContract::PositionCoupledV1;
        for (U32 step = 0; step < 8U; ++step) {
          ds.absolute_position = prompt.size() + step;
          const auto dr = fwd_w4a8->run(std::span(&dec_tok, 1U), ds, false, true);
          dec_tok = dr.token;
        }
        const auto tel_after_dec = fwd_w4a8->prefill_telemetry();
        require(tel_after_dec.prefill_w4a8_tokens ==
                    tel_after_pf.prefill_w4a8_tokens,
                label + " decode T=1..8 unexpectedly used W4A8 instead of W4A16");

        const double cos_sim = cosine_similarity(ref_logits, w4a8_logits);
        require(cos_sim >= 0.96,
                label + " W4A8 cosine similarity too low: " +
                    std::to_string(cos_sim));

        // Verify NativeBackend end-to-end stats and decode W4A16 isolation.
        auto eng = Engine::create(make_load_options(
            *model, libs, 1024, PrefillLinearMode::W4A8,
            PrefillAttentionMode::ExactNax, false, false, false, true));
        static_cast<void>(run_engine_request(*eng, prompt, 1));
        const auto st1 = eng->stats();
        require(st1.prefill_linear_mode == "w4a8" &&
                    st1.prefill_w4a8_tokens >= 128ULL,
                label + " Engine W4A8 telemetry missing");
        static_cast<void>(run_engine_request(*eng, prompt, 8));
        const auto st2 = eng->stats();
        require(st2.prefill_w4a8_tokens == 2ULL * st1.prefill_w4a8_tokens,
                label + " Engine decode steps unexpectedly incremented W4A8 count");
        eng->shutdown();

        case2[label] = {
            {"cos_sim", cos_sim},
            {"prefill_w4a8_tokens", st1.prefill_w4a8_tokens},
        };
      }
      report["case2_w4a8_exact_nax"] = case2;
    }

    // =========================================================================
    // 3. bf16 + sage2 & bf16 + sage2 + minference (L<4096 bypass, L>=4096 active)
    // 4. w4a8 + sage2 + minference (orthogonal composition)
    // =========================================================================
    {
      constexpr U32 kLongCtx = 5120U;
      const auto short_prompt = make_tokens(256U, 33U);
      const auto long_prompt = make_tokens(4608U, 44U);

      auto ld = LoadedDirect::open(dense_oq, kLongCtx);
      auto fwd_exact = make_forward(ld, dense_oq, libs, kLongCtx, {});
      auto fwd_sage2 = make_forward(
          ld, dense_oq, libs, kLongCtx,
          QwenPrefillOptions{.attention_mode = PrefillAttentionMode::Sage2});
      auto fwd_minf = make_forward(
          ld, dense_oq, libs, kLongCtx,
          QwenPrefillOptions{.attention_mode = PrefillAttentionMode::Sage2,
                             .sparse_minference = true});
      auto fwd_w4a8_minf = make_forward(
          ld, dense_oq, libs, kLongCtx,
          QwenPrefillOptions{.linear_mode = PrefillLinearMode::W4A8,
                             .attention_mode = PrefillAttentionMode::Sage2,
                             .sparse_minference = true});

      // Short context (< 4096): Sage2 active, MInference cleanly bypassed.
      const auto short_ref =
          run_prefill_and_last_logits(*fwd_exact, short_prompt);
      const auto short_minf =
          run_prefill_and_last_logits(*fwd_minf, short_prompt);
      const auto short_tel = fwd_minf->prefill_telemetry();
      require(short_tel.prefill_sage2_tokens >= 255ULL,
              "Sage2 short-context tokens not counted");
      require(short_tel.prefill_minference_blocks_skipped == 0ULL,
              "MInference must be bypassed for context < 4096");
      const double short_cos = cosine_similarity(short_ref, short_minf);
      require(short_cos >= 0.985,
              "Sage2 short-context cosine similarity too low: " +
                  std::to_string(short_cos));

      // Long context (>= 4096): Exact vs Sage2 vs Sage2+MInference vs W4A8+Sage2+MInference.
      const auto long_ref = run_prefill_and_last_logits(*fwd_exact, long_prompt);
      const auto long_sage2 =
          run_prefill_and_last_logits(*fwd_sage2, long_prompt);
      const auto long_minf = run_prefill_and_last_logits(*fwd_minf, long_prompt);
      const auto long_w4a8_minf =
          run_prefill_and_last_logits(*fwd_w4a8_minf, long_prompt);

      const auto tel_minf = fwd_minf->prefill_telemetry();
      require(tel_minf.prefill_sage2_tokens >= 4607ULL,
              "Sage2+MInference long-context sage2 tokens missing");
      require(tel_minf.prefill_minference_blocks_skipped > 0ULL &&
                  tel_minf.prefill_minference_blocks_total >
                      tel_minf.prefill_minference_blocks_skipped,
              "MInference did not skip KV blocks at L=4608");

      const auto tel_w4a8_minf = fwd_w4a8_minf->prefill_telemetry();
      require(tel_w4a8_minf.prefill_w4a8_tokens >= 4607ULL &&
                  tel_w4a8_minf.prefill_sage2_tokens >= 4607ULL &&
                  tel_w4a8_minf.prefill_minference_blocks_skipped > 0ULL,
              "W4A8 + Sage2 + MInference composition counters incomplete");

      const double cos_sage2 = cosine_similarity(long_ref, long_sage2);
      const double cos_minf = cosine_similarity(long_ref, long_minf);
      const double cos_w4a8_minf = cosine_similarity(long_ref, long_w4a8_minf);
      require(cos_sage2 >= 0.98,
              "Sage2 L=4608 cosine too low: " + std::to_string(cos_sage2));
      require(cos_minf >= 0.95,
              "Sage2+MInference L=4608 cosine too low: " +
                  std::to_string(cos_minf));
      require(cos_w4a8_minf >= 0.93,
              "W4A8+Sage2+MInference L=4608 cosine too low: " +
                  std::to_string(cos_w4a8_minf));

      // Verify end-to-end Engine::create for W4A8 + Sage2 + MInference.
      auto eng = Engine::create(make_load_options(
          dense_oq, libs, static_cast<std::int32_t>(kLongCtx),
          PrefillLinearMode::W4A8, PrefillAttentionMode::Sage2, true, false,
          false, true));
      static_cast<void>(run_engine_request(*eng, long_prompt, 4));
      const auto st = eng->stats();
      require(st.prefill_linear_mode == "w4a8" &&
                  st.prefill_attention_mode == "sage2" &&
                  st.prefill_sparse_minference &&
                  st.prefill_sparse_minference_effective &&
                  st.prefill_w4a8_tokens >= 4608ULL &&
                  st.prefill_sage2_tokens >= 4608ULL &&
                  st.prefill_minference_blocks_skipped > 0ULL,
              "Engine W4A8+Sage2+MInference telemetry mismatch");
      eng->shutdown();

      report["case3_4_sage2_minference"] = {
          {"short_cos", short_cos},
          {"long_sage2_cos", cos_sage2},
          {"long_minf_cos", cos_minf},
          {"long_w4a8_minf_cos", cos_w4a8_minf},
          {"minf_blocks_skipped", st.prefill_minference_blocks_skipped},
          {"minf_blocks_total", st.prefill_minference_blocks_total},
      };
    }

    // =========================================================================
    // 5. bf16 + ane_offload & w4a8 + ane_offload (T >= 1024 orthogonal composition)
    // =========================================================================
    {
      constexpr U32 kCtx = 2048U;
      const auto chunk1024 = make_tokens(1024U, 55U);
      const auto prompt1056 = make_tokens(1056U, 55U);
      const bool hw_ane = ane::is_ane_available();

      auto ld = LoadedDirect::open(dense_oq, kCtx);
      auto fwd_gpu_bf16 = make_forward(ld, dense_oq, libs, kCtx, {});
      auto fwd_ane_bf16 = make_forward(
          ld, dense_oq, libs, kCtx, QwenPrefillOptions{.ane_offload = true});
      auto fwd_ane_w4a8 = make_forward(
          ld, dense_oq, libs, kCtx,
          QwenPrefillOptions{.linear_mode = PrefillLinearMode::W4A8,
                             .ane_offload = true});

      const auto tail64 = make_tokens(64U, 61U);
      const auto ref_logits =
          run_chunk_then_probe_logits(*fwd_gpu_bf16, chunk1024, tail64);
      const auto ane_bf16_logits =
          run_chunk_then_probe_logits(*fwd_ane_bf16, chunk1024, tail64);
      const auto ane_w4a8_logits =
          run_chunk_then_probe_logits(*fwd_ane_w4a8, chunk1024, tail64);

      const double cos_ane_bf16 = cosine_similarity(ref_logits, ane_bf16_logits);
      const double cos_ane_w4a8 = cosine_similarity(ref_logits, ane_w4a8_logits);
      require(cos_ane_bf16 >= 0.96,
              "bf16+ane_offload cosine too low: " +
                  std::to_string(cos_ane_bf16));
      require(cos_ane_w4a8 >= 0.94,
              "w4a8+ane_offload cosine too low: " +
                  std::to_string(cos_ane_w4a8));

      const auto t_bf16 = fwd_ane_bf16->prefill_telemetry();
      const auto t_w4a8 = fwd_ane_w4a8->prefill_telemetry();
      require(t_w4a8.prefill_w4a8_tokens >= 1024ULL,
              "w4a8+ane_offload disabled W4A8 (violates strict orthogonality!)");
      if (hw_ane && fwd_ane_bf16->ane_offload_active()) {
        require(std::abs(t_bf16.prefill_ane_split_alpha - 0.25) < 1e-6,
                "bf16+ane_offload split alpha != 0.25");
        require(t_bf16.prefill_ane_offload_tokens >= 1024ULL,
                "bf16+ane_offload did not offload T=1024 chunk");
      }
      if (hw_ane && fwd_ane_w4a8->ane_offload_active()) {
        require(std::abs(t_w4a8.prefill_ane_split_alpha - 0.15) < 1e-6,
                "w4a8+ane_offload split alpha != 0.15");
        require(t_w4a8.prefill_ane_offload_tokens >= 1024ULL,
                "w4a8+ane_offload did not offload T=1024 chunk");
      }

      // Engine::create end-to-end check for W4A8 + ANE with max_context=4096
      // (ws_rows == 2048) and a 2080-token prompt (1024 + 1024 + 32) verifying
      // ANE is NOT silently bypassed on >= 2048-token prompts!
      constexpr std::int32_t kEngCtx = 4096;
      const auto prompt2080 = make_tokens(2080U, 55U);
      auto eng = Engine::create(make_load_options(
          dense_oq, libs, kEngCtx, PrefillLinearMode::W4A8,
          PrefillAttentionMode::ExactNax, false, false, true, true));
      static_cast<void>(run_engine_request(*eng, prompt2080, 4));
      const auto st = eng->stats();
      require(st.prefill_linear_mode == "w4a8" && st.prefill_ane_offload &&
                  st.prefill_w4a8_tokens >= 2080ULL,
              "Engine w4a8+ane_offload 2080-tok telemetry mismatch");
      if (st.prefill_ane_active) {
        require(std::abs(st.prefill_ane_split_alpha - 0.15) < 1e-6 &&
                    st.prefill_ane_offload_tokens >= 2048ULL &&
                    st.prefill_ane_fallback_tokens >= 32ULL,
                "Engine w4a8+ane_offload >=2048 prompt chunk/tail counters wrong (offload=" +
                    std::to_string(st.prefill_ane_offload_tokens) +
                    ", fallback=" +
                    std::to_string(st.prefill_ane_fallback_tokens) + ")");
      }
      eng->shutdown();

      report["case5_dual_ane"] = {
          {"hw_ane_available", hw_ane},
          {"bf16_ane_active", fwd_ane_bf16->ane_offload_active()},
          {"w4a8_ane_active", fwd_ane_w4a8->ane_offload_active()},
          {"bf16_ane_cos", cos_ane_bf16},
          {"w4a8_ane_cos", cos_ane_w4a8},
          {"engine_ane_offload_tokens", st.prefill_ane_offload_tokens},
          {"engine_ane_fallback_tokens", st.prefill_ane_fallback_tokens},
          {"engine_w4a8_tokens", st.prefill_w4a8_tokens},
      };
    }

    // =========================================================================
    // 6. cache_blend = true: cold populate -> warm reordered [Doc B][Doc A][Query]
    //    under BOTH bf16 + exact_nax AND w4a8 + sage2 + minference!
    // =========================================================================
    {
      constexpr U32 kCtx = 2048U;
      auto make_doc = [](U32 len, U32 seed) {
        std::vector<U32> d = make_tokens(len, seed, 240U);
        d.front() = 250U; // prefix delimiter
        d.back() = 251U;  // suffix delimiter
        return d;
      };
      const auto doc_a = make_doc(320U, 71U);
      const auto doc_b = make_doc(320U, 83U);
      const auto query = make_tokens(128U, 97U, 240U);

      std::vector<U32> prompt_ab;
      prompt_ab.insert(prompt_ab.end(), doc_a.begin(), doc_a.end());
      prompt_ab.insert(prompt_ab.end(), doc_b.begin(), doc_b.end());
      prompt_ab.insert(prompt_ab.end(), query.begin(), query.end());

      std::vector<U32> prompt_ba;
      prompt_ba.insert(prompt_ba.end(), doc_b.begin(), doc_b.end());
      prompt_ba.insert(prompt_ba.end(), doc_a.begin(), doc_a.end());
      prompt_ba.insert(prompt_ba.end(), query.begin(), query.end());

      detail::BlendSegmentConfig seg_cfg;
      seg_cfg.prefix_delimiters = {{250U}};
      seg_cfg.suffix_delimiters = {{251U}};

      auto ld = LoadedDirect::open(dense_oq, kCtx);
      Json case6 = Json::object();

      struct BlendModeCase {
        std::string name;
        QwenPrefillOptions opts;
      };
      const std::array<BlendModeCase, 2> modes{{
          {"bf16_exact_nax",
           QwenPrefillOptions{.linear_mode = PrefillLinearMode::BFloat16,
                              .attention_mode = PrefillAttentionMode::ExactNax,
                              .sparse_minference = false,
                              .cache_blend = true}},
          {"w4a8_sage2_minference",
           QwenPrefillOptions{.linear_mode = PrefillLinearMode::W4A8,
                              .attention_mode = PrefillAttentionMode::Sage2,
                              .sparse_minference = true,
                              .cache_blend = true}},
      }};

      for (const auto &mc : modes) {
        auto fwd = make_forward(ld, dense_oq, libs, kCtx, mc.opts);
        require(fwd->cache_blend_eligible(),
                mc.name + " cache_blend_eligible() returned false");

        // Fresh cold baseline on reordered [Doc B][Doc A][Query].
        const auto cold_ba_logits = run_prefill_and_last_logits(*fwd, prompt_ba);

        detail::ChunkBlendCache cache(1ULL << 30);
        const U64 exec_hash = detail::compute_blend_model_exec_hash(
            ld.digest, "affine4_g128",
            prefill_linear_mode_to_string(mc.opts.linear_mode),
            prefill_attention_mode_to_string(mc.opts.attention_mode),
            mc.opts.sparse_minference,
            mc.opts.ane_mode != PrefillAneMode::Off
                ? prefill_ane_mode_to_string(mc.opts.ane_mode)
                : (mc.opts.ane_offload ? "auto" : "off"));

        GpuSamplingOptions samp{};
        samp.absolute_position = prompt_ab.size() - 1U;
        samp.contract = SamplingContract::PositionCoupledV1;

        // Cold populate [Doc A][Doc B][Query].
        fwd->reset();
        const auto r_cold = fwd->run_blend_prefill(
            prompt_ab, 0U, cache, exec_hash, samp, true, &seg_cfg);
        require(cache.size() >= 2U,
                mc.name + " cold blend did not populate both Doc A & Doc B");
        require(r_cold.prefill_delta.prefill_cache_blend_hit_tokens == 0ULL,
                mc.name + " cold blend unexpectedly reported hits");

        // Warm reordered hit [Doc B][Doc A][Query].
        fwd->reset();
        const auto r_warm = fwd->run_blend_prefill(
            prompt_ba, 0U, cache, exec_hash, samp, true, &seg_cfg);
        const auto &wd = r_warm.prefill_delta;
        require(wd.prefill_cache_blend_hit_tokens == 640ULL,
                mc.name + " warm reordered hit_tokens != 640 (got " +
                    std::to_string(wd.prefill_cache_blend_hit_tokens) + ")");
        require(wd.prefill_cache_blend_recomputed_tokens > 0ULL &&
                    wd.prefill_cache_blend_recomputed_tokens <
                        wd.prefill_cache_blend_hit_tokens,
                mc.name + " warm recomputed_tokens not strictly < hit_tokens");
        if (mc.opts.linear_mode == PrefillLinearMode::W4A8) {
          require(wd.prefill_w4a8_tokens > 0ULL &&
                      wd.prefill_sage2_tokens > 0ULL,
                  mc.name + " warm blend did not compose W4A8 + Sage2");
        }
        const double cos_ba = cosine_similarity(cold_ba_logits, r_warm.logits);
        require(cos_ba >= 0.94,
                mc.name + " warm reordered CacheBlend cosine too low: " +
                    std::to_string(cos_ba));

        case6[mc.name] = {
            {"cos_sim_vs_cold", cos_ba},
            {"hit_tokens", wd.prefill_cache_blend_hit_tokens},
            {"recomputed_tokens", wd.prefill_cache_blend_recomputed_tokens},
            {"w4a8_tokens", wd.prefill_w4a8_tokens},
            {"sage2_tokens", wd.prefill_sage2_tokens},
        };
      }

      // End-to-end Engine::create with prefill_cache_blend = true (Tier-2 hit).
      auto eng = Engine::create(make_load_options(
          dense_oq, libs, static_cast<std::int32_t>(kCtx),
          PrefillLinearMode::W4A8, PrefillAttentionMode::Sage2, false, true,
          false, true));
      const auto shared_doc = make_tokens(512U, 123U, 240U);
      const auto q1 = make_tokens(128U, 201U, 240U);
      const auto q2 = make_tokens(128U, 202U, 240U);
      std::vector<U32> req1 = shared_doc;
      req1.insert(req1.end(), q1.begin(), q1.end());
      std::vector<U32> req2 = shared_doc;
      req2.insert(req2.end(), q2.begin(), q2.end());

      static_cast<void>(run_engine_request(*eng, req1, 2));
      const auto st_cold = eng->stats();
      require(st_cold.prefill_cache_blend &&
                  st_cold.prefill_cache_blend_hit_tokens == 0ULL,
              "Engine cold CacheBlend reported spurious hit");
      static_cast<void>(run_engine_request(*eng, req2, 2));
      const auto st_warm = eng->stats();
      require(st_warm.prefill_cache_blend_hit_tokens >= 512ULL &&
                  st_warm.prefill_cache_blend_recomputed_tokens <
                      st_warm.prefill_cache_blend_hit_tokens &&
                  st_warm.prefill_w4a8_tokens > st_cold.prefill_w4a8_tokens &&
                  st_warm.prefill_sage2_tokens > st_cold.prefill_sage2_tokens,
              "Engine warm CacheBlend Tier-2 telemetry mismatch");
      eng->shutdown();

      case6["engine_tier2"] = {
          {"hit_tokens", st_warm.prefill_cache_blend_hit_tokens},
          {"recomputed_tokens", st_warm.prefill_cache_blend_recomputed_tokens},
      };

      // Concurrent B=2 packed decode with CacheBlend + Dual-ANE enabled on Slot 0
      // and disabled on Slot 1+: shared_contract_identity must match across slots
      // so concurrent requests never fail with "Qwen packed batch slot differs in
      // execution contract".
      {
        auto b2_opts = make_load_options(
            dense_oq, libs, static_cast<std::int32_t>(kCtx),
            PrefillLinearMode::W4A8, PrefillAttentionMode::Sage2, false, true,
            true, true);
        b2_opts.max_concurrency = 2;
        auto eng_b2 = Engine::create(b2_opts);
        SubmitRequest sr1;
        sr1.prompt_tokens = to_i32(req1);
        sr1.max_new_tokens = 8;
        sr1.sampling.temperature = 0.0F;
        sr1.sampling.seed = 42ULL;
        SubmitRequest sr2 = sr1;
        sr2.prompt_tokens = to_i32(req2);
        const auto id1 = eng_b2->submit(sr1);
        const auto id2 = eng_b2->submit(sr2);
        std::size_t c1 = 0, c2 = 0;
        bool d1 = false, d2 = false;
        const auto deadline = std::chrono::steady_clock::now() + 180s;
        while ((!d1 || !d2) && std::chrono::steady_clock::now() < deadline) {
          for (const auto &ev : eng_b2->poll_events(10ms)) {
            if (ev.kind == EventKind::tokens) {
              if (ev.request_id == id1) c1 += ev.tokens.size();
              else if (ev.request_id == id2) c2 += ev.tokens.size();
            } else if (ev.kind == EventKind::finished) {
              if (ev.request_id == id1) d1 = true;
              else if (ev.request_id == id2) d2 = true;
            } else {
              throw std::runtime_error("Concurrent B=2 CacheBlend+ANE failed: " + ev.error);
            }
          }
        }
        require(d1 && d2 && c1 == 8U && c2 == 8U,
                "Concurrent B=2 CacheBlend+ANE did not finish cleanly");
        eng_b2->shutdown();
        case6["concurrent_b2_cache_blend_ane"] = "passed";
      }

      // Multi-window CacheBlend (suffix_len = 3200 > workspace_rows_ = 2048):
      // 3 x 1024-tok docs + 128-tok query, cold populate -> warm reordered [C][A][B][Q]!
      {
        constexpr U32 kMultiCtx = 4096U;
        auto ld4k = LoadedDirect::open(dense_oq, kMultiCtx);
        const auto mw_a = make_doc(1024U, 131U);
        const auto mw_b = make_doc(1024U, 149U);
        const auto mw_c = make_doc(1024U, 167U);
        const auto mw_q = make_tokens(128U, 181U, 240U);

        std::vector<U32> p_abc;
        p_abc.insert(p_abc.end(), mw_a.begin(), mw_a.end());
        p_abc.insert(p_abc.end(), mw_b.begin(), mw_b.end());
        p_abc.insert(p_abc.end(), mw_c.begin(), mw_c.end());
        p_abc.insert(p_abc.end(), mw_q.begin(), mw_q.end());

        std::vector<U32> p_cab;
        p_cab.insert(p_cab.end(), mw_c.begin(), mw_c.end());
        p_cab.insert(p_cab.end(), mw_a.begin(), mw_a.end());
        p_cab.insert(p_cab.end(), mw_b.begin(), mw_b.end());
        p_cab.insert(p_cab.end(), mw_q.begin(), mw_q.end());

        QwenPrefillOptions mw_opts{
            .linear_mode = PrefillLinearMode::W4A8,
            .attention_mode = PrefillAttentionMode::Sage2,
            .sparse_minference = false,
            .cache_blend = true,
            .ane_offload = false};
        auto fwd_mw = make_forward(ld4k, dense_oq, libs, kMultiCtx, mw_opts);
        require(fwd_mw->memory_plan().workspace_rows == 2048U &&
                    p_cab.size() > fwd_mw->memory_plan().workspace_rows,
                "multi-window CacheBlend test did not exceed workspace_rows");
        const auto cold_cab_logits =
            run_prefill_and_last_logits(*fwd_mw, p_cab);

        detail::ChunkBlendCache mw_cache(1ULL << 30);
        const U64 mw_hash = detail::compute_blend_model_exec_hash(
            ld4k.digest, "affine4_g128", "w4a8", "sage2", false, "off");
        GpuSamplingOptions mw_samp{};
        mw_samp.absolute_position = p_abc.size() - 1U;
        mw_samp.contract = SamplingContract::PositionCoupledV1;

        fwd_mw->reset();
        const auto r_mw_cold = fwd_mw->run_blend_prefill(
            p_abc, 0U, mw_cache, mw_hash, mw_samp, true, &seg_cfg);
        require(mw_cache.size() >= 3U,
                "multi-window cold blend did not capture all 3 x 1024-tok docs");
        require(r_mw_cold.prefill_delta.prefill_cache_blend_hit_tokens == 0ULL,
                "multi-window cold blend unexpectedly reported hits");

        fwd_mw->reset();
        const auto r_mw_warm = fwd_mw->run_blend_prefill(
            p_cab, 0U, mw_cache, mw_hash, mw_samp, true, &seg_cfg);
        const auto &mwd = r_mw_warm.prefill_delta;
        require(mwd.prefill_cache_blend_hit_tokens == 3072ULL,
                "multi-window warm hit_tokens != 3072 (got " +
                    std::to_string(mwd.prefill_cache_blend_hit_tokens) + ")");
        require(mwd.prefill_cache_blend_recomputed_tokens > 0ULL &&
                    mwd.prefill_cache_blend_recomputed_tokens < 3072ULL,
                "multi-window warm recomputed_tokens not strictly < 3072");
        require(mwd.prefill_w4a8_tokens > 0ULL &&
                    mwd.prefill_sage2_tokens > 0ULL,
                "multi-window warm blend did not compose W4A8 + Sage2");
        const double cos_cab =
            cosine_similarity(cold_cab_logits, r_mw_warm.logits);
        require(cos_cab >= 0.94,
                "multi-window warm reordered CacheBlend cosine too low: " +
                    std::to_string(cos_cab));
        case6["multi_window_3200"] = {
            {"cos_sim_vs_cold", cos_cab},
            {"hit_tokens", mwd.prefill_cache_blend_hit_tokens},
            {"recomputed_tokens", mwd.prefill_cache_blend_recomputed_tokens},
            {"w4a8_tokens", mwd.prefill_w4a8_tokens},
            {"sage2_tokens", mwd.prefill_sage2_tokens},
        };

        // Stage-2 Dual-ANE + CacheBlend + W4A8 + Sage2 simultaneous composition
        // (M_active >= 1024 inside a single warm window: 1024-tok HIT + 960-tok dense tail).
        QwenPrefillOptions all5_opts{
            .linear_mode = PrefillLinearMode::W4A8,
            .attention_mode = PrefillAttentionMode::Sage2,
            .sparse_minference = false,
            .cache_blend = true,
            .ane_offload = true};
        auto fwd_all5 = make_forward(ld4k, dense_oq, libs, kMultiCtx, all5_opts);
        const auto prime_tail = make_tokens(128U, 211U, 240U);
        const auto large_tail = make_tokens(960U, 223U, 240U);
        std::vector<U32> p_prime = mw_a;
        p_prime.insert(p_prime.end(), prime_tail.begin(), prime_tail.end());
        std::vector<U32> p_warm_ane = mw_a;
        p_warm_ane.insert(p_warm_ane.end(), large_tail.begin(),
                          large_tail.end());

        const auto cold_ane_blend_logits =
            run_prefill_and_last_logits(*fwd_all5, p_warm_ane);

        detail::ChunkBlendCache ane_cache(1ULL << 30);
        const U64 all5_hash = detail::compute_blend_model_exec_hash(
            ld4k.digest, "affine4_g128", "w4a8", "sage2", false, "auto");
        detail::BlendSegmentConfig tail960_cfg = seg_cfg;
        tail960_cfg.dense_tail_tokens = 960U;

        GpuSamplingOptions s_prime{};
        s_prime.absolute_position = p_prime.size() - 1U;
        s_prime.contract = SamplingContract::PositionCoupledV1;
        fwd_all5->reset();
        static_cast<void>(fwd_all5->run_blend_prefill(
            p_prime, 0U, ane_cache, all5_hash, s_prime, false, &seg_cfg));
        require(ane_cache.size() >= 1U,
                "Stage-2 ANE+CacheBlend prime did not insert Doc A");

        GpuSamplingOptions s_warm_ane{};
        s_warm_ane.absolute_position = p_warm_ane.size() - 1U;
        s_warm_ane.contract = SamplingContract::PositionCoupledV1;
        fwd_all5->reset();
        const auto r_all5 = fwd_all5->run_blend_prefill(
            p_warm_ane, 0U, ane_cache, all5_hash, s_warm_ane, true,
            &tail960_cfg);
        const auto &ad = r_all5.prefill_delta;
        require(ad.prefill_cache_blend_hit_tokens == 1024ULL &&
                    ad.prefill_cache_blend_recomputed_tokens < 1024ULL &&
                    ad.prefill_w4a8_tokens >= 1024ULL &&
                    ad.prefill_sage2_tokens >= 1024ULL,
                "Stage-2 ANE+CacheBlend+W4A8+Sage2 telemetry mismatch");
        if (ane::is_ane_available() && fwd_all5->ane_offload_active()) {
          require(ad.prefill_ane_offload_tokens >= 1024ULL,
                  "Stage-2 CacheBlend did not compose Dual-ANE when M_active >= 1024 (got " +
                      std::to_string(ad.prefill_ane_offload_tokens) + ")");
        }
        const double cos_all5 =
            cosine_similarity(cold_ane_blend_logits, r_all5.logits);
        require(cos_all5 >= 0.94,
                "Stage-2 ANE+CacheBlend+W4A8+Sage2 cosine too low: " +
                    std::to_string(cos_all5));
        case6["stage2_ane_w4a8_sage2_blend"] = {
            {"cos_sim_vs_cold", cos_all5},
            {"hit_tokens", ad.prefill_cache_blend_hit_tokens},
            {"recomputed_tokens", ad.prefill_cache_blend_recomputed_tokens},
            {"ane_offload_tokens", ad.prefill_ane_offload_tokens},
            {"w4a8_tokens", ad.prefill_w4a8_tokens},
            {"sage2_tokens", ad.prefill_sage2_tokens},
        };
      }

      report["case6_cache_blend"] = case6;
    }

    // -------------------------------------------------------------------------
    // Case 7: Safe Auto-Fallback when requesting W4A4 on non-MXFP4 (oQ4e) model
    // -------------------------------------------------------------------------
    {
      auto fb_opts = make_load_options(
          dense_oq, libs, 1024, PrefillLinearMode::W4A4,
          PrefillAttentionMode::ExactNax, false, false, false, true);
      auto eng_fb = Engine::create(fb_opts);
      const auto prompt_fb = make_tokens(64U, 307U, 240U);
      const auto toks = run_engine_request(*eng_fb, prompt_fb, 4);
      const auto st_fb = eng_fb->stats();
      eng_fb->shutdown();

      require(toks.size() == 4U,
              "Case 7 W4A4->W4A8 fallback engine did not generate tokens");
      require(st_fb.prefill_linear_mode == "w4a4",
              "Case 7 requested prefill_linear_mode must stay w4a4");
      require(st_fb.prefill_linear_fallback,
              "Case 7 must report prefill_linear_fallback=true on oQ4e");
      require(st_fb.prefill_linear_fallback_reason == "requires_mxfp4_model",
              "Case 7 fallback reason mismatch: " +
                  st_fb.prefill_linear_fallback_reason);
      require(st_fb.model_quant_format == "oq_affine",
              "Case 7 model_quant_format mismatch: " + st_fb.model_quant_format);
      require(st_fb.prefill_linear_mode_effective == "w4a8" ||
                  st_fb.prefill_linear_mode_effective == "bf16",
              "Case 7 effective mode must fall back to w4a8 or bf16, got: " +
                  st_fb.prefill_linear_mode_effective);
      report["case7_w4a4_non_mxfp4_fallback"] = {
          {"requested", st_fb.prefill_linear_mode},
          {"effective", st_fb.prefill_linear_mode_effective},
          {"fallback", st_fb.prefill_linear_fallback},
          {"reason", st_fb.prefill_linear_fallback_reason},
          {"model_quant_format", st_fb.model_quant_format},
      };
    }

    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
