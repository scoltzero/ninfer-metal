#include "omlx_m5/artifact.h"
#include "omlx_m5/engine.h"
#include "omlx_m5/execution_contract.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/qwen_forward.h"
#include "omlx_m5/topology_profile.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using namespace omlx::m5;
using namespace std::chrono_literals;
using Json = nlohmann::json;
using U32 = std::uint32_t;
using U64 = std::uint64_t;

void require(bool cond, std::string_view msg) {
  if (!cond) {
    throw std::runtime_error("UNIVERSAL-CAPABILITY FAIL: " + std::string(msg));
  }
}

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

float bf16_to_f32(std::uint16_t b) {
  return std::bit_cast<float>(static_cast<U32>(b) << 16U);
}

std::uint16_t f32_to_f16(float v) {
  const U32 bits = std::bit_cast<U32>(v);
  const U32 sign = (bits >> 16U) & 0x8000U;
  const std::int32_t exp =
      static_cast<std::int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
  U32 mant = bits & 0x007FFFFFU;
  if (exp <= 0) {
    return static_cast<std::uint16_t>(sign);
  }
  if (exp >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  const U32 round_bit = (mant >> 12U) & 1U;
  const U32 sticky = (mant & 0xFFFU) != 0U ? 1U : 0U;
  const U32 lsb = (mant >> 13U) & 1U;
  mant = (mant >> 13U) + (round_bit & (sticky | lsb));
  U32 out_exp = static_cast<U32>(exp);
  if (mant >= 0x400U) {
    mant = 0U;
    ++out_exp;
    if (out_exp >= 31U) {
      return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
  }
  return static_cast<std::uint16_t>(sign | (out_exp << 10U) | (mant & 0x3FFU));
}

float f16_to_f32(std::uint16_t h) {
  const U32 sign = (static_cast<U32>(h) & 0x8000U) << 16U;
  const U32 exp = (static_cast<U32>(h) >> 10U) & 0x1FU;
  const U32 mant = static_cast<U32>(h) & 0x03FFU;
  if (exp == 0U) {
    if (mant == 0U) {
      return std::bit_cast<float>(sign);
    }
    float m = static_cast<float>(mant) * (1.0F / 1024.0F);
    float val = std::ldexp(m, -14);
    return (sign != 0U) ? -val : val;
  }
  if (exp == 31U) {
    return std::bit_cast<float>(sign | 0x7F800000U | (mant << 13U));
  }
  const U32 bits = sign | ((exp + 112U) << 23U) | (mant << 13U);
  return std::bit_cast<float>(bits);
}

float decode_fp4_e2m1(U32 code) {
  const float mag =
      0.5F * static_cast<float>((0xC8643210U >> ((code & 7U) * 4U)) & 15U);
  return (code & 8U) == 0U ? mag : -mag;
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

double cosine_similarity(std::span<const float> a, std::span<const double> b) {
  require(a.size() == b.size() && !a.empty(), "cosine_similarity size mismatch");
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    require(std::isfinite(a[i]) && std::isfinite(b[i]),
            "non-finite value in logit comparison");
    dot += double(a[i]) * b[i];
    na += double(a[i]) * double(a[i]);
    nb += b[i] * b[i];
  }
  const double denom = std::sqrt(na) * std::sqrt(nb);
  return denom > 0.0 ? (dot / denom) : 0.0;
}

double max_abs_diff(std::span<const float> a, std::span<const double> b) {
  require(a.size() == b.size(), "max_abs_diff size mismatch");
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(double(a[i]) - b[i]));
  }
  return m;
}

enum class WeightEnc {
  DirectBf16,
  DirectF16,
  AffineQ4G64Tiled,
  AffineQ8G64Tiled,
  AffineQ4G64RowSplit,
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

struct AliasBinding {
  std::string name;
  U32 object_id = 0;
  std::string format;
  U64 out_features = 0;
  U64 in_features = 0;
  std::string use;
  std::string activation_policy = "A16Only";
};

class UniversalArtifactBuilder {
public:
  U32 add_norm_1d(std::string name, U32 width, U32 seed, bool as_f16,
                  std::vector<double> *host_out) {
    host_out->resize(width);
    BuiltObject obj{
        .id = static_cast<U32>(objects_.size()),
        .name = std::move(name),
        .format = as_f16 ? "direct_f16_v1" : "direct_bf16_v1",
        .layout = "contiguous_le_v1",
        .shape = {width},
        .activation_policy = "A16Only",
    };
    std::vector<std::uint16_t> words(width);
    for (U32 i = 0; i < width; ++i) {
      const float raw = 1.0F + det_val(seed, i, 0.08F);
      if (as_f16) {
        words[i] = f32_to_f16(raw);
        (*host_out)[i] = double(f16_to_f32(words[i]));
      } else {
        words[i] = f32_to_bf16(raw);
        (*host_out)[i] = double(bf16_to_f32(words[i]));
      }
    }
    const auto bytes = std::as_bytes(std::span<const std::uint16_t>(words));
    obj.planes.push_back({"data", {bytes.begin(), bytes.end()}});
    objects_.push_back(std::move(obj));
    return objects_.back().id;
  }

  U32 add_matrix_2d(std::string name, U32 n, U32 k, WeightEnc enc, U32 seed,
                    std::string binding_use, std::vector<double> *host_out,
                    float amp = 0.05F) {
    host_out->assign(std::size_t(n) * k, 0.0);
    const U32 id = static_cast<U32>(objects_.size());
    if (enc == WeightEnc::DirectBf16 || enc == WeightEnc::DirectF16) {
      const bool is_f16 = (enc == WeightEnc::DirectF16);
      std::vector<std::uint16_t> words(std::size_t(n) * k);
      for (std::size_t i = 0; i < words.size(); ++i) {
        const float raw = det_val(seed, i, amp);
        if (is_f16) {
          words[i] = f32_to_f16(raw);
          (*host_out)[i] = double(f16_to_f32(words[i]));
        } else {
          words[i] = f32_to_bf16(raw);
          (*host_out)[i] = double(bf16_to_f32(words[i]));
        }
      }
      const auto bytes = std::as_bytes(std::span<const std::uint16_t>(words));
      objects_.push_back(BuiltObject{
          .id = id,
          .name = std::move(name),
          .format = is_f16 ? "direct_f16_v1" : "direct_bf16_v1",
          .layout = "contiguous_le_v1",
          .shape = {n, k},
          .planes = {{"data", {bytes.begin(), bytes.end()}}},
          .binding_use = std::move(binding_use),
          .activation_policy = "A16Only",
      });
      return id;
    }

    const U32 padded_k = ((k + 127U) / 128U) * 128U;
    BuiltObject obj{
        .id = id,
        .name = std::move(name),
        .shape = {n, k},
        .padded_k = padded_k,
        .binding_use = std::move(binding_use),
        .activation_policy = "A16Only",
    };

    if ((enc == WeightEnc::AffineQ4G64Tiled ||
         enc == WeightEnc::AffineQ8G64Tiled) &&
        k >= 512U && (k % 64U) == 0U) {
      const U32 bits = (enc == WeightEnc::AffineQ4G64Tiled) ? 4U : 8U;
      const U32 max_q = (1U << bits) - 1U;
      obj.format = "oq_affine_q" + std::to_string(bits) + "_g64_bf16_v1";
      obj.layout = "oq_nax_tiled_g64_v1";
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
        const U32 tn = r / bn;
        const U32 rn = r % bn;
        for (U32 g64 = 0; g64 < groups64; ++g64) {
          const float span =
              2.0F * amp * (0.6F + 0.1F * float((r + g64 + seed) % 5U));
          const std::uint16_t sc_bf = f32_to_bf16(span / float(max_q));
          const std::uint16_t bi_bf = f32_to_bf16(-0.5F * span);
          const double sc = double(bf16_to_f32(sc_bf));
          const double bi = double(bf16_to_f32(bi_bf));
          const U32 packed_sb =
              static_cast<U32>(sc_bf) | (static_cast<U32>(bi_bf) << 16U);
          std::memcpy(nax_sb.data() + (std::size_t(g64) * padded_n + r) * 4U,
                      &packed_sb, 4U);
          const std::size_t tile_idx = std::size_t(tn) * groups64 + g64;
          for (U32 off = 0; off < 64U; ++off) {
            const U32 col = g64 * 64U + off;
            const U32 q =
                static_cast<U32>(det_hash(seed, U64(r) * U64(k) + U64(col))) &
                max_q;
            (*host_out)[std::size_t(r) * k + col] = double(q) * sc + bi;
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
      }
      obj.planes.push_back({"nax_codes", std::move(nax_codes)});
      obj.planes.push_back({"nax_sb", std::move(nax_sb)});
    } else if (enc == WeightEnc::AffineQ4G64RowSplit ||
               enc == WeightEnc::AffineQ4G64Tiled ||
               enc == WeightEnc::AffineQ8G64Tiled) {
      const U32 bits = (enc == WeightEnc::AffineQ8G64Tiled) ? 8U : 4U;
      const U32 max_q = (1U << bits) - 1U;
      obj.format = "oq_affine_q" + std::to_string(bits) + "_g64_bf16_v1";
      obj.layout = "oq_rowsplit_k128_v1";
      const U32 gs = 64U;
      const U32 padded_groups = padded_k / gs;
      std::vector<std::byte> code(
          bits == 8U ? (std::size_t(n) * padded_k)
                     : (std::size_t(n) * (padded_k / 2U)),
          std::byte{0});
      std::vector<std::uint16_t> scale(std::size_t(n) * padded_groups, 0U);
      std::vector<std::uint16_t> bias(std::size_t(n) * padded_groups, 0U);
      for (U32 r = 0; r < n; ++r) {
        for (U32 sg = 0; sg < k / gs; ++sg) {
          const float span =
              2.0F * amp * (0.6F + 0.1F * float((r + sg + seed) % 5U));
          const std::uint16_t sc_bf = f32_to_bf16(span / float(max_q));
          const std::uint16_t bi_bf = f32_to_bf16(-0.5F * span);
          scale[std::size_t(r) * padded_groups + sg] = sc_bf;
          bias[std::size_t(r) * padded_groups + sg] = bi_bf;
          const double sc = double(bf16_to_f32(sc_bf));
          const double bi = double(bf16_to_f32(bi_bf));
          if (bits == 8U) {
            for (U32 off = 0; off < gs; ++off) {
              const U32 c0 = sg * gs + off;
              const U32 q0 =
                  static_cast<U32>(det_hash(seed, U64(r) * k + c0)) & 0xFFU;
              code[std::size_t(r) * padded_k + c0] = static_cast<std::byte>(q0);
              (*host_out)[std::size_t(r) * k + c0] = double(q0) * sc + bi;
            }
          } else {
            for (U32 off = 0; off < gs; off += 2U) {
              const U32 c0 = sg * gs + off;
              const U32 q0 =
                  static_cast<U32>(det_hash(seed, U64(r) * k + c0)) & 0xFU;
              const U32 q1 =
                  static_cast<U32>(det_hash(seed, U64(r) * k + c0 + 1U)) & 0xFU;
              code[std::size_t(r) * (padded_k / 2U) + (c0 >> 1U)] =
                  static_cast<std::byte>(q0 | (q1 << 4U));
              (*host_out)[std::size_t(r) * k + c0] = double(q0) * sc + bi;
              (*host_out)[std::size_t(r) * k + c0 + 1U] = double(q1) * sc + bi;
            }
          }
        }
      }
      const auto s_raw = std::as_bytes(std::span<const std::uint16_t>(scale));
      const auto b_raw = std::as_bytes(std::span<const std::uint16_t>(bias));
      obj.planes.push_back(
          {bits == 8U ? "code" : "base_code", std::move(code)});
      obj.planes.push_back({"scale", {s_raw.begin(), s_raw.end()}});
      obj.planes.push_back({"bias", {b_raw.begin(), b_raw.end()}});
    } else {
      throw std::invalid_argument("unsupported 2D encoding");
    }

    objects_.push_back(std::move(obj));
    return id;
  }

  U32 add_moe_mxfp4_3d(std::string name, U32 num_experts, U32 n, U32 k,
                       U32 seed, std::vector<double> *host_out) {
    const U32 id = static_cast<U32>(objects_.size());
    const U32 padded_k = ((k + 127U) / 128U) * 128U;
    const std::size_t rows = std::size_t(num_experts) * n;
    host_out->assign(rows * k, 0.0);
    std::vector<std::byte> data(rows * (padded_k / 2U), std::byte{0});
    std::vector<std::byte> scale(rows * (padded_k / 32U), std::byte{118});
    for (U32 e = 0; e < num_experts; ++e) {
      for (U32 r = 0; r < n; ++r) {
        const std::size_t flat_r = std::size_t(e) * n + r;
        for (U32 g = 0; g < k / 32U; ++g) {
          const std::uint8_t sc_code = static_cast<std::uint8_t>(
              119U + (det_hash(seed + 1U, flat_r * 256ULL + g) % 3U));
          scale[flat_r * (padded_k / 32U) + g] = static_cast<std::byte>(sc_code);
          const double sc = std::ldexp(1.0, static_cast<int>(sc_code) - 127);
          for (U32 j = 0; j < 16U; ++j) {
            const U32 c0 = g * 32U + 2U * j;
            const U64 h0 = det_hash(seed, flat_r * U64(k) + U64(c0));
            const U32 q0 = static_cast<U32>(h0 & 0xFU);
            const U32 q1 = static_cast<U32>((h0 >> 8U) & 0xFU);
            data[flat_r * (padded_k / 2U) + g * 16U + j] =
                static_cast<std::byte>(q0 | (q1 << 4U));
            (*host_out)[flat_r * k + c0] = double(decode_fp4_e2m1(q0)) * sc;
            (*host_out)[flat_r * k + c0 + 1U] =
                double(decode_fp4_e2m1(q1)) * sc;
          }
        }
      }
    }
    objects_.push_back(BuiltObject{
        .id = id,
        .name = std::move(name),
        .format = "mxfp4_e2m1_g32_e8m0_v1",
        .layout = "mx_expert_rowsplit_k128_v1",
        .shape = {num_experts, n, k},
        .padded_k = padded_k,
        .planes = {{"data", std::move(data)}, {"scale", std::move(scale)}},
        .binding_use = "moe",
        .activation_policy = "AllowA4",
    });
    return id;
  }

  void add_alias_binding(AliasBinding b) {
    alias_bindings_.push_back(std::move(b));
  }

  std::vector<std::byte> serialize(std::string_view source_profile,
                                   std::string_view source_repo = "test/universal",
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

    for (const auto &alias : alias_bindings_) {
      j_bindings.push_back({
          {"activation_policy", alias.activation_policy},
          {"format", alias.format},
          {"in_features", alias.in_features},
          {"leading_shape", Json::array()},
          {"name", alias.name},
          {"object_id", alias.object_id},
          {"out_features", alias.out_features},
          {"use", alias.use},
      });
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
  std::vector<AliasBinding> alias_bindings_;
};

// ============================================================================
// Independent FP64 CPU Reference Oracle for Generalized Pure-GQA & MoE Models
// ============================================================================

struct CpuLayerWeights {
  std::vector<double> input_norm;
  std::vector<double> q_proj; // [q_out, h]
  std::vector<double> k_proj; // [kv, h]
  std::vector<double> v_proj; // [kv, h]
  std::vector<double> o_proj; // [h, q]
  std::vector<double> q_norm; // [hd] if has_qk_norm
  std::vector<double> k_norm; // [hd] if has_qk_norm
  std::vector<double> post_norm;

  // Dense MLP
  std::vector<double> mlp_gate; // [f, h]
  std::vector<double> mlp_up;   // [f, h]
  std::vector<double> mlp_down; // [h, f]

  // MoE MLP
  std::vector<double> router;      // [E, h]
  std::vector<double> exp_gate;    // [E, mf, h]
  std::vector<double> exp_up;      // [E, mf, h]
  std::vector<double> exp_down;    // [E, h, mf]
  std::vector<double> shared_gate; // [sf, h]
  std::vector<double> shared_up;   // [sf, h]
  std::vector<double> shared_down; // [h, sf]
  std::vector<double> shared_expert_gate; // [1, h]
};

struct CpuModelOracle {
  QwenModelConfig cfg;
  double norm_offset = 0.0;
  std::vector<double> embedding; // [v, h]
  std::vector<double> final_norm; // [h]
  std::vector<double> lm_head;   // [v, h]
  std::vector<CpuLayerWeights> layers;

  static void matvec(std::span<const double> w, U32 n, U32 k,
                     std::span<const double> x, std::span<double> out) {
    for (U32 r = 0; r < n; ++r) {
      double acc = 0.0;
      const double *row = w.data() + std::size_t(r) * k;
      for (U32 c = 0; c < k; ++c) {
        acc += row[c] * x[c];
      }
      out[r] = acc;
    }
  }

  void rmsnorm(std::span<const double> x, std::span<const double> w,
               std::span<double> out) const {
    const std::size_t n = x.size();
    double sq = 0.0;
    for (double v : x) {
      sq += v * v;
    }
    const double inv = 1.0 / std::sqrt(sq / double(n) + cfg.rms_norm_eps);
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = x[i] * inv * (w[i] + norm_offset);
    }
  }

  static double silu(double x) { return x / (1.0 + std::exp(-x)); }
  static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

  struct OracleTrace {
    std::vector<double> logits; // [T, V]
    std::vector<std::vector<U32>> topk_ids; // [layer][T * top_k]
    std::vector<std::vector<double>> topk_scores;
  };

  OracleTrace forward(std::span<const U32> tokens,
                      bool quantize_kv_affine4 = false) const {
    const U32 T = static_cast<U32>(tokens.size());
    const U32 h = static_cast<U32>(cfg.hidden_size);
    const U32 v = static_cast<U32>(cfg.vocab_size);
    const U32 qh = static_cast<U32>(cfg.num_attention_heads);
    const U32 kvh = static_cast<U32>(cfg.num_key_value_heads);
    const U32 hd = static_cast<U32>(cfg.head_dim);
    const U32 q_width = qh * hd;
    const U32 q_out = cfg.has_attn_output_gate ? (2U * q_width) : q_width;
    const U32 kv_width = kvh * hd;
    const U32 rot_dim =
        static_cast<U32>(std::llround(double(hd) * cfg.partial_rotary_factor));
    const U32 half_rot = rot_dim / 2U;

    std::vector<double> hidden(std::size_t(T) * h, 0.0);
    for (U32 t = 0; t < T; ++t) {
      std::copy_n(embedding.data() + std::size_t(tokens[t]) * h, h,
                  hidden.data() + std::size_t(t) * h);
    }

    OracleTrace trace;
    trace.topk_ids.resize(layers.size());
    trace.topk_scores.resize(layers.size());

    std::vector<double> norm_h(h);
    for (std::size_t li = 0; li < layers.size(); ++li) {
      const auto &lw = layers[li];
      std::vector<double> q_all(std::size_t(T) * q_out, 0.0);
      std::vector<double> k_all(std::size_t(T) * kv_width, 0.0);
      std::vector<double> v_all(std::size_t(T) * kv_width, 0.0);

      for (U32 t = 0; t < T; ++t) {
        rmsnorm(std::span<const double>(hidden.data() + std::size_t(t) * h, h),
                lw.input_norm, norm_h);
        matvec(lw.q_proj, q_out, h, norm_h,
               std::span<double>(q_all.data() + std::size_t(t) * q_out, q_out));
        matvec(lw.k_proj, kv_width, h, norm_h,
               std::span<double>(k_all.data() + std::size_t(t) * kv_width,
                                 kv_width));
        matvec(lw.v_proj, kv_width, h, norm_h,
               std::span<double>(v_all.data() + std::size_t(t) * kv_width,
                                 kv_width));

        const auto apply_qkn_rope = [&](double *head_ptr,
                                        const std::vector<double> &nw) {
          double inv_rms = 1.0;
          if (cfg.has_qk_norm) {
            double sq = 0.0;
            for (U32 d = 0; d < hd; ++d) {
              sq += head_ptr[d] * head_ptr[d];
            }
            inv_rms = 1.0 / std::sqrt(sq / double(hd) + cfg.rms_norm_eps);
            for (U32 d = 0; d < hd; ++d) {
              head_ptr[d] *= inv_rms * (nw[d] + norm_offset);
            }
          }
          for (U32 p = 0; p < half_rot; ++p) {
            const double x0 = head_ptr[p];
            const double x1 = head_ptr[p + half_rot];
            const double exp_f = double(2U * p) / double(rot_dim);
            const double angle = double(t) / std::pow(cfg.rope_theta, exp_f);
            const double cs = std::cos(angle);
            const double sn = std::sin(angle);
            head_ptr[p] = x0 * cs - x1 * sn;
            head_ptr[p + half_rot] = x0 * sn + x1 * cs;
          }
          for (U32 d = 0; d < hd; ++d) {
            head_ptr[d] = static_cast<double>(
                bf16_to_f32(f32_to_bf16(static_cast<float>(head_ptr[d]))));
          }
        };

        const U32 q_stride = cfg.has_attn_output_gate ? (2U * hd) : hd;
        for (U32 hi = 0; hi < qh; ++hi) {
          apply_qkn_rope(q_all.data() + std::size_t(t) * q_out + hi * q_stride,
                         lw.q_norm);
        }
        for (U32 kh = 0; kh < kvh; ++kh) {
          apply_qkn_rope(k_all.data() + std::size_t(t) * kv_width + kh * hd,
                         lw.k_norm);
        }
        if (quantize_kv_affine4) {
          const auto sim_a4 = [&](double *head_ptr) {
            const U32 gs = std::min<U32>(hd, 128U);
            const U32 groups = hd / gs;
            for (U32 g = 0; g < groups; ++g) {
              float gmin = 1e30F;
              float gmax = -1e30F;
              std::vector<float> vf(gs);
              for (U32 i = 0; i < gs; ++i) {
                vf[i] = bf16_to_f32(
                    f32_to_bf16(static_cast<float>(head_ptr[g * gs + i])));
                gmin = std::min(gmin, vf[i]);
                gmax = std::max(gmax, vf[i]);
              }
              const float z_f = f16_to_f32(f32_to_f16(gmin));
              const float span = std::max(0.0F, gmax - z_f);
              const float s_f = f16_to_f32(f32_to_f16(span / 15.0F));
              const float inv_s = (s_f > 0.0F) ? (1.0F / s_f) : 0.0F;
              for (U32 i = 0; i < gs; ++i) {
                const float q = std::clamp(
                    std::rint((vf[i] - z_f) * inv_s), 0.0F, 15.0F);
                head_ptr[g * gs + i] = static_cast<double>(q * s_f + z_f);
              }
            }
          };
          for (U32 kh = 0; kh < kvh; ++kh) {
            sim_a4(k_all.data() + std::size_t(t) * kv_width + kh * hd);
            sim_a4(v_all.data() + std::size_t(t) * kv_width + kh * hd);
          }
        }
      }

      // Causal GQA attention
      const double scale = 1.0 / std::sqrt(double(hd));
      const U32 repeat = qh / kvh;
      const U32 q_stride = cfg.has_attn_output_gate ? (2U * hd) : hd;
      std::vector<double> mixer(q_width, 0.0);
      std::vector<double> delta(h, 0.0);
      for (U32 t = 0; t < T; ++t) {
        std::fill(mixer.begin(), mixer.end(), 0.0);
        for (U32 hi = 0; hi < qh; ++hi) {
          const U32 kh = hi / repeat;
          const double *q_vec =
              q_all.data() + std::size_t(t) * q_out + hi * q_stride;
          std::vector<double> scores(t + 1U, 0.0);
          double max_s = -1e30;
          for (U32 s = 0; s <= t; ++s) {
            const double *k_vec =
                k_all.data() + std::size_t(s) * kv_width + kh * hd;
            double dot = 0.0;
            for (U32 d = 0; d < hd; ++d) {
              dot += q_vec[d] * k_vec[d];
            }
            scores[s] = dot * scale;
            max_s = std::max(max_s, scores[s]);
          }
          double sum_e = 0.0;
          for (U32 s = 0; s <= t; ++s) {
            scores[s] = std::exp(scores[s] - max_s);
            sum_e += scores[s];
          }
          for (U32 d = 0; d < hd; ++d) {
            double acc = 0.0;
            for (U32 s = 0; s <= t; ++s) {
              const double *v_vec =
                  v_all.data() + std::size_t(s) * kv_width + kh * hd;
              acc += (scores[s] / sum_e) * v_vec[d];
            }
            if (cfg.has_attn_output_gate) {
              const double g = q_vec[hd + d];
              acc *= sigmoid(g);
            }
            mixer[hi * hd + d] = acc;
          }
        }
        matvec(lw.o_proj, h, q_width, mixer, delta);
        for (U32 d = 0; d < h; ++d) {
          hidden[std::size_t(t) * h + d] += delta[d];
        }
      }

      // MLP (Dense or Generalized MoE)
      for (U32 t = 0; t < T; ++t) {
        rmsnorm(std::span<const double>(hidden.data() + std::size_t(t) * h, h),
                lw.post_norm, norm_h);
        if (cfg.num_experts == 0) {
          const U32 f = static_cast<U32>(cfg.intermediate_size);
          std::vector<double> g_buf(f), u_buf(f), prod(f);
          matvec(lw.mlp_gate, f, h, norm_h, g_buf);
          matvec(lw.mlp_up, f, h, norm_h, u_buf);
          for (U32 i = 0; i < f; ++i) {
            prod[i] = silu(g_buf[i]) * u_buf[i];
          }
          matvec(lw.mlp_down, h, f, prod, delta);
        } else {
          const U32 E = static_cast<U32>(cfg.num_experts);
          const U32 K = static_cast<U32>(cfg.num_experts_per_token);
          const U32 mf = static_cast<U32>(cfg.intermediate_size);
          std::vector<double> r_logits(E);
          matvec(lw.router, E, h, norm_h, r_logits);
          for (U32 e = 0; e < E; ++e) {
            r_logits[e] = static_cast<double>(
                bf16_to_f32(f32_to_bf16(static_cast<float>(r_logits[e]))));
          }
          std::vector<bool> used(E, false);
          std::vector<U32> sel_ids(K, 0U);
          std::vector<double> sel_vals(K, 0.0);
          for (U32 ki = 0; ki < K; ++ki) {
            double best_v = -1e300;
            U32 best_e = 0U;
            for (U32 e = 0; e < E; ++e) {
              if (!used[e] && r_logits[e] >= best_v) {
                best_v = r_logits[e];
                best_e = e;
              }
            }
            used[best_e] = true;
            sel_ids[ki] = best_e;
            sel_vals[ki] = best_v;
          }
          std::vector<double> weights(K, 0.0);
          {
            const double mx = sel_vals[0];
            double denom = 0.0;
            for (U32 ki = 0; ki < K; ++ki) {
              weights[ki] = std::exp(sel_vals[ki] - mx);
              denom += weights[ki];
            }
            for (U32 ki = 0; ki < K; ++ki) {
              weights[ki] /= denom;
            }
          }
          for (U32 ki = 0; ki < K; ++ki) {
            trace.topk_ids[li].push_back(sel_ids[ki]);
            trace.topk_scores[li].push_back(weights[ki]);
          }

          std::fill(delta.begin(), delta.end(), 0.0);
          std::vector<double> eg(mf), eu(mf), eprod(mf), edown(h);
          for (U32 ki = 0; ki < K; ++ki) {
            const U32 eid = sel_ids[ki];
            matvec(std::span<const double>(
                       lw.exp_gate.data() + std::size_t(eid) * mf * h,
                       std::size_t(mf) * h),
                   mf, h, norm_h, eg);
            matvec(std::span<const double>(
                       lw.exp_up.data() + std::size_t(eid) * mf * h,
                       std::size_t(mf) * h),
                   mf, h, norm_h, eu);
            for (U32 i = 0; i < mf; ++i) {
              eprod[i] = silu(eg[i]) * eu[i];
            }
            matvec(std::span<const double>(
                       lw.exp_down.data() + std::size_t(eid) * h * mf,
                       std::size_t(h) * mf),
                   h, mf, eprod, edown);
            for (U32 d = 0; d < h; ++d) {
              delta[d] += weights[ki] * edown[d];
            }
          }

          if (cfg.shared_expert_intermediate_size > 0) {
            const U32 sf =
                static_cast<U32>(cfg.shared_expert_intermediate_size);
            std::vector<double> sg(sf), su(sf), sprod(sf), sdown(h);
            matvec(lw.shared_gate, sf, h, norm_h, sg);
            matvec(lw.shared_up, sf, h, norm_h, su);
            for (U32 i = 0; i < sf; ++i) {
              sprod[i] = silu(sg[i]) * su[i];
            }
            matvec(lw.shared_down, h, sf, sprod, sdown);
            double s_scale = 1.0;
            if (cfg.has_shared_expert_gate) {
              double g_dot = 0.0;
              for (U32 d = 0; d < h; ++d) {
                g_dot += lw.shared_expert_gate[d] * norm_h[d];
              }
              s_scale = sigmoid(g_dot);
            }
            for (U32 d = 0; d < h; ++d) {
              delta[d] += s_scale * sdown[d];
            }
          }
        }

        for (U32 d = 0; d < h; ++d) {
          hidden[std::size_t(t) * h + d] += delta[d];
        }
      }
    }

    trace.logits.resize(std::size_t(T) * v, 0.0);
    for (U32 t = 0; t < T; ++t) {
      rmsnorm(std::span<const double>(hidden.data() + std::size_t(t) * h, h),
              final_norm, norm_h);
      matvec(lm_head, v, h, norm_h,
             std::span<double>(trace.logits.data() + std::size_t(t) * v, v));
    }
    return trace;
  }
};

std::int32_t run_engine_single_token(const std::filesystem::path &model_dir,
                                     const std::filesystem::path &metallib_dir,
                                     std::span<const std::int32_t> prompt,
                                     std::string kv_dtype) {
  LoadOptions opts;
  opts.model_dir = model_dir.string();
  opts.max_context = 1024;
  opts.max_concurrency = 1;
  opts.kv_dtype = std::move(kv_dtype);
  opts.sampling_contract = kDefaultServingSamplingContract;
  opts.path_overrides["artifact"] = (model_dir / "model.miv").string();
  opts.path_overrides["kernel_directory"] = metallib_dir.string();
  auto engine = Engine::create(opts);
  SubmitRequest req;
  req.prompt_tokens.assign(prompt.begin(), prompt.end());
  req.max_new_tokens = 1;
  req.sampling.temperature = 0.0F;
  const U64 req_id = engine->submit(std::move(req));
  std::optional<std::int32_t> tok;
  const auto deadline = std::chrono::steady_clock::now() + 15s;
  while (std::chrono::steady_clock::now() < deadline) {
    const auto events = engine->poll_events(50ms);
    bool finished = false;
    for (const auto &ev : events) {
      if (ev.request_id != req_id) {
        continue;
      }
      if (ev.kind == EventKind::failed) {
        engine->shutdown();
        throw std::runtime_error("Engine request failed: " + ev.error);
      }
      if (!ev.tokens.empty()) {
        tok = ev.tokens.front();
      }
      if (ev.kind == EventKind::finished) {
        finished = true;
      }
    }
    if (finished) {
      break;
    }
  }
  engine->shutdown();
  require(tok.has_value(), "Engine failed to emit token within timeout");
  return *tok;
}

// ============================================================================
// Suite 1: Config Generalization, Reconciliation & Tied lm_head
// ============================================================================
void test_config_generalization_and_tied_lm_head(
    const MetalContext &ctx, const std::filesystem::path &metallib_dir,
    const std::filesystem::path &tmp_root) {
  std::cout << "[1/5] Testing config generalization, reconciliation & tied lm_head...\n";

  // 1A: Alias fields ("quantization_config", "n_routed_experts") in flat config
  {
    const auto cfg_dir = tmp_root / "cfg_aliases";
    std::filesystem::create_directories(cfg_dir);
    Json flat_moe_cfg = {
        {"model_type", "deepseek_v3_like_custom"},
        {"torch_dtype", "float16"},
        {"hidden_size", 256},
        {"intermediate_size", 1024},
        {"moe_intermediate_size", 128},
        {"vocab_size", 128},
        {"num_hidden_layers", 2},
        {"num_attention_heads", 4},
        {"num_key_value_heads", 2},
        {"n_routed_experts", 16},
        {"num_experts_per_token", 4},
        {"rms_norm_eps", 1e-6},
        {"rope_theta", 10000.0},
        {"tie_word_embeddings", true},
        {"quantization_config", {{"bits", 4}, {"group_size", 64}, {"mode", "affine"}}},
    };
    write_text_file(cfg_dir / "config.json", flat_moe_cfg.dump(2));
    const auto parsed = load_qwen_model_config(cfg_dir / "config.json");
    require(parsed.model_type == "deepseek_v3_like_custom", "custom model_type");
    require(parsed.dtype == "float16", "torch_dtype fallback");
    require(parsed.head_dim == 64, "derived head_dim = 256/4");
    require(parsed.full_attention_layers() == 2 &&
                parsed.linear_attention_layers() == 0,
            "default layer_types = all full_attention");
    require(parsed.num_experts == 16 && parsed.num_experts_per_token == 4,
            "n_routed_experts alias");
    require(parsed.intermediate_size == 128,
            "moe_intermediate_size preferred over dense intermediate_size on MoE");
    require(parsed.shared_expert_intermediate_size == 0,
            "omitted shared_expert_intermediate_size defaults to 0");
    require(parsed.tie_word_embeddings, "tie_word_embeddings parsed");
    require(parsed.default_quantization.bits == 4 &&
                parsed.default_quantization.group_size == 64,
            "quantization_config parsed");
    std::string reason;
    require(QwenForward::supports_config(parsed, &reason),
            "supports_config should accept 16-expert top-4 pure-GQA: " + reason);
  }

  // 1B: Tied lm_head across 3 modes:
  //   mode 0: direct_bf16_v1 embed_tokens, NO lm_head object/binding, flat "layers.<i>." names
  //   mode 1: direct_bf16_v1 embed_tokens + use="lm_head" alias binding sharing object_id
  //   mode 2: quantized oq_affine_q4_g64_bf16_v1 embed_tokens + use="lm_head" alias binding
  for (int mode = 0; mode < 3; ++mode) {
    const auto dir = tmp_root / ("tied_mode_" + std::to_string(mode));
    std::filesystem::create_directories(dir);
    constexpr U32 H = 256U, F = 256U, V = 128U, QH = 2U, KVH = 1U, HD = 128U;
    Json cfg_json = {
        {"model_type", "community_custom_arch"},
        {"torch_dtype", "bfloat16"},
        {"hidden_size", H},
        {"intermediate_size", F},
        {"vocab_size", V},
        {"num_hidden_layers", 2},
        {"num_attention_heads", QH},
        {"num_key_value_heads", KVH},
        {"rms_norm_eps", 1e-6},
        {"rope_theta", 500000.0},
        {"tie_word_embeddings", true},
    };
    write_text_file(dir / "config.json", cfg_json.dump(2));

    CpuModelOracle oracle;
    oracle.cfg = load_qwen_model_config(dir / "config.json");
    oracle.cfg.has_attn_output_gate = false;
    oracle.cfg.has_qk_norm = true;
    oracle.layers.resize(2);

    UniversalArtifactBuilder b;
    const std::string pfx = (mode == 0) ? "" : "model.";
    const WeightEnc emb_enc =
        (mode == 2) ? WeightEnc::AffineQ4G64RowSplit : WeightEnc::DirectBf16;
    const U32 emb_id = b.add_matrix_2d(
        pfx + "embed_tokens.weight", V, H, emb_enc, 101U,
        (mode == 2) ? "embedding" : "", &oracle.embedding, 0.06F);
    oracle.lm_head = oracle.embedding;
    if (mode == 1) {
      b.add_alias_binding({.name = pfx + "lm_head.weight",
                           .object_id = emb_id,
                           .format = "direct_bf16_v1",
                           .out_features = V,
                           .in_features = H,
                           .use = "lm_head"});
    } else if (mode == 2) {
      b.add_alias_binding({.name = pfx + "lm_head.weight",
                           .object_id = emb_id,
                           .format = "oq_affine_q4_g64_bf16_v1",
                           .out_features = V,
                           .in_features = H,
                           .use = "lm_head"});
    }

    for (U32 li = 0; li < 2U; ++li) {
      auto &lw = oracle.layers[li];
      const std::string lp = pfx + "layers." + std::to_string(li) + ".";
      const U32 s = 200U + li * 30U;
      b.add_norm_1d(lp + "input_layernorm.weight", H, s + 1U, false,
                    &lw.input_norm);
      b.add_matrix_2d(lp + "self_attn.q_proj.weight", QH * HD, H,
                      WeightEnc::AffineQ4G64Tiled, s + 2U, "linear",
                      &lw.q_proj);
      b.add_matrix_2d(lp + "self_attn.k_proj.weight", KVH * HD, H,
                      WeightEnc::AffineQ4G64Tiled, s + 3U, "linear",
                      &lw.k_proj);
      b.add_matrix_2d(lp + "self_attn.v_proj.weight", KVH * HD, H,
                      WeightEnc::AffineQ4G64Tiled, s + 4U, "linear",
                      &lw.v_proj);
      b.add_matrix_2d(lp + "self_attn.o_proj.weight", H, QH * HD,
                      WeightEnc::AffineQ4G64Tiled, s + 5U, "linear",
                      &lw.o_proj);
      b.add_norm_1d(lp + "self_attn.q_norm.weight", HD, s + 6U, false,
                    &lw.q_norm);
      b.add_norm_1d(lp + "self_attn.k_norm.weight", HD, s + 7U, false,
                    &lw.k_norm);
      b.add_norm_1d(lp + "post_attention_layernorm.weight", H, s + 8U, false,
                    &lw.post_norm);
      b.add_matrix_2d(lp + "mlp.gate_proj.weight", F, H,
                      WeightEnc::AffineQ4G64Tiled, s + 9U, "linear",
                      &lw.mlp_gate);
      b.add_matrix_2d(lp + "mlp.up_proj.weight", F, H,
                      WeightEnc::AffineQ4G64Tiled, s + 10U, "linear",
                      &lw.mlp_up);
      b.add_matrix_2d(lp + "mlp.down_proj.weight", H, F,
                      WeightEnc::AffineQ4G64Tiled, s + 11U, "linear",
                      &lw.mlp_down);
    }
    b.add_norm_1d(pfx + "norm.weight", H, 999U, false, &oracle.final_norm);

    const auto art_path = dir / "model.miv";
    write_binary_file(art_path, b.serialize("universal_custom_tied"));

    auto art = ArtifactModel::open(art_path, true);
    auto cfg = load_qwen_model_config(dir / "config.json");
    reconcile_qwen_config_with_artifact(cfg, art);
    require(!cfg.has_attn_output_gate,
            "reconcile should detect ungated q_proj shape [q, h]");
    require(cfg.has_qk_norm, "reconcile should detect q_norm/k_norm");

    auto weights =
        materialize_model(art, plan_model_materialization(art, ctx), ctx);
    auto catalog = build_qwen_layer_catalog(art, cfg, 256U);
    QwenForward fwd(ctx, weights, catalog, cfg, 256U, metallib_dir, {}, false,
                    false, true, std::nullopt, false, false, std::nullopt,
                    LinearNumericPolicy::RowExactPortableV1,
                    SamplingContract::PositionCoupledV1,
                    AttentionNumericPolicy::RowExactV1, true, "bf16");
    require(fwd.embedding_weight().object_id == fwd.head_weight().object_id,
            "tied lm_head must share object_id with embed_tokens");

    const std::array<U32, 4> prompt{3U, 17U, 42U, 91U};
    GpuSamplingOptions samp{.temperature = 0.0F,
                            .top_k = 1U,
                            .top_p = 1.0F,
                            .contract = SamplingContract::PositionCoupledV1};
    const auto res = fwd.run(prompt, samp, /*inspect_tensors=*/true);
    const auto ref = oracle.forward(prompt);
    const double cos = cosine_similarity(res.logits, ref.logits);
    require(cos > 0.9999,
            "tied mode " + std::to_string(mode) +
                " cosine too low: " + std::to_string(cos));
    const auto ref_last =
        std::span<const double>(ref.logits.data() + 3U * V, V);
    const U32 ref_argmax = static_cast<U32>(
        std::distance(ref_last.begin(), std::ranges::max_element(ref_last)));
    require(res.token == ref_argmax,
            "tied mode " + std::to_string(mode) + " greedy token mismatch");

    // Verify Engine::create loads & generates exact token on bf16, and matches
    // QwenForward(affine4_g128) on affine4_g128!
    const std::array<std::int32_t, 4> eng_prompt{3, 17, 42, 91};
    const std::int32_t eng_bf16 =
        run_engine_single_token(dir, metallib_dir, eng_prompt, "bf16");
    require(static_cast<U32>(eng_bf16) == ref_argmax,
            "Engine(bf16) token mismatch on tied mode " + std::to_string(mode));

    QwenForward fwd_a4(ctx, weights, catalog, cfg, 256U, metallib_dir, {},
                       false, false, true, std::nullopt, false, false,
                       std::nullopt, LinearNumericPolicy::RowExactPortableV1,
                       SamplingContract::PositionCoupledV1,
                       AttentionNumericPolicy::RowExactV1, true,
                       "affine4_g128");
    const auto res_a4 = fwd_a4.run(prompt, samp, /*inspect_tensors=*/true);
    const auto ref_a4 = oracle.forward(prompt, /*quantize_kv_affine4=*/true);
    const double cos_a4_exact = cosine_similarity(res_a4.logits, ref_a4.logits);
    const double cos_a4_bf16 = cosine_similarity(res_a4.logits, ref.logits);
    require(cos_a4_exact > 0.999 && cos_a4_bf16 > 0.985,
            "QwenForward(affine4_g128) cosine on tied mode " +
                std::to_string(mode) + " cos_exact=" +
                std::to_string(cos_a4_exact) + " cos_bf16=" +
                std::to_string(cos_a4_bf16));
    const std::int32_t eng_a4 =
        run_engine_single_token(dir, metallib_dir, eng_prompt, "affine4_g128");
    require(static_cast<U32>(eng_a4) == res_a4.token,
            "Engine(affine4_g128) vs QwenForward(affine4_g128) mismatch on tied mode " +
                std::to_string(mode));
  }
}

// ============================================================================
// Suite 2: 0-GDN Pure GQA + Ungated q_proj + Optional QK-Norm + KV Dtype Guard
// ============================================================================
void test_pure_gqa_ungated_optional_qknorm_and_kv_guard(
    const MetalContext &ctx, const std::filesystem::path &metallib_dir,
    const std::filesystem::path &tmp_root) {
  std::cout << "[2/5] Testing 0-GDN pure GQA, ungated q_proj, optional QK-norm & KV dtype guard...\n";

  struct CaseSpec {
    std::string tag;
    U32 h;
    U32 f;
    U32 qh;
    U32 kvh;
    U32 hd;
    bool has_gate;
    bool has_qkn;
  };
  const std::array<CaseSpec, 4> cases{{
      // Qwen3-style pure GQA: hd=128, ungated, with QK-norm
      {"qwen3_hd128", 256U, 256U, 2U, 1U, 128U, false, true},
      // Llama-style pure GQA: hd=64, ungated, NO QK-norm (tests affine4 exact hd=64!)
      {"llama_hd64_noqkn", 256U, 256U, 4U, 2U, 64U, false, false},
      // Non-pow2 head_dim=96 (H=192): tests NativeQwenState affine4->bf16 fallback!
      {"llama_hd96_fallback", 192U, 256U, 2U, 1U, 96U, false, false},
      // HD=256 NAX Flash-Prefill + Affine4 KV: ungated, NO QK-norm!
      {"nax_hd256_noqkn", 512U, 512U, 2U, 1U, 256U, false, false},
  }};

  for (const auto &cs : cases) {
    const auto dir = tmp_root / ("gqa_" + cs.tag);
    std::filesystem::create_directories(dir);
    constexpr U32 V = 128U;
    Json cfg_json = {
        {"model_type", cs.has_qkn ? "qwen3" : "llama"},
        {"dtype", "bfloat16"},
        {"hidden_size", cs.h},
        {"intermediate_size", cs.f},
        {"vocab_size", V},
        {"num_hidden_layers", 2},
        {"num_attention_heads", cs.qh},
        {"num_key_value_heads", cs.kvh},
        {"head_dim", cs.hd},
        {"rms_norm_eps", 1e-5},
        {"rope_theta", 100000.0},
    };
    write_text_file(dir / "config.json", cfg_json.dump(2));

    CpuModelOracle oracle;
    oracle.cfg = load_qwen_model_config(dir / "config.json");
    oracle.cfg.has_attn_output_gate = cs.has_gate;
    oracle.cfg.has_qk_norm = cs.has_qkn;
    oracle.layers.resize(2);

    UniversalArtifactBuilder b;
    b.add_matrix_2d("model.embed_tokens.weight", V, cs.h, WeightEnc::DirectBf16,
                    301U, "", &oracle.embedding, 0.06F);
    const U32 q_rows =
        cs.has_gate ? (2U * cs.qh * cs.hd) : (cs.qh * cs.hd);
    for (U32 li = 0; li < 2U; ++li) {
      auto &lw = oracle.layers[li];
      const std::string lp = "model.layers." + std::to_string(li) + ".";
      const U32 s = 400U + li * 30U;
      // Note: when H=192 (not multiple of 128), use DirectBf16 for K=192 weights
      const WeightEnc k_h_enc =
          (cs.h % 128U == 0U) ? WeightEnc::AffineQ4G64Tiled
                              : WeightEnc::DirectBf16;
      const WeightEnc k_q_enc =
          ((cs.qh * cs.hd) % 128U == 0U) ? WeightEnc::AffineQ4G64Tiled
                                         : WeightEnc::DirectBf16;
      b.add_norm_1d(lp + "input_layernorm.weight", cs.h, s + 1U, false,
                    &lw.input_norm);
      b.add_matrix_2d(lp + "self_attn.q_proj.weight", q_rows, cs.h, k_h_enc,
                      s + 2U, "linear", &lw.q_proj);
      b.add_matrix_2d(lp + "self_attn.k_proj.weight", cs.kvh * cs.hd, cs.h,
                      k_h_enc, s + 3U, "linear", &lw.k_proj);
      b.add_matrix_2d(lp + "self_attn.v_proj.weight", cs.kvh * cs.hd, cs.h,
                      k_h_enc, s + 4U, "linear", &lw.v_proj);
      b.add_matrix_2d(lp + "self_attn.o_proj.weight", cs.h, cs.qh * cs.hd,
                      k_q_enc, s + 5U, "linear", &lw.o_proj);
      if (cs.has_qkn) {
        b.add_norm_1d(lp + "self_attn.q_norm.weight", cs.hd, s + 6U, false,
                      &lw.q_norm);
        b.add_norm_1d(lp + "self_attn.k_norm.weight", cs.hd, s + 7U, false,
                      &lw.k_norm);
      }
      b.add_norm_1d(lp + "post_attention_layernorm.weight", cs.h, s + 8U, false,
                    &lw.post_norm);
      b.add_matrix_2d(lp + "mlp.gate_proj.weight", cs.f, cs.h, k_h_enc, s + 9U,
                      "linear", &lw.mlp_gate);
      b.add_matrix_2d(lp + "mlp.up_proj.weight", cs.f, cs.h, k_h_enc, s + 10U,
                      "linear", &lw.mlp_up);
      b.add_matrix_2d(lp + "mlp.down_proj.weight", cs.h, cs.f,
                      WeightEnc::AffineQ4G64Tiled, s + 11U, "linear",
                      &lw.mlp_down);
    }
    b.add_norm_1d("model.norm.weight", cs.h, 888U, false, &oracle.final_norm);
    b.add_matrix_2d("lm_head.weight", V, cs.h, WeightEnc::DirectBf16, 889U, "",
                    &oracle.lm_head, 0.06F);

    const auto art_path = dir / "model.miv";
    write_binary_file(art_path, b.serialize("universal_pure_gqa"));

    auto art = ArtifactModel::open(art_path, true);
    auto cfg = load_qwen_model_config(dir / "config.json");
    reconcile_qwen_config_with_artifact(cfg, art);
    require(cfg.has_attn_output_gate == cs.has_gate, "reconciled gate flag");
    require(cfg.has_qk_norm == cs.has_qkn, "reconciled qk_norm flag");

    auto weights =
        materialize_model(art, plan_model_materialization(art, ctx), ctx);
    auto catalog = build_qwen_layer_catalog(art, cfg, 256U);

    // Test 20-token prompt on hd=256 so Flash-NAX prefill (>=16 tokens) runs live!
    std::vector<U32> prompt;
    const U32 prompt_len = (cs.hd == 256U) ? 20U : 5U;
    for (U32 i = 0; i < prompt_len; ++i) {
      prompt.push_back((i * 13U + 5U) % V);
    }
    const U32 next_tok_in = 77U;
    std::vector<U32> full_seq = prompt;
    full_seq.push_back(next_tok_in);
    const auto ref_full = oracle.forward(full_seq, /*quantize_kv_affine4=*/false);
    const auto ref_full_a4 =
        oracle.forward(full_seq, /*quantize_kv_affine4=*/true);

    for (std::string_view kv_dt : {"bf16", "affine4_g128"}) {
      if (kv_dt == "affine4_g128" && cs.hd == 96U) {
        // hd=96 is guarded at NativeQwenState / Engine level (falls back to bf16)
        continue;
      }
      const auto &cur_ref = (kv_dt == "affine4_g128") ? ref_full_a4 : ref_full;
      QwenForward fwd(ctx, weights, catalog, cfg, 256U, metallib_dir, {}, false,
                      false, true, std::nullopt, /*split=*/true, /*batch=*/true,
                      std::nullopt, LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1,
                      AttentionNumericPolicy::RowExactV1, true, kv_dt, nullptr,
                      /*enable_flash_nax_prefill=*/(cs.hd == 256U));
      // Verify 0 GDN buffers were allocated!
      for (const auto &buf : fwd.memory_plan().buffers) {
        require(buf.name != "qkv" && buf.name != "convolved" &&
                    buf.name != "recurrent" && buf.name != "gdn_chunk_ws",
                "0-GDN model must not allocate GDN buffer: " + buf.name);
      }
      GpuSamplingOptions samp{.temperature = 0.0F,
                              .top_k = 1U,
                              .top_p = 1.0F,
                              .contract = SamplingContract::PositionCoupledV1};
      const auto r_prefill = fwd.run(prompt, samp, /*inspect_tensors=*/true);
      const auto ref_prefill = std::span<const double>(
          cur_ref.logits.data(), std::size_t(prompt_len) * V);
      const double min_cos = (kv_dt == "bf16") ? 0.9995 : 0.997;
      const double cos_p = cosine_similarity(r_prefill.logits, ref_prefill);
      require(cos_p > min_cos, cs.tag + " (" + std::string(kv_dt) +
                                   ") prefill cos=" + std::to_string(cos_p));

      const std::array<U32, 1> step1{next_tok_in};
      const auto r_dec = fwd.run(step1, samp, /*inspect_tensors=*/true);
      const auto ref_dec = std::span<const double>(
          cur_ref.logits.data() + std::size_t(prompt_len) * V, V);
      const double cos_d = cosine_similarity(r_dec.logits, ref_dec);
      require(cos_d > min_cos, cs.tag + " (" + std::string(kv_dt) +
                                   ") decode cos=" + std::to_string(cos_d));

      // Also test run_packed_batch across 2 slots (exercises compile_packed_batch_profile!)
      if (kv_dt == "bf16") {
        fwd.reset();
        QwenForward slot1(
            ctx, weights, catalog, cfg, 256U, metallib_dir, {}, false, false,
            true, std::nullopt, true, true, std::nullopt,
            LinearNumericPolicy::RowExactPortableV1,
            SamplingContract::PositionCoupledV1,
            AttentionNumericPolicy::RowExactV1, false, kv_dt, &fwd,
            (cs.hd == 256U));
        const std::array<U32, 3> s0_toks{prompt[0], prompt[1], prompt[2]};
        const std::array<U32, 2> s1_toks{11U, 29U};
        const std::array<PackedSlotSlice, 2> slices{{
            {.slot_program = &fwd,
             .slot_index = 0U,
             .tokens = s0_toks,
             .sampling = samp,
             .verify = false},
            {.slot_program = &slot1,
             .slot_index = 1U,
             .tokens = s1_toks,
             .sampling = samp,
             .verify = false},
        }};
        (void)fwd.run_packed_batch(slices, /*use_icb=*/true,
                                   /*scatter_slot_logits=*/true);
        const auto s1_raw = slot1.read_last_logits_bf16();
        std::vector<float> s1_logits(V);
        for (U32 i = 0; i < V; ++i) {
          std::uint16_t u = 0;
          std::memcpy(&u, s1_raw.data() + i * 2U, 2U);
          s1_logits[i] = bf16_to_f32(u);
        }
        const auto s1_ref = oracle.forward(s1_toks);
        const double cos_b = cosine_similarity(
            s1_logits, std::span<const double>(s1_ref.logits.data() + V, V));
        require(cos_b > 0.9995,
                cs.tag + " packed_batch slot1 cos=" + std::to_string(cos_b));
      }
    }

    // Engine::create with default "affine4_g128" MUST succeed for ALL head_dims
    // (including hd=64, hd=96 fallback, hd=128, hd=256)!
    std::vector<std::int32_t> eng_prompt;
    for (U32 tok : prompt) {
      eng_prompt.push_back(static_cast<std::int32_t>(tok));
    }
    const std::int32_t eng_tok =
        run_engine_single_token(dir, metallib_dir, eng_prompt, "affine4_g128");
    require(eng_tok >= 0 && static_cast<U32>(eng_tok) < V,
            "Engine valid token on " + cs.tag);
  }
}

// ============================================================================
// Suite 3: Generalized MoE (64 exp top-4 no shared; 128 exp top-8 +/- shared_gate)
// ============================================================================
void test_generalized_moe_execution(const MetalContext &ctx,
                                    const std::filesystem::path &metallib_dir,
                                    const std::filesystem::path &tmp_root) {
  std::cout << "[3/5] Testing generalized MoE (64/128 experts, top-4/top-8, optional shared_expert)...\n";

  struct MoeSpec {
    std::string tag;
    U32 num_experts;
    U32 top_k;
    U32 shared_inter;
    bool has_shared_gate;
  };
  const std::array<MoeSpec, 3> specs{{
      {"e64_k4_noshared", 64U, 4U, 0U, false},
      {"e128_k8_shared_nogate", 128U, 8U, 128U, false},
      {"e128_k8_shared_gated", 128U, 8U, 128U, true},
  }};

  for (const auto &ms : specs) {
    const auto dir = tmp_root / ("moe_" + ms.tag);
    std::filesystem::create_directories(dir);
    constexpr U32 H = 256U, MF = 128U, V = 128U, QH = 2U, KVH = 1U, HD = 128U;
    Json cfg_json = {
        {"model_type", "qwen3_moe"},
        {"dtype", "bfloat16"},
        {"hidden_size", H},
        {"moe_intermediate_size", MF},
        {"shared_expert_intermediate_size", ms.shared_inter},
        {"num_experts", ms.num_experts},
        {"num_experts_per_token", ms.top_k},
        {"norm_topk_prob", true},
        {"vocab_size", V},
        {"num_hidden_layers", 2},
        {"num_attention_heads", QH},
        {"num_key_value_heads", KVH},
        {"head_dim", HD},
        {"rms_norm_eps", 1e-6},
        {"rope_theta", 100000.0},
    };
    write_text_file(dir / "config.json", cfg_json.dump(2));

    CpuModelOracle oracle;
    oracle.cfg = load_qwen_model_config(dir / "config.json");
    oracle.cfg.has_attn_output_gate = false;
    oracle.cfg.has_qk_norm = true;
    oracle.cfg.has_shared_expert_gate = ms.has_shared_gate;
    oracle.layers.resize(2);

    UniversalArtifactBuilder b;
    b.add_matrix_2d("model.embed_tokens.weight", V, H, WeightEnc::DirectBf16,
                    501U, "", &oracle.embedding, 0.08F);
    for (U32 li = 0; li < 2U; ++li) {
      auto &lw = oracle.layers[li];
      const std::string lp = "model.layers." + std::to_string(li) + ".";
      const U32 s = 600U + li * 40U;
      b.add_norm_1d(lp + "input_layernorm.weight", H, s + 1U, false,
                    &lw.input_norm);
      b.add_matrix_2d(lp + "self_attn.q_proj.weight", QH * HD, H,
                      WeightEnc::AffineQ4G64Tiled, s + 2U, "linear",
                      &lw.q_proj);
      b.add_matrix_2d(lp + "self_attn.k_proj.weight", KVH * HD, H,
                      WeightEnc::AffineQ4G64Tiled, s + 3U, "linear",
                      &lw.k_proj);
      b.add_matrix_2d(lp + "self_attn.v_proj.weight", KVH * HD, H,
                      WeightEnc::AffineQ4G64Tiled, s + 4U, "linear",
                      &lw.v_proj);
      b.add_matrix_2d(lp + "self_attn.o_proj.weight", H, QH * HD,
                      WeightEnc::AffineQ4G64Tiled, s + 5U, "linear",
                      &lw.o_proj);
      b.add_norm_1d(lp + "self_attn.q_norm.weight", HD, s + 6U, false,
                    &lw.q_norm);
      b.add_norm_1d(lp + "self_attn.k_norm.weight", HD, s + 7U, false,
                    &lw.k_norm);
      b.add_norm_1d(lp + "post_attention_layernorm.weight", H, s + 8U, false,
                    &lw.post_norm);
      // Distinct router weights (amp=0.15) so top-k ordering is well-separated
      b.add_matrix_2d(lp + "mlp.gate.weight", ms.num_experts, H,
                      WeightEnc::DirectBf16, s + 9U, "router", &lw.router,
                      0.15F);
      b.add_moe_mxfp4_3d(lp + "mlp.switch_mlp.gate_proj", ms.num_experts, MF, H,
                         s + 10U, &lw.exp_gate);
      b.add_moe_mxfp4_3d(lp + "mlp.switch_mlp.up_proj", ms.num_experts, MF, H,
                         s + 11U, &lw.exp_up);
      b.add_moe_mxfp4_3d(lp + "mlp.switch_mlp.down_proj", ms.num_experts, H, MF,
                         s + 12U, &lw.exp_down);
      if (ms.shared_inter > 0U) {
        b.add_matrix_2d(lp + "mlp.shared_expert.gate_proj.weight",
                        ms.shared_inter, H, WeightEnc::AffineQ4G64Tiled,
                        s + 13U, "linear", &lw.shared_gate);
        b.add_matrix_2d(lp + "mlp.shared_expert.up_proj.weight",
                        ms.shared_inter, H, WeightEnc::AffineQ4G64Tiled,
                        s + 14U, "linear", &lw.shared_up);
        b.add_matrix_2d(lp + "mlp.shared_expert.down_proj.weight", H,
                        ms.shared_inter, WeightEnc::AffineQ4G64Tiled, s + 15U,
                        "linear", &lw.shared_down);
        if (ms.has_shared_gate) {
          b.add_matrix_2d(lp + "mlp.shared_expert_gate.weight", 1U, H,
                          WeightEnc::DirectBf16, s + 16U, "shared_gate",
                          &lw.shared_expert_gate, 0.10F);
        }
      }
    }
    b.add_norm_1d("model.norm.weight", H, 777U, false, &oracle.final_norm);
    b.add_matrix_2d("lm_head.weight", V, H, WeightEnc::DirectBf16, 778U, "",
                    &oracle.lm_head, 0.06F);

    const auto art_path = dir / "model.miv";
    write_binary_file(art_path, b.serialize("universal_moe"));

    auto art = ArtifactModel::open(art_path, true);
    auto cfg = load_qwen_model_config(dir / "config.json");
    reconcile_qwen_config_with_artifact(cfg, art);
    require(cfg.shared_expert_intermediate_size == int(ms.shared_inter),
            "reconciled shared_expert_intermediate_size");
    require(cfg.has_shared_expert_gate == ms.has_shared_gate,
            "reconciled has_shared_expert_gate");

    auto weights =
        materialize_model(art, plan_model_materialization(art, ctx), ctx);
    auto catalog = build_qwen_layer_catalog(art, cfg, 256U);
    QwenForward fwd(ctx, weights, catalog, cfg, 256U, metallib_dir, {},
                    /*inspect_routes=*/true, /*speculative=*/true, true,
                    std::nullopt, false, true, std::nullopt,
                    LinearNumericPolicy::RowExactPortableV1,
                    SamplingContract::PositionCoupledV1,
                    AttentionNumericPolicy::RowExactV1, true, "bf16");

    if (ms.shared_inter == 0U) {
      for (const auto &buf : fwd.memory_plan().buffers) {
        require(buf.name != "shared" && buf.name != "shared_gate",
                "0-shared_expert MoE must not allocate " + buf.name);
      }
    }

    const std::array<U32, 4> prompt{5U, 23U, 61U, 109U};
    GpuSamplingOptions samp{.temperature = 0.0F,
                            .top_k = 1U,
                            .top_p = 1.0F,
                            .contract = SamplingContract::PositionCoupledV1};
    const auto res = fwd.run(prompt, samp, /*inspect_tensors=*/true);
    const auto ref = oracle.forward(prompt);
    const double cos_p = cosine_similarity(res.logits, ref.logits);
    require(cos_p > 0.999,
            ms.tag + " MoE T=4 prefill cos=" + std::to_string(cos_p));

    // Verify route trace dimensions & selected expert IDs!
    require(res.layer_routes.size() == 4U * 2U * ms.top_k,
            ms.tag + " layer_routes size");
    require(res.router_logits.size() == 4U * 2U * ms.num_experts,
            ms.tag + " router_logits size");
    for (U32 li = 0; li < 2U; ++li) {
      const auto snap = fwd.read_moe_route_trace(li);
      require(snap.expert_ids.size() == 4U * ms.top_k,
              ms.tag + " read_moe_route_trace expert_ids size");
      for (U32 tok = 0; tok < 4U; ++tok) {
        const float *rl =
            res.router_logits.data() + (tok * 2U + li) * ms.num_experts;
        std::vector<bool> used(ms.num_experts, false);
        for (U32 ki = 0; ki < ms.top_k; ++ki) {
          float best_v = -1e30F;
          U32 best_e = 0U;
          for (U32 e = 0; e < ms.num_experts; ++e) {
            if (!used[e] && rl[e] >= best_v) {
              best_v = rl[e];
              best_e = e;
            }
          }
          used[best_e] = true;
          const std::size_t idx = std::size_t(tok) * ms.top_k + ki;
          require(snap.expert_ids[idx] == best_e,
                  ms.tag + " layer " + std::to_string(li) +
                      " route trace expert_id mismatch at " +
                      std::to_string(idx));
        }
      }
    }

    // Also test T=1 decode step (exercises T=1 MoE NAX / portable path!)
    const std::array<U32, 1> step1{39U};
    const std::array<U32, 5> seq5{5U, 23U, 61U, 109U, 39U};
    const auto res1 = fwd.run(step1, samp, /*inspect_tensors=*/true);
    const auto ref5 = oracle.forward(seq5);
    const double cos_d = cosine_similarity(
        res1.logits, std::span<const double>(ref5.logits.data() + 4U * V, V));
    require(cos_d > 0.999,
            ms.tag + " MoE T=1 decode cos=" + std::to_string(cos_d));

    // Test T=8 speculative verify on generalized MoE!
    const std::array<U32, 8> verify8{12U, 24U, 36U, 48U, 60U, 72U, 84U, 96U};
    std::vector<U32> seq13(seq5.begin(), seq5.end());
    seq13.insert(seq13.end(), verify8.begin(), verify8.end());
    const auto res_v8 = fwd.verify(verify8, samp, /*inspect_tensors=*/true);
    (void)fwd.commit_prefix(8U);
    const auto ref13 = oracle.forward(seq13);
    const double cos_v8 = cosine_similarity(
        res_v8.logits,
        std::span<const double>(ref13.logits.data() + 5U * V, 8U * V));
    require(cos_v8 > 0.999,
            ms.tag + " MoE T=8 verify cos=" + std::to_string(cos_v8));

    // Test B=2 run_packed_batch on generalized MoE (exercises batched MoE)!
    fwd.reset();
    QwenForward slot1(ctx, weights, catalog, cfg, 256U, metallib_dir, {},
                      /*inspect_routes=*/false, /*speculative=*/true, true,
                      std::nullopt, false, true, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1,
                      AttentionNumericPolicy::RowExactV1, false, "bf16", &fwd);
    const std::array<U32, 2> b_s0{5U, 23U};
    const std::array<U32, 2> b_s1{61U, 109U};
    const std::array<PackedSlotSlice, 2> moe_slices{{
        {.slot_program = &fwd,
         .slot_index = 0U,
         .tokens = b_s0,
         .sampling = samp,
         .verify = false},
        {.slot_program = &slot1,
         .slot_index = 1U,
         .tokens = b_s1,
         .sampling = samp,
         .verify = false},
    }};
    (void)fwd.run_packed_batch(moe_slices, /*use_icb=*/true,
                               /*scatter_slot_logits=*/true);
    const auto s1_moe_raw = slot1.read_last_logits_bf16();
    std::vector<float> s1_moe_logits(V);
    for (U32 i = 0; i < V; ++i) {
      std::uint16_t u = 0;
      std::memcpy(&u, s1_moe_raw.data() + i * 2U, 2U);
      s1_moe_logits[i] = bf16_to_f32(u);
    }
    const auto ref_bs1 = oracle.forward(b_s1);
    const double cos_mb = cosine_similarity(
        s1_moe_logits, std::span<const double>(ref_bs1.logits.data() + V, V));
    require(cos_mb > 0.999,
            ms.tag + " MoE packed_batch slot1 cos=" + std::to_string(cos_mb));

    // Verify Engine::create on generalized MoE
    const std::array<std::int32_t, 4> eng_prompt{5, 23, 61, 109};
    const std::int32_t eng_tok =
        run_engine_single_token(dir, metallib_dir, eng_prompt, "bf16");
    const auto ref_last =
        std::span<const double>(ref.logits.data() + 3U * V, V);
    const U32 ref_argmax = static_cast<U32>(
        std::distance(ref_last.begin(), std::ranges::max_element(ref_last)));
    require(static_cast<U32>(eng_tok) == ref_argmax,
            ms.tag + " Engine greedy token mismatch");
  }
}

// ============================================================================
// Suite 4: Mixed-Precision (.miv combining Q4 + Q8 + direct_f16_v1 + direct_bf16_v1)
// ============================================================================
void test_mixed_precision_and_direct_f16_execution(
    const MetalContext &ctx, const std::filesystem::path &metallib_dir,
    const std::filesystem::path &tmp_root) {
  std::cout << "[4/5] Testing mixed-precision & direct_f16_v1 execution...\n";

  const auto dir = tmp_root / "mixed_prec_f16";
  std::filesystem::create_directories(dir);
  constexpr U32 H = 512U, F = 512U, V = 128U, QH = 4U, KVH = 2U, HD = 128U;
  Json cfg_json = {
      {"model_type", "mistral"},
      {"torch_dtype", "float16"},
      {"hidden_size", H},
      {"intermediate_size", F},
      {"vocab_size", V},
      {"num_hidden_layers", 2},
      {"num_attention_heads", QH},
      {"num_key_value_heads", KVH},
      {"rms_norm_eps", 1e-5},
      {"rope_theta", 10000.0},
  };
  write_text_file(dir / "config.json", cfg_json.dump(2));

  CpuModelOracle oracle;
  oracle.cfg = load_qwen_model_config(dir / "config.json");
  oracle.cfg.has_attn_output_gate = false;
  oracle.cfg.has_qk_norm = false;
  oracle.layers.resize(2);

  UniversalArtifactBuilder b;
  // FP16 embedding!
  b.add_matrix_2d("model.embed_tokens.weight", V, H, WeightEnc::DirectF16, 701U,
                  "", &oracle.embedding, 0.06F);
  for (U32 li = 0; li < 2U; ++li) {
    auto &lw = oracle.layers[li];
    const std::string lp = "model.layers." + std::to_string(li) + ".";
    const U32 s = 800U + li * 30U;
    // FP16 RMSNorm weight!
    b.add_norm_1d(lp + "input_layernorm.weight", H, s + 1U, /*as_f16=*/true,
                  &lw.input_norm);
    // Q4_g64 + Q8_g64 + DirectF16 + DirectBf16 mixed in the same layer!
    b.add_matrix_2d(lp + "self_attn.q_proj.weight", QH * HD, H,
                    WeightEnc::AffineQ4G64Tiled, s + 2U, "linear", &lw.q_proj);
    b.add_matrix_2d(lp + "self_attn.k_proj.weight", KVH * HD, H,
                    WeightEnc::AffineQ8G64Tiled, s + 3U, "linear", &lw.k_proj);
    b.add_matrix_2d(lp + "self_attn.v_proj.weight", KVH * HD, H,
                    WeightEnc::AffineQ4G64Tiled, s + 4U, "linear", &lw.v_proj);
    b.add_matrix_2d(lp + "self_attn.o_proj.weight", H, QH * HD,
                    WeightEnc::DirectF16, s + 5U, "linear", &lw.o_proj);
    b.add_norm_1d(lp + "post_attention_layernorm.weight", H, s + 6U,
                  /*as_f16=*/true, &lw.post_norm);
    b.add_matrix_2d(lp + "mlp.gate_proj.weight", F, H,
                    WeightEnc::AffineQ4G64Tiled, s + 7U, "linear",
                    &lw.mlp_gate);
    b.add_matrix_2d(lp + "mlp.up_proj.weight", F, H,
                    WeightEnc::AffineQ4G64Tiled, s + 8U, "linear", &lw.mlp_up);
    b.add_matrix_2d(lp + "mlp.down_proj.weight", H, F,
                    WeightEnc::AffineQ8G64Tiled, s + 9U, "linear",
                    &lw.mlp_down);
  }
  b.add_norm_1d("model.norm.weight", H, 950U, /*as_f16=*/true,
                &oracle.final_norm);
  // DirectBf16 lm_head alongside DirectF16 embedding & Q4/Q8 backbone!
  b.add_matrix_2d("lm_head.weight", V, H, WeightEnc::DirectBf16, 951U, "",
                  &oracle.lm_head, 0.06F);

  const auto art_path = dir / "model.miv";
  write_binary_file(art_path, b.serialize("universal_mixed_f16"));

  auto art = ArtifactModel::open(art_path, true);
  auto cfg = load_qwen_model_config(dir / "config.json");
  reconcile_qwen_config_with_artifact(cfg, art);
  auto weights =
      materialize_model(art, plan_model_materialization(art, ctx), ctx);
  auto catalog = build_qwen_layer_catalog(art, cfg, 256U);
  // Topology profile classifier must accept mixed Q4 + Q8 + direct_f16_v1 + direct_bf16_v1
  const auto topo = describe_qwen_topology(art, cfg, catalog, 256U);
  require(topo.quant_profile == QwenQuantProfile::oq_affine,
          "mixed Q4+Q8+F16+BF16 should classify as oq_affine");
  QwenForward fwd(ctx, weights, catalog, cfg, 256U, metallib_dir, {}, false,
                  false, true, std::nullopt, false, false, std::nullopt,
                  LinearNumericPolicy::RowExactPortableV1,
                  SamplingContract::PositionCoupledV1,
                  AttentionNumericPolicy::RowExactV1, true, "bf16");

  const std::array<U32, 4> prompt{9U, 34U, 72U, 118U};
  GpuSamplingOptions samp{.temperature = 0.0F,
                          .top_k = 1U,
                          .top_p = 1.0F,
                          .contract = SamplingContract::PositionCoupledV1};
  const auto res = fwd.run(prompt, samp, /*inspect_tensors=*/true);
  const auto ref = oracle.forward(prompt);
  const double cos = cosine_similarity(res.logits, ref.logits);
  const double mad = max_abs_diff(res.logits, ref.logits);
  require(cos > 0.9999 && mad < 0.05,
          "mixed-precision + direct_f16_v1 cos=" + std::to_string(cos) +
              " mad=" + std::to_string(mad));
}

// ============================================================================
// Suite 5: QwenForward::supports_config() Boundaries & mseries_backendd --describe
// ============================================================================
void test_supports_config_and_backendd_describe(
    const std::filesystem::path &backendd_bin) {
  std::cout << "[5/5] Testing supports_config boundaries & mseries_backendd --describe...\n";

  QwenModelConfig base;
  base.model_type = "custom_arch";
  base.hidden_size = 256;
  base.intermediate_size = 512;
  base.vocab_size = 32000;
  base.num_hidden_layers = 4;
  base.layer_types = {LayerType::full_attention, LayerType::full_attention,
                      LayerType::full_attention, LayerType::full_attention};
  base.num_attention_heads = 4;
  base.num_key_value_heads = 2;
  base.head_dim = 64;
  base.partial_rotary_factor = 1.0;
  base.rms_norm_eps = 1e-6;
  base.rope_theta = 10000.0;
  // 0-GDN with all linear_* == 0 MUST pass!
  base.linear_num_key_heads = 0;
  base.linear_num_value_heads = 0;
  base.linear_key_head_dim = 0;
  base.linear_value_head_dim = 0;
  base.linear_conv_kernel_dim = 0;

  std::string reason;
  require(QwenForward::supports_config(base, &reason) && reason.empty(),
          "base 0-GDN config must be supported");

  {
    auto bad = base;
    bad.hidden_size = 224; // not multiple of 64
    require(!QwenForward::supports_config(bad, &reason) && !reason.empty(),
            "hidden_size % 64 != 0 must fail");
  }
  {
    auto bad = base;
    bad.head_dim = 512; // > 256
    require(!QwenForward::supports_config(bad, &reason),
            "head_dim > 256 must fail");
  }
  {
    auto bad = base;
    bad.layer_types[0] = LayerType::linear_attention; // >0 GDN with linear_*=0
    require(!QwenForward::supports_config(bad, &reason),
            "GDN layer with linear_*=0 must fail");
  }
  {
    auto bad = base;
    bad.num_experts = 1; // must be 0 or 2..256
    require(!QwenForward::supports_config(bad, &reason),
            "num_experts == 1 must fail");
  }
  {
    auto bad = base;
    bad.num_experts = 512; // > 256
    bad.num_experts_per_token = 8;
    bad.intermediate_size = 128;
    require(!QwenForward::supports_config(bad, &reason),
            "num_experts > 256 must fail");
  }
  {
    auto bad = base;
    bad.num_experts = 4;
    bad.num_experts_per_token = 6; // > num_experts
    bad.intermediate_size = 128;
    require(!QwenForward::supports_config(bad, &reason),
            "top_k > num_experts must fail");
  }
  {
    auto bad = base;
    bad.num_experts = 16;
    bad.num_experts_per_token = 2;
    bad.intermediate_size = 128;
    bad.shared_expert_intermediate_size = 32; // not multiple of 64
    require(!QwenForward::supports_config(bad, &reason),
            "shared_expert_intermediate_size % 64 != 0 must fail");
  }

  // Run mseries_backendd --describe via POSIX pipe and inspect JSON contract
  int pipefd[2]{-1, -1};
  require(::pipe(pipefd) == 0, "pipe() failed");
  const pid_t pid = ::fork();
  require(pid >= 0, "fork() failed");
  if (pid == 0) {
    ::close(pipefd[0]);
    ::dup2(pipefd[1], STDOUT_FILENO);
    ::close(pipefd[1]);
    ::execl(backendd_bin.c_str(), backendd_bin.c_str(), "--describe",
            static_cast<char *>(nullptr));
    std::_Exit(127);
  }
  ::close(pipefd[1]);
  std::string out_str;
  std::array<char, 4096> buf{};
  for (;;) {
    const ssize_t n = ::read(pipefd[0], buf.data(), buf.size());
    if (n <= 0) {
      break;
    }
    out_str.append(buf.data(), static_cast<std::size_t>(n));
  }
  ::close(pipefd[0]);
  int status = 0;
  require(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
              WEXITSTATUS(status) == 0,
          "mseries_backendd --describe exited non-zero");

  const Json desc = Json::parse(out_str);
  require(desc.contains("architecture_capabilities"),
          "--describe missing architecture_capabilities");
  const auto &ac = desc.at("architecture_capabilities");
  require(ac.at("topologies") ==
              Json::array({"gdn_hybrid_gqa", "pure_gqa_swiglu", "dflash2_draft"}),
          "topologies mismatch");
  require(!ac.at("model_type_whitelist_enforced").get<bool>(),
          "model_type_whitelist_enforced must be false");
  require(ac.at("attn_output_gate") == Json::array({true, false}),
          "attn_output_gate mismatch");
  require(ac.at("qk_norm") == Json::array({true, false}), "qk_norm mismatch");
  require(ac.at("head_dims") == Json::array({64, 128, 256}),
          "head_dims mismatch");
  require(ac.at("moe").at("min_experts").get<int>() == 2 &&
              ac.at("moe").at("max_experts").get<int>() == 256 &&
              ac.at("moe").at("min_top_k").get<int>() == 1 &&
              ac.at("moe").at("max_top_k").get<int>() == 8 &&
              ac.at("moe").at("shared_expert").get<std::string>() == "optional",
          "moe capabilities contract mismatch");
  const auto &qc = ac.at("quantization");
  require(qc.at("affine_bits") == Json::array({3, 4, 5, 6, 8}) &&
              qc.at("affine_group_sizes") == Json::array({64, 128}) &&
              qc.at("mxfp4_group_sizes") == Json::array({32}) &&
              qc.at("direct_dtypes") == Json::array({"bf16", "f16"}) &&
              qc.at("per_tensor_mixed_precision").get<bool>() &&
              qc.at("strip_non_text_vlm").get<bool>(),
          "quantization capabilities contract mismatch");
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc >= 3,
            "usage: omlx_m5_universal_capability_tests <metallib_dir> <mseries_backendd>");
    const std::filesystem::path metallib_dir = argv[1];
    const std::filesystem::path backendd_bin = argv[2];

    const TempDirGuard tmp_guard{
        std::filesystem::temp_directory_path() /
        ("omlx_m5_universal_cap_" + std::to_string(::getpid()))};
    std::filesystem::create_directories(tmp_guard.path);

    MetalContext ctx = MetalContext::create();
    test_config_generalization_and_tied_lm_head(ctx, metallib_dir,
                                                tmp_guard.path);
    test_pure_gqa_ungated_optional_qknorm_and_kv_guard(ctx, metallib_dir,
                                                       tmp_guard.path);
    test_generalized_moe_execution(ctx, metallib_dir, tmp_guard.path);
    test_mixed_precision_and_direct_f16_execution(ctx, metallib_dir,
                                                  tmp_guard.path);
    test_supports_config_and_backendd_describe(backendd_bin);

    std::cout << "ALL UNIVERSAL CAPABILITY TESTS PASSED\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "ERROR: " << ex.what() << "\n";
    return 1;
  }
}
