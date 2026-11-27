#include "affine_nax_cache.h"
#include "lora_runtime.h"
#include "mxfp4_nax_cache.h"
#include "omlx_m5/artifact.h"
#include "omlx_m5/dflash2.h"
#include "omlx_m5/dflash_forward.h"
#include "omlx_m5/engine.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/lora.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/qwen_forward.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {

using namespace omlx::m5;
using Json = nlohmann::json;
using U32 = std::uint32_t;
using U64 = std::uint64_t;

void require(bool cond, std::string_view msg) {
  if (!cond) {
    throw std::runtime_error("LORA-DUAL-MODE TEST FAIL: " + std::string(msg));
  }
}

std::uint16_t f32_to_bf16(float v) {
  const U32 bits = std::bit_cast<U32>(v);
  const U32 lsb = (bits >> 16U) & 1U;
  return static_cast<std::uint16_t>((bits + 0x7FFFU + lsb) >> 16U);
}

float bf16_to_f32(std::uint16_t b) {
  return std::bit_cast<float>(static_cast<U32>(b) << 16U);
}

float round_bf16(float v) { return bf16_to_f32(f32_to_bf16(v)); }

std::uint16_t f32_to_fp16(float value) {
  const U32 f = std::bit_cast<U32>(value);
  const U32 sign = (f >> 16U) & 0x8000U;
  const std::int32_t exp = static_cast<std::int32_t>((f >> 23U) & 0xFFU) - 127 + 15;
  U32 mant = f & 0x007FFFFFU;
  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    mant = (mant | 0x00800000U) >> static_cast<U32>(1 - exp);
    return static_cast<std::uint16_t>(sign | ((mant + 0x00001000U) >> 13U));
  }
  if (exp >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<U32>(exp) << 10U) | ((mant + 0x00001000U) >> 13U));
}

float det_val(U32 seed, U64 idx, float amp = 0.08F) {
  U64 x = (U64(seed) * 0x9E3779B97F4A7C15ULL) ^ (idx * 0xBF58476D1CE4E5B9ULL);
  x = (x ^ (x >> 30U)) * 0x94D049BB133111EBULL;
  x = x ^ (x >> 31U);
  const float u = static_cast<float>(static_cast<std::int32_t>(x & 0xFFFFU) - 32768) / 32768.0F;
  return u * amp;
}

float decode_e2m1(U32 code) {
  constexpr std::array<float, 8> kMag{0.0F, 0.5F, 1.0F, 1.5F,
                                      2.0F, 3.0F, 4.0F, 6.0F};
  const float v = kMag[code & 7U];
  return (code & 8U) ? -v : v;
}

float decode_e8m0(std::uint8_t sc) {
  return std::ldexp(1.0F, sc == 0U ? -127 : static_cast<int>(sc) - 127);
}

std::string hex_crc32(U32 crc) {
  std::ostringstream out;
  out << std::hex << std::nouppercase << std::setfill('0') << std::setw(8) << crc;
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
  const auto bytes = std::as_bytes(std::span<const char>(text.data(), text.size()));
  write_binary_file(path, bytes);
}

// Minimal single-file .safetensors writer supporting BF16, F16, F32.
struct StTensorSpec {
  std::string name;
  std::string dtype; // "BF16", "F16", "F32"
  std::vector<U64> shape;
  std::vector<float> values;
};

void write_safetensors_file(const std::filesystem::path &path,
                            std::span<const StTensorSpec> specs) {
  std::vector<StTensorSpec> sorted(specs.begin(), specs.end());
  std::ranges::sort(sorted, {}, &StTensorSpec::name);
  std::vector<std::byte> body;
  Json header = Json::object();
  for (const auto &sp : sorted) {
    U64 count = 1;
    for (U64 d : sp.shape) {
      count *= d;
    }
    require(sp.values.size() == count,
            "StTensorSpec element count mismatch for " + sp.name);
    const U64 begin_off = body.size();
    if (sp.dtype == "F32") {
      const auto raw = std::as_bytes(std::span<const float>(sp.values));
      body.insert(body.end(), raw.begin(), raw.end());
    } else if (sp.dtype == "BF16") {
      std::vector<std::uint16_t> tmp(count);
      for (std::size_t i = 0; i < count; ++i) {
        tmp[i] = f32_to_bf16(sp.values[i]);
      }
      const auto raw = std::as_bytes(std::span<const std::uint16_t>(tmp));
      body.insert(body.end(), raw.begin(), raw.end());
    } else if (sp.dtype == "F16") {
      std::vector<std::uint16_t> tmp(count);
      for (std::size_t i = 0; i < count; ++i) {
        tmp[i] = f32_to_fp16(sp.values[i]);
      }
      const auto raw = std::as_bytes(std::span<const std::uint16_t>(tmp));
      body.insert(body.end(), raw.begin(), raw.end());
    } else {
      throw std::invalid_argument("unsupported test safetensor dtype: " + sp.dtype);
    }
    const U64 end_off = body.size();
    header[sp.name] = {
        {"dtype", sp.dtype},
        {"shape", sp.shape},
        {"data_offsets", {begin_off, end_off}},
    };
  }
  std::string hdr_str = header.dump();
  while ((hdr_str.size() % 8U) != 0U) {
    hdr_str.push_back(' ');
  }
  std::vector<std::byte> file;
  append_u64(file, hdr_str.size());
  const auto hdr_bytes = std::as_bytes(std::span<const char>(hdr_str.data(), hdr_str.size()));
  file.insert(file.end(), hdr_bytes.begin(), hdr_bytes.end());
  file.insert(file.end(), body.begin(), body.end());
  write_binary_file(path, file);
}

// Artifact builder producing canonical .miv / .omlxm5 files + exact FP32 reference weights.
enum class WeightEnc {
  DirectBf16,
  Mxfp4Tiled,
  Mxfp4Untiled,
  AffineQ4G64Tiled,
  AffineQ6G64Tiled,
  AffineQ8G64Tiled,
  AffineQ8G128Tiled,
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
  std::string binding_use; // empty if no binding entry
  std::string activation_policy;
};

class SyntheticArtifactBuilder {
public:
  std::map<std::string, std::vector<float>> ref_weights;

  U32 add_bf16_1d_or_nd(std::string name, std::vector<U64> shape, U32 seed,
                        float base_bias = 0.0F, float amp = 0.05F,
                        std::string binding_use = "") {
    U64 count = 1;
    for (U64 d : shape) {
      count *= d;
    }
    std::vector<std::uint16_t> bf(count);
    std::vector<float> ref(count);
    for (U64 i = 0; i < count; ++i) {
      const float v = base_bias + det_val(seed, i, amp);
      bf[i] = f32_to_bf16(v);
      ref[i] = bf16_to_f32(bf[i]);
    }
    ref_weights[name] = std::move(ref);
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
                    std::string use = "linear", float amp = 0.08F) {
    const U32 id = static_cast<U32>(objects_.size());
    if (enc == WeightEnc::DirectBf16) {
      return add_bf16_1d_or_nd(std::move(name), {n, k}, seed, 0.0F, amp,
                               std::move(use));
    }
    const U32 padded_k = ((k + 127U) / 128U) * 128U;
    std::vector<float> ref(std::size_t(n) * k, 0.0F);
    BuiltObject obj{
        .id = id,
        .name = name,
        .shape = {n, k},
        .padded_k = padded_k,
        .binding_use = std::move(use),
    };

    if (enc == WeightEnc::Mxfp4Tiled || enc == WeightEnc::Mxfp4Untiled) {
      obj.format = "mxfp4_e2m1_g32_e8m0_v1";
      obj.activation_policy = "AllowA4";
      const U32 bn = (n >= 65536U) ? 32U : 16U;
      const U32 padded_n = ((n + bn - 1U) / bn) * bn;
      if (enc == WeightEnc::Mxfp4Tiled) {
        obj.layout = "mx_nax_tiled_bk128_v1";
        const U32 k_blocks = k / 128U;
        std::vector<std::byte> nax_data(std::size_t(padded_n) * (k / 2U),
                                        std::byte{0});
        std::vector<std::byte> nax_scale(std::size_t(padded_n) * (k / 32U),
                                         std::byte{120});
        for (U32 r = 0; r < n; ++r) {
          const U32 tn = r / bn;
          const U32 rn = r % bn;
          for (U32 blk = 0; blk < k_blocks; ++blk) {
            const std::size_t tile_idx = std::size_t(tn) * k_blocks + blk;
            for (U32 sub = 0; sub < 4U; ++sub) {
              const std::uint8_t sc_code =
                  static_cast<std::uint8_t>(118U + ((seed + r + blk + sub) % 4U));
              nax_scale[tile_idx * (bn * 4U) + rn * 4U + sub] =
                  static_cast<std::byte>(sc_code);
              const float sc = decode_e8m0(sc_code);
              for (U32 j = 0; j < 16U; ++j) {
                const U32 c0 = blk * 128U + sub * 32U + 2U * j;
                const U32 q0 = (seed * 17U + r * 31U + c0 * 7U) & 0xFU;
                const U32 q1 = (seed * 23U + r * 13U + (c0 + 1U) * 11U) & 0xFU;
                nax_data[tile_idx * (bn * 64U) + rn * 64U + sub * 16U + j] =
                    static_cast<std::byte>(q0 | (q1 << 4U));
                ref[std::size_t(r) * k + c0] = decode_e2m1(q0) * sc;
                ref[std::size_t(r) * k + c0 + 1U] = decode_e2m1(q1) * sc;
              }
            }
          }
        }
        obj.planes.push_back({"nax_data", std::move(nax_data)});
        obj.planes.push_back({"nax_scale", std::move(nax_scale)});
      } else {
        obj.layout = "mx_rowsplit_k128_v1";
        std::vector<std::byte> data(std::size_t(n) * (padded_k / 2U),
                                    std::byte{0});
        std::vector<std::byte> scale(std::size_t(n) * (padded_k / 32U),
                                     std::byte{120});
        for (U32 r = 0; r < n; ++r) {
          for (U32 g = 0; g < k / 32U; ++g) {
            const std::uint8_t sc_code =
                static_cast<std::uint8_t>(118U + ((seed + r + g) % 4U));
            scale[std::size_t(r) * (padded_k / 32U) + g] =
                static_cast<std::byte>(sc_code);
            const float sc = decode_e8m0(sc_code);
            for (U32 j = 0; j < 16U; ++j) {
              const U32 c0 = g * 32U + 2U * j;
              const U32 q0 = (seed * 19U + r * 29U + c0 * 5U) & 0xFU;
              const U32 q1 = (seed * 11U + r * 17U + (c0 + 1U) * 3U) & 0xFU;
              data[std::size_t(r) * (padded_k / 2U) + g * 16U + j] =
                  static_cast<std::byte>(q0 | (q1 << 4U));
              ref[std::size_t(r) * k + c0] = decode_e2m1(q0) * sc;
              ref[std::size_t(r) * k + c0 + 1U] = decode_e2m1(q1) * sc;
            }
          }
        }
        obj.planes.push_back({"data", std::move(data)});
        obj.planes.push_back({"scale", std::move(scale)});
      }
    } else {
      U32 bits = 8U;
      U32 gs = 64U;
      bool tiled = true;
      if (enc == WeightEnc::AffineQ4G64Tiled) {
        bits = 4U;
        gs = 64U;
      } else if (enc == WeightEnc::AffineQ6G64Tiled) {
        bits = 6U;
        gs = 64U;
      } else if (enc == WeightEnc::AffineQ8G64Tiled) {
        bits = 8U;
        gs = 64U;
      } else if (enc == WeightEnc::AffineQ8G128Tiled) {
        bits = 8U;
        gs = 128U;
      } else if (enc == WeightEnc::AffineQ8G128Untiled) {
        bits = 8U;
        gs = 128U;
        tiled = false;
      }
      obj.format = "oq_affine_q" + std::to_string(bits) + "_g" +
                   std::to_string(gs) + "_bf16_v1";
      obj.activation_policy = "A16Only";
      const U32 max_q = (1U << bits) - 1U;
      if (tiled) {
        obj.layout = "oq_nax_tiled_g64_v1";
        const U32 bn = (n >= 65536U) ? 32U : 16U;
        const U32 padded_n = ((n + bn - 1U) / bn) * bn;
        const U32 groups64 = k / 64U;
        const U32 sb_groups = k / gs;
        std::vector<std::byte> nax_codes(
            bits == 8U ? (std::size_t(padded_n) * k)
                       : (std::size_t(padded_n) * (k / 2U)),
            std::byte{0});
        std::vector<std::byte> nax_high2(
            bits == 6U ? (std::size_t(padded_n) * (k / 4U)) : 0U,
            std::byte{0});
        std::vector<std::byte> nax_sb(std::size_t(groups64) * padded_n * 4U,
                                      std::byte{0});
        for (U32 r = 0; r < n; ++r) {
          for (U32 sg = 0; sg < sb_groups; ++sg) {
            const float span = 2.0F * amp * (0.6F + 0.1F * float((r + sg + seed) % 5U));
            const std::uint16_t sc_bf = f32_to_bf16(span / float(max_q));
            const std::uint16_t bi_bf = f32_to_bf16(-0.5F * span);
            const U32 packed_sb =
                static_cast<U32>(sc_bf) | (static_cast<U32>(bi_bf) << 16U);
            if (gs == 128U) {
              std::memcpy(
                  nax_sb.data() + (std::size_t(2U * sg) * padded_n + r) * 4U,
                  &packed_sb, 4U);
              std::memcpy(nax_sb.data() +
                              (std::size_t(2U * sg + 1U) * padded_n + r) * 4U,
                          &packed_sb, 4U);
            } else {
              std::memcpy(nax_sb.data() + (std::size_t(sg) * padded_n + r) * 4U,
                          &packed_sb, 4U);
            }
          }
          const U32 tn = r / bn;
          const U32 rn = r % bn;
          for (U32 col = 0; col < k; ++col) {
            const U32 q =
                (seed * 29U + r * 41U + col * 17U + (col >> 3U)) & max_q;
            const U32 g64 = col / 64U;
            const U32 off = col % 64U;
            const std::size_t tile_idx = std::size_t(tn) * groups64 + g64;
            if (bits == 8U) {
              nax_codes[tile_idx * (bn * 64U) + rn * 64U + off] =
                  static_cast<std::byte>(q);
            } else {
              const std::size_t cb_idx =
                  tile_idx * (bn * 32U) + rn * 32U + (off >> 1U);
              const std::uint8_t cur = std::to_integer<std::uint8_t>(nax_codes[cb_idx]);
              nax_codes[cb_idx] = static_cast<std::byte>(
                  cur | ((q & 0xFU) << ((off & 1U) * 4U)));
              if (bits == 6U) {
                const std::size_t hb_idx =
                    tile_idx * (bn * 16U) + rn * 16U + (off >> 2U);
                const std::uint8_t hcur =
                    std::to_integer<std::uint8_t>(nax_high2[hb_idx]);
                nax_high2[hb_idx] = static_cast<std::byte>(
                    hcur | (((q >> 4U) & 0x3U) << ((off & 3U) * 2U)));
              }
            }
            U32 packed_sb = 0;
            std::memcpy(&packed_sb,
                        nax_sb.data() + (std::size_t(g64) * padded_n + r) * 4U,
                        4U);
            const float sc =
                bf16_to_f32(static_cast<std::uint16_t>(packed_sb & 0xFFFFU));
            const float bi =
                bf16_to_f32(static_cast<std::uint16_t>(packed_sb >> 16U));
            ref[std::size_t(r) * k + col] = std::fma(sc, float(q), bi);
          }
        }
        obj.planes.push_back({"nax_codes", std::move(nax_codes)});
        if (bits == 6U) {
          obj.planes.push_back({"nax_high2", std::move(nax_high2)});
        }
        obj.planes.push_back({"nax_sb", std::move(nax_sb)});
      } else {
        obj.layout = "oq_rowsplit_k128_v1";
        const U32 padded_groups = padded_k / gs;
        std::vector<std::byte> code(std::size_t(n) * padded_k, std::byte{0});
        std::vector<std::uint16_t> scale(std::size_t(n) * padded_groups, 0U);
        std::vector<std::uint16_t> bias(std::size_t(n) * padded_groups, 0U);
        for (U32 r = 0; r < n; ++r) {
          for (U32 sg = 0; sg < k / gs; ++sg) {
            const float span = 2.0F * amp * (0.6F + 0.1F * float((r + sg + seed) % 5U));
            scale[std::size_t(r) * padded_groups + sg] =
                f32_to_bf16(span / 255.0F);
            bias[std::size_t(r) * padded_groups + sg] =
                f32_to_bf16(-0.5F * span);
          }
          for (U32 col = 0; col < k; ++col) {
            const U32 q = (seed * 31U + r * 19U + col * 23U) & 0xFFU;
            code[std::size_t(r) * padded_k + col] = static_cast<std::byte>(q);
            const U32 sg = col / gs;
            const float sc =
                bf16_to_f32(scale[std::size_t(r) * padded_groups + sg]);
            const float bi =
                bf16_to_f32(bias[std::size_t(r) * padded_groups + sg]);
            ref[std::size_t(r) * k + col] = std::fma(sc, float(q), bi);
          }
        }
        const auto s_raw = std::as_bytes(std::span<const std::uint16_t>(scale));
        const auto b_raw = std::as_bytes(std::span<const std::uint16_t>(bias));
        obj.planes.push_back({"code", std::move(code)});
        obj.planes.push_back({"scale", {s_raw.begin(), s_raw.end()}});
        obj.planes.push_back({"bias", {b_raw.begin(), b_raw.end()}});
      }
    }

    ref_weights[name] = std::move(ref);
    objects_.push_back(std::move(obj));
    return id;
  }

  std::vector<std::byte> serialize(std::string_view source_profile = "oq_mixed",
                                   std::string_view source_repo = "test/qwen-lora",
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
          obj.shape.size() == 2U) {
        j_bindings.push_back({
            {"activation_policy", obj.activation_policy},
            {"format", obj.format},
            {"in_features", obj.shape[1]},
            {"leading_shape", Json::array()},
            {"name", obj.name},
            {"object_id", obj.id},
            {"out_features", obj.shape[0]},
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
        {"schema", "omlx-m5-artifact-v1"},
        {"views", Json::array()},
    };
    const std::string manifest_str = manifest.dump();
    const auto manifest_bytes =
        std::as_bytes(std::span<const char>(manifest_str.data(), manifest_str.size()));
    const U64 raw_end = 64ULL + manifest_bytes.size();
    const U64 payload_offset = (raw_end + 16383ULL) & ~16383ULL;
    const U32 manifest_crc = artifact_crc32(manifest_bytes);

    std::vector<std::byte> file;
    constexpr char kMagic[8] = {'O', 'M', 'L', 'X', 'M', '5', 'A', '1'};
    for (char c : kMagic) {
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

struct QwenFixtureSpec {
  bool pure_mxfp4 = false;
  bool pure_oq = false;
  bool include_embedded_mtp = false;
};

struct BuiltQwenFixture {
  QwenModelConfig config;
  std::filesystem::path model_dir;
  std::filesystem::path artifact_path;
  std::map<std::string, std::vector<float>> ref_weights;
};

BuiltQwenFixture build_qwen_fixture(const std::filesystem::path &dir,
                                    const QwenFixtureSpec &spec) {
  std::filesystem::create_directories(dir);
  Json cfg = {
      {"model_type", "qwen3_5"},
      {"text_config",
       {{"model_type", "qwen3_5_text"},
        {"dtype", "bfloat16"},
        {"hidden_size", 512},
        {"intermediate_size", 512},
        {"vocab_size", 256},
        {"num_hidden_layers", 2},
        {"num_attention_heads", 4},
        {"num_key_value_heads", 2},
        {"head_dim", 128},
        {"max_position_embeddings", 2048},
        {"rms_norm_eps", 1e-6},
        {"rope_parameters",
         {{"rope_theta", 10000.0},
          {"partial_rotary_factor", 0.25},
          {"mrope_interleaved", true}}},
        {"linear_num_key_heads", 4},
        {"linear_num_value_heads", 4},
        {"linear_key_head_dim", 128},
        {"linear_value_head_dim", 128},
        {"linear_conv_kernel_dim", 4},
        {"layer_types", {"full_attention", "linear_attention"}},
        {"tie_word_embeddings", false}}},
  };
  write_text_file(dir / "config.json", cfg.dump(2));

  SyntheticArtifactBuilder b;
  const U32 h = 512U;
  const U32 f = 512U;
  const U32 v = 256U;
  const U32 q = 512U;  // 4 * 128
  const U32 kv = 256U; // 2 * 128
  const U32 gk = 512U;
  const U32 gv = 512U;
  const U32 conv = 2U * gk + gv; // 1536
  const U32 hv = 4U;

  b.add_bf16_1d_or_nd("model.embed_tokens.weight", {v, h}, 101U, 0.0F, 0.12F,
                      "embedding");
  b.add_bf16_1d_or_nd("model.norm.weight", {h}, 102U, 0.0F, 0.05F);
  b.add_linear_2d("lm_head.weight", v, h,
                  spec.pure_mxfp4 ? WeightEnc::Mxfp4Tiled
                                  : WeightEnc::AffineQ8G64Tiled,
                  103U, "lm_head", 0.08F);

  // Layer 0: full_attention + dense MLP
  b.add_bf16_1d_or_nd("model.layers.0.input_layernorm.weight", {h}, 201U, 0.0F,
                      0.05F);
  b.add_linear_2d("model.layers.0.self_attn.q_proj.weight", 2U * q, h,
                  spec.pure_oq      ? WeightEnc::AffineQ4G64Tiled
                  : spec.pure_mxfp4 ? WeightEnc::Mxfp4Tiled
                                    : WeightEnc::AffineQ8G64Tiled,
                  202U);
  b.add_linear_2d("model.layers.0.self_attn.k_proj.weight", kv, h,
                  spec.pure_oq ? WeightEnc::AffineQ6G64Tiled
                               : WeightEnc::Mxfp4Tiled,
                  203U);
  b.add_linear_2d("model.layers.0.self_attn.v_proj.weight", kv, h,
                  spec.pure_oq ? WeightEnc::AffineQ8G64Tiled
                               : WeightEnc::Mxfp4Tiled,
                  204U);
  b.add_bf16_1d_or_nd("model.layers.0.self_attn.q_norm.weight", {128U}, 205U,
                      0.0F, 0.05F);
  b.add_bf16_1d_or_nd("model.layers.0.self_attn.k_norm.weight", {128U}, 206U,
                      0.0F, 0.05F);
  b.add_linear_2d("model.layers.0.self_attn.o_proj.weight", h, q,
                  spec.pure_mxfp4 ? WeightEnc::Mxfp4Tiled
                                  : WeightEnc::AffineQ8G128Tiled,
                  207U);
  b.add_bf16_1d_or_nd("model.layers.0.post_attention_layernorm.weight", {h},
                      208U, 0.0F, 0.05F);
  b.add_linear_2d("model.layers.0.mlp.gate_proj.weight", f, h,
                  spec.pure_oq      ? WeightEnc::AffineQ4G64Tiled
                  : spec.pure_mxfp4 ? WeightEnc::Mxfp4Tiled
                                    : WeightEnc::AffineQ8G64Tiled,
                  209U);
  b.add_linear_2d("model.layers.0.mlp.up_proj.weight", f, h,
                  spec.pure_oq      ? WeightEnc::AffineQ6G64Tiled
                  : spec.pure_mxfp4 ? WeightEnc::Mxfp4Tiled
                                    : WeightEnc::AffineQ8G64Tiled,
                  210U);
  b.add_linear_2d("model.layers.0.mlp.down_proj.weight", h, f,
                  spec.pure_oq ? WeightEnc::AffineQ8G64Tiled
                               : WeightEnc::Mxfp4Tiled,
                  211U);

  // Layer 1: linear_attention (GDN) + dense MLP
  b.add_bf16_1d_or_nd("model.layers.1.input_layernorm.weight", {h}, 301U, 0.0F,
                      0.05F);
  b.add_linear_2d("model.layers.1.linear_attn.in_proj_qkv.weight", conv, h,
                  spec.pure_oq      ? WeightEnc::AffineQ4G64Tiled
                  : spec.pure_mxfp4 ? WeightEnc::Mxfp4Tiled
                                    : WeightEnc::AffineQ8G64Tiled,
                  302U);
  b.add_linear_2d("model.layers.1.linear_attn.in_proj_z.weight", gv, h,
                  spec.pure_oq      ? WeightEnc::AffineQ6G64Tiled
                  : spec.pure_mxfp4 ? WeightEnc::Mxfp4Untiled
                                    : WeightEnc::AffineQ8G64Tiled,
                  303U);
  b.add_linear_2d("model.layers.1.linear_attn.in_proj_a.weight", hv, h,
                  spec.pure_mxfp4 ? WeightEnc::Mxfp4Tiled
                                  : WeightEnc::AffineQ8G128Untiled,
                  304U);
  b.add_linear_2d("model.layers.1.linear_attn.in_proj_b.weight", hv, h,
                  spec.pure_mxfp4 ? WeightEnc::Mxfp4Tiled
                                  : WeightEnc::AffineQ8G128Untiled,
                  305U);
  // HF conv1d layout: [conv, 1, 4] (matches delta-centered RMSNorm offset +1.0)
  b.add_bf16_1d_or_nd("model.layers.1.linear_attn.conv1d.weight",
                      {conv, 1U, 4U}, 306U, 0.0F, 0.06F);
  b.add_bf16_1d_or_nd("model.layers.1.linear_attn.A_log", {hv}, 307U, 0.5F,
                      0.05F);
  b.add_bf16_1d_or_nd("model.layers.1.linear_attn.dt_bias", {hv}, 308U, -1.0F,
                      0.05F);
  // Note: GDN norm uses standard weight (around 1.0)
  b.add_bf16_1d_or_nd("model.layers.1.linear_attn.norm.weight", {128U}, 309U,
                      1.0F, 0.05F);
  b.add_linear_2d("model.layers.1.linear_attn.out_proj.weight", h, gv,
                  WeightEnc::AffineQ8G64Tiled, 310U);
  b.add_bf16_1d_or_nd("model.layers.1.post_attention_layernorm.weight", {h},
                      311U, 0.0F, 0.05F);
  b.add_linear_2d("model.layers.1.mlp.gate_proj.weight", f, h,
                  spec.pure_oq ? WeightEnc::AffineQ4G64Tiled
                               : WeightEnc::Mxfp4Tiled,
                  312U);
  b.add_linear_2d("model.layers.1.mlp.up_proj.weight", f, h,
                  spec.pure_oq ? WeightEnc::AffineQ6G64Tiled
                               : WeightEnc::Mxfp4Tiled,
                  313U);
  b.add_linear_2d("model.layers.1.mlp.down_proj.weight", h, f,
                  spec.pure_oq ? WeightEnc::AffineQ8G64Tiled
                               : WeightEnc::Mxfp4Tiled,
                  314U);

  if (spec.include_embedded_mtp) {
    b.add_bf16_1d_or_nd("mtp.pre_fc_norm_embedding.weight", {h}, 401U, 0.0F,
                        0.05F);
    b.add_bf16_1d_or_nd("mtp.pre_fc_norm_hidden.weight", {h}, 402U, 0.0F,
                        0.05F);
    b.add_linear_2d("mtp.fc.weight", h, 2U * h, WeightEnc::Mxfp4Tiled, 403U);
    b.add_bf16_1d_or_nd("mtp.layers.0.input_layernorm.weight", {h}, 404U, 0.0F,
                        0.05F);
    b.add_linear_2d("mtp.layers.0.self_attn.q_proj.weight", 2U * q, h,
                    WeightEnc::AffineQ8G64Tiled, 405U);
    b.add_linear_2d("mtp.layers.0.self_attn.k_proj.weight", kv, h,
                    WeightEnc::AffineQ8G64Tiled, 406U);
    b.add_linear_2d("mtp.layers.0.self_attn.v_proj.weight", kv, h,
                    WeightEnc::AffineQ8G64Tiled, 407U);
    b.add_bf16_1d_or_nd("mtp.layers.0.self_attn.q_norm.weight", {128U}, 408U,
                        0.0F, 0.05F);
    b.add_bf16_1d_or_nd("mtp.layers.0.self_attn.k_norm.weight", {128U}, 409U,
                        0.0F, 0.05F);
    b.add_linear_2d("mtp.layers.0.self_attn.o_proj.weight", h, q,
                    WeightEnc::Mxfp4Tiled, 410U);
    b.add_bf16_1d_or_nd("mtp.layers.0.post_attention_layernorm.weight", {h},
                        411U, 0.0F, 0.05F);
    b.add_linear_2d("mtp.layers.0.mlp.gate_proj.weight", f, h,
                    WeightEnc::Mxfp4Tiled, 412U);
    b.add_linear_2d("mtp.layers.0.mlp.up_proj.weight", f, h,
                    WeightEnc::Mxfp4Tiled, 413U);
    b.add_linear_2d("mtp.layers.0.mlp.down_proj.weight", h, f,
                    WeightEnc::Mxfp4Tiled, 414U);
    b.add_bf16_1d_or_nd("mtp.norm.weight", {h}, 415U, 0.0F, 0.05F);
  }

  const auto artifact_path = dir / "model.miv";
  write_binary_file(artifact_path, b.serialize());
  return BuiltQwenFixture{
      .config = load_qwen_model_config(dir / "config.json"),
      .model_dir = dir,
      .artifact_path = artifact_path,
      .ref_weights = std::move(b.ref_weights),
  };
}

std::filesystem::path build_dflash2_companion(const std::filesystem::path &dir) {
  std::filesystem::create_directories(dir);
  Json cfg = {
      {"architectures", {"DFlash2DraftModel"}},
      {"model_type", "qwen3"},
      {"is_causal", false},
      {"hidden_size", 512},
      {"intermediate_size", 512},
      {"vocab_size", 256},
      {"num_hidden_layers", 1},
      {"num_attention_heads", 4},
      {"num_key_value_heads", 2},
      {"head_dim", 128},
      {"max_position_embeddings", 2048},
      {"sliding_window", 512},
      {"rms_norm_eps", 1e-6},
      {"rope_parameters", {{"rope_theta", 10000.0}}},
      {"layer_types", {"sliding_attention"}},
      {"dflash_config",
       {{"block_size", 8},
        {"conv_group_size", 64},
        {"conv_kernel_size", 2},
        {"selector_rank", 16},
        {"selector_top_k", 32},
        {"mask_token_id", 0},
        {"target_layer_ids", {0, 1}},
        {"use_bidirectional_attention", true}}},
  };
  write_text_file(dir / "config.json", cfg.dump(2));

  SyntheticArtifactBuilder b;
  const U32 h = 512U;
  const U32 inter = 512U;
  const U32 v = 256U;
  const U32 q = 512U;
  const U32 kv = 256U;
  const U32 rank = 16U;
  const U32 groups = h / 64U; // 8

  b.add_bf16_1d_or_nd("dflash2.fc.weight", {h, 2U * h}, 501U, 0.0F, 0.04F);
  b.add_bf16_1d_or_nd("dflash2.hidden_norm.weight", {h}, 502U, 1.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.norm.weight", {h}, 503U, 1.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.candidate_selector.hidden_projection.weight",
                      {rank, h}, 504U, 0.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.candidate_selector.predecessor_codebook",
                      {v, rank}, 505U, 0.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.candidate_selector.successor_codebook",
                      {v, rank}, 506U, 0.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.layers.0.input_layernorm.weight", {h}, 507U,
                      1.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.layers.0.post_attention_layernorm.weight", {h},
                      508U, 1.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.layers.0.self_attn.q_proj.weight", {q, h}, 509U,
                      0.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.layers.0.self_attn.k_proj.weight", {kv, h}, 510U,
                      0.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.layers.0.self_attn.v_proj.weight", {kv, h}, 511U,
                      0.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.layers.0.self_attn.q_norm.weight", {128U}, 512U,
                      1.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.layers.0.self_attn.k_norm.weight", {128U}, 513U,
                      1.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.layers.0.self_attn.o_proj.weight", {h, q}, 514U,
                      0.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.layers.0.mlp.gate_proj.weight", {inter, h}, 515U,
                      0.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.layers.0.mlp.up_proj.weight", {inter, h}, 516U,
                      0.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.layers.0.mlp.down_proj.weight", {h, inter}, 517U,
                      0.0F, 0.05F);
  b.add_bf16_1d_or_nd("dflash2.layers.0.attention_conv.base_kernel",
                      {2U, 2U, h}, 518U, 0.0F, 0.02F);
  b.add_bf16_1d_or_nd(
      "dflash2.layers.0.attention_conv.kernel_projection.weight",
      {4U * groups, h}, 519U, 0.0F, 0.02F);
  b.add_bf16_1d_or_nd("dflash2.layers.0.mlp_conv.base_kernel", {2U, 2U, h},
                      520U, 0.0F, 0.02F);
  b.add_bf16_1d_or_nd("dflash2.layers.0.mlp_conv.kernel_projection.weight",
                      {4U * groups, h}, 521U, 0.0F, 0.02F);

  const auto path = dir / "model.miv";
  write_binary_file(path, b.serialize("dflash2_bf16", "test/dflash2", "rev1"));
  return dir;
}

// Exact CPU FP32 Qwen3.5/3.8 forward reference over W_eff = W_base + scale * B @ A.
struct CpuQwenRef {
  const QwenModelConfig &cfg;
  std::map<std::string, std::vector<float>> w;

  // Stateful KV & GDN caches for sequential T=1 / multi-step reference checks
  U32 pos = 0;
  std::vector<float> k_cache; // [max_ctx, kvh * hd]
  std::vector<float> v_cache; // [max_ctx, kvh * hd]
  std::vector<float> gdn_hist; // [3, conv]
  std::vector<float> gdn_state; // [hv, vd, kd]

  CpuQwenRef(const QwenModelConfig &c,
             std::map<std::string, std::vector<float>> base_w,
             const LoadedLoraAdapter *lora = nullptr)
      : cfg(c), w(std::move(base_w)) {
    const U32 kv_dim = U32(cfg.num_key_value_heads * cfg.head_dim);
    const U32 gk = U32(cfg.linear_num_key_heads * cfg.linear_key_head_dim);
    const U32 gv = U32(cfg.linear_num_value_heads * cfg.linear_value_head_dim);
    const U32 conv = 2U * gk + gv;
    k_cache.assign(256U * kv_dim, 0.0F);
    v_cache.assign(256U * kv_dim, 0.0F);
    gdn_hist.assign(std::size_t(cfg.linear_conv_kernel_dim - 1) * conv, 0.0F);
    gdn_state.assign(std::size_t(gv) * U32(cfg.linear_key_head_dim), 0.0F);

    if (lora != nullptr) {
      for (const auto &u : lora->updates) {
        std::string wname;
        if (u.target.is_global) {
          require(u.target.role == "lm_head", "unexpected global LoRA role");
          wname = "lm_head.weight";
        } else {
          const std::string pfx =
              "model.layers." + std::to_string(u.target.layer_index) + ".";
          const auto &r = u.target.role;
          if (r == "attention_q")
            wname = pfx + "self_attn.q_proj.weight";
          else if (r == "attention_k")
            wname = pfx + "self_attn.k_proj.weight";
          else if (r == "attention_v")
            wname = pfx + "self_attn.v_proj.weight";
          else if (r == "attention_o")
            wname = pfx + "self_attn.o_proj.weight";
          else if (r == "gdn_qkv")
            wname = pfx + "linear_attn.in_proj_qkv.weight";
          else if (r == "gdn_z")
            wname = pfx + "linear_attn.in_proj_z.weight";
          else if (r == "gdn_a")
            wname = pfx + "linear_attn.in_proj_a.weight";
          else if (r == "gdn_b")
            wname = pfx + "linear_attn.in_proj_b.weight";
          else if (r == "gdn_out")
            wname = pfx + "linear_attn.out_proj.weight";
          else if (r == "mlp_gate")
            wname = pfx + "mlp.gate_proj.weight";
          else if (r == "mlp_up")
            wname = pfx + "mlp.up_proj.weight";
          else if (r == "mlp_down")
            wname = pfx + "mlp.down_proj.weight";
          else
            throw std::runtime_error("unhandled CPU ref LoRA role: " + r);
        }
        auto &mat = w.at(wname);
        require(mat.size() == std::size_t(u.out_features) * u.in_features,
                "CPU ref matrix size mismatch on " + wname);
        for (U32 row = 0; row < u.out_features; ++row) {
          for (U32 col = 0; col < u.in_features; ++col) {
            double dot = 0.0;
            for (U32 ri = 0; ri < u.rank; ++ri) {
              dot += double(u.b_f32[std::size_t(row) * u.rank + ri]) *
                     double(u.a_f32[std::size_t(ri) * u.in_features + col]);
            }
            mat[std::size_t(row) * u.in_features + col] +=
                static_cast<float>(double(u.scale) * dot);
          }
        }
      }
    }
  }

  static std::vector<float> rmsnorm(std::span<const float> x,
                                    std::span<const float> nw, U32 rows,
                                    U32 dim, float eps, float w_offset) {
    std::vector<float> out(std::size_t(rows) * dim);
    for (U32 r = 0; r < rows; ++r) {
      double sq = 0.0;
      for (U32 c = 0; c < dim; ++c) {
        const double v = round_bf16(x[std::size_t(r) * dim + c]);
        sq += v * v;
      }
      const float inv =
          1.0F / std::sqrt(static_cast<float>(sq / double(dim)) + eps);
      for (U32 c = 0; c < dim; ++c) {
        const float xv = round_bf16(x[std::size_t(r) * dim + c]);
        const float wv = nw[c] + w_offset;
        out[std::size_t(r) * dim + c] = round_bf16(xv * inv * wv);
      }
    }
    return out;
  }

  static std::vector<float> matmul_nt(std::span<const float> x,
                                      std::span<const float> weight, U32 rows,
                                      U32 n, U32 k,
                                      bool round_out_bf16 = true) {
    std::vector<float> y(std::size_t(rows) * n, 0.0F);
    for (U32 r = 0; r < rows; ++r) {
      for (U32 ni = 0; ni < n; ++ni) {
        double acc = 0.0;
        for (U32 ci = 0; ci < k; ++ci) {
          acc += double(round_bf16(x[std::size_t(r) * k + ci])) *
                 double(weight[std::size_t(ni) * k + ci]);
        }
        const float vf = static_cast<float>(acc);
        y[std::size_t(r) * n + ni] = round_out_bf16 ? round_bf16(vf) : vf;
      }
    }
    return y;
  }

  static float silu(float x) { return x / (1.0F + std::exp(-x)); }
  static float sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }
  static float softplus(float x) {
    return x > 20.0F ? x : std::log1p(std::exp(x));
  }

  // Returns all [T, V] logits in FP32 and advances stateful KV & GDN caches.
  std::vector<float> forward(std::span<const U32> tokens) {
    const U32 T = static_cast<U32>(tokens.size());
    const U32 h = static_cast<U32>(cfg.hidden_size);
    const U32 f = static_cast<U32>(cfg.intermediate_size);
    const U32 v = static_cast<U32>(cfg.vocab_size);
    const U32 qh = static_cast<U32>(cfg.num_attention_heads);
    const U32 kvh = static_cast<U32>(cfg.num_key_value_heads);
    const U32 hd = static_cast<U32>(cfg.head_dim);
    const U32 q_dim = qh * hd;
    const U32 kv_dim = kvh * hd;
    const U32 rot_dim =
        static_cast<U32>(std::llround(cfg.head_dim * cfg.partial_rotary_factor));
    const float eps = static_cast<float>(cfg.rms_norm_eps);

    // 1. Embedding
    std::vector<float> hidden(std::size_t(T) * h);
    const auto &emb = w.at("model.embed_tokens.weight");
    for (U32 t = 0; t < T; ++t) {
      std::copy_n(emb.data() + std::size_t(tokens[t]) * h, h,
                  hidden.data() + std::size_t(t) * h);
    }

    // 2. Layer 0: full_attention
    {
      auto norm = rmsnorm(hidden, w.at("model.layers.0.input_layernorm.weight"),
                          T, h, eps, 1.0F);
      auto q_gate =
          matmul_nt(norm, w.at("model.layers.0.self_attn.q_proj.weight"), T,
                    2U * q_dim, h);
      auto k_cur = matmul_nt(norm, w.at("model.layers.0.self_attn.k_proj.weight"),
                             T, kv_dim, h);
      auto v_cur = matmul_nt(norm, w.at("model.layers.0.self_attn.v_proj.weight"),
                             T, kv_dim, h);
      const auto &q_nw = w.at("model.layers.0.self_attn.q_norm.weight");
      const auto &k_nw = w.at("model.layers.0.self_attn.k_norm.weight");

      auto apply_head_norm_rope = [&](float *vec, const std::vector<float> &nw,
                                      U32 abs_pos) {
        double sq = 0.0;
        for (U32 d = 0; d < hd; ++d) {
          sq += double(vec[d]) * double(vec[d]);
        }
        const float inv = 1.0F / std::sqrt(static_cast<float>(sq / hd) + eps);
        const U32 half_rot = rot_dim / 2U;
        for (U32 i = 0; i < half_rot; ++i) {
          const float x0 = vec[i] * inv * (nw[i] + 1.0F);
          const float x1 =
              vec[i + half_rot] * inv * (nw[i + half_rot] + 1.0F);
          const double freq =
              double(abs_pos) /
              std::pow(cfg.rope_theta, double(2U * i) / double(rot_dim));
          const float cs = static_cast<float>(std::cos(freq));
          const float sn = static_cast<float>(std::sin(freq));
          vec[i] = round_bf16(x0 * cs - x1 * sn);
          vec[i + half_rot] = round_bf16(x1 * cs + x0 * sn);
        }
        for (U32 d = rot_dim; d < hd; ++d) {
          vec[d] = round_bf16(vec[d] * inv * (nw[d] + 1.0F));
        }
      };

      for (U32 t = 0; t < T; ++t) {
        for (U32 head_i = 0; head_i < qh; ++head_i) {
          apply_head_norm_rope(
              q_gate.data() + (std::size_t(t) * qh + head_i) * 2U * hd, q_nw,
              pos + t);
        }
        for (U32 head_i = 0; head_i < kvh; ++head_i) {
          apply_head_norm_rope(
              k_cur.data() + (std::size_t(t) * kvh + head_i) * hd, k_nw,
              pos + t);
        }
        std::copy_n(k_cur.data() + std::size_t(t) * kv_dim, kv_dim,
                    k_cache.data() + std::size_t(pos + t) * kv_dim);
        std::copy_n(v_cur.data() + std::size_t(t) * kv_dim, kv_dim,
                    v_cache.data() + std::size_t(pos + t) * kv_dim);
      }

      const float attn_scale = 1.0F / std::sqrt(static_cast<float>(hd));
      std::vector<float> mixer(std::size_t(T) * q_dim, 0.0F);
      for (U32 t = 0; t < T; ++t) {
        const U32 seq_len = pos + t + 1U;
        for (U32 head_i = 0; head_i < qh; ++head_i) {
          const U32 kv_h = head_i / (qh / kvh);
          const float *q_ptr =
              q_gate.data() + (std::size_t(t) * qh + head_i) * 2U * hd;
          const float *gate_ptr = q_ptr + hd;
          std::vector<float> scores(seq_len);
          float max_s = -1e30F;
          for (U32 s = 0; s < seq_len; ++s) {
            const float *kp =
                k_cache.data() + std::size_t(s) * kv_dim + kv_h * hd;
            double dot = 0.0;
            for (U32 d = 0; d < hd; ++d) {
              dot += double(q_ptr[d]) * double(kp[d]);
            }
            scores[s] = static_cast<float>(dot) * attn_scale;
            max_s = std::max(max_s, scores[s]);
          }
          double sum_e = 0.0;
          for (U32 s = 0; s < seq_len; ++s) {
            scores[s] = std::exp(scores[s] - max_s);
            sum_e += scores[s];
          }
          float *out_p = mixer.data() + (std::size_t(t) * qh + head_i) * hd;
          for (U32 d = 0; d < hd; ++d) {
            double acc = 0.0;
            for (U32 s = 0; s < seq_len; ++s) {
              const float *vp =
                  v_cache.data() + std::size_t(s) * kv_dim + kv_h * hd;
              acc += (double(scores[s]) / sum_e) * double(vp[d]);
            }
            out_p[d] =
                round_bf16(static_cast<float>(acc) * sigmoid(gate_ptr[d]));
          }
        }
      }
      auto attn_delta = matmul_nt(
          mixer, w.at("model.layers.0.self_attn.o_proj.weight"), T, h, q_dim);
      for (std::size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = round_bf16(hidden[i] + attn_delta[i]);
      }

      auto post_norm = rmsnorm(
          hidden, w.at("model.layers.0.post_attention_layernorm.weight"), T, h,
          eps, 1.0F);
      auto gate = matmul_nt(post_norm,
                            w.at("model.layers.0.mlp.gate_proj.weight"), T, f, h);
      auto up = matmul_nt(post_norm, w.at("model.layers.0.mlp.up_proj.weight"),
                          T, f, h);
      std::vector<float> prod(std::size_t(T) * f);
      for (std::size_t i = 0; i < prod.size(); ++i) {
        prod[i] = round_bf16(silu(gate[i]) * up[i]);
      }
      auto mlp_delta = matmul_nt(
          prod, w.at("model.layers.0.mlp.down_proj.weight"), T, h, f);
      for (std::size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = round_bf16(hidden[i] + mlp_delta[i]);
      }
    }

    // 3. Layer 1: linear_attention (GDN)
    {
      const U32 kh = static_cast<U32>(cfg.linear_num_key_heads);
      const U32 hv = static_cast<U32>(cfg.linear_num_value_heads);
      const U32 kd = static_cast<U32>(cfg.linear_key_head_dim);
      const U32 vd = static_cast<U32>(cfg.linear_value_head_dim);
      const U32 gk = kh * kd;
      const U32 gv = hv * vd;
      const U32 conv_dim = 2U * gk + gv;
      const U32 k_conv = static_cast<U32>(cfg.linear_conv_kernel_dim);

      auto norm = rmsnorm(hidden, w.at("model.layers.1.input_layernorm.weight"),
                          T, h, eps, 1.0F);
      auto qkv_in = matmul_nt(
          norm, w.at("model.layers.1.linear_attn.in_proj_qkv.weight"), T,
          conv_dim, h);
      auto z_in = matmul_nt(
          norm, w.at("model.layers.1.linear_attn.in_proj_z.weight"), T, gv, h);
      auto a_in = matmul_nt(
          norm, w.at("model.layers.1.linear_attn.in_proj_a.weight"), T, hv, h);
      auto b_in = matmul_nt(
          norm, w.at("model.layers.1.linear_attn.in_proj_b.weight"), T, hv, h);

      const auto &conv_w = w.at("model.layers.1.linear_attn.conv1d.weight");
      const auto &a_log = w.at("model.layers.1.linear_attn.A_log");
      const auto &dt_bias = w.at("model.layers.1.linear_attn.dt_bias");
      const auto &gdn_nw = w.at("model.layers.1.linear_attn.norm.weight");

      std::vector<float> convolved(std::size_t(T) * conv_dim, 0.0F);
      for (U32 t = 0; t < T; ++t) {
        for (U32 c = 0; c < conv_dim; ++c) {
          float acc = 0.0F;
          for (U32 tap = 0; tap < k_conv; ++tap) {
            const int src_t = static_cast<int>(t) - static_cast<int>(k_conv - 1U - tap);
            const float x_val =
                (src_t >= 0)
                    ? qkv_in[std::size_t(src_t) * conv_dim + c]
                    : gdn_hist[std::size_t(src_t + int(k_conv - 1U)) * conv_dim + c];
            acc += x_val * conv_w[std::size_t(c) * k_conv + tap];
          }
          convolved[std::size_t(t) * conv_dim + c] = round_bf16(silu(acc));
        }
      }
      // Update conv history (last k_conv - 1 rows)
      std::vector<float> next_hist((k_conv - 1U) * conv_dim, 0.0F);
      for (U32 hi = 0; hi < k_conv - 1U; ++hi) {
        const int src_t = static_cast<int>(T) - static_cast<int>(k_conv - 1U - hi);
        if (src_t >= 0) {
          std::copy_n(qkv_in.data() + std::size_t(src_t) * conv_dim, conv_dim,
                      next_hist.data() + std::size_t(hi) * conv_dim);
        } else {
          std::copy_n(
              gdn_hist.data() + std::size_t(src_t + int(k_conv - 1U)) * conv_dim,
              conv_dim, next_hist.data() + std::size_t(hi) * conv_dim);
        }
      }
      gdn_hist = std::move(next_hist);

      std::vector<float> mixer(std::size_t(T) * gv, 0.0F);
      for (U32 t = 0; t < T; ++t) {
        const float *row_c = convolved.data() + std::size_t(t) * conv_dim;
        for (U32 h_v = 0; h_v < hv; ++h_v) {
          const U32 h_k = h_v / (hv / kh);
          const float *q_raw = row_c + h_k * kd;
          const float *k_raw = row_c + gk + h_k * kd;
          const float *v_raw = row_c + 2U * gk + h_v * vd;

          double q_sq = 0.0;
          double k_sq = 0.0;
          for (U32 d = 0; d < kd; ++d) {
            q_sq += double(q_raw[d]) * double(q_raw[d]);
            k_sq += double(k_raw[d]) * double(k_raw[d]);
          }
          const float inv_scale = 1.0F / std::sqrt(static_cast<float>(kd));
          const float q_inv =
              1.0F / std::sqrt(static_cast<float>(q_sq / kd) + 1.0e-6F);
          const float k_inv =
              1.0F / std::sqrt(static_cast<float>(k_sq / kd) + 1.0e-6F);
          std::vector<float> q_n(kd), k_n(kd);
          for (U32 d = 0; d < kd; ++d) {
            q_n[d] = q_raw[d] * q_inv * inv_scale * inv_scale;
            k_n[d] = k_raw[d] * k_inv * inv_scale;
          }

          const float g_decay = std::exp(
              -std::exp(a_log[h_v]) *
              softplus(a_in[std::size_t(t) * hv + h_v] + dt_bias[h_v]));
          const float beta = sigmoid(b_in[std::size_t(t) * hv + h_v]);

          std::vector<float> y_head(vd, 0.0F);
          for (U32 vi = 0; vi < vd; ++vi) {
            float *st = gdn_state.data() + (std::size_t(h_v) * vd + vi) * kd;
            float kv_mem = 0.0F;
            for (U32 d = 0; d < kd; ++d) {
              st[d] *= g_decay;
              kv_mem += st[d] * k_n[d];
            }
            const float delta = (v_raw[vi] - kv_mem) * beta;
            float out_v = 0.0F;
            for (U32 d = 0; d < kd; ++d) {
              st[d] += k_n[d] * delta;
              out_v += st[d] * q_n[d];
            }
            y_head[vi] = round_bf16(out_v);
          }
          // GDN RMSNorm + SiLU(z) gate
          double y_sq = 0.0;
          for (U32 vi = 0; vi < vd; ++vi) {
            y_sq += double(y_head[vi]) * double(y_head[vi]);
          }
          const float y_inv =
              1.0F / std::sqrt(static_cast<float>(y_sq / vd) + eps);
          for (U32 vi = 0; vi < vd; ++vi) {
            const float z_val = z_in[std::size_t(t) * gv + h_v * vd + vi];
            mixer[std::size_t(t) * gv + h_v * vd + vi] =
                round_bf16(y_head[vi] * y_inv * gdn_nw[vi] * silu(z_val));
          }
        }
      }

      auto gdn_delta = matmul_nt(
          mixer, w.at("model.layers.1.linear_attn.out_proj.weight"), T, h, gv);
      for (std::size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = round_bf16(hidden[i] + gdn_delta[i]);
      }

      auto post_norm = rmsnorm(
          hidden, w.at("model.layers.1.post_attention_layernorm.weight"), T, h,
          eps, 1.0F);
      auto gate = matmul_nt(post_norm,
                            w.at("model.layers.1.mlp.gate_proj.weight"), T, f, h);
      auto up = matmul_nt(post_norm, w.at("model.layers.1.mlp.up_proj.weight"),
                          T, f, h);
      std::vector<float> prod(std::size_t(T) * f);
      for (std::size_t i = 0; i < prod.size(); ++i) {
        prod[i] = round_bf16(silu(gate[i]) * up[i]);
      }
      auto mlp_delta = matmul_nt(
          prod, w.at("model.layers.1.mlp.down_proj.weight"), T, h, f);
      for (std::size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = round_bf16(hidden[i] + mlp_delta[i]);
      }
    }

    pos += T;
    auto final_n =
        rmsnorm(hidden, w.at("model.norm.weight"), T, h, eps, 1.0F);
    return matmul_nt(final_n, w.at("lm_head.weight"), T, v, h,
                     /*round_out_bf16=*/false);
  }
};

struct DiffMetrics {
  float max_abs = 0.0F;
  double cosine = 1.0;
};

DiffMetrics compare_vectors(std::span<const float> got,
                            std::span<const float> ref) {
  require(got.size() == ref.size(), "compare_vectors length mismatch");
  float max_abs = 0.0F;
  double dot = 0.0;
  double ng = 0.0;
  double nr = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    require(std::isfinite(got[i]) && std::isfinite(ref[i]),
            "nonfinite value in vector comparison");
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
    dot += double(got[i]) * double(ref[i]);
    ng += double(got[i]) * double(got[i]);
    nr += double(ref[i]) * double(ref[i]);
  }
  const double denom = std::sqrt(ng) * std::sqrt(nr);
  return {.max_abs = max_abs, .cosine = denom > 0.0 ? (dot / denom) : 1.0};
}

std::vector<std::int32_t> run_engine_sync(Engine &engine,
                                          std::vector<std::int32_t> prompt,
                                          int max_new_tokens) {
  std::vector<std::int32_t> generated;
  SubmitRequest req;
  req.prompt_tokens = std::move(prompt);
  req.max_new_tokens = max_new_tokens;
  req.sampling.temperature = 0.0F;
  req.sampling.top_k = 1;
  req.sampling.top_p = 1.0F;
  const auto req_id = engine.submit(std::move(req));
  for (int iter = 0; iter < 4000; ++iter) {
    const auto events = engine.poll_events(std::chrono::milliseconds(50));
    for (const auto &ev : events) {
      if (ev.request_id != req_id) {
        continue;
      }
      if (ev.kind == EventKind::tokens) {
        generated.insert(generated.end(), ev.tokens.begin(), ev.tokens.end());
      } else if (ev.kind == EventKind::finished) {
        return generated;
      } else if (ev.kind == EventKind::failed || ev.kind == EventKind::aborted) {
        throw std::runtime_error("engine request failed: " + ev.error);
      }
    }
  }
  throw std::runtime_error("engine request timed out in run_engine_sync");
}

// ---------------------------------------------------------------------------
// Test 1: Task 1 Mixed MXFP4 + affine8 (q8_g64 + q8_g128) Zero-Copy & Accuracy
// ---------------------------------------------------------------------------
void test_task1_mixed_mxfp4_affine8_accuracy(
    const std::filesystem::path &root_dir,
    const std::filesystem::path &metallib_dir) {
  std::cout << "[1/5] Task 1: Mixed MXFP4 + affine8 (q8_g64/g128) zero-copy & numerical accuracy...\n";
  const auto fx =
      build_qwen_fixture(root_dir / "task1_mixed_aff8", {.pure_mxfp4 = false});

  auto ctx = MetalContext::create();
  auto art = ArtifactModel::open(fx.artifact_path, true);
  auto mplan = plan_model_materialization(art, ctx, {.require_verified_payload = true});
  auto weights = materialize_model(art, std::move(mplan), ctx);
  auto catalog = build_qwen_layer_catalog(art, fx.config, 1024U, 64U);

  // Verify zero-copy binding in AffineNaxCache for both q8_g64 and q8_g128 pre-tiled views!
  if (detail::forward::affine_nax_runtime_eligible(ctx, metallib_dir)) {
    MetalBuffer zero_buf = ctx.make_buffer(256ULL);
    zero_buf.zero();
    const detail::forward::Ref zero_ref{&zero_buf, 0, 256ULL};
    const auto q_view = weights.parent_view("model.layers.0.self_attn.q_proj.weight");
    const auto o_view = weights.parent_view("model.layers.0.self_attn.o_proj.weight");
    const auto a_untiled =
        weights.parent_view("model.layers.1.linear_attn.in_proj_a.weight");
    std::vector<MaterializedWeightView> cands{q_view, o_view, a_untiled};
    auto cache = detail::forward::AffineNaxCache::acquire(
        &cands, ctx, metallib_dir, zero_ref, cands);
    const auto *tw_q = cache->find(q_view.object_id);
    const auto *tw_o = cache->find(o_view.object_id);
    const auto *tw_a = cache->find(a_untiled.object_id);
    const bool q_is_v2 = detail::forward::is_oq_nax_lane_v2_layout(q_view.layout);
    const bool o_is_v2 = detail::forward::is_oq_nax_lane_v2_layout(o_view.layout);
    require(tw_q != nullptr && tw_q->bits == 8U && tw_q->group_size == 64U,
            "q8_g64 pre-tiled missing from AffineNaxCache");
    require((!q_is_v2 || (tw_q->codes.buffer == q_view.planes[0].buffer &&
                          tw_q->codes.offset == q_view.planes[0].offset)) &&
                tw_q->codes.buffer != nullptr &&
                tw_q->sb.buffer == q_view.planes[1].buffer &&
                tw_q->sb.offset == q_view.planes[1].offset,
            "q8_g64 pre-tiled binding mismatch in AffineNaxCache");
    require(tw_o != nullptr && tw_o->bits == 8U && tw_o->group_size == 128U,
            "q8_g128 pre-tiled missing from AffineNaxCache");
    require((!o_is_v2 || (tw_o->codes.buffer == o_view.planes[0].buffer &&
                          tw_o->codes.offset == o_view.planes[0].offset)) &&
                tw_o->codes.buffer != nullptr &&
                tw_o->sb.buffer == o_view.planes[1].buffer &&
                tw_o->sb.offset == o_view.planes[1].offset,
            "q8_g128 pre-tiled binding mismatch in AffineNaxCache");
    require(tw_a != nullptr && tw_a->bits == 8U && tw_a->group_size == 128U &&
                tw_a->codes.buffer != a_untiled.planes[0].buffer,
            "untiled q8_g128 fallback pre-tile failed in AffineNaxCache");
  }

  // Run QwenForward across T=32 (prefill), T=4 (small-T), and T=1 (decode) vs exact FP32 reference!
  QwenForward fwd(ctx, weights, catalog, fx.config, 1024U, metallib_dir, {},
                  false, true, false, std::nullopt, false, false, std::nullopt,
                  LinearNumericPolicy::RowExactPortableV1,
                  SamplingContract::PositionCoupledV1,
                  AttentionNumericPolicy::RowExactV1, true);
  CpuQwenRef cpu_ref(fx.config, fx.ref_weights);
  GpuSamplingOptions samp{.temperature = 0.0F,
                          .top_k = 1U,
                          .top_p = 1.0F,
                          .contract = SamplingContract::PositionCoupledV1};

  std::vector<U32> tok32(32U);
  for (U32 i = 0; i < 32U; ++i) {
    tok32[i] = (i * 7U + 3U) % 256U;
  }
  const auto gpu32 = fwd.run(tok32, samp, /*inspect_tensors=*/true);
  const auto ref32 = cpu_ref.forward(tok32);
  const auto dm32 = compare_vectors(gpu32.logits, ref32);
  require(dm32.max_abs < 0.035F && dm32.cosine > 0.9999,
          "Task 1 T=32 logits mismatch: max_abs=" + std::to_string(dm32.max_abs) +
              " cos=" + std::to_string(dm32.cosine));

  const std::array<U32, 4> tok4{19U, 42U, 108U, 211U};
  const auto gpu4 = fwd.run(tok4, samp, /*inspect_tensors=*/true);
  const auto ref4 = cpu_ref.forward(tok4);
  const auto dm4 = compare_vectors(gpu4.logits, ref4);
  require(dm4.max_abs < 0.035F && dm4.cosine > 0.9999,
          "Task 1 T=4 logits mismatch: max_abs=" + std::to_string(dm4.max_abs));

  const std::array<U32, 1> tok1{77U};
  const auto gpu1 = fwd.run(tok1, samp, /*inspect_tensors=*/true);
  const auto ref1 = cpu_ref.forward(tok1);
  const auto dm1 = compare_vectors(gpu1.logits, ref1);
  require(dm1.max_abs < 0.035F && dm1.cosine > 0.9999,
          "Task 1 T=1 logits mismatch: max_abs=" + std::to_string(dm1.max_abs));
  std::cout << "  Task 1 accuracy: T=32 max_abs=" << dm32.max_abs
            << " cos=" << dm32.cosine << " | T=4 max_abs=" << dm4.max_abs
            << " cos=" << dm4.cosine << " | T=1 max_abs=" << dm1.max_abs
            << " cos=" << dm1.cosine << "\n";
}

// ---------------------------------------------------------------------------
// Test 2: Standard LoRA Loader (HF PEFT + mlx-lm + validation edge cases)
// ---------------------------------------------------------------------------
void test_lora_loader_peft_and_mlxlm(const std::filesystem::path &root_dir,
                                     const QwenModelConfig &cfg) {
  std::cout << "[2/5] Task 3.1: HF PEFT & mlx-lm LoRA directory loader semantics...\n";

  // 2A. HuggingFace PEFT with rank_pattern, alpha_pattern, use_rslora, fan_in_fan_out,
  // BF16/F16/F32 dtypes, split q_proj + q_gate_proj, and fused in_proj_qkvz + in_proj_ba!
  const auto peft_dir = root_dir / "peft_loader_test";
  Json peft_cfg = {
      {"peft_type", "LORA"},
      {"r", 8},
      {"lora_alpha", 16.0},
      {"use_rslora", true},
      {"rank_pattern", {{"o_proj", 16}}},
      {"alpha_pattern", {{"o_proj", 32.0}}},
      {"fan_in_fan_out", {"mlp.down_proj"}},
  };
  write_text_file(peft_dir / "adapter_config.json", peft_cfg.dump(2));

  const auto make_vec = [](std::size_t n, U32 seed, float amp = 0.05F) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = det_val(seed, i, amp);
    }
    return v;
  };
  std::vector<StTensorSpec> peft_tensors{
      // Split q_proj (BF16) + q_gate_proj (F16), r=8
      {"base_model.model.language_model.model.layers.0.self_attn.q_proj.lora_A.weight",
       "BF16",
       {8, 512},
       make_vec(8 * 512, 11U)},
      {"base_model.model.language_model.model.layers.0.self_attn.q_proj.lora_B.weight",
       "BF16",
       {512, 8},
       make_vec(512 * 8, 12U)},
      {"base_model.model.language_model.model.layers.0.self_attn.q_gate_proj.lora_A.weight",
       "F16",
       {8, 512},
       make_vec(8 * 512, 13U)},
      {"base_model.model.language_model.model.layers.0.self_attn.q_gate_proj.lora_B.weight",
       "F16",
       {512, 8},
       make_vec(512 * 8, 14U)},
      // o_proj with rank_pattern=16, alpha_pattern=32, use_rslora -> scale = 32 / sqrt(16) = 8.0 (F32)
      {"base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight",
       "F32",
       {16, 512},
       make_vec(16 * 512, 15U)},
      {"base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight",
       "F32",
       {512, 16},
       make_vec(512 * 16, 16U)},
      // mlp.down_proj with per-module fan_in_fan_out: A=[K=512, r=8], B=[r=8, N=512]
      {"model.layers.0.mlp.down_proj.lora_A.weight",
       "BF16",
       {512, 8},
       make_vec(512 * 8, 17U)},
      {"model.layers.0.mlp.down_proj.lora_B.weight",
       "BF16",
       {8, 512},
       make_vec(8 * 512, 18U)},
      // Fused GDN in_proj_qkvz ([2048, 512]) + in_proj_ba ([8, 512])
      {"layers.1.linear_attn.in_proj_qkvz.lora_A.weight",
       "F32",
       {8, 512},
       make_vec(8 * 512, 19U)},
      {"layers.1.linear_attn.in_proj_qkvz.lora_B.weight",
       "F32",
       {2048, 8},
       make_vec(2048 * 8, 20U)},
      {"layers.1.linear_attn.in_proj_ba.lora_A.weight",
       "F32",
       {8, 512},
       make_vec(8 * 512, 21U)},
      {"layers.1.linear_attn.in_proj_ba.lora_B.weight",
       "F32",
       {8, 8},
       make_vec(8 * 8, 22U)},
  };
  write_safetensors_file(peft_dir / "adapter_model.safetensors", peft_tensors);

  const auto loaded_peft = load_lora_adapter_for_qwen(peft_dir, cfg, false);
  require(loaded_peft.adapted_layers == 2U,
          "expected 2 adapted layers in PEFT test");
  // Expected canonical updates:
  // layer 0: attention_q (rank 16 = 8+8!), attention_o (rank 16, scale 8.0), mlp_down (rank 8)
  // layer 1: gdn_qkv, gdn_z, gdn_a, gdn_b -> total 7 updates!
  require(loaded_peft.updates.size() == 7U,
          "expected 7 canonical updates from PEFT test, got " +
              std::to_string(loaded_peft.updates.size()));
  for (const auto &u : loaded_peft.updates) {
    if (!u.target.is_global && u.target.layer_index == 0U &&
        u.target.role == "attention_o") {
      require(u.rank == 16U && std::abs(u.scale - 8.0F) < 1e-5F,
              "o_proj rank_pattern/alpha_pattern/rslora scale mismatch");
    }
    if (!u.target.is_global && u.target.layer_index == 0U &&
        u.target.role == "attention_q") {
      require(u.rank == 16U && u.out_features == 1024U && u.in_features == 512U,
              "split q_proj + q_gate_proj merge geometry mismatch");
    }
  }

  // 2B. mlx-lm format (adapters.safetensors + nested lora_parameters + [in, r] / [r, out])
  const auto mlx_dir = root_dir / "mlxlm_loader_test";
  Json mlx_cfg = {
      {"fine_tune_type", "lora"},
      {"num_layers", 2},
      {"lora_parameters", {{"rank", 4}, {"alpha", 8.0}, {"scale", 2.5}}},
  };
  write_text_file(mlx_dir / "adapter_config.json", mlx_cfg.dump(2));
  const auto a_mlx = make_vec(512 * 4, 31U); // [in=512, r=4]
  const auto b_mlx = make_vec(4 * 256, 32U); // [r=4, out=256]
  std::vector<StTensorSpec> mlx_tensors{
      {"language_model.model.layers.0.self_attn.k_proj.lora_a",
       "F32",
       {512, 4},
       a_mlx},
      {"language_model.model.layers.0.self_attn.k_proj.lora_b",
       "F32",
       {4, 256},
       b_mlx},
  };
  write_safetensors_file(mlx_dir / "adapters.safetensors", mlx_tensors);
  const auto loaded_mlx = load_lora_adapter_for_qwen(mlx_dir, cfg, false);
  require(loaded_mlx.updates.size() == 1U && loaded_mlx.adapted_layers == 1U,
          "mlx-lm loader update count mismatch");
  const auto &mu = loaded_mlx.updates[0];
  require(mu.rank == 4U && std::abs(mu.scale - 2.5F) < 1e-6F,
          "mlx-lm explicit lora_parameters.scale mismatch");
  // Verify exact transpose: A_canon[ri, ci] == a_mlx[ci * 4 + ri], B_canon[ni, ri] == b_mlx[ri * 256 + ni]
  for (U32 ri = 0; ri < 4U; ++ri) {
    for (U32 ci = 0; ci < 512U; ++ci) {
      require(mu.a_f32[ri * 512U + ci] == a_mlx[ci * 4U + ri],
              "mlx-lm lora_a transpose mismatch");
    }
    for (U32 ni = 0; ni < 256U; ++ni) {
      require(mu.b_f32[ni * 4U + ri] == b_mlx[ri * 256U + ni],
              "mlx-lm lora_b transpose mismatch");
    }
  }

  // 2C. Negative checks: tied lm_head rejection & NaN weight rejection
  const auto bad_head_dir = root_dir / "bad_tied_head";
  write_text_file(bad_head_dir / "adapter_config.json", R"({"r": 4, "lora_alpha": 8})");
  write_safetensors_file(
      bad_head_dir / "adapter_model.safetensors",
      std::array<StTensorSpec, 2>{{
          {"lm_head.lora_A.weight", "F32", {4, 512}, make_vec(4 * 512, 41U)},
          {"lm_head.lora_B.weight", "F32", {256, 4}, make_vec(256 * 4, 42U)},
      }});
  bool rejected_tied = false;
  try {
    (void)load_lora_adapter_for_qwen(bad_head_dir, cfg, /*lm_head_tied=*/true);
  } catch (const std::invalid_argument &) {
    rejected_tied = true;
  }
  require(rejected_tied, "lm_head LoRA on tied embedding must be rejected");

  const auto bad_nan_dir = root_dir / "bad_nan";
  write_text_file(bad_nan_dir / "adapter_config.json", R"({"r": 4, "lora_alpha": 8})");
  auto nan_vec = make_vec(4 * 512, 43U);
  nan_vec[7] = std::numeric_limits<float>::quiet_NaN();
  write_safetensors_file(
      bad_nan_dir / "adapter_model.safetensors",
      std::array<StTensorSpec, 2>{{
          {"model.layers.0.self_attn.k_proj.lora_A.weight", "F32", {4, 512},
           nan_vec},
          {"model.layers.0.self_attn.k_proj.lora_B.weight", "F32", {256, 4},
           make_vec(256 * 4, 44U)},
      }});
  bool rejected_nan = false;
  try {
    (void)load_lora_adapter_for_qwen(bad_nan_dir, cfg, false);
  } catch (const std::invalid_argument &) {
    rejected_nan = true;
  }
  require(rejected_nan, "NaN LoRA weights must be rejected");
}

// Helper to create a rich multi-rank LoRA adapter directory covering ranks 4, 8, 16, 32, 64!
std::filesystem::path build_multirank_peft_adapter(
    const std::filesystem::path &dir, U32 seed_base, float global_alpha = 8.0F) {
  Json cfg = {
      {"peft_type", "LORA"},
      {"r", 8},
      {"lora_alpha", global_alpha},
      {"rank_pattern",
       {{"q_proj", 4},
        {"q_gate_proj", 4},
        {"k_proj", 8},
        {"v_proj", 16},
        {"o_proj", 32},
        {"layers.0.mlp.gate_proj", 8},
        {"layers.0.mlp.up_proj", 16},
        {"layers.0.mlp.down_proj", 64},
        {"in_proj_qkvz", 8},
        {"in_proj_ba", 4},
        {"out_proj", 16},
        {"layers.1.mlp.gate_proj", 32},
        {"layers.1.mlp.down_proj", 8},
        {"lm_head", 16}}},
  };
  write_text_file(dir / "adapter_config.json", cfg.dump(2));

  const auto mv = [](U32 rows, U32 cols, U32 seed, float amp = 0.04F) {
    std::vector<float> out(std::size_t(rows) * cols);
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = round_bf16(det_val(seed, i, amp));
    }
    return out;
  };

  std::vector<StTensorSpec> specs{
      // Layer 0 full_attention (q=4+4, k=8, v=16, o=32) + MLP (gate=8, up=16, down=64)
      {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", "BF16",
       {4, 512}, mv(4, 512, seed_base + 1U)},
      {"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", "BF16",
       {512, 4}, mv(512, 4, seed_base + 2U)},
      {"base_model.model.model.layers.0.self_attn.q_gate_proj.lora_A.weight",
       "BF16", {4, 512}, mv(4, 512, seed_base + 3U)},
      {"base_model.model.model.layers.0.self_attn.q_gate_proj.lora_B.weight",
       "BF16", {512, 4}, mv(512, 4, seed_base + 4U)},
      {"base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight", "BF16",
       {8, 512}, mv(8, 512, seed_base + 5U)},
      {"base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight", "BF16",
       {256, 8}, mv(256, 8, seed_base + 6U)},
      {"base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight", "BF16",
       {16, 512}, mv(16, 512, seed_base + 7U)},
      {"base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight", "BF16",
       {256, 16}, mv(256, 16, seed_base + 8U)},
      {"base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight", "BF16",
       {32, 512}, mv(32, 512, seed_base + 9U)},
      {"base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight", "BF16",
       {512, 32}, mv(512, 32, seed_base + 10U)},
      {"base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight", "BF16",
       {8, 512}, mv(8, 512, seed_base + 11U)},
      {"base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight", "BF16",
       {512, 8}, mv(512, 8, seed_base + 12U)},
      {"base_model.model.model.layers.0.mlp.up_proj.lora_A.weight", "BF16",
       {16, 512}, mv(16, 512, seed_base + 13U)},
      {"base_model.model.model.layers.0.mlp.up_proj.lora_B.weight", "BF16",
       {512, 16}, mv(512, 16, seed_base + 14U)},
      {"base_model.model.model.layers.0.mlp.down_proj.lora_A.weight", "BF16",
       {64, 512}, mv(64, 512, seed_base + 15U)},
      {"base_model.model.model.layers.0.mlp.down_proj.lora_B.weight", "BF16",
       {512, 64}, mv(512, 64, seed_base + 16U)},
      // Layer 1 GDN + MLP
      {"base_model.model.model.layers.1.linear_attn.in_proj_qkvz.lora_A.weight",
       "BF16", {8, 512}, mv(8, 512, seed_base + 17U)},
      {"base_model.model.model.layers.1.linear_attn.in_proj_qkvz.lora_B.weight",
       "BF16", {2048, 8}, mv(2048, 8, seed_base + 18U)},
      {"base_model.model.model.layers.1.linear_attn.in_proj_ba.lora_A.weight",
       "BF16", {4, 512}, mv(4, 512, seed_base + 19U)},
      {"base_model.model.model.layers.1.linear_attn.in_proj_ba.lora_B.weight",
       "BF16", {8, 4}, mv(8, 4, seed_base + 20U)},
      {"base_model.model.model.layers.1.linear_attn.out_proj.lora_A.weight",
       "BF16", {16, 512}, mv(16, 512, seed_base + 21U)},
      {"base_model.model.model.layers.1.linear_attn.out_proj.lora_B.weight",
       "BF16", {512, 16}, mv(512, 16, seed_base + 22U)},
      {"base_model.model.model.layers.1.mlp.gate_proj.lora_A.weight", "BF16",
       {32, 512}, mv(32, 512, seed_base + 23U)},
      {"base_model.model.model.layers.1.mlp.gate_proj.lora_B.weight", "BF16",
       {512, 32}, mv(512, 32, seed_base + 24U)},
      {"base_model.model.model.layers.1.mlp.down_proj.lora_A.weight", "BF16",
       {8, 512}, mv(8, 512, seed_base + 25U)},
      {"base_model.model.model.layers.1.mlp.down_proj.lora_B.weight", "BF16",
       {512, 8}, mv(512, 8, seed_base + 26U)},
      // Untied lm_head (strongly shifts output distribution so LoRA effect is unmistakable!)
      {"base_model.model.lm_head.lora_A.weight", "BF16", {16, 512},
       mv(16, 512, seed_base + 27U, 0.08F)},
      {"base_model.model.lm_head.lora_B.weight", "BF16", {256, 16},
       mv(256, 16, seed_base + 28U, 0.08F)},
  };
  write_safetensors_file(dir / "adapter_model.safetensors", specs);
  return dir;
}

std::filesystem::path build_mlxlm_adapter(const std::filesystem::path &dir,
                                          U32 seed_base) {
  Json cfg = {
      {"fine_tune_type", "lora"},
      {"lora_parameters", {{"rank", 16}, {"alpha", 24.0}, {"scale", 1.5}}},
  };
  write_text_file(dir / "adapter_config.json", cfg.dump(2));
  const auto mv = [](U32 rows, U32 cols, U32 seed, float amp = 0.05F) {
    std::vector<float> out(std::size_t(rows) * cols);
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = round_bf16(det_val(seed, i, amp));
    }
    return out;
  };
  // mlx-lm orientation: lora_a = [in_features, r], lora_b = [r, out_features]
  std::vector<StTensorSpec> specs{
      {"model.layers.0.mlp.gate_proj.lora_a", "BF16", {512, 16},
       mv(512, 16, seed_base + 1U)},
      {"model.layers.0.mlp.gate_proj.lora_b", "BF16", {16, 512},
       mv(16, 512, seed_base + 2U)},
      {"model.layers.0.mlp.down_proj.lora_a", "BF16", {512, 16},
       mv(512, 16, seed_base + 3U)},
      {"model.layers.0.mlp.down_proj.lora_b", "BF16", {16, 512},
       mv(16, 512, seed_base + 4U)},
      {"lm_head.lora_a", "BF16", {512, 16}, mv(512, 16, seed_base + 5U, 0.09F)},
      {"lm_head.lora_b", "BF16", {16, 256}, mv(16, 256, seed_base + 6U, 0.09F)},
  };
  write_safetensors_file(dir / "adapters.safetensors", specs);
  return dir;
}

std::string sha256_file(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  require(in.good(), "failed to read file for sha256: " + path.string());
  const auto sz = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  std::vector<std::byte> buf(sz);
  in.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(sz));
  return artifact_sha256_hex(buf);
}

// ---------------------------------------------------------------------------
// Test 3: Mode A ("merge") & Mode B ("bypass") vs Exact FP32 Reference
// across ranks r = 4, 8, 16, 32, 64 and T = 32, T = 4, T = 1
// ---------------------------------------------------------------------------
void test_merge_and_bypass_exact_parity(
    const std::filesystem::path &root_dir,
    const std::filesystem::path &metallib_dir) {
  std::cout << "[3/5] Task 3.1: Mode A (merge) & Mode B (bypass) vs FP32 reference across r={4,8,16,32,64}, T={32,4,1}...\n";

  // Pure MXFP4 Layer 0 + mixed oQ4/oQ6/oQ8/MXFP4 Layer 1 specifically stresses:
  // - Mode A unpacking MXFP4 tiled + Affine q4/q6/q8 tiled -> q8_g64 NAX override
  // - Mode B disabling block_input_nax, block_mlp_input_nax, fused SwiGLU, and add_sq_t1!
  const auto fx =
      build_qwen_fixture(root_dir / "lora_parity_model", {.pure_mxfp4 = true});
  const std::string disk_sha_before = sha256_file(fx.artifact_path);

  const auto adapter_dir =
      build_multirank_peft_adapter(root_dir / "lora_multirank_peft", 700U);
  const auto loaded = load_lora_adapter_for_qwen(adapter_dir, fx.config, false);

  auto ctx = MetalContext::create();
  auto art = ArtifactModel::open(fx.artifact_path, true);
  auto mplan = plan_model_materialization(art, ctx, {.require_verified_payload = true});
  auto weights = materialize_model(art, std::move(mplan), ctx);
  auto catalog = build_qwen_layer_catalog(art, fx.config, 1024U, 64U);

  auto merged_rt = detail::lora_rt::build_merged_lora_overrides(
      ctx, metallib_dir, weights, catalog, loaded);
  auto bypass_rt = detail::lora_rt::build_bypass_lora_runtime(
      ctx, weights, catalog, loaded);
  require(merged_rt->buffer_bytes > 0 && merged_rt->adapted_layers == 3U,
          "merged_rt buffer_bytes / adapted_layers invalid");
  require(bypass_rt->weight_buffer_bytes > 0 && bypass_rt->adapted_layers == 3U,
          "bypass_rt weight_buffer_bytes / adapted_layers invalid");

  // Also run base unadapted model to confirm LoRA strongly changes logits!
  QwenForward fwd_base(ctx, weights, catalog, fx.config, 1024U, metallib_dir,
                       {}, false, false, false, std::nullopt, false, false,
                       std::nullopt, LinearNumericPolicy::RowExactPortableV1,
                       SamplingContract::PositionCoupledV1,
                       AttentionNumericPolicy::RowExactV1, true);
  QwenForward fwd_merge(ctx, weights, catalog, fx.config, 1024U, metallib_dir,
                        {}, false, false, false, std::nullopt, false, false,
                        std::nullopt, LinearNumericPolicy::RowExactPortableV1,
                        SamplingContract::PositionCoupledV1,
                        AttentionNumericPolicy::RowExactV1, true, "bf16",
                        nullptr, false, 0U, merged_rt, nullptr);
  QwenForward fwd_bypass(ctx, weights, catalog, fx.config, 1024U, metallib_dir,
                         {}, false, false, false, std::nullopt, false, false,
                         std::nullopt, LinearNumericPolicy::RowExactPortableV1,
                         SamplingContract::PositionCoupledV1,
                         AttentionNumericPolicy::RowExactV1, true, "bf16",
                         nullptr, false, 0U, nullptr, bypass_rt);

  // Verify Mode A ("merge") has ZERO runtime LoRA kernel dispatches and overrides to q8!
  for (U32 cap : {1U, 32U}) {
    const auto pdesc = fwd_merge.inspect_profile(cap, false);
    bool any_q8_override = false;
    for (const auto &wr : pdesc.weights) {
      require(wr.library != "lora",
              "merge mode must have 0 runtime lora library dispatches");
      if (wr.object == "lm_head.weight") {
        require(wr.format == "oq_affine_q8_g64_bf16_v1",
                "merged lm_head must have format oq_affine_q8_g64_bf16_v1");
        any_q8_override = true;
      }
    }
    require(any_q8_override, "merge mode profile missing q8 override route");
  }

  CpuQwenRef cpu_lora_ref(fx.config, fx.ref_weights, &loaded);
  GpuSamplingOptions samp{.temperature = 0.0F,
                          .top_k = 1U,
                          .top_p = 1.0F,
                          .contract = SamplingContract::PositionCoupledV1};

  // Step A: T = 32 prefill
  std::vector<U32> tok32(32U);
  for (U32 i = 0; i < 32U; ++i) {
    tok32[i] = (i * 11U + 5U) % 256U;
  }
  const auto base32 = fwd_base.run(tok32, samp, true);
  const auto merge32 = fwd_merge.run(tok32, samp, true);
  const auto bypass32 = fwd_bypass.run(tok32, samp, true);
  const auto ref32 = cpu_lora_ref.forward(tok32);

  const auto diff_vs_base = compare_vectors(bypass32.logits, base32.logits);
  require(diff_vs_base.max_abs > 0.25F,
          "LoRA adapter had negligible effect vs base logits");

  float max_ref32 = 1e-6F;
  for (float v : ref32) {
    max_ref32 = std::max(max_ref32, std::abs(v));
  }
  const auto dm_merge32 = compare_vectors(merge32.logits, ref32);
  const auto dm_bypass32 = compare_vectors(bypass32.logits, ref32);
  require((dm_merge32.max_abs / max_ref32) < 0.025F && dm_merge32.cosine > 0.9999,
          "merge T=32 vs FP32 ref mismatch: max_abs=" +
              std::to_string(dm_merge32.max_abs) +
              " max_ref=" + std::to_string(max_ref32) +
              " cos=" + std::to_string(dm_merge32.cosine));
  require((dm_bypass32.max_abs / max_ref32) < 0.025F &&
              dm_bypass32.cosine > 0.9999,
          "bypass T=32 vs FP32 ref mismatch: max_abs=" +
              std::to_string(dm_bypass32.max_abs) +
              " max_ref=" + std::to_string(max_ref32) +
              " cos=" + std::to_string(dm_bypass32.cosine));

  // Step B: T = 4 small-T multi-row continuation
  const std::array<U32, 4> tok4{13U, 88U, 144U, 233U};
  const auto merge4 = fwd_merge.run(tok4, samp, true);
  const auto bypass4 = fwd_bypass.run(tok4, samp, true);
  const auto ref4 = cpu_lora_ref.forward(tok4);
  float max_ref4 = 1e-6F;
  for (float v : ref4) {
    max_ref4 = std::max(max_ref4, std::abs(v));
  }
  const auto dm_merge4 = compare_vectors(merge4.logits, ref4);
  const auto dm_bypass4 = compare_vectors(bypass4.logits, ref4);
  require((dm_merge4.max_abs / max_ref4) < 0.025F && dm_merge4.cosine > 0.9999,
          "merge T=4 vs FP32 ref mismatch: max_abs=" +
              std::to_string(dm_merge4.max_abs));
  require((dm_bypass4.max_abs / max_ref4) < 0.025F && dm_bypass4.cosine > 0.9999,
          "bypass T=4 vs FP32 ref mismatch: max_abs=" +
              std::to_string(dm_bypass4.max_abs));

  // Step C: T = 1 decode continuation (also test non-inspect T=1 fast decode!)
  const std::array<U32, 1> tok1{55U};
  const auto merge1 = fwd_merge.run(tok1, samp, true);
  const auto bypass1 = fwd_bypass.run(tok1, samp, true);
  const auto ref1 = cpu_lora_ref.forward(tok1);
  float max_ref1 = 1e-6F;
  for (float v : ref1) {
    max_ref1 = std::max(max_ref1, std::abs(v));
  }
  const auto dm_merge1 = compare_vectors(merge1.logits, ref1);
  const auto dm_bypass1 = compare_vectors(bypass1.logits, ref1);
  require((dm_merge1.max_abs / max_ref1) < 0.025F && dm_merge1.cosine > 0.9999,
          "merge T=1 vs FP32 ref mismatch: max_abs=" +
              std::to_string(dm_merge1.max_abs));
  require((dm_bypass1.max_abs / max_ref1) < 0.025F && dm_bypass1.cosine > 0.9999,
          "bypass T=1 vs FP32 ref mismatch: max_abs=" +
              std::to_string(dm_bypass1.max_abs));

  require(sha256_file(fx.artifact_path) == disk_sha_before,
          "on-disk .miv artifact was mutated by LoRA merge/bypass!");

  // Step D: Pure oQ Affine (q4 + q6 + q8 tiled + q8_g128 untiled) merge & bypass vs FP32 ref!
  {
    const auto fx_oq =
        build_qwen_fixture(root_dir / "lora_parity_oq", {.pure_oq = true});
    auto art_oq = ArtifactModel::open(fx_oq.artifact_path, true);
    auto mplan_oq = plan_model_materialization(
        art_oq, ctx, {.require_verified_payload = true});
    auto weights_oq = materialize_model(art_oq, std::move(mplan_oq), ctx);
    auto cat_oq = build_qwen_layer_catalog(art_oq, fx_oq.config, 1024U, 64U);
    auto m_oq = detail::lora_rt::build_merged_lora_overrides(
        ctx, metallib_dir, weights_oq, cat_oq, loaded);
    auto b_oq = detail::lora_rt::build_bypass_lora_runtime(
        ctx, weights_oq, cat_oq, loaded);
    QwenForward fm_oq(ctx, weights_oq, cat_oq, fx_oq.config, 1024U,
                      metallib_dir, {}, false, false, false, std::nullopt,
                      false, false, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1,
                      AttentionNumericPolicy::RowExactV1, true, "bf16", nullptr,
                      false, 0U, m_oq, nullptr);
    QwenForward fb_oq(ctx, weights_oq, cat_oq, fx_oq.config, 1024U,
                      metallib_dir, {}, false, false, false, std::nullopt,
                      false, false, std::nullopt,
                      LinearNumericPolicy::RowExactPortableV1,
                      SamplingContract::PositionCoupledV1,
                      AttentionNumericPolicy::RowExactV1, true, "bf16", nullptr,
                      false, 0U, nullptr, b_oq);
    CpuQwenRef ref_oq(fx_oq.config, fx_oq.ref_weights, &loaded);
    const auto gm_oq = fm_oq.run(tok32, samp, true);
    const auto gb_oq = fb_oq.run(tok32, samp, true);
    const auto gr_oq = ref_oq.forward(tok32);
    const auto d_moq = compare_vectors(gm_oq.logits, gr_oq);
    const auto d_boq = compare_vectors(gb_oq.logits, gr_oq);
    require(d_moq.cosine > 0.9995 && d_boq.cosine > 0.9995,
            "oQ q4/q6/q8 LoRA merge/bypass parity failed: cos_merge=" +
                std::to_string(d_moq.cosine) +
                " cos_bypass=" + std::to_string(d_boq.cosine));
  }
  std::cout << "  Task 3.1 merge  accuracy: T=32 max_abs=" << dm_merge32.max_abs
            << " cos=" << dm_merge32.cosine << " | T=4 max_abs=" << dm_merge4.max_abs
            << " | T=1 max_abs=" << dm_merge1.max_abs << "\n"
            << "  Task 3.1 bypass accuracy: T=32 max_abs=" << dm_bypass32.max_abs
            << " cos=" << dm_bypass32.cosine << " | T=4 max_abs=" << dm_bypass4.max_abs
            << " | T=1 max_abs=" << dm_bypass1.max_abs << "\n";
}

// ---------------------------------------------------------------------------
// Test 4: Engine::reload_lora Hot-Swapping & Memory Budget Enforcement
// ---------------------------------------------------------------------------
void test_engine_hotswap_and_budget(const std::filesystem::path &root_dir,
                                    const std::filesystem::path &metallib_dir) {
  std::cout << "[4/5] Task 3.1: Engine::reload_lora bypass hot-swap & UMA budget enforcement...\n";
  const auto fx =
      build_qwen_fixture(root_dir / "hotswap_model", {.pure_mxfp4 = true});
  const auto adapter_a =
      build_multirank_peft_adapter(root_dir / "hotswap_adapter_a", 900U, 8.0F);
  const auto adapter_b =
      build_mlxlm_adapter(root_dir / "hotswap_adapter_b", 1100U);

  const std::vector<std::int32_t> prompt{4, 15, 29, 63, 102, 199};

  // Reference Engine runs for Adapter A (merge) and Adapter B (merge)
  auto make_opts = [&](std::string lora_dir, std::string mode) {
    LoadOptions lo;
    lo.model_dir = fx.model_dir.string();
    lo.max_context = 1024;
    lo.max_concurrency = 2;
    lo.lora_adapter = std::move(lora_dir);
    lo.lora_mode = std::move(mode);
    lo.path_overrides["kernel_directory"] = metallib_dir.string();
    lo.path_overrides["prefix_cache_bytes"] = std::to_string(16ULL << 20);
    return lo;
  };

  std::vector<std::int32_t> expected_base, expected_a, expected_b;
  {
    auto eng_a = Engine::create(make_opts(adapter_a.string(), "merge"));
    const auto st_a = eng_a->stats();
    require(st_a.lora_adapter == adapter_a.string() && st_a.lora_mode == "merge" &&
                st_a.lora_buffer_bytes > 0 && st_a.lora_adapted_layers == 3U,
            "merge Engine stats mismatch");
    expected_a = run_engine_sync(*eng_a, prompt, 6);
    std::string merge_err;
    require(!eng_a->reload_lora(adapter_b.string(), "bypass", &merge_err) &&
                merge_err.find("merge mode") != std::string::npos,
            "reload_lora on merge-loaded Engine must return false");
    eng_a->shutdown();
  }
  {
    auto eng_b = Engine::create(make_opts(adapter_b.string(), "merge"));
    expected_b = run_engine_sync(*eng_b, prompt, 6);
    eng_b->shutdown();
  }

  // Live single Engine hot-swapping: None -> Adapter A -> Adapter B -> Invalid -> Clear ("")
  auto eng = Engine::create(make_opts("", "bypass"));
  {
    const auto st0 = eng->stats();
    require(st0.lora_adapter.empty() && st0.lora_mode.empty() &&
                st0.lora_buffer_bytes == 0 && st0.lora_adapted_layers == 0U,
            "initial unadapted Engine stats must have empty lora fields");
    expected_base = run_engine_sync(*eng, prompt, 6);
    require(expected_base != expected_a && expected_a != expected_b,
            "base, Adapter A, and Adapter B must produce distinct token sequences");
  }

  // Hot-swap -> Adapter A (also verifies prefix cache from base is invalidated!)
  std::string err;
  require(eng->reload_lora(adapter_a.string(), "bypass", &err),
          "hot-swap to Adapter A failed: " + err);
  {
    const auto st = eng->stats();
    require(st.lora_adapter == adapter_a.string() && st.lora_mode == "bypass" &&
                st.lora_buffer_bytes > 0 && st.lora_adapted_layers == 3U,
            "stats after hot-swap to Adapter A mismatch");
    const auto got_a = run_engine_sync(*eng, prompt, 6);
    require(got_a == expected_a,
            "bypass hot-swap Adapter A tokens differ from merge reference");
  }

  // Hot-swap -> Adapter B (mlx-lm format)
  require(eng->reload_lora(adapter_b.string(), "bypass", &err),
          "hot-swap to Adapter B failed: " + err);
  {
    const auto st = eng->stats();
    require(st.lora_adapter == adapter_b.string() && st.lora_mode == "bypass" &&
                st.lora_buffer_bytes > 0 && st.lora_adapted_layers == 2U,
            "stats after hot-swap to Adapter B mismatch");
    const auto got_b = run_engine_sync(*eng, prompt, 6);
    require(got_b == expected_b,
            "bypass hot-swap Adapter B tokens differ from merge reference");
  }

  // Invalid adapter hot-swap must fail cleanly while keeping Adapter B intact!
  require(!eng->reload_lora((root_dir / "does_not_exist").string(), "bypass", &err),
          "nonexistent adapter hot-swap must return false");
  require(eng->stats().lora_adapter == adapter_b.string(),
          "failed hot-swap corrupted active Adapter B stats");
  require(run_engine_sync(*eng, prompt, 6) == expected_b,
          "failed hot-swap corrupted active Adapter B generation");

  // Switching to merge mode via reload_lora must return false so sidecar falls back to full reload
  require(!eng->reload_lora(adapter_a.string(), "merge", &err) &&
              err.find("merge mode") != std::string::npos,
          "reload_lora(..., 'merge') must return false");

  // Clear LoRA ("") -> restores exact base generation and frees lora_buffer_bytes!
  require(eng->reload_lora("", "bypass", &err),
          "clearing bypass LoRA failed: " + err);
  {
    const auto st_clear = eng->stats();
    require(st_clear.lora_adapter.empty() && st_clear.lora_mode.empty() &&
                st_clear.lora_buffer_bytes == 0 &&
                st_clear.lora_adapted_layers == 0U,
            "cleared LoRA stats must be zero/empty");
    const auto got_base = run_engine_sync(*eng, prompt, 6);
    require(got_base == expected_base,
            "cleared bypass LoRA did not restore exact base token sequence");
  }
  eng->shutdown();

  // Memory budget enforcement check (tiny 1 MiB budget must reject load)
  {
    auto tiny_opts = make_opts(adapter_a.string(), "merge");
    tiny_opts.memory_budget_bytes = 1ULL << 20;
    bool rejected_budget = false;
    try {
      (void)Engine::create(tiny_opts);
    } catch (const std::runtime_error &) {
      rejected_budget = true;
    }
    require(rejected_budget,
            "1 MiB memory_budget_bytes must reject native LoRA load");
  }
}

// ---------------------------------------------------------------------------
// Test 5: Speculative Decoding Compatibility (MTP & DFlash2 + merge & bypass)
// ---------------------------------------------------------------------------
void test_speculative_mtp_and_dflash_lora(
    const std::filesystem::path &root_dir,
    const std::filesystem::path &metallib_dir) {
  std::cout << "[5/5] Task 3.1: Speculative decoding (MTP & DFlash2) compatibility with merge & bypass LoRA...\n";
  const auto fx = build_qwen_fixture(
      root_dir / "spec_qwen_mtp",
      {.pure_mxfp4 = false, .include_embedded_mtp = true});
  const auto dflash_dir = build_dflash2_companion(root_dir / "spec_dflash2");
  const auto adapter_dir =
      build_multirank_peft_adapter(root_dir / "spec_lora_adapter", 1300U, 8.0F);

  const std::vector<std::int32_t> prompt{7, 19, 33, 81, 120, 145, 200, 222};

  for (const std::string mode : {"merge", "bypass"}) {
    // Ground-truth pure AR target sequence with active LoRA (enable_mtp=off)
    LoadOptions ar_opts;
    ar_opts.model_dir = fx.model_dir.string();
    ar_opts.max_context = 1024;
    ar_opts.lora_adapter = adapter_dir.string();
    ar_opts.lora_mode = mode;
    ar_opts.path_overrides["kernel_directory"] = metallib_dir.string();
    ar_opts.path_overrides["enable_mtp"] = "off";
    auto ar_eng = Engine::create(ar_opts);
    const auto ref_tokens = run_engine_sync(*ar_eng, prompt, 10);
    ar_eng->shutdown();

    // 5A. MTP Speculative Decoding + active LoRA
    LoadOptions mtp_opts = ar_opts;
    mtp_opts.path_overrides.erase("enable_mtp");
    mtp_opts.speculative.backend = SpeculativeBackend::mtp;
    mtp_opts.speculative.draft_tokens = 3;
    auto mtp_eng = Engine::create(mtp_opts);
    const auto mtp_tokens = run_engine_sync(*mtp_eng, prompt, 10);
    const auto mtp_st = mtp_eng->stats();
    require(mtp_st.spec_rounds > 0 && mtp_st.spec_target_verify_calls > 0,
            "MTP speculative engine did not execute verify rounds under LoRA mode=" +
                mode);
    require(mtp_tokens == ref_tokens,
            "MTP speculative tokens diverged from target AR under LoRA mode=" +
                mode);
    mtp_eng->shutdown();

    // 5B. DFlash2 Speculative Decoding + active LoRA
    LoadOptions df_opts = ar_opts;
    df_opts.draft_dir = dflash_dir.string();
    df_opts.speculative.backend = SpeculativeBackend::dflash2;
    df_opts.speculative.draft_tokens = 3;
    auto df_eng = Engine::create(df_opts);
    const auto df_tokens = run_engine_sync(*df_eng, prompt, 10);
    const auto df_st = df_eng->stats();
    require(df_st.spec_rounds > 0 && df_st.spec_target_verify_calls > 0,
            "DFlash2 speculative engine did not execute verify rounds under LoRA mode=" +
                mode);
    require(df_tokens == ref_tokens,
            "DFlash2 speculative tokens diverged from target AR under LoRA mode=" +
                mode);
    df_eng->shutdown();
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc >= 2, "usage: omlx_m5_lora_dual_mode_tests <metallib_dir>");
    const std::filesystem::path metallib_dir(argv[1]);
    const auto tmp_dir = std::filesystem::temp_directory_path() /
                         ("omlx-m5-lora-dual-" + std::to_string(::getpid()));
    std::filesystem::remove_all(tmp_dir);
    std::filesystem::create_directories(tmp_dir);
    struct Cleanup {
      std::filesystem::path dir;
      ~Cleanup() { std::filesystem::remove_all(dir); }
    } cleanup{tmp_dir};

    test_task1_mixed_mxfp4_affine8_accuracy(tmp_dir, metallib_dir);
    const auto loader_fx =
        build_qwen_fixture(tmp_dir / "loader_cfg_fx", {.pure_mxfp4 = true});
    test_lora_loader_peft_and_mlxlm(tmp_dir, loader_fx.config);
    test_merge_and_bypass_exact_parity(tmp_dir, metallib_dir);
    test_engine_hotswap_and_budget(tmp_dir, metallib_dir);
    test_speculative_mtp_and_dflash_lora(tmp_dir, metallib_dir);

    std::cout << "ALL LORA DUAL-MODE & AFFINE8 C++ TESTS PASSED!\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "lora_dual_mode_tests FAILED: " << ex.what() << '\n';
    return 1;
  }
}
