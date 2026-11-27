#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::uint32_t reference_scalar_crc32(std::span<const std::byte> bytes,
                                     std::uint32_t initial = 0) {
  std::uint32_t crc = ~initial;
  for (const auto byte : bytes) {
    crc ^= static_cast<std::uint8_t>(byte);
    for (int bit = 0; bit < 8; ++bit) {
      const auto mask = (crc & 1U) != 0U ? 0xEDB88320U : 0U;
      crc = (crc >> 1) ^ mask;
    }
  }
  return ~crc;
}

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

std::vector<std::byte> build_test_artifact(std::uint64_t payload_alignment) {
  // Object 0: multi-plane oq_affine_q4_g64_bf16_v1 [2, 128] with 256B plane alignment
  // Object 1: single-plane direct_bf16_v1 [4]
  std::vector<std::byte> base_code(128, std::byte{0x5AU});
  std::vector<std::byte> scale(8, std::byte{0x3CU});
  std::vector<std::byte> bias(8, std::byte{0x00U});
  std::vector<std::byte> norm_data(8, std::byte{0x3FU});

  std::vector<std::byte> payload;
  const auto align_payload = [&](std::size_t alignment) {
    while (payload.size() % alignment != 0) {
      payload.push_back(std::byte{0});
    }
  };

  // Object 0 at offset 0
  const std::uint64_t obj0_offset = 0;
  const std::uint64_t p0_offset = payload.size();
  payload.insert(payload.end(), base_code.begin(), base_code.end());
  align_payload(256);
  const std::uint64_t p1_offset = payload.size();
  payload.insert(payload.end(), scale.begin(), scale.end());
  align_payload(256);
  const std::uint64_t p2_offset = payload.size();
  payload.insert(payload.end(), bias.begin(), bias.end());
  const std::uint64_t obj0_bytes = payload.size() - obj0_offset;
  const auto obj0_sha = omlx::m5::artifact_sha256_hex(
      std::span<const std::byte>(payload).subspan(obj0_offset, obj0_bytes));

  // Object 1 aligned to 256
  align_payload(256);
  const std::uint64_t obj1_offset = payload.size();
  const std::uint64_t obj1_p0_offset = payload.size();
  payload.insert(payload.end(), norm_data.begin(), norm_data.end());
  const std::uint64_t obj1_bytes = payload.size() - obj1_offset;
  const auto obj1_sha = omlx::m5::artifact_sha256_hex(
      std::span<const std::byte>(payload).subspan(obj1_offset, obj1_bytes));

  // Canonical JSON (sorted keys, no whitespace)
  std::ostringstream json;
  json << "{\"bindings\":[{\"activation_policy\":\"A16Only\","
       << "\"format\":\"oq_affine_q4_g64_bf16_v1\",\"in_features\":128,"
       << "\"leading_shape\":[],\"name\":\"linear\",\"object_id\":0,"
       << "\"out_features\":2,\"use\":\"linear\"}],"
       << "\"container_version\":1,"
       << "\"model\":{\"source_profile\":\"oq_affine\","
       << "\"source_repo\":\"test/repo\",\"source_revision\":\"rev1\"},"
       << "\"objects\":["
       << "{\"format\":\"oq_affine_q4_g64_bf16_v1\",\"id\":0,"
       << "\"layout\":\"oq_rowsplit_k128_v1\","
       << "\"name\":\"linear\",\"object_bytes\":" << obj0_bytes
       << ",\"object_offset\":" << obj0_offset
       << ",\"object_sha256\":\"" << obj0_sha << "\",\"padded_k\":128,"
       << "\"planes\":["
       << "{\"alignment\":256,\"bytes\":" << base_code.size() << ",\"crc32\":\""
       << hex_crc32(omlx::m5::artifact_crc32(base_code))
       << "\",\"kind\":\"base_code\",\"offset\":" << p0_offset
       << ",\"sha256\":\"" << omlx::m5::artifact_sha256_hex(base_code) << "\"},"
       << "{\"alignment\":256,\"bytes\":" << scale.size() << ",\"crc32\":\""
       << hex_crc32(omlx::m5::artifact_crc32(scale))
       << "\",\"kind\":\"scale\",\"offset\":" << p1_offset
       << ",\"sha256\":\"" << omlx::m5::artifact_sha256_hex(scale) << "\"},"
       << "{\"alignment\":256,\"bytes\":" << bias.size() << ",\"crc32\":\""
       << hex_crc32(omlx::m5::artifact_crc32(bias))
       << "\",\"kind\":\"bias\",\"offset\":" << p2_offset
       << ",\"sha256\":\"" << omlx::m5::artifact_sha256_hex(bias) << "\"}"
       << "],\"shape\":[2,128],\"source_tensors\":[\"linear.weight\"]},"
       << "{\"format\":\"direct_bf16_v1\",\"id\":1,"
       << "\"layout\":\"contiguous_le_v1\","
       << "\"name\":\"norm.weight\",\"object_bytes\":" << obj1_bytes
       << ",\"object_offset\":" << obj1_offset
       << ",\"object_sha256\":\"" << obj1_sha << "\","
       << "\"planes\":["
       << "{\"alignment\":256,\"bytes\":" << norm_data.size() << ",\"crc32\":\""
       << hex_crc32(omlx::m5::artifact_crc32(norm_data))
       << "\",\"kind\":\"data\",\"offset\":" << obj1_p0_offset
       << ",\"sha256\":\"" << omlx::m5::artifact_sha256_hex(norm_data) << "\"}"
       << "],\"shape\":[4],\"source_tensors\":[\"norm.weight\"]}"
       << "],\"payload_bytes\":" << payload.size()
       << ",\"schema\":\"omlx-m5-artifact-v1\",\"views\":[]}";

  const auto manifest_str = json.str();
  const auto manifest_bytes =
      std::as_bytes(std::span(manifest_str.data(), manifest_str.size()));
  const std::uint64_t raw_end = 64ULL + manifest_bytes.size();
  const std::uint64_t payload_offset =
      (raw_end + payload_alignment - 1) & ~(payload_alignment - 1);
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
  append_u32(file, 2U);
  append_u32(file, manifest_crc);
  const std::uint32_t header_crc =
      omlx::m5::artifact_crc32(std::span<const std::byte>(file.data(), 56));
  append_u32(file, header_crc);
  append_u32(file, 0U);
  file.insert(file.end(), manifest_bytes.begin(), manifest_bytes.end());
  file.resize(static_cast<std::size_t>(payload_offset), std::byte{0});
  file.insert(file.end(), payload.begin(), payload.end());
  return file;
}

void test_crc32_contract_and_speed() {
  constexpr char input[] = "123456789";
  const auto bytes = std::as_bytes(std::span(input, sizeof(input) - 1));
  expect(omlx::m5::artifact_crc32(bytes) == 0xCBF43926U,
         "artifact CRC32 does not match the standard IEEE vector");

  const auto first = std::as_bytes(std::span(input, std::size_t{4}));
  const auto second = std::as_bytes(std::span(input + 4, std::size_t{5}));
  const auto chained =
      omlx::m5::artifact_crc32(second, omlx::m5::artifact_crc32(first));
  expect(chained == 0xCBF43926U,
         "artifact CRC32 chaining disagrees with zlib semantics");

  // Sweep alignments 0..15 and diverse lengths against scalar IEEE oracle
  std::vector<std::byte> buffer(65537 + 16);
  for (std::size_t i = 0; i < buffer.size(); ++i) {
    buffer[i] = static_cast<std::byte>((i * 131U + 17U) & 0xFFU);
  }
  constexpr std::array<std::size_t, 20> kLengths = {
      0,  1,  2,  3,  4,   5,   6,   7,   8,    9,
      15, 16, 31, 63, 64, 65, 127, 255, 4097, 65537};
  for (std::size_t align = 0; align < 16; ++align) {
    for (const auto len : kLengths) {
      const std::span<const std::byte> slice(buffer.data() + align, len);
      const auto actual = omlx::m5::artifact_crc32(slice);
      const auto expected = reference_scalar_crc32(slice);
      expect(actual == expected,
             "HW CRC32 mismatch at align=" + std::to_string(align) +
                 " len=" + std::to_string(len));
      if (len >= 9) {
        const auto split = len / 3;
        const auto c1 = omlx::m5::artifact_crc32(slice.first(split));
        const auto c2 = omlx::m5::artifact_crc32(slice.subspan(split), c1);
        expect(c2 == expected, "HW CRC32 split chaining mismatch");
      }
    }
  }

  // Throughput benchmark on 16 MiB
  std::vector<std::byte> bench_buf(16 * 1024 * 1024);
  for (std::size_t i = 0; i < bench_buf.size(); ++i) {
    bench_buf[i] = static_cast<std::byte>((i * 73U + 9U) & 0xFFU);
  }
  const auto t0 = std::chrono::steady_clock::now();
  const auto scalar_crc = reference_scalar_crc32(bench_buf);
  const auto t1 = std::chrono::steady_clock::now();
  std::uint32_t hw_crc = 0;
  for (int iter = 0; iter < 8; ++iter) {
    hw_crc = omlx::m5::artifact_crc32(bench_buf);
  }
  const auto t2 = std::chrono::steady_clock::now();
  expect(hw_crc == scalar_crc, "16 MiB benchmark CRC32 mismatch");
  const double scalar_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double hw_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count() / 8.0;
  const double hw_gbps =
      (static_cast<double>(bench_buf.size()) / 1e9) / (hw_ms / 1e3);
  std::cout << "CRC32 16MiB: scalar=" << scalar_ms << "ms, hw=" << hw_ms
            << "ms (" << hw_gbps << " GB/s, " << (scalar_ms / hw_ms)
            << "x speedup)\n";
}

void test_16kib_and_legacy_4kib_artifacts() {
  expect(omlx::m5::kPayloadAlignment == 16384 &&
             omlx::m5::kLegacyPayloadAlignment == 4096,
         "unexpected payload alignment constants");

  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       ("omlx-m5-b0-c1-" + std::to_string(::getpid()));
  std::filesystem::create_directories(tmp_dir);
  struct Cleanup {
    std::filesystem::path dir;
    ~Cleanup() { std::filesystem::remove_all(dir); }
  } cleanup{tmp_dir};

  const auto write_bytes = [](const std::filesystem::path &p,
                              const std::vector<std::byte> &data) {
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data.data()),
              static_cast<std::streamsize>(data.size()));
  };

  // 1. Newly generated 16 KiB-aligned artifact
  const auto path_16k = tmp_dir / "aligned_16k.omlxm5";
  const auto raw_16k = build_test_artifact(16384);
  write_bytes(path_16k, raw_16k);
  auto model_16k = omlx::m5::ArtifactModel::open(path_16k, false);
  expect(model_16k.summary().payload_offset == 16384 &&
             !model_16k.summary().payload_verified,
         "16 KiB artifact summary mismatch");
  model_16k.verify_payload();
  expect(model_16k.summary().payload_verified,
         "in-place verify_payload did not mark summary verified");
  const auto verified_16k = omlx::m5::ArtifactModel::open(path_16k, true);
  expect(verified_16k.summary().payload_verified &&
             verified_16k.objects().size() == 2,
         "open(..., true) failed on 16 KiB artifact");

  // 2. Existing 4 KiB-aligned v1 artifact (payload_offset == 4096)
  const auto path_4k = tmp_dir / "legacy_4k.omlxm5";
  const auto raw_4k = build_test_artifact(4096);
  write_bytes(path_4k, raw_4k);
  const auto verified_4k = omlx::m5::ArtifactModel::open(path_4k, true);
  expect(verified_4k.summary().payload_offset == 4096 &&
             verified_4k.summary().payload_verified,
         "legacy 4096-aligned v1 artifact rejected");

  // 3. Unaligned 2 KiB payload_offset must still fail
  const auto path_2k = tmp_dir / "bad_2k.omlxm5";
  write_bytes(path_2k, build_test_artifact(2048));
  bool rejected_2k = false;
  try {
    static_cast<void>(omlx::m5::ArtifactModel::open(path_2k, false));
  } catch (const std::runtime_error &) {
    rejected_2k = true;
  }
  expect(rejected_2k, "2048-aligned payload_offset was accepted");

  // 4. Payload byte corruption caught by single-pass streaming verification
  auto corrupted = raw_16k;
  corrupted.back() ^= std::byte{0x01};
  const auto path_corrupt = tmp_dir / "corrupt.omlxm5";
  write_bytes(path_corrupt, corrupted);
  auto unverified_corrupt = omlx::m5::ArtifactModel::open(path_corrupt, false);
  bool caught_corrupt = false;
  try {
    unverified_corrupt.verify_payload();
  } catch (const std::runtime_error &) {
    caught_corrupt = true;
  }
  expect(caught_corrupt, "single-pass verify_payload missed plane corruption");
}

std::uint16_t float_to_bf16_bits(float v) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(v);
  const std::uint32_t lsb = (bits >> 16U) & 1U;
  return static_cast<std::uint16_t>((bits + 0x7FFFU + lsb) >> 16U);
}

float bf16_bits_to_float(std::uint16_t b) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16U);
}

std::vector<std::byte> build_affine8_pretiled_artifact(
    std::uint32_t n, std::uint32_t k, std::uint32_t group_size,
    bool inject_spurious_high2 = false, bool half_size_codes = false,
    std::vector<float> *out_ref = nullptr) {
  const std::uint32_t bn = (n >= 65536U) ? 32U : 16U;
  const std::uint32_t padded_n = ((n + bn - 1U) / bn) * bn;
  const std::uint32_t padded_k = ((k + 127U) / 128U) * 128U;
  const std::uint32_t k_groups64 = k / 64U;
  const std::uint32_t sb_groups = k / group_size;

  std::vector<std::byte> nax_codes(
      half_size_codes ? (std::size_t(padded_n) * k / 2U)
                      : (std::size_t(padded_n) * k),
      std::byte{0});
  std::vector<std::byte> nax_sb(std::size_t(k_groups64) * padded_n * 4U,
                                std::byte{0});
  if (out_ref != nullptr) {
    out_ref->assign(std::size_t(n) * k, 0.0F);
  }

  for (std::uint32_t row = 0; row < n; ++row) {
    const std::uint32_t tile_n = row / bn;
    const std::uint32_t rn = row % bn;
    for (std::uint32_t sg = 0; sg < sb_groups; ++sg) {
      const float sc_raw = 0.0125F * float((sg % 7U) + 1U) + 0.001F * float(row % 5U);
      const float bi_raw = -1.25F + 0.05F * float((row + sg) % 11U);
      const std::uint16_t sc_bf = float_to_bf16_bits(sc_raw);
      const std::uint16_t bi_bf = float_to_bf16_bits(bi_raw);
      const std::uint32_t packed_sb =
          static_cast<std::uint32_t>(sc_bf) |
          (static_cast<std::uint32_t>(bi_bf) << 16U);
      if (group_size == 128U) {
        std::memcpy(nax_sb.data() + (std::size_t(2U * sg) * padded_n + row) * 4U,
                    &packed_sb, 4U);
        std::memcpy(
            nax_sb.data() + (std::size_t(2U * sg + 1U) * padded_n + row) * 4U,
            &packed_sb, 4U);
      } else {
        std::memcpy(nax_sb.data() + (std::size_t(sg) * padded_n + row) * 4U,
                    &packed_sb, 4U);
      }
    }
    if (!half_size_codes) {
      for (std::uint32_t col = 0; col < k; ++col) {
        const std::uint8_t q =
            static_cast<std::uint8_t>((row * 37U + col * 13U + 5U) & 0xFFU);
        const std::uint32_t g64 = col / 64U;
        const std::uint32_t off = col % 64U;
        const std::size_t c_idx =
            (std::size_t(tile_n) * k_groups64 + g64) * (std::size_t(bn) * 64U) +
            std::size_t(rn) * 64U + off;
        nax_codes[c_idx] = static_cast<std::byte>(q);

        if (out_ref != nullptr) {
          std::uint32_t packed_sb = 0;
          std::memcpy(&packed_sb,
                      nax_sb.data() + (std::size_t(g64) * padded_n + row) * 4U,
                      4U);
          const float sc =
              bf16_bits_to_float(static_cast<std::uint16_t>(packed_sb & 0xFFFFU));
          const float bi =
              bf16_bits_to_float(static_cast<std::uint16_t>(packed_sb >> 16U));
          (*out_ref)[std::size_t(row) * k + col] = std::fma(sc, float(q), bi);
        }
      }
    }
  }

  std::vector<std::byte> payload;
  const auto align_payload = [&](std::size_t alignment) {
    while (payload.size() % alignment != 0U) {
      payload.push_back(std::byte{0});
    }
  };

  const std::uint64_t p0_offset = payload.size();
  payload.insert(payload.end(), nax_codes.begin(), nax_codes.end());
  align_payload(256U);
  std::uint64_t p_high_offset = 0;
  std::vector<std::byte> spurious_high(std::size_t(padded_n) * (k / 4U),
                                       std::byte{0});
  if (inject_spurious_high2) {
    p_high_offset = payload.size();
    payload.insert(payload.end(), spurious_high.begin(), spurious_high.end());
    align_payload(256U);
  }
  const std::uint64_t p1_offset = payload.size();
  payload.insert(payload.end(), nax_sb.begin(), nax_sb.end());
  const std::uint64_t obj0_bytes = payload.size();
  const auto obj0_sha = omlx::m5::artifact_sha256_hex(payload);

  const std::string fmt =
      (group_size == 128U) ? "oq_affine_q8_g128_bf16_v1"
                           : "oq_affine_q8_g64_bf16_v1";
  std::ostringstream json;
  json << "{\"bindings\":[{\"activation_policy\":\"A16Only\","
       << "\"format\":\"" << fmt << "\",\"in_features\":" << k
       << ",\"leading_shape\":[],\"name\":\"linear\",\"object_id\":0,"
       << "\"out_features\":" << n << ",\"use\":\"linear\"}],"
       << "\"container_version\":1,"
       << "\"model\":{\"source_profile\":\"oq_affine\","
       << "\"source_repo\":\"test/affine8\",\"source_revision\":\"rev8\"},"
       << "\"objects\":[{\"format\":\"" << fmt << "\",\"id\":0,"
       << "\"layout\":\"oq_nax_tiled_g64_v1\",\"name\":\"linear\","
       << "\"object_bytes\":" << obj0_bytes << ",\"object_offset\":0,"
       << "\"object_sha256\":\"" << obj0_sha << "\",\"padded_k\":" << padded_k
       << ",\"planes\":["
       << "{\"alignment\":256,\"bytes\":" << nax_codes.size() << ",\"crc32\":\""
       << hex_crc32(omlx::m5::artifact_crc32(nax_codes))
       << "\",\"kind\":\"nax_codes\",\"offset\":" << p0_offset
       << ",\"sha256\":\"" << omlx::m5::artifact_sha256_hex(nax_codes) << "\"},";
  if (inject_spurious_high2) {
    json << "{\"alignment\":256,\"bytes\":" << spurious_high.size()
         << ",\"crc32\":\""
         << hex_crc32(omlx::m5::artifact_crc32(spurious_high))
         << "\",\"kind\":\"nax_high2\",\"offset\":" << p_high_offset
         << ",\"sha256\":\"" << omlx::m5::artifact_sha256_hex(spurious_high)
         << "\"},";
  }
  json << "{\"alignment\":256,\"bytes\":" << nax_sb.size() << ",\"crc32\":\""
       << hex_crc32(omlx::m5::artifact_crc32(nax_sb))
       << "\",\"kind\":\"nax_sb\",\"offset\":" << p1_offset
       << ",\"sha256\":\"" << omlx::m5::artifact_sha256_hex(nax_sb) << "\"}"
       << "],\"shape\":[" << n << "," << k
       << "],\"source_tensors\":[\"linear.weight\"]}],"
       << "\"payload_bytes\":" << payload.size()
       << ",\"schema\":\"omlx-m5-artifact-v1\",\"views\":[]}";

  const auto manifest_str = json.str();
  const auto manifest_bytes =
      std::as_bytes(std::span(manifest_str.data(), manifest_str.size()));
  const std::uint64_t raw_end = 64ULL + manifest_bytes.size();
  const std::uint64_t payload_offset = (raw_end + 16383ULL) & ~16383ULL;
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
  append_u32(file, 1U);
  append_u32(file, manifest_crc);
  const std::uint32_t header_crc =
      omlx::m5::artifact_crc32(std::span<const std::byte>(file.data(), 56U));
  append_u32(file, header_crc);
  append_u32(file, 0U);
  file.insert(file.end(), manifest_bytes.begin(), manifest_bytes.end());
  file.resize(static_cast<std::size_t>(payload_offset), std::byte{0});
  file.insert(file.end(), payload.begin(), payload.end());
  return file;
}

void test_affine8_nax_pretile_contract_and_decode() {
  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       ("omlx-m5-b0-aff8-" + std::to_string(::getpid()));
  std::filesystem::create_directories(tmp_dir);
  struct Cleanup {
    std::filesystem::path dir;
    ~Cleanup() { std::filesystem::remove_all(dir); }
  } cleanup{tmp_dir};

  const auto write_bytes = [](const std::filesystem::path &p,
                              const std::vector<std::byte> &data) {
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data.data()),
              static_cast<std::streamsize>(data.size()));
  };

  // 1. q8_g64 pre-tiled (N=37 -> padded_n=48, K=512) + zero-copy materialization
  std::vector<float> ref_g64;
  const auto p_g64 = tmp_dir / "q8_g64.omlxm5";
  write_bytes(p_g64,
              build_affine8_pretiled_artifact(37U, 512U, 64U, false, false,
                                              &ref_g64));
  const auto art_g64 = omlx::m5::ArtifactModel::open(p_g64, true);
  expect(art_g64.objects().size() == 1U &&
             art_g64.objects()[0].planes.size() == 2U,
         "q8_g64 pre-tiled must have exactly 2 planes (nax_codes, nax_sb)");
  for (std::uint32_t row : {0U, 15U, 16U, 36U}) {
    const auto decoded =
        art_g64.decode_weight_row(art_g64.objects()[0], row, 0U, 512U);
    expect(decoded.size() == 512U, "q8_g64 decoded row length mismatch");
    for (std::size_t col = 0; col < 512U; ++col) {
      expect(decoded[col] == ref_g64[std::size_t(row) * 512U + col],
             "q8_g64 decode_weight_row 0-ULP mismatch");
    }
  }
  auto ctx = omlx::m5::MetalContext::create();
  auto plan = omlx::m5::plan_model_materialization(
      art_g64, ctx, {.require_verified_payload = true});
  auto mat = omlx::m5::materialize_model(art_g64, std::move(plan), ctx);
  const auto view = mat.parent_view(0U);
  expect(view.format == "oq_affine_q8_g64_bf16_v1" &&
             view.layout == "oq_nax_tiled_g64_v1" && view.planes.size() == 2U &&
             view.planes[0].kind == "nax_codes" &&
             view.planes[0].bytes == 48ULL * 512ULL &&
             view.planes[1].kind == "nax_sb" &&
             view.planes[1].bytes == 48ULL * 8ULL * 4ULL,
         "q8_g64 materialized view plane contract mismatch");

  // 2. q8_g128 pre-tiled (N=20 -> padded_n=32, K=512)
  std::vector<float> ref_g128;
  const auto p_g128 = tmp_dir / "q8_g128.omlxm5";
  write_bytes(p_g128,
              build_affine8_pretiled_artifact(20U, 512U, 128U, false, false,
                                              &ref_g128));
  const auto art_g128 = omlx::m5::ArtifactModel::open(p_g128, true);
  for (std::uint32_t row : {0U, 15U, 16U, 19U}) {
    const auto decoded =
        art_g128.decode_weight_row(art_g128.objects()[0], row, 0U, 512U);
    for (std::size_t col = 0; col < 512U; ++col) {
      expect(decoded[col] == ref_g128[std::size_t(row) * 512U + col],
             "q8_g128 decode_weight_row 0-ULP mismatch");
    }
  }

  // 3. Non-128-multiple K=576 (padded_k=640) q8_g64 pre-tiled
  std::vector<float> ref_k576;
  const auto p_k576 = tmp_dir / "q8_g64_k576.omlxm5";
  write_bytes(p_k576,
              build_affine8_pretiled_artifact(18U, 576U, 64U, false, false,
                                              &ref_k576));
  const auto art_k576 = omlx::m5::ArtifactModel::open(p_k576, true);
  for (std::uint32_t row : {0U, 15U, 16U, 17U}) {
    const auto decoded =
        art_k576.decode_weight_row(art_k576.objects()[0], row, 0U, 576U);
    expect(decoded.size() == 576U, "q8_g64 K=576 decoded length mismatch");
    for (std::size_t col = 0; col < 576U; ++col) {
      expect(decoded[col] == ref_k576[std::size_t(row) * 576U + col],
             "q8_g64 K=576 decode_weight_row 0-ULP mismatch");
    }
  }

  // 4. Negative: spurious nax_high2 plane on bits == 8 must be rejected
  const auto p_spurious = tmp_dir / "bad_high2.omlxm5";
  write_bytes(p_spurious,
              build_affine8_pretiled_artifact(16U, 512U, 64U, true, false));
  bool rejected_spurious = false;
  try {
    static_cast<void>(omlx::m5::ArtifactModel::open(p_spurious, false));
  } catch (const std::exception &) {
    rejected_spurious = true;
  }
  expect(rejected_spurious,
         "spurious nax_high2 plane on q8 oq_nax_tiled_g64_v1 was accepted");

  // 5. Negative: 4-bit half-size nax_codes on bits == 8 must be rejected
  const auto p_half = tmp_dir / "bad_half_codes.omlxm5";
  write_bytes(p_half,
              build_affine8_pretiled_artifact(16U, 512U, 64U, false, true));
  bool rejected_half = false;
  try {
    static_cast<void>(omlx::m5::ArtifactModel::open(p_half, false));
  } catch (const std::exception &) {
    rejected_half = true;
  }
  expect(rejected_half,
         "half-size nax_codes on q8 oq_nax_tiled_g64_v1 was accepted");
}

std::uint32_t pi64_ref(std::uint32_t off) {
  return ((off & 12U) << 2U) | ((off >> 2U) & 12U) | (off & 3U);
}

void fwht64_ref(std::vector<float> &v) {
  expect(v.size() % 64U == 0U, "fwht64_ref length must be multiple of 64");
  for (std::size_t base = 0; base < v.size(); base += 64U) {
    for (std::size_t h = 1; h < 64U; h <<= 1) {
      for (std::size_t i = 0; i < 64U; i += (h << 1)) {
        for (std::size_t j = 0; j < h; ++j) {
          const float a = v[base + i + j];
          const float b = v[base + i + j + h];
          v[base + i + j] = a + b;
          v[base + i + j + h] = a - b;
        }
      }
    }
    for (std::size_t i = 0; i < 64U; ++i) {
      v[base + i] *= 0.125F;
    }
  }
}

std::vector<std::byte> build_v2_affine_lane_artifact(
    std::uint32_t bits, std::uint32_t group_size, std::uint32_t n,
    std::uint32_t k, bool with_fwht64, std::uint32_t version_override,
    std::uint32_t flags_override, const std::string &layout_override,
    const std::string &transform_override,
    std::vector<float> *out_raw_stored = nullptr,
    std::vector<float> *out_logical_unrotated = nullptr) {
  const std::uint32_t bn = (n >= 65536U) ? 32U : 16U;
  const std::uint32_t padded_n = ((n + bn - 1U) / bn) * bn;
  const std::uint32_t padded_k = ((k + 127U) / 128U) * 128U;
  const std::uint32_t k_groups64 = k / 64U;
  const std::uint32_t sb_groups = k / group_size;

  const std::size_t codes_bytes =
      bits == 8U ? std::size_t(padded_n) * k
      : bits == 3U ? std::size_t(padded_n) * (k / 4U)
                   : std::size_t(padded_n) * (k / 2U);
  const std::size_t high_bytes =
      bits == 3U ? std::size_t(padded_n) * (k / 8U)
      : (bits == 5U || bits == 6U) ? std::size_t(padded_n) * (k / 4U)
                                   : 0U;
  std::vector<std::byte> nax_codes(codes_bytes, std::byte{0});
  std::vector<std::byte> nax_high(high_bytes, std::byte{0});
  std::vector<std::byte> nax_sb(std::size_t(k_groups64) * padded_n * 4U,
                                std::byte{0});

  if (out_raw_stored != nullptr) {
    out_raw_stored->assign(std::size_t(n) * k, 0.0F);
  }
  const std::uint32_t qmask = (1U << bits) - 1U;
  for (std::uint32_t row = 0; row < n; ++row) {
    const std::uint32_t tile_n = row / bn;
    const std::uint32_t rn = row % bn;
    for (std::uint32_t sg = 0; sg < sb_groups; ++sg) {
      const float sc_raw = 0.02F * float((sg % 5U) + 1U) + 0.002F * float(row % 7U);
      const float bi_raw = -0.75F + 0.04F * float((row + sg) % 9U);
      const std::uint32_t packed_sb =
          static_cast<std::uint32_t>(float_to_bf16_bits(sc_raw)) |
          (static_cast<std::uint32_t>(float_to_bf16_bits(bi_raw)) << 16U);
      if (group_size == 128U) {
        std::memcpy(nax_sb.data() + (std::size_t(2U * sg) * padded_n + row) * 4U,
                    &packed_sb, 4U);
        std::memcpy(
            nax_sb.data() + (std::size_t(2U * sg + 1U) * padded_n + row) * 4U,
            &packed_sb, 4U);
      } else {
        std::memcpy(nax_sb.data() + (std::size_t(sg) * padded_n + row) * 4U,
                    &packed_sb, 4U);
      }
    }
    for (std::uint32_t col = 0; col < k; ++col) {
      const std::uint32_t q = (row * 29U + col * 17U + 3U) & qmask;
      const std::uint32_t g64 = col / 64U;
      const std::uint32_t off = col % 64U;
      const std::uint32_t slot = pi64_ref(off);
      const std::size_t tile_idx = std::size_t(tile_n) * k_groups64 + g64;
      if (bits == 8U) {
        nax_codes[tile_idx * (std::size_t(bn) * 64U) + std::size_t(rn) * 64U + slot] =
            static_cast<std::byte>(q);
      } else if (bits == 3U) {
        auto &cb = nax_codes[tile_idx * (std::size_t(bn) * 16U) +
                             std::size_t(rn) * 16U + slot / 4U];
        cb = static_cast<std::byte>(
            static_cast<std::uint8_t>(cb) |
            static_cast<std::uint8_t>((q & 3U) << ((slot & 3U) * 2U)));
        auto &hb = nax_high[tile_idx * (std::size_t(bn) * 8U) +
                            std::size_t(rn) * 8U + slot / 8U];
        hb = static_cast<std::byte>(
            static_cast<std::uint8_t>(hb) |
            static_cast<std::uint8_t>(((q >> 2U) & 1U) << (slot & 7U)));
      } else {
        auto &cb = nax_codes[tile_idx * (std::size_t(bn) * 32U) +
                             std::size_t(rn) * 32U + slot / 2U];
        cb = static_cast<std::byte>(
            static_cast<std::uint8_t>(cb) |
            static_cast<std::uint8_t>((q & 0x0FU) << ((slot & 1U) * 4U)));
        if (bits == 5U || bits == 6U) {
          auto &hb = nax_high[tile_idx * (std::size_t(bn) * 16U) +
                              std::size_t(rn) * 16U + slot / 4U];
          const std::uint32_t hval =
              bits == 5U ? ((q >> 4U) & 1U) : ((q >> 4U) & 3U);
          hb = static_cast<std::byte>(
              static_cast<std::uint8_t>(hb) |
              static_cast<std::uint8_t>(hval << ((slot & 3U) * 2U)));
        }
      }
      if (out_raw_stored != nullptr) {
        std::uint32_t packed_sb = 0;
        std::memcpy(&packed_sb,
                    nax_sb.data() + (std::size_t(g64) * padded_n + row) * 4U, 4U);
        const float sc =
            bf16_bits_to_float(static_cast<std::uint16_t>(packed_sb & 0xFFFFU));
        const float bi =
            bf16_bits_to_float(static_cast<std::uint16_t>(packed_sb >> 16U));
        (*out_raw_stored)[std::size_t(row) * k + col] = std::fma(sc, float(q), bi);
      }
    }
  }
  if (out_logical_unrotated != nullptr && out_raw_stored != nullptr) {
    *out_logical_unrotated = *out_raw_stored;
    if (with_fwht64) {
      for (std::uint32_t row = 0; row < n; ++row) {
        std::vector<float> r(out_logical_unrotated->begin() + std::ptrdiff_t(row * k),
                             out_logical_unrotated->begin() +
                                 std::ptrdiff_t((row + 1U) * k));
        fwht64_ref(r);
        std::copy(r.begin(), r.end(),
                  out_logical_unrotated->begin() + std::ptrdiff_t(row * k));
      }
    }
  }

  std::vector<std::byte> payload;
  const auto align_payload = [&](std::size_t alignment) {
    while (payload.size() % alignment != 0U) {
      payload.push_back(std::byte{0});
    }
  };
  const std::uint64_t p_codes_off = payload.size();
  payload.insert(payload.end(), nax_codes.begin(), nax_codes.end());
  align_payload(256U);
  std::uint64_t p_high_off = 0;
  if (!nax_high.empty()) {
    p_high_off = payload.size();
    payload.insert(payload.end(), nax_high.begin(), nax_high.end());
    align_payload(256U);
  }
  const std::uint64_t p_sb_off = payload.size();
  payload.insert(payload.end(), nax_sb.begin(), nax_sb.end());
  const std::uint64_t obj_bytes = payload.size();
  const auto obj_sha = omlx::m5::artifact_sha256_hex(payload);

  const std::string fmt = "oq_affine_q" + std::to_string(bits) + "_g" +
                          std::to_string(group_size) + "_bf16_v1";
  const std::string layout =
      layout_override.empty()
          ? (group_size == 128U ? "oq_nax_lane_g128_v2" : "oq_nax_lane_g64_v2")
          : layout_override;
  const std::string schema = (version_override == 2U)
                                 ? "omlx-m5-artifact-v2"
                                 : "omlx-m5-artifact-v1";
  const std::string transform =
      transform_override.empty() ? (with_fwht64 ? "fwht64" : "")
                                 : transform_override;

  std::ostringstream json;
  json << "{\"bindings\":[{\"activation_policy\":\"A16Only\","
       << "\"format\":\"" << fmt << "\",\"in_features\":" << k
       << ",\"leading_shape\":[],\"name\":\"model.layers.0.mlp.down_proj\","
       << "\"object_id\":0,\"out_features\":" << n << ",\"use\":\"mlp_down\"}],"
       << "\"container_version\":" << version_override << ",";
  if (version_override == 2U) {
    json << "\"format_features\":{\"fwht64_prerotated\":"
         << (with_fwht64 ? "true" : "false")
         << ",\"fwht64_targets\":[\"mlp_down\"],"
         << "\"nax_lane_layout\":\"pi64_quartet_v2\"},";
  }
  json << "\"model\":{\"source_profile\":\"oq_affine\","
       << "\"source_repo\":\"test/v2\",\"source_revision\":\"rev2\"},"
       << "\"objects\":[{\"format\":\"" << fmt << "\",\"id\":0,";
  if (!transform.empty()) {
    json << "\"input_transform\":\"" << transform << "\",";
  }
  json << "\"layout\":\"" << layout
       << "\",\"name\":\"model.layers.0.mlp.down_proj\","
       << "\"object_bytes\":" << obj_bytes << ",\"object_offset\":0,"
       << "\"object_sha256\":\"" << obj_sha << "\",\"padded_k\":" << padded_k
       << ",\"planes\":["
       << "{\"alignment\":256,\"bytes\":" << nax_codes.size() << ",\"crc32\":\""
       << hex_crc32(omlx::m5::artifact_crc32(nax_codes))
       << "\",\"kind\":\"nax_codes\",\"offset\":" << p_codes_off
       << ",\"sha256\":\"" << omlx::m5::artifact_sha256_hex(nax_codes) << "\"},";
  if (!nax_high.empty()) {
    const std::string hkind = (bits == 3U) ? "nax_high1" : "nax_high2";
    json << "{\"alignment\":256,\"bytes\":" << nax_high.size() << ",\"crc32\":\""
         << hex_crc32(omlx::m5::artifact_crc32(nax_high)) << "\",\"kind\":\""
         << hkind << "\",\"offset\":" << p_high_off << ",\"sha256\":\""
         << omlx::m5::artifact_sha256_hex(nax_high) << "\"},";
  }
  json << "{\"alignment\":256,\"bytes\":" << nax_sb.size() << ",\"crc32\":\""
       << hex_crc32(omlx::m5::artifact_crc32(nax_sb))
       << "\",\"kind\":\"nax_sb\",\"offset\":" << p_sb_off << ",\"sha256\":\""
       << omlx::m5::artifact_sha256_hex(nax_sb) << "\"}"
       << "],\"shape\":[" << n << "," << k
       << "],\"source_tensors\":[\"model.layers.0.mlp.down_proj.weight\"]}],"
       << "\"payload_bytes\":" << payload.size() << ",\"schema\":\"" << schema
       << "\",\"views\":[]}";

  const auto manifest_str = json.str();
  const auto manifest_bytes =
      std::as_bytes(std::span(manifest_str.data(), manifest_str.size()));
  const std::uint64_t raw_end = 64ULL + manifest_bytes.size();
  const std::uint64_t payload_offset = (raw_end + 16383ULL) & ~16383ULL;
  const std::uint32_t manifest_crc = omlx::m5::artifact_crc32(manifest_bytes);

  std::vector<std::byte> file;
  constexpr char magic[8] = {'O', 'M', 'L', 'X', 'M', '5', 'A', '1'};
  for (char c : magic) {
    file.push_back(static_cast<std::byte>(c));
  }
  append_u32(file, version_override);
  append_u32(file, 64U);
  append_u64(file, 64ULL);
  append_u64(file, manifest_bytes.size());
  append_u64(file, payload_offset);
  append_u64(file, payload.size());
  append_u32(file, 1U);
  append_u32(file, manifest_crc);
  const std::uint32_t header_crc =
      omlx::m5::artifact_crc32(std::span<const std::byte>(file.data(), 56U));
  append_u32(file, header_crc);
  append_u32(file, flags_override);
  file.insert(file.end(), manifest_bytes.begin(), manifest_bytes.end());
  file.resize(static_cast<std::size_t>(payload_offset), std::byte{0});
  file.insert(file.end(), payload.begin(), payload.end());
  return file;
}

void test_v2_artifact_pi64_fwht64_and_materialization() {
  // Verify pi64 involution on all 64 offsets
  for (std::uint32_t off = 0; off < 64U; ++off) {
    expect(pi64_ref(pi64_ref(off)) == off, "pi64 is not involutive");
  }

  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       ("omlx-m5-b0-v2-" + std::to_string(::getpid()));
  std::filesystem::create_directories(tmp_dir);
  struct Cleanup {
    std::filesystem::path dir;
    ~Cleanup() { std::filesystem::remove_all(dir); }
  } cleanup{tmp_dir};

  const auto write_bytes = [](const std::filesystem::path &p,
                              const std::vector<std::byte> &data) {
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data.data()),
              static_cast<std::streamsize>(data.size()));
  };

  auto ctx = omlx::m5::MetalContext::create();
  constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 6> kConfigs = {
      {{3U, 64U}, {4U, 64U}, {5U, 64U}, {6U, 64U}, {8U, 64U}, {5U, 128U}}};

  for (const auto [bits, gs] : kConfigs) {
    for (const bool with_fwht : {false, true}) {
      const std::uint32_t flags =
          omlx::m5::kArtifactFlagNaxLaneV2 |
          (with_fwht ? omlx::m5::kArtifactFlagFwht64 : 0U);
      std::vector<float> raw_ref;
      std::vector<float> logical_ref;
      const auto p = tmp_dir / ("v2_q" + std::to_string(bits) + "_g" +
                                std::to_string(gs) +
                                (with_fwht ? "_fwht.miv" : "_plain.miv"));
      write_bytes(p, build_v2_affine_lane_artifact(
                         bits, gs, 20U, 512U, with_fwht, 2U, flags, "", "",
                         &raw_ref, &logical_ref));
      const auto art = omlx::m5::ArtifactModel::open(p, true);
      expect(art.summary().container_version == 2U &&
                 art.summary().container_flags == flags,
             "v2 summary version/flags mismatch");
      const auto &obj = art.object(0U);
      expect(obj.has_fwht64() == with_fwht &&
                 obj.input_transform == (with_fwht ? "fwht64" : "none"),
             "ArtifactObject input_transform mismatch");

      for (const std::uint32_t row : {0U, 15U, 16U, 19U}) {
        const auto dec_raw =
            art.decode_weight_row(obj, row, 0U, 512U, /*logical_unrotated=*/false);
        const auto dec_log =
            art.decode_weight_row(obj, row, 0U, 512U, /*logical_unrotated=*/true);
        const auto dec_slice =
            art.decode_weight_row(obj, row, 11U, 83U, /*logical_unrotated=*/true);
        for (std::size_t col = 0; col < 512U; ++col) {
          expect(dec_raw[col] == raw_ref[std::size_t(row) * 512U + col],
                 "v2 pi64 raw decode_weight_row mismatch");
          expect(dec_log[col] == logical_ref[std::size_t(row) * 512U + col],
                 "v2 fwht64 logical decode_weight_row mismatch");
        }
        for (std::size_t i = 0; i < 83U; ++i) {
          expect(dec_slice[i] == logical_ref[std::size_t(row) * 512U + 11U + i],
                 "v2 unaligned sub-slice decode_weight_row mismatch");
        }
      }

      // Materialization propagation through parent_view and binding_view
      auto plan = omlx::m5::plan_model_materialization(
          art, ctx, {.require_verified_payload = true});
      auto mat = omlx::m5::materialize_model(art, std::move(plan), ctx);
      const auto pv = mat.parent_view(0U);
      const auto bv = mat.binding_view("model.layers.0.mlp.down_proj");
      expect(pv.has_fwht64() == with_fwht &&
                 pv.input_transform == obj.input_transform &&
                 bv.weight.has_fwht64() == with_fwht &&
                 bv.weight.input_transform == obj.input_transform,
             "MaterializedWeightView did not propagate input_transform");
    }
  }

  // Negative checks: invalid version, invalid flags, v1 + v2 features, mismatched group layout
  const auto expect_open_fail = [&](const std::filesystem::path &path,
                                    const std::string &why) {
    bool failed = false;
    try {
      static_cast<void>(omlx::m5::ArtifactModel::open(path, false));
    } catch (const std::exception &) {
      failed = true;
    }
    expect(failed, why);
  };

  const auto p_bad_flags = tmp_dir / "bad_v2_flags.miv";
  write_bytes(p_bad_flags,
              build_v2_affine_lane_artifact(4U, 64U, 16U, 512U, true, 2U, 0x4U,
                                            "", ""));
  expect_open_fail(p_bad_flags, "v2 artifact with reserved flags=0x4 accepted");

  const auto p_bad_ver = tmp_dir / "bad_v3.miv";
  write_bytes(p_bad_ver,
              build_v2_affine_lane_artifact(4U, 64U, 16U, 512U, false, 3U, 0U,
                                            "", ""));
  expect_open_fail(p_bad_ver, "unsupported version=3 artifact accepted");

  const auto p_v1_lane = tmp_dir / "bad_v1_lane_v2.miv";
  write_bytes(p_v1_lane,
              build_v2_affine_lane_artifact(4U, 64U, 16U, 512U, false, 1U, 0U,
                                            "oq_nax_lane_g64_v2", ""));
  expect_open_fail(p_v1_lane, "v1 artifact with oq_nax_lane_g64_v2 accepted");

  const auto p_v1_fwht = tmp_dir / "bad_v1_fwht.miv";
  write_bytes(p_v1_fwht,
              build_v2_affine_lane_artifact(4U, 64U, 16U, 512U, true, 1U, 0U,
                                            "oq_nax_tiled_g64_v1", "fwht64"));
  expect_open_fail(p_v1_fwht, "v1 artifact with input_transform=fwht64 accepted");

  const auto p_bad_xf = tmp_dir / "bad_transform.miv";
  write_bytes(p_bad_xf,
              build_v2_affine_lane_artifact(4U, 64U, 16U, 512U, false, 2U, 0x2U,
                                            "oq_nax_lane_g64_v2", "hadamard128"));
  expect_open_fail(p_bad_xf, "v2 artifact with unknown input_transform accepted");

  const auto p_bad_gs = tmp_dir / "bad_gs_layout.miv";
  write_bytes(p_bad_gs,
              build_v2_affine_lane_artifact(4U, 128U, 16U, 512U, false, 2U, 0x2U,
                                            "oq_nax_lane_g64_v2", ""));
  expect_open_fail(p_bad_gs, "g128 affine format with oq_nax_lane_g64_v2 accepted");
}

} // namespace

int main() {
  try {
    test_crc32_contract_and_speed();
    test_16kib_and_legacy_4kib_artifacts();
    test_affine8_nax_pretile_contract_and_decode();
    test_v2_artifact_pi64_fwht64_and_materialization();
    std::cout << "b0 artifact native tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "b0 artifact native test failure: " << error.what() << '\n';
    return 1;
  }
}
