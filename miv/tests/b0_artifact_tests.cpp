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

} // namespace

int main() {
  try {
    test_crc32_contract_and_speed();
    test_16kib_and_legacy_4kib_artifacts();
    test_affine8_nax_pretile_contract_and_decode();
    std::cout << "b0 artifact native tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "b0 artifact native test failure: " << error.what() << '\n';
    return 1;
  }
}
