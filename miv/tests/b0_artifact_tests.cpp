#include <array>
#include <chrono>
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

} // namespace

int main() {
  try {
    test_crc32_contract_and_speed();
    test_16kib_and_legacy_4kib_artifacts();
    std::cout << "b0 artifact native tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "b0 artifact native test failure: " << error.what() << '\n';
    return 1;
  }
}
