#include "omlx_m5/artifact.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <CommonCrypto/CommonDigest.h>
#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

#include "omlx_m5/quantization.h"

namespace omlx::m5 {
namespace {

using Json = nlohmann::json;

constexpr std::uint32_t kVersionV1 = 1;
constexpr std::uint32_t kVersionV2 = 2;
constexpr std::uint32_t kHeaderBytes = 64;
constexpr std::uint64_t kObjectAlignment = 256;
constexpr std::uint64_t kPlaneAlignment = 256;
constexpr std::size_t kVerifyStreamChunkBytes = 256 * 1024;

bool is_supported_magic(const std::byte *header) noexcept {
  const auto *magic = reinterpret_cast<const char *>(header);
  return std::equal(kArtifactMagic.begin(), kArtifactMagic.end(), magic) ||
         std::equal(kLegacyArtifactMagic.begin(), kLegacyArtifactMagic.end(),
                    magic);
}

bool is_supported_schema(std::string_view schema,
                         std::uint32_t version) noexcept {
  if (version == kVersionV2) {
    return schema == kArtifactSchemaV2 || schema == kLegacyArtifactSchemaV2;
  }
  if (version == kVersionV1) {
    return schema == kArtifactSchemaV1 || schema == kLegacyArtifactSchemaV1;
  }
  return false;
}

constexpr bool is_affine_nax_v2_layout(std::string_view layout) noexcept {
  return layout == "oq_nax_lane_g64_v2" || layout == "oq_nax_lane_g128_v2";
}

constexpr std::uint64_t pi64_slot(std::uint64_t off) noexcept {
  return ((off & 12ULL) << 2U) | ((off >> 2U) & 12ULL) | (off & 3ULL);
}

void fwht64_block_inplace(float *block) noexcept {
  for (std::size_t h = 1; h < 64; h <<= 1) {
    const std::size_t step = h << 1;
    for (std::size_t i = 0; i < 64; i += step) {
      for (std::size_t j = 0; j < h; ++j) {
        const float a = block[i + j];
        const float b = block[i + j + h];
        block[i + j] = a + b;
        block[i + j + h] = a - b;
      }
    }
  }
  for (std::size_t i = 0; i < 64; ++i) {
    block[i] *= 0.125F;
  }
}

std::uint32_t read_u32(const std::byte *source) {
  return static_cast<std::uint32_t>(static_cast<std::uint8_t>(source[0])) |
         (static_cast<std::uint32_t>(static_cast<std::uint8_t>(source[1]))
          << 8) |
         (static_cast<std::uint32_t>(static_cast<std::uint8_t>(source[2]))
          << 16) |
         (static_cast<std::uint32_t>(static_cast<std::uint8_t>(source[3]))
          << 24);
}

std::uint64_t read_u64(const std::byte *source) {
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < 8; ++byte) {
    value |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(source[byte]))
             << (byte * 8);
  }
  return value;
}

std::uint32_t parse_crc(std::string_view value) {
  if (value.size() != 8) {
    throw std::runtime_error("artifact CRC32 must have eight hex digits");
  }
  std::size_t parsed = 0;
  const auto result = std::stoul(std::string(value), &parsed, 16);
  if (parsed != value.size() ||
      result > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("artifact CRC32 is invalid");
  }
  return static_cast<std::uint32_t>(result);
}

class Sha256Hasher {
public:
  Sha256Hasher() {
    if (CC_SHA256_Init(&context_) != 1) {
      throw std::runtime_error("unable to initialize SHA-256");
    }
  }

  void update(std::span<const std::byte> bytes) {
    auto remaining = bytes.size();
    const auto *current = bytes.data();
    while (remaining != 0) {
      const auto chunk = static_cast<CC_LONG>(
          std::min<std::size_t>(remaining, std::numeric_limits<CC_LONG>::max()));
      if (CC_SHA256_Update(&context_, current, chunk) != 1) {
        throw std::runtime_error("unable to update SHA-256");
      }
      current += chunk;
      remaining -= chunk;
    }
  }

  [[nodiscard]] std::string finish_hex() {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    if (CC_SHA256_Final(digest.data(), &context_) != 1) {
      throw std::runtime_error("unable to finalize SHA-256");
    }
    constexpr std::string_view kHex = "0123456789abcdef";
    std::string output;
    output.reserve(digest.size() * 2);
    for (const auto byte : digest) {
      output.push_back(kHex[byte >> 4]);
      output.push_back(kHex[byte & 0x0FU]);
    }
    return output;
  }

private:
  CC_SHA256_CTX context_{};
};

std::string sha256_hex(std::span<const std::byte> bytes) {
  Sha256Hasher hasher;
  hasher.update(bytes);
  return hasher.finish_hex();
}

bool is_power_of_two(std::uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          std::string_view label) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::overflow_error(std::string(label) + " overflow");
  }
  return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               std::string_view label) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error(std::string(label) + " overflow");
  }
  return left * right;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment,
                       std::string_view label) {
  if (!is_power_of_two(alignment)) {
    throw std::invalid_argument(std::string(label) + " alignment is invalid");
  }
  const auto mask = alignment - 1;
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
    throw std::overflow_error(std::string(label) + " alignment overflow");
  }
  return (value + mask) & ~mask;
}

bool is_lower_hex(std::string_view value, std::size_t digits) {
  return value.size() == digits &&
         std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

void require_zero(std::span<const std::byte> bytes, std::string_view label) {
  const auto *ptr = reinterpret_cast<const std::uint8_t *>(bytes.data());
  std::size_t len = bytes.size();
  while (len != 0 && (reinterpret_cast<std::uintptr_t>(ptr) & 7U) != 0) {
    if (*ptr++ != 0) {
      throw std::runtime_error(std::string(label) + " is not zero-filled");
    }
    --len;
  }
  while (len >= 8) {
    std::uint64_t word = 0;
    std::memcpy(&word, ptr, sizeof(word));
    if (word != 0) {
      throw std::runtime_error(std::string(label) + " is not zero-filled");
    }
    ptr += 8;
    len -= 8;
  }
  while (len != 0) {
    if (*ptr++ != 0) {
      throw std::runtime_error(std::string(label) + " is not zero-filled");
    }
    --len;
  }
}

const ArtifactPlane &require_plane(const ArtifactObject &object,
                                   std::size_t index, std::string_view kind,
                                   std::uint64_t bytes) {
  if (index >= object.planes.size() || object.planes[index].kind != kind ||
      object.planes[index].bytes != bytes) {
    throw std::runtime_error("artifact quantized plane contract mismatch: " +
                             object.name);
  }
  return object.planes[index];
}

struct AffineFormatSpec {
  std::string_view id;
  std::uint32_t bits;
  std::uint32_t group_size;
};

const AffineFormatSpec *affine_format(std::string_view id) {
  static constexpr std::array<AffineFormatSpec, 10> formats = {{
      {"oq_affine_q3_g64_bf16_v1", 3, 64},
      {"oq_affine_q3_g128_bf16_v1", 3, 128},
      {"oq_affine_q4_g64_bf16_v1", 4, 64},
      {"oq_affine_q4_g128_bf16_v1", 4, 128},
      {"oq_affine_q5_g64_bf16_v1", 5, 64},
      {"oq_affine_q5_g128_bf16_v1", 5, 128},
      {"oq_affine_q6_g64_bf16_v1", 6, 64},
      {"oq_affine_q6_g128_bf16_v1", 6, 128},
      {"oq_affine_q8_g64_bf16_v1", 8, 64},
      {"oq_affine_q8_g128_bf16_v1", 8, 128},
  }};
  const auto found = std::ranges::find(formats, id, &AffineFormatSpec::id);
  return found == formats.end() ? nullptr : &*found;
}

std::uint64_t flattened_rows(const ArtifactObject &object) {
  if (object.shape.size() < 2) {
    throw std::runtime_error("artifact quantized rank is invalid");
  }
  std::uint64_t rows = 1;
  for (std::size_t index = 0; index + 1 < object.shape.size(); ++index) {
    if (object.shape[index] == 0) {
      throw std::runtime_error("artifact quantized dimension is zero");
    }
    rows = checked_multiply(rows, object.shape[index], "quantized rows");
  }
  return rows;
}

void validate_format_contract(const ArtifactObject &object,
                              std::uint32_t container_version) {
  if (object.input_transform != "none" && object.input_transform != "fwht64") {
    throw std::runtime_error("unsupported artifact input_transform: " +
                             object.input_transform);
  }
  if (object.input_transform == "fwht64") {
    if (container_version < kVersionV2) {
      throw std::runtime_error(
          "artifact input_transform=fwht64 requires container version 2");
    }
    if (object.format.starts_with("direct_") || object.shape.size() < 2 ||
        object.shape.back() < 64 || (object.shape.back() % 64) != 0) {
      throw std::runtime_error(
          "artifact input_transform=fwht64 requires quantized weight with K >= 64 and K % 64 == 0");
    }
  }
  if (object.format.starts_with("direct_")) {
    if (object.layout != "contiguous_le_v1" || object.planes.size() != 1 ||
        object.planes.front().kind != "data" || object.padded_k != 0) {
      throw std::runtime_error("artifact direct layout contract mismatch");
    }
    constexpr std::array<std::pair<std::string_view, std::uint64_t>, 16>
        kDtypeBytes = {{{"bool", 1},
                        {"u8", 1},
                        {"i8", 1},
                        {"f8_e4m3", 1},
                        {"f8_e5m2", 1},
                        {"f8_e8m0", 1},
                        {"u16", 2},
                        {"i16", 2},
                        {"f16", 2},
                        {"bf16", 2},
                        {"u32", 4},
                        {"i32", 4},
                        {"f32", 4},
                        {"u64", 8},
                        {"i64", 8},
                        {"f64", 8}}};
    if (!object.format.ends_with("_v1")) {
      throw std::runtime_error("artifact direct format version is invalid");
    }
    const auto dtype =
        std::string_view(object.format)
            .substr(std::string_view("direct_").size(),
                    object.format.size() - std::string_view("direct_").size() -
                        std::string_view("_v1").size());
    const auto dtype_entry = std::ranges::find_if(
        kDtypeBytes, [&](const auto &entry) { return entry.first == dtype; });
    if (dtype_entry == kDtypeBytes.end()) {
      throw std::runtime_error("artifact direct dtype is unsupported");
    }
    std::uint64_t elements = 1;
    for (const auto dimension : object.shape) {
      elements = checked_multiply(elements, dimension, "direct tensor");
    }
    const auto expected =
        checked_multiply(elements, dtype_entry->second, "direct tensor");
    if (object.planes.front().bytes != expected) {
      throw std::runtime_error("artifact direct plane size mismatch");
    }
    return;
  }
  if (object.shape.size() < 2 || object.shape.back() == 0 ||
      object.padded_k < object.shape.back() || object.padded_k % 128 != 0) {
    throw std::runtime_error("artifact quantized geometry is invalid");
  }
  const auto rows = flattened_rows(object);
  const auto padded_k = object.padded_k;
  if (const auto *affine = affine_format(object.format); affine != nullptr) {
    const bool is_nax_v1 = (object.layout == "oq_nax_tiled_g64_v1");
    const bool is_nax_v2 = is_affine_nax_v2_layout(object.layout);
    if (is_nax_v2 && container_version < kVersionV2) {
      throw std::runtime_error(
          "artifact v2 NAX lane layout requires container version 2");
    }
    if (is_nax_v1 || is_nax_v2) {
      const bool v2_gs_ok =
          !is_nax_v2 ||
          (object.layout == "oq_nax_lane_g64_v2" && affine->group_size == 64) ||
          (object.layout == "oq_nax_lane_g128_v2" && affine->group_size == 128);
      if (!v2_gs_ok || object.shape.size() != 2 ||
          (affine->group_size != 64 && affine->group_size != 128) ||
          (affine->bits != 3 && affine->bits != 4 && affine->bits != 5 &&
           affine->bits != 6 && affine->bits != 8) ||
          object.shape.back() < 512 ||
          (object.shape.back() % affine->group_size) != 0) {
        throw std::runtime_error("artifact affine NAX tiled geometry mismatch");
      }
      const auto k = object.shape.back();
      const std::uint64_t bn = rows >= 65536 ? 32 : 16;
      const auto padded_n = ((rows + bn - 1) / bn) * bn;
      std::size_t index = 0;
      if (affine->bits == 3) {
        static_cast<void>(require_plane(
            object, index++, "nax_codes",
            checked_multiply(padded_n, k / 4, "affine Q3 NAX codes")));
        static_cast<void>(require_plane(
            object, index++, "nax_high1",
            checked_multiply(padded_n, k / 8, "affine Q3 NAX high1")));
      } else if (affine->bits == 8) {
        static_cast<void>(require_plane(
            object, index++, "nax_codes",
            checked_multiply(padded_n, k, "affine Q8 NAX codes")));
      } else {
        static_cast<void>(require_plane(
            object, index++, "nax_codes",
            checked_multiply(padded_n, k / 2, "affine NAX codes")));
        if (affine->bits == 5 || affine->bits == 6) {
          static_cast<void>(require_plane(
              object, index++, "nax_high2",
              checked_multiply(padded_n, k / 4, "affine NAX high2")));
        }
      }
      static_cast<void>(require_plane(
          object, index++, "nax_sb",
          checked_multiply(padded_n, (k / 64) * 4, "affine NAX sb")));
      if (index != object.planes.size()) {
        throw std::runtime_error("artifact affine NAX plane count mismatch");
      }
      return;
    }
    const auto expected_layout = object.shape.size() == 2
                                     ? "oq_rowsplit_k128_v1"
                                     : "oq_expert_rowsplit_k128_v1";
    if (object.layout != expected_layout) {
      throw std::runtime_error("artifact affine layout contract mismatch");
    }
    std::size_t index = 0;
    if (affine->bits == 8) {
      static_cast<void>(require_plane(object, index++, "code",
                                      checked_multiply(rows, padded_k, "Q8")));
    } else if (affine->bits == 3) {
      static_cast<void>(require_plane(
          object, index++, "low_2bit",
          checked_multiply(rows, padded_k / 4, "Q3 low2")));
      static_cast<void>(require_plane(
          object, index++, "high_bit",
          checked_multiply(rows, padded_k / 8, "Q3 high1")));
    } else {
      static_cast<void>(
          require_plane(object, index++, "base_code",
                        checked_multiply(rows, padded_k / 2, "affine base")));
      if (affine->bits == 5) {
        static_cast<void>(
            require_plane(object, index++, "high_bit",
                          checked_multiply(rows, padded_k / 8, "Q5 high")));
      } else if (affine->bits == 6) {
        static_cast<void>(
            require_plane(object, index++, "high_2bit",
                          checked_multiply(rows, padded_k / 4, "Q6 high")));
      }
    }
    const auto group_bytes = checked_multiply(
        rows, padded_k / affine->group_size * 2, "affine groups");
    static_cast<void>(require_plane(object, index++, "scale", group_bytes));
    static_cast<void>(require_plane(object, index++, "bias", group_bytes));
    if (index != object.planes.size()) {
      throw std::runtime_error("artifact affine plane count mismatch");
    }
    return;
  }
  if (object.format == "mxfp4_e2m1_g32_e8m0_v1") {
    if (object.layout == "mx_nax_tiled_bk128_v1") {
      if (object.shape.size() != 2 || object.shape.back() < 512 ||
          (object.shape.back() % 128) != 0 || object.planes.size() != 2) {
        throw std::runtime_error("artifact MXFP4 NAX tiled geometry mismatch");
      }
      const auto k = object.shape.back();
      const std::uint64_t bn = rows >= 65536 ? 32 : 16;
      const auto padded_n = ((rows + bn - 1) / bn) * bn;
      static_cast<void>(require_plane(
          object, 0, "nax_data",
          checked_multiply(padded_n, k / 2, "MXFP4 NAX data")));
      static_cast<void>(require_plane(
          object, 1, "nax_scale",
          checked_multiply(padded_n, k / 32, "MXFP4 NAX scale")));
      return;
    }
    const auto expected_layout = object.shape.size() == 2
                                     ? "mx_rowsplit_k128_v1"
                                     : "mx_expert_rowsplit_k128_v1";
    if (object.layout != expected_layout || object.planes.size() != 2) {
      throw std::runtime_error("artifact MXFP4 layout contract mismatch");
    }
    static_cast<void>(require_plane(
        object, 0, "data", checked_multiply(rows, padded_k / 2, "MXFP4")));
    static_cast<void>(require_plane(
        object, 1, "scale", checked_multiply(rows, padded_k / 32, "MXFP4")));
    return;
  }
  throw std::runtime_error("artifact format is unsupported: " + object.format);
}

class MappedFile {
public:
  explicit MappedFile(const std::filesystem::path &path) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("unable to open artifact " + path.string());
    }
    struct stat status {};
    if (::fstat(fd_, &status) != 0 || status.st_size <= 0) {
      close_file();
      throw std::runtime_error("unable to stat artifact " + path.string());
    }
    size_ = static_cast<std::uint64_t>(status.st_size);
    mapping_ = ::mmap(nullptr, static_cast<std::size_t>(size_), PROT_READ,
                      MAP_PRIVATE, fd_, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      close_file();
      throw std::runtime_error("unable to mmap artifact " + path.string());
    }
  }

  MappedFile(MappedFile &&other) noexcept
      : path_(std::move(other.path_)), fd_(std::exchange(other.fd_, -1)),
        mapping_(std::exchange(other.mapping_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}
  MappedFile &operator=(MappedFile &&other) noexcept {
    if (this != &other) {
      reset();
      path_ = std::move(other.path_);
      fd_ = std::exchange(other.fd_, -1);
      mapping_ = std::exchange(other.mapping_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }
  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;
  ~MappedFile() { reset(); }

  [[nodiscard]] const std::byte *data() const {
    return static_cast<const std::byte *>(mapping_);
  }
  [[nodiscard]] std::uint64_t size() const { return size_; }

  void advise_sequential(bool sequential) const {
    if (mapping_ != nullptr && size_ != 0) {
      static_cast<void>(::posix_madvise(
          mapping_, static_cast<std::size_t>(size_),
          sequential ? POSIX_MADV_SEQUENTIAL : POSIX_MADV_NORMAL));
    }
  }

private:
  void close_file() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }
  void reset() {
    if (mapping_ != nullptr) {
      ::munmap(mapping_, static_cast<std::size_t>(size_));
      mapping_ = nullptr;
    }
    close_file();
    size_ = 0;
  }

  std::filesystem::path path_;
  int fd_ = -1;
  void *mapping_ = nullptr;
  std::uint64_t size_ = 0;
};

float decode_e2m1(std::uint8_t code) {
  constexpr std::array<float, 8> magnitudes = {0.0F, 0.5F, 1.0F, 1.5F,
                                               2.0F, 3.0F, 4.0F, 6.0F};
  const auto magnitude = magnitudes[code & 0x07U];
  return (code & 0x08U) == 0 ? magnitude : -magnitude;
}

float decode_e8m0(std::uint8_t code) {
  if (code == 0xFFU) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (code == 0U) {
    return std::ldexp(1.0F, -127);
  }
  return std::ldexp(1.0F, static_cast<int>(code) - 127);
}

void require_range(std::uint64_t begin, std::uint64_t bytes,
                   std::uint64_t limit, std::string_view label) {
  if (begin > limit || bytes > limit - begin) {
    throw std::runtime_error(std::string(label) + " exceeds artifact range");
  }
}

std::size_t checked_size(std::uint64_t value, std::string_view label) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error(std::string(label) + " exceeds address space");
  }
  return static_cast<std::size_t>(value);
}

#if !defined(__ARM_FEATURE_CRC32)
constexpr std::array<std::uint32_t, 256> make_crc32_table() {
  std::array<std::uint32_t, 256> table{};
  for (std::uint32_t index = 0; index < 256; ++index) {
    std::uint32_t crc = index;
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ ((crc & 1U) != 0U ? 0xEDB88320U : 0U);
    }
    table[index] = crc;
  }
  return table;
}
#endif

void verify_object_payload(const std::byte *payload_base,
                           const ArtifactObject &object) {
  const bool single_exact_plane =
      object.planes.size() == 1 &&
      object.planes.front().offset == object.object_offset &&
      object.planes.front().bytes == object.object_bytes;
  std::optional<Sha256Hasher> object_hasher;
  if (!single_exact_plane) {
    object_hasher.emplace();
  }
  std::string first_plane_sha256;
  std::uint64_t cursor = object.object_offset;
  for (const auto &plane : object.planes) {
    if (object_hasher.has_value() && plane.offset > cursor) {
      object_hasher->update(
          {payload_base + cursor,
           checked_size(plane.offset - cursor, "plane padding")});
    }
    const auto *plane_data = payload_base + plane.offset;
    auto remaining = checked_size(plane.bytes, "plane");
    std::uint32_t crc = 0;
    Sha256Hasher plane_hasher;
    std::size_t offset = 0;
    while (remaining != 0) {
      const auto chunk_len = std::min(remaining, kVerifyStreamChunkBytes);
      const std::span<const std::byte> chunk(plane_data + offset, chunk_len);
      crc = artifact_crc32(chunk, crc);
      plane_hasher.update(chunk);
      if (object_hasher.has_value()) {
        object_hasher->update(chunk);
      }
      offset += chunk_len;
      remaining -= chunk_len;
    }
    auto plane_sha = plane_hasher.finish_hex();
    if (crc != plane.crc32 || plane_sha != plane.sha256) {
      throw std::runtime_error("artifact plane checksum mismatch: " +
                               object.name + "/" + plane.kind);
    }
    if (single_exact_plane) {
      first_plane_sha256 = std::move(plane_sha);
    }
    cursor = plane.offset + plane.bytes;
  }
  const auto object_sha =
      single_exact_plane ? first_plane_sha256 : object_hasher->finish_hex();
  if (object_sha != object.object_sha256) {
    throw std::runtime_error("artifact object checksum mismatch: " +
                             object.name);
  }
}

void verify_objects_parallel(const std::byte *payload_base,
                             const std::vector<ArtifactObject> &objects,
                             std::uint64_t payload_bytes) {
  if (objects.empty()) {
    return;
  }
  const std::size_t hw = std::max(1U, std::thread::hardware_concurrency());
  const std::size_t workers =
      (objects.size() < 2 || payload_bytes < (4ULL << 20))
          ? 1
          : std::min<std::size_t>({hw, 16, objects.size()});
  if (workers <= 1) {
    for (const auto &object : objects) {
      verify_object_payload(payload_base, object);
    }
    return;
  }
  std::atomic<std::size_t> next{0};
  std::atomic<std::size_t> first_fail{objects.size()};
  std::mutex err_mutex;
  std::exception_ptr captured;
  std::size_t captured_idx = objects.size();

  auto worker_fn = [&]() {
    while (true) {
      const std::size_t idx = next.fetch_add(1, std::memory_order_relaxed);
      if (idx >= objects.size() ||
          idx >= first_fail.load(std::memory_order_relaxed)) {
        break;
      }
      try {
        verify_object_payload(payload_base, objects[idx]);
      } catch (...) {
        std::scoped_lock lock(err_mutex);
        if (idx < captured_idx) {
          captured_idx = idx;
          captured = std::current_exception();
          first_fail.store(idx, std::memory_order_relaxed);
        }
        break;
      }
    }
  };

  std::vector<std::thread> pool;
  pool.reserve(workers - 1);
  for (std::size_t w = 1; w < workers; ++w) {
    pool.emplace_back(worker_fn);
  }
  worker_fn();
  for (auto &t : pool) {
    t.join();
  }
  if (captured) {
    std::rethrow_exception(captured);
  }
}

} // namespace

std::uint32_t artifact_crc32(std::span<const std::byte> bytes,
                             std::uint32_t initial) {
  std::uint32_t crc = ~initial;
  const auto *ptr = reinterpret_cast<const std::uint8_t *>(bytes.data());
  std::size_t len = bytes.size();
#if defined(__ARM_FEATURE_CRC32)
  while (len != 0 && (reinterpret_cast<std::uintptr_t>(ptr) & 7U) != 0) {
    crc = __crc32b(crc, *ptr++);
    --len;
  }
  while (len >= 64) {
    std::uint64_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
    std::uint64_t w4 = 0, w5 = 0, w6 = 0, w7 = 0;
    std::memcpy(&w0, ptr + 0, 8);
    std::memcpy(&w1, ptr + 8, 8);
    std::memcpy(&w2, ptr + 16, 8);
    std::memcpy(&w3, ptr + 24, 8);
    std::memcpy(&w4, ptr + 32, 8);
    std::memcpy(&w5, ptr + 40, 8);
    std::memcpy(&w6, ptr + 48, 8);
    std::memcpy(&w7, ptr + 56, 8);
    crc = __crc32d(crc, w0);
    crc = __crc32d(crc, w1);
    crc = __crc32d(crc, w2);
    crc = __crc32d(crc, w3);
    crc = __crc32d(crc, w4);
    crc = __crc32d(crc, w5);
    crc = __crc32d(crc, w6);
    crc = __crc32d(crc, w7);
    ptr += 64;
    len -= 64;
  }
  while (len >= 8) {
    std::uint64_t word = 0;
    std::memcpy(&word, ptr, 8);
    crc = __crc32d(crc, word);
    ptr += 8;
    len -= 8;
  }
  while (len != 0) {
    crc = __crc32b(crc, *ptr++);
    --len;
  }
#else
  static constexpr auto kTable = make_crc32_table();
  while (len != 0) {
    crc = (crc >> 8) ^ kTable[(crc ^ *ptr++) & 0xFFU];
    --len;
  }
#endif
  return ~crc;
}

std::string artifact_sha256_hex(std::span<const std::byte> bytes) {
  return sha256_hex(bytes);
}

class ArtifactModel::Impl {
public:
  std::filesystem::path path;
  MappedFile file;
  ArtifactSummary summary;
  std::vector<ArtifactObject> objects;
  std::vector<ArtifactBinding> bindings;
  std::unordered_map<std::string, std::uint32_t> object_ids;

  Impl(std::filesystem::path selected_path, MappedFile selected_file)
      : path(std::move(selected_path)), file(std::move(selected_file)) {}
};

ArtifactModel ArtifactModel::open(const std::filesystem::path &path,
                                  bool verify_payload) {
  auto file = MappedFile(path);
  if (file.size() < kHeaderBytes) {
    throw std::runtime_error("artifact header is truncated");
  }
  const auto *header = file.data();
  if (!is_supported_magic(header)) {
    throw std::runtime_error("artifact magic mismatch");
  }
  const std::string detected_magic(reinterpret_cast<const char *>(header), 8);
  const auto version = read_u32(header + 8);
  const auto header_bytes = read_u32(header + 12);
  const auto manifest_offset = read_u64(header + 16);
  const auto manifest_bytes = read_u64(header + 24);
  const auto payload_offset = read_u64(header + 32);
  const auto payload_bytes = read_u64(header + 40);
  const auto object_count = read_u32(header + 48);
  const auto manifest_crc = read_u32(header + 52);
  const auto header_crc = read_u32(header + 56);
  const auto flags = read_u32(header + 60);
  const bool valid_version = (version == kVersionV1 || version == kVersionV2);
  const bool valid_flags = (version == kVersionV1)
                               ? (flags == 0)
                               : ((flags & ~kArtifactValidFlagsMask) == 0);
  if (!valid_version || header_bytes != kHeaderBytes || !valid_flags) {
    throw std::runtime_error("unsupported artifact header");
  }
  if (artifact_crc32({header, 56}) != header_crc) {
    throw std::runtime_error("artifact header checksum mismatch");
  }
  if (manifest_offset != kHeaderBytes ||
      payload_offset % kLegacyPayloadAlignment != 0) {
    throw std::runtime_error("artifact offsets violate container contract");
  }
  require_range(manifest_offset, manifest_bytes, payload_offset,
                "artifact manifest");
  require_range(payload_offset, payload_bytes, file.size(), "artifact payload");
  if (checked_add(payload_offset, payload_bytes, "artifact file end") !=
      file.size()) {
    throw std::runtime_error("artifact has trailing or missing payload bytes");
  }
  const auto manifest_span = std::span<const std::byte>(
      file.data() + manifest_offset, checked_size(manifest_bytes, "manifest"));
  if (artifact_crc32(manifest_span) != manifest_crc) {
    throw std::runtime_error("artifact manifest checksum mismatch");
  }
  Json manifest;
  try {
    const auto *begin = reinterpret_cast<const char *>(manifest_span.data());
    manifest = Json::parse(begin, begin + manifest_span.size());
  } catch (const Json::exception &error) {
    throw std::runtime_error(std::string("invalid artifact manifest JSON: ") +
                             error.what());
  }
  const auto *manifest_begin =
      reinterpret_cast<const char *>(manifest_span.data());
  if (manifest.dump() !=
      std::string_view(manifest_begin, manifest_span.size())) {
    throw std::runtime_error("artifact manifest is not canonical JSON");
  }
  const std::string manifest_schema = manifest.value("schema", "");
  if (!is_supported_schema(manifest_schema, version) ||
      manifest.value("container_version", 0U) != version ||
      manifest.value("payload_bytes", std::uint64_t{0}) != payload_bytes ||
      !manifest.contains("model") || !manifest.contains("objects") ||
      !manifest.contains("views") || !manifest.contains("bindings") ||
      !manifest.at("objects").is_array() || !manifest.at("views").is_array() ||
      !manifest.at("views").empty() || !manifest.at("bindings").is_array() ||
      manifest.at("objects").size() != object_count ||
      (version == kVersionV2 && manifest.contains("format_features") &&
       !manifest.at("format_features").is_object())) {
    throw std::runtime_error("artifact manifest contract mismatch");
  }
  const auto manifest_end =
      checked_add(manifest_offset, manifest_bytes, "artifact manifest end");
  require_zero(
      {file.data() + manifest_end,
       checked_size(payload_offset - manifest_end, "manifest padding")},
      "artifact manifest padding");

  auto impl = std::make_unique<Impl>(std::filesystem::weakly_canonical(path),
                                     std::move(file));
  impl->summary = {
      .container_version = version,
      .artifact_bytes = impl->file.size(),
      .manifest_bytes = manifest_bytes,
      .payload_offset = payload_offset,
      .payload_bytes = payload_bytes,
      .source_repo = manifest.at("model").value("source_repo", ""),
      .source_revision = manifest.at("model").value("source_revision", ""),
      .source_profile = manifest.at("model").value("source_profile", ""),
      .payload_verified = verify_payload,
      .container_flags = flags,
      .container_magic = detected_magic,
      .manifest_schema = manifest_schema,
  };
  if (verify_payload) {
    impl->file.advise_sequential(true);
  }
  std::uint64_t previous_object_end = 0;
  for (std::size_t index = 0; index < manifest.at("objects").size(); ++index) {
    const auto &record = manifest.at("objects").at(index);
    ArtifactObject object;
    object.id = record.at("id").get<std::uint32_t>();
    object.name = record.at("name").get<std::string>();
    object.format = record.at("format").get<std::string>();
    object.layout = record.at("layout").get<std::string>();
    object.shape = record.at("shape").get<std::vector<std::uint64_t>>();
    object.padded_k = record.value("padded_k", 0ULL);
    object.object_offset = record.at("object_offset").get<std::uint64_t>();
    object.object_bytes = record.at("object_bytes").get<std::uint64_t>();
    object.object_sha256 = record.at("object_sha256").get<std::string>();
    object.source_tensors =
        record.at("source_tensors").get<std::vector<std::string>>();
    if (record.contains("input_transform")) {
      if (!record.at("input_transform").is_string()) {
        throw std::runtime_error(
            "artifact object input_transform must be a string");
      }
      object.input_transform = record.at("input_transform").get<std::string>();
    } else {
      object.input_transform = "none";
    }
    if (object.id != index || object.name.empty() || object.object_bytes == 0 ||
        object.object_offset % kObjectAlignment != 0 ||
        object.object_offset < previous_object_end ||
        !is_lower_hex(object.object_sha256, 64) ||
        object.source_tensors.empty()) {
      throw std::runtime_error("artifact object order or identity is invalid");
    }
    require_zero({impl->file.data() + payload_offset + previous_object_end,
                  checked_size(object.object_offset - previous_object_end,
                               "object padding")},
                 "artifact object padding");
    const auto object_end = checked_add(
        object.object_offset, object.object_bytes, "artifact object end");
    if (object_end > payload_bytes) {
      throw std::runtime_error("artifact object exceeds payload");
    }
    std::uint64_t previous_plane_end = object.object_offset;
    for (const auto &plane_record : record.at("planes")) {
      ArtifactPlane plane;
      plane.kind = plane_record.at("kind").get<std::string>();
      plane.offset = plane_record.at("offset").get<std::uint64_t>();
      plane.bytes = plane_record.at("bytes").get<std::uint64_t>();
      plane.alignment = plane_record.at("alignment").get<std::uint64_t>();
      plane.crc32 = parse_crc(plane_record.at("crc32").get<std::string>());
      plane.sha256 = plane_record.at("sha256").get<std::string>();
      if (plane.kind.empty() || plane.bytes == 0 ||
          plane.alignment != kPlaneAlignment ||
          plane.offset % plane.alignment != 0 ||
          plane.offset < previous_plane_end ||
          !is_lower_hex(plane.sha256, 64) ||
          checked_add(plane.offset, plane.bytes, "artifact plane end") >
              object_end) {
        throw std::runtime_error("artifact plane range is invalid");
      }
      if (std::ranges::any_of(object.planes, [&](const ArtifactPlane &value) {
            return value.kind == plane.kind;
          })) {
        throw std::runtime_error("artifact object has duplicate plane kinds");
      }
      require_zero(
          {impl->file.data() + payload_offset + previous_plane_end,
           checked_size(plane.offset - previous_plane_end, "plane padding")},
          "artifact plane padding");
      previous_plane_end = plane.offset + plane.bytes;
      object.planes.push_back(std::move(plane));
    }
    if (previous_plane_end != object_end) {
      throw std::runtime_error("artifact object has trailing bytes");
    }
    validate_format_contract(object, version);
    const auto [_, inserted] = impl->object_ids.emplace(object.name, object.id);
    if (!inserted) {
      throw std::runtime_error("artifact object name is duplicated");
    }
    previous_object_end = object_end;
    impl->objects.push_back(std::move(object));
  }
  if (previous_object_end != payload_bytes) {
    throw std::runtime_error("artifact payload has trailing bytes");
  }
  if (verify_payload) {
    verify_objects_parallel(impl->file.data() + payload_offset, impl->objects,
                            payload_bytes);
    impl->file.advise_sequential(false);
  }
  std::unordered_set<std::uint32_t> bound_ids;
  std::unordered_set<std::uint32_t> primary_bound_ids;
  std::unordered_set<std::string> binding_names;
  for (const auto &record : manifest.at("bindings")) {
    ArtifactBinding binding{
        .name = record.at("name").get<std::string>(),
        .object_id = record.at("object_id").get<std::uint32_t>(),
        .format = record.at("format").get<std::string>(),
        .use = record.at("use").get<std::string>(),
        .activation_policy = record.at("activation_policy").get<std::string>(),
        .leading_shape =
            record.value("leading_shape", std::vector<std::uint64_t>{}),
        .out_features = record.at("out_features").get<std::uint64_t>(),
        .in_features = record.at("in_features").get<std::uint64_t>(),
    };
    if (binding.object_id >= impl->objects.size() || binding.use.empty() ||
        !binding_names.emplace(binding.name).second ||
        binding.format != impl->objects[binding.object_id].format ||
        (binding.activation_policy != "A16Only" &&
         binding.activation_policy != "AllowA4")) {
      throw std::runtime_error("artifact binding is invalid");
    }
    const auto &object = impl->objects[binding.object_id];
    const bool is_primary = (binding.name == object.name);
    const bool is_lm_head_alias =
        !is_primary && binding.use == "lm_head" &&
        (binding.name == "lm_head" || binding.name == "lm_head.weight" ||
         binding.name.ends_with(".lm_head") ||
         binding.name.ends_with(".lm_head.weight"));
    if (!is_primary && !is_lm_head_alias) {
      throw std::runtime_error("artifact binding is invalid");
    }
    if (is_primary && !primary_bound_ids.emplace(binding.object_id).second) {
      throw std::runtime_error("artifact binding is invalid");
    }
    bound_ids.insert(binding.object_id);
    if (object.shape.size() < 2) {
      throw std::runtime_error("artifact binding rank is invalid");
    }
    const auto expected_leading = std::vector<std::uint64_t>(
        object.shape.begin(), object.shape.end() - 2);
    if (binding.leading_shape != expected_leading ||
        binding.out_features != object.shape[object.shape.size() - 2] ||
        binding.in_features != object.shape.back() ||
        binding.activation_policy != (binding.format == "mxfp4_e2m1_g32_e8m0_v1"
                                          ? "AllowA4"
                                          : "A16Only")) {
      throw std::runtime_error("artifact binding geometry is invalid");
    }
    impl->bindings.push_back(std::move(binding));
  }
  for (const auto &object : impl->objects) {
    const bool quantized = !object.format.starts_with("direct_");
    if (quantized && !bound_ids.contains(object.id)) {
      throw std::runtime_error(
          "artifact quantized binding coverage is invalid");
    }
  }
  return ArtifactModel(std::move(impl));
}

ArtifactModel::ArtifactModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ArtifactModel::ArtifactModel(ArtifactModel &&) noexcept = default;
ArtifactModel &ArtifactModel::operator=(ArtifactModel &&) noexcept = default;
ArtifactModel::~ArtifactModel() = default;

const std::filesystem::path &ArtifactModel::path() const { return impl_->path; }
const ArtifactSummary &ArtifactModel::summary() const { return impl_->summary; }
void ArtifactModel::verify_payload() {
  if (impl_->summary.payload_verified) {
    return;
  }
  impl_->file.advise_sequential(true);
  const auto *payload_base = impl_->file.data() + impl_->summary.payload_offset;
  verify_objects_parallel(payload_base, impl_->objects,
                          impl_->summary.payload_bytes);
  impl_->file.advise_sequential(false);
  impl_->summary.payload_verified = true;
}
const std::vector<ArtifactObject> &ArtifactModel::objects() const {
  return impl_->objects;
}
const std::vector<ArtifactBinding> &ArtifactModel::bindings() const {
  return impl_->bindings;
}

const ArtifactObject &ArtifactModel::object(std::uint32_t id) const {
  if (id >= impl_->objects.size()) {
    throw std::out_of_range("artifact object id is out of range");
  }
  return impl_->objects[id];
}

const ArtifactObject &ArtifactModel::object(std::string_view name) const {
  const auto it = impl_->object_ids.find(std::string(name));
  if (it == impl_->object_ids.end()) {
    throw std::out_of_range("unknown artifact object: " + std::string(name));
  }
  return object(it->second);
}

const ArtifactPlane &ArtifactModel::plane(const ArtifactObject &selected_object,
                                          std::string_view kind) const {
  const auto it = std::ranges::find_if(
      selected_object.planes,
      [&](const ArtifactPlane &value) { return value.kind == kind; });
  if (it == selected_object.planes.end()) {
    throw std::out_of_range("artifact plane is missing: " +
                            selected_object.name + "/" + std::string(kind));
  }
  return *it;
}

std::span<const std::byte>
ArtifactModel::plane_bytes(const ArtifactObject &selected_object,
                           const ArtifactPlane &selected_plane) const {
  const auto owner = std::ranges::find_if(
      selected_object.planes,
      [&](const ArtifactPlane &value) { return &value == &selected_plane; });
  if (owner == selected_object.planes.end()) {
    throw std::invalid_argument("artifact plane does not belong to object");
  }
  return {impl_->file.data() + impl_->summary.payload_offset +
              selected_plane.offset,
          checked_size(selected_plane.bytes, "plane")};
}

ArtifactMaterializationPlan
ArtifactModel::materialization_plan(std::uint64_t alignment) const {
  if (alignment < kPlaneAlignment || !is_power_of_two(alignment)) {
    throw std::invalid_argument(
        "materialization alignment must be a power of two at least 256");
  }
  ArtifactMaterializationPlan plan;
  plan.alignment = alignment;
  plan.payload_bytes = impl_->summary.payload_bytes;
  for (const auto &object : impl_->objects) {
    for (const auto &selected_plane : object.planes) {
      const auto destination =
          align_up(plan.arena_bytes, alignment, "materialization arena");
      plan.arena_padding_bytes =
          checked_add(plan.arena_padding_bytes, destination - plan.arena_bytes,
                      "materialization padding");
      plan.upload_bytes = checked_add(plan.upload_bytes, selected_plane.bytes,
                                      "materialization upload bytes");
      plan.arena_bytes = checked_add(destination, selected_plane.bytes,
                                     "materialization arena bytes");
      plan.spans.push_back({
          .object_id = object.id,
          .object_name = object.name,
          .plane_kind = selected_plane.kind,
          .source_file_offset =
              checked_add(impl_->summary.payload_offset, selected_plane.offset,
                          "materialization source offset"),
          .destination_offset = destination,
          .bytes = selected_plane.bytes,
      });
    }
  }
  return plan;
}

std::vector<float>
ArtifactModel::decode_weight_row(const ArtifactObject &selected_object,
                                 std::uint64_t row, std::uint64_t start,
                                 std::uint64_t count,
                                 bool logical_unrotated) const {
  const auto rows = flattened_rows(selected_object);
  const auto logical_k = selected_object.shape.back();
  if (row >= rows || start > logical_k || count > logical_k - start ||
      selected_object.padded_k < logical_k) {
    throw std::out_of_range("artifact decode range is invalid");
  }
  if (count == 0) {
    return {};
  }
  const bool unrotate = logical_unrotated && selected_object.has_fwht64();
  std::uint64_t decode_start = start;
  std::uint64_t decode_count = count;
  if (unrotate) {
    if (logical_k < 64 || (logical_k % 64) != 0) {
      throw std::runtime_error(
          "artifact fwht64 decode requires K >= 64 and divisible by 64");
    }
    decode_start = (start / 64ULL) * 64ULL;
    const auto decode_end = ((start + count + 63ULL) / 64ULL) * 64ULL;
    decode_count = decode_end - decode_start;
  }
  const auto finalize_row = [&](std::vector<float> raw) -> std::vector<float> {
    if (!unrotate) {
      return raw;
    }
    for (std::size_t off = 0; off < raw.size(); off += 64U) {
      fwht64_block_inplace(raw.data() + off);
    }
    const auto slice_off = static_cast<std::ptrdiff_t>(start - decode_start);
    return std::vector<float>(raw.begin() + slice_off,
                              raw.begin() + slice_off +
                                  static_cast<std::ptrdiff_t>(count));
  };
  const auto padded_k = selected_object.padded_k;
  std::vector<float> result;
  result.reserve(static_cast<std::size_t>(decode_count));

  if (const auto *affine = affine_format(selected_object.format);
      affine != nullptr) {
    const bool is_nax_v2 = is_affine_nax_v2_layout(selected_object.layout);
    if (selected_object.layout == "oq_nax_tiled_g64_v1" || is_nax_v2) {
      const std::uint64_t bn = rows >= 65536 ? 32 : 16;
      const auto padded_n = ((rows + bn - 1) / bn) * bn;
      const auto groups = logical_k / 64;
      const auto codes =
          plane_bytes(selected_object, plane(selected_object, "nax_codes"));
      const auto sb =
          plane_bytes(selected_object, plane(selected_object, "nax_sb"));
      const auto high =
          affine->bits == 3
              ? plane_bytes(selected_object, plane(selected_object, "nax_high1"))
          : (affine->bits == 5 || affine->bits == 6)
              ? plane_bytes(selected_object, plane(selected_object, "nax_high2"))
              : std::span<const std::byte>{};
      const auto rn = row % bn;
      for (std::uint64_t column = decode_start;
           column < decode_start + decode_count; ++column) {
        const auto g = column / 64;
        const auto col_in_g = column % 64;
        const auto slot = is_nax_v2 ? pi64_slot(col_in_g) : col_in_g;
        const auto tile_idx = (row / bn) * groups + g;
        std::uint32_t code = 0;
        if (affine->bits == 3) {
          const auto c_byte = static_cast<std::uint8_t>(
              codes[tile_idx * (bn * 16) + rn * 16 + slot / 4]);
          const auto h_byte = static_cast<std::uint8_t>(
              high[tile_idx * (bn * 8) + rn * 8 + slot / 8]);
          code = ((c_byte >> ((slot & 3U) * 2U)) & 0x03U) |
                 (((h_byte >> (slot & 7U)) & 1U) << 2U);
        } else if (affine->bits == 8) {
          code = static_cast<std::uint8_t>(
              codes[tile_idx * (bn * 64) + rn * 64 + slot]);
        } else {
          const auto c_byte = static_cast<std::uint8_t>(
              codes[tile_idx * (bn * 32) + rn * 32 + slot / 2]);
          code = (c_byte >> ((slot & 1U) * 4U)) & 0x0FU;
          if (affine->bits == 5 || affine->bits == 6) {
            const auto h_byte = static_cast<std::uint8_t>(
                high[tile_idx * (bn * 16) + rn * 16 + slot / 4]);
            const auto h2 = (h_byte >> ((slot & 3U) * 2U)) & 0x03U;
            code |= (affine->bits == 5 ? (h2 & 1U) : h2) << 4U;
          }
        }
        std::uint32_t packed_sb = 0;
        std::memcpy(&packed_sb, sb.data() + (g * padded_n + row) * 4, 4);
        const auto scale_word = static_cast<std::uint16_t>(packed_sb & 0xFFFFU);
        const auto bias_word = static_cast<std::uint16_t>(packed_sb >> 16U);
        result.push_back(std::fma(bf16_to_float(scale_word),
                                  static_cast<float>(code),
                                  bf16_to_float(bias_word)));
      }
      return finalize_row(std::move(result));
    }
    const auto code_kind = affine->bits == 8   ? "code"
                           : affine->bits == 3 ? "low_2bit"
                                               : "base_code";
    const auto base =
        plane_bytes(selected_object, plane(selected_object, code_kind));
    const auto scale =
        plane_bytes(selected_object, plane(selected_object, "scale"));
    const auto bias =
        plane_bytes(selected_object, plane(selected_object, "bias"));
    const auto high =
        (affine->bits == 3 || affine->bits == 5)
            ? plane_bytes(selected_object, plane(selected_object, "high_bit"))
        : affine->bits == 6
            ? plane_bytes(selected_object, plane(selected_object, "high_2bit"))
            : std::span<const std::byte>{};
    const auto base_row = affine->bits == 8   ? padded_k
                          : affine->bits == 3 ? padded_k / 4
                                              : padded_k / 2;
    const auto high_row =
        (affine->bits == 3 || affine->bits == 5) ? padded_k / 8 : padded_k / 4;
    const auto group_row = padded_k / affine->group_size;
    for (std::uint64_t column = decode_start;
         column < decode_start + decode_count; ++column) {
      std::uint32_t code = 0;
      if (affine->bits == 8) {
        code = static_cast<std::uint8_t>(base[row * base_row + column]);
      } else if (affine->bits == 3) {
        const auto low_byte =
            static_cast<std::uint8_t>(base[row * base_row + column / 4]);
        const auto high_byte =
            static_cast<std::uint8_t>(high[row * high_row + column / 8]);
        code = ((low_byte >> ((column & 3U) * 2U)) & 0x03U) |
               (((high_byte >> (column & 7U)) & 1U) << 2U);
      } else {
        const auto base_byte =
            static_cast<std::uint8_t>(base[row * base_row + column / 2]);
        code = (base_byte >> ((column & 1U) * 4U)) & 0x0FU;
      }
      if (affine->bits == 5) {
        const auto high_byte =
            static_cast<std::uint8_t>(high[row * high_row + column / 8]);
        code |= ((high_byte >> (column & 7U)) & 1U) << 4U;
      } else if (affine->bits == 6) {
        const auto high_byte =
            static_cast<std::uint8_t>(high[row * high_row + column / 4]);
        code |= ((high_byte >> ((column & 3U) * 2U)) & 3U) << 4U;
      }
      const auto group = column / affine->group_size;
      const auto scale_word =
          static_cast<std::uint16_t>(
              static_cast<std::uint8_t>(scale[(row * group_row + group) * 2])) |
          (static_cast<std::uint16_t>(static_cast<std::uint8_t>(
               scale[(row * group_row + group) * 2 + 1]))
           << 8);
      const auto bias_word =
          static_cast<std::uint16_t>(
              static_cast<std::uint8_t>(bias[(row * group_row + group) * 2])) |
          (static_cast<std::uint16_t>(static_cast<std::uint8_t>(
               bias[(row * group_row + group) * 2 + 1]))
           << 8);
      result.push_back(std::fma(bf16_to_float(scale_word),
                                static_cast<float>(code),
                                bf16_to_float(bias_word)));
    }
    return finalize_row(std::move(result));
  }

  if (selected_object.format == "mxfp4_e2m1_g32_e8m0_v1") {
    if (selected_object.layout == "mx_nax_tiled_bk128_v1") {
      const std::uint64_t bn = rows >= 65536 ? 32 : 16;
      const auto k_blocks = logical_k / 128;
      const auto data =
          plane_bytes(selected_object, plane(selected_object, "nax_data"));
      const auto scale =
          plane_bytes(selected_object, plane(selected_object, "nax_scale"));
      const auto rn = row % bn;
      for (std::uint64_t column = decode_start;
           column < decode_start + decode_count; ++column) {
        const auto blk = column / 128;
        const auto col_in_blk = column % 128;
        const auto tile_idx = (row / bn) * k_blocks + blk;
        const auto packed = static_cast<std::uint8_t>(
            data[tile_idx * (bn * 64) + rn * 64 + col_in_blk / 2]);
        const auto code = static_cast<std::uint8_t>(
            (packed >> ((col_in_blk & 1U) * 4U)) & 0x0FU);
        const auto scale_code = static_cast<std::uint8_t>(
            scale[tile_idx * (bn * 4) + rn * 4 + col_in_blk / 32]);
        const auto value = decode_e2m1(code) * decode_e8m0(scale_code);
        if (!std::isfinite(value)) {
          throw std::runtime_error("MXFP4 decode produced a non-finite value");
        }
        result.push_back(value);
      }
      return finalize_row(std::move(result));
    }
    const auto data =
        plane_bytes(selected_object, plane(selected_object, "data"));
    const auto scale =
        plane_bytes(selected_object, plane(selected_object, "scale"));
    const auto data_row = padded_k / 2;
    const auto scale_row = padded_k / 32;
    for (std::uint64_t column = decode_start;
         column < decode_start + decode_count; ++column) {
      const auto packed =
          static_cast<std::uint8_t>(data[row * data_row + column / 2]);
      const auto code =
          static_cast<std::uint8_t>((packed >> ((column & 1U) * 4U)) & 0x0FU);
      const auto scale_code =
          static_cast<std::uint8_t>(scale[row * scale_row + column / 32]);
      const auto value = decode_e2m1(code) * decode_e8m0(scale_code);
      if (!std::isfinite(value)) {
        throw std::runtime_error("MXFP4 decode produced a non-finite value");
      }
      result.push_back(value);
    }
    return finalize_row(std::move(result));
  }

  throw std::invalid_argument("artifact object is not a decodable R1 weight: " +
                              selected_object.format);
}

} // namespace omlx::m5
