#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#import <objc/message.h>

#include "ane/ane_dual_offload.h"
#include "omlx_m5/quantization.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>

@protocol OmlxM5AneMetalDeviceIOSurface <NSObject>
- (nullable id<MTLBuffer>)newBufferWithIOSurface:(IOSurfaceRef)surface;
@end

namespace omlx::m5::ane {
namespace {

using Json = nlohmann::json;

constexpr std::uint32_t kAneCacheSchemaVersion = 1u;
constexpr unsigned int kAneQoS = 21u;
constexpr std::chrono::milliseconds kAneCacheLockTimeout{30000};
constexpr std::chrono::milliseconds kAneCacheLockRetry{50};

struct AnePackParamsHost {
  std::uint32_t T = 0;
  std::uint32_t K = 0;
  std::uint32_t src_stride = 0;
  std::uint32_t reserved = 0;
};

struct AneUnpackParamsHost {
  std::uint32_t T = 0;
  std::uint32_t ane0_n = 0;
  std::uint32_t ane1_n = 0;
  std::uint32_t out_stride = 0;
  std::uint32_t dst_col_offset = 0;
  std::uint32_t reserved = 0;
};

// Exact IEEE-754 float32 <-> float16 bit conversions (pure C++20, -Wpedantic clean).
std::uint16_t float_to_f16_bits(float value) noexcept {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16u) & 0x8000u;
  const std::uint32_t exp = (bits >> 23u) & 0xFFu;
  const std::uint32_t mant = bits & 0x007FFFFFu;

  if (exp == 0xFFu) {
    if (mant == 0u) {
      return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    return static_cast<std::uint16_t>(sign | 0x7E00u | (mant >> 13u));
  }

  const int unbiased = static_cast<int>(exp) - 127;
  if (unbiased > 15) {
    return static_cast<std::uint16_t>(sign | 0x7BFFu); // Clamp to finite max fp16
  }
  if (unbiased < -24) {
    return static_cast<std::uint16_t>(sign);
  }
  if (unbiased < -14) {
    // Subnormal fp16
    const std::uint32_t sig = 0x00800000u | mant;
    const int shift = -14 - unbiased + 13; // in [14..23]
    const std::uint32_t half_lsb = 1u << (shift - 1);
    const std::uint32_t mask = (1u << shift) - 1u;
    std::uint32_t sub = sig >> shift;
    const std::uint32_t rem = sig & mask;
    if (rem > half_lsb || (rem == half_lsb && (sub & 1u) != 0u)) {
      ++sub;
    }
    return static_cast<std::uint16_t>(sign | sub);
  }

  const std::uint32_t h_exp = static_cast<std::uint32_t>(unbiased + 15);
  std::uint32_t h_mant = mant >> 13u;
  const std::uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h_mant & 1u) != 0u)) {
    ++h_mant;
    if (h_mant == 0x400u) {
      h_mant = 0u;
      if (h_exp + 1u >= 31u) {
        return static_cast<std::uint16_t>(sign | 0x7BFFu);
      }
      return static_cast<std::uint16_t>(sign | ((h_exp + 1u) << 10u));
    }
  }
  return static_cast<std::uint16_t>(sign | (h_exp << 10u) | h_mant);
}

float f16_bits_to_float(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16u;
  const std::uint32_t exp = (h >> 10u) & 0x1Fu;
  const std::uint32_t mant = h & 0x03FFu;
  if (exp == 0u) {
    if (mant == 0u) {
      return std::bit_cast<float>(sign);
    }
    // Subnormal
    float v = std::ldexp(static_cast<float>(mant), -24);
    return sign != 0u ? -v : v;
  }
  if (exp == 31u) {
    const std::uint32_t f_bits =
        sign | 0x7F800000u | (mant << 13u);
    return std::bit_cast<float>(f_bits);
  }
  const std::uint32_t f_bits =
      sign | ((exp + 112u) << 23u) | (mant << 13u);
  return std::bit_cast<float>(f_bits);
}

std::string sha256_hex_bytes(const void *data, std::size_t length) {
  CC_SHA256_CTX ctx{};
  if (CC_SHA256_Init(&ctx) != 1) {
    throw std::runtime_error("ANE SHA-256 init failed");
  }
  const auto *ptr = static_cast<const unsigned char *>(data);
  std::size_t remaining = length;
  while (remaining > 0) {
    const auto chunk = static_cast<CC_LONG>(
        std::min<std::size_t>(remaining, std::numeric_limits<CC_LONG>::max()));
    if (CC_SHA256_Update(&ctx, ptr, chunk) != 1) {
      throw std::runtime_error("ANE SHA-256 update failed");
    }
    ptr += chunk;
    remaining -= chunk;
  }
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  if (CC_SHA256_Final(digest.data(), &ctx) != 1) {
    throw std::runtime_error("ANE SHA-256 finalize failed");
  }
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(digest.size() * 2);
  for (unsigned char b : digest) {
    out.push_back(kHex[b >> 4u]);
    out.push_back(kHex[b & 0x0Fu]);
  }
  return out;
}

std::string sha256_hex_string(std::string_view s) {
  return sha256_hex_bytes(s.data(), s.size());
}

std::string ns_error_text(NSString *prefix, NSError *error) {
  NSString *msg =
      error ? [NSString stringWithFormat:@"%@: %@", prefix, error] : prefix;
  return msg ? std::string([msg UTF8String]) : std::string("ANE error");
}

std::string sanitize_os_build() {
  @autoreleasepool {
    NSString *os_build =
        [[NSProcessInfo processInfo] operatingSystemVersionString];
    if (!os_build) {
      return "unknown_os";
    }
    os_build = [os_build stringByReplacingOccurrencesOfString:@"/" withString:@"_"];
    os_build = [os_build stringByReplacingOccurrencesOfString:@":" withString:@"_"];
    os_build = [os_build stringByReplacingOccurrencesOfString:@" " withString:@"_"];
    return std::string([os_build UTF8String]);
  }
}

bool ensure_ane_framework_loaded() noexcept {
  static const bool loaded = [] {
    return dlopen(
               "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
               "AppleNeuralEngine",
               RTLD_NOW | RTLD_LOCAL) != nullptr;
  }();
  return loaded;
}

std::uint32_t detect_iokit_ane_dies() noexcept {
  io_iterator_t iter = 0;
  CFMutableDictionaryRef matching = IOServiceMatching("H11ANEIn");
  if (matching == nullptr) {
    return 0u;
  }
  if (IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iter) !=
      KERN_SUCCESS) {
    return 0u;
  }
  std::uint32_t count = 0u;
  io_object_t service = 0;
  while ((service = IOIteratorNext(iter)) != 0) {
    ++count;
    IOObjectRelease(service);
  }
  IOObjectRelease(iter);
  return count;
}

std::uint32_t detect_ane_die_count() noexcept {
  const std::uint32_t iokit_dies = detect_iokit_ane_dies();
  if (iokit_dies > 0u) {
    return iokit_dies;
  }
  Class dev_info = NSClassFromString(@"_ANEDeviceInfo");
  if (dev_info != Nil) {
    if ([dev_info respondsToSelector:@selector(numANEs)]) {
      auto fn = reinterpret_cast<int (*)(Class, SEL)>(&objc_msgSend);
      const int n = fn(dev_info, @selector(numANEs));
      if (n > 0) {
        return static_cast<std::uint32_t>(n);
      }
    }
    if ([dev_info respondsToSelector:@selector(hasANE)]) {
      auto fn = reinterpret_cast<BOOL (*)(Class, SEL)>(&objc_msgSend);
      if (fn(dev_info, @selector(hasANE))) {
        return 1u;
      }
    }
  }
  return 0u;
}

// Cross-process flock() guard for disk cache and temp staging safety.
class ScopedFileLock final {
public:
  explicit ScopedFileLock(const std::filesystem::path &lock_path) {
    if (lock_path.has_parent_path()) {
      std::error_code ec;
      std::filesystem::create_directories(lock_path.parent_path(), ec);
      if (ec) {
        throw std::runtime_error("failed to create ANE cache lock directory: " +
                                 ec.message());
      }
    }
    fd_ = ::open(lock_path.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd_ < 0) {
      throw std::runtime_error("failed to open ANE cache lock file: " +
                               lock_path.string());
    }
    const auto deadline =
        std::chrono::steady_clock::now() + kAneCacheLockTimeout;
    for (;;) {
      if (::flock(fd_, LOCK_EX | LOCK_NB) == 0) {
        return;
      }
      if (errno != EWOULDBLOCK && errno != EAGAIN) {
        break;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        break;
      }
      std::this_thread::sleep_for(kAneCacheLockRetry);
    }
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("timed out acquiring ANE cache flock: " +
                             lock_path.string());
  }

  ~ScopedFileLock() {
    if (fd_ >= 0) {
      ::flock(fd_, LOCK_UN);
      ::close(fd_);
      fd_ = -1;
    }
  }

  ScopedFileLock(const ScopedFileLock &) = delete;
  ScopedFileLock &operator=(const ScopedFileLock &) = delete;

private:
  int fd_ = -1;
};

void write_file_atomic(const std::filesystem::path &dest, const void *bytes,
                       std::size_t size) {
  if (dest.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(dest.parent_path(), ec);
    if (ec) {
      throw std::runtime_error("failed to create ANE cache parent dir: " +
                               ec.message());
    }
  }
  auto tmp = dest;
  tmp += ".tmp." + std::to_string(static_cast<long>(::getpid()));
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("failed to open temp ANE cache file: " +
                               tmp.string());
    }
    if (size > 0) {
      out.write(static_cast<const char *>(bytes),
                static_cast<std::streamsize>(size));
      if (!out) {
        throw std::runtime_error("failed to write temp ANE cache file: " +
                                 tmp.string());
      }
    }
    out.flush();
    if (!out) {
      throw std::runtime_error("failed to flush temp ANE cache file: " +
                               tmp.string());
    }
  }
  std::error_code ec;
  std::filesystem::rename(tmp, dest, ec);
  if (ec) {
    std::filesystem::remove(tmp, ec);
    throw std::runtime_error("failed to commit ANE cache file: " +
                             dest.string());
  }
}

std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::runtime_error("failed to open ANE cache file: " + path.string());
  }
  const auto end_pos = in.tellg();
  if (end_pos < 0) {
    throw std::runtime_error("failed to stat ANE cache file: " + path.string());
  }
  std::vector<std::uint8_t> buf(static_cast<std::size_t>(end_pos));
  in.seekg(0, std::ios::beg);
  if (!buf.empty()) {
    in.read(reinterpret_cast<char *>(buf.data()),
            static_cast<std::streamsize>(buf.size()));
    if (!in) {
      throw std::runtime_error("failed to read ANE cache file: " +
                               path.string());
    }
  }
  return buf;
}

// Weight decoding helpers matching miv/src/artifact.cpp exactly.
struct AffineFormatInfo {
  std::uint32_t bits = 0;
  std::uint32_t group_size = 0;
};

std::optional<AffineFormatInfo> parse_affine_format(std::string_view fmt) noexcept {
  struct Entry {
    std::string_view id;
    std::uint32_t bits;
    std::uint32_t gs;
  };
  static constexpr std::array<Entry, 10> kFormats = {{
      {"oq_affine_q3_g64_bf16_v1", 3u, 64u},
      {"oq_affine_q3_g128_bf16_v1", 3u, 128u},
      {"oq_affine_q4_g64_bf16_v1", 4u, 64u},
      {"oq_affine_q4_g128_bf16_v1", 4u, 128u},
      {"oq_affine_q5_g64_bf16_v1", 5u, 64u},
      {"oq_affine_q5_g128_bf16_v1", 5u, 128u},
      {"oq_affine_q6_g64_bf16_v1", 6u, 64u},
      {"oq_affine_q6_g128_bf16_v1", 6u, 128u},
      {"oq_affine_q8_g64_bf16_v1", 8u, 64u},
      {"oq_affine_q8_g128_bf16_v1", 8u, 128u},
  }};
  for (const auto &e : kFormats) {
    if (e.id == fmt) {
      return AffineFormatInfo{e.bits, e.gs};
    }
  }
  return std::nullopt;
}

float decode_e2m1(std::uint8_t code) noexcept {
  static constexpr std::array<float, 8> kMag = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float m = kMag[code & 0x07u];
  return (code & 0x08u) == 0u ? m : -m;
}

float decode_e8m0(std::uint8_t code) noexcept {
  if (code == 0xFFu) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (code == 0u) {
    return std::ldexp(1.0f, -127);
  }
  return std::ldexp(1.0f, static_cast<int>(code) - 127);
}

const MaterializedPlaneView &require_view_plane(
    const MaterializedWeightView &w, std::string_view kind) {
  for (const auto &p : w.planes) {
    if (p.kind == kind) {
      if (p.buffer == nullptr || p.bytes == 0) {
        throw std::runtime_error("ANE weight plane is empty: " + w.name + "/" +
                                 std::string(kind));
      }
      return p;
    }
  }
  throw std::out_of_range("ANE weight plane missing: " + w.name + "/" +
                          std::string(kind));
}

std::span<const std::byte> plane_span(const MaterializedPlaneView &p) {
  return p.buffer->view(p.offset, p.bytes);
}

std::uint16_t load_u16_le(const std::byte *ptr) noexcept {
  std::uint16_t v = 0;
  std::memcpy(&v, ptr, sizeof(v));
  return v;
}

std::uint32_t load_u32_le(const std::byte *ptr) noexcept {
  std::uint32_t v = 0;
  std::memcpy(&v, ptr, sizeof(v));
  return v;
}

float load_f32_le(const std::byte *ptr) noexcept {
  float v = 0.0f;
  std::memcpy(&v, ptr, sizeof(v));
  return v;
}

// Quantize one single row [in_features] of FP32 weights into groupwise G=64 asymmetric INT8.
void quantize_row_g64_int8(const float *src_row, std::uint32_t in_features,
                           std::int8_t *dst_data, std::uint16_t *dst_scale,
                           std::int8_t *dst_offset) noexcept {
  constexpr float kMinNormalFp16 = 6.103515625e-5f; // 2^-14
  const std::uint32_t ng = in_features / kAneGroupSize;
  for (std::uint32_t g = 0; g < ng; ++g) {
    const float *grp = src_row + static_cast<std::size_t>(g) * kAneGroupSize;
    float lo = 0.0f;
    float hi = 0.0f;
    for (std::uint32_t i = 0; i < kAneGroupSize; ++i) {
      const float v = std::isfinite(grp[i]) ? grp[i] : 0.0f;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const float range = hi - lo;
    std::int8_t *q_out = dst_data + static_cast<std::size_t>(g) * kAneGroupSize;
    if (range <= 1.0e-9f) {
      dst_scale[g] = float_to_f16_bits(1.0f);
      dst_offset[g] = 0;
      std::memset(q_out, 0, kAneGroupSize);
      continue;
    }

    const float raw_scale = std::max(range / 255.0f, kMinNormalFp16);
    std::uint16_t sc_bits = float_to_f16_bits(raw_scale);
    float sc = f16_bits_to_float(sc_bits);
    // Ensure fp16 scale rounds up enough that [lo, hi] fits in 255 steps without clipping.
    if (sc * 255.0f < range && sc_bits < 0x7BFFu) {
      ++sc_bits;
      sc = f16_bits_to_float(sc_bits);
    }
    const float inv_sc = 1.0f / sc;
    const int zp = std::clamp<int>(
        static_cast<int>(std::nearbyint(-128.0f - lo * inv_sc)), -128, 127);

    dst_scale[g] = sc_bits;
    dst_offset[g] = static_cast<std::int8_t>(zp);

    for (std::uint32_t i = 0; i < kAneGroupSize; ++i) {
      const float v = std::isfinite(grp[i]) ? grp[i] : 0.0f;
      const int q = std::clamp<int>(
          static_cast<int>(std::nearbyint(v * inv_sc)) + zp, -128, 127);
      q_out[i] = static_cast<std::int8_t>(q);
    }
  }
}

// MIL Weight Blob v2 helper (matching CoreML/ANE BLOBFILE format).
std::uint64_t append_mil_blob_chunk(NSMutableData *blob, const void *bytes,
                                    std::size_t byte_count,
                                    std::uint32_t dtype_code) {
  const NSUInteger aligned =
      (blob.length + 63u) & ~static_cast<NSUInteger>(63u);
  if (aligned > blob.length) {
    [blob increaseLengthBy:(aligned - blob.length)];
  }
  const std::uint64_t chunk_offset = static_cast<std::uint64_t>(blob.length);
  std::array<std::uint8_t, 64> header{};
  header[0] = 0xEFu;
  header[1] = 0xBEu;
  header[2] = 0xADu;
  header[3] = 0xDEu;
  std::memcpy(header.data() + 4, &dtype_code, sizeof(dtype_code));
  const std::uint64_t sz64 = static_cast<std::uint64_t>(byte_count);
  const std::uint64_t data_off = chunk_offset + header.size();
  std::memcpy(header.data() + 8, &sz64, sizeof(sz64));
  std::memcpy(header.data() + 16, &data_off, sizeof(data_off));
  [blob appendBytes:header.data() length:header.size()];
  if (byte_count > 0) {
    [blob appendBytes:bytes length:byte_count];
  }
  return chunk_offset;
}

constexpr std::uint32_t kBlobDtypeFp16 = 1u;
constexpr std::uint32_t kBlobDtypeInt8 = 4u;

struct MilProcedureOffsets {
  bool fused_swiglu = false;
  bool fp16_weights = false;
  std::uint32_t in_features = 0;
  std::uint32_t out_features = 0;
  std::uint64_t w0_data_off = 0;
  std::uint64_t w0_scale_off = 0;
  std::uint64_t w0_offset_off = 0;
  std::uint64_t w1_data_off = 0;
  std::uint64_t w1_scale_off = 0;
  std::uint64_t w1_offset_off = 0;
};

// Emit multi-procedure MIL program(1.3) using either direct FP16 weights or
// 2D G=64 constexpr_blockwise_shift_scale + reshape to 4D conv weight, with
// fused silu(gate)*up for MlpGateUpSwiGlu!
NSString *build_g64_bank_mil(std::span<const MilProcedureOffsets> procs,
                             std::uint32_t chunk_tokens) {
  NSMutableString *mil = [NSMutableString
      stringWithString:
          @"program(1.3)\n"
           "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", "
           "\"3520.4.1\"}, {\"coremlc-version\", \"3520.5.1\"}})]\n{\n"];
  for (std::size_t idx = 0; idx < procs.size(); ++idx) {
    const auto &p = procs[idx];
    const std::uint32_t K = p.in_features;
    const std::uint32_t N = p.out_features;
    const std::uint32_t ng = K / kAneGroupSize;
    if (!p.fused_swiglu) {
      if (p.fp16_weights) {
        [mil appendFormat:
                 @"  func procedure%03zu<ios18>(tensor<fp16, [1, %u, 1, %u]> x) {\n"
                  "    tensor<fp16, [%u, %u, 1, 1]> w = const()[name=string(\"w\"), val=tensor<fp16, [%u, %u, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%llu)))];\n"
                  "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                  "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                  "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                  "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                  "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                  "    tensor<fp16, [1, %u, 1, %u]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string(\"conv\")];\n"
                  "  } -> (y);\n",
             idx, K, chunk_tokens,
             N, K, N, K, static_cast<unsigned long long>(p.w0_data_off),
             N, chunk_tokens];
      } else {
        [mil appendFormat:
                 @"  func procedure%03zu<ios18>(tensor<fp16, [1, %u, 1, %u]> x) {\n"
                  "    tensor<fp16, [%u, %u]> w2d = constexpr_blockwise_shift_scale("
                  "data=tensor<int8, [%u, %u]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%llu))), "
                  "scale=tensor<fp16, [%u, %u]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%llu))), "
                  "offset=tensor<int8, [%u, %u]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%llu))))"
                  "[name=string(\"dq\")];\n"
                  "    tensor<int32, [4]> wsh = const()[name=string(\"wsh\"), val=tensor<int32, [4]>([%u, %u, 1, 1])];\n"
                  "    tensor<fp16, [%u, %u, 1, 1]> w = reshape(shape=wsh, x=w2d)[name=string(\"w4d\")];\n"
                  "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                  "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                  "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                  "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                  "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                  "    tensor<fp16, [1, %u, 1, %u]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string(\"conv\")];\n"
                  "  } -> (y);\n",
             idx, K, chunk_tokens,
             N, K,
             N, K, static_cast<unsigned long long>(p.w0_data_off),
             N, ng, static_cast<unsigned long long>(p.w0_scale_off),
             N, ng, static_cast<unsigned long long>(p.w0_offset_off),
             N, K, N, K, N, chunk_tokens];
      }
    } else if (p.fp16_weights) {
      [mil appendFormat:
               @"  func procedure%03zu<ios18>(tensor<fp16, [1, %u, 1, %u]> x) {\n"
                "    tensor<fp16, [%u, %u, 1, 1]> gw = const()[name=string(\"gw\"), val=tensor<fp16, [%u, %u, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%llu)))];\n"
                "    tensor<fp16, [%u, %u, 1, 1]> uw = const()[name=string(\"uw\"), val=tensor<fp16, [%u, %u, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%llu)))];\n"
                "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                "    tensor<fp16, [1, %u, 1, %u]> gate = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=gw, x=x)[name=string(\"gate\")];\n"
                "    tensor<fp16, [1, %u, 1, %u]> up = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=uw, x=x)[name=string(\"up\")];\n"
                "    tensor<fp16, [1, %u, 1, %u]> silu_out = silu(x=gate)[name=string(\"silu\")];\n"
                "    tensor<fp16, [1, %u, 1, %u]> y = mul(x=silu_out, y=up)[name=string(\"swiglu\")];\n"
                "  } -> (y);\n",
           idx, K, chunk_tokens,
           N, K, N, K, static_cast<unsigned long long>(p.w0_data_off),
           N, K, N, K, static_cast<unsigned long long>(p.w1_data_off),
           N, chunk_tokens, N, chunk_tokens, N, chunk_tokens, N, chunk_tokens];
    } else {
      // Qwen SwiGLU convention: SiLU(gate) * up fused inside MIL procedure!
      [mil appendFormat:
               @"  func procedure%03zu<ios18>(tensor<fp16, [1, %u, 1, %u]> x) {\n"
                "    tensor<fp16, [%u, %u]> gw2 = constexpr_blockwise_shift_scale("
                "data=tensor<int8, [%u, %u]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%llu))), "
                "scale=tensor<fp16, [%u, %u]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%llu))), "
                "offset=tensor<int8, [%u, %u]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%llu))))"
                "[name=string(\"dqg\")];\n"
                "    tensor<fp16, [%u, %u]> uw2 = constexpr_blockwise_shift_scale("
                "data=tensor<int8, [%u, %u]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%llu))), "
                "scale=tensor<fp16, [%u, %u]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%llu))), "
                "offset=tensor<int8, [%u, %u]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%llu))))"
                "[name=string(\"dqu\")];\n"
                "    tensor<int32, [4]> wsh = const()[name=string(\"wsh\"), val=tensor<int32, [4]>([%u, %u, 1, 1])];\n"
                "    tensor<fp16, [%u, %u, 1, 1]> gw = reshape(shape=wsh, x=gw2)[name=string(\"gw4\")];\n"
                "    tensor<fp16, [%u, %u, 1, 1]> uw = reshape(shape=wsh, x=uw2)[name=string(\"uw4\")];\n"
                "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                "    tensor<fp16, [1, %u, 1, %u]> gate = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=gw, x=x)[name=string(\"gate\")];\n"
                "    tensor<fp16, [1, %u, 1, %u]> up = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=uw, x=x)[name=string(\"up\")];\n"
                "    tensor<fp16, [1, %u, 1, %u]> silu_out = silu(x=gate)[name=string(\"silu\")];\n"
                "    tensor<fp16, [1, %u, 1, %u]> y = mul(x=silu_out, y=up)[name=string(\"swiglu\")];\n"
                "  } -> (y);\n",
           idx, K, chunk_tokens,
           N, K,
           N, K, static_cast<unsigned long long>(p.w0_data_off),
           N, ng, static_cast<unsigned long long>(p.w0_scale_off),
           N, ng, static_cast<unsigned long long>(p.w0_offset_off),
           N, K,
           N, K, static_cast<unsigned long long>(p.w1_data_off),
           N, ng, static_cast<unsigned long long>(p.w1_scale_off),
           N, ng, static_cast<unsigned long long>(p.w1_offset_off),
           N, K, N, K, N, K,
           N, chunk_tokens, N, chunk_tokens, N, chunk_tokens, N, chunk_tokens];
    }
  }
  [mil appendString:@"}\n"];
  return mil;
}

NSDictionary *ane_instance_options(int ane_instance) {
  if (ane_instance <= 0) {
    return @{};
  }
  return @{
    @"kANEFProcedureVariantHint" : @1,
    @"kANEFAneInstanceHint" : @(ane_instance),
  };
}

IOSurfaceRef create_ane_iosurface(std::size_t byte_count,
                                  std::size_t *out_alloc_bytes) {
  const std::size_t alloc =
      std::max<std::size_t>(65536u, (byte_count + 65535u) & ~std::size_t{65535u});
  if (out_alloc_bytes) {
    *out_alloc_bytes = alloc;
  }
  return IOSurfaceCreate((__bridge CFDictionaryRef) @{
    (id)kIOSurfaceWidth : @(alloc),
    (id)kIOSurfaceHeight : @1,
    (id)kIOSurfaceBytesPerElement : @1,
    (id)kIOSurfaceBytesPerRow : @(alloc),
    (id)kIOSurfaceAllocSize : @(alloc),
    (id)kIOSurfacePixelFormat : @0,
  });
}

std::filesystem::path resolve_metallib_file(
    const std::filesystem::path &hint) {
  if (!hint.empty()) {
    if (std::filesystem::is_regular_file(hint)) {
      return hint;
    }
    const auto candidate = hint / "omlx_m5_ane_pack_unpack.metallib";
    if (std::filesystem::is_regular_file(candidate)) {
      return candidate;
    }
  }
  const std::filesystem::path local("omlx_m5_ane_pack_unpack.metallib");
  if (std::filesystem::is_regular_file(local)) {
    return local;
  }
  throw std::runtime_error(
      "unable to locate omlx_m5_ane_pack_unpack.metallib at: " +
      hint.string());
}

struct CompiledAneBank {
  id model = nil;
  NSDictionary *execution_options = nil;
  NSString *staging_directory = nil;
  std::filesystem::path lock_path;
  bool disk_cache_hit = false;
  bool native_compiled_hit = false;
  std::uint64_t weight_bytes = 0;

  ~CompiledAneBank() {
    @autoreleasepool {
      if (model != nil) {
        NSError *err = nil;
        auto unload_fn =
            reinterpret_cast<BOOL (*)(id, SEL, unsigned int, NSError **)>(
                &objc_msgSend);
        unload_fn(model, @selector(unloadWithQoS:error:), kAneQoS, &err);
        model = nil;
      }
      if (staging_directory != nil) {
        if (!lock_path.empty()) {
          try {
            ScopedFileLock lk(lock_path);
            [[NSFileManager defaultManager]
                removeItemAtPath:staging_directory
                           error:nil];
          } catch (...) {
            // Defer staging cleanup safely if lock fails during teardown.
          }
        } else {
          [[NSFileManager defaultManager]
              removeItemAtPath:staging_directory
                         error:nil];
        }
        staging_directory = nil;
      }
    }
  }
};

struct SurfaceBundle {
  IOSurfaceRef surface = nullptr;
  id ane_object = nil;
  id<MTLBuffer> mtl_buffer = nil;
  std::size_t alloc_bytes = 0;

  ~SurfaceBundle() {
    mtl_buffer = nil;
    ane_object = nil;
    if (surface != nullptr) {
      CFRelease(surface);
      surface = nullptr;
    }
  }
};

struct OpDieBinding {
  CompiledAneBank *bank = nullptr;
  std::array<id, 2> requests{nil, nil};             // ping-pong slot 0 & 1
  std::array<id<MTLBuffer>, 2> in_bufs{nil, nil};   // ping-pong input IOSurface buffers
  std::array<id<MTLBuffer>, 2> out_bufs{nil, nil};  // ping-pong output IOSurface buffers
};

struct AneJoinState {
  std::atomic<std::uint32_t> remaining{0};
  std::uint64_t seq = 0;
};

struct AneEvalJob {
  CompiledAneBank *bank = nullptr;
  id request = nil;
  std::shared_ptr<AneJoinState> join;
};

// Persistent per-die worker thread: guarantees strict FIFO ordering per physical die
// and zero per-op thread creation overhead.
class AneDieWorker final {
public:
  using CompletionFn =
      std::function<void(const std::shared_ptr<AneJoinState> &, bool,
                         const std::string &)>;

  explicit AneDieWorker(CompletionFn on_done) : on_done_(std::move(on_done)) {
    thread_ = std::thread([this] { run_loop(); });
  }

  ~AneDieWorker() { stop(); }

  void enqueue(AneEvalJob job) {
    {
      std::lock_guard<std::mutex> lk(mutex_);
      if (stop_) {
        return;
      }
      queue_.push_back(std::move(job));
    }
    cv_.notify_one();
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lk(mutex_);
      if (stop_) {
        return;
      }
      stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

private:
  void run_loop() {
    auto eval_fn = reinterpret_cast<BOOL (*)(id, SEL, unsigned int, id, id,
                                             NSError **)>(&objc_msgSend);
    for (;;) {
      AneEvalJob job;
      {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) {
          return;
        }
        job = std::move(queue_.front());
        queue_.pop_front();
      }
      bool ok = false;
      std::string err_msg;
      @autoreleasepool {
        NSError *error = nil;
        if (job.bank != nullptr && job.bank->model != nil &&
            job.request != nil) {
          ok = eval_fn(job.bank->model,
                       @selector(evaluateWithQoS:options:request:error:),
                       kAneQoS, job.bank->execution_options, job.request,
                       &error);
          if (!ok) {
            err_msg = ns_error_text(@"ANE evaluateWithQoS failed", error);
          }
        } else {
          err_msg = "ANE eval job missing bank or request";
        }
      }
      on_done_(job.join, ok, err_msg);
    }
  }

  CompletionFn on_done_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<AneEvalJob> queue_;
  bool stop_ = false;
  std::thread thread_;
};

// Compile or warm-load one ANE procedure bank with persistent flock()-guarded disk cache.
std::unique_ptr<CompiledAneBank> compile_or_load_bank(
    const std::filesystem::path &cache_dir,
    bool enable_disk_cache,
    std::string_view cache_key,
    std::string_view os_build,
    std::uint32_t die_idx,
    std::uint32_t shard_idx,
    int ane_instance,
    std::uint32_t chunk_tokens,
    bool use_fp16_weights,
    std::span<const AneOffloadOpSpec *const> shard_specs,
    std::span<const AneCompiledOpInfo> shard_infos) {
  auto bank = std::make_unique<CompiledAneBank>();
  bank->execution_options = ane_instance_options(ane_instance);

  const std::string tag =
      "bank_d" + std::to_string(die_idx) + "_s" + std::to_string(shard_idx);
  const auto meta_path = cache_dir / (tag + ".meta.json");
  const auto mil_path = cache_dir / (tag + ".mil");
  const auto weights_path = cache_dir / (tag + ".weights.bin");
  const auto lock_path = cache_dir / (tag + ".lock");

  std::unique_ptr<ScopedFileLock> disk_lock;
  if (enable_disk_cache && !cache_dir.empty()) {
    try {
      disk_lock = std::make_unique<ScopedFileLock>(lock_path);
      bank->lock_path = lock_path;
    } catch (...) {
      // Fail open: disk cache lock unavailable degrades to uncached compile.
      disk_lock.reset();
      enable_disk_cache = false;
    }
  }

  @autoreleasepool {
    NSData *mil_data = nil;
    NSData *weight_blob = nil;

    // 1. Probe persistent disk cache (skips .miv weight extraction & G=64 quantization on hit!).
    if (enable_disk_cache && std::filesystem::is_regular_file(meta_path) &&
        std::filesystem::is_regular_file(mil_path) &&
        std::filesystem::is_regular_file(weights_path)) {
      try {
        const auto meta_bytes = read_file_bytes(meta_path);
        const Json meta = Json::parse(meta_bytes.begin(), meta_bytes.end());
        if (meta.value("schema_version", 0u) == kAneCacheSchemaVersion &&
            meta.value("cache_key", "") == cache_key &&
            meta.value("os_build", "") == os_build &&
            meta.value("die_index", UINT32_MAX) == die_idx &&
            meta.value("shard_index", UINT32_MAX) == shard_idx &&
            meta.value("chunk_tokens", 0u) == chunk_tokens &&
            meta.value("procedure_count", std::size_t{0}) ==
                shard_specs.size()) {
          const auto mil_vec = read_file_bytes(mil_path);
          const auto w_vec = read_file_bytes(weights_path);
          const std::string mil_sha =
              sha256_hex_bytes(mil_vec.data(), mil_vec.size());
          if (mil_sha == meta.value("mil_sha256", "") &&
              w_vec.size() == meta.value("weight_blob_bytes", std::size_t{0})) {
            mil_data = [NSData dataWithBytes:mil_vec.data()
                                      length:mil_vec.size()];
            weight_blob = [NSData dataWithBytes:w_vec.data()
                                         length:w_vec.size()];
            bank->weight_bytes = static_cast<std::uint64_t>(w_vec.size());
            bank->disk_cache_hit = true;
          }
        }
      } catch (...) {
        // Corrupt cache entry: rebuild fresh below.
        mil_data = nil;
        weight_blob = nil;
        bank->disk_cache_hit = false;
      }
    }

    // 2. Cold build: extract .miv weights directly -> G=64 int8 MIL + weight.bin.
    if (mil_data == nil || weight_blob == nil) {
      NSMutableData *mutable_blob = [NSMutableData dataWithLength:64];
      std::vector<MilProcedureOffsets> proc_offsets;
      proc_offsets.reserve(shard_specs.size());
      std::uint32_t total_chunks = 0u;

      for (std::size_t i = 0; i < shard_specs.size(); ++i) {
        const auto &spec = *shard_specs[i];
        const auto &info = shard_infos[i];
        const std::uint64_t row_start =
            static_cast<std::uint64_t>(die_idx) * info.split.die_rows;
        const std::uint64_t row_count = info.split.die_rows;

        MilProcedureOffsets off{};
        off.fused_swiglu = (spec.role == AneProjectionRole::MlpGateUpSwiGlu);
        off.fp16_weights = use_fp16_weights;
        off.in_features = info.in_features;
        off.out_features = info.split.die_rows;

        if (use_fp16_weights) {
          auto pack_fp16 = [&](const MaterializedWeightView &w) {
            const auto f32 = dequantize_miv_weight_slice_f32(w, row_start, row_count);
            std::vector<std::uint16_t> fp16(f32.size());
            for (std::size_t k = 0; k < f32.size(); ++k) {
              fp16[k] = float_to_f16_bits(f32[k]);
            }
            return append_mil_blob_chunk(mutable_blob, fp16.data(),
                                         fp16.size() * sizeof(std::uint16_t),
                                         kBlobDtypeFp16);
          };
          off.w0_data_off = pack_fp16(spec.weight);
          total_chunks += 1u;
          if (off.fused_swiglu) {
            if (!spec.up_weight.has_value()) {
              throw std::invalid_argument(
                  "MlpGateUpSwiGlu requires both gate and up weights");
            }
            off.w1_data_off = pack_fp16(*spec.up_weight);
            total_chunks += 1u;
          }
        } else {
          const auto q0 =
              extract_miv_weight_slice_g64_int8(spec.weight, row_start, row_count);
          off.w0_data_off = append_mil_blob_chunk(
              mutable_blob, q0.data.data(), q0.data.size(), kBlobDtypeInt8);
          off.w0_scale_off = append_mil_blob_chunk(
              mutable_blob, q0.scale.data(),
              q0.scale.size() * sizeof(std::uint16_t), kBlobDtypeFp16);
          off.w0_offset_off = append_mil_blob_chunk(
              mutable_blob, q0.offset.data(), q0.offset.size(), kBlobDtypeInt8);
          total_chunks += 3u;

          if (off.fused_swiglu) {
            if (!spec.up_weight.has_value()) {
              throw std::invalid_argument(
                  "MlpGateUpSwiGlu requires both gate and up weights");
            }
            const auto q1 = extract_miv_weight_slice_g64_int8(
                *spec.up_weight, row_start, row_count);
            off.w1_data_off = append_mil_blob_chunk(
                mutable_blob, q1.data.data(), q1.data.size(), kBlobDtypeInt8);
            off.w1_scale_off = append_mil_blob_chunk(
                mutable_blob, q1.scale.data(),
                q1.scale.size() * sizeof(std::uint16_t), kBlobDtypeFp16);
            off.w1_offset_off = append_mil_blob_chunk(
                mutable_blob, q1.offset.data(), q1.offset.size(), kBlobDtypeInt8);
            total_chunks += 3u;
          }
        }
        proc_offsets.push_back(off);
      }

      auto *hdr = static_cast<std::uint32_t *>(mutable_blob.mutableBytes);
      hdr[0] = total_chunks;
      hdr[1] = 2u; // MIL weight blob format v2

      NSString *mil_str = build_g64_bank_mil(proc_offsets, chunk_tokens);
      mil_data = [mil_str dataUsingEncoding:NSUTF8StringEncoding];
      weight_blob = mutable_blob;
      bank->weight_bytes = static_cast<std::uint64_t>(weight_blob.length);

      if (enable_disk_cache) {
        try {
          write_file_atomic(mil_path, mil_data.bytes, mil_data.length);
          write_file_atomic(weights_path, weight_blob.bytes,
                            weight_blob.length);
          Json meta = {
              {"schema_version", kAneCacheSchemaVersion},
              {"cache_key", std::string(cache_key)},
              {"os_build", std::string(os_build)},
              {"die_index", die_idx},
              {"shard_index", shard_idx},
              {"chunk_tokens", chunk_tokens},
              {"procedure_count", shard_specs.size()},
              {"weight_blob_bytes", static_cast<std::size_t>(weight_blob.length)},
              {"mil_sha256", sha256_hex_bytes(mil_data.bytes, mil_data.length)},
          };
          const std::string meta_str = meta.dump(2);
          write_file_atomic(meta_path, meta_str.data(), meta_str.size());
        } catch (...) {
          // Cache persistence failure is non-fatal; continue compiling in-memory.
        }
      }
    }

    // 3. Instantiate private _ANEInMemoryModel and leverage Apple's native compiledModelExists cache.
    NSDictionary *weight_map = @{
      @"@model_path/weights/weight.bin" :
          @{@"offset" : @0, @"data" : weight_blob},
    };
    Class desc_cls = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class model_cls = NSClassFromString(@"_ANEInMemoryModel");
    if (desc_cls == Nil || model_cls == Nil) {
      throw std::runtime_error("Private ANE runtime classes are unavailable");
    }
    auto make_desc =
        reinterpret_cast<id (*)(Class, SEL, id, id, id)>(&objc_msgSend);
    id descriptor =
        make_desc(desc_cls, @selector(modelWithMILText:weights:optionsPlist:),
                  mil_data, weight_map, nil);
    auto make_model = reinterpret_cast<id (*)(Class, SEL, id)>(&objc_msgSend);
    id model =
        make_model(model_cls, @selector(inMemoryModelWithDescriptor:),
                   descriptor);
    if (model == nil) {
      throw std::runtime_error("Failed to create _ANEInMemoryModel");
    }

    auto hex_fn = reinterpret_cast<id (*)(id, SEL)>(&objc_msgSend);
    NSString *identifier = hex_fn(model, @selector(hexStringIdentifier));
    NSString *staging_dir =
        [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];

    bool restored = false;
    bool staged = false;
    if (enable_disk_cache &&
        [model respondsToSelector:@selector(compiledModelExists)]) {
      auto exists_fn = reinterpret_cast<BOOL (*)(id, SEL)>(&objc_msgSend);
      restored = exists_fn(model, @selector(compiledModelExists));
    }

    auto compile_fresh = [&]() {
      [[NSFileManager defaultManager] removeItemAtPath:staging_dir error:nil];
      NSString *weight_dir =
          [staging_dir stringByAppendingPathComponent:@"weights"];
      [[NSFileManager defaultManager] createDirectoryAtPath:weight_dir
                                withIntermediateDirectories:YES
                                                 attributes:nil
                                                      error:nil];
      [mil_data
          writeToFile:[staging_dir stringByAppendingPathComponent:@"model.mil"]
           atomically:YES];
      [weight_blob
          writeToFile:[weight_dir stringByAppendingPathComponent:@"weight.bin"]
           atomically:YES];

      NSError *err = nil;
      auto compile_fn =
          reinterpret_cast<BOOL (*)(id, SEL, unsigned int, id, NSError **)>(
              &objc_msgSend);
      BOOL ok = compile_fn(model, @selector(compileWithQoS:options:error:),
                           kAneQoS, bank->execution_options, &err);
      if (!ok) {
        [[NSFileManager defaultManager] removeItemAtPath:staging_dir error:nil];
        throw std::runtime_error(
            ns_error_text(@"ANE bank compilation failed", err));
      }
      staged = true;
    };

    if (!restored) {
      compile_fresh();
    }

    auto load_fn =
        reinterpret_cast<BOOL (*)(id, SEL, unsigned int, id, NSError **)>(
            &objc_msgSend);
    NSError *load_err = nil;
    BOOL loaded = load_fn(model, @selector(loadWithQoS:options:error:), kAneQoS,
                          bank->execution_options, &load_err);
    if (!loaded && restored) {
      if ([model respondsToSelector:@selector(purgeCompiledModel)]) {
        auto purge_fn = reinterpret_cast<void (*)(id, SEL)>(&objc_msgSend);
        purge_fn(model, @selector(purgeCompiledModel));
      }
      restored = false;
      compile_fresh();
      load_err = nil;
      loaded = load_fn(model, @selector(loadWithQoS:options:error:), kAneQoS,
                       bank->execution_options, &load_err);
    }
    if (!loaded) {
      [[NSFileManager defaultManager] removeItemAtPath:staging_dir error:nil];
      throw std::runtime_error(ns_error_text(@"ANE bank load failed", load_err));
    }

    bank->model = model;
    bank->staging_directory = staged ? staging_dir : nil;
    bank->native_compiled_hit = restored;
    return bank;
  }
}

} // namespace

// -----------------------------------------------------------------------------
// Public helpers: Roles, Hardware Probe, Alignment, and Zero-Reconversion Extract
// -----------------------------------------------------------------------------

std::string_view to_string(AneProjectionRole role) noexcept {
  switch (role) {
  case AneProjectionRole::MlpGateUpSwiGlu:
    return "mlp_gate_up_swiglu";
  case AneProjectionRole::MlpDown:
    return "mlp_down";
  case AneProjectionRole::GdnInProjZ:
    return "gdn_z";
  case AneProjectionRole::GdnOutProj:
    return "gdn_out";
  case AneProjectionRole::AttnOProj:
    return "attention_o";
  }
  return "unknown";
}

std::string_view to_string(AneLinearRoute route) noexcept {
  switch (route) {
  case AneLinearRoute::BF16:
    return "bf16";
  case AneLinearRoute::W4A8:
    return "w4a8";
  }
  return "bf16";
}

bool ane_stage_role_eligible(std::string_view role_or_name) noexcept {
  // Explicit hard denylist for GDN recurrent projections, Attn QKV, MoE, & embeddings.
  constexpr std::array<std::string_view, 16> kDeniedSubstrings = {{
      "in_proj_qkv",
      "gdn_qkv",
      "in_proj_a",
      "gdn_a",
      "in_proj_b",
      "gdn_b",
      "q_proj",
      "k_proj",
      "v_proj",
      "attention_q",
      "attention_k",
      "attention_v",
      "lm_head",
      "embed_tokens",
      "switch_mlp",
      "moe_",
  }};
  for (const auto denied : kDeniedSubstrings) {
    if (role_or_name.find(denied) != std::string_view::npos) {
      return false;
    }
  }

  std::string_view normalized = role_or_name;
  if (normalized.ends_with(".weight")) {
    normalized.remove_suffix(std::string_view(".weight").size());
  }

  // Exact catalog stage role allowlist.
  if (normalized == "mlp_gate" || normalized == "mlp_up" ||
      normalized == "mlp_gate_up_swiglu" || normalized == "mlp_down" ||
      normalized == "gdn_z" || normalized == "gdn_out" ||
      normalized == "attention_o") {
    return true;
  }

  // Tensor-name suffix allowlist (non-recurrent projections only).
  if (normalized.ends_with(".mlp.gate_proj") ||
      normalized.ends_with(".mlp.up_proj") ||
      normalized.ends_with(".mlp.down_proj") ||
      normalized.ends_with(".linear_attn.in_proj_z") ||
      normalized.ends_with(".linear_attn.out_proj") ||
      normalized.ends_with(".self_attn.o_proj")) {
    return true;
  }
  return false;
}

AneHardwareInfo probe_ane_hardware() noexcept {
  AneHardwareInfo info;
  @autoreleasepool {
    info.os_build = sanitize_os_build();
    info.framework_loaded = ensure_ane_framework_loaded();
    if (!info.framework_loaded) {
      info.unavailable_reason = "AppleNeuralEngine.framework failed to dlopen";
      return info;
    }
    Class desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class mdl = NSClassFromString(@"_ANEInMemoryModel");
    Class req = NSClassFromString(@"_ANERequest");
    Class srf = NSClassFromString(@"_ANEIOSurfaceObject");
    info.classes_available =
        (desc != Nil && mdl != Nil && req != Nil && srf != Nil);
    if (!info.classes_available) {
      info.unavailable_reason =
          "Required private AppleNeuralEngine ObjC classes are absent";
      return info;
    }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    info.metal_iosurface_supported =
        (dev != nil &&
         [dev respondsToSelector:@selector(newBufferWithIOSurface:)]);
    if (!info.metal_iosurface_supported) {
      info.unavailable_reason =
          "MTLDevice newBufferWithIOSurface: is unavailable";
      return info;
    }

    info.physical_die_count = detect_ane_die_count();
    info.hardware_available = (info.physical_die_count > 0u);
    if (!info.hardware_available) {
      info.unavailable_reason = "No Apple Neural Engine hardware detected";
    }
  }
  return info;
}

bool is_ane_available() noexcept { return probe_ane_hardware().is_available(); }

AneChannelSplit compute_ane_channel_split(std::uint32_t out_features,
                                          double target_alpha,
                                          std::uint32_t active_dies,
                                          double max_alpha,
                                          std::uint32_t min_gpu_rows) noexcept {
  AneChannelSplit split{};
  split.total_out_features = out_features;
  split.active_dies = active_dies;
  split.gpu_rows = out_features;
  if (active_dies == 0u || target_alpha <= 0.0 ||
      out_features < kAneRowAlign) {
    return split;
  }

  const double clamped_alpha =
      std::clamp(target_alpha, 0.0, std::clamp(max_alpha, 0.0, 0.95));
  const double raw_per_die =
      (static_cast<double>(out_features) * clamped_alpha) /
      static_cast<double>(active_dies);
  auto die_rows = static_cast<std::uint32_t>(
      std::llround(raw_per_die / static_cast<double>(kAneRowAlign))) *
      kAneRowAlign;
  if (die_rows == 0u && raw_per_die >= static_cast<double>(kAneRowAlign) * 0.5) {
    die_rows = kAneRowAlign;
  }

  const std::uint64_t required_gpu =
      std::min<std::uint64_t>(min_gpu_rows, out_features);
  while (die_rows > 0u &&
         static_cast<std::uint64_t>(die_rows) * active_dies + required_gpu >
             static_cast<std::uint64_t>(out_features)) {
    die_rows -= kAneRowAlign;
  }

  split.die_rows = die_rows;
  split.ane_total_rows = die_rows * active_dies;
  split.gpu_row_offset = split.ane_total_rows; // Guaranteed % 128 == 0 (% 64 == 0)
  split.gpu_rows = out_features - split.ane_total_rows;
  split.effective_alpha =
      out_features > 0u
          ? static_cast<double>(split.ane_total_rows) /
                static_cast<double>(out_features)
          : 0.0;
  return split;
}

bool ane_weight_view_supported(const MaterializedWeightView &w) noexcept {
  if (w.parent_shape.size() != 2 || w.row_count == 0 || w.in_features == 0 ||
      (w.in_features % kAneGroupSize) != 0) {
    return false;
  }
  if (w.format == "mxfp4_e2m1_g32_e8m0_v1") {
    return (w.layout == "mx_rowsplit_k128_v1" ||
            w.layout == "mx_nax_tiled_bk128_v1") &&
           w.padded_k >= w.in_features && (w.padded_k % 128u) == 0;
  }
  if (const auto af = parse_affine_format(w.format); af.has_value()) {
    return (w.layout == "oq_rowsplit_k128_v1" ||
            w.layout == "oq_nax_tiled_g64_v1" ||
            w.layout == "oq_nax_tiled_g128_v1") &&
           w.padded_k >= w.in_features && (w.padded_k % 128u) == 0;
  }
  if (w.format == "direct_bf16_v1" || w.format == "direct_f16_v1" ||
      w.format == "direct_f32_v1") {
    return w.layout == "contiguous_le_v1";
  }
  return false;
}

std::vector<float> dequantize_miv_weight_slice_f32(
    const MaterializedWeightView &w, std::uint64_t row_start,
    std::uint64_t row_count) {
  if (!ane_weight_view_supported(w)) {
    throw std::invalid_argument("unsupported .miv weight view for ANE: " +
                                w.name + " (" + w.format + "/" + w.layout + ")");
  }
  if (row_count == 0 || row_start > w.row_count ||
      row_count > w.row_count - row_start) {
    throw std::out_of_range("ANE weight extraction row range out of bounds: " +
                            w.name);
  }
  const std::uint64_t K = w.in_features;
  const std::uint64_t padded_k = w.padded_k;
  const std::uint64_t total_parent_rows = w.parent_shape[0];
  std::vector<float> out(static_cast<std::size_t>(row_count * K), 0.0f);

  // 1. Direct dense weights (BF16 / FP16 / FP32)
  if (w.format == "direct_bf16_v1" || w.format == "direct_f16_v1" ||
      w.format == "direct_f32_v1") {
    const auto data = plane_span(require_view_plane(w, "data"));
    for (std::uint64_t lr = 0; lr < row_count; ++lr) {
      const std::uint64_t r = row_start + lr;
      float *dst = out.data() + static_cast<std::size_t>(lr * K);
      if (w.format == "direct_bf16_v1") {
        const auto *src = data.data() + (r * K) * 2;
        for (std::uint64_t c = 0; c < K; ++c) {
          dst[c] = bf16_to_float(load_u16_le(src + c * 2));
        }
      } else if (w.format == "direct_f16_v1") {
        const auto *src = data.data() + (r * K) * 2;
        for (std::uint64_t c = 0; c < K; ++c) {
          dst[c] = f16_bits_to_float(load_u16_le(src + c * 2));
        }
      } else {
        const auto *src = data.data() + (r * K) * 4;
        for (std::uint64_t c = 0; c < K; ++c) {
          dst[c] = load_f32_le(src + c * 4);
        }
      }
    }
    return out;
  }

  // 2. MXFP4 (mxfp4_e2m1_g32_e8m0_v1) — both untiled and NAX pre-tiled
  if (w.format == "mxfp4_e2m1_g32_e8m0_v1") {
    if (w.layout == "mx_nax_tiled_bk128_v1") {
      const std::uint64_t bn = total_parent_rows >= 65536u ? 32u : 16u;
      const std::uint64_t k_blocks = K / 128u;
      const auto data = plane_span(require_view_plane(w, "nax_data"));
      const auto scale = plane_span(require_view_plane(w, "nax_scale"));
      for (std::uint64_t lr = 0; lr < row_count; ++lr) {
        const std::uint64_t r = w.row_start + row_start + lr;
        const std::uint64_t rn = r % bn;
        const std::uint64_t row_tile_base = (r / bn) * k_blocks;
        float *dst = out.data() + static_cast<std::size_t>(lr * K);
        for (std::uint64_t blk = 0; blk < k_blocks; ++blk) {
          const std::uint64_t tile_idx = row_tile_base + blk;
          const auto *d_ptr = reinterpret_cast<const std::uint8_t *>(
              data.data() + tile_idx * (bn * 64u) + rn * 64u);
          const auto *s_ptr = reinterpret_cast<const std::uint8_t *>(
              scale.data() + tile_idx * (bn * 4u) + rn * 4u);
          for (std::uint64_t sg = 0; sg < 4u; ++sg) {
            const float sc = decode_e8m0(s_ptr[sg]);
            if (!std::isfinite(sc)) {
              throw std::runtime_error("MXFP4 NAX tiled decode produced NaN scale");
            }
            const std::uint64_t col_base = blk * 128u + sg * 32u;
            const auto *dp = d_ptr + sg * 16u;
            for (std::uint64_t b = 0; b < 16u; ++b) {
              const std::uint8_t packed = dp[b];
              dst[col_base + b * 2u] = decode_e2m1(packed & 0x0Fu) * sc;
              dst[col_base + b * 2u + 1u] = decode_e2m1(packed >> 4u) * sc;
            }
          }
        }
      }
      return out;
    }

    // Untiled mx_rowsplit_k128_v1
    const auto data = plane_span(require_view_plane(w, "data"));
    const auto scale = plane_span(require_view_plane(w, "scale"));
    const std::uint64_t data_row = padded_k / 2u;
    const std::uint64_t scale_row = padded_k / 32u;
    const std::uint64_t g32_count = K / 32u;
    for (std::uint64_t lr = 0; lr < row_count; ++lr) {
      const std::uint64_t r = row_start + lr;
      const auto *d_row =
          reinterpret_cast<const std::uint8_t *>(data.data() + r * data_row);
      const auto *s_row =
          reinterpret_cast<const std::uint8_t *>(scale.data() + r * scale_row);
      float *dst = out.data() + static_cast<std::size_t>(lr * K);
      for (std::uint64_t g = 0; g < g32_count; ++g) {
        const float sc = decode_e8m0(s_row[g]);
        if (!std::isfinite(sc)) {
          throw std::runtime_error("MXFP4 untiled decode produced NaN scale");
        }
        const auto *dp = d_row + g * 16u;
        float *dp_out = dst + g * 32u;
        for (std::uint64_t b = 0; b < 16u; ++b) {
          const std::uint8_t packed = dp[b];
          dp_out[b * 2u] = decode_e2m1(packed & 0x0Fu) * sc;
          dp_out[b * 2u + 1u] = decode_e2m1(packed >> 4u) * sc;
        }
      }
    }
    return out;
  }

  // 3. oQ Affine (q3/q4/q5/q6/q8, g64/g128) — both untiled and NAX pre-tiled
  const auto af = parse_affine_format(w.format);
  if (!af.has_value()) {
    throw std::invalid_argument("unexpected affine format: " + w.format);
  }
  const std::uint32_t bits = af->bits;
  const std::uint32_t gs = af->group_size;

  if (w.layout == "oq_nax_tiled_g64_v1" || w.layout == "oq_nax_tiled_g128_v1") {
    const std::uint64_t bn = total_parent_rows >= 65536u ? 32u : 16u;
    const std::uint64_t padded_n = ((total_parent_rows + bn - 1u) / bn) * bn;
    const std::uint64_t groups64 = K / 64u;
    const auto codes = plane_span(require_view_plane(w, "nax_codes"));
    const auto sb = plane_span(require_view_plane(w, "nax_sb"));
    std::span<const std::byte> high{};
    if (bits == 3u) {
      high = plane_span(require_view_plane(w, "nax_high1"));
    } else if (bits == 5u || bits == 6u) {
      high = plane_span(require_view_plane(w, "nax_high2"));
    }

    for (std::uint64_t lr = 0; lr < row_count; ++lr) {
      const std::uint64_t r = w.row_start + row_start + lr;
      const std::uint64_t rn = r % bn;
      const std::uint64_t row_tile_base = (r / bn) * groups64;
      float *dst = out.data() + static_cast<std::size_t>(lr * K);
      for (std::uint64_t g = 0; g < groups64; ++g) {
        const std::uint64_t tile_idx = row_tile_base + g;
        const std::uint32_t packed_sb =
            load_u32_le(sb.data() + (g * padded_n + r) * 4u);
        const float sc =
            bf16_to_float(static_cast<std::uint16_t>(packed_sb & 0xFFFFu));
        const float bi =
            bf16_to_float(static_cast<std::uint16_t>(packed_sb >> 16u));
        float *g_out = dst + g * 64u;

        if (bits == 3u) {
          const auto *cp = reinterpret_cast<const std::uint8_t *>(
              codes.data() + tile_idx * (bn * 16u) + rn * 16u);
          const auto *hp = reinterpret_cast<const std::uint8_t *>(
              high.data() + tile_idx * (bn * 8u) + rn * 8u);
          for (std::uint64_t c = 0; c < 64u; ++c) {
            const std::uint32_t code =
                ((cp[c / 4u] >> ((c & 3u) * 2u)) & 0x03u) |
                (((hp[c / 8u] >> (c & 7u)) & 1u) << 2u);
            g_out[c] = std::fma(sc, static_cast<float>(code), bi);
          }
        } else if (bits == 8u) {
          const auto *cp = reinterpret_cast<const std::uint8_t *>(
              codes.data() + tile_idx * (bn * 64u) + rn * 64u);
          for (std::uint64_t c = 0; c < 64u; ++c) {
            g_out[c] = std::fma(sc, static_cast<float>(cp[c]), bi);
          }
        } else {
          const auto *cp = reinterpret_cast<const std::uint8_t *>(
              codes.data() + tile_idx * (bn * 32u) + rn * 32u);
          const auto *hp =
              (bits == 5u || bits == 6u)
                  ? reinterpret_cast<const std::uint8_t *>(
                        high.data() + tile_idx * (bn * 16u) + rn * 16u)
                  : nullptr;
          for (std::uint64_t c = 0; c < 64u; ++c) {
            std::uint32_t code = (cp[c / 2u] >> ((c & 1u) * 4u)) & 0x0Fu;
            if (hp != nullptr) {
              const std::uint32_t h2 = (hp[c / 4u] >> ((c & 3u) * 2u)) & 0x03u;
              code |= (bits == 5u ? (h2 & 1u) : h2) << 4u;
            }
            g_out[c] = std::fma(sc, static_cast<float>(code), bi);
          }
        }
      }
    }
    return out;
  }

  // Untiled oq_rowsplit_k128_v1
  const char *code_kind =
      bits == 8u ? "code" : (bits == 3u ? "low_2bit" : "base_code");
  const auto base = plane_span(require_view_plane(w, code_kind));
  const auto scale = plane_span(require_view_plane(w, "scale"));
  const auto bias = plane_span(require_view_plane(w, "bias"));
  std::span<const std::byte> high{};
  if (bits == 3u || bits == 5u) {
    high = plane_span(require_view_plane(w, "high_bit"));
  } else if (bits == 6u) {
    high = plane_span(require_view_plane(w, "high_2bit"));
  }
  const std::uint64_t base_row =
      bits == 8u ? padded_k : (bits == 3u ? padded_k / 4u : padded_k / 2u);
  const std::uint64_t high_row =
      (bits == 3u || bits == 5u) ? padded_k / 8u : padded_k / 4u;
  const std::uint64_t group_row = padded_k / gs;

  for (std::uint64_t lr = 0; lr < row_count; ++lr) {
    const std::uint64_t r = row_start + lr;
    const auto *b_row =
        reinterpret_cast<const std::uint8_t *>(base.data() + r * base_row);
    const auto *h_row =
        high.empty()
            ? nullptr
            : reinterpret_cast<const std::uint8_t *>(high.data() + r * high_row);
    const auto *s_row = scale.data() + r * group_row * 2u;
    const auto *bi_row = bias.data() + r * group_row * 2u;
    float *dst = out.data() + static_cast<std::size_t>(lr * K);

    const std::uint64_t num_groups = K / gs;
    for (std::uint64_t g = 0; g < num_groups; ++g) {
      const float sc = bf16_to_float(load_u16_le(s_row + g * 2u));
      const float bi = bf16_to_float(load_u16_le(bi_row + g * 2u));
      const std::uint64_t col_start = g * gs;
      for (std::uint64_t i = 0; i < gs; ++i) {
        const std::uint64_t c = col_start + i;
        std::uint32_t code = 0u;
        if (bits == 8u) {
          code = b_row[c];
        } else if (bits == 3u) {
          code = ((b_row[c / 4u] >> ((c & 3u) * 2u)) & 0x03u) |
                 (((h_row[c / 8u] >> (c & 7u)) & 1u) << 2u);
        } else {
          code = (b_row[c / 2u] >> ((c & 1u) * 4u)) & 0x0Fu;
          if (bits == 5u) {
            code |= ((h_row[c / 8u] >> (c & 7u)) & 1u) << 4u;
          } else if (bits == 6u) {
            code |= ((h_row[c / 4u] >> ((c & 3u) * 2u)) & 3u) << 4u;
          }
        }
        dst[c] = std::fma(sc, static_cast<float>(code), bi);
      }
    }
  }
  return out;
}

AneGroup64Int8Weight quantize_f32_to_g64_int8(std::span<const float> weights,
                                              std::uint32_t rows,
                                              std::uint32_t in_features) {
  if (rows == 0 || in_features == 0 || (in_features % kAneGroupSize) != 0) {
    throw std::invalid_argument(
        "ANE G=64 quantization requires positive rows and in_features % 64 == 0");
  }
  const std::size_t expected =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(in_features);
  if (weights.size() != expected) {
    throw std::invalid_argument("ANE G=64 quantization input size mismatch");
  }

  AneGroup64Int8Weight q;
  q.rows = rows;
  q.in_features = in_features;
  q.group_size = kAneGroupSize;
  const std::uint32_t ng = in_features / kAneGroupSize;
  q.data.resize(expected);
  q.scale.resize(static_cast<std::size_t>(rows) * ng);
  q.offset.resize(static_cast<std::size_t>(rows) * ng);

  const std::size_t hw = std::max<std::size_t>(1u, std::thread::hardware_concurrency());
  const std::size_t num_threads =
      rows >= 256u ? std::min<std::size_t>({hw, 8u, rows / 64u}) : 1u;

  auto work = [&](std::uint32_t r0, std::uint32_t r1) {
    for (std::uint32_t r = r0; r < r1; ++r) {
      quantize_row_g64_int8(
          weights.data() + static_cast<std::size_t>(r) * in_features,
          in_features,
          q.data.data() + static_cast<std::size_t>(r) * in_features,
          q.scale.data() + static_cast<std::size_t>(r) * ng,
          q.offset.data() + static_cast<std::size_t>(r) * ng);
    }
  };

  if (num_threads <= 1u) {
    work(0u, rows);
  } else {
    std::vector<std::thread> threads;
    threads.reserve(num_threads - 1u);
    for (std::size_t t = 1; t < num_threads; ++t) {
      const auto r0 = static_cast<std::uint32_t>((rows * t) / num_threads);
      const auto r1 = static_cast<std::uint32_t>((rows * (t + 1u)) / num_threads);
      threads.emplace_back(work, r0, r1);
    }
    work(0u, static_cast<std::uint32_t>(rows / num_threads));
    for (auto &th : threads) {
      th.join();
    }
  }
  return q;
}

AneGroup64Int8Weight extract_miv_weight_slice_g64_int8(
    const MaterializedWeightView &weight, std::uint64_t row_start,
    std::uint64_t row_count) {
  const auto f32 = dequantize_miv_weight_slice_f32(weight, row_start, row_count);
  return quantize_f32_to_g64_int8(
      f32, static_cast<std::uint32_t>(row_count),
      static_cast<std::uint32_t>(weight.in_features));
}

std::vector<float> dequantize_g64_int8_to_f32(
    const AneGroup64Int8Weight &q) {
  const std::uint32_t ng = q.in_features / kAneGroupSize;
  std::vector<float> out(static_cast<std::size_t>(q.rows) * q.in_features);
  for (std::uint32_t r = 0; r < q.rows; ++r) {
    for (std::uint32_t g = 0; g < ng; ++g) {
      const float sc = f16_bits_to_float(q.scale[static_cast<std::size_t>(r) * ng + g]);
      const float zp = static_cast<float>(q.offset[static_cast<std::size_t>(r) * ng + g]);
      const std::size_t base =
          static_cast<std::size_t>(r) * q.in_features +
          static_cast<std::size_t>(g) * kAneGroupSize;
      for (std::uint32_t i = 0; i < kAneGroupSize; ++i) {
        out[base + i] = (static_cast<float>(q.data[base + i]) - zp) * sc;
      }
    }
  }
  return out;
}

// -----------------------------------------------------------------------------
// AneSharedBuffer
// -----------------------------------------------------------------------------

class AneSharedBuffer::Impl {
public:
  id<MTLBuffer> buffer = nil;
  std::uint64_t length = 0;
};

AneSharedBuffer::AneSharedBuffer() = default;
AneSharedBuffer::AneSharedBuffer(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
AneSharedBuffer::AneSharedBuffer(AneSharedBuffer &&) noexcept = default;
AneSharedBuffer &AneSharedBuffer::operator=(AneSharedBuffer &&) noexcept = default;
AneSharedBuffer::~AneSharedBuffer() = default;

bool AneSharedBuffer::valid() const noexcept {
  return impl_ != nullptr && impl_->buffer != nil;
}
std::uint64_t AneSharedBuffer::size() const noexcept {
  return impl_ != nullptr ? impl_->length : 0u;
}
void *AneSharedBuffer::data() noexcept {
  return valid() ? [impl_->buffer contents] : nullptr;
}
const void *AneSharedBuffer::data() const noexcept {
  return valid() ? [impl_->buffer contents] : nullptr;
}
void *AneSharedBuffer::native_mtl_buffer() const noexcept {
  return valid() ? (__bridge void *)impl_->buffer : nullptr;
}

// -----------------------------------------------------------------------------
// AneDualOffloadEngine::Impl
// -----------------------------------------------------------------------------

class AneDualOffloadEngine::Impl {
public:
  Impl() = default;

  ~Impl() {
    // 1. Drain any pending GPU/ANE operations cleanly before shutting down workers.
    try {
      flush_and_wait(5000u);
    } catch (...) {
      // Never throw from destructor.
    }
    // 2. Ensure serial listener queue has finished all scheduled callbacks.
    if (listener_queue_ != nil) {
      dispatch_sync(listener_queue_, ^{});
    }
    // 3. Stop persistent ANE worker threads.
    for (auto &w : workers_) {
      if (w) {
        w->stop();
      }
    }
    // 4. Release pre-built requests before freeing surfaces & banks.
    op_bindings_.clear();
    // 5. Release IOSurface pools & unload ANE banks cleanly.
    for (auto &die_map : surface_pools_) {
      die_map.clear();
    }
    for (auto &die_banks : banks_) {
      die_banks.clear();
    }
  }

  void latch_error(const std::string &msg) {
    std::lock_guard<std::mutex> lk(error_mutex_);
    if (latched_error_.empty()) {
      latched_error_ = msg;
    }
  }

  std::string current_error() const {
    std::lock_guard<std::mutex> lk(error_mutex_);
    return latched_error_;
  }

  bool has_error() const {
    std::lock_guard<std::mutex> lk(error_mutex_);
    return !latched_error_.empty();
  }

  void signal_ane_done(std::uint64_t seq) {
    {
      std::lock_guard<std::mutex> lk(done_mutex_);
      if (seq > signaled_done_) {
        signaled_done_ = seq;
        [ane_done_event_ setSignaledValue:seq];
      }
    }
    done_cv_.notify_all();
  }

  void on_die_job_done(const std::shared_ptr<AneJoinState> &join, bool ok,
                       const std::string &err_msg) {
    if (!ok) {
      latch_error(err_msg.empty() ? "ANE evaluation failed" : err_msg);
    }
    if (join->remaining.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
      completed_ops_.fetch_add(1u, std::memory_order_relaxed);
      signal_ane_done(join->seq);
    }
  }

  const SurfaceBundle &get_or_create_surface(std::uint32_t die_idx,
                                             std::uint32_t slot,
                                             bool is_output,
                                             std::size_t byte_count) {
    const std::uint64_t key =
        (static_cast<std::uint64_t>(slot) << 63u) |
        (static_cast<std::uint64_t>(is_output ? 1u : 0u) << 62u) |
        static_cast<std::uint64_t>(byte_count);
    auto &pool = surface_pools_[die_idx];
    if (auto it = pool.find(key); it != pool.end()) {
      return *it->second;
    }

    auto bundle = std::make_unique<SurfaceBundle>();
    bundle->surface = create_ane_iosurface(byte_count, &bundle->alloc_bytes);
    if (bundle->surface == nullptr) {
      throw std::runtime_error("IOSurfaceCreate failed for ANE ping-pong pool");
    }

    Class srf_cls = NSClassFromString(@"_ANEIOSurfaceObject");
    auto make_obj =
        reinterpret_cast<id (*)(Class, SEL, IOSurfaceRef)>(&objc_msgSend);
    bundle->ane_object =
        make_obj(srf_cls, @selector(objectWithIOSurface:), bundle->surface);
    bundle->mtl_buffer =
        [(id<OmlxM5AneMetalDeviceIOSurface>)device_
            newBufferWithIOSurface:bundle->surface];
    if (bundle->ane_object == nil || bundle->mtl_buffer == nil) {
      throw std::runtime_error(
          "Failed to wrap ANE IOSurface with _ANEIOSurfaceObject / MTLBuffer");
    }

    stats_.iosurface_pool_bytes += bundle->alloc_bytes;
    auto *raw = bundle.get();
    pool.emplace(key, std::move(bundle));
    return *raw;
  }

  void init_metal_pipelines(const AneDualOffloadConfig &config) {
    @autoreleasepool {
      if (config.external_mtl_device != nullptr) {
        device_ = (__bridge id<MTLDevice>)config.external_mtl_device;
      } else {
        device_ = MTLCreateSystemDefaultDevice();
      }
      if (device_ == nil) {
        throw std::runtime_error("MTLDevice unavailable for Dual-ANE offload");
      }
      if (config.external_mtl_command_queue != nullptr) {
        queue_ =
            (__bridge id<MTLCommandQueue>)config.external_mtl_command_queue;
      } else {
        queue_ = [device_ newCommandQueue];
      }
      if (queue_ == nil) {
        throw std::runtime_error(
            "MTLCommandQueue unavailable for Dual-ANE offload");
      }

      const auto lib_path = resolve_metallib_file(config.metallib_path);
      NSURL *url =
          [NSURL fileURLWithPath:[NSString stringWithUTF8String:lib_path.c_str()]];
      NSError *err = nil;
      id<MTLLibrary> lib = [device_ newLibraryWithURL:url error:&err];
      if (lib == nil) {
        throw std::runtime_error(
            ns_error_text(@"Failed to load omlx_m5_ane_pack_unpack.metallib", err));
      }

      auto make_pso = [&](NSString *fn_name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [lib newFunctionWithName:fn_name];
        if (fn == nil) {
          throw std::runtime_error(
              "Missing Metal function in ane_pack_unpack.metallib: " +
              std::string([fn_name UTF8String]));
        }
        NSError *pso_err = nil;
        id<MTLComputePipelineState> pso =
            [device_ newComputePipelineStateWithFunction:fn error:&pso_err];
        if (pso == nil) {
          throw std::runtime_error(
              ns_error_text(@"Failed to build ANE pack/unpack pipeline", pso_err));
        }
        return pso;
      };

      pack_single_pso_ =
          make_pso(@"omlx_m5_ane_pack_bf16_to_fp16_planar_single");
      pack_dual_pso_ = make_pso(@"omlx_m5_ane_pack_bf16_to_fp16_planar_dual");
      unpack_single_pso_ =
          make_pso(@"omlx_m5_ane_unpack_fp16_planar_to_bf16_single");
      unpack_dual_pso_ =
          make_pso(@"omlx_m5_ane_unpack_fp16_planar_to_bf16_dual");

      ane_ready_event_ = [device_ newSharedEvent];
      ane_done_event_ = [device_ newSharedEvent];
      unpack_done_event_ = [device_ newSharedEvent];
      if (ane_ready_event_ == nil || ane_done_event_ == nil ||
          unpack_done_event_ == nil) {
        throw std::runtime_error(
            "Failed to create MTLSharedEvent objects for Dual-ANE pipeline");
      }

      dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
          DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
      listener_queue_ =
          dispatch_queue_create("io.ninfer.miv.ane.shared_event_listener", attr);
      event_listener_ =
          [[MTLSharedEventListener alloc] initWithDispatchQueue:listener_queue_];
      if (event_listener_ == nil) {
        throw std::runtime_error(
            "Failed to create MTLSharedEventListener for Dual-ANE pipeline");
      }
    }
  }

  void bind_op_requests() {
    @autoreleasepool {
      Class req_cls = NSClassFromString(@"_ANERequest");
      auto get_model_fn = reinterpret_cast<id (*)(id, SEL)>(&objc_msgSend);
      auto idx_fn =
          reinterpret_cast<id (*)(id, SEL, unsigned int)>(&objc_msgSend);
      auto make_req =
          reinterpret_cast<id (*)(Class, SEL, id, id, id, id, id, id, id)>(
              &objc_msgSend);

      op_bindings_.resize(ops_.size());
      for (std::size_t op_idx = 0; op_idx < ops_.size(); ++op_idx) {
        const auto &info = ops_[op_idx];
        const std::size_t in_bytes =
            static_cast<std::size_t>(info.in_features) * stats_.chunk_tokens * 2u;
        const std::size_t out_bytes =
            static_cast<std::size_t>(info.split.die_rows) * stats_.chunk_tokens * 2u;

        for (std::uint32_t die = 0; die < stats_.active_die_count; ++die) {
          CompiledAneBank *bank = banks_[die].at(info.shard_index).get();
          auto &binding = op_bindings_[op_idx][die];
          binding.bank = bank;

          id ane_model = get_model_fn(bank->model, @selector(model));
          NSIndexSet *in_set =
              idx_fn(ane_model, @selector(inputSymbolIndicesForProcedureIndex:),
                     info.procedure_index);
          NSIndexSet *out_set = idx_fn(
              ane_model, @selector(outputSymbolIndicesForProcedureIndex:),
              info.procedure_index);
          if (in_set == nil || out_set == nil || in_set.count != 1 ||
              out_set.count != 1) {
            throw std::runtime_error(
                "ANE procedure symbol indices mismatch for op " + info.name);
          }
          NSArray *in_indices = @[ @(in_set.firstIndex) ];
          NSArray *out_indices = @[ @(out_set.firstIndex) ];

          for (std::uint32_t slot = 0; slot < 2u; ++slot) {
            const auto &in_surf =
                get_or_create_surface(die, slot, false, in_bytes);
            const auto &out_surf =
                get_or_create_surface(die, slot, true, out_bytes);
            id req = make_req(
                req_cls,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                          weightsBuffer:perfStats:procedureIndex:),
                @[ in_surf.ane_object ], in_indices, @[ out_surf.ane_object ],
                out_indices, nil, nil, @(info.procedure_index));
            if (req == nil) {
              throw std::runtime_error(
                  "Failed to pre-build _ANERequest for op " + info.name);
            }
            binding.requests[slot] = req;
            binding.in_bufs[slot] = in_surf.mtl_buffer;
            binding.out_bufs[slot] = out_surf.mtl_buffer;
          }
        }
      }
    }
  }

  void start_workers() {
    for (std::uint32_t die = 0; die < stats_.active_die_count; ++die) {
      workers_[die] = std::make_unique<AneDieWorker>(
          [this](const std::shared_ptr<AneJoinState> &join, bool ok,
                 const std::string &err) { on_die_job_done(join, ok, err); });
    }
  }

  void enqueue_op(
      std::uint32_t op_id,
      void *src_mtl_buffer,
      std::uint64_t src_offset,
      void *dst_mtl_buffer,
      std::uint64_t dst_offset,
      std::uint32_t out_stride,
      std::function<void(void *mtl_command_buffer)> encode_gpu_suffix,
      std::uint32_t src_stride,
      std::uint32_t dst_col_offset) {
    if (!stats_.active) {
      throw std::runtime_error("Dual-ANE engine is not active");
    }
    if (has_error()) {
      throw std::runtime_error("Prior Dual-ANE error: " + current_error());
    }
    if (op_id >= ops_.size()) {
      throw std::out_of_range("ANE op_id out of range");
    }
    if (src_mtl_buffer == nullptr || dst_mtl_buffer == nullptr) {
      throw std::invalid_argument("ANE enqueue_op buffers must be non-null");
    }

    const auto &info = ops_[op_id];
    const std::uint32_t T = stats_.chunk_tokens;
    const std::uint32_t K = info.in_features;
    const std::uint32_t eff_src_stride = src_stride != 0u ? src_stride : K;
    const bool dual = (stats_.active_die_count == 2u);

    const std::uint64_t seq = ++next_seq_;
    const std::uint32_t slot = static_cast<std::uint32_t>((seq - 1u) & 1u);

    const auto &bind0 = op_bindings_[op_id][0];
    const auto &bind1 = op_bindings_[op_id][dual ? 1u : 0u];

    auto join = std::make_shared<AneJoinState>();
    join->remaining.store(stats_.active_die_count, std::memory_order_relaxed);
    join->seq = seq;

    @autoreleasepool {
      id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)src_mtl_buffer;
      id<MTLBuffer> dst_buf = (__bridge id<MTLBuffer>)dst_mtl_buffer;

      // Register listener BEFORE committing CB_A so signal can never be missed.
      // When GPU pack finishes, listener dispatches ane0 (and ane1 in parallel!)
      // on persistent per-die worker threads with ZERO CPU blocking wait!
      Impl *self = this;
      CompiledAneBank *bank0 = bind0.bank;
      id req0 = bind0.requests[slot];
      CompiledAneBank *bank1 = dual ? bind1.bank : nullptr;
      id req1 = dual ? bind1.requests[slot] : nil;

      [ane_ready_event_
          notifyListener:event_listener_
                 atValue:seq
                   block:^(id<MTLSharedEvent> /*ev*/, std::uint64_t /*val*/) {
                     if (self->has_error()) {
                       self->signal_ane_done(seq);
                       return;
                     }
                     self->workers_[0]->enqueue(AneEvalJob{bank0, req0, join});
                     if (dual) {
                       self->workers_[1]->enqueue(AneEvalJob{bank1, req1, join});
                     }
                   }];

      // CB_A: (wait unpack_done[seq-2]) -> Pack BF16 -> FP16 planar IOSurface(s) -> signal ane_ready
      id<MTLCommandBuffer> cb_a = [queue_ commandBuffer];
      if (cb_a == nil) {
        latch_error("Failed to create CB_A command buffer");
        signal_ane_done(seq);
        throw std::runtime_error("Failed to create CB_A command buffer");
      }
      if (seq > 2u) {
        [cb_a encodeWaitForEvent:unpack_done_event_ value:(seq - 2u)];
      }
      id<MTLComputeCommandEncoder> pack_enc = [cb_a computeCommandEncoder];
      const AnePackParamsHost pack_p{T, K, eff_src_stride, 0u};
      if (dual) {
        [pack_enc setComputePipelineState:pack_dual_pso_];
        [pack_enc setBuffer:src_buf offset:src_offset atIndex:0];
        [pack_enc setBuffer:bind0.in_bufs[slot] offset:0 atIndex:1];
        [pack_enc setBuffer:bind1.in_bufs[slot] offset:0 atIndex:2];
        [pack_enc setBytes:&pack_p length:sizeof(pack_p) atIndex:3];
      } else {
        [pack_enc setComputePipelineState:pack_single_pso_];
        [pack_enc setBuffer:src_buf offset:src_offset atIndex:0];
        [pack_enc setBuffer:bind0.in_bufs[slot] offset:0 atIndex:1];
        [pack_enc setBytes:&pack_p length:sizeof(pack_p) atIndex:2];
      }
      const MTLSize pack_tg = MTLSizeMake((K + 31u) / 32u, (T + 31u) / 32u, 1u);
      const MTLSize threads = MTLSizeMake(8u, 32u, 1u);
      [pack_enc dispatchThreadgroups:pack_tg threadsPerThreadgroup:threads];
      [pack_enc endEncoding];
      [cb_a encodeSignalEvent:ane_ready_event_ value:seq];
      [cb_a addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        if (completed.status == MTLCommandBufferStatusError) {
          self->latch_error(
              ns_error_text(@"ANE pack CB_A failed", completed.error));
          // Wake ane_done_event_ so CB_B2 never hangs on Metal error.
          self->signal_ane_done(seq);
        }
      }];
      [cb_a commit];

      // CB_B1 (optional): GPU bf16/w4a8 suffix [N_ane, N) submitted IMMEDIATELY
      // to execute concurrently in hardware while ane0 || ane1 evaluate [0, N_ane)!
      if (encode_gpu_suffix) {
        id<MTLCommandBuffer> cb_b1 = [queue_ commandBuffer];
        if (cb_b1 == nil) {
          latch_error("Failed to create CB_B1 GPU suffix command buffer");
          throw std::runtime_error(
              "Failed to create CB_B1 GPU suffix command buffer");
        }
        encode_gpu_suffix((__bridge void *)cb_b1);
        [cb_b1 addCompletedHandler:^(id<MTLCommandBuffer> completed) {
          if (completed.status == MTLCommandBufferStatusError) {
            self->latch_error(ns_error_text(
                @"ANE co-execution GPU suffix CB_B1 failed", completed.error));
          }
        }];
        [cb_b1 commit];
      }

      // CB_B2: wait(ane_done, seq) -> Unpack FP16 planar IOSurface(s) -> dst[:, 0..N_ane) BF16
      //        -> signal(unpack_done, seq)
      id<MTLCommandBuffer> cb_b2 = [queue_ commandBuffer];
      if (cb_b2 == nil) {
        latch_error("Failed to create CB_B2 unpack command buffer");
        throw std::runtime_error("Failed to create CB_B2 unpack command buffer");
      }
      [cb_b2 encodeWaitForEvent:ane_done_event_ value:seq];
      id<MTLComputeCommandEncoder> unpack_enc = [cb_b2 computeCommandEncoder];
      const AneUnpackParamsHost unpack_p{
          T,
          info.split.die_rows,
          dual ? info.split.die_rows : 0u,
          out_stride,
          dst_col_offset,
          0u,
      };
      if (dual) {
        [unpack_enc setComputePipelineState:unpack_dual_pso_];
        [unpack_enc setBuffer:bind0.out_bufs[slot] offset:0 atIndex:0];
        [unpack_enc setBuffer:bind1.out_bufs[slot] offset:0 atIndex:1];
        [unpack_enc setBuffer:dst_buf offset:dst_offset atIndex:2];
        [unpack_enc setBytes:&unpack_p length:sizeof(unpack_p) atIndex:3];
      } else {
        [unpack_enc setComputePipelineState:unpack_single_pso_];
        [unpack_enc setBuffer:bind0.out_bufs[slot] offset:0 atIndex:0];
        [unpack_enc setBuffer:dst_buf offset:dst_offset atIndex:1];
        [unpack_enc setBytes:&unpack_p length:sizeof(unpack_p) atIndex:2];
      }
      const MTLSize unpack_tg =
          MTLSizeMake((info.split.ane_total_rows + 31u) / 32u,
                      (T + 31u) / 32u, 1u);
      [unpack_enc dispatchThreadgroups:unpack_tg threadsPerThreadgroup:threads];
      [unpack_enc endEncoding];
      [cb_b2 encodeSignalEvent:unpack_done_event_ value:seq];
      [cb_b2 addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        if (completed.status == MTLCommandBufferStatusError) {
          self->latch_error(
              ns_error_text(@"ANE unpack CB_B2 failed", completed.error));
        }
      }];
      [cb_b2 commit];

      last_cb_ = cb_b2;
      last_enqueued_seq_ = seq;
      enqueued_ops_.fetch_add(1u, std::memory_order_relaxed);
      stats_.total_enqueued_ops =
          enqueued_ops_.load(std::memory_order_relaxed);
    }
  }

  void flush_and_wait(std::uint32_t timeout_ms) {
    const std::uint64_t target = last_enqueued_seq_;
    id<MTLCommandBuffer> cb = last_cb_;
    last_cb_ = nil;

    if (target == 0u || cb == nil) {
      stats_.total_completed_ops =
          completed_ops_.load(std::memory_order_relaxed);
      if (has_error()) {
        stats_.active = false;
        throw std::runtime_error(current_error());
      }
      return;
    }

    // 1. Bounded wait on ANE completion event before waiting on Metal CB_B2.
    {
      std::unique_lock<std::mutex> lk(done_mutex_);
      const bool finished = done_cv_.wait_for(
          lk, std::chrono::milliseconds(timeout_ms),
          [this, target] { return signaled_done_ >= target; });
      if (!finished) {
        latch_error("ANE evaluation timed out waiting for shared event");
        if (target > signaled_done_) {
          signaled_done_ = target;
          [ane_done_event_ setSignaledValue:target];
        }
      }
    }

    // 2. Single chunk-boundary host wait for the final unpack command buffer.
    @autoreleasepool {
      [cb waitUntilCompleted];
      if (cb.status == MTLCommandBufferStatusError) {
        latch_error(ns_error_text(@"ANE final unpack CB failed", cb.error));
      }
    }
    stats_.total_completed_ops = completed_ops_.load(std::memory_order_relaxed);
    if (has_error()) {
      stats_.active = false;
      throw std::runtime_error(current_error());
    }
  }

  AneDualOffloadConfig config_{};
  AneEngineStats stats_{};
  std::vector<AneCompiledOpInfo> ops_;
  // (layer_index, role) -> op_id
  std::unordered_map<std::uint64_t, std::uint32_t> op_lookup_;

  // Metal objects (ARC managed)
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLComputePipelineState> pack_single_pso_ = nil;
  id<MTLComputePipelineState> pack_dual_pso_ = nil;
  id<MTLComputePipelineState> unpack_single_pso_ = nil;
  id<MTLComputePipelineState> unpack_dual_pso_ = nil;
  id<MTLSharedEvent> ane_ready_event_ = nil;
  id<MTLSharedEvent> ane_done_event_ = nil;
  id<MTLSharedEvent> unpack_done_event_ = nil;
  dispatch_queue_t listener_queue_ = nil;
  MTLSharedEventListener *event_listener_ = nil;

  // Per-die banks, IOSurface pools, pre-built requests, and persistent workers
  std::array<std::vector<std::unique_ptr<CompiledAneBank>>, 2> banks_;
  std::array<std::unordered_map<std::uint64_t, std::unique_ptr<SurfaceBundle>>,
             2>
      surface_pools_;
  std::vector<std::array<OpDieBinding, 2>> op_bindings_;
  std::array<std::unique_ptr<AneDieWorker>, 2> workers_;

  // Synchronization state
  std::atomic<std::uint64_t> next_seq_{0};
  std::uint64_t last_enqueued_seq_ = 0;
  id<MTLCommandBuffer> last_cb_ = nil;
  std::mutex done_mutex_;
  std::condition_variable done_cv_;
  std::uint64_t signaled_done_ = 0;
  mutable std::mutex error_mutex_;
  std::string latched_error_;
  std::atomic<std::uint64_t> enqueued_ops_{0};
  std::atomic<std::uint64_t> completed_ops_{0};
};

// -----------------------------------------------------------------------------
// AneDualOffloadEngine public methods
// -----------------------------------------------------------------------------

std::filesystem::path AneDualOffloadEngine::default_cache_root() {
  @autoreleasepool {
    NSArray<NSString *> *paths = NSSearchPathForDirectoriesInDomains(
        NSCachesDirectory, NSUserDomainMask, YES);
    NSString *base =
        paths.firstObject != nil ? paths.firstObject : NSTemporaryDirectory();
    NSString *dir =
        [[base stringByAppendingPathComponent:@"io.ninfer.miv"]
            stringByAppendingPathComponent:@"ane_banks_v1"];
    return std::filesystem::path([dir UTF8String]);
  }
}

std::string AneDualOffloadEngine::compute_cache_key(
    const AneDualOffloadConfig &config,
    std::span<const AneOffloadOpSpec> specs,
    std::uint32_t active_dies,
    std::string_view os_build) {
  const double alpha =
      config.alpha_override.value_or(
          config.linear_route == AneLinearRoute::W4A8 ? config.alpha_w4a8
                                                      : config.alpha_bf16);
  const auto alpha_permille = static_cast<std::uint32_t>(
      std::llround(std::clamp(alpha, 0.0, config.max_alpha) * 1000.0));

  std::ostringstream oss;
  oss << "ane_schema_v" << kAneCacheSchemaVersion
      << "|digest=" << config.artifact_digest
      << "|T=" << config.chunk_tokens
      << "|route=" << to_string(config.linear_route)
      << "|alpha=" << alpha_permille
      << "|dies=" << active_dies
      << "|max_bank=" << config.max_bank_bytes_per_die
      << "|max_procs=" << config.max_procedures_per_bank
      << "|fp16=" << (config.use_fp16_weights ? 1 : 0)
      << "|swiglu_only=" << (config.mlp_swiglu_only ? 1 : 0)
      << "|max_k=" << config.max_in_features
      << "|os=" << os_build
      << "|ops=" << specs.size();
  for (const auto &s : specs) {
    oss << ";" << s.layer_index << ":" << static_cast<int>(s.role) << ":"
        << s.name << ":" << s.weight.object_id << ":" << s.weight.format << ":"
        << s.weight.layout << ":" << s.weight.row_count << "x"
        << s.weight.in_features;
    if (s.up_weight.has_value()) {
      oss << "+" << s.up_weight->object_id << ":" << s.up_weight->format;
    }
  }
  return sha256_hex_string(oss.str());
}

std::unique_ptr<AneDualOffloadEngine> AneDualOffloadEngine::build(
    std::span<const AneOffloadOpSpec> specs,
    const AneDualOffloadConfig &config) {
  auto impl = std::make_unique<Impl>();
  impl->config_ = config;
  impl->stats_.chunk_tokens = config.chunk_tokens;
  impl->stats_.linear_route = config.linear_route;
  const double target_alpha =
      config.alpha_override.value_or(
          config.linear_route == AneLinearRoute::W4A8 ? config.alpha_w4a8
                                                      : config.alpha_bf16);
  impl->stats_.target_alpha = std::clamp(target_alpha, 0.0, config.max_alpha);

  // 1. Validate strict GDN recurrence exclusion on every supplied op spec BEFORE
  //    checking hardware so misconfigurations are caught deterministically.
  for (const auto &spec : specs) {
    if (!spec.name.empty() && !ane_stage_role_eligible(spec.name)) {
      throw std::invalid_argument(
          "Strict GDN recurrence exclusion violated by ANE op spec: " +
          spec.name);
    }
    if (!spec.weight.name.empty() &&
        !ane_stage_role_eligible(spec.weight.name)) {
      throw std::invalid_argument(
          "Strict GDN recurrence exclusion violated by ANE weight tensor: " +
          spec.weight.name);
    }
    if (spec.role == AneProjectionRole::MlpGateUpSwiGlu) {
      if (!spec.up_weight.has_value()) {
        throw std::invalid_argument(
            "MlpGateUpSwiGlu spec missing up_weight: " + spec.name);
      }
      if (!spec.up_weight->name.empty() &&
          !ane_stage_role_eligible(spec.up_weight->name)) {
        throw std::invalid_argument(
            "Strict GDN recurrence exclusion violated by ANE up_weight tensor: " +
            spec.up_weight->name);
      }
      if (spec.up_weight->row_count != spec.weight.row_count ||
          spec.up_weight->in_features != spec.weight.in_features) {
        throw std::invalid_argument(
            "MlpGateUpSwiGlu gate/up shape mismatch: " + spec.name);
      }
    }
  }

  // 2. Hardware probe (graceful no-op fallback if ANE unavailable).
  const AneHardwareInfo hw = probe_ane_hardware();
  impl->stats_.physical_die_count = hw.physical_die_count;
  if (!hw.is_available()) {
    impl->stats_.available = false;
    impl->stats_.active = false;
    impl->stats_.status_message = hw.unavailable_reason;
    return std::unique_ptr<AneDualOffloadEngine>(
        new AneDualOffloadEngine(std::move(impl)));
  }
  impl->stats_.available = true;

  if (config.chunk_tokens < 32u || (config.chunk_tokens % 32u) != 0u) {
    throw std::invalid_argument(
        "ANE chunk_tokens must be >= 32 and a multiple of 32");
  }

  // 3. Determine active die topology (2-die pinned on Ultra, 1-die fallback on single-die Mac).
  std::uint32_t active_dies = 1u;
  bool use_instance_hints = false;
  if (!config.force_single_die && hw.physical_die_count >= 2u) {
    active_dies = 2u;
    use_instance_hints = true;
  } else if (config.simulate_dual_banks_on_single_die) {
    active_dies = 2u;
    use_instance_hints = false;
  } else {
    active_dies = 1u;
    use_instance_hints = false;
  }
  impl->stats_.active_die_count = active_dies;

  // 4. Plan 128-row-aligned splits & shard procedures into < 2.5 GiB banks per die.
  struct ShardPlan {
    std::vector<const AneOffloadOpSpec *> specs;
    std::vector<AneCompiledOpInfo> infos;
    std::uint64_t estimated_bytes = 64u;
  };
  std::vector<ShardPlan> shards;
  std::uint64_t total_per_die_bytes = 0u;

  for (const auto &spec : specs) {
    if (config.mlp_swiglu_only &&
        spec.role != AneProjectionRole::MlpGateUpSwiGlu) {
      continue;
    }
    if (!ane_weight_view_supported(spec.weight)) {
      continue;
    }
    if (spec.role == AneProjectionRole::MlpGateUpSwiGlu &&
        !ane_weight_view_supported(*spec.up_weight)) {
      continue;
    }
    const auto out_features = static_cast<std::uint32_t>(spec.weight.row_count);
    const auto in_features =
        static_cast<std::uint32_t>(spec.weight.in_features);
    if (config.max_in_features > 0u && in_features > config.max_in_features) {
      continue;
    }
    const AneChannelSplit split = compute_ane_channel_split(
        out_features, target_alpha, active_dies, config.max_alpha);
    if (!split.enabled()) {
      continue;
    }

    const std::uint32_t mult =
        (spec.role == AneProjectionRole::MlpGateUpSwiGlu) ? 2u : 1u;
    const std::uint64_t ng = in_features / kAneGroupSize;
    const std::uint64_t single_w_bytes =
        config.use_fp16_weights
            ? (static_cast<std::uint64_t>(split.die_rows) * in_features * 2u + 128u)
            : (static_cast<std::uint64_t>(split.die_rows) * in_features +
               static_cast<std::uint64_t>(split.die_rows) * ng * 3u + 3u * 128u);
    const std::uint64_t op_bytes = single_w_bytes * mult;

    // Respect per-die total resident cap (< 4.0 GiB ANE address window).
    if (total_per_die_bytes + op_bytes > config.max_total_bytes_per_die) {
      break;
    }

    if (shards.empty() ||
        shards.back().specs.size() >= config.max_procedures_per_bank ||
        shards.back().estimated_bytes + op_bytes >
            config.max_bank_bytes_per_die) {
      shards.emplace_back();
    }

    auto &cur_shard = shards.back();
    AneCompiledOpInfo info{};
    info.op_id = static_cast<std::uint32_t>(impl->ops_.size());
    info.layer_index = spec.layer_index;
    info.role = spec.role;
    info.name = spec.name.empty() ? spec.weight.name : spec.name;
    info.in_features = in_features;
    info.split = split;
    info.shard_index = static_cast<std::uint32_t>(shards.size() - 1u);
    info.procedure_index = static_cast<std::uint32_t>(cur_shard.specs.size());
    info.per_die_weight_bytes = op_bytes;

    cur_shard.specs.push_back(&spec);
    cur_shard.infos.push_back(info);
    cur_shard.estimated_bytes += op_bytes;
    total_per_die_bytes += op_bytes;

    const std::uint64_t key =
        (static_cast<std::uint64_t>(info.layer_index) << 8u) |
        static_cast<std::uint64_t>(info.role);
    impl->op_lookup_.emplace(key, info.op_id);
    impl->ops_.push_back(std::move(info));
  }

  if (impl->ops_.empty()) {
    impl->stats_.active = false;
    impl->stats_.status_message = "No eligible projections met ANE split criteria";
    return std::unique_ptr<AneDualOffloadEngine>(
        new AneDualOffloadEngine(std::move(impl)));
  }

  // 5. Compute cache key & directory.
  const std::filesystem::path cache_root =
      config.cache_root.empty() ? default_cache_root() : config.cache_root;
  const std::string cache_key =
      compute_cache_key(config, specs, active_dies, hw.os_build);
  const std::filesystem::path cache_dir = cache_root / cache_key;
  impl->stats_.cache_key = cache_key;
  impl->stats_.cache_dir = cache_dir;

  // 6. Initialize Metal pack/unpack pipelines and MTLSharedEvents.
  impl->init_metal_pipelines(config);

  // 7. Compile or warm-load procedure banks for each active die.
  //    On 2-die Ultra hardware, pin ane0 -> instance 1 and ane1 -> instance 2.
  //    If pinned instance compilation fails unexpectedly, degrade gracefully to 1 unpinned die.
  auto compile_all_dies = [&](std::uint32_t dies, bool pinned) {
    impl->banks_[0].clear();
    impl->banks_[1].clear();
    impl->stats_.disk_cache_hits = 0u;
    impl->stats_.disk_cache_misses = 0u;
    impl->stats_.native_compiled_cache_hits = 0u;
    impl->stats_.total_weight_bytes_per_die = 0u;
    impl->stats_.die_pinned[0] = false;
    impl->stats_.die_pinned[1] = false;

    for (std::uint32_t die = 0; die < dies; ++die) {
      const int ane_inst = pinned ? static_cast<int>(die + 1u) : 0;
      impl->stats_.die_pinned[die] = pinned;
      for (std::size_t s = 0; s < shards.size(); ++s) {
        auto bank = compile_or_load_bank(
            cache_dir, config.enable_disk_cache, cache_key, hw.os_build, die,
            static_cast<std::uint32_t>(s), ane_inst, config.chunk_tokens,
            config.use_fp16_weights, shards[s].specs, shards[s].infos);
        if (bank->disk_cache_hit) {
          ++impl->stats_.disk_cache_hits;
        } else {
          ++impl->stats_.disk_cache_misses;
        }
        if (bank->native_compiled_hit) {
          ++impl->stats_.native_compiled_cache_hits;
        }
        if (die == 0u) {
          impl->stats_.total_weight_bytes_per_die += bank->weight_bytes;
        }
        impl->banks_[die].push_back(std::move(bank));
      }
    }
  };

  try {
    compile_all_dies(active_dies, use_instance_hints);
  } catch (const std::exception &e) {
    if (use_instance_hints && active_dies == 2u) {
      // Fallback gracefully to unpinned single-die if instance hints rejected.
      active_dies = 1u;
      impl->stats_.active_die_count = 1u;
      for (auto &info : impl->ops_) {
        info.split.active_dies = 1u;
        info.split.ane_total_rows = info.split.die_rows;
        info.split.gpu_row_offset = info.split.die_rows;
        info.split.gpu_rows = info.split.total_out_features - info.split.die_rows;
        info.split.effective_alpha =
            static_cast<double>(info.split.ane_total_rows) /
            static_cast<double>(info.split.total_out_features);
      }
      compile_all_dies(1u, false);
      impl->stats_.status_message =
          std::string("Pinned 2-die init fell back to 1-die unpinned: ") +
          e.what();
    } else {
      throw;
    }
  }

  impl->stats_.compiled_ops = static_cast<std::uint32_t>(impl->ops_.size());
  impl->stats_.bank_shards_per_die =
      static_cast<std::uint32_t>(shards.size());

  // 8. Pre-build ping-pong IOSurfaces & _ANERequests and start persistent workers.
  impl->bind_op_requests();
  impl->start_workers();
  impl->stats_.active = true;
  if (impl->stats_.status_message.empty()) {
    impl->stats_.status_message = "ok";
  }

  return std::unique_ptr<AneDualOffloadEngine>(
      new AneDualOffloadEngine(std::move(impl)));
}

std::unique_ptr<AneDualOffloadEngine> AneDualOffloadEngine::build_from_model(
    const MaterializedModel &model, const QwenLayerCatalog &catalog,
    const AneDualOffloadConfig &config) {
  std::vector<AneOffloadOpSpec> specs;
  for (const auto &layer : catalog.layers()) {
    const LayerStageRecord *mlp_gate = nullptr;
    const LayerStageRecord *mlp_up = nullptr;
    const LayerStageRecord *mlp_down = nullptr;
    const LayerStageRecord *gdn_z = nullptr;
    const LayerStageRecord *gdn_out = nullptr;
    const LayerStageRecord *attn_o = nullptr;

    for (const auto &stage : layer.stages) {
      if (stage.kind != LayerStageKind::linear ||
          stage.object_id == UINT32_MAX ||
          !model.has_object(stage.object_id)) {
        continue;
      }
      // Enforce strict GDN recurrence exclusion & Attn QKV exclusion.
      if (!ane_stage_role_eligible(stage.role) ||
          !ane_stage_role_eligible(stage.object_name)) {
        continue;
      }
      if (!layer.moe) {
        if (stage.role == "mlp_gate") {
          mlp_gate = &stage;
        } else if (stage.role == "mlp_up") {
          mlp_up = &stage;
        } else if (stage.role == "mlp_down") {
          mlp_down = &stage;
        }
      }
      if (layer.type == LayerType::linear_attention) {
        if (stage.role == "gdn_z") {
          gdn_z = &stage;
        } else if (stage.role == "gdn_out") {
          gdn_out = &stage;
        }
      } else if (layer.type == LayerType::full_attention) {
        if (stage.role == "attention_o") {
          attn_o = &stage;
        }
      }
    }

    // Add ONLY the 5 eligible non-recurrent projections in execution order.
    if (gdn_z != nullptr) {
      specs.push_back(AneOffloadOpSpec{
          .layer_index = layer.layer,
          .role = AneProjectionRole::GdnInProjZ,
          .name = gdn_z->object_name,
          .weight = model.parent_view(gdn_z->object_id),
      });
    }
    if (gdn_out != nullptr) {
      specs.push_back(AneOffloadOpSpec{
          .layer_index = layer.layer,
          .role = AneProjectionRole::GdnOutProj,
          .name = gdn_out->object_name,
          .weight = model.parent_view(gdn_out->object_id),
      });
    }
    if (attn_o != nullptr) {
      specs.push_back(AneOffloadOpSpec{
          .layer_index = layer.layer,
          .role = AneProjectionRole::AttnOProj,
          .name = attn_o->object_name,
          .weight = model.parent_view(attn_o->object_id),
      });
    }
    if (mlp_gate != nullptr && mlp_up != nullptr) {
      specs.push_back(AneOffloadOpSpec{
          .layer_index = layer.layer,
          .role = AneProjectionRole::MlpGateUpSwiGlu,
          .name = "mlp_gate_up_swiglu",
          .weight = model.parent_view(mlp_gate->object_id),
          .up_weight = model.parent_view(mlp_up->object_id),
      });
    }
    if (mlp_down != nullptr) {
      specs.push_back(AneOffloadOpSpec{
          .layer_index = layer.layer,
          .role = AneProjectionRole::MlpDown,
          .name = mlp_down->object_name,
          .weight = model.parent_view(mlp_down->object_id),
      });
    }
  }
  return build(specs, config);
}

AneDualOffloadEngine::AneDualOffloadEngine(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
AneDualOffloadEngine::AneDualOffloadEngine(AneDualOffloadEngine &&) noexcept =
    default;
AneDualOffloadEngine &
AneDualOffloadEngine::operator=(AneDualOffloadEngine &&) noexcept = default;
AneDualOffloadEngine::~AneDualOffloadEngine() = default;

bool AneDualOffloadEngine::is_available() const noexcept {
  return impl_ != nullptr && impl_->stats_.available;
}

bool AneDualOffloadEngine::is_active() const noexcept {
  return impl_ != nullptr && impl_->stats_.active;
}

const AneEngineStats &AneDualOffloadEngine::stats() const noexcept {
  return impl_->stats_;
}

std::span<const AneCompiledOpInfo> AneDualOffloadEngine::ops() const noexcept {
  return impl_->ops_;
}

const AneCompiledOpInfo *AneDualOffloadEngine::find_op(
    std::uint32_t layer_index, AneProjectionRole role) const noexcept {
  const std::uint64_t key =
      (static_cast<std::uint64_t>(layer_index) << 8u) |
      static_cast<std::uint64_t>(role);
  const auto it = impl_->op_lookup_.find(key);
  if (it == impl_->op_lookup_.end()) {
    return nullptr;
  }
  return &impl_->ops_[it->second];
}

const AneCompiledOpInfo *AneDualOffloadEngine::find_op_by_id(
    std::uint32_t op_id) const noexcept {
  if (op_id >= impl_->ops_.size()) {
    return nullptr;
  }
  return &impl_->ops_[op_id];
}

bool AneDualOffloadEngine::can_offload_chunk(
    std::uint32_t tokens) const noexcept {
  return is_active() && tokens == impl_->stats_.chunk_tokens;
}

AneSharedBuffer AneDualOffloadEngine::make_shared_buffer(
    std::uint64_t bytes) const {
  if (impl_ == nullptr || impl_->device_ == nil) {
    throw std::runtime_error("ANE engine Metal device is not initialized");
  }
  if (bytes == 0u) {
    throw std::invalid_argument("ANE shared buffer size must be positive");
  }
  @autoreleasepool {
    id<MTLBuffer> buf =
        [impl_->device_ newBufferWithLength:static_cast<NSUInteger>(bytes)
                                    options:MTLResourceStorageModeShared |
                                            MTLResourceHazardTrackingModeTracked];
    if (buf == nil) {
      throw std::bad_alloc();
    }
    auto bimpl = std::make_unique<AneSharedBuffer::Impl>();
    bimpl->buffer = buf;
    bimpl->length = bytes;
    return AneSharedBuffer(std::move(bimpl));
  }
}

AneCalibrationResult AneDualOffloadEngine::calibrate(
    double gpu_only_ms, double gpu_suffix_ms,
    std::uint32_t representative_op_id) {
  AneCalibrationResult res{};
  res.gpu_only_ms = gpu_only_ms;
  res.gpu_suffix_ms = gpu_suffix_ms;
  res.calibrated_op_id = representative_op_id;
  if (!is_active() || representative_op_id >= impl_->ops_.size()) {
    res.active = is_active();
    res.reason = "engine_inactive_or_invalid_op";
    return res;
  }

  const auto &info = impl_->ops_[representative_op_id];
  const std::uint32_t T = impl_->stats_.chunk_tokens;
  const std::uint32_t K = info.in_features;
  const std::uint32_t N = info.split.total_out_features;

  auto src = make_shared_buffer(static_cast<std::uint64_t>(T) * K * 2u);
  auto dst = make_shared_buffer(static_cast<std::uint64_t>(T) * N * 2u);
  std::memset(src.data(), 0, static_cast<std::size_t>(src.size()));
  std::memset(dst.data(), 0, static_cast<std::size_t>(dst.size()));

  // 2 warmup iterations (primes ANE JIT & page residency) + 2 timed iterations.
  for (int i = 0; i < 2; ++i) {
    run_op_sync(representative_op_id, src.native_mtl_buffer(), 0,
                dst.native_mtl_buffer(), 0, N);
  }
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 2; ++i) {
    run_op_sync(representative_op_id, src.native_mtl_buffer(), 0,
                dst.native_mtl_buffer(), 0, N);
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double ane_pair_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / 2.0;

  res.performed = true;
  res.ane_pair_ms = ane_pair_ms;

  const double effective_hybrid_ms = std::max(ane_pair_ms, gpu_suffix_ms);
  const double threshold_ms =
      gpu_only_ms * impl_->config_.calibration_bypass_ratio;
  if (gpu_only_ms > 0.0 && effective_hybrid_ms > threshold_ms) {
    impl_->stats_.active = false;
    impl_->stats_.auto_bypassed = true;
    impl_->stats_.status_message = "auto_bypassed_by_calibration";
    res.active = false;
    res.auto_bypassed = true;
    res.reason = "ane_pair_slower_than_gpu_only";
  } else {
    impl_->stats_.active = true;
    impl_->stats_.auto_bypassed = false;
    res.active = true;
    res.auto_bypassed = false;
    res.reason = "ane_beneficial";
  }
  return res;
}

void AneDualOffloadEngine::enqueue_op(
    std::uint32_t op_id,
    void *src_mtl_buffer,
    std::uint64_t src_offset,
    void *dst_mtl_buffer,
    std::uint64_t dst_offset,
    std::uint32_t out_stride,
    std::function<void(void *mtl_command_buffer)> encode_gpu_suffix,
    std::uint32_t src_stride,
    std::uint32_t dst_col_offset) {
  impl_->enqueue_op(op_id, src_mtl_buffer, src_offset, dst_mtl_buffer,
                    dst_offset, out_stride, std::move(encode_gpu_suffix),
                    src_stride, dst_col_offset);
}

void AneDualOffloadEngine::flush_and_wait(std::uint32_t timeout_ms) {
  impl_->flush_and_wait(timeout_ms);
}

void AneDualOffloadEngine::run_op_sync(
    std::uint32_t op_id,
    void *src_mtl_buffer,
    std::uint64_t src_offset,
    void *dst_mtl_buffer,
    std::uint64_t dst_offset,
    std::uint32_t out_stride,
    std::function<void(void *mtl_command_buffer)> encode_gpu_suffix,
    std::uint32_t src_stride,
    std::uint32_t dst_col_offset) {
  impl_->enqueue_op(op_id, src_mtl_buffer, src_offset, dst_mtl_buffer,
                    dst_offset, out_stride, std::move(encode_gpu_suffix),
                    src_stride, dst_col_offset);
  impl_->flush_and_wait(30000u);
}

void *AneDualOffloadEngine::native_mtl_device() const noexcept {
  return impl_ != nullptr ? (__bridge void *)impl_->device_ : nullptr;
}

void *AneDualOffloadEngine::native_mtl_command_queue() const noexcept {
  return impl_ != nullptr ? (__bridge void *)impl_->queue_ : nullptr;
}

} // namespace omlx::m5::ane
