#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/metal.h"

namespace omlx::m5 {

enum class KvStorageFormat : std::uint8_t {
  BFloat16 = 0,
  Affine4Group128 = 1,
};

[[nodiscard]] constexpr std::string_view
kv_storage_format_name(KvStorageFormat format) noexcept {
  switch (format) {
  case KvStorageFormat::BFloat16:
    return "bf16";
  case KvStorageFormat::Affine4Group128:
    return "affine4_g128";
  }
  return "bf16";
}

[[nodiscard]] KvStorageFormat parse_kv_storage_format(std::string_view name);

// Per-page bounding-box summary stored at the tail of each Affine4Group128
// physical page (32 B per KV head: 8 subbands x FP16 min + 8 subbands x FP16
// max) to enable O(1)-metadata QSA / Combined-Sparse top-k page selection.
struct alignas(16) KvAffine4PageSummary {
  std::uint16_t subband_min_f16[8];
  std::uint16_t subband_max_f16[8];
};
static_assert(sizeof(KvAffine4PageSummary) == 32);

// Per-vector scale/zero metadata in Affine4Group128 (16 B per vector: up to 4
// groups of G=128, i.e. up to head_dim=512; for head_dim=256: 128 B packed
// uint4 codes + 16 B half scale/zero = 144 B/vec).
struct alignas(16) KvAffine4VecMeta {
  std::uint16_t scale_f16[4];
  std::uint16_t zero_f16[4];
};
static_assert(sizeof(KvAffine4VecMeta) == 16);

[[nodiscard]] std::uint64_t kv_page_bytes(KvStorageFormat format,
                                          std::uint32_t page_tokens,
                                          std::uint32_t kv_heads,
                                          std::uint32_t head_dim);

[[nodiscard]] std::uint64_t kv_slab_bytes(KvStorageFormat format,
                                          std::uint32_t physical_pages,
                                          std::uint32_t page_tokens,
                                          std::uint32_t kv_heads,
                                          std::uint32_t head_dim);

[[nodiscard]] std::uint32_t kv_copy_words16_per_page(KvStorageFormat format,
                                                     std::uint32_t page_tokens,
                                                     std::uint32_t kv_heads,
                                                     std::uint32_t head_dim);

struct KvPageCopy {
  std::uint32_t source = 0;
  std::uint32_t destination = 0;
};

// Shared physical KV page allocator across multi-slot QwenForward instances so
// B=2..8 slots share one physical KV slab pool without allocating 8x slabs.
class SharedPhysicalPageAllocator final {
public:
  explicit SharedPhysicalPageAllocator(std::uint32_t physical_pages);
  SharedPhysicalPageAllocator(const SharedPhysicalPageAllocator &) = delete;
  SharedPhysicalPageAllocator &
  operator=(const SharedPhysicalPageAllocator &) = delete;

  [[nodiscard]] std::uint32_t physical_pages() const noexcept;
  [[nodiscard]] std::uint32_t free_pages() const;
  [[nodiscard]] std::uint32_t ref_count(std::uint32_t physical_slot) const;

private:
  friend class KvPagePool;
  mutable std::mutex mutex_;
  std::uint32_t physical_ = 0;
  std::vector<std::uint32_t> refs_;
  std::vector<std::uint32_t> free_;
};

// C1 page ownership shared by standalone Attention, QwenForward, and cross-slot
// COW SharedPrefixCache. Slots are reserved in a bounded GPU slab; logical
// pages are mapped only as needed.
class KvPagePool final {
public:
  static constexpr std::uint32_t unmapped = UINT32_MAX;
  KvPagePool(const MetalContext &, std::uint32_t capacity_tokens,
             std::uint32_t page_tokens = 64, std::uint32_t physical_pages = 0,
             KvStorageFormat storage_format = KvStorageFormat::BFloat16);
  KvPagePool(const MetalContext &, std::uint32_t capacity_tokens,
             std::uint32_t page_tokens,
             std::shared_ptr<SharedPhysicalPageAllocator> shared_allocator,
             KvStorageFormat storage_format = KvStorageFormat::BFloat16);
  ~KvPagePool();
  KvPagePool(KvPagePool &&) noexcept;
  KvPagePool &operator=(KvPagePool &&) noexcept;
  KvPagePool(const KvPagePool &) = delete;
  KvPagePool &operator=(const KvPagePool &) = delete;

  [[nodiscard]] std::shared_ptr<SharedPhysicalPageAllocator>
  shared_allocator() const noexcept;
  [[nodiscard]] KvStorageFormat storage_format() const noexcept;
  [[nodiscard]] std::uint32_t tokens() const;
  [[nodiscard]] std::uint32_t logical_pages() const;
  [[nodiscard]] std::uint32_t physical_pages() const;
  [[nodiscard]] std::uint32_t free_pages() const;
  [[nodiscard]] std::uint32_t mapped_pages() const;
  [[nodiscard]] std::uint32_t page_tokens() const;
  [[nodiscard]] std::uint32_t physical_slot(std::uint32_t logical) const;
  [[nodiscard]] const MetalBuffer &table_buffer() const;

  // Reserve the entire append before any GPU write. Failure leaves the old
  // mapping/frontier untouched. Copy moves must finish before append dispatch.
  [[nodiscard]] std::vector<KvPageCopy> begin_append(std::uint32_t count);
  void commit(std::uint32_t count);
  void abort();
  void truncate(std::uint32_t count);
  void reset();

  [[nodiscard]] std::uint64_t retain_prefix();
  void restore_prefix(std::uint64_t lease);
  void release_prefix(std::uint64_t lease);

  // Direct physical-page COW reference counting for cross-slot SharedPrefixCache.
  void retain_page(std::uint32_t physical_slot);
  void release_page(std::uint32_t physical_slot);
  [[nodiscard]] std::uint32_t ref_count(std::uint32_t physical_slot) const;
  [[nodiscard]] std::optional<KvPageCopy>
  cow_page_if_shared(std::uint32_t logical_page);
  [[nodiscard]] std::vector<std::uint32_t> mapped_slots() const;
  void install_shared_prefix(std::uint32_t token_count,
                             std::span<const std::uint32_t> physical_slots);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Installation-time GPU copy of an entire key/value page for tail COW.
class KvPageCopier final {
public:
  KvPageCopier(const MetalContext &, const MetalBuffer &keys,
               const MetalBuffer &values, std::uint32_t page_tokens,
               std::uint32_t kv_width, std::uint32_t physical_pages,
               const std::filesystem::path &libraries, std::string key);
  KvPageCopier(const MetalContext &, const MetalBuffer &keys,
               const MetalBuffer &values, KvStorageFormat format,
               std::uint32_t page_tokens, std::uint32_t kv_heads,
               std::uint32_t head_dim, std::uint32_t physical_pages,
               const std::filesystem::path &libraries, std::string key);
  ~KvPageCopier();
  KvPageCopier(KvPageCopier &&) noexcept;
  KvPageCopier &operator=(KvPageCopier &&) noexcept;
  KvPageCopier(const KvPageCopier &) = delete;
  KvPageCopier &operator=(const KvPageCopier &) = delete;
  [[nodiscard]] MetalSubmission execute(KvPageCopy move);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5
