#include "omlx_m5/kv_page_pool.h"
#include "forward_plan.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace omlx::m5 {
namespace {
using U32 = std::uint32_t;
using U64 = std::uint64_t;

U32 pages_for(U32 tokens, U32 page_tokens) {
  return U32((U64(tokens) + page_tokens - 1) / page_tokens);
}

U64 checked_mul(U64 a, U64 b) {
  if (a && b > std::numeric_limits<U64>::max() / a)
    throw std::overflow_error("KV page byte geometry overflow");
  return a * b;
}

U64 checked_add(U64 a, U64 b) {
  if (b > std::numeric_limits<U64>::max() - a)
    throw std::overflow_error("KV page byte size overflow");
  return a + b;
}
} // namespace

KvStorageFormat parse_kv_storage_format(std::string_view name) {
  if (name.empty() || name == "bf16" || name == "bfloat16")
    return KvStorageFormat::BFloat16;
  if (name == "affine4_g128" || name == "affine4")
    return KvStorageFormat::Affine4Group128;
  throw std::invalid_argument("unsupported KV storage format: " +
                              std::string(name));
}

U64 kv_page_bytes(KvStorageFormat format, U32 page_tokens, U32 kv_heads,
                  U32 head_dim) {
  if (!page_tokens || !kv_heads || !head_dim)
    throw std::invalid_argument("KV page geometry must be positive");
  switch (format) {
  case KvStorageFormat::BFloat16:
    return checked_mul(checked_mul(page_tokens, kv_heads),
                       checked_mul(head_dim, 2));
  case KvStorageFormat::Affine4Group128: {
    if ((head_dim % 2) != 0 || head_dim > 512)
      throw std::invalid_argument(
          "Affine4Group128 KV requires even head_dim <= 512");
    const U64 vec_bytes = U64(head_dim / 2) + sizeof(KvAffine4VecMeta);
    const U64 head_bytes = checked_add(
        checked_mul(page_tokens, vec_bytes), sizeof(KvAffine4PageSummary));
    return checked_mul(kv_heads, head_bytes);
  }
  }
  throw std::invalid_argument("unknown KV storage format");
}

U64 kv_slab_bytes(KvStorageFormat format, U32 physical_pages, U32 page_tokens,
                  U32 kv_heads, U32 head_dim) {
  if (!physical_pages)
    throw std::invalid_argument("KV slab physical_pages must be positive");
  return checked_mul(physical_pages,
                     kv_page_bytes(format, page_tokens, kv_heads, head_dim));
}

U32 kv_copy_words16_per_page(KvStorageFormat format, U32 page_tokens,
                             U32 kv_heads, U32 head_dim) {
  const U64 bytes = kv_page_bytes(format, page_tokens, kv_heads, head_dim);
  if ((bytes % 2) != 0 || (bytes / 2) > UINT32_MAX)
    throw std::overflow_error("KV page copy element count overflow");
  return U32(bytes / 2);
}

SharedPhysicalPageAllocator::SharedPhysicalPageAllocator(U32 physical_pages)
    : physical_(physical_pages), refs_(physical_pages, 0) {
  if (!physical_pages || physical_pages == UINT32_MAX)
    throw std::invalid_argument("KV physical page pool size invalid");
  free_.reserve(physical_pages);
  for (U32 slot = physical_pages; slot > 0; --slot)
    free_.push_back(slot - 1);
}

U32 SharedPhysicalPageAllocator::physical_pages() const noexcept {
  return physical_;
}

U32 SharedPhysicalPageAllocator::free_pages() const {
  std::scoped_lock lock(mutex_);
  return U32(free_.size());
}

U32 SharedPhysicalPageAllocator::ref_count(U32 physical_slot) const {
  std::scoped_lock lock(mutex_);
  if (physical_slot >= physical_)
    throw std::out_of_range("KV ref_count slot outside pool");
  return refs_[physical_slot];
}

class KvPagePool::Impl {
public:
  struct Lease {
    U32 tokens;
    std::vector<U32> pages;
  };
  struct Before {
    std::vector<U32> mapping;
    U32 tokens = 0;
    U32 cow_old_slot = KvPagePool::unmapped;
  };
  KvStorageFormat format = KvStorageFormat::BFloat16;
  U32 capacity = 0, page_size = 0, frontier = 0;
  U32 pending_width = 0;
  std::shared_ptr<SharedPhysicalPageAllocator> alloc;
  std::vector<U32> mapping;
  std::vector<U32> uploaded;
  std::map<std::uint64_t, Lease> leases;
  std::uint64_t next_lease = 1;
  std::unique_ptr<Before> before;
  MetalBuffer table;

  void upload(const std::vector<U32> &value) {
    if (uploaded.size() != value.size()) {
      table.write(std::as_bytes(std::span(value)));
      uploaded = value;
      return;
    }
    std::size_t first = value.size();
    std::size_t last = 0;
    for (std::size_t i = 0; i < value.size(); ++i) {
      if (value[i] != uploaded[i]) {
        if (first == value.size())
          first = i;
        last = i;
      }
    }
    if (first == value.size())
      return;
    const auto span_slice =
        std::span<const U32>(value).subspan(first, last - first + 1);
    table.write(std::as_bytes(span_slice), U64(first) * sizeof(U32));
    std::copy(span_slice.begin(), span_slice.end(),
              uploaded.begin() + std::ptrdiff_t(first));
  }

  static U32 alloc_locked(SharedPhysicalPageAllocator &a) {
    if (a.free_.empty())
      throw std::overflow_error("KV physical page pool exhausted");
    const U32 slot = a.free_.back();
    a.free_.pop_back();
    if (a.refs_[slot]++)
      throw std::logic_error("KV free slot has an active reference");
    return slot;
  }

  static void release_locked(SharedPhysicalPageAllocator &a, U32 slot) {
    if (slot == KvPagePool::unmapped || slot >= a.refs_.size() || !a.refs_[slot])
      throw std::logic_error("KV page reference underflow");
    if (--a.refs_[slot] == 0)
      a.free_.push_back(slot);
  }
};

KvPagePool::KvPagePool(const MetalContext &context, U32 capacity_tokens,
                       U32 page_tokens, U32 physical_pages,
                       KvStorageFormat storage_format)
    : KvPagePool(
          context, capacity_tokens, page_tokens,
          [&]() -> std::shared_ptr<SharedPhysicalPageAllocator> {
            if (!capacity_tokens || !page_tokens)
              throw std::invalid_argument("KV page pool geometry is empty");
            const auto logical = pages_for(capacity_tokens, page_tokens);
            const U32 phys = physical_pages ? physical_pages : logical + 1;
            if (phys < logical || phys == UINT32_MAX)
              throw std::invalid_argument("KV physical page pool too small");
            return std::make_shared<SharedPhysicalPageAllocator>(phys);
          }(),
          storage_format) {}

KvPagePool::KvPagePool(const MetalContext &context, U32 capacity_tokens,
                       U32 page_tokens,
                       std::shared_ptr<SharedPhysicalPageAllocator> shared_alloc,
                       KvStorageFormat storage_format)
    : impl_(std::make_unique<Impl>()) {
  if (!capacity_tokens || !page_tokens || !shared_alloc)
    throw std::invalid_argument("KV page pool geometry is empty");
  auto &p = *impl_;
  p.format = storage_format;
  p.capacity = capacity_tokens;
  p.page_size = page_tokens;
  const auto logical = pages_for(capacity_tokens, page_tokens);
  if (shared_alloc->physical_pages() < logical ||
      shared_alloc->physical_pages() == UINT32_MAX)
    throw std::invalid_argument("KV physical page pool too small");
  p.alloc = std::move(shared_alloc);
  p.mapping.assign(logical, unmapped);
  p.table = context.make_buffer(U64(logical) * sizeof(U32));
  p.upload(p.mapping);
}

KvPagePool::~KvPagePool() {
  if (!impl_ || !impl_->alloc)
    return;
  auto &p = *impl_;
  std::scoped_lock lock(p.alloc->mutex_);
  if (p.before) {
    for (std::size_t i = 0; i < p.mapping.size(); ++i) {
      if (p.mapping[i] != p.before->mapping[i] && p.mapping[i] != unmapped)
        Impl::release_locked(*p.alloc, p.mapping[i]);
    }
    p.mapping = std::move(p.before->mapping);
    p.before.reset();
  }
  for (U32 slot : p.mapping) {
    if (slot != unmapped && slot < p.alloc->refs_.size() && p.alloc->refs_[slot])
      Impl::release_locked(*p.alloc, slot);
  }
  for (auto &[id, lease] : p.leases) {
    (void)id;
    for (U32 slot : lease.pages) {
      if (slot < p.alloc->refs_.size() && p.alloc->refs_[slot])
        Impl::release_locked(*p.alloc, slot);
    }
  }
}

KvPagePool::KvPagePool(KvPagePool &&) noexcept = default;
KvPagePool &KvPagePool::operator=(KvPagePool &&) noexcept = default;
std::shared_ptr<SharedPhysicalPageAllocator>
KvPagePool::shared_allocator() const noexcept {
  return impl_->alloc;
}
KvStorageFormat KvPagePool::storage_format() const noexcept {
  return impl_->format;
}
U32 KvPagePool::tokens() const { return impl_->frontier; }
U32 KvPagePool::logical_pages() const { return U32(impl_->mapping.size()); }
U32 KvPagePool::physical_pages() const {
  return impl_->alloc->physical_pages();
}
U32 KvPagePool::free_pages() const { return impl_->alloc->free_pages(); }
U32 KvPagePool::page_tokens() const { return impl_->page_size; }
U32 KvPagePool::mapped_pages() const {
  return U32(std::count_if(impl_->mapping.begin(), impl_->mapping.end(),
                           [](U32 page) { return page != unmapped; }));
}
U32 KvPagePool::physical_slot(U32 logical) const {
  if (logical >= impl_->mapping.size() || impl_->mapping[logical] == unmapped)
    throw std::out_of_range("KV logical page is unmapped");
  return impl_->mapping[logical];
}
const MetalBuffer &KvPagePool::table_buffer() const { return impl_->table; }

std::vector<KvPageCopy> KvPagePool::begin_append(U32 count) {
  auto &p = *impl_;
  if (p.before || !count || count > p.capacity - p.frontier)
    throw std::invalid_argument("KV append outside available frontier");
  auto next_mapping = p.mapping;
  std::vector<KvPageCopy> copies;
  U32 cow_old = unmapped;
  {
    std::scoped_lock lock(p.alloc->mutex_);
    const U32 target_pages = pages_for(p.frontier + count, p.page_size);
    bool need_cow = false;
    if (p.frontier % p.page_size) {
      const U32 tail = p.frontier / p.page_size;
      const U32 old = next_mapping[tail];
      if (old == unmapped)
        throw std::logic_error("KV committed tail has no page");
      need_cow = (p.alloc->refs_[old] > 1);
    }
    U32 needed = need_cow ? 1U : 0U;
    for (U32 logical = 0; logical < target_pages; ++logical) {
      if (next_mapping[logical] == unmapped)
        ++needed;
    }
    if (p.alloc->free_.size() < needed)
      throw std::overflow_error("KV physical page pool exhausted");
    if (need_cow) {
      const U32 tail = p.frontier / p.page_size;
      cow_old = next_mapping[tail];
      const U32 replacement = Impl::alloc_locked(*p.alloc);
      next_mapping[tail] = replacement;
      copies.push_back({cow_old, replacement});
    }
    for (U32 logical = 0; logical < target_pages; ++logical) {
      if (next_mapping[logical] == unmapped)
        next_mapping[logical] = Impl::alloc_locked(*p.alloc);
    }
  }
  p.upload(next_mapping);
  p.before = std::make_unique<Impl::Before>(
      Impl::Before{p.mapping, p.frontier, cow_old});
  p.mapping = std::move(next_mapping);
  p.pending_width = count;
  return copies;
}

void KvPagePool::commit(U32 count) {
  auto &p = *impl_;
  if (!p.before || count > p.pending_width)
    throw std::invalid_argument("KV commit without reserved append");
  if (!count) {
    abort();
    return;
  }
  auto next_mapping = p.mapping;
  const U32 frontier = p.before->tokens + count;
  {
    std::scoped_lock lock(p.alloc->mutex_);
    if (p.before->cow_old_slot != unmapped) {
      Impl::release_locked(*p.alloc, p.before->cow_old_slot);
    }
    for (U32 logical = pages_for(frontier, p.page_size);
         logical < next_mapping.size(); ++logical) {
      if (next_mapping[logical] != unmapped) {
        Impl::release_locked(*p.alloc, next_mapping[logical]);
        next_mapping[logical] = unmapped;
      }
    }
  }
  p.upload(next_mapping);
  p.mapping = std::move(next_mapping);
  p.frontier = frontier;
  p.pending_width = 0;
  p.before.reset();
}

void KvPagePool::abort() {
  auto &p = *impl_;
  if (!p.before)
    throw std::logic_error("KV abort without reserved append");
  {
    std::scoped_lock lock(p.alloc->mutex_);
    for (std::size_t i = 0; i < p.mapping.size(); ++i) {
      if (p.mapping[i] != p.before->mapping[i] && p.mapping[i] != unmapped)
        Impl::release_locked(*p.alloc, p.mapping[i]);
    }
  }
  p.upload(p.before->mapping);
  p.mapping = std::move(p.before->mapping);
  p.frontier = p.before->tokens;
  p.pending_width = 0;
  p.before.reset();
}

void KvPagePool::truncate(U32 count) {
  auto &p = *impl_;
  if (p.before || count > p.frontier)
    throw std::invalid_argument("KV truncate outside committed frontier");
  auto next_mapping = p.mapping;
  {
    std::scoped_lock lock(p.alloc->mutex_);
    for (U32 logical = pages_for(count, p.page_size);
         logical < next_mapping.size(); ++logical) {
      if (next_mapping[logical] != unmapped) {
        Impl::release_locked(*p.alloc, next_mapping[logical]);
        next_mapping[logical] = unmapped;
      }
    }
  }
  p.upload(next_mapping);
  p.mapping = std::move(next_mapping);
  p.frontier = count;
}

void KvPagePool::reset() {
  if (impl_->before)
    abort();
  truncate(0);
}

std::uint64_t KvPagePool::retain_prefix() {
  auto &p = *impl_;
  if (p.before)
    throw std::logic_error("KV cannot retain an uncommitted prefix");
  std::vector<U32> pages(p.mapping.begin(),
                         p.mapping.begin() +
                             pages_for(p.frontier, p.page_size));
  for (auto slot : pages)
    if (slot == unmapped)
      throw std::logic_error("KV snapshot has an unmapped committed page");
  if (p.next_lease == UINT64_MAX)
    throw std::overflow_error("KV page lease ID exhausted");
  const auto id = p.next_lease;
  {
    std::scoped_lock lock(p.alloc->mutex_);
    for (auto slot : pages)
      ++p.alloc->refs_[slot];
  }
  auto [entry, inserted] =
      p.leases.emplace(id, Impl::Lease{p.frontier, std::move(pages)});
  if (!inserted)
    throw std::logic_error("KV page lease ID duplicated");
  (void)entry;
  ++p.next_lease;
  return id;
}

void KvPagePool::restore_prefix(std::uint64_t lease) {
  auto &p = *impl_;
  if (p.before)
    throw std::logic_error("KV cannot restore during pending append");
  const auto found = p.leases.find(lease);
  if (found == p.leases.end())
    throw std::out_of_range("KV page lease not found");
  auto next_mapping = std::vector<U32>(p.mapping.size(), unmapped);
  {
    std::scoped_lock lock(p.alloc->mutex_);
    for (auto slot : found->second.pages) {
      if (!p.alloc->refs_[slot])
        throw std::logic_error("KV retained page lost its lease");
      ++p.alloc->refs_[slot];
    }
    for (auto slot : p.mapping) {
      if (slot != unmapped)
        Impl::release_locked(*p.alloc, slot);
    }
  }
  std::copy(found->second.pages.begin(), found->second.pages.end(),
            next_mapping.begin());
  p.upload(next_mapping);
  p.mapping = std::move(next_mapping);
  p.frontier = found->second.tokens;
}

void KvPagePool::release_prefix(std::uint64_t lease) {
  auto &p = *impl_;
  const auto found = p.leases.find(lease);
  if (found == p.leases.end())
    throw std::out_of_range("KV page lease not found");
  {
    std::scoped_lock lock(p.alloc->mutex_);
    for (auto slot : found->second.pages)
      Impl::release_locked(*p.alloc, slot);
  }
  p.leases.erase(found);
}

void KvPagePool::retain_page(U32 physical_slot) {
  auto &p = *impl_;
  if (p.before)
    throw std::logic_error("KV cannot retain page during pending append");
  std::scoped_lock lock(p.alloc->mutex_);
  if (physical_slot >= p.alloc->physical_)
    throw std::out_of_range("KV retain_page slot outside pool");
  if (p.alloc->refs_[physical_slot] == 0)
    throw std::logic_error("KV cannot retain an unallocated physical page");
  if (p.alloc->refs_[physical_slot] == UINT32_MAX)
    throw std::overflow_error("KV page reference count overflow");
  ++p.alloc->refs_[physical_slot];
}

void KvPagePool::release_page(U32 physical_slot) {
  auto &p = *impl_;
  if (p.before)
    throw std::logic_error("KV cannot release page during pending append");
  std::scoped_lock lock(p.alloc->mutex_);
  if (physical_slot >= p.alloc->physical_)
    throw std::out_of_range("KV release_page slot outside pool");
  Impl::release_locked(*p.alloc, physical_slot);
}

U32 KvPagePool::ref_count(U32 physical_slot) const {
  return impl_->alloc->ref_count(physical_slot);
}

std::optional<KvPageCopy> KvPagePool::cow_page_if_shared(U32 logical_page) {
  auto &p = *impl_;
  if (p.before)
    throw std::logic_error("KV cannot COW page during pending append");
  if (logical_page >= p.mapping.size() || p.mapping[logical_page] == unmapped)
    throw std::out_of_range("KV logical page is unmapped for COW");
  const U32 old_slot = p.mapping[logical_page];
  U32 replacement = unmapped;
  {
    std::scoped_lock lock(p.alloc->mutex_);
    if (p.alloc->refs_[old_slot] <= 1)
      return std::nullopt;
    replacement = Impl::alloc_locked(*p.alloc);
    Impl::release_locked(*p.alloc, old_slot);
  }
  p.mapping[logical_page] = replacement;
  p.upload(p.mapping);
  return KvPageCopy{old_slot, replacement};
}

std::vector<U32> KvPagePool::mapped_slots() const {
  const auto &p = *impl_;
  const U32 count = pages_for(p.frontier, p.page_size);
  return std::vector<U32>(p.mapping.begin(), p.mapping.begin() + count);
}

void KvPagePool::install_shared_prefix(U32 token_count,
                                       std::span<const U32> physical_slots) {
  auto &p = *impl_;
  if (p.before)
    throw std::logic_error("KV cannot install shared prefix during append");
  if (token_count > p.capacity ||
      physical_slots.size() != pages_for(token_count, p.page_size))
    throw std::invalid_argument("KV shared prefix page count mismatch");
  auto next_mapping = std::vector<U32>(p.mapping.size(), unmapped);
  {
    std::scoped_lock lock(p.alloc->mutex_);
    for (U32 slot : physical_slots) {
      if (slot >= p.alloc->physical_ || p.alloc->refs_[slot] == 0)
        throw std::invalid_argument("KV shared prefix references invalid page");
    }
    for (U32 slot : physical_slots) {
      if (p.alloc->refs_[slot] == UINT32_MAX)
        throw std::overflow_error("KV page reference count overflow");
      ++p.alloc->refs_[slot];
    }
    for (U32 slot : p.mapping) {
      if (slot != unmapped)
        Impl::release_locked(*p.alloc, slot);
    }
  }
  std::copy(physical_slots.begin(), physical_slots.end(), next_mapping.begin());
  p.upload(next_mapping);
  p.mapping = std::move(next_mapping);
  p.frontier = token_count;
}

class KvPageCopier::Impl {
public:
  const MetalContext &context;
  U32 physical, values_per_page;
  detail::forward::Profile profile;
  Impl(const MetalContext &ctx, const MetalBuffer &keys,
       const MetalBuffer &values, U32 words16_per_page, U32 physical_pages,
       const std::filesystem::path &libraries, std::string key)
      : context(ctx), physical(physical_pages),
        values_per_page(words16_per_page) {
    if (key.empty() || !words16_per_page || physical_pages < 2 ||
        U64(physical_pages) * values_per_page * 2 > keys.size() ||
        U64(physical_pages) * values_per_page * 2 > values.size())
      throw std::invalid_argument("KV copy resource geometry invalid");
    detail::forward::Encoder e(ctx, libraries, 1, "p05.kv.copy." + key);
    auto &d = e.emit("attention_postprocess", "omlx_m5_kv_page_copy",
                     {.width = values_per_page}, {.width = 128},
                     {{&keys, 0, keys.size()}, {&values, 0, values.size()}});
    e.profile.sampling_offset =
        e.dynamic(d, 2, detail::forward::words({0, 0, values_per_page}));
    profile = e.finish();
  }
};

KvPageCopier::KvPageCopier(const MetalContext &context, const MetalBuffer &keys,
                           const MetalBuffer &values, U32 page_tokens,
                           U32 kv_width, U32 physical_pages,
                           const std::filesystem::path &libraries,
                           std::string key)
    : impl_(std::make_unique<Impl>(
          context, keys, values,
          [&]() -> U32 {
            if (!page_tokens || !kv_width ||
                U64(page_tokens) * kv_width > UINT32_MAX)
              throw std::invalid_argument("KV copy resource geometry invalid");
            return U32(U64(page_tokens) * kv_width);
          }(),
          physical_pages, libraries, std::move(key))) {}

KvPageCopier::KvPageCopier(const MetalContext &context, const MetalBuffer &keys,
                           const MetalBuffer &values, KvStorageFormat format,
                           U32 page_tokens, U32 kv_heads, U32 head_dim,
                           U32 physical_pages,
                           const std::filesystem::path &libraries,
                           std::string key)
    : impl_(std::make_unique<Impl>(
          context, keys, values,
          kv_copy_words16_per_page(format, page_tokens, kv_heads, head_dim),
          physical_pages, libraries, std::move(key))) {}

KvPageCopier::~KvPageCopier() = default;
KvPageCopier::KvPageCopier(KvPageCopier &&) noexcept = default;
KvPageCopier &KvPageCopier::operator=(KvPageCopier &&) noexcept = default;

MetalSubmission KvPageCopier::execute(KvPageCopy move) {
  auto &p = *impl_;
  if (move.source >= p.physical || move.destination >= p.physical ||
      move.source == move.destination)
    throw std::invalid_argument("KV copy page ID outside pool");
  const std::array<U32, 3> params{move.source, move.destination,
                                  p.values_per_page};
  std::memcpy(p.profile.parameter_data.data() + p.profile.sampling_offset,
              params.data(), sizeof(params));
  p.profile.parameters.write("step", p.profile.parameter_data);
  const std::array invocation = {
      MetalCommandInvocation{.plan = &p.profile.plan,
                             .bindings = p.profile.bindings,
                             .parameters = &p.profile.parameters}};
  return p.context.execute_batch(invocation);
}
} // namespace omlx::m5
