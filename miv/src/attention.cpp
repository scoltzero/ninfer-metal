#include "omlx_m5/attention.h"
#include "omlx_m5/kv_page_pool.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace omlx::m5 {
namespace {

struct AttentionPostprocessParams {
  std::uint32_t query_heads = 0;
  std::uint32_t key_value_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t rotary_dim = 0;
  std::uint32_t tokens = 0;
  std::uint32_t query_token_stride = 0;
  std::uint32_t query_head_stride = 0;
  std::uint32_t key_token_stride = 0;
  std::uint32_t key_head_stride = 0;
  std::uint32_t start_position = 0;
  float epsilon = 0.0F;
  float theta = 0.0F;
  float weight_offset = 0.0F;
};

struct PagedAttentionParams {
  std::uint32_t query_heads = 0;
  std::uint32_t key_value_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t tokens = 0;
  std::uint32_t cache_start = 0;
  std::uint32_t cache_capacity = 0;
  std::uint32_t page_tokens = 0;
  std::uint32_t page_count = 0;
  std::uint32_t query_token_stride = 0;
  std::uint32_t query_head_stride = 0;
  std::uint32_t key_token_stride = 0;
  std::uint32_t value_token_stride = 0;
  std::uint32_t gate_token_stride = 0;
  std::uint32_t gate_head_stride = 0;
};

static_assert(sizeof(AttentionPostprocessParams) == 52);
static_assert(sizeof(PagedAttentionParams) == 56);

std::uint32_t checked_u32(std::uint64_t value, const char *label) {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error(std::string(label) + " exceeds uint32");
  }
  return static_cast<std::uint32_t>(value);
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               const char *label) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error(std::string(label) + " overflow");
  }
  return left * right;
}

bool token_domain(std::uint32_t tokens) {
  return tokens >= 1 && (tokens <= 128 || tokens == 512 || tokens == 1024);
}

std::uint64_t strided_span(std::uint32_t tokens,
                           std::uint64_t token_stride_bytes,
                           std::uint64_t heads, std::uint64_t head_stride_bytes,
                           std::uint64_t head_bytes) {
  return checked_multiply(tokens - 1, token_stride_bytes,
                          "attention token span") +
         checked_multiply(heads - 1, head_stride_bytes, "attention head span") +
         head_bytes;
}

} // namespace

class AttentionPostprocessPlan::Impl {
public:
  AttentionPostprocessInfo info;
  AttentionPostprocessParams params;
  MetalCommandPlan command_plan;
  MetalBuffer params_buffer;
};

AttentionPostprocessPlan::AttentionPostprocessPlan() = default;
AttentionPostprocessPlan::AttentionPostprocessPlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
AttentionPostprocessPlan::AttentionPostprocessPlan(
    AttentionPostprocessPlan &&) noexcept = default;
AttentionPostprocessPlan &AttentionPostprocessPlan::operator=(
    AttentionPostprocessPlan &&) noexcept = default;
AttentionPostprocessPlan::~AttentionPostprocessPlan() = default;

bool AttentionPostprocessPlan::valid() const { return impl_ != nullptr; }

const AttentionPostprocessInfo &AttentionPostprocessPlan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("attention postprocess plan is empty");
  }
  return impl_->info;
}

AttentionPostprocessPlan prepare_attention_postprocess(
    const MetalContext &context, const std::filesystem::path &metallib,
    const QwenModelConfig &config, std::uint32_t tokens) {
  const auto head_dim = checked_u32(config.head_dim, "attention head dim");
  const auto query_heads =
      checked_u32(config.num_attention_heads, "attention query heads");
  const auto key_value_heads =
      checked_u32(config.num_key_value_heads, "attention KV heads");
  const auto rotary_double =
      static_cast<double>(config.head_dim) * config.partial_rotary_factor;
  const auto rotary_dim =
      static_cast<std::uint32_t>(std::llround(rotary_double));
  if (!token_domain(tokens) || head_dim == 0 || query_heads == 0 ||
      key_value_heads == 0 || rotary_dim == 0 || rotary_dim > head_dim ||
      rotary_dim % 2 != 0 || std::abs(rotary_double - rotary_dim) > 1.0e-9 ||
      !std::isfinite(config.rms_norm_eps) || config.rms_norm_eps <= 0.0 ||
      !std::isfinite(config.rope_theta) || config.rope_theta <= 0.0) {
    throw std::invalid_argument("attention postprocess domain is invalid");
  }
  const auto query_head_bytes = checked_multiply(head_dim, 2, "query head");
  const auto query_head_stride =
      checked_multiply(head_dim, 4, "query/gate head stride");
  const auto query_token_stride = checked_multiply(
      query_heads, query_head_stride, "query/gate token stride");
  const auto key_head_stride = checked_multiply(head_dim, 2, "key head stride");
  const auto key_token_stride =
      checked_multiply(key_value_heads, key_head_stride, "key token stride");
  auto impl = std::make_unique<AttentionPostprocessPlan::Impl>();
  impl->info = {
      .executor_key =
          "attention.qk_norm_rope.h" + std::to_string(query_heads) + ".kv" +
          std::to_string(key_value_heads) + ".d" + std::to_string(head_dim) +
          ".r" + std::to_string(rotary_dim) + ".t" + std::to_string(tokens),
      .tokens = tokens,
      .query_heads = query_heads,
      .key_value_heads = key_value_heads,
      .head_dim = head_dim,
      .rotary_dim = rotary_dim,
      .epsilon = static_cast<float>(config.rms_norm_eps),
      .theta = config.rope_theta,
      .query_span_bytes = strided_span(tokens, query_token_stride, query_heads,
                                       query_head_stride, query_head_bytes),
      .key_span_bytes = strided_span(tokens, key_token_stride, key_value_heads,
                                     key_head_stride, key_head_stride),
      .norm_weight_bytes = key_head_stride,
  };
  impl->params = {
      .query_heads = query_heads,
      .key_value_heads = key_value_heads,
      .head_dim = head_dim,
      .rotary_dim = rotary_dim,
      .tokens = tokens,
      .query_token_stride = checked_u32(query_token_stride / 2, "query stride"),
      .query_head_stride =
          checked_u32(query_head_stride / 2, "query head stride"),
      .key_token_stride = checked_u32(key_token_stride / 2, "key stride"),
      .key_head_stride = head_dim,
      .start_position = 0,
      .epsilon = static_cast<float>(config.rms_norm_eps),
      .theta = static_cast<float>(config.rope_theta),
      .weight_offset = 0.0F,
  };
  context.register_library({
      .key = "b3.attention_postprocess.library",
      .path = metallib,
  });
  static_cast<void>(context.warmup_pipeline({
      .key = "b3.attention_postprocess.qk_norm_rope",
      .library_key = "b3.attention_postprocess.library",
      .function_name = "omlx_m5_attention_qk_norm_rope",
      .variant = "half_split_bf16",
  }));
  impl->params_buffer = context.make_buffer(sizeof(AttentionPostprocessParams));
  impl->command_plan = context.prepare_command_plan({
      .key = impl->info.executor_key,
      .dispatches = {{
          .pipeline_key = "b3.attention_postprocess.qk_norm_rope",
          .grid = {.width = 32,
                   .height = tokens * (query_heads + key_value_heads)},
          .threads_per_threadgroup = {.width = 32},
          .resources =
              {
                  {.argument_index = 0,
                   .resource_slot = 0,
                   .minimum_bytes = impl->info.query_span_bytes},
                  {.argument_index = 1,
                   .resource_slot = 1,
                   .minimum_bytes = impl->info.key_span_bytes},
                  {.argument_index = 2,
                   .resource_slot = 2,
                   .minimum_bytes = impl->info.norm_weight_bytes},
                  {.argument_index = 3,
                   .resource_slot = 3,
                   .minimum_bytes = impl->info.norm_weight_bytes},
                  {.argument_index = 4,
                   .resource_slot = 4,
                   .minimum_bytes = sizeof(AttentionPostprocessParams)},
              },
      }},
  });
  return AttentionPostprocessPlan(std::move(impl));
}

MetalSubmission execute_attention_postprocess(
    const MetalContext &context, const AttentionPostprocessPlan &plan,
    const ProjectionSplitOutputView &query,
    const ProjectionSplitOutputView &key, const MetalBuffer &query_norm,
    std::uint64_t query_norm_offset, const MetalBuffer &key_norm,
    std::uint64_t key_norm_offset, std::uint64_t start_position) {
  if (plan.impl_ == nullptr || query.buffer == nullptr ||
      key.buffer == nullptr || query.tokens != plan.impl_->info.tokens ||
      key.tokens != plan.impl_->info.tokens ||
      query.features !=
          plan.impl_->info.query_heads * plan.impl_->info.head_dim ||
      key.features !=
          plan.impl_->info.key_value_heads * plan.impl_->info.head_dim ||
      query.block_count != plan.impl_->info.query_heads ||
      query.block_features != plan.impl_->info.head_dim ||
      query.block_stride_bytes != plan.impl_->info.head_dim * 4ULL ||
      query.token_stride_bytes !=
          plan.impl_->info.query_heads * plan.impl_->info.head_dim * 4ULL ||
      !key.contiguous_per_token() ||
      key.token_stride_bytes !=
          plan.impl_->info.key_value_heads * plan.impl_->info.head_dim * 2ULL ||
      query.span_bytes < plan.impl_->info.query_span_bytes ||
      key.span_bytes < plan.impl_->info.key_span_bytes ||
      query_norm_offset % 2 != 0 || key_norm_offset % 2 != 0 ||
      start_position > std::numeric_limits<std::uint32_t>::max() ||
      start_position + plan.impl_->info.tokens - 1 >
          std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("attention postprocess invocation is invalid");
  }
  auto params = plan.impl_->params;
  params.start_position = static_cast<std::uint32_t>(start_position);
  std::vector<std::byte> params_bytes(sizeof(params));
  std::memcpy(params_bytes.data(), &params, sizeof(params));
  plan.impl_->params_buffer.write(params_bytes);
  const std::vector<MetalBufferBinding> bindings = {
      {0, query.buffer, query.offset},     {1, key.buffer, key.offset},
      {2, &query_norm, query_norm_offset}, {3, &key_norm, key_norm_offset},
      {4, &plan.impl_->params_buffer, 0},
  };
  return context.execute(plan.impl_->command_plan, bindings);
}

class PagedKvCache::Impl {
public:
  PagedKvCacheInfo info;
  std::unique_ptr<KvPagePool> pool;
  MetalBuffer keys;
  MetalBuffer values;
};

PagedKvCache::PagedKvCache(const MetalContext &context,
                           std::uint32_t key_value_heads,
                           std::uint32_t head_dim,
                           std::uint32_t capacity_tokens,
                           std::uint32_t page_tokens)
    : impl_(std::make_unique<Impl>()) {
  if (key_value_heads == 0 || head_dim == 0 || capacity_tokens == 0 ||
      page_tokens == 0) {
    throw std::invalid_argument("paged KV cache geometry is invalid");
  }
  const auto page_count =
      (static_cast<std::uint64_t>(capacity_tokens) + page_tokens - 1) /
      page_tokens;
  const auto bytes_per_token = checked_multiply(
      checked_multiply(key_value_heads, head_dim, "KV token values"), 2,
      "KV token bytes");
  const auto bytes_per_page =
      checked_multiply(bytes_per_token, page_tokens, "KV page bytes");
  const auto physical_pages = page_count + 1;
  const auto total_bytes =
      checked_multiply(physical_pages, bytes_per_page, "KV cache bytes");
  impl_->info = {
      .key_value_heads = key_value_heads,
      .head_dim = head_dim,
      .capacity_tokens = capacity_tokens,
      .page_tokens = page_tokens,
      .page_count = checked_u32(page_count, "KV page count"),
      .physical_page_count = checked_u32(physical_pages, "KV physical pages"),
      .bytes_per_token = bytes_per_token,
      .bytes_per_page = bytes_per_page,
      .key_bytes = total_bytes,
      .value_bytes = total_bytes,
  };
  impl_->keys = context.make_buffer(total_bytes);
  impl_->values = context.make_buffer(total_bytes);
  impl_->pool = std::make_unique<KvPagePool>(
      context, capacity_tokens, page_tokens, impl_->info.physical_page_count);
}

PagedKvCache::PagedKvCache(PagedKvCache &&) noexcept = default;
PagedKvCache &PagedKvCache::operator=(PagedKvCache &&) noexcept = default;
PagedKvCache::~PagedKvCache() = default;

const PagedKvCacheInfo &PagedKvCache::info() const { return impl_->info; }
std::uint32_t PagedKvCache::tokens() const { return impl_->pool->tokens(); }

void PagedKvCache::truncate(std::uint32_t tokens) {
  if (tokens > impl_->pool->tokens())
    throw std::out_of_range("paged KV truncate exceeds current tokens");
  impl_->pool->truncate(tokens);
}

void PagedKvCache::reset() { impl_->pool->reset(); }
const MetalBuffer &PagedKvCache::keys() const { return impl_->keys; }
const MetalBuffer &PagedKvCache::values() const { return impl_->values; }
const MetalBuffer &PagedKvCache::page_table() const {
  return impl_->pool->table_buffer();
}

class PagedAttentionPlan::Impl {
public:
  PagedAttentionInfo info;
  PagedAttentionParams params;
  MetalCommandPlan command_plan;
  MetalBuffer params_buffer;
};

PagedAttentionPlan::PagedAttentionPlan() = default;
PagedAttentionPlan::PagedAttentionPlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PagedAttentionPlan::PagedAttentionPlan(PagedAttentionPlan &&) noexcept =
    default;
PagedAttentionPlan &
PagedAttentionPlan::operator=(PagedAttentionPlan &&) noexcept = default;
PagedAttentionPlan::~PagedAttentionPlan() = default;

bool PagedAttentionPlan::valid() const { return impl_ != nullptr; }

const PagedAttentionInfo &PagedAttentionPlan::info() const {
  if (impl_ == nullptr) {
    throw std::logic_error("paged attention plan is empty");
  }
  return impl_->info;
}

PagedAttentionPlan
prepare_paged_attention(const MetalContext &context,
                        const std::filesystem::path &metallib,
                        const QwenModelConfig &config, std::uint32_t tokens,
                        const PagedKvCacheInfo &cache) {
  const auto query_heads =
      checked_u32(config.num_attention_heads, "paged query heads");
  const auto key_value_heads =
      checked_u32(config.num_key_value_heads, "paged KV heads");
  const auto head_dim = checked_u32(config.head_dim, "paged head dim");
  if (!token_domain(tokens) || query_heads == 0 || key_value_heads == 0 ||
      query_heads % key_value_heads != 0 || head_dim == 0 || head_dim > 256 ||
      cache.key_value_heads != key_value_heads || cache.head_dim != head_dim ||
      cache.capacity_tokens == 0 || cache.page_tokens == 0 ||
      cache.page_count == 0 || cache.physical_page_count == 0) {
    throw std::invalid_argument("paged attention domain is invalid");
  }
  const auto query_head_bytes = checked_multiply(head_dim, 2, "paged query");
  const auto query_head_stride =
      checked_multiply(head_dim, 4, "paged query/gate stride");
  const auto query_token_stride = checked_multiply(
      query_heads, query_head_stride, "paged query token stride");
  const auto kv_token_stride = checked_multiply(
      checked_multiply(key_value_heads, head_dim, "paged KV token"), 2,
      "paged KV token bytes");
  auto impl = std::make_unique<PagedAttentionPlan::Impl>();
  impl->info = {
      .executor_key = "attention.paged.h" + std::to_string(query_heads) +
                      ".kv" + std::to_string(key_value_heads) + ".d" +
                      std::to_string(head_dim) + ".t" + std::to_string(tokens) +
                      ".p" + std::to_string(cache.page_tokens),
      .tokens = tokens,
      .query_heads = query_heads,
      .key_value_heads = key_value_heads,
      .head_dim = head_dim,
      .query_span_bytes = strided_span(tokens, query_token_stride, query_heads,
                                       query_head_stride, query_head_bytes),
      .key_span_bytes = checked_multiply(tokens, kv_token_stride, "paged key"),
      .value_span_bytes =
          checked_multiply(tokens, kv_token_stride, "paged value"),
      .gate_span_bytes = strided_span(tokens, query_token_stride, query_heads,
                                      query_head_stride, query_head_bytes),
      .output_bytes = checked_multiply(
          checked_multiply(
              tokens,
              checked_multiply(query_heads, head_dim, "paged output width"),
              "paged output values"),
          2, "paged output bytes"),
  };
  impl->params = {
      .query_heads = query_heads,
      .key_value_heads = key_value_heads,
      .head_dim = head_dim,
      .tokens = tokens,
      .cache_capacity = cache.capacity_tokens,
      .page_tokens = cache.page_tokens,
      .page_count = cache.physical_page_count,
      .query_token_stride = checked_u32(query_token_stride / 2, "query stride"),
      .query_head_stride = checked_u32(query_head_stride / 2, "query head"),
      .key_token_stride = checked_u32(kv_token_stride / 2, "key stride"),
      .value_token_stride = checked_u32(kv_token_stride / 2, "value stride"),
      .gate_token_stride = checked_u32(query_token_stride / 2, "gate stride"),
      .gate_head_stride = checked_u32(query_head_stride / 2, "gate head"),
  };
  context.register_library({
      .key = "b3.attention_postprocess.library",
      .path = metallib,
  });
  static_cast<void>(context.warmup_pipeline({
      .key = "b3.attention.paged_append",
      .library_key = "b3.attention_postprocess.library",
      .function_name = "omlx_m5_paged_kv_append",
      .variant = "bf16_page_major",
  }));
  static_cast<void>(context.warmup_pipeline({
      .key = "b3.attention.paged_causal",
      .library_key = "b3.attention_postprocess.library",
      .function_name = "omlx_m5_paged_causal_attention",
      .variant = "bf16_gqa",
  }));
  impl->params_buffer = context.make_buffer(sizeof(PagedAttentionParams));
  const auto kv_values = checked_multiply(
      tokens, checked_multiply(key_value_heads, head_dim, "append width"),
      "append values");
  impl->command_plan = context.prepare_command_plan({
      .key = impl->info.executor_key,
      .dispatches =
          {
              {
                  .pipeline_key = "b3.attention.paged_append",
                  .grid = {.width = kv_values},
                  .threads_per_threadgroup = {.width = 128},
                  .resources =
                      {
                          {.argument_index = 0,
                           .resource_slot = 1,
                           .minimum_bytes = impl->info.key_span_bytes},
                          {.argument_index = 1,
                           .resource_slot = 2,
                           .minimum_bytes = impl->info.value_span_bytes},
                          {.argument_index = 2,
                           .resource_slot = 4,
                           .minimum_bytes = cache.key_bytes},
                          {.argument_index = 3,
                           .resource_slot = 5,
                           .minimum_bytes = cache.value_bytes},
                          {.argument_index = 4,
                           .resource_slot = 8,
                           .minimum_bytes = cache.page_count * 4ULL},
                          {.argument_index = 5,
                           .resource_slot = 7,
                           .minimum_bytes = sizeof(PagedAttentionParams)},
                      },
              },
              {
                  .pipeline_key = "b3.attention.paged_causal",
                  .grid = {.width = 32, .height = tokens * query_heads},
                  .threads_per_threadgroup = {.width = 32},
                  .resources =
                      {
                          {.argument_index = 0,
                           .resource_slot = 0,
                           .minimum_bytes = impl->info.query_span_bytes},
                          {.argument_index = 1,
                           .resource_slot = 3,
                           .minimum_bytes = impl->info.gate_span_bytes},
                          {.argument_index = 2,
                           .resource_slot = 4,
                           .minimum_bytes = cache.key_bytes},
                          {.argument_index = 3,
                           .resource_slot = 5,
                           .minimum_bytes = cache.value_bytes},
                          {.argument_index = 4,
                           .resource_slot = 6,
                           .minimum_bytes = impl->info.output_bytes},
                          {.argument_index = 5,
                           .resource_slot = 8,
                           .minimum_bytes = cache.page_count * 4ULL},
                          {.argument_index = 6,
                           .resource_slot = 7,
                           .minimum_bytes = sizeof(PagedAttentionParams)},
                      },
              },
          },
  });
  return PagedAttentionPlan(std::move(impl));
}

MetalSubmission execute_paged_attention(const MetalContext &context,
                                        const PagedAttentionPlan &plan,
                                        const ProjectionSplitOutputView &query,
                                        const ProjectionSplitOutputView &key,
                                        const ProjectionSplitOutputView &value,
                                        const ProjectionSplitOutputView &gate,
                                        PagedKvCache &cache,
                                        MetalBuffer &output,
                                        std::uint64_t output_offset) {
  if (plan.impl_ == nullptr || cache.impl_ == nullptr ||
      query.buffer == nullptr || key.buffer == nullptr ||
      value.buffer == nullptr || gate.buffer == nullptr ||
      query.offset % 2 != 0 || key.offset % 2 != 0 ||
      value.offset % 2 != 0 || gate.offset % 2 != 0 ||
      output_offset % 2 != 0 || query.tokens != plan.impl_->info.tokens ||
      key.tokens != plan.impl_->info.tokens ||
      value.tokens != plan.impl_->info.tokens ||
      gate.tokens != plan.impl_->info.tokens ||
      query.features !=
          plan.impl_->info.query_heads * plan.impl_->info.head_dim ||
      gate.features != query.features ||
      key.features !=
          plan.impl_->info.key_value_heads * plan.impl_->info.head_dim ||
      value.features != key.features ||
      query.block_count != plan.impl_->info.query_heads ||
      gate.block_count != plan.impl_->info.query_heads ||
      query.block_features != plan.impl_->info.head_dim ||
      gate.block_features != plan.impl_->info.head_dim ||
      query.block_stride_bytes != plan.impl_->info.head_dim * 4ULL ||
      gate.block_stride_bytes != plan.impl_->info.head_dim * 4ULL ||
      query.token_stride_bytes !=
          static_cast<std::uint64_t>(plan.impl_->params.query_token_stride) *
              2ULL ||
      gate.token_stride_bytes !=
          static_cast<std::uint64_t>(plan.impl_->params.gate_token_stride) *
              2ULL ||
      !key.contiguous_per_token() || !value.contiguous_per_token() ||
      key.token_stride_bytes !=
          static_cast<std::uint64_t>(plan.impl_->params.key_token_stride) *
              2ULL ||
      value.token_stride_bytes !=
          static_cast<std::uint64_t>(plan.impl_->params.value_token_stride) *
              2ULL ||
      query.span_bytes < plan.impl_->info.query_span_bytes ||
      key.span_bytes < plan.impl_->info.key_span_bytes ||
      value.span_bytes < plan.impl_->info.value_span_bytes ||
      gate.span_bytes < plan.impl_->info.gate_span_bytes ||
      cache.impl_->info.key_value_heads != plan.impl_->info.key_value_heads ||
      cache.impl_->info.head_dim != plan.impl_->info.head_dim ||
      cache.impl_->info.capacity_tokens != plan.impl_->params.cache_capacity ||
      cache.impl_->info.page_tokens != plan.impl_->params.page_tokens ||
      cache.impl_->info.physical_page_count != plan.impl_->params.page_count ||
      static_cast<std::uint64_t>(cache.impl_->pool->tokens()) +
              plan.impl_->info.tokens >
          cache.impl_->info.capacity_tokens) {
    throw std::invalid_argument("paged attention invocation is invalid");
  }
  auto params = plan.impl_->params;
  params.cache_start = cache.impl_->pool->tokens();
  std::vector<std::byte> params_bytes(sizeof(params));
  std::memcpy(params_bytes.data(), &params, sizeof(params));
  plan.impl_->params_buffer.write(params_bytes);
  const std::vector<MetalBufferBinding> bindings = {
      {0, query.buffer, query.offset},
      {1, key.buffer, key.offset},
      {2, value.buffer, value.offset},
      {3, gate.buffer, gate.offset},
      {4, &cache.impl_->keys, 0},
      {5, &cache.impl_->values, 0},
      {6, &output, output_offset},
      {7, &plan.impl_->params_buffer, 0},
      {8, &cache.impl_->pool->table_buffer(), 0},
  };
  const auto copies = cache.impl_->pool->begin_append(plan.impl_->info.tokens);
  if (!copies.empty()) {
    cache.impl_->pool->abort();
    throw std::logic_error(
        "standalone KV cache has no retained-page COW owner");
  }
  try {
    auto submission = context.execute(plan.impl_->command_plan, bindings);
    cache.impl_->pool->commit(plan.impl_->info.tokens);
    return submission;
  } catch (...) {
    cache.impl_->pool->abort();
    throw;
  }
}

} // namespace omlx::m5
