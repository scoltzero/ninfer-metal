#include "omlx_m5/qwen_memory.h"
#include "omlx_m5/kv_page_pool.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace omlx::m5 {
namespace {
using U32 = std::uint32_t;
using U64 = std::uint64_t;

U64 add(U64 a, U64 b) {
  if (b > std::numeric_limits<U64>::max() - a)
    throw std::overflow_error("Qwen memory size overflow");
  return a + b;
}
U64 multiply(U64 a, U64 b) {
  if (a && b > std::numeric_limits<U64>::max() / a)
    throw std::overflow_error("Qwen memory geometry overflow");
  return a * b;
}
U64 positive(std::int32_t value) {
  if (value <= 0)
    throw std::invalid_argument("Qwen memory dimension must be positive");
  return U64(value);
}

U32 resolve_feature_ring_rows(U32 max_context, U32 requested_ring_rows) {
  if (requested_ring_rows != 0) {
    if (requested_ring_rows > (1U << 30))
      throw std::invalid_argument("Qwen feature_ring_rows overflow");
    return std::max<U32>(std::bit_ceil(requested_ring_rows), 64U);
  }
  const U32 clamped = std::min<U32>(max_context, 4096U);
  return std::max<U32>(std::bit_ceil(clamped), 64U);
}
} // namespace

U64 QwenForwardMemoryPlan::buffer_bytes() const {
  return add(workspace_bytes, state_bytes);
}

QwenForwardMemoryPlan
plan_qwen_forward_memory(const QwenModelConfig &c, U32 max_context,
                         U32 feature_count, bool inspect_routes,
                         bool speculative, bool split_attention,
                         bool position_sampling, bool exact_attention,
                         bool inspect_tensors, U32 feature_ring_rows,
                         U32 workspace_rows, bool affine4_kv,
                         U32 kv_pool_pages, bool flash_nax_prefill,
                         bool enable_w4a8_workspace,
                         bool enable_minference_workspace) {
  const U32 max_ws_rows = c.num_experts > 1 ? 4096U : 2048U;
  if (max_context == 0 || workspace_rows == 0 || workspace_rows > max_ws_rows ||
      c.num_hidden_layers <= 0 ||
      c.layer_types.size() != std::size_t(c.num_hidden_layers) ||
      feature_count > U32(c.num_hidden_layers) || c.linear_conv_kernel_dim <= 1)
    throw std::invalid_argument(
        "Qwen memory context/layer geometry is invalid");
  const U64 h = positive(c.hidden_size), f = positive(c.intermediate_size);
  const U64 v = positive(c.vocab_size);
  const U64 q = multiply(positive(c.num_attention_heads), positive(c.head_dim));
  const U64 kv =
      multiply(positive(c.num_key_value_heads), positive(c.head_dim));
  const U64 gk = multiply(positive(c.linear_num_key_heads),
                          positive(c.linear_key_head_dim));
  const U64 gv = multiply(positive(c.linear_num_value_heads),
                          positive(c.linear_value_head_dim));
  const U64 conv = add(multiply(gk, 2), gv);
  const U64 shared_f =
      c.num_experts ? positive(c.shared_expert_intermediate_size) : 0;
  const U64 assignments = c.num_experts ? positive(c.num_experts_per_token) : 1;
  const U64 feature_width = feature_count ? multiply(h, feature_count) : 0;
  const U64 width =
      std::max({h, f, q * 2, kv, conv, gv, shared_f, feature_width});
  const U32 ring_rows = (speculative && feature_count)
                            ? resolve_feature_ring_rows(max_context,
                                                        feature_ring_rows)
                            : 0U;
  // Enforce element-count bounds across per-chunk workspaces and cyclic feature_history.
  if (width > U64(std::numeric_limits<std::int32_t>::max()) ||
      multiply(std::max(multiply(width, assignments), v), workspace_rows) >
          UINT32_MAX ||
      (ring_rows && multiply(feature_width, ring_rows) > UINT32_MAX))
    throw std::overflow_error("Qwen execution exceeds Metal element indexing");

  QwenForwardMemoryPlan plan;
  plan.feature_ring_rows = ring_rows;
  plan.workspace_rows = workspace_rows;
  const auto append = [&](std::string name, U64 bytes, bool state = false,
                          bool clear_on_reset = false) {
    if (!bytes)
      throw std::invalid_argument("empty Qwen allocation");
    auto &total = state ? plan.state_bytes : plan.workspace_bytes;
    total = add(total, bytes);
    plan.largest_buffer_bytes = std::max(plan.largest_buffer_bytes, bytes);
    plan.buffers.push_back({std::move(name), bytes, state, clear_on_reset});
  };
  const auto tensor = [&](std::string name, U64 columns, U64 rows = 0,
                          U64 element_bytes = 2) {
    const U64 actual_rows = rows ? rows : U64(workspace_rows);
    append(std::move(name),
           multiply(multiply(actual_rows, columns), element_bytes));
  };
  tensor("token_ids", 1, workspace_rows, 4);
  tensor("hidden", h);
  tensor("norm", h);
  tensor("query", q * 2);
  tensor("key", kv);
  tensor("value", kv);
  tensor("qkv", conv);
  tensor("z", gv);
  tensor("a", positive(c.linear_num_value_heads));
  tensor("b", positive(c.linear_num_value_heads));
  tensor("convolved", conv);
  tensor("recurrent", gv);
  tensor("mixer", std::max(q, gv));
  if (split_attention) {
    const bool strict_two_pass_attn = []() {
      const char *val = std::getenv("OMLX_M5_STRICT_TWO_PASS_ATTN");
      if (!val)
        return false;
      const std::string_view sv(val);
      return sv == "1" || sv == "true" || sv == "TRUE" || sv == "on" ||
             sv == "ON";
    }();
    const bool bf16_two_pass =
        exact_attention && !affine4_kv && strict_two_pass_attn;
    const U64 raw_splits = (U64(max_context) + 511ULL) / 512ULL;
    const U64 max_splits =
        (!affine4_kv && !bf16_two_pass) ? std::min<U64>(256ULL, raw_splits)
                                        : raw_splits;
    const U64 records =
        multiply(multiply(8, positive(c.num_attention_heads)), max_splits);
    append("attention_split_max", multiply(records, 4));
    if (!bf16_two_pass) {
      append("attention_split_sum", multiply(records, 4));
      append("attention_split_output",
             multiply(multiply(records, positive(c.head_dim)), 4));
    }
  }
  if ((exact_attention || affine4_kv || flash_nax_prefill) &&
      c.full_attention_layers() > 0)
    tensor("attention_positions", 1, workspace_rows, 4);
  tensor("delta", h);
  tensor("gate", std::max(multiply(f, assignments), shared_f));
  tensor("up", std::max(multiply(f, assignments), shared_f));
  tensor("product", std::max(multiply(f, assignments), shared_f));
  tensor("sums", (width + 31) / 32, U64(workspace_rows) * assignments, 4);
  if (c.num_experts) {
    tensor("router", positive(c.num_experts));
    tensor("route_ids", assignments, workspace_rows, 4);
    tensor("route_scores", assignments);
    tensor("expert_output", h, U64(workspace_rows) * assignments);
    tensor("routed", h);
    tensor("shared", h);
    tensor("shared_gate", 1);
  }
  const U64 max_logits_rows =
      inspect_tensors ? U64(workspace_rows) : std::min<U64>(workspace_rows, 8ULL);
  tensor("logits", v, max_logits_rows);
  tensor("last_logits", v, 1);
  tensor("sampled", 1, 1, position_sampling ? 8 : 4);
  if (position_sampling) {
    tensor("position_sample_ids", 64, 1, 4);
    tensor("position_sample_scores", 64, 1, 4);
  }
  if (feature_count)
    tensor("features", multiply(h, feature_count));
  if (inspect_routes && c.num_experts) {
    tensor("route_trace", multiply(c.num_hidden_layers, assignments),
           workspace_rows, 4);
    tensor("router_logits_trace", multiply(c.num_hidden_layers, c.num_experts));
    tensor("router_scores_trace", multiply(c.num_hidden_layers, assignments));
  }
  if (ring_rows)
    tensor("feature_history", multiply(h, feature_count), ring_rows);
  if (workspace_rows >= 64U && c.linear_attention_layers() > 0 &&
      c.linear_key_head_dim == 128 && c.linear_value_head_dim == 128) {
    const U64 max_chunks = (U64(workspace_rows) + 63ULL) / 64ULL;
    const U64 qk_bytes =
        multiply(multiply(max_chunks, positive(c.linear_num_key_heads)),
                 2ULL * 64ULL * 128ULL * 2ULL);
    const U64 ctrl_bytes =
        multiply(multiply(max_chunks, positive(c.linear_num_value_heads)),
                 17152ULL);
    append("qwen.workspace.gdn_chunk", add(qk_bytes, ctrl_bytes));
  }
  if (enable_w4a8_workspace) {
    const U64 a8_mp = ((U64(workspace_rows) + 63ULL) / 64ULL) * 64ULL;
    const U64 raw_max_k = std::max({h, q, gv, f, shared_f});
    const U64 a8_k = ((raw_max_k + 63ULL) / 64ULL) * 64ULL;
    const U64 a8_groups = a8_k / 64ULL;
    append("a8_qa", multiply(a8_mp, a8_k));
    append("a8_sa", multiply(multiply(a8_groups, a8_mp), 4ULL));
    append("a8_ra", multiply(multiply(a8_groups, a8_mp), 4ULL));
  }
  if (enable_minference_workspace && c.full_attention_layers() > 0 &&
      c.head_dim == 256) {
    const U64 qh = positive(c.num_attention_heads);
    const U64 kvh = positive(c.num_key_value_heads);
    const U64 repeat = std::max<U64>(1ULL, qh / kvh);
    const auto q_blocks_for = [&](U64 t_tier, U64 wm_tier) -> U64 {
      const U64 eff_t = std::min<U64>(U64(workspace_rows), t_tier);
      const U64 bm_tok = std::max<U64>(1ULL, (wm_tier * 16ULL) / repeat);
      return std::max<U64>(1ULL, (eff_t + bm_tok - 1ULL) / bm_tok);
    };
    const U64 max_kv_blocks =
        std::min<U64>(8192ULL, std::max<U64>(128ULL, (U64(max_context) + 31ULL) / 32ULL));
    const U64 max_q_blocks = std::max({
        q_blocks_for(U64(workspace_rows), 8ULL),
        q_blocks_for(96ULL, 6ULL),
        q_blocks_for(16ULL, 4ULL),
    });
    const U64 split_count =
        std::clamp<U64>((max_kv_blocks + 127ULL) / 128ULL, 1ULL, 64ULL);
    const auto align256 = [](U64 x) { return (x + 255ULL) & ~255ULL; };
    U64 cursor = 0;
    const auto add_sub = [&](U64 nbytes) {
      cursor = align256(add(cursor, std::max<U64>(256ULL, nbytes)));
    };
    const U64 probe_rows = multiply(64ULL, qh);
    add_sub(multiply(multiply(probe_rows, split_count), 4ULL));
    add_sub(multiply(multiply(probe_rows, split_count), 4ULL));
    add_sub(multiply(probe_rows, 4ULL));
    add_sub(multiply(probe_rows, 4ULL));
    add_sub(multiply(multiply(kvh, max_kv_blocks), 4ULL));
    add_sub(multiply(multiply(kvh, max_kv_blocks), 4ULL));
    add_sub(multiply(multiply(kvh, 256ULL), 2ULL));
    add_sub(multiply(multiply(kvh, 256ULL), 2ULL));
    add_sub(multiply(multiply(kvh, 2ULL), 4ULL));
    add_sub(multiply(multiply(multiply(kvh, max_q_blocks), 512ULL), 2ULL));
    add_sub(multiply(multiply(kvh, max_q_blocks), 4ULL));
    append("minference_scratch", cursor);
  }
  append("zero", 256);
  append("argmax_partials", 1024ULL);
  append("rms_sq_buf",
         std::max<U64>(
             4096ULL,
             multiply(add(multiply(U64(c.num_hidden_layers), 2ULL), 16ULL),
                      32ULL)));
  append("moe_tg_done_buf",
         std::max<U64>(512ULL,
                       multiply(add(U64(c.num_hidden_layers), 16ULL), 32ULL)));
  const U64 logical_pages = (U64(max_context) + 63) / 64;
  const U32 physical_pages = [&]() -> U32 {
    if (logical_pages + 1 > UINT32_MAX)
      throw std::overflow_error("Qwen KV physical page count overflow");
    const U32 min_pages = U32(logical_pages + 1);
    return kv_pool_pages ? std::max<U32>(min_pages, kv_pool_pages) : min_pages;
  }();
  if (c.full_attention_layers() > 0)
    append("kv_page_table", multiply(logical_pages, 4), true);
  const KvStorageFormat kv_fmt =
      affine4_kv ? KvStorageFormat::Affine4Group128 : KvStorageFormat::BFloat16;
  for (U32 layer = 0; layer < c.layer_types.size(); ++layer) {
    const auto prefix = "state." + std::to_string(layer);
    if (c.layer_types[layer] == LayerType::full_attention) {
      const U64 kv_bytes =
          kv_slab_bytes(kv_fmt, physical_pages, 64U,
                        U32(positive(c.num_key_value_heads)),
                        U32(positive(c.head_dim)));
      append(prefix + ".0", kv_bytes, true);
      append(prefix + ".1", kv_bytes, true);
    } else {
      append(prefix + ".0",
             multiply(multiply(gv, positive(c.linear_key_head_dim)), 4), true,
             true);
      append(prefix + ".1",
             multiply(multiply(c.linear_conv_kernel_dim - 1, conv), 2), true,
             true);
      if (speculative) {
        // Dead write-only `.scratch` (151 MiB/slot) eliminated: verify reads
        // `state.0` as `device const float*` without mutating it.
        append(prefix + ".history",
               multiply(multiply(c.linear_conv_kernel_dim - 1, conv), 2));
        tensor(prefix + ".qkv", conv, 8);
        tensor(prefix + ".keys", gk, 8, 4);
        tensor(prefix + ".decays", c.linear_num_value_heads, 8, 4);
        tensor(prefix + ".deltas", gv, 8, 4);
      }
    }
  }
  return plan;
}

QwenLoadBudget qwen_load_budget(U64 weight_bytes,
                                const QwenForwardMemoryPlan &plan,
                                U64 recommended_bytes, U64 allocated_bytes) {
  QwenLoadBudget result;
  result.required_bytes =
      add(add(weight_bytes, plan.buffer_bytes()), result.runtime_reserve_bytes);
  result.available_bytes = recommended_bytes > allocated_bytes
                               ? recommended_bytes - allocated_bytes
                               : 0;
  result.fits = result.required_bytes <= result.available_bytes;
  return result;
}
} // namespace omlx::m5
