#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "omlx_m5/kv_page_pool.h"
#include "omlx_m5/metal.h"

namespace omlx::m5::detail {

// ============================================================================
// Module 3: Hybrid 48-GDN + 16-Full-Attn Cross-Chunk Reuse (`GDN-CacheBlend`)
// ============================================================================

inline constexpr std::uint32_t kBlendDefaultLMin = 256U;
inline constexpr std::uint32_t kBlendDefaultLMax = 4096U;
inline constexpr std::uint32_t kBlendDefaultBoundaryWindow = 64U;
inline constexpr std::uint32_t kBlendDefaultDenseTailTokens = 128U;
inline constexpr std::uint32_t kBlendDefaultMinSuffixTokens = 512U;
inline constexpr std::uint32_t kBlendDefaultGearWindow = 16U;
inline constexpr std::uint32_t kBlendDefaultGearMask = (1U << 11U) - 1U;
inline constexpr std::uint32_t kBlendDefaultGearMinLen = 512U;
inline constexpr std::uint32_t kBlendDefaultPageTokens = 64U;
inline constexpr std::uint32_t kBlendDefaultHeadDim = 256U;
inline constexpr std::uint32_t kBlendDefaultRotaryDim = 64U;
inline constexpr float kBlendDefaultHkvdRatio = 0.15f;

// Token classification flags consumed by `omlx_m5_blend_hkvd_l2_score` and
// `omlx_m5_blend_hkvd_topk_select`.
inline constexpr std::uint32_t kBlendTokenSkip = 0U;
inline constexpr std::uint32_t kBlendTokenMandatory = 1U;
inline constexpr std::uint32_t kBlendTokenInteriorCandidate = 2U;

// Storage precision for interior `[W_bnd, L_m)` GDN transition operators (M_C, B_C).
enum class BlendGdnOperatorFormat : std::uint32_t {
  BFloat16 = 0U,        // 2 B/elem BF16 + per-row FP16 scale
  Fp8E4M3RowScale = 1U, // 1 B/elem OCP FP8 E4M3 + per-row FP16 scale (compact ~1.016 B/elem)
  Float16RowScale = 2U, // 2 B/elem FP16 + per-row FP16 scale (10-bit mantissa high precision)
};

// Per-vector `affine4_g128` cached KV payload for `head_dim == 256` (144 B/vec).
// Group 0 covers dims [0..128) (which strictly encloses Qwen3.8 rotary_dim = 64).
// Group 1 covers dims [128..256) (100% RoPE-free -> zero re-quantization on shift).
struct alignas(16) BlendAffine4Vec256 {
  std::uint8_t codes_g0[64];
  std::uint8_t codes_g1[64];
  KvAffine4VecMeta meta;
};
static_assert(sizeof(BlendAffine4Vec256) == 144);

// ============================================================================
// GPU Parameter ABI Structs (must match `miv/kernels/cache_blend.metal` exactly)
// ============================================================================

struct BlendExtractPreRopeKvParams {
  std::uint32_t key_value_heads = 0;
  std::uint32_t head_dim = 256;
  std::uint32_t rotary_dim = 64;
  std::uint32_t tokens = 0;
  std::uint32_t key_token_stride = 0;
  std::uint32_t value_token_stride = 0;
  std::uint32_t apply_k_norm = 0;
  std::uint32_t write_rotary_bf16 = 1;
  float epsilon = 1.0e-6f;
  float weight_offset = 0.0f;
  std::uint32_t _pad0 = 0;
  std::uint32_t _pad1 = 0;
};
static_assert(sizeof(BlendExtractPreRopeKvParams) == 48);

struct BlendInstantiateKvParams {
  std::uint32_t key_value_heads = 0;
  std::uint32_t head_dim = 256;
  std::uint32_t rotary_dim = 64;
  std::uint32_t chunk_tokens = 0;
  std::uint32_t dst_start_pos = 0;
  std::uint32_t page_tokens = 64;
  float theta = 10000000.0f;
  std::uint32_t src_token_offset = 0;
  std::uint32_t use_rotary_bf16 = 1;
  std::uint32_t _pad0 = 0;
  std::uint32_t _pad1 = 0;
  std::uint32_t _pad2 = 0;
};
static_assert(sizeof(BlendInstantiateKvParams) == 48);

struct BlendRefreshPageSummaryParams {
  std::uint32_t key_value_heads = 0;
  std::uint32_t head_dim = 256;
  std::uint32_t page_tokens = 64;
  std::uint32_t first_logical_page = 0;
  std::uint32_t num_logical_pages = 0;
  std::uint32_t total_valid_tokens = 0;
};
static_assert(sizeof(BlendRefreshPageSummaryParams) == 24);

struct BlendGdnExtractMParams {
  std::uint32_t key_heads = 16;
  std::uint32_t value_heads = 48;
  std::uint32_t key_dim = 128;
  std::uint32_t value_dim = 128;
  std::uint32_t tokens = 0;
  std::uint32_t _pad0 = 0;
};
static_assert(sizeof(BlendGdnExtractMParams) == 24);

struct BlendGdnPackOpParams {
  std::uint32_t value_heads = 48;
  std::uint32_t dim = 128;
  std::uint32_t op_format = 0; // BlendGdnOperatorFormat
  std::uint32_t _pad0 = 0;
};
static_assert(sizeof(BlendGdnPackOpParams) == 16);

struct BlendGdnStateComposeParams {
  std::uint32_t key_heads = 16;
  std::uint32_t value_heads = 48;
  std::uint32_t dim = 128;
  std::uint32_t op_format = 0;       // BlendGdnOperatorFormat
  std::uint32_t chunk_tokens = 0;    // L_m (length of G_t table)
  std::uint32_t num_hkvd_perturb = 0;
  std::uint32_t apply_bias = 1;      // 1 = add B_C, 0 = homogeneous S_in * M_C
  std::uint32_t _pad0 = 0;
};
static_assert(sizeof(BlendGdnStateComposeParams) == 32);

struct BlendHkvdSelectParams {
  std::uint32_t total_tokens = 0;
  std::uint32_t mix_width = 0;
  std::uint32_t cached_is_int8 = 1;
  std::uint32_t compute_l2_inline = 0;
  std::uint32_t explicit_hkvd_k = 0; // 0 => ceil(hkvd_ratio * n_cand)
  float hkvd_ratio = kBlendDefaultHkvdRatio;
  std::uint32_t _pad0 = 0;
  std::uint32_t _pad1 = 0;
};
static_assert(sizeof(BlendHkvdSelectParams) == 32);

struct BlendGatherScatterParams {
  std::uint32_t max_active_tokens = 0;
  std::uint32_t width = 0;
  std::uint32_t use_device_count = 0;
  std::uint32_t residual_add = 0;
};
static_assert(sizeof(BlendGatherScatterParams) == 16);

struct BlendQkNormRopeScatterParams {
  std::uint32_t query_heads = 0;
  std::uint32_t key_value_heads = 0;
  std::uint32_t head_dim = 256;
  std::uint32_t rotary_dim = 64;
  std::uint32_t max_active_tokens = 0;
  std::uint32_t query_token_stride = 0;
  std::uint32_t query_head_stride = 0;
  std::uint32_t key_token_stride = 0;
  std::uint32_t key_head_stride = 0;
  std::uint32_t value_token_stride = 0;
  std::uint32_t page_tokens = 64;
  std::uint32_t use_device_count = 0;
  float epsilon = 1.0e-6f;
  float theta = 10000000.0f;
  float weight_offset = 0.0f;
  std::uint32_t _pad0 = 0;
};
static_assert(sizeof(BlendQkNormRopeScatterParams) == 64);

struct BlendInt8QuantParams {
  std::uint32_t tokens = 0;
  std::uint32_t width = 0;
};
static_assert(sizeof(BlendInt8QuantParams) == 8);

struct BlendGdnCumDecayParams {
  std::uint32_t value_heads = 48;
  std::uint32_t tokens = 0;
};
static_assert(sizeof(BlendGdnCumDecayParams) == 8);

// ============================================================================
// Content-Defined + Structural Delimiter-Anchored Chunk Segmentation
// ============================================================================

enum class BlendSegmentKind : std::uint8_t {
  Prefix = 0,         // Covered by Tier-1 SharedPrefixCache
  CacheableChunk = 1, // Eligible for Tier-2 ChunkBlendCache lookup/populate
  UncachedDense = 2,  // Mandatory dense execution (e.g. trailing query frontier)
};

struct BlendSegmentConfig {
  std::uint32_t l_min = kBlendDefaultLMin;
  std::uint32_t l_max = kBlendDefaultLMax;
  std::uint32_t boundary_window = kBlendDefaultBoundaryWindow;
  std::uint32_t dense_tail_tokens = kBlendDefaultDenseTailTokens;
  std::uint32_t min_suffix_for_blend = kBlendDefaultMinSuffixTokens;
  std::uint32_t gear_window = kBlendDefaultGearWindow;
  std::uint32_t gear_mask = kBlendDefaultGearMask;
  std::uint32_t gear_min_len = kBlendDefaultGearMinLen;
  // Structural delimiter sequences:
  // - prefix_delimiters: cut immediately BEFORE a matching sequence (e.g. <|im_start|>, <doc, ---)
  // - suffix_delimiters: cut immediately AFTER a matching sequence (e.g. <|im_end|>, </doc>)
  std::vector<std::vector<std::uint32_t>> prefix_delimiters = {{151644U}, {248045U}};
  std::vector<std::vector<std::uint32_t>> suffix_delimiters = {{151645U}, {248046U}};
};

struct BlendChunkSegment {
  std::uint32_t start = 0;
  std::uint32_t length = 0;
  BlendSegmentKind kind = BlendSegmentKind::UncachedDense;
  // Leading boundary recomputation window (`W_bnd = 64` per stitched non-prefix chunk).
  std::uint32_t boundary_window = 0;
  std::uint64_t content_hash = 0;
};

[[nodiscard]] std::uint64_t
fnv1a_64_blend_bytes(const void *data, std::size_t bytes,
                     std::uint64_t seed = 14695981039346656037ULL) noexcept;

[[nodiscard]] std::uint64_t
fnv1a_64_blend_tokens(std::span<const std::uint32_t> tokens,
                      std::uint64_t seed = 14695981039346656037ULL) noexcept;

[[nodiscard]] std::uint64_t compute_blend_model_exec_hash(
    std::string_view artifact_digest, std::string_view kv_dtype,
    std::string_view prefill_linear_mode,
    std::string_view prefill_attention_mode,
    bool prefill_sparse_minference = false,
    std::string_view prefill_ane_mode = "off") noexcept;

[[nodiscard]] std::vector<BlendChunkSegment>
segment_blend_chunks(std::span<const std::uint32_t> tokens,
                     std::uint32_t prefix_tokens = 0U,
                     const BlendSegmentConfig &cfg = {});

// ============================================================================
// BlendChunkKey, Geometry, GPU Payload & Thread-Safe Byte-Capped LRU Cache
// ============================================================================

struct BlendChunkKey {
  std::uint64_t model_exec_hash = 0;
  std::uint64_t token_content_hash = 0;
  std::uint32_t token_length = 0;
  std::uint32_t first_token = 0;
  std::uint32_t last_token = 0;

  bool operator==(const BlendChunkKey &) const noexcept = default;
};

struct BlendChunkKeyHash {
  [[nodiscard]] std::size_t operator()(const BlendChunkKey &k) const noexcept;
};

[[nodiscard]] BlendChunkKey
make_blend_chunk_key(std::uint64_t model_exec_hash,
                     std::span<const std::uint32_t> tokens) noexcept;

struct BlendChunkGeometry {
  std::uint32_t num_full_attn_layers = 16;
  std::uint32_t num_gdn_layers = 48;
  std::uint32_t q_heads = 24;
  std::uint32_t kv_heads = 4;
  std::uint32_t head_dim = 256;
  std::uint32_t rotary_dim = 64;
  std::uint32_t hidden_size = 5120;
  std::uint32_t gdn_key_heads = 16;
  std::uint32_t gdn_value_heads = 48;
  std::uint32_t gdn_key_dim = 128;
  std::uint32_t gdn_value_dim = 128;
  std::uint32_t gdn_conv_width = 10240;
  std::uint32_t gdn_conv_kernel = 4;
  std::uint32_t token_length = 0;
  std::uint32_t interior_start = kBlendDefaultBoundaryWindow;
  std::uint32_t probe_layer_index = 3;
  BlendGdnOperatorFormat gdn_op_format = BlendGdnOperatorFormat::BFloat16;

  [[nodiscard]] std::uint64_t attn_vec_bytes() const noexcept;
  [[nodiscard]] std::uint64_t attn_k_pre_bytes() const noexcept;
  [[nodiscard]] std::uint64_t attn_k_rotary_bf16_bytes() const noexcept;
  [[nodiscard]] std::uint64_t attn_v_bytes() const noexcept;
  [[nodiscard]] std::uint64_t h3_res_i8_bytes() const noexcept;
  [[nodiscard]] std::uint64_t h3_res_scale_bytes() const noexcept;
  [[nodiscard]] std::uint64_t q3_pre_i8_bytes() const noexcept;
  [[nodiscard]] std::uint64_t q3_pre_scale_bytes() const noexcept;
  [[nodiscard]] std::uint64_t mix3_cached_i8_bytes() const noexcept;
  [[nodiscard]] std::uint64_t mix3_cached_scale_bytes() const noexcept;
  [[nodiscard]] std::uint64_t gdn_tail_qkv_bytes() const noexcept;
  [[nodiscard]] std::uint64_t gdn_op_data_bytes() const noexcept;
  [[nodiscard]] std::uint64_t gdn_op_scale_bytes() const noexcept;
  [[nodiscard]] std::uint64_t gdn_cum_decay_bytes() const noexcept;
  [[nodiscard]] std::uint64_t total_payload_bytes() const noexcept;
};

struct BlendChunkPayload {
  // 16 Full-Attn layers: pre-RoPE K_pre (post k_norm) + V in affine4_g128 (144 B/vec)
  // plus compact pre-RoPE BF16 sidecar for the `rotary_dim = 64` dims only so
  // RoPE repositioning at arbitrary `p_new` is true single-quantize (cos_sim >= 0.9998).
  MetalBuffer attn_k_pre;
  MetalBuffer attn_k_rotary_bf16;
  MetalBuffer attn_v;

  // Layer-3 HKVD probe baseline in compact int8_g128:
  //   data: int8_t[token_length, width], scales: half[token_length, width / 128].
  MetalBuffer h3_res_i8;
  MetalBuffer h3_res_scales;
  MetalBuffer q3_pre_i8;
  MetalBuffer q3_pre_scales;
  MetalBuffer mix3_cached_i8;
  MetalBuffer mix3_cached_scales;

  // 48 GDN layers:
  //   - tail_qkv: [num_gdn_layers, conv_kernel - 1, conv_width] BF16
  //   - interior [interior_start, token_length) affine transition (M_C, B_C):
  //     data: [num_gdn_layers, Hv, 128, 128] BF16 or FP8 E4M3
  //     row_scales: [num_gdn_layers, Hv, 128] FP16
  //   - cum_decay_f16: [num_gdn_layers, token_length, Hv] FP16
  MetalBuffer gdn_tail_qkv;
  MetalBuffer gdn_m_data;
  MetalBuffer gdn_m_row_scales;
  MetalBuffer gdn_b_data;
  MetalBuffer gdn_b_row_scales;
  MetalBuffer gdn_cum_decay_f16;
};

[[nodiscard]] BlendChunkPayload
allocate_blend_chunk_payload(const MetalContext &context,
                             const BlendChunkGeometry &geom);

struct BlendChunkEntry {
  BlendChunkKey key;
  std::vector<std::uint32_t> tokens; // Exact token-span equality guard
  BlendChunkGeometry geometry;
  BlendChunkPayload payload;
  std::uint64_t accounted_bytes = 0;
  std::uint64_t hit_count = 0;
};

struct BlendCacheStats {
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t insertions = 0;
  std::uint64_t evictions = 0;
  std::uint64_t cached_bytes = 0;
  std::uint64_t budget_bytes = 0;
  std::size_t entry_count = 0;
};

struct BlendPlannedSegment {
  BlendChunkSegment segment;
  std::shared_ptr<const BlendChunkEntry> hit_entry;
  [[nodiscard]] bool is_cache_hit() const noexcept { return hit_entry != nullptr; }
};

struct BlendPromptPlan {
  std::uint32_t prefix_tokens = 0;
  std::uint32_t total_tokens = 0;
  std::uint32_t hit_tokens = 0;
  std::uint32_t miss_tokens = 0;
  std::uint32_t boundary_tokens = 0;
  std::uint32_t dense_tail_tokens = 0;
  std::vector<BlendPlannedSegment> segments;
  // Per-suffix-token classification flags (`kBlendTokenMandatory` vs
  // `kBlendTokenInteriorCandidate`) and logical prompt positions for
  // direct upload to `omlx_m5_blend_hkvd_topk_select`.
  std::vector<std::uint32_t> suffix_token_flags;
  std::vector<std::uint32_t> suffix_prompt_positions;
};

// Thread-safe byte-capped LRU cache for position-independent RAG chunks.
// Entries are returned as `std::shared_ptr<const BlendChunkEntry>` so in-flight
// GPU command buffers safely retain their UMA buffers even across evictions.
class ChunkBlendCache final {
public:
  explicit ChunkBlendCache(std::uint64_t budget_bytes = 0);

  ChunkBlendCache(const ChunkBlendCache &) = delete;
  ChunkBlendCache &operator=(const ChunkBlendCache &) = delete;

  [[nodiscard]] bool enabled() const noexcept;
  [[nodiscard]] std::uint64_t budget_bytes() const noexcept;
  void set_budget_bytes(std::uint64_t budget_bytes);
  [[nodiscard]] std::uint64_t cached_bytes() const;
  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] bool empty() const;
  [[nodiscard]] BlendCacheStats stats() const;

  [[nodiscard]] std::shared_ptr<const BlendChunkEntry>
  lookup(std::uint64_t model_exec_hash, std::span<const std::uint32_t> tokens);

  bool insert(std::shared_ptr<BlendChunkEntry> entry);

  std::uint64_t trim_to_bytes(std::uint64_t target_bytes);
  std::uint64_t clear();

  [[nodiscard]] BlendPromptPlan
  plan_prompt(std::uint64_t model_exec_hash,
              std::span<const std::uint32_t> prompt,
              std::uint32_t prefix_tokens = 0U,
              const BlendSegmentConfig &cfg = {});

private:
  using EntryPtr = std::shared_ptr<BlendChunkEntry>;
  using LruList = std::list<EntryPtr>;
  using IndexMap =
      std::unordered_multimap<BlendChunkKey, LruList::iterator, BlendChunkKeyHash>;

  void erase_locked(IndexMap::iterator map_it) noexcept;
  void evict_to_locked(std::uint64_t target_bytes) noexcept;

  mutable std::mutex mutex_;
  std::uint64_t budget_bytes_ = 0;
  std::uint64_t cached_bytes_ = 0;
  std::uint64_t hits_ = 0;
  std::uint64_t misses_ = 0;
  std::uint64_t insertions_ = 0;
  std::uint64_t evictions_ = 0;
  LruList lru_; // Front = MRU, Back = LRU
  IndexMap index_;
};

} // namespace omlx::m5::detail
