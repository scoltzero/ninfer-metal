#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/model_config.h"

namespace omlx::m5::ane {

// Fixed-chunk geometry and alignment contracts for Dual-ANE + GPU co-execution.
inline constexpr std::uint32_t kAneDefaultChunkTokens = 1024u;
inline constexpr std::uint32_t kAneRowAlign = 128u;
inline constexpr std::uint32_t kAneGroupSize = 64u;
inline constexpr std::uint64_t kAneDefaultMaxBankBytesPerDie =
    320ULL * 1024ULL * 1024ULL; // 320 MiB per shard (~10 fast-compiling shards/die on 27B)
inline constexpr std::uint64_t kAneDefaultMaxTotalBytesPerDie =
    3584ULL * 1024ULL * 1024ULL; // 3.5 GiB total resident cap per die
inline constexpr std::uint32_t kAneMaxProceduresPerBank = 24u;
inline constexpr double kAneDefaultAlphaBf16 = 0.25;
inline constexpr double kAneDefaultAlphaW4A8 = 0.15;
inline constexpr double kAneMaxAlpha = 0.30;

// Mirrors Module 4a's PrefillLinearMode without touching types.h in Wave 1.
enum class AneLinearRoute : std::uint8_t {
  BF16 = 0,
  W4A8 = 1,
};

// Strictly allowlisted non-recurrent prefill projections eligible for Dual-ANE
// output-channel-split co-execution.
//
// NON-NEGOTIABLE INVARIANT (GDN Recurrence Exclusion):
//   linear_attn.in_proj_qkv, linear_attn.in_proj_a, linear_attn.in_proj_b, and
//   self_attn.{q_proj, k_proj, v_proj} are NEVER represented here and are
//   explicitly rejected by ane_stage_role_eligible().
enum class AneProjectionRole : std::uint8_t {
  MlpGateUpSwiGlu = 0, // Fused SiLU(gate_slice) * up_slice inside MIL procedure
  MlpDown = 1,         // mlp.down_proj
  GdnInProjZ = 2,      // linear_attn.in_proj_z (token-local output gate only)
  GdnOutProj = 3,      // linear_attn.out_proj
  AttnOProj = 4,       // self_attn.o_proj
};

[[nodiscard]] std::string_view to_string(AneProjectionRole role) noexcept;
[[nodiscard]] std::string_view to_string(AneLinearRoute route) noexcept;

// Returns true iff a QwenLayerCatalog stage role or tensor name is eligible for
// ANE offload. Explicitly rejects GDN recurrent projections (in_proj_qkv,
// in_proj_a, in_proj_b), attention q/k/v projections, MoE routed experts,
// embeddings, norms, and lm_head.
[[nodiscard]] bool ane_stage_role_eligible(std::string_view role_or_name) noexcept;

// Runtime hardware & private AppleNeuralEngine.framework probe result.
struct AneHardwareInfo {
  bool framework_loaded = false;
  bool classes_available = false;
  bool metal_iosurface_supported = false;
  bool hardware_available = false;
  std::uint32_t physical_die_count = 0; // 0 if unavailable, 1 on M1-M5 Max, 2 on Ultra
  std::string os_build;
  std::string unavailable_reason;

  [[nodiscard]] bool is_available() const noexcept {
    return framework_loaded && classes_available && metal_iosurface_supported &&
           hardware_available && physical_die_count > 0;
  }
};

[[nodiscard]] AneHardwareInfo probe_ane_hardware() noexcept;
[[nodiscard]] bool is_ane_available() noexcept;

// Output-channel split geometry aligned to 128 rows (% 128 == 0 => % 64 == 0).
struct AneChannelSplit {
  std::uint32_t total_out_features = 0;
  std::uint32_t active_dies = 0;
  std::uint32_t die_rows = 0;       // N_die per die (% 128 == 0)
  std::uint32_t ane_total_rows = 0; // active_dies * die_rows (% 128 == 0)
  std::uint32_t gpu_row_offset = 0; // == ane_total_rows (% 128 == 0, % 64 == 0)
  std::uint32_t gpu_rows = 0;       // total_out_features - ane_total_rows
  double effective_alpha = 0.0;

  [[nodiscard]] bool enabled() const noexcept { return ane_total_rows > 0; }
};

[[nodiscard]] AneChannelSplit compute_ane_channel_split(
    std::uint32_t out_features, double target_alpha, std::uint32_t active_dies,
    double max_alpha = kAneMaxAlpha,
    std::uint32_t min_gpu_rows = kAneRowAlign) noexcept;

// Groupwise G=64 asymmetric INT8 weight representation consumed by MIL
// constexpr_blockwise_shift_scale(data=int8[N,K], scale=fp16[N,K/64], offset=int8[N,K/64]).
// Dequantized value: (float(data[r, c]) - float(offset[r, c/64])) * fp16(scale[r, c/64]).
struct AneGroup64Int8Weight {
  std::uint32_t rows = 0;
  std::uint32_t in_features = 0;
  std::uint32_t group_size = kAneGroupSize;
  std::vector<std::int8_t> data;    // [rows * in_features]
  std::vector<std::uint16_t> scale; // [rows * (in_features / 64)] IEEE-754 fp16
  std::vector<std::int8_t> offset;  // [rows * (in_features / 64)] int8 zero-point

  [[nodiscard]] std::uint64_t payload_bytes() const noexcept {
    return static_cast<std::uint64_t>(data.size()) +
           static_cast<std::uint64_t>(scale.size()) * sizeof(std::uint16_t) +
           static_cast<std::uint64_t>(offset.size());
  }
};

// Zero-reconversion direct extraction from MaterializedModel UMA weight planes.
// Supports:
//   - mxfp4_e2m1_g32_e8m0_v1 (mx_rowsplit_k128_v1 AND mx_nax_tiled_bk128_v1)
//   - oq_affine_q{3,4,5,6,8}_g{64,128}_bf16_v1 (oq_rowsplit_k128_v1 AND
//     oq_nax_tiled_g64_v1 / oq_nax_tiled_g128_v1)
//   - direct_bf16_v1 / direct_f16_v1 / direct_f32_v1
[[nodiscard]] bool ane_weight_view_supported(
    const MaterializedWeightView &weight) noexcept;

[[nodiscard]] std::vector<float> dequantize_miv_weight_slice_f32(
    const MaterializedWeightView &weight, std::uint64_t row_start,
    std::uint64_t row_count);

[[nodiscard]] AneGroup64Int8Weight quantize_f32_to_g64_int8(
    std::span<const float> weights, std::uint32_t rows,
    std::uint32_t in_features);

[[nodiscard]] AneGroup64Int8Weight extract_miv_weight_slice_g64_int8(
    const MaterializedWeightView &weight, std::uint64_t row_start,
    std::uint64_t row_count);

[[nodiscard]] std::vector<float> dequantize_g64_int8_to_f32(
    const AneGroup64Int8Weight &quantized);

// Descriptor for one eligible prefill operation compiled into the Dual-ANE banks.
struct AneOffloadOpSpec {
  std::uint32_t layer_index = 0;
  AneProjectionRole role = AneProjectionRole::MlpDown;
  std::string name;
  // Primary weight (or gate_proj when role == MlpGateUpSwiGlu).
  MaterializedWeightView weight;
  // Secondary weight (required iff role == MlpGateUpSwiGlu, holds up_proj).
  std::optional<MaterializedWeightView> up_weight;
};

// Resolved plan entry after 128-row alignment and bank sharding.
struct AneCompiledOpInfo {
  std::uint32_t op_id = 0;
  std::uint32_t layer_index = 0;
  AneProjectionRole role = AneProjectionRole::MlpDown;
  std::string name;
  std::uint32_t in_features = 0;
  AneChannelSplit split{};
  std::uint32_t shard_index = 0;
  std::uint32_t procedure_index = 0;
  std::uint64_t per_die_weight_bytes = 0;
};

struct AneDualOffloadConfig {
  std::uint32_t chunk_tokens = kAneDefaultChunkTokens; // Fixed T_chunk (default 1024)
  AneLinearRoute linear_route = AneLinearRoute::BF16;
  double alpha_bf16 = kAneDefaultAlphaBf16;
  double alpha_w4a8 = kAneDefaultAlphaW4A8;
  std::optional<double> alpha_override;
  double max_alpha = kAneMaxAlpha;
  std::uint64_t max_bank_bytes_per_die = kAneDefaultMaxBankBytesPerDie;
  std::uint64_t max_total_bytes_per_die = kAneDefaultMaxTotalBytesPerDie;
  std::uint32_t max_procedures_per_bank = kAneMaxProceduresPerBank;
  // Direct FP16 MIL weight blobs instead of G=64 INT8 blockwise dequant:
  // ~1.7x higher ANE TFLOPS on Apple M5 Ultra with zero weight quantization loss.
  bool use_fp16_weights = false;
  // Exclude projections with in_features > max_in_features (0 = no cap).
  // Avoids Apple ANE's C_in > 8192 input-SRAM spill cliff (e.g. mlp.down_proj K=17408).
  std::uint32_t max_in_features = 0u;
  // Offload only fused MlpGateUpSwiGlu projections (highest FLOP/dispatch ratio).
  bool mlp_swiglu_only = false;
  // Optional overrides for single-die testing / dual-bank simulation on 1-die Macs.
  bool force_single_die = false;
  bool simulate_dual_banks_on_single_die = false;
  // Persistent flock()-guarded disk cache configuration.
  // Empty cache_root resolves to ~/Library/Caches/io.ninfer.miv/ane_banks_v1.
  bool enable_disk_cache = true;
  std::filesystem::path cache_root;
  std::string artifact_digest; // Used in SHA-256 cache key
  // Path to omlx_m5_ane_pack_unpack.metallib (or directory containing it).
  std::filesystem::path metallib_path;
  // Optional external MTLDevice / MTLCommandQueue (opaque id<MTLDevice>/id<MTLCommandQueue>).
  // When nullptr, the engine creates its own system default Metal device & queue.
  void *external_mtl_device = nullptr;
  void *external_mtl_command_queue = nullptr;
  // Auto-bypass calibration guard margin: bypass if t_ane_pair > t_gpu_only * bypass_ratio.
  double calibration_bypass_ratio = 1.0;
};

struct AneCalibrationResult {
  bool performed = false;
  bool active = false;
  bool auto_bypassed = false;
  std::uint32_t calibrated_op_id = 0;
  double ane_pair_ms = 0.0;
  double gpu_only_ms = 0.0;
  double gpu_suffix_ms = 0.0;
  std::string reason;
};

struct AneEngineStats {
  bool available = false;
  bool active = false;
  bool auto_bypassed = false;
  std::uint32_t physical_die_count = 0;
  std::uint32_t active_die_count = 0;
  bool die_pinned[2] = {false, false};
  std::uint32_t chunk_tokens = 0;
  AneLinearRoute linear_route = AneLinearRoute::BF16;
  double target_alpha = 0.0;
  std::uint32_t compiled_ops = 0;
  std::uint32_t bank_shards_per_die = 0;
  std::uint64_t total_weight_bytes_per_die = 0;
  std::uint64_t iosurface_pool_bytes = 0;
  std::uint32_t disk_cache_hits = 0;
  std::uint32_t disk_cache_misses = 0;
  std::uint32_t native_compiled_cache_hits = 0;
  std::uint64_t total_enqueued_ops = 0;
  std::uint64_t total_completed_ops = 0;
  std::string cache_key;
  std::filesystem::path cache_dir;
  std::string status_message;
};

// Lightweight RAII shared MTLBuffer bound to the engine's MTLDevice.
// Enables pure-C++ callers and tests to stage BF16 activations/outputs without
// exposing Objective-C headers in C++ translation units.
class AneSharedBuffer final {
public:
  AneSharedBuffer();
  AneSharedBuffer(AneSharedBuffer &&) noexcept;
  AneSharedBuffer &operator=(AneSharedBuffer &&) noexcept;
  AneSharedBuffer(const AneSharedBuffer &) = delete;
  AneSharedBuffer &operator=(const AneSharedBuffer &) = delete;
  ~AneSharedBuffer();

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::uint64_t size() const noexcept;
  [[nodiscard]] void *data() noexcept;
  [[nodiscard]] const void *data() const noexcept;
  [[nodiscard]] void *native_mtl_buffer() const noexcept;

private:
  class Impl;
  friend class AneDualOffloadEngine;
  explicit AneSharedBuffer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Native miv C++/ObjC++ Dual-ANE Fixed-Chunk Output-Channel-Split Co-Execution Engine.
//
// Owns:
//  - Multi-procedure MIL program(1.3) banks pinned to ane0 (instance 1) & ane1 (instance 2),
//    with automatic 1-die fallback on single-die Apple Silicon Macs.
//  - Persistent flock()-guarded disk cache under ~/Library/Caches/io.ninfer.miv/ane_banks_v1/<sha256>/.
//  - Ping-pong IOSurfaceRef pools backed by zero-copy Metal buffers (newBufferWithIOSurface:).
//  - Zero-CPU-stall MTLSharedEvent + MTLSharedEventListener async pipeline overlapping
//    ane0||ane1 on [0, N_ane) with GPU bf16/w4a8 suffix on [N_ane, N).
class AneDualOffloadEngine final {
public:
  // Build from explicit eligible op specs (validates strict GDN recurrence exclusion).
  [[nodiscard]] static std::unique_ptr<AneDualOffloadEngine> build(
      std::span<const AneOffloadOpSpec> specs,
      const AneDualOffloadConfig &config);

  // Build directly from a MaterializedModel + QwenLayerCatalog.
  // Automatically extracts ONLY eligible non-recurrent projections:
  //   - Dense MLP: mlp_gate + mlp_up (fused SiLU(gate)*up) and mlp_down
  //   - GDN: gdn_z (in_proj_z) and gdn_out (out_proj) -- NEVER gdn_qkv/a/b!
  //   - Full-Attn: attention_o (o_proj) -- NEVER attention_q/k/v!
  [[nodiscard]] static std::unique_ptr<AneDualOffloadEngine> build_from_model(
      const MaterializedModel &model, const QwenLayerCatalog &catalog,
      const AneDualOffloadConfig &config);

  // Returns default cache directory: ~/Library/Caches/io.ninfer.miv/ane_banks_v1
  [[nodiscard]] static std::filesystem::path default_cache_root();

  // Compute deterministic SHA-256 cache key for the given configuration and ops.
  [[nodiscard]] static std::string compute_cache_key(
      const AneDualOffloadConfig &config,
      std::span<const AneOffloadOpSpec> specs,
      std::uint32_t active_dies,
      std::string_view os_build);

  AneDualOffloadEngine(AneDualOffloadEngine &&) noexcept;
  AneDualOffloadEngine &operator=(AneDualOffloadEngine &&) noexcept;
  AneDualOffloadEngine(const AneDualOffloadEngine &) = delete;
  AneDualOffloadEngine &operator=(const AneDualOffloadEngine &) = delete;
  ~AneDualOffloadEngine();

  [[nodiscard]] bool is_available() const noexcept;
  [[nodiscard]] bool is_active() const noexcept;
  [[nodiscard]] const AneEngineStats &stats() const noexcept;
  [[nodiscard]] std::span<const AneCompiledOpInfo> ops() const noexcept;

  // Lookup compiled op by (layer_index, role); returns nullptr if not offloaded.
  [[nodiscard]] const AneCompiledOpInfo *find_op(
      std::uint32_t layer_index, AneProjectionRole role) const noexcept;
  [[nodiscard]] const AneCompiledOpInfo *find_op_by_id(
      std::uint32_t op_id) const noexcept;

  // Returns true iff a prefill chunk of `tokens` can execute on this engine
  // (i.e. engine is active and tokens == chunk_tokens).
  [[nodiscard]] bool can_offload_chunk(std::uint32_t tokens) const noexcept;

  // Allocate a shared MTLBuffer on the engine's MTLDevice.
  [[nodiscard]] AneSharedBuffer make_shared_buffer(std::uint64_t bytes) const;

  // Load-time 2-iteration warmup timing micro-calibration.
  // Compares measured ANE pair time (pack + ane0||ane1 + unpack) against
  // caller-measured GPU-only time (t_gpu_only_ms) and optional GPU suffix time.
  // If t_ane_pair > t_gpu_only * bypass_ratio, sets is_active() = false so
  // prefill_ane_offload = true can NEVER regress throughput.
  AneCalibrationResult calibrate(double gpu_only_ms,
                                 double gpu_suffix_ms = 0.0,
                                 std::uint32_t representative_op_id = 0);

  // Zero-CPU-stall async dispatch of one Dual-ANE + GPU suffix co-execution op.
  //
  // Pipeline encoded onto MTLCommandQueue with ZERO waitUntilCompleted() calls:
  //   CB_A: (wait unpack_done[seq-2]) -> Pack [T_chunk, K] BF16 -> IOSurface_in[slot] FP16
  //         -> signal(ane_ready, seq) -> MTLSharedEventListener dispatches ane0 || ane1
  //   CB_B1 (optional): caller encodes GPU bf16/w4a8 suffix [N_ane, N) into dst
  //   CB_B2: wait(ane_done, seq) -> Unpack IOSurface_out0/1[slot] FP16 -> dst[:, 0..N_ane) BF16
  //         -> signal(unpack_done, seq)
  //
  // src_mtl_buffer: opaque id<MTLBuffer> holding [T_chunk, src_stride] BF16
  // dst_mtl_buffer: opaque id<MTLBuffer> holding [T_chunk, out_stride] BF16
  // encode_gpu_suffix: optional callback receiving opaque id<MTLCommandBuffer>
  //                    for CB_B1 (runs concurrently on GPU while ane0||ane1 execute).
  void enqueue_op(
      std::uint32_t op_id,
      void *src_mtl_buffer,
      std::uint64_t src_offset,
      void *dst_mtl_buffer,
      std::uint64_t dst_offset,
      std::uint32_t out_stride,
      std::function<void(void *mtl_command_buffer)> encode_gpu_suffix = {},
      std::uint32_t src_stride = 0,
      std::uint32_t dst_col_offset = 0);

  // Single host wait at the end of a T_chunk pass (drains all in-flight ops
  // and throws std::runtime_error if any ANE or Metal error occurred).
  void flush_and_wait(std::uint32_t timeout_ms = 30000u);

  // Convenience synchronous single-op wrapper (enqueue_op + flush_and_wait).
  void run_op_sync(
      std::uint32_t op_id,
      void *src_mtl_buffer,
      std::uint64_t src_offset,
      void *dst_mtl_buffer,
      std::uint64_t dst_offset,
      std::uint32_t out_stride,
      std::function<void(void *mtl_command_buffer)> encode_gpu_suffix = {},
      std::uint32_t src_stride = 0,
      std::uint32_t dst_col_offset = 0);

  // Opaque native handles for Wave 2 integration / testing.
  [[nodiscard]] void *native_mtl_device() const noexcept;
  [[nodiscard]] void *native_mtl_command_queue() const noexcept;

private:
  class Impl;
  explicit AneDualOffloadEngine(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5::ane
