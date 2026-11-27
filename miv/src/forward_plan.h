#pragma once

#include "omlx_m5/qwen_forward.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace omlx::m5::detail::forward {
using U32 = std::uint32_t;
using U64 = std::uint64_t;
struct Ref {
  const MetalBuffer *buffer = nullptr;
  U64 offset = 0;
  U64 bytes = 0;
};
struct Patch {
  U64 offset;
  U32 word;
  bool position;
  U32 multiplier = 1;
};
struct SampleParams {
  U32 vocabulary, top_k;
  float temperature, top_p;
  U64 seed, stream, absolute_position;
};
static_assert(sizeof(SampleParams) == 40);

inline U64 mul(U64 a, U64 b) {
  if (a && b > std::numeric_limits<U64>::max() / a)
    throw std::overflow_error("Qwen execution geometry overflow");
  return a * b;
}
inline U32 word(float value) { return std::bit_cast<U32>(value); }
inline std::vector<std::byte> words(std::initializer_list<U32> data) {
  auto bytes = std::as_bytes(std::span(data.begin(), data.size()));
  return {bytes.begin(), bytes.end()};
}
inline Ref sub(Ref ref, U64 offset) {
  if (offset > ref.bytes)
    throw std::out_of_range("Qwen view offset");
  return {ref.buffer, ref.offset + offset, ref.bytes - offset};
}
inline Ref plane(const MaterializedWeightView &weight, std::string_view kind) {
  for (const auto &p : weight.planes)
    if (p.kind == kind)
      return {p.buffer, p.offset, p.bytes};
  throw std::invalid_argument("Qwen plane absent: " + weight.name + "/" +
                              std::string(kind));
}
inline std::optional<Ref> try_plane(const MaterializedWeightView &weight,
                                    std::string_view kind) {
  for (const auto &p : weight.planes)
    if (p.kind == kind)
      return Ref{p.buffer, p.offset, p.bytes};
  return std::nullopt;
}
inline std::vector<float> bf16_values(const MetalBuffer &buffer, U64 values) {
  auto bytes = buffer.read(0, mul(values, 2));
  std::vector<float> result(values);
  for (U64 i = 0; i < values; ++i) {
    std::uint16_t code;
    std::memcpy(&code, bytes.data() + 2 * i, 2);
    result[i] = std::bit_cast<float>(U32(code) << 16);
  }
  return result;
}

// Shared FNV-1a 64-bit plan-key hasher (Ponytail P1-4 deduplication).
struct Fnv1a64 {
  U64 h = 14695981039346656037ULL;
  void bytes(const void *data, std::size_t len) noexcept {
    const auto *p = static_cast<const unsigned char *>(data);
    for (std::size_t i = 0; i < len; ++i) {
      h ^= static_cast<U64>(p[i]);
      h *= 1099511628211ULL;
    }
  }
  void str(std::string_view s) noexcept { bytes(s.data(), s.size()); }
  void u64(U64 v) noexcept { bytes(&v, sizeof(v)); }
  void weight(const MaterializedWeightView &w) noexcept {
    u64(w.object_id);
    str(w.name);
    str(w.format);
    str(w.layout);
    str(w.input_transform);
    u64(w.row_start);
    u64(w.row_count);
    u64(w.in_features);
    u64(w.padded_k);
  }
  [[nodiscard]] std::string hex() const {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(16, '0');
    U64 v = h;
    for (int i = 15; i >= 0; --i) {
      out[static_cast<std::size_t>(i)] = kHex[v & 0xFULL];
      v >>= 4U;
    }
    return out;
  }
};

// M4-only promotions measured on original 27B/35B slices. The owning Program
// additionally checks the physical device; M5 keeps its existing route until
// target-hardware qualification.
inline bool qualified_m4_matrix_t8(const MaterializedWeightView &weight, U32 n,
                                   U32 k, U32 capacity) {
  if (capacity != 8)
    return false;
  return (weight.format == "oq_affine_q4_g64_bf16_v1" && n == 248320 &&
          k == 5120) ||
         (weight.format == "oq_affine_q5_g64_bf16_v1" && n == 5120 &&
          k == 17408) ||
         (weight.format == "oq_affine_q5_g128_bf16_v1" && n == 8192 &&
          k == 2048) ||
         (weight.format == "oq_affine_q6_g64_bf16_v1" && n == 8192 &&
          k == 2048) ||
         (weight.format == "oq_affine_q8_g64_bf16_v1" && n == 248320 &&
          k == 2048) ||
         (weight.format == "mxfp4_e2m1_g32_e8m0_v1" && n == 248320 &&
          (k == 5120 || k == 2048));
}

inline bool qualified_m4_lane_t1(const MaterializedWeightView &weight, U32 n,
                                 U32 k, U32 capacity) {
  if (capacity != 1)
    return false;
  return (weight.format == "oq_affine_q4_g64_bf16_v1" && n == 248320 &&
          k == 5120) ||
         (weight.format == "oq_affine_q5_g64_bf16_v1" && n == 5120 &&
          k == 17408) ||
         (weight.format == "mxfp4_e2m1_g32_e8m0_v1" && n == 248320 &&
          k == 5120);
}

struct ForwardStage {
  std::string name;
  U32 location = 0;
  U32 length = 0;
};
struct ForwardWeightRoute {
  std::string stage, object, format, library, function, family, source,
      fallback_reason;
  U32 n = 0, k = 0, t = 0;
  MetalDispatchSize grid, threads;
  std::string numeric_contract_id = "legacy_route_v1";
  std::string row_exact_group = {};
  bool row_exact_eligible = false;
};
struct Profile {
  U32 capacity = 0;
  bool split_attention = false;
  U32 split_count = 0;
  U32 context_envelope = 0;
  U64 split_workspace_bytes = 0;
  std::string attention_kernel, attention_merge_kernel;
  MetalDispatchSize attention_grid, attention_merge_grid;
  std::vector<ForwardStage> stages;
  std::vector<ForwardWeightRoute> weight_routes;
  U32 short_matrix_calls = 0;
  U32 short_lane_calls = 0;
  MetalCommandPlan plan;
  std::vector<MetalBufferBinding> bindings;
  MetalParameterTable parameters;
  std::vector<std::byte> parameter_data;
  std::vector<Patch> patches;
  U64 sampling_offset = 0;
  std::optional<MetalIcbProfile> icb;
  std::string numeric_contract_id = "legacy_route_v1";
  std::string sampling_contract_id = "legacy_rejection_v1";
  std::string attention_numeric_contract_id = "legacy_attention_v1";
  bool attention_row_exact_eligible = false;
  std::string gdn_numeric_contract_id = "gdn.sequential.simd32.f32_state.conv_bf16.v1";
  bool gdn_row_exact_eligible = false;
  bool fused_final_norm_t1 = false;
  bool used_w4a8 = false;
  bool used_sage2 = false;
  bool used_minference = false;
  U32 minf_context_envelope = 0;
};

inline void ensure_profile_icb(const MetalContext &context, Profile &profile) {
  if (context.capabilities().profile.compute_icb_eligible &&
      !profile.icb.has_value()) {
    profile.icb.emplace(context.prepare_icb_profile(
        profile.plan.key(), {.plan = &profile.plan,
                             .bindings = profile.bindings,
                             .parameters = &profile.parameters}));
  }
}

// Installation-time lowering of explicit Qwen calls into immutable Metal plans.
// No artifact lookup, pipeline compilation or route selection occurs in run().
class Encoder {
public:
  struct GroupSumKey {
    const MetalBuffer *sums_buffer = nullptr;
    U64 sums_offset = 0;
    const MetalBuffer *input_buffer = nullptr;
    U64 input_offset = 0;
    U32 cols = 0;
    U32 group_size = 0;
    U32 capacity_rows = 0;
    bool dynamic_rows = false;
    U32 multiplier = 1;
    bool fwht64 = false;
    bool operator==(const GroupSumKey &) const = default;
  };
  struct F16CastKey {
    const MetalBuffer *dst_buffer = nullptr;
    U64 dst_offset = 0;
    const MetalBuffer *scale_buffer = nullptr;
    U64 scale_offset = 0;
    const MetalBuffer *input_buffer = nullptr;
    U64 input_offset = 0;
    U32 cols = 0;
    U32 capacity_rows = 0;
    bool dynamic_rows = false;
    bool fwht64 = false;
    bool operator==(const F16CastKey &) const = default;
  };
  struct A8QuantKey {
    const MetalBuffer *input_buffer = nullptr;
    U64 input_offset = 0;
    const MetalBuffer *qa_buffer = nullptr;
    U64 qa_offset = 0;
    const MetalBuffer *sa_buffer = nullptr;
    U64 sa_offset = 0;
    const MetalBuffer *ra_buffer = nullptr;
    U64 ra_offset = 0;
    U32 k = 0;
    U32 rows = 0;
    U32 mp = 0;
    bool dynamic_tokens = false;
    bool fwht64 = false;
    bool operator==(const A8QuantKey &) const = default;
  };

  Encoder(const MetalContext &context, const std::filesystem::path &directory,
          U32 capacity, std::string prefix = "qwen.forward")
      : context(context), directory(directory) {
    profile.capacity = capacity;
    descriptor.key = prefix + ".t" + std::to_string(capacity);
    descriptor.icb_qualified =
        context.capabilities().profile.compute_icb_eligible;
  }

  void begin_stage(std::string name) {
    if (name.empty())
      throw std::invalid_argument("empty Qwen diagnostic stage");
    close_stage();
    cached_group_sum.reset();
    cached_f16_cast.reset();
    cached_a8_quant.reset();
    active_stage = std::move(name);
    stage_start = U32(descriptor.dispatches.size());
  }

  [[nodiscard]] bool has_group_sum(const GroupSumKey &key) const {
    return cached_group_sum.has_value() && *cached_group_sum == key;
  }

  void set_group_sum(GroupSumKey key) { cached_group_sum = std::move(key); }

  [[nodiscard]] bool has_f16_cast(const F16CastKey &key) const {
    return cached_f16_cast.has_value() && *cached_f16_cast == key;
  }

  void set_f16_cast(F16CastKey key) { cached_f16_cast = std::move(key); }

  [[nodiscard]] bool has_a8_quant(const A8QuantKey &key) const {
    return cached_a8_quant.has_value() && *cached_a8_quant == key;
  }

  void set_a8_quant(A8QuantKey key) { cached_a8_quant = std::move(key); }

  void clear_a8_quant() { cached_a8_quant.reset(); }

  void clear_dynamic_parameter_cache() {
    dynamic_parameters.clear();
    cached_group_sum.reset();
    cached_f16_cast.reset();
    cached_a8_quant.reset();
  }

  void record_weight(const MaterializedWeightView &weight, U32 n, U32 k,
                     std::string family, std::string source,
                     std::string fallback_reason,
                     std::string numeric_contract = "legacy_route_v1",
                     std::string exact_group = {}, bool exact_eligible = false,
                     std::optional<U32> explicit_t = {}) {
    if (descriptor.dispatches.empty() || active_stage.empty())
      throw std::logic_error("Qwen weight route has no active dispatch stage");
    profile.weight_routes.push_back(
        {active_stage, weight.name, weight.format, last_library, last_function,
         std::move(family), std::move(source), std::move(fallback_reason), n, k,
         explicit_t.value_or(profile.capacity), last_grid, last_threads});
    auto &record = profile.weight_routes.back();
    record.numeric_contract_id = std::move(numeric_contract);
    record.row_exact_group = std::move(exact_group);
    record.row_exact_eligible = exact_eligible;
  }

  MetalDispatchDescriptor &emit(std::string lib, std::string function,
                                MetalDispatchSize grid,
                                MetalDispatchSize threads,
                                std::initializer_list<Ref> refs) {
    last_library = lib;
    last_function = function;
    last_grid = grid;
    last_threads = threads;
    const auto library = "qwen.forward.library." + lib;
    context.register_library(
        {.key = library, .path = directory / ("omlx_m5_" + lib + ".metallib")});
    const auto pipeline = "qwen.forward." + function;
    const auto pipeline_info = context.warmup_pipeline(
        {.key = pipeline,
         .library_key = library,
         .function_name = function,
         .variant = "fixed-qwen",
         .support_indirect_command_buffers =
             context.capabilities().profile.compute_icb_eligible});
    MetalDispatchDescriptor command{.pipeline_key = pipeline,
                                    .grid = grid,
                                    .threads_per_threadgroup = threads};
    std::vector<const MetalBuffer *> dispatch_reads;
    std::vector<const MetalBuffer *> dispatch_writes;
    bool unknown_access = std::any_of(
        pipeline_info.bindings.begin(), pipeline_info.bindings.end(),
        [](const auto &binding) { return binding.kind != "buffer"; });
    U32 argument = 0;
    for (const auto ref : refs) {
      const auto index = argument++;
      // Q4/Q8 compile out the unused high-bit plane; the Metal adapter owns
      // zero bindings for inactive slots in the complete argument signature.
      const auto binding_it = std::find_if(
          pipeline_info.bindings.begin(), pipeline_info.bindings.end(),
          [&](const auto &binding) { return binding.index == index; });
      if (binding_it == pipeline_info.bindings.end())
        continue;
      if (!ref.buffer || !ref.bytes)
        throw std::invalid_argument("empty Qwen operand");
      if (binding_it->access == "read") {
        dispatch_reads.push_back(ref.buffer);
      } else if (binding_it->access == "write") {
        dispatch_writes.push_back(ref.buffer);
      } else {
        dispatch_reads.push_back(ref.buffer);
        dispatch_writes.push_back(ref.buffer);
        if (binding_it->access != "read_write")
          unknown_access = true;
      }
      const BindingKey bkey{ref.buffer, ref.offset};
      U32 slot = 0;
      if (const auto found = binding_slots_.find(bkey);
          found != binding_slots_.end()) {
        slot = found->second;
      } else {
        slot = static_cast<U32>(profile.bindings.size());
        binding_slots_.emplace(bkey, slot);
        profile.bindings.push_back({slot, ref.buffer, ref.offset});
      }
      command.resources.push_back({.argument_index = index,
                                   .resource_slot = slot,
                                   .minimum_bytes = ref.bytes});
    }
    bool need_barrier = false;
    if (!descriptor.dispatches.empty()) {
      need_barrier = unknown_access;
      if (!need_barrier) {
        for (const auto *buf : dispatch_reads) {
          if (written_since_barrier.contains(buf)) {
            need_barrier = true;
            break;
          }
        }
      }
      if (!need_barrier) {
        for (const auto *buf : dispatch_writes) {
          if (written_since_barrier.contains(buf) ||
              read_since_barrier.contains(buf)) {
            need_barrier = true;
            break;
          }
        }
      }
      if (need_barrier) {
        read_since_barrier.clear();
        written_since_barrier.clear();
      }
    }
    for (const auto *buf : dispatch_reads)
      read_since_barrier.insert(buf);
    for (const auto *buf : dispatch_writes)
      written_since_barrier.insert(buf);
    if (cached_group_sum) {
      for (const auto *buf : dispatch_writes) {
        if (buf == cached_group_sum->input_buffer ||
            buf == cached_group_sum->sums_buffer) {
          cached_group_sum.reset();
          break;
        }
      }
    }
    if (cached_f16_cast) {
      for (const auto *buf : dispatch_writes) {
        if (buf == cached_f16_cast->input_buffer ||
            buf == cached_f16_cast->dst_buffer ||
            buf == cached_f16_cast->scale_buffer) {
          cached_f16_cast.reset();
          break;
        }
      }
    }
    if (cached_a8_quant) {
      for (const auto *buf : dispatch_writes) {
        if (buf == cached_a8_quant->input_buffer ||
            buf == cached_a8_quant->qa_buffer ||
            buf == cached_a8_quant->sa_buffer ||
            buf == cached_a8_quant->ra_buffer) {
          cached_a8_quant.reset();
          break;
        }
      }
    }
    command.barrier_before = need_barrier;
    descriptor.dispatches.push_back(std::move(command));
    return descriptor.dispatches.back();
  }

  void constant(MetalDispatchDescriptor &command, U32 argument,
                std::vector<std::byte> value) {
    command.constant_arguments.push_back({argument, std::move(value)});
  }

  U64 dynamic(MetalDispatchDescriptor &command, U32 argument,
              std::vector<std::byte> value, std::optional<U32> tokens_word = {},
              std::optional<U32> position_word = {}, U32 multiplier = 1) {
    const auto key =
        std::make_tuple(value, tokens_word, position_word, multiplier);
    if (const auto found = dynamic_parameters.find(key);
        found != dynamic_parameters.end()) {
      command.dynamic_arguments.push_back(
          {argument, found->second, value.size()});
      return found->second;
    }
    const U64 offset = (profile.parameter_data.size() + 255) / 256 * 256;
    dynamic_parameters.emplace(key, offset);
    profile.parameter_data.resize(offset + value.size());
    std::copy(value.begin(), value.end(),
              profile.parameter_data.begin() + offset);
    command.dynamic_arguments.push_back({argument, offset, value.size()});
    if (tokens_word)
      profile.patches.push_back({offset, *tokens_word, false, multiplier});
    if (position_word)
      profile.patches.push_back({offset, *position_word, true});
    return offset;
  }

  Profile finish(bool prepare_icb = false) {
    close_stage();
    descriptor.dispatches.front().barrier_before = false;
    descriptor.ranges.push_back(
        {"full", 0, static_cast<U32>(descriptor.dispatches.size())});
    for (const auto &stage : profile.stages)
      descriptor.ranges.push_back({stage.name, stage.location, stage.length});
    profile.plan = context.prepare_command_plan(std::move(descriptor));
    profile.parameters = context.make_parameter_table(
        {.key = profile.plan.key() + ".parameters",
         .byte_size = profile.parameter_data.size(),
         .fields = {{"step", 0, profile.parameter_data.size(), 256}}});
    profile.parameters.write("step", profile.parameter_data);
    if (prepare_icb)
      ensure_profile_icb(context, profile);
    return std::move(profile);
  }

  Profile profile;
  bool delegated_prefill = false;

private:
  struct BindingKey {
    const MetalBuffer *buffer = nullptr;
    U64 offset = 0;
    bool operator==(const BindingKey &) const = default;
  };
  struct BindingKeyHash {
    std::size_t operator()(const BindingKey &k) const noexcept {
      const auto h1 = std::hash<const MetalBuffer *>{}(k.buffer);
      const auto h2 = std::hash<U64>{}(k.offset);
      return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6U) + (h1 >> 2U));
    }
  };

  void close_stage() {
    if (active_stage.empty())
      return;
    const U32 end = U32(descriptor.dispatches.size());
    if (end == stage_start)
      throw std::logic_error("Qwen diagnostic stage has no dispatch");
    profile.stages.push_back(
        {std::move(active_stage), stage_start, end - stage_start});
    active_stage.clear();
  }

  const MetalContext &context;
  std::string active_stage, last_library, last_function;
  MetalDispatchSize last_grid, last_threads;
  U32 stage_start = 0;
  const std::filesystem::path &directory;
  MetalCommandPlanDescriptor descriptor;
  std::unordered_map<BindingKey, U32, BindingKeyHash> binding_slots_;
  std::map<std::tuple<std::vector<std::byte>, std::optional<U32>,
                      std::optional<U32>, U32>,
           U64>
      dynamic_parameters;
  std::unordered_set<const MetalBuffer *> read_since_barrier;
  std::unordered_set<const MetalBuffer *> written_since_barrier;
  std::optional<GroupSumKey> cached_group_sum;
  std::optional<F16CastKey> cached_f16_cast;
  std::optional<A8QuantKey> cached_a8_quant;
};
struct ForwardLinearSelection {
  std::string library, function, group_function, family;
  U32 bits = 0, group = 0;
  bool matrix = false, lane = false, broad = false;
  bool row_exact = false;
  std::string numeric_contract_id = "legacy_route_v1";
  std::string row_exact_group = {};
};
struct ForwardLinearDecision {
  ForwardLinearSelection route;
  std::string source, fallback_reason;
  bool override_matched = false;
};

inline MetalDispatchSize linear_grid(const ForwardLinearSelection &route,
                                     std::string_view format, U32 n, U32 t) {
  if (route.matrix)
    return {.width = U64((n + 31) / 32) * 128};
  return {.width = U64((n + 3) / 4) * 128,
          .height = route.row_exact || format == "direct_bf16_v1" ||
                            route.bits == 3
                        ? t
                    : route.broad ? U64((t + 15) / 16)
                                  : 1};
}

inline U32 small_t_bucket(U32 t) {
  if (t <= 2)
    return 2;
  if (t <= 4)
    return 4;
  if (t <= 8)
    return 8;
  return 12;
}

// This selector is shared by production lowering and the metadata-only route
// inspector. It never materializes weight planes or probes a new runtime path.
inline ForwardLinearSelection
select_linear_route(std::string_view format, U32 n, U32 k, U32 t,
                    bool candidate_t1 = false, bool candidate_t8 = false,
                    bool candidate_matrix = false) {
  if (!n || !k || !t)
    throw std::invalid_argument("Qwen Linear route geometry is empty");
  ForwardLinearSelection route;
  if (format == "direct_bf16_v1") {
    route.library = "qwen_forward";
    route.function = "omlx_m5_forward_bf16_linear";
    route.family = "direct_bf16";
    return route;
  }
  if (format == "mxfp4_e2m1_g32_e8m0_v1") {
    if ((t == 1 || t == 8) && candidate_matrix) {
      route.library = "quantized_matrix";
      route.function = "omlx_m5_matrix_q0_g32";
      route.family = "simdgroup_matrix";
      route.matrix = true;
    } else if (t == 1) {
      route.library = "qmv_t1";
      route.function = candidate_t1 ? "omlx_m5_mxfp4_qmv_t1_lane_accum"
                                    : "omlx_m5_mxfp4_qmv_t1";
      route.family = candidate_t1 ? "lane_accum" : "qmv_t1";
      route.lane = candidate_t1;
    } else if (t <= 12) {
      const U32 bucket = small_t_bucket(t);
      route.library = "mxfp4_small_t";
      route.lane = candidate_t8 && t == 8;
      route.function =
          route.lane ? "omlx_m5_mxfp4_w4a16_c8_lane_accum"
                     : "omlx_m5_mxfp4_w4a16_c" + std::to_string(bucket);
      route.family = route.lane ? "lane_accum" : "w4a16_small_t";
    } else {
      route.library = "qmm_broad";
      route.function = "omlx_m5_mxfp4_w4a16_broad_t16";
      route.family = "w4a16_broad";
      route.broad = true;
    }
    return route;
  }
  if (!format.starts_with("oq_affine_q"))
    throw std::invalid_argument("Qwen Linear format unsupported: " +
                                std::string(format));
  const auto group_offset = format.find("_g");
  if (group_offset == std::string_view::npos)
    throw std::invalid_argument("Qwen Linear group absent");
  route.bits = U32(std::stoul(std::string(format.substr(11))));
  route.group = U32(std::stoul(std::string(format.substr(group_offset + 2))));
  if ((route.bits != 3 && route.bits != 4 && route.bits != 5 &&
       route.bits != 6 && route.bits != 8) ||
      (route.group != 64 && route.group != 128) || k % route.group)
    throw std::invalid_argument("Qwen affine route geometry unsupported");
  if (route.bits == 3) {
    route.library = "qwen_forward";
    route.function =
        "omlx_m5_forward_q3_g" + std::to_string(route.group) + "_linear";
    route.family = "q3_simd32";
    route.row_exact = true;
    route.numeric_contract_id = "linear_row_exact_portable_v1";
    route.row_exact_group =
        "oq.simd32.groupdot_scale_then_bias.v1/" + std::string(format);
    return route;
  }
  const std::string suffix =
      "q" + std::to_string(route.bits) + "_g" + std::to_string(route.group);
  if ((t == 1 || t == 8) && candidate_matrix) {
    route.library = "quantized_matrix";
    route.function = "omlx_m5_matrix_" + suffix;
    route.family = "simdgroup_matrix";
    route.matrix = true;
  } else if (t == 1) {
    route.library = "qmv_t1";
    route.group_function =
        "omlx_m5_activation_group_sum_g" + std::to_string(route.group);
    route.function = "omlx_m5_oq_" + suffix + "_qmv_t1";
    route.lane = candidate_t1;
    if (route.lane)
      route.function += "_lane_accum";
    route.family = route.lane ? "lane_accum" : "qmv_t1";
  } else if (t <= 12) {
    const U32 bucket = small_t_bucket(t);
    route.library = "qmm_small_t";
    route.group_function =
        "omlx_m5_activation_group_sum_small_t_g" + std::to_string(route.group);
    if (bucket == 12) {
      route.function = "omlx_m5_oq_" + suffix + "_coop_c12";
      route.family = "cooperative";
    } else {
      route.function =
          "omlx_m5_oq_" + suffix + "_row_c" + std::to_string(bucket);
      route.lane = candidate_t8 && t == 8;
      if (route.lane)
        route.function += "_lane_accum";
      route.family = route.lane ? "lane_accum" : "row_split";
    }
  } else {
    route.library = "qmm_broad";
    route.group_function =
        "omlx_m5_broad_group_sum_g" + std::to_string(route.group);
    route.function = "omlx_m5_oq_" + suffix + "_broad_t16";
    route.family = "broad";
    route.broad = true;
  }
  return route;
}

inline ForwardLinearSelection select_row_exact_linear_route(
    std::string_view format, U32 n, U32 k, U32 capacity) {
  // Validate the same physical format/geometry as the production selector.
  auto route = select_linear_route(format, n, k, capacity);
  if (route.bits == 3)
    return route;
  route.library = "row_exact";
  route.family = "row_exact_portable";
  route.matrix = route.lane = route.broad = false;
  route.row_exact = true;
  route.numeric_contract_id = "linear_row_exact_portable_v1";
  if (format == "direct_bf16_v1") {
    route.function = "omlx_m5_row_exact_bf16";
    route.row_exact_group = "bf16.simd32.sequential_fma.v1";
  } else if (format == "mxfp4_e2m1_g32_e8m0_v1") {
    if (k % 32) throw std::invalid_argument("row-exact MXFP4 K alignment");
    route.function = "omlx_m5_row_exact_mxfp4";
    route.row_exact_group = "mxfp4.simd32.groupdot_scale_fma.v1";
  } else {
    const std::string canonical = "oq_affine_q" + std::to_string(route.bits) +
        "_g" + std::to_string(route.group) + "_bf16_v1";
    if (format != canonical)
      throw std::invalid_argument("row-exact affine format version unsupported");
    route.function = "omlx_m5_row_exact_q" + std::to_string(route.bits) +
                     "_g" + std::to_string(route.group);
    route.group_function = "omlx_m5_row_exact_group_sum_g" +
                           std::to_string(route.group);
    route.row_exact_group = "oq.simd32.groupdot_scale_then_bias.v1/" +
                            std::string(format);
  }
  return route;
}

inline bool row_exact_routes_compatible(const ForwardLinearSelection &a,
                                       const ForwardLinearSelection &b,
                                       U32 a_k, U32 b_k) {
  return a.row_exact && b.row_exact && a_k == b_k &&
         a.numeric_contract_id == b.numeric_contract_id &&
         !a.row_exact_group.empty() && a.row_exact_group == b.row_exact_group;
}

// The one policy decision used by metadata inspection and profile lowering.
// Probe candidates outside the qualified table stay in isolated benchmarks.
inline ForwardLinearDecision resolve_linear_route(
    const MaterializedWeightView &weight, U32 n, U32 k, U32 capacity,
    std::string_view device_name, bool enable_m4_short_routes,
    const std::optional<QwenLinearRouteOverride> &diagnostic_override = {},
    LinearNumericPolicy numeric_policy = LinearNumericPolicy::Legacy,
    std::optional<U32> apple_gpu_family = {}) {
  if (numeric_policy != LinearNumericPolicy::Legacy &&
      numeric_policy != LinearNumericPolicy::RowExactPortableV1)
    throw std::invalid_argument("unknown Linear numeric policy");
  if (numeric_policy == LinearNumericPolicy::RowExactPortableV1) {
    if (diagnostic_override)
      throw std::invalid_argument("row-exact policy rejects legacy A/B overrides");
    return {select_row_exact_linear_route(weight.format, n, k, capacity),
            "row_exact_portable_v1", "legacy_short_routes_not_row_exact", false};
  }
  const bool override_supported =
      apple_gpu_family ? (*apple_gpu_family >= 9)
                       : (device_name.starts_with("Apple M4") ||
                          device_name.starts_with("Apple M5"));
  if (diagnostic_override && !override_supported)
    throw std::invalid_argument("linear A/B override requires Apple GPU family 9+");
  const bool matched =
      diagnostic_override && diagnostic_override->object == weight.name &&
      diagnostic_override->format == weight.format &&
      diagnostic_override->n == n && diagnostic_override->k == k &&
      diagnostic_override->capacity == capacity;
  if (matched) {
    const auto &choice = diagnostic_override->choice;
    if (weight.format == "direct_bf16_v1" ||
        (choice != "baseline" && choice != "lane" && choice != "matrix") ||
        (choice == "lane" && capacity != 1 && capacity != 8) ||
        (choice == "matrix" && capacity != 1 && capacity != 8))
      throw std::invalid_argument("unsupported exact linear A/B candidate");
    auto route = select_linear_route(
        weight.format, n, k, capacity, choice == "lane" && capacity == 1,
        choice == "lane" && capacity == 8, choice == "matrix");
    return {std::move(route), "diagnostic_exact_override", {}, true};
  }
  const bool m4 = device_name == "Apple M4";
  const bool lane = m4 && enable_m4_short_routes &&
                    qualified_m4_lane_t1(weight, n, k, capacity);
  const bool matrix = m4 && enable_m4_short_routes &&
                      qualified_m4_matrix_t8(weight, n, k, capacity);
  auto route =
      select_linear_route(weight.format, n, k, capacity, lane, false, matrix);
  if (lane || matrix)
    return {std::move(route), "m4_qualified_shape", {}, false};
  return {std::move(route), "baseline",
          !m4                       ? "device_unqualified"
          : !enable_m4_short_routes ? "m4_short_routes_disabled"
                                    : "shape_or_width_unqualified",
          false};
}

inline std::string select_grouped_expert_kernel(std::string_view format) {
  if (format == "mxfp4_e2m1_g32_e8m0_v1")
    return "omlx_m5_mxfp4_moe_w4a16_grouped";
  if (!format.starts_with("oq_affine_q"))
    throw std::invalid_argument("Qwen expert format unsupported");
  const auto group_offset = format.find("_g");
  if (group_offset == std::string_view::npos)
    throw std::invalid_argument("Qwen expert group absent");
  const U32 bits = U32(std::stoul(std::string(format.substr(11))));
  const U32 group =
      U32(std::stoul(std::string(format.substr(group_offset + 2))));
  if ((bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
      (group != 64 && group != 128))
    throw std::invalid_argument("Qwen expert format unsupported");
  return "omlx_m5_oq_q" + std::to_string(bits) + "_g" + std::to_string(group) +
         "_moe_grouped";
}

inline void linear(Encoder &e, const MaterializedWeightView &w, Ref x, Ref y,
                   U32 expected_n, U32 expected_k, Ref sums, Ref zero,
                   const ForwardLinearSelection &route,
                   std::optional<U32> explicit_t = {}) {
  if (w.parent_shape.size() != 2 || w.row_count != expected_n ||
      w.in_features != expected_k)
    throw std::invalid_argument("Qwen Linear shape mismatch: " + w.name);
  const U32 n = expected_n, k = expected_k,
            t = explicit_t.value_or(e.profile.capacity);
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3);
  if (route.row_exact)
    e.profile.numeric_contract_id = route.numeric_contract_id;

  // Bug #1 guard: pre-tiled NAX weights have no raw "data"/"base_code" planes.
  // When running on non-NAX hosts (e.g. M4 dev CTests) or fallback paths,
  // execute portable SIMD32 kernels directly over the pre-tiled planes.
  if (w.layout == "mx_nax_tiled_bk128_v1") {
    const MetalDispatchSize pgrid{.width = U64((n + 3) / 4) * 128, .height = t};
    auto &d = e.emit("qwen_forward", "omlx_m5_forward_pretiled_mxfp4_linear",
                     pgrid, {.width = 128},
                     {x, plane(w, "nax_data"), plane(w, "nax_scale"), y});
    const U32 pk_flags =
        U32(w.padded_k) | (w.has_fwht64() ? (1u << 31u) : 0u);
    e.dynamic(d, 4, words({n, k, pk_flags, t}), tokens_patch);
    return;
  }
  if (w.layout == "oq_nax_tiled_g64_v1" ||
      w.layout == "oq_nax_tiled_g128_v1" ||
      w.layout == "oq_nax_lane_g64_v2" ||
      w.layout == "oq_nax_lane_g128_v2") {
    if ((k % 64u) != 0u) {
      throw std::invalid_argument(
          "Qwen pre-tiled affine linear requires K divisible by 64");
    }
    const bool is_lane_v2 = (w.layout == "oq_nax_lane_g64_v2" ||
                             w.layout == "oq_nax_lane_g128_v2");
    const bool is_g128 = (w.layout == "oq_nax_tiled_g128_v1" ||
                          w.layout == "oq_nax_lane_g128_v2");
    const U32 bn = (n >= 65536u) ? 32u : 16u;
    const U32 padded_n = ((n + bn - 1u) / bn) * bn;
    const Ref sb = plane(w, "nax_sb");
    const U64 sb_bytes_g64 = mul(mul(U64(padded_n), k / 64u), 4u);
    const U64 sb_bytes_g128 =
        (k % 128u == 0u) ? mul(mul(U64(padded_n), k / 128u), 4u) : 0ULL;
    const U32 sb_group_size =
        (is_g128 && (k % 128u) == 0u &&
         sb.bytes < sb_bytes_g64 && sb.bytes >= sb_bytes_g128)
            ? 128u
            : 64u;
    const U64 min_sb_bytes =
        (sb_group_size == 128u) ? sb_bytes_g128 : sb_bytes_g64;
    if (sb.bytes < min_sb_bytes) {
      throw std::invalid_argument(
          "Qwen pre-tiled affine nax_sb plane is smaller than required");
    }
    const MetalDispatchSize pgrid{.width = U64((n + 3) / 4) * 128, .height = t};
    Ref high = route.bits == 3 ? plane(w, "nax_high1")
               : (route.bits == 5 || route.bits == 6) ? plane(w, "nax_high2")
                                                      : zero;
    const U32 bits_flags = route.bits |
                           (is_lane_v2 ? (1u << 30u) : 0u) |
                           (w.has_fwht64() ? (1u << 31u) : 0u);
    auto &d = e.emit("qwen_forward", "omlx_m5_forward_pretiled_affine_linear",
                     pgrid, {.width = 128},
                     {x, plane(w, "nax_codes"), high, sb, y});
    e.dynamic(d, 5, words({n, k, bits_flags, t, sb_group_size}), tokens_patch);
    return;
  }

  auto params = words({n, k, U32(w.padded_k), t});
  const MetalDispatchSize grid = linear_grid(route, w.format, n, t);
  if (route.matrix)
    ++e.profile.short_matrix_calls;
  if (route.lane)
    ++e.profile.short_lane_calls;
  if (w.format == "direct_bf16_v1") {
    auto &d = e.emit(route.library, route.function, grid, {.width = 128},
                     {x, plane(w, "data"), y});
    e.dynamic(d, 3, std::move(params), tokens_patch);
    return;
  }
  if (w.format == "mxfp4_e2m1_g32_e8m0_v1") {
    if (route.matrix) {
      auto &d = e.emit(route.library, route.function, grid, {.width = 128},
                       {x, plane(w, "data"), zero, plane(w, "scale"), zero, y});
      e.dynamic(d, 6, std::move(params), tokens_patch);
    } else {
      auto &d = e.emit(route.library, route.function, grid, {.width = 128},
                       {x, plane(w, "data"), plane(w, "scale"), y});
      e.dynamic(d, 4, std::move(params), tokens_patch);
    }
    return;
  }
  if (route.bits == 3) {
    auto &d = e.emit(route.library, route.function, grid, {.width = 128},
                     {x, plane(w, "low_2bit"), plane(w, "high_bit"),
                      plane(w, "scale"), plane(w, "bias"), y});
    e.dynamic(d, 6, std::move(params), tokens_patch);
    return;
  }
  const auto codes = plane(w, route.bits == 8 ? "code" : "base_code");
  const auto high = route.bits == 5   ? plane(w, "high_bit")
                    : route.bits == 6 ? plane(w, "high_2bit")
                                      : zero;
  if (route.matrix) {
    auto &d = e.emit(route.library, route.function, grid, {.width = 128},
                     {x, codes, high, plane(w, "scale"), plane(w, "bias"), y});
    e.dynamic(d, 6, std::move(params), tokens_patch);
    return;
  }
  const bool dyn_rows = !explicit_t.has_value() && t != 1;
  const Encoder::GroupSumKey sum_key{sums.buffer, sums.offset, x.buffer,
                                     x.offset,    k,           route.group,
                                     t,           dyn_rows,    1};
  if (!e.has_group_sum(sum_key)) {
    auto &sum = e.emit(route.library, route.group_function,
                       {.width = U64(t) * (k / route.group) * 32}, {.width = 32},
                       {x, sums});
    if (t != 1 || route.row_exact)
      e.dynamic(sum, 2, params, tokens_patch);
    e.set_group_sum(sum_key);
  }
  auto &d =
      e.emit(route.library, route.function, grid, {.width = 128},
             {x, codes, high, plane(w, "scale"), plane(w, "bias"), sums, y});
  e.dynamic(d, 7, std::move(params), tokens_patch);
}

// Universal BF16 FWHT-64 transform (supports both out-of-place and in-place).
inline void emit_fwht64_bf16(Encoder &e, Ref src, Ref dst, U32 k,
                             std::optional<U32> explicit_t = {},
                             U32 row_mul = 1u) {
  if ((k % 64u) != 0u || k == 0u) {
    throw std::invalid_argument("emit_fwht64_bf16 requires k % 64 == 0");
  }
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const U32 eff_mul = (row_mul > 0u) ? row_mul : 1u;
  const std::optional<U32> tokens_patch =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>(3u);
  auto &d = e.emit("qwen_forward", "omlx_m5_fwht64_bf16",
                   {.width = U64(k / 64u) * 32u, .height = U64(t) * eff_mul},
                   {.width = 32u}, {src, dst});
  e.dynamic(d, 2, words({k, eff_mul, 0u, t}), tokens_patch);
}

// NInfer-style fused row-exact gate + up + SwiGLU projection.
inline void linear_swiglu(Encoder &e, const MaterializedWeightView &gate_w,
                          const MaterializedWeightView &up_w, Ref x, Ref y,
                          U32 expected_n, U32 expected_k, Ref sums, Ref zero,
                          const ForwardLinearSelection &route) {
  if (gate_w.parent_shape.size() != 2 || up_w.parent_shape.size() != 2 ||
      gate_w.row_count != expected_n || gate_w.in_features != expected_k ||
      up_w.row_count != expected_n || up_w.in_features != expected_k ||
      gate_w.format != up_w.format || gate_w.padded_k != up_w.padded_k ||
      !route.row_exact || gate_w.format == "direct_bf16_v1") {
    throw std::invalid_argument("Qwen fused SwiGLU Linear geometry mismatch");
  }
  if (gate_w.layout == "mx_nax_tiled_bk128_v1" ||
      gate_w.layout == "oq_nax_tiled_g64_v1" ||
      gate_w.layout == "oq_nax_tiled_g128_v1" ||
      gate_w.layout == "oq_nax_lane_g64_v2" ||
      gate_w.layout == "oq_nax_lane_g128_v2" || route.bits == 3) {
    throw std::invalid_argument("pre-tiled/q3 SwiGLU requires separate or NAX route");
  }
  const U32 n = expected_n, k = expected_k, t = e.profile.capacity;
  e.profile.numeric_contract_id = route.numeric_contract_id;
  auto params = words({n, k, U32(gate_w.padded_k), t});
  const MetalDispatchSize grid = linear_grid(route, gate_w.format, n, t);
  const std::string fn = route.function + "_swiglu";
  if (gate_w.format == "mxfp4_e2m1_g32_e8m0_v1") {
    auto &d = e.emit(route.library, fn, grid, {.width = 128},
                     {x, plane(gate_w, "data"), plane(gate_w, "scale"),
                      plane(up_w, "data"), plane(up_w, "scale"), y});
    e.dynamic(d, 6, std::move(params), 3);
    return;
  }
  const auto g_codes = plane(gate_w, route.bits == 8 ? "code" : "base_code");
  const auto g_high = route.bits == 5   ? plane(gate_w, "high_bit")
                      : route.bits == 6 ? plane(gate_w, "high_2bit")
                                        : zero;
  const auto u_codes = plane(up_w, route.bits == 8 ? "code" : "base_code");
  const auto u_high = route.bits == 5   ? plane(up_w, "high_bit")
                      : route.bits == 6 ? plane(up_w, "high_2bit")
                                        : zero;
  const bool dyn_rows = (t != 1);
  const Encoder::GroupSumKey sum_key{sums.buffer, sums.offset, x.buffer,
                                     x.offset,    k,           route.group,
                                     t,           dyn_rows,    1};
  if (!e.has_group_sum(sum_key)) {
    auto &sum = e.emit(route.library, route.group_function,
                       {.width = U64(t) * (k / route.group) * 32}, {.width = 32},
                       {x, sums});
    e.dynamic(sum, 2, params, 3);
    e.set_group_sum(sum_key);
  }
  auto &d = e.emit(route.library, fn, grid, {.width = 128},
                   {x, g_codes, g_high, plane(gate_w, "scale"),
                    plane(gate_w, "bias"), u_codes, u_high,
                    plane(up_w, "scale"), plane(up_w, "bias"), sums, y});
  e.dynamic(d, 11, std::move(params), 3);
}

// Isolated operator probes explicitly use the old baseline without installing
// a device policy. Production target/draft call the preselected-route overload.
inline void linear(Encoder &e, const MaterializedWeightView &w, Ref x, Ref y,
                   U32 n, U32 k, Ref sums, Ref zero) {
  linear(e, w, x, y, n, k, sums, zero,
         select_linear_route(w.format, n, k, e.profile.capacity));
}

inline void embedding_lookup(Encoder &e,
                             const MaterializedWeightView &embedding,
                             Ref token_ids, Ref hidden, Ref zero, U32 v,
                             U32 h, std::optional<U32> explicit_t = {}) {
  const U32 t = explicit_t.value_or(e.profile.capacity);
  const std::optional<U32> tok_word =
      explicit_t.has_value() ? std::nullopt : std::optional<U32>{5U};
  std::string function = "omlx_m5_embedding_";
  std::vector<std::byte> embed_params =
      words({v, 0, v, h, U32(embedding.padded_k), t, 0, 0});
  if (embedding.format == "direct_bf16_v1") {
    auto &d =
        e.emit("vocab", function + "bf16", {.width = U64(t) * h},
               {.width = 256}, {token_ids, plane(embedding, "data"), hidden});
    e.dynamic(d, 3, embed_params, tok_word);
  } else if (embedding.format == "mxfp4_e2m1_g32_e8m0_v1") {
    auto &d = e.emit("vocab", function + "mxfp4", {.width = U64(t) * h},
                     {.width = 256},
                     {token_ids, plane(embedding, "data"),
                      plane(embedding, "scale"), hidden});
    e.dynamic(d, 4, embed_params, tok_word);
  } else if (embedding.format.starts_with("oq_affine_q")) {
    U32 bits = std::stoul(embedding.format.substr(11));
    U32 group =
        std::stoul(embedding.format.substr(embedding.format.find("_g") + 2));
    if (bits == 3) {
      auto &d = e.emit("qwen_forward",
                       "omlx_m5_forward_embedding_q3_g" + std::to_string(group),
                       {.width = U64(t) * h}, {.width = 256},
                       {token_ids, plane(embedding, "low_2bit"),
                        plane(embedding, "high_bit"), plane(embedding, "scale"),
                        plane(embedding, "bias"), hidden});
      e.dynamic(d, 6, embed_params, tok_word);
      return;
    }
    Ref high = bits == 5   ? plane(embedding, "high_bit")
               : bits == 6 ? plane(embedding, "high_2bit")
                           : zero;
    auto &d = e.emit(
        "vocab",
        function + "q" + std::to_string(bits) + "_g" + std::to_string(group),
        {.width = U64(t) * h}, {.width = 256},
        {token_ids, plane(embedding, bits == 8 ? "code" : "base_code"), high,
         plane(embedding, "scale"), plane(embedding, "bias"), hidden});
    e.dynamic(d, 6, embed_params, tok_word);
  } else
    throw std::invalid_argument("Qwen embedding format unsupported");
}
} // namespace omlx::m5::detail::forward
