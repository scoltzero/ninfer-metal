#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/materialization.h"

namespace omlx::m5 {

enum class ActivationPolicy {
  a16_only,
  allow_a4,
};

enum class ActivationProfile {
  a16,
  a4,
};

[[nodiscard]] std::string_view to_string(ActivationPolicy policy);
[[nodiscard]] std::string_view to_string(ActivationProfile profile);
[[nodiscard]] ActivationPolicy parse_activation_policy(std::string_view policy);
[[nodiscard]] bool allows_a4(ActivationPolicy policy);

struct ActivationUse {
  std::string binding_name;
  std::string role;
  ActivationPolicy policy = ActivationPolicy::a16_only;
  std::optional<std::uint32_t> expert;
};

[[nodiscard]] ActivationUse
make_activation_use(const ArtifactBinding &binding,
                    std::optional<std::uint32_t> expert = std::nullopt);
[[nodiscard]] ActivationPolicy
common_activation_policy(std::span<const ActivationUse> uses);
[[nodiscard]] ActivationProfile
resolve_activation_profile(ActivationPolicy policy,
                           ActivationProfile preferred);

struct ActivationReuseKey {
  std::uint64_t source_id = 0;
  std::uint64_t normalization_id = 0;
  std::uint32_t input_features = 0;
  std::uint32_t tokens = 0;

  bool operator==(const ActivationReuseKey &) const = default;
};

struct ActivationSlotSpec {
  std::uint32_t max_input_features = 0;
  std::uint32_t max_tokens = 0;
  ActivationPolicy policy = ActivationPolicy::a16_only;

  bool operator==(const ActivationSlotSpec &) const = default;
};

struct ActivationWorkspacePlan {
  std::uint64_t alignment = 256;
  std::uint64_t capacity_bytes = 0;
  std::uint64_t a16_bytes = 0;
  std::uint64_t a4_code_bytes = 0;
  std::uint64_t a4_scale_bytes = 0;
  std::uint64_t padding_bytes = 0;
  std::vector<ActivationSlotSpec> slots;
};

[[nodiscard]] ActivationWorkspacePlan
plan_activation_workspace(std::span<const ActivationSlotSpec> slots,
                          std::uint64_t alignment = 256);

struct ActivationPlaneView {
  MetalBuffer *buffer = nullptr;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};

struct Mxfp4ActivationView {
  ActivationPlaneView codes;
  ActivationPlaneView scales;
  std::uint32_t group_size = 32;
};

struct ActivationBundle {
  std::uint64_t epoch = 0;
  ActivationReuseKey key;
  ActivationPolicy common_policy = ActivationPolicy::a16_only;
  ActivationProfile profile = ActivationProfile::a16;
  std::optional<ActivationPlaneView> normalized_a16;
  std::optional<Mxfp4ActivationView> a4;
};

struct ActivationLease {
  ActivationBundle bundle;
  bool produce_a16 = false;
  bool produce_a4 = false;
};

struct ActivationWorkspaceStats {
  std::uint64_t capacity_bytes = 0;
  std::uint64_t used_bytes = 0;
  std::uint64_t peak_bytes = 0;
  std::uint64_t epoch = 0;
  std::uint64_t epoch_count = 0;
  std::uint64_t slot_count = 0;
  std::uint64_t current_entries = 0;
  std::uint64_t acquire_calls = 0;
  std::uint64_t consumer_uses = 0;
  std::uint64_t cache_hits = 0;
  std::uint64_t policy_fallbacks = 0;
  std::uint64_t a16_materializations = 0;
  std::uint64_t a4_materializations = 0;
  std::uint64_t avoided_a16_materializations = 0;
  std::uint64_t avoided_a4_quantizations = 0;
  std::uint64_t cross_projection_reuses = 0;
  std::uint64_t cross_expert_reuses = 0;
  std::uint64_t a16_bytes = 0;
  std::uint64_t a4_code_bytes = 0;
  std::uint64_t a4_scale_bytes = 0;
};

class ActivationWorkspace final {
public:
  ActivationWorkspace(const MetalContext &context,
                      ActivationWorkspacePlan plan);
  ActivationWorkspace(ActivationWorkspace &&) noexcept;
  ActivationWorkspace &operator=(ActivationWorkspace &&) noexcept;
  ActivationWorkspace(const ActivationWorkspace &) = delete;
  ActivationWorkspace &operator=(const ActivationWorkspace &) = delete;
  ~ActivationWorkspace();

  void begin_epoch(std::uint64_t epoch);
  [[nodiscard]] ActivationLease acquire(const ActivationReuseKey &key,
                                        std::span<const ActivationUse> uses,
                                        ActivationProfile preferred,
                                        bool also_materialize_a16 = false);
  [[nodiscard]] const ActivationWorkspacePlan &plan() const;
  [[nodiscard]] ActivationWorkspaceStats stats() const;
  [[nodiscard]] MetalBuffer &buffer();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5
