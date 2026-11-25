#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace omlx::m5 {

inline constexpr std::uint64_t kPayloadAlignment = 16384;
inline constexpr std::uint64_t kLegacyPayloadAlignment = 4096;

struct ArtifactPlane {
  std::string kind;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 1;
  std::uint32_t crc32 = 0;
  std::string sha256;
};

struct ArtifactObject {
  std::uint32_t id = 0;
  std::string name;
  std::string format;
  std::string layout;
  std::vector<std::uint64_t> shape;
  std::uint64_t padded_k = 0;
  std::uint64_t object_offset = 0;
  std::uint64_t object_bytes = 0;
  std::string object_sha256;
  std::vector<ArtifactPlane> planes;
  std::vector<std::string> source_tensors;
};

struct ArtifactBinding {
  std::string name;
  std::uint32_t object_id = 0;
  std::string format;
  std::string use;
  std::string activation_policy;
  std::vector<std::uint64_t> leading_shape;
  std::uint64_t out_features = 0;
  std::uint64_t in_features = 0;
};

struct ArtifactUploadSpan {
  std::uint32_t object_id = 0;
  std::string object_name;
  std::string plane_kind;
  std::uint64_t source_file_offset = 0;
  std::uint64_t destination_offset = 0;
  std::uint64_t bytes = 0;
};

struct ArtifactMaterializationPlan {
  std::uint64_t alignment = 0;
  std::uint64_t payload_bytes = 0;
  std::uint64_t upload_bytes = 0;
  std::uint64_t arena_bytes = 0;
  std::uint64_t arena_padding_bytes = 0;
  std::vector<ArtifactUploadSpan> spans;
};

struct ArtifactSummary {
  std::uint32_t container_version = 0;
  std::uint64_t artifact_bytes = 0;
  std::uint64_t manifest_bytes = 0;
  std::uint64_t payload_offset = 0;
  std::uint64_t payload_bytes = 0;
  std::string source_repo;
  std::string source_revision;
  std::string source_profile;
  bool payload_verified = false;
};

class ArtifactModel final {
public:
  static ArtifactModel open(const std::filesystem::path &path,
                            bool verify_payload = false);

  ArtifactModel(ArtifactModel &&) noexcept;
  ArtifactModel &operator=(ArtifactModel &&) noexcept;
  ArtifactModel(const ArtifactModel &) = delete;
  ArtifactModel &operator=(const ArtifactModel &) = delete;
  ~ArtifactModel();

  [[nodiscard]] const std::filesystem::path &path() const;
  [[nodiscard]] const ArtifactSummary &summary() const;
  void verify_payload();
  [[nodiscard]] const std::vector<ArtifactObject> &objects() const;
  [[nodiscard]] const std::vector<ArtifactBinding> &bindings() const;
  [[nodiscard]] const ArtifactObject &object(std::uint32_t id) const;
  [[nodiscard]] const ArtifactObject &object(std::string_view name) const;
  [[nodiscard]] const ArtifactPlane &plane(const ArtifactObject &object,
                                           std::string_view kind) const;
  [[nodiscard]] std::span<const std::byte>
  plane_bytes(const ArtifactObject &object, const ArtifactPlane &plane) const;
  [[nodiscard]] ArtifactMaterializationPlan
  materialization_plan(std::uint64_t alignment = 256) const;
  [[nodiscard]] std::vector<float>
  decode_weight_row(const ArtifactObject &object, std::uint64_t row,
                    std::uint64_t start, std::uint64_t count) const;

private:
  class Impl;
  explicit ArtifactModel(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::uint32_t artifact_crc32(std::span<const std::byte> bytes,
                                           std::uint32_t initial = 0);
[[nodiscard]] std::string artifact_sha256_hex(std::span<const std::byte> bytes);

} // namespace omlx::m5
