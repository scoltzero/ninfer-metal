#include "omlx_m5/execution_contract.h"
#include "execution_build_identity.h"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <span>
#include <stdexcept>
namespace omlx::m5 {
namespace {
std::string digest(std::string_view value) {
  return artifact_sha256_hex(std::as_bytes(std::span(value.data(), value.size())));
}
void frame(std::string &out, std::string_view value) {
  out += std::to_string(value.size()); out += ':'; out.append(value);
}
std::string numbers(std::span<const std::uint64_t> values) {
  std::string out; for (auto value : values) frame(out, std::to_string(value)); return out;
}
}
std::string canonical_execution_identity(const ExecutionIdentityFields &fields) {
  if (fields.empty()) throw std::invalid_argument("empty execution identity");
  std::string out = "omlx.execution.identity.v1|";
  for (const auto &[key, value] : fields) {
    if (key.empty() || key.size() > 256 || key.find('\0') != std::string::npos || value.empty())
      throw std::invalid_argument("execution identity field invalid");
    frame(out, key); frame(out, value);
  }
  return out;
}
std::string execution_identity_digest(const ExecutionIdentityFields &fields) {
  return digest(canonical_execution_identity(fields));
}
std::string execution_file_digest(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::invalid_argument("execution identity file unavailable: " + path.string());
  const auto end = stream.tellg();
  if (end < 0) throw std::runtime_error("execution identity file read failed");
  std::string bytes(static_cast<std::size_t>(end), '\0');
  stream.seekg(0);
  if (!bytes.empty() && !stream.read(bytes.data(), end))
    throw std::runtime_error("execution identity file read failed");
  return digest(bytes);
}
std::string execution_kernel_bundle_digest(const std::filesystem::path &directory) {
  ExecutionIdentityFields files;
  for (const auto &entry : std::filesystem::directory_iterator(directory))
    if (entry.is_regular_file() && entry.path().extension() == ".metallib")
      files.emplace(entry.path().filename().string(), execution_file_digest(entry.path()));
  if (files.empty()) throw std::invalid_argument("execution kernel bundle has no metallibs");
  return execution_identity_digest(files);
}
std::string execution_artifact_digest(const ArtifactModel &artifact) {
  if (!artifact.summary().payload_verified)
    throw std::invalid_argument("execution artifact identity requires verified payload");
  ExecutionIdentityFields fields{{"source_repo", artifact.summary().source_repo.empty() ? "none" : artifact.summary().source_repo},
    {"source_revision", artifact.summary().source_revision.empty() ? "none" : artifact.summary().source_revision},
    {"source_profile", artifact.summary().source_profile.empty() ? "none" : artifact.summary().source_profile},
    {"container_version", std::to_string(artifact.summary().container_version)}};
  for (const auto &object : artifact.objects()) {
    ExecutionIdentityFields data{{"name",object.name},{"format",object.format},{"layout",object.layout},
      {"shape",numbers(object.shape)},{"padded_k",std::to_string(object.padded_k)},
      {"object_sha256",object.object_sha256}};
    for (const auto &plane : object.planes)
      data.emplace("plane."+plane.kind, canonical_execution_identity({{"sha256",plane.sha256},{"bytes",std::to_string(plane.bytes)}}));
    fields.emplace("object."+std::to_string(object.id),execution_identity_digest(data));
  }
  for (const auto &binding : artifact.bindings())
    fields.emplace("binding."+binding.name,execution_identity_digest({{"object",std::to_string(binding.object_id)},
      {"format",binding.format},{"use",binding.use.empty()?"none":binding.use},
      {"activation",binding.activation_policy.empty()?"none":binding.activation_policy},
      {"shape",binding.leading_shape.empty()?"scalar":numbers(binding.leading_shape)},
      {"n",std::to_string(binding.out_features)},{"k",std::to_string(binding.in_features)}}));
  return execution_identity_digest(fields);
}
std::string_view execution_source_id() { return OMLX_M5_EXECUTION_SOURCE_ID; }
std::string_view prefill_chunk_policy_id(SamplingContract contract) {
  (void)sampling_contract_name(contract);
  return contract == SamplingContract::PositionCoupledV1 ? "prefill.fixed1024.absolute-boundaries.v1" : "prefill.legacy1024.resume-tail.v1";
}
std::size_t prefill_next_chunk(std::size_t position, std::size_t total, SamplingContract contract) {
  (void)sampling_contract_name(contract);
  if (position >= total) throw std::invalid_argument("prefill position outside prompt");
  const auto room = contract == SamplingContract::PositionCoupledV1 ? 1024-position%1024 : 1024;
  return std::min(room,total-position);
}
std::size_t prefill_next_uncontended_chunk(std::size_t position, std::size_t total, SamplingContract contract, std::size_t max_chunk_tokens) {
  (void)sampling_contract_name(contract);
  if (position >= total) throw std::invalid_argument("prefill position outside prompt");
  if (max_chunk_tokens < 2048) {
    const std::size_t cap = max_chunk_tokens == 0 ? 1024 : max_chunk_tokens;
    return std::min(prefill_next_chunk(position, total, contract), cap);
  }
  const std::size_t rem = total - position;
  const bool aligned =
      (contract != SamplingContract::PositionCoupledV1 ||
       (position % 1024 == 0));
  if (max_chunk_tokens >= 4096 && rem >= 4096 && aligned) {
    return 4096;
  }
  if (rem >= 2048 && aligned) {
    return 2048;
  }
  return prefill_next_chunk(position, total, contract);
}
std::size_t prefill_next_contended_chunk(std::size_t position, std::size_t total, SamplingContract contract, std::size_t max_contended_tokens) {
  const auto canonical = prefill_next_chunk(position, total, contract);
  if (max_contended_tokens == 0 || max_contended_tokens >= canonical) return canonical;
  // Align to 64-token KvPagePool boundaries while never straddling a 1024-token
  // PositionCoupledV1 absolute checkpoint boundary.
  const std::size_t page_rem = position % 64;
  if (page_rem != 0) {
    const std::size_t to_page = 64 - page_rem;
    if (max_contended_tokens <= to_page) return std::min(canonical, to_page);
    const std::size_t extra_pages = ((max_contended_tokens - to_page) / 64) * 64;
    return std::min(canonical, to_page + extra_pages);
  }
  const std::size_t page_aligned = std::max<std::size_t>(64, (max_contended_tokens / 64) * 64);
  return std::min(canonical, page_aligned);
}
bool prefill_checkpoint_compatible(std::size_t position, std::size_t total, SamplingContract contract) {
  (void)sampling_contract_name(contract);
  return position && position <= total && (contract == SamplingContract::LegacyRejectionV1 || position == total || position%1024 == 0);
}
} // namespace omlx::m5
