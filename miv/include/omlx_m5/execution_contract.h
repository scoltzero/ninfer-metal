#pragma once
#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include "omlx_m5/artifact.h"
#include "omlx_m5/position_sampling.h"
namespace omlx::m5 {
using ExecutionIdentityFields = std::map<std::string, std::string>;
[[nodiscard]] std::string canonical_execution_identity(const ExecutionIdentityFields &);
[[nodiscard]] std::string execution_identity_digest(const ExecutionIdentityFields &);
[[nodiscard]] std::string execution_file_digest(const std::filesystem::path &);
[[nodiscard]] std::string execution_kernel_bundle_digest(const std::filesystem::path &);
[[nodiscard]] std::string execution_artifact_digest(const ArtifactModel &);
[[nodiscard]] std::string_view execution_source_id();
[[nodiscard]] std::string_view prefill_chunk_policy_id(SamplingContract);
[[nodiscard]] std::size_t prefill_next_chunk(std::size_t position, std::size_t total, SamplingContract);
[[nodiscard]] std::size_t prefill_next_uncontended_chunk(std::size_t position, std::size_t total, SamplingContract, std::size_t max_chunk_tokens);
[[nodiscard]] std::size_t prefill_next_contended_chunk(std::size_t position, std::size_t total, SamplingContract, std::size_t max_contended_tokens);
[[nodiscard]] bool prefill_checkpoint_compatible(std::size_t position, std::size_t total, SamplingContract);
} // namespace omlx::m5
