#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "omlx_m5/materialization.h"

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <class Exception, class Function>
void expect_throws(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

std::string hex_crc32(std::uint32_t crc) {
  std::ostringstream out;
  out << std::hex << std::nouppercase << std::setfill('0') << std::setw(8)
      << crc;
  return out.str();
}

void append_u32(std::vector<std::byte> &out, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<std::byte>((v >> (i * 8)) & 0xFFU));
  }
}

void append_u64(std::vector<std::byte> &out, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::byte>((v >> (i * 8)) & 0xFFU));
  }
}

void test_component_classification() {
  using omlx::m5::classify_model_component;
  using omlx::m5::ModelComponent;
  expect(classify_model_component("language_model.model.layers.0.weight") ==
             ModelComponent::text,
         "text component classification failed");
  expect(classify_model_component("vision_model.encoder.weight") ==
             ModelComponent::vision,
         "vision component classification failed");
  expect(classify_model_component("vision_tower.blocks.0.weight") ==
             ModelComponent::vision,
         "vision tower component classification failed");
  expect(classify_model_component("language_model.mtp.layers.0.weight") ==
             ModelComponent::mtp,
         "MTP component classification failed");
  expect(classify_model_component("language_model.draft_model.weight") ==
             ModelComponent::speculative,
         "speculative component classification failed");
}

void test_mutable_arenas() {
  auto context = omlx::m5::MetalContext::create();
  const auto before = context.registry_stats();
  {
    omlx::m5::WorkspaceArena workspace(context, 4096);
    omlx::m5::StateArena state(context, 8192);
    const auto after_create = context.registry_stats();
    expect(after_create.current_buffer_count == before.current_buffer_count + 2,
           "workspace/state buffers are missing from the resource ledger");
    expect(after_create.current_buffer_bytes ==
               before.current_buffer_bytes + 4096 + 8192,
           "workspace/state bytes disagree with the resource ledger");

    auto first = workspace.allocate(3, 1);
    auto second = workspace.allocate(512);
    expect(first.offset == 0 && first.bytes == 3,
           "workspace first allocation is wrong");
    expect(second.offset == 256 && second.bytes == 512,
           "workspace alignment is wrong");
    const std::array<std::byte, 3> marker = {std::byte{0x12}, std::byte{0x34},
                                             std::byte{0x56}};
    first.buffer->write(marker, first.offset);
    expect(first.buffer->read(first.offset, marker.size()) ==
               std::vector<std::byte>(marker.begin(), marker.end()),
           "workspace allocation does not reference its backing buffer");
    expect(workspace.stats().used_bytes == 768 &&
               workspace.stats().peak_bytes == 768 &&
               workspace.stats().allocation_count == 2,
           "workspace accounting is wrong");

    workspace.reset();
    expect(workspace.stats().used_bytes == 0 &&
               workspace.stats().peak_bytes == 768 &&
               workspace.stats().generation == 1,
           "workspace reset accounting is wrong");
    auto reused = workspace.allocate(256);
    expect(reused.offset == 0,
           "workspace reset did not make scratch storage reusable");

    const auto state_slice = state.allocate(1024, 512);
    expect(state_slice.offset == 0 && state.stats().used_bytes == 1024,
           "state arena allocation is wrong");
    expect_throws<std::bad_alloc>(
        [&] { static_cast<void>(state.allocate(8192)); },
        "state arena accepted an allocation beyond capacity");
    expect_throws<std::invalid_argument>(
        [&] { static_cast<void>(workspace.allocate(1, 3)); },
        "workspace accepted a non-power-of-two alignment");

    omlx::m5::WorkspaceArena moved(std::move(workspace));
    expect(moved.allocate(128).offset == 256,
           "moved workspace lost its bump position");
  }
  const auto after_destroy = context.registry_stats();
  expect(after_destroy.current_buffer_count == before.current_buffer_count &&
             after_destroy.current_buffer_bytes == before.current_buffer_bytes,
         "workspace/state buffers leaked after destruction");
}

void test_16kib_mmap_direct_materialization() {
  const auto tmp_dir = std::filesystem::temp_directory_path() /
                       ("omlx-m5-b1-c1-" + std::to_string(::getpid()));
  std::filesystem::create_directories(tmp_dir);
  struct Cleanup {
    std::filesystem::path dir;
    ~Cleanup() { std::filesystem::remove_all(dir); }
  } cleanup{tmp_dir};

  std::vector<std::byte> payload(512);
  for (std::size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<std::byte>((i * 37U + 5U) & 0xFFU);
  }
  const auto sha = omlx::m5::artifact_sha256_hex(payload);
  const auto crc = hex_crc32(omlx::m5::artifact_crc32(payload));
  const std::uint64_t payload_offset = omlx::m5::kPayloadAlignment;

  struct BrandingCase {
    std::array<char, 8> magic;
    std::string_view schema;
    const char *filename;
  };
  const std::array<BrandingCase, 2> cases = {{
      {omlx::m5::kArtifactMagic, omlx::m5::kArtifactSchemaV1, "direct_16k.miv"},
      {omlx::m5::kLegacyArtifactMagic, omlx::m5::kLegacyArtifactSchemaV1,
       "direct_16k_legacy.omlxm5"},
  }};

  auto context = omlx::m5::MetalContext::create();
  for (const auto &item : cases) {
    std::ostringstream json;
    json << "{\"bindings\":[],\"container_version\":1,"
         << "\"model\":{\"source_profile\":\"dflash2_bf16\","
         << "\"source_repo\":\"test/b1\",\"source_revision\":\"r1\"},"
         << "\"objects\":[{\"format\":\"direct_bf16_v1\",\"id\":0,"
         << "\"layout\":\"contiguous_le_v1\","
         << "\"name\":\"language_model.model.norm.weight\","
         << "\"object_bytes\":512,\"object_offset\":0,"
         << "\"object_sha256\":\"" << sha << "\","
         << "\"planes\":[{\"alignment\":256,\"bytes\":512,\"crc32\":\"" << crc
         << "\",\"kind\":\"data\",\"offset\":0,\"sha256\":\"" << sha << "\"}],"
         << "\"shape\":[256],\"source_tensors\":[\"norm.weight\"]}],"
         << "\"payload_bytes\":512,\"schema\":\"" << item.schema
         << "\",\"views\":[]}";
    const auto manifest_str = json.str();
    const auto manifest_span =
        std::as_bytes(std::span(manifest_str.data(), manifest_str.size()));

    std::vector<std::byte> file;
    for (char c : item.magic) {
      file.push_back(static_cast<std::byte>(c));
    }
    append_u32(file, 1U);
    append_u32(file, 64U);
    append_u64(file, 64ULL);
    append_u64(file, manifest_span.size());
    append_u64(file, payload_offset);
    append_u64(file, payload.size());
    append_u32(file, 1U);
    append_u32(file, omlx::m5::artifact_crc32(manifest_span));
    append_u32(file, omlx::m5::artifact_crc32(
                         std::span<const std::byte>(file.data(), 56)));
    append_u32(file, 0U);
    file.insert(file.end(), manifest_span.begin(), manifest_span.end());
    file.resize(static_cast<std::size_t>(payload_offset), std::byte{0});
    file.insert(file.end(), payload.begin(), payload.end());

    const auto artifact_path = tmp_dir / item.filename;
    {
      std::ofstream out(artifact_path, std::ios::binary);
      out.write(reinterpret_cast<const char *>(file.data()),
                static_cast<std::streamsize>(file.size()));
    }

    const auto before = context.registry_stats();
    {
      auto artifact = omlx::m5::ArtifactModel::open(artifact_path, false);
      expect(artifact.summary().container_magic ==
                     std::string(item.magic.data(), item.magic.size()) &&
                 artifact.summary().manifest_schema == item.schema,
             "artifact summary magic/schema mismatch");
      expect(artifact.summary().payload_offset == 16384,
             "16 KiB payload alignment missing");
      const auto &obj = artifact.object(0U);
      const auto mmap_ptr = reinterpret_cast<std::uintptr_t>(
          artifact.plane_bytes(obj, artifact.plane(obj, "data")).data());
      expect(mmap_ptr % 16384 == 0,
             "mapped payload start is not 16 KiB page-aligned");
      artifact.verify_payload();
      auto plan = omlx::m5::plan_model_materialization(artifact, context);
      auto materialized =
          omlx::m5::materialize_model(artifact, std::move(plan), context);
      const auto mid = context.registry_stats();
      expect(mid.current_buffer_count == before.current_buffer_count + 1 &&
                 mid.upload_count == before.upload_count + 1 &&
                 mid.uploaded_bytes == before.uploaded_bytes + 512,
             "direct mmap materialization created unexpected buffers/uploads");
      const auto view =
          materialized.parent_view("language_model.model.norm.weight");
      expect(view.planes.front().buffer->read(0, 512) == payload,
             "materialized GPU segment bytes differ from mmap payload");
    }
  }
}

} // namespace

int main() {
  try {
    test_component_classification();
    test_mutable_arenas();
    test_16kib_mmap_direct_materialization();
    std::cout << "B1.2 materialization arena tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B1.2 materialization test failure: " << error.what() << '\n';
    return 1;
  }
}
