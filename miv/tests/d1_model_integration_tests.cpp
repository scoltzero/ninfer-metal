#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "omlx_m5/memory_plan.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/safetensors.h"

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main(int argc, const char *const *argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument(
          "usage: d1_model_integration_tests MODEL_DIR");
    }
    const std::filesystem::path model_dir = argv[1];
    const auto config =
        omlx::m5::load_qwen_model_config(model_dir / "config.json");
    omlx::m5::validate_qwen38_27b_target(config);
    const auto model = omlx::m5::SafeTensorModel::open(
        model_dir, "04dc5509edd8670fc78cc8c2f74bf9b77b1f2acc");
    expect(model.shard_count() == 4, "target must have four shards");
    expect(model.tensor_count() == 2209, "target tensor count mismatch");
    expect(model.total_tensor_bytes() == 16971398624ULL,
           "target tensor payload byte count mismatch");
    expect(model.declared_total_size() == 16971681558ULL,
           "target physical shard byte count mismatch");

    const auto modules = omlx::m5::quantized_module_names(model);
    expect(modules.size() == 505, "target quantized module count mismatch");
    std::map<std::int32_t, std::size_t> counts;
    for (const auto &name : modules) {
      const auto binding = omlx::m5::bind_quantized_module(model, config, name);
      ++counts[binding.bits];
    }
    expect(counts[4] == 339, "target q4 module count mismatch");
    expect(counts[5] == 166, "target q5 module count mismatch");

    const auto q5 = omlx::m5::bind_quantized_module(
        model, config, "language_model.model.layers.0.linear_attn.in_proj_a");
    const auto decoded = omlx::m5::dequantize_row(q5, 0, 0, 64);
    expect(decoded.size() == 64, "q5 sample decode length mismatch");
    expect(std::ranges::all_of(
               decoded, [](float value) { return std::isfinite(value); }),
           "q5 sample decode contains non-finite values");

    const auto plan = omlx::m5::plan_sequence_memory(config, 32768);
    expect(plan.kv_bytes_per_token == 65536, "target KV geometry mismatch");
    expect(plan.gdn_state_bytes == 150994944, "target GDN state mismatch");

    auto metal = omlx::m5::MetalContext::create(false);
    expect(!metal.capabilities().device_name.empty(),
           "Metal device name is empty");
    expect(metal.capabilities().nax_capability_source == "injected",
           "D1 capability override was not recorded");
    const auto weight = model.tensor(
        "language_model.model.layers.0.linear_attn.in_proj_a.weight");
    const auto upload_size = std::min<std::uint64_t>(weight.nbytes, 4096);
    const auto source = std::span<const std::byte>(weight.data, upload_size);
    auto buffer = metal.upload(source);
    expect(buffer.read() ==
               std::vector<std::byte>(source.begin(), source.end()),
           "Metal layer-slice upload round trip failed");

    auto heap = metal.make_heap(1ULL << 20);
    auto heap_buffer = heap.make_buffer(upload_size);
    heap_buffer.write(source);
    expect(heap_buffer.read() ==
               std::vector<std::byte>(source.begin(), source.end()),
           "Metal heap layer-slice round trip failed");
    const auto timing = metal.submit_barrier();
    expect(timing.wall_ms >= 0.0, "Metal timing is invalid");

    std::cout << "D1 model integration passed: " << model.tensor_count()
              << " tensors, " << modules.size() << " quantized modules\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "D1 model integration failed: " << error.what() << '\n';
    return 1;
  }
}
