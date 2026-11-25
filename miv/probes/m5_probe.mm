#include <sys/sysctl.h>

#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/metal.h"

#ifndef OMLX_M5_CLASSIC_METALLIB_PATH
#define OMLX_M5_CLASSIC_METALLIB_PATH ""
#endif

#ifndef OMLX_M5_NAX_METALLIB_PATH
#define OMLX_M5_NAX_METALLIB_PATH ""
#endif

namespace {

struct Options {
  std::uint64_t bytes = 64ULL * 1024ULL * 1024ULL;
  std::uint32_t iterations = 8;
  std::string output;
};

std::uint64_t parse_u64(std::string_view text, std::string_view name) {
  std::uint64_t value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value == 0) {
    throw std::invalid_argument(std::string(name) +
                                " must be a positive integer");
  }
  return value;
}

Options parse_args(int argc, const char *const *argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--bytes" && i + 1 < argc) {
      options.bytes = parse_u64(argv[++i], "--bytes");
    } else if (arg == "--iterations" && i + 1 < argc) {
      const auto value = parse_u64(argv[++i], "--iterations");
      if (value > UINT32_MAX) {
        throw std::invalid_argument("--iterations is too large");
      }
      options.iterations = static_cast<std::uint32_t>(value);
    } else if (arg == "--output" && i + 1 < argc) {
      options.output = argv[++i];
    } else if (arg == "--help") {
      std::cout << "usage: omlx_m5_probe [--bytes N] [--iterations N] "
                   "[--output FILE]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  std::string(arg));
    }
  }
  options.bytes -= options.bytes % sizeof(std::uint32_t);
  if (options.bytes == 0 ||
      options.bytes > UINT32_MAX * sizeof(std::uint32_t)) {
    throw std::invalid_argument("--bytes is outside the supported range");
  }
  return options;
}

std::string json_escape(std::string_view value) {
  std::ostringstream stream;
  for (const unsigned char character : value) {
    switch (character) {
    case '\\':
      stream << "\\\\";
      break;
    case '"':
      stream << "\\\"";
      break;
    case '\n':
      stream << "\\n";
      break;
    case '\r':
      stream << "\\r";
      break;
    case '\t':
      stream << "\\t";
      break;
    default:
      if (character < 0x20) {
        stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
               << static_cast<int>(character) << std::dec;
      } else {
        stream << character;
      }
    }
  }
  return stream.str();
}

std::uint64_t system_memory_bytes() {
  std::uint64_t value = 0;
  std::size_t size = sizeof(value);
  if (sysctlbyname("hw.memsize", &value, &size, nullptr, 0) != 0) {
    return 0;
  }
  return value;
}

std::vector<std::byte> constant_u32(std::uint32_t value) {
  std::vector<std::byte> result(sizeof(value));
  std::memcpy(result.data(), &value, sizeof(value));
  return result;
}

struct LibraryStatus {
  bool file_exists = false;
  bool loaded = false;
  std::string error;
};

LibraryStatus check_library(omlx::m5::MetalContext &context, std::string key,
                            const std::string &path) {
  LibraryStatus status;
  if (path.empty()) {
    status.error = "path not configured";
    return status;
  }
  status.file_exists = std::filesystem::is_regular_file(path);
  if (!status.file_exists) {
    status.error = "file not found";
    return status;
  }
  try {
    context.register_library({.key = std::move(key), .path = path});
    status.loaded = true;
  } catch (const std::exception &error) {
    status.error = error.what();
  }
  return status;
}

double run_copy_bandwidth(omlx::m5::MetalContext &context, std::uint64_t bytes,
                          std::uint32_t iterations) {
  const auto elements =
      static_cast<std::uint32_t>(bytes / sizeof(std::uint32_t));
  const auto pipeline = context.warmup_pipeline({
      .key = "d0.probe.copy.pipeline",
      .library_key = "d0.probe.classic.library",
      .function_name = "omlx_m5_copy_u32",
      .variant = "metal4-direct",
  });
  const auto plan = context.prepare_command_plan({
      .key = "d0.probe.copy." + std::to_string(elements),
      .dispatches = {{
          .pipeline_key = pipeline.descriptor.key,
          .grid = {.width = elements},
          .threads_per_threadgroup = {.width = pipeline.thread_execution_width},
          .resources =
              {
                  {.argument_index = 0,
                   .resource_slot = 0,
                   .minimum_bytes = bytes},
                  {.argument_index = 1,
                   .resource_slot = 1,
                   .minimum_bytes = bytes},
              },
          .constant_arguments = {{.argument_index = 2,
                                  .bytes = constant_u32(elements)}},
      }},
  });
  std::vector<std::uint32_t> values(elements);
  for (std::uint32_t index = 0; index < elements; ++index) {
    values[index] = index;
  }
  auto source = context.upload(std::as_bytes(std::span(values)));
  auto destination = context.make_buffer(bytes);
  const std::array bindings = {
      omlx::m5::MetalBufferBinding{0, &source, 0},
      omlx::m5::MetalBufferBinding{1, &destination, 0},
  };
  static_cast<void>(context.execute(plan, bindings));
  const auto start = std::chrono::steady_clock::now();
  for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
    static_cast<void>(context.execute(plan, bindings));
  }
  const auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
  const auto expected = std::as_bytes(std::span(values));
  if (destination.read() !=
      std::vector<std::byte>(expected.begin(), expected.end())) {
    throw std::runtime_error("Metal 4 copy probe output mismatch");
  }
  const auto traffic =
      static_cast<double>(bytes) * 2.0 * static_cast<double>(iterations);
  return traffic / elapsed.count() / 1.0e9;
}

std::string render_json(const omlx::m5::MetalCapabilities &capabilities,
                        const omlx::m5::MetalRegistryStats &stats,
                        const Options &options,
                        const LibraryStatus &classic_status,
                        const LibraryStatus &nax_status,
                        double copy_bandwidth_gbs) {
  const auto target_m5_ultra =
      capabilities.device_name.find("M5 Ultra") != std::string::npos;
  std::ostringstream json;
  json << std::fixed << std::setprecision(3);
  json << "{\n"
       << "  \"schema_version\": 2,\n"
       << "  \"probe\": \"d0_metal4_smoke\",\n"
       << "  \"measurement_scope\": \""
       << (target_m5_ultra ? "M5 Ultra measurement"
                           : "M4 Metal 4 correctness and harness only")
       << "\",\n"
       << "  \"hardware\": {\n"
       << "    \"device_name\": \"" << json_escape(capabilities.device_name)
       << "\",\n"
       << "    \"system_memory_bytes\": " << system_memory_bytes() << ",\n"
       << "    \"recommended_working_set_bytes\": "
       << capabilities.recommended_working_set_bytes << ",\n"
       << "    \"max_buffer_length\": " << capabilities.max_buffer_length
       << "\n"
       << "  },\n"
       << "  \"software\": {\n"
       << "    \"os\": \"" << json_escape(capabilities.runtime_os) << "\"\n"
       << "  },\n"
       << "  \"submission\": {\n"
       << "    \"metal4_direct_eligible\": "
       << (capabilities.profile.metal4_direct_eligible ? "true" : "false")
       << ",\n"
       << "    \"metal4_command_buffers\": "
       << stats.metal4_command_buffers_submitted << ",\n"
       << "    \"classic_command_buffers\": "
       << stats.classic_command_buffers_submitted << "\n"
       << "  },\n"
       << "  \"classic_metallib\": {\n"
       << "    \"built\": " << (classic_status.file_exists ? "true" : "false")
       << ",\n"
       << "    \"loaded\": " << (classic_status.loaded ? "true" : "false")
       << ",\n"
       << "    \"error\": \"" << json_escape(classic_status.error) << "\"\n"
       << "  },\n"
       << "  \"nax_metallib\": {\n"
       << "    \"built\": " << (nax_status.file_exists ? "true" : "false")
       << ",\n"
       << "    \"loaded_on_current_host\": "
       << (nax_status.loaded ? "true" : "false") << ",\n"
       << "    \"loaded_on_m5_ultra\": "
       << (nax_status.loaded && target_m5_ultra ? "true" : "false") << ",\n"
       << "    \"runtime_validated_on_m5\": false,\n"
       << "    \"error\": \"" << json_escape(nax_status.error) << "\"\n"
       << "  },\n"
       << "  \"copy_probe\": {\n"
       << "    \"bytes\": " << options.bytes << ",\n"
       << "    \"iterations\": " << options.iterations << ",\n"
       << "    \"read_plus_write_gbs\": " << copy_bandwidth_gbs << "\n"
       << "  }\n"
       << "}\n";
  return json.str();
}

std::string resolve_metallib_path(const std::string &compiled,
                                  const char *argv0) {
  if (!compiled.empty() && std::filesystem::is_regular_file(compiled)) {
    return compiled;
  }
  if (!compiled.empty() && argv0 && *argv0) {
    const auto name = std::filesystem::path(compiled).filename();
    const auto exe_dir =
        std::filesystem::weakly_canonical(argv0).parent_path();
    for (const auto &candidate : {exe_dir / "metallib" / name,
                                  exe_dir.parent_path() / "metallib" / name}) {
      if (std::filesystem::is_regular_file(candidate)) {
        return candidate.string();
      }
    }
  }
  return compiled;
}

} // namespace

int main(int argc, const char *const *argv) {
  try {
    const auto options = parse_args(argc, argv);
    auto context = omlx::m5::MetalContext::create();
    const std::string classic_path =
        resolve_metallib_path(OMLX_M5_CLASSIC_METALLIB_PATH, argv[0]);
    const std::string nax_path =
        resolve_metallib_path(OMLX_M5_NAX_METALLIB_PATH, argv[0]);
    const auto classic_status =
        check_library(context, "d0.probe.classic.library", classic_path);
    const auto nax_status =
        check_library(context, "d0.probe.nax.library", nax_path);
    if (!classic_status.loaded) {
      throw std::runtime_error("classic metallib is not loadable: " +
                               classic_status.error);
    }
    const auto bandwidth =
        run_copy_bandwidth(context, options.bytes, options.iterations);
    const auto json =
        render_json(context.capabilities(), context.registry_stats(), options,
                    classic_status, nax_status, bandwidth);
    if (options.output.empty()) {
      std::cout << json;
    } else {
      std::ofstream output(options.output);
      if (!output) {
        throw std::runtime_error("unable to open output file");
      }
      output << json;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "omlx_m5_probe: " << error.what() << '\n';
    return 1;
  }
}
