#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/metal.h"
#include "omlx_m5/quantization.h"

#ifndef OMLX_M5_NAX_METALLIB_PATH
#define OMLX_M5_NAX_METALLIB_PATH ""
#endif

namespace {

struct Options {
  std::uint32_t m = 1;
  std::uint32_t n = 64;
  std::uint32_t k = 64;
  std::uint32_t bits = 4;
  std::uint32_t iterations = 10;
  std::string output;
};

std::uint32_t parse_u32(std::string_view text, std::string_view name) {
  std::uint32_t value = 0;
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
    if (arg == "--m" && i + 1 < argc) {
      options.m = parse_u32(argv[++i], "--m");
    } else if (arg == "--n" && i + 1 < argc) {
      options.n = parse_u32(argv[++i], "--n");
    } else if (arg == "--k" && i + 1 < argc) {
      options.k = parse_u32(argv[++i], "--k");
    } else if (arg == "--bits" && i + 1 < argc) {
      options.bits = parse_u32(argv[++i], "--bits");
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = parse_u32(argv[++i], "--iterations");
    } else if (arg == "--output" && i + 1 < argc) {
      options.output = argv[++i];
    } else if (arg == "--help") {
      std::cout << "usage: omlx_m5_nax_qmm_probe [--m N] [--n N] [--k N] "
                   "[--bits 4|5] [--iterations N] [--output FILE]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  std::string(arg));
    }
  }
  if (options.bits != 4 && options.bits != 5) {
    throw std::invalid_argument("--bits must be 4 or 5");
  }
  if (options.k % 64 != 0) {
    throw std::invalid_argument("--k must be divisible by group size 64");
  }
  if (options.n % 64 != 0) {
    throw std::invalid_argument("--n must be divisible by the D0 NAX tile 64");
  }
  return options;
}

template <typename T> std::vector<std::byte> constant_bytes(const T &value) {
  std::vector<std::byte> result(sizeof(value));
  std::memcpy(result.data(), &value, sizeof(value));
  return result;
}

std::string json_escape(std::string_view value) {
  std::ostringstream stream;
  for (const unsigned char character : value) {
    if (character == '\\') {
      stream << "\\\\";
    } else if (character == '"') {
      stream << "\\\"";
    } else if (character == '\n') {
      stream << "\\n";
    } else if (character < 0x20) {
      stream << ' ';
    } else {
      stream << character;
    }
  }
  return stream.str();
}

struct Result {
  bool metallib_built = false;
  bool skipped = false;
  bool runtime_available = false;
  bool numerical_verified = false;
  std::string error;
  double average_wall_ms = 0.0;
  double average_gpu_ms = 0.0;
  double max_abs_error = 0.0;
  double rel_l2_error = 0.0;
  double reference_l2_norm = 0.0;
  double actual_l2_norm = 0.0;
  std::uint64_t metal4_command_buffers = 0;
  std::uint64_t classic_command_buffers = 0;
};

std::uint32_t unpack_affine_code(const std::byte *row, std::uint64_t row_bytes,
                                 std::uint32_t column, std::uint32_t bits) {
  const auto bit_offset =
      static_cast<std::uint64_t>(column) * static_cast<std::uint64_t>(bits);
  const auto byte_offset = bit_offset / 8U;
  const auto shift = static_cast<std::uint32_t>(bit_offset % 8U);
  std::uint32_t packed = static_cast<std::uint8_t>(row[byte_offset]);
  if (byte_offset + 1U < row_bytes) {
    packed |= static_cast<std::uint32_t>(
                  static_cast<std::uint8_t>(row[byte_offset + 1U]))
              << 8U;
  }
  return (packed >> shift) & ((1U << bits) - 1U);
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

Result run(omlx::m5::MetalContext &context, const Options &options,
           const char *argv0) {
  Result result;
  const std::string path =
      resolve_metallib_path(OMLX_M5_NAX_METALLIB_PATH, argv0);
  result.metallib_built =
      !path.empty() && std::filesystem::is_regular_file(path);
  if (!result.metallib_built) {
    result.skipped = true;
    result.error =
        "legacy MLX NAX smoke metallib absent (OMLX_M5_BUILD_MLX_LEGACY_SMOKE=OFF); skipped";
    return result;
  }
  try {
    context.register_library({
        .key = "d0.nax.library",
        .path = path,
    });
    const auto function_name = "omlx_m5_q" + std::to_string(options.bits) +
                               "_affine_g64_nax_bfloat16_t";
    const auto pipeline = context.warmup_pipeline({
        .key = "d0.nax.q" + std::to_string(options.bits) + ".pipeline",
        .library_key = "d0.nax.library",
        .function_name = function_name,
        .variant = "metal4-nax",
    });

    const auto weight_words_per_row =
        static_cast<std::uint64_t>(options.k) * options.bits / 32;
    const auto row_weight_bytes =
        weight_words_per_row * sizeof(std::uint32_t);
    const auto groups_per_row = options.k / 64;
    const auto weights_bytes =
        static_cast<std::uint64_t>(options.n) * row_weight_bytes;
    const auto parameters_bytes = static_cast<std::uint64_t>(options.n) *
                                  groups_per_row * sizeof(std::uint16_t);
    const auto input_bytes = static_cast<std::uint64_t>(options.m) * options.k *
                             sizeof(std::uint16_t);
    const auto output_bytes = static_cast<std::uint64_t>(options.m) *
                              options.n * sizeof(std::uint16_t);

    const auto max_code = (1U << options.bits) - 1U;
    std::vector<std::byte> weight_host(weights_bytes, std::byte{0});
    std::vector<std::uint16_t> scale_words(
        static_cast<std::size_t>(options.n) * groups_per_row);
    std::vector<std::uint16_t> bias_words(
        static_cast<std::size_t>(options.n) * groups_per_row);
    for (std::uint32_t row = 0; row < options.n; ++row) {
      for (std::uint32_t group = 0; group < groups_per_row; ++group) {
        const auto param_index =
            static_cast<std::size_t>(row) * groups_per_row + group;
        const float raw_scale =
            (0.125F +
             0.03125F * static_cast<float>((row * 5U + group * 3U) % 8U)) /
            static_cast<float>(max_code);
        const float raw_bias =
            -0.0625F +
            0.015625F * static_cast<float>((row * 7U + group * 11U) % 9U);
        scale_words[param_index] = omlx::m5::float_to_bf16(raw_scale);
        bias_words[param_index] = omlx::m5::float_to_bf16(raw_bias);
      }
      const auto row_base = static_cast<std::uint64_t>(row) * row_weight_bytes;
      for (std::uint32_t col = 0; col < options.k; ++col) {
        const std::uint32_t code =
            ((row * 13U + col * 7U + (col / 64U) * 5U + 3U) % max_code) + 1U;
        const auto bit_offset = static_cast<std::uint64_t>(col) *
                                static_cast<std::uint64_t>(options.bits);
        const auto byte_offset = row_base + (bit_offset / 8U);
        const auto shift = static_cast<unsigned>(bit_offset % 8U);
        const auto packed = static_cast<std::uint16_t>(code) << shift;
        weight_host[byte_offset] |= static_cast<std::byte>(packed & 0xFFU);
        if (shift + options.bits > 8U) {
          weight_host[byte_offset + 1U] |=
              static_cast<std::byte>((packed >> 8U) & 0xFFU);
        }
      }
    }

    const float input_norm_scale =
        8.0F / std::sqrt(static_cast<float>(options.k));
    std::vector<std::uint16_t> input_words(
        static_cast<std::size_t>(options.m) * options.k);
    std::vector<float> input_f32(input_words.size());
    for (std::uint32_t token = 0; token < options.m; ++token) {
      for (std::uint32_t col = 0; col < options.k; ++col) {
        const auto idx = static_cast<std::size_t>(token) * options.k + col;
        const int centered =
            static_cast<int>((token * 19U + col * 11U + (col / 16U) * 3U) %
                             17U) -
            7;
        const float raw_x =
            static_cast<float>(centered) * 0.125F * input_norm_scale;
        input_words[idx] = omlx::m5::float_to_bf16(raw_x);
        input_f32[idx] = omlx::m5::bf16_to_float(input_words[idx]);
      }
    }

    auto weights = context.upload(weight_host);
    auto scales = context.upload(std::as_bytes(std::span(scale_words)));
    auto biases = context.upload(std::as_bytes(std::span(bias_words)));
    auto input = context.upload(std::as_bytes(std::span(input_words)));
    auto output = context.make_buffer(output_bytes);
    output.write(std::vector<std::byte>(output_bytes, std::byte{0}));

    const auto k = static_cast<std::int32_t>(options.k);
    const auto n = static_cast<std::int32_t>(options.n);
    const auto m = static_cast<std::int32_t>(options.m);
    const auto plan = context.prepare_command_plan({
        .key = "d0.nax." + std::to_string(options.bits) + "." +
               std::to_string(options.m) + "." + std::to_string(options.n) +
               "." + std::to_string(options.k),
        .dispatches = {{
            .pipeline_key = pipeline.descriptor.key,
            .kind = omlx::m5::MetalDispatchKind::threadgroups,
            .grid =
                {
                    .width = (options.n + 63) / 64,
                    .height = (options.m + 63) / 64,
                },
            .threads_per_threadgroup = {.width = 32, .height = 2, .depth = 2},
            .resources =
                {
                    {.argument_index = 0,
                     .resource_slot = 0,
                     .minimum_bytes = weights_bytes},
                    {.argument_index = 1,
                     .resource_slot = 1,
                     .minimum_bytes = parameters_bytes},
                    {.argument_index = 2,
                     .resource_slot = 2,
                     .minimum_bytes = parameters_bytes},
                    {.argument_index = 3,
                     .resource_slot = 3,
                     .minimum_bytes = input_bytes},
                    {.argument_index = 4,
                     .resource_slot = 4,
                     .minimum_bytes = output_bytes},
                },
            .constant_arguments =
                {
                    {.argument_index = 5, .bytes = constant_bytes(k)},
                    {.argument_index = 6, .bytes = constant_bytes(n)},
                    {.argument_index = 7, .bytes = constant_bytes(m)},
                },
        }},
    });
    const std::array bindings = {
        omlx::m5::MetalBufferBinding{0, &weights, 0},
        omlx::m5::MetalBufferBinding{1, &scales, 0},
        omlx::m5::MetalBufferBinding{2, &biases, 0},
        omlx::m5::MetalBufferBinding{3, &input, 0},
        omlx::m5::MetalBufferBinding{4, &output, 0},
    };
    static_cast<void>(context.execute(plan, bindings));
    double total_wall_ms = 0.0;
    double total_gpu_ms = 0.0;
    for (std::uint32_t iteration = 0; iteration < options.iterations;
         ++iteration) {
      const auto submission = context.execute(plan, bindings);
      total_wall_ms += submission.timing.wall_ms;
      total_gpu_ms += submission.timing.gpu_ms;
    }
    const auto stats = context.registry_stats();
    result.runtime_available = true;
    result.average_wall_ms =
        total_wall_ms / static_cast<double>(options.iterations);
    result.average_gpu_ms =
        total_gpu_ms / static_cast<double>(options.iterations);
    result.metal4_command_buffers = stats.metal4_command_buffers_submitted;
    result.classic_command_buffers = stats.classic_command_buffers_submitted;

    const auto output_data = output.read();
    if (output_data.size() != output_bytes) {
      throw std::runtime_error("NAX output buffer size mismatch");
    }

    std::vector<float> weight_row(options.k);
    double max_abs_error = 0.0;
    double max_ref_abs = 0.0;
    double diff_sq_sum = 0.0;
    double ref_sq_sum = 0.0;
    double actual_sq_sum = 0.0;
    for (std::uint32_t row = 0; row < options.n; ++row) {
      const auto *row_ptr =
          weight_host.data() + static_cast<std::uint64_t>(row) * row_weight_bytes;
      for (std::uint32_t group = 0; group < groups_per_row; ++group) {
        const auto param_index =
            static_cast<std::size_t>(row) * groups_per_row + group;
        const float scale = omlx::m5::bf16_to_float(scale_words[param_index]);
        const float bias = omlx::m5::bf16_to_float(bias_words[param_index]);
        const std::uint32_t col_begin = group * 64U;
        for (std::uint32_t lane = 0; lane < 64U; ++lane) {
          const std::uint32_t col = col_begin + lane;
          const auto code =
              unpack_affine_code(row_ptr, row_weight_bytes, col, options.bits);
          const auto dequant_bf16 =
              omlx::m5::float_to_bf16(static_cast<float>(code) * scale + bias);
          weight_row[col] = omlx::m5::bf16_to_float(dequant_bf16);
        }
      }
      for (std::uint32_t token = 0; token < options.m; ++token) {
        const float *x_row =
            input_f32.data() + static_cast<std::size_t>(token) * options.k;
        float sum = 0.0F;
        for (std::uint32_t col = 0; col < options.k; ++col) {
          sum += x_row[col] * weight_row[col];
        }
        const auto out_index =
            (static_cast<std::size_t>(token) * options.n + row) *
            sizeof(std::uint16_t);
        std::uint16_t actual_word = 0;
        std::memcpy(&actual_word, output_data.data() + out_index,
                    sizeof(actual_word));
        const float actual = omlx::m5::bf16_to_float(actual_word);
        if (!std::isfinite(actual)) {
          throw std::runtime_error("NAX output contains non-finite value");
        }
        const double expected = static_cast<double>(sum);
        const double actual_d = static_cast<double>(actual);
        const double abs_err = std::abs(actual_d - expected);
        max_abs_error = std::max(max_abs_error, abs_err);
        max_ref_abs = std::max(max_ref_abs, std::abs(expected));
        diff_sq_sum += abs_err * abs_err;
        ref_sq_sum += expected * expected;
        actual_sq_sum += actual_d * actual_d;
      }
    }

    result.max_abs_error = max_abs_error;
    result.reference_l2_norm = std::sqrt(ref_sq_sum);
    result.actual_l2_norm = std::sqrt(actual_sq_sum);
    result.rel_l2_error =
        std::sqrt(diff_sq_sum) / std::max(result.reference_l2_norm, 1e-12);
    if (!(result.reference_l2_norm > 0.0) || !(result.actual_l2_norm > 0.0)) {
      throw std::runtime_error("NAX output or reference norm is zero");
    }
    const double abs_tolerance = std::max(0.05, 0.02 * max_ref_abs);
    if (result.rel_l2_error > 0.02 || result.max_abs_error > abs_tolerance) {
      std::ostringstream msg;
      msg << "NAX numerical verification failed: max_abs_error="
          << result.max_abs_error << " (limit " << abs_tolerance
          << "), rel_l2_error=" << result.rel_l2_error;
      throw std::runtime_error(msg.str());
    }
    result.numerical_verified = true;
  } catch (const std::exception &error) {
    result.error = error.what();
  }
  return result;
}

std::string render_json(const omlx::m5::MetalCapabilities &capabilities,
                        const Options &options, const Result &result) {
  const auto target_m5_ultra =
      capabilities.device_name.find("M5 Ultra") != std::string::npos;
  const auto target_m5 =
      target_m5_ultra ||
      capabilities.profile.probe.apple_gpu_family >= 10 ||
      capabilities.device_name.find("M5") != std::string::npos;
  const bool validated = result.runtime_available && result.numerical_verified;
  std::ostringstream json;
  json << std::fixed << std::setprecision(6);
  json << "{\n"
       << "  \"schema_version\": 2,\n"
       << "  \"probe\": \"d0_nax_qmm_metal4\",\n"
       << "  \"measurement_scope\": \""
       << (target_m5_ultra && validated ? "M5 Ultra measurement"
           : target_m5 && validated     ? "M5 measurement"
           : validated
               ? "M4 Metal 4/NAX API runtime measurement; not M5 performance"
               : "M4 correctness and harness only")
       << "\",\n"
       << "  \"device_name\": \"" << json_escape(capabilities.device_name)
       << "\",\n"
       << "  \"m\": " << options.m << ",\n"
       << "  \"n\": " << options.n << ",\n"
       << "  \"k\": " << options.k << ",\n"
       << "  \"bits\": " << options.bits << ",\n"
       << "  \"dtype\": \"bfloat16\",\n"
       << "  \"iterations\": " << options.iterations << ",\n"
       << "  \"metallib_built\": " << (result.metallib_built ? "true" : "false")
       << ",\n"
       << "  \"skipped\": " << (result.skipped ? "true" : "false") << ",\n"
       << "  \"runtime_available\": "
       << (result.runtime_available ? "true" : "false") << ",\n"
       << "  \"numerical_verified\": "
       << (result.numerical_verified ? "true" : "false") << ",\n"
       << "  \"runtime_validated_on_current_device\": "
       << (validated ? "true" : "false") << ",\n"
       << "  \"runtime_validated_on_current_m4\": "
       << (validated && !target_m5 ? "true" : "false")
       << ",\n"
       << "  \"runtime_validated_on_m5\": "
       << (validated && target_m5 ? "true" : "false")
       << ",\n"
       << "  \"max_abs_error\": " << result.max_abs_error << ",\n"
       << "  \"rel_l2_error\": " << result.rel_l2_error << ",\n"
       << "  \"reference_l2_norm\": " << result.reference_l2_norm << ",\n"
       << "  \"actual_l2_norm\": " << result.actual_l2_norm << ",\n"
       << "  \"metal4_command_buffers\": " << result.metal4_command_buffers
       << ",\n"
       << "  \"classic_command_buffers\": " << result.classic_command_buffers
       << ",\n"
       << "  \"average_wall_ms\": " << result.average_wall_ms << ",\n"
       << "  \"average_gpu_ms\": " << result.average_gpu_ms << ",\n"
       << "  \"error\": \"" << json_escape(result.error) << "\"\n"
       << "}\n";
  return json.str();
}

} // namespace

int main(int argc, const char *const *argv) {
  try {
    const auto options = parse_args(argc, argv);
    auto context = omlx::m5::MetalContext::create();
    const auto result = run(context, options, argv[0]);
    const auto json = render_json(context.capabilities(), options, result);
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
    std::cerr << "omlx_m5_nax_qmm_probe: " << error.what() << '\n';
    return 1;
  }
}
