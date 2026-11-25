#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/dispatch.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/qwen_program.h"
#include "omlx_m5/topology_profile.h"

#ifndef OMLX_M5_QWEN_AR_METALLIB_PATH
#error "OMLX_M5_QWEN_AR_METALLIB_PATH is required"
#endif

namespace {

struct Options {
  std::filesystem::path artifact;
  std::filesystem::path config;
  std::vector<std::uint32_t> prompt;
  std::uint32_t max_context = 1024;
  std::uint32_t maximum_new_tokens = 16;
  std::uint32_t repeat = 1;
  std::optional<std::uint32_t> cancel_after;
  std::optional<std::uint64_t> fail_after;
  omlx::m5::QwenArSamplingOptions sampling;
  bool self_check = false;
};

std::vector<std::uint32_t> parse_tokens(std::string_view value) {
  std::vector<std::uint32_t> result;
  std::stringstream input{std::string(value)};
  std::string part;
  while (std::getline(input, part, ',')) {
    if (part.empty()) {
      throw std::invalid_argument("prompt contains an empty token");
    }
    result.push_back(static_cast<std::uint32_t>(std::stoul(part)));
  }
  if (result.empty()) {
    throw std::invalid_argument("prompt is empty");
  }
  return result;
}

Options parse_options(int argc, char **argv) {
  Options result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto require_value = [&]() -> std::string {
      if (++index >= argc) {
        throw std::invalid_argument("missing value for " + argument);
      }
      return argv[index];
    };
    if (argument == "--artifact") {
      result.artifact = require_value();
    } else if (argument == "--config") {
      result.config = require_value();
    } else if (argument == "--prompt") {
      result.prompt = parse_tokens(require_value());
    } else if (argument == "--max-context") {
      result.max_context =
          static_cast<std::uint32_t>(std::stoul(require_value()));
    } else if (argument == "--max-new-tokens") {
      result.maximum_new_tokens =
          static_cast<std::uint32_t>(std::stoul(require_value()));
    } else if (argument == "--temperature") {
      result.sampling.temperature = std::stof(require_value());
    } else if (argument == "--top-k") {
      result.sampling.top_k =
          static_cast<std::uint32_t>(std::stoul(require_value()));
    } else if (argument == "--top-p") {
      result.sampling.top_p = std::stof(require_value());
    } else if (argument == "--seed") {
      result.sampling.seed =
          static_cast<std::uint32_t>(std::stoul(require_value()));
    } else if (argument == "--repeat") {
      result.repeat = static_cast<std::uint32_t>(std::stoul(require_value()));
    } else if (argument == "--cancel-after") {
      result.cancel_after =
          static_cast<std::uint32_t>(std::stoul(require_value()));
    } else if (argument == "--fail-after") {
      result.fail_after = std::stoull(require_value());
    } else if (argument == "--self-check") {
      result.self_check = true;
    } else {
      throw std::invalid_argument("unknown runner argument: " + argument);
    }
  }
  if (result.artifact.empty() || result.config.empty() ||
      result.prompt.empty() || result.max_context < 1024 ||
      result.maximum_new_tokens == 0 || result.repeat == 0) {
    throw std::invalid_argument(
        "runner requires artifact, config, prompt, max-context>=1024, and "
        "positive token/repeat counts");
  }
  return result;
}

omlx::m5::QwenProgram make_program(const Options &options) {
  auto artifact = omlx::m5::ArtifactModel::open(options.artifact, true);
  const auto config = omlx::m5::load_qwen_model_config(options.config);
  auto context = omlx::m5::MetalContext::create();
  auto materialization = omlx::m5::plan_model_materialization(
      artifact, context,
      {.components = {omlx::m5::ModelComponent::text},
       .require_verified_payload = true});
  auto layers = omlx::m5::build_qwen_layer_catalog(artifact, config,
                                                   options.max_context, 64);
  auto profiles =
      omlx::m5::compile_topology_profiles(omlx::m5::describe_qwen_topology(
          artifact, config, layers, options.max_context, 64));
  auto dispatch =
      omlx::m5::build_qwen_program_dispatch_table(artifact, layers, profiles);
  const auto build_plan = omlx::m5::plan_qwen_program(
      artifact, config, materialization, layers, profiles, dispatch.stats(),
      {.max_context = options.max_context, .kv_page_tokens = 64});
  auto weights = omlx::m5::materialize_model(
      artifact, std::move(materialization), context);
  auto program = omlx::m5::assemble_qwen_program(
      build_plan, std::move(context), std::move(weights), std::move(layers),
      std::move(dispatch), std::move(profiles));
  program.install_autoregressive_executor(OMLX_M5_QWEN_AR_METALLIB_PATH);
  return program;
}

void emit(const nlohmann::json &event) { std::cout << event.dump() << '\n'; }

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = parse_options(argc, argv);
    auto program = make_program(options);
    const auto addresses = program.autoregressive_address_snapshot();
    const auto registry_before = program.registry_stats();
    emit({{"event", "ready"},
          {"model_execution_kind", "control_fixture"},
          {"architecture",
           omlx::m5::to_string(program.build_plan().architecture)},
          {"quant_profile",
           omlx::m5::to_string(program.build_plan().quant_profile)},
          {"profile_count", program.profiles().profiles().size()},
          {"topology_signature", program.build_plan().topology_signature},
          {"hidden_size", program.build_plan().hidden_size},
          {"vocabulary_size", program.build_plan().vocabulary_size},
          {"expert_count", program.build_plan().expert_count},
          {"expert_top_k", program.build_plan().expert_top_k},
          {"layer_count", program.build_plan().layer_count},
          {"icb_recordings", program.autoregressive_stats().icb_recordings}});

    if (options.self_check) {
      nlohmann::json contexts = nlohmann::json::array();
      for (const std::uint32_t count : {1U, 5U, 17U, 129U}) {
        std::vector<std::uint32_t> tokens(count);
        for (std::uint32_t index = 0; index < count; ++index) {
          tokens[index] = index % program.build_plan().vocabulary_size;
        }
        program.reset_autoregressive();
        const auto mode = count == 1 ? omlx::m5::TopologyProfileMode::decode
                                     : omlx::m5::TopologyProfileMode::prefill;
        const auto step = program.execute_ar(
            tokens, mode, omlx::m5::QwenArExecutionRoute::execute_batch,
            options.sampling);
        contexts.push_back({{"tokens", count},
                            {"capacity", step.profile_capacity},
                            {"oracle_max_abs_error", step.oracle_max_abs_error},
                            {"kv_checksum", step.kv_checksum},
                            {"gdn_checksum", step.gdn_checksum}});
        contexts.back()["sampled_token"] = step.sampled_token;
      }
      const std::array<std::uint32_t, 1> token = {11};
      program.reset_autoregressive();
      const auto direct = program.execute_ar(
          token, omlx::m5::TopologyProfileMode::decode,
          omlx::m5::QwenArExecutionRoute::execute_batch, options.sampling);
      program.reset_autoregressive();
      const auto icb = program.execute_ar(
          token, omlx::m5::TopologyProfileMode::decode,
          omlx::m5::QwenArExecutionRoute::icb_profile, options.sampling);
      emit(
          {{"event", "self_check"},
           {"contexts", std::move(contexts)},
           {"icb_batch_token_equal", direct.sampled_token == icb.sampled_token},
           {"icb_batch_kv_equal", direct.kv_checksum == icb.kv_checksum},
           {"icb_batch_gdn_equal", direct.gdn_checksum == icb.gdn_checksum},
           {"icb_dispatches", icb.submission.dispatch_count}});
    }

    if (options.fail_after.has_value()) {
      program.inject_failure_after(*options.fail_after);
    }
    std::vector<std::uint32_t> previous_tokens;
    bool repeat_identical = true;
    bool any_failure = false;
    for (std::uint32_t request = 0; request < options.repeat; ++request) {
      std::uint32_t cancellation_checks = 0;
      const auto run = program.generate(
          options.prompt, options.maximum_new_tokens, options.sampling, [&] {
            return options.cancel_after.has_value() &&
                   cancellation_checks++ >= *options.cancel_after;
          });
      std::vector<std::uint32_t> current_tokens;
      current_tokens.reserve(run.events.size());
      for (const auto &record : run.events) {
        current_tokens.push_back(record.token);
        emit({{"event", "token"},
              {"request", request},
              {"index", record.index},
              {"position", record.position},
              {"token", record.token},
              {"route", record.route}});
      }
      if (request != 0) {
        repeat_identical =
            repeat_identical && current_tokens == previous_tokens;
      }
      previous_tokens = std::move(current_tokens);
      any_failure = any_failure || run.failed;
      emit({{"event", run.failed ? "error" : "request_done"},
            {"request", request},
            {"tokens", run.events.size()},
            {"cancelled", run.cancelled},
            {"failed", run.failed},
            {"error", run.error}});
      if (run.failed) {
        break;
      }
    }
    const auto registry_after = program.registry_stats();
    const auto &stats = program.autoregressive_stats();
    emit({{"event", "done"},
          {"repeat_identical", repeat_identical},
          {"addresses_stable",
           program.autoregressive_address_snapshot() == addresses},
          {"batch_steps", stats.batch_steps},
          {"icb_steps", stats.icb_steps},
          {"fallback_steps", stats.fallback_steps},
          {"profile_switches", stats.profile_switches},
          {"icb_recordings", stats.icb_recordings},
          {"icb_replays", stats.icb_replays},
          {"token_loop_plan_builds", stats.token_loop_plan_builds},
          {"token_loop_pipeline_compiles", stats.token_loop_pipeline_compiles},
          {"token_loop_weight_uploads", stats.token_loop_weight_uploads},
          {"registry_delta",
           {{"buffer_count", registry_after.current_buffer_count -
                                 registry_before.current_buffer_count},
            {"upload_count",
             registry_after.upload_count - registry_before.upload_count},
            {"pipeline_compiles", registry_after.pipelines_compiled -
                                      registry_before.pipelines_compiled},
            {"icb_recordings", registry_after.icb_profile_recordings -
                                   registry_before.icb_profile_recordings},
            {"icb_replays", registry_after.icb_profile_replays -
                                registry_before.icb_profile_replays}}}});
    return any_failure ? 2 : 0;
  } catch (const std::exception &error) {
    emit({{"event", "fatal"}, {"error", error.what()}});
    return 1;
  }
}
