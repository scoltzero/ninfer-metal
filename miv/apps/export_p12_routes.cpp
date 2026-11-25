#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <cmath>
#include <charconv>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/execution_contract.h"
#include "../src/route_trace_schema.h"
#include "omlx_m5/qwen_forward.h"
#include "omlx_m5/quantization.h"

using namespace omlx::m5;
using Json = nlohmann::json;
int main(int argc, char **argv) {
  try {
    if (argc != 7)
      throw std::invalid_argument(
          "export_p12_routes MODEL_DIR ARTIFACT LIBRARIES TOKENS_JSON LAYER OUTPUT");
    const auto config = load_qwen_model_config(
        std::filesystem::path(argv[1]) / "config.json");
    if (config.num_experts != 256 || config.num_experts_per_token != 8)
      throw std::invalid_argument("P12 exporter requires 256/top8 MoE model");
    Json input;
    std::ifstream(argv[4]) >> input;
    if (!input.is_object() || !input.contains("tokens") ||
        !input.at("tokens").is_array() ||
        input.value("origin", "").empty() ||
        !input.contains("synthetic") || !input.at("synthetic").is_boolean())
      throw std::invalid_argument("P12 input provenance/tokens missing");
    const auto contract=parse_sampling_contract(input.value("sampling_contract","legacy_rejection_v1"));
    std::vector<std::uint32_t> tokens;
    for(const auto &token:input.at("tokens")) {
      if(!token.is_number_integer() || (!token.is_number_unsigned() && token.get<std::int64_t>()<0) ||
        token.get<std::uint64_t>()>=std::uint64_t(config.vocab_size))throw std::invalid_argument("P12 input token integer/domain invalid");
      tokens.push_back(token.get<std::uint32_t>());
    }
    std::uint32_t layer=0;const std::string_view layer_text(argv[5]);
    const auto [end,error]=std::from_chars(layer_text.data(),layer_text.data()+layer_text.size(),layer);
    if(error!=std::errc{} || end!=layer_text.data()+layer_text.size())throw std::invalid_argument("P12 layer integer invalid");
    if (tokens.empty() || tokens.size() > 1024 ||
        layer >= std::uint32_t(config.num_hidden_layers) ||
        std::ranges::any_of(tokens, [&](auto token) {
          return token >= std::uint32_t(config.vocab_size);
        }))
      throw std::invalid_argument("P12 input token/layer domain invalid");
    auto context = MetalContext::create();
    auto artifact = ArtifactModel::open(argv[2], false);
    const auto planned = plan_model_materialization(
        artifact, context, {.components = {ModelComponent::text},
                            .require_verified_payload = false});
    const auto memory = plan_qwen_forward_memory(config,1024,0,true,false,false,contract==SamplingContract::PositionCoupledV1,contract==SamplingContract::PositionCoupledV1);
    const auto budget = qwen_load_budget(
        planned.resident_bytes, memory,
        context.capabilities().recommended_working_set_bytes,
        context.current_allocated_bytes());
    if (!budget.fits)
      throw std::runtime_error("P12 route export exceeds local Metal budget; "
                               "use target machine, no partial load");
    artifact = ArtifactModel::open(argv[2], true);
    auto weights = materialize_model(
        artifact,
        plan_model_materialization(
            artifact, context, {.components = {ModelComponent::text},
                                .require_verified_payload = true}),
        context);
    auto layers = build_qwen_layer_catalog(artifact, config, 1024, 64);
    QwenForward program(context,weights,layers,config,1024,argv[3],{},true,false,false,{},false,false,{},
      contract==SamplingContract::PositionCoupledV1?LinearNumericPolicy::RowExactPortableV1:LinearNumericPolicy::Legacy,
      contract,contract==SamplingContract::PositionCoupledV1?AttentionNumericPolicy::RowExactV1:AttentionNumericPolicy::Legacy);
    const auto forward = program.run(tokens,GpuSamplingOptions{.contract=contract}); // selected token only, no logits
    const auto route = program.read_moe_route_trace(layer);
    const std::string object_name = "language_model.model.layers." +
                                    std::to_string(layer) +
                                    ".mlp.switch_mlp.gate_proj";
    const auto &object = artifact.object(object_name);
    const auto profile = program.inspect_profile(std::uint32_t(tokens.size()));
    const auto found = std::find_if(
        profile.weights.begin(), profile.weights.end(),
        [&](const auto &record) { return record.object == object_name; });
    if (found == profile.weights.end())
      throw std::logic_error("P12 expert profile does not match captured route");
    const std::string router_name = "language_model.model.layers." +
                                    std::to_string(layer) + ".mlp.gate";
    const auto router = std::find_if(
        profile.weights.begin(), profile.weights.end(),
        [&](const auto &record) { return record.object == router_name; });
    if (router == profile.weights.end())
      throw std::logic_error("P12 router projection profile absent");
    Json ids = Json::array(), scores = Json::array(), raw = Json::array();
    for (std::uint32_t token = 0; token < route.tokens; ++token) {
      Json id_row = Json::array(), score_row = Json::array(), raw_row = Json::array();
      float mass = 0;
      for (std::uint32_t slot = 0; slot < 8; ++slot) {
        const auto index = std::uint64_t(token) * 8 + slot;
        const auto id = route.expert_ids[index];
        const auto code = route.scores_bf16[index];
        const float score = bf16_to_float(code);
        if (id >= 256 || !std::isfinite(score) || score < 0)
          throw std::runtime_error("P12 captured route/score invalid");
        id_row.push_back(id);
        score_row.push_back(score);
        raw_row.push_back(code);
        mass += score;
      }
      if (std::abs(mass - 1.0f) > 0.02f)
        throw std::runtime_error("P12 captured score mass invalid");
      ids.push_back(std::move(id_row));
      scores.push_back(std::move(score_row));
      raw.push_back(std::move(raw_row));
    }
    const auto byte_view = std::as_bytes(std::span(tokens));
    Json report = {
        {"schema", "omlx-m5-p12-route-trace-v2"},
        {"origin", input.at("origin")},
        {"synthetic", input.at("synthetic")},
        {"source_kind", artifact.summary().source_repo.starts_with("fixture/")
            ? "small_control_fixture" : "provided_target_tokens"},
        {"source_artifact", artifact.summary().source_repo + "@" +
                                artifact.summary().source_revision},
        {"input_tokens_sha256", artifact_sha256_hex(byte_view)},
        {"source_object", object_name},
        {"router_weight_object", router_name},
        {"router_projection_function", router->function},
        {"router_topk_kernel", "omlx_m5_forward_router_top8_logits"},
        {"source_format", object.format}, {"source_layout", object.layout},
        {"source_profile_function", found->function},
        {"source_profile_family", found->family},
        {"source_profile_capacity", found->capacity},
        {"layer", layer}, {"tokens", route.tokens},
        {"start_position", route.start_position},
        {"route_readback_bytes", route.readback_bytes},
        {"ordinary_forward_host_read_bytes", forward.host_read_bytes},
        {"experts", 256}, {"top_k", 8},
        {"expert_ids", ids}, {"scores", scores},
        {"scores_bf16_u16", raw},
        {"production_route_changed", false}};
    const auto profile_identity=[](const QwenProfileWeightRoute &r){return canonical_execution_identity({
      {"object",r.object},{"format",r.format},{"library",r.library},{"function",r.function},{"family",r.family},
      {"numeric",r.numeric_contract_id},{"n",std::to_string(r.n)},{"k",std::to_string(r.k)},{"capacity",std::to_string(r.capacity)},
      {"grid",std::to_string(r.grid.width)+","+std::to_string(r.grid.height)+","+std::to_string(r.grid.depth)}});};
    ExecutionIdentityFields provenance{{"artifact",execution_artifact_digest(artifact)},
      {"config",execution_file_digest(std::filesystem::path(argv[1])/"config.json")},
      {"compiled_source",std::string(execution_source_id())},{"kernels",execution_kernel_bundle_digest(argv[3])},
      {"execution",std::string(program.execution_identity())},{"numeric",profile.numeric_contract_id},
      {"sampling",std::string(sampling_contract_name(contract))},{"chunk",std::string(prefill_chunk_policy_id(contract))},
      {"row_layout","single-slot.absolute-position.rows.v1"},{"router_profile",profile_identity(*router)},
      {"expert_profile",profile_identity(*found)},{"input",artifact_sha256_hex(byte_view)},
      {"route_ids",artifact_sha256_hex(std::as_bytes(std::span(route.expert_ids)))},
      {"route_scores",artifact_sha256_hex(std::as_bytes(std::span(route.scores_bf16)))},{"scope","committed_forward"}};
    report["provenance"]=provenance;report["provenance_sha256"]=execution_identity_digest(provenance);
    report["row_positions"]=Json::array();report["row_slots"]=Json::array();report["row_valid"]=Json::array();
    for(std::uint32_t row=0;row<route.tokens;++row){report["row_positions"].push_back(route.start_position+row);report["row_slots"].push_back(0);report["row_valid"].push_back(1);}
    detail::validate_route_trace_v2(report);
    std::ofstream output(argv[6]);
    output << report.dump(2) << '\n';
    if (!output) throw std::runtime_error("P12 trace output write failed");
    std::cout << argv[6] << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "P12 route export failed: " << error.what() << '\n';
    return 1;
  }
}
