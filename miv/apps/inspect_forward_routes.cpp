#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../src/forward_plan.h"
#include "omlx_m5/artifact.h"
#include "omlx_m5/materialization.h"

namespace {
using omlx::m5::detail::forward::linear_grid;
using omlx::m5::detail::forward::resolve_linear_route;
using omlx::m5::detail::forward::select_grouped_expert_kernel;
using omlx::m5::detail::forward::U32;

U32 profile_capacity(U32 tokens) {
  for (const U32 capacity : {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U, 512U, 1024U})
    if (tokens <= capacity)
      return capacity;
  throw std::invalid_argument("requested T exceeds installed forward profiles");
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 4)
      throw std::invalid_argument(
          "usage: inspect_forward_routes ARTIFACT DEVICE_NAME T [T ...] [--row-exact]");
    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    const std::string device = argv[2];
    const bool short_routes = device == "Apple M4";
    const bool row_exact = std::string_view(argv[argc - 1]) == "--row-exact";
    const int numeric_end = argc - int(row_exact);
    if (numeric_end < 4) throw std::invalid_argument("no requested row width");
    nlohmann::json rows = nlohmann::json::array();
    for (const auto &object : artifact.objects()) {
      if ((object.shape.size() != 2 && object.shape.size() != 3) ||
          (object.format != "mxfp4_e2m1_g32_e8m0_v1" &&
           object.format != "direct_bf16_v1" &&
           !object.format.starts_with("oq_affine_q")))
        continue;
      if (object.shape[0] > UINT32_MAX || object.shape[1] > UINT32_MAX)
        throw std::overflow_error("forward route object shape exceeds U32");
      const bool expert = object.shape.size() == 3;
      if (expert && object.shape[0] != 256)
        continue;
      if (expert && object.shape[2] > UINT32_MAX)
        throw std::overflow_error("expert input dimension exceeds U32");
      const auto n = U32(object.shape[expert ? 1 : 0]);
      const auto k = U32(object.shape[expert ? 2 : 1]);
      omlx::m5::MaterializedWeightView metadata;
      metadata.format = object.format;
      for (int arg = 3; arg < numeric_end; ++arg) {
        const std::string text(argv[arg]);
        std::size_t consumed = 0;
        const auto parsed = std::stoull(text, &consumed);
        if (text.empty() || text.front() == '-' || consumed != text.size() ||
            parsed == 0 || parsed > 1024)
          throw std::invalid_argument("requested width outside 1..1024");
        const U32 requested = U32(parsed);
        const U32 capacity = profile_capacity(requested);
        metadata.name = object.name;
        const auto decision = expert
            ? omlx::m5::detail::forward::ForwardLinearDecision{
                  {.library = "moe",
                   .function = select_grouped_expert_kernel(object.format),
                   .family = "grouped_a16"}, "baseline",
                  "expert_candidate_unqualified", false}
            : resolve_linear_route(metadata, n, k, capacity, device,
                                   short_routes, {}, row_exact
                                       ? omlx::m5::LinearNumericPolicy::RowExactPortableV1
                                       : omlx::m5::LinearNumericPolicy::Legacy);
        const auto &route = decision.route;
        const auto grid = linear_grid(route, object.format, n, capacity);
        nlohmann::json row = {{"object", object.name},
                        {"format", object.format},
                        {"n", n},
                        {"k", k},
                        {"requested_tokens", requested},
                        {"profile_capacity", capacity},
                        {"library", route.library},
                        {"kernel", route.function},
                        {"family", route.family},
                        {"selection_source", decision.source},
                        {"fallback_reason", decision.fallback_reason},
                        {"batch", 1},
                        {"width", capacity},
                        {"short_route", route.lane || route.matrix}};
        row["numeric_contract_id"] = route.numeric_contract_id;
        row["row_exact_group"] = route.row_exact_group;
        row["row_exact_eligible"] = route.row_exact;
        row["whole_program_row_exact_qualified"] = false;
        if (expert && row_exact)
          row["fallback_reason"] = "expert_row_exact_unqualified";
        if (!expert) {
          row["grid"] = {{"width", grid.width},
                          {"height", grid.height},
                          {"depth", grid.depth}};
          row["threads_per_threadgroup"] = 128;
        } else {
          // Grouped MoE uses routed token geometry, not the rank-2 Linear grid.
          row["grid"] = nullptr;
          row["threads_per_threadgroup"] = nullptr;
        }
        rows.push_back(std::move(row));
      }
    }
    std::cout << nlohmann::json(
                     {
                         {"schema", "omlx-m5-forward-route-inspection-v1"},
                         {"scope",
                          "metadata-only prediction from the exact selector "
                          "used by production lowering; not executed on GPU"},
                         {"artifact", argv[1]},
                         {"source_revision",
                          artifact.summary().source_revision},
                         {"device_name", device},
                         {"routes", rows},
                     })
                     .dump(2)
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "forward route inspection failed: " << error.what() << '\n';
    return 1;
  }
}
