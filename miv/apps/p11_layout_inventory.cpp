#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/p11_tensor_view.h"

using namespace omlx::m5;
using Json = nlohmann::json;
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument("p11_layout_inventory ARTIFACT OUTPUT");
    auto artifact = ArtifactModel::open(argv[1], false);
    Json records = Json::array();
    std::map<std::string, std::uint64_t> counts;
    for (const auto &object : artifact.objects()) {
      if (!object.format.starts_with("mxfp4_") &&
          !object.format.starts_with("oq_affine_"))
        continue;
      Json record = {{"name", object.name}, {"format", object.format},
                     {"layout", object.layout}, {"shape", object.shape},
                     {"padded_k", object.padded_k}, {"eligible", false}};
      std::string category;
      try {
        const auto view = native_mxfp4_weight_view(
            object, 0, std::uint32_t(object.shape.at(0)));
        record["eligible"] = true;
        record["code_row_bytes"] = view.data_row_bytes;
        record["scale_row_bytes"] = view.scale_row_bytes;
        category = "direct_mxfp4_parent_view";
      } catch (const std::invalid_argument &error) {
        category = object.format.starts_with("oq_affine_")
                       ? "affine_scale_bias_not_native_e8m0"
                       : object.shape.size() != 2
                           ? "expert_or_nonmatrix_layout"
                           : object.padded_k % 256
                               ? "fp4_outer_stride_below_128_bytes"
                               : "other_layout_or_plane_rejection";
        record["reason"] = error.what();
      }
      record["category"] = category;
      ++counts[category];
      records.push_back(std::move(record));
    }
    const Json report = {
        {"schema", "omlx-m5-p11-layout-inventory-v1"},
        {"artifact", argv[1]},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"payload_verified", false},
        {"categories", counts},
        {"records", records},
        {"production_route_changed", false}};
    std::ofstream output(argv[2]);
    output << report.dump(2) << '\n';
    if (!output) throw std::runtime_error("P11 inventory output failed");
    std::cout << argv[2] << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "P11 inventory failure: " << error.what() << '\n';
    return 1;
  }
}
