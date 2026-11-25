#pragma once
#include "omlx_m5/execution_contract.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <array>
#include <bit>
#include <set>
namespace omlx::m5::detail {
inline void validate_route_trace_v2(const nlohmann::json &trace) {
 try {
  if(trace.at("schema")!="omlx-m5-p12-route-trace-v2")throw std::invalid_argument("schema");
  const auto &raw=trace.at("provenance");if(!raw.is_object())throw std::invalid_argument("provenance object");
  ExecutionIdentityFields fields=raw.get<ExecutionIdentityFields>();
  const std::array<const char*,15> required{"artifact","config","compiled_source","kernels","execution","numeric","sampling","chunk","row_layout","router_profile","expert_profile","input","route_ids","route_scores","scope"};
  for(const auto *key:required)if(!fields.contains(key) || fields.at(key).empty())throw std::invalid_argument("identity field");
  for(const auto *key:{"artifact","config","compiled_source","kernels","execution","input","route_ids","route_scores"}){
   const auto &hash=fields.at(key);if(hash.size()!=64 || hash.find_first_not_of("0123456789abcdef")!=std::string::npos)throw std::invalid_argument("identity hash");
  }
  const auto contract=parse_sampling_contract(fields.at("sampling"));
  if(fields.at("chunk")!=prefill_chunk_policy_id(contract) || fields.at("row_layout")!="single-slot.absolute-position.rows.v1" ||
    fields.at("scope")!="committed_forward" ||
    (fields.at("numeric")!="legacy_route_v1" && fields.at("numeric")!="linear_row_exact_portable_v1"))throw std::invalid_argument("contract");
  if(trace.at("provenance_sha256").get<std::string>()!=execution_identity_digest(fields))throw std::invalid_argument("provenance digest");
  const auto unsigned_value=[](const nlohmann::json &v)->std::uint64_t{
   if(!v.is_number_integer() || (v.is_number_integer() && !v.is_number_unsigned() && v.get<std::int64_t>()<0))throw std::invalid_argument("unsigned integer");
   return v.get<std::uint64_t>();
  };
  const auto count=unsigned_value(trace.at("tokens")),start=unsigned_value(trace.at("start_position"));
  if(!count || count>1024 || start>UINT32_MAX-count || unsigned_value(trace.at("experts"))!=256 || unsigned_value(trace.at("top_k"))!=8)throw std::invalid_argument("row envelope");
  std::vector<std::uint32_t> ids;std::vector<std::uint16_t> scores;
  for(const auto *name:{"expert_ids","scores","scores_bf16_u16","row_positions","row_slots","row_valid"})
   if(!trace.at(name).is_array() || trace.at(name).size()!=count)throw std::invalid_argument("row layout shape");
  for(std::size_t row=0;row<count;++row){
   if(unsigned_value(trace.at("row_positions").at(row))!=start+row || unsigned_value(trace.at("row_slots").at(row))!=0 || unsigned_value(trace.at("row_valid").at(row))!=1)throw std::invalid_argument("row identity");
   if(trace.at("expert_ids").at(row).size()!=8 || trace.at("scores_bf16_u16").at(row).size()!=8 || trace.at("scores").at(row).size()!=8)throw std::invalid_argument("route width");
   std::set<std::uint32_t> unique;double mass=0;
   for(std::size_t slot=0;slot<8;++slot){auto id=unsigned_value(trace.at("expert_ids").at(row).at(slot)),code=unsigned_value(trace.at("scores_bf16_u16").at(row).at(slot));
    if(id>=256 || code>65535 || !unique.insert(std::uint32_t(id)).second)throw std::invalid_argument("route ID/code");
    const float value=std::bit_cast<float>(std::uint32_t(code)<<16),declared=trace.at("scores").at(row).at(slot).get<float>();
    if(!std::isfinite(value) || value<0 || value!=declared)throw std::invalid_argument("route score");mass+=value;ids.push_back(std::uint32_t(id));scores.push_back(std::uint16_t(code));
   }
   if(std::abs(mass-1.0)>.02)throw std::invalid_argument("score mass");
  }
  if(fields.at("route_ids")!=artifact_sha256_hex(std::as_bytes(std::span(ids))) || fields.at("route_scores")!=artifact_sha256_hex(std::as_bytes(std::span(scores))))throw std::invalid_argument("route payload digest");
 }catch(const std::exception &e){throw std::invalid_argument(std::string("P12 v2 trace provenance/layout invalid: ")+e.what());}
}
} // namespace omlx::m5::detail
