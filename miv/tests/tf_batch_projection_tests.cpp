#include "omlx_m5/bounded_batch.h"
#include "omlx_m5/artifact.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <nlohmann/json.hpp>
using namespace omlx::m5;using U32=std::uint32_t;using Json=nlohmann::json;
namespace {
void require(bool b,const char*m){if(!b)throw std::runtime_error(m);}
Json mode(const std::filesystem::path&fixtures,const std::filesystem::path&libs,const char*name,U32 b){
 auto cfg=load_qwen_model_config(fixtures/name/"source/config.json");auto artifact=ArtifactModel::open(fixtures/name/"model.omlxm5",true);auto context=MetalContext::create();
 auto weights=materialize_model(artifact,plan_model_materialization(artifact,context),context);auto layers=build_qwen_layer_catalog(artifact,cfg,2048,64);
 std::vector<std::unique_ptr<QwenForward>> owned;std::vector<QwenForward*> programs;
 for(U32 i=0;i<b;++i){owned.push_back(std::make_unique<QwenForward>(context,weights,layers,cfg,2048,libs,std::span<const U32>{},false,false,false,std::nullopt,false,false,std::nullopt,
  LinearNumericPolicy::RowExactPortableV1,SamplingContract::PositionCoupledV1,AttentionNumericPolicy::RowExactV1));programs.push_back(owned.back().get());}
 SharedHeadProjection projection(context,programs,libs);CoupledBatchSampler sampler(context,programs,libs);
 const auto projection_addresses=projection.resource_addresses(),sampler_addresses=sampler.resource_addresses();U32 cases=0;
 for(U32 w=1;w<=8;++w){std::vector<BatchRow> rows;std::vector<PositionSamplingRow> keys;std::vector<std::vector<std::byte>> expected;std::vector<U32> tokens;
  for(U32 slot=0;slot<b;++slot){auto&p=*programs[slot];p.reset();const U32 width=1+(w+slot-1)%8;std::vector<U32> input(width);for(U32 t=0;t<width;++t)input[t]=(71*t+256+slot)%257;
   GpuSamplingOptions s{.temperature=.7f,.top_k=20,.top_p=.95f,.seed=UINT64_MAX-slot,.contract=SamplingContract::PositionCoupledV1};auto r=p.run(input,s,true);
   rows.push_back({slot,width,1});keys.push_back({.seed=s.seed,.position=p.position(),.top_k=s.top_k,.temperature=s.temperature,.top_p=s.top_p});tokens.push_back(r.token);
   expected.push_back(p.batch_projection_output_buffer().read(0,std::uint64_t(width)*cfg.vocab_size*2));}
  for(bool icb:{false,true}){auto result=projection.project(rows,icb);require(result.dispatch_count>0 && projection.last_rows()>b,"projection did not execute true multi-row Linear");
   for(U32 slot=0;slot<b;++slot)require(programs[slot]->batch_projection_output_buffer().read(0,expected[slot].size())==expected[slot],"shared head row bit mismatch");
   auto sampled=sampler.sample(rows,keys,icb);require(sampled.tokens==tokens,"shared head actual sampling differs from solo");++cases;}
  std::reverse(rows.begin(),rows.end());std::reverse(keys.begin(),keys.end());std::reverse(tokens.begin(),tokens.end());
  (void)projection.project(rows);auto shuffled=sampler.sample(rows,keys);require(shuffled.tokens==tokens,"compact slot order changed draw");++cases;
  rows[0].valid=0;(void)projection.project(rows);auto masked=sampler.sample(rows,keys);for(U32 i=1;i<b;++i)require(masked.tokens[i]==tokens[i],"inactive slot changed another row");++cases;
  require(projection.resource_addresses()==projection_addresses && sampler.resource_addresses()==sampler_addresses,"batch resource addresses moved");
 }
 bool denied=false;const std::array<BatchRow,2> repeated{{{0,1,1},{0,1,1}}};try{(void)projection.project(repeated);}catch(const std::invalid_argument&){denied=true;}require(denied,"duplicate active slot accepted");
 const auto actual=projection.execution_route();Json widths=Json::array(),layout=Json::array();
 for(U32 slot=0;slot<b;++slot){const U32 width=1+(7+slot)%8;widths.push_back(width);
  for(U32 position=0;position<width;++position)layout.push_back({{"logical_request",slot},{"position",position},{"valid",slot!=b-1}});}
 return {{"model",name},{"b",b},{"cases",cases},{"real_qwen_norm_head_caller",true},{"ragged_w1_8",true},{"reorder_mask_stable_resources",true},
  {"workload_widths",widths},{"row_layout",layout},{"actual_m",projection.last_rows()},
  {"executed_route",{{"function",actual.function},{"library",actual.library},{"format",actual.format},{"activation","A16"},
    {"n",actual.n},{"k",actual.k},{"m",projection.last_rows()},{"kind","shared_projection"},
    {"numeric_contract_id",actual.numeric_contract_id},{"row_exact_group",actual.row_exact_group},
    {"grid",{actual.grid.width,actual.grid.height,actual.grid.depth}},
    {"threads",{actual.threads_per_threadgroup.width,actual.threads_per_threadgroup.height,actual.threads_per_threadgroup.depth}}}},
  {"passed",true}};
}
}
int main(int argc,char**argv){try{if(argc!=3)throw std::invalid_argument("batch_projection_tests FIXTURES LIBRARIES");Json modes=Json::array();
 for(const char*name:{"affine","mxfp4","moe-affine","moe-mxfp4"})for(U32 b:{2U,4U,8U})modes.push_back(mode(argv[1],argv[2],name,b));
 std::cout<<Json({{"schema","omlx-m5-p10-tf-projection-v1"},{"modes",modes},{"passed",true}}).dump(2)<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<"P10 projection: "<<e.what()<<'\n';return 1;}}
