#include "omlx_m5/qwen_forward.h"
#include "omlx_m5/artifact.h"
#include "omlx_m5/target_verify_gpu.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
using namespace omlx::m5;using U32=std::uint32_t;using Json=nlohmann::json;
namespace {
void require(bool b,const char *m){if(!b)throw std::runtime_error(m);}
Json mode(const std::filesystem::path &fixtures,const std::filesystem::path &libraries,const char *name){
 const auto directory=fixtures/name;const auto cfg=load_qwen_model_config(directory/"source/config.json");
 auto artifact=ArtifactModel::open(directory/"model.omlxm5",true);
 auto context=MetalContext::create(),teacher_context=MetalContext::create();
 auto weights=materialize_model(artifact,plan_model_materialization(artifact,context),context);
 auto teacher_weights=materialize_model(artifact,plan_model_materialization(artifact,teacher_context),teacher_context);
 auto layers=build_qwen_layer_catalog(artifact,cfg,2048,64);const std::array<U32,2> taps{0,1};
 QwenForward target(context,weights,layers,cfg,2048,libraries,taps,false,true,false,{},false,false,{},
   LinearNumericPolicy::RowExactPortableV1,SamplingContract::PositionCoupledV1,AttentionNumericPolicy::RowExactV1);
 QwenForward teacher(teacher_context,teacher_weights,layers,cfg,2048,libraries,taps,false,true,false,{},false,false,{},
   LinearNumericPolicy::RowExactPortableV1,SamplingContract::PositionCoupledV1,AttentionNumericPolicy::RowExactV1);
 TargetGpuVerifier verifier(context,target.verification_logits_buffer(),cfg.vocab_size,libraries,SamplingContract::PositionCoupledV1);
 U32 cases=0,abort_cases=0;
 for(U32 depth=0;depth<=7;++depth)for(float temperature:{0.f,.7f}){
   GpuSamplingOptions sampling{.temperature=temperature,.top_k=20,.top_p=.95f,.seed=UINT64_MAX,
      .contract=SamplingContract::PositionCoupledV1};
   std::vector<U32> prompt(63+depth%3);for(U32 i=0;i<prompt.size();++i)prompt[i]=(i*71+256)%cfg.vocab_size;
   teacher.reset();auto initial=teacher.run(prompt,sampling);std::vector<U32> chain{initial.token},samples;
   for(U32 row=0;row<=depth;++row){const std::array<U32,1> input{chain.back()};auto step=teacher.run(input,sampling,true);
     samples.push_back(step.token);if(row<depth)chain.push_back(step.token);}
   for(U32 accepted=0;accepted<=depth;++accepted){
     auto input=chain;if(accepted<depth)input[accepted+1]=(input[accepted+1]+1)%cfg.vocab_size;
     target.reset();teacher.reset();(void)target.run(prompt,sampling);(void)teacher.run(prompt,sampling);
     const auto before=target.inspect_recurrent_state();const auto verified=target.verify(input,sampling,true);
     require(target.position()==prompt.size() && target.inspect_recurrent_state()==before,"verify modified committed state");
     require(std::all_of(verified.logits.begin(),verified.logits.end(),[](float x){return std::isfinite(x);}),"verify logits nonfinite");
     std::vector<SelectorStep> proposals(depth);for(U32 row=0;row<depth;++row)proposals[row].token=input[row+1];
     auto selected=verifier.select(proposals,sampling,target.position());
     std::vector<std::int32_t> expected(samples.begin(),samples.begin()+accepted+1);
     require(selected.verification.licensed_outputs==expected && selected.verification.accepted_proposals==accepted &&
       selected.verification.used_correction==(accepted<depth),"actual forward equality acceptance differs from serial path");
     const auto count=U32(expected.size());(void)target.commit_prefix(count);
     for(U32 row=0;row<count;++row){auto step=teacher.run(std::span(input).subspan(row,1),sampling,true);
       require(std::equal(step.logits.begin(),step.logits.end(),verified.logits.begin()+U32(cfg.vocab_size)*row),"accepted-row logits not exact");}
     require(target.position()==teacher.position() && target.inspect_recurrent_state()==teacher.inspect_recurrent_state(),"coupled Replay state differs");
     const auto feature_bytes=std::uint64_t(target.position())*taps.size()*cfg.hidden_size*2;
     require(target.feature_history().read(0,feature_bytes)==teacher.feature_history().read(0,feature_bytes),"coupled feature frontier differs");
     const std::array<U32,1> continuation{U32(expected.back())};auto a=target.run(continuation,sampling,true),b=teacher.run(continuation,sampling,true);
     require(a.logits==b.logits && a.hidden==b.hidden && a.token==b.token && target.inspect_recurrent_state()==teacher.inspect_recurrent_state(),"pending correction/bonus continuation differs");
     const auto committed=target.inspect_recurrent_state();const auto frontier=target.position();
     (void)target.verify(input,sampling);target.abort_verify();
     require(target.position()==frontier && target.inspect_recurrent_state()==committed,"aborted coupled verify polluted state");
     ++cases;++abort_cases;
   }
 }
 return {{"model",name},{"accepted_prefix_cases",cases},{"abort_cases",abort_cases},{"same_contract_state_logits_features",true},{"passed",true}};
}
}
int main(int argc,char **argv){try{
 if(argc!=3)throw std::invalid_argument("tf_coupled_state_tests FIXTURES LIBRARIES");Json modes=Json::array();
 for(const char *name:{"affine","mxfp4","moe-affine","moe-mxfp4"})modes.push_back(mode(argv[1],argv[2],name));
 std::cout<<Json({{"schema","omlx-m5-p04-tf-state-v1"},{"modes",modes},{"d0_7_all_prefixes",true},{"passed",true}}).dump(2)<<'\n';return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
