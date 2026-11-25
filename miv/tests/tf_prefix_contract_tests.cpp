#include "omlx_m5/qwen_forward.h"
#include "omlx_m5/execution_contract.h"
#include "omlx_m5/engine.h"
#include <array>
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
using namespace omlx::m5;using U32=std::uint32_t;using Json=nlohmann::json;using namespace std::chrono_literals;
namespace {
void require(bool b,const char *m){if(!b)throw std::runtime_error(m);}
template<class F> void rejects(F f){bool denied=false;try{f();}catch(const std::invalid_argument&){denied=true;}require(denied,"tampered continuation accepted");}
std::vector<U32> prompt(U32 n,U32 salt=256){std::vector<U32> out(n);for(U32 i=0;i<n;++i)out[i]=(i*71+salt)%257;return out;}
GpuSamplingOptions sampling(){return {.temperature=.7f,.top_k=20,.top_p=.95f,.seed=UINT64_MAX,.contract=SamplingContract::PositionCoupledV1};}
void prefill(QwenForward &p,const std::vector<U32>&tokens){for(std::size_t pos=0;pos<tokens.size();){auto n=prefill_next_chunk(pos,tokens.size(),SamplingContract::PositionCoupledV1);(void)p.run(std::span(tokens).subspan(pos,n),sampling());pos+=n;}}
Json continuation(const std::filesystem::path &fixtures,const std::filesystem::path &libraries,const char *name){
 const auto dir=fixtures/name;auto config=load_qwen_model_config(dir/"source/config.json");auto artifact=ArtifactModel::open(dir/"model.omlxm5",true);
 auto context=MetalContext::create(),reference_context=MetalContext::create();
 auto weights=materialize_model(artifact,plan_model_materialization(artifact,context),context);
 auto reference_weights=materialize_model(artifact,plan_model_materialization(artifact,reference_context),reference_context);
 auto layers=build_qwen_layer_catalog(artifact,config,2048,64);const std::array<U32,2> taps{0,1};
 QwenForward p(context,weights,layers,config,2048,libraries,taps,false,true,false,{},false,false,{},LinearNumericPolicy::RowExactPortableV1,SamplingContract::PositionCoupledV1,AttentionNumericPolicy::RowExactV1);
 QwenForward reference(reference_context,reference_weights,layers,config,2048,libraries,taps,false,true,false,{},false,false,{},LinearNumericPolicy::RowExactPortableV1,SamplingContract::PositionCoupledV1,AttentionNumericPolicy::RowExactV1);
 p.bind_execution_identity(execution_identity_digest({{"model",execution_artifact_digest(artifact)},{"draft","fixture"},{"chunk",std::string(prefill_chunk_policy_id(SamplingContract::PositionCoupledV1))}}));
 const std::array<U32,3> suffix{37,43,53};U32 cases=0,negatives=0;
 for(U32 width:{1U,63U,64U,65U,127U,128U,129U,1023U,1024U,1025U}){
  p.reset();reference.reset();auto tokens=prompt(width);prefill(p,tokens);prefill(reference,tokens);
  auto saved=p.checkpoint();require(saved.execution_identity==p.execution_identity() && saved.payload_digest.size()==64 && saved.accounted_bytes==p.checkpoint_bytes(),"checkpoint omitted contract or bytes");
  auto expected=p.run(suffix,sampling(),true);const auto expected_state=p.inspect_recurrent_state();
  p.reset();
  for(U32 mutation=0;mutation<7;++mutation){auto bad=saved;
    if(mutation==0)bad.execution_identity[0]^=1;
    if(mutation==1)bad.payload_digest[0]^=1;
    if(mutation==2)++bad.position;
    if(mutation==3)++bad.kv_lease;
    if(mutation==4)bad.recurrent[0][0]^=std::byte{1};
    if(mutation==5)bad.feature_history[0]^=std::byte{1};
    if(mutation==6)++bad.owner_id;
    const auto before=p.inspect_recurrent_state();rejects([&]{p.restore(bad);});require(before==p.inspect_recurrent_state() && p.position()==0,"failed restore mutated state");++negatives;
  }
  p.restore(saved);const auto original_pages=p.inspect_kv_page_table();auto actual=p.run(suffix,sampling(),true);
  if(width%64)require(original_pages.at(width/64)!=p.inspect_kv_page_table().at(width/64),"shared partial tail did not use COW");
  require(actual.token==expected.token && actual.hidden==expected.hidden && actual.logits==expected.logits && p.inspect_recurrent_state()==expected_state,"restored branch differs");
  const auto bytes=std::uint64_t(p.position())*taps.size()*config.hidden_size*2;
  auto teacher=reference.run(suffix,sampling(),true);
  require(actual.logits==teacher.logits && p.feature_history().read(0,bytes)==reference.feature_history().read(0,bytes),"restored feature/KV continuation differs");
  p.release(saved);p.reset();rejects([&]{p.restore(saved);});++negatives;++cases;
 }
 rejects([&]{p.bind_execution_identity("replacement");});++negatives;
 return {{"model",name},{"restore_cases",cases},{"negative_cases",negatives},{"state_feature_logits_equal",true},{"passed",true}};
}
std::vector<std::int32_t> run(Engine &e,const SubmitRequest&r){const auto id=e.submit(r);std::vector<std::int32_t> out;auto deadline=std::chrono::steady_clock::now()+60s;
 while(std::chrono::steady_clock::now()<deadline)for(const auto &event:e.poll_events(10ms)){
  require(event.request_id==id,"foreign prefix event");if(event.kind==EventKind::tokens)out.insert(out.end(),event.tokens.begin(),event.tokens.end());
  else{if(event.kind==EventKind::failed)throw std::runtime_error(event.error);require(event.kind==EventKind::finished,"prefix request not finished");return out;}}
 throw std::runtime_error("prefix request timeout");}
LoadOptions options(const std::filesystem::path &fixtures,const std::filesystem::path &draft,const std::filesystem::path &libraries,const char *name,U32 depth,bool cache){
 LoadOptions o;o.model_dir=(fixtures/name/"source").string();o.max_context=2048;o.sampling_contract="position_coupled_v1";
 o.path_overrides={{"artifact",(fixtures/name/"model.omlxm5").string()},{"kernel_directory",libraries.string()}};
 if(cache)o.path_overrides["prefix_cache_bytes"]="67108864";
 if(depth){o.draft_dir=(draft/"source").string();o.path_overrides["draft_artifact"]=(draft/"model.omlxm5").string();o.speculative={.backend=SpeculativeBackend::dflash2,.draft_tokens=depth};}
 return o;
}
Json engines(const std::filesystem::path &fixtures,const std::filesystem::path &draft,const std::filesystem::path &libraries,const char *name,U32 depth){
 auto cached=Engine::create(options(fixtures,draft,libraries,name,depth,true)),cold=Engine::create(options(fixtures,draft,libraries,name,depth,false));
 SubmitRequest r;r.max_new_tokens=8;r.sampling={.temperature=.7f,.top_p=.95f,.top_k=20,.seed=UINT64_MAX};U32 cases=0;
 for(U32 length:{63U,65U,1023U,1024U,1025U}){
  auto tokens=prompt(length);r.prompt_tokens.assign(tokens.begin(),tokens.end());auto before=cached->stats();auto expected=run(*cold,r);
  require(run(*cached,r)==expected,"prefix miss/partial hit differs from cold");auto after=cached->stats();
  if(length==65 || length==1023 || length==1024)require(after.prefix_hits==before.prefix_hits,"illegal old tail reused");
  if(length==1025)require(after.prefix_partial_hits==before.prefix_partial_hits+1,"legal absolute chunk did not resume");
  before=cached->stats();require(run(*cached,r)==expected,"exact prefix hit differs");require(cached->stats().prefix_hits==before.prefix_hits+1,"exact hit missing");cases+=2;
 }
 // Cross-seed and cross-temperature prefix cache HITs with exact PositionCoupledV1 teacher parity!
 for(const auto &s:std::array<SamplingSpec,3>{{
   {.temperature=.7f,.top_p=.95f,.top_k=20,.seed=123},
   {.temperature=.25f,.top_p=.8f,.top_k=8,.seed=987654321ULL},
   {.temperature=0.0f,.top_p=1.0f,.top_k=0,.seed=42},
 }}){
  auto before=cached->stats();r.sampling=s;
  require(run(*cached,r)==run(*cold,r),"PositionCoupledV1 cross-seed/temp hit differs from cold teacher");
  require(cached->stats().prefix_hits==before.prefix_hits+1 && cached->stats().prefix_misses==before.prefix_misses,
          "PositionCoupledV1 cross-seed/temp missed prefix cache");
  ++cases;
 }
 // Cross-seed partial chunk hit from cached 1024 boundary onto new 1026-token suffix
 {
  auto ext=prompt(1026);ext[1024]=11;ext[1025]=29;
  SubmitRequest pr=r;pr.prompt_tokens.assign(ext.begin(),ext.end());
  pr.sampling={.temperature=.9f,.top_p=.9f,.top_k=16,.seed=777};
  auto before=cached->stats();
  require(run(*cached,pr)==run(*cold,pr),"PositionCoupledV1 cross-seed 1024 partial hit differs");
  require(cached->stats().prefix_partial_hits==before.prefix_partial_hits+1 && cached->stats().prefix_misses==before.prefix_misses,
          "PositionCoupledV1 cross-seed 1024 partial hit missing");
  ++cases;
 }
 // Multi-entry LRU retention: a 63-token MRU entry (1 KV page) coexists with a
 // divergent 1025-token miss (17 KV pages; 1 + 17 <= 33 physical pages) without wipe!
 {
  auto short_mru=r;auto t63=prompt(63,99);short_mru.prompt_tokens.assign(t63.begin(),t63.end());
  require(run(*cached,short_mru)==run(*cold,short_mru),"short MRU entry miss differs");++cases;
  r.sampling={.temperature=.7f,.top_p=.95f,.top_k=20,.seed=123};
  r.prompt_tokens[1000]^=1;require(run(*cached,r)==run(*cold,r),"divergent suffix differs");++cases;
  short_mru.sampling.seed=55555;
  auto before=cached->stats();
  require(run(*cached,short_mru)==run(*cold,short_mru),"retained MRU entry differs after divergent miss");
  require(cached->stats().prefix_hits==before.prefix_hits+1,"divergent miss wiped retained MRU prefix");
  ++cases;
 }
 r.max_new_tokens=256;auto id=cached->submit(r);require(cached->cancel(id),"prefix cancel not accepted");bool aborted=false;auto deadline=std::chrono::steady_clock::now()+60s;
 while(!aborted && std::chrono::steady_clock::now()<deadline)for(const auto &event:cached->poll_events(10ms))if(event.request_id==id && event.kind==EventKind::aborted)aborted=true;
 require(aborted,"prefix cancel did not terminate");r.max_new_tokens=8;require(run(*cached,r)==run(*cold,r),"cancel polluted cached continuation");++cases;
 cached->shutdown();cold->shutdown();
 auto tiny_options=options(fixtures,draft,libraries,name,depth,true);tiny_options.path_overrides["prefix_cache_bytes"]="1";
 auto tiny=Engine::create(tiny_options);r.prompt_tokens={1,7,3,9};r.max_new_tokens=1;
 (void)run(*tiny,r);(void)run(*tiny,r);require(tiny->stats().prefix_hits==0 && tiny->stats().prefix_misses==2,"tiny coupled cache budget retained state");tiny->shutdown();
 return {{"model",name},{"d",depth},{"requests_compared",cases},{"legal_chunk_partial_hit",true},{"illegal_tail_not_reused",true},{"seed_invalidation",true},{"cross_seed_temperature_hit",true},{"multi_entry_lru_retained",true},{"cancel_reuse",true},{"tiny_budget_checked",true},{"passed",true}};
}
}
int main(int argc,char **argv){try{
 if(argc!=4)throw std::invalid_argument("tf_prefix_contract_tests FIXTURES DRAFT LIBRARIES");Json states=Json::array(),requests=Json::array();
 for(const char *name:{"affine","mxfp4","moe-affine","moe-mxfp4"}){states.push_back(continuation(argv[1],argv[3],name));for(U32 d=0;d<=7;++d)requests.push_back(engines(argv[1],argv[2],argv[3],name,d));}
 std::cout<<Json({{"schema","omlx-m5-p08-tf-prefix-v1"},{"continuations",states},{"engines",requests},{"passed",true}}).dump(2)<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<"P08-TF: "<<e.what()<<'\n';return 1;}}
