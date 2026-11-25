#include "../src/forward_plan.h"
#include "omlx_m5/position_sampling.h"
#include "omlx_m5/engine.h"
#include "omlx_m5/quantization.h"
#include <iostream>
#include <numeric>
#include <nlohmann/json.hpp>

using namespace omlx::m5;using namespace omlx::m5::detail::forward;using Json=nlohmann::json;
namespace {
void check(bool b,std::string_view m){if(!b)throw std::runtime_error(std::string(m));}
template<class F>void denied(F f){try{f();}catch(const std::invalid_argument&){return;}throw std::runtime_error("invalid keyed input admitted");}
Json batch(const std::filesystem::path &libs,U32 vocab,U32 rows) {
  auto c=MetalContext::create();std::vector<std::uint16_t> logits(U64(rows)*vocab);
  for(U32 r=0;r<rows;++r)for(U32 v=0;v<vocab;++v) {
    const float value=r%5==0?0.f:float(int((v*17+r*31)%73)-36)/8;
    logits[U64(r)*vocab+v]=float_to_bf16(value);
  }
  auto gpu=c.upload(std::as_bytes(std::span(logits)));PositionKeyedSampler sampler(c,gpu,vocab,rows,libs);
  std::vector<PositionSamplingRow> opts(rows);
  for(U32 r=0;r<rows;++r)opts[r]={.seed=r%7==0?UINT64_MAX:U64(r)*100003,
    .position=(r%3==0?(1ULL<<40):0)+r,.purpose=r%2?target_sampling_purpose:draft_sampling_purpose,
    .top_k=std::min(vocab,r%4==0?1U:r%4==1?2U:r%4==2?20U:64U),
    .temperature=r%7==0?0.f:r%3==0?.5f:r%3==1?1.f:3.f,.top_p=r%4==0?0.f:r%4==1?.5f:r%4==2?.95f:1.f};
  auto result=sampler.sample(opts);auto direct=sampler.sample(opts,false);check(result.tokens==direct.tokens,"keyed ICB/direct mismatch");
  std::vector<U32> ids(vocab);std::iota(ids.begin(),ids.end(),0);std::vector<float> scores(vocab);
  U64 compared=0;
  for(U32 row=0;row<rows;++row) {
    for(U32 v=0;v<vocab;++v)scores[v]=bf16_to_float(logits[U64(row)*vocab+v]);
    check(position_sample_candidates(ids,scores,opts[row])==result.tokens[row],"keyed host/GPU sample mismatch");++compared;
    gpu.write(std::as_bytes(std::span(logits).subspan(U64(row)*vocab,vocab)));
    auto one=sampler.sample(std::span(opts).subspan(row,1));check(one.tokens[0]==result.tokens[row],"keyed solo/window mismatch");
  }
  std::vector<std::uint16_t> shuffled(logits.size());std::vector<PositionSamplingRow> permuted(rows);
  for(U32 r=0;r<rows;++r){U32 from=rows-1-r;std::copy_n(logits.begin()+U64(from)*vocab,vocab,shuffled.begin()+U64(r)*vocab);permuted[r]=opts[from];}
  gpu.write(std::as_bytes(std::span(shuffled)));auto reordered=sampler.sample(permuted);
  for(U32 r=0;r<rows;++r)check(reordered.tokens[r]==result.tokens[rows-1-r],"keyed row order changed draw");
  permuted[0].valid=0;auto masked=sampler.sample(permuted);for(U32 r=1;r<rows;++r)check(masked.tokens[r]==reordered.tokens[r],"inactive row consumed another RNG");
  auto invalid=opts[0];invalid.temperature=-1;denied([&]{(void)sampler.sample(std::span(&invalid,1));});
  invalid=opts[0];invalid.temperature=1;invalid.top_k=0;denied([&]{(void)sampler.sample(std::span(&invalid,1));});
  invalid=opts[0];invalid.top_p=1.1f;denied([&]{(void)sampler.sample(std::span(&invalid,1));});
  invalid=opts[0];invalid.purpose=123;denied([&]{(void)sampler.sample(std::span(&invalid,1));});
  shuffled[0]=0x7fc0;gpu.write(std::as_bytes(std::span(shuffled)));permuted[0].valid=1;
  bool rejected=false;try{(void)sampler.sample(permuted);}catch(const std::runtime_error&){rejected=true;}check(rejected,"NaN logits silently ignored");
  return {{"vocab",vocab},{"rows",rows},{"host_gpu_rows",compared},{"solo_equal",true},{"reorder_equal",true},
          {"inactive_mask_checked",true},{"nonfinite_rejected",true},{"compact_fallback_rows",result.compact_fallback_rows},{"passed",true}};
}
}
int main(int argc,char **argv) {
  try {
    if(argc!=2)throw std::invalid_argument("tf_position_sampling_tests LIBRARIES");
    LoadOptions load;load.sampling_contract="position_coupled_v1";load.max_context=1024;
    denied([&]{(void)Engine::create_fake(load);});
    const std::filesystem::path libraries(argv[1]);
    Json results=Json::array();results.push_back(batch(argv[1],257,128));results.push_back(batch(argv[1],248320,8));
    auto c=MetalContext::create();constexpr U32 count=64;
    std::vector<PositionSamplingRow> params(count);std::vector<U32> ids(count);
    for(U32 i=0;i<count;++i){params[i]={.seed=i%2?UINT64_MAX:U64(i)*92821,.position=(1ULL<<40)+i,
      .purpose=i%2?target_sampling_purpose:draft_sampling_purpose};ids[i]=i*7919;}
    auto pb=c.upload(std::as_bytes(std::span(params))),ib=c.upload(std::as_bytes(std::span(ids))),yb=c.make_buffer(count*4);
    Encoder e(c,libraries,1,"tf.position.uniform");auto &d=e.emit("position_sampling","omlx_m5_position_uniform_vectors",{.width=count},{.width=32},
      {{&pb,0,pb.size()},{&ib,0,ib.size()},{&yb,0,yb.size()}});e.dynamic(d,3,words({count}));auto p=e.finish();
    const std::array calls{MetalCommandInvocation{.plan=&p.plan,.bindings=p.bindings,.parameters=&p.parameters}};(void)c.execute_batch(calls);
    auto bytes=yb.read();std::vector<float> uniforms(count);std::memcpy(uniforms.data(),bytes.data(),bytes.size());
    for(U32 i=0;i<count;++i){float expected=position_sampling_uniform(params[i].seed,params[i].position,ids[i],params[i].purpose);
      check(expected>0&&expected<1&&std::bit_cast<U32>(expected)==std::bit_cast<U32>(uniforms[i]),"uniform hash CPU/GPU bit mismatch");}
    // Distribution and top-p boundary through the actual GPU, 20k positions.
    constexpr U32 rows=128,vocab=3;std::vector<std::uint16_t> logits(rows*vocab);
    for(U32 r=0;r<rows;++r)for(U32 v=0;v<3;++v)logits[r*3+v]=float_to_bf16(std::log(v==0?.5f:v==1?.3f:.2f));
    auto xb=c.upload(std::as_bytes(std::span(logits)));PositionKeyedSampler sampler(c,xb,3,rows,argv[1]);
    std::array<U32,3> counts{};std::vector<PositionSamplingRow> opts(rows);
    for(U32 first=0;first<20000;first+=rows) {
      const U32 active=std::min(rows,20000-first);
      for(U32 r=0;r<active;++r)opts[r]={.seed=77,.position=first+r,.top_k=3,.temperature=1,.top_p=1};
      auto result=sampler.sample(std::span(opts).first(active));for(auto token:result.tokens)++counts[token];
    }
    check(std::abs(double(counts[0])/20000-.5)<.02 && std::abs(double(counts[1])/20000-.3)<.02 &&
          std::abs(double(counts[2])/20000-.2)<.02,"keyed categorical frequency check");
    // Candidate permutation invariant, including ties and a fixed top-p cut.
    const std::array<U32,4> a{40,10,30,20},b{20,30,10,40};const std::array<float,4> sa{5,5,5,1},sb{1,5,5,5};
    for(U32 pos=0;pos<1000;++pos)check(position_sample_candidates(a,sa,{.seed=4,.position=pos,.top_k=2,.temperature=1,.top_p=.95f})==
      position_sample_candidates(b,sb,{.seed=4,.position=pos,.top_k=2,.temperature=1,.top_p=.95f}),"candidate permutation changed keyed choice");
    std::cout<<Json({{"schema","omlx-m5-p03-tf-position-v1"},{"cases",results},{"uniform_vectors",count},
      {"distribution_draws",20000},{"distribution_counts",counts},{"candidate_permutations",1000},
      {"unqualified_fake_rejected",true},
      {"native_spec_acceptance_qualified",false},{"passed",true}}).dump(2)<<'\n';return 0;
  }catch(const std::exception&e){std::cerr<<"P03-TF: "<<e.what()<<'\n';return 1;}
}
