#include "omlx_m5/target_verify_gpu.h"
#include "omlx_m5/position_sampling.h"
#include "omlx_m5/quantization.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <nlohmann/json.hpp>

using namespace omlx::m5;using U32=std::uint32_t;using U64=std::uint64_t;using Json=nlohmann::json;
namespace {
void require(bool b,std::string_view m){if(!b)throw std::runtime_error(std::string(m));}
Json run(const std::filesystem::path &libs,U32 vocab) {
  auto ctx=MetalContext::create();std::vector<std::uint16_t> x(U64(8)*vocab);std::vector<U32> ids(vocab);std::iota(ids.begin(),ids.end(),0);
  auto buffer=ctx.make_buffer(x.size()*2);TargetGpuVerifier verifier(ctx,buffer,vocab,libs,SamplingContract::PositionCoupledV1);
  U32 cases=0,fallbacks=0;U64 max_read=0;
  for(bool tied:{false,true}) {
    for(U32 r=0;r<8;++r)for(U32 i=0;i<vocab;++i)x[U64(r)*vocab+i]=float_to_bf16(tied?0.f:float(int((i*17+r*31)%73)-36)/8);
    buffer.write(std::as_bytes(std::span(x)));
    for(U32 k=0;k<=7;++k)for(U32 accepted=0;accepted<=k;++accepted)for(float temperature:{0.f,.7f})for(float top_p:{.5f,1.f}) {
      const U64 seed=73+k*31+accepted,position=(1ULL<<35)+19;
      GpuSamplingOptions opts{.temperature=temperature,.top_k=8,.top_p=top_p,.seed=seed,.contract=SamplingContract::PositionCoupledV1};
      std::vector<U32> target(k+1);std::vector<float> scores(vocab);
      for(U32 r=0;r<=k;++r) {
        for(U32 i=0;i<vocab;++i)scores[i]=bf16_to_float(x[U64(r)*vocab+i]);
        target[r]=position_sample_candidates(ids,scores,{.seed=seed,.position=position+r+1,.top_k=8,.temperature=temperature,.top_p=top_p});
      }
      std::vector<SelectorStep> proposals(k);
      for(U32 r=0;r<k;++r)proposals[r].token=std::int32_t(r<accepted?target[r]:(target[r]+1)%vocab);
      std::vector<std::int32_t> expected;
      for(U32 r=0;r<accepted;++r)expected.push_back(target[r]);expected.push_back(target[accepted]);
      for(bool icb:{false,true}) {
        auto actual=verifier.select(proposals,opts,position,icb);
        require(actual.verification.licensed_outputs==expected,"coupled equality/correction/bonus token differs");
        require(actual.verification.accepted_proposals==accepted && actual.verification.used_correction==(accepted<k),"coupled accepted-prefix metadata differs");
        max_read=std::max(max_read,actual.host_read_bytes);fallbacks+=actual.compact_cpu_fallback;
      }
      ++cases;
    }
  }
  return {{"vocab",vocab},{"cases",cases},{"compact_fallbacks",fallbacks},{"max_host_read_bytes",max_read},{"d0_7_all_prefixes",true},{"passed",true}};
}
}
int main(int argc,char **argv) {
  try{if(argc!=2)throw std::invalid_argument("tf_target_equality_tests LIBRARIES");Json tests=Json::array();
    tests.push_back(run(argv[1],257));tests.push_back(run(argv[1],248320));
    std::cout<<Json({{"schema","omlx-m5-p04-tf-target-equality-v1"},{"cases",tests},{"passed",true}}).dump(2)<<'\n';return 0;
  }catch(const std::exception&e){std::cerr<<"P04-TF: "<<e.what()<<'\n';return 1;}
}
