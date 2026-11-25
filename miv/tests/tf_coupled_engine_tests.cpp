#include "omlx_m5/engine.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
using namespace omlx::m5;using Json=nlohmann::json;using namespace std::chrono_literals;
namespace {
void require(bool b,std::string_view m){if(!b)throw std::runtime_error(std::string(m));}
struct Result {std::vector<std::int32_t> tokens;EventKind terminal=EventKind::failed;std::string reason;};
Result run(Engine &e,const SubmitRequest &r,std::size_t cancel_after=SIZE_MAX,bool allow_failure=false) {
  const auto before=e.stats();auto id=e.submit(r);Result out;bool cancelled=false;const auto end=std::chrono::steady_clock::now()+60s;
  while(std::chrono::steady_clock::now()<end)for(auto &ev:e.poll_events(10ms)) {
    require(ev.request_id==id,"coupled engine foreign event");
    if(ev.kind==EventKind::tokens){out.tokens.insert(out.tokens.end(),ev.tokens.begin(),ev.tokens.end());
      if(!cancelled&&out.tokens.size()>=cancel_after){(void)e.cancel(id);cancelled=true;}}
    else {require(allow_failure || ev.kind!=EventKind::failed,ev.error);out.terminal=ev.kind;out.reason=ev.finish_reason;
      const auto after=e.stats();const auto rounds=after.spec_rounds-before.spec_rounds;
      const auto verifies=after.spec_target_verify_calls-before.spec_target_verify_calls;
      const auto replays=after.spec_replay_calls-before.spec_replay_calls;
      require(verifies==rounds,"coupled round did not have exactly one target verify");
      require(replays<=rounds,"coupled round replayed more than once");
      if(!cancelled)require(replays==rounds,"committed coupled round omitted Replay");
      else require(rounds-replays<=1,"cancellation left more than one uncommitted round");
      return out;}
  }
  throw std::runtime_error("coupled engine timeout");
}
LoadOptions options(const std::filesystem::path &target,const std::filesystem::path &draft,
                    const std::filesystem::path &libs,std::uint32_t d) {
  LoadOptions o;o.model_dir=(target/"affine/source").string();o.max_context=2048;o.sampling_contract="position_coupled_v1";
  o.path_overrides={{"artifact",(target/"affine/model.omlxm5").string()},{"kernel_directory",libs.string()}};
  if(d){o.draft_dir=(draft/"source").string();o.path_overrides["draft_artifact"]=(draft/"model.omlxm5").string();
    o.speculative={.backend=SpeculativeBackend::dflash2,.draft_tokens=d};}
  return o;
}
}
int main(int argc,char **argv) {
  try{if(argc!=4)throw std::invalid_argument("tf_coupled_engine_tests TARGET DRAFT LIBRARIES");
    auto ar=Engine::create(options(argv[1],argv[2],argv[3],0));Json records=Json::array();
    for(std::uint32_t d=1;d<=7;++d) {
      auto spec=Engine::create(options(argv[1],argv[2],argv[3],d));
      for(std::uint64_t seed:std::array<std::uint64_t,3>{0,123,UINT64_MAX})for(float temp:{0.f,.7f}) {
        SubmitRequest r;for(int i=0;i<65;++i)r.prompt_tokens.push_back((71*i+256)%257);
        r.max_new_tokens=24;r.sampling={.temperature=temp,.top_p=.95f,.top_k=20,.seed=seed};
        const auto baseline=run(*ar,r),actual=run(*spec,r);require(actual.tokens==baseline.tokens,"sampled AR/spec token sequence differs");
        require(run(*spec,r).tokens==baseline.tokens,"second coupled request changed output");
        for(int budget:{1,2,3,4,5,6,7,8,9}){r.max_new_tokens=budget;auto bounded=run(*spec,r);require(bounded.tokens==std::vector<std::int32_t>(baseline.tokens.begin(),baseline.tokens.begin()+budget),"coupled D tail changed output");}
        r.max_new_tokens=24;r.stop_token_seqs={{baseline.tokens.at(5),baseline.tokens.at(6)}};
        require(run(*spec,r).tokens==run(*ar,r).tokens,"coupled stop sequence differs");
        r.stop_token_seqs.clear();r.max_new_tokens=100;auto cancelled=run(*spec,r,3);
        require(cancelled.terminal==EventKind::aborted || cancelled.terminal==EventKind::finished,"coupled cancel terminal invalid");
        r.max_new_tokens=24;require(run(*spec,r).tokens==baseline.tokens,"cancel polluted next coupled request");
        auto invalid=r;invalid.prompt_tokens.push_back(257);
        require(run(*spec,invalid,SIZE_MAX,true).terminal==EventKind::failed,"invalid prompt did not produce failure");
        require(run(*spec,r).tokens==baseline.tokens,"failed request polluted next coupled request");
        records.push_back({{"d",d},{"temperature",temp},{"seed",seed},{"tokens",baseline.tokens},{"passed",true}});
      }
      auto s=spec->stats();require(s.spec_rounds>0 && s.spec_target_verify_calls==s.spec_rounds && s.spec_replay_calls<=s.spec_rounds,"coupled spec repeated verify or Replay");
      spec->shutdown();
    }
    ar->shutdown();std::cout<<Json({{"schema","omlx-m5-p04-tf-engine-v1"},{"cases",records},{"terminal_budget_stop_cancel_checked",true},{"error_reset_checked",true},{"passed",true}}).dump(2)<<'\n';return 0;
  }catch(const std::exception&e){std::cerr<<"P04-TF Engine: "<<e.what()<<'\n';return 1;}
}
