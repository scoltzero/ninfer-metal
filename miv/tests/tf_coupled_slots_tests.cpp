#include "omlx_m5/engine.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <thread>
#include <nlohmann/json.hpp>
using namespace omlx::m5;using Json=nlohmann::json;using U32=std::uint32_t;using namespace std::chrono_literals;
namespace {
bool grouped_mode=false,projection_mode=false;
void require(bool b,std::string_view m){if(!b)throw std::runtime_error(std::string(m));}
struct Result{std::vector<std::int32_t> tokens;EventKind terminal=EventKind::tokens;};
std::map<std::uint64_t,Result> collect(Engine&e,const std::vector<std::uint64_t>&ids){
 std::map<std::uint64_t,Result> out;const std::set<std::uint64_t> expected(ids.begin(),ids.end());auto deadline=std::chrono::steady_clock::now()+60s;
 while(std::chrono::steady_clock::now()<deadline){for(const auto&ev:e.poll_events(10ms)){
  require(expected.contains(ev.request_id),"foreign slot event");auto&r=out[ev.request_id];
  if(ev.kind==EventKind::tokens){require(r.terminal==EventKind::tokens,"token after terminal");r.tokens.insert(r.tokens.end(),ev.tokens.begin(),ev.tokens.end());}
  else{require(ev.kind!=EventKind::failed,ev.error);require(r.terminal==EventKind::tokens,"duplicate slot terminal");r.terminal=ev.kind;}}
  if(out.size()==ids.size() && std::all_of(out.begin(),out.end(),[](const auto&p){return p.second.terminal!=EventKind::tokens;}))return out;
 }
 throw std::runtime_error("slot fairness/termination timeout");
}
LoadOptions options(const std::filesystem::path&fixtures,const std::filesystem::path&draft,const std::filesystem::path&libs,const char*name,U32 b,U32 d,bool cache){
 LoadOptions o;o.model_dir=(fixtures/name/"source").string();o.max_context=2048;o.max_concurrency=b;o.sampling_contract="position_coupled_v1";
 o.path_overrides={{"artifact",(fixtures/name/"model.omlxm5").string()},{"kernel_directory",libs.string()}};
 if(b>1 && grouped_mode)o.path_overrides["batch_mode"]="on";
 if(b>1 && projection_mode)o.path_overrides["batch_projection_prototype"]="on";
 if(cache)o.path_overrides["prefix_cache_bytes"]="67108864";
 if(d){o.draft_dir=(draft/"source").string();o.path_overrides["draft_artifact"]=(draft/"model.omlxm5").string();o.speculative={.backend=SpeculativeBackend::dflash2,.draft_tokens=d};}
 return o;
}
Json check(const std::filesystem::path&fixtures,const std::filesystem::path&draft,const std::filesystem::path&libs,const char*name,U32 b,U32 d,float temperature){
 auto solo=Engine::create(options(fixtures,draft,libs,name,1,d,false)),slots=Engine::create(options(fixtures,draft,libs,name,b,d,true));
 std::vector<SubmitRequest> requests(b);std::vector<std::vector<std::int32_t>> expected(b);U32 comparisons=0;
 for(U32 i=0;i<b;++i){auto&r=requests[i];for(U32 row=0;row<33+i*7;++row)r.prompt_tokens.push_back((row*71+256+i)%257);
  r.max_new_tokens=12+i%4;r.sampling={.temperature=temperature,.top_p=.95f,.top_k=20,.seed=i%2?UINT64_MAX:(1ULL<<40)+i};
  auto id=solo->submit(r);expected[i]=collect(*solo,{id}).at(id).tokens;
 }
 for(U32 wave=0;wave<3;++wave){std::vector<std::uint64_t> ids(b),order;
  for(U32 j=0;j<b;++j){const U32 i=wave%2?b-1-j:j;ids[i]=slots->submit(requests[i]);order.push_back(ids[i]);}
  auto out=collect(*slots,order);for(U32 i=0;i<b;++i){require(out.at(ids[i]).terminal==EventKind::finished && out.at(ids[i]).tokens==expected[i],"slot/arrival/reuse changed same-contract output");++comparisons;}
 }
 const auto normal=slots->stats();
 if(d)require(normal.spec_target_verify_calls==normal.spec_rounds && normal.spec_replay_calls==normal.spec_rounds,"normal slot round omitted verify/Replay");
 const bool checked_cancel=temperature>0 && (b==2 || d==0 || d==7);
 if(checked_cancel){
 // Cancel an active middle slot, then compare the other actual streams.
 std::vector<std::uint64_t> ids;for(auto&r:requests){r.max_new_tokens=64;ids.push_back(slots->submit(r));}
 auto deadline=std::chrono::steady_clock::now()+3s;
 while(slots->capacity().active_requests<int(b) && std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(50us);
 require(slots->capacity().active_requests==int(b),"active slots did not admit concurrently");const U32 victim=b/2;
 require(slots->cancel(ids[victim]),"active slot cancellation refused");auto cancelled=collect(*slots,ids);
 require(cancelled.at(ids[victim]).terminal==EventKind::aborted,"cancelled slot not aborted");
 for(U32 i=0;i<b;++i)if(i!=victim){auto id=solo->submit(requests[i]);auto baseline=collect(*solo,{id}).at(id);
  require(cancelled.at(ids[i]).tokens==baseline.tokens && cancelled.at(ids[i]).terminal==EventKind::finished,"cancel crossed another slot state");++comparisons;}
 for(U32 i=0;i<b;++i){requests[i].max_new_tokens=12+i%4;auto id=slots->submit(requests[i]);require(collect(*slots,{id}).at(id).tokens==expected[i],"cancelled slot reuse polluted output");++comparisons;}
 }
 const auto stats=slots->stats();
 if(grouped_mode){require(stats.batch_submissions>0,"P10 grouped forward did not execute");
  if(!d)require(stats.batch_sampling_host_read_bytes>0,"coupled cross-row sampler did not execute");
  if(projection_mode){
    require(stats.shared_head_multirow_calls>0 && stats.shared_head_projection_rows>=stats.shared_head_projection_calls,
      "actual shared head multi-row prototype did not execute");
  }
 } else require(stats.batch_submissions==0,"P09 silently enabled grouped mode");
 if(d)require(stats.spec_rounds && stats.spec_target_verify_calls==stats.spec_rounds && stats.spec_replay_calls<=stats.spec_rounds,"slot lost verify/Replay invariant");
 slots->shutdown();solo->shutdown();return {{"model",name},{"b",b},{"d",d},{"temperature",temperature},{"request_comparisons",comparisons},{"arrival_reorder",true},{"middle_cancel_reuse",checked_cancel},{"passed",true}};
}
}
int main(int argc,char**argv){try{
 if(argc!=4 && argc!=5)throw std::invalid_argument("coupled_slots_tests FIXTURES DRAFT LIBRARIES [--batch|--batch-prototype]");
 if(argc==5){const std::string_view mode(argv[4]);if(mode!="--batch" && mode!="--batch-prototype")throw std::invalid_argument("unknown slot test mode");grouped_mode=true;projection_mode=mode=="--batch-prototype";}
 Json records=Json::array();
 for(const char*name:{"affine","mxfp4","moe-affine","moe-mxfp4"}){
  for(U32 b:{2U,4U,8U})for(U32 d=0;d<=7;++d)for(float t:{0.f,.7f})records.push_back(check(argv[1],argv[2],argv[3],name,b,d,t));
  std::cerr<<"P09 mode passed: "<<name<<'\n';
 }
 for(U32 b:{3U,5U,6U,7U})for(U32 d:{0U,7U})records.push_back(check(argv[1],argv[2],argv[3],"affine",b,d,.7f));
 std::cout<<Json({{"schema","omlx-m5-coupled-slots-v2"},{"grouped_mode",grouped_mode},{"head_projection_prototype",projection_mode},{"records",records},{"passed",true}}).dump(2)<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<"P09-TF: "<<e.what()<<'\n';return 1;}}
