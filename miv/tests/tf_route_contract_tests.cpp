#include "omlx_m5/qwen_forward.h"
#include "omlx_m5/quantization.h"
#include "../src/forward_plan.h"
#include "../src/route_trace_schema.h"
#include <fstream>
#include <iostream>
#include <numeric>
using namespace omlx::m5;using namespace omlx::m5::detail::forward;using Json=nlohmann::json;
namespace {
void require(bool b,const char*m){if(!b)throw std::runtime_error(m);}
void gpu_router(const std::filesystem::path &libs){
 auto c=MetalContext::create();auto logits=c.make_buffer(8*256*2),ids=c.make_buffer(8*8*4),scores=c.make_buffer(8*8*2);
 Encoder encoder(c,libs,8,"p12.tf.router");auto&d=encoder.emit("qwen_forward","omlx_m5_forward_router_top8_logits",{.width=32,.height=8},{.width=32},{{&logits,0,logits.size()},{&ids,0,ids.size()},{&scores,0,scores.size()}});encoder.dynamic(d,3,words({8,256}),0);auto profile=encoder.finish();
 const auto run=[&](const std::vector<std::uint16_t>&x){logits.write(std::as_bytes(std::span(x)));const std::array calls{MetalCommandInvocation{.plan=&profile.plan,.bindings=profile.bindings,.parameters=&profile.parameters}};(void)c.execute_batch(calls);return std::pair{ids.read(),scores.read()};};
 std::vector<std::uint16_t>x(8*256);
 for(U32 row=0;row<8;++row)for(U32 expert=0;expert<256;++expert)x[row*256+expert]=float_to_bf16(row%2?float(int((expert*17+row*31)%41)-20):0.f);
 auto a=run(x);std::vector<std::uint16_t> reversed(x.size());for(U32 row=0;row<8;++row)std::copy_n(x.begin()+(7-row)*256,256,reversed.begin()+row*256);auto b=run(reversed);
 for(U32 row=0;row<8;++row){require(std::equal(a.first.begin()+row*32,a.first.begin()+(row+1)*32,b.first.begin()+(7-row)*32),"GPU router reorder crossed expert IDs");require(std::equal(a.second.begin()+row*16,a.second.begin()+(row+1)*16,b.second.begin()+(7-row)*16),"GPU router reorder changed BF16 scores");
  if(row%2==0)for(U32 k=0;k<8;++k){U32 id;std::uint16_t score;std::memcpy(&id,a.first.data()+row*32+k*4,4);std::memcpy(&score,a.second.data()+row*16+k*2,2);require(id==255-k && score==float_to_bf16(.125f),"protected router tie rule changed");}
 }
}
}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::invalid_argument("route_contract_tests TRACE LIBRARIES");Json trace;std::ifstream(argv[1])>>trace;detail::validate_route_trace_v2(trace);auto roundtrip=Json::parse(trace.dump());detail::validate_route_trace_v2(roundtrip);U32 rejected=0;
 const auto bad=[&](auto edit){auto copy=trace;edit(copy);bool failed=false;try{detail::validate_route_trace_v2(copy);}catch(const std::invalid_argument&){failed=true;}require(failed,"malformed trace passed v2 validation");++rejected;};
 for(const auto *field:{"artifact","config","compiled_source","kernels","execution","numeric","sampling","chunk","row_layout","router_profile","expert_profile","input","route_ids","route_scores","scope"})bad([&](auto&x){x["provenance"].erase(field);});
 bad([](auto&x){x["provenance"]["sampling"]="bad";});bad([](auto&x){x["provenance"]["chunk"]="bad";});bad([](auto&x){x["provenance_sha256"]="bad";});
 bad([](auto&x){x["expert_ids"][0][1]=x["expert_ids"][0][0];});bad([](auto&x){x["scores_bf16_u16"][0][0]=65536;});
 bad([](auto&x){x["scores_bf16_u16"][0][0]=32704;});bad([](auto&x){x["scores"][0][0]=.25;});
 bad([](auto&x){x["row_positions"][0]=4294967296ULL;});bad([](auto&x){x["row_valid"][0]=false;});bad([](auto&x){x["row_slots"][0]=1;});
 bad([](auto&x){x["tokens"]=-1;});bad([](auto&x){x["tokens"]=1.0;});bad([](auto&x){x["top_k"]=8.0;});
 bad([](auto&x){std::swap(x["expert_ids"][0],x["expert_ids"][1]);x["scores_bf16_u16"][0][0]=0;});
 gpu_router(argv[2]);std::cout<<Json({{"schema","omlx-m5-p12-tf-contract-v1"},{"negative_cases",rejected},{"strict_roundtrip",true},{"gpu_router_ties_reorder",true},{"passed",true}}).dump(2)<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<"P12-TF: "<<e.what()<<'\n';return 1;}}
