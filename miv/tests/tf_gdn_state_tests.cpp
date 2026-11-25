#include "omlx_m5/qwen_forward.h"
#include "omlx_m5/artifact.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace omlx::m5;
using U32=std::uint32_t;using Json=nlohmann::json;
namespace {
void require(bool b,std::string_view m){if(!b)throw std::runtime_error(std::string(m));}
std::vector<U32> tokens(U32 count,U32 offset=0){std::vector<U32> t(count);for(U32 i=0;i<count;++i)t[i]=(71*(i+offset)+256)%257;return t;}
void finite_state(const std::vector<std::byte> &bytes,const QwenModelConfig &c) {
  const std::uint64_t state=std::uint64_t(c.linear_num_value_heads)*c.linear_value_head_dim*c.linear_key_head_dim;
  const std::uint64_t conv=(2ULL*c.linear_num_key_heads*c.linear_key_head_dim+
    std::uint64_t(c.linear_num_value_heads)*c.linear_value_head_dim)*(c.linear_conv_kernel_dim-1);
  std::uint64_t offset=0;
  for(int layer=0;layer<c.linear_attention_layers();++layer) {
    for(std::uint64_t i=0;i<state;++i){float x;std::memcpy(&x,bytes.data()+offset+i*4,4);require(std::isfinite(x),"nonfinite FP32 recurrent state");}
    offset+=state*4;
    for(std::uint64_t i=0;i<conv;++i){std::uint16_t bits;std::memcpy(&bits,bytes.data()+offset+i*2,2);require(std::isfinite(std::bit_cast<float>(std::uint32_t(bits)<<16)),"nonfinite conv history");}
    offset+=conv*2;
  }
  require(offset==bytes.size(),"state inspection geometry mismatch");
}
void equal_row(const QwenForwardResult &batch,const QwenForwardResult &solo,U32 row,U32 h,U32 v) {
  require(batch.hidden.size()>=(row+1)*h && solo.hidden.size()==h,"state test hidden geometry");
  require(std::equal(solo.hidden.begin(),solo.hidden.end(),batch.hidden.begin()+row*h),"GDN chain solo/window hidden bit mismatch");
  require(std::equal(solo.logits.begin(),solo.logits.end(),batch.logits.begin()+row*v),"GDN chain solo/window logits bit mismatch");
}
Json mode(const std::filesystem::path &fixtures,const std::filesystem::path &libs,const char *name) {
  const auto root=fixtures/name;auto cfg=load_qwen_model_config(root/"source/config.json");
  auto artifact=ArtifactModel::open(root/"model.omlxm5",true);
  auto ca=MetalContext::create(),cb=MetalContext::create();
  auto wa=materialize_model(artifact,plan_model_materialization(artifact,ca),ca);
  auto wb=materialize_model(artifact,plan_model_materialization(artifact,cb),cb);
  auto layers=build_qwen_layer_catalog(artifact,cfg,2048,64);const std::array<U32,2> taps{0,1};
  QwenForward batch(ca,wa,layers,cfg,2048,libs,taps,false,true,false,{},false,false,{},
    LinearNumericPolicy::RowExactPortableV1,SamplingContract::LegacyRejectionV1,AttentionNumericPolicy::RowExactV1);
  QwenForward solo(cb,wb,layers,cfg,2048,libs,taps,false,true,false,{},false,false,{},
    LinearNumericPolicy::RowExactPortableV1,SamplingContract::LegacyRejectionV1,AttentionNumericPolicy::RowExactV1);
  require(batch.inspect_profile(8).gdn_row_exact_eligible && batch.inspect_profile(8,true).gdn_row_exact_eligible &&
    batch.inspect_profile(8).gdn_numeric_contract_id==batch.inspect_profile(1).gdn_numeric_contract_id,
    "actual installed GDN contract missing across run/verify widths");
  U32 width_cases=0,prefix_cases=0,abort_cases=0,long_steps=0;
  for(U32 history:{0U,1U,3U,63U,64U,65U,127U,129U})for(U32 width=1;width<=8;++width) {
    const auto prompt=tokens(history),input=tokens(width,history);
    batch.reset();solo.reset();if(history){(void)batch.run(prompt);(void)solo.run(prompt);}
    const auto full=batch.run(input,{},true);
    for(U32 row=0;row<width;++row){const auto step=solo.run(std::span(input).subspan(row,1),{},true);equal_row(full,step,row,cfg.hidden_size,cfg.vocab_size);}
    require(batch.inspect_recurrent_state()==solo.inspect_recurrent_state(),"GDN/conv final state differs across W");++width_cases;
    finite_state(batch.inspect_recurrent_state(),cfg);
    for(U32 accepted=0;accepted<=width;++accepted) {
      batch.reset();solo.reset();if(history){(void)batch.run(prompt);(void)solo.run(prompt);}
      const auto committed=batch.inspect_recurrent_state();const auto verified=batch.verify(input,{},true);
      require(batch.inspect_recurrent_state()==committed && batch.position()==history,"verify modified committed GDN/conv");
      (void)batch.commit_prefix(accepted);
      for(U32 r=0;r<accepted;++r){auto step=solo.run(std::span(input).subspan(r,1),{},true);equal_row(verified,step,r,cfg.hidden_size,cfg.vocab_size);}
      require(batch.position()==solo.position() && batch.inspect_recurrent_state()==solo.inspect_recurrent_state(),"accepted-prefix Replay FP32/conv bit mismatch");
      const auto next=tokens(1,history+accepted);const auto a=batch.run(next,{},true),b=solo.run(next,{},true);
      require(a.logits==b.logits && a.hidden==b.hidden && batch.inspect_recurrent_state()==solo.inspect_recurrent_state(),"Replay continuation differs");++prefix_cases;
    }
    batch.reset();if(history)(void)batch.run(prompt);const auto before=batch.inspect_recurrent_state();
    (void)batch.verify(input);batch.abort_verify();require(before==batch.inspect_recurrent_state()&&batch.position()==history,"abort polluted GDN/conv");++abort_cases;
  }
  batch.reset();solo.reset();(void)batch.run(tokens(33));(void)solo.run(tokens(33));
  for(U32 round=0;round<128;++round) {
    const U32 width=round%8+1,accepted=round%(width+1);auto input=tokens(width,batch.position());
    (void)batch.verify(input);(void)batch.commit_prefix(accepted);
    for(U32 r=0;r<accepted;++r)(void)solo.run(std::span(input).subspan(r,1));
    const auto next=tokens(1,batch.position());auto a=batch.run(next,{},true),b=solo.run(next,{},true);
    require(a.logits==b.logits && batch.inspect_recurrent_state()==solo.inspect_recurrent_state(),"long Replay state drift");++long_steps;
    finite_state(batch.inspect_recurrent_state(),cfg);
  }
  return {{"model",name},{"width_cases",width_cases},{"accepted_prefix_cases",prefix_cases},
    {"abort_cases",abort_cases},{"long_replay_rounds",long_steps},{"final_position",batch.position()},
    {"state_bytes",batch.inspect_recurrent_state().size()},{"whole_model_scope","small control target"},{"passed",true}};
}
}
int main(int argc,char **argv) {
  try{if(argc!=3)throw std::invalid_argument("tf_gdn_state_tests FIXTURES LIBRARIES");Json modes=Json::array();
    for(const char *name:{"affine","mxfp4","moe-affine","moe-mxfp4"})modes.push_back(mode(argv[1],argv[2],name));
    std::cout<<Json({{"schema","omlx-m5-p07-tf-state-v1"},{"modes",modes},{"passed",true},
      {"m5_device_qualified",false},{"full_target_model_claimed",false}}).dump(2)<<'\n';return 0;
  }catch(const std::exception&e){std::cerr<<"P07-TF: "<<e.what()<<'\n';return 1;}
}
