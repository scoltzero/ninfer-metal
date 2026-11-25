#include "omlx_m5/affine_lane.h"
#include "omlx_m5/artifact.h"
#include "omlx_m5/quantization.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>

using namespace omlx::m5;
using U32=std::uint32_t;using U64=std::uint64_t;using Json=nlohmann::json;
namespace {
void require(bool b,std::string_view m){if(!b)throw std::runtime_error(std::string(m));}
template<class F>void denied(F f){try{f();}catch(const std::invalid_argument&){return;}throw std::runtime_error("invalid affine lane input admitted");}
struct Source {
  MaterializedWeightView view;
  std::vector<MetalBuffer> buffers;
  std::map<std::string,std::vector<std::byte>> raw;
  void add(const MetalContext &c,std::string kind,std::vector<std::byte> bytes) {
    buffers.push_back(c.upload(bytes));
    view.planes.push_back({.kind=kind,.buffer=&buffers.back(),.bytes=bytes.size(),.row_stride=bytes.size()/view.row_count});
    raw.emplace(std::move(kind),std::move(bytes));
  }
};
Source synthetic(const MetalContext &c,U32 n,U32 k) {
  Source s;s.buffers.reserve(3);
  s.view={.name="tf.affine.synthetic",.format="oq_affine_q4_g64_bf16_v1",.layout="oq_rowsplit_k128_v1",
          .parent_shape={n,k},.padded_k=U64((k+127)/128+1)*128,.row_count=n,.in_features=k};
  const U64 kp=s.view.padded_k,gp=kp/64;
  std::vector<std::byte> code(U64(n)*kp/2);
  std::vector<std::uint16_t> scale(U64(n)*gp),bias(scale.size());
  for(U32 row=0;row<n;++row) {
    for(U32 i=0;i<k;++i)code[U64(row)*kp/2+i/2]|=std::byte(((row*11+i*7)%16)<<((i&1)*4));
    for(U32 g=0;g<gp;++g) {
      const float a=.006f*(1+(row+g)%11);
      scale[U64(row)*gp+g]=float_to_bf16(a);bias[U64(row)*gp+g]=float_to_bf16(-a*7.5f);
    }
  }
  s.add(c,"base_code",std::move(code));
  auto sb=std::as_bytes(std::span(scale)),bb=std::as_bytes(std::span(bias));
  s.add(c,"scale",{sb.begin(),sb.end()});s.add(c,"bias",{bb.begin(),bb.end()});return s;
}
U32 byte(const Source &s,std::string_view key,U64 at){return std::to_integer<unsigned char>(s.raw.at(std::string(key)).at(at));}
float bf(const Source &s,std::string_view key,U64 at){return bf16_to_float(std::uint16_t(byte(s,key,at*2)|(byte(s,key,at*2+1)<<8)));}
float decode(const Source &s,U32 n,U32 k) {
  U32 q=(byte(s,"base_code",U64(n)*s.view.padded_k/2+k/2)>>((k&1)*4))&15;
  const U64 at=U64(n)*(s.view.padded_k/64)+k/64;
  return std::fma(bf(s,"scale",at),float(q),bf(s,"bias",at));
}
std::vector<std::uint16_t> read(const MetalBuffer &b,U64 count){auto bytes=b.read(0,count*2);std::vector<std::uint16_t> out(count);std::memcpy(out.data(),bytes.data(),bytes.size());return out;}
Json check(const MetalContext &c,const Source &s,const std::filesystem::path &libs,bool real,
           AffineLaneExecution mode) {
  AffineLaneWeight prepared(c,s.view);const auto &g=prepared.geometry();
  auto codes=prepared.codes().read(),pairs=prepared.scale_bias().read();
  for(U32 n=0;n<g.padded_n;++n)for(U32 group=0;group<g.groups;++group) {
    const U64 base=(U64(n/32)*g.groups+group)*1024+U64(n%32)*32;
    for(U32 i=0;i<32;++i) {
      const auto expected=n<g.n?s.raw.at("base_code")[U64(n)*s.view.padded_k/2+group*32+i]:std::byte{};
      require(codes[base+i]==expected,"prepared code changed original byte");
    }
    for(U32 i=0;i<4;++i) {
      const auto expected=n<g.n?s.raw.at(i<2?"scale":"bias")[(U64(n)*(s.view.padded_k/64)+group)*2+i%2]:std::byte{};
      require(pairs[(U64(group)*g.padded_n+n)*4+i]==expected,"prepared scale/bias changed original bit");
    }
  }
  std::vector<std::uint16_t> input(U64(128)*g.k);
  for(U32 r=0;r<128;++r)for(U32 k=0;k<g.k;++k)
    input[U64(r)*g.k+k]=float_to_bf16(float(int((r*31+k*17)%257)-128)/256.f*(r%3==0?.001f:r%3==1?1.f:16.f));
  auto x=c.upload(std::as_bytes(std::span(input))),y=c.make_buffer(U64(128)*g.n*2);
  AffineLaneProjection p(c,prepared,x,y,libs,mode);
  if(!affine_lane_m5_diagnostic_eligible(c))denied([&]{AffineLaneProjection bad(c,prepared,x,y,libs,AffineLaneExecution::M5TensorDiagnostic);});
  const auto before=c.registry_stats();
  auto run=[&](U32 rows,bool icb=true){
    std::vector<std::uint16_t> sentinel(U64(128)*g.n,0xa5a5);y.write(std::as_bytes(std::span(sentinel)));
    auto submission=p.run(rows,icb);require(submission.dispatch_count==2,"affine lane dispatch count");
    auto all=read(y,U64(128)*g.n);
    require(std::all_of(all.begin()+U64(rows)*g.n,all.end(),[](auto b){return b==0xa5a5;}),"invalid row/tail overwrite");
    all.resize(U64(rows)*g.n);
    require(std::all_of(all.begin(),all.end(),[](auto b){return std::isfinite(bf16_to_float(b));}),"nonfinite affine output");
    return all;
  };
  const auto full=run(128);
  for(U32 m:{1U,2U,4U,7U,8U,9U,16U,17U,31U,32U,33U,48U,64U,65U,127U,128U}) {
    auto part=run(m);require(std::equal(part.begin(),part.end(),full.begin()),"affine row width changed bits");
  }
  require(run(128,false)==full,"affine ICB/direct mismatch");
  for(U32 row=0;row<128;++row) {
    x.write(std::as_bytes(std::span(input).subspan(U64(row)*g.k,g.k)));
    auto one=run(1);require(std::equal(one.begin(),one.end(),full.begin()+U64(row)*g.n),"affine solo/window mismatch");
  }
  std::vector<std::uint16_t> shuffled(input.size());
  for(U32 r=0;r<128;++r)std::copy_n(input.begin()+U64((r*37)%128)*g.k,g.k,shuffled.begin()+U64(r)*g.k);
  x.write(std::as_bytes(std::span(shuffled)));auto reordered=run(128);
  for(U32 r=0;r<128;++r)require(std::equal(reordered.begin()+U64(r)*g.n,reordered.begin()+U64(r+1)*g.n,
    full.begin()+U64((r*37)%128)*g.n),"affine reorder changed bits");
  // Independent mathematical dot reference, all N for four amplitude/row cases.
  double error=0,power=0,max_abs=0;
  for(U32 r:{0U,1U,2U,127U})for(U32 n=0;n<g.n;++n) {
    double expected=0;
    for(U32 k=0;k<g.k;++k)expected+=double(bf16_to_float(input[U64(r)*g.k+k]))*decode(s,n,k);
    const double delta=bf16_to_float(full[U64(r)*g.n+n])-expected;
    error+=delta*delta;power+=expected*expected;max_abs=std::max(max_abs,std::abs(delta));
  }
  const double relative=std::sqrt(error/std::max(power,1e-30));require(relative<=.01,"affine FP64 storage-math reference error");
  std::vector<std::uint16_t> eye(input.size());
  const std::array<U32,14> columns{0,1,2,3,4,5,6,7,31,32,63,std::min(64U,g.k-1),std::min(127U,g.k-1),g.k-1};
  for(U32 r=0;r<columns.size();++r)eye[U64(r)*g.k+columns[r]]=float_to_bf16(1);
  x.write(std::as_bytes(std::span(eye)));auto values=run(U32(columns.size()));
  for(U32 r=0;r<columns.size();++r)for(U32 n=0;n<g.n;++n) {
    float expected=decode(s,n,columns[r]);require(values[U64(r)*g.n+n]==float_to_bf16(expected==0.f?0.f:expected),"affine one-hot value changed");
  }
  const auto after=c.registry_stats();
  require(before.pipelines_compiled==after.pipelines_compiled && before.command_plans_built==after.command_plans_built &&
    before.current_buffer_bytes==after.current_buffer_bytes && before.upload_count==after.upload_count &&
    before.icb_profile_recordings==after.icb_profile_recordings,"affine run allocated/repacked/compiled/recorded");
  denied([&]{(void)p.run(0);});denied([&]{(void)p.run(129);});
  return {{"object",s.view.name},{"n",g.n},{"k",g.k},{"actual_executed_n",g.n},{"real_weight",real},
          {"split_k",g.split_k},{"padded_n",g.padded_n},{"numeric_contract_id",p.numeric_contract_id()},
          {"prepared_code_bytes",g.code_bytes},{"prepared_scale_bias_bytes",g.scale_bias_bytes},
          {"initialization_readback_bytes",prepared.preparation_readback_bytes()},
          {"row_widths",{1,2,4,7,8,9,16,17,31,32,33,48,64,65,127,128}},
          {"solo_rows",128},{"one_hot_values",14U*U64(g.n)},
          {"fp64_reference_full_n_rows",4},{"relative_l2",relative},{"max_abs",max_abs},
          {"layout_bits_preserved",true},{"row_reorder_equal",true},{"icb_direct_equal",true},
          {"no_hot_repack_allocation_compile",true},{"m5_guard_checked",!affine_lane_m5_diagnostic_eligible(c)},
          {"m5_tensor_candidate_kernel_executed",mode==AffineLaneExecution::M5TensorDiagnostic},
          {"nax_physical_execution_verified",false},{"passed",true}};
}
}
int main(int argc,char **argv) {
  try {
    const bool tensor=argc>2 && std::string_view(argv[argc-1])=="--tensor";
    const int arguments=argc-int(tensor);
    if(arguments!=2 && arguments!=4)throw std::invalid_argument("tf_affine_lane_tests LIBRARIES [ARTIFACT OBJECT] [--tensor]");
    const auto mode=tensor?AffineLaneExecution::M5TensorDiagnostic:AffineLaneExecution::PortableReference;
    auto c=MetalContext::create();Json results=Json::array();
    if(arguments==4) {
      auto a=ArtifactModel::open(argv[2],false);const auto &o=a.object(argv[3]);
      Source s;s.buffers.reserve(o.planes.size());
      s.view={.object_id=o.id,.name=o.name,.format=o.format,.layout=o.layout,.parent_shape=o.shape,
              .padded_k=o.padded_k,.row_count=o.shape[0],.in_features=o.shape[1]};
      for(const auto &p:o.planes){auto bytes=a.plane_bytes(o,p);require(artifact_sha256_hex(bytes)==p.sha256,"real source hash mismatch");s.add(c,p.kind,{bytes.begin(),bytes.end()});}
      results.push_back(check(c,s,argv[1],true,mode));
    } else {
      for(const auto [n,k]:{std::pair<U32,U32>{17,64},{48,1024},{65,2048},{128,4096}}) {
        auto s=synthetic(c,n,k);results.push_back(check(c,s,argv[1],false,mode));
        auto wrong=s.view;wrong.format="mxfp4_e2m1_g32_e8m0_v1";denied([&]{AffineLaneWeight p(c,wrong);});
        wrong=s.view;wrong.format="oq_affine_q5_g64_bf16_v1";denied([&]{AffineLaneWeight p(c,wrong);});
        wrong=s.view;wrong.format="oq_affine_q4_g128_bf16_v1";denied([&]{AffineLaneWeight p(c,wrong);});
        wrong=s.view;wrong.planes[0].row_stride++;denied([&]{AffineLaneWeight p(c,wrong);});
        wrong=s.view;wrong.planes[2]=wrong.planes[1];denied([&]{AffineLaneWeight p(c,wrong);});
      }
      AffineLaneGeometry a{.n=17,.k=4096,.split_k=8},b{.n=32,.k=4096,.split_k=8};
      require(affine_lane_fusion_compatible(a,b),"equal split fusion refused");b.split_k=4;
      require(!affine_lane_fusion_compatible(a,b),"different split fusion accepted");
      // Equal members may have a different default split after concatenating N.
      // Carry their original split explicitly; recomputing it would change bits.
      auto parent=synthetic(c,8192,4096);
      auto left=parent.view,right=parent.view;
      left.row_count=right.row_count=4096;left.parent_shape=right.parent_shape={4096,4096};
      for(auto &plane:left.planes)plane.bytes=4096*plane.row_stride;
      for(auto &plane:right.planes){plane.offset+=4096*plane.row_stride;plane.bytes=4096*plane.row_stride;}
      AffineLaneWeight lp(c,left),rp(c,right),fused(c,parent.view,lp.geometry().split_k);
      require(affine_lane_fusion_compatible(lp.geometry(),rp.geometry()) &&
              affine_lane_split_k(8192,4096)!=lp.geometry().split_k && fused.geometry().explicit_split_k,
              "fusion test did not preserve a changed-N split seam");
      std::vector<std::uint16_t> input(4096,float_to_bf16(.125f));
      auto x=c.upload(std::as_bytes(std::span(input))),y=c.make_buffer(8192*2);
      auto yl=c.make_buffer(4096*2),yr=c.make_buffer(4096*2);
      AffineLaneProjection pp(c,fused,x,y,argv[1],mode,1),pl(c,lp,x,yl,argv[1],mode,1),pr(c,rp,x,yr,argv[1],mode,1);
      (void)pp.run(1);(void)pl.run(1);(void)pr.run(1);
      const auto whole=read(y,8192),al=read(yl,4096),ar=read(yr,4096);
      require(std::equal(al.begin(),al.end(),whole.begin()) && std::equal(ar.begin(),ar.end(),whole.begin()+4096),
              "preserved-split fused/separate arithmetic changed");
      denied([&]{AffineLaneWeight bad(c,parent.view,3);});
    }
    std::cout<<Json({{"schema","omlx-m5-p11-tf-affine-v1"},{"device",c.capabilities().device_name},
      {"m5_diagnostic_eligible",affine_lane_m5_diagnostic_eligible(c)},
      {"scope","prepared representation + portable GPU reference; M5 tensor fragment requires H0"},
      {"m5_nax_qualified",false},{"fusion_split_seam_checked",arguments==2},
      {"cases",results},{"passed",true}}).dump(2)<<'\n';return 0;
  } catch(const std::exception &e){std::cerr<<"P11-TF: "<<e.what()<<'\n';return 1;}
}
