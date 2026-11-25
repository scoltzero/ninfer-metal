#include "omlx_m5/affine_lane.h"
#include "omlx_m5/quantization.h"
#include "forward_plan.h"
#include <atomic>

namespace omlx::m5 {
using namespace detail::forward;
std::uint32_t affine_lane_split_k(U32 n,U32 k) {
  if(!n||!k||k%64)throw std::invalid_argument("affine lane requires N>0 and K64");
  U32 sk=1;
  const U64 tiles=(U64(n)+31)/32;
  while(sk<8 && tiles*sk<1024 && (k/64)/(sk*2)>=8)sk*=2;
  return sk;
}
bool affine_lane_fusion_compatible(const AffineLaneGeometry &a,const AffineLaneGeometry &b) {
  return a.n && b.n && a.k==b.k && a.split_k==b.split_k && a.layout_id==b.layout_id;
}
bool affine_lane_m5_diagnostic_eligible(const MetalContext &ctx) {
  const auto &c=ctx.capabilities();const auto &p=c.profile.probe;
  return (c.device_name.starts_with("Apple M5") || p.apple_gpu_family>=10) &&
         p.runtime_os_major>=26 && p.metal4_compiler && p.tensor_api;
}
struct AffineLaneWeight::Impl {
  AffineLaneGeometry geometry;
  MetalBuffer codes,sb;
  U64 read_bytes=0;
  Impl(const MetalContext &ctx,const MaterializedWeightView &w,std::optional<U32> fixed_split) {
    if(w.format!="oq_affine_q4_g64_bf16_v1" || w.layout!="oq_rowsplit_k128_v1" ||
       w.parent_shape.size()!=2 || w.row_start || w.row_count!=w.parent_shape[0] ||
       !w.row_count || w.row_count>INT32_MAX-31 || w.in_features!=w.parent_shape[1] ||
       !w.in_features || w.in_features>INT32_MAX || w.in_features%64 ||
       w.padded_k<w.in_features || w.padded_k%128 || w.planes.size()!=3)
      throw std::invalid_argument("affine lane accepts a complete Q4/G64 BF16-scale parent only; MXFP4/high-bit/G128 are separate routes");
    geometry.n=U32(w.row_count);geometry.k=U32(w.in_features);
    geometry.padded_n=(geometry.n+31)/32*32;geometry.groups=geometry.k/64;
    geometry.split_k=fixed_split.value_or(affine_lane_split_k(geometry.n,geometry.k));
    geometry.explicit_split_k=fixed_split.has_value();
    if((geometry.split_k!=1 && geometry.split_k!=2 && geometry.split_k!=4 && geometry.split_k!=8) ||
       geometry.split_k>geometry.groups)
      throw std::invalid_argument("affine lane fixed split must be 1/2/4/8 and fit the K groups");
    geometry.code_bytes=mul(geometry.padded_n,geometry.k/2);
    geometry.scale_bias_bytes=mul(mul(geometry.padded_n,geometry.groups),4);
    if(geometry.code_bytes>ctx.capabilities().max_buffer_length ||
       geometry.scale_bias_bytes>ctx.capabilities().max_buffer_length)
      throw std::invalid_argument("affine lane prepared buffer exceeds device limit");
    auto load=[&](std::string_view kind,U64 stride) {
      const MaterializedPlaneView *found=nullptr;
      for(const auto &p:w.planes)if(p.kind==kind) {
        if(found)throw std::invalid_argument("duplicate affine lane plane");found=&p;
      }
      if(!found || !found->buffer || found->row_stride!=stride ||
         found->bytes!=mul(geometry.n,stride) || found->offset>found->buffer->size() ||
         found->bytes>found->buffer->size()-found->offset)
        throw std::invalid_argument("affine lane plane size/stride/range mismatch");
      auto bytes=found->buffer->read(found->offset,found->bytes);read_bytes+=bytes.size();return bytes;
    };
    auto base=load("base_code",w.padded_k/2);
    auto scales=load("scale",w.padded_k/64*2),biases=load("bias",w.padded_k/64*2);
    std::vector<std::byte> packed(geometry.code_bytes),pairs(geometry.scale_bias_bytes);
    for(U32 n=0;n<geometry.n;++n)for(U32 g=0;g<geometry.groups;++g) {
      const U64 dst=(U64(n/32)*geometry.groups+g)*32*32+U64(n%32)*32;
      const U64 src=U64(n)*(w.padded_k/2)+g*32;
      std::copy_n(base.begin()+src,32,packed.begin()+dst);
      const U64 original=(U64(n)*(w.padded_k/64)+g)*2;
      const U64 pair=(U64(g)*geometry.padded_n+n)*4;
      std::uint16_t s,b;std::memcpy(&s,scales.data()+original,2);std::memcpy(&b,biases.data()+original,2);
      if(!std::isfinite(bf16_to_float(s))||!std::isfinite(bf16_to_float(b)))
        throw std::invalid_argument("nonfinite affine lane scale/bias");
      std::copy_n(scales.begin()+original,2,pairs.begin()+pair);
      std::copy_n(biases.begin()+original,2,pairs.begin()+pair+2);
    }
    codes=ctx.upload(packed);sb=ctx.upload(pairs);
  }
};
AffineLaneWeight::AffineLaneWeight(const MetalContext &c,const MaterializedWeightView &w,std::optional<U32> sk):impl_(std::make_unique<Impl>(c,w,sk)){}
AffineLaneWeight::~AffineLaneWeight()=default;
AffineLaneWeight::AffineLaneWeight(AffineLaneWeight &&) noexcept=default;
AffineLaneWeight &AffineLaneWeight::operator=(AffineLaneWeight &&) noexcept=default;
const AffineLaneGeometry &AffineLaneWeight::geometry() const{return impl_->geometry;}
const MetalBuffer &AffineLaneWeight::codes() const{return impl_->codes;}
const MetalBuffer &AffineLaneWeight::scale_bias() const{return impl_->sb;}
U64 AffineLaneWeight::preparation_readback_bytes() const{return impl_->read_bytes;}

struct AffineLaneProjection::Impl {
  struct Params {U32 n,np,k,rows,mp,sk;};
  struct Installed {Profile profile;U64 offset=0;U32 block=16;};
  const MetalContext &ctx;
  AffineLaneGeometry g;
  U32 max_rows;
  AffineLaneExecution mode;
  MetalBuffer xs;
  std::vector<Installed> installed;
  Impl(const MetalContext &ctx,const AffineLaneWeight &w,const MetalBuffer &x,
       MetalBuffer &y,const std::filesystem::path &libs,AffineLaneExecution mode,U32 max_rows)
    :ctx(ctx),g(w.geometry()),max_rows(max_rows),mode(mode) {
    if(!max_rows||max_rows>128 || x.size()<mul(mul(max_rows,g.k),2) ||
       y.size()<mul(mul(max_rows,g.n),2))throw std::invalid_argument("affine lane workspace geometry");
    if(mode!=AffineLaneExecution::PortableReference && mode!=AffineLaneExecution::M5TensorDiagnostic)
      throw std::invalid_argument("unknown affine lane execution mode");
    if(mode==AffineLaneExecution::M5TensorDiagnostic && !affine_lane_m5_diagnostic_eligible(ctx))
      throw std::invalid_argument("M5 lane fragment layout unavailable on this physical GPU; no silent portable substitution");
    const U32 max_mp=(max_rows+15)/16*16;
    xs=ctx.make_buffer(mul(mul(g.groups,max_mp),4));
    static std::atomic<U64> uid{0};const auto owner=++uid;
    installed.reserve(2);
    for(U32 block:{16U,32U}) {
      Params p{g.n,g.padded_n,g.k,max_rows,max_mp,g.split_k};
      auto data=std::as_bytes(std::span(&p,1));std::vector<std::byte> params(data.begin(),data.end());
      Encoder e(ctx,libs,max_rows,"affine.lane."+std::to_string(owner)+".r"+std::to_string(block));
      e.begin_stage("activation_group_sums");
      auto &sum=e.emit("affine_lane","omlx_m5_affine_lane_xsum",
        {.width=g.groups,.height=max_mp},{.width=std::min(g.groups,256U)},{{&x,0,x.size()},{&xs,0,xs.size()}});
      const U64 offset=e.dynamic(sum,2,params);
      e.begin_stage("affine_group_projection");
      const std::string function=mode==AffineLaneExecution::PortableReference?"omlx_m5_affine_lane_portable"
        :"omlx_m5_affine_lane_sk"+std::to_string(g.split_k)+"_r"+std::to_string(block);
      const MetalDispatchSize grid=mode==AffineLaneExecution::PortableReference
        ?MetalDispatchSize{.width=U64((g.n+3)/4)*128,.height=max_rows}
        :MetalDispatchSize{.width=U64(g.padded_n/32)*32*g.split_k,.height=(max_mp+block-1)/block};
      const MetalDispatchSize threads{.width=mode==AffineLaneExecution::PortableReference?128U:32U*g.split_k};
      auto &d=e.emit("affine_lane",function,grid,threads,
        {{&x,0,x.size()},{&w.codes(),0,w.codes().size()},{&w.scale_bias(),0,w.scale_bias().size()},
         {&xs,0,xs.size()},{&y,0,y.size()}});
      e.dynamic(d,5,params);
      auto profile=e.finish();
      installed.push_back({std::move(profile),offset,block});
      auto &stable=installed.back().profile;
      if(ctx.capabilities().profile.compute_icb_eligible)
        stable.icb.emplace(ctx.prepare_icb_profile(stable.plan.key(),
          {.plan=&stable.plan,.bindings=stable.bindings,.parameters=&stable.parameters}));
    }
  }
};
AffineLaneProjection::AffineLaneProjection(const MetalContext &c,const AffineLaneWeight &w,
  const MetalBuffer &x,MetalBuffer &y,const std::filesystem::path &l,AffineLaneExecution m,U32 r)
  :impl_(std::make_unique<Impl>(c,w,x,y,l,m,r)){}
AffineLaneProjection::~AffineLaneProjection()=default;
MetalSubmission AffineLaneProjection::run(U32 rows,bool icb) {
  auto &p=*impl_;if(!rows||rows>p.max_rows)throw std::invalid_argument("affine lane rows outside installed envelope");
  const U32 mp=(rows+15)/16*16;
  auto &i=p.installed[mp<=16?0:1];
  Impl::Params params{p.g.n,p.g.padded_n,p.g.k,rows,mp,p.g.split_k};
  std::memcpy(i.profile.parameter_data.data()+i.offset,&params,sizeof(params));
  i.profile.parameters.write("step",i.profile.parameter_data);
  if(icb && i.profile.icb)return p.ctx.replay_icb(*i.profile.icb);
  const std::array calls{MetalCommandInvocation{.plan=&i.profile.plan,.bindings=i.profile.bindings,.parameters=&i.profile.parameters}};
  return p.ctx.execute_batch(calls);
}
std::string AffineLaneProjection::numeric_contract_id() const {
  return std::string(impl_->mode==AffineLaneExecution::PortableReference?"affine_lane_portable_v1.sk":"affine_lane_m5_candidate_v1.sk")+
    std::to_string(impl_->g.split_k);
}
} // namespace omlx::m5
