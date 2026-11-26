#include "omlx_m5/position_sampling.h"
#include "forward_plan.h"
#include <atomic>
#include <numeric>

namespace omlx::m5 {
using namespace detail::forward;
SamplingContract parse_sampling_contract(std::string_view s) {
  if(s=="legacy_rejection_v1")return SamplingContract::LegacyRejectionV1;
  if(s=="position_coupled_v1")return SamplingContract::PositionCoupledV1;
  throw std::invalid_argument("sampling_contract must be legacy_rejection_v1 or position_coupled_v1");
}
std::string_view sampling_contract_name(SamplingContract c) {
  if(c==SamplingContract::LegacyRejectionV1)return "legacy_rejection_v1";
  if(c==SamplingContract::PositionCoupledV1)return "position_coupled_v1";
  throw std::invalid_argument("unknown sampling contract");
}
namespace {
U64 mix(U64 v){v=(v^(v>>30))*0xBF58476D1CE4E5B9ULL;v=(v^(v>>27))*0x94D049BB133111EBULL;return v^(v>>31);}
void validate(const PositionSamplingRow &p,U32 candidates) {
  if(!std::isfinite(p.temperature)||p.temperature<0 || !std::isfinite(p.top_p)||p.top_p<0||p.top_p>1 ||
     p.top_k>64 || (p.temperature>0 && (!p.top_k||p.top_k>candidates)) || p.valid>1 ||
     (p.purpose!=target_sampling_purpose && p.purpose!=draft_sampling_purpose))
    throw std::invalid_argument("position sampling row/options invalid");
}
}
float position_sampling_uniform(U64 seed,U64 pos,U32 id,U64 purpose) {
  U64 bits=mix(seed^purpose);bits=mix(bits^(pos*0xD1B54A32D192ED03ULL));bits=mix(bits^id);
  return (float(bits>>41)+.5f)*0x1p-23f;
}
U32 position_sample_candidates(std::span<const U32> ids,std::span<const float> scores,const PositionSamplingRow &p) {
  if(ids.empty()||ids.size()!=scores.size())throw std::invalid_argument("position candidate geometry");
  validate(p,U32(ids.size()));
  if(!p.valid)throw std::invalid_argument("cannot draw inactive row");
  std::set<U32> unique(ids.begin(),ids.end());
  if(unique.size()!=ids.size() || std::any_of(scores.begin(),scores.end(),[](float x){return !std::isfinite(x);}))
    throw std::invalid_argument("position candidate IDs/scores invalid");
  std::vector<U32> order(ids.size());std::iota(order.begin(),order.end(),0);
  std::sort(order.begin(),order.end(),[&](U32 a,U32 b){return scores[a]>scores[b]||(scores[a]==scores[b]&&ids[a]<ids[b]);});
  if(p.temperature==0)return ids[order[0]];
  order.resize(p.top_k);std::vector<double> scaled(order.size()),prob(order.size());double total=0;
  for(U32 i=0;i<order.size();++i){scaled[i]=(double(scores[order[i]])-scores[order[0]])/p.temperature;prob[i]=std::exp(scaled[i]);total+=prob[i];}
  double cumulative=0;U32 retained=U32(order.size());
  for(U32 i=0;i<order.size();++i){cumulative+=prob[i]/total;if(cumulative>=p.top_p){retained=i+1;break;}}
  double best=-INFINITY;U32 selected=UINT32_MAX;
  for(U32 i=0;i<retained;++i) {
    const U32 token=ids[order[i]];const double u=position_sampling_uniform(p.seed,p.position,token,p.purpose);
    const double score=scaled[i]-std::log(-std::log(u));
    if(score>best||(score==best&&token<selected)){best=score;selected=token;}
  }
  return selected;
}
struct PositionKeyedSampler::Impl {
  const MetalContext &ctx;U32 vocab,max_rows;MetalBuffer parts,meta,ids,scores;Profile profile;U64 options_offset=0,geometry_offset=0;
  Impl(const MetalContext &c,const MetalBuffer &x,U32 v,U32 rows,
       const std::filesystem::path &libs,std::string_view plan_key):ctx(c),vocab(v),max_rows(rows) {
    if(!v||!rows||rows>128||x.size()<mul(mul(rows,v),2))throw std::invalid_argument("position sampler geometry");
    parts=c.make_buffer(U64(rows)*64*16);meta=c.make_buffer(U64(rows)*8);ids=c.make_buffer(U64(rows)*64*4);scores=c.make_buffer(ids.size());
    static std::atomic<U64> id{0};
    Encoder e(c,libs,rows,plan_key.empty() ? "position.sampler.v2."+std::to_string(++id)
                                          : std::string(plan_key));
    const std::vector<std::byte> init_opts(U64(rows)*sizeof(PositionSamplingRow));
    const auto init_geom=words({v,rows,rows,0});
    auto &d1=e.emit("position_sampling","omlx_m5_position_sample_logits_stage1",{.width=64U*256U,.height=rows},{.width=256},
      {{&x,0,x.size()},{&parts,0,parts.size()}});
    options_offset=e.dynamic(d1,2,init_opts);
    geometry_offset=e.dynamic(d1,3,init_geom);
    auto &d2=e.emit("position_sampling","omlx_m5_position_sample_logits_stage2",{.width=256,.height=rows},{.width=256},
      {{&x,0,x.size()},{&parts,0,parts.size()},{&meta,0,meta.size()},{&ids,0,ids.size()},{&scores,0,scores.size()}});
    (void)e.dynamic(d2,5,init_opts);
    (void)e.dynamic(d2,6,init_geom);
    profile=e.finish();
    if(c.capabilities().profile.compute_icb_eligible)profile.icb.emplace(c.prepare_icb_profile(profile.plan.key(),
      {.plan=&profile.plan,.bindings=profile.bindings,.parameters=&profile.parameters}));
  }
};
PositionKeyedSampler::PositionKeyedSampler(const MetalContext &c,const MetalBuffer &x,U32 v,U32 r,
                                           const std::filesystem::path &l,std::string_view plan_key)
  :impl_(std::make_unique<Impl>(c,x,v,r,l,plan_key)){}
PositionKeyedSampler::~PositionKeyedSampler()=default;
MetalSubmission PositionKeyedSampler::sample_device(std::span<const PositionSamplingRow> rows,bool icb) {
  auto &p=*impl_;if(rows.empty()||rows.size()>p.max_rows)throw std::invalid_argument("position rows outside installed envelope");
  for(const auto &r:rows)validate(r,p.vocab);
  std::memcpy(p.profile.parameter_data.data()+p.options_offset,rows.data(),rows.size_bytes());
  const U32 count=U32(rows.size());std::memcpy(p.profile.parameter_data.data()+p.geometry_offset+4,&count,4);
  p.profile.parameters.write("step",p.profile.parameter_data);
  if(icb&&p.profile.icb)return p.ctx.replay_icb(*p.profile.icb);
  const std::array calls{MetalCommandInvocation{.plan=&p.profile.plan,.bindings=p.profile.bindings,.parameters=&p.profile.parameters}};
  return p.ctx.execute_batch(calls);
}
const MetalBuffer &PositionKeyedSampler::tokens_and_flags() const{return impl_->meta;}
PositionSamplingResult PositionKeyedSampler::resolve_result(std::span<const PositionSamplingRow> rows) {
  auto &p=*impl_;if(rows.empty()||rows.size()>p.max_rows)throw std::invalid_argument("position resolution envelope");
  const U32 count=U32(rows.size());PositionSamplingResult out;
  auto raw=p.meta.read(0,U64(count)*8);out.host_read_bytes=raw.size();out.tokens.resize(count);
  for(U32 i=0;i<count;++i) {
    std::array<U32,2> m;std::memcpy(m.data(),raw.data()+U64(i)*8,8);
    if(m[1]&2)throw std::runtime_error("nonfinite position logits");
    out.tokens[i]=m[0];if(m[1]&1) {
      const U32 k=rows[i].top_k;auto ib=p.ids.read(U64(i)*64*4,U64(k)*4),sb=p.scores.read(U64(i)*64*4,U64(k)*4);
      std::vector<U32> ids(k);std::vector<float> scores(k);std::memcpy(ids.data(),ib.data(),ib.size());std::memcpy(scores.data(),sb.data(),sb.size());
      out.tokens[i]=position_sample_candidates(ids,scores,rows[i]);out.host_read_bytes+=ib.size()+sb.size();++out.compact_fallback_rows;
      const std::array<U32,2> resolved{out.tokens[i],0};
      p.meta.write(std::as_bytes(std::span(resolved)),U64(i)*8);
    }
  }
  return out;
}
PositionSamplingResult PositionKeyedSampler::sample(std::span<const PositionSamplingRow> rows,bool icb) {
  auto submission=sample_device(rows,icb);auto result=resolve_result(rows);
  result.submission=std::move(submission);return result;
}
} // namespace omlx::m5
