#include "../src/forward_plan.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/artifact.h"
#include <iostream>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <nlohmann/json.hpp>
using namespace omlx::m5;using namespace omlx::m5::detail::forward;using Json=nlohmann::json;
namespace {
void require(bool b,std::string_view m){if(!b)throw std::runtime_error(std::string(m));}
std::vector<std::uint16_t> read(const MetalBuffer &b,U64 n){auto x=b.read(0,n*2);std::vector<std::uint16_t> y(n);std::memcpy(y.data(),x.data(),x.size());return y;}
std::uint16_t data(U32 p,U32 h,U32 d,U32 salt,U32 pattern) {
  const int value=int((U64(p)*17+h*31+d*7+salt)%257)-128;
  const float amplitude=pattern==1?4.f:pattern==2?.001f:1.f;
  return float_to_bf16(amplitude*float(value)/512.f);
}
Json operators(const std::filesystem::path &libs,U32 qh,U32 kvh,U32 hd,U32 history,U32 pattern) {
  constexpr U32 rows=8,page=64;const U32 qs=2*qh*hd,kw=kvh*hd,pages=(history+rows+page-1)/page;
  auto c=MetalContext::create();std::vector<U32> table(pages),positions(rows);
  std::iota(table.begin(),table.end(),0);std::reverse(table.begin(),table.end());
  for(U32 r=0;r<rows;++r)positions[r]=history+r;
  std::vector<std::uint16_t> query(U64(rows)*qs),keys(U64(pages)*page*kw,0x7fc0),values(keys.size(),0x7fc0);
  for(U32 r=0;r<rows;++r)for(U32 h=0;h<qh;++h)for(U32 d=0;d<2*hd;++d)
    query[U64(r)*qs+h*hd*2+d]=data(history+r,h,d,113,pattern);
  for(U32 at=0;at<history+rows;++at)for(U32 h=0;h<kvh;++h)for(U32 d=0;d<hd;++d) {
    const U64 off=(U64(table[at/page])*page+at%page)*kw+h*hd+d;
    keys[off]=data(at,h,d,11,pattern);values[off]=data(at,h,d,67,pattern);
  }
  auto qb=c.upload(std::as_bytes(std::span(query))),kb=c.upload(std::as_bytes(std::span(keys))),vb=c.upload(std::as_bytes(std::span(values)));
  auto tb=c.upload(std::as_bytes(std::span(table))),pb=c.upload(std::as_bytes(std::span(positions)));
  auto out=c.make_buffer(U64(rows)*qh*hd*2);const U32 splits=(history+rows+511)/512;
  auto maxima=c.make_buffer(U64(rows)*qh*splits*4),sums=c.make_buffer(maxima.size());
  auto parts=c.make_buffer(U64(rows)*qh*splits*hd*4);
  const auto refs=[](const MetalBuffer &b){return Ref{&b,0,b.size()};};
  const auto params=words({qh,kvh,hd,rows,history,pages*page,page,pages,qs,2*hd,kw,kw,qs,2*hd,splits,512});
  auto build=[&](U32 kind) {
    Encoder e(c,libs,rows,"tf.attention.kind"+std::to_string(kind));
    if(kind==0) {
      auto &d=e.emit("attention_exact","omlx_m5_paged_causal_attention_exact",{.width=32,.height=U64(rows)*qh},{.width=32},
        {refs(qb),sub(refs(qb),hd*2),refs(kb),refs(vb),refs(out),refs(tb),refs(pb)});
      e.dynamic(d,7,std::vector<std::byte>(params.begin(),params.begin()+14*4),3);
    } else if(kind==1) {
      auto &m=e.emit("attention_exact","omlx_m5_paged_attention_exact_max",{.width=U64(splits)*32,.height=U64(rows)*qh},{.width=32},
        {refs(qb),refs(kb),refs(maxima),refs(tb),refs(pb)});e.dynamic(m,5,params,3);
      auto &d=e.emit("attention_exact","omlx_m5_paged_attention_exact_merge",{.width=32,.height=U64(rows)*qh},{.width=32},
        {refs(qb),sub(refs(qb),hd*2),refs(kb),refs(vb),refs(out),refs(tb),refs(maxima),refs(pb)});e.dynamic(d,8,params,3);
    } else if(kind==2) {
      auto &m=e.emit("attention_split_kv","omlx_m5_paged_attention_split_partial",{.width=U64(splits)*32,.height=U64(rows)*qh},{.width=32},
        {refs(qb),refs(kb),refs(vb),refs(maxima),refs(sums),refs(parts),refs(tb)});e.dynamic(m,7,params,3);
      auto &d=e.emit("attention_split_kv","omlx_m5_paged_attention_split_merge",{.width=32,.height=U64(rows)*qh},{.width=32},
        {sub(refs(qb),hd*2),refs(maxima),refs(sums),refs(parts),refs(out)});e.dynamic(d,5,params,3);
    } else if(kind==3) {
      auto &d=e.emit("attention_exact","omlx_m5_paged_causal_attention_exact_1pass",{.width=32,.height=U64(rows)*qh},{.width=32},
        {refs(qb),sub(refs(qb),hd*2),refs(kb),refs(vb),refs(out),refs(tb),refs(pb)});
      e.dynamic(d,7,std::vector<std::byte>(params.begin(),params.begin()+14*4),3);
    } else {
      auto &m=e.emit("attention_exact","omlx_m5_paged_attention_exact_1pass_partial",{.width=U64(splits)*32,.height=U64(rows)*qh},{.width=32},
        {refs(qb),refs(kb),refs(vb),refs(maxima),refs(sums),refs(parts),refs(tb),refs(pb)});e.dynamic(m,8,params,3);
      auto &d=e.emit("attention_exact","omlx_m5_paged_attention_exact_1pass_merge",{.width=32,.height=U64(rows)*qh},{.width=32},
        {sub(refs(qb),hd*2),refs(maxima),refs(sums),refs(parts),refs(out)});e.dynamic(d,5,params,3);
    }
    return e.finish();
  };
  auto serial=build(0),split=build(1),legacy=build(2),serial_1p=build(3),split_1p=build(4);
  if(c.capabilities().profile.compute_icb_eligible)
    for(auto *p:{&serial,&split,&legacy,&serial_1p,&split_1p})p->icb.emplace(c.prepare_icb_profile(p->plan.key(),
      {.plan=&p->plan,.bindings=p->bindings,.parameters=&p->parameters}));
  auto run=[&](Profile &p,U32 live,bool icb=true) {
    for(const auto &patch:p.patches){const U32 v=live;std::memcpy(p.parameter_data.data()+patch.offset+patch.word*4,&v,4);}
    p.parameters.write("step",p.parameter_data);std::vector<std::uint16_t> sentinel(U64(rows)*qh*hd,0xa5a5);out.write(std::as_bytes(std::span(sentinel)));
    if(icb&&p.icb)(void)c.replay_icb(*p.icb);else{const std::array calls{MetalCommandInvocation{.plan=&p.plan,.bindings=p.bindings,.parameters=&p.parameters}};(void)c.execute_batch(calls);}
    auto all=read(out,sentinel.size());require(std::all_of(all.begin()+U64(live)*qh*hd,all.end(),[](auto x){return x==0xa5a5;}),"Attention invalid-row overwrite");
    all.resize(U64(live)*qh*hd);return all;
  };
  const auto full=run(serial,8);require(std::all_of(full.begin(),full.end(),[](auto x){return std::isfinite(bf16_to_float(x));}),"Attention read unused page/tail NaN");
  require(run(split,8)==full && run(split,8,false)==full,"exact max/ordered merge differs from global scan");
  const auto full_1p=run(serial_1p,8);
  require(std::all_of(full_1p.begin(),full_1p.end(),[](auto x){return std::isfinite(bf16_to_float(x));}),"1-pass Attention read unused page/tail NaN");
  require(run(split_1p,8)==full_1p && run(split_1p,8,false)==full_1p,"1-pass split differs from 1-pass single");
  for(U32 w=1;w<=8;++w){
    auto part=run(split,w);require(std::equal(part.begin(),part.end(),full.begin()),"Attention width prefix changed");
    auto part_1p=run(split_1p,w);require(std::equal(part_1p.begin(),part_1p.end(),full_1p.begin()),"1-pass Attention width prefix changed");
  }
  U64 legacy_diff=0;auto old=run(legacy,8);for(U64 i=0;i<old.size();++i)legacy_diff+=old[i]!=full[i];
  for(U32 r=0;r<8;++r) {
    qb.write(std::as_bytes(std::span(query).subspan(U64(r)*qs,qs)));pb.write(std::as_bytes(std::span(positions).subspan(r,1)));
    auto one=run(split,1);require(std::equal(one.begin(),one.end(),full.begin()+U64(r)*qh*hd),"Attention solo logical frontier changed bits");
    auto one_1p=run(split_1p,1);require(std::equal(one_1p.begin(),one_1p.end(),full_1p.begin()+U64(r)*qh*hd),"1-pass Attention solo logical frontier changed bits");
  }
  std::vector<std::uint16_t> permuted(query.size());std::vector<U32> newpositions(8);
  for(U32 r=0;r<8;++r){std::copy_n(query.begin()+U64(7-r)*qs,qs,permuted.begin()+U64(r)*qs);newpositions[r]=positions[7-r];}
  qb.write(std::as_bytes(std::span(permuted)));pb.write(std::as_bytes(std::span(newpositions)));
  auto reordered=run(split,8);auto reordered_1p=run(split_1p,8);
  for(U32 r=0;r<8;++r){
    require(std::equal(reordered.begin()+U64(r)*qh*hd,reordered.begin()+U64(r+1)*qh*hd,
      full.begin()+U64(7-r)*qh*hd),"Attention reordered rows lost position identity");
    require(std::equal(reordered_1p.begin()+U64(r)*qh*hd,reordered_1p.begin()+U64(r+1)*qh*hd,
      full_1p.begin()+U64(7-r)*qh*hd),"1-pass Attention reordered rows lost position identity");
  }
  // Independent FP64 softmax/value/gate mathematics, every target Q head and
  // every output dimension of query row 0; no shared Metal helper arithmetic.
  double error=0,power=0,maximum_error=0,input_value_max=0,max_tolerance_ratio=0;
  for(U32 at=0;at<=history;++at)for(U32 d=0;d<kw;++d)
    input_value_max=std::max(input_value_max,std::abs(double(bf16_to_float(values[(U64(table[at/page])*page+at%page)*kw+d]))));
  for(U32 head=0;head<qh;++head) {
    const U32 kvhead=head/(qh/kvh);std::vector<double> score(history+1),prob(history+1);
    double maximum=-INFINITY,denom=0;
    for(U32 at=0;at<=history;++at) {
      double dot=0;const U64 physical=(U64(table[at/page])*page+at%page)*kw+kvhead*hd;
      for(U32 d=0;d<hd;++d)dot+=double(bf16_to_float(query[head*2*hd+d]))*bf16_to_float(keys[physical+d]);
      score[at]=dot/std::sqrt(double(hd));maximum=std::max(maximum,score[at]);
    }
    for(U32 at=0;at<=history;++at){prob[at]=std::exp(score[at]-maximum);denom+=prob[at];}
    for(U32 d=0;d<hd;++d) {
      double sum=0;for(U32 at=0;at<=history;++at)sum+=prob[at]*bf16_to_float(values[(U64(table[at/page])*page+at%page)*kw+kvhead*hd+d]);
      const double expected=(sum/denom)/(1+std::exp(-double(bf16_to_float(query[head*2*hd+hd+d]))));
      const double delta=bf16_to_float(full[head*hd+d])-expected;
      error+=delta*delta;power+=expected*expected;maximum_error=std::max(maximum_error,std::abs(delta));
      const double tolerance=1e-6*input_value_max+.01*std::abs(expected);
      max_tolerance_ratio=std::max(max_tolerance_ratio,std::abs(delta)/std::max(tolerance,1e-30));
    }
  }
  const double relative=std::sqrt(error/std::max(power,1e-30));
  if(max_tolerance_ratio>1) {
    std::ostringstream message;message<<std::scientific<<std::setprecision(9)<<"Attention FP64 h="<<history<<" pattern="<<pattern
      <<" relative="<<relative<<" max="<<maximum_error<<" rms_reference="<<std::sqrt(power/(qh*hd))<<" tolerance_ratio="<<max_tolerance_ratio;
    throw std::runtime_error(message.str());
  }
  // Causal exclusion is checked by corrupting every future position for row 0.
  qb.write(std::as_bytes(std::span(query)));pb.write(std::as_bytes(std::span(positions)));
  for(U32 at=history+1;at<history+8;++at)for(U32 d=0;d<kw;++d)keys[(U64(table[at/page])*page+at%page)*kw+d]=0x7fc0;
  kb.write(std::as_bytes(std::span(keys)));auto causal=run(split,1);require(std::equal(causal.begin(),causal.end(),full.begin()),"Attention accessed future keys");
  auto causal_1p=run(split_1p,1);require(std::equal(causal_1p.begin(),causal_1p.end(),full_1p.begin()),"1-pass Attention accessed future keys");
  return {{"query_heads",qh},{"kv_heads",kvh},{"head_dim",hd},{"history",history},{"pattern",pattern},
    {"split_count",splits},{"exact_partial_bytes",U64(rows)*qh*splits*4},{"legacy_mismatched_values",legacy_diff},
    {"all_heads_executed",true},{"w1_8_equal",true},{"solo_equal",true},{"reorder_equal",true},
    {"nonmonotonic_pages",true},{"unused_tail_nan_ignored",true},{"future_keys_ignored",true},{"fp64_reference_query_rows",1},{"fp64_reference_all_heads",true},
    {"fp64_relative_l2",relative},{"fp64_max_abs",maximum_error},{"input_value_max",input_value_max},
    {"fp64_element_tolerance","1e-6 * max_abs_V + 0.01 * abs(reference)"},
    {"fp64_max_tolerance_ratio",max_tolerance_ratio},{"passed",true}};
}
Json norm(const std::filesystem::path &libs,U32 qh,U32 kvh,U32 hd,float rotary_factor) {
  constexpr U32 rows=8;const U32 qs=2*qh*hd,ks=kvh*hd,rotary=U32(std::llround(hd*rotary_factor));
  auto c=MetalContext::create();std::vector<std::uint16_t> query(U64(rows)*qs),key(U64(rows)*ks),qn(hd),kn(hd);std::vector<U32> pos(rows);
  for(U32 r=0;r<rows;++r){pos[r]=63+r;for(U32 d=0;d<qs;++d)query[U64(r)*qs+d]=data(r,0,d,77,1);for(U32 d=0;d<ks;++d)key[U64(r)*ks+d]=data(r,0,d,43,1);}
  for(U32 d=0;d<hd;++d){qn[d]=float_to_bf16(1+float(int(d%31)-15)/256);kn[d]=float_to_bf16(1+float(int(d%29)-14)/256);}
  auto q=c.upload(std::as_bytes(std::span(query))),k=c.upload(std::as_bytes(std::span(key))),qw=c.upload(std::as_bytes(std::span(qn))),kw=c.upload(std::as_bytes(std::span(kn))),positions=c.upload(std::as_bytes(std::span(pos)));
  auto qf=c.make_buffer(query.size()*4),kf=c.make_buffer(key.size()*4);
  const auto ref=[](const MetalBuffer &b){return Ref{&b,0,b.size()};};
  auto params=words({qh,kvh,hd,rotary,rows,qs,2*hd,ks,hd,63,word(1e-6f),word(1000000.f),0});
  auto build=[&](U32 mode) {
    Encoder e(c,libs,8,"tf.norm.mode"+std::to_string(mode));MetalDispatchSize grid{.width=32,.height=U64(rows)*(qh+kvh)};
    if(mode==0){auto &d=e.emit("attention_exact","omlx_m5_attention_qk_norm_rope_exact",grid,{.width=32},{ref(q),ref(k),ref(qw),ref(kw),ref(positions)});e.dynamic(d,5,params,4);}
    else if(mode==1) {
      auto &d=e.emit("attention_exact","omlx_m5_qk_norm_fp32_reference",grid,{.width=32},{ref(q),ref(k),ref(qw),ref(kw),ref(qf),ref(kf)});e.dynamic(d,6,params,4);
      auto &r=e.emit("attention_exact","omlx_m5_qk_rope_fp32_reference",grid,{.width=32},{ref(qf),ref(kf),ref(q),ref(k),ref(positions)});e.dynamic(r,5,params,4);
    } else {
      auto &d=e.emit("attention_postprocess","omlx_m5_attention_qk_norm_only",grid,{.width=32},{ref(q),ref(k),ref(qw),ref(kw)});e.dynamic(d,4,params,4);
      auto &r=e.emit("attention_postprocess","omlx_m5_attention_qk_rope_only",grid,{.width=32},{ref(q),ref(k)});e.dynamic(r,2,params,4);
    }return e.finish();
  };
  auto fused=build(0),fp32=build(1),bf16=build(2);
  auto run=[&](Profile &p,U32 live){q.write(std::as_bytes(std::span(query)));k.write(std::as_bytes(std::span(key)));
    for(const auto &patch:p.patches)std::memcpy(p.parameter_data.data()+patch.offset+patch.word*4,&live,4);
    p.parameters.write("step",p.parameter_data);const std::array calls{MetalCommandInvocation{.plan=&p.plan,.bindings=p.bindings,.parameters=&p.parameters}};(void)c.execute_batch(calls);
    return std::make_pair(read(q,query.size()),read(k,key.size()));};
  const auto full=run(fused,8),reference=run(fp32,8);require(full==reference,"FP32 norm/RoPE reference does not preserve fused rounding");
  U64 old_diff=0;auto old=run(bf16,8);for(U64 i=0;i<query.size();++i)old_diff+=old.first[i]!=full.first[i];for(U64 i=0;i<key.size();++i)old_diff+=old.second[i]!=full.second[i];
  for(U32 w=1;w<=8;++w){auto part=run(fused,w);require(std::equal(part.first.begin(),part.first.begin()+U64(w)*qs,full.first.begin()) &&
    std::equal(part.second.begin(),part.second.begin()+U64(w)*ks,full.second.begin()),"norm/RoPE width changed bits");}
  const auto original_q=query,original_k=key;
  const auto original_positions=pos;
  for(U32 r=0;r<8;++r) {
    std::copy_n(original_q.begin()+U64(r)*qs,qs,query.begin());std::copy_n(original_k.begin()+U64(r)*ks,ks,key.begin());
    pos[0]=original_positions[r];positions.write(std::as_bytes(std::span(pos)));
    auto one=run(fused,1);require(std::equal(one.first.begin(),one.first.begin()+qs,full.first.begin()+U64(r)*qs) &&
      std::equal(one.second.begin(),one.second.begin()+ks,full.second.begin()+U64(r)*ks),"norm/RoPE solo absolute position changed bits");
  }
  for(U32 r=0;r<8;++r) {
    std::copy_n(original_q.begin()+U64(7-r)*qs,qs,query.begin()+U64(r)*qs);std::copy_n(original_k.begin()+U64(7-r)*ks,ks,key.begin()+U64(r)*ks);pos[r]=original_positions[7-r];
  }
  positions.write(std::as_bytes(std::span(pos)));auto reordered=run(fused,8);
  for(U32 r=0;r<8;++r)require(std::equal(reordered.first.begin()+U64(r)*qs,reordered.first.begin()+U64(r+1)*qs,full.first.begin()+U64(7-r)*qs) &&
    std::equal(reordered.second.begin()+U64(r)*ks,reordered.second.begin()+U64(r+1)*ks,full.second.begin()+U64(7-r)*ks),"norm/RoPE reorder changed absolute position");
  query=original_q;key=original_k;
  for(U32 r=0;r<8;++r)for(U32 h=0;h<qh;++h)for(U32 d=0;d<hd;++d)require(full.first[U64(r)*qs+h*hd*2+hd+d]==query[U64(r)*qs+h*hd*2+hd+d],"qk norm overwrote gate");
  return {{"head_dim",hd},{"rotary_dim",rotary},{"fp32_reference_bit_equal",true},{"legacy_bf16_intermediate_mismatches",old_diff},
    {"w1_8_equal",true},{"solo_equal",true},{"reorder_equal",true},{"gate_untouched",true},{"passed",true}};
}

Json one_pass_long_hd256(const std::filesystem::path &libs) {
  constexpr U32 qh=24,kvh=4,hd=256,rows=8,page=64,max_history=32768;
  const U32 qs=2*qh*hd,kw=kvh*hd,max_pages=(max_history+rows+page-1)/page,max_splits=(max_history+rows+511)/512;
  auto c=MetalContext::create();
  c.register_library({.key="test.attn_exact",.path=libs/"omlx_m5_attention_exact.metallib"});
  const auto pinfo_part=c.warmup_pipeline({.key="test.1p.partial",.library_key="test.attn_exact",
    .function_name="omlx_m5_paged_attention_exact_1pass_partial",.variant="v1",
    .support_indirect_command_buffers=c.capabilities().profile.compute_icb_eligible});
  const auto pinfo_single=c.warmup_pipeline({.key="test.1p.single",.library_key="test.attn_exact",
    .function_name="omlx_m5_paged_causal_attention_exact_1pass",.variant="v1",
    .support_indirect_command_buffers=c.capabilities().profile.compute_icb_eligible});
  const auto pinfo_merge=c.warmup_pipeline({.key="test.1p.merge",.library_key="test.attn_exact",
    .function_name="omlx_m5_paged_attention_exact_1pass_merge",.variant="v1",
    .support_indirect_command_buffers=c.capabilities().profile.compute_icb_eligible});
  require(pinfo_part.static_threadgroup_memory_bytes<=32768 &&
          pinfo_single.static_threadgroup_memory_bytes<=32768 &&
          pinfo_merge.static_threadgroup_memory_bytes<=32768,
          "1-pass BF16 attention exceeded Apple10 32 KiB threadgroup memory limit");

  std::vector<U32> table(max_pages);
  std::iota(table.begin(),table.end(),0);std::reverse(table.begin(),table.end());
  std::vector<std::uint16_t> keys(U64(max_pages)*page*kw,0x7fc0),values(keys.size(),0x7fc0);
  for(U32 at=0;at<max_history+rows;++at)for(U32 h=0;h<kvh;++h)for(U32 d=0;d<hd;++d) {
    const U64 off=(U64(table[at/page])*page+at%page)*kw+h*hd+d;
    keys[off]=data(at,h,d,11,0);values[off]=data(at,h,d,67,0);
  }
  auto kb=c.upload(std::as_bytes(std::span(keys))),vb=c.upload(std::as_bytes(std::span(values)));
  auto tb=c.upload(std::as_bytes(std::span(table)));
  auto qb=c.make_buffer(U64(rows)*qs*2),pb=c.make_buffer(rows*4),out=c.make_buffer(U64(rows)*qh*hd*2);
  auto maxima=c.make_buffer(U64(rows)*qh*max_splits*4),sums=c.make_buffer(maxima.size());
  auto parts=c.make_buffer(U64(rows)*qh*max_splits*hd*4);
  const auto refs=[](const MetalBuffer &b){return Ref{&b,0,b.size()};};

  Json results=Json::array();
  for(U32 history:{2048u,8192u,32768u}) {
    const U32 splits=(history+rows+511)/512;
    std::vector<std::uint16_t> query(U64(rows)*qs);std::vector<U32> positions(rows);
    for(U32 r=0;r<rows;++r) {
      positions[r]=history+r;
      for(U32 h=0;h<qh;++h)for(U32 d=0;d<2*hd;++d)query[U64(r)*qs+h*hd*2+d]=data(history+r,h,d,113,0);
    }
    qb.write(std::as_bytes(std::span(query)));pb.write(std::as_bytes(std::span(positions)));
    const auto params_t8=words({qh,kvh,hd,rows,history,max_pages*page,page,max_pages,qs,2*hd,kw,kw,qs,2*hd,splits,512});
    const auto params_t1=words({qh,kvh,hd,1u,history,max_pages*page,page,max_pages,qs,2*hd,kw,kw,qs,2*hd,splits,512});

    auto exec=[&](Profile &p,U32 live,bool icb=true) {
      if(icb&&c.capabilities().profile.compute_icb_eligible&&!p.icb)
        p.icb.emplace(c.prepare_icb_profile(p.plan.key(),{.plan=&p.plan,.bindings=p.bindings,.parameters=&p.parameters}));
      std::vector<std::uint16_t> sentinel(U64(rows)*qh*hd,0xa5a5);out.write(std::as_bytes(std::span(sentinel)));
      if(icb&&p.icb)(void)c.replay_icb(*p.icb);
      else{const std::array calls{MetalCommandInvocation{.plan=&p.plan,.bindings=p.bindings,.parameters=&p.parameters}};(void)c.execute_batch(calls);}
      auto all=read(out,sentinel.size());
      require(std::all_of(all.begin()+U64(live)*qh*hd,all.end(),[](auto x){return x==0xa5a5;}),"Long HD256 invalid-row overwrite");
      all.resize(U64(live)*qh*hd);return all;
    };

    // 1. 2-pass reference (T=8, GQA TG tptg.y=12)
    auto p_ref=[&]{
      Encoder e(c,libs,rows,"tf.long.2p.l"+std::to_string(history));
      auto &d=e.emit("attention_exact","omlx_m5_paged_causal_attention_exact",{.width=32,.height=U64(rows)*qh},{.width=32,.height=12},
        {refs(qb),sub(refs(qb),hd*2),refs(kb),refs(vb),refs(out),refs(tb),refs(pb)});
      e.dynamic(d,7,std::vector<std::byte>(params_t8.begin(),params_t8.begin()+14*4));return e.finish();
    }();
    // 2. 1-pass single (T=8, scalar tptg.y=1 & GQA TG tptg.y=12)
    auto p_1p_single_s1=[&]{
      Encoder e(c,libs,rows,"tf.long.1ps.y1.l"+std::to_string(history));
      auto &d=e.emit("attention_exact","omlx_m5_paged_causal_attention_exact_1pass",{.width=32,.height=U64(rows)*qh},{.width=32,.height=1},
        {refs(qb),sub(refs(qb),hd*2),refs(kb),refs(vb),refs(out),refs(tb),refs(pb)});
      e.dynamic(d,7,std::vector<std::byte>(params_t8.begin(),params_t8.begin()+14*4));return e.finish();
    }();
    auto p_1p_single_gqa=[&]{
      Encoder e(c,libs,rows,"tf.long.1ps.gqa.l"+std::to_string(history));
      auto &d=e.emit("attention_exact","omlx_m5_paged_causal_attention_exact_1pass",{.width=32,.height=U64(rows)*qh},{.width=32,.height=12},
        {refs(qb),sub(refs(qb),hd*2),refs(kb),refs(vb),refs(out),refs(tb),refs(pb)});
      e.dynamic(d,7,std::vector<std::byte>(params_t8.begin(),params_t8.begin()+14*4));return e.finish();
    }();
    // 3. 1-pass split (T=8, GQA TG tptg.y=12 & scalar tptg.y=1)
    auto p_1p_split_gqa=[&]{
      Encoder e(c,libs,rows,"tf.long.1pkv.t8.gqa.l"+std::to_string(history));
      auto &m=e.emit("attention_exact","omlx_m5_paged_attention_exact_1pass_partial",{.width=U64(splits)*32,.height=U64(rows)*qh},{.width=32,.height=12},
        {refs(qb),refs(kb),refs(vb),refs(maxima),refs(sums),refs(parts),refs(tb),refs(pb)});e.dynamic(m,8,params_t8);
      auto &d=e.emit("attention_exact","omlx_m5_paged_attention_exact_1pass_merge",{.width=32,.height=U64(rows)*qh},{.width=32},
        {sub(refs(qb),hd*2),refs(maxima),refs(sums),refs(parts),refs(out)});e.dynamic(d,5,params_t8);return e.finish();
    }();
    auto p_1p_split_y1=[&]{
      Encoder e(c,libs,rows,"tf.long.1pkv.t8.y1.l"+std::to_string(history));
      auto &m=e.emit("attention_exact","omlx_m5_paged_attention_exact_1pass_partial",{.width=U64(splits)*32,.height=U64(rows)*qh},{.width=32,.height=1},
        {refs(qb),refs(kb),refs(vb),refs(maxima),refs(sums),refs(parts),refs(tb),refs(pb)});e.dynamic(m,8,params_t8);
      auto &d=e.emit("attention_exact","omlx_m5_paged_attention_exact_1pass_merge",{.width=32,.height=U64(rows)*qh},{.width=32},
        {sub(refs(qb),hd*2),refs(maxima),refs(sums),refs(parts),refs(out)});e.dynamic(d,5,params_t8);return e.finish();
    }();
    // 4. 1-pass split (T=1 AR, GQA TG tptg.y=6)
    auto p_1p_split_t1=[&]{
      Encoder e(c,libs,1,"tf.long.1pkv.t1.gqa.l"+std::to_string(history));
      auto &m=e.emit("attention_exact","omlx_m5_paged_attention_exact_1pass_partial",{.width=U64(splits)*32,.height=U64(qh)},{.width=32,.height=6},
        {refs(qb),refs(kb),refs(vb),refs(maxima),refs(sums),refs(parts),refs(tb),refs(pb)});e.dynamic(m,8,params_t1);
      auto &d=e.emit("attention_exact","omlx_m5_paged_attention_exact_1pass_merge",{.width=32,.height=U64(qh)},{.width=32},
        {sub(refs(qb),hd*2),refs(maxima),refs(sums),refs(parts),refs(out)});e.dynamic(d,5,params_t1);return e.finish();
    }();

    const auto ref_t8=exec(p_ref,rows);
    const auto out_t8_split_icb=exec(p_1p_split_gqa,rows,true);
    const auto out_t8_split_dir=exec(p_1p_split_gqa,rows,false);
    const auto out_t8_split_y1=exec(p_1p_split_y1,rows,true);
    const auto out_t8_single_gqa=exec(p_1p_single_gqa,rows,true);
    const auto out_t8_single_s1=exec(p_1p_single_s1,rows,true);

    require(out_t8_split_icb==out_t8_split_dir,"Long HD256 1-pass ICB vs direct mismatch");
    require(out_t8_split_icb==out_t8_split_y1,"Long HD256 1-pass GQA TG vs scalar mismatch");
    require(out_t8_split_icb==out_t8_single_gqa && out_t8_split_icb==out_t8_single_s1,
            "Long HD256 1-pass split vs single-pass 0-ULP mismatch");

    // Verify T=1 AR (col=0 at history+r) == T=8 verify (col=r at history) 0 ULP across all r=0..7!
    for(U32 r=0;r<rows;++r) {
      qb.write(std::as_bytes(std::span(query).subspan(U64(r)*qs,qs)));
      pb.write(std::as_bytes(std::span(positions).subspan(r,1)));
      const auto out_t1=exec(p_1p_split_t1,1,true);
      require(std::equal(out_t1.begin(),out_t1.end(),out_t8_split_icb.begin()+U64(r)*qh*hd),
              "Long HD256 T=1 vs T=8 0-ULP parity failed at L="+std::to_string(history)+" row="+std::to_string(r));
    }

    // Compare 1-pass vs 2-pass reference within BF16 tolerance across all 8*24*256 elements
    const auto bf16_ord=[](std::uint16_t x)->int{return (x&0x8000u)?-int(x&0x7fffu):int(x);};
    U64 exact_matches=0;int max_ulp=0;double max_abs=0,sq_err=0,sq_ref=0;
    for(std::size_t i=0;i<ref_t8.size();++i) {
      const auto a=out_t8_split_icb[i],b=ref_t8[i];
      if(a==b)++exact_matches;
      const int ulp=std::abs(bf16_ord(a)-bf16_ord(b));
      max_ulp=std::max(max_ulp,ulp);
      const double fa=bf16_to_float(a),fb=bf16_to_float(b),diff=std::abs(fa-fb);
      max_abs=std::max(max_abs,diff);sq_err+=diff*diff;sq_ref+=fb*fb;
    }
    const double rel_l2=std::sqrt(sq_err/std::max(sq_ref,1e-30));
    require(max_ulp<=2 && rel_l2<=1e-3,
            "Long HD256 1-pass vs 2-pass exceeded BF16 tolerance at L="+std::to_string(history)+
            " max_ulp="+std::to_string(max_ulp)+" rel_l2="+std::to_string(rel_l2));
    results.push_back({{"history",history},{"splits",splits},{"t1_eq_t8_zero_ulp",true},
      {"split_eq_single_zero_ulp",true},{"gqa_eq_scalar_zero_ulp",true},{"icb_eq_direct_zero_ulp",true},
      {"elements",ref_t8.size()},{"exact_2pass_bit_matches",exact_matches},
      {"max_bf16_ulp_to_2pass",max_ulp},{"max_abs_to_2pass",max_abs},{"relative_l2_to_2pass",rel_l2}});
  }
  return {{"query_heads",qh},{"kv_heads",kvh},{"head_dim",hd},
    {"partial_static_tg_bytes",pinfo_part.static_threadgroup_memory_bytes},
    {"single_static_tg_bytes",pinfo_single.static_threadgroup_memory_bytes},
    {"merge_static_tg_bytes",pinfo_merge.static_threadgroup_memory_bytes},
    {"depths",results},{"passed",true}};
}
}
int main(int argc,char **argv) {
  try {
    if(argc!=2 && argc!=3)throw std::invalid_argument("tf_attention_exact_tests LIBRARIES [TARGET_CONFIG]");
    const std::filesystem::path libs(argv[1]);U32 qh=4,kvh=2,hd=64;float rotary=.5;
    if(argc==3){const auto c=load_qwen_model_config(argv[2]);qh=c.num_attention_heads;kvh=c.num_key_value_heads;hd=c.head_dim;rotary=c.partial_rotary_factor;}
    Json cases=Json::array();
    for(U32 history:argc==3?std::vector<U32>{2048,8192,32768}:std::vector<U32>{0,1,63,64,65,511,512,513})
      for(U32 p=0;p<3;++p)cases.push_back(operators(libs,qh,kvh,hd,history,p));
    Json out={{"schema","omlx-m5-p06-tf-attention-v1"},{"scope",argc==3?"real target geometry, synthetic KV/queries":"small geometry/page/causal operators"},
      {"cases",cases},{"norm",norm(libs,qh,kvh,hd,rotary)},
      {"full_model_exact_qualified",false},{"m5_performance_qualified",false},{"passed",true}};
    if(argc==2)out["one_pass_long_hd256"]=one_pass_long_hd256(libs);
    std::cout<<out.dump(2)<<'\n';return 0;
  }catch(const std::exception&e){std::cerr<<"P06-TF: "<<e.what()<<'\n';return 1;}
}
