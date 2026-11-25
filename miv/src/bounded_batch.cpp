#include "omlx_m5/bounded_batch.h"
#include "forward_plan.h"
#include <atomic>
#include <set>
namespace omlx::m5 {
using namespace detail::forward;
namespace {
struct BatchLayout {U32 count=0,total=0,columns=0,capacity=0;std::array<U32,8> slots{},widths{},offsets{},valid{};};
static_assert(sizeof(BatchLayout)==144);
BatchLayout layout(std::span<const BatchRow> rows,U32 slots,U32 columns,bool multi){
 if(rows.empty() || rows.size()>slots)throw std::invalid_argument("batch layout outside slot envelope");
 BatchLayout out;out.count=U32(rows.size());out.columns=columns;out.capacity=slots;std::set<U32> active;
 for(U32 i=0;i<rows.size();++i){auto r=rows[i];if(r.valid>1 || r.slot>=slots || r.width<1 || r.width>8 || (r.valid && !active.insert(r.slot).second))throw std::invalid_argument("batch slot/width/mask invalid");
  out.slots[i]=r.slot;out.widths[i]=r.width;out.valid[i]=r.valid;out.offsets[i]=out.total;if(r.valid)out.total+=multi?r.width:1;}
 return out;
}
Ref ref(const MetalBuffer &b){return {&b,0,b.size()};}
MetalSubmission submit(const MetalContext &c,Profile &p,bool icb){if(icb && p.icb)return c.replay_icb(*p.icb);const std::array calls{MetalCommandInvocation{.plan=&p.plan,.bindings=p.bindings,.parameters=&p.parameters}};return c.execute_batch(calls);}
void prepare_icb(const MetalContext &c,Profile &p){if(c.capabilities().profile.compute_icb_eligible)p.icb.emplace(c.prepare_icb_profile(p.plan.key(),{.plan=&p.plan,.bindings=p.bindings,.parameters=&p.parameters}));}
std::vector<std::byte> bytes(const BatchLayout &p){const auto data=std::as_bytes(std::span(&p,1));return {data.begin(),data.end()};}
}
struct CoupledBatchSampler::Impl {
 const MetalContext &context;std::filesystem::path libraries;U32 slots,vocab;MetalBuffer gathered;Profile profile;U64 offset=0;std::unique_ptr<PositionKeyedSampler> sampler;std::unique_ptr<PositionKeyedSampler> direct_sampler;
 Impl(const MetalContext&c,std::span<QwenForward *const> programs,const std::filesystem::path&libs):context(c),libraries(libs),slots(U32(programs.size())),vocab(programs.empty()?0:programs[0]->vocabulary_size()){
  if(slots<2 || slots>8 || !vocab)throw std::invalid_argument("coupled sampler requires B2-8");
  for(auto *program:programs)if(!program || program->vocabulary_size()!=vocab)throw std::invalid_argument("coupled sampler mixed vocabulary");
  gathered=c.make_buffer(U64(slots)*vocab*2);static std::atomic<U64> id{0};Encoder e(c,libraries,slots,"coupled.batch.sample."+std::to_string(++id));
  std::vector<Ref> inputs;for(U32 i=0;i<8;++i)inputs.push_back(ref(programs[std::min(i,slots-1)]->batch_logits_buffer()));inputs.push_back(ref(gathered));
  auto &g=e.emit("bounded_batch","omlx_m5_gather_slot_logits",{.width=U64(slots)*vocab},{.width=256},{inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6],inputs[7],inputs[8]});offset=e.dynamic(g,9,bytes({}));profile=e.finish();prepare_icb(c,profile);
  sampler=std::make_unique<PositionKeyedSampler>(c,gathered,vocab,slots,libraries);
  const auto &primary_logits=programs[0]->batch_projection_output_buffer();
  if(primary_logits.size()>=mul(mul(slots,vocab),2U)){
   direct_sampler=std::make_unique<PositionKeyedSampler>(c,primary_logits,vocab,slots,libraries);
  }
 }
};
CoupledBatchSampler::CoupledBatchSampler(const MetalContext&c,std::span<QwenForward *const>p,const std::filesystem::path&l):impl_(std::make_unique<Impl>(c,p,l)){}
CoupledBatchSampler::~CoupledBatchSampler()=default;
bool CoupledBatchSampler::uses_icb() const{return impl_->profile.icb.has_value();}
bool CoupledBatchSampler::can_sample_contiguous() const noexcept{return impl_->direct_sampler!=nullptr;}
std::vector<U64> CoupledBatchSampler::resource_addresses() const{return {impl_->gathered.gpu_address(),impl_->profile.parameters.gpu_address(),impl_->sampler->tokens_and_flags().gpu_address()};}
PositionSamplingResult CoupledBatchSampler::sample_contiguous(std::span<const PositionSamplingRow> keys,bool icb){
 auto &p=*impl_;
 if(!p.direct_sampler)throw std::logic_error("contiguous batch sampler unavailable");
 if(keys.empty() || keys.size()>p.slots)throw std::invalid_argument("contiguous batch sampling layout invalid");
 return p.direct_sampler->sample(keys,icb);
}
PositionSamplingResult CoupledBatchSampler::sample(std::span<const BatchRow> rows,std::span<const PositionSamplingRow> keys,bool icb){
 auto&p=*impl_;if(rows.size()!=keys.size())throw std::invalid_argument("batch sampling key layout mismatch");auto params=layout(rows,p.slots,p.vocab,false);std::vector<PositionSamplingRow> selected(keys.begin(),keys.end());
 for(U32 i=0;i<rows.size();++i)selected[i].valid=rows[i].valid;
 std::memcpy(p.profile.parameter_data.data()+p.offset,&params,sizeof(params));p.profile.parameters.write("step",p.profile.parameter_data);auto gathered=submit(p.context,p.profile,icb);
 auto out=p.sampler->sample(selected,icb);out.submission.dispatch_count+=gathered.dispatch_count;out.submission.timing.gpu_ms+=gathered.timing.gpu_ms;out.submission.timing.wall_ms+=gathered.timing.wall_ms;return out;
}
struct SharedHeadProjection::Impl {
 const MetalContext &context;std::filesystem::path libraries;U32 slots,h,v,capacity,last=0;MetalBuffer input,output,sums,zero;Profile profile;U64 gather_offset=0,scatter_offset=0;
 Impl(const MetalContext&c,std::span<QwenForward *const>programs,const std::filesystem::path&libs):context(c),libraries(libs),slots(U32(programs.size())),h(programs.empty()?0:U32(programs[0]->head_weight().in_features)),v(programs.empty()?0:programs[0]->vocabulary_size()),capacity(slots*8){
  if(slots<2 || slots>8 || !h || !v)throw std::invalid_argument("shared head requires B2-8");
  const auto head=programs[0]->head_weight();
  if(head.layout.find("nax_tiled")!=std::string::npos)
   throw std::invalid_argument("SharedHeadProjection requires raw weight planes; use run_packed_batch for pre-tiled NAX weights");
  auto route=select_row_exact_linear_route(head.format,v,h,capacity);
  for(auto *p:programs)if(!p || p->vocabulary_size()!=v || p->head_weight().object_id!=head.object_id || p->head_weight().in_features!=h || p->inspect_profile(1).numeric_contract_id!="linear_row_exact_portable_v1")throw std::invalid_argument("shared head mixed weights/numeric contracts");
  input=c.make_buffer(U64(capacity)*h*2);output=c.make_buffer(U64(capacity)*v*2);sums=c.make_buffer(U64(capacity)*(route.group?h/route.group:1)*4);zero=c.make_buffer(4);zero.write(std::array<std::byte,4>{});
  static std::atomic<U64> id{0};Encoder e(c,libraries,capacity,"coupled.shared.head."+std::to_string(++id));e.begin_stage("gather_norm");std::vector<Ref> refs;
  for(U32 i=0;i<8;++i)refs.push_back(ref(programs[std::min(i,slots-1)]->batch_projection_input_buffer()));refs.push_back(ref(input));
  auto&g=e.emit("bounded_batch","omlx_m5_gather_projection_rows",{.width=U64(capacity)*h},{.width=256},{refs[0],refs[1],refs[2],refs[3],refs[4],refs[5],refs[6],refs[7],refs[8]});gather_offset=e.dynamic(g,9,bytes(BatchLayout{.columns=h,.capacity=capacity}));
  e.begin_stage("shared_head_linear");linear(e,head,ref(input),ref(output),v,h,ref(sums),ref(zero),route);
  e.record_weight(head,v,h,route.family,"actual_shared_head_rerun_v1",{},route.numeric_contract_id,route.row_exact_group,true);
  e.begin_stage("scatter_head");refs={ref(output)};for(U32 i=0;i<8;++i)refs.push_back(ref(programs[std::min(i,slots-1)]->batch_projection_output_buffer()));for(U32 i=0;i<8;++i)refs.push_back(ref(programs[std::min(i,slots-1)]->batch_logits_buffer()));
  auto&s=e.emit("bounded_batch","omlx_m5_scatter_projection_rows",{.width=U64(capacity)*v},{.width=256},{refs[0],refs[1],refs[2],refs[3],refs[4],refs[5],refs[6],refs[7],refs[8],refs[9],refs[10],refs[11],refs[12],refs[13],refs[14],refs[15],refs[16]});scatter_offset=e.dynamic(s,17,bytes(BatchLayout{.columns=v,.capacity=capacity+1}));profile=e.finish();prepare_icb(c,profile);
 }
};
SharedHeadProjection::SharedHeadProjection(const MetalContext&c,std::span<QwenForward *const>p,const std::filesystem::path&l):impl_(std::make_unique<Impl>(c,p,l)){}
SharedHeadProjection::~SharedHeadProjection()=default;
U32 SharedHeadProjection::last_rows() const{return impl_->last;}
QwenProfileWeightRoute SharedHeadProjection::execution_route() const {
 const auto&r=impl_->profile.weight_routes.at(0);
 return {r.stage,r.object,r.format,r.library,r.function,r.family,r.source,r.fallback_reason,
  r.n,r.k,r.t,r.grid,r.threads,r.numeric_contract_id,r.row_exact_group,r.row_exact_eligible};
}
std::vector<U64> SharedHeadProjection::resource_addresses() const{return {impl_->input.gpu_address(),impl_->output.gpu_address(),impl_->sums.gpu_address(),impl_->profile.parameters.gpu_address()};}
MetalSubmission SharedHeadProjection::project(std::span<const BatchRow> rows,bool icb){auto&p=*impl_;auto params=layout(rows,p.slots,p.h,true);p.last=params.total;if(!p.last)return {};
 std::memcpy(p.profile.parameter_data.data()+p.gather_offset,&params,sizeof(params));params.columns=p.v;std::memcpy(p.profile.parameter_data.data()+p.scatter_offset,&params,sizeof(params));
 for(const auto &patch:p.profile.patches){if(patch.position)throw std::logic_error("shared head unexpected position patch");const U32 count=p.last*patch.multiplier;std::memcpy(p.profile.parameter_data.data()+patch.offset+patch.word*4,&count,4);}
 p.profile.parameters.write("step",p.profile.parameter_data);return submit(p.context,p.profile,icb);
}
} // namespace omlx::m5
