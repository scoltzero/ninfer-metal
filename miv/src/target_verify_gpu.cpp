#include "omlx_m5/target_verify_gpu.h"

#include "forward_plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace omlx::m5 {
namespace {
using namespace detail::forward;
constexpr U32 max_topk = 64, max_proposal_k = 32, max_width = 8;
constexpr U32 chunk = 256;
struct VerifyParams {
  U32 vocab, topk, blocks, groups, width, proposal_k, position;
  float temperature, top_p;
  float accept_uniform[7], correction_uniform[7], bonus_uniform;
};
static_assert(sizeof(VerifyParams) == 96);
} // namespace

std::uint64_t target_gpu_verify_workspace_bytes(U32 vocabulary,SamplingContract contract) {
  if (!vocabulary)
    throw std::invalid_argument("target verifier vocabulary is empty");
  if(contract==SamplingContract::PositionCoupledV1)return 7*4+8*4+16+8*8+2*8*64*4+8*64*16;
  (void)sampling_contract_name(contract);
  const U64 blocks = (U64(vocabulary) + chunk - 1) / chunk;
  const U64 groups = (blocks + 31) / 32;
  return (2 * max_width * (blocks + groups + 1) * max_topk + max_width + 7 +
          2 * 7 * max_proposal_k + 4) *
         4;
}

class TargetGpuVerifier::Impl {
public:
  struct CoupledProfile {
    Profile profile;
    U64 options_offset = 0;
  };

  Impl(const MetalContext &selected, const MetalBuffer &target_logits,
       U32 vocabulary,const std::filesystem::path &libraries,SamplingContract contract)
      : context(selected), logits(target_logits), vocab(vocabulary),contract(contract) {
    (void)sampling_contract_name(contract);
    if(contract==SamplingContract::PositionCoupledV1) {
      if(!vocab || logits.size()<mul(mul(8U,vocab),2U))
        throw std::invalid_argument("position sampler geometry");
      allocate("parts",8*64*16);
      allocate("sampled",8*8);
      allocate("compact_ids",8*64*4);
      allocate("compact_scores",8*64*4);
      allocate("proposal_tokens",7*4);
      allocate("licensed",8*4);
      allocate("meta",16);
      const Ref logits_ref{&logits,0,logits.size()};
      for(U32 w=1;w<=8;++w) {
        auto &cp=coupled_profiles[w-1];
        Encoder e(context,libraries,w,"target.verify.position-coupled-v2.w"+std::to_string(w));
        const std::vector<std::byte> init_opts(U64(w)*sizeof(PositionSamplingRow));
        const auto geom=words({vocab,w,w,0});
        e.begin_stage("sample_logits");
        auto &s1=e.emit("position_sampling","omlx_m5_position_sample_logits_stage1",
          {.width=64U*256U,.height=w},{.width=256U},{logits_ref,get("parts")});
        cp.options_offset=e.dynamic(s1,2,init_opts);
        e.constant(s1,3,geom);
        auto &s2=e.emit("position_sampling","omlx_m5_position_sample_logits_stage2",
          {.width=256U,.height=w},{.width=256U},
          {logits_ref,get("parts"),get("sampled"),get("compact_ids"),get("compact_scores")});
        (void)e.dynamic(s2,5,init_opts);
        e.constant(s2,6,geom);
        e.begin_stage("verify_equality");
        auto &eq=e.emit("position_sampling","omlx_m5_position_verify_equality",
          {.width=1U},{.width=32U},
          {get("sampled"),get("proposal_tokens"),get("licensed"),get("meta")});
        e.constant(eq,4,words({w}));
        cp.profile=e.finish(true);
      }
      {
        Encoder e(context,libraries,8,"target.verify.position-coupled-v2.eq-only");
        auto &d=e.emit("position_sampling","omlx_m5_position_verify_equality",
          {.width=1U},{.width=32U},
          {get("sampled"),get("proposal_tokens"),get("licensed"),get("meta")});
        e.profile.sampling_offset=e.dynamic(d,4,words({8}));
        equality_only_profile=e.finish(true);
      }
      return;
    }
    if (vocab < max_topk || U64(max_width) * vocab * 2 > logits.size() ||
        target_gpu_verify_workspace_bytes(vocab) >
            context.capabilities().max_buffer_length)
      throw std::invalid_argument("target verifier resource geometry invalid");
    const U64 blocks = (U64(vocab) + chunk - 1) / chunk;
    const U64 groups = (blocks + 31) / 32;
    allocate("partial_ids", max_width * blocks * max_topk * 4);
    allocate("partial_scores", max_width * blocks * max_topk * 4);
    allocate("group_ids", max_width * groups * max_topk * 4);
    allocate("group_scores", max_width * groups * max_topk * 4);
    allocate("ids", max_width * max_topk * 4);
    allocate("scores", max_width * max_topk * 4);
    allocate("proposal_tokens", 7 * 4);
    allocate("proposal_ids", 7 * max_proposal_k * 4);
    allocate("proposal_q", 7 * max_proposal_k * 4);
    allocate("licensed", max_width * 4);
    allocate("meta", 4 * 4);
    Encoder e(context, libraries, 8, "target.verify.selector.v1");
    const VerifyParams initial{.vocab = vocab,
                               .topk = max_topk,
                               .blocks = U32(blocks),
                               .groups = U32(groups),
                               .width = max_width};
    const auto raw = std::as_bytes(std::span(&initial, 1));
    const std::vector<std::byte> params(raw.begin(), raw.end());
    e.begin_stage("target_topk_chunks");
    auto &chunks =
        e.emit("target_verify_gpu", "omlx_m5_target_topk_chunks",
               {.width = blocks, .height = max_width}, {.width = 128},
               {{&logits, 0, logits.size()},
                get("partial_ids"),
                get("partial_scores")});
    e.profile.sampling_offset = e.dynamic(chunks, 3, params);
    e.begin_stage("target_topk_reduce");
    auto &reduce =
        e.emit("target_verify_gpu", "omlx_m5_target_topk_reduce32",
               {.width = groups, .height = max_width}, {.width = 128},
               {get("partial_ids"), get("partial_scores"), get("group_ids"),
                get("group_scores")});
    e.dynamic(reduce, 4, params);
    e.begin_stage("target_topk_merge");
    auto &merge = e.emit(
        "target_verify_gpu", "omlx_m5_target_topk_merge", {.width = max_width},
        {.width = 32},
        {get("group_ids"), get("group_scores"), get("ids"), get("scores")});
    e.dynamic(merge, 4, params);
    e.begin_stage("target_verify_decision");
    auto &decision = e.emit("target_verify_gpu", "omlx_m5_target_verify_select",
                            {.width = 1}, {.width = 32},
                            {get("ids"), get("scores"), get("proposal_tokens"),
                             get("proposal_ids"), get("proposal_q"),
                             get("licensed"), get("meta")});
    e.dynamic(decision, 7, params);
    profile = e.finish();
    if (context.capabilities().profile.compute_icb_eligible)
      profile.icb.emplace(context.prepare_icb_profile(
          profile.plan.key(), {.plan = &profile.plan,
                               .bindings = profile.bindings,
                               .parameters = &profile.parameters}));
  }

  void validate_coupled_row(const PositionSamplingRow &r) const {
    if(!std::isfinite(r.temperature)||r.temperature<0||
       !std::isfinite(r.top_p)||r.top_p<0||r.top_p>1||
       r.top_k>64||(r.temperature>0&&(!r.top_k||r.top_k>vocab))||r.valid>1||
       (r.purpose!=target_sampling_purpose&&r.purpose!=draft_sampling_purpose))
      throw std::invalid_argument("position sampling row/options invalid");
  }

  U64 resolve_coupled_compact_rows(std::span<const PositionSamplingRow> rows) {
    const U32 count=U32(rows.size());
    auto raw=get("sampled").buffer->read(0,U64(count)*8);
    U64 read_bytes=raw.size();
    for(U32 i=0;i<count;++i) {
      std::array<U32,2> m{};
      std::memcpy(m.data(),raw.data()+U64(i)*8,8);
      if(m[1]&2)throw std::runtime_error("nonfinite coupled target logits");
      if(m[1]&1) {
        const U32 k=rows[i].top_k;
        auto ib=get("compact_ids").buffer->read(U64(i)*64*4,U64(k)*4);
        auto sb=get("compact_scores").buffer->read(U64(i)*64*4,U64(k)*4);
        std::vector<U32> cids(k);
        std::vector<float> cscores(k);
        std::memcpy(cids.data(),ib.data(),ib.size());
        std::memcpy(cscores.data(),sb.data(),sb.size());
        const U32 tok=position_sample_candidates(cids,cscores,rows[i]);
        read_bytes+=ib.size()+sb.size();
        const std::array<U32,2> resolved{tok,0};
        const_cast<MetalBuffer *>(get("sampled").buffer)
            ->write(std::as_bytes(std::span(resolved)),U64(i)*8);
      }
    }
    return read_bytes;
  }

  MetalSubmission run_equality_only(U32 width, bool use_icb) {
    auto &cp=coupled_profiles[width-1];
    if(use_icb&&cp.profile.icb) {
      return context.replay_icb(*cp.profile.icb,"verify_equality");
    }
    auto &eq=equality_only_profile;
    std::memcpy(eq.parameter_data.data()+eq.sampling_offset,&width,4);
    eq.parameters.write("step",eq.parameter_data);
    const std::array calls{MetalCommandInvocation{
        .plan=&eq.plan,.bindings=eq.bindings,.parameters=&eq.parameters}};
    return context.execute_batch(calls);
  }

  TargetGpuVerifyDecision finish_coupled_submission(
      std::span<const PositionSamplingRow> rows, U32 width, U32 proposal_count,
      bool use_icb, MetalSubmission &&initial_sub) {
    TargetGpuVerifyDecision result;
    result.submission=std::move(initial_sub);
    auto read_meta=[&]{
      auto raw=get("meta").buffer->read(0,16);
      result.host_read_bytes+=16;
      std::array<U32,4> m{};
      std::memcpy(m.data(),raw.data(),16);
      return m;
    };
    auto meta=read_meta();
    if(meta[3]&2)throw std::runtime_error("nonfinite coupled target logits");
    if(meta[3]&1) {
      result.host_read_bytes+=resolve_coupled_compact_rows(rows);
      result.compact_cpu_fallback=true;
      auto rerun=run_equality_only(width,use_icb);
      result.submission.dispatch_count+=rerun.dispatch_count;
      result.submission.timing.gpu_ms+=rerun.timing.gpu_ms;
      result.submission.timing.wall_ms+=rerun.timing.wall_ms;
      meta=read_meta();
    }
    if(!meta[0]||meta[0]>width||meta[1]>proposal_count||meta[3])
      throw std::runtime_error("position verification metadata invalid");
    auto licensed=get("licensed").buffer->read(0,U64(meta[0])*4);
    result.host_read_bytes+=licensed.size();
    result.verification.licensed_outputs.resize(meta[0]);
    std::memcpy(result.verification.licensed_outputs.data(),licensed.data(),licensed.size());
    result.verification.accepted_proposals=meta[1];
    result.verification.used_correction=meta[2]!=0;
    return result;
  }

  Ref get(std::string_view name) const { return refs.at(std::string(name)); }
  void allocate(std::string name, U64 bytes) {
    auto buffer = std::make_unique<MetalBuffer>(context.make_buffer(bytes));
    refs.emplace(std::move(name), Ref{buffer.get(), 0, bytes});
    buffers.push_back(std::move(buffer));
  }
  const MetalContext &context;
  const MetalBuffer &logits;
  U32 vocab;
  std::vector<std::unique_ptr<MetalBuffer>> buffers;
  std::map<std::string, Ref> refs;
  Profile profile;
  std::array<CoupledProfile, 8> coupled_profiles{};
  Profile equality_only_profile;
  SamplingContract contract;
};

TargetGpuVerifier::TargetGpuVerifier(const MetalContext &context,
                                     const MetalBuffer &logits, U32 vocabulary,
                                     const std::filesystem::path &libraries,SamplingContract contract)
    : impl_(std::make_unique<Impl>(context,logits,vocabulary,libraries,contract)) {}
TargetGpuVerifier::~TargetGpuVerifier() = default;

bool TargetGpuVerifier::can_enqueue_select(bool use_icb) const noexcept {
  const auto &p = *impl_;
  return use_icb && p.contract == SamplingContract::PositionCoupledV1 &&
         p.coupled_profiles[0].profile.icb.has_value();
}

TargetGpuPendingSelect
TargetGpuVerifier::enqueue_select(std::span<const SelectorStep> proposals,
                                  const GpuSamplingOptions &sampling,
                                  U64 absolute_position, bool use_icb) {
  auto &p = *impl_;
  if (!can_enqueue_select(use_icb)) {
    TargetGpuPendingSelect out;
    out.sync_result = select(proposals, sampling, absolute_position, use_icb);
    return out;
  }
  const U32 width = U32(proposals.size()) + 1;
  if (sampling.contract != p.contract)
    throw std::invalid_argument("target verification contract mismatch");
  if (width > 8 || absolute_position > UINT64_MAX - width)
    throw std::invalid_argument("position verify width/position overflow");
  TargetGpuPendingSelect out;
  out.rows.resize(width);
  out.width = width;
  out.proposal_count = U32(proposals.size());
  out.use_icb = use_icb;
  std::array<U32, 7> tokens{};
  for (U32 r = 0; r < width; ++r) {
    out.rows[r] = {.seed = sampling.seed,
                   .position = absolute_position + r + 1,
                   .top_k = sampling.top_k,
                   .temperature = sampling.temperature,
                   .top_p = sampling.top_p};
    p.validate_coupled_row(out.rows[r]);
  }
  for (U32 r = 0; r < proposals.size(); ++r) {
    if (proposals[r].token < 0 || U32(proposals[r].token) >= p.vocab)
      throw std::invalid_argument("position proposal token invalid");
    tokens[r] = U32(proposals[r].token);
  }
  const_cast<MetalBuffer *>(p.get("proposal_tokens").buffer)
      ->write(std::as_bytes(std::span(tokens)));
  auto &cp = p.coupled_profiles[width - 1];
  std::memcpy(cp.profile.parameter_data.data() + cp.options_offset,
              out.rows.data(), std::span(out.rows).size_bytes());
  cp.profile.parameters.write("step", cp.profile.parameter_data);
  out.pending = p.context.replay_icb_async(*cp.profile.icb);
  out.async_icb = true;
  return out;
}

TargetGpuVerifyDecision
TargetGpuVerifier::finish_select(TargetGpuPendingSelect &&pending) {
  if (!pending.async_icb) {
    return std::move(pending.sync_result);
  }
  auto &p = *impl_;
  auto sub = p.context.wait_submission(std::move(pending.pending));
  return p.finish_coupled_submission(pending.rows, pending.width,
                                     pending.proposal_count, pending.use_icb,
                                     std::move(sub));
}

TargetGpuPreparedSelect
TargetGpuVerifier::prepare_select(std::span<const SelectorStep> proposals,
                                  const GpuSamplingOptions &sampling,
                                  U64 absolute_position, bool use_icb) {
  auto &p = *impl_;
  const U32 width = U32(proposals.size()) + 1;
  if (sampling.contract != p.contract)
    throw std::invalid_argument("target verification contract mismatch");
  TargetGpuPreparedSelect prep;
  prep.absolute_position = absolute_position;
  prep.width = width;
  prep.proposal_count = U32(proposals.size());
  prep.use_icb = use_icb;
  if (p.contract == SamplingContract::PositionCoupledV1) {
    if (width > 8 || absolute_position > UINT64_MAX - width)
      throw std::invalid_argument("position verify width/position overflow");
    prep.coupled = true;
    prep.rows.resize(width);
    std::array<U32, 7> tokens{};
    for (U32 r = 0; r < width; ++r) {
      prep.rows[r] = {.seed = sampling.seed,
                      .position = absolute_position + r + 1,
                      .top_k = sampling.top_k,
                      .temperature = sampling.temperature,
                      .top_p = sampling.top_p};
      p.validate_coupled_row(prep.rows[r]);
    }
    for (U32 r = 0; r < proposals.size(); ++r) {
      if (proposals[r].token < 0 || U32(proposals[r].token) >= p.vocab)
        throw std::invalid_argument("position proposal token invalid");
      tokens[r] = U32(proposals[r].token);
    }
    const_cast<MetalBuffer *>(p.get("proposal_tokens").buffer)
        ->write(std::as_bytes(std::span(tokens)));
    auto &cp = p.coupled_profiles[width - 1];
    std::memcpy(cp.profile.parameter_data.data() + cp.options_offset,
                prep.rows.data(), std::span(prep.rows).size_bytes());
    cp.profile.parameters.write("step", cp.profile.parameter_data);
    prep.icb = (use_icb && cp.profile.icb) ? &(*cp.profile.icb) : nullptr;
    return prep;
  }
  if (width > max_width || absolute_position > UINT32_MAX ||
      !std::isfinite(sampling.temperature) || sampling.temperature < 0 ||
      !std::isfinite(sampling.top_p) || sampling.top_p < 0 ||
      sampling.top_p > 1 || sampling.top_k > max_topk ||
      (sampling.temperature > 0 && sampling.top_k == 0))
    throw std::invalid_argument("target GPU verification request invalid");
  const U32 topk = sampling.temperature > 0 ? sampling.top_k : 1;
  const U32 proposal_k =
      proposals.empty() ? 0 : U32(proposals.front().candidate_ids.size());
  if (proposal_k > max_proposal_k || (!proposals.empty() && !proposal_k))
    throw std::invalid_argument("target GPU proposal width invalid");
  std::array<U32, 7> proposed{};
  std::array<U32, 7 * max_proposal_k> ids{};
  std::array<float, 7 * max_proposal_k> q{};
  for (U32 row = 0; row < proposals.size(); ++row) {
    const auto &step = proposals[row];
    if (step.token < 0 || U32(step.token) >= p.vocab ||
        step.candidate_ids.size() != proposal_k ||
        step.probabilities.size() != proposal_k)
      throw std::invalid_argument("target GPU proposal geometry mismatch");
    proposed[row] = U32(step.token);
    for (U32 index = 0; index < proposal_k; ++index) {
      if (step.candidate_ids[index] < 0 ||
          U32(step.candidate_ids[index]) >= p.vocab ||
          !std::isfinite(step.probabilities[index]) ||
          step.probabilities[index] < 0)
        throw std::invalid_argument("target GPU proposal candidate invalid");
      ids[row * max_proposal_k + index] = U32(step.candidate_ids[index]);
      q[row * max_proposal_k + index] = step.probabilities[index];
    }
  }
  const auto upload = [&](std::string_view name, auto &values) {
    const auto ref = p.get(name);
    const_cast<MetalBuffer *>(ref.buffer)
        ->write(std::as_bytes(std::span(values)), ref.offset);
  };
  upload("proposal_tokens", proposed);
  upload("proposal_ids", ids);
  upload("proposal_q", q);
  VerifyParams params{.vocab = p.vocab,
                      .topk = topk,
                      .blocks = U32((U64(p.vocab) + chunk - 1) / chunk),
                      .groups =
                          U32((U64(p.vocab) + chunk * 32 - 1) / (chunk * 32)),
                      .width = width,
                      .proposal_k = proposal_k,
                      .position = U32(absolute_position),
                      .temperature = sampling.temperature,
                      .top_p = sampling.top_p};
  for (U32 row = 0; row < proposals.size(); ++row) {
    params.accept_uniform[row] = float(speculative_uniform(
        sampling.seed, absolute_position + row, 0xACCE9701ULL));
    params.correction_uniform[row] = float(speculative_uniform(
        sampling.seed, absolute_position + row, 0xC022EC71ULL));
  }
  params.bonus_uniform = float(speculative_uniform(
      sampling.seed, absolute_position + proposals.size(), 0xB0A05EEDULL));
  auto &profile = p.profile;
  std::memcpy(profile.parameter_data.data() + profile.sampling_offset, &params,
              sizeof(params));
  profile.parameters.write("step", profile.parameter_data);
  prep.legacy_proposals.assign(proposals.begin(), proposals.end());
  prep.legacy_sampling = sampling;
  prep.legacy_topk = topk;
  prep.icb = (use_icb && profile.icb) ? &(*profile.icb) : nullptr;
  return prep;
}

TargetGpuVerifyDecision
TargetGpuVerifier::finish_prepared_select(TargetGpuPreparedSelect &&prep,
                                          bool already_executed,
                                          MetalSubmission batched_submission) {
  auto &p = *impl_;
  const U32 width = prep.width;
  if (prep.coupled) {
    MetalSubmission sub;
    if (already_executed) {
      sub = std::move(batched_submission);
    } else if (prep.icb) {
      sub = p.context.replay_icb(*prep.icb);
    } else {
      auto &cp = p.coupled_profiles[width - 1];
      const std::array calls{
          MetalCommandInvocation{.plan = &cp.profile.plan,
                                 .bindings = cp.profile.bindings,
                                 .parameters = &cp.profile.parameters}};
      sub = p.context.execute_batch(calls);
    }
    return p.finish_coupled_submission(prep.rows, width, prep.proposal_count,
                                       prep.use_icb, std::move(sub));
  }
  TargetGpuVerifyDecision result;
  if (already_executed) {
    result.submission = std::move(batched_submission);
  } else if (prep.icb) {
    result.submission = p.context.replay_icb(*prep.icb);
  } else {
    auto &profile = p.profile;
    const std::array calls = {
        MetalCommandInvocation{.plan = &profile.plan,
                               .bindings = profile.bindings,
                               .parameters = &profile.parameters}};
    result.submission = p.context.execute_batch(calls);
  }
  const auto status = p.get("meta").buffer->read(0, 16);
  result.host_read_bytes += status.size();
  std::array<U32, 4> meta;
  std::memcpy(meta.data(), status.data(), status.size());
  if (!meta[0] || meta[0] > width || meta[1] > prep.proposal_count ||
      meta[2] > 1 || meta[3] > 1)
    throw std::runtime_error("target GPU verification metadata invalid");
  if (meta[3]) {
    result.compact_cpu_fallback = true;
    const auto top_ids =
        p.get("ids").buffer->read(0, U64(width) * max_topk * 4);
    const auto top_scores =
        p.get("scores").buffer->read(0, U64(width) * max_topk * 4);
    result.host_read_bytes += top_ids.size() + top_scores.size();
    std::vector<TargetDistribution> distributions;
    distributions.reserve(width);
    for (U32 row = 0; row < width; ++row) {
      std::array<U32, max_topk> row_ids;
      std::array<float, max_topk> row_scores;
      const auto offset = U64(row) * max_topk * 4;
      std::memcpy(row_ids.data(), top_ids.data() + offset,
                  prep.legacy_topk * 4);
      std::memcpy(row_scores.data(), top_scores.data() + offset,
                  prep.legacy_topk * 4);
      distributions.push_back(target_distribution_from_topk(
          std::span(row_ids).first(prep.legacy_topk),
          std::span(row_scores).first(prep.legacy_topk), p.vocab,
          prep.legacy_sampling.temperature, prep.legacy_sampling.top_p));
    }
    result.verification = verify_speculative(
        prep.legacy_proposals, distributions, prep.legacy_sampling.temperature,
        prep.legacy_sampling.seed, prep.absolute_position);
  } else {
    const auto data = p.get("licensed").buffer->read(0, U64(meta[0]) * 4);
    result.host_read_bytes += data.size();
    result.verification.licensed_outputs.resize(meta[0]);
    std::memcpy(result.verification.licensed_outputs.data(), data.data(),
                data.size());
    result.verification.accepted_proposals = meta[1];
    result.verification.used_correction = meta[2] != 0;
    for (auto token : result.verification.licensed_outputs)
      if (token < 0 || U32(token) >= p.vocab)
        throw std::runtime_error("target GPU licensed token invalid");
  }
  return result;
}

TargetGpuVerifyDecision
TargetGpuVerifier::select(std::span<const SelectorStep> proposals,
                          const GpuSamplingOptions &sampling,
                          U64 absolute_position, bool use_icb) {
  auto prep = prepare_select(proposals, sampling, absolute_position, use_icb);
  return finish_prepared_select(std::move(prep), false);
}

} // namespace omlx::m5
