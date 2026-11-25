#pragma once
#include "omlx_m5/types.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <utility>
namespace omlx::m5::detail {
class RoundCostRecorder final {
public:
 enum Phase : std::size_t {draft,verify,sampling,replay,publish};
 using Clock=std::chrono::steady_clock;
 RoundCostRecorder(StatsReport &stats,std::mutex &mutex,std::uint32_t depth)
  :stats_(stats),mutex_(mutex),depth_(depth),start_(Clock::now()) {
   if(depth>7)throw std::invalid_argument("round cost depth outside D0-7");
 }
 RoundCostRecorder(const RoundCostRecorder&)=delete;
 RoundCostRecorder &operator=(const RoundCostRecorder&)=delete;
 ~RoundCostRecorder(){if(!finished_)finish(committed_,false);}
 template<class Function>decltype(auto) measure(Phase phase,Function &&function){
  struct Timer{double &value;Clock::time_point begin=Clock::now();~Timer(){value+=milliseconds(begin);}} timer{phases_[phase]};
  return std::forward<Function>(function)();
 }
 void add_draft(double milliseconds){phases_[draft]+=milliseconds;}
 void add_verify(double milliseconds){phases_[verify]+=milliseconds;}
 void add_sampling(double milliseconds){phases_[sampling]+=milliseconds;}
 void add_select(double milliseconds){phases_[sampling]+=milliseconds;}
 void add_replay(double milliseconds){phases_[replay]+=milliseconds;}
 void mark_committed(std::uint32_t count){committed_=count;}
 [[nodiscard]] std::uint32_t depth() const noexcept { return depth_; }
 [[nodiscard]] double phase_ms(Phase phase) const noexcept { return phases_[phase]; }
 [[nodiscard]] double wall_ms() const noexcept { return last_wall_ms_ > 0.0 ? last_wall_ms_ : milliseconds(start_); }
 void finish(std::uint32_t committed,bool completed=true){
  if(finished_)return;finished_=true;
  const auto wall=milliseconds(start_);last_wall_ms_=wall;
  const auto sum=std::accumulate(phases_.begin(),phases_.end(),0.0);
  std::scoped_lock lock(mutex_);
  if(completed)++stats_.spec_cost_completed_rounds;else ++stats_.spec_cost_aborted_rounds;
  stats_.spec_cost_committed_tokens+=committed;stats_.spec_cost_round_wall_ms+=wall;
  ++stats_.spec_cost_depth_rounds.at(depth_);stats_.spec_cost_depth_committed_tokens.at(depth_)+=committed;
  stats_.spec_cost_depth_wall_ms.at(depth_)+=wall;
  for(std::size_t i=0;i<phases_.size();++i)stats_.spec_cost_phase_wall_ms.at(i)+=phases_[i];
  stats_.spec_cost_phase_wall_ms.at(5)+=std::max(0.0,wall-sum);
 }
 static double milliseconds(Clock::time_point begin){return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();}
private:
 StatsReport &stats_;std::mutex &mutex_;std::uint32_t depth_;Clock::time_point start_;
 std::array<double,5> phases_{};double last_wall_ms_=0.0;std::uint32_t committed_=0;bool finished_=false;
};
} // namespace omlx::m5::detail
