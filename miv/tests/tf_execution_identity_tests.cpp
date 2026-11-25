#include "omlx_m5/execution_contract.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace omlx::m5;
void require(bool b,const char *m){if(!b)throw std::runtime_error(m);}
template<class F> void rejects(F operation){bool failed=false;try{operation();}catch(const std::invalid_argument&){failed=true;}require(failed,"malformed identity accepted");}
int main(int argc,char **argv){try{
  const ExecutionIdentityFields a{{"model","a|b"},{"draft","c"}},b{{"model","a"},{"draft","b|c"}};
  require(canonical_execution_identity(a)!=canonical_execution_identity(b),"delimiter collision");
  require(execution_identity_digest(a)!=execution_identity_digest(b),"digest collision");
  const ExecutionIdentityFields order{{"z","3"},{"a","1"},{"m","2"}};
  require(canonical_execution_identity(order)=="omlx.execution.identity.v1|1:a1:11:m1:21:z1:3","canonical golden differs");
  require(execution_identity_digest(order).size()==64,"digest not SHA256");
  for(const auto *key:{"model","draft","numeric","sampling","chunk","kernel","source","layout"}){
    auto one=order,two=order;one[key]="v1";two[key]="v2";
    require(execution_identity_digest(one)!=execution_identity_digest(two),"contract mutation failed to invalidate");
  }
  rejects([]{(void)canonical_execution_identity({});});rejects([]{(void)canonical_execution_identity({{"","x"}});});
  rejects([]{(void)canonical_execution_identity({{"x",""}});});
  rejects([]{(void)prefill_next_chunk(3,3,SamplingContract::PositionCoupledV1);});
  rejects([]{(void)prefill_chunk_policy_id(static_cast<SamplingContract>(999));});
  require(!prefill_checkpoint_compatible(63,65,SamplingContract::PositionCoupledV1),"old tail accepted for expanded exact prompt");
  require(prefill_checkpoint_compatible(63,63,SamplingContract::PositionCoupledV1),"exact hit rejected");
  require(prefill_checkpoint_compatible(1024,1025,SamplingContract::PositionCoupledV1),"legal chunk checkpoint rejected");
  require(prefill_checkpoint_compatible(63,65,SamplingContract::LegacyRejectionV1),"legacy tail semantics changed");
  for(std::size_t total:{1,63,64,65,1023,1024,1025,2047,2048,2049}){
    std::size_t position=0;while(position<total){auto count=prefill_next_chunk(position,total,SamplingContract::PositionCoupledV1);
      require(count && count<=1024 && position%1024+count<=1024,"absolute chunk crossed boundary");position+=count;}
    require(position==total,"chunk plan lost rows");
  }
  auto directory=std::filesystem::temp_directory_path()/("omlx-contract-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(directory);
  {std::ofstream(directory/"a.metallib",std::ios::binary)<<"one";}
  const auto old=execution_kernel_bundle_digest(directory);
  {std::ofstream(directory/"a.metallib",std::ios::binary)<<"two";}
  require(old!=execution_kernel_bundle_digest(directory),"kernel content change not detected");
  std::filesystem::rename(directory/"a.metallib",directory/"b.metallib");
  {std::ofstream(directory/"a.metallib",std::ios::binary)<<"one";}
  require(old!=execution_kernel_bundle_digest(directory),"kernel membership change not detected");
  std::filesystem::remove_all(directory);
  require(execution_source_id().size()==64,"build source ID missing");
  if(argc!=2)throw std::invalid_argument("execution identity tests require fixtures");
  auto first=ArtifactModel::open(std::filesystem::path(argv[1])/"affine/model.omlxm5",true);
  auto second=ArtifactModel::open(std::filesystem::path(argv[1])/"mxfp4/model.omlxm5",true);
  require(execution_artifact_digest(first)!=execution_artifact_digest(second),"actual weight/format identity collision");
  auto unchecked=ArtifactModel::open(std::filesystem::path(argv[1])/"affine/model.omlxm5",false);
  rejects([&]{(void)execution_artifact_digest(unchecked);});
  std::cout<<"{\"schema\":\"omlx-m5-execution-identity-test-v1\",\"canonical_golden\":true,\"collision_mutation_negative_checks\":true,\"chunk_policy_checked\":true,\"kernel_invalidation_checked\":true,\"passed\":true}\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
