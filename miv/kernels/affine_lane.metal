// Tensor path adapted from TensorFold lane_qmm.py _MAIN/_MAIN_TILED at cb2ebf05.
// Copyright 2026 TensorFold contributors. Notices/licenses retained under
// m5rt/third_party/tensorfold/. Changes: native parameter ABI, padded-N prepared
// buffers, per-element tails and physical-device guard in the host installer.
// The portable scalar reference below is separately implemented here.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct AffineLaneParams {uint n,np,k,rows,mp,sk;};

kernel void omlx_m5_affine_lane_xsum(
    device const bfloat *x [[buffer(0)]], device float *xs [[buffer(1)]],
    constant AffineLaneParams &p [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint g=at.x,r=at.y;
  if(g>=p.k/64 || r>=p.mp)return;
  float sum=0.f;
  if(r<p.rows)for(uint i=0;i<64;++i)sum+=float(x[r*p.k+g*64+i]);
  xs[g*p.mp+r]=sum;
}

// Same prepared bytes and ordered group/slice formula, without assuming the
// M5 cooperative fragment layout. This is not a NAX fallback or speed claim.
kernel void omlx_m5_affine_lane_portable(
    device const bfloat *x [[buffer(0)]], device const uchar *codes [[buffer(1)]],
    device const uint *sb [[buffer(2)]], device const float *xs [[buffer(3)]],
    device bfloat *y [[buffer(4)]], constant AffineLaneParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint n=tg.x*4+sg,r=tg.y,kg=p.k/64;
  if(n>=p.n || r>=p.rows)return;
  float total=0.f;
  for(uint slice=0;slice<p.sk;++slice) {
    float partial=0.f;
    for(uint g=slice*kg/p.sk;g<(slice+1)*kg/p.sk;++g) {
      const ulong offset=(ulong(n/32)*kg+g)*1024+ulong(n%32)*32;
      float dot=0.f;
      for(uint i=lane;i<64;i+=32) {
        const uint code=(uint(codes[offset+i/2])>>((i&1)*4))&15;
        dot=fma(float(x[r*p.k+g*64+i]),float(code),dot);
      }
      dot=simd_sum(dot);
      if(lane==0) {
        const vec<bfloat,2> pair=as_type<vec<bfloat,2>>(sb[g*p.np+n]);
        partial=fma(float(pair[0]),dot,fma(float(pair[1]),xs[g*p.mp+r],partial));
      }
    }
    if(lane==0)total+=partial;
  }
  if(lane==0)y[r*p.n+n]=bfloat(total);
}

template<uint SK,uint TMR>
inline void affine_lane_tensor(
    device const bfloat *x,device const uchar *codes,device const uint4 *sb,
    device const float *xs,device bfloat *y,constant AffineLaneParams &p,
    threadgroup float *parts,uint lane,uint sg,uint3 tg) {
  constexpr uint NT=32,GS=64,NF=2;
  const uint rb=tg.y*(16*TMR),n0=tg.x*NT,kg=p.k/GS;
  if(rb>=p.rows || n0>=p.np)return;
  const short qid=short(lane>>2);
  const short fm=(qid&4)|((short(lane)>>1)&3);
  const short fn=((qid&2)|(short(lane)&1))*4;
  constexpr auto desc=matmul2d_descriptor(16*TMR,NT,GS,false,true,false,
                                        matmul2d_descriptor::mode::multiply);
  matmul2d<desc,execution_simdgroup> op;
  tensor<device bfloat,dextents<int32_t,2>,tensor_inline> a_parent(
      (device bfloat *)x+ulong(rb)*p.k,dextents<int32_t,2>(p.k,p.rows-rb));
  float acc[TMR][NF*8];
  for(uint t=0;t<TMR;++t)for(uint i=0;i<NF*8;++i)acc[t][i]=0.f;
  for(uint g=sg*kg/SK;g<(sg+1)*kg/SK;++g) {
    float scale[NF][4],bias[NF][4];
    for(uint f=0;f<NF;++f) {
      const vec<bfloat,8> pair=as_type<vec<bfloat,8>>(sb[(g*p.np+n0+f*16+fn)/4]);
      for(uint j=0;j<4;++j){scale[f][j]=float(pair[j*2]);bias[f][j]=float(pair[j*2+1]);}
    }
    auto a=a_parent.slice(g*GS,0);
    tensor<device uint4b_format,dextents<int32_t,2>,tensor_inline> b(
        (device uchar *)codes+(ulong(tg.x)*kg+g)*(NT*GS/2),dextents<int32_t,2>(GS,NT));
    auto product=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    op.run(a,b,product);
    for(uint t=0;t<TMR;++t)for(uint f=0;f<NF;++f)for(uint r=0;r<2;++r) {
      const uint row=rb+t*16+fm+r*8;
      const float sum=row<p.mp?xs[g*p.mp+row]:0.f;
      for(uint j=0;j<4;++j) {
        const uint i=f*8+r*4+j;
        acc[t][i]=fma(scale[f][j],product[t*NF*8+i],fma(bias[f][j],sum,acc[t][i]));
      }
    }
  }
  for(uint t=0;t<TMR;++t) {
    if(SK>1) {
      if(sg>0)for(uint i=0;i<NF*8;++i)parts[((sg-1)*NF*8+i)*32+lane]=acc[t][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(sg==0)for(uint s=1;s<SK;++s)for(uint i=0;i<NF*8;++i)
        acc[t][i]+=parts[((s-1)*NF*8+i)*32+lane];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(sg==0)for(uint f=0;f<NF;++f)for(uint r=0;r<2;++r) {
      const uint row=rb+t*16+fm+8*r,n=n0+f*16+fn;
      if(row<p.rows)for(uint j=0;j<4;++j)if(n+j<p.n)
        y[row*p.n+n+j]=bfloat(acc[t][f*8+r*4+j]);
    }
  }
}

#define LANE_ENTRY(sk,rows,tmr)                                               \
kernel void omlx_m5_affine_lane_sk##sk##_r##rows(                              \
    device const bfloat *x [[buffer(0)]],device const uchar *w [[buffer(1)]],   \
    device const uint4 *sb [[buffer(2)]],device const float *xs [[buffer(3)]], \
    device bfloat *y [[buffer(4)]],constant AffineLaneParams &p [[buffer(5)]],  \
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]], \
    uint3 tg [[threadgroup_position_in_grid]]) {                             \
  threadgroup float parts[(sk>1?sk-1:1)*16*32];                              \
  affine_lane_tensor<sk,tmr>(x,w,sb,xs,y,p,parts,lane,sg,tg);                   \
}
LANE_ENTRY(1,16,1)
LANE_ENTRY(2,16,1)
LANE_ENTRY(4,16,1)
LANE_ENTRY(8,16,1)
LANE_ENTRY(1,32,2)
LANE_ENTRY(2,32,2)
LANE_ENTRY(4,32,2)
LANE_ENTRY(8,32,2)
