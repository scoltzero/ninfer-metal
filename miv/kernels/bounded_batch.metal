#include <metal_stdlib>
using namespace metal;
struct BatchLayout {uint count,total,columns,capacity;uint slots[8],widths[8],offsets[8],valid[8];};
kernel void omlx_m5_gather_slot_logits(
 const device ushort *x0 [[buffer(0)]],
 const device ushort *x1 [[buffer(1)]],
 const device ushort *x2 [[buffer(2)]],
 const device ushort *x3 [[buffer(3)]],
 const device ushort *x4 [[buffer(4)]],
 const device ushort *x5 [[buffer(5)]],
 const device ushort *x6 [[buffer(6)]],
 const device ushort *x7 [[buffer(7)]],
 device ushort *y [[buffer(8)]],constant BatchLayout &p [[buffer(9)]],uint index [[thread_position_in_grid]]) {
 uint row=index/p.columns,col=index%p.columns;if(row>=p.count)return;
 uint member=row,local=0;
 if(!p.valid[member]){y[index]=0;return;}
 uint offset=local*p.columns+col;ushort value=0;switch(p.slots[member]){
 case 0:value=x0[offset];break;
 case 1:value=x1[offset];break;
 case 2:value=x2[offset];break;
 case 3:value=x3[offset];break;
 case 4:value=x4[offset];break;
 case 5:value=x5[offset];break;
 case 6:value=x6[offset];break;
 case 7:value=x7[offset];break;
 }y[index]=value;
}
kernel void omlx_m5_gather_projection_rows(
 const device ushort *x0 [[buffer(0)]],
 const device ushort *x1 [[buffer(1)]],
 const device ushort *x2 [[buffer(2)]],
 const device ushort *x3 [[buffer(3)]],
 const device ushort *x4 [[buffer(4)]],
 const device ushort *x5 [[buffer(5)]],
 const device ushort *x6 [[buffer(6)]],
 const device ushort *x7 [[buffer(7)]],
 device ushort *y [[buffer(8)]],constant BatchLayout &p [[buffer(9)]],uint index [[thread_position_in_grid]]) {
 uint row=index/p.columns,col=index%p.columns;if(row>=p.total)return;
 uint member=row,local=0;
 member=0;while(member<p.count && !(p.valid[member] && row>=p.offsets[member] && row<p.offsets[member]+p.widths[member]))++member;if(member==p.count)return;local=row-p.offsets[member];
 uint offset=local*p.columns+col;ushort value=0;switch(p.slots[member]){
 case 0:value=x0[offset];break;
 case 1:value=x1[offset];break;
 case 2:value=x2[offset];break;
 case 3:value=x3[offset];break;
 case 4:value=x4[offset];break;
 case 5:value=x5[offset];break;
 case 6:value=x6[offset];break;
 case 7:value=x7[offset];break;
 }y[index]=value;
}
kernel void omlx_m5_scatter_projection_rows(const device ushort *x [[buffer(0)]],
 device ushort *y0 [[buffer(1)]],
 device ushort *y1 [[buffer(2)]],
 device ushort *y2 [[buffer(3)]],
 device ushort *y3 [[buffer(4)]],
 device ushort *y4 [[buffer(5)]],
 device ushort *y5 [[buffer(6)]],
 device ushort *y6 [[buffer(7)]],
 device ushort *y7 [[buffer(8)]],
 device ushort *last0 [[buffer(9)]],
 device ushort *last1 [[buffer(10)]],
 device ushort *last2 [[buffer(11)]],
 device ushort *last3 [[buffer(12)]],
 device ushort *last4 [[buffer(13)]],
 device ushort *last5 [[buffer(14)]],
 device ushort *last6 [[buffer(15)]],
 device ushort *last7 [[buffer(16)]],
 constant BatchLayout &p [[buffer(17)]],uint index [[thread_position_in_grid]]) {
 uint row=index/p.columns,col=index%p.columns;if(row>=p.total)return;uint member=0;while(member<p.count && !(p.valid[member] && row>=p.offsets[member] && row<p.offsets[member]+p.widths[member]))++member;if(member==p.count)return;uint local=row-p.offsets[member];uint offset=local*p.columns+col;ushort value=x[index];switch(p.slots[member]){
 case 0:y0[offset]=value;if(local+1==p.widths[member])last0[col]=value;break;
 case 1:y1[offset]=value;if(local+1==p.widths[member])last1[col]=value;break;
 case 2:y2[offset]=value;if(local+1==p.widths[member])last2[col]=value;break;
 case 3:y3[offset]=value;if(local+1==p.widths[member])last3[col]=value;break;
 case 4:y4[offset]=value;if(local+1==p.widths[member])last4[col]=value;break;
 case 5:y5[offset]=value;if(local+1==p.widths[member])last5[col]=value;break;
 case 6:y6[offset]=value;if(local+1==p.widths[member])last6[col]=value;break;
 case 7:y7[offset]=value;if(local+1==p.widths[member])last7[col]=value;break;
 }
}
