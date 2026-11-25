// SDK27 diagnostic: consume two FP4/E8M0 multi-plane tensors directly.
// No BF16 decode tile and no production route promotion.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
using Weight = tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>,
                      tensor_handle, tensor_blockwise<tensor_plane_scales,
                                       device metal_fp8_ue8m0_format, 32, 1>>;
kernel void omlx_m5_p11_fp4_a4_native(
    Weight A [[buffer(0)]],
    Weight B [[buffer(1)]],
    tensor<device float, dextents<int32_t, 2>> C [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]]) {
  constexpr matmul2d_descriptor desc(16, 32, int(dynamic_extent), false, true);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto left = A.slice(0, group.y * 16);
  auto right = B.slice(0, group.x * 32);
  auto output = C.slice(group.x * 32, group.y * 16);
  op.run(left, right, output);
}
