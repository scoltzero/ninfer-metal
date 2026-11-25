#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/p11_tensor_view.h"

#ifndef OMLX_M5_P11_METALLIB_PATH
#define OMLX_M5_P11_METALLIB_PATH ""
#endif

namespace {
using Json = nlohmann::json;
std::string text(NSString *value) {
  return value ? std::string(value.UTF8String) : "";
}
std::string error_text(NSError *error) {
  return error ? text(error.localizedDescription) : "no NSError detail";
}
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 270000
MTLTensorDescriptor *fp4_descriptor(std::uint32_t rows, std::uint32_t k) {
  MTLTensorDescriptor *descriptor = [[MTLTensorDescriptor alloc] init];
  const NSInteger dimensions[] = {NSInteger(k), NSInteger(rows)};
  const NSInteger strides[] = {1, NSInteger(k)};
  const NSInteger blocks[] = {32, 1};
  descriptor.dimensions =
      [[MTLTensorExtents alloc] initWithRank:2 values:dimensions];
  descriptor.strides = [[MTLTensorExtents alloc] initWithRank:2 values:strides];
  descriptor.dataType = MTLTensorDataTypeMetalFloat4E2M1;
  descriptor.usage = MTLTensorUsageCompute;
  MTLTensorAuxiliaryPlaneDescriptor *scale =
      [[MTLTensorAuxiliaryPlaneDescriptor alloc] init];
  scale.dataType = MTLTensorDataTypeMetalFloat8UE8M0;
  scale.blockFactors =
      [[MTLTensorExtents alloc] initWithRank:2 values:blocks];
  MTLTensorAuxiliaryPlaneDescriptorMap *planes =
      [[MTLTensorAuxiliaryPlaneDescriptorMap alloc] init];
  [planes setDescriptor:scale forPlane:MTLTensorPlaneTypeScales];
  descriptor.auxiliaryPlanes = planes;
  return descriptor;
}
id<MTLTensor> fp4_view(id<MTLDevice> device, id<MTLBuffer> codes,
                       id<MTLBuffer> scales, std::uint32_t rows,
                       std::uint32_t k, NSError **error) {
  auto *descriptor = fp4_descriptor(rows, k);
  MTLTensorBufferAttachments *attachments =
      [[MTLTensorBufferAttachments alloc] init];
  [attachments setBuffer:codes offset:0 forPlane:MTLTensorPlaneTypeData];
  [attachments setBuffer:scales offset:0 forPlane:MTLTensorPlaneTypeScales];
  return [device newTensorWithDescriptor:descriptor
                              attachments:attachments error:error];
}
#endif
Json probe(std::string_view metallib, std::uint32_t n,
           std::uint32_t k) {
  Json result = {{"schema", "omlx-m5-p11-fp4-native-canary-v1"},
                 {"sdk_fp4_symbols", __MAC_OS_X_VERSION_MAX_ALLOWED >= 270000},
                 {"descriptor_created", false},
                 {"multi_plane_view_created", false},
                 {"pipeline_compiled", false},
                 {"runtime_executed", false},
                 {"zero_checked", false},
                 {"scale_sign_checked", false},
                 {"actual_nax_verified", false},
                 {"production_route_changed", false}};
  if (n < 32 || n > 16384 || k < 256 || k > 32768 || k % 256)
    throw std::invalid_argument("P11 canary needs N=32..16384, K=256..32768/256");
  result["n"] = n;
  result["k"] = k;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  result["device"] = device ? text(device.name) : "unavailable";
  if (device == nil) {
    result["reason"] = "no Metal device";
    return result;
  }
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 270000
  if (@available(macOS 27.0, *)) {
    constexpr std::uint32_t m = 16;
    std::vector<std::uint8_t> a(m * k / 2), as(m * k / 32, 127);
    std::vector<std::uint8_t> b(n * k / 2), bs(n * k / 32, 127);
    const auto make = [&](std::span<const std::uint8_t> bytes) {
      return [device newBufferWithBytes:bytes.data()
                                 length:bytes.size()
                                options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> a_codes = make(a), a_scales = make(as),
                  b_codes = make(b), b_scales = make(bs);
    id<MTLBuffer> output = [device newBufferWithLength:m * n * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    if (!a_codes || !a_scales || !b_codes || !b_scales || !output) {
      result["reason"] = "canary buffer allocation failed";
      return result;
    }
    auto *descriptor = fp4_descriptor(n, k);
    result["descriptor_created"] = descriptor != nil;
    // tensorSizeAndAlignWithDescriptor rejects multi-plane descriptors under
    // Metal Validation. Bounds are checked per attachment instead.
    result["data_plane_bytes"] = b_codes.length;
    result["scale_plane_bytes"] = b_scales.length;
    NSError *error = nil;
    id<MTLTensor> a_tensor = fp4_view(device, a_codes, a_scales, m, k, &error);
    if (!a_tensor) {
      result["reason"] = "A view: " + error_text(error);
      return result;
    }
    error = nil;
    id<MTLTensor> b_tensor = fp4_view(device, b_codes, b_scales, n, k, &error);
    if (!b_tensor) {
      result["reason"] = "B view: " + error_text(error);
      return result;
    }
    MTLTensorDescriptor *out_descriptor = [[MTLTensorDescriptor alloc] init];
    const NSInteger out_dimensions[] = {NSInteger(n), NSInteger(m)};
    const NSInteger out_strides[] = {1, NSInteger(n)};
    out_descriptor.dimensions =
        [[MTLTensorExtents alloc] initWithRank:2 values:out_dimensions];
    out_descriptor.strides =
        [[MTLTensorExtents alloc] initWithRank:2 values:out_strides];
    out_descriptor.dataType = MTLTensorDataTypeFloat32;
    out_descriptor.usage = MTLTensorUsageCompute;
    error = nil;
    id<MTLTensor> c_tensor =
        [output newTensorWithDescriptor:out_descriptor offset:0 error:&error];
    if (!c_tensor) {
      result["reason"] = "C view: " + error_text(error);
      return result;
    }
    result["multi_plane_view_created"] = true;
    result["backing_planes_match"] =
        b_tensor.buffer == b_codes && b_tensor.bufferOffset == 0 &&
        b_tensor.auxiliaryPlanes.count == 1 &&
        b_tensor.auxiliaryPlanes.firstObject.buffer == b_scales &&
        b_tensor.auxiliaryPlanes.firstObject.bufferOffset == 0;
    result["block_factor_k"] =
        [b_tensor.auxiliaryPlanes.firstObject.blockFactors extentAtDimensionIndex:0];
    if (!result["backing_planes_match"].get<bool>() ||
        result["block_factor_k"].get<int>() != 32) {
      result["reason"] = "tensor backing/block metadata mismatch";
      return result;
    }
    error = nil;
    NSString *path = [NSString stringWithUTF8String:std::string(metallib).c_str()];
    id<MTLLibrary> library =
        [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    if (!library) {
      result["reason"] = "library: " + error_text(error);
      return result;
    }
    MTL4CompilerDescriptor *compiler_descriptor =
        [[MTL4CompilerDescriptor alloc] init];
    id<MTL4Compiler> compiler =
        [device newCompilerWithDescriptor:compiler_descriptor error:&error];
    if (!compiler) {
      result["reason"] = "compiler: " + error_text(error);
      return result;
    }
    MTL4LibraryFunctionDescriptor *function =
        [[MTL4LibraryFunctionDescriptor alloc] init];
    function.name = @"omlx_m5_p11_fp4_a4_native";
    function.library = library;
    MTL4ComputePipelineDescriptor *pipeline_descriptor =
        [[MTL4ComputePipelineDescriptor alloc] init];
    pipeline_descriptor.computeFunctionDescriptor = function;
    id<MTLComputePipelineState> pipeline =
        [compiler newComputePipelineStateWithDescriptor:pipeline_descriptor
                                     compilerTaskOptions:nil error:&error];
    if (!pipeline) {
      result["reason"] = "pipeline: " + error_text(error);
      return result;
    }
    result["pipeline_compiled"] = true;
    MTL4ArgumentTableDescriptor *table_descriptor =
        [[MTL4ArgumentTableDescriptor alloc] init];
    table_descriptor.maxBufferBindCount = 3;
    table_descriptor.initializeBindings = YES;
    id<MTL4ArgumentTable> table =
        [device newArgumentTableWithDescriptor:table_descriptor error:&error];
    if (!table) {
      result["reason"] = "argument table: " + error_text(error);
      return result;
    }
    [table setResource:a_tensor.gpuResourceID atBufferIndex:0];
    [table setResource:b_tensor.gpuResourceID atBufferIndex:1];
    [table setResource:c_tensor.gpuResourceID atBufferIndex:2];
    MTLResidencySetDescriptor *residency_descriptor =
        [[MTLResidencySetDescriptor alloc] init];
    id<MTLResidencySet> residency =
        [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    if (!residency) {
      result["reason"] = "residency: " + error_text(error);
      return result;
    }
    for (id<MTLAllocation> allocation in
         @[a_codes, a_scales, b_codes, b_scales, output,
           a_tensor, b_tensor, c_tensor, pipeline])
      [residency addAllocation:allocation];
    [residency commit];
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    if (!queue || !allocator) {
      result["reason"] = "Metal 4 queue/allocator unavailable";
      return result;
    }
    const auto run = [&]() -> std::string {
      id<MTL4CommandBuffer> command = [device newCommandBuffer];
      [command beginCommandBufferWithAllocator:allocator];
      [command useResidencySet:residency];
      id<MTL4ComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setArgumentTable:table];
      [encoder dispatchThreadgroups:MTLSizeMake((n + 31) / 32, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder endEncoding];
      [command endCommandBuffer];
      dispatch_semaphore_t done = dispatch_semaphore_create(0);
      __block NSError *feedback_error = nil;
      MTL4CommitOptions *options = [[MTL4CommitOptions alloc] init];
      [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        feedback_error = feedback.error;
        dispatch_semaphore_signal(done);
      }];
      const id<MTL4CommandBuffer> commands[] = {command};
      [queue commit:commands count:1 options:options];
      if (dispatch_semaphore_wait(done,
              dispatch_time(DISPATCH_TIME_NOW, 10LL * NSEC_PER_SEC)))
        return "command timeout";
      return error_text(feedback_error);
    };
    std::memset(output.contents, 0, output.length);
    auto runtime_error = run();
    if (runtime_error != "no NSError detail") {
      result["reason"] = "zero dispatch: " + runtime_error;
      return result;
    }
    result["runtime_executed"] = true;
    auto *values = static_cast<const float *>(output.contents);
    bool zero_ok = true;
    for (std::size_t i = 0; i < m * n; ++i) zero_ok &= values[i] == 0.0f;
    result["zero_checked"] = zero_ok;
    if (!zero_ok) {
      result["reason"] = "zero output mismatch";
      return result;
    }
    std::memset(a_codes.contents, 0x22, a_codes.length);
    std::memset(b_codes.contents, 0x11, b_codes.length);
    auto *b_scale = static_cast<std::uint8_t *>(b_scales.contents);
    for (std::size_t row = 0; row < n; ++row)
      std::memset(b_scale + row * (k / 32), row % 2 ? 127 : 128, k / 32);
    const auto check = [&](bool negative) {
      std::memset(output.contents, 0, output.length);
      const auto issue = run();
      if (issue != "no NSError detail") return false;
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t token = 0; token < m; ++token) {
          const float expected = float(row % 2 ? k / 2 : k) *
                                 (negative ? -1.0f : 1.0f);
          if (std::abs(values[token * n + row] - expected) > 0.01f)
            return false;
        }
      return true;
    };
    const bool positive = check(false);
    std::memset(b_codes.contents, 0x99, b_codes.length);
    result["scale_sign_checked"] = positive && check(true);
    if (!result["scale_sign_checked"].get<bool>())
      result["reason"] = "nonzero scale/sign output mismatch";
    return result;
  }
#endif
  result["reason"] = "SDK/runtime FP4 tensor API unavailable";
  return result;
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 2 && argc != 3 && argc != 5)
      throw std::invalid_argument(
          "p11_tensor_canary METALLIB [OUTPUT [N K]]");
    const auto n = argc == 5 ? std::uint32_t(std::stoul(argv[3])) : 32U;
    const auto k = argc == 5 ? std::uint32_t(std::stoul(argv[4])) : 256U;
    auto result = probe(argv[1], n, k);
    if (argc >= 3) {
      std::ofstream output(argv[2]);
      output << result.dump(2) << '\n';
      if (!output) throw std::runtime_error("P11 JSON write failed");
    }
    std::cout << result.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "P11 canary: " << error.what() << '\n';
    return 1;
  }
}
