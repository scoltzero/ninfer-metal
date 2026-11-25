#include "../src/forward_plan.h"
#include "omlx_m5/artifact.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/target_verify_gpu.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <nlohmann/json.hpp>

using namespace omlx::m5;
using namespace omlx::m5::detail::forward;
using Json = nlohmann::json;
namespace {
void require(bool b, std::string_view s) {
  if (!b) throw std::runtime_error(std::string(s));
}
float value(std::uint16_t b) { return std::bit_cast<float>(U32(b) << 16); }
std::vector<std::uint16_t> read(const MetalBuffer &b, U64 count) {
  auto bytes = b.read(0, count * 2);
  std::vector<std::uint16_t> out(count);
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}
struct Weight {
  MaterializedWeightView view;
  std::vector<std::unique_ptr<MetalBuffer>> storage;
  void add(const MetalContext &ctx, std::string kind, std::vector<std::byte> data) {
    storage.push_back(std::make_unique<MetalBuffer>(ctx.upload(data)));
    view.planes.push_back({.kind = std::move(kind), .buffer = storage.back().get(),
                          .bytes = data.size(), .row_stride = data.size() / view.row_count});
  }
};
Weight synthetic(const MetalContext &ctx, U32 bits, U32 group, bool fp4, bool bf16) {
  Weight w;
  const U32 n = 19, k = 256, kp = bf16 ? k : k + 128;
  w.view.name = "tf.synthetic.weight";
  w.view.format = bf16 ? "direct_bf16_v1" : fp4 ? "mxfp4_e2m1_g32_e8m0_v1"
      : "oq_affine_q" + std::to_string(bits) + "_g" + std::to_string(group) + "_bf16_v1";
  w.view.parent_shape = {n, k}; w.view.row_count = n;
  w.view.in_features = k; w.view.padded_k = kp;
  if (bf16) {
    std::vector<std::uint16_t> data(U64(n) * k);
    for (U64 i = 0; i < data.size(); ++i)
      data[i] = float_to_bf16(float(int((i * 17) % 251) - 125) / 128);
    auto bytes = std::as_bytes(std::span(data));
    w.add(ctx, "data", {bytes.begin(), bytes.end()}); return w;
  }
  const U32 stride = bits == 8 && !fp4 ? kp : kp / 2;
  const U32 hs = bits == 5 ? kp / 8 : bits == 6 ? kp / 4 : 0;
  std::vector<std::byte> base(U64(n) * stride), high(U64(n) * hs);
  for (U32 row = 0; row < n; ++row) for (U32 col = 0; col < k; ++col) {
    const U32 q = (row * 5 + col * 7 + (col >> 5) * 3) & ((1U << bits) - 1);
    if (bits == 8) base[U64(row) * stride + col] = std::byte(q);
    else base[U64(row) * stride + col / 2] |= std::byte((q & 15) << ((col & 1) * 4));
    if (bits == 5) high[U64(row) * hs + col / 8] |= std::byte((q >> 4) << (col & 7));
    if (bits == 6) high[U64(row) * hs + col / 4] |= std::byte((q >> 4) << ((col & 3) * 2));
  }
  w.add(ctx, fp4 ? "data" : bits == 8 ? "code" : "base_code", std::move(base));
  if (hs) w.add(ctx, bits == 5 ? "high_bit" : "high_2bit", std::move(high));
  if (fp4) {
    std::vector<std::byte> s(U64(n) * (kp / 32));
    for (U64 i = 0; i < s.size(); ++i) s[i] = std::byte(117 + i % 13);
    w.add(ctx, "scale", std::move(s));
  } else {
    std::vector<std::uint16_t> s(U64(n) * (kp / group)), b(s.size());
    for (U64 i = 0; i < s.size(); ++i) {
      const float scale = .006f * (1 + i % 11);
      s[i] = float_to_bf16(scale); b[i] = float_to_bf16(-scale * ((1U << bits) - 1) / 2);
    }
    auto sb = std::as_bytes(std::span(s)), bb = std::as_bytes(std::span(b));
    w.add(ctx, "scale", {sb.begin(), sb.end()}); w.add(ctx, "bias", {bb.begin(), bb.end()});
  }
  return w;
}
Json audit(const MetalContext &ctx, const MaterializedWeightView &w,
           const std::filesystem::path &libs, std::string prefix) {
  const U32 n = U32(w.row_count), k = U32(w.in_features), max_rows = 128;
  std::vector<std::uint16_t> input(U64(max_rows) * k);
  for (U32 row = 0; row < max_rows; ++row) for (U32 col = 0; col < k; ++col) {
    const float amplitude = row % 3 == 0 ? .001f : row % 3 == 1 ? 1.f : 16.f;
    input[U64(row) * k + col] = float_to_bf16(
        amplitude * float(int((col * 17 + row * 31) % 257) - 128) / 127.f);
  }
  auto x = ctx.upload(std::as_bytes(std::span(input)));
  auto y = ctx.make_buffer(U64(max_rows) * n * 2);
  auto sums = ctx.make_buffer(U64(max_rows) * k * 4), zero = ctx.make_buffer(256);
  zero.write(std::vector<std::byte>(256));
  std::map<std::pair<bool,U32>, Profile> plans;
  auto plan = [&](U32 capacity, bool exact) -> Profile & {
    auto key = std::make_pair(exact, capacity);
    if (auto it = plans.find(key); it != plans.end()) return it->second;
    Encoder e(ctx, libs, capacity, prefix + (exact ? ".exact" : ".legacy"));
    const auto route = exact ? select_row_exact_linear_route(w.format, n, k, capacity)
                             : resolve_linear_route(w,n,k,capacity,
                                 ctx.capabilities().device_name,true).route;
    linear(e, w, {&x, 0, x.size()}, {&y, 0, y.size()}, n, k,
           {&sums,0,sums.size()}, {&zero,0,zero.size()}, route);
    return plans.emplace(key, e.finish()).first->second;
  };
  auto run = [&](U32 capacity, U32 live, bool exact) {
    auto &p = plan(capacity, exact);
    for (const auto &patch : p.patches) {
      require(!patch.position, "unexpected position patch");
      const U32 v = live * patch.multiplier;
      std::memcpy(p.parameter_data.data() + patch.offset + patch.word * 4, &v, 4);
    }
    p.parameters.write("step", p.parameter_data);
    std::vector<std::uint16_t> sentinel(U64(max_rows) * n, 0xa5a5);
    y.write(std::as_bytes(std::span(sentinel)));
    const std::array calls{MetalCommandInvocation{.plan=&p.plan,.bindings=p.bindings,.parameters=&p.parameters}};
    (void)ctx.execute_batch(calls);
    auto all = read(y, U64(max_rows) * n);
    require(std::all_of(all.begin() + U64(live) * n, all.end(), [](auto v){return v==0xa5a5;}),
            "invalid rows wrote output");
    all.resize(U64(live) * n);
    require(std::all_of(all.begin(),all.end(),[](auto b){return std::isfinite(value(b));}),
            "nonfinite row result");
    require(std::any_of(all.begin(),all.end(),[](auto b){return b!=0xa5a5;}), "output was not written");
    return all;
  };
  const auto batched = run(128,128,true);
  const auto original = input;
  U64 compared = 0, legacy_mismatches = 0;
  for (U32 row = 0; row < max_rows; ++row) {
    x.write(std::as_bytes(std::span(original).subspan(U64(row)*k,k)));
    const auto solo = run(1,1,true);
    if (!std::equal(solo.begin(),solo.end(),batched.begin()+U64(row)*n)) {
      for (U32 col=0;col<n;++col) if(solo[col]!=batched[U64(row)*n+col])
        throw std::runtime_error("solo/window " + w.format + " row=" + std::to_string(row) +
            " col=" + std::to_string(col) + " solo=" + std::to_string(solo[col]) +
            " batch=" + std::to_string(batched[U64(row)*n+col]));
    }
    compared += n;
  }
  x.write(std::as_bytes(std::span(original)));
  for (U32 width : {1U,2U,3U,4U,5U,6U,7U,8U,9U,15U,16U,17U,31U,32U,33U,63U,64U,65U,127U,128U}) {
    const auto result = run(width,width,true);
    require(std::equal(result.begin(),result.end(),batched.begin()), "width seam bit mismatch");
    compared += result.size();
    U32 legacy_capacity = 128;
    for (U32 c : {1U,2U,4U,8U,16U,32U,64U,128U})
      if (width <= c) { legacy_capacity = c; break; }
    const auto legacy = run(legacy_capacity,width,false);
    for (U64 i=0;i<result.size();++i) legacy_mismatches += result[i]!=legacy[i];
  }
  // Capacity 128 with only seven live rows exercises masking independently of dispatch capacity.
  const auto partial = run(128,7,true);
  require(std::equal(partial.begin(),partial.end(),batched.begin()), "capacity/live mask mismatch");
  std::vector<std::uint16_t> permuted(original.size());
  for(U32 row=0;row<128;++row)
    std::copy_n(original.begin()+U64((row*37)%128)*k,k,permuted.begin()+U64(row)*k);
  x.write(std::as_bytes(std::span(permuted)));
  auto shuffled=run(128,128,true);
  for(U32 row=0;row<128;++row)
    require(std::equal(shuffled.begin()+U64(row)*n,shuffled.begin()+U64(row+1)*n,
                       batched.begin()+U64((row*37)%128)*n),"row reorder bit mismatch");
  // Fused parent outputs: two disjoint row views share the parent strides and numerical body.
  x.write(std::as_bytes(std::span(original)));
  U32 split=n/2;
  U64 output_first=0;
  if (n >= 2) for (const auto [first,count] : {std::pair<U32,U32>{0,split},{split,n-split}}) {
    auto member=w; member.row_count=count; member.parent_shape={count,k};
    for(auto &p:member.planes) {p.offset+=U64(first)*p.row_stride;p.bytes=U64(count)*p.row_stride;}
    Encoder e(ctx,libs,8,prefix+".member"+std::to_string(first));
    linear(e,member,{&x,0,x.size()},{&y,0,y.size()},count,k,
           {&sums,0,sums.size()},{&zero,0,zero.size()},select_row_exact_linear_route(member.format,count,k,8));
    auto p=e.finish();const std::array calls{MetalCommandInvocation{.plan=&p.plan,.bindings=p.bindings,.parameters=&p.parameters}};
    (void)ctx.execute_batch(calls);const auto ys=read(y,U64(count)*8);
    for(U32 r=0;r<8;++r)require(std::equal(ys.begin()+U64(r)*count,ys.begin()+U64(r+1)*count,
        batched.begin()+U64(r)*n+first),"fused/separate column bit mismatch");
    output_first+=ys.size();
  }
  // Independent scalar storage decoder and one-hot rows verify actual values,
  // not merely two equally wrong GPU invocations.
  std::map<std::string,std::vector<std::byte>> raw;
  for(const auto &p:w.planes)raw.emplace(p.kind,p.buffer->read(p.offset,p.bytes));
  const auto u8=[&](std::string_view kind,U64 index){return U32(std::to_integer<unsigned char>(raw.at(std::string(kind)).at(index)));};
  const auto b16=[&](std::string_view kind,U64 index){return value(std::uint16_t(u8(kind,index*2)|(u8(kind,index*2+1)<<8)));};
  const auto decode=[&](U32 row,U32 col){
    if(w.format=="direct_bf16_v1")return b16("data",U64(row)*k+col);
    const auto route=select_row_exact_linear_route(w.format,n,k,1);
    if(w.format=="mxfp4_e2m1_g32_e8m0_v1") {
      const U32 code=(u8("data",U64(row)*w.padded_k/2+col/2)>>((col&1)*4))&15;
      constexpr float levels[8]={0,.5f,1,1.5f,2,3,4,6};
      const U32 exponent=u8("scale",U64(row)*(w.padded_k/32)+col/32);
      return std::ldexp((code&8?-1.f:1.f)*levels[code&7],int(exponent)-127);
    }
    U32 code=route.bits==8?u8("code",U64(row)*w.padded_k+col)
      :(u8("base_code",U64(row)*w.padded_k/2+col/2)>>((col&1)*4))&15;
    if(route.bits==5)code|=((u8("high_bit",U64(row)*w.padded_k/8+col/8)>>(col&7))&1)<<4;
    if(route.bits==6)code|=((u8("high_2bit",U64(row)*w.padded_k/4+col/4)>>((col&3)*2))&3)<<4;
    const U64 index=U64(row)*(w.padded_k/route.group)+col/route.group;
    return std::fma(b16("scale",index),float(code),b16("bias",index));
  };
  const std::array<U32,14> columns{0,1,2,3,4,5,6,7,31,32,63,64,127,k-1};
  std::vector<std::uint16_t> eye(original.size());
  for(U32 r=0;r<columns.size();++r)eye[U64(r)*k+columns[r]]=float_to_bf16(1.f);
  x.write(std::as_bytes(std::span(eye)));
  auto actual_values=run(32,U32(columns.size()),true);
  for(U32 r=0;r<columns.size();++r)for(U32 col=0;col<n;++col) {
    const float decoded = decode(col,columns[r]);
    const auto expected = float_to_bf16(decoded == 0.f ? 0.f : decoded);
    if(actual_values[U64(r)*n+col]!=expected)
      throw std::runtime_error("one-hot " + w.format + " input=" + std::to_string(columns[r]) +
          " output=" + std::to_string(col) + " actual=" + std::to_string(actual_values[U64(r)*n+col]) +
          " expected=" + std::to_string(expected));
  }
  auto exact=select_row_exact_linear_route(w.format,n,k,8);
  auto legacy=select_linear_route(w.format,n,k,8);
  require(row_exact_routes_compatible(exact,select_row_exact_linear_route(w.format,n,k,1),k,k),"exact group lost across widths");
  require(!row_exact_routes_compatible(exact,legacy,k,k),"legacy candidate admitted as exact");
  require(!row_exact_routes_compatible(exact,exact,k,k+32),"different K admitted for fusion");
  return {{"object",w.name},{"format",w.format},{"n",n},{"k",k},{"actual_executed_n",n},
          {"padded_k",w.padded_k},{"numeric_contract_id",exact.numeric_contract_id},
          {"row_exact_group",exact.row_exact_group},{"widths",{1,2,3,4,5,6,7,8,9,15,16,17,31,32,33,63,64,65,127,128}},
          {"solo_rows",128},{"bit_values_compared",compared},{"reorder_rows",128},
          {"fusion_values",output_first},{"invalid_rows_preserved",true},
          {"fusion_geometry_eligible",n>=2},
          {"one_hot_values_checked",U64(columns.size())*n},
          {"legacy_m4_short_routes_enabled",true},
          {"legacy_mismatched_values",legacy_mismatches},{"passed",true}};
}
void check_p2_kernel_regressions(const MetalContext &ctx,
                                 const std::filesystem::path &libs) {
  auto zero = ctx.make_buffer(256);
  zero.write(std::vector<std::byte>(256));
  const Ref zero_ref{&zero, 0, zero.size()};

  // 1. MIVK-4: BF16 linear with padded_k > in_features on both row_exact and qwen_forward.
  {
    constexpr U32 n = 12, k = 96, padded_k = 128, t = 4;
    Weight w;
    w.view.name = "p2.bf16.padded";
    w.view.format = "direct_bf16_v1";
    w.view.layout = "contiguous_le_v1";
    w.view.parent_shape = {n, k};
    w.view.row_count = n;
    w.view.in_features = k;
    w.view.padded_k = padded_k;
    std::vector<std::uint16_t> w_words(U64(n) * padded_k, float_to_bf16(42.0F));
    for (U32 r = 0; r < n; ++r) {
      for (U32 c = 0; c < k; ++c) {
        w_words[U64(r) * padded_k + c] =
            float_to_bf16(float(int((r * 23 + c * 11) % 97) - 48) / 64.0F);
      }
    }
    auto w_bytes = std::as_bytes(std::span(w_words));
    w.add(ctx, "data", {w_bytes.begin(), w_bytes.end()});
    std::vector<std::uint16_t> x_words(U64(t) * k);
    for (U32 r = 0; r < t; ++r) {
      for (U32 c = 0; c < k; ++c) {
        x_words[U64(r) * k + c] =
            float_to_bf16(float(int((r * 19 + c * 7) % 61) - 30) / 32.0F);
      }
    }
    auto x_buf = ctx.upload(std::as_bytes(std::span(x_words)));
    auto y_buf = ctx.make_buffer(U64(t) * n * 2ULL);
    auto sums_buf = ctx.make_buffer(U64(t) * k * 4ULL);
    for (bool exact : {true, false}) {
      Encoder e(ctx, libs, t,
                exact ? "p2.bf16.padded.exact" : "p2.bf16.padded.legacy");
      const auto route = exact
                             ? select_row_exact_linear_route(w.view.format, n, k, t)
                             : select_linear_route(w.view.format, n, k, t);
      linear(e, w.view, {&x_buf, 0, x_buf.size()}, {&y_buf, 0, y_buf.size()},
             n, k, {&sums_buf, 0, sums_buf.size()}, zero_ref, route, t);
      auto prof = e.finish();
      const std::array calls{MetalCommandInvocation{
          .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
      static_cast<void>(ctx.execute_batch(calls));
      const auto got = read(y_buf, U64(t) * n);
      for (U32 row = 0; row < t; ++row) {
        for (U32 col = 0; col < n; ++col) {
          double acc = 0.0;
          for (U32 j = 0; j < k; ++j) {
            acc += double(value(x_words[U64(row) * k + j])) *
                   double(value(w_words[U64(col) * padded_k + j]));
          }
          const float actual = value(got[U64(row) * n + col]);
          const float tol = 0.02F * std::max(1.0F, std::abs(float(acc)));
          require(std::isfinite(actual) && std::abs(actual - float(acc)) <= tol,
                  exact ? "row_exact BF16 ignored padded_input_features"
                        : "qwen_forward BF16 ignored padded_k");
        }
      }
    }
  }

  // 2. MIVE-13 + MIVK-5: pre-tiled oq_nax_tiled_g64_v1 Q8 and
  // oq_nax_tiled_g128_v1 Q4/Q8 (both repeated g64 nax_sb and compact g128 nax_sb).
  {
    constexpr U32 n = 19, k = 128, bn = 16, padded_n = 32, groups64 = k / 64U, t = 2;
    std::vector<std::uint16_t> x_words(U64(t) * k);
    for (U32 r = 0; r < t; ++r) {
      for (U32 c = 0; c < k; ++c) {
        x_words[U64(r) * k + c] =
            float_to_bf16(float(int((r * 13 + c * 5) % 43) - 21) / 32.0F);
      }
    }
    auto x_buf = ctx.upload(std::as_bytes(std::span(x_words)));
    auto y_buf = ctx.make_buffer(U64(t) * n * 2ULL);
    auto sums_buf = ctx.make_buffer(U64(t) * k * 4ULL);

    const auto run_pretiled = [&](U32 bits, U32 format_gs,
                                  const std::string &layout, bool compact_sb) {
      Weight w;
      w.view.name = "p2.pretiled." + layout + ".q" + std::to_string(bits);
      w.view.format = "oq_affine_q" + std::to_string(bits) + "_g" +
                      std::to_string(format_gs) + "_bf16_v1";
      w.view.layout = layout;
      w.view.parent_shape = {n, k};
      w.view.row_count = n;
      w.view.in_features = k;
      w.view.padded_k = k;

      const U32 sb_groups = compact_sb ? (k / 128U) : groups64;
      const U64 code_bytes = U64(padded_n) * (bits == 8U ? k : (k / 2U));
      std::vector<std::uint8_t> codes(code_bytes, 0);
      std::vector<U32> sb(U64(sb_groups) * padded_n, 0);
      std::vector<float> ref_w(U64(n) * k, 0.0F);

      for (U32 row = 0; row < n; ++row) {
        const U32 tile_row = row / bn;
        const U32 rn = row % bn;
        for (U32 g = 0; g < groups64; ++g) {
          const U32 sb_g = compact_sb ? (g >> 1U) : g;
          const U32 logical_g = (format_gs == 128U) ? (g >> 1U) : g;
          const std::uint16_t sc_b =
              float_to_bf16(0.015625F * float(1U + (row + logical_g) % 5U));
          const std::uint16_t bi_b =
              float_to_bf16(-0.03125F * float(1U + (row % 3U)));
          sb[U64(sb_g) * padded_n + row] = U32(sc_b) | (U32(bi_b) << 16U);
          const float sc = value(sc_b);
          const float bi = value(bi_b);
          const U64 tile_idx = U64(tile_row) * groups64 + g;
          for (U32 off = 0; off < 64U; ++off) {
            const U32 col = g * 64U + off;
            const U32 q = (row * 17U + col * 9U) & ((1U << bits) - 1U);
            if (bits == 8U) {
              codes[tile_idx * (U64(bn) * 64ULL) + U64(rn) * 64ULL + off] =
                  static_cast<std::uint8_t>(q);
            } else {
              codes[tile_idx * (U64(bn) * 32ULL) + U64(rn) * 32ULL + off / 2U] |=
                  static_cast<std::uint8_t>((q & 0xFU) << ((off & 1U) * 4U));
            }
            ref_w[U64(row) * k + col] = std::fma(sc, float(q), bi);
          }
        }
      }
      auto c_raw = std::as_bytes(std::span(codes));
      auto sb_raw = std::as_bytes(std::span(sb));
      w.add(ctx, "nax_codes", {c_raw.begin(), c_raw.end()});
      w.add(ctx, "nax_sb", {sb_raw.begin(), sb_raw.end()});

      Encoder e(ctx, libs, t,
                "p2.pretiled." + layout + ".q" + std::to_string(bits) +
                    (compact_sb ? ".compact" : ".full"));
      const auto route = select_linear_route(w.view.format, n, k, t);
      linear(e, w.view, {&x_buf, 0, x_buf.size()}, {&y_buf, 0, y_buf.size()},
             n, k, {&sums_buf, 0, sums_buf.size()}, zero_ref, route, t);
      auto prof = e.finish();
      const std::array calls{MetalCommandInvocation{
          .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
      static_cast<void>(ctx.execute_batch(calls));
      const auto got = read(y_buf, U64(t) * n);
      for (U32 row = 0; row < t; ++row) {
        for (U32 col = 0; col < n; ++col) {
          double acc = 0.0;
          for (U32 j = 0; j < k; ++j) {
            acc += double(value(x_words[U64(row) * k + j])) *
                   double(ref_w[U64(col) * k + j]);
          }
          const float actual = value(got[U64(row) * n + col]);
          const float tol = 0.03F * std::max(1.0F, std::abs(float(acc)));
          require(std::isfinite(actual) && std::abs(actual - float(acc)) <= tol,
                  "pre-tiled affine fallback diverged for " + w.view.name);
        }
      }
    };

    run_pretiled(8U, 64U, "oq_nax_tiled_g64_v1", false);
    run_pretiled(4U, 128U, "oq_nax_tiled_g128_v1", false);
    run_pretiled(8U, 128U, "oq_nax_tiled_g128_v1", false);
    run_pretiled(4U, 128U, "oq_nax_tiled_g128_v1", true);
    run_pretiled(8U, 128U, "oq_nax_tiled_g128_v1", true);
  }

  // 3. MIVK-6: affine4 KV quantizer must clamp BF16 outliers (|v| > 65504)
  // before FP16 half conversion so metadata stays finite and attention never NaNs.
  {
    constexpr U32 qh = 1, kvh = 1, hd = 128, page_tokens = 64;
    const U64 vec_bytes = hd / 2ULL + 16ULL;
    const U64 head_bytes = U64(page_tokens) * vec_bytes + 32ULL;
    const U64 code_bytes = U64(page_tokens) * (hd / 2ULL);
    std::vector<std::uint16_t> kv_words(hd, float_to_bf16(1.0F));
    kv_words[0] = float_to_bf16(-1.0e12F);
    kv_words[1] = float_to_bf16(1.0e12F);
    kv_words[2] = float_to_bf16(100000.0F);
    kv_words[3] = float_to_bf16(-100000.0F);
    std::vector<std::uint16_t> q_words(hd, float_to_bf16(0.05F));
    std::vector<std::uint16_t> g_words(hd, float_to_bf16(1.0F));
    const std::array<U32, 1> pt_words{0U};

    auto k_buf = ctx.upload(std::as_bytes(std::span(kv_words)));
    auto v_buf = ctx.upload(std::as_bytes(std::span(kv_words)));
    auto q_buf = ctx.upload(std::as_bytes(std::span(q_words)));
    auto g_buf = ctx.upload(std::as_bytes(std::span(g_words)));
    auto ck_buf = ctx.make_buffer(head_bytes);
    auto cv_buf = ctx.make_buffer(head_bytes);
    auto out_buf = ctx.make_buffer(U64(hd) * 2ULL);
    auto pt_buf = ctx.upload(std::as_bytes(std::span(pt_words)));
    ck_buf.zero();
    cv_buf.zero();

    const auto p_words = words({qh, kvh, hd, 1U, 0U, page_tokens, page_tokens,
                                1U, hd, hd, hd, hd, hd, hd});
    auto verify_half_meta_finite = [&](const MetalBuffer &cache, const char *label) {
      const auto raw_meta = cache.read(code_bytes, 16ULL);
      std::array<std::uint16_t, 8> half_bits{};
      std::memcpy(half_bits.data(), raw_meta.data(), raw_meta.size());
      for (std::uint16_t hbits : half_bits) {
        require((hbits & 0x7C00U) != 0x7C00U,
                std::string(label) + " wrote non-finite FP16 scale/zero");
      }
    };

    Encoder e(ctx, libs, 1U, "p2.affine4.clamp.qf");
    auto &app = e.emit("qwen_forward", "omlx_m5_forward_paged_kv_append_affine4",
                       {.width = 32ULL, .height = kvh}, {.width = 32ULL},
                       {{&k_buf, 0, k_buf.size()},
                        {&v_buf, 0, v_buf.size()},
                        {&ck_buf, 0, ck_buf.size()},
                        {&cv_buf, 0, cv_buf.size()},
                        {&pt_buf, 0, pt_buf.size()}});
    e.dynamic(app, 5, p_words);
    auto &att = e.emit("qwen_forward",
                       "omlx_m5_forward_paged_causal_attention_affine4_exact",
                       {.width = 32ULL, .height = qh}, {.width = 32ULL},
                       {{&q_buf, 0, q_buf.size()},
                        {&g_buf, 0, g_buf.size()},
                        {&ck_buf, 0, ck_buf.size()},
                        {&cv_buf, 0, cv_buf.size()},
                        {&out_buf, 0, out_buf.size()},
                        {&pt_buf, 0, pt_buf.size()}});
    e.dynamic(att, 6, p_words);
    auto prof = e.finish();
    const std::array calls{MetalCommandInvocation{
        .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
    static_cast<void>(ctx.execute_batch(calls));
    verify_half_meta_finite(ck_buf, "qwen_forward K affine4");
    verify_half_meta_finite(cv_buf, "qwen_forward V affine4");
    const auto att_out = read(out_buf, hd);
    for (auto word_out : att_out) {
      require(std::isfinite(value(word_out)),
              "qwen_forward affine4 attention produced non-finite output on clamped KV");
    }

    if (std::filesystem::exists(libs / "omlx_m5_attention_affine4.metallib")) {
      ck_buf.zero();
      cv_buf.zero();
      Encoder e4(ctx, libs, 1U, "p2.affine4.clamp.nax");
      auto &app4 = e4.emit("attention_affine4",
                           "omlx_m5_paged_kv_append_affine4_g128",
                           {.width = 32ULL, .height = kvh}, {.width = 32ULL},
                           {{&k_buf, 0, k_buf.size()},
                            {&v_buf, 0, v_buf.size()},
                            {&ck_buf, 0, ck_buf.size()},
                            {&cv_buf, 0, cv_buf.size()},
                            {&pt_buf, 0, pt_buf.size()}});
      e4.dynamic(app4, 5, p_words);
      auto prof4 = e4.finish();
      const std::array calls4{MetalCommandInvocation{
          .plan = &prof4.plan, .bindings = prof4.bindings, .parameters = &prof4.parameters}};
      static_cast<void>(ctx.execute_batch(calls4));
      verify_half_meta_finite(ck_buf, "attention_affine4 K");
      verify_half_meta_finite(cv_buf, "attention_affine4 V");
    }
  }

  // 4. MIVK-3: moe9_router_fused_top8_impl (T=1) must self-zero moe_tg_done_buf
  // so consecutive independent dispatches update route_ids_out without external zeroing.
  if (std::filesystem::exists(libs / "omlx_m5_moe_nax.metallib")) {
    constexpr U32 h = 64U, experts = 256U;
    std::vector<std::uint16_t> rw(U64(experts) * h, float_to_bf16(0.0F));
    for (U32 e_idx = 0; e_idx < 8U; ++e_idx) {
      rw[U64(10U + e_idx) * h + 0U] = float_to_bf16(4.0F + float(e_idx));
      rw[U64(100U + e_idx) * h + 1U] = float_to_bf16(4.0F + float(e_idx));
    }
    std::vector<std::uint16_t> x1(h, float_to_bf16(0.0F));
    x1[0] = float_to_bf16(1.0F);
    auto x_buf = ctx.upload(std::as_bytes(std::span(x1)));
    auto rw_buf = ctx.upload(std::as_bytes(std::span(rw)));
    auto rlogits = ctx.make_buffer(U64(experts) * 2ULL);
    auto rids = ctx.make_buffer(32ULL);
    auto rscores = ctx.make_buffer(16ULL);
    auto done_buf = ctx.make_buffer(16ULL);
    done_buf.zero();

    Encoder em(ctx, libs, 1U, "p2.moe9.self_reset");
    auto &d = em.emit("moe_nax", "omlx_m5_moe9_router_qmv_fused_top8_t1",
                      {.width = 64ULL * 64ULL}, {.width = 64ULL},
                      {{&x_buf, 0, x_buf.size()},
                       {&rw_buf, 0, rw_buf.size()},
                       zero_ref,
                       zero_ref,
                       zero_ref,
                       zero_ref,
                       zero_ref,
                       {&rlogits, 0, rlogits.size()},
                       {&rids, 0, rids.size()},
                       {&rscores, 0, rscores.size()},
                       {&rlogits, 0, rlogits.size()},
                       {&done_buf, 0, done_buf.size()}});
    em.dynamic(d, 12,
               words({h, h, 64U, 0U, h, 64U, 16U, 0U, 0U, 0U, 0U, 0U, 0U}));
    auto prof = em.finish();
    const std::array calls{MetalCommandInvocation{
        .plan = &prof.plan, .bindings = prof.bindings, .parameters = &prof.parameters}};
    static_cast<void>(ctx.execute_batch(calls));

    U32 done_val = 999U;
    std::array<U32, 8> ids1{};
    const auto d1 = done_buf.read(0, 4);
    const auto r1 = rids.read(0, 32);
    std::memcpy(&done_val, d1.data(), 4);
    std::memcpy(ids1.data(), r1.data(), 32);
    require(done_val == 0U, "moe9_router_fused_top8_t1 failed to self-zero moe_tg_done_buf");
    require(ids1[0] >= 10U && ids1[0] < 18U, "moe9 step 1 routed wrong expert");

    // Step 2 WITHOUT zeroing done_buf: must switch to experts 100..107!
    std::vector<std::uint16_t> x2(h, float_to_bf16(0.0F));
    x2[1] = float_to_bf16(1.0F);
    x_buf.write(std::as_bytes(std::span(x2)));
    static_cast<void>(ctx.execute_batch(calls));
    const auto d2 = done_buf.read(0, 4);
    const auto r2 = rids.read(0, 32);
    std::memcpy(&done_val, d2.data(), 4);
    std::array<U32, 8> ids2{};
    std::memcpy(ids2.data(), r2.data(), 32);
    require(done_val == 0U, "moe9_router_fused_top8_t1 done_buf non-zero after step 2");
    require(ids2[0] >= 100U && ids2[0] < 108U,
            "moe9_router_fused_top8_t1 stuck on stale step 1 routes");
  }

  // 5. MIVK-7: TargetGpuVerifier LegacyRejectionV1 must reject NaN & all -Inf
  // logits under both greedy (temperature == 0) and stochastic (temperature > 0).
  if (std::filesystem::exists(libs / "omlx_m5_target_verify_gpu.metallib")) {
    constexpr U32 tv_vocab = 256U;
    std::vector<std::uint16_t> logits(8ULL * tv_vocab, float_to_bf16(0.0F));
    logits[5] = float_to_bf16(3.0F);
    auto logits_buf = ctx.upload(std::as_bytes(std::span(logits)));
    TargetGpuVerifier verifier(ctx, logits_buf, tv_vocab, libs,
                               SamplingContract::LegacyRejectionV1);
    const auto expect_verify_throws = [&](float temp, const char *msg) {
      GpuSamplingOptions opts;
      opts.contract = SamplingContract::LegacyRejectionV1;
      opts.temperature = temp;
      opts.top_k = temp > 0.0F ? 8U : 1U;
      opts.top_p = 1.0F;
      bool threw = false;
      try {
        static_cast<void>(verifier.select({}, opts, 0ULL, false));
      } catch (const std::exception &) {
        threw = true;
      }
      require(threw, msg);
    };

    // Single NaN in row 0 among finite logits: must propagate and throw!
    logits[17] = 0x7FC0U;
    logits_buf.write(std::as_bytes(std::span(logits)));
    expect_verify_throws(0.0F, "target_verify_gpu greedy ignored NaN logit");
    expect_verify_throws(0.8F, "target_verify_gpu stochastic ignored NaN logit");

    // All -Inf in row 0: must trigger meta[3] = 1 and throw!
    std::fill(logits.begin(), logits.begin() + tv_vocab,
              static_cast<std::uint16_t>(0xFF80U));
    logits_buf.write(std::as_bytes(std::span(logits)));
    expect_verify_throws(0.0F, "target_verify_gpu greedy accepted all -Inf logits");
    expect_verify_throws(0.8F, "target_verify_gpu stochastic accepted all -Inf logits");
  }
}
}
int main(int argc,char **argv) {
  try {
    if(argc!=2 && argc!=4)throw std::invalid_argument("tf_row_exact_tests LIBRARIES [ARTIFACT OBJECT]");
    auto ctx=MetalContext::create();Json rows=Json::array();
    if(argc==4) {
      auto a=ArtifactModel::open(argv[2],false);const auto &o=a.object(argv[3]);
      require(o.shape.size()==2,"real row test requires rank-2");Weight w;
      w.view={.object_id=o.id,.name=o.name,.format=o.format,.layout=o.layout,.parent_shape=o.shape,
              .padded_k=o.padded_k,.row_count=o.shape[0],.in_features=o.shape[1]};
      Json hashes=Json::object();
      for(const auto &p:o.planes) {
        auto bytes=a.plane_bytes(o,p);hashes[p.kind]=artifact_sha256_hex(bytes);
        require(hashes[p.kind].get<std::string>()==p.sha256,"executed real plane hash mismatch");
        w.add(ctx,p.kind,{bytes.begin(),bytes.end()});
      }
      auto result=audit(ctx,w.view,argv[1],"tf.real");result["executed_plane_sha256"]=hashes;rows.push_back(result);
    } else {
      U32 index=0;
      for(U32 bits:{4U,5U,6U,8U})for(U32 gs:{64U,128U}) {
        auto w=synthetic(ctx,bits,gs,false,false);rows.push_back(audit(ctx,w.view,argv[1],"tf.synthetic"+std::to_string(index++)));
      }
      for(bool bf16:{false,true}) {
        auto w=synthetic(ctx,4,32,!bf16,bf16);rows.push_back(audit(ctx,w.view,argv[1],"tf.synthetic"+std::to_string(index++)));
      }
      check_p2_kernel_regressions(ctx, argv[1]);
    }
    std::cout<<Json({{"schema","omlx-m5-p02-tf-row-exact-v1"},{"device",ctx.capabilities().device_name},
      {"scope",argc==4?"real weight full-N operator":"synthetic format/tail/operator"},
      {"whole_program_exact_qualified",false},{"m5_nax_qualified",false},{"cases",rows},{"passed",true}}).dump(2)<<'\n';
    return 0;
  } catch(const std::exception &e){std::cerr<<"P02-TF: "<<e.what()<<'\n';return 1;}
}
