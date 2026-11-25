#include "affine_nax_cache.h"
#include "mxfp4_nax_cache.h"
#include "omlx_m5/artifact.h"
#include "omlx_m5/engine.h"
#include "omlx_m5/execution_contract.h"
#include "omlx_m5/kv_page_pool.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/qwen_forward.h"
#include "silicon_tuning.h"
#include "spec_controller.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace omlx::m5;
using Json = nlohmann::json;
using U32 = std::uint32_t;
using U64 = std::uint64_t;

namespace {

void require(bool ok, std::string_view message) {
  if (!ok) {
    throw std::runtime_error("W2-TEST FAIL: " + std::string(message));
  }
}

struct ScopedEnv {
  std::string name;
  std::optional<std::string> previous;
  ScopedEnv(std::string key, const char *value) : name(std::move(key)) {
    if (const char *old = std::getenv(name.c_str())) {
      previous = std::string(old);
    }
    if (value != nullptr) {
      setenv(name.c_str(), value, 1);
    } else {
      unsetenv(name.c_str());
    }
  }
  ~ScopedEnv() {
    if (previous.has_value()) {
      setenv(name.c_str(), previous->c_str(), 1);
    } else {
      unsetenv(name.c_str());
    }
  }
};

std::string shell_quote(const std::filesystem::path &p) {
  std::string s = p.string();
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out.push_back(c);
    }
  }
  out.push_back('\'');
  return out;
}

void run_cmd(const std::string &cmd) {
  const int rc = std::system(cmd.c_str());
  if (rc != 0) {
    throw std::runtime_error("Subcommand failed (" + std::to_string(rc) +
                             "): " + cmd);
  }
}

std::vector<U32> make_tokens(U32 count, U32 seed_offset, U32 vocab) {
  std::vector<U32> out(count);
  for (U32 i = 0; i < count; ++i) {
    out[i] = (73U * (i + seed_offset) + 19U) % vocab;
  }
  return out;
}

Json test_silicon_tuning(const MetalContext &ctx,
                         const std::filesystem::path &repo_root,
                         const std::filesystem::path &tmp_dir) {
  const auto live = detail::resolve_silicon_tuning(ctx);
  require(live.gpu_cores >= 8U && !live.profile_id.empty(),
          "live SiliconTuningConfig invalid");

  {
    ScopedEnv env_cores("OMLX_M5_GPU_CORES", "64");
    const auto m5u64 = detail::resolve_silicon_tuning(ctx);
    require(m5u64.gpu_cores == 64U &&
                m5u64.profile_id == "apple-m5-ultra-64gpu" &&
                m5u64.split_kv_min_context == 2048U &&
                m5u64.moe_nax_min_tokens == 1U,
            "64-core M5 Ultra SiliconTuningConfig mismatch");
  }
  {
    ScopedEnv env_cores("OMLX_M5_GPU_CORES", "80");
    const auto m5u80 = detail::resolve_silicon_tuning(ctx);
    require(m5u80.gpu_cores == 80U &&
                m5u80.profile_id == "apple-m5-ultra-80gpu" &&
                m5u80.split_kv_partitions == 16U,
            "80-core M5 Ultra SiliconTuningConfig mismatch");
  }
  {
    ScopedEnv env_cores("OMLX_M5_GPU_CORES", "16");
    const auto m5pro = detail::resolve_silicon_tuning(ctx);
    require(m5pro.gpu_cores == 16U &&
                m5pro.profile_id == "apple-m5-pro-16gpu" &&
                m5pro.split_kv_min_context == 8192U,
            "16-core M5 Pro SiliconTuningConfig mismatch");
  }

  const auto custom_json = tmp_dir / "custom_dispatch_table.json";
  {
    std::ofstream out(custom_json);
    out << Json({{"profile_id", "apple-m5-ultra-64gpu"},
                 {"dispatch_policies",
                  {{"attention", {{"split_kv_min_context", 1792}}},
                   {"gdn_prefill", {{"tb8_min_tokens", 8}}},
                   {"moe_routing", {{"prefill_nax_min_tokens", 1}}}}}})
               .dump(2);
  }
  {
    ScopedEnv env_cores("OMLX_M5_GPU_CORES", "64");
    ScopedEnv env_tbl("OMLX_M5_STATIC_DISPATCH_TABLE",
                      custom_json.string().c_str());
    const auto loaded = detail::resolve_silicon_tuning(ctx);
    require(loaded.profile_id == "apple-m5-ultra-64gpu" &&
                loaded.split_kv_min_context == 1792U &&
                loaded.loaded_from_json,
            "JSON static dispatch table override failed");
  }

  auto repo_table =
      repo_root / "tests/fixtures/m5-static-dispatch-table-apple-m5-ultra-64gpu-v1.json";
  if (!std::filesystem::is_regular_file(repo_table)) {
    repo_table =
        repo_root / "docs/backend/m5-static-dispatch-table-apple-m5-ultra-64gpu-v1.json";
  }
  bool repo_table_checked = false;
  if (std::filesystem::is_regular_file(repo_table)) {
    ScopedEnv env_cores("OMLX_M5_GPU_CORES", "64");
    const auto loaded = detail::resolve_silicon_tuning(ctx, repo_table);
    require(loaded.loaded_from_json,
            "repo static dispatch table JSON failed to parse");
    repo_table_checked = true;
  }
  require(repo_table_checked, "repo static dispatch table JSON missing");

  // Verify omlx-m5-silicon-calibration-v1 loading, SKU isolation, and split-K scaling.
  const auto custom_cal = tmp_dir / "apple-m5-max-40gpu.json";
  {
    std::ofstream out(custom_cal);
    out << Json({{"schema", "omlx-m5-silicon-calibration-v1"},
                 {"profile_id", "apple-m5-max-40gpu"},
                 {"measured_hardware_constants",
                  {{"icb_safe_weight_segment_bytes", 1073741824ULL}}},
                 {"measured_dispatch_overrides",
                  {{"attention_split_kv_min_context", 4096},
                   {"attention_split_kv_partitions", 8},
                   {"gdn_prefill_tb8_min_tokens", 16},
                   {"linear_nax_min_tokens", 1},
                   {"linear_split_k_target_slices", 4},
                   {"moe_prefill_nax_min_tokens", 8},
                   {"dflash2_draft_tokens", 5}}}})
               .dump(2);
  }
  {
    ScopedEnv env_cores("OMLX_M5_GPU_CORES", "40");
    ScopedEnv env_cal("OMLX_M5_SILICON_CALIBRATION_FILE",
                      custom_cal.string().c_str());
    const auto loaded = detail::resolve_silicon_tuning(ctx);
    require(loaded.loaded_from_json &&
                loaded.profile_id == "apple-m5-max-40gpu" &&
                loaded.linear_split_k_target_slices == 4U &&
                loaded.split_kv_min_context == 4096U &&
                loaded.split_kv_partitions == 8U &&
                loaded.dflash_draft_tokens == 5U &&
                loaded.max_weight_segment_bytes == 1073741824ULL,
            "omlx-m5-silicon-calibration-v1 load failed");
    require(detail::forward::mxfp4_nax_split_k(
                5120U, 5120U, loaded.linear_split_k_target_slices) == 4U &&
                detail::forward::mxfp4_nax_split_k(
                    248320U, 5120U, loaded.linear_split_k_target_slices) == 2U &&
                detail::forward::affine_nax_split_k(
                    5120U, 5120U, loaded.linear_split_k_target_slices) == 4U,
            "split_k scaling from linear_split_k_target_slices failed");
  }
  {
    // Strict SKU isolation: 40-GPU calibration file must NOT apply to 80-GPU profile.
    ScopedEnv env_cores("OMLX_M5_GPU_CORES", "80");
    ScopedEnv env_cal("OMLX_M5_SILICON_CALIBRATION_FILE",
                      custom_cal.string().c_str());
    const auto isolated = detail::resolve_silicon_tuning(ctx);
    require(!isolated.loaded_from_json &&
                isolated.profile_id == "apple-m5-ultra-80gpu" &&
                isolated.linear_split_k_target_slices == 8U,
            "calibration SKU isolation failed");
  }

  // Verify Q3 affine quantization is accepted in model_config.cpp.
  {
    const auto q3_cfg_path = tmp_dir / "q3_model_config.json";
    {
      std::ofstream out(q3_cfg_path);
      out << Json({{"text_config",
                    {{"hidden_size", 512},
                     {"intermediate_size", 1024},
                     {"num_hidden_layers", 2},
                     {"num_attention_heads", 4},
                     {"num_key_value_heads", 2},
                     {"head_dim", 128},
                     {"vocab_size", 1024},
                     {"max_position_embeddings", 4096},
                     {"rms_norm_eps", 1e-6},
                     {"rope_theta", 10000.0},
                     {"linear_num_key_heads", 2},
                     {"linear_num_value_heads", 4},
                     {"linear_key_head_dim", 128},
                     {"linear_value_head_dim", 128},
                     {"linear_conv_kernel_dim", 4},
                     {"layer_types",
                      {"linear_attention", "full_attention"}}}},
                   {"quantization",
                    {{"bits", 3}, {"group_size", 128}, {"mode", "affine"}}}})
                 .dump(2);
    }
    const auto parsed_q3 = load_qwen_model_config(q3_cfg_path);
    require(parsed_q3.default_quantization.bits == 3 &&
                parsed_q3.default_quantization.group_size == 128 &&
                parsed_q3.default_quantization.mode == "affine",
            "model_config.cpp failed to accept affine bits=3");
  }

  return {{"live_profile", live.profile_id},
          {"live_gpu_cores", live.gpu_cores},
          {"repo_table_checked", repo_table_checked},
          {"passed", true}};
}

Json test_qda_q3_g128_and_pretile(const std::filesystem::path &qwen_fixtures,
                                  const std::filesystem::path &libs,
                                  const std::filesystem::path &python_exe,
                                  const std::filesystem::path &repo_root,
                                  const std::filesystem::path &tmp_dir) {
  const auto affine_src = qwen_fixtures / "affine/source";
  const auto mxfp4_src = qwen_fixtures / "mxfp4/source";
  const auto tool = repo_root / "tools/m5_artifact.py";

  const auto qda_recipe = tmp_dir / "qda_recipe_3_6bpw.json";
  const auto qda_art_untiled = tmp_dir / "qda_q3_untiled.omlxm5";
  const auto qda_art_pretiled = tmp_dir / "qda_q3_pretiled.omlxm5";
  const auto mx_art_untiled = qwen_fixtures / "mxfp4/model.omlxm5";
  const auto mx_art_pretiled = tmp_dir / "mxfp4_pretiled.omlxm5";

  // 1. Dequantize the small Qwen fixture into a clean BF16 checkpoint in tmp_dir,
  // run `qda-search` CLI, ensure both q3_g64 and q3_g128 are present, and emit
  // both untiled and `--nax-pretile` oQe artifacts + pretiled MXFP4!
  const auto bf16_src = tmp_dir / "bf16_src";
  const auto py_driver = tmp_dir / "run_qda_prep.py";
  {
    std::ofstream py(py_driver);
    py << R"PY(
import json, pathlib, struct, sys
import numpy as np

repo = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(repo / "tools"))
import m5_artifact as ma

aff_art = pathlib.Path(sys.argv[2])
aff_cfg_path = pathlib.Path(sys.argv[3])
bf16_dir = pathlib.Path(sys.argv[4])
bf16_dir.mkdir(parents=True, exist_ok=True)

cfg = json.loads(aff_cfg_path.read_text())
cfg.pop("quantization", None)
(bf16_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

_, manifest = ma.read_artifact(aff_art, False)
obj_by_name = {item["name"]: item for item in manifest["objects"]}
src = ma.SafeTensorSource(aff_cfg_path.parent, None)
tensors = {}
try:
    for name, info in sorted(src.tensors.items()):
        if name.endswith((".scales", ".biases")):
            continue
        if info.dtype == "BF16":
            v = src.view(name)
            tensors[name] = ("BF16", info.shape, bytes(v))
            v.release()
            continue
        mname = name[:-len(".weight")] if name.endswith(".weight") else name
        item = obj_by_name[mname]
        shape = tuple(int(x) for x in item["shape"])
        rows_n = int(np.prod(shape[:-1]))
        cols_n = shape[-1]
        rows = [ma.decode_artifact_row(aff_art, manifest, mname, r, 0, cols_n) for r in range(rows_n)]
        arr = np.asarray(rows, dtype=np.float32).reshape(shape)
        tensors[name] = ("BF16", shape, ma._float32_to_bf16_u16(arr).tobytes())
finally:
    src.close()

hdr = {}
chunks = []
off = 0
for k_name, (dt, shp, raw) in tensors.items():
    hdr[k_name] = {"dtype": dt, "shape": list(shp), "data_offsets": [off, off + len(raw)]}
    chunks.append(raw)
    off += len(raw)
hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
pad = (-len(hb)) % 8
if pad:
    hb += b" " * pad
with (bf16_dir / "model.safetensors").open("wb") as f:
    f.write(struct.pack("<Q", len(hb)))
    f.write(hb)
    for c in chunks:
        f.write(c)
)PY";
  }
  run_cmd(shell_quote(python_exe) + " " + shell_quote(py_driver) + " " +
          shell_quote(repo_root) + " " +
          shell_quote(qwen_fixtures / "affine/model.omlxm5") + " " +
          shell_quote(affine_src / "config.json") + " " +
          shell_quote(bf16_src));

  run_cmd(shell_quote(python_exe) + " " + shell_quote(tool) +
          " qda-search --source " + shell_quote(bf16_src) +
          " --output-recipe " + shell_quote(qda_recipe) +
          " --target-bpw 3.55 --head-max-bits 4 --min-bits 3 >/dev/null");
  require(std::filesystem::is_regular_file(qda_recipe),
          "qda-search did not emit recipe JSON");

  Json recipe_doc;
  {
    std::ifstream in(qda_recipe);
    in >> recipe_doc;
  }
  require(recipe_doc.at("schema") == "omlx-qda-recipe-v1" &&
              recipe_doc.at("effective_bpw").get<double>() <= 4.0,
          "qda recipe schema check");

  // Ensure both q3_g64 and q3_g128 are explicitly exercised in the oQe artifacts.
  bool saw_q3_g128 = false;
  for (auto &[mname, spec] : recipe_doc.at("quantization").items()) {
    if (!spec.is_object())
      continue;
    if (spec.value("bits", 0) == 3 && spec.value("group_size", 0) == 128) {
      saw_q3_g128 = true;
    }
    if (mname.find("self_attn.v_proj") != std::string::npos) {
      spec["bits"] = 3;
      spec["group_size"] = 64;
    }
  }
  require(saw_q3_g128, "qda-search at 3.55 bpw did not assign any q3_g128 layer");
  const auto patched_recipe = tmp_dir / "qda_recipe_q3_mixed.json";
  {
    std::ofstream out(patched_recipe);
    out << recipe_doc.dump(2);
  }

  const auto py_emit = tmp_dir / "run_oqe_emit.py";
  {
    std::ofstream py(py_emit);
    py << R"PY(
import json, pathlib, sys
repo = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(repo / "tools"))
import m5_artifact as ma

bf16_dir = pathlib.Path(sys.argv[2])
recipe = json.loads(pathlib.Path(sys.argv[3]).read_text())
ma.quantize_source_with_recipe_to_artifact(bf16_dir, recipe, pathlib.Path(sys.argv[4]), "test/qda-q3", nax_pretile=False)
ma.quantize_source_with_recipe_to_artifact(bf16_dir, recipe, pathlib.Path(sys.argv[5]), "test/qda-q3", nax_pretile=True)
)PY";
  }
  run_cmd(shell_quote(python_exe) + " " + shell_quote(py_emit) + " " +
          shell_quote(repo_root) + " " + shell_quote(bf16_src) + " " +
          shell_quote(patched_recipe) + " " + shell_quote(qda_art_untiled) +
          " " + shell_quote(qda_art_pretiled));
  run_cmd(shell_quote(python_exe) + " " + shell_quote(tool) +
          " pretile --input " + shell_quote(mx_art_untiled) + " --output " +
          shell_quote(mx_art_pretiled) + " >/dev/null");

  // 3. Open with parallel SHA-256 verification and verify decode_weight_row 0-ULP match!
  const auto art_u = ArtifactModel::open(qda_art_untiled, true);
  const auto art_p = ArtifactModel::open(qda_art_pretiled, true);
  require(art_u.objects().size() == art_p.objects().size(),
          "untiled vs pretiled object count mismatch");
  U32 pretiled_q3_objects = 0;
  for (std::size_t idx = 0; idx < art_u.objects().size(); ++idx) {
    const auto &ou = art_u.objects()[idx];
    const auto &op = art_p.objects()[idx];
    if (op.layout == "oq_nax_tiled_g64_v1") {
      if (op.format.starts_with("oq_affine_q3_")) {
        ++pretiled_q3_objects;
      }
      const auto row_u = art_u.decode_weight_row(ou, 0, 0, ou.shape.back());
      const auto row_p = art_p.decode_weight_row(op, 0, 0, op.shape.back());
      require(row_u == row_p,
              "decode_weight_row mismatch between untiled and pretiled " +
                  op.name);
    }
  }
  require(pretiled_q3_objects > 0, "no pre-tiled q3 objects produced");

  // 4. Forward parity + Bug #1 test (Legacy + OMLX_M5_T_DOMAIN_DISPATCH=1 on pre-tiled artifacts!).
  const auto cfg_aff = load_qwen_model_config(affine_src / "config.json");
  auto ctx = MetalContext::create();
  const auto w_u =
      materialize_model(art_u, plan_model_materialization(art_u, ctx), ctx);
  const auto w_p =
      materialize_model(art_p, plan_model_materialization(art_p, ctx), ctx);
  const auto cat_u = build_qwen_layer_catalog(art_u, cfg_aff, 512, 64);
  const auto cat_p = build_qwen_layer_catalog(art_p, cfg_aff, 512, 64);

  {
    ScopedEnv env_td("OMLX_M5_T_DOMAIN_DISPATCH", "1");
    // Bug #1 check on pre-tiled affine (q3_g64 + q3_g128 + q4_g128) under Legacy policy!
    QwenForward fwd_legacy_u(ctx, w_u, cat_u, cfg_aff, 512, libs, {}, false,
                             false, false, std::nullopt, false, false,
                             std::nullopt, LinearNumericPolicy::Legacy);
    QwenForward fwd_legacy_p(ctx, w_p, cat_p, cfg_aff, 512, libs, {}, false,
                             false, false, std::nullopt, false, false,
                             std::nullopt, LinearNumericPolicy::Legacy);
    for (U32 width : {1U, 8U, 12U, 24U}) {
      fwd_legacy_u.reset();
      fwd_legacy_p.reset();
      const auto toks = make_tokens(width, width * 5U, U32(cfg_aff.vocab_size));
      const auto ru = fwd_legacy_u.run(toks, {}, true);
      const auto rp = fwd_legacy_p.run(toks, {}, true);
      require(ru.logits == rp.logits && ru.hidden == rp.hidden &&
                  ru.token == rp.token,
              "untiled vs pretiled q3/g128 Legacy forward mismatch at T=" +
                  std::to_string(width));
    }

    // Also check RowExactPortableV1 + PositionCoupledV1!
    GpuSamplingOptions coupled_samp{.temperature = 0.7f,
                                    .top_k = 16,
                                    .top_p = 0.9f,
                                    .seed = 42,
                                    .contract = SamplingContract::PositionCoupledV1};
    QwenForward fwd_exact_u(ctx, w_u, cat_u, cfg_aff, 512, libs, {}, false,
                            false, false, std::nullopt, false, false,
                            std::nullopt,
                            LinearNumericPolicy::RowExactPortableV1,
                            SamplingContract::PositionCoupledV1,
                            AttentionNumericPolicy::RowExactV1);
    QwenForward fwd_exact_p(ctx, w_p, cat_p, cfg_aff, 512, libs, {}, false,
                            false, false, std::nullopt, false, false,
                            std::nullopt,
                            LinearNumericPolicy::RowExactPortableV1,
                            SamplingContract::PositionCoupledV1,
                            AttentionNumericPolicy::RowExactV1);
    const auto prompt12 = make_tokens(12U, 3U, U32(cfg_aff.vocab_size));
    const auto eu = fwd_exact_u.run(prompt12, coupled_samp, true);
    const auto ep = fwd_exact_p.run(prompt12, coupled_samp, true);
    require(eu.logits == ep.logits && eu.token == ep.token,
            "untiled vs pretiled q3/g128 RowExact T=12 mismatch");

    // Bug #1 check on pre-tiled MXFP4 under Legacy + T_DOMAIN_DISPATCH=1!
    const auto cfg_mx = load_qwen_model_config(mxfp4_src / "config.json");
    const auto art_mx_u = ArtifactModel::open(mx_art_untiled, true);
    const auto art_mx_p = ArtifactModel::open(mx_art_pretiled, true);
    const auto w_mx_u = materialize_model(
        art_mx_u, plan_model_materialization(art_mx_u, ctx), ctx);
    const auto w_mx_p = materialize_model(
        art_mx_p, plan_model_materialization(art_mx_p, ctx), ctx);
    const auto cat_mx_u = build_qwen_layer_catalog(art_mx_u, cfg_mx, 512, 64);
    const auto cat_mx_p = build_qwen_layer_catalog(art_mx_p, cfg_mx, 512, 64);
    QwenForward mx_leg_u(ctx, w_mx_u, cat_mx_u, cfg_mx, 512, libs, {}, false,
                         false, false, std::nullopt, false, false, std::nullopt,
                         LinearNumericPolicy::Legacy);
    QwenForward mx_leg_p(ctx, w_mx_p, cat_mx_p, cfg_mx, 512, libs, {}, false,
                         false, false, std::nullopt, false, false, std::nullopt,
                         LinearNumericPolicy::Legacy);
    for (U32 width : {1U, 8U, 12U}) {
      mx_leg_u.reset();
      mx_leg_p.reset();
      const auto toks = make_tokens(width, 91U + width, U32(cfg_mx.vocab_size));
      const auto rmu = mx_leg_u.run(toks, {}, true);
      const auto rmp = mx_leg_p.run(toks, {}, true);
      require(rmu.logits == rmp.logits && rmu.token == rmp.token,
              "MXFP4 pretiled Legacy Bug #1 check mismatch at T=" +
                  std::to_string(width));
    }
  }

  return {{"pretiled_q3_objects", pretiled_q3_objects},
          {"effective_bpw", recipe_doc.at("effective_bpw")},
          {"passed", true}};
}

Json test_memory_squeeze_and_packed_batch(
    const std::filesystem::path &qwen_fixtures,
    const std::filesystem::path &libs) {
  const auto root = qwen_fixtures / "affine";
  const auto cfg = load_qwen_model_config(root / "source/config.json");
  const auto artifact = ArtifactModel::open(root / "model.omlxm5", true);
  auto ctx = MetalContext::create();
  const auto weights =
      materialize_model(artifact, plan_model_materialization(artifact, ctx), ctx);
  const auto layers = build_qwen_layer_catalog(artifact, cfg, 1024, 64);
  const std::array<U32, 2> taps{0, 1};

  // 1. Memory squeeze invariants: zero `.scratch`, capped 4096-row feature_history, 16-row secondary slot!
  const auto plan_64k_primary = plan_qwen_forward_memory(
      cfg, 65536U, 2U, false, true, false, true, true, false, 0U, 1024U, false);
  const auto plan_64k_secondary = plan_qwen_forward_memory(
      cfg, 65536U, 2U, false, true, false, true, true, false, 0U, 16U, true);
  for (const auto &b : plan_64k_primary.buffers) {
    require(b.name.find(".scratch") == std::string::npos,
            "dead GDN .scratch buffer still allocated: " + b.name);
  }
  require(plan_64k_primary.feature_ring_rows == 4096U,
          "64K context feature_ring_rows did not cap at 4096");
  require(plan_64k_secondary.workspace_bytes * 5ULL <
              plan_64k_primary.workspace_bytes,
          "16-row secondary slot workspace did not shrink as expected");
  require(plan_64k_secondary.state_bytes * 2ULL <
              plan_64k_primary.state_bytes,
          "affine4_g128 KV did not shrink state_bytes by >2x");

  // 2. Construct B=3 slots (slot0 primary + slot1, slot2 secondary sharing slot0's 1024-row workspace)
  // and 3 standalone 1024-row oracle programs!
  GpuSamplingOptions samp{.temperature = 0.6f,
                          .top_k = 16,
                          .top_p = 0.95f,
                          .seed = 1234,
                          .contract = SamplingContract::PositionCoupledV1};

  QwenForward slot0(ctx, weights, layers, cfg, 1024, libs, taps, false, true,
                    false, std::nullopt, false, true, std::nullopt,
                    LinearNumericPolicy::RowExactPortableV1,
                    SamplingContract::PositionCoupledV1,
                    AttentionNumericPolicy::RowExactV1, false, "bf16", nullptr);
  QwenForward slot1(ctx, weights, layers, cfg, 1024, libs, taps, false, true,
                    false, std::nullopt, false, true, std::nullopt,
                    LinearNumericPolicy::RowExactPortableV1,
                    SamplingContract::PositionCoupledV1,
                    AttentionNumericPolicy::RowExactV1, false, "bf16", &slot0);
  QwenForward slot2(ctx, weights, layers, cfg, 1024, libs, taps, false, true,
                    false, std::nullopt, false, true, std::nullopt,
                    LinearNumericPolicy::RowExactPortableV1,
                    SamplingContract::PositionCoupledV1,
                    AttentionNumericPolicy::RowExactV1, false, "bf16", &slot0);
  require(slot1.memory_plan().workspace_rows == 16U &&
              slot0.memory_plan().workspace_rows == 1024U,
          "secondary slot workspace_rows mismatch");

  QwenForward solo0(ctx, weights, layers, cfg, 1024, libs, taps, false, true,
                    false, std::nullopt, false, true, std::nullopt,
                    LinearNumericPolicy::RowExactPortableV1,
                    SamplingContract::PositionCoupledV1,
                    AttentionNumericPolicy::RowExactV1);
  QwenForward solo1(ctx, weights, layers, cfg, 1024, libs, taps, false, true,
                    false, std::nullopt, false, true, std::nullopt,
                    LinearNumericPolicy::RowExactPortableV1,
                    SamplingContract::PositionCoupledV1,
                    AttentionNumericPolicy::RowExactV1);
  QwenForward solo2(ctx, weights, layers, cfg, 1024, libs, taps, false, true,
                    false, std::nullopt, false, true, std::nullopt,
                    LinearNumericPolicy::RowExactPortableV1,
                    SamplingContract::PositionCoupledV1,
                    AttentionNumericPolicy::RowExactV1);

  // Prefill heterogeneous lengths: 19, 73 (>16 rows on secondary slot1!), 131 (>16 rows on secondary slot2!)
  const auto p0 = make_tokens(19U, 1U, U32(cfg.vocab_size));
  const auto p1 = make_tokens(73U, 200U, U32(cfg.vocab_size));
  const auto p2 = make_tokens(131U, 400U, U32(cfg.vocab_size));

  const auto r_s0 = slot0.run(p0, samp);
  const auto r_s1 = slot1.run(p1, samp); // delegates 73-tok prefill via slot0!
  const auto r_s2 = slot2.run(p2, samp); // delegates 131-tok prefill via slot0!
  const auto r_o0 = solo0.run(p0, samp);
  const auto r_o1 = solo1.run(p1, samp);
  const auto r_o2 = solo2.run(p2, samp);
  require(r_s0.token == r_o0.token && r_s1.token == r_o1.token &&
              r_s2.token == r_o2.token,
          "secondary 16-row delegated prefill token mismatch");
  require(slot1.read_last_logits_bf16() == solo1.read_last_logits_bf16() &&
              slot2.read_last_logits_bf16() == solo2.read_last_logits_bf16(),
          "secondary 16-row delegated prefill last_logits mismatch");
  require(slot1.inspect_recurrent_state() == solo1.inspect_recurrent_state() &&
              slot2.inspect_recurrent_state() == solo2.inspect_recurrent_state(),
          "secondary 16-row delegated prefill recurrent state mismatch");

  // 3. Cross-slot `run_packed_batch` AR parity (heterogeneous widths {1, 3, 2})!
  const auto ar0 = make_tokens(1U, 700U, U32(cfg.vocab_size));
  const auto ar1 = make_tokens(3U, 710U, U32(cfg.vocab_size));
  const auto ar2 = make_tokens(2U, 720U, U32(cfg.vocab_size));
  const std::array<PackedSlotSlice, 3> ar_slices{{
      {&slot0, 0U, ar0, samp, false},
      {&slot1, 1U, ar1, samp, false},
      {&slot2, 2U, ar2, samp, false},
  }};
  const auto packed_ar = slot0.run_packed_batch(ar_slices, true);
  require(packed_ar.slot_results.size() == 3U, "packed AR slot count");
  (void)solo0.run(ar0, samp);
  (void)solo1.run(ar1, samp);
  (void)solo2.run(ar2, samp);
  require(slot0.batch_logits_buffer().read() ==
                  solo0.read_last_logits_bf16() &&
              slot1.batch_logits_buffer().read() ==
                  solo1.read_last_logits_bf16() &&
              slot2.batch_logits_buffer().read() ==
                  solo2.read_last_logits_bf16(),
          "run_packed_batch AR batch_logits 0-ULP mismatch");
  require(slot0.read_last_logits_bf16() == solo0.read_last_logits_bf16() &&
              slot1.read_last_logits_bf16() == solo1.read_last_logits_bf16() &&
              slot2.read_last_logits_bf16() == solo2.read_last_logits_bf16(),
          "run_packed_batch AR read_last_logits_bf16 0-ULP mismatch");
  require(slot0.inspect_recurrent_state() == solo0.inspect_recurrent_state() &&
              slot1.inspect_recurrent_state() == solo1.inspect_recurrent_state() &&
              slot2.inspect_recurrent_state() == solo2.inspect_recurrent_state(),
          "run_packed_batch AR recurrent state mismatch");

  // 4. Cross-slot `run_packed_batch` speculative `verify` parity ({4, 8, 5}) + `commit_prefix` ({2, 5, 0})!
  const auto vf0 = make_tokens(4U, 800U, U32(cfg.vocab_size));
  const auto vf1 = make_tokens(8U, 810U, U32(cfg.vocab_size));
  const auto vf2 = make_tokens(5U, 820U, U32(cfg.vocab_size));
  const std::array<PackedSlotSlice, 3> vf_slices{{
      {&slot0, 0U, vf0, samp, true},
      {&slot1, 1U, vf1, samp, true},
      {&slot2, 2U, vf2, samp, true},
  }};
  (void)slot0.run_packed_batch(vf_slices, true);
  (void)solo0.verify(vf0, samp);
  (void)solo1.verify(vf1, samp);
  (void)solo2.verify(vf2, samp);
  require(slot0.read_logits(4U) == solo0.read_logits(4U) &&
              slot1.read_logits(8U) == solo1.read_logits(8U) &&
              slot2.read_logits(5U) == solo2.read_logits(5U),
          "run_packed_batch verify logits 0-ULP mismatch");

  (void)slot0.commit_prefix(2U);
  (void)slot1.commit_prefix(5U);
  (void)slot2.commit_prefix(0U);
  (void)solo0.commit_prefix(2U);
  (void)solo1.commit_prefix(5U);
  (void)solo2.commit_prefix(0U);

  const auto next_tok = make_tokens(1U, 950U, U32(cfg.vocab_size));
  require(slot0.run(next_tok, samp, true).logits ==
                  solo0.run(next_tok, samp, true).logits &&
              slot1.run(next_tok, samp, true).logits ==
                  solo1.run(next_tok, samp, true).logits &&
              slot2.run(next_tok, samp, true).logits ==
                  solo2.run(next_tok, samp, true).logits,
          "post-packed-verify commit continuation logits mismatch");

  // 5. Cross-slot `restore_shared` parity: checkpoint on slot0 -> restore_shared into slot2!
  const U64 prev_epoch = slot2.feature_epoch();
  auto ckpt = slot0.checkpoint();
  slot2.reset();
  slot2.restore_shared(ckpt, &slot0);
  require(slot2.feature_epoch() > prev_epoch &&
              slot2.position() == slot0.position(),
          "restore_shared did not advance feature_epoch or restore position");
  const auto cont_tok = make_tokens(3U, 999U, U32(cfg.vocab_size));
  const auto out0 = slot0.run(cont_tok, samp, true);
  const auto out2 = slot2.run(cont_tok, samp, true);
  require(out0.logits == out2.logits && out0.token == out2.token &&
              slot0.inspect_recurrent_state() == slot2.inspect_recurrent_state(),
          "restore_shared cross-slot continuation diverged from source slot");
  slot0.release(ckpt);

  return {{"primary_ws_bytes", plan_64k_primary.workspace_bytes},
          {"secondary_ws_bytes", plan_64k_secondary.workspace_bytes},
          {"passed", true}};
}

Json test_spec_controller_and_native_e2e(
    const std::filesystem::path &qwen_fixtures,
    const std::filesystem::path &dflash_fixtures,
    const std::filesystem::path &libs) {
  // 1. Unit test AdaptiveSpecController (MoE unique-expert coupon-collector + SpecSearch gating).
  const double u1 = detail::AdaptiveSpecController::expected_unique_experts(256, 8, 1);
  const double u8 = detail::AdaptiveSpecController::expected_unique_experts(256, 8, 8);
  const double u32 = detail::AdaptiveSpecController::expected_unique_experts(256, 8, 32);
  require(std::abs(u1 - 8.0) < 1e-6 && u8 > 40.0 && u8 < 50.0 &&
              u32 > 125.0 && u32 < 145.0,
          "MoE unique-expert formula out of calibrated H1-M range");

  detail::AdaptiveSpecController moe_ctrl(detail::SpecControllerConfig{
      .mode = detail::SpecSearchMode::On,
      .max_draft_tokens = 7,
      .max_slots = 4,
      .hidden_size = 2048,
      .num_layers = 40,
      .full_attn_layers = 10,
      .num_experts = 256,
      .experts_per_token = 8,
      .warmup_rounds = 2,
  });
  // Feed low-acceptance rounds on slot 0 -> should gate to K=0 and then emit periodic K=1 probe.
  for (U32 r = 0; r < 6; ++r) {
    moe_ctrl.observe_round(0, 7, 0, true, 5.5, 9.0, 0.8, 8);
  }
  const auto gated_dec = moe_ctrl.select_single_depth({0, 7, 512});
  require(gated_dec.k == 0 && gated_dec.gated,
          "AdaptiveSpecController failed to gate 0%-acceptance MoE stream");

  // 2. NativeBackend end-to-end test:
  //    (a) affine4_g128 KV cache AR == DFlash2 exact token parity!
  //    (b) B=3 concurrent batch + cross-slot SharedPrefixCache (`prefix_shared_hits >= 1`, `packed_batch_calls >= 1`)!
  const auto model_dir = qwen_fixtures / "affine/source";
  const auto model_art = qwen_fixtures / "affine/model.omlxm5";
  const auto draft_dir = dflash_fixtures / "source";
  const auto draft_art = dflash_fixtures / "model.omlxm5";
  const auto cfg = load_qwen_model_config(model_dir / "config.json");

  auto run_single_engine = [&](SpeculativeBackend spec_mode,
                               const std::string &kv_dtype,
                               const std::vector<std::int32_t> &prompt) {
    LoadOptions opts;
    opts.model_dir = model_dir.string();
    opts.draft_dir =
        (spec_mode == SpeculativeBackend::none) ? "" : draft_dir.string();
    opts.max_context = 1024;
    opts.max_concurrency = 1;
    opts.kv_dtype = kv_dtype;
    opts.sampling_contract = "position_coupled_v1";
    opts.speculative.backend = spec_mode;
    opts.speculative.draft_tokens =
        (spec_mode == SpeculativeBackend::none) ? 0 : 5;
    opts.path_overrides["artifact"] = model_art.string();
    if (spec_mode != SpeculativeBackend::none) {
      opts.path_overrides["draft_artifact"] = draft_art.string();
    }
    opts.path_overrides["kernel_directory"] = libs.string();
    auto eng = Engine::create(opts);
    SubmitRequest req;
    req.prompt_tokens = prompt;
    req.max_new_tokens = 18;
    req.sampling.temperature = 0.6f;
    req.sampling.top_k = 16;
    req.sampling.top_p = 0.9f;
    req.sampling.seed = 777;
    std::vector<std::int32_t> generated;
    const auto id = eng->submit(req);
    bool finished = false;
    while (!finished) {
      const auto evs = eng->poll_events(std::chrono::milliseconds(500));
      for (const auto &ev : evs) {
        if (ev.request_id != id)
          continue;
        if (ev.kind == EventKind::tokens) {
          generated.insert(generated.end(), ev.tokens.begin(), ev.tokens.end());
        } else if (ev.kind == EventKind::finished) {
          finished = true;
          break;
        } else if (ev.kind == EventKind::failed) {
          throw std::runtime_error("Engine request failed: " + ev.error);
        }
      }
    }
    eng->shutdown();
    return generated;
  };

  std::vector<std::int32_t> prompt70(70);
  for (std::size_t i = 0; i < prompt70.size(); ++i) {
    prompt70[i] = std::int32_t((37 * i + 11) % cfg.vocab_size);
  }
  const auto af4_ar =
      run_single_engine(SpeculativeBackend::none, "affine4_g128", prompt70);
  const auto af4_df =
      run_single_engine(SpeculativeBackend::dflash2, "affine4_g128", prompt70);
  require(af4_ar.size() == 18 && af4_ar == af4_df,
          "affine4_g128 KV cache AR != DFlash2 under position_coupled_v1");

  // 3. Multi-slot B=3 batch + cross-slot SharedPrefixCache + DFlash2 exact parity!
  const auto bf16_ar_ref =
      run_single_engine(SpeculativeBackend::none, "bf16", prompt70);

  LoadOptions b_opts;
  b_opts.model_dir = model_dir.string();
  b_opts.draft_dir = draft_dir.string();
  b_opts.max_context = 1024;
  b_opts.max_concurrency = 3;
  b_opts.kv_dtype = "bf16";
  b_opts.sampling_contract = "position_coupled_v1";
  b_opts.speculative.backend = SpeculativeBackend::dflash2;
  b_opts.speculative.draft_tokens = 5;
  b_opts.path_overrides["artifact"] = model_art.string();
  b_opts.path_overrides["draft_artifact"] = draft_art.string();
  b_opts.path_overrides["kernel_directory"] = libs.string();
  b_opts.path_overrides["batch_mode"] = "on";
  b_opts.path_overrides["prefix_cache_bytes"] = std::to_string(64ULL << 20);
  auto batch_eng = Engine::create(b_opts);

  // Warm prefix cache on slot 0 with prompt70 (pinned!).
  {
    SubmitRequest warm_req;
    warm_req.prompt_tokens = prompt70;
    warm_req.max_new_tokens = 2;
    warm_req.pin_prefix = true;
    warm_req.sampling.temperature = 0.0f;
    warm_req.sampling.top_k = 1;
    warm_req.sampling.seed = 1;
    const auto wid = batch_eng->submit(warm_req);
    bool warm_done = false;
    while (!warm_done) {
      const auto evs = batch_eng->poll_events(std::chrono::milliseconds(500));
      for (const auto &ev : evs) {
        if (ev.request_id == wid &&
            (ev.kind == EventKind::finished || ev.kind == EventKind::failed)) {
          require(ev.kind == EventKind::finished,
                  "warm request failed: " + ev.error);
          warm_done = true;
          break;
        }
      }
    }
  }

  // Submit 3 concurrent requests sharing prompt70 across slots 0, 1, 2 (distinct seeds on 1, 2)!
  std::array<std::uint64_t, 3> rids{};
  std::array<std::vector<std::int32_t>, 3> outs{};
  for (std::size_t s = 0; s < 3; ++s) {
    SubmitRequest req;
    req.prompt_tokens = prompt70;
    req.max_new_tokens = 18;
    req.sampling.temperature = 0.6f;
    req.sampling.top_k = 16;
    req.sampling.top_p = 0.9f;
    req.sampling.seed = 777 + s * 111;
    rids[s] = batch_eng->submit(req);
  }
  std::size_t done = 0;
  while (done < 3) {
    const auto evs = batch_eng->poll_events(std::chrono::milliseconds(500));
    for (const auto &ev : evs) {
      for (std::size_t s = 0; s < 3; ++s) {
        if (ev.request_id != rids[s])
          continue;
        if (ev.kind == EventKind::tokens) {
          outs[s].insert(outs[s].end(), ev.tokens.begin(), ev.tokens.end());
        } else if (ev.kind == EventKind::finished) {
          ++done;
        } else if (ev.kind == EventKind::failed) {
          throw std::runtime_error("concurrent request failed: " + ev.error);
        }
      }
    }
  }
  require(outs[0] == bf16_ar_ref,
          "concurrent B=3 cross-slot COW + packed batch DFlash2 diverged from AR oracle!");
  const auto st = batch_eng->stats();
  batch_eng->shutdown();
  require(st.prefix_hits >= 3 && st.prefix_shared_hits >= 2,
          "cross-slot SharedPrefixCache did not record prefix_shared_hits");
  require(st.packed_batch_calls >= 1,
          "NativeBackend did not execute packed_batch_calls on B=3");

  return {{"moe_u8", u8},
          {"moe_u32", u32},
          {"prefix_hits", st.prefix_hits},
          {"prefix_shared_hits", st.prefix_shared_hits},
          {"packed_batch_calls", st.packed_batch_calls},
          {"packed_batch_tokens", st.packed_batch_tokens},
          {"passed", true}};
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 6) {
      throw std::invalid_argument(
          "usage: omlx_m5_w2_ninfer_features_tests <qwen_fixtures> "
          "<dflash_fixtures> <metallib_dir> <python_exe> <repo_root>");
    }
    const std::filesystem::path qwen_fixtures = argv[1];
    const std::filesystem::path dflash_fixtures = argv[2];
    const std::filesystem::path libs = argv[3];
    const std::filesystem::path python_exe = argv[4];
    const std::filesystem::path repo_root = argv[5];

    const auto tmp_dir =
        std::filesystem::temp_directory_path() /
        ("omlx_m5_w2_features_" + std::to_string(::getpid()));
    std::filesystem::remove_all(tmp_dir);
    std::filesystem::create_directories(tmp_dir);
    struct Cleanup {
      std::filesystem::path dir;
      ~Cleanup() {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
      }
    } cleanup{tmp_dir};

    auto ctx = MetalContext::create();
    const auto r_tuning = test_silicon_tuning(ctx, repo_root, tmp_dir);
    const auto r_qda = test_qda_q3_g128_and_pretile(qwen_fixtures, libs,
                                                    python_exe, repo_root,
                                                    tmp_dir);
    const auto r_packed =
        test_memory_squeeze_and_packed_batch(qwen_fixtures, libs);
    const auto r_e2e =
        test_spec_controller_and_native_e2e(qwen_fixtures, dflash_fixtures, libs);

    std::cout << Json({{"schema", "omlx-m5-w2-ninfer-features-v1"},
                       {"silicon_tuning", r_tuning},
                       {"qda_q3_g128_pretile", r_qda},
                       {"memory_squeeze_and_packed_batch", r_packed},
                       {"spec_controller_and_native_e2e", r_e2e},
                       {"passed", true}})
                     .dump(2)
              << '\n';
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
