#include "omlx_m5/artifact.h"
#include "omlx_m5/engine.h"
#include "omlx_m5/execution_contract.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/mtp_forward.h"
#include "omlx_m5/qwen_forward.h"
#include "../src/spec_controller.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace omlx::m5;
using Json = nlohmann::json;
using U32 = std::uint32_t;
using U64 = std::uint64_t;

namespace {

void require(bool ok, std::string_view message) {
  if (!ok) {
    throw std::runtime_error("M5-MTP TEST FAIL: " + std::string(message));
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
  const std::string s = p.string();
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

std::vector<std::int32_t> make_prompt_i32(std::size_t count, U32 seed_offset,
                                          U32 vocab) {
  std::vector<std::int32_t> out(count);
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = std::int32_t((73U * (U32(i) + seed_offset) + 19U) % vocab);
  }
  return out;
}

struct RunOutcome {
  std::vector<std::int32_t> tokens;
  std::string finish_reason;
  EventKind terminal_kind = EventKind::finished;
  std::optional<RequestStats> request_stats;
};

RunOutcome collect_request(Engine &engine, U64 request_id) {
  RunOutcome out;
  for (;;) {
    auto events = engine.poll_events(std::chrono::milliseconds(50));
    for (auto &ev : events) {
      if (ev.request_id != request_id) {
        continue;
      }
      if (ev.kind == EventKind::tokens) {
        out.tokens.insert(out.tokens.end(), ev.tokens.begin(), ev.tokens.end());
      } else {
        out.terminal_kind = ev.kind;
        out.finish_reason = ev.finish_reason;
        out.request_stats = std::move(ev.request_stats);
        if (ev.kind == EventKind::failed) {
          throw std::runtime_error("Engine request failed: " + ev.error);
        }
        return out;
      }
    }
  }
}

void generate_mtp_fixtures_via_python(
    const std::filesystem::path &python_exe,
    const std::filesystem::path &repo_root,
    const std::filesystem::path &qwen_fixtures,
    const std::filesystem::path &out_dir) {
  const auto script_path = out_dir / "build_mtp_fixtures.py";
  {
    std::ofstream f(script_path);
    f << R"PY(
import json
import math
import sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

repo_root = Path(sys.argv[1])
qwen_fix = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
sys.path.insert(0, str(repo_root / "tools"))
import m5_artifact
from mlx_lm.models.qwen3_5 import Attention, MLP, SparseMoeBlock, TextModelArgs, create_attention_mask
from mlx_lm.models.cache import KVCache

def make_mtp_ckpt(base_source_dir: Path, dst_dir: Path, *, mode: str, moe: bool,
                  prefix: str, legacy_unshifted: bool = False, seed: int = 9901):
    dst_dir.mkdir(parents=True, exist_ok=True)
    raw_cfg = json.loads((base_source_dir / "config.json").read_text())
    tcfg = dict(raw_cfg.get("text_config") or raw_cfg)
    args = TextModelArgs.from_dict(tcfg)

    class TempMtp(nn.Module):
        def __init__(self, a):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(a.hidden_size, eps=a.rms_norm_eps)
            self.pre_fc_norm_embedding = nn.RMSNorm(a.hidden_size, eps=a.rms_norm_eps)
            self.fc = nn.Linear(a.hidden_size * 2, a.hidden_size, bias=False)
            self.self_attn_layer_in = nn.RMSNorm(a.hidden_size, eps=a.rms_norm_eps)
            self.self_attn = Attention(a)
            self.post_attn = nn.RMSNorm(a.hidden_size, eps=a.rms_norm_eps)
            self.mlp = SparseMoeBlock(a) if moe else MLP(a.hidden_size, a.intermediate_size)
            self.norm = nn.RMSNorm(a.hidden_size, eps=a.rms_norm_eps)

    m = TempMtp(args)
    rng = np.random.default_rng(seed)
    init_w = []
    for name, val in tree_flatten(m.parameters()):
        shp = tuple(val.shape)
        if "norm" in name or name in ("self_attn_layer_in.weight", "post_attn.weight"):
            arr = 1.15 + rng.normal(0.0, 0.05, shp)
        else:
            arr = rng.normal(0.0, 0.55 / math.sqrt(shp[-1]), shp)
        init_w.append((name, mx.array(arr.astype(np.float32)).astype(mx.bfloat16)))
    m.load_weights(init_w)

    protected = {"fc": False}
    if moe:
        protected["mlp.gate"] = {"bits": 8, "group_size": 64, "mode": "affine"}
        protected["mlp.shared_expert_gate"] = {"bits": 8, "group_size": 64, "mode": "affine"}

    def qpred(path, mod):
        if path in protected:
            return protected[path]
        return hasattr(mod, "to_quantized")

    nn.quantize(
        m,
        group_size=64 if mode == "affine" else 32,
        bits=4,
        mode=mode,
        class_predicate=qpred,
    )
    m.eval()

    mapped = {}
    for name, val in tree_flatten(m.parameters()):
        if name.startswith("self_attn_layer_in."):
            sub = "layers.0.input_layernorm." + name.removeprefix("self_attn_layer_in.")
        elif name.startswith("self_attn."):
            sub = "layers.0." + name
        elif name.startswith("post_attn."):
            sub = "layers.0.post_attention_layernorm." + name.removeprefix("post_attn.")
        elif name.startswith("mlp."):
            sub = "layers.0." + name
        else:
            sub = name
        full_name = prefix + sub
        if legacy_unshifted and any(
            sub == k
            for k in (
                "layers.0.self_attn.q_norm.weight",
                "layers.0.self_attn.k_norm.weight",
                "layers.0.post_attention_layernorm.weight",
                "norm.weight",
            )
        ):
            val = (val.astype(mx.float32) - 1.0).astype(mx.bfloat16)
        mapped[full_name] = val

    mx.save_safetensors(str(dst_dir / "model.safetensors"), mapped)
    out_cfg = dict(raw_cfg)
    qtable = dict(out_cfg.get("quantization") or {})
    if moe:
        qtable[prefix + "layers.0.mlp.gate"] = {"bits": 8, "group_size": 64, "mode": "affine"}
        qtable[prefix + "layers.0.mlp.shared_expert_gate"] = {"bits": 8, "group_size": 64, "mode": "affine"}
    out_cfg["quantization"] = qtable
    (dst_dir / "config.json").write_text(json.dumps(out_cfg, indent=2) + "\n")
    (dst_dir / "REVISION").write_text("mtp-fixture-v1\n")
    return m, args, mapped

# 1. Dense affine MTP (29 tensors: shifted + legacy unshifted twin)
m_aff, args_aff, mapped_aff = make_mtp_ckpt(
    qwen_fix / "affine/source", out_dir / "mtp_affine_ckpt",
    mode="affine", moe=False, prefix="language_model.mtp.", legacy_unshifted=False, seed=4105
)
assert len(mapped_aff) == 29, f"expected 29 dense affine MTP tensors, got {len(mapped_aff)}"

make_mtp_ckpt(
    qwen_fix / "affine/source", out_dir / "mtp_affine_legacy_ckpt",
    mode="affine", moe=False, prefix="language_model.mtp.", legacy_unshifted=True, seed=4105
)

# Compute MLX ground-truth oracle for m_aff chained K=4 on prompt [5, 12, 33, 7]
from mlx_lm.models.qwen3_5 import TextModel
base_cfg_raw = json.loads((qwen_fix / "affine/source/config.json").read_text())
base_tm = TextModel(TextModelArgs.from_dict(base_cfg_raw["text_config"]))
base_qpred = lambda p, mod: hasattr(mod, "to_quantized")
nn.quantize(base_tm, group_size=64, bits=4, mode="affine", class_predicate=base_qpred)
base_weights = mx.load(str(qwen_fix / "affine/source/model.safetensors"))
base_tm.load_weights([(k.removeprefix("language_model."), v) for k, v in base_weights.items()])
base_tm.eval()

prompt_ids = [5, 12, 33, 7]
x_in = mx.array([prompt_ids], dtype=mx.int32)
trunk_h = base_tm.model(x_in)
trunk_normed = base_tm.model.norm(trunk_h)
logits_p = base_tm.lm_head(trunk_normed) if hasattr(base_tm, "lm_head") else base_tm.model.embed_tokens.as_linear(trunk_normed)
anchor_0 = int(mx.argmax(logits_p[0, -1]).item())

def run_mtp_step(m_head, cond_h, tok_ids, cache):
    emb = base_tm.model.embed_tokens(tok_ids)
    e = m_head.pre_fc_norm_embedding(emb)
    h = m_head.pre_fc_norm_hidden(cond_h)
    fused = m_head.fc(mx.concatenate([e, h], axis=-1))
    mask = create_attention_mask(fused, cache)
    r = m_head.self_attn(m_head.self_attn_layer_in(fused), mask, cache)
    h2 = fused + r
    out = m_head.norm(h2 + m_head.mlp(m_head.post_attn(h2)))
    return out

mtp_cache = KVCache()
# Prime prompt pairs (trunk_normed[0..2], prompt[1..3])
_ = run_mtp_step(m_aff, trunk_normed[:, :-1, :], mx.array([prompt_ids[1:]], dtype=mx.int32), mtp_cache)
# Cycle 1 d=0..3 chain
cur_h = trunk_normed[:, -1:, :]
cur_tok = anchor_0
oracle_drafts = []
for _ in range(4):
    h_out = run_mtp_step(m_aff, cur_h, mx.array([[cur_tok]], dtype=mx.int32), mtp_cache)
    lg = base_tm.lm_head(h_out) if hasattr(base_tm, "lm_head") else base_tm.model.embed_tokens.as_linear(h_out)
    nxt = int(mx.argmax(lg[0, -1]).item())
    oracle_drafts.append(nxt)
    cur_h = h_out
    cur_tok = nxt

(out_dir / "oracle.json").write_text(json.dumps({
    "prompt": prompt_ids,
    "anchor_0": anchor_0,
    "drafts": oracle_drafts,
}, indent=2) + "\n")

# 2. Dense MXFP4 MTP with "mtp." prefix (RedHatAI naming convention, without "language_model.")
make_mtp_ckpt(
    qwen_fix / "mxfp4/source", out_dir / "mtp_mxfp4_ckpt",
    mode="mxfp4", moe=False, prefix="mtp.", legacy_unshifted=False, seed=4202
)

# 3. Dense HF zero-centered MTP (for target_norm_offset == 1.0f test)
_, _, mapped_hf = make_mtp_ckpt(
    qwen_fix / "hf-mxfp4/source", out_dir / "mtp_hf_ckpt",
    mode="mxfp4", moe=False, prefix="language_model.mtp.", legacy_unshifted=False, seed=4303
)
for k in list(mapped_hf.keys()):
    if k.endswith("norm.weight"):
        mapped_hf[k] = (mapped_hf[k].astype(mx.float32) - 1.0).astype(mx.bfloat16)
mx.save_safetensors(str(out_dir / "mtp_hf_ckpt/model.safetensors"), mapped_hf)

# 4. 256-Expert MoE MTP (Qwen3.6-35B-A3B topology)
make_mtp_ckpt(
    qwen_fix / "moe-affine/source", out_dir / "mtp_moe_ckpt",
    mode="affine", moe=True, prefix="language_model.mtp.", legacy_unshifted=False, seed=4404
)
)PY";
  }

  const std::string cmd = shell_quote(python_exe) + " " +
                          shell_quote(script_path) + " " +
                          shell_quote(repo_root) + " " +
                          shell_quote(qwen_fixtures) + " " +
                          shell_quote(out_dir);
  run_cmd(cmd);
}

Json test_artifact_cli_roundtrip(const std::filesystem::path &python_exe,
                                 const std::filesystem::path &repo_root,
                                 const std::filesystem::path &qwen_fixtures,
                                 const std::filesystem::path &work_dir) {
  const auto tool = repo_root / "tools/m5_artifact.py";
  const auto base_aff = qwen_fixtures / "affine/model.omlxm5";
  const auto mtp_only = work_dir / "mtp_affine_only.omlxm5";
  const auto bundled_aff = work_dir / "qwen_affine_bundled_mtp.omlxm5";
  const auto reextracted = work_dir / "mtp_reextracted.omlxm5";
  const auto pretiled_mtp = work_dir / "mtp_pretiled.omlxm5";

  // 1. extract-mtp from checkpoint directory
  run_cmd(shell_quote(python_exe) + " " + shell_quote(tool) +
          " extract-mtp --source " + shell_quote(work_dir / "mtp_affine_ckpt") +
          " --output " + shell_quote(mtp_only) +
          " --source-repo test/mtp-affine > /dev/null");

  const auto mtp_art = ArtifactModel::open(mtp_only, true);
  require(mtp_art.objects().size() == 15U,
          "dense affine MTP sub-artifact must have 15 objects (from 29 tensors)");
  require(has_mtp_weights(mtp_art), "has_mtp_weights(mtp_art) false");

  // 2. bundle-mtp onto base backbone .omlxm5 without mutating backbone plane digests
  run_cmd(shell_quote(python_exe) + " " + shell_quote(tool) +
          " bundle-mtp --base " + shell_quote(base_aff) + " --mtp " +
          shell_quote(mtp_only) + " --output " + shell_quote(bundled_aff) +
          " > /dev/null");

  const auto base_art = ArtifactModel::open(base_aff, true);
  const auto bundled_art = ArtifactModel::open(bundled_aff, true);
  require(!has_mtp_weights(base_art), "base affine unexpectedly had MTP weights");
  require(has_mtp_weights(bundled_art), "bundled affine missing MTP weights");
  require(bundled_art.objects().size() ==
              base_art.objects().size() + mtp_art.objects().size(),
          "bundled object count mismatch");

  for (const auto &base_obj : base_art.objects()) {
    const auto &bundled_obj = bundled_art.object(base_obj.name);
    require(bundled_obj.object_sha256 == base_obj.object_sha256,
            "bundle-mtp mutated base object sha256: " + base_obj.name);
  }

  // 3. extract-mtp from bundled .omlxm5 + --nax-pretile check
  run_cmd(shell_quote(python_exe) + " " + shell_quote(tool) +
          " extract-mtp --source " + shell_quote(bundled_aff) + " --output " +
          shell_quote(reextracted) + " > /dev/null");
  const auto re_art = ArtifactModel::open(reextracted, true);
  require(re_art.objects().size() == 15U, "re-extracted MTP object count != 15");
  for (std::size_t i = 0; i < mtp_art.objects().size(); ++i) {
    require(re_art.objects()[i].name == mtp_art.objects()[i].name &&
                re_art.objects()[i].object_sha256 ==
                    mtp_art.objects()[i].object_sha256,
            "re-extracted MTP object digest mismatch");
  }

  run_cmd(shell_quote(python_exe) + " " + shell_quote(tool) +
          " extract-mtp --source " + shell_quote(bundled_aff) + " --output " +
          shell_quote(pretiled_mtp) + " --nax-pretile > /dev/null");
  const auto pt_art = ArtifactModel::open(pretiled_mtp, true);
  require(pt_art.summary().payload_verified && has_mtp_weights(pt_art),
          "pretiled MTP artifact verification failed");

  // Also bundle MXFP4, HF, and MoE test artifacts for downstream tests!
  run_cmd(shell_quote(python_exe) + " " + shell_quote(tool) +
          " bundle-mtp --base " + shell_quote(qwen_fixtures / "mxfp4/model.omlxm5") +
          " --mtp " + shell_quote(work_dir / "mtp_mxfp4_ckpt") + " --output " +
          shell_quote(work_dir / "qwen_mxfp4_bundled_mtp.omlxm5") +
          " > /dev/null");
  run_cmd(shell_quote(python_exe) + " " + shell_quote(tool) +
          " bundle-mtp --base " + shell_quote(qwen_fixtures / "hf-mxfp4/model.omlxm5") +
          " --mtp " + shell_quote(work_dir / "mtp_hf_ckpt") + " --output " +
          shell_quote(work_dir / "qwen_hf_bundled_mtp.omlxm5") +
          " > /dev/null");
  run_cmd(shell_quote(python_exe) + " " + shell_quote(tool) +
          " bundle-mtp --base " + shell_quote(qwen_fixtures / "moe-affine/model.omlxm5") +
          " --mtp " + shell_quote(work_dir / "mtp_moe_ckpt") + " --output " +
          shell_quote(work_dir / "qwen_moe_bundled_mtp.omlxm5") +
          " > /dev/null");
  run_cmd(shell_quote(python_exe) + " " + shell_quote(tool) +
          " extract-mtp --source " +
          shell_quote(work_dir / "mtp_affine_legacy_ckpt") + " --output " +
          shell_quote(work_dir / "mtp_affine_legacy.omlxm5") + " > /dev/null");

  return {{"mtp_objects", mtp_art.objects().size()},
          {"bundled_objects", bundled_art.objects().size()},
          {"base_digests_preserved", true}};
}

Json test_mlx_oracle_and_rollback(const MetalContext &ctx,
                                  const std::filesystem::path &qwen_fixtures,
                                  const std::filesystem::path &libraries,
                                  const std::filesystem::path &work_dir) {
  const auto oracle =
      Json::parse(std::ifstream(work_dir / "oracle.json"));
  std::vector<U32> prompt;
  for (int tok : oracle.at("prompt").get<std::vector<int>>()) {
    prompt.push_back(U32(tok));
  }
  const U32 expected_anchor = U32(oracle.at("anchor_0").get<int>());
  const auto expected_drafts = oracle.at("drafts").get<std::vector<int>>();

  const auto cfg =
      load_qwen_model_config(qwen_fixtures / "affine/source/config.json");
  const auto bundled_art =
      ArtifactModel::open(work_dir / "qwen_affine_bundled_mtp.omlxm5", true);
  auto mat_plan = plan_model_materialization(
      bundled_art, ctx,
      {.components = {ModelComponent::text, ModelComponent::mtp},
       .require_verified_payload = true});
  const auto weights =
      materialize_model(bundled_art, std::move(mat_plan), ctx);
  const auto catalog = build_qwen_layer_catalog(bundled_art, cfg, 2048U, 64U);

  QwenForward target(ctx, weights, catalog, cfg, 2048U, libraries, {}, false,
                     true, false, {}, false, false, {},
                     LinearNumericPolicy::RowExactPortableV1,
                     SamplingContract::PositionCoupledV1,
                     AttentionNumericPolicy::RowExactV1);

  MtpForward mtp(ctx, weights, cfg, 2048U, target.embedding_weight(),
                 target.head_weight(), target.final_norm_weight(),
                 target.trunk_hidden_buffer(),
                 target.normalization_weight_offset(), libraries, false,
                 LinearNumericPolicy::RowExactPortableV1,
                 SamplingContract::PositionCoupledV1, "bf16", &catalog,
                 &weights);

  require(mtp.prefix() == "language_model.mtp.", "unexpected MTP prefix");
  require(!mtp.is_moe_layer(), "dense MTP reported as MoE");
  require(!mtp.norm_offsets().repaired_any,
          "shifted MTP norms unexpectedly triggered repair");

  // 1. Verify MLX ground-truth oracle parity & single-CB chain execution!
  const GpuSamplingOptions greedy{.temperature = 0.0F,
                                  .top_k = 0U,
                                  .top_p = 1.0F,
                                  .seed = 0ULL,
                                  .contract =
                                      SamplingContract::PositionCoupledV1};
  const auto pf_res = target.run(prompt, greedy);
  require(pf_res.token == expected_anchor,
          "QwenForward greedy anchor differs from MLX oracle");
  static_cast<void>(mtp.prime_prefill_chunk(prompt, true, true));
  require(mtp.committed_tokens() == prompt.size() - 1U &&
              mtp.has_pending_trunk(),
          "MTP post-priming frontier mismatch");

  const std::array<U32, 1> c1_toks{expected_anchor};
  const auto prop1 =
      mtp.propose(c1_toks, target.position(), 4U, greedy, true, true);
  if (prop1.selection.tokens != expected_drafts) {
    Json diag{{"got", prop1.selection.tokens}, {"expected", expected_drafts}};
    require(false,
            "MtpForward K=4 chained draft tokens differ from MLX MTPModule oracle: " +
                diag.dump());
  }
  require(prop1.draft_dispatches > 0U, "MtpForward recorded zero dispatches");
  // committed_tokens advanced by c=1 (to 4), while kv_tokens holds 4 + 3 = 7!
  require(mtp.committed_tokens() == 4U && mtp.kv_tokens() == 7U,
          "MTP speculative KV reservation mismatch after K=4 chain");

  // 2. Verify legacy unshifted norm auto-repair (repair_legacy_head_norms)!
  const auto legacy_art =
      ArtifactModel::open(work_dir / "mtp_affine_legacy.omlxm5", true);
  auto leg_plan = plan_model_materialization(
      legacy_art, ctx,
      {.components = {ModelComponent::mtp}, .require_verified_payload = true});
  const auto leg_weights =
      materialize_model(legacy_art, std::move(leg_plan), ctx);

  // Re-run prompt on fresh target so trunk_hidden holds prompt[0..3] again.
  target.reset();
  static_cast<void>(target.run(prompt, greedy));
  MtpForward mtp_legacy(
      ctx, leg_weights, cfg, 2048U, target.embedding_weight(),
      target.head_weight(), target.final_norm_weight(),
      target.trunk_hidden_buffer(), target.normalization_weight_offset(),
      libraries, false, LinearNumericPolicy::RowExactPortableV1,
      SamplingContract::PositionCoupledV1, "bf16", &catalog, &weights);
  const auto &noff = mtp_legacy.norm_offsets();
  require(noff.repaired_any && noff.pre_fc_norm_embedding == 0.0F &&
              noff.pre_fc_norm_hidden == 0.0F && noff.input_layernorm == 0.0F &&
              noff.q_norm == 1.0F && noff.k_norm == 1.0F &&
              noff.post_attention_layernorm == 1.0F && noff.final_norm == 1.0F,
          "repair_legacy_head_norms failed to detect exact 4 unshifted norms");
  static_cast<void>(mtp_legacy.prime_prefill_chunk(prompt, true, true));
  const auto prop_legacy =
      mtp_legacy.propose(c1_toks, target.position(), 4U, greedy, true, true);
  require(prop_legacy.selection.tokens == expected_drafts,
          "repaired legacy MTP norms did not match pre-shifted MTP tokens");

  // 3. Verify HF zero-centered checkpoint norm offsets (all +1.0f)!
  {
    const auto hf_cfg =
        load_qwen_model_config(qwen_fixtures / "hf-mxfp4/source/config.json");
    const auto hf_art =
        ArtifactModel::open(work_dir / "qwen_hf_bundled_mtp.omlxm5", true);
    auto hf_plan = plan_model_materialization(
        hf_art, ctx,
        {.components = {ModelComponent::text, ModelComponent::mtp},
         .require_verified_payload = true});
    const auto hf_weights = materialize_model(hf_art, std::move(hf_plan), ctx);
    const auto hf_cat = build_qwen_layer_catalog(hf_art, hf_cfg, 1024U, 64U);
    QwenForward hf_target(ctx, hf_weights, hf_cat, hf_cfg, 1024U, libraries);
    require(hf_target.normalization_weight_offset() == 1.0F,
            "HF target norm offset != 1.0f");
    MtpForward hf_mtp(ctx, hf_weights, hf_cfg, 1024U,
                      hf_target.embedding_weight(), hf_target.head_weight(),
                      hf_target.final_norm_weight(),
                      hf_target.trunk_hidden_buffer(),
                      hf_target.normalization_weight_offset(), libraries, false,
                      LinearNumericPolicy::Legacy,
                      SamplingContract::LegacyRejectionV1, "bf16", &hf_cat,
                      &hf_weights);
    const auto &hfo = hf_mtp.norm_offsets();
    require(!hfo.repaired_any && hfo.pre_fc_norm_embedding == 1.0F &&
                hfo.pre_fc_norm_hidden == 1.0F && hfo.input_layernorm == 1.0F &&
                hfo.q_norm == 1.0F && hfo.k_norm == 1.0F &&
                hfo.post_attention_layernorm == 1.0F && hfo.final_norm == 1.0F,
            "HF zero-centered MTP norms must all use +1.0f offset");
  }

  // 4. Verify multi-subchunk (>64) & multi-chunk prompt priming seam invariance
  //    + multi-round O(1) truncate(hist_offset) rollback!
  std::vector<U32> long_prompt(135);
  for (std::size_t i = 0; i < long_prompt.size(); ++i) {
    long_prompt[i] = U32((31U * i + 7U) % U32(cfg.vocab_size));
  }

  target.reset();
  mtp.reset();
  const auto single_pf = target.run(long_prompt, greedy);
  static_cast<void>(mtp.prime_prefill_chunk(long_prompt, true, true));
  require(mtp.committed_tokens() == 134U && mtp.kv_tokens() == 134U,
          "135-token single-chunk priming frontier mismatch");
  const std::array<U32, 1> anchor_long{single_pf.token};
  const auto prop_single_chunk =
      mtp.propose(anchor_long, target.position(), 4U, greedy, true, true);

  // Now prefill & prime the exact same 135 tokens split across 2 chunks (64 + 71)!
  target.reset();
  mtp.reset();
  const auto cA = std::span<const U32>(long_prompt).first(64U);
  const auto cB = std::span<const U32>(long_prompt).subspan(64U);
  static_cast<void>(target.run(cA, greedy));
  static_cast<void>(mtp.prime_prefill_chunk(cA, true, true));
  require(mtp.committed_tokens() == 63U && mtp.has_pending_trunk(),
          "chunk A priming frontier mismatch");
  const auto split_pf = target.run(cB, greedy);
  static_cast<void>(mtp.prime_prefill_chunk(cB, false, true));
  require(split_pf.token == single_pf.token,
          "QwenForward 64+71 prefill anchor mismatch");
  require(mtp.committed_tokens() == 134U && mtp.has_pending_trunk(),
          "chunk B priming frontier mismatch");
  const auto prop_two_chunk =
      mtp.propose(anchor_long, target.position(), 4U, greedy, true, true);
  require(prop_two_chunk.selection.tokens == prop_single_chunk.selection.tokens,
          "multi-chunk prompt priming seam diverged from single-chunk priming");

  // Simulate multi-round partial acceptances: m=0 (c=1), m=2 (c=3), m=4 (c=5)
  for (U32 sim_c : {1U, 3U, 5U, 2U}) {
    const U32 prev_hist = mtp.committed_tokens();
    require(mtp.kv_tokens() == prev_hist + 3U,
            "expected 3 speculative KV tail tokens before rollback");
    std::vector<U32> verify_in(sim_c);
    for (U32 r = 0; r < sim_c; ++r) {
      verify_in[r] = (prev_hist + r) % U32(cfg.vocab_size);
    }
    static_cast<void>(target.verify(verify_in, greedy));
    static_cast<void>(target.commit_prefix(sim_c));
    const auto p_next =
        mtp.propose(verify_in, target.position(), 4U, greedy, true, true);
    require(p_next.selection.tokens.size() == 4U, "K=4 proposal size != 4");
    require(mtp.committed_tokens() == prev_hist + sim_c,
            "hist_offset did not advance by exact committed count after rollback");
    require(mtp.kv_tokens() == prev_hist + sim_c + 3U,
            "kv_tokens did not reflect truncated + re-appended speculative tail");
  }

  // 5. Regression test for MIVE-6: PositionCoupledV1 proposal_k up to 64 and
  //    FP32 top_p truncation consistency (sampled token must never get q == 0).
  {
    const auto verify_step_dist = [](const SelectorStep &step, U32 expected_k,
                                     const std::string &label) {
      require(step.candidate_ids.size() == expected_k &&
                  step.probabilities.size() == expected_k,
              label + ": unexpected proposal_k size");
      double sum = 0.0;
      float q_tok = -1.0F;
      for (std::size_t i = 0; i < step.candidate_ids.size(); ++i) {
        const float pr = step.probabilities[i];
        require(std::isfinite(pr) && pr >= 0.0F,
                label + ": nonfinite or negative probability");
        sum += double(pr);
        if (step.candidate_ids[i] == step.token) {
          q_tok = pr;
        }
      }
      require(std::abs(sum - 1.0) < 1.0e-3,
              label + ": proposal probabilities not normalized");
      require(q_tok > 0.0F,
              label + ": sampled draft token missing or zeroed (q_token <= 0)");
    };

    target.reset();
    mtp.reset();
    const auto pf_tok = target.run(prompt, greedy).token;
    static_cast<void>(mtp.prime_prefill_chunk(prompt, true, true));
    const std::array<U32, 1> anchor_c1{pf_tok};

    // PositionCoupledV1 with top_k = 48 (> 32) and tokens_only = false.
    const GpuSamplingOptions pc48{.temperature = 1.0F,
                                  .top_k = 48U,
                                  .top_p = 0.95F,
                                  .seed = 42ULL,
                                  .contract =
                                      SamplingContract::PositionCoupledV1};
    const auto pc_res =
        mtp.propose(anchor_c1, target.position(), 4U, pc48, true, false);
    for (U32 d = 0U; d < 4U; ++d) {
      verify_step_dist(pc_res.selection.steps[d], 48U,
                       "PositionCoupledV1 top_k=48 single step=" +
                           std::to_string(d));
    }

    // LegacyRejectionV1 razor-edge top_p boundary sweep.
    mtp.reset();
    static_cast<void>(mtp.prime_prefill_chunk(prompt, true, true));
    const GpuSamplingOptions leg_full{.temperature = 1.0F,
                                      .top_k = 32U,
                                      .top_p = 1.0F,
                                      .seed = 1ULL,
                                      .contract =
                                          SamplingContract::LegacyRejectionV1};
    const auto baseline =
        mtp.propose(anchor_c1, target.position(), 4U, leg_full, true, false);
    std::array<float, 4> cums{};
    float acc = 0.0F;
    for (U32 i = 0U; i < 4U; ++i) {
      acc += baseline.selection.steps[0].probabilities[i];
      cums[i] = std::min(acc, std::nextafter(1.0F, 0.0F));
    }
    const std::array<float, 8> boundary_ps{
        cums[0],
        std::max(0.0F, std::nextafter(cums[0], 0.0F)),
        std::min(1.0F, std::nextafter(cums[0], 1.0F)),
        cums[2],
        std::max(0.0F, std::nextafter(cums[2], 0.0F)),
        std::min(1.0F, std::nextafter(cums[2], 1.0F)),
        0.5F,
        0.9F};
    for (const float tp : boundary_ps) {
      for (const U64 seed : {1ULL, 7ULL, 42ULL, 1337ULL}) {
        mtp.reset();
        static_cast<void>(mtp.prime_prefill_chunk(prompt, true, true));
        const GpuSamplingOptions leg_edge{
            .temperature = 1.0F,
            .top_k = 32U,
            .top_p = tp,
            .seed = seed,
            .contract = SamplingContract::LegacyRejectionV1};
        const auto edge_res =
            mtp.propose(anchor_c1, target.position(), 4U, leg_edge, true, false);
        for (U32 d = 0U; d < 4U; ++d) {
          verify_step_dist(edge_res.selection.steps[d], 32U,
                           "LegacyRejectionV1 boundary top_p single step=" +
                               std::to_string(d));
        }
      }
    }

    // Packed batch (B = 2): slot 0 LegacyRejectionV1 boundary top_p,
    // slot 1 PositionCoupledV1 top_k = 48, both tokens_only = false.
    MtpForward mtp_slot2(
        ctx, weights, cfg, 2048U, target.embedding_weight(),
        target.head_weight(), target.final_norm_weight(),
        target.trunk_hidden_buffer(), target.normalization_weight_offset(),
        libraries, false, LinearNumericPolicy::RowExactPortableV1,
        SamplingContract::PositionCoupledV1, "bf16", &catalog, &weights);
    mtp.reset();
    mtp_slot2.reset();
    static_cast<void>(mtp.prime_prefill_chunk(prompt, true, true));
    static_cast<void>(mtp_slot2.prime_prefill_chunk(prompt, true, true));
    const GpuSamplingOptions pb_leg{
        .temperature = 1.0F,
        .top_k = 32U,
        .top_p = std::min(1.0F, std::nextafter(cums[2], 1.0F)),
        .seed = 99ULL,
        .contract = SamplingContract::LegacyRejectionV1};
    const std::array<MtpPackedSlotSlice, 2> pb_slices{{
        {.slot_mtp = &mtp,
         .committed_tokens = anchor_c1,
         .target_position = target.position(),
         .draft_tokens = 4U,
         .sampling = pb_leg,
         .tokens_only = false},
        {.slot_mtp = &mtp_slot2,
         .committed_tokens = anchor_c1,
         .target_position = target.position(),
         .draft_tokens = 4U,
         .sampling = pc48,
         .tokens_only = false},
    }};
    const auto pb_res = mtp.propose_packed_batch(pb_slices, true);
    require(pb_res.slot_results.size() == 2U, "packed batch slot count != 2");
    for (U32 d = 0U; d < 4U; ++d) {
      verify_step_dist(pb_res.slot_results[0].selection.steps[d], 32U,
                       "packed LegacyRejectionV1 step=" + std::to_string(d));
      verify_step_dist(pb_res.slot_results[1].selection.steps[d], 48U,
                       "packed PositionCoupledV1 top_k=48 step=" +
                           std::to_string(d));
    }
  }

  return {{"oracle_drafts", expected_drafts},
          {"legacy_norm_repair_verified", true},
          {"priming_seam_parity", true},
          {"multi_round_rollback_verified", true},
          {"sampling_postprocess_parity_verified", true}};
}

Json test_depth_controller() {
  MtpDepthController fixed(4U, true);
  require(fixed.select_depth(4U) == 4U && fixed.select_depth(2U) == 2U,
          "fixed MtpDepthController depth mismatch");
  fixed.observe(4U, 0U, 25.0, true);
  require(fixed.select_depth(4U) == 4U && !fixed.should_exit(),
          "fixed MtpDepthController must never mutate depth or exit");

  // Adaptive controller: warmup sweeps 4, 3, 2, 1, 0, 0, 0
  MtpDepthController adaptive(4U, false);
  const std::array<U32, 7> expected_warmup{4U, 3U, 2U, 1U, 0U, 0U, 0U};
  for (std::size_t i = 0; i < expected_warmup.size(); ++i) {
    const U32 d = adaptive.select_depth(4U);
    require(d == expected_warmup[i],
            "adaptive warmup sequence mismatch at step " + std::to_string(i));
    // High acceptance + flat cycle cost -> deep speculation wins!
    adaptive.observe(d, d, d == 0U ? 10.0 : (11.0 + 0.5 * double(d)), true);
  }
  require(adaptive.select_depth(4U) == 4U && !adaptive.should_exit(),
          "adaptive controller should pick K=4 when acceptance is 100%");

  // Losing speculation controller: 0% acceptance and verify is 2x slower than AR!
  MtpDepthController losing(4U, false);
  for (U32 step = 0U; step < 64U && !losing.should_exit(); ++step) {
    const U32 d = losing.select_depth(4U);
    losing.observe(d, 0U, d == 0U ? 8.0 : (16.0 + 4.0 * double(d)), true);
  }
  require(losing.should_exit(),
          "adaptive controller failed to trigger should_exit on sustained 0% acceptance");

  // UDS/Serve `speculative.draft_tokens` must act as adaptive ceiling (not
  // fixed_depth lock) on real models (hidden_size >= 512), and clamp > 4 to 4.
  {
    LoadOptions sidecar_opts;
    sidecar_opts.spec_search = "auto";
    sidecar_opts.speculative.backend = SpeculativeBackend::mtp;
    sidecar_opts.speculative.draft_tokens = 3U;
    const auto pol3 =
        detail::resolve_mtp_depth_policy(sidecar_opts, 5120U, 4U, 4U);
    require(pol3.max_draft_tokens == 3U && !pol3.fixed_depth,
            "sidecar draft_tokens=3 unexpectedly locked fixed_depth=true");
    sidecar_opts.path_overrides["mtp_fixed_depth"] = "on";
    const auto fixed3 =
        detail::resolve_mtp_depth_policy(sidecar_opts, 5120U, 4U, 4U);
    require(fixed3.max_draft_tokens == 3U && fixed3.fixed_depth,
            "sidecar explicitly fixed MTP depth was ignored");
    sidecar_opts.path_overrides.erase("mtp_fixed_depth");

    sidecar_opts.speculative.draft_tokens = 7U;
    const auto pol7 =
        detail::resolve_mtp_depth_policy(sidecar_opts, 5120U, 4U, 4U);
    require(pol7.max_draft_tokens == 4U && !pol7.fixed_depth,
            "sidecar draft_tokens=7 failed to clamp to 4 with adaptive depth");
  }

  return {{"warmup_verified", true},
          {"high_accept_k4_verified", true},
          {"sustained_loss_exit_verified", true},
          {"sidecar_adaptive_policy_verified", true}};
}

Json test_end_to_end_engine_parity(const std::filesystem::path &qwen_fixtures,
                                   const std::filesystem::path &libraries,
                                   const std::filesystem::path &work_dir) {
  struct ModelCase {
    std::string label;
    std::filesystem::path model_dir;
    std::filesystem::path bundled_artifact;
    U32 vocab_size;
  };
  const std::vector<ModelCase> cases{
      {"dense_affine", qwen_fixtures / "affine/source",
       work_dir / "qwen_affine_bundled_mtp.omlxm5", 257U},
      {"dense_mxfp4_bare_prefix", qwen_fixtures / "mxfp4/source",
       work_dir / "qwen_mxfp4_bundled_mtp.omlxm5", 257U},
      {"moe_256_affine", qwen_fixtures / "moe-affine/source",
       work_dir / "qwen_moe_bundled_mtp.omlxm5", 257U},
  };

  const auto qwen_metallib = (libraries / "omlx_m5_qwen_ar.metallib").string();
  U32 total_parity_checks = 0;

  for (const auto &mc : cases) {
    // Build pure-AR engine (enable_mtp=0 opt-out) on the EXACT SAME bundled MTP artifact!
    LoadOptions ar_opts;
    ar_opts.model_dir = mc.model_dir.string();
    ar_opts.max_context = 2048;
    ar_opts.sampling_contract = "position_coupled_v1";
    ar_opts.spec_search = "off";
    ar_opts.path_overrides = {
        {"artifact", mc.bundled_artifact.string()},
        {"qwen_ar_metallib", qwen_metallib},
        {"enable_mtp", "0"},
    };
    auto ar_engine = Engine::create(ar_opts);
    require(ar_engine->stats().kernel_family == "qwen_forward_a16_metal4",
            mc.label + ": enable_mtp=0 did not select pure AR kernel_family");

    // Build auto-enabled MTP engine (default LoadOptions without enable_mtp override!)
    LoadOptions mtp_opts;
    mtp_opts.model_dir = mc.model_dir.string();
    mtp_opts.max_context = 2048;
    mtp_opts.sampling_contract = "position_coupled_v1";
    mtp_opts.spec_search = "off";
    mtp_opts.path_overrides = {
        {"artifact", mc.bundled_artifact.string()},
        {"qwen_ar_metallib", qwen_metallib},
    };
    auto mtp_engine = Engine::create(mtp_opts);
    require(mtp_engine->stats().kernel_family == "qwen_mtp_forward_a16_metal4",
            mc.label + ": auto-enabled MTP did not select MTP kernel_family");

    // Verify greedy (T=0) AND stochastic position_coupled_v1 (T=0.7) 0-ULP parity!
    const std::vector<std::size_t> prompt_lens =
        (mc.label == "dense_affine")
            ? std::vector<std::size_t>{1U, 19U, 1050U}
            : std::vector<std::size_t>{17U};

    for (std::size_t plen : prompt_lens) {
      SubmitRequest req_greedy;
      req_greedy.prompt_tokens = make_prompt_i32(plen, 11U, mc.vocab_size);
      req_greedy.max_new_tokens = 28;
      req_greedy.sampling.temperature = 0.0F;
      req_greedy.sampling.top_p = 1.0F;
      req_greedy.sampling.top_k = 0;
      req_greedy.sampling.seed = 0ULL;

      const auto ar_out = collect_request(*ar_engine, ar_engine->submit(req_greedy));
      const auto mtp_out = collect_request(*mtp_engine, mtp_engine->submit(req_greedy));
      require(ar_out.tokens.size() == 28U && ar_out.tokens == mtp_out.tokens,
              mc.label + ": greedy AR != MTP token parity failed at plen=" +
                  std::to_string(plen));
      ++total_parity_checks;
    }

    // Stochastic coupled parity (temperature=0.7, top_k=16, top_p=0.9)
    for (U64 seed : {42ULL, 1337ULL}) {
      SubmitRequest req_stoch;
      req_stoch.prompt_tokens = make_prompt_i32(23U, U32(seed), mc.vocab_size);
      req_stoch.max_new_tokens = 26;
      req_stoch.sampling.temperature = 0.7F;
      req_stoch.sampling.top_k = 16;
      req_stoch.sampling.top_p = 0.9F;
      req_stoch.sampling.seed = seed;

      const auto ar_out = collect_request(*ar_engine, ar_engine->submit(req_stoch));
      const auto mtp_out = collect_request(*mtp_engine, mtp_engine->submit(req_stoch));
      require(ar_out.tokens.size() == 26U && ar_out.tokens == mtp_out.tokens,
              mc.label + ": coupled stochastic AR != MTP parity failed for seed=" +
                  std::to_string(seed));
      ++total_parity_checks;
    }

    const auto ar_st = ar_engine->stats();
    const auto mtp_st = mtp_engine->stats();
    require(ar_st.spec_rounds == 0U && ar_st.ar_steps > 0U,
            mc.label + ": pure AR engine unexpectedly ran speculative rounds");
    require(mtp_st.spec_rounds > 0U && mtp_st.spec_proposal_tokens > 0U,
            mc.label + ": MTP engine did not execute speculative verify rounds");
  }

  // Test OMLX_M5_DISABLE_MTP=1 env opt-out!
  {
    ScopedEnv dis_env("OMLX_M5_DISABLE_MTP", "1");
    LoadOptions env_opts;
    env_opts.model_dir = (qwen_fixtures / "affine/source").string();
    env_opts.max_context = 1024;
    env_opts.sampling_contract = "position_coupled_v1";
    env_opts.path_overrides = {
        {"artifact", (work_dir / "qwen_affine_bundled_mtp.omlxm5").string()},
        {"qwen_ar_metallib", qwen_metallib},
    };
    auto env_engine = Engine::create(env_opts);
    require(env_engine->stats().kernel_family == "qwen_forward_a16_metal4",
            "OMLX_M5_DISABLE_MTP=1 failed to opt out of MTP");
  }

  // Test standalone external mtp_artifact override + stop-sequence clipping +
  // adaptive spec_search="on" + LegacyRejectionV1 + B=2 concurrency!
  {
    LoadOptions ext_opts;
    ext_opts.model_dir = (qwen_fixtures / "affine/source").string();
    ext_opts.max_context = 1024;
    ext_opts.sampling_contract = "position_coupled_v1";
    ext_opts.spec_search = "on";
    ext_opts.path_overrides = {
        {"artifact", (qwen_fixtures / "affine/model.omlxm5").string()},
        {"mtp_artifact", (work_dir / "mtp_affine_only.omlxm5").string()},
        {"mtp_draft_tokens", "3"},
        {"qwen_ar_metallib", qwen_metallib},
    };
    auto ext_engine = Engine::create(ext_opts);
    require(ext_engine->stats().kernel_family == "qwen_mtp_forward_a16_metal4",
            "external mtp_artifact failed to activate MTP");

    LoadOptions ref_ar_opts;
    ref_ar_opts.model_dir = (qwen_fixtures / "affine/source").string();
    ref_ar_opts.max_context = 1024;
    ref_ar_opts.sampling_contract = "position_coupled_v1";
    ref_ar_opts.path_overrides = {
        {"artifact", (qwen_fixtures / "affine/model.omlxm5").string()},
        {"qwen_ar_metallib", qwen_metallib},
    };
    auto ref_ar = Engine::create(ref_ar_opts);

    SubmitRequest full_req;
    full_req.prompt_tokens = make_prompt_i32(15U, 55U, 257U);
    full_req.max_new_tokens = 24;
    const auto ref_full = collect_request(*ref_ar, ref_ar->submit(full_req));
    const auto ext_full = collect_request(*ext_engine, ext_engine->submit(full_req));
    require(ref_full.tokens == ext_full.tokens,
            "external mtp_artifact + spec_search=on parity mismatch");
    ++total_parity_checks;

    // Stop sequence clipping inside a multi-token speculative batch!
    require(ref_full.tokens.size() >= 10U, "expected >= 10 reference tokens");
    SubmitRequest stop_req = full_req;
    stop_req.stop_token_seqs = {{ref_full.tokens[5], ref_full.tokens[6]}};
    const auto ar_stop = collect_request(*ref_ar, ref_ar->submit(stop_req));
    const auto mtp_stop = collect_request(*ext_engine, ext_engine->submit(stop_req));
    require(ar_stop.finish_reason == "stop" && mtp_stop.finish_reason == "stop" &&
                ar_stop.tokens == mtp_stop.tokens,
            "stop-sequence clipping AR != MTP mismatch");
    ++total_parity_checks;

    // Multi-slot B=2 concurrent MTP with >16-token delegated prefill on slot 1!
    LoadOptions b2_opts;
    b2_opts.model_dir = (qwen_fixtures / "affine/source").string();
    b2_opts.max_context = 1024;
    b2_opts.max_concurrency = 2;
    b2_opts.sampling_contract = "position_coupled_v1";
    b2_opts.spec_search = "off";
    b2_opts.path_overrides = {
        {"artifact", (work_dir / "qwen_affine_bundled_mtp.omlxm5").string()},
        {"qwen_ar_metallib", qwen_metallib},
        {"batch_mode", "on"},
    };
    auto b2_engine = Engine::create(b2_opts);
    SubmitRequest r0, r1;
    r0.prompt_tokens = make_prompt_i32(25U, 101U, 257U);
    r0.max_new_tokens = 20;
    r1.prompt_tokens = make_prompt_i32(42U, 202U, 257U); // >16 -> delegated prefill on slot 1!
    r1.max_new_tokens = 20;

    const auto ar_r0 = collect_request(*ref_ar, ref_ar->submit(r0));
    const auto ar_r1 = collect_request(*ref_ar, ref_ar->submit(r1));

    const U64 id0 = b2_engine->submit(r0);
    const U64 id1 = b2_engine->submit(r1);
    std::vector<std::int32_t> got0, got1;
    bool done0 = false, done1 = false;
    while (!done0 || !done1) {
      for (auto &ev : b2_engine->poll_events(std::chrono::milliseconds(50))) {
        if (ev.request_id == id0) {
          if (ev.kind == EventKind::tokens) {
            got0.insert(got0.end(), ev.tokens.begin(), ev.tokens.end());
          } else {
            require(ev.kind == EventKind::finished, "B=2 slot 0 failed");
            done0 = true;
          }
        } else if (ev.request_id == id1) {
          if (ev.kind == EventKind::tokens) {
            got1.insert(got1.end(), ev.tokens.begin(), ev.tokens.end());
          } else {
            require(ev.kind == EventKind::finished, "B=2 slot 1 failed");
            done1 = true;
          }
        }
      }
    }
    require(got0 == ar_r0.tokens && got1 == ar_r1.tokens,
            "B=2 concurrent MTP (with delegated prefill) diverged from AR");
    ++total_parity_checks;
  }

  // LegacyRejectionV1 greedy & stochastic check!
  {
    LoadOptions leg_ar_opts;
    leg_ar_opts.model_dir = (qwen_fixtures / "affine/source").string();
    leg_ar_opts.max_context = 1024;
    leg_ar_opts.sampling_contract = "legacy_rejection_v1";
    leg_ar_opts.spec_search = "off";
    leg_ar_opts.path_overrides = {
        {"artifact", (work_dir / "qwen_affine_bundled_mtp.omlxm5").string()},
        {"qwen_ar_metallib", qwen_metallib},
        {"enable_mtp", "0"},
    };
    auto leg_ar = Engine::create(leg_ar_opts);

    LoadOptions leg_mtp_opts = leg_ar_opts;
    leg_mtp_opts.path_overrides.erase("enable_mtp");
    auto leg_mtp = Engine::create(leg_mtp_opts);

    SubmitRequest g_req;
    g_req.prompt_tokens = make_prompt_i32(14U, 77U, 257U);
    g_req.max_new_tokens = 20;
    require(collect_request(*leg_ar, leg_ar->submit(g_req)).tokens ==
                collect_request(*leg_mtp, leg_mtp->submit(g_req)).tokens,
            "LegacyRejectionV1 greedy AR != MTP mismatch");
    ++total_parity_checks;

    SubmitRequest s_req = g_req;
    s_req.sampling.temperature = 0.8F;
    s_req.sampling.top_k = 16;
    s_req.sampling.top_p = 0.9F;
    s_req.sampling.seed = 999ULL;
    const auto s_out = collect_request(*leg_mtp, leg_mtp->submit(s_req));
    require(s_out.tokens.size() == 20U,
            "LegacyRejectionV1 stochastic MTP output length != 20");
  }

  // Rolling Prefix Cache multi-turn MTP KV frontier retention: Turn 2 must hit
  // Rolling Prefix Cache AND keep executing speculative MTP verify rounds with
  // 0-ULP greedy parity against pure AR!
  {
    LoadOptions roll_ar_opts;
    roll_ar_opts.model_dir = (qwen_fixtures / "affine/source").string();
    roll_ar_opts.max_context = 2048;
    roll_ar_opts.sampling_contract = "position_coupled_v1";
    roll_ar_opts.spec_search = "off";
    roll_ar_opts.path_overrides = {
        {"artifact", (work_dir / "qwen_affine_bundled_mtp.omlxm5").string()},
        {"qwen_ar_metallib", qwen_metallib},
        {"enable_mtp", "0"},
        {"prefix_cache_bytes", "67108864"},
        {"rolling_prefix_cache", "on"},
    };
    auto roll_ar = Engine::create(roll_ar_opts);

    LoadOptions roll_mtp_opts = roll_ar_opts;
    roll_mtp_opts.path_overrides.erase("enable_mtp");
    auto roll_mtp = Engine::create(roll_mtp_opts);

    SubmitRequest t1;
    t1.prompt_tokens = make_prompt_i32(25U, 33U, 257U);
    t1.max_new_tokens = 18;
    const auto ar_t1 = collect_request(*roll_ar, roll_ar->submit(t1));
    const auto mtp_t1 = collect_request(*roll_mtp, roll_mtp->submit(t1));
    require(ar_t1.tokens == mtp_t1.tokens,
            "Rolling MTP Turn 1 greedy parity mismatch");
    const auto rounds_after_t1 = roll_mtp->stats().spec_rounds;
    require(rounds_after_t1 > 0U, "Rolling MTP Turn 1 ran 0 spec rounds");

    SubmitRequest t2;
    t2.prompt_tokens = t1.prompt_tokens;
    t2.prompt_tokens.insert(t2.prompt_tokens.end(), mtp_t1.tokens.begin(),
                            mtp_t1.tokens.end());
    const auto suffix = make_prompt_i32(9U, 88U, 257U);
    t2.prompt_tokens.insert(t2.prompt_tokens.end(), suffix.begin(),
                            suffix.end());
    t2.max_new_tokens = 18;
    const auto ar_t2 = collect_request(*roll_ar, roll_ar->submit(t2));
    const auto mtp_t2 = collect_request(*roll_mtp, roll_mtp->submit(t2));
    const auto st_after_t2 = roll_mtp->stats();
    require(st_after_t2.prefix_hits >= 1U &&
                st_after_t2.prefix_restored_tokens >= 42U,
            "Rolling MTP Turn 2 did not hit Rolling Prefix Cache");
    require(st_after_t2.spec_rounds > rounds_after_t1,
            "Rolling MTP Turn 2 dropped out of MTP instead of retaining KV frontier");
    require(ar_t2.tokens == mtp_t2.tokens,
            "Rolling MTP Turn 2 greedy parity mismatch after restore_frontier");
    require(mtp_t1.request_stats && mtp_t1.request_stats->prefix_restored_tokens == 0U,
            "Rolling MTP cold request reported a prefix restore");
    require(mtp_t2.request_stats &&
                mtp_t2.request_stats->prefix_restored_tokens >= 42U &&
                mtp_t2.request_stats->spec_rounds > 0U &&
                mtp_t2.request_stats->fallback_reason.empty(),
            "Rolling MTP warm request telemetry did not report continued speculation");
    ++total_parity_checks;
  }

  return {{"parity_checks_passed", total_parity_checks},
          {"zero_ulp_ar_eq_mtp", true}};
}

Json test_mtp_pipelines_and_memory(const MetalContext &ctx,
                                   const std::filesystem::path &libraries) {
  ctx.register_library(
      {.key = "qwen.forward.library.mtp_forward",
       .path = libraries / "omlx_m5_mtp_forward.metallib"});
  Json tg_bytes = Json::object();
  const std::array<const char *, 5> fns{
      "omlx_m5_mtp_save_trunk_row", "omlx_m5_mtp_stem_norms",
      "omlx_m5_mtp_fc_dual_bf16", "omlx_m5_mtp_qk_norm_rope",
      "omlx_m5_mtp_latch_chain_step"};
  U64 max_tg = 0;
  for (const char *fn : fns) {
    const auto info = ctx.warmup_pipeline(
        {.key = std::string("qwen.forward.") + fn,
         .library_key = "qwen.forward.library.mtp_forward",
         .function_name = fn,
         .variant = "fixed-qwen",
         .support_indirect_command_buffers =
             ctx.capabilities().profile.compute_icb_eligible});
    require(info.static_threadgroup_memory_bytes <= 32768ULL,
            std::string("MTP kernel exceeds 32 KiB TG SRAM: ") + fn);
    if (ctx.capabilities().profile.compute_icb_eligible) {
      require(info.indirect_command_buffers_enabled,
              std::string("MTP kernel not ICB-enabled: ") + fn);
    }
    tg_bytes[fn] = info.static_threadgroup_memory_bytes;
    max_tg = std::max(max_tg, info.static_threadgroup_memory_bytes);
  }

  // Verify 27B dense & 35B MoE real-geometry memory plans obey <= 2 GiB ICB rule.
  QwenModelConfig c27;
  c27.hidden_size = 5120;
  c27.intermediate_size = 17408;
  c27.vocab_size = 248320;
  c27.num_attention_heads = 32;
  c27.num_key_value_heads = 4;
  c27.head_dim = 256;
  const auto p27 = plan_mtp_forward_memory(c27, 32768U, false);
  require(p27.largest_buffer_bytes <= (2ULL << 30),
          "27B MTP buffer exceeds 2 GiB ICB limit");
  const auto p27_aff4 = plan_mtp_forward_memory(c27, 65536U, true);
  require(p27_aff4.largest_buffer_bytes <= (2ULL << 30),
          "27B affine4_g128 64K MTP buffer exceeds 2 GiB ICB limit");
  for (const auto &b : p27_aff4.buffers) {
    require(b.name.find(".scratch") == std::string::npos,
            "MTP affine4 buffer contains forbidden .scratch substring: " +
                b.name);
  }

  QwenModelConfig c35 = c27;
  c35.hidden_size = 2048;
  c35.intermediate_size = 512;
  c35.shared_expert_intermediate_size = 512;
  c35.num_experts = 256;
  c35.num_experts_per_token = 8;
  c35.num_attention_heads = 16;
  c35.num_key_value_heads = 2;
  const auto p35 = plan_mtp_forward_memory(c35, 32768U, false);
  require(p35.largest_buffer_bytes <= (2ULL << 30),
          "35B MoE MTP buffer exceeds 2 GiB ICB limit");

  return {{"max_static_tg_bytes", max_tg},
          {"tg_bytes_by_kernel", tg_bytes},
          {"ws_27b_bytes", p27.workspace_bytes},
          {"ws_35b_moe_bytes", p35.workspace_bytes}};
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 5) {
      throw std::invalid_argument(
          "usage: omlx_m5_mtp_forward_tests QWEN_FIXTURES METAL_DIR PYTHON REPO_ROOT");
    }
    ScopedEnv clear_dis("OMLX_M5_DISABLE_MTP", nullptr);
    ScopedEnv clear_en("OMLX_M5_ENABLE_MTP", nullptr);
    ScopedEnv clear_dt("OMLX_M5_MTP_DRAFT_TOKENS", nullptr);

    const std::filesystem::path qwen_fixtures(argv[1]);
    const std::filesystem::path libraries(argv[2]);
    const std::filesystem::path python_exe(argv[3]);
    const std::filesystem::path repo_root(argv[4]);

    const auto work_dir =
        std::filesystem::temp_directory_path() /
        ("omlx-m5-mtp-test-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(work_dir);

    struct Cleanup {
      std::filesystem::path dir;
      ~Cleanup() {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
      }
    } cleanup{work_dir};

    auto ctx = MetalContext::create();
    generate_mtp_fixtures_via_python(python_exe, repo_root, qwen_fixtures,
                                     work_dir);

    Json report;
    report["schema"] = "omlx-m5-step5-mtp-tests-v1";
    report["pipelines_and_memory"] =
        test_mtp_pipelines_and_memory(ctx, libraries);
    report["artifact_cli"] =
        test_artifact_cli_roundtrip(python_exe, repo_root, qwen_fixtures, work_dir);
    if (std::getenv("OMLX_M5_MTP_SHADER_VAL_SMOKE") != nullptr) {
      LoadOptions leg_mtp_opts;
      leg_mtp_opts.model_dir = (qwen_fixtures / "affine/source").string();
      leg_mtp_opts.max_context = 1024;
      leg_mtp_opts.sampling_contract = "legacy_rejection_v1";
      leg_mtp_opts.spec_search = "off";
      leg_mtp_opts.path_overrides = {
          {"artifact", (work_dir / "qwen_affine_bundled_mtp.omlxm5").string()},
          {"qwen_ar_metallib", (libraries / "omlx_m5_qwen_ar.metallib").string()},
      };
      auto leg_mtp = Engine::create(leg_mtp_opts);
      SubmitRequest req;
      req.prompt_tokens = make_prompt_i32(70U, 11U, 257U); // >64 -> multi-subchunk priming!
      req.max_new_tokens = 12;
      const auto out = collect_request(*leg_mtp, leg_mtp->submit(req));
      require(out.tokens.size() == 12U, "shader-val smoke length != 12");
      report["shader_val_smoke"] = true;
      report["passed"] = true;
      std::cout << report.dump() << '\n';
      return 0;
    }
    report["oracle_and_rollback"] =
        test_mlx_oracle_and_rollback(ctx, qwen_fixtures, libraries, work_dir);
    report["depth_controller"] = test_depth_controller();
    report["engine_parity"] =
        test_end_to_end_engine_parity(qwen_fixtures, libraries, work_dir);
    report["passed"] = true;

    std::cout << report.dump() << '\n';
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
