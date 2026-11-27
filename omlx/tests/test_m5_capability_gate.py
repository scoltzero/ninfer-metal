# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for the `< 5 ms` O(1) Two-Layer Structural & Per-Tensor
Capability Gate (`inspect_m5rt_source_fast`), `model_type`-decoupled auto-routing,
`head_dim`-driven `kv_dtype` selection, DFlash2 structural pairing, and UI blocker
reasons in `omlx.engine.m5_sidecar`.
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from omlx.admin import routes as admin_routes
from omlx.engine import m5_sidecar as m5_mod
from omlx.engine.m5_sidecar import (
    M5RT_ENGINE_ID,
    M5RT_SUPPORTED_DFLASH2_MODEL_TYPES,
    M5RT_SUPPORTED_DFLASH2_REPOS,
    M5RT_SUPPORTED_DFLASH2_TARGET_GEOMETRIES,
    M5RT_SUPPORTED_MODEL_TYPES,
    M5Options,
    M5Target,
    build_m5rt_options,
    can_auto_convert_m5rt_fast,
    can_convert_to_miv,
    detect_adapted,
    estimate_m5rt_memory_bytes,
    find_m5rt_binary,
    inspect_m5rt_source_fast,
    resolve_m5rt_target,
)
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.model_settings import ModelSettings
from omlx.settings import GlobalSettings

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
}


def _write_header_only_safetensors(
    shard_path: Path,
    tensor_specs: dict[str, tuple[str, list[int] | tuple[int, ...]]],
    *,
    pad_bytes: int = 64,
) -> Path:
    """Write a valid `.safetensors` header with large virtual `data_offsets` and zero payload read."""
    header: dict[str, Any] = {"__metadata__": {"format": "mlx"}}
    cursor = 0
    for name, (dtype, shape) in tensor_specs.items():
        elems = 1
        for d in shape:
            elems *= int(d)
        nbytes = elems * _DTYPE_BYTES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": [int(d) for d in shape],
            "data_offsets": [cursor, cursor + nbytes],
        }
        cursor += nbytes
    raw_hdr = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    # Write ONLY the 8-byte header size + JSON header + tiny pad (not multi-GB payload);
    # any code attempting to mmap/read tensor payload bytes at high offsets would fail!
    shard_path.write_bytes(struct.pack("<Q", len(raw_hdr)) + raw_hdr + (b"\x00" * pad_bytes))
    return shard_path


def _add_affine_linear(
    specs: dict[str, tuple[str, list[int]]],
    module: str,
    out_features: int,
    in_features: int,
    *,
    bits: int = 4,
    group_size: int = 64,
    scales_dtype: str = "BF16",
    leading: tuple[int, ...] = (),
) -> None:
    specs[f"{module}.weight"] = ("U32", [*leading, out_features, (in_features * bits) // 32])
    specs[f"{module}.scales"] = (scales_dtype, [*leading, out_features, in_features // group_size])
    specs[f"{module}.biases"] = (scales_dtype, [*leading, out_features, in_features // group_size])


def _build_pure_gqa_tensors(
    *,
    hidden_size: int = 2048,
    intermediate_size: int = 6144,
    vocab_size: int = 32768,
    num_layers: int = 2,
    num_q_heads: int = 16,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    gated_q: bool = False,
    with_qk_norm: bool = True,
    bits: int = 4,
    group_size: int = 64,
    lm_head_mode: str = "affine",
    prefix: str = "model",
) -> dict[str, tuple[str, list[int]]]:
    specs: dict[str, tuple[str, list[int]]] = {}
    q_rows = (2 if gated_q else 1) * num_q_heads * head_dim
    kv_rows = num_kv_heads * head_dim
    o_cols = num_q_heads * head_dim

    _add_affine_linear(specs, f"{prefix}.embed_tokens", vocab_size, hidden_size, bits=bits, group_size=group_size)
    specs[f"{prefix}.norm.weight"] = ("BF16", [hidden_size])

    for i in range(num_layers):
        lp = f"{prefix}.layers.{i}"
        specs[f"{lp}.input_layernorm.weight"] = ("BF16", [hidden_size])
        specs[f"{lp}.post_attention_layernorm.weight"] = ("BF16", [hidden_size])
        _add_affine_linear(specs, f"{lp}.self_attn.q_proj", q_rows, hidden_size, bits=bits, group_size=group_size)
        _add_affine_linear(specs, f"{lp}.self_attn.k_proj", kv_rows, hidden_size, bits=bits, group_size=group_size)
        _add_affine_linear(specs, f"{lp}.self_attn.v_proj", kv_rows, hidden_size, bits=bits, group_size=group_size)
        _add_affine_linear(specs, f"{lp}.self_attn.o_proj", hidden_size, o_cols, bits=bits, group_size=group_size)
        if with_qk_norm:
            specs[f"{lp}.self_attn.q_norm.weight"] = ("BF16", [head_dim])
            specs[f"{lp}.self_attn.k_norm.weight"] = ("BF16", [head_dim])
        _add_affine_linear(specs, f"{lp}.mlp.gate_proj", intermediate_size, hidden_size, bits=bits, group_size=group_size)
        _add_affine_linear(specs, f"{lp}.mlp.up_proj", intermediate_size, hidden_size, bits=bits, group_size=group_size)
        _add_affine_linear(specs, f"{lp}.mlp.down_proj", hidden_size, intermediate_size, bits=bits, group_size=group_size)

    head_prefix = "language_model.lm_head" if prefix.startswith("language_model.") else "lm_head"
    if lm_head_mode == "bf16":
        specs[f"{head_prefix}.weight"] = ("BF16", [vocab_size, hidden_size])
    elif lm_head_mode == "fp16":
        specs[f"{head_prefix}.weight"] = ("F16", [vocab_size, hidden_size])
    elif lm_head_mode == "affine8":
        _add_affine_linear(specs, head_prefix, vocab_size, hidden_size, bits=8, group_size=64)
    elif lm_head_mode == "tied":
        pass
    else:
        _add_affine_linear(specs, head_prefix, vocab_size, hidden_size, bits=bits, group_size=group_size)
    return specs


def _setup_mock_miv_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    prov_root = tmp_path / "Providers" / "io.ninfer.metal" / "versions" / "miv-v0.03"
    (prov_root / "bin").mkdir(parents=True, exist_ok=True)
    (prov_root / "metallib").mkdir(parents=True, exist_ok=True)
    (prov_root / "tools").mkdir(parents=True, exist_ok=True)
    fake_bin = prov_root / "bin" / "mseries_backendd"
    fake_bin.write_text(
        '#!/bin/sh\nif [ "$1" = "--version" ]; then echo "mseries-backendd miv-v0.03 protocol=2"; exit 0; fi\nexit 0\n',
        encoding="utf-8",
    )
    fake_bin.chmod(0o755)
    (prov_root / "metallib" / "omlx_m5_qwen_ar.metallib").write_bytes(b"MTLB")
    (prov_root / "tools" / "m5_artifact.py").write_text(
        "import pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.write_bytes(b'MIV00001' + b'\\x00' * 128)\n",
        encoding="utf-8",
    )
    # Intentionally keep legacy supported_model_types in provider.json to verify
    # built-in m5rt ignores string whitelists in favor of structural probing!
    (prov_root / "provider.json").write_text(
        json.dumps(
            {
                "provider_id": "io.ninfer.metal",
                "engine_id": "m5rt",
                "protocol_version": 2,
                "entrypoint": "bin/mseries_backendd",
                "metallib_dir": "metallib",
                "supported_model_types": ["qwen3_5", "qwen3_6", "qwen3_8"],
                "artifact_extensions": [".miv", ".omlxm5"],
                "converter": {
                    "entrypoint": "tools/m5_artifact.py",
                    "subcommand": "convert",
                    "output_filename": "model.miv",
                    "artifact_extension": ".miv",
                    "default_args": ["--nax-pretile"],
                    "supported_quant_modes": ["affine", "mxfp4"],
                },
            }
        ),
        encoding="utf-8",
    )
    current = prov_root.parent.parent / "current"
    if current.exists() or current.is_symlink():
        current.unlink()
    current.symlink_to(prov_root)
    monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(tmp_path / "Providers"))
    monkeypatch.setattr(m5_mod, "_DEFAULT_ARTIFACT_CACHE_RELPATH", tmp_path / "isolated_cache")
    monkeypatch.delenv("OMLX_M5RT_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("OMLX_M5RT_BIN", raising=False)
    monkeypatch.delenv("NINFER_METAL_BIN", raising=False)
    return fake_bin


class TestM5TwoLayerCapabilityGate:
    def test_backward_compatible_module_constants_remain_importable(self):
        assert "qwen3_5" in M5RT_SUPPORTED_MODEL_TYPES
        assert len(M5RT_SUPPORTED_DFLASH2_REPOS) > 0
        assert (5120, 64) in M5RT_SUPPORTED_DFLASH2_TARGET_GEOMETRIES
        assert M5RT_SUPPORTED_DFLASH2_MODEL_TYPES == ("qwen3",)
        for mode in ("affine", "mxfp4", "mxfp4-pack-quantized", "bf16", "fp16", "mixed"):
            assert mode in m5_mod.M5RT_DEFAULT_CONVERTER["supported_quant_modes"]

    def test_1_custom_model_type_auto_convert_routing_and_adapted_detection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        fake_bin = _setup_mock_miv_provider(tmp_path, monkeypatch)
        model_dir = tmp_path / "models" / "MyCommunityFinetune-8B"
        cfg = {
            "model_type": "my_community_finetune",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 2,
            "vocab_size": 32768,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rope_theta": 1000000.0,
            "hidden_act": "silu",
            "mtp_num_hidden_layers": 1,
            "quantization": {"group_size": 64, "bits": 4},
        }
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        tensors = _build_pure_gqa_tensors(
            hidden_size=2048,
            intermediate_size=6144,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=256,
            gated_q=True,
        )
        tensors["mtp.fc.weight"] = ("BF16", [2048, 4096])
        _write_header_only_safetensors(model_dir / "model.safetensors", tensors)

        probe = inspect_m5rt_source_fast(model_dir)
        assert probe["eligible"] is True, probe["reason"]
        assert probe["model_type"] == "my_community_finetune"
        assert probe["model_type_hint_matched"] is False
        assert probe["has_attn_output_gate"] is True
        assert probe["has_qk_norm"] is True
        assert probe["companion_mtp_tensors"] == ["mtp.fc.weight"]
        assert probe["elapsed_ms"] < 5.0

        # Zero-subprocess auto-convert & EnginePool routing
        with patch.object(m5_mod.subprocess, "run", side_effect=AssertionError("must not spawn")):
            assert can_auto_convert_m5rt_fast(model_dir, str(fake_bin)) is True
            assert detect_adapted(model_dir).adapted is False

            gs = GlobalSettings(base_path=tmp_path / "base")
            monkeypatch.setattr("omlx.engine_pool.get_global_settings", lambda: gs)
            pool = EnginePool()
            pool._entries["MyCommunityFinetune-8B"] = EngineEntry(
                model_id="MyCommunityFinetune-8B",
                model_path=str(model_dir),
                model_type="llm",
                engine_type="batched",
                estimated_size=4096,
            )
            assert pool.external_engine_id("MyCommunityFinetune-8B", ModelSettings()) == M5RT_ENGINE_ID

        ok, reason = can_convert_to_miv(model_dir, str(fake_bin))
        assert ok is True and reason == ""

        # Admin MTP & DFlash UI unlock works for custom model_type too
        mtp_ok, mtp_reason = m5_mod.external_engine_mtp_compatibility(
            M5RT_ENGINE_ID, {"model_path": str(model_dir), "config_model_type": "my_community_finetune"}
        )
        assert mtp_ok is True, mtp_reason
        df_ok, df_reason = m5_mod.external_engine_dflash_compatibility(
            M5RT_ENGINE_ID,
            {"model_path": str(model_dir), "config_model_type": "my_community_finetune", "engine_type": "batched"},
        )
        assert df_ok is True, df_reason

        # Once model.miv exists, detect_adapted & resolve_m5rt_target recognize it as adapted!
        (model_dir / "model.miv").write_bytes(b"MIV00001" + b"\x00" * 256)
        adapted = detect_adapted(model_dir)
        assert adapted.adapted is True
        assert adapted.artifact_path == (model_dir / "model.miv").resolve()
        assert can_auto_convert_m5rt_fast(model_dir, str(fake_bin)) is False

        target, t_err = resolve_m5rt_target(model_dir, str(fake_bin), ModelSettings())
        assert t_err == "" and target is not None
        assert target.artifact_path == (model_dir / "model.miv").resolve()

    def test_2_zero_gdn_ungated_pure_gqa_and_head_dim_kv_dtype_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        fake_bin = _setup_mock_miv_provider(tmp_path, monkeypatch)
        binary, err = find_m5rt_binary(str(fake_bin))
        assert binary is not None and err == ""

        # Case A: 0-GDN + ungated q_proj + implicit head_dim = 2048 // 16 = 128 + no QK-norm (Llama-style)
        gqa128_dir = tmp_path / "PureGQA-hd128"
        cfg128 = {
            "model_type": "llama",
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 2,
            "vocab_size": 128256,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            # Note: head_dim and layer_types and linear_* intentionally omitted!
            "rope_theta": 500000.0,
            "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
        }
        gqa128_dir.mkdir()
        (gqa128_dir / "config.json").write_text(json.dumps(cfg128), encoding="utf-8")
        _write_header_only_safetensors(
            gqa128_dir / "model.safetensors",
            _build_pure_gqa_tensors(
                hidden_size=2048,
                intermediate_size=8192,
                vocab_size=128256,
                num_q_heads=16,
                num_kv_heads=4,
                head_dim=128,
                gated_q=False,
                with_qk_norm=False,
            ),
        )

        p128 = inspect_m5rt_source_fast(gqa128_dir)
        assert p128["eligible"] is True, p128["reason"]
        assert p128["topology"] == "pure_gqa_swiglu"
        assert p128["linear_attention_layers"] == 0
        assert p128["full_attention_layers"] == 2
        assert p128["head_dim"] == 128
        assert p128["has_attn_output_gate"] is False
        assert p128["has_qk_norm"] is False
        assert p128["kv_dtype_default"] == "bf16"
        assert can_auto_convert_m5rt_fast(gqa128_dir, str(fake_bin)) is True

        # Case B: 0-GDN + ungated q_proj + implicit head_dim = 4096 // 16 = 256 + QK-norm (Qwen3-style)
        gqa256_dir = tmp_path / "PureGQA-hd256"
        cfg256 = {
            "model_type": "qwen3_custom_gqa",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 2,
            "vocab_size": 151936,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "rope_theta": 1000000.0,
            "quantization": {"mode": "affine", "bits": 4, "group_size": 128},
        }
        gqa256_dir.mkdir()
        (gqa256_dir / "config.json").write_text(json.dumps(cfg256), encoding="utf-8")
        _write_header_only_safetensors(
            gqa256_dir / "model.safetensors",
            _build_pure_gqa_tensors(
                hidden_size=4096,
                intermediate_size=12288,
                vocab_size=151936,
                num_q_heads=16,
                num_kv_heads=4,
                head_dim=256,
                gated_q=False,
                with_qk_norm=True,
                group_size=128,
            ),
        )

        p256 = inspect_m5rt_source_fast(gqa256_dir)
        assert p256["eligible"] is True, p256["reason"]
        assert p256["head_dim"] == 256
        assert p256["has_attn_output_gate"] is False
        assert p256["has_qk_norm"] is True
        assert p256["kv_dtype_default"] == "affine4_g128"

        # Verify runtime kv_dtype selection in BOTH default TOML and no-TOML modes!
        t128 = M5Target(binary=binary, artifact_path=None, metallib_dir=fake_bin.parent.parent / "metallib", auto_convert_output=gqa128_dir / "model.miv")
        t256 = M5Target(binary=binary, artifact_path=None, metallib_dir=fake_bin.parent.parent / "metallib", auto_convert_output=gqa256_dir / "model.miv")

        for no_default_cfg in ("0", "1"):
            monkeypatch.setenv("NINFER_METAL_NO_DEFAULT_CONFIG", no_default_cfg)
            opts128 = build_m5rt_options(
                gqa128_dir, t128, ModelSettings(), default_context=8192, memory_budget_bytes=0
            )
            opts256 = build_m5rt_options(
                gqa256_dir, t256, ModelSettings(), default_context=8192, memory_budget_bytes=0
            )
            assert opts128.kv_dtype == "bf16", f"no_default_cfg={no_default_cfg}"
            assert opts256.kv_dtype == "affine4_g128", f"no_default_cfg={no_default_cfg}"

        # Also verify _estimate_shared_kv_pool_bytes with kv_dtype="auto" uses head_dim == 256, NOT model_type!
        est128_auto = estimate_m5rt_memory_bytes(
            gqa128_dir, t128, 8192, plan=M5Options(max_context=8192, max_concurrency=2, kv_dtype="auto")
        )
        est128_bf16 = estimate_m5rt_memory_bytes(
            gqa128_dir, t128, 8192, plan=M5Options(max_context=8192, max_concurrency=2, kv_dtype="bf16")
        )
        assert est128_auto == est128_bf16

        est256_auto = estimate_m5rt_memory_bytes(
            gqa256_dir, t256, 8192, plan=M5Options(max_context=8192, max_concurrency=2, kv_dtype="auto")
        )
        est256_aff4 = estimate_m5rt_memory_bytes(
            gqa256_dir, t256, 8192, plan=M5Options(max_context=8192, max_concurrency=2, kv_dtype="affine4_g128")
        )
        assert est256_auto == est256_aff4

    @pytest.mark.parametrize(
        "num_experts,top_k,with_shared",
        [
            (64, 4, False),
            (64, 4, True),
            (128, 8, False),
            (128, 8, True),
        ],
    )
    def test_3_generalized_64_and_128_expert_moe_with_and_without_shared_expert(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        num_experts: int,
        top_k: int,
        with_shared: bool,
    ):
        fake_bin = _setup_mock_miv_provider(tmp_path, monkeypatch)
        moe_dir = tmp_path / f"MoE-e{num_experts}-k{top_k}-sh{int(with_shared)}"
        cfg: dict[str, Any] = {
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "moe_intermediate_size": 768,
            "num_hidden_layers": 1,
            "vocab_size": 151936,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_experts": num_experts,
            "num_experts_per_tok": top_k,
            "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
        }
        if with_shared:
            cfg["shared_expert_intermediate_size"] = 1024

        specs: dict[str, tuple[str, list[int]]] = {}
        _add_affine_linear(specs, "model.embed_tokens", 151936, 2048)
        specs["model.norm.weight"] = ("BF16", [2048])
        _add_affine_linear(specs, "lm_head", 151936, 2048)

        lp = "model.layers.0"
        specs[f"{lp}.input_layernorm.weight"] = ("BF16", [2048])
        specs[f"{lp}.post_attention_layernorm.weight"] = ("BF16", [2048])
        _add_affine_linear(specs, f"{lp}.self_attn.q_proj", 2048, 2048)
        _add_affine_linear(specs, f"{lp}.self_attn.k_proj", 512, 2048)
        _add_affine_linear(specs, f"{lp}.self_attn.v_proj", 512, 2048)
        _add_affine_linear(specs, f"{lp}.self_attn.o_proj", 2048, 2048)
        specs[f"{lp}.mlp.gate.weight"] = ("BF16", [num_experts, 2048])
        _add_affine_linear(specs, f"{lp}.mlp.switch_mlp.gate_proj", 768, 2048, leading=(num_experts,))
        _add_affine_linear(specs, f"{lp}.mlp.switch_mlp.up_proj", 768, 2048, leading=(num_experts,))
        _add_affine_linear(specs, f"{lp}.mlp.switch_mlp.down_proj", 2048, 768, leading=(num_experts,))
        if with_shared:
            specs[f"{lp}.mlp.shared_expert_gate.weight"] = ("BF16", [1, 2048])
            _add_affine_linear(specs, f"{lp}.mlp.shared_expert.gate_proj", 1024, 2048)
            _add_affine_linear(specs, f"{lp}.mlp.shared_expert.up_proj", 1024, 2048)
            _add_affine_linear(specs, f"{lp}.mlp.shared_expert.down_proj", 2048, 1024)

        moe_dir.mkdir()
        (moe_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        _write_header_only_safetensors(moe_dir / "model.safetensors", specs)

        probe = inspect_m5rt_source_fast(moe_dir)
        assert probe["eligible"] is True, probe["reason"]
        assert probe["num_experts"] == num_experts
        assert probe["num_experts_per_tok"] == top_k
        assert probe["has_shared_expert"] is with_shared
        assert can_auto_convert_m5rt_fast(moe_dir, str(fake_bin)) is True
        ok, reason = can_convert_to_miv(moe_dir, str(fake_bin))
        assert ok is True and reason == ""

        # Also verify adapted .miv recognition on 64-expert MoE
        (moe_dir / "model.miv").write_bytes(b"MIV00001" + b"\x00" * 128)
        assert detect_adapted(moe_dir).adapted is True

    def test_4_mixed_precision_affine_trunk_plus_bf16_fp16_8bit_and_tied_lm_head(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        fake_bin = _setup_mock_miv_provider(tmp_path, monkeypatch)

        for variant, lm_head_mode, tie_flag in (
            ("bf16_lm_head", "bf16", False),
            ("fp16_lm_head", "fp16", False),
            ("affine8_lm_head", "affine8", False),
            ("tied_embeddings", "tied", True),
        ):
            mdir = tmp_path / f"Mixed-{variant}"
            cfg = {
                "model_type": "custom_mixed_arch",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 2,
                "vocab_size": 65536,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "tie_word_embeddings": tie_flag,
                "quantization": {
                    "mode": "affine",
                    "bits": 4,
                    "group_size": 64,
                    "lm_head": {"bits": 8, "group_size": 64} if lm_head_mode == "affine8" else False,
                },
            }
            mdir.mkdir()
            (mdir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            tensors = _build_pure_gqa_tensors(
                hidden_size=2048,
                intermediate_size=6144,
                vocab_size=65536,
                lm_head_mode=lm_head_mode,
            )
            _write_header_only_safetensors(mdir / "model.safetensors", tensors)

            probe = inspect_m5rt_source_fast(mdir)
            assert probe["eligible"] is True, f"{variant}: {probe['reason']}"
            if lm_head_mode in ("bf16", "fp16", "affine8"):
                assert probe["is_mixed_precision"] is True
            if tie_flag:
                assert probe["tie_word_embeddings"] is True
            assert can_auto_convert_m5rt_fast(mdir, str(fake_bin)) is True
            ok, reason = can_convert_to_miv(mdir, str(fake_bin))
            assert ok is True and reason == ""

        # Pure BF16 checkpoint (no quantization dict in config.json)
        all_bf16_dir = tmp_path / "Pure-BF16-Valid"
        all_bf16_dir.mkdir()
        (all_bf16_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "custom_bf16_arch",
                    "hidden_size": 1024,
                    "intermediate_size": 3072,
                    "num_hidden_layers": 1,
                    "vocab_size": 32000,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                    "head_dim": 128,
                }
            ),
            encoding="utf-8",
        )
        bf16_specs: dict[str, tuple[str, list[int]]] = {
            "model.embed_tokens.weight": ("BF16", [32000, 1024]),
            "model.norm.weight": ("BF16", [1024]),
            "lm_head.weight": ("BF16", [32000, 1024]),
            "model.layers.0.input_layernorm.weight": ("BF16", [1024]),
            "model.layers.0.post_attention_layernorm.weight": ("BF16", [1024]),
            "model.layers.0.self_attn.q_proj.weight": ("BF16", [1024, 1024]),
            "model.layers.0.self_attn.k_proj.weight": ("BF16", [256, 1024]),
            "model.layers.0.self_attn.v_proj.weight": ("BF16", [256, 1024]),
            "model.layers.0.self_attn.o_proj.weight": ("BF16", [1024, 1024]),
            "model.layers.0.mlp.gate_proj.weight": ("BF16", [3072, 1024]),
            "model.layers.0.mlp.up_proj.weight": ("BF16", [3072, 1024]),
            "model.layers.0.mlp.down_proj.weight": ("BF16", [1024, 3072]),
        }
        _write_header_only_safetensors(all_bf16_dir / "model.safetensors", bf16_specs)
        bf16_probe = inspect_m5rt_source_fast(all_bf16_dir)
        assert bf16_probe["eligible"] is True, bf16_probe["reason"]
        assert bf16_probe["quant_modes"] == ["bf16"]
        assert can_convert_to_miv(all_bf16_dir, str(fake_bin)) == (True, "")

    def test_5_vlm_checkpoint_with_visual_tensors_safely_accepted_as_skip_non_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        fake_bin = _setup_mock_miv_provider(tmp_path, monkeypatch)
        vlm_dir = tmp_path / "CustomVLM-7B-4bit"
        cfg = {
            "model_type": "custom_vlm",
            "vision_config": {"hidden_size": 1152, "depth": 27},
            "text_config": {
                "model_type": "custom_vlm_text",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 2,
                "vocab_size": 151936,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 128,
            },
            "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
        }
        specs = _build_pure_gqa_tensors(
            hidden_size=2048,
            intermediate_size=6144,
            vocab_size=151936,
            prefix="language_model.model",
        )
        # Non-text VLM tensors with unaligned dimensions, Conv3D shapes, and .bias tensors
        # must all be safely categorized as skip_non_text without blocking conversion!
        non_text = {
            "visual.patch_embed.proj.weight": ("BF16", [1152, 3, 2, 14, 14]),
            "visual.patch_embed.proj.bias": ("BF16", [1152]),
            "model.visual.blocks.0.attn.qkv.weight": ("BF16", [3456, 1152]),
            "model.visual.blocks.0.attn.qkv.bias": ("BF16", [3456]),
            "vision_tower.encoder.layers.0.fc1.weight": ("F16", [4304, 1152]),
            "multi_modal_projector.linear_1.weight": ("BF16", [2048, 1152]),
            "multi_modal_projector.linear_1.bias": ("BF16", [2048]),
            "aligner.proj.weight": ("BF16", [2048, 1152]),
            "audio_tower.conv1.weight": ("BF16", [1280, 128, 3]),
        }
        specs.update(non_text)

        vlm_dir.mkdir()
        (vlm_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        _write_header_only_safetensors(vlm_dir / "model.safetensors", specs)

        probe = inspect_m5rt_source_fast(vlm_dir)
        assert probe["eligible"] is True, probe["reason"]
        assert set(probe["skipped_non_text_tensors"]) == set(non_text.keys())
        assert probe["repack_counts"]["skip_non_text"] == len(non_text)
        assert can_auto_convert_m5rt_fast(vlm_dir, str(fake_bin)) is True
        assert can_convert_to_miv(vlm_dir, str(fake_bin)) == (True, "")

    def test_6_dflash2_structural_validation_and_target_compatibility(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        fake_bin = _setup_mock_miv_provider(tmp_path, monkeypatch)

        # Target model with non-248320 vocab (151936) and custom model_type
        target_dir = tmp_path / "models" / "Community-Qwen3-14B"
        target_cfg = {
            "model_type": "community_qwen3",
            "hidden_size": 5120,
            "intermediate_size": 15360,
            "num_hidden_layers": 40,
            "vocab_size": 151936,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
        }
        target_dir.mkdir(parents=True)
        (target_dir / "config.json").write_text(json.dumps(target_cfg), encoding="utf-8")
        _write_header_only_safetensors(
            target_dir / "model.safetensors",
            _build_pure_gqa_tensors(
                hidden_size=5120,
                intermediate_size=15360,
                vocab_size=151936,
                num_layers=2,
                num_q_heads=40,
                num_kv_heads=8,
                head_dim=128,
            ),
        )
        # Avoid generating 40 layer headers in target_dir by marking target already adapted for pairing test:
        (target_dir / "model.miv").write_bytes(b"MIV00001" + b"\x00" * 256)

        # Custom repo name & custom model_type & non-248320 vocab DFlash2 draft
        draft_dir = tmp_path / "models" / "MyCustomOrg-14B-DFlash2-Draft"
        draft_cfg = {
            "architectures": ["DFlash2DraftModel"],
            "model_type": "custom_draft_arch",
            "is_causal": False,
            "dtype": "bfloat16",
            "hidden_size": 5120,
            "intermediate_size": 15360,
            "vocab_size": 151936,
            "num_hidden_layers": 4,
            "num_target_layers": 40,
            "layer_types": ["sliding_attention"] * 4,
            "dflash_config": {
                "block_size": 8,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "mask_token_id": 151643,
                "selector_rank": 256,
                "selector_top_k": 16,
                "target_layer_ids": [3, 13, 23, 39],
            },
        }
        draft_dir.mkdir(parents=True)
        (draft_dir / "config.json").write_text(json.dumps(draft_cfg), encoding="utf-8")
        _write_header_only_safetensors(
            draft_dir / "model.safetensors",
            {"model.embed_tokens.weight": ("BF16", [151936, 5120])},
        )

        # Standalone & matched target pass
        df_probe = inspect_m5rt_source_fast(draft_dir, target_model_dir=target_dir)
        assert df_probe["eligible"] is True, df_probe["reason"]
        assert df_probe["is_dflash2_draft"] is True
        assert df_probe["elapsed_ms"] < 5.0
        assert can_convert_to_miv(draft_dir, str(fake_bin), target_model_dir=target_dir) == (True, "")

        # resolve_m5rt_target pairs & schedules auto-conversion for custom DFlash2 draft
        ms_df = ModelSettings(dflash_enabled=True, dflash_draft_model=str(draft_dir))
        resolved_t, err_t = resolve_m5rt_target(target_dir, str(fake_bin), ms_df)
        assert err_t == "" and resolved_t is not None
        assert resolved_t.draft_dir == draft_dir.resolve()
        assert resolved_t.auto_convert_draft_output == draft_dir / "model.miv"

        # Mismatched target vocab_size -> rejected with specific reason
        ok_v, reason_v = m5_mod._is_supported_dflash2_draft_config(
            draft_cfg,
            target_config={"vocab_size": 248320, "hidden_size": 5120, "num_hidden_layers": 40},
        )
        assert ok_v is False and "vocab_size" in reason_v

        # Mismatched target hidden_size -> rejected
        ok_h, reason_h = m5_mod._is_supported_dflash2_draft_config(
            draft_cfg,
            target_config={"vocab_size": 151936, "hidden_size": 4096, "num_hidden_layers": 40},
        )
        assert ok_h is False and "hidden_size" in reason_h

        # target_layer_ids out of range (>= target.num_hidden_layers) -> rejected
        ok_l, reason_l = m5_mod._is_supported_dflash2_draft_config(
            draft_cfg,
            target_config={"vocab_size": 151936, "hidden_size": 5120, "num_hidden_layers": 32},
        )
        assert ok_l is False and ("num_target_layers" in reason_l or "target_layer_ids" in reason_l)

        # resolve_m5rt_target rejects mismatched explicit DFlash2 draft when dflash_enabled=True
        bad_draft_dir = tmp_path / "models" / "WrongVocab-DFlash2"
        bad_draft_dir.mkdir()
        (bad_draft_dir / "config.json").write_text(
            json.dumps({**draft_cfg, "vocab_size": 248320}), encoding="utf-8"
        )
        _write_header_only_safetensors(
            bad_draft_dir / "model.safetensors",
            {"model.embed_tokens.weight": ("BF16", [248320, 5120])},
        )
        bad_t, bad_err = resolve_m5rt_target(
            target_dir,
            str(fake_bin),
            ModelSettings(dflash_enabled=True, dflash_draft_model=str(bad_draft_dir)),
        )
        assert bad_t is None and "vocab_size" in bad_err

    def test_7_sub_5ms_negative_blocker_reasons_mla_mamba2_awq_and_affine_gs32(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        fake_bin = _setup_mock_miv_provider(tmp_path, monkeypatch)

        # (a) DeepSeek MLA checkpoint
        mla_dir = tmp_path / "DeepSeek-V3-MLA"
        mla_dir.mkdir()
        (mla_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "deepseek_v3",
                    "hidden_size": 2048,
                    "intermediate_size": 6144,
                    "num_hidden_layers": 2,
                    "vocab_size": 129280,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 16,
                    "kv_lora_rank": 512,
                    "qk_rope_head_dim": 64,
                }
            ),
            encoding="utf-8",
        )
        _write_header_only_safetensors(
            mla_dir / "model.safetensors",
            {
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": ("BF16", [576, 2048]),
            },
        )
        t0 = time.perf_counter()
        mla_probe = inspect_m5rt_source_fast(mla_dir)
        mla_ms = (time.perf_counter() - t0) * 1000.0
        assert mla_probe["eligible"] is False
        assert mla_ms < 5.0 and mla_probe["elapsed_ms"] < 5.0
        assert "MLA" in mla_probe["reason"]
        assert "model.layers.0.self_attn.kv_a_proj_with_mqa.weight" in mla_probe["reason"]
        assert "model_type is not in" not in mla_probe["reason"]
        ok_mla, reason_mla = can_convert_to_miv(mla_dir, str(fake_bin))
        assert ok_mla is False and "MLA" in reason_mla
        # UI compatibility check via resolve_m5rt_target surfaces the exact MLA blocker!
        rt_mla, rt_mla_reason = resolve_m5rt_target(mla_dir, str(fake_bin), ModelSettings())
        assert rt_mla is None and "MLA" in rt_mla_reason

        # (b) Mamba2 / hybrid SSM checkpoint
        mamba_dir = tmp_path / "Nemotron-Mamba2"
        mamba_dir.mkdir()
        (mamba_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "nemotron_h",
                    "hidden_size": 2048,
                    "intermediate_size": 6144,
                    "num_hidden_layers": 2,
                    "vocab_size": 65536,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 4,
                    "head_dim": 128,
                    "layer_types": ["mamba", "full_attention"],
                    "mamba_d_state": 128,
                }
            ),
            encoding="utf-8",
        )
        _write_header_only_safetensors(
            mamba_dir / "model.safetensors",
            {
                "model.layers.0.mamba.in_proj.weight": ("BF16", [4096, 2048]),
            },
        )
        t0 = time.perf_counter()
        mamba_probe = inspect_m5rt_source_fast(mamba_dir)
        mamba_ms = (time.perf_counter() - t0) * 1000.0
        assert mamba_probe["eligible"] is False
        assert mamba_ms < 5.0 and mamba_probe["elapsed_ms"] < 5.0
        assert "Mamba" in mamba_probe["reason"]
        assert "model.layers.0.mamba.in_proj.weight" in mamba_probe["reason"]
        assert "model_type is not in" not in mamba_probe["reason"]

        # (c) AWQ checkpoint (.qweight / .qzeros + quant_method="awq")
        awq_dir = tmp_path / "Qwen3-8B-AWQ"
        awq_dir.mkdir()
        (awq_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3",
                    "hidden_size": 2048,
                    "intermediate_size": 6144,
                    "num_hidden_layers": 1,
                    "vocab_size": 151936,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 4,
                    "head_dim": 128,
                    "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128},
                }
            ),
            encoding="utf-8",
        )
        _write_header_only_safetensors(
            awq_dir / "model.safetensors",
            {
                "model.layers.0.self_attn.q_proj.qweight": ("I32", [2048, 256]),
                "model.layers.0.self_attn.q_proj.qzeros": ("I32", [16, 256]),
            },
        )
        t0 = time.perf_counter()
        awq_probe = inspect_m5rt_source_fast(awq_dir)
        awq_ms = (time.perf_counter() - t0) * 1000.0
        assert awq_probe["eligible"] is False
        assert awq_ms < 5.0 and awq_probe["elapsed_ms"] < 5.0
        assert "AWQ" in awq_probe["reason"]
        assert "model.layers.0.self_attn.q_proj.qweight" in awq_probe["reason"]
        assert "[2048, 256]" in awq_probe["reason"]
        assert awq_probe["first_blocker"] is not None
        assert awq_probe["first_blocker"]["tensor"] == "model.layers.0.self_attn.q_proj.qweight"
        assert awq_probe["first_blocker"]["shape"] == [2048, 256]

        # (d) Unsupported affine group_size=32 inferred per-tensor even when global config claims g64!
        gs32_dir = tmp_path / "Qwen3-8B-Affine-GS32"
        gs32_dir.mkdir()
        (gs32_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3",
                    "hidden_size": 2048,
                    "intermediate_size": 6144,
                    "num_hidden_layers": 1,
                    "vocab_size": 32768,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 4,
                    "head_dim": 128,
                    # Global config claims g64, while layer 0 k_proj actually uses gs=32 (2048 // 64 = 32)!
                    "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
                }
            ),
            encoding="utf-8",
        )
        gs32_specs = _build_pure_gqa_tensors(
            hidden_size=2048,
            intermediate_size=6144,
            vocab_size=32768,
            num_layers=1,
            num_q_heads=16,
            num_kv_heads=4,
            head_dim=128,
        )
        # Override k_proj with affine gs=32 (groups = 2048 // 32 = 64 instead of 32)
        _add_affine_linear(
            gs32_specs,
            "model.layers.0.self_attn.k_proj",
            512,
            2048,
            bits=4,
            group_size=32,
        )
        _write_header_only_safetensors(gs32_dir / "model.safetensors", gs32_specs)

        t0 = time.perf_counter()
        gs32_probe = inspect_m5rt_source_fast(gs32_dir)
        gs32_ms = (time.perf_counter() - t0) * 1000.0
        assert gs32_probe["eligible"] is False
        assert gs32_ms < 5.0 and gs32_probe["elapsed_ms"] < 5.0
        assert "group_size=32" in gs32_probe["reason"]
        assert "model.layers.0.self_attn.k_proj.weight" in gs32_probe["reason"]
        assert "[512, 256]" in gs32_probe["reason"]
        assert gs32_probe["first_blocker"] is not None
        assert gs32_probe["first_blocker"]["tensor"] == "model.layers.0.self_attn.k_proj.weight"
        assert gs32_probe["first_blocker"]["shape"] == [512, 256]

        ok_gs32, reason_gs32 = can_convert_to_miv(gs32_dir, str(fake_bin))
        assert ok_gs32 is False
        assert "group_size=32" in reason_gs32
        assert "model.layers.0.self_attn.k_proj.weight" in reason_gs32
        assert "model_type is not in" not in reason_gs32

    def test_8_hybrid_gdn_dt_bias_sharded_index_and_additional_structural_blockers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _setup_mock_miv_provider(tmp_path, monkeypatch)

        # (a) Sharded hybrid GDN + Full-Attention GQA with .dt_bias passes in < 5 ms
        hybrid_dir = tmp_path / "Hybrid-GDN-Sharded"
        hybrid_dir.mkdir()
        (hybrid_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "custom_hybrid_gdn",
                    "hidden_size": 1024,
                    "intermediate_size": 3072,
                    "num_hidden_layers": 2,
                    "vocab_size": 32768,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                    "head_dim": 128,
                    "layer_types": ["linear_attention", "full_attention"],
                    "linear_num_key_heads": 4,
                    "linear_num_value_heads": 8,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "linear_conv_kernel_dim": 4,
                    "quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
                }
            ),
            encoding="utf-8",
        )
        shard1 = {
            "model.embed_tokens.weight": ("BF16", [32768, 1024]),
            "model.layers.0.input_layernorm.weight": ("BF16", [1024]),
            "model.layers.0.post_attention_layernorm.weight": ("BF16", [1024]),
            "model.layers.0.linear_attn.in_proj_qkv.weight": ("U32", [2048, 128]),
            "model.layers.0.linear_attn.in_proj_qkv.scales": ("U8", [2048, 32]),
            "model.layers.0.linear_attn.in_proj_z.weight": ("U32", [1024, 128]),
            "model.layers.0.linear_attn.in_proj_z.scales": ("U8", [1024, 32]),
            "model.layers.0.linear_attn.in_proj_a.weight": ("BF16", [8, 1024]),
            "model.layers.0.linear_attn.in_proj_b.weight": ("BF16", [8, 1024]),
            "model.layers.0.linear_attn.conv1d.weight": ("BF16", [2048, 1, 4]),
            "model.layers.0.linear_attn.A_log": ("F32", [8]),
            "model.layers.0.linear_attn.dt_bias": ("BF16", [8]),
            "model.layers.0.linear_attn.norm.weight": ("BF16", [128]),
            "model.layers.0.linear_attn.out_proj.weight": ("U32", [1024, 128]),
            "model.layers.0.linear_attn.out_proj.scales": ("U8", [1024, 32]),
            "model.layers.0.mlp.gate_proj.weight": ("U32", [3072, 128]),
            "model.layers.0.mlp.gate_proj.scales": ("U8", [3072, 32]),
            "model.layers.0.mlp.up_proj.weight": ("U32", [3072, 128]),
            "model.layers.0.mlp.up_proj.scales": ("U8", [3072, 32]),
            "model.layers.0.mlp.down_proj.weight": ("U32", [1024, 384]),
            "model.layers.0.mlp.down_proj.scales": ("U8", [1024, 96]),
        }
        shard2 = {
            "model.layers.1.input_layernorm.weight": ("BF16", [1024]),
            "model.layers.1.post_attention_layernorm.weight": ("BF16", [1024]),
            "model.layers.1.self_attn.q_proj.weight": ("U32", [2048, 128]),
            "model.layers.1.self_attn.q_proj.scales": ("U8", [2048, 32]),
            "model.layers.1.self_attn.k_proj.weight": ("U32", [256, 128]),
            "model.layers.1.self_attn.k_proj.scales": ("U8", [256, 32]),
            "model.layers.1.self_attn.v_proj.weight": ("U32", [256, 128]),
            "model.layers.1.self_attn.v_proj.scales": ("U8", [256, 32]),
            "model.layers.1.self_attn.o_proj.weight": ("U32", [1024, 128]),
            "model.layers.1.self_attn.o_proj.scales": ("U8", [1024, 32]),
            "model.layers.1.self_attn.q_norm.weight": ("BF16", [128]),
            "model.layers.1.self_attn.k_norm.weight": ("BF16", [128]),
            "model.layers.1.mlp.gate_proj.weight": ("U32", [3072, 128]),
            "model.layers.1.mlp.gate_proj.scales": ("U8", [3072, 32]),
            "model.layers.1.mlp.up_proj.weight": ("U32", [3072, 128]),
            "model.layers.1.mlp.up_proj.scales": ("U8", [3072, 32]),
            "model.layers.1.mlp.down_proj.weight": ("U32", [1024, 384]),
            "model.layers.1.mlp.down_proj.scales": ("U8", [1024, 96]),
            "model.norm.weight": ("BF16", [1024]),
            "lm_head.weight": ("BF16", [32768, 1024]),
        }
        _write_header_only_safetensors(hybrid_dir / "model-00001-of-00002.safetensors", shard1)
        _write_header_only_safetensors(hybrid_dir / "model-00002-of-00002.safetensors", shard2)
        weight_map = {
            **{k: "model-00001-of-00002.safetensors" for k in shard1},
            **{k: "model-00002-of-00002.safetensors" for k in shard2},
        }
        (hybrid_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )
        hp = inspect_m5rt_source_fast(hybrid_dir)
        assert hp["eligible"] is True, hp["reason"]
        assert hp["topology"] == "gdn_hybrid_gqa"
        assert hp["linear_attention_layers"] == 1
        assert hp["full_attention_layers"] == 1
        assert hp["has_attn_output_gate"] is True
        assert hp["is_mixed_precision"] is True
        assert hp["elapsed_ms"] < 5.0

        # (b) Additive linear .bias in text backbone rejected (unlike .dt_bias and .biases)
        bias_dir = tmp_path / "Linear-Bias-Blocked"
        bias_dir.mkdir()
        (bias_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen2_5",
                    "hidden_size": 1024,
                    "intermediate_size": 3072,
                    "num_hidden_layers": 1,
                    "vocab_size": 32768,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                    "head_dim": 128,
                    "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
                }
            ),
            encoding="utf-8",
        )
        b_specs = _build_pure_gqa_tensors(
            hidden_size=1024, intermediate_size=3072, num_layers=1, num_q_heads=8, num_kv_heads=2
        )
        b_specs["model.layers.0.self_attn.q_proj.bias"] = ("BF16", [1024])
        _write_header_only_safetensors(bias_dir / "model.safetensors", b_specs)
        bp = inspect_m5rt_source_fast(bias_dir)
        assert bp["eligible"] is False
        assert bp["blocker_code"] == "unsupported_linear_bias"
        assert "model.layers.0.self_attn.q_proj.bias" in bp["reason"]

        # (c) GPTQ (.g_idx) and FP8 (.weight_scale_inv) blockers
        for name_tag, extra_tensor, expected_sub in (
            ("GPTQ", ("model.layers.0.self_attn.q_proj.g_idx", ("I32", [1024])), "GPTQ"),
            ("FP8", ("model.layers.0.self_attn.q_proj.weight_scale_inv", ("F32", [8, 8])), "FP8"),
        ):
            qdir = tmp_path / f"Bad-{name_tag}"
            qdir.mkdir()
            (qdir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "hidden_size": 1024,
                        "intermediate_size": 3072,
                        "num_hidden_layers": 1,
                        "vocab_size": 32768,
                        "num_attention_heads": 8,
                        "num_key_value_heads": 2,
                        "head_dim": 128,
                    }
                ),
                encoding="utf-8",
            )
            _write_header_only_safetensors(qdir / "model.safetensors", {extra_tensor[0]: extra_tensor[1]})
            qp = inspect_m5rt_source_fast(qdir)
            assert qp["eligible"] is False
            assert expected_sub in qp["reason"]
            assert extra_tensor[0] in qp["reason"]

        # (d) Invalid q_proj output dim (3 * q_heads * head_dim) and >256 MoE experts
        bad_q_dir = tmp_path / "Bad-QProj-Shape"
        bad_q_dir.mkdir()
        (bad_q_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "custom_arch",
                    "hidden_size": 1024,
                    "intermediate_size": 3072,
                    "num_hidden_layers": 1,
                    "vocab_size": 32768,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                    "head_dim": 128,
                    "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
                }
            ),
            encoding="utf-8",
        )
        bq_specs = _build_pure_gqa_tensors(
            hidden_size=1024, intermediate_size=3072, num_layers=1, num_q_heads=8, num_kv_heads=2
        )
        _add_affine_linear(bq_specs, "model.layers.0.self_attn.q_proj", 1536, 1024)
        _write_header_only_safetensors(bad_q_dir / "model.safetensors", bq_specs)
        bqp = inspect_m5rt_source_fast(bad_q_dir)
        assert bqp["eligible"] is False
        assert bqp["blocker_code"] == "invalid_q_proj_shape"
        assert "1536" in bqp["reason"]

