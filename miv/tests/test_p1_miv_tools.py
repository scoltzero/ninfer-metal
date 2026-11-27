"""Targeted P1 regression tests for TOOL-1, TOOL-2, TOOL-4, and TOOL-5."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MIV_TOOLS = REPO_ROOT / "miv" / "tools"
if str(MIV_TOOLS) not in sys.path:
    sys.path.insert(0, str(MIV_TOOLS))

import download_b0_models
import build_m5_release
import m5_artifact
import m5_env
import m5_ultra_tuning

MIV_TESTS = REPO_ROOT / "miv" / "tests"
if str(MIV_TESTS) not in sys.path:
    sys.path.insert(0, str(MIV_TESTS))

import d1_quantization_parity
import d2_real_layer_parity
import d2_real_projection_parity


def _bf16_bytes(values: np.ndarray) -> bytes:
    arr = np.asarray(values, dtype=np.float32)
    bits = arr.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype("<u2").tobytes()


def _pack_codes(codes: np.ndarray, bits: int) -> bytes:
    rows, k = codes.shape
    bit_mat = ((codes[:, :, None] >> np.arange(bits, dtype=np.uint8)) & 1).reshape(
        rows, k * bits
    )
    packed = np.packbits(bit_mat, axis=1, bitorder="little")
    return np.ascontiguousarray(packed).tobytes()


def _write_safetensors(
    path: Path, tensors: dict[str, tuple[str, list[int], bytes]]
) -> int:
    header: dict[str, dict] = {}
    offset = 0
    for name in sorted(tensors):
        dtype, shape, payload = tensors[name]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for name in sorted(tensors):
            handle.write(tensors[name][2])
    return path.stat().st_size


def _write_checkpoint(
    model_dir: Path,
    *,
    revision: str,
    config: dict,
    tensors: dict[str, tuple[str, list[int], bytes]],
    with_index: bool = True,
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    shard = "model.safetensors"
    size = _write_safetensors(model_dir / shard, tensors)
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if with_index:
        index = {
            "metadata": {"total_size": size},
            "weight_map": {name: shard for name in sorted(tensors)},
        }
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (model_dir / "REVISION").write_text(revision + "\n", encoding="utf-8")
    return model_dir


def _header_manifest_crc(path: Path) -> str:
    with path.open("rb") as handle:
        header = handle.read(m5_artifact.HEADER_BYTES)
    manifest_crc = m5_artifact.HEADER_STRUCT.unpack(header)[8]
    return f"{manifest_crc:08x}"


def _make_oq_checkpoint(model_dir: Path, *, include_mtp: bool = False) -> Path:
    rng = np.random.default_rng(7)
    n, k = 16, 512
    q4_codes = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
    q6_codes = rng.integers(0, 64, size=(n, k), dtype=np.uint8)
    scales_g64 = np.full((n, k // 64), 0.25, dtype=np.float32)
    biases_g64 = np.full((n, k // 64), -0.5, dtype=np.float32)
    scales_g128 = np.full((n, k // 128), 0.125, dtype=np.float32)
    biases_g128 = np.full((n, k // 128), -1.0, dtype=np.float32)

    tensors: dict[str, tuple[str, list[int], bytes]] = {
        "model.layers.0.self_attn.q_proj.weight": (
            "U32",
            [n, k * 4 // 32],
            _pack_codes(q4_codes, 4),
        ),
        "model.layers.0.self_attn.q_proj.scales": (
            "BF16",
            [n, k // 64],
            _bf16_bytes(scales_g64),
        ),
        "model.layers.0.self_attn.q_proj.biases": (
            "BF16",
            [n, k // 64],
            _bf16_bytes(biases_g64),
        ),
        "model.layers.0.mlp.down_proj.weight": (
            "U32",
            [n, k * 6 // 32],
            _pack_codes(q6_codes, 6),
        ),
        "model.layers.0.mlp.down_proj.scales": (
            "BF16",
            [n, k // 128],
            _bf16_bytes(scales_g128),
        ),
        "model.layers.0.mlp.down_proj.biases": (
            "BF16",
            [n, k // 128],
            _bf16_bytes(biases_g128),
        ),
        "model.norm.weight": ("BF16", [k], _bf16_bytes(np.ones(k, dtype=np.float32))),
    }
    quant_cfg: dict = {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "model.layers.0.mlp.down_proj": {
            "bits": 6,
            "group_size": 128,
            "mode": "affine",
        },
    }
    if include_mtp:
        mtp_codes = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
        tensors["mtp.fc.weight"] = (
            "U32",
            [n, k * 4 // 32],
            _pack_codes(mtp_codes, 4),
        )
        tensors["mtp.fc.scales"] = (
            "BF16",
            [n, k // 64],
            _bf16_bytes(scales_g64),
        )
        tensors["mtp.fc.biases"] = (
            "BF16",
            [n, k // 64],
            _bf16_bytes(biases_g64),
        )
    return _write_checkpoint(
        model_dir,
        revision="rev-oq-test-001",
        config={"model_type": "qwen3_8", "quantization": quant_cfg},
        tensors=tensors,
    )


def _make_bf16_checkpoint(model_dir: Path) -> Path:
    rng = np.random.default_rng(13)
    n, k = 16, 512
    w_q = rng.normal(0.0, 0.25, size=(n, k)).astype(np.float32)
    w_down = rng.normal(0.0, 0.25, size=(n, k)).astype(np.float32)
    tensors: dict[str, tuple[str, list[int], bytes]] = {
        "model.layers.0.self_attn.q_proj.weight": ("BF16", [n, k], _bf16_bytes(w_q)),
        "model.layers.0.mlp.down_proj.weight": ("BF16", [n, k], _bf16_bytes(w_down)),
        "model.norm.weight": ("BF16", [k], _bf16_bytes(np.ones(k, dtype=np.float32))),
    }
    return _write_checkpoint(
        model_dir,
        revision="rev-bf16-test-002",
        config={"model_type": "qwen3_8", "hidden_size": k},
        tensors=tensors,
    )


def _make_mxfp4_checkpoint(model_dir: Path) -> Path:
    rng = np.random.default_rng(23)
    n, k = 16, 512
    codes = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
    packed_nibs = (codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)
    scales = np.full((n, k // 32), 127, dtype=np.uint8)
    q8_codes = rng.integers(0, 256, size=(4, 128), dtype=np.uint8)
    tensors: dict[str, tuple[str, list[int], bytes]] = {
        "model.layers.0.self_attn.q_proj.weight": (
            "U32",
            [n, k // 8],
            np.ascontiguousarray(packed_nibs).tobytes(),
        ),
        "model.layers.0.self_attn.q_proj.scales": (
            "U8",
            [n, k // 32],
            scales.tobytes(),
        ),
        "model.layers.0.mlp.gate.weight": (
            "U32",
            [4, 128 * 8 // 32],
            _pack_codes(q8_codes, 8),
        ),
        "model.layers.0.mlp.gate.scales": (
            "BF16",
            [4, 2],
            _bf16_bytes(np.full((4, 2), 0.0625, dtype=np.float32)),
        ),
        "model.layers.0.mlp.gate.biases": (
            "BF16",
            [4, 2],
            _bf16_bytes(np.full((4, 2), -1.0, dtype=np.float32)),
        ),
    }
    return _write_checkpoint(
        model_dir,
        revision="rev-mx-test-003",
        config={
            "model_type": "qwen3_6_moe",
            "quantization": {
                "bits": 4,
                "group_size": 32,
                "mode": "mxfp4",
                "model.layers.0.mlp.gate": {"bits": 8, "group_size": 64},
            },
        },
        tensors=tensors,
    )


# ---------------------------------------------------------------------------
# TOOL-1: --nax-pretile conversion sidecar, orphan cleanup & report metadata
# ---------------------------------------------------------------------------


def test_tool1_convert_nax_pretile_writes_final_sidecar_and_cleans_tmp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _make_oq_checkpoint(tmp_path / "src_oq")
    out_art = tmp_path / "out" / "model.omlxm5"
    args = m5_artifact.build_parser().parse_args(
        [
            "convert",
            "--source",
            str(src),
            "--output",
            str(out_art),
            "--source-repo",
            "test/oq-pretile",
            "--expected-revision",
            "rev-oq-test-001",
            "--profile",
            "oq_affine",
            "--nax-pretile",
        ]
    )
    m5_artifact.command_convert(args)
    stdout_report = json.loads(capsys.readouterr().out)

    # No orphan .untiled.tmp or .untiled.tmp.conversion.json files remain.
    out_files = sorted(p.name for p in out_art.parent.iterdir())
    assert out_files == ["model.omlxm5", "model.omlxm5.conversion.json"]

    summary, manifest = m5_artifact.read_artifact(out_art, verify_payload=True)
    conv_report = json.loads(
        (out_art.parent / "model.omlxm5.conversion.json").read_text(encoding="utf-8")
    )
    actual_sha = m5_artifact.sha256_file(out_art)
    actual_crc = _header_manifest_crc(out_art)
    expected_cfg_sha = m5_artifact.sha256_file(src / "config.json")

    for rep in (stdout_report, conv_report):
        assert rep["artifact"] == str(out_art.resolve())
        assert rep["artifact_bytes"] == out_art.stat().st_size
        assert rep["artifact_sha256"] == actual_sha
        assert rep["manifest_crc32"] == actual_crc
        assert rep["manifest_sha256"] == summary["manifest_sha256"]
        assert rep["payload_bytes"] == summary["payload_bytes"]
        assert rep["pretiled_objects"] == 2
        assert rep["source_config_sha256"] == expected_cfg_sha
        assert rep["source_revision"] == "rev-oq-test-001"

    assert len(conv_report["objects"]) == len(manifest["objects"])
    for rep_obj, man_obj in zip(conv_report["objects"], manifest["objects"], strict=True):
        assert rep_obj["id"] == man_obj["id"]
        assert rep_obj["name"] == man_obj["name"]
        assert rep_obj["bytes"] == man_obj["object_bytes"]
        assert rep_obj["sha256"] == man_obj["object_sha256"]


def test_tool1_extract_mtp_nax_pretile_updates_all_hashes_and_cleans_tmp(
    tmp_path: Path,
) -> None:
    src = _make_oq_checkpoint(tmp_path / "src_oq_mtp", include_mtp=True)
    mtp_out = tmp_path / "out" / "mtp.omlxm5"

    report = m5_artifact.extract_mtp_artifact(
        src,
        mtp_out,
        "test/oq-mtp",
        expected_revision="rev-oq-test-001",
        profile="oq_affine",
        nax_pretile=True,
    )

    out_files = sorted(p.name for p in mtp_out.parent.iterdir())
    assert out_files == ["mtp.omlxm5"]

    summary, manifest = m5_artifact.read_artifact(mtp_out, verify_payload=True)
    assert report["artifact"] == str(mtp_out.resolve())
    assert report["artifact_bytes"] == mtp_out.stat().st_size
    assert report["artifact_sha256"] == m5_artifact.sha256_file(mtp_out)
    assert report["manifest_crc32"] == _header_manifest_crc(mtp_out)
    assert report["manifest_sha256"] == summary["manifest_sha256"]
    assert report["payload_bytes"] == summary["payload_bytes"]
    assert report["pretiled_objects"] == 1
    assert manifest["objects"][0]["layout"] == "oq_nax_tiled_g64_v1"


# ---------------------------------------------------------------------------
# TOOL-2: qda-search --output-artifact provenance + already-quantized sources
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nax_pretile", [False, True])
def test_tool2_qda_bf16_preserves_source_provenance_and_sidecar(
    tmp_path: Path, nax_pretile: bool
) -> None:
    src = _make_bf16_checkpoint(tmp_path / "src_bf16")
    out_art = tmp_path / "out" / f"qda_{'pt' if nax_pretile else 'raw'}.omlxm5"

    recipe = m5_artifact.qda_search_checkpoint(src, target_bpw=4.5)
    rep = m5_artifact.quantize_source_with_recipe_to_artifact(
        src, recipe, out_art, "test/bf16-qda", nax_pretile=nax_pretile
    )

    out_files = sorted(p.name for p in out_art.parent.iterdir())
    assert out_files == [out_art.name, f"{out_art.name}.conversion.json"]

    summary, manifest = m5_artifact.read_artifact(out_art, verify_payload=True)
    conv = json.loads(
        (out_art.parent / f"{out_art.name}.conversion.json").read_text(encoding="utf-8")
    )

    expected_cfg_sha = m5_artifact.sha256_file(src / "config.json")
    expected_idx_sha = m5_artifact.sha256_file(src / "model.safetensors.index.json")

    assert manifest["model"]["source_config_sha256"] == expected_cfg_sha
    assert manifest["model"]["source_index_sha256"] == expected_idx_sha
    assert manifest["model"]["source_revision"] == "rev-bf16-test-002"

    for doc in (rep, conv):
        assert doc["artifact"] == str(out_art.resolve())
        assert doc["artifact_bytes"] == out_art.stat().st_size
        assert doc["artifact_sha256"] == m5_artifact.sha256_file(out_art)
        assert doc["manifest_crc32"] == _header_manifest_crc(out_art)
        assert doc["manifest_sha256"] == summary["manifest_sha256"]
        assert doc["payload_bytes"] == summary["payload_bytes"]
        assert doc["source_config_sha256"] == expected_cfg_sha
        assert doc["source_index_sha256"] == expected_idx_sha
        assert doc["source_revision"] == "rev-bf16-test-002"
    if nax_pretile:
        assert rep["pretiled_objects"] == 2


@pytest.mark.parametrize("nax_pretile", [False, True])
def test_tool2_qda_already_quantized_oq_checkpoint_succeeds(
    tmp_path: Path, nax_pretile: bool
) -> None:
    src = _make_oq_checkpoint(tmp_path / "src_quant_oq")
    out_art = tmp_path / "out" / f"oq_qda_{'pt' if nax_pretile else 'raw'}.omlxm5"

    # Choose a bpw that assigns different tiers than the source checkpoint (Q4/Q6)
    # to ensure passthrough preserves the source tensors' real quantization specs.
    recipe = m5_artifact.qda_search_checkpoint(src, target_bpw=3.5)
    rep = m5_artifact.quantize_source_with_recipe_to_artifact(
        src, recipe, out_art, "test/already-oq", nax_pretile=nax_pretile
    )

    summary, manifest = m5_artifact.read_artifact(out_art, verify_payload=True)
    assert manifest["model"]["source_config_sha256"] == m5_artifact.sha256_file(
        src / "config.json"
    )
    assert manifest["model"]["source_revision"] == "rev-oq-test-001"
    assert rep["artifact_sha256"] == m5_artifact.sha256_file(out_art)
    assert rep["manifest_sha256"] == summary["manifest_sha256"]
    formats = {obj["name"]: obj["format"] for obj in manifest["objects"]}
    assert formats["model.layers.0.self_attn.q_proj"] == "oq_affine_q4_g64_bf16_v1"
    assert formats["model.layers.0.mlp.down_proj"] == "oq_affine_q6_g128_bf16_v1"


def test_tool2_qda_already_quantized_mxfp4_checkpoint_succeeds(tmp_path: Path) -> None:
    src = _make_mxfp4_checkpoint(tmp_path / "src_quant_mx")
    out_art = tmp_path / "out" / "mx_qda.omlxm5"

    recipe = m5_artifact.qda_search_checkpoint(src, target_bpw=4.25)
    rep = m5_artifact.quantize_source_with_recipe_to_artifact(
        src, recipe, out_art, "test/already-mx", nax_pretile=True
    )

    _, manifest = m5_artifact.read_artifact(out_art, verify_payload=True)
    assert manifest["model"]["source_config_sha256"] == m5_artifact.sha256_file(
        src / "config.json"
    )
    assert manifest["model"]["source_revision"] == "rev-mx-test-003"
    assert rep["artifact_sha256"] == m5_artifact.sha256_file(out_art)
    formats = {obj["name"]: obj["format"] for obj in manifest["objects"]}
    assert formats["model.layers.0.self_attn.q_proj"] == "mxfp4_e2m1_g32_e8m0_v1"
    assert formats["model.layers.0.mlp.gate"] == "oq_affine_q8_g64_bf16_v1"


# ---------------------------------------------------------------------------
# TOOL-4: download_b0_models.py full 64-hex SHA-256 & pin validation
# ---------------------------------------------------------------------------


def test_tool4_all_pins_have_valid_64_hex_sha256_and_validation_enforced() -> None:
    assert (
        download_b0_models.PINS["q38_27_oq"].lfs_sha256[
            "model-00003-of-00004.safetensors"
        ]
        == "d3555aff4d37c9b4d76ad42a654a62ef3054cded714922d011968b410fa9359f"
    )
    for key, pin in download_b0_models.PINS.items():
        for fname, digest in pin.lfs_sha256.items():
            assert re.fullmatch(r"[0-9a-f]{64}", digest), (
                f"{key}/{fname} invalid digest: {digest!r}"
            )

    with pytest.raises(ValueError, match="expected 64 lowercase hex chars"):
        download_b0_models.ModelPin(
            key="bad_len",
            repo="org/model",
            revision="0" * 40,
            files=("model.safetensors",),
            lfs_sha256={"model.safetensors": "a" * 60},
        )

    with pytest.raises(ValueError, match="expected 64 lowercase hex chars"):
        download_b0_models.ModelPin(
            key="bad_hex",
            repo="org/model",
            revision="0" * 40,
            files=("model.safetensors",),
            lfs_sha256={"model.safetensors": "G" * 64},
        )


# ---------------------------------------------------------------------------
# TOOL-5: no hardcoded root tools/ paths across miv/tools/
# ---------------------------------------------------------------------------


def test_tool5_stage_a_scripts_have_no_root_tools_invocations() -> None:
    for script_name in (
        "run_stage_a_27b_pipeline.sh",
        "run_stage_a_batch2_batch3.sh",
    ):
        text = (MIV_TOOLS / script_name).read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            assert not re.search(r"(^|[\s\"'])\.?/?tools/", stripped), (
                f"{script_name}:{lineno} still references root tools/: {stripped}"
            )


def test_tool5_m5_env_works_in_clean_clone_without_root_tools(tmp_path: Path) -> None:
    fake_repo = tmp_path / "clean_clone"
    (fake_repo / "miv" / "tools").mkdir(parents=True)
    (fake_repo / ".gitignore").write_text("/tools/\n", encoding="utf-8")
    (fake_repo / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.24)\n", encoding="utf-8")
    (fake_repo / "CMakePresets.json").write_text("{}\n", encoding="utf-8")
    shutil.copy2(MIV_TOOLS / "m5_env.py", fake_repo / "miv" / "tools" / "m5_env.py")
    (fake_repo / "miv" / "tools" / "validate_d0_assets.py").write_text(
        "# stub\n", encoding="utf-8"
    )

    tree_id_1 = m5_env.source_tree_id(fake_repo)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", tree_id_1)

    (fake_repo / "miv" / "tools" / "validate_d0_assets.py").write_text(
        "# stub v2\n", encoding="utf-8"
    )
    tree_id_2 = m5_env.source_tree_id(fake_repo)
    assert tree_id_2 != tree_id_1

    # Also invoke copied m5_env.py CLI in the clean clone (no tools/ and no models/).
    out_json = tmp_path / "env.json"
    proc = subprocess.run(
        [sys.executable, str(fake_repo / "miv" / "tools" / "m5_env.py"), "--output", str(out_json)],
        cwd=fake_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["source_id"] == tree_id_2
    assert payload["model_revision"] is None


def test_tool5_m5_ultra_tuning_does_not_depend_on_missing_b2_scripts(
    tmp_path: Path,
) -> None:
    assert m5_ultra_tuning.discover_b2_scripts(tmp_path) == []

    # Plan-only run must not emit nonexistent root tools/reproduce_b2_*.sh paths.
    plan_out = tmp_path / "plan_out"
    proc = subprocess.run(
        [
            sys.executable,
            str(MIV_TOOLS / "m5_ultra_tuning.py"),
            "--output-dir",
            str(plan_out),
            "--build-dir",
            str(tmp_path / "nonexistent_build"),
            "--plan-only",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(proc.stdout)
    assert plan["scripts"] == []

    # Default --stages all must skip absent legacy b2 scripts instead of raising FileNotFoundError.
    run_out = tmp_path / "run_out"
    h1_l_dir = run_out / "h1_l"
    h1_l_dir.mkdir(parents=True)
    (h1_l_dir / "sample.json").write_text(
        json.dumps(
            {
                "schema": "omlx-m5-projection-candidate-v1",
                "artifact": "sample.omlxm5",
                "object": "layers.0.q_proj",
                "format": "oq_affine_q4_g64_bf16_v1",
                "t": 1,
                "n": 64,
                "k": 512,
                "candidate": "row_exact_portable",
                "old_gpu_ms": [1.0, 1.1],
                "new_gpu_ms": [0.8, 0.9],
            }
        ),
        encoding="utf-8",
    )
    proc_run = subprocess.run(
        [
            sys.executable,
            str(MIV_TOOLS / "m5_ultra_tuning.py"),
            "--output-dir",
            str(run_out),
            "--build-dir",
            str(tmp_path / "nonexistent_build"),
            "--allow-non-target",
            "--iterations",
            "2",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    gate = json.loads(proc_run.stdout)
    assert "h1_complete" in gate
    result = json.loads((run_out / "tuning.json").read_text(encoding="utf-8"))
    assert result["scripts"] == []
    assert result["eligible_curve_count"] == 2


@pytest.mark.parametrize(
    "module_name",
    [
        "m5_ultra_tuning",
        "m5_ultra_capability",
        "m5_ultra_full_model",
        "m5_hw_trace_stats",
        "bench_omlx_vs_m5rt",
    ],
)
def test_tool5_python_tools_import_without_root_tools_dir(
    tmp_path: Path, module_name: str
) -> None:
    # Copy the module + its miv/tools dependencies into an isolated fake repo with NO root tools/.
    src_text = (MIV_TOOLS / f"{module_name}.py").read_text(encoding="utf-8")
    assert 'ROOT / "tools"' not in src_text and "ROOT / 'tools'" not in src_text
    fake_repo = tmp_path / f"iso_{module_name}"
    fake_miv_tools = fake_repo / "miv" / "tools"
    fake_miv_tools.mkdir(parents=True)
    for helper in ("m5_target_guard.py", "m5_silicon_profile.py", f"{module_name}.py"):
        shutil.copy2(MIV_TOOLS / helper, fake_miv_tools / helper)

    # Import by path from fake_repo CWD with clean sys.path (no miv/tools pre-inserted).
    code = (
        "import importlib.util, pathlib, sys; "
        f"p = pathlib.Path({str(fake_miv_tools / (module_name + '.py'))!r}); "
        f"spec = importlib.util.spec_from_file_location({module_name!r}, p); "
        "mod = importlib.util.module_from_spec(spec); "
        f"sys.modules[{module_name!r}] = mod; "
        "spec.loader.exec_module(mod); "
        "bad = [e for e in sys.path if e.endswith('/tools') and not e.endswith('/miv/tools')]; "
        "assert not bad, f'unexpected root tools on sys.path: {bad}'"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=fake_repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"


# ---------------------------------------------------------------------------
# TOOL-3: mx_nax_tiled_bk128_v1 & oq_nax_tiled_g64_v1 geometry parity with C++
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "padded_k"),
    [
        ([16, 256], 256),        # logical_k < 512
        ([16, 576], 640),        # logical_k % 128 != 0
        ([2, 16, 512], 512),     # rank != 2
    ],
)
def test_tool3_mx_nax_tiled_rejects_invalid_geometry_in_validator_and_decoder(
    tmp_path: Path, shape: list[int], padded_k: int
) -> None:
    record = {
        "id": 0,
        "name": "bad_mx",
        "format": "mxfp4_e2m1_g32_e8m0_v1",
        "layout": "mx_nax_tiled_bk128_v1",
        "shape": shape,
        "padded_k": padded_k,
        "planes": [
            {"kind": "nax_data", "offset": 0, "bytes": 16 * padded_k // 2},
            {"kind": "nax_scale", "offset": 16 * padded_k // 2, "bytes": 16 * padded_k // 32},
        ],
    }
    with pytest.raises(ValueError, match="MXFP4 NAX tiled geometry mismatch"):
        m5_artifact._expected_plane_bytes(record)

    # Also verify decode_artifact_row rejects directly on an unvalidated manifest.
    dummy_art = tmp_path / "dummy.omlxm5"
    dummy_art.write_bytes(b"\x00" * (m5_artifact.HEADER_BYTES + 4096))
    with pytest.raises(ValueError, match="MXFP4 NAX tiled geometry mismatch"):
        m5_artifact.decode_artifact_row(
            dummy_art, {"objects": [record]}, "bad_mx", 0, 0, 16
        )


def test_tool3_oq_nax_tiled_rejects_rank3_and_valid_mx_passes(tmp_path: Path) -> None:
    bad_oq = {
        "id": 0,
        "name": "bad_oq_rank3",
        "format": "oq_affine_q4_g64_bf16_v1",
        "layout": "oq_nax_tiled_g64_v1",
        "shape": [2, 16, 512],
        "padded_k": 512,
        "planes": [],
    }
    with pytest.raises(ValueError, match="affine NAX tiled geometry mismatch"):
        m5_artifact._expected_plane_bytes(bad_oq)

    dummy_art = tmp_path / "dummy_oq.omlxm5"
    dummy_art.write_bytes(b"\x00" * (m5_artifact.HEADER_BYTES + 4096))
    with pytest.raises(ValueError, match="affine NAX tiled geometry mismatch"):
        m5_artifact.decode_artifact_row(
            dummy_art, {"objects": [bad_oq]}, "bad_oq_rank3", 0, 0, 16
        )

    valid_mx = {
        "id": 0,
        "name": "good_mx",
        "format": "mxfp4_e2m1_g32_e8m0_v1",
        "layout": "mx_nax_tiled_bk128_v1",
        "shape": [16, 512],
        "padded_k": 512,
    }
    assert m5_artifact._expected_plane_bytes(valid_mx) == {
        "nax_data": 16 * 512 // 2,
        "nax_scale": 16 * 512 // 32,
    }


# ---------------------------------------------------------------------------
# TOOL-6: build_m5_release.py install.sh atomic macOS symlink flip
# ---------------------------------------------------------------------------


def _make_stub_release_bundle(bundle_dir: Path, version: str) -> Path:
    for sub in ("bin", "metallib", "python", "tools", "calibrations", "config", "scripts"):
        (bundle_dir / sub).mkdir(parents=True, exist_ok=True)
    for exe in ("mseries_backendd", "ninfer-metal-serve", "omlx_m5_probe"):
        path = bundle_dir / "bin" / exe
        path.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
        path.chmod(0o755)
    (bundle_dir / "metallib" / "dummy.metallib").write_bytes(b"MTLB")
    (bundle_dir / "tools" / "dummy.py").write_text("# stub\n", encoding="utf-8")
    (bundle_dir / "RELEASE-MANIFEST.json").write_text("{}\n", encoding="utf-8")
    (bundle_dir / "provider.json").write_text(
        json.dumps({"version": version}) + "\n", encoding="utf-8"
    )
    (bundle_dir / "README-RELEASE.md").write_text("# test\n", encoding="utf-8")
    install_sh = bundle_dir / "install.sh"
    install_sh.write_text(
        build_m5_release.build_install_script(version), encoding="utf-8"
    )
    install_sh.chmod(0o755)
    return install_sh


def test_tool6_install_script_atomic_symlink_flip_on_macos(tmp_path: Path) -> None:
    script_v1 = build_m5_release.build_install_script("v0.8.0-test.1")
    assert "mv -T" not in script_v1

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    provider_root = (
        fake_home / "Library" / "Application Support" / "oMLX" / "Providers" / "io.ninfer.metal"
    )
    env = dict(os.environ, HOME=str(fake_home))

    # 1) Fresh install
    install_v1 = _make_stub_release_bundle(tmp_path / "bundle_v1", "v0.8.0-test.1")
    subprocess.run(["zsh", "-n", str(install_v1)], check=True)
    subprocess.run(["bash", "-n", str(install_v1)], check=True)
    res1 = subprocess.run(
        [str(install_v1)], env=env, capture_output=True, text=True
    )
    assert res1.returncode == 0, res1.stderr
    current_link = provider_root / "current"
    assert current_link.is_symlink()
    assert current_link.resolve() == (provider_root / "versions" / "v0.8.0-test.1").resolve()
    assert not any("current.tmp" in p.name for p in provider_root.iterdir())

    # 2) Upgrade install atomically flips existing symlink without leaving temp files
    install_v2 = _make_stub_release_bundle(tmp_path / "bundle_v2", "v0.8.0-test.2")
    res2 = subprocess.run(
        [str(install_v2)], env=env, capture_output=True, text=True
    )
    assert res2.returncode == 0, res2.stderr
    assert current_link.is_symlink()
    assert current_link.resolve() == (provider_root / "versions" / "v0.8.0-test.2").resolve()
    assert not any("current.tmp" in p.name for p in provider_root.iterdir())

    # 3) Real directory at current must fail safely without nesting a symlink inside it
    current_link.unlink()
    current_link.mkdir()
    res3 = subprocess.run(
        [str(install_v2)], env=env, capture_output=True, text=True
    )
    assert res3.returncode != 0
    assert "real directory" in res3.stderr
    assert list(current_link.iterdir()) == []
    assert not any("current.tmp" in p.name for p in provider_root.iterdir())


# ---------------------------------------------------------------------------
# TEST-4: d2_real_layer_parity, d2_real_projection_parity, d1_quantization_parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mod",
    [d2_real_layer_parity, d2_real_projection_parity],
)
def test_tool_test4_d2_parity_catches_scale_error_and_nan(mod) -> None:
    expected = [0.25, -0.5, 0.75, 1.0]
    sim, err = mod.verify_parity("ok", list(expected), expected)
    assert sim == pytest.approx(1.0)
    assert err == 0.0

    # Global positive scaling preserves cosine == 1.0 but must fail max_abs_tol!
    scaled = [2.0 * x for x in expected]
    assert mod.cosine(scaled, expected) == pytest.approx(1.0)
    with pytest.raises(AssertionError, match="max_abs_error"):
        mod.verify_parity("scaled", scaled, expected)

    # NaN in actual or expected (both index 0 and non-zero index) must fail.
    with pytest.raises(AssertionError, match="non-finite"):
        mod.verify_parity("nan_actual_0", [float("nan"), -0.5, 0.75, 1.0], expected)
    with pytest.raises(AssertionError, match="non-finite"):
        mod.verify_parity("nan_actual_mid", [0.25, -0.5, float("nan"), 1.0], expected)
    with pytest.raises(AssertionError, match="non-finite"):
        mod.verify_parity("nan_expected", expected, [0.25, float("nan"), 0.75, 1.0])
    with pytest.raises(AssertionError, match="non-finite"):
        mod.verify_parity("inf_actual", [0.25, float("inf"), 0.75, 1.0], expected)

    # Length mismatch / empty must fail.
    with pytest.raises(AssertionError, match="length mismatch"):
        mod.verify_parity("truncated", expected[:2], expected)


def test_tool_test4_d1_quantization_parity_catches_truncation_and_nan() -> None:
    ref = [float(i) * 0.01 for i in range(64)]
    err = d1_quantization_parity.verify_sample_parity("m", 4, list(ref), ref)
    assert err == 0.0

    # Truncated output (previously silently passed due to non-strict zip) must fail.
    with pytest.raises(AssertionError, match="sample length mismatch"):
        d1_quantization_parity.verify_sample_parity("m", 4, ref[:1], ref)
    with pytest.raises(AssertionError, match="sample length mismatch"):
        d1_quantization_parity.verify_sample_parity("m", 4, ref[:63], ref[:63])

    # Non-first-element NaN (previously masked by max([0.0, nan]) == 0.0) must fail.
    bad_mid = list(ref)
    bad_mid[37] = float("nan")
    with pytest.raises(AssertionError, match="non-finite"):
        d1_quantization_parity.verify_sample_parity("m", 4, bad_mid, ref)
    with pytest.raises(AssertionError, match="non-finite"):
        d1_quantization_parity.verify_sample_parity("m", 4, ref, bad_mid)

    # First-element NaN (previously bypassed `nan > 1e-4` == False) must fail.
    bad_first = list(ref)
    bad_first[0] = float("nan")
    with pytest.raises(AssertionError, match="non-finite"):
        d1_quantization_parity.verify_sample_parity("m", 4, bad_first, ref)

    # Error above 1e-4 must fail.
    bad_err = list(ref)
    bad_err[10] += 2e-4
    with pytest.raises(AssertionError, match="decoder mismatch"):
        d1_quantization_parity.verify_sample_parity("m", 4, bad_err, ref)


# ---------------------------------------------------------------------------
# TASK-1: affine8 (q8_g64 & q8_g128) NAX pre-tile, mixed MXFP4+affine8 & matmul
# ---------------------------------------------------------------------------


def _read_plane_raw(path: Path, manifest: dict, obj_name: str, kind: str) -> bytes:
    rec = next(item for item in manifest["objects"] if item["name"] == obj_name)
    plane = next(p for p in rec["planes"] if p["kind"] == kind)
    with path.open("rb") as handle:
        payload_offset = m5_artifact.HEADER_STRUCT.unpack(
            handle.read(m5_artifact.HEADER_BYTES)
        )[5]
        handle.seek(payload_offset + int(plane["offset"]))
        return handle.read(int(plane["bytes"]))


def _bf16_round_f32(values: np.ndarray) -> np.ndarray:
    raw = _bf16_bytes(values)
    u16 = np.frombuffer(raw, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(values.shape)


def _dequant_affine_ref(
    codes: np.ndarray, scales_f32: np.ndarray, biases_f32: np.ndarray, group_size: int
) -> np.ndarray:
    n, k = codes.shape
    sc = np.repeat(_bf16_round_f32(scales_f32), group_size, axis=1)
    bi = np.repeat(_bf16_round_f32(biases_f32), group_size, axis=1)
    assert sc.shape == (n, k) and bi.shape == (n, k)
    return sc * codes.astype(np.float32) + bi


def _dequant_mxfp4_ref(codes: np.ndarray, scales_e8m0: np.ndarray) -> np.ndarray:
    lut = np.array(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=np.float32,
    )
    vals = lut[codes & 0xF]
    exps = np.where(scales_e8m0 == 0, -127, scales_e8m0.astype(np.int32) - 127)
    sc = np.ldexp(np.ones_like(exps, dtype=np.float32), exps)
    return vals * np.repeat(sc, 32, axis=1)


def _decode_full_matrix(path: Path, manifest: dict, obj_name: str) -> np.ndarray:
    rec = next(item for item in manifest["objects"] if item["name"] == obj_name)
    n, k = int(rec["shape"][0]), int(rec["shape"][1])
    rows = [
        m5_artifact.decode_artifact_row(path, manifest, obj_name, r, 0, k)
        for r in range(n)
    ]
    return np.asarray(rows, dtype=np.float32)


def _simulate_affine8_nax_tiled_matmul(
    x_f32: np.ndarray,
    nax_codes_bytes: bytes,
    nax_sb_bytes: bytes,
    n: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Metal 4 NAX group-64 tiled matmul directly from raw nax_codes & nax_sb planes.

    Returns (y_valid [T, n], y_padded_tail [T, padded_n - n]).
    """
    bn = 32 if n >= 65536 else 16
    padded_n = m5_artifact.align_up(n, bn)
    groups = k // 64
    x_bf16 = _bf16_round_f32(x_f32)
    t_rows = x_bf16.shape[0]

    codes_tiled = np.frombuffer(nax_codes_bytes, dtype=np.uint8).reshape(
        padded_n // bn, groups, bn, 64
    )
    sb_u32 = np.frombuffer(nax_sb_bytes, dtype="<u4").reshape(groups, padded_n)
    sc_f32 = ((sb_u32 & 0xFFFF) << 16).view(np.float32)
    bi_f32 = (sb_u32 & 0xFFFF0000).view(np.float32)

    acc = np.zeros((t_rows, padded_n), dtype=np.float32)
    for g in range(groups):
        xg = x_bf16[:, g * 64 : (g + 1) * 64]  # [T, 64]
        xsum = np.sum(xg, axis=1, keepdims=True)  # [T, 1]
        cg = codes_tiled[:, g, :, :].reshape(padded_n, 64).astype(np.float32)
        dot = xg @ cg.T  # [T, padded_n]
        acc += dot * sc_f32[g][None, :] + xsum * bi_f32[g][None, :]
    return acc[:, :n], acc[:, n:]


def test_affine8_pure_q8_g64_and_g128_nax_pretile_roundtrip_and_matmul(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rng = np.random.default_rng(101)
    # Use unaligned row counts (19 -> padded_n=32, 24 -> padded_n=32) and
    # non-128-multiple K=576 (padded_k=640) for q8_g64 alongside K=512 for q8_g128.
    n_q, k_q = 19, 576
    n_d, k_d = 24, 512
    q_codes = rng.integers(0, 256, size=(n_q, k_q), dtype=np.uint8)
    q_scales = rng.uniform(0.01, 0.2, size=(n_q, k_q // 64)).astype(np.float32)
    q_biases = rng.uniform(-3.0, 1.0, size=(n_q, k_q // 64)).astype(np.float32)

    d_codes = rng.integers(0, 256, size=(n_d, k_d), dtype=np.uint8)
    d_scales = rng.uniform(0.01, 0.2, size=(n_d, k_d // 128)).astype(np.float32)
    d_biases = rng.uniform(-2.0, 2.0, size=(n_d, k_d // 128)).astype(np.float32)

    emb_codes = rng.integers(0, 256, size=(16, 512), dtype=np.uint8)
    emb_scales = np.full((16, 512 // 64), 0.0625, dtype=np.float32)
    emb_biases = np.full((16, 512 // 64), -1.0, dtype=np.float32)

    exp_codes = rng.integers(0, 256, size=(2 * 4, 512), dtype=np.uint8)
    exp_scales = np.full((2 * 4, 512 // 64), 0.125, dtype=np.float32)
    exp_biases = np.full((2 * 4, 512 // 64), -0.5, dtype=np.float32)

    tensors: dict[str, tuple[str, list[int], bytes]] = {
        "model.embed_tokens.weight": (
            "U32",
            [16, 512 * 8 // 32],
            _pack_codes(emb_codes, 8),
        ),
        "model.embed_tokens.scales": ("BF16", [16, 512 // 64], _bf16_bytes(emb_scales)),
        "model.embed_tokens.biases": ("BF16", [16, 512 // 64], _bf16_bytes(emb_biases)),
        "model.layers.0.self_attn.q_proj.weight": (
            "U32",
            [n_q, k_q * 8 // 32],
            _pack_codes(q_codes, 8),
        ),
        "model.layers.0.self_attn.q_proj.scales": (
            "BF16",
            [n_q, k_q // 64],
            _bf16_bytes(q_scales),
        ),
        "model.layers.0.self_attn.q_proj.biases": (
            "BF16",
            [n_q, k_q // 64],
            _bf16_bytes(q_biases),
        ),
        "model.layers.0.mlp.down_proj.weight": (
            "U32",
            [n_d, k_d * 8 // 32],
            _pack_codes(d_codes, 8),
        ),
        "model.layers.0.mlp.down_proj.scales": (
            "BF16",
            [n_d, k_d // 128],
            _bf16_bytes(d_scales),
        ),
        "model.layers.0.mlp.down_proj.biases": (
            "BF16",
            [n_d, k_d // 128],
            _bf16_bytes(d_biases),
        ),
        "model.layers.0.mlp.switch_mlp.gate_proj.weight": (
            "U32",
            [2, 4, 512 * 8 // 32],
            _pack_codes(exp_codes, 8),
        ),
        "model.layers.0.mlp.switch_mlp.gate_proj.scales": (
            "BF16",
            [2, 4, 512 // 64],
            _bf16_bytes(exp_scales),
        ),
        "model.layers.0.mlp.switch_mlp.gate_proj.biases": (
            "BF16",
            [2, 4, 512 // 64],
            _bf16_bytes(exp_biases),
        ),
        "model.norm.weight": (
            "BF16",
            [512],
            _bf16_bytes(np.ones(512, dtype=np.float32)),
        ),
    }
    src = _write_checkpoint(
        tmp_path / "src_pure_q8",
        revision="rev-pure-q8-001",
        config={
            "model_type": "qwen3_8",
            "quantization": {
                "bits": 8,
                "group_size": 64,
                "mode": "affine",
                "model.layers.0.mlp.down_proj": {
                    "bits": 8,
                    "group_size": 128,
                    "mode": "affine",
                },
            },
        },
        tensors=tensors,
    )

    untiled_art = tmp_path / "out_untiled" / "model.omlxm5"
    pretiled_art = tmp_path / "out_pretiled" / "model.omlxm5"
    two_step_art = tmp_path / "out_twostep" / "model.omlxm5"

    # 1) Untiled convert + two-step pretile vs single-step convert --nax-pretile
    m5_artifact.command_convert(
        m5_artifact.build_parser().parse_args(
            [
                "convert",
                "--source",
                str(src),
                "--output",
                str(untiled_art),
                "--source-repo",
                "test/pure-q8",
                "--expected-revision",
                "rev-pure-q8-001",
                "--artifact-version",
                "1",
            ]
        )
    )
    _ = capsys.readouterr()

    m5_artifact.command_convert(
        m5_artifact.build_parser().parse_args(
            [
                "convert",
                "--source",
                str(src),
                "--output",
                str(pretiled_art),
                "--source-repo",
                "test/pure-q8",
                "--expected-revision",
                "rev-pure-q8-001",
                "--artifact-version",
                "1",
                "--nax-pretile",
            ]
        )
    )
    pt_stdout = json.loads(capsys.readouterr().out)
    assert pt_stdout["pretiled_objects"] == 2

    two_step_art.parent.mkdir(parents=True, exist_ok=True)
    pt_summary = m5_artifact.pretile_artifact(untiled_art, two_step_art)
    assert pt_summary["pretiled_objects"] == 2
    assert pretiled_art.read_bytes() == two_step_art.read_bytes()

    _, untiled_manifest = m5_artifact.read_artifact(untiled_art, verify_payload=True)
    _, pretiled_manifest = m5_artifact.read_artifact(pretiled_art, verify_payload=True)

    objs = {obj["name"]: obj for obj in pretiled_manifest["objects"]}
    # Embedding and 3D expert tensors must remain untiled; rank-2 linear Q8 layers pre-tile.
    assert objs["model.embed_tokens"]["layout"] == "oq_rowsplit_k128_v1"
    assert [p["kind"] for p in objs["model.embed_tokens"]["planes"]] == [
        "code",
        "scale",
        "bias",
    ]
    assert (
        objs["model.layers.0.mlp.switch_mlp.gate_proj"]["layout"]
        == "oq_expert_rowsplit_k128_v1"
    )

    q_obj = objs["model.layers.0.self_attn.q_proj"]
    assert q_obj["format"] == "oq_affine_q8_g64_bf16_v1"
    assert q_obj["layout"] == "oq_nax_tiled_g64_v1"
    assert q_obj["padded_k"] == 640
    assert [(p["kind"], p["bytes"]) for p in q_obj["planes"]] == [
        ("nax_codes", 32 * 576),
        ("nax_sb", 32 * (576 // 64) * 4),
    ]

    d_obj = objs["model.layers.0.mlp.down_proj"]
    assert d_obj["format"] == "oq_affine_q8_g128_bf16_v1"
    assert d_obj["layout"] == "oq_nax_tiled_g64_v1"
    assert d_obj["padded_k"] == 512
    assert [(p["kind"], p["bytes"]) for p in d_obj["planes"]] == [
        ("nax_codes", 32 * 512),
        ("nax_sb", 32 * (512 // 64) * 4),
    ]

    # Verify g128 nax_sb repeats each 128-group (scale, bias) across 2 consecutive g64 slots.
    d_sb_raw = _read_plane_raw(
        pretiled_art, pretiled_manifest, "model.layers.0.mlp.down_proj", "nax_sb"
    )
    d_sb_u32 = np.frombuffer(d_sb_raw, dtype="<u4").reshape(k_d // 64, 32)
    for g128 in range(k_d // 128):
        np.testing.assert_array_equal(
            d_sb_u32[2 * g128, :n_d], d_sb_u32[2 * g128 + 1, :n_d]
        )
    # Padding rows [n_d:32] must be strictly zero in both nax_sb and nax_codes.
    assert np.all(d_sb_u32[:, n_d:] == 0)
    d_codes_raw = _read_plane_raw(
        pretiled_art, pretiled_manifest, "model.layers.0.mlp.down_proj", "nax_codes"
    )
    d_codes_tiled = np.frombuffer(d_codes_raw, dtype=np.uint8).reshape(2, 8, 16, 64)
    assert np.all(d_codes_tiled[1, :, 8:, :] == 0)

    # 2) Full-matrix dequantization bit-exactness (untiled == pretiled == BF16 reference)
    w_q_ref = _dequant_affine_ref(q_codes, q_scales, q_biases, 64)
    w_q_untiled = _decode_full_matrix(
        untiled_art, untiled_manifest, "model.layers.0.self_attn.q_proj"
    )
    w_q_pretiled = _decode_full_matrix(
        pretiled_art, pretiled_manifest, "model.layers.0.self_attn.q_proj"
    )
    np.testing.assert_array_equal(w_q_untiled, w_q_ref)
    np.testing.assert_array_equal(w_q_pretiled, w_q_ref)

    w_d_ref = _dequant_affine_ref(d_codes, d_scales, d_biases, 128)
    w_d_untiled = _decode_full_matrix(
        untiled_art, untiled_manifest, "model.layers.0.mlp.down_proj"
    )
    w_d_pretiled = _decode_full_matrix(
        pretiled_art, pretiled_manifest, "model.layers.0.mlp.down_proj"
    )
    np.testing.assert_array_equal(w_d_untiled, w_d_ref)
    np.testing.assert_array_equal(w_d_pretiled, w_d_ref)

    # 3) Numerical matmul consistency across T=1 decode and T=16 prefill
    q_codes_raw = _read_plane_raw(
        pretiled_art, pretiled_manifest, "model.layers.0.self_attn.q_proj", "nax_codes"
    )
    q_sb_raw = _read_plane_raw(
        pretiled_art, pretiled_manifest, "model.layers.0.self_attn.q_proj", "nax_sb"
    )
    for t_batch in (1, 16):
        x_q = _bf16_round_f32(rng.normal(0.0, 0.5, size=(t_batch, k_q)).astype(np.float32))
        y_q_ref = x_q @ w_q_ref.T
        y_q_untiled = x_q @ w_q_untiled.T
        y_q_pretiled = x_q @ w_q_pretiled.T
        y_q_nax, y_q_pad = _simulate_affine8_nax_tiled_matmul(
            x_q, q_codes_raw, q_sb_raw, n_q, k_q
        )
        np.testing.assert_array_equal(y_q_pretiled, y_q_untiled)
        np.testing.assert_allclose(y_q_nax, y_q_ref, rtol=5e-5, atol=5e-4)
        assert np.all(y_q_pad == 0.0)

        x_d = _bf16_round_f32(rng.normal(0.0, 0.5, size=(t_batch, k_d)).astype(np.float32))
        y_d_ref = x_d @ w_d_ref.T
        y_d_untiled = x_d @ w_d_untiled.T
        y_d_pretiled = x_d @ w_d_pretiled.T
        y_d_nax, y_d_pad = _simulate_affine8_nax_tiled_matmul(
            x_d, d_codes_raw, d_sb_raw, n_d, k_d
        )
        np.testing.assert_array_equal(y_d_pretiled, y_d_untiled)
        np.testing.assert_allclose(y_d_nax, y_d_ref, rtol=5e-5, atol=5e-4)
        assert np.all(y_d_pad == 0.0)


@pytest.mark.parametrize("group_size", [64, 128])
def test_affine8_large_n_bn32_pretile_and_boundary_decode(
    tmp_path: Path, group_size: int
) -> None:
    rng = np.random.default_rng(202 + group_size)
    n, k = 65540, 512  # n >= 65536 triggers bn = 32 -> padded_n = 65568
    codes = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    scales = rng.uniform(0.02, 0.1, size=(n, k // group_size)).astype(np.float32)
    biases = rng.uniform(-1.5, 1.5, size=(n, k // group_size)).astype(np.float32)

    src = _write_checkpoint(
        tmp_path / f"src_bn32_g{group_size}",
        revision="rev-bn32-q8",
        config={
            "model_type": "qwen3_8",
            "quantization": {"bits": 8, "group_size": group_size, "mode": "affine"},
        },
        tensors={
            "lm_head.weight": ("U32", [n, k * 8 // 32], _pack_codes(codes, 8)),
            "lm_head.scales": ("BF16", [n, k // group_size], _bf16_bytes(scales)),
            "lm_head.biases": ("BF16", [n, k // group_size], _bf16_bytes(biases)),
        },
    )
    out_art = tmp_path / f"out_bn32_g{group_size}.omlxm5"
    args = m5_artifact.build_parser().parse_args(
        [
            "convert",
            "--source",
            str(src),
            "--output",
            str(out_art),
            "--source-repo",
            "test/bn32-q8",
            "--artifact-version",
            "1",
            "--nax-pretile",
        ]
    )
    m5_artifact.command_convert(args)

    _, manifest = m5_artifact.read_artifact(out_art, verify_payload=True)
    obj = manifest["objects"][0]
    assert obj["format"] == f"oq_affine_q8_g{group_size}_bf16_v1"
    assert obj["layout"] == "oq_nax_tiled_g64_v1"
    padded_n = 65568  # align_up(65540, 32)
    assert [(p["kind"], p["bytes"]) for p in obj["planes"]] == [
        ("nax_codes", padded_n * k),
        ("nax_sb", padded_n * (k // 64) * 4),
    ]

    # Spot-check rows straddling bn=16 and bn=32 tile boundaries and the final row.
    sc_bf16 = _bf16_round_f32(scales)
    bi_bf16 = _bf16_round_f32(biases)
    for row in (0, 15, 16, 31, 32, 65535, 65536, 65539):
        decoded = m5_artifact.decode_artifact_row(out_art, manifest, "lm_head", row, 0, k)
        expected = (
            np.repeat(sc_bf16[row], group_size) * codes[row].astype(np.float32)
            + np.repeat(bi_bf16[row], group_size)
        )
        np.testing.assert_array_equal(np.asarray(decoded, dtype=np.float32), expected)

    # Verify tail padding rows 65540..65567 in nax_codes and nax_sb are zero.
    raw_codes = _read_plane_raw(out_art, manifest, "lm_head", "nax_codes")
    last_tile = np.frombuffer(raw_codes, dtype=np.uint8).reshape(
        padded_n // 32, k // 64, 32, 64
    )[-1]
    assert np.all(last_tile[:, 4:, :] == 0)  # 65540 % 32 == 4


@pytest.mark.parametrize("default_mode", ["mxfp4", "affine"])
def test_mixed_mxfp4_and_affine8_per_layer_pretile_and_matmul_consistency(
    tmp_path: Path, default_mode: str
) -> None:
    rng = np.random.default_rng(303)
    k = 512
    n_mx = 16
    n_q8_g64 = 20
    n_q8_g128 = 18

    mx_codes = rng.integers(0, 16, size=(n_mx, k), dtype=np.uint8)
    mx_packed = (mx_codes[:, 0::2] & 0xF) | ((mx_codes[:, 1::2] & 0xF) << 4)
    mx_scales = rng.integers(124, 130, size=(n_mx, k // 32), dtype=np.uint8)

    o_codes = rng.integers(0, 256, size=(n_q8_g64, k), dtype=np.uint8)
    o_scales = rng.uniform(0.01, 0.15, size=(n_q8_g64, k // 64)).astype(np.float32)
    o_biases = rng.uniform(-2.0, 2.0, size=(n_q8_g64, k // 64)).astype(np.float32)

    down_codes = rng.integers(0, 256, size=(n_q8_g128, k), dtype=np.uint8)
    down_scales = rng.uniform(0.01, 0.15, size=(n_q8_g128, k // 128)).astype(np.float32)
    down_biases = rng.uniform(-2.0, 2.0, size=(n_q8_g128, k // 128)).astype(np.float32)

    router_codes = rng.integers(0, 256, size=(4, 128), dtype=np.uint8)
    router_scales = np.full((4, 2), 0.0625, dtype=np.float32)
    router_biases = np.full((4, 2), -1.0, dtype=np.float32)

    tensors: dict[str, tuple[str, list[int], bytes]] = {
        "model.layers.0.self_attn.q_proj.weight": (
            "U32",
            [n_mx, k // 8],
            np.ascontiguousarray(mx_packed).tobytes(),
        ),
        "model.layers.0.self_attn.q_proj.scales": (
            "U8",
            [n_mx, k // 32],
            mx_scales.tobytes(),
        ),
        "model.layers.0.self_attn.o_proj.weight": (
            "U32",
            [n_q8_g64, k * 8 // 32],
            _pack_codes(o_codes, 8),
        ),
        "model.layers.0.self_attn.o_proj.scales": (
            "BF16",
            [n_q8_g64, k // 64],
            _bf16_bytes(o_scales),
        ),
        "model.layers.0.self_attn.o_proj.biases": (
            "BF16",
            [n_q8_g64, k // 64],
            _bf16_bytes(o_biases),
        ),
        "model.layers.0.mlp.down_proj.weight": (
            "U32",
            [n_q8_g128, k * 8 // 32],
            _pack_codes(down_codes, 8),
        ),
        "model.layers.0.mlp.down_proj.scales": (
            "BF16",
            [n_q8_g128, k // 128],
            _bf16_bytes(down_scales),
        ),
        "model.layers.0.mlp.down_proj.biases": (
            "BF16",
            [n_q8_g128, k // 128],
            _bf16_bytes(down_biases),
        ),
        "model.layers.0.mlp.gate.weight": (
            "U32",
            [4, 128 * 8 // 32],
            _pack_codes(router_codes, 8),
        ),
        "model.layers.0.mlp.gate.scales": (
            "BF16",
            [4, 2],
            _bf16_bytes(router_scales),
        ),
        "model.layers.0.mlp.gate.biases": (
            "BF16",
            [4, 2],
            _bf16_bytes(router_biases),
        ),
    }

    if default_mode == "mxfp4":
        quant_cfg: dict = {
            "bits": 4,
            "group_size": 32,
            "mode": "mxfp4",
            "model.layers.0.self_attn.o_proj": {"bits": 8, "group_size": 64},
            "model.layers.0.mlp.down_proj": {
                "bits": 8,
                "group_size": 128,
                "mode": "affine",
            },
            "model.layers.0.mlp.gate": {"bits": 8, "group_size": 64},
        }
    else:
        quant_cfg = {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
            "model.layers.0.self_attn.q_proj": {
                "bits": 4,
                "group_size": 32,
                "mode": "mxfp4",
            },
            "model.layers.0.mlp.down_proj": {
                "bits": 8,
                "group_size": 128,
                "mode": "affine",
            },
        }

    src = _write_checkpoint(
        tmp_path / f"src_mixed_{default_mode}",
        revision=f"rev-mixed-{default_mode}",
        config={"model_type": "qwen3_6_moe", "quantization": quant_cfg},
        tensors=tensors,
    )
    untiled_art = tmp_path / f"mixed_{default_mode}_untiled.omlxm5"
    pretiled_art = tmp_path / f"mixed_{default_mode}_pretiled.omlxm5"

    for out_p, flag in ((untiled_art, []), (pretiled_art, ["--nax-pretile"])):
        args = m5_artifact.build_parser().parse_args(
            [
                "convert",
                "--source",
                str(src),
                "--output",
                str(out_p),
                "--source-repo",
                f"test/mixed-{default_mode}",
                "--artifact-version",
                "1",
                *flag,
            ]
        )
        m5_artifact.command_convert(args)

    _, u_manifest = m5_artifact.read_artifact(untiled_art, verify_payload=True)
    pt_summary, p_manifest = m5_artifact.read_artifact(pretiled_art, verify_payload=True)
    assert pt_summary["formats"] == {
        "mxfp4_e2m1_g32_e8m0_v1": 1,
        "oq_affine_q8_g128_bf16_v1": 1,
        "oq_affine_q8_g64_bf16_v1": 2,
    }

    objs = {obj["name"]: obj for obj in p_manifest["objects"]}
    assert objs["model.layers.0.self_attn.q_proj"]["layout"] == "mx_nax_tiled_bk128_v1"
    assert [
        p["kind"] for p in objs["model.layers.0.self_attn.q_proj"]["planes"]
    ] == ["nax_data", "nax_scale"]

    assert objs["model.layers.0.self_attn.o_proj"]["layout"] == "oq_nax_tiled_g64_v1"
    assert [(p["kind"], p["bytes"]) for p in objs["model.layers.0.self_attn.o_proj"]["planes"]] == [
        ("nax_codes", 32 * k),
        ("nax_sb", 32 * (k // 64) * 4),
    ]

    assert objs["model.layers.0.mlp.down_proj"]["layout"] == "oq_nax_tiled_g64_v1"
    assert [(p["kind"], p["bytes"]) for p in objs["model.layers.0.mlp.down_proj"]["planes"]] == [
        ("nax_codes", 32 * k),
        ("nax_sb", 32 * (k // 64) * 4),
    ]

    # Router with K=128 (< 512) remains untiled oq_rowsplit_k128_v1.
    assert objs["model.layers.0.mlp.gate"]["layout"] == "oq_rowsplit_k128_v1"
    assert [p["kind"] for p in objs["model.layers.0.mlp.gate"]["planes"]] == [
        "code",
        "scale",
        "bias",
    ]

    bindings = {b["name"]: b for b in p_manifest["bindings"]}
    assert bindings["model.layers.0.self_attn.q_proj"]["activation_policy"] == "AllowA4"
    assert bindings["model.layers.0.self_attn.o_proj"]["activation_policy"] == "A16Only"
    assert bindings["model.layers.0.mlp.down_proj"]["activation_policy"] == "A16Only"

    # Verify dequantization & matmul consistency across all mixed layers.
    w_mx_ref = _dequant_mxfp4_ref(mx_codes, mx_scales)
    w_o_ref = _dequant_affine_ref(o_codes, o_scales, o_biases, 64)
    w_down_ref = _dequant_affine_ref(down_codes, down_scales, down_biases, 128)

    for obj_name, w_ref in (
        ("model.layers.0.self_attn.q_proj", w_mx_ref),
        ("model.layers.0.self_attn.o_proj", w_o_ref),
        ("model.layers.0.mlp.down_proj", w_down_ref),
    ):
        w_u = _decode_full_matrix(untiled_art, u_manifest, obj_name)
        w_p = _decode_full_matrix(pretiled_art, p_manifest, obj_name)
        np.testing.assert_array_equal(w_u, w_ref)
        np.testing.assert_array_equal(w_p, w_ref)

    for t_batch in (1, 12):
        x = _bf16_round_f32(rng.normal(0.0, 0.5, size=(t_batch, k)).astype(np.float32))
        y_mx_p = x @ _decode_full_matrix(
            pretiled_art, p_manifest, "model.layers.0.self_attn.q_proj"
        ).T
        np.testing.assert_allclose(y_mx_p, x @ w_mx_ref.T, rtol=1e-6, atol=1e-6)

        for obj_name, n_rows, w_ref in (
            ("model.layers.0.self_attn.o_proj", n_q8_g64, w_o_ref),
            ("model.layers.0.mlp.down_proj", n_q8_g128, w_down_ref),
        ):
            y_nax, y_pad = _simulate_affine8_nax_tiled_matmul(
                x,
                _read_plane_raw(pretiled_art, p_manifest, obj_name, "nax_codes"),
                _read_plane_raw(pretiled_art, p_manifest, obj_name, "nax_sb"),
                n_rows,
                k,
            )
            np.testing.assert_allclose(y_nax, x @ w_ref.T, rtol=5e-5, atol=5e-4)
            assert np.all(y_pad == 0.0)


def test_affine8_nax_tiled_rejects_high_planes_and_invalid_geometry(
    tmp_path: Path,
) -> None:
    dummy_art = tmp_path / "dummy_q8.omlxm5"
    dummy_art.write_bytes(b"\x00" * (m5_artifact.HEADER_BYTES + 4096))

    # 1) Invalid geometries rejected by both _expected_plane_bytes and decode_artifact_row
    for fmt, shape, padded_k in (
        ("oq_affine_q8_g64_bf16_v1", [16, 256], 256),     # k < 512
        ("oq_affine_q8_g128_bf16_v1", [16, 576], 640),    # k % 128 != 0
        ("oq_affine_q8_g64_bf16_v1", [2, 16, 512], 512),  # rank != 2
    ):
        bad_rec = {
            "id": 0,
            "name": "bad_q8",
            "format": fmt,
            "layout": "oq_nax_tiled_g64_v1",
            "shape": shape,
            "padded_k": padded_k,
            "planes": [],
        }
        with pytest.raises(ValueError, match="affine NAX tiled geometry mismatch"):
            m5_artifact._expected_plane_bytes(bad_rec)
        with pytest.raises(ValueError, match="affine NAX tiled geometry mismatch"):
            m5_artifact.decode_artifact_row(
                dummy_art, {"objects": [bad_rec]}, "bad_q8", 0, 0, 16
            )

    # 2) Spurious nax_high2 / nax_high1 plane or 4-bit nax_codes size rejected when bits == 8
    base_valid = {
        "id": 0,
        "name": "q8_obj",
        "format": "oq_affine_q8_g64_bf16_v1",
        "layout": "oq_nax_tiled_g64_v1",
        "shape": [16, 512],
        "padded_k": 512,
        "object_offset": 0,
        "object_bytes": 16 * 512 + 16 * 8 * 4,
        "object_sha256": "0" * 64,
        "source_tensors": ["q8_obj.weight", "q8_obj.scales", "q8_obj.biases"],
        "planes": [
            {"kind": "nax_codes", "offset": 0, "bytes": 16 * 512},
            {"kind": "nax_sb", "offset": 16 * 512, "bytes": 16 * 8 * 4},
        ],
    }
    m5_artifact._validate_object_contract(base_valid, 0)

    for high_kind in ("nax_high1", "nax_high2"):
        with_high = {
            **base_valid,
            "planes": [
                {"kind": "nax_codes", "offset": 0, "bytes": 16 * 512},
                {"kind": high_kind, "offset": 16 * 512, "bytes": 16 * 128},
                {"kind": "nax_sb", "offset": 16 * 512 + 16 * 128, "bytes": 16 * 8 * 4},
            ],
        }
        with pytest.raises(ValueError, match="quantized layout contract mismatch"):
            m5_artifact._validate_object_contract(with_high, 0)

    wrong_code_size = {
        **base_valid,
        "planes": [
            {"kind": "nax_codes", "offset": 0, "bytes": 16 * 512 // 2},
            {"kind": "nax_sb", "offset": 16 * 512 // 2, "bytes": 16 * 8 * 4},
        ],
    }
    with pytest.raises(ValueError, match="quantized plane size mismatch"):
        m5_artifact._validate_object_contract(wrong_code_size, 0)


def test_miv_v2_cli_defaults_qda_bf16_prerotate_dflash2_and_mtp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1) Case 1: unquantized BF16 checkpoint -> qda-search v2 rotates in FP32 BEFORE quantization
    bf16_src = _make_bf16_checkpoint(tmp_path / "bf16_v2_src")
    v2_art = tmp_path / "bf16_v2.miv"
    args = m5_artifact.build_parser().parse_args(
        [
            "qda-search",
            "--source",
            str(bf16_src),
            "--target-bpw",
            "4.5",
            "--output-artifact",
            str(v2_art),
            "--source-repo",
            "test/bf16-v2",
            "--nax-pretile",
            "--jobs",
            "2",
        ]
    )
    assert args.artifact_version == 2
    assert args.fwht64 is None
    assert args.fwht64_o_proj is True
    m5_artifact.command_qda_search(args)
    _ = capsys.readouterr()

    summary, manifest = m5_artifact.read_artifact(v2_art, verify_payload=True)
    assert v2_art.read_bytes()[:8] == b"MIV00001"
    assert summary["container_magic"] == "MIV00001"
    assert summary["container_version"] == 2
    assert summary["container_flags"] == (m5_artifact.FLAG_FWHT64 | m5_artifact.FLAG_NAX_LANE_V2)
    assert summary["schema"] == "miv-artifact-v2"
    assert manifest["converter"]["name"] == "miv-artifact"
    assert manifest["format_features"]["fwht64_prerotated"] is True
    assert manifest["format_features"]["nax_lane_layout"] == "pi64_quartet_v2"
    objs = {o["name"]: o for o in manifest["objects"]}
    assert objs["model.layers.0.mlp.down_proj"]["input_transform"] == "fwht64"
    assert objs["model.layers.0.mlp.down_proj"]["layout"] in (
        "oq_nax_lane_g64_v2",
        "oq_nax_lane_g128_v2",
    )
    assert objs["model.layers.0.self_attn.q_proj"].get("input_transform", "none") == "none"

    # 2) dflash2_bf16 profile auto-disables fwht64 even under v2 default
    df_src = _write_checkpoint(
        tmp_path / "dflash2_src",
        revision="rev-df2",
        config={"architectures": ["DFlash2DraftModel"], "model_type": "qwen3"},
        tensors={
            "layers.0.mlp.down_proj.weight": (
                "BF16",
                [16, 64],
                _bf16_bytes(np.ones((16, 64), dtype=np.float32)),
            ),
        },
    )
    df_out = tmp_path / "dflash2_v2.miv"
    m5_artifact.command_convert(
        m5_artifact.build_parser().parse_args(
            [
                "convert",
                "--source",
                str(df_src),
                "--output",
                str(df_out),
                "--source-repo",
                "test/dflash2",
                "--profile",
                "dflash2_bf16",
            ]
        )
    )
    _ = capsys.readouterr()
    df_sum, df_man = m5_artifact.read_artifact(df_out, verify_payload=True)
    assert df_sum["container_version"] == 2
    assert df_sum["container_flags"] == 0
    assert df_man["format_features"]["fwht64_prerotated"] is False
    assert all(o.get("input_transform", "none") == "none" for o in df_man["objects"])

    # 3) extract-mtp & bundle-mtp preserve v2 format/flags/input_transform
    mtp_ckpt = _make_oq_checkpoint(tmp_path / "oq_mtp_v2_src", include_mtp=True)
    full_v2 = tmp_path / "full_v2.miv"
    mtp_v2 = tmp_path / "mtp_v2.miv"
    m5_artifact.command_convert(
        m5_artifact.build_parser().parse_args(
            [
                "convert",
                "--source",
                str(mtp_ckpt),
                "--output",
                str(full_v2),
                "--source-repo",
                "test/oq-mtp-v2",
                "--nax-pretile",
                "--jobs",
                "2",
            ]
        )
    )
    _ = capsys.readouterr()
    m5_artifact.command_extract_mtp(
        m5_artifact.build_parser().parse_args(
            [
                "extract-mtp",
                "--source",
                str(full_v2),
                "--output",
                str(mtp_v2),
            ]
        )
    )
    _ = capsys.readouterr()
    mtp_sum, mtp_man = m5_artifact.read_artifact(mtp_v2, verify_payload=True)
    assert mtp_sum["container_magic"] == "MIV00001"
    assert mtp_sum["schema"] == "miv-artifact-v2"
    assert mtp_man["converter"]["name"] == "miv-artifact"
    assert mtp_sum["container_version"] == 2
    assert (mtp_sum["container_flags"] & m5_artifact.FLAG_NAX_LANE_V2) != 0
    assert mtp_man["objects"][0]["layout"] in ("oq_nax_lane_g64_v2", "oq_nax_lane_g128_v2")


def test_dual_magic_and_schema_branding_interop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import struct
    import zlib

    src = _make_oq_checkpoint(tmp_path / "brand_src", include_mtp=True)

    # 1) Default writes emit MIV00001 + miv-artifact-v{1,2} + converter.name == "miv-artifact"
    #    --legacy-magic emits OMLXM5A1 + omlx-m5-artifact-v{1,2} + converter.name == "omlx-m5-artifact"
    cases = [
        ("new_v2.miv", [], b"MIV00001", 2, "miv-artifact-v2", "miv-artifact"),
        ("new_v1.miv", ["--artifact-version", "1"], b"MIV00001", 1, "miv-artifact-v1", "miv-artifact"),
        ("legacy_v2.miv", ["--legacy-magic"], b"OMLXM5A1", 2, "omlx-m5-artifact-v2", "omlx-m5-artifact"),
        (
            "legacy_v1.miv",
            ["--artifact-version", "1", "--legacy-magic"],
            b"OMLXM5A1",
            1,
            "omlx-m5-artifact-v1",
            "omlx-m5-artifact",
        ),
    ]
    paths: dict[str, Path] = {}
    for filename, extra_args, exp_magic, exp_ver, exp_schema, exp_conv in cases:
        out = tmp_path / filename
        paths[filename] = out
        m5_artifact.command_convert(
            m5_artifact.build_parser().parse_args(
                [
                    "convert",
                    "--source",
                    str(src),
                    "--output",
                    str(out),
                    "--source-repo",
                    "test/brand-interop",
                    "--jobs",
                    "2",
                    *extra_args,
                ]
            )
        )
        _ = capsys.readouterr()
        assert out.read_bytes()[:8] == exp_magic
        summary, manifest = m5_artifact.read_artifact(out, verify_payload=True)
        assert summary["container_magic"] == exp_magic.decode("ascii")
        assert summary["container_version"] == exp_ver
        assert summary["schema"] == exp_schema
        assert manifest["converter"]["name"] == exp_conv

    # 2) pretile reads legacy OMLXM5A1 v1 and emits new MIV00001 v2 by default (or legacy with --legacy-magic)
    upgraded = tmp_path / "legacy_to_new_v2.miv"
    m5_artifact.command_pretile(
        m5_artifact.build_parser().parse_args(
            [
                "pretile",
                "--input",
                str(paths["legacy_v1.miv"]),
                "--output",
                str(upgraded),
                "--jobs",
                "2",
            ]
        )
    )
    _ = capsys.readouterr()
    assert upgraded.read_bytes()[:8] == b"MIV00001"
    up_sum, up_man = m5_artifact.read_artifact(upgraded, verify_payload=True)
    assert up_sum["container_magic"] == "MIV00001"
    assert up_sum["container_version"] == 2
    assert up_sum["schema"] == "miv-artifact-v2"
    assert up_man["converter"]["name"] == "miv-artifact"

    kept_legacy = tmp_path / "legacy_to_legacy_v1.miv"
    m5_artifact.command_pretile(
        m5_artifact.build_parser().parse_args(
            [
                "pretile",
                "--input",
                str(paths["legacy_v1.miv"]),
                "--output",
                str(kept_legacy),
                "--artifact-version",
                "1",
                "--legacy-magic",
                "--jobs",
                "2",
            ]
        )
    )
    _ = capsys.readouterr()
    assert kept_legacy.read_bytes()[:8] == b"OMLXM5A1"
    kl_sum, kl_man = m5_artifact.read_artifact(kept_legacy, verify_payload=True)
    assert kl_sum["container_magic"] == "OMLXM5A1"
    assert kl_sum["container_version"] == 1
    assert kl_sum["schema"] == "omlx-m5-artifact-v1"
    assert kl_man["converter"]["name"] == "omlx-m5-artifact"

    # 3) extract-mtp and bundle-mtp support both reading legacy and emitting new/legacy
    mtp_legacy = tmp_path / "mtp_legacy.miv"
    m5_artifact.command_extract_mtp(
        m5_artifact.build_parser().parse_args(
            [
                "extract-mtp",
                "--source",
                str(paths["legacy_v1.miv"]),
                "--output",
                str(mtp_legacy),
                "--legacy-magic",
            ]
        )
    )
    _ = capsys.readouterr()
    assert mtp_legacy.read_bytes()[:8] == b"OMLXM5A1"
    ml_sum, ml_man = m5_artifact.read_artifact(mtp_legacy, verify_payload=True)
    assert ml_sum["schema"] == "omlx-m5-artifact-v1"
    assert ml_man["converter"]["name"] == "omlx-m5-artifact"

    bundled_new = tmp_path / "bundled_new.miv"
    m5_artifact.command_bundle_mtp(
        m5_artifact.build_parser().parse_args(
            [
                "bundle-mtp",
                "--base",
                str(paths["legacy_v1.miv"]),
                "--mtp",
                str(mtp_legacy),
                "--output",
                str(bundled_new),
            ]
        )
    )
    _ = capsys.readouterr()
    assert bundled_new.read_bytes()[:8] == b"MIV00001"
    bn_sum, bn_man = m5_artifact.read_artifact(bundled_new, verify_payload=True)
    assert bn_sum["schema"] == "miv-artifact-v1"
    assert bn_man["converter"]["name"] == "miv-artifact"

    # 4) Reject invalid magics (including rejected alternate magic) and version/schema mismatches
    for bad_magic in (b"MIVART" + b"01", b"BADMAGIC"):
        bad_path = tmp_path / f"bad_magic_{bad_magic.decode('ascii')}.miv"
        raw = bytearray(paths["new_v2.miv"].read_bytes())
        raw[:8] = bad_magic
        struct.pack_into("<I", raw, 56, zlib.crc32(raw[:56]) & 0xFFFFFFFF)
        bad_path.write_bytes(raw)
        with pytest.raises(ValueError, match="unsupported artifact header"):
            m5_artifact.read_artifact(bad_path, verify_payload=False)

    def _rewrite_schema(src_art: Path, dst_art: Path, new_schema: str) -> None:
        raw = src_art.read_bytes()
        hdr = m5_artifact.HEADER_STRUCT.unpack(raw[: m5_artifact.HEADER_BYTES])
        magic, ver, _, m_off, m_len, old_pay_off, pay_len, _, _, _, flags = hdr
        man = json.loads(raw[m_off : m_off + m_len])
        man["schema"] = new_schema
        enc = m5_artifact.canonical_json(man)
        pay_off = m5_artifact.align_up(m5_artifact.HEADER_BYTES + len(enc), m5_artifact.PAYLOAD_ALIGNMENT)
        m_crc = m5_artifact.crc32(enc)
        prefix = m5_artifact.HEADER_STRUCT.pack(
            magic, ver, m5_artifact.HEADER_BYTES, m5_artifact.HEADER_BYTES,
            len(enc), pay_off, pay_len, len(man["objects"]), m_crc, 0, flags
        )
        h_crc = m5_artifact.crc32(prefix[:56])
        full_hdr = m5_artifact.HEADER_STRUCT.pack(
            magic, ver, m5_artifact.HEADER_BYTES, m5_artifact.HEADER_BYTES,
            len(enc), pay_off, pay_len, len(man["objects"]), m_crc, h_crc, flags
        )
        dst_art.write_bytes(
            full_hdr + enc + bytes(pay_off - m5_artifact.HEADER_BYTES - len(enc)) + raw[old_pay_off : old_pay_off + pay_len]
        )

    for src_key, bad_schema in (
        ("new_v1.miv", "miv-artifact-v2"),
        ("legacy_v1.miv", "omlx-m5-artifact-v2"),
        ("new_v2.miv", "miv-artifact-v1"),
        ("legacy_v2.miv", "omlx-m5-artifact-v1"),
        ("new_v2.miv", "bogus-artifact-v2"),
    ):
        dst = tmp_path / f"mismatch_{src_key}_{bad_schema}.miv"
        _rewrite_schema(paths[src_key], dst, bad_schema)
        with pytest.raises(ValueError, match="artifact manifest contract mismatch"):
            m5_artifact.read_artifact(dst, verify_payload=False)
