"""Regression tests for Phase 2 P1 fixes: EXT-1 through EXT-7."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import zlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
OMLX_ROOT = REPO_ROOT / "omlx"
if str(OMLX_ROOT) not in sys.path:
    sys.path.insert(0, str(OMLX_ROOT))

from fastapi import HTTPException  # noqa: E402
from omlx.admin import routes as admin_routes  # noqa: E402
from omlx.engine import m5_sidecar as m5_mod  # noqa: E402
from omlx.engine.m5_sidecar import (  # noqa: E402
    M5RT_ENGINE_ID,
    M5SidecarEngine,
    M5Target,
    find_m5rt_binary,
)
from omlx.model_settings import ModelSettings  # noqa: E402

SYNC_MK_SCRIPT = REPO_ROOT / "omlx-extend" / "scripts" / "sync_microkernels.sh"
START_SERVE_SCRIPT = REPO_ROOT / "omlx-extend" / "scripts" / "start_ninfer_serve.sh"
SYNC_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "sync-omlx-extend.yml"


def _write_omlxm5_container(path: Path, object_names: list[str]) -> Path:
    """Create a minimal valid OMLXM5A1 v1 container with the given manifest object names."""
    manifest = {
        "schema": "omlx-m5-artifact-v1",
        "container_version": 1,
        "payload_bytes": 0,
        "model": {},
        "objects": [{"name": name} for name in object_names],
        "views": [],
        "bindings": [],
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    manifest_crc = zlib.crc32(manifest_bytes) & 0xFFFFFFFF
    payload_offset = ((64 + len(manifest_bytes) + 4095) // 4096) * 4096
    prefix = struct.pack(
        "<8sIIQQQQII",
        b"OMLXM5A1",
        1,
        64,
        64,
        len(manifest_bytes),
        payload_offset,
        0,
        len(object_names),
        manifest_crc,
    )
    header_crc = zlib.crc32(prefix) & 0xFFFFFFFF
    header = prefix + struct.pack("<II", header_crc, 0)
    assert len(header) == 64
    padding = b"\x00" * (payload_offset - 64 - len(manifest_bytes))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + manifest_bytes + padding)
    return path


def _make_fake_daemon(path: Path, version: str = "miv-v0.01", protocol: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "--version" ]; then echo "mseries-backendd {version} protocol={protocol}"; exit 0; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _make_model_dir(
    model_dir: Path,
    *,
    model_type: str = "qwen3_8",
    mtp_layers: int = 0,
    artifact_objects: list[str] | None = None,
    placeholder_artifact: bool = False,
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    text_cfg: dict[str, object] = {
        "model_type": f"{model_type}_text",
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "vocab_size": 512,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "max_position_embeddings": 262144,
        "layer_types": ["linear_attention", "full_attention"],
    }
    if mtp_layers > 0:
        text_cfg["mtp_num_hidden_layers"] = mtp_layers
    cfg = {"model_type": model_type, "text_config": text_cfg}
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    if artifact_objects is not None:
        _write_omlxm5_container(model_dir / "model.omlxm5", artifact_objects)
    elif placeholder_artifact:
        (model_dir / "model.omlxm5").write_bytes(b"OMLXM5" + b"\x00" * 512)
    return model_dir


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_provider_tarball(
    tar_path: Path,
    *,
    provider_id: str = "io.example.testmk",
    engine_id: str = "testmk",
    version: str = "0.1.0",
    extra_entries: list[tuple[tarfile.TarInfo, bytes | None]] | None = None,
) -> Path:
    manifest = {
        "provider_id": provider_id,
        "engine_id": engine_id,
        "display_name": "Test MK",
        "display_version": version,
        "protocol_version": 2,
        "entrypoint": "bin/test_daemon",
    }
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    daemon_bytes = b"#!/bin/sh\necho 'test_daemon 0.1.0 protocol=2'\n"
    with tarfile.open(tar_path, "w:gz") as tf:
        for name, data, mode in (
            ("bundle/provider.json", manifest_bytes, 0o644),
            ("bundle/bin/test_daemon", daemon_bytes, 0o755),
        ):
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            ti.mode = mode
            tf.addfile(ti, io.BytesIO(data))
        if extra_entries:
            for ti, payload in extra_entries:
                if payload is not None:
                    ti.size = len(payload)
                    tf.addfile(ti, io.BytesIO(payload))
                else:
                    tf.addfile(ti)
    return tar_path


def _write_registry_and_latest(
    tmp_path: Path,
    *,
    tar_path: Path,
    provider_id: str = "io.example.testmk",
    engine_id: str = "testmk",
    version: str = "0.1.0",
    filename: str | None = None,
) -> tuple[Path, Path]:
    latest_path = tmp_path / "latest.json"
    asset: dict[str, object] = {
        "url": f"file://{tar_path}",
        "sha256": _sha256(tar_path),
    }
    if filename is not None:
        asset["filename"] = filename
    latest_path.write_text(
        json.dumps(
            {
                "provider_id": provider_id,
                "protocol_version": 2,
                "display_version": version,
                "assets": {"macos-arm64": asset},
            }
        ),
        encoding="utf-8",
    )
    registry_path = tmp_path / "microkernels.json"
    registry_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "provider_id": provider_id,
                        "engine_id": engine_id,
                        "latest_manifest_url": f"file://{latest_path}",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return registry_path, latest_path


# ---------------------------------------------------------------------------
# EXT-1: sync_microkernels.sh path traversal & tar extraction safety
# ---------------------------------------------------------------------------


class TestExt1SyncMicrokernels:
    def test_rejects_traversal_filename_before_download(self, tmp_path: Path):
        tar_path = _build_provider_tarball(tmp_path / "good.tar.gz")
        escape_marker = tmp_path / "escaped.tar.gz"
        registry_path, _ = _write_registry_and_latest(
            tmp_path,
            tar_path=tar_path,
            filename=f"../../../../../..{escape_marker}",
        )
        providers_dir = tmp_path / "Providers"
        res = subprocess.run(
            [
                "bash",
                str(SYNC_MK_SCRIPT),
                "--registry",
                str(registry_path),
                "--providers-dir",
                str(providers_dir),
                "--provider",
                "testmk",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode != 0
        assert "Refusing unsafe filename" in res.stderr
        assert not escape_marker.exists()

    def test_rejects_traversal_version(self, tmp_path: Path):
        tar_path = _build_provider_tarball(tmp_path / "good.tar.gz")
        registry_path, _ = _write_registry_and_latest(
            tmp_path,
            tar_path=tar_path,
            version="../../outside",
            filename="good.tar.gz",
        )
        providers_dir = tmp_path / "Providers"
        res = subprocess.run(
            [
                "bash",
                str(SYNC_MK_SCRIPT),
                "--registry",
                str(registry_path),
                "--providers-dir",
                str(providers_dir),
                "--provider",
                "testmk",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode != 0
        assert "Refusing unsafe version" in res.stderr
        assert not (tmp_path / "outside").exists()

    def test_rejects_symlink_escape_even_in_python_pre_311_fallback(self, tmp_path: Path):
        # Create a tarball with an escaping symlink.
        sym = tarfile.TarInfo(name="bundle/evil_link")
        sym.type = tarfile.SYMTYPE
        sym.linkname = "../../../../tmp/pwned"
        tar_path = _build_provider_tarball(
            tmp_path / "evil_sym.tar.gz", extra_entries=[(sym, None)]
        )
        registry_path, _ = _write_registry_and_latest(
            tmp_path, tar_path=tar_path, filename="evil_sym.tar.gz"
        )

        # Shim python3 so tarfile.TarFile.extractall raises TypeError on filter=... (Python 3.9 behavior).
        shim_dir = tmp_path / "shim_bin"
        shim_dir.mkdir()
        sitecust = tmp_path / "shim_site"
        sitecust.mkdir()
        (sitecust / "sitecustomize.py").write_text(
            "import tarfile\n"
            "_orig = tarfile.TarFile.extractall\n"
            "def _no_filter(self, path='.', members=None, *, numeric_owner=False, **kw):\n"
            "    if 'filter' in kw:\n"
            "        raise TypeError(\"extractall() got an unexpected keyword argument 'filter'\")\n"
            "    return _orig(self, path=path, members=members, numeric_owner=numeric_owner)\n"
            "tarfile.TarFile.extractall = _no_filter\n",
            encoding="utf-8",
        )
        py_shim = shim_dir / "python3"
        py_shim.write_text(
            f"#!/bin/sh\nPYTHONPATH={sitecust}${{PYTHONPATH:+:$PYTHONPATH}} exec {sys.executable} \"$@\"\n",
            encoding="utf-8",
        )
        py_shim.chmod(0o755)

        env = dict(os.environ)
        env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"
        providers_dir = tmp_path / "Providers"

        res = subprocess.run(
            [
                "bash",
                str(SYNC_MK_SCRIPT),
                "--registry",
                str(registry_path),
                "--providers-dir",
                str(providers_dir),
                "--provider",
                "testmk",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res.returncode != 0
        assert "escapes staging dir" in res.stderr

        # Also test symlink-chain attack (sub/up -> .., then sub/up/escape -> ..) in fallback mode!
        sub_dir = tarfile.TarInfo(name="bundle/sub")
        sub_dir.type = tarfile.DIRTYPE
        sub_dir.mode = 0o755
        up_link = tarfile.TarInfo(name="bundle/sub/up")
        up_link.type = tarfile.SYMTYPE
        up_link.linkname = ".."
        chain_link = tarfile.TarInfo(name="bundle/sub/up/escape")
        chain_link.type = tarfile.SYMTYPE
        chain_link.linkname = "../.."
        tar_chain = _build_provider_tarball(
            tmp_path / "evil_chain.tar.gz",
            extra_entries=[(sub_dir, None), (up_link, None), (chain_link, None)],
        )
        reg_chain, _ = _write_registry_and_latest(
            tmp_path, tar_path=tar_chain, filename="evil_chain.tar.gz"
        )
        res_chain = subprocess.run(
            [
                "bash",
                str(SYNC_MK_SCRIPT),
                "--registry",
                str(reg_chain),
                "--providers-dir",
                str(providers_dir),
                "--provider",
                "testmk",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res_chain.returncode != 0

        # Valid bundle with safe intra-bundle relative symlink succeeds even in fallback mode.
        safe_link = tarfile.TarInfo(name="bundle/bin/test_daemon_alias")
        safe_link.type = tarfile.SYMTYPE
        safe_link.linkname = "test_daemon"
        tar_good = _build_provider_tarball(
            tmp_path / "good_safe.tar.gz",
            version="0.1.0-rc1+m5",
            extra_entries=[(safe_link, None)],
        )
        reg_good, _ = _write_registry_and_latest(
            tmp_path,
            tar_path=tar_good,
            version="0.1.0-rc1+m5",
            filename="good_safe.tar.gz",
        )
        res_good = subprocess.run(
            [
                "bash",
                str(SYNC_MK_SCRIPT),
                "--registry",
                str(reg_good),
                "--providers-dir",
                str(providers_dir),
                "--provider",
                "testmk",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res_good.returncode == 0, res_good.stderr
        assert (providers_dir / "io.example.testmk" / "current" / "provider.json").is_file()


# ---------------------------------------------------------------------------
# EXT-2: start_ninfer_serve.sh two-word passthrough flag parsing
# ---------------------------------------------------------------------------


class TestExt2StartNinferServeArgs:
    def test_two_word_passthrough_flags_do_not_consume_model_dir(self, tmp_path: Path):
        fake_bin = tmp_path / "ninfer-metal-serve"
        fake_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_bin.chmod(0o755)

        model_dir = _make_model_dir(tmp_path / "models" / "Qwen3.8-Test", placeholder_artifact=True)
        cfg_file = tmp_path / "ninfer-metal.toml"
        cfg_file.write_text("[server]\nport = 8080\n", encoding="utf-8")

        env = dict(os.environ)
        env["HOME"] = str(tmp_path / "home")
        env["NINFER_METAL_SERVE_BIN"] = str(fake_bin)
        env["NINFER_METAL_CONFIG"] = str(cfg_file)
        (tmp_path / "home").mkdir()

        # Case 1: two-word options BEFORE positional MODEL_DIR
        res = subprocess.run(
            [
                "bash",
                str(START_SERVE_SCRIPT),
                "--dry-run",
                "--kv-dtype",
                "bf16",
                "--max-concurrency",
                "4",
                "-c",
                "2",
                "--max-context",
                "16384",
                "--spec-backend",
                "mtp",
                "--draft-tokens",
                "3",
                "--cors",
                "off",
                str(model_dir),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert f"Model:     {model_dir}" in res.stdout
        assert "--model-dir bf16" not in res.stdout
        assert "--kv-dtype bf16" in res.stdout
        assert "--max-concurrency 4" in res.stdout
        assert "-c 2" in res.stdout
        assert "--cors off" in res.stdout
        assert f"--model-dir {model_dir}" in res.stdout

        # Case 2: two-word options with NO positional MODEL_DIR (auto-discovers ./models/...)
        res_auto = subprocess.run(
            [
                "bash",
                str(START_SERVE_SCRIPT),
                "--dry-run",
                "--kv-dtype",
                "bf16",
                "--prefix-cache-gb",
                "2.5",
                "--api-key",
                "secret-token",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "--model-dir bf16" not in res_auto.stdout
        assert "Qwen3.8-Test (auto-discovered)" in res_auto.stdout
        assert "--kv-dtype bf16" in res_auto.stdout
        assert "--prefix-cache-gb 2.5" in res_auto.stdout
        assert "--api-key secret-token" in res_auto.stdout

        # Case 3: --cors without value followed by positional MODEL_DIR
        res_cors = subprocess.run(
            [
                "bash",
                str(START_SERVE_SCRIPT),
                "--dry-run",
                "--cors",
                str(model_dir),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert f"Model:     {model_dir}" in res_cors.stdout
        assert f"--model-dir {model_dir}" in res_cors.stdout

        # Case 4: `--` passthrough delimiter
        res_dash = subprocess.run(
            [
                "bash",
                str(START_SERVE_SCRIPT),
                "--dry-run",
                str(model_dir),
                "--",
                "--kv-dtype",
                "bf16",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "--kv-dtype bf16" in res_dash.stdout
        assert " -- " not in res_dash.stdout


# ---------------------------------------------------------------------------
# EXT-3: batch_mode parsing & max_concurrency linkage in m5_sidecar.py
# ---------------------------------------------------------------------------


class TestExt3BatchModeContract:
    def test_batch_mode_parsing_validation_and_load_params(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("NINFER_METAL_NO_DEFAULT_CONFIG", "1")
        assert m5_mod.validate_m5rt_options({"batch_mode": "off"}) == {"batch_mode": "off"}
        assert m5_mod.validate_m5rt_options({"batch_mode": False}) == {"batch_mode": False}
        assert m5_mod.validate_m5rt_options({"batch_mode": "on"}) == {"batch_mode": "on"}
        with pytest.raises(ValueError, match="batch_mode"):
            m5_mod.validate_m5rt_options({"batch_mode": "maybe"})

        model_dir = _make_model_dir(tmp_path / "model", placeholder_artifact=True)
        fake_bin = _make_fake_daemon(tmp_path / "bin" / "mseries_backendd")
        binary, err = find_m5rt_binary(str(fake_bin))
        assert binary is not None and err == ""
        target = M5Target(
            binary=binary,
            artifact_path=model_dir / "model.omlxm5",
            metallib_dir=tmp_path / "bin",
            fake=False,
        )

        # 1) batch_mode="off" parses to False (NOT True!) and never sends "on"
        ms_off = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            m5rt_options={"max_concurrency": 1, "batch_mode": "off"},
        )
        opts_off = m5_mod.build_m5rt_options(
            model_dir, target, ms_off, default_context=4096, memory_budget_bytes=0
        )
        assert opts_off.batch_mode is False
        eng_off = M5SidecarEngine(
            model_name=str(model_dir),
            model_id="bm-off",
            target=target,
            log_dir=tmp_path / "logs",
            options=opts_off,
        )
        payload_off = eng_off._build_load_params()
        assert payload_off["path_overrides"]["batch_mode"] == "off"

        # 2) batch_mode unset + max_concurrency=4 -> auto-enables batch_mode="on" (matches C++ config.cpp)
        ms_auto4 = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            m5rt_options={"max_concurrency": 4},
        )
        opts_auto4 = m5_mod.build_m5rt_options(
            model_dir, target, ms_auto4, default_context=4096, memory_budget_bytes=0
        )
        assert opts_auto4.batch_mode is None
        eng_auto4 = M5SidecarEngine(
            model_name=str(model_dir),
            model_id="bm-auto4",
            target=target,
            log_dir=tmp_path / "logs",
            options=opts_auto4,
        )
        assert eng_auto4._build_load_params()["path_overrides"]["batch_mode"] == "on"

        # 3) batch_mode="off" + max_concurrency=4 -> explicitly disabled ("off")
        ms_off4 = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            m5rt_options={"max_concurrency": 4, "batch_mode": "off"},
        )
        opts_off4 = m5_mod.build_m5rt_options(
            model_dir, target, ms_off4, default_context=4096, memory_budget_bytes=0
        )
        eng_off4 = M5SidecarEngine(
            model_name=str(model_dir),
            model_id="bm-off4",
            target=target,
            log_dir=tmp_path / "logs",
            options=opts_off4,
        )
        assert eng_off4._build_load_params()["path_overrides"]["batch_mode"] == "off"

        # 4) batch_mode="on" + max_concurrency=1 -> guard suppresses "on" (avoids B>=2 throw)
        ms_on1 = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            m5rt_options={"max_concurrency": 1, "batch_mode": "on"},
        )
        opts_on1 = m5_mod.build_m5rt_options(
            model_dir, target, ms_on1, default_context=4096, memory_budget_bytes=0
        )
        assert opts_on1.batch_mode is True
        eng_on1 = M5SidecarEngine(
            model_name=str(model_dir),
            model_id="bm-on1",
            target=target,
            log_dir=tmp_path / "logs",
            options=opts_on1,
        )
        assert eng_on1._build_load_params()["path_overrides"].get("batch_mode") != "on"

        # 5) Runtime signature reflects batch_mode changes
        assert m5_mod.m5rt_runtime_signature(ms_off4, None) != m5_mod.m5rt_runtime_signature(
            ms_auto4, None
        )


# ---------------------------------------------------------------------------
# EXT-4: UI TurboQuant KV toggle semantics with shared ninfer-metal.toml
# ---------------------------------------------------------------------------


class TestExt4TurboQuantKvToggle:
    def test_ui_toggle_on_default_scheme_and_explicit_ui_off(self, tmp_path: Path, monkeypatch):
        shared_toml = tmp_path / "ninfer-metal.toml"
        shared_toml.write_text(
            '[engine]\nmax_concurrency = 4\nkv_dtype = "affine4_g128"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("NINFER_METAL_CONFIG", str(shared_toml))
        monkeypatch.delenv("NINFER_METAL_NO_DEFAULT_CONFIG", raising=False)

        model_dir = _make_model_dir(tmp_path / "kv-model", placeholder_artifact=True)
        fake_bin = _make_fake_daemon(tmp_path / "bin" / "mseries_backendd")
        binary, _ = find_m5rt_binary(str(fake_bin))
        assert binary is not None
        target = M5Target(
            binary=binary,
            artifact_path=model_dir / "model.omlxm5",
            metallib_dir=tmp_path / "bin",
            fake=False,
        )

        # 1) Untouched ModelSettings() inherits shared TOML kv_dtype="affine4_g128"
        opts_default = m5_mod.build_m5rt_options(
            model_dir, target, ModelSettings(), default_context=4096, memory_budget_bytes=0
        )
        assert opts_default.kv_dtype == "affine4_g128"

        # 2) Turning UI KV Compression ON with default scheme="turboquant", bits=4 -> "affine4_g128"
        ms_on_default_scheme = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            turboquant_kv_enabled=True,
            # Default: turboquant_kv_scheme="turboquant", turboquant_kv_bits=4
        )
        opts_on = m5_mod.build_m5rt_options(
            model_dir, target, ms_on_default_scheme, default_context=4096, memory_budget_bytes=0
        )
        assert opts_on.kv_dtype == "affine4_g128"

        # 3) Explicit 8-bit scheme still never maps to int8_g64 (stays bf16)
        ms_8bit = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            turboquant_kv_enabled=True,
            turboquant_kv_scheme="turboquant",
            turboquant_kv_bits=8,
        )
        opts_8bit = m5_mod.build_m5rt_options(
            model_dir, target, ms_8bit, default_context=4096, memory_budget_bytes=0
        )
        assert opts_8bit.kv_dtype == "bf16"
        assert m5_mod.m5rt_runtime_signature(ms_on_default_scheme, None) != (
            m5_mod.m5rt_runtime_signature(ms_8bit, None)
        )

        # 4) Explicit UI disable (PUT turboquant_kv_enabled=False) overrides TOML -> "bf16",
        #    and re-enabling (PUT turboquant_kv_enabled=True) restores -> "affine4_g128".
        ms_live = ModelSettings(external_engine=M5RT_ENGINE_ID)
        req_off = admin_routes.ModelSettingsRequest(turboquant_kv_enabled=False)
        ms_live.turboquant_kv_enabled = False
        m5_mod.apply_m5rt_model_settings_update(
            ms_live, req_off, req_off.model_fields_set, is_diffusion_model=False
        )
        opts_after_off = m5_mod.build_m5rt_options(
            model_dir, target, ms_live, default_context=4096, memory_budget_bytes=0
        )
        assert opts_after_off.kv_dtype == "bf16"

        req_on = admin_routes.ModelSettingsRequest(turboquant_kv_enabled=True)
        ms_live.turboquant_kv_enabled = True
        m5_mod.apply_m5rt_model_settings_update(
            ms_live, req_on, req_on.model_fields_set, is_diffusion_model=False
        )
        opts_after_on = m5_mod.build_m5rt_options(
            model_dir, target, ms_live, default_context=4096, memory_budget_bytes=0
        )
        assert opts_after_on.kv_dtype == "affine4_g128"

        # 5) Stock MLX model is never polluted with m5rt_options on UI disable
        ms_stock = ModelSettings(external_engine="stock")
        m5_mod.apply_m5rt_model_settings_update(
            ms_stock, req_off, req_off.model_fields_set, is_diffusion_model=False
        )
        assert ms_stock.m5rt_options is None


# ---------------------------------------------------------------------------
# EXT-5: speculative.backend="auto" detects embedded MTP & explicit mtp_artifact
# ---------------------------------------------------------------------------


class TestExt5SpeculativeAutoMtp:
    def test_auto_detects_embedded_mtp_and_explicit_mtp_artifact(self, tmp_path: Path, monkeypatch):
        shared_toml = tmp_path / "ninfer-metal.toml"
        shared_toml.write_text(
            '[speculative]\nbackend = "auto"\ndraft_tokens = 0\n', encoding="utf-8"
        )
        monkeypatch.setenv("NINFER_METAL_CONFIG", str(shared_toml))
        monkeypatch.delenv("NINFER_METAL_NO_DEFAULT_CONFIG", raising=False)

        fake_bin = _make_fake_daemon(tmp_path / "bin" / "mseries_backendd")
        binary, _ = find_m5rt_binary(str(fake_bin))
        assert binary is not None

        # 1) Embedded MTP weights inside single-file OMLXM5A1 model.omlxm5
        emb_dir = _make_model_dir(
            tmp_path / "embedded-mtp",
            artifact_objects=["model.layers.0.q_proj.weight", "language_model.mtp.norm.weight"],
        )
        emb_target = M5Target(
            binary=binary,
            artifact_path=emb_dir / "model.omlxm5",
            metallib_dir=tmp_path / "bin",
            fake=False,
        )
        opts_emb = m5_mod.build_m5rt_options(
            emb_dir, emb_target, ModelSettings(), default_context=4096, memory_budget_bytes=0
        )
        assert opts_emb.speculative_backend == "mtp"
        assert opts_emb.draft_tokens == 3

        # 2) Valid OMLXM5A1 WITHOUT MTP objects stays "none" even if config.json declares mtp layers
        stripped_dir = _make_model_dir(
            tmp_path / "stripped-mtp",
            mtp_layers=1,
            artifact_objects=["model.layers.0.q_proj.weight"],
        )
        stripped_target = M5Target(
            binary=binary,
            artifact_path=stripped_dir / "model.omlxm5",
            metallib_dir=tmp_path / "bin",
            fake=False,
        )
        opts_stripped = m5_mod.build_m5rt_options(
            stripped_dir,
            stripped_target,
            ModelSettings(),
            default_context=4096,
            memory_budget_bytes=0,
        )
        assert opts_stripped.speculative_backend == "none"
        assert opts_stripped.draft_tokens == 0

        # 3) Explicit m5rt_options.mtp_artifact_path triggers "mtp" and forwards path_overrides
        ext_mtp_file = _write_omlxm5_container(
            tmp_path / "external_mtp.omlxm5", ["mtp.fc.weight"]
        )
        ms_ext_mtp = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            m5rt_options={"mtp_artifact_path": str(ext_mtp_file)},
        )
        opts_ext = m5_mod.build_m5rt_options(
            stripped_dir, stripped_target, ms_ext_mtp, default_context=4096, memory_budget_bytes=0
        )
        assert opts_ext.speculative_backend == "mtp"
        assert opts_ext.draft_tokens == 3
        assert opts_ext.path_overrides["mtp_artifact"] == str(ext_mtp_file)

        # 4) Synthetic placeholder artifact falls back to config.json mtp_num_hidden_layers
        ph_dir = _make_model_dir(
            tmp_path / "placeholder-mtp", mtp_layers=1, placeholder_artifact=True
        )
        ph_target = M5Target(
            binary=binary,
            artifact_path=ph_dir / "model.omlxm5",
            metallib_dir=tmp_path / "bin",
            fake=False,
        )
        opts_ph = m5_mod.build_m5rt_options(
            ph_dir, ph_target, ModelSettings(), default_context=4096, memory_budget_bytes=0
        )
        assert opts_ph.speculative_backend == "mtp"


# ---------------------------------------------------------------------------
# EXT-6: Admin API _mtp_compat_for_model, _dflash_compat_for_model & auto-route
# ---------------------------------------------------------------------------


class TestExt6AdminCompatAndAutoRoute:
    def test_m5rt_qwen38_and_qwen36_omlxm5_compat_and_external_blocking(self, tmp_path: Path):
        # Pure .omlxm5 Qwen3.8 model (NO .safetensors files at all!) with embedded MTP weights
        q38_dir = _make_model_dir(
            tmp_path / "Qwen3.8-27B-m5",
            model_type="qwen3_8",
            mtp_layers=1,
            artifact_objects=["model.layers.0.weight", "mtp.norm.weight"],
        )
        info_q38 = {
            "id": "Qwen3.8-27B-m5",
            "model_path": str(q38_dir),
            "config_model_type": "qwen3_8",
            "engine_type": "batched",
        }
        mtp_ok, mtp_reason = admin_routes._mtp_compat_for_model(
            info_q38, external_engine=M5RT_ENGINE_ID
        )
        assert mtp_ok is True, mtp_reason

        df_ok, df_reason = admin_routes._dflash_compat_for_model(
            info_q38, external_engine=M5RT_ENGINE_ID
        )
        assert df_ok is True, df_reason

        # Qwen3.6 MoE on m5rt also passes DFlash compat
        q36_dir = _make_model_dir(
            tmp_path / "Qwen3.6-35B-m5",
            model_type="qwen3_6_moe",
            artifact_objects=["model.layers.0.weight"],
        )
        info_q36 = {
            "id": "Qwen3.6-35B-m5",
            "model_path": str(q36_dir),
            "config_model_type": "qwen3_6_moe",
            "engine_type": "batched",
        }
        df36_ok, df36_reason = admin_routes._dflash_compat_for_model(
            info_q36, external_engine=M5RT_ENGINE_ID
        )
        assert df36_ok is True, df36_reason

        # Stripped OMLXM5A1 (no embedded MTP and no mtp.omlxm5) rejected for m5rt MTP
        mtp36_ok, _ = admin_routes._mtp_compat_for_model(
            info_q36, external_engine=M5RT_ENGINE_ID
        )
        assert mtp36_ok is False

        # External engine without dflash_enabled (e.g. splash) blocks DFlash even for qwen3_5
        q35_info = {
            "id": "Qwen3.5-Splash",
            "model_path": str(q36_dir),
            "config_model_type": "qwen3_5",
            "engine_type": "batched",
        }
        splash_df_ok, splash_df_reason = admin_routes._dflash_compat_for_model(
            q35_info, external_engine="splash"
        )
        assert splash_df_ok is False
        assert "external engine" in splash_df_reason

    def test_update_model_settings_uses_resolved_auto_routed_engine(
        self, tmp_path: Path, monkeypatch
    ):
        # Create a Path B provider that does NOT support mtp_enabled or dflash_enabled
        providers_root = tmp_path / "Providers"
        prov_dir = providers_root / "io.community.llama-metal" / "current"
        (prov_dir / "bin").mkdir(parents=True)
        daemon = _make_fake_daemon(prov_dir / "bin" / "llama_daemon", version="0.2.0", protocol=2)
        (prov_dir / "provider.json").write_text(
            json.dumps(
                {
                    "provider_id": "io.community.llama-metal",
                    "engine_id": "llama-metal",
                    "display_name": "Llama Metal",
                    "protocol_version": 2,
                    "entrypoint": str(daemon.relative_to(prov_dir)),
                    "supported_standard_settings": ["max_context_window"],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(providers_root))

        # Model A: auto-routed to Path B "llama-metal" (no MTP/DFlash support), even with qwen3_5 + mtp safetensors!
        path_b_dir = _make_model_dir(
            tmp_path / "AutoPathB", model_type="qwen3_5", mtp_layers=1
        )
        (path_b_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"mtp.fc.weight": "model.safetensors"}}),
            encoding="utf-8",
        )

        # Model B: auto-routed to "m5rt", Qwen3.8 pure .omlxm5 with embedded MTP
        m5_dir = _make_model_dir(
            tmp_path / "AutoM5",
            model_type="qwen3_8",
            mtp_layers=1,
            artifact_objects=["model.layers.0.weight", "language_model.mtp.norm.weight"],
        )

        entries = {
            "AutoPathB": MagicMock(
                model_id="AutoPathB",
                model_path=str(path_b_dir),
                config_model_type="qwen3_5",
                engine_type="batched",
                model_type="llm",
                model_format="safetensors",
                engine=None,
                load_failed=False,
                is_pinned=False,
            ),
            "AutoM5": MagicMock(
                model_id="AutoM5",
                model_path=str(m5_dir),
                config_model_type="qwen3_8",
                engine_type="batched",
                model_type="llm",
                model_format="safetensors",
                engine=None,
                load_failed=False,
                is_pinned=False,
            ),
        }
        stored_settings = {
            "AutoPathB": ModelSettings(),  # external_engine=None -> auto-routed!
            "AutoM5": ModelSettings(),  # external_engine=None -> auto-routed!
        }

        pool = MagicMock()
        pool.get_entry.side_effect = lambda mid: entries.get(mid)
        pool._entries = entries
        pool._engine_runtime_signature.return_value = ("sig",)
        pool.external_engine_id.side_effect = lambda mid, ms: (
            "llama-metal" if mid == "AutoPathB" else M5RT_ENGINE_ID
        )

        sm = MagicMock()
        sm.get_settings.side_effect = lambda mid: stored_settings[mid]
        sm.get_all_settings.return_value = stored_settings

        monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
        monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: sm)
        monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)

        # 1) Auto-routed Path B model must be blocked from enabling MTP & DFlash (HTTP 400)
        with pytest.raises(HTTPException) as exc_mtp:
            asyncio.run(
                admin_routes.update_model_settings(
                    "AutoPathB",
                    admin_routes.ModelSettingsRequest(mtp_enabled=True),
                    is_admin=True,
                )
            )
        assert exc_mtp.value.status_code == 400

        with pytest.raises(HTTPException) as exc_df:
            asyncio.run(
                admin_routes.update_model_settings(
                    "AutoPathB",
                    admin_routes.ModelSettingsRequest(dflash_enabled=True),
                    is_admin=True,
                )
            )
        assert exc_df.value.status_code == 400

        # 2) Auto-routed m5rt Qwen3.8 pure-.omlxm5 model succeeds for MTP, DFlash, and UI KV off!
        res_mtp = asyncio.run(
            admin_routes.update_model_settings(
                "AutoM5",
                admin_routes.ModelSettingsRequest(mtp_enabled=True),
                is_admin=True,
            )
        )
        assert res_mtp["success"] is True
        assert stored_settings["AutoM5"].mtp_enabled is True

        res_df = asyncio.run(
            admin_routes.update_model_settings(
                "AutoM5",
                admin_routes.ModelSettingsRequest(mtp_enabled=False, dflash_enabled=True),
                is_admin=True,
            )
        )
        assert res_df["success"] is True
        assert stored_settings["AutoM5"].dflash_enabled is True

        # Also verify EXT-4 auto-routed UI KV disable persists bf16 override
        res_kv = asyncio.run(
            admin_routes.update_model_settings(
                "AutoM5",
                admin_routes.ModelSettingsRequest(turboquant_kv_enabled=False),
                is_admin=True,
            )
        )
        assert res_kv["success"] is True
        assert stored_settings["AutoM5"].m5rt_options == {"kv_dtype": "bf16"}


# ---------------------------------------------------------------------------
# EXT-7: Offline bundled release config + embedded Providers/ discovery
# ---------------------------------------------------------------------------


class TestExt7OfflineBundleDiscovery:
    def test_workflow_builds_dmg_only_without_wheel_tarball_bloat(self):
        text = SYNC_WORKFLOW.read_text(encoding="utf-8")
        assert "./omlx-extend/scripts/build_omlx_extend_dmg.sh" in text
        assert "ASSETS=(dist/omlx-extend/*.dmg dist/omlx-extend/*.dmg.sha256)" in text
        assert "python -m build --wheel" not in text
        assert "-slim.tar.gz" not in text

    def test_workflow_defers_release_until_sync_pr_merged(self):
        import yaml

        wf = yaml.safe_load(SYNC_WORKFLOW.read_text(encoding="utf-8"))
        steps = wf["jobs"]["sync-patch-test-release"]["steps"]
        by_name = {step["name"]: step for step in steps if "name" in step}

        release_step = by_name["Tag and publish oMLX-extend GitHub Release"]
        cond = release_step["if"]
        # Release only runs when main has the synced code (unchanged or pushed directly to main),
        # never on an unmerged PR branch when branch protection blocks direct push.
        assert "steps.sync.outputs.tree_changed == 'false' || steps.commit_sync.outputs.pushed_to_main == 'true'" in cond
        assert "steps.upstream.outputs.already_released != 'true' || steps.upstream.outputs.release_mode == 'mix' || steps.upstream.outputs.release_mode == 'force'" in cond
        assert "inputs.dry_run != true" in cond

        run_script = release_step["run"]
        # Force tag push must only occur when release_mode == force
        assert 'if [[ "${RELEASE_MODE}" == "force" ]]; then' in run_script
        assert 'git tag "${EXTEND_TAG}" HEAD' in run_script
        assert 'git push origin "refs/tags/${EXTEND_TAG}"' in run_script

        # Sync commit step must stage omlx + UPSTREAM_VERSION + patches before committing/pushing
        commit_step = by_name["Commit synced monorepo changes to main (or open sync PR if branch-protected)"]
        commit_run = commit_step["run"]
        add_idx = commit_run.index("git add -A -- omlx omlx-extend/UPSTREAM_VERSION omlx-extend/patches")
        commit_idx = commit_run.index('git commit -m "${COMMIT_MSG}"')
        push_idx = commit_run.index("if git push origin HEAD:main; then")
        diff_idx = commit_run.index(
            'git diff --quiet "origin/${SYNC_BRANCH}" HEAD -- omlx omlx-extend/UPSTREAM_VERSION omlx-extend/patches'
        )
        assert add_idx < commit_idx < push_idx < diff_idx

        bundle_step = by_name["Build macOS oMLX-extend DMG installer"]
        assert "./omlx-extend/scripts/build_omlx_extend_dmg.sh" in bundle_step["run"]

    def test_bundled_offline_archive_discovery_in_launcher_and_sidecar(
        self, tmp_path: Path, monkeypatch
    ):
        # Construct exact offline bundled release tree from sync-omlx-extend.yml
        bundle_dir = tmp_path / "oMLX-v0.8.0-extend+ninfer-metal-miv-v0.01-macos27-arm64"
        (bundle_dir / "scripts").mkdir(parents=True)
        (bundle_dir / "config").mkdir(parents=True)
        shutil.copyfile(START_SERVE_SCRIPT, bundle_dir / "scripts" / "start_ninfer_serve.sh")
        (bundle_dir / "scripts" / "start_ninfer_serve.sh").chmod(0o755)
        shutil.copyfile(
            REPO_ROOT / "omlx-extend" / "config" / "ninfer-metal.toml",
            bundle_dir / "config" / "ninfer-metal.toml",
        )

        prov_ver = bundle_dir / "Providers" / "io.ninfer.metal" / "versions" / "miv-v0.01"
        (prov_ver / "bin").mkdir(parents=True)
        serve_bin = prov_ver / "bin" / "ninfer-metal-serve"
        serve_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        serve_bin.chmod(0o755)
        daemon_bin = _make_fake_daemon(prov_ver / "bin" / "mseries_backendd")
        (prov_ver / "provider.json").write_text(
            json.dumps(
                {
                    "provider_id": "io.ninfer.metal",
                    "engine_id": "m5rt",
                    "display_name": "miv Core Backend",
                    "display_version": "miv-v0.01",
                    "protocol_version": 2,
                    "entrypoint": "bin/mseries_backendd",
                }
            ),
            encoding="utf-8",
        )
        os.symlink("versions/miv-v0.01", bundle_dir / "Providers" / "io.ninfer.metal" / "current")

        fake_home = tmp_path / "clean_home"
        fake_home.mkdir()
        model_dir = _make_model_dir(tmp_path / "ModelA", placeholder_artifact=True)

        # 1) Launcher discovers bundled binary, config/ninfer-metal.toml, and --providers-dir
        env = {
            "HOME": str(fake_home),
            "PATH": "/usr/bin:/bin",
        }
        res = subprocess.run(
            [
                "bash",
                str(bundle_dir / "scripts" / "start_ninfer_serve.sh"),
                "--dry-run",
                str(model_dir),
            ],
            cwd=fake_home,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        expected_bin = bundle_dir / "Providers" / "io.ninfer.metal" / "current" / "bin" / "ninfer-metal-serve"
        expected_cfg = bundle_dir / "config" / "ninfer-metal.toml"
        expected_prov = bundle_dir / "Providers"
        assert f"Binary:    {expected_bin}" in res.stdout
        assert f"Config:    {expected_cfg}" in res.stdout
        assert f"Providers: {expected_prov}" in res.stdout
        assert f"--providers-dir {expected_prov}" in res.stdout

        # 2) m5_sidecar discovers bundled Providers/ & config/ninfer-metal.toml when run from bundle dir
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.delenv("OMLX_PROVIDERS_DIR", raising=False)
        monkeypatch.delenv("NINFER_METAL_BIN", raising=False)
        monkeypatch.delenv("OMLX_M5RT_BIN", raising=False)
        monkeypatch.delenv("NINFER_METAL_CONFIG", raising=False)
        monkeypatch.delenv("NINFER_METAL_NO_DEFAULT_CONFIG", raising=False)
        monkeypatch.chdir(bundle_dir)

        assert m5_mod.has_m5rt_binary_candidate() is True
        found_bin, err = m5_mod.find_m5rt_binary()
        assert found_bin is not None and err == ""
        assert found_bin.path.resolve() == daemon_bin.resolve()

        found_cfg = m5_mod.discover_ninfer_metal_toml_path()
        assert found_cfg is not None
        assert found_cfg.resolve() == expected_cfg.resolve()


# ---------------------------------------------------------------------------
# EXT-8: SPI v2 LoRA provider manifest, Path B capability gating & patch check
# ---------------------------------------------------------------------------


class TestExt8SpiV2LoRAAndManifestContract:
    def test_provider_manifest_template_declares_lora_and_lora_mode_schema(self):
        tmpl_path = REPO_ROOT / "omlx-extend" / "provider" / "provider.json.template"
        raw = tmpl_path.read_text(encoding="utf-8")
        rendered = (
            raw.replace("@DISPLAY_VERSION@", "miv-v0.02")
            .replace("@SEMVER@", "0.2.0")
            .replace("@GIT_SHA@", "deadbeef")
            .replace("@BUILT_AT@", "2026-11-25T00:00:00Z")
        )
        manifest = json.loads(rendered)

        assert "lora" in manifest["capabilities"]
        assert "lora_adapter" in manifest["supported_standard_settings"]

        by_key = {item["key"]: item for item in manifest["options_schema"]}
        assert "lora_mode" in by_key
        lora_mode_schema = by_key["lora_mode"]
        assert lora_mode_schema["type"] == "enum"
        assert lora_mode_schema["options"] == ["merge", "bypass"]
        assert lora_mode_schema["default"] == "merge"
        assert lora_mode_schema["reload_required"] is True
        assert lora_mode_schema.get("label")

    def test_path_b_lora_capability_gating_and_enum_validation(
        self, tmp_path: Path, monkeypatch
    ):
        from omlx.model_settings import lora_adapter_conflict

        providers_dir = tmp_path / "Providers"

        # Path B provider 1: declares lora_adapter + lora_mode enum
        lora_prov = providers_dir / "io.vendor.lora_mk" / "current"
        (lora_prov / "bin").mkdir(parents=True)
        _make_fake_daemon(lora_prov / "bin" / "vendor_daemon", version="0.1.0", protocol=2)
        (lora_prov / "provider.json").write_text(
            json.dumps(
                {
                    "provider_id": "io.vendor.lora_mk",
                    "engine_id": "vendor-lora-mk",
                    "display_name": "Vendor LoRA MK",
                    "protocol_version": 2,
                    "entrypoint": "bin/vendor_daemon",
                    "capabilities": ["ar", "lora"],
                    "supported_standard_settings": ["max_context_window", "lora_adapter"],
                    "options_schema": [
                        {
                            "key": "lora_mode",
                            "label": "LoRA Mode",
                            "type": "enum",
                            "options": ["merge", "bypass"],
                            "default": "merge",
                            "reload_required": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        # Path B provider 2: does NOT declare lora_adapter
        plain_prov = providers_dir / "io.vendor.plain_mk" / "current"
        (plain_prov / "bin").mkdir(parents=True)
        _make_fake_daemon(plain_prov / "bin" / "plain_daemon", version="0.1.0", protocol=2)
        (plain_prov / "provider.json").write_text(
            json.dumps(
                {
                    "provider_id": "io.vendor.plain_mk",
                    "engine_id": "vendor-plain-mk",
                    "display_name": "Vendor Plain MK",
                    "protocol_version": 2,
                    "entrypoint": "bin/plain_daemon",
                    "capabilities": ["ar"],
                    "supported_standard_settings": ["max_context_window"],
                }
            ),
            encoding="utf-8",
        )

        monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(providers_dir))

        assert (
            lora_adapter_conflict(
                {"lora_adapter": "/tmp/a", "external_engine": "vendor-lora-mk"}
            )
            is None
        )
        assert (
            lora_adapter_conflict(
                {"lora_adapter": "/tmp/a", "external_engine": "io.vendor.lora_mk"}
            )
            is None
        )
        assert (
            lora_adapter_conflict(
                {"lora_adapter": "/tmp/a", "external_engine": "vendor-plain-mk"}
            )
            == "an external engine"
        )

        # Schema-driven enum validation rejects invalid lora_mode on Path B too
        ms = ModelSettings()
        req_ok = admin_routes.ModelSettingsRequest(
            external_engine_options={"vendor-lora-mk": {"lora_mode": "bypass"}}
        )
        m5_mod.apply_m5rt_model_settings_update(
            ms,
            req_ok,
            {"external_engine_options"},
            is_diffusion_model=False,
            external_engine="vendor-lora-mk",
        )
        assert ms.external_engine_options == {"vendor-lora-mk": {"lora_mode": "bypass"}}

        req_bad = admin_routes.ModelSettingsRequest(
            external_engine_options={"vendor-lora-mk": {"lora_mode": "invalid"}}
        )
        with pytest.raises(ValueError, match="lora_mode"):
            m5_mod.apply_m5rt_model_settings_update(
                ms,
                req_bad,
                {"external_engine_options"},
                is_diffusion_model=False,
                external_engine="vendor-lora-mk",
            )

