# SPDX-License-Identifier: Apache-2.0
"""Comprehensive unit, plugin, live `mseries_backendd` (fake + Metal `.omlxm5`),
and real FastAPI `server.py` E2E tests for `M5SidecarEngine` on oMLX v0.8.0.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import shutil
import signal
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

import omlx.server as srv
from omlx.admin import routes as admin_routes
from omlx.api.responses_utils import ResponseStore
from omlx.engine import m5_sidecar as m5_mod
from omlx.engine.base import BaseEngine, GenerationOutput
from omlx.engine.m5_sidecar import (
    EXTERNAL_ENGINE_ENTRY_POINTS_GROUP,
    M5RT_ENGINE_ID,
    ExternalEnginePlugin,
    M5Binary,
    M5Options,
    M5SidecarEngine,
    M5Target,
    _IncrementalDetokenizer,
    _StopMatcher,
    discover_external_engine_plugins,
    find_m5rt_binary,
    get_external_engines,
    m5rt_memory_estimate,
    resolve_m5rt_target,
    validate_m5rt_artifact_file,
)
from omlx.engine.splash import EXTERNAL_ENGINES
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.exceptions import InvalidRequestError, ModelLoadingError
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.scheduler import SchedulerConfig
from omlx.settings import GlobalSettings
from omlx.utils.proc_memory import list_process_group_pids

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_DAEMON_PATH = REPO_ROOT / "build/b6-xcode27/m5rt/mseries_backendd"
LIVE_METALLIB_DIR = REPO_ROOT / "build/b6-xcode27/m5rt/metallib"
AFFINE_FIXTURE_DIR = REPO_ROOT / "build/b6-xcode27/m5rt/fixtures/qwen-execution/affine"


# -- Test helpers & fixtures ---------------------------------------------------


def _build_bpe_chat_tokenizer(dest_dir: Path) -> PreTrainedTokenizerFast:
    """Create a real HuggingFace byte-level BPE tokenizer with ChatML + UTF-8 support."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Sort alphabet deterministically (tokenizers' Rust HashSet varies per call).
    byte_alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab: dict[str, int] = {tok: idx for idx, tok in enumerate(byte_alphabet)}
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    for sp in specials:
        if sp not in vocab:
            vocab[sp] = len(vocab)

    tok_obj = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok_obj.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok_obj.decoder = decoders.ByteLevel()

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        eos_token="<|im_end|>",
        bos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=specials,
    )
    fast.chat_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    fast.save_pretrained(str(dest_dir))
    return fast


def _write_test_model(
    model_dir: Path,
    *,
    vocab_size: int = 512,
    with_artifact: bool = True,
    with_tokenizer: bool = True,
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    tok = _build_bpe_chat_tokenizer(model_dir) if with_tokenizer else None
    actual_vocab = max(vocab_size, len(tok) if tok is not None else vocab_size)
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>") if tok is not None else 256
    endoftext_id = (
        tok.convert_tokens_to_ids("<|endoftext|>") if tok is not None else 255
    )
    config = {
        "model_type": "qwen3_5",
        "eos_token_id": [im_end_id, endoftext_id],
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "vocab_size": actual_vocab,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "max_position_embeddings": 4096,
            "eos_token_id": [im_end_id, endoftext_id],
            "layer_types": ["linear_attention", "full_attention"],
        },
        "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"\x00" * 512)
    if with_artifact:
        (model_dir / "model.omlxm5").write_bytes(b"OMLXM5" + b"\x00" * 1024)
    return model_dir


def _fake_daemon_script(
    path: Path, version: str = "0.1.0", protocol: int = 1
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "--version" ]; then echo "mseries-backendd {version} protocol={protocol}"; exit 0; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


class _MockUdsSidecarServer:
    """In-process asyncio AF_UNIX mock `mseries_backendd` for deterministic protocol tests."""

    def __init__(
        self,
        socket_path: Path,
        *,
        support_push: bool = True,
        support_v2_coexistence: bool = True,
        token_plan: list[list[int]] | None = None,
        step_delay_s: float = 0.0,
        drop_mid_stream: bool = False,
        early_push_before_submit_reply: bool = False,
    ) -> None:
        self.socket_path = socket_path
        self.support_push = support_push
        self.support_v2_coexistence = support_v2_coexistence
        self.token_plan = token_plan
        self.step_delay_s = step_delay_s
        self.drop_mid_stream = drop_mid_stream
        self.early_push_before_submit_reply = early_push_before_submit_reply

        self.server: asyncio.AbstractServer | None = None
        self.load_params: list[dict[str, Any]] = []
        self.trim_calls: list[int] = []
        self.scheduler_hints: list[bool] = []
        self.resident_bytes = 128 * 1024 * 1024
        self.submitted_requests: list[dict[str, Any]] = []
        self.cancelled_ids: list[int] = []
        self.shutdown_called = False
        self._next_req_id = 1
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._active_tasks: dict[int, asyncio.Task[None]] = {}
        self._writers: list[asyncio.StreamWriter] = []

    async def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        self.server = await asyncio.start_unix_server(
            self._handle_client, path=str(self.socket_path)
        )

    async def close(self) -> None:
        for task in list(self._active_tasks.values()):
            task.cancel()
        for writer in list(self._writers):
            writer.close()
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        self.socket_path.unlink(missing_ok=True)

    async def _emit(self, event: dict[str, Any]) -> None:
        await self._event_queue.put(event)

    async def _run_request(self, req_id: int, params: dict[str, Any]) -> None:
        stop_seqs = [list(s) for s in (params.get("stop_token_seqs") or []) if s]
        max_new = int(params.get("max_new_tokens", 1))
        batches = (
            list(self.token_plan)
            if self.token_plan is not None
            else [[65 + (i % 26)] for i in range(max_new)]
        )
        emitted: list[int] = []
        try:
            for idx, batch in enumerate(batches):
                if len(emitted) >= max_new:
                    break
                if self.step_delay_s > 0:
                    await asyncio.sleep(self.step_delay_s)
                if self.drop_mid_stream and idx == 1:
                    for w in list(self._writers):
                        w.close()
                    return
                out_batch: list[int] = []
                stopped = False
                for tok in batch:
                    if len(emitted) + len(out_batch) >= max_new:
                        break
                    candidate = emitted + out_batch + [tok]
                    if any(
                        len(seq) <= len(candidate) and candidate[-len(seq) :] == seq
                        for seq in stop_seqs
                    ):
                        stopped = True
                        break
                    out_batch.append(tok)
                if out_batch:
                    emitted.extend(out_batch)
                    await self._emit(
                        {
                            "request_id": req_id,
                            "tokens": out_batch,
                            "kind": "tokens",
                            "finish_reason": "",
                            "error": "",
                        }
                    )
                if stopped:
                    await self._emit(
                        {
                            "request_id": req_id,
                            "tokens": [],
                            "kind": "finished",
                            "finish_reason": "stop",
                            "error": "",
                        }
                    )
                    return

            reason = "length" if len(emitted) >= max_new else "stop"
            await self._emit(
                {
                    "request_id": req_id,
                    "tokens": [],
                    "kind": "finished",
                    "finish_reason": reason,
                    "error": "",
                }
            )
        except asyncio.CancelledError:
            await self._emit(
                {
                    "request_id": req_id,
                    "tokens": [],
                    "kind": "aborted",
                    "finish_reason": "cancelled",
                    "error": "",
                }
            )
            raise

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._writers.append(writer)
        try:
            while raw := await reader.readline():
                msg = json.loads(raw.decode("utf-8"))
                rpc_id = msg["id"]
                method = msg["method"]
                params = msg.get("params") or {}

                if method == "hello":
                    caps = ["ar", "dflash2", "cancel", "stats", "capacity"]
                    if self.support_push:
                        caps.append("subscribe_events")
                    if self.support_v2_coexistence:
                        caps.extend(["mtp", "kv_affine4_g128", "memory_budget_bytes", "trim_cache", "set_scheduler_hint"])
                    res = {
                        "provider_id": "io.ninfer.metal" if self.support_v2_coexistence else "com.omlx.mseries",
                        "provider_version": "miv-v0.01" if self.support_v2_coexistence else "0.1.0",
                        "protocol_version": 2 if self.support_v2_coexistence else 1,
                        "engine_contract_version": 1,
                        "capabilities": caps,
                        "pid": os.getpid(),
                        "phys_footprint_bytes": self.resident_bytes,
                    }
                    await self._send_ok(writer, rpc_id, res)
                elif method == "load":
                    self.load_params.append(dict(params))
                    lora_path = str(params.get("lora_adapter") or "")
                    lora_mode = str(params.get("lora_mode") or "merge")
                    lora_buf = (4 * 1024 * 1024 if lora_mode == "bypass" else 16 * 1024 * 1024) if lora_path else 0
                    self._mock_lora_state = {
                        "lora_adapter": lora_path,
                        "lora_mode": lora_mode,
                        "lora_buffer_bytes": lora_buf,
                        "lora_adapted_layers": 2 if lora_path else 0,
                    }
                    await self._send_ok(
                        writer,
                        rpc_id,
                        {
                            "backend_kind": "mock",
                            "prefix_hits": 3,
                            "prefix_partial_hits": 1,
                            "prefix_misses": 2,
                            "prefix_evictions": 0,
                            "prefix_restored_tokens": 64,
                            "pid": os.getpid(),
                            "phys_footprint_bytes": self.resident_bytes,
                            **self._mock_lora_state,
                        },
                    )
                elif method == "reload_lora":
                    if not getattr(self, "support_reload_lora", True):
                        await self._send_err(writer, rpc_id, "unknown sidecar method")
                        continue
                    lora_path = str(params.get("lora_adapter") or "")
                    lora_mode = str(params.get("lora_mode") or "merge")
                    lora_buf = (4 * 1024 * 1024 if lora_mode == "bypass" else 16 * 1024 * 1024) if lora_path else 0
                    self._mock_lora_state = {
                        "lora_adapter": lora_path,
                        "lora_mode": lora_mode,
                        "lora_buffer_bytes": lora_buf,
                        "lora_adapted_layers": 2 if lora_path else 0,
                    }
                    if not hasattr(self, "reload_lora_calls"):
                        self.reload_lora_calls = []
                    self.reload_lora_calls.append(dict(params))
                    await self._send_ok(
                        writer,
                        rpc_id,
                        {
                            "reloaded": True,
                            "hot_reloaded": lora_mode == "bypass",
                            "pid": os.getpid(),
                            "phys_footprint_bytes": self.resident_bytes,
                            **self._mock_lora_state,
                        },
                    )
                elif method == "capacity":
                    await self._send_ok(
                        writer,
                        rpc_id,
                        {
                            "max_concurrency": 1,
                            "active_requests": len(self._active_tasks),
                            "queued_requests": 0,
                            "accepting": True,
                            "pid": os.getpid(),
                            "phys_footprint_bytes": self.resident_bytes,
                        },
                    )
                elif method == "stats":
                    lora_state = getattr(self, "_mock_lora_state", {
                        "lora_adapter": "",
                        "lora_mode": "merge",
                        "lora_buffer_bytes": 0,
                        "lora_adapted_layers": 0,
                    })
                    await self._send_ok(
                        writer,
                        rpc_id,
                        {
                            "submitted": len(self.submitted_requests),
                            "completed": 1,
                            "aborted": len(self.cancelled_ids),
                            "failed": 0,
                            "emitted_tokens": 10,
                            "backend_kind": "mock",
                            "prefix_hits": 3,
                            "prefix_partial_hits": 1,
                            "prefix_misses": 2,
                            "prefix_evictions": 0,
                            "prefix_restored_tokens": 64,
                            "pid": os.getpid(),
                            "phys_footprint_bytes": self.resident_bytes,
                            **lora_state,
                        },
                    )
                elif method == "submit":
                    req_id = self._next_req_id
                    self._next_req_id += 1
                    self.submitted_requests.append(params)
                    task = asyncio.create_task(self._run_request(req_id, params))
                    self._active_tasks[req_id] = task
                    if self.early_push_before_submit_reply:
                        # Yield so push stream delivers events BEFORE submit replies!
                        await asyncio.sleep(0.02)
                    await self._send_ok(writer, rpc_id, {"request_id": req_id})
                elif method == "cancel":
                    target_id = int(params["request_id"])
                    self.cancelled_ids.append(target_id)
                    task = self._active_tasks.pop(target_id, None)
                    cancelled = False
                    if task is not None and not task.done():
                        task.cancel()
                        cancelled = True
                    await self._send_ok(writer, rpc_id, {"cancelled": cancelled})
                elif method == "poll":
                    events = []
                    try:
                        ev = await asyncio.wait_for(
                            self._event_queue.get(), timeout=0.02
                        )
                        events.append(ev)
                        while not self._event_queue.empty():
                            events.append(self._event_queue.get_nowait())
                    except asyncio.TimeoutError:
                        pass
                    await self._send_ok(writer, rpc_id, {"events": events})
                elif method == "subscribe_events":
                    if not self.support_push:
                        writer.write(
                            (
                                json.dumps(
                                    {
                                        "id": rpc_id,
                                        "ok": False,
                                        "error": {
                                            "code": "request_failed",
                                            "message": "unknown sidecar method",
                                        },
                                    }
                                )
                                + "\n"
                            ).encode("utf-8")
                        )
                        await writer.drain()
                        continue
                    await self._send_ok(writer, rpc_id, {"subscribed": True})
                    while not self.shutdown_called:
                        ev = await self._event_queue.get()
                        writer.write(
                            (json.dumps({"event": ev}) + "\n").encode("utf-8")
                        )
                        await writer.drain()
                        if ev.get("kind") == "eof":
                            return
                elif method == "trim_cache":
                    if not self.support_v2_coexistence:
                        await self._send_err(writer, rpc_id, "unknown sidecar method")
                        continue
                    target_free = int(params.get("target_free_bytes", 0))
                    self.trim_calls.append(target_free)
                    freed = min(target_free, max(0, self.resident_bytes - 32 * 1024 * 1024))
                    self.resident_bytes -= freed
                    await self._send_ok(
                        writer,
                        rpc_id,
                        {
                            "freed_bytes": freed,
                            "resident_bytes": self.resident_bytes,
                            "phys_footprint_bytes": self.resident_bytes,
                        },
                    )
                elif method == "set_scheduler_hint":
                    if not self.support_v2_coexistence:
                        await self._send_err(writer, rpc_id, "unknown sidecar method")
                        continue
                    contended = bool(params.get("gpu_contended", False))
                    self.scheduler_hints.append(contended)
                    await self._send_ok(
                        writer,
                        rpc_id,
                        {"gpu_contended": contended},
                    )
                elif method == "shutdown":
                    self.shutdown_called = True
                    await self._event_queue.put({"kind": "eof"})
                    await self._send_ok(writer, rpc_id, {"shutdown": True})
                    return
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass

    @staticmethod
    async def _send_ok(
        writer: asyncio.StreamWriter, rpc_id: Any, result: dict[str, Any]
    ) -> None:
        writer.write(
            (json.dumps({"id": rpc_id, "ok": True, "result": result}) + "\n").encode(
                "utf-8"
            )
        )
        await writer.drain()

    @staticmethod
    async def _send_err(
        writer: asyncio.StreamWriter, rpc_id: Any, message: str
    ) -> None:
        writer.write(
            (
                json.dumps(
                    {
                        "id": rpc_id,
                        "ok": False,
                        "error": {"code": "request_failed", "message": message},
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
        await writer.drain()


# -- 1. Resolution, Detokenizer, Stop & Plugin Discovery Unit Tests ------------


class TestResolutionAndPrimitives:
    def test_external_engines_registers_splash_first_and_m5rt(self):
        assert list(EXTERNAL_ENGINES.keys())[:2] == ["splash", "m5rt"]
        assert EXTERNAL_ENGINES["m5rt"] == "miv Core Backend"

    def test_binary_and_artifact_resolution_orders(self, tmp_path, monkeypatch):
        bin_script = _fake_daemon_script(tmp_path / "bin" / "mseries_backendd")
        metallibs = tmp_path / "bin" / "metallib"
        metallibs.mkdir()
        (metallibs / "omlx_m5_qwen_ar.metallib").write_bytes(b"MTLB")

        # 1) Explicit path wins
        binary, reason = find_m5rt_binary(str(bin_script))
        assert binary is not None and reason == ""
        assert binary.path == bin_script.resolve()
        assert binary.auto_detected is False

        # 2) OMLX_M5RT_BIN env
        monkeypatch.setenv("OMLX_M5RT_BIN", str(bin_script))
        binary_env, _ = find_m5rt_binary(None)
        assert binary_env is not None and binary_env.auto_detected is True

        # 3) Artifact resolution via .cache/b0/artifacts source_config_sha256
        model_dir = _write_test_model(tmp_path / "models" / "Qwen3.8-Test", with_artifact=False)
        cache_dir = tmp_path / "artifact-cache"
        cache_dir.mkdir()
        cached_art = cache_dir / "qwen38-test-oq4e.omlxm5"
        cached_art.write_bytes(b"OMLXM5" + b"\x00" * 2048)
        cfg_sha = hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest()
        (cache_dir / "qwen38-test-oq4e.omlxm5.conversion.json").write_text(
            json.dumps(
                {
                    "source_config_sha256": cfg_sha,
                    "source_repo": "Jundot/Qwen3.8-Test",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("OMLX_M5RT_ARTIFACT_DIR", str(cache_dir))

        target, err = resolve_m5rt_target(model_dir, None, ModelSettings())
        assert err == ""
        assert target is not None
        assert target.artifact_path == cached_art.resolve()
        assert target.metallib_dir == metallibs.resolve()
        assert m5rt_memory_estimate(model_dir, target, 4096, ModelSettings()) > 2048

    def test_validate_artifact_file_accepts_omlxm5_files_and_rejects_dirs(self, tmp_path):
        art = tmp_path / "weights.omlxm5"
        art.write_bytes(b"OMLXM5")
        assert validate_m5rt_artifact_file(art) == art.resolve()

        with pytest.raises(ValueError, match="not a directory"):
            validate_m5rt_artifact_file(tmp_path)
        bad_ext = tmp_path / "weights.safetensors"
        bad_ext.write_bytes(b"SAFE")
        with pytest.raises(ValueError, match=r"\.omlxm5"):
            validate_m5rt_artifact_file(bad_ext)

    def test_rolling_prefix_cache_option_validation_and_load_param_forwarding(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("NINFER_METAL_NO_DEFAULT_CONFIG", "1")
        for valid in (True, False, "on", "off", "1", "0", "true", "false", "ON", None):
            validated = m5_mod.validate_m5rt_options({"rolling_prefix_cache": valid})
            assert validated == {"rolling_prefix_cache": valid}

        for invalid in ("maybe", 2, []):
            with pytest.raises(ValueError, match="rolling_prefix_cache"):
                m5_mod.validate_m5rt_options({"rolling_prefix_cache": invalid})

        model_dir = _write_test_model(tmp_path / "rolling-model")
        fake_bin = _fake_daemon_script(tmp_path / "bin" / "mseries_backendd")
        binary, _ = find_m5rt_binary(str(fake_bin))
        assert binary is not None
        target = M5Target(
            binary=binary,
            artifact_path=model_dir / "model.omlxm5",
            metallib_dir=tmp_path / "bin",
            fake=False,
        )

        # Unset: omitted from load params so daemon default / env applies
        ms_default = ModelSettings(external_engine=M5RT_ENGINE_ID, m5rt_options={})
        opts_default = m5_mod.build_m5rt_options(
            model_dir, target, ms_default, default_context=4096, memory_budget_bytes=0
        )
        assert opts_default.rolling_prefix_cache is None
        eng_default = M5SidecarEngine(
            model_name=str(model_dir),
            model_id="rolling-default",
            target=target,
            log_dir=tmp_path / "logs",
            options=opts_default,
        )
        params_default = eng_default._build_load_params()
        assert "rolling_prefix_cache" not in params_default
        assert "rolling_prefix_cache" not in params_default["path_overrides"]

        # Explicit "on" / True -> forwarded to both top-level JSON and path_overrides
        ms_on = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            m5rt_options={
                "rolling_prefix_cache": "on",
                "prefix_cache_bytes": 4 * (1 << 30),
            },
        )
        opts_on = m5_mod.build_m5rt_options(
            model_dir, target, ms_on, default_context=4096, memory_budget_bytes=0
        )
        assert opts_on.rolling_prefix_cache is True
        eng_on = M5SidecarEngine(
            model_name=str(model_dir),
            model_id="rolling-on",
            target=target,
            log_dir=tmp_path / "logs",
            options=opts_on,
        )
        params_on = eng_on._build_load_params()
        assert params_on["rolling_prefix_cache"] == "on"
        assert params_on["path_overrides"]["rolling_prefix_cache"] == "on"
        assert params_on["path_overrides"]["prefix_cache_bytes"] == str(4 * (1 << 30))

        # Explicit False -> forwarded as "off"
        ms_off = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            m5rt_options={"rolling_prefix_cache": False},
        )
        opts_off = m5_mod.build_m5rt_options(
            model_dir, target, ms_off, default_context=4096, memory_budget_bytes=0
        )
        assert opts_off.rolling_prefix_cache is False
        eng_off = M5SidecarEngine(
            model_name=str(model_dir),
            model_id="rolling-off",
            target=target,
            log_dir=tmp_path / "logs",
            options=opts_off,
        )
        params_off = eng_off._build_load_params()
        assert params_off["rolling_prefix_cache"] == "off"
        assert params_off["path_overrides"]["rolling_prefix_cache"] == "off"

        # Runtime signature reflects rolling_prefix_cache changes
        assert m5_mod.m5rt_runtime_signature(ms_on, None) != m5_mod.m5rt_runtime_signature(
            ms_off, None
        )

    def test_sync_m5rt_ui_kv_setting_updates_external_engine_options(self):
        ms = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            turboquant_kv_enabled=False,
            m5rt_options={"kv_dtype": "bf16", "max_context": 65536},
            external_engine_options={"m5rt": {"kv_dtype": "bf16", "max_context": 65536}},
        )
        req_enable = SimpleNamespace(turboquant_kv_enabled=True)
        m5_mod.sync_m5rt_ui_kv_setting(ms, req_enable, {"turboquant_kv_enabled"})
        assert ms.m5rt_options == {"max_context": 65536}
        assert ms.external_engine_options == {"m5rt": {"max_context": 65536}}

        req_disable = SimpleNamespace(turboquant_kv_enabled=False)
        m5_mod.sync_m5rt_ui_kv_setting(ms, req_disable, {"turboquant_kv_enabled"})
        assert ms.m5rt_options["kv_dtype"] == "bf16"
        assert ms.external_engine_options["m5rt"]["kv_dtype"] == "bf16"

    def test_incremental_detokenizer_holds_back_incomplete_utf8(self, tmp_path):
        tok = _build_bpe_chat_tokenizer(tmp_path / "tok")
        text = "Hello 世界 🚀!"
        token_ids = tok.encode(text, add_special_tokens=False)
        assert len(token_ids) >= len("Hello ") + 7  # multi-byte UTF-8 spans multiple tokens

        special_ids = {
            tok.convert_tokens_to_ids("<|im_end|>"),
            tok.convert_tokens_to_ids("<|endoftext|>"),
        }
        detok = _IncrementalDetokenizer(tok, tmp_path / "tok", special_ids)
        pieces: list[str] = []
        for tid in token_ids:
            piece = detok.add_tokens([tid])
            assert "\ufffd" not in piece, f"Leaked replacement char in piece={piece!r}"
            pieces.append(piece)
        pieces.append(detok.finalize())
        assert "".join(pieces) == text

        # Also verify the sliding-window fallback when mlx_lm BPE detokenizer is absent:
        fallback_detok = _IncrementalDetokenizer(tok, tmp_path / "tok", special_ids)
        fallback_detok._streaming = None
        fb_pieces: list[str] = []
        for tid in token_ids:
            piece = fallback_detok.add_tokens([tid])
            assert "\ufffd" not in piece
            fb_pieces.append(piece)
        fb_pieces.append(fallback_detok.finalize())
        assert "".join(fb_pieces) == text

    def test_stop_matcher_holds_partial_prefixes_and_truncates_cleanly(self):
        matcher = _StopMatcher(stop_token_seqs=[[999]], stop_strings=["</tool>", "STOP"])
        assert matcher.push_text("Hello </to") == "Hello "
        assert not matcher.stopped
        # Disproved prefix flushes cleanly:
        assert matcher.push_text("pic> world ST") == "</topic> world "
        assert not matcher.stopped
        # Completed stop string truncates and marks stopped:
        assert matcher.push_text("OP trailing") == ""
        assert matcher.stopped

    async def test_entry_points_external_engine_plugin_discovery(
        self, tmp_path, monkeypatch
    ):
        model_dir = _write_test_model(tmp_path / "models" / "CustomModel", with_artifact=False)

        class DummyPluginEngine(BaseEngine):
            _prefill_memory_guard_managed_externally = True
            external_engine_id = "custom_sidecar"

            def __init__(self, path: str) -> None:
                self._model_name = path
                self._tok = MagicMock()
                self._tok.encode.return_value = [1, 2]

            @property
            def model_name(self) -> str:
                return self._model_name

            @property
            def tokenizer(self) -> Any:
                return self._tok

            @property
            def model_type(self) -> str | None:
                return "qwen3_5"

            async def start(self) -> None:
                return None

            async def stop(self) -> None:
                return None

            async def generate(self, prompt, **kwargs) -> GenerationOutput:
                return GenerationOutput(text="plugin-ok")

            async def stream_generate(self, prompt, **kwargs):
                yield GenerationOutput(text="plugin-ok")

            async def chat(self, messages, **kwargs) -> GenerationOutput:
                return GenerationOutput(text="plugin-ok")

            async def stream_chat(self, messages, **kwargs):
                yield GenerationOutput(text="plugin-ok")

            def get_stats(self) -> dict[str, Any]:
                return {"engine_type": "custom_sidecar"}

            def get_cache_stats(self) -> dict[str, Any] | None:
                return None

            def process_memory_bytes(self) -> int:
                return 42 * 1024 * 1024

        plugin_descriptor = {
            "id": "custom_sidecar",
            "label": "Custom Sidecar",
            "factory": lambda entry, ms, gs, sc: DummyPluginEngine(entry.model_path),
            "validator": lambda path, ms, gs: (True, ""),
            "status": lambda gs, pool: {"available": True, "version": "2.0.0"},
            "estimate_memory_bytes": lambda entry, ms, gs: 128 * 1024 * 1024,
            "runtime_signature": lambda ms, gs: ("custom_v2",),
        }
        fake_ep = SimpleNamespace(
            name="custom_sidecar",
            group=EXTERNAL_ENGINE_ENTRY_POINTS_GROUP,
            load=lambda: plugin_descriptor,
        )
        monkeypatch.setattr(
            m5_mod.importlib.metadata,
            "entry_points",
            lambda group=None: [fake_ep]
            if group == EXTERNAL_ENGINE_ENTRY_POINTS_GROUP
            else [],
        )

        plugins = discover_external_engine_plugins()
        assert "custom_sidecar" in plugins
        engines_map = get_external_engines()
        assert engines_map["custom_sidecar"] == "Custom Sidecar"

        gs = GlobalSettings(base_path=tmp_path / "base")
        monkeypatch.setattr("omlx.engine_pool.get_global_settings", lambda: gs)
        pool = EnginePool()
        pool._entries["CustomModel"] = EngineEntry(
            model_id="CustomModel",
            model_path=str(model_dir),
            model_type="llm",
            engine_type="batched",
            estimated_size=1024,
        )
        ms = ModelSettings(external_engine="custom_sidecar")
        eng = pool._external_engine_for(pool._entries["CustomModel"], ms, "batched")
        assert isinstance(eng, DummyPluginEngine)

        compat_rows = admin_routes._external_engines_for_model(
            pool, {"id": "CustomModel", "engine_type": "batched", "is_helper": False}, ms
        )
        assert any(
            r["id"] == "custom_sidecar" and r["compatible"] is True for r in compat_rows
        )


# -- 2. Dual-Connection Async UDS Protocol & Showstopper Unit Tests ------------


class TestDualUdsTransportAndShowstoppers:
    async def _attach_mock_engine(
        self,
        tmp_path: Path,
        server: _MockUdsSidecarServer,
    ) -> M5SidecarEngine:
        model_dir = _write_test_model(tmp_path / "model")
        fake_bin = _fake_daemon_script(tmp_path / "bin" / "mseries_backendd")
        binary, _ = find_m5rt_binary(str(fake_bin))
        assert binary is not None
        target = M5Target(
            binary=binary,
            artifact_path=model_dir / "model.omlxm5",
            metallib_dir=tmp_path / "bin",
            fake=False,
        )
        engine = M5SidecarEngine(
            model_name=str(model_dir),
            model_id="test-model",
            target=target,
            log_dir=tmp_path / "logs",
            options=M5Options(max_context=4096, memory_budget_bytes=256 * 1024 * 1024),
        )
        await asyncio.to_thread(engine._load_tokenizer)
        engine._socket_path = server.socket_path
        engine._process = SimpleNamespace(pid=os.getpid(), poll=lambda: None, returncode=None)  # type: ignore[assignment]
        engine._ctrl_reader, engine._ctrl_writer = await engine._open_uds()
        engine._hello_info = await engine._rpc("hello", timeout=5.0)
        engine._last_stats = await engine._rpc("load", engine._build_load_params(), timeout=5.0)
        engine._last_capacity = await engine._rpc("capacity", timeout=5.0)
        await engine._setup_event_stream()
        return engine

    async def test_request_controls_api_and_wire(self, tmp_path, monkeypatch):
        model_dir = _write_test_model(tmp_path / "prompt-model")
        tokenizer = _build_bpe_chat_tokenizer(model_dir)
        sock = Path("/tmp") / f"m5-mock-{os.getpid()}-controls.sock"
        mock_srv = _MockUdsSidecarServer(
            sock, support_push=True, token_plan=[tokenizer.encode("Answer")]
        )
        await mock_srv.start()
        engine = await self._attach_mock_engine(tmp_path / "controls", mock_srv)

        async def get_engine(*args, **kwargs):
            return engine

        pool = MagicMock()
        pool.get_entry.return_value = None
        state = srv.ServerState(default_model="test-model", engine_pool=pool)
        monkeypatch.setattr(srv, "_server_state", state)
        monkeypatch.setattr(srv, "get_engine_for_model", get_engine)
        try:
            transport = httpx.ASGITransport(app=srv.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                messages = [{"role": "user", "content": "Question"}]
                for path, payload, field in (
                    ("/v1/chat/completions", {"messages": messages, "thinking_budget": 0}, "thinking_budget"),
                    ("/v1/completions", {"prompt": "Question", "repetition_context_size": 30}, "repetition_context_size"),
                    ("/v1/responses", {"input": "Question", "frequency_penalty": 0.5}, "frequency_penalty"),
                    ("/v1/messages", {"messages": messages, "thinking": {"budget_tokens": 0}}, "thinking.budget_tokens"),
                ):
                    response = await client.post(
                        path,
                        json={"model": "test-model", "max_tokens": 32, "stream": True, **payload},
                    )
                    assert response.status_code == 400, response.text
                    assert response.json()["error"]["param"] == field
                assert mock_srv.submitted_requests == []

                # An inherited setting must not break existing clients on older daemons.
                state.sampling.repetition_penalty = 1.25
                response = await client.post(
                    "/v1/completions",
                    json={"model": "test-model", "prompt": "Question", "max_tokens": 32},
                )
                assert response.status_code == 200, response.text
                assert "Inherited repetition_penalty is unsupported" in response.headers["warning"]
                assert "repetition_penalty" not in mock_srv.submitted_requests[-1]["sampling"]

                response = await client.post(
                    "/v1/completions",
                    json={"model": "test-model", "prompt": "Question", "max_tokens": 32,
                          "repetition_penalty": 1.0},
                )
                assert response.status_code == 200, response.text
                assert "warning" not in response.headers

                engine._hello_info["capabilities"] += ["thinking_budget", "sampling_penalties"]
                engine._options = replace(
                    engine._options, sampling_contract="legacy_rejection_v1"
                )
                # A daemon can announce the feature while this model uses the legacy contract.
                response = await client.post(
                    "/v1/completions",
                    json={"model": "test-model", "prompt": "Question", "repetition_penalty": 1.5,
                          "stream": True},
                )
                assert response.status_code == 400, response.text
                assert response.json()["error"]["param"] == "repetition_penalty"
                assert "position_coupled_v1" in response.text

                engine._options = replace(
                    engine._options, sampling_contract="position_coupled_v1"
                )
                start_id = engine.tokenizer.convert_tokens_to_ids("<think>")
                end_id = engine.tokenizer.convert_tokens_to_ids("</think>")
                response = await client.post(
                    "/v1/completions",
                    json={"model": "test-model", "prompt": "Question <think>\n",
                          "max_tokens": 32, "thinking_budget": 0, "stream": True},
                )
                assert response.status_code == 200, response.text
                submitted = mock_srv.submitted_requests[-1]
                assert submitted["thinking_budget"] == {
                    "budget": 0, "start_in_thinking": True,
                    "start_token_ids": [start_id], "end_token_ids": [end_id],
                }
                assert submitted["sampling"]["repetition_penalty"] == 1.25
                assert "warning" not in response.headers

                render_calls = 0
                prepare = engine.prepare_chat_prompt

                def count_render(*args, **kwargs):
                    nonlocal render_calls
                    render_calls += 1
                    return prepare(*args, **kwargs)

                monkeypatch.setattr(engine, "prepare_chat_prompt", count_render)
                response = await client.post(
                    "/v1/responses",
                    json={"model": "test-model", "input": "Question", "max_output_tokens": 32,
                          "repetition_penalty": 1.5, "repetition_context_size": 40,
                          "presence_penalty": -0.25, "frequency_penalty": 0.5,
                          "thinking_budget": 0},
                )
                assert response.status_code == 200, response.text
                assert render_calls == 1
                submitted = mock_srv.submitted_requests[-1]
                assert submitted["thinking_budget"]["budget"] == 0
                sampling = submitted["sampling"]
                assert {field: sampling[field] for field in (
                    "repetition_penalty", "repetition_context_size",
                    "presence_penalty", "frequency_penalty",
                )} == {
                    "repetition_penalty": 1.5, "repetition_context_size": 40,
                    "presence_penalty": -0.25, "frequency_penalty": 0.5,
                }

                # A closed block with whitespace in the prompt must not reopen.
                await engine.generate(
                    "Question <think></think>\n", max_tokens=32, thinking_budget=0
                )
                assert mock_srv.submitted_requests[-1]["thinking_budget"]["start_in_thinking"] is False
        finally:
            engine._process = None
            await engine.stop()
            await mock_srv.close()

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "supports_mid_system, keep_prefix, expected_mid_role",
        [(True, True, "system"), (False, True, "user"), (False, False, "merged_system")],
    )
    async def test_responses_replay_preserves_prepared_prompt(
        self, tmp_path, monkeypatch, stream, supports_mid_system, keep_prefix, expected_mid_role
    ):
        sock = Path("/tmp") / f"m5-mock-{os.getpid()}-responses.sock"
        mock_srv = _MockUdsSidecarServer(sock, support_push=True)
        await mock_srv.start()
        engine = await self._attach_mock_engine(tmp_path / "responses", mock_srv)
        tokenizer = engine.tokenizer
        tokenizer._omlx_supports_mid_system_messages = supports_mid_system
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] }}: {{ message['content'] }}"
            "{{ message.get('tool_call_id', '') }}"
            "{{ message.get('tool_calls', []) | tojson }}\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        )

        rendered = []
        apply_template = tokenizer.apply_chat_template

        def capture_template(messages, *args, **kwargs):
            rendered.append(messages)
            if not supports_mid_system:
                assert all(m["role"] != "system" for m in messages[1:])
            return apply_template(messages, *args, **kwargs)

        monkeypatch.setattr(tokenizer, "apply_chat_template", capture_template)

        async def get_engine(*args, **kwargs):
            return engine

        pool = MagicMock()
        pool.get_entry.return_value = None
        settings = GlobalSettings(base_path=tmp_path)
        settings.server.preserve_mid_system_cache = keep_prefix
        state_dir = tmp_path / "response-state"
        state = srv.ServerState(
            default_model="test-model",
            engine_pool=pool,
            global_settings=settings,
            responses_store=ResponseStore(state_dir=state_dir),
        )
        monkeypatch.setattr(srv, "_server_state", state)
        monkeypatch.setattr(srv, "get_engine_for_model", get_engine)
        common = {
            "model": "test-model",
            "max_output_tokens": 256,
            "stream": stream,
            "tools": [{"type": "function", "name": "lookup"}],
        }

        async def respond(client, **kwargs):
            before = len(rendered)
            response = await client.post("/v1/responses", json={**common, **kwargs})
            assert response.status_code == 200, response.text
            assert len(rendered) == before + 1
            if not stream:
                return response.json()
            return next(
                event["response"]
                for line in response.text.splitlines()
                if line.startswith("data: ")
                for event in [json.loads(line[6:])]
                if event["type"] == "response.completed"
            )

        try:
            transport = httpx.ASGITransport(app=srv.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                first_input = [
                    {"role": "system", "content": "Persistent system"},
                    {"role": "user", "content": "Find the value"},
                ]
                mock_srv.token_plan = [
                    tokenizer.encode('<tool_call>{"name":"lookup","arguments":{}}</tool_call>')
                ]
                first = await respond(
                    client, input=first_input, instructions="First request only"
                )
                call = next(item for item in first["output"] if item["type"] == "function_call")
                result = {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": "Blue",
                }
                second_input = [
                    result,
                    {"role": "developer", "content": "Mid-turn marker"},
                    {"role": "user", "content": "Summarize it"},
                ]
                mock_srv.token_plan = [tokenizer.encode("It is blue.")]
                second = await respond(
                    client, input=second_input, previous_response_id=first["id"]
                )

                state.responses_store = ResponseStore(state_dir=state_dir)
                final = {"role": "user", "content": "Repeat the answer"}
                await respond(
                    client, input=[final], previous_response_id=second["id"],
                    instructions="Current request only",
                )
                replay_tokens = mock_srv.submitted_requests[-1]["prompt_tokens"]
                replay_messages = rendered[-1]
                await respond(
                    client,
                    input=[
                        *first_input,
                        *first["output"],
                        *second_input,
                        *second["output"],
                        final,
                    ],
                    instructions="Current request only",
                )
                assert mock_srv.submitted_requests[-1]["prompt_tokens"] == replay_tokens
                assert rendered[-1] == replay_messages
                assert "First request only" not in tokenizer.decode(replay_tokens)
                assert "Current request only" in tokenizer.decode(replay_tokens)

                tool_index = next(
                    i for i, msg in enumerate(replay_messages) if msg["role"] == "tool"
                )
                assert replay_messages[tool_index]["tool_call_id"] == call["call_id"]
                assert replay_messages[tool_index - 1]["tool_calls"][0]["id"] == call["call_id"]
                mid_index = next(
                    i
                    for i, msg in enumerate(replay_messages)
                    if "Mid-turn marker" in msg["content"]
                )
                if expected_mid_role == "merged_system":
                    assert mid_index == 0
                else:
                    assert mid_index > tool_index
                    assert replay_messages[mid_index]["role"] == expected_mid_role

                if supports_mid_system and not stream:
                    chat = {
                        "model": "test-model",
                        "messages": [
                            *first_input,
                            {"role": "system", "content": "Chat mid-turn marker"},
                        ],
                    }
                    before = len(rendered)
                    response = await client.post("/v1/chat/completions", json=chat)
                    assert response.status_code == 200, response.text
                    assert len(rendered) == before + 1
                    assert rendered[-1][-1]["role"] == "system"
                    chat_ids = mock_srv.submitted_requests[-1]["prompt_tokens"]
                    response = await client.post(
                        "/v1/tokenize", json={**chat, "add_generation_prompt": True}
                    )
                    assert response.status_code == 200, response.text
                    assert len(rendered) == before + 2
                    assert response.json()["tokens"] == chat_ids

                # Calling the engine directly still applies its old strict preparation.
                engine.prepare_chat_prompt(
                    [*first_input, {"role": "system", "content": "Direct"}]
                )
                assert rendered[-1][0]["content"].endswith("Direct")
                assert [m["role"] for m in rendered[-1]] == ["system", "user"]
        finally:
            engine._process = None
            await engine.stop()
            await mock_srv.close()

    async def test_push_stream_early_event_race_eos_and_utf8_streaming(self, tmp_path):
        model_dir = _write_test_model(tmp_path / "proto-model")
        tok = AutoTokenizer_or_fast = _build_bpe_chat_tokenizer(model_dir)
        utf8_ids = tok.encode("你好世界!", add_special_tokens=False)
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        # Split UTF-8 bytes one token per push frame, followed by <|im_end|> + trailing tokens
        plan = [[tid] for tid in utf8_ids] + [[im_end_id], [65, 66]]

        sock = Path("/tmp") / f"m5-mock-{os.getpid()}-1.sock"
        mock_srv = _MockUdsSidecarServer(
            sock,
            support_push=True,
            token_plan=plan,
            early_push_before_submit_reply=True,
        )
        await mock_srv.start()
        engine = await self._attach_mock_engine(tmp_path / "run1", mock_srv)
        try:
            assert engine._push_subscribed is True
            assert engine._prefill_memory_guard_managed_externally is True
            assert engine.accepts_image_input is False
            assert engine.max_context_tokens == 4096
            assert engine.reserved_bytes() == 256 * 1024 * 1024
            assert engine.process_memory_bytes() >= 128 * 1024 * 1024

            # Showstopper #1: count_chat_tokens
            msgs = [{"role": "user", "content": "Say hello in Chinese"}]
            counted = engine.count_chat_tokens(msgs)
            assert counted > 4

            # Showstopper #2 + #3: EOS stop_token_seqs + UTF-8 streaming + telemetry
            chunks: list[GenerationOutput] = []
            async for out in engine.stream_chat(msgs, max_tokens=32, temperature=0.7):
                assert "\ufffd" not in out.new_text
                chunks.append(out)

            assert chunks[-1].finished is True
            assert chunks[-1].finish_reason == "stop"
            assert chunks[-1].text == "你好世界!"
            assert chunks[-1].prompt_tokens == counted
            assert chunks[-1].completion_tokens == len(utf8_ids)
            assert chunks[-1].first_token_at is not None

            submitted = mock_srv.submitted_requests[-1]
            assert [im_end_id] in submitted["stop_token_seqs"]
            # Default temperature=0.7, top_k=0 normalized to hardware-valid top_k=64
            assert submitted["sampling"]["top_k"] == 64

            cache_stats = engine.get_runtime_cache_stats()
            assert cache_stats is not None
            assert cache_stats["cache_rates"]["cumulative"]["prefix_hits"] == 3
        finally:
            engine._process = None
            await engine.stop()
            await mock_srv.close()

    async def test_poll_fallback_and_abort_on_disconnect(self, tmp_path):
        sock = Path("/tmp") / f"m5-mock-{os.getpid()}-2.sock"
        mock_srv = _MockUdsSidecarServer(
            sock,
            support_push=False,
            token_plan=[[72], [105], [33], [65], [66], [67]],
            step_delay_s=0.05,
        )
        await mock_srv.start()
        engine = await self._attach_mock_engine(tmp_path / "run2", mock_srv)
        try:
            assert engine._push_subscribed is False
            assert engine.supports_request_scoped_abort is True

            # Abort mid-stream via generator close (simulates client disconnect)
            gen = engine.stream_generate("Prompt", max_tokens=16, _request_id="req-disc")
            first = await anext(gen)
            assert first.finished is False
            live = engine.get_live_requests()
            assert len(live["generating"]) == 1
            await gen.aclose()
            await asyncio.sleep(0.05)
            assert len(mock_srv.cancelled_ids) >= 1
            assert engine.active_request_count() == 0
        finally:
            engine._process = None
            await engine.stop()
            await mock_srv.close()

    async def test_unexpected_socket_drop_sets_runtime_failed_and_wakes_waiters(
        self, tmp_path
    ):
        sock = Path("/tmp") / f"m5-mock-{os.getpid()}-3.sock"
        mock_srv = _MockUdsSidecarServer(
            sock,
            support_push=True,
            token_plan=[[72], [105], [33]],
            step_delay_s=0.02,
            drop_mid_stream=True,
        )
        await mock_srv.start()
        engine = await self._attach_mock_engine(tmp_path / "run3", mock_srv)
        try:
            with pytest.raises(RuntimeError, match="M5 sidecar failed"):
                async for _ in engine.stream_generate("Crash test", max_tokens=8):
                    pass
            assert engine.runtime_failed_reason is not None
        finally:
            engine._process = None
            await engine.stop()
            await mock_srv.close()


# -- 3. Live `mseries_backendd` Daemon + Real Metal `.omlxm5` + FastAPI E2E ----


@pytest.mark.skipif(
    not LIVE_DAEMON_PATH.is_file(),
    reason=f"Live mseries_backendd not built at {LIVE_DAEMON_PATH}",
)
class TestLiveDaemonAndFastAPIE2E:
    async def test_live_daemon_fake_lifecycle_eos_abort_and_watchdog_cleanup(
        self, tmp_path
    ):
        model_dir = _write_test_model(tmp_path / "live-fake-model", with_artifact=False)
        ms = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            m5rt_options={
                "fake": True,
                "fake_options": {"event_batch_size": 1, "step_delay_us": 15_000},
            },
        )
        target, reason = resolve_m5rt_target(model_dir, str(LIVE_DAEMON_PATH), ms)
        assert target is not None, reason
        opts = m5_mod.build_m5rt_options(
            model_dir, target, ms, default_context=2048, memory_budget_bytes=64 * 1024**2
        )
        engine = M5SidecarEngine(
            model_name=str(model_dir),
            model_id="live-fake",
            target=target,
            log_dir=tmp_path / "logs",
            options=opts,
        )
        await engine.start()
        try:
            group_pid = engine._process.pid
            sock_path = engine._socket_path
            assert sock_path is not None and sock_path.exists()
            assert len(list_process_group_pids(group_pid)) >= 1
            assert engine.process_memory_bytes() > 0

            # Test EOS stopping on live daemon: FakeBackend emits base + 1, base + 2, ...
            eos_id = engine._eos_token_ids[0]
            out = await engine.generate(
                prompt=[max(0, eos_id - 3)],
                max_tokens=20,
                temperature=0.0,
            )
            assert out.finish_reason == "stop"
            assert out.completion_tokens == 2  # stopped right on eos_id!

            # Test mid-generation request-scoped abort over live Control UDS:
            seen_chunks = 0

            async def _consume_and_abort():
                nonlocal seen_chunks
                async for chunk in engine.stream_generate(
                    prompt=[1, 2, 3],
                    max_tokens=100,
                    temperature=0.0,
                    _request_id="live-abort-1",
                ):
                    if not chunk.finished:
                        seen_chunks += 1
                        if seen_chunks == 2:
                            cancelled = await engine.abort_request("live-abort-1")
                            assert cancelled is True

            await asyncio.wait_for(_consume_and_abort(), timeout=5.0)
            assert seen_chunks >= 2
            assert engine.active_request_count() == 0
        finally:
            await engine.stop()

        assert list_process_group_pids(group_pid) == []
        assert not sock_path.exists()

    @pytest.mark.skipif(
        not (AFFINE_FIXTURE_DIR / "model.omlxm5").is_file()
        or not (LIVE_METALLIB_DIR / "omlx_m5_qwen_ar.metallib").is_file(),
        reason="2-layer Metal affine .omlxm5 fixture or metallib not present",
    )
    async def test_live_daemon_real_2layer_metal_omlxm5_fixture(self, tmp_path):
        # Stage a model folder pairing the 2-layer affine C++ fixture config with its .omlxm5
        staged = tmp_path / "qwen-2layer-affine"
        shutil.copytree(AFFINE_FIXTURE_DIR / "source", staged)
        artifact_file = AFFINE_FIXTURE_DIR / "model.omlxm5"

        ms = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            m5rt_artifact_path=str(artifact_file),
            m5rt_options={"metallib_dir": str(LIVE_METALLIB_DIR), "max_context": 1024},
        )
        target, reason = resolve_m5rt_target(staged, str(LIVE_DAEMON_PATH), ms)
        assert target is not None, reason

        opts = m5_mod.build_m5rt_options(
            staged, target, ms, default_context=1024, memory_budget_bytes=256 * 1024**2
        )
        engine = M5SidecarEngine(
            model_name=str(staged),
            model_id="qwen-2layer-affine",
            target=target,
            log_dir=tmp_path / "logs",
            options=opts,
        )
        await engine.start()
        try:
            stats = engine.get_stats()
            assert stats["backend_stats"]["backend_kind"] == "native"
            assert stats["backend_stats"]["model_execution_kind"] == "qwen_dense"
            assert engine.process_memory_bytes() >= artifact_file.stat().st_size

            # Verify real Metal GPU execution through both generate() and stream_chat()
            gen_out = await engine.generate(
                prompt=[1, 7, 3, 9, 2],
                max_tokens=6,
                temperature=0.0,
                ignore_eos=True,
            )
            assert gen_out.completion_tokens == 6
            assert len(gen_out.tokens) == 6
            assert gen_out.finish_reason == "length"

            oracle_path = AFFINE_FIXTURE_DIR / "native-first-token.json"
            if oracle_path.is_file():
                oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
                if isinstance(oracle.get("prompt_tokens"), list) and isinstance(
                    oracle.get("first_token"), int
                ):
                    oracle_out = await engine.generate(
                        prompt=oracle["prompt_tokens"],
                        max_tokens=1,
                        temperature=0.0,
                        ignore_eos=True,
                    )
                    assert oracle_out.tokens[0] == oracle["first_token"]

            chat_chunks: list[GenerationOutput] = []
            async for chunk in engine.stream_chat(
                [{"role": "user", "content": "Hi"}],
                max_tokens=4,
                temperature=0.7,
            ):
                chat_chunks.append(chunk)
            assert chat_chunks[-1].finished is True
            assert chat_chunks[-1].prompt_tokens > 0
            assert chat_chunks[-1].completion_tokens >= 1
        finally:
            await engine.stop()

    async def test_real_fastapi_server_e2e_endpoints_admin_and_crash_recovery(
        self, tmp_path, monkeypatch
    ):
        """Full FastAPI E2E test over /v1/chat/completions, /v1/completions, /v1/messages,
        PUT /admin/api/models/{id}/settings (.omlxm5 file validation without HTTP 422),
        and automatic EnginePool crash recovery via `runtime_failed_reason`.
        """
        base_dir = tmp_path / "omlx-base"
        models_root = tmp_path / "models"
        model_id = "Qwen3.8-Sidecar-E2E"
        model_dir = _write_test_model(models_root / model_id, with_artifact=True)
        external_art = tmp_path / "artifacts" / "custom-target.omlxm5"
        external_art.parent.mkdir(parents=True)
        external_art.write_bytes(b"OMLXM5" + b"\x00" * 4096)

        global_settings = GlobalSettings(base_path=base_dir)
        global_settings.integrations.m5rt_path = str(LIVE_DAEMON_PATH)
        global_settings.auth.allow_unauthenticated_inference = True
        global_settings.auth.skip_api_key_verification = True

        settings_mgr = ModelSettingsManager(base_path=base_dir)
        sched_cfg = SchedulerConfig()
        sched_cfg.paged_ssd_cache_dir = None
        pool = EnginePool(scheduler_config=sched_cfg)
        pool._settings_manager = settings_mgr
        pool.discover_models([str(models_root)])
        assert model_id in pool._entries

        # Patch global state for FastAPI server & admin routes
        monkeypatch.setattr("omlx.engine_pool.get_global_settings", lambda: global_settings)
        monkeypatch.setattr("omlx.settings._global_settings", global_settings, raising=False)
        monkeypatch.setattr(srv._server_state, "global_settings", global_settings, raising=False)
        monkeypatch.setattr(srv._server_state, "engine_pool", pool)
        monkeypatch.setattr(srv._server_state, "settings_manager", settings_mgr)
        monkeypatch.setattr(srv._server_state, "responses_store", ResponseStore())
        monkeypatch.setattr(srv._server_state, "api_key", None)

        monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
        monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: settings_mgr)
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: global_settings)
        monkeypatch.setattr(admin_routes, "_get_server_state", lambda: srv._server_state)
        srv.app.dependency_overrides[admin_routes.require_admin] = lambda: True

        transport = httpx.ASGITransport(app=srv.app)
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                # 1) Admin PUT /admin/api/models/{id}/settings accepts .omlxm5 file (NO HTTP 422!)
                put_resp = await client.put(
                    f"/admin/api/models/{model_id}/settings",
                    json={
                        "external_engine": M5RT_ENGINE_ID,
                        "m5rt_artifact_path": str(external_art),
                        "m5rt_options": {
                            "fake": True,
                            "fake_options": {"event_batch_size": 2, "step_delay_us": 1000},
                        },
                    },
                )
                assert put_resp.status_code == 200, put_resp.text
                saved = settings_mgr.get_settings(model_id)
                assert saved.external_engine == M5RT_ENGINE_ID
                assert saved.m5rt_artifact_path == str(external_art.resolve())

                # Passing a directory instead of a .omlxm5 file returns HTTP 400 (never 422)
                bad_resp = await client.put(
                    f"/admin/api/models/{model_id}/settings",
                    json={"m5rt_artifact_path": str(tmp_path)},
                )
                assert bad_resp.status_code == 400

                # Global settings reports m5rt availability
                gs_resp = await client.get("/admin/api/global-settings")
                assert gs_resp.status_code == 200
                m5_info = gs_resp.json()["external_engines"]["m5rt"]
                assert m5_info["available"] is True
                assert m5_info["path"] == str(LIVE_DAEMON_PATH.resolve())

                # 2) Non-streaming POST /v1/chat/completions (exercises count_chat_tokens!)
                chat_resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": "Hello M5 sidecar!"}],
                        "max_tokens": 6,
                        "temperature": 0.0,
                        "stream": False,
                    },
                )
                assert chat_resp.status_code == 200, chat_resp.text
                chat_data = chat_resp.json()
                assert chat_data["choices"][0]["finish_reason"] in ("length", "stop")
                assert chat_data["usage"]["prompt_tokens"] > 0
                assert chat_data["usage"]["completion_tokens"] == 6

                loaded_engine = pool._entries[model_id].engine
                assert isinstance(loaded_engine, M5SidecarEngine)
                assert loaded_engine.process_memory_bytes() > 0
                assert pool.external_process_memory_bytes() > 0

                # 3) Streaming POST /v1/chat/completions SSE
                sse_lines: list[str] = []
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": "Stream test"}],
                        "max_tokens": 5,
                        "temperature": 0.0,
                        "stream": True,
                    },
                ) as sse_resp:
                    assert sse_resp.status_code == 200
                    async for line in sse_resp.aiter_lines():
                        if line.startswith("data:"):
                            sse_lines.append(line)
                assert any("[DONE]" in ln for ln in sse_lines)
                assert len(sse_lines) >= 2

                # 4) POST /v1/completions (non-streaming + streaming)
                comp_resp = await client.post(
                    "/v1/completions",
                    json={
                        "model": model_id,
                        "prompt": "Plain completion prompt",
                        "max_tokens": 4,
                        "temperature": 0.0,
                    },
                )
                assert comp_resp.status_code == 200, comp_resp.text
                assert comp_resp.json()["usage"]["completion_tokens"] == 4

                comp_sse: list[str] = []
                async with client.stream(
                    "POST",
                    "/v1/completions",
                    json={
                        "model": model_id,
                        "prompt": "Stream plain completion",
                        "max_tokens": 4,
                        "temperature": 0.0,
                        "stream": True,
                    },
                ) as comp_stream_resp:
                    assert comp_stream_resp.status_code == 200
                    async for line in comp_stream_resp.aiter_lines():
                        if line.startswith("data:"):
                            comp_sse.append(line)
                assert any("[DONE]" in ln for ln in comp_sse)

                # 5) Anthropic POST /v1/messages (non-streaming + streaming)
                anth_resp = await client.post(
                    "/v1/messages",
                    json={
                        "model": model_id,
                        "max_tokens": 5,
                        "messages": [{"role": "user", "content": "Anthropic hello"}],
                    },
                )
                assert anth_resp.status_code == 200, anth_resp.text
                anth_data = anth_resp.json()
                assert anth_data["type"] == "message"
                assert anth_data["usage"]["input_tokens"] > 0
                assert anth_data["usage"]["output_tokens"] == 5

                anth_events: list[str] = []
                async with client.stream(
                    "POST",
                    "/v1/messages",
                    json={
                        "model": model_id,
                        "max_tokens": 4,
                        "stream": True,
                        "messages": [{"role": "user", "content": "Anthropic stream"}],
                    },
                ) as anth_stream_resp:
                    assert anth_stream_resp.status_code == 200
                    async for line in anth_stream_resp.aiter_lines():
                        if line.startswith("event:"):
                            anth_events.append(line)
                assert any("message_stop" in ev for ev in anth_events)

                # 5b) Mid-stream HTTP client disconnect cleanly aborts request without leaking active count:
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": "Slow stream abort"}],
                        "max_tokens": 64,
                        "temperature": 0.0,
                        "stream": True,
                    },
                ) as slow_resp:
                    assert slow_resp.status_code == 200
                    async for line in slow_resp.aiter_lines():
                        if line.startswith("data:") and "[DONE]" not in line:
                            break  # Disconnect right after first SSE chunk!
                await asyncio.sleep(0.1)
                assert loaded_engine.active_request_count() == 0
                # Same daemon stayed resident across chat, completions, and messages!
                assert pool._entries[model_id].engine is loaded_engine

                # 6) Crash recovery: SIGKILL child daemon -> runtime_failed_reason -> auto reload!
                old_pid = loaded_engine._process.pid
                os.killpg(old_pid, signal.SIGKILL)
                await asyncio.sleep(0.15)
                assert loaded_engine.runtime_failed_reason is not None

                recover_resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": "After crash"}],
                        "max_tokens": 4,
                        "temperature": 0.0,
                    },
                )
                assert recover_resp.status_code == 200, recover_resp.text
                reloaded_engine = pool._entries[model_id].engine
                assert isinstance(reloaded_engine, M5SidecarEngine)
                assert reloaded_engine is not loaded_engine
                assert reloaded_engine._process.pid != old_pid
                assert reloaded_engine.runtime_failed_reason is None
        finally:
            srv.app.dependency_overrides.pop(admin_routes.require_admin, None)
            await pool.shutdown()

    @pytest.mark.skipif(
        not (AFFINE_FIXTURE_DIR / "model.omlxm5").is_file()
        or not (LIVE_METALLIB_DIR / "omlx_m5_qwen_ar.metallib").is_file(),
        reason="2-layer Metal affine .omlxm5 fixture or metallib not present",
    )
    async def test_real_fastapi_server_e2e_with_native_metal_omlxm5_fixture(
        self, tmp_path, monkeypatch
    ):
        """End-to-end FastAPI HTTP test backed by real Metal GPU `.omlxm5` execution."""
        base_dir = tmp_path / "omlx-metal-base"
        models_root = tmp_path / "models"
        model_id = "Qwen3.5-2Layer-Metal-E2E"
        staged = models_root / model_id
        shutil.copytree(AFFINE_FIXTURE_DIR / "source", staged)
        artifact_file = AFFINE_FIXTURE_DIR / "model.omlxm5"

        global_settings = GlobalSettings(base_path=base_dir)
        global_settings.integrations.m5rt_path = str(LIVE_DAEMON_PATH)
        global_settings.auth.allow_unauthenticated_inference = True
        global_settings.auth.skip_api_key_verification = True

        settings_mgr = ModelSettingsManager(base_path=base_dir)
        sched_cfg = SchedulerConfig()
        sched_cfg.paged_ssd_cache_dir = None
        pool = EnginePool(scheduler_config=sched_cfg)
        pool._settings_manager = settings_mgr
        pool.discover_models([str(models_root)])

        monkeypatch.setattr("omlx.engine_pool.get_global_settings", lambda: global_settings)
        monkeypatch.setattr("omlx.settings._global_settings", global_settings, raising=False)
        monkeypatch.setattr(srv._server_state, "global_settings", global_settings, raising=False)
        monkeypatch.setattr(srv._server_state, "engine_pool", pool)
        monkeypatch.setattr(srv._server_state, "settings_manager", settings_mgr)
        monkeypatch.setattr(srv._server_state, "responses_store", ResponseStore())
        monkeypatch.setattr(srv._server_state, "api_key", None)

        monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
        monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: settings_mgr)
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: global_settings)
        monkeypatch.setattr(admin_routes, "_get_server_state", lambda: srv._server_state)
        srv.app.dependency_overrides[admin_routes.require_admin] = lambda: True

        transport = httpx.ASGITransport(app=srv.app)
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                put_resp = await client.put(
                    f"/admin/api/models/{model_id}/settings",
                    json={
                        "external_engine": M5RT_ENGINE_ID,
                        "m5rt_artifact_path": str(artifact_file),
                        "m5rt_options": {
                            "metallib_dir": str(LIVE_METALLIB_DIR),
                            "max_context": 1024,
                        },
                    },
                )
                assert put_resp.status_code == 200, put_resp.text

                chat_resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": "Metal GPU"}],
                        "max_tokens": 4,
                        "temperature": 0.0,
                    },
                )
                assert chat_resp.status_code == 200, chat_resp.text
                body = chat_resp.json()
                assert body["usage"]["prompt_tokens"] > 0
                assert body["usage"]["completion_tokens"] >= 1

                eng = pool._entries[model_id].engine
                assert isinstance(eng, M5SidecarEngine)
                assert eng._push_subscribed is True
                assert eng.get_stats()["backend_stats"]["backend_kind"] == "native"
                assert eng.process_memory_bytes() >= artifact_file.stat().st_size
        finally:
            srv.app.dependency_overrides.pop(admin_routes.require_admin, None)
            await pool.shutdown()


# -- 4. miv-v0.01 Settings Sync, Auto-Discovery, Coexistence & UI Unlock Tests -


class TestMivV001SettingsCoexistenceAndUI:
    def test_kv_dtypes_bytes_per_token_and_no_affine8_to_int8_mapping(self, tmp_path):
        assert m5_mod.VALID_KV_DTYPES == frozenset({"bf16", "affine4_g128"})
        assert "mtp" in m5_mod.VALID_SPEC_BACKENDS
        with pytest.raises(ValueError, match="kv_dtype"):
            m5_mod.validate_m5rt_options({"kv_dtype": "int8_g64"})
        assert m5_mod.validate_m5rt_options({"kv_dtype": "affine4_g128"}) == {
            "kv_dtype": "affine4_g128"
        }
        assert m5_mod.validate_m5rt_options({"speculative_backend": "mtp", "draft_tokens": 3}) == {
            "speculative_backend": "mtp",
            "draft_tokens": 3,
        }

        # Qwen3.8-27B geometry: 16 full layers, 4 KV heads, head_dim=256 -> 32768 values/tok
        qwen_cfg = {
            "layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
        }
        assert m5_mod._kv_bytes_per_token(qwen_cfg, "bf16") == 65536
        # Exact 0.5625 B/val (9/16) -> 18432 B/tok
        assert m5_mod._kv_bytes_per_token(qwen_cfg, "affine4_g128") == 18432

        # Standard ModelSettings KV sync: affine4 -> affine4_g128; affine8 must NOT map to int8_g64!
        model_dir = _write_test_model(tmp_path / "kv-model")
        # Raise native max_position_embeddings to 262144 to verify 32768 cap removal
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        cfg["text_config"]["max_position_embeddings"] = 262144
        (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

        fake_bin = _fake_daemon_script(tmp_path / "bin" / "mseries_backendd", "miv-v0.01", protocol=2)
        binary, err = find_m5rt_binary(str(fake_bin))
        assert binary is not None and err == ""
        assert binary.protocol_version == 2
        target = M5Target(
            binary=binary,
            artifact_path=model_dir / "model.omlxm5",
            metallib_dir=tmp_path / "bin",
            fake=False,
        )

        ms_aff4 = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            max_context_window=131072,
            turboquant_kv_enabled=True,
            turboquant_kv_scheme="affine4",
            turboquant_kv_bits=4,
        )
        opts_aff4 = m5_mod.build_m5rt_options(
            model_dir, target, ms_aff4, default_context=None, memory_budget_bytes=8 << 30
        )
        assert opts_aff4.kv_dtype == "affine4_g128"
        assert opts_aff4.max_context == 131072

        ms_aff8 = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            turboquant_kv_enabled=True,
            turboquant_kv_scheme="affine8",
            turboquant_kv_bits=8,
        )
        opts_aff8 = m5_mod.build_m5rt_options(
            model_dir, target, ms_aff8, default_context=4096, memory_budget_bytes=0
        )
        assert opts_aff8.kv_dtype == "bf16"  # Never "int8_g64"!

        # Explicit model windows are used by both the planner and the estimator.
        est_32k = m5rt_memory_estimate(
            model_dir, target, 262144, ModelSettings(max_context_window=32768)
        )
        est_256k = m5rt_memory_estimate(
            model_dir, target, 32768, ModelSettings(max_context_window=262144)
        )
        assert est_256k > est_32k

    def test_standard_settings_toml_and_global_concurrency_sync(self, tmp_path, monkeypatch):
        model_dir = _write_test_model(tmp_path / "sync-model")
        draft_dir = _write_test_model(tmp_path / "draft-model")
        fake_bin = _fake_daemon_script(tmp_path / "bin" / "mseries_backendd", "miv-v0.01", protocol=2)
        binary, _ = find_m5rt_binary(str(fake_bin))
        assert binary is not None
        target = M5Target(
            binary=binary,
            artifact_path=model_dir / "model.omlxm5",
            metallib_dir=tmp_path / "bin",
            draft_dir=draft_dir,
            draft_artifact_path=draft_dir / "model.omlxm5",
            fake=False,
        )

        monkeypatch.setenv("NINFER_METAL_CONFIG", str(tmp_path / "no-shared.toml"))
        monkeypatch.setenv("NINFER_METAL_NO_DEFAULT_CONFIG", "1")
        gs = GlobalSettings(base_path=tmp_path / "base")
        gs.scheduler.max_concurrent_requests = 4

        # 1) Stock MTP settings sync + global max_concurrent_requests default
        ms_mtp = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            mtp_enabled=True,
            mtp_fixed_depth=4,
        )
        opts_mtp = m5_mod.build_m5rt_options(
            model_dir,
            target,
            ms_mtp,
            default_context=4096,
            memory_budget_bytes=4 << 30,
            global_settings=gs,
        )
        assert opts_mtp.speculative_backend == "mtp"
        assert opts_mtp.draft_tokens == 4
        assert opts_mtp.max_concurrency == 4

        eng_mtp = M5SidecarEngine(
            model_name=str(model_dir),
            model_id="sync-mtp",
            target=target,
            log_dir=tmp_path / "logs",
            options=opts_mtp,
        )
        load_payload = eng_mtp._build_load_params()
        assert load_payload["speculative"] == {"backend": "mtp", "draft_tokens": 4}
        assert load_payload["path_overrides"]["mtp_fixed_depth"] == "on"
        adaptive = m5_mod.build_m5rt_options(
            model_dir, target, ModelSettings(mtp_enabled=True, mtp_adaptive_max_depth=4),
            default_context=4096, memory_budget_bytes=0, global_settings=gs,
        )
        assert adaptive.draft_tokens == 4
        assert adaptive.path_overrides["mtp_fixed_depth"] == "off"
        assert load_payload["memory_budget_bytes"] == (4 << 30)

        # 2) Stock DFlash settings sync (block_size=6 -> draft_tokens=5)
        ms_dflash = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            dflash_enabled=True,
            dflash_draft_model=str(draft_dir),
            dflash_block_size=6,
        )
        opts_df = m5_mod.build_m5rt_options(
            model_dir,
            target,
            ms_dflash,
            default_context=4096,
            memory_budget_bytes=0,
            global_settings=gs,
        )
        assert opts_df.speculative_backend == "dflash2"
        assert opts_df.draft_tokens == 5

        # 3) ninfer-metal.toml + external_engine_options override precedence
        toml_path = tmp_path / "ninfer-metal.toml"
        toml_path.write_text(
            "[engine]\n"
            "max_concurrency = 6\n"
            "rolling_prefix_cache = true\n"
            "prefix_cache_gb = 2.0\n"
            "sampling_contract = \"position_coupled_v1\"\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("NINFER_METAL_CONFIG", str(toml_path))
        ms_ext = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            external_engine_options={"ninfer-metal": {"max_concurrency": 3}},
        )
        opts_merged = m5_mod.build_m5rt_options(
            model_dir,
            target,
            ms_ext,
            default_context=4096,
            memory_budget_bytes=0,
            global_settings=gs,
        )
        assert opts_merged.max_concurrency == 3  # external_engine_options overrides TOML
        assert opts_merged.rolling_prefix_cache is True  # inherited from TOML
        assert opts_merged.prefix_cache_bytes == 2 * (1 << 30)  # prefix_cache_gb converted
        assert opts_merged.sampling_contract == "position_coupled_v1"

        # Runtime signature reflects standard setting changes
        assert m5_mod.m5rt_runtime_signature(ms_mtp, None, global_settings=gs) != (
            m5_mod.m5rt_runtime_signature(ms_dflash, None, global_settings=gs)
        )

    def test_provider_json_auto_discovery_and_o1_auto_router(self, tmp_path, monkeypatch):
        providers_root = tmp_path / "Providers"
        prov_dir = providers_root / "io.ninfer.metal" / "versions" / "miv-v0.01"
        bin_path = _fake_daemon_script(prov_dir / "bin" / "mseries_backendd", "miv-v0.01", protocol=2)
        metallib_dir = prov_dir / "metallib"
        metallib_dir.mkdir(parents=True)
        (metallib_dir / "omlx_m5_qwen_ar.metallib").write_bytes(b"MTLB")
        (prov_dir / "provider.json").write_text(
            json.dumps(
                {
                    "provider_id": "io.ninfer.metal",
                    "codename": "miv",
                    "display_version": "miv-v0.01",
                    "semver": "0.0.1",
                    "core_abi": 1,
                    "protocol_version": 2,
                    "entrypoint": "bin/mseries_backendd",
                    "metallib_dir": "metallib",
                }
            ),
            encoding="utf-8",
        )
        current_link = providers_root / "io.ninfer.metal" / "current"
        current_link.symlink_to(prov_dir)

        monkeypatch.delenv("OMLX_M5RT_BIN", raising=False)
        monkeypatch.delenv("NINFER_METAL_BIN", raising=False)
        monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(providers_root))

        binary, err = find_m5rt_binary(None)
        assert binary is not None and err == ""
        assert binary.path == bin_path.resolve()
        assert binary.version == "miv-v0.01"
        assert binary.protocol_version == 2
        assert binary.provider_id == "io.ninfer.metal"
        assert binary.codename == "miv"

        # Fast O(1) detect_adapted: zero subprocess spawn!
        adapted_dir = _write_test_model(tmp_path / "models" / "Adapted", with_artifact=True)
        unadapted_dir = _write_test_model(tmp_path / "models" / "StockOnly", with_artifact=False)
        pack_dir = _write_test_model(tmp_path / "models" / "Packed", with_artifact=False)
        (pack_dir / "weights.omlxm5").write_bytes(b"OMLXM5" + b"\x00" * 512)
        (pack_dir / "model_pack.json").write_text(
            json.dumps({"artifact_path": "weights.omlxm5"}), encoding="utf-8"
        )

        with patch.object(m5_mod.subprocess, "run", side_effect=AssertionError("must not spawn")):
            assert m5_mod.detect_adapted(adapted_dir).adapted is True
            assert m5_mod.detect_adapted(pack_dir).adapted is True
            assert m5_mod.detect_adapted(pack_dir).artifact_path == (pack_dir / "weights.omlxm5").resolve()
            assert m5_mod.detect_adapted(unadapted_dir).adapted is False

            gs = GlobalSettings(base_path=tmp_path / "base")
            monkeypatch.setattr("omlx.engine_pool.get_global_settings", lambda: gs)
            pool = EnginePool()
            pool._entries["Adapted"] = EngineEntry(
                model_id="Adapted",
                model_path=str(adapted_dir),
                model_type="llm",
                engine_type="batched",
                estimated_size=1024,
            )
            pool._entries["StockOnly"] = EngineEntry(
                model_id="StockOnly",
                model_path=str(unadapted_dir),
                model_type="llm",
                engine_type="batched",
                estimated_size=1024,
            )

            # Adapted model auto-routes to m5rt; unadapted returns None (100% stock oMLX)
            assert pool.external_engine_id("Adapted", ModelSettings()) == M5RT_ENGINE_ID
            assert pool.external_engine_id("StockOnly", ModelSettings()) is None
            # Explicit "stock" opts adapted model back into stock MLX
            assert pool.external_engine_id("Adapted", ModelSettings(external_engine="stock")) is None

    async def test_coexistence_rpc_trim_cache_scheduler_hints_and_v1_noop(self, tmp_path):
        from omlx.process_memory_enforcer import ProcessMemoryEnforcer

        # 1) v2 daemon: memory_budget_bytes in load + trim_cache + set_scheduler_hint
        sock_v2 = Path("/tmp") / f"m5-coex-v2-{os.getpid()}.sock"
        srv_v2 = _MockUdsSidecarServer(sock_v2, support_push=True, support_v2_coexistence=True)
        await srv_v2.start()

        model_dir = _write_test_model(tmp_path / "coex-model")
        fake_bin = _fake_daemon_script(tmp_path / "bin" / "mseries_backendd", "miv-v0.01", protocol=2)
        binary, _ = find_m5rt_binary(str(fake_bin))
        assert binary is not None
        target = M5Target(
            binary=binary,
            artifact_path=model_dir / "model.omlxm5",
            metallib_dir=tmp_path / "bin",
            fake=False,
        )
        engine = M5SidecarEngine(
            model_name=str(model_dir),
            model_id="coex-m5",
            target=target,
            log_dir=tmp_path / "logs",
            options=M5Options(max_context=4096, memory_budget_bytes=512 * 1024 * 1024),
        )
        await asyncio.to_thread(engine._load_tokenizer)
        engine._socket_path = sock_v2
        engine._process = SimpleNamespace(pid=os.getpid(), poll=lambda: None, returncode=None)  # type: ignore[assignment]
        engine._ctrl_reader, engine._ctrl_writer = await engine._open_uds()
        engine._hello_info = await engine._rpc("hello", timeout=5.0)
        engine._last_stats = await engine._rpc("load", engine._build_load_params(), timeout=5.0)
        try:
            assert srv_v2.load_params[-1]["memory_budget_bytes"] == 512 * 1024 * 1024

            # Hook into EnginePool + ProcessMemoryEnforcer for coexistence test
            pool = EnginePool()
            mlx_engine = MagicMock()
            mlx_engine.has_active_requests.return_value = True
            pool._entries["stock-mlx"] = EngineEntry(
                model_id="stock-mlx",
                model_path=str(model_dir),
                model_type="llm",
                engine_type="batched",
                estimated_size=1024,
                engine=mlx_engine,
                in_use=1,
            )
            pool._entries["coex-m5"] = EngineEntry(
                model_id="coex-m5",
                model_path=str(model_dir),
                model_type="llm",
                engine_type="batched",
                estimated_size=1024,
                engine=engine,
                in_use=1,
            )

            # Both active -> gpu_contended=True
            assert await pool.sync_external_scheduler_hints() is True
            assert srv_v2.scheduler_hints == [True]
            # Redundant sync is deduplicated
            assert await pool.sync_external_scheduler_hints() is True
            assert srv_v2.scheduler_hints == [True]
            # Stock MLX finishes -> gpu_contended=False
            pool._entries["stock-mlx"].in_use = 0
            mlx_engine.has_active_requests.return_value = False
            assert await pool.sync_external_scheduler_hints() is False
            assert srv_v2.scheduler_hints == [True, False]

            # ProcessMemoryEnforcer soft pressure calls trim_cache BEFORE evicting coexisting model!
            enforcer = ProcessMemoryEnforcer(
                engine_pool=pool,
                soft_threshold=0.80,
                hard_threshold=0.95,
            )
            enforcer._running = True
            ceiling = 200 * 1024 * 1024  # soft = 160 MiB
            usage_samples = iter([180 * 1024 * 1024, 150 * 1024 * 1024])
            enforcer._get_hard_limit_bytes = lambda: ceiling  # type: ignore[method-assign]
            enforcer._get_abort_limit_bytes = lambda: ceiling  # type: ignore[method-assign]
            enforcer._current_usage_bytes = lambda: next(usage_samples, 150 * 1024 * 1024)  # type: ignore[method-assign]
            enforcer._propagate_memory_limit = lambda: None  # type: ignore[method-assign]
            enforcer._request_scheduler_cache_reclaim = lambda _b: 0  # type: ignore[method-assign]
            enforcer._walk_store_cache_caps = lambda: None  # type: ignore[method-assign]

            await enforcer._check_and_enforce()
            assert len(srv_v2.trim_calls) == 1
            assert srv_v2.trim_calls[0] == 20 * 1024 * 1024
            # Both coexisting models remain loaded (no eviction needed!)
            assert pool._entries["coex-m5"].engine is engine
            assert pool._entries["stock-mlx"].engine is mlx_engine
        finally:
            engine._process = None
            await engine.stop()
            assert engine.process_memory_bytes() == 0
            await srv_v2.close()

        # 2) Legacy v1 daemon: trim_cache & set_scheduler_hint gracefully no-op without failing runtime
        sock_v1 = Path("/tmp") / f"m5-coex-v1-{os.getpid()}.sock"
        srv_v1 = _MockUdsSidecarServer(sock_v1, support_push=False, support_v2_coexistence=False)
        await srv_v1.start()
        eng_v1 = M5SidecarEngine(
            model_name=str(model_dir),
            model_id="coex-v1",
            target=target,
            log_dir=tmp_path / "logs",
            options=M5Options(max_context=4096, memory_budget_bytes=0),
        )
        await asyncio.to_thread(eng_v1._load_tokenizer)
        eng_v1._socket_path = sock_v1
        eng_v1._process = SimpleNamespace(pid=os.getpid(), poll=lambda: None, returncode=None)  # type: ignore[assignment]
        eng_v1._ctrl_reader, eng_v1._ctrl_writer = await eng_v1._open_uds()
        try:
            trim_res = await eng_v1.trim_cache(16 * 1024 * 1024)
            assert trim_res["freed_bytes"] == 0 and trim_res["supported"] is False
            assert await eng_v1.set_scheduler_hint(gpu_contended=True) is False
            assert eng_v1.runtime_failed_reason is None
        finally:
            eng_v1._process = None
            await eng_v1.stop()
            await srv_v1.close()

    def test_admin_api_exposes_supported_standard_settings_and_unlocks_m5rt_mtp(self, tmp_path):
        model_dir = _write_test_model(tmp_path / "models" / "Qwen3.8-MTP")
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        cfg["text_config"]["mtp_num_hidden_layers"] = 1
        (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"mtp.fc.weight": "model.safetensors"}}),
            encoding="utf-8",
        )

        # Splash blocks MTP (upstream behavior), while m5rt unlocks MTP!
        splash_ok, splash_reason = admin_routes._mtp_compat_for_model(
            {"id": "Qwen3.8-MTP", "model_path": str(model_dir)},
            external_engine="splash",
        )
        assert splash_ok is False
        assert "external engine" in splash_reason

        m5_ok, m5_reason = admin_routes._mtp_compat_for_model(
            {"id": "Qwen3.8-MTP", "model_path": str(model_dir)},
            external_engine=M5RT_ENGINE_ID,
        )
        assert m5_ok is True, m5_reason

        # _external_engines_for_model exposes supported_standard_settings on m5rt row,
        # while omitting it on splash row to keep upstream Splash contract intact.
        pool = MagicMock()
        pool.external_engine_compatibility.side_effect = (
            lambda eid, mid, ms: (eid == M5RT_ENGINE_ID, "", None)
        )
        rows = admin_routes._external_engines_for_model(
            pool,
            {"id": "Qwen3.8-MTP", "engine_type": "batched", "is_helper": False},
            ModelSettings(external_engine=M5RT_ENGINE_ID),
        )
        by_id = {r["id"]: r for r in rows}
        assert "supported_standard_settings" not in by_id["splash"]
        assert "mtp_enabled" in by_id[M5RT_ENGINE_ID]["supported_standard_settings"]
        assert "dflash_enabled" in by_id[M5RT_ENGINE_ID]["supported_standard_settings"]
        assert "turboquant_kv_enabled" in by_id[M5RT_ENGINE_ID]["supported_standard_settings"]
        assert "max_context_window" in by_id[M5RT_ENGINE_ID]["supported_standard_settings"]


# -- 5. Path B Multi-Microkernel Parallel Sidecar Discovery & Coexistence -----


def _install_mock_spi_provider(
    providers_root: Path,
    *,
    provider_id: str,
    engine_id: str,
    display_name: str,
    version: str = "0.2.0",
    protocol_version: int = 2,
    supported_topologies: list[str] | None = None,
    supported_model_types: list[str] | None = None,
    artifact_extensions: list[str] | None = None,
    supported_standard_settings: list[str] | None = None,
    use_current_symlink: bool = True,
    executable: bool = True,
    with_metallib: bool = False,
    binary_name: str = "llama_metal_backendd",
    version_output: str | None = None,
) -> Path:
    """Create a realistic Path B SPI provider bundle under `providers_root`."""
    if use_current_symlink:
        ver_dir = providers_root / provider_id / "versions" / f"v{version}"
        ver_dir.mkdir(parents=True, exist_ok=True)
        bundle_dir = ver_dir
        current_link = providers_root / provider_id / "current"
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        current_link.symlink_to(ver_dir)
    else:
        bundle_dir = providers_root / provider_id
        bundle_dir.mkdir(parents=True, exist_ok=True)

    bin_path = bundle_dir / "bin" / binary_name
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    ver_line = version_output or f"{binary_name} {version} protocol={protocol_version}"
    bin_path.write_text(
        f'#!/bin/sh\nif [ "$1" = "--version" ]; then echo "{ver_line}"; exit 0; fi\nexit 0\n',
        encoding="utf-8",
    )
    bin_path.chmod(0o755 if executable else 0o644)

    manifest: dict[str, Any] = {
        "schema": "omlx-provider-manifest-v2",
        "provider_id": provider_id,
        "engine_id": engine_id,
        "display_name": display_name,
        "display_version": version,
        "semver": version,
        "core_abi": 1,
        "protocol_version": protocol_version,
        "entrypoint": f"bin/{binary_name}",
        "supported_topologies": supported_topologies or [],
        "supported_model_types": supported_model_types or [],
        "artifact_extensions": artifact_extensions or [".omlxm5"],
        "supported_standard_settings": supported_standard_settings
        or ["max_context_window", "turboquant_kv_enabled", "turboquant_kv_scheme"],
        "kv_formats": ["bf16", "affine4_g128"],
    }
    if with_metallib:
        mdir = bundle_dir / "metallib"
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "llama_nax.metallib").write_bytes(b"MTLB")
        manifest["metallib_dir"] = "metallib"

    (bundle_dir / "provider.json").write_text(json.dumps(manifest), encoding="utf-8")
    return bundle_dir


def _write_custom_model(
    model_dir: Path,
    *,
    model_type: str,
    artifact_name: str | None = "model.omlxm5",
    pack_manifest: dict[str, Any] | None = None,
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": model_type,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "vocab_size": 512,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "max_position_embeddings": 8192,
        "eos_token_id": [2],
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"\x00" * 512)
    if artifact_name:
        (model_dir / artifact_name).write_bytes(b"OMLXM5" + b"\x00" * 4096)
    if pack_manifest is not None:
        (model_dir / "model_pack.json").write_text(
            json.dumps(pack_manifest), encoding="utf-8"
        )
    return model_dir


class TestPathBMultiMicrokernelSidecars:
    def test_generic_spi_providers_discovery_validation_and_builtin_skip(
        self, tmp_path, monkeypatch
    ):
        providers_root = tmp_path / "Providers"

        # Built-in io.ninfer.metal & legacy com.omlx.mseries MUST be skipped by plugin scanner
        _install_mock_spi_provider(
            providers_root,
            provider_id="io.ninfer.metal",
            engine_id="m5rt",
            display_name="ninfer-metal",
            supported_model_types=["qwen3_5"],
            binary_name="mseries_backendd",
            version_output="mseries-backendd miv-v0.01 protocol=2",
        )
        _install_mock_spi_provider(
            providers_root,
            provider_id="com.omlx.mseries",
            engine_id="m5rt",
            display_name="legacy",
            binary_name="mseries_backendd",
            version_output="mseries-backendd 0.1.0 protocol=1",
        )

        # Valid Path B provider via current/ symlink
        _install_mock_spi_provider(
            providers_root,
            provider_id="io.community.llama-metal",
            engine_id="llama-metal",
            display_name="Llama Metal 4",
            version="0.3.0",
            protocol_version=2,
            supported_topologies=["llama3_gqa_v1"],
            supported_model_types=["llama"],
            artifact_extensions=[".omlxm5"],
            with_metallib=True,
        )

        # Valid Path B provider via direct */provider.json fallback (no current/ symlink)
        _install_mock_spi_provider(
            providers_root,
            provider_id="io.community.deepseek-mla",
            engine_id="deepseek-mla",
            display_name="DeepSeek MLA",
            version="0.1.0",
            protocol_version=2,
            supported_topologies=["deepseek_mla_v1"],
            supported_model_types=["deepseek_v3"],
            artifact_extensions=[".dskm5"],
            use_current_symlink=False,
            with_metallib=False,
            binary_name="deepseek_mla_daemon",
            version_output="deepseek-mla-daemon 0.1.0",
        )

        # Invalid protocol_version (99) and non-executable binary MUST be rejected
        _install_mock_spi_provider(
            providers_root,
            provider_id="io.community.bad-proto",
            engine_id="bad-proto",
            display_name="Bad Proto",
            protocol_version=99,
        )
        _install_mock_spi_provider(
            providers_root,
            provider_id="io.community.not-exec",
            engine_id="not-exec",
            display_name="Not Exec",
            executable=False,
        )

        monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(providers_root))
        monkeypatch.setattr(m5_mod.importlib.metadata, "entry_points", lambda group=None: [])

        # Discovery must run in O(1) without spawning subprocesses!
        with patch.object(m5_mod.subprocess, "run", side_effect=AssertionError("must not spawn")):
            plugins = discover_external_engine_plugins()

        assert set(plugins.keys()) == {"llama-metal", "deepseek-mla"}
        assert "m5rt" not in plugins
        assert "bad-proto" not in plugins
        assert "not-exec" not in plugins

        engines = get_external_engines()
        assert engines["llama-metal"] == "Llama Metal 4"
        assert engines["deepseek-mla"] == "DeepSeek MLA"
        assert "max_context_window" in m5_mod.get_external_engine_supported_settings("llama-metal")

        # Generic --version output works (with and without explicit `protocol=2` token)
        gs = GlobalSettings(base_path=tmp_path / "base")
        llama_status = plugins["llama-metal"].status(gs, None)
        assert llama_status["available"] is True
        assert llama_status["provider_id"] == "io.community.llama-metal"
        assert llama_status["version"] == "0.3.0"
        assert llama_status["protocol_version"] == 2

        ds_status = plugins["deepseek-mla"].status(gs, None)
        assert ds_status["available"] is True
        assert ds_status["version"] == "0.1.0"
        assert ds_status["protocol_version"] == 2

    def test_non_hijack_routing_between_m5rt_and_path_b_providers(
        self, tmp_path, monkeypatch
    ):
        providers_root = tmp_path / "Providers"
        _install_mock_spi_provider(
            providers_root,
            provider_id="io.ninfer.metal",
            engine_id="m5rt",
            display_name="ninfer-metal",
            supported_topologies=["qwen35_hybrid_dense_v1"],
            supported_model_types=["qwen3_5", "qwen3_6", "qwen3_8"],
            binary_name="mseries_backendd",
            version_output="mseries-backendd miv-v0.01 protocol=2",
        )
        _install_mock_spi_provider(
            providers_root,
            provider_id="io.community.llama-metal",
            engine_id="llama-metal",
            display_name="Llama Metal 4",
            supported_topologies=["llama3_gqa_v1"],
            supported_model_types=["llama"],
            artifact_extensions=[".omlxm5"],
        )
        _install_mock_spi_provider(
            providers_root,
            provider_id="io.community.qwen-alt",
            engine_id="qwen-alt",
            display_name="Qwen Alt Experimental",
            supported_topologies=["qwen_alt_v1"],
            supported_model_types=["qwen3_5"],
            artifact_extensions=[".omlxm5"],
        )

        monkeypatch.delenv("OMLX_M5RT_BIN", raising=False)
        monkeypatch.delenv("NINFER_METAL_BIN", raising=False)
        monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(providers_root))
        monkeypatch.setattr(m5_mod.importlib.metadata, "entry_points", lambda group=None: [])

        # 1) Llama model with .omlxm5 (no model_pack.json): m5rt MUST NOT hijack!
        llama_dir = _write_custom_model(tmp_path / "models" / "Llama3-8B", model_type="llama")
        # 2) Qwen model with explicit io.ninfer.metal provider_id -> routes to m5rt even when qwen-alt exists
        qwen_m5_dir = _write_custom_model(
            tmp_path / "models" / "Qwen3.5-Mainline",
            model_type="qwen3_5",
            pack_manifest={"provider_id": "io.ninfer.metal", "artifact_path": "model.omlxm5"},
        )
        # 3) Qwen model with explicit io.community.qwen-alt provider_id -> routes to qwen-alt (no m5rt hijack)
        qwen_alt_dir = _write_custom_model(
            tmp_path / "models" / "Qwen3.5-Alt",
            model_type="qwen3_5",
            pack_manifest={"provider_id": "io.community.qwen-alt", "artifact_path": "model.omlxm5"},
        )
        # 4) Unadapted stock Llama model (no .omlxm5) -> routes to None (stock MLX)
        stock_llama_dir = _write_custom_model(
            tmp_path / "models" / "Llama3-Stock", model_type="llama", artifact_name=None
        )
        # 5) Pure Qwen3.8 model (not in qwen-alt's supported_model_types) -> routes to built-in m5rt
        qwen38_dir = _write_custom_model(tmp_path / "models" / "Qwen3.8-27B", model_type="qwen3_8")

        gs = GlobalSettings(base_path=tmp_path / "base")
        monkeypatch.setattr("omlx.engine_pool.get_global_settings", lambda: gs)
        pool = EnginePool()
        for mid, mdir in (
            ("Llama3-8B", llama_dir),
            ("Qwen3.5-Mainline", qwen_m5_dir),
            ("Qwen3.5-Alt", qwen_alt_dir),
            ("Llama3-Stock", stock_llama_dir),
            ("Qwen3.8-27B", qwen38_dir),
        ):
            pool._entries[mid] = EngineEntry(
                model_id=mid,
                model_path=str(mdir),
                model_type="llm",
                engine_type="batched",
                estimated_size=1024,
            )

        with patch.object(m5_mod.subprocess, "run", side_effect=AssertionError("must not spawn")):
            assert m5_mod.detect_adapted(llama_dir).adapted is False
            assert m5_mod.detect_adapted(qwen_alt_dir).adapted is False
            assert m5_mod.detect_adapted(qwen_m5_dir).adapted is True
            assert m5_mod.detect_adapted(qwen38_dir).adapted is True

            assert pool.external_engine_id("Llama3-8B", ModelSettings()) == "llama-metal"
            assert pool.external_engine_id("Qwen3.5-Mainline", ModelSettings()) == M5RT_ENGINE_ID
            assert pool.external_engine_id("Qwen3.5-Alt", ModelSettings()) == "qwen-alt"
            assert pool.external_engine_id("Qwen3.8-27B", ModelSettings()) == M5RT_ENGINE_ID
            assert pool.external_engine_id("Llama3-Stock", ModelSettings()) is None
            assert pool.external_engine_id("Llama3-8B", ModelSettings(external_engine="stock")) is None

    def test_uma_budget_forwarding_per_engine_isolation_and_optional_metallib(
        self, tmp_path, monkeypatch
    ):
        providers_root = tmp_path / "Providers"
        _install_mock_spi_provider(
            providers_root,
            provider_id="io.community.llama-metal",
            engine_id="llama-metal",
            display_name="Llama Metal 4",
            supported_topologies=["llama3_gqa_v1"],
            supported_model_types=["llama"],
            artifact_extensions=[".omlxm5"],
            with_metallib=False,  # No omlx_m5_qwen_ar.metallib required!
        )
        monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(providers_root))
        monkeypatch.setattr(m5_mod.importlib.metadata, "entry_points", lambda group=None: [])

        llama_dir = _write_custom_model(tmp_path / "models" / "Llama3-Budget", model_type="llama")
        gs = GlobalSettings(base_path=tmp_path / "base")
        monkeypatch.setattr("omlx.engine_pool.get_global_settings", lambda: gs)

        pool = EnginePool()
        pool._entries["Llama3-Budget"] = EngineEntry(
            model_id="Llama3-Budget",
            model_path=str(llama_dir),
            model_type="llm",
            engine_type="batched",
            estimated_size=1024,
        )

        # Simulate UMA admission limit = 512 MiB with 384 MiB already used -> 128 MiB budget room
        pool._splash_admission_target = lambda: 512 * 1024 * 1024  # type: ignore[method-assign]
        pool._current_model_memory = 384 * 1024 * 1024

        ms = ModelSettings(
            max_context_window=4096,
            external_engine_options={
                "io.community.llama-metal": {"max_concurrency": 2, "prefix_cache_gb": 1.0},
                "llama-metal": {"rolling_prefix_cache": True},
            },
        )
        engine = pool._external_engine_for(pool._entries["Llama3-Budget"], ms, "batched")
        assert isinstance(engine, M5SidecarEngine)
        assert engine.external_engine_id == "llama-metal"
        assert engine.reserved_bytes() == 128 * 1024 * 1024
        assert engine._options.max_concurrency == 2
        assert engine._options.rolling_prefix_cache is True
        assert engine._log_path.parent.name == "llama-metal"

        # get_stats reports engine_type == engine_id, and load params omit qwen_ar_metallib
        stats = engine.get_stats()
        assert stats["engine_type"] == "llama-metal"
        assert stats["provider_id"] == "io.community.llama-metal"
        load_params = engine._build_load_params()
        assert "qwen_ar_metallib" not in load_params["path_overrides"]
        assert load_params["memory_budget_bytes"] == 128 * 1024 * 1024

        # Per-engine socket prefix stays well below macOS 104-byte sun_path limit
        sock_path = m5_mod._short_socket_path("llama-metal")
        assert "llama-metal" in str(sock_path)
        assert len(str(sock_path).encode("utf-8")) < 104

        # m5rt_status filters out Path B sidecars; Path B status reports its own loaded model
        pool._entries["Llama3-Budget"].engine = engine
        m5_stat = m5_mod.m5rt_status(gs, pool)
        assert all(m["id"] != "Llama3-Budget" for m in m5_stat["models"])
        llama_plugin = m5_mod.get_external_engine_plugin("llama-metal")
        assert llama_plugin is not None and callable(llama_plugin.status)
        llama_stat = llama_plugin.status(gs, pool)
        assert [m["id"] for m in llama_stat["models"]] == ["Llama3-Budget"]

        # When zero UMA room remains, EnginePool raises ModelLoadingError cleanly
        pool._current_model_memory = 512 * 1024 * 1024
        with pytest.raises(ModelLoadingError, match="memory limit"):
            pool._external_engine_for(pool._entries["Llama3-Budget"], ms, "batched")

    async def test_gpu_contended_true_for_two_parallel_external_sidecars_without_mlx(
        self, tmp_path
    ):
        sock_a = Path("/tmp") / f"m5-par-a-{os.getpid()}.sock"
        sock_b = Path("/tmp") / f"m5-par-b-{os.getpid()}.sock"
        srv_a = _MockUdsSidecarServer(sock_a, support_push=True, support_v2_coexistence=True)
        srv_b = _MockUdsSidecarServer(sock_b, support_push=True, support_v2_coexistence=True)
        await srv_a.start()
        await srv_b.start()

        qwen_dir = _write_test_model(tmp_path / "qwen-par")
        llama_dir = _write_custom_model(tmp_path / "llama-par", model_type="llama")
        bin_a = _fake_daemon_script(tmp_path / "bin_a" / "mseries_backendd", "miv-v0.01", protocol=2)
        bin_b = _fake_daemon_script(
            tmp_path / "bin_b" / "llama_metal_backendd", "0.2.0", protocol=2
        )
        m5_bin, _ = find_m5rt_binary(str(bin_a))
        llama_manifest = {
            "provider_id": "io.community.llama-metal",
            "engine_id": "llama-metal",
            "display_name": "Llama Metal",
            "protocol_version": 2,
        }
        llama_bin, _ = m5_mod.inspect_m5rt_binary(
            bin_b, auto_detected=True, provider_manifest=llama_manifest, allow_generic=True
        )
        assert m5_bin is not None and llama_bin is not None

        eng_a = M5SidecarEngine(
            model_name=str(qwen_dir),
            model_id="qwen-par",
            target=M5Target(binary=m5_bin, artifact_path=qwen_dir / "model.omlxm5", metallib_dir=tmp_path / "bin_a"),
            log_dir=tmp_path / "logs" / "m5rt",
            options=M5Options(max_context=4096, memory_budget_bytes=256 * 1024 * 1024),
            engine_id=M5RT_ENGINE_ID,
            provider_id="io.ninfer.metal",
        )
        eng_b = M5SidecarEngine(
            model_name=str(llama_dir),
            model_id="llama-par",
            target=M5Target(binary=llama_bin, artifact_path=llama_dir / "model.omlxm5", metallib_dir=None),
            log_dir=tmp_path / "logs" / "llama-metal",
            options=M5Options(max_context=4096, memory_budget_bytes=256 * 1024 * 1024),
            engine_id="llama-metal",
            provider_id="io.community.llama-metal",
            manifest=llama_manifest,
        )

        for eng, sock in ((eng_a, sock_a), (eng_b, sock_b)):
            await asyncio.to_thread(eng._load_tokenizer)
            eng._socket_path = sock
            eng._process = SimpleNamespace(pid=os.getpid(), poll=lambda: None, returncode=None)  # type: ignore[assignment]
            eng._ctrl_reader, eng._ctrl_writer = await eng._open_uds()
            eng._hello_info = await eng._rpc("hello", timeout=5.0)
            eng._last_stats = await eng._rpc("load", eng._build_load_params(), timeout=5.0)

        try:
            pool = EnginePool()
            # NO stock MLX model in the pool — only two parallel external sidecars!
            pool._entries["qwen-par"] = EngineEntry(
                model_id="qwen-par",
                model_path=str(qwen_dir),
                model_type="llm",
                engine_type="batched",
                estimated_size=1024,
                engine=eng_a,
                in_use=1,
            )
            pool._entries["llama-par"] = EngineEntry(
                model_id="llama-par",
                model_path=str(llama_dir),
                model_type="llm",
                engine_type="batched",
                estimated_size=1024,
                engine=eng_b,
                in_use=1,
            )

            # Both sidecars active -> gpu_contended=True sent to BOTH sidecars!
            assert await pool.sync_external_scheduler_hints() is True
            assert srv_a.scheduler_hints == [True]
            assert srv_b.scheduler_hints == [True]

            # One sidecar finishes -> single active sidecar gets gpu_contended=False
            pool._entries["llama-par"].in_use = 0
            assert await pool.sync_external_scheduler_hints() is False
            assert srv_a.scheduler_hints == [True, False]
            assert srv_b.scheduler_hints == [True, False]
        finally:
            eng_a._process = None
            eng_b._process = None
            await eng_a.stop()
            await eng_b.stop()
            await srv_a.close()
            await srv_b.close()

    def test_auto_routed_plugin_runtime_settings_signature_detects_option_changes(
        self, tmp_path, monkeypatch
    ):
        providers_root = tmp_path / "Providers"
        _install_mock_spi_provider(
            providers_root,
            provider_id="io.community.llama-metal",
            engine_id="llama-metal",
            display_name="Llama Metal 4",
            supported_topologies=["llama3_gqa_v1"],
            supported_model_types=["llama"],
            artifact_extensions=[".omlxm5"],
        )
        monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(providers_root))
        monkeypatch.setattr(m5_mod.importlib.metadata, "entry_points", lambda group=None: [])

        llama_dir = _write_custom_model(tmp_path / "models" / "Llama3-Sig", model_type="llama")
        gs = GlobalSettings(base_path=tmp_path / "base")
        monkeypatch.setattr("omlx.engine_pool.get_global_settings", lambda: gs)

        pool = EnginePool()
        pool._entries["Llama3-Sig"] = EngineEntry(
            model_id="Llama3-Sig",
            model_path=str(llama_dir),
            model_type="llm",
            engine_type="batched",
            estimated_size=1024,
        )

        # Auto-routed (external_engine=None) -> signature MUST still invoke plugin.runtime_signature!
        ms_a = ModelSettings(
            external_engine=None,
            external_engine_options={"llama-metal": {"max_concurrency": 1}},
        )
        ms_b = ModelSettings(
            external_engine=None,
            external_engine_options={"llama-metal": {"max_concurrency": 4}},
        )
        assert pool.external_engine_id("Llama3-Sig", ms_a) == "llama-metal"
        sig_a = pool._engine_runtime_signature("Llama3-Sig", ms_a)
        sig_b = pool._engine_runtime_signature("Llama3-Sig", ms_b)
        assert sig_a is not None and sig_b is not None
        assert sig_a != sig_b
        assert any(k == "external_plugin_signature" for k, _ in sig_a)

    def test_global_settings_external_engines_includes_microkernels_list(
        self, tmp_path, monkeypatch
    ):
        providers_root = tmp_path / "Providers"
        _install_mock_spi_provider(
            providers_root,
            provider_id="io.ninfer.metal",
            engine_id="m5rt",
            display_name="miv Core Backend (ninfer-metal)",
            version="miv-v0.01",
            protocol_version=2,
            supported_topologies=["qwen35_hybrid_dense_v1", "qwen36_hybrid_moe256_k8_shared_v1"],
            supported_model_types=["qwen3_5", "qwen3_6_moe"],
            binary_name="mseries_backendd",
            version_output="mseries-backendd miv-v0.01 protocol=2",
        )
        _install_mock_spi_provider(
            providers_root,
            provider_id="io.community.llama-metal",
            engine_id="llama-metal",
            display_name="Llama Metal 4 (Path B)",
            version="0.3.0",
            protocol_version=2,
            supported_topologies=["llama3_gqa_v1"],
            supported_model_types=["llama"],
        )
        monkeypatch.delenv("OMLX_M5RT_BIN", raising=False)
        monkeypatch.delenv("NINFER_METAL_BIN", raising=False)
        monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(providers_root))
        monkeypatch.setattr(m5_mod.importlib.metadata, "entry_points", lambda group=None: [])
        monkeypatch.setattr(admin_routes, "_get_engine_pool", None)

        gs = GlobalSettings(base_path=tmp_path / "base")
        status = admin_routes._external_engines_status(gs)

        assert "splash" in status
        assert status["m5rt"]["label"] == "miv Core Backend"
        assert status["m5rt"]["available"] is True
        assert "microkernels" in status
        by_provider = {m["provider_id"]: m for m in status["microkernels"]}
        assert "io.ninfer.metal" in by_provider
        assert "io.community.llama-metal" in by_provider

        miv = by_provider["io.ninfer.metal"]
        assert miv["engine_id"] == "m5rt"
        assert miv["display_name"] == "miv Core Backend"
        assert miv["version"] == "miv-v0.01"
        assert miv["path_tier"] == "Path A"
        assert miv["protocol_version"] == 2
        assert miv["installed"] is True
        assert miv["binary_path"].endswith("mseries_backendd")
        assert "qwen3_5" in miv["supported_model_types"]

        llama = by_provider["io.community.llama-metal"]
        assert llama["engine_id"] == "llama-metal"
        assert llama["display_name"] == "Llama Metal 4 (Path B)"
        assert llama["version"] == "0.3.0"
        assert llama["path_tier"] == "Path B"
        assert llama["protocol_version"] == 2
        assert llama["installed"] is True
        assert llama["supported_model_types"] == ["llama"]

    @pytest.mark.asyncio
    async def test_admin_models_exposes_active_engine_and_preserves_auto_route_under_stock_override(
        self, tmp_path, monkeypatch
    ):
        providers_root = tmp_path / "Providers"
        _install_mock_spi_provider(
            providers_root,
            provider_id="io.ninfer.metal",
            engine_id="m5rt",
            display_name="miv Core Backend (ninfer-metal)",
            version="miv-v0.01",
            protocol_version=2,
            supported_topologies=["qwen35_hybrid_dense_v1"],
            supported_model_types=["qwen3_5"],
            binary_name="mseries_backendd",
            version_output="mseries-backendd miv-v0.01 protocol=2",
        )
        _install_mock_spi_provider(
            providers_root,
            provider_id="io.community.llama-metal",
            engine_id="llama-metal",
            display_name="Llama Metal 4 (Path B)",
            version="0.3.0",
            protocol_version=2,
            supported_topologies=["llama3_gqa_v1"],
            supported_model_types=["llama"],
        )
        monkeypatch.delenv("OMLX_M5RT_BIN", raising=False)
        monkeypatch.delenv("NINFER_METAL_BIN", raising=False)
        monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(providers_root))
        monkeypatch.setattr(m5_mod.importlib.metadata, "entry_points", lambda group=None: [])

        models_dir = tmp_path / "models"
        qwen_auto = _write_custom_model(
            models_dir / "Qwen3.5-Auto",
            model_type="qwen3_5",
            pack_manifest={"provider_id": "io.ninfer.metal", "artifact_path": "model.omlxm5"},
        )
        qwen_stock = _write_custom_model(
            models_dir / "Qwen3.5-StockOverride",
            model_type="qwen3_5",
            pack_manifest={"provider_id": "io.ninfer.metal", "artifact_path": "model.omlxm5"},
        )
        llama_path_b = _write_custom_model(
            models_dir / "Llama3-PathB",
            model_type="llama",
        )
        stock_plain = _write_custom_model(
            models_dir / "Plain-MLX",
            model_type="llama",
            artifact_name=None,
        )

        gs = GlobalSettings(base_path=tmp_path / "base")
        gs.model.model_dir = str(models_dir)
        gs.model.model_dirs = [str(models_dir)]
        monkeypatch.setattr("omlx.engine_pool.get_global_settings", lambda: gs)
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)

        pool = EnginePool()
        for mid, mdir in [
            ("Qwen3.5-Auto", qwen_auto),
            ("Qwen3.5-StockOverride", qwen_stock),
            ("Llama3-PathB", llama_path_b),
            ("Plain-MLX", stock_plain),
        ]:
            pool._entries[mid] = EngineEntry(
                model_id=mid,
                model_path=str(mdir),
                model_type="llm",
                engine_type="batched",
                estimated_size=1024,
            )

        settings_mgr = ModelSettingsManager(tmp_path / "model_settings.json")
        settings_mgr.set_settings(
            "Qwen3.5-StockOverride", ModelSettings(external_engine="stock")
        )
        monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
        monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: settings_mgr)
        monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)

        resp = await admin_routes.list_models(is_admin=True)
        by_id = {m["id"]: m for m in resp["models"]}

        # Auto-routed miv Core Backend model
        assert by_id["Qwen3.5-Auto"]["auto_external_engine"] == "m5rt"
        assert by_id["Qwen3.5-Auto"]["active_engine_id"] == "m5rt"
        assert by_id["Qwen3.5-Auto"]["active_engine_label"] == "miv Core Backend"

        # Adapted model overridden to "stock" MUST still report auto_external_engine="m5rt"
        assert by_id["Qwen3.5-StockOverride"]["auto_external_engine"] == "m5rt"
        assert by_id["Qwen3.5-StockOverride"]["active_engine_id"] == "mlx"
        assert by_id["Qwen3.5-StockOverride"]["active_engine_label"] == "MLX"

        # Auto-routed Path B microkernel model
        assert by_id["Llama3-PathB"]["auto_external_engine"] == "llama-metal"
        assert by_id["Llama3-PathB"]["active_engine_id"] == "llama-metal"
        assert by_id["Llama3-PathB"]["active_engine_label"] == "Llama Metal 4 (Path B)"

        # Unadapted model stays 100% Stock MLX
        assert by_id["Plain-MLX"]["auto_external_engine"] is None
        assert by_id["Plain-MLX"]["active_engine_id"] == "mlx"
        assert by_id["Plain-MLX"]["active_engine_label"] == "MLX"

    def test_shared_default_ninfer_metal_toml_reuse_and_precedence(
        self, tmp_path, monkeypatch
    ):
        """m5_sidecar.py reuses omlx-extend/config/ninfer-metal.toml with exact
        auto-discovery order, speculative.backend='auto', and UI > [sidecar.models] > shared TOML precedence.
        """
        fake_home = tmp_path / "home"
        work_dir = tmp_path / "work"
        fake_home.mkdir()
        work_dir.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.delenv("NINFER_METAL_CONFIG", raising=False)
        monkeypatch.delenv("NINFER_METAL_NO_DEFAULT_CONFIG", raising=False)
        monkeypatch.delenv("OMLX_PROVIDERS_DIR", raising=False)
        monkeypatch.chdir(work_dir)

        # 1) Auto-discovers repo omlx-extend/config/ninfer-metal.toml
        discovered = m5_mod.discover_ninfer_metal_toml_path()
        assert discovered is not None
        assert discovered.resolve() == (
            REPO_ROOT / "omlx-extend" / "config" / "ninfer-metal.toml"
        ).resolve()

        # Model with native max_position_embeddings=262144 so shared 65536 default applies cleanly
        model_dir = _write_test_model(tmp_path / "models" / "Qwen3.8-SharedDefault")
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        cfg["text_config"]["max_position_embeddings"] = 262144
        (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

        fake_bin = _fake_daemon_script(
            tmp_path / "bin" / "mseries_backendd", "miv-v0.01", protocol=2
        )
        binary, err = find_m5rt_binary(str(fake_bin))
        assert binary is not None and err == ""
        target = M5Target(
            binary=binary,
            artifact_path=model_dir / "model.omlxm5",
            metallib_dir=tmp_path / "bin",
            fake=False,
        )

        # Plain ModelSettings() inherits all shared defaults from omlx-extend/config/ninfer-metal.toml
        opts = m5_mod.build_m5rt_options(
            model_dir,
            target,
            ModelSettings(),
            default_context=32768,
            memory_budget_bytes=8 << 30,
            model_id="Qwen3.8-SharedDefault",
        )
        assert opts.max_concurrency == 4
        assert opts.max_context == 262144
        assert opts.kv_dtype == "affine4_g128"
        assert opts.rolling_prefix_cache is True
        assert opts.prefix_cache_bytes == 4 * (1 << 30)
        assert opts.sampling_contract == "position_coupled_v1"
        assert opts.prefill_linear_mode == "bf16"
        assert opts.prefill_attention_mode == "exact_nax"
        assert opts.prefill_sparse_minference is False
        assert opts.prefill_cache_blend is False
        assert opts.prefill_ane_offload is False
        # speculative.backend = "auto" on plain model resolves cleanly to "none" without error
        assert opts.speculative_backend == "none"
        assert opts.draft_tokens == 0

        # 2) speculative.backend = "auto" detects companion mtp.omlxm5 and dflash2/ cleanly
        mtp_model = _write_test_model(tmp_path / "models" / "Qwen3.8-WithMTP")
        (mtp_model / "mtp.omlxm5").write_bytes(b"OMLXM5" + b"\x00" * 512)
        opts_mtp_auto = m5_mod.build_m5rt_options(
            mtp_model,
            target,
            ModelSettings(),
            default_context=4096,
            memory_budget_bytes=0,
        )
        assert opts_mtp_auto.speculative_backend == "mtp"
        assert opts_mtp_auto.draft_tokens == 0
        assert opts_mtp_auto.path_overrides["mtp_artifact"] == str(
            (mtp_model / "mtp.omlxm5").resolve()
        )

        dflash_model = _write_test_model(tmp_path / "models" / "Qwen3.8-WithDFlash")
        _write_test_model(dflash_model / "dflash2")
        metallib_dir = tmp_path / "bin" / "metallib"
        metallib_dir.mkdir(exist_ok=True)
        (metallib_dir / "omlx_m5_qwen_ar.metallib").write_bytes(b"MTLB")
        df_target, df_err = resolve_m5rt_target(
            dflash_model, str(fake_bin), ModelSettings()
        )
        assert df_target is not None and df_err == ""
        assert df_target.draft_dir == (dflash_model / "dflash2").resolve()
        assert df_target.draft_artifact_path == (
            dflash_model / "dflash2" / "model.omlxm5"
        ).resolve()
        opts_df_auto = m5_mod.build_m5rt_options(
            dflash_model,
            df_target,
            ModelSettings(),
            default_context=4096,
            memory_budget_bytes=0,
        )
        assert opts_df_auto.speculative_backend == "dflash2"
        assert opts_df_auto.draft_tokens == 0

        # 3) Precedence: explicit UI settings > [sidecar.models."<name>"] > [engine] in shared TOML
        custom_toml = tmp_path / "layered.toml"
        custom_toml.write_text(
            "[engine]\n"
            "max_concurrency = 4\n"
            "max_context = 65536\n"
            'kv_dtype = "affine4_g128"\n'
            "rolling_prefix_cache = true\n"
            "prefix_cache_gb = 4.0\n\n"
            "[speculative]\n"
            'backend = "auto"\n'
            "draft_tokens = 0\n\n"
            '[sidecar.models."Qwen3.8-SharedDefault"]\n'
            "max_concurrency = 2\n"
            "max_context = 32768\n"
            'kv_dtype = "bf16"\n'
            "prefix_cache_gb = 6.0\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("NINFER_METAL_CONFIG", str(custom_toml))

        # [sidecar.models."Qwen3.8-SharedDefault"] beats shared [engine]
        opts_sidecar_model = m5_mod.build_m5rt_options(
            model_dir,
            target,
            ModelSettings(),
            default_context=None,
            memory_budget_bytes=0,
            model_id="Qwen3.8-SharedDefault",
        )
        assert opts_sidecar_model.max_concurrency == 2
        assert opts_sidecar_model.max_context == 32768
        assert opts_sidecar_model.kv_dtype == "bf16"
        assert opts_sidecar_model.prefix_cache_bytes == 6 * (1 << 30)

        # Explicit oMLX per-model UI settings beat [sidecar.models."..."] AND shared [engine]!
        ms_ui = ModelSettings(
            max_context_window=131072,
            turboquant_kv_enabled=True,
            turboquant_kv_scheme="affine4",
            turboquant_kv_bits=4,
            mtp_enabled=True,
            mtp_fixed_depth=4,
        )
        opts_ui = m5_mod.build_m5rt_options(
            model_dir,
            target,
            ms_ui,
            default_context=None,
            memory_budget_bytes=0,
            model_id="Qwen3.8-SharedDefault",
        )
        assert opts_ui.max_context == 131072
        assert opts_ui.kv_dtype == "affine4_g128"
        assert opts_ui.speculative_backend == "mtp"
        assert opts_ui.draft_tokens == 4
        assert opts_ui.max_concurrency == 2  # still inherited from [sidecar.models]

        # 4) Auto-discovery precedence chain:
        #    $NINFER_METAL_CONFIG > ./ninfer-metal.toml > ~/.config/... > Providers/.../config/... > repo
        monkeypatch.delenv("NINFER_METAL_CONFIG", raising=False)
        prov_cfg = (
            fake_home
            / "Library/Application Support/oMLX/Providers/io.ninfer.metal/current/config/ninfer-metal.toml"
        )
        prov_cfg.parent.mkdir(parents=True)
        prov_cfg.write_text("[engine]\nmax_concurrency = 5\n", encoding="utf-8")
        assert m5_mod.discover_ninfer_metal_toml_path() == prov_cfg

        user_cfg = fake_home / ".config/ninfer-metal/ninfer-metal.toml"
        user_cfg.parent.mkdir(parents=True)
        user_cfg.write_text("[engine]\nmax_concurrency = 6\n", encoding="utf-8")
        assert m5_mod.discover_ninfer_metal_toml_path() == user_cfg

        cwd_cfg = work_dir / "ninfer-metal.toml"
        cwd_cfg.write_text("[engine]\nmax_concurrency = 7\n", encoding="utf-8")
        assert m5_mod.discover_ninfer_metal_toml_path() == cwd_cfg

        monkeypatch.setenv("NINFER_METAL_CONFIG", str(custom_toml))
        assert m5_mod.discover_ninfer_metal_toml_path() == custom_toml


class TestMivDiscoveryAutoRepackAndFoundryConverter:
    def test_pure_miv_and_legacy_omlxm5_discovery_with_stale_index_and_hf_cache(
        self, tmp_path: Path
    ):
        from omlx.model_discovery import discover_models

        models_root = tmp_path / "models"
        models_root.mkdir()

        # 1) Pure .miv local folder that still retains model.safetensors.index.json
        pure_dir = models_root / "Qwen3.8-27B-PureMiv"
        pure_dir.mkdir()
        (pure_dir / "config.json").write_text(
            json.dumps({"model_type": "qwen3_8", "hidden_size": 4096}),
            encoding="utf-8",
        )
        (pure_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (pure_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
                        "lm_head.weight": "model-00004-of-00004.safetensors",
                    }
                }
            ),
            encoding="utf-8",
        )
        miv_payload = b"MIV00001" + (b"\x00" * 4096)
        (pure_dir / "model.miv").write_bytes(miv_payload)

        # 2) Legacy pure .omlxm5 HF Hub cache snapshot remains backward-compatible
        hf_snap = (
            models_root
            / "models--scoltzero--Qwen3.5-9B-miv"
            / "snapshots"
            / "abc123456789"
        )
        hf_snap.mkdir(parents=True)
        (models_root / "models--scoltzero--Qwen3.5-9B-miv" / "refs").mkdir()
        (models_root / "models--scoltzero--Qwen3.5-9B-miv" / "refs" / "main").write_text(
            "abc123456789\n", encoding="utf-8"
        )
        (hf_snap / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5", "hidden_size": 2048}),
            encoding="utf-8",
        )
        legacy_payload = b"OMLXM5A1" + (b"\x00" * 4096)
        (hf_snap / "model.omlxm5").write_bytes(legacy_payload)

        discovered = discover_models(models_root)
        assert "Qwen3.8-27B-PureMiv" in discovered
        assert discovered["Qwen3.8-27B-PureMiv"].model_format == "miv"
        assert discovered["Qwen3.8-27B-PureMiv"].estimated_size >= len(miv_payload)

        assert "scoltzero--Qwen3.5-9B-miv" in discovered
        assert discovered["scoltzero--Qwen3.5-9B-miv"].model_format == "omlxm5"

    def test_can_convert_to_miv_colocated_and_bf16_guard(
        self, tmp_path: Path, monkeypatch
    ):
        prov_root = tmp_path / "provider"
        (prov_root / "bin").mkdir(parents=True)
        (prov_root / "metallib").mkdir(parents=True)
        (prov_root / "tools").mkdir(parents=True)
        fake_bin = prov_root / "bin" / "mseries_backendd"
        fake_bin.write_text(
            '#!/bin/sh\necho "mseries-backendd miv-v0.01 (codename=miv semver=0.0.1 core_abi=1) protocol=2"\n',
            encoding="utf-8",
        )
        fake_bin.chmod(0o755)
        (prov_root / "metallib" / "omlx_m5_qwen_ar.metallib").write_bytes(b"MTLB")
        fake_tool = prov_root / "tools" / "m5_artifact.py"
        fake_tool.write_text(
            "import pathlib, sys\n"
            "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
            "out.write_bytes(b'MIV00001' + b'\\x00' * 128)\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("NINFER_METAL_ARTIFACT_TOOL", raising=False)

        # Unquantized BF16 Qwen checkpoint must NOT claim auto-conversion (stays on Stock MLX)
        bf16_dir = tmp_path / "Qwen3.5-BF16"
        bf16_dir.mkdir()
        (bf16_dir / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5"}), encoding="utf-8"
        )
        (bf16_dir / "model.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
        bf16_ok, _ = m5_mod.can_convert_to_miv(bf16_dir, str(fake_bin))
        assert bf16_ok is False
        assert m5_mod.can_auto_convert_m5rt_fast(bf16_dir, str(fake_bin)) is False

        # Quantized 4-bit affine Qwen checkpoint converts in-place to model.miv alongside model.safetensors
        qwen_dir = tmp_path / "Qwen3.5-4bit"
        qwen_dir.mkdir()
        (qwen_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
                }
            ),
            encoding="utf-8",
        )
        orig_st_bytes = b"\x08\x00\x00\x00\x00\x00\x00\x00{}      "
        (qwen_dir / "model.safetensors").write_bytes(orig_st_bytes)

        ok, reason = m5_mod.can_convert_to_miv(qwen_dir, str(fake_bin))
        assert ok is True and reason == ""
        fmt_before = m5_mod.detect_model_weight_format_info(
            qwen_dir, active_external_engine="m5rt", configured_m5rt_binary=str(fake_bin)
        )
        assert fmt_before["weight_format"] == "safetensors"
        assert fmt_before["pending_auto_repack"] is True
        assert fmt_before["repack_target_format"] == "miv"

        res = m5_mod.convert_model_to_miv(
            qwen_dir, source_repo="mlx-community/Qwen3.5-9B-4bit", configured_binary=str(fake_bin)
        )
        assert res["already_existed"] is False
        assert res["artifact_format"] == "miv"
        assert (qwen_dir / "model.miv").is_file()
        # Original .safetensors remains intact right next to model.miv!
        assert (qwen_dir / "model.safetensors").read_bytes() == orig_st_bytes
        assert m5_mod.detect_adapted(qwen_dir).adapted is True

        fmt_after = m5_mod.detect_model_weight_format_info(
            qwen_dir, active_external_engine="m5rt", configured_m5rt_binary=str(fake_bin)
        )
        assert fmt_after["weight_format"] == "miv"
        assert fmt_after["has_miv_weights"] is True
        assert fmt_after["pending_auto_repack"] is False

        # When overridden back to Stock MLX, active weight_format reports "safetensors"
        fmt_stock = m5_mod.detect_model_weight_format_info(
            qwen_dir, active_external_engine="stock", configured_m5rt_binary=str(fake_bin)
        )
        assert fmt_stock["weight_format"] == "safetensors"
        assert fmt_stock["has_miv_weights"] is True

        ok_after, _ = m5_mod.can_convert_to_miv(qwen_dir, str(fake_bin))
        assert ok_after is False

    async def test_auto_repack_on_first_load_and_foundry_path_b_converter(
        self, tmp_path: Path, monkeypatch
    ):
        # Set up both Path A (io.ninfer.metal -> .miv) and Path B (io.community.llama-metal -> .llama-m5)
        providers_root = tmp_path / "Providers"
        miv_ver = providers_root / "io.ninfer.metal" / "versions" / "miv-v0.01"
        (miv_ver / "bin").mkdir(parents=True)
        (miv_ver / "metallib").mkdir(parents=True)
        (miv_ver / "tools").mkdir(parents=True)
        miv_sock_srv = _MockUdsSidecarServer(
            Path("/tmp") / f"m5-autorepack-{os.getpid()}.sock",
            support_push=True,
            support_v2_coexistence=True,
            token_plan=[[65, 66]],
        )
        await miv_sock_srv.start()
        try:
            miv_bin = _fake_daemon_script(miv_ver / "bin" / "mseries_backendd", "miv-v0.01", protocol=2)
            (miv_ver / "metallib" / "omlx_m5_qwen_ar.metallib").write_bytes(b"MTLB")
            (miv_ver / "tools" / "m5_artifact.py").write_text(
                "import pathlib, sys\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
                "out.write_bytes(b'MIV00001' + b'\\x00' * 256)\n",
                encoding="utf-8",
            )
            (miv_ver / "provider.json").write_text(
                json.dumps(
                    {
                        "provider_id": "io.ninfer.metal",
                        "engine_id": "m5rt",
                        "protocol_version": 2,
                        "entrypoint": "bin/mseries_backendd",
                        "metallib_dir": "metallib",
                        "supported_model_types": ["qwen3_5"],
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
            (providers_root / "io.ninfer.metal" / "current").symlink_to(miv_ver)

            # Path B provider with Microkernel Foundry `converter` contract
            llama_ver = providers_root / "io.community.llama-metal" / "versions" / "0.2.0"
            (llama_ver / "bin").mkdir(parents=True)
            (llama_ver / "tools").mkdir(parents=True)
            _fake_daemon_script(llama_ver / "bin" / "llama_backendd", "0.2.0", protocol=2)
            (llama_ver / "tools" / "repack_llama.py").write_text(
                "import pathlib, sys\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
                "out.write_bytes(b'LLAMAM5!' + b'\\x00' * 256)\n",
                encoding="utf-8",
            )
            (llama_ver / "provider.json").write_text(
                json.dumps(
                    {
                        "provider_id": "io.community.llama-metal",
                        "engine_id": "llama-metal",
                        "display_name": "Community Llama Metal 4",
                        "protocol_version": 2,
                        "entrypoint": "bin/llama_backendd",
                        "supported_model_types": ["llama"],
                        "artifact_extensions": [".llama-m5"],
                        "converter": {
                            "entrypoint": "tools/repack_llama.py",
                            "subcommand": "convert",
                            "output_filename": "model.llama-m5",
                            "artifact_extension": ".llama-m5",
                            "supported_quant_modes": ["affine"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (providers_root / "io.community.llama-metal" / "current").symlink_to(llama_ver)

            monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(providers_root))
            monkeypatch.setattr(m5_mod, "_DEFAULT_ARTIFACT_CACHE_RELPATH", tmp_path / "isolated_repo_cache")
            monkeypatch.delenv("OMLX_M5RT_ARTIFACT_DIR", raising=False)
            monkeypatch.delenv("OMLX_M5RT_BIN", raising=False)
            monkeypatch.delenv("NINFER_METAL_BIN", raising=False)
            monkeypatch.setattr(m5_mod.importlib.metadata, "entry_points", lambda group=None: [])

            # Create unrepacked quantized Qwen & Llama checkpoints
            qwen_unrepacked = tmp_path / "models" / "Qwen3.5-Mirror"
            qwen_unrepacked.mkdir(parents=True)
            (qwen_unrepacked / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "vocab_size": 512,
                        "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
                    }
                ),
                encoding="utf-8",
            )
            (qwen_unrepacked / "model.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")

            llama_unrepacked = tmp_path / "models" / "Llama3-Mirror"
            llama_unrepacked.mkdir(parents=True)
            (llama_unrepacked / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "llama",
                        "vocab_size": 512,
                        "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
                    }
                ),
                encoding="utf-8",
            )
            (llama_unrepacked / "model.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")

            gs = GlobalSettings(base_path=tmp_path / "base")
            monkeypatch.setattr("omlx.engine_pool.get_global_settings", lambda: gs)
            pool = EnginePool()
            for mid, mdir in (("Qwen3.5-Mirror", qwen_unrepacked), ("Llama3-Mirror", llama_unrepacked)):
                pool._entries[mid] = EngineEntry(
                    model_id=mid,
                    model_path=str(mdir),
                    model_type="llm",
                    engine_type="batched",
                    estimated_size=4096,
                )

            # Fast O(1) auto-route BEFORE conversion (zero subprocess!)
            with patch.object(m5_mod.subprocess, "run", side_effect=AssertionError("must not spawn")):
                assert pool.external_engine_id("Qwen3.5-Mirror", ModelSettings()) == "m5rt"
                assert pool.external_engine_id("Llama3-Mirror", ModelSettings()) == "llama-metal"
                assert pool.external_engine_id("Qwen3.5-Mirror", ModelSettings(external_engine="stock")) is None

            # Verify m5rt_target sets auto_convert_output = model.miv when not yet converted
            target, err = pool.m5rt_target("Qwen3.5-Mirror", ModelSettings())
            assert err == "" and target is not None
            assert target.artifact_path is None
            assert target.auto_convert_output == qwen_unrepacked / "model.miv"

            # Verify first Load (`engine.start()`) auto-repacks model.miv in-place and passes it to load RPC!
            eng = pool._external_engine_for(pool._entries["Qwen3.5-Mirror"], ModelSettings(), "batched")
            assert isinstance(eng, M5SidecarEngine)
            monkeypatch.setattr(m5_mod, "_short_socket_path", lambda _eid=None: miv_sock_srv.socket_path)
            real_popen = m5_mod.subprocess.Popen
            monkeypatch.setattr(
                m5_mod.subprocess,
                "Popen",
                lambda cmd, *a, **kw: (
                    SimpleNamespace(pid=os.getpid(), poll=lambda: None, returncode=None, wait=lambda: 0)
                    if isinstance(cmd, list) and "m5rt-watchdog" in cmd
                    else real_popen(cmd, *a, **kw)
                ),
            )
            await eng.start()
            try:
                assert (qwen_unrepacked / "model.miv").is_file()
                assert (qwen_unrepacked / "model.safetensors").is_file()
                assert Path(miv_sock_srv.load_params[-1]["path_overrides"]["artifact"]).name == "model.miv"
            finally:
                eng._process = None
                await eng.stop()

            # Also verify Path B Foundry converter produces model.llama-m5 alongside model.safetensors
            llama_plugin = m5_mod.get_external_engine_plugin("llama-metal")
            assert llama_plugin is not None and llama_plugin.convert_model is not None
            llama_res = llama_plugin.convert_model(llama_unrepacked, source_repo="meta-llama/Llama-3-8B-4bit")
            assert Path(llama_res["artifact_path"]).name == "model.llama-m5"
            assert (llama_unrepacked / "model.llama-m5").is_file()
            assert (llama_unrepacked / "model.safetensors").is_file()
        finally:
            await miv_sock_srv.close()


def _write_test_peft_lora(adapter_dir: Path, *, hidden: int = 128, rank: int = 8) -> Path:
    import struct as _struct

    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": rank,
                "lora_alpha": 16,
                "base_model_name_or_path": "Qwen/Qwen3.5-Test",
                "target_modules": ["q_proj", "v_proj"],
            }
        ),
        encoding="utf-8",
    )
    tensors = {}
    offset = 0
    for layer_idx in (0, 1):
        for proj in ("q_proj", "v_proj"):
            for suffix, shape in (("lora_A.weight", [rank, hidden]), ("lora_B.weight", [hidden, rank])):
                nbytes = shape[0] * shape[1] * 2
                tensors[f"base_model.model.model.layers.{layer_idx}.self_attn.{proj}.{suffix}"] = {
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [offset, offset + nbytes],
                }
                offset += nbytes
    header_bytes = json.dumps(tensors).encode("utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(
        _struct.pack("<Q", len(header_bytes)) + header_bytes + (b"\x00" * offset)
    )
    return adapter_dir


class TestSidecarLoRAIntegration:
    """SPI v2 LoRA forwarding, admin auth, reload triggers, and UMA memory accounting."""

    @pytest.mark.asyncio
    async def test_lora_forwarding_modes_reload_and_memory_enforcer_accounting(
        self, tmp_path: Path, monkeypatch
    ):
        from omlx.lora_adapters import estimate_sidecar_lora_bytes
        from omlx.process_memory_enforcer import ProcessMemoryEnforcer
        from omlx.settings import SubKeyEntry

        model_dir = _write_test_model(tmp_path / "models" / "Qwen3.5-LoRA")
        adapter_a = _write_test_peft_lora(tmp_path / "models" / "lora-alpha", hidden=128, rank=8)
        adapter_b = _write_test_peft_lora(tmp_path / "external_loras" / "lora-beta", hidden=128, rank=4)

        fake_bin = _fake_daemon_script(tmp_path / "bin" / "mseries_backendd", version="miv-v0.01", protocol=2)
        metallib_dir = tmp_path / "metallib"
        metallib_dir.mkdir()
        (metallib_dir / "omlx_m5_qwen_ar.metallib").write_bytes(b"MTLB")

        # Shared ninfer-metal.toml sets default lora_adapter = "" and lora_mode = "merge"
        toml_cfg = tmp_path / "ninfer-metal.toml"
        toml_cfg.write_text(
            '[engine]\nlora_adapter = ""\nlora_mode = "merge"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("NINFER_METAL_CONFIG", str(toml_cfg))
        monkeypatch.setenv("OMLX_M5RT_BIN", str(fake_bin))
        monkeypatch.setenv("OMLX_M5RT_METALLIB_DIR", str(metallib_dir))

        gs = GlobalSettings(base_path=tmp_path / "base")
        gs.auth.api_key = "admin-secret-key"
        gs.auth.sub_keys = [SubKeyEntry(key="sub-inference-only-key", name="sub1")]
        monkeypatch.setattr("omlx.engine_pool.get_global_settings", lambda: gs)

        pool = EnginePool()
        pool.discover_models([str(tmp_path / "models")])
        assert "lora-alpha" in pool.lora_adapters()
        settings_mgr = ModelSettingsManager(tmp_path / "base")
        pool._settings_manager = settings_mgr

        # 1) Memory estimate & budget incorporate LoRA UMA footprint (merge > bypass > base)
        target, reason = pool.m5rt_target("Qwen3.5-LoRA", ModelSettings())
        assert reason == "" and target is not None

        est_base = pool._entry_runtime_resident_size(
            pool._entries["Qwen3.5-LoRA"],
            ModelSettings(external_engine="m5rt"),
        )
        est_bypass = pool._entry_runtime_resident_size(
            pool._entries["Qwen3.5-LoRA"],
            ModelSettings(
                external_engine="m5rt",
                lora_adapter="lora-alpha",
                external_engine_options={"ninfer-metal": {"lora_mode": "bypass"}},
            ),
        )
        est_merge = pool._entry_runtime_resident_size(
            pool._entries["Qwen3.5-LoRA"],
            ModelSettings(
                external_engine="m5rt",
                lora_adapter="lora-alpha",
                external_engine_options={"ninfer-metal": {"lora_mode": "merge"}},
            ),
        )
        expected_bypass_lora = estimate_sidecar_lora_bytes(adapter_a, lora_mode="bypass")
        expected_merge_lora = estimate_sidecar_lora_bytes(adapter_a, lora_mode="merge")
        assert expected_bypass_lora > 0
        assert expected_merge_lora > expected_bypass_lora
        assert est_bypass == est_base + expected_bypass_lora
        assert est_merge == est_base + expected_merge_lora

        # 2) Start sidecar against _MockUdsSidecarServer: verify resolved absolute path + lora_mode
        sock_path = Path(f"/tmp/omlx-lora-{os.getpid()}.sock")
        uds_srv = _MockUdsSidecarServer(sock_path)
        await uds_srv.start()
        try:
            ms_bypass = ModelSettings(
                external_engine="ninfer-metal",
                lora_adapter="lora-alpha",
                external_engine_options={"ninfer-metal": {"lora_mode": "bypass"}},
            )
            eng = pool._external_engine_for(
                pool._entries["Qwen3.5-LoRA"], ms_bypass, "batched"
            )
            assert isinstance(eng, M5SidecarEngine)
            assert eng._options.lora_adapter == str(adapter_a.resolve())
            assert eng._options.lora_mode == "bypass"
            assert eng._options.memory_budget_bytes == est_bypass

            monkeypatch.setattr(m5_mod, "_short_socket_path", lambda _eid=None: sock_path)
            fake_proc = SimpleNamespace(pid=os.getpid(), poll=lambda: None, returncode=None, wait=lambda: 0)
            monkeypatch.setattr(m5_mod.subprocess, "Popen", lambda *a, **kw: fake_proc)
            monkeypatch.setattr(m5_mod, "list_process_group_pids", lambda _pid: [os.getpid()])
            monkeypatch.setattr(m5_mod, "get_phys_footprint", lambda _pid: 16 * 1024 * 1024)

            # Set a smaller sidecar base footprint so mmap weights + reported lora_buffer_bytes sets floor
            uds_srv.resident_bytes = 8 * 1024 * 1024
            await eng.start()
            try:
                assert uds_srv.load_params[-1]["lora_adapter"] == str(adapter_a.resolve())
                assert uds_srv.load_params[-1]["lora_mode"] == "bypass"

                stats = eng.get_stats()
                assert stats["lora_adapter"] == str(adapter_a.resolve())
                assert stats["lora_mode"] == "bypass"
                assert stats["lora_buffer_bytes"] == 4 * 1024 * 1024
                assert stats["lora_adapted_layers"] == 2

                # ProcessMemoryEnforcer includes external sidecar weights + lora_buffer_bytes
                pool._entries["Qwen3.5-LoRA"].engine = eng
                enforcer = ProcessMemoryEnforcer(engine_pool=pool)
                monkeypatch.setattr("omlx.process_memory_enforcer.get_phys_footprint", lambda: 32 * 1024 * 1024)
                monkeypatch.setattr("omlx.process_memory_enforcer.mx.get_active_memory", lambda: 0)
                monkeypatch.setattr("omlx.process_memory_enforcer.mx.get_cache_memory", lambda: 0)
                monkeypatch.setattr("omlx.process_memory_enforcer.unreleased_graphics_bytes", lambda *_a, **_kw: 0)
                expected_sidecar_mem = max(
                    16 * 1024 * 1024,
                    max(8 * 1024 * 1024, eng._weights_bytes() + 4 * 1024 * 1024),
                )
                assert eng.process_memory_bytes() == expected_sidecar_mem
                assert enforcer._current_usage_bytes() == 32 * 1024 * 1024 + expected_sidecar_mem

                # When sidecar footprint drops below mmap weights + 16 MiB LoRA, floor dominates
                monkeypatch.setattr(m5_mod, "get_phys_footprint", lambda _pid: 2 * 1024 * 1024)

                # 3) reload_lora RPC updates active adapter, mode, and lora_buffer_bytes (4 MiB -> 16 MiB)
                reload_res = await eng.reload_lora(adapter_b, lora_mode="merge")
                assert reload_res["lora_adapter"] == str(adapter_b.resolve())
                assert reload_res["lora_mode"] == "merge"
                assert eng.lora_memory_bytes() == 16 * 1024 * 1024
                assert eng.process_memory_bytes() == eng._weights_bytes() + 16 * 1024 * 1024
                assert enforcer._current_usage_bytes() == 32 * 1024 * 1024 + eng._weights_bytes() + 16 * 1024 * 1024

                # Fallback when daemon does not implement reload_lora -> re-calls load RPC
                uds_srv.support_reload_lora = False
                prev_loads = len(uds_srv.load_params)
                await eng.reload_lora("lora-alpha", lora_mode="bypass", lora_adapters=pool.lora_adapters())
                assert len(uds_srv.load_params) == prev_loads + 1
                assert uds_srv.load_params[-1]["lora_adapter"] == str(adapter_a.resolve())
                assert uds_srv.load_params[-1]["lora_mode"] == "bypass"
            finally:
                pool._entries["Qwen3.5-LoRA"].engine = None
                eng._process = None
                await eng.stop()
        finally:
            await uds_srv.close()

        # 4) Admin HTTP permission checks: main admin key allowed, sub-key and unauth rejected (401)
        from omlx.admin import auth as admin_auth

        monkeypatch.setattr(srv._server_state, "global_settings", gs)
        monkeypatch.setattr(srv._server_state, "api_key", gs.auth.api_key)
        monkeypatch.setattr(srv._server_state, "bind_host", "127.0.0.1")
        monkeypatch.setattr(srv._server_state, "engine_pool", pool)
        monkeypatch.setattr(srv._server_state, "settings_manager", settings_mgr)
        monkeypatch.setattr(admin_auth, "_get_global_settings", lambda: gs)
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)
        monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
        monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: settings_mgr)
        monkeypatch.setattr(admin_routes, "_get_server_state", lambda: srv._server_state)

        transport = httpx.ASGITransport(app=srv.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Unauthenticated -> 401
            r_unauth = await client.put(
                "/admin/api/models/Qwen3.5-LoRA/settings",
                json={"lora_adapter": "lora-alpha"},
            )
            assert r_unauth.status_code == 401

            # Inference sub-key -> 401 on both PUT settings and GET lora-adapters
            sub_headers = {"Authorization": "Bearer sub-inference-only-key"}
            r_sub_put = await client.put(
                "/admin/api/models/Qwen3.5-LoRA/settings",
                headers=sub_headers,
                json={"lora_adapter": "lora-alpha"},
            )
            assert r_sub_put.status_code == 401
            r_sub_get = await client.get(
                "/admin/api/models/Qwen3.5-LoRA/lora-adapters",
                headers=sub_headers,
            )
            assert r_sub_get.status_code == 401

            # Main admin key -> 200 for setting lora_adapter + lora_mode (merge & bypass)
            admin_headers = {"Authorization": "Bearer admin-secret-key"}
            for opt_key, mode in (("ninfer-metal", "merge"), ("io.ninfer.metal", "bypass"), ("m5rt", "merge")):
                r_ok = await client.put(
                    "/admin/api/models/Qwen3.5-LoRA/settings",
                    headers=admin_headers,
                    json={
                        "external_engine": "ninfer-metal",
                        "lora_adapter": "lora-alpha",
                        "external_engine_options": {opt_key: {"lora_mode": mode}},
                    },
                )
                assert r_ok.status_code == 200, r_ok.text
                body = r_ok.json()
                assert body["settings"]["external_engine"] == "m5rt"
                assert body["settings"]["lora_adapter"] == "lora-alpha"
                assert body["settings"]["external_engine_options"][opt_key]["lora_mode"] == mode


class TestDFlash2WhitelistAndAutoRepack:
    """DFlash2 companion draft whitelist, first-Load auto-repack, and broadened artifact discovery."""

    def test_dflash2_whitelist_conversion_and_format_badges(self, tmp_path: Path, monkeypatch):
        prov_root = tmp_path / "Providers" / "io.ninfer.metal" / "versions" / "miv-v0.01"
        (prov_root / "bin").mkdir(parents=True)
        (prov_root / "metallib").mkdir(parents=True)
        (prov_root / "tools").mkdir(parents=True)
        fake_bin = _fake_daemon_script(prov_root / "bin" / "mseries_backendd", "miv-v0.01", protocol=2)
        (prov_root / "metallib" / "omlx_m5_qwen_ar.metallib").write_bytes(b"MTLB")
        captured_args_file = tmp_path / "converter_argv.json"
        fake_tool = prov_root / "tools" / "m5_artifact.py"
        fake_tool.write_text(
            "import json, pathlib, struct, sys, zlib\n"
            f"pathlib.Path({str(captured_args_file)!r}).write_text(json.dumps(sys.argv[1:]))\n"
            "args = sys.argv[1:]\n"
            "out = pathlib.Path(args[args.index('--output') + 1])\n"
            "manifest = b'{\"objects\":[]}'\n"
            "hdr_prefix = struct.pack('<8sIIQQQQII', b'MIV00001', 2, 64, 64, len(manifest), 64 + len(manifest), 0, 0, zlib.crc32(manifest) & 0xFFFFFFFF)\n"
            "hdr = hdr_prefix + struct.pack('<II', zlib.crc32(hdr_prefix) & 0xFFFFFFFF, 0)\n"
            "out.write_bytes(hdr + manifest)\n",
            encoding="utf-8",
        )
        (prov_root.parent.parent / "current").symlink_to(prov_root)
        monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(tmp_path / "Providers"))
        monkeypatch.delenv("OMLX_M5RT_BIN", raising=False)
        monkeypatch.delenv("NINFER_METAL_BIN", raising=False)

        def _make_dflash2_cfg(hidden_size: int, target_layers: int, *, vocab_size: int = 248320, model_type: str = "qwen3", arch: str = "DFlash2DraftModel") -> dict:
            return {
                "architectures": [arch],
                "model_type": model_type,
                "is_causal": False,
                "dtype": "bfloat16",
                "hidden_size": hidden_size,
                "intermediate_size": hidden_size * 3,
                "vocab_size": vocab_size,
                "num_hidden_layers": 5,
                "num_target_layers": target_layers,
                "layer_types": ["sliding_attention"] * 5,
                "dflash_config": {
                    "block_size": 8,
                    "conv_group_size": 16,
                    "conv_kernel_size": 2,
                    "mask_token_id": 100,
                    "selector_rank": 256,
                    "selector_top_k": 16,
                    "target_layer_ids": [1, 5, 9, 13, 17],
                },
            }

        orig_st_bytes = b"\x08\x00\x00\x00\x00\x00\x00\x00{}      "

        # 1) All miv-supported Qwen3.5/3.6/3.8 Dense & MoE DFlash2 target geometries pass whitelist
        for label, (h, tl) in (
            ("Qwen3.8-27B-DFlash2", (5120, 64)),
            ("Qwen3.6-35B-A3B-DFlash2", (2048, 40)),
            ("Ornith-1.5-122B-DFlash2", (3072, 48)),
            ("Qwen3.5-9B-DFlash2", (4096, 32)),
            ("Qwen3.5-4B-DFlash2", (2560, 32)),
            ("Qwen3.5-2B-DFlash2", (2048, 24)),
            ("Qwen3.5-0.8B-DFlash2", (1024, 24)),
        ):
            ddir = tmp_path / label
            ddir.mkdir()
            (ddir / "config.json").write_text(json.dumps(_make_dflash2_cfg(h, tl)), encoding="utf-8")
            (ddir / "model.safetensors").write_bytes(orig_st_bytes)
            ok, reason = m5_mod.can_convert_to_miv(ddir, str(fake_bin))
            assert ok is True and reason == "", f"{label}: {reason}"
            # Helper DFlash2 models must NOT auto-route as standalone chat LLMs
            assert m5_mod.detect_adapted(ddir).adapted is False
            assert m5_mod.can_auto_convert_m5rt_fast(ddir, str(fake_bin)) is False

        # 2) Convert 27B DFlash2 draft in-place and verify CLI flags & UI format badges
        df27_dir = tmp_path / "Qwen3.8-27B-DFlash2"
        fmt_before = m5_mod.detect_model_weight_format_info(
            df27_dir, active_external_engine=None, configured_m5rt_binary=str(fake_bin)
        )
        assert fmt_before["weight_format"] == "safetensors"
        assert fmt_before["pending_auto_repack"] is True
        assert fmt_before["repack_target_format"] == "miv"

        res = m5_mod.convert_model_to_miv(
            df27_dir, source_repo="z-lab/Qwen3.8-27B-DFlash2", configured_binary=str(fake_bin)
        )
        assert res["already_existed"] is False
        assert res["artifact_format"] == "miv"
        assert (df27_dir / "model.miv").is_file()
        argv_used = json.loads(captured_args_file.read_text(encoding="utf-8"))
        assert "--nax-pretile" not in argv_used
        assert argv_used[argv_used.index("--profile") + 1] == "dflash2_bf16"

        fmt_after = m5_mod.detect_model_weight_format_info(
            df27_dir, active_external_engine=None, configured_m5rt_binary=str(fake_bin)
        )
        assert fmt_after["weight_format"] == "miv"
        assert fmt_after["has_miv_weights"] is True
        assert fmt_after["pending_auto_repack"] is False
        assert m5_mod.detect_adapted(df27_dir).adapted is False
        ok_again, _ = m5_mod.can_convert_to_miv(df27_dir, str(fake_bin))
        assert ok_again is False

        # 3) Reject DFlash v1, non-Qwen Flint-DFlash2, and unsupported 512-expert Flash-Next geometry
        for bad_name, bad_cfg in (
            ("Qwen3.5-9B-DFlash-v1", _make_dflash2_cfg(4096, 32, arch="DFlashDraftModel")),
            ("Flint-DFlash2", _make_dflash2_cfg(2048, 49, vocab_size=150144, model_type="flint")),
            ("Qwen3.8-Flash-Next-DFlash2", _make_dflash2_cfg(2560, 48)),
        ):
            bdir = tmp_path / bad_name
            bdir.mkdir()
            (bdir / "config.json").write_text(json.dumps(bad_cfg), encoding="utf-8")
            (bdir / "model.safetensors").write_bytes(orig_st_bytes)
            b_ok, b_reason = m5_mod.can_convert_to_miv(bdir, str(fake_bin))
            assert b_ok is False and b_reason != ""

    async def test_dflash2_auto_repack_on_first_load_and_artifact_discovery(
        self, tmp_path: Path, monkeypatch
    ):
        providers_root = tmp_path / "Providers"
        miv_ver = providers_root / "io.ninfer.metal" / "versions" / "miv-v0.01"
        (miv_ver / "bin").mkdir(parents=True)
        (miv_ver / "metallib").mkdir(parents=True)
        (miv_ver / "tools").mkdir(parents=True)
        miv_sock_srv = _MockUdsSidecarServer(
            Path("/tmp") / f"m5-df2-{os.getpid()}.sock",
            support_push=True,
            support_v2_coexistence=True,
            token_plan=[[65, 66]],
        )
        await miv_sock_srv.start()
        try:
            _fake_daemon_script(miv_ver / "bin" / "mseries_backendd", "miv-v0.01", protocol=2)
            (miv_ver / "metallib" / "omlx_m5_qwen_ar.metallib").write_bytes(b"MTLB")
            (miv_ver / "metallib" / "omlx_m5_dflash2_program.metallib").write_bytes(b"MTLB")
            (miv_ver / "tools" / "m5_artifact.py").write_text(
                "import pathlib, struct, sys, zlib\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
                "manifest = b'{\"objects\":[]}'\n"
                "hdr_prefix = struct.pack('<8sIIQQQQII', b'MIV00001', 2, 64, 64, len(manifest), 64 + len(manifest), 0, 0, zlib.crc32(manifest) & 0xFFFFFFFF)\n"
                "hdr = hdr_prefix + struct.pack('<II', zlib.crc32(hdr_prefix) & 0xFFFFFFFF, 0)\n"
                "out.write_bytes(hdr + manifest)\n",
                encoding="utf-8",
            )
            (miv_ver / "provider.json").write_text(
                json.dumps(
                    {
                        "provider_id": "io.ninfer.metal",
                        "engine_id": "m5rt",
                        "display_name": "miv Core Backend",
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
            (providers_root / "io.ninfer.metal" / "current").symlink_to(miv_ver)
            monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(providers_root))
            monkeypatch.setattr(m5_mod, "_DEFAULT_ARTIFACT_CACHE_RELPATH", tmp_path / "isolated_repo_cache")
            monkeypatch.delenv("OMLX_M5RT_ARTIFACT_DIR", raising=False)
            monkeypatch.delenv("OMLX_M5RT_BIN", raising=False)
            monkeypatch.delenv("NINFER_METAL_BIN", raising=False)

            qwen_27b = tmp_path / "models" / "Qwen3.8-27B-mxfp4"
            qwen_27b.mkdir(parents=True)
            (qwen_27b / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "hidden_size": 5120,
                        "num_hidden_layers": 64,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                        "vocab_size": 248320,
                        "quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
                    }
                ),
                encoding="utf-8",
            )
            (qwen_27b / "model.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")

            # Unrelated 9B DFlash v1 helper in pool + matching 27B DFlash2 helper (unrepacked .safetensors)
            df_9b_v1 = tmp_path / "models" / "Qwen3.5-9B-DFlash"
            df_9b_v1.mkdir(parents=True)
            (df_9b_v1 / "config.json").write_text(
                json.dumps({"architectures": ["DFlashDraftModel"], "model_type": "qwen3", "hidden_size": 4096, "vocab_size": 248320}),
                encoding="utf-8",
            )

            df_27b = tmp_path / "models" / "Qwen3.8-27B-DFlash2"
            df_27b.mkdir(parents=True)
            (df_27b / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["DFlash2DraftModel"],
                        "model_type": "qwen3",
                        "is_causal": False,
                        "dtype": "bfloat16",
                        "hidden_size": 5120,
                        "intermediate_size": 17408,
                        "vocab_size": 248320,
                        "num_hidden_layers": 5,
                        "num_target_layers": 64,
                        "layer_types": ["sliding_attention"] * 5,
                        "dflash_config": {
                            "block_size": 8,
                            "conv_group_size": 16,
                            "conv_kernel_size": 2,
                            "mask_token_id": 248070,
                            "selector_rank": 256,
                            "selector_top_k": 16,
                            "target_layer_ids": [5, 19, 33, 47, 61],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (df_27b / "model.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00" + b"{}" + b" " * 4096)

            gs = GlobalSettings(base_path=tmp_path / "base")
            monkeypatch.setattr("omlx.engine_pool.get_global_settings", lambda: gs)
            pool = EnginePool()
            pool._entries["Qwen3.8-27B-mxfp4"] = EngineEntry(
                model_id="Qwen3.8-27B-mxfp4",
                model_path=str(qwen_27b),
                model_type="llm",
                engine_type="batched",
                estimated_size=4096,
            )
            pool._entries["Qwen3.5-9B-DFlash"] = EngineEntry(
                model_id="Qwen3.5-9B-DFlash",
                model_path=str(df_9b_v1),
                model_type="llm",
                engine_type="batched",
                estimated_size=1024,
                is_helper=True,
            )
            pool._entries["Qwen3.8-27B-DFlash2"] = EngineEntry(
                model_id="Qwen3.8-27B-DFlash2",
                model_path=str(df_27b),
                model_type="llm",
                engine_type="batched",
                estimated_size=2048,
                is_helper=True,
            )

            # Without explicit dflash_draft_model, candidate scan picks matching 27B DFlash2 over 9B DFlash v1!
            ms = ModelSettings(
                dflash_enabled=True,
                turboquant_kv_enabled=True,
                turboquant_kv_scheme="turboquant",
                turboquant_kv_bits=4.0,
            )
            target, err = pool.m5rt_target("Qwen3.8-27B-mxfp4", ms)
            assert err == "" and target is not None
            assert target.draft_dir == df_27b.resolve()
            assert target.draft_artifact_path is None
            assert target.auto_convert_output == qwen_27b / "model.miv"
            assert target.auto_convert_draft_output == df_27b / "model.miv"

            # First Load auto-repacks BOTH target model.miv AND draft model.miv + enables affine4_g128 KV!
            eng = pool._external_engine_for(pool._entries["Qwen3.8-27B-mxfp4"], ms, "batched")
            assert isinstance(eng, M5SidecarEngine)
            monkeypatch.setattr(m5_mod, "_short_socket_path", lambda _eid=None: miv_sock_srv.socket_path)
            monkeypatch.setattr(eng, "_load_tokenizer", lambda: None)
            real_popen = m5_mod.subprocess.Popen
            monkeypatch.setattr(
                m5_mod.subprocess,
                "Popen",
                lambda cmd, *a, **kw: (
                    SimpleNamespace(pid=os.getpid(), poll=lambda: None, returncode=None, wait=lambda: 0)
                    if isinstance(cmd, list) and "m5rt-watchdog" in cmd
                    else real_popen(cmd, *a, **kw)
                ),
            )
            await eng.start()
            try:
                assert (qwen_27b / "model.miv").is_file()
                assert (df_27b / "model.miv").is_file()
                lp = miv_sock_srv.load_params[-1]
                assert lp["kv_dtype"] == "affine4_g128"
                assert lp["speculative"] == {"backend": "dflash2", "draft_tokens": 0}
                assert Path(lp["path_overrides"]["artifact"]) == (qwen_27b / "model.miv").resolve()
                assert Path(lp["path_overrides"]["draft_artifact"]) == (df_27b / "model.miv").resolve()
            finally:
                eng._process = None
                await eng.stop()

            # Verify broadened draft filename discovery: draft.miv, custom *.omlxm5, and .cache/b0/artifacts
            (df_27b / "model.miv").rename(df_27b / "qwen38-27b-dflash2.omlxm5")
            t_custom, err_custom = pool.m5rt_target("Qwen3.8-27B-mxfp4", ms)
            assert err_custom == "" and t_custom is not None
            assert t_custom.draft_artifact_path == (df_27b / "qwen38-27b-dflash2.omlxm5").resolve()

            cache_dir = tmp_path / "artifact_cache"
            cache_dir.mkdir()
            cached_art = cache_dir / "qwen38-27b-dflash2.miv"
            (df_27b / "qwen38-27b-dflash2.omlxm5").rename(cached_art)
            monkeypatch.setenv("OMLX_M5RT_ARTIFACT_DIR", str(cache_dir))
            t_cached, err_cached = pool.m5rt_target("Qwen3.8-27B-mxfp4", ms)
            assert err_cached == "" and t_cached is not None
            assert t_cached.draft_artifact_path == cached_art.resolve()

            # Verify companion dflash2/ subfolder without its own artifact does NOT steal parent model.miv
            monkeypatch.delenv("OMLX_M5RT_ARTIFACT_DIR", raising=False)
            sub_df = qwen_27b / "dflash2"
            sub_df.mkdir()
            (sub_df / "config.json").write_text((df_27b / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
            ms_sub = ModelSettings(dflash_enabled=True, dflash_draft_model=str(sub_df))
            t_sub, err_sub = m5_mod.resolve_m5rt_target(qwen_27b, None, ms_sub, allow_auto_convert=False)
            assert t_sub is None and "requires a draft .miv/.omlxm5 artifact" in err_sub
        finally:
            await miv_sock_srv.close()

    def test_missing_tokenizer_raises_for_real_vocab_but_allows_small_fixtures(self, tmp_path):
        import omlx.engine.m5_sidecar as m5_mod

        fake_bin = _fake_daemon_script(tmp_path / "bin" / "mseries_backendd")
        binary, err = m5_mod.inspect_m5rt_binary(fake_bin, auto_detected=False)
        assert err == "" and binary is not None

        # 1) Real-model vocab (> 1024) without HF tokenizer files must fail fast with ModelLoadingError
        real_dir = _write_test_model(
            tmp_path / "models" / "Qwen3.8-27B-mxfp4",
            vocab_size=248320,
            with_tokenizer=False,
        )
        eng_real = M5SidecarEngine(
            model_name=str(real_dir),
            model_id="Qwen3.8-27B-mxfp4",
            target=M5Target(binary=binary, artifact_path=real_dir / "model.omlxm5", metallib_dir=tmp_path),
            log_dir=tmp_path / "logs",
            options=M5Options(max_context=4096),
        )
        with pytest.raises(ModelLoadingError, match="missing HuggingFace tokenizer files"):
            eng_real._load_tokenizer()

        # 2) Small C++ unit-test fixture (vocab_size <= 1024) without HF tokenizer keeps byte fallback
        fix_dir = _write_test_model(
            tmp_path / "models" / "qwen-fixture",
            vocab_size=257,
            with_tokenizer=False,
        )
        eng_fix = M5SidecarEngine(
            model_name=str(fix_dir),
            model_id="qwen-fixture",
            target=M5Target(binary=binary, artifact_path=fix_dir / "model.omlxm5", metallib_dir=tmp_path),
            log_dir=tmp_path / "logs",
            options=M5Options(max_context=4096),
        )
        eng_fix._load_tokenizer()
        assert eng_fix.tokenizer is not None







def test_miv_launch_plan_uses_same_window_for_load_estimate_and_signature(tmp_path, monkeypatch):
    fixture = Path(__file__).with_name("fixtures") / "ninfer_shared_serve_contract.toml"
    monkeypatch.setenv("NINFER_METAL_CONFIG", str(fixture))
    model_dir = _write_test_model(tmp_path / "source-dir")
    config_file = model_dir / "config.json"
    config = json.loads(config_file.read_text())
    config["text_config"]["max_position_embeddings"] = 262144
    config_file.write_text(json.dumps(config))
    binary = m5_mod.M5Binary(path=tmp_path / "daemon", version="miv-v0.02", protocol_version=2, auto_detected=False)
    target = M5Target(binary=binary, artifact_path=model_dir / "model.omlxm5", metallib_dir=None)
    gs = GlobalSettings(base_path=tmp_path / "base")
    gs.sampling.max_context_window_policy = 8192
    settings = ModelSettings()

    def plan():
        return m5_mod.build_m5rt_options(
            model_dir, target, settings, default_context=8192, memory_budget_bytes=0,
            global_settings=gs, model_id="shared-fixture",
        )

    first = plan()
    assert (first.max_context, first.max_concurrency, first.kv_dtype) == (8192, 4, "affine4_g128")
    assert (first.speculative_backend, first.draft_tokens, first.spec_search) == ("dflash2", 0, "on")
    assert first.batch_mode is False
    assert first.prefix_cache_bytes == 0

    engine = M5SidecarEngine(str(model_dir), "shared-fixture", target, tmp_path / "logs", first)
    params = engine._build_load_params()
    assert (params["max_context"], params["max_concurrency"], params["kv_dtype"]) == (
        first.max_context, first.max_concurrency, first.kv_dtype,
    )
    assert params["path_overrides"]["prefix_cache_bytes"] == "0"

    def estimate():
        return m5rt_memory_estimate(
            model_dir, target, 8192, settings, global_settings=gs, model_id="shared-fixture",
        )

    def signature():
        return m5_mod.m5rt_runtime_signature(
            settings, None, model_path=model_dir, target=target, default_context=8192,
            model_id="shared-fixture", global_settings=gs,
        )

    before, old_size = signature(), estimate()
    gs.scheduler.max_concurrent_requests = 2
    assert signature() == before
    gs.sampling.max_context_window_policy = None
    assert plan().max_context == 65536
    assert signature() != before
    assert estimate() > old_size


def test_miv_provider_field_patch_and_kv_button_preserve_unedited_options():
    from omlx.admin.routes import ModelSettingsRequest

    settings = ModelSettings(
        external_engine="m5rt", mtp_fixed_depth=4,
        external_engine_options={"m5rt": {"kv_dtype": "bf16", "max_concurrency": 1}},
        m5rt_options={"kv_dtype": "bf16"},
    )
    req = ModelSettingsRequest(
        turboquant_kv_enabled=True,
        external_engine_options_patch={"m5rt": {"max_concurrency": 4}},
    )
    m5_mod.apply_m5rt_model_settings_update(
        settings, req, req.model_fields_set, is_diffusion_model=False,
    )
    assert settings.mtp_fixed_depth == 4
    assert settings.m5rt_options is None
    assert settings.external_engine_options == {"m5rt": {"max_concurrency": 4}}


@pytest.mark.parametrize("opened", [False, True])
def test_sidecar_stop_ignores_streamed_think_text(opened):
    matcher = m5_mod._StopMatcher([], ["STOP"], skip_reasoning=True, prompt_opens_think=opened)
    prefix = [] if opened else ["<th", "ink>"]
    emitted = "".join(matcher.push_text(x) for x in prefix + ["STOP ex", "ample</th", "ink>visible ST", "OP hidden"])
    assert emitted == ("<think>" if not opened else "") + "STOP example</think>visible "
    assert matcher.stopped


def test_sidecar_stop_supports_custom_thinking_markers():
    matcher = m5_mod._StopMatcher(
        [],
        ["STOP"],
        skip_reasoning=True,
        prompt_opens_think=False,
        think_open="<|thought|>",
        think_close="<|/thought|>",
    )
    chunks = ["<|tho", "ught|>STOP inside<|/th", "ought|>answer ST", "OP tail"]
    emitted = "".join(matcher.push_text(c) for c in chunks)
    assert emitted == "<|thought|>STOP inside<|/thought|>answer "
    assert matcher.stopped


def test_sync_m5rt_ui_kv_setting_normalizes_leftover_stock_affine8():
    from omlx.admin.routes import ModelSettingsRequest

    settings = ModelSettings(
        external_engine="m5rt",
        turboquant_kv_enabled=True,
        turboquant_kv_scheme="affine8",
        turboquant_kv_bits=8.0,
        m5rt_options={"kv_dtype": "bf16"},
        external_engine_options={"m5rt": {"kv_dtype": "bf16"}},
    )
    req = ModelSettingsRequest(turboquant_kv_enabled=True)
    m5_mod.apply_m5rt_model_settings_update(
        settings, req, req.model_fields_set, is_diffusion_model=False,
    )
    assert settings.turboquant_kv_scheme == "affine4"
    assert settings.turboquant_kv_bits == 4.0
    assert settings.m5rt_options is None
    assert settings.external_engine_options is None
    assert m5_mod._resolve_effective_kv_dtype({}, settings) == "affine4_g128"

    # Switching from Stock MLX (with 6-bit turboquant enabled) to m5rt also normalizes.
    switched = ModelSettings(
        external_engine=None,
        turboquant_kv_enabled=True,
        turboquant_kv_scheme="turboquant",
        turboquant_kv_bits=6.0,
    )
    req_switch = ModelSettingsRequest(external_engine="m5rt")
    m5_mod.apply_m5rt_model_settings_update(
        switched, req_switch, req_switch.model_fields_set, is_diffusion_model=False,
    )
    assert switched.turboquant_kv_scheme == "affine4"
    assert switched.turboquant_kv_bits == 4.0
    assert m5_mod._resolve_effective_kv_dtype({}, switched) == "affine4_g128"


def test_m5rt_repetition_context_size_rejects_oversized_window():
    engine = M5SidecarEngine.__new__(M5SidecarEngine)
    engine._model_id = "test-model"
    engine._options = M5Options(sampling_contract="position_coupled_v1")
    engine._last_stats = {"sampling_contract_id": "position_coupled_v1"}
    engine._hello_info = {"capabilities": ["sampling_penalties", "thinking_budget"]}
    engine._warned_request_controls = set()
    engine._tokenizer = None
    engine._vocab_size = 32000

    with pytest.raises(InvalidRequestError):
        engine.reconcile_request_controls(
            {"repetition_penalty": 1.1, "repetition_context_size": m5_mod.M5RT_MAX_REPETITION_CONTEXT_SIZE + 1},
            {"repetition_context_size": "repetition_context_size"},
        )


@pytest.mark.asyncio
async def test_sidecar_string_stop_cancel_waits_for_async_request_stats(tmp_path):
    model_dir = _write_test_model(tmp_path / "cancel-stats")
    tok = _build_bpe_chat_tokenizer(model_dir)
    bin_path = _fake_daemon_script(tmp_path / "bin" / "mseries_backendd", "miv-v0.03", protocol=2)
    m5_bin, _ = find_m5rt_binary(str(bin_path))
    assert m5_bin is not None
    engine = M5SidecarEngine(
        model_name=str(model_dir),
        model_id="cancel-stats",
        target=M5Target(binary=m5_bin, artifact_path=model_dir / "model.omlxm5", metallib_dir=None),
        log_dir=tmp_path / "logs",
        options=M5Options(max_context=4096, sampling_contract="position_coupled_v1"),
    )
    engine._tokenizer = tok
    engine._vocab_size = len(tok)
    engine._eos_token_ids = [tok.eos_token_id] if tok.eos_token_id is not None else [0]
    engine._process = SimpleNamespace(pid=os.getpid(), poll=lambda: None, returncode=None)  # type: ignore[assignment]
    engine._hello_info = {"capabilities": ["subscribe_events"]}

    hello_ids = tok.encode("hello STOP world", add_special_tokens=False)

    async def fake_rpc(method: str, params: dict | None = None, timeout: float = 30.0):
        if method == "submit":
            q = engine._request_queues.setdefault(42, asyncio.Queue())
            q.put_nowait(m5_mod._SidecarEvent(request_id=42, kind="tokens", tokens=tuple(hello_ids)))
            return {"request_id": 42}
        return {"ok": True}

    async def fake_cancel(backend_id: int) -> bool:
        async def push_later():
            await asyncio.sleep(0.02)
            q = engine._request_queues.get(backend_id)
            if q is not None:
                q.put_nowait(
                    m5_mod._SidecarEvent(
                        request_id=backend_id,
                        kind="aborted",
                        finish_reason="cancelled",
                        request_stats={
                            "prefix_restored_tokens": 5,
                            "spec_rounds": 2,
                            "ar_steps": 0,
                            "decode_mode": "mtp",
                            "fallback_reason": "",
                        },
                    )
                )

        asyncio.create_task(push_later())
        return True

    engine._rpc = fake_rpc  # type: ignore[assignment]
    engine._cancel_backend_request = fake_cancel  # type: ignore[assignment]

    final = None
    async for chunk in engine.stream_generate([1, 2, 3, 4, 5, 6, 7], max_tokens=32, stop=["STOP"]):
        final = chunk
    assert final is not None
    assert final.finished is True
    assert final.finish_reason == "stop"
    assert final.cached_tokens == 5


def test_stock_dflash_block_size_clamps_for_m5rt_and_m5options_default_contract(tmp_path):
    assert M5Options().sampling_contract == "position_coupled_v1"

    model_dir = _write_test_model(tmp_path / "models" / "Qwen3.8-DFlash-Clamp")
    _write_test_model(model_dir / "dflash2")
    fake_bin = _fake_daemon_script(tmp_path / "bin" / "mseries_backendd", "miv-v0.03", protocol=2)
    metallib_dir = tmp_path / "bin" / "metallib"
    metallib_dir.mkdir(exist_ok=True)
    (metallib_dir / "omlx_m5_qwen_ar.metallib").write_bytes(b"MTLB")
    target, err = resolve_m5rt_target(model_dir, str(fake_bin), ModelSettings())
    assert target is not None and err == ""

    # Leftover Stock MLX dflash_block_size=16 must clamp cleanly to draft_tokens=7 (not crash).
    opts = m5_mod.build_m5rt_options(
        model_dir,
        target,
        ModelSettings(dflash_enabled=True, dflash_block_size=16),
        default_context=32768,
        memory_budget_bytes=0,
    )
    assert opts.speculative_backend == "dflash2"
    assert opts.draft_tokens == 7

    # Explicit m5rt_options.draft_tokens out of range must still raise ValueError.
    with pytest.raises(ValueError, match="dflash2 draft_tokens must be in"):
        m5_mod.build_m5rt_options(
            model_dir,
            target,
            ModelSettings(dflash_enabled=True, m5rt_options={"draft_tokens": 8}),
            default_context=32768,
            memory_budget_bytes=0,
        )

    # Switching external_engine to m5rt without sending dflash_block_size normalizes 16 -> 8.
    from omlx.admin.routes import ModelSettingsRequest

    ms = ModelSettings(external_engine="stock", dflash_enabled=True, dflash_block_size=16)
    req = ModelSettingsRequest(external_engine="m5rt")
    m5_mod.sync_m5rt_ui_kv_setting(ms, req, req.model_fields_set)
    assert ms.dflash_block_size == 8


@pytest.mark.asyncio
async def test_apply_max_concurrent_requests_reloads_only_global_sidecars(tmp_path):
    from omlx.engine_pool import EngineEntry, EnginePool

    class _FakeSidecar:
        external_engine_id = "m5rt"

        def __init__(self, opts: M5Options) -> None:
            self._options = opts

    pool = EnginePool()

    def _make_entry(model_id: str, conc: int, source: str) -> EngineEntry:
        eng = _FakeSidecar(M5Options(max_concurrency=conc, concurrency_source=source))
        entry = EngineEntry(
            model_id=model_id,
            model_path=str(tmp_path / model_id),
            model_type="llm",
            engine_type="llm",
            estimated_size=0,
        )
        entry.engine = eng  # type: ignore[assignment]
        return entry

    pool._entries["global-stale"] = _make_entry("global-stale", 2, "global")
    pool._entries["global-same"] = _make_entry("global-same", 4, "global")
    pool._entries["toml-pinned"] = _make_entry("toml-pinned", 1, "shared_toml")
    pool._entries["model-override"] = _make_entry("model-override", 2, "model_options")

    reloaded: list[tuple[str, bool, str]] = []

    async def fake_request_unload(model_id: str, *, reason: str = "", abort_active: bool = True):
        reloaded.append((model_id, abort_active, reason))
        return True

    pool.request_unload = fake_request_unload  # type: ignore[method-assign]
    await pool.apply_max_concurrent_requests(4)
    assert reloaded == [("global-stale", False, "max_concurrent_requests changed")]


def test_pre_convert_safetensors_mtp_detection_prevents_stripped_mtp_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("NINFER_METAL_CONFIG", str(tmp_path / "none.toml"))
    binary = M5Binary(
        path=Path("/fake/mseries_backendd"),
        version="miv-v0.03",
        protocol_version=2,
        auto_detected=False,
    )

    # 1) OsaurusAI/Qwen3.6-35B-A3B-mxfp4 style: config.json keeps mtp_num_hidden_layers=1
    #    while model.safetensors.index.json has 0 MTP keys -> must resolve to "none" pre-convert.
    stripped_dir = tmp_path / "Qwen3.6-35B-A3B-mxfp4"
    stripped_dir.mkdir()
    (stripped_dir / "config.json").write_text(
        json.dumps({"text_config": {"mtp_num_hidden_layers": 1, "max_position_embeddings": 262144}}),
        encoding="utf-8",
    )
    (stripped_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"language_model.lm_head.weight": "model-00001.safetensors"}}),
        encoding="utf-8",
    )
    t_stripped = M5Target(
        binary=binary,
        artifact_path=None,
        metallib_dir=tmp_path,
        auto_convert_output=stripped_dir / "model.miv",
    )
    p_stripped = m5_mod.build_m5rt_options(
        stripped_dir, t_stripped, ModelSettings(), default_context=65536, memory_budget_bytes=0
    )
    assert p_stripped.speculative_backend == "none"

    # 2) Jundot/Qwen3.6-35B-A3B-oQ4e-mtp style: weight_map retains language_model.mtp.* -> "mtp".
    mtp_dir = tmp_path / "Qwen3.6-35B-A3B-oQ4e-mtp"
    mtp_dir.mkdir()
    (mtp_dir / "config.json").write_text(
        json.dumps({"text_config": {"mtp_num_hidden_layers": 1, "max_position_embeddings": 262144}}),
        encoding="utf-8",
    )
    (mtp_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "language_model.lm_head.weight": "model-00001.safetensors",
                    "language_model.mtp.fc.weight": "model-00004.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    t_mtp = M5Target(
        binary=binary,
        artifact_path=None,
        metallib_dir=tmp_path,
        auto_convert_output=mtp_dir / "model.miv",
    )
    p_mtp = m5_mod.build_m5rt_options(
        mtp_dir, t_mtp, ModelSettings(), default_context=65536, memory_budget_bytes=0
    )
    assert p_mtp.speculative_backend == "mtp"


def test_m4a_orthogonal_prefill_switches_all_96_combinations_and_legacy_shim(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("NINFER_METAL_NO_DEFAULT_CONFIG", "1")
    monkeypatch.delenv("NINFER_METAL_CONFIG", raising=False)

    schema_by_key = {item["key"]: item for item in m5_mod.M5RT_DEFAULT_OPTIONS_SCHEMA}
    assert schema_by_key["max_context"]["max"] >= 262144
    assert schema_by_key["max_context"]["default"] == 262144
    assert schema_by_key["prefill_linear_mode"]["options"] == ["bf16", "w4a8", "w4a4"]
    assert schema_by_key["prefill_linear_mode"]["default"] == "bf16"
    assert schema_by_key["prefill_attention_mode"]["options"] == ["exact_nax", "sage2"]
    assert schema_by_key["prefill_attention_mode"]["default"] == "exact_nax"
    for bkey in ("prefill_sparse_minference", "prefill_cache_blend"):
        assert schema_by_key[bkey]["type"] == "bool"
        assert schema_by_key[bkey]["default"] is False
    assert schema_by_key["prefill_ane_mode"]["type"] == "enum"
    assert schema_by_key["prefill_ane_mode"]["options"] == [
        "off",
        "auto",
        "compact_int8",
        "ultra_fp16_swiglu",
    ]
    assert schema_by_key["prefill_ane_mode"]["default"] == "off"
    assert "prefill_ane_offload" not in schema_by_key

    model_dir = _write_test_model(tmp_path / "Qwen3.8-27B-SPI")
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg["model_type"] = "qwen3_8"
    cfg["text_config"]["max_position_embeddings"] = 262144
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    fake_bin = _fake_daemon_script(tmp_path / "bin" / "mseries_backendd", "miv-v0.03", protocol=2)
    binary, err = find_m5rt_binary(str(fake_bin))
    assert binary is not None and err == ""
    target = M5Target(
        binary=binary,
        artifact_path=model_dir / "model.omlxm5",
        metallib_dir=tmp_path / "bin",
        fake=False,
    )

    combos = 0
    seen_sigs = set()
    for lm in ("bf16", "w4a8", "w4a4"):
        for am in ("exact_nax", "sage2"):
            for sp in (False, True):
                for cb in (False, True):
                    for ane_mode in ("off", "auto", "compact_int8", "ultra_fp16_swiglu"):
                        combos += 1
                        ane_bool = ane_mode != "off"
                        raw = {
                            "max_context": 262144,
                            "prefill_linear_mode": lm,
                            "prefill_attention_mode": am,
                            "prefill_sparse_minference": sp,
                            "prefill_cache_blend": cb,
                            "prefill_ane_mode": ane_mode,
                        }
                        validated = m5_mod.validate_m5rt_options(raw)
                        assert validated is not None
                        assert validated["prefill_linear_mode"] == lm
                        assert validated["prefill_attention_mode"] == am
                        assert validated["prefill_sparse_minference"] is sp
                        assert validated["prefill_cache_blend"] is cb
                        assert validated["prefill_ane_mode"] == ane_mode
                        assert validated["prefill_ane_offload"] is ane_bool

                        ms = ModelSettings(
                            external_engine=M5RT_ENGINE_ID,
                            external_engine_options={M5RT_ENGINE_ID: raw},
                        )
                        plan = m5_mod.build_m5rt_options(
                            model_dir,
                            target,
                            ms,
                            default_context=32768,
                            memory_budget_bytes=0,
                        )
                        assert plan.max_context == 262144
                        assert plan.prefill_linear_mode == lm
                        assert plan.prefill_attention_mode == am
                        assert plan.prefill_sparse_minference is sp
                        assert plan.prefill_cache_blend is cb
                        assert plan.prefill_ane_mode == ane_mode
                        assert plan.prefill_ane_offload is ane_bool

                        rpc = plan.to_rpc_dict()
                        assert rpc["prefill_linear_mode"] == lm
                        assert rpc["prefill_attention_mode"] == am
                        assert rpc["prefill_sparse_minference"] is sp
                        assert rpc["prefill_cache_blend"] is cb
                        assert rpc["prefill_ane_mode"] == ane_mode
                        assert rpc["prefill_ane_offload"] is ane_bool

                        eng = M5SidecarEngine(
                            str(model_dir), "spi-test", target, tmp_path / "logs", plan
                        )
                        load_params = eng._build_load_params()
                        assert load_params["prefill_linear_mode"] == lm
                        assert load_params["prefill_attention_mode"] == am
                        assert load_params["prefill_sparse_minference"] is sp
                        assert load_params["prefill_cache_blend"] is cb
                        assert load_params["prefill_ane_mode"] == ane_mode
                        assert load_params["prefill_ane_offload"] is ane_bool

                        sig = m5_mod.m5rt_runtime_signature(
                            ms,
                            None,
                            model_path=model_dir,
                            target=target,
                            default_context=32768,
                        )
                        seen_sigs.add(json.dumps(sig, sort_keys=True))

    assert combos == 96
    assert len(seen_sigs) == 96

    # Invalid values must raise ValueError
    with pytest.raises(ValueError, match="prefill_linear_mode"):
        m5_mod.validate_m5rt_options({"prefill_linear_mode": "int4"})
    with pytest.raises(ValueError, match="prefill_attention_mode"):
        m5_mod.validate_m5rt_options({"prefill_attention_mode": "flash2"})
    with pytest.raises(ValueError, match="prefill_ane_mode"):
        m5_mod.validate_m5rt_options({"prefill_ane_mode": "turbo_ane"})

    # Legacy prefill_engine = "nax_ane" shim maps prefill_ane_mode="auto" & prefill_ane_offload=True
    # WITHOUT touching w4a4 / sage2 / minference / cache_blend!
    ms_legacy = ModelSettings(
        external_engine=M5RT_ENGINE_ID,
        external_engine_options={
            M5RT_ENGINE_ID: {
                "prefill_engine": "nax_ane",
                "prefill_linear_mode": "w4a4",
                "prefill_attention_mode": "sage2",
                "prefill_sparse_minference": True,
                "prefill_cache_blend": True,
            }
        },
    )
    plan_legacy = m5_mod.build_m5rt_options(
        model_dir, target, ms_legacy, default_context=32768, memory_budget_bytes=0
    )
    assert plan_legacy.prefill_ane_mode == "auto"
    assert plan_legacy.prefill_ane_offload is True
    assert plan_legacy.prefill_linear_mode == "w4a4"
    assert plan_legacy.prefill_attention_mode == "sage2"
    assert plan_legacy.prefill_sparse_minference is True
    assert plan_legacy.prefill_cache_blend is True

    # Legacy prefill_ane_offload=True -> prefill_ane_mode="auto"; explicit prefill_ane_mode wins
    ms_legacy_bool = ModelSettings(
        external_engine=M5RT_ENGINE_ID,
        external_engine_options={M5RT_ENGINE_ID: {"prefill_ane_offload": True}},
    )
    plan_legacy_bool = m5_mod.build_m5rt_options(
        model_dir, target, ms_legacy_bool, default_context=32768, memory_budget_bytes=0
    )
    assert plan_legacy_bool.prefill_ane_mode == "auto"
    assert plan_legacy_bool.prefill_ane_offload is True

    # Explicit prefill_ane_offload=False beats legacy prefill_engine="nax_ane"
    ms_override = ModelSettings(
        external_engine=M5RT_ENGINE_ID,
        external_engine_options={
            M5RT_ENGINE_ID: {
                "prefill_engine": "nax_ane",
                "prefill_ane_offload": False,
                "prefill_linear_mode": "w4a8",
            }
        },
    )
    plan_override = m5_mod.build_m5rt_options(
        model_dir, target, ms_override, default_context=32768, memory_budget_bytes=0
    )
    assert plan_override.prefill_ane_mode == "off"
    assert plan_override.prefill_ane_offload is False
    assert plan_override.prefill_linear_mode == "w4a8"

    # Explicit prefill_ane_mode="off" beats legacy prefill_ane_offload=True & prefill_engine="nax_ane"
    ms_mode_wins = ModelSettings(
        external_engine=M5RT_ENGINE_ID,
        external_engine_options={
            M5RT_ENGINE_ID: {
                "prefill_engine": "nax_ane",
                "prefill_ane_offload": True,
                "prefill_ane_mode": "off",
            }
        },
    )
    plan_mode_wins = m5_mod.build_m5rt_options(
        model_dir, target, ms_mode_wins, default_context=32768, memory_budget_bytes=0
    )
    assert plan_mode_wins.prefill_ane_mode == "off"
    assert plan_mode_wins.prefill_ane_offload is False


def test_m4a_c5_max_context_policy_unclamping_and_c6_shared_kv_memory_estimate(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("NINFER_METAL_NO_DEFAULT_CONFIG", "1")
    monkeypatch.delenv("NINFER_METAL_CONFIG", raising=False)

    # Qwen3.8-27B geometry: 48 GDN + 16 Full-Attn, 4 KV heads, head_dim=256, 256K native ctx
    model_dir = _write_test_model(tmp_path / "Qwen3.8-27B-256K")
    cfg = {
        "model_type": "qwen3_8",
        "text_config": {
            "model_type": "qwen3_8",
            "max_position_embeddings": 262144,
            "num_hidden_layers": 64,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    fake_bin = _fake_daemon_script(tmp_path / "bin" / "mseries_backendd", "miv-v0.03", protocol=2)
    binary, _ = find_m5rt_binary(str(fake_bin))
    assert binary is not None
    target = M5Target(
        binary=binary,
        artifact_path=model_dir / "model.omlxm5",
        metallib_dir=tmp_path / "bin",
        fake=False,
    )

    # C5: global max_context_window_policy MUST NOT clamp explicit ext_opts["max_context"]
    gs = GlobalSettings(base_path=tmp_path / "base")
    gs.sampling.max_context_window_policy = 32768

    ms_explicit_ext = ModelSettings(
        external_engine=M5RT_ENGINE_ID,
        external_engine_options={M5RT_ENGINE_ID: {"max_context": 262144}},
    )
    plan_ext = m5_mod.build_m5rt_options(
        model_dir,
        target,
        ms_explicit_ext,
        default_context=32768,
        memory_budget_bytes=0,
        global_settings=gs,
    )
    assert plan_ext.max_context == 262144

    # When neither max_context_window nor ext_opts["max_context"] is set, policy still clamps
    plan_clamped = m5_mod.build_m5rt_options(
        model_dir,
        target,
        ModelSettings(),
        default_context=262144,
        memory_budget_bytes=0,
        global_settings=gs,
    )
    assert plan_clamped.max_context == 32768

    # C6: estimate_m5rt_memory_bytes resolves "auto" -> "affine4_g128" for Qwen3.8
    # and uses shared KvPagePool page count (4128 pages at 262144, B=4 -> ~4.55 GiB KV),
    # NOT B * max_context * bf16 (~64 GiB)!
    plan_auto = m5_mod.M5Options(
        max_context=262144,
        max_concurrency=4,
        kv_dtype="auto",
        prefix_cache_bytes=0,
    )
    plan_aff4 = m5_mod.M5Options(
        max_context=262144,
        max_concurrency=4,
        kv_dtype="affine4_g128",
        prefix_cache_bytes=0,
    )
    est_auto = m5_mod.estimate_m5rt_memory_bytes(
        model_dir, target, 262144, plan=plan_auto
    )
    est_aff4 = m5_mod.estimate_m5rt_memory_bytes(
        model_dir, target, 262144, plan=plan_aff4
    )
    assert est_auto == est_aff4

    # Exact shared KvPagePool formula: 4128 * 1,183,744 = 4,886,495,232 (~4.55 GiB)
    expected_kv = 4128 * 1183744
    weights_sz = (model_dir / "model.omlxm5").stat().st_size
    expected_ws = max(64 * 1024**2, 1024 * 248320 * 2 * 4)
    assert est_auto == weights_sz + expected_kv + expected_ws
    assert est_auto < 10 * (1 << 30)  # Far below old 64 GiB bf16*B overestimate!


def test_effective_load_settings_forwards_live_prefill_telemetry(tmp_path: Path) -> None:
    engine = m5_mod.M5SidecarEngine.__new__(m5_mod.M5SidecarEngine)
    engine._options = m5_mod.M5Options(
        max_context=262144,
        max_concurrency=4,
        kv_dtype="affine4_g128",
        concurrency_source="model",
        prefill_linear_mode="w4a4",
        prefill_attention_mode="sage2",
        prefill_sparse_minference=True,
        prefill_cache_blend=True,
        prefill_ane_mode="ultra_fp16_swiglu",
        prefill_ane_offload=True,
    )
    engine._last_stats = {
        "max_context": 262144,
        "max_concurrency": 4,
        "kv_dtype": "affine4_g128",
        "sampling_contract_id": "position_coupled_v1",
        "speculative_backend": "dflash2",
        "max_draft_tokens": 7,
        "prefill_linear_mode": "w4a4",
        "prefill_linear_mode_effective": "w4a8",
        "prefill_linear_fallback": True,
        "prefill_linear_fallback_reason": "requires_mxfp4_model",
        "model_quant_format": "oq_affine",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": True,
        "prefill_sparse_minference_effective": True,
        "prefill_cache_blend": True,
        "prefill_ane_mode": "ultra_fp16_swiglu",
        "prefill_ane_mode_effective": "ultra_fp16_swiglu",
        "prefill_ane_inactive_reason": "",
        "prefill_ane_offload": True,
        "prefill_ane_active": True,
        "prefill_ane_split_alpha": 0.03,
        "prefill_w4a8_tokens": 4096,
        "prefill_w4a4_tokens": 0,
        "prefill_sage2_tokens": 4096,
        "prefill_minference_blocks_skipped": 120,
        "prefill_minference_blocks_total": 256,
        "prefill_cache_blend_hit_tokens": 2048,
        "prefill_cache_blend_recomputed_tokens": 384,
        "prefill_ane_offload_tokens": 4096,
        "prefill_ane_fallback_tokens": 0,
    }
    eff = engine.effective_load_settings()
    assert eff is not None
    assert eff["prefill_linear_mode"] == "w4a4"
    assert eff["prefill_linear_mode_effective"] == "w4a8"
    assert eff["prefill_linear_fallback"] is True
    assert eff["prefill_linear_fallback_reason"] == "requires_mxfp4_model"
    assert eff["model_quant_format"] == "oq_affine"
    assert eff["prefill_attention_mode"] == "sage2"
    assert eff["prefill_sparse_minference_effective"] is True
    assert eff["prefill_cache_blend"] is True
    assert eff["prefill_ane_mode"] == "ultra_fp16_swiglu"
    assert eff["prefill_ane_mode_effective"] == "ultra_fp16_swiglu"
    assert eff["prefill_ane_active"] is True
    assert eff["prefill_w4a8_tokens"] == 4096
    assert eff["prefill_w4a4_tokens"] == 0
    assert eff["prefill_minference_blocks_skipped"] == 120
    assert eff["prefill_cache_blend_hit_tokens"] == 2048
    assert eff["prefill_ane_offload_tokens"] == 4096


def test_detect_model_weight_format_info_quant_format_and_mxfp4_w4a4_support(
    tmp_path: Path,
) -> None:
    import struct
    import zlib

    def _write_miv(path: Path, *, magic: bytes, version: int, objects: list[dict]) -> None:
        schema = (
            ("omlx-m5-artifact-v2" if version == 2 else "omlx-m5-artifact-v1")
            if magic == b"OMLXM5A1"
            else ("miv-artifact-v2" if version == 2 else "miv-artifact-v1")
        )
        manifest = {
            "schema": schema,
            "container_version": version,
            "objects": objects,
        }
        raw_manifest = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
        manifest_crc = zlib.crc32(raw_manifest) & 0xFFFFFFFF
        payload_offset = ((64 + len(raw_manifest) + 4095) // 4096) * 4096
        prefix = struct.pack(
            "<8sIIQQQQII",
            magic,
            version,
            64,
            64,
            len(raw_manifest),
            payload_offset,
            0,
            len(objects),
            manifest_crc,
        )
        header_crc = zlib.crc32(prefix) & 0xFFFFFFFF
        flags = 3 if version == 2 else 0
        hdr = prefix + struct.pack("<II", header_crc, flags)
        path.write_bytes(hdr + raw_manifest + b"\x00" * (payload_offset - 64 - len(raw_manifest)))

    # 1) MXFP4 .miv v2 (MIV00001) with direct_bf16_v1 control/router projections -> "mxfp4"
    mxfp4_dir = tmp_path / "Qwen3.8-27B-mxfp4"
    mxfp4_dir.mkdir()
    _write_miv(
        mxfp4_dir / "model.miv",
        magic=b"MIV00001",
        version=2,
        objects=[
            {"name": "model.embed_tokens.weight", "format": "direct_bf16_v1"},
            {"name": "model.layers.0.linear_attn.in_proj_a", "format": "direct_bf16_v1"},
            {"name": "model.layers.0.mlp.gate", "format": "direct_bf16_v1"},
            {"name": "model.layers.0.mlp.gate_proj", "format": "mxfp4_e2m1_g32_e8m0_v1"},
            {"name": "model.layers.0.mlp.down_proj", "format": "mxfp4_e2m1_g32_e8m0_v1"},
            {"name": "mtp.layers.0.mlp.gate_proj", "format": "oq_affine_4bit_g64_v1"},
        ],
    )
    info_mxfp4 = m5_mod.detect_model_weight_format_info(mxfp4_dir, active_external_engine="m5rt")
    assert info_mxfp4["quant_format"] == "mxfp4"
    assert info_mxfp4["mxfp4_w4a4_supported"] is True

    # 2) oQ4e .omlxm5 v1 (OMLXM5A1) -> "oq_affine", mxfp4_w4a4_supported == False
    oq_dir = tmp_path / "Qwen3.8-27B-oQ4e"
    oq_dir.mkdir()
    _write_miv(
        oq_dir / "model.omlxm5",
        magic=b"OMLXM5A1",
        version=1,
        objects=[
            {"name": "model.layers.0.mlp.gate_proj", "format": "oq_affine_4bit_g64_v1"},
            {"name": "model.layers.0.mlp.down_proj", "format": "oq_affine_4bit_g64_v1"},
        ],
    )
    info_oq = m5_mod.detect_model_weight_format_info(oq_dir, active_external_engine="m5rt")
    assert info_oq["quant_format"] == "oq_affine"
    assert info_oq["mxfp4_w4a4_supported"] is False

    # 3) config.json fallback (no .miv file yet)
    cfg_mxfp4_dir = tmp_path / "safetensors-mxfp4"
    cfg_mxfp4_dir.mkdir()
    (cfg_mxfp4_dir / "config.json").write_text(
        json.dumps({"quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32}}),
        encoding="utf-8",
    )
    (cfg_mxfp4_dir / "model.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
    info_cfg_mxfp4 = m5_mod.detect_model_weight_format_info(cfg_mxfp4_dir)
    assert info_cfg_mxfp4["quant_format"] == "mxfp4"
    assert info_cfg_mxfp4["mxfp4_w4a4_supported"] is True

    cfg_affine_dir = tmp_path / "safetensors-oq4e"
    cfg_affine_dir.mkdir()
    (cfg_affine_dir / "config.json").write_text(
        json.dumps({"quantization": {"mode": "affine", "bits": 4, "group_size": 64}}),
        encoding="utf-8",
    )
    (cfg_affine_dir / "model.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
    info_cfg_oq = m5_mod.detect_model_weight_format_info(cfg_affine_dir)
    assert info_cfg_oq["quant_format"] == "oq_affine"
    assert info_cfg_oq["mxfp4_w4a4_supported"] is False


def test_inspect_mtp_manifest_accepts_miv_v1_and_v2_and_rejects_invalid_headers(
    tmp_path: Path, monkeypatch
) -> None:
    import struct
    import zlib

    def _write_container(
        path: Path,
        *,
        version: int,
        flags: int,
        object_names: list[str],
        magic: bytes = b"MIV00001",
        schema: str | None = None,
    ) -> Path:
        if schema is None:
            if magic == b"OMLXM5A1":
                schema = "omlx-m5-artifact-v2" if version == 2 else "omlx-m5-artifact-v1"
            else:
                schema = "miv-artifact-v2" if version == 2 else "miv-artifact-v1"
        manifest = {
            "schema": schema,
            "container_version": version,
            "objects": [{"name": name} for name in object_names],
        }
        raw_manifest = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
        manifest_crc = zlib.crc32(raw_manifest) & 0xFFFFFFFF
        payload_offset = ((64 + len(raw_manifest) + 4095) // 4096) * 4096
        prefix = struct.pack(
            "<8sIIQQQQII",
            magic,
            version,
            64,
            64,
            len(raw_manifest),
            payload_offset,
            0,
            len(object_names),
            manifest_crc,
        )
        header_crc = zlib.crc32(prefix) & 0xFFFFFFFF
        hdr = prefix + struct.pack("<II", header_crc, flags)
        path.write_bytes(hdr + raw_manifest + b"\x00" * (payload_offset - 64 - len(raw_manifest)))
        return path

    # Accept BOTH new MIV00001 + miv-artifact-v{1,2} AND legacy OMLXM5A1 + omlx-m5-artifact-v{1,2}
    for tag, magic in (("new", b"MIV00001"), ("legacy", b"OMLXM5A1")):
        v1_mtp = _write_container(
            tmp_path / f"{tag}_v1_mtp.miv",
            version=1,
            flags=0,
            object_names=[
                "language_model.model.layers.0.mlp.down_proj",
                "language_model.mtp.layers.0.mlp.down_proj",
            ],
            magic=magic,
        )
        v1_no_mtp = _write_container(
            tmp_path / f"{tag}_v1_no_mtp.miv",
            version=1,
            flags=0,
            object_names=["language_model.model.layers.0.mlp.down_proj"],
            magic=magic,
        )
        assert m5_mod.inspect_omlxm5_has_mtp_weights(v1_mtp) is True
        assert m5_mod.inspect_omlxm5_has_mtp_weights(v1_no_mtp) is False

        # v2 valid for every valid flag bitmask (0, 1=FWHT64, 2=NAX_LANE_V2, 3=both)
        for flags in (0, 1, 2, 3):
            v2_mtp = _write_container(
                tmp_path / f"{tag}_v2_f{flags}_mtp.miv",
                version=2,
                flags=flags,
                object_names=["language_model.mtp.layers.0.self_attn.o_proj"],
                magic=magic,
            )
            v2_no_mtp = _write_container(
                tmp_path / f"{tag}_v2_f{flags}_nomtp.miv",
                version=2,
                flags=flags,
                object_names=["language_model.model.layers.0.self_attn.o_proj"],
                magic=magic,
            )
            assert m5_mod.inspect_omlxm5_has_mtp_weights(v2_mtp) is True
            assert m5_mod.inspect_omlxm5_has_mtp_weights(v2_no_mtp) is False

    # Invalid headers/magics/version-schema mismatches must return None
    invalid_cases = [
        ("bad_v1_flags.miv", dict(version=1, flags=1)),
        ("bad_v2_flags.miv", dict(version=2, flags=4)),
        ("bad_v3.miv", dict(version=3, flags=0)),
        ("bad_alt_magic_v1.miv", dict(version=1, flags=0, magic=b"MIVART" + b"01", schema="miv-artifact-v1")),
        ("bad_alt_magic_v2.miv", dict(version=2, flags=3, magic=b"MIVART" + b"01", schema="miv-artifact-v2")),
        ("bad_random_magic.miv", dict(version=2, flags=0, magic=b"BADMAGIC", schema="miv-artifact-v2")),
        ("v1_new_v2_schema.miv", dict(version=1, flags=0, magic=b"MIV00001", schema="miv-artifact-v2")),
        ("v1_legacy_v2_schema.miv", dict(version=1, flags=0, magic=b"OMLXM5A1", schema="omlx-m5-artifact-v2")),
        ("v2_new_v1_schema.miv", dict(version=2, flags=3, magic=b"MIV00001", schema="miv-artifact-v1")),
        ("v2_legacy_v1_schema.miv", dict(version=2, flags=3, magic=b"OMLXM5A1", schema="omlx-m5-artifact-v1")),
        ("unknown_schema.miv", dict(version=2, flags=0, magic=b"MIV00001", schema="other-artifact-v2")),
    ]
    for fname, kwargs in invalid_cases:
        bad = _write_container(
            tmp_path / fname,
            object_names=["language_model.mtp.layers.0.mlp.down_proj"],
            **kwargs,
        )
        assert m5_mod.inspect_omlxm5_has_mtp_weights(bad) is None, f"expected None for {fname}"

    # Post-conversion header validator accepts BOTH MIV00001 and legacy OMLXM5A1, rejects invalid magics
    prov_root = tmp_path / "provider_magic_check"
    (prov_root / "bin").mkdir(parents=True)
    (prov_root / "metallib").mkdir(parents=True)
    (prov_root / "tools").mkdir(parents=True)
    fake_bin = _fake_daemon_script(prov_root / "bin" / "mseries_backendd", "miv-v0.01", protocol=2)
    (prov_root / "metallib" / "omlx_m5_qwen_ar.metallib").write_bytes(b"MTLB")
    fake_tool = prov_root / "tools" / "m5_artifact.py"
    monkeypatch.delenv("NINFER_METAL_ARTIFACT_TOOL", raising=False)

    for label, emitted_magic, should_pass in (
        ("new_magic", b"MIV00001", True),
        ("legacy_magic", b"OMLXM5A1", True),
        ("rejected_alt_magic", b"MIVART" + b"01", False),
    ):
        fake_tool.write_text(
            "import pathlib, sys\n"
            "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
            f"out.write_bytes({emitted_magic!r} + b'\\x00' * 128)\n",
            encoding="utf-8",
        )
        mdir = tmp_path / f"qwen_conv_{label}"
        mdir.mkdir()
        (mdir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
                }
            ),
            encoding="utf-8",
        )
        (mdir / "model.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
        if should_pass:
            res = m5_mod.convert_model_to_miv(
                mdir, source_repo=f"test/{label}", configured_binary=str(fake_bin)
            )
            assert res["already_existed"] is False
            assert (mdir / "model.miv").read_bytes()[:8] == emitted_magic
        else:
            with pytest.raises(ValueError, match="valid MIV00001/OMLXM5A1 header"):
                m5_mod.convert_model_to_miv(
                    mdir, source_repo=f"test/{label}", configured_binary=str(fake_bin)
                )
            assert not (mdir / "model.miv").exists()



