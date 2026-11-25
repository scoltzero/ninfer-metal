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
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
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
        additional_special_tokens=["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
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
                            "phys_footprint_bytes": 128 * 1024 * 1024,
                        },
                    )
                elif method == "stats":
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
                            "phys_footprint_bytes": 128 * 1024 * 1024,
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
            assert cache_stats["cache_rates"]["cumulative"]["prefix_hits"] == 4
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

        # Stale 32768 cap removed in m5rt_memory_estimate()
        est_32k = m5rt_memory_estimate(model_dir, target, 32768, ModelSettings())
        est_256k = m5rt_memory_estimate(model_dir, target, 262144, ModelSettings())
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

        gs = GlobalSettings(base_path=tmp_path / "base")
        gs.scheduler.max_concurrent_requests = 4

        # 1) Stock MTP settings sync + global max_concurrent_requests default
        ms_mtp = ModelSettings(
            external_engine=M5RT_ENGINE_ID,
            mtp_enabled=True,
            mtp_fixed_depth=5,
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
        assert opts_mtp.draft_tokens == 5
        assert opts_mtp.max_concurrency == 4

        eng_mtp = M5SidecarEngine(
            model_name=str(model_dir),
            model_id="sync-mtp",
            target=target,
            log_dir=tmp_path / "logs",
            options=opts_mtp,
        )
        load_payload = eng_mtp._build_load_params()
        assert load_payload["speculative"] == {"backend": "mtp", "draft_tokens": 5}
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
        assert opts.max_context == 65536
        assert opts.kv_dtype == "affine4_g128"
        assert opts.rolling_prefix_cache is True
        assert opts.prefix_cache_bytes == 4 * (1 << 30)
        assert opts.sampling_contract == "legacy_rejection_v1"
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
        assert opts_mtp_auto.draft_tokens == 3
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
        assert opts_df_auto.draft_tokens == 4

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
            mtp_fixed_depth=5,
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
        assert opts_ui.draft_tokens == 5
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



