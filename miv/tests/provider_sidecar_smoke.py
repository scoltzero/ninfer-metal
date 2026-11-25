#!/usr/bin/env python3
"""Exercise fake, native AR, native DFlash2, pipelining, dual-conn push streaming, and stale-socket recovery over the sidecar RPC."""

from __future__ import annotations

import json
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


class Client:
    def __init__(self, path: Path):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        deadline = time.monotonic() + 10
        while True:
            try:
                self.socket.connect(str(path))
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        self.reader = self.socket.makefile("r", encoding="utf-8")
        self.writer = self.socket.makefile("w", encoding="utf-8")
        self.next_id = 1

    def call(self, method: str, params: dict | None = None) -> dict:
        request_id = self.next_id
        self.next_id += 1
        json.dump(
            {"id": request_id, "method": method, "params": params or {}},
            self.writer,
        )
        self.writer.write("\n")
        self.writer.flush()
        line = self.reader.readline()
        assert line, f"unexpected EOF waiting for {method} response"
        response = json.loads(line)
        assert response["id"] == request_id
        if not response["ok"]:
            raise RuntimeError(response["error"]["message"])
        return response["result"]

    def close(self):
        self.reader.close()
        self.writer.close()
        self.socket.close()


def collect(client: Client, request_id: int) -> tuple[list[int], dict]:
    tokens: list[int] = []
    terminal = None
    deadline = time.monotonic() + 30
    while terminal is None and time.monotonic() < deadline:
        for event in client.call("poll", {"max_wait_us": 20_000})["events"]:
            assert event["request_id"] == request_id
            if event["kind"] == "tokens":
                tokens.extend(event["tokens"])
            else:
                terminal = event
    assert terminal is not None
    return tokens, terminal


def test_pipelined_jsonl_and_reconnect(daemon: Path) -> None:
    """Verify (a) multi-line JSONL in a single sendall() and transient disconnect survival."""
    with tempfile.TemporaryDirectory(prefix="omlx-mseries-pipeline-") as temporary:
        socket_path = Path(temporary) / "provider.sock"
        process = subprocess.Popen(
            [str(daemon), "--socket", str(socket_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        raw_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        deadline = time.monotonic() + 10
        while True:
            try:
                raw_sock.connect(str(socket_path))
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

        try:
            pipelined_requests = [
                {"id": 1, "method": "hello", "params": {}},
                {
                    "id": 2,
                    "method": "load",
                    "params": {
                        "fake": True,
                        "max_concurrency": 2,
                        "max_context": 1024,
                        "prefix_cache_bytes": 4096,
                        "batch_mode": True,
                        "sampling_contract": "legacy_rejection_v1",
                        "fake_options": {"event_batch_size": 2, "step_delay_us": 0},
                    },
                },
                {"id": 3, "method": "capacity", "params": {}},
                {"id": 4, "method": "stats", "params": {}},
            ]
            payload = "".join(
                json.dumps(item) + "\n" for item in pipelined_requests
            ).encode("utf-8")
            # Send all 4 JSONL frames in a single sendall() so recv() reads them together.
            raw_sock.sendall(payload)

            reader = raw_sock.makefile("r", encoding="utf-8")
            responses = [json.loads(reader.readline()) for _ in range(4)]
            assert [resp["id"] for resp in responses] == [1, 2, 3, 4], responses
            assert all(resp["ok"] for resp in responses), responses
            assert "subscribe_events" in responses[0]["result"]["capabilities"]
            assert responses[0]["result"]["pid"] == process.pid
            assert responses[0]["result"]["phys_footprint_bytes"] > 0
            assert responses[2]["result"]["max_concurrency"] == 2
            assert responses[2]["result"]["phys_footprint_bytes"] > 0
            assert responses[3]["result"]["phys_footprint_bytes"] > 0

            # Disconnect first client without shutting down; daemon + loaded engine must stay alive.
            reader.close()
            raw_sock.close()

            reconnected = Client(socket_path)
            try:
                stats_after = reconnected.call("stats")
                assert stats_after["backend_kind"] == "fake"
                assert stats_after["pid"] == process.pid
                assert stats_after["phys_footprint_bytes"] > 0
                reconnected.call("shutdown")
            finally:
                reconnected.close()
        finally:
            raw_sock.close()
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            assert process.returncode == 0, (stdout, stderr)
            assert not socket_path.exists()


def test_dual_connection_subscribe_and_cancel(daemon: Path) -> None:
    """Verify (b) concurrent dual-connection subscribe_events push streaming + non-blocking control + cancel."""
    with tempfile.TemporaryDirectory(prefix="omlx-mseries-dualconn-") as temporary:
        socket_path = Path(temporary) / "provider.sock"
        process = subprocess.Popen(
            [str(daemon), "--socket", str(socket_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        control = Client(socket_path)
        event_conn = Client(socket_path)
        try:
            hello = control.call("hello")
            assert "subscribe_events" in hello["capabilities"]
            assert hello["phys_footprint_bytes"] > 0

            load_stats = control.call(
                "load",
                {
                    "fake": True,
                    "max_concurrency": 2,
                    "max_context": 1024,
                    "fake_options": {"event_batch_size": 2, "step_delay_us": 4_000},
                },
            )
            assert load_stats["phys_footprint_bytes"] > 0

            # Subscribe on connection 2 with a 500ms poll interval so we can prove:
            # 1) control RPCs on connection 1 never wait on connection 2's poll_events, and
            # 2) condition-variable wakeups push events immediately without 500ms lag.
            sub_ack = event_conn.call(
                "subscribe_events", {"poll_interval_us": 500_000}
            )
            assert sub_ack == {"subscribed": True}

            pushed_frames: list[dict] = []
            saw_eof = threading.Event()
            first_finish = threading.Event()
            cancel_abort = threading.Event()
            lock = threading.Lock()
            req1_id_box: list[int] = []
            req2_id_box: list[int] = []

            def pump_events() -> None:
                while True:
                    line = event_conn.reader.readline()
                    if not line:
                        break
                    frame = json.loads(line)
                    event = frame["event"]
                    with lock:
                        pushed_frames.append(event)
                        if event["kind"] == "eof":
                            saw_eof.set()
                            break
                        if (
                            req1_id_box
                            and event.get("request_id") == req1_id_box[0]
                            and event["kind"] == "finished"
                        ):
                            first_finish.set()
                        if (
                            req2_id_box
                            and event.get("request_id") == req2_id_box[0]
                            and event["kind"] == "aborted"
                        ):
                            cancel_abort.set()

            pump_thread = threading.Thread(target=pump_events, daemon=True)
            pump_thread.start()

            # Control RPCs must return immediately while event_conn is parked in poll_events(500ms).
            t0 = time.monotonic()
            cap = control.call("capacity")
            stats = control.call("stats")
            control_elapsed = time.monotonic() - t0
            assert control_elapsed < 0.2, f"control blocked on event stream: {control_elapsed}s"
            assert cap["accepting"] is True
            assert cap["phys_footprint_bytes"] > 0
            assert stats["phys_footprint_bytes"] > 0

            with lock:
                req1_id = control.call(
                    "submit", {"prompt_tokens": [10, 20], "max_new_tokens": 6}
                )["request_id"]
                req1_id_box.append(req1_id)
                req2_id = control.call(
                    "submit", {"prompt_tokens": [30, 40], "max_new_tokens": 250}
                )["request_id"]
                req2_id_box.append(req2_id)

            assert control.call("cancel", {"request_id": req2_id})["cancelled"] is True
            assert first_finish.wait(timeout=10.0), "timed out waiting for req1 finish push"
            assert cancel_abort.wait(timeout=10.0), "timed out waiting for req2 abort push"

            control.call("shutdown")
            assert saw_eof.wait(timeout=5.0), "timed out waiting for subscribe_events eof frame"
            pump_thread.join(timeout=5.0)

            req1_tokens: list[int] = []
            req1_terminal = None
            req2_terminal = None
            for ev in pushed_frames:
                if ev.get("request_id") == req1_id:
                    if ev["kind"] == "tokens":
                        req1_tokens.extend(ev["tokens"])
                    else:
                        req1_terminal = ev
                elif ev.get("request_id") == req2_id and ev["kind"] != "tokens":
                    req2_terminal = ev

            assert len(req1_tokens) == 6, req1_tokens
            assert req1_terminal is not None and req1_terminal["kind"] == "finished"
            assert req2_terminal is not None and req2_terminal["kind"] == "aborted"
            assert pushed_frames[-1] == {"kind": "eof"}
        finally:
            control.close()
            event_conn.close()
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            assert process.returncode == 0, (stdout, stderr)
            assert not socket_path.exists()


def test_stale_socket_recovery_and_signals(daemon: Path) -> None:
    """Verify (c) active socket protection, stale-socket recovery after SIGKILL, and clean SIGTERM unlink."""
    with tempfile.TemporaryDirectory(prefix="omlx-mseries-stale-") as temporary:
        socket_path = Path(temporary) / "provider.sock"
        proc1 = subprocess.Popen(
            [str(daemon), "--socket", str(socket_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            c1 = Client(socket_path)
            hello1 = c1.call("hello")
            assert hello1["pid"] == proc1.pid
            c1.close()

            # Starting a second daemon on a LIVE socket must fail without harming proc1.
            conflict = subprocess.run(
                [str(daemon), "--socket", str(socket_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert conflict.returncode != 0
            assert "Unix socket path already exists" in conflict.stderr

            # Confirm proc1 is still alive and serving connections after the conflict probe.
            c1_verify = Client(socket_path)
            assert c1_verify.call("hello")["pid"] == proc1.pid
            c1_verify.close()

            # Simulate crash with SIGKILL leaving a stale socket file on disk.
            proc1.send_signal(signal.SIGKILL)
            proc1.communicate(timeout=10)
            assert socket_path.exists(), "SIGKILL must leave stale socket behind"
        finally:
            if proc1.poll() is None:
                proc1.kill()
                proc1.communicate()

        # New daemon must probe stale socket (ECONNREFUSED), unlink it, and bind cleanly
        # (also exercising positional socket path form alongside --socket PATH).
        proc2 = subprocess.Popen(
            [str(daemon), str(socket_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            c2 = Client(socket_path)
            hello2 = c2.call("hello")
            assert hello2["pid"] == proc2.pid
            assert hello2["phys_footprint_bytes"] > 0

            # Exercise subscriber disconnect + SIGPIPE immunity without killing daemon.
            sub_tmp = Client(socket_path)
            assert sub_tmp.call("subscribe_events", {"poll_interval_us": 10_000}) == {
                "subscribed": True
            }
            sub_tmp.close()
            time.sleep(0.03)
            assert c2.call("hello")["pid"] == proc2.pid
            c2.close()

            # SIGTERM must shut down cleanly (rc=0) and unlink the socket via SocketPathGuard.
            proc2.send_signal(signal.SIGTERM)
            stdout2, stderr2 = proc2.communicate(timeout=10)
            assert proc2.returncode == 0, (stdout2, stderr2)
            assert not socket_path.exists(), "SIGTERM must unlink socket_path cleanly"
        finally:
            if proc2.poll() is None:
                proc2.kill()
                proc2.communicate()


def run_session(
    daemon: Path, load: dict, *, expected: dict, load_error: str = ""
) -> None:
    with tempfile.TemporaryDirectory(prefix="omlx-mseries-sidecar-") as temporary:
        socket_path = Path(temporary) / "provider.sock"
        process = subprocess.Popen(
            [str(daemon), "--socket", str(socket_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        client = Client(socket_path)
        try:
            hello = client.call("hello")
            assert hello["provider_id"] == "io.ninfer.metal"
            assert hello["provider_version"] == "miv-v0.01"
            assert hello["codename"] == "miv"
            assert hello["semver"] == "0.0.1"
            assert hello["core_abi"] == 1
            assert hello["protocol_version"] == 2
            for cap in (
                "ar",
                "mtp",
                "dflash2",
                "kv_affine4_g128",
                "rolling_prefix_cache",
                "trim_cache",
                "memory_budget_bytes",
                "set_scheduler_hint",
                "subscribe_events",
            ):
                assert cap in hello["capabilities"], (cap, hello["capabilities"])
            assert hello["kv_formats"] == ["bf16", "affine4_g128"]
            assert "mtp_enabled" in hello["supported_standard_settings"]
            assert "turboquant_kv_scheme" in hello["supported_standard_settings"]
            schema_by_key = {item["key"]: item for item in hello["options_schema"]}
            assert schema_by_key["max_concurrency"]["label"] == "Concurrency (B/C)"
            assert schema_by_key["prefix_cache_gb"]["type"] == "bytes_gb"
            assert schema_by_key["prefix_cache_gb"]["min"] == 0
            assert schema_by_key["prefix_cache_gb"]["max"] == 128
            assert schema_by_key["sampling_contract"]["default"] == "legacy_rejection_v1"
            assert schema_by_key["artifact_path"]["type"] == "string"
            assert schema_by_key["artifact_path"]["default"] == ""
            assert schema_by_key["draft_artifact_path"]["type"] == "string"
            assert schema_by_key["draft_artifact_path"]["default"] == ""
            assert hello["pid"] == process.pid
            assert hello["phys_footprint_bytes"] > 0
            if load_error:
                try:
                    client.call("load", load)
                except RuntimeError as error:
                    assert load_error in str(error)
                else:
                    raise AssertionError(
                        "production sidecar accepted a control fixture"
                    )
                client.call("shutdown")
                return
            load_result = client.call("load", load)
            assert load_result["pid"] == process.pid
            assert load_result["phys_footprint_bytes"] > 0
            request = {
                "prompt_tokens": [1, 7, 3, 9, 2],
                "max_new_tokens": 8,
                "sampling": {"temperature": 0, "top_p": 1, "seed": 7},
            }
            request_id = client.call("submit", request)["request_id"]
            tokens, terminal = collect(client, request_id)
            assert len(tokens) == 8
            assert terminal["kind"] == "finished"
            stats = client.call("stats")
            assert stats["pid"] == process.pid
            assert stats["phys_footprint_bytes"] > 0
            for key, value in expected.items():
                assert stats[key] == value, (key, stats[key], value)
            capacity = client.call("capacity")
            assert capacity["max_concurrency"] == 1
            assert capacity["pid"] == process.pid
            assert capacity["phys_footprint_bytes"] > 0

            hint = client.call("set_scheduler_hint", {"gpu_contended": True})
            assert hint["gpu_contended"] is True
            hint_off = client.call("set_scheduler_hint", {"gpu_contended": False})
            assert hint_off["gpu_contended"] is False
            trimmed = client.call("trim_cache", {"target_free_bytes": 1024})
            assert trimmed["freed_bytes"] >= 0
            assert trimmed["resident_bytes"] > 0

            blocker_id = client.call(
                "submit",
                {"prompt_tokens": [1, 2], "max_new_tokens": 512},
            )["request_id"]
            cancel_id = client.call(
                "submit",
                {"prompt_tokens": [3, 4], "max_new_tokens": 8},
            )["request_id"]
            assert client.call("cancel", {"request_id": cancel_id})["cancelled"]
            cancelled = None
            deadline = time.monotonic() + 30
            while cancelled is None and time.monotonic() < deadline:
                for event in client.call("poll", {"max_wait_us": 20_000})["events"]:
                    if event["request_id"] == cancel_id and event["kind"] != "tokens":
                        cancelled = event
            assert cancelled is not None and cancelled["kind"] == "aborted"
            assert blocker_id != cancel_id
            client.call("shutdown")
        finally:
            client.close()
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            assert process.returncode == 0, (stdout, stderr)
            assert not socket_path.exists()


def main() -> int:
    if len(sys.argv) != 7:
        raise SystemExit(
            "usage: provider_sidecar_smoke.py DAEMON TARGET_FIXTURES "
            "QWEN_METALLIB DRAFT_ARTIFACT DRAFT_CONFIG DFLASH_METALLIB"
        )
    daemon, fixtures, qwen_metallib, draft_artifact, draft_config, dflash_lib = (
        Path(value).resolve() for value in sys.argv[1:]
    )
    version = subprocess.check_output([str(daemon), "--version"], text=True)
    assert "protocol=2" in version and "miv-v0.01" in version, version

    test_pipelined_jsonl_and_reconnect(daemon)
    test_dual_connection_subscribe_and_cancel(daemon)
    test_stale_socket_recovery_and_signals(daemon)

    run_session(
        daemon,
        {
            "fake": True,
            "max_context": 1024,
            "fake_options": {"event_batch_size": 3, "step_delay_us": 1000},
        },
        expected={"backend_kind": "fake", "actual_k": 0},
    )
    base = {
        "model_dir": str(fixtures / "ar-dense-oq"),
        "max_context": 1024,
        "path_overrides": {
            "artifact": str(fixtures / "ar-dense-oq.omlxm5"),
            "qwen_ar_metallib": str(qwen_metallib),
        },
    }
    run_session(
        daemon,
        {
            **base,
            "model_dir": str(fixtures.parent / "qwen-execution/affine/source"),
            "path_overrides": {
                **base["path_overrides"],
                "artifact": str(fixtures.parent / "qwen-execution/affine/model.omlxm5"),
            },
        },
        expected={
            "backend_kind": "native",
            "model_execution_kind": "qwen_dense",
            "actual_t": 1,
            "actual_k": 0,
        },
    )
    run_session(
        daemon,
        {
            **base,
            "draft_dir": str(draft_config.parent),
            "speculative": {"backend": "dflash2", "draft_tokens": 7},
            "path_overrides": {
                **base["path_overrides"],
                "draft_artifact": str(draft_artifact),
                "dflash2_metallib": str(dflash_lib),
            },
        },
        expected={
            "backend_kind": "native",
            "model_execution_kind": "control_fixture",
            "actual_t": 8,
            "actual_m": 8,
            "actual_k": 7,
            "kernel_family": "dflash2_metal4_icb",
        },
        load_error="companion/target geometry mismatch",
    )
    run_session(
        daemon,
        {
            **base,
            "model_dir": str(fixtures.parent / "qwen-execution/affine/source"),
            "draft_dir": str(fixtures.parent / "dflash-execution/source"),
            "speculative": {"backend": "dflash2", "draft_tokens": 7},
            "path_overrides": {
                **base["path_overrides"],
                "artifact": str(fixtures.parent / "qwen-execution/affine/model.omlxm5"),
                "draft_artifact": str(
                    fixtures.parent / "dflash-execution/model.omlxm5"
                ),
            },
        },
        expected={
            "backend_kind": "native",
            "model_execution_kind": "qwen_dense",
            "kernel_family": "qwen_dflash2_forward_a16_metal4",
        },
    )
    print(json.dumps({"schema": "omlx-mseries-sidecar-smoke-v1", "passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
