#!/usr/bin/env python3
"""Exercise the production native Engine through nanobind."""

from __future__ import annotations

import json
import pathlib
import sys
import time


def collect(native, engine, request_id: int) -> tuple[list[int], object]:
    tokens: list[int] = []
    terminal = None
    deadline = time.monotonic() + 10.0
    while terminal is None and time.monotonic() < deadline:
        for event in engine.poll_events(20_000):
            assert event.request_id == request_id
            if event.kind == native.EventKind.TOKENS:
                tokens.extend(event.tokens)
            else:
                terminal = event
    assert terminal is not None
    return tokens, terminal


def load_options(native, fixtures: pathlib.Path, qwen_metallib: pathlib.Path):
    options = native.LoadOptions()
    real = fixtures.parent / "qwen-execution" / "affine"
    options.model_dir = str(real / "source")
    options.max_context = 1024
    options.path_overrides = {
        "artifact": str(real / "model.omlxm5"),
        "qwen_ar_metallib": str(qwen_metallib),
    }
    return options


def main() -> int:
    if len(sys.argv) != 7:
        raise SystemExit(
            "usage: python_binding_native.py EXTENSION_DIR TARGET_FIXTURES "
            "QWEN_METALLIB DRAFT_ARTIFACT DRAFT_CONFIG DFLASH_METALLIB"
        )
    extension_dir = pathlib.Path(sys.argv[1]).resolve()
    fixtures = pathlib.Path(sys.argv[2]).resolve()
    qwen_metallib = pathlib.Path(sys.argv[3]).resolve()
    draft_artifact = pathlib.Path(sys.argv[4]).resolve()
    draft_config = pathlib.Path(sys.argv[5]).resolve()
    dflash_metallib = pathlib.Path(sys.argv[6]).resolve()
    sys.path.insert(0, str(extension_dir))

    import omlx_m5_native as native

    options = load_options(native, fixtures, qwen_metallib)
    engine = native.Engine.create(options)
    request = native.SubmitRequest()
    request.prompt_tokens = [1, 7, 3, 9, 2]
    request.max_new_tokens = 8
    tokens, terminal = collect(native, engine, engine.submit(request))
    assert len(tokens) == 8
    assert terminal.kind == native.EventKind.FINISHED
    stats = engine.stats()
    assert stats.backend_kind == "native"
    assert stats.model_execution_kind == "qwen_dense"
    assert stats.actual_t == 1 and stats.actual_m == 1 and stats.actual_k == 0
    assert stats.artifact_id and stats.route_table_id
    ar_record = {
        "backend_kind": stats.backend_kind,
        "artifact_id": stats.artifact_id,
        "route_table_id": stats.route_table_id,
        "actual_t": stats.actual_t,
        "actual_m": stats.actual_m,
        "actual_k": stats.actual_k,
        "kernel_family": stats.kernel_family,
    }

    cancel_request = native.SubmitRequest()
    cancel_request.prompt_tokens = [1, 2]
    cancel_request.max_new_tokens = 512
    request_id = engine.submit(cancel_request)
    assert engine.cancel(request_id)
    _, cancelled = collect(native, engine, request_id)
    assert cancelled.kind == native.EventKind.ABORTED
    engine.shutdown()

    options = load_options(native, fixtures, qwen_metallib)
    options.draft_dir = str(draft_config.parent)
    options.speculative.backend = native.SpeculativeBackend.DFLASH2
    options.speculative.draft_tokens = 7
    # The historical fixture must remain rejected by the production factory.
    options.model_dir = str(fixtures / "ar-dense-oq")
    options.path_overrides = {
        **options.path_overrides,
        "artifact": str(fixtures / "ar-dense-oq.omlxm5"),
        "draft_artifact": str(draft_artifact),
        "dflash2_metallib": str(dflash_metallib),
    }
    try:
        native.Engine.create(options)
    except ValueError as error:
        assert "companion/target geometry mismatch" in str(error)
        dflash_record = {"control_fixture_rejected": True, "error": str(error)}
    else:
        raise AssertionError("production factory accepted a DFlash control fixture")
    options = load_options(native, fixtures, qwen_metallib)
    real_draft = fixtures.parent / "dflash-execution"
    options.draft_dir = str(real_draft / "source")
    options.speculative.backend = native.SpeculativeBackend.DFLASH2
    options.speculative.draft_tokens = 7
    options.path_overrides = {
        **options.path_overrides,
        "draft_artifact": str(real_draft / "model.omlxm5"),
    }
    engine = native.Engine.create(options)
    actual, terminal = collect(native, engine, engine.submit(request))
    assert actual == tokens and terminal.kind == native.EventKind.FINISHED
    stats = engine.stats()
    assert (
        stats.spec_rounds
        == stats.spec_target_verify_calls
        == stats.spec_replay_calls
        > 0
    )
    assert stats.spec_context_rows_projected > 0
    assert stats.spec_context_rows_reused > 0
    assert stats.ar_steps == 1 and stats.model_execution_kind == "qwen_dense"
    dflash_record.update(
        real_target_passed=True,
        one_verify_per_round=True,
        spec_rounds=stats.spec_rounds,
        spec_host_read_bytes=stats.spec_host_read_bytes,
    )
    engine.shutdown()
    print(
        json.dumps(
            {
                "schema": "omlx-m5-b6-python-binding-v1",
                "ar": ar_record,
                "dflash2": dflash_record,
                "cancelled": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
