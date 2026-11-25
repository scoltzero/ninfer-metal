#!/usr/bin/env python3
"""Exercise the nanobind D0 contract without importing oMLX."""

from __future__ import annotations

import pathlib
import sys
import time


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python_binding_smoke.py EXTENSION_DIR")
    sys.path.insert(0, str(pathlib.Path(sys.argv[1]).resolve()))

    import omlx_m5_native as native

    fake = native.FakeBackendOptions()
    fake.event_batch_size = 2
    speculative = native.SpeculativeOptions()
    speculative.backend = native.SpeculativeBackend.DFLASH2
    speculative.draft_tokens = 7
    load = native.LoadOptions()
    load.speculative = speculative
    engine = native.Engine.create_fake(load_options=load, fake_options=fake)

    request = native.SubmitRequest()
    request.prompt_tokens = [40]
    request.max_new_tokens = 3
    request_id = engine.submit(request)

    tokens: list[int] = []
    terminal = None
    deadline = time.monotonic() + 2.0
    while terminal is None and time.monotonic() < deadline:
        for event in engine.poll_events(20_000):
            assert event.request_id == request_id
            if event.kind == native.EventKind.TOKENS:
                tokens.extend(event.tokens)
            else:
                terminal = event

    assert tokens == [41, 42, 43], tokens
    assert terminal is not None
    assert terminal.kind == native.EventKind.FINISHED
    assert terminal.finish_reason == "length"
    assert engine.stats().completed == 1
    engine.shutdown()
    print("python binding smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
