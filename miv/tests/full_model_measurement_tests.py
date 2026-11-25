"""Check timing semantics for batched speculative token events."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import m5_ultra_full_model as benchmark


class MeasurementTests(unittest.TestCase):
    def measure(self, batches, event_times, finished_at):
        native = SimpleNamespace(EventKind=SimpleNamespace(TOKENS=1, FINISHED=2))
        events = [SimpleNamespace(request_id=7, kind=1, tokens=t) for t in batches]
        events.append(
            SimpleNamespace(request_id=7, kind=2, finish_reason="length", error="")
        )
        engine = SimpleNamespace(poll_events=lambda _: events)
        # One timeout check, one time per nonempty token batch, then completion.
        with patch.object(
            benchmark.time,
            "perf_counter",
            side_effect=[100.0, *event_times, finished_at],
        ):
            return benchmark.collect(native, engine, 7, 30, start=99.0)

    def test_speculative_batch_excludes_whole_first_batch(self):
        result = self.measure([[1, 2, 3], [], [4, 5]], [101.0, 102.0], 103.0)
        self.assertEqual(result["ttft_seconds"], 2.0)
        self.assertEqual(result["request_output_tokens_per_second"], 1.25)
        self.assertEqual(result["decode_token_count"], 2)
        self.assertEqual(result["decode_tokens_per_second"], 2.0)

    def test_single_batch_does_not_invent_decode_rate(self):
        result = self.measure([[1, 2, 3]], [101.0], 103.0)
        self.assertIsNone(result["decode_tokens_per_second"])
        self.assertEqual(result["decode_token_count"], 0)

    def test_coalesced_arrivals_do_not_divide_by_zero(self):
        result = self.measure([[1], [2]], [101.0, 101.0], 102.0)
        self.assertIsNone(result["decode_tokens_per_second"])


if __name__ == "__main__":
    unittest.main()
