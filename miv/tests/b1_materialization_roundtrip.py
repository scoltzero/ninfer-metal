#!/usr/bin/env python3
"""B1.2 segmented materialization and non-owning view integration test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "miv" / "tools"))
sys.path.insert(0, str(ROOT / "miv" / "tests"))

import b0_artifact_roundtrip as b0
import m5_artifact as artifact


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_component_fixture(root: Path) -> Path:
    q4 = [index % 16 for index in range(64)]
    packed = b0.pack_codes(q4, 4)

    def affine(
        rows: int, leading: list[int]
    ) -> dict[str, tuple[str, list[int], bytes]]:
        return {
            "weight": ("U32", [*leading, 8], packed * rows),
            "scales": ("BF16", [*leading, 1], b0.bf16([0.5] * rows)),
            "biases": ("BF16", [*leading, 1], b0.bf16([-1.0] * rows)),
        }

    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    modules = {
        "language_model.model.linear": (2, [2]),
        "language_model.model.experts": (6, [2, 3]),
        "language_model.mtp.linear": (2, [2]),
        "vision_model.encoder.linear": (2, [2]),
        "language_model.draft_model.linear": (2, [2]),
    }
    for module, (rows, leading) in modules.items():
        for suffix, value in affine(rows, leading).items():
            tensors[f"{module}.{suffix}"] = value
    tensors["language_model.model.norm.weight"] = (
        "BF16",
        [2],
        b0.bf16([1.0, 2.0]),
    )
    return b0.write_model(
        root,
        "b1-component-fixture",
        {
            "model_type": "fixture",
            "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        },
        tensors,
    )


def inspect(inspector: Path, path: Path, *arguments: str) -> dict:
    output = subprocess.check_output([str(inspector), str(path), *arguments], text=True)
    return json.loads(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspector", type=Path, required=True)
    args = parser.parse_args()
    inspector = args.inspector.resolve()
    with tempfile.TemporaryDirectory(prefix="omlx-m5-b1-2-") as temporary:
        root = Path(temporary)
        source = make_component_fixture(root / "source")
        model = artifact.SafeTensorSource(source, "b1-component-fixture")
        output = root / "fixture.omlxm5"
        try:
            artifact.write_artifact(model, output, "fixture/b1", "oq_affine")
        finally:
            model.close()

        component_counts = {}
        for component in ("text", "vision", "mtp", "speculative"):
            record = inspect(inspector, output, "--components", component)
            expect(not record["materialized"], "plan-only inspection materialized data")
            expect(
                record["plan"]["selected_object_count"] > 0,
                f"{component} selection is empty",
            )
            component_counts[component] = record["plan"]["selected_object_count"]

        record = inspect(
            inspector,
            output,
            "--components",
            "text,mtp",
            "--max-segment-bytes",
            "512",
            "--materialize",
            "--verify-resident",
        )
        plan = record["plan"]
        stats = record["stats"]
        delta = record["registry_delta"]
        checks = record["view_checks"]
        expect(
            record["payload_verified"], "materialization skipped payload verification"
        )
        expect(record["materialized"], "fixture was not materialized")
        expect(plan["segment_count"] > 1, "forced segmented arena did not segment")
        expect(
            plan["resident_bytes"] == plan["upload_bytes"] + plan["padding_bytes"],
            "plan byte accounting is inconsistent",
        )
        expect(stats["uploaded_bytes"] == plan["upload_bytes"], "upload byte mismatch")
        expect(stats["resident_bytes"] == plan["resident_bytes"], "resident mismatch")
        expect(
            delta["buffer_count"] == plan["segment_count"], "segment ledger mismatch"
        )
        expect(
            delta["buffer_bytes"] == plan["resident_bytes"], "resident ledger mismatch"
        )
        expect(
            delta["upload_count"] == plan["placement_count"], "upload count mismatch"
        )
        expect(
            delta["uploaded_bytes"] == plan["upload_bytes"], "upload ledger mismatch"
        )
        expect(all(checks.values()), "one or more non-owning view checks failed")
        expect(
            plan["selected_object_count"]
            == component_counts["text"] + component_counts["mtp"],
            "component selection included an unrequested object",
        )

        failed = subprocess.run(
            [
                str(inspector),
                str(output),
                "--components",
                "text",
                "--max-segment-bytes",
                "256",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        expect(failed.returncode != 0, "planner accepted a plane larger than a segment")

        print(
            json.dumps(
                {
                    "schema": record["schema"],
                    "component_counts": component_counts,
                    "segments": plan["segment_count"],
                    "uploaded_bytes": stats["uploaded_bytes"],
                    "all_view_checks": True,
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
