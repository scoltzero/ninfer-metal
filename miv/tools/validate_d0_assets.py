#!/usr/bin/env python3
"""Cross-check D0 fixtures against the fixed checkpoint analysis."""

from __future__ import annotations

import json
import pathlib


def main() -> int:
    miv_root = pathlib.Path(__file__).resolve().parents[1]
    repo_root = miv_root.parent if miv_root.name == "miv" else miv_root
    fixtures = miv_root / "tests" / "fixtures"

    def resolve_fixture(name: str, fallback: str) -> pathlib.Path:
        cand = fixtures / name
        if cand.is_file():
            return cand
        return repo_root / fallback

    shapes = json.loads(
        resolve_fixture(
            "qwen38_27b_oq4e_shapes.json", "bench/probes/qwen38_27b_oq4e_shapes.json"
        ).read_text(encoding="utf-8")
    )
    analysis = json.loads(
        resolve_fixture("analysis.json", "docs/p4/analysis.json").read_text(
            encoding="utf-8"
        )
    )["Jundot/Qwen3.8-27B-oQ4e-mtp"]
    assert shapes["model_revision"] == analysis["revision"]
    assert shapes["quantization"]["group_size"] == analysis["group_size"]
    assert shapes["quantization"]["bits"] == [4, 5]
    assert sum(item["tensor_count"] for item in shapes["shapes"]) == sum(
        analysis["by_bits_n"][bits] for bits in ("4", "5")
    )
    assert {item["bits"] for item in shapes["shapes"]} == {4, 5}
    assert shapes["m_values"] == list(range(1, 9)) + [16]
    assert len({item["id"] for item in shapes["shapes"]}) == len(shapes["shapes"])

    prompts = [
        json.loads(line)
        for line in resolve_fixture("d0.jsonl", "bench/prompts/d0.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    categories = {
        "code",
        "reasoning",
        "chat-story",
        "structured",
        "translation",
    }
    assert {item["category"] for item in prompts} == categories
    assert all(
        sum(item["category"] == category for item in prompts) >= 3
        for category in categories
    )
    assert len({item["id"] for item in prompts}) == len(prompts)
    print("D0 assets match the fixed checkpoint analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

