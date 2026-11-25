#!/usr/bin/env python3
"""Collect non-secret local environment facts for benchmark provenance."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import subprocess
from collections.abc import Sequence


def run(command: Sequence[str], *, env: dict[str, str] | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_revision(path: pathlib.Path) -> str | None:
    return run(["git", "-C", str(path), "rev-parse", "HEAD"])


def git_worktree_id(path: pathlib.Path) -> str | None:
    if git_revision(path) is None:
        return None
    digest = hashlib.sha256()
    diff = subprocess.run(
        ["git", "-C", str(path), "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    digest.update(diff)
    untracked = subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for raw in sorted(item for item in untracked if item):
        relative = raw.decode()
        content = (path / relative).read_bytes()
        digest.update(len(raw).to_bytes(4, "big"))
        digest.update(raw)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


def source_tree_id(root: pathlib.Path) -> str:
    paths = [
        path
        for path in (
            root / ".gitignore",
            root / "CMakeLists.txt",
            root / "CMakePresets.json",
        )
        if path.is_file()
    ]
    for directory in (
        root / "miv",
        root / "m5rt",
        root / "bench/schema",
        root / "bench/probes",
        root / "bench/prompts",
    ):
        if not directory.is_dir():
            continue
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    tools_dir = root / "miv/tools"
    if tools_dir.is_dir():
        paths.extend(path for path in tools_dir.glob("m5_*.py") if path.is_file())
        paths.extend(
            path
            for path in (
                tools_dir / "bootstrap_d0.sh",
                tools_dir / "reproduce_d0.sh",
                tools_dir / "reproduce_d1.sh",
                tools_dir / "reproduce_d2.sh",
                tools_dir / "reproduce_d3.sh",
                tools_dir / "reproduce_d4.sh",
                tools_dir / "validate_d0_assets.py",
                tools_dir / "validate_d1_inspection.py",
                tools_dir / "validate_d2_results.py",
                tools_dir / "validate_d3_probe.py",
            )
            if path.is_file()
        )
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    script_parent = pathlib.Path(__file__).resolve().parents[1]
    root = script_parent.parent if script_parent.name == "miv" else script_parent
    developer_dir = "/Applications/Xcode.app/Contents/Developer"
    xcode_env = dict(os.environ, DEVELOPER_DIR=developer_dir)
    payload = {
        "schema_version": 1,
        "source_id": source_tree_id(root),
        "workspace_commit": git_revision(root),
        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "machine_id": platform.node(),
        "architecture": platform.machine(),
        "macos": run(["sw_vers", "-productVersion"]),
        "macos_build": run(["sw_vers", "-buildVersion"]),
        "xcode": run(
            [f"{developer_dir}/usr/bin/xcodebuild", "-version"],
            env=xcode_env,
        ),
        "metal_compiler": run(["xcrun", "--find", "metal"], env=xcode_env),
        "macos_sdk": run(
            ["xcrun", "-sdk", "macosx", "--show-sdk-version"],
            env=xcode_env,
        ),
        "cmake": run(["cmake", "--version"]),
        "omlx_commit": git_revision(root / "omlx"),
        "omlx_worktree_id": git_worktree_id(root / "omlx"),
        "ninfer_commit": git_revision(root / "ninfer"),
        "ninfer_worktree_id": git_worktree_id(root / "ninfer"),
        "model_revision": (
            (root / "models/Jundot/Qwen3.8-27B-oQ4e-mtp/REVISION")
            .read_text(encoding="utf-8")
            .strip()
            if (root / "models/Jundot/Qwen3.8-27B-oQ4e-mtp/REVISION").is_file()
            else None
        ),
        "measurement_scope": "M4 correctness and harness only",
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
