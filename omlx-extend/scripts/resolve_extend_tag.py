#!/usr/bin/env python3
"""Resolve oMLX-extend release tag names (`omlx-extend-vX.Y.Z` / `-mix1..N`).

Tag convention:
  - First release for upstream `jundot/omlx` tag `vX.Y.Z`:
      tag   = `omlx-extend-vX.Y.Z`
      title = `oMLX-extend vX.Y.Z`
  - Subsequent extend-only releases on the same upstream `vX.Y.Z`:
      tag   = `omlx-extend-vX.Y.Z-mix1`, `omlx-extend-vX.Y.Z-mix2`, ...
      title = `oMLX-extend vX.Y.Z-mix1`, `oMLX-extend vX.Y.Z-mix2`, ...
  - Next upstream release `vX.Y.(Z+1)`:
      resets cleanly to `omlx-extend-vX.Y.(Z+1)` (carrying all prior mix edits).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def resolve_extend_tag(
    upstream_tag: str,
    existing_tags: set[str],
    mode: str = "auto",
) -> dict[str, str | bool | int]:
    tag = upstream_tag.strip()
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?", tag):
        raise ValueError(f"Invalid upstream tag: {upstream_tag!r}")

    base_tag = f"omlx-extend-{tag}"
    mix_re = re.compile(rf"^{re.escape(base_tag)}-mix([1-9][0-9]*)$")

    mix_numbers: list[int] = []
    for t in existing_tags:
        m = mix_re.match(t.strip())
        if m:
            mix_numbers.append(int(m.group(1)))
    max_mix = max(mix_numbers) if mix_numbers else 0
    base_exists = base_tag in existing_tags

    if mode == "auto":
        extend_tag = base_tag
        already_released = base_exists
        is_mix = False
        mix_number = 0
    elif mode == "mix":
        if not base_exists and max_mix == 0:
            # Base release for this upstream version does not exist yet:
            # publish the base tag first instead of jumping straight to -mix1.
            extend_tag = base_tag
            already_released = False
            is_mix = False
            mix_number = 0
        else:
            mix_number = max_mix + 1
            extend_tag = f"{base_tag}-mix{mix_number}"
            already_released = False
            is_mix = True
    elif mode == "force":
        if max_mix > 0:
            mix_number = max_mix
            extend_tag = f"{base_tag}-mix{mix_number}"
            already_released = True
            is_mix = True
        else:
            mix_number = 0
            extend_tag = base_tag
            already_released = base_exists
            is_mix = False
    else:
        raise ValueError(f"Unsupported mode: {mode!r} (expected auto|mix|force)")

    release_title = (
        f"oMLX-extend {tag}-mix{mix_number}" if is_mix else f"oMLX-extend {tag}"
    )

    return {
        "upstream_tag": tag,
        "base_tag": base_tag,
        "extend_tag": extend_tag,
        "release_title": release_title,
        "already_released": already_released,
        "is_mix": is_mix,
        "mix_number": mix_number,
        "highest_existing_mix": max_mix,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Resolve oMLX-extend release tag.")
    parser.add_argument("--upstream-tag", required=True, help="Upstream tag (e.g. v0.8.3)")
    parser.add_argument(
        "--mode",
        choices=("auto", "mix", "force"),
        default="auto",
        help="Tag resolution mode (auto, mix, force)",
    )
    parser.add_argument(
        "--existing-tags-file",
        help="File with one existing tag/release name per line (use '-' for stdin)",
    )
    parser.add_argument(
        "--format",
        choices=("json", "github-output"),
        default="json",
        help="Output format",
    )
    args = parser.parse_args(argv[1:])

    existing: set[str] = set()
    if args.existing_tags_file:
        if args.existing_tags_file == "-":
            lines = sys.stdin.read().splitlines()
        else:
            lines = Path(args.existing_tags_file).read_text(encoding="utf-8").splitlines()
        existing = {line.strip() for line in lines if line.strip()}

    info = resolve_extend_tag(args.upstream_tag, existing, mode=args.mode)
    if args.format == "github-output":
        for k, v in info.items():
            if isinstance(v, bool):
                val = "true" if v else "false"
            else:
                val = str(v)
            print(f"{k}={val}")
    else:
        print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
