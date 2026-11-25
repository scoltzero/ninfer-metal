#!/usr/bin/env python3
"""Deterministic 3-way JSON merger for Xcode Localizable.xcstrings catalogs.

Usage:
    merge_xcstrings.py <base.xcstrings> <ours.xcstrings> <theirs.xcstrings> <output.xcstrings>

Preserves upstream (`theirs`) key ordering and Xcode `" : "` JSON formatting,
carries forward all disjoint keys added or updated in `ours` (oMLX-extend / mix
updates), and exits non-zero only when the exact same `(key, locale)` was
modified differently on both sides.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


_MISSING = object()


def _merge_entry(key: str, base: Any, ours: Any, theirs: Any) -> tuple[Any, list[str]]:
    if ours == base:
        return theirs, []
    if theirs == base:
        return ours, []
    if ours == theirs:
        return ours, []

    # Both sides modified the same key differently: attempt per-locale 3-way merge
    # when all non-missing sides are dicts.
    if (
        isinstance(ours, dict)
        and isinstance(theirs, dict)
        and (base is _MISSING or isinstance(base, dict))
    ):
        base_dict = base if isinstance(base, dict) else {}
        merged: dict[str, Any] = {}
        conflicts: list[str] = []

        # Preserve theirs top-level field order, then ours-only fields.
        field_order = list(theirs.keys()) + [k for k in ours.keys() if k not in theirs]
        for field in field_order:
            b_val = base_dict.get(field, _MISSING)
            o_val = ours.get(field, _MISSING)
            t_val = theirs.get(field, _MISSING)

            if field == "localizations" and isinstance(o_val, dict) and isinstance(t_val, dict):
                b_loc = b_val if isinstance(b_val, dict) else {}
                loc_merged: dict[str, Any] = {}
                loc_order = list(t_val.keys()) + [k for k in o_val.keys() if k not in t_val]
                for loc in loc_order:
                    bl = b_loc.get(loc, _MISSING)
                    ol = o_val.get(loc, _MISSING)
                    tl = t_val.get(loc, _MISSING)
                    if ol == bl:
                        val = tl
                    elif tl == bl:
                        val = ol
                    elif ol == tl:
                        val = ol
                    else:
                        conflicts.append(f"{key} [locale={loc}]")
                        continue
                    if val is not _MISSING:
                        loc_merged[loc] = val
                merged[field] = loc_merged
            else:
                if o_val == b_val:
                    val = t_val
                elif t_val == b_val:
                    val = o_val
                elif o_val == t_val:
                    val = o_val
                else:
                    conflicts.append(f"{key} [field={field}]")
                    continue
                if val is not _MISSING:
                    merged[field] = val

        return merged, conflicts

    return _MISSING, [key]


def merge_xcstrings(
    base_data: dict[str, Any],
    ours_data: dict[str, Any],
    theirs_data: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    conflicts: list[str] = []
    result: dict[str, Any] = {}

    for top_key in ("sourceLanguage", "strings", "version"):
        if top_key == "strings":
            base_s = base_data.get("strings", {})
            ours_s = ours_data.get("strings", {})
            theirs_s = theirs_data.get("strings", {})

            merged_strings: dict[str, Any] = {}
            key_order = list(theirs_s.keys()) + [k for k in ours_s.keys() if k not in theirs_s]
            for k in key_order:
                val, key_conflicts = _merge_entry(
                    k,
                    base_s.get(k, _MISSING),
                    ours_s.get(k, _MISSING),
                    theirs_s.get(k, _MISSING),
                )
                if key_conflicts:
                    conflicts.extend(key_conflicts)
                elif val is not _MISSING:
                    merged_strings[k] = val
            result["strings"] = merged_strings
        else:
            b = base_data.get(top_key, _MISSING)
            o = ours_data.get(top_key, _MISSING)
            t = theirs_data.get(top_key, _MISSING)
            if o == b:
                val = t
            elif t == b or o == t:
                val = o
            else:
                conflicts.append(f"<top-level:{top_key}>")
                continue
            if val is not _MISSING:
                result[top_key] = val

    return result, conflicts


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(
            "Usage: merge_xcstrings.py <base.xcstrings> <ours.xcstrings> <theirs.xcstrings> <output.xcstrings>",
            file=sys.stderr,
        )
        return 2

    base_p, ours_p, theirs_p, out_p = (Path(p) for p in argv[1:5])
    base_data = json.loads(base_p.read_text(encoding="utf-8"))
    ours_data = json.loads(ours_p.read_text(encoding="utf-8"))
    theirs_data = json.loads(theirs_p.read_text(encoding="utf-8"))

    merged, conflicts = merge_xcstrings(base_data, ours_data, theirs_data)
    if conflicts:
        print("xcstrings merge conflict on keys:", file=sys.stderr)
        for item in conflicts:
            print(f"  - {item}", file=sys.stderr)
        return 1

    out_p.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, separators=(",", " : ")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
