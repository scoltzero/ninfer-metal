#!/usr/bin/env python3
"""Statistical Hardware Counter & Metal System Trace capture/analyzer CLI.

Wraps `xcrun xctrace record --template 'Metal System Trace'` + XML table export
(with automatic fallback when `xctrace`/Xcode GUI Instruments is unavailable on
CommandLineTools-only headless SSH hosts) and integrates `omlx_m5_hw_counter_probe`
(`MTLCounterSampleBuffer` + `MTL4CounterHeap`) into a compact JSON report.
Raw multi-hundred-MB `.trace` bundles are recorded in temporary storage outside
the repository and cleaned up automatically after statistical extraction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPT_PARENT = _SCRIPT_DIR.parent
ROOT = _SCRIPT_PARENT.parent if _SCRIPT_PARENT.name == "miv" else _SCRIPT_PARENT
sys.path.insert(0, str(_SCRIPT_DIR))

from m5_target_guard import hardware_facts  # noqa: E402


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def distribution_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "stddev": 0.0,
            "cv": 0.0,
        }
    ordered = sorted(float(v) for v in values)
    n = len(ordered)

    def pct(p: float) -> float:
        if n == 1:
            return ordered[0]
        rank = max(0.0, min(100.0, p)) / 100.0 * (n - 1)
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        frac = rank - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    mean_val = sum(ordered) / n
    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in ordered) / (n - 1)
        std_val = math.sqrt(variance)
    else:
        std_val = 0.0
    cv_val = (std_val / abs(mean_val)) if abs(mean_val) > 1e-12 else 0.0
    return {
        "count": n,
        "min": ordered[0],
        "p50": pct(50.0),
        "p90": pct(90.0),
        "p99": pct(99.0),
        "max": ordered[-1],
        "mean": mean_val,
        "stddev": std_val,
        "cv": cv_val,
    }


def classify_kernel_label(label: str, channel: str = "") -> str:
    lower = f"{channel} {label}".lower()
    if any(tok in lower for tok in ("moe", "expert", "router", "switch_mlp")):
        return "moe"
    if any(tok in lower for tok in ("nax", "tensorops", "w4a4", "fp4_native")):
        return "nax_gemm"
    if any(tok in lower for tok in ("attention", "rope", "split_kv", "kv_page")):
        return "attention"
    if "gdn" in lower or "deltanet" in lower:
        return "gdn"
    if any(tok in lower for tok in ("rmsnorm", "normalize", "norm")):
        return "rmsnorm_residual"
    if any(tok in lower for tok in ("blit", "copy", "fill", "resolve", "sync")):
        return "blit_copy_sync"
    if any(tok in lower for tok in ("vocab", "sample_logits", "position_sample", "embed")):
        return "vocab_sampling"
    if any(tok in lower for tok in ("qmv", "qmm", "mxfp4_small_t", "row_exact", "affine_lane", "mlp")):
        return "simd_gemv_gemm"
    if "compute" in channel.lower():
        return "other_compute"
    return "other"


def probe_xctrace_availability(developer_dir: str | None) -> dict[str, Any]:
    env = os.environ.copy()
    if developer_dir and Path(developer_dir).is_dir():
        env["DEVELOPER_DIR"] = developer_dir
    cmd = ["xcrun", "xctrace", "version"]
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=15, check=False
        )
    except FileNotFoundError as exc:
        return {
            "available": False,
            "version": None,
            "reason": f"xcrun not found: {exc}",
        }
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "version": None,
            "reason": "xcrun xctrace version timed out",
        }
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    if proc.returncode != 0:
        return {
            "available": False,
            "version": None,
            "reason": combined or f"xcrun xctrace exited with {proc.returncode}",
        }
    first_line = combined.splitlines()[0] if combined else "unknown"
    return {
        "available": True,
        "version": first_line,
        "reason": "ok",
    }


def _resolve_xml_cell(elem: ET.Element, registry: dict[tuple[str, str], ET.Element]) -> ET.Element:
    elem_id = elem.attrib.get("id")
    if elem_id is not None:
        registry[(elem.tag, elem_id)] = elem
        for child in elem:
            _resolve_xml_cell(child, registry)
        return elem
    ref_id = elem.attrib.get("ref")
    if ref_id is not None:
        return registry.get((elem.tag, ref_id), elem)
    for child in elem:
        _resolve_xml_cell(child, registry)
    return elem


def parse_xctrace_gpu_intervals_xml(
    xml_text: str, target_process_substr: str | None
) -> dict[str, Any]:
    if not xml_text.strip():
        return {"interval_count": 0}
    root = ET.fromstring(xml_text)
    table_node = root.find(".//node")
    if table_node is None:
        return {"interval_count": 0}
    schema = table_node.find("schema")
    mnemonics: list[str] = []
    if schema is not None:
        for col in schema.findall("col"):
            mnem = col.findtext("mnemonic") or ""
            mnemonics.append(mnem)

    col_idx = {name: idx for idx, name in enumerate(mnemonics)}
    registry: dict[tuple[str, str], ET.Element] = {}

    intervals: list[tuple[int, int, str, str, int | None]] = []
    for row in table_node.findall("row"):
        cells = [_resolve_xml_cell(child, registry) for child in list(row)]
        if len(cells) < 3:
            continue

        def cell_at(name: str) -> ET.Element | None:
            idx = col_idx.get(name)
            if idx is None or idx >= len(cells):
                return None
            c = cells[idx]
            return None if c.tag == "sentinel" else c

        start_el = cell_at("start")
        dur_el = cell_at("duration")
        if start_el is None or dur_el is None:
            continue
        try:
            start_ns = int((start_el.text or "0").strip())
            dur_ns = int((dur_el.text or "0").strip())
        except ValueError:
            continue
        if dur_ns <= 0:
            continue

        proc_el = cell_at("process")
        proc_fmt = (proc_el.attrib.get("fmt", "") if proc_el is not None else "")
        label_el = cell_at("event-label")
        label_fmt = (label_el.attrib.get("fmt", "") if label_el is not None else "")
        sub_el = cell_at("channel-subtitle")
        sub_fmt = (sub_el.attrib.get("fmt", "") if sub_el is not None else "")
        chan_el = cell_at("channel-name")
        chan_str = (
            (chan_el.text or chan_el.attrib.get("fmt", ""))
            if chan_el is not None
            else ""
        ).strip()

        if target_process_substr:
            haystack = f"{proc_fmt} {label_fmt}"
            if target_process_substr not in haystack:
                continue

        lat_el = cell_at("start-latency")
        lat_ns: int | None = None
        if lat_el is not None and (lat_el.text or "").strip().isdigit():
            lat_ns = int((lat_el.text or "0").strip())

        combined_label = f"{sub_fmt} {label_fmt}".strip()
        intervals.append((start_ns, dur_ns, chan_str, combined_label, lat_ns))

    if not intervals:
        return {"interval_count": 0}

    intervals.sort(key=lambda item: (item[0], item[0] + item[1]))

    # Compute merged active GPU timeline, idle encoder gaps, and concurrency overlap.
    merged: list[list[int]] = []
    idle_gaps_ns: list[float] = []
    sum_interval_ns = 0
    latencies_ns: list[float] = []
    family_ns: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    channel_ns: dict[str, int] = {}

    for start_ns, dur_ns, chan_str, combined_label, lat_ns in intervals:
        end_ns = start_ns + dur_ns
        sum_interval_ns += dur_ns
        if lat_ns is not None:
            latencies_ns.append(float(lat_ns))
        fam = classify_kernel_label(combined_label, chan_str)
        family_ns[fam] = family_ns.get(fam, 0) + dur_ns
        family_counts[fam] = family_counts.get(fam, 0) + 1
        ch_key = chan_str or "Unknown"
        channel_ns[ch_key] = channel_ns.get(ch_key, 0) + dur_ns

        if not merged:
            merged.append([start_ns, end_ns])
        else:
            prev_start, prev_end = merged[-1]
            if start_ns > prev_end:
                idle_gaps_ns.append(float(start_ns - prev_end))
                merged.append([start_ns, end_ns])
            else:
                merged[-1][1] = max(prev_end, end_ns)

    union_active_ns = sum(seg[1] - seg[0] for seg in merged)
    wall_window_ns = max(1, merged[-1][1] - merged[0][0])
    overlap_ratio = (
        float(sum_interval_ns) / float(union_active_ns)
        if union_active_ns > 0
        else 1.0
    )
    duty_cycle_pct = 100.0 * float(union_active_ns) / float(wall_window_ns)

    family_rows = [
        {
            "family": fam,
            "interval_count": family_counts[fam],
            "total_gpu_ns": ns_val,
            "total_gpu_ms": ns_val * 1e-6,
            "share_of_gpu_pct": (100.0 * ns_val / sum_interval_ns)
            if sum_interval_ns > 0
            else 0.0,
        }
        for fam, ns_val in sorted(
            family_ns.items(), key=lambda kv: kv[1], reverse=True
        )
    ]
    channel_rows = [
        {
            "channel": ch,
            "total_gpu_ns": ns_val,
            "total_gpu_ms": ns_val * 1e-6,
            "share_of_gpu_pct": (100.0 * ns_val / sum_interval_ns)
            if sum_interval_ns > 0
            else 0.0,
        }
        for ch, ns_val in sorted(
            channel_ns.items(), key=lambda kv: kv[1], reverse=True
        )
    ]

    return {
        "interval_count": len(intervals),
        "window_span_ns": wall_window_ns,
        "union_active_gpu_ns": union_active_ns,
        "sum_interval_gpu_ns": sum_interval_ns,
        "gpu_duty_cycle_pct": duty_cycle_pct,
        "concurrency_overlap_ratio": overlap_ratio,
        "encoder_idle_gap_ns": distribution_stats(idle_gaps_ns),
        "cpu_to_gpu_start_latency_ns": distribution_stats(latencies_ns),
        "kernel_family_breakdown": family_rows,
        "channel_breakdown": channel_rows,
    }


def parse_xctrace_toc(toc_xml: str) -> dict[str, Any]:
    if not toc_xml.strip():
        return {}
    root = ET.fromstring(toc_xml)
    run_el = root.find("./run")
    if run_el is None:
        return {}
    summary_el = run_el.find("./info/summary")
    proc_el = run_el.find("./info/target/process")
    dev_el = run_el.find("./info/target/device")
    schemas = [
        tbl.attrib.get("schema", "")
        for tbl in run_el.findall("./data/table")
        if tbl.attrib.get("schema")
    ]
    # Intentionally omit device UUID, user host name, and unrelated system processes!
    return {
        "template_name": summary_el.findtext("template-name")
        if summary_el is not None
        else None,
        "instruments_version": summary_el.findtext("instruments-version")
        if summary_el is not None
        else None,
        "duration_s": float(summary_el.findtext("duration") or 0.0)
        if summary_el is not None
        else 0.0,
        "target_process_name": proc_el.attrib.get("name")
        if proc_el is not None
        else None,
        "target_return_status": proc_el.attrib.get("return-exit-status")
        if proc_el is not None
        else None,
        "device_model": dev_el.attrib.get("model") if dev_el is not None else None,
        "os_version": dev_el.attrib.get("os-version") if dev_el is not None else None,
        "table_schema_count": len(schemas),
        "has_metal_gpu_intervals": "metal-gpu-intervals" in schemas,
        "has_metal_gpu_state_intervals": "metal-gpu-state-intervals" in schemas,
    }


def export_and_analyze_trace_bundle(
    trace_bundle: Path,
    developer_dir: str | None,
    target_process_name: str | None,
) -> dict[str, Any]:
    env = os.environ.copy()
    if developer_dir and Path(developer_dir).is_dir():
        env["DEVELOPER_DIR"] = developer_dir

    toc_proc = subprocess.run(
        ["xcrun", "xctrace", "export", "--input", str(trace_bundle), "--toc"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    toc_info = parse_xctrace_toc(toc_proc.stdout)
    proc_filter = target_process_name or toc_info.get("target_process_name")

    gpu_proc = subprocess.run(
        [
            "xcrun",
            "xctrace",
            "export",
            "--input",
            str(trace_bundle),
            "--xpath",
            '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]',
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    gpu_analysis = parse_xctrace_gpu_intervals_xml(gpu_proc.stdout, proc_filter)
    if gpu_analysis.get("interval_count", 0) == 0 and proc_filter:
        # Retry unfiltered if Metal 4 queue submissions aren't tagged with process name.
        gpu_analysis = parse_xctrace_gpu_intervals_xml(gpu_proc.stdout, None)
        gpu_analysis["process_filter_note"] = (
            "unfiltered_device_gpu_intervals_used_for_metal4_queue"
        )

    return {
        "toc": toc_info,
        "gpu_intervals": gpu_analysis,
    }


def record_and_analyze_xctrace(
    launch_command: list[str],
    developer_dir: str | None,
    keep_trace_dir: Path | None,
) -> dict[str, Any]:
    avail = probe_xctrace_availability(developer_dir)
    if not avail["available"]:
        return {
            "available": False,
            "recorded": False,
            "fallback_used": True,
            "reason": avail["reason"],
        }

    env = os.environ.copy()
    if developer_dir and Path(developer_dir).is_dir():
        env["DEVELOPER_DIR"] = developer_dir

    target_name = Path(launch_command[0]).name
    temp_ctx = tempfile.TemporaryDirectory(prefix="omlx_m5_xctrace_")
    try:
        if keep_trace_dir is not None:
            keep_trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = keep_trace_dir / f"{target_name}.trace"
            if trace_path.exists():
                shutil.rmtree(trace_path)
        else:
            trace_path = Path(temp_ctx.name) / f"{target_name}.trace"

        record_cmd = [
            "xcrun",
            "xctrace",
            "record",
            "--template",
            "Metal System Trace",
            "--output",
            str(trace_path),
            "--launch",
            "--",
            *launch_command,
        ]
        rec_proc = subprocess.run(
            record_cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if rec_proc.returncode != 0 or not trace_path.exists():
            msg = (rec_proc.stdout + "\n" + rec_proc.stderr).strip()
            return {
                "available": True,
                "version": avail["version"],
                "recorded": False,
                "fallback_used": True,
                "reason": msg or f"xctrace record exited with {rec_proc.returncode}",
            }

        raw_size_bytes = sum(
            f.stat().st_size for f in trace_path.rglob("*") if f.is_file()
        )
        parsed = export_and_analyze_trace_bundle(
            trace_path, developer_dir, target_name
        )
        return {
            "available": True,
            "version": avail["version"],
            "recorded": True,
            "fallback_used": False,
            "raw_trace_bytes_extracted": raw_size_bytes,
            "raw_trace_cleaned_up": keep_trace_dir is None,
            "kept_outside_repo_path": str(trace_path)
            if keep_trace_dir is not None
            else None,
            **parsed,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": True,
            "version": avail["version"],
            "recorded": False,
            "fallback_used": True,
            "reason": f"xctrace capture/export failed: {exc}",
        }
    finally:
        temp_ctx.cleanup()


def summarize_hw_counter_report(hw_data: dict[str, Any]) -> dict[str, Any]:
    caps = hw_data.get("device_counter_capabilities", {})
    summary: dict[str, Any] = {
        "device_name": caps.get("device_name"),
        "architecture_name": caps.get("architecture_name"),
        "apple_gpu_family": caps.get("apple_gpu_family"),
        "timestamp_counter_supported": caps.get("timestamp_counter_supported"),
        "stage_utilization_supported": caps.get("stage_utilization_supported"),
        "statistic_counter_supported": caps.get("statistic_counter_supported"),
        "metal4_counter_heap_supported": caps.get("metal4_counter_heap_supported"),
        "metal4_timestamp_frequency_hz": caps.get("metal4_timestamp_frequency_hz"),
        "zero_overhead_disabled_check": hw_data.get("zero_overhead_disabled_check"),
    }

    h0 = hw_data.get("h0_microbenchmarks")
    if isinstance(h0, dict):
        gap = h0.get("dispatch_gap_analysis") or {}
        nax_list = h0.get("nax_gemm_counter_sweep") or []
        nax_compact = []
        for entry in nax_list:
            if "gpu_ns" in entry:
                nax_compact.append(
                    {
                        "label": entry.get("label"),
                        "gpu_ms_p50": entry.get("gpu_ms_p50"),
                        "p50_effective_gbs": entry.get("p50_effective_gbs"),
                        "p50_effective_tflops": entry.get("p50_effective_tflops"),
                        "bottleneck_class": entry.get("bottleneck_class"),
                    }
                )
        summary["h0_summary"] = {
            "key_findings": h0.get("key_findings"),
            "intra_encoder_dispatch_gap_ns": gap.get("intra_encoder_dispatch_gap_ns"),
            "inter_encoder_boundary_gap_ns": gap.get("inter_encoder_boundary_gap_ns"),
            "encoder_coalescing_speedup": gap.get("encoder_coalescing_speedup"),
            "nax_sweep_highlights": nax_compact,
            "collector_family_summary": h0.get("collector_family_summary"),
        }

    model_bd = hw_data.get("model_hardware_breakdown")
    if isinstance(model_bd, dict):
        profiles = model_bd.get("profiles", {})
        summary["model_summary"] = {
            "artifact": model_bd.get("artifact"),
            "source_repo": model_bd.get("source_repo"),
            "resident_weight_bytes": model_bd.get("resident_weight_bytes"),
            "continuous_decode_loop": model_bd.get("continuous_decode_loop"),
            "decode_t1_bottleneck": profiles.get("decode_t1", {}).get(
                "bottleneck_class"
            ),
            "decode_t1_effective_gbs": profiles.get("decode_t1", {}).get(
                "monolithic_effective_weight_gbs"
            ),
            "decode_t1_1200gbs_utilization_pct": profiles.get("decode_t1", {}).get(
                "nominal_1200gbs_utilization_pct"
            ),
            "decode_t1_sublayer_breakdown": profiles.get("decode_t1", {}).get(
                "sublayer_breakdown"
            ),
            "verify_t8_sublayer_breakdown": profiles.get("verify_t8", {}).get(
                "sublayer_breakdown"
            ),
            "prefill_sublayer_breakdown": profiles.get("prefill", {}).get(
                "sublayer_breakdown"
            ),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--xctrace-mode",
        choices=("auto", "require", "off"),
        default="auto",
        help="Capture Metal System Trace via xcrun xctrace if available.",
    )
    parser.add_argument(
        "--input-trace",
        type=Path,
        default=None,
        help="Analyze an existing .trace bundle instead of recording a new one.",
    )
    parser.add_argument(
        "--keep-trace-outside-repo",
        type=Path,
        default=None,
        help="Optional directory OUTSIDE the git repo to retain raw .trace bundle.",
    )
    parser.add_argument(
        "--developer-dir",
        type=str,
        default=os.environ.get(
            "OMLX_M5_DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer"
        ),
    )
    parser.add_argument(
        "--hw-counter-probe",
        type=Path,
        default=None,
        help="Path to omlx_m5_hw_counter_probe executable.",
    )
    parser.add_argument(
        "--hw-counters-json",
        type=Path,
        default=None,
        help="Ingest an already-recorded hw_counter_probe JSON file.",
    )
    parser.add_argument(
        "--hw-counters-output",
        type=Path,
        default=None,
        help="Where to save the raw omlx_m5_hw_counter_probe JSON output.",
    )
    parser.add_argument("--mode", default="h0", choices=("enumerate", "h0", "model", "all"))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--model-artifact", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument("--verify-tokens", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=20)
    parser.add_argument("--stage-repeats", type=int, default=5)
    parser.add_argument(
        "--xctrace-launch",
        nargs=argparse.REMAINDER,
        default=None,
        help="Optional explicit command to record under xctrace (or pass after --).",
    )
    raw_argv = sys.argv[1:]
    trailing_launch: list[str] = []
    if "--" in raw_argv:
        sep = raw_argv.index("--")
        trailing_launch = raw_argv[sep + 1 :]
        raw_argv = raw_argv[:sep]
        if raw_argv and raw_argv[-1] == "--xctrace-launch":
            raw_argv = raw_argv[:-1]
    args = parser.parse_args(raw_argv)
    if trailing_launch:
        args.xctrace_launch = trailing_launch

    if args.keep_trace_outside_repo is not None:
        resolved_keep = args.keep_trace_outside_repo.resolve()
        if resolved_keep == ROOT or ROOT in resolved_keep.parents:
            raise ValueError(
                f"--keep-trace-outside-repo must lie outside repo root {ROOT}"
            )

    hw_counter_data: dict[str, Any] | None = None
    hw_counters_saved_path: Path | None = None

    if args.hw_counters_json is not None and args.hw_counters_json.is_file():
        hw_counter_data = json.loads(
            args.hw_counters_json.read_text(encoding="utf-8")
        )
        hw_counters_saved_path = args.hw_counters_json.resolve()
    elif args.hw_counter_probe is not None and args.hw_counter_probe.is_file():
        probe_cmd = [
            str(args.hw_counter_probe.resolve()),
            "--mode",
            args.mode,
            "--warmups",
            str(args.warmups),
            "--iterations",
            str(args.iterations),
        ]
        if args.model_artifact is not None:
            probe_cmd.extend(["--model-artifact", str(args.model_artifact.resolve())])
        if args.model_config is not None:
            probe_cmd.extend(["--model-config", str(args.model_config.resolve())])
            probe_cmd.extend(
                [
                    "--max-context",
                    str(args.max_context),
                    "--prefill-tokens",
                    str(args.prefill_tokens),
                    "--verify-tokens",
                    str(args.verify_tokens),
                    "--decode-steps",
                    str(args.decode_steps),
                    "--stage-repeats",
                    str(args.stage_repeats),
                ]
            )
        if args.hw_counters_output is not None:
            hw_counters_saved_path = args.hw_counters_output.resolve()
            probe_cmd.extend(["--output", str(hw_counters_saved_path)])
            subprocess.run(probe_cmd, check=True)
            hw_counter_data = json.loads(
                hw_counters_saved_path.read_text(encoding="utf-8")
            )
        else:
            proc = subprocess.run(
                probe_cmd, check=True, capture_output=True, text=True
            )
            hw_counter_data = json.loads(proc.stdout)

    xctrace_section: dict[str, Any]
    if args.input_trace is not None:
        parsed = export_and_analyze_trace_bundle(
            args.input_trace.resolve(), args.developer_dir, None
        )
        xctrace_section = {
            "available": True,
            "recorded": False,
            "analyzed_existing_trace": True,
            **parsed,
        }
    elif args.xctrace_mode == "off":
        xctrace_section = {
            "available": probe_xctrace_availability(args.developer_dir)["available"],
            "recorded": False,
            "fallback_used": True,
            "reason": "xctrace_mode=off",
        }
    else:
        launch_cmd = list(args.xctrace_launch or [])
        if launch_cmd and launch_cmd[0] == "--":
            launch_cmd = launch_cmd[1:]
        if not launch_cmd and args.hw_counter_probe is not None and args.hw_counter_probe.is_file():
            nax_probe = args.hw_counter_probe.parent / "omlx_m5_nax_qmm_probe"
            if nax_probe.is_file():
                launch_cmd = [
                    str(nax_probe),
                    "--m",
                    "8",
                    "--n",
                    "5120",
                    "--k",
                    "5120",
                    "--bits",
                    "4",
                    "--iterations",
                    "20",
                ]
            else:
                launch_cmd = [
                    str(args.hw_counter_probe.resolve()),
                    "--mode",
                    "h0",
                    "--warmups",
                    "2",
                    "--iterations",
                    "6",
                ]

        if launch_cmd:
            xctrace_section = record_and_analyze_xctrace(
                launch_cmd, args.developer_dir, args.keep_trace_outside_repo
            )
        else:
            avail = probe_xctrace_availability(args.developer_dir)
            xctrace_section = {
                "available": avail["available"],
                "recorded": False,
                "fallback_used": True,
                "reason": avail["reason"] if not avail["available"] else "no launch target specified",
            }

        if args.xctrace_mode == "require" and not xctrace_section.get("recorded"):
            raise RuntimeError(
                f"xctrace recording required but failed: {xctrace_section.get('reason')}"
            )

    has_hw_counters = bool(
        hw_counter_data
        and (
            hw_counter_data.get("device_counter_capabilities", {}).get(
                "timestamp_counter_supported"
            )
            or hw_counter_data.get("device_counter_capabilities", {}).get(
                "metal4_counter_heap_supported"
            )
        )
    )
    has_xctrace = bool(xctrace_section.get("recorded") or xctrace_section.get("analyzed_existing_trace"))
    evidence_valid = has_hw_counters or has_xctrace

    evidence_sources: list[str] = []
    if has_hw_counters:
        evidence_sources.append("metal_hw_counter_sampler")
    if has_xctrace:
        evidence_sources.append("xctrace_metal_system_trace")

    report: dict[str, Any] = {
        "schema": "omlx-m5-hw-trace-stats-v1",
        "attached": evidence_valid,
        "valid": evidence_valid,
        "evidence_sources": evidence_sources,
        "reason": "hardware_counter_and_trace_verified"
        if evidence_valid
        else "no_hardware_counter_or_xctrace_evidence",
        "hardware": hardware_facts(),
        "xctrace": xctrace_section,
        "hw_counters_summary": summarize_hw_counter_report(hw_counter_data)
        if hw_counter_data
        else None,
        "hw_counters_file": hw_counters_saved_path.name
        if hw_counters_saved_path is not None
        else None,
    }

    atomic_json(args.output.resolve(), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
