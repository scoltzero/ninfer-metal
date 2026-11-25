"""Unit tests for Apple Silicon SKU recognition, theoretical priors, and calibration overlays."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

MIV_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MIV_ROOT / "tools"))

import m5_silicon_profile as sp
import m5_target_guard as guard


class SiliconProfileTests(unittest.TestCase):
    def _facts(self, chip: str, gpu_cores: int | None, ram_gb: int, cpu_cores: int = 30):
        return {
            "chip": chip,
            "cpu_total_cores": cpu_cores,
            "cpu_perf_cores": None,
            "cpu_eff_cores": None,
            "gpu_core_count": gpu_cores,
            "agx_io_class": "AGXAcceleratorG17D",
            "metal_plugin_name": "AGXMetalG17D",
            "gpu_probe_source": "unit_test",
            "system_memory_bytes": ram_gb * sp.GIB,
            "system_memory_gib": float(ram_gb),
            "macos": "27.0",
            "macos_build": "27A000",
            "xcode": "Xcode 27.0",
            "macos_sdk": "27.0",
        }

    def test_recognizes_m5_ultra_64gpu_96g_binned(self):
        profile = sp.resolve_silicon_profile(
            self._facts("Apple M5 Ultra", 64, 96, cpu_cores=30),
            auto_load_calibration=False,
        )
        self.assertEqual(profile["profile_id"], "apple-m5-ultra-64gpu")
        self.assertEqual(profile["matched_by"], "exact_chip_and_gpu_cores")
        self.assertEqual(profile["result_namespace"], "m5-ultra-64gpu-96g")
        self.assertEqual(profile["parameter_origin"], "theoretical_prior")
        self.assertEqual(profile["calibration_status"], "uncalibrated_prior")
        self.assertFalse(profile["device_performance_qualified"])
        self.assertFalse(profile["production_enabled"])
        self.assertTrue(profile["target_flags"]["is_m5_ultra"])
        self.assertTrue(profile["target_flags"]["is_m5_ultra_binned_64gpu"])
        self.assertFalse(profile["target_flags"]["is_m5_ultra_full_core_80gpu"])
        self.assertEqual(profile["silicon_spec_prior"]["compute_scale_vs_ultra_80"], 0.80)
        self.assertEqual(profile["silicon_spec_prior"]["bandwidth_scale_vs_ultra_80"], 1.00)

        # 96 GiB memory admission check:
        # estimated working set = 72 GiB, OS headroom = 16 GiB => usable = 56 GiB.
        admission = profile["memory_budget"]["safe_max_concurrency"]
        # 27B + DFlash2 at 128K (19.39 + 15.52 + 0.5 = 35.41 GiB for B=1; B=2 = 50.93 GiB; B=4 exceeds)
        self.assertLessEqual(admission["qwen38-27b-oq4e-dflash2"]["128k"]["max_safe_concurrency"], 2)
        # 27B + DFlash2 at 32K fits multi-slot
        self.assertGreaterEqual(admission["qwen38-27b-oq4e-dflash2"]["32k"]["max_safe_concurrency"], 4)
        # 35B MoE AR at 128K fits full B=8 (20.13 + 8*3.19 + 0.5 = 46.15 <= 56)
        self.assertEqual(admission["qwen36-35b-a3b-oq4e-ar"]["128k"]["max_safe_concurrency"], 8)

    def test_recognizes_m5_ultra_80gpu_256g_full_core(self):
        profile = sp.resolve_silicon_profile(
            self._facts("Apple M5 Ultra", 80, 256, cpu_cores=36),
            auto_load_calibration=False,
        )
        self.assertEqual(profile["profile_id"], "apple-m5-ultra-80gpu")
        self.assertEqual(profile["result_namespace"], "m5-ultra-80gpu-256g")
        self.assertTrue(profile["target_flags"]["is_m5_ultra_full_core_80gpu"])
        self.assertFalse(profile["target_flags"]["is_m5_ultra_binned_64gpu"])
        self.assertEqual(profile["theoretical_dispatch_prior"]["linear_split_k_target_slices"], 8)
        self.assertEqual(profile["theoretical_dispatch_prior"]["attention_split_kv_partitions"], 16)

        admission = profile["memory_budget"]["safe_max_concurrency"]
        self.assertEqual(admission["qwen38-27b-oq4e-ar"]["128k"]["max_safe_concurrency"], 8)
        self.assertEqual(admission["qwen38-27b-oq4e-dflash2"]["64k"]["max_safe_concurrency"], 8)
        self.assertEqual(admission["qwen38-27b-oq4e-dflash2"]["128k"]["max_safe_concurrency"], 8)

    def test_recognizes_m5_family_tiers_and_dynamic_bin_fallback(self):
        cases = [
            ("Apple M5 Max", 40, 128, "apple-m5-max-40gpu", 0.50),
            ("Apple M5 Max", 32, 64, "apple-m5-max-32gpu", 0.40),
            ("Apple M5 Pro", 20, 64, "apple-m5-pro-20gpu", 0.25),
            ("Apple M5 Pro", 16, 48, "apple-m5-pro-16gpu", 0.20),
            ("Apple M5", 10, 32, "apple-m5-10gpu", 0.125),
            ("Apple M4", 10, 16, "apple-m4-dev", 0.0),
        ]
        for chip, gpus, ram, expected_id, expected_scale in cases:
            profile = sp.resolve_silicon_profile(
                self._facts(chip, gpus, ram), auto_load_calibration=False
            )
            self.assertEqual(profile["profile_id"], expected_id)
            self.assertEqual(
                profile["silicon_spec_prior"]["compute_scale_vs_ultra_80"],
                expected_scale,
            )

        # Unlisted binned M5 Ultra (e.g. hypothetical 72 GPU) synthesizes ratio prior
        dyn = sp.resolve_silicon_profile(
            self._facts("Apple M5 Ultra", 72, 192), auto_load_calibration=False
        )
        self.assertEqual(dyn["profile_id"], "apple-m5-ultra-72gpu")
        self.assertEqual(dyn["matched_by"], "dynamic_m5_ratio_prior")
        self.assertAlmostEqual(
            dyn["silicon_spec_prior"]["compute_scale_vs_ultra_80"], 0.90
        )

    def test_calibration_overlay_updates_effective_params_and_tracks_diff(self):
        facts = self._facts("Apple M5 Ultra", 64, 96, cpu_cores=30)
        base = sp.resolve_silicon_profile(facts, auto_load_calibration=False)
        template = sp.build_calibration_template(base)
        template["measured_at"] = "2026-11-19T23:00:00Z"
        template["measured_dispatch_overrides"]["linear_split_k_target_slices"] = 8
        template["measured_dispatch_overrides"]["attention_split_kv_min_context"] = 16384

        with tempfile.TemporaryDirectory() as tmp:
            cal_path = Path(tmp) / "apple-m5-ultra-64gpu.json"
            cal_path.write_text(json.dumps(template), encoding="utf-8")
            calibrated = sp.resolve_silicon_profile(
                facts, calibration_path=cal_path, auto_load_calibration=False
            )
            self.assertEqual(calibrated["calibration_status"], "measured_calibrated")
            self.assertEqual(calibrated["parameter_origin"], "measured_calibration_overlay")
            self.assertEqual(
                calibrated["effective_dispatch_parameters"]["linear_split_k_target_slices"],
                8,
            )
            self.assertEqual(
                calibrated["effective_dispatch_parameters"]["attention_split_kv_min_context"],
                16384,
            )
            # Unmodified keys retain prior values
            self.assertEqual(
                calibrated["effective_dispatch_parameters"]["dflash2_draft_tokens"],
                7,
            )
            diff = calibrated["calibration"]["prior_vs_measured_diff"]
            self.assertEqual(
                diff["linear_split_k_target_slices"], {"prior": 6, "measured": 8}
            )
            self.assertEqual(
                diff["attention_split_kv_min_context"],
                {"prior": 8192, "measured": 16384},
            )

    def test_target_guard_includes_silicon_profile_fields(self):
        facts = guard.hardware_facts()
        for required_key in (
            "chip",
            "gpu_core_count",
            "system_memory_bytes",
            "silicon_profile_id",
            "result_namespace",
            "calibration_status",
            "target_device",
            "measurement_scope",
        ):
            self.assertIn(required_key, facts)


if __name__ == "__main__":
    unittest.main()
