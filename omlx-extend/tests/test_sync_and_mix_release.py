"""Tests for oMLX-extend automatic 3-way upstream sync and omlx-extend-vX.Y.Z-mixN releases."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "omlx-extend" / "scripts"
SYNC_SCRIPT = SCRIPTS_DIR / "sync_omlx.sh"
MERGE_XCSTRINGS_SCRIPT = SCRIPTS_DIR / "merge_xcstrings.py"
RESOLVE_TAG_SCRIPT = SCRIPTS_DIR / "resolve_extend_tag.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


resolve_mod = _load_module("resolve_extend_tag", RESOLVE_TAG_SCRIPT)
merge_xc_mod = _load_module("merge_xcstrings", MERGE_XCSTRINGS_SCRIPT)


def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return res.stdout.strip()


class TestResolveExtendTag:
    def test_auto_mode_first_and_repeat_release(self):
        first = resolve_mod.resolve_extend_tag("v0.8.3", set(), mode="auto")
        assert first["extend_tag"] == "omlx-extend-v0.8.3"
        assert first["release_title"] == "oMLX-extend v0.8.3"
        assert first["already_released"] is False
        assert first["is_mix"] is False

        repeat = resolve_mod.resolve_extend_tag("v0.8.3", {"omlx-extend-v0.8.3"}, mode="auto")
        assert repeat["extend_tag"] == "omlx-extend-v0.8.3"
        assert repeat["already_released"] is True

    def test_mix_mode_increments_numerically_and_resets_on_new_upstream(self):
        # Before base exists, mix publishes base tag first instead of skipping to -mix1
        before_base = resolve_mod.resolve_extend_tag("v0.8.3", set(), mode="mix")
        assert before_base["extend_tag"] == "omlx-extend-v0.8.3"
        assert before_base["is_mix"] is False

        # Base exists -> -mix1
        mix1 = resolve_mod.resolve_extend_tag("v0.8.3", {"omlx-extend-v0.8.3"}, mode="mix")
        assert mix1["extend_tag"] == "omlx-extend-v0.8.3-mix1"
        assert mix1["release_title"] == "oMLX-extend v0.8.3-mix1"
        assert mix1["is_mix"] is True
        assert mix1["mix_number"] == 1

        # Existing mix1..mix9 -> mix10 (numeric sort, not lexicographic)
        existing = {"omlx-extend-v0.8.3"} | {f"omlx-extend-v0.8.3-mix{i}" for i in range(1, 10)}
        mix10 = resolve_mod.resolve_extend_tag("v0.8.3", existing, mode="mix")
        assert mix10["extend_tag"] == "omlx-extend-v0.8.3-mix10"
        assert mix10["mix_number"] == 10

        # Next upstream v0.8.4 resets cleanly to omlx-extend-v0.8.4
        next_upstream = resolve_mod.resolve_extend_tag("v0.8.4", existing, mode="auto")
        assert next_upstream["extend_tag"] == "omlx-extend-v0.8.4"
        assert next_upstream["already_released"] is False
        assert next_upstream["is_mix"] is False

    def test_force_mode_targets_latest_mix_if_present(self):
        existing = {"omlx-extend-v0.8.3", "omlx-extend-v0.8.3-mix1", "omlx-extend-v0.8.3-mix2"}
        forced = resolve_mod.resolve_extend_tag("v0.8.3", existing, mode="force")
        assert forced["extend_tag"] == "omlx-extend-v0.8.3-mix2"
        assert forced["already_released"] is True

    def test_sync_workflow_does_not_clobber_upstream_repo_slug(self):
        wf_text = (REPO_ROOT / ".github" / "workflows" / "sync-omlx-extend.yml").read_text(
            encoding="utf-8"
        )
        # UPSTREAM_VERSION defines UPSTREAM_REPO=https://github.com/jundot/omlx,
        # so sourcing it must happen inside a subshell to preserve owner/repo for gh api.
        assert 'CURRENT_TAG="$(' in wf_text
        assert "--latest=false" in wf_text
        assert "./omlx-extend/scripts/build_omlx_extend_dmg.sh" in wf_text
        assert "dist/omlx-extend/*.dmg" in wf_text
        rel_text = (REPO_ROOT / ".github" / "workflows" / "release-ninfer-metal.yml").read_text(
            encoding="utf-8"
        )
        assert '"!omlx-extend-*"' in rel_text


class TestMergeXcstrings:
    def test_disjoint_additions_and_locale_edits_merge_cleanly(self):
        base = {
            "sourceLanguage": "en",
            "strings": {
                "shared.title": {
                    "localizations": {
                        "en": {"stringUnit": {"state": "translated", "value": "Title"}}
                    }
                }
            },
            "version": "1.0",
        }
        ours = {
            "sourceLanguage": "en",
            "strings": {
                "shared.title": {
                    "localizations": {
                        "en": {"stringUnit": {"state": "translated", "value": "Title"}},
                        "zh-Hans": {"stringUnit": {"state": "translated", "value": "标题"}},
                    }
                },
                "microkernels.mix1.badge": {
                    "localizations": {
                        "en": {"stringUnit": {"state": "translated", "value": "Mix1 Badge"}}
                    }
                },
            },
            "version": "1.0",
        }
        theirs = {
            "sourceLanguage": "en",
            "strings": {
                "upstream.v084.banner": {
                    "localizations": {
                        "en": {"stringUnit": {"state": "translated", "value": "New in 0.8.4"}}
                    }
                },
                "shared.title": {
                    "localizations": {
                        "en": {"stringUnit": {"state": "translated", "value": "Title"}},
                        "ja": {"stringUnit": {"state": "translated", "value": "タイトル"}},
                    }
                },
            },
            "version": "1.0",
        }

        merged, conflicts = merge_xc_mod.merge_xcstrings(base, ours, theirs)
        assert conflicts == []
        assert list(merged["strings"].keys()) == [
            "upstream.v084.banner",
            "shared.title",
            "microkernels.mix1.badge",
        ]
        assert set(merged["strings"]["shared.title"]["localizations"].keys()) == {
            "en",
            "ja",
            "zh-Hans",
        }

    def test_true_same_key_same_locale_conflict_fails(self):
        base = {
            "sourceLanguage": "en",
            "strings": {
                "dup.key": {
                    "localizations": {"en": {"stringUnit": {"state": "translated", "value": "Old"}}}
                }
            },
            "version": "1.0",
        }
        ours = {
            "sourceLanguage": "en",
            "strings": {
                "dup.key": {
                    "localizations": {"en": {"stringUnit": {"state": "translated", "value": "Ours"}}}
                }
            },
            "version": "1.0",
        }
        theirs = {
            "sourceLanguage": "en",
            "strings": {
                "dup.key": {
                    "localizations": {"en": {"stringUnit": {"state": "translated", "value": "Theirs"}}}
                }
            },
            "version": "1.0",
        }
        _, conflicts = merge_xc_mod.merge_xcstrings(base, ours, theirs)
        assert conflicts == ["dup.key [locale=en]"]


class TestThreeWaySyncCarriesMixForward:
    def _make_upstream_repo(self, upstream_dir: Path) -> None:
        upstream_dir.mkdir(parents=True)
        _git(upstream_dir, "init", "-q")
        _git(upstream_dir, "config", "user.name", "Upstream")
        _git(upstream_dir, "config", "user.email", "up@example.com")

        # v0.8.3 base
        (upstream_dir / "omlx").mkdir(parents=True)
        (upstream_dir / "omlx" / "_version.py").write_text('__version__ = "0.8.3"\n', encoding="utf-8")
        (upstream_dir / "omlx" / "server.py").write_text(
            "def start():\n    return 'v0.8.3'\n", encoding="utf-8"
        )
        xc_rel = upstream_dir / "apps" / "omlx-mac" / "Resources" / "Localizable.xcstrings"
        xc_rel.parent.mkdir(parents=True)
        xc_base = {
            "sourceLanguage": "en",
            "strings": {
                "app.name": {
                    "localizations": {"en": {"stringUnit": {"state": "translated", "value": "oMLX"}}}
                }
            },
            "version": "1.0",
        }
        xc_rel.write_text(
            json.dumps(xc_base, ensure_ascii=False, indent=2, separators=(",", " : ")) + "\n",
            encoding="utf-8",
        )
        _git(upstream_dir, "add", "-A")
        _git(upstream_dir, "commit", "-q", "-m", "release v0.8.3")
        _git(upstream_dir, "tag", "v0.8.3")

        # v0.8.4 upstream release (updates server.py, _version.py, and Localizable.xcstrings)
        (upstream_dir / "omlx" / "_version.py").write_text('__version__ = "0.8.4"\n', encoding="utf-8")
        (upstream_dir / "omlx" / "server.py").write_text(
            "def start():\n    return 'v0.8.4'\n\ndef new_upstream_endpoint():\n    return True\n",
            encoding="utf-8",
        )
        xc_v084 = {
            "sourceLanguage": "en",
            "strings": {
                "app.name": {
                    "localizations": {"en": {"stringUnit": {"state": "translated", "value": "oMLX"}}}
                },
                "upstream.v084.feature": {
                    "localizations": {"en": {"stringUnit": {"state": "translated", "value": "New 0.8.4"}}}
                },
            },
            "version": "1.0",
        }
        xc_rel.write_text(
            json.dumps(xc_v084, ensure_ascii=False, indent=2, separators=(",", " : ")) + "\n",
            encoding="utf-8",
        )
        _git(upstream_dir, "add", "-A")
        _git(upstream_dir, "commit", "-q", "-m", "release v0.8.4")
        _git(upstream_dir, "tag", "v0.8.4")

    def test_sync_carries_mix_edits_binary_assets_and_xcstrings_to_next_upstream(
        self, tmp_path: Path
    ):
        upstream_dir = tmp_path / "upstream"
        self._make_upstream_repo(upstream_dir)

        # Build live monorepo omlx/ atop v0.8.3 + unpatched mix1 edits + binary PNG
        live_omlx = tmp_path / "live_omlx"
        _git(tmp_path, "clone", "--quiet", "--branch", "v0.8.3", str(upstream_dir), str(live_omlx))
        subprocess.run(["rm", "-rf", str(live_omlx / ".git")], check=True)

        # Mix1 edits made directly in live omlx/ (NOT exported to any static patch!)
        mix_mod = live_omlx / "omlx" / "engine" / "microkernel_manager.py"
        mix_mod.parent.mkdir(parents=True, exist_ok=True)
        mix_mod.write_text('MIX_VERSION = "mix1-hotfix"\n', encoding="utf-8")

        logo_png = live_omlx / "apps" / "omlx-mac" / "Resources" / "logo.png"
        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRmix1binary"
        logo_png.write_bytes(png_bytes)

        xc_live = live_omlx / "apps" / "omlx-mac" / "Resources" / "Localizable.xcstrings"
        xc_data = json.loads(xc_live.read_text(encoding="utf-8"))
        xc_data["strings"]["extend.mix1.label"] = {
            "localizations": {"en": {"stringUnit": {"state": "translated", "value": "Mix1 Label"}}}
        }
        xc_live.write_text(
            json.dumps(xc_data, ensure_ascii=False, indent=2, separators=(",", " : ")) + "\n",
            encoding="utf-8",
        )

        # Run sync_omlx.sh from v0.8.3 -> v0.8.4
        res = subprocess.run(
            [
                "bash",
                str(SYNC_SCRIPT),
                "--tag",
                "v0.8.4",
                "--base-tag",
                "v0.8.3",
                "--omlx-dir",
                str(live_omlx),
                "--upstream",
                str(upstream_dir),
                "--skip-tests",
                "--allow-dirty",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "3-way merge onto v0.8.4 succeeded" in res.stdout

        # 1) Upstream v0.8.4 updates arrived
        assert '__version__ = "0.8.4"' in (live_omlx / "omlx" / "_version.py").read_text(encoding="utf-8")
        assert "new_upstream_endpoint" in (live_omlx / "omlx" / "server.py").read_text(encoding="utf-8")

        # 2) All mix1 code, binary assets, and localization keys carried forward cleanly
        assert mix_mod.read_text(encoding="utf-8") == 'MIX_VERSION = "mix1-hotfix"\n'
        assert logo_png.read_bytes() == png_bytes
        merged_xc = json.loads(xc_live.read_text(encoding="utf-8"))
        assert "upstream.v084.feature" in merged_xc["strings"]
        assert "extend.mix1.label" in merged_xc["strings"]
        assert not (live_omlx / ".git").exists()

    def test_sync_fails_cleanly_on_real_code_conflict_without_touching_live_dir(
        self, tmp_path: Path
    ):
        upstream_dir = tmp_path / "upstream"
        self._make_upstream_repo(upstream_dir)

        live_omlx = tmp_path / "live_omlx"
        _git(tmp_path, "clone", "--quiet", "--branch", "v0.8.3", str(upstream_dir), str(live_omlx))
        subprocess.run(["rm", "-rf", str(live_omlx / ".git")], check=True)

        # Create conflicting edit on the exact same line upstream changed in v0.8.4
        server_py = live_omlx / "omlx" / "server.py"
        original_mix_text = "def start():\n    return 'mix-conflict-override'\n"
        server_py.write_text(original_mix_text, encoding="utf-8")

        res = subprocess.run(
            [
                "bash",
                str(SYNC_SCRIPT),
                "--tag",
                "v0.8.4",
                "--base-tag",
                "v0.8.3",
                "--omlx-dir",
                str(live_omlx),
                "--upstream",
                str(upstream_dir),
                "--skip-tests",
                "--allow-dirty",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 3
        assert "omlx/server.py" in res.stderr
        # Live directory was NOT corrupted or partially overwritten
        assert server_py.read_text(encoding="utf-8") == original_mix_text
        assert '__version__ = "0.8.3"' in (live_omlx / "omlx" / "_version.py").read_text(encoding="utf-8")

    def test_exported_spi_patch_matches_live_omlx_and_applies_cleanly(self, tmp_path: Path):
        patch_file = REPO_ROOT / "omlx-extend" / "patches" / "omlx-v0.8.3-provider-spi.patch"
        apply_script = SCRIPTS_DIR / "apply_spi_patch.sh"
        assert patch_file.is_file() and patch_file.stat().st_size > 0
        patch_text = patch_file.read_text(encoding="utf-8", errors="replace")
        assert "resolve_sidecar_lora_settings" in patch_text
        assert "estimate_sidecar_lora_bytes" in patch_text
        assert "lora_mode" in patch_text

        # Copy tracked omlx/ files into a clean git repo, reverse-apply the SPI patch to
        # reconstruct upstream v0.8.3, and verify apply_spi_patch.sh applies cleanly forward!
        stage = tmp_path / "omlx_stage"
        subprocess.run(
            [
                "rsync",
                "-ac",
                "--exclude=/.git/",
                "--exclude=/.venv/",
                "--exclude=/venv/",
                "--exclude=/build/",
                "--exclude=/dist/",
                "--exclude=/.pytest_cache/",
                "--exclude=/.mypy_cache/",
                "--exclude=/.ruff_cache/",
                "--exclude=/.mlx_cache/",
                "--exclude=/*.egg-info/",
                "--exclude=/packaging/_build/",
                "--exclude=/packaging/_export/",
                "--exclude=/packaging/_wheels/",
                "--exclude=/apps/omlx-mac/.build/",
                "--exclude=/apps/omlx-mac/.swiftpm/",
                "--exclude=/apps/omlx-mac/DerivedData/",
                "--exclude=/apps/omlx-mac/oMLX.xcodeproj/project.xcworkspace/xcshareddata/",
                "--exclude=/node_modules/",
                "--exclude=__pycache__/",
                "--exclude=*.pyc",
                "--exclude=*.pyo",
                "--exclude=.DS_Store",
                "--exclude=*.dylib",
                "--exclude=*.so",
                "--exclude=*.metallib",
                "--exclude=*.dmg",
                "--exclude=*.pkg",
                "--exclude=*.db",
                "--exclude=*.sqlite",
                "--exclude=*.sqlite3",
                "--exclude=*.log",
                f"{REPO_ROOT / 'omlx'}/",
                f"{stage}/",
            ],
            check=True,
        )
        _git(stage, "init", "-q")
        _git(stage, "config", "user.name", "Test")
        _git(stage, "config", "user.email", "test@example.com")
        _git(stage, "add", "-A")
        _git(stage, "commit", "-q", "-m", "live omlx-extend state")

        # Reverse-apply the exported patch to reach exact upstream v0.8.3 base
        subprocess.run(
            ["git", "-C", str(stage), "apply", "--index", "--reverse", "--binary", str(patch_file)],
            check=True,
            capture_output=True,
            text=True,
        )
        _git(stage, "commit", "-q", "-m", "upstream v0.8.3 base")

        # Forward check & apply via apply_spi_patch.sh
        chk = subprocess.run(
            [
                "bash",
                str(apply_script),
                "--omlx-dir",
                str(stage),
                "--patch",
                str(patch_file),
                "--check-only",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Patch applies cleanly" in chk.stdout

        app = subprocess.run(
            [
                "bash",
                str(apply_script),
                "--omlx-dir",
                str(stage),
                "--patch",
                str(patch_file),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Patch applied successfully" in app.stdout
        assert (stage / "omlx" / "engine" / "m5_sidecar.py").read_bytes() == (
            REPO_ROOT / "omlx" / "omlx" / "engine" / "m5_sidecar.py"
        ).read_bytes()

