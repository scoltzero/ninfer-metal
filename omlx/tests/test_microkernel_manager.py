"""Tests for native Python microkernel registry, download, hot-update, enable toggle, and delete."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin import routes as admin_routes
from omlx.engine import m5_sidecar, microkernel_manager
from omlx.engine.microkernel_manager import (
    MicrokernelManagerError,
    delete_provider,
    install_or_update_provider,
    is_microkernel_enabled,
    list_catalog,
    reset_manager_state_for_tests,
    set_provider_enabled,
)
from omlx.settings import GlobalSettings


def _make_spi_tarball(
    dest_tar: Path,
    *,
    provider_id: str,
    engine_id: str,
    version: str,
    display_name: str = "Community Llama Metal",
    model_types: list[str] | None = None,
    extra_bad_member: str | None = None,
) -> str:
    """Create a valid SPI v2 provider tarball and return its SHA-256 hex digest."""
    manifest = {
        "provider_id": provider_id,
        "engine_id": engine_id,
        "display_name": display_name,
        "display_version": version,
        "version": version,
        "protocol_version": 2,
        "entrypoint": "bin/mock_daemon",
        "supported_topologies": ["llama3_gqa_v1"],
        "supported_model_types": model_types or ["llama", "mistral"],
        "artifact_extensions": [".omlxm5"],
    }
    daemon_script = (
        "#!/usr/bin/env bash\n"
        f'if [[ "$1" == "--version" ]]; then echo "{engine_id} {version} protocol=2"; exit 0; fi\n'
        "exit 0\n"
    ).encode("utf-8")

    with tarfile.open(dest_tar, "w:gz") as tf:
        root_name = f"{engine_id}-{version}"
        m_bytes = json.dumps(manifest).encode("utf-8")
        mi = tarfile.TarInfo(name=f"{root_name}/provider.json")
        mi.size = len(m_bytes)
        mi.mode = 0o644
        tf.addfile(mi, io.BytesIO(m_bytes))

        bi = tarfile.TarInfo(name=f"{root_name}/bin/mock_daemon")
        bi.size = len(daemon_script)
        bi.mode = 0o755
        tf.addfile(bi, io.BytesIO(daemon_script))

        if extra_bad_member is not None:
            bad_bytes = b"evil"
            ei = tarfile.TarInfo(name=extra_bad_member)
            ei.size = len(bad_bytes)
            ei.mode = 0o644
            tf.addfile(ei, io.BytesIO(bad_bytes))

    return hashlib.sha256(dest_tar.read_bytes()).hexdigest()


def _write_latest(
    dest: Path, *, provider_id: str, version: str, tar_path: Path, sha256: str
) -> None:
    payload = {
        "provider_id": provider_id,
        "protocol_version": 2,
        "display_version": version,
        "semver": version.lstrip("v"),
        "assets": {
            "macos-arm64": {
                "filename": tar_path.name,
                "url": tar_path.as_uri(),
                "sha256": sha256,
            }
        },
    }
    dest.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_manager_state():
    reset_manager_state_for_tests()
    yield
    reset_manager_state_for_tests()


def _make_settings(tmp_path: Path) -> GlobalSettings:
    settings_path = tmp_path / "settings.json"
    gs = GlobalSettings()
    gs._settings_file = settings_path  # type: ignore[attr-defined]
    gs.save = lambda: settings_path.write_text(  # type: ignore[method-assign]
        json.dumps(gs.to_dict()), encoding="utf-8"
    )
    return gs


def test_full_lifecycle_refresh_download_hot_update_toggle_and_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    providers_dir = tmp_path / "Providers"
    providers_dir.mkdir()
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()

    pid = "io.community.llama-metal"
    eid = "llama-metal"
    tar_v1 = remote_dir / "llama-metal-v0.1.0.tar.gz"
    sha_v1 = _make_spi_tarball(
        tar_v1, provider_id=pid, engine_id=eid, version="v0.1.0"
    )
    latest_path = remote_dir / "latest.json"
    _write_latest(
        latest_path, provider_id=pid, version="v0.1.0", tar_path=tar_v1, sha256=sha_v1
    )

    registry_path = remote_dir / "microkernels.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": "omlx-microkernel-registry-v1",
                "providers": [
                    {
                        "provider_id": pid,
                        "engine_id": eid,
                        "display_name": "Community Llama Metal",
                        "path": "B",
                        "current_version": "v0.1.0",
                        "latest_manifest_url": latest_path.as_uri(),
                        "supported_model_types": ["llama", "mistral"],
                        "supported_topologies": ["llama3_gqa_v1"],
                        "artifact_extensions": [".omlxm5"],
                        "protocol_version": 2,
                        "min_macos_version": "27.0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(providers_dir))
    monkeypatch.setenv("OMLX_MICROKERNEL_REGISTRY_URL", registry_path.as_uri())
    monkeypatch.setenv("OMLX_MICROKERNEL_REGISTRY_FILE", str(registry_path))
    # Prevent picking up host dev build miv binary during isolated test
    monkeypatch.setattr(
        microkernel_manager,
        "m5rt_status",
        lambda gs, pool=None: {"available": False},
    )

    gs = _make_settings(tmp_path)

    # 1. Before refresh: zero network fetch, catalog loaded from local file, nothing installed yet
    initial = list_catalog(gs, None, refresh=False)
    assert initial["refreshed"] is False
    assert len(initial["catalog"]) == 1
    assert initial["catalog"][0]["supported_model_types"] == ["llama", "mistral"]
    assert initial["catalog"][0]["installed"] is False
    assert initial["installed"] == []

    # 2. Download & install v0.1.0
    res_install = install_or_update_provider(pid, gs, None)
    assert res_install["success"] is True
    assert res_install["version"] == "v0.1.0"
    assert (providers_dir / pid / "current").is_symlink()
    assert (providers_dir / pid / "current").resolve().name == "v0.1.0"
    assert len(res_install["installed"]) == 1
    row = res_install["installed"][0]
    assert row["enabled"] is True
    assert row["managed"] is True
    assert row["can_delete"] is True
    assert row["can_update"] is False  # Not refreshed with a newer release yet!

    # 3. Publish v0.2.0 remotely; BEFORE refresh, can_update MUST remain False!
    tar_v2 = remote_dir / "llama-metal-v0.2.0.tar.gz"
    sha_v2 = _make_spi_tarball(
        tar_v2, provider_id=pid, engine_id=eid, version="v0.2.0"
    )
    _write_latest(
        latest_path, provider_id=pid, version="v0.2.0", tar_path=tar_v2, sha256=sha_v2
    )
    before_refresh = list_catalog(gs, None, refresh=False)
    assert before_refresh["installed"][0]["can_update"] is False
    assert before_refresh["installed"][0]["has_update"] is False

    # 4. Click Refresh -> now has_update & can_update light up!
    after_refresh = list_catalog(gs, None, refresh=True)
    assert after_refresh["refreshed"] is True
    inst_row = after_refresh["installed"][0]
    assert inst_row["version"] == "v0.1.0"
    assert inst_row["remote_version"] == "v0.2.0"
    assert inst_row["has_update"] is True
    assert inst_row["can_update"] is True

    # 5. Hot-update to v0.2.0 -> atomic symlink flip + .previous_version recorded + can_update resets False
    res_upd = install_or_update_provider(pid, gs, None)
    assert res_upd["version"] == "v0.2.0"
    assert (providers_dir / pid / "current").resolve().name == "v0.2.0"
    assert (providers_dir / pid / ".previous_version").read_text().strip() == "v0.1.0"
    assert res_upd["installed"][0]["version"] == "v0.2.0"
    assert res_upd["installed"][0]["can_update"] is False

    # 6. Checkbox disable persists to settings.json and disables routing helper
    res_dis = set_provider_enabled(pid, False, gs, None)
    assert res_dis["enabled"] is False
    assert res_dis["installed"][0]["enabled"] is False
    assert pid in gs.integrations.disabled_microkernels
    assert is_microkernel_enabled(gs, provider_id=pid, engine_id=eid) is False

    res_en = set_provider_enabled(pid, True, gs, None)
    assert res_en["enabled"] is True
    assert is_microkernel_enabled(gs, provider_id=pid, engine_id=eid) is True

    # 7. Delete blocked with 409 if an active model is served on this core; succeeds when unload_active=True
    fake_engine = m5_sidecar.M5SidecarEngine.__new__(m5_sidecar.M5SidecarEngine)
    fake_engine.external_engine_id = eid
    fake_engine.provider_id = pid
    fake_engine.get_stats = lambda: {
        "process_memory_bytes": 1024,
        "memory_budget_bytes": 4096,
        "log_path": "/tmp/mock.log",
    }
    unload_mock = AsyncMock()

    async def _unload(mid: str) -> None:
        await unload_mock(mid)
        fake_pool._entries.pop(mid, None)

    fake_pool = SimpleNamespace(
        _entries={"llama-3-8b": SimpleNamespace(engine=fake_engine)},
        unload_model=_unload,
    )

    import asyncio

    with pytest.raises(MicrokernelManagerError) as exc_info:
        asyncio.run(delete_provider(pid, gs, fake_pool, unload_active=False))
    assert exc_info.value.status_code == 409
    assert exc_info.value.active_models == ["llama-3-8b"]
    assert (providers_dir / pid).is_dir()

    res_del = asyncio.run(delete_provider(pid, gs, fake_pool, unload_active=True))
    assert res_del["deleted"] is True
    assert res_del["unloaded_models"] == ["llama-3-8b"]
    assert not (providers_dir / pid).exists()
    assert res_del["installed"] == []


def test_security_rejects_sha256_mismatch_and_tar_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    providers_dir = tmp_path / "Providers"
    providers_dir.mkdir()
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()

    pid = "io.community.bad-core"
    eid = "bad-core"
    tar_ok = remote_dir / "bad-core-v0.1.0.tar.gz"
    _make_spi_tarball(tar_ok, provider_id=pid, engine_id=eid, version="v0.1.0")
    latest_path = remote_dir / "latest.json"
    _write_latest(
        latest_path,
        provider_id=pid,
        version="v0.1.0",
        tar_path=tar_ok,
        sha256="0" * 64,  # wrong hash!
    )
    registry_path = remote_dir / "microkernels.json"
    registry_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "provider_id": pid,
                        "engine_id": eid,
                        "display_name": "Bad Core",
                        "path": "B",
                        "latest_manifest_url": latest_path.as_uri(),
                        "protocol_version": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(providers_dir))
    monkeypatch.setenv("OMLX_MICROKERNEL_REGISTRY_URL", registry_path.as_uri())
    monkeypatch.setenv("OMLX_MICROKERNEL_REGISTRY_FILE", str(registry_path))
    gs = _make_settings(tmp_path)

    with pytest.raises(MicrokernelManagerError, match="SHA-256 mismatch"):
        install_or_update_provider(pid, gs, None)
    assert not (providers_dir / pid).exists()

    # Now valid SHA-256 but tar-slip ../escape entry
    tar_slip = remote_dir / "bad-core-v0.1.1.tar.gz"
    sha_slip = _make_spi_tarball(
        tar_slip,
        provider_id=pid,
        engine_id=eid,
        version="v0.1.1",
        extra_bad_member="../escaped.txt",
    )
    _write_latest(
        latest_path,
        provider_id=pid,
        version="v0.1.1",
        tar_path=tar_slip,
        sha256=sha_slip,
    )
    with pytest.raises(MicrokernelManagerError, match="unsafe tar path"):
        install_or_update_provider(pid, gs, None)
    assert not (providers_dir / pid).exists()
    assert not (tmp_path / "escaped.txt").exists()


def test_admin_microkernel_http_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    providers_dir = tmp_path / "Providers"
    providers_dir.mkdir()
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()

    pid = "io.community.gemma-metal"
    eid = "gemma-metal"
    tar_v1 = remote_dir / "gemma-metal-v1.0.0.tar.gz"
    sha_v1 = _make_spi_tarball(
        tar_v1,
        provider_id=pid,
        engine_id=eid,
        version="v1.0.0",
        display_name="Gemma Metal 4",
        model_types=["gemma3", "gemma4"],
    )
    latest_path = remote_dir / "latest.json"
    _write_latest(
        latest_path, provider_id=pid, version="v1.0.0", tar_path=tar_v1, sha256=sha_v1
    )
    registry_path = remote_dir / "microkernels.json"
    registry_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "provider_id": pid,
                        "engine_id": eid,
                        "display_name": "Gemma Metal 4",
                        "path": "B",
                        "current_version": "v1.0.0",
                        "latest_manifest_url": latest_path.as_uri(),
                        "supported_model_types": ["gemma3", "gemma4"],
                        "protocol_version": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("OMLX_PROVIDERS_DIR", str(providers_dir))
    monkeypatch.setenv("OMLX_MICROKERNEL_REGISTRY_URL", registry_path.as_uri())
    monkeypatch.setenv("OMLX_MICROKERNEL_REGISTRY_FILE", str(registry_path))
    monkeypatch.setattr(
        microkernel_manager, "m5rt_status", lambda gs, pool=None: {"available": False}
    )

    gs = _make_settings(tmp_path)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: None)

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = lambda: True
    client = TestClient(app)

    r_get = client.get("/admin/api/microkernels")
    assert r_get.status_code == 200
    assert r_get.json()["refreshed"] is False
    assert r_get.json()["catalog"][0]["provider_id"] == pid

    r_ref = client.post("/admin/api/microkernels/refresh")
    assert r_ref.status_code == 200
    assert r_ref.json()["refreshed"] is True

    r_inst = client.post(f"/admin/api/microkernels/{pid}/install")
    assert r_inst.status_code == 200
    assert r_inst.json()["installed"][0]["version"] == "v1.0.0"

    r_toggle = client.put(
        f"/admin/api/microkernels/{pid}/enabled", json={"enabled": False}
    )
    assert r_toggle.status_code == 200
    assert r_toggle.json()["installed"][0]["enabled"] is False

    r_del = client.delete(f"/admin/api/microkernels/{pid}")
    assert r_del.status_code == 200
    assert r_del.json()["deleted"] is True
    assert r_del.json()["installed"] == []
