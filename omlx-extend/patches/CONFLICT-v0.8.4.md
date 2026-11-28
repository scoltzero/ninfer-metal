# oMLX-extend Monorepo Sync Conflict (v0.8.4)

Automated monorepo 3-way sync against `jundot/omlx@v0.8.4` (target `omlx-extend-v0.8.4`) encountered a merge conflict while carrying forward live `oMLX-extend` / `mix` changes in `omlx/`.

## Sync Log
```text
==> Monorepo direct-sync: upstream v0.8.4 -> /Users/runner/work/ninfer-metal/ninfer-metal/omlx (target release: omlx-extend-v0.8.4)
==> 3-way merge mode: base=v0.8.3, ours=live /Users/runner/work/ninfer-metal/ninfer-metal/omlx, theirs=v0.8.4
==> Fetching upstream tags v0.8.3 and v0.8.4 (depth=1)...
==> Running 3-way Git merge onto upstream v0.8.4...
Auto-merging apps/omlx-mac/Resources/Localizable.xcstrings
Auto-merging apps/omlx-mac/Sources/AppView/Screens/IntegrationsScreen.swift
Auto-merging apps/omlx-mac/Sources/AppView/Screens/ModelSettingsScreen.swift
Auto-merging apps/omlx-mac/Sources/AppView/ViewModels/IntegrationsScreenVM.swift
Auto-merging apps/omlx-mac/Sources/AppView/ViewModels/ModelSettingsScreenVM.swift
Auto-merging apps/omlx-mac/Sources/Net/DTO/GlobalSettingsDTO.swift
CONFLICT (content): Merge conflict in apps/omlx-mac/Sources/Net/DTO/GlobalSettingsDTO.swift
Auto-merging apps/omlx-mac/Sources/Net/DTO/ModelsDTO.swift
Auto-merging apps/omlx-mac/Sources/Net/Endpoints.swift
Auto-merging apps/omlx-mac/Sources/Net/OMLXClient.swift
Auto-merging apps/omlx-mac/Sources/Updater/AppUpdater.swift
Auto-merging apps/omlx-mac/Tests/oMLXTests/DTOFixtureTests.swift
Auto-merging apps/omlx-mac/Tests/oMLXTests/IntegrationsScreenVMTests.swift
CONFLICT (content): Merge conflict in apps/omlx-mac/Tests/oMLXTests/IntegrationsScreenVMTests.swift
Auto-merging apps/omlx-mac/Tests/oMLXTests/ModelSettingsScreenVMTests.swift
Auto-merging apps/omlx-web/omlx_web/i18n/en.json
Auto-merging apps/omlx-web/omlx_web/i18n/zh-TW.json
Auto-merging apps/omlx-web/omlx_web/i18n/zh.json
Auto-merging apps/omlx-web/omlx_web/static/js/dashboard.js
Auto-merging apps/omlx-web/omlx_web/templates/dashboard/_modal_model_settings.html
Auto-merging apps/omlx-web/omlx_web/templates/dashboard/_settings.html
Auto-merging docs/admin-api.md
Auto-merging omlx/admin/routes.py
CONFLICT (content): Merge conflict in omlx/admin/routes.py
Auto-merging omlx/engine/splash.py
Auto-merging omlx/engine_pool.py
CONFLICT (content): Merge conflict in omlx/engine_pool.py
Auto-merging omlx/model_profiles.py
Auto-merging omlx/model_settings.py
Auto-merging omlx/server.py
CONFLICT (content): Merge conflict in omlx/server.py
Auto-merging omlx/settings.py
CONFLICT (content): Merge conflict in omlx/settings.py
Auto-merging tests/integration/test_server_endpoints.py
Auto-merging tests/test_admin_model_settings.py
Auto-merging tests/test_admin_server_aliases.py
Auto-merging tests/test_splash_engine.py
Automatic merge failed; fix conflicts and then commit the result.
ERROR: 3-way merge conflict against upstream v0.8.4 in:
apps/omlx-mac/Sources/Net/DTO/GlobalSettingsDTO.swift
apps/omlx-mac/Tests/oMLXTests/IntegrationsScreenVMTests.swift
omlx/admin/routes.py
omlx/engine_pool.py
omlx/server.py
omlx/settings.py
```

## Resolution Steps (Monorepo Mode)
1. Run `./omlx-extend/scripts/sync_omlx.sh --tag v0.8.4 --keep-staging` locally and inspect/resolve conflicts in the preserved staging repo.
2. Verify `(cd omlx && pytest tests/test_microkernel_manager.py tests/test_m5_sidecar_engine.py tests/test_splash_engine.py tests/test_process_memory_enforcer.py -q)` and `pytest omlx-extend/tests/test_p1_ext_spi.py -q`.
3. Re-export the binary-safe patch via `./omlx-extend/scripts/sync_omlx.sh --export-patch --tag v0.8.4`, update `omlx-extend/UPSTREAM_VERSION`, and delete this conflict file.
