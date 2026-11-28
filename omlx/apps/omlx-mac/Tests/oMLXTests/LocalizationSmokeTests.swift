// Smoke test for the Localizable.xcstrings catalog. Pinned here so the
// Welcome-wizard Phase 1 wiring (and every later phase that adds keys)
// can't silently desync: if a key is renamed in code but not the catalog,
// or vice-versa, this test fails the build.
//
// We deliberately avoid asserting on every single key — that turns the
// test into a maintenance burden. Instead we check:
//   • the catalog parses
//   • a known-stable subset of keys resolves to its English value
//   • the source language is en
//   • welcome.* keys added in Phase 1 are all present
//
// String(localized:) with a key that's missing from the catalog returns
// the key itself, so the equality check below catches missing entries.

import XCTest
@testable import oMLX

final class LocalizationSmokeTests: XCTestCase {
    private static let englishBundle: Bundle = {
        guard let path = Bundle.main.path(forResource: "en", ofType: "lproj"),
              let bundle = Bundle(path: path) else {
            return .main
        }
        return bundle
    }()

    private static let simplifiedChineseErrorKeys: [String] = [
        "quant.error.cancel_failed",
        "quant.error.load_models",
        "quant.error.remove_failed",
        "quant.error.start_failed",
        "quant.upload.error.cancel_failed",
        "quant.upload.error.remove_failed",
    ]

    /// Hard-coded baseline of common.* keys → English values. Only the
    /// primitives actually used by at least one wrapped call site live here;
    /// any drift means someone touched the catalog without updating call
    /// sites (or vice versa).
    private static let commonBaseline: [(key: String, en: String)] = [
        ("common.cancel", "Cancel"),
        ("common.copy",   "Copy"),
        ("common.create", "Create"),
        ("common.open",   "Open"),
        ("common.save",   "Save"),
    ]

    /// Sentinel keys from every wrapped screen / surface. Presence-only check —
    /// if any of these resolves to the key string itself, the catalog is out
    /// of sync with the wrapped call sites. Two per surface keeps it cheap to
    /// run but catches drift on the most-visible strings.
    private static let sentinelKeys: [String] = [
        // Welcome wizard
        "welcome.window.title", "welcome.button.start_server",
        // Main app shell
        "about.section.project", "about.license.name",
        "logs.section.title", "network.section.proxies.title",
        // Server-side screens
        "server.section.advanced", "server.row.base_path",
        "security.section.api_key", "security.api_key.row_label",
        "integrations.section.applications", "integrations.tool.claude",
        "integrations.section.external_engines", "integrations.panel.model",
        "integrations.section.markitdown", "integrations.markitdown.pdf_processing_missing_model",
        "integrations.websearch.section_label", "integrations.websearch.test_success",
        "performance.section.cache", "performance.cache.enabled",
        "askai.row.enabled", "askai.section.presets",
        "settings.search.prompt", "settings.search.empty",
        "status.section.system", "status.section.active_now",
        // High-density screens
        "models.active.title", "models.library.title",
        "models.updates.check", "models.updates.available",
        "downloads.hf.section.title", "downloads.active.title",
        "quant.header.title", "quant.about.title",
        // Profile + bench
        "profile.scope.preset", "profile.detail.section.sampling",
        "bench.accuracy.header.title", "bench.accuracy.section.queue",
        "bench.throughput.header.title", "bench.throughput.section.configuration",
        "bench.context.header.title", "bench.context.section.configuration",
        // Settings + helpers
        "settings.section.basic", "settings.advanced.experimental.section",
        "settings.actions.reset", "settings.apply.choose.group_pp",
        "appearance.row.menubar_icon", "appearance.row.menubar_icon.restore",
        // Menubar + updates
        "menubar.item.quit", "menubar.stats.session_section",
        "menubar.item.settings", "menubar.item.web_dashboard",
        "update.channel.stable", "update.confirm.title",
    ]

    func testCatalogResolvesCommonBaseline() {
        // Force English so the assertion holds regardless of host locale.
        for (key, expected) in Self.commonBaseline {
            let resolved = NSLocalizedString(key, bundle: Self.englishBundle,
                                             value: key, comment: "")
            XCTAssertEqual(resolved, expected,
                           "common key \(key) resolved to \(resolved); expected \(expected)")
        }
    }

    func testSimplifiedChineseErrorTemplatesPreserveDetails() {
        guard let path = Bundle.main.path(forResource: "zh-Hans", ofType: "lproj"),
              let bundle = Bundle(path: path) else {
            XCTFail("Simplified Chinese localization bundle is missing")
            return
        }

        for key in Self.simplifiedChineseErrorKeys {
            let resolved = NSLocalizedString(key, bundle: bundle,
                                             value: key, comment: "")
            XCTAssertNotEqual(resolved, key,
                              "zh-Hans localization for \(key) exposes its key")
            XCTAssertTrue(resolved.contains("%@"),
                          "zh-Hans localization for \(key) drops the error placeholder")
        }
    }

    func testSentinelKeysArePresentInCatalog() {
        // Presence-only check: NSLocalizedString returns the key itself
        // when missing. We pass `value: <sentinel>` so a real missing key
        // resolves to the sentinel and never accidentally equals the key.
        let sentinel = "__missing__"
        // Resolve against the source-language bundle rather than `.main`.
        // With a partial catalog for the host's language (Korean, Japanese,
        // Traditional Chinese, and now German), `.main` picks that language's
        // table, and an explicit `value:` suppresses the fallback to English,
        // so every sentinel would come back unresolved.
        for key in Self.sentinelKeys {
            let resolved = NSLocalizedString(key, bundle: Self.englishBundle,
                                             value: sentinel, comment: "")
            XCTAssertNotEqual(resolved, sentinel,
                              "key \(key) is wired in code but missing from xcstrings")
            XCTAssertFalse(resolved.isEmpty,
                           "key \(key) resolved to an empty string")
        }
    }

    func testMivCoreBackendAndTurboquantLocalizations() throws {
        let zhHansPath = try XCTUnwrap(Bundle.main.path(forResource: "zh-Hans", ofType: "lproj"))
        let zhHans = try XCTUnwrap(Bundle(path: zhHansPath))
        let zhHantPath = try XCTUnwrap(Bundle.main.path(forResource: "zh-Hant", ofType: "lproj"))
        let zhHant = try XCTUnwrap(Bundle(path: zhHantPath))

        XCTAssertEqual(
            NSLocalizedString("external_engine.m5rt.label", bundle: Self.englishBundle, value: "", comment: ""),
            "miv Core Backend"
        )
        XCTAssertEqual(
            NSLocalizedString("external_engine.m5rt.label", bundle: zhHans, value: "", comment: ""),
            "miv 核心后端"
        )
        XCTAssertEqual(
            NSLocalizedString("external_engine.m5rt.label", bundle: zhHant, value: "", comment: ""),
            "miv 核心後端"
        )
        XCTAssertEqual(
            NSLocalizedString("settings.summary.turboquant", bundle: zhHans, value: "", comment: ""),
            "TurboQuant KV"
        )
        XCTAssertEqual(
            NSLocalizedString("settings.advanced.external_engine.stock_override", bundle: Self.englishBundle, value: "", comment: ""),
            "Using Stock MLX (auto-route overridden)"
        )
    }

    func testMivSettingsAndRouteBadgesExistInEveryAppLanguage() throws {
        let languages = try XCTUnwrap(Bundle.main.object(forInfoDictionaryKey: "CFBundleLocalizations") as? [String])
        XCTAssertTrue(languages.contains("zh-Hans"))
        let keys = [
            "settings.advanced.external_engine.opt.max_context",
            "settings.advanced.external_engine.opt.speculative_backend",
            "settings.advanced.external_engine.hint.sampling_contract",
            "settings.advanced.external_engine.choice.position_coupled",
            "settings.advanced.lora_mode.bypass", "settings.miv.forwarded",
            "settings.miv.prefill.hint.cache_blend", "external_engine.auto.short",
            "models.format.miv_available.help", "models.format.miv_routed.help",
        ]
        for language in languages {
            let url = try XCTUnwrap(Bundle.main.url(forResource: "Localizable", withExtension: "strings",
                                                    subdirectory: nil, localization: language))
            let table = try XCTUnwrap(PropertyListSerialization.propertyList(from: Data(contentsOf: url),
                                                                             format: nil) as? [String: String])
            for key in keys {
                XCTAssertFalse(table[key, default: ""].isEmpty, "\(language) is missing \(key)")
            }
            if language == "zh-Hans" {
                XCTAssertEqual(table["settings.advanced.external_engine.opt.max_context"], "最大上下文长度")
                XCTAssertEqual(table["settings.advanced.external_engine.choice.position_coupled"], "按位置耦合 v1（GPU）")
            }
        }
    }

    func testCatalogIsValidJSON() {
        // Direct file-level parse so a catalog corruption (extra trailing
        // comma, bad nesting) shows up here rather than as a missing-string
        // mystery at runtime.
        guard let url = Bundle.main.url(forResource: "Localizable",
                                        withExtension: "xcstrings") else {
            // Some test hosts strip xcstrings; treat as non-fatal so the
            // suite stays green when run outside Xcode's resource bundle.
            return
        }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            XCTFail("Couldn't read Localizable.xcstrings: \(error)")
            return
        }
        do {
            let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            XCTAssertEqual(root?["sourceLanguage"] as? String, "en",
                           "catalog sourceLanguage should be en")
            let strings = root?["strings"] as? [String: Any] ?? [:]
            XCTAssertGreaterThan(strings.count, 800,
                                 "catalog suspiciously small (\(strings.count) keys)")
        } catch {
            XCTFail("Localizable.xcstrings is not valid JSON: \(error)")
        }
    }
}
