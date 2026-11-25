import XCTest
@testable import oMLX

@MainActor
final class IntegrationsScreenVMTests: XCTestCase {
    private var server: IntegrationsSettingsServer { IntegrationsSettingsURLProtocol.server }

    private func makeClient() throws -> (OMLXClient, URLSession) {
        try server.reset()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [IntegrationsSettingsURLProtocol.self]
        let session = URLSession(configuration: configuration)
        return (OMLXClient(port: 18161, session: session), session)
    }

    func testConfiguredSettingsAndOcrChoicesPreserveHiddenPreference() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        server.setIntegrations([
            "markitdown_enabled": false, "markitdown_expose_model": true,
            "markitdown_max_file_size_mb": 40, "markitdown_max_files_per_request": 9,
            "markitdown_pdf_processing_engine": "GLM-OCR"
        ])
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        XCTAssertNil(vm.lastError)
        XCTAssertFalse(vm.markitdownModelVisible)
        vm.markitdownModelVisible = false
        XCTAssertTrue(vm.markitdownExposeModel)
        XCTAssertEqual(vm.markitdownMaxFileSizeText, "40")
        XCTAssertEqual(vm.markitdownMaxFilesText, "9")
        XCTAssertEqual(vm.markitdownPdfOptions.map(\.0), ["markitdown", "GLM-OCR", "DeepSeek-OCR"])
        XCTAssertNil(vm.markitdownPdfWarning)
        vm.markitdownEnabled = true
        XCTAssertTrue(vm.markitdownModelVisible)
    }

    func testMissingOcrSelectionIsRetainedWithBothWarnings() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        server.setIntegrations(["markitdown_pdf_processing_engine": "missing-ocr"])
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        XCTAssertTrue(vm.markitdownOcrModelMissing)
        XCTAssertEqual(vm.markitdownPdfOptions.map(\.0), ["markitdown", "missing-ocr", "GLM-OCR", "DeepSeek-OCR"])
        let missingWarning = try XCTUnwrap(vm.markitdownPdfWarning)
        vm.markitdownPdfProcessingEngine = "markitdown"
        let warning = try XCTUnwrap(vm.markitdownPdfWarning)
        XCTAssertTrue(missingWarning.hasSuffix("\n\(warning)"))
        XCTAssertTrue(server.patches.isEmpty)
    }

    func testVisibilitySavesRefreshModelsAndSurviveRefreshFailure() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        vm.markitdownMaxFilesText = "12"
        vm.markitdownModelVisible = true
        await vm.save(.markitdownExposeModel, client: client)
        XCTAssertEqual(server.patches.last?.count, 1)
        XCTAssertEqual(server.patches.last?["markitdown_expose_model"] as? Bool, true)

        vm.markitdownEnabled = false
        await vm.save(.markitdownEnabled, client: client)
        XCTAssertFalse(vm.markitdownModelVisible)
        XCTAssertTrue(vm.markitdownExposeModel)
        XCTAssertEqual(server.patches.last?.count, 1)
        XCTAssertEqual(server.patches.last?["markitdown_enabled"] as? Bool, false)
        XCTAssertEqual(vm.markitdownMaxFilesText, "12")
        XCTAssertEqual(server.modelReads, 3)

        server.rejectModelReads(true)
        vm.markitdownEnabled = true
        await vm.save(.markitdownEnabled, client: client)
        XCTAssertTrue(vm.markitdownEnabled)
        XCTAssertNotNil(vm.lastError)
    }

    func testLimitsAndEngineSaveOnlyTheirChangedFieldWhileDisabled() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        server.setIntegrations(["markitdown_enabled": false])
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        vm.markitdownMaxFileSizeText = " 40 "
        vm.markitdownMaxFilesText = "9"
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.count, 1)
        XCTAssertEqual(server.patches.last?.count, 1)
        XCTAssertEqual(server.patches.last?["markitdown_max_file_size_mb"] as? Int, 40)
        XCTAssertEqual(vm.markitdownMaxFileSizeText, "40")
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.count, 1)
        await vm.save(.markitdownMaxFiles, client: client)
        XCTAssertEqual(server.patches.last?["markitdown_max_files_per_request"] as? Int, 9)
        vm.markitdownPdfProcessingEngine = "GLM-OCR"
        await vm.save(.markitdownPdfProcessingEngine, client: client)
        XCTAssertEqual(server.patches.last?.count, 1)
        XCTAssertEqual(server.patches.last?["markitdown_pdf_processing_engine"] as? String, "GLM-OCR")
        await vm.save(.markitdownPdfProcessingEngine, client: client)
        XCTAssertEqual(server.patches.count, 3)
        XCTAssertEqual(server.modelReads, 1)
        XCTAssertNil(vm.lastError)
    }

    func testInvalidLimitsKeepDraftsAndSendNoRequests() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        for text in ["0", "1.5", "abc"] {
            vm.markitdownMaxFileSizeText = text
            await vm.save(.markitdownMaxFileSize, client: client)
            XCTAssertNotNil(vm.lastError, text)
            XCTAssertEqual(vm.markitdownMaxFileSizeText, text)
        }
        XCTAssertTrue(server.patches.isEmpty)
    }

    func testFailedSavesRestoreSwitchesAndPickerAndKeepNumericDraftsForRetry() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        server.setIntegrations(["markitdown_expose_model": true, "markitdown_pdf_processing_engine": "GLM-OCR"])
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        server.rejectSaves(true)
        vm.markitdownEnabled = false
        await vm.save(.markitdownEnabled, client: client)
        XCTAssertTrue(vm.markitdownEnabled)
        XCTAssertNotNil(vm.lastError)
        vm.markitdownModelVisible = false
        await vm.save(.markitdownExposeModel, client: client)
        XCTAssertTrue(vm.markitdownModelVisible)
        vm.markitdownPdfProcessingEngine = "markitdown"
        await vm.save(.markitdownPdfProcessingEngine, client: client)
        XCTAssertEqual(vm.markitdownPdfProcessingEngine, "GLM-OCR")
        vm.markitdownMaxFileSizeText = "60"
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(vm.markitdownMaxFileSizeText, "60")
        XCTAssertTrue(vm.savingMarkitdownFields.isEmpty)

        server.rejectSaves(false)
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.last?["markitdown_max_file_size_mb"] as? Int, 60)
        XCTAssertNil(vm.lastError)
    }

    func testDuplicateSubmitAndBlurSavesAreSuppressedWhileInFlight() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        vm.markitdownMaxFileSizeText = "30"
        let started = XCTestExpectation(description: "Numeric save started")
        server.holdNextPost(started: started)
        let submit = Task { await vm.save(.markitdownMaxFileSize, client: client) }
        let result = await XCTWaiter.fulfillment(of: [started], timeout: 3)
        XCTAssertEqual(result, .completed)
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.count, 1)
        server.finishHeldPost()
        await submit.value
        XCTAssertTrue(vm.savingMarkitdownFields.isEmpty)
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.count, 1)
    }

    func testWebSearchLoadsConfiguredValues() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        server.setIntegrations([
            "web_search_provider": "ddgs_custom", "web_search_brave_api_key": "saved-key",
            "web_search_searxng_url": "http://localhost:8888", "web_search_ddgs_backends": "mojeek,yahoo",
            "web_search_max_results": 7, "web_search_content_mode": "full",
            "web_search_content_truncate": false, "web_search_content_max_chars": 500
        ])
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        XCTAssertEqual(vm.webSearch, .init(provider: "ddgs_custom", braveApiKey: "saved-key",
                                         searxngUrl: "http://localhost:8888", ddgsBackends: "mojeek,yahoo",
                                         maxResults: "7", contentMode: "full", contentTruncate: false,
                                         contentMaxChars: "500"))
    }

    func testWebSearchSavesExactlyOneFieldAndPreservesOtherDrafts() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        vm.webSearch = .init(provider: "brave", braveApiKey: " draft-key ",
                             searxngUrl: " http://localhost:8888 ", ddgsBackends: "mojeek,yahoo",
                             maxResults: " 7 ", contentMode: "full", contentTruncate: false,
                             contentMaxChars: "500")
        let cases: [(IntegrationsScreenVM.WebSearchField, String, Any)] = [
            (.provider, "web_search_provider", "brave"),
            (.braveApiKey, "web_search_brave_api_key", "draft-key"),
            (.searxngUrl, "web_search_searxng_url", "http://localhost:8888"),
            (.ddgsBackends, "web_search_ddgs_backends", "mojeek,yahoo"),
            (.maxResults, "web_search_max_results", 7),
            (.contentMode, "web_search_content_mode", "full"),
            (.contentTruncate, "web_search_content_truncate", false),
            (.contentMaxChars, "web_search_content_max_chars", 500)
        ]
        for (field, key, value) in cases {
            let previousCount = server.patches.count
            await vm.saveWebSearch(field, client: client)
            XCTAssertNil(vm.lastError)
            XCTAssertEqual(server.patches.count, previousCount + 1)
            XCTAssertEqual(server.patches.last! as NSDictionary, [key: value] as NSDictionary)
            await vm.saveWebSearch(field, client: client)
            XCTAssertEqual(server.patches.count, previousCount + 1)
            if field == .provider { XCTAssertEqual(vm.webSearch.braveApiKey, " draft-key ") }
        }
        XCTAssertEqual(vm.webSearch.braveApiKey, "draft-key")
        XCTAssertEqual(vm.webSearch.maxResults, "7")
        vm.setDdgsBackend("mojeek", selected: false)
        vm.setDdgsBackend("brave", selected: true)
        vm.setDdgsBackend("brave", selected: true)
        XCTAssertEqual(vm.webSearch.ddgsBackends, "yahoo,brave")
    }

    func testWebSearchRejectsInvalidNumbersWithoutSaving() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        vm.webSearch.maxResults = "11"
        await vm.saveWebSearch(.maxResults, client: client)
        XCTAssertNotNil(vm.lastError)
        vm.webSearch.contentMaxChars = "0"
        await vm.saveWebSearch(.contentMaxChars, client: client)
        XCTAssertNotNil(vm.lastError)
        XCTAssertTrue(server.patches.isEmpty)
    }

    func testWebSearchFailedSavesRestoreControlsAndRetainTypedValuesForRetry() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        server.rejectSaves(true)
        vm.webSearch = .init(provider: "brave", braveApiKey: "retry-key",
                             searxngUrl: "http://localhost:8888", ddgsBackends: "yahoo",
                             maxResults: "8", contentMode: "full", contentTruncate: false,
                             contentMaxChars: "500")
        for field in IntegrationsScreenVM.WebSearchField.allCases {
            await vm.saveWebSearch(field, client: client)
            XCTAssertNotNil(vm.lastError)
        }
        XCTAssertEqual(vm.webSearch.provider, "ddgs")
        XCTAssertEqual(vm.webSearch.ddgsBackends, "")
        XCTAssertEqual(vm.webSearch.contentMode, "snippet")
        XCTAssertTrue(vm.webSearch.contentTruncate)
        XCTAssertEqual(vm.webSearch.braveApiKey, "retry-key")
        XCTAssertEqual(vm.webSearch.searxngUrl, "http://localhost:8888")
        XCTAssertEqual(vm.webSearch.maxResults, "8")
        XCTAssertEqual(vm.webSearch.contentMaxChars, "500")
        XCTAssertTrue(vm.savingWebSearchFields.isEmpty)
        server.rejectSaves(false)
        await vm.saveWebSearch(.braveApiKey, client: client)
        XCTAssertNil(vm.lastError)
        XCTAssertEqual(server.patches.last?["web_search_brave_api_key"] as? String, "retry-key")
    }

    func testWebSearchConnectivityUsesUnsavedValuesAndDoesNotPersist() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        vm.webSearch = .init(provider: "brave", braveApiKey: "pending-key",
                             searxngUrl: "http://localhost:8888", ddgsBackends: "yahoo",
                             maxResults: "7", contentMode: "full", contentTruncate: false,
                             contentMaxChars: "500")
        await vm.testWebSearch(client: client)
        XCTAssertEqual(server.searchRequests.last! as NSDictionary, [
            "provider": "brave", "brave_api_key": "pending-key",
            "searxng_url": "http://localhost:8888", "ddgs_backends": "yahoo", "max_results": 7
        ] as NSDictionary)
        XCTAssertTrue(server.patches.isEmpty)
        XCTAssertEqual(vm.webSearchTestOK, true)
        XCTAssertEqual(vm.webSearchTestMessage, "Search OK: 1 results")
        vm.webSearch.contentMaxChars = "600"
        XCTAssertNil(vm.webSearchTestMessage)
        server.setSearchResponse(["ok": false, "error": ["message": "Provider rate limited"]])
        await vm.testWebSearch(client: client)
        XCTAssertEqual(vm.webSearchTestOK, false)
        XCTAssertEqual(vm.webSearchTestMessage, "Provider rate limited")
    }

    /// Local mode shows the API key masked and copies the real one.
    func testClaudeEnvRecipeMasksTheApiKey() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let noKey = IntegrationsScreenVM()
        await noKey.load(client: client)
        noKey.claudeMode = "local"
        XCTAssertEqual(noKey.claudeEnvRecipe, noKey.claudeEnvRecipeDisplay)
        XCTAssertFalse(noKey.claudeEnvRecipe.contains("ANTHROPIC_AUTH_TOKEN="))

        server.setApiKey("sk-live-secret")
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        vm.claudeMode = "local"
        XCTAssertTrue(vm.claudeEnvRecipe.contains("sk-live-secret"))
        XCTAssertFalse(vm.claudeEnvRecipeDisplay.contains("sk-live-secret"))
        XCTAssertEqual(vm.claudeEnvRecipeDisplay.replacingOccurrences(of: "••••••••", with: "sk-live-secret"),
                       vm.claudeEnvRecipe)

        vm.claudeMode = "cloud"
        XCTAssertEqual(vm.claudeEnvRecipe, vm.claudeEnvRecipeDisplay)
    }

    func testMicrokernelManagerLifecycleRefreshDownloadToggleUpdateAndDelete() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        server.enableMicrokernelFixture()
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)

        // Initial load: catalog present, miv installed, update button MUST be dimmed before refresh!
        XCTAssertEqual(vm.microkernelCatalog.count, 2)
        XCTAssertEqual(vm.selectedCatalogProviderId, "io.ninfer.metal")
        XCTAssertFalse(vm.microkernelsRefreshed)
        XCTAssertEqual(vm.installedMicrokernels.count, 1)
        XCTAssertFalse(vm.installedMicrokernels[0].isUpdateReady)
        XCTAssertFalse(vm.canDownloadSelectedCatalogItem) // miv already installed & managed

        // Select second core (Path B Llama) in dropdown -> shows supported models & Download enabled!
        vm.selectedCatalogProviderId = "io.community.llama-metal"
        XCTAssertEqual(vm.selectedCatalogItem?.supportedModelTypes, ["llama", "mistral"])
        XCTAssertTrue(vm.canDownloadSelectedCatalogItem)

        // Download Path B core -> appears in installed list enabled by default
        await vm.downloadSelectedMicrokernel(client: client)
        XCTAssertEqual(vm.installedMicrokernels.count, 2)
        let llamaInstalled = try XCTUnwrap(vm.installedMicrokernels.first(where: { $0.providerId == "io.community.llama-metal" }))
        XCTAssertTrue(llamaInstalled.isEnabled)
        XCTAssertFalse(llamaInstalled.isUpdateReady)

        // Click Refresh -> now miv has update available and lights up its Update button!
        await vm.refreshMicrokernels(client: client)
        XCTAssertTrue(vm.microkernelsRefreshed)
        let mivAfterRefresh = try XCTUnwrap(vm.installedMicrokernels.first(where: { $0.providerId == "io.ninfer.metal" }))
        XCTAssertTrue(mivAfterRefresh.isUpdateReady)
        XCTAssertEqual(mivAfterRefresh.remoteVersion, "miv-v0.02")

        // Click Update on miv -> updates version and turns Update button back off
        await vm.updateMicrokernel(mivAfterRefresh, client: client)
        let mivUpdated = try XCTUnwrap(vm.installedMicrokernels.first(where: { $0.providerId == "io.ninfer.metal" }))
        XCTAssertEqual(mivUpdated.version, "miv-v0.02")
        XCTAssertFalse(mivUpdated.isUpdateReady)

        // Uncheck Llama checkbox -> persists enabled = false
        await vm.toggleMicrokernelEnabled(llamaInstalled, enabled: false, client: client)
        let llamaDisabled = try XCTUnwrap(vm.installedMicrokernels.first(where: { $0.providerId == "io.community.llama-metal" }))
        XCTAssertFalse(llamaDisabled.isEnabled)

        // Delete Llama -> removed from installed list
        await vm.deleteMicrokernel(llamaDisabled, unloadActive: false, client: client)
        XCTAssertEqual(vm.installedMicrokernels.count, 1)
        XCTAssertFalse(vm.installedMicrokernels.contains(where: { $0.providerId == "io.community.llama-metal" }))
    }
}

private final class IntegrationsSettingsServer: @unchecked Sendable {
    private let lock = NSLock()
    private var settings: [String: Any] = [:]
    private var recordedPatches: [[String: Any]] = []
    private var modelReadCount = 0
    private var savesRejected = false
    private var modelReadsRejected = false
    private var postStarted: XCTestExpectation?
    private var heldPost: (@Sendable () -> Void)?
    private var recordedSearchRequests: [[String: Any]] = []
    private var searchResponse: [String: Any] = [:]
    private var stats: [String: Any] = [:]
    private var mkEnabled = false
    private var mkRefreshed = false
    private var mkMivVersion = "miv-v0.01"
    private var mkLlamaInstalled = false
    private var mkLlamaEnabled = true

    var searchRequests: [[String: Any]] { lock.withLock { recordedSearchRequests } }
    var patches: [[String: Any]] { lock.withLock { recordedPatches } }
    var modelReads: Int { lock.withLock { modelReadCount } }

    func reset() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/global-settings.json")
        let fixture = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        lock.withLock {
            settings = fixture
            recordedPatches = []
            modelReadCount = 0
            savesRejected = false
            modelReadsRejected = false
            postStarted = nil
            heldPost = nil
            recordedSearchRequests = []
            searchResponse = ["ok": true, "results": [["title": "Example", "url": "https://example.com", "snippet": "Example result"]]]
            stats = Self.statsBody(apiKey: "")
            mkEnabled = false
            mkRefreshed = false
            mkMivVersion = "miv-v0.01"
            mkLlamaInstalled = false
            mkLlamaEnabled = true
        }
    }

    func enableMicrokernelFixture() {
        lock.withLock { mkEnabled = true }
    }

    private func microkernelPayload() -> [String: Any] {
        let mivHasUpdate = mkRefreshed && mkMivVersion == "miv-v0.01"
        var catalog: [[String: Any]] = [
            [
                "provider_id": "io.ninfer.metal",
                "engine_id": "m5rt",
                "display_name": "miv Core Backend (ninfer-metal)",
                "path_tier": "Path A",
                "current_version": "miv-v0.01",
                "remote_version": mkRefreshed ? "miv-v0.02" : "miv-v0.01",
                "protocol_version": 2,
                "min_macos_version": "27.0",
                "supported_model_types": ["qwen3_5", "qwen3_6", "qwen3_8"],
                "supported_topologies": ["qwen35_hybrid_dense_v1"],
                "artifact_extensions": [".omlxm5"],
                "installed": true,
                "installed_version": mkMivVersion,
                "managed": true,
                "enabled": true,
                "has_update": mivHasUpdate,
                "can_download": false,
            ],
            [
                "provider_id": "io.community.llama-metal",
                "engine_id": "llama-metal",
                "display_name": "Community Llama Metal",
                "path_tier": "Path B",
                "current_version": "v0.1.0",
                "remote_version": "v0.1.0",
                "protocol_version": 2,
                "min_macos_version": "27.0",
                "supported_model_types": ["llama", "mistral"],
                "supported_topologies": ["llama3_gqa_v1"],
                "artifact_extensions": [".omlxm5"],
                "installed": mkLlamaInstalled,
                "installed_version": mkLlamaInstalled ? "v0.1.0" : NSNull(),
                "managed": mkLlamaInstalled,
                "enabled": mkLlamaEnabled,
                "has_update": false,
                "can_download": !mkLlamaInstalled,
            ],
        ]
        var installed: [[String: Any]] = [
            [
                "provider_id": "io.ninfer.metal",
                "engine_id": "m5rt",
                "display_name": "miv Core Backend (ninfer-metal)",
                "version": mkMivVersion,
                "remote_version": mkRefreshed ? "miv-v0.02" : NSNull(),
                "has_update": mivHasUpdate,
                "can_update": mivHasUpdate,
                "path_tier": "Path A",
                "protocol_version": 2,
                "installed": true,
                "enabled": true,
                "managed": true,
                "can_delete": true,
                "binary_path": "/Users/test/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/bin/mseries_backendd",
                "supported_model_types": ["qwen3_5", "qwen3_6", "qwen3_8"],
                "models": [Any](),
            ]
        ]
        if mkLlamaInstalled {
            installed.append([
                "provider_id": "io.community.llama-metal",
                "engine_id": "llama-metal",
                "display_name": "Community Llama Metal",
                "version": "v0.1.0",
                "remote_version": mkRefreshed ? "v0.1.0" : NSNull(),
                "has_update": false,
                "can_update": false,
                "path_tier": "Path B",
                "protocol_version": 2,
                "installed": true,
                "enabled": mkLlamaEnabled,
                "managed": true,
                "can_delete": true,
                "binary_path": "/Users/test/Library/Application Support/oMLX/Providers/io.community.llama-metal/current/bin/llama_daemon",
                "supported_model_types": ["llama", "mistral"],
                "models": [Any](),
            ])
        }
        return [
            "registry_source": mkRefreshed ? "remote" : "local",
            "refreshed": mkRefreshed,
            "catalog": catalog,
            "installed": installed,
        ]
    }

    /// Required stats aggregates plus the fields the command builders read.
    private static func statsBody(apiKey: String) -> [String: Any] {
        [
            "total_tokens_served": 0, "total_cached_tokens": 0, "cache_efficiency": 0.0,
            "total_prompt_tokens": 0, "total_completion_tokens": 0, "total_requests": 0,
            "avg_prefill_tps": 0.0, "avg_generation_tps": 0.0, "uptime_seconds": 0.0,
            "host": "127.0.0.1", "port": 8000, "api_key": apiKey, "cli_prefix": "omlx",
            "active_models": ["models": [Any]()],
        ]
    }

    func setApiKey(_ apiKey: String) { lock.withLock { stats = Self.statsBody(apiKey: apiKey) } }

    func setSearchResponse(_ body: [String: Any]) { lock.withLock { searchResponse = body } }

    func setIntegrations(_ values: [String: Any]) {
        lock.withLock {
            var integrations = settings["integrations"] as! [String: Any]
            integrations.merge(values) { _, new in new }
            settings["integrations"] = integrations
        }
    }

    func rejectSaves(_ reject: Bool) { lock.withLock { savesRejected = reject } }
    func rejectModelReads(_ reject: Bool) { lock.withLock { modelReadsRejected = reject } }
    func holdNextPost(started: XCTestExpectation) { lock.withLock { postStarted = started } }

    func finishHeldPost() {
        let completion = lock.withLock {
            let completion = heldPost
            heldPost = nil
            return completion
        }
        completion?()
    }

    func respond(to request: URLRequest, completion: @escaping @Sendable (Int, Data) -> Void) throws {
        lock.lock()
        defer { lock.unlock() }
        var status = 200
        var body: [String: Any]
        switch (request.httpMethod, request.url?.path) {
        case ("GET", "/admin/api/global-settings"):
            body = settings
        case ("GET", "/admin/api/stats"):
            body = stats
        case ("GET", "/admin/api/models"):
            modelReadCount += 1
            if modelReadsRejected {
                status = 503
                body = ["detail": "Model list unavailable"]
            } else {
                body = ["models": [
                    model("GLM-OCR", configType: "GLM_OCR"),
                    model("DeepSeek-OCR", configType: "deepseek_ocr2"),
                    model("ocr-in-name-only", configType: "qwen3_5"),
                    model("unknown", configType: nil)
                ]]
            }
        case ("POST", "/admin/api/web-search/test"):
            recordedSearchRequests.append(try JSONSerialization.jsonObject(with: requestBody(request)) as! [String: Any])
            body = searchResponse
        case ("POST", "/admin/api/global-settings"):
            let patch = try JSONSerialization.jsonObject(with: requestBody(request)) as! [String: Any]
            recordedPatches.append(patch)
            if savesRejected {
                status = 400
                body = ["detail": "Settings save rejected"]
            } else {
                var integrations = settings["integrations"] as! [String: Any]
                integrations.merge(patch) { _, new in new }
                settings["integrations"] = integrations
                body = ["success": true, "runtime_applied": ["integrations"]]
            }
        case ("GET", "/admin/api/microkernels"):
            if mkEnabled {
                body = microkernelPayload()
            } else {
                status = 404
                body = ["detail": "Not enabled in test"]
            }
        case ("POST", "/admin/api/microkernels/refresh"):
            mkRefreshed = true
            body = microkernelPayload()
        case ("POST", "/admin/api/microkernels/io.ninfer.metal/install"):
            mkMivVersion = "miv-v0.02"
            body = microkernelPayload()
        case ("POST", "/admin/api/microkernels/io.community.llama-metal/install"):
            mkLlamaInstalled = true
            mkLlamaEnabled = true
            body = microkernelPayload()
        case ("PUT", "/admin/api/microkernels/io.community.llama-metal/enabled"):
            let req = try JSONSerialization.jsonObject(with: requestBody(request)) as! [String: Any]
            mkLlamaEnabled = (req["enabled"] as? Bool) ?? true
            body = microkernelPayload()
        case ("DELETE", "/admin/api/microkernels/io.community.llama-metal"):
            mkLlamaInstalled = false
            body = microkernelPayload()
        default:
            body = [:]
        }
        let data = try JSONSerialization.data(withJSONObject: body)
        if request.httpMethod == "POST", let started = postStarted {
            postStarted = nil
            let responseStatus = status
            heldPost = { completion(responseStatus, data) }
            started.fulfill()
        } else {
            completion(status, data)
        }
    }

    private func model(_ id: String, configType: String?) -> [String: Any] {
        var model: [String: Any] = ["id": id, "loaded": false, "is_loading": false, "estimated_size": 0]
        model["config_model_type"] = configType
        return model
    }

    private func requestBody(_ request: URLRequest) throws -> Data {
        if let data = request.httpBody { return data }
        let stream = try XCTUnwrap(request.httpBodyStream)
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 1024)
        while stream.hasBytesAvailable {
            let count = stream.read(&buffer, maxLength: buffer.count)
            if count < 0 { throw stream.streamError ?? URLError(.cannotDecodeRawData) }
            if count == 0 { break }
            data.append(contentsOf: buffer.prefix(count))
        }
        return data
    }
}

private final class IntegrationsSettingsURLProtocol: URLProtocol, @unchecked Sendable {
    static let server = IntegrationsSettingsServer()

    override class func canInit(with request: URLRequest) -> Bool { request.url?.port == 18161 }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        do {
            try Self.server.respond(to: request) { [self] status, data in
                let response = HTTPURLResponse(url: request.url!, statusCode: status,
                                               httpVersion: nil, headerFields: nil)!
                client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
                client?.urlProtocol(self, didLoad: data)
                client?.urlProtocolDidFinishLoading(self)
            }
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
