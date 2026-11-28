import Foundation
import XCTest
@testable import MIVApp

private final class CoreResponseFixture: URLProtocol {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let url = request.url!
        let key = request.value(forHTTPHeaderField: "Authorization")
        let modern = url.host == "current.test"
        var headers = ["Content-Type": "application/json"]
        var status = 200
        var body = ""
        switch url.path {
        case "", "/":
            if modern, request.httpMethod == "HEAD" {
                headers = ["Content-Type": "text/html; charset=utf-8", "X-MIV-Dashboard": "1"]
            } else { status = 404 }
        case "/metrics":
            body = """
            # TYPE ninfer_metal_generated_tokens_total counter
            ninfer_metal_generated_tokens_total 42
            ninfer_metal_model_generated_tokens_total{model="hello"} 42
            """
        case "/health":
            status = modern ? 200 : 503
            body = modern
                ? #"{"status":"ok","version":"test","backend":"metal","models":[{"id":"test","max_context":2048,"max_concurrency":4,"kv_dtype":"affine4_g128","spec_backend":"mtp","resident_memory_bytes":64,"prefill_ane_mode":"auto","prefill_ane_mode_effective":"off","prefill_ane_active":false,"prefill_ane_inactive_reason":"calibration_bypassed"}]}"#
                : #"{"status":"shutting_down","version":"old","model":"test"}"#
        case "/v1/models":
            if key == "Bearer sample" {
                body = modern
                    ? #"{"data":[{"id":"test","max_model_len":2048,"provider_id":"io.ninfer.metal","backend_mode":"inproc","spec_backend":"mtp","prefill_ane_mode":"auto","prefill_ane_mode_effective":"compact_int8","prefill_ane_active":true}]}"#
                    : #"{"data":[{"id":"test","max_model_len":2048,"backend_mode":"inproc","prefill_ane_offload":true}]}"#
            } else { status = 401 }
        default: status = 404
        }
        if url.path != "/v1/models" && key != nil { status = 400 }
        let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

final class CoreServiceTests: XCTestCase {
    private func client(host: String, key: String) -> CoreConnection {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [CoreResponseFixture.self]
        return CoreConnection(base: URL(string: "http://\(host)")!, key: key,
                              session: URLSession(configuration: configuration))
    }

    func testThreeReadRoutesAndDashboardWithKeyOnlyOnProtectedModels() async throws {
        let connection = client(host: "current.test", key: "sample")
        let health = try await connection.health()
        let model = try await connection.models().first
        let stats = try await connection.metrics()
        let hasDashboard = await connection.dashboardAvailable()
        XCTAssertEqual(health.models?.first?.maxContext, 2048)
        XCTAssertEqual(health.models?.first?.maxConcurrency, 4)
        XCTAssertEqual(health.models?.first?.kvDtype, "affine4_g128")
        XCTAssertEqual(model?.maxModelLen, 2048)
        XCTAssertEqual(model?.specBackend, "mtp")
        XCTAssertEqual(model?.providerId, "io.ninfer.metal")
        XCTAssertEqual(model?.prefillAneMode, "auto")
        XCTAssertEqual(model?.prefillAneModeEffective, "compact_int8")
        XCTAssertEqual(model?.prefillAneActive, true)
        XCTAssertNil(model?.prefillAneInactiveReason)
        XCTAssertEqual(health.models?.first?.prefillAneModeEffective, "off")
        XCTAssertEqual(health.models?.first?.prefillAneActive, false)
        XCTAssertEqual(health.models?.first?.prefillAneInactiveReason, "calibration_bypassed")
        XCTAssertEqual(stats, ["ninfer_metal_generated_tokens_total": 42])
        XCTAssertTrue(hasDashboard)
    }

    func testLegacyOffloadFlagDoesNotImplyANERuntimeTelemetry() async throws {
        let models = try await client(host: "older.test", key: "sample").models()
        let model = try XCTUnwrap(models.first)
        XCTAssertNil(model.prefillAneMode)
        XCTAssertNil(model.prefillAneModeEffective)
        XCTAssertNil(model.prefillAneActive)
        XCTAssertNil(model.prefillAneInactiveReason)
    }

    func testOlderCoreAndRejectedKeyRemainDistinctFromHealth() async throws {
        let connection = client(host: "older.test", key: "wrong")
        let health = try await connection.health()
        let hasDashboard = await connection.dashboardAvailable()
        XCTAssertEqual(health.status, "shutting_down")
        XCTAssertFalse(hasDashboard)
        do {
            _ = try await connection.models()
            XCTFail("Expected HTTP 401")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("401"))
            XCTAssertFalse(error.localizedDescription.contains("wrong"))
        }
    }

    func testLaunchFlagsAndInheritance() throws {
        var settings = CoreSettings()
        XCTAssertEqual(try settings.launchArguments(), ["--host", "127.0.0.1", "--port", "8080"])
        settings.port = 11234
        settings.configPath = "/tmp/miv settings.toml"
        settings.modelDirectory = "/tmp/model path"
        settings.maxContext = 2048
        settings.maxConcurrency = 2
        settings.kvDtype = "affine4_g128"
        settings.speculativeBackend = "mtp"
        XCTAssertEqual(try settings.launchArguments(), [
            "--host", "127.0.0.1", "--port", "11234", "--config", "/tmp/miv settings.toml",
            "--model-dir", "/tmp/model path", "--max-context", "2048", "--max-concurrency", "2",
            "--kv-dtype", "affine4_g128", "--spec-backend", "mtp"
        ])
        settings.maxConcurrency = 9
        XCTAssertThrowsError(try settings.launchArguments())
    }

    func testLANIsOptInIncludingPreferencesFromBeforeTheFlag() throws {
        let old = #"{"mode":"managed","endpoint":"http://localhost","port":4545,"modelDirectory":"","configPath":""}"#
        var settings = try JSONDecoder().decode(CoreSettings.self, from: Data(old.utf8))
        XCTAssertFalse(settings.allowLAN)
        XCTAssertEqual(try settings.launchArguments().prefix(2), ["--host", "127.0.0.1"])
        settings.allowLAN = true
        XCTAssertEqual(try settings.launchArguments().prefix(2), ["--host", "0.0.0.0"])
        XCTAssertEqual(try settings.baseURL().absoluteString, "http://127.0.0.1:4545")
        let saved = try JSONEncoder().encode(settings)
        XCTAssertTrue(try JSONDecoder().decode(CoreSettings.self, from: saved).allowLAN)
    }

    @MainActor
    func testConnectionSettingsAllowOriginOrV1AndKeepSecretsInMemory() throws {
        let suite = UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let service = CoreService(defaults: defaults)
        var settings = CoreSettings()
        settings.mode = .external
        settings.endpoint = "http://[::1]:8080/v1/"
        try service.apply(settings, apiKey: "sample")
        XCTAssertEqual(service.browserURL?.absoluteString, "http://[::1]:8080")
        XCTAssertThrowsError(try service.start())
        let relaunched = CoreService(defaults: defaults)
        XCTAssertEqual(relaunched.settings, settings)
        XCTAssertTrue(relaunched.apiKey.isEmpty)
        for invalid in ["file:///tmp/server", "https://host:70000", "https://user:pw@here", "https://host/?token=value"] {
            settings.endpoint = invalid
            XCTAssertThrowsError(try service.apply(settings, apiKey: ""))
        }
        XCTAssertEqual(service.apiKey, "sample")
    }
}

extension CoreServiceTests {
    @MainActor
    func testManagedStopExitAndFailedLaunchRetainOnlyOwnedLogs() async throws {
        let suite = UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        let script = FileManager.default.temporaryDirectory.appendingPathComponent("miv-test-\(suite)")
        defer {
            defaults.removePersistentDomain(forName: suite)
            try? FileManager.default.removeItem(at: script)
        }
        func install(_ body: String) throws {
            try ("#!/bin/sh\n" + body + "\n").write(to: script, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: script.path)
        }
        func waitForExit(_ service: CoreService) async throws {
            let deadline = Date().addingTimeInterval(3)
            while service.isRunning && Date() < deadline {
                try await Task.sleep(nanoseconds: 20_000_000)
            }
            XCTAssertFalse(service.isRunning)
        }

        try install("echo loading-fixture; exec /bin/sleep 30")
        var settings = CoreSettings()
        settings.port = Int.random(in: 49152...65535)
        let service = CoreService(defaults: defaults, executable: script)
        defer { service.shutdown() }
        try service.apply(settings, apiKey: "")
        try service.start()
        XCTAssertTrue(service.isRunning)
        var changed = settings
        changed.mode = .external
        XCTAssertThrowsError(try service.apply(changed, apiKey: ""))
        XCTAssertThrowsError(try service.start())
        for _ in 0..<50 where !service.logText.contains("loading-fixture") {
            try await Task.sleep(nanoseconds: 15_000_000)
            await service.refresh()
        }
        XCTAssertNil(service.health)
        XCTAssertNil(service.healthError)
        XCTAssertTrue(service.logText.contains("loading-fixture"))
        service.stop()
        try await waitForExit(service)
        XCTAssertNil(service.serviceMessage)

        try install("echo model-not-found-fixture >&2; exit 7")
        try service.start()
        try await waitForExit(service)
        XCTAssertTrue(service.serviceMessage?.contains("7") == true)
        XCTAssertTrue(service.logText.contains("model-not-found-fixture"))
        XCTAssertFalse(service.logText.contains("loading-fixture"))

        try install("exec /bin/sleep 30")
        try service.start()
        service.shutdown()
        XCTAssertFalse(service.isRunning)
        XCTAssertNil(service.serviceMessage)
    }
}

extension CoreServiceTests {
    @MainActor func testManagedOutputFloodPreservesReadySignalAndFinalBytes() async throws {
        let fm = FileManager.default, suite = UUID().uuidString
        let dir = fm.temporaryDirectory.appendingPathComponent("MIV-output-" + suite)
        try fm.createDirectory(at: dir, withIntermediateDirectories: false)
        let script = dir.appendingPathComponent("fixture.py")
        let ready = dir.appendingPathComponent("fixture.py.ready")
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? fm.removeItem(at: dir) }
        let fixture = #"""
        #!/usr/bin/python3
        import argparse, http.server, json, os, pathlib, signal, stat, time
        args = argparse.ArgumentParser(add_help=False)
        args.add_argument('--port', type=int)
        port = args.parse_known_args()[0].port
        assert all(stat.S_ISFIFO(os.fstat(fd).st_mode) for fd in [1, 2])

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_HEAD(self):
                self.send_response(200); self.send_header('Content-Type', 'text/html')
                self.send_header('X-MIV-Dashboard','1'); self.end_headers()
            def do_GET(self):
                if self.path == '/health': body = '{"status":"ok","version":"log-fixture"}'
                elif self.path == '/metrics': body = 'ninfer_metal_requests_total 0\n'
                else: body = '{"data":[{"id":"log-fixture"}]}'
                self.send_response(200); self.end_headers(); self.wfile.write(body.encode())

        httpd = http.server.HTTPServer(('127.0.0.1',port), Handler)
        def stop(signum, frame):
            # Write more than one OS pipe capacity immediately before exit.
            for i in range(48): os.write(2, b'E' * 8192)
            os.write(1,b'FINAL-STDOUT\n'); os.write(2,b'FINAL-STDERR\n')
            os._exit(0)
        signal.signal(signal.SIGTERM, stop)
        os.write(2,b'[ninfer-metal-serve] Listening ')
        time.sleep(.05)
        os.write(2,f'on http://127.0.0.1:{port} (fixture)\n'.encode())
        # Flood both streams before letting the UI make its very first refresh.
        for i in range(1024): os.write(1 if i % 2 == 0 else 2, b'X' * 8192)
        os.write(2,b'AFTER-8-MIB\n')
        pathlib.Path(__file__ + '.ready').touch()
        httpd.serve_forever()
        """#
        try fixture.write(to: script, atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: script.path)
        var settings = CoreSettings()
        settings.port = Int.random(in: 49152...65535)
        let service = CoreService(defaults: defaults, executable: script)
        defer { service.shutdown() }
        try service.apply(settings, apiKey: "")
        try service.start()
        let deadline = Date().addingTimeInterval(20)
        while !fm.fileExists(atPath: ready.path) && service.isRunning && Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue(fm.fileExists(atPath: ready.path), service.serviceMessage ?? "log flood did not drain")
        await service.refresh()
        XCTAssertEqual(service.health?.version, "log-fixture")
        XCTAssertEqual(service.logText.utf8.count, 48 * 1024)
        XCTAssertTrue(service.logText.hasSuffix("AFTER-8-MIB\n"))
        XCTAssertFalse(service.logText.contains("Listening on"))
        service.stop()
        let stopDeadline = Date().addingTimeInterval(4)
        while service.isRunning && Date() < stopDeadline { try await Task.sleep(nanoseconds: 20_000_000) }
        XCTAssertFalse(service.isRunning)
        XCTAssertNil(service.serviceMessage)
        XCTAssertEqual(service.logText.utf8.count, 48 * 1024)
        XCTAssertTrue(service.logText.hasSuffix("FINAL-STDOUT\nFINAL-STDERR\n"))
        XCTAssertFalse(service.logText.contains("AFTER-8-MIB"))
    }
}
