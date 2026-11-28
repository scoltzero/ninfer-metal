import Foundation

enum CoreMode: String, Codable, CaseIterable {
    case managed, external
}

struct CoreSettings: Codable, Equatable {
    var mode: CoreMode = .managed
    var endpoint = "http://127.0.0.1:8080"
    var port = 8080
    // Older preferences have no LAN flag. Absence is the loopback-only default.
    private var lanEnabled: Bool?
    var allowLAN: Bool {
        get { lanEnabled == true }
        set { lanEnabled = newValue ? true : nil }
    }
    private var ipv6Enabled: Bool?
    var allowIPv6: Bool {
        get { ipv6Enabled == true }
        set { ipv6Enabled = newValue ? true : nil }
    }
    var bindHost: String { allowLAN ? (allowIPv6 ? "::" : "0.0.0.0") : "127.0.0.1" }

    var modelDirectory = ""
    var configPath = ""
    var maxContext: Int?
    var maxConcurrency: Int?
    var kvDtype: String?
    var speculativeBackend: String?

    init() {}

    func baseURL() throws -> URL {
        if mode == .managed {
            guard (1...65535).contains(port) else { throw CoreFailure("端口须在 1 至 65535 之间。") }
            let clientHost = allowLAN && allowIPv6 ? "[::1]" : "127.0.0.1"
            return URL(string: "http://\(clientHost):\(port)")!
        }
        let text = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard var parts = URLComponents(string: text),
              ["http", "https"].contains(parts.scheme?.lowercased()),
              let host = parts.host, !host.isEmpty,
              parts.user == nil, parts.password == nil, parts.query == nil, parts.fragment == nil,
              ["", "/", "/v1", "/v1/"].contains(parts.path),
              (1...65535).contains(parts.port ?? 80)
        else { throw CoreFailure("请输入完整的 HTTP 或 HTTPS 服务地址，例如 http://127.0.0.1:8080。") }
        parts.path = ""
        guard let url = parts.url else { throw CoreFailure("服务地址无效。") }
        return url
    }

    func launchArguments() throws -> [String] {
        guard mode == .managed else { throw CoreFailure("请切换到本机托管模式。") }
        _ = try baseURL()
        if let maxContext, maxContext <= 0 || maxContext > Int(Int32.max) {
            throw CoreFailure("上下文长度须为正的 32 位整数。")
        }
        if let maxConcurrency, !(1...8).contains(maxConcurrency) {
            throw CoreFailure("并发槽位须在 1 至 8 之间。")
        }
        if let kvDtype, !["bf16", "affine4_g128"].contains(kvDtype) {
            throw CoreFailure("当前核心不支持该 KV 缓存格式。")
        }
        if let speculativeBackend, !["auto", "none", "mtp", "dflash2"].contains(speculativeBackend) {
            throw CoreFailure("当前核心不支持该推测解码后端。")
        }
        var args = ["--host", bindHost, "--port", String(port)]
        if !configPath.isEmpty { args += ["--config", (configPath as NSString).expandingTildeInPath] }
        if !modelDirectory.isEmpty { args += ["--model-dir", (modelDirectory as NSString).expandingTildeInPath] }
        if let maxContext { args += ["--max-context", String(maxContext)] }
        if let maxConcurrency { args += ["--max-concurrency", String(maxConcurrency)] }
        if let kvDtype { args += ["--kv-dtype", kvDtype] }
        if let speculativeBackend { args += ["--spec-backend", speculativeBackend] }
        return args
    }
}

struct CoreHealth: Decodable {
    let status, version: String
    let model, backend: String?
    let maxConcurrency, activeRequests, memoryBudgetBytes: Int?
    let models: [CoreModel]?
}

struct CoreModel: Decodable, Identifiable {
    let id: String
    let maxModelLen, maxContext, maxConcurrency, activeRequests, residentMemoryBytes: Int?
    let status, backend: String?
    let providerId, backendMode, loraAdapter, loraMode, kvDtype, specBackend: String?
    let prefillAneMode, prefillAneModeEffective, prefillAneInactiveReason: String?
    let prefillAneActive: Bool?
}

struct CoreFailure: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

private final class NoRedirects: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}

struct CoreConnection {
    let base: URL
    let key: String
    var session: URLSession = CoreConnection.ephemeralSession

    private static let ephemeralSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 5
        config.timeoutIntervalForResource = 8
        config.httpShouldSetCookies = false
        config.httpCookieStorage = nil
        config.urlCredentialStorage = nil
        config.urlCache = nil
        return URLSession(configuration: config, delegate: NoRedirects(), delegateQueue: nil)
    }()

    private func request(_ path: String, method: String = "GET") async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: path.isEmpty ? base : base.appendingPathComponent(path))
        request.httpMethod = method
        if path == "v1/models", !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw CoreFailure("服务未返回 HTTP 响应。") }
        if path == "health" && http.statusCode == 503 { return (data, http) }
        switch http.statusCode {
        case 200: return (data, http)
        case 401 where path == "v1/models", 403 where path == "v1/models":
            throw CoreFailure("模型接口认证失败，请检查 API 密钥（HTTP \(http.statusCode)）。")
        default: throw CoreFailure("服务返回 HTTP \(http.statusCode)。")
        }
    }

    private func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        do { return try decoder.decode(type, from: data) }
        catch { throw CoreFailure("服务返回的数据不符合 MIV 接口格式。") }
    }

    func health() async throws -> CoreHealth {
        try await decode(CoreHealth.self, from: request("health").0)
    }

    func models() async throws -> [CoreModel] {
        struct ModelList: Decodable { let data: [CoreModel] }
        return try await decode(ModelList.self, from: request("v1/models").0).data
    }

    func dashboardAvailable() async -> Bool {
        guard let (_, http) = try? await request("", method: "HEAD") else { return false }
        return http.mimeType == "text/html" && http.value(forHTTPHeaderField: "X-MIV-Dashboard") == "1"
    }

    func metrics() async throws -> [String: Double] {
        let (data, _) = try await request("metrics")
        return try Self.parseMetrics(data).total
    }

    typealias UsageCounters = [String: (requests: Double, output: Double)]
    typealias MetricsSnapshot = (total: [String: Double], perModel: UsageCounters)

    func metricsSnapshot() async throws -> MetricsSnapshot {
        let (data, _) = try await request("metrics")
        return try Self.parseMetrics(data)
    }

    private static func parseMetrics(_ data: Data) throws -> MetricsSnapshot {
        let text = String(decoding: data, as: UTF8.self)
        let total = aggregateMetrics(text)
        guard !total.isEmpty else { throw CoreFailure("服务未返回 MIV 性能指标。") }
        var models: UsageCounters = [:]
        for line in text.split(whereSeparator: \.isNewline) {
            let requests = line.hasPrefix("ninfer_metal_model_requests_total{")
            guard requests || line.hasPrefix("ninfer_metal_model_generated_tokens_total{"),
                  let label = line.range(of: "{model="),
                  let space = line.lastIndex(of: " "),
                  let count = Double(line[line.index(after: space)...]), count.isFinite else { continue }
            // The server emits the model label first and uses JSON-compatible escapes.
            let tail = line[label.upperBound...]
            var escaped = false
            guard tail.first == "\"" else { continue }
            for index in tail.indices.dropFirst() {
                if escaped { escaped = false; continue }
                if tail[index] == "\\" { escaped = true; continue }
                if tail[index] == "\"" {
                    guard let id = try? JSONDecoder().decode(String.self, from: Data(tail[...index].utf8)) else { break }
                    if requests { models[id, default: (0, 0)].requests = count }
                    else { models[id, default: (0, 0)].output = count }
                    break
                }
            }
        }
        return (total, models)
    }

    static func aggregateMetrics(_ text: String) -> [String: Double] {
        var result: [String: Double] = [:]
        for line in text.split(whereSeparator: \.isNewline) where line.hasPrefix("ninfer_metal_") {
            let fields = line.split(whereSeparator: \.isWhitespace)
            guard fields.count == 2, !fields[0].contains("{"),
                  let value = Double(fields[1]), value.isFinite else { continue }
            result[String(fields[0])] = value
        }
        return result
    }
}
