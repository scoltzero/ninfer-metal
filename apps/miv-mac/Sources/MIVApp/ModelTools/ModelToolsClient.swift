import Foundation

struct ToolStatus: Decodable { let success: Bool? }
struct MirrorPatch {
    var hfEndpoint: String?
    var msEndpoint: String?
}
struct ToolMirrors {
    struct Mirror { let endpoint: String }
    var huggingface: Mirror?
    var modelscope: Mirror?
}
struct ModelToolError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}
func toolByteCount(_ bytes: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
}

struct ModelToolsClient {
    let root: URL
    var roots: [URL] = []

    private func request<R: Decodable>(_ path: String, params: [String: Any]) async throws -> R {
        let env = await MainActor.run { MIVManagement.workerEnvironment() }
        let mirrors = try await getGlobalSettings()
        var network = env.filter { ["https_proxy", "http_proxy", "all_proxy", "no_proxy", "hf_endpoint", "modelscope_domain", "hf_token", "modelscope_api_token", "ssl_cert_file", "requests_ca_bundle", "curl_ca_bundle"].contains($0.key.lowercased()) }
        if let mirror = mirrors.huggingface?.endpoint, !mirror.isEmpty { network["HF_ENDPOINT"] = mirror }
        if let mirror = mirrors.modelscope?.endpoint, !mirror.isEmpty { network["MODELSCOPE_DOMAIN"] = mirror }
        let data = try JSONSerialization.data(withJSONObject: ["path": path, "params": params, "network": network, "roots": roots.map(\.path)])
        let bytes = try await ToolWorkers.shared.call(root: root, data: data, environment: env)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(R.self, from: bytes)
    }
    func get<R: Decodable>(_ path: String, query: [URLQueryItem] = []) async throws -> R {
        try await request(path, params: Dictionary(uniqueKeysWithValues: query.map { ($0.name, $0.value ?? "") }))
    }
    func post<R: Decodable, B: Encodable>(_ path: String, body: B) async throws -> R {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let object = try JSONSerialization.jsonObject(with: encoder.encode(body)) as? [String: Any] ?? [:]
        return try await request(path, params: object)
    }
    func postEmpty<R: Decodable>(_ path: String) async throws -> R { try await request(path, params: [:]) }
    func delete<R: Decodable>(_ path: String) async throws -> R { try await request(path, params: [:]) }

    func getGlobalSettings() async throws -> ToolMirrors {
        ToolMirrors(huggingface: .init(endpoint: UserDefaults.standard.string(forKey: "mivHFEndpoint") ?? ""),
                    modelscope: .init(endpoint: UserDefaults.standard.string(forKey: "mivMSEndpoint") ?? ""))
    }
    func updateGlobalSettings(_ patch: MirrorPatch) async throws -> ToolStatus {
        for (key, newValue) in [("mivHFEndpoint", patch.hfEndpoint), ("mivMSEndpoint", patch.msEndpoint)] {
            guard let raw = newValue else { continue }
            if !raw.isEmpty {
                guard let url = URLComponents(string: raw), ["https","http"].contains(url.scheme),
                      url.host != nil, url.user == nil, url.password == nil, url.fragment == nil,
                      url.query == nil else { throw ModelToolError(message: "镜像地址应为完整的 HTTP(S) 地址。") }
            }
            UserDefaults.standard.set(raw, forKey: key)
        }
        return ToolStatus(success: true)
    }
}

/// The worker owns long-running jobs across navigation. stdin/EOF is its
/// lifetime signal, so even an unexpected App exit releases the worker.
private actor ToolWorkers {
    static let shared = ToolWorkers()
    private var sessions: [URL: ToolWorker] = [:]
    func call(root: URL, data: Data, environment: [String:String]) async throws -> Data {
        let key = root.standardizedFileURL
        let worker = sessions[key] ?? ToolWorker()
        sessions[key] = worker
        return try await worker.call(root: key, data: data, environment: environment)
    }
}

struct ModelToolsRuntime {
    let resources: URL
    var python: URL { resources.appendingPathComponent("Python/cpython-3.11/bin/python3") }
    static func bundled() throws -> Self {
        guard let dir = Bundle.main.resourceURL else { throw ModelToolError(message: "缺少 App 工具资源。") }
        let runtime = Self(resources: dir.appendingPathComponent("ModelTools"))
        guard FileManager.default.isExecutableFile(atPath: runtime.python.path) else {
            throw ModelToolError(message: "当前 App 缺少内置模型工具运行时。")
        }
        return runtime
    }
    func environment(_ extra: [String:String]) -> [String:String] {
        var env = ProcessInfo.processInfo.environment
        env.merge(extra) { _, new in new }
        let paths = ["/opt/homebrew/bin", "/usr/local/bin", FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".local/bin").path]
        env["PATH"] = ([env["PATH"] ?? "/usr/bin:/bin"] + paths).joined(separator: ":")
        env["PYTHONHOME"] = resources.appendingPathComponent("Python/cpython-3.11").path
        env["PYTHONPATH"] = resources.path + ":" + resources.appendingPathComponent("Python/framework-mlx-base/lib/python3.11/site-packages").path
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        return env
    }
}

private actor ToolWorker {
    var child: Process?
    var input: FileHandle?
    var pending: [String: CheckedContinuation<Data, Error>] = [:]

    private func start(root: URL, environment: [String:String]) throws {
        let runtime = try ModelToolsRuntime.bundled()
        let process = Process(), requests = Pipe(), replies = Pipe()
        process.executableURL = runtime.python
        process.arguments = [runtime.resources.appendingPathComponent("worker.py").path, root.path]
        process.environment = runtime.environment(environment)
        process.standardInput = requests
        process.standardOutput = replies
        process.standardError = FileHandle.nullDevice
        try process.run()
        input = requests.fileHandleForWriting
        child = process
        let reader = replies.fileHandleForReading
        Task.detached {
            var buffer = Data()
            while true {
                let bytes = reader.availableData
                if bytes.isEmpty { break }
                buffer.append(bytes)
                while let end = buffer.firstIndex(of: 10) {
                    let line = Data(buffer[..<end])
                    buffer.removeSubrange(...end)
                    await self.deliver(line)
                }
            }
            await self.ended(process)
        }
    }
    private func ended(_ process: Process) {
        guard child === process else { return }
        child = nil; input = nil
        for item in pending.values { item.resume(throwing: ModelToolError(message: "模型工具已退出。请检查 App 的工具运行时。")) }
        pending.removeAll()
    }
    private func deliver(_ line: Data) {
        guard let object = try? JSONSerialization.jsonObject(with: line) as? [String:Any],
              let id = object["id"] as? String, let waiter = pending.removeValue(forKey: id) else { return }
        if let message = object["error"] as? String { waiter.resume(throwing: ModelToolError(message: message)); return }
        do { waiter.resume(returning: try JSONSerialization.data(withJSONObject: object["result"] ?? [:])) }
        catch { waiter.resume(throwing: error) }
    }
    func call(root: URL, data: Data, environment: [String:String]) async throws -> Data {
        if child == nil { try start(root: root, environment: environment) }
        let id = UUID().uuidString
        var request = try JSONSerialization.jsonObject(with: data) as? [String:Any] ?? [:]
        request["id"] = id
        var line = try JSONSerialization.data(withJSONObject: request)
        line.append(10)
        return try await withCheckedThrowingContinuation { continuation in
            pending[id] = continuation
            do { try input?.write(contentsOf: line) }
            catch { pending.removeValue(forKey: id); continuation.resume(throwing: error) }
        }
    }
}
