import Combine
import Darwin
import Foundation

@MainActor
final class CoreService: ObservableObject {
    @Published private(set) var settings: CoreSettings
    @Published private(set) var health: CoreHealth?
    @Published private(set) var models: [CoreModel] = []
    @Published private(set) var metrics: [String: Double] = [:]
    @Published private(set) var healthError: String?
    @Published private(set) var modelsError: String?
    @Published private(set) var metricsError: String?
    @Published private(set) var serviceMessage: String?
    @Published private(set) var isRunning = false
    @Published private(set) var isStopping = false
    @Published private(set) var isRefreshing = false
    @Published private(set) var dashboardAvailable = false
    @Published private(set) var outputRate: Double?
    @Published private(set) var logText = ""
    private(set) var apiKey = ""

    private let defaults: UserDefaults
    private let explicitExecutable: URL?
    private var process: Process?
    private var logReader: FileHandle?
    private var hasBoundListener = false
    private var generation = 0
    private var lastSample: (time: TimeInterval, tokens: Double)?
    private static let defaultsKey = "coreSettings"

    init(defaults: UserDefaults = .standard, executable: URL? = nil) {
        self.defaults = defaults
        self.explicitExecutable = executable
        settings = defaults.data(forKey: Self.defaultsKey)
            .flatMap { try? JSONDecoder().decode(CoreSettings.self, from: $0) } ?? CoreSettings()
    }

    var browserURL: URL? { try? settings.baseURL() }

    private var coreExecutable: URL? {
        if let explicitExecutable { return explicitExecutable }
        let env = ProcessInfo.processInfo.environment
        let providers = env["OMLX_PROVIDERS_DIR"].map { ($0 as NSString).expandingTildeInPath }
        let providerPath = "Library/Application Support/oMLX/Providers/io.ninfer.metal/current/bin/ninfer-metal-serve"
        let locations = [
            Bundle.main.resourceURL?.appendingPathComponent("Core/bin/ninfer-metal-serve"),
            providers.map { URL(fileURLWithPath: $0).appendingPathComponent("io.ninfer.metal/current/bin/ninfer-metal-serve") },
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(providerPath)
        ]
        return locations.compactMap { $0 }.first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    /// Mirrors the locations supported by both the native core and the oMLX-extend launcher.
    var effectiveConfigURL: URL? {
        guard settings.mode == .managed else { return nil }
        func file(_ path: String) -> URL { URL(fileURLWithPath: (path as NSString).expandingTildeInPath) }
        if !settings.configPath.isEmpty { return file(settings.configPath) }
        let candidates = [
            ProcessInfo.processInfo.environment["NINFER_METAL_CONFIG"].map(file),
            file("ninfer-metal.toml"),
            file("~/.config/ninfer-metal/ninfer-metal.toml"),
            file("~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/config/ninfer-metal.toml"),
            coreExecutable?.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("config/ninfer-metal.toml")
        ]
        return candidates.compactMap { $0 }.first { FileManager.default.isReadableFile(atPath: $0.path) }
    }

    func apply(_ newSettings: CoreSettings, apiKey newKey: String) throws {
        guard process == nil || newSettings == settings else { throw CoreFailure("请先停止托管服务，再更改启动或连接设置。") }
        _ = try newSettings.baseURL()
        if newSettings.mode == .managed { _ = try newSettings.launchArguments() }
        guard newKey.rangeOfCharacter(from: .controlCharacters) == nil else { throw CoreFailure("API 密钥包含无效字符。") }
        let stored = try JSONEncoder().encode(newSettings)
        if settings != newSettings || apiKey != newKey {
            invalidateSnapshot()
            serviceMessage = nil
        }
        settings = newSettings
        apiKey = newKey
        defaults.set(stored, forKey: Self.defaultsKey)
    }

    private func invalidateSnapshot() {
        generation += 1
        health = nil
        models = []
        metrics = [:]
        healthError = nil
        modelsError = nil
        metricsError = nil
        lastSample = nil
        outputRate = nil
        dashboardAvailable = false
        isRefreshing = false
    }

    func refresh() async {
        if settings.mode == .managed {
            readLog()
            guard let process, process.isRunning, !isStopping, hasBoundListener else { return }
        }
        guard !isRefreshing, let base = browserURL else { return }
        isRefreshing = true
        let version = generation
        let client = CoreConnection(base: base, key: apiKey)
        async let healthResult = result { try await client.health() }
        async let modelsResult = result { try await client.models() }
        async let metricsResult = result { try await client.metrics() }
        async let hasDashboard = client.dashboardAvailable()
        let (h, m, stats, dashboard) = await (healthResult, modelsResult, metricsResult, hasDashboard)
        guard generation == version else { return }
        (health, healthError) = h
        models = m.0 ?? []
        modelsError = m.1
        metrics = stats.0 ?? [:]
        metricsError = stats.1
        let now = ProcessInfo.processInfo.systemUptime
        outputRate = nil
        if let tokens = metrics["ninfer_metal_generated_tokens_total"] {
            if let previous = lastSample, tokens >= previous.tokens, now > previous.time {
                outputRate = (tokens - previous.tokens) / (now - previous.time)
            }
            lastSample = (now, tokens)
        } else {
            lastSample = nil
        }
        dashboardAvailable = dashboard
        isRefreshing = false
    }

    private func result<T>(_ load: () async throws -> T) async -> (T?, String?) {
        do { return (try await load(), nil) }
        catch { return (nil, error.localizedDescription) }
    }

    func start() throws {
        guard process == nil else { throw CoreFailure("托管服务已经启动。") }
        let args = try settings.launchArguments()
        guard let binary = coreExecutable, FileManager.default.isExecutableFile(atPath: binary.path) else {
            throw CoreFailure("未找到 ninfer-metal-serve。请使用配有核心的 MIV.app 或安装 oMLX 的 ninfer-metal provider。")
        }
        try Self.checkPort(settings.port, host: settings.bindHost)
        let child = Process()
        child.executableURL = binary
        child.arguments = args
        // Keep only a private, unlinked log for the running child. The UI reads the most recent 48 KiB.
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("miv-\(UUID().uuidString).log").path
        let fd = Darwin.open(path, O_WRONLY | O_CREAT | O_EXCL, 0o600)
        guard fd >= 0 else { throw CoreFailure("无法创建服务日志。") }
        let writer = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        defer { try? writer.close() }
        let reader: FileHandle
        do { reader = try FileHandle(forReadingFrom: URL(fileURLWithPath: path)) }
        catch { unlink(path); throw error }
        unlink(path)
        child.standardOutput = writer
        child.standardError = writer
        child.terminationHandler = { [weak self] ended in
            Task { @MainActor in self?.finished(ended) }
        }
        do { try child.run() }
        catch { try? reader.close(); throw error }
        invalidateSnapshot()
        logText = ""
        serviceMessage = nil
        hasBoundListener = false
        logReader = reader
        process = child
        isRunning = true
        isStopping = false
    }

    private static func checkPort(_ port: Int, host: String) throws {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw CoreFailure("无法检查本机端口。") }
        defer { close(fd) }
        var reuse: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout.size(ofValue: reuse)))
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(port).bigEndian
        addr.sin_addr.s_addr = inet_addr(host)
        let status = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard status == 0 else { throw CoreFailure("无法监听本机端口 \(port)（\(String(cString: strerror(errno)))）。") }
    }

    private func readLog() {
        guard let reader = logReader,
              let size = try? reader.seekToEnd() else { return }
        let tailSize: UInt64 = 48 * 1024
        do {
            try reader.seek(toOffset: size > tailSize ? size - tailSize : 0)
            let data = try reader.read(upToCount: Int(tailSize)) ?? Data()
            let text = String(decoding: data, as: UTF8.self)
            if text != logText { logText = text }
            // Do not attach to a pre-existing listener while our child is still loading weights.
            if !hasBoundListener {
                hasBoundListener = text.contains("[ninfer-metal-serve] Listening on http://\(settings.bindHost):\(settings.port) (")
            }
        } catch { return }
    }

    private func finished(_ child: Process) {
        guard process === child else { return }
        readLog()
        try? logReader?.close()
        logReader = nil
        process = nil
        if !isStopping {
            serviceMessage = "核心已退出（\(child.terminationStatus)），详情见运行日志。"
        }
        isRunning = false
        isStopping = false
        hasBoundListener = false
        invalidateSnapshot()
    }

    func stop() {
        guard let child = process, child.isRunning, !isStopping else { return }
        isStopping = true
        invalidateSnapshot()
        child.terminate()
        Task { [weak self, weak child] in
            try? await Task.sleep(nanoseconds: 10_000_000_000)
            if let child, self?.process === child, child.isRunning {
                kill(child.processIdentifier, SIGKILL)
            }
        }
    }

    func shutdown() {
        guard let child = process else { return }
        stop()
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while child.isRunning && ProcessInfo.processInfo.systemUptime < deadline {
            Thread.sleep(forTimeInterval: 0.04)
        }
        if child.isRunning { kill(child.processIdentifier, SIGKILL) }
        child.waitUntilExit()
        finished(child)
    }
}
