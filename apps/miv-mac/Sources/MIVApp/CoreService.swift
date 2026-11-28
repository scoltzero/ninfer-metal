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
    @Published private(set) var modelUsageCounters: CoreConnection.UsageCounters = [:]
    @Published private(set) var logText = ""
    private(set) var apiKey = ""

    @Published var hfDiscoveryEnabled: Bool
    @Published var hfCacheModels: [HFCachedModel] = []
    @Published var fallbackPolicy: MIVFallbackPolicy
    private(set) var runningRoutingFile: URL?
    @Published var modelLibrary: [MIVLibraryModel]
    @Published var globalGeneration: [String:String]
    @Published var configuredModels: [MIVConfiguredModel] = []
    @Published var scannedModelPaths: [String] = []
    @Published var libraryError: String?
    @Published var isChangingModels = false
    var libraryRevision = 0
    private var runningModelOverlay: URL?
    let defaults: UserDefaults
    private let explicitExecutable: URL?
    private var process: Process?
    private var outputTail: ManagedOutputTail?
    private var hasBoundListener = false
    private var generation = 0
    private var lastSample: (time: TimeInterval, tokens: Double)?
    private static let defaultsKey = "coreSettings"

    init(defaults: UserDefaults = .standard, executable: URL? = nil) {
        self.defaults = defaults
        hfDiscoveryEnabled = defaults.object(forKey: "hfCacheDiscovery") as? Bool ?? true
        fallbackPolicy = defaults.data(forKey: "modelFallbackPolicy").flatMap {
            try? JSONDecoder().decode(MIVFallbackPolicy.self, from: $0)
        } ?? MIVFallbackPolicy()
        self.explicitExecutable = executable
        self.globalGeneration = defaults.dictionary(forKey: "globalGeneration") as? [String:String] ?? [:]
        self.modelLibrary = defaults.data(forKey: "modelLibrary").flatMap {
            try? JSONDecoder().decode([MIVLibraryModel].self, from: $0)
        } ?? []
        settings = defaults.data(forKey: Self.defaultsKey)
            .flatMap { try? JSONDecoder().decode(CoreSettings.self, from: $0) } ?? CoreSettings()
    }

    var browserURL: URL? { try? settings.baseURL() }

    var coreExecutable: URL? {
        if let explicitExecutable { return explicitExecutable }
        let fm = FileManager.default
        let urls = [Bundle.main.resourceURL?.appendingPathComponent("Core/bin/ninfer-metal-serve"),
                    fm.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/MIV/Providers/io.ninfer.metal/current/bin/ninfer-metal-serve")]
        return urls.compactMap { $0 }.first { fm.isExecutableFile(atPath: $0.path) }
    }

    var effectiveConfigURL: URL? {
        guard settings.mode == .managed else { return nil }
        if !settings.configPath.isEmpty {
            return URL(fileURLWithPath: (settings.configPath as NSString).expandingTildeInPath)
        }
        let fm = FileManager.default
        let urls = [fm.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/MIV/ninfer-metal.toml"),
                    coreExecutable?.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("config/ninfer-metal.toml")]
        return urls.compactMap { $0 }.first { fm.isReadableFile(atPath: $0.path) }
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
        modelUsageCounters = [:]
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
        let key = settings.mode == .managed && apiKey.isEmpty ? MIVManagement.localMasterKey() : apiKey
        let client = CoreConnection(base: base, key: key)
        async let healthResult = result { try await client.health() }
        async let modelsResult = result { try await client.models() }
        async let metricsResult = result { try await client.metricsSnapshot() }
        async let hasDashboard = client.dashboardAvailable()
        let (h, m, stats, dashboard) = await (healthResult, modelsResult, metricsResult, hasDashboard)
        guard generation == version else { return }
        (health, healthError) = h
        models = m.0 ?? []
        modelsError = m.1
        metrics = stats.0?.total ?? [:]
        modelUsageCounters = stats.0?.perModel ?? [:]
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
        var args = try settings.launchArguments()
        guard let binary = coreExecutable, FileManager.default.isExecutableFile(atPath: binary.path) else {
            throw CoreFailure("未找到 ninfer-metal-serve。请使用配有核心的 MIV.app。")
        }
        try Self.checkPort(settings.port, host: settings.bindHost)
        let child = Process()
        child.executableURL = binary
        if settings.configPath.isEmpty {
            args += effectiveConfigURL.map { ["--config", $0.path] } ?? ["--no-default-config"]
        }
        args.append("--explicit-providers-only")
        args += MIVManagement.launchArguments()
        child.environment = try managementEnvironment(for: settings)
        let routing = try createRoutingFile()
        defer {
            if process !== child { try? FileManager.default.removeItem(at: routing.deletingLastPathComponent()) }
        }
        child.environment?["MIV_MODEL_FALLBACK_FILE"] = routing.path
        let overlay = try writeModelOverlay()
        defer {
            if process !== child, let overlay { try? FileManager.default.removeItem(at: overlay.deletingLastPathComponent()) }
        }
        if let overlay { args += ["--model-overrides", overlay.path] }
        child.arguments = args
        let output = try ManagedOutputTail(host: settings.bindHost, port: settings.port)
        child.standardOutput = output.writer
        child.standardError = output.writer
        defer { try? output.writer.close() }
        child.terminationHandler = { [weak self] ended in
            Task { @MainActor in self?.finished(ended) }
        }
        try child.run()
        invalidateSnapshot()
        logText = ""
        serviceMessage = nil
        hasBoundListener = false
        outputTail = output
        runningRoutingFile = routing
        runningModelOverlay = overlay
        process = child
        isRunning = true
        isStopping = false
    }

    private static func checkPort(_ port: Int, host: String) throws {
        let family = host == "::" ? AF_INET6 : AF_INET
        let fd = socket(family, SOCK_STREAM, 0)
        guard fd >= 0 else { throw CoreFailure("无法检查本机端口。") }
        defer { close(fd) }
        var reuse: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout.size(ofValue: reuse)))
        let status: Int32
        if family == AF_INET6 {
            var addr = sockaddr_in6()
            addr.sin6_len = UInt8(MemoryLayout<sockaddr_in6>.size)
            addr.sin6_family = sa_family_t(AF_INET6)
            addr.sin6_port = in_port_t(port).bigEndian
            status = withUnsafePointer(to: &addr) { ptr in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in6>.size))
                }
            }
        } else {
            var addr = sockaddr_in()
            addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
            addr.sin_family = sa_family_t(AF_INET)
            addr.sin_port = in_port_t(port).bigEndian
            addr.sin_addr.s_addr = inet_addr(host)
            status = withUnsafePointer(to: &addr) { ptr in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
        }
        guard status == 0 else { throw CoreFailure("无法监听本机端口 \(port)（\(String(cString: strerror(errno)))）。") }
    }

    private func readLog(final: Bool = false) {
        guard let outputTail else { return }
        let snapshot = outputTail.snapshot(final: final)
        if logText != snapshot.text { logText = snapshot.text }
        hasBoundListener = snapshot.listening
    }

    private func finished(_ child: Process) {
        guard process === child else { return }
        readLog(final: true)
        outputTail = nil
        if let url = runningRoutingFile {
            try? FileManager.default.removeItem(at: url.deletingLastPathComponent())
            runningRoutingFile = nil
        }
        if let url = runningModelOverlay {
            try? FileManager.default.removeItem(at: url.deletingLastPathComponent())
            runningModelOverlay = nil
        }
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

/// A single pipe preserves the ordering of the child's stdout and stderr writes.
/// A serial queue owns the fixed-size ring, independently of UI polling.
private final class ManagedOutputTail {
    private static let capacity = 48 * 1024
    let writer: FileHandle
    private let reader: FileHandle
    private let source: DispatchSourceRead
    private let queue = DispatchQueue(label: "io.ninfer.miv.core-output", qos: .utility)
    private var ring = [UInt8](repeating: 0, count: capacity)
    private var scratch = [UInt8](repeating: 0, count: 8192)
    private var cursor = 0
    private var used = 0
    private let marker: Data
    private var overlap = Data()
    private var listening = false

    init(host: String, port: Int) throws {
        let pipe = Pipe()
        reader = pipe.fileHandleForReading
        writer = pipe.fileHandleForWriting
        marker = Data("[ninfer-metal-serve] Listening on http://\(host):\(port) (".utf8)
        let fd = reader.fileDescriptor
        let flags = fcntl(fd, F_GETFL)
        guard flags >= 0, fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        source = DispatchSource.makeReadSource(fileDescriptor: fd, queue: queue)
        source.setEventHandler { [weak self] in self?.drain(upTo: 64 * 1024) }
        source.setCancelHandler { [reader] in try? reader.close() }
        source.resume()
    }

    deinit { source.cancel() }

    private func drain(upTo maximum: Int) {
        guard !source.isCancelled else { return }
        var remaining = maximum
        while remaining > 0 {
            let bytes = scratch.withUnsafeMutableBytes {
                Darwin.read(reader.fileDescriptor, $0.baseAddress!, min(remaining, $0.count))
            }
            if bytes < 0 && errno == EINTR { continue }
            if bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) { return }
            guard bytes > 0 else { source.cancel(); return }

            if !listening {
                overlap.append(contentsOf: scratch.prefix(bytes))
                listening = overlap.range(of: marker) != nil
                // Keep a possible partial marker even when it spans two reads.
                overlap = listening ? Data() : Data(overlap.suffix(marker.count - 1))
            }
            let first = min(bytes, Self.capacity - cursor)
            ring.replaceSubrange(cursor..<(cursor + first), with: scratch[..<first])
            if first < bytes { ring.replaceSubrange(0..<(bytes - first), with: scratch[first..<bytes]) }
            cursor = (cursor + bytes) % Self.capacity
            used = min(Self.capacity, used + bytes)
            remaining -= bytes
        }
    }

    func snapshot(final: Bool) -> (text: String, listening: Bool) {
        queue.sync {
            if !source.isCancelled {
                // Include bytes already written even if Dispatch has not delivered
                // the next read event yet. Darwin reports queued pipe bytes in st_size.
                // At exit this also avoids waiting for helpers that inherited the writer.
                var status = stat()
                if fstat(reader.fileDescriptor, &status) == 0 {
                    drain(upTo: Int(status.st_size))
                } else if final {
                    drain(upTo: Int.max)
                }
                if final { source.cancel() }
            }
            let start = (cursor + Self.capacity - used) % Self.capacity
            let first = min(used, Self.capacity - start)
            var bytes = Array(ring[start..<(start + first)])
            bytes.append(contentsOf: ring.prefix(used - first))
            let decoded = String(decoding: bytes, as: UTF8.self)
            // Malformed UTF-8 can expand during replacement. Keep the display
            // string within the same byte limit and begin at a scalar boundary.
            let suffix = decoded.utf8.suffix(Self.capacity).drop(while: { ($0 & 0xc0) == 0x80 })
            return (String(decoding: suffix, as: UTF8.self), listening)
        }
    }
}
