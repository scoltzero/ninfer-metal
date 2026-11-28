import Foundation

struct MIVManagementSettings: Codable, Equatable {
    var autoStart = false
    var httpProxy = ""
    var httpsProxy = ""
    var noProxy = ""
    var caBundle = ""
    var memoryBudget: Double?
    var prefixEnabled: Bool?
    var prefixBudget: Double?
    var logLevel: String?
    var recordUsage = true
}
struct MIVSubKey: Codable, Identifiable, Equatable {
    var id = UUID()
    var name: String
    var key: String
    var created = Date()
}
struct MIVAccess: Codable, Equatable {
    var master = ""
    var mode = "inherit"
    var keys: [MIVSubKey] = []
}

enum MIVManagement {
    static var root: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MIV", isDirectory: true)
    }
    static var settings: MIVManagementSettings {
        get {
            guard let data = UserDefaults.standard.data(forKey: "management") else { return .init() }
            return (try? JSONDecoder().decode(MIVManagementSettings.self, from: data)) ?? .init()
        }
        set { UserDefaults.standard.set(try? JSONEncoder().encode(newValue), forKey: "management") }
    }
    static func workerEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        let s = settings
        for (name, value) in [("HTTP_PROXY", s.httpProxy), ("HTTPS_PROXY", s.httpsProxy), ("NO_PROXY", s.noProxy)] where !value.isEmpty {
            env[name] = value; env[name.lowercased()] = value
        }
        if !s.caBundle.isEmpty {
            env["SSL_CERT_FILE"] = s.caBundle; env["REQUESTS_CA_BUNDLE"] = s.caBundle
            env["CURL_CA_BUNDLE"] = s.caBundle
        }
        return env
    }
    static func launchArguments() -> [String] {
        let s = settings
        var args: [String] = []
        if let budget = s.memoryBudget { args += ["--memory-budget-gb", String(budget)] }
        if let enabled = s.prefixEnabled { args += ["--rolling-prefix-cache", enabled ? "on" : "off"] }
        if let budget = s.prefixBudget { args += ["--prefix-cache-gb", String(budget)] }
        return args
    }
    static var logLevelURL: URL { root.appendingPathComponent("log-level") }
    static func applyLogLevel(_ level: String?) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try (level ?? "INFO").write(to: logLevelURL, atomically: true, encoding: .utf8)
    }
    static var securityURL: URL { root.appendingPathComponent("access.json") }
    static func access(at url: URL = securityURL) throws -> MIVAccess {
        guard FileManager.default.fileExists(atPath: url.path) else { return .init() }
        return try JSONDecoder().decode(MIVAccess.self, from: Data(contentsOf: url))
    }
    static func write(_ access: MIVAccess, at url: URL = securityURL) throws {
        if access.mode == "required" && access.master.isEmpty && access.keys.isEmpty {
            throw CoreFailure("请先设置主密钥或创建子密钥。")
        }
        let keys = ([access.master] + access.keys.map(\.key)).filter { !$0.isEmpty }
        guard Set(keys).count == keys.count, keys.allSatisfy({ k in
            k.count >= 4 && k.utf8.allSatisfy { (33...126).contains($0) }
        }) else { throw CoreFailure("密钥须为至少 4 位的可打印 ASCII 字符，不能重复或含空格。") }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        let data = try JSONEncoder().encode(access)
        let tmp = url.deletingLastPathComponent().appendingPathComponent(".access-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tmp) }
        guard FileManager.default.createFile(atPath: tmp.path, contents: data, attributes: [.posixPermissions: 0o600]) else {
            throw CoreFailure("无法写入本机密钥文件。")
        }
        guard rename(tmp.path, url.path) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
    }
    static func localMasterKey() -> String { (try? access().master) ?? "" }
    static func generateKey() -> String {
        "sk-miv-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    }
}
extension CoreService {
    func managementEnvironment(for settings: CoreSettings) throws -> [String: String] {
        var env = MIVManagement.workerEnvironment()
        if !FileManager.default.fileExists(atPath: MIVManagement.securityURL.path) { try MIVManagement.write(.init()) }
        _ = try MIVManagement.access()
        try MIVManagement.applyLogLevel(MIVManagement.settings.logLevel)
        env["MIV_LOG_LEVEL_FILE"] = MIVManagement.logLevelURL.path
        env["MIV_ACCESS_FILE"] = MIVManagement.securityURL.path
        return env
    }
}
