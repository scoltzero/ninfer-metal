import Foundation
import SwiftUI

struct MIVFallbackPolicy: Codable, Equatable {
    var enabled = false
    var preferred = ""
}

struct HFCachedModel: Equatable {
    let directory, routeID, repoID: String
    let hasNativeWeights: Bool
}

enum HFCacheDiscovery {
    static func hub(environment: [String:String] = ProcessInfo.processInfo.environment,
                    home: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
        func path(_ raw: String) -> URL {
            URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
        }
        if let override = environment["HF_HUB_CACHE"], !override.isEmpty { return path(override) }
        if let base = environment["HF_HOME"], !base.isEmpty { return path(base).appendingPathComponent("hub") }
        return home.appendingPathComponent(".cache/huggingface/hub")
    }

    /// Mirrors Hub refs/main, then refs/master, and finally uses the newest usable revision.
    /// Snapshots usually contain symlinks into blobs; the checks follow them without changing either.
    static func scan(_ hub: URL) -> [HFCachedModel] {
        let fm = FileManager.default
        let repos = (try? fm.contentsOfDirectory(at: hub, includingPropertiesForKeys: [.isDirectoryKey],
                                                 options: [.skipsHiddenFiles])) ?? []
        var result: [HFCachedModel] = []
        for repo in repos where repo.lastPathComponent.hasPrefix("models--") {
            let route = String(repo.lastPathComponent.dropFirst("models--".count))
            guard !route.isEmpty else { continue }
            let snapshots = repo.appendingPathComponent("snapshots")
            let candidates = (try? fm.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey],
                                                          options: [.skipsHiddenFiles])) ?? []
            let refs = ["main", "master"].compactMap { name in
                (try? String(contentsOf: repo.appendingPathComponent("refs/" + name), encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            let ordered = candidates.sorted { lhs, rhs in
                let l = refs.firstIndex(of: lhs.lastPathComponent) ?? Int.max
                let r = refs.firstIndex(of: rhs.lastPathComponent) ?? Int.max
                if l != r { return l < r }
                let lm = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let rm = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return lm == rm ? lhs.lastPathComponent > rhs.lastPathComponent : lm > rm
            }
            for snapshot in ordered {
                guard fm.isReadableFile(atPath: snapshot.appendingPathComponent("config.json").path) else { continue }
                let native = ["model.miv", "model.omlxm5"].contains { readableWeight(snapshot.appendingPathComponent($0)) }
                guard native || hasCompleteSafetensors(at: snapshot) else { continue }
                let parts = route.components(separatedBy: "--")
                let source = parts.count > 1 ? parts[0] + "/" + parts.dropFirst().joined(separator: "--") : route
                result.append(HFCachedModel(directory: snapshot.resolvingSymlinksInPath().standardizedFileURL.path,
                                            routeID: route, repoID: source, hasNativeWeights: native))
                break
            }
        }
        return result.sorted { $0.repoID < $1.repoID }
    }

    private static func readableWeight(_ path: URL) -> Bool {
        let file = path.resolvingSymlinksInPath()
        guard let attrs = try? file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
              attrs.isRegularFile == true, (attrs.fileSize ?? 0) > 0 else { return false }
        return FileManager.default.isReadableFile(atPath: file.path)
    }

    private static func hasCompleteSafetensors(at folder: URL) -> Bool {
        let fm = FileManager.default
        let index = folder.appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: index.path) {
            struct Index: Decodable { let weight_map: [String:String] }
            guard let data = try? Data(contentsOf: index),
                  let map = try? JSONDecoder().decode(Index.self, from: data).weight_map, !map.isEmpty else { return false }
            return Set(map.values).allSatisfy { shard in
                !shard.isEmpty && !shard.contains("/") && shard.hasSuffix(".safetensors") &&
                    readableWeight(folder.appendingPathComponent(shard))
            }
        }
        return readableWeight(folder.appendingPathComponent("model.safetensors"))
    }
}

extension CoreService {
    func setHFDiscovery(_ enabled: Bool) {
        defaults.set(enabled, forKey: "hfCacheDiscovery")
        hfDiscoveryEnabled = enabled
        if !enabled { hfCacheModels = [] }
        Task { await refreshModelLibrary() }
    }

    func setFallbackPolicy(_ value: MIVFallbackPolicy) throws {
        let data = try JSONEncoder().encode(value)
        if let runningRoutingFile { try data.write(to: runningRoutingFile, options: .atomic) }
        defaults.set(data, forKey: "modelFallbackPolicy")
        fallbackPolicy = value
    }

    func createRoutingFile() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("MIV-routing-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        let file = root.appendingPathComponent("policy.json")
        do { try JSONEncoder().encode(fallbackPolicy).write(to: file, options: .atomic) }
        catch { try? FileManager.default.removeItem(at: root); throw error }
        return file
    }
}

struct MIVHFDiscoveryRow: View {
    @ObservedObject var service: CoreService
    var body: some View {
        Row(label: "从 Hugging Face 本地缓存发现模型",
            sublabel: "读取 Hub 快照目录，优先选择 main 或 master 对应的完整版本。") {
            RowSwitch(isOn: Binding(get: { service.hfDiscoveryEnabled }, set: service.setHFDiscovery))
        }
    }
}

struct MIVFallbackRow: View {
    @ObservedObject var service: CoreService
    @State private var error: String?
    var body: some View {
        Row(label: "模型回退", sublabel: "推理请求指定的模型未加载时，路由到已加载的模型。", warning: error) {
            RowSwitch(isOn: Binding(get: { service.fallbackPolicy.enabled },
                                    set: { save(.init(enabled: $0, preferred: service.fallbackPolicy.preferred)) }))
        }
        if service.fallbackPolicy.enabled {
            Row(label: "优先回退模型", sublabel: "优先模型不可用时，按加载顺序选择首个可用模型。") {
                let ids = Set(service.libraryRows.map(\.apiID) + service.models.map(\.id) + [service.fallbackPolicy.preferred]).subtracting([""])
                Picker("优先回退模型", selection: Binding(get: { service.fallbackPolicy.preferred },
                                                  set: { save(.init(enabled: true, preferred: $0)) })) {
                    Text("按加载顺序").tag("")
                    ForEach(ids.sorted(), id: \.self) { Text($0).tag($0) }
                }.labelsHidden().frame(maxWidth: 190)
            }
        }
    }
    private func save(_ value: MIVFallbackPolicy) {
        do { try service.setFallbackPolicy(value); error = nil }
        catch { self.error = error.localizedDescription }
    }
}
