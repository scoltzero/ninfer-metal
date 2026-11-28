import Foundation

struct MIVLibraryModel: Codable, Equatable, Identifiable {
    var directory: String
    var sourceID: String?
    var alias = ""
    var displayName = ""
    var favorite = false
    var enabled: Bool?
    var engine: [String: String] = [:]
    var generation: [String: String] = [:]
    var profiles: [String: [String: String]] = [:]
    var id: String { directory + "\0" + (sourceID ?? "") }
    var apiID: String { sourceID ?? URL(fileURLWithPath: directory).lastPathComponent }
    var title: String { displayName.isEmpty ? apiID : displayName }

    var migratingLegacyANE: Self {
        guard engine[MIVANEMode.legacyKey] != nil || engine[MIVANEMode.key]?.isEmpty == true else { return self }
        var model = self
        model.engine[MIVANEMode.key] = MIVANEMode.selection(in: engine)
        model.engine.removeValue(forKey: MIVANEMode.legacyKey)
        return model
    }

    func overlay() throws -> [String: Any] {
        var record: [String: Any] = ["directory": directory,
                                      "engine": try ModelField.typed(migratingLegacyANE.engine, fields: ModelField.engine),
                                      "generation": try ModelField.typed(generation, fields: ModelField.generation)]
        if let sourceID { record["source_id"] = sourceID }
        if let enabled { record["enabled"] = enabled }
        if !alias.isEmpty { record["alias"] = alias }
        return record
    }
}

struct MIVConfiguredModel: Codable {
    let id, directory: String
    let max_context, max_concurrency: Int
    let kv_dtype, spec_backend: String
    let engine: [String:String]?
}

extension CoreService {
    var libraryRoot: URL {
        if let custom = defaults.string(forKey: "modelLibraryRoot") { return URL(fileURLWithPath: custom) }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/MIV/Models")
    }

    var libraryRoots: [URL] {
        let additional = defaults.stringArray(forKey: "additionalModelRoots") ?? []
        return [libraryRoot] + additional.filter { $0 != libraryRoot.path }.map { URL(fileURLWithPath: $0) }
    }

    func addLibraryRoot(_ url: URL) throws {
        let path = Self.normalPath(url.path)
        var directory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: path, isDirectory: &directory), directory.boolValue else {
            throw CoreFailure("请选择已有的模型库目录。")
        }
        if !libraryRoots.contains(where: { $0.path == path }) {
            defaults.set(Array(libraryRoots.dropFirst().map(\.path)) + [path], forKey: "additionalModelRoots")
        }
        Task { await refreshModelLibrary() }
    }

    func removeLibraryRoot(_ url: URL) {
        var paths = libraryRoots.map(\.path).filter { $0 != Self.normalPath(url.path) }
        if !paths.isEmpty { defaults.set(paths.removeFirst(), forKey: "modelLibraryRoot") }
        else { defaults.removeObject(forKey: "modelLibraryRoot") }
        defaults.set(paths, forKey: "additionalModelRoots")
        Task { await refreshModelLibrary() }
    }

    func setLibraryRoot(_ url: URL) throws {
        var directory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &directory), directory.boolValue else {
            throw CoreFailure("请选择已有的模型库目录。")
        }
        let path = Self.normalPath(url.path)
        let others = libraryRoots.map(\.path).filter { $0 != path }
        defaults.set(path, forKey: "modelLibraryRoot")
        defaults.set(others, forKey: "additionalModelRoots")
        Task { await refreshModelLibrary() }
    }

    static func normalPath(_ path: String) -> String {
        URL(fileURLWithPath: (path as NSString).expandingTildeInPath).resolvingSymlinksInPath().standardizedFileURL.path
    }

    func registerModel(directory url: URL) throws {
        let path = Self.normalPath(url.path)
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw CoreFailure("未找到可用的模型目录。")
        }
        if !modelLibrary.contains(where: { $0.directory == path }) {
            let matching = configuredModels.filter { Self.normalPath($0.directory) == path }
            // Explicit TOML aliases distinguish variants sharing one directory.
            if matching.isEmpty {
                let cached = hfCacheModels.first { $0.directory == path }
                var added = MIVLibraryModel(directory: path, sourceID: cached?.routeID)
                added.displayName = cached?.repoID ?? ""
                try saveModel(added)
            }
            else { for item in matching { try saveModel(MIVLibraryModel(directory: path, sourceID: item.id)) } }
        }
    }

    func setGlobalGeneration(_ values: [String:String]) throws {
        _ = try ModelField.typed(values, fields: ModelField.generation)
        let cleaned = values.filter { !$0.value.isEmpty }
        defaults.set(cleaned, forKey: "globalGeneration")
        globalGeneration = cleaned
    }

    func saveModel(_ edited: MIVLibraryModel) throws {
        let model = edited.migratingLegacyANE
        _ = try model.overlay()
        var records = modelLibrary.filter {
            $0.id != model.id && !(model.sourceID != nil && $0.sourceID == nil && $0.directory == model.directory)
        }
        records.append(model)
        defaults.set(try JSONEncoder().encode(records), forKey: "modelLibrary")
        modelLibrary = records
    }

    func applyMeasuredContext(_ length: Int, modelID: String) throws {
        guard settings.mode == .managed else { throw CoreFailure("只能为本机托管的模型应用上下文设置。") }
        guard var model = libraryRows.first(where: { $0.apiID == modelID || $0.alias == modelID }) else {
            throw CoreFailure("请先在模型库中登记这个模型的目录。")
        }
        model.engine["max_context"] = String(length)
        try saveModel(model)
    }

    var libraryRows: [MIVLibraryModel] {
        var rows = modelLibrary
        for item in configuredModels {
            let directory = Self.normalPath(item.directory)
            if let index = rows.firstIndex(where: { $0.directory == directory && $0.sourceID == nil }) {
                rows[index].sourceID = item.id
            }
            let candidate = MIVLibraryModel(directory: directory, sourceID: item.id)
            if !rows.contains(where: { $0.id == candidate.id }) { rows.append(candidate) }
        }
        for cached in hfCacheModels where !rows.contains(where: { $0.directory == cached.directory }) {
            var item = MIVLibraryModel(directory: cached.directory, sourceID: cached.routeID)
            item.displayName = cached.repoID
            rows.append(item)
        }
        for path in scannedModelPaths where !rows.contains(where: { $0.directory == path }) {
            rows.append(MIVLibraryModel(directory: path))
        }
        return rows.sorted {
            if $0.favorite != $1.favorite { return $0.favorite }
            return $0.title.localizedStandardCompare($1.title) == .orderedAscending
        }
    }

    func needsPreparedHFWeights(_ model: MIVLibraryModel) -> Bool {
        hfCacheModels.contains { $0.directory == model.directory && !$0.hasNativeWeights } &&
            (model.engine["artifact"] ?? "").isEmpty &&
            !configuredModels.contains { Self.normalPath($0.directory) == model.directory }
    }

    func setModelLoaded(_ model: MIVLibraryModel, loaded: Bool) async throws {
        guard settings.mode == .managed, !isChangingModels else { throw CoreFailure("服务正在切换模型。") }
        if loaded && needsPreparedHFWeights(model) {
            throw CoreFailure("此快照包含 Safetensors。请先准备 MIV 权重，再在此模型的「MIV 核心」中选择主模型权重文件。")
        }
        isChangingModels = true
        defer { isChangingModels = false }
        var next = model
        next.enabled = loaded
        try saveModel(next)
        if isRunning {
            stop()
            while isRunning { try await Task.sleep(nanoseconds: 100_000_000) }
        }
        // Start only if there is a selected model; unloading the last one stops the core.
        if loaded || libraryRows.contains(where: { isModelSelected($0) }) { try start() }
    }

    func isModelSelected(_ model: MIVLibraryModel) -> Bool {
        model.enabled ?? configuredModels.contains {
            Self.normalPath($0.directory) == model.directory && (model.sourceID == nil || $0.id == model.sourceID)
        }
    }

    func refreshModelLibrary() async {
        guard settings.mode == .managed else { return }
        let roots = libraryRoots
        let cache = hfDiscoveryEnabled ? HFCacheDiscovery.hub() : nil
        let configuration = settings
        let source = effectiveConfigURL
        let executable = coreExecutable
        libraryRevision += 1
        let revision = libraryRevision
        let (paths, cached, items, error) = await Task.detached {
            let paths = Self.scanDirectories(roots)
            let cached = cache.map(HFCacheDiscovery.scan) ?? []
            guard let executable else { return (paths, cached, [MIVConfiguredModel](), Optional<String>.none) }
            do {
                let task = Process(), output = Pipe()
                task.executableURL = executable
                var args = try configuration.launchArguments()
                if configuration.configPath.isEmpty {
                    args += source.map { ["--config", $0.path] } ?? ["--no-default-config"]
                }
                task.arguments = args + ["--describe-models"]
                task.standardOutput = output
                task.standardError = FileHandle.nullDevice
                try task.run()
                let data = output.fileHandleForReading.readDataToEndOfFile()
                task.waitUntilExit()
                guard task.terminationStatus == 0 else { throw CoreFailure("核心无法读取当前配置，请检查 TOML 文件。") }
                return (paths, cached, try JSONDecoder().decode([MIVConfiguredModel].self, from: data), nil)
            } catch { return (paths, cached, [], Optional(error.localizedDescription)) }
        }.value
        guard settings == configuration, libraryRevision == revision else { return }
        hfCacheModels = cached
        scannedModelPaths = paths
        configuredModels = items
        libraryError = error
    }

    private nonisolated static func scanDirectories(_ roots: [URL]) -> [String] {
        let fm = FileManager.default
        var paths = Set<String>()
        for root in roots {
            if isModelDirectory(root) { paths.insert(root.resolvingSymlinksInPath().path); continue }
            guard let entries = fm.enumerator(at: root, includingPropertiesForKeys: [.isDirectoryKey],
                                               options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { continue }
            for case let url as URL in entries {
                guard (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true,
                      isModelDirectory(url) else { continue }
                paths.insert(url.resolvingSymlinksInPath().standardizedFileURL.path)
                entries.skipDescendants()
            }
        }
        return paths.sorted()
    }

    private nonisolated static func isModelDirectory(_ url: URL) -> Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: url.appendingPathComponent("config.json").path) ||
               fm.fileExists(atPath: url.appendingPathComponent("model.miv").path) ||
               fm.fileExists(atPath: url.appendingPathComponent("model.omlxm5").path)
    }

    func writeModelOverlay() throws -> URL? {
        guard !modelLibrary.isEmpty || !globalGeneration.isEmpty else { return nil }
        let contents: [String: Any] = ["models": try modelLibrary.map { try $0.overlay() },
            "defaults": try ModelField.typed(globalGeneration, fields: ModelField.generation)]
        let data = try JSONSerialization.data(withJSONObject: contents)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("MIV-models-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        let file = directory.appendingPathComponent("models.json")
        do { try data.write(to: file, options: .atomic) }
        catch { try? FileManager.default.removeItem(at: directory); throw error }
        return file
    }
}
