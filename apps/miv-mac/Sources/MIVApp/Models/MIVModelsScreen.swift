import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct MIVModelsScreen: View {
    @ObservedObject var service: CoreService
    @State private var selected: MIVLibraryModel?
    @State private var inspected: CoreModel?
    @State private var search = ""
    @State private var notice: String?
    private var managed: Bool { service.settings.mode == .managed }

    var body: some View {
        VStack(spacing: 0) {
            if let model = selected, managed {
                MIVModelSettings(service: service, initial: model) { selected = nil }
                    .id(model.id)
            } else if let model = inspected {
                Button("‹  模型") { inspected = nil }.frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 24)
                SectionHeader(model.id)
                ListGroup {
                    Row(label: "上下文窗口") { Text(model.maxModelLen.map(String.init) ?? "—") }
                    Row(label: "并发槽位") { Text(model.maxConcurrency.map(String.init) ?? "—") }
                    Row(label: "KV 缓存格式") { Text(model.kvDtype ?? "—") }
                    Row(label: "推测解码后端") { Text(model.specBackend ?? "—") }
                    Row(label: "运行后端", isLast: true) { Text(model.backendMode ?? "—") }
                }
            } else {
                SectionHeader("已加载的模型") { Text("\(service.models.count)").foregroundStyle(.secondary) }
                ListGroup {
                    if service.models.isEmpty {
                        Row(label: service.modelsError ?? "当前没有已加载的模型", isLast: true) { EmptyView() }
                    }
                    ForEach(service.models) { model in
                        Row(label: model.id, sublabel: [model.backendMode, model.kvDtype, model.specBackend].compactMap { $0 }.joined(separator: " · ")) {
                            if managed, let local = service.libraryRows.first(where: { $0.apiID == model.id }) {
                                Button("卸载") { change(local, load: false) }
                                Button("设置") { selected = local }
                            } else {
                                Button("详情") { inspected = model }
                            }
                        }
                    }
                }
                if managed {
                    SectionHeader("模型库", subtitle: "加载或卸载会重新启动本机核心，并重新加载其余选中的模型。") {
                        Button("添加模型…") { addModel() }
                        Button { Task { await service.refreshModelLibrary() } } label: { Image(systemName: "arrow.clockwise") }
                            .accessibilityLabel("重新扫描模型库")
                    }
                    TextField("搜索模型名称、标识或目录", text: $search).textFieldStyle(.roundedBorder)
                        .padding(.horizontal, 14).padding(.bottom, 6)
                    ListGroup {
                        let rows = service.libraryRows.filter {
                            search.isEmpty || [$0.title, $0.alias, $0.directory].contains { $0.localizedCaseInsensitiveContains(search) }
                        }
                        if rows.isEmpty {
                            Row(label: search.isEmpty ? "模型库为空" : "没有匹配的模型",
                                sublabel: search.isEmpty ? "添加单个模型，或在服务器设置中指定模型目录。" : nil, isLast: true) { EmptyView() }
                        }
                        ForEach(rows) { item in
                            let active = service.models.contains { $0.id == item.apiID }
                            FreeRow(isLast: item.id == rows.last?.id) {
                                HStack(spacing: 10) {
                                    ModelIconTile(systemSymbol: "text.alignleft")
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(item.title).font(.system(size: 13, weight: .medium))
                                        Text(item.alias.isEmpty ? item.directory : item.alias + " · " + item.directory)
                                            .font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                                    }
                                    Spacer()
                                    Button {
                                        var edited = item; edited.favorite.toggle()
                                        do { try service.saveModel(edited) } catch { notice = error.localizedDescription }
                                    } label: {
                                        Image(systemName: item.favorite ? "star.fill" : "star")
                                            .foregroundStyle(item.favorite ? Color.orange : .secondary)
                                    }.buttonStyle(.plain).accessibilityLabel(item.favorite ? "取消收藏" : "收藏")
                                    if !active && service.needsPreparedHFWeights(item) {
                                        Button("选择 MIV 权重…") { selected = item }
                                    } else {
                                        Button(active ? "卸载" : "加载") { change(item, load: !active) }
                                            .disabled(service.isChangingModels || service.isStopping)
                                    }
                                    Button { selected = item } label: { Image(systemName: "chevron.right") }
                                        .accessibilityLabel("设置 " + item.title)
                                }
                                .contentShape(Rectangle())
                                .contextMenu {
                                    Button("模型设置…") { selected = item }
                                    Button("复制模型标识") { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(item.apiID, forType: .string) }
                                    Button("在访达中显示") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: item.directory)]) }
                                }
                            }
                        }
                    }
                    if let error = service.libraryError { Text(error).foregroundStyle(.red).padding(8) }
                }
                if let notice { Text(notice).foregroundStyle(.red).padding(8) }
            }
        }
        .controlSize(.small)
        .onChange(of: selected?.id) { _, _ in NotificationCenter.default.post(name: .init("MIVScrollToTop"), object: nil) }
        .task(id: service.settings) { await service.refreshModelLibrary() }
        .onReceive(NotificationCenter.default.publisher(for: .init("MIVShowModel"))) { notification in
            guard let id = notification.object as? String else { return }
            selected = service.libraryRows.first { $0.id == id || $0.apiID == id }
        }
    }

    private func change(_ item: MIVLibraryModel, load: Bool) {
        Task {
            do { try await service.setModelLoaded(item, loaded: load); notice = nil }
            catch { notice = error.localizedDescription }
        }
    }

    private func addModel() {
        let picker = NSOpenPanel()
        picker.canChooseDirectories = true; picker.canChooseFiles = false
        picker.title = "选择模型目录"; picker.prompt = "添加"
        if picker.runModal() == .OK, let url = picker.url {
            do { try service.registerModel(directory: url); notice = nil }
            catch { notice = error.localizedDescription }
        }
    }
}

private enum ModelPage: String, CaseIterable, Identifiable {
    case overview = "基本与生成", acceleration = "推理加速", memory = "内存与缓存", advanced = "高级", miv = "MIV 核心", profiles = "采样配置"
    var id: Self { self }
}

private struct MIVModelSettings: View {
    @ObservedObject var service: CoreService
    @State private var draft: MIVLibraryModel
    @State private var page = ModelPage.overview
    @State private var message: String?
    @State private var profile = ""
    var back: () -> Void

    init(service: CoreService, initial: MIVLibraryModel, back: @escaping () -> Void) {
        self.service = service
        _draft = State(initialValue: initial)
        self.back = back
    }

    private var running: CoreModel? { service.models.first { $0.id == draft.apiID } }
    private var configured: MIVConfiguredModel? {
        service.configuredModels.first { CoreService.normalPath($0.directory) == draft.directory && (draft.sourceID == nil || $0.id == draft.sourceID) }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Button {
                    if page == .overview { back() } else { page = .overview }
                } label: { Label(page == .overview ? "模型" : "模型设置", systemImage: "chevron.backward") }
                Spacer()
                Button("在访达中显示") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: draft.directory)]) }
            }.padding(.horizontal, 18)
            SectionHeader(draft.title, subtitle: draft.apiID)
            if let model = running {
                Text("已加载 · " + [model.kvDtype, model.specBackend, model.maxModelLen.map { "\($0) tokens" }].compactMap { $0 }.joined(separator: " · "))
                    .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 28).font(.caption)
            }
            switch page {
            case .overview:
                ListGroup {
                    pageRow(.profiles, symbol: "square.stack.3d.up.fill", gradient: SquircleGradient.models, last: true)
                }
                SectionHeader("基本设置")
                ListGroup {
                    Row(label: "显示名称", sublabel: "在本机模型库和菜单栏中显示。") {
                        TextField(draft.apiID, text: $draft.displayName).frame(width: 172)
                    }
                    Row(label: "API 别名", sublabel: "客户端可使用别名或原有模型标识。") {
                        TextField(draft.apiID, text: $draft.alias).frame(width: 172)
                    }
                    Row(label: "收藏", sublabel: "在模型库中优先显示。", isLast: true) {
                        RowSwitch(isOn: $draft.favorite)
                    }
                }
                fields([.context, .concurrency], title: "加载设置")
                fields(ModelField.generation.filter { !["reasoning_effort","chat_template_kwargs"].contains($0.id) },
                       title: "生成默认值", generation: true)
                Button("读取模型的 generation_config.json") {
                    do { draft.generation.merge(try generationFile()) { _, new in new }; message = "已读取生成参数，保存后生效。" }
                    catch { message = error.localizedDescription }
                }.padding(.bottom, 4)
                ListGroup {
                    pageRow(.acceleration, symbol: "bolt.fill", gradient: SquircleGradient.performance)
                    pageRow(.memory, symbol: "memorychip.fill", gradient: SquircleGradient.downloads)
                    pageRow(.advanced, symbol: "gearshape.fill", gradient: SquircleGradient.server)
                    pageRow(.miv, symbol: "cpu.fill", gradient: SquircleGradient.quantization, last: true)
                }.padding(.top, 14)
            case .acceleration: fields(ModelField.acceleration, title: "推测解码")
            case .memory: fields(ModelField.memory, title: "统一内存")
            case .advanced:
                fields(ModelField.advanced, title: "适配器与模板")
                fields(ModelField.generation.filter { ["reasoning_effort","chat_template_kwargs"].contains($0.id) },
                       title: "对话模板参数", generation: true)
            case .miv:
                SectionHeader("配置来源", subtitle: "单模型设置优先于本机启动默认值及 TOML。留空继承原配置；调整这些参数需要重新加载模型。生成请求显式指定的值优先于单模型和全局生成默认值。")
                ListGroup {
                    Row(label: "外部配置文件", sublabel: service.effectiveConfigURL?.path ?? "使用核心内置默认值", isLast: true) {
                        if let url = service.effectiveConfigURL {
                            Button("在访达中显示") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                        }
                    }
                }
                fields(ModelField.miv, title: "MIV 核心参数")
            case .profiles:
                SectionHeader("生成配置", subtitle: "配置按此模型单独保存。应用配置后请保存设置并重新加载。")
                ListGroup {
                    Row(label: "保存当前生成参数") {
                        TextField("配置名称", text: $profile).frame(width: 130)
                        Button("保存配置") {
                            do {
                                _ = try ModelField.typed(draft.generation, fields: ModelField.generation)
                                draft.profiles[profile] = draft.generation
                                save(reload: false)
                            } catch { message = error.localizedDescription }
                        }.disabled(profile.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    ForEach(draft.profiles.keys.sorted(), id: \.self) { name in
                        Row(label: name) {
                            Button("应用") { draft.generation = draft.profiles[name] ?? [:]; message = "已应用到当前编辑，保存后生效。" }
                            Button("删除") { draft.profiles.removeValue(forKey: name); save(reload: false) }
                        }
                    }
                }
            }
            HStack {
                Button("重置当前分栏") { resetPage() }
                Spacer()
                Button("保存设置") { save(reload: false) }
                Button(running == nil ? "保存并加载" : "保存并重新加载") { save(reload: true) }
                    .buttonStyle(.borderedProminent).disabled(service.isChangingModels || service.isStopping)
            }.padding(.horizontal, 18).padding(.vertical, 12)
            if let message { Text(message).font(.callout).textSelection(.enabled).padding(.horizontal, 18) }
        }
        .onChange(of: page) { _, _ in NotificationCenter.default.post(name: .init("MIVScrollToTop"), object: nil) }
    }

    private func pageRow(_ destination: ModelPage, symbol: String, gradient: [Color], last: Bool = false) -> some View {
        NavRow(label: destination.rawValue, isLast: last, action: { page = destination }) {
            Squircle(systemSymbol: symbol, gradient: gradient)
        }
    }

    private func fields(_ fields: [ModelField], title: String, generation: Bool = false) -> some View {
        VStack(spacing: 0) {
            SectionHeader(title, subtitle: generation ? "留空继承全局默认值。请求中显式传入的参数优先。" : "留空或选择「继承」来使用启动配置。")
            ListGroup {
                ForEach(fields) { field in
                    Row(label: field.label, sublabel: field.detail, mivSetting: !generation, isLast: field.id == fields.last?.id) {
                        control(field, generation: generation)
                    }
                }
            }
        }
    }

    @ViewBuilder private func control(_ field: ModelField, generation: Bool) -> some View {
        let value = Binding(get: { (generation ? draft.generation : draft.engine)[field.id] ?? "" },
                            set: { new in
                                if generation { draft.generation[field.id] = new }
                                else { draft.engine[field.id] = new }
                            })
        switch field.kind {
        case .choice(let choices):
            Picker(field.label, selection: value) {
                Text(inherited(field, generation: generation)).tag("")
                ForEach(choices, id: \.0) { Text($0.1).tag($0.0) }
            }.labelsHidden().frame(maxWidth: 155)
        case .directory, .file:
            HStack {
                TextField(inherited(field, generation: generation), text: value).frame(minWidth: 60, maxWidth: 138)
                Button("选择…") {
                    let picker = NSOpenPanel()
                    if case .directory = field.kind { picker.canChooseDirectories = true; picker.canChooseFiles = false }
                    else { picker.canChooseDirectories = false; picker.canChooseFiles = true }
                    if picker.runModal() == .OK, let url = picker.url { value.wrappedValue = url.path }
                }
            }
        default:
            TextField(inherited(field, generation: generation), text: value).frame(width: 115)
        }
    }

    private func inherited(_ field: ModelField, generation: Bool) -> String {
        if generation { return service.globalGeneration[field.id].map { "继承 · " + $0 } ?? "继承" }
        guard let configured else { return "继承" }
        if let raw = configured.engine?[field.id], !raw.isEmpty {
            if case .choice(let options) = field.kind, let localized = options.first(where: { $0.0 == raw })?.1 {
                return "继承 · " + localized
            }
            return "继承 · " + raw
        }
        switch field.id {
        case "max_context": return "继承 · \(configured.max_context)"
        case "max_concurrency": return "继承 · \(configured.max_concurrency)"
        default: return "继承"
        }
    }

    private func generationFile() throws -> [String: String] {
        let url = URL(fileURLWithPath: draft.directory).appendingPathComponent("generation_config.json")
        let raw = try Data(contentsOf: url)
        guard let object = try JSONSerialization.jsonObject(with: raw) as? [String:Any] else {
            throw CoreFailure("generation_config.json 必须是 JSON 对象。")
        }
        var values: [String:String] = [:]
        for field in ModelField.generation {
            guard let value = object[field.id] ?? (field.id == "max_tokens" ? object["max_new_tokens"] : nil) else { continue }
            if let text = value as? String { values[field.id] = text }
            else if let number = value as? NSNumber {
                values[field.id] = CFGetTypeID(number) == CFBooleanGetTypeID() ? String(number.boolValue) : number.stringValue
            } else if JSONSerialization.isValidJSONObject(value) {
                values[field.id] = String(decoding: try JSONSerialization.data(withJSONObject: value), as: UTF8.self)
            }
        }
        if object["do_sample"] as? Bool == false { values["temperature"] = "0" }
        guard !values.isEmpty else { throw CoreFailure("该文件没有可用于 MIV 的生成参数。") }
        _ = try ModelField.typed(values, fields: ModelField.generation)
        return values
    }

    private func save(reload: Bool) {
        do {
            try service.saveModel(draft)
            if reload {
                Task {
                    do { try await service.setModelLoaded(draft, loaded: true); draft.enabled = true; message = "核心正在加载新配置。" }
                    catch { message = error.localizedDescription }
                }
            } else { message = "已保存。核心下次加载此模型时生效。" }
        } catch { message = error.localizedDescription }
    }

    private func resetPage() {
        let keys: [ModelField]
        switch page {
        case .overview: draft.generation = [:]; draft.displayName = ""; draft.alias = ""; draft.favorite = false; keys = [.context, .concurrency]
        case .acceleration: keys = ModelField.acceleration
        case .memory: keys = ModelField.memory
        case .advanced: draft.generation.removeValue(forKey: "chat_template_kwargs"); draft.generation.removeValue(forKey: "reasoning_effort"); keys = ModelField.advanced
        case .miv: keys = ModelField.miv
        case .profiles: profile = ""; return
        }
        for field in keys { draft.engine.removeValue(forKey: field.id) }
        message = "当前分栏已恢复继承，保存后生效。"
    }
}
