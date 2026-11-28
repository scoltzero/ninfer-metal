import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct ServerManagementPage: View {
    let section: AppSection
    @ObservedObject var service: CoreService
    @State private var startup: CoreSettings
    @State private var prefs = MIVManagement.settings
    @State private var port: String
    @State private var context: String
    @State private var memory: String
    @State private var prefix: String
    @State private var generations: [String: String]
    @State private var clientKey: String
    @State private var note = ""
    @State private var failed = false
    @State private var saving = false
    @State private var showSampling = false

    init(section: AppSection, service: CoreService) {
        self.section = section; self.service = service
        _startup = State(initialValue: service.settings)
        _port = State(initialValue: String(service.settings.port))
        _context = State(initialValue: service.settings.maxContext.map(String.init) ?? "")
        _memory = State(initialValue: MIVManagement.settings.memoryBudget.map { String($0) } ?? "")
        _prefix = State(initialValue: MIVManagement.settings.prefixBudget.map { String($0) } ?? "")
        _generations = State(initialValue: service.globalGeneration)
        _clientKey = State(initialValue: service.apiKey)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if section == .security { MIVSecurityPage(service: service) }
            else {
                switch section {
                case .server: server
                case .network: network
                case .performance: performance
                default: EmptyView()
                }
                if section != .performance || service.settings.mode == .managed {
                    HStack {
                        if !note.isEmpty { Text(note).font(.caption).foregroundStyle(failed ? Color.red : .secondary) }
                        Spacer()
                        Button("恢复默认值", action: reset).buttonStyle(.omlx(.normal))
                        Button(saving ? "正在应用…" : "应用") { Task { await apply() } }
                            .buttonStyle(.omlx(.primary))
                    }.padding(14).disabled(saving)
                }
            }
        }
        .onChange(of: startup.endpoint) { _, _ in clientKey = "" }
        .onChange(of: startup.mode) { _, _ in clientKey = "" }
        .onChange(of: port) { _, _ in clientKey = "" }
    }

    @ViewBuilder private var server: some View {
        MIVServerHero(service: service)
        SectionHeader("网络")
        ListGroup {
            Row(label: "运行方式") {
                Picker("运行方式", selection: $startup.mode) {
                    Text("本机托管").tag(CoreMode.managed); Text("连接现有服务").tag(CoreMode.external)
                }.labelsHidden().frame(width: 158)
            }
            if startup.mode == .managed {
                Row(label: "监听地址") {
                    Picker("监听地址", selection: Binding(get: { startup.allowLAN ? (startup.allowIPv6 ? 2 : 1) : 0 },
                                                     set: { startup.allowLAN = $0 != 0; startup.allowIPv6 = $0 == 2 })) {
                        Text("127.0.0.1（仅本机）").tag(0)
                        Text("0.0.0.0（IPv4）").tag(1)
                        Text("::（IPv4 和 IPv6）").tag(2)
                    }.labelsHidden().frame(width: 178)
                }
                field("端口", "默认 8080。运行中的服务将在应用后重启。", $port)
            } else {
                field("服务地址", "现有 MIV 服务的 HTTP 或 HTTPS 地址。", $startup.endpoint)
            }
            Row(label: "连接密钥", sublabel: "仅在本次会话中用于连接所选服务。", isLast: true) {
                SecureField("API 密钥", text: $clientKey).frame(maxWidth: 210)
            }
        }
        if let base = service.browserURL {
            SectionHeader("API 端点")
            ListGroup {
                ForEach([("OpenAI 兼容", "v1"), ("健康检查", "health"), ("性能指标", "metrics")], id: \.1) { label, path in
                    Row(label: label) {
                        let url = base.appendingPathComponent(path)
                        Text(url.absoluteString).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                        Button { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(url.absoluteString, forType: .string) }
                        label: { Image(systemName: "doc.on.doc") }.buttonStyle(.plain).help("复制端点")
                    }
                }
            }
        }
        if startup.mode == .managed {
            SectionHeader("默认配置", subtitle: "单模型设置可覆盖以下值，请求中明确指定的采样参数优先。")
            ListGroup {
                field("上下文窗口", "留空继承所选 TOML。", $context)
                ForEach(Array(Self.sampling.prefix(showSampling ? Self.sampling.count : 4)), id: \.0) { key, name in
                    field(name, nil, Binding(get: { generations[key] ?? "" }, set: { generations[key] = $0 }))
                }
                Row(label: nil, isLast: true) { Button(showSampling ? "收起" : "显示全部字段…") { showSampling.toggle() } }
            }
            SectionHeader("服务启动")
            ListGroup { Row(label: "应用启动时自动启动服务器", isLast: true) { RowSwitch(isOn: $prefs.autoStart) } }
            SectionHeader("日志")
            ListGroup { Row(label: "请求日志级别", sublabel: "应用后立即影响本机核心记录的 HTTP 请求。", isLast: true) {
                Picker("日志级别", selection: Binding(get: { prefs.logLevel ?? "INFO" }, set: { prefs.logLevel = $0 })) {
                    ForEach(["ERROR","WARNING","INFO","DEBUG","TRACE"], id: \.self) { Text($0).tag($0) }
                }.labelsHidden().frame(width: 125)
            } }
            SectionHeader("用量历史")
            ListGroup { Row(label: "记录用量历史", sublabel: "关闭后保留已有记录。", isLast: true) { RowSwitch(isOn: $prefs.recordUsage) } }
            SectionHeader("存储")
            ListGroup {
                MIVHFDiscoveryRow(service: service)
                path("模型目录", "含已准备权重的目录；留空时使用模型库或 TOML。", $startup.modelDirectory, true)
                path("核心配置", "可显式选择已有的 ninfer-metal.toml。", $startup.configPath, false)
                if let u = service.effectiveConfigURL { Row(label: nil) { Button("在访达中显示生效配置") { NSWorkspace.shared.activateFileViewerSelecting([u]) } } }
            }
            SectionHeader("模型目录", subtitle: "第一项用于下载，所有目录都用于发现本地模型。")
            ListGroup {
                ForEach(service.libraryRoots, id: \.path) { root in
                    Row(label: root.lastPathComponent, sublabel: root.path) {
                        if root == service.libraryRoot { Text("主目录").foregroundStyle(.secondary) }
                        else { Button("设为主目录") { modify { try service.setLibraryRoot(root) } } }
                        if service.libraryRoots.count > 1 { Button("移除") { service.removeLibraryRoot(root) } }
                    }
                }
                Row(label: nil, isLast: true) { Button("添加模型目录…") {
                    if let url = Self.choose(directory: true) { modify { try service.addLibraryRoot(url) } }
                } }
            }
        }
    }
    static let sampling = [("max_tokens","最大输出 tokens"),("temperature","温度"),("top_p","Top P"),("top_k","Top K"),
                           ("repetition_penalty","重复惩罚"),("presence_penalty","存在惩罚"),("frequency_penalty","频率惩罚"),
                           ("seed","随机种子"),("thinking_budget","思考预算")]

    @ViewBuilder private var network: some View {
        SectionHeader("代理", subtitle: "用于此 App 发起的模型下载、上传等出站连接。")
        ListGroup {
            field("HTTP 代理", nil, $prefs.httpProxy); field("HTTPS 代理", nil, $prefs.httpsProxy)
            field("绕过代理", "以逗号分隔的主机、域名或网段。", $prefs.noProxy)
        }
        SectionHeader("TLS")
        ListGroup { path("CA 证书包", "自定义 PEM 根证书文件。", $prefs.caBundle, false) }
    }
    @ViewBuilder private var performance: some View {
        if service.settings.mode == .external {
            SectionHeader("远程实例", subtitle: "加载参数在托管这台服务器的 MIV App 上配置。")
        } else {
            SectionHeader("调度器", subtitle: "全局默认值可由单模型配置覆盖。")
            ListGroup { Row(label: "最大并发请求数", isLast: true) {
                Picker("最大并发请求数", selection: $startup.maxConcurrency) {
                    Text("继承配置").tag(Int?.none)
                    ForEach(1...8, id: \.self) { Text(String($0)).tag(Optional($0)) }
                }.labelsHidden().frame(width: 120)
            } }
            SectionHeader("内存与生命周期")
            ListGroup {
                field("统一内存预算（GiB）", "总模型驻留预算。留空使用核心默认值。", $memory)
                MIVFallbackRow(service: service)
            }
            SectionHeader("缓存", subtitle: "MIV 使用统一内存中的滚动前缀缓存。")
            ListGroup {
                Row(label: "滚动前缀缓存") { Picker("滚动前缀缓存", selection: $prefs.prefixEnabled) {
                    Text("继承配置").tag(Bool?.none); Text("开启").tag(Optional(true)); Text("关闭").tag(Optional(false))
                }.labelsHidden().frame(width: 120) }
                field("前缀缓存预算（GiB）", "留空继承配置。", $prefix)
            }
        }
        SectionHeader("实时负载")
        ListGroup {
            ForEach([("活动请求","active_requests"),("排队请求","queued_requests"),("输入 tokens","prompt_tokens_total"),("输出 tokens","generated_tokens_total"),("推测轮数","spec_rounds_total"),("前缀完整命中","prefix_cache_hits_total")], id: \.1) { label, key in
                Row(label: label) { Text(service.metrics["ninfer_metal_" + key]?.formatted(.number.precision(.fractionLength(0))) ?? "—").monospacedDigit() }
            }
        }
    }
    private func field(_ label: String, _ help: String?, _ value: Binding<String>) -> some View {
        Row(label: label, sublabel: help) { TextField(label, text: value).labelsHidden().frame(maxWidth: 210).accessibilityLabel(label) }
    }
    private func path(_ label: String, _ help: String, _ value: Binding<String>, _ directory: Bool) -> some View {
        Row(label: label, sublabel: help) {
            TextField(label,text: value).labelsHidden().frame(maxWidth: 210)
            Button("选择…") { if let url = Self.choose(directory: directory) { value.wrappedValue = url.path } }.accessibilityLabel("选择" + label)
        }
    }
    static func choose(directory: Bool) -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = directory; panel.canChooseFiles = !directory
        return panel.runModal() == .OK ? panel.url : nil
    }
    private func modify(_ body: () throws -> Void) {
        do { try body(); note = ""; failed = false } catch { note = error.localizedDescription; failed = true }
    }
    private func reset() {
        switch section {
        case .server: startup.allowLAN = false; startup.allowIPv6 = false; port = "8080"; context = ""; generations = [:]; prefs.autoStart = false; prefs.recordUsage = true; prefs.logLevel = nil
        case .network: prefs.httpProxy = ""; prefs.httpsProxy = ""; prefs.noProxy = ""; prefs.caBundle = ""
        case .performance: memory = ""; prefix = ""; startup.maxConcurrency = nil; prefs.prefixEnabled = nil
        default: break
        }
        note = "点击应用以保存默认值。"; failed = false
    }
    private func apply() async {
        saving = true; defer { saving = false }
        do {
            var fresh = MIVManagement.settings
            var settings = service.settings
            if section == .network {
                for proxy in [prefs.httpProxy,prefs.httpsProxy] where !proxy.isEmpty {
                    guard let u = URLComponents(string: proxy), let scheme = u.scheme,
                          ["http","https","socks5","socks5h"].contains(scheme), u.host != nil else { throw CoreFailure("代理地址格式无效。") }
                }
                if !prefs.caBundle.isEmpty && !FileManager.default.isReadableFile(atPath: prefs.caBundle) { throw CoreFailure("无法读取 CA 文件。") }
                fresh.httpProxy = prefs.httpProxy; fresh.httpsProxy = prefs.httpsProxy
                fresh.noProxy = prefs.noProxy; fresh.caBundle = prefs.caBundle
            } else if section == .server {
                if startup.mode == .managed {
                    guard let n = Int(port), (1...65535).contains(n) else { throw CoreFailure("端口须为 1–65535。") }
                    startup.port = n
                    if !context.isEmpty, Int(context) == nil { throw CoreFailure("上下文窗口须为正整数。") }
                    startup.maxContext = Int(context)
                    _ = try startup.launchArguments()
                }
                settings = startup; _ = try settings.baseURL()
                fresh.autoStart = prefs.autoStart; fresh.recordUsage = prefs.recordUsage; fresh.logLevel = prefs.logLevel
            } else if section == .performance, settings.mode == .managed {
                func budget(_ text: String) throws -> Double? {
                    if text.isEmpty { return nil }
                    guard let n = Double(text), n.isFinite, n >= 0 else { throw CoreFailure("预算须为非负数。") }; return n
                }
                fresh.memoryBudget = try budget(memory); fresh.prefixBudget = try budget(prefix)
                fresh.prefixEnabled = prefs.prefixEnabled; settings.maxConcurrency = startup.maxConcurrency
            }
            let previous = service.settings, oldPrefs = MIVManagement.settings, oldKey = service.apiKey
            let oldGeneration = service.globalGeneration
            let generation = generations.filter { !$0.value.isEmpty }
            let changeGeneration = section == .server && settings.mode == .managed && generation != oldGeneration
            if changeGeneration { try service.setGlobalGeneration(generation) }
            let restart = service.isRunning && (settings != previous || changeGeneration || (section == .performance && fresh != oldPrefs))
            do {
                if restart {
                    service.stop()
                    while service.isStopping { try await Task.sleep(for: .milliseconds(100)) }
                }
                if section == .server && fresh.logLevel != oldPrefs.logLevel { try MIVManagement.applyLogLevel(fresh.logLevel) }
                MIVManagement.settings = fresh
                if section == .server || section == .performance {
                    try service.apply(settings, apiKey: section == .server ? clientKey : oldKey)
                }
                if restart && settings.mode == .managed { try service.start() }
            } catch {
                if fresh.logLevel != oldPrefs.logLevel { try? MIVManagement.applyLogLevel(oldPrefs.logLevel) }
                MIVManagement.settings = oldPrefs
                if changeGeneration { try? service.setGlobalGeneration(oldGeneration) }
                if restart { try? service.apply(previous, apiKey: oldKey); if !service.isRunning { try? service.start() } }
                throw error
            }
            await service.refresh(); note = "已应用。"; failed = false
        } catch { note = error.localizedDescription; failed = true }
    }
}
