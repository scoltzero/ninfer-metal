import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct ServiceScreen: View {
    @ObservedObject var service: CoreService
    @State private var draft: CoreSettings
    @State private var port: String
    @State private var context: String
    @State private var key: String
    @State private var notice: String?
    @State private var failed = false

    init(service: CoreService) {
        self.service = service
        _draft = State(initialValue: service.settings)
        _port = State(initialValue: String(service.settings.port))
        _context = State(initialValue: service.settings.maxContext.map(String.init) ?? "")
        _key = State(initialValue: service.apiKey)
    }

    var body: some View {
        Form {
            Section {
                Picker("运行方式", selection: $draft.mode) {
                    Text("本机托管").tag(CoreMode.managed)
                    Text("连接现有服务").tag(CoreMode.external)
                }.pickerStyle(.segmented)
                if draft.mode == .external {
                    TextField("服务地址", text: $draft.endpoint, prompt: Text("http://127.0.0.1:8080"))
                        .autocorrectionDisabled().accessibilityLabel("服务地址")
                } else {
                    TextField("监听端口", text: $port).monospacedDigit().accessibilityLabel("监听端口")
                    Toggle("允许局域网连接", isOn: $draft.allowLAN).accessibilityLabel("允许局域网连接")
                    if draft.allowLAN {
                        Text("同一网络的设备可通过本机局域网地址访问服务和网页。推理 API 的认证密钥由 TOML 的 [server].api_key 设置。")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                    pathField("模型目录", value: $draft.modelDirectory, directory: true, prompt: "使用 TOML 中指定的模型")
                    pathField("共享配置", value: $draft.configPath, directory: false, prompt: "自动查找 ninfer-metal.toml")
                    if let url = configURL {
                        Button("在 Finder 中显示配置文件") {
                            NSWorkspace.shared.activateFileViewerSelecting([url])
                        }
                        .help(url.path)
                    }
                }
            }.disabled(service.isRunning || service.isStopping)
            if draft.mode == .managed {
                Section {
                    TextField("上下文长度", text: $context, prompt: Text("继承配置（tokens）")).accessibilityLabel("上下文长度")
                    Picker("最大并发数", selection: $draft.maxConcurrency) {
                        Text("继承配置").tag(Int?.none)
                        ForEach(1...8, id: \.self) { Text(String($0)).tag(Optional($0)) }
                    }.accessibilityLabel("最大并发数")
                    Picker("KV 缓存格式", selection: $draft.kvDtype) {
                        Text("继承配置").tag(String?.none)
                        Text("BF16").tag(Optional("bf16"))
                        Text("Affine 4-bit · G128").tag(Optional("affine4_g128"))
                    }.accessibilityLabel("KV 缓存格式")
                    Picker("推测解码", selection: $draft.speculativeBackend) {
                        Text("继承配置").tag(String?.none)
                        Text("自动检测").tag(Optional("auto"))
                        Text("关闭").tag(Optional("none"))
                        Text("MTP").tag(Optional("mtp"))
                        Text("DFlash2").tag(Optional("dflash2"))
                    }.accessibilityLabel("推测解码")
                } header: { Text("推理设置") } footer: {
                    Text("设置在下次启动时生效。DFlash2 草稿模型路径和推测深度可在共享 TOML 中配置。")
                }.disabled(service.isRunning || service.isStopping)
            }
            Section {
                SecureField("API 密钥", text: $key, prompt: Text("需要认证时填写")).accessibilityLabel("API 密钥")
            } footer: {
                Text("App 用此密钥访问受保护的推理 API，仅在本次会话中使用。")
            }
            Section {
                HStack {
                    if service.isRunning && service.settings.mode == .managed {
                        Button(service.isStopping ? "正在停止…" : "停止服务") { service.stop() }
                            .disabled(service.isStopping)
                        Button("应用连接密钥") { apply(start: false) }.disabled(service.isStopping)
                    } else if draft.mode == .managed {
                        Button("启动服务") { apply(start: true) }.buttonStyle(.borderedProminent)
                        Button("保存设置") { apply(start: false) }
                    } else {
                        Button("应用并连接") { apply(start: false) }.buttonStyle(.borderedProminent)
                    }
                }
                if let notice {
                    Text(notice).foregroundStyle(failed ? Color.red : .secondary).textSelection(.enabled)
                }
                if let message = service.serviceMessage {
                    Text(message).foregroundStyle(.secondary).textSelection(.enabled)
                }
                if let message = service.modelsError ?? service.healthError, draft.mode == .external || service.isRunning {
                    Text(message).foregroundStyle(.secondary).textSelection(.enabled)
                }
            }
        }
        .formStyle(.grouped)
        .onChange(of: draft.endpoint) { _, _ in key = ""; notice = nil }
        .onChange(of: draft.mode) { _, _ in key = ""; notice = nil }
        .onChange(of: port) { _, _ in key = ""; notice = nil }
    }

    private var configURL: URL? {
        if !draft.configPath.isEmpty {
            let path = (draft.configPath as NSString).expandingTildeInPath
            return FileManager.default.fileExists(atPath: path) ? URL(fileURLWithPath: path) : nil
        }
        guard service.settings.configPath.isEmpty else { return nil }
        return service.effectiveConfigURL
    }

    private func pathField(_ label: String, value: Binding<String>, directory: Bool, prompt: String) -> some View {
        LabeledContent(label) {
            HStack {
                TextField(label, text: value, prompt: Text(prompt)).labelsHidden()
                    .autocorrectionDisabled().help(value.wrappedValue)
                Button("选择…") {
                    let panel = NSOpenPanel()
                    panel.title = label
                    panel.canChooseDirectories = directory
                    panel.canChooseFiles = !directory
                    panel.allowsMultipleSelection = false
                    if !directory, let toml = UTType(filenameExtension: "toml") { panel.allowedContentTypes = [toml] }
                    if panel.runModal() == .OK, let url = panel.url { value.wrappedValue = url.path }
                }.accessibilityLabel("选择" + label)
            }
        }
    }

    private func apply(start: Bool) {
        do {
            if draft.mode == .managed {
                guard let number = Int(port), (1...65535).contains(number) else {
                    throw InputError("监听端口必须是 1 至 65535 的整数。")
                }
                draft.port = number
                let text = context.trimmingCharacters(in: .whitespacesAndNewlines)
                if text.isEmpty {
                    draft.maxContext = nil
                } else {
                    guard let number = Int(text), number > 0, number <= Int(Int32.max) else {
                        throw InputError("上下文长度必须是正整数，留空可继承配置。")
                    }
                    draft.maxContext = number
                }
            }
            try service.apply(draft, apiKey: key)
            if start { try service.start() }
            failed = false
            notice = start ? nil : "设置已应用。"
            Task { await service.refresh() }
        } catch {
            failed = true
            notice = error.localizedDescription
        }
    }

    private struct InputError: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }
}
