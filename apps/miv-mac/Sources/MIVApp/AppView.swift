import AppKit
import SwiftUI

// Adapted from oMLX's AppView and AppSection. Keep the split navigation and
// section routing, with a flexible detail column for standard macOS resizing.
enum AppSection: String, CaseIterable, Identifiable {
    case status, models, performance, service, logs
    var id: Self { self }
    var title: String {
        switch self {
        case .status: "运行状态"
        case .models: "模型"
        case .performance: "性能"
        case .service: "服务设置"
        case .logs: "服务日志"
        }
    }
    var symbol: String {
        switch self {
        case .status: "gauge.with.dots.needle.50percent"
        case .models: "cube.transparent"
        case .performance: "waveform.path.ecg"
        case .service: "server.rack"
        case .logs: "text.alignleft"
        }
    }
}

@MainActor final class NavigationState: ObservableObject {
    @Published var section: AppSection = .status
    @Published var notice: String?
}

extension CoreService {
    var chatURL: URL? {
        guard let url = browserURL, var parts = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return nil }
        parts.fragment = "chat"
        return parts.url
    }

    var statusText: String {
        if isStopping { return "正在停止" }
        if let health {
            switch health.status {
            case "ok": return "运行中"
            case "degraded": return "部分可用"
            case "unhealthy": return "异常"
            case "shutting_down": return "正在停止"
            default: return health.status
            }
        }
        if settings.mode == .managed { return isRunning ? "等待核心就绪" : "已停止" }
        return healthError == nil ? "正在连接" : "无法连接"
    }
}

func displayRate(_ value: Double) -> String {
    value.formatted(.number.precision(.fractionLength(1))) + " tok/s"
}

private func count(_ number: Int?) -> String { number?.formatted() ?? "—" }
private func bytes(_ number: Int?) -> String {
    guard let number else { return "—" }
    let gibibytes = Double(number) / (1024 * 1024 * 1024)
    return gibibytes.formatted(.number.precision(.fractionLength(2))) + " GiB"
}

struct AppView: View {
    @ObservedObject var service: CoreService
    @ObservedObject var navigation: NavigationState

    var body: some View {
        NavigationSplitView {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 9) {
                    Image(nsImage: NSApp.applicationIconImage).resizable().frame(width: 38, height: 38)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("MIV").font(.title3.bold())
                        Text("ninfer-metal").font(.caption).foregroundStyle(.secondary)
                    }
                }.padding(.horizontal, 16).padding(.top, 18)
                List(selection: $navigation.section) {
                    ForEach(AppSection.allCases.filter { $0 != .logs || service.settings.mode == .managed }) { page in
                        Label(page.title, systemImage: page.symbol).tag(page)
                    }
                }.listStyle(.sidebar)
                HStack {
                    Circle().fill(statusColor).frame(width: 7, height: 7)
                    Text(service.statusText).font(.callout)
                }.padding([.leading, .bottom], 18)
            }.navigationSplitViewColumnWidth(min: 155, ideal: 185, max: 245)
        } detail: {
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Text(navigation.section.title).font(.largeTitle.weight(.semibold))
                        .accessibilityAddTraits(.isHeader)
                    Spacer()
                    if service.dashboardAvailable, let url = service.browserURL {
                        Link(destination: url) { Label("打开网页", systemImage: "safari") }
                    }
                    Button {
                        Task { await service.refresh() }
                    } label: { Image(systemName: "arrow.clockwise") }
                    .disabled(service.isRefreshing || (service.settings.mode == .managed && !service.isRunning))
                    .help("刷新核心状态").accessibilityLabel("刷新核心状态")
                }.padding(.horizontal, 24).padding(.top, 20).padding(.bottom, 14)
                Divider()
                switch navigation.section {
                case .service: ServiceScreen(service: service)
                case .logs: logs
                case .status, .performance, .models:
                    Form {
                        if navigation.section == .status { overview }
                        if navigation.section == .models { models }
                        if navigation.section == .performance { performance }
                    }.formStyle(.grouped)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .navigationSplitViewStyle(.balanced)
        .alert("服务操作失败", isPresented: Binding(get: { navigation.notice != nil }, set: { if !$0 { navigation.notice = nil } })) {
            Button("好") { navigation.notice = nil }
        } message: { Text(navigation.notice ?? "") }
    }

    private var statusColor: Color {
        if service.health?.status == "ok" { return .green }
        if service.health != nil || service.isRunning || service.isStopping { return .orange }
        return .secondary
    }

    @ViewBuilder private var overview: some View {
        Section {
            LabeledContent("核心状态") { Label(service.statusText, systemImage: "circle.fill").foregroundStyle(statusColor) }
            LabeledContent("连接方式", value: service.settings.mode == .managed ? "本机托管" : "现有服务")
            if let url = service.browserURL {
                LabeledContent("API 基址") {
                    Text(url.appendingPathComponent("v1").absoluteString).textSelection(.enabled)
                }
            }
            if let health = service.health {
                LabeledContent("核心版本", value: health.version)
                if let backend = health.backend { LabeledContent("后端", value: backend) }
                LabeledContent("活动请求 / 并发上限", value: count(health.activeRequests) + " / " + count(health.maxConcurrency))
                if let budget = health.memoryBudgetBytes, budget > 0 { LabeledContent("内存预算", value: bytes(budget)) }
            } else {
                if let message = service.serviceMessage ?? service.healthError {
                    Text(message).foregroundStyle(.secondary).textSelection(.enabled)
                }
                Button("配置服务…") { navigation.section = .service }
            }
        } header: { Text("服务") }
        if let health = service.health {
            Section {
                if let primary = health.model { LabeledContent("首选模型", value: primary) }
                Button("查看模型") { navigation.section = .models }
                if !service.dashboardAvailable {
                    Text("该核心未提供网页控制台。").font(.callout).foregroundStyle(.secondary)
                }
            } header: { Text("模型服务") }
        }
    }

    @ViewBuilder private var models: some View {
        if let error = service.modelsError {
            Section {
                Text(error).textSelection(.enabled)
                Button("打开服务设置") { navigation.section = .service }
            } header: { Text("无法读取模型列表") }
        } else if service.models.isEmpty {
            Section {
                Text(service.health == nil ? "连接核心后即可查看模型。" : "核心尚未列出模型。")
                Button("打开服务设置") { navigation.section = .service }
            }
        } else {
            ForEach(service.models) { model in
                Section {
                    if let context = model.maxModelLen ?? model.maxContext { LabeledContent("上下文窗口", value: count(context) + " tokens") }
                    if let kv = model.kvDtype { LabeledContent("KV 缓存格式", value: kv) }
                    if let spec = model.specBackend { LabeledContent("推测解码", value: spec) }
                    if let provider = model.providerId { LabeledContent("提供方", value: provider) }
                    if let backend = model.backendMode { LabeledContent("运行模式", value: backend) }
                    if let details = service.health?.models?.first(where: { $0.id == model.id }) {
                        if let status = details.status { LabeledContent("状态", value: status == "ok" ? "运行中" : status) }
                        if model.kvDtype == nil, let kv = details.kvDtype { LabeledContent("KV 缓存格式", value: kv) }
                        if model.specBackend == nil, let spec = details.specBackend { LabeledContent("推测解码", value: spec) }
                        if let resident = details.residentMemoryBytes { LabeledContent("驻留内存", value: bytes(resident)) }
                        if let capacity = details.maxConcurrency {
                            LabeledContent("活动请求 / 并发上限", value: count(details.activeRequests) + " / " + count(capacity))
                        }
                    }
                    if let lora = model.loraAdapter, !lora.isEmpty {
                        LabeledContent("LoRA 适配器", value: lora)
                        if let mode = model.loraMode { LabeledContent("LoRA 模式", value: mode) }
                    }
                } header: {
                    Text(model.id).font(.headline).textSelection(.enabled)
                }
            }
            if service.dashboardAvailable, let url = service.chatURL {
                Section { Link("在网页中试用模型", destination: url) }
            }
        }
    }

    @ViewBuilder private var performance: some View {
        if service.metrics.isEmpty {
            Section {
                Text(service.metricsError ?? "连接核心后即可查看性能指标。").textSelection(.enabled)
                Button("打开服务设置") { navigation.section = .service }
            }
        } else {
            Section {
                LabeledContent("输出吞吐（采样区间）", value: service.outputRate.map(displayRate) ?? "等待下一次采样")
                metric("正在处理", "active_requests")
                metric("排队请求", "queued_requests")
                if let footprint = service.metrics["ninfer_metal_phys_footprint_bytes"] {
                    LabeledContent("核心进程内存", value: bytes(Int(exactly: footprint)))
                }
            } header: { Text("实时负载") }
            Section {
                metric("总请求", "requests_total")
                metric("成功", "requests_succeeded_total")
                metric("失败", "requests_failed_total")
                metric("取消", "requests_cancelled_total")
                metric("输入 tokens", "prompt_tokens_total")
                metric("输出 tokens", "generated_tokens_total")
            } header: { Text("本次核心运行") }
            Section {
                metric("前缀完整命中", "prefix_cache_hits_total")
                metric("前缀部分命中", "prefix_cache_partial_hits_total")
                metric("前缀未命中", "prefix_cache_misses_total")
                metric("推测轮数", "spec_rounds_total")
                metric("提议 tokens", "spec_proposal_tokens_total")
                metric("接受 tokens", "spec_accepted_tokens_total")
            } header: { Text("缓存与推测解码") }
        }
    }

    @ViewBuilder private func metric(_ title: String, _ key: String) -> some View {
        if let value = service.metrics["ninfer_metal_" + key] {
            LabeledContent(title, value: value.formatted(.number.precision(.fractionLength(0))))
                .monospacedDigit()
        }
    }

    private var logs: some View {
        ScrollView([.vertical, .horizontal]) {
            Text(service.logText.isEmpty ? "本机核心启动后的输出会显示在这里。" : service.logText)
                .font(.system(.callout, design: .monospaced)).textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading).padding(18)
        }.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
