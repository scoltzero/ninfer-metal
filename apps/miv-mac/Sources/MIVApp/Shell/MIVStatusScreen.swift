import SwiftUI

extension CoreService {
    func restartManaged() async throws {
        guard settings.mode == .managed, isRunning, !isStopping else { return }
        let previous = settings
        stop()
        let deadline = Date().addingTimeInterval(10)
        while isRunning, Date() < deadline { try await Task.sleep(for: .milliseconds(100)) }
        guard !isRunning, settings == previous else { throw CoreFailure("核心仍在退出或设置已更改，请从服务器页启动。") }
        try start()
    }
}

struct MIVServerHero: View {
    @ObservedObject var service: CoreService
    @EnvironmentObject private var navigation: NavigationState
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 12) {
            Image(nsImage: NSApp.applicationIconImage).resizable().interpolation(.high).frame(width: 53, height: 53)
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("MIV").font(.omlxText(DesignTokens.FontSize.title2, weight: .semibold))
                    StatusPill(status: .custom(color: service.health?.status == "ok" ? theme.greenDot : theme.textTertiary,
                                               label: service.statusText, fillBg: true))
                }
                Text(service.serviceMessage ?? service.healthError ?? service.browserURL?.absoluteString ?? "—")
                    .font(.omlxText(DesignTokens.FontSize.callout)).foregroundStyle(theme.textSecondary)
                    .lineLimit(2).textSelection(.enabled)
            }
            Spacer(minLength: 8)
            if service.settings.mode == .managed {
                if service.isRunning {
                    Button("重启", systemImage: "arrow.clockwise") {
                        Task {
                            do { try await service.restartManaged() }
                            catch { navigation.notice = error.localizedDescription }
                        }
                    }.buttonStyle(.omlx(.normal)).disabled(service.isStopping)
                    Button("停止", systemImage: "stop.fill") { service.stop() }
                        .buttonStyle(.omlx(.destructive)).disabled(service.isStopping)
                } else {
                    Button("启动服务器", systemImage: "play.fill") {
                        do { try service.start() }
                        catch { navigation.section = .server; navigation.notice = error.localizedDescription }
                    }.buttonStyle(.omlx(.primary))
                }
            } else {
                Button("刷新", systemImage: "arrow.clockwise") { Task { await service.refresh() } }
                    .buttonStyle(.omlx(.normal)).disabled(service.isRefreshing)
            }
        }
        .padding(14)
        .background(theme.groupBg)
        .clipShape(RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: theme.cornerRadius).strokeBorder(theme.groupBorder, lineWidth: 0.5))
        .padding(.horizontal, 14).padding(.bottom, 14)
    }
}

struct MIVStatusScreen: View {
    @ObservedObject var service: CoreService
    @EnvironmentObject private var activity: DashboardActivity
    @EnvironmentObject private var navigation: NavigationState
    @Environment(\.omlxTheme) private var theme
    @AppStorage(NumberScale.storageKey) private var numberScale = NumberScale.compact.rawValue
    @State private var scope = "session"
    @State private var confirmClear = false

    var body: some View {
        let total = scope == "session" ? activity.served : activity.lifetime.usage
        let rates = scope == "session" ? activity.session.rates : activity.lifetime.rates
        VStack(alignment: .leading, spacing: 0) {
            MIVServerHero(service: service)
            SectionHeader("服务统计") {
                HStack(spacing: 10) {
                    Segmented(selection: $scope, options: [("session", "本次会话"), ("alltime", "全部时间")])
                    Button { confirmClear = true } label: { Image(systemName: "trash") }
                        .buttonStyle(.omlx(.plain, size: .small)).help("清除该范围的本地统计")
                }
            }
            HStack(spacing: 10) {
                StatTile(label: "预填充 Tokens", value: formatted(total.prompt))
                StatTile(label: "缓存 Tokens", value: formatted(total.cached))
                StatTile(label: "缓存效率", value: formattedEfficiency(total))
            }.padding(.horizontal, 14)
            SectionHeader("平均速度", subtitle: "按本机观测到的采样时间计算，含空闲时间")
            HStack(spacing: 10) {
                StatTile(label: "提示处理", value: rates?.promptTps.map(displayRate) ?? "—")
                StatTile(label: "Token 生成", value: rates?.generationTps.map(displayRate) ?? "—")
            }.padding(.horizontal, 14)
            SectionHeader("当前活动")
            ListGroup {
                if let models = service.health?.models, !models.isEmpty {
                    ForEach(models) { model in
                        Row(label: model.id, sublabel: model.backend ?? model.backendMode, isLast: model.id == models.last?.id) {
                            Text("\(model.activeRequests ?? 0) / \(model.maxConcurrency ?? 0) 请求").monospacedDigit()
                        }
                    }
                } else { Row(label: "暂无活动模型", isLast: true) }
            }
            UsageHistory(buckets: activity.savedHours, recording: activity.recording)
            SectionHeader("系统", subtitle: "运行 MIV App 的 Mac")
            ListGroup {
                Row(label: "系统内存") {
                    Text(SystemStatsSampler.formatBytes(activity.system.memory.usedBytes) + " / " + SystemStatsSampler.formatBytes(activity.system.memory.totalBytes))
                        .monospacedDigit()
                }
                Row(label: "CPU 利用率") { Text(percent(activity.system.cpuTotalUsage)) }
                Row(label: "GPU 利用率") { Text(percent(activity.system.gpuUsage)) }
                Row(label: "GPU 内存") { Text(activity.system.gpuMemoryInUseBytes.map(SystemStatsSampler.formatBytes) ?? "—") }
                Row(label: "系统运行时间", isLast: true) { Text(SystemStatsSampler.formatUptime(activity.system.uptimeSeconds)) }
            }
            SectionHeader("API 端点")
            ListGroup {
                Row(label: "OpenAI 兼容 API") {
                    if let url = service.browserURL?.appendingPathComponent("v1") { CodeChip(value: url.absoluteString) }
                }
                Row(label: "核心版本") { Text(service.health?.version ?? "—") }
                if service.dashboardAvailable, let url = service.browserURL {
                    LinkRow(label: "网页控制台", sublabel: url.absoluteString, icon: "safari", url: url, isLast: true)
                } else {
                    Row(label: "网页控制台", sublabel: service.health == nil ? "等待连接核心" : "该核心未提供网页控制台。", isLast: true)
                }
            }
        }.alert("清除本地统计？", isPresented: $confirmClear) {
            Button("取消", role: .cancel) {}
            Button("清除", role: .destructive) { activity.clear(allTime: scope == "alltime") }
        } message: { Text("将在此 App 中重新累计所选端点和时间范围的数据。") }
    }

    private func formattedEfficiency(_ usage: DashboardActivity.Usage) -> String {
        if scope == "session", service.health == nil { return "—" }
        let ratio = usage.prompt > 0 ? usage.cached / usage.prompt : 0
        return ratio.formatted(.percent.precision(.fractionLength(1)))
    }

    private func percent(_ fraction: Double?) -> String {
        fraction?.formatted(.percent.precision(.fractionLength(0))) ?? "—"
    }

    private func formatted(_ value: Double) -> String {
        guard value.isFinite, let number = Int(exactly: value.rounded()) else { return "—" }
        if scope == "session", service.health == nil { return "—" }
        return numberScale == NumberScale.exact.rawValue ? number.formatted() : CountFormat.compact(number)
    }
}

private struct StatTile: View {
    enum AccentRole { case neutral, success, warning, danger }

    let label: String
    let value: String
    var sub: String?
    var accentRole: AccentRole = .neutral

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.omlxText(DesignTokens.FontSize.subheadline, weight: .medium))
                .foregroundStyle(theme.textSecondary)
                .textCase(.uppercase)
                .kerning(0.6)
                .multilineTextAlignment(.center)
                .frame(maxWidth: .infinity)
            Text(value)
                .font(.omlxText(DesignTokens.FontSize.title1, weight: .semibold))
                .kerning(-0.5)
                .foregroundStyle(accentColor)
                .frame(maxWidth: .infinity)
            if let sub {
                Text(sub)
                    .font(.omlxText(DesignTokens.FontSize.subheadline))
                    .foregroundStyle(theme.textTertiary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, minHeight: 96, alignment: .center)
        .background(theme.groupBg)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private var accentColor: Color {
        switch accentRole {
        case .neutral: return theme.text
        case .success: return theme.greenDot
        case .warning: return theme.amberDot
        case .danger:  return theme.redDot
        }
    }
}

