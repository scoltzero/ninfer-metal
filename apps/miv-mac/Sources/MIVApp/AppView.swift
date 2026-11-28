import AppKit
import SwiftUI

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

// MIV uses the donor's five sidebar groups, SF symbols, toolbar title and
// 720-pt content column. Only the outer fixed width was dropped for resizing.
struct AppView: View {
    @ObservedObject var service: CoreService
    @ObservedObject var navigation: NavigationState
    @Environment(\.colorScheme) private var scheme
    @AppStorage(EnhancedReadability.enabledKey) private var enhancedReadability = false
    @State private var settingsQuery = ""
    @State private var highlightedSettingID: String?
    @FocusState private var searchFocused: Bool

    var body: some View {
        let theme = OMLXTheme.resolved(for: scheme, enhancedReadability: enhancedReadability)
        NavigationSplitView(columnVisibility: .constant(.all)) {
            Group {
                if settingsQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    SettingsSidebar(selection: Binding(get: { navigation.section }, set: { if let page = $0 { navigation.section = page } }))
                } else {
                    SettingsSearchResults(results: SettingsSearch.results(for: settingsQuery), selection: $highlightedSettingID, activate: openSetting)
                }
            }
            .toolbar(removing: .sidebarToggle)
            .navigationSplitViewColumnWidth(min: 180, ideal: 195, max: 215)
        } detail: {
            Group {
                if navigation.section.fillsContentArea {
                    page.frame(maxWidth: 720, maxHeight: .infinity, alignment: .topLeading)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                        .padding(.bottom, 18)
                } else {
                    ScrollViewReader { proxy in
                        ScrollView {
                            VStack(spacing: 0) {
                                Color.clear.frame(height: 0).id("detailTop")
                                page.frame(maxWidth: 720, alignment: .topLeading)
                                    .frame(maxWidth: .infinity, alignment: .top)
                                    .padding(.bottom, 36)
                            }
                        }
                        .onReceive(NotificationCenter.default.publisher(for: .init("MIVScrollToTop"))) { _ in
                            Task { @MainActor in
                                await Task.yield()
                                proxy.scrollTo("detailTop", anchor: .top)
                            }
                        }
                    }.id(navigation.section)
                }
            }
            .navigationTitle(navigation.section.title)
            .modifier(ToolbarHeightKeeper())
            .background(theme.windowBg)
        }
        .searchable(text: $settingsQuery, placement: .sidebar, prompt: "搜索设置")
        .searchFocused($searchFocused)
        .onSubmit(of: .search) { selectHighlightedSetting() }
        .onChange(of: settingsQuery) { _, _ in highlightedSettingID = nil }
        .navigationSplitViewStyle(.balanced)
        .background(theme.windowBg)
        .environment(\.omlxTheme, theme)
        .environmentObject(navigation)
        .alert("服务操作失败", isPresented: Binding(get: { navigation.notice != nil }, set: { if !$0 { navigation.notice = nil } })) {
            Button("好") { navigation.notice = nil }
        } message: { Text(navigation.notice ?? "") }
    }

    private func selectHighlightedSetting() {
        let results = SettingsSearch.results(for: settingsQuery)
        if let entry = results.first(where: { $0.id == highlightedSettingID }) ?? results.first { openSetting(entry) }
    }

    private func openSetting(_ entry: SettingsSearchEntry) {
        navigation.section = entry.section
        settingsQuery = ""
        searchFocused = false
    }

    @ViewBuilder private var page: some View {
        switch navigation.section {
        case .status: MIVStatusScreen(service: service)
        case .appearance: AppearanceScreen()
        case .about: MIVAboutScreen(service: service)
        case .logs: LogsScreen(service: service)
        case .models: MIVModelsScreen(service: service)
        case .downloads: MIVDownloadsScreen(service: service)
        case .quantization: MIVQuantizationScreen(service: service)
        case .integrations: MIVIntegrationsScreen(service: service)
        case .throughputBench, .accuracyBench, .contextBench: BenchmarksPage(section: navigation.section, service: service)
        case .server, .network, .performance, .security: ServerManagementPage(section: navigation.section, service: service)
        }
    }
}


private struct SettingsSidebar: View {
    @Binding var selection: AppSection?

    var body: some View {
        List(selection: $selection) {
            Section {
                SidebarRow(section: .status)
                SidebarRow(section: .server)
                SidebarRow(section: .network)
                SidebarRow(section: .performance)
                SidebarRow(section: .logs)
            } header: {
                Text(String(localized: "sidebar.group.server",
                            defaultValue: "Server",
                            comment: "Sidebar group heading for server-related screens"))
            }
            Section {
                SidebarRow(section: .appearance)
            } header: {
                Text(String(localized: "sidebar.group.app",
                            defaultValue: "App",
                            comment: "Sidebar group heading for settings stored by the Mac app itself"))
            }
            Section {
                SidebarRow(section: .models)
                SidebarRow(section: .downloads)
                SidebarRow(section: .integrations)
                SidebarRow(section: .quantization)
            } header: {
                Text(String(localized: "sidebar.group.models",
                            defaultValue: "Models",
                            comment: "Sidebar group heading for models/downloads/quant screens"))
            }
            Section {
                SidebarRow(section: .throughputBench)
                SidebarRow(section: .accuracyBench)
                SidebarRow(section: .contextBench)
            } header: {
                Text(String(localized: "sidebar.group.benchmark",
                            defaultValue: "Benchmark",
                            comment: "Sidebar group heading for accuracy + throughput bench screens"))
            }
            Section {
                SidebarRow(section: .security)
                SidebarRow(section: .about)
            } header: {
                Text(String(localized: "sidebar.group.general",
                            defaultValue: "General",
                            comment: "Sidebar group heading for the about/integrations/logs screens"))
            }
        }
        .listStyle(.sidebar)
    }
}

private struct SettingsSearchResults: View {
    let results: [SettingsSearchEntry]
    @Binding var selection: String?
    let activate: (SettingsSearchEntry) -> Void

    var body: some View {
        List(selection: $selection) {
            ForEach(results) { entry in
                Button {
                    activate(entry)
                } label: {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(entry.title())
                            .font(.body)
                        Text(entry.breadcrumb())
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 3)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .tag(entry.id)
                .help("\(entry.title()) — \(entry.breadcrumb())")
                .accessibilityLabel("\(entry.title()), \(entry.breadcrumb())")
                .accessibilityIdentifier("settings-search-result-\(entry.id)")
            }
        }
        .listStyle(.sidebar)
        .overlay {
            if results.isEmpty {
                Text(String(localized: "settings.search.empty", defaultValue: "No settings found",
                            comment: "Empty state for global settings search results"))
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding()
            }
        }
    }
}

private struct SidebarRow: View {
    let section: AppSection

    var body: some View {
        NavigationLink(value: section) {
            Label(section.title, systemImage: section.symbol)
        }
    }
}

private struct ToolbarHeightKeeper: ViewModifier {
    func body(content: Content) -> some View {
        if #available(macOS 26.0, *) {
            content.toolbar {
                ToolbarItem {
                    Color.clear.frame(width: 1, height: 1)
                }
                .sharedBackgroundVisibility(.hidden)
            }
        } else {
            content.toolbar {
                ToolbarItem {
                    Color.clear.frame(width: 1, height: 1)
                }
            }
        }
    }
}
