import AppKit
import SwiftUI

struct IntegrationTarget: Encodable {
    var tool: String
    var model: String
    var address: String
    var apiKey: String
    var contextWindow: Int?
    var toolsProfile: String?
    var catalog: [String]
    var opusModel: String?
    var sonnetModel: String?
    var haikuModel: String?
}
struct IntegrationProfile: Decodable { var profile: String }
struct ClientAvailability: Decodable { let installed: Bool; let hint: String }

struct MIVIntegrationsScreen: View {
    @ObservedObject var service: CoreService
    @Environment(\.omlxTheme) private var theme
    @State private var selected: IntegrationTool?
    @AppStorage("miv.integration.models") private var savedModels = Data()
    @AppStorage("miv.integration.tools") private var toolsProfile = "coding"
    @AppStorage("miv.integration.claude.mode") private var claudeMode = "local"
    @AppStorage("miv.integration.claude.desktop") private var desktopEnabled = false
    @State private var availability: [String: ClientAvailability] = [:]
    @State private var commands: [String: Command] = [:]
    @State private var result = ""

    private struct Command { var text: String; var profile: String }

    private var models: [(String,String)] { service.models.map { ($0.id,$0.id) } }
    private var saved: [String:String] { (try? JSONDecoder().decode([String:String].self, from: savedModels)) ?? [:] }
    private var client: ModelToolsClient { ModelToolsClient(root: service.libraryRoot, roots: service.libraryRoots) }
    private func selection(_ key: String) -> Binding<String> {
        Binding(get: {
            if let preferred = saved[key], service.models.contains(where: { $0.id == preferred }) { return preferred }
            return service.models.first?.id ?? ""
        }, set: { id in
            var prefs = saved
            prefs[key] = id
            if let data = try? JSONEncoder().encode(prefs) { savedModels = data }
        })
    }
    private var variants: [String] {
        guard let selected else { return [] }
        if selected == .codex { return ["codex","codex_app"] }
        if selected == .claude && desktopEnabled && claudeMode == "local" { return ["claude","claude_desktop"] }
        return [selected.rawValue]
    }
    private var requestIdentity: [String] {
        [selected?.rawValue ?? "", toolsProfile, claudeMode, String(desktopEnabled),
         service.browserURL?.absoluteString ?? "", service.apiKey, savedModels.base64EncodedString()] + service.models.map(\.id)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader("应用程序", subtitle: "选择连接 MIV 的应用，配置默认模型和启动命令。")
            ListGroup {
                FreeRow(verticalPadding: 12) {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 108, maximum: 140), spacing: 8)],
                              alignment: .leading, spacing: 8) {
                        ForEach(IntegrationTool.allCases) { tool in
                            ApplicationTile(tool: tool, isSelected: selected == tool) {
                                selected = selected == tool ? nil : tool
                            }
                        }
                    }
                }
                if let selected { panel(selected) }
                else {
                    FreeRow(isLast: true, verticalPadding: 8) {
                        Text("选择应用以查看启动命令。").font(.omlxText(11.5)).foregroundStyle(theme.textTertiary)
                    }
                }
            }
            FooterBar(error: result.isEmpty ? nil : result)
        }
        .task(id: service.libraryRoot) {
            do { availability = try await client.get("/integrations/availability") }
            catch { if !Task.isCancelled { result = error.localizedDescription } }
        }
        .task(id: requestIdentity) { await prepareCommands() }
    }

    @ViewBuilder private func panel(_ tool: IntegrationTool) -> some View {
        FreeRow { ApplicationPanelHead(tool: tool) }
        if tool == .claude {
            Row(label: "运行模式") {
                Popup(selection: $claudeMode, width: .controlMedium, options: [("local","本地模型"),("cloud","Claude 云端")])
            }
        }
        if tool != .claude || claudeMode == "local" {
            modelRow("默认模型", key: tool.rawValue)
            if tool == .claude {
                ForEach(["Opus","Sonnet","Haiku"], id: \.self) { tier in modelRow(tier, key: "claude." + tier) }
                Row(label: "Claude Desktop", sublabel: "通过本地网关连接桌面客户端。") {
                    Toggle("启用桌面入口", isOn: $desktopEnabled).labelsHidden().toggleStyle(.switch)
                }
            }
            if tool == .openclaw {
                Row(label: "工具配置") {
                    Popup(selection: $toolsProfile, width: .controlMedium,
                          options: ["minimal","coding","messaging","full"].map { ($0,$0) })
                }
            }
        }
        ForEach(variants, id: \.self) { variant in
            FreeRow(isLast: variant == variants.last) {
                VStack(alignment: .leading, spacing: 6) {
                    if let command = commands[variant] {
                        CommandBlock(command: command.text, caption: label(variant))
                    } else {
                        Text(service.models.isEmpty ? "服务就绪并加载模型后可使用本地启动命令。" : "正在准备启动命令…")
                            .font(.omlxText(11.5)).foregroundStyle(theme.textTertiary)
                    }
                    HStack {
                        if let status = availability[variant] {
                            Text(status.installed ? "已安装" : "未检测到客户端。" + status.hint)
                                .font(.omlxText(11)).foregroundStyle(theme.textSecondary).textSelection(.enabled)
                        }
                        Spacer()
                        Button("在终端启动") { Task { await open(variant) } }
                            .disabled(commands[variant] == nil || availability[variant]?.installed != true)
                    }
                }
            }
        }
    }
    private func modelRow(_ title: String, key: String) -> some View {
        Row(label: title) {
            Popup(selection: selection(key), width: .controlMedium, options: models.isEmpty ? [("","暂无可用模型")] : models)
                .disabled(models.isEmpty).accessibilityLabel(title)
        }
    }
    private func label(_ id: String) -> String {
        switch id { case "codex_app": "Codex App"; case "claude_desktop": "Claude Desktop"
        default: selected?.title ?? id }
    }
    private func prepareCommands() async {
        commands = [:]; result = ""
        guard let tool = selected else { return }
        if tool == .claude && claudeMode == "cloud" {
            let env = "env -u ANTHROPIC_BASE_URL -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_DEFAULT_OPUS_MODEL -u ANTHROPIC_DEFAULT_SONNET_MODEL -u ANTHROPIC_DEFAULT_HAIKU_MODEL -u ANTHROPIC_MODEL -u API_TIMEOUT_MS -u CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC claude"
            commands["claude"] = .init(text: env, profile: "")
            return
        }
        guard let address = service.browserURL?.absoluteString else { return }
        let model = selection(tool.rawValue).wrappedValue
        guard !model.isEmpty else { return }
        let key = service.settings.mode == .managed && service.apiKey.isEmpty ? MIVManagement.localMasterKey() : service.apiKey
        do {
            let launcher = try ModelToolsRuntime.bundled().resources.appendingPathComponent("miv-launch").path
            for variant in variants {
                var target = IntegrationTarget(tool: variant, model: model, address: address, apiKey: key,
                    contextWindow: service.models.first { $0.id == model }?.maxContext,
                    toolsProfile: tool == .openclaw ? toolsProfile : nil, catalog: service.models.map(\.id))
                if tool == .claude {
                    target.opusModel = selection("claude.Opus").wrappedValue
                    target.sonnetModel = selection("claude.Sonnet").wrappedValue
                    target.haikuModel = selection("claude.Haiku").wrappedValue
                }
                let profile: IntegrationProfile = try await client.post("/integrations/prepare", body: target)
                try Task.checkCancellation()
                let cmd = [launcher,profile.profile].map { "'" + $0.replacingOccurrences(of: "'", with: "'\\''") + "'" }.joined(separator: " ")
                commands[variant] = .init(text: cmd, profile: profile.profile)
            }
        } catch { if !Task.isCancelled { result = error.localizedDescription } }
    }

    private func open(_ variant: String) async {
        guard let command = commands[variant] else { return }
        do {
            let url: URL
            if command.profile.isEmpty {
                let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
                url = dir.appendingPathComponent("Claude.command")
            } else {
                url = URL(fileURLWithPath: command.profile).deletingPathExtension().appendingPathExtension("command")
            }
            let cleanup = command.profile.isEmpty ? "rm -f -- \"$0\"\nrmdir -- \"$(dirname \"$0\")\"\n" : ""
            try ("#!/bin/bash\n" + cleanup + "exec " + command.text + "\n").write(to: url, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: url.path)
            guard let terminal = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Terminal") else {
                throw ModelToolError(message: "找不到 macOS 终端。")
            }
            _ = try await NSWorkspace.shared.open([url], withApplicationAt: terminal, configuration: .init())
        } catch { result = error.localizedDescription }
    }
}

private enum IntegrationTool: String, CaseIterable, Identifiable {
    case claude, codex, dsh, opencode, openclaw, hermes, pi, vibe, copilot

    var id: String { rawValue }

    var assetName: String {
        switch self {
        case .claude: return "IntegrationClaude"
        case .codex: return "IntegrationCodex"
        case .dsh: return "IntegrationDsh"
        case .opencode: return "IntegrationOpenCode"
        case .openclaw: return "IntegrationOpenClaw"
        case .hermes: return "IntegrationHermes"
        case .pi: return "IntegrationPi"
        case .vibe: return "IntegrationVibe"
        case .copilot: return "IntegrationCopilot"
        }
    }

    var title: String {
        switch self {
        case .claude:
            return String(localized: "integrations.tool.claude",
                          defaultValue: "Claude Code",
                          comment: "Display name for the Claude Code integration")
        case .codex:
            return String(localized: "integrations.tool.codex",
                          defaultValue: "Codex",
                          comment: "Display name for the Codex integration")
        case .dsh:
            return String(localized: "integrations.tool.dsh",
                          defaultValue: "DeepSeek Harness",
                          comment: "Display name for the DeepSeek Harness (dsh) integration")
        case .opencode:
            return String(localized: "integrations.tool.opencode",
                          defaultValue: "OpenCode",
                          comment: "Display name for the OpenCode integration")
        case .openclaw:
            return String(localized: "integrations.tool.openclaw",
                          defaultValue: "OpenClaw",
                          comment: "Display name for the OpenClaw integration")
        case .hermes:
            return String(localized: "integrations.tool.hermes",
                          defaultValue: "Hermes Agent",
                          comment: "Display name for the Hermes Agent integration")
        case .pi:
            return String(localized: "integrations.tool.pi",
                          defaultValue: "Pi",
                          comment: "Display name for the Pi integration")
        case .vibe:
            return String(localized: "integrations.tool.vibe",
                          defaultValue: "Mistral Vibe",
                          comment: "Display name for the Mistral Vibe integration")
        case .copilot:
            return String(localized: "integrations.tool.copilot",
                          defaultValue: "Copilot CLI",
                          comment: "Display name for the Copilot CLI integration")
        }
    }
}

private struct ApplicationTile: View {
    let tool: IntegrationTool
    let isSelected: Bool
    let action: () -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Button(action: action) {
            VStack(spacing: 6) {
                IntegrationIcon(asset: tool.assetName, size: 40)
                Text(tool.title)
                    .font(.omlxText(11.5, weight: .medium))
                    .foregroundStyle(isSelected ? theme.text : theme.textSecondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(2, reservesSpace: true)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity)
            }
            .padding(.vertical, 8)
            .padding(.horizontal, 6)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(isSelected ? theme.selBg : Color.clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(isSelected ? theme.accent.opacity(0.55) : Color.clear,
                                  lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
        .buttonStyle(.plain)
        .help(tool.title)
        .accessibilityLabel(Text(tool.title))
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }
}

/// Rounded white chip that holds a tool mark, as in the web launcher's tiles.
private struct IntegrationIcon: View {
    let asset: String
    var size: CGFloat = 40

    @Environment(\.omlxTheme) private var theme

    /// The console draws the mark on a white chip in light appearance and on
    /// its dark tile colour in dark, and the assets carry that background, so
    /// the chip fill has to follow or the rounded corners would show a seam.
    private var chipBackground: Color {
        theme.isDark ? Color(red: 29 / 255, green: 29 / 255, blue: 32 / 255) : .white
    }

    /// The web tile rounds its chip to 8 of 36 points; keep that ratio here.
    private var cornerRadius: CGFloat { size * 8 / 36 }

    var body: some View {
        let filename = asset + (theme.isDark ? "-dark" : "-light")
        let path = Bundle.main.resourceURL?.appendingPathComponent("ModelTools/Icons/").appendingPathComponent(filename + ".png")
        let icon = path.flatMap { NSImage(contentsOf: $0) } ?? NSImage()
        Image(nsImage: icon)
            .resizable()
            .interpolation(.high)
            .scaledToFill()
            .frame(width: size, height: size)
            .background(chipBackground)
            .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(theme.groupBorder, lineWidth: 0.5)
            )
            .accessibilityHidden(true)
    }
}


private struct ApplicationPanelHead: View {
    let tool: IntegrationTool

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 10) {
            IntegrationIcon(asset: tool.assetName, size: 28)
            Text(tool.title)
                .font(.omlxText(DesignTokens.FontSize.body, weight: .semibold))
                .foregroundStyle(theme.text)
            Spacer(minLength: 0)
        }
    }
}

private struct CommandBlock: View {
    let command: String
    var copyLoader: (() async throws -> String)? = nil
    var caption: String? = nil
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            VStack(alignment: .leading, spacing: 4) {
                Text(caption ?? String(localized: "integrations.command.terminal_caption",
                                       defaultValue: "$ Terminal",
                                       comment: "Caption above each shell command block"))
                    .font(.omlxText(DesignTokens.FontSize.caption, weight: .semibold))
                    .foregroundStyle(theme.textTertiary)
                    .textCase(.uppercase)
                    .kerning(0.6)
                Text(command)
                    .font(.omlxMono(DesignTokens.FontSize.callout))
                    .foregroundStyle(theme.text)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            CopyIconButton(
                value: command,
                valueLoader: copyLoader,
                helpText: String(localized: "integrations.command.copy",
                                 defaultValue: "Copy command",
                                 comment: "Tooltip for the copy button on a shell command block")
            )
        }
        .padding(12)
        .background(theme.codeBg)
        .clipShape(RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
    }
}

