import AppKit
import SwiftUI

struct MIVAboutScreen: View {
    @ObservedObject var service: CoreService
    @Environment(\.omlxTheme) private var theme
    static let repository = URL(string: "https://github.com/scoltzero/ninfer-metal")!
    private var version: String { Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "开发版" }
    private var build: String { Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "—" }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 16) {
                Image(nsImage: NSApp.applicationIconImage).resizable().interpolation(.high).frame(width: 69, height: 69)
                VStack(alignment: .leading, spacing: 4) {
                    Text("MIV").font(.omlxText(DesignTokens.FontSize.title1, weight: .semibold))
                    Text("Apple Silicon 原生 Metal 推理引擎").font(.omlxText(DesignTokens.FontSize.callout)).foregroundStyle(theme.textSecondary)
                    Text("版本 \(version) · 构建 \(build)").font(.omlxMono(DesignTokens.FontSize.subheadline)).foregroundStyle(theme.textTertiary)
                }
                Spacer()
            }.padding(18).background(theme.groupBg)
                .clipShape(RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: theme.cornerRadius).strokeBorder(theme.groupBorder, lineWidth: 0.5))
                .padding(.horizontal, 14).padding(.bottom, 14)
            SectionHeader("项目")
            ListGroup {
                LinkRow(label: "GitHub 仓库", sublabel: "源代码和路线图", icon: "chevron.left.forwardslash.chevron.right", url: Self.repository)
                LinkRow(label: "发布版本", sublabel: "核心和应用的发布记录", icon: "shippingbox", url: Self.repository.appendingPathComponent("releases"))
                LinkRow(label: "文档", sublabel: "模型、配置与集成", icon: "book.closed", url: Self.repository.appendingPathComponent("tree/main/docs"))
                LinkRow(label: "报告问题", sublabel: "向项目提交错误报告与功能建议", icon: "exclamationmark.bubble", url: Self.repository.appendingPathComponent("issues/new"), isLast: true)
            }
            SectionHeader("许可证")
            ListGroup {
                Row(label: "ninfer-metal", sublabel: "GNU General Public License v3.0") {
                    licenseButton("MIV-GPL-3.0")
                }
                Row(label: "oMLX 原生界面", sublabel: "Apache License 2.0", isLast: true) {
                    licenseButton("oMLX-Apache-2.0")
                }
            }
            SectionHeader("致谢")
            ListGroup {
                LinkRow(label: "oMLX", sublabel: "Swift 外壳和主题组件", icon: "macwindow",
                        url: URL(string: "https://github.com/jundot/omlx")!, isLast: true)
            }
        }
    }
    @ViewBuilder private func licenseButton(_ name: String) -> some View {
        if let url = Bundle.main.resourceURL?.appendingPathComponent("Licenses").appendingPathComponent(name),
           FileManager.default.isReadableFile(atPath: url.path) {
            Button("显示文件") { NSWorkspace.shared.activateFileViewerSelecting([url]) }.buttonStyle(.omlx(.normal, size: .small))
        }
    }
}
