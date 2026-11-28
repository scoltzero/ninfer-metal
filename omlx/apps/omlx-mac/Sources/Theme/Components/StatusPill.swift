// PR 3 — colored pill indicating server / model state.
// Mirrors JSX `StatusPill` but accepts a model-state enum so screens
// can render any lifecycle stage uniformly.

import SwiftUI

struct StatusPill: View {
    enum Status: Equatable, Sendable {
        case running
        case starting
        case stopping
        case stopped
        case error
        /// Free-form variant: caller supplies the dot color and label.
        case custom(color: Color, label: String, fillBg: Bool)
    }

    let status: Status
    var compact: Bool = false

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        let cfg = config
        HStack(spacing: 5) {
            Circle()
                .fill(cfg.dot)
                .frame(width: 6, height: 6)
            if !compact {
                Text(cfg.label)
                    .font(.omlxText(DesignTokens.FontSize.subheadline, weight: .medium))
                    .foregroundStyle(cfg.text)
            }
        }
        .padding(.horizontal, compact ? 6 : 8)
        .padding(.vertical, 2.5)
        .background(
            Capsule().fill(cfg.bg)
        )
    }

    private var config: (dot: Color, text: Color, bg: Color, label: String) {
        switch status {
        case .running:
            return (theme.greenDot, theme.successText, theme.successBg, "Running")
        case .starting:
            return (theme.blueDot, theme.blueDot, theme.accentSoft, "Starting")
        case .stopping:
            return (theme.amberDot, theme.warningText, theme.warningBg, "Stopping")
        case .stopped:
            return (theme.textTertiary, theme.textSecondary,
                    theme.controlBg, "Stopped")
        case .error:
            return (theme.redDot, theme.redDot, theme.warningBg, "Error")
        case .custom(let color, let label, let fillBg):
            return (color, color, fillBg ? color.opacity(0.16) : .clear, label)
        }
    }
}

private extension OMLXTheme {
    var mivRouteBadgeColors: (text: Color, bg: Color) { (accent, accentSoft) }
}

/// Subtle pill identifying which inference engine serves a model
/// (`miv Core`, Path B microkernel, `Splash`, or neutral `MLX`).
struct EngineBadgePill: View {
    let engineId: String
    let label: String
    var tooltip: String? = nil

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        let style = palette
        HStack(spacing: 4) {
            Circle()
                .fill(style.dot)
                .frame(width: 5, height: 5)
            Text(label)
                .font(.omlxText(DesignTokens.FontSize.subheadline, weight: .medium))
                .foregroundStyle(style.text)
                .lineLimit(1)
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 2)
        .background(Capsule().fill(style.bg))
        .help(tooltip ?? label)
    }

    private var palette: (dot: Color, text: Color, bg: Color) {
        switch engineId {
        case "m5rt":
            let colors = theme.mivRouteBadgeColors
            return (colors.text, colors.text, colors.bg)
        case "mlx", "", "stock":
            return (theme.textTertiary, theme.textSecondary, theme.controlBg)
        case "splash":
            let teal = Color(nsColor: .systemTeal)
            return (teal, teal, teal.opacity(0.16))
        default:
            let indigo = Color(nsColor: .systemIndigo)
            return (indigo, indigo, indigo.opacity(0.16))
        }
    }
}

/// Compact metadata tag pill used on Integrations microkernel cards
/// (`miv-v0.01`, `Path A`, `Path B`, `SPI v2`).
struct MicrokernelTagPill: View {
    enum Tone {
        case accent
        case indigo
        case neutral
    }

    let label: String
    var tone: Tone = .neutral
    var mono: Bool = false

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        let style = palette
        Text(label)
            .font(mono
                  ? .omlxMono(DesignTokens.FontSize.caption, weight: .medium)
                  : .omlxText(DesignTokens.FontSize.caption, weight: .medium))
            .foregroundStyle(style.text)
            .lineLimit(1)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .fill(style.bg)
            )
    }

    private var palette: (text: Color, bg: Color) {
        switch tone {
        case .accent:
            return (theme.accent, theme.accentSoft)
        case .indigo:
            let indigo = Color(nsColor: .systemIndigo)
            return (indigo, indigo.opacity(0.16))
        case .neutral:
            return (theme.textSecondary, theme.codeBg)
        }
    }
}

/// Monospaced weight-format badge used on `Models` rows and `ModelSettingsScreen.Header`
/// to distinguish `.miv` microkernel weights, pending auto-repack `.safetensors → .miv`,
/// legacy `.omlxm5`, `safetensors`, and `GGUF`.
struct WeightFormatPill: View {
    enum StyleKind {
        case miv
        case mivRouted
        case pendingRepack
        case neutral
    }

    let label: String
    let kind: StyleKind
    var tooltip: String? = nil

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        let style = palette
        Text(label)
            .font(.omlxMono(DesignTokens.FontSize.caption, weight: .medium))
            .foregroundStyle(style.text)
            .lineLimit(1)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .fill(style.bg)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .strokeBorder(style.border, lineWidth: 0.5)
            )
            .help(tooltip ?? label)
            .accessibilityHint(Text(tooltip ?? ""))
    }

    private var palette: (text: Color, bg: Color, border: Color) {
        switch kind {
        case .mivRouted:
            let colors = theme.mivRouteBadgeColors
            return (colors.text, colors.bg, .clear)
        case .miv:
            let teal = Color(nsColor: .systemTeal)
            return (teal, teal.opacity(0.14), teal.opacity(0.35))
        case .pendingRepack:
            let amber = Color(nsColor: .systemOrange)
            return (amber, amber.opacity(0.14), amber.opacity(0.35))
        case .neutral:
            return (theme.textSecondary, theme.codeBg, theme.groupBorder)
        }
    }
}

extension ModelDTO {
    var weightFormatPillKind: WeightFormatPill.StyleKind {
        if pendingAutoRepack == true {
            return .pendingRepack
        }
        switch effectiveWeightFormat {
        case "safetensors", "gguf", "bin":
            return .neutral
        default:
            return .miv
        }
    }
}

#Preview("StatusPill") {
    VStack(alignment: .leading, spacing: 10) {
        StatusPill(status: .running)
        StatusPill(status: .starting)
        StatusPill(status: .stopping)
        StatusPill(status: .stopped)
        StatusPill(status: .error)
        StatusPill(status: .custom(
            color: Color(rgb24: 0x5E5CE6),
            label: "Prefilling 42%",
            fillBg: true
        ))
    }
    .padding(24)
    .omlxThemed()
}
