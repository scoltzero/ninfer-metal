import SwiftUI

struct ModelIconTile: View {
    let systemSymbol: String
    let size: CGFloat

    @Environment(\.omlxTheme) private var theme

    init(systemSymbol: String, size: CGFloat = 26) {
        self.systemSymbol = systemSymbol
        self.size = size
    }

    init(modelType: String?, size: CGFloat = 26) {
        self.init(systemSymbol: Self.symbol(for: modelType), size: size)
    }

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: size * 0.27, style: .continuous)
        Image(systemName: systemSymbol)
            .font(.system(size: size * 0.5, weight: .medium))
            .foregroundStyle(theme.textSecondary)
            .frame(width: size, height: size)
            .background(shape.fill(theme.controlBg))
            .overlay(shape.strokeBorder(theme.groupBorder, lineWidth: 0.5))
    }

    static func symbol(for modelType: String?) -> String {
        switch modelType {
        case "vlm": return "text.below.photo"
        case "embedding": return "square.stack.3d.up"
        case "reranker": return "text.line.first.and.arrowtriangle.forward"
        case "decision": return "arrow.triangle.branch"
        case "audio_stt": return "waveform.and.mic"
        case "audio_tts": return "waveform"
        case "audio_sts": return "person.wave.2"
        default: return "text.alignleft"
        }
    }
}

