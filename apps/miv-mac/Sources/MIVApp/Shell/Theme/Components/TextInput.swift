// PR 3 — text field with focus ring matching the JSX `TextInput` / `BoxedInput`.

import SwiftUI

struct TextInput: View {
    @Binding var text: String
    var titleKey: LocalizedStringKey
    var placeholder: String
    var isSecure: Bool
    var mono: Bool
    var isNumeric: Bool
    var range: ClosedRange<Double>?
    var step: Double?
    var suffix: String?
    var width: CGFloat?
    var onStep: (() -> Void)?

    @Environment(\.omlxTheme) private var theme

    init(_ titleKey: LocalizedStringKey = "", text: Binding<String>, placeholder: String = "", isSecure: Bool = false, mono: Bool = false, isNumeric: Bool = false, range: ClosedRange<Double>? = nil, step: Double? = nil, suffix: String? = nil, width: CGFloat? = nil, onStep: (() -> Void)? = nil) {
        self._text = text
        self.titleKey = titleKey
        self.placeholder = placeholder
        self.isSecure = isSecure
        self.mono = mono
        self.isNumeric = isNumeric
        self.range = range
        self.step = step
        self.suffix = suffix
        self.width = width
        self.onStep = onStep
    }

    private var textAsDouble: Binding<Double> {
        Binding(
            get: { Double(text) ?? 0 },
            set: {
                text = $0.formatted(.number.grouping(.never))
                onStep?()
            }
        )
    }

    var body: some View {
        field
            .lineLimit(1)
            .textFieldStyle(.roundedBorder)
            // Regular weight: native macOS fields never render medium-weight
            // content, and the bolder text read as a foreign control.
            .font(mono ? .omlxMono(DesignTokens.FontSize.body) : .omlxText(DesignTokens.FontSize.body))
            .foregroundStyle(theme.text)
            .overlay(alignment: .trailing) {
                HStack(spacing: 0) {
                    if let suffix {
                        Text(suffix)
                            .font(.omlxText(DesignTokens.FontSize.subheadline))
                            .foregroundStyle(theme.textSecondary)
                            .padding(.horizontal, 10)
                    }
                    if isNumeric {
                        stepper
                    }
                }
            }
            // Fixed, not max: under `maxWidth` a long row label squeezes the
            // field, so fields sharing a width token rendered ragged widths.
            // A fixed frame makes equal tokens render equal; labels wrap.
            .frame(width: width)
    }

    private var field: some View {
        Group {
            if isSecure {
                SecureField(titleKey, text: $text, prompt: Text(placeholder))
            } else {
                TextField(titleKey, text: $text, prompt: Text(placeholder))
            }
        }
    }

    private var stepper: some View {
        Group {
            switch (range, step) {
            case let (range?, step?):
                Stepper(titleKey, value: textAsDouble, in: range, step: step)
            case let (range?, nil):
                Stepper(titleKey, value: textAsDouble, in: range)
            default:
                Stepper(titleKey, value: textAsDouble)
            }
        }
        .labelsHidden()
        .controlSize(.small)
        .padding(.trailing, 1)
    }
}
