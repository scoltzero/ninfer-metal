// PR 3 — rounded card that hosts a stack of `Row`s. Mirrors JSX `ListGroup`.

import SwiftUI

struct ListGroup<Content: View>: View {
    let content: () -> Content

    @Environment(\.omlxTheme) private var theme

    init(@ViewBuilder content: @escaping () -> Content) {
        self.content = content
    }

    var body: some View {
        VStack(spacing: 0) {
            content()
        }
        .background(theme.groupBg)
        .clipShape(RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous))
        .padding(.horizontal, 14)
        .padding(.bottom, 6)
    }
}
