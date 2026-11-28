import AppKit
import SwiftUI

@main
enum MIVApp {
    @MainActor static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.setActivationPolicy(.regular)
        withExtendedLifetime(delegate) { app.run() }
    }
}

// The oMLX shell keeps one window and runs its service from the menu bar
// after the window closes. Both entry points share the same CoreService.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, NSMenuDelegate {
    let service = CoreService()
    let navigation = NavigationState()
    private(set) var window: NSWindow?
    private(set) var statusItem: NSStatusItem?
    private var polling: Task<Void, Never>?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.mainMenu = makeMainMenu()
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        let image = Bundle.main.url(forResource: "MenuIconTemplate", withExtension: "png")
            .flatMap { NSImage(contentsOf: $0) }
            ?? NSImage(systemSymbolName: "cpu", accessibilityDescription: "MIV")
        image?.size = NSSize(width: 18, height: 18)
        image?.isTemplate = true
        item.button?.image = image
        item.button?.setAccessibilityLabel("MIV")
        item.button?.toolTip = "MIV"
        let menu = NSMenu()
        menu.delegate = self
        menuNeedsUpdate(menu)
        item.menu = menu
        statusItem = item
        showWindow()
        polling = Task { [weak self] in
            while !Task.isCancelled {
                await self?.service.refresh()
                try? await Task.sleep(for: .seconds(3))
            }
        }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
        showWindow()
        return true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func applicationWillTerminate(_ notification: Notification) {
        polling?.cancel()
        service.shutdown()
    }

    @objc func showWindow() {
        if window == nil {
            let frame = NSRect(x: 0, y: 0, width: 932, height: 650)
            let main = NSWindow(contentRect: frame, styleMask: [.titled, .closable, .miniaturizable, .resizable],
                                backing: .buffered, defer: false)
            main.identifier = .init("MIVMain")
            main.title = "MIV"
            main.isReleasedWhenClosed = false
            main.collectionBehavior.insert(.fullScreenPrimary)
            main.contentMinSize = .init(width: 680, height: 460)
            let content = NSHostingView(rootView: AppView(service: service, navigation: navigation))
            content.sizingOptions = []
            main.contentView = content
            main.center()
            main.setFrameAutosaveName("MIVMainWindow")
            main.delegate = self
            window = main
        }
        NSApp.setActivationPolicy(.regular)
        if window?.isMiniaturized == true { window?.deminiaturize(nil) }
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func windowWillClose(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
    }

    @objc private func openSection(_ sender: NSMenuItem) {
        guard let name = sender.representedObject as? String, let page = AppSection(rawValue: name) else { return }
        navigation.section = page
        showWindow()
    }

    @objc private func openWeb() {
        guard let url = service.browserURL else { return }
        NSWorkspace.shared.open(url)
    }

    @objc private func toggleService() {
        if service.isRunning {
            service.stop()
        } else {
            do { try service.start() }
            catch {
                navigation.section = .service
                navigation.notice = error.localizedDescription
                showWindow()
            }
        }
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()
        menu.autoenablesItems = false
        let status = menu.addItem(withTitle: "MIV · " + service.statusText, action: nil, keyEquivalent: "")
        status.isEnabled = false
        if let rate = service.outputRate {
            menu.addItem(withTitle: "输出 " + displayRate(rate), action: nil, keyEquivalent: "").isEnabled = false
        }
        menu.addItem(.separator())
        _ = add("打开控制台", to: menu, action: #selector(showWindow))
        for section in [AppSection.models, .performance, .service] {
            let link = add(section.title, to: menu, action: #selector(openSection(_:)))
            link.representedObject = section.rawValue
        }
        add("打开网页", to: menu, action: #selector(openWeb)).isEnabled = service.dashboardAvailable
        if service.settings.mode == .managed {
            menu.addItem(.separator())
            let title = service.isStopping ? "正在停止…" : service.isRunning ? "停止服务" : "启动服务"
            add(title, to: menu, action: #selector(toggleService)).isEnabled = !service.isStopping
        }
        menu.addItem(.separator())
        menu.addItem(withTitle: "退出 MIV", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
    }

    @discardableResult
    private func add(_ title: String, to menu: NSMenu, action: Selector) -> NSMenuItem {
        let item = menu.addItem(withTitle: title, action: action, keyEquivalent: "")
        item.target = self
        return item
    }

    private func makeMainMenu() -> NSMenu {
        let main = NSMenu()
        func submenu(_ name: String) -> NSMenu {
            let item = NSMenuItem()
            let menu = NSMenu(title: name)
            main.addItem(item)
            item.submenu = menu
            return menu
        }
        let app = submenu("MIV")
        app.addItem(withTitle: "关于 MIV", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        app.addItem(.separator())
        app.addItem(withTitle: "隐藏 MIV", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        app.addItem(withTitle: "退出 MIV", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let edit = submenu("编辑")
        edit.addItem(withTitle: "撤销", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "重做", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "剪切", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "拷贝", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "粘贴", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "全选", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        let windows = submenu("窗口")
        windows.addItem(withTitle: "关闭窗口", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        windows.addItem(withTitle: "最小化", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        windows.addItem(withTitle: "缩放", action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
        let full = windows.addItem(withTitle: "切换全屏", action: #selector(NSWindow.toggleFullScreen(_:)), keyEquivalent: "f")
        full.keyEquivalentModifierMask = [.command, .control]
        windows.addItem(.separator())
        add("打开控制台", to: windows, action: #selector(showWindow))
        NSApp.windowsMenu = windows
        return main
    }
}
