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
    let activity = DashboardActivity()
    private let systemItems = SystemMenubarItemsController()
    private lazy var metricItems = MenubarMetricItemsController(store: activity.graph,
        openAppearanceSettings: { [weak self] in self?.navigate(.appearance) },
        openDashboard: { [weak self] in self?.openWeb() })
    static let restoreIconRequestNotification = Notification.Name("MIVRestoreMenuIcon")


    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.mainMenu = makeMainMenu()
        installStatusItem()
        NotificationCenter.default.addObserver(self, selector: #selector(restoreStatusItem), name: Self.restoreIconRequestNotification, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(preferencesChanged), name: UserDefaults.didChangeNotification, object: nil)
        metricItems.willShowPopover = { [weak self] in self?.systemItems.closeAllPopovers() }
        systemItems.willShowPopover = { [weak self] in self?.metricItems.closeAllPopovers() }
        showWindow()
        if MIVManagement.settings.autoStart && service.settings.mode == .managed {
            do { try service.start() } catch { navigation.notice = error.localizedDescription }
        }
        Task { await service.refreshModelLibrary() }
        polling = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await service.refresh()
                activity.sample(service: service)
                metricItems.sync()
                systemItems.apply(activity.system)
                try? await Task.sleep(for: .seconds(MenubarMetricPrefs.refreshInterval))
            }
        }
    }

    @objc private func preferencesChanged() {
        metricItems.sync()
        systemItems.sync()
        if window?.isVisible == false { NSApp.setActivationPolicy(MenubarMetricPrefs.showDockIcon ? .regular : .accessory) }
    }

    @objc private func restoreStatusItem() {
        if let item = statusItem { NSStatusBar.system.removeStatusItem(item) }
        installStatusItem()
        metricItems.rebuild()
        systemItems.rebuild()
    }

    private func installStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        let image = Bundle.main.url(forResource: "MenuIconTemplate", withExtension: "png")
            .flatMap { NSImage(contentsOf: $0) }
            ?? NSImage(systemSymbolName: "cpu", accessibilityDescription: "MIV")
        image?.size = NSSize(width: 18, height: 18)
        image?.isTemplate = true
        item.autosaveName = "miv.main"
        item.isVisible = true
        item.button?.image = image
        item.button?.setAccessibilityLabel("MIV")
        item.button?.toolTip = "MIV"
        let menu = NSMenu()
        menu.delegate = self
        menuNeedsUpdate(menu)
        item.menu = menu
        statusItem = item

    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
        showWindow()
        return true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func applicationWillTerminate(_ notification: Notification) {
        polling?.cancel()
        NotificationCenter.default.removeObserver(self)
        activity.save()
        BenchmarkQueue.shared.shutdown()
        service.shutdown()
    }

    @objc func showWindow() {
        if window == nil {
            let frame = NSRect(x: 0, y: 0, width: 920, height: 600)
            let main = NSWindow(contentRect: frame, styleMask: [.titled, .closable, .miniaturizable, .resizable],
                                backing: .buffered, defer: false)
            main.identifier = .init("MIVMain")
            main.title = "MIV"
            main.toolbarStyle = .unified
            main.isReleasedWhenClosed = false
            main.collectionBehavior.insert(.fullScreenPrimary)
            main.contentMinSize = .init(width: 680, height: 460)
            let content = NSHostingView(rootView: AppView(service: service, navigation: navigation).environmentObject(activity))
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
        NSApp.setActivationPolicy(MenubarMetricPrefs.showDockIcon ? .regular : .accessory)
    }

    @objc private func openSection(_ sender: NSMenuItem) {
        guard let name = sender.representedObject as? String, let page = AppSection(rawValue: name) else { return }
        navigate(page)
    }

    private func navigate(_ page: AppSection) {
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
                navigation.section = .server
                navigation.notice = error.localizedDescription
                showWindow()
            }
        }
    }

    @objc private func restart() {
        Task {
            do { try await service.restartManaged() }
            catch { navigation.notice = error.localizedDescription; navigate(.server) }
        }
    }

    @objc private func openChat() { if let url = service.chatURL { NSWorkspace.shared.open(url) } }
    @objc private func openReleases() { NSWorkspace.shared.open(MIVAboutScreen.repository.appendingPathComponent("releases")) }
    @objc private func showAbout() { navigate(.about) }
    @objc private func showModel(_ sender: NSMenuItem) {
        navigate(.models)
        guard let id = sender.representedObject as? String else { return }
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(100))
            NotificationCenter.default.post(name: .init("MIVShowModel"), object: id)
        }
    }
    @objc private func toggleModel(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? String, let model = service.libraryRows.first(where: { $0.id == id }) else { return }
        let loaded = Set((service.health?.models ?? []).map(\.id))
        let isLoaded = loaded.contains(model.apiID)
        Task {
            do { try await service.setModelLoaded(model, loaded: !isLoaded) }
            catch { navigation.notice = error.localizedDescription; navigate(.models) }
        }
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()
        menu.autoenablesItems = false
        func line(_ title: String, in menu: NSMenu) { menu.addItem(withTitle: title, action: nil, keyEquivalent: "").isEnabled = false }
        func item(_ title: String, _ symbol: String, _ action: Selector?, enabled: Bool = true, in target: NSMenu? = nil) -> NSMenuItem {
            let row = (target ?? menu).addItem(withTitle: title, action: action, keyEquivalent: "")
            row.target = self
            row.image = NSImage(systemSymbolName: symbol, accessibilityDescription: nil)
            row.isEnabled = enabled
            return row
        }
        func submenu(_ title: String, _ symbol: String) -> NSMenu {
            let child = NSMenu()
            child.autoenablesItems = false
            item(title, symbol, nil).submenu = child
            return child
        }

        line("服务器：" + service.statusText, in: menu)
        menu.addItem(.separator())
        if service.settings.mode == .managed {
            if service.isRunning {
                _ = item("重启服务器", "arrow.clockwise.circle", #selector(restart), enabled: !service.isStopping)
                _ = item("停止服务器", "stop.circle", #selector(toggleService), enabled: !service.isStopping)
            } else {
                _ = item("启动服务器", "play.circle", #selector(toggleService))
            }
        }
        let models = submenu("模型", "shippingbox")
        let loaded = Set((service.health?.models ?? []).map(\.id))
        let library = service.settings.mode == .managed ? service.libraryRows : []
        let visible = library.filter { model in
            MenubarMetricPrefs.modelLibraryScope == .all || model.favorite || loaded.contains(model.apiID)
        }
        var represented = Set<String>()
        for model in visible {
            let apiID = model.apiID
            represented.insert(apiID)
            let row = item(model.title, model.favorite ? "star.fill" : "cube", nil, in: models)
            row.state = loaded.contains(apiID) ? .on : .off
            let detail = NSMenu()
            detail.autoenablesItems = false
            row.submenu = detail
            item(loaded.contains(apiID) ? "卸载" : "加载", "power", #selector(toggleModel(_:)), enabled: !service.isChangingModels && !service.isStopping, in: detail).representedObject = model.id
            item("模型设置…", "slider.horizontal.3", #selector(showModel(_:)), in: detail).representedObject = model.id
        }
        for id in loaded.subtracting(represented).sorted() { line(id, in: models) }
        if models.items.isEmpty { line("暂无可用模型", in: models) }
        models.addItem(.separator())
        let all = add("管理模型…", to: models, action: #selector(openSection(_:)))
        all.representedObject = AppSection.models.rawValue

        menu.addItem(.separator())
        let system = submenu("系统统计", "cpu")
        let host = NSHostingView(rootView: SystemStatsPanelStack(kinds: [.cpu, .gpu, .memory], snapshot: activity.system,
                                        refreshInterval: MenubarMetricPrefs.refreshInterval).omlxThemed())
        host.setFrameSize(host.fittingSize)
        let panel = NSMenuItem()
        panel.view = host
        system.addItem(panel)
        let stats = submenu("服务统计", "chart.bar")
        line("本次核心运行", in: stats)
        line("预填充 Tokens   " + CountFormat.compact(Int(exactly: activity.served.prompt)), in: stats)
        line("生成 Tokens   " + CountFormat.compact(Int(exactly: activity.served.output)), in: stats)
        line("缓存 Tokens   " + CountFormat.compact(Int(exactly: activity.served.cached)), in: stats)
        if let rate = activity.live?.generationTps { line("当前输出   " + displayRate(rate), in: stats) }
        stats.addItem(.separator())
        line("本机累计观测", in: stats)
        line("请求   " + CountFormat.compact(Int(exactly: activity.lifetime.usage.requests)), in: stats)
        let performance = add("性能设置…", to: stats, action: #selector(openSection(_:)))
        performance.representedObject = AppSection.performance.rawValue
        menu.addItem(.separator())
        _ = item("网页控制台", "safari", #selector(openWeb), enabled: service.dashboardAvailable)
        _ = item("聊天", "bubble.left.and.bubble.right", #selector(openChat), enabled: service.dashboardAvailable)
        menu.addItem(.separator())
        _ = item("查看发布版本…", "arrow.down.circle", #selector(openReleases))
        _ = item("管理面板", "slider.horizontal.3", #selector(showWindow))
        _ = item("关于 MIV", "info.circle", #selector(showAbout))
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
        add("关于 MIV", to: app, action: #selector(showAbout))
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
