import AppKit
import XCTest
@testable import MIVApp

final class WindowTests: XCTestCase {
    // Exercise the actual NSHostingView and status-menu selectors. A fixed-width
    // SwiftUI root can silently disable zoom even on an NSWindow marked resizable.
    @MainActor func testWindowResizeZoomAndMenuReopen() async throws {
        guard NSScreen.main != nil else { throw XCTSkip("WindowServer display required") }
        let app = NSApplication.shared
        let delegate = AppDelegate()
        delegate.applicationDidFinishLaunching(Notification(name: NSApplication.didFinishLaunchingNotification))
        let window = try XCTUnwrap(delegate.window)
        defer {
            delegate.applicationWillTerminate(Notification(name: NSApplication.willTerminateNotification))
            window.orderOut(nil)
            if let item = delegate.statusItem { NSStatusBar.system.removeStatusItem(item) }
        }
        XCTAssertTrue(window.isVisible)
        XCTAssertTrue(window.styleMask.contains(.resizable))
        XCTAssertTrue(window.collectionBehavior.contains(.fullScreenPrimary))
        XCTAssertTrue(try XCTUnwrap(window.standardWindowButton(.zoomButton)).isEnabled)
        let original = window.frame
        defer { window.setFrame(original, display: false) }
        window.setContentSize(NSSize(width: 720, height: 490))
        let small = window.frame.size
        window.setContentSize(NSSize(width: 870, height: 610))
        XCTAssertGreaterThan(window.frame.width, small.width)
        XCTAssertGreaterThan(window.frame.height, small.height)
        let beforeZoom = window.frame
        window.performZoom(nil)
        XCTAssertNotEqual(window.frame.size, beforeZoom.size)
        window.performZoom(nil)
        XCTAssertEqual(window.frame.size, beforeZoom.size)

        let entered = expectation(forNotification: NSWindow.didEnterFullScreenNotification, object: window)
        window.toggleFullScreen(nil)
        await fulfillment(of: [entered], timeout: 8)
        XCTAssertTrue(window.styleMask.contains(.fullScreen))
        let exited = expectation(forNotification: NSWindow.didExitFullScreenNotification, object: window)
        window.toggleFullScreen(nil)
        await fulfillment(of: [exited], timeout: 8)
        XCTAssertFalse(window.styleMask.contains(.fullScreen))

        let status = try XCTUnwrap(delegate.statusItem)
        XCTAssertTrue(try XCTUnwrap(status.button?.image).isTemplate)
        let menu = try XCTUnwrap(status.menu)
        delegate.menuNeedsUpdate(menu)
        for page in [AppSection.models, .performance, .service] {
            delegate.navigation.section = .status
            let index = menu.indexOfItem(withTitle: page.title)
            XCTAssertGreaterThanOrEqual(index, 0)
            if index >= 0 { menu.performActionForItem(at: index) }
            XCTAssertEqual(delegate.navigation.section, page)
        }
        window.performMiniaturize(nil)
        delegate.showWindow()
        XCTAssertFalse(window.isMiniaturized)
        window.performClose(nil)
        XCTAssertFalse(window.isVisible)
        XCTAssertFalse(delegate.applicationShouldTerminateAfterLastWindowClosed(app))
        let reopen = menu.indexOfItem(withTitle: "打开控制台")
        XCTAssertGreaterThanOrEqual(reopen, 0)
        if reopen >= 0 { menu.performActionForItem(at: reopen) }
        XCTAssertTrue(window.isVisible)
        XCTAssertTrue(delegate.window === window)
    }
}
