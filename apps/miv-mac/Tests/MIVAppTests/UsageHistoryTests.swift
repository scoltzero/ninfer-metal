import XCTest
@testable import MIVApp

final class UsageHistoryTests: XCTestCase {
    func testNativeHTTPLogThresholdFiltersTableAndRawView() {
        let lines = LogLevel.allCases.map { "[ninfer-metal-serve] " + $0.rawValue + " GET /health" }
        let text = lines.joined(separator: "\n")
        let records = LogParser.parse(text, totalLines: lines.count)
        XCTAssertEqual(records.compactMap(\.level), LogLevel.allCases)
        XCTAssertEqual(LogParser.rows(records, minLevel: .info).count, 4)
        XCTAssertEqual(LogParser.filter(text, minLevel: .debug), lines.dropFirst().joined(separator: "\n"))
    }

    @MainActor func testHourlyDeltasRemainSeparateAcrossModelsEndpointsAndRecordingGaps() throws {
        let domain = "io.ninfer.miv.historytest." + UUID().uuidString
        let defaults = try XCTUnwrap(UserDefaults(suiteName: domain))
        defer { defaults.removePersistentDomain(forName: domain) }
        let activity = DashboardActivity(defaults: defaults)
        func tick(_ time: Double, _ count: Double, endpoint: String = "first", record: Bool = true) {
            activity.observe(endpoint: endpoint, running: true,
                metrics: ["ninfer_metal_prompt_tokens_total": count * 2,
                          "ninfer_metal_generated_tokens_total": count,
                          "ninfer_metal_requests_total": count],
                modelCounters: ["model": (requests: count, output: count)], record: record, now: time)
        }
        tick(100, 20)
        tick(102, 26)
        XCTAssertEqual(activity.live?.generationTps, 3)
        XCTAssertEqual(activity.savedHours.values.first?.models["model"]?.output, 6)
        tick(104, 35, record: false)
        tick(106, 40)
        tick(108, 41)
        XCTAssertEqual(activity.savedHours.values.first?.all.prompt, 14)
        XCTAssertEqual(activity.lifetime.usage.output, 7)
        tick(110, 90, endpoint: "other")
        XCTAssertTrue(activity.savedHours.isEmpty)
        XCTAssertNil(activity.live)
        tick(112, 41)
        XCTAssertEqual(activity.savedHours.values.first?.all.output, 7)
        activity.clear(allTime: false)
        tick(114, 44)
        XCTAssertEqual(activity.served.output, 3)
        activity.save()
        let restored = DashboardActivity(defaults: defaults)
        restored.observe(endpoint: "first", running: false, metrics: [:], modelCounters: [:], record: false, now: 120)
        XCTAssertEqual(restored.savedHours.values.first?.all.output, 10)
        restored.clear(allTime: true)
        XCTAssertTrue(restored.savedHours.isEmpty)
    }
}
