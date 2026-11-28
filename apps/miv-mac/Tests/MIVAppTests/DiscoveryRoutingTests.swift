import Foundation
import XCTest
@testable import MIVApp

final class DiscoveryRoutingTests: XCTestCase {
    func testStandardHubLocationsAndUsableRevisions() throws {
        let fm = FileManager.default
        let home = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? fm.removeItem(at: home) }
        XCTAssertEqual(HFCacheDiscovery.hub(environment: [:], home: home).path, home.path + "/.cache/huggingface/hub")
        let hub = HFCacheDiscovery.hub(environment: ["HF_HOME": home.path], home: home)
        XCTAssertEqual(hub.path, home.path + "/hub")
        XCTAssertEqual(HFCacheDiscovery.hub(environment: ["HF_HUB_CACHE": home.path, "HF_HOME":"/ignored"]).path, home.path)

        func put(_ relative: String, _ value: String = "") throws {
            let target = hub.appendingPathComponent(relative)
            try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(value.utf8).write(to: target)
        }
        let repo = "models--Example--Chat"
        try put(repo + "/refs/main", "older")
        try put(repo + "/snapshots/older/config.json", "{}")
        try put(repo + "/snapshots/older/model.safetensors.index.json", #"{"weight_map":{"w":"part1.safetensors","v":"part2.safetensors"}}"#)
        try put(repo + "/snapshots/older/part1.safetensors", "weights")
        try put(repo + "/snapshots/newer/config.json", "{}")
        try put(repo + "/blobs/prepared", "fixture")
        let weights = hub.appendingPathComponent(repo + "/snapshots/newer/model.miv")
        try fm.createSymbolicLink(atPath: weights.path, withDestinationPath: "../../blobs/prepared")
        try put("models--Incomplete/snapshots/hash/config.json")
        try put("datasets--Ignored/snapshots/hash/config.json")
        try put("datasets--Ignored/snapshots/hash/model.miv")
        let first = HFCacheDiscovery.scan(hub)
        XCTAssertEqual(first.map(\.repoID), ["Example/Chat"])
        XCTAssertEqual(first.first?.routeID, "Example--Chat")
        XCTAssertEqual(URL(fileURLWithPath: try XCTUnwrap(first.first?.directory)).lastPathComponent, "newer")
        XCTAssertEqual(first.first?.hasNativeWeights, true)
        try put(repo + "/snapshots/older/part2.safetensors")
        XCTAssertEqual(HFCacheDiscovery.scan(hub).first?.hasNativeWeights, true, "an empty shard is incomplete")
        try put(repo + "/snapshots/older/part2.safetensors", "weights")
        let second = HFCacheDiscovery.scan(hub)
        XCTAssertEqual(URL(fileURLWithPath: try XCTUnwrap(second.first?.directory)).lastPathComponent, "older")
        XCTAssertEqual(second.first?.hasNativeWeights, false)
        XCTAssertEqual(try fm.destinationOfSymbolicLink(atPath: weights.path), "../../blobs/prepared")
        XCTAssertEqual(try String(contentsOf: hub.appendingPathComponent(repo + "/refs/main"), encoding: .utf8), "older")
    }

    @MainActor func testFallbackDisabledByDefaultAndDiscoveryCanBeTurnedOff() throws {
        let name = UUID().uuidString, defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let service = CoreService(defaults: defaults)
        XCTAssertFalse(service.fallbackPolicy.enabled)
        XCTAssertTrue(service.hfDiscoveryEnabled)
        service.hfCacheModels = [.init(directory:"/cache/revision", routeID:"org--model",repoID:"org/model",hasNativeWeights:false)]
        let cached = try XCTUnwrap(service.libraryRows.first)
        XCTAssertTrue(service.needsPreparedHFWeights(cached))
        service.setHFDiscovery(false)
        XCTAssertTrue(service.libraryRows.isEmpty)
        try service.setFallbackPolicy(.init(enabled:true, preferred:"loaded-model"))
        let restored = CoreService(defaults: defaults)
        XCTAssertFalse(restored.hfDiscoveryEnabled)
        XCTAssertEqual(restored.fallbackPolicy, MIVFallbackPolicy(enabled:true, preferred:"loaded-model"))
    }
}
