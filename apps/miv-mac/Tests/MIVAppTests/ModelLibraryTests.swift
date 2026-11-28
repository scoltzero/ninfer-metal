import Foundation
import XCTest
@testable import MIVApp

final class ModelLibraryTests: XCTestCase {
    @MainActor func testEditingOneTOMLVariantPreservesSiblingAndGlobalDefaults() throws {
        let suite = UUID().uuidString, dir = "/models/weights"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let service = CoreService(defaults: defaults)
        service.configuredModels = ["variant-a", "variant-b"].map {
            .init(id: $0, directory: dir, max_context: 4096, max_concurrency: 1, kv_dtype: "bf16", spec_backend: "none", engine: nil)
        }
        try service.setGlobalGeneration(["max_tokens":"21", "enable_thinking":"false"])
        try service.applyMeasuredContext(8192, modelID: "variant-b")
        let restored = CoreService(defaults: defaults)
        XCTAssertEqual(restored.modelLibrary.map(\.sourceID), ["variant-b"])
        XCTAssertEqual(restored.globalGeneration["max_tokens"], "21")
        let overlay = try XCTUnwrap(try restored.writeModelOverlay())
        defer { try? FileManager.default.removeItem(at: overlay.deletingLastPathComponent()) }
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: overlay)) as? [String:Any])
        XCTAssertEqual((json["defaults"] as? [String:Any])?["enable_thinking"] as? Bool, false)
        let record = try XCTUnwrap((json["models"] as? [[String:Any]])?.first)
        XCTAssertEqual(record["source_id"] as? String, "variant-b")
        XCTAssertNil(record["enabled"])
        XCTAssertEqual(record["engine"] as? [String:Int], ["max_context":8192])
    }

    func testModelOptionsRejectInvalidNumbersAndPreserveInheritedFields() throws {
        XCTAssertThrowsError(try ModelField.typed(["temperature":"nan"], fields: ModelField.generation))
        XCTAssertThrowsError(try ModelField.typed(["top_p":"0"], fields: ModelField.generation))
        XCTAssertThrowsError(try ModelField.typed(["repetition_context_size":"4097"], fields: ModelField.generation))
        XCTAssertThrowsError(try ModelField.typed(["spec_backend":"mtp","draft_tokens":"5"], fields: ModelField.engine))
        XCTAssertThrowsError(try ModelField.typed(["prefix_cache_gb":"129"], fields: ModelField.engine))
        let typed = try ModelField.typed(["draft_tokens":"", "prefill_sparse_minference":"false",
                                        "prefill_attention_mode":"sage2", "prefix_cache_gb":"3.5"], fields: ModelField.engine)
        XCTAssertNil(typed["draft_tokens"])
        XCTAssertEqual(typed["prefill_sparse_minference"] as? Bool, false)
        XCTAssertEqual(typed["prefix_cache_gb"] as? Double, 3.5)
    }

    func testEveryModelFieldRoundTripsThroughItsControlType() throws {
        for field in ModelField.engine + ModelField.generation {
            let choices: [String]
            switch field.kind {
            case .choice(let values): choices = values.map(\.0)
            case .text: choices = field.id == "chat_template_kwargs" ? ["{}", #"{"style":"plain"}"#] : ["END"]
            case .file, .directory: choices = ["/tmp/model A"]
            case .integer, .number: choices = ["1"]
            }
            for choice in choices {
                let value = try ModelField.typed([field.id:choice], fields: [field])
                XCTAssertNotNil(value[field.id], field.id)
            }
        }
    }
}
