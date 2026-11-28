import XCTest
@testable import oMLX

final class ModelsScreenSortingTests: XCTestCase {

    func testSortModelsByNameIgnoresCase() throws {
        let models = try [
            makeModel("Qwen"),
            makeModel("gpt"),
            makeModel("Llama"),
            makeModel("mistral"),
        ]

        let ids = sortModelsByName(models).map(\.id)

        XCTAssertEqual(ids, ["gpt", "Llama", "mistral", "Qwen"])
    }

    func testSortModelsByNamePreservesInputOrderForCaseOnlyTies() throws {
        let models = try [
            makeModel("qwen"),
            makeModel("Qwen"),
            makeModel("QWEN"),
        ]

        let ids = sortModelsByName(models).map(\.id)

        XCTAssertEqual(ids, ["qwen", "Qwen", "QWEN"])
    }

    func testSortModelsByNameUsesDisplayNameWhenPresent() throws {
        let models = try [
            makeModel("llama", displayName: "Meta/llama"),
            makeModel("Qwen", displayName: "deepsweet/Qwen"),
            makeModel("gemma", displayName: "Google/gemma"),
        ]

        let ids = sortModelsByName(models).map(\.id)

        XCTAssertEqual(ids, ["Qwen", "gemma", "llama"])
    }

    func testAliasIsDisplayedAndCopiedWithoutChangingIdentity() throws {
        let id = "Qwen3.8-27B-oQ6e-mtp"
        let model = try makeModel(
            id, displayName: "Jundot/\(id)", settings: ["model_alias": "test"]
        )

        XCTAssertEqual(model.displayAlias, "test")
        XCTAssertEqual(model.apiModelName, "test")
        XCTAssertEqual(model.id, id)
        XCTAssertEqual(model.displayTitle, "Jundot/\(id)")
    }

    func testAbsentAliasesHideBadgeAndCopyModelID() throws {
        let settingsCases: [Any?] = [
            nil, NSNull(), [String: String](),
            ["model_alias": NSNull()], ["model_alias": ""],
        ]
        for settings in settingsCases {
            let model = try makeModel("model-id", displayName: "Display name", settings: settings)

            XCTAssertNil(model.displayAlias)
            XCTAssertEqual(model.apiModelName, "model-id")
            XCTAssertEqual(model.displayTitle, "Display name")
        }
    }

    func testSortModelsByNameDoesNotUseAliases() throws {
        let models = try [
            makeModel("first", displayName: "Alpha", settings: ["model_alias": "zulu"]),
            makeModel("second", displayName: "Zulu", settings: ["model_alias": "alpha"]),
        ]

        XCTAssertEqual(sortModelsByName(Array(models.reversed())).map(\.id), ["first", "second"])
    }

    func testUnconvertedRowsHideTheFormatAndMivRoutePills() throws {
        let model = try makeModel("source", metadata: [
            "weight_format": "safetensors", "has_miv_weights": false,
            "pending_auto_repack": true, "repack_target_format": "miv", "can_convert": true,
            "active_engine_id": "m5rt", "miv_route_enabled": true,
        ])
        XCTAssertNil(model.listMivBadge)
        XCTAssertFalse(model.showsOtherListWeightFormat)
        XCTAssertFalse(model.showsListEngineBadge(inActiveModels: true))
        XCTAssertFalse(model.showsListEngineBadge(inActiveModels: false))
        XCTAssertTrue(model.canConvertMiv)
    }

    func testConvertedBadgeUsesEffectiveRoutingEvenWithColocatedWeightsOrAStaleEngine() throws {
        let available = try makeModel("mixed", metadata: [
            "weight_format": "safetensors", "miv_artifact_format": "miv",
            "active_engine_id": "m5rt", "auto_external_engine": "m5rt", "miv_route_enabled": false,
        ])
        XCTAssertEqual(available.listMivBadge, .available)
        XCTAssertFalse(available.showsOtherListWeightFormat)
        XCTAssertFalse(available.showsListEngineBadge(inActiveModels: false))

        let routed = try makeModel("cached", metadata: [
            "weight_format": "safetensors", "miv_artifact_format": "omlxm5",
            "active_engine_id": "mlx", "miv_route_enabled": true,
        ])
        XCTAssertEqual(routed.listMivBadge, .routed)
        XCTAssertFalse(routed.showsOtherListWeightFormat)
    }

    func testLegacyFormatMetadataAndStockOverride() throws {
        let mixed = try makeModel("mixed", settings: ["external_engine": "stock"], metadata: [
            "weight_format": "safetensors", "has_miv_weights": true,
            "active_engine_id": "m5rt", "auto_external_engine": "m5rt",
        ])
        XCTAssertEqual(mixed.listMivBadge, .available)

        let legacy = try makeModel("legacy", metadata: [
            "model_format": "omlxm5", "active_engine_id": "io.ninfer.metal",
        ])
        XCTAssertEqual(legacy.listMivBadge, .routed)
        XCTAssertFalse(legacy.showsListEngineBadge(inActiveModels: true))

        let stock = try makeModel("stock", settings: ["external_engine": "stock"], metadata: [
            "has_miv_weights": true, "active_engine_id": "mlx", "auto_external_engine": "m5rt",
        ])
        XCTAssertEqual(stock.listMivBadge, .available)
        XCTAssertTrue(stock.showsListEngineBadge(inActiveModels: false))
    }

    func testUnrelatedFormatsAndEnginesStillHaveTheirOwnBadges() throws {
        let gguf = try makeModel("gguf", metadata: ["model_format": "gguf", "active_engine_id": "splash"])
        XCTAssertNil(gguf.listMivBadge)
        XCTAssertTrue(gguf.showsOtherListWeightFormat)
        XCTAssertTrue(gguf.showsListEngineBadge(inActiveModels: true))

        var custom = try makeModel("path-b", metadata: [
            "weight_format": "other", "has_miv_weights": true, "active_engine_id": "other-provider",
        ])
        XCTAssertNil(custom.listMivBadge)
        XCTAssertTrue(custom.showsOtherListWeightFormat)
        XCTAssertTrue(custom.showsListEngineBadge(inActiveModels: false))
        custom.mivArtifactFormat = "miv"
        XCTAssertEqual(custom.listMivBadge, .available)
        XCTAssertTrue(custom.showsOtherListWeightFormat)
    }

    func testCurrentServerMissingArtifactTakesPrecedenceOverLegacyFormatHints() throws {
        for format in ["safetensors", "omlxm5"] {
            let model = try makeModel("path-b", metadata: [
                "weight_format": format, "has_miv_weights": true, "active_engine_id": "other-provider",
                "miv_artifact_format": NSNull(), "miv_route_enabled": false,
            ])
            XCTAssertNil(model.listMivBadge)
            XCTAssertEqual(model.showsOtherListWeightFormat, format == "omlxm5")
            XCTAssertTrue(model.showsListEngineBadge(inActiveModels: false))
            if format == "omlxm5" { XCTAssertEqual(model.weightFormatBadgeLabel, ".omlxm5") }
        }
    }

    private func makeModel(
        _ id: String, displayName: String? = nil, settings: Any? = nil, metadata: [String: Any] = [:]
    ) throws -> ModelDTO {
        var json: [String: Any] = [
            "id": id, "loaded": false, "is_loading": false, "estimated_size": 0,
        ]
        json["display_name"] = displayName
        json["settings"] = settings
        json.merge(metadata) { _, new in new }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(ModelDTO.self, from: JSONSerialization.data(withJSONObject: json))
    }
}
