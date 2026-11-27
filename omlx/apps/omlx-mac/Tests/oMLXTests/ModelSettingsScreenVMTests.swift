import XCTest
import SwiftUI
@testable import oMLX

@MainActor
final class ModelSettingsScreenVMTests: XCTestCase {

    /// Some locales put a (narrow) no-break space before "%", e.g. "42 %".
    private func normalizedPercent(_ label: String) -> String {
        label
            .replacingOccurrences(of: "\u{202F}", with: "")
            .replacingOccurrences(of: "\u{00A0}", with: "")
            .replacingOccurrences(of: " ", with: "")
    }

    func testSnapshotResultHelpersFormatSkippedStatsAndAppliedJSON() {
        let skipped = [
            SkippedFeatureDTO(feature: "dflash", reason: "draft model 'x' is not installed"),
            SkippedFeatureDTO(feature: "a8_prefill", reason: "cannot be combined with qwen35_ane_prefill_enabled"),
        ]
        XCTAssertEqual(
            ModelSettingsScreenVM.summarizeSkipped(skipped),
            ["dflash: draft model 'x' is not installed",
             "a8_prefill: cannot be combined with qwen35_ane_prefill_enabled"]
        )
        XCTAssertEqual(ModelSettingsScreenVM.summarizeSkipped(nil), [])

        XCTAssertEqual(
            ModelSettingsScreenVM.candidateStats(pp: 1309.1, tg: 59.6, memoryGb: 128, quantization: "4bit", omlxVersion: "0.7.0"),
            "PP 1309.1 tok/s · TG 59.6 tok/s · 128 GB · 4bit · oMLX 0.7.0"
        )
        XCTAssertEqual(
            ModelSettingsScreenVM.candidateStats(pp: 12.0, tg: nil, quantization: nil, omlxVersion: ""),
            "PP 12.0 tok/s"
        )

        let json = ModelSettingsScreenVM.appliedJSON([
            "turboquant_kv_enabled": AnyCodable(true),
            "temperature": AnyCodable(0.6),
        ])
        XCTAssertEqual(
            json,
            """
            {
              "temperature" : 0.6,
              "turboquant_kv_enabled" : true
            }
            """
        )
        XCTAssertEqual(ModelSettingsScreenVM.appliedJSON(nil), "{}")
    }

    func testModelTypeOptionsMatchServerValues() {
        let values = ModelSettingsScreenVM.modelTypeOptions.map(\.0)

        XCTAssertEqual(
            values,
            [
                "",
                "llm",
                "vlm",
                "embedding",
                "reranker",
                "audio_stt",
                "audio_tts",
                "audio_sts",
                "decision",
            ]
        )
    }

    func testAffine8IsIncludedInWorkingProfile() {
        let vm = ModelSettingsScreenVM()
        vm.turboquantKvEnabled = true
        vm.turboquantKvScheme = "affine8"
        vm.turboquantKvBits = "4"

        let settings = vm.currentSettingsDict()

        XCTAssertEqual(settings["turboquant_kv_enabled"]?.value as? Bool, true)
        XCTAssertEqual(settings["turboquant_kv_scheme"]?.value as? String, "affine8")
        XCTAssertEqual(settings["turboquant_kv_bits"]?.value as? Double, 8)
    }

    func testLightningMtpAllowsTurboQuantInWorkingProfile() {
        let vm = ModelSettingsScreenVM()
        vm.mtpEnabled = true
        vm.turboquantKvEnabled = true

        XCTAssertNil(vm.mtpConflictReason)

        let settings = vm.currentSettingsDict()
        XCTAssertEqual(settings["mtp_enabled"]?.value as? Bool, true)
        XCTAssertEqual(settings["turboquant_kv_enabled"]?.value as? Bool, true)
    }

    func testLightningMtpAdaptiveMaxDepthInWorkingProfile() throws {
        let vm = ModelSettingsScreenVM()
        vm.mtpEnabled = true
        XCTAssertNil(vm.currentSettingsDict()[ProfileSettingsKey.mtpAdaptiveMaxDepth])
        XCTAssertEqual(ModelSettingsScreenVM.mtpDepthOptions.map(\.0), ["", "3", "4", "5", "6", "7", "8"])

        // The model default must send JSON null, not omit the key.
        var patch = ModelSettingsPatch()
        patch.mtpAdaptiveMaxDepth = .some(Int(vm.mtpAdaptiveMaxDepth))
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let object = try JSONSerialization.jsonObject(with: encoder.encode(patch)) as? [String: Any]
        XCTAssertTrue(object?["mtp_adaptive_max_depth"] is NSNull)

        for depth in 3...8 {
            vm.mtpAdaptiveMaxDepth = String(depth)
            let settings = vm.currentSettingsDict()
            XCTAssertEqual(settings[ProfileSettingsKey.mtpAdaptiveMaxDepth]?.value as? Int, depth)
            XCTAssertNil(settings["mtp_fixed_depth"])
        }

        vm.mtpEnabled = false
        XCTAssertNil(vm.currentSettingsDict()[ProfileSettingsKey.mtpAdaptiveMaxDepth])
    }

    func testProfileFieldEditAutoSavesWithoutAProfile() async {
        let vm = ModelSettingsScreenVM()
        vm.autoSaveDelay = .zero
        // The closed port only wires the client; every request fails.
        let client = OMLXClient(host: "127.0.0.1", port: 9)
        await vm.load(modelID: "m", client: client)
        vm.resetWorkingBaseline()
        vm.lastError = nil
        let temperature = vm.bindProfile(Binding(
            get: { vm.temperature },
            set: { vm.temperature = $0 }
        ))

        // An invalid value is reported and never sent.
        temperature.wrappedValue = "-1"
        await vm.waitForAutoSave()
        guard case .failure(let invalid) = SamplingValidator.temperature("-1") else {
            return XCTFail("negative temperature must fail validation")
        }
        XCTAssertEqual(vm.lastError, invalid.message)

        // A valid value is written, so the closed port reports a transport error.
        temperature.wrappedValue = "0.7"
        await vm.waitForAutoSave()
        XCTAssertNotNil(vm.lastError)
        XCTAssertNotEqual(vm.lastError, invalid.message)
    }

    func testVlmMtpDraftModelOptionsIncludeQwenMtpConfigType() {
        let vm = ModelSettingsScreenVM()
        vm.modelID = "Qwopus3.6-35B-A3B-v1-4bit-MLXVLM-Target"
        vm.allModels = [
            makeModel(
                id: "Qwopus3.6-35B-A3B-v1-4bit-MLXVLM-Target",
                configModelType: "qwen3_5_moe"
            ),
            makeModel(
                id: "Qwopus3.6-35B-A3B-v1-4bit-MLXVLM-MTP-Drafter",
                configModelType: "qwen3_5_mtp"
            ),
            makeModel(id: "Qwen3.6-Regular-Model", configModelType: "qwen3_5_moe"),
        ]

        let values = vm.vlmMtpDraftModelOptions().map(\.0)

        XCTAssertTrue(values.contains("Qwopus3.6-35B-A3B-v1-4bit-MLXVLM-MTP-Drafter"))
        XCTAssertFalse(values.contains("Qwopus3.6-35B-A3B-v1-4bit-MLXVLM-Target"))
        XCTAssertFalse(values.contains("Qwen3.6-Regular-Model"))
    }

    func testVlmMtpDraftModelOptionsKeepAssistantAndStandaloneMtpFallbacks() {
        let vm = ModelSettingsScreenVM()
        vm.modelID = "target"
        vm.allModels = [
            makeModel(id: "gemma-assistant-draft", configModelType: nil),
            makeModel(id: "model-MTP-draft", configModelType: nil),
            makeModel(id: "model-MTPLX-runtime", configModelType: nil),
        ]

        let values = vm.vlmMtpDraftModelOptions().map(\.0)

        XCTAssertTrue(values.contains("gemma-assistant-draft"))
        XCTAssertTrue(values.contains("model-MTP-draft"))
        XCTAssertFalse(values.contains("model-MTPLX-runtime"))
    }

    func testQwenAneControlsUseMeasuredDefaults() {
        let vm = ModelSettingsScreenVM()

        XCTAssertFalse(vm.qwen35AnePrefillEnabled)
        XCTAssertEqual(vm.qwen35AnePrefillSequenceLength, "2048")
        XCTAssertEqual(vm.qwen35AnePrefillFraction, "0.53")
        XCTAssertEqual(vm.qwen35AnePrefillMaxLayers, "64")
        XCTAssertTrue(vm.qwen35AnePrefillDualAne)
        XCTAssertTrue(vm.qwen35AnePrefillGdn)
        XCTAssertEqual(vm.qwen35AnePrefillGdnFraction, "0.5")
        XCTAssertEqual(vm.qwen35AnePrefillGdnMaxLayers, "48")
        XCTAssertFalse(vm.qwen35AnePrefillCpuEnabled)
        XCTAssertEqual(vm.qwen35AnePrefillCpuFraction, "0.135")
        XCTAssertEqual(vm.qwen35AnePrefillCpuDownFraction, "0")
        XCTAssertEqual(vm.qwen35AnePrefillCpuThreads, "8")
        XCTAssertTrue(vm.qwen35AnePrefillCpuSharedResource)
    }

    func testQwenAneFractionFormatterPreservesSettingsValues() {
        XCTAssertEqual(ModelSettingsScreenVM.formatPct(0.5), "0.5")
        XCTAssertEqual(ModelSettingsScreenVM.formatPct(0.53), "0.53")
        XCTAssertEqual(ModelSettingsScreenVM.formatPct(0.527), "0.527")
    }

    func testQwenAneArbitraryInputValidation() {
        XCTAssertEqual(try? QwenAneSettingsValidator.promptBlock("2112").get(), 2112)
        XCTAssertThrowsError(try QwenAneSettingsValidator.promptBlock("2100").get())
        XCTAssertEqual(
            try? QwenAneSettingsValidator.tailPadding("1357", sequenceLength: "2048").get(),
            1357
        )
        XCTAssertThrowsError(
            try QwenAneSettingsValidator.tailPadding("2048", sequenceLength: "2048").get()
        )
        XCTAssertEqual(
            try? QwenAneSettingsValidator.mlpFraction("0.467", cpuFraction: "0.137").get(),
            0.467
        )
        XCTAssertThrowsError(
            try QwenAneSettingsValidator.mlpFraction("0.9", cpuFraction: "0.1").get()
        )
        XCTAssertEqual(
            try? QwenAneSettingsValidator.cpuFraction("0.137", mlpFraction: "0.467").get(),
            0.137
        )
        XCTAssertThrowsError(try QwenAneSettingsValidator.cpuThreads("8.5").get())
        XCTAssertThrowsError(try QwenAneSettingsValidator.cpuThreads("65").get())
        XCTAssertEqual(try? QwenAneSettingsValidator.gdnFraction("0.527").get(), 0.527)
        XCTAssertEqual(
            try? QwenAneSettingsValidator.cpuGdnFraction("0.047", gdnFraction: "0.527").get(),
            0.047
        )
        XCTAssertThrowsError(
            try QwenAneSettingsValidator.cpuGdnFraction("0.5", gdnFraction: "0.5").get()
        )
    }

    func testQwenAneSettingsAreIncludedInWorkingProfile() {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "qwen", configModelType: "qwen3_5")
        vm.qwen35AnePrefillEnabled = true
        vm.qwen35AnePrefillCpuEnabled = true

        let settings = vm.currentSettingsDict()

        XCTAssertEqual(settings["qwen35_ane_prefill_enabled"]?.value as? Bool, true)
        XCTAssertEqual(settings["qwen35_ane_prefill_sequence_length"]?.value as? Int, 2048)
        XCTAssertEqual(settings["qwen35_ane_prefill_fraction"]?.value as? Double, 0.53)
        XCTAssertEqual(settings["qwen35_ane_prefill_max_layers"]?.value as? Int, 64)
        XCTAssertEqual(settings["qwen35_ane_prefill_dual_ane"]?.value as? Bool, true)
        XCTAssertEqual(settings["qwen35_ane_prefill_gdn"]?.value as? Bool, true)
        XCTAssertEqual(settings["qwen35_ane_prefill_gdn_fraction"]?.value as? Double, 0.5)
        XCTAssertEqual(settings["qwen35_ane_prefill_gdn_max_layers"]?.value as? Int, 48)
        XCTAssertEqual(settings["qwen35_ane_prefill_cpu_enabled"]?.value as? Bool, true)
        XCTAssertEqual(settings["qwen35_ane_prefill_cpu_fraction"]?.value as? Double, 0.135)
        XCTAssertEqual(settings["qwen35_ane_prefill_cpu_down_fraction"]?.value as? Double, 0.0)
        XCTAssertEqual(settings["qwen35_ane_prefill_cpu_gdn_fraction"]?.value as? Double, 0.0)
        XCTAssertEqual(settings["qwen35_ane_prefill_cpu_threads"]?.value as? Int, 8)
        XCTAssertEqual(settings["qwen35_ane_prefill_cpu_shared_resource"]?.value as? Bool, true)
    }

    func testDisabledQwenAneSettingIsIncludedInWorkingProfile() {
        let settings = ModelSettingsScreenVM().currentSettingsDict()

        XCTAssertEqual(settings["qwen35_ane_prefill_enabled"]?.value as? Bool, false)
        XCTAssertNil(settings["qwen35_ane_prefill_sequence_length"])
    }

    func testQwenAneProfileBindingCreatesWorkingState() {
        let vm = ModelSettingsScreenVM()
        let binding = vm.bindProfile(Binding(
            get: { vm.qwen35AnePrefillEnabled },
            set: { vm.qwen35AnePrefillEnabled = $0 }
        ))

        binding.wrappedValue = true

        XCTAssertTrue(vm.qwen35AnePrefillEnabled)
        XCTAssertTrue(vm.profileDirty)
    }

    func testApplyingQwenAneTunerResultStagesWorkingProfile() {
        let vm = ModelSettingsScreenVM()
        vm.aneTuningStatus = ANETuningStatusResponse(
            tuningId: "tune-1",
            modelId: "qwen",
            status: "complete",
            phase: "complete",
            message: "Done",
            current: 1,
            total: 1,
            results: [],
            recommendation: ANETuningRecommendationDTO(
                enabled: true,
                mlpFraction: 0.467,
                gdnEnabled: true,
                gdnFraction: 0.527,
                cpuEnabled: nil,
                cpuFraction: nil,
                cpuDownFraction: nil,
                cpuGdnFraction: nil,
                fusedDown: true,
                cpuThreads: nil,
                cpuSharedResource: nil,
                processingTps: 123.4,
                speedupPercent: 12.3,
                sequenceLength: 2112,
                tailPaddingMinTokens: 1400
            ),
            error: nil,
            terminationReason: nil
        )

        // The tuner ran in whatever ANE mode the model had; applying its
        // result must not flip dual_ane and invalidate the measurement.
        vm.qwen35AnePrefillDualAne = false

        vm.applyANETuningRecommendation()

        XCTAssertTrue(vm.profileDirty)
        XCTAssertTrue(vm.qwen35AnePrefillEnabled)
        XCTAssertEqual(vm.qwen35AnePrefillSequenceLength, "2112")
        XCTAssertEqual(vm.qwen35AnePrefillTailPaddingMinTokens, "1400")
        XCTAssertEqual(vm.qwen35AnePrefillFraction, "0.467")
        XCTAssertTrue(vm.qwen35AnePrefillFusedDown)
        XCTAssertFalse(vm.qwen35AnePrefillDualAne)
        XCTAssertTrue(vm.qwen35AnePrefillGdn)
        XCTAssertEqual(vm.qwen35AnePrefillGdnFraction, "0.527")
    }

    func testK2AnePresetsKeepOneThirdAndExistingValues() {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "mova", configModelType: "k2_horizon")
        vm.qwen35AnePrefillEnabled = true
        let presets = ModelSettingsScreenVM.aneFractionOptions(current: "0.42", presets: [1.0 / 3.0, 0.5])
        XCTAssertEqual(presets.map(\.0), ["0.42", String(1.0 / 3.0), "0.5"])
        XCTAssertEqual(presets.map { normalizedPercent($0.1) }, ["42%", "33%", "50%"])
        vm.qwen35AnePrefillFraction = presets[0].0
        XCTAssertEqual(vm.currentSettingsDict()[ProfileSettingsKey.qwen35AnePrefillFraction]?.value as? Double, 0.42)
        vm.qwen35AnePrefillFraction = presets[1].0
        XCTAssertEqual(vm.currentSettingsDict()[ProfileSettingsKey.qwen35AnePrefillFraction]?.value as? Double, 1.0 / 3.0)
        XCTAssertEqual(ModelSettingsScreenVM.aneFractionOptions(current: vm.qwen35AnePrefillFraction, presets: [1.0 / 3.0, 0.5]).count, 2)
    }

    func testK2SharedPresetsKeepLoadedStringSelection() {
        for current in ["0", "0.0", "1", "1.0", String(1.0 / 3.0)] {
            let options = ModelSettingsScreenVM.aneFractionOptions(current: current, presets: [0, 1.0 / 3.0, 1])
            XCTAssertEqual(options.map { normalizedPercent($0.1) }, ["0%", "33%", "100%"])
            XCTAssertEqual(options.filter { $0.0 == current }.count, 1)
        }
    }

    func testMovaAneRecommendationUsesSharedProfileFields() throws {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "mova", configModelType: "k2_horizon")
        let data = Data(#"{"tuning_id":"k2","model_id":"mova","status":"completed","phase":"completed","message":"Done","current":1,"total":1,"results":[],"recommendation":{"backend":"k2","enabled":true,"mlp_fraction":0.3333333333333333,"shared_fraction":1,"gdn_enabled":false,"processing_tps":100,"speedup_percent":4,"sequence_length":2048}}"#.utf8)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        vm.aneTuningStatus = try decoder.decode(ANETuningStatusResponse.self, from: data)
        XCTAssertEqual(vm.aneTuningStatus?.recommendation?.processingTps, 100)
        XCTAssertEqual(vm.aneTuningStatus?.recommendation?.speedupPercent, 4)
        vm.applyANETuningRecommendation()
        XCTAssertTrue(vm.thinkingForced)
        XCTAssertTrue(vm.qwen35AnePrefillEnabled)
        XCTAssertTrue(vm.qwen35AnePrefillGdn)
        XCTAssertEqual(Double(vm.qwen35AnePrefillFraction), 1.0 / 3.0)
        XCTAssertEqual(Double(vm.qwen35AnePrefillSharedFraction), 1)
        XCTAssertTrue(vm.profileDirty)
        let fields = vm.currentSettingsDict()
        XCTAssertNotNil(fields[ProfileSettingsKey.qwen35AnePrefillEnabled])
    }

    func testControlsUseServerCapabilitiesForAnyModelFamily() throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let data = Data(#"{"id":"future","loaded":false,"is_loading":false,"estimated_size":0,"config_model_type":"future_model","thinking_forced":true,"reasoning_effort_options":["medium","high"],"reasoning_effort_default":"medium","reasoning_effort_custom":false,"ane_prefill_backend":"qwen","ane_prefill_mlp_fractions":[0.25]}"#.utf8)
        let vm = ModelSettingsScreenVM()
        vm.model = try decoder.decode(ModelDTO.self, from: data)
        XCTAssertTrue(vm.thinkingForced)
        XCTAssertTrue(vm.isQwen35AnePrefillModel)
        XCTAssertEqual(vm.reasoningEffortPresets, ["medium", "high"])
        XCTAssertEqual(vm.model?.anePrefillMlpFractions, [0.25])
        vm.addKwarg(.enableThinking)
        XCTAssertTrue(vm.chatTemplateEntries.isEmpty)
        vm.addKwarg(.reasoningEffort)
        XCTAssertEqual(vm.chatTemplateEntries.first?.value, "medium")
        XCTAssertNil(vm.currentSettingsDict()["enable_thinking"])
        vm.model?.anePrefillBackend = nil
        XCTAssertFalse(vm.isQwen35AnePrefillModel)
    }

    func testA8PrefillModelGateCoversExactFamilies() {
        let vm = ModelSettingsScreenVM()
        for type in ["qwen3_5", "qwen3_5_moe", "qwen3_6", "qwen3_8", "qwen4_exp", "Qwen4-Exp", "deepseek_v41",
                     "minimax_m3", "minimax_m3_vl"] {
            vm.model = makeModel(id: "m", configModelType: type)
            XCTAssertTrue(vm.isA8PrefillModel, type)
        }
        for type in ["qwen4", "qwen4_exp_x", "qwen3", "llama", "k2_horizon", "deepseek_v4", "minimax_m2"] {
            vm.model = makeModel(id: "m", configModelType: type)
            XCTAssertFalse(vm.isA8PrefillModel, type)
        }
        vm.model = makeModel(id: "m", configModelType: nil)
        XCTAssertFalse(vm.isA8PrefillModel)
    }

    func testQwen4SsdOffloadWireKeysAndCompatibility() throws {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "qwen4", configModelType: "qwen4_exp")
        XCTAssertTrue(vm.isQwen4Exp)

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let dto = try decoder.decode(
            ModelSettingsDTO.self,
            from: Data(#"{"qwen4_ple_ssd_offload":true}"#.utf8)
        )
        XCTAssertEqual(dto.qwen4PleSsdOffload, true)

        var patch = ModelSettingsPatch()
        patch.qwen4PleSsdOffload = true
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let object = try JSONSerialization.jsonObject(
            with: encoder.encode(patch)
        ) as? [String: Any]
        XCTAssertEqual(object?["qwen4_ple_ssd_offload"] as? Bool, true)
    }

    func testExternalEngineWireKeys() throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let dto = try decoder.decode(
            ModelSettingsDTO.self,
            from: Data(#"{"trust_remote_code":false,"external_engine":"splash"}"#.utf8)
        )
        XCTAssertEqual(dto.externalEngine, "splash")
        let option = try decoder.decode(
            ExternalEngineOption.self,
            from: Data(#"{"id":"splash","label":"Splash","compatible":false,"reason":"no draft"}"#.utf8)
        )
        XCTAssertEqual(option.reason, "no draft")

        // Turning the engine off must send JSON null, not omit the key.
        var patch = ModelSettingsPatch()
        patch.externalEngine = .some(nil)
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let object = try JSONSerialization.jsonObject(
            with: encoder.encode(patch)
        ) as? [String: Any]
        XCTAssertTrue(object?["external_engine"] is NSNull)
    }

    func testQwenAneSettingsDecodeFromServerAndEncodeForPatch() throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let json = #"""
        {
            "qwen35_ane_prefill_enabled": true,
            "qwen35_ane_prefill_sequence_length": 2048,
            "qwen35_ane_prefill_fraction": 0.53,
            "qwen35_ane_prefill_max_layers": 64,
            "qwen35_ane_prefill_dual_ane": true,
            "qwen35_ane_prefill_gdn": true,
            "qwen35_ane_prefill_gdn_fraction": 0.5,
            "qwen35_ane_prefill_gdn_max_layers": 48,
            "qwen35_ane_prefill_cpu_enabled": true,
            "qwen35_ane_prefill_cpu_fraction": 0.135,
            "qwen35_ane_prefill_cpu_down_fraction": 0.2,
            "qwen35_ane_prefill_cpu_gdn_fraction": 0.05,
            "qwen35_ane_prefill_cpu_threads": 8,
            "qwen35_ane_prefill_cpu_shared_resource": true
        }
        """#
        let dto = try decoder.decode(ModelSettingsDTO.self, from: Data(json.utf8))
        XCTAssertEqual(dto.qwen35AnePrefillFraction, 0.53)
        XCTAssertEqual(dto.qwen35AnePrefillGdnFraction, 0.5)
        XCTAssertEqual(dto.qwen35AnePrefillCpuEnabled, true)
        XCTAssertEqual(dto.qwen35AnePrefillCpuFraction, 0.135)
        XCTAssertEqual(dto.qwen35AnePrefillCpuDownFraction, 0.2)
        XCTAssertEqual(dto.qwen35AnePrefillCpuGdnFraction, 0.05)
        XCTAssertEqual(dto.qwen35AnePrefillCpuThreads, 8)
        XCTAssertEqual(dto.qwen35AnePrefillCpuSharedResource, true)

        var patch = ModelSettingsPatch()
        patch.qwen35AnePrefillEnabled = true
        patch.qwen35AnePrefillFraction = 0.53
        patch.qwen35AnePrefillCpuEnabled = true
        patch.qwen35AnePrefillCpuFraction = 0.135
        patch.qwen35AnePrefillCpuDownFraction = 0.2
        patch.qwen35AnePrefillCpuGdnFraction = 0.05
        patch.qwen35AnePrefillCpuThreads = 8
        patch.qwen35AnePrefillCpuSharedResource = true
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let object = try JSONSerialization.jsonObject(with: encoder.encode(patch)) as? [String: Any]
        XCTAssertEqual(object?["qwen35_ane_prefill_enabled"] as? Bool, true)
        XCTAssertEqual(object?["qwen35_ane_prefill_fraction"] as? Double, 0.53)
        XCTAssertEqual(object?["qwen35_ane_prefill_cpu_enabled"] as? Bool, true)
        XCTAssertEqual(object?["qwen35_ane_prefill_cpu_fraction"] as? Double, 0.135)
        XCTAssertEqual(object?["qwen35_ane_prefill_cpu_down_fraction"] as? Double, 0.2)
        XCTAssertEqual(object?["qwen35_ane_prefill_cpu_gdn_fraction"] as? Double, 0.05)
        XCTAssertEqual(object?["qwen35_ane_prefill_cpu_threads"] as? Int, 8)
        XCTAssertEqual(object?["qwen35_ane_prefill_cpu_shared_resource"] as? Bool, true)
    }

    func testANETunerOverridesEncodeForStartRequest() throws {
        let request = ANETuningStartRequest(
            modelId: "qwen",
            sequenceLength: 2048,
            repeats: 2,
            allowCpu: false,
            allowCpuGate: false,
            allowCpuDown: true,
            allowAneGdn: false,
            allowCpuGdn: false,
            allowCpuSharedResource: false
        )
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase

        let data = try encoder.encode(request)
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        XCTAssertEqual(object?["allow_cpu"] as? Bool, false)
        XCTAssertEqual(object?["allow_cpu_gate"] as? Bool, false)
        XCTAssertEqual(object?["allow_cpu_down"] as? Bool, true)
        XCTAssertEqual(object?["allow_ane_gdn"] as? Bool, false)
        XCTAssertEqual(object?["allow_cpu_gdn"] as? Bool, false)
        XCTAssertEqual(object?["allow_cpu_shared_resource"] as? Bool, false)
    }

    func testK2UsesExistingReasoningEffortAndBudgetSettings() {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "mova", configModelType: "k2_horizon")
        XCTAssertEqual(vm.reasoningEffortPresets, ["low", "medium", "high"])
        vm.addKwarg(.enableThinking)
        XCTAssertTrue(vm.chatTemplateEntries.isEmpty)
        vm.addKwarg(.reasoningEffort)
        XCTAssertEqual(vm.chatTemplateEntries.first?.value, "high")
        vm.thinkingBudgetEnabled = true
        vm.thinkingBudgetTokens = "1"
        let settings = vm.currentSettingsDict()
        XCTAssertNil(settings["enable_thinking"])
        XCTAssertEqual(settings["thinking_budget_tokens"]?.value as? Int, 1)
        let kwargs = settings["chat_template_kwargs"]?.value as? [String: AnyCodable]
        XCTAssertEqual(kwargs?["reasoning_effort"]?.value as? String, "high")
        vm.model = makeModel(id: "qwen", configModelType: "qwen3")
        XCTAssertTrue(vm.reasoningEffortPresets.contains("xhigh"))
        XCTAssertNotNil(vm.currentSettingsDict()["enable_thinking"])
    }




    func testThinkingPatchCanClearK2SettingWithoutChangingOtherWireValues() throws {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let cases: [(Bool??, String)] = [
            (nil, "{}"),
            (.some(nil), #"{"enable_thinking":null}"#),
            (true, #"{"enable_thinking":true}"#),
            (false, #"{"enable_thinking":false}"#),
        ]
        for (value, expected) in cases {
            var patch = ModelSettingsPatch()
            patch.enableThinking = value
            XCTAssertEqual(String(decoding: try encoder.encode(patch), as: UTF8.self), expected)
        }
    }

    func testSharedAneSettingsPreserveSplitAndExcludeOtherBackendOptions() {
        let vm = ModelSettingsScreenVM()
        vm.qwen35AnePrefillEnabled = true
        vm.qwen35AnePrefillFraction = String(1.0 / 3.0)
        vm.qwen35AnePrefillSharedFraction = "0"
        vm.qwen35AnePrefillCpuEnabled = true
        vm.model = makeModel(id: "k2", configModelType: "k2_horizon")
        let k2 = vm.currentSettingsDict()
        XCTAssertEqual(k2["qwen35_ane_prefill_enabled"]?.value as? Bool, true)
        XCTAssertEqual(k2["qwen35_ane_prefill_fraction"]?.value as? Double, 1.0 / 3.0)
        XCTAssertEqual(k2["qwen35_ane_prefill_shared_fraction"]?.value as? Double, 0)
        XCTAssertNil(k2["qwen35_ane_prefill_cpu_enabled"])
        XCTAssertFalse(k2.keys.contains { $0.hasPrefix("k2_ane_") })
        vm.model = makeModel(id: "qwen", configModelType: "qwen3_5")
        let qwen = vm.currentSettingsDict()
        XCTAssertEqual(qwen["qwen35_ane_prefill_enabled"]?.value as? Bool, true)
        XCTAssertEqual(qwen["qwen35_ane_prefill_cpu_enabled"]?.value as? Bool, true)
        XCTAssertNil(qwen["qwen35_ane_prefill_shared_fraction"])
    }

    func testMtplxImportIsOfferedOnlyForTheServerSideSidecarMarker() {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(
            id: "Qwen3.8-27B-MTPLX-Optimized-Speed",
            configModelType: "qwen3_5",
            mtpCompatible: false,
            mtpCompatibilityReason: "MTPLX side-car detected but not imported. Import it to merge the MTP head into the checkpoint index."
        )
        XCTAssertTrue(vm.canImportMtplxSidecar)

        vm.model = makeModel(
            id: "missing-mtp",
            configModelType: "qwen3_5",
            mtpCompatible: false,
            mtpCompatibilityReason: "Config declares MTP layers but the weight files contain neither mtp.* tensors nor native nextn layers."
        )
        XCTAssertFalse(vm.canImportMtplxSidecar)

        vm.model = makeModel(
            id: "already-compatible",
            configModelType: "qwen3_5",
            mtpCompatible: true,
            mtpCompatibilityReason: "MTPLX side-car detected but not imported."
        )
        XCTAssertFalse(vm.canImportMtplxSidecar)
    }

    func testMtplxImportEndpointUsesModelScopedAdminRoute() {
        XCTAssertEqual(
            AdminAPI.importMtplx("Qwen3.8-27B-MTPLX-Optimized-Speed"),
            "/admin/api/models/Qwen3.8-27B-MTPLX-Optimized-Speed/import-mtplx"
        )
    }

    func testPagesFollowModelTypeAndSummarizeEnabledFeatures() {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "llm", configModelType: "qwen3_5")
        XCTAssertTrue(vm.showsGenerationSection)
        XCTAssertTrue(vm.showsEnginePages)

        XCTAssertFalse(vm.showsEmbeddingContextWindow)
        XCTAssertFalse(vm.showsRerankerContextWindow)

        vm.model = makeModel(id: "embed", configModelType: "bert", modelType: "embedding")
        XCTAssertFalse(vm.showsGenerationSection)
        XCTAssertFalse(vm.showsEnginePages)
        XCTAssertTrue(vm.showsEmbeddingContextWindow)
        XCTAssertFalse(vm.showsRerankerContextWindow)

        vm.model = makeModel(id: "rerank", configModelType: "xlm-roberta", modelType: "reranker")
        XCTAssertFalse(vm.showsEmbeddingContextWindow)
        XCTAssertTrue(vm.showsRerankerContextWindow)

        vm.model = makeModel(id: "diffusion", configModelType: "diffusion_gemma")
        XCTAssertTrue(vm.showsGenerationSection)
        XCTAssertFalse(vm.showsEnginePages)

        vm.model = makeModel(id: "llm", configModelType: "qwen3_5")
        vm.mtpEnabled = true
        vm.dflashEnabled = true
        vm.vlmMtpEnabled = true
        // CED prefill only counts on the family that shows its row.
        vm.deepseekV41CedPrefillEnabled = true
        XCTAssertEqual(vm.enabledFeatures(on: .acceleration).count, 3)
        XCTAssertEqual(ModelSettingsScreenVM.summary(["A", "B", "C"], empty: "Off"), "A, B +1")
        XCTAssertEqual(ModelSettingsScreenVM.summary(["A"], empty: "Off"), "A")
        XCTAssertEqual(ModelSettingsScreenVM.summary([], empty: "Off"), "Off")

        let summary = vm.summary(for: .acceleration)
        vm.externalEngine = "splash"
        XCTAssertNotEqual(vm.summary(for: .acceleration), summary)
    }

    func testSavingEmptyAliasAndModelTypeSendsExplicitClears() async {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [RejectingSettingsURLProtocol.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let client = OMLXClient(host: "127.0.0.1", port: 18160, session: session)
        let vm = ModelSettingsScreenVM()
        vm.modelID = "clear"

        await vm.save(.alias, client: client)
        await vm.save(.modelType, client: client)

        XCTAssertNil(vm.lastError)
    }

    func testRejectedAutoSaveRestoresTheStoredValues() async {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [RejectingSettingsURLProtocol.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let client = OMLXClient(host: "127.0.0.1", port: 18160, session: session)
        let vm = ModelSettingsScreenVM()
        vm.autoSaveDelay = .zero
        await vm.load(modelID: "m", client: client)
        XCTAssertNil(vm.lastError)
        XCTAssertTrue(vm.mtpEnabled)

        let dflash = vm.bindProfile(Binding(
            get: { vm.dflashEnabled },
            set: { vm.dflashEnabled = $0 }
        ))
        dflash.wrappedValue = true
        await vm.waitForAutoSave()

        XCTAssertFalse(vm.dflashEnabled)
        XCTAssertFalse(vm.profileDirty)
        XCTAssertTrue(vm.lastError?.contains("cannot both be enabled") == true)
    }

    func testLightningMtpAndDflashLockEachOther() {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "qwen", configModelType: "qwen3_5")
        vm.mtpEnabled = true
        XCTAssertNotNil(vm.dflashConflictReason)
        XCTAssertNil(vm.mtpConflictReason)

        vm.mtpEnabled = false
        vm.dflashEnabled = true
        XCTAssertNotNil(vm.mtpConflictReason)
        XCTAssertNil(vm.dflashConflictReason)
    }

    func testMoeExpertOffloadLocksSpeculativeDecodersUnlessMtpHeadStaysResident() {
        let vm = ModelSettingsScreenVM()
        var model = makeModel(id: "moe", configModelType: "qwen3_moe")
        model.moeExpertOffloadSupported = true
        vm.model = model
        vm.moeExpertOffloadEnabled = true
        XCTAssertNotNil(vm.mtpConflictReason)
        XCTAssertNotNil(vm.dflashConflictReason)
        XCTAssertNotNil(vm.vlmMtpConflictReason)

        vm.model?.moeOffloadAllowsMtp = true
        XCTAssertNil(vm.mtpConflictReason)
        XCTAssertNotNil(vm.dflashConflictReason)

        vm.moeExpertOffloadEnabled = false
        vm.mtpEnabled = true
        XCTAssertNil(vm.moeExpertOffloadConflictReason)
        vm.model?.moeOffloadAllowsMtp = false
        XCTAssertNotNil(vm.moeExpertOffloadConflictReason)
    }

    func testMoeResidentPercentIsSavedAsFractionAndValidated() async {
        let vm = ModelSettingsScreenVM()
        var model = makeModel(id: "moe", configModelType: "qwen3_moe")
        model.moeExpertOffloadSupported = true
        vm.model = model
        vm.resetWorkingBaseline()
        vm.moeExpertOffloadEnabled = true
        vm.moeExpertOffloadResidentPercent = "30"
        vm.refreshProfileDirty()

        let settings = vm.currentSettingsDict()
        XCTAssertEqual(settings[ProfileSettingsKey.moeExpertOffloadEnabled]?.value as? Bool, true)
        XCTAssertEqual(settings[ProfileSettingsKey.moeExpertOffloadResidentFraction]?.value as? Double, 0.3)

        // An out-of-range edit stops the profile write before any request.
        vm.moeExpertOffloadResidentPercent = "2"
        XCTAssertNil(vm.moeExpertOffloadResidentFraction)
        await vm.saveWorkingAs(scope: .model, name: "moe",
                               client: OMLXClient(host: "127.0.0.1", port: 9))
        XCTAssertEqual(vm.lastError, vm.moeResidentRangeError)
    }

    func testGuidedGrammarIsUniversalProfileSettingAndBlocksVlmMtp() {
        let vm = ModelSettingsScreenVM()
        vm.resetWorkingBaseline()
        vm.guidedGrammarEnabled = true
        vm.guidedGrammar = #"root ::= "YES""#
        vm.refreshProfileDirty()

        // Applying a profile resets missing universal keys, so the grammar
        // must travel with every profile write.
        let settings = vm.currentSettingsDict()
        XCTAssertEqual(settings[ProfileSettingsKey.guidedGrammarEnabled]?.value as? Bool, true)
        XCTAssertEqual(settings[ProfileSettingsKey.guidedGrammar]?.value as? String, #"root ::= "YES""#)
        XCTAssertTrue(ProfileSettingsKey.templateKeys.contains(ProfileSettingsKey.guidedGrammar))

        XCTAssertTrue(vm.vlmMtpProcessorConflict)
        XCTAssertNotNil(vm.vlmMtpConflictReason)
    }

    func testForcedSsdNgramOffloadKeepsTheStoredChoice() async throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let model = try decoder.decode(ModelDTO.self, from: Data(#"{"id":"dsv41","loaded":false,"is_loading":false,"estimated_size":0,"config_model_type":"deepseek_v41","deepseek_v41_engram_ssd_offload_supported":true,"deepseek_v41_engram_ssd_offload_forced":true,"moe_expert_offload_supported":true,"moe_offload_allows_mtp":true}"#.utf8))
        XCTAssertEqual(model.moeOffloadAllowsMtp, true)

        let vm = ModelSettingsScreenVM()
        vm.model = model
        vm.deepseekV41EngramSsdOffloadSupported = true
        vm.deepseekV41EngramSsdOffloadForced = true
        XCTAssertTrue(vm.ssdNgramOffloadSupported)
        XCTAssertTrue(vm.ssdNgramOffloadForced)
        vm.ssdNgramOffload = true
        XCTAssertTrue(vm.deepseekV41EngramSsdOffload)

        // Forced: nothing is sent, so the closed port never reports an error.
        let client = OMLXClient(host: "127.0.0.1", port: 9)
        await vm.save(.ssdNgramOffload, client: client)
        XCTAssertNil(vm.lastError)
        vm.deepseekV41EngramSsdOffloadForced = false
        await vm.save(.ssdNgramOffload, client: client)
        XCTAssertNotNil(vm.lastError)

        var patch = ModelSettingsPatch()
        patch.deepseekV41EngramSsdOffload = true
        patch.deepseekV41CedPrefillEnabled = true
        patch.qwen4GdnDecodeWideProj = true
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let object = try JSONSerialization.jsonObject(with: encoder.encode(patch)) as? [String: Any]
        XCTAssertEqual(object?["deepseek_v41_engram_ssd_offload"] as? Bool, true)
        XCTAssertEqual(object?["deepseek_v41_ced_prefill_enabled"] as? Bool, true)
        XCTAssertEqual(object?["qwen4_gdn_decode_wide_proj"] as? Bool, true)
    }

    func testUndoingAnEditClearsTheUnsavedState() {
        let vm = ModelSettingsScreenVM()
        vm.resetWorkingBaseline()
        let binding = vm.bindProfile(Binding(
            get: { vm.mtpEnabled },
            set: { vm.mtpEnabled = $0 }
        ))
        binding.wrappedValue = true
        XCTAssertTrue(vm.profileDirty)
        binding.wrappedValue = false
        XCTAssertFalse(vm.profileDirty)
    }

    func testDirectSaveSendsNullForClearedFields() throws {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "m", configModelType: "qwen3_5")
        vm.temperature = "0.6"

        let body = vm.workingSettingsBody()
        XCTAssertEqual(body[ProfileSettingsKey.temperature]?.value as? Double, 0.6)
        XCTAssertEqual(body[ProfileSettingsKey.indexCacheFreq]?.value as? Int, 0)
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let object = try JSONSerialization.jsonObject(with: encoder.encode(body)) as? [String: Any]
        XCTAssertTrue(object?["top_p"] is NSNull)
        XCTAssertTrue(object?["reasoning_parser"] is NSNull)
        XCTAssertNil(object?["mtp_fixed_depth"])
        XCTAssertEqual(object?["qwen35_ane_prefill_enabled"] as? Bool, false)
    }

    func testMivProfileEditPreservesFixedDepthUntilAdaptiveDepthChanges() {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "m", configModelType: "qwen3_5")
        vm.mtpEnabled = true
        let baseline = vm.currentSettingsDict()

        vm.temperature = "0.6"
        let unrelatedEdit = vm.workingSettingsBody(changesSince: baseline)
        XCTAssertEqual(unrelatedEdit.keys.sorted(), [ProfileSettingsKey.temperature])
        XCTAssertNil(unrelatedEdit["mtp_fixed_depth"])

        vm.mtpAdaptiveMaxDepth = "4"
        let adaptiveEdit = vm.workingSettingsBody(changesSince: baseline)
        XCTAssertTrue(adaptiveEdit["mtp_fixed_depth"]?.value is NSNull)
        XCTAssertEqual(adaptiveEdit[ProfileSettingsKey.mtpAdaptiveMaxDepth]?.value as? Int, 4)
    }

    func testLoRAAdapterPickerAndWorkingBody() throws {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "m", configModelType: "qwen3_5")
        vm.loraAdapters = [
            LoRAAdapterDTO(id: "fits", path: "/models/fits", baseModel: "Qwen/Base", rank: 32,
                           targets: ["q_proj"], sizeFormatted: "638 MB", problem: nil),
            LoRAAdapterDTO(id: "misfit", path: "/models/misfit", baseModel: nil, rank: 8,
                           targets: ["q_proj"], sizeFormatted: nil, problem: "too many layers"),
        ]
        XCTAssertTrue(vm.workingSettingsBody()[ProfileSettingsKey.loraAdapter]?.value is NSNull)

        // A saved adapter stays selectable even when it no longer fits.
        vm.loraAdapter = "/models/misfit"
        XCTAssertEqual(vm.loraAdapterOptions().map(\.0), ["", "/models/fits", "/models/misfit"])
        vm.loraAdapter = "/elsewhere/adapter"
        XCTAssertEqual(vm.loraAdapterOptions().map(\.0), ["", "/elsewhere/adapter", "/models/fits"])
        XCTAssertEqual(
            vm.workingSettingsBody()[ProfileSettingsKey.loraAdapter]?.value as? String,
            "/elsewhere/adapter"
        )
    }

    func testGPUPowerLimitSendsZeroWhenOff() throws {
        let vm = ModelSettingsScreenVM()
        XCTAssertEqual(vm.currentSettingsDict()[ProfileSettingsKey.gpuPowerLimitW]?.value as? Int, 0)
        vm.limitGpuPower = true
        vm.gpuPowerLimitWatts = "12"
        XCTAssertEqual(vm.currentSettingsDict()[ProfileSettingsKey.gpuPowerLimitW]?.value as? Int, 12)

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        vm.model = try decoder.decode(ModelDTO.self, from: Data(#"{"id":"q","loaded":false,"is_loading":false,"estimated_size":0,"gpu_power_limit_supported":true,"settings":{"gpu_power_limit_w":20}}"#.utf8))
        XCTAssertTrue(vm.gpuPowerLimitSupported)
        XCTAssertEqual(vm.model?.settings?.gpuPowerLimitW, 20)
    }

    func testQwenInt8RowNeedsSupportedHardwareUnlessAlreadyOn() throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        func model(_ extra: String) throws -> ModelDTO {
            try decoder.decode(ModelDTO.self, from: Data(#"{"id":"q","loaded":false,"is_loading":false,"estimated_size":0,"config_model_type":"qwen3_5"\#(extra)}"#.utf8))
        }
        let vm = ModelSettingsScreenVM()
        vm.model = try model(#","a8_prefill_supported":false"#)
        XCTAssertTrue(vm.isA8PrefillModel)
        XCTAssertFalse(vm.showsA8Prefill)
        vm.model = try model(#","a8_prefill_supported":true"#)
        XCTAssertTrue(vm.showsA8Prefill)
        vm.model = try model(#","a8_prefill_supported":false,"settings":{"a8_prefill_enabled":true}"#)
        XCTAssertTrue(vm.showsA8Prefill)

        // A Prefill Accelerator Mac shows the row where the kernels cannot run here.
        vm.model = try model(#","a8_prefill_supported":false"#)
        vm.prefillHelperNodeID = "m5"
        XCTAssertTrue(vm.showsA8Prefill)
        vm.prefillHelperNodeID = ""
        XCTAssertFalse(vm.showsA8Prefill)
    }

    func testLoadDefaultsStagesGenerationFieldsOnly() {
        let vm = ModelSettingsScreenVM()
        vm.maxTokens = "512"
        vm.minP = "0.1"
        vm.forceSampling = true
        vm.ttlSeconds = "600"
        vm.mtpEnabled = true
        vm.resetWorkingBaseline()

        let defaults = GenerationConfigDTO(
            maxContextWindow: 32768, temperature: 0.6, topP: 0.95, topK: 20
        )
        vm.applyGenerationDefaults(defaults)
        XCTAssertEqual(vm.contextLength, "32768")
        XCTAssertEqual(vm.temperature, "0.6")
        XCTAssertEqual(vm.topP, "0.95")
        XCTAssertEqual(vm.topK, "20")
        XCTAssertEqual(vm.repetitionPenalty, "")
        XCTAssertEqual(vm.maxTokens, "")
        XCTAssertEqual(vm.minP, "")
        XCTAssertFalse(vm.forceSampling)
        XCTAssertEqual(vm.ttlSeconds, "600")
        XCTAssertTrue(vm.mtpEnabled)
        XCTAssertTrue(vm.profileDirty)

        vm.resetWorkingBaseline()
        vm.applyGenerationDefaults(defaults)
        XCTAssertFalse(vm.profileDirty)
    }

    private func makeModel(
        id: String,
        configModelType: String?,
        modelType: String? = nil,
        mtpCompatible: Bool? = nil,
        mtpCompatibilityReason: String? = nil
    ) -> ModelDTO {
        var model = ModelDTO(
            id: id,
            displayName: nil,
            modelPath: nil,
            loaded: false,
            isLoading: false,
            estimatedSize: 0,
            estimatedSizeFormatted: nil,
            actualSize: nil,
            actualSizeFormatted: nil,
            pinned: nil,
            isDefault: nil,
            isFavorite: nil,
            engineType: nil,
            modelType: modelType,
            configModelType: configModelType,
            modelContextLength: nil,
            thinkingDefault: nil,
            dflashCompatible: nil,
            dflashCompatibilityReason: nil,
            mtpCompatible: mtpCompatible,
            mtpCompatibilityReason: mtpCompatibilityReason,
            qwen4PleSsdOffloadSupported: nil,
            qwen4PleSsdOffloadForced: nil,
            qwen4PleResidentBytes: nil,
            qwen4PleMmapBytes: nil,
            virtual: nil,
            settings: nil
        )
        model.thinkingForced = configModelType == "k2_horizon"
        model.reasoningEffortOptions = model.thinkingForced == true ? ["low", "medium", "high"] : ["low", "medium", "high", "xhigh", "max"]
        model.reasoningEffortDefault = model.thinkingForced == true ? "high" : "low"
        model.reasoningEffortCustom = model.thinkingForced != true
        model.anePrefillBackend = configModelType == "k2_horizon" ? "k2" : (configModelType?.hasPrefix("qwen3_5") == true ? "qwen" : nil)
        model.anePrefillDefaultFraction = model.anePrefillBackend == "k2" ? 1.0 / 3.0 : 0.53
        model.anePrefillMlpFractions = [1.0 / 3.0, 0.5]
        model.anePrefillSharedFractions = [0, 1.0 / 3.0, 1]
        return model
    }

    func testAutoRoutedExternalEngineStockOverrideAndRestore() {
        var model = makeModel(id: "Qwen3.6-35B-A3B", configModelType: "qwen3_5_moe")
        model.autoExternalEngine = "m5rt"
        model.activeEngineId = "m5rt"
        model.activeEngineLabel = "miv Core Backend"
        model.externalEngines = [
            ExternalEngineOption(
                id: "m5rt",
                label: "miv Core Backend",
                compatible: true,
                reason: "",
                supportedStandardSettings: ["mtp_enabled", "dflash_enabled", "turboquant_kv_enabled"]
            ),
            ExternalEngineOption(
                id: "llama-metal",
                label: "Llama Metal 4 (Path B)",
                compatible: true,
                reason: ""
            ),
        ]

        let vm = ModelSettingsScreenVM()
        vm.model = model
        vm.externalEngine = ""

        // Auto-routed state: effectiveExternalEngine is active even though externalEngine is empty.
        XCTAssertEqual(vm.effectiveExternalEngine, "m5rt")
        XCTAssertTrue(vm.isAutoRoutedExternalEngine)
        XCTAssertFalse(vm.isStockOverride)
        XCTAssertEqual(vm.headerEngineId, "m5rt")

        // Turning OFF an adapted model writes "stock" (not "") so Stock MLX override sticks.
        vm.setExternalEngineEnabled(false, usableEngine: "m5rt")
        XCTAssertEqual(vm.externalEngine, "stock")
        XCTAssertEqual(vm.effectiveExternalEngine, "")
        XCTAssertTrue(vm.isStockOverride)
        XCTAssertEqual(vm.headerEngineId, "mlx")

        // Turning ON again restores "" (auto-route) instead of hard-pinning.
        vm.setExternalEngineEnabled(true, usableEngine: "m5rt")
        XCTAssertEqual(vm.externalEngine, "")
        XCTAssertEqual(vm.effectiveExternalEngine, "m5rt")
        XCTAssertTrue(vm.isAutoRoutedExternalEngine)
        XCTAssertFalse(vm.isStockOverride)

        // Switching to a Path B provider pins it cleanly.
        vm.externalEngine = "llama-metal"
        XCTAssertEqual(vm.effectiveExternalEngine, "llama-metal")
        XCTAssertFalse(vm.isAutoRoutedExternalEngine)
        XCTAssertEqual(vm.headerEngineId, "llama-metal")
    }

    func testMivFormatBadgesAutoRepackAndPureMivDetection() throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        // 1) Downloaded quantized .safetensors model pending auto-repack on first Load
        let convertibleJSON = Data(#"""
        {
          "id": "Qwen3.5-9B-4bit",
          "loaded": false,
          "is_loading": false,
          "estimated_size": 1000,
          "config_model_type": "qwen3_5",
          "weight_format": "safetensors",
          "has_miv_weights": false,
          "pending_auto_repack": true,
          "repack_target_format": "miv",
          "auto_external_engine": "m5rt",
          "external_engines": [
            {
              "id": "m5rt",
              "label": "miv Core Backend",
              "compatible": true,
              "can_convert": true,
              "reason": ""
            }
          ]
        }
        """#.utf8)
        let convertibleModel = try decoder.decode(ModelDTO.self, from: convertibleJSON)
        let vm = ModelSettingsScreenVM()
        vm.model = convertibleModel
        vm.modelID = convertibleModel.id
        XCTAssertTrue(vm.canConvertMiv)
        XCTAssertTrue(vm.canConvertOMLXM5)
        XCTAssertFalse(convertibleModel.isMivOnly)
        XCTAssertEqual(convertibleModel.weightFormatBadgeLabel, "safetensors → .miv")
        XCTAssertEqual(convertibleModel.resolvedEngineId, "m5rt")

        // 2) Repacked .miv model co-located alongside original .safetensors
        let repackedJSON = Data(#"""
        {
          "id": "Qwen3.8-27B-4bit",
          "loaded": true,
          "is_loading": false,
          "estimated_size": 16000,
          "config_model_type": "qwen3_8",
          "weight_format": "miv",
          "has_miv_weights": true,
          "pending_auto_repack": false,
          "auto_external_engine": "m5rt"
        }
        """#.utf8)
        let repackedModel = try decoder.decode(ModelDTO.self, from: repackedJSON)
        XCTAssertFalse(repackedModel.isMivOnly)
        XCTAssertEqual(repackedModel.weightFormatBadgeLabel, ".miv")
        XCTAssertEqual(repackedModel.resolvedEngineId, "m5rt")

        // 3) Pure .miv model (and legacy .omlxm5 compatibility)
        let pureJSON = Data(#"""
        {
          "id": "Qwen3.8-PureMiv",
          "loaded": false,
          "is_loading": false,
          "estimated_size": 2000,
          "config_model_type": "qwen3_8",
          "model_format": "miv",
          "weight_format": "miv",
          "has_miv_weights": true,
          "auto_external_engine": "m5rt"
        }
        """#.utf8)
        let pureModel = try decoder.decode(ModelDTO.self, from: pureJSON)
        XCTAssertTrue(pureModel.isMivOnly)
        XCTAssertTrue(pureModel.isOMLXM5Only)
        XCTAssertEqual(pureModel.weightFormatBadgeLabel, ".miv")
        XCTAssertEqual(pureModel.resolvedEngineId, "m5rt")
    }

    func testExternalEngineLoRACapabilityAndLoraModeRoundTrip() {
        var model = makeModel(id: "Qwen3.6-LoRA", configModelType: "qwen3_5")
        model.autoExternalEngine = "m5rt"
        model.externalEngines = [
            ExternalEngineOption(
                id: "splash",
                label: "Splash",
                compatible: true,
                reason: ""
            ),
            ExternalEngineOption(
                id: "m5rt",
                label: "miv Core Backend",
                compatible: true,
                reason: "",
                supportedStandardSettings: ["max_context_window", "lora_adapter"],
                providerId: "io.ninfer.metal",
                optionsSchema: [["key": AnyCodable("lora_mode"), "type": AnyCodable("enum")]]
            ),
        ]

        let vm = ModelSettingsScreenVM()
        vm.model = model
        vm.externalEngine = ""
        vm.externalEngineOptions = ["ninfer-metal": ["lora_mode": AnyCodable("bypass")]]

        XCTAssertEqual(vm.effectiveExternalEngine, "m5rt")
        XCTAssertTrue(vm.externalEngineSupports("lora_adapter"))
        XCTAssertTrue(vm.showsExternalEngineLoraMode)
        XCTAssertEqual(vm.externalEngineLoraMode, "bypass")

        vm.externalEngineLoraMode = "merge"
        XCTAssertEqual(vm.externalEngineLoraMode, "merge")
        XCTAssertEqual(vm.externalEngineOptions["m5rt"]?["lora_mode"]?.value as? String, "merge")
        XCTAssertEqual(vm.externalEngineOptions["ninfer-metal"]?["lora_mode"]?.value as? String, "merge")

        vm.externalEngine = "splash"
        XCTAssertFalse(vm.externalEngineSupports("lora_adapter"))
        XCTAssertFalse(vm.showsExternalEngineLoraMode)
    }

    func testExternalEngineGeneralOptionsExposeConcurrencyAndSamplingContract() {
        let vm = ModelSettingsScreenVM()
        let m5Opt = ExternalEngineOption(
            id: "m5rt",
            label: "miv Core Backend (ninfer-metal)",
            compatible: true,
            reason: nil,
            supportedStandardSettings: ["max_context_window", "dflash_enabled", "lora_adapter"],
            providerId: "io.ninfer.metal",
            optionsSchema: [
                [
                    "key": AnyCodable("max_concurrency"),
                    "label": AnyCodable("Concurrency (B/C)"),
                    "type": AnyCodable("int"),
                    "min": AnyCodable(1),
                    "max": AnyCodable(8),
                    "default": AnyCodable(1),
                ],
                [
                    "key": AnyCodable("rolling_prefix_cache"),
                    "label": AnyCodable("Rolling Prefix Cache"),
                    "type": AnyCodable("bool"),
                    "default": AnyCodable(true),
                ],
                [
                    "key": AnyCodable("lora_mode"),
                    "label": AnyCodable("LoRA Mode"),
                    "type": AnyCodable("enum"),
                    "options": AnyCodable([AnyCodable("merge"), AnyCodable("bypass")]),
                    "default": AnyCodable("merge"),
                ],
                [
                    "key": AnyCodable("sampling_contract"),
                    "label": AnyCodable("Sampling Contract"),
                    "type": AnyCodable("enum"),
                    "options": AnyCodable([AnyCodable("position_coupled_v1"), AnyCodable("legacy_rejection_v1")]),
                    "default": AnyCodable("position_coupled_v1"),
                ],
            ]
        )
        var model = makeModel(id: "Qwen3.8-27B-mxfp4", configModelType: "qwen3_5")
        model.externalEngines = [m5Opt]
        model.autoExternalEngine = "m5rt"
        vm.model = model
        vm.externalEngine = ""
        vm.externalEngineOptions = ["m5rt": ["max_concurrency": AnyCodable(4), "sampling_contract": AnyCodable("legacy_rejection_v1")]]

        let general = vm.externalEngineGeneralOptions
        XCTAssertEqual(general.map(\.key), ["max_concurrency", "rolling_prefix_cache", "sampling_contract"])
        XCTAssertEqual(vm.externalEngineIntOption(general[0]), 4)
        XCTAssertEqual(vm.externalEngineBoolOption(general[1]), true)
        XCTAssertEqual(vm.externalEngineEnumOption(general[2]), "legacy_rejection_v1")

        vm.setExternalEngineIntOption(general[0], 1)
        vm.setExternalEngineBoolOption(general[1], false)
        vm.setExternalEngineEnumOption(general[2], "position_coupled_v1")
        XCTAssertEqual(vm.externalEngineIntOption(general[0]), 1)
        XCTAssertEqual(vm.externalEngineBoolOption(general[1]), false)
        XCTAssertEqual(vm.externalEngineEnumOption(general[2]), "position_coupled_v1")
        XCTAssertEqual(vm.externalEngineOptions["m5rt"]?["max_concurrency"]?.value as? Int, 1)
        XCTAssertEqual(vm.externalEngineOptions["m5rt"]?["sampling_contract"]?.value as? String, "position_coupled_v1")
    }

    func testExternalEnginePrefillOptionsSplitFromGeneralAndSupportModesA4GatingAndTelemetry() throws {
        let vm = ModelSettingsScreenVM()
        let m5Opt = ExternalEngineOption(
            id: "m5rt",
            label: "miv Core Backend (ninfer-metal)",
            compatible: true,
            reason: nil,
            supportedStandardSettings: ["max_context_window", "dflash_enabled", "mtp_enabled"],
            providerId: "io.ninfer.metal",
            optionsSchema: [
                ["key": AnyCodable("max_concurrency"), "label": AnyCodable("Concurrency"), "type": AnyCodable("int"), "default": AnyCodable(1)],
                ["key": AnyCodable("prefill_linear_mode"), "label": AnyCodable("Linear"), "type": AnyCodable("enum"), "options": AnyCodable([AnyCodable("bf16"), AnyCodable("w4a8"), AnyCodable("w4a4")]), "default": AnyCodable("bf16")],
                ["key": AnyCodable("prefill_attention_mode"), "label": AnyCodable("Sage2"), "type": AnyCodable("enum"), "options": AnyCodable([AnyCodable("exact_nax"), AnyCodable("sage2")]), "default": AnyCodable("exact_nax")],
                ["key": AnyCodable("prefill_sparse_minference"), "label": AnyCodable("MInference"), "type": AnyCodable("bool"), "default": AnyCodable(false)],
                ["key": AnyCodable("prefill_cache_blend"), "label": AnyCodable("CacheBlend"), "type": AnyCodable("bool"), "default": AnyCodable(false)],
                ["key": AnyCodable("prefill_ane_mode"), "label": AnyCodable("ANE Mode"), "type": AnyCodable("enum"), "options": AnyCodable([AnyCodable("off"), AnyCodable("auto"), AnyCodable("compact_int8"), AnyCodable("ultra_fp16_swiglu")]), "default": AnyCodable("off")],
            ]
        )

        // 1) Non-MXFP4 (oQ4e) model: A4 is gated & saved w4a4 auto-falls back to A8 (w4a8)
        var oqModel = makeModel(id: "Qwen3.8-27B-oQ4e-mtp", configModelType: "qwen3_5")
        oqModel.externalEngines = [m5Opt]
        oqModel.autoExternalEngine = "m5rt"
        oqModel.quantFormat = "oq_affine"
        oqModel.mxfp4W4A4Supported = false
        vm.model = oqModel
        vm.externalEngine = ""
        vm.externalEngineOptions = ["m5rt": ["prefill_linear_mode": AnyCodable("w4a4")]]

        XCTAssertFalse(vm.isMxfp4ExternalEngineModel)
        XCTAssertEqual(vm.externalEngineGeneralOptions.map(\.key), ["max_concurrency"])
        XCTAssertEqual(vm.externalEnginePrefillOptions.map(\.key), [
            "prefill_linear_mode",
            "prefill_attention_mode",
            "prefill_sparse_minference",
            "prefill_cache_blend",
            "prefill_ane_mode",
        ])
        XCTAssertEqual(vm.activeMivLinearMode, "w4a8")
        XCTAssertEqual(vm.activeMivLinearSubTier, .a8)
        XCTAssertEqual(vm.normalizeMivPrefillOptionsForCurrentModel(), ["prefill_linear_mode"])
        XCTAssertEqual(vm.externalEngineOptions["m5rt"]?["prefill_linear_mode"]?.value as? String, "w4a8")
        XCTAssertEqual(vm.activeMivPrefillMode, .routeA)

        // Selecting A4 on oQ4e auto-falls back to A8 ("w4a8")
        let a4KeysOnOq = vm.stageMivLinearSubTier(.a4)
        XCTAssertEqual(a4KeysOnOq, ["prefill_linear_mode"])
        XCTAssertEqual(vm.externalEngineOptions["m5rt"]?["prefill_linear_mode"]?.value as? String, "w4a8")
        XCTAssertEqual(vm.activeMivLinearSubTier, .a8)

        // 2) MXFP4 model: A4 enabled, 4-Tier mode preserves A4 and NEVER touches MInference / CacheBlend / ANE
        var mxfp4Model = makeModel(id: "Qwen3.8-27B-mxfp4", configModelType: "qwen3_8")
        mxfp4Model.externalEngines = [m5Opt]
        mxfp4Model.autoExternalEngine = "m5rt"
        mxfp4Model.quantFormat = "mxfp4"
        mxfp4Model.mxfp4W4A4Supported = true
        vm.model = mxfp4Model
        vm.externalEngineOptions = [:]
        vm.syncPreferredMivLinearSubTier()

        XCTAssertTrue(vm.isMxfp4ExternalEngineModel)
        XCTAssertEqual(vm.activeMivPrefillMode, .lossless)
        XCTAssertFalse(vm.isExternalEngineSage2Active)

        // Enable CacheBlend + ANE=ultra_fp16_swiglu independently
        let blendOpt = try XCTUnwrap(vm.externalEnginePrefillOption(forKey: "prefill_cache_blend"))
        XCTAssertEqual(vm.setExternalEnginePrefillToggle(blendOpt, isOn: true), ["prefill_cache_blend"])
        XCTAssertEqual(vm.stageMivAneMode(.ultraFp16SwiGlu), ["prefill_ane_mode"])
        XCTAssertEqual(vm.activeMivAneMode, .ultraFp16SwiGlu)

        // Route B (Sage2) does NOT force-enable MInference (strictly independent)
        XCTAssertEqual(vm.stageMivPrefillMode(.routeB), ["prefill_linear_mode", "prefill_attention_mode"])
        XCTAssertEqual(vm.activeMivPrefillMode, .routeB)
        XCTAssertTrue(vm.isExternalEngineSage2Active)
        let minfOpt = try XCTUnwrap(vm.externalEnginePrefillOption(forKey: "prefill_sparse_minference"))
        XCTAssertFalse(vm.externalEnginePrefillToggleValue(for: minfOpt))
        XCTAssertTrue(vm.externalEnginePrefillToggleValue(for: blendOpt))
        XCTAssertEqual(vm.activeMivAneMode, .ultraFp16SwiGlu)

        // Route A -> A4 -> Route A+B preserves A4 and leaves MInference/CacheBlend/ANE untouched
        _ = vm.stageMivPrefillMode(.routeA)
        XCTAssertEqual(vm.activeMivLinearMode, "w4a8")
        XCTAssertEqual(vm.stageMivLinearSubTier(.a4), ["prefill_linear_mode"])
        XCTAssertEqual(vm.activeMivLinearMode, "w4a4")
        _ = vm.stageMivPrefillMode(.routeAB)
        XCTAssertEqual(vm.activeMivPrefillMode, .routeAB)
        XCTAssertEqual(vm.activeMivLinearMode, "w4a4")
        XCTAssertFalse(vm.externalEnginePrefillToggleValue(for: minfOpt))
        XCTAssertTrue(vm.externalEnginePrefillToggleValue(for: blendOpt))
        XCTAssertEqual(vm.activeMivAneMode, .ultraFp16SwiGlu)

        // Toggle MInference ON independently
        XCTAssertEqual(vm.setExternalEnginePrefillToggle(minfOpt, isOn: true), ["prefill_sparse_minference"])
        XCTAssertTrue(vm.externalEnginePrefillToggleValue(for: minfOpt))
        XCTAssertEqual(vm.enabledFeatures(on: .acceleration).count, 5)

        // Legacy prefill_ane_offload=true hydrates activeMivAneMode as .auto and syncs on write
        vm.externalEngineOptions = ["m5rt": ["prefill_ane_offload": AnyCodable(true)]]
        XCTAssertEqual(vm.activeMivAneMode, .auto)
        let aneKeys = vm.stageMivAneMode(.off)
        XCTAssertEqual(aneKeys, ["prefill_ane_mode", "prefill_ane_offload"])
        XCTAssertEqual(vm.externalEngineOptions["m5rt"]?["prefill_ane_mode"]?.value as? String, "off")
        XCTAssertEqual(vm.externalEngineOptions["m5rt"]?["prefill_ane_offload"]?.value as? Bool, false)

        // Verify live telemetry decoding on MivEffectiveSettings (including W4A4 fallback & ANE mode)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let eff = try decoder.decode(
            MivEffectiveSettings.self,
            from: Data(#"""
            {
              "max_context": 262144,
              "max_concurrency": 1,
              "kv_dtype": "affine4_g128",
              "sampling_contract_id": "position_coupled_v1",
              "speculative_backend": "mtp",
              "max_draft_tokens": 4,
              "concurrency_source": "default",
              "prefill_linear_mode": "w4a4",
              "prefill_linear_mode_effective": "w4a8",
              "prefill_linear_fallback": true,
              "prefill_linear_fallback_reason": "requires_mxfp4_model",
              "model_quant_format": "oq_affine",
              "prefill_attention_mode": "sage2",
              "prefill_sparse_minference": true,
              "prefill_sparse_minference_effective": true,
              "prefill_cache_blend": false,
              "prefill_ane_mode": "ultra_fp16_swiglu",
              "prefill_ane_mode_effective": "ultra_fp16_swiglu",
              "prefill_ane_offload": true,
              "prefill_ane_active": true,
              "prefill_w4a8_tokens": 8192,
              "prefill_w4a4_tokens": 0,
              "prefill_sage2_tokens": 8192,
              "prefill_minference_blocks_skipped": 420
            }
            """#.utf8)
        )
        XCTAssertEqual(eff.prefillLinearMode, "w4a4")
        XCTAssertEqual(eff.prefillLinearModeEffective, "w4a8")
        XCTAssertEqual(eff.prefillLinearFallback, true)
        XCTAssertEqual(eff.prefillLinearFallbackReason, "requires_mxfp4_model")
        XCTAssertEqual(eff.modelQuantFormat, "oq_affine")
        XCTAssertEqual(eff.prefillAttentionMode, "sage2")
        XCTAssertEqual(eff.prefillSparseMinferenceEffective, true)
        XCTAssertEqual(eff.prefillAneMode, "ultra_fp16_swiglu")
        XCTAssertEqual(eff.prefillAneModeEffective, "ultra_fp16_swiglu")
        XCTAssertEqual(eff.prefillAneActive, true)
        XCTAssertEqual(eff.prefillW4A8Tokens, 8192)
        XCTAssertEqual(eff.prefillW4A4Tokens, 0)
        XCTAssertEqual(eff.prefillMinferenceBlocksSkipped, 420)

        // Verify snake_case JSON decoding of quant_format & mxfp4_w4a4_supported on ModelDTO
        let decodedMxfp4 = try decoder.decode(
            ModelDTO.self,
            from: Data(#"{"id":"qwen-mxfp4","loaded":false,"is_loading":false,"estimated_size":0,"quant_format":"mxfp4","mxfp4_w4a4_supported":true}"#.utf8)
        )
        XCTAssertEqual(decodedMxfp4.quantFormat, "mxfp4")
        XCTAssertEqual(decodedMxfp4.mxfp4W4A4Supported, true)
        let decodedOq = try decoder.decode(
            ModelDTO.self,
            from: Data(#"{"id":"qwen-oq4e","loaded":false,"is_loading":false,"estimated_size":0,"quant_format":"oq_affine","mxfp4_w4a4_supported":false}"#.utf8)
        )
        XCTAssertEqual(decodedOq.quantFormat, "oq_affine")
        XCTAssertEqual(decodedOq.mxfp4W4A4Supported, false)
    }
}

/// Serves one model with Lightning MTP on and rejects its settings writes
/// the way the server rejects an MTP + DFlash pair. Writes to "clear" must
/// send JSON null.
private final class RejectingSettingsURLProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        do {
            let path = request.url?.path ?? ""
            var status = 200
            let body: Data
            switch path {
            case "/admin/api/models":
                body = Data(#"{"models":[{"id":"m","loaded":false,"is_loading":false,"estimated_size":0,"config_model_type":"qwen3_5","settings":{"mtp_enabled":true}}]}"#.utf8)
            case "/admin/api/models/m/profiles":
                body = Data(#"{"profiles":[]}"#.utf8)
            case "/admin/api/profile-templates":
                body = Data(#"{"templates":[]}"#.utf8)
            case "/admin/api/global-settings":
                body = try Data(contentsOf: URL(fileURLWithPath: #filePath)
                    .deletingLastPathComponent()
                    .appendingPathComponent("Fixtures/global-settings.json"))
            case "/admin/api/models/clear/settings":
                let stream = try XCTUnwrap(request.httpBodyStream)
                stream.open()
                defer { stream.close() }
                let settings = try XCTUnwrap(JSONSerialization.jsonObject(with: stream) as? [String: Any])
                XCTAssertTrue(!settings.isEmpty && settings.values.allSatisfy { $0 is NSNull })
                body = Data(#"{"status":"ok"}"#.utf8)
            case "/admin/api/models/m/settings":
                status = 400
                body = Data(#"{"detail":"MTP and DFlash cannot both be enabled; choose one speculative-decoding path."}"#.utf8)
            default:
                status = 404
                body = Data()
            }
            let response = HTTPURLResponse(url: request.url!, statusCode: status,
                                           httpVersion: nil, headerFields: nil)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: body)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
