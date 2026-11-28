// PR 8 (+ PR 14 advanced/experimental expansion) — GET /admin/api/models
// response + per-model settings shape, plus the patch body for
// PUT /admin/api/models/{id}/settings.
//
// All fields the HTML admin modal renders are now exposed here so the
// Swift Advanced tab can reach feature parity. Experimental flags
// (TurboQuant KV, IndexCache, SpecPrefill, DFlash, native MTP) decode
// alongside compatibility readouts (`dflash_compatible`,
// `mtp_compatible`, etc.) sourced from the server so the UI can disable
// switches the runtime won't accept.

import Foundation

struct ListModelsResponse: Codable, Sendable {
    let models: [ModelDTO]
}

struct ModelDTO: Codable, Equatable, Sendable, Identifiable {
    let id: String
    let displayName: String?
    let modelPath: String?
    let loaded: Bool
    let isLoading: Bool
    let estimatedSize: Int64
    let estimatedSizeFormatted: String?
    let actualSize: Int64?
    let actualSizeFormatted: String?
    let pinned: Bool?
    let isDefault: Bool?
    let isFavorite: Bool?
    let engineType: String?
    let modelType: String?
    /// Lower-level config-derived model class (e.g. `deepseek_v32`,
    /// `glm_moe_dsa`). Used to gate the IndexCache row to DSA models.
    let configModelType: String?
    /// Native context window from the model's config.json. The Context
    /// Bench target selector hides presets beyond it.
    let modelContextLength: Int?
    /// Server-side default for `enable_thinking` derived from the model
    /// (chat template, config). UI shows it as the inherited value when
    /// `enable_thinking` is unset and offers a one-click reset to it.
    let thinkingDefault: Bool?
    var thinkingForced: Bool? = nil
    var reasoningEffortOptions: [String]? = nil
    var reasoningEffortDefault: String? = nil
    var reasoningEffortCustom: Bool? = nil
    var anePrefillBackend: String? = nil
    var anePrefillDefaultFraction: Double? = nil
    var anePrefillMlpFractions: [Double]? = nil
    var anePrefillSharedFractions: [Double]? = nil
    var embeddingAudioSupported: Bool? = nil
    /// True when the model can hold a spoken conversation on /v1/realtime.
    var realtimeVoice: Bool? = nil
    /// External engines that could serve this model, with why not when they
    /// cannot. Empty for models only oMLX's own engines run.
    var externalEngines: [ExternalEngineOption]? = nil
    /// External engine chosen automatically for an adapted model, if any.
    var autoExternalEngine: String? = nil
    /// Resolved or currently-serving engine identifier (`"m5rt"`, Path B id, `"splash"`, or `"mlx"`).
    var activeEngineId: String? = nil
    /// Human-readable display label for the active/resolved engine.
    var activeEngineLabel: String? = nil
    var mivEffectiveSettings: MivEffectiveSettings? = nil
    /// Standard `ModelSettings` keys honored by the active external engine.
    var supportedStandardSettings: [String]? = nil
    /// True when the model is structurally compatible with DFlash (block
    /// diffusion speculative decoding). The toggle stays disabled when false.
    let dflashCompatible: Bool?
    /// Human-readable explanation when `dflashCompatible` is false. Surfaced
    /// as a tooltip on the disabled DFlash toggle.
    let dflashCompatibilityReason: String?
    /// True when the model is structurally compatible with native MTP.
    let mtpCompatible: Bool?
    let mtpCompatibilityReason: String?
    /// Qwen4-Exp PLE mmap capability and server-side forced residency decision.
    let qwen4PleSsdOffloadSupported: Bool?
    let qwen4PleSsdOffloadForced: Bool?
    let qwen4PleResidentBytes: Int64?
    let qwen4PleMmapBytes: Int64?
    /// DeepSeek V4.1 Engram counterpart of the Qwen4 PLE fields above.
    var deepseekV41EngramSsdOffloadSupported: Bool? = nil
    var deepseekV41EngramSsdOffloadForced: Bool? = nil
    var deepseekV41EngramResidentBytes: Int64? = nil
    var deepseekV41EngramMmapBytes: Int64? = nil
    var moeExpertOffloadSupported: Bool? = nil
    /// True on M1/M2 for checkpoints that still carry bfloat16 leaves.
    var activationFp16Supported: Bool? = nil
    /// True for families whose MTP draft head stays resident while experts stream.
    var moeOffloadAllowsMtp: Bool? = nil
    /// True when this Mac can run the Qwen INT8 activation prefill kernels.
    var a8PrefillSupported: Bool? = nil
    /// True when this Mac's GPU accepts a power cap.
    var gpuPowerLimitSupported: Bool? = nil
    /// True for families another Mac can help prefill (Prefill Accelerator).
    var prefillHelperSupported: Bool? = nil
    /// The helper's state while the model is loaded; nil otherwise.
    var prefillHelper: PrefillHelperStatusDTO? = nil
    /// "hf_cache" for models found in the shared Hugging Face Hub cache.
    var sourceType: String? = nil
    /// "gguf", "miv", or "omlxm5" when only an external engine serves the model.
    var modelFormat: String? = nil
    /// Active weight container format (`"miv"`, `"omlxm5"`, `"safetensors"`, `"gguf"`).
    var weightFormat: String? = nil
    /// True when repacked microkernel weights (`.miv` / `.omlxm5`) exist in the model directory.
    var hasMivWeights: Bool? = nil
    /// Built-in artifact present on disk or in the artifact cache, independent of the selected engine.
    var mivArtifactFormat: String? = nil
    /// Effective MIV routing from the backend, including provider enablement and model overrides.
    var mivRouteEnabled: Bool? = nil
    /// True when the model currently has `.safetensors` and will auto-repack to `.miv` on first Load.
    var pendingAutoRepack: Bool? = nil
    /// Target format suffix when `pendingAutoRepack` is true (e.g. `"miv"`).
    var repackTargetFormat: String? = nil
    /// True when `.safetensors` weights (including companion DFlash2 drafts) can be repacked to `.miv`.
    var canConvert: Bool? = nil
    /// True for builtin virtual entries (e.g. the MarkItDown document
    /// converter) that have no real load/unload lifecycle.
    let virtual: Bool?
    let settings: ModelSettingsDTO?
}

extension ModelDTO {
    /// Single size figure for compact UI: the observed footprint once the
    /// model has settled, the estimate while loading or before one exists.
    var sizeLabel: String {
        if isLoading {
            return estimatedSizeFormatted ?? ""
        }
        return actualSizeFormatted ?? estimatedSizeFormatted ?? ""
    }

    var isInHFCache: Bool { sourceType == "hf_cache" }
    var isGGUF: Bool { modelFormat == "gguf" }
    var isMivOnly: Bool { modelFormat == "miv" || modelFormat == "omlxm5" }
    var isOMLXM5Only: Bool { isMivOnly }

    var canConvertMiv: Bool {
        canConvert == true || externalEngines?.contains(where: { $0.canConvert == true }) == true
    }

    var effectiveWeightFormat: String {
        if let wf = weightFormat, !wf.isEmpty {
            return wf.lowercased()
        }
        if isGGUF { return "gguf" }
        if let mf = modelFormat, !mf.isEmpty {
            return mf.lowercased()
        }
        return "safetensors"
    }

    enum ListMivBadge: Equatable {
        case available, routed

        var help: String {
            switch self {
            case .available:
                return String(localized: "models.format.miv_available.help",
                              defaultValue: "This model has .miv weights. MIV routing is off.")
            case .routed:
                return String(localized: "models.format.miv_routed.help",
                              defaultValue: "This model has .miv weights and is routed to the MIV engine.")
            }
        }
    }

    static func isBuiltinMivEngine(_ id: String?) -> Bool {
        switch id {
        case "m5rt", "io.ninfer.metal", "ninfer-metal", "com.omlx.mseries": return true
        default: return false
        }
    }

    var listMivBadge: ListMivBadge? {
        guard virtual != true else { return nil }
        if let format = mivArtifactFormat {
            guard format == "miv" || format == "omlxm5" else { return nil }
        } else {
            // New servers send an explicit route flag even when the artifact is absent.
            // Only older servers need the fallback; their boolean also counts Path B artifacts.
            let format = effectiveWeightFormat
            guard mivRouteEnabled == nil, hasMivWeights != false,
                  format == "miv" || format == "omlxm5" || (format == "safetensors" && hasMivWeights == true)
            else { return nil }
        }
        if let mivRouteEnabled { return mivRouteEnabled ? .routed : .available }
        // Older servers can still report the previous live engine while it unloads.
        let saved = settings?.externalEngine ?? ""
        let route = saved.isEmpty || saved == "auto" ? activeEngineId ?? autoExternalEngine : saved
        return Self.isBuiltinMivEngine(route) ? .routed : .available
    }

    var showsOtherListWeightFormat: Bool {
        guard virtual != true, pendingAutoRepack != true else { return false }
        switch effectiveWeightFormat {
        case "safetensors", "miv": return false
        case "omlxm5": return mivRouteEnabled != nil && mivArtifactFormat != "omlxm5"
        default: return true
        }
    }

    func showsListEngineBadge(inActiveModels: Bool) -> Bool {
        guard virtual != true, !Self.isBuiltinMivEngine(resolvedEngineId) else { return false }
        return inActiveModels || resolvedEngineId != "mlx" || isStockEngineOverride
    }

    var weightFormatBadgeLabel: String {
        if pendingAutoRepack == true {
            let target = (repackTargetFormat?.isEmpty == false) ? repackTargetFormat! : "miv"
            return "safetensors → .\(target)"
        }
        let fmt = effectiveWeightFormat
        switch fmt {
        case "gguf":
            return "GGUF"
        case "safetensors":
            return "safetensors"
        default:
            return fmt.hasPrefix(".") ? fmt : ".\(fmt)"
        }
    }

    var weightFormatTooltip: String {
        if pendingAutoRepack == true {
            let target = (repackTargetFormat?.isEmpty == false) ? repackTargetFormat! : "miv"
            return String(
                localized: "models.format.pending_repack.help",
                defaultValue: "Currently .safetensors; first Load automatically repacks .\(target) alongside original weights.",
                comment: "Tooltip when a convertible .safetensors model will auto-repack on first Load; placeholder is target format"
            )
        }
        let fmt = effectiveWeightFormat
        if fmt == "gguf" {
            return String(
                localized: "models.library.gguf.help",
                defaultValue: "A GGUF file, served by an external engine",
                comment: "Tooltip on the GGUF badge in the model library"
            )
        }
        if fmt == "safetensors" {
            if hasMivWeights == true {
                return String(
                    localized: "models.format.safetensors_with_miv.help",
                    defaultValue: "Running on original .safetensors weights (.miv weights are also present in this folder).",
                    comment: "Tooltip when a model runs on .safetensors while .miv also exists in the directory"
                )
            }
            return String(
                localized: "models.format.safetensors.help",
                defaultValue: "Standard .safetensors weights",
                comment: "Tooltip for standard .safetensors model format badge"
            )
        }
        if isMivOnly {
            return String(
                localized: "models.format.miv_only.help",
                defaultValue: "Pure .\(fmt) Metal 4 microkernel weights (no .safetensors in folder).",
                comment: "Tooltip for a pure .miv microkernel weight folder; placeholder is format name"
            )
        }
        return String(
            localized: "models.format.miv_colocated.help",
            defaultValue: "Repacked .\(fmt) Metal 4 microkernel weights (co-located alongside original .safetensors).",
            comment: "Tooltip when repacked .miv weights sit alongside original .safetensors; placeholder is format name"
        )
    }

    /// Localized display name for a given engine ID, translating built-in `m5rt`
    /// while preserving manifest display names for Path B providers and Splash.
    static func localizedEngineLabel(id: String, fallback: String? = nil) -> String {
        switch id {
        case "m5rt":
            return String(localized: "external_engine.m5rt.label",
                          defaultValue: "miv Core Backend",
                          comment: "Display name for the built-in miv Core Backend (m5rt) engine")
        case "mlx", "", "stock":
            return "MLX"
        default:
            if let fallback, !fallback.isEmpty { return fallback }
            return id
        }
    }

    /// Resolved engine serving (or selected to serve) this model.
    var resolvedEngineId: String {
        if let activeEngineId, !activeEngineId.isEmpty {
            return activeEngineId
        }
        let saved = settings?.externalEngine ?? ""
        if saved == "stock" { return "mlx" }
        if !saved.isEmpty { return saved }
        if let autoExternalEngine, !autoExternalEngine.isEmpty {
            return autoExternalEngine
        }
        if isGGUF {
            return externalEngines?.first(where: \.compatible)?.id
                ?? externalEngines?.first?.id
                ?? "splash"
        }
        return "mlx"
    }

    /// True when the model is served via automatic adapted-model routing (`external_engine=null`).
    var isAutoRoutedEngine: Bool {
        let saved = settings?.externalEngine ?? ""
        guard saved.isEmpty, let auto = autoExternalEngine, !auto.isEmpty else {
            return false
        }
        return resolvedEngineId == auto
    }

    /// True when an adapted model has been explicitly forced back to Stock MLX.
    var isStockEngineOverride: Bool {
        settings?.externalEngine == "stock" && !(autoExternalEngine ?? "").isEmpty
    }

    /// Full display label for the resolved engine.
    var resolvedEngineDisplayLabel: String {
        let id = resolvedEngineId
        if id == "mlx" {
            return isStockEngineOverride
                ? String(localized: "external_engine.stock_mlx.label",
                         defaultValue: "Stock MLX",
                         comment: "Display label when running on Stock MLX")
                : "MLX"
        }
        let fallback = activeEngineLabel
            ?? externalEngines?.first(where: { $0.id == id })?.label
        return Self.localizedEngineLabel(id: id, fallback: fallback)
    }

    /// Compact pill label for `Active Models` and `Model Library` rows.
    var compactEngineBadgeLabel: String {
        let id = resolvedEngineId
        switch id {
        case "m5rt":
            return String(localized: "external_engine.m5rt.compact",
                          defaultValue: "miv Core",
                          comment: "Compact badge label for miv Core Backend on model rows")
        case "mlx":
            return isStockEngineOverride
                ? String(localized: "external_engine.stock_mlx.label",
                         defaultValue: "Stock MLX",
                         comment: "Display label when running on Stock MLX")
                : "MLX"
        default:
            return resolvedEngineDisplayLabel
        }
    }

    /// Compact badge label for `Model Library` rows, including an `Auto` suffix when auto-routed.
    var libraryEngineBadgeLabel: String {
        if isAutoRoutedEngine {
            let autoSuffix = String(localized: "external_engine.auto.short",
                                    defaultValue: "Auto",
                                    comment: "Short suffix indicating automatic external-engine routing")
            return "\(compactEngineBadgeLabel) · \(autoSuffix)"
        }
        return compactEngineBadgeLabel
    }

    var activeEngineTooltip: String {
        if isAutoRoutedEngine {
            return String(localized: "external_engine.tooltip.auto",
                          defaultValue: "\(resolvedEngineDisplayLabel) (auto-routed)",
                          comment: "Tooltip for an auto-routed external engine badge; placeholder is engine name")
        }
        if isStockEngineOverride {
            return String(localized: "settings.advanced.external_engine.stock_override",
                          defaultValue: "Using Stock MLX (auto-route overridden)",
                          comment: "Hint shown when an adapted model is overridden back to Stock MLX")
        }
        return resolvedEngineDisplayLabel
    }
}

struct MivEffectiveSettings: Codable, Equatable, Sendable {
    let maxContext: Int
    let maxConcurrency: Int
    let kvDtype: String
    let samplingContractId: String
    let speculativeBackend: String
    let maxDraftTokens: Int
    let concurrencySource: String
    var prefillLinearMode: String? = nil
    var prefillAttentionMode: String? = nil
    var prefillSparseMinference: Bool? = nil
    var prefillSparseMinferenceEffective: Bool? = nil
    var prefillSparseMinferenceInactiveReason: String? = nil
    var prefillCacheBlend: Bool? = nil
    var prefillAneOffload: Bool? = nil
    var prefillAneActive: Bool? = nil
    var prefillAneSplitAlpha: Double? = nil
    var prefillW4a8Tokens: Int? = nil
    var prefillSage2Tokens: Int? = nil
    var prefillMinferenceBlocksSkipped: Int? = nil
    var prefillMinferenceBlocksTotal: Int? = nil
    var prefillCacheBlendHitTokens: Int? = nil
    var prefillCacheBlendRecomputedTokens: Int? = nil
    var prefillAneOffloadTokens: Int? = nil
    var prefillAneFallbackTokens: Int? = nil
}

/// One entry of `external_engines` in GET /admin/api/models.
struct ExternalEngineOption: Codable, Equatable, Sendable, Identifiable {
    let id: String
    let label: String
    let compatible: Bool
    let reason: String?
    /// Hugging Face repo of the draft the engine lacks, when one provides it.
    var draftRepo: String? = nil
    /// True when `.safetensors` weights can be converted in-place to `.miv` for this engine.
    var canConvert: Bool? = nil
    /// Standard `ModelSettings` keys honored when this engine serves the model.
    var supportedStandardSettings: [String]? = nil
    var requestSettings: [String]? = nil
    var speculativeLimits: [String: Int]? = nil
    /// Canonical SPI provider ID when backed by an out-of-process provider.
    var providerId: String? = nil
    /// Provider-specific per-model options schema.
    var optionsSchema: [[String: AnyCodable]]? = nil
}

struct ModelSettingsDTO: Codable, Equatable, Sendable {
    let modelAlias: String?
    let modelTypeOverride: String?
    let maxContextWindow: Int?
    let maxTokens: Int?
    let temperature: Double?
    let topP: Double?
    let topK: Int?
    let minP: Double?
    let presencePenalty: Double?
    let repetitionPenalty: Double?
    let forceSampling: Bool?
    let maxToolResultTokens: Int?
    let enableThinking: Bool?
    let qwen4PleSsdOffload: Bool?
    var qwen4GdnDecodeWideProj: Bool? = nil
    var activationFp16Enabled: Bool? = nil
    var deepseekV41EngramSsdOffload: Bool? = nil
    var deepseekV41CedPrefillEnabled: Bool? = nil
    var prefillHelperNodeId: String? = nil
    var prefillHelperLayers: Int? = nil
    let thinkingBudgetEnabled: Bool?
    let thinkingBudgetTokens: Int?
    var gpuPowerLimitW: Int? = nil
    let reasoningParser: String?
    let ttlSeconds: Int?
    let isPinned: Bool?
    let isDefault: Bool?
    let isFavorite: Bool?
    let displayName: String?
    let activeProfileName: String?
    // Security
    let trustRemoteCode: Bool?
    /// External engine serving the model ("splash"); nil runs it in-process.
    var externalEngine: String? = nil
    var embeddingAudioEnabled: Bool? = nil
    var embeddingAudioMaxSeconds: Double? = nil
    // Chat-template kwargs (free-form dict + a sibling list of keys the
    // user wants to *force* — those go to `forced_ct_kwargs` server-side
    // and override the request's `chat_template_kwargs`).
    let chatTemplateKwargs: [String: AnyCodable]?
    let forcedCtKwargs: [String]?
    var guidedGrammarEnabled: Bool? = nil
    var guidedGrammar: String? = nil
    var moeExpertOffloadEnabled: Bool? = nil
    var moeExpertOffloadResidentFraction: Double? = nil
    var loopcdEnabled: Bool? = nil
    var loopcdSkipThinking: Bool? = nil
    // Experimental: TurboQuant KV cache
    let turboquantKvEnabled: Bool?
    let turboquantKvScheme: String?
    let turboquantKvBits: Double?
    // Experimental: private Qwen3.5/3.6/3.8 ANE/GPU prefill
    var qwen35AnePrefillSharedFraction: Double? = nil
    let qwen35AnePrefillEnabled: Bool?
    let qwen35AnePrefillSequenceLength: Int?
    let qwen35AnePrefillTailPaddingMinTokens: Int?
    let qwen35AnePrefillFraction: Double?
    let qwen35AnePrefillFusedDown: Bool?
    let qwen35AnePrefillMaxLayers: Int?
    let qwen35AnePrefillDualAne: Bool?
    let qwen35AnePrefillGdn: Bool?
    let qwen35AnePrefillGdnFraction: Double?
    let qwen35AnePrefillGdnMaxLayers: Int?
    let qwen35AnePrefillCpuEnabled: Bool?
    let qwen35AnePrefillCpuFraction: Double?
    let qwen35AnePrefillCpuDownFraction: Double?
    let qwen35AnePrefillCpuGdnFraction: Double?
    let qwen35AnePrefillCpuThreads: Int?
    let qwen35AnePrefillCpuSharedResource: Bool?
    // Experimental: oQ mixed-bit INT8-activation prefill kernels
    let a8PrefillEnabled: Bool?
    let a8PrefillMinTokens: Int?
    // Experimental: IndexCache (DSA models only)
    let indexCacheFreq: Int?
    // Experimental: SpecPrefill
    let specprefillEnabled: Bool?
    let specprefillDraftModel: String?
    let specprefillKeepPct: Double?
    let specprefillThreshold: Int?
    // Experimental: DFlash (block diffusion speculative decoding)
    let dflashEnabled: Bool?
    let dflashDraftModel: String?
    let dflashDraftQuantEnabled: Bool?
    let dflashDraftQuantWeightBits: Int?
    let dflashDraftQuantGroupSize: Int?
    let dflashDraftWindowSize: Int?
    let dflashBlockSize: Int?
    let dflashVerifyMode: String?
    // Experimental: native MTP (mlx-lm PR 990 / PR 15 monkey-patch)
    let mtpEnabled: Bool?
    let mtpAdaptiveMaxDepth: Int?
    // Experimental: VLM MTP (mlx-vlm assistant-drafter speculative decoding)
    let vlmMtpEnabled: Bool?
    let vlmMtpDraftModel: String?
    let vlmMtpDraftBlockSize: Int?
    // LoRA adapter folder applied on top of the weights at load.
    let loraAdapter: String?
    /// Per-provider external engine options (e.g. `lora_mode`, `max_concurrency`, `sampling_contract`).
    var externalEngineOptions: [String: [String: AnyCodable]]? = nil
    /// Legacy `m5rt_options` dict from older `model_settings.json` files.
    var m5rtOptions: [String: AnyCodable]? = nil
}

/// Patch body for PUT /admin/api/models/{id}/settings. Flat snake-cased
/// keys (Encoder converts via `.convertToSnakeCase`). `nil` fields are
/// omitted by `encodeIfPresent` so the server merges instead of resetting.
struct ModelSettingsPatch: Encodable, Equatable, Sendable {
    var modelAlias: String?? = nil
    var modelTypeOverride: String?? = nil
    var maxContextWindow: Int? = nil
    var maxTokens: Int? = nil
    var temperature: Double? = nil
    var topP: Double? = nil
    var topK: Int? = nil
    var minP: Double? = nil
    var presencePenalty: Double? = nil
    var repetitionPenalty: Double? = nil
    var ttlSeconds: Int?? = nil  // Explicit null clears the model override.
    var enableThinking: Bool?? = nil // nil omits the key. .some(nil) sends JSON null.
    var qwen4PleSsdOffload: Bool? = nil
    var qwen4GdnDecodeWideProj: Bool? = nil
    var activationFp16Enabled: Bool? = nil
    var deepseekV41EngramSsdOffload: Bool? = nil
    var deepseekV41CedPrefillEnabled: Bool? = nil
    var prefillHelperNodeId: String?? = nil // .some(nil) sends JSON null (off).
    var prefillHelperLayers: Int?? = nil // .some(nil) sends JSON null (auto).
    var thinkingBudgetEnabled: Bool? = nil
    var thinkingBudgetTokens: Int? = nil
    var maxToolResultTokens: Int? = nil
    var gpuPowerLimitW: Int? = nil // 0 turns the limit off.
    var forceSampling: Bool? = nil
    var isPinned: Bool? = nil
    var isFavorite: Bool? = nil
    // Security
    var trustRemoteCode: Bool? = nil
    var externalEngine: String?? = nil // .some(nil) sends JSON null (in-process).
    var embeddingAudioEnabled: Bool? = nil
    var embeddingAudioMaxSeconds: Double?? = nil // .some(nil) sends JSON null (30 s default).
    var reasoningParser: String? = nil
    // Chat-template kwargs
    var chatTemplateKwargs: [String: AnyCodable]? = nil
    var forcedCtKwargs: [String]? = nil
    // Experimental: TurboQuant KV
    var turboquantKvEnabled: Bool? = nil
    var turboquantKvScheme: String? = nil
    var turboquantKvBits: Double? = nil
    // Experimental: private Qwen3.5/3.6/3.8 ANE/GPU prefill
    var qwen35AnePrefillSharedFraction: Double? = nil
    var qwen35AnePrefillEnabled: Bool? = nil
    var qwen35AnePrefillSequenceLength: Int? = nil
    var qwen35AnePrefillTailPaddingMinTokens: Int? = nil
    var qwen35AnePrefillFraction: Double? = nil
    var qwen35AnePrefillFusedDown: Bool? = nil
    var qwen35AnePrefillMaxLayers: Int? = nil
    var qwen35AnePrefillDualAne: Bool? = nil
    var qwen35AnePrefillGdn: Bool? = nil
    var qwen35AnePrefillGdnFraction: Double? = nil
    var qwen35AnePrefillGdnMaxLayers: Int? = nil
    var qwen35AnePrefillCpuEnabled: Bool? = nil
    var qwen35AnePrefillCpuFraction: Double? = nil
    var qwen35AnePrefillCpuDownFraction: Double? = nil
    var qwen35AnePrefillCpuGdnFraction: Double? = nil
    var qwen35AnePrefillCpuThreads: Int? = nil
    var qwen35AnePrefillCpuSharedResource: Bool? = nil
    // Experimental: oQ mixed-bit INT8-activation prefill kernels
    var a8PrefillEnabled: Bool? = nil
    var a8PrefillMinTokens: Int? = nil
    // Experimental: IndexCache
    var indexCacheFreq: Int? = nil
    // Experimental: SpecPrefill
    var specprefillEnabled: Bool? = nil
    var specprefillDraftModel: String? = nil
    var specprefillKeepPct: Double? = nil
    var specprefillThreshold: Int? = nil
    // Experimental: DFlash
    var dflashEnabled: Bool? = nil
    var dflashDraftModel: String? = nil
    var dflashDraftQuantEnabled: Bool? = nil
    var dflashDraftQuantWeightBits: Int? = nil
    var dflashDraftQuantGroupSize: Int? = nil
    var dflashDraftWindowSize: Int? = nil
    var dflashBlockSize: Int? = nil
    var dflashVerifyMode: String? = nil
    // Experimental: native MTP
    var mtpEnabled: Bool? = nil
    var mtpAdaptiveMaxDepth: Int?? = nil // .some(nil) sends JSON null (model default).
    var mtpFixedDepth: Int?? = nil // .some(nil) sends JSON null (adaptive).
    // Experimental: VLM MTP
    var vlmMtpEnabled: Bool? = nil
    var vlmMtpDraftModel: String? = nil
    var vlmMtpDraftBlockSize: Int? = nil
    var loraAdapter: String? = nil // "" clears the adapter.
    var externalEngineOptions: [String: [String: AnyCodable]]? = nil
    var externalEngineOptionsPatch: [String: [String: AnyCodable]]? = nil
}

/// GET /admin/api/models/{id}/generation_config: the sampling values a new
/// model starts with plus the native context window. Absent keys have no
/// recommendation.
struct GenerationConfigDTO: Decodable, Equatable, Sendable {
    var maxContextWindow: Int? = nil
    var temperature: Double? = nil
    var topP: Double? = nil
    var topK: Int? = nil
    var repetitionPenalty: Double? = nil
}

/// GET /admin/api/models/{id}/lora-adapters: adapters in the model folders.
/// `problem` says why one does not fit this model; nil when it does.
struct LoRAAdapterDTO: Decodable, Equatable, Sendable, Identifiable {
    let id: String
    let path: String
    let baseModel: String?
    let rank: Int
    let targets: [String]
    let sizeFormatted: String?
    let problem: String?
}

struct LoRAAdapterListResponse: Decodable, Sendable {
    let adapters: [LoRAAdapterDTO]
}

/// Body for POST /admin/api/models/{id}/settings/recipe.
struct ApplyRecipeRequest: Encodable, Sendable {
    let recipe: String
}

/// Body for POST /admin/api/models/{id}/settings/optimal.
struct ApplyOptimalRequest: Encodable, Sendable {
    let benchmarkId: String
}

struct SkippedFeatureDTO: Codable, Equatable, Sendable {
    let feature: String
    let reason: String
}

/// One omlx.ai benchmark row offered by GET /settings/optimal.
struct OptimalCandidateDTO: Decodable, Identifiable, Sendable {
    let benchmarkId: String
    let benchmarkUrl: String?
    let ppTps: Double?
    let tgTps: Double?
    let quantization: String?
    let omlxVersion: String?
    let createdAt: String?
    let contextProfile: String?
    let memoryGb: Int?

    var id: String { benchmarkId }
}

/// Response of GET /admin/api/models/{id}/settings/optimal: best rows by
/// prompt processing and by token generation for this device and model.
struct OptimalCandidatesDTO: Decodable, Sendable {
    let found: Bool
    let modelName: String?
    let contextLength: Int?
    let byPp: [OptimalCandidateDTO]
    let byTg: [OptimalCandidateDTO]
    let searchUrl: String?
}

/// Response of the settings snapshot endpoints (reset / recipe / optimal
/// apply). Carries the persisted `settings` plus the scoped `applied` values
/// and the features `skipped` on this machine; the optimal apply adds the
/// benchmark summary.
struct SettingsApplyResultDTO: Decodable, Sendable {
    let success: Bool?
    let benchmarkId: String?
    let benchmarkUrl: String?
    let ppTps: Double?
    let tgTps: Double?
    let quantization: String?
    let omlxVersion: String?
    let requiresReload: Bool?
    let autoUnloaded: Bool?
    let autoReloaded: Bool?
    let changed: Bool?
    let applied: [String: AnyCodable]?
    let skipped: [SkippedFeatureDTO]?
    let settings: ModelSettingsDTO?
}

/// Generic acknowledgment shape returned by non-streaming admin endpoints
/// that just need to signal completion (model load/unload, settings patch,
/// task cancel/remove, stats clear, sub-key CRUD, etc.). Server responses
/// vary in which subset of `status`/`message`/`success` they populate, so
/// all three are optional and callers typically only check that the call
/// did not throw.
struct SimpleStatusResponse: Codable, Sendable {
    let status: String?
    let message: String?
    let success: Bool?
}

// MARK: - Prefill Accelerator

/// GET /admin/api/prefill-helper/models/{id}, also `prefill_helper` in the
/// model list. `state` is off, connecting, needs_files, copying, starting,
/// ready, working, offline, stopped, error or unloaded.
struct PrefillHelperStatusDTO: Codable, Equatable, Sendable {
    struct LastPrompt: Codable, Equatable, Sendable {
        let tokens: Int
        let seconds: Double
    }

    let state: String
    var name: String? = nil
    var nodeId: String? = nil
    var link: String? = nil
    var held: Int? = nil
    var layers: Int? = nil
    var reason: String? = nil
    var path: String? = nil
    var bytes: Int64? = nil
    var free: Int64? = nil
    var done: Int64? = nil
    var secondsLeft: Int? = nil
    var onBattery: Bool? = nil
    var lastPrompt: LastPrompt? = nil
    /// ready/working: a prompt went past the helper's GPU memory and ran here.
    var outOfMemory: Bool? = nil
    /// needs_files only: make_room or low_space, the disk left after copying
    /// and the older copies make_room deletes first.
    var prompt: String? = nil
    var left: Int64? = nil
    var delete: [Copy]? = nil

    struct Copy: Codable, Equatable, Sendable {
        let name: String
        let bytes: Int64
    }
}

/// GET /admin/api/prefill-helper/candidates?model_id=: paired Macs and
/// whether each can help the model.
struct PrefillHelperCandidatesDTO: Codable, Equatable, Sendable {
    struct Device: Codable, Equatable, Sendable, Identifiable {
        let nodeId: String
        let name: String
        var chip: String? = nil
        var ramGb: Double? = nil
        let online: Bool
        var reason: String? = nil
        var link: String? = nil
        var version: String? = nil
        var versionOk: Bool? = nil
        var layersFit: Int? = nil
        var onBattery: Bool? = nil
        /// What the Mac helps now, when it is another model.
        var lending: Lending? = nil

        struct Lending: Codable, Equatable, Sendable {
            var host: String? = nil
            var model: String? = nil
        }

        var id: String { nodeId }
        var usable: Bool { online && versionOk == true && (layersFit ?? 0) > 0 }
    }

    let version: String
    let devices: [Device]
}

/// GET /admin/api/prefill-helper/lending: the helper this Mac runs for
/// another Mac, if any.
struct PrefillLendingDTO: Codable, Equatable, Sendable {
    let active: Bool
    var host: String? = nil
    var model: String? = nil
    var layers: Int? = nil
    var activeGib: Double? = nil
}
