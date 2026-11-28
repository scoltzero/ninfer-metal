import Foundation

enum MIVANEMode: String, CaseIterable {
    case off, auto
    case compactInt8 = "compact_int8"
    case ultraFP16SwiGLU = "ultra_fp16_swiglu"

    static let key = "prefill_ane_mode"
    static let legacyKey = "prefill_ane_offload"

    var label: String {
        switch self {
        case .off: "关闭（仅 GPU）"
        case .auto: "自动选择"
        case .compactInt8: "紧凑 INT8"
        case .ultraFP16SwiGLU: "Ultra FP16 SwiGLU"
        }
    }

    // An explicit mode takes precedence, including an empty selection for inheritance.
    static func selection(in engine: [String: String]) -> String? {
        if let mode = engine[key] { return mode.isEmpty ? nil : mode }
        guard let old = engine[legacyKey], !old.isEmpty else { return nil }
        switch old {
        case "true", "on": return MIVANEMode.auto.rawValue
        case "false", "off": return MIVANEMode.off.rawValue
        default: return old
        }
    }

    static func reported(_ raw: String) -> String? {
        raw.isEmpty ? nil : (MIVANEMode(rawValue: raw)?.label ?? raw)
    }
}

struct ModelField: Identifiable {
    enum Kind {
        case integer(ClosedRange<Int>), number, choice([(String, String)]), text, file, directory
    }
    let id, label, detail: String
    let kind: Kind
    init(_ id: String, _ label: String, _ detail: String, _ kind: Kind) {
        self.id = id; self.label = label; self.detail = detail; self.kind = kind
    }
    static let switchValues: Kind = .choice([("true","开启"), ("false","关闭")])
    static let context = ModelField("max_context", "上下文窗口", "提示词和输出共享的最大 token 数。", .integer(1...Int(Int32.max)))
    static let concurrency = ModelField("max_concurrency", "并发槽位", "此模型可同时运行的请求数。", .integer(1...8))
    static let generation = [
        ModelField("max_tokens","最大输出","客户端未指定上限时，本次最多生成多少个 token。",.integer(1...Int(Int32.max))),
        ModelField("temperature","温度","0 使用贪心解码，较高的值增加随机性。",.number),
        ModelField("top_p","Top P","保留累计概率达到阈值的候选；范围 (0, 1]。",.number),
        ModelField("top_k","Top K","保留的高概率候选数，0 使用核心默认。",.integer(0...64)),
        ModelField("seed","随机种子","固定采样的随机数种子。",.integer(Int.min...Int.max)),
        ModelField("repetition_penalty","重复惩罚","大于 0；1 表示不调整重复 token 的分数。",.number),
        ModelField("presence_penalty","出现惩罚","按 token 是否已经出现调整分数。",.number),
        ModelField("frequency_penalty","频率惩罚","按 token 已出现的次数调整分数。",.number),
        ModelField("repetition_context_size","惩罚回看长度","用于统计重复 token 的最近上下文长度，最多 4096。",.integer(1...4096)),
        ModelField("enable_thinking","思考模式","通过对话模板的 enable_thinking 控制是否生成思考内容。",switchValues),
        ModelField("thinking_budget","思考预算","最多生成的思考 token 数，0 表示立即结束思考。",.integer(0...Int(Int32.max))),
        ModelField("stop","停止字符串","遇到此文本时结束；多个停止词可填 JSON 字符串数组。",.text),
        ModelField("reasoning_effort","推理强度","将该值提供给对话模板；可用档位取决于模型模板。",.text),
        ModelField("chat_template_kwargs","自定义模板参数","填入 JSON 对象。请求传入的同名模板参数优先。",.text)
    ]
    static let acceleration = [
        ModelField("spec_backend","推测解码后端","自动模式优先发现 DFlash2 草稿，其次使用 MTP 头。",.choice([("auto","自动识别"),("none","关闭"),("mtp","MTP 多 token 预测"),("dflash2","DFlash2 草稿模型")])),
        ModelField("draft_tokens","每轮草稿 token 数","0 由核心选择；MTP 最多 4，DFlash2 最多 7。",.integer(0...7)),
        ModelField("draft_dir","DFlash2 草稿模型目录","选择与主模型匹配的草稿模型目录。",.directory),
        ModelField("draft_artifact","DFlash2 权重文件","指定要加载的草稿 .miv 文件。",.file),
        ModelField("mtp_artifact","MTP 预测头","指定独立的 MTP 预测头 .miv 文件。",.file)
    ]
    static let memory = [
        ModelField("kv_dtype","KV 缓存格式","BF16 保留原始精度；Affine 4 位压缩每组 128 个值。",.choice([("bf16","BF16"),("affine4_g128","Affine 4 位 · 128 个值一组")])),
        ModelField("rolling_prefix_cache","滚动前缀缓存","保存已经处理过的前缀状态供后续请求复用。",switchValues),
        ModelField("prefix_cache_gb","前缀缓存预算（GiB）","核心可用于可复用前缀的内存，范围 0 至 128。",.number),
        ModelField("memory_budget_gb","模型内存预算（GiB）","限制加载该模型允许使用的统一内存；0 采用自动预算。",.number)
    ]
    static let advanced = [
        ModelField("lora_adapter","LoRA 适配器目录","加载 Hugging Face PEFT 或 mlx-lm 的本地适配器。",.directory),
        ModelField("lora_mode","LoRA 执行方式","合并在加载时融合权重；旁路在运行时计算低秩残差。",.choice([("merge","加载时合并"),("bypass","运行时旁路")])),
        ModelField("chat_template","对话模板","为此模型指定 Jinja 对话模板文件。",.file)
    ]
    static let miv = [
        ModelField("artifact","主模型权重文件","覆盖自动发现的 model.miv 或 model.omlxm5。",.file),
        ModelField("metallib_dir","Metal 着色器目录","选择此模型加载的预编译 Metal 内核目录。",.directory),
        ModelField("prefill_linear_mode","预填充线性计算","W4A8 在预填充阶段量化激活，解码保持原线性计算路径。",.choice([("bf16","BF16 精确计算"),("w4a8","W4A8 激活量化")])),
        ModelField("prefill_attention_mode","预填充注意力","精确 NAX 使用原计算；Sage2 使用低位宽 QK 和 FP16 PV。",.choice([("exact_nax","精确 NAX"),("sage2","Sage2 低位注意力")])),
        ModelField("prefill_sparse_minference","MInference 稀疏注意力","配合 Sage2 跳过低权重的 KV 块，可改变输出。",switchValues),
        ModelField("prefill_cache_blend","CacheBlend 分段复用","尝试复用提示中分散匹配的 KV 和 GDN 状态。",switchValues),
        ModelField(MIVANEMode.key, "ANE 协同模式", "自动模式按模型和硬件选择。也可指定紧凑 INT8 或 Ultra FP16 SwiGLU；重新加载模型后生效。", .choice(MIVANEMode.allCases.map { ($0.rawValue, $0.label) })),
        ModelField("batch_mode","批处理","允许核心将兼容的并发请求组合到同一批次。",.choice([("on","开启"),("off","关闭")])),
        ModelField("sampling_contract","推测采样契约","位置耦合在 GPU 上校验；旧版拒绝采样用于对照调试。",.choice([("position_coupled_v1","位置耦合 v1"),("legacy_rejection_v1","旧版拒绝采样 v1")]))
    ]
    static let engine = [context, concurrency] + acceleration + memory + advanced + miv

    static func typed(_ input: [String: String], fields: [ModelField]) throws -> [String: Any] {
        var result: [String: Any] = [:]
        for field in fields {
            guard let raw = input[field.id], !raw.isEmpty else { continue }
            let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            let invalid = CoreFailure("「\(field.label)」的值无效。\(field.detail)")
            switch field.kind {
            case .integer(let range):
                guard let n = Int(text), range.contains(n) else { throw invalid }
                result[field.id] = n
            case .number:
                guard let n = Double(text), n.isFinite, abs(n) <= Double(Float.greatestFiniteMagnitude) else { throw invalid }
                switch field.id {
                case "temperature", "memory_budget_gb": guard n >= 0 else { throw invalid }
                case "top_p": guard n > 0, n <= 1 else { throw invalid }
                case "prefix_cache_gb": guard (0...128).contains(n) else { throw invalid }
                case "repetition_penalty": guard n > 0 else { throw invalid }
                default: break
                }
                result[field.id] = n
            case .choice(let options):
                guard options.contains(where: { $0.0 == text }) else { throw invalid }
                if let boolean = Bool(text) { result[field.id] = boolean }
                else { result[field.id] = text }
            case .file, .directory: result[field.id] = (text as NSString).expandingTildeInPath
            case .text:
                if field.id == "chat_template_kwargs" {
                    guard let object = (try? JSONSerialization.jsonObject(with: Data(raw.utf8))) as? [String:Any] else { throw invalid }
                    result[field.id] = object
                } else if field.id == "stop", raw.hasPrefix("[") {
                    guard let stops = try? JSONDecoder().decode([String].self, from: Data(raw.utf8)),
                          !stops.isEmpty, stops.allSatisfy({ !$0.isEmpty }) else { throw invalid }
                    result[field.id] = stops
                } else { result[field.id] = raw }
            }
        }
        if result["spec_backend"] as? String == "mtp", let depth = result["draft_tokens"] as? Int, depth > 4 {
            throw CoreFailure("MTP 每轮最多 4 个草稿 token。")
        }
        return result
    }
}
