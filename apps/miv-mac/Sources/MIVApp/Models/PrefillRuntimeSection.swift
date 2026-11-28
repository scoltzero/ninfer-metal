import SwiftUI

struct PrefillRuntimeSection: View {
    let model: CoreModel

    var body: some View {
        let requested = model.prefillAneMode.flatMap(MIVANEMode.reported)
        let effective = model.prefillAneModeEffective.flatMap(MIVANEMode.reported)
        let reason = model.prefillAneInactiveReason ?? ""
        SectionHeader("ANE 运行状态")
        ListGroup {
            if requested == nil && effective == nil && model.prefillAneActive == nil && reason.isEmpty {
                Row(label: "运行状态", isLast: true) { Text("核心未报告") }
            } else {
                if let requested { Row(label: "本次加载模式") { Text(requested) } }
                Row(label: "实际生效模式") { Text(effective ?? "核心未报告") }
                Row(label: "协同状态", isLast: reason.isEmpty) {
                    Text(model.prefillAneActive.map { $0 ? "已激活（active）" : "未激活（idle）" } ?? "核心未报告")
                }
                if !reason.isEmpty {
                    Row(label: "未激活或回退原因", isLast: true) { Text(Self.reasonLabel(reason)).textSelection(.enabled) }
                }
            }
        }
    }

    private static func reasonLabel(_ reason: String) -> String {
        switch reason {
        case "moe_unsupported": "当前 MoE 模型不支持 ANE 协同"
        case "ane_unavailable": "ANE 不可用"
        case "chunk_tokens_unavailable": "没有可用的 ANE 分块长度"
        case "calibration_bypassed": "本次校准未启用 ANE"
        case "no_layers": "未找到可参与校准的模型层"
        case "gpu_suffix_unsupported": "剩余计算不支持当前 GPU 路径"
        case "no_valid_split": "没有可用的 GPU 和 ANE 计算划分"
        default: reason
        }
    }
}
