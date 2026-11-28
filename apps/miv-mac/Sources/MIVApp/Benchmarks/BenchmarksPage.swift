import AppKit
import SwiftUI

struct BenchmarksPage: View {
    let section: AppSection
    @ObservedObject var service: CoreService
    @ObservedObject private var queue = BenchmarkQueue.shared
    @State private var selection = ""
    @State private var lengths = "512, 2048"
    @State private var output = "128"
    @State private var slots = 1
    @State private var limit = "32768"
    @State private var samples = "100"
    @State private var counts: [String:String] = [:]
    @State private var thinking = false
    @State private var chosen: Set<String> = ["mmlu"]
    @State private var message = ""
    private var kind: String { section == .accuracyBench ? "accuracy" : section == .contextBench ? "context" : "throughput" }
    static let tests = [("知识",["mmlu","mmlu_pro","kmmlu","cmmlu","jmmlu"]),("推理",["hellaswag","truthfulqa","arc_challenge","winogrande"]),("数学",["gsm8k","mathqa"]),("代码",["humaneval","mbpp","livecodebench"]),("安全",["bbq","safetybench"])]
    private var current: CoreModel? { service.models.first(where: { $0.id == selection }) }
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader("配置", subtitle: kind == "context" ? "发送真实预填充请求，验证当前实例能够处理的最大上下文。" : nil)
            ListGroup {
                Row(label: "模型", sublabel: "已加载的模型可以直接测试。") {
                    Picker("模型", selection: $selection) {
                        Text("选择模型…").tag("")
                        ForEach(service.models) { Text($0.id).tag($0.id) }
                    }.labelsHidden().frame(maxWidth: 280)
                }
                if kind == "context" {
                    box("最大测试上下文", "输入与输出 token 总数。", $limit)
                } else {
                    Row(label: "批大小") { Stepper("\(slots)", value: $slots, in: 1...max(1,min(8,current?.maxConcurrency ?? 1))).fixedSize() }
                    if kind == "throughput" {
                        box("输入长度", "以逗号分隔。每次使用独立 token 序列。", $lengths)
                        box("生成长度", "每个请求的最大输出 tokens。", $output)
                    } else {
                        box("每个评测集的样本数", "0 表示完整数据集。", $samples)
                        Row(label: "扩展思考", isLast: true) { RowSwitch(isOn: $thinking) }
                    }
                }
            }
            if kind == "accuracy" {
                SectionHeader("评测集")
                ListGroup {
                    ForEach(Self.tests, id: \.0) { group, ids in
                        Row(label: group) {
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: 115))], alignment: .leading) {
                                ForEach(ids, id: \.self) { id in
                                    Toggle(id.uppercased(), isOn: Binding(get: { chosen.contains(id) }, set: { on in
                                        if on { chosen.insert(id) } else { chosen.remove(id) }
                                    })).toggleStyle(.checkbox)
                                    if chosen.contains(id) {
                                        TextField("样本数（继承）", text: Binding(get: { counts[id] ?? "" }, set: { counts[id] = $0 }))
                                            .accessibilityLabel(id + " 样本数").frame(maxWidth: 105)
                                    }
                                }
                            }.frame(maxWidth: 430, alignment: .leading)
                        }
                    }
                }
            }
            HStack { Text(message).font(.caption); Spacer(); Button("加入队列并运行") { enqueue() }
                .buttonStyle(.omlx(.primary)).disabled(current == nil || (kind == "accuracy" && chosen.isEmpty))
            }.padding(14)
            SectionHeader("队列")
            ListGroup {
                if let active = queue.active {
                    Row(label: active.model, sublabel: queue.progress) { ProgressView().controlSize(.small); Button("取消") { queue.cancel() } }
                }
                ForEach(queue.waiting) { job in
                    Row(label: job.model, sublabel: jobName(job.kind)) { Button("移除") { queue.remove(job.id) } }
                }
                if queue.active == nil {
                    Row(label: queue.waiting.isEmpty ? "没有正在运行的测试" : "等待继续", sublabel: queue.error, isLast: true) {
                        if !queue.waiting.isEmpty { Button("继续队列") { queue.resume(service) } }
                    }
                }
            }
            SectionHeader("结果") {
                if queue.results.contains(where: { $0.job.kind == kind }) { Button("清除结果") { queue.clear(kind) } }
            }
            ListGroup {
                let records = queue.results.filter { $0.job.kind == kind }
                if records.isEmpty { Row(label: "暂无结果", isLast: true) { EmptyView() } }
                ForEach(records) { record in
                    VStack(alignment: .leading, spacing: 7) {
                        Text(record.job.model).font(.headline)
                        Text(record.date.formatted()).font(.caption).foregroundStyle(.secondary)
                        let text = resultText(record)
                        Text(text).monospacedDigit().textSelection(.enabled)
                        HStack {
                            Button("复制结果") { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(text, forType: .string) }
                            if record.job.kind == "accuracy", FileManager.default.isReadableFile(atPath: queue.report(record.id).path) {
                                Button("导出逐题结果…") {
                                    let panel = NSSavePanel(); panel.nameFieldStringValue = "MIV-" + record.job.model.replacingOccurrences(of: "/", with: "-") + ".jsonl"
                                    if panel.runModal() == .OK, let url = panel.url {
                                        do { try Data(contentsOf: queue.report(record.id), options: .mappedIfSafe).write(to: url, options: .atomic) }
                                        catch { message = error.localizedDescription }
                                    }
                                }
                            }
                            if let n = object(record)["verified_context"] as? Int, service.settings.mode == .managed, record.job.base == service.browserURL?.absoluteString {
                                Button("应用到模型上下文") {
                                    do { try service.applyMeasuredContext(n, modelID: record.job.model); message = "已保存模型设置，下次加载时生效。" }
                                    catch { message = error.localizedDescription }
                                }
                            }
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(12)
                }
            }
        }
        .onAppear { pick() }
        .onChange(of: service.models.map(\.id)) { _, _ in pick() }
        .onChange(of: selection) { _, _ in slots = min(slots, max(1,current?.maxConcurrency ?? 1)) }
    }
    private func pick() { if current == nil { selection = service.models.first?.id ?? "" } }
    private func box(_ label: String, _ subtitle: String, _ value: Binding<String>) -> some View {
        Row(label: label, sublabel: subtitle) { TextField(label, text: value).labelsHidden().frame(width: 140) }
    }
    private func jobName(_ kind: String) -> String { kind == "accuracy" ? "准确率" : kind == "context" ? "上下文" : "吞吐量" }
    private func enqueue() {
        guard let base = service.browserURL?.absoluteString, let current else { return }
        var job = BenchmarkJob(kind: kind, model: current.id, base: base)
        job.batch = slots
        if kind == "throughput" {
            let parts = lengths.replacingOccurrences(of: "，", with: ",").components(separatedBy: ",")
            let values = parts.compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            guard !values.isEmpty, values.count == parts.count, values.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) }),
                  let n = Int(output), n > 0, n <= Int(Int32.max) else { message = "输入长度和生成长度须为正整数。"; return }
            job.lengths = values; job.output = n
        } else if kind == "accuracy" {
            guard let n = Int(samples), n >= 0, n <= Int(Int32.max) else { message = "请填写有效样本数。"; return }
            var perSuite: [String:Int] = [:]
            for (id, raw) in counts where chosen.contains(id) && !raw.isEmpty {
                guard let value = Int(raw), (0...Int(Int32.max)).contains(value) else { message = id + " 样本数无效。"; return }
                perSuite[id] = value
            }
            job.sampleCounts = perSuite; job.samples = n; job.suites = chosen.sorted(); job.thinking = thinking
        } else {
            guard let n = Int(limit), n > 1, n <= Int(Int32.max) else { message = "最大上下文须大于 1。"; return }
            job.limit = n
        }
        message = ""; queue.enqueue(job, service: service)
    }
    private func object(_ record: BenchmarkRecord) -> [String: Any] {
        (try? JSONSerialization.jsonObject(with: record.result)) as? [String: Any] ?? [:]
    }
    private func resultText(_ record: BenchmarkRecord) -> String {
        let payload = object(record)
        if let rows = payload["measurements"] as? [[String: Any]] {
            return rows.map { row in
                func v(_ key: String) -> String { String(describing: row[key] ?? "—") }
                return "输入 \(v("input")) · 并发 \(v("batch")) · 首 token \(v("ttft_ms")) ms\n预填充 \(v("prefill_tps")) tok/s · 解码 \(v("decode_tps")) tok/s · 端到端 \(v("output_tps")) tok/s"
            }.joined(separator: "\n\n")
        }
        if let scores = payload["scores"] as? [[String: Any]] {
            return scores.map { row in
                let correct = row["correct"] as? Int ?? 0, total = row["total"] as? Int ?? 0
                let pct = total > 0 ? 100 * Double(correct) / Double(total) : 0
                return "\((row["suite"] as? String ?? "").uppercased()) · \(correct) / \(total) · \(pct.formatted(.number.precision(.fractionLength(1))))%"
            }.joined(separator: "\n")
        }
        return "已验证上下文 \(payload["verified_context"] ?? "—") tokens · 实例上限 \(payload["configured_limit"] ?? "—")"
    }
}
