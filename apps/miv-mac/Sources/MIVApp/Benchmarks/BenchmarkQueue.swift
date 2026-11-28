import Foundation
import Combine

struct BenchmarkJob: Codable, Identifiable {
    var id = UUID()
    var kind: String
    var model: String
    var base: String
    var lengths = [512, 2048]
    var output = 128
    var batch = 1
    var suites = ["mmlu"]
    var sampleCounts: [String: Int]?
    var samples = 100
    var thinking = false
    var limit = 32768
}

struct BenchmarkRecord: Codable, Identifiable {
    var job: BenchmarkJob
    var result: Data
    var date = Date()
    var id: UUID { job.id }
}

@MainActor
final class BenchmarkQueue: ObservableObject {
    static let shared = BenchmarkQueue()
    @Published private(set) var waiting: [BenchmarkJob] = []
    @Published private(set) var active: BenchmarkJob?
    @Published private(set) var results: [BenchmarkRecord] = []
    @Published private(set) var progress = ""
    @Published private(set) var error: String?
    private var closing = false
    private var process: Process?
    private var task: Task<Void, Never>?
    private let defaults: UserDefaults
    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        let d = JSONDecoder()
        waiting = defaults.data(forKey: "bench.queue").flatMap { try? d.decode([BenchmarkJob].self, from: $0) } ?? []
        results = defaults.data(forKey: "bench.results").flatMap { try? d.decode([BenchmarkRecord].self, from: $0) } ?? []
    }
    private func save() {
        defaults.set(try? JSONEncoder().encode((active.map { [$0] } ?? []) + waiting), forKey: "bench.queue")
        defaults.set(try? JSONEncoder().encode(results), forKey: "bench.results")
    }
    func enqueue(_ job: BenchmarkJob, service: CoreService) {
        waiting.append(job); save(); resume(service)
    }
    func remove(_ id: UUID) { waiting.removeAll { $0.id == id }; save() }
    func report(_ id: UUID) -> URL { MIVManagement.root.appendingPathComponent("Benchmarks/").appendingPathComponent(id.uuidString + ".jsonl") }
    func clear(_ kind: String) {
        for record in results where record.job.kind == kind { try? FileManager.default.removeItem(at: report(record.id)) }
        results.removeAll { $0.job.kind == kind }; save()
    }
    func shutdown() {
        closing = true
        save()
        task?.cancel()
        if let process, process.isRunning { process.terminate() }
    }
    func cancel() {
        task?.cancel()
        if let process, process.isRunning { process.terminate() }
    }
    func resume(_ service: CoreService) {
        guard task == nil, let job = waiting.first else { return }
        guard service.browserURL?.absoluteString == job.base, service.models.contains(where: { $0.id == job.model }) else {
            error = "队首模型尚未连接，请连接对应服务并加载模型后继续。"; return
        }
        waiting.removeFirst(); active = job; progress = "准备中…"; error = nil; save()
        task = Task {
            do {
                let key = service.apiKey.isEmpty && service.settings.mode == .managed ? MIVManagement.localMasterKey() : service.apiKey
                let result = try await execute(job, key: key)
                try Task.checkCancellation()
                results.insert(BenchmarkRecord(job: job, result: result), at: 0)
            } catch {
                if job.kind == "accuracy" { try? FileManager.default.removeItem(at: report(job.id)) }
                self.error = Task.isCancelled ? "已取消本次测试。" : error.localizedDescription
            }
            process = nil; active = nil; task = nil; progress = ""
            guard !closing else { return }; save()
            if self.error == nil { resume(service) }
        }
    }

    private func execute(_ job: BenchmarkJob, key: String) async throws -> Data {
        guard let res = Bundle.main.resourceURL else { throw CoreFailure("找不到应用资源。") }
        let python = res.appendingPathComponent("ModelTools/Python")
        let proc = Process(), input = Pipe(), output = Pipe()
        proc.executableURL = python.appendingPathComponent("cpython-3.11/bin/python3")
        proc.arguments = ["-u", res.appendingPathComponent("BenchTools/benchmark.py").path]
        var env = MIVManagement.workerEnvironment()
        env["PYTHONHOME"] = python.appendingPathComponent("cpython-3.11").path
        env["PYTHONPATH"] = python.appendingPathComponent("framework-mlx-base/lib/python3.11/site-packages").path
        env["PYTHONNOUSERSITE"] = "1"; env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc.environment = env
        proc.standardInput = input; proc.standardOutput = output; proc.standardError = output
        guard var fields = try JSONSerialization.jsonObject(with: JSONEncoder().encode(job)) as? [String: Any] else {
            throw CoreFailure("评测任务参数无效。")
        }
        let detail = report(job.id)
        if job.kind == "accuracy" {
            try FileManager.default.createDirectory(at: detail.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            if FileManager.default.fileExists(atPath: detail.path) { try FileManager.default.removeItem(at: detail) }
            fields["detailPath"] = detail.path
        }
        fields["key"] = key
        let data = try JSONSerialization.data(withJSONObject: fields)
        try proc.run(); process = proc
        defer {
            if proc.isRunning { proc.terminate() }
            try? input.fileHandleForWriting.close()
            try? output.fileHandleForReading.close()
        }
        try input.fileHandleForWriting.write(contentsOf: data)
        try input.fileHandleForWriting.close()
        var result: Data?, failure: String?
        for try await line in output.fileHandleForReading.bytes.lines {
            try Task.checkCancellation()
            guard let e = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] else {
                if !line.isEmpty { failure = String(line.prefix(400)) }; continue
            }
            if let text = e["progress"] as? String { progress = text }
            if let text = e["error"] as? String { failure = text }
            if let payload = e["result"] { result = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]) }
        }
        proc.waitUntilExit()
        guard proc.terminationStatus == 0, let result else { throw CoreFailure(failure ?? "评测进程已退出（\(proc.terminationStatus)）。") }
        return result
    }
}
