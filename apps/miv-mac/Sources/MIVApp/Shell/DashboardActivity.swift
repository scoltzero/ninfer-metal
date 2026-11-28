import SwiftUI

// The standalone core exports monotonic token counters. Speeds here are
// observed counter deltas per wall second, including idle time.
@MainActor final class DashboardActivity: ObservableObject {
    struct Usage: Codable, Equatable {
        var prompt = 0.0, output = 0.0, cached = 0.0, requests = 0.0
        init() {}
        init(_ metrics: [String: Double]) {
            prompt = metrics["ninfer_metal_prompt_tokens_total"] ?? 0
            output = metrics["ninfer_metal_generated_tokens_total"] ?? 0
            cached = metrics["ninfer_metal_prefix_cache_restored_tokens_total"] ?? 0
            requests = metrics["ninfer_metal_requests_total"] ?? 0
        }
        mutating func add(_ values: Usage) {
            prompt += values.prompt; output += values.output; cached += values.cached; requests += values.requests
        }
        func since(_ previous: Usage) -> Usage {
            var delta = Usage()
            delta.prompt = max(0, prompt - previous.prompt)
            delta.output = max(0, output - previous.output)
            delta.cached = max(0, cached - previous.cached)
            delta.requests = max(0, requests - previous.requests)
            return delta
        }
    }

    struct Record: Codable {
        var usage = Usage()
        var seconds = 0.0
        var rates: MetricRates? {
            guard seconds > 0 else { return nil }
            return MetricRates(promptTps: usage.prompt / seconds, generationTps: usage.output / seconds)
        }
    }

    @Published private(set) var system = SystemStatsSnapshot()
    @Published private(set) var served = Usage()
    private var baseline = Usage()
    @Published private(set) var session = Record()
    @Published private(set) var lifetime = Record()
    @Published private(set) var live: MetricRates?
    let graph = MenubarMetricsStore()
    private let sampler = SystemStatsSampler()
    private let defaults: UserDefaults
    private var records: [String: Record]
    private var origin: String?
    private var previous: (TimeInterval, Usage)?
    @Published private(set) var savedHours: [Int: UsageHour] = [:]
    @Published private(set) var recording = true
    private var hourlyByOrigin: [String: [Int: UsageHour]]
    private var lastModelCounters: [String: (requests: Double, output: Double)] = [:]
    private var historyDirty = false
    private var lastSave = 0.0
    private static let key = "dashboard.observedUsage.v1"

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        hourlyByOrigin = defaults.data(forKey: "dashboard.hours.v1").flatMap { try? JSONDecoder().decode([String:[Int:UsageHour]].self, from: $0) } ?? [:]
        let oldestHour = Int(Date().timeIntervalSince1970 / 3600) - 400 * 24
        hourlyByOrigin = hourlyByOrigin.mapValues { $0.filter { $0.key >= oldestHour } }
        records = defaults.data(forKey: Self.key).flatMap { try? JSONDecoder().decode([String: Record].self, from: $0) } ?? [:]
    }

    func sample(service: CoreService) {
        system = sampler.sample()
        graph.dashboardAvailable = service.dashboardAvailable
        observe(endpoint: service.browserURL?.absoluteString, running: service.health != nil, metrics: service.metrics,
                modelCounters: service.modelUsageCounters, record: MIVManagement.settings.recordUsage,
                now: ProcessInfo.processInfo.systemUptime)
    }

    func observe(endpoint next: String?, running: Bool, metrics: [String: Double], modelCounters: CoreConnection.UsageCounters,
                 record: Bool, now: TimeInterval) {
        let wasRecording = recording
        recording = record
        if origin != next {
            save()
            origin = next
            session = Record()
            baseline = Usage()
            served = Usage()
            lifetime = next.flatMap { records[$0] } ?? Record()
            previous = nil
            lastModelCounters = [:]
            savedHours = next.flatMap { hourlyByOrigin[$0] } ?? [:]
            graph.resetHistory()
        }
        guard let origin, running, metrics["ninfer_metal_prompt_tokens_total"] != nil else {
            live = nil
            previous = nil
            graph.markServerStopped()
            return
        }
        let sample = Usage(metrics)
        live = nil
        if let (before, counts) = previous, sample.prompt >= counts.prompt, sample.output >= counts.output,
           sample.requests >= counts.requests, now > before, now - before < 20 {
            let delta = sample.since(counts)
            let seconds = now - before
            session.usage.add(delta)
            session.seconds += seconds
            if recording && wasRecording {
                lifetime.usage.add(delta)
                lifetime.seconds += seconds
                records[origin] = lifetime
                if delta != Usage() {
                    let hour = Int(Date().timeIntervalSince1970 / 3600)
                    var bucket = savedHours[hour] ?? UsageHour()
                    bucket.all.add(delta)
                    for (model, counters) in modelCounters {
                        guard let before = lastModelCounters[model] else { continue }
                        var modelDelta = Usage()
                        modelDelta.requests = max(0, counters.requests - before.requests)
                        modelDelta.output = max(0, counters.output - before.output)
                        if modelDelta != Usage() { bucket.models[model, default: Usage()].add(modelDelta) }
                    }
                    if savedHours[hour] == nil { savedHours = savedHours.filter { $0.key >= hour - 400 * 24 } }
                    savedHours[hour] = bucket
                    historyDirty = true
                }
            }
            live = MetricRates(promptTps: delta.prompt / seconds, generationTps: delta.output / seconds)
            if now - lastSave > 30 { save(); lastSave = now }
        } else if let (_, counts) = previous, sample.prompt < counts.prompt || sample.output < counts.output {
            session = Record()
        }
        if sample.requests < baseline.requests || sample.output < baseline.output || sample.prompt < baseline.prompt {
            baseline = Usage()
        }
        served = sample.since(baseline)
        lastModelCounters = modelCounters
        previous = (now, sample)
        graph.applyTick(live: live, average: session.rates, alltime: lifetime.rates, serverRunning: true)
    }

    func clear(allTime: Bool) {
        if allTime {
            lifetime = Record()
            savedHours = [:]
            historyDirty = true
            if let origin { records[origin] = lifetime }
            save()
        } else {
            baseline = previous?.1 ?? Usage()
            served = Usage()
            session = Record()
        }
    }

    func save() {
        if historyDirty, let origin {
            hourlyByOrigin[origin] = savedHours
            if let data = try? JSONEncoder().encode(hourlyByOrigin) {
                defaults.set(data, forKey: "dashboard.hours.v1")
                historyDirty = false
            }
        }
        if let data = try? JSONEncoder().encode(records) { defaults.set(data, forKey: Self.key) }
    }
}
