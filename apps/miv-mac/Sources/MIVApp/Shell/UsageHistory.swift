import SwiftUI

// Store deltas in hourly buckets. The public core reports input/cache totals
// globally, and requests/output per model, so never apportion input by guess.
struct UsageHour: Codable {
    var all = DashboardActivity.Usage()
    var models: [String: DashboardActivity.Usage] = [:]
}

struct UsageHistory: View {
    let buckets: [Int: UsageHour]
    let recording: Bool
    @EnvironmentObject private var navigation: NavigationState
    @State private var range = "today"
    @State private var model = ""

    private let ranges = [("today", "今天"), ("yesterday", "昨天"), ("7d", "7 天"), ("30d", "30 天"), ("90d", "90 天"), ("month", "本月")]
    private var dates: (Date, Date) {
        let cal = Calendar.current
        let today = cal.startOfDay(for: Date())
        let end = cal.date(byAdding: .day, value: 1, to: today)!
        if range == "yesterday" { return (cal.date(byAdding: .day, value: -1, to: today)!, today) }
        if range == "month" { return (cal.dateInterval(of: .month, for: today)!.start, end) }
        let count = Int(range.dropLast()) ?? 1
        return (cal.date(byAdding: .day, value: 1 - count, to: today)!, end)
    }

    private var selection: [(date: Date, value: UsageHour)] {
        let (start, end) = dates
        return buckets.compactMap { hour, value in
            let date = Date(timeIntervalSince1970: Double(hour) * 3600)
            return date >= start && date < end ? (date, value) : nil
        }.sorted { $0.date < $1.date }
    }
    private func values(_ hour: UsageHour) -> DashboardActivity.Usage {
        model.isEmpty ? hour.all : hour.models[model] ?? .init()
    }

    var body: some View {
        let selected = selection
        let available = Set(buckets.values.flatMap { $0.models.keys }).sorted()
        let totals = selected.reduce(into: DashboardActivity.Usage()) { $0.add(values($1.value)) }
        VStack(alignment: .leading, spacing: 12) {
            SectionHeader("使用历史", subtitle: "按本机时间汇总 · 保留 400 天")
            HStack {
                Picker("时间范围", selection: $range) {
                    ForEach(ranges, id: \.0) { key, label in Text(label).tag(key) }
                }
                Picker("模型", selection: $model) {
                    Text("所有模型").tag("")
                    ForEach(available, id: \.self) { Text($0).tag($0) }
                }
            }.padding(.horizontal, 18)
            if !recording {
                HStack {
                    Text("历史记录已暂停。").foregroundStyle(.secondary)
                    Button("打开服务器设置") { navigation.section = .server }
                }.padding(.horizontal, 18)
            }
            HStack {
                tile("请求", totals.requests)
                if model.isEmpty { tile("输入 Tokens", totals.prompt) }
                tile("输出 Tokens", totals.output)
                if model.isEmpty { tile("缓存 Tokens", totals.cached) }
            }.padding(.horizontal, 18)
            heatmap(selected)
        }
        .onChange(of: available) { _, names in if !names.contains(model) { model = "" } }
    }

    private func tile(_ label: String, _ value: Double) -> some View {
        VStack(spacing: 4) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Text(CountFormat.count(Int(exactly: value) ?? 0)).font(.headline.monospacedDigit())
        }.frame(maxWidth: .infinity)
    }

    private func heatmap(_ selected: [(date: Date, value: UsageHour)]) -> some View {
        let cal = Calendar.current
        let (start, end) = dates
        let days = (0..<max(1, cal.dateComponents([.day], from: start, to: end).day ?? 1)).compactMap {
            cal.date(byAdding: .day, value: $0, to: start)
        }
        var hours: [Date: [Int: Double]] = [:]
        for row in selected {
            let use = values(row.value)
            hours[cal.startOfDay(for: row.date), default: [:]][cal.component(.hour, from: row.date), default: 0] += use.prompt + use.output
        }
        let peak = max(1, hours.values.flatMap { $0.values }.max() ?? 1)
        return ScrollView {
            LazyVStack(spacing: 3) {
                ForEach(days, id: \.self) { day in
                    HStack(spacing: 2) {
                        Text(day.formatted(.dateTime.month(.twoDigits).day(.twoDigits))).font(.system(size: 9).monospacedDigit()).frame(width: 40)
                        ForEach(0..<24, id: \.self) { hour in
                            let count = hours[day]?[hour] ?? 0
                            RoundedRectangle(cornerRadius: 2)
                                .fill(count > 0 ? Color.green.opacity(0.2 + 0.8 * sqrt(count / peak)) : Color.secondary.opacity(0.1))
                                .frame(height: 14)
                                .help("\(day.formatted(date: .abbreviated, time: .omitted)) \(hour):00 · \(Int(exactly: count) ?? 0) tokens")
                        }
                    }
                }
            }
        }.frame(height: min(220, CGFloat(days.count * 17))).padding(.horizontal, 18)
    }
}
