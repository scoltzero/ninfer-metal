import Foundation

enum ThermalSeverity { case nominal, fair, serious, critical }

enum SystemMetricsPoller {
    nonisolated static func severity(for state: ProcessInfo.ThermalState) -> ThermalSeverity {
        switch state {
        case .nominal:  return .nominal
        case .fair:     return .fair
        case .serious:  return .serious
        case .critical: return .critical
        @unknown default: return .critical
        }
    }

    nonisolated static func label(for severity: ThermalSeverity) -> String {
        switch severity {
        case .nominal:
            return String(localized: "metrics.thermal.nominal",
                          defaultValue: "Nominal",
                          comment: "Thermal severity label shown in the Status screen's System block")
        case .fair:
            return String(localized: "metrics.thermal.fair",
                          defaultValue: "Fair",
                          comment: "Thermal severity label shown in the Status screen's System block")
        case .serious:
            return String(localized: "metrics.thermal.serious",
                          defaultValue: "Serious",
                          comment: "Thermal severity label shown in the Status screen's System block")
        case .critical:
            return String(localized: "metrics.thermal.critical",
                          defaultValue: "Critical",
                          comment: "Thermal severity label shown in the Status screen's System block")
        }
    }

    /// Format bytes as a 1-decimal GB string ("34.6"). Uses decimal GB
    /// (10^9) to match how macOS reports system memory in About This Mac
    /// — using GiB here would print 32 GB machines as "29.8".
    nonisolated static func formatBytesAsGB(_ bytes: UInt64) -> String {
        let gb = Double(bytes) / 1_000_000_000.0
        return String(format: "%.1f", gb)
    }

    /// Format binary memory units for the macOS system-memory panel.
    nonisolated static func formatBytesAsGiB(_ bytes: UInt64) -> String {
        let gib = Double(bytes) / 1_073_741_824.0
        return String(format: "%.1f", gib)
    }
}
