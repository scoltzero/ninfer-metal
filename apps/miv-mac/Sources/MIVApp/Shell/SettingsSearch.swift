import Foundation

struct SettingsSearchEntry: Identifiable {
    let section: AppSection
    let label: String
    let aliases: String
    var id: String { section.rawValue + ":" + label }
    func title() -> String { label }
    func breadcrumb() -> String { section.title }
}

enum SettingsSearch {
    static func results(for query: String) -> [SettingsSearchEntry] {
        let terms = query.split(whereSeparator: \.isWhitespace)
        guard !terms.isEmpty else { return [] }
        let pages = AppSection.allCases.map { SettingsSearchEntry(section: $0, label: $0.title, aliases: $0.rawValue) }
        return (pages + fields).filter { entry in
            let text = [entry.section.title, entry.section.rawValue, entry.label, entry.aliases].joined(separator: " ")
            return terms.allSatisfy { text.localizedCaseInsensitiveContains(String($0)) }
        }
    }

    private static let fields: [SettingsSearchEntry] = [
        .init(section: .status, label: "服务统计", aliases: "tokens requests activity usage cache 命中率 吞吐 活动模型"),
        .init(section: .status, label: "系统", aliases: "GPU CPU RAM 内存 占用"),
        .init(section: .server, label: "模型存储", aliases: "模型目录 模型库 路径 library directory storage"),
        .init(section: .server, label: "共享配置", aliases: "toml 打开 配置文件 config"),
        .init(section: .network, label: "服务地址", aliases: "endpoint host 连接 局域网 LAN 端口 port"),
        .init(section: .performance, label: "推理设置", aliases: "调度 上下文 并发 max context concurrency batch 缓存 KV MTP DFlash2 推测解码"),
        .init(section: .appearance, label: "刷新间隔", aliases: "refresh interval 频率 菜单栏"),
        .init(section: .appearance, label: "显示 Dock 图标", aliases: "dock icon 程序坞"),
        .init(section: .appearance, label: "恢复菜单栏图标", aliases: "menu bar restore"),
        .init(section: .appearance, label: "数字显示", aliases: "number scale compact exact 万"),
        .init(section: .appearance, label: "增强可读性", aliases: "contrast readability 对比度"),
        .init(section: .appearance, label: "菜单栏项目", aliases: "CPU GPU MEM LIV AVG ALL 速度 内存 菜单栏 模型收藏"),
        .init(section: .logs, label: "服务日志", aliases: "level log log file 筛选 级别 表格 原始 复制"),
        .init(section: .models, label: "模型库", aliases: "模型设置 载入 加载 卸载 model settings profile"),
        .init(section: .downloads, label: "下载模型", aliases: "Hugging Face ModelScope 下载"),
        .init(section: .integrations, label: "应用集成", aliases: "API 应用 地址 clients"),
        .init(section: .quantization, label: "模型转换", aliases: "精度 量化 权重 safetensors miv"),
        .init(section: .throughputBench, label: "吞吐测试", aliases: "benchmark TTFT TPS prefill decode"),
        .init(section: .accuracyBench, label: "准确度测试", aliases: "accuracy quality 准确率"),
        .init(section: .contextBench, label: "上下文测试", aliases: "context length 长文本"),
        .init(section: .security, label: "API 密钥", aliases: "token authentication key 密码 认证"),
        .init(section: .about, label: "版本和许可证", aliases: "license release 版权 GitHub")
    ]
}
