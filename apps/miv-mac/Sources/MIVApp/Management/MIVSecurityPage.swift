import AppKit
import SwiftUI

struct MIVSecurityPage: View {
    @ObservedObject var service: CoreService
    @State private var policy = MIVAccess()
    @State private var visible = false
    @State private var adding = false
    @State private var name = ""
    @State private var subKey = ""
    @State private var message = ""
    @State private var loadFailed = false
    @State private var confirmMode: String?
    @State private var revoking: MIVSubKey?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if service.settings.mode == .external {
                SectionHeader("连接认证", subtitle: "外部服务的主密钥和子密钥由服务所在机器管理。可在「服务器」更换本次连接凭证。")
            } else {
                SectionHeader("API 密钥", subtitle: "用于本机托管的 /v1 请求。")
                ListGroup {
                    Row(label: "主密钥", sublabel: "留空时继承所选 TOML 中的 server.api_key。") {
                        Group { if visible { TextField("主密钥", text: $policy.master) }
                            else { SecureField("主密钥", text: $policy.master) }
                        }.frame(maxWidth: 190)
                        Button { visible.toggle() } label: { Image(systemName: visible ? "eye.slash" : "eye") }.help("显示或隐藏密钥")
                        Button("生成") { policy.master = MIVManagement.generateKey() }
                        Button("复制") { copy(policy.master) }.disabled(policy.master.isEmpty)
                    }
                }
                SectionHeader("身份验证")
                ListGroup {
                    Row(label: "认证方式", sublabel: "推理免密允许生成和模型列表；完整关闭也会开放单模型详情。") {
                        Picker("认证方式", selection: Binding(get: { policy.mode }, set: { value in
                            if ["open","inference"].contains(value) { confirmMode = value }
                            else { policy.mode = value }
                        })) {
                            Text("继承配置").tag("inherit"); Text("要求密钥").tag("required")
                            Text("允许免密推理").tag("inference"); Text("关闭所有认证").tag("open")
                        }.labelsHidden().frame(width: 150)
                    }
                }
                SectionHeader("子密钥", subtitle: "为应用或用户分别发放，可单独撤销。保存后对下一次请求生效。")
                ListGroup {
                    ForEach(policy.keys) { item in
                        Row(label: item.name.isEmpty ? "未命名" : item.name,
                            sublabel: item.created.formatted(date: .abbreviated, time: .shortened) + " · …" + String(item.key.suffix(4))) {
                            Button("复制") { copy(item.key) }
                            Button("撤销") { revoking = item }
                        }
                    }
                    Row(label: policy.keys.isEmpty ? "尚无子密钥" : nil, isLast: true) {
                        Button("新建…") { name = ""; subKey = MIVManagement.generateKey(); adding = true }
                    }
                }
                HStack { Text(message).font(.caption); Spacer(); Button("保存") { save() }.buttonStyle(.omlx(.primary)).disabled(loadFailed) }.padding(14)
            }
        }
        .onAppear {
            do { policy = try MIVManagement.access(); loadFailed = false }
            catch { message = "无法读取密钥文件：" + error.localizedDescription; loadFailed = true }
        }
        .alert("允许免密访问？", isPresented: Binding(get: { confirmMode != nil }, set: { if !$0 { confirmMode = nil } })) {
            Button("取消", role: .cancel) {}
            Button("允许") { if let value = confirmMode { policy.mode = value }; confirmMode = nil }
        } message: { Text("能连接到此服务的客户端将可直接发起推理请求。") }
        .alert("撤销这枚子密钥？", isPresented: Binding(get: { revoking != nil }, set: { if !$0 { revoking = nil } })) {
            Button("取消", role: .cancel) {}
            Button("撤销", role: .destructive) { if let item = revoking { policy.keys.removeAll { $0.id == item.id }; save() } }
        }
        .sheet(isPresented: $adding) {
            VStack(alignment: .leading, spacing: 15) {
                Text("新建子密钥").font(.title2)
                TextField("名称", text: $name)
                HStack { SecureField("子密钥", text: $subKey); Button("复制") { copy(subKey) } }
                HStack { Spacer(); Button("取消") { adding = false }
                    Button("创建") {
                        let item = MIVSubKey(name: name, key: subKey)
                        policy.keys.append(item)
                        if save() { adding = false } else { policy.keys.removeAll { $0.id == item.id }; adding = false }
                    }.buttonStyle(.borderedProminent)
                }
            }.padding(22).frame(width: 390)
        }
    }
    @discardableResult private func save() -> Bool {
        do { try MIVManagement.write(policy); message = "安全设置已生效。"; Task { await service.refresh() }; return true }
        catch { message = error.localizedDescription; return false }
    }
    private func copy(_ value: String) { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(value, forType: .string) }
}
