# oMLX Provider、整合DMG与Sidecar发布路线 v2

> 状态：U0–U4按2026-09-22历史报告保留；**2026-11-19 Sidecar 集成已落地至主仓库 `omlx` `v0.8.0`**（[`omlx/omlx/engine/m5_sidecar.py`](../../omlx/omlx/engine/m5_sidecar.py) + [`miv/apps/mseries_backendd.cpp`](../../miv/apps/mseries_backendd.cpp)，采用 `EXTERNAL_ENGINES["m5rt"]`、`omlx.external_engines` 插件发现及 Unix Domain Socket 多连接 push-streaming）；U5A/U5B（签名与公证 DMG 打包）仍待后续实机发布闭环。
> 日期：2026-09-22（2026-11-19 同步状态）
> 当前权威：实施计划 `v6.0`（[`omlx-m5-backend-nlnfer-项目实施计划书.md`](../../omlx-m5-backend-nlnfer-项目实施计划书.md)，历史工作包表归档于 [`pre-post-m5ultra-implementation-plan-v5.1.md`](../versions/v0.1-v5.1-pre-silicon/plans/pre-post-m5ultra-implementation-plan-v5.1.md)）；联调与实机验收见 [`m5-ultra-arrival-runbook.md`](../../m5-ultra-arrival-runbook.md)；本文件保留既定 Provider/DMG 合同。

## 1. 结论

采用两阶段交付：

1. **整合DMG阶段**：从最新版oMLX重新建立集成分支，把通用Provider层、M-Series
   Provider、sidecar、UI和生命周期管理一起打入自有签名DMG。普通用户仍然只需拖入
   Applications并在UI里启用；
2. **官方插件阶段**：向oMLX上游提交不含M-Series专用kernel/模型逻辑的通用Provider SPI。
   上游合并并发布后，停止要求用户安装整套fork DMG，只发布独立签名sidecar DMG。

不采用“安装后修改`/Applications/oMLX.app`内部Python环境”的方案。官方App使用内嵌
venvstacks、代码签名和应用内更新；后改site-packages会破坏签名、版本一致性和升级路径。

## 2. 当前事实与边界

截至U4完成后的事实：

- 官方oMLX上游没有正式backend plugin规范、provider entry point或插件manifest；
- 集成worktree已经实现通用Provider SPI，但它仍是本项目代码，尚未进入官方release；
- 社区PR中出现过“建立plugin system”的建议，但尚未成为上游接口；
- macOS App会在App启动后按`auto_start_on_launch`启动server，并监督server崩溃重启；
- U4已用`SMAppService.mainApp`增加登录启动，同时保留`auto_start_on_launch`作为独立的
  server开关；
- 稳定集成worktree为`/Users/luochuhan/repo/omlx-provider-integration`，分支
  `codex/mseries-provider-integration-v064`，基线为官方`v0.6.4@1d782618`；
- 2026-09-22期间`github.com/jundot/omlx`、API与raw入口均返回404，GitHub整体状态正常；
  因此当前不声称同步到最新`main`，也不使用第三方fork替代。

## 3. 产品架构

```text
oMLX.app
├── Swift/Web UI
│   ├── Provider安装、更新、启停、诊断
│   ├── 登录时启动App
│   └── 每模型 Auto / Stock / M-Series
├── oMLX Server
│   └── EngineProviderRegistry
│       ├── StockProvider
│       └── SidecarProviderProxy
└── ProviderRegistry
    └── 每个已加载Provider模型的SidecarSession
        └── mseries-backendd
        ├── m5rt NativeBackend
        ├── Metal libraries/kernels
        ├── artifact converter/cache
        └── DeviceProfile + dispatch table
```

### 3.1 为什么使用sidecar

- 不绑定官方DMG内的Python、MLX与nanobind ABI；
- oMLX升级时，只校验Provider SPI/RPC版本，不必把native模块装入只读venv；
- sidecar崩溃不直接破坏server进程；
- 当前B6的`load/submit/cancel/poll/stats/shutdown`可直接映射到Unix socket RPC；
- prompt与批量token event的IPC开销远小于权重计算；大权重、KV、GDN和workspace不会跨进程。

### 3.2 Provider SPI v1

上游PR只提交通用能力：

- `EngineProviderRegistry`；
- Provider manifest发现与版本检查；
- `probe / supports / create / health / stats / shutdown`；
- engine策略：`auto / stock / <provider-id>`；
- 不支持请求、启动失败和首token前失败的stock fallback；
- 首token后失败不重放；
- Provider状态与fallback reason管理API；
- fake sidecar与contract tests。

上游PR不包含：

- M-Series品牌、固定芯片表；
- `.miv`转换器；
- oQ/MXFP4 Metal kernel；
- DFlash2 companion实现；
- 我们的静态dispatch table。

这样可以降低上游把PR视为“为单一量化/硬件引入专用依赖”的风险。

## 4. 已执行的同步/稳定基线方法

没有在B7工作树中直接merge。实际执行结果：

1. B7历史冻结点为`675b58a`；v5.1当前根仓库为`fc9ffa5`加真实前向修复/诊断等未提交修改，U5A须按实际工作树manifest集成，不能只取旧B7点；
2. 因上游地址404，选择可验证的官方`v0.6.4`，不把最后观察但未抓取的main SHA作为代码基线；
3. 创建独立worktree和分支，不修改根仓库历史`omlx/`；
4. stock Python、Swift与Release Stage基线通过，并制作可恢复bundle/tar/wheel备份；
5. 按U1→U2→U3→U4分层加入SPI、sidecar、EnginePool/admin和UI；
6. stock-only、provider-auto/forced、首token故障边界、真实native AR与UI构建分别验证。

预计高冲突文件：

- `omlx/engine_pool.py`；
- `omlx/model_settings.py`、`model_profiles.py`；
- `omlx/admin/routes.py`与Swift DTO/Model Settings页面；
- `pyproject.toml`与venvstacks/DMG packaging；
- Server/App生命周期与自动更新。

不要用一次大merge解决这些冲突；每层都保留独立commit和回归结果。

## 5. 整合DMG阶段

### 5.1 分发身份

自有DMG必须使用：

- 独立bundle identifier；
- 独立Developer ID签名与notarization；
- 独立版本号、更新feed和下载域名；
- Apache-2.0及第三方NOTICE；
- 与官方oMLX并存/迁移规则。

不能继续使用官方更新feed，否则官方升级可能覆盖Provider、UI和sidecar。若暂时没有Apple
Developer ID，只能发布明确标注的开发测试包，不能把ad-hoc签名包当普通用户正式版本。

### 5.2 用户流程

```text
下载DMG → 拖入Applications → 首次向导
       → 检测芯片/内存/Metal能力
       → 选择模型目录
       → 自动发现或生成.miv
       → Backend默认Auto
       → 启动server与sidecar
```

全局UI：

- 登录时启动oMLX；
- oMLX启动时启动Provider；
- Provider启用/停用/重启；
- 版本、兼容性、签名和健康状态；
- 自动更新与回滚；
- artifact缓存目录、占用、验证和删除；
- 导出脱敏诊断。

每模型UI：

- `Auto / Stock / M-Series`；
- `AR / DFlash2`；
- artifact未生成、转换中、可用、损坏、版本过期状态；
- 当前实际route、T/M/K和fallback reason；
- 一键正确性测试和基准；
- unsupported请求自动stock fallback。

### 5.3 启动与故障语义

```text
macOS登录
  → SMAppService启动oMLX.app
  → auto_start_on_launch启动oMLX server
  → ProviderRegistry验证manifest并显示installed状态
  → 模型首次请求/Pin时启动其mseries-backendd并加载权重
  → Probe操作只启动短生命周期sidecar完成握手后退出
```

- sidecar未启动：先重启sidecar，不加载模型；
- ABI/RPC不兼容：禁用Provider并提示更新，stock继续可用；
- sidecar崩溃：限次退避重启，避免crash loop；
- 请求首token前失败：stock fallback；
- 首token后失败：报错，不重放；
- 用户停用Provider：先停止接收、排空/取消请求、卸载权重、退出进程；
- oMLX升级后首次启动：先跑兼容握手，再允许Auto选择Provider。

“每次都能使用”指App和server按各自开关启动、Provider manifest每次被发现；sidecar与大模型
权重按请求或Pin策略加载。这样不在登录时保留无模型的空进程，也不无条件占满统一内存。

## 6. 官方Sidecar阶段

当Provider SPI进入上游release后：

1. 官方oMLX DMG负责Provider发现、UI、supervisor与stock fallback；
2. 我们只发布`M-Series Native Backend.dmg`；
3. DMG安装签名sidecar与manifest到用户级Application Support目录；
4. 官方oMLX通过Provider SPI连接，不修改App bundle；
5. sidecar使用独立更新feed；
6. 整合DMG进入维护期，最终停止新增功能。

上游合并不是可控Gate。如果Provider SPI未合并或未进入已发布的官方DMG，未修改的
官方oMLX就不会发现sidecar，此时不能转入“只发布sidecar”。项目必须继续维护
自有整合DMG，或等待上游接口实际落地，不得依赖尚未发布的PR。

建议插件位置：

```text
~/Library/Application Support/oMLX/Providers/com.omlx.mseries/
├── provider.json
├── bin/mseries-backendd
├── metallib/
└── receipts/
```

manifest必须包含`provider_id`、provider/RPC版本、支持的oMLX范围、最低macOS/Metal版本、
芯片profile、可执行文件hash、签名Team ID、安装来源与更新feed。

## 7. 实施阶段与Gate

| 阶段 | 交付 | Gate |
|---|---|---|
| ✅ U0 | 官方v0.6.4基线、独立worktree、可恢复备份 | stock Python/Swift/Stage通过；已完成 |
| ✅ U1 | 通用Provider SPI与fake provider | 不含M-Series逻辑；合同测试通过 |
| ✅ U2 | `mseries_backendd`、异步RPC、ProviderEngine | fake/native AR/DFlash2与首token边界通过 |
| ✅ U3 | Provider设置、EnginePool、artifact/admin API | 迁移、reload、disable unload、stock fallback通过 |
| ✅ U4 | Swift/Web UI与登录启动 | 原生/Web管理、路由解释、SMAppService、构建测试通过 |
| ⬜ U5A | 到货前整合验证DMG | 当前机Release Stage、安装/升级/回滚和故障注入通过 |
| ⬜ H0–H2 | M5 Ultra实测、route/tile晋级、长稳 | 不把M4候选冒充M5结论 |
| ⬜ U5B | 最终调优签名/公证DMG | clean Mac拖拽、签名、公证与最终性能Gate通过 |
| ⏸ U6 | 上游通用SPI PR | 上游恢复后提交；PR只含通用SPI、文档和测试 |
| ⏸ U7 | 官方release适配 | SPI进入官方DMG后，官方DMG + sidecar DMG通过 |

## 8. 当前执行边界

- 产品化下一步仍是U5A；实施v5.1已授权的P01–P12与其协调，包内只显示实际通过的功能，不把历史B7资产等同最新生产资格；
- 不在U5A阶段声称M5 Ultra性能，不提前写最终dispatch表；
- 不修改已安装官方App，不push，不使用第三方fork替代上游；
- U6只准备可审查patch，仓库恢复前不实际提交；
- U7不以PR合并替代正式官方DMG验证；
- M0–M9继续延期。

U0–U4的实现与验证证据见[`u0-u4-completion-report.md`](../versions/v0.1-v5.1-pre-silicon/b0-b7-u0-u4-reports/u0-u4-completion-report.md)。
