# MIV for macOS

MIV 的 Swift 界面、菜单、图标、中文资源和工具源码均位于本目录。推理由仓库中的原生 `ninfer-metal-serve` 执行。模型下载、oQ 量化和评测使用随 App 打包的独立 Python 环境。两个 App 可以分别修改、构建和运行。

## 构建

在 Apple Silicon、macOS 27 和 Xcode 27 上安装 CMake、Ninja 和 uv，从仓库根目录运行：

```sh
./apps/miv-mac/build-app.sh
open dist/MIV.app
```

首次构建从本目录的运行时配方生成 `build/miv-python`。已有对应布局的可重定位 CPython 3.11 与依赖集时，可以用 `MIV_PYTHON_RUNTIME` 指定。发布包包含运行时，使用者不需要安装 Python 或 oMLX。

## 配置和数据

- 在「服务器」中选用端口、模型库和可选的外部 TOML。原生实例可通过「连接现有服务」查看状态和运行基准。
- 本机托管使用 MIV 自己的设置和应用支持目录。单模型的「MIV 核心」分栏覆盖选中模型的启动设置，留空继承。加载参数在模型重新加载后生效。请求中显式指定的生成参数优先于单模型和全局默认值。
- 量化页可执行 oQ 量化。MIVFP4 的独立位置已预留，当前没有可执行的 MIVFP4 转换器。
- 代码评测在单独的 macOS 沙箱内执行。准确率可分评测集选择样本数，支持导出逐题结果。上下文测试验证当前已加载实例的可用长度，结果可以保存到本机模型设置。
