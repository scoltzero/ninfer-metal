// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "MIV",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "MIVApp", targets: ["MIVApp"])],
    targets: [
        .executableTarget(name: "MIVApp"),
        .testTarget(name: "MIVAppTests", dependencies: ["MIVApp"])
    ]
)
