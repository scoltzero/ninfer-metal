#!/bin/bash
# Build the native dashboard and its matching relocatable Metal server.
set -euo pipefail

project=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$project/../.." && pwd)
build="$repo/build/miv-app"
dist="$repo/dist"
runtime="${MIV_PYTHON_RUNTIME:-$repo/build/miv-python}"
"$project/Packaging/runtime/prepare.sh" "$runtime"
cmake -S "$repo" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$runtime/cpython-3.11/bin/python3" \
    -DOMLX_M5_BUILD_NAX_METALLIB=ON -DOMLX_M5_BUILD_MLX_LEGACY_SMOKE=OFF \
    -DOMLX_M5_BUILD_PYTHON=OFF -DOMLX_M5_BUILD_MLX_FIXTURES=OFF
cmake --build "$build" --target ninfer-metal-serve --parallel "${MIV_BUILD_JOBS:-6}"
swift build --package-path "$project" --configuration release --product MIVApp
swift_bin=$(swift build --package-path "$project" --configuration release --show-bin-path)

mkdir -p "$dist"
stage=$(mktemp -d "$dist/.miv-app.XXXXXX")
trap 'rm -rf "$stage"' EXIT
app="$stage/MIV.app"
res="$app/Contents/Resources"
core="$res/Core"
licenses="$res/Licenses"
icons="$stage/AppIcon.iconset"
mkdir -p "$app/Contents/MacOS" "$core/bin" "$core/metallib" "$core/config" "$licenses" "$icons"
install -m 755 "$swift_bin/MIVApp" "$app/Contents/MacOS/MIVApp"
install -m 755 "$build/miv/ninfer-metal-serve" "$core/bin/ninfer-metal-serve"
cp "$build/miv/metallib/"*.metallib "$core/metallib/"
cp "$project/Packaging/Core/ninfer-metal.toml" "$core/config/"
cp -R "$project/Packaging/Localizations/"*.lproj "$res/"
cp -R "$project/BenchTools" "$res/BenchTools"
"$project/ModelTools/stage.sh" "$res/ModelTools"
ditto "$runtime" "$res/ModelTools/Python"
find "$res" -type d -name __pycache__ -prune -exec rm -rf {} +
py="$res/ModelTools/Python"
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHOME="$py/cpython-3.11" \
    PYTHONPATH="$res/ModelTools:$res/BenchTools:$py/framework-mlx-base/lib/python3.11/site-packages" \
    "$py/cpython-3.11/bin/python3" -c 'import aiohttp, httpx, mlx.core, huggingface_hub, modelscope, miv_eval; assert len(miv_eval.BENCHMARKS) == 16'

cp "$project/Packaging/Info.plist" "$app/Contents/"
cp "$repo/LICENSE" "$licenses/MIV-GPL-3.0"
cp "$project/Packaging/oMLX-Apache-2.0" "$licenses/"
cp "$repo/miv/third_party/THIRD_PARTY_NOTICES.md" "$licenses/"
for notice in "$repo"/miv/third_party/*/{LICENSE*,NOTICE}; do
    [[ -f "$notice" ]] || continue
    component=$(basename "$(dirname "$notice")")
    cp "$notice" "$licenses/${component}-$(basename "$notice")"
done

logo="$repo/docs/assets/logo.png"
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$logo" --out "$icons/icon_${size}x${size}.png" >/dev/null
    if [[ "$size" == 512 ]]; then
        cp "$logo" "$icons/icon_512x512@2x.png"
    else
        sips -z "$((size*2))" "$((size*2))" "$logo" --out "$icons/icon_${size}x${size}@2x.png" >/dev/null
    fi
done
iconutil -c icns "$icons" -o "$res/AppIcon.icns"
# Alpha-only monochrome twin blade extracted from the README SVG, without its squircle.
cp "$project/Packaging/MenuIconTemplate.png" "$res/"
codesign --force --sign - "$core/bin/ninfer-metal-serve"
codesign --force --sign - "$app"
codesign --verify --deep --strict "$app"
"$core/bin/ninfer-metal-serve" --version

if [[ -e "$dist/MIV.app" ]]; then
    old_id=$(/usr/libexec/PlistBuddy -c 'Print CFBundleIdentifier' "$dist/MIV.app/Contents/Info.plist")
    [[ "$old_id" == io.ninfer.miv.mac ]] || { echo "Another application occupies $dist/MIV.app" >&2; exit 1; }
    mv "$dist/MIV.app" "$stage/previous.app"
fi
if ! mv "$app" "$dist/MIV.app"; then
    [[ ! -d "$stage/previous.app" ]] || mv "$stage/previous.app" "$dist/MIV.app"
    exit 1
fi
echo "$dist/MIV.app"
