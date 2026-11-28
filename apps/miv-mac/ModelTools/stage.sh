#!/bin/bash
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
out=${1:?expected MIV ModelTools resource directory}
mkdir -p "$out"
ditto "$src/Snapshot/omlx" "$out/omlx"
# The helper imports private snapshot modules, never the host's oMLX package.
touch "$out/omlx/__init__.py" "$out/omlx/admin/__init__.py" "$out/omlx/api/__init__.py"
cp "$src"/worker.py "$src"/bridge.py "$src"/launch.py "$out/"
cp "$src"/settings.py "$out/omlx/"
install -m 755 "$src/miv-launch" "$out/"
ditto "$src/Icons" "$out/Icons"
