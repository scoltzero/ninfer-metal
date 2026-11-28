#!/bin/bash
# A portable third-party Python environment. Neither MIV build nor launch needs an oMLX checkout.
set -euo pipefail
dest=${1:?usage: prepare.sh OUTPUT_DIR}
src=$(cd "$(dirname "$0")" && pwd)
if [[ -x "$dest/cpython-3.11/bin/python3" && -d "$dest/framework-mlx-base/lib/python3.11/site-packages/httpx" ]]; then exit 0; fi
mkdir -p "$(dirname "$dest")"
work=$(mktemp -d "$(dirname "$dest")/.miv-runtime.XXXXXX")
trap 'rm -rf "$work"' EXIT
uv python install 3.11.10 --install-dir "$work/pythons"
mkdir -p "$work/runtime/framework-mlx-base/lib/python3.11/site-packages"
for python in "$work"/pythons/cpython-3.11.10-*; do
    [[ -x "$python/bin/python3" ]] || continue
    ditto "$python" "$work/runtime/cpython-3.11"
    break
done
py="$work/runtime/cpython-3.11/bin/python3"
uv pip install --python "$py" --target "$work/runtime/framework-mlx-base/lib/python3.11/site-packages" \
    -c "$src/constraints.txt" -r "$src/requirements.txt"
PYTHONNOUSERSITE=1 PYTHONPATH="$work/runtime/framework-mlx-base/lib/python3.11/site-packages" \
    "$py" -B -c 'import httpx, huggingface_hub, modelscope, mlx.core, mlx_lm, mlx_vlm'
[[ ! -e "$dest" ]] || { echo "Incomplete runtime exists at $dest" >&2; exit 1; }
mv "$work/runtime" "$dest"
