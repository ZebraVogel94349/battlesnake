#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="python3"

export PATH="$(dirname -- "$PYTHON"):$PATH"

NVCC="${CUDACXX:-$(command -v nvcc || true)}"
if [[ -z "$NVCC" || ! -x "$NVCC" ]]; then
    echo "nvcc not found. The NVIDIA driver/nvidia-smi alone is insufficient." >&2
    echo "Install a CUDA toolkit, then set CUDACXX=/path/to/nvcc." >&2
    exit 2
fi

for command_name in cmake ninja; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "$command_name is missing." >&2
        echo "Install build requirements first:" >&2
        echo "  $PYTHON -m pip install -r $SCRIPT_DIR/requirements-build.txt" >&2
        exit 2
    fi
done
if ! "$PYTHON" -c 'import scikit_build_core' >/dev/null 2>&1; then
    echo "scikit-build-core is missing." >&2
    echo "Install build requirements first:" >&2
    echo "  $PYTHON -m pip install -r $SCRIPT_DIR/requirements-build.txt" >&2
    exit 2
fi

CUDA_ARCHITECTURES="${HISSS_CUDA_ARCHITECTURES:-native}"
export CUDACXX="$NVCC"
export CMAKE_ARGS="${CMAKE_ARGS:-} -DHISSS_REQUIRE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

echo "[hisss-build] source=$PROJECT_DIR"
echo "[hisss-build] python=$PYTHON"
echo "[hisss-build] nvcc=$CUDACXX"
echo "[hisss-build] CUDA architectures=$CUDA_ARCHITECTURES"

# --no-deps prevents the starter's PyPI requirement from being considered;
# --no-build-isolation prevents a second, hidden build environment.  The
# explicit local editable project replaces an already installed PyPI hisss.
"$PYTHON" -m pip install \
    --no-build-isolation \
    --no-deps \
    --force-reinstall \
    --verbose \
    --editable "$PROJECT_DIR"

"$PYTHON" - "$PROJECT_DIR" <<'PY'
import importlib.metadata
import sys
from pathlib import Path

project = Path(sys.argv[1]).resolve()
import hisss

module = Path(hisss.__file__).resolve()
try:
    module.relative_to(project / "src")
except ValueError as exc:
    raise SystemExit(
        f"wrong hisss imported after build: {module}; expected {project / 'src'}"
    ) from exc
if not hisss.cuda_available():
    raise SystemExit(f"local hisss has no usable CUDA backend: {hisss.cuda_last_error()}")
direct_url = importlib.metadata.distribution("hisss").read_text("direct_url.json")
print(f"[hisss-build] module={module}")
print(f"[hisss-build] editable_metadata={direct_url}")
print("[hisss-build] CUDA backend is available")
PY
