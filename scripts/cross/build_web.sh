#!/usr/bin/env bash
# Build ncnn (static, single-threaded) + the GDExtension for the web (WASM) target via emscripten.
# Single-threaded by design: NCNN_THREADS=OFF + scons threads=no, so the exported game needs NO
# COOP/COEP headers and deploys to any static host (itch.io, GitHub Pages). See docs/dev/building.md.
#
# Usage: source your emsdk env first, then: scripts/cross/build_web.sh
# Requires: emsdk activated (emcc on PATH), cmake, scons, python3; ./godot-cpp + ./thirdparty/ncnn.
# Env: NCNN_JOBS caps ncnn compile parallelism (default = CPU count).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

command -v emcc >/dev/null || { echo "emcc not found; source your emsdk_env.sh first" >&2; exit 2; }

CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
NCNN_JOBS="${NCNN_JOBS:-$CPUS}"

ncnn_build="thirdparty/ncnn/build-web-wasm32"
if [ ! -f "$ncnn_build/install/lib/libncnn.a" ]; then
  emcmake cmake -S thirdparty/ncnn -B "$ncnn_build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DNCNN_SHARED_LIB=OFF \
    -DNCNN_OPENMP=OFF -DNCNN_THREADS=OFF -DNCNN_VULKAN=OFF -DNCNN_SIMD=ON
  cmake --build "$ncnn_build" --parallel "$NCNN_JOBS"
  cmake --install "$ncnn_build" --prefix "$repo/$ncnn_build/install"
fi

for cfg in template_debug template_release; do
  scons platform=web arch=wasm32 target="$cfg" threads=no ncnn_openmp=no -j"$CPUS"
done

echo "== built web wasm32 =="
ls -la bin/ | grep -i web || true
