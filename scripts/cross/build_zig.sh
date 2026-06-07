#!/usr/bin/env bash
# Cross-build ncnn (static) + the GDExtension for a zig target — Windows or Linux x86_64 —
# from any host. Used by the dev recipe (docs/dev/building.md) and the cross-build CI.
#
# Usage: scripts/cross/build_zig.sh <linux|windows>
# Requires: zig, cmake, scons, python3; ./godot-cpp and ./thirdparty/ncnn checked out.
# Env: NCNN_JOBS caps ncnn's compile parallelism (default = CPU count; CI sets 2 to avoid OOM).
set -euo pipefail

target="${1:?usage: build_zig.sh <linux|windows>}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

case "$target" in
  linux)   plat=linux;   sysname=Linux ;;
  windows) plat=windows; sysname=Windows ;;
  *) echo "unknown target '$target' (expected linux|windows)" >&2; exit 2 ;;
esac
arch=x86_64
CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
NCNN_JOBS="${NCNN_JOBS:-$CPUS}"

shimdir="$repo/thirdparty/_xbuild/shims-$target"
scripts/cross/make_shims.sh "$target" "$shimdir"

ncnn_build="thirdparty/ncnn/build-$plat-$arch"
if [ ! -f "$ncnn_build/install/lib/libncnn.a" ]; then
  cmake -S thirdparty/ncnn -B "$ncnn_build" \
    -DCMAKE_TOOLCHAIN_FILE="$repo/scripts/cross/zig-toolchain.cmake" \
    -DZIG_SHIM_DIR="$shimdir" -DZIG_SYSTEM_NAME="$sysname" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DNCNN_SHARED_LIB=OFF \
    -DNCNN_OPENMP=OFF -DNCNN_VULKAN=OFF -DNCNN_THREADS=ON
  cmake --build "$ncnn_build" --parallel "$NCNN_JOBS"
  cmake --install "$ncnn_build" --prefix "$repo/$ncnn_build/install"
fi

export PATH="$shimdir:$PATH"
for cfg in template_debug template_release; do
  scons platform="$plat" arch="$arch" target="$cfg" ncnn_openmp=no -j"$CPUS"
done

echo "== built $plat $arch =="
ls -la addons/godot_native_rl/bin/ | grep "$plat" || true
