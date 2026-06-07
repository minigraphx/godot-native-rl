#!/usr/bin/env bash
# Cross-build ncnn (static) + the GDExtension for Android, one ABI at a time.
# Used by the dev recipe (docs/dev/building.md) and the cross-build CI.
#
# Usage: scripts/cross/build_android.sh <arm64|x86_64>
# Requires: ANDROID_NDK_ROOT (path to an NDK), cmake, scons; ./godot-cpp + ./thirdparty/ncnn.
# Env: NCNN_JOBS caps ncnn's compile parallelism (default = CPU count).
set -euo pipefail

arch="${1:?usage: build_android.sh <arm64|x86_64>}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

case "$arch" in
  arm64)  abi=arm64-v8a ;;
  x86_64) abi=x86_64 ;;
  *) echo "unknown arch '$arch' (expected arm64|x86_64)" >&2; exit 2 ;;
esac
: "${ANDROID_NDK_ROOT:?set ANDROID_NDK_ROOT to the NDK path}"
CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
NCNN_JOBS="${NCNN_JOBS:-$CPUS}"

ncnn_build="thirdparty/ncnn/build-android-$arch"
if [ ! -f "$ncnn_build/install/lib/libncnn.a" ]; then
  cmake -S thirdparty/ncnn -B "$ncnn_build" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$abi" -DANDROID_PLATFORM=android-24 -DCMAKE_BUILD_TYPE=Release \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DNCNN_SHARED_LIB=OFF \
    -DNCNN_OPENMP=OFF -DNCNN_VULKAN=OFF -DNCNN_THREADS=ON
  cmake --build "$ncnn_build" --parallel "$NCNN_JOBS"
  cmake --install "$ncnn_build" --prefix "$repo/$ncnn_build/install"
fi

# ANDROID_HOME= (empty) is required: godot-cpp's android.py does `if env["ANDROID_HOME"]` but its
# default resolves to None so the key is never created (KeyError) — empty falls through to
# ANDROID_NDK_ROOT.
for cfg in template_debug template_release; do
  scons platform=android arch="$arch" target="$cfg" ncnn_openmp=no ANDROID_HOME= -j"$CPUS"
done

echo "== built android $arch =="
ls -la addons/godot_native_rl/bin/ | grep android || true
