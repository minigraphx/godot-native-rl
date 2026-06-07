#!/usr/bin/env bash
# Build ncnn (static) + the GDExtension for iOS — device (arm64) and simulator (arm64+x86_64) —
# and package each config into an .xcframework. macOS + full Xcode only.
# Used by the dev recipe (docs/dev/building.md) and the cross-build CI.
#
# Usage: scripts/cross/build_ios.sh
# Requires: Xcode (iOS SDKs), cmake, scons; ./godot-cpp + ./thirdparty/ncnn.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
CPUS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
NCNN_JOBS="${NCNN_JOBS:-$CPUS}"

ncnn_common=(
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 -DCMAKE_BUILD_TYPE=Release
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF
  -DNCNN_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DNCNN_SHARED_LIB=OFF
  -DNCNN_OPENMP=OFF -DNCNN_VULKAN=OFF -DNCNN_THREADS=ON
)

build_ncnn() { # <build-subdir> <sysroot> <arch>
  local dir="thirdparty/ncnn/$1"
  [ -f "$dir/install/lib/libncnn.a" ] && return 0
  cmake -S thirdparty/ncnn -B "$dir" "${ncnn_common[@]}" \
    -DCMAKE_OSX_SYSROOT="$2" -DCMAKE_OSX_ARCHITECTURES="$3"
  cmake --build "$dir" --parallel "$NCNN_JOBS"
  cmake --install "$dir" --prefix "$repo/$dir/install"
}

build_ncnn build-ios-arm64     iphoneos        arm64
build_ncnn build-iossim-arm64  iphonesimulator arm64
build_ncnn build-iossim-x86_64 iphonesimulator x86_64

# Fat simulator lib (arm64 + x86_64) at the dir SConstruct expects for arch=universal.
uni=thirdparty/ncnn/build-ios-universal
if [ ! -f "$uni/install/lib/libncnn.a" ]; then
  mkdir -p "$uni/install/lib"
  cp -R thirdparty/ncnn/build-iossim-arm64/install/include "$uni/install/include"
  lipo -create \
    thirdparty/ncnn/build-iossim-arm64/install/lib/libncnn.a \
    thirdparty/ncnn/build-iossim-x86_64/install/lib/libncnn.a \
    -output "$uni/install/lib/libncnn.a"
fi

for cfg in template_debug template_release; do
  scons platform=ios arch=arm64     target="$cfg"                   -j"$CPUS"
  scons platform=ios arch=universal target="$cfg" ios_simulator=yes -j"$CPUS"
  out="bin/libncnn_runner.ios.$cfg.xcframework"
  rm -rf "$out"
  xcodebuild -create-xcframework \
    -library "bin/libncnn_runner.ios.$cfg.arm64.dylib" \
    -library "bin/libncnn_runner.ios.$cfg.universal.simulator.dylib" \
    -output "$out"
done

echo "== built ios =="
ls -d bin/*ios* || true
