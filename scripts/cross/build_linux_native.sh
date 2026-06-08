#!/usr/bin/env bash
# Native Linux build of ncnn (static) + the GDExtension, using the host's system toolchain
# (gcc/g++ + libstdc++). Runs on a Linux host (CI's ubuntu runner, or a Linux dev box).
#
# Why native, not zig: the zig cross-compile (build_zig.sh) produces a Linux .so that links but
# fails to LOAD on an Ubuntu host — libc++'s out-of-line std::string / std::bad_array_new_length
# instantiations aren't linked in, so Godot reports
#   undefined symbol: std::__1::basic_string::push_back
# Native GCC uses libstdc++ and avoids that entirely. This is the same toolchain ci.yml builds +
# tests with, so the result is proven to load. zig stays for Windows cross-compile and local
# macOS->Linux dev builds.
#
# Requires: build-essential, scons, cmake, git, python3; ./godot-cpp + ./thirdparty/ncnn checked out.
# Env: NCNN_JOBS caps ncnn's compile parallelism (default = CPU count).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
NCNN_JOBS="${NCNN_JOBS:-$CPUS}"

ncnn_build="thirdparty/ncnn/build"
if [ ! -f "$ncnn_build/install/lib/libncnn.a" ]; then
  cmake -S thirdparty/ncnn -B "$ncnn_build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF
  CMAKE_BUILD_PARALLEL_LEVEL="$NCNN_JOBS" cmake --build "$ncnn_build" --config Release
  cmake --install "$ncnn_build" --prefix "$repo/$ncnn_build/install"
fi

for cfg in template_debug template_release; do
  scons platform=linux arch=x86_64 target="$cfg" -j"$CPUS"
done

echo "== built linux x86_64 (native gcc) =="
ls -la addons/godot_native_rl/bin/ | grep linux || true
