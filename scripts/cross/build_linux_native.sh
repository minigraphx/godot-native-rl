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
  # NCNN_OPENMP=OFF (#103): ship a self-contained .so with no libgomp.so.1 runtime
  # dependency. ncnn falls back to its own simple thread pool / single-thread; deploy
  # inference is typically single-sample, and the batched crowd path (run_inference_batch)
  # uses its own std::threads with opt.num_threads pinned to 1, so nothing relies on OpenMP.
  cmake -S thirdparty/ncnn -B "$ncnn_build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DNCNN_OPENMP=OFF
  CMAKE_BUILD_PARALLEL_LEVEL="$NCNN_JOBS" cmake --build "$ncnn_build" --config Release
  cmake --install "$ncnn_build" --prefix "$repo/$ncnn_build/install"
fi

# Guard: a libncnn.a left over from a pre-#103 build (OpenMP on) would make the .so carry
# undefined GOMP_* symbols once we stop linking gomp — link succeeds, load fails. Fail loud.
# Anchor on the symbol name (last nm field) — stricter than an unanchored grep, and
# covers libgomp (GOMP_*) + LLVM/Intel libomp (omp_*/__kmpc_*), matching build_macos.sh.
# `grep ... >/dev/null`, NOT `grep -q`: -q exits on first match, which SIGPIPEs the still-writing
# nm/awk under `pipefail` (status 141 -> the `if` takes the no-match branch *exactly when* OpenMP
# symbols are present). Reading to EOF keeps the exit status honest (#158).
if nm "$ncnn_build/install/lib/libncnn.a" 2>/dev/null | awk '{print $NF}' \
    | grep -E '^(GOMP_|omp_|__kmpc_)' >/dev/null; then
  echo "ERROR: stale OpenMP-enabled libncnn.a at $ncnn_build/install/lib/libncnn.a." >&2
  echo "       rm -rf $ncnn_build and re-run to rebuild with NCNN_OPENMP=OFF (#103)." >&2
  exit 1
fi

for cfg in template_debug template_release; do
  scons platform=linux arch=x86_64 target="$cfg" ncnn_openmp=no -j"$CPUS"
done

# Validate the shipped .so is self-contained (#103): no libgomp in DT_NEEDED, and no undefined
# OpenMP symbols that would fail at load time on a libgomp-less host. `grep >/dev/null` not `-q`
# for the same SIGPIPE-under-pipefail reason as above (#158). Count audited binaries and fail if
# the glob matched nothing (#155) — an unaudited "pass" would silently re-open the #103 gap.
audited=0
for so in addons/godot_native_rl/bin/libncnn_runner.linux.*.so; do
  [ -f "$so" ] || continue
  audited=$((audited + 1))
  if readelf -d "$so" | grep -E "libgomp" >/dev/null; then
    echo "ERROR: $so still has a libgomp DT_NEEDED entry" >&2
    exit 1
  fi
  if nm -D --undefined-only "$so" | awk '{print $NF}' | grep -E '^(GOMP_|omp_|__kmpc_)' >/dev/null; then
    echo "ERROR: $so has undefined OpenMP symbols (stale OpenMP libncnn.a?)" >&2
    exit 1
  fi
  echo "OK: $so is self-contained (no libgomp NEEDED, no undefined OpenMP symbols)"
done
[ "$audited" -ge 2 ] || { echo "ERROR: expected debug+release .so to audit, found $audited (glob matched nothing?)" >&2; exit 1; }

echo "== built linux x86_64 (native gcc) =="
ls -la addons/godot_native_rl/bin/ | grep linux || true
