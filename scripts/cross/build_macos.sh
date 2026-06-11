#!/usr/bin/env bash
# Native macOS (Apple Silicon, arm64) build of ncnn (static) + the GDExtension, using the host
# Xcode toolchain. Runs on a macOS host (CI's macos-14 runner, or a macOS dev box).
#
# Ships a self-contained dylib (#152, the macOS analogue of #103's Linux libgomp fix): ncnn is
# built with NCNN_OPENMP=OFF so cmake's find_package(OpenMP) never injects a homebrew
# libomp.dylib load path (e.g. /opt/homebrew/opt/libomp/lib/libomp.dylib) into libncnn.a — which
# would fail to load on any user machine without homebrew libomp. The post-build audit fails the
# build loud if the dylib references anything outside the system paths.
#
# Requires: Xcode (cmake, clang), scons, git, python3; ./godot-cpp + ./thirdparty/ncnn checked out.
# Env: NCNN_JOBS caps ncnn's compile parallelism (default = CPU count).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
CPUS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
NCNN_JOBS="${NCNN_JOBS:-$CPUS}"

# SConstruct's macOS host path looks for the static lib at thirdparty/ncnn/build/install (the
# generic host dir), so build there — same dir build_linux_native.sh uses for Linux.
ncnn_build="thirdparty/ncnn/build"
if [ ! -f "$ncnn_build/install/lib/libncnn.a" ]; then
  cmake -S thirdparty/ncnn -B "$ncnn_build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DNCNN_OPENMP=OFF
  CMAKE_BUILD_PARALLEL_LEVEL="$NCNN_JOBS" cmake --build "$ncnn_build" --config Release
  cmake --install "$ncnn_build" --prefix "$repo/$ncnn_build/install"
fi

# Guard: a libncnn.a left over from a pre-#152 build (OpenMP on) would have undefined OpenMP
# runtime symbols that scons pulls libomp in to satisfy at link time — fail loud instead.
# Anchor on the symbol name (last nm field) so e.g. `_compute_*` can't false-match `omp_`.
# `grep ... >/dev/null`, NOT `grep -q`: -q exits on first match, SIGPIPEing the still-writing
# nm/awk under `pipefail` (status 141 -> no-match branch *exactly when* OpenMP symbols exist) (#158).
if nm "$ncnn_build/install/lib/libncnn.a" 2>/dev/null | awk '{print $NF}' \
    | grep -E '^(_GOMP_|_omp_|___kmpc_)' >/dev/null; then
  echo "ERROR: stale OpenMP-enabled libncnn.a at $ncnn_build/install/lib/libncnn.a." >&2
  echo "       rm -rf $ncnn_build and re-run to rebuild with NCNN_OPENMP=OFF (#152)." >&2
  exit 1
fi

for cfg in template_debug template_release; do
  scons platform=macos arch=arm64 target="$cfg" -j"$CPUS"
done

# Validate the shipped dylib is self-contained (#152). otool -L line 1 is the file header and
# line 2 is the dylib's own install name (LC_ID_DYLIB, often @rpath/... or the build path), so
# allow self/relocatable refs and system paths (/usr/lib, /System/Library); hard-deny anything
# whose basename looks like an OpenMP runtime (libomp/libgomp/libiomp5) regardless of path; flag
# any other non-system dependency. Then dlopen with eager symbol resolution.
self_basename_ok() { [ "$(basename "$1")" = "$2" ]; }
audited=0
for dylib in addons/godot_native_rl/bin/libncnn_runner.macos.*.dylib; do
  [ -f "$dylib" ] || continue
  audited=$((audited + 1))
  self="$(basename "$dylib")"
  bad=""
  while IFS= read -r dep; do
    [ -z "$dep" ] && continue
    base="$(basename "$dep")"
    case "$base" in
      libomp.dylib|libgomp*.dylib|libiomp5.dylib)
        bad="${bad}${dep} (OpenMP runtime)"$'\n' ; continue ;;
    esac
    case "$dep" in
      /usr/lib/*|/System/Library/*) ;;                      # system: ok
      @rpath/*|@loader_path/*|@executable_path/*) ;;        # relocatable self/loader refs: ok
      *) self_basename_ok "$dep" "$self" || bad="${bad}${dep}"$'\n' ;;  # the dylib's own id: ok
    esac
  done < <(otool -L "$dylib" | tail -n +2 | awk '{print $1}')
  if [ -n "$bad" ]; then
    echo "ERROR: $dylib has non-system / OpenMP load dependencies:" >&2
    printf '%s' "$bad" >&2
    exit 1
  fi
  python3 - "$dylib" <<'PY'
import ctypes, sys
ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL | 2)  # RTLD_NOW
print("OK: %s is self-contained (no OpenMP/non-system deps) and dlopens" % sys.argv[1])
PY
done
# Fail if the glob matched nothing (#155) — an unaudited "pass" would silently re-open the #152 gap.
[ "$audited" -ge 2 ] || { echo "ERROR: expected debug+release dylib to audit, found $audited (glob matched nothing?)" >&2; exit 1; }

echo "== built macos arm64 (native) =="
ls -la addons/godot_native_rl/bin/ | grep macos || true
