#!/usr/bin/env bash
# Static undefined-symbol audit for an Android GDExtension .so — catches the #95 bug class (a
# library that links fine but fails to *load* because a symbol it imports isn't provided by any of
# its DT_NEEDED libraries) without needing to emulate the target ABI.
#
# How it works: every UNDEFINED dynamic symbol the .so imports must be satisfiable, at load time, by
# some library named in its DT_NEEDED list. We resolve that list against the NDK sysroot (the same
# system libs Godot's Android runtime provides) and fail if any *strong* (non-weak) undefined symbol
# is not defined by any of them. This mirrors what bionic's linker does at dlopen() — a missing
# strong symbol there aborts with "cannot locate symbol", exactly the #95 failure.
#
# Usage: scripts/cross/audit_android_symbols.sh <arm64|x86_64>
# Requires: ANDROID_NDK_LATEST_HOME (or ANDROID_NDK_ROOT); the built .so in addons/.../bin/.
set -euo pipefail

arch="${1:?usage: audit_android_symbols.sh <arm64|x86_64>}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

ndk="${ANDROID_NDK_LATEST_HOME:-${ANDROID_NDK_ROOT:-}}"
: "${ndk:?set ANDROID_NDK_LATEST_HOME or ANDROID_NDK_ROOT to the NDK path}"

# API 24 matches build_android.sh's ANDROID_PLATFORM.
api=24
case "$arch" in
  arm64)  triple=aarch64-linux-android ;;
  x86_64) triple=x86_64-linux-android ;;
  *) echo "unknown arch '$arch' (expected arm64|x86_64)" >&2; exit 2 ;;
esac

so="addons/godot_native_rl/bin/libncnn_runner.android.template_release.$arch.so"
test -f "$so" || { echo "::error::missing $so (build it first)"; exit 1; }

# NDK prebuilt toolchain bin dir (host = linux-x86_64 on the GitHub runner).
toolbin="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin"
readelf="$toolbin/llvm-readelf"
nm="$toolbin/llvm-nm"
for t in "$readelf" "$nm"; do
  test -x "$t" || { echo "::error::missing NDK tool $t"; exit 1; }
done

# System libs the .so is allowed to import from, as bionic provides at runtime. We resolve
# DT_NEEDED against this sysroot. libc++_shared.so lives in the NDK (Godot's APK ships it too).
sysroot="$ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/$triple/$api"
cxxlib="$ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/$triple/libc++_shared.so"

echo "== auditing $so (NDK $ndk) =="

# 1) DT_NEEDED list (informational + sanity).
echo "-- DT_NEEDED --"
"$readelf" -d "$so" | awk '/NEEDED/{print $NF}'

# 2) Collect every UNDEFINED, non-weak dynamic symbol the .so imports.
#    llvm-nm -D -u prints undefined dynamic symbols; 'w'/'v' type = weak (ok to be missing).
und="$(mktemp)"
"$nm" -D -u "$so" | awk '{print $NF}' | sort -u > "$und"
# Weak undefined symbols are allowed to stay unresolved at load (the linker just leaves them 0).
weak="$(mktemp)"
"$nm" -D "$so" | awk '$1 ~ /^[wv]$/ {print $NF}' | sort -u > "$weak"
comm -23 "$und" "$weak" > "$und.strong"
echo "-- strong undefined symbols: $(wc -l < "$und.strong") --"

# 3) Build the set of symbols DEFINED by the runtime libs (sysroot system libs + libc++_shared).
defined="$(mktemp)"
: > "$defined"
shopt -s nullglob
for lib in "$sysroot"/lib*.so "$cxxlib"; do
  [ -f "$lib" ] || continue
  # Defined dynamic symbols (T/W/B/D/R/V/i...) — anything not 'U'.
  "$nm" -D --defined-only "$lib" 2>/dev/null | awk '{print $NF}'
done | sort -u > "$defined"

# 4) Any strong undefined symbol not defined anywhere in the runtime libs is a load-time failure.
missing="$(comm -23 "$und.strong" "$defined" || true)"
if [ -n "$missing" ]; then
  echo "::error::android-$arch .so imports symbols no runtime lib provides (would fail dlopen — #95 bug class):"
  echo "$missing" | sed 's/^/    /'
  exit 1
fi

echo "OK: every strong undefined symbol in the android-$arch .so is satisfiable at load."
