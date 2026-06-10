#!/usr/bin/env bash
# Static undefined-symbol audit for an Android GDExtension .so — catches the #95 bug class (a
# library that links fine but fails to *load* because a symbol it imports isn't provided by any of
# its DT_NEEDED libraries) without needing to emulate the target ABI.
#
# How it works: we ask the NDK linker to do exactly what bionic's loader does at dlopen(). A trivial
# stub is *linked against* the built .so with `--no-allow-shlib-undefined`, which makes lld error if
# the shared library imports any symbol that none of the libraries on the link line provides. The
# NDK clang++ driver puts the same runtime libs on that line that the device ships (libc++_shared,
# libc, libm, libdl), and — unlike hand-harvesting symbol tables from the sysroot — the linker
# already knows their real locations and versioned-symbol rules. A missing strong symbol here is the
# "cannot locate symbol" abort from #95; weak undefined symbols are left unresolved exactly as at
# runtime, so they don't trip it.
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
# clang++ (not clang) so the driver links libc++_shared.so by default — the same C++ runtime the
# Android app provides, which defines the std::/operator-new/__cxa_* symbols ncnn imports.
clangxx="$toolbin/${triple}${api}-clang++"
for t in "$readelf" "$clangxx"; do
  test -x "$t" || { echo "::error::missing NDK tool $t"; exit 1; }
done

echo "== auditing $so (NDK $ndk) =="

# Informational: the libraries bionic will resolve this .so's imports against at load.
echo "-- DT_NEEDED --"
"$readelf" -d "$so" | awk '/NEEDED/{print $NF}'

# Link a do-nothing stub against the .so and let the linker verify every import resolves.
#  --no-as-needed  : force the linker to actually pull in (and thus inspect) our .so even though the
#                    stub references nothing from it.
#  --no-allow-shlib-undefined : error if the .so needs a symbol no library on the link line defines —
#                    i.e. the exact dlopen-time failure #95 was. (lld defaults to *allowing* these.)
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf 'int main(void){return 0;}\n' > "$tmp/stub.c"

if ! "$clangxx" "$tmp/stub.c" -o "$tmp/stub" \
      -Wl,--no-as-needed \
      -L"$(dirname "$so")" -l:"$(basename "$so")" \
      -Wl,--no-allow-shlib-undefined 2> "$tmp/link.err"; then
  echo "::error::android-$arch .so imports symbols no runtime lib provides (would fail dlopen — #95 bug class):"
  sed 's/^/    /' "$tmp/link.err"
  exit 1
fi

echo "OK: every strong undefined symbol in the android-$arch .so resolves against the Android runtime libs."
