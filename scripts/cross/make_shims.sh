#!/usr/bin/env bash
# Generate zig cross-compiler shims for the GDExtension build.
#
# Usage: scripts/cross/make_shims.sh <linux|windows> <outdir>
#
# Produces, in <outdir>:
#   cc, c++, ar, ranlib           — generic names (used by the ncnn CMake toolchain)
#   x86_64-w64-mingw32-{gcc,g++,gcc-ar,ranlib}  (windows only — the names godot-cpp expects)
# Put <outdir> on PATH for the scons build; point the ncnn CMake toolchain at <outdir>/cc,c++.
set -euo pipefail

target_os="${1:?usage: make_shims.sh <linux|windows> <outdir>}"
outdir="${2:?usage: make_shims.sh <linux|windows> <outdir>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shim="$here/zigshim.py"

case "$target_os" in
  linux)   triple="x86_64-linux-gnu" ;;
  windows) triple="x86_64-windows-gnu" ;;
  *) echo "unknown target '$target_os' (expected linux|windows)" >&2; exit 2 ;;
esac

mkdir -p "$outdir"

cc_shim() { # $1=path $2=lang(cc|c++)
  printf '#!/bin/sh\nZIG_LANG=%s ZIG_TARGET=%s exec python3 "%s" "$@"\n' "$2" "$triple" "$shim" >"$1"
  chmod +x "$1"
}
zig_tool() { # $1=path $2=zig subcommand (ar|ranlib)
  printf '#!/bin/sh\nexec zig %s "$@"\n' "$2" >"$1"
  chmod +x "$1"
}

cc_shim  "$outdir/cc"  cc
cc_shim  "$outdir/c++" c++
zig_tool "$outdir/ar"     ar
zig_tool "$outdir/ranlib" ranlib

if [ "$target_os" = windows ]; then
  # godot-cpp's mingw cross path calls these exact names.
  cc_shim  "$outdir/x86_64-w64-mingw32-gcc" cc
  cc_shim  "$outdir/x86_64-w64-mingw32-g++" c++
  zig_tool "$outdir/x86_64-w64-mingw32-gcc-ar" ar
  zig_tool "$outdir/x86_64-w64-mingw32-ranlib" ranlib
fi

echo "wrote $target_os shims ($triple) to $outdir"
