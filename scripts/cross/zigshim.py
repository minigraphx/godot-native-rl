#!/usr/bin/env python3
"""Drop-in gcc/mingw cross compiler backed by ``zig cc`` / ``zig c++``.

Lets zig stand in for a gcc/mingw cross toolchain under SCons (godot-cpp) and CMake
(ncnn) so the GDExtension can be cross-compiled to Windows and Linux from any host.
Sanitizes gcc/mingw-only flags that zig's clang/lld reject and bakes in the ``-target``
triple. Used both by the local dev recipe (docs/dev/building.md) and the cross-build CI.

Invoked via small per-name wrapper scripts (see make_shims.sh) that set:
  ZIG_LANG    "cc" or "c++"
  ZIG_TARGET  e.g. x86_64-linux-gnu / x86_64-windows-gnu
"""
import os
import subprocess
import sys
import tempfile

lang = os.environ["ZIG_LANG"]
target = os.environ["ZIG_TARGET"]

# GCC/mingw-only flags clang/lld doesn't accept. The -target triple already pins the base
# ISA, so -m64/-march=x86-64 are redundant and zig rejects them ("unknown CPU"). Only the
# generic arch flags are dropped — ncnn's per-file SIMD flags (-mavx2/-msse2/-mfma/...) pass
# through untouched.
DROP = {
    "-static",          # conflicts with -shared under lld (zig static-links its runtime anyway)
    "-fno-gnu-unique",  # GCC-only (godot-cpp hot-reload); clang rejects it
    "-m64", "-m32",
    "-march=x86-64", "-march=x86_64", "-march=i686",
}

out = []
for arg in sys.argv[1:]:
    if arg in DROP:
        continue
    if arg.startswith("-Wl,-R,"):  # GNU rpath shorthand -> portable form lld understands
        out.append("-Wl,-rpath," + arg[len("-Wl,-R,"):])
        continue
    out.append(arg)

cmd = ["zig", lang, "-target", target] + out

# SCons/CMake invoke us with a scrubbed environment (no HOME), so zig can't locate its
# default cache dir. Pin it (honour an explicit override if the caller set one).
child_env = os.environ.copy()
cache_root = os.path.join(tempfile.gettempdir(), "zig-cross-cache")
child_env.setdefault("ZIG_GLOBAL_CACHE_DIR", os.path.join(cache_root, "global"))
child_env.setdefault("ZIG_LOCAL_CACHE_DIR", os.path.join(cache_root, "local"))
sys.exit(subprocess.call(cmd, env=child_env))
