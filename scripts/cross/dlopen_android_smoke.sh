#!/usr/bin/env bash
# Runtime load check for the Android GDExtension .so: dlopen() it on a real Android emulator
# against actual bionic, and dlsym() the GDExtension entry symbol. This is the device-equivalent of
# the Linux/Windows ClassDB.instantiate smoke — if the .so links but fails to *load* (the #95 bug
# class: an undefined libc++/system symbol), dlopen() returns NULL and we print dlerror() and fail.
#
# Runs inside reactivecircus/android-emulator-runner (an emulator is already booted; adb is on PATH).
# We don't run Godot in the emulator (out of scope) — a tiny NDK-compiled C host is enough to prove
# the .so loads against bionic + libc++_shared. The host links libandroid + liblog so the platform
# symbols the extension imports (but leaves to the host, as Godot does) are present in the process.
#
# Usage: scripts/cross/dlopen_android_smoke.sh <x86_64|arm64>
# Requires: ANDROID_NDK_LATEST_HOME (or ANDROID_NDK_ROOT), adb (running emulator), the built .so.
set -euo pipefail

arch="${1:?usage: dlopen_android_smoke.sh <x86_64|arm64>}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

ndk="${ANDROID_NDK_LATEST_HOME:-${ANDROID_NDK_ROOT:-}}"
: "${ndk:?set ANDROID_NDK_LATEST_HOME or ANDROID_NDK_ROOT to the NDK path}"

api=24   # matches build_android.sh's ANDROID_PLATFORM=android-24
case "$arch" in
  arm64)  triple=aarch64-linux-android ;;
  x86_64) triple=x86_64-linux-android ;;
  *) echo "unknown arch '$arch' (expected arm64|x86_64)" >&2; exit 2 ;;
esac

so="addons/godot_native_rl/bin/libncnn_runner.android.template_release.$arch.so"
test -f "$so" || { echo "::error::missing $so (build it first)"; exit 1; }

toolbin="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin"
clang="$toolbin/${triple}${api}-clang"
test -x "$clang" || { echo "::error::missing NDK clang $clang"; exit 1; }
# godot-cpp links the GDExtension against libc++_shared.so (see docs/dev/building.md), which the
# Godot APK ships but a bare emulator does not — push it alongside the .so.
cxxlib="$ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/$triple/libc++_shared.so"
test -f "$cxxlib" || { echo "::error::missing $cxxlib"; exit 1; }

work="$(mktemp -d)"
cat > "$work/host.c" <<'EOF'
#include <dlfcn.h>
#include <stdio.h>
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: host <path-to-.so>\n"); return 2; }
    /* The extension imports Android platform APIs (AAsset / __android_log_print) but does not
       DT_NEEDED libandroid/liblog -- Godot's runtime provides them. Preload them RTLD_GLOBAL so
       those symbols are in the global scope when we load the extension, mirroring the engine.
       Linking the host against them would not work: --as-needed drops unreferenced DT_NEEDEDs. */
    dlopen("libandroid.so", RTLD_NOW | RTLD_GLOBAL);
    dlopen("liblog.so", RTLD_NOW | RTLD_GLOBAL);
    /* RTLD_NOW forces every symbol to resolve immediately — exactly the #95 failure surface. */
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen FAILED: %s\n", dlerror()); return 1; }
    /* The GDExtension entry symbol Godot itself looks up (see ncnn_runner.gdextension). */
    void *sym = dlsym(h, "ncnn_runner_library_init");
    if (!sym) { fprintf(stderr, "dlsym(ncnn_runner_library_init) FAILED: %s\n", dlerror()); return 1; }
    printf("OK: dlopen + dlsym(ncnn_runner_library_init) succeeded\n");
    return 0;
}
EOF

# Build the host for the emulator ABI. Only -ldl is linked; the Android platform libs are pulled in
# at runtime via dlopen(RTLD_GLOBAL) above (the emulator ships them in /system/lib*).
"$clang" -o "$work/host" "$work/host.c" -ldl

dev=/data/local/tmp/gnrl_smoke
adb wait-for-device
adb shell "rm -rf $dev && mkdir -p $dev"
adb push "$work/host" "$dev/host" >/dev/null
adb push "$so" "$dev/$(basename "$so")" >/dev/null
adb push "$cxxlib" "$dev/libc++_shared.so" >/dev/null
adb shell "chmod 755 $dev/host"

# LD_LIBRARY_PATH=$dev so the .so's DT_NEEDED libc++_shared.so resolves to the one we pushed.
set +e
out="$(adb shell "cd $dev && LD_LIBRARY_PATH=$dev ./host $dev/$(basename "$so"); echo EXIT:\$?")"
set -e
echo "$out"
echo "$out" | grep -q "EXIT:0" || { echo "::error::android-$arch dlopen smoke failed (#95 bug class)"; exit 1; }
echo "OK: android-$arch .so loaded on a real emulator."
