#!/usr/bin/env bash
# Static test-link audit for the iOS GDExtension .xcframework — catches the #95 bug class (a binary
# that links but fails to *load* because an imported symbol isn't provided at runtime) for a target
# we can't dlopen on the macOS host (wrong platform).
#
# The xcframework wraps per-slice *dylibs* (see build_ios.sh). We prove each slice resolves by
# *test-linking* a trivial main against the dylib with -isysroot the matching iOS SDK: the static
# linker resolves the dylib's transitive imports against the SDK exactly as dyld does at app launch,
# so an undefined libc++/system symbol fails here with "Undefined symbols", the same as on-device.
#
# Usage: scripts/cross/audit_ios_link.sh
# Requires: Xcode (iphoneos + iphonesimulator SDKs); the built .xcframework in addons/.../bin/.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

# Audit the release xcframework (what ships); debug uses the same toolchain + link line.
xcf="addons/godot_native_rl/bin/libncnn_runner.ios.template_release.xcframework"
test -d "$xcf" || { echo "::error::missing $xcf (build it first)"; exit 1; }

work="$(mktemp -d)"
cat > "$work/main.c" <<'EOF'
/* Reference the GDExtension entry symbol (a function — godot-cpp's GDExtensionInit) so the linker
   must pull in the dylib and resolve it; linking the dylib then forces its LC_LOAD_DYLIB deps +
   two-level-namespace imports to resolve against the iOS SDK, the same as dyld at app launch. */
extern int ncnn_runner_library_init(void *p_get_proc, const void *p_lib, void *r_init);
void *gnrl_force = (void *)&ncnn_runner_library_init;
int main(void) { return gnrl_force != 0 ? 0 : 0; }
EOF

# Find each slice's dylib inside the xcframework and link against it with the matching SDK.
status=0
audited=0
shopt -s nullglob
for dylib in "$xcf"/*/*.dylib; do
  audited=$((audited + 1))
  slice_dir="$(dirname "$dylib")"
  slice="$(basename "$slice_dir")"   # e.g. ios-arm64 or ios-arm64_x86_64-simulator
  case "$slice" in
    *simulator*) sdk=iphonesimulator; min=12.0 ; suffix=ios-simulator ;;
    *)           sdk=iphoneos;        min=12.0 ; suffix=ios ;;
  esac
  sysroot="$(xcrun --sdk "$sdk" --show-sdk-path)"

  # Link target arch from the slice dir name: prefer arm64 (device + sim both have it).
  case "$slice" in
    *arm64*) arch=arm64 ;;
    *x86_64*) arch=x86_64 ;;
    *) echo "::warning::can't infer arch for slice $slice, defaulting arm64"; arch=arm64 ;;
  esac
  target="${arch}-apple-ios${min}"
  [ "$suffix" = "ios-simulator" ] && target="${arch}-apple-ios${min}-simulator"

  echo "== test-linking slice $slice ($target, SDK $sdk) =="
  echo "-- dylib load commands (LC_LOAD_DYLIB) --"
  xcrun otool -L "$dylib" | sed 's/^/    /'

  # Re-export the dylib's interface into the executable (-Wl,-reexport not needed). Linking against
  # the dylib makes ld load each of its LC_LOAD_DYLIB dependencies from the SDK and resolve the
  # dylib's two-level-namespace imports — a dependency or symbol the SDK can't provide fails here,
  # the same as dyld at launch. -Wl,-undefined,error keeps any stray flat symbol fatal.
  if xcrun --sdk "$sdk" clang -target "$target" -isysroot "$sysroot" \
        -Wl,-undefined,error \
        "$work/main.c" "$dylib" -o "$work/probe-$slice" 2>"$work/err-$slice"; then
    echo "OK: $slice links clean (all dependencies + symbols resolve against $sdk SDK)."
  else
    echo "::error::ios slice $slice failed to link — unresolved deps/symbols (#95 bug class):"
    sed 's/^/    /' "$work/err-$slice"
    status=1
  fi
done

# Fail if the slice glob matched nothing (#180, same vacuous-glob class as #155/#175): with
# `nullglob` a zero-match loop runs zero times and would print OK without auditing anything,
# silently re-opening the #95 iOS link-but-fail-to-load gap. The xcframework ships a device slice
# + a simulator slice, so require >= 2.
if [ "$audited" -lt 2 ]; then
  echo "::error::expected >=2 xcframework slices to audit, found $audited (slice glob matched nothing?)" >&2
  exit 1
fi
if [ "$status" -ne 0 ]; then
  exit 1
fi
echo "OK: every iOS xcframework slice resolves its symbols against the iOS SDK ($audited slices)."
