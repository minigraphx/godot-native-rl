# Building From Source

> **Contributor / from-source build.** Game developers should start at
> [docs/guide/getting-started.md](../guide/getting-started.md) — a prebuilt extension is the
> intended happy path (see Releases, coming). Until that ships, this from-source build is the
> working way to get a `NcnnRunner` for your platform.

## Prerequisites

- Godot `4.5+`
- C++ toolchain for your platform
- [SCons](https://scons.org/)
- CMake (to build ncnn)
- `godot-cpp` checked out in `./godot-cpp` (matching your Godot version)

## Platform Setup (macOS / Linux / Windows)

### macOS

```bash
brew install scons cmake git
xcode-select --install
```

Apple Silicon note:
- If your ncnn static library is `arm64`-only, build the extension with `arch=arm64`.
- `arch=universal` requires ncnn to include both `arm64` and `x86_64`.

### Linux (Ubuntu/Debian example)

```bash
sudo apt update
sudo apt install -y build-essential scons cmake git
```

### Windows

Use **x64 Native Tools Command Prompt for VS** (or Developer PowerShell with MSVC set up), then:

```powershell
python -m pip install --upgrade pip
python -m pip install scons
```

Install CMake and Git with your preferred method (`winget`, installer, etc.).

## Project Setup

### 1) Clone dependencies

From repository root, clone dependencies into the expected paths:

```bash
# Build against the minimum Godot you want to support (4.5) — the resulting binary also runs on newer.
git clone -b 4.5 https://github.com/godotengine/godot-cpp.git
mkdir -p thirdparty
git clone https://github.com/Tencent/ncnn.git thirdparty/ncnn
```

Directory layout:

```text
.
├── godot-cpp/
└── thirdparty/
    └── ncnn/
```

### 2) Build godot-cpp bindings

From repository root:

```bash
cd godot-cpp
scons platform=<platform> target=template_debug
scons platform=<platform> target=template_release
cd ..
```

Use `platform=macos`, `platform=linux`, or `platform=windows`.

### 3) Build ncnn as static library

From repository root:

```bash
cmake -S thirdparty/ncnn -B thirdparty/ncnn/build \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF \
  -DBUILD_SHARED_LIBS=OFF

cmake --build thirdparty/ncnn/build --config Release
cmake --install thirdparty/ncnn/build --prefix thirdparty/ncnn/build/install
```

`SConstruct` looks for static ncnn here:

- macOS/Linux: `thirdparty/ncnn/build/install/lib/libncnn.a` (fallback: `thirdparty/ncnn/build/src/libncnn.a`)
- Windows: `thirdparty/ncnn/build/install/lib/ncnn.lib` (fallback: `libncnn.a`)

## Build The GDExtension

From repository root:

```bash
scons platform=<platform> target=template_debug
scons platform=<platform> target=template_release
```

Examples:

```bash
scons platform=macos arch=arm64 target=template_debug
scons platform=linux target=template_debug
scons platform=windows target=template_debug
```

Build output is written to `addons/godot_native_rl/bin/` and matched by `ncnn_runner.gdextension`.

### Enable the plugin in Godot

Open the project in the Godot editor, then go to **Project → Project Settings → Plugins** and enable **Godot Native RL**. This is a one-time step per clone. Headless training does not require this — the plugin only affects editor tooling.

## Universal / Multi-Architecture Builds

### macOS (single universal dylib)

Build ncnn per-architecture, then merge static libs with `lipo`:

- Do **not** build ncnn universal in one CMake build directory with `-DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"` on Apple Silicon.
- Use separate build directories per architecture.
- For the `x86_64` build on Apple Silicon, run CMake under Rosetta (`arch -x86_64`).

```bash
# arm64 build
cmake -S thirdparty/ncnn -B thirdparty/ncnn/build-arm64 \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build thirdparty/ncnn/build-arm64 --config Release
cmake --install thirdparty/ncnn/build-arm64 --prefix thirdparty/ncnn/install-arm64

# x86_64 build
arch -x86_64 cmake -S thirdparty/ncnn -B thirdparty/ncnn/build-x86_64 \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_OSX_ARCHITECTURES=x86_64
arch -x86_64 cmake --build thirdparty/ncnn/build-x86_64 --config Release
arch -x86_64 cmake --install thirdparty/ncnn/build-x86_64 --prefix thirdparty/ncnn/install-x86_64

# merge to universal lib path expected by SConstruct
mkdir -p thirdparty/ncnn/build/install/lib
lipo -create \
  thirdparty/ncnn/install-arm64/lib/libncnn.a \
  thirdparty/ncnn/install-x86_64/lib/libncnn.a \
  -output thirdparty/ncnn/build/install/lib/libncnn.a
```

Then build extension as universal:

```bash
scons platform=macos arch=universal target=template_debug
scons platform=macos arch=universal target=template_release
```

### Linux (ship multiple architecture-specific `.so` files)

Linux does not typically use one universal/fat `.so` for Godot extensions. Build once per architecture and package both artifacts:

```bash
scons platform=linux arch=x86_64 target=template_debug
scons platform=linux arch=arm64 target=template_debug
```

Add architecture-specific entries in `.gdextension`, for example:

```ini
linux.debug.x86_64 = "res://addons/godot_native_rl/bin/libncnn_runner.linux.template_debug.x86_64.so"
linux.debug.arm64 = "res://addons/godot_native_rl/bin/libncnn_runner.linux.template_debug.arm64.so"
```

### Windows (ship multiple architecture-specific `.dll` files)

Windows also normally uses one DLL per architecture. Build each architecture and package both:

```powershell
scons platform=windows arch=x86_64 target=template_debug
scons platform=windows arch=arm64 target=template_debug
```

Add architecture-specific entries in `.gdextension`, for example:

```ini
windows.debug.x86_64 = "res://addons/godot_native_rl/bin/libncnn_runner.windows.template_debug.x86_64.dll"
windows.debug.arm64 = "res://addons/godot_native_rl/bin/libncnn_runner.windows.template_debug.arm64.dll"
```

## Cross-Compiling From a macOS Host (Windows / Linux / Android / iOS)

All four non-host platforms can be built from a macOS (Apple Silicon) host. Each needs **its own
ncnn static lib** (built for the target) plus the extension; the toolchain differs per target:

| Target | Toolchain | Install |
| --- | --- | --- |
| Windows x86_64 | **zig** (`zig cc`/`zig c++`) via gcc/mingw-named shims | `brew install zig` |
| Linux x86_64 | **zig** via gcc-named shims | `brew install zig` |
| iOS (device + simulator) | **Apple clang** (full Xcode, not just CLT) | Xcode.app |
| Android (arm64-v8a, x86_64) | **Android NDK** clang | `brew install --cask android-ndk` (+ see trap) |

`SConstruct` has three knobs that make this work (native macOS builds are unaffected):

- It prefers a **per-target ncnn** at `thirdparty/ncnn/build-<platform>-<arch>/install` over the
  host `build/install`, so each target links the right arch.
- `ncnn_openmp=no` skips the libgomp link (cross ncnn is built `NCNN_OPENMP=OFF` — no GNU OpenMP
  runtime exists for these toolchains).
- For linux/windows/android it pins the shared-object suffix to `.o` (clang rejects SCons's default
  `.os` extension) and the library suffix to `.so`/`.dll`.

### Windows + Linux (zig)

zig ships a full cross clang/lld but no Android/iOS system libs, so it covers **Windows and Linux
only**. Create shims with the exact compiler names godot-cpp expects and have them inject the
`-target` triple, pin zig's cache dir (SCons/CMake scrub `HOME`), and drop gcc/mingw-only flags
clang rejects (`-static`, `-fno-gnu-unique`, `-m64`, `-march=x86-64`; translate `-Wl,-R,` →
`-Wl,-rpath,`). Keep ncnn's per-file SIMD flags (`-mavx2`/`-msse2`/...).

```bash
# Shim driver (one file). Names: linux -> cc/c++/gcc/g++/ar/ranlib;
# windows -> x86_64-w64-mingw32-{gcc,g++,gcc-ar,ranlib} + ar/ranlib.
# Each compiler shim sets ZIG_LANG=cc|c++ and ZIG_TARGET=x86_64-linux-gnu|x86_64-windows-gnu
# then execs the driver; ar/ranlib shims exec `zig ar`/`zig ranlib`.
```

ncnn (per target, via a CMake toolchain file pointing `CMAKE_C/CXX_COMPILER` at the shims and
`CMAKE_SYSTEM_NAME=Linux|Windows`):

```bash
cmake -S thirdparty/ncnn -B thirdparty/ncnn/build-linux-x86_64 \
  -DCMAKE_TOOLCHAIN_FILE=<toolchain-linux.cmake> -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
  -DNCNN_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DNCNN_SHARED_LIB=OFF \
  -DNCNN_OPENMP=OFF -DNCNN_VULKAN=OFF -DNCNN_THREADS=ON
cmake --build thirdparty/ncnn/build-linux-x86_64 -j8
cmake --install thirdparty/ncnn/build-linux-x86_64 --prefix thirdparty/ncnn/build-linux-x86_64/install
```

Extension (shims on `PATH`):

```bash
PATH="<shims-linux>:$PATH"   scons platform=linux   arch=x86_64 target=template_debug ncnn_openmp=no
PATH="<shims-windows>:$PATH" scons platform=windows arch=x86_64 target=template_debug ncnn_openmp=no
```

Result: a Linux `.so` depending only on libc/libm/libpthread/libdl (no libstdc++/libgomp), and a
self-contained Windows `.dll` importing only UCRT + KERNEL32 (no mingw runtime DLLs).

### iOS (device + simulator)

Needs **full Xcode** (the iOS SDK ships only inside Xcode.app; the standalone Command Line Tools
carry the macOS SDK only). No zig. Build ncnn for device arm64 and both simulator arches, then
`lipo` the two simulator arches into one fat lib:

```bash
# device (iphoneos) and simulator (iphonesimulator) builds, per arch:
cmake -S thirdparty/ncnn -B thirdparty/ncnn/build-ios-arm64 \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 -DNCNN_OPENMP=OFF -DNCNN_VULKAN=OFF <...static flags...>
# repeat with SYSROOT=iphonesimulator for arm64 and x86_64, then:
lipo -create <sim-arm64>/libncnn.a <sim-x86_64>/libncnn.a -output build-ios-universal/install/lib/libncnn.a
cp -R <sim-arm64-install>/include build-ios-universal/install/include
```

Extension (device = arm64; simulator = universal):

```bash
scons platform=ios arch=arm64     target=template_debug                    # -> build-ios-arm64
scons platform=ios arch=universal target=template_debug ios_simulator=yes  # -> build-ios-universal
```

godot-cpp appends `.simulator` to the simulator output's suffix, so device and simulator never
collide (`...arm64.dylib` vs `...universal.simulator.dylib`). Bundle them into an `.xcframework`
per config and reference that (no arch key) in the manifest:

```bash
xcodebuild -create-xcframework \
  -library addons/godot_native_rl/bin/libncnn_runner.ios.template_debug.arm64.dylib \
  -library addons/godot_native_rl/bin/libncnn_runner.ios.template_debug.universal.simulator.dylib \
  -output  addons/godot_native_rl/bin/libncnn_runner.ios.template_debug.xcframework
```
```ini
ios.debug = "res://addons/godot_native_rl/bin/libncnn_runner.ios.template_debug.xcframework"
ios.release = "res://addons/godot_native_rl/bin/libncnn_runner.ios.template_release.xcframework"
```

### Android (arm64-v8a + x86_64)

Needs the **Android NDK** (zig can't target Android — no bionic libc).
`brew install --cask android-ndk`.

> **Gatekeeper trap.** The cask sets `com.apple.quarantine` and the NDK is a *code-signed `.app`
> bundle* — every NDK clang call then hangs on a first-run Gatekeeper assessment, and the xattr
> can't be stripped in place (sealed bundle → "Operation not permitted"; Homebrew has removed
> `--no-quarantine`). Fix: copy the toolchain **out** of the bundle (detaches the seal):
> ```bash
> ditto --noqtn /opt/homebrew/Caskroom/android-ndk/*/AndroidNDK*.app/Contents/NDK ~/android-ndk-r29
> xattr -cr ~/android-ndk-r29
> ```
> Use `~/android-ndk-r29` as the NDK root thereafter.

ncnn (per ABI, via the NDK's CMake toolchain):

```bash
NDK=~/android-ndk-r29
cmake -S thirdparty/ncnn -B thirdparty/ncnn/build-android-arm64 \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_OPENMP=OFF -DNCNN_VULKAN=OFF <...static flags...>   # repeat with ANDROID_ABI=x86_64
```

Extension — the **`ANDROID_HOME=` (empty) arg is required**: godot-cpp's `android.py` does
`if env["ANDROID_HOME"]` but its default resolves to `None` so the key is never created
(`KeyError`); passing empty makes it fall through to `ANDROID_NDK_ROOT`:

```bash
ANDROID_NDK_ROOT=~/android-ndk-r29 scons platform=android arch=arm64  target=template_debug ncnn_openmp=no ANDROID_HOME=
ANDROID_NDK_ROOT=~/android-ndk-r29 scons platform=android arch=x86_64 target=template_debug ncnn_openmp=no ANDROID_HOME=
```
```ini
android.debug.arm64 = "res://addons/godot_native_rl/bin/libncnn_runner.android.template_debug.arm64.so"
android.debug.x86_64 = "res://addons/godot_native_rl/bin/libncnn_runner.android.template_debug.x86_64.so"
```

The Android `.so` depends on `libc++_shared.so`, which Godot's Android export template ships.
arm64-v8a covers real devices and the Apple-Silicon emulator; x86_64 covers the emulator on x86
hosts.

> **Status:** these cross-builds are verified by binary inspection (correct ELF/PE/Mach-O, the
> `ncnn_runner_library_init` entry symbol exported, clean dependencies) but have **not yet been
> runtime-tested** on each target OS. The macOS arm64 build is the only one exercised by the test
> suite today.

## Web (WASM)

The web build is **single-threaded by design** (`NCNN_THREADS=OFF` + `scons threads=no`). That
means the exported game needs **no `SharedArrayBuffer`, and therefore no cross-origin-isolation
(COOP/COEP) headers** — it deploys to any static host (itch.io, GitHub Pages, an S3 bucket) with
zero server configuration. Our policies are small MLPs, so single-threaded inference is plenty.
This is the deployment story neither godot_rl (Python server / ONNX) nor a .NET runtime can match.

### 1) Install + activate emscripten (emsdk)

```bash
git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
cd ~/emsdk && ./emsdk install 3.1.64 && ./emsdk activate 3.1.64
source ~/emsdk/emsdk_env.sh      # puts emcc on PATH for the current shell
emcc --version                   # expect 3.1.64
```

Pin **3.1.64** — it is known to compile against godot-cpp `4.5`. Other versions may work but are
untested here; the CI `web` leg pins the same version.

### 2) Build ncnn (static, single-threaded) + the GDExtension

```bash
cd /path/to/godot-native-rl
source ~/emsdk/emsdk_env.sh
scripts/cross/build_web.sh        # NCNN_JOBS=N to cap ncnn compile parallelism
```

This produces:

```
bin/libncnn_runner.web.template_debug.wasm32.nothreads.wasm
bin/libncnn_runner.web.template_release.wasm32.nothreads.wasm
```

The `.nothreads` suffix is godot-cpp's single-threaded variant tag. The manifest
(`ncnn_runner.gdextension`) maps these to the **`web.debug.wasm32` / `web.release.wasm32`** keys
(no `threads` feature tag) — matching godot-cpp's own convention for a no-threads web library.

### 3) Export a game to web (single-threaded, extension support)

In the Godot editor: **Project → Export → Add… → Web**, then in the preset options:

- **Thread Support: OFF** (selects the single-threaded template; pairs with our `nothreads` binary)
- **Extension Support: ON** (selects the `dlink` template, required for any GDExtension on web)

Export to `build/web/index.html`. (Or headless, once a `Web` preset exists:
`godot --headless --path . --export-debug "Web" build/web/index.html`.) A correct export bundles
`index.side.wasm` (the extension-capable engine side-module) **and** the
`libncnn_runner.web.…nothreads.wasm` extension binary next to `index.html`; the model
`.param`/`.bin` ride inside `index.pck` and are read game-side via `FileAccess` (the deploy-side
controllers load models from byte buffers, not file paths — ncnn cannot `fopen` inside the
browser-served `.pck`).

> **CRITICAL — pack the model files (every platform, not just web).** Godot's exporter only
> auto-includes recognized *resources*. The ncnn `.param`/`.bin` are raw data files referenced by
> string path, so the dependency scanner **does not pack them** and the controller fails at runtime
> with `cannot read model files …`. You must add them to the export preset's
> **Resources → "Filters to export non-resource files/folders"** include filter:
>
> ```
> *.ncnn.param, *.ncnn.bin
> ```
>
> (In `export_presets.cfg` this is `include_filter="*.ncnn.param, *.ncnn.bin"`.) This applies to
> desktop and mobile exports too — running from the editor masks it because `res://` is the real
> filesystem there; an exported `.pck` only contains what the filter packs. Verify with
> `strings build/web/index.pck | grep 7767517` (ncnn's param magic — a hit means the model is in).

### 4) Serve it — no special headers

```bash
cd build/web && python3 -m http.server 8060
# open http://localhost:8060/
```

A plain static server sends **no COOP/COEP headers** — and the game still runs, because the build
is single-threaded. That is the proof the deployment works on any host.

**Verified in-browser:** the `chase_the_target` policy runs native ncnn inference in the browser
(the agent chases the target) served this way. See `docs/dev/img/web-chase-proof.png`.

> If you instead build a **multi-threaded** web variant (`threads=yes`, `NCNN_THREADS=ON`), the
> manifest key becomes `web.debug.threads.wasm32` and the host **must** send COOP/COEP headers
> (`Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy: require-corp`) so the
> browser grants `SharedArrayBuffer`. itch.io and plain GitHub Pages do not, which is why the
> default here is single-threaded.

## Manual ONNX → ncnn conversion (internals)

> The one-command path (`scripts/export_to_ncnn.py`) is documented for game developers in
> [docs/guide/deploying.md](../guide/deploying.md). This section is the manual pnnx breakdown.

### 1) Install pnnx

```bash
python3 -m pip install --user numpy pnnx
```

### 2) Convert model

Basic conversion:

```bash
pnnx your_model.onnx
```

For models that need explicit shape hints, add `inputshape`:

```bash
pnnx your_model.onnx inputshape=[1,8]
```

Generated outputs include:

- `your_model.ncnn.param`
- `your_model.ncnn.bin`

### 4) Verify the conversion (recommended)

`pnnx` is reliable for the simple MLP policies typical of RL agents, but a conversion can
silently go wrong (an unsupported operator, a wrong blob, numerical drift). For an RL policy this
failure is **silent** — the agent behaves subtly wrong instead of crashing — so verifying is worth
the few seconds. `scripts/verify_ncnn_parity.py` runs the ONNX (via onnxruntime) and the ncnn
model over 50 random observations and confirms the **argmax matches** on every one:

```bash
python scripts/verify_ncnn_parity.py \
  your_model.onnx your_model.ncnn.param your_model.ncnn.bin in0 out0
```

Expected: `PARITY OK: 50/50 argmax match between ONNX and ncnn`. (godot-rl-exported policies
convert to the blob names `in0` / `out0` — `pnnx` prunes the vestigial `state_ins` input.)
Verification requires `onnxruntime` and the `ncnn` Python package.

### The "fast path" (skip verification)

Verification is the **default**; skipping it is an explicit opt-out, never the reverse — defaulting
to "skip" would quietly ship broken models.

Skipping is reasonable when:

- You have already verified this model **architecture** once and are only re-converting new weights
  of the same shape (`pnnx` is deterministic, so re-verification is redundant), or
- You are in a CI/iteration loop where speed matters and a separate periodic verification gate exists.

The fast path also avoids the `onnxruntime` + `ncnn` dependencies — plain `pnnx` is enough to
convert; those two packages are only needed to verify.

> A one-command convenience helper (`scripts/export_to_ncnn.py`) that wraps convert + verify with a
> `--skip-verify` opt-out is planned. Until then, run `pnnx` and `verify_ncnn_parity.py` as above.
