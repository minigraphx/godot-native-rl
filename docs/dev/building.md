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

Build output is written to `bin/` and matched by `ncnn_runner.gdextension`.

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
linux.debug.x86_64 = "res://bin/libncnn_runner.linux.template_debug.x86_64.so"
linux.debug.arm64 = "res://bin/libncnn_runner.linux.template_debug.arm64.so"
```

### Windows (ship multiple architecture-specific `.dll` files)

Windows also normally uses one DLL per architecture. Build each architecture and package both:

```powershell
scons platform=windows arch=x86_64 target=template_debug
scons platform=windows arch=arm64 target=template_debug
```

Add architecture-specific entries in `.gdextension`, for example:

```ini
windows.debug.x86_64 = "res://bin/libncnn_runner.windows.template_debug.x86_64.dll"
windows.debug.arm64 = "res://bin/libncnn_runner.windows.template_debug.arm64.dll"
```

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
