# Godot Native RL (ncnn GDExtension)

Minimal Godot 4.6+ GDExtension (C++) for running ncnn inference from Godot.

## What This Repository Provides

- `NcnnRunner` C++ node class exposed to Godot.
- `load_model(param_path, bin_path)` to load ncnn models.
- `run_inference(input: PackedFloat32Array)` to run a forward pass.
- Static ncnn linking from `thirdparty/ncnn`.

## Repository Layout

- `SConstruct` - build script for the GDExtension.
- `ncnn_runner.gdextension` - Godot extension manifest.
- `src/` - C++ source for `NcnnRunner` and registration.
- `thirdparty/ncnn/` - expected location of ncnn sources/build artifacts.
- `godot-cpp/` - expected local checkout of Godot C++ bindings.

## Prerequisites

- Godot `4.6+`
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
git clone -b 4.6 https://github.com/godotengine/godot-cpp.git
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

## Godot Usage

Example helper script:

```gdscript
class_name NcnnAgentHelper
extends Node

@export_file("*.param") var model_param_path: String
@export_file("*.bin") var model_bin_path: String
@export var input_blob_name: String = "input"
@export var output_blob_name: String = "output"

var _native_runner: NcnnRunner

func _ready() -> void:
    _native_runner = NcnnRunner.new()
    add_child(_native_runner)

    _native_runner.input_blob_name = input_blob_name
    _native_runner.output_blob_name = output_blob_name

    var absolute_param = ProjectSettings.globalize_path(model_param_path)
    var absolute_bin = ProjectSettings.globalize_path(model_bin_path)

    var ok = _native_runner.load_model(absolute_param, absolute_bin)
    if not ok:
        push_error("Failed to load ncnn model.")

func get_action(observations: Array[float]) -> PackedFloat32Array:
    return _native_runner.run_inference(PackedFloat32Array(observations))
```

## Notes

- `run_inference` currently maps input to a 1D `ncnn::Mat` of float32.
- Output is returned as a flattened `PackedFloat32Array`.
- Default blob names are `"input"` and `"output"`; override when your model uses different tensor names.

## Test Model Guide

- See [docs/pytorch_mlp_test_model.md](/Users/andreas/Documents/Godot Native RL/docs/pytorch_mlp_test_model.md) for creating and exporting a tiny PyTorch MLP test model to ncnn.
