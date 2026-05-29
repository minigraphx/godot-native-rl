# Godot Native RL (ncnn GDExtension)

Minimal Godot 4.6+ GDExtension (C++) for running ncnn inference from Godot.

## What This Repository Provides

- `NcnnRunner` C++ node class exposed to Godot.
- `load_model(param_path, bin_path)` to load ncnn models.
- `run_inference(input: PackedFloat32Array)` to run a forward pass.
- `input_shape` (optional) to map flat float arrays to 1D/2D/3D ncnn input tensors.
- `run_inference_image(image: Image, normalize_to_zero_one := true)` for RGB image input.
- `run_discrete_action(input: PackedFloat32Array)` for argmax-based discrete action selection.
- `is_model_loaded()` to guard inference calls from GDScript.
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

## Convert ONNX To ncnn

Use `pnnx` (recommended) to convert ONNX models to ncnn files.

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

### 3) Use in this Godot project

Move/copy the files to a project folder such as `models/`:

```bash
mkdir -p models
cp your_model.ncnn.param models/
cp your_model.ncnn.bin models/
```

Then set:

- `model_param_path = "res://models/your_model.ncnn.param"`
- `model_bin_path = "res://models/your_model.ncnn.bin"`

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

enum ActionMode {
    CONTINUOUS,
    DISCRETE_ARGMAX,
}

@export_file("*.param") var model_param_path: String
@export_file("*.bin") var model_bin_path: String
@export var input_blob_name: String = "input"
@export var output_blob_name: String = "output"
@export var input_shape: PackedInt32Array = PackedInt32Array()
@export_enum("Continuous", "Discrete Argmax") var action_mode: int = ActionMode.CONTINUOUS

var _native_runner: NcnnRunner

func _ready() -> void:
    _native_runner = NcnnRunner.new()
    add_child(_native_runner)

    _native_runner.input_blob_name = input_blob_name
    _native_runner.output_blob_name = output_blob_name
    _native_runner.input_shape = input_shape

    var absolute_param = ProjectSettings.globalize_path(model_param_path)
    var absolute_bin = ProjectSettings.globalize_path(model_bin_path)

    var ok = _native_runner.load_model(absolute_param, absolute_bin)
    if not ok:
        push_error("Failed to load ncnn model.")

func get_action(observations: Array[float]) -> Variant:
    if not _native_runner.is_model_loaded():
        push_error("NcnnAgentHelper.get_action: model not loaded.")
        return null

    var packed_obs := PackedFloat32Array(observations)
    if action_mode == ActionMode.DISCRETE_ARGMAX:
        return _native_runner.run_discrete_action(packed_obs)

    return _native_runner.run_inference(packed_obs)

func get_action_from_image(image: Image, normalize_to_zero_one: bool = true) -> PackedFloat32Array:
    return _native_runner.run_inference_image(image, normalize_to_zero_one)
```

## Training Bridge (Milestone 3 Step 1/2)

Two new scripts are included:

- `tcp_client.gd` (`TcpClientBridge`): TCP client with reconnect, request timeout, and response matching via `request_id`.
- `sync_node.gd` (`SyncNode`): finds agents in a group, batches observations, sends one request, and routes actions back.

### Agent Contract

Agents participating in training should:

- be added to group `ncnn_training_agents` (or configure `agent_group_name` on `SyncNode`),
- implement `collect_observation() -> Array` or `PackedFloat32Array`,
- implement `apply_training_action(action)` to consume one returned action.

### Wire-Up In Scene

1. Add one node with script `tcp_client.gd`.
2. Add one node with script `sync_node.gd`.
3. Set `sync_node.tcp_client_path` to your TCP client node.
4. Add your agent nodes to group `ncnn_training_agents`.

### JSON Line Protocol

`TcpClientBridge` uses newline-delimited JSON (`\\n` framing).

Request example:

```json
{"type":"action_request","request_id":1,"observations":[[0.1,0.2],[0.3,0.4]],"metadata":{"agent_count":2}}
```

Response example:

```json
{"request_id":1,"actions":[1,0]}
```

Optional error response:

```json
{"request_id":1,"ok":false,"error":"bad payload"}
```

## Notes

- If `input_shape` is empty, `run_inference` maps input to a 1D `ncnn::Mat`.
- Set `input_shape` to `[w]`, `[w, h]`, or `[w, h, c]` to reshape flat float arrays before inference.
- `run_inference_image` converts to RGB8 internally and can normalize pixels to `[0, 1]`.
- Output is returned as a flattened `PackedFloat32Array`.
- `run_discrete_action` returns the argmax index over the output tensor (flattened).
- Default blob names are `"input"` and `"output"`; override when your model uses different tensor names.

## Test Model Guide

- See [docs/pytorch_mlp_test_model.md](/Users/andreas/Documents/Godot Native RL/docs/pytorch_mlp_test_model.md) for creating and exporting a tiny PyTorch MLP test model to ncnn.
