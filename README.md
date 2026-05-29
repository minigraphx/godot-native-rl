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

@export_file("*.param") var model_param_path: String = "res://models/test_mlp.ncnn.param"
@export_file("*.bin") var model_bin_path: String = "res://models/test_mlp.ncnn.bin"
@export var input_blob_name: String = "in0"
@export var output_blob_name: String = "out0"
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
    if _native_runner == null or not _native_runner.is_model_loaded():
        push_error("NcnnAgentHelper.get_action: model not loaded.")
        return null

    var packed_obs := PackedFloat32Array(observations)
    if action_mode == ActionMode.DISCRETE_ARGMAX:
        return _native_runner.run_discrete_action(packed_obs)

    return _native_runner.run_inference(packed_obs)

func get_action_from_image(image: Image, normalize_to_zero_one: bool = true) -> PackedFloat32Array:
    if _native_runner == null or not _native_runner.is_model_loaded():
        push_error("NcnnAgentHelper.get_action_from_image: native runner is not ready.")
        return PackedFloat32Array()
    return _native_runner.run_inference_image(image, normalize_to_zero_one)
```

## Training Bridge (godot_rl_agents-compatible)

Training uses a bridge that speaks the [`godot_rl_agents`](https://github.com/edbeeching/godot_rl_agents) wire protocol, so you can train with the existing `godot-rl` Python package (Stable-Baselines3 / CleanRL / Sample-Factory) — no custom Python required. Two reusable scripts provide it:

- `sync.gd` (`NcnnSync`): connects to the Python trainer as a TCP client (default port `11008`), performs the handshake, sends `env_info`, then runs the synchronous step loop, pausing the SceneTree between steps and honoring `action_repeat` / `speed_up` / command-line args.
- `ncnn_ai_controller_2d.gd` (`NcnnAIController2D`): the base class your agents extend. Agents are discovered via the `"AGENT"` group.

### Agent Contract

Agents extend `NcnnAIController2D` (auto-added to group `"AGENT"`) and implement:

- `get_obs() -> Dictionary` returning `{"obs": [...]}`
- `get_reward() -> float`
- `get_action_space() -> Dictionary`, e.g. `{"move": {"size": 5, "action_type": "discrete"}}`
- `set_action(action)` to apply one action

`get_obs_space`, `get_done`, `reset`, and the other contract methods are provided by the base class.

### Wire-Up In Scene

1. Add one node with script `sync.gd` (`NcnnSync`) and set its `control_mode` to `Training`.
2. Add your agent node(s) extending `NcnnAIController2D` (they auto-join group `"AGENT"`).
3. Start the Python trainer (e.g. `gdrl`); Godot connects to it on launch.

### Protocol

Messages are length-prefixed (4-byte little-endian) JSON via `StreamPeerTCP.put_string` / `get_string`, matching godot_rl. Per step Godot sends:

```json
{"type":"step","obs":[{"obs":[0.1,0.2]}],"reward":[0.0],"done":[false]}
```

and the trainer replies:

```json
{"type":"action","action":[{"move":2}]}
```

> A complete, runnable 2D example (chase-the-target) with an end-to-end train → convert → ncnn-inference walkthrough is coming as a dedicated example and tutorial.

## Notes

- If `input_shape` is empty, `run_inference` maps input to a 1D `ncnn::Mat`.
- Set `input_shape` to `[w]`, `[w, h]`, or `[w, h, c]` to reshape flat float arrays before inference.
- `run_inference_image` converts to RGB8 internally and can normalize pixels to `[0, 1]`.
- Output is returned as a flattened `PackedFloat32Array`.
- `run_discrete_action` returns the argmax index over the output tensor (flattened).
- Default blob names are `"input"` and `"output"`; override when your model uses different tensor names.

## Test Model Guide

- See [docs/pytorch_mlp_test_model.md](/Users/andreas/Documents/Godot Native RL/docs/pytorch_mlp_test_model.md) for creating and exporting a tiny PyTorch MLP test model to ncnn.
