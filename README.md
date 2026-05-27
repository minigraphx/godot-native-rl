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

## Setup

### 1) Add dependencies

Clone or place dependencies in this structure:

```text
.
├── godot-cpp/
└── thirdparty/
    └── ncnn/
```

### 2) Build ncnn as static library

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

Example:

```bash
scons platform=macos target=template_debug
```

Other common targets:

```bash
scons platform=linux target=template_debug
scons platform=windows target=template_debug
```

Build output is written to `bin/` and matched by `ncnn_runner.gdextension`.

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
