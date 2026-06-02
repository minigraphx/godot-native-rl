# Godot Native RL (ncnn GDExtension)

Minimal Godot 4.6+ GDExtension (C++) for running ncnn inference from Godot.

> **Should you use ncnn or stick with ONNX Runtime?** See the balanced decision guide:
> [docs/ncnn_vs_onnx.md](docs/ncnn_vs_onnx.md) — honest pros/cons on both sides (web/console/mobile,
> INT8, conversion, licensing), including where ONNX Runtime is genuinely the better choice.

> **Library moved (2026-05-31):** the reusable scripts now live under
> `addons/godot_native_rl/` (controllers, `sync.gd`, `reward/`, `sensors/`). `class_name`-based
> usage is unchanged — `extends NcnnAIController2D` / `NcnnAIController3D`, `NcnnSync`,
> `RewardBuilder`, `RaycastSensor2D/3D`, etc. all still resolve in the editor. If you `preload`
> old paths like `res://sync.gd` or `res://reward/…`, update them to `res://addons/godot_native_rl/…`.
> In headless/CLI runs (no editor cache), prefer path-based `extends "res://addons/godot_native_rl/…"`
> over bare `class_name` (see CLAUDE.md). The compiled GDExtension (`ncnn_runner.gdextension`,
> `bin/`) still lives at the project root.

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

> New here? Read [docs/ncnn_vs_onnx.md](docs/ncnn_vs_onnx.md) first to decide whether converting to
> ncnn is the right call for your deployment target — for fast server-side iteration the stock
> `godot_rl_agents` ONNX path may suit you better.

### One command (recommended)

Convert and verify in a single step — auto-derives `inputshape` from the ONNX, checks ncnn↔ONNX
argmax/logit parity, and cleans up pnnx intermediates:

    .venv-train/bin/python scripts/export_to_ncnn.py models/your_model.onnx

Useful flags: `--skip-verify`, `--keep-intermediates`, `--inputshape '[1,N],[1]'`, `--outdir DIR`.
The manual `pnnx` + `verify_ncnn_parity.py` steps below are the underlying operations it wraps.

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

**VecNormalize policies:** if you trained with SB3 `VecNormalize`, export its stats with
`.venv-train/bin/python scripts/export_vecnormalize.py path/to/vec_normalize.pkl` and set the
controller's `obs_norm_stats_path` to the generated JSON. The addon replays the running mean/std
game-side before inference (the network itself does not carry them).

### INT8 quantization (mobile/edge deployment)

INT8 quantization reduces model size by ~4× and inference time by ~2–4× on mobile and edge devices.
This is a deployment moat: ONNX Runtime / Barracuda lack game-side INT8 support, but ncnn (statically
linked in this project) has INT8 built in — deploying an INT8 model requires **no C++ changes**.

**1. Build the quantize tools once** (not included in the pip `ncnn` wheel):

```bash
./scripts/build_ncnn_tools.sh
```

This builds `ncnn2table`, `ncnn2int8`, and `ncnnoptimize` from the vendored `thirdparty/ncnn/` source
into `thirdparty/ncnn/tools-bin/`. Uses `NCNN_SIMPLEOCV=ON` so OpenCV is not required.

**2. Export to INT8 (one command):**

```bash
.venv-train/bin/python scripts/export_int8.py models/your_model.ncnn.param \
  models/your_model.ncnn.bin --width W --height H --channels C --outdir models
```

This runs the full pipeline: optimize → KL-calibrate → `ncnn2int8` → argmax-agreement parity check.
Produces `your_model_int8.ncnn.{param,bin}` in `--outdir`. Flags: `--samples` (calibration images,
default 256), `--threshold` (argmax agreement rate, default 0.9), `--skip-verify`,
`--keep-intermediates`, `--in-blob`/`--out-blob`, `--tools-dir`.

For the committed synthetic-CNN fixture (`models/synthetic_cnn.ncnn.*`, 8×8×3):

```bash
.venv-train/bin/python scripts/export_int8.py models/synthetic_cnn.ncnn.param \
  models/synthetic_cnn.ncnn.bin --width 8 --height 8 --channels 3 --outdir models
```

**Calibration guidance:** for real policies, calibrate on **captured game frames** that are
representative of the actual observation distribution — the synthetic set is a regression fixture.
More samples (e.g. `--samples 1024`) improve histogram coverage for larger inputs.

**3. Deploy:** load the `*_int8.ncnn.{param,bin}` files with `NcnnRunner` exactly as you would an
fp32 model — the runtime INT8 path is compiled in, no changes required:

```gdscript
runner.load_model("res://models/your_model_int8.ncnn.param",
                  "res://models/your_model_int8.ncnn.bin")
```

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
You may optionally override `get_info() -> Dictionary` (default `{}`) — see **Per-agent `info`** below.

### Wire-Up In Scene

1. Add one node with script `sync.gd` (`NcnnSync`) and set its `control_mode` to `Training`.
2. Add your agent node(s) extending `NcnnAIController2D` (they auto-join group `"AGENT"`).
3. Start the Python trainer (e.g. `gdrl`); Godot connects to it on launch.

### Protocol

Messages are length-prefixed (4-byte little-endian) JSON via `StreamPeerTCP.put_string` / `get_string`, matching godot_rl. Per step Godot sends:

```json
{"type":"step","obs":[{"obs":[0.1,0.2]}],"reward":[0.0],"done":[false],"info":[{}]}
```

and the trainer replies:

```json
{"type":"action","action":[{"move":2}]}
```

### Socket timeouts

`NcnnSync` bounds both its connect and read loops so a missing or silent trainer can't hang a
headless run:

- `connect_timeout_sec` (default `10.0`) — give up connecting and fall back to human controls.
- `read_timeout_sec` (default `60.0`, matching godot_rl's `DEFAULT_TIMEOUT`) — if the trainer
  sends nothing, quit cleanly (exit code 0) instead of blocking forever.

Override per run via cmdline: `... res://scene.tscn read_timeout=120 connect_timeout=5`. A value
`<= 0` disables the timeout (waits forever).

### Per-agent `info`

Agents may override `get_info() -> Dictionary` (default `{}`) to attach per-step metadata sent to
the trainer in the step message's `info` field (godot_rl reads `info`, e.g. `{"is_success": true}`
for success-rate metrics). Backward-compatible: older trainers ignore it.

## Sensors

Reusable observation sources implementing the shared sensor interface
(`get_observation() -> Array`, `obs_size() -> int`). Compose them manually inside your
agent's `get_obs()` and concatenate with your other features.

- **`RaycastSensor2D`** (`sensors/raycast_sensor_2d.gd`) — an even fan of `n_rays` 2D rays
  across `cone_degrees`, centered on the node's forward. Each ray emits a *closeness* float:
  `0.0` for no hit, up to `~1.0` for a near obstacle. Configurable `ray_length`,
  `collision_mask`, `collide_with_areas`, `collide_with_bodies`.
- **`RaycastSensor3D`** (`sensors/raycast_sensor_3d.gd`) — an `n_rays_width × n_rays_height`
  grid of 3D rays across `horizontal_fov × vertical_fov`, centered on forward (−Z). Same
  closeness encoding and physics options.
- **`RelativePositionSensor2D`** (`sensors/relative_position_sensor_2d.gd`) — the egocentric
  position of a `target_path` node: a unit direction in the sensor's local frame plus a
  clipped, normalized distance, `[dir_x, dir_y, dist_norm]` (3 floats). `dist_norm =
  clamp(distance / max_distance, 0, 1)`. Answers "where is my target relative to me?"
  (`godot_rl` issue #177).
- **`RelativePositionSensor3D`** (`sensors/relative_position_sensor_3d.gd`) — the 3D form:
  `[dir_x, dir_y, dir_z, dist_norm]` (4 floats), direction in the sensor's local frame
  (forward = −Z), same `max_distance` clipping.
- **`CameraSensor`** (`sensors/camera_sensor.gd`) — image observations from a `SubViewport`
  (`godot_rl` issue #78). Dimension-agnostic: point it at a `SubViewport` holding a `Camera2D` or
  `Camera3D`. Unlike the float sensors above, it returns a **hex-encoded `String`** of raw `uint8`
  pixels (HWC, `[H, W, 3]` RGB or `[H, W, 1]` with `grayscale = true`), and contributes a
  `{"space": "box", "size": [...]}` obs-space entry rather than a flat size. Compose it manually:
  `obs[sensor.get_observation_key()] = sensor.get_observation()` and merge
  `sensor.get_obs_space_entry()` into your `get_obs_space()`. The `observation_key` **must contain
  `"2d"`** even for a `Camera3D` view (name it e.g. `"camera_3d_2d"`) — `godot_rl` routes image obs
  on that substring, decoding to `Box(0, 255, uint8)` for
  SB3's `MultiInputPolicy`/`NatureCNN` (which does its own `/255`). Size the obs by sizing the
  `SubViewport`. *Native ncnn **deploy** works for **discrete, RGB** image policies: set the agent's
  `control_mode = NCNN_INFERENCE` and override `get_inference_image()` to return
  `camera.get_image()` — the controller feeds it to `NcnnRunner.run_inference_image` (RGB8 + `/255`)
  and acts on the argmax. Grayscale and continuous image policies are follow-ups (backlog item 38/21).*

Pure ray geometry lives in `sensors/raycast_math.gd`; the relative-position frame/clip math
lives in `sensors/relative_position_math.gd`; the camera shape + hex encoding lives in
`sensors/camera_obs_math.gd` (all headless-unit-tested).
This encoding matches `godot_rl`'s raycast convention, so ported environments behave the same —
and the observations feed `NcnnRunner` for zero-runtime deployment on mobile/web/console.

## Examples

### Chase The Target (2D)

A complete, runnable 2D example: an agent learns to chase a relocating target, trained with
`godot-rl` over the `NcnnSync` bridge and deployed via native `NcnnRunner` inference. It ships
with a pre-trained model so it runs out of the box.

- Scene: `examples/chase_the_target/chase_the_target.tscn`
- From-scratch tutorial: [docs/examples/chase_the_target_tutorial.md](docs/examples/chase_the_target_tutorial.md)

Run the headless checks (unit tests + protocol + inference smoke + trained-chase):

```bash
./test/run_tests.sh
```

### 3D Raycast Rover

A tank-steered 3D rover (`examples/rover_3d/`) that uses a `RaycastSensor3D` to avoid a fixed
obstacle field and reach a goal it senses egocentrically. Demonstrates `NcnnAIController3D` +
`RaycastSensor3D` + declarative `RewardBuilder`/`RewardAdapter` reward. Discrete tank actions
(`idle / forward / turn-left / turn-right`); observation = 5 ray closeness values + `[sin, cos]`
of the goal bearing + normalized distance. It ships with a pre-trained ncnn model
(`examples/rover_3d/models/rover_policy.ncnn.*`), a deterministic trained-rover behavioral check, and
a golden-inference regression. The headless smoke test (`test/integration/rover_smoke_scene.tscn`)
exercises the full obs + physics-raycast pipeline.

Train with `./scripts/train_rover.sh`. Training is **checkpoint/resume-capable**: it saves to
`models/rover_checkpoints/` every 25k steps and **auto-resumes** from the latest checkpoint on
re-run, so an interrupted run continues instead of restarting. `FRESH=1` starts from scratch;
`CHECKPOINT_FREQ=N` changes the interval.

- **Train faster (parallel):** `SCENE=res://examples/rover_3d/rover_3d_train_parallel.tscn
  ./scripts/train_rover.sh` tiles 8 rover worlds in one Godot process via `ParallelArena`, so
  godot-rl vectorizes over 8 agents (~Nx samples/sec) — see "Parallel training" below.
- **Refine later:** because checkpoints are kept, raise the target and re-run to keep improving:
  `TIMESTEPS=600000 ./scripts/train_rover.sh` resumes from the latest checkpoint toward 600k.
- **Export a checkpoint without finishing:** `scripts/export_checkpoint.py` loads a checkpoint (latest
  by default) and writes the ONNX (then `scripts/export_to_ncnn.py` converts it) — non-destructive, so
  the checkpoints remain for further refinement.
- **⚠️ Apple Silicon / macOS — do not let the machine sleep while training.** Sleep suspends the
  headless Godot client; the Godot side now self-terminates on `read_timeout_sec` (default 60s), but
  the trainer still blocks on the dead socket and the run stalls (you'll
  see `total_timesteps` stop advancing at 0% CPU). Keep it awake for the whole run, e.g.
  `caffeinate -is ./scripts/train_rover.sh` (`-i` prevents idle sleep, `-s` prevents sleep on AC). If
  a run does stall, kill it and just re-run — it resumes from the last checkpoint (≤25k steps lost).
  For unattended long runs prefer an always-on machine or CI; the local trainer + Godot can't survive
  the host sleeping.

#### Parallel training (faster)

Training throughput is bottlenecked by the Godot environment (physics + raycasts + per-step socket
round-trip), not the tiny PPO net. `ParallelArena` (in `addons/godot_native_rl/training/`) tiles N
copies of an agent "world" sub-scene in one Godot process, spaced far enough apart that each
agent's raycasts only see its own obstacles. `NcnnSync` already batches every agent in the `AGENT`
group, and godot-rl auto-detects `n_agents` from the handshake, so this is a scene-only change —
**the Python trainer is unchanged**.

> Run training scenes via `./scripts/train_rover.sh` (it starts the Python trainer first). Launching
> a training scene headless on its own will hang — the `Sync` node waits for a trainer on port 11008.

Run the parallel rover training scene (8 agents) instead of the single-agent one:

```bash
SCENE=res://examples/rover_3d/rover_3d_train_parallel.tscn ./scripts/train_rover.sh
```

Measured on this machine via `./scripts/throughput_compare.sh` (8k timesteps, fresh): single-agent
**55.9 samples/s** vs parallel-×8 **347.8 samples/s** — a **6.2× speedup** (sub-linear at short run
lengths because of fixed startup/handshake overhead; it approaches 8× over a full run).

To reuse it for your own env: make a world sub-scene containing exactly one `AGENT`-group agent
(keep its game logic in the world's local frame so it's tile-offset-safe), add a `ParallelArena`
node, set `world_scene` to it, and pick `count` / `spacing` (spacing must exceed your arena extent
+ ray length).

### Hide & Seek (2D, parameter-sharing self-play)

- **Hide & Seek** (`examples/hide_and_seek/`) — 2D 1v1 self-play (parameter sharing): a seeker vs a hider trained by one shared PPO policy, with line-of-sight-gated vision and occluding walls. See `examples/hide_and_seek/README.md`.

## Notes

- If `input_shape` is empty, `run_inference` maps input to a 1D `ncnn::Mat`.
- Set `input_shape` to `[w]`, `[w, h]`, or `[w, h, c]` to reshape flat float arrays before inference.
- `run_inference_image` converts to RGB8 internally and can normalize pixels to `[0, 1]`.
- Output is returned as a flattened `PackedFloat32Array`.
- `run_discrete_action` returns the argmax index over the output tensor (flattened).
- Default blob names are `"input"` and `"output"`; override when your model uses different tensor names.

## Test Model Guide

- See [docs/pytorch_mlp_test_model.md](/Users/andreas/Documents/Godot Native RL/docs/pytorch_mlp_test_model.md) for creating and exporting a tiny PyTorch MLP test model to ncnn.
