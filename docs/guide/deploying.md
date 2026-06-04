# Deploying (native ncnn inference)

## Convert your trained model

### One command (recommended)

Convert and verify in a single step — auto-derives `inputshape` from the ONNX, checks ncnn↔ONNX
argmax/logit parity, and cleans up pnnx intermediates:

```bash
.venv-train/bin/python scripts/export_to_ncnn.py models/your_model.onnx
```

Useful flags: `--skip-verify`, `--keep-intermediates`, `--inputshape '[1,N],[1]'`, `--outdir DIR`,
`--via {onnx,torchscript,auto}`.
The manual `pnnx` + `verify_ncnn_parity.py` steps are the underlying operations it wraps — see
[../dev/building.md#manual-onnx--ncnn-conversion-internals](../dev/building.md#manual-onnx--ncnn-conversion-internals).

### From TorchScript (skip ONNX)

If you already have a TorchScript policy (`.pt`/`.ptl`), convert it **directly** — one fewer hop,
and often better numerical parity since pnnx's native format *is* TorchScript. A `.pt` carries no
readable shape metadata, so the tool **auto-derives `inputshape`** from a `<model>.shape.json`
sidecar (`{"inputshape": "[1,5]"}` or `{"shape": [1, 5]}`), else best-effort from the first
`Linear` layer:

```bash
.venv-train/bin/python scripts/export_to_ncnn.py models/policy.pt              # auto-shape
.venv-train/bin/python scripts/export_to_ncnn.py models/policy.pt --inputshape '[1,5]'  # override
```

To produce the `.pt` + sidecar from a trained SB3 checkpoint (an ONNX-free alternative):

```bash
.venv-train/bin/python scripts/export_torchscript.py --checkpoint models/rover_checkpoints/<ckpt>.zip
.venv-train/bin/python scripts/export_to_ncnn.py models/policy.pt    # sidecar -> auto-shape
```

`--via` defaults to `auto` (routes by extension: `.onnx` → onnx, `.pt`/`.ptl` → torchscript).
Pass `--via` explicitly to force a path. The ONNX path remains the default fallback.

## Use it in Godot

After conversion, load the model at runtime using `NcnnRunner`:

```gdscript
@export_file("*.param") var model_param_path: String = "res://models/your_model.ncnn.param"
@export_file("*.bin") var model_bin_path: String = "res://models/your_model.ncnn.bin"

var _runner: NcnnRunner

func _ready() -> void:
    _runner = NcnnRunner.new()
    add_child(_runner)
    _runner.input_blob_name = "in0"
    _runner.output_blob_name = "out0"
    var ok = _runner.load_model(
        ProjectSettings.globalize_path(model_param_path),
        ProjectSettings.globalize_path(model_bin_path))
    if not ok:
        push_error("Failed to load ncnn model.")

func get_action(observations: Array[float]) -> PackedFloat32Array:
    if not _runner.is_model_loaded():
        return PackedFloat32Array()
    return _runner.run_inference(PackedFloat32Array(observations))

func get_action_from_image(image: Image) -> PackedFloat32Array:
    if not _runner.is_model_loaded():
        return PackedFloat32Array()
    return _runner.run_inference_image(image, true)  # normalize to [0,1]
```

For the high-level controller (recommended): extend `NcnnAIController2D` or `NcnnAIController3D`
and set `control_mode = NCNN_INFERENCE` — the controller handles `load_model`/`run_inference` and
action decoding automatically. Set `model_param_path` and `model_bin_path` as exports on the
controller node.

Key `NcnnRunner` methods:

- `load_model(param_path, bin_path) -> bool`
- `run_inference(input: PackedFloat32Array) -> PackedFloat32Array`
- `run_inference_image(image: Image, normalize_to_zero_one := true) -> PackedFloat32Array`
- `run_discrete_action(input: PackedFloat32Array) -> int` — argmax over output
- `is_model_loaded() -> bool`
- `input_blob_name`, `output_blob_name` — set to `"in0"` / `"out0"` for godot-rl-exported models
- `input_shape: PackedInt32Array` — optional: reshapes flat floats to 1D/2D/3D ncnn tensor

## INT8 quantization (mobile/edge)

INT8 quantization reduces model size by ~4× and inference time by ~2–4× on mobile and edge devices.
This is a deployment moat: ONNX Runtime / Barracuda lack game-side INT8 support, but ncnn
(statically linked in this project) has INT8 built in — deploying an INT8 model requires **no C++
changes**.

**1. Build the quantize tools once** (not included in the pip `ncnn` wheel):

```bash
./scripts/build_ncnn_tools.sh
```

This builds `ncnn2table`, `ncnn2int8`, and `ncnnoptimize` from the vendored `thirdparty/ncnn/`
source into `thirdparty/ncnn/tools-bin/`. Uses `NCNN_SIMPLEOCV=ON` so OpenCV is not required.

**2. Export to INT8 (one command):**

```bash
.venv-train/bin/python scripts/export_int8.py models/your_model.ncnn.param \
  models/your_model.ncnn.bin --width W --height H --channels C --outdir models
```

This runs the full pipeline: optimize → KL-calibrate → `ncnn2int8` → argmax-agreement parity check.
Produces `your_model_int8.ncnn.{param,bin}` in `--outdir`. Flags: `--samples` (calibration images,
default 256), `--threshold` (argmax agreement rate, default 0.9), `--skip-verify`,
`--keep-intermediates`, `--in-blob`/`--out-blob`, `--tools-dir`.

**3. Deploy:** load the `*_int8.ncnn.{param,bin}` files with `NcnnRunner` exactly as you would an
fp32 model — the runtime INT8 path is compiled in, no changes required:

```gdscript
runner.load_model("res://models/your_model_int8.ncnn.param",
                  "res://models/your_model_int8.ncnn.bin")
```

## VecNormalize obs stats

If you trained with SB3 `VecNormalize`, the policy network expects pre-normalized observations.
Export the running stats to JSON and set the controller's `obs_norm_stats_path` so `ObsNormalize`
replays the mean/std game-side before inference:

```bash
.venv-train/bin/python scripts/export_vecnormalize.py path/to/vec_normalize.pkl
```

This writes a JSON file. Point the controller's `obs_norm_stats_path` export at it. The network
itself does not carry the stats — this step is required for policies trained with `VecNormalize`.

## Platform targets (the moat)

ncnn is statically linked via C++ (no .NET, no external runtime). This enables deployment targets
that godot_rl's ONNX/.NET path and Unity ML-Agents simply cannot reach:

- **Web / WASM** — godot_rl's ONNX Runtime can't run in a browser; ncnn compiles to WASM with
  Emscripten.
- **Console** — no .NET certification issues; the extension is a plain C++ Godot GDExtension.
- **Mobile (iOS/Android)** — static lib, no managed runtime to ship alongside the app.
- **Edge devices** — INT8 quantization game-side, async inference threads, LOD policy switching;
  none of these are replicable by a Python-server or managed-runtime framework.

The deploy workflow is algorithm-agnostic: PPO logits and DQN Q-values both use argmax under
discrete decoding; PPO/TD3 mean and SAC `tanh(mean)` both use the continuous output directly.
Train with any godot_rl-compatible algorithm and deploy the same ncnn model unchanged.
