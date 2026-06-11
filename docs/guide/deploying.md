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
    # Load from bytes (works in an exported .pck and on web, where there is no real file to fopen).
    var param_bytes := FileAccess.get_file_as_bytes(model_param_path)
    var bin_bytes := FileAccess.get_file_as_bytes(model_bin_path)
    var ok = _runner.load_model_from_buffers(param_bytes, bin_bytes)
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

- `load_model_from_buffers(param: PackedByteArray, bin: PackedByteArray) -> bool` — load from
  bytes (read with `FileAccess.get_file_as_bytes`); **use this for exported games / web**
- `load_model(param_path, bin_path) -> bool` — load from filesystem paths (editor / desktop tools)
- `run_inference(input: PackedFloat32Array) -> PackedFloat32Array`
- `run_inference_image(image: Image, normalize_to_zero_one := true) -> PackedFloat32Array`
- `run_discrete_action(input: PackedFloat32Array) -> int` — argmax over output
- `run_inference_multi(inputs, output_names) -> Dictionary` — multi-input/multi-output pass (used
  for recurrent state-carry; see "Recurrent (LSTM) policies" below)
- `is_model_loaded() -> bool`
- `input_blob_name`, `output_blob_name` — set to `"in0"` / `"out0"` for godot-rl-exported models
- `input_shape: PackedInt32Array` — optional: reshapes flat floats to 1D/2D/3D ncnn tensor

### Level-of-Detail policy switching (`NcnnLODRunner`)

For many agents (or an expensive policy), `NcnnLODRunner` runs a cheap **reflex** net most frames and
an accurate **deliberative** net only every `deliberative_interval` frames (or when you pass
`state_changed = true`) — exactly one inference per frame, so the big net's cost is paid at
~1/interval the rate. Set the two model paths (`reflex_*` / `deliberative_*`, which must share the obs
and output contract) and call `decide(obs)` each frame:

```gdscript
var out := $NcnnLODRunner.decide(obs)   # { logits, tier: "reflex"|"deliberative", ran_deliberative }
```

`deliberative_interval` is live-editable; call `reset()` on episode boundaries to re-arm the cadence.
Only viable because both nets are statically linked and resident — switching them is free at runtime.

## Exporting your game

**Enable the addon** (Project → Project Settings → Plugins → *Godot Native RL*). Beyond surfacing a
clear error if the native binary is missing for your platform, enabling it registers an export hook
that **auto-packs your `.ncnn.param` / `.ncnn.bin` files into every export**. These are raw data
files Godot's exporter skips by default (they're referenced by string path, not as resources), so
without this an exported game crashes at runtime with `cannot read model files …` — on *every*
platform. With the addon enabled there is nothing else to configure.

> Not enabling the plugin? Then add the model files to your export preset by hand: Project →
> Export → *Resources* → "Filters to export non-resource files/folders" →
> `*.ncnn.param, *.ncnn.bin`.

### Web (WASM)

The native ncnn extension runs in the browser — something godot_rl's ONNX/.NET path cannot do. In
the **Web** export preset set:

- **Extension Support: ON** — required for any GDExtension on web (selects the `dlink` template).
- **Thread Support: OFF** — pairs with the single-threaded `…wasm32.nothreads` binary the addon
  ships. This means **no `SharedArrayBuffer`, so no COOP/COEP headers** — the game deploys to
  itch.io / GitHub Pages / any static host with zero server configuration.

Serve the exported folder with any static file server and open `index.html`. (Build the web binary
from source via `scripts/cross/build_web.sh` — see `docs/dev/building.md`.)

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

## Continuous action sampling (DiagGaussian std sidecar)

A PPO policy with a continuous (Box) action space stores its std as a state-independent learned
parameter (`policy.log_std`) that is never part of the network output — so an ncnn-converted policy
emits only the action mean. To sample (explore at deploy / vary behavior) instead of always taking
the mean, export the std to a sidecar and replay the DiagGaussian draw game-side:

```bash
.venv-train/bin/python scripts/export_action_dist.py models/policy.zip   # -> models/policy_action_dist.json
```

Point the controller's `action_dist_stats_path` export at the JSON and set
`deterministic_inference = false`. The deploy side then samples `mean + std·N(0,1)` (seeded by
`inference_seed` for reproducible eval), applying the optional per-key `tanh` squash after the draw.
With `deterministic_inference = true` (default) the std is ignored and the mean is used unchanged.
This **exceeds godot_rl**, whose export drops the std entirely. (PPO/A2C continuous only; SAC's
state-dependent std is out of scope — its actor already exports `tanh(mean)`.)

## Recurrent (LSTM) policies (`recurrent_stats_path`)

Recurrent LSTM policies deploy natively: `NcnnControllerCore` carries the network's hidden state
across frames via the C++ `NcnnRunner.run_inference_multi` multi-IO path. Point the controller's
`recurrent_stats_path` export at a `<model>.recurrent.json` sidecar that declares which blobs carry
state:

```json
{
  "obs_input": "in0",
  "obs_shape": [5],
  "action_output": "out0",
  "state_pairs": [
    { "in": "in1", "out": "out1", "shape": [8] },
    { "in": "in2", "out": "out2", "shape": [8] }
  ]
}
```

State zero-inits on load and re-zeroes on `reset()` / `reset_recurrent_state()` (so memory never
bleeds across episodes); the action output decodes through the normal action path. Float-obs path
only.

> Deploy plumbing — the synthetic-LSTM fixture proves the round-trip; real `RecurrentPPO` training +
> general export tooling are follow-ups. The C++ multi-IO method changed the extension ABI, so
> **rebuild `NcnnRunner` after pulling** (see [../dev/building.md](../dev/building.md)). Full
> contract: [The recurrent deploy contract](../dev/DEVELOPMENT.md#the-recurrent-deploy-contract-lstm).

## Driving animation from policy actions (`AnimationPolicyAdapter`)

For continuous-control agents, `AnimationPolicyAdapter` writes a policy's action vector straight to
an `AnimationTree`'s blend parameters — production animation with no hand-written blending layer.
Each mapping routes one action element to one blend-param path with an affine remap + clamp
(`scale·a + offset`, clamped) so a `tanh`/`[-1,1]` or raw action adapts to whatever range the
parameter expects:

```gdscript
adapter.add_mapping(0, "parameters/locomotion/blend_amount", 0.5, 0.5, 0.0, 1.0)  # [-1,1] -> [0,1]
adapter.add_mapping(1, "parameters/lean/blend_position")
adapter.apply(action)   # call each frame after inference
```

Out-of-range / unmapped action elements are skipped; a freed or unset tree is a safe no-op (the
"no tree" error logs once, not every frame).

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
