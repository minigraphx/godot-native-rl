# Developer Notes

Contributor-facing reference for working **inside** this repo: architecture, data flow, and the
longer-form "why" behind decisions. This is the home for deep-dives that don't belong in the
always-loaded `CLAUDE.md` (which keeps the terse triggers + commands) or the user-facing `README.md`.

> New here? Read `CLAUDE.md` first (commands, venvs, headless gotchas), then this for the bigger
> picture. Design rationale lives in `docs/superpowers/specs/`; step-by-step plans in
> `docs/superpowers/plans/`; the actionable backlog in `docs/BACKLOG.md`.

## Architecture at a glance

Two layers, deliberately separated:

- **C++ GDExtension (inference)** — `src/ncnn_runner.{h,cpp}` exposes `NcnnRunner` (load ncnn
  `.param`/`.bin`, run a forward pass, `run_discrete_action` = argmax). Statically links ncnn from
  `thirdparty/ncnn`. Manifest `ncnn_runner.gdextension`, binaries in `bin/`. This is the moat: native
  inference with no .NET/runtime, deployable to web/console/mobile.
- **GDScript library (training bridge + authoring)** — `addons/godot_native_rl/`:
  - `sync.gd` (`NcnnSync`) — the godot_rl wire-protocol bridge (TCP client, port 11008).
  - `controllers/` — `NcnnControllerCore` (RefCounted episode/reward state machine) + thin
    `NcnnAIController2D`/`3D` node wrappers that forward to it.
  - `reward/` — declarative reward authoring (`RewardBuilder` + `RewardAdapter` + terms).
  - `sensors/` — `RaycastSensor2D/3D` + pure `raycast_math` (incl. an opt-in `class_sensor` mode:
    per-ray multi-hot collision-layer segments via `detection_classes` + optional `other`/closeness
    slots, encoded by `raycast_math.encode_ray_class`).
  - `training/` — `ParallelArena` (tiles N agent worlds in one process for ~Nx-faster training).

Examples (`examples/chase_the_target/`, `examples/rover_3d/`) compose these into runnable scenes.

## Training data flow (godot_rl protocol)

```
Python trainer (SB3 PPO, .venv-train)            Godot (headless client)
  opens TCP server on :11008  ──── connect ────►  NcnnSync.connect_to_server()
  handshake / env_info        ◄──── n_agents ───  NcnnSync collects the "AGENT" group
  loop:  action  ─────────────────────────────►   set_action() on each agent
         step (obs/reward/done) ◄───────────────   get_obs()/get_reward()/get_done()
```

- Agents self-register via `add_to_group("AGENT")` in the controller's `_ready`. `NcnnSync` derives
  obs/action space from `agents_training[0]` (homogeneous agents assumed).
- `control_mode` resolution: an agent left at `INHERIT_FROM_SYNC` adopts the Sync node's mode
  (`TRAINING=1` / `NCNN_INFERENCE=2` / `HUMAN=0`). This is why `ParallelArena`'s replicated agents
  become training agents automatically under a TRAINING `Sync`.
- **Parallelism is scene-only:** `ParallelArena` spawns N copies of a "world" sub-scene tiled on a
  square XZ grid (`spacing` must exceed an agent world's reach: arena extent + ray length). One
  `Sync` collects all N agents → the trainer vectorizes over `n_agents = N`. The Python side is
  unchanged. Measured ~6.2× at 8 agents (sub-linear vs 8× from fixed startup/handshake overhead).

## Convert + deploy flow

`train_rover.py` (SB3) → ONNX (`export_model_as_onnx`) → `scripts/export_to_ncnn.py` (pnnx in an
isolated temp dir + parity verify) → `models/*.ncnn.{param,bin}` → loaded by `NcnnRunner` at deploy.
Parity is checked at `atol=1e-2` (torch-dynamo vs ncnn InnerProduct drift; argmax is stable). godot_rl
policies convert to blob names `in0`/`out0` (pnnx prunes the vestigial `state_ins` input).

## The inference-backend boundary (swappable runtime)

The deploy runtime is a **pluggable seam, not hardcoded**. `NcnnControllerCore.choose_and_apply_action(agent,
runner)` takes the `runner` as an injected, duck-typed dependency and only ever calls a tiny inference
surface — it has no idea it's talking to ncnn. `NcnnRunner` is the only implementation today, but a second
backend (e.g. an ExecuTorch `.pte` runner — see **issue #54**) drops in by implementing the same surface,
with **no changes to the controller, `ActionDecode`, `ObsNormalize`, the sensors, or the wire protocol**.

The inference surface the controller depends on (backend-**neutral**):

- `is_model_loaded() -> bool`
- `run_inference(PackedFloat32Array obs) -> PackedFloat32Array` — float-vector deploy
- `run_inference_image(Image, normalize: bool) -> PackedFloat32Array` — image deploy

Everything downstream touches only the **output float vector** (sliced + argmax/mean-decoded by
`ActionDecode`, identically regardless of the training algorithm) — never the runtime that produced it. So
"what runs the forward pass" is fully decoupled from "what the agent does with the result."

**Neutral vs backend-specific.** The inference surface above is neutral; **model loading is the one
backend-specific spot**. `NcnnRunner.load_model(param_path, bin_path)` takes ncnn's *two* files (plus
`set_input_blob_name`/`set_output_blob_name`/`set_input_shape` config); a `.pte` runner would load a *single*
program file. So a swap means a new C++ GDExtension implementing this surface **plus** a new `PyTorch ->
<format>` export path (alongside the existing ONNX / TorchScript / state_dict → ncnn paths) — and nothing in
the GDScript deploy logic.

**Why we don't ship a second backend yet.** Two native runtimes = a doubled build/CI/binary-shipping matrix
across every platform, for no current payoff (ncnn still wins the decisive web/WASM axis — see #54). The seam
keeps the decision **reversible at low cost**: add a runner when a trigger fires, don't pre-build the
abstraction. This section *is* the contract a future backend implements.

## Known robustness gaps (see docs/BACKLOG.md)

- **No socket timeout** — `NcnnSync.connect_to_server()` / `_get_dict_json_message()` poll in
  unbounded `while` loops, so a silent/dead socket blocks forever. This is the root cause behind both
  the "launch a training scene headless without a trainer → hang" and the macOS-sleep hang. Fix is
  folded into **backlog item 9** (protocol v0.8: connect/read timeout).

## INT8 quantization pipeline

INT8 export converts a float32 ncnn model to an INT8-quantized version using KL-divergence
calibration. The pipeline has three stages, all orchestrated by `scripts/export_int8.py`:

### Stage 1 — optimize

`ncnnoptimize` (from `thirdparty/ncnn/tools-bin/`) fuses and simplifies the fp32 model (e.g. folds
BatchNorm into Conv weights) before quantization. This step produces a clean `*_opt.ncnn.{param,bin}`
that `ncnn2table` and `ncnn2int8` operate on.

### Stage 2 — KL-calibrate (`ncnn2table`)

Calibration measures the activation range of each quantizable blob across a sample set, then
computes a per-blob INT8 quantization scale using KL-divergence minimization over a 2048-bin
histogram.

**Calibration tensor format:** tensors are CHW float32 `.npy` files, normalized `/255` — the same
layout and scale that `NcnnRunner.run_inference_image` produces at deploy time. The `ncnn2table`
`shape=` argument takes WHC order and reverses it internally; `type=1` selects the `.npy` path (no
OpenCV dependency).

`scripts/int8_calibration.py` generates these tensors from random (or supplied) pixel data. For
real policies, generate calibration tensors from **captured game frames** that are representative of
the actual observation distribution — out-of-distribution calibration images will give inaccurate
scale estimates and degrade accuracy.

**Why sample count matters:** KL calibration builds a 2048-bin activation histogram per blob. For
the synthetic fixture (8×8×3 input), each sample contributes only ~192 activation values per
quantizable blob, so 256 samples → roughly 24–32 values per bin — sparse but sufficient for this
tiny model. Larger real inputs (e.g. 84×84×3) contribute many more values per sample; the default
256 samples is usually adequate. If you see poor INT8 parity on a real policy, try `--samples 1024`
or more.

### Stage 3 — ncnn2int8 + parity verify

`ncnn2int8` applies the per-blob scales from the calibration table to produce the INT8 model
(`*_int8.ncnn.{param,bin}`). `scripts/verify_int8_parity.py` then runs both the INT8 and fp32
ncnn models over random inputs and measures **argmax agreement** (not logit closeness).

**Why argmax agreement, not logit closeness:** quantization intentionally shifts activations to fit
INT8 precision — logits will drift by design (sometimes several percent). For RL control what
matters is that the agent picks the same action, so the metric is the fraction of random inputs on
which both models return the same argmax. The default threshold is 0.9 (90% agreement); models with
≥2 distinct outputs (i.e. not degenerate all-same-action) must also be verified.

### No runner changes needed

The static `libncnn.a` is built with `NCNN_INT8=ON`, so `NcnnRunner` already handles INT8 models
transparently — quantized and fp32 models are loaded and called identically from GDScript. There
were no C++ changes in backlog item 13.

### Deploy

Load `*_int8.ncnn.{param,bin}` with `NcnnRunner` exactly like fp32. The committed
`models/synthetic_cnn_int8.ncnn.*` fixture (1.74× smaller than fp32 for this tiny 8×8×3 model;
real CNN policies see larger gains) is verified by `test/unit/test_int8_deploy.gd`.

## Where things live

| Need | Path |
|------|------|
| Commands, venvs, headless gotchas (always-loaded) | `CLAUDE.md` |
| User-facing setup / usage / examples | `README.md` |
| Design rationale (per feature) | `docs/superpowers/specs/` |
| Implementation plans (per feature) | `docs/superpowers/plans/` |
| Actionable backlog (pick up by number) | `docs/BACKLOG.md` |
| ncnn vs ONNX Runtime decision guide | `docs/ncnn_vs_onnx.md` |
