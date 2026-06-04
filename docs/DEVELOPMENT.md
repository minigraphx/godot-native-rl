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

## The deploy contract (algorithm-agnostic)

**PPO is the only algorithm we've *proven*, not one we depend on.** The deploy path is a stable,
narrow contract that is independent of the RL algorithm that trained the weights — this is what lets
the project grow (SAC, DQN, TD3, A2C, …) without churning the runtime. The contract:

```
obs vector ──(optional ObsNormalize)──► policy network (ncnn) ──► output vector ──(ActionDecode)──► action dict
```

- **The runtime is a pure forward pass.** `NcnnRunner` (C++) loads `.param`/`.bin` and runs
  `obs → output`; it has no notion of which algorithm trained the net.
- **Decode keys off output *shape* + `action_type` only, never the algorithm.** `ActionDecode`
  (`controllers/action_decode.gd`) slices the output per action_space key and:
  - `discrete` → **argmax** over the segment. PPO/A2C pre-softmax **logits**, DQN action-value
    **Q-values**, and any other discrete head all argmax identically — same code path, same action.
  - `continuous` → the segment as-is, optionally `tanh`-squashed (`"squash": true`). PPO/A2C and
    TD3/DDPG deterministic actors pass the **mean** through; SAC's squashed-Gaussian deploys as
    `tanh(mean)` via the `squash` flag. All share the one continuous path.
- **Obs normalization is a VecEnv wrapper, not an algorithm feature.** `ObsNormalize` replays SB3
  `VecNormalize` running mean/var (it lives in a side `.pkl`, not the network), used by PPO/A2C/SAC/…
  alike — so it's algorithm-independent too.
- **Training is backend-agnostic via the godot_rl wire protocol.** Anything that speaks it works:
  SB3 (PPO + A2C/DQN/SAC/TD3), the shipped CleanRL backend (`scripts/train_cleanrl.py`), and later
  SampleFactory/SKRL/RLlib. The only PPO/godot_rl-flavored vestige is the `state_ins` wire input,
  which pnnx **prunes at conversion** → inert at deploy.

**Guarded by** `test/unit/test_algorithm_agnostic_decode.gd` (DQN Q-values / SAC / TD3 / hybrid heads
all decode through the same path). That's a decode/runtime guard needing no training run; the full
trained non-PPO regression (SB3 SAC/DQN end-to-end → ncnn → behavioral check) is the separate
`needs-training-run` slice of issue #45.

**Recurrent (LSTM) policies** carry hidden state across frames — the runtime-feature extension
of the contract, not a violation; see the next subsection. Feed-forward policies of every algorithm
above already satisfy the base contract as-is.

## The recurrent deploy contract (LSTM)

Recurrent policies extend the base contract with **carried hidden state**: the network has extra
input/output blobs that thread state from one frame to the next. This is **deploy plumbing only** —
the training/export side (real `RecurrentPPO` from sb3-contrib + tooling that emits the sidecar from
an arbitrary trained model) is **deferred** (issue #33); what's shipped is the runtime path plus a
hand-built synthetic fixture.

**Generic multi-IO runner (C++).** `NcnnRunner.run_inference_multi(inputs, output_names)` runs a net
with any number of inputs and outputs (the single-IO `run_inference`/`run_inference_image` are
unchanged and now share a `build_mat_from_shape` helper).
- `inputs`: `Array` of `{ "name": String, "data": PackedFloat32Array, "shape": PackedInt32Array }`.
- `output_names`: `PackedStringArray` of blobs to extract.
- Returns `{ blob_name: PackedFloat32Array }`; **empty `Dictionary` on any error** (bad shape,
  missing blob, extract failure) — fail loud, no partial results.

**Sidecar (`<model>.recurrent.json`).** Declares the contract so nothing is hardcoded; parsed +
validated at load by pure `controllers/recurrent_state.gd` (`RecurrentState`, the recurrent analogue
of `obs_normalize.gd`). Schema:

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

Each `state_pairs` entry maps a state **input** blob to the **output** blob that produces its next
value, with the blob's `shape` (an LSTM contributes two pairs — cell + hidden; a GRU would contribute
one — the sidecar format is the same, though only LSTM is currently verified).

**State lifecycle.** `NcnnControllerCore` holds the carried state in `recurrent_state`:
1. **Zero-init on load** — `recurrent_stats_path` is read in the controllers' `_ready()`
   (`NCNN_INFERENCE` mode, mirroring `obs_norm_stats_path`); a valid contract zero-fills each
   `pair.in` to `product(shape)` floats.
2. **Each frame** — `choose_and_apply_action` takes the recurrent branch: it feeds `obs` + the
   carried `*_in` blobs into `run_inference_multi`, decodes the `action_output` blob via the same
   `ActionDecode` path, and **stores the returned `*_out` blobs** as the next frame's `*_in`. On any
   failure it push_errors and skips `set_action` **without advancing state**.
3. **Re-zero on episode boundary** — `reset()` (and the public `reset_recurrent_state()` on the
   controllers, for games that manage their own episode boundaries) re-zeroes the state so memory
   never bleeds across episodes.

**Scope.** Float-obs path only — image-obs + recurrent is **out of scope** (the recurrent branch
does not run `run_inference_image`). Batched multi-agent recurrent inference is separate (issue #34).

**Conversion + fixture.** pnnx was confirmed to **preserve the LSTM's 3-in/3-out state blobs**
through ONNX→ncnn conversion. The synthetic fixture (`scripts/make_synthetic_lstm.py` →
`models/synthetic_lstm.ncnn.{param,bin}` + `models/synthetic_lstm.recurrent.json` + `models/synthetic_lstm_golden.json`) verifies the full
path: end-to-end per-step argmax + logit parity (`atol 1e-2`) and reset-reproduction. If a future
model's conversion ever prunes those blobs, the fallback is to **hand-author the `.param`** state
blobs (and the sidecar names them either way).

**Rebuild required.** `run_inference_multi` changed the C++ ABI and `bin/` is gitignored, so a fresh
clone (or anyone pulling this branch) **must rebuild** the extension —
`scons platform=... target=template_debug` **and** `target=template_release` — or `NcnnRunner` won't
expose the new method.

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
