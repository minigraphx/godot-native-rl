# SampleFactory Backend — Design

**Date:** 2026-06-05
**Status:** Implemented — with one pivot from this design (see note below).

> **Implementation note (post-build):** §3–§5 below specify an **ONNX** intermediate. During
> implementation we found `.venv-sf` cannot `torch.onnx.export` (broken onnx/ml_dtypes; onnxscript
> needs numpy≥2, colliding with SampleFactory's gymnasium<1.0 / numpy<2 pin). Since the
> checkpoint→intermediate step must run in `.venv-sf`, the shipped exporter emits **TorchScript**
> (`scripts/export_sf_to_torchscript.py` → `.pt` + `.pt.shape.json` sidecar → `export_to_ncnn.py`
> torchscript path). The deploy contract (raw per-segment action logits) is unchanged. Also: SF 2.1.1
> has no MultiDiscrete support (action space is `Tuple([Discrete…])`), and `train_sf.py` calls
> `run_rl` with a module-level picklable env factory rather than `sample_factory_training()` to work
> around godot_rl-0.8.2/SF-2.1.1 incompatibilities. See the commit history on `feat/sample-factory-backend`.
**Backlog item:** 18 (SampleFactory backend — async high-throughput training backend)
**GitHub issue:** #24 (the closing PR should `Closes #24` and tick `docs/BACKLOG.md` item 18 in the same change)

## 1. Purpose

Add a **third** training backend alongside Stable-Baselines3 (`scripts/train_chase.py`) and the
CleanRL single-file PPO (`scripts/train_cleanrl.py`): a **SampleFactory** (async PPO / APPO) trainer
that trains the existing **Chase The Target** example over the godot_rl bridge, then **exports the
trained actor** so it flows unchanged into `scripts/export_to_ncnn.py` → native ncnn deploy.

This proves the framework is trainer-agnostic with a *third, architecturally different* algorithm
(async actor-critic with separate rollout/learner workers) and closes the train → convert → deploy
loop for it.

**Explicit non-goal:** throughput. SampleFactory's whole reason for existing is samples/sec via its
async architecture, but measuring/showcasing that (against the parallel arena) is a separate
follow-up. This phase optimizes for **correctness and a green smoke test**, so SF's parallelism is
deliberately dialed *down* (single-agent chase, serial/sync mode) to keep the loop deterministic and
macOS-robust.

Mirror `scripts/train_cleanrl.py` / `scripts/train_cleanrl.sh` in structure and CLI ergonomics;
reuse the chase training scene (`examples/chase_the_target/chase_the_target_train.tscn`).

## 2. Context — researched facts that drive the design

Researched against the installed `godot_rl==0.8.2` and a `pip install --dry-run sample_factory`:

- **godot_rl ships a SampleFactory wrapper** (`godot_rl/wrappers/sample_factory_wrapper.py`):
  `SampleFactoryEnvWrapperBatched` / `...NonBatched` (both subclass `GodotEnv`), plus
  `make_godot_env_func`, `register_gdrl_env`, `parse_gdrl_args`, and the entry points
  `sample_factory_training(args, extras)` / `sample_factory_enjoy(args, extras)`. This is the
  supported path; we build on it rather than re-implementing the env glue.
- **No built-in SF→ONNX export.** godot_rl only ships `wrappers/onnx/stable_baselines_export.py`
  (SB3 only). So, like the CleanRL backend, we **hand-roll** the actor → ONNX export.
- **`sample_factory==2.1.1` resolves on Python 3.13 but downgrades `gymnasium 1.0.0 → 0.29.1`**
  (SF pins `gymnasium<1.0`). It does **not** pin torch (torch 2.12 stays). It pulls SF's async
  machinery: `faster-fifo`, `signal-slot-mp`, `wandb`, `tensorboardX`, `pyglet`, `psutil`.
  → Installing into `.venv-train` would silently change the gym version that the SB3 + CleanRL
    backends (and all green tests) are validated under. **Decision: isolate SF in a separate venv.**
- **godot_rl's SF defaults** (`gdrl_override_defaults`): `use_rnn=False`, `normalize_input=True`,
  `normalize_returns=True`, `async_rl=True`, `serial_mode=False`, `num_workers=1`,
  `num_envs_per_worker=2`, `worker_num_splits=2`, `env_agents=16`, `nonlinearity=relu`,
  `batched_sampling=False`. We override the ones that hurt correctness/parity/macOS (see §4).
- **In-editor port offset:** `make_godot_env_func` computes `port = cfg.base_port; if env_config:
  port += 1 + env_config.env_id`. With one serial worker (`env_id=0`) the Godot client must connect
  on `base_port + 1`. The orchestrator launches Godot on the matching port. Nailing the exact
  worker/port/`env_agents` wiring is the main implementation unknown.

## 3. The deploy contract (why the ONNX output must be raw action logits)

The native deploy path decodes a discrete action via `ActionDecode.decode_actions`
(`addons/godot_native_rl/controllers/action_decode.gd`): it slices the policy output into one
segment per action key and takes **argmax over the `size` logits** per discrete key. Chase is
`{"move": {"size": 5, "action_type": "discrete"}}`, so the exported ONNX `out0` must be a length-5
**logit vector** — exactly as the SB3 and CleanRL exporters produce.

Therefore the SF exporter must wrap the trained `ActorCritic` so its forward returns the **raw
action logits** (pre-sampling categorical logits, length `sum(nvec)`) with input name `obs` and
output name `output` (+ vestigial `state_ins`/`state_outs`), matching godot_rl's
`export_model_as_onnx` naming so `export_to_ncnn.py`'s `derive_inputshape` (keys on `obs`, appends
`,[1]` for `state_ins`) and the `in0`/`out0` parity check work unchanged.

## 4. Architecture

Three new scripts + one new requirements file + one new venv, mirroring the CleanRL backend layout.
Pure, import-light helpers live at module top (unit-tested with stdlib `unittest`); all heavy
imports (`torch`, `sample_factory`, `godot_rl`, `numpy`) live **inside `main()`** (hard repo
convention).

### 4.1 `requirements-sf.txt`
```
godot-rl-agents==0.8.2     # for the SF wrapper + GodotEnv glue
sample_factory==2.1.1      # pulls gymnasium 0.29.1, faster-fifo, signal-slot-mp, ...
```
(Exact pin set finalized during implementation from a real install; keep it minimal.)

### 4.2 `.venv-sf` (via `setup_training.sh`)
Add a third idempotent `create_venv` call:
```
create_venv "$PYTHON_SF" ".venv-sf" "$REQ_SF"     # PYTHON_SF default python3.13
```
`.venv-train` (3.13, godot-rl+SB3, gymnasium 1.0.0) and `.venv` (3.14, pnnx+torch) are untouched.
`--check` mode lists the third venv too. Document that `.venv-sf` is heavier and only needed for the
SF backend.

### 4.3 `scripts/train_sf.py`
```
── pure helpers (module top, no heavy imports) ──
SFConfig (NamedTuple)            timesteps/seed/ports/paths/scene-agent-count (immutable)
parse_args(argv) -> SFConfig     argparse → frozen config
build_sf_argv(cfg) -> list[str]  translate SFConfig → the argv list godot_rl's parse_gdrl_args wants
                                 (serial/sync/normalize-off overrides live here, testable as strings)
client_port(base_port) -> int    base_port + 1 (the in-editor single-worker offset)

── main() (lazy heavy imports) ──
  builds the args namespace godot_rl's sample_factory_training expects, calls it (env_path=None →
  in-editor: SF opens server, waits for the launched Godot client), trains for cfg.timesteps,
  SF writes a checkpoint under train_dir/experiment/.
```
macOS-safe / parity-safe overrides baked into `build_sf_argv`: `--serial_mode=True`,
`--async_rl=False`, `--num_workers=1`, `--num_envs_per_worker=1`, `--normalize_input=False`,
`--normalize_returns=False`, `--use_rnn=False`, single `--env_agents` matching the chase scene.

### 4.4 `scripts/export_sf_to_onnx.py`
```
── pure helpers ──
actor_logit_layout(action_space) -> (total_logits, nvec)   reuse the discrete_action_dims pattern

── main() (lazy heavy imports) ──
  - load the SF checkpoint (latest under train_dir/experiment/checkpoint_p0/)
  - rebuild the ActorCritic via SF's model factory from the saved cfg
  - load_state_dict from the checkpoint
  - wrap: obs → encoder → core (identity, no RNN) → action_parameterization → raw action logits
  - export to ONNX with input_names=["obs","state_ins"], output_names=["output","state_outs"],
    dynamic batch axis (identical naming/axes to scripts/train_cleanrl.py::export_actor_as_onnx)
```
Output = raw logits (length `sum(nvec)`); deploy ActionDecode argmaxes per segment. Pinned to the
SF 2.1.1 model API.

### 4.5 `scripts/train_sf.sh`
Orchestrator mirroring `train_cleanrl.sh`:
1. start `train_sf.py` (in `.venv-sf`) — opens the SF server on `base_port + 1`, blocks for client;
2. `sleep`, then launch headless Godot `chase_the_target_train.tscn` on the matching port;
3. `wait` the trainer; kill Godot;
4. run `export_sf_to_onnx.py` (in `.venv-sf`) → ONNX;
5. run `export_to_ncnn.py <onnx>` (in `.venv`) → `.ncnn.{param,bin}` + parity.

Env overrides: `GODOT`, `PY_SF` (`.venv-sf/bin/python`), `PY_CONVERT` (`.venv/bin/python`),
`TIMESTEPS`, `SPEEDUP`, `OUTDIR` (default `models/`; the smoke passes a temp dir).

## 5. Data flow

```
chase_the_target_train.tscn  ⇄  train_sf.py / sample_factory_training  (.venv-sf)
   (Godot client, port=base+1)        (SF server, serial/sync, 1 agent)
                                          │ SF checkpoint (.pth) under train_dir/experiment/
                                          ▼
                            export_sf_to_onnx.py  (.venv-sf)
                                          │ ONNX  (obs → output, raw logits)
                                          ▼
                            export_to_ncnn.py  (.venv, pnnx)
                                          │ .ncnn.{param,bin}
                                          ▼
                            verify_ncnn_parity.py  (.venv, torch/onnxruntime vs ncnn, atol 1e-2)
```

## 6. Testing

### 6.1 Pure unit tests (always run, no SF dep)
Stdlib `unittest` under `test/python/test_train_sf.py` for the import-light helpers:
`parse_args`, `build_sf_argv` (asserts the serial/sync/normalize-off override strings are present),
`client_port`, `actor_logit_layout`. Discovered by the existing `run_tests.sh` Python-helper step;
no `sample_factory` import required.

### 6.2 End-to-end smoke in `run_tests.sh`
A new step, **guarded by `[ -x .venv-sf/bin/python ]`**:
- if absent → print `SKIP: .venv-sf not present (run scripts/setup_training.sh to enable the SF smoke)` and continue;
- if present → run a **tiny-timestep** `train_sf.sh` into a **temp dir** (`OUTDIR=$(mktemp -d)`,
  never touches `models/`), then assert the `.ncnn.{param,bin}` exist and `verify_ncnn_parity.py`
  passes (argmax/atol). Mirrors the INT8 and hide-seek smoke wiring already in the suite.

Tiny budget tuned during implementation so the step is short but actually produces a usable
checkpoint; serial/sync mode keeps it deterministic and macOS-robust.

## 7. Docs (same-change, per repo convention)
- `README.md` — add the SF backend to the training-backends list.
- `CLAUDE.md` — add the `train_sf.sh` key command; update the "Two venvs" gotcha to **three**
  (`.venv-sf` for SampleFactory); note item 18 done.
- `docs/dev/DEVELOPMENT.md` — backend/data-flow note if warranted.
- `docs/godot-rl-gap-analysis-2026-06-02.md` — flip the SampleFactory-backend row.
- `docs/BACKLOG.md` — tick item 18.
- GitHub — `Closes #24`.

## 8. Key risks & mitigations
- **macOS multiprocessing flakiness** (SF's `faster-fifo`/`signal-slot-mp` async workers) →
  `serial_mode=True`, `async_rl=False`, `num_workers=1` for the smoke.
- **In-editor port offset** (`base_port + 1 + env_id`) → orchestrator launches Godot on
  `client_port(base_port)`; this is the main wiring unknown, resolved during TDD.
- **`normalize_input`/`normalize_returns` must be False** or the exported actor carries a
  `RunningMeanStd` that breaks ncnn parity (game-side obs-norm replay is item 24/47, out of scope).
- **SF 2.1.1 model/checkpoint API** — the exporter is pinned to that version; a future SF bump is a
  separate maintenance task.
- **Third heavy venv** — documented as opt-in; the smoke auto-skips without it, so the merge gate
  stays green on machines that haven't created `.venv-sf`.

## 9. Out of scope (explicit)
- Throughput measurement / parallel-arena SF showcase (separate follow-up).
- Game-side replay of SF's input normalization (items 24/47).
- SF recurrent (LSTM) policies — `use_rnn=False` here; recurrent deploy is item 22.
- `sample_factory_enjoy` / SF-native evaluation — we deploy via ncnn, not SF's enjoy.
