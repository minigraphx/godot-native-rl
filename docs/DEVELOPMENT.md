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
  - `sensors/` — `RaycastSensor2D/3D` + pure `raycast_math`.
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

## Known robustness gaps (see docs/BACKLOG.md)

- **No socket timeout** — `NcnnSync.connect_to_server()` / `_get_dict_json_message()` poll in
  unbounded `while` loops, so a silent/dead socket blocks forever. This is the root cause behind both
  the "launch a training scene headless without a trainer → hang" and the macOS-sleep hang. Fix is
  folded into **backlog item 9** (protocol v0.8: connect/read timeout).

## Where things live

| Need | Path |
|------|------|
| Commands, venvs, headless gotchas (always-loaded) | `CLAUDE.md` |
| User-facing setup / usage / examples | `README.md` |
| Design rationale (per feature) | `docs/superpowers/specs/` |
| Implementation plans (per feature) | `docs/superpowers/plans/` |
| Actionable backlog (pick up by number) | `docs/BACKLOG.md` |
| ncnn vs ONNX Runtime decision guide | `docs/ncnn_vs_onnx.md` |
