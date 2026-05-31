# Godot Native RL — Project Memory

## What this is

A GDExtension-based reinforcement-learning framework for Godot 4.6+ that uses Tencent's **ncnn**
for native inference (statically linked C++, **no C#/.NET, no external runtime**). It speaks the
`godot_rl_agents` wire protocol for training, so you train with the stock `godot-rl` Python package
and deploy with native ncnn — on mobile, web, console, desktop, and edge.

**Positioning (north star):** focused superiority first (be clearly better at deployment), then
godot_rl feature parity, then Unity ML-Agents parity (long-term stretch). Strategy: start as a
complement to godot_rl, grow toward full replacement.

## Current state (working, on `main`)

- Full **train → convert → deploy loop** works end-to-end and is in CI-style headless tests.
- The reusable library lives under **`addons/godot_native_rl/`** (item 5): `sync.gd` (`NcnnSync`,
  the bridge), `controllers/` (`NcnnControllerCore` RefCounted core + thin `NcnnAIController2D`/
  `NcnnAIController3D`), `reward/` (`RewardBuilder`/`RewardAdapter`/terms), `sensors/`
  (`RaycastSensor2D`/`RaycastSensor3D` + pure `raycast_math`), `plugin.cfg`. The C++ GDExtension
  stays at the repo root: `src/ncnn_runner.{h,cpp}` (`NcnnRunner`), `ncnn_runner.gdextension`, `bin/`.
- Examples: `examples/chase_the_target/` (2D, ships a pre-trained ncnn model) and
  `examples/rover_3d/` (3D tank-steered raycast obstacle-avoidance rover; scaffold + headless tests
  done, trained model + golden regression pending).
- Wire protocol is **fully godot_rl v0.8.2-compatible** (proven by real SB3 PPO training).

## Key commands

- **Build the extension:** `scons platform=macos arch=arm64 target=template_debug` (see README for other platforms). `godot` binary: `/opt/homebrew/bin/godot` (4.6.2).
- **Run all tests:** `./test/run_tests.sh` — headless GDScript unit tests + Python protocol test +
  inference smoke + trained-chase + golden regression + rover-3D smoke + Python helper tests. Must be
  green before merge. (The full suite should pass from a **clean cache** — `rm .godot/global_script_class_cache.cfg` first to be sure.)
- **Train (chase):** `TIMESTEPS=120000 ./scripts/train_chase.sh` (starts SB3 trainer, launches headless
  Godot training scene which connects on port 11008). ~34 min at 120k steps.
- **Train (rover, resumable):** `./scripts/train_rover.sh` — checkpoints to `models/rover_checkpoints/`
  every 25k steps and **auto-resumes** on re-run (survives shutdown / interruption). `FRESH=1` to
  restart from scratch; `CHECKPOINT_FREQ=N` to tune.
- **Convert + verify (one command):** `.venv-train/bin/python scripts/export_to_ncnn.py models/model.onnx`
  (auto-derives inputshape, runs pnnx, verifies parity, cleans intermediates). Flags: `--skip-verify`,
  `--keep-intermediates`, `--inputshape`, `--outdir`. Underlying manual steps: `../.venv/bin/pnnx model.onnx
  'inputshape=[1,5],[1]'` then `scripts/verify_ncnn_parity.py <onnx> <param> <bin> in0 out0`.

## Operational gotchas (learned the hard way)

- **Two venvs:** `.venv` (Python 3.14) has `pnnx`+torch for conversion. `.venv-train` (Python
  **3.13** — torch wheels don't exist for 3.14) has `godot-rl onnxruntime ncnn onnxscript`. Keep
  them separate. Both gitignored.
- **`onnxscript` is required** for ONNX export (torch 2.12 dynamo exporter needs it) — not pulled
  in by godot-rl automatically.
- **Do NOT pass `seed=` to `PPO()`** — godot-rl's env wrapper raises `NotImplementedError` on
  `env.seed()`. Seed via the env constructor only.
- **pnnx `inputshape` must be quoted** (`'inputshape=[1,5],[1]'`) or zsh globs the brackets. The
  second `[1]` is godot-rl's vestigial `state_ins` input — pnnx prunes it → clean `in0`/`out0`.
- **Parity tolerance is `atol=1e-2`** — torch dynamo exporter vs ncnn InnerProduct differ by
  ~1e-3 to 5e-3 in float32; argmax is stable.
- **The bridge sets `done` at `reset_after`** (godot_rl convention) so episodes terminate and
  `ep_rew_mean` appears. (A future chip splits this into `terminated`/`truncated`.)
- **`class_name` is unreliable headless:** the global class registry comes from
  `.godot/global_script_class_cache.cfg`, which is gitignored and is **not** rebuilt by
  `--headless`/`--script` runs (only an editor/import pass writes it). So `extends SomeClassName`
  fails (`Could not find base class`) on a fresh clone or after moving a `class_name` file. **Use
  path-based `extends "res://addons/godot_native_rl/.../foo.gd"`** for in-repo subclasses (the reward
  terms + example agents do this); reference scripts via `preload` consts, not bare `class_name`.
- **Don't commit Godot-generated `*.gd.uid` files** — an editor/import pass scatters them (and can
  re-materialize moved scripts at their old paths); `git clean -f -- '*.gd.uid'` and delete stray
  root duplicates before committing.

## Conventions

- GDScript uses **TAB** indentation. Dependency-free headless test harness at `test/harness.gd`
  (tests `extends SceneTree`, run via `godot --headless --path . --script res://test/...`).
- The reusable library lives under `addons/godot_native_rl/`; reference moved scripts by their
  full `res://addons/godot_native_rl/...` path and prefer **path-based `extends`** over bare
  `class_name` (see the headless gotcha above). Favor pure helpers + thin node wrappers and small,
  focused files.
- Python: 4-space indentation; tests are stdlib `unittest` under `test/python/` (auto-discovered by
  `run_tests.sh`); keep heavy imports (torch/SB3) lazy inside `main()` so pure helpers stay testable.
- Use the **superpowers workflow**: brainstorm → spec (`docs/superpowers/specs/`) → plan
  (`docs/superpowers/plans/`) → TDD implement on a feature branch. Don't push to `main` directly.
- **Before every push, check and update the docs** so they match the change: README, this
  `CLAUDE.md`, and `docs/BACKLOG.md`. Stale paths/commands/state count as a bug — fix them in the
  same change, not later.

## Roadmap & backlog

- **Strategy + gap analysis:** `docs/superpowers/specs/2026-05-30-feature-parity-roadmap-design.md`
  (four tracks: Sensors, Multi-Agent, Training Algorithms, DX/Distribution).
- **Novel addons + protocol findings:**
  `docs/superpowers/specs/2026-05-30-novel-addons-and-protocol-design.md` (10 addons in neither
  godot_rl nor Unity; 4 protocol upgrades incl. the `terminated`/`truncated` correctness fix).
- **Actionable backlog (pick up by number):** `docs/BACKLOG.md` — any session (incl. mobile) can
  start an item without clicking. Say "do backlog item N".
  - **Done:** 1 (Signal→Reward + RewardBuilder), 2 (export_to_ncnn helper), 3 (RaycastSensor2D/3D),
    4 (ncnn_vs_onnx guide), 5 (addon structure + controller refactor). **In progress:** 6 (3D rover —
    scaffold + resumable training done; trained model + golden regression pending).
  - **Newer items surfaced this work:** 21–24 (deploy-side inference gaps: continuous/multi-key
    actions, recurrent/LSTM, batched multi-agent, VecNormalize parity) and 25 (Asset Library release —
    move the GDExtension + prebuilt binaries into the addon and submit).

## The moat (why this beats godot_rl + Unity)

ncnn statically linked via C++ enables: web/WASM deployment (godot_rl's ONNX/.NET can't),
console deployment (no .NET cert issues), INT8 quantization game-side, async inference threads,
LOD policy switching, and Godot-native ideas (Signal→Reward, NavMesh sensor) — none replicable by
a Python-server framework or a managed-runtime one. Lead with these in all docs.
