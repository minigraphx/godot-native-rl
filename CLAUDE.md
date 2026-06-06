# Godot Native RL — Project Memory

## What this is

A GDExtension-based reinforcement-learning framework for Godot 4.5+ that uses Tencent's **ncnn**
for native inference (statically linked C++, **no C#/.NET, no external runtime**). It speaks the
`godot_rl_agents` wire protocol for training, so you train with the stock `godot-rl` Python package
and deploy with native ncnn — on mobile, web, console, desktop, and edge.

**Positioning (north star):** focused superiority first (be clearly better at deployment), then
godot_rl feature parity, then Unity ML-Agents parity (long-term stretch). Strategy: start as a
complement to godot_rl, grow toward full replacement.

## Current state (working, on `main`)

Full train → convert → deploy loop works end-to-end (headless CI tests). Reusable library in
`addons/godot_native_rl/` (`sync.gd`/`NcnnSync`, `controllers/`, `reward/`, `sensors/`,
`training/`, `net/`); C++ GDExtension at repo root (`src/ncnn_runner.{h,cpp}`). Examples:
`chase_the_target` (2D), `rover_3d` (3D), `hide_and_seek` (2D self-play), `ball_chase` (2D continuous-control / SAC). Wire protocol is
godot_rl v0.8.2-compatible. **Architecture + data flow + deploy contract:
[docs/dev/DEVELOPMENT.md](docs/dev/DEVELOPMENT.md).**

## Key commands

- **Build the extension:** `scons platform=macos arch=arm64 target=template_debug` (see README for other platforms). Project minimum is **Godot 4.5**; developed/tested on both 4.5 and 4.6 (e.g. `/opt/homebrew/bin/godot-mono` is 4.5.1). Set `GODOT=` to pick the binary for `run_tests.sh`.
- **Run all tests:** `./test/run_tests.sh` — headless GDScript unit tests + Python protocol test +
  inference smoke + trained-chase + golden regression + rover-3D smoke + Python helper tests. Must be
  green before merge. (`run_tests.sh` now **regenerates the script-class cache fresh on every run**
  — `rm` + a `--headless --editor --quit` import pass — so it self-heals both a *missing* cache (fresh
  clone) and a *stale* one (after a branch switch that moved/removed a `class_name` file); you no longer
  need to `rm` it manually. See the fresh-clone trap below.)
- **Train (chase):** `TIMESTEPS=120000 ./scripts/train_chase.sh` (starts SB3 trainer, launches headless
  Godot training scene which connects on port 11008). ~34 min at 120k steps.
- **Train (rover, resumable):** `./scripts/train_rover.sh` — checkpoints to `models/rover_checkpoints/`
  every 25k steps and **auto-resumes** on re-run. `FRESH=1` restart; `CHECKPOINT_FREQ=N` tune;
  `TIMESTEPS=N` raise the target to **refine** an existing model further. On macOS/Apple Silicon wrap
  it: `caffeinate -is ./scripts/train_rover.sh` (see sleep gotcha below).
- **Train (rover, parallel — fast):** `SCENE=res://examples/rover_3d/rover_3d_train_parallel.tscn
  ./scripts/train_rover.sh` — tiles 8 rover worlds in one process (`ParallelArena`), so godot-rl
  vectorizes over 8 agents (~Nx samples/sec). Trainer code is unchanged.
- **Train (hide & seek self-play):** `./scripts/train_hide_seek.sh` (one shared PPO policy over a
  seeker+hider AGENT group; `SCENE=res://examples/hide_and_seek/hide_and_seek_train_parallel.tscn`
  for 8 tiled worlds via `ParallelArena2D`).
- **Train (hide & seek, two distinct policies):** `./scripts/train_hide_seek_multipolicy.sh` — custom
  single-file multi-policy PPO; seeker + hider learn separate networks (distinct `policy_name`s via the
  `--multi-policy` cmdline gate read by `HideSeekAgent`), each exported to ncnn via
  `export_to_ncnn.py --via torchscript`. `SCENE=`/`TIMESTEPS=` overrides; the trained example for the
  `agent_policy_names` wire field. Deploy/regress in `hide_and_seek_multipolicy_eval.tscn`.
- **Train (BallChase, SAC):** `./scripts/train_ball_chase.sh` — SB3 SAC (continuous-control) over the
  BallChase env (port 11008). Exports the deterministic actor (tanh(mean)) as **TorchScript** (godot_rl's
  SAC ONNX export breaks under torch 2.x dynamo), then `scripts/export_to_ncnn.py models/ball_chase_sac.pt
  --via torchscript`.
- **Train (chase, CleanRL backend):** `./scripts/train_cleanrl.sh` — single-file CleanRL-style PPO over
  godot_rl's `CleanRLGodotEnv` (same chase scene + port 11008; `TIMESTEPS`/`SPEEDUP`/`ACTION_REPEAT`
  overrides). Exports ONNX (`models/chase_cleanrl_policy.onnx`) consumable unchanged by `export_to_ncnn.py`.
- **Train (chase, SampleFactory backend):** `./scripts/train_sf.sh` — SampleFactory async PPO over
  godot_rl's bridge (same chase scene; serial/sync + `normalize_input=False` so the actor is a plain
  MLP). Runs in the isolated **`.venv-sf`** (SF pins `gymnasium<1.0`); exports the SF checkpoint to
  **TorchScript** via `export_sf_to_torchscript.py` → ncnn (`.venv-sf` can't onnx-export).
  `TIMESTEPS`/`SPEEDUP`/`ACTION_REPEAT`/`BASE_PORT`/`EXPERIMENT`/`TRAIN_DIR`/`OUTDIR` overrides.
- **Throughput check:** `./scripts/throughput_compare.sh` — short fresh runs of the parallel vs
  single-agent scene into temp dirs (never touches `models/`); prints samples/sec + speedup **plus a
  per-step phase breakdown** (`collect_obs` / `serialize_send` / `await_action`) so you can see whether
  the sim, JSON serialization, or the socket round-trip dominates. The breakdown comes from `NcnnSync`'s
  opt-in `StepProfiler`, enabled with the `profile=true` cmdline arg (zero overhead otherwise).
- **Export a checkpoint (no full run):** `.venv-train/bin/python scripts/export_checkpoint.py`
  (latest checkpoint → `models/rover_policy.onnx`, non-destructive) then `scripts/export_to_ncnn.py`.
- **Export a checkpoint → TorchScript (ONNX-free):** `.venv-train/bin/python scripts/export_torchscript.py
  --checkpoint <ckpt.zip>` — traces the deterministic actor to `models/policy.pt` **and writes a
  `models/policy.pt.shape.json` sidecar**, so `export_to_ncnn.py models/policy.pt` auto-derives the shape
  with no flag (no `onnxscript`/dynamo hop). Same pnnx parity check; keep the ONNX path as default/fallback.
- **Convert + verify (one command):** `.venv-train/bin/python scripts/export_to_ncnn.py models/model.onnx`
  (auto-derives inputshape, runs pnnx, verifies parity, cleans intermediates). Flags: `--skip-verify`,
  `--keep-intermediates`, `--inputshape`, `--outdir`, `--via {onnx,torchscript,auto}`. Underlying manual
  steps: `../.venv/bin/pnnx model.onnx 'inputshape=[1,5],[1]'` then `scripts/verify_ncnn_parity.py <onnx>
  <param> <bin> in0 out0`.
- **Convert TorchScript → ncnn (skip ONNX):** `.venv-train/bin/python scripts/export_to_ncnn.py
  models/policy.pt` — runs pnnx on a `.pt`/`.ptl` directly (one fewer hop, pnnx's native format).
  `--via` defaults to `auto` (routes by extension). `inputshape` is **auto-derived** for `.pt` too: from
  a `<model>.shape.json` sidecar (`{"inputshape": "[1,5]"}` or `{"shape": [1,5]}`), else best-effort
  from the first `Linear` layer (MLPs); pass `--inputshape '[1,5]'` to override (still **required** for a
  conv-first stem — spatial dims aren't recoverable from weights). Parity = `torch.jit` vs ncnn at atol=1e-2.
- **Direct module → ncnn (no ONNX/TorchScript/pnnx):** `.venv-train/bin/python
  scripts/export_statedict_to_ncnn.py --checkpoint <ckpt.zip>` — writes `.ncnn.{param,bin}` straight
  from an SB3 MLP policy by hand-mapping layers (Input/Linear/ReLU/Tanh/Sigmoid/Flatten only; fails
  loud otherwise). Zero toolchain-deprecation exposure; simple feed-forward nets only. **Validate with
  `verify_ncnn_parity.py` before deploy** (the format writer is unit-tested; the round-trip isn't yet).
- **Export VecNormalize stats (deploy):** `.venv-train/bin/python scripts/export_vecnormalize.py
  vec_normalize.pkl` → JSON; set the controller's `obs_norm_stats_path` so `ObsNormalize` replays
  the obs mean/std game-side before inference (policies trained with SB3 `VecNormalize`).
- **Quantize to INT8 (deploy):** `./scripts/build_ncnn_tools.sh` (once) then
  `.venv-train/bin/python scripts/export_int8.py models/m.ncnn.param models/m.ncnn.bin
  --width W --height H --channels C --outdir models` (optimize → KL-calibrate → ncnn2int8 →
  argmax-parity). Produces `m_int8.ncnn.{param,bin}`; deploy via `NcnnRunner` like fp32.
- **Record expert demos:** `godot --headless --path . res://examples/chase_the_target/record_chase_demos.tscn -- --demo-out=PATH --demo-trajectories=N`
  (offline — no trainer/socket; `gnrl_v1` default format; set `demo_format="godot_rl"` on the `NcnnSync` node for stock-tooling interop).
- **Behavior cloning:** `.venv-train/bin/python scripts/train_bc.py --demos PATH --out models/bc.pt`
  then `.venv-train/bin/python scripts/export_to_ncnn.py models/bc.pt` (deploys via the normal ncnn pipeline).

## Operational gotchas

Full list (learned the hard way): **[docs/dev/gotchas.md](docs/dev/gotchas.md)**. The few that bite
daily:
- **`class_name` is unreliable headless** — prefer path-based `extends "res://addons/..."`.
- **Three venvs** — `.venv` (3.14, pnnx+torch) convert; `.venv-train` (3.13, godot-rl+SB3) train
  (also runs `export_to_ncnn.py`); `.venv-sf` (3.13, SampleFactory — pins `gymnasium<1.0`, so
  isolated) for the SF backend only. Create all three with `./scripts/setup_training.sh`.
- **macOS: never sleep during training** — wrap in `caffeinate -is`.
- **Rebuild the extension on a fresh clone** — `bin/` is gitignored.

## Conventions

- **Deeper dev reference** (architecture, data flow, the long-form "why") lives in
  `docs/dev/DEVELOPMENT.md`. Keep CLAUDE.md terse (it's always-loaded); put new deep-dives there.
- GDScript uses **TAB** indentation. Dependency-free headless test harness at `test/harness.gd`
  (tests `extends SceneTree`, run via `godot --headless --path . --script res://test/...`).
- The reusable library lives under `addons/godot_native_rl/`; reference moved scripts by their
  full `res://addons/godot_native_rl/...` path and prefer **path-based `extends`** over bare
  `class_name` (see the headless gotcha above). Favor pure helpers + thin node wrappers and small,
  focused files.
- **Godot 4.6 `:=` can't infer from an untyped value** — `var xs := some_untyped_var.get_children()`
  fails to parse (`Cannot infer the type`). Annotate explicitly: `var xs: Array = ...`.
- Python: 4-space indentation; tests are stdlib `unittest` under `test/python/` (auto-discovered by
  `run_tests.sh`); keep heavy imports (torch/SB3) lazy inside `main()` so pure helpers stay testable.
- Use the **superpowers workflow**: brainstorm → spec (`docs/superpowers/specs/`) → plan
  (`docs/superpowers/plans/`) → TDD implement on a feature branch. Don't push to `main` directly.
- **Before every push, check and update the docs** so they match the change: README, this
  `CLAUDE.md`, and `docs/godot-rl-gap-analysis-2026-06-02.md`. Also flip the checkbox in
  `docs/BACKLOG.md` and close the GitHub issue (`Closes #NN`) if the change ships a listed item.
  Stale paths/commands/state count as a bug — fix them in the same change, not later.

## Roadmap & backlog

- **Strategy + gap analysis:** `docs/superpowers/specs/2026-05-30-feature-parity-roadmap-design.md`
  (four tracks: Sensors, Multi-Agent, Training Algorithms, DX/Distribution).
- **Novel addons + protocol findings:**
  `docs/superpowers/specs/2026-05-30-novel-addons-and-protocol-design.md` (10 addons in neither
  godot_rl nor Unity; 4 protocol upgrades incl. the `terminated`/`truncated` correctness fix).
- **Open work (primary SSOT):** GitHub issues — labelled `backlog` + `area:*` + `priority:1–5`.
  Say "do issue #N". New items go to GitHub only.
  `docs/BACKLOG.md` tracks the originally-listed items until they're all done (then retires);
  it's updated by each closing PR but not extended with new entries.
  - **Done:** 1 (Signal→Reward + RewardBuilder), 2 (export_to_ncnn helper), 3 (RaycastSensor2D/3D),
    4 (ncnn_vs_onnx guide), 5 (addon structure + controller refactor), 6 (3D rover + trained model +
    golden regression), 7 (RelativePositionSensor2D/3D), 42 (RelativePositionSensor multi-target + PositionSensor parity),
    8 (CameraSensor — hex image obs protocol),
    36 (deploy-side image inference — `run_inference_image` glue + synthetic-CNN golden),
    30 (ParallelArena — parallel multi-agent training, ~6.2× speedup measured),
    12 (Hide & Seek example — 2D 1v1 parameter-sharing self-play, scaffold + smoke test),
    21 (continuous + multi-key action deploy),
    24 (obs-normalization VecNormalize parity),
    13 (INT8 quantization export), 17 (CleanRL backend — single-file PPO over `CleanRLGodotEnv`),
    33 (TorchScript → ncnn direct export, `--via {onnx,torchscript,auto}`),
    11 (GridSensor2D/3D — query-based cell detection, per-layer overlap counts),
    39 (`get_obs_space()` on controllers — already present),
    40 (ISensor2D/3D interface + `collect_sensors()` sensor auto-discovery),
    41 (RaycastSensor2D/3D `class_sensor` multi-class detection — per-ray multi-hot layer slots +
    optional other/closeness, pure `raycast_math.encode_ray_class`; 2D added alongside 3D),
    44 (`INHERIT_FROM_SYNC` per-agent control mode — already present in `NcnnSync._get_agents()`),
    20 (multi-policy `policy_name` wire field — `agent_policy_names` in env_info; the rest of the
    old item-20 catalog line was split 2026-06-03 into items 46–54, trained example is item 45),
    45 (multi-policy trained example — Hide & Seek seeker+hider as two distinct policies via a custom
    single-file multi-policy PPO over `CleanRLGodotEnv`, `--multi-policy` cmdline identity gate,
    TorchScript→ncnn export, golden-inference + deterministic LOS behavioral regression; #73 tracks a
    cleaner identity mechanism),
    43 (stochastic action sampling — `deterministic_inference`/`inference_seed` on controllers,
    discrete softmax-sample via seedable RNG in core),
    22 (recurrent/LSTM deploy — `NcnnRunner.run_inference_multi` multi-IO + `NcnnControllerCore`
    hidden-state carry + `recurrent.json` sidecar; synthetic-LSTM golden; real RecurrentPPO
    train/export deferred),
    10 (expert-demo recording — pure `DemoRecorder` + `NcnnSync` `RECORD_EXPERT_DEMOS` mode,
    `gnrl_v1`/`godot_rl` formats, Python loader + `train_bc.py` BC trainer, chase scripted-expert
    example + committed sample + headless smoke),
    46 (ObsHistoryBuffer — frame-stacking sensor wrapper, #17; pure FrameRing + dimension-agnostic
    Node, zero-filled window, per-episode reset),
    47 (RunningNormSensor — online VecNormalize-parity normalization, #18; pure RunningStats Welford,
    freeze + JSON sidecar persistence, game-side so no Python at deploy).
    GitHub #45 (algorithm-agnostic train/deploy contract — note: GitHub issue #45, **not** internal
    item 45 which is the multi-policy trained example/#26; proven for non-PPO by synthetic DQN
    unbounded-Q argmax + SAC tanh-squash fixtures through the real ncnn pipeline,
    `test_algorithm_agnostic_golden_inference.gd`; live-trained SB3 SAC regression filed as
    follow-up #74).
    GitHub #74 (trained SB3 SAC non-PPO regression — live train → TorchScript export → ncnn →
    behavioral check; continuous BallChase env ported from godot_rl_agents_examples; the
    live-trained follow-up to #45. Note: GitHub issue #74.)
    18 (SampleFactory training backend — async PPO, chase example, TorchScript→ncnn export,
    headless smoke; committed golden-inference regression `test_chase_sf_golden_inference.gd` +
    `models/chase_sf_policy.ncnn.*` fixture added in #79).
    9 partial (socket
    timeout + per-agent `info`; `terminated`/`truncated` blocked upstream).
  - **Newer items surfaced this work:** 23 (deploy-side inference gap: batched multi-agent
    inference; 21/22/24 — continuous/multi-key actions, recurrent/LSTM, VecNormalize parity — now
    done) and 25 (Asset Library release —
    move the GDExtension + prebuilt binaries into the addon and submit).

## The moat (why this beats godot_rl + Unity)

ncnn statically linked via C++ enables: web/WASM deployment (godot_rl's ONNX/.NET can't),
console deployment (no .NET cert issues), INT8 quantization game-side, async inference threads,
LOD policy switching, and Godot-native ideas (Signal→Reward, NavMesh sensor) — none replicable by
a Python-server framework or a managed-runtime one. Lead with these in all docs.
