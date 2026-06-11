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

Full train → convert → deploy loop works end-to-end (headless CI tests). All four examples have
standalone headless-compatible play scenes; trained inference is available for chase, rover,
multi-policy hide & seek, and BallChase. Reusable library in
`addons/godot_native_rl/` (`sync.gd`/`NcnnSync`, `controllers/`, `reward/`, `sensors/` (+ drop-in `scenes/`),
`training/`, `net/`, `script_templates/` (controller scaffold, auto-installed on plugin enable)); C++ GDExtension at repo root (`src/ncnn_runner.{h,cpp}`). Examples:
`chase_the_target` (2D, + `chase_crowd.tscn` batched shared-policy crowd via `run_inference_batch` + `NcnnCrowdController`), `rover_3d` (3D), `hide_and_seek` (2D self-play), `ball_chase` (2D continuous-control / SAC), `fly_by` (3D continuous-control / PPO, ships the #64 DiagGaussian-sampling demo), `quadruped_walk` (3D continuous-control locomotion — code-built 8-hinge-joint articulated quadruped on the **Jolt** backend; #60 PR1 harness landed: rig+agent+train/track scenes+physics smoke, the trained walking net is the follow-up training run). Wire protocol is
godot_rl v0.8.2-compatible. **Architecture + data flow + deploy contract:
[docs/dev/DEVELOPMENT.md](docs/dev/DEVELOPMENT.md).**

## Key commands

- **Build the extension:** `scons platform=macos arch=arm64 target=template_debug` (see README for other platforms). Project minimum is **Godot 4.5**; developed/tested on both 4.5 and 4.6 (e.g. `/opt/homebrew/bin/godot-mono` is 4.5.1). Set `GODOT=` to pick the binary for `run_tests.sh`.
- **Build the extension (web/WASM):** `source ~/emsdk/emsdk_env.sh && scripts/cross/build_web.sh` (single-threaded; needs emsdk 3.1.64). No COOP/COEP headers required at deploy — see `docs/dev/building.md`. Model `*.ncnn.param`/`*.ncnn.bin` are auto-packed into exports by the enabled addon's `EditorExportPlugin` (`addons/godot_native_rl/export/`); without the plugin, set an `include_filter` by hand. (Raw data files the Godot exporter skips otherwise — affects all platforms.)
- **Cut a release:** bump `addons/godot_native_rl/plugin.cfg` `version=`, then `git tag vX.Y.Z &&
  git push origin vX.Y.Z` → `.github/workflows/release.yml` builds all platforms, **runtime/symbol-validates
  each binary** (shared `validate-binaries.yml`, also used by `cross-build.yml`; publish is gated on it),
  assembles the addon + examples zips, smoke-tests the packaged addon, and publishes a GitHub Release. Then
  update the Asset Library entry by hand (`Custom` download → the addon-zip URL + sha256). Full
  runbook: [docs/dev/RELEASING.md](docs/dev/RELEASING.md).
- **Run all tests:** `./test/run_tests.sh` — headless GDScript unit tests + Python protocol test +
  inference smoke + trained-chase + golden regression + rover-3D smoke + Python helper tests. Must be
  green before merge. (`run_tests.sh` now **regenerates the script-class cache fresh on every run**
  — `rm` + a `--headless --editor --quit` import pass — so it self-heals both a *missing* cache (fresh
  clone) and a *stale* one (after a branch switch that moved/removed a `class_name` file); you no longer
  need to `rm` it manually. See the fresh-clone trap below.)
- **CI:** `.github/workflows/ci.yml` runs on PRs/pushes to `main`. A `build` job compiles the
  GDExtension once (godot-cpp `4.5` + ncnn tag `20260526`, both cached) and uploads `addons/godot_native_rl/bin/`; a `test`
  matrix runs the full `test/run_tests.sh` under Godot **4.5 + 4.6** (`.venv`/`.venv-train` cached).
  The SF smoke auto-skips (no `.venv-sf` in CI). The **Build GDExtension** step is skipped on a
  content-addressed `bin/` cache hit (key = `hashFiles('src/**','SConstruct')` + `GODOT_CPP_BRANCH` +
  `NCNN_TAG` + `runner.os` + `CACHE_VERSION`), so Python/GDScript/docs-only PRs reuse the prior binary
  and the `build` job drops to ~1–2 min; the `test` matrix always runs. On a `bin/` miss the
  extension build passes `build_library=no`, linking the **prebuilt** godot-cpp libs from the
  godot-cpp cache instead of recompiling the bindings (#85) — C++-change builds are ~2–3 min, not
  ~15. Bump `CACHE_VERSION` in the
  workflow to force a cold rebuild (also busts this `bin/` cache); bump the Godot patch versions in the
  `test` matrix to track new releases.
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
- **Train (multi-policy, PettingZoo interop):** `./scripts/train_pettingzoo.sh` — multi-policy PPO over
  our own `GodotParallelEnv` PettingZoo `ParallelEnv` adapter (`scripts/godot_pettingzoo_env.py`; the
  godot_rl `GDRLPettingZooEnv` functionality without depending on the upstream class). Reads
  `agent_policy_names`, one learner per policy, each actor → TorchScript → `export_to_ncnn.py`. Interop
  proven deterministically via PettingZoo's `parallel_api_test`; live-trained fixtures committed
  (`models/pettingzoo_{seeker,hider}.ncnn.*`) with golden-inference + LOS behavioral regression
  (#118). `SCENE`/`TIMESTEPS`/`NUM_STEPS`
  overrides; exits loud if `TIMESTEPS` < one rollout batch (`NUM_STEPS` × n_agents) instead of
  silently exporting an untrained policy (#119) — lower `NUM_STEPS` for short smoke runs.
- **Train (BallChase, SAC):** `./scripts/train_ball_chase.sh` — SB3 SAC (continuous-control) over the
  BallChase env (port 11008). Exports the deterministic actor (tanh(mean)) as **TorchScript** (godot_rl's
  SAC ONNX export breaks under torch 2.x dynamo), then `scripts/export_to_ncnn.py models/ball_chase_sac.pt
  --via torchscript`. Re-export a saved SAC checkpoint without retraining via
  `scripts/export_sac_torchscript.py --checkpoint models/ball_chase_sac.zip` (see issue #81 / `docs/ncnn_vs_onnx.md`).
- **Train (FlyBy, PPO continuous):** `./scripts/train_fly_by.sh` — SB3 PPO over the FlyBy plane env
  (port 11008), 2 continuous actions (`pitch`/`turn`), 8-dim plane-local obs. Exports the deterministic
  actor (action mean) as **TorchScript** → `export_to_ncnn.py models/fly_by_policy.pt`,
  plus the std sidecar via `scripts/export_action_dist.py`. (The original numpy<2/onnx-1.19 blocker
  that forced TorchScript here is gone since #126 moved `.venv-train` to numpy≥2 + onnx 1.21, but
  TorchScript still works fine, so the script is unchanged.) The play scene (`fly_by.tscn`) ships
  `deterministic_inference=true`; flip it to `false` to demo continuous DiagGaussian sampling (#64).
  `TIMESTEPS`/`SCENE` overrides.
- **Train (quadruped walk, PPO continuous):** `./scripts/train_quadruped.sh` — SB3 PPO over the
  code-built articulated quadruped (8 hinge-joint motor targets = one continuous `motors` action,
  ~29-dim obs: joint angles/vels + torso up + body-local velocity + dir-to-finish + foot contacts).
  Defaults to the tiled `quadruped_walk_train_parallel.tscn` (8 worlds via `ParallelArena`); **Jolt**
  backend (set in `project.godot`). Exports the deterministic actor as **TorchScript** →
  `export_to_ncnn.py models/quadruped_walk.pt` (+ `export_action_dist.py` for the std sidecar).
  `TIMESTEPS`/`SPEEDUP`/`ACTION_REPEAT`/`SCENE` overrides; wrap in `caffeinate -is` on macOS. Reward
  shaping is finicky (#60) — the committed walking net is the PR2 follow-up training run.
- **Train (chase, CleanRL backend):** `./scripts/train_cleanrl.sh` — single-file CleanRL-style PPO over
  godot_rl's `CleanRLGodotEnv` (same chase scene + port 11008; `TIMESTEPS`/`SPEEDUP`/`ACTION_REPEAT`
  overrides). Exports ONNX (`models/chase_cleanrl_policy.onnx`) consumable unchanged by `export_to_ncnn.py`.
- **Train (chase, SampleFactory backend):** `./scripts/train_sf.sh` — SampleFactory async PPO over
  godot_rl's bridge (same chase scene; serial/sync + `normalize_input=False` so the actor is a plain
  MLP). Runs in the isolated **`.venv-sf`** (SF pins `gymnasium<1.0`); exports the SF checkpoint to
  **TorchScript** via `export_sf_to_torchscript.py` → ncnn (`.venv-sf` can't onnx-export).
  `TIMESTEPS`/`SPEEDUP`/`ACTION_REPEAT`/`BASE_PORT`/`EXPERIMENT`/`TRAIN_DIR`/`OUTDIR` overrides.
- **Train (chase, RLlib backend):** `./scripts/train_rllib.sh` — stock Ray/RLlib PPO (new API
  stack) over the godot_rl wire protocol via a thin custom gymnasium adapter (the stock
  `RayVectorGodotEnv` is old-API-stack only). Shares **`.venv-train`** (ray add-on installed on top;
  `gymnasium==1.2.2`, godot-rl `--no-deps` — #126); single socket (`num_env_runners=0`);
  exports the RLModule actor → TorchScript → `export_to_ncnn.py`. Ecosystem interop (#110), not a
  replacement for the custom trainers. `TIMESTEPS`/`SPEEDUP`/`ACTION_REPEAT`/`BASE_PORT`/
  `EXPERIMENT`/`TRAIN_DIR`/`OUTDIR`/`SCENE` overrides.
- **Throughput check:** `./scripts/throughput_compare.sh` — short fresh runs of the parallel vs
  single-agent scene into temp dirs (never touches `models/`); prints samples/sec + speedup **plus a
  per-step phase breakdown** (`collect_obs` / `serialize_send` / `await_action`) so you can see whether
  the sim, JSON serialization, or the socket round-trip dominates. The breakdown comes from `NcnnSync`'s
  opt-in `StepProfiler`, enabled with the `profile=true` cmdline arg (zero overhead otherwise).
- **Export a checkpoint (no full run):** `.venv-train/bin/python scripts/export_checkpoint.py`
  (latest checkpoint → `models/rover_policy.onnx`, non-destructive) then `scripts/export_to_ncnn.py`.
  Checkpoint selection is shared/`scripts/checkpoints.py`: trainers resume by **highest step
  count** (mtime-free, so `FRESH`/`cp -p`/backups can't pick a weaker ckpt); exporters deploy by
  **best-reward (#138, when present) → highest-step → mtime(legacy)** (#105).
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
- **Export continuous action std (deploy):** `.venv-train/bin/python scripts/export_action_dist.py
  models/policy.zip` → `*_action_dist.json`; set the controller's `action_dist_stats_path` and
  `deterministic_inference=false` so `ActionDecode` samples `mean + std·N(0,1)` (PPO DiagGaussian,
  game-side) instead of the mean. PPO continuous only (SAC std is state-dependent — out of scope).
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
- **Three venvs** — `.venv` (3.14, pnnx+torch) convert; `.venv-train` (3.13) train + verify +
  **RLlib** (SB3 2.8.0 + gymnasium 1.2.2 + numpy≥2 + ray add-on; godot-rl `--no-deps`; also runs
  `export_to_ncnn.py`); `.venv-sf` (3.13, SampleFactory — pins `gymnasium<1.0`, so still isolated)
  for the SF backend only. Create all with `./scripts/setup_training.sh`. (RLlib folded into
  `.venv-train` in #126, retiring `.venv-rllib` and the old numpy<2/onnx==1.17.0 pin hack; SF stays
  separate until SampleFactory ships a gymnasium-1.x release.)
- **macOS: never sleep during training** — wrap in `caffeinate -is`.
- **Rebuild the extension on a fresh clone** — `addons/godot_native_rl/bin/` is gitignored.

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
    GitHub #81 (SAC ONNX export broken under torch 2.x — standardized on TorchScript: promoted
    `export_sac_actor_as_torchscript` into `scripts/export_sac_torchscript.py` with a standalone
    `--checkpoint` CLI; documented + test-guarded the `dynamo=False` legacy-ONNX fallback. Note:
    GitHub issue #81.)
    18 (SampleFactory training backend — async PPO, chase example, TorchScript→ncnn export,
    headless smoke; committed golden-inference regression `test_chase_sf_golden_inference.gd` +
    `models/chase_sf_policy.ncnn.*` fixture added in #79).
    49 (in-editor Policy Debugger — drop-in PolicyDebugOverlay node + inference_step signal on
    controllers + pure PolicyDebug formatter; live obs/action-probs/identity/get_debug_status overlay,
    auto-discovery + F3 toggle + debug-build gate; headless helper/overlay/emit tests + chase debug
    scene),
    GitHub #111 (PettingZoo `ParallelEnv` interop adapter — `GodotParallelEnv` in
    `scripts/godot_pettingzoo_env.py` provides `GDRLPettingZooEnv` functionality without depending on
    the upstream class; `train_pettingzoo.sh` drives multi-policy PPO one learner per `agent_policy_names`
    each actor → TorchScript → ncnn; conformance proven via PettingZoo's `parallel_api_test`;
    live full training run shipped as #118).
    GitHub #118 (PettingZoo live-trained two-policy regression — full multi-policy run through
    `train_pettingzoo.sh`, committed `models/pettingzoo_{seeker,hider}.ncnn.*` fixtures,
    `test_pettingzoo_golden_inference.gd` golden + `trained_pettingzoo_eval.tscn` LOS behavioral
    check reusing the multipolicy checker. Note: GitHub issue #118.)
    GitHub #110 (RLlib training backend — stock Ray/RLlib PPO on the new API stack over the
    godot_rl wire via a custom gymnasium adapter (`GodotRLlibEnv` in `scripts/train_rllib.py`;
    the stock `RayVectorGodotEnv` is old-API-stack only); shares `.venv-train` since #126
    (gymnasium 1.2.2, godot-rl `--no-deps`); RLModule actor → TorchScript →
    `export_to_ncnn.py`; guarded end-to-end smoke in `run_tests.sh` + committed
    golden-inference fixture `models/chase_rllib_policy.ncnn.*` +
    `test_chase_rllib_golden_inference.gd`. Note: GitHub issue #110.)
    9 partial (socket
    timeout + per-agent `info`; `terminated`/`truncated` blocked upstream).
  - **Newer items surfaced this work:** 23 (deploy-side inference gap: batched multi-agent
    inference; 21/22/24 — continuous/multi-key actions, recurrent/LSTM, VecNormalize parity — now
    done) and 25 (Asset Library release —
    move the GDExtension + prebuilt binaries into the addon and submit).

## The moat (why this beats godot_rl + Unity)

ncnn statically linked via C++ enables: web/WASM deployment (godot_rl's ONNX/.NET can't),
console deployment (no .NET cert issues), INT8 quantization game-side, async inference threads,
thread-parallel batched crowd inference (`run_inference_batch` + `NcnnCrowdController`, one shared `Net`),
LOD policy switching, and Godot-native ideas (Signal→Reward, NavMesh sensor) — none replicable by
a Python-server framework or a managed-runtime one. Lead with these in all docs.
