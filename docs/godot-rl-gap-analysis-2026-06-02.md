# godot_rl Ecosystem Gap Analysis

**Date:** 2026-06-02 · **status refreshed 2026-06-09**  
**Repos audited:** `edbeeching/godot_rl_agents` · `edbeeching/godot_rl_agents_plugin` · `edbeeching/godot_rl_agents_examples`  
**This repo state:** backlog items 1–8, 10–13, 17–18, 20–22, 24–25, 30, 33, 36, 39–47, 49 done;
item 9 partial (terminated/truncated blocked upstream). GitHub #45, #64, #74, #79, #81, #111 also closed.
Open gaps tracked as GitHub issues (see table below).
(2026-06-03 refresh: GridSensor + ISensor interface shipped; `INHERIT_FROM_SYNC` already wired;
`policy_name`/`agent_policy_names` wire field shipped — RLlib/PettingZoo *trainers* now unblocked,
tracked as issue #26)
(2026-06-04 refresh: expert-demo recording shipped — `RECORD_EXPERT_DEMOS` mode + `gnrl_v1`/`godot_rl`
formats + Python loader + `train_bc.py` BC trainer + chase scripted-expert example; item 10 done)
(2026-06-06 refresh: continuous BallChase example added — 2D SAC-trained agent ported from
`edbeeching/godot_rl_agents_examples`, logic reimplemented against this addon (NcnnSync, RewardBuilder),
upstream plugin not vendored. Trains with SB3 SAC via `SBGSingleObsEnv`; exports deterministic actor
(tanh(mean)) as TorchScript (godot_rl's SAC ONNX export breaks under torch 2.x dynamo); converts to
ncnn via `export_to_ncnn.py --via torchscript`. Behavioral regression in CI. Closes #74 — live-trained
non-PPO follow-up to #45.)
(2026-06-07 refresh: Asset Library release shipped (item 25) — `release.yml` builds all platforms
including web/WASM on every `vX.Y.Z` tag; EditorExportPlugin auto-packs `*.ncnn.*` into game exports;
web proven in-browser without COOP/COEP headers. ObsHistoryBuffer (item 46) + RunningNormSensor
(item 47) shipped 2026-06-05. Multi-policy trained example (item 45) shipped 2026-06-05 — custom
single-file multi-policy PPO, seeker+hider with distinct ncnn models. SampleFactory golden regression
added (#79).)
(2026-06-09 refresh: In-editor Policy Debugger shipped (item 49) — `PolicyDebugOverlay`, `inference_step`
signal, F3 toggle, debug-build gate, auto-discovery; closes #23. Continuous DiagGaussian action
sampling via log_std sidecar shipped (#64) — game-side `mean + std·N(0,1)` for PPO continuous
policies via `action_dist_stats_path`; closes #64. SAC ncnn export standardised on TorchScript (#81).)
(2026-06-09 audit: re-checked every gap row against open issues. The remaining godot_rl_agents
compatibility gaps that had **no open tracking issue** were filed into the `v0.2 — godot_rl complement`
milestone — RLlib `RayVectorGodotEnv` (#110), PettingZoo `GDRLPettingZooEnv` (#111), plugin editor-DX
parity / sensor `.tscn` scenes + `script_templates/AIController` (#112), Optuna HP-tuning example (#113).
The old `#26` references on the RLlib/PettingZoo rows were stale — #26 shipped a custom multi-policy PPO
and is closed; the stock-wrapper interop is now tracked separately.)
(2026-06-09 refresh: PettingZoo `ParallelEnv` interop shipped (#111) — `GodotParallelEnv` adapter in
`scripts/godot_pettingzoo_env.py` provides `GDRLPettingZooEnv` functionality without depending on the
upstream class; `train_pettingzoo.sh` drives multi-policy PPO; conformance proven via PettingZoo's
`parallel_api_test`. Live training run is a follow-up. RLlib (#110) remains open.)

---

## Sensors

| Sensor | Upstream plugin | This repo | Status |
|---|---|---|---|
| `RaycastSensor2D` | ✅ | ✅ | — |
| `RaycastSensor3D` (distance) | ✅ | ✅ | — |
| `RaycastSensor3D` class mode | ✅ `class_sensor` + `boolean_class_mask` — one-hot per class per ray | ✅ `class_sensor` mode on both 2D+3D, per-ray multi-hot layer segments via `detection_classes` | ✅ done (#42) |
| `ISensor2D` / `ISensor3D` interface | ✅ shared base all sensors implement | ✅ + `collect_sensors()` auto-discovery | ✅ done (item 40) |
| `PositionSensor2D/3D` | ✅ multi-target `Array[Node2D]`, optional dir/dist split | ✅ multi-target `objects_to_observe`, both modes + axis toggles | ✅ done (#15) |
| `RGBCameraSensor2D/3D` | ✅ configurable render res + downscale + RGBA/RGB + editor preview | ✅ fixed viewport res, RGB only, no downscale | ⚠️ partial (#36) |
| `GridSensor2D` | ✅ area/body occupancy grid, multi-layer, debug view | ✅ query-based, per-layer counts | ✅ done (item 11) |
| `GridSensor3D` | ✅ | ✅ | ✅ done (item 11) |
| Pre-built sensor `.tscn` scenes | ✅ RaycastSensor2D.tscn, RGBCameraSensor2D.tscn + examples | ❌ | Minor (#112) |
| `script_templates/AIController` | ✅ controller scaffold template in plugin | ❌ | Minor (#112) |

### CameraSensor detail gap
Upstream `RGBCameraSensor2D` exports: `render_image_resolution` (default 36×36),
`downscale_image`, `resized_image_resolution`, RGBA/RGB auto-detect, live editor preview,
`camera_zoom_factor`. This repo's `CameraSensor` captures at the viewport's current resolution
with no resize, RGB-only, no editor preview. Grayscale (1-channel) deploy also missing in the
C++ runner (needs a `PIXEL_GRAY` path in `NcnnRunner`).

---

## Controller / Agent interface

| Feature | Upstream | This repo | Status |
|---|---|---|---|
| `AIController2D` / `AIController3D` base | ✅ | ✅ (`NcnnAIController2D/3D`) | — |
| `HUMAN` / `TRAINING` modes | ✅ | ✅ | — |
| `ONNX_INFERENCE` (requires C#/.NET) | ✅ | ❌ → replaced by `NCNN_INFERENCE` | By design |
| `INHERIT_FROM_SYNC` mode | ✅ per-agent can override scene-level default | ✅ wired in `NcnnSync._get_agents()` — INHERIT agents adopt sync mode, others override | ✅ done (item 44) |
| `RECORD_EXPERT_DEMOS` mode | ✅ | ✅ offline mode on `NcnnSync`; `gnrl_v1` default format + `godot_rl` interop | ✅ done (item 10) |
| `policy_name` export | ✅ default `"shared_policy"` | ✅ default `"shared_policy"` on `NcnnAIController2D/3D` | ✅ done (item 20) |
| `get_obs_space()` method | ✅ required on every agent | ✅ implemented — delegates to `obs_space_from_obs()` | — (item 39 ✅) |
| `get_action()` for demo recording | ✅ required when recording | ✅ hook on controllers | ✅ done (item 10) |
| `expert_demo_save_path` export | ✅ | ✅ on `NcnnSync` | ✅ done (item 10) |
| `remove_last_episode_key` binding | ✅ undo bad demonstration | ✅ `remove_last_episode_action` export on `NcnnSync` | ✅ done (item 10) |
| Stochastic action sampling | ✅ `deterministic_inference` flag (softmax vs argmax) | ✅ `deterministic_inference` + `inference_seed`; discrete softmax-sample **+ continuous DiagGaussian sample via a `std` sidecar (godot_rl's export drops the std; we keep it game-side)** | ✅ done (#16, #64) |
| VecNormalize obs replay | ❌ upstream | ✅ `obs_norm_stats_path` | **Advantage** |

---

## Sync node / wire protocol

| Feature | Upstream | This repo | Status |
|---|---|---|---|
| Training bridge (protocol v0.7) | ✅ | ✅ | — |
| `agent_policy_names` in env_info | ✅ | ✅ always emitted (one entry per training agent, obs order) | ✅ done (item 20) |
| `call()` remote method invocation | ✅ Python can invoke arbitrary Godot methods | ✅ handled in `NcnnSync` | — |
| `terminated`/`truncated` split | ❌ TODO both sides | ❌ | Parity (#12) |
| Connect / read timeouts | ❌ | ✅ | **Advantage** |
| Per-agent `info` field | ❌ | ✅ | **Advantage** |
| `deterministic_inference` export on Sync | ✅ | ✅ on `NcnnAIController2D/3D` (per-agent) | ✅ done (#16) |

---

## Python trainer wrappers

| Wrapper | Upstream | This repo | Status |
|---|---|---|---|
| `StableBaselinesGodotEnv` (SB3 VecEnv, n_parallel) | ✅ | ✅ proven | — |
| `SBGSingleObsEnv` (SB3 + `MlpPolicy` compat) | ✅ | ✅ used by `train_ball_chase.py` (SAC) | ✅ done (#74) |
| `CleanRLGodotEnv` | ✅ | ✅ item 17 done | — |
| `RayVectorGodotEnv` (RLlib) | ✅ | ❌ no training script | **Gap** (#110) |
| `GDRLPettingZooEnv` (PettingZoo, multi-policy) | ✅ | ✅ `GodotParallelEnv` in `scripts/godot_pettingzoo_env.py` — `GDRLPettingZooEnv` functionality without the upstream class; `parallel_api_test` conformance; live training run is a follow-up | ✅ done (#111) |
| `SampleFactoryEnvWrapper` (batched + non-batched) | ✅ | ✅ done (#24) — `train_sf.sh`, async PPO, TorchScript→ncnn, isolated `.venv-sf` | — |
| ONNX export helper (`OnnxablePolicy`) | ✅ SB3/SAC → ONNX | ✅ `export_to_ncnn.py` ONNX+TorchScript→ncnn | Different, covered |
| Optuna HP tuning example | ✅ | ❌ | Nice-to-have (#113) |

---

## Deploy-side inference

| Feature | Upstream | This repo | Status |
|---|---|---|---|
| In-game ONNX inference (C#/.NET) | ✅ | ❌ | By design |
| In-game ncnn inference | ❌ | ✅ | **Advantage** |
| Discrete action deploy | ✅ | ✅ | — |
| Continuous + multi-key deploy | ❌ | ✅ item 21 | **Advantage** |
| Camera/image deploy | ❌ | ✅ item 36 | **Advantage** |
| Grayscale (1-channel) camera deploy | ❌ | ❌ needs C++ `PIXEL_GRAY` | **Gap** (#36) |
| VecNormalize obs parity | ❌ | ✅ item 24 | **Advantage** |
| INT8 quantization | ❌ | ✅ item 13 | **Advantage** |
| TorchScript → ncnn export | ❌ | ✅ item 33 | **Advantage** |
| Recurrent / LSTM deploy | ❌ | ✅ item 22 | **Advantage** (deploy; training/export pending #33) |
| Batched multi-agent inference | ❌ | ✅ `run_inference_batch` (thread-parallel) + `CrowdController` + `chase_crowd` | **Advantage** (#34) |

---

## Unique to this repo (not in upstream)

| Feature | Notes |
|---|---|
| Native ncnn inference (no .NET/C#) | Enables web, mobile, console, edge deploy |
| `ParallelArena` / `ParallelArena2D` | Scene-level agent tiling → ~Nx training speedup |
| INT8 quantization pipeline | `export_int8.py` + `build_ncnn_tools.sh` |
| VecNormalize obs replay game-side | `ObsNormalize` + `export_vecnormalize.py` |
| Continuous + multi-key action deploy | `action_decode.gd` + C++ fix |
| Recurrent / LSTM deploy (hidden-state carry) | `run_inference_multi` + `recurrent_state.gd` + `.recurrent.json` |
| TorchScript → ncnn direct export | `--via torchscript` in `export_to_ncnn.py` |
| Socket connect/read timeouts | Clean exit on dead trainer |
| Per-agent `info` field | `get_info()` hook on controllers |
| `RewardBuilder` / `RewardAdapter` | More expressive than upstream's `ApproachNodeReward` |
| `ObsHistoryBuffer` (frame-stacking sensor wrapper) | `ISensor`-conforming ring-buffer, `N × inner.obs_size()`, auto-discovered by `collect_sensors()` |
| `RunningNormSensor` (online Welford normalisation) | No Python `VecNormalize` at deploy; Welford mean/var, freeze + JSON sidecar |
| In-editor Policy Debugger (`PolicyDebugOverlay`) | Live obs / action-probs / identity overlay, F3 toggle, debug-build gate, auto-discovery |
| Web/WASM GDExtension (no COOP/COEP) | Single-threaded ncnn WASM; proven in-browser on itch.io / GitHub Pages unmodified |
| Continuous DiagGaussian action sampling (game-side) | `action_dist_stats_path` + log_std sidecar → `mean + std·N(0,1)` without Python at inference |

---

## Prioritised gap summary

| Priority | Gap | Issue |
|---|---|---|
| ✅ Done | `policy_name` + `agent_policy_names` wire field — unblocks RLlib & PettingZoo | — |
| ✅ Done | `GridSensor2D/3D` — last major sensor type | — |
| ✅ Done | `ISensor2D/3D` interface + `collect_sensors()` | — |
| ✅ Done | `get_obs_space()` on agents — already implemented | — |
| ✅ Done | `INHERIT_FROM_SYNC` — already wired in `NcnnSync._get_agents()` | — |
| ✅ Done | `RaycastSensor3D` (and 2D) multi-class detection mode (`class_sensor`) | #42 |
| ✅ Done | `RECORD_EXPERT_DEMOS` + demo infra — `gnrl_v1`/`godot_rl` formats, Python loader + `train_bc.py`, chase scripted-expert | #13 |
| ✅ Done | Recurrent / LSTM **deploy** (hidden-state carry; training/export still pending) | #33 |
| ✅ Done | `SBGSingleObsEnv` + SB3 SAC continuous training — BallChase example, live-trained non-PPO regression | #74 |
| ✅ Done | Multi-policy trained example (custom single-file PPO, seeker+hider, item 45) | #26 partial |
| ✅ Done | Asset Library release + web/WASM GDExtension — prebuilt binaries on all platforms, EditorExportPlugin auto-packs models | #32 |
| ✅ Done | `ObsHistoryBuffer` (frame-stacking) + `RunningNormSensor` (online Welford) | #17, #18 |
| ✅ Done | In-editor Policy Debugger — live obs/action-probs overlay, F3 toggle | #23 |
| ✅ Done | Continuous DiagGaussian action sampling via log_std sidecar | #64 |
| ✅ Done | SampleFactory backend (godot_rl wrapper, `SampleFactoryEnvWrapper`) | #24 |
| ✅ Done | Batched multi-agent inference — `run_inference_batch` (thread-parallel, one shared `Net`) + `CrowdController` + `chase_crowd` example | #34 |
| ✅ Done | PettingZoo `ParallelEnv` interop — `GodotParallelEnv` adapter + `parallel_api_test` conformance; live training run is a follow-up | #111 |
| 🟡 P3 | RLlib `RayVectorGodotEnv` training-script interop | #110 |
| ⚪ P4 | Plugin editor-DX parity: pre-built sensor `.tscn` scenes + `script_templates/AIController` | #112 |
| ⚪ P5 | Optuna hyperparameter-tuning example (nice-to-have) | #113 |
| ⚪ P4 | CameraSensor: configurable render res + downscale + RGBA | #36 |
| ⚪ P4 | Grayscale camera deploy (C++ `PIXEL_GRAY` path) | #36 |
| 🔴 P5 | `terminated`/`truncated` split — wire semantics change; blocked on upstream godot_rl TODO | #12 |
| 🔵 By design | `ONNX_INFERENCE` mode — replaced by ncnn | — |
