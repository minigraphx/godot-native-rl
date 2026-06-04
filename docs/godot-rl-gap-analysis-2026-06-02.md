# godot_rl Ecosystem Gap Analysis

**Date:** 2026-06-02 · **status refreshed 2026-06-03**  
**Repos audited:** `edbeeching/godot_rl_agents` · `edbeeching/godot_rl_agents_plugin` · `edbeeching/godot_rl_agents_examples`  
**This repo state:** backlog items 1–8, 11, 12–13, 17, 20 (wire-field slice), 21, 24, 30, 33, 36,
39, 40, 41, 44 done; item 9 partial. Open gaps tracked as GitHub issues (see table below).
(2026-06-03 refresh: GridSensor + ISensor interface shipped; `INHERIT_FROM_SYNC` already wired;
`policy_name`/`agent_policy_names` wire field shipped — RLlib/PettingZoo *trainers* now unblocked,
tracked as issue #26)

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
| Pre-built sensor `.tscn` scenes | ✅ RaycastSensor2D.tscn, RGBCameraSensor2D.tscn + examples | ❌ | Minor |
| `script_templates/AIController` | ✅ controller scaffold template in plugin | ❌ | Minor |

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
| `RECORD_EXPERT_DEMOS` mode | ✅ | ❌ | **Gap** (#13) |
| `policy_name` export | ✅ default `"shared_policy"` | ✅ default `"shared_policy"` on `NcnnAIController2D/3D` | ✅ done (item 20) |
| `get_obs_space()` method | ✅ required on every agent | ✅ implemented — delegates to `obs_space_from_obs()` | — (item 39 ✅) |
| `get_action()` for demo recording | ✅ required when recording | ❌ | **Gap** (#13) |
| `expert_demo_save_path` export | ✅ | ❌ | **Gap** (#13) |
| `remove_last_episode_key` binding | ✅ undo bad demonstration | ❌ | **Gap** (#13) |
| Stochastic action sampling | ✅ `deterministic_inference` flag (softmax vs argmax) | ✅ `deterministic_inference` + `inference_seed`; discrete softmax-sample (continuous follow-up #64) | ✅ done (#16) |
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
| `SBGSingleObsEnv` (SB3 + `MlpPolicy` compat) | ✅ | ❌ | Minor |
| `CleanRLGodotEnv` | ✅ | ✅ item 17 done | — |
| `RayVectorGodotEnv` (RLlib) | ✅ | ❌ no training script | **Gap** (#26 — `policy_name` now shipped) |
| `GDRLPettingZooEnv` (PettingZoo, multi-policy) | ✅ | ❌ | **Gap** (#26 — `policy_name` shipped; needs trainer/example) |
| `SampleFactoryEnvWrapper` (batched + non-batched) | ✅ | ❌ | **Gap** (#24) |
| ONNX export helper (`OnnxablePolicy`) | ✅ SB3/SAC → ONNX | ✅ `export_to_ncnn.py` ONNX+TorchScript→ncnn | Different, covered |
| Optuna HP tuning example | ✅ | ❌ | Nice-to-have |

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
| Recurrent / LSTM deploy | ❌ | ❌ | Parity gap (#33) |
| Batched multi-agent inference | ❌ | ❌ | Parity gap (#34) |

---

## Unique to this repo (not in upstream)

| Feature | Notes |
|---|---|
| Native ncnn inference (no .NET/C#) | Enables web, mobile, console, edge deploy |
| `ParallelArena` / `ParallelArena2D` | Scene-level agent tiling → ~Nx training speedup |
| INT8 quantization pipeline | `export_int8.py` + `build_ncnn_tools.sh` |
| VecNormalize obs replay game-side | `ObsNormalize` + `export_vecnormalize.py` |
| Continuous + multi-key action deploy | `action_decode.gd` + C++ fix |
| TorchScript → ncnn direct export | `--via torchscript` in `export_to_ncnn.py` |
| Socket connect/read timeouts | Clean exit on dead trainer |
| Per-agent `info` field | `get_info()` hook on controllers |
| `RewardBuilder` / `RewardAdapter` | More expressive than upstream's `ApproachNodeReward` |

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
| 🟠 P1 | `RECORD_EXPERT_DEMOS` + demo infra | #13 |
| 🟠 P1 | Recurrent / LSTM deploy | #33 |
| 🟡 P2 | RLlib + PettingZoo multi-policy trained example | #26 |
| 🟡 P2 | SampleFactory backend (godot_rl wrapper, `SampleFactoryEnvWrapper`) | #24 |
| 🔵 P3 | Batched multi-agent inference | #34 |
| ⚪ P4 | CameraSensor: configurable render res + downscale + RGBA | #36 |
| ⚪ P4 | Grayscale camera deploy (C++ `PIXEL_GRAY` path) | #36 |
| ⚪ P4 | `SBGSingleObsEnv` compat wrapper | — |
| 🔴 P5 | `terminated`/`truncated` split — wire semantics change; blocked on upstream godot_rl TODO | #12 |
| 🔵 By design | `ONNX_INFERENCE` mode — replaced by ncnn | — |
