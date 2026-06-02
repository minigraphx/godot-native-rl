# godot_rl Ecosystem Gap Analysis

**Date:** 2026-06-02  
**Repos audited:** `edbeeching/godot_rl_agents` · `edbeeching/godot_rl_agents_plugin` · `edbeeching/godot_rl_agents_examples`  
**This repo state:** `main` @ `cf68fd2` (items 1–9 partial, 12–13, 17, 21, 24, 30, 33, 36 done)

---

## Sensors

| Sensor | Upstream plugin | This repo | Status |
|---|---|---|---|
| `RaycastSensor2D` | ✅ | ✅ | — |
| `RaycastSensor3D` (distance) | ✅ | ✅ | — |
| `RaycastSensor3D` class mode | ✅ `class_sensor` + `boolean_class_mask` — one-hot per class per ray | ❌ distance only | **Gap** (item 41) |
| `ISensor2D` / `ISensor3D` interface | ✅ shared base all sensors implement | ❌ no interface | **Gap** (item 40) |
| `PositionSensor2D/3D` | ✅ multi-target `Array[Node2D]`, optional dir/dist split | ✅ single `target_path` only | ⚠️ partial (item 42) |
| `RGBCameraSensor2D/3D` | ✅ configurable render res + downscale + RGBA/RGB + editor preview | ✅ fixed viewport res, RGB only, no downscale | ⚠️ partial (item 38) |
| `GridSensor2D` | ✅ area/body occupancy grid, multi-layer, debug view | ❌ | **Gap** (item 11) |
| `GridSensor3D` | ✅ | ❌ | **Gap** (item 11) |
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
| `INHERIT_FROM_SYNC` mode | ✅ per-agent can override scene-level default | ⚠️ enum value declared in controllers but not wired in `NcnnSync` | **Partial** (item 44) |
| `RECORD_EXPERT_DEMOS` mode | ✅ | ❌ | **Gap** (item 10) |
| `policy_name` export | ✅ default `"shared_policy"` | ❌ | **Gap** (item 20) |
| `get_obs_space()` method | ✅ required on every agent | ✅ implemented — delegates to `obs_space_from_obs()` | — (item 39 ✅) |
| `get_action()` for demo recording | ✅ required when recording | ❌ | **Gap** (item 10) |
| `expert_demo_save_path` export | ✅ | ❌ | **Gap** (item 10) |
| `remove_last_episode_key` binding | ✅ undo bad demonstration | ❌ | **Gap** (item 10) |
| Stochastic action sampling | ✅ `deterministic_inference` flag (softmax vs argmax) | ❌ always deterministic argmax | **Gap** (item 43) |
| VecNormalize obs replay | ❌ upstream | ✅ `obs_norm_stats_path` | **Advantage** |

---

## Sync node / wire protocol

| Feature | Upstream | This repo | Status |
|---|---|---|---|
| Training bridge (protocol v0.7) | ✅ | ✅ | — |
| `agent_policy_names` in env_info | ✅ | ❌ Python defaults gracefully (single-policy) | **Gap** (item 20) |
| `call()` remote method invocation | ✅ Python can invoke arbitrary Godot methods | ✅ handled in `NcnnSync` | — |
| `terminated`/`truncated` split | ❌ TODO both sides | ❌ | Parity (item 9) |
| Connect / read timeouts | ❌ | ✅ | **Advantage** |
| Per-agent `info` field | ❌ | ✅ | **Advantage** |
| `deterministic_inference` export on Sync | ✅ | ❌ (goes with item 43) | **Gap** |

---

## Python trainer wrappers

| Wrapper | Upstream | This repo | Status |
|---|---|---|---|
| `StableBaselinesGodotEnv` (SB3 VecEnv, n_parallel) | ✅ | ✅ proven | — |
| `SBGSingleObsEnv` (SB3 + `MlpPolicy` compat) | ✅ | ❌ | Minor |
| `CleanRLGodotEnv` | ✅ | ✅ item 17 done | — |
| `RayVectorGodotEnv` (RLlib) | ✅ | ❌ no training script | **Gap** (item 20) |
| `GDRLPettingZooEnv` (PettingZoo, multi-policy) | ✅ | ❌ | **Gap** (item 20, needs `policy_name`) |
| `SampleFactoryEnvWrapper` (batched + non-batched) | ✅ | ❌ | **Gap** (item 18) |
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
| Grayscale (1-channel) camera deploy | ❌ | ❌ needs C++ `PIXEL_GRAY` | **Gap** (item 38) |
| VecNormalize obs parity | ❌ | ✅ item 24 | **Advantage** |
| INT8 quantization | ❌ | ✅ item 13 | **Advantage** |
| TorchScript → ncnn export | ❌ | ✅ item 33 | **Advantage** |
| Recurrent / LSTM deploy | ❌ | ❌ | Parity gap (item 22) |
| Batched multi-agent inference | ❌ | ❌ | Parity gap (item 23) |

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

| Priority | Gap | Backlog item |
|---|---|---|
| 🔴 High | `GridSensor2D/3D` — last major sensor type | 11 |
| 🔴 High | `policy_name` + `agent_policy_names` — blocks RLlib & PettingZoo | 20 |
| ✅ Done | `get_obs_space()` on agents — already implemented | 39 |
| 🟡 Medium | `ISensor2D/3D` interface + `collect_sensors()` | 40 |
| 🟡 Medium | `RaycastSensor3D` multi-class detection mode | 41 |
| 🟡 Medium | `RelativePositionSensor` multi-target | 42 |
| 🟡 Medium | Stochastic action sampling (`deterministic_inference`) | 43 |
| 🟡 Medium | `INHERIT_FROM_SYNC` — enum exists, not wired in NcnnSync | 44 |
| 🟡 Medium | `RECORD_EXPERT_DEMOS` + demo infra | 10 |
| 🟡 Medium | CameraSensor: configurable render res + downscale + RGBA | 38 |
| 🟠 Lower | RLlib training script (after `policy_name` lands) | 20 |
| 🟠 Lower | SampleFactory backend | 18 |
| 🟠 Lower | Grayscale camera deploy (C++ `PIXEL_GRAY` path) | 38 |
| 🟠 Lower | `SBGSingleObsEnv` compat wrapper | — |
| 🔵 By design | `ONNX_INFERENCE` mode — replaced by ncnn | — |
