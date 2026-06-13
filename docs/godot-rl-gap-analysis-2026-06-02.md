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
(2026-06-13 refresh: ICM intrinsic reward shipped (item 51 / #201) — `--intrinsic icm` in
`train_cleanrl.py`, the phase-2 follow-up to #27's RND. A learned encoder + inverse model (predicts
the action, shaping the encoder toward controllable features) + forward model (predicts next
features; its error is the bonus); the rollout passes `(obs, action, next_obs)` into the signal since
ICM is transition-based, not state-only. Torch-guarded tests + guarded CleanRL+ICM CI smoke;
training-only, deploy unchanged. Completes the intrinsic-reward parity item (both RND and ICM).)
(2026-06-13 refresh: GAIL adversarial imitation shipped (#61, the GAIL half) — `scripts/gail.py`: a
discriminator D(obs, action)→P(expert), reward = softplus(D), trained adversarially against the
recorded expert demos (#10). Wired into `train_cleanrl.py` as `--imitation gail` — REPLACES the env
reward so the policy imitates with zero env reward. Pure helpers + torch-guarded discriminator tests +
guarded CleanRL+GAIL CI smoke. Shipped as a training *method* (like RND/#27 — no committed deploy net):
pure GAIL on the small chase demo set learns to chase but is sample-inefficient (4 catches in a 300k
run, not yet robust). AMP — adversarial *reference-motion* priors — is the deferred half; it needs
motion-clip data this repo doesn't record. #62 Eureka (LLM reward design) is the last batch item.)
(2026-06-13 refresh: #60 M4 the generation race shipped — `quadruped_race.tscn` runs the committed
500k/2.5M/6M training generations as a SEQUENTIAL race (one creature, model-swapped between runs in
clean solo physics) onto a leaderboard. The learning arc: 500k ~12 m, 2.5M ~21 m, 6M ~26 m —
CI-asserted (later generation out-distances earlier by >=8 m). No training run. Sequential, not
side-by-side: multiple articulated ragdolls in one Jolt space contend for the solver and all gaits
collapse (gotcha documented). #60 M5 (record-to-video, needs #40) is the last sub-milestone.)
(2026-06-13 refresh: #60 M3 multiple morphologies shipped — a 6-leg **hexapod** ('many-legged')
locomotion example. The quadruped's game + agent were generalized to be leg-count-agnostic
(`range(4)` → `range(_leg_count())`; agent derives motors=joint_count and obs=5·legs+9), so the
hexapod reuses ALL the obs/reward/locomotion logic — only `HexapodBuilder` differs. The quadruped's
v3 reward transfers unchanged: the trained 12-motor/39-obs net walks ~21 m at ~1.0 m/s. Behavioral
+ golden regressions; quadruped tests unchanged. #60 M4 race + M5 record-to-video remain.)
(2026-06-13 refresh: MA-POCA M3 posthumous credit shipped (item 54 M3 / #30) — an `early_finish`
'bank and leave' env variant (an agent banks out after contributing; a per-active-agent step penalty
makes collect-then-bank optimal) + `--early-finish` masking that drops banked agents' inert steps
from the actor loss while their pre-bank steps keep advantage from the team's later return — the
defining MA-POCA property. Trained 17-dim actor collects 3/4 items AND an agent banks. This completes
#30 (M1 env + M2 centralized critic + M3 posthumous credit) — full MA-POCA MARL parity.)
(2026-06-13 refresh: MA-POCA cooperative training shipped (item 54 M2 / #30) — single-file
`scripts/train_coop_mapoca.py`: shared decentralized actor + a centralized attention critic over the
team's obs + a per-agent leave-one-out counterfactual baseline, over coop_collect via CleanRLGodotEnv.
Pure credit/masking helpers unit-tested; world-major team grouping verified at runtime via the
shared-reward invariant; the trained actor collects 4/4 items cooperatively under ncnn. This closes
the last Unity-ML-Agents MARL-parity gap that had a trained deliverable (centralized-critic credit
assignment); M3 posthumous credit is the remaining stretch, masking helpers already in place.)
(2026-06-13 refresh: trained CNN visual example shipped (item 37 / #35) — `examples/visual_chase`,
pixels-only chase via a code-rasterized 36×36×3 `camera_2d` obs (godot_rl's "*2d"→uint8 mapping →
SB3 NatureCNN), trained fully headless, deployed through the item-36 image route
(`get_inference_image()` → `run_inference_image`, first trained consumer). TorchScript→ncnn conv
export: godot_rl's `export_model_as_onnx` KeyErrors on MultiInputPolicy under torch 2.x dynamo —
the SAC/#81 breakage class again, TorchScript remains our standard answer.)
(2026-06-09 audit: re-checked every gap row against open issues. The remaining godot_rl_agents
compatibility gaps that had **no open tracking issue** were filed into the `v0.2 — godot_rl complement`
milestone — RLlib `RayVectorGodotEnv` (#110), PettingZoo `GDRLPettingZooEnv` (#111), plugin editor-DX
parity / sensor `.tscn` scenes + `script_templates/AIController` (#112), Optuna HP-tuning example (#113).
The old `#26` references on the RLlib/PettingZoo rows were stale — #26 shipped a custom multi-policy PPO
and is closed; the stock-wrapper interop is now tracked separately.)
(2026-06-09 refresh: PettingZoo `ParallelEnv` interop shipped (#111) — `GodotParallelEnv` adapter in
`scripts/godot_pettingzoo_env.py` provides `GDRLPettingZooEnv` functionality without depending on the
upstream class; `train_pettingzoo.sh` drives multi-policy PPO; conformance proven via PettingZoo's
`parallel_api_test`. Live training run is a follow-up.)
(2026-06-10 refresh: Ray/RLlib backend shipped (#110) — stock RLlib PPO on the new API stack over the
godot_rl wire via a custom gymnasium adapter (`GodotRLlibEnv`; the stock `RayVectorGodotEnv` is
old-API-stack only), shares `.venv-train` (#126), RLModule actor → TorchScript → ncnn, guarded smoke +
committed golden-inference fixture.)
(2026-06-10 strategy note: **native-ONNX-in-godot_rl is a latent moat risk** — see "Strategic note" below.)

---

## Strategic note — native ONNX is interesting *for godot_rl* (moat risk)

godot_rl's inability to web-export is **not** an ONNX-Runtime limitation; it's a property of *how godot_rl
integrated ONNX* — its *stock* no-Python path runs through Godot **Mono/.NET**, and .NET can't web-export
([godot#70796](https://github.com/godotengine/godot/issues/70796)). ONNX Runtime itself has a native C/C++ core,
and a community **native ORT GDExtension already exists** ([`godot_onnx_extension`](https://github.com/joemarshall/godot_onnx_extension),
the subject of godot_rl issue [#249](https://github.com/edbeeching/godot_rl_agents/issues/249)): it drops .NET
and reaches desktop + Android **with no conversion step**, but **does not web-export today** — and it's an
**unmaintained POC** (last commit Feb 2024, no releases, godot-cpp pinned to a 2024-era commit). So right now
ncnn is still the only native path proven in the browser, and the risk is doubly latent: someone must *revive*
that extension **and** *add* a WASM target before godot_rl's biggest gap against this project closes — which is
why "native ONNX integration is interesting" really means "interesting *for godot_rl*."

Implications for positioning:
- **Web/WASM is our headline pillar but the most contestable one.** If godot_rl grows a native (non-.NET) ORT
  backend, "godot_rl literally can't reach the browser" stops being true. Lean the moat narrative on the
  pillars a native-ORT backend does **not** neutralise: console certification (no managed runtime / smaller
  audit surface), lean edge footprint (~3.4 MB static `.so` vs ORT-WASM's heavier payload, same brittle
  `wasm32` dlink pipeline either way), and game-side INT8 — all in the "Deploy-side inference" / "Unique to
  this repo" tables above.
- **We could ship it too.** The swappable inference seam (`docs/dev/DEVELOPMENT.md`, "The inference-backend
  boundary"; ExecuTorch tracked as #54) means a native-ORT runner drops in with no GDScript/decode/protocol
  changes. Doing so *as an upstream godot_rl contribution* fits the "complement first" strategy but narrows
  our own differentiation. This is a deliberate positioning call, **not** a queued implementation task.
- Neither runtime does **on-device learning** — both are forward-pass-only; training stays in Python. That's
  not a differentiator either way.

See `docs/ncnn_vs_onnx.md` §"Web / HTML5 deployment" for the same note in the deployment-decision framing.

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
| Pre-built sensor `.tscn` scenes | ✅ RaycastSensor2D.tscn, RGBCameraSensor2D.tscn + examples | ✅ `sensors/scenes/` — Raycast2D/3D + Camera2D/3D (pre-wired SubViewport) | ✅ done (#112) |
| `script_templates/AIController` | ✅ controller scaffold template in plugin | ✅ `NcnnAIController2D/3D` templates, auto-installed on plugin enable | ✅ done (#112) |

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
| `RayVectorGodotEnv` (RLlib) | ✅ | ✅ done (#110) — `train_rllib.sh`, new-API-stack PPO via a custom gymnasium adapter (the stock wrapper is old-API-stack only), TorchScript→ncnn, shares `.venv-train` (#126) | — |
| `GDRLPettingZooEnv` (PettingZoo, multi-policy) | ✅ | ✅ `GodotParallelEnv` in `scripts/godot_pettingzoo_env.py` — `GDRLPettingZooEnv` functionality without the upstream class; `parallel_api_test` conformance; live-trained two-policy fixtures + golden/LOS regression shipped (#118) | ✅ done (#111) |
| `SampleFactoryEnvWrapper` (batched + non-batched) | ✅ | ✅ done (#24) — `train_sf.sh`, async PPO, TorchScript→ncnn, isolated `.venv-sf` | — |
| ONNX export helper (`OnnxablePolicy`) | ✅ SB3/SAC → ONNX | ✅ `export_to_ncnn.py` ONNX+TorchScript→ncnn | Different, covered |
| Optuna HP tuning example | ✅ | ✅ done (#113) — `tune_optuna.sh`/`tune_optuna.py`: Optuna PPO study over an example, maximizes `ep_rew_mean`, one Godot client per trial, isolated `optuna` dep (`requirements-tune.txt`), pure helpers unit-tested | — |

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
| ✅ Done | **Unity 3DBall parity** — tilting-platform ball balance, trained net balances 1800 frames/0 falls (continuous starter) | #47 |
| ✅ Done | **Unity GridWorld parity** — grid navigation, the `GridSensor2D` worked example (discrete starter) | #48 |
| ✅ Done | **Competitive self-play** (Unity ML-Agents self-play parity, league-style): native-ncnn ghost opponents (invisible to the trainer), opponent pool + ELO ledger, alternating-role phases (`train_selfplay.sh`) | #29 |
| ✅ Done | **Curriculum learning** (Unity ML-Agents `environment_parameters` parity) — game-side staged difficulty (`CurriculumController`, all backends, zero protocol change) + additive `curriculum` wire override for custom loops; 3-stage chase demo | #28 |
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
| ✅ Done | PettingZoo `ParallelEnv` interop — `GodotParallelEnv` adapter + `parallel_api_test` conformance; live training run shipped (#118) | #111 |
| ✅ Done | RLlib training-script interop — new-API-stack PPO, custom gymnasium adapter (stock `RayVectorGodotEnv` is old-API-stack only), shares `.venv-train` (#126) | #110 |
| 🟡 M1 done | Continuous-control **locomotion** showcase (Unity Crawler/Walker territory): `quadruped_walk` — code-built 8-hinge-joint articulated quadruped on Jolt; M1 ships a trained PPO ncnn net that walks ~21m straight at ~1.1 m/s (forward-velocity + lateral-penalty reward) + learning-stage spread + behavioral/golden regressions. Epic ongoing (hurdles, race, morphologies, video = M2–M5) | #60 |
| ✅ Done | Plugin editor-DX parity: drop-in sensor scenes (`sensors/scenes/`) + `NcnnAIController` script templates auto-installed on enable | #112 |
| ✅ Done | Optuna hyperparameter-tuning example — `tune_optuna.sh`/`.py`, PPO study maximizing `ep_rew_mean`, per-trial Godot client, isolated `optuna` dep, unit-tested helpers | #113 |
| ⚪ P4 | CameraSensor: configurable render res + downscale + RGBA | #36 |
| ⚪ P4 | Grayscale camera deploy (C++ `PIXEL_GRAY` path) | #36 |
| 🔴 P5 | `terminated`/`truncated` split — wire semantics change; blocked on upstream godot_rl TODO | #12 |
| 🔵 By design | `ONNX_INFERENCE` mode — replaced by ncnn | — |
