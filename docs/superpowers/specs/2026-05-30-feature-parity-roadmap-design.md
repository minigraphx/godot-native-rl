# Feature Parity Roadmap: Godot Native RL

**Date:** 2026-05-30
**Status:** Approved for decomposition

## 1. Vision & Positioning

**Focused superiority first, godot_rl parity second, Unity ML-Agents parity long-term.**

Godot Native RL is a GDExtension-based RL framework for Godot 4.6+ that uses Tencent's `ncnn` for native inference — statically linked, no C#/.NET, no external runtime. This gives it a deployment advantage that neither godot_rl (requires .NET for ONNX inference) nor Unity ML-Agents (Unity-only) can match:

- **Mobile (iOS/Android):** ARM + Vulkan, no runtime install
- **Web (HTML5/WASM):** ncnn compiles to WASM; godot_rl's ONNX inference path requires .NET which is blocked on web
- **Console (Switch/PS/Xbox):** Static linking, no .NET, small binary — required for cert compliance
- **Desktop + Edge:** Works everywhere Godot runs

**Strategy:** Start as a complement to godot_rl (train with their Python tooling, deploy with ncnn). Build toward full replacement as each feature gap closes. The complete replacement target is achieved when a user can do *everything* through this repo without installing godot_rl at all.

**Current state (updated 2026-06-03):**
- ✅ Full godot_rl wire-protocol compatibility (proven by real SB3 + CleanRL training)
- ✅ `NcnnSync` + `NcnnAIController2D`/`3D` + `NcnnRunner` GDExtension
- ✅ End-to-end Chase The Target 2D + Rover 3D examples (train → pnnx → ncnn → inference) +
  Hide & Seek 2D self-play scaffold
- ✅ Headless test suite (unit + protocol + inference smoke + trained-chase/rover checks + golden regressions)
- ✅ Conversion verification (`verify_ncnn_parity.py`: argmax + atol + diversity); INT8 + obs-norm + TorchScript paths
- ✅ **Full sensor parity** — Raycast/Grid/Camera/RelativePosition (2D+3D) all shipped (was the
  biggest switching friction; now closed, and RelativePosition puts us *ahead* of godot_rl)
- ⚠️ Addon structure (`addons/godot_native_rl/` + `plugin.cfg`) in place; Asset Library **binary
  packaging/release** still deferred (item 25)
- ⚠️ Two training backends (SB3 + CleanRL; SampleFactory/SKRL/RLlib queued)

---

## 2. Gap Analysis

> **Status refreshed 2026-06-03.** The `Ours` columns below reflect shipped state as of this date;
> see `docs/BACKLOG.md` for per-item detail. Original analysis was 2026-05-30.

### 2A. vs godot_rl + godot_rl_agents

#### Sensors (was the most critical gap — now full parity)
| Sensor | godot_rl | Ours |
|---|---|---|
| `RaycastSensor2D` | ✅ (n_rays, ray_length, collision mask) | ✅ (item 3) |
| `RaycastSensor3D` | ✅ (n_rays_width × height grid) | ✅ (item 3) — *distance-only; per-class one-hot queued (item 41)* |
| `GridSensor2D` | ✅ (cell-based spatial detection) | ✅ (item 11) |
| `GridSensor3D` | ✅ | ✅ (item 11) |
| `RGBCameraSensor3D` | ✅ (SubViewport → obs) | ✅ `CameraSensor` (item 8) + deploy-side image inference (item 36) |
| `RelativePositionSensor` | ❌ (open issue #177) | ✅ (item 7) — **ahead of godot_rl**; multi-target queued (item 42) |
| `ISensor2D`/`ISensor3D` interface + auto-discovery | ✅ | ✅ (item 40, `collect_sensors()`) |

#### Multi-agent & policies
| Feature | godot_rl | Ours |
|---|---|---|
| Parallel multi-agent training (tiled worlds) | ✅ | ✅ `ParallelArena`/`ParallelArena2D` (item 30, ~6.2×) |
| Parameter-sharing self-play example | ⚠️ | ✅ Hide & Seek 2D scaffold (item 12) |
| Per-agent `policy_name` in `env_info` | ✅ | ✅ (item 20 wire-field slice, done 2026-06-03) |
| PettingZoo wrapper | ✅ | ❌ (`agent_policy_names` now shipped — trained example queued, item 45) |
| Multi-policy training (RLlib) | ✅ | ❌ (`agent_policy_names` now shipped — trained example queued, item 45) |
| Expert-demo recording (RECORD_EXPERT_DEMOS) | ✅ | ❌ (queued — item 10) |
| Imitation learning (BC + GAIL, `imitation` lib) | ✅ | ❌ (follows expert-demo recording) |

#### Training backends
| Backend | godot_rl | Ours |
|---|---|---|
| Stable-Baselines3 | ✅ | ✅ |
| CleanRL | ✅ | ✅ (item 17, trained model shipped) |
| SampleFactory | ✅ | ❌ (queued — item 18) |
| RLlib | ✅ | ❌ (queued — item 45; multi-policy `policy_name` field now shipped) |
| SKRL | ❌ | ❌ (queued — item 19) |

#### Distribution & DX
| Feature | godot_rl | Ours |
|---|---|---|
| Godot Asset Library plugin | ✅ | ⚠️ addon structure + `plugin.cfg` done; binary release deferred (item 25) |
| `gdrl` CLI launcher | ✅ | ❌ |
| Hugging Face Hub integration | ✅ | ❌ (Later — item 20) |
| HP tuning (Optuna) | ✅ | ❌ |
| Troubleshooting docs | ✅ (extensive) | ⚠️ strong dev-facing gotchas (CLAUDE.md, `DEVELOPMENT.md`, `ncnn_vs_onnx.md`); no user-facing troubleshooting guide yet |

### 2B. vs Unity ML-Agents (stretch goals)

| Feature | Unity | godot_rl | Ours |
|---|---|---|---|
| SAC (off-policy) | ✅ | ❌ | ❌ |
| Curiosity intrinsic reward | ✅ | ❌ | ❌ |
| RND intrinsic reward | ✅ | ❌ | ❌ |
| Competitive self-play | ✅ | ❌ | ⚠️ (parameter-sharing self-play example, item 12; no opponent-snapshot pool) |
| Cooperative MA-POCA | ✅ | ❌ | ❌ |
| Curriculum learning | ✅ | ❌ | ❌ |
| Environment parameter randomization | ✅ | ❌ | ❌ |
| Variable-length obs (attention) | ✅ | ❌ | ❌ |
| RNN/LSTM memory | ✅ | ⚠️ (CleanRL GRU) | ❌ (deploy-side queued — item 22) |
| Physics body sensor (articulation joints) | ✅ | ❌ | ❌ |
| Buffer sensor (variable-count entities) | ✅ | ❌ | ❌ |
| Side channels (config/stats/float/string) | ✅ | ❌ | ❌ |
| Behavior snapshots for self-play | ✅ | ❌ | ❌ |

---

## 3. Four Independent Tracks

Each track produces independently shippable software. They can run in parallel. Each sub-project within a track gets its own spec → plan → implementation cycle.

### Track A — Sensor Library *(highest godot_rl switching friction)*

The single most important track for usability parity. Every non-trivial godot_rl example uses raycasts. Without sensors, users cannot replicate godot_rl's showcase environments.

**Shared sensor interface** (all sensors implement):
```gdscript
func get_observation() -> Array  # flat float array
func obs_size() -> int           # declared size (for get_obs_space)
```

Priority order:
1. **`RaycastSensor2D`** — most-used, highest switching friction. Configurable `n_rays`, `ray_length`, `collision_mask`, `collide_with_areas/bodies`. Emits 1 float per ray (normalized hit distance, or 0 if no hit). Ships with headless unit tests.
2. **`RaycastSensor3D`** — grid of rays (`n_rays_width × n_rays_height`), same interface.
3. **`GridSensor2D` / `GridSensor3D`** — spatial cell detection, configurable `cell_width/height`, `grid_size_x/y`.
4. **`CameraSensor`** (already queued) — SubViewport → resized Image → float array.
5. **`RelativePositionSensor`** (already queued) — direction + normalized distance, unit-tested.
6. **`PhysicsBodySensor`** (Unity parity stretch) — joint angles/velocities for articulated bodies.
7. **`BufferSensor`** (Unity parity stretch) — variable-length entity observations with attention.

**Addon structure:** sensors live in `addons/godot_native_rl/sensors/`. The sensor library is the first thing that requires the proper addon layout (Track D dependency).

**Our deployment advantage here:** All sensors deploy to mobile/web/console with zero runtime. godot_rl's camera/ONNX inference path requires .NET. Our sensors feed `NcnnRunner` which works everywhere.

---

### Track B — Multi-Agent & Policy Architecture

Sub-projects in dependency order:
1. **`NcnnAIController3D` + base refactor** (already queued) — prerequisite for all 3D multi-agent work.
2. **Per-agent `policy_name`** — add to `env_info`, route through `NcnnSync`. Enables godot_rl-protocol multi-policy (RLlib). *(✅ done 2026-06-03 — `agent_policy_names` wire field, item 20 slice; the trained RLlib/PettingZoo example is item 45.)*
3. **PettingZoo wrapper** — Python-side parallel-env API. Enables MARL with any PettingZoo-compatible trainer.
4. **Expert-demo recording (RECORD_EXPERT_DEMOS)** (already queued) — prerequisite for imitation learning.
5. **Competitive self-play** — ghost controller (frozen policy snapshot), league training, ELO tracking.
6. **MA-POCA cooperative** — shared reward, centralized critic.

---

### Track C — Training Algorithms & Reward Signals

All Python-side. Independent of Godot sensor/controller work.
1. **SAC training script** — SB3 already has SAC; one new `train_chase_sac.py`. 5–10× more sample-efficient than PPO for heavier envs. Quick win.
2. **Curiosity / ICM** (intrinsic reward) — critical for sparse-reward games (most real games). Pluggable reward signal added to any training script.
3. **RND** (Random Network Distillation) — alternative intrinsic reward, simpler than ICM.
4. **Imitation learning (BC + GAIL)** — depends on expert-demo recording (Track B).
5. **Curriculum learning** — progressive difficulty via environment parameter randomization. Requires side-channels or command-line parameterization.
6. **HP tuning (Optuna)** — copy/adapt godot_rl's existing `stable_baselines3_hp_tuning.py`.
7. **CleanRL / SampleFactory / SKRL backends** (all already queued).

---

### Track D — Developer Experience & Distribution

Makes this a *product* rather than a library. Unlocks adoption.
1. **Godot Asset Library plugin** (`addons/godot_native_rl/`, `plugin.cfg`) — installable from the Godot editor. **Prerequisite for Track A sensors** (proper addon structure needed first).
2. **`gdrl`-equivalent CLI or documented training entry point** — right now, users run `./scripts/train_chase.sh`. A consistent `mlagents-learn`-style entry point reduces confusion.
3. **Hugging Face Hub** — push trained ncnn models, pull pretrained ones. One-command: `godot-ncnn push examples/chase_the_target/models/ my-org/chase-agent`.
4. **Comprehensive docs** — `CUSTOM_ENV.md`, `NODE_REFERENCE.md`, `TROUBLESHOOTING.md` equivalents. godot_rl has 10+ doc files; we have 2.
5. **TensorBoard integration** — already logging (logs/sb3); just needs documentation + potentially an editor plugin shortcut.

---

## 4. Recommended Implementation Order

### Phase 1: Sensor foundation + proper addon structure (removes #1 switching friction)

**1A. Addon structure** (Track D prerequisite — ~1 day)
Reorganize into `addons/godot_native_rl/`:
```
addons/godot_native_rl/
  plugin.cfg
  sensors/
  controllers/
  sync.gd
  ncnn_ai_controller.gd       ← base (part of 3D refactor)
  ncnn_ai_controller_2d.gd
  ncnn_ai_controller_3d.gd    ← 3D refactor
```

**1B. RaycastSensor2D + RaycastSensor3D** (Track A priority 1 — ~1 week)
The sensors that every godot_rl example uses. With raycasts, users can replicate BallChase, Lander, DefendTheGoal.

**1C. 3D controller + navigate-to-target example** (already queued — parallel to 1B)

### Phase 2: Fills the remaining godot_rl sensor gap + SAC quick win

- GridSensor2D + GridSensor3D
- CameraSensor + RelativePositionSensor (already queued)
- SAC training script (Track C quick win — half a day)

### Phase 3: Multi-agent + imitation

- Expert-demo recording (already queued)
- PettingZoo wrapper + per-agent policy_name *(policy_name wire field ✅ done — item 20; wrapper/trained example is item 45)*
- Imitation learning (BC + GAIL)
- CleanRL + SampleFactory backends (already queued)

### Phase 4: DX + Unity-parity stretch

- Hugging Face Hub integration
- Curiosity / RND intrinsic reward
- Self-play (competitive)
- MA-POCA (cooperative)
- Curriculum learning

---

## 5. What Makes Us Uniquely Better (Superiority Dimensions)

These are the things we can do that godot_rl + Unity ML-Agents *cannot*, and should be emphasized in all documentation:

| Dimension | Our advantage |
|---|---|
| **Web deployment** | ncnn compiles to WASM; godot_rl ONNX/C# is blocked on web entirely |
| **Console deployment** | Static linking, no .NET cert issues |
| **Mobile performance** | ncnn's ARM + Vulkan backend, purpose-built for mobile inference |
| **No C# required** | GDScript-only workflow; no .NET SDK, no .csproj, no Mono |
| **Binary size** | ncnn is ~1MB statically linked; ONNX Runtime is tens of MB |
| **Conversion verification** | Our `verify_ncnn_parity.py` (argmax + atol + diversity) is more rigorous than godot_rl's manual export |
| **Offline inference** | No Python server needed at runtime; model runs entirely in C++ |

These should be the *first things* in every README, tutorial, and doc — not buried in technical sections.

---

## 6. Next Immediate Action

The first implementation cycle is **Phase 1A (addon structure) + 1B (RaycastSensor2D/3D)**. These should be brainstormed as separate sub-projects with their own specs:

1. `addon-structure` spec → plan → implementation (dependency for sensors)
2. `raycast-sensors-2d-3d` spec → plan → implementation

Each of these is independently shippable software with clear acceptance criteria.
