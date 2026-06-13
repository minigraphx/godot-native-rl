# Godot Native RL (ncnn GDExtension)

[![CI](https://github.com/minigraphx/godot-native-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/minigraphx/godot-native-rl/actions/workflows/ci.yml)

Reinforcement learning for **Godot 4.5+** with **native ncnn inference** — statically linked C++,
no C#/.NET, no external runtime. Train with the standard `godot-rl` Python stack; deploy native on
web/WASM, console, mobile, desktop, and edge.

> **ncnn vs ONNX Runtime?** Honest decision guide:
> [docs/ncnn_vs_onnx.md](docs/ncnn_vs_onnx.md).

## Quick start (game developers)

1. **Install** — get the extension and enable the plugin:
   [docs/guide/getting-started.md](docs/guide/getting-started.md).
2. **Run an example** — pre-trained models, no Python needed:
   [docs/guide/running-examples.md](docs/guide/running-examples.md).
3. **Train your own AI** — `./scripts/setup_training.sh` then train → convert → deploy:
   [docs/guide/training.md](docs/guide/training.md).

## Guides
- [Getting started](docs/guide/getting-started.md) — install + enable the plugin
- [Running the examples](docs/guide/running-examples.md) — chase / rover / hide & seek / ball chase
- [Training your own AI](docs/guide/training.md) — setup, train, the parallel-training fast path
- [Deploying](docs/guide/deploying.md) — NcnnRunner, INT8, VecNormalize, continuous action sampling, platform targets
- [Sensors](docs/guide/sensors.md) — raycast, relative-position, camera, grid, navmesh
- [Building an agent in your scene](docs/guide/building-your-agent.md)

## Examples
- `examples/chase_the_target` — 2D discrete-action agent, trained with SB3 PPO
- `examples/rover_3d` — runnable 3D discrete-action rover with native inference, trained with SB3 PPO
- `examples/hide_and_seek` — 2D 1v1 self-play with a persistent trained two-policy demo
- `examples/ball_chase` — runnable 2D continuous-action SAC agent with native inference (`./scripts/train_ball_chase.sh`); exports the deterministic actor via TorchScript → ncnn; `SCENE=res://examples/ball_chase/ball_chase_train_parallel.tscn` tiles 8 worlds (`ParallelArena2D`) for ~3.4× measured training throughput
- `examples/fly_by` — runnable 3D continuous-action plane (PPO); ships a trained ncnn net + a `fly_by_action_dist.json` std sidecar for deploy-side DiagGaussian sampling (`./scripts/train_fly_by.sh`)
- `examples/quadruped_walk` — 3D continuous-control **locomotion**: a code-built articulated quadruped (8 hinge-joint motors, Jolt physics) trained with PPO (`./scripts/train_quadruped.sh`). Ships a trained ncnn net that **walks ~21 m straight toward the finish** (sustained ~1.1 m/s), deployed in `quadruped_walk_track.tscn` (camera + distance HUD), plus a learning-stage spread under `models/stages/` (500k/2.5M/6M steps) so you can watch the creature progress from flailing to walking. Behavioral forward-distance + golden-inference regressions ([#60](https://github.com/minigraphx/godot-native-rl/issues/60)). **M2 — run + jump hurdles**: 6 forward hurdle-closeness rays (`RaycastSensor3D` on the hurdle collision layer), a clear-the-hurdle bonus, and a game-side 3-stage curriculum (flat → low → race spacing, per-world `CurriculumController`); the shipped race-stage net **runs ~31 m and clears 4 of 6 hurdles** (`quadruped_hurdles_track.tscn`; `OUT=models/quadruped_hurdles SCENE=res://examples/quadruped_walk/quadruped_hurdles_train_parallel.tscn ./scripts/train_quadruped.sh`). **M3 — multiple morphologies**: a 6-leg **hexapod** (the more-stable 'many-legged' body) reusing the same game+agent generalized to be leg-count-agnostic; the quadruped's locomotion reward transfers unchanged and the trained hexapod **walks ~21 m at ~1.0 m/s** (`hexapod_walk_track.tscn`; `OUT=models/hexapod_walk SCENE=res://examples/quadruped_walk/hexapod_walk_train_parallel.tscn ./scripts/train_quadruped.sh`). **M4 — the generation race**: `quadruped_race.tscn` runs the committed 500k / 2.5M / 6M training *generations* (one creature, model-swapped between runs in clean solo physics) and ranks them on a leaderboard — the learning arc as a race (500k ~12 m, 2.5M ~21 m, 6M ~26 m). No training run: it composes the committed checkpoint spread. (Sequential, not side-by-side: multiple articulated ragdolls in one Jolt space contend for the solver and all gaits collapse — see `docs/dev/gotchas.md`)
- `examples/3dball` — **Unity 3DBall parity**: balance a ball on a tilting platform (2 continuous tilt actions, 8-dim Unity-matching obs, Jolt). Ships a trained ncnn net that balances indefinitely — 1800-frame eval, zero falls (`./scripts/train_ball_balance.sh`) ([#47](https://github.com/minigraphx/godot-native-rl/issues/47))
- `examples/gridworld` — **Unity GridWorld parity** + the `GridSensor2D` worked example: navigate an 8×8 grid to the goal, avoid pits (5 discrete actions, 5×5×2-layer grid-sensor obs + goal vector) (`./scripts/train_gridworld.sh`) ([#48](https://github.com/minigraphx/godot-native-rl/issues/48))
- `examples/visual_chase` — the chase task observed through **pixels only**: a code-rasterized 36×36×3 frame on the `camera_2d` wire key feeds SB3's CNN (NatureCNN) for training — fully headless, no rendering — and the trained net deploys through the native **image route** (`get_inference_image()` → `NcnnRunner.run_inference_image`). Ships a trained ncnn CNN + a portable golden-inference regression (fixed frames → correct, decisive argmax, verified identically on x86 and ARM) + a live integration smoke. Note: a discrete CNN policy's full-trajectory *catch count* isn't portable across architectures (ncnn runs convs in fp16 on ARM, fp32 on x86; ~3-magnitude logit drift flips the occasional argmax) — locally it catches 9–11/3600 frames, but CI gates per-frame correctness, not the trajectory (`./scripts/train_visual_chase.sh`, [#35](https://github.com/minigraphx/godot-native-rl/issues/35))
- `examples/coop_collect` — **cooperative multi-agent** (MA-POCA, #30): a shared-team-reward "collect" task where a centralized attention critic + per-agent counterfactual baseline assign credit for the team reward. Ships a trained 1.5M-step actor that collects **4/4 items cooperatively** under ncnn; trained with `./scripts/train_coop_mapoca.sh` (decentralized shared actor exported; critic discarded at deploy) ([#30](https://github.com/minigraphx/godot-native-rl/issues/30))
- `examples/chase_the_target/chase_crowd.tscn` — batched shared-policy crowd: many chasers driven by **one** shared net in a single `run_inference_batch` call per frame (reuses the committed chase net)

## Batched / crowd inference
For crowds of shared-policy agents, `NcnnRunner.run_inference_batch(inputs, num_threads)` runs all N
agents' forward passes in one C++ call, fanned across CPU threads (serial fallback on WASM). ncnn has
no CPU batch dimension, so this doesn't cut FLOPs — the win is collapsing N GDScript↔C++ round-trips
into one, parallelizing the passes across cores, and sharing **one** loaded `Net`. The reusable
`NcnnCrowdController` node owns the shared runner, gathers `get_obs()` from its child agents, runs one
batch, decodes each via `ActionDecode`, and scatters `set_action()` back. See `examples/.../chase_crowd.tscn`.

## Level-of-Detail policy switching
`NcnnLODRunner` runs a cheap "reflex" net most frames and an accurate "deliberative" net only every
N frames (or on a significant state change) — exactly one inference per frame, so the expensive net's
cost is paid at ~1/N the rate. `decide(obs)` returns the action plus which tier ran; only viable
because we statically link two resident nets and switch them game-side at no runtime cost.

## What you get
- `NcnnRunner` C++ node: `load_model`, `run_inference`, `run_inference_image`,
  `run_discrete_action`, `run_inference_multi` (recurrent/LSTM state-carry), `run_inference_batch` (crowds).
- `NcnnAIController2D` / `NcnnAIController3D` + auto-discovered sensors + a Signal→Reward builder.
- Editor DX: drop-in sensor scenes (`addons/godot_native_rl/sensors/scenes/` — raycast 2D/3D +
  camera 2D/3D with a pre-wired `SubViewport`) and an "NCNN AI Controller" script template,
  auto-installed to your project's script-template folder (`res://script_templates/` by
  default) when the plugin is enabled.
- **Curriculum learning** (`training/curriculum_controller.gd`): staged environment difficulty with
  performance-gated promotion, decided **game-side** so it works with every training backend
  unchanged (stage visible to trainers via the per-agent `info` field); custom loops can override
  via an additive `curriculum` wire message. Demo:
  `SCENE=res://examples/chase_the_target/chase_the_target_train_curriculum.tscn ./scripts/train_chase.sh`.
- **Competitive self-play with native ghosts** (`training/self_play_manager.gd`): the frozen
  opponent is an ordinary `NCNN_INFERENCE` agent running **in-engine ncnn** — invisible to the
  trainer, so any stock single-policy backend trains against it. Opponent pool + ELO ledger,
  per-episode snapshot swapping (`reload_model`), alternating-role league via
  `./scripts/train_selfplay.sh` (Hide & Seek demo).
- **Episode replay** (`training/replay_recorder.gd` + `replay_player.gd`): drop a `ReplayRecorder`
  into any training scene to save per-episode trajectories (actions + rewards + an opt-in
  initial-state snapshot — zero agent changes), then replay them deterministically in Godot
  (`chase_replay.tscn`). Exact for kinematic seeded games; approximate for physics envs (Jolt is
  not cross-run deterministic). Foundation for record-to-video (#40).
- godot_rl v0.8.2-compatible training bridge (`NcnnSync`) incl. multi-policy + parallel arenas.
  Training backends: SB3 (`train_chase.sh`), CleanRL (`train_cleanrl.sh`), SampleFactory async PPO
  (`train_sf.sh`, isolated `.venv-sf`, exports via TorchScript→ncnn), Ray/RLlib new-API-stack PPO
  (`train_rllib.sh`, shares `.venv-train` — stock RLlib trains against an unmodified env over the
  godot_rl wire protocol, exports via TorchScript→ncnn). PettingZoo `ParallelEnv`
  interop via our own `GodotParallelEnv` adapter (`train_pettingzoo.sh`; conformance proven with
  PettingZoo's `parallel_api_test`).
- Convert (`scripts/export_to_ncnn.py`) and INT8 quantize for deployment.

## Policy Debugger
Drop a `PolicyDebugOverlay` node (`addons/godot_native_rl/debug/policy_debug_overlay.gd`) into any
scene running ncnn inference. With its `controllers` list left empty it auto-discovers your agents and
overlays live observations, action probabilities, the loaded policy/model, and any `get_debug_status()`
you expose. Press **F3** to toggle; in release builds it removes itself at startup unless you set
`debug_build_only = false`. Worked example: `examples/chase_the_target/chase_the_target_debug.tscn`.

That same debug scene also carries a **live policy switcher** (`chase_model_switcher.gd`): a dropdown
that hot-swaps the deployed `.ncnn` model at runtime via the controllers' `swap_model(param, bin)` —
same scene, same engine, a different model file, visibly different behaviour, no recompile and no
Python. It's the most direct way to show native inference is real and model-driven (great for the
web demo); pair it with the overlay to watch the obs/action-probabilities change as you swap.

## The moat
ncnn statically linked enables web/WASM and console deployment (ONNX/.NET can't), game-side INT8
quantization, async inference, LOD policy switching (`NcnnLODRunner`), and Godot-native ideas (Signal→Reward, `NavMeshSensor`, `AnimationPolicyAdapter`) — none
replicable by a Python-server or managed-runtime framework.

## Installation (use the addon — no build needed)

You don't need the C++/SCons/ncnn toolchain to *use* this framework — just the prebuilt addon.

- **Asset Library (in-editor):** open the **AssetLib** tab in Godot 4.5+, search
  "Godot Native RL", install. It drops `addons/godot_native_rl/` (with native binaries for
  macOS/Windows/Linux/Android/iOS/web) into your project.
- **Manual:** download `godot-native-rl-addon-<version>.zip` from
  [Releases](../../releases) and unzip at your project root. For the demo scenes, also grab
  `godot-native-rl-examples-<version>.zip` (drop it in alongside the addon).

Then enable the plugin in **Project → Project Settings → Plugins**.

Building from source is covered in [CONTRIBUTING.md](CONTRIBUTING.md) → [docs/dev/](docs/dev/).

## Compatibility

- **Godot:** 4.5+ (`compatibility_minimum = 4.5`); the test suite runs in CI on 4.5.2 and 4.6.3.
- **Platforms** — prebuilt binaries ship for all; runtime-verification status:

| Platform              | Toolchain   | Status                          |
|-----------------------|-------------|---------------------------------|
| Linux x86_64          | native GCC  | ✅ verified (CI smoke + tests)  |
| macOS arm64           | native      | ✅ verified                     |
| Web / WASM            | emscripten  | ✅ verified (in-browser)        |
| Windows x86_64        | zig         | ✅ verified (CI: Godot --headless loads NcnnRunner) |
| Android x86_64        | Android NDK | ✅ verified (CI: dlopen on a real emulator) |
| Android arm64         | Android NDK | 🧪 symbol-audited in CI; device runtime check pending |
| iOS arm64             | Xcode       | 🧪 symbol-audited in CI; device runtime check pending |

"🧪 symbol-audited" means CI statically proves the binary's symbols all resolve at load (Android
arm64: the NDK linker resolves every imported symbol against the runtime libs; iOS: the `.xcframework` slices
test-link against the iOS SDK) — the same #95 load-failure class the verified targets catch by
actually loading — but it hasn't yet been loaded on a physical device. Contributions running these
on real hardware are welcome.

## Contributing / building from source
Building the GDExtension, architecture, and dev notes:
[CONTRIBUTING.md](CONTRIBUTING.md) → [docs/dev/](docs/dev/).

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).

The prebuilt addon binaries statically link ncnn (BSD 3-Clause) and godot-cpp (MIT); their
notices are reproduced in
[addons/godot_native_rl/THIRD_PARTY_LICENSES.md](addons/godot_native_rl/THIRD_PARTY_LICENSES.md).
