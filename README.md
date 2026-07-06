# Godot Native RL (ncnn GDExtension)

[![CI](https://github.com/minigraphx/godot-native-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/minigraphx/godot-native-rl/actions/workflows/ci.yml)

Reinforcement learning for **Godot 4.5+** with **native ncnn inference** — statically linked C++,
no C#/.NET, no external runtime. Train with the standard [`godot-rl`](https://github.com/edbeeching/godot_rl_agents)
Python stack; deploy native on web/WASM, console, mobile, desktop, and edge.

> **ncnn vs ONNX Runtime?** Honest decision guide:
> [docs/ncnn_vs_onnx.md](docs/ncnn_vs_onnx.md).

## Demo

[![Watch the demo](https://img.youtube.com/vi/Cud1gvbjg0I/hqdefault.jpg)](https://youtu.be/Cud1gvbjg0I)

▶ **[Watch the demo on YouTube](https://youtu.be/Cud1gvbjg0I)** — trained agents running on native ncnn, no Python at runtime.

🕹️ **[Play it in your browser](https://minigraphx.github.io/godot-native-rl/)** — the demo launcher as a
web build (single-threaded WASM, published from `main` by `.github/workflows/deploy-web-demo.yml`).
Pick **"Evolution Lab (train LIVE)"** and the browser tab *trains* a neural net in front of you —
no Python, no server, nothing but the page. (One-time repo setup: Settings → Pages → Source:
GitHub Actions.) Append **`?debug`** to the URL for an on-page dev console ([Eruda](https://github.com/liriliri/eruda),
loaded only when asked) that also captures errors from the very first boot instant — field-debug
the web build from any phone, no USB inspector needed.

| Chase the target | Quadruped (locomotion) | Rover (3D) |
|:---:|:---:|:---:|
| ![Chase the target](media/Catch.jpeg) | ![Quadruped locomotion](media/Quadruped-Debug.jpeg) | ![Rover 3D](media/Rover.jpg) |

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
- [Training in-engine (ES, no Python)](docs/guide/es-training.md) — ESTrainer, common random numbers, warm-start fine-tuning, when to use it
- [Deploying](docs/guide/deploying.md) — NcnnRunner, INT8, VecNormalize, continuous action sampling, platform targets
- [Sensors](docs/guide/sensors.md) — raycast, relative-position, camera, grid, navmesh
- [Building an agent in your scene](docs/guide/building-your-agent.md)

## Examples
- `examples/chase_the_target` — 2D discrete-action agent, trained with SB3 PPO
- `examples/rover_3d` — runnable 3D discrete-action rover with native inference, trained with SB3 PPO
- `examples/hide_and_seek` — 2D 1v1 self-play with a persistent trained two-policy demo
- `examples/seek_target` — the **`RelativePositionSensor2D` worked example** ([#38](https://github.com/minigraphx/godot-native-rl/issues/38)): reach a goal while dodging a patrolling hazard, where the agent's ENTIRE observation is one drop-in sensor observing both (goal + hazard slots, 6 floats — unit direction + distance per target, zero hand-coded obs — `collect_sensors()` auto-discovery). The shipped net is the first example trained **entirely in-engine** by `ESTrainer` with sep-CMA-ES: no Python, no export step, the checkpoint is the deploy artifact
- `examples/ball_chase` — runnable 2D continuous-action SAC agent with native inference (`./scripts/train_ball_chase.sh`); exports the deterministic actor via TorchScript → ncnn; `SCENE=res://examples/ball_chase/ball_chase_train_parallel.tscn` tiles 8 worlds (`ParallelArena2D`) for ~3.4× measured training throughput
- `examples/fly_by` — runnable 3D continuous-action plane (PPO); ships a trained ncnn net + a `fly_by_action_dist.json` std sidecar for deploy-side DiagGaussian sampling (`./scripts/train_fly_by.sh`)
- `examples/quadruped_walk` — 3D continuous-control **locomotion**: a code-built articulated quadruped (8 hinge-joint motors, Jolt physics) trained with PPO (`./scripts/train_quadruped.sh`). Ships a trained ncnn net that **walks ~21 m straight toward the finish** (sustained ~1.1 m/s), deployed in `quadruped_walk_track.tscn` (camera + distance HUD), plus a learning-stage spread under `models/stages/` (500k/2.5M/6M steps) so you can watch the creature progress from flailing to walking. Behavioral forward-distance + golden-inference regressions ([#60](https://github.com/minigraphx/godot-native-rl/issues/60)). **M2 — run the hurdles**: 6 forward hurdle-closeness rays (`RaycastSensor3D` on the hurdle collision layer), a clear-the-hurdle bonus, and a game-side 3-stage curriculum (flat → low → race spacing, per-world `CurriculumController`); the shipped race-stage net **runs the full ~40 m and clears all 6 hurdles in-lane**, rendered as low, ascending athletic-hurdle gates the trotting gait visibly clears (the hurdles are perception-only by design — the sensor reads them but the creature doesn't physically collide — so the gate visual is cosmetic, [#277](https://github.com/minigraphx/godot-native-rl/issues/277)) (`quadruped_hurdles_track.tscn`; `OUT=models/quadruped_hurdles SCENE=res://examples/quadruped_walk/quadruped_hurdles_train_parallel.tscn ./scripts/train_quadruped.sh`). **M3 — multiple morphologies**: a 6-leg **hexapod** (the more-stable 'many-legged' body) reusing the same game+agent generalized to be leg-count-agnostic; the quadruped's locomotion reward transfers unchanged and the trained hexapod **walks ~21 m at ~1.0 m/s** (`hexapod_walk_track.tscn`; `OUT=models/hexapod_walk SCENE=res://examples/quadruped_walk/hexapod_walk_train_parallel.tscn ./scripts/train_quadruped.sh`). **M4 — the generation race**: `quadruped_race.tscn` runs the committed 500k / 2.5M / 6M training *generations* (one creature, model-swapped between runs in clean solo physics) and ranks them on a leaderboard — the learning arc as a race (500k ~12 m, 2.5M ~21 m, 6M ~26 m). No training run: it composes the committed checkpoint spread. (Sequential, not side-by-side: multiple articulated ragdolls in one Jolt space contend for the solver and all gaits collapse — see `docs/dev/gotchas.md`)
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

## Train inside the engine (evolutionary strategies — no Python at all)
`ESTrainer` (`addons/godot_native_rl/training/es_trainer.gd`) is a native, in-engine training loop:
an evolutionary optimizer — OpenAI-style ES by default, or **sep-CMA-ES** (see the `optimizer`
switch below) — perturbs a flat weight vector, turns every candidate into a live ncnn
net in memory (`training/ncnn_weights.gd` θ⇄buffers codec + `load_model_from_buffers`), and scores
it by episodic return from the existing reward system. No Python, no socket, no backprop — it runs
on every deploy target the static ncnn build reaches, **including web**. The training artifact IS
the deploy artifact (checkpoints are ncnn `.param`/`.bin`), and the codec is bijective, so you can
warm-start from a shipped net and fine-tune on-device — including **pnnx-exported PPO actors**
(a structural adapter parses foreign MLP params and fp16-tagged bins; plain MLP policies only —
CNN policies like visual_chase are refused loud). Drop it into a scene in place of `NcnnSync`:
`godot --headless --path . res://examples/chase_the_target/chase_es_train.tscn` (single world), or
`chase_es_train_parallel.tscn` for 8 tiled worlds via `ParallelArena2D` — that run learns chase
from scratch in ~25 min (mean fitness −0.9 → 13.5 over 400 generations), and the resulting net
ships in `examples/chase_the_target/models/chase_es.ncnn.*` with a CI behavioral regression. Set
`optimizer = "cma_es"` (or cmdline `optimizer=cma_es`) to switch the update rule to **sep-CMA-ES**
— self-adapting step size + per-coordinate variances; in a paired 200-generation chase benchmark
(identical seeds/budget) it reached best generation mean **13.4 vs 2.3** for plain ES and passed
the plain-ES whole-run best by generation 22 (see `docs/guide/es-training.md`). The
`examples/seek_target` net was trained this way end-to-end. ES is
sample-inefficient — small nets and dense rewards, not a PPO/SAC replacement (issue #131).

**Watch it happen: the Evolution Lab demo** (launcher → "Evolution Lab (train LIVE)", #291) —
8 worlds train on screen while a highlighted **champion world** runs the best-so-far net through
the ordinary inference path, hot-swapped on every improvement: population evolving on the left,
today's best brain deployed on the right, zero export in between. HUD shows the live learning
curve; keys 1/2/3 set speed. First visible competence in ~5–8 minutes.

**On-device adaptation, measured honestly:** `chase_es_finetune_parallel.tscn` warm-starts the
shipped chase net (`warm_start_*_path`; ES-trained or pnnx-exported alike) against a **fleeing**
target it never trained on. The measured value is **time-to-competence**: the warm-started
population outscores 300 generations of from-scratch training *at generation 1* (mean fitness
3.9 vs 0.67 blessed after the full cold run; replicated at two difficulty settings). The equally
honest limit: same-architecture fine-tuning has ~no headroom when the environment shift demands
features the observations lack (no target velocity here → informed pursuit is already the
representable optimum) — details in the ES spec. The adapted net ships with a pipeline
regression (`trained_es_drift_scene.tscn`).

The adapter also adopts **pnnx-exported PPO nets** (`quadruped_es_finetune_parallel.tscn`
warm-starts the shipped quadruped walker, bit-for-bit). The measured limit there is physical:
**unseedable-physics envs have no common random numbers** (Jolt is cross-run nondeterministic),
so — adequately budgeted (λ=32, k=3, 8 tiled worlds) — ES **preserves** the warm-start (never
collapses) but **can't climb** it; an under-sized run collapses, a budget artifact rather than a
physics law. To *improve* a physics policy use the gradient backends (PPO/SAC); in-engine ES is
for kinematic/seedable envs. Corrected experiment + numbers in the ES spec.

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
It also covers **batched crowds**: `NcnnCrowdController` emits a per-unit `inference_step` through each
child agent, so the overlay shows one live block per crowd unit (see `chase_crowd.tscn`).

That same debug scene also carries a **live policy switcher** (`chase_model_switcher.gd`): a dropdown
that hot-swaps the deployed `.ncnn` model at runtime via the controllers' `swap_model(param, bin)` —
same scene, same engine, a different model file, visibly different behaviour, no recompile and no
Python. It's the most direct way to show native inference is real and model-driven (great for the
web demo); pair it with the overlay to watch the obs/action-probabilities change as you swap.

## Orbit camera (3D demos)
The 3D demos (quadruped walk/hurdles/race, hexapod, rover_3d, fly_by) carry a drop-in `OrbitCamera`
(`addons/godot_native_rl/camera/orbit_camera.gd`): press **C** to toggle a free orbit camera,
right-drag to rotate, scroll to zoom — inspect the gait/jumps from any angle. It defaults to the
fixed follow view and is cosmetic + inert headless (no input → no change), so training/CI are
unaffected.

## Resizable windows (launcher + demos)
The launcher and every demo run in a **resizable window** and scale to fit (project-wide
`canvas_items` stretch, base 1280×720) — drag any corner and the content stays crisp, no clipping
(#271/#272). The 2D demos carry a drop-in `FitCamera2D`
(`addons/godot_native_rl/camera/fit_camera_2d.gd`): it centers each demo's world and zoom-fits it to
the window, re-fitting on resize. Like the orbit camera it's cosmetic + inert headless and never
touches the simulation's `arena_size`, so observations and the trained nets are unaffected.

## The moat
ncnn statically linked enables web/WASM and console deployment (ONNX/.NET can't), game-side INT8
quantization, async inference, LOD policy switching (`NcnnLODRunner`), **in-engine ES training with
no Python** (`ESTrainer`), and Godot-native ideas (Signal→Reward, `NavMeshSensor`, `AnimationPolicyAdapter`) — none
replicable by a Python-server or managed-runtime framework.

## Installation (use the addon — no build needed)

You don't need the C++/SCons/ncnn toolchain to *use* this framework — just the prebuilt addon.

- **Asset Library (in-editor):** open the **AssetLib** tab in Godot 4.5+, search
  "Godot Native RL", install. It drops `addons/godot_native_rl/` (with native binaries for
  macOS/Windows/Linux/Android/iOS/web) into your project.
- **Manual:** download `godot-native-rl-addon-<version>.zip` from
  [Releases](../../releases) and unzip at your project root. For the demo scenes, also grab
  `godot-native-rl-examples-<version>.zip` (it ships a ready-to-run `project.godot` + a demo
  launcher — extract the addon into the same folder, open it, press F5).

> **macOS:** the prebuilt native library isn't Apple-notarized, so a browser download tags it
> `com.apple.quarantine` and Godot refuses to load it (every example silently falls back to "no
> inference" with a parse error on `NcnnAIController2D`). Clear the quarantine once after unzipping:
> ```
> xattr -dr com.apple.quarantine addons/godot_native_rl/bin
> ```
> (or right-click each `.dylib` → Open once). AssetLib installs aren't affected.

Then enable the plugin in **Project → Project Settings → Plugins**.

Building from source is covered in [CONTRIBUTING.md](CONTRIBUTING.md) → [docs/dev/](docs/dev/).

## Credits & relationship to godot_rl_agents

This project grew out of — and stays wire-compatible with —
**[godot_rl_agents](https://github.com/edbeeching/godot_rl_agents)** by Edward Beeching (the
`godot-rl` Python package). You train with its stock package and speak its training protocol — that
side is pure Python, **no .NET**. What we add is the **deployment** half: a native **ncnn** inference
path (statically-linked C++ — web/WASM, console, mobile, edge, INT8; no external runtime). It's a
*complement* to godot_rl_agents, not a hard fork of its code. Big thanks to Edward and the
godot_rl_agents contributors.

The piece we replace is specifically **in-Godot inference**: godot_rl_agents runs trained models
inside Godot via **ONNX on the Mono/.NET build of the editor** (its in-engine ONNX inference needs the
Godot .NET/Mono editor — see its [README](https://github.com/edbeeching/godot_rl_agents)).
Native ncnn is the **no-.NET** alternative for that step. Prefer ONNX anyway? You can export ONNX from
`godot-rl` and run it with **ONNX Runtime** off-engine (server / desktop / NVIDIA), or load it in
Godot with a **native C++ ONNX GDExtension** —
[joemarshall/godot_onnx_extension](https://github.com/joemarshall/godot_onnx_extension) or
[mat490/Godot-ONNX-AI-Models-Loaders](https://github.com/mat490/Godot-ONNX-AI-Models-Loaders) — to
skip .NET. Our honest [ncnn vs ONNX Runtime guide](docs/ncnn_vs_onnx.md) compares all of these.

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
