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
- [Sensors](docs/guide/sensors.md) — raycast, relative-position, camera, grid
- [Building an agent in your scene](docs/guide/building-your-agent.md)

## Examples
- `examples/chase_the_target` — 2D discrete-action agent, trained with SB3 PPO
- `examples/rover_3d` — runnable 3D discrete-action rover with native inference, trained with SB3 PPO
- `examples/hide_and_seek` — 2D 1v1 self-play with a persistent trained two-policy demo
- `examples/ball_chase` — runnable 2D continuous-action SAC agent with native inference (`./scripts/train_ball_chase.sh`); exports the deterministic actor via TorchScript → ncnn
- `examples/fly_by` — runnable 3D continuous-action plane (PPO); ships a trained ncnn net + a `fly_by_action_dist.json` std sidecar for deploy-side DiagGaussian sampling (`./scripts/train_fly_by.sh`)
- `examples/chase_the_target/chase_crowd.tscn` — batched shared-policy crowd: many chasers driven by **one** shared net in a single `run_inference_batch` call per frame (reuses the committed chase net)

## Batched / crowd inference
For crowds of shared-policy agents, `NcnnRunner.run_inference_batch(inputs, num_threads)` runs all N
agents' forward passes in one C++ call, fanned across CPU threads (serial fallback on WASM). ncnn has
no CPU batch dimension, so this doesn't cut FLOPs — the win is collapsing N GDScript↔C++ round-trips
into one, parallelizing the passes across cores, and sharing **one** loaded `Net`. The reusable
`NcnnCrowdController` node owns the shared runner, gathers `get_obs()` from its child agents, runs one
batch, decodes each via `ActionDecode`, and scatters `set_action()` back. See `examples/.../chase_crowd.tscn`.

## What you get
- `NcnnRunner` C++ node: `load_model`, `run_inference`, `run_inference_image`,
  `run_discrete_action`, `run_inference_multi` (recurrent/LSTM state-carry), `run_inference_batch` (crowds).
- `NcnnAIController2D` / `NcnnAIController3D` + auto-discovered sensors + a Signal→Reward builder.
- Editor DX: drop-in sensor scenes (`addons/godot_native_rl/sensors/scenes/` — raycast 2D/3D +
  camera 2D/3D with a pre-wired `SubViewport`) and an "NCNN AI Controller" script template,
  auto-installed to `res://script_templates/` when the plugin is enabled.
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

## The moat
ncnn statically linked enables web/WASM and console deployment (ONNX/.NET can't), game-side INT8
quantization, async inference, and Godot-native ideas (Signal→Reward, NavMesh sensor) — none
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
