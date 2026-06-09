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

## What you get
- `NcnnRunner` C++ node: `load_model`, `run_inference`, `run_inference_image`,
  `run_discrete_action`, `run_inference_multi` (recurrent/LSTM state-carry).
- `NcnnAIController2D` / `NcnnAIController3D` + auto-discovered sensors + a Signal→Reward builder.
- godot_rl v0.8.2-compatible training bridge (`NcnnSync`) incl. multi-policy + parallel arenas.
  Training backends: SB3 (`train_chase.sh`), CleanRL (`train_cleanrl.sh`), SampleFactory async PPO
  (`train_sf.sh`, isolated `.venv-sf`, exports via TorchScript→ncnn).
- Convert (`scripts/export_to_ncnn.py`) and INT8 quantize for deployment.

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
| Windows x86_64        | zig         | 🔨 builds; runtime check pending |
| Android arm64/x86_64  | Android NDK | 🔨 builds; runtime check pending |
| iOS arm64             | Xcode       | 🔨 builds; runtime check pending |

"🔨 builds" means the binary compiles and links in CI but hasn't yet been loaded on a real
device/runtime — contributions verifying these are welcome.

## Contributing / building from source
Building the GDExtension, architecture, and dev notes:
[CONTRIBUTING.md](CONTRIBUTING.md) → [docs/dev/](docs/dev/).

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).

The prebuilt addon binaries statically link ncnn (BSD 3-Clause) and godot-cpp (MIT); their
notices are reproduced in
[addons/godot_native_rl/THIRD_PARTY_LICENSES.md](addons/godot_native_rl/THIRD_PARTY_LICENSES.md).
