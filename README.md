# Godot Native RL (ncnn GDExtension)

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
- [Running the examples](docs/guide/running-examples.md) — chase / rover / hide & seek
- [Training your own AI](docs/guide/training.md) — setup, train, the parallel-training fast path
- [Deploying](docs/guide/deploying.md) — NcnnRunner, INT8, VecNormalize, platform targets
- [Sensors](docs/guide/sensors.md) — raycast, relative-position, camera, grid
- [Building an agent in your scene](docs/guide/building-your-agent.md)

## What you get
- `NcnnRunner` C++ node: `load_model`, `run_inference`, `run_inference_image`,
  `run_discrete_action`.
- `NcnnAIController2D` / `NcnnAIController3D` + auto-discovered sensors + a Signal→Reward builder.
- godot_rl v0.8.2-compatible training bridge (`NcnnSync`) incl. multi-policy + parallel arenas.
- Convert (`scripts/export_to_ncnn.py`) and INT8 quantize for deployment.

## The moat
ncnn statically linked enables web/WASM and console deployment (ONNX/.NET can't), game-side INT8
quantization, async inference, and Godot-native ideas (Signal→Reward, NavMesh sensor) — none
replicable by a Python-server or managed-runtime framework.

## Contributing / building from source
Building the GDExtension, architecture, and dev notes:
[CONTRIBUTING.md](CONTRIBUTING.md) → [docs/dev/](docs/dev/).

## License
See repository license.
