# Training Your Own AI

You train with the standard `godot-rl` Python stack and deploy with native ncnn.

## 1. Python setup (once)

Install **Python 3.13** (training) and **Python 3.14** (conversion), then:

```bash
./scripts/setup_training.sh
```

This creates `.venv-train` (godot-rl + verify deps) and `.venv` (pnnx + torch) from
`requirements-train.txt` / `requirements-convert.txt`. Re-running is safe (existing venvs are
reused). Override interpreters with `PYTHON_TRAIN=` / `PYTHON_CONVERT=`. Validate without
installing: `./scripts/setup_training.sh --check`.

**conda alternative:** create two envs and `pip install -r` the same files; see
[../dev/DEVELOPMENT.md](../dev/DEVELOPMENT.md) for the two-env rationale.

## 2. Train

Training always requires two processes running simultaneously: the **Godot environment** (the scene)
and the **Python trainer** (SB3/CleanRL). The training scripts start both for you.

The training scripts launch Godot headless and the Python trainer in one command:

```bash
# Chase the Target (2D)
TIMESTEPS=120000 ./scripts/train_chase.sh

# 3D Rover (checkpoint/resume-capable)
./scripts/train_rover.sh

# 3D Rover, parallel ×8 (~6× faster)
SCENE=res://examples/rover_3d/rover_3d_train_parallel.tscn ./scripts/train_rover.sh

# Hide & Seek self-play
./scripts/train_hide_seek.sh

# Chase via CleanRL backend
./scripts/train_cleanrl.sh
```

The rover trainer is **checkpoint/resume-capable**: it saves to `models/rover_checkpoints/` every
25k steps and **auto-resumes** from the latest checkpoint on re-run. `FRESH=1` starts from scratch;
`CHECKPOINT_FREQ=N` changes the interval; `TIMESTEPS=N` raises the target to refine an existing
model further.

The parallel rover scene (`rover_3d_train_parallel.tscn`) tiles 8 rover worlds in one Godot process
via `ParallelArena`, so godot-rl vectorizes over 8 agents (~Nx samples/sec). The Python trainer is
unchanged — it auto-detects `n_agents` from the handshake. Measured speedup: **6.2×** over the
single-agent scene.

> **macOS:** wrap long runs in `caffeinate -is` — sleep kills the training socket
> (see [../dev/gotchas.md](../dev/gotchas.md)).

## 3. Convert + deploy

After training produces a checkpoint/ONNX, convert to ncnn and deploy — see
[deploying.md](deploying.md).
