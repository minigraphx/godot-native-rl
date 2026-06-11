# BallChase Parallel/Tiled SAC Training Scene — Design

**Date:** 2026-06-11
**Issue:** #82 — Parallel/tiled BallChase SAC training scene (ParallelArena2D) for throughput
**Status:** Approved

## Problem

The 200k-step BallChase SAC run (#74 / PR #80) trains single-agent at ~59 fps (~55 min
wall-clock). The repo already vectorizes many worlds in one Godot process via
`ParallelArena` / `ParallelArena2D` (rover, hide & seek) for ~Nx samples/sec, with the trainer
unchanged. BallChase lacks a parallel scene.

## Constraints & facts established up front

- **Tile-safety is free.** BallChase uses only parent-local `position` — no physics bodies, no
  raycasts, no `global_position`. Worlds cannot interact even when overlapping; spacing matters
  only for visualization.
- **Structural gap:** unlike hide & seek (separate `hide_seek_world.tscn`),
  `ball_chase_train.tscn` is monolithic — game + agent + Sync in one scene. A world sub-scene
  must be extracted for `ParallelArena2D` to replicate.
- **SAC update-to-data ratio:** the trainer uses `train_freq=1, gradient_steps=1`. With 8 tiled
  agents each `env.step()` collects 8 transitions but does 1 gradient update — 8× fewer updates
  per sample, which can hurt final quality and eat the throughput win.
- `checkpoint_freq // env.num_envs` in the trainer already keeps checkpoints in total-timestep
  units under vectorization. Resume across env-count changes is safe because the replay buffer
  restarts empty anyway (CheckpointCallback saves only the policy).
- `scripts/throughput_compare.sh` is rover-hardcoded (scenes + `train_rover.py` +
  `--onnx_export_path`).

## Decisions (with user)

1. **Run scope: short measured compare.** Ship scene + scripts + tests now; run a short
   (~20–30k step) parallel-vs-single comparison to measure speedup and confirm a learning
   policy. The existing committed ball_chase fixture stays; a full 200k parallel retrain is
   optional later.
2. **SAC updates: `gradient_steps=-1`.** One-line trainer change so SB3 does as many gradient
   updates as transitions collected per step (8 with 8 envs, 1 with 1 env — identical behavior
   for the single-world scene, fully backward compatible). Keeps the update-to-data ratio at 1
   regardless of tiling.
3. **Approach: full mirror + parameterized throughput script** (over a one-off manual
   measurement, and over duplicating nodes without a world extraction).

## Design

### Scenes

- **`examples/ball_chase/ball_chase_world.tscn`** (new) — `BallChaseGame` (with
  `AgentBody`/`Target` children) + `BallChaseAgent` (`control_mode = 2`), **no Sync**. The
  replicable unit.
- **`examples/ball_chase/ball_chase_train.tscn`** (refactored) — instance of
  `ball_chase_world.tscn` + `Sync` (`control_mode = 1`), mirroring how
  `hide_and_seek_train.tscn` composes `hide_seek_world.tscn`. Wire-identical to today
  (1 agent, same obs/action spaces), so the committed trained model and golden regression are
  untouched.
- **`examples/ball_chase/ball_chase_train_parallel.tscn`** (new) — `ParallelArena2D`
  (`world_scene = ball_chase_world.tscn`, `count = 8`, `spacing = 1400.0`) + `Sync`
  (`control_mode = 1`), mirroring `hide_and_seek_train_parallel.tscn`.

### Trainer (`scripts/train_ball_chase.py`)

- `gradient_steps=1` → `gradient_steps=-1`, with a comment explaining the update-to-data ratio
  under tiling. No other trainer changes; godot-rl auto-detects n_agents and vectorizes.

### Throughput script (`scripts/throughput_compare.sh`)

- Add env-var overrides with rover defaults (existing bare invocation unchanged):
  - `SINGLE_SCENE` / `PARALLEL_SCENE` — scenes to compare
  - `TRAINER` — trainer script (default `scripts/train_rover.py`)
  - `EXPORT_ARG` — the trainer's export-path flag name (default `--onnx_export_path`;
    BallChase uses `--pt_export_path`)
- BallChase compare becomes one documented command.

### Tests (headless, wired into `run_tests.sh`)

- A BallChase parallel-arena smoke in the style of `parallel_arena_smoke_checker.gd`: asserts
  the arena spawned exactly N agents, each produces a finite 5-dim obs, spawned worlds sit at
  distinct tile origins ≥ spacing apart, and random **continuous** actions drive frames without
  errors. The existing checker is rover-shaped (discrete `action_count`); the implementation
  plan decides between parameterizing it and adding a small 2D-continuous sibling, based on how
  much actually shares.
- The existing `trained_ball_chase_scene.tscn` regression keeps guarding the deploy path, but it
  duplicates its own game/agent nodes and does NOT instance the train scene — so a new
  scene-structure unit test (instantiate without entering the tree; assert world/train/parallel
  composition and exported properties) guards the refactor instead.

### Measurement (acceptance evidence)

- Short fresh compare (~20–30k steps) via the generalized script:
  - parallel beats single on samples/sec (script already exits non-zero otherwise), and
  - the parallel run's episode-reward trend is positive (learning, not just fast).
- Record numbers in issue #82 and a line in the BallChase docs/README.
- Full 200k parallel retrain + fixture refresh: explicitly out of scope (existing fixture
  stays; can be a follow-up if wanted).

## Error handling

- `ParallelArena2D` already pushes errors on missing `world_scene` / `count < 1`.
- The smoke checker fails loud (non-zero exit) on agent-count, obs-shape, non-finite-obs, or
  tiling violations.
- `throughput_compare.sh` keeps `set -euo pipefail`, fails on trainer non-zero exit, and exits
  non-zero when parallel is not faster.

## Out of scope

- Trainer code changes beyond `gradient_steps=-1` (no `n_parallel` > 1 multi-process work).
- Replacing the committed BallChase model/golden fixtures.
- Any change to the deploy/inference path.

## Docs to update in the same change

- `CLAUDE.md` (BallChase train command gains the `SCENE=` parallel override; backlog state)
- `README` / example docs (parallel scene + measured speedup note)
- Close #82 via the PR (`Closes #82`). No `docs/BACKLOG.md` change: #82 is a GitHub-only item
  (BACKLOG.md tracks only the originally-listed items and is not extended).
