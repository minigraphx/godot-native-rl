# Parallel Multi-Agent Training (ParallelArena) — Design

**Date:** 2026-05-31
**Backlog:** new item (A1 — training speedup); roadmap DX/throughput
**Status:** Approved (brainstorm → spec)

## Problem & motivation

Training currently runs **one** agent in one Godot process (`n_parallel=1`), bottlenecked by
environment throughput (~40–60 env-fps: physics + raycasts + per-step socket round-trip). The
PPO net is a tiny CPU MLP — negligible. The standard godot-rl speedup is **many agents per
process**: `NcnnSync` already collects every node in the `AGENT` group and batches obs/rewards/
actions, and godot-rl auto-detects `n_agents` from the handshake to vectorize PPO. So the only
missing piece is a **scene** that hosts N agent worlds, spatially isolated so each agent's
raycasts hit only its own obstacles. We build this as a **reusable** component so every example
benefits, not just the rover.

## Decisions (from brainstorming)

1. **Reusable component:** a `ParallelArena` node in the addon that instances N copies of *any*
   agent "world" sub-scene; plus the rover world extracted as a sub-scene and a parallel rover
   training scene using it.
2. **Isolation:** **spatial tiling** in one shared `World3D` — each world placed at a large grid
   offset (default 200 units) so rays (length 20, arena 40) never reach another tile. Lightweight
   (single physics space), idiomatic for godot-rl. (SubViewport-per-agent rejected as heavier.)
3. **Trainer unchanged:** godot-rl auto-detects `n_agents = count`; `train_rover.py` needs no
   change. `train_rover.sh` gets a `SCENE=` env override to target the parallel scene.
4. **JAX/NumPy + Gymnasium env twin** (training without Godot) is recorded as a separate backlog
   item for later (see below) — out of scope here.

## Architecture

### `addons/godot_native_rl/training/parallel_arena.gd` — `ParallelArena extends Node3D`

Exports:
- `world_scene: PackedScene` — the agent world to replicate (must contain exactly one `AGENT`-group
  agent and keep its game logic in its own local frame, i.e. tile-offset-safe).
- `count: int = 8` — number of parallel worlds (= `n_agents` the trainer vectorizes over).
- `spacing: float = 200.0` — distance between tile origins (must exceed an agent world's reach:
  arena extent + `ray_length`, with margin).

Behavior:
- `_ready()`: for `i in range(count)`, `instantiate()` `world_scene`, set its `position =
  tile_offset(i, spacing, _cols())`, `add_child(it)`. (Children enter the `AGENT` group; `NcnnSync`
  in the same scene collects them all.)
- Pure helper `static tile_offset(index: int, spacing: float, cols: int) -> Vector3` — lays tiles
  in a roughly-square grid on the XZ plane: `Vector3((index % cols) * spacing, 0, (index / cols) *
  spacing)`. `cols = ceil(sqrt(count))` via `_cols()`. Unit-tested.
- Guards: `count < 1` → `push_warning`, instance nothing; `world_scene == null` → `push_error`,
  instance nothing.

### `examples/rover_3d/rover_world.tscn` — reusable rover world sub-scene

The rover world WITHOUT a Sync node: `RoverGame` (+ `AgentBody`/`RaycastSensor3D`, `Goal`,
`Obstacles`, `RoverAgent`) — identical node structure to the existing `rover_3d.tscn` play scene,
minus Sync. This is what the arena replicates. (The existing `rover_3d.tscn` / `_train.tscn` /
smoke scenes are left untouched to avoid churning tested scenes.)

### `examples/rover_3d/rover_3d_train_parallel.tscn` — fast training scene

A root node containing a `ParallelArena` (`world_scene = rover_world.tscn`, `count = 8`) and a
`Sync` (`NcnnSync`, `control_mode = 1`). The arena spawns 8 tiled rover worlds; Sync collects all 8
agents → godot-rl trains over 8 parallel envs from one process.

### Tile-offset safety (one change to `RoverGame`)

`RoverGame.read_obstacles()` currently stores `child.global_position` as obstacle centers, while
`move_agent`/`is_blocked`/`reset_positions` operate on `_agent_body.position` (local to RoverGame).
At the origin these coincide, but under a tile offset they diverge → blocking breaks. Fix:
`read_obstacles` stores `to_local(child.global_position)` (obstacle centers in RoverGame's local
frame), matching the local agent positions. It is offset-invariant and **identical at the origin**
(existing `test_rover_game*` unaffected). The sensor's raycasts run in **world** space, so the
200-unit spacing keeps each rover sensing only its own tile. Goal/bearing obs use local positions
(relative geometry is offset-invariant) → unchanged.

## Data flow

`ParallelArena._ready` spawns N worlds → each `RoverAgent` joins `AGENT` → `NcnnSync._get_agents`
puts all N in `agents_training` → handshake reports `n_agents = N` → godot-rl creates an N-wide
vectorized env → PPO collects N× samples per physics tick. No trainer code changes.

## Error handling

- `count < 1` or `world_scene == null` → warn/error, spawn nothing (NcnnSync then finds 0 agents →
  its existing "no agents" path; the scene still loads).
- Homogeneity: `NcnnSync` derives obs/action space from `agents_training[0]`; all tiled worlds are
  identical, so this holds. (Heterogeneous agents are out of scope.)

## Testing strategy

- **Unit (`test/unit/test_parallel_arena.gd`):** `tile_offset` grid math — index 0 → origin;
  within-row spacing; row wrap at `cols`; a few (count, cols) cases. Pure, headless.
- **Headless smoke (`test/integration/parallel_arena_smoke_scene.tscn` + checker):** an arena with
  `world_scene = rover_world.tscn`, `count = 4`. Steps N physics frames; asserts exactly 4 nodes in
  the `AGENT` group, each agent's `get_obs()` returns 8 finite values, and the four agents sit at
  distinct world offsets (≥ spacing apart) — proving isolation + spawning. Wired into
  `run_tests.sh`. Drives random actions like the rover smoke.
- **Throughput validation (deferred, needs the training port):** a short `count=8`
  `rover_3d_train_parallel.tscn` run vs the single-agent baseline, comparing samples/sec. Run after
  the current rover model run frees port 11008.
- Full `./test/run_tests.sh` green from a clean cache.

## Scope boundaries / deferrals

- No refactor of the existing rover scenes (leave the tested ones alone).
- SubViewport-per-agent isolation: deferred (only needed if a future env can't be tile-offset-safe).
- Heterogeneous agents / curriculum: out of scope.
- **Sequencing:** implement *after* the rover trained-model branch ships and merges, so A1 builds on
  the shipped rover and the throughput validation doesn't collide with the in-flight training run.

## Follow-ups to record on completion

1. New backlog item: **JAX/NumPy + Gymnasium env "twin"** — reimplement an example's dynamics as a
   vectorized pure-Python/JAX Gymnasium env for 100–1000× faster training, deploy the policy back in
   Godot (only viable for simple envs; reintroduces a sim-to-deploy gap to validate). Recorded "for
   later."
2. Optionally retrofit the parallel arena into the chase example.
3. Optionally measure and document the achieved speedup in the README/ncnn docs.
