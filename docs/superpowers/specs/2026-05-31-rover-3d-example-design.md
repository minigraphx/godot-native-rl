# 3D Raycast Rover Example — Design

**Date:** 2026-05-31
**Backlog item:** 6 (`docs/BACKLOG.md`) — reframed from "navigate-to-target" to a raycast obstacle-avoidance rover
**Status:** Approved (brainstorm → spec)

## Problem & motivation

We need a 3D example that exercises the new `NcnnAIController3D` (item 5) and `RaycastSensor3D`
(item 3) together, reusing the existing godot_rl + SB3 PPO training pipeline unchanged and
deploying via native ncnn. A raycast obstacle-avoidance rover is a recognizable godot_rl-style
environment that ties items 1 (reward authoring), 3 (sensors), and 5 (controllers) into one
showcase, while staying within the deploy path's current limits (discrete, single action-key,
feed-forward).

## Decisions (from brainstorming)

1. **Example:** raycast obstacle-avoidance rover (not the original navigate-to-target).
2. **Movement:** tank-style, discrete actions `{idle, forward, turn-left, turn-right}` (4). The
   rover has a heading; rays fan from the heading so "what's in front of me" drives behavior.
3. **Goal sensing:** egocentric — `[sin(bearing), cos(bearing), clamp(dist/max)]` where `bearing`
   is the angle to the goal relative to the rover's heading (XZ plane).
4. **Collision:** block movement + small per-step penalty; episode does **not** terminate on
   collision (dense, forgiving signal → trains reliably headless).
5. **Layout:** fixed obstacle field; randomize rover start pose + goal each episode; on reaching
   the goal, relocate it (chase-style) and continue the episode.
6. **Artifact scope:** build the full scaffold + headless tests now and merge; the real PPO
   training run → ncnn conversion → shipped model → golden-inference regression is a clearly
   scoped **final explicit step** after the scaffold lands.

## Architecture

New directory `examples/rover_3d/`, mirroring `examples/chase_the_target/`.

### `rover_game.gd` — `RoverGame extends Node3D`

World state + the bug-prone logic as **pure, headless-unit-testable helpers**.

Exports/config: `arena_size: Vector2` (XZ extent), `move_speed`, `turn_speed`, `goal_radius`
(reach threshold), `agent_body_path`, `goal_path`. Obstacles are authored in the scene as
`StaticBody3D` + `CollisionShape3D` (so the `RaycastSensor3D` hits them); the game also holds an
`obstacles` data list (`{center: Vector3, half_extent: Vector3}`) used by the pure helpers and
the movement blocking. (Source-of-truth detail resolved in the plan: the game reads obstacle
AABBs from its `StaticBody3D` children at `_ready`, so the authored colliders are the single
source and the data list is derived — no hand-kept duplication. Pure tests inject the data list
directly.)

Pure helpers:
- `clamp_to_bounds(pos: Vector3) -> Vector3` — clamp X/Z to the arena (Y fixed on the plane).
- `is_blocked(pos: Vector3, obstacles: Array) -> bool` — true if `pos` lies within any obstacle
  AABB (center ± half_extent, XZ).
- `random_free_position(rng, obstacles) -> Vector3` — a random in-bounds position not blocked.
- `max_distance() -> float` — arena diagonal (for distance normalization).
- `bearing_to(agent_pos, agent_yaw, goal_pos) -> float` — signed angle (radians, XZ) from the
  rover's heading to the goal direction.
- `seed_rng(s: int) -> void` — non-negative seed (uint64).

Runtime helpers (exercised by the scene + smoke test):
- `get_agent_pos()`, `get_agent_yaw()`, `get_goal_pos()`, `distance()`.
- `move_agent(forward: float, yaw_delta: float, delta: float) -> void` — apply yaw, then attempt
  forward translation along the new heading; if the target cell `is_blocked`, **block** the move
  (keep position) and `emit_signal("bumped")`. Always `clamp_to_bounds`.
- `relocate_goal()` — `reaches += 1`, move goal to `random_free_position`, `emit goal_reached`.
- `reset_positions()` — randomize agent pose + goal to free positions.

Signals: `goal_reached`, `bumped`.

### `rover_agent.gd` — `RoverAgent`

`extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd"` (path-based, per
CLAUDE.md / item 5 — cache-independent headless resolution). Holds a `RaycastSensor3D` child
reference (set in the scene) and the `RoverGame` via `game_path`.

- `ACTION_KEY := "move"`, `ACTION_COUNT := 4`.
- `get_action_space()` → `{"move": {"size": 4, "action_type": "discrete"}}`.
- **Pure helpers:**
  - `action_index_to_motion(idx, move_speed, turn_speed) -> Dictionary` →
    `{"forward": f, "yaw": y}`: `0`=idle `{0,0}`, `1`=forward `{move_speed,0}`,
    `2`=turn-left `{0,-turn_speed}`, `3`=turn-right `{0,+turn_speed}`.
  - `compute_goal_obs(agent_pos, agent_yaw, goal_pos, max_dist) -> Array` →
    `[sin(bearing), cos(bearing), clampf(dist/max_dist, 0, 1)]`.
  - `compose_obs(ray_obs: Array, goal_obs: Array) -> Array` → `ray_obs + goal_obs` (manual
    composition per item 3; ray_obs comes from the sensor at runtime).
- `get_obs()` → `{"obs": compose_obs(_sensor.get_observation(), compute_goal_obs(...))}`; null-game
  fallback returns a correctly-sized zero vector.
- `set_action(action)` → store the discrete index (validated in range).
- Reward (reuses item 1): `reward_source = RewardBuilder.new().add_progress_shaping(game.distance,
  game.max_distance, ["goal_reached"]).add_event_bonus("goal_reached", goal_bonus)
  .add_step_penalty(step_penalty).build()`; plus a `RewardAdapter` mapping the game's `bumped`
  signal → a small negative `collision_penalty` (showcases signal→reward).
- `_physics_process(delta)`: `super._physics_process(delta)`; if game present →
  `action_index_to_motion` → `game.move_agent(...)`; `accumulate_reward()`; if
  `game.distance() < game.goal_radius` → `game.relocate_goal()`; handle `needs_reset` (reset
  positions, `reset()`, `zero_reward()`, `reward_source.reset()` — chase pattern).

### Sensor

`RaycastSensor3D` child of the rover: a horizontal fan — `n_rays_width = 5`, `n_rays_height = 1`,
`horizontal_fov ≈ 120`, `vertical_fov = 0`, `ray_length` ≈ arena-scale, `collision_mask` matching
the obstacle bodies. `obs_size() == 5`. Total observation = 5 rays + 3 goal = **8 floats**.

### Scenes

- `rover_3d.tscn` — play/inference scene (RoverGame + AgentBody + Goal + RoverAgent with a
  RaycastSensor3D child + obstacles as StaticBody3D).
- `rover_3d_train.tscn` — training scene: same world + a `Sync` (`NcnnSync`) node, `control_mode`
  wiring exactly like `chase_the_target_train.tscn` (agent `control_mode = TRAINING`, sync
  `control_mode = HUMAN`/INHERIT per chase convention).

### Training scripts

- `scripts/train_rover.py` — clone of `train_chase.py`: `StableBaselinesGodotEnv(env_path=None)`,
  `VecMonitor`, `PPO("MultiInputPolicy", …)`, save `.zip`, export ONNX. Defaults:
  `--save_model_path models/rover_policy.zip`, `--onnx_export_path models/rover_policy.onnx`.
- `scripts/train_rover.sh` — orchestration mirroring `train_chase.sh` (start trainer on the port,
  launch the headless training scene).

## Data flow

`NcnnSync` ↔ `RoverAgent` uses the unchanged contract (`get_obs_space`, `get_obs`, `get_reward`,
`get_action_space`, `set_action`, `get_done`, …) inherited from `NcnnAIController3D`. The rover's
`get_obs()` concatenates the sensor output with egocentric goal features (manual composition).
Action shape (1 discrete head) and obs shape (8-vector) keep the training pipeline identical to
chase.

## Error handling

- `RoverAgent` with no game → `get_obs()` returns a zero vector of the correct length (8), with a
  one-time `push_warning` (mirrors `ChaseAgent`).
- `action_index_to_motion` asserts the index is in `[0, ACTION_COUNT)`.
- `random_free_position` is bounded-retry: after N attempts it returns the last candidate (avoids
  an infinite loop if the arena is over-packed) — N and arena/obstacle sizes chosen so free space
  always exists.

## Testing strategy (TDD, headless `extends SceneTree` harness, `preload` refs)

- **`test_rover_game.gd`** (pure): `clamp_to_bounds`; `is_blocked` inside/outside obstacle AABBs;
  `random_free_position` never blocked + in-bounds (seeded, many samples); `max_distance`;
  `bearing_to` (goal ahead → ~0, left → negative, right → positive, behind → ±π); seeded RNG
  determinism; `relocate_goal` increments `reaches` and emits `goal_reached`.
- **`test_rover_game_blocking.gd`** (or folded in): `move_agent` blocked by an obstacle keeps
  position and emits `bumped`; an unobstructed move advances along heading and clamps to bounds.
- **`test_rover_agent.gd`** (pure): `action_index_to_motion` for all 4 indices; `compute_goal_obs`
  shape + values (ahead/left/right/distance saturation); `compose_obs` length (5+3=8) and order;
  `get_action_space`.
- **Scene smoke** (`test/integration/` style): `rover_3d_train.tscn` loads and steps N physics
  frames headless with no errors (sensor + agent + game wired, NcnnSync in human/training mode
  without a server prints the expected warning and continues).
- **Full gate:** `./test/run_tests.sh` green (incl. existing trained-chase + golden), from a clean
  cache state.
- **Deferred (final explicit step):** run real PPO training → `export_to_ncnn.py` →
  `models/rover_policy.ncnn.*` → a `trained_rover_scene` + golden-inference regression wired into
  `run_tests.sh`, matching chase's bar.

## Scope boundaries / deferrals

- Discrete single-key action only (continuous/multi-key/LSTM are items 21–22).
- Fixed obstacle layout; randomized obstacles deferred.
- No new sensor types — reuses `RaycastSensor3D` as-is.
- The trained ncnn model + golden regression land after the scaffold is merged (the long training
  run is isolated as the final step).

## Follow-ups to record on completion

1. Final step: train the rover, convert to ncnn, ship `models/rover_policy.ncnn.*`, add the
   trained-rover golden regression to `run_tests.sh`.
2. Tutorial doc (`docs/examples/rover_3d_tutorial.md`) paralleling the chase tutorial, once the
   trained model exists.
