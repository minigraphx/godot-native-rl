# RelativePositionSensor2D + RelativePositionSensor3D — Design

**Date:** 2026-06-01
**Backlog item:** 7 (`docs/BACKLOG.md`)
**Upstream motivation:** `godot_rl` issue #177 ("a relative position vector from an object to
another object, clipped at some distance and normalized")
**Roadmap reference:** `docs/superpowers/specs/2026-05-30-feature-parity-roadmap-design.md` — Track A (Sensors)
**Status:** Approved (brainstorm → spec)

## Problem & motivation

After raycasts, "where is my target relative to me?" is the next most common observation in
`godot_rl` environments (navigate-to-goal, seek/flee, pursuit). Today every agent hand-codes
it — the `rover_3d` example computes `[sin(bearing), cos(bearing), dist_norm]` inline in
`RoverAgent.compute_goal_obs`. `godot_rl` issue #177 requests a reusable sensor for exactly
this, and we ship none. This item delivers `RelativePositionSensor2D` and
`RelativePositionSensor3D` implementing the shared sensor interface already used by the
raycast sensors:

```gdscript
func get_observation() -> Array  # flat float array
func obs_size() -> int           # declared size (for obs-space declaration)
```

**Our deployment advantage:** like the raycast sensors, these feed `NcnnRunner`, which runs
on mobile/web/console/desktop/edge with zero runtime. `godot_rl`'s comparable inference path
requires .NET.

## Decisions (from brainstorming)

1. **Two variants — 2D and 3D**, mirroring `RaycastSensor2D`/`RaycastSensor3D`.
2. **Location:** `addons/godot_native_rl/sensors/` directly (backlog item 5 already landed,
   so the addon layout is in place — no root-level staging like item 3 had).
3. **Output encoding: unit direction + scalar normalized distance** (literal reading of the
   backlog's "normalized direction + clipped distance"). 2D → `[dir_x, dir_y, dist_norm]`
   (3 floats); 3D → `[dir_x, dir_y, dir_z, dist_norm]` (4 floats). The direction is always
   unit-length, so bearing and distance are **decoupled** signals for the policy.
   - *Rejected alternative:* a single clipped-and-normalized offset vector (2/3 floats). More
     compact, but a near-zero offset loses direction, and it couples the two signals.
4. **Frame: egocentric (sensor-local).** The offset is rotated into the sensor node's local
   frame, so "forward / left / right" is relative to the agent's heading. This matches the
   rover's existing bearing obs and is the RL standard.
   - *Rejected alternative:* world-frame direction — simpler, but the policy must learn its
     own heading separately; rarely what RL envs want.
5. **Testability:** pure static math core + thin node wrappers, same as the raycast sensors.
   The egocentric rotation + normalize + clip all live in the pure core (no tree, no physics),
   so the full encoding is unit-tested headlessly.
6. **Demo scope:** sensor scripts + headless unit tests only. **No** example rewire, **no**
   retraining (the rover's obs stays as-is, so the trained-rover/golden regressions are
   untouched). A **new backlog item** is opened for an example that uses the sensor.

## Architecture

Three new small, focused files under `addons/godot_native_rl/sensors/`.

### `relative_position_math.gd` — pure static helpers

`extends RefCounted`; all functions `static`. This holds the bug-prone logic (frame rotation,
normalization, clipping, edge cases), fully testable headlessly with no node/tree/physics.

- `encode_2d(world_offset: Vector2, sensor_rotation: float, max_distance: float) -> Array`
  Returns `[dir_x, dir_y, dist_norm]`.
  - `local := world_offset.rotated(-sensor_rotation)` (rotate world→sensor-local frame).
  - `dir := local.normalized()` (unit, or `Vector2.ZERO` when `local` is zero-length).
  - `dist_norm := clampf(world_offset.length() / max_distance, 0.0, 1.0)`.

- `encode_3d(world_offset: Vector3, sensor_basis: Basis, max_distance: float) -> Array`
  Returns `[dir_x, dir_y, dir_z, dist_norm]`.
  - `local := sensor_basis.inverse() * world_offset` (world→sensor-local).
  - `dir := local.normalized()` (unit, or `Vector3.ZERO` when zero-length).
  - `dist_norm := clampf(world_offset.length() / max_distance, 0.0, 1.0)`.

  Note: distance is computed from `world_offset.length()` (rotation-invariant), so the local
  rotation only affects the direction, never the magnitude.

- **Guards (both):**
  - `world_offset` zero-length → direction is the zero vector, `dist_norm = 0.0`.
  - `max_distance <= 0.0` → `dist_norm = 0.0` (avoids divide-by-zero; degenerate config).

### `relative_position_sensor_2d.gd` — `extends Node2D`

`class_name RelativePositionSensor2D` for editor ergonomics, **but** tests reference it via
`preload` (headless `class_name` is unreliable per project conventions).

Exports:
- `target_path: NodePath` — the node whose position is observed.
- `max_distance: float = 1000.0`.

Behavior:
- `obs_size() -> int`: `3`.
- `get_observation() -> Array`:
  - Resolve `target := get_node_or_null(target_path)`. If null → one-time `push_error` and
    return a zero-filled `Array` of length `obs_size()` (stable shape).
  - `sensor_xform := global_transform if is_inside_tree() else transform` (the same
    tree-fallback `RaycastSensor3D` uses, so the path is headless-testable);
    `sensor_pos := sensor_xform.origin`, `sensor_rotation := sensor_xform.get_rotation()`.
  - `target_pos := target.global_position if target.is_inside_tree() else target.position`.
  - `world_offset := target_pos - sensor_pos`.
  - return `RelativePositionMath.encode_2d(world_offset, sensor_rotation, max_distance)`.

### `relative_position_sensor_3d.gd` — `extends Node3D`

`class_name RelativePositionSensor3D` (tests use `preload`).

Exports:
- `target_path: NodePath`.
- `max_distance: float = 50.0`.

Behavior mirrors 2D:
- `obs_size() -> int`: `4`.
- `get_observation() -> Array`: resolve target (null → `push_error` + zero-filled array of
  length `obs_size()`); `sensor_xform := global_transform if is_inside_tree() else transform`;
  `world_offset := target_pos - sensor_xform.origin`; return
  `RelativePositionMath.encode_3d(world_offset, sensor_xform.basis, max_distance)`.

## Data flow

```
agent.get_obs():
    var rel := relative_position_sensor.get_observation()   # Array[float], length obs_size()
    return {"obs": other_features + rel}                     # manual concatenation

agent obs-space declaration:
    size includes relative_position_sensor.obs_size()
```

Manual composition only (same as the raycast sensors). The controller is untouched, so the
existing trained-chase/trained-rover inference and golden regressions are unaffected.

## Error handling

Validate at boundaries; never silently swallow (per project conventions):
- Empty / unresolvable `target_path` → one-time `push_error`, return zero-filled array of
  `obs_size()` (the trainer/runner contract needs a stable observation shape every step).
- `max_distance <= 0.0` → `dist_norm = 0.0`, guarded in pure math.
- Zero offset (target coincident with sensor) → zero direction + `dist_norm = 0.0`.

## Testing strategy

TDD, headless `extends SceneTree` harness (`test/harness.gd`), all references via `preload`
(no bare `class_name`). New `test/unit/test_*.gd` files are auto-discovered by
`test/run_tests.sh`.

**`test_relative_position_math.gd` (pure, no node/tree):**
- 2D: target dead-ahead of an unrotated sensor → `dir ≈ (1, 0)` (or the project's forward
  convention) + correct `dist_norm`; rotating `sensor_rotation` rotates the local direction
  predictably (e.g. a target to the world-right reads as "behind/ahead" after a 90° turn);
  `dist_norm` is rotation-invariant; distance beyond `max_distance` clips to `1.0`; half
  `max_distance` → `0.5`; zero offset → `[0, 0, 0]`; `max_distance <= 0` → `dist_norm == 0`;
  direction is unit length for non-zero offsets.
- 3D: analogous with `Basis`; target along `-Z` (Godot forward) of an unrotated sensor →
  `dir ≈ (0, 0, -1)`; yaw rotation rotates the local direction; `dist_norm` rotation-invariant
  and clips; zero offset → `[0, 0, 0, 0]`; unit-length direction.

**`test_relative_position_sensor_2d.gd` / `_3d.gd` (node wrappers, headless):**
- `obs_size()` is `3` / `4`; `get_observation()` length matches.
- A target node at a known relative position yields the expected encoding (use detached nodes
  with `position`/`rotation` set, exercising the `is_inside_tree()` fallback).
- Rotating the sensor node rotates the egocentric direction.
- Missing/empty `target_path` → zero-filled array of the correct length (and no crash).

**Full suite gate:** `./test/run_tests.sh` must be fully green, including the unchanged
trained-chase + trained-rover inference and golden-inference regressions.

## Scope boundaries (YAGNI / explicit deferrals)

- **No** multi-target or tag-based selection (issue #177 "possible extension").
- **No** extra target properties (velocity, custom user properties — issue #177 extension).
- **No** example rewire or retraining; the rover keeps its inline goal obs.
- **No** controller auto-discovery of sensors (`collect_sensors()` — still an item-5 follow-up).
- **No** raw (un-normalized) output mode.

## Follow-ups to record on completion

1. **New backlog item:** an example that uses `RelativePositionSensor` (e.g. a small 2D
   seek/navigate-to-target demo, or migrate the rover's goal obs onto it with a retrain).
2. Multi-target / tag selection + extra target properties (velocity), if a ported `godot_rl`
   env needs them (issue #177 extensions).
3. Sensor auto-discovery `collect_sensors()` in the controller core — fold into the item-5
   follow-up that also covers the raycast sensors.
