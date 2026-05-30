# RaycastSensor2D + RaycastSensor3D — Design

**Date:** 2026-05-30
**Backlog item:** 3 (`docs/BACKLOG.md`)
**Roadmap reference:** `docs/superpowers/specs/2026-05-30-feature-parity-roadmap-design.md` — Track A.1
**Status:** Approved (brainstorm → spec)

## Problem & motivation

Raycasts are the most-used observation type in `godot_rl` — nearly every non-trivial
`godot_rl` environment uses them. We ship none. This is the single biggest
switching-friction gap for users migrating from `godot_rl`: without raycast sensors they
cannot replicate `godot_rl`'s showcase environments on top of our native-ncnn deployment
path.

This item delivers `RaycastSensor2D` and `RaycastSensor3D` implementing the shared sensor
interface from the roadmap:

```gdscript
func get_observation() -> Array  # flat float array
func obs_size() -> int           # declared size (for obs-space declaration)
```

**Our deployment advantage:** these sensors feed `NcnnRunner`, which runs on
mobile/web/console/desktop/edge with zero runtime. `godot_rl`'s comparable inference path
requires .NET.

## Decisions (from brainstorming)

1. **Location:** root-level `sensors/` directory (mirrors the existing `reward/` module).
   The proper `addons/godot_native_rl/sensors/` layout is **backlog item 5** — sensors
   migrate there when that lands. Not in scope here.
2. **Per-ray encoding:** **closeness**, one float per ray. Miss → `0.0`; hit →
   `1 − clamp(distance / ray_length, 0, 1)`, so a close obstacle ≈ `1.0` and a far one ≈
   `0.0`. This is `godot_rl`'s convention and the standard monotonic RL proximity signal
   (no-hit and far-hit both ≈ `0.0`).
3. **Testability:** pure math core + thin node wrapper, with the physics cast isolated
   behind an **injectable seam** so the full `get_observation()` path is testable headlessly
   without a ticking physics world. Verifying against real physics is **deferred** to a
   follow-up.
4. **Composition:** **manual** — the agent calls `sensor.get_observation()` inside its own
   `get_obs()` and concatenates. No controller-contract change. Auto-discovery
   (`collect_sensors()`) belongs to the item-5 base-controller refactor.
5. **Demo scope:** sensor scripts + headless unit tests only. **No** new RL example, **no**
   retraining, **no** per-ray class one-hot.

## Architecture

A new root-level `sensors/` module of small, focused files.

### `sensors/raycast_math.gd` — pure static helpers

`extends RefCounted`; all functions `static`. This is the bug-prone math, fully testable
headlessly with no physics.

- `ray_directions_2d(n_rays: int, cone_degrees: float, forward_radians: float) -> Array`
  Returns an `Array` of `Vector2` unit direction vectors.
  - `n_rays < 1` → empty array.
  - `n_rays == 1` → a single ray along `forward_radians`.
  - `n_rays > 1` → rays spread **evenly** across `[forward − cone/2, forward + cone/2]`
    inclusive of both endpoints (step = `cone / (n_rays − 1)`). Order: from
    `forward − cone/2` to `forward + cone/2`.

- `ray_directions_3d(n_w: int, n_h: int, h_fov_deg: float, v_fov_deg: float) -> Array`
  Returns an `Array` of `Vector3` unit direction vectors, an `n_w × n_h` grid centered on
  forward (`−Z`, Godot's 3D forward). Yaw spread across `h_fov_deg`, pitch across
  `v_fov_deg`, both using the same even-endpoint-inclusive rule as 2D (a count of `1` along
  an axis means centered/zero offset on that axis). Order: row-major, height (pitch) outer,
  width (yaw) inner. `n_w < 1` or `n_h < 1` → empty array.

- `closeness(distance: float, ray_length: float) -> float`
  - `ray_length <= 0.0` → `0.0` (guard).
  - `distance < 0.0` (miss sentinel) → `0.0`.
  - otherwise → `clampf(1.0 - distance / ray_length, 0.0, 1.0)`.

### `sensors/raycast_sensor_2d.gd` — `extends Node2D`

`class_name RaycastSensor2D` for editor ergonomics, **but** tests reference it via `preload`
(headless `class_name` is unreliable per project conventions).

Exports:
- `n_rays: int = 8`
- `ray_length: float = 200.0`
- `cone_degrees: float = 90.0`
- `collision_mask: int = 1`
- `collide_with_areas: bool = false`
- `collide_with_bodies: bool = true`

Injectable cast seam:
- `var _cast_fn = null` — when set (a `Callable(origin: Vector2, dir: Vector2) -> float`),
  used instead of the physics query. Set via `set_cast_fn_for_test(fn: Callable)`.
- Production path `_cast(origin, dir)`: queries
  `get_world_2d().direct_space_state.intersect_ray(PhysicsRayQueryParameters2D.create(...))`
  with `collision_mask`, `collide_with_areas`, `collide_with_bodies`; returns the hit
  distance (`origin.distance_to(result.position)`), or `-1.0` on miss.

Behavior:
- `get_observation() -> Array`: directions = `raycast_math.ray_directions_2d(n_rays,
  cone_degrees, global_rotation)`; for each, distance = `_cast(global_position, dir)`; map
  through `raycast_math.closeness(distance, ray_length)`; return the flat `Array`.
- `obs_size() -> int`: `max(n_rays, 0)`.

### `sensors/raycast_sensor_3d.gd` — `extends Node3D`

`class_name RaycastSensor3D` (tests use `preload`).

Exports:
- `n_rays_width: int = 4`
- `n_rays_height: int = 2`
- `ray_length: float = 20.0`
- `horizontal_fov: float = 90.0`
- `vertical_fov: float = 45.0`
- `collision_mask: int = 1`
- `collide_with_areas: bool = false`
- `collide_with_bodies: bool = true`

Injectable cast seam mirrors 2D: `_cast_fn` is a `Callable(origin: Vector3, dir: Vector3)
-> float`; production path uses `get_world_3d().direct_space_state.intersect_ray(
PhysicsRayQueryParameters3D.create(...))`. Directions are produced by `ray_directions_3d`
and rotated into world space by the node's `global_transform.basis`. `obs_size()` =
`max(n_rays_width, 0) * max(n_rays_height, 0)`.

## Data flow

```
agent.get_obs():
    var rays := raycast_sensor.get_observation()   # Array[float], length obs_size()
    return {"obs": other_features + rays}           # manual concatenation

agent obs-space declaration:
    size includes raycast_sensor.obs_size()
```

Manual composition only. The controller is untouched, so the existing trained-chase
inference and golden regression tests are unaffected.

## Error handling

Validate at boundaries; never silently swallow:
- `n_rays < 1` (2D) / `n_rays_width < 1` or `n_rays_height < 1` (3D) → `get_observation()`
  returns `[]`, `obs_size()` returns `0`, plus a one-time `push_warning`.
- `ray_length <= 0.0` → `closeness` returns `0.0` (guarded in pure math).
- Production cast with no world/space state available (e.g. not in tree) and no injected
  `_cast_fn` → `push_error` and return a zero-filled array of length `obs_size()` (stable
  shape so the trainer/runner contract never breaks).

## Testing strategy

TDD, headless `extends SceneTree` harness (`test/harness.gd`), all references via `preload`
(no bare `class_name`). New `test/unit/test_*.gd` files are auto-discovered by
`test/run_tests.sh`.

**`test_raycast_math.gd` (pure, no physics):**
- 2D: direction count equals `n_rays`; `n_rays == 1` aligns with forward; `n_rays < 1` → empty;
  endpoints land at `forward ± cone/2`; rotating `forward_radians` rotates all dirs; unit length.
- 3D: direction count equals `n_w * n_h`; center alignment near `−Z`; fov spread; degenerate
  counts → empty; unit length.
- `closeness`: miss → 0; near hit → ~1; far hit → ~0; clamp beyond range; `ray_length <= 0` → 0.

**`test_raycast_sensor_2d.gd` (injected stub caster, no physics):**
- `obs_size()` equals `n_rays`; `get_observation()` length equals `obs_size()`.
- All-miss stub → all `0.0`; close-hit stub → values near `1.0`; mixed distances map in order.
- Node rotation flows into the cast directions (assert dirs passed to stub rotate with
  `global_rotation`).
- `n_rays < 1` → `[]` and `obs_size() == 0`.

**`test_raycast_sensor_3d.gd` (injected stub caster, no physics):**
- `obs_size()` equals `n_w * n_h`; observation length matches; all-miss → zeros;
  close-hit → ~1; degenerate counts → `[]`.

**Full suite gate:** `./test/run_tests.sh` must be fully green, including the unchanged
trained-chase inference and golden-inference regression.

**Deferred:** a real ticking-physics `.tscn` integration scene that builds colliders and
asserts a true `RayCast`/space-state hit registers. Tracked as a follow-up.

## Scope boundaries (YAGNI / explicit deferrals)

- **No** per-ray detectable-class one-hot (roadmap scopes item 3 to "1 float per ray").
- **No** new RL example, scene, or retraining (a raycast-driven trained example is a future
  backlog item).
- **No** addon-layout move (backlog item 5).
- **No** controller auto-discovery of sensors (`collect_sensors()` — item 5 follow-up).
- **No** real-physics integration scene (deferred follow-up).

## Follow-ups to record on completion

1. Auto-discovery `collect_sensors()` in the base controller — fold into item 5.
2. Real-physics `.tscn` verification scene for raycast sensors.
3. Per-ray class one-hot detection, if a ported `godot_rl` env requires it.
4. Migrate `sensors/` into `addons/godot_native_rl/sensors/` with item 5.
