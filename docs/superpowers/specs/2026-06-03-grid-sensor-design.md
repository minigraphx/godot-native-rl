# GridSensor2D + GridSensor3D — Design

**Date:** 2026-06-03
**Backlog item:** 11 — *GridSensor2D + GridSensor3D — cell-based spatial detection* (roadmap spec Track A.3)
**Status:** design approved, pre-implementation

## Goal

Add cell-based spatial-detection sensors for 2D and 3D, reaching godot_rl `GridSensor2D`/
`GridSensor3D` feature parity. A grid of cells centered on the sensor reports, per cell, how
many objects of each detection layer overlap it. The flat float array feeds an agent's
`get_obs()` (manual composition, same as the existing raycast/relative-position sensors) and
ultimately `NcnnRunner`.

## Key decisions (locked)

1. **Query-based detection (repo style), not event-driven.** godot_rl spawns live `Area2D`/
   `Area3D` cells and mutates a persistent buffer on enter/exit signals. We instead compute the
   observation **fresh on each `get_observation()` call** by querying the physics space, building
   a new buffer immutably. This matches the existing `RaycastSensor2D/3D` and
   `RelativePositionSensor2D/3D` pattern, obeys the repo immutability rule, and is fully
   headless-testable via an injectable query seam. (Trade-off: slightly more per-frame cost than
   signal accumulation — acceptable; cell counts are small and queries are cheap.)
2. **Per-layer count encoding (godot_rl parity).** One float per active detection-layer bit per
   cell = count of overlapping objects on that layer. `obs_size = grid_a * grid_b * n_layers`.
   This keeps both the **index layout and the semantics identical to godot_rl**, so a GridSensor
   agent is portable in/out of the upstream plugin. Counts are unbounded but tiny in practice;
   users who want normalization already have `ObsNormalize`/VecNormalize.
3. **Both 2D and 3D this item.** The encoding is dimension-agnostic and lives in a shared pure
   `grid_sensor_math.gd`; only the thin node wrapper + physics query seam differ between 2D and
   3D — exactly how `raycast_math` backs `raycast_sensor_2d/3d`.

## Architecture

Three files under `addons/godot_native_rl/sensors/`, mirroring the raycast split:

```
grid_sensor_math.gd     (RefCounted, pure, dimension-agnostic)  ← all encoding logic + unit tests
grid_sensor_2d.gd       (Node2D)   thin wrapper, physics query seam
grid_sensor_3d.gd       (Node3D)   thin wrapper, physics query seam
```

### Shared sensor interface (matches Track A)

```gdscript
func get_observation() -> Array  # flat float array, fresh each call
func obs_size() -> int           # declared size (for get_obs_space)
```

## Component 1 — `grid_sensor_math.gd` (pure, RefCounted)

Dimension-agnostic. `grid_a`/`grid_b` are the two grid axes (2D: x,y; 3D: x,z). `step_a`/`step_b`
are the per-cell spacing on each axis.

- `collision_mapping(detection_mask: int) -> Dictionary`
  Maps each **set bit index** of the mask to a sequential obs slot, e.g. mask `0b101`
  → `{0: 0, 2: 1}`. Identical to godot_rl's `_get_collision_mapping`.
- `n_layers(detection_mask: int) -> int` — number of set bits (`= collision_mapping.size()`).
- `obs_size(grid_a: int, grid_b: int, detection_mask: int) -> int`
  `= maxi(grid_a,0) * maxi(grid_b,0) * n_layers(detection_mask)`.
- `obs_index(cell_i: int, cell_j: int, layer_slot: int, grid_b: int, n_layers: int) -> int`
  godot_rl formula: `(cell_i * grid_b * n_layers) + (cell_j * n_layers) + layer_slot`.
- `cell_offsets(grid_a: int, grid_b: int, step_a: float, step_b: float) -> Array` (of `Vector2`)
  Local cell-center offsets relative to the sensor origin, using godot_rl's **integer-division**
  shift: `shift = Vector2(-(grid_a/2)*step_a, -(grid_b/2)*step_b)` (int division — odd grids are
  symmetric about origin, even grids are offset by half a cell, matching godot_rl). Order is
  `i` outer / `j` inner (row-major over the grid), so element index `= i*grid_b + j`. 2D wrappers
  use the `Vector2` directly; the 3D wrapper maps `Vector2(x, y) → Vector3(x, 0, y)`.
- `build_obs(cell_layers: Array, grid_a: int, grid_b: int, detection_mask: int) -> Array` (of float)
  **Core encoding.** `cell_layers` is a flat array of length `grid_a*grid_b`, indexed
  `i*grid_b + j`, where each element is an `Array[int]` of the `collision_layer` values of objects
  overlapping that cell. For each cell, for each overlapping object's `collision_layer`, for each
  set bit that is in `collision_mapping`, increment `obs[obs_index(...)]` by 1. Returns a fresh
  flat `Array` of floats of size `obs_size`. Pure → the bulk of unit testing lives here.

## Component 2 — `grid_sensor_2d.gd` (Node2D)

Exports (defaults match godot_rl GridSensor2D):
- `@export_flags_2d_physics var detection_mask: int = 1`
- `@export var collide_with_areas: bool = false`
- `@export var collide_with_bodies: bool = true`
- `@export var cell_width: float = 20.0`
- `@export var cell_height: float = 20.0`
- `@export var grid_size_x: int = 3`
- `@export var grid_size_y: int = 3`

Test seam: `set_overlap_fn_for_test(fn: Callable)` where
`fn(cell_center: Vector2, cell_size: Vector2) -> Array[int]` returns the collision layers of
objects overlapping the cell. `null` → real physics.

- `obs_size() -> int` → `GridSensorMath.obs_size(grid_size_x, grid_size_y, detection_mask)`.
- `get_observation() -> Array`:
  - Degenerate guards (matching raycast sensors): `grid_size_x < 1` or `grid_size_y < 1` or
    `n_layers == 0` → one-time `push_warning` + return `[]`. No world and no seam →
    `push_error` + zeros of `obs_size()`.
  - Transform cell offsets into world space using `global_transform if is_inside_tree() else
    transform` (so detached unit-test instances still resolve, per the headless gotcha; in normal
    use cells rotate/translate with the node — same as godot_rl child cells).
  - For each offset, world center `= xform * offset`; gather overlap layers (seam or real) →
    build `cell_layers`; return `GridSensorMath.build_obs(...)`.
- `_overlap(center: Vector2, size: Vector2) -> Array`: real path uses
  `get_world_2d().direct_space_state.intersect_shape` with a `RectangleShape2D(size)` positioned
  at `center` (with the node rotation), `collision_mask = detection_mask`, `collide_with_areas`/
  `collide_with_bodies` from exports; reads each result's `collider.collision_layer`. Null-world
  guard returns `[]`.

## Component 3 — `grid_sensor_3d.gd` (Node3D)

Same shape, with godot_rl GridSensor3D differences:
- `@export_flags_3d_physics var detection_mask: int = 1`
- `@export var collide_with_areas: bool = false`
- `@export var collide_with_bodies: bool = false`  *(godot_rl note: won't detect StaticBody3D
  without an Area; the default mirrors upstream)*
- `cell_width: float = 1.0`, `cell_height: float = 1.0`
- `grid_size_x: int = 3`, `grid_size_z: int = 3`  *(grid is on the X/Z horizontal plane)*

Placement uses `cell_width` on **both** grid axes (`step_a = step_b = cell_width`); `cell_height`
is only the box's **Y extent**, not a grid step. Box shape `BoxShape3D(cell_width, cell_height,
cell_width)`. Seam `fn(cell_center: Vector3, cell_size: Vector3) -> Array[int]`. Real path uses
`get_world_3d().direct_space_state.intersect_shape` with `PhysicsShapeQueryParameters3D`.
`obs_index` uses `grid_b = grid_size_z`.

## Data flow

```
get_observation()
  → cell_offsets(grid_a, grid_b, step_a, step_b)         [pure]
  → for each cell: world center = xform * offset
                   layers = overlap(center, cell_size)    [physics or seam]
  → build_obs(cell_layers, grid_a, grid_b, detection_mask) [pure]
  → flat Array[float]  →  agent.get_obs() concatenation  →  NcnnRunner
```

## Error handling

- `n_rays`-style degenerate guards: empty grid or empty detection mask → warn once + empty obs;
  `obs_size()` returns 0 so a caller never mis-sizes the obs space.
- No physics world and no injected seam → `push_error` + zeros (never crash, never silently
  return wrong-length data).
- All failure paths return arrays of the declared length (or empty when the sensor is degenerate),
  so obs-space inference stays consistent.

## Testing (headless, `test/unit/`, auto-discovered by `run_tests.sh`)

- `test_grid_sensor_math.gd` — the encoding core:
  - `collision_mapping` for single/multi/sparse masks; `n_layers`.
  - `obs_size` incl. zero-grid and zero-mask.
  - `obs_index` against hand-computed values.
  - `cell_offsets`: centering for odd grids (symmetric about origin), even grids (half-cell
    offset), 1×N grids, order `i*grid_b+j`.
  - `build_obs`: empty cells → zeros; single object on one layer → count 1 at the right index;
    multiple objects same cell+layer → count accumulates; object spanning multiple mapped layers
    → increments each slot; layer bit **outside** the mask → ignored; output length == obs_size.
- `test_grid_sensor_2d.gd` / `test_grid_sensor_3d.gd` — wrapper via injected overlap stub:
  - empty stub → all-zero obs of length `obs_size()`;
  - known overlaps at specific cells → correct indexed counts;
  - cell centers translate/rotate with node transform (record centers passed to the stub);
  - degenerate grid → empty obs + `obs_size()==0`.

Use path-based `preload`/`extends` (no bare `class_name` base), per the headless class-cache
gotcha. Run the full suite from a clean cache before merge.

## Docs to update (same change)

- `CLAUDE.md` — add GridSensor2D/3D to the `sensors/` description.
- `docs/BACKLOG.md` — mark item 11 ✅ with spec + plan paths.
- `README` — if it enumerates sensors, add the grid sensors.

## YAGNI / out of scope

- No `debug_view` (godot_rl's editor cell visualization) — not needed for headless train/deploy.
- No event-driven buffer, no live `Area` cell nodes, no `@tool` editor spawning.
- No example scene this item (sensors compose into agents manually; an example can follow if a
  showcase env needs it).
