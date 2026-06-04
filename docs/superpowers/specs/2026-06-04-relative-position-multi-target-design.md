# RelativePositionSensor2D/3D — Multi-Target + Full Upstream Parity — Design

**Date:** 2026-06-04
**Backlog item:** 42 (`docs/BACKLOG.md`) — GitHub issue **#15**
**Builds on:** item 7 (`RelativePositionSensor2D/3D`, single target) and item 40 (`ISensor2D/3D` interface)
**Upstream reference:** `edbeeching/godot_rl_agents_plugin` →
`addons/godot_rl_agents/sensors/sensors_2d/PositionSensor2D.gd` and `sensors_3d/PositionSensor3D.gd`
**Roadmap reference:** `docs/superpowers/specs/2026-05-30-feature-parity-roadmap-design.md` — Track A (Sensors)
**Status:** Approved (brainstorm → spec)

## Problem & motivation

This repo's `RelativePositionSensor2D/3D` (item 7) observes a **single** target via a `target_path:
NodePath`, emitting an egocentric unit direction + clipped normalized distance (3 floats in 2D, 4 in
3D). Upstream's `PositionSensor2D/3D` is richer: it observes an **array** of targets and exposes
per-axis toggles plus two encoding modes. The gap analysis flags this as partial parity (`#15`).

The driving constraint from brainstorming: **godot_rl compatibility is the priority** (the addon has
no downstream users yet, so our own legacy API shape carries no weight). "Compatibility" here means a
policy trained against upstream `PositionSensor` — and a scene authored against it — should behave the
same here. That requires mirroring upstream's *full* API and *both* encoding modes, not merely adding a
target array.

Crucially, this repo's current encoding (`[dir, dist]`, always emitted) matches **only** upstream's
**non-default** mode (`use_separate_direction = true`). Upstream's **default** mode emits a normalized,
clamped offset vector with **no** separate distance scalar. So minimal "add an array" work would still
diverge from upstream's default behavior.

**Deployment advantage (unchanged):** like all sensors here, this feeds `NcnnRunner` — native ncnn
inference on mobile/web/console/desktop/edge with zero runtime. Upstream's comparable inference path
requires .NET.

## Upstream behavior being matched (verified from source)

```gdscript
@export var objects_to_observe: Array[Node2D]            # direct node refs (Array[Node3D] in 3D)
@export var include_x := true
@export var include_y := true
@export var include_z := true                            # 3D only
@export_range(0.01, 20_000) var max_distance := 1.0      # 3D range is 0.01 .. 2_500
@export var use_separate_direction: bool = false
```

Per target, with egocentric local offset `L = to_local(obj.global_position)`:

- **`use_separate_direction = false` (DEFAULT):**
  `relative = L.limit_length(max_distance) / max_distance`, then append the **enabled axes**
  (`relative.x` if `include_x`, `relative.y` if `include_y`, `relative.z` if `include_z`).
  **No distance scalar is appended.**
- **`use_separate_direction = true`:**
  `direction = L.normalized()`, `distance = min(L.length() / max_distance, 1.0)`; append the enabled
  **direction axes**, **then** `distance`.

Other upstream behaviors:
- **Order:** iterate `objects_to_observe` in array order; each target contributes its slot in order.
- **Freed/invalid target:** `relative_position` stays `Vector2.ZERO` (`is_instance_valid(obj)` is
  false), so its slot zero-fills. The array length is fixed at author time, so the slot count — and
  therefore `obs_size` — never changes. This is exactly what a fixed-width RL policy input needs.
- Upstream also draws debug lines (`Line2D` / `ImmediateMesh`). **Not** part of obs/training parity.

## Decisions (from brainstorming)

1. **Full parity scope.** Mirror upstream's full API and both encoding modes — not the minimal
   "Array of targets, existing encoding only" interpretation in the backlog one-liner.
2. **Keep the class names** `RelativePositionSensor2D` / `RelativePositionSensor3D` (repo convention,
   issue title, and `ISensor2D/3D` conformance). Compatibility is delivered by matching the **obs
   encoding and export semantics**, not the class identity — a ported policy cares about the obs
   vector, not the node's class name.
3. **Match upstream's export names exactly:** `objects_to_observe`, `include_x`, `include_y`,
   `include_z` (3D), `max_distance`, `use_separate_direction`. Maximizes familiarity for users coming
   from godot_rl.
4. **Direct node references** `Array[Node2D]` / `Array[Node3D]` (true upstream parity), replacing the
   single `target_path: NodePath`. The existing detached-node fallback keeps it headless-testable (see
   §Testing); the test seam becomes "assign the array directly."
5. **Adopt upstream's `max_distance` defaults and ranges:** default `1.0`; `@export_range(0.01,
   20_000)` in 2D, `@export_range(0.01, 2_500)` in 3D. (Replaces this repo's `1000.0` / `50.0`.)
6. **Omit debug rendering** (`Line2D` / `ImmediateMesh`). It is visualization only, does not affect the
   obs vector or training, and `_ready()`-time mesh/line creation would complicate the headless tests.
   Deliberate non-goal; can be opened as a separate backlog item if a visual aid is later wanted.
7. **Architecture unchanged in spirit:** pure static math core + thin node wrapper + headless test
   seam, same as the raycast sensors. All bug-prone logic (frame rotation, normalize, `limit_length`
   clamp, axis masking, mode selection, edge cases) lives in the pure core and is unit-tested with no
   tree/physics.

## Encoding (exact replication)

For a target with egocentric local offset `L` (the world offset rotated into the sensor's local frame):

**Mode `use_separate_direction = false` (default):**
```
scaled = L.limit_length(max_distance) / max_distance     # vector clamped to length max_distance, then normalized by it
append scaled.x if include_x
append scaled.y if include_y
append scaled.z if include_z        # 3D only
```

**Mode `use_separate_direction = true`:**
```
dir  = L.normalized()               # zero vector when L is zero-length
dist = min(L.length() / max_distance, 1.0)
append dir.x if include_x
append dir.y if include_y
append dir.z if include_z            # 3D only
append dist
```

**Guards (both modes):**
- `max_distance <= 0.0` → emit the slot as all zeros (avoid divide-by-zero; degenerate config).
- Zero-length `L` → zeros in both modes (normalize → zero vector; `limit_length(...)/max` → zero;
  `dist` → 0).
- A freed / `null` / invalid target is treated as `L = ZERO`, so its slot emits the correct count of
  zeros and the total length is unchanged.

Note: distance and the clamp use vector length, which is rotation-invariant — the egocentric frame
rotation only affects direction/axis components, never the magnitude.

## Architecture / files

Three files under `addons/godot_native_rl/sensors/`, all already present from item 7 — extended, not
created.

### `relative_position_math.gd` — pure static helpers

`extends RefCounted`; all functions `static`. Replaces the current single-mode `encode_2d`/`encode_3d`
with mode- and axis-aware versions, plus a size helper.

- `encode_2d(world_offset: Vector2, sensor_rotation: float, max_distance: float,
  use_separate_direction: bool, include_x: bool, include_y: bool) -> Array`
  - `L := world_offset.rotated(-sensor_rotation)`
  - dispatch on `use_separate_direction` per §Encoding; apply axis masks; apply guards.
- `encode_3d(world_offset: Vector3, sensor_basis: Basis, max_distance: float,
  use_separate_direction: bool, include_x: bool, include_y: bool, include_z: bool) -> Array`
  - `L := sensor_basis.inverse() * world_offset`
  - same dispatch with the extra `z` axis.
- `per_target_size(use_separate_direction: bool, include_x: bool, include_y: bool,
  include_z: bool = false) -> int`
  - enabled-axis count `+ (1 if use_separate_direction else 0)`.

Keeping the frame rotation inside the pure core (rather than in the wrapper) preserves the item-7
property that the **full** egocentric encoding is unit-tested headlessly.

### `relative_position_sensor_2d.gd` / `relative_position_sensor_3d.gd` — thin node wrappers

`extends` the `i_sensor_2d.gd` / `i_sensor_3d.gd` path (unchanged conformance).

Exports (replacing `target_path` / the old `max_distance`):
```gdscript
@export var objects_to_observe: Array[Node2D]     # Array[Node3D] in 3D
@export var include_x: bool = true
@export var include_y: bool = true
@export var include_z: bool = true                # 3D only
@export_range(0.01, 20_000) var max_distance: float = 1.0   # 0.01 .. 2_500 in 3D
@export var use_separate_direction: bool = false
```

- `obs_size() -> int`: `objects_to_observe.size() * RelativePositionMath.per_target_size(...)`.
- `get_observation() -> Array`: iterate `objects_to_observe` in order; for each, compute
  `world_offset` (`ZERO` when `!is_instance_valid(obj)`), call `encode_2d`/`encode_3d`, append its
  result. Concatenation order = array order.
  - Sensor frame: `global_transform` when `is_inside_tree()`, else `transform` (detached fallback).
  - Target position: `obj.global_position` when `obj.is_inside_tree()`, else `obj.position`.
- **Test seam:** assign `objects_to_observe` directly (the array is the seam — no separate
  `set_target_for_test` override needed; the detached-position fallback already handles nodes added in
  a `--script` test's `_initialize()` that are not yet inside the tree).
- Keep the existing "warn once when a target is missing" behavior, adapted to the array (warn at most
  once when any slot resolves invalid, reset when all valid) so logs aren't spammed per frame.

## Composition (unchanged)

Sensors are auto-discovered by `NcnnControllerCore.collect_sensors(agent)` / the controllers'
`collect_sensors()` (duck-typed `is ISensor2D/3D`, tree order). Agents that build `get_obs()` by hand
call `get_observation()` and concatenate; `obs_size()` declares the contributed width. No controller
changes.

## Testing

Pure-core tests (headless, no tree, no physics — the bulk of coverage):
- Both modes × axis-toggle combinations (e.g. `include_x` only, `x+y`, all on; 3D adds `z`).
- `use_separate_direction = true` appends `dist` as the trailing element after the enabled axes.
- `use_separate_direction = false` appends **no** `dist`; verify the `limit_length` clamp: a target
  beyond `max_distance` yields a unit-length-scaled vector (components sum-of-squares ≈ 1 after the
  clamp at the boundary), a target at half distance yields half-scaled components.
- Guards: zero offset → zeros; `max_distance <= 0` → zeros.
- Egocentric rotation: a known world offset + sensor rotation/basis maps to the expected local axes.
- `per_target_size` matches the emitted per-target length across mode/axis configs.

Wrapper tests (assign `objects_to_observe` directly):
- Multi-target concatenation order matches array order.
- A freed / `null` slot zero-fills while sibling slots stay correct, and total length is unchanged.
- `obs_size()` equals the emitted observation length across mode/axis/target-count configs.
- Detached-node fallback: nodes assigned but not inside the tree still produce a finite observation
  (no `global_position` error).

Update existing tests:
- `test/unit/test_relative_position_sensor_2d.gd` and `_3d.gd` currently assert the old always-`[dir,
  dist]` 3/4-float encoding and `obs_size` 3/4. Rework them for the new default mode
  (normalized-offset, 2/3 floats for a single target) and add the mode/axis/multi-target cases above.
- `test/unit/test_sensor_interface_conformance.gd` only checks `is ISensor2D/3D` — stays green.

All tests run under `./test/run_tests.sh`, which must be green before merge.

## Docs to update (per CLAUDE.md "before every push")

- **README** — sensor list entry for `RelativePositionSensor2D/3D` (multi-target + modes).
- **CLAUDE.md** — the sensors paragraph (`RelativePositionSensor2D/3D` description).
- **`docs/godot-rl-gap-analysis-2026-06-02.md`** — flip the `PositionSensor` row from `⚠️ partial
  (#15)` to ✅ and update the priority table row.
- **`docs/BACKLOG.md`** — check item 42, and add to the Done list in CLAUDE.md's roadmap section.
- Close **#15** via `Closes #15` in the PR.

## Non-goals

- Debug line/mesh rendering (upstream's `Line2D` / `ImmediateMesh`).
- Renaming the classes to upstream's `PositionSensor2D/3D`.
- Any example rewire or retraining (the rover keeps its inline goal obs; trained-model/golden
  regressions are untouched). An example using the sensor remains backlog item 32.
- Backward compatibility with the old single `target_path` export (no downstream users; clean break).

## Risks

- **Breaking the existing unit tests' assumptions** is expected and intended (the default encoding
  changes); the risk is forgetting a doc/gap-analysis update — covered by the docs checklist above.
- **`@export_range` + typed `Array[Node2D]`** must round-trip through the editor inspector correctly;
  verified by the wrapper tests assigning the array programmatically (the path the controller uses).
