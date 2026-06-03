# ISensor2D / ISensor3D Interface + `collect_sensors()` — Design

**Date:** 2026-06-03
**Backlog item:** 40 — *`ISensor2D` / `ISensor3D` interface* (roadmap Track A; upstream-plugin portability)
**Status:** design approved, pre-implementation

## Goal

Give the flat-float sensors a shared, lightweight base interface (`ISensor2D` / `ISensor3D`) and
add `NcnnControllerCore.collect_sensors(root)` so an agent can auto-discover its child sensors and
build the `obs` vector without hand-written concatenation. This matches the upstream
`godot_rl_agents` plugin's `ISensor2D`/`ISensor3D` concept (improving cross-plugin familiarity) and
removes the manual obs-composition boilerplate flagged when the sensor library was built (deferred
from backlog items 3 and 5).

## Key decisions (locked in brainstorming)

1. **Lightweight, query-based interface — not upstream's stateful shape.** Upstream's `ISensor*`
   carries `_obs`/`_active`/`activate`/`deactivate`/`reset`/`_update_observation`. Our sensors are
   pure and query-based, so our interface declares only the two methods our sensors already expose:
   `get_observation() -> Array` and `obs_size() -> int`. This matches the backlog wording and adds
   no unused state (YAGNI).
2. **Flat-float sensors only; `CameraSensor` stays separate.** `CameraSensor` returns a hex
   `String` (not `Array`), contributes a *named* obs key, and a `box` space entry via its own
   `get_observation_key()` / `get_obs_space_entry()`. Forcing it under a `get_observation() ->
   Array` / `obs_size() -> int` contract would be a leaky abstraction. It keeps its purpose-built
   contract and is composed manually as today. The interface is additive — an image path can be
   added later if a real mixed case arises.
3. **`collect_sensors()` is a helper that builds the obs vector; the agent stays in control.**
   It returns one concatenated `Array`; the agent calls it inside its own `get_obs()` and may add
   manual extras and the `CameraSensor` key alongside. This preserves the existing required
   `get_obs()` contract and the `get_obs_space()` inference path (which reads `get_obs()`).
4. **Recursive descendant discovery, in stable scene-tree order.** Sensors may be nested under
   pivots/rigs/mounts. Order is depth-first pre-order over `get_children()` → deterministic obs
   layout. (Documented caveat: reordering sensor nodes changes the obs layout.)
5. **Discovery is duck-typed, and sensors extend the interface BY PATH.** The global class-name
   cache is unreliable headless (documented gotcha), so: (a) sensors use
   `extends "res://addons/godot_native_rl/sensors/i_sensor_2d.gd"`, never `extends ISensor2D`;
   (b) `collect_sensors()` discovers via `has_method("get_observation") and has_method("obs_size")`,
   never `is ISensor2D`. The interface's `class_name` exists only for in-editor recognition /
   upstream familiarity and is never relied on at runtime headless.

## Architecture

Two tiny interface scripts + one discovery helper on the controller core, plus re-parenting the
six existing flat sensors onto the interface (behavior-preserving). Mirrors the
pure-helper + thin-wrapper conventions already in `sensors/`.

```
sensors/i_sensor_2d.gd          (class_name ISensor2D, extends Node2D)   ← new
sensors/i_sensor_3d.gd          (class_name ISensor3D, extends Node3D)   ← new
controllers/ncnn_controller_core.gd   + collect_sensors() / _gather_sensor_obs()  ← modify
controllers/ncnn_ai_controller_2d.gd  + thin collect_sensors() convenience        ← modify
controllers/ncnn_ai_controller_3d.gd  + thin collect_sensors() convenience        ← modify
sensors/{raycast,relative_position,grid}_sensor_{2,3}d.gd   extends → interface (by path)  ← modify (6)
```

## Component 1 — `sensors/i_sensor_2d.gd` / `i_sensor_3d.gd`

```gdscript
class_name ISensor2D
extends Node2D

# Shared base for 2D flat-float sensors: each contributes a flat Array of floats to the
# agent observation. Subclasses override both methods.
#
# Subclasses MUST extend this BY PATH:
#     extends "res://addons/godot_native_rl/sensors/i_sensor_2d.gd"
# never `extends ISensor2D` — the global class-name cache is unreliable headless (see
# CLAUDE.md). The class_name above is for in-editor recognition only. Sensor discovery
# (NcnnControllerCore.collect_sensors) is duck-typed, never `is ISensor2D`.

func get_observation() -> Array:
	return []

func obs_size() -> int:
	return 0
```

`i_sensor_3d.gd` is identical with `class_name ISensor3D` / `extends Node3D`. Because the interface
extends `Node2D`/`Node3D`, subclasses remain `Node2D`/`Node3D` as before.

## Component 2 — Retrofit the six flat sensors

For each of `raycast_sensor_2d`, `relative_position_sensor_2d`, `grid_sensor_2d` change
`extends Node2D` → `extends "res://addons/godot_native_rl/sensors/i_sensor_2d.gd"`; for
`raycast_sensor_3d`, `relative_position_sensor_3d`, `grid_sensor_3d` change `extends Node3D` →
`extends "res://addons/godot_native_rl/sensors/i_sensor_3d.gd"`. Each keeps its own `class_name`
and all existing code. Each already defines `get_observation()` and `obs_size()`, so this is a
**zero-behavior-change** re-parenting — the methods now formally override the interface stubs.
`CameraSensor` is untouched (`extends Node`).

## Component 3 — `NcnnControllerCore.collect_sensors(root)`

```gdscript
# Recursively gather flat sensors under `root` (duck-typed) in stable scene-tree order and
# concatenate their observations into one flat Array. Nodes without obs_size() (e.g.
# CameraSensor, which returns a hex String under its own obs key) are skipped — compose
# those manually. Depth-first pre-order over get_children() → deterministic obs layout.
static func collect_sensors(root: Node) -> Array:
	var out: Array = []
	_gather_sensor_obs(root, out)
	return out

static func _gather_sensor_obs(node: Node, out: Array) -> void:
	for child in node.get_children():
		if child.has_method("get_observation") and child.has_method("obs_size"):
			out.append_array(child.get_observation())
		_gather_sensor_obs(child, out)
```

Discovery predicate `has_method("get_observation") and has_method("obs_size")` matches the flat
sensors and excludes `CameraSensor` (which has `get_observation` but **no** `obs_size`). Recursion
visits every node so nested sensors are found; a matched sensor's own children are still visited
(harmless unless a user nests a sensor under a sensor, which is their explicit choice).

## Component 4 — Controller convenience method

Add to both `NcnnAIController2D` and `NcnnAIController3D`:

```gdscript
# Convenience: concatenate all child flat-sensor observations (recursive, tree order).
# Agents can write `return {"obs": collect_sensors()}` in get_obs().
func collect_sensors() -> Array:
	return NcnnControllerCore.collect_sensors(self)
```

Agent usage becomes, e.g.:
```gdscript
func get_obs() -> Dictionary:
	return {"obs": collect_sensors()}
# or with manual extras / a camera key:
func get_obs() -> Dictionary:
	return {"obs": own_floats + collect_sensors(), "camera_2d": cam.get_observation()}
```
`get_obs_space()` is unchanged — it still infers from `get_obs()` via `obs_space_from_obs()`.

## Data flow

```
agent.get_obs()
  → controller.collect_sensors()                       (instance convenience)
    → NcnnControllerCore.collect_sensors(agent)         (static, testable)
      → _gather_sensor_obs(agent, out)                  (recursive, duck-typed, tree order)
  → {"obs": [...concatenated flat sensor floats...], (+ manual extras / camera key)}
  → obs_space_from_obs(get_obs())  (unchanged)  →  NcnnRunner
```

## Error handling

- No sensors under `root` → `collect_sensors()` returns `[]` (agent decides how to handle an
  empty obs; degenerate but not a crash).
- A sensor whose `get_observation()` returns `[]` (its own degenerate guard) contributes nothing —
  consistent with each sensor's existing behavior.
- Duck-typing guards against calling sensor methods on arbitrary nodes (physics bodies, meshes,
  pivots) — only nodes with both methods contribute.
- The interface stubs return `[]` / `0`, so an un-overridden subclass fails safe (contributes
  nothing) rather than crashing.

## Testing (headless, `test/unit/`, auto-discovered by `run_tests.sh`)

- `test_collect_sensors.gd` — a synthetic `Node` tree built in `_initialize()` with mock flat
  sensors (tiny scripts/objects exposing `get_observation()` + `obs_size()`) at varying depths,
  plus a camera-like node (`get_observation()` but **no** `obs_size()`):
  - empty tree → `[]`;
  - single sensor → its obs;
  - multiple sensors → concatenation in depth-first tree order (assert exact order with distinct
    per-sensor values);
  - nested sensor (under a non-sensor pivot) is found;
  - camera-like node is **skipped** (its values never appear);
  - a non-sensor node (no methods) is ignored.
- `test_i_sensor.gd` — base stubs: a bare `ISensor2D`/`ISensor3D` returns `[]` / `0`; a tiny
  subclass that `extends` the interface **by path** and overrides both methods works and is
  discovered by `collect_sensors()`.
- **Retrofit safety:** the existing per-sensor unit tests
  (`test_raycast_sensor_2d/3d`, `test_relative_position_sensor_2d/3d`, `test_grid_sensor_2d/3d`)
  must pass unchanged — re-parenting is behavior-preserving. Full suite green from a clean cache.

No trained-model migration: re-ordering any trained example's obs would risk its golden regression
for zero functional gain, so `collect_sensors()` is proven by the synthetic tests above and the
unchanged sensor suites — not by migrating the rover/chase agents.

## Docs to update (same change)

- `CLAUDE.md` — sensors line: note the `ISensor2D/3D` base + `collect_sensors()` auto-discovery.
- `README.md` — sensors section: agents can `collect_sensors()` instead of manual concatenation;
  note `CameraSensor` is composed separately.
- `docs/BACKLOG.md` — mark item 40 ✅ with spec + plan paths; add 40 to the "Done" summary line.

## YAGNI / out of scope

- No upstream-style stateful interface (`_obs`/`activate`/`reset`/`_update_observation`).
- No image/`CameraSensor` path in `collect_sensors()` (composed manually).
- No `obs_size`-summing helper (`get_obs_space()` already infers from a live `get_obs()`).
- No migration of existing trained example agents onto `collect_sensors()`.
- Items 41 (raycast multi-class) and 42 (relative-position multi-target) are separate cycles.
