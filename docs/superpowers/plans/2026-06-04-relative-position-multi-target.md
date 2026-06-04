# RelativePositionSensor2D/3D — Multi-Target + Full Upstream Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `RelativePositionSensor2D/3D` from a single `target_path` to an array of targets with full godot_rl `PositionSensor` parity — direct node refs, per-axis toggles, and both encoding modes (default normalized-offset; optional unit-direction + distance).

**Architecture:** Keep the repo's pure-static-math-core + thin-node-wrapper + headless-test-seam pattern. All bug-prone logic (frame rotation, `limit_length` clamp, normalize, axis masking, mode dispatch, guards) lives in `relative_position_math.gd` and is unit-tested with no tree/physics. The two node wrappers loop their `objects_to_observe` array, compute each egocentric world offset, and concatenate the math core's per-target output. `obs_size()` derives from mode + enabled axes × target count, so the policy input width is fixed at author time.

**Tech Stack:** GDScript (Godot 4.6, TAB indentation), dependency-free headless test harness (`test/harness.gd`, tests `extends SceneTree`, auto-discovered by `test/run_tests.sh` via `test/unit/test_*.gd`).

**Spec:** `docs/superpowers/specs/2026-06-04-relative-position-multi-target-design.md` (item 42 / issue #15).

---

## Critical gotchas (read before starting)

- **Typed-array property assignment hangs headless.** Assigning an *untyped* array literal to a typed
  `Array[Node2D]` `@export` errors and hangs the test. ALWAYS declare a typed local first:
  `var targets: Array[Node2D] = [t1, t2]` then `s.objects_to_observe = targets`. Never
  `s.objects_to_observe = [t1, t2]` directly.
- **`:=` can't infer from an untyped value** (Godot 4.6). Use explicit types where the RHS is untyped
  (e.g. `var out: Array = []` is fine; `var obs: Array = s.get_observation()`).
- **Subclass `extends` BY PATH**, never `extends ISensor2D` (class-name cache is unreliable headless).
  The existing wrappers already do `extends "res://addons/godot_native_rl/sensors/i_sensor_2d.gd"` —
  keep that line unchanged.
- **`global_position`/`global_transform` error when a node is not `is_inside_tree()`** (headless tests
  add nodes via `.new()` without adding to the tree). The wrappers must read `global_*` only when
  `is_inside_tree()`, else fall back to `transform` / `.position`. The existing wrappers already do
  this — preserve it.
- **Run the suite via `./test/run_tests.sh`** (it regenerates the script-class cache fresh each run).
  Do NOT run a single test in isolation expecting `class_name` to resolve.

---

## File Structure

- **Modify** `addons/godot_native_rl/sensors/relative_position_math.gd` — replace the single-mode
  `encode_2d`/`encode_3d` with mode + axis-aware versions; add `per_target_size`.
- **Modify** `addons/godot_native_rl/sensors/relative_position_sensor_2d.gd` — new exports
  (`objects_to_observe`, `include_x/y`, `max_distance`, `use_separate_direction`); array loop;
  `obs_size` from config. Remove `target_path` / `set_target_for_test`.
- **Modify** `addons/godot_native_rl/sensors/relative_position_sensor_3d.gd` — same, with `include_z`.
- **Create** `test/unit/test_relative_position_math.gd` — pure-core tests (both modes, axes, guards).
- **Modify** `test/unit/test_relative_position_sensor_2d.gd` — rewrite for the multi-target wrapper.
- **Modify** `test/unit/test_relative_position_sensor_3d.gd` — rewrite for the multi-target wrapper.
- **Modify** docs: `README.md`, `CLAUDE.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`,
  `docs/BACKLOG.md`.
- **Unchanged:** `test/unit/test_sensor_interface_conformance.gd` (only checks `is ISensor2D/3D` and
  `.new()` — must stay green).

---

## Task 1: Pure math core — modes, axes, `per_target_size`

**Files:**
- Modify: `addons/godot_native_rl/sensors/relative_position_math.gd`
- Test: `test/unit/test_relative_position_math.gd` (create)

- [ ] **Step 1: Write the failing test** — create `test/unit/test_relative_position_math.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RelativePositionMath = preload("res://addons/godot_native_rl/sensors/relative_position_math.gd")

func _initialize() -> void:
	var h := Harness.new()

	# --- per_target_size ---
	h.assert_eq(RelativePositionMath.per_target_size(false, true, true, false), 2, "2D default mode -> 2")
	h.assert_eq(RelativePositionMath.per_target_size(true, true, true, false), 3, "2D separate mode -> 3 (dir+dist)")
	h.assert_eq(RelativePositionMath.per_target_size(false, true, true, true), 3, "3D default mode -> 3")
	h.assert_eq(RelativePositionMath.per_target_size(true, true, true, true), 4, "3D separate mode -> 4")
	h.assert_eq(RelativePositionMath.per_target_size(false, true, false, false), 1, "x-only default -> 1")
	h.assert_eq(RelativePositionMath.per_target_size(true, false, false, false), 1, "all-axes-off separate -> 1 (dist only)")

	# --- 2D default mode (non-separate): normalized clamped offset, NO distance ---
	var d: Array = RelativePositionMath.encode_2d(Vector2(50, 0), 0.0, 100.0, false, true, true)
	h.assert_eq(d.size(), 2, "default mode emits 2 (no dist)")
	h.assert_true(absf(d[0] - 0.5) < 1e-5 and absf(d[1]) < 1e-5, "half distance +X -> (0.5,0)")

	var c: Array = RelativePositionMath.encode_2d(Vector2(200, 0), 0.0, 100.0, false, true, true)
	h.assert_true(absf(c[0] - 1.0) < 1e-5 and absf(c[1]) < 1e-5, "beyond max -> clamped unit (1,0)")

	# --- 2D separate mode: unit dir + dist ---
	var s: Array = RelativePositionMath.encode_2d(Vector2(50, 0), 0.0, 100.0, true, true, true)
	h.assert_eq(s.size(), 3, "separate mode emits 3")
	h.assert_true(absf(s[0] - 1.0) < 1e-5 and absf(s[1]) < 1e-5 and absf(s[2] - 0.5) < 1e-5, "separate -> dir(1,0)+dist0.5")

	# --- axis mask: include_y only, separate -> [dir.y, dist] ---
	var m: Array = RelativePositionMath.encode_2d(Vector2(0, 50), 0.0, 100.0, true, false, true)
	h.assert_eq(m.size(), 2, "include_y only separate -> 2")
	h.assert_true(absf(m[0] - 1.0) < 1e-5 and absf(m[1] - 0.5) < 1e-5, "y-axis dir + dist")

	# --- egocentric rotation: target +X, sensor +90deg -> local (0,-1) ---
	var r: Array = RelativePositionMath.encode_2d(Vector2(10, 0), PI / 2.0, 100.0, true, true, true)
	h.assert_true(absf(r[0]) < 1e-5 and absf(r[1] + 1.0) < 1e-5, "rotation maps +X to local -Y")

	# --- guards ---
	var z: Array = RelativePositionMath.encode_2d(Vector2.ZERO, 0.0, 100.0, true, true, true)
	h.assert_true(absf(z[0]) < 1e-6 and absf(z[1]) < 1e-6 and absf(z[2]) < 1e-6, "zero offset -> zeros")
	var g: Array = RelativePositionMath.encode_2d(Vector2(10, 0), 0.0, 0.0, false, true, true)
	h.assert_true(g.size() == 2 and absf(g[0]) < 1e-6 and absf(g[1]) < 1e-6, "max<=0 -> zeros (correct count)")

	# --- 3D separate: target -Z forward -> [0,0,-1, 0.1] ---
	var t3: Array = RelativePositionMath.encode_3d(Vector3(0, 0, -10), Basis(), 100.0, true, true, true, true)
	h.assert_eq(t3.size(), 4, "3D separate -> 4")
	h.assert_true(absf(t3[0]) < 1e-5 and absf(t3[1]) < 1e-5 and absf(t3[2] + 1.0) < 1e-5 and absf(t3[3] - 0.1) < 1e-5, "3D forward -> [0,0,-1,0.1]")

	# --- 3D default mode: normalized clamped offset, no dist ---
	var d3: Array = RelativePositionMath.encode_3d(Vector3(0, 0, -50), Basis(), 100.0, false, true, true, true)
	h.assert_eq(d3.size(), 3, "3D default -> 3 (no dist)")
	h.assert_true(absf(d3[2] + 0.5) < 1e-5, "3D half -Z -> z=-0.5")

	# --- 3D axis mask: include_z only, default -> [scaled.z] ---
	var mz: Array = RelativePositionMath.encode_3d(Vector3(0, 0, -50), Basis(), 100.0, false, false, false, true)
	h.assert_eq(mz.size(), 1, "z-only default -> 1")
	h.assert_true(absf(mz[0] + 0.5) < 1e-5, "z-only -> -0.5")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_relative_position_math.gd`
Expected: FAIL — `encode_2d` is currently called with the old 3-arg signature, so the new 6-arg calls and `per_target_size` error out / mismatch.

- [ ] **Step 3: Replace the math core.** Overwrite `addons/godot_native_rl/sensors/relative_position_math.gd` with:

```gdscript
class_name RelativePositionMath
extends RefCounted

# Pure, stateless helpers for relative-position sensors. No physics, no node state — fully
# unit-testable headlessly. Mirrors godot_rl's PositionSensor2D/3D encoding exactly.
#
# Each target contributes an EGOCENTRIC slot (world offset rotated into the sensor's local
# frame), in one of two modes:
#   use_separate_direction = false (DEFAULT): the normalized, clamped offset components
#     scaled = local.limit_length(max_distance) / max_distance   (no separate distance scalar)
#   use_separate_direction = true: the unit direction components, then a clamped normalized
#     distance scalar appended last.
# Per-axis include_x/y(/z) toggles drop axes from the output. Guards: max_distance <= 0 -> the
# slot is all zeros; a zero offset -> zeros.

# Number of floats one target contributes for a given config.
static func per_target_size(use_separate_direction: bool, include_x: bool, include_y: bool, include_z: bool = false) -> int:
	var n := 0
	if include_x:
		n += 1
	if include_y:
		n += 1
	if include_z:
		n += 1
	if use_separate_direction:
		n += 1
	return n

static func _zeros(n: int) -> Array:
	var out: Array = []
	out.resize(n)
	out.fill(0.0)
	return out

# world_offset: target_pos - sensor_pos, in world space.
# sensor_rotation: the sensor node's world rotation (radians).
static func encode_2d(world_offset: Vector2, sensor_rotation: float, max_distance: float, use_separate_direction: bool, include_x: bool, include_y: bool) -> Array:
	if max_distance <= 0.0:
		return _zeros(per_target_size(use_separate_direction, include_x, include_y, false))
	var local := world_offset.rotated(-sensor_rotation)
	var out: Array = []
	if use_separate_direction:
		var dir := local.normalized()
		var dist := minf(local.length() / max_distance, 1.0)
		if include_x:
			out.append(dir.x)
		if include_y:
			out.append(dir.y)
		out.append(dist)
	else:
		var scaled := local.limit_length(max_distance) / max_distance
		if include_x:
			out.append(scaled.x)
		if include_y:
			out.append(scaled.y)
	return out

# world_offset: target_pos - sensor_pos, in world space.
# sensor_basis: the sensor node's world-transform basis.
static func encode_3d(world_offset: Vector3, sensor_basis: Basis, max_distance: float, use_separate_direction: bool, include_x: bool, include_y: bool, include_z: bool) -> Array:
	if max_distance <= 0.0:
		return _zeros(per_target_size(use_separate_direction, include_x, include_y, include_z))
	var local := sensor_basis.inverse() * world_offset
	var out: Array = []
	if use_separate_direction:
		var dir := local.normalized()
		var dist := minf(local.length() / max_distance, 1.0)
		if include_x:
			out.append(dir.x)
		if include_y:
			out.append(dir.y)
		if include_z:
			out.append(dir.z)
		out.append(dist)
	else:
		var scaled := local.limit_length(max_distance) / max_distance
		if include_x:
			out.append(scaled.x)
		if include_y:
			out.append(scaled.y)
		if include_z:
			out.append(scaled.z)
	return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_relative_position_math.gd`
Expected: PASS — `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/relative_position_math.gd test/unit/test_relative_position_math.gd
git commit -m "feat(sensors): RelativePositionMath modes + axis toggles + per_target_size (#15)"
```

---

## Task 2: `RelativePositionSensor2D` wrapper — multi-target

**Files:**
- Modify: `addons/godot_native_rl/sensors/relative_position_sensor_2d.gd`
- Test: `test/unit/test_relative_position_sensor_2d.gd` (rewrite)

- [ ] **Step 1: Write the failing test** — overwrite `test/unit/test_relative_position_sensor_2d.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RelativePositionSensor2D = preload("res://addons/godot_native_rl/sensors/relative_position_sensor_2d.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = RelativePositionSensor2D.new()
	s.max_distance = 100.0
	s.position = Vector2.ZERO
	s.rotation = 0.0

	var t1 := Node2D.new()
	t1.position = Vector2(50, 0)    # half distance, +X
	var t2 := Node2D.new()
	t2.position = Vector2(0, 100)   # at max distance, +Y
	# Typed local FIRST (untyped literal -> typed @export hangs headless).
	var targets: Array[Node2D] = [t1, t2]
	s.objects_to_observe = targets

	# Default mode = non-separate, include x+y -> 2 floats/target.
	h.assert_eq(s.obs_size(), 4, "two targets, default mode -> obs_size 4")
	var obs: Array = s.get_observation()
	h.assert_eq(obs.size(), 4, "obs length == obs_size")
	h.assert_true(absf(obs[0] - 0.5) < 1e-5 and absf(obs[1]) < 1e-5, "slot0 = (0.5,0)")
	h.assert_true(absf(obs[2]) < 1e-5 and absf(obs[3] - 1.0) < 1e-5, "slot1 = (0,1)")

	# Separate mode -> 3 floats/target -> obs_size 6.
	s.use_separate_direction = true
	h.assert_eq(s.obs_size(), 6, "separate mode two targets -> 6")
	var obs2: Array = s.get_observation()
	h.assert_eq(obs2.size(), 6, "separate obs length 6")
	# slot0 separate: dir (1,0) + dist 0.5
	h.assert_true(absf(obs2[0] - 1.0) < 1e-5 and absf(obs2[1]) < 1e-5 and absf(obs2[2] - 0.5) < 1e-5, "slot0 separate -> (1,0,0.5)")

	# Back to default mode; free a target -> its slot zero-fills, sibling intact, length unchanged.
	s.use_separate_direction = false
	t2.free()
	var obs3: Array = s.get_observation()
	h.assert_eq(obs3.size(), 4, "freed slot keeps length 4")
	h.assert_true(absf(obs3[0] - 0.5) < 1e-5 and absf(obs3[1]) < 1e-5, "sibling slot0 intact")
	h.assert_true(absf(obs3[2]) < 1e-6 and absf(obs3[3]) < 1e-6, "freed slot1 zero-filled")

	t1.free()
	s.free()

	# Empty targets -> obs_size 0, empty obs (no crash).
	var s2 = RelativePositionSensor2D.new()
	h.assert_eq(s2.obs_size(), 0, "no targets -> obs_size 0")
	h.assert_eq(s2.get_observation().size(), 0, "no targets -> empty obs")
	s2.free()

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_relative_position_sensor_2d.gd`
Expected: FAIL — `objects_to_observe` / `use_separate_direction` don't exist yet; `obs_size` still returns fixed 3.

- [ ] **Step 3: Replace the wrapper.** Overwrite `addons/godot_native_rl/sensors/relative_position_sensor_2d.gd`:

```gdscript
class_name RelativePositionSensor2D
extends "res://addons/godot_native_rl/sensors/i_sensor_2d.gd"

# Egocentric relative-position observation for a set of target nodes, matching godot_rl's
# PositionSensor2D. Each target in `objects_to_observe` contributes a slot (see
# RelativePositionMath): the normalized clamped offset (default) or a unit direction + distance
# (use_separate_direction). Per-axis include_x/y toggles drop axes. obs_size() is fixed by the
# config and target count, so the policy input width is stable; a freed/invalid target zero-fills
# its slot rather than shrinking the vector.

const RelativePositionMath = preload("res://addons/godot_native_rl/sensors/relative_position_math.gd")

## Targets to observe, in order; each contributes a slot. Freed/invalid entries zero-fill.
@export var objects_to_observe: Array[Node2D]
## Include the relative x component in each slot.
@export var include_x: bool = true
## Include the relative y component in each slot.
@export var include_y: bool = true
## Distance normalizer. Obs values are normalized so 0 is closest and 1 is at/over this distance.
@export_range(0.01, 20000.0) var max_distance: float = 1.0
## false: emit the normalized clamped offset. true: emit unit direction + a distance scalar.
@export var use_separate_direction: bool = false

var _warned_invalid := false

func obs_size() -> int:
	return objects_to_observe.size() * RelativePositionMath.per_target_size(use_separate_direction, include_x, include_y, false)

func get_observation() -> Array:
	# World transform when in the tree; local transform fallback when detached (unit tests).
	var sensor_xform := global_transform if is_inside_tree() else transform
	var sensor_rotation := sensor_xform.get_rotation()
	var out: Array = []
	var any_invalid := false
	for obj in objects_to_observe:
		var world_offset := Vector2.ZERO
		if is_instance_valid(obj):
			var target_pos := obj.global_position if obj.is_inside_tree() else obj.position
			world_offset = target_pos - sensor_xform.origin
		else:
			any_invalid = true
		out.append_array(RelativePositionMath.encode_2d(world_offset, sensor_rotation, max_distance, use_separate_direction, include_x, include_y))
	if any_invalid and not _warned_invalid:
		push_error("RelativePositionSensor2D: one or more objects_to_observe are invalid; their slots are zero-filled.")
		_warned_invalid = true
	elif not any_invalid:
		_warned_invalid = false
	return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_relative_position_sensor_2d.gd`
Expected: PASS — `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/relative_position_sensor_2d.gd test/unit/test_relative_position_sensor_2d.gd
git commit -m "feat(sensors): RelativePositionSensor2D multi-target + modes + axis toggles (#15)"
```

---

## Task 3: `RelativePositionSensor3D` wrapper — multi-target

**Files:**
- Modify: `addons/godot_native_rl/sensors/relative_position_sensor_3d.gd`
- Test: `test/unit/test_relative_position_sensor_3d.gd` (rewrite)

- [ ] **Step 1: Write the failing test** — overwrite `test/unit/test_relative_position_sensor_3d.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RelativePositionSensor3D = preload("res://addons/godot_native_rl/sensors/relative_position_sensor_3d.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = RelativePositionSensor3D.new()
	s.max_distance = 100.0
	s.position = Vector3.ZERO
	s.rotation = Vector3.ZERO

	var t1 := Node3D.new()
	t1.position = Vector3(0, 0, -50)   # half distance, forward (-Z)
	var t2 := Node3D.new()
	t2.position = Vector3(100, 0, 0)   # at max distance, +X
	var targets: Array[Node3D] = [t1, t2]
	s.objects_to_observe = targets

	# Default mode = non-separate, include x+y+z -> 3 floats/target.
	h.assert_eq(s.obs_size(), 6, "two targets, default mode -> obs_size 6")
	var obs: Array = s.get_observation()
	h.assert_eq(obs.size(), 6, "obs length == obs_size")
	# slot0 forward half: (0,0,-0.5); slot1 +X max: (1,0,0)
	h.assert_true(absf(obs[2] + 0.5) < 1e-5 and absf(obs[0]) < 1e-5 and absf(obs[1]) < 1e-5, "slot0 = (0,0,-0.5)")
	h.assert_true(absf(obs[3] - 1.0) < 1e-5 and absf(obs[4]) < 1e-5 and absf(obs[5]) < 1e-5, "slot1 = (1,0,0)")

	# Separate mode -> 4 floats/target -> obs_size 8.
	s.use_separate_direction = true
	h.assert_eq(s.obs_size(), 8, "separate mode two targets -> 8")
	var obs2: Array = s.get_observation()
	h.assert_eq(obs2.size(), 8, "separate obs length 8")
	# slot0 separate: dir (0,0,-1) + dist 0.5
	h.assert_true(absf(obs2[2] + 1.0) < 1e-5 and absf(obs2[3] - 0.5) < 1e-5, "slot0 separate -> (...,-1,0.5)")

	# Yaw +90deg about Y rotates the forward target (-Z) to local +X (separate mode).
	s.rotation = Vector3(0.0, PI / 2.0, 0.0)
	var obs_rot: Array = s.get_observation()
	h.assert_true(absf(obs_rot[0] - 1.0) < 1e-5, "sensor yaw rotates slot0 dir to local +X")
	s.rotation = Vector3.ZERO

	# Back to default mode; free a target -> slot zero-fills, sibling intact, length unchanged.
	s.use_separate_direction = false
	t2.free()
	var obs3: Array = s.get_observation()
	h.assert_eq(obs3.size(), 6, "freed slot keeps length 6")
	h.assert_true(absf(obs3[2] + 0.5) < 1e-5, "sibling slot0 intact")
	h.assert_true(absf(obs3[3]) < 1e-6 and absf(obs3[4]) < 1e-6 and absf(obs3[5]) < 1e-6, "freed slot1 zero-filled")

	t1.free()
	s.free()

	# Empty targets -> obs_size 0, empty obs.
	var s2 = RelativePositionSensor3D.new()
	h.assert_eq(s2.obs_size(), 0, "no targets -> obs_size 0")
	h.assert_eq(s2.get_observation().size(), 0, "no targets -> empty obs")
	s2.free()

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_relative_position_sensor_3d.gd`
Expected: FAIL — new exports don't exist; `obs_size` still fixed at 4.

- [ ] **Step 3: Replace the wrapper.** Overwrite `addons/godot_native_rl/sensors/relative_position_sensor_3d.gd`:

```gdscript
class_name RelativePositionSensor3D
extends "res://addons/godot_native_rl/sensors/i_sensor_3d.gd"

# 3D egocentric relative-position observation for a set of target nodes, matching godot_rl's
# PositionSensor3D. Each target in `objects_to_observe` contributes a slot (see
# RelativePositionMath): the normalized clamped offset (default) or a unit direction (local frame,
# forward = -Z) + distance (use_separate_direction). Per-axis include_x/y/z toggles drop axes.
# obs_size() is fixed by config + target count; a freed/invalid target zero-fills its slot.

const RelativePositionMath = preload("res://addons/godot_native_rl/sensors/relative_position_math.gd")

## Targets to observe, in order; each contributes a slot. Freed/invalid entries zero-fill.
@export var objects_to_observe: Array[Node3D]
## Include the relative x component in each slot.
@export var include_x: bool = true
## Include the relative y component in each slot.
@export var include_y: bool = true
## Include the relative z component in each slot.
@export var include_z: bool = true
## Distance normalizer. Obs values are normalized so 0 is closest and 1 is at/over this distance.
@export_range(0.01, 2500.0) var max_distance: float = 1.0
## false: emit the normalized clamped offset. true: emit unit direction + a distance scalar.
@export var use_separate_direction: bool = false

var _warned_invalid := false

func obs_size() -> int:
	return objects_to_observe.size() * RelativePositionMath.per_target_size(use_separate_direction, include_x, include_y, include_z)

func get_observation() -> Array:
	var sensor_xform := global_transform if is_inside_tree() else transform
	var out: Array = []
	var any_invalid := false
	for obj in objects_to_observe:
		var world_offset := Vector3.ZERO
		if is_instance_valid(obj):
			var target_pos := obj.global_position if obj.is_inside_tree() else obj.position
			world_offset = target_pos - sensor_xform.origin
		else:
			any_invalid = true
		out.append_array(RelativePositionMath.encode_3d(world_offset, sensor_xform.basis, max_distance, use_separate_direction, include_x, include_y, include_z))
	if any_invalid and not _warned_invalid:
		push_error("RelativePositionSensor3D: one or more objects_to_observe are invalid; their slots are zero-filled.")
		_warned_invalid = true
	elif not any_invalid:
		_warned_invalid = false
	return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_relative_position_sensor_3d.gd`
Expected: PASS — `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/relative_position_sensor_3d.gd test/unit/test_relative_position_sensor_3d.gd
git commit -m "feat(sensors): RelativePositionSensor3D multi-target + modes + axis toggles (#15)"
```

---

## Task 4: Full suite green (incl. interface conformance)

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite** (regenerates the script-class cache fresh, then runs all tests):

Run: `./test/run_tests.sh`
Expected: every unit test prints `Results: N passed, 0 failed`, including
`test_sensor_interface_conformance.gd` (it only checks `RelativePositionSensor2D/3D is ISensor2D/3D`
and `.new()`, which still hold). The Python protocol/timeout/helper tests and golden regressions are
untouched and must remain green. Final line: the suite's overall success message.

- [ ] **Step 2: Clean stray UID files** (an import pass scatters them — do not commit):

Run: `git clean -fn -- '*.gd.uid'` then, if it lists files, `git clean -f -- '*.gd.uid'`
Expected: no `*.gd.uid` files staged or committed.

- [ ] **Step 3 (only if a test failed): debug.** Use superpowers:systematic-debugging. Common causes:
  the typed-array-literal hang (Step-1 gotcha — declare `var targets: Array[Node3D] = [...]` first),
  or a `global_position` call on a detached node (must be guarded by `is_inside_tree()`). Do NOT
  weaken a production type to satisfy a test stub. Re-run `./test/run_tests.sh` after the fix.

---

## Task 5: Documentation updates

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/godot-rl-gap-analysis-2026-06-02.md`
- Modify: `docs/BACKLOG.md`

- [ ] **Step 1: README — replace the two sensor bullets** (currently lines ~517–524). Find the
  `RelativePositionSensor2D` and `RelativePositionSensor3D` bullets and replace both with:

```markdown
- **`RelativePositionSensor2D`** (`sensors/relative_position_sensor_2d.gd`) — egocentric positions of
  a set of `objects_to_observe` (`Array[Node2D]`), matching `godot_rl`'s `PositionSensor2D`. Two
  modes: `use_separate_direction = false` (default) emits the normalized clamped offset
  `[x, y]` per target; `true` emits a unit direction plus a clipped normalized distance
  `[dir_x, dir_y, dist_norm]`. Per-axis `include_x`/`include_y` toggles, `max_distance` normalizer.
  Freed/invalid targets zero-fill their slot, so `obs_size()` stays fixed. Answers "where are my
  targets relative to me?" (`godot_rl` issue #177).
- **`RelativePositionSensor3D`** (`sensors/relative_position_sensor_3d.gd`) — the 3D form over
  `objects_to_observe` (`Array[Node3D]`), direction in the sensor's local frame (forward = −Z), with
  `include_x`/`include_y`/`include_z` toggles and the same two modes + `max_distance` clipping.
```

- [ ] **Step 2: CLAUDE.md — update the sensors paragraph.** Find the substring
  `+ \`RelativePositionSensor2D\`/\`RelativePositionSensor3D\` +` (around line 27) and replace it with:

```
+ `RelativePositionSensor2D`/`RelativePositionSensor3D` (multi-target `objects_to_observe`, godot_rl
PositionSensor parity: normalized-offset default + optional unit-dir/dist split, per-axis include
toggles) +
```

- [ ] **Step 3: CLAUDE.md — mark item 42 done.** In the "Done:" list (around line 195), find
  `7 (RelativePositionSensor2D/3D),` and append after it ` 42 (RelativePositionSensor multi-target +
  PositionSensor parity),` so the line reads `... 7 (RelativePositionSensor2D/3D), 42
  (RelativePositionSensor multi-target + PositionSensor parity), 8 (CameraSensor ...`.

- [ ] **Step 4: gap-analysis — flip the PositionSensor rows.** In
  `docs/godot-rl-gap-analysis-2026-06-02.md`:
  - Replace line 21's row with:
    `| \`PositionSensor2D/3D\` | ✅ multi-target \`Array[Node2D]\`, optional dir/dist split | ✅ multi-target \`objects_to_observe\`, both modes + axis toggles | ✅ done (#15) |`
  - Remove the priority-table row `| 🟡 P2 | \`RelativePositionSensor\` multi-target | #15 |`
    (line ~132) — the item is complete.

- [ ] **Step 5: BACKLOG — check item 42.** Replace the `42. ⬜ **...` line marker with `42. ✅`
  (change only the `⬜` to `✅` on the item-42 line ~232; leave the descriptive text).

- [ ] **Step 6: Verify docs reference no stale API.** Run:
  `grep -rn "target_path\|set_target_for_test" README.md CLAUDE.md docs/`
  Expected: no matches referencing the *RelativePosition* sensor (raycast/other sensors are unrelated;
  confirm none of the hits are about `RelativePositionSensor`). Fix any that remain.

- [ ] **Step 7: Commit**

```bash
git add README.md CLAUDE.md docs/godot-rl-gap-analysis-2026-06-02.md docs/BACKLOG.md
git commit -m "docs: RelativePositionSensor multi-target + PositionSensor parity (#15)"
```

---

## Task 6: PR

**Files:** none

- [ ] **Step 1: Re-fetch and rebase onto origin/main** (main moves fast in this repo):

Run: `git fetch origin && git rebase origin/main`
Expected: clean rebase (or resolve conflicts, re-run `./test/run_tests.sh`).

- [ ] **Step 2: Push and open the PR** with `Closes #15` in the body, summarizing: multi-target
  `objects_to_observe`, both encoding modes (`use_separate_direction`), per-axis include toggles,
  `max_distance` parity defaults, fixed `obs_size`, freed-target zero-fill; pure-core + wrapper tests;
  docs synced. Confirm `./test/run_tests.sh` is green in the PR description's test plan.

---

## Self-review notes

- **Spec coverage:** §Encoding both modes → Task 1; export API + class names + `obs_size` →
  Tasks 2–3; `max_distance` parity defaults/ranges → Tasks 2–3 (`@export_range(0.01, 20000.0)` 2D,
  `(0.01, 2500.0)` 3D, default `1.0`); freed-target zero-fill + fixed width → Tasks 2–3 tests; debug
  rendering omitted (non-goal, not implemented); docs checklist → Task 5; `Closes #15` → Task 6.
- **No backward-compat shim** for `target_path`/`set_target_for_test` (spec non-goal) — the rewritten
  tests are the only consumers, and the conformance test doesn't touch those members.
- **Type consistency:** `per_target_size(use_separate_direction, include_x, include_y, include_z=false)`
  and `encode_2d(world_offset, sensor_rotation, max_distance, use_separate_direction, include_x,
  include_y)` / `encode_3d(..., include_x, include_y, include_z)` are used with matching arity/names in
  every wrapper and test.
