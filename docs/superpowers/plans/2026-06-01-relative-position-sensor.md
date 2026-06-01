# RelativePositionSensor2D + RelativePositionSensor3D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship reusable `RelativePositionSensor2D`/`RelativePositionSensor3D` observation sources that emit an egocentric unit direction + clipped normalized distance to a target node, implementing the existing sensor interface (`get_observation() -> Array`, `obs_size() -> int`).

**Architecture:** Mirror the raycast sensors exactly — a pure static math core (`relative_position_math.gd`) holding all frame-rotation/normalization/clipping logic (headless-unit-testable, no tree/physics), plus two thin node wrappers (`Node2D`/`Node3D`) that resolve the target, compute the world-space offset, and delegate to the math. A `set_target_for_test()` seam (analogous to the raycast `set_cast_fn_for_test`) makes the wrappers headless-testable without tree-dependent `NodePath` resolution or `global_position`.

**Tech Stack:** GDScript (Godot 4.6), TAB indentation, dependency-free `extends SceneTree` test harness (`test/harness.gd`). All in-repo references via `preload` (no bare `class_name`, per project conventions).

**Spec:** `docs/superpowers/specs/2026-06-01-relative-position-sensor-design.md`

**Conventions reused from `addons/godot_native_rl/sensors/raycast_*`:**
- Run one unit test: `godot --headless --path . --script res://test/unit/test_NAME.gd`
  (the `godot` binary is `/opt/homebrew/bin/godot`; tests are auto-discovered by `test/run_tests.sh` from `test/unit/test_*.gd`).
- Float assertions: `h.assert_true(absf(actual - expected) < 1e-5, "label")` (the harness's `assert_eq` only auto-tolerances when *both* args are floats; array elements are compared one-by-one).
- 2D forward at `rotation = 0` is **+X** (`Vector2.from_angle(0) == (1, 0)`). 3D forward is **−Z** (Godot convention, matches `RaycastSensor3D`).

---

### Task 1: Pure math core — `encode_2d`

**Files:**
- Create: `addons/godot_native_rl/sensors/relative_position_math.gd`
- Test: `test/unit/test_relative_position_math.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_relative_position_math.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RelativePositionMath = preload("res://addons/godot_native_rl/sensors/relative_position_math.gd")

func _approx(h: Harness, out: Array, expected: Array, label: String) -> void:
	var ok := out.size() == expected.size()
	for i in range(mini(out.size(), expected.size())):
		if absf(float(out[i]) - float(expected[i])) > 1e-5:
			ok = false
	h.assert_true(ok, "%s (got %s, want %s)" % [label, str(out), str(expected)])

func _initialize() -> void:
	var h := Harness.new()

	# Target straight ahead (+X), unrotated sensor -> dir (1,0), dist 10/100
	_approx(h, RelativePositionMath.encode_2d(Vector2(10, 0), 0.0, 100.0), [1.0, 0.0, 0.1], "2d ahead, no rotation")

	# Sensor yawed +90deg: a world +X target reads as local (0,-1)
	_approx(h, RelativePositionMath.encode_2d(Vector2(10, 0), PI / 2.0, 100.0), [0.0, -1.0, 0.1], "2d rotation rotates direction")

	# Distance is rotation-invariant and clips at max_distance
	_approx(h, RelativePositionMath.encode_2d(Vector2(200, 0), 0.0, 100.0), [1.0, 0.0, 1.0], "2d distance clips to 1")
	_approx(h, RelativePositionMath.encode_2d(Vector2(50, 0), 0.0, 100.0), [1.0, 0.0, 0.5], "2d half distance -> 0.5")

	# Zero offset -> zero direction + zero distance
	_approx(h, RelativePositionMath.encode_2d(Vector2.ZERO, 0.0, 100.0), [0.0, 0.0, 0.0], "2d zero offset")

	# max_distance <= 0 -> dist_norm guarded to 0 (direction still valid)
	_approx(h, RelativePositionMath.encode_2d(Vector2(10, 0), 0.0, 0.0), [1.0, 0.0, 0.0], "2d max_distance 0 guard")

	# Direction is unit length for a non-axis-aligned offset
	var out: Array = RelativePositionMath.encode_2d(Vector2(3, 4), 0.0, 100.0)
	h.assert_true(absf(Vector2(out[0], out[1]).length() - 1.0) < 1e-5, "2d direction is unit length")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_relative_position_math.gd`
Expected: FAIL — `Could not find type "RelativePositionMath"` / parse error (file does not exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/sensors/relative_position_math.gd`:

```gdscript
class_name RelativePositionMath
extends RefCounted

# Pure, stateless helpers for relative-position sensors. No physics, no node state — fully
# unit-testable headlessly. Output is an EGOCENTRIC unit direction (the offset rotated into
# the sensor's local frame, then normalized) followed by a clipped, normalized distance:
# dist_norm = clamp(offset_length / max_distance, 0, 1). The direction is unit-length so
# bearing and distance are decoupled signals. Guards: a zero offset -> zero direction; a
# non-positive max_distance -> dist_norm 0.

static func _dist_norm(offset_length: float, max_distance: float) -> float:
	if max_distance <= 0.0:
		return 0.0
	return clampf(offset_length / max_distance, 0.0, 1.0)

# world_offset: target_pos - sensor_pos, in world space.
# sensor_rotation: the sensor node's world rotation (radians).
# Returns [dir_x, dir_y, dist_norm].
static func encode_2d(world_offset: Vector2, sensor_rotation: float, max_distance: float) -> Array:
	var local := world_offset.rotated(-sensor_rotation)
	var dir := local.normalized()  # Vector2.ZERO when local is zero-length
	return [dir.x, dir.y, _dist_norm(world_offset.length(), max_distance)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_relative_position_math.gd`
Expected: PASS — `Results: 7 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/relative_position_math.gd test/unit/test_relative_position_math.gd
git commit -m "feat: relative_position_math.encode_2d (egocentric dir + clipped distance)"
```

---

### Task 2: Pure math core — `encode_3d`

**Files:**
- Modify: `addons/godot_native_rl/sensors/relative_position_math.gd`
- Test: `test/unit/test_relative_position_math.gd` (add cases)

- [ ] **Step 1: Write the failing test**

In `test/unit/test_relative_position_math.gd`, insert these lines just before `h.finish(self)`:

```gdscript
	# 3D: target along -Z (forward) of an unrotated sensor -> dir (0,0,-1), dist 10/100
	_approx(h, RelativePositionMath.encode_3d(Vector3(0, 0, -10), Basis.IDENTITY, 100.0), [0.0, 0.0, -1.0, 0.1], "3d forward, no rotation")

	# Sensor yawed +90deg about Y: a world-forward (-Z) target reads as local +X
	var yaw := Basis(Vector3(0, 1, 0), PI / 2.0)
	_approx(h, RelativePositionMath.encode_3d(Vector3(0, 0, -10), yaw, 100.0), [1.0, 0.0, 0.0, 0.1], "3d yaw rotates direction")

	# Distance clips and is rotation-invariant
	_approx(h, RelativePositionMath.encode_3d(Vector3(0, 0, -200), Basis.IDENTITY, 100.0), [0.0, 0.0, -1.0, 1.0], "3d distance clips to 1")

	# Zero offset -> zeros; max_distance <= 0 -> dist guarded
	_approx(h, RelativePositionMath.encode_3d(Vector3.ZERO, Basis.IDENTITY, 100.0), [0.0, 0.0, 0.0, 0.0], "3d zero offset")
	_approx(h, RelativePositionMath.encode_3d(Vector3(0, 0, -10), Basis.IDENTITY, 0.0), [0.0, 0.0, -1.0, 0.0], "3d max_distance 0 guard")

	# Direction unit length for an arbitrary offset
	var out3: Array = RelativePositionMath.encode_3d(Vector3(1, 2, 2), Basis.IDENTITY, 100.0)
	h.assert_true(absf(Vector3(out3[0], out3[1], out3[2]).length() - 1.0) < 1e-5, "3d direction is unit length")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_relative_position_math.gd`
Expected: FAIL — `Invalid call. Nonexistent function 'encode_3d'`.

- [ ] **Step 3: Write minimal implementation**

Append to `addons/godot_native_rl/sensors/relative_position_math.gd`:

```gdscript
# world_offset: target_pos - sensor_pos, in world space.
# sensor_basis: the sensor node's world-transform basis.
# Returns [dir_x, dir_y, dir_z, dist_norm].
static func encode_3d(world_offset: Vector3, sensor_basis: Basis, max_distance: float) -> Array:
	var local := sensor_basis.inverse() * world_offset
	var dir := local.normalized()  # Vector3.ZERO when local is zero-length
	return [dir.x, dir.y, dir.z, _dist_norm(world_offset.length(), max_distance)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_relative_position_math.gd`
Expected: PASS — `Results: 13 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/relative_position_math.gd test/unit/test_relative_position_math.gd
git commit -m "feat: relative_position_math.encode_3d (egocentric dir + clipped distance)"
```

---

### Task 3: `RelativePositionSensor2D` node wrapper

**Files:**
- Create: `addons/godot_native_rl/sensors/relative_position_sensor_2d.gd`
- Test: `test/unit/test_relative_position_sensor_2d.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_relative_position_sensor_2d.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RelativePositionSensor2D = preload("res://addons/godot_native_rl/sensors/relative_position_sensor_2d.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = RelativePositionSensor2D.new()
	s.max_distance = 100.0

	# obs_size is fixed at 3
	h.assert_eq(s.obs_size(), 3, "obs_size == 3")

	# Target ahead (+X) of an unrotated, origin sensor -> [1, 0, 0.1]
	var target := Node2D.new()
	target.position = Vector2(10, 0)
	s.set_target_for_test(target)
	s.position = Vector2.ZERO
	s.rotation = 0.0
	var obs: Array = s.get_observation()
	h.assert_eq(obs.size(), 3, "observation length == obs_size")
	h.assert_true(absf(obs[0] - 1.0) < 1e-5 and absf(obs[1]) < 1e-5 and absf(obs[2] - 0.1) < 1e-5, "target ahead -> [1,0,0.1]")

	# Rotating the sensor +90deg rotates the egocentric direction to (0,-1)
	s.rotation = PI / 2.0
	var obs_rot: Array = s.get_observation()
	h.assert_true(absf(obs_rot[0]) < 1e-5 and absf(obs_rot[1] + 1.0) < 1e-5, "sensor rotation rotates direction")

	target.free()

	# No target -> zero-filled array of obs_size (no crash)
	var s2 = RelativePositionSensor2D.new()
	var obs_none: Array = s2.get_observation()
	h.assert_eq(obs_none.size(), 3, "no target -> length 3")
	var all_zero := true
	for v in obs_none:
		if absf(v) > 1e-6:
			all_zero = false
	h.assert_true(all_zero, "no target -> zeros")

	s.free()
	s2.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_relative_position_sensor_2d.gd`
Expected: FAIL — parse error / file does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/sensors/relative_position_sensor_2d.gd`:

```gdscript
class_name RelativePositionSensor2D
extends Node2D

# Egocentric relative-position observation for a target node: a unit direction (in the
# sensor's local frame) + a clipped, normalized distance. See RelativePositionMath.encode_2d.
# Mirrors the raycast sensors: pure math core + thin node wrapper, with target resolution
# isolated behind set_target_for_test so the full observation path is headless-testable.
# Composition into an agent's get_obs() is manual: call get_observation() and concatenate;
# obs_size() declares the contributed size.

const RelativePositionMath = preload("res://addons/godot_native_rl/sensors/relative_position_math.gd")

@export var target_path: NodePath
@export var max_distance: float = 1000.0

# Test seam: a target node injected directly, bypassing target_path resolution (which needs
# tree membership). When null, target_path is resolved via get_node_or_null.
var _target_override: Node2D = null
var _warned_no_target := false

func set_target_for_test(node: Node2D) -> void:
	_target_override = node

func obs_size() -> int:
	return 3

func get_observation() -> Array:
	var target: Node2D = _target_override if _target_override != null else get_node_or_null(target_path) as Node2D
	if target == null:
		if not _warned_no_target:
			push_error("RelativePositionSensor2D: target_path resolves to no Node2D; returning zeros.")
			_warned_no_target = true
		var zeros := []
		zeros.resize(obs_size())
		zeros.fill(0.0)
		return zeros
	_warned_no_target = false
	# Use the world transform when in the tree; fall back to the local transform when detached
	# (e.g. unit tests) so the path resolves without tree-dependent global_position errors.
	var sensor_xform := global_transform if is_inside_tree() else transform
	var target_pos := target.global_position if target.is_inside_tree() else target.position
	var world_offset := target_pos - sensor_xform.origin
	return RelativePositionMath.encode_2d(world_offset, sensor_xform.get_rotation(), max_distance)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_relative_position_sensor_2d.gd`
Expected: PASS — `Results: 6 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/relative_position_sensor_2d.gd test/unit/test_relative_position_sensor_2d.gd
git commit -m "feat: RelativePositionSensor2D (egocentric relative-position observation)"
```

---

### Task 4: `RelativePositionSensor3D` node wrapper

**Files:**
- Create: `addons/godot_native_rl/sensors/relative_position_sensor_3d.gd`
- Test: `test/unit/test_relative_position_sensor_3d.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_relative_position_sensor_3d.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RelativePositionSensor3D = preload("res://addons/godot_native_rl/sensors/relative_position_sensor_3d.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = RelativePositionSensor3D.new()
	s.max_distance = 100.0

	# obs_size is fixed at 4
	h.assert_eq(s.obs_size(), 4, "obs_size == 4")

	# Target along -Z (forward) of an unrotated, origin sensor -> [0,0,-1,0.1]
	var target := Node3D.new()
	target.position = Vector3(0, 0, -10)
	s.set_target_for_test(target)
	s.position = Vector3.ZERO
	s.rotation = Vector3.ZERO
	var obs: Array = s.get_observation()
	h.assert_eq(obs.size(), 4, "observation length == obs_size")
	h.assert_true(absf(obs[0]) < 1e-5 and absf(obs[1]) < 1e-5 and absf(obs[2] + 1.0) < 1e-5 and absf(obs[3] - 0.1) < 1e-5, "target forward -> [0,0,-1,0.1]")

	# Sensor yawed +90deg about Y rotates the forward target to local +X
	s.rotation = Vector3(0.0, PI / 2.0, 0.0)
	var obs_rot: Array = s.get_observation()
	h.assert_true(absf(obs_rot[0] - 1.0) < 1e-5 and absf(obs_rot[1]) < 1e-5 and absf(obs_rot[2]) < 1e-5, "sensor yaw rotates direction")

	target.free()

	# No target -> zero-filled array of obs_size (no crash)
	var s2 = RelativePositionSensor3D.new()
	var obs_none: Array = s2.get_observation()
	h.assert_eq(obs_none.size(), 4, "no target -> length 4")
	var all_zero := true
	for v in obs_none:
		if absf(v) > 1e-6:
			all_zero = false
	h.assert_true(all_zero, "no target -> zeros")

	s.free()
	s2.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_relative_position_sensor_3d.gd`
Expected: FAIL — parse error / file does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/sensors/relative_position_sensor_3d.gd`:

```gdscript
class_name RelativePositionSensor3D
extends Node3D

# Egocentric relative-position observation for a target node (3D): a unit direction in the
# sensor's local frame + a clipped, normalized distance. See RelativePositionMath.encode_3d.
# Mirrors RelativePositionSensor2D and the raycast sensors: pure math core + thin node
# wrapper, with target resolution isolated behind set_target_for_test for headless testing.

const RelativePositionMath = preload("res://addons/godot_native_rl/sensors/relative_position_math.gd")

@export var target_path: NodePath
@export var max_distance: float = 50.0

# Test seam: a target node injected directly, bypassing target_path resolution.
var _target_override: Node3D = null
var _warned_no_target := false

func set_target_for_test(node: Node3D) -> void:
	_target_override = node

func obs_size() -> int:
	return 4

func get_observation() -> Array:
	var target: Node3D = _target_override if _target_override != null else get_node_or_null(target_path) as Node3D
	if target == null:
		if not _warned_no_target:
			push_error("RelativePositionSensor3D: target_path resolves to no Node3D; returning zeros.")
			_warned_no_target = true
		var zeros := []
		zeros.resize(obs_size())
		zeros.fill(0.0)
		return zeros
	_warned_no_target = false
	# World transform when in the tree; local transform fallback when detached (unit tests).
	var sensor_xform := global_transform if is_inside_tree() else transform
	var target_pos := target.global_position if target.is_inside_tree() else target.position
	var world_offset := target_pos - sensor_xform.origin
	return RelativePositionMath.encode_3d(world_offset, sensor_xform.basis, max_distance)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_relative_position_sensor_3d.gd`
Expected: PASS — `Results: 6 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/relative_position_sensor_3d.gd test/unit/test_relative_position_sensor_3d.gd
git commit -m "feat: RelativePositionSensor3D (egocentric relative-position observation)"
```

---

### Task 5: Docs — README, CLAUDE.md, BACKLOG

**Files:**
- Modify: `README.md` (Sensors section, ~line 412–428)
- Modify: `CLAUDE.md` (sensors mention in "Current state")
- Modify: `docs/BACKLOG.md` (mark item 7 done; add a new follow-up item)

- [ ] **Step 1: Extend the README Sensors section**

In `README.md`, after the `RaycastSensor3D` bullet (the line ending `Same / closeness encoding and physics options.`) and before the `Pure ray geometry…` paragraph, add:

```markdown
- **`RelativePositionSensor2D`** (`sensors/relative_position_sensor_2d.gd`) — the egocentric
  position of a `target_path` node: a unit direction in the sensor's local frame plus a
  clipped, normalized distance, `[dir_x, dir_y, dist_norm]` (3 floats). `dist_norm =
  clamp(distance / max_distance, 0, 1)`. Answers "where is my target relative to me?"
  (`godot_rl` issue #177).
- **`RelativePositionSensor3D`** (`sensors/relative_position_sensor_3d.gd`) — the 3D form:
  `[dir_x, dir_y, dir_z, dist_norm]` (4 floats), direction in the sensor's local frame
  (forward = −Z), same `max_distance` clipping.
```

Then update the closing paragraph's first sentence to mention the new math file:

Change:
```markdown
Pure ray geometry and normalization live in `sensors/raycast_math.gd` (headless-unit-tested).
```
to:
```markdown
Pure ray geometry lives in `sensors/raycast_math.gd`; the relative-position frame/clip math
lives in `sensors/relative_position_math.gd` (both headless-unit-tested).
```

- [ ] **Step 2: Update CLAUDE.md sensors mention**

In `CLAUDE.md`, in the "Current state" bullet that lists the addon modules, update the `sensors/` description:

Change:
```markdown
`sensors/` (`RaycastSensor2D`/`RaycastSensor3D` + pure `raycast_math`),
```
to:
```markdown
`sensors/` (`RaycastSensor2D`/`RaycastSensor3D` + `RelativePositionSensor2D`/`RelativePositionSensor3D` + pure `raycast_math`/`relative_position_math`),
```

- [ ] **Step 3: Mark BACKLOG item 7 done + add follow-up item**

In `docs/BACKLOG.md`, replace the item 7 line:

```markdown
7. ⬜ **RelativePositionSensor** (godot_rl issue #177) — normalized direction + clipped distance.
```
with:
```markdown
7. ✅ **RelativePositionSensor2D + RelativePositionSensor3D** (godot_rl issue #177) — egocentric
   unit direction + clipped normalized distance to a `target_path` node.
   **Done 2026-06-01** — spec `docs/superpowers/specs/2026-06-01-relative-position-sensor-design.md`,
   plan `docs/superpowers/plans/2026-06-01-relative-position-sensor.md`. Shipped
   `addons/godot_native_rl/sensors/relative_position_math.gd` (pure `encode_2d`/`encode_3d`,
   headless-unit-tested) + `relative_position_sensor_2d.gd`/`_3d.gd` (thin node wrappers with a
   `set_target_for_test` seam). Output: 2D `[dir_x, dir_y, dist_norm]` (3 floats), 3D
   `[dir_x, dir_y, dir_z, dist_norm]` (4 floats); direction egocentric (sensor-local frame),
   `dist_norm = clamp(distance / max_distance, 0, 1)`. Manual composition (no controller change);
   missing target → stable zero-filled obs. Full suite green from a clean cache.
   **Deferred:** multi-target / tag selection + extra target properties (velocity) — issue #177
   extensions; sensor auto-discovery `collect_sensors()` (shared item-5 follow-up).
```

Then add a new item under the "Later" section (item 20's group), as item 32:

```markdown
32. ⬜ **Example using `RelativePositionSensor`** — a small 2D seek/navigate-to-target demo (or
    migrate the rover's inline goal obs onto `RelativePositionSensor3D` with a retrain), to show
    the sensor end-to-end and provide a trained regression. *(follow-up from item 7)*
```

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md docs/BACKLOG.md
git commit -m "docs: document RelativePositionSensor2D/3D; mark backlog item 7 done"
```

---

### Task 6: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite from a clean class cache**

Run:
```bash
rm -f .godot/global_script_class_cache.cfg && ./test/run_tests.sh
```
Expected: ends with `All tests passed.` — including the three new unit tests
(`test_relative_position_math`, `test_relative_position_sensor_2d`, `test_relative_position_sensor_3d`)
and the unchanged trained-chase, trained-rover, and golden-inference regressions.

- [ ] **Step 2: Confirm no stray generated files**

Run: `git status --short && git clean -n -- '*.gd.uid'`
Expected: only intended files; if any `*.gd.uid` files appear, `git clean -f -- '*.gd.uid'` and re-check (per the CLAUDE.md gotcha).

- [ ] **Step 3: (Optional) Finishing the branch**

Use the `superpowers:finishing-a-development-branch` skill to decide merge/PR. Do **not** push to `main` directly (project convention).
```
