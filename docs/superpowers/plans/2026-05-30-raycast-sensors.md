# RaycastSensor2D + RaycastSensor3D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `RaycastSensor2D` and `RaycastSensor3D` (godot_rl's most-used observation type) implementing the shared `get_observation() -> Array` / `obs_size() -> int` interface, feeding the native ncnn deployment path.

**Architecture:** A root-level `sensors/` module (mirrors `reward/`). Pure trig + normalization live in `sensors/raycast_math.gd` (static funcs, headless-unit-tested). Two thin node wrappers (`raycast_sensor_2d.gd` / `raycast_sensor_3d.gd`) isolate the physics cast behind an injectable seam (`set_cast_fn_for_test`) so the whole `get_observation()` path is testable without a ticking physics world. Per-ray encoding is **closeness**: miss → `0.0`, hit → `1 − clamp(dist/len)`.

**Tech Stack:** Godot 4.6 GDScript (TAB indentation), headless `extends SceneTree` test harness (`test/harness.gd`), references via `preload` (no bare `class_name` in headless runs).

**Spec:** `docs/superpowers/specs/2026-05-30-raycast-sensors-design.md`

**Conventions reminder for every task:**
- GDScript uses **TAB** indentation.
- Tests are `extends SceneTree` with `func _initialize()`, use `preload` for all script refs, end with `h.finish(self)`.
- New `test/unit/test_*.gd` files are auto-discovered by `test/run_tests.sh`.
- Run a single unit test with: `godot --headless --path . --script "res://test/unit/test_NAME.gd"` (the binary is `/opt/homebrew/bin/godot`; `godot` is on PATH).
- Float asserts: `h.assert_eq(a, b, label)` uses a `1e-6` tolerance when both args are floats. For `Vector2`/`Vector3` compare components or use `(a - b).length() < eps` inside `assert_true`.

---

## File structure

- **Create** `sensors/raycast_math.gd` — pure static helpers: `closeness`, `ray_directions_2d`, `ray_directions_3d`. `extends RefCounted`.
- **Create** `sensors/raycast_sensor_2d.gd` — `class_name RaycastSensor2D extends Node2D`. Thin physics wrapper + injectable cast.
- **Create** `sensors/raycast_sensor_3d.gd` — `class_name RaycastSensor3D extends Node3D`. Same pattern, 3D grid.
- **Create** `test/unit/test_raycast_math.gd` — pure-math unit tests (grown across Tasks 1–3).
- **Create** `test/unit/test_raycast_sensor_2d.gd` — sensor tests via injected stub caster.
- **Create** `test/unit/test_raycast_sensor_3d.gd` — sensor tests via injected stub caster.
- **Modify** `README.md` — add a short "Sensors" subsection.

---

## Task 1: `closeness` normalization (pure math)

**Files:**
- Create: `sensors/raycast_math.gd`
- Test: `test/unit/test_raycast_math.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_raycast_math.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RaycastMath = preload("res://sensors/raycast_math.gd")

func _initialize() -> void:
	var h := Harness.new()

	# --- closeness ---
	h.assert_eq(RaycastMath.closeness(-1.0, 100.0), 0.0, "miss (negative distance) -> 0")
	h.assert_eq(RaycastMath.closeness(0.0, 100.0), 1.0, "hit at origin -> 1")
	h.assert_eq(RaycastMath.closeness(50.0, 100.0), 0.5, "half distance -> 0.5")
	h.assert_eq(RaycastMath.closeness(100.0, 100.0), 0.0, "hit at max range -> 0")
	h.assert_eq(RaycastMath.closeness(200.0, 100.0), 0.0, "beyond range clamps to 0")
	h.assert_eq(RaycastMath.closeness(50.0, 0.0), 0.0, "zero ray_length guard -> 0")
	h.assert_eq(RaycastMath.closeness(50.0, -5.0), 0.0, "negative ray_length guard -> 0")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script "res://test/unit/test_raycast_math.gd"`
Expected: FAIL — script `res://sensors/raycast_math.gd` does not exist / parse error (cannot load preload).

- [ ] **Step 3: Write minimal implementation**

Create `sensors/raycast_math.gd`:

```gdscript
class_name RaycastMath
extends RefCounted

# Pure, stateless helpers for raycast sensors. No physics, no node state — fully
# unit-testable headlessly. Per-ray encoding is "closeness": a miss reads 0.0 and a
# hit reads 1 - clamp(distance / ray_length), so a near obstacle ~1.0 and a far one ~0.0.

static func closeness(distance: float, ray_length: float) -> float:
	if ray_length <= 0.0:
		return 0.0
	if distance < 0.0:
		return 0.0
	return clampf(1.0 - distance / ray_length, 0.0, 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script "res://test/unit/test_raycast_math.gd"`
Expected: PASS — `Results: 7 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add sensors/raycast_math.gd test/unit/test_raycast_math.gd
git commit -m "feat: raycast_math closeness normalization (pure)"
```

---

## Task 2: `ray_directions_2d` (pure math)

**Files:**
- Modify: `sensors/raycast_math.gd`
- Test: `test/unit/test_raycast_math.gd`

- [ ] **Step 1: Write the failing test**

Append these assertions to `test/unit/test_raycast_math.gd` **before** the `h.finish(self)` line:

```gdscript
	# --- ray_directions_2d ---
	h.assert_eq(RaycastMath.ray_directions_2d(0, 90.0, 0.0).size(), 0, "n_rays 0 -> empty")
	h.assert_eq(RaycastMath.ray_directions_2d(-3, 90.0, 0.0).size(), 0, "n_rays negative -> empty")

	var single: Array = RaycastMath.ray_directions_2d(1, 90.0, 0.0)
	h.assert_eq(single.size(), 1, "single ray -> 1 dir")
	h.assert_true((single[0] - Vector2(1.0, 0.0)).length() < 1e-5, "single ray at forward 0 points +X")

	var fan: Array = RaycastMath.ray_directions_2d(3, 90.0, 0.0)
	h.assert_eq(fan.size(), 3, "n_rays 3 -> 3 dirs")
	h.assert_true((fan[0] - Vector2.from_angle(-PI / 4.0)).length() < 1e-5, "fan start at forward - cone/2")
	h.assert_true((fan[1] - Vector2(1.0, 0.0)).length() < 1e-5, "fan middle at forward")
	h.assert_true((fan[2] - Vector2.from_angle(PI / 4.0)).length() < 1e-5, "fan end at forward + cone/2")

	var rotated: Array = RaycastMath.ray_directions_2d(1, 90.0, PI / 2.0)
	h.assert_true((rotated[0] - Vector2(0.0, 1.0)).length() < 1e-5, "forward PI/2 points +Y")

	var unit_ok := true
	for d in RaycastMath.ray_directions_2d(7, 120.0, 0.3):
		if absf(d.length() - 1.0) > 1e-5:
			unit_ok = false
	h.assert_true(unit_ok, "all 2D dirs are unit length")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script "res://test/unit/test_raycast_math.gd"`
Expected: FAIL — `Invalid call. Nonexistent function 'ray_directions_2d'` (or parse error).

- [ ] **Step 3: Write minimal implementation**

Add to `sensors/raycast_math.gd` (after `closeness`):

```gdscript
# Even fan of unit direction vectors across a cone centered on forward_radians.
# n_rays < 1 -> empty; n_rays == 1 -> single ray on forward; n_rays > 1 -> endpoints
# land exactly at forward +/- cone/2. Order runs from forward - cone/2 to forward + cone/2.
static func ray_directions_2d(n_rays: int, cone_degrees: float, forward_radians: float) -> Array:
	var dirs := []
	if n_rays < 1:
		return dirs
	if n_rays == 1:
		dirs.append(Vector2.from_angle(forward_radians))
		return dirs
	var cone := deg_to_rad(cone_degrees)
	var start := forward_radians - cone / 2.0
	var step := cone / float(n_rays - 1)
	for i in range(n_rays):
		dirs.append(Vector2.from_angle(start + step * float(i)))
	return dirs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script "res://test/unit/test_raycast_math.gd"`
Expected: PASS — all assertions pass (now 15+ passed, 0 failed).

- [ ] **Step 5: Commit**

```bash
git add sensors/raycast_math.gd test/unit/test_raycast_math.gd
git commit -m "feat: raycast_math ray_directions_2d fan geometry (pure)"
```

---

## Task 3: `ray_directions_3d` (pure math)

**Files:**
- Modify: `sensors/raycast_math.gd`
- Test: `test/unit/test_raycast_math.gd`

- [ ] **Step 1: Write the failing test**

Append to `test/unit/test_raycast_math.gd` **before** `h.finish(self)`:

```gdscript
	# --- ray_directions_3d ---
	h.assert_eq(RaycastMath.ray_directions_3d(0, 2, 90.0, 45.0).size(), 0, "n_w 0 -> empty")
	h.assert_eq(RaycastMath.ray_directions_3d(4, 0, 90.0, 45.0).size(), 0, "n_h 0 -> empty")

	var grid: Array = RaycastMath.ray_directions_3d(4, 2, 90.0, 45.0)
	h.assert_eq(grid.size(), 8, "4x2 grid -> 8 dirs")

	var center: Array = RaycastMath.ray_directions_3d(1, 1, 90.0, 45.0)
	h.assert_eq(center.size(), 1, "1x1 grid -> 1 dir")
	h.assert_true((center[0] - Vector3(0.0, 0.0, -1.0)).length() < 1e-5, "1x1 dir points forward -Z")

	var yaw_row: Array = RaycastMath.ray_directions_3d(3, 1, 90.0, 0.0)
	h.assert_eq(yaw_row.size(), 3, "3x1 -> 3 dirs")
	h.assert_true((yaw_row[1] - Vector3(0.0, 0.0, -1.0)).length() < 1e-5, "3x1 middle dir is forward")

	var unit3_ok := true
	for d in grid:
		if absf(d.length() - 1.0) > 1e-5:
			unit3_ok = false
	h.assert_true(unit3_ok, "all 3D dirs are unit length")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script "res://test/unit/test_raycast_math.gd"`
Expected: FAIL — `Invalid call. Nonexistent function 'ray_directions_3d'` (or parse error).

- [ ] **Step 3: Write minimal implementation**

Add to `sensors/raycast_math.gd` (after `ray_directions_2d`):

```gdscript
# Grid of unit direction vectors centered on forward (-Z, Godot's 3D forward).
# Yaw spreads across h_fov, pitch across v_fov, both endpoint-inclusive; a count of 1
# on an axis means zero offset (centered) on that axis. Order is row-major: pitch
# (height) outer, yaw (width) inner. n_w < 1 or n_h < 1 -> empty.
static func ray_directions_3d(n_w: int, n_h: int, h_fov_deg: float, v_fov_deg: float) -> Array:
	var dirs := []
	if n_w < 1 or n_h < 1:
		return dirs
	var h_fov := deg_to_rad(h_fov_deg)
	var v_fov := deg_to_rad(v_fov_deg)
	var yaw_start := 0.0 if n_w == 1 else -h_fov / 2.0
	var yaw_step := 0.0 if n_w == 1 else h_fov / float(n_w - 1)
	var pitch_start := 0.0 if n_h == 1 else -v_fov / 2.0
	var pitch_step := 0.0 if n_h == 1 else v_fov / float(n_h - 1)
	for hi in range(n_h):
		var pitch := pitch_start + pitch_step * float(hi)
		for wi in range(n_w):
			var yaw := yaw_start + yaw_step * float(wi)
			var d := Vector3(0.0, 0.0, -1.0)
			d = d.rotated(Vector3(1.0, 0.0, 0.0), pitch)
			d = d.rotated(Vector3(0.0, 1.0, 0.0), yaw)
			dirs.append(d)
	return dirs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script "res://test/unit/test_raycast_math.gd"`
Expected: PASS — all assertions pass.

- [ ] **Step 5: Commit**

```bash
git add sensors/raycast_math.gd test/unit/test_raycast_math.gd
git commit -m "feat: raycast_math ray_directions_3d grid geometry (pure)"
```

---

## Task 4: `RaycastSensor2D` node

**Files:**
- Create: `sensors/raycast_sensor_2d.gd`
- Test: `test/unit/test_raycast_sensor_2d.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_raycast_sensor_2d.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RaycastSensor2D = preload("res://sensors/raycast_sensor_2d.gd")
const RaycastMath = preload("res://sensors/raycast_math.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = RaycastSensor2D.new()
	s.n_rays = 4
	s.ray_length = 100.0
	s.cone_degrees = 90.0
	# Add to the tree so global_position / global_rotation resolve.
	get_root().add_child(s)

	# obs_size reflects n_rays
	h.assert_eq(s.obs_size(), 4, "obs_size == n_rays")

	# All-miss stub -> all zeros, length == n_rays
	var miss_fn := func(_o: Vector2, _d: Vector2) -> float:
		return -1.0
	s.set_cast_fn_for_test(miss_fn)
	var obs_miss: Array = s.get_observation()
	h.assert_eq(obs_miss.size(), 4, "obs length == n_rays")
	var all_zero := true
	for v in obs_miss:
		if absf(v) > 1e-6:
			all_zero = false
	h.assert_true(all_zero, "all-miss -> zeros")

	# Hit-at-origin stub (distance 0) -> all ones
	var hit_fn := func(_o: Vector2, _d: Vector2) -> float:
		return 0.0
	s.set_cast_fn_for_test(hit_fn)
	var obs_hit: Array = s.get_observation()
	var all_one := true
	for v in obs_hit:
		if absf(v - 1.0) > 1e-6:
			all_one = false
	h.assert_true(all_one, "hit at origin -> ones")

	# Half-distance stub -> 0.5 closeness
	var half_fn := func(_o: Vector2, _d: Vector2) -> float:
		return 50.0
	s.set_cast_fn_for_test(half_fn)
	var obs_half: Array = s.get_observation()
	h.assert_true(absf(obs_half[0] - 0.5) < 1e-6, "distance 50 / length 100 -> 0.5")

	# Directions passed to the cast match ray_directions_2d at rotation 0
	var recorded := []
	s.rotation = 0.0
	var record_fn := func(_o: Vector2, d: Vector2) -> float:
		recorded.append(d)
		return -1.0
	s.set_cast_fn_for_test(record_fn)
	s.get_observation()
	var expected: Array = RaycastMath.ray_directions_2d(4, 90.0, 0.0)
	var dirs_match := recorded.size() == expected.size()
	for i in range(mini(recorded.size(), expected.size())):
		if (recorded[i] - expected[i]).length() > 1e-5:
			dirs_match = false
	h.assert_true(dirs_match, "cast directions match ray_directions_2d at rotation 0")

	# n_rays < 1 -> empty obs + obs_size 0
	s.n_rays = 0
	h.assert_eq(s.get_observation().size(), 0, "n_rays 0 -> empty obs")
	h.assert_eq(s.obs_size(), 0, "n_rays 0 -> obs_size 0")

	s.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script "res://test/unit/test_raycast_sensor_2d.gd"`
Expected: FAIL — `res://sensors/raycast_sensor_2d.gd` does not exist (preload parse error).

- [ ] **Step 3: Write minimal implementation**

Create `sensors/raycast_sensor_2d.gd`:

```gdscript
class_name RaycastSensor2D
extends Node2D

# A fan of 2D rays emitting one "closeness" float each (see RaycastMath.closeness).
# The physics cast is isolated behind _cast_fn so the full observation path is testable
# headlessly via set_cast_fn_for_test. Composition into an agent's get_obs() is manual:
# call get_observation() and concatenate; obs_size() declares the contributed size.

const RaycastMath = preload("res://sensors/raycast_math.gd")

@export var n_rays: int = 8
@export var ray_length: float = 200.0
@export var cone_degrees: float = 90.0
@export_flags_2d_physics var collision_mask: int = 1
@export var collide_with_areas: bool = false
@export var collide_with_bodies: bool = true

# Test seam: a Callable(origin: Vector2, dir: Vector2) -> float returning hit distance,
# or a negative value for a miss. When null, the real physics query is used.
var _cast_fn = null

func set_cast_fn_for_test(fn: Callable) -> void:
	_cast_fn = fn

func obs_size() -> int:
	return maxi(n_rays, 0)

func get_observation() -> Array:
	if n_rays < 1:
		push_warning("RaycastSensor2D: n_rays < 1; returning empty observation.")
		return []
	if _cast_fn == null and get_world_2d() == null:
		push_error("RaycastSensor2D: no world_2d available and no injected cast; returning zeros.")
		var zeros := []
		zeros.resize(obs_size())
		zeros.fill(0.0)
		return zeros
	var dirs := RaycastMath.ray_directions_2d(n_rays, cone_degrees, global_rotation)
	var origin := global_position
	var out := []
	for dir in dirs:
		out.append(RaycastMath.closeness(_cast(origin, dir), ray_length))
	return out

func _cast(origin: Vector2, dir: Vector2) -> float:
	if _cast_fn != null:
		return _cast_fn.call(origin, dir)
	var world := get_world_2d()
	if world == null:
		return -1.0
	var to := origin + dir * ray_length
	var query := PhysicsRayQueryParameters2D.create(origin, to, collision_mask)
	query.collide_with_areas = collide_with_areas
	query.collide_with_bodies = collide_with_bodies
	var result := world.direct_space_state.intersect_ray(query)
	if result.is_empty():
		return -1.0
	return origin.distance_to(result.position)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script "res://test/unit/test_raycast_sensor_2d.gd"`
Expected: PASS — `Results: 8 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add sensors/raycast_sensor_2d.gd test/unit/test_raycast_sensor_2d.gd
git commit -m "feat: RaycastSensor2D node with injectable cast seam"
```

---

## Task 5: `RaycastSensor3D` node

**Files:**
- Create: `sensors/raycast_sensor_3d.gd`
- Test: `test/unit/test_raycast_sensor_3d.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_raycast_sensor_3d.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RaycastSensor3D = preload("res://sensors/raycast_sensor_3d.gd")
const RaycastMath = preload("res://sensors/raycast_math.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = RaycastSensor3D.new()
	s.n_rays_width = 4
	s.n_rays_height = 2
	s.ray_length = 20.0
	s.horizontal_fov = 90.0
	s.vertical_fov = 45.0
	get_root().add_child(s)

	# obs_size == n_w * n_h
	h.assert_eq(s.obs_size(), 8, "obs_size == n_w * n_h")

	# All-miss stub -> zeros, length == obs_size
	var miss_fn := func(_o: Vector3, _d: Vector3) -> float:
		return -1.0
	s.set_cast_fn_for_test(miss_fn)
	var obs_miss: Array = s.get_observation()
	h.assert_eq(obs_miss.size(), 8, "obs length == obs_size")
	var all_zero := true
	for v in obs_miss:
		if absf(v) > 1e-6:
			all_zero = false
	h.assert_true(all_zero, "all-miss -> zeros")

	# Hit-at-origin stub -> ones
	var hit_fn := func(_o: Vector3, _d: Vector3) -> float:
		return 0.0
	s.set_cast_fn_for_test(hit_fn)
	var obs_hit: Array = s.get_observation()
	var all_one := true
	for v in obs_hit:
		if absf(v - 1.0) > 1e-6:
			all_one = false
	h.assert_true(all_one, "hit at origin -> ones")

	# Half-distance stub -> 0.5
	var half_fn := func(_o: Vector3, _d: Vector3) -> float:
		return 10.0
	s.set_cast_fn_for_test(half_fn)
	var obs_half: Array = s.get_observation()
	h.assert_true(absf(obs_half[0] - 0.5) < 1e-6, "distance 10 / length 20 -> 0.5")

	# Directions match ray_directions_3d at identity transform
	var recorded := []
	var record_fn := func(_o: Vector3, d: Vector3) -> float:
		recorded.append(d)
		return -1.0
	s.set_cast_fn_for_test(record_fn)
	s.get_observation()
	var expected: Array = RaycastMath.ray_directions_3d(4, 2, 90.0, 45.0)
	var dirs_match := recorded.size() == expected.size()
	for i in range(mini(recorded.size(), expected.size())):
		if (recorded[i] - expected[i]).length() > 1e-5:
			dirs_match = false
	h.assert_true(dirs_match, "cast directions match ray_directions_3d at identity")

	# Degenerate counts -> empty
	s.n_rays_width = 0
	h.assert_eq(s.get_observation().size(), 0, "n_rays_width 0 -> empty obs")
	h.assert_eq(s.obs_size(), 0, "n_rays_width 0 -> obs_size 0")

	s.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script "res://test/unit/test_raycast_sensor_3d.gd"`
Expected: FAIL — `res://sensors/raycast_sensor_3d.gd` does not exist (preload parse error).

- [ ] **Step 3: Write minimal implementation**

Create `sensors/raycast_sensor_3d.gd`:

```gdscript
class_name RaycastSensor3D
extends Node3D

# A grid of 3D rays emitting one "closeness" float each (see RaycastMath.closeness).
# Mirrors RaycastSensor2D: the physics cast is isolated behind _cast_fn for headless
# testing via set_cast_fn_for_test. Composition into an agent's get_obs() is manual.

const RaycastMath = preload("res://sensors/raycast_math.gd")

@export var n_rays_width: int = 4
@export var n_rays_height: int = 2
@export var ray_length: float = 20.0
@export var horizontal_fov: float = 90.0
@export var vertical_fov: float = 45.0
@export_flags_3d_physics var collision_mask: int = 1
@export var collide_with_areas: bool = false
@export var collide_with_bodies: bool = true

# Test seam: a Callable(origin: Vector3, dir: Vector3) -> float returning hit distance,
# or a negative value for a miss. When null, the real physics query is used.
var _cast_fn = null

func set_cast_fn_for_test(fn: Callable) -> void:
	_cast_fn = fn

func obs_size() -> int:
	return maxi(n_rays_width, 0) * maxi(n_rays_height, 0)

func get_observation() -> Array:
	if n_rays_width < 1 or n_rays_height < 1:
		push_warning("RaycastSensor3D: n_rays_width/height < 1; returning empty observation.")
		return []
	if _cast_fn == null and get_world_3d() == null:
		push_error("RaycastSensor3D: no world_3d available and no injected cast; returning zeros.")
		var zeros := []
		zeros.resize(obs_size())
		zeros.fill(0.0)
		return zeros
	var dirs := RaycastMath.ray_directions_3d(n_rays_width, n_rays_height, horizontal_fov, vertical_fov)
	var origin := global_position
	var basis := global_transform.basis
	var out := []
	for local_dir in dirs:
		var world_dir: Vector3 = basis * local_dir
		out.append(RaycastMath.closeness(_cast(origin, world_dir), ray_length))
	return out

func _cast(origin: Vector3, dir: Vector3) -> float:
	if _cast_fn != null:
		return _cast_fn.call(origin, dir)
	var world := get_world_3d()
	if world == null:
		return -1.0
	var to := origin + dir * ray_length
	var query := PhysicsRayQueryParameters3D.create(origin, to, collision_mask)
	query.collide_with_areas = collide_with_areas
	query.collide_with_bodies = collide_with_bodies
	var result := world.direct_space_state.intersect_ray(query)
	if result.is_empty():
		return -1.0
	return origin.distance_to(result.position)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script "res://test/unit/test_raycast_sensor_3d.gd"`
Expected: PASS — `Results: 8 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add sensors/raycast_sensor_3d.gd test/unit/test_raycast_sensor_3d.gd
git commit -m "feat: RaycastSensor3D node with injectable cast seam"
```

---

## Task 6: Full-suite gate + README docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full suite (must be green before docs)**

Run: `./test/run_tests.sh`
Expected: ends with `All tests passed.` — including the unchanged trained-chase inference and golden-inference regression (the controller is untouched, so these must still pass).

If anything fails, STOP and fix before continuing (use the systematic-debugging skill).

- [ ] **Step 2: Add a Sensors section to the README**

Find the existing feature/component list in `README.md` and add this subsection (place it after the existing components overview; match surrounding heading style):

```markdown
### Sensors

Reusable observation sources implementing the shared sensor interface
(`get_observation() -> Array`, `obs_size() -> int`). Compose them manually inside your
agent's `get_obs()` and concatenate with your other features.

- **`RaycastSensor2D`** (`sensors/raycast_sensor_2d.gd`) — an even fan of `n_rays` 2D rays
  across `cone_degrees`, centered on the node's forward. Each ray emits a *closeness* float:
  `0.0` for no hit, up to `~1.0` for a near obstacle. Configurable `ray_length`,
  `collision_mask`, `collide_with_areas`, `collide_with_bodies`.
- **`RaycastSensor3D`** (`sensors/raycast_sensor_3d.gd`) — an `n_rays_width × n_rays_height`
  grid of 3D rays across `horizontal_fov × vertical_fov`, centered on forward (−Z). Same
  closeness encoding and physics options.

Pure ray geometry and normalization live in `sensors/raycast_math.gd` (headless-unit-tested).
This encoding matches `godot_rl`'s raycast convention, so ported environments behave the same —
and the observations feed `NcnnRunner` for zero-runtime deployment on mobile/web/console.
```

- [ ] **Step 3: Re-run the full suite to confirm docs change broke nothing**

Run: `./test/run_tests.sh`
Expected: `All tests passed.`

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document RaycastSensor2D/3D in README"
```

---

## Self-review notes (author)

- **Spec coverage:** closeness encoding (Task 1), 2D fan geometry + config (Tasks 2, 4), 3D grid geometry + config (Tasks 3, 5), injectable test seam (Tasks 4, 5), manual composition documented (Task 6 README), error handling for `n_rays < 1` / missing world (Tasks 4, 5), full-suite gate incl. trained-chase + golden (Task 6). Deferrals (real-physics scene, auto-discovery, class one-hot, addon move) intentionally have no task.
- **Type consistency:** `RaycastMath.closeness/ray_directions_2d/ray_directions_3d`, `set_cast_fn_for_test`, `get_observation`, `obs_size` names are identical across all tasks and tests. `_cast_fn` is the same seam name in both sensors.
- **Placeholders:** none — every code step shows complete code.
```
