# GridSensor2D + GridSensor3D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add query-based, godot_rl-parity `GridSensor2D` + `GridSensor3D` cell-detection sensors (backlog item 11), backed by a shared pure math helper, fully headless-tested.

**Architecture:** One pure RefCounted math helper (`grid_sensor_math.gd`) holds all encoding logic (collision-layer mapping, cell offsets, flat-buffer indexing, count accumulation). Two thin node wrappers (`grid_sensor_2d.gd` Node2D, `grid_sensor_3d.gd` Node3D) query the physics space per cell behind an injectable seam and delegate encoding to the math helper. Detection is computed fresh each `get_observation()` call (immutable), mirroring the existing `RaycastSensor2D/3D` split.

**Tech Stack:** Godot 4.6 GDScript (TAB indentation), headless `SceneTree` test harness at `res://test/harness.gd`, auto-discovered by `test/run_tests.sh` (glob `test/unit/test_*.gd`).

**Reference spec:** `docs/superpowers/specs/2026-06-03-grid-sensor-design.md`

**Conventions (from CLAUDE.md):**
- TAB indentation in GDScript.
- Use path-based `preload`/`extends` (`res://addons/godot_native_rl/...`), NOT bare `class_name` bases — the global class cache is unreliable headless.
- Pure helpers + thin node wrappers; small focused files.
- Godot 4.6 `:=` can't infer from an untyped value — annotate `var xs: Array = ...` explicitly.
- Each unit test is `extends SceneTree`, preloads `Harness`, runs in `_initialize()`, ends with `h.finish(self)`.

---

## File Structure

- Create: `addons/godot_native_rl/sensors/grid_sensor_math.gd` — pure encoding (RefCounted).
- Create: `addons/godot_native_rl/sensors/grid_sensor_2d.gd` — Node2D wrapper.
- Create: `addons/godot_native_rl/sensors/grid_sensor_3d.gd` — Node3D wrapper.
- Create: `test/unit/test_grid_sensor_math.gd` — math unit tests.
- Create: `test/unit/test_grid_sensor_2d.gd` — 2D wrapper tests (injected seam).
- Create: `test/unit/test_grid_sensor_3d.gd` — 3D wrapper tests (injected seam).
- Modify: `CLAUDE.md` — add grid sensors to the `sensors/` description.
- Modify: `docs/BACKLOG.md` — mark item 11 ✅.
- Modify: `README.md` — add grid sensors if sensors are enumerated.

---

## Task 1: Pure math helper — collision mapping, sizes, indexing

**Files:**
- Create: `addons/godot_native_rl/sensors/grid_sensor_math.gd`
- Test: `test/unit/test_grid_sensor_math.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_grid_sensor_math.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const GridSensorMath = preload("res://addons/godot_native_rl/sensors/grid_sensor_math.gd")

func _initialize() -> void:
	var h := Harness.new()

	# collision_mapping: each set bit -> sequential slot
	var m1: Dictionary = GridSensorMath.collision_mapping(0b101)
	h.assert_eq(m1.size(), 2, "mask 0b101 -> 2 layers")
	h.assert_eq(m1[0], 0, "bit 0 -> slot 0")
	h.assert_eq(m1[2], 1, "bit 2 -> slot 1")
	h.assert_eq(GridSensorMath.collision_mapping(0).size(), 0, "mask 0 -> empty mapping")

	# n_layers
	h.assert_eq(GridSensorMath.n_layers(0b101), 2, "n_layers(0b101) == 2")
	h.assert_eq(GridSensorMath.n_layers(0), 0, "n_layers(0) == 0")
	h.assert_eq(GridSensorMath.n_layers(1), 1, "n_layers(1) == 1")

	# obs_size = grid_a * grid_b * n_layers
	h.assert_eq(GridSensorMath.obs_size(3, 3, 1), 9, "3x3x1 -> 9")
	h.assert_eq(GridSensorMath.obs_size(3, 3, 0b101), 18, "3x3x2 -> 18")
	h.assert_eq(GridSensorMath.obs_size(0, 3, 1), 0, "zero grid -> 0")
	h.assert_eq(GridSensorMath.obs_size(3, 3, 0), 0, "zero mask -> 0")

	# obs_index: (i*grid_b*n) + (j*n) + slot
	h.assert_eq(GridSensorMath.obs_index(0, 0, 0, 3, 1), 0, "cell 0,0 slot 0")
	h.assert_eq(GridSensorMath.obs_index(1, 0, 0, 3, 1), 3, "cell 1,0 -> 3")
	h.assert_eq(GridSensorMath.obs_index(0, 2, 0, 3, 1), 2, "cell 0,2 -> 2")
	h.assert_eq(GridSensorMath.obs_index(1, 2, 1, 3, 2), 11, "cell 1,2 slot1 n2 -> 11")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_grid_sensor_math.gd`
Expected: parse/load error or FAIL — `grid_sensor_math.gd` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/sensors/grid_sensor_math.gd`:

```gdscript
class_name GridSensorMath
extends RefCounted

# Pure, stateless helpers for grid sensors. No physics, no node state — fully
# unit-testable headlessly. Encoding matches godot_rl's GridSensor: one float per
# active detection-layer bit per cell = count of overlapping objects on that layer.
# grid_a/grid_b are the two grid axes (2D: x,y; 3D: x,z). Buffer index = i*grid_b+j.

# Each set bit of detection_mask -> a sequential obs slot, low bit first.
static func collision_mapping(detection_mask: int) -> Dictionary:
	var mapping := {}
	var total := 0
	for i in range(32):
		if (detection_mask & (1 << i)) != 0:
			mapping[i] = total
			total += 1
	return mapping

static func n_layers(detection_mask: int) -> int:
	return collision_mapping(detection_mask).size()

static func obs_size(grid_a: int, grid_b: int, detection_mask: int) -> int:
	return maxi(grid_a, 0) * maxi(grid_b, 0) * n_layers(detection_mask)

# Flat-buffer index for a cell's layer slot (godot_rl formula).
static func obs_index(cell_i: int, cell_j: int, layer_slot: int, grid_b: int, layers: int) -> int:
	return (cell_i * grid_b * layers) + (cell_j * layers) + layer_slot
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_grid_sensor_math.gd`
Expected: PASS for all assertions, `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/grid_sensor_math.gd test/unit/test_grid_sensor_math.gd
git commit -m "feat: GridSensorMath mapping/size/index helpers (backlog 11)"
```

---

## Task 2: Pure math helper — cell_offsets + build_obs

**Files:**
- Modify: `addons/godot_native_rl/sensors/grid_sensor_math.gd`
- Test: `test/unit/test_grid_sensor_math.gd:_initialize` (extend before `h.finish`)

- [ ] **Step 1: Write the failing test**

In `test/unit/test_grid_sensor_math.gd`, insert before `h.finish(self)`:

```gdscript
	# cell_offsets: odd grid is symmetric about origin (shift = -(grid/2)*step)
	var off3: Array = GridSensorMath.cell_offsets(3, 3, 10.0, 10.0)
	h.assert_eq(off3.size(), 9, "3x3 -> 9 offsets")
	# i outer, j inner: index 0 = (i0,j0); shift = (-10,-10) since 3/2==1
	h.assert_true((off3[0] - Vector2(-10.0, -10.0)).length() < 1e-5, "cell 0,0 at (-10,-10)")
	# center cell (i1,j1) at index 1*3+1 = 4 sits on origin
	h.assert_true(off3[4].length() < 1e-5, "center cell at origin")
	# even grid offset by half-cell: 2/2==1 -> shift -(1)*10 = -10
	var off2: Array = GridSensorMath.cell_offsets(2, 2, 10.0, 10.0)
	h.assert_eq(off2.size(), 4, "2x2 -> 4 offsets")
	h.assert_true((off2[0] - Vector2(-10.0, -10.0)).length() < 1e-5, "even cell 0,0 at (-10,-10)")
	# asymmetric steps
	var offab: Array = GridSensorMath.cell_offsets(1, 2, 5.0, 7.0)
	h.assert_eq(offab.size(), 2, "1x2 -> 2 offsets")

	# build_obs: empty cells -> all zeros
	var empty_cells: Array = [[], [], [], [], [], [], [], [], []]
	var ob0: Array = GridSensorMath.build_obs(empty_cells, 3, 3, 1)
	h.assert_eq(ob0.size(), 9, "build_obs length == obs_size")
	var all_zero := true
	for v in ob0:
		if absf(v) > 1e-9:
			all_zero = false
	h.assert_true(all_zero, "empty -> zeros")

	# build_obs: one object on layer bit 0 in cell index 4 (i1,j1) -> count 1 at obs_index 4
	var cells1: Array = [[], [], [], [], [0b1], [], [], [], []]
	var ob1: Array = GridSensorMath.build_obs(cells1, 3, 3, 1)
	h.assert_eq(ob1[4], 1.0, "single hit -> count 1 at index 4")

	# build_obs: two objects same cell+layer -> count accumulates
	var cells2: Array = [[0b1, 0b1], [], [], [], [], [], [], [], []]
	var ob2: Array = GridSensorMath.build_obs(cells2, 3, 3, 1)
	h.assert_eq(ob2[0], 2.0, "two objects -> count 2")

	# build_obs: object on two mapped layers -> increments both slots
	var cells3: Array = [[0b101], [], [], [], [], [], [], [], []]
	var ob3: Array = GridSensorMath.build_obs(cells3, 3, 3, 0b101)
	h.assert_eq(ob3[0], 1.0, "slot 0 incremented")
	h.assert_eq(ob3[1], 1.0, "slot 1 incremented")

	# build_obs: layer bit outside the mask is ignored
	var cells4: Array = [[0b10], [], [], [], [], [], [], [], []]
	var ob4: Array = GridSensorMath.build_obs(cells4, 3, 3, 0b1)
	var ignored_zero := true
	for v in ob4:
		if absf(v) > 1e-9:
			ignored_zero = false
	h.assert_true(ignored_zero, "out-of-mask layer ignored")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_grid_sensor_math.gd`
Expected: FAIL — `cell_offsets`/`build_obs` not defined (or runtime error on the nil return).

- [ ] **Step 3: Write minimal implementation**

Append to `addons/godot_native_rl/sensors/grid_sensor_math.gd`:

```gdscript
# Local cell-center offsets (Vector2) relative to the sensor origin. Integer-division
# shift matches godot_rl: odd grids are symmetric about origin, even grids offset by
# half a cell. Order is i outer / j inner -> element index = i*grid_b + j.
static func cell_offsets(grid_a: int, grid_b: int, step_a: float, step_b: float) -> Array:
	var offsets: Array = []
	if grid_a < 1 or grid_b < 1:
		return offsets
	var shift_a := -float(grid_a / 2) * step_a
	var shift_b := -float(grid_b / 2) * step_b
	for i in range(grid_a):
		for j in range(grid_b):
			offsets.append(Vector2(float(i) * step_a + shift_a, float(j) * step_b + shift_b))
	return offsets

# Build the flat float observation buffer from per-cell overlapping collision layers.
# cell_layers: flat Array (len grid_a*grid_b, index i*grid_b+j) of Array[int] of the
# collision_layer values overlapping each cell. Returns a fresh Array of floats.
static func build_obs(cell_layers: Array, grid_a: int, grid_b: int, detection_mask: int) -> Array:
	var mapping := collision_mapping(detection_mask)
	var layers := mapping.size()
	var out: Array = []
	out.resize(maxi(grid_a, 0) * maxi(grid_b, 0) * layers)
	out.fill(0.0)
	if layers == 0 or grid_a < 1 or grid_b < 1:
		return out
	for i in range(grid_a):
		for j in range(grid_b):
			var cell_index := i * grid_b + j
			if cell_index >= cell_layers.size():
				continue
			var layers_here: Array = cell_layers[cell_index]
			for collision_layer in layers_here:
				for bit in mapping:
					if (collision_layer & (1 << bit)) != 0:
						out[obs_index(i, j, mapping[bit], grid_b, layers)] += 1.0
	return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_grid_sensor_math.gd`
Expected: PASS, `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/grid_sensor_math.gd test/unit/test_grid_sensor_math.gd
git commit -m "feat: GridSensorMath cell_offsets + build_obs encoding (backlog 11)"
```

---

## Task 3: GridSensor2D node wrapper

**Files:**
- Create: `addons/godot_native_rl/sensors/grid_sensor_2d.gd`
- Test: `test/unit/test_grid_sensor_2d.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_grid_sensor_2d.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const GridSensor2D = preload("res://addons/godot_native_rl/sensors/grid_sensor_2d.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = GridSensor2D.new()
	s.grid_size_x = 3
	s.grid_size_y = 3
	s.cell_width = 10.0
	s.cell_height = 10.0
	s.detection_mask = 1

	# obs_size reflects grid * layers
	h.assert_eq(s.obs_size(), 9, "obs_size == 3*3*1")

	# Empty-overlap stub -> all zeros, length == obs_size
	var empty_fn := func(_c: Vector2, _sz: Vector2) -> Array:
		return []
	s.set_overlap_fn_for_test(empty_fn)
	var obs_empty: Array = s.get_observation()
	h.assert_eq(obs_empty.size(), 9, "obs length == obs_size")
	var all_zero := true
	for v in obs_empty:
		if absf(v) > 1e-9:
			all_zero = false
	h.assert_true(all_zero, "empty overlap -> zeros")

	# Stub that reports a layer-1 hit only for the cell nearest origin -> count 1 there
	var centers: Array = []
	var hit_center_fn := func(c: Vector2, _sz: Vector2) -> Array:
		centers.append(c)
		if c.length() < 1e-5:
			return [0b1]
		return []
	s.set_overlap_fn_for_test(hit_center_fn)
	var obs_hit: Array = s.get_observation()
	# center cell is i1,j1 -> index 4
	h.assert_eq(obs_hit[4], 1.0, "hit at center cell -> count 1 at index 4")
	h.assert_eq(centers.size(), 9, "queried 9 cells")

	# Cell centers translate with node position
	centers.clear()
	s.position = Vector2(100.0, 0.0)
	s.set_overlap_fn_for_test(hit_center_fn)
	s.get_observation()
	var found_translated := false
	for c in centers:
		if (c - Vector2(100.0, 0.0)).length() < 1e-5:
			found_translated = true
	h.assert_true(found_translated, "center cell shifted to node position")

	# Degenerate grid -> empty obs + obs_size 0
	s.grid_size_x = 0
	h.assert_eq(s.get_observation().size(), 0, "grid 0 -> empty obs")
	h.assert_eq(s.obs_size(), 0, "grid 0 -> obs_size 0")

	s.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_grid_sensor_2d.gd`
Expected: FAIL — `grid_sensor_2d.gd` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/sensors/grid_sensor_2d.gd`:

```gdscript
class_name GridSensor2D
extends Node2D

# A grid of 2D cells emitting per-cell, per-layer overlap counts (see GridSensorMath).
# Query-based: each get_observation() queries the physics space fresh and builds a new
# buffer immutably. The physics query is isolated behind _overlap_fn so the full
# observation path is testable headlessly via set_overlap_fn_for_test. Composition into
# an agent's get_obs() is manual: call get_observation() and concatenate.

const GridSensorMath = preload("res://addons/godot_native_rl/sensors/grid_sensor_math.gd")

@export_flags_2d_physics var detection_mask: int = 1
@export var collide_with_areas: bool = false
@export var collide_with_bodies: bool = true
@export var cell_width: float = 20.0
@export var cell_height: float = 20.0
@export var grid_size_x: int = 3
@export var grid_size_y: int = 3

# Test seam: a Callable(cell_center: Vector2, cell_size: Vector2) -> Array of overlapping
# collision_layer ints. When null, the real physics query is used.
var _overlap_fn = null
var _warned_degenerate := false

func set_overlap_fn_for_test(fn: Callable) -> void:
	_overlap_fn = fn

func obs_size() -> int:
	return GridSensorMath.obs_size(grid_size_x, grid_size_y, detection_mask)

func get_observation() -> Array:
	if grid_size_x < 1 or grid_size_y < 1 or GridSensorMath.n_layers(detection_mask) == 0:
		if not _warned_degenerate:
			push_warning("GridSensor2D: empty grid or detection_mask; returning empty observation.")
			_warned_degenerate = true
		return []
	_warned_degenerate = false
	if _overlap_fn == null and get_world_2d() == null:
		push_error("GridSensor2D: no world_2d available and no injected overlap; returning zeros.")
		var zeros: Array = []
		zeros.resize(obs_size())
		zeros.fill(0.0)
		return zeros
	var offsets: Array = GridSensorMath.cell_offsets(grid_size_x, grid_size_y, cell_width, cell_height)
	var xform := global_transform if is_inside_tree() else transform
	var size := Vector2(cell_width, cell_height)
	var cell_layers: Array = []
	for offset in offsets:
		var center: Vector2 = xform * offset
		cell_layers.append(_overlap(center, size))
	return GridSensorMath.build_obs(cell_layers, grid_size_x, grid_size_y, detection_mask)

func _overlap(center: Vector2, size: Vector2) -> Array:
	if _overlap_fn != null:
		return _overlap_fn.call(center, size)
	var world := get_world_2d()
	if world == null:
		return []
	var space := world.direct_space_state
	if space == null:
		return []
	var shape := RectangleShape2D.new()
	shape.size = size
	var params := PhysicsShapeQueryParameters2D.new()
	params.shape = shape
	params.transform = Transform2D(global_rotation, center)
	params.collision_mask = detection_mask
	params.collide_with_areas = collide_with_areas
	params.collide_with_bodies = collide_with_bodies
	var results := space.intersect_shape(params, 32)
	var layers: Array = []
	for r in results:
		var collider = r.get("collider")
		if collider != null and "collision_layer" in collider:
			layers.append(collider.collision_layer)
	return layers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_grid_sensor_2d.gd`
Expected: PASS, `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/grid_sensor_2d.gd test/unit/test_grid_sensor_2d.gd
git commit -m "feat: GridSensor2D query-based cell sensor (backlog 11)"
```

---

## Task 4: GridSensor3D node wrapper

**Files:**
- Create: `addons/godot_native_rl/sensors/grid_sensor_3d.gd`
- Test: `test/unit/test_grid_sensor_3d.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_grid_sensor_3d.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const GridSensor3D = preload("res://addons/godot_native_rl/sensors/grid_sensor_3d.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = GridSensor3D.new()
	s.grid_size_x = 3
	s.grid_size_z = 3
	s.cell_width = 2.0
	s.cell_height = 2.0
	s.detection_mask = 1

	# obs_size reflects grid_x * grid_z * layers
	h.assert_eq(s.obs_size(), 9, "obs_size == 3*3*1")

	# Empty-overlap stub -> zeros, length == obs_size
	var empty_fn := func(_c: Vector3, _sz: Vector3) -> Array:
		return []
	s.set_overlap_fn_for_test(empty_fn)
	var obs_empty: Array = s.get_observation()
	h.assert_eq(obs_empty.size(), 9, "obs length == obs_size")
	var all_zero := true
	for v in obs_empty:
		if absf(v) > 1e-9:
			all_zero = false
	h.assert_true(all_zero, "empty overlap -> zeros")

	# Hit only at the cell nearest origin -> count 1 at center index 4
	var centers: Array = []
	var hit_center_fn := func(c: Vector3, _sz: Vector3) -> Array:
		centers.append(c)
		if c.length() < 1e-5:
			return [0b1]
		return []
	s.set_overlap_fn_for_test(hit_center_fn)
	var obs_hit: Array = s.get_observation()
	h.assert_eq(obs_hit[4], 1.0, "hit at center cell -> count 1 at index 4")
	h.assert_eq(centers.size(), 9, "queried 9 cells")

	# Cells live on the X/Z plane (y == 0 for all centers)
	var all_y_zero := true
	for c in centers:
		if absf(c.y) > 1e-5:
			all_y_zero = false
	h.assert_true(all_y_zero, "cells on X/Z plane (y==0)")

	# Degenerate grid -> empty obs + obs_size 0
	s.grid_size_x = 0
	h.assert_eq(s.get_observation().size(), 0, "grid 0 -> empty obs")
	h.assert_eq(s.obs_size(), 0, "grid 0 -> obs_size 0")

	s.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_grid_sensor_3d.gd`
Expected: FAIL — `grid_sensor_3d.gd` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/sensors/grid_sensor_3d.gd`:

```gdscript
class_name GridSensor3D
extends Node3D

# A grid of 3D box cells on the X/Z plane emitting per-cell, per-layer overlap counts
# (see GridSensorMath). Mirrors GridSensor2D: query-based, physics isolated behind
# _overlap_fn for headless testing via set_overlap_fn_for_test. cell_width is the grid
# step on BOTH X and Z; cell_height is only the box's Y extent. collide_with_bodies
# defaults false (godot_rl note: StaticBody3D needs an Area to be detected).

const GridSensorMath = preload("res://addons/godot_native_rl/sensors/grid_sensor_math.gd")

@export_flags_3d_physics var detection_mask: int = 1
@export var collide_with_areas: bool = false
@export var collide_with_bodies: bool = false
@export var cell_width: float = 1.0
@export var cell_height: float = 1.0
@export var grid_size_x: int = 3
@export var grid_size_z: int = 3

# Test seam: a Callable(cell_center: Vector3, cell_size: Vector3) -> Array of overlapping
# collision_layer ints. When null, the real physics query is used.
var _overlap_fn = null
var _warned_degenerate := false

func set_overlap_fn_for_test(fn: Callable) -> void:
	_overlap_fn = fn

func obs_size() -> int:
	return GridSensorMath.obs_size(grid_size_x, grid_size_z, detection_mask)

func get_observation() -> Array:
	if grid_size_x < 1 or grid_size_z < 1 or GridSensorMath.n_layers(detection_mask) == 0:
		if not _warned_degenerate:
			push_warning("GridSensor3D: empty grid or detection_mask; returning empty observation.")
			_warned_degenerate = true
		return []
	_warned_degenerate = false
	if _overlap_fn == null and get_world_3d() == null:
		push_error("GridSensor3D: no world_3d available and no injected overlap; returning zeros.")
		var zeros: Array = []
		zeros.resize(obs_size())
		zeros.fill(0.0)
		return zeros
	# cell_width is the step on both grid axes; planar offsets map (x,y) -> (x,0,y).
	var offsets: Array = GridSensorMath.cell_offsets(grid_size_x, grid_size_z, cell_width, cell_width)
	var xform := global_transform if is_inside_tree() else transform
	var size := Vector3(cell_width, cell_height, cell_width)
	var cell_layers: Array = []
	for offset in offsets:
		var local := Vector3(offset.x, 0.0, offset.y)
		var center: Vector3 = xform * local
		cell_layers.append(_overlap(center, size))
	return GridSensorMath.build_obs(cell_layers, grid_size_x, grid_size_z, detection_mask)

func _overlap(center: Vector3, size: Vector3) -> Array:
	if _overlap_fn != null:
		return _overlap_fn.call(center, size)
	var world := get_world_3d()
	if world == null:
		return []
	var space := world.direct_space_state
	if space == null:
		return []
	var shape := BoxShape3D.new()
	shape.size = size
	var params := PhysicsShapeQueryParameters3D.new()
	params.shape = shape
	params.transform = Transform3D(global_transform.basis, center)
	params.collision_mask = detection_mask
	params.collide_with_areas = collide_with_areas
	params.collide_with_bodies = collide_with_bodies
	var results := space.intersect_shape(params, 32)
	var layers: Array = []
	for r in results:
		var collider = r.get("collider")
		if collider != null and "collision_layer" in collider:
			layers.append(collider.collision_layer)
	return layers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_grid_sensor_3d.gd`
Expected: PASS, `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/grid_sensor_3d.gd test/unit/test_grid_sensor_3d.gd
git commit -m "feat: GridSensor3D query-based cell sensor (backlog 11)"
```

---

## Task 5: Full suite green + docs

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/BACKLOG.md`
- Modify: `README.md` (only if it enumerates sensors)

- [ ] **Step 1: Run the full test suite from a clean cache**

Run:
```bash
rm -f .godot/global_script_class_cache.cfg
./test/run_tests.sh
```
Expected: all tests pass, including the three new `test_grid_sensor_*` files (run_tests.sh
self-heals the class cache, then globs `test/unit/test_*.gd`). If it hangs, see the fresh-clone
trap in CLAUDE.md (the cache regen pass).

- [ ] **Step 2: Update CLAUDE.md sensors description**

In `CLAUDE.md`, in the `sensors/` bullet of the "Current state" section, add the grid sensors
alongside the existing list, e.g. after the `CameraSensor` mention add:
`+ GridSensor2D/GridSensor3D (cell-based spatial detection, per-layer overlap counts, query-based
+ pure grid_sensor_math)`.

- [ ] **Step 3: Mark backlog item 11 done**

In `docs/BACKLOG.md`, change line 162 from:
```
11. ⬜ **GridSensor2D + GridSensor3D** — cell-based spatial detection. *(roadmap spec Track A.3)*
```
to:
```
11. ✅ **GridSensor2D + GridSensor3D** — cell-based spatial detection. *(roadmap spec Track A.3)*
    **Done 2026-06-03** — spec `docs/superpowers/specs/2026-06-03-grid-sensor-design.md`, plan
    `docs/superpowers/plans/2026-06-03-grid-sensor.md`. Query-based (fresh each call, immutable),
    per-layer count encoding (godot_rl-parity index layout), shared pure `grid_sensor_math.gd`
    (collision mapping, cell_offsets, build_obs) + thin `grid_sensor_2d.gd`/`grid_sensor_3d.gd`
    wrappers with an injectable overlap seam. Headless unit tests for math + both wrappers.
```
Also update the "Done:" summary line near the top of the backlog file (the one listing completed
item numbers) to include `11`.

- [ ] **Step 4: Update README sensor list (if present)**

Run: `grep -n -i "RaycastSensor\|RelativePositionSensor\|CameraSensor" README.md`
If a sensor list exists, add `GridSensor2D`/`GridSensor3D` (cell-based spatial detection) in the
same style. If no sensor enumeration exists, skip this step.

- [ ] **Step 5: Clean stray uid files and commit docs**

```bash
git clean -f -- '*.gd.uid'
git add CLAUDE.md docs/BACKLOG.md README.md
git commit -m "docs: document GridSensor2D/3D, mark backlog 11 done"
```

---

## Self-Review notes

- **Spec coverage:** math (mapping/n_layers/obs_size/obs_index/cell_offsets/build_obs) → Tasks 1–2;
  GridSensor2D wrapper + seam + degenerate guards → Task 3; GridSensor3D (X/Z plane, box, body
  default false) → Task 4; testing → each task's tests; docs → Task 5. All spec sections covered.
- **Type consistency:** `collision_mapping`, `n_layers`, `obs_size(grid_a, grid_b, mask)`,
  `obs_index(i, j, slot, grid_b, layers)`, `cell_offsets(grid_a, grid_b, step_a, step_b)`,
  `build_obs(cell_layers, grid_a, grid_b, mask)`, `set_overlap_fn_for_test`, `obs_size()`,
  `get_observation()` are used identically across tasks. 2D uses `grid_size_y`, 3D `grid_size_z`.
- **No placeholders:** every code step shows complete code; every run step gives the exact command
  and expected result.
- **Headless safety:** all cross-file references use path-based `preload`; tests are `extends
  SceneTree` + `Harness`; suite run from a clean cache per CLAUDE.md.
