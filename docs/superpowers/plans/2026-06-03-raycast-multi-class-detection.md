# RaycastSensor Multi-Class Detection (`class_sensor`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `class_sensor` mode to `RaycastSensor2D`/`RaycastSensor3D` that encodes *what* each ray hit (by collision layer) as a multi-hot class segment + optional `other` slot + optional distance slot, with the existing distance-only behavior unchanged when off.

**Architecture:** All encoding lives in one pure, dimension-agnostic helper `RaycastMath.encode_ray_class(...)` (headless-unit-testable, no physics). Each sensor node gains four exports, a class-aware cast (`_cast_class -> {distance, layer}`) with its own test seam, and a branch in `get_observation()` that concatenates the per-ray segment. The non-class code path and its `set_cast_fn_for_test` seam are untouched.

**Tech Stack:** GDScript (Godot 4.6), TAB indentation, dependency-free headless harness at `test/harness.gd`. Tests are `extends SceneTree` scripts auto-discovered by `test/run_tests.sh` (glob `test/unit/test_*.gd`).

**Spec:** `docs/superpowers/specs/2026-06-03-raycast-multi-class-detection-design.md`

## Key conventions (read before starting)

- **GDScript uses TAB indentation** — match the existing files exactly.
- Sensors reference helpers via `preload` consts and **path-based `extends`** (never bare `class_name`) — the global class cache is unreliable headless. The files already follow this; don't change it.
- **Layer numbering:** `detection_classes` entries are **1-based collision-layer numbers**. An object "on layer L" has `collision_layer` bit `1 << (L - 1)` set. So layer 2 → value `2`, layer 3 → value `4`, layer 5 → value `16`.
- **Harness `assert_eq` on Arrays** uses `==` (exact element compare). All distances in tests are chosen so closeness is an exact binary fraction (e.g. `closeness(10, 20) == 0.5`, `closeness(0, n) == 1.0`).
- **Run a single test:** `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/<file>.gd`
- **Run the full suite (final gate):** `./test/run_tests.sh`

## File Structure

- **Modify** `addons/godot_native_rl/sensors/raycast_math.gd` — add the pure `encode_ray_class(...)` (the only new logic; the node wrappers stay thin).
- **Modify** `addons/godot_native_rl/sensors/raycast_sensor_3d.gd` — 4 exports, `_cast_class`, `set_class_cast_fn_for_test`, `obs_size()` branch, `get_observation()` branch.
- **Modify** `addons/godot_native_rl/sensors/raycast_sensor_2d.gd` — same set of changes, 2D variants.
- **Modify** `test/unit/test_raycast_math.gd` — add `encode_ray_class` cases.
- **Modify** `test/unit/test_raycast_sensor_3d.gd` — add class-mode cases + off-regression.
- **Modify** `test/unit/test_raycast_sensor_2d.gd` — add class-mode cases + off-regression.
- **Modify** `CLAUDE.md`, `docs/BACKLOG.md`, and (if it documents sensor encodings) `docs/DEVELOPMENT.md`.

---

### Task 1: Pure `encode_ray_class` in `raycast_math.gd`

**Files:**
- Modify: `addons/godot_native_rl/sensors/raycast_math.gd`
- Test: `test/unit/test_raycast_math.gd`

- [ ] **Step 1: Write the failing tests**

In `test/unit/test_raycast_math.gd`, insert this block immediately **before** the final `h.finish(self)` line (use TABs):

```gdscript
	# --- encode_ray_class ---
	var classes := [2, 3, 5]   # 1-based layers -> bit values 2, 4, 16

	# miss -> all zeros; segment len = n_classes + other + distance = 5
	var enc_miss: Array = RaycastMath.encode_ray_class(-1.0, 0, 20.0, classes, true, true)
	h.assert_eq(enc_miss.size(), 5, "encode: segment len = n_classes + other + distance")
	var enc_miss_zero := true
	for v in enc_miss:
		if absf(v) > 1e-6:
			enc_miss_zero = false
	h.assert_true(enc_miss_zero, "encode: miss -> all zeros")

	# hit on layer 3 (bit value 4) at half distance -> middle class slot + closeness 0.5
	var enc_c3: Array = RaycastMath.encode_ray_class(10.0, 4, 20.0, classes, true, true)
	h.assert_eq(enc_c3, [0.0, 1.0, 0.0, 0.0, 0.5], "encode: layer-3 hit -> middle slot + closeness")

	# multi-layer hit: object on layers 2 AND 5 (bit values 2 | 16 = 18), at origin
	var enc_multi: Array = RaycastMath.encode_ray_class(0.0, 18, 20.0, classes, true, true)
	h.assert_eq(enc_multi, [1.0, 0.0, 1.0, 0.0, 1.0], "encode: multi-layer -> multi-hot, other 0")

	# unlisted-layer hit: layer 4 (bit value 8), not in classes -> other slot lit
	var enc_other: Array = RaycastMath.encode_ray_class(0.0, 8, 20.0, classes, true, true)
	h.assert_eq(enc_other, [0.0, 0.0, 0.0, 1.0, 1.0], "encode: unlisted layer -> other slot")

	# include_other off: unlisted-layer hit -> class slots zero + distance only
	var enc_no_other: Array = RaycastMath.encode_ray_class(0.0, 8, 20.0, classes, false, true)
	h.assert_eq(enc_no_other, [0.0, 0.0, 0.0, 1.0], "encode: include_other off -> no other slot")

	# include_distance off: layer-2 hit -> classes + other, no closeness slot
	var enc_no_dist: Array = RaycastMath.encode_ray_class(10.0, 2, 20.0, classes, true, false)
	h.assert_eq(enc_no_dist, [1.0, 0.0, 0.0, 0.0], "encode: include_distance off -> no closeness slot")

	# both flags off: class slots only
	var enc_only: Array = RaycastMath.encode_ray_class(10.0, 2, 20.0, classes, false, false)
	h.assert_eq(enc_only, [1.0, 0.0, 0.0], "encode: both flags off -> class slots only")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_raycast_math.gd`
Expected: parse/runtime error or FAILs — `encode_ray_class` is not defined yet (`Invalid call. Nonexistent function 'encode_ray_class'`).

- [ ] **Step 3: Write the minimal implementation**

In `addons/godot_native_rl/sensors/raycast_math.gd`, append this function at the end of the file (after `ray_directions_3d`, use TABs):

```gdscript
# Per-ray class/distance segment for class_sensor mode. Pure, no physics. A miss is
# hit_distance < 0 (matching closeness()). Segment order:
#   [ class_0 .. class_{n-1}, (other), (closeness) ]
# Each class slot is 1.0 when the ray hit AND the hit collider's layer bitmask has that
# layer's bit set (multi-hot — several may be 1.0); detection_classes entries are 1-based
# layer numbers (layer L -> bit 1 << (L - 1)). The optional 'other' slot is 1.0 when the
# ray hit but matched no listed class. The optional closeness slot is closeness(distance).
static func encode_ray_class(
		hit_distance: float, hit_layer: int, ray_length: float,
		detection_classes: Array, include_other: bool, include_distance: bool) -> Array:
	var seg := []
	var hit := hit_distance >= 0.0
	var matched_any := false
	for class_layer in detection_classes:
		var li := int(class_layer)
		var matched := hit and li >= 1 and (hit_layer & (1 << (li - 1))) != 0
		seg.append(1.0 if matched else 0.0)
		if matched:
			matched_any = true
	if include_other:
		seg.append(1.0 if (hit and not matched_any) else 0.0)
	if include_distance:
		seg.append(closeness(hit_distance, ray_length))
	return seg
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_raycast_math.gd`
Expected: all PASS, `0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/raycast_math.gd test/unit/test_raycast_math.gd
git commit -m "feat: add pure RaycastMath.encode_ray_class for class_sensor encoding"
```

---

### Task 2: `class_sensor` mode on `RaycastSensor3D`

**Files:**
- Modify: `addons/godot_native_rl/sensors/raycast_sensor_3d.gd`
- Test: `test/unit/test_raycast_sensor_3d.gd`

- [ ] **Step 1: Write the failing tests**

In `test/unit/test_raycast_sensor_3d.gd`, insert this block immediately **before** the existing `# Degenerate counts -> empty` comment (use TABs). It uses a fresh sensor instance so the earlier `s` rotation/state doesn't leak in:

```gdscript
	# --- class_sensor mode ---
	var cs = RaycastSensor3D.new()
	cs.n_rays_width = 2
	cs.n_rays_height = 1
	cs.ray_length = 20.0
	cs.class_sensor = true
	cs.detection_classes = [2, 3]    # bit values 2 and 4
	cs.include_other = true
	cs.include_distance = true
	# per ray = 2 classes + other + distance = 4; 2 rays -> 8
	h.assert_eq(cs.obs_size(), 8, "class obs_size = n_rays * (n_classes + other + distance)")

	# ray 0 hits layer 3 (bit value 4) at half distance; ray 1 misses
	var cs_calls := {"n": 0}
	var cs_class_fn := func(_o: Vector3, _d: Vector3) -> Dictionary:
		cs_calls["n"] += 1
		if cs_calls["n"] == 1:
			return {"distance": 10.0, "layer": 4}
		return {"distance": -1.0, "layer": 0}
	cs.set_class_cast_fn_for_test(cs_class_fn)
	var cs_obs: Array = cs.get_observation()
	h.assert_eq(cs_obs.size(), 8, "class obs length == obs_size")
	h.assert_eq(cs_obs, [0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], "class obs: ray0 layer-3 hit, ray1 miss")

	# class_sensor off -> distance-only path is byte-identical to before
	cs.class_sensor = false
	var cs_half_fn := func(_o: Vector3, _d: Vector3) -> float:
		return 10.0
	cs.set_cast_fn_for_test(cs_half_fn)
	h.assert_eq(cs.get_observation(), [0.5, 0.5], "class_sensor=false -> distance-only path unchanged")
	cs.free()

```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_raycast_sensor_3d.gd`
Expected: error/FAIL — `class_sensor` / `set_class_cast_fn_for_test` don't exist yet (`Invalid set index 'class_sensor'` or `Nonexistent function`).

- [ ] **Step 3: Write the minimal implementation**

In `addons/godot_native_rl/sensors/raycast_sensor_3d.gd`:

(a) Add the four exports right after the existing `@export var collide_with_bodies: bool = true` line:

```gdscript
@export var class_sensor: bool = false
@export var detection_classes: Array[int] = []
@export var include_other: bool = true
@export var include_distance: bool = true
```

(b) Add a class-cast seam field next to the existing `var _cast_fn = null` line:

```gdscript
var _class_cast_fn = null
```

(c) Add the test-seam setter right after the existing `set_cast_fn_for_test`:

```gdscript
func set_class_cast_fn_for_test(fn: Callable) -> void:
	_class_cast_fn = fn
```

(d) Replace the whole `obs_size()` function with:

```gdscript
func obs_size() -> int:
	var n_rays := maxi(n_rays_width, 0) * maxi(n_rays_height, 0)
	if not class_sensor:
		return n_rays
	var per_ray := detection_classes.size()
	if include_other:
		per_ray += 1
	if include_distance:
		per_ray += 1
	return n_rays * per_ray
```

(e) In `get_observation()`, change the no-cast guard condition from
`if _cast_fn == null and get_world_3d() == null:` to
`if _cast_fn == null and _class_cast_fn == null and get_world_3d() == null:`,
then replace the final ray loop (the `for local_dir in dirs:` block and its `return out`) with:

```gdscript
	for local_dir in dirs:
		var world_dir: Vector3 = basis * local_dir
		if class_sensor:
			var hit: Dictionary = _cast_class(origin, world_dir)
			out.append_array(RaycastMath.encode_ray_class(
				hit.get("distance", -1.0), hit.get("layer", 0), ray_length,
				detection_classes, include_other, include_distance))
		else:
			out.append(RaycastMath.closeness(_cast(origin, world_dir), ray_length))
	return out
```

(f) Add the class-aware cast right after the existing `_cast` function:

```gdscript
func _cast_class(origin: Vector3, dir: Vector3) -> Dictionary:
	if _class_cast_fn != null:
		return _class_cast_fn.call(origin, dir)
	var world := get_world_3d()
	if world == null:
		return {"distance": -1.0, "layer": 0}
	var space := world.direct_space_state
	if space == null:
		return {"distance": -1.0, "layer": 0}
	var to := origin + dir * ray_length
	var query := PhysicsRayQueryParameters3D.create(origin, to, collision_mask)
	query.collide_with_areas = collide_with_areas
	query.collide_with_bodies = collide_with_bodies
	var result := space.intersect_ray(query)
	if result.is_empty():
		return {"distance": -1.0, "layer": 0}
	var collider = result.collider
	var layer := 0
	if collider != null:
		layer = collider.collision_layer
	return {"distance": origin.distance_to(result.position), "layer": layer}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_raycast_sensor_3d.gd`
Expected: all PASS, `0 failed` (including the pre-existing distance-only cases).

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/raycast_sensor_3d.gd test/unit/test_raycast_sensor_3d.gd
git commit -m "feat: add class_sensor multi-class detection to RaycastSensor3D"
```

---

### Task 3: `class_sensor` mode on `RaycastSensor2D`

**Files:**
- Modify: `addons/godot_native_rl/sensors/raycast_sensor_2d.gd`
- Test: `test/unit/test_raycast_sensor_2d.gd`

- [ ] **Step 1: Write the failing tests**

In `test/unit/test_raycast_sensor_2d.gd`, insert this block immediately **before** the existing `# n_rays < 1 -> empty obs + obs_size 0` comment (use TABs):

```gdscript
	# --- class_sensor mode ---
	var cs = RaycastSensor2D.new()
	cs.n_rays = 2
	cs.ray_length = 100.0
	cs.cone_degrees = 90.0
	cs.class_sensor = true
	cs.detection_classes = [2, 3]    # bit values 2 and 4
	cs.include_other = true
	cs.include_distance = true
	# per ray = 2 classes + other + distance = 4; 2 rays -> 8
	h.assert_eq(cs.obs_size(), 8, "class obs_size = n_rays * (n_classes + other + distance)")

	# ray 0 hits layer 3 (bit value 4) at half distance; ray 1 misses
	var cs_calls := {"n": 0}
	var cs_class_fn := func(_o: Vector2, _d: Vector2) -> Dictionary:
		cs_calls["n"] += 1
		if cs_calls["n"] == 1:
			return {"distance": 50.0, "layer": 4}
		return {"distance": -1.0, "layer": 0}
	cs.set_class_cast_fn_for_test(cs_class_fn)
	var cs_obs: Array = cs.get_observation()
	h.assert_eq(cs_obs.size(), 8, "class obs length == obs_size")
	h.assert_eq(cs_obs, [0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], "class obs: ray0 layer-3 hit, ray1 miss")

	# class_sensor off -> distance-only path is byte-identical to before
	cs.class_sensor = false
	var cs_half_fn := func(_o: Vector2, _d: Vector2) -> float:
		return 50.0
	cs.set_cast_fn_for_test(cs_half_fn)
	h.assert_eq(cs.get_observation(), [0.5, 0.5], "class_sensor=false -> distance-only path unchanged")
	cs.free()

```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_raycast_sensor_2d.gd`
Expected: error/FAIL — `class_sensor` / `set_class_cast_fn_for_test` don't exist yet.

- [ ] **Step 3: Write the minimal implementation**

In `addons/godot_native_rl/sensors/raycast_sensor_2d.gd`:

(a) Add the four exports right after the existing `@export var collide_with_bodies: bool = true` line:

```gdscript
@export var class_sensor: bool = false
@export var detection_classes: Array[int] = []
@export var include_other: bool = true
@export var include_distance: bool = true
```

(b) Add a class-cast seam field next to the existing `var _cast_fn = null` line:

```gdscript
var _class_cast_fn = null
```

(c) Add the test-seam setter right after the existing `set_cast_fn_for_test`:

```gdscript
func set_class_cast_fn_for_test(fn: Callable) -> void:
	_class_cast_fn = fn
```

(d) Replace the whole `obs_size()` function with:

```gdscript
func obs_size() -> int:
	var n := maxi(n_rays, 0)
	if not class_sensor:
		return n
	var per_ray := detection_classes.size()
	if include_other:
		per_ray += 1
	if include_distance:
		per_ray += 1
	return n * per_ray
```

(e) In `get_observation()`, change the no-cast guard condition from
`if _cast_fn == null and get_world_2d() == null:` to
`if _cast_fn == null and _class_cast_fn == null and get_world_2d() == null:`,
then replace the final ray loop (the `for dir in dirs:` block and its `return out`) with:

```gdscript
	for dir in dirs:
		if class_sensor:
			var hit: Dictionary = _cast_class(origin, dir)
			out.append_array(RaycastMath.encode_ray_class(
				hit.get("distance", -1.0), hit.get("layer", 0), ray_length,
				detection_classes, include_other, include_distance))
		else:
			out.append(RaycastMath.closeness(_cast(origin, dir), ray_length))
	return out
```

(f) Add the class-aware cast right after the existing `_cast` function (at the end of the file):

```gdscript
func _cast_class(origin: Vector2, dir: Vector2) -> Dictionary:
	if _class_cast_fn != null:
		return _class_cast_fn.call(origin, dir)
	var world := get_world_2d()
	if world == null:
		return {"distance": -1.0, "layer": 0}
	var space := world.direct_space_state
	if space == null:
		return {"distance": -1.0, "layer": 0}
	var to := origin + dir * ray_length
	var query := PhysicsRayQueryParameters2D.create(origin, to, collision_mask)
	query.collide_with_areas = collide_with_areas
	query.collide_with_bodies = collide_with_bodies
	var result := space.intersect_ray(query)
	if result.is_empty():
		return {"distance": -1.0, "layer": 0}
	var collider = result.collider
	var layer := 0
	if collider != null:
		layer = collider.collision_layer
	return {"distance": origin.distance_to(result.position), "layer": layer}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_raycast_sensor_2d.gd`
Expected: all PASS, `0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/raycast_sensor_2d.gd test/unit/test_raycast_sensor_2d.gd
git commit -m "feat: add class_sensor multi-class detection to RaycastSensor2D"
```

---

### Task 4: Docs + full-suite gate

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/BACKLOG.md`
- Modify: `docs/DEVELOPMENT.md` (only if it documents sensor obs encodings)

- [ ] **Step 1: Update `CLAUDE.md` sensors blurb**

In the `sensors/` description (the parenthetical listing `RaycastSensor2D/RaycastSensor3D` …), note the new mode. Find the text `RaycastSensor2D`/`RaycastSensor3D` and append after the raycast mention:

```
(both support an opt-in class_sensor mode: per-ray multi-hot collision-layer segments via
detection_classes + optional other/closeness slots, encoded by pure raycast_math.encode_ray_class)
```

Keep the edit terse — CLAUDE.md is always-loaded.

- [ ] **Step 2: Mark backlog item 41 done**

In `docs/BACKLOG.md`, change item 41's leading `⬜` to `✅` and append a short completion note in the same style as neighboring done items (e.g. item 40), including that **2D was added alongside 3D** and pointing at the spec/plan:

```
**Done 2026-06-03** — both RaycastSensor2D and RaycastSensor3D. Opt-in `class_sensor`:
`detection_classes` (1-based layer numbers) → per-ray multi-hot class slots + optional `other`
catch-all + optional `closeness`, all encoded by pure `RaycastMath.encode_ray_class`. New
`_cast_class`/`set_class_cast_fn_for_test` seam; distance-only path unchanged when off. Spec
`docs/superpowers/specs/2026-06-03-raycast-multi-class-detection-design.md`, plan
`docs/superpowers/plans/2026-06-03-raycast-multi-class-detection.md`.
```

Also update the "Done:" roll-up list near the top of `docs/BACKLOG.md` (and the matching line in `CLAUDE.md`'s "Done:" enumeration) to include `41`.

- [ ] **Step 3: Check `docs/DEVELOPMENT.md` for sensor-encoding docs**

Run: `grep -n "RaycastSensor\|closeness\|per-ray\|sensor" docs/DEVELOPMENT.md`
If a section documents the raycast obs encoding, add a short paragraph describing `class_sensor` (multi-hot layer slots + `other` + `closeness`, controlled by the four exports). If there's no such section, skip — do not invent one.

- [ ] **Step 4: Run the full test suite**

Run: `./test/run_tests.sh`
Expected: the suite regenerates the script-class cache, then all GDScript unit tests + Python tests + smoke/golden/rover tests pass — final line shows no failures. The three modified unit tests (`test_raycast_math.gd`, `test_raycast_sensor_2d.gd`, `test_raycast_sensor_3d.gd`) are green.

If anything fails, fix before committing (do NOT edit tests to pass — fix the implementation, per the project testing rule).

- [ ] **Step 5: Clean stray uid files and commit**

```bash
git clean -f -- '*.gd.uid'
git add CLAUDE.md docs/BACKLOG.md docs/DEVELOPMENT.md
git commit -m "docs: mark backlog item 41 done (RaycastSensor class_sensor, 2D+3D)"
```

(If `docs/DEVELOPMENT.md` was not changed in Step 3, drop it from the `git add`.)

---

## Self-Review notes

- **Spec coverage:** §"Public API" → Tasks 2(a)/3(a); §"Encoding"/`encode_ray_class` → Task 1; §"Cast seam change" → Tasks 2(c,f)/3(c,f); §`obs_size()` + degenerate guard → Tasks 2(d)/3(d) (empty `detection_classes` with both flags off yields `per_ray == 0` → empty obs, and the existing `n_rays < 1` guard still returns `[]`); §"Testing" → Tasks 1/2/3; §"Scope & docs" (both 2D+3D, doc updates) → Tasks 3 + 4. All sections mapped.
- **Type consistency:** the cast dictionary keys are `"distance"`/`"layer"` everywhere; the helper is `RaycastMath.encode_ray_class(hit_distance, hit_layer, ray_length, detection_classes, include_other, include_distance)` with the same arg order in the helper definition (Task 1) and both call sites (Tasks 2e/3e); the seam is `set_class_cast_fn_for_test` / field `_class_cast_fn` in both sensors.
- **Layer-value arithmetic** in tests is consistent with `1 << (L - 1)`: layer 2→`2`, layer 3→`4`, layer 5→`16`; multi `2|16 = 18`; unlisted layer 4→`8`. Verified against expected arrays.
