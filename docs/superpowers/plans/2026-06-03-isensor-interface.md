# ISensor2D / ISensor3D Interface + collect_sensors() Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lightweight `ISensor2D`/`ISensor3D` base interfaces, retrofit the six flat sensors onto them, and add `NcnnControllerCore.collect_sensors(root)` (+ a controller convenience method) so agents auto-discover child sensors instead of hand-concatenating the obs vector (backlog item 40).

**Architecture:** Two tiny interface scripts (`get_observation() -> Array`, `obs_size() -> int`) that the flat sensors `extends` **by path** (the global class-name cache is unreliable headless). A static `collect_sensors(root)` on `NcnnControllerCore` recursively gathers flat sensors via **duck typing** (`has_method`), concatenating their observations in scene-tree order; `CameraSensor` (no `obs_size`) is skipped. A thin instance method on `NcnnAIController2D/3D` forwards to the static helper.

**Tech Stack:** Godot 4.6 GDScript (TAB indentation), headless `SceneTree` test harness at `res://test/harness.gd` (`Harness.new()`, `h.assert_eq(actual, expected, label)`, `h.assert_true(cond, label)`, `h.finish(self)`), auto-discovered by `test/run_tests.sh` (glob `test/unit/test_*.gd`).

**Reference spec:** `docs/superpowers/specs/2026-06-03-isensor-interface-design.md`

**Conventions (from CLAUDE.md):**
- TAB indentation in GDScript.
- Use path-based `preload`/`extends` (`res://addons/godot_native_rl/...`), NOT bare `class_name` bases.
- Godot 4.6 `:=` can't infer from an untyped value — annotate `var xs: Array = ...` explicitly.
- Each unit test is `extends SceneTree`, preloads `Harness`, runs in `_initialize()`, ends with `h.finish(self)`.
- GDScript `is` works against a **preloaded script const** (`node is SomePreloadedScript`) and follows the `extends` chain — used here to verify the retrofit headlessly without the class registry.

---

## File Structure

- Create: `addons/godot_native_rl/sensors/i_sensor_2d.gd` — `ISensor2D` base (Node2D).
- Create: `addons/godot_native_rl/sensors/i_sensor_3d.gd` — `ISensor3D` base (Node3D).
- Modify: `addons/godot_native_rl/controllers/ncnn_controller_core.gd` — add `collect_sensors` + `_gather_sensor_obs`.
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` — add `collect_sensors()` convenience.
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd` — add `collect_sensors()` convenience.
- Modify (6, retrofit `extends`): `sensors/{raycast,relative_position,grid}_sensor_{2,3}d.gd`.
- Create: `test/unit/test_i_sensor.gd` — interface base-stub tests.
- Create: `test/unit/test_collect_sensors.gd` — discovery/ordering/skip tests (mock tree).
- Create: `test/unit/test_sensor_interface_conformance.gd` — real sensors `is` the interface (retrofit proof).
- Create: `test/unit/test_controller_collect_sensors.gd` — controller convenience method.
- Modify: `CLAUDE.md`, `README.md`, `docs/BACKLOG.md`.

---

## Task 1: Interface base scripts

**Files:**
- Create: `addons/godot_native_rl/sensors/i_sensor_2d.gd`
- Create: `addons/godot_native_rl/sensors/i_sensor_3d.gd`
- Test: `test/unit/test_i_sensor.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_i_sensor.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const ISensor2D = preload("res://addons/godot_native_rl/sensors/i_sensor_2d.gd")
const ISensor3D = preload("res://addons/godot_native_rl/sensors/i_sensor_3d.gd")

func _initialize() -> void:
	var h := Harness.new()

	var s2 = ISensor2D.new()
	h.assert_eq(s2.get_observation(), [], "ISensor2D base get_observation -> []")
	h.assert_eq(s2.obs_size(), 0, "ISensor2D base obs_size -> 0")
	h.assert_true(s2 is Node2D, "ISensor2D is a Node2D")
	s2.free()

	var s3 = ISensor3D.new()
	h.assert_eq(s3.get_observation(), [], "ISensor3D base get_observation -> []")
	h.assert_eq(s3.obs_size(), 0, "ISensor3D base obs_size -> 0")
	h.assert_true(s3 is Node3D, "ISensor3D is a Node3D")
	s3.free()

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_i_sensor.gd`
Expected: parse/load error — the interface files do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/sensors/i_sensor_2d.gd`:

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

Create `addons/godot_native_rl/sensors/i_sensor_3d.gd`:

```gdscript
class_name ISensor3D
extends Node3D

# Shared base for 3D flat-float sensors: each contributes a flat Array of floats to the
# agent observation. Subclasses override both methods.
#
# Subclasses MUST extend this BY PATH:
#     extends "res://addons/godot_native_rl/sensors/i_sensor_3d.gd"
# never `extends ISensor3D` — the global class-name cache is unreliable headless (see
# CLAUDE.md). The class_name above is for in-editor recognition only. Sensor discovery
# (NcnnControllerCore.collect_sensors) is duck-typed, never `is ISensor3D`.

func get_observation() -> Array:
	return []

func obs_size() -> int:
	return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_i_sensor.gd`
Expected: PASS, `Results: 6 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/i_sensor_2d.gd addons/godot_native_rl/sensors/i_sensor_3d.gd test/unit/test_i_sensor.gd
git commit -m "feat: ISensor2D/ISensor3D base interface scripts (backlog 40)"
```

---

## Task 2: collect_sensors() discovery helper

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_controller_core.gd` (add two static funcs after `obs_space_from_obs`)
- Test: `test/unit/test_collect_sensors.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_collect_sensors.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")

# Mock flat sensor: has both get_observation() and obs_size() (duck-typed match).
class MockSensor extends Node:
	var _obs: Array = []
	func setup(obs: Array) -> void:
		_obs = obs
	func get_observation() -> Array:
		return _obs
	func obs_size() -> int:
		return _obs.size()

# Mock image-like sensor: has get_observation() but NO obs_size() (must be skipped).
class MockCamera extends Node:
	func get_observation() -> String:
		return "deadbeef"

func _make_sensor(obs: Array) -> MockSensor:
	var s := MockSensor.new()
	s.setup(obs)
	return s

func _initialize() -> void:
	var h := Harness.new()

	# Empty tree -> []
	var empty_root := Node.new()
	h.assert_eq(NcnnControllerCore.collect_sensors(empty_root), [], "empty tree -> []")
	empty_root.free()

	# Single sensor -> its obs
	var root1 := Node.new()
	root1.add_child(_make_sensor([1.0, 2.0]))
	h.assert_eq(NcnnControllerCore.collect_sensors(root1), [1.0, 2.0], "single sensor -> its obs")
	root1.free()

	# Multiple + nested + camera-like skip + plain node ignored.
	# Tree (insertion order): sensorA[1,2], pivot{ sensorB[3] }, camera, plain, sensorC[4,5]
	var root := Node.new()
	root.add_child(_make_sensor([1.0, 2.0]))          # sensorA
	var pivot := Node.new()
	pivot.add_child(_make_sensor([3.0]))              # sensorB (nested)
	root.add_child(pivot)
	root.add_child(MockCamera.new())                 # skipped (no obs_size)
	root.add_child(Node.new())                        # plain node, ignored
	root.add_child(_make_sensor([4.0, 5.0]))          # sensorC
	var obs: Array = NcnnControllerCore.collect_sensors(root)
	h.assert_eq(obs, [1.0, 2.0, 3.0, 4.0, 5.0], "depth-first tree order, camera+plain skipped")
	root.free()

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_collect_sensors.gd`
Expected: FAIL — `collect_sensors` is not defined on `NcnnControllerCore` (parse error or runtime error on the nil call).

- [ ] **Step 3: Write minimal implementation**

In `addons/godot_native_rl/controllers/ncnn_controller_core.gd`, append after the existing `obs_space_from_obs(...)` static function (end of file):

```gdscript
# Recursively gather flat sensors under `root` (duck-typed) in stable scene-tree order and
# concatenate their observations into one flat Array. Nodes without obs_size() (e.g.
# CameraSensor, which returns a hex String under its own obs key) are skipped — compose
# those manually. Depth-first pre-order over get_children() -> deterministic obs layout.
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

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_collect_sensors.gd`
Expected: PASS, `Results: 3 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_controller_core.gd test/unit/test_collect_sensors.gd
git commit -m "feat: NcnnControllerCore.collect_sensors recursive duck-typed discovery (backlog 40)"
```

---

## Task 3: Retrofit the six flat sensors onto the interface

**Files:**
- Modify: `addons/godot_native_rl/sensors/raycast_sensor_2d.gd` (line 2)
- Modify: `addons/godot_native_rl/sensors/raycast_sensor_3d.gd` (line 2)
- Modify: `addons/godot_native_rl/sensors/relative_position_sensor_2d.gd` (line 2)
- Modify: `addons/godot_native_rl/sensors/relative_position_sensor_3d.gd` (line 2)
- Modify: `addons/godot_native_rl/sensors/grid_sensor_2d.gd` (line 2)
- Modify: `addons/godot_native_rl/sensors/grid_sensor_3d.gd` (line 2)
- Test: `test/unit/test_sensor_interface_conformance.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_sensor_interface_conformance.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const ISensor2D = preload("res://addons/godot_native_rl/sensors/i_sensor_2d.gd")
const ISensor3D = preload("res://addons/godot_native_rl/sensors/i_sensor_3d.gd")
const RaycastSensor2D = preload("res://addons/godot_native_rl/sensors/raycast_sensor_2d.gd")
const RaycastSensor3D = preload("res://addons/godot_native_rl/sensors/raycast_sensor_3d.gd")
const RelativePositionSensor2D = preload("res://addons/godot_native_rl/sensors/relative_position_sensor_2d.gd")
const RelativePositionSensor3D = preload("res://addons/godot_native_rl/sensors/relative_position_sensor_3d.gd")
const GridSensor2D = preload("res://addons/godot_native_rl/sensors/grid_sensor_2d.gd")
const GridSensor3D = preload("res://addons/godot_native_rl/sensors/grid_sensor_3d.gd")

func _check(h, node, iface, label: String) -> void:
	h.assert_true(node is iface, label)
	node.free()

func _initialize() -> void:
	var h := Harness.new()
	_check(h, RaycastSensor2D.new(), ISensor2D, "RaycastSensor2D is ISensor2D")
	_check(h, RelativePositionSensor2D.new(), ISensor2D, "RelativePositionSensor2D is ISensor2D")
	_check(h, GridSensor2D.new(), ISensor2D, "GridSensor2D is ISensor2D")
	_check(h, RaycastSensor3D.new(), ISensor3D, "RaycastSensor3D is ISensor3D")
	_check(h, RelativePositionSensor3D.new(), ISensor3D, "RelativePositionSensor3D is ISensor3D")
	_check(h, GridSensor3D.new(), ISensor3D, "GridSensor3D is ISensor3D")
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_sensor_interface_conformance.gd`
Expected: FAIL — all six `is` assertions are false because the sensors still `extends Node2D/Node3D`, not the interface. (`Results: 0 passed, 6 failed`.)

- [ ] **Step 3: Retrofit the six `extends` lines**

In each file, replace line 2 only (keep line 1 `class_name ...` and the rest unchanged):

`raycast_sensor_2d.gd`, `relative_position_sensor_2d.gd`, `grid_sensor_2d.gd`:
```gdscript
extends "res://addons/godot_native_rl/sensors/i_sensor_2d.gd"
```

`raycast_sensor_3d.gd`, `relative_position_sensor_3d.gd`, `grid_sensor_3d.gd`:
```gdscript
extends "res://addons/godot_native_rl/sensors/i_sensor_3d.gd"
```

(Each was previously `extends Node2D` or `extends Node3D`; the interface itself extends Node2D/Node3D, so the sensors remain Node2D/Node3D — no behavior change.)

- [ ] **Step 4: Run the new test AND the existing sensor suites to verify behavior is preserved**

Run:
```bash
/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_sensor_interface_conformance.gd
/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_raycast_sensor_2d.gd
/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_raycast_sensor_3d.gd
/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_relative_position_sensor_2d.gd
/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_relative_position_sensor_3d.gd
/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_grid_sensor_2d.gd
/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_grid_sensor_3d.gd
```
Expected: conformance test `Results: 6 passed, 0 failed`; every existing sensor test still ends `0 failed` (unchanged behavior).

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/raycast_sensor_2d.gd addons/godot_native_rl/sensors/raycast_sensor_3d.gd addons/godot_native_rl/sensors/relative_position_sensor_2d.gd addons/godot_native_rl/sensors/relative_position_sensor_3d.gd addons/godot_native_rl/sensors/grid_sensor_2d.gd addons/godot_native_rl/sensors/grid_sensor_3d.gd test/unit/test_sensor_interface_conformance.gd
git commit -m "feat: flat sensors extend ISensor2D/3D by path (backlog 40)"
```

---

## Task 4: Controller convenience method

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` (add method near `get_obs`/`get_obs_space`, ~line 128)
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd` (add method near `get_obs`/`get_obs_space`, ~line 128)
- Test: `test/unit/test_controller_collect_sensors.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_controller_collect_sensors.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const NcnnAIController2D = preload("res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd")
const NcnnAIController3D = preload("res://addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd")

class MockSensor extends Node:
	var _obs: Array = []
	func setup(obs: Array) -> void:
		_obs = obs
	func get_observation() -> Array:
		return _obs
	func obs_size() -> int:
		return _obs.size()

func _mock(obs: Array) -> MockSensor:
	var s := MockSensor.new()
	s.setup(obs)
	return s

func _initialize() -> void:
	var h := Harness.new()

	var c2 = NcnnAIController2D.new()
	c2.add_child(_mock([1.0, 2.0]))
	c2.add_child(_mock([3.0]))
	h.assert_eq(c2.collect_sensors(), [1.0, 2.0, 3.0], "2D controller concatenates child sensors")
	c2.free()

	var c3 = NcnnAIController3D.new()
	c3.add_child(_mock([7.0]))
	h.assert_eq(c3.collect_sensors(), [7.0], "3D controller concatenates child sensors")
	c3.free()

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_controller_collect_sensors.gd`
Expected: FAIL — `collect_sensors()` is not a method on the controllers.

- [ ] **Step 3: Add the convenience method to both controllers**

In `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd`, add after the `get_obs_space()` function (around line 128):

```gdscript
# Convenience: concatenate all child flat-sensor observations (recursive, tree order).
# Agents can write `return {"obs": collect_sensors()}` in get_obs().
func collect_sensors() -> Array:
	return NcnnControllerCore.collect_sensors(self)
```

In `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd`, add the identical method after its `get_obs_space()` (around line 128):

```gdscript
# Convenience: concatenate all child flat-sensor observations (recursive, tree order).
# Agents can write `return {"obs": collect_sensors()}` in get_obs().
func collect_sensors() -> Array:
	return NcnnControllerCore.collect_sensors(self)
```

(Both files already `const NcnnControllerCore = preload(".../ncnn_controller_core.gd")` at the top, so no new import is needed.)

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_controller_collect_sensors.gd`
Expected: PASS, `Results: 2 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd test/unit/test_controller_collect_sensors.gd
git commit -m "feat: collect_sensors() convenience on NcnnAIController2D/3D (backlog 40)"
```

---

## Task 5: Full suite green + docs

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/BACKLOG.md`
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite from a clean cache**

Run:
```bash
rm -f .godot/global_script_class_cache.cfg
./test/run_tests.sh
```
Expected: `All tests passed.` (the four new `test_i_sensor` / `test_collect_sensors` / `test_sensor_interface_conformance` / `test_controller_collect_sensors` suites and all existing suites green). If it hangs, see the fresh-clone class-cache trap in CLAUDE.md.

- [ ] **Step 2: Update CLAUDE.md sensors description**

In `CLAUDE.md`, in the `sensors/` bullet of the "Current state" section, after the list of sensors add a note about the interface + discovery. Find the text `+ pure
  \`raycast_math\`/\`relative_position_math\`/\`camera_obs_math\`/\`grid_sensor_math\`)` and insert before the closing paren:
```
; all flat sensors extend `ISensor2D`/`ISensor3D` and are auto-discovered by
  `NcnnControllerCore.collect_sensors(agent)` / the controllers' `collect_sensors()`
```
Keep it terse; CLAUDE.md is always-loaded.

- [ ] **Step 3: Mark backlog item 40 done**

In `docs/BACKLOG.md`, change the line beginning `40. ⬜ **\`ISensor2D\` / \`ISensor3D\` interface**` to `40. ✅ ...` and append an indented note:
```
    **Done 2026-06-03** — spec `docs/superpowers/specs/2026-06-03-isensor-interface-design.md`,
    plan `docs/superpowers/plans/2026-06-03-isensor-interface.md`. Lightweight `ISensor2D`/`ISensor3D`
    (Node2D/3D base, `get_observation() -> Array` + `obs_size() -> int`); the six flat sensors extend
    them **by path** (headless-safe); `NcnnControllerCore.collect_sensors(root)` recursively gathers
    flat sensors via duck typing in tree order (CameraSensor skipped — no `obs_size`), plus a
    `collect_sensors()` convenience on `NcnnAIController2D/3D`. Headless unit tests for base stubs,
    discovery/ordering/skip, real-sensor `is`-conformance, and the controller method.
```
Also add `40` to the "Done:" summary line in `CLAUDE.md` (the one near the sensor-track list that enumerates completed item numbers), in the form `40 (ISensor2D/3D interface + collect_sensors auto-discovery)`.

- [ ] **Step 4: Update the README sensors section**

In `README.md`, in the Sensors section (after the per-sensor bullets, near the "Pure ray geometry lives in..." paragraph), add a short paragraph:
```
All flat-float sensors (`RaycastSensor2D/3D`, `RelativePositionSensor2D/3D`, `GridSensor2D/3D`)
extend `ISensor2D`/`ISensor3D` and expose `get_observation() -> Array` + `obs_size() -> int`. An
agent can let the controller gather them automatically instead of concatenating by hand:
`func get_obs() -> Dictionary: return {"obs": collect_sensors()}` —
`collect_sensors()` walks the agent's child sensors depth-first in scene-tree order.
`CameraSensor` returns image obs under its own key and is composed separately.
```

- [ ] **Step 5: Clean stray uid files and commit docs**

```bash
git clean -f -- '*.gd.uid'
git add CLAUDE.md README.md docs/BACKLOG.md
git commit -m "docs: document ISensor2D/3D + collect_sensors, mark backlog 40 done"
```

---

## Self-Review notes

- **Spec coverage:** interface scripts → Task 1; `collect_sensors` static + duck-typed recursion + Camera skip → Task 2; retrofit six sensors by path → Task 3; controller convenience → Task 4; tests in each task; docs → Task 5. All spec sections covered. (No trained-model migration — matches spec's out-of-scope.)
- **Type consistency:** `get_observation() -> Array`, `obs_size() -> int`, `collect_sensors(root: Node) -> Array`, `_gather_sensor_obs(node, out)`, controller `collect_sensors() -> Array` used identically across tasks. Mock classes expose the same duck-typed surface the helper checks.
- **No placeholders:** every code step shows complete code; every run step gives the exact command + expected result.
- **Headless safety:** sensors extend the interface by path; discovery is duck-typed (never `is ISensor*`); conformance test uses preloaded-script `is`; suite run from a clean cache.
