# Editor-DX Parity (#112) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship drop-in sensor `.tscn` scenes (Raycast 2D/3D + Camera 2D/3D) and an `NcnnAIController2D/3D` script template that the plugin auto-installs into `res://script_templates/` — closing both `Minor (#112)` rows in the gap analysis.

**Architecture:** Four hand-written `.tscn` scenes under `addons/godot_native_rl/sensors/scenes/` referencing existing sensor scripts by `res://` path. Templates live inside the addon behind a `.gdignore` (they contain `extends _BASE_`, which is not valid GDScript); a pure-helper installer (`build_plan` / `execute_plan`, mirroring the `plugin_runtime_check.gd` pattern) is wired into `plugin.gd._enter_tree` to copy them to `res://script_templates/` — copy-if-missing, never overwrite, never delete.

**Tech Stack:** GDScript only (no C++/Python changes). Headless tests via `test/harness.gd` (`extends SceneTree`, auto-discovered by `test/run_tests.sh` from `test/unit/test_*.gd`).

**Spec:** `docs/superpowers/specs/2026-06-10-editor-dx-parity-design.md`
**Branch:** `feat/112-editor-dx-parity` (already created; spec committed).

---

## Conventions you must follow

- **GDScript uses TAB indentation** (repo-wide convention). All code blocks below use tabs.
- Reference scripts by full `res://` path (`class_name` is unreliable headless).
- Run a single test with:
  ```bash
  GODOT="${GODOT:-$(command -v godot || command -v godot-mono)}"
  "$GODOT" --headless --path . --script res://test/unit/test_X.gd
  ```
  A passing run prints `Results: N passed, 0 failed` and exits 0. (The Godot binary path is
  machine-specific — always probe via `command -v`, never hardcode.)
- The editor import pass scatters `*.gd.uid` files — **never commit them** (`git clean -f -- '*.gd.uid'` if they appear in `git status`).
- Commit with conventional-commit types; no attribution footer (disabled globally).

---

### Task 1: Raycast sensor scenes

**Files:**
- Test: `test/unit/test_sensor_scenes.gd` (create)
- Create: `addons/godot_native_rl/sensors/scenes/RaycastSensor2D.tscn`
- Create: `addons/godot_native_rl/sensors/scenes/RaycastSensor3D.tscn`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_sensor_scenes.gd`:

```gdscript
extends SceneTree

# Smoke tests for the drop-in sensor scenes (#112): each .tscn must instantiate headlessly,
# carry the right script (referenced by res:// path), and be usable with its defaults.

const Harness = preload("res://test/harness.gd")

const SCENES_DIR := "res://addons/godot_native_rl/sensors/scenes"

func _initialize() -> void:
	var h := Harness.new()

	# --- RaycastSensor2D.tscn ---
	var packed_2d: PackedScene = load("%s/RaycastSensor2D.tscn" % SCENES_DIR)
	h.assert_true(packed_2d != null, "RaycastSensor2D.tscn loads")
	var ray2d = packed_2d.instantiate()
	h.assert_true(ray2d is Node2D, "RaycastSensor2D root is Node2D")
	h.assert_true(ray2d.has_method("get_observation"), "RaycastSensor2D has sensor script")
	h.assert_true(ray2d.obs_size() > 0, "RaycastSensor2D obs_size > 0 with defaults")
	ray2d.free()

	# --- RaycastSensor3D.tscn ---
	var packed_3d: PackedScene = load("%s/RaycastSensor3D.tscn" % SCENES_DIR)
	h.assert_true(packed_3d != null, "RaycastSensor3D.tscn loads")
	var ray3d = packed_3d.instantiate()
	h.assert_true(ray3d is Node3D, "RaycastSensor3D root is Node3D")
	h.assert_true(ray3d.has_method("get_observation"), "RaycastSensor3D has sensor script")
	h.assert_true(ray3d.obs_size() > 0, "RaycastSensor3D obs_size > 0 with defaults")
	ray3d.free()

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
GODOT="${GODOT:-$(command -v godot || command -v godot-mono)}"
"$GODOT" --headless --path . --script res://test/unit/test_sensor_scenes.gd
```

Expected: FAIL — engine errors about the missing `.tscn` files, then either a script error on the null `packed_2d` or `FAIL: RaycastSensor2D.tscn loads`. Non-zero exit either way.

- [ ] **Step 3: Create the two scenes**

Create `addons/godot_native_rl/sensors/scenes/RaycastSensor2D.tscn` (script defaults kept — no property overrides):

```
[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://addons/godot_native_rl/sensors/raycast_sensor_2d.gd" id="1"]

[node name="RaycastSensor2D" type="Node2D"]
script = ExtResource("1")
```

Create `addons/godot_native_rl/sensors/scenes/RaycastSensor3D.tscn`:

```
[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://addons/godot_native_rl/sensors/raycast_sensor_3d.gd" id="1"]

[node name="RaycastSensor3D" type="Node3D"]
script = ExtResource("1")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
"$GODOT" --headless --path . --script res://test/unit/test_sensor_scenes.gd
```

Expected: PASS — `Results: 8 passed, 0 failed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add test/unit/test_sensor_scenes.gd addons/godot_native_rl/sensors/scenes/RaycastSensor2D.tscn addons/godot_native_rl/sensors/scenes/RaycastSensor3D.tscn
git commit -m "feat: drop-in RaycastSensor2D/3D scenes (#112)"
```

---

### Task 2: Camera sensor scenes (pre-wired SubViewport)

**Files:**
- Test: `test/unit/test_sensor_scenes.gd` (modify — append camera asserts before `h.finish`)
- Create: `addons/godot_native_rl/sensors/scenes/CameraSensor2D.tscn`
- Create: `addons/godot_native_rl/sensors/scenes/CameraSensor3D.tscn`

- [ ] **Step 1: Extend the test (failing)**

In `test/unit/test_sensor_scenes.gd`, insert before `h.finish(self)`:

```gdscript
	# --- CameraSensor2D.tscn: SubViewport + Camera2D pre-wired ---
	var packed_cam2d: PackedScene = load("%s/CameraSensor2D.tscn" % SCENES_DIR)
	h.assert_true(packed_cam2d != null, "CameraSensor2D.tscn loads")
	var cam2d = packed_cam2d.instantiate()
	h.assert_true(cam2d.viewport is SubViewport, "CameraSensor2D viewport export pre-wired")
	h.assert_eq(cam2d.viewport.size, Vector2i(36, 36), "CameraSensor2D SubViewport is 36x36")
	h.assert_eq(cam2d.viewport.render_target_update_mode, SubViewport.UPDATE_ALWAYS,
		"CameraSensor2D SubViewport renders every frame")
	h.assert_true(cam2d.viewport.get_node_or_null("Camera2D") is Camera2D,
		"CameraSensor2D has a Camera2D inside the SubViewport")
	h.assert_true(cam2d.is_key_valid(cam2d.observation_key),
		"CameraSensor2D observation_key valid (contains \"2d\")")
	cam2d.free()

	# --- CameraSensor3D.tscn: SubViewport + Camera3D pre-wired ---
	var packed_cam3d: PackedScene = load("%s/CameraSensor3D.tscn" % SCENES_DIR)
	h.assert_true(packed_cam3d != null, "CameraSensor3D.tscn loads")
	var cam3d = packed_cam3d.instantiate()
	h.assert_true(cam3d.viewport is SubViewport, "CameraSensor3D viewport export pre-wired")
	h.assert_true(cam3d.viewport.get_node_or_null("Camera3D") is Camera3D,
		"CameraSensor3D has a Camera3D inside the SubViewport")
	h.assert_true(cam3d.is_key_valid(cam3d.observation_key),
		"CameraSensor3D observation_key valid (godot_rl routes image obs on the \"2d\" substring)")
	cam3d.free()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
"$GODOT" --headless --path . --script res://test/unit/test_sensor_scenes.gd
```

Expected: FAIL on `CameraSensor2D.tscn loads` (missing file), non-zero exit.

- [ ] **Step 3: Create the two camera scenes**

Create `addons/godot_native_rl/sensors/scenes/CameraSensor2D.tscn`. Note `node_paths=PackedStringArray("viewport")` — that is how Godot serializes an exported Node property; the path resolves to the child SubViewport at instantiation. `render_target_update_mode = 4` is `UPDATE_ALWAYS`; 36×36 matches upstream `RGBCameraSensor2D`'s default render resolution:

```
[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://addons/godot_native_rl/sensors/camera_sensor.gd" id="1"]

[node name="CameraSensor2D" type="Node" node_paths=PackedStringArray("viewport")]
script = ExtResource("1")
viewport = NodePath("SubViewport")

[node name="SubViewport" type="SubViewport" parent="."]
size = Vector2i(36, 36)
render_target_update_mode = 4

[node name="Camera2D" type="Camera2D" parent="SubViewport"]
```

Create `addons/godot_native_rl/sensors/scenes/CameraSensor3D.tscn` (same shape, `Camera3D` child; `observation_key` keeps its `"camera_2d"` default on purpose — godot_rl routes image obs on the `"2d"` substring even for 3D captures):

```
[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://addons/godot_native_rl/sensors/camera_sensor.gd" id="1"]

[node name="CameraSensor3D" type="Node" node_paths=PackedStringArray("viewport")]
script = ExtResource("1")
viewport = NodePath("SubViewport")

[node name="SubViewport" type="SubViewport" parent="."]
size = Vector2i(36, 36)
render_target_update_mode = 4

[node name="Camera3D" type="Camera3D" parent="SubViewport"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
"$GODOT" --headless --path . --script res://test/unit/test_sensor_scenes.gd
```

Expected: PASS — `Results: 18 passed, 0 failed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add test/unit/test_sensor_scenes.gd addons/godot_native_rl/sensors/scenes/CameraSensor2D.tscn addons/godot_native_rl/sensors/scenes/CameraSensor3D.tscn
git commit -m "feat: drop-in CameraSensor2D/3D scenes with pre-wired SubViewport (#112)"
```

---

### Task 3: Script template files + `.gdignore` + content test

**Files:**
- Test: `test/unit/test_script_template_content.gd` (create)
- Create: `addons/godot_native_rl/script_templates/.gdignore` (empty file)
- Create: `addons/godot_native_rl/script_templates/NcnnAIController2D/controller_template.gd`
- Create: `addons/godot_native_rl/script_templates/NcnnAIController3D/controller_template.gd`

The `.gdignore` is load-bearing: the templates contain `extends _BASE_` (Godot's template
placeholder, not valid GDScript). `.gdignore` keeps the editor filesystem scan — and the headless
import pass in `run_tests.sh` — from parsing them, while `FileAccess`/`DirAccess` can still read
the files (ignore only affects the resource scanner).

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_script_template_content.gd`:

```gdscript
extends SceneTree

# Guards the controller script templates (#112): valid Godot template headers, the
# `extends _BASE_` placeholder, all four required stubs, and the .gdignore that keeps
# the (intentionally non-parseable) templates out of the resource scan.

const Harness = preload("res://test/harness.gd")

const TEMPLATES := [
	"res://addons/godot_native_rl/script_templates/NcnnAIController2D/controller_template.gd",
	"res://addons/godot_native_rl/script_templates/NcnnAIController3D/controller_template.gd",
]

func _initialize() -> void:
	var h := Harness.new()

	for path in TEMPLATES:
		var text := FileAccess.get_file_as_string(path)
		h.assert_true(text != "", "%s readable via FileAccess (despite .gdignore)" % path)
		h.assert_true(text.contains("# meta-name:"), "%s: meta-name header" % path)
		h.assert_true(text.contains("# meta-default: true"), "%s: meta-default header" % path)
		h.assert_true(text.contains("extends _BASE_"), "%s: extends _BASE_ placeholder" % path)
		for stub in ["func get_obs()", "func get_reward()", "func get_action_space()", "func set_action(action)"]:
			h.assert_true(text.contains(stub), "%s: has %s stub" % [path, stub])
		h.assert_true(text.contains("collect_sensors()"), "%s: mentions sensor auto-discovery" % path)

	h.assert_true(FileAccess.file_exists("res://addons/godot_native_rl/script_templates/.gdignore"),
		".gdignore present (templates are not valid GDScript and must stay unscanned)")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
"$GODOT" --headless --path . --script res://test/unit/test_script_template_content.gd
```

Expected: FAIL — `FAIL: ... readable via FileAccess ...` for both paths, non-zero exit.

- [ ] **Step 3: Create the template files**

Create the empty marker:

```bash
mkdir -p addons/godot_native_rl/script_templates/NcnnAIController2D addons/godot_native_rl/script_templates/NcnnAIController3D
touch addons/godot_native_rl/script_templates/.gdignore
```

Create `addons/godot_native_rl/script_templates/NcnnAIController2D/controller_template.gd` with exactly this content (tabs for indentation):

```gdscript
# meta-name: NCNN AI Controller
# meta-description: Agent scaffold for Godot Native RL — implement the obs/reward/action contract
# meta-default: true
extends _BASE_

# The four methods below define your agent. Each stub fails loud (push_error) so a
# forgotten override surfaces immediately instead of silently training on garbage.

func get_obs() -> Dictionary:
	# Compose a flat float Array. With ISensor2D/3D children, auto-discovery does it:
	#     var obs := []
	#     for sensor in collect_sensors():
	#         obs.append_array(sensor.get_observation())
	#     return {"obs": obs}
	push_error("get_obs() not implemented — return {\"obs\": [floats...]}")
	return {"obs": []}

func get_reward() -> float:
	# Return the reward accumulated since the last step — e.g. a RewardBuilder total,
	# or a hand-computed shaping term (distance delta, goal bonus, time penalty).
	push_error("get_reward() not implemented")
	return 0.0

func get_action_space() -> Dictionary:
	# Describe each action head. Examples:
	#     "move": {"size": 4, "action_type": "discrete"}     # one of 4 choices
	#     "steer": {"size": 2, "action_type": "continuous"}  # 2 floats in [-1, 1]
	push_error("get_action_space() not implemented")
	return {}

func set_action(action) -> void:
	# Apply the chosen action to your agent, e.g.:
	#     var idx := int(action["move"])
	#     velocity = DIRECTIONS[idx] * speed
	push_error("set_action() not implemented")

#func get_obs_space() -> Dictionary:
#	# Override only for complex obs (images, multiple keys). The base class derives
#	# {"obs": {"size": [len], "space": "box"}} from get_obs() automatically.
#	return super.get_obs_space()
```

Create `addons/godot_native_rl/script_templates/NcnnAIController3D/controller_template.gd` with the **identical content** (the directory name is what binds a template to its base class; the body is class-agnostic because it extends `_BASE_`).

- [ ] **Step 4: Run test to verify it passes**

```bash
"$GODOT" --headless --path . --script res://test/unit/test_script_template_content.gd
```

Expected: PASS — `Results: 19 passed, 0 failed`, exit 0.

- [ ] **Step 5: Sanity-check the scan still works**

The headless import pass must not choke on `extends _BASE_`:

```bash
"$GODOT" --headless --editor --quit 2>&1 | grep -i "_BASE_" ; echo "grep exit: $?"
git clean -f -- '*.gd.uid'
```

Expected: no `_BASE_` parse errors (grep exit 1).

- [ ] **Step 6: Commit**

```bash
git add test/unit/test_script_template_content.gd addons/godot_native_rl/script_templates/
git commit -m "feat: NcnnAIController2D/3D script templates behind .gdignore (#112)"
```

---

### Task 4: Installer — pure `build_plan`

**Files:**
- Test: `test/unit/test_script_template_installer.gd` (create)
- Create: `addons/godot_native_rl/script_template_installer.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_script_template_installer.gd`:

```gdscript
extends SceneTree

# Tests the script-template installer (#112). build_plan is pure (file_exists injected);
# execute_plan is exercised against user:// with real DirAccess (Task 5 extends this file).

const Harness = preload("res://test/harness.gd")
const Installer = preload("res://addons/godot_native_rl/script_template_installer.gd")

const SOURCES := [
	"res://addons/x/script_templates/Class2D/tmpl.gd",
	"res://addons/x/script_templates/Class3D/tmpl.gd",
]

func _initialize() -> void:
	var h := Harness.new()

	# --- build_plan: nothing installed yet -> everything planned, ClassDir/file preserved ---
	var none_exist := func(_p: String) -> bool: return false
	var plan: Array = Installer.build_plan(SOURCES, "res://script_templates", none_exist)
	h.assert_eq(plan.size(), 2, "all missing -> both planned")
	h.assert_eq(plan[0]["src"], SOURCES[0], "plan keeps source path")
	h.assert_eq(plan[0]["dst"], "res://script_templates/Class2D/tmpl.gd", "dst = root/ClassDir/file")
	h.assert_eq(plan[1]["dst"], "res://script_templates/Class3D/tmpl.gd", "second dst correct")

	# --- build_plan: everything installed -> empty plan (never overwrite) ---
	var all_exist := func(_p: String) -> bool: return true
	h.assert_eq(Installer.build_plan(SOURCES, "res://script_templates", all_exist).size(), 0,
		"all present -> nothing planned")

	# --- build_plan: partial install -> only the missing one ---
	var only_2d_exists := func(p: String) -> bool: return p.contains("Class2D")
	var partial: Array = Installer.build_plan(SOURCES, "res://script_templates", only_2d_exists)
	h.assert_eq(partial.size(), 1, "one present -> one planned")
	h.assert_eq(partial[0]["dst"], "res://script_templates/Class3D/tmpl.gd", "the missing 3D one")

	# --- build_plan: returns a new array, inputs untouched ---
	h.assert_eq(SOURCES.size(), 2, "sources array not mutated")

	# --- the addon's real constants point at files that exist ---
	for src in Installer.TEMPLATE_SOURCES:
		h.assert_true(FileAccess.file_exists(src), "%s exists" % src)
	h.assert_eq(Installer.DEST_ROOT, "res://script_templates", "dest root is the editor default")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
"$GODOT" --headless --path . --script res://test/unit/test_script_template_installer.gd
```

Expected: FAIL — the preload of the missing `script_template_installer.gd` aborts the script (engine parse/load error), non-zero exit.

- [ ] **Step 3: Write the installer (planning half)**

Create `addons/godot_native_rl/script_template_installer.gd`:

```gdscript
@tool
extends RefCounted

# Installs the addon's controller script templates into the project-level
# res://script_templates/ (the editor's default `editor/script/templates_search_path`),
# where Godot's "new script from template" flow discovers them. The canonical templates
# live inside the addon (so they ship in the addon zip) behind a .gdignore, because
# `extends _BASE_` is not valid GDScript and must stay out of the resource scan.
# Planning is pure (file_exists injected) so it is testable headless; plugin.gd wires
# it up on enable. Copy-if-missing only: a user's edited copy is never overwritten.

const TEMPLATE_SOURCES: Array[String] = [
	"res://addons/godot_native_rl/script_templates/NcnnAIController2D/controller_template.gd",
	"res://addons/godot_native_rl/script_templates/NcnnAIController3D/controller_template.gd",
]
const DEST_ROOT := "res://script_templates"

# Returns a new Array of {"src": String, "dst": String} for each source whose destination
# (dest_root/<ClassDir>/<file>) is missing. file_exists: Callable(String) -> bool.
static func build_plan(sources: Array, dest_root: String, file_exists: Callable) -> Array:
	var plan: Array = []
	for src_v in sources:
		var src := String(src_v)
		var parts := src.split("/")
		if parts.size() < 2:
			push_error("script_template_installer: malformed template source path %s" % src)
			continue
		var dst := "%s/%s/%s" % [dest_root, parts[parts.size() - 2], parts[parts.size() - 1]]
		if not file_exists.call(dst):
			plan.append({"src": src, "dst": dst})
	return plan
```

- [ ] **Step 4: Run test to verify it passes**

```bash
"$GODOT" --headless --path . --script res://test/unit/test_script_template_installer.gd
```

Expected: PASS — `Results: 11 passed, 0 failed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add test/unit/test_script_template_installer.gd addons/godot_native_rl/script_template_installer.gd
git commit -m "feat: script-template installer — pure copy-if-missing planner (#112)"
```

---

### Task 5: Installer — `execute_plan` (real copy round-trip)

**Files:**
- Test: `test/unit/test_script_template_installer.gd` (modify — append before `h.finish`)
- Modify: `addons/godot_native_rl/script_template_installer.gd` (append `execute_plan`)

- [ ] **Step 1: Extend the test (failing)**

In `test/unit/test_script_template_installer.gd`, insert before `h.finish(self)`:

```gdscript
	# --- execute_plan: real copy into user://, content round-trips ---
	var src_path: String = Installer.TEMPLATE_SOURCES[0]
	var dst := "user://test_script_templates/NcnnAIController2D/controller_template.gd"
	var errors: Array = Installer.execute_plan([{"src": src_path, "dst": dst}])
	h.assert_eq(errors, [], "execute_plan: no errors on a valid copy")
	h.assert_eq(FileAccess.get_file_as_string(dst), FileAccess.get_file_as_string(src_path),
		"execute_plan: copied content matches source")
	DirAccess.remove_absolute(dst)
	DirAccess.remove_absolute("user://test_script_templates/NcnnAIController2D")
	DirAccess.remove_absolute("user://test_script_templates")

	# --- execute_plan: missing source -> error collected, not swallowed ---
	# (the engine also prints its own error line here — that's the intentional failure path)
	var bad: Array = Installer.execute_plan(
		[{"src": "res://addons/godot_native_rl/script_templates/does_not_exist.gd",
			"dst": "user://test_script_templates/x.gd"}])
	h.assert_eq(bad.size(), 1, "execute_plan: missing source reported as one error")
	DirAccess.remove_absolute("user://test_script_templates")

	# --- execute_plan: empty plan is a no-op ---
	h.assert_eq(Installer.execute_plan([]), [], "execute_plan: empty plan -> no errors")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
"$GODOT" --headless --path . --script res://test/unit/test_script_template_installer.gd
```

Expected: FAIL — script error `Invalid call. Nonexistent function 'execute_plan'`, non-zero exit.

- [ ] **Step 3: Implement `execute_plan`**

Append to `addons/godot_native_rl/script_template_installer.gd`:

```gdscript
# Executes a plan from build_plan: mkdir -p the destination dir, then copy. Returns an
# Array of error strings ([] on full success); one failed entry does not stop the others.
static func execute_plan(plan: Array) -> Array:
	var errors: Array = []
	for entry_v in plan:
		var entry: Dictionary = entry_v
		var src := String(entry["src"])
		var dst := String(entry["dst"])
		var dir_err := DirAccess.make_dir_recursive_absolute(dst.get_base_dir())
		if dir_err != OK and dir_err != ERR_ALREADY_EXISTS:
			errors.append("mkdir failed for %s (error %d)" % [dst.get_base_dir(), dir_err])
			continue
		var copy_err := DirAccess.copy_absolute(src, dst)
		if copy_err != OK:
			errors.append("copy failed %s -> %s (error %d)" % [src, dst, copy_err])
	return errors
```

- [ ] **Step 4: Run test to verify it passes**

```bash
"$GODOT" --headless --path . --script res://test/unit/test_script_template_installer.gd
```

Expected: PASS — `Results: 15 passed, 0 failed`, exit 0. (One engine-printed error line from the
intentional missing-source case is expected and fine — the suite gates on exit code, not stderr.)

- [ ] **Step 5: Commit**

```bash
git add test/unit/test_script_template_installer.gd addons/godot_native_rl/script_template_installer.gd
git commit -m "feat: script-template installer execute_plan with collected errors (#112)"
```

---

### Task 6: Plugin glue + `.gitignore`

**Files:**
- Modify: `addons/godot_native_rl/plugin.gd`
- Modify: `.gitignore`

No new headless test: the glue is two calls into the already-tested halves, and `EditorPlugin`
lifecycle can't run headless. Both halves (`build_plan`, `execute_plan`) are covered by Tasks 4–5.

- [ ] **Step 1: Wire the installer into `plugin.gd`**

In `addons/godot_native_rl/plugin.gd`, add the preload after the existing ones:

```gdscript
const TemplateInstaller = preload("res://addons/godot_native_rl/script_template_installer.gd")
```

Extend the file-head comment's enumeration — replace this part of the comment:

```gdscript
# (b) registers an EditorExportPlugin that auto-packs your ncnn model files into game exports
# (without it, exported games crash with "cannot read model files" — see export/).
```

with:

```gdscript
# (b) registers an EditorExportPlugin that auto-packs your ncnn model files into game exports
# (without it, exported games crash with "cannot read model files" — see export/), and
# (c) installs the NcnnAIController script templates into res://script_templates/ (copy-if-
# missing — your edited copies are never touched; see script_template_installer.gd).
```

At the end of `_enter_tree()`, append:

```gdscript
	var plan := TemplateInstaller.build_plan(
		TemplateInstaller.TEMPLATE_SOURCES,
		TemplateInstaller.DEST_ROOT,
		func(p: String) -> bool: return FileAccess.file_exists(p)
	)
	for err in TemplateInstaller.execute_plan(plan):
		push_error("Godot Native RL: script template install failed: %s" % err)
```

(`_exit_tree` stays as-is — the install is one-way; disabling the plugin never deletes the
user's templates.)

- [ ] **Step 2: Gitignore the installed copies in this repo**

Enabling the plugin in this repo would copy templates to the repo root as a side effect.
The canonical files live in the addon; the root copies are machine-local noise. Append to
`.gitignore` (after the `# Editor export output` block):

```
# Installed by the plugin from addons/godot_native_rl/script_templates/ (editor-enable side effect)
/script_templates/
```

- [ ] **Step 3: Verify the plugin script still parses + suite-relevant tests pass**

```bash
"$GODOT" --headless --path . --script res://test/unit/test_script_template_installer.gd
"$GODOT" --headless --editor --quit 2>&1 | grep -iE "parse error|script_template" ; echo "grep exit: $?"
git clean -f -- '*.gd.uid'
git status --short   # must show only plugin.gd and .gitignore modified
```

Expected: installer test passes; no parse errors (grep exit 1, or only benign non-error lines); status clean apart from the two intended files.

- [ ] **Step 4: Commit**

```bash
git add addons/godot_native_rl/plugin.gd .gitignore
git commit -m "feat: auto-install script templates on plugin enable (#112)"
```

---

### Task 7: Docs

**Files:**
- Modify: `README.md` (the "What you get" list)
- Modify: `docs/guide/sensors.md` (intro)
- Modify: `docs/guide/building-your-agent.md` ("Agent contract" section)
- Modify: `docs/godot-rl-gap-analysis-2026-06-02.md` (two Sensors rows + one priority row)
- Modify: `CLAUDE.md` (library description line)

- [ ] **Step 1: README — add one "What you get" bullet**

In `README.md`, under `## What you get`, insert after the `NcnnAIController2D / NcnnAIController3D` bullet:

```markdown
- Editor DX: drag-in sensor scenes (`addons/godot_native_rl/sensors/scenes/` — raycast 2D/3D +
  camera 2D/3D with a pre-wired `SubViewport`) and an "NCNN AI Controller" script template,
  auto-installed to `res://script_templates/` when the plugin is enabled.
```

- [ ] **Step 2: Sensors guide — drop-in scenes note**

In `docs/guide/sensors.md`, after the opening paragraph (the one ending `...it joins the observation.`), insert:

```markdown
Prefer scenes? `addons/godot_native_rl/sensors/scenes/` ships drop-in `.tscn`s —
`RaycastSensor2D/3D` plus `CameraSensor2D/3D` with a pre-wired 36×36 `SubViewport` + camera.
Instance one under your agent and tweak the exports.
```

- [ ] **Step 3: Agent guide — template note**

In `docs/guide/building-your-agent.md`, at the end of the `## Agent contract` section (after the `get_info()` paragraph), insert:

```markdown
Tip: with the plugin enabled, **Attach Script → Template → "NCNN AI Controller"** starts you
from a scaffold with these four methods stubbed (the template is auto-installed to
`res://script_templates/`).
```

- [ ] **Step 4: Gap analysis — flip three rows**

In `docs/godot-rl-gap-analysis-2026-06-02.md`:

Replace:
```markdown
| Pre-built sensor `.tscn` scenes | ✅ RaycastSensor2D.tscn, RGBCameraSensor2D.tscn + examples | ❌ | Minor (#112) |
| `script_templates/AIController` | ✅ controller scaffold template in plugin | ❌ | Minor (#112) |
```
with:
```markdown
| Pre-built sensor `.tscn` scenes | ✅ RaycastSensor2D.tscn, RGBCameraSensor2D.tscn + examples | ✅ `sensors/scenes/` — Raycast2D/3D + Camera2D/3D (pre-wired SubViewport) | ✅ done (#112) |
| `script_templates/AIController` | ✅ controller scaffold template in plugin | ✅ `NcnnAIController2D/3D` templates, auto-installed on plugin enable | ✅ done (#112) |
```

Replace the priority-table row:
```markdown
| ⚪ P4 | Plugin editor-DX parity: pre-built sensor `.tscn` scenes + `script_templates/AIController` | #112 |
```
with:
```markdown
| ✅ Done | Plugin editor-DX parity: drop-in sensor scenes (`sensors/scenes/`) + `NcnnAIController` script templates auto-installed on enable | #112 |
```

(The 2026-06-09 audit paragraph near the top says the gaps "were filed" as issues — that is a
historical statement and stays as-is.)

- [ ] **Step 5: CLAUDE.md — one-line mention**

In `CLAUDE.md`, the "Current state" paragraph lists the library contents:
`` `addons/godot_native_rl/` (`sync.gd`/`NcnnSync`, `controllers/`, `reward/`, `sensors/`, `training/`, `net/`) ``.
Change `` `sensors/` `` to `` `sensors/` (+ drop-in `scenes/`) `` and append after `` `net/` ``:
`` , `script_templates/` (controller scaffold, auto-installed on plugin enable) ``.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/guide/sensors.md docs/guide/building-your-agent.md docs/godot-rl-gap-analysis-2026-06-02.md CLAUDE.md
git commit -m "docs: drop-in sensor scenes + controller script template (#112)"
```

---

### Task 8: Full suite, push, PR

- [ ] **Step 1: Run the full test suite**

```bash
GODOT="${GODOT:-$(command -v godot || command -v godot-mono)}" ./test/run_tests.sh
```

Expected: ends with `All tests passed.`, exit 0. Gate on that exact signal / the exit code —
do **not** grep for `failed`/`ERROR` (both appear in passing runs).

- [ ] **Step 2: Clean stray uid files, rebase, push**

```bash
git clean -f -- '*.gd.uid'
git fetch origin main && git rebase origin/main
git push -u origin feat/112-editor-dx-parity
```

(main moves fast in this repo — always rebase onto `origin/main` before every push. If the
rebase pulled in changes, re-run the full suite before pushing.)

- [ ] **Step 3: Open the PR**

```bash
gh pr create --title "feat: editor-DX parity — drop-in sensor scenes + controller script templates" --body "$(cat <<'EOF'
## Summary
- Drop-in sensor scenes under `addons/godot_native_rl/sensors/scenes/`: `RaycastSensor2D/3D.tscn` (script defaults) and `CameraSensor2D/3D.tscn` with a pre-wired 36×36 `SubViewport` + camera — the manual wiring users previously had to do by hand.
- `NcnnAIController2D/3D` script templates (Godot `# meta-name` format, `extends _BASE_`, four loud-failing stubs with guided comments incl. `collect_sensors()` composition). Canonical copies ship inside the addon behind a `.gdignore`; the plugin auto-installs them to `res://script_templates/` on enable — copy-if-missing, never overwrites, never deletes.
- Installer follows the pure-helper + thin-glue pattern (`script_template_installer.gd`, mirroring `plugin_runtime_check.gd`).
- Docs: README "What you get" bullet, sensors + agent guides, gap-analysis rows flipped, CLAUDE.md.

Closes #112

## Test plan
- [x] `test/unit/test_sensor_scenes.gd` — all four scenes instantiate headlessly; raycast `obs_size() > 0`; camera `viewport` export resolves to the 36×36 SubViewport with a camera inside
- [x] `test/unit/test_script_template_content.gd` — template headers, `extends _BASE_`, all four stubs, `.gdignore` present
- [x] `test/unit/test_script_template_installer.gd` — pure plan logic (missing→planned, present→skipped, no input mutation) + real `user://` copy round-trip + collected errors
- [x] Full `./test/run_tests.sh` green (incl. the fresh import pass over the `.gdignore`'d templates)
EOF
)"
```

- [ ] **Step 4: Verify CI is green**

```bash
gh pr checks --watch
```

Expected: all checks pass. If the build job is skipped on a `bin/` cache hit, that's normal
(this PR has no C++ changes).
