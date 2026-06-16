# Responsive Window + Demo Scaling — Implementation Plan (#271, #272)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the launcher and every demo window resizable and legibly scaled, and center the 2D
demos, without touching any simulation.

**Architecture:** One project-wide `[display]` block (`canvas_items` + `expand`, resizable, base
1280×720) fixes the shared root cause. The launcher bumps its fonts and centers its menu column. A
new reusable `FitCamera2D` node (2D analogue of #265's OrbitCamera) centers + zoom-fits each 2D
scene's fixed world rect, re-fitting on resize.

**Tech Stack:** Godot 4.5/4.6, GDScript (TAB indent), headless `SceneTree` test harness
(`test/harness.gd`), `.tscn` format=3.

**Spec:** `docs/superpowers/specs/2026-06-16-responsive-window-design.md`

**Branch:** `feature/271-272-responsive-window` (already created, spec committed).

---

## ⚠️ Read before Task 1 — the project.godot local-edit trap

The working tree carries **local-only** edits to `project.godot` that must **never** be committed
(Godot-4.6 editor resave artifacts; project minimum is **4.5**): `config/features` `"4.5"`→`"4.6"`,
a new `[animation]` section, and `run/main_scene` `uid://…`→`res://examples/launcher.tscn`. The same
applies to local edits in `export_presets.cfg` and `examples/fly_by/sky.hdr.import` — **never stage
them**. Task 1 uses a stash-scoped commit so the `[display]` block lands on the committed base only.

When you `git add`, always name **exact paths**. Never `git add -A` / `git add .` / `git add -u`.

---

## Task 1: `[display]` block + regression test

**Files:**
- Modify: `project.godot` (add a `[display]` section at end of file)
- Create: `test/unit/test_display_settings.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_display_settings.gd`:

```gdscript
extends SceneTree
# Regression (#271/#272): the project-wide [display] block must stay set so every scene is
# resizable and content-scaled. Reads ProjectSettings (loaded from project.godot in a headless run).

const Harness = preload("res://test/harness.gd")

func _initialize() -> void:
	var h := Harness.new()
	h.assert_eq(ProjectSettings.get_setting("display/window/stretch/mode", ""), "canvas_items", "stretch mode is canvas_items")
	h.assert_eq(ProjectSettings.get_setting("display/window/stretch/aspect", ""), "expand", "stretch aspect is expand")
	h.assert_eq(bool(ProjectSettings.get_setting("display/window/size/resizable", false)), true, "window is resizable")
	h.assert_eq(int(ProjectSettings.get_setting("display/window/size/viewport_width", 0)), 1280, "base viewport width 1280")
	h.assert_eq(int(ProjectSettings.get_setting("display/window/size/viewport_height", 0)), 720, "base viewport height 720")
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_display_settings.gd`
Expected: FAIL — `stretch mode is canvas_items` (current value is the default `""`/`disabled`).

- [ ] **Step 3: Add the `[display]` block on the committed base (stash-scoped)**

```bash
git stash push -- project.godot          # working project.godot reverts to committed base (4.5)
```

Append this block to the **end** of `project.godot` (EOF — keeps it far from the top-of-file local
edits so the later `stash pop` merges cleanly):

```ini

[display]

window/size/viewport_width=1280
window/size/viewport_height=720
window/size/resizable=true
window/stretch/mode="canvas_items"
window/stretch/aspect="expand"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_display_settings.gd`
Expected: PASS — `Results: 5 passed, 0 failed`.

- [ ] **Step 5: Commit, then restore local edits**

```bash
git add project.godot test/unit/test_display_settings.gd
git commit -m "feat(display): project-wide stretch + resizable window (#271, #272)"
git stash pop                            # restore the local 4.6/animation/main_scene edits on top
```

Verify the commit contains the `[display]` block and **not** `features=4.6`:
`git show HEAD:project.godot | grep -E 'config/features|stretch/mode'`
Expected: `config/features=PackedStringArray("4.5")` and `window/stretch/mode="canvas_items"`.
Then `git status -s` should again show `project.godot` as locally modified (the restored edits).

---

## Task 2: Launcher legibility — fonts + centered column (#271)

**Files:**
- Modify: `examples/launcher.gd:35-56` (`_ready()` UI build)
- Create: `test/unit/test_launcher_layout.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_launcher_layout.gd`:

```gdscript
extends SceneTree
# Regression (#271): the launcher title + buttons carry bumped font-size overrides so the menu is
# legible. Instances the launcher in the headless tree so _ready() builds the UI, then walks it.

const Harness = preload("res://test/harness.gd")
const Launcher = preload("res://examples/launcher.gd")

func _collect(node: Node, labels: Array, buttons: Array) -> void:
	if node is Label:
		labels.append(node)
	if node is Button:
		buttons.append(node)
	for c in node.get_children():
		_collect(c, labels, buttons)

func _initialize() -> void:
	var h := Harness.new()
	var l := Launcher.new()
	root.add_child(l)   # entering the tree fires _ready(), which builds the menu
	var labels: Array = []
	var buttons: Array = []
	_collect(l, labels, buttons)
	h.assert_true(labels.size() >= 1, "launcher has a title label")
	h.assert_true(buttons.size() >= 8, "launcher has demo buttons (got %d)" % buttons.size())
	h.assert_true(labels[0].has_theme_font_size_override("font_size"), "title has a font_size override")
	h.assert_true(labels[0].get_theme_font_size("font_size") >= 24, "title font >= 24")
	for b in buttons:
		h.assert_true(b.has_theme_font_size_override("font_size"), "button has a font_size override")
		h.assert_true(b.get_theme_font_size("font_size") >= 16, "button font >= 16")
	l.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_launcher_layout.gd`
Expected: FAIL — `title has a font_size override` (the current launcher sets no overrides).

- [ ] **Step 3: Update `examples/launcher.gd` `_ready()`**

Replace the body of `_ready()` (lines 35-56) with:

```gdscript
func _ready() -> void:
	var scroll := ScrollContainer.new()
	scroll.set_anchors_preset(Control.PRESET_FULL_RECT)
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED  # let the row fill width
	add_child(scroll)

	var row := HBoxContainer.new()
	row.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.alignment = BoxContainer.ALIGNMENT_CENTER  # centers the fixed-width column
	scroll.add_child(row)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 10)
	vbox.custom_minimum_size = Vector2(640, 0)
	row.add_child(vbox)

	var title := Label.new()
	title.text = "Godot Native RL — Demos   (☰ Menu or Esc returns here)"
	title.add_theme_font_size_override("font_size", 28)
	vbox.add_child(title)

	for d in DEMOS:
		var path: String = d[0]
		var btn := Button.new()
		btn.text = "%s\n    %s" % [d[1], d[2]]
		btn.add_theme_font_size_override("font_size", 18)
		btn.alignment = HORIZONTAL_ALIGNMENT_LEFT
		btn.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		btn.disabled = not ResourceLoader.exists(path)
		btn.pressed.connect(func() -> void: _run(path))
		vbox.add_child(btn)
```

Do **not** change `DEMOS`, `demo_scenes()`, or `_run()`.

- [ ] **Step 4: Run both launcher tests to verify they pass**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_launcher_layout.gd`
Expected: PASS.
Run: `"$GODOT" --headless --path . --script res://test/unit/test_launcher.gd`
Expected: PASS (the curated-list smoke is unaffected).

- [ ] **Step 5: Commit**

```bash
git add examples/launcher.gd test/unit/test_launcher_layout.gd
git commit -m "feat(launcher): bigger fonts + centered menu column (#271)"
```

---

## Task 3: `FitCamera2D` node + pure-helper test

**Files:**
- Create: `addons/godot_native_rl/camera/fit_camera_2d.gd`
- Create: `test/unit/test_fit_camera_2d.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_fit_camera_2d.gd`:

```gdscript
extends SceneTree
# Unit tests for FitCamera2D's pure fit_zoom helper (#272). Node lifecycle (_ready/resize) needs
# the tree and is covered by the scene-structure test.

const Harness = preload("res://test/harness.gd")
const FitCamera2D = preload("res://addons/godot_native_rl/camera/fit_camera_2d.gd")

func _vec_approx(h, got: Vector2, want: Vector2, msg: String) -> void:
	h.assert_true(got.is_equal_approx(want), "%s (got %v want %v)" % [msg, got, want])

func _initialize() -> void:
	var h := Harness.new()
	# contain-fit: zoom = min(vp.x/world.x, vp.y/world.y) / margin, uniform on both axes.
	_vec_approx(h, FitCamera2D.fit_zoom(Vector2(1000, 600), Vector2(1280, 720), 1.0), Vector2(1.2, 1.2), "1000x600 into 1280x720 -> min axis ratio 1.2")
	_vec_approx(h, FitCamera2D.fit_zoom(Vector2(320, 320), Vector2(1280, 720), 1.0), Vector2(2.25, 2.25), "320x320 into 1280x720 -> 2.25")
	# margin > 1 zooms out proportionally (adds padding).
	_vec_approx(h, FitCamera2D.fit_zoom(Vector2(1000, 600), Vector2(1280, 720), 2.0), Vector2(0.6, 0.6), "margin 2.0 halves the zoom")
	# degenerate inputs -> identity zoom, no divide-by-zero.
	_vec_approx(h, FitCamera2D.fit_zoom(Vector2(0, 600), Vector2(1280, 720), 1.0), Vector2.ONE, "zero world width -> identity")
	_vec_approx(h, FitCamera2D.fit_zoom(Vector2(1000, 600), Vector2(0, 0), 1.0), Vector2.ONE, "zero viewport -> identity")
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_fit_camera_2d.gd`
Expected: FAIL — the preload errors / script does not exist yet.

- [ ] **Step 3: Create `addons/godot_native_rl/camera/fit_camera_2d.gd`**

```gdscript
extends Camera2D
# Reusable 2D framing camera (#272) — the 2D analogue of OrbitCamera (#265). Centers on a
# configurable world_rect and zooms so the rect is contained in the viewport with a small margin,
# re-fitting whenever the window resizes. Cosmetic + inert headless (no rendering needed). Lets the
# 2D demos fill and center in a resizable window WITHOUT touching the simulation's arena_size.

# --- Pure helper (unit-testable; no tree needed) ---
# Camera2D zoom semantics: visible world size = viewport_size / zoom. To CONTAIN world_size we need
# zoom <= viewport/world on both axes -> take the min ratio. margin (>=1) divides the zoom to add
# breathing room. Returns Vector2.ONE on degenerate input (no divide-by-zero).
static func fit_zoom(world_size: Vector2, viewport_size: Vector2, margin: float = 1.1) -> Vector2:
	if world_size.x <= 0.0 or world_size.y <= 0.0 or viewport_size.x <= 0.0 or viewport_size.y <= 0.0:
		return Vector2.ONE
	var z := minf(viewport_size.x / world_size.x, viewport_size.y / world_size.y) / maxf(margin, 0.0001)
	return Vector2(z, z)

# --- Configuration ---
@export var world_rect := Rect2(0, 0, 1000, 600)  ## the world region to center + fit (in world units)
@export var margin := 1.1                           ## >1 adds padding around world_rect

func _ready() -> void:
	_apply()
	var vp := get_viewport()
	if vp != null and not vp.size_changed.is_connected(_on_resize):
		vp.size_changed.connect(_on_resize)
	make_current()

func _on_resize() -> void:
	_apply()

func _apply() -> void:
	position = world_rect.position + world_rect.size * 0.5
	var vp := get_viewport()
	if vp != null:
		zoom = fit_zoom(world_rect.size, vp.get_visible_rect().size, margin)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_fit_camera_2d.gd`
Expected: PASS — `Results: 5 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/camera/fit_camera_2d.gd test/unit/test_fit_camera_2d.gd
git commit -m "feat(camera): reusable FitCamera2D centered zoom-fit node (#272)"
```

---

## Task 4: Wire `FitCamera2D` into the six 2D play scenes + scene-structure test

**Files:**
- Modify: `examples/chase_the_target/chase_the_target.tscn`
- Modify: `examples/ball_chase/ball_chase.tscn`
- Modify: `examples/hide_and_seek/hide_and_seek_multipolicy.tscn`
- Modify: `examples/coop_collect/coop_collect.tscn`
- Modify: `examples/gridworld/gridworld.tscn`
- Modify: `examples/visual_chase/visual_chase.tscn`
- Create: `test/unit/test_fit_camera_2d_in_scenes.gd`

`chase_the_target_debug.tscn` instances `chase_the_target.tscn`, so it inherits the camera — do NOT
add one there.

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_fit_camera_2d_in_scenes.gd`:

```gdscript
extends SceneTree
# Structure regression (#272): each 2D play scene carries exactly one FitCamera2D. Scenes are
# instantiated WITHOUT entering the tree (no _ready / no ncnn) — we only assert the node is wired in.
# chase_the_target_debug.tscn inherits chase's camera via instancing, so it must also report one.

const Harness = preload("res://test/harness.gd")
const FITCAM := "res://addons/godot_native_rl/camera/fit_camera_2d.gd"

const SCENES: Array[String] = [
	"res://examples/chase_the_target/chase_the_target.tscn",
	"res://examples/chase_the_target/chase_the_target_debug.tscn",
	"res://examples/ball_chase/ball_chase.tscn",
	"res://examples/hide_and_seek/hide_and_seek_multipolicy.tscn",
	"res://examples/coop_collect/coop_collect.tscn",
	"res://examples/gridworld/gridworld.tscn",
	"res://examples/visual_chase/visual_chase.tscn",
]

func _count(node: Node) -> int:
	var n := 0
	var s: Variant = node.get_script()
	if s != null and s.resource_path == FITCAM:
		n += 1
	for c in node.get_children():
		n += _count(c)
	return n

func _initialize() -> void:
	var h := Harness.new()
	for path in SCENES:
		var packed := load(path) as PackedScene
		h.assert_true(packed != null, "%s loads" % path)
		if packed == null:
			continue
		var rootn := packed.instantiate()
		h.assert_eq(_count(rootn), 1, "%s has exactly one FitCamera2D" % path)
		rootn.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_fit_camera_2d_in_scenes.gd`
Expected: FAIL — every scene reports `0` FitCamera2D nodes.

- [ ] **Step 3: Add a `FitCamera2D` to each of the six scenes**

For **each** of the six `.tscn` files, make three edits. Using the literal string ext_resource id
`"fitcam2d"` (format=3 allows string ids) keeps the edit identical across files and avoids
renumbering existing ids:

1. In the `[gd_scene load_steps=N format=3]` header, increment `N` by 1.
2. After the **last** existing `[ext_resource …]` line, add:

```
[ext_resource type="Script" path="res://addons/godot_native_rl/camera/fit_camera_2d.gd" id="fitcam2d"]
```

3. At the **end of the file** (so the camera is the LAST child of the root — preserving every
   existing `index=` override), append:

```
[node name="FitCamera2D" type="Camera2D" parent="."]
script = ExtResource("fitcam2d")
world_rect = Rect2(0, 0, 1000, 600)
```

Use these per-scene `world_rect` values (all 1000×600 except gridworld):

| Scene | `world_rect` line |
|---|---|
| `chase_the_target/chase_the_target.tscn` | `world_rect = Rect2(0, 0, 1000, 600)` |
| `ball_chase/ball_chase.tscn` | `world_rect = Rect2(0, 0, 1000, 600)` |
| `hide_and_seek/hide_and_seek_multipolicy.tscn` | `world_rect = Rect2(0, 0, 1000, 600)` |
| `coop_collect/coop_collect.tscn` | `world_rect = Rect2(0, 0, 1000, 600)` |
| `visual_chase/visual_chase.tscn` | `world_rect = Rect2(0, 0, 1000, 600)` |
| `gridworld/gridworld.tscn` | `world_rect = Rect2(0, 0, 320, 320)` |

- [ ] **Step 4: Run the structure test + the behavioral regressions**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_fit_camera_2d_in_scenes.gd`
Expected: PASS — every scene (including the debug scene) reports exactly one FitCamera2D.

Then catch any malformed `.tscn` (every curated scene must still load):

Run: `"$GODOT" --headless --path . --script res://test/unit/test_launcher.gd`
Expected: PASS.

(The full trained-net behavioral + golden regressions run in Task 5 via `run_tests.sh`. The camera
is cosmetic and cannot change obs/actions, but the suite is the authoritative confirmation.)

- [ ] **Step 5: Commit**

```bash
git add examples/chase_the_target/chase_the_target.tscn examples/ball_chase/ball_chase.tscn \
        examples/hide_and_seek/hide_and_seek_multipolicy.tscn examples/coop_collect/coop_collect.tscn \
        examples/gridworld/gridworld.tscn examples/visual_chase/visual_chase.tscn \
        test/unit/test_fit_camera_2d_in_scenes.gd
git commit -m "feat(examples): center+fit 2D demos with FitCamera2D (#272)"
```

---

## Task 5: Full suite, visual verification, docs, close issues

**Files:**
- Modify: `README.md` (only if a screenshot/framing note is warranted)
- Modify: `CLAUDE.md` (one line if a convention changed — likely the FitCamera2D library mention)
- Modify: `docs/BACKLOG.md` (tick #271/#272 if listed)

- [ ] **Step 1: Run the full suite**

Run: `./test/run_tests.sh`
Expected: ends with `All tests passed.` (the four new unit tests are auto-discovered by the
`test/unit/test_*.gd` glob).

- [ ] **Step 2: Visual verification (windowed, NOT headless)**

Launch the launcher and a 2D demo windowed and resize each. Capture **≥3 screenshots** at
different window sizes (e.g. 1280×720, 1920×1080, and a tall/narrow window):

```bash
"$GODOT" --path . --resolution 1280x720 res://examples/launcher.tscn
"$GODOT" --path . --resolution 1920x1080 res://examples/gridworld/gridworld.tscn
```

Confirm: launcher menu is legible and centered; 2D demos are centered and fill the window without
clipping or distortion; gridworld is comfortably large. Spot-check one 3D demo (e.g.
`rover_3d.tscn`) still frames correctly.

- [ ] **Step 3: Update docs + backlog**

- In `CLAUDE.md`, add `camera/` (FitCamera2D / OrbitCamera) to the library description if not
  already covered, noting the 2D demos use a centered fit-camera.
- Tick #271 and #272 in `docs/BACKLOG.md` if they are listed there.
- Add a one-line note to `README.md` only if framing changes warrant a refreshed screenshot.

- [ ] **Step 4: Commit docs**

```bash
git add CLAUDE.md docs/BACKLOG.md README.md
git commit -m "docs: note responsive window + FitCamera2D; tick #271/#272"
```

- [ ] **Step 5: Open the PR**

Push the branch and open a PR that `Closes #271` and `Closes #272`. Verify CI is green before merge
(the local project.godot edits are NOT part of the branch — confirm `git status` still shows them as
local-only).

---

## Notes for the implementer

- **Set `GODOT`** to your engine binary first (e.g. `export GODOT=/opt/homebrew/bin/godot-mono`);
  `run_tests.sh` honors it. `class_name` is unreliable headless — these tests/scripts use
  path-based `preload`/`extends`, keep it that way.
- **Never** stage `project.godot` (beyond the Task 1 stash-scoped commit), `export_presets.cfg`, or
  `examples/fly_by/sky.hdr.import` — they carry local-only edits.
- The simulation `arena_size` on each game is **read-only** for this work — changing it would alter
  observations and break the trained nets. `FitCamera2D.world_rect` mirrors it but never writes it.
