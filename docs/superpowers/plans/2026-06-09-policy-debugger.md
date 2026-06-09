# In-Editor Policy Debugger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a drop-in, in-game overlay that visualizes live observation, action probabilities, model identity, and optional game status during native ncnn inference.

**Architecture:** The shared `NcnnControllerCore` emits an immutable `inference_step` payload (declared as a signal on the controller node base classes) right after `run_inference`. A pure static helper `PolicyDebug` turns that payload + identity + status into display lines (all the testable math). A thin `PolicyDebugOverlay` (`CanvasLayer`) auto-discovers controllers, polls an optional `get_debug_status()` hook, and renders via the helper. Toggle key + debug-build gate make it ship-safe.

**Tech Stack:** GDScript (Godot 4.5+), the project's dependency-free headless harness (`test/harness.gd`), native ncnn via the existing `NcnnRunner` GDExtension.

**Spec:** `docs/superpowers/specs/2026-06-09-policy-debugger-design.md`

**Conventions to honor (from CLAUDE.md / memory):**
- GDScript uses **TAB** indentation.
- Prefer path-based `extends`/`preload` over bare `class_name` for things used headlessly.
- `test/unit/test_*.gd` are auto-discovered by `test/run_tests.sh`; each `extends SceneTree`, builds a `Harness`, and **must** reach `h.finish(self)` (which calls `tree.quit`) or headless Godot hangs forever.
- Godot 4.6 `:=` can't infer from an untyped value — annotate the local type explicitly.
- Run a single unit test with: `"$GODOT" --headless --path . --script res://test/unit/<file>.gd` (set `GODOT` to your binary; probe with `which godot godot-mono`).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` (modify) | declare `signal inference_step(debug: Dictionary)` |
| `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd` (modify) | declare `signal inference_step(debug: Dictionary)` |
| `addons/godot_native_rl/controllers/ncnn_controller_core.gd` (modify) | build + emit the `debug` payload in `choose_and_apply_action` |
| `addons/godot_native_rl/debug/policy_debug.gd` (create) | pure formatting: header, status, obs, action rows, bars, composer |
| `addons/godot_native_rl/debug/policy_debug_overlay.gd` (create) | `CanvasLayer` node: discover, connect, poll status, render, toggle, gate |
| `examples/chase_the_target/chase_agent.gd` (modify) | optional `get_debug_status()` (distance + reward) |
| `examples/chase_the_target/chase_the_target_debug.tscn` (create) | demo scene = inference scene instance + overlay |
| `test/unit/test_policy_debug.gd` (create) | headless tests for the pure helper |
| `test/unit/test_policy_debug_emit.gd` (create) | headless test for the core signal emission |
| `test/unit/test_policy_debug_overlay.gd` (create) | headless test for overlay discovery + render |
| `README.md`, `CLAUDE.md`, `docs/BACKLOG.md` (modify) | document + flip backlog item 49 |

---

## Task 1: Declare the `inference_step` signal on both controller base classes

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd`
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd`
- Test: `test/unit/test_policy_debug_emit.gd` (signal-existence portion)

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_policy_debug_emit.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const Controller2D = preload("res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd")
const Controller3D = preload("res://addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd")

func _initialize() -> void:
	var h := Harness.new()

	var c2 = Controller2D.new()
	h.assert_true(c2.has_signal("inference_step"), "2D controller declares inference_step signal")
	c2.free()

	var c3 = Controller3D.new()
	h.assert_true(c3.has_signal("inference_step"), "3D controller declares inference_step signal")
	c3.free()

	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_policy_debug_emit.gd`
Expected: FAIL — "2D controller declares inference_step signal (expected true, got false)".

- [ ] **Step 3: Add the signal to the 2D controller**

In `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd`, immediately after the `const RecurrentState = preload(...)` line (line 7) and before the `enum ControlModes` line, add:

```gdscript

# Emitted once per inference decision with an immutable debug payload consumed by
# PolicyDebugOverlay. Keys: agent_name:String, obs:PackedFloat32Array (normalized vector fed to
# the net; [] on the image path), obs_image:Dictionary ({"w","h","c"} or {}), logits:PackedFloat32Array
# (raw network output, pre-decode), action_space:Dictionary, action:Dictionary (decoded),
# deterministic:bool. Inert when no listener is connected.
signal inference_step(debug: Dictionary)
```

- [ ] **Step 4: Add the signal to the 3D controller**

In `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd`, immediately after the `const RecurrentState = preload(...)` line (line 7) and before the `enum ControlModes` line, add the identical block:

```gdscript

# Emitted once per inference decision with an immutable debug payload consumed by
# PolicyDebugOverlay. Keys: agent_name:String, obs:PackedFloat32Array (normalized vector fed to
# the net; [] on the image path), obs_image:Dictionary ({"w","h","c"} or {}), logits:PackedFloat32Array
# (raw network output, pre-decode), action_space:Dictionary, action:Dictionary (decoded),
# deterministic:bool. Inert when no listener is connected.
signal inference_step(debug: Dictionary)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_policy_debug_emit.gd`
Expected: PASS (2 passed, 0 failed).

- [ ] **Step 6: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd \
        addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd \
        test/unit/test_policy_debug_emit.gd
git commit -m "feat: declare inference_step signal on ncnn controllers (#23)"
```

---

## Task 2: Emit the debug payload from `NcnnControllerCore`

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_controller_core.gd:84-115`
- Test: `test/unit/test_policy_debug_emit.gd` (extend it)

- [ ] **Step 1: Extend the test with a fake agent + runner that capture the emitted payload**

Replace the entire body of `test/unit/test_policy_debug_emit.gd` with:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const Controller2D = preload("res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd")
const Controller3D = preload("res://addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")

# Minimal fake runner: always "loaded", returns a fixed logit vector.
class FakeRunner:
	extends RefCounted
	var out := PackedFloat32Array([2.0, 0.0, -1.0])
	func is_model_loaded() -> bool:
		return true
	func run_inference(_v: PackedFloat32Array) -> PackedFloat32Array:
		return out

# Minimal fake agent Node that declares the signal and the controller contract.
class FakeAgent:
	extends Node2D
	signal inference_step(debug: Dictionary)
	var last_action := {}
	func get_inference_image() -> Image:
		return null
	func get_obs() -> Dictionary:
		return {"obs": [0.5, -0.5]}
	func get_action_space() -> Dictionary:
		return {"move": {"size": 3, "action_type": "discrete"}}
	func set_action(action) -> void:
		last_action = action

func _initialize() -> void:
	var h := Harness.new()

	# --- signal existence on the real controllers ---
	var c2 = Controller2D.new()
	h.assert_true(c2.has_signal("inference_step"), "2D controller declares inference_step signal")
	c2.free()
	var c3 = Controller3D.new()
	h.assert_true(c3.has_signal("inference_step"), "3D controller declares inference_step signal")
	c3.free()

	# --- core emits the payload through the agent ---
	var captured := {"hit": false, "payload": {}}
	var agent := FakeAgent.new()
	agent.name = "Bot"
	agent.inference_step.connect(func(debug): captured["hit"] = true; captured["payload"] = debug)
	var core := NcnnControllerCore.new()
	core.choose_and_apply_action(agent, FakeRunner.new())

	h.assert_true(captured["hit"], "core emitted inference_step")
	var p: Dictionary = captured["payload"]
	h.assert_eq(p.get("agent_name", ""), "Bot", "payload agent_name")
	h.assert_eq(PackedFloat32Array(p.get("obs", [])), PackedFloat32Array([0.5, -0.5]), "payload obs vector")
	h.assert_eq(PackedFloat32Array(p.get("logits", [])), PackedFloat32Array([2.0, 0.0, -1.0]), "payload raw logits")
	h.assert_eq(int(p.get("action", {}).get("move", -1)), 0, "payload decoded action (argmax of logits)")
	h.assert_true(p.get("action_space", {}).has("move"), "payload action_space present")
	h.assert_true((p.get("obs_image", {}) as Dictionary).is_empty(), "payload obs_image empty on float path")
	agent.free()

	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_policy_debug_emit.gd`
Expected: FAIL — "core emitted inference_step (expected true, got false)" (signal not emitted yet).

- [ ] **Step 3: Implement the emission in the core**

In `addons/godot_native_rl/controllers/ncnn_controller_core.gd`, modify `choose_and_apply_action` (lines 84-115). Add two capture locals at the top, set them in each branch, and emit after a successful decode. The full updated method:

```gdscript
func choose_and_apply_action(agent, runner) -> void:
	if runner == null or not runner.is_model_loaded():
		return
	var output: PackedFloat32Array
	var debug_obs := PackedFloat32Array()
	var debug_img := {}
	var img: Image = agent.get_inference_image()
	if img != null:
		if not recurrent_contract.is_empty() and not _warned_image_recurrent:
			push_warning("NcnnControllerCore: a recurrent contract is set but get_inference_image() returned a frame — recurrent hidden state is NOT used on the image path (float-obs only).")
			_warned_image_recurrent = true
		output = runner.run_inference_image(img, true)
		debug_img = {"w": img.get_width(), "h": img.get_height(), "c": 0}
	else:
		var obs_dict: Dictionary = agent.get_obs()
		assert("obs" in obs_dict, "get_obs() must return a dictionary with an 'obs' key")
		var obs_vec := PackedFloat32Array(obs_dict["obs"])
		if not obs_norm_stats.is_empty():
			obs_vec = ObsNormalize.normalize(obs_vec, obs_norm_stats["mean"], obs_norm_stats["var"],
				obs_norm_stats["epsilon"], obs_norm_stats["clip_obs"])
			if obs_vec.is_empty():
				push_error("NcnnControllerCore.choose_and_apply_action: obs normalization failed (size mismatch); skipping action.")
				return
		if recurrent_contract.is_empty():
			output = runner.run_inference(obs_vec)
		else:
			output = _run_recurrent_and_advance(runner, obs_vec)
		debug_obs = obs_vec
	var action: Dictionary = ActionDecode.decode_actions(output, agent.get_action_space(), deterministic_inference, rng)
	if action.is_empty():
		push_error("NcnnControllerCore.choose_and_apply_action: action decode failed (empty/mismatched output); skipping action.")
		return
	_emit_debug(agent, debug_obs, debug_img, output, action)
	agent.set_action(action)
```

Then add this helper method directly below `choose_and_apply_action` (before the `obs_space_from_obs` static function at line 120):

```gdscript
# Emit the immutable debug payload through the agent (the node owns the signal; the core is
# node-agnostic). Inert when nothing declares/listens for the signal — only the small Dictionary
# is built, at decision cadence. See PolicyDebugOverlay for the consumer.
func _emit_debug(agent, obs_vec: PackedFloat32Array, obs_image: Dictionary, logits: PackedFloat32Array, action: Dictionary) -> void:
	if not agent.has_signal("inference_step"):
		return
	agent.emit_signal("inference_step", {
		"agent_name": String(agent.name),
		"obs": obs_vec,
		"obs_image": obs_image,
		"logits": logits,
		"action_space": agent.get_action_space(),
		"action": action,
		"deterministic": deterministic_inference,
	})
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_policy_debug_emit.gd`
Expected: PASS (8 passed, 0 failed).

- [ ] **Step 5: Run the two inference golden tests to confirm no regression**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_chase_golden_inference.gd`
Run: `"$GODOT" --headless --path . --script res://test/unit/test_controller_inference.gd`
Expected: both PASS (the emission is additive; behavior unchanged for agents without a listener).

- [ ] **Step 6: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_controller_core.gd \
        test/unit/test_policy_debug_emit.gd
git commit -m "feat: emit inference_step debug payload from controller core (#23)"
```

---

## Task 3: The pure `PolicyDebug` formatting helper

**Files:**
- Create: `addons/godot_native_rl/debug/policy_debug.gd`
- Test: `test/unit/test_policy_debug.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_policy_debug.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const PolicyDebug = preload("res://addons/godot_native_rl/debug/policy_debug.gd")

# Join a PackedStringArray into one string for substring assertions.
func _joined(lines: PackedStringArray) -> String:
	return "\n".join(lines)

func _initialize() -> void:
	var h := Harness.new()

	# --- bar(): magnitude in [0,1] -> fill chars, clamped ---
	h.assert_eq(PolicyDebug.bar(0.0, 8), "", "bar 0 -> empty")
	h.assert_eq(PolicyDebug.bar(1.0, 8).length(), 8, "bar 1 -> full width")
	h.assert_eq(PolicyDebug.bar(-1.0, 8).length(), 8, "bar uses magnitude (negative)")
	h.assert_eq(PolicyDebug.bar(5.0, 8).length(), 8, "bar clamps above 1")
	h.assert_eq(PolicyDebug.bar(0.5, 8).length(), 4, "bar 0.5 -> half width")

	# --- header_line() ---
	var hdr := PolicyDebug.header_line({"policy_name": "shared_policy", "model": "chase.ncnn.param", "deterministic": true, "seed": -1})
	h.assert_true(hdr.contains("shared_policy") and hdr.contains("chase.ncnn.param") and hdr.contains("det"),
		"header shows policy, model, det")
	var hdr2 := PolicyDebug.header_line({"policy_name": "p", "model": "m", "deterministic": false, "seed": 7})
	h.assert_true(hdr2.contains("stochastic"), "header shows stochastic when not deterministic")

	# --- status_rows() ---
	h.assert_eq(PolicyDebug.status_rows({}).size(), 0, "empty status -> no rows")
	var srows := PolicyDebug.status_rows({"dist": 0.34, "step": 87})
	h.assert_true(_joined(srows).contains("dist") and _joined(srows).contains("0.34") and _joined(srows).contains("87"),
		"status rows render labels and values")

	# --- obs_rows() ---
	var orows := PolicyDebug.obs_rows(PackedFloat32Array([0.5, -0.25]), 8)
	h.assert_true(orows[0].contains("OBS (2)"), "obs header shows count")
	h.assert_true(_joined(orows).contains("[0]") and _joined(orows).contains("[1]"), "obs rows indexed")

	# --- action_rows(): discrete, chosen marker from decoded action ---
	var arows := PolicyDebug.action_rows(
		PackedFloat32Array([2.0, 0.0, -1.0]),
		{"move": {"size": 3, "action_type": "discrete"}},
		{"move": 0},
		8)
	var ajoined := _joined(arows)
	h.assert_true(ajoined.contains("move (discrete, 3)"), "discrete action header")
	h.assert_true(ajoined.contains("chosen"), "discrete action marks the chosen index")
	# index 0 has the largest logit -> highest probability; it is the chosen one.
	h.assert_true(arows[1].contains("chosen"), "chosen marker on the argmax row")

	# --- action_rows(): continuous with squash ---
	var crows := PolicyDebug.action_rows(
		PackedFloat32Array([0.0, 10.0]),
		{"steer": {"size": 2, "action_type": "continuous", "squash": true}},
		{"steer": [0.0, 1.0]},
		8)
	var cjoined := _joined(crows)
	h.assert_true(cjoined.contains("steer (continuous, 2") and cjoined.contains("tanh"), "continuous squash header + tanh")

	# --- action_rows(): logits/action_space size mismatch is flagged, no crash ---
	var mrows := PolicyDebug.action_rows(
		PackedFloat32Array([1.0]),
		{"move": {"size": 3, "action_type": "discrete"}},
		{"move": 0},
		8)
	h.assert_true(_joined(mrows).contains("mismatch"), "size mismatch flagged")

	# --- render_lines(): image-obs path shows dims, skips obs vector ---
	var img_lines := PolicyDebug.render_lines(
		{"agent_name": "Cam", "obs": PackedFloat32Array(), "obs_image": {"w": 84, "h": 84, "c": 0},
		 "logits": PackedFloat32Array([1.0, 2.0]), "action_space": {"a": {"size": 2, "action_type": "discrete"}},
		 "action": {"a": 1}, "deterministic": true},
		{"policy_name": "p", "model": "m", "deterministic": true, "seed": -1},
		{},
		8)
	var ijoined := _joined(img_lines)
	h.assert_true(ijoined.contains("Cam"), "render shows agent name")
	h.assert_true(ijoined.contains("84") and ijoined.contains("OBS image"), "render shows image dims on image path")
	h.assert_true(not ijoined.contains("OBS (0)"), "render skips numeric obs section on image path")

	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_policy_debug.gd`
Expected: FAIL — script load error / "bar 0 -> empty" failing (helper does not exist yet).

- [ ] **Step 3: Implement the helper**

Create `addons/godot_native_rl/debug/policy_debug.gd`:

```gdscript
class_name PolicyDebug
extends RefCounted

# Pure, node-free formatting for the in-game Policy Debugger overlay. Turns an inference_step
# payload + identity + optional game status into display lines. All probability/segmentation math
# lives here so it is unit-testable headless; PolicyDebugOverlay only routes data and renders.

const InferenceMath = preload("res://addons/godot_native_rl/controllers/inference_math.gd")
const FILL := "#"

# Magnitude bar: |value| clamped to [0,1] -> rounded count of fill chars (0..width).
static func bar(value: float, width: int) -> String:
	var mag := clampf(absf(value), 0.0, 1.0)
	var n := int(round(mag * float(width)))
	return FILL.repeat(n)

static func _fmt_num(v) -> String:
	if typeof(v) == TYPE_FLOAT:
		return "%.2f" % v
	return str(v)

static func _fmt_signed(v: float) -> String:
	return "%+.2f" % v

# Header: "policy: <name>   model: <basename>   <det|stochastic>".
static func header_line(identity: Dictionary) -> String:
	var policy: String = str(identity.get("policy_name", "?"))
	var model: String = str(identity.get("model", "?"))
	var det: bool = bool(identity.get("deterministic", true))
	return "policy: %s   model: %s   %s" % [policy, model, "det" if det else "stochastic"]

# One STATUS line of "label value" pairs; empty status -> no rows.
static func status_rows(status: Dictionary) -> PackedStringArray:
	var out := PackedStringArray()
	if status.is_empty():
		return out
	var parts := PackedStringArray()
	for key in status.keys():
		parts.append("%s %s" % [str(key), _fmt_num(status[key])])
	out.append("STATUS   " + "   ".join(parts))
	return out

# "OBS (n)" header + one indexed, signed, bar-annotated row per element.
static func obs_rows(obs: PackedFloat32Array, bar_width: int) -> PackedStringArray:
	var out := PackedStringArray()
	out.append("OBS (%d)" % obs.size())
	for i in range(obs.size()):
		out.append("  [%d] %s  %s" % [i, _fmt_signed(obs[i]), bar(obs[i], bar_width)])
	return out

# Per action key (insertion order over the logit vector): discrete -> softmax % + bar + chosen
# marker (from the decoded action); continuous -> raw (+ tanh when squash) + bar. A size mismatch
# is flagged inline and stops further walking (never indexes out of range).
static func action_rows(logits: PackedFloat32Array, action_space: Dictionary, action: Dictionary, bar_width: int) -> PackedStringArray:
	var out := PackedStringArray()
	var index := 0
	for key in action_space.keys():
		var entry: Dictionary = action_space[key]
		var size: int = int(entry.get("size", 0))
		var action_type: String = str(entry.get("action_type", "discrete"))
		if size <= 0 or index + size > logits.size():
			out.append("ACTION  %s  [logits/action_space size mismatch]" % str(key))
			return out
		var segment: PackedFloat32Array = logits.slice(index, index + size)
		if action_type == "discrete":
			out.append("ACTION  %s (discrete, %d)" % [str(key), size])
			var probs := InferenceMath.softmax(segment)
			var chosen: int = int(action.get(key, -1))
			for i in range(size):
				var marker := "  <-chosen" if i == chosen else ""
				out.append("  %d  %3d%%  %s%s" % [i, int(round(probs[i] * 100.0)), bar(probs[i], bar_width), marker])
		elif action_type == "continuous":
			var squash: bool = bool(entry.get("squash", false))
			out.append("ACTION  %s (continuous, %d%s)" % [str(key), size, ", tanh" if squash else ""])
			for i in range(size):
				var raw := segment[i]
				if squash:
					out.append("  [%d] raw %s  tanh %s  %s" % [i, _fmt_signed(raw), _fmt_signed(tanh(raw)), bar(tanh(raw), bar_width)])
				else:
					out.append("  [%d] %s  %s" % [i, _fmt_signed(raw), bar(raw, bar_width)])
		else:
			out.append("ACTION  %s  [unknown action_type '%s']" % [str(key), action_type])
		index += size
	return out

# Top-level composer used by the overlay: title + header + status + obs (or image dims) + actions.
static func render_lines(debug: Dictionary, identity: Dictionary, status: Dictionary, bar_width: int) -> PackedStringArray:
	var out := PackedStringArray()
	out.append("POLICY DEBUG  -  %s" % str(debug.get("agent_name", "?")))
	out.append(header_line(identity))
	out.append_array(status_rows(status))
	var obs: PackedFloat32Array = debug.get("obs", PackedFloat32Array())
	var obs_image: Dictionary = debug.get("obs_image", {})
	if obs.is_empty() and not obs_image.is_empty():
		out.append("OBS image  %dx%dx%d" % [int(obs_image.get("w", 0)), int(obs_image.get("h", 0)), int(obs_image.get("c", 0))])
	else:
		out.append_array(obs_rows(obs, bar_width))
	out.append_array(action_rows(
		debug.get("logits", PackedFloat32Array()),
		debug.get("action_space", {}),
		debug.get("action", {}),
		bar_width))
	return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_policy_debug.gd`
Expected: PASS (all assertions pass, 0 failed).

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/debug/policy_debug.gd test/unit/test_policy_debug.gd
git commit -m "feat: PolicyDebug pure formatting helper (#23)"
```

---

## Task 4: The `PolicyDebugOverlay` node

**Files:**
- Create: `addons/godot_native_rl/debug/policy_debug_overlay.gd`
- Test: `test/unit/test_policy_debug_overlay.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_policy_debug_overlay.gd`. It adds a fake controller under the scene root, then the overlay (so auto-discovery finds it), emits a payload, and asserts `build_text()` renders it. `debug_build_only` is set false so `_ready` never frees the node.

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const PolicyDebugOverlay = preload("res://addons/godot_native_rl/debug/policy_debug_overlay.gd")

# Fake controller: declares the signal, exposes identity props + an optional status hook.
class FakeController:
	extends Node2D
	signal inference_step(debug: Dictionary)
	var policy_name := "shared_policy"
	var model_param_path := "res://models/chase.ncnn.param"
	var deterministic_inference := true
	var inference_seed := -1
	func get_debug_status() -> Dictionary:
		return {"dist": 0.34}

func _initialize() -> void:
	var h := Harness.new()
	var root := get_root()

	# --- _basename(): static path -> file name ---
	h.assert_eq(PolicyDebugOverlay._basename("res://models/chase.ncnn.param"), "chase.ncnn.param", "_basename extracts file")
	h.assert_eq(PolicyDebugOverlay._basename(""), "?", "_basename empty -> ?")

	# Fake controller must be in the tree BEFORE the overlay so auto-discovery finds it.
	var ctrl := FakeController.new()
	ctrl.name = "Bot"
	root.add_child(ctrl)

	var overlay := PolicyDebugOverlay.new()
	overlay.debug_build_only = false   # do not free in _ready regardless of build type
	overlay.start_visible = true
	root.add_child(overlay)            # _ready() runs discovery + connects

	# Emit a payload as the core would.
	ctrl.inference_step.emit({
		"agent_name": "Bot",
		"obs": PackedFloat32Array([0.5, -0.5]),
		"obs_image": {},
		"logits": PackedFloat32Array([2.0, 0.0, -1.0]),
		"action_space": {"move": {"size": 3, "action_type": "discrete"}},
		"action": {"move": 0},
		"deterministic": true,
	})

	var text := overlay.build_text()
	h.assert_true(text.contains("POLICY DEBUG  -  Bot"), "overlay renders agent title")
	h.assert_true(text.contains("shared_policy") and text.contains("chase.ncnn.param"), "overlay renders identity header")
	h.assert_true(text.contains("STATUS") and text.contains("dist") and text.contains("0.34"), "overlay renders polled status")
	h.assert_true(text.contains("move (discrete, 3)") and text.contains("chosen"), "overlay renders action rows")

	overlay.free()
	ctrl.free()
	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_policy_debug_overlay.gd`
Expected: FAIL — script load error (overlay does not exist yet).

- [ ] **Step 3: Implement the overlay**

Create `addons/godot_native_rl/debug/policy_debug_overlay.gd`:

```gdscript
class_name PolicyDebugOverlay
extends CanvasLayer

# Drop-in in-game overlay for the Policy Debugger. Add one node to a scene running ncnn inference:
# with `controllers` empty it auto-discovers every node that emits `inference_step`; otherwise it
# tracks the listed controllers. Press `toggle_key` to show/hide. With `debug_build_only` it frees
# itself in release exports, so it is safe to leave in a shipped scene. Rendering math is in the
# pure PolicyDebug helper.

const PolicyDebug = preload("res://addons/godot_native_rl/debug/policy_debug.gd")

@export var controllers: Array[NodePath] = []   # empty = auto-discover all inference_step emitters
@export var toggle_key: Key = KEY_F3
@export var start_visible: bool = false
@export var debug_build_only: bool = true       # free in release exports (OS.is_debug_build() == false)
@export var bar_width: int = 8

var _panel: PanelContainer = null
var _label: Label = null
var _tracked: Array = []          # controller Node refs
var _latest: Dictionary = {}      # instance_id -> debug payload
var _identities: Dictionary = {}  # instance_id -> identity dict

func _ready() -> void:
	if debug_build_only and not OS.is_debug_build():
		queue_free()
		return
	_build_ui()
	_resolve_controllers()
	_connect_controllers()
	_set_visible(start_visible)

func _build_ui() -> void:
	_panel = PanelContainer.new()
	_panel.position = Vector2(8, 8)
	add_child(_panel)
	var margin := MarginContainer.new()
	for side in ["left", "top", "right", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 8)
	_panel.add_child(margin)
	_label = Label.new()
	margin.add_child(_label)

func _resolve_controllers() -> void:
	_tracked.clear()
	if controllers.is_empty():
		_discover_all(get_tree().get_root())
		return
	for path in controllers:
		var node := get_node_or_null(path)
		if node == null or not node.has_signal("inference_step"):
			push_warning("PolicyDebugOverlay: '%s' is not a controller emitting inference_step; skipping." % str(path))
			continue
		_tracked.append(node)

func _discover_all(node: Node) -> void:
	if node != self and node.has_signal("inference_step"):
		_tracked.append(node)
	for child in node.get_children():
		_discover_all(child)

func _connect_controllers() -> void:
	for c in _tracked:
		var id: int = c.get_instance_id()
		_identities[id] = _identity_of(c)
		c.connect("inference_step", _on_inference_step.bind(id))

func _identity_of(c) -> Dictionary:
	return {
		"policy_name": c.get("policy_name") if c.get("policy_name") != null else "?",
		"model": _basename(c.get("model_param_path")),
		"deterministic": c.get("deterministic_inference") if c.get("deterministic_inference") != null else true,
		"seed": c.get("inference_seed") if c.get("inference_seed") != null else -1,
	}

static func _basename(path) -> String:
	if path == null or String(path).is_empty():
		return "?"
	return String(path).get_file()

func _on_inference_step(debug: Dictionary, id: int) -> void:
	_latest[id] = debug

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == toggle_key:
		_set_visible(not _panel.visible)

func _set_visible(v: bool) -> void:
	if _panel != null:
		_panel.visible = v

# Build the full overlay text from the latest payloads + freshly polled status. Pure-ish: only
# reads node state, no side effects — exposed so it is unit-testable headless.
func build_text() -> String:
	var lines := PackedStringArray()
	for c in _tracked:
		var id: int = c.get_instance_id()
		if not _latest.has(id):
			continue
		var status: Dictionary = {}
		if c.has_method("get_debug_status"):
			var s = c.get_debug_status()
			if s is Dictionary:
				status = s
		lines.append_array(PolicyDebug.render_lines(_latest[id], _identities[id], status, bar_width))
		lines.append("")
	return "\n".join(lines)

func _process(_delta: float) -> void:
	if _panel == null or not _panel.visible:
		return
	_label.text = build_text()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_policy_debug_overlay.gd`
Expected: PASS (all assertions, 0 failed).

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/debug/policy_debug_overlay.gd test/unit/test_policy_debug_overlay.gd
git commit -m "feat: PolicyDebugOverlay drop-in in-game node (#23)"
```

---

## Task 5: Wire the chase example + debug scene + docs

**Files:**
- Modify: `examples/chase_the_target/chase_agent.gd`
- Create: `examples/chase_the_target/chase_the_target_debug.tscn`
- Modify: `README.md`, `CLAUDE.md`, `docs/BACKLOG.md`

- [ ] **Step 1: Add the optional status hook to ChaseAgent**

In `examples/chase_the_target/chase_agent.gd`, add this method after `get_reward()` (after line 64):

```gdscript

# Optional Policy Debugger hook (duck-typed; not part of the controller contract). Surfaces
# game-specific progress in the PolicyDebugOverlay STATUS line. Absent -> no STATUS section.
func get_debug_status() -> Dictionary:
	if _game == null:
		return {}
	return {"dist_to_target": _game.distance(), "episode_reward": reward, "step": n_steps}
```

- [ ] **Step 2: Verify it parses (headless import) and the existing chase tests still pass**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_chase_golden_inference.gd`
Expected: PASS (the hook is additive; inference unchanged).

- [ ] **Step 3: Author the debug scene**

Create `examples/chase_the_target/chase_the_target_debug.tscn` — instance the existing inference scene and add the overlay (visible by default for the demo):

```
[gd_scene load_steps=3 format=3]

[ext_resource type="PackedScene" path="res://examples/chase_the_target/chase_the_target.tscn" id="1"]
[ext_resource type="Script" path="res://addons/godot_native_rl/debug/policy_debug_overlay.gd" id="2"]

[node name="ChaseDebug" type="Node"]

[node name="ChaseTheTarget" parent="." instance=ExtResource("1")]

[node name="PolicyDebugOverlay" type="CanvasLayer" parent="."]
script = ExtResource("2")
start_visible = true
debug_build_only = false
```

(`controllers` left empty → the overlay auto-discovers the `ChaseAgent` inside the instanced scene. `debug_build_only = false` so the demo overlay shows even in a release web export.)

- [ ] **Step 4: Run the full headless suite to confirm nothing regressed**

Run: `./test/run_tests.sh`
Expected: ends with `All tests passed.` and exit code 0. (Gate on that line / exit code — do NOT grep for `failed`/`ERROR`, which appear in passing runs.)

- [ ] **Step 5: Manual non-headless verification (the issue's explicit requirement)**

This cannot run headless — it needs a viewport. Perform it in a desktop Godot:

```
<godot-binary> --path . res://examples/chase_the_target/chase_the_target_debug.tscn
```

Confirm and screenshot:
- the overlay panel renders top-left with the agent title + identity header (`policy: ... model: chase_the_target.ncnn.param ...`);
- the STATUS line shows `dist_to_target`, `episode_reward`, `step`, updating as the agent moves;
- the OBS rows and the `move (discrete, 5)` action rows update live, with `<-chosen` tracking the taken action;
- pressing **F3** toggles the overlay off/on.

If the base inference scene needs anything to drive decisions that the trained-chase demo doesn't, note it — but it is the same scene the project already uses for trained inference, so it should run as-is.

- [ ] **Step 6: Update docs and the backlog**

1. In `docs/BACKLOG.md`, tick item **49** (In-editor Policy Debugger) to done.
2. In `README.md`, add a short "Policy Debugger" blurb under the features/usage section: "Drop a `PolicyDebugOverlay` (`addons/godot_native_rl/debug/policy_debug_overlay.gd`) into a scene running ncnn inference — it auto-discovers your agents and overlays live observations, action probabilities, the loaded policy/model, and any `get_debug_status()` you expose. Press F3 to toggle; it auto-hides in release builds unless `debug_build_only` is off. See `examples/chase_the_target/chase_the_target_debug.tscn`."
3. In `CLAUDE.md`, add item 49 to the **Done** list in the roadmap section (keyed by backlog item number), matching the existing entry style, e.g.: `49 (in-editor Policy Debugger — drop-in PolicyDebugOverlay + inference_step signal + pure PolicyDebug formatter; live obs/action-probs/identity/get_debug_status overlay, headless helper tests + chase debug scene),`.

- [ ] **Step 7: Commit**

```bash
git add examples/chase_the_target/chase_agent.gd \
        examples/chase_the_target/chase_the_target_debug.tscn \
        README.md CLAUDE.md docs/BACKLOG.md
git commit -m "feat: chase Policy Debugger demo scene + docs (Closes #23)"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- In-game overlay, F6/web-capable → Task 4 (`CanvasLayer`) + Task 5 (debug scene). ✓
- Signal + separate overlay node → Task 1 (signal) + Task 2 (emit) + Task 4 (overlay). ✓
- Raw payload, helper computes softmax → Task 2 payload (raw `logits`) + Task 3 `action_rows`. ✓
- Obs vector + both action types + multi-key + chosen marker → Task 3 tests/impl. ✓
- Identity header (policy/model/generation) → Task 4 `_identity_of` + Task 3 `header_line`. ✓
- Optional `get_debug_status()` hook → Task 4 `build_text` poll + Task 5 chase impl. ✓
- Auto-discover + toggle key + debug-build gate → Task 4 `_resolve_controllers`/`_unhandled_input`/`_ready`. ✓
- Headless unit tests + non-headless manual verification → Tasks 1-4 tests + Task 5 Step 5. ✓
- Error handling (null/non-controller, size mismatch, image path, missing hook) → Task 3 mismatch + image tests, Task 4 warn/skip. ✓
- No golden/regression scene changes → overlay only in the new `_debug.tscn`; Task 2 Step 5 + Task 5 Step 4 confirm. ✓

**Placeholder scan:** none — every code/test/command step is concrete.

**Type consistency:** `inference_step(debug: Dictionary)` payload keys identical across Task 2 emit and Task 3/4 consumers; `bar/header_line/status_rows/obs_rows/action_rows/render_lines` signatures match between helper impl (Task 3) and overlay/tests (Tasks 3-4); `build_text()` / `_basename()` names consistent between Task 4 impl and test.
