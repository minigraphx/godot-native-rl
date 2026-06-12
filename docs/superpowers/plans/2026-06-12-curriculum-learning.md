# Curriculum Learning Implementation Plan (#28)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Game-side curriculum learning (staged env params, performance-gated promotion) with an optional trainer-override wire message, demonstrated by a real trained chase run whose promotions gate on reward.

**Architecture:** Pure `Curriculum` logic (RefCounted, rolling-window promotion) + thin `CurriculumController` node that applies stage params to the game at episode-reset boundaries and reports via signal/info. `NcnnSync` gains one additive `"curriculum"` message; a stdlib Python client lets custom trainers drive it. Chase demo: 3 stages over `touch_radius` + `arena_size`.

**Tech Stack:** GDScript (TAB indent, path-based extends), `test/harness.gd` headless tests, stdlib Python `unittest`, existing godot_rl JSON wire framing.

**Spec:** `docs/superpowers/specs/2026-06-12-curriculum-learning-design.md`
**Run tests with:** `GODOT=/opt/homebrew/bin/godot-mono` (plain `godot` is not on PATH on this machine).

---

## File structure

- Create `addons/godot_native_rl/training/curriculum.gd` — pure stage/promotion logic.
- Create `addons/godot_native_rl/training/curriculum_controller.gd` — Node wrapper (apply/report/external-control).
- Modify `addons/godot_native_rl/sync.gd` — `"curriculum"` case in `handle_message()`.
- Create `scripts/curriculum_client.py` — stdlib send helpers for custom trainers.
- Create `examples/chase_the_target/chase_curriculum.json` — 3-stage demo config.
- Modify `examples/chase_the_target/chase_game.gd` — `apply_curriculum(params)`.
- Modify `examples/chase_the_target/chase_agent.gd` — episode reporting + `get_info()` stage.
- Create `examples/chase_the_target/chase_the_target_train_curriculum.tscn` — train-scene variant.
- Tests: `test/unit/test_curriculum.gd`, `test/unit/test_curriculum_controller.gd`,
  `test/integration/curriculum_smoke_checker.gd` + `curriculum_smoke_scene.tscn`,
  `test/python/test_curriculum_client.py`; extend `test/integration/run_protocol_test.py` for the wire path.
- Modify `test/run_tests.sh` — register the curriculum smoke.
- Docs: README, CLAUDE.md, gap-analysis, BACKLOG (item 52), `Closes #28`.

---

### Task 1: Pure curriculum logic

**Files:** Create `addons/godot_native_rl/training/curriculum.gd` · Test `test/unit/test_curriculum.gd`

- [ ] **Step 1: failing test** — `test/unit/test_curriculum.gd`:

```gdscript
extends SceneTree
# Unit tests for the pure Curriculum stage/promotion logic (no scene dependencies).

const Harness = preload("res://test/harness.gd")
const Curriculum = preload("res://addons/godot_native_rl/training/curriculum.gd")

func _stages() -> Array:
	return [
		{"name": "easy", "params": {"touch_radius": 120.0},
			"promote": {"metric": "mean_reward", "threshold": 5.0, "window": 4, "min_episodes": 3}},
		{"name": "mid", "params": {"touch_radius": 80.0},
			"promote": {"metric": "success_rate", "threshold": 0.75, "window": 4, "min_episodes": 4}},
		{"name": "hard", "params": {"touch_radius": 40.0}},
	]

func _initialize() -> void:
	var h = Harness.new()

	var c = Curriculum.new()
	h.assert_true(c.set_stages(_stages()), "valid stages accepted")
	h.assert_eq(c.stage_count(), 3, "3 stages")
	h.assert_eq(c.stage_index(), 0, "starts at stage 0")
	h.assert_eq(c.stage_name(), "easy", "stage 0 name")
	h.assert_eq(c.current_params()["touch_radius"], 120.0, "stage 0 params")
	h.assert_true(not c.is_final(), "not final at 0")

	# min_episodes gate: 2 great episodes are not enough (needs 3)
	c.record_episode(10.0, true)
	c.record_episode(10.0, true)
	h.assert_true(not c.should_promote(), "min_episodes gate holds")
	c.record_episode(10.0, true)
	h.assert_true(c.should_promote(), "mean_reward 10 >= 5 over 3 eps promotes")

	# advance clears the window
	h.assert_true(c.advance(), "advance to stage 1")
	h.assert_eq(c.stage_index(), 1, "now stage 1")
	h.assert_true(not c.should_promote(), "fresh window after advance")

	# success_rate metric: 3/4 successes = 0.75 >= 0.75
	c.record_episode(0.0, true)
	c.record_episode(0.0, true)
	c.record_episode(0.0, false)
	c.record_episode(0.0, true)
	h.assert_true(c.should_promote(), "success_rate 0.75 promotes")
	h.assert_true(c.advance(), "advance to final")
	h.assert_true(c.is_final(), "final stage reached")

	# final stage: never promotes, advance returns false
	c.record_episode(100.0, true)
	c.record_episode(100.0, true)
	c.record_episode(100.0, true)
	c.record_episode(100.0, true)
	h.assert_true(not c.should_promote(), "final stage never promotes")
	h.assert_true(not c.advance(), "advance refuses past final")

	# rolling window: low episodes push the good ones out
	var c2 = Curriculum.new()
	c2.set_stages(_stages())
	for i in range(4):
		c2.record_episode(10.0, true)
	h.assert_true(c2.should_promote(), "window full of 10s promotes")
	for i in range(4):
		c2.record_episode(0.0, false)
	h.assert_true(not c2.should_promote(), "window rolled to 0s: no promote")

	# set_stage jump + bounds
	h.assert_true(c2.set_stage(2), "set_stage in range")
	h.assert_eq(c2.stage_index(), 2, "jumped to 2")
	h.assert_true(not c2.set_stage(7), "set_stage out of range refused")
	h.assert_eq(c2.stage_index(), 2, "index unchanged after refusal")

	# malformed stages fail loud (return false)
	var bad = Curriculum.new()
	h.assert_true(not bad.set_stages([]), "empty stages rejected")
	h.assert_true(not bad.set_stages([{"params": {}}]), "missing name rejected")
	h.assert_true(not bad.set_stages([{"name": "x"}]), "missing params rejected")
	h.assert_true(not bad.set_stages([
		{"name": "a", "params": {}, "promote": {"metric": "bogus", "threshold": 1.0, "window": 2, "min_episodes": 1}},
		{"name": "b", "params": {}},
	]), "unknown metric rejected")

	h.finish(self)
```

- [ ] **Step 2:** run `"$GODOT" --headless --path . --script res://test/unit/test_curriculum.gd` → FAIL (missing file).
- [ ] **Step 3: implement** `addons/godot_native_rl/training/curriculum.gd`:

```gdscript
extends RefCounted
# Pure curriculum stage/promotion logic — no scene or node dependencies.
# Stages: ordered Array of { "name": String, "params": Dictionary,
#   "promote": { "metric": "mean_reward"|"success_rate", "threshold": float,
#                "window": int, "min_episodes": int } }   (final stage: no "promote")
# Promotion: over a rolling window of the last `window` episodes, promote when at least
# `min_episodes` have been recorded AND the metric clears `threshold`.
# Spec: docs/superpowers/specs/2026-06-12-curriculum-learning-design.md (#28)

const METRICS := ["mean_reward", "success_rate"]

var _stages: Array = []
var _index := 0
var _rewards: Array = []    # rolling window (parallel arrays)
var _successes: Array = []

func set_stages(stages: Array) -> bool:
	if stages.is_empty():
		push_error("Curriculum: stages must be a non-empty Array.")
		return false
	for s in stages:
		if not (s is Dictionary) or not s.has("name") or not s.has("params"):
			push_error("Curriculum: every stage needs 'name' and 'params'.")
			return false
		if s.has("promote"):
			var p = s["promote"]
			if not (p is Dictionary) or not METRICS.has(p.get("metric", "")) \
					or not p.has("threshold") or not p.has("window") or not p.has("min_episodes"):
				push_error("Curriculum: stage '%s' has a malformed 'promote' block." % s["name"])
				return false
	_stages = stages
	_index = 0
	_clear_window()
	return true

func stage_count() -> int:
	return _stages.size()

func stage_index() -> int:
	return _index

func stage_name() -> String:
	return str(_stages[_index]["name"]) if _index < _stages.size() else ""

func current_params() -> Dictionary:
	return _stages[_index]["params"] if _index < _stages.size() else {}

func is_final() -> bool:
	return _index >= _stages.size() - 1

func record_episode(reward: float, success: bool) -> void:
	var window := _window_size()
	_rewards.append(reward)
	_successes.append(success)
	while _rewards.size() > window:
		_rewards.pop_front()
		_successes.pop_front()

func should_promote() -> bool:
	if is_final() or not _stages[_index].has("promote"):
		return false
	var p: Dictionary = _stages[_index]["promote"]
	if _rewards.size() < int(p["min_episodes"]):
		return false
	match str(p["metric"]):
		"mean_reward":
			return _mean(_rewards) >= float(p["threshold"])
		"success_rate":
			return _success_rate() >= float(p["threshold"])
	return false

func advance() -> bool:
	if is_final():
		return false
	_index += 1
	_clear_window()
	return true

func set_stage(i: int) -> bool:
	if i < 0 or i >= _stages.size():
		push_warning("Curriculum: set_stage(%d) out of range [0, %d)." % [i, _stages.size()])
		return false
	_index = i
	_clear_window()
	return true

func _window_size() -> int:
	if _index < _stages.size() and _stages[_index].has("promote"):
		return int(_stages[_index]["promote"]["window"])
	return 1

func _clear_window() -> void:
	_rewards.clear()
	_successes.clear()

func _mean(xs: Array) -> float:
	if xs.is_empty():
		return 0.0
	var s := 0.0
	for x in xs:
		s += float(x)
	return s / xs.size()

func _success_rate() -> float:
	if _successes.is_empty():
		return 0.0
	var n := 0
	for v in _successes:
		if v:
			n += 1
	return float(n) / _successes.size()
```

- [ ] **Step 4:** run the test → `Results: N passed, 0 failed`.
- [ ] **Step 5:** `git add addons/godot_native_rl/training/curriculum.gd test/unit/test_curriculum.gd && git commit -m "feat: pure curriculum stage/promotion logic (#28)"`

---

### Task 2: CurriculumController node

**Files:** Create `addons/godot_native_rl/training/curriculum_controller.gd` · Test `test/unit/test_curriculum_controller.gd`

- [ ] **Step 1: failing test** — `test/unit/test_curriculum_controller.gd`:

```gdscript
extends SceneTree
# CurriculumController: applies stage params to the game at episode boundaries, emits
# stage_changed, honors external control, loads stages from JSON, errors loud on a missing
# apply method. No NcnnSync/socket involved.

const Harness = preload("res://test/harness.gd")
const Controller = preload("res://addons/godot_native_rl/training/curriculum_controller.gd")

class StubGame:
	extends Node
	var applied: Array = []
	func apply_curriculum(params: Dictionary) -> void:
		applied.append(params)

var _signal_log: Array = []

func _on_stage_changed(index: int, name: String, params: Dictionary) -> void:
	_signal_log.append([index, name, params])

func _stages() -> Array:
	return [
		{"name": "easy", "params": {"touch_radius": 120.0},
			"promote": {"metric": "mean_reward", "threshold": 5.0, "window": 2, "min_episodes": 2}},
		{"name": "hard", "params": {"touch_radius": 40.0}},
	]

func _initialize() -> void:
	var h = Harness.new()

	var game := StubGame.new()
	get_root().add_child(game)
	var ctrl = Controller.new()
	get_root().add_child(ctrl)
	ctrl.game_path = ctrl.get_path_to(game)
	ctrl.set_stages(_stages())
	ctrl.stage_changed.connect(_on_stage_changed)
	# _ready ran on add_child (root tree is active in SceneTree scripts only via manual call):
	ctrl._ready()  # idempotent re-run to ensure group registration with paths set

	h.assert_true(ctrl.is_in_group("CURRICULUM"), "joins CURRICULUM group")
	# Stage 0 params applied once initially:
	h.assert_eq(game.applied.size(), 1, "initial stage params applied")
	h.assert_eq(game.applied[0]["touch_radius"], 120.0, "initial params are stage 0")

	# Two good episodes -> promotion applies stage 1 params at the SAME record boundary
	ctrl.record_episode(10.0, true)
	h.assert_eq(game.applied.size(), 1, "no promotion after 1 episode")
	ctrl.record_episode(10.0, true)
	h.assert_eq(game.applied.size(), 2, "promotion applied at episode boundary")
	h.assert_eq(game.applied[1]["touch_radius"], 40.0, "stage 1 params applied")
	h.assert_eq(_signal_log.size(), 1, "stage_changed emitted once")
	h.assert_eq(_signal_log[0][0], 1, "signal carries new index")
	h.assert_eq(ctrl.stage_index(), 1, "controller reports stage 1")

	# External control disables auto-promotion and supports direct jumps
	var ctrl2 = Controller.new()
	get_root().add_child(ctrl2)
	ctrl2.game_path = ctrl2.get_path_to(game)
	ctrl2.set_stages(_stages())
	ctrl2._ready()
	game.applied.clear()
	ctrl2.set_external_control(true)
	ctrl2.record_episode(100.0, true)
	ctrl2.record_episode(100.0, true)
	h.assert_eq(ctrl2.stage_index(), 0, "external control blocks auto-promotion")
	h.assert_true(ctrl2.jump_to_stage(1), "external jump works")
	h.assert_eq(game.applied.back()["touch_radius"], 40.0, "jump applied params")
	h.assert_true(not ctrl2.jump_to_stage(9), "out-of-range jump refused")

	# Direct params injection (trainer 'params' override)
	ctrl2.apply_external_params({"touch_radius": 7.0})
	h.assert_eq(game.applied.back()["touch_radius"], 7.0, "external params applied")

	# JSON loading
	var ctrl3 = Controller.new()
	get_root().add_child(ctrl3)
	ctrl3.game_path = ctrl3.get_path_to(game)
	ctrl3.stages_json_path = "res://test/unit/fixtures/curriculum_two_stage.json"
	ctrl3._ready()
	h.assert_eq(ctrl3.stage_count(), 2, "stages loaded from JSON")

	# Missing apply method: loud but not crashing
	var bare := Node.new()
	get_root().add_child(bare)
	var ctrl4 = Controller.new()
	get_root().add_child(ctrl4)
	ctrl4.game_path = ctrl4.get_path_to(bare)
	ctrl4.set_stages(_stages())
	ctrl4._ready()  # push_error expected; must not crash
	h.assert_true(true, "missing apply method did not crash")

	h.finish(self)
```

Also create fixture `test/unit/fixtures/curriculum_two_stage.json`:

```json
{
	"stages": [
		{"name": "easy", "params": {"touch_radius": 120.0},
		 "promote": {"metric": "mean_reward", "threshold": 5.0, "window": 2, "min_episodes": 2}},
		{"name": "hard", "params": {"touch_radius": 40.0}}
	]
}
```

- [ ] **Step 2:** run → FAIL (missing controller).
- [ ] **Step 3: implement** `addons/godot_native_rl/training/curriculum_controller.gd`:

```gdscript
extends Node
# Thin Node wrapper around the pure Curriculum: applies stage params to the game node at
# episode boundaries (never mid-episode), reports promotions via stage_changed + print, and
# supports trainer override (external control) via NcnnSync's "curriculum" wire message.
# Spec: docs/superpowers/specs/2026-06-12-curriculum-learning-design.md (#28)

const CurriculumScript = preload("res://addons/godot_native_rl/training/curriculum.gd")

signal stage_changed(index: int, name: String, params: Dictionary)

@export var game_path: NodePath
@export var apply_method := "apply_curriculum"
@export var stages_json_path := ""  ## optional JSON {"stages": [...]}; set_stages() takes precedence

var _curriculum = CurriculumScript.new()
var _game: Node
var _external := false
var _stages_set := false
var _initial_applied := false

func _ready() -> void:
	if not is_in_group("CURRICULUM"):
		add_to_group("CURRICULUM")
	if _game == null:
		_game = get_node_or_null(game_path)
	if not _stages_set and stages_json_path != "":
		var loaded := _load_stages_json(stages_json_path)
		if not loaded.is_empty():
			set_stages(loaded)
	if _stages_set and not _initial_applied:
		_apply(_curriculum.current_params())
		_initial_applied = true

func set_stages(stages: Array) -> bool:
	_stages_set = _curriculum.set_stages(stages)
	return _stages_set

func set_external_control(on: bool) -> void:
	_external = on

func record_episode(reward: float, success: bool) -> void:
	_curriculum.record_episode(reward, success)
	if _external:
		return
	if _curriculum.should_promote() and _curriculum.advance():
		print("Curriculum: promoted to stage %d \"%s\"" % [_curriculum.stage_index(), _curriculum.stage_name()])
		_apply(_curriculum.current_params())
		stage_changed.emit(_curriculum.stage_index(), _curriculum.stage_name(), _curriculum.current_params())

func jump_to_stage(i: int) -> bool:
	if not _curriculum.set_stage(i):
		return false
	print("Curriculum: externally set to stage %d \"%s\"" % [i, _curriculum.stage_name()])
	_apply(_curriculum.current_params())
	stage_changed.emit(_curriculum.stage_index(), _curriculum.stage_name(), _curriculum.current_params())
	return true

func apply_external_params(params: Dictionary) -> void:
	set_external_control(true)
	_apply(params)

func stage_index() -> int:
	return _curriculum.stage_index()

func stage_name() -> String:
	return _curriculum.stage_name()

func stage_count() -> int:
	return _curriculum.stage_count()

func _apply(params: Dictionary) -> void:
	if _game == null:
		_game = get_node_or_null(game_path)
	if _game == null or not _game.has_method(apply_method):
		push_error("CurriculumController: game at '%s' has no method '%s' — params not applied." % [str(game_path), apply_method])
		return
	_game.call(apply_method, params)

func _load_stages_json(path: String) -> Array:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("CurriculumController: cannot open stages JSON '%s'." % path)
		return []
	var parsed = JSON.parse_string(f.get_as_text())
	if not (parsed is Dictionary) or not (parsed.get("stages") is Array):
		push_error("CurriculumController: '%s' must be {\"stages\": [...]}." % path)
		return []
	return parsed["stages"]
```

- [ ] **Step 4:** run → PASS. (Note the `_ready()` manual re-run in the test: headless SceneTree scripts don't fire `_ready` reliably for nodes configured after add_child — see memory note; the controller's `_ready` is written idempotent for exactly this.)
- [ ] **Step 5:** `git add addons/godot_native_rl/training/curriculum_controller.gd test/unit/test_curriculum_controller.gd test/unit/fixtures/curriculum_two_stage.json && git commit -m "feat: CurriculumController node — boundary application, signal, external control (#28)"`

---

### Task 3: NcnnSync "curriculum" wire message

**Files:** Modify `addons/godot_native_rl/sync.gd` (handle_message match) · Test: extend `test/integration/run_protocol_test.py`

- [ ] **Step 1:** read `test/integration/run_protocol_test.py` + its stub scene to mirror the framing; add a test segment that, after the handshake, sends `{"type":"curriculum","stage":1}` and asserts (via a `call` message round-trip to a stub method `get_curriculum_stage`) the stage changed; then sends `{"type":"curriculum","params":{"touch_radius":7}}` and asserts applied. (The stub scene gains a `CurriculumController` + stub game exposing `get_curriculum_stage()` through the agent's `_call_method_on_agents` path or a dedicated probe — choose whichever the existing protocol test supports with the least new machinery.)
- [ ] **Step 2:** run the protocol test → FAIL (unhandled message type warning, no stage change).
- [ ] **Step 3: implement** — in `sync.gd`'s `handle_message()` match, before the fallthrough:

```gdscript
		"curriculum":
			_handle_curriculum_message(message)
			return handle_message()
```

and the handler + helper:

```gdscript
# Trainer-driven curriculum override (#28): {"type":"curriculum","stage":N} jumps the scene's
# CurriculumController to stage N; {"type":"curriculum","params":{...}} applies raw params.
# Additive + optional: stock trainers never send it; absent controller -> warn and drop.
func _handle_curriculum_message(message: Dictionary) -> void:
	var ctrl := get_tree().get_first_node_in_group("CURRICULUM")
	if ctrl == null:
		push_warning("NcnnSync: 'curriculum' message but no CurriculumController in scene; ignored.")
		return
	ctrl.set_external_control(true)
	if message.has("stage"):
		ctrl.jump_to_stage(int(message["stage"]))
	elif message.has("params") and message["params"] is Dictionary:
		ctrl.apply_external_params(message["params"])
	else:
		push_warning("NcnnSync: 'curriculum' message needs 'stage' or 'params'.")
```

- [ ] **Step 4:** run the protocol test → PASS; also run `test/unit/test_sync_messages.gd` to confirm no regression.
- [ ] **Step 5:** `git add addons/godot_native_rl/sync.gd test/integration/run_protocol_test.py <stub files> && git commit -m "feat: additive 'curriculum' wire message for trainer-driven stage override (#28)"`

---

### Task 4: Python curriculum client

**Files:** Create `scripts/curriculum_client.py` · Test `test/python/test_curriculum_client.py`

- [ ] **Step 1: failing test** — `test/python/test_curriculum_client.py`:

```python
import json
import struct
import unittest

from scripts.curriculum_client import encode_curriculum_stage, encode_curriculum_params


class TestCurriculumClient(unittest.TestCase):
    def _decode(self, payload: bytes):
        # godot_rl framing: 4-byte little-endian length prefix + utf-8 JSON
        (length,) = struct.unpack("<I", payload[:4])
        self.assertEqual(length, len(payload) - 4)
        return json.loads(payload[4:].decode("utf-8"))

    def test_stage_message(self):
        msg = self._decode(encode_curriculum_stage(2))
        self.assertEqual(msg, {"type": "curriculum", "stage": 2})

    def test_params_message(self):
        msg = self._decode(encode_curriculum_params({"touch_radius": 7.0}))
        self.assertEqual(msg["type"], "curriculum")
        self.assertEqual(msg["params"], {"touch_radius": 7.0})

    def test_stage_must_be_int(self):
        with self.assertRaises(TypeError):
            encode_curriculum_stage("two")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2:** run `.venv-train/bin/python -m unittest test.python.test_curriculum_client` → FAIL (no module).
- [ ] **Step 3: implement** `scripts/curriculum_client.py` (verify the exact framing against `run_protocol_test.py` first — if the wire uses a different prefix (e.g. ascii length), match THAT and fix the unit test accordingly; the framing constant must equal what `NcnnSync._get_dict_json_message` expects):

```python
#!/usr/bin/env python3
"""Send curriculum-override messages to a running NcnnSync (trainer-driven curriculum, #28).

Game-side promotion is the default and needs nothing from the trainer. These helpers are for
custom training loops (e.g. the single-file CleanRL-style trainers) that want explicit control:

    sock.sendall(encode_curriculum_stage(2))          # jump the scene to stage 2
    sock.sendall(encode_curriculum_params({"x": 1}))  # inject raw env params

Framing matches the godot_rl JSON wire NcnnSync speaks (length-prefixed UTF-8 JSON).
Stdlib only; pure encoders so they unit-test without a socket.
"""
import json
import struct


def _encode(message: dict) -> bytes:
    payload = json.dumps(message).encode("utf-8")
    return struct.pack("<I", len(payload)) + payload


def encode_curriculum_stage(stage: int) -> bytes:
    if not isinstance(stage, int) or isinstance(stage, bool):
        raise TypeError("stage must be an int")
    return _encode({"type": "curriculum", "stage": stage})


def encode_curriculum_params(params: dict) -> bytes:
    return _encode({"type": "curriculum", "params": dict(params)})
```

- [ ] **Step 4:** run → PASS (auto-discovered by run_tests.sh's unittest discover).
- [ ] **Step 5:** `git add scripts/curriculum_client.py test/python/test_curriculum_client.py && git commit -m "feat: stdlib curriculum-override client for custom trainers (#28)"`

---

### Task 5: Chase demo wiring + headless smoke

**Files:** Create `examples/chase_the_target/chase_curriculum.json`, `chase_the_target_train_curriculum.tscn`; Modify `chase_game.gd`, `chase_agent.gd`; Test `test/integration/curriculum_smoke_checker.gd` + `curriculum_smoke_scene.tscn`

- [ ] **Step 1:** `chase_curriculum.json` (3 stages; tune thresholds during the trained run):

```json
{
	"stages": [
		{"name": "easy", "params": {"touch_radius": 120.0, "arena_size_x": 500.0, "arena_size_y": 300.0},
		 "promote": {"metric": "mean_reward", "threshold": 8.0, "window": 20, "min_episodes": 20}},
		{"name": "mid", "params": {"touch_radius": 80.0, "arena_size_x": 750.0, "arena_size_y": 450.0},
		 "promote": {"metric": "mean_reward", "threshold": 8.0, "window": 20, "min_episodes": 20}},
		{"name": "hard", "params": {"touch_radius": 40.0, "arena_size_x": 1000.0, "arena_size_y": 600.0}}
	]
}
```

- [ ] **Step 2:** `ChaseGame.apply_curriculum` (params flat floats — Vector2 isn't JSON):

```gdscript
# Curriculum hook (#28): stage params applied at episode boundaries by CurriculumController.
func apply_curriculum(params: Dictionary) -> void:
	if params.has("touch_radius"):
		touch_radius = float(params["touch_radius"])
	if params.has("arena_size_x"):
		arena_size.x = float(params["arena_size_x"])
	if params.has("arena_size_y"):
		arena_size.y = float(params["arena_size_y"])
```

- [ ] **Step 3:** `ChaseAgent` — in its `needs_reset` branch (read the exact current code first; insert before `reset()`):

```gdscript
		# Curriculum (#28): report the finished episode (reward + did we catch at least once).
		if _curriculum != null:
			_curriculum.record_episode(reward, _episode_catches > 0)
		_episode_catches = 0
```

plus: `var _curriculum: Node` resolved in `_ready` via `get_tree().get_first_node_in_group("CURRICULUM")` (null-safe — the plain train scene has none), `_episode_catches` incremented in the existing `target_caught` adapter path (connect the game signal directly: `_game.target_caught.connect(func(): _episode_catches += 1)`), and:

```gdscript
func get_info() -> Dictionary:
	if _curriculum == null:
		return {}
	return {"curriculum_stage": _curriculum.stage_index()}
```

- [ ] **Step 4:** `chase_the_target_train_curriculum.tscn` — copy of `chase_the_target_train.tscn` + a `CurriculumController` node (`game_path` → ChaseGame, `stages_json_path` → the JSON). Verify headless load with `--quit-after 3`.
- [ ] **Step 5: smoke** — `test/integration/curriculum_smoke_checker.gd` in a scene instancing the real chase game + agent + controller (no socket): drive fake episodes by calling `controller.record_episode(10.0, true)` 20× and assert `game.touch_radius` moved 120 → 80 (stage promotion applied to the real game), then 20× more → 40 (final). Assert `stage_changed` fired twice and `agent.get_info()["curriculum_stage"] == 2`. Quit 0/1 like the other checkers. Register in `run_tests.sh` after the quadruped smoke:

```bash
echo "== Curriculum promotion smoke (headless) =="
"$GODOT" --headless --path . res://test/integration/curriculum_smoke_scene.tscn
```

- [ ] **Step 6:** run the new smoke + the full suite → all green.
- [ ] **Step 7:** `git add examples/chase_the_target test/integration/curriculum_smoke_* test/run_tests.sh && git commit -m "feat: 3-stage chase curriculum demo + headless promotion smoke (#28)"`

---

### Task 6: Real trained demonstration

- [ ] **Step 1:** `SCENE=res://examples/chase_the_target/chase_the_target_train_curriculum.tscn TIMESTEPS=120000 GODOT=/opt/homebrew/bin/godot-mono ./scripts/train_chase.sh` in the background (`caffeinate -is`), log to `logs/chase_curriculum.log`.
- [ ] **Step 2:** monitor for `Curriculum: promoted to stage` lines; verify promotions happen mid-run (not instantly — else raise thresholds in the JSON and rerun) and reward climbs across stages.
- [ ] **Step 3:** capture the promotion timeline (step counts at each promotion + reward curve summary) for the PR body. Do NOT commit the resulting model (deploy contract unchanged; existing chase fixtures stay).

### Task 7: Docs + PR

- [ ] README: curriculum feature bullet (game-side, all backends, optional trainer override) + demo command.
- [ ] CLAUDE.md: key-command entry (curriculum train variant + JSON knobs) and example-list mention.
- [ ] `docs/godot-rl-gap-analysis-2026-06-02.md`: curriculum row (Unity ML-Agents parity) → done.
- [ ] `docs/BACKLOG.md`: tick item 52.
- [ ] Full suite green; push `feature/curriculum-learning`; `gh pr create` (body via repo-local `--body-file`, `Closes #28`), include the trained-run promotion evidence.

---

## Self-review

- **Spec coverage:** pure logic (T1), controller (T2), wire message (T3), Python client (T4), chase demo + info reporting + smoke (T5), trained run (T6), docs (T7). Non-goals untouched. ✔
- **Placeholders:** none — full code for every new file; T3/T5 include explicit "read the existing file first" steps where insertion points depend on current line content (executor verifies framing/branch shape before editing). ✔
- **Type consistency:** `set_stages`, `record_episode(reward, success)`, `should_promote`, `advance`, `set_stage` / controller `jump_to_stage`, `apply_external_params`, `set_external_control`, `stage_index/name/count` used consistently across tasks and tests. ✔
