# Expert-Demo Recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record human/scripted-expert `(observation, action)` demonstrations from a Godot example to disk in our native `gnrl_v1` format (or legacy godot_rl format), then behavior-clone a policy that exports through the existing ncnn pipeline.

**Architecture:** A pure `DemoRecorder` accumulator (godot_rl trajectory layout) + thin `NcnnSync` `RECORD_EXPERT_DEMOS` wiring + a `get_action()` agent hook. A version-aware Python loader feeds a minimal torch behavior-cloning trainer that emits a TorchScript model consumable by `export_to_ncnn.py`.

**Tech Stack:** GDScript (Godot 4.6, TAB indent, path-based `extends`), Python 3.13 in `.venv-train` (torch via SB3), the project's dependency-free headless test harness (`test/harness.gd`).

**Spec:** `docs/superpowers/specs/2026-06-04-expert-demo-recording-design.md`

---

## File Structure

New:
- `addons/godot_native_rl/training/demo_recorder.gd` — pure RefCounted accumulator + JSON serializer.
- `examples/chase_the_target/chase_expert_agent.gd` — scripted expert (`get_action()` steers toward target).
- `examples/chase_the_target/record_chase_demos.tscn` — record scene (Sync in RECORD mode + expert + checker).
- `test/integration/record_demos_smoke_checker.gd` — drives + verifies the record scene, self-quits.
- `examples/chase_the_target/demos/chase_expert_demos.json` — committed `gnrl_v1` sample (generated from the scene).
- `scripts/load_expert_demos.py` — version-aware loader (`load_demos`, `flatten_pairs`).
- `scripts/train_bc.py` — minimal torch behavior cloning → TorchScript + shape sidecar.
- `test/unit/test_demo_recorder.gd`, `test/unit/test_chase_expert_action.gd`
- `test/python/test_demo_loader.py`, `test/python/test_train_bc.py`

Modified:
- `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` / `_3d.gd` — add `get_action()` hook.
- `addons/godot_native_rl/sync.gd` — enum value, exports, record init + process, save.
- `test/run_tests.sh` — wire the record-demos smoke scene.
- Docs: `README.md`, `CLAUDE.md`, `docs/BACKLOG.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`.

---

## Task 1: `DemoRecorder` pure accumulator

**Files:**
- Create: `addons/godot_native_rl/training/demo_recorder.gd`
- Test: `test/unit/test_demo_recorder.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_demo_recorder.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const DemoRecorder = preload("res://addons/godot_native_rl/training/demo_recorder.gd")

func _initialize() -> void:
	var h := Harness.new()
	var r = DemoRecorder.new()

	# Two non-terminal steps, then a terminal step => one trajectory, 3 obs / 2 acts.
	r.record_step([0.0, 1.0], [1.0], false)
	r.record_step([0.1, 1.1], [2.0], false)
	r.record_step([0.2, 1.2], [9.0], true)  # terminal: obs kept, action dropped
	h.assert_eq(r.trajectory_count(), 1, "terminal step finalizes one trajectory")
	h.assert_eq(r.step_count(), 2, "two actions recorded (terminal action dropped)")

	# gnrl_v1 envelope.
	var parsed = JSON.parse_string(
		r.to_json("gnrl_v1", {"move": {"size": 5, "action_type": "discrete"}}))
	h.assert_eq(parsed["format_version"], "gnrl_v1", "envelope carries format_version")
	h.assert_true(parsed.has("action_space"), "envelope carries action_space")
	var traj = parsed["demo_trajectories"][0]
	h.assert_eq(traj[0].size(), 3, "obs list keeps terminal frame (len acts + 1)")
	h.assert_eq(traj[1].size(), 2, "acts list excludes terminal action")

	# Legacy godot_rl format is the bare trajectory array.
	var bare = JSON.parse_string(r.to_json("godot_rl", {}))
	h.assert_true(bare is Array, "godot_rl format is a bare top-level array")
	h.assert_eq(bare.size(), 1, "one trajectory in bare array")

	# Input arrays must not be aliased: mutating the caller's array after record_step
	# must not change recorded data.
	var obs_in := [5.0]
	var act_in := [6.0]
	r.record_step(obs_in, act_in, false)
	obs_in[0] = 999.0
	act_in[0] = 999.0
	r.record_step([7.0], [8.0], true)
	var t2 = JSON.parse_string(r.to_json("godot_rl", {}))[1]
	h.assert_eq(t2[1][0][0], 6.0, "recorded action is a copy, not aliased to caller")

	# remove_last_episode pops; guarded on empty.
	r.remove_last_episode()
	r.remove_last_episode()
	h.assert_eq(r.trajectory_count(), 0, "remove_last_episode pops both, then guards empty")
	r.remove_last_episode()  # must not crash when empty

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_demo_recorder.gd`
Expected: FAIL — cannot load `demo_recorder.gd` (file does not exist).

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/training/demo_recorder.gd`:

```gdscript
extends RefCounted
# Pure accumulator for expert demonstrations in godot_rl trajectory layout.
# Each trajectory is [obs_list, acts_list] with len(obs_list) == len(acts_list) + 1
# (the terminal observation has no action), mirroring godot_rl_agents' recorder.
# No file I/O here — serialization returns a String; NcnnSync writes it.

const FORMAT_GNRL_V1 := "gnrl_v1"
const FORMAT_GODOT_RL := "godot_rl"

var _trajectories: Array = []      # Array of [obs_list, acts_list]
var _current: Array = [[], []]     # in-progress [obs_list, acts_list]

func record_step(obs: Array, action: Array, done: bool) -> void:
	_current[0].append(obs.duplicate())  # copy so callers can't mutate recorded data
	if done:
		_trajectories.append(_current.duplicate(true))
		_current[0] = []
		_current[1] = []
	else:
		_current[1].append(action.duplicate())

func remove_last_episode() -> void:
	if _trajectories.size() > 0:
		_trajectories.remove_at(_trajectories.size() - 1)

func trajectory_count() -> int:
	return _trajectories.size()

# Total recorded actions (transitions), completed + in-progress.
func step_count() -> int:
	var n := _current[1].size()
	for traj in _trajectories:
		n += traj[1].size()
	return n

func to_json(demo_format: String, action_space: Dictionary) -> String:
	if demo_format == FORMAT_GODOT_RL:
		return JSON.stringify(_trajectories, "", false)
	assert(demo_format == FORMAT_GNRL_V1,
		"DemoRecorder: unknown demo_format '%s'" % demo_format)
	var envelope := {
		"format_version": FORMAT_GNRL_V1,
		"action_space": action_space,
		"demo_trajectories": _trajectories,
	}
	return JSON.stringify(envelope, "", false)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_demo_recorder.gd`
Expected: PASS — last line `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/training/demo_recorder.gd test/unit/test_demo_recorder.gd
git commit -m "feat(demos): pure DemoRecorder accumulator + gnrl_v1/godot_rl serialization (#13)"
```

---

## Task 2: `get_action()` hook on the controllers

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` (after `set_action`, ~line 142)
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd` (same spot)
- Test: `test/unit/test_chase_expert_action.gd` covers the override path in Task 4; this task just adds the abstract hook (the default-asserts contract mirrors the existing `get_obs`/`set_action` asserts, which are likewise contract-only).

- [ ] **Step 1: Add the hook to the 2D controller**

In `ncnn_ai_controller_2d.gd`, immediately after the existing `set_action(_action)` method, add:

```gdscript
# Return the flat action array for expert-demo recording (godot_rl get_action() parity).
# A scripted-expert or human-controlled agent overrides this: it decides the action,
# applies it (so the avatar moves via the agent's own _physics_process), and returns the
# flat array recorded into the demo file. Default asserts — only required when recording.
func get_action() -> Array:
	assert(false, "get_action must be implemented by the agent to record expert demos")
	return []
```

- [ ] **Step 2: Add the identical hook to the 3D controller**

In `ncnn_ai_controller_3d.gd`, after its `set_action(_action)` method, add the same method verbatim (the comment and body are identical — `NcnnAIController3D` agents record the same way).

- [ ] **Step 3: Verify the controllers still parse**

Run: `godot --headless --path . --script res://test/unit/test_controller.gd`
Expected: PASS — adding an unused method must not break existing controller tests.

Run: `godot --headless --path . --script res://test/unit/test_controller_3d.gd`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd
git commit -m "feat(demos): add get_action() recording hook to NcnnAIController2D/3D (#13)"
```

---

## Task 3: `NcnnSync` RECORD_EXPERT_DEMOS mode

**Files:**
- Modify: `addons/godot_native_rl/sync.gd`

This is thin orchestration glue; it needs a live `SceneTree`, so its behavioral test is the integration smoke (Task 5). Make the edits, then verify existing sync tests still pass.

- [ ] **Step 1: Extend the control-mode enum**

In `sync.gd` line 4, change:

```gdscript
enum ControlModes { HUMAN, TRAINING, NCNN_INFERENCE }
```

to:

```gdscript
enum ControlModes { HUMAN, TRAINING, NCNN_INFERENCE, RECORD_EXPERT_DEMOS }
```

- [ ] **Step 2: Add the DemoRecorder preload + exports + state**

Near the other preloads at the top (after the `StepProfiler` preload, ~line 7), add:

```gdscript
const DemoRecorder = preload("res://addons/godot_native_rl/training/demo_recorder.gd")
```

After the existing socket-timeout exports (~line 56), add:

```gdscript
# --- Expert-demo recording (control_mode == RECORD_EXPERT_DEMOS) ---
@export_global_file("*.json") var expert_demo_save_path: String = ""
@export var demo_format: String = "gnrl_v1"  # "gnrl_v1" (default) | "godot_rl" (legacy/interop)
# InputMap action that pops the last recorded episode (undo a bad demo). Acted on only if mapped.
@export var remove_last_episode_action: StringName = &"remove_last_demo_episode"
# Headless bound: after this many recorded actions, save + quit. 0 = unlimited (editor/human play).
@export var max_record_steps: int = 0
```

With the other `var` state declarations (~line 70), add:

```gdscript
var _recorder = null
var _record_agent = null
var _record_action_space: Dictionary = {}
```

- [ ] **Step 3: Branch `_initialize()` into the recording path**

In `_initialize()` (~line 86), replace the single `_initialize_training_agents()` call with a branch:

```gdscript
	_set_heuristic("human", all_agents)
	if control_mode == ControlModes.RECORD_EXPERT_DEMOS:
		_initialize_demo_recording()
	else:
		_initialize_training_agents()
	_set_seed()
```

- [ ] **Step 4: Add the recording init + per-step + save methods**

After `_initialize_training_agents()` (~line 109), add:

```gdscript
func _initialize_demo_recording() -> void:
	# godot_rl parity: a single agent is recorded. RECORD mode is OFFLINE — it never opens
	# the TCP socket, so the "training scene without a trainer hangs" gotcha does not apply.
	assert(all_agents.size() == 1,
		"RECORD_EXPERT_DEMOS records a single agent (got %d)" % all_agents.size())
	_record_agent = all_agents[0]
	# The agent was routed to agents_heuristic by _get_agents(); take it back so
	# _heuristic_process() doesn't also reset it — the agent's own _physics_process does.
	agents_heuristic.erase(_record_agent)
	_record_action_space = _record_agent.get_action_space()
	_recorder = DemoRecorder.new()

func _demo_record_process() -> void:
	if _recorder == null:
		return
	var obs: Array = _record_agent.get_obs()["obs"]
	var action: Array = _record_agent.get_action()
	var done: bool = _record_agent.get_done()
	_recorder.record_step(obs, action, done)
	if done:
		_record_agent.set_done_false()
	if _remove_last_episode_pressed():
		_recorder.remove_last_episode()
	if max_record_steps > 0 and _recorder.step_count() >= max_record_steps:
		save_expert_demos()
		get_tree().quit(0)

func _remove_last_episode_pressed() -> bool:
	if String(remove_last_episode_action).is_empty():
		return false
	if not InputMap.has_action(remove_last_episode_action):
		return false
	return Input.is_action_just_pressed(remove_last_episode_action)

func save_expert_demos() -> void:
	if _recorder == null:
		return
	if expert_demo_save_path.is_empty():
		push_error("NcnnSync: expert_demo_save_path is empty; cannot save demos.")
		return
	var abs_path := ProjectSettings.globalize_path(expert_demo_save_path)
	DirAccess.make_dir_recursive_absolute(abs_path.get_base_dir())
	var f := FileAccess.open(abs_path, FileAccess.WRITE)
	if f == null:
		push_error("NcnnSync: cannot open expert_demo_save_path '%s'." % expert_demo_save_path)
		return
	f.store_line(_recorder.to_json(demo_format, _record_action_space))
	f.close()

func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		if _recorder != null and not expert_demo_save_path.is_empty():
			save_expert_demos()
```

- [ ] **Step 5: Call `_demo_record_process()` each action-step**

In `_physics_process()` (~line 117), add the record call after `_heuristic_process()`:

```gdscript
	_training_process()
	_inference_process()
	_heuristic_process()
	_demo_record_process()
```

- [ ] **Step 6: Verify existing sync tests still pass**

Run: `godot --headless --path . --script res://test/unit/test_sync_messages.gd`
Expected: PASS.

Run: `godot --headless --path . --script res://test/unit/test_sync_inference.gd`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add addons/godot_native_rl/sync.gd
git commit -m "feat(demos): NcnnSync RECORD_EXPERT_DEMOS mode (offline record + save) (#13)"
```

---

## Task 4: Chase scripted-expert agent

**Files:**
- Create: `examples/chase_the_target/chase_expert_agent.gd`
- Test: `test/unit/test_chase_expert_action.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_chase_expert_action.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const ChaseExpertAgent = preload("res://examples/chase_the_target/chase_expert_agent.gd")

func _initialize() -> void:
	var h := Harness.new()

	# expert_action_index maps the relative offset (target - agent) to a chase direction.
	# Indices match ChaseAgent.action_index_to_velocity: 1=up,2=down,3=left,4=right.
	h.assert_eq(ChaseExpertAgent.expert_action_index(Vector2(10.0, 1.0)), 4, "target right -> move right")
	h.assert_eq(ChaseExpertAgent.expert_action_index(Vector2(-10.0, 1.0)), 3, "target left -> move left")
	h.assert_eq(ChaseExpertAgent.expert_action_index(Vector2(1.0, 10.0)), 2, "target below -> move down")
	h.assert_eq(ChaseExpertAgent.expert_action_index(Vector2(1.0, -10.0)), 1, "target above -> move up")
	# Ties on |x| >= |y| pick the horizontal axis.
	h.assert_eq(ChaseExpertAgent.expert_action_index(Vector2(5.0, 5.0)), 4, "tie favors horizontal")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_chase_expert_action.gd`
Expected: FAIL — cannot load `chase_expert_agent.gd`.

- [ ] **Step 3: Write minimal implementation**

Create `examples/chase_the_target/chase_expert_agent.gd`:

```gdscript
# Scripted expert for expert-demo recording. Greedily steps toward the target, so demos
# can be generated headlessly with no human input. Path-based extends (no bare class_name)
# so it resolves in headless/CLI runs — see CLAUDE.md.
extends "res://examples/chase_the_target/chase_agent.gd"

# Pure: map the target-relative offset to a discrete chase direction index.
# Matches ChaseAgent.action_index_to_velocity (1=up, 2=down, 3=left, 4=right).
static func expert_action_index(rel: Vector2) -> int:
	if absf(rel.x) >= absf(rel.y):
		return 4 if rel.x > 0.0 else 3
	return 2 if rel.y > 0.0 else 1

# godot_rl get_action() contract: decide, apply (store _action_index so the base
# _physics_process moves the avatar), and return the flat action array for recording.
func get_action() -> Array:
	if _game == null:
		return [0.0]
	var rel: Vector2 = _game.get_target_pos() - _game.get_agent_pos()
	var idx := expert_action_index(rel)
	_action_index = idx
	return [float(idx)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_chase_expert_action.gd`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/chase_the_target/chase_expert_agent.gd test/unit/test_chase_expert_action.gd
git commit -m "feat(demos): chase scripted-expert agent (get_action steers to target) (#13)"
```

---

## Task 5: Record scene + integration smoke + committed sample

**Files:**
- Create: `test/integration/record_demos_smoke_checker.gd`
- Create: `examples/chase_the_target/record_chase_demos.tscn`
- Create: `examples/chase_the_target/demos/chase_expert_demos.json` (generated, committed)
- Modify: `test/run_tests.sh`

- [ ] **Step 1: Write the smoke checker (the integration test gate)**

Create `test/integration/record_demos_smoke_checker.gd`:

```gdscript
extends Node
# Drives the RECORD_EXPERT_DEMOS scene headlessly: waits until the recorder has enough
# completed trajectories, saves, re-loads the file, asserts gnrl_v1 shape, and quits.
# Save path / trajectory count are overridable via user cmdline args so the same scene
# both runs the suite smoke (user://) and generates the committed sample (res://).

@export var sync_path: NodePath
@export var save_path: String = "user://chase_demos_smoke.json"
@export var target_trajectories: int = 2

var _sync = null
var _done := false

func _ready() -> void:
	_sync = get_node(sync_path)
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--demo-out="):
			save_path = arg.substr("--demo-out=".length())
		elif arg.begins_with("--demo-trajectories="):
			target_trajectories = arg.substr("--demo-trajectories=".length()).to_int()
	_sync.expert_demo_save_path = save_path

func _physics_process(_delta) -> void:
	if _done or _sync == null or _sync._recorder == null:
		return
	if _sync._recorder.trajectory_count() < target_trajectories:
		return
	_done = true
	_sync.save_expert_demos()
	_verify_and_quit()

func _verify_and_quit() -> void:
	var text := FileAccess.get_file_as_string(ProjectSettings.globalize_path(save_path))
	var parsed = JSON.parse_string(text)
	var ok: bool = parsed is Dictionary \
		and parsed.get("format_version") == "gnrl_v1" \
		and parsed.get("demo_trajectories", []).size() >= target_trajectories
	if ok:
		var traj = parsed["demo_trajectories"][0]
		ok = traj[0].size() == traj[1].size() + 1  # obs keeps the terminal frame
	if ok:
		print("record-demos smoke: PASS (%d trajectories)" % parsed["demo_trajectories"].size())
		get_tree().quit(0)
	else:
		printerr("record-demos smoke: FAIL")
		get_tree().quit(1)
```

- [ ] **Step 2: Write the record scene**

Create `examples/chase_the_target/record_chase_demos.tscn`. Mirror the structure of `test/integration/inference_smoke_scene.tscn` (ChaseGame root with AgentBody + Target children), but use the expert agent and RECORD mode. `Sync.control_mode = 3` is `RECORD_EXPERT_DEMOS`; the agent's `control_mode = 0` is `INHERIT_FROM_SYNC`. Small `reset_after` makes episodes complete fast; `action_repeat = 1` samples every frame.

```
[gd_scene load_steps=6 format=3]

[ext_resource type="Script" path="res://examples/chase_the_target/chase_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/chase_the_target/chase_expert_agent.gd" id="2"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="3"]
[ext_resource type="Script" path="res://test/integration/record_demos_smoke_checker.gd" id="4"]

[node name="ChaseGame" type="Node2D"]
script = ExtResource("1")
agent_body_path = NodePath("AgentBody")
target_path = NodePath("Target")

[node name="AgentBody" type="Node2D" parent="."]

[node name="Target" type="Node2D" parent="."]

[node name="ChaseExpertAgent" type="Node2D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
control_mode = 0
reset_after = 30

[node name="Sync" type="Node" parent="."]
script = ExtResource("3")
control_mode = 3
action_repeat = 1
demo_format = "gnrl_v1"

[node name="RecordChecker" type="Node" parent="."]
script = ExtResource("4")
sync_path = NodePath("../Sync")
```

Confirm the exact `ChaseGame` exported property names (`agent_body_path`, `target_path`) and node layout against `test/integration/inference_smoke_scene.tscn` before finalizing; copy whatever that scene uses.

- [ ] **Step 3: Run the smoke scene to verify it passes**

Run: `godot --headless --path . res://examples/chase_the_target/record_chase_demos.tscn`
Expected: prints `record-demos smoke: PASS (>=2 trajectories)` and exits 0.

If it hangs: the agent isn't reaching `done` (check `reset_after` is small and the agent's `_physics_process` sets `needs_reset`/`done`), or `_sync._recorder` is null (RECORD init didn't run — check `control_mode = 3`).

- [ ] **Step 4: Generate and commit the sample demo file**

Run (writes a richer sample straight into the repo via the user-arg override):

```bash
mkdir -p examples/chase_the_target/demos
godot --headless --path . res://examples/chase_the_target/record_chase_demos.tscn -- \
  --demo-out=res://examples/chase_the_target/demos/chase_expert_demos.json \
  --demo-trajectories=8
```

Expected: exits 0; `examples/chase_the_target/demos/chase_expert_demos.json` exists, is a single-line `gnrl_v1` envelope with `"demo_trajectories"` of length ≥ 8. Verify:

```bash
python3 -c "import json; d=json.load(open('examples/chase_the_target/demos/chase_expert_demos.json')); print(d['format_version'], len(d['demo_trajectories']))"
```

Expected: `gnrl_v1 8` (or more).

- [ ] **Step 5: Wire the smoke into `run_tests.sh`**

In `test/run_tests.sh`, after the `Parallel arena smoke test` block, add:

```bash
echo "== Expert-demo record smoke test (headless) =="
"$GODOT" --headless --path . res://examples/chase_the_target/record_chase_demos.tscn
```

- [ ] **Step 6: Commit**

```bash
git add test/integration/record_demos_smoke_checker.gd examples/chase_the_target/record_chase_demos.tscn examples/chase_the_target/demos/chase_expert_demos.json test/run_tests.sh
git commit -m "test(demos): record scene + integration smoke + committed gnrl_v1 sample (#13)"
```

---

## Task 6: Python loader `load_expert_demos.py`

**Files:**
- Create: `scripts/load_expert_demos.py`
- Test: `test/python/test_demo_loader.py`

- [ ] **Step 1: Write the failing test**

Create `test/python/test_demo_loader.py`:

```python
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import load_expert_demos as ld  # noqa: E402


def _write(tmp, name, obj):
    p = Path(tmp) / name
    p.write_text(json.dumps(obj))
    return str(p)


# One trajectory: 3 obs (2-dim), 2 acts (1-dim). obs has the terminal frame (acts + 1).
TRAJ = [[[0.0, 1.0], [0.1, 1.1], [0.2, 1.2]], [[1.0], [2.0]]]


class DemoLoaderTest(unittest.TestCase):
    def test_loads_gnrl_v1_with_action_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "d.json", {
                "format_version": "gnrl_v1",
                "action_space": {"move": {"size": 5, "action_type": "discrete"}},
                "demo_trajectories": [TRAJ],
            })
            ds = ld.load_demos(path)
            self.assertEqual(ds.action_space["move"]["size"], 5)
            obs, acts = ds.trajectories[0]
            self.assertEqual(obs.shape, (3, 2))
            self.assertEqual(acts.shape, (2, 1))

    def test_loads_legacy_godot_rl_bare_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "d.json", [TRAJ])
            ds = ld.load_demos(path)
            self.assertIsNone(ds.action_space)
            self.assertEqual(ds.trajectories[0][0].shape, (3, 2))

    def test_flatten_pairs_drops_terminal_obs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "d.json", [TRAJ])
            x, y = ld.flatten_pairs(ld.load_demos(path))
            self.assertEqual(x.shape, (2, 2))  # 2 obs paired with 2 acts
            self.assertEqual(y.shape, (2, 1))

    def test_rejects_unknown_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "d.json", {"format_version": "bogus", "demo_trajectories": []})
            with self.assertRaises(ValueError):
                ld.load_demos(path)

    def test_rejects_length_rule_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 2 obs but 2 acts violates len(obs) == len(acts) + 1.
            bad = [[[[0.0], [0.1]], [[1.0], [2.0]]]]
            path = _write(tmp, "d.json", bad)
            with self.assertRaises(ValueError):
                ld.load_demos(path)

    def test_rejects_bad_top_level_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "d.json", 42)
            with self.assertRaises(ValueError):
                ld.load_demos(path)

    def test_loads_committed_sample(self):
        sample = Path(__file__).resolve().parents[2] / \
            "examples/chase_the_target/demos/chase_expert_demos.json"
        ds = ld.load_demos(str(sample))
        self.assertIsNotNone(ds.action_space)
        x, y = ld.flatten_pairs(ds)
        self.assertEqual(x.shape[0], y.shape[0])
        self.assertGreater(x.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_demo_loader -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'load_expert_demos'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/load_expert_demos.py`:

```python
"""Load expert-demo files (gnrl_v1 envelope or legacy godot_rl bare array).

A demo file is one of:
  * gnrl_v1:  {"format_version": "gnrl_v1", "action_space": {...},
               "demo_trajectories": [[obs_list, acts_list], ...]}
  * godot_rl: [[obs_list, acts_list], ...]   (bare top-level array)

Each trajectory keeps one more observation than action (the terminal obs has no action).
"""
import json
from pathlib import Path

import numpy as np

GNRL_V1 = "gnrl_v1"


class DemoSet:
    def __init__(self, trajectories, action_space):
        self.trajectories = trajectories  # list[(obs: (T+1, od), acts: (T, ad))]
        self.action_space = action_space  # dict | None (None for legacy godot_rl)


def _to_arrays(demo_trajectories):
    out = []
    for i, traj in enumerate(demo_trajectories):
        if not isinstance(traj, list) or len(traj) != 2:
            raise ValueError(f"trajectory {i} must be [obs_list, acts_list]")
        obs_list, acts_list = traj
        if len(obs_list) != len(acts_list) + 1:
            raise ValueError(
                f"trajectory {i}: expected len(obs) == len(acts) + 1, "
                f"got {len(obs_list)} obs / {len(acts_list)} acts")
        obs = np.asarray(obs_list, dtype=np.float32)
        acts = np.asarray(acts_list, dtype=np.float32)
        if obs.ndim != 2:
            raise ValueError(f"trajectory {i}: obs is ragged or not 2-D ({obs.shape})")
        if acts.size and acts.ndim != 2:
            raise ValueError(f"trajectory {i}: acts is ragged or not 2-D ({acts.shape})")
        out.append((obs, acts))
    return out


def load_demos(path) -> DemoSet:
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        if raw.get("format_version") != GNRL_V1:
            raise ValueError(f"unknown demo format_version: {raw.get('format_version')!r}")
        return DemoSet(_to_arrays(raw["demo_trajectories"]), raw.get("action_space"))
    if isinstance(raw, list):
        return DemoSet(_to_arrays(raw), None)
    raise ValueError(f"unrecognized demo top-level type: {type(raw).__name__}")


def flatten_pairs(demoset: DemoSet):
    """Stack all trajectories into (X=obs[:-1], Y=acts) supervised pairs for BC."""
    xs, ys = [], []
    for obs, acts in demoset.trajectories:
        if acts.size == 0:
            continue
        xs.append(obs[:-1])  # drop terminal obs (no action)
        ys.append(acts)
    if not xs:
        raise ValueError("no (obs, action) pairs in demo set")
    return np.concatenate(xs), np.concatenate(ys)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_demo_loader -v`
Expected: PASS — all 7 tests OK. (Requires Task 5's committed sample to exist for `test_loads_committed_sample`.)

- [ ] **Step 5: Commit**

```bash
git add scripts/load_expert_demos.py test/python/test_demo_loader.py
git commit -m "feat(demos): version-aware Python demo loader (gnrl_v1 + legacy godot_rl) (#13)"
```

---

## Task 7: BC trainer `train_bc.py`

**Files:**
- Create: `scripts/train_bc.py`
- Test: `test/python/test_train_bc.py`

- [ ] **Step 1: Write the failing test**

Create `test/python/test_train_bc.py`:

```python
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_bc as bc  # noqa: E402


class TrainBCTest(unittest.TestCase):
    def test_resolve_branches_from_action_space(self):
        space = {"move": {"size": 5, "action_type": "discrete"}}
        branches = bc.resolve_branches(space, None, y_width=1)
        self.assertEqual(branches, [{"type": "discrete", "size": 5}])

    def test_resolve_branches_legacy_requires_action_type(self):
        with self.assertRaises(ValueError):
            bc.resolve_branches(None, None, y_width=2)

    def test_train_discrete_reduces_loss_and_exports(self):
        import numpy as np
        # Separable synthetic discrete demo: action = 1 if x0 > 0 else 0.
        rng = np.random.default_rng(0)
        x = rng.normal(size=(256, 3)).astype("float32")
        y = (x[:, :1] > 0).astype("float32")  # shape (256, 1), classes {0,1}
        branches = [{"type": "discrete", "size": 2}]
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "bc.pt")
            first, last = bc.train(x, y, branches, epochs=80, lr=0.05, hidden=32, out_path=out)
            self.assertLess(last, first, "BC loss should decrease")
            self.assertTrue(Path(out).exists(), "TorchScript model written")
            self.assertTrue(Path(out + ".shape.json").exists(), "shape sidecar written")
            # Exported model takes obs_dim=3 and returns 2 logits.
            import torch
            m = torch.jit.load(out)
            self.assertEqual(tuple(m(torch.zeros(1, 3)).shape), (1, 2))

    def test_train_continuous_exports_matching_width(self):
        import numpy as np
        rng = np.random.default_rng(1)
        x = rng.normal(size=(128, 4)).astype("float32")
        y = (x[:, :2] * 0.5).astype("float32")  # 2-D continuous target
        branches = [{"type": "continuous", "size": 2}]
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "bc.pt")
            first, last = bc.train(x, y, branches, epochs=80, lr=0.05, hidden=32, out_path=out)
            self.assertLess(last, first)
            import torch
            m = torch.jit.load(out)
            self.assertEqual(tuple(m(torch.zeros(1, 4)).shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_train_bc -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_bc'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/train_bc.py`:

```python
"""Behavior cloning over expert demos -> TorchScript model + shape sidecar.

The model matches the deploy contract (logits for discrete, means for continuous), so
`export_to_ncnn.py models/<bc>.pt` consumes it unchanged. Run in .venv-train (torch via SB3).

Usage:
  .venv-train/bin/python scripts/train_bc.py --demos demos.json --out models/bc_policy.pt
  # legacy godot_rl files (no action_space metadata) need --action-type:
  .venv-train/bin/python scripts/train_bc.py --demos old.json --out m.pt --action-type discrete
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from load_expert_demos import load_demos, flatten_pairs  # noqa: E402


def resolve_branches(action_space, action_type, y_width):
    """Return [{"type", "size"}] action branches. Sizes: discrete = #classes (output logits),
    continuous = action dim. Legacy files (no action_space) use a single --action-type branch."""
    if action_space:
        branches = []
        for spec in action_space.values():
            branches.append({"type": spec["action_type"], "size": int(spec["size"])})
        return branches
    if action_type is None:
        raise ValueError("legacy godot_rl demos have no action_space; pass --action-type")
    if action_type == "continuous":
        return [{"type": "continuous", "size": int(y_width)}]
    return [{"type": "discrete", "size": None}]  # size resolved from data in train()


def _build_model(obs_dim, out_dim, hidden):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


def _bc_loss(out, y, branches):
    import torch.nn.functional as F
    loss = None
    o_off = 0  # output-column offset
    y_off = 0  # target-column offset
    for b in branches:
        size = b["size"]
        if b["type"] == "discrete":
            term = F.cross_entropy(out[:, o_off:o_off + size], y[:, y_off].long())
            o_off += size
            y_off += 1
        else:
            term = F.mse_loss(out[:, o_off:o_off + size], y[:, y_off:y_off + size])
            o_off += size
            y_off += size
        loss = term if loss is None else loss + term
    return loss


def train(x, y, branches, epochs, lr, hidden, out_path):
    import torch
    # Resolve any data-derived discrete sizes (legacy single-branch path).
    for j, b in enumerate(branches):
        if b["type"] == "discrete" and b["size"] is None:
            b["size"] = int(y[:, j].max()) + 1
    out_dim = sum(b["size"] for b in branches)
    obs_dim = x.shape[1]
    model = _build_model(obs_dim, out_dim, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)

    first_loss = None
    last_loss = None
    for _ in range(epochs):
        opt.zero_grad()
        loss = _bc_loss(model(xt), yt, branches)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        if first_loss is None:
            first_loss = last_loss

    model.eval()
    scripted = torch.jit.trace(model, torch.zeros(1, obs_dim))
    scripted.save(out_path)
    Path(out_path + ".shape.json").write_text(json.dumps({"inputshape": f"[1,{obs_dim}]"}))
    return first_loss, last_loss


def main():
    ap = argparse.ArgumentParser(description="Behavior cloning over expert demos.")
    ap.add_argument("--demos", required=True)
    ap.add_argument("--out", default="models/bc_policy.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--action-type", choices=["discrete", "continuous"], default=None)
    args = ap.parse_args()

    ds = load_demos(args.demos)
    x, y = flatten_pairs(ds)
    branches = resolve_branches(ds.action_space, args.action_type, y_width=y.shape[1])
    first, last = train(x, y, branches, args.epochs, args.lr, args.hidden, args.out)
    print(f"BC done: loss {first:.4f} -> {last:.4f}; wrote {args.out} (+ .shape.json)")
    print(f"Next: .venv-train/bin/python scripts/export_to_ncnn.py {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_train_bc -v`
Expected: PASS — 4 tests OK.

- [ ] **Step 5: End-to-end sanity over the committed sample (manual, not in suite)**

Run:

```bash
.venv-train/bin/python scripts/train_bc.py \
  --demos examples/chase_the_target/demos/chase_expert_demos.json \
  --out /tmp/chase_bc.pt --epochs 200
```

Expected: prints `BC done: loss A -> B` with B < A and writes `/tmp/chase_bc.pt` + `.shape.json`. (Optional further check: `scripts/export_to_ncnn.py /tmp/chase_bc.pt` produces ncnn artifacts — proves the BC model is deployable.)

- [ ] **Step 6: Commit**

```bash
git add scripts/train_bc.py test/python/test_train_bc.py
git commit -m "feat(demos): behavior-cloning trainer (demos -> TorchScript -> ncnn) (#13)"
```

---

## Task 8: Documentation + close issue

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `docs/BACKLOG.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`

- [ ] **Step 1: Update `docs/BACKLOG.md` item 10**

Change the item-10 line (`183`) from `⬜` to `✅` and append a "Done 2026-06-04" note with the spec/plan paths and a one-line summary (mirror the format of the already-done items, e.g. item 11).

- [ ] **Step 2: Update the gap-analysis rows**

In `docs/godot-rl-gap-analysis-2026-06-02.md`, flip the four `#13` rows (`RECORD_EXPERT_DEMOS mode`, `get_action() for demo recording`, `expert_demo_save_path export`, `remove_last_episode_key binding`) from **Gap** to ✅ done, noting our `gnrl_v1` default format + `godot_rl` interop, and update the priority table row (`130`).

- [ ] **Step 3: Update `CLAUDE.md`**

- Add a "Key commands" bullet for recording + BC:
  - record: `godot --headless --path . res://examples/chase_the_target/record_chase_demos.tscn -- --demo-out=PATH`
  - clone: `.venv-train/bin/python scripts/train_bc.py --demos PATH --out models/bc.pt` → `export_to_ncnn.py`.
- Add item 10 to the **Done** list with a terse description (recorder + gnrl_v1/godot_rl formats + loader + BC).
- One line in the addon paragraph: `training/demo_recorder.gd` records expert demos; `RECORD_EXPERT_DEMOS` mode on `NcnnSync`.

- [ ] **Step 4: Update `README.md`**

Add a short "Imitation learning / expert demos" subsection: how to record (RECORD_EXPERT_DEMOS scene), the two on-disk formats (`gnrl_v1` default, `godot_rl` interop), and the `train_bc.py` → `export_to_ncnn.py` path. Follow the README's existing section style.

- [ ] **Step 5: Run the full suite**

Run: `./test/run_tests.sh`
Expected: ends with `All tests passed.` (gate on that line + exit 0, per CLAUDE.md — do NOT grep for "failed").

- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md docs/BACKLOG.md docs/godot-rl-gap-analysis-2026-06-02.md
git commit -m "docs: expert-demo recording + behavior cloning (Closes #13)"
```

---

## Self-Review Notes

- **Spec coverage:** `gnrl_v1`/`godot_rl` formats (Task 1 + 6), version-aware loader reading both (Task 6), `action_space` metadata in envelope (Task 1, consumed Task 7), `DemoRecorder` godot_rl-faithful loop (Task 1), `get_action()` hook (Task 2), `NcnnSync` RECORD mode single-agent + offline + save (Task 3), scripted-expert + human note (Task 4), `load_expert_demos.py` + `train_bc.py` (Tasks 6–7), example + committed sample (Tasks 4–5), all four test rows from the spec + `run_tests.sh` wiring (Tasks 1,5,6,7), docs + gap rows + `Closes #13` (Task 8). The human-input `set_action()` path is documented-only (headless can't supply Input), as the spec's risk note allows.
- **Naming consistency:** `record_step`, `remove_last_episode`, `trajectory_count`, `step_count`, `to_json(demo_format, action_space)`, `save_expert_demos`, `expert_demo_save_path`, `demo_format`, `expert_action_index`, `resolve_branches`, `train`, `flatten_pairs`, `load_demos`, `DemoSet` are used identically across the tasks that define and call them.
- **No placeholders:** every code step shows complete code; the two prose doc steps (README/CLAUDE) describe exact content and section placement.
