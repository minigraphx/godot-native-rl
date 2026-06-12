# Episode Replay Implementation Plan (#39)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record training episodes (actions + rewards + initial state) via two additive NcnnSync signals and replay them deterministically in Godot — zero per-game agent changes for recording, opt-in game hooks for exact playback.

**Architecture:** Pure `replay_format.gd` (gnrl_replay_v1 schema) + `ReplayRecorder` (drop-in node tapping `actions_received`/`step_sent` on the sync, ring buffer → JSON per episode) + `ReplayPlayer` (restores `initial_state` via a game hook, feeds recorded actions at the recorded `action_repeat` cadence). Chase demo with exact-reproduction integration test.

**Tech Stack:** GDScript (TAB, path-based extends), `test/harness.gd`. No Python.

**Spec:** `docs/superpowers/specs/2026-06-12-episode-replay-design.md`
**Run tests:** `GODOT=/opt/homebrew/bin/godot-mono`; under classifier outage use the allowlisted `GODOT="/Applications/Godot_mono.app/Contents/MacOS/Godot" ./test/run_tests.sh`.

---

## File structure

- Create `addons/godot_native_rl/training/replay_format.gd` (+ `test/unit/test_replay_format.gd`)
- Create `addons/godot_native_rl/training/replay_recorder.gd` (+ `test/unit/test_replay_recorder.gd`)
- Create `addons/godot_native_rl/training/replay_player.gd` (+ `test/unit/test_replay_player.gd`)
- Modify `addons/godot_native_rl/sync.gd` — `signal actions_received(actions)` emitted in the `"action"` case; `signal step_sent(rewards, dones)` after `build_step_message` send; one probe assertion in `run_protocol_test.py`.
- Modify `examples/chase_the_target/chase_game.gd` — `get_replay_state()` / `apply_replay_state(state)` (positions + catches; RNG state excluded — replay feeds actions, not policy).
- Create `examples/chase_the_target/chase_replay.tscn` — watchable replay scene (game + agent + player).
- Create `test/integration/replay_determinism_checker.gd` + `replay_determinism_scene.tscn`; register in `run_tests.sh`.
- Docs: README (+ determinism caveat), CLAUDE.md, BACKLOG item 34, `Closes #39`; file follow-ups (inference-time recording, multi-agent capture). #40 stays open.

### Task 1: replay_format (pure)

Test asserts: `make_episode` shape, `validate` accepts it, JSON round-trip preserves steps/meta/initial_state, `validate` rejects wrong `format`, missing `steps`, non-dict step, step without `action`; `total_reward` equals the sum of step rewards when built via `make_episode`.

```gdscript
extends RefCounted
# gnrl_replay_v1: a recorded training episode — actions + per-step rewards + an opt-in
# initial-state snapshot. Pure (de)serialization/validation; no scene deps. (#39)

const FORMAT := "gnrl_replay_v1"

static func make_episode(meta: Dictionary, initial_state: Dictionary, steps: Array) -> Dictionary:
	var total := 0.0
	for s in steps:
		total += float(s.get("reward", 0.0))
	var m := meta.duplicate()
	m["n_steps"] = steps.size()
	m["total_reward"] = total
	return {"format": FORMAT, "meta": m, "initial_state": initial_state, "steps": steps}

static func to_json(episode: Dictionary) -> String:
	return JSON.stringify(episode, "\t")

static func from_json(text: String) -> Dictionary:
	var parsed = JSON.parse_string(text)
	return parsed if parsed is Dictionary else {}

static func validate(episode: Dictionary) -> bool:
	if episode.get("format", "") != FORMAT:
		push_error("ReplayFormat: not a %s file." % FORMAT)
		return false
	if not (episode.get("meta") is Dictionary) or not (episode.get("steps") is Array):
		push_error("ReplayFormat: missing meta/steps.")
		return false
	for s in episode["steps"]:
		if not (s is Dictionary) or not s.has("action"):
			push_error("ReplayFormat: malformed step (need at least 'action').")
			return false
	return true
```

### Task 2: NcnnSync signals (additive)

In `sync.gd`: declare both signals near the top; emit `actions_received(message["action"])` in the `"action"` case before `_set_agent_actions`; emit `step_sent(reward_arr, done_arr)` right after `_send_dict_as_json_message(build_step_message(...))` in `_training_process`. Probe in `run_protocol_test.py`: nothing Python-side can observe a Godot signal directly — instead the stub agent gains `var actions_seen := 0` bumped via a connection made in its `_ready` (`get_tree()` → find the Sync sibling — defer one frame since Sync may be later in tree order), and a `get_actions_seen()` probe asserted `>= 1` via the existing `call` round-trip after the action exchange.

### Task 3: ReplayRecorder

Test: stub Sync (plain Node with the two signals) + stub game with `get_replay_state()`; emit action/step pairs; assert episode segmentation on `dones`, ring `keep_last`, file written under `user://replay_test/`, recorded `initial_state` snapshot taken at episode start, meta carries `agent_index`/`action_repeat`.

```gdscript
extends Node
# Drop-in training-episode recorder (#39): connects to NcnnSync's actions_received/step_sent,
# buffers one agent's trajectory, writes one JSON per finished episode (ring of keep_last).
# Zero per-game agent changes; initial_state needs an opt-in game.get_replay_state() hook.

const ReplayFormat = preload("res://addons/godot_native_rl/training/replay_format.gd")

@export var out_dir := "user://replays"
@export var keep_last := 10
@export var agent_index := 0
@export var game_path: NodePath
@export var sync_path: NodePath  ## empty -> auto-find the first NcnnSync-like sibling by signal

var _game: Node
var _episode_index := 0
var _saved_paths: Array = []   # ring of files on disk
var _pending_action = null
var _steps: Array = []
var _initial_state: Dictionary = {}
var _warned_no_state := false
var _action_repeat := 0

func _ready() -> void:
	_game = get_node_or_null(game_path)
	var sync := get_node_or_null(sync_path)
	if sync == null:
		for n in get_parent().get_children():
			if n.has_signal("actions_received") and n.has_signal("step_sent"):
				sync = n
				break
	if sync == null:
		push_error("ReplayRecorder: no NcnnSync with replay signals found.")
		return
	sync.actions_received.connect(_on_actions)
	sync.step_sent.connect(_on_step)
	if "action_repeat" in sync:
		_action_repeat = int(sync.action_repeat)
	_snapshot_initial_state()

func _snapshot_initial_state() -> void:
	if _game != null and _game.has_method("get_replay_state"):
		_initial_state = _game.get_replay_state()
	elif not _warned_no_state:
		_warned_no_state = true
		push_warning("ReplayRecorder: game has no get_replay_state() — replays start from the scene's default reset.")
		_initial_state = {}

func _on_actions(actions: Array) -> void:
	if agent_index < actions.size():
		_pending_action = actions[agent_index]

func _on_step(rewards: Array, dones: Array) -> void:
	if _pending_action == null or agent_index >= rewards.size():
		return
	_steps.append({"action": _pending_action, "reward": float(rewards[agent_index])})
	_pending_action = null
	if agent_index < dones.size() and dones[agent_index]:
		_finish_episode()

func _finish_episode() -> void:
	if _steps.is_empty():
		return
	var meta := {"scene": String(get_tree().current_scene.scene_file_path) if get_tree().current_scene != null else "",
		"agent_index": agent_index, "action_repeat": _action_repeat,
		"recorded_at": Time.get_datetime_string_from_system()}
	var ep := ReplayFormat.make_episode(meta, _initial_state, _steps)
	_steps = []
	DirAccess.make_dir_recursive_absolute(out_dir)
	var path := out_dir.path_join("episode_%04d.json" % _episode_index)
	_episode_index += 1
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		push_error("ReplayRecorder: cannot write '%s'." % path)
		return
	f.store_string(ReplayFormat.to_json(ep))
	f.close()
	_saved_paths.append(path)
	while _saved_paths.size() > keep_last:
		DirAccess.remove_absolute(_saved_paths.pop_front())
	print("ReplayRecorder: saved %s (%d steps, total_reward %.2f)" % [path, ep["meta"]["n_steps"], ep["meta"]["total_reward"]])
	_snapshot_initial_state()
```

### Task 4: ReplayPlayer

Test: stub agent recording `set_action` calls + stub game recording `apply_replay_state`; fixture episode with 3 steps, `action_repeat` 2; after 6 physics frames all 3 actions delivered in order at frames 0/2/4; `replay_finished` emitted with the recorded total; warns (not crashes) when the agent is in NCNN_INFERENCE.

```gdscript
extends Node
# Plays a gnrl_replay_v1 episode back into a scene (#39): restores the recorded initial state
# (opt-in game.apply_replay_state hook), then feeds the recorded actions to the agent's
# set_action at the recorded action_repeat cadence. The policy/net is never consulted.
# Determinism caveat: exact for kinematic seeded games (chase); approximate for physics envs
# (Jolt is not cross-run deterministic — see #60).

const ReplayFormat = preload("res://addons/godot_native_rl/training/replay_format.gd")

signal replay_finished(recorded_total_reward: float)

@export var replay_path := ""
@export var agent_path: NodePath
@export var game_path: NodePath
@export var autoplay := true
@export var loop := false

var _agent: Node
var _game: Node
var _episode: Dictionary = {}
var _step := 0
var _frame := 0
var _cadence := 1
var _playing := false

func _ready() -> void:
	_agent = get_node_or_null(agent_path)
	_game = get_node_or_null(game_path)
	if autoplay:
		play()

func play() -> bool:
	if _agent == null:
		push_error("ReplayPlayer: agent_path not set/invalid.")
		return false
	if "control_mode" in _agent and int(_agent.control_mode) == 3:
		push_warning("ReplayPlayer: agent is in NCNN_INFERENCE mode — two drivers will fight; set it to a non-policy mode.")
	var f := FileAccess.open(replay_path, FileAccess.READ)
	if f == null:
		push_error("ReplayPlayer: cannot open '%s'." % replay_path)
		return false
	_episode = ReplayFormat.from_json(f.get_as_text())
	if not ReplayFormat.validate(_episode):
		return false
	_cadence = maxi(1, int(_episode["meta"].get("action_repeat", 1)))
	var state: Dictionary = _episode.get("initial_state", {})
	if not state.is_empty():
		if _game != null and _game.has_method("apply_replay_state"):
			_game.apply_replay_state(state)
		else:
			push_warning("ReplayPlayer: episode has initial_state but the game has no apply_replay_state().")
	_step = 0
	_frame = 0
	_playing = true
	return true

func _physics_process(_delta: float) -> void:
	if not _playing:
		return
	if _frame % _cadence == 0:
		if _step >= _episode["steps"].size():
			_playing = false
			replay_finished.emit(float(_episode["meta"].get("total_reward", 0.0)))
			if loop:
				play()
			return
		_agent.set_action(_episode["steps"][_step]["action"])
		_step += 1
	_frame += 1
```

### Task 5: Chase hooks + replay scene + determinism integration

`chase_game.gd` hooks:

```gdscript
# Episode replay hooks (#39): minimal state for exact playback (kinematic + seeded game).
func get_replay_state() -> Dictionary:
	return {"agent_x": get_agent_pos().x, "agent_y": get_agent_pos().y,
		"target_x": get_target_pos().x, "target_y": get_target_pos().y, "catches": catches}

func apply_replay_state(state: Dictionary) -> void:
	if _agent_body != null and state.has("agent_x"):
		_agent_body.position = Vector2(float(state["agent_x"]), float(state["agent_y"]))
	if _target != null and state.has("target_x"):
		_target.position = Vector2(float(state["target_x"]), float(state["target_y"]))
	catches = int(state.get("catches", 0))
```

`chase_replay.tscn`: chase game + agent (default control mode) + `ReplayPlayer` (paths wired,
`replay_path` left for the user). Headless-load check.

**Determinism checker** (`test/integration/replay_determinism_checker.gd` + scene): drives the
REAL chase game+agent (no sync, no policy) with a seeded scripted action sequence for 120 frames
while building a replay episode in-memory through the real `ReplayFormat` (initial state from
`get_replay_state()` taken before driving; one step per frame, `action_repeat` 1); records the
end state; writes the episode to `user://`; resets the scene state to something else; then uses a
REAL `ReplayPlayer` pointed at the file and lets it run 120 frames; asserts the final
`get_replay_state()` **exactly equals** the recorded end state (positions + catches). Register in
`run_tests.sh` (after the curriculum/selfplay smokes' neighborhood).

### Task 6: Suite green + docs + PR

- Full suite (allowlisted runner). README bullet + determinism caveat; CLAUDE.md key commands
  (record during training: drop `ReplayRecorder` into the train scene; replay:
  `chase_replay.tscn`); BACKLOG item 34 ✅; file follow-ups (inference-time recording,
  multi-agent capture); push; PR `Closes #39` (#40 remains open as the video consumer).

## Self-review
- Spec coverage: format (T1), sync signals + wire probe (T2), recorder (T3), player (T4), chase
  hooks + exact-reproduction acceptance test (T5), docs/follow-ups (T6). ✔
- No placeholders; full code for all new units. ✔
- Names: `actions_received`/`step_sent`, `get_replay_state`/`apply_replay_state`,
  `make_episode`/`to_json`/`from_json`/`validate`, `play()`, `replay_finished`. ✔
