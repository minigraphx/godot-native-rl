extends Node
# Drop-in episode recorder (#39): buffers trajectories and writes one JSON per finished episode
# (ring of keep_last per stream). Two capture paths, zero per-game agent changes:
#
# - TRAINING (attach_sync): taps NcnnSync's additive actions_received/step_sent signals. Records
#   one agent (agent_index) or — #195 — EVERY agent slot (record_all_agents), each with its own
#   episode segmentation (per-agent dones) and file stream (episode_NNNN_train_K.json).
# - INFERENCE (attach_agent, #194): taps a deployed controller's existing inference_step signal —
#   no trainer, no socket; record shipped agents in a play scene. Episode boundaries are detected
#   by the controller's n_steps wrapping (robust regardless of who clears `done`); per-step reward
#   is the delta of the agent's accumulator. Honest limit: reward accrued between the last
#   decision and the reset is zeroed by the agent's own reset before we can read it, so the final
#   partial window's reward is not captured (actions are exact; rewards are exact per decision
#   window otherwise). Attach several agents to capture a multi-agent inference scene.
#
# initial_state needs an opt-in game.get_replay_state() hook (exact restore for seeded kinematic
# games). Spec: docs/superpowers/specs/2026-06-12-episode-replay-design.md

const ReplayFormat = preload("res://addons/godot_native_rl/training/replay_format.gd")

@export var out_dir := "user://replays"
@export var keep_last := 10
@export var agent_index := 0
@export var record_all_agents := false  ## training path: capture every agent slot (#195)
@export var game_path: NodePath
@export var sync_path: NodePath  ## empty -> auto-find a sibling exposing the replay signals
## Decision cadence stamped into inference-path meta (training-path meta reads the sync's).
@export var inference_action_repeat := 8

var _game: Node
var _saved_paths: Dictionary = {}   # stream key -> Array[String] (ring per stream)
var _episode_index: Dictionary = {} # stream key -> int
var _pending_actions: Array = []
var _steps: Dictionary = {}         # stream key -> Array[step]
var _initial_state: Dictionary = {}
var _warned_no_state := false
var _action_repeat := 0
var _agent_state: Dictionary = {}   # instance_id -> {"last_reward": float, "last_n_steps": int}

func _ready() -> void:
	_game = get_node_or_null(game_path)
	var sync := get_node_or_null(sync_path)
	if sync == null and get_parent() != null:
		for n in get_parent().get_children():
			if n.has_signal("actions_received") and n.has_signal("step_sent"):
				sync = n
				break
	if sync == null:
		push_error("ReplayRecorder: no NcnnSync with replay signals found (use attach_agent() for inference-time capture).")
		return
	attach_sync(sync)
	_snapshot_initial_state()

# --- Training path (NcnnSync signals) ---

# Split out so tests can attach a stub emitter directly.
func attach_sync(sync: Node) -> void:
	sync.actions_received.connect(_on_actions)
	sync.step_sent.connect(_on_step)
	if "action_repeat" in sync:
		_action_repeat = int(sync.action_repeat)

func _on_actions(actions: Array) -> void:
	_pending_actions = actions

func _on_step(rewards: Array, dones: Array) -> void:
	if _pending_actions.is_empty():
		return
	var indices: Array = range(rewards.size()) if record_all_agents else [agent_index]
	for idx in indices:
		if idx >= _pending_actions.size() or idx >= rewards.size():
			continue
		var key := _stream_key(idx)
		if not _steps.has(key):
			_steps[key] = []
		_steps[key].append({"action": _pending_actions[idx], "reward": float(rewards[idx])})
		if idx < dones.size() and dones[idx]:
			_finish_episode(key, {"mode": "training", "agent_index": idx, "action_repeat": _action_repeat})
	_pending_actions = []

# --- Inference path (controller inference_step signal, #194) ---

## Record a DEPLOYED agent (NCNN_INFERENCE / any consumer of NcnnControllerCore): call once per
## agent to capture; several attached agents record side by side. No Sync required.
func attach_agent(agent: Node) -> void:
	if not agent.has_signal("inference_step"):
		push_error("ReplayRecorder.attach_agent: '%s' has no inference_step signal." % agent.name)
		return
	_agent_state[agent.get_instance_id()] = {"last_reward": 0.0, "last_n_steps": -1}
	agent.inference_step.connect(_on_inference_step.bind(agent))
	if _game == null:
		_game = get_node_or_null(game_path)
	if _initial_state.is_empty():
		_snapshot_initial_state()

func _on_inference_step(payload: Dictionary, agent: Node) -> void:
	var state: Dictionary = _agent_state[agent.get_instance_id()]
	var key := "inf_" + String(agent.name)
	var n_steps := int(agent.get("n_steps"))
	if n_steps < int(state["last_n_steps"]):
		# The controller reset since the last decision — the buffered episode is complete.
		_finish_episode(key, {"mode": "inference", "agent": String(agent.name),
			"action_repeat": inference_action_repeat})
		state["last_reward"] = 0.0
	var reward_now := float(agent.get("reward"))
	var delta := reward_now - float(state["last_reward"])
	if reward_now < float(state["last_reward"]):
		delta = reward_now  # accumulator was zeroed by a reset between decisions
	if not _steps.has(key):
		_steps[key] = []
	_steps[key].append({"action": payload.get("action", {}), "reward": delta})
	state["last_reward"] = reward_now
	state["last_n_steps"] = n_steps

## Flush any buffered inference steps as final (partial) episodes — call before quitting a
## recording session so the tail isn't lost (training episodes flush on their done flags).
func flush_inference_episodes() -> void:
	for key in _steps.keys():
		if String(key).begins_with("inf_") and not (_steps[key] as Array).is_empty():
			_finish_episode(key, {"mode": "inference", "agent": String(key).trim_prefix("inf_"),
				"action_repeat": inference_action_repeat, "partial": true})

# --- Shared internals ---

func _stream_key(idx: int) -> String:
	return "train_%d" % idx

func _snapshot_initial_state() -> void:
	if _game != null and _game.has_method("get_replay_state"):
		_initial_state = _game.get_replay_state()
		return
	if not _warned_no_state:
		_warned_no_state = true
		push_warning("ReplayRecorder: game has no get_replay_state() — replays start from the scene's default reset.")
	_initial_state = {}

func _finish_episode(key: String, extra_meta: Dictionary) -> void:
	var steps: Array = _steps.get(key, [])
	if steps.is_empty():
		return
	var scene_path := ""
	if get_tree() != null and get_tree().current_scene != null:
		scene_path = String(get_tree().current_scene.scene_file_path)
	var meta := {"scene": scene_path, "recorded_at": Time.get_datetime_string_from_system()}
	meta.merge(extra_meta, true)
	var ep := ReplayFormat.make_episode(meta, _initial_state, steps)
	_steps[key] = []
	DirAccess.make_dir_recursive_absolute(out_dir)
	var index := int(_episode_index.get(key, 0))
	_episode_index[key] = index + 1
	# The classic single-agent training stream keeps its historical file names
	# (episode_NNNN.json); every other stream carries its key as a suffix.
	var suffix := "" if (key == _stream_key(agent_index) and not record_all_agents) else "_%s" % key
	var path := out_dir.path_join("episode_%04d%s.json" % [index, suffix])
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		push_error("ReplayRecorder: cannot write '%s'." % path)
		return
	f.store_string(ReplayFormat.to_json(ep))
	f.close()
	if not _saved_paths.has(key):
		_saved_paths[key] = []
	_saved_paths[key].append(path)
	while (_saved_paths[key] as Array).size() > keep_last:
		DirAccess.remove_absolute(_saved_paths[key].pop_front())
	print("ReplayRecorder: saved %s (%d steps, total_reward %.2f)" % [path, ep["meta"]["n_steps"], ep["meta"]["total_reward"]])
	_snapshot_initial_state()
