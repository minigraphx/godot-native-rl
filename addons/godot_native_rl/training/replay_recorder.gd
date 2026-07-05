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
## FALLBACK decision cadence for inference-path meta when an episode is too short to observe it
## (#311): the real cadence is derived per episode from the controller's n_steps delta between
## decisions, so a deploy-time `action_repeat=` override can never desync playback.
@export var inference_action_repeat := 8

var _game: Node
var _saved_paths: Dictionary = {}   # stream key -> Array[String] (ring per stream)
var _episode_index: Dictionary = {} # stream key -> int
var _pending_actions: Array = []
var _steps: Dictionary = {}         # stream key -> Array[step]
var _initial_state: Dictionary = {} # scene-start snapshot (fallback for a stream's first episode)
var _stream_initial: Dictionary = {} # stream key -> per-episode snapshot, taken at episode START (#310)
var _warned_no_state := false
var _action_repeat := 0
# instance_id -> {"last_reward","last_n_steps","cadence","name","key"} — keyed by instance id,
# NOT node name: Godot only enforces name uniqueness among siblings, and multi-agent capture
# instances the same subscene N times (#309).
var _agent_state: Dictionary = {}

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
	# A stream whose buffer is empty is STARTING an episode with this action window: snapshot the
	# game now (post-reset, pre-action). Snapshotting at finish time instead corrupts every other
	# in-progress stream's initial state under multi-agent capture (#310).
	var indices: Array = range(actions.size()) if record_all_agents else [agent_index]
	for idx in indices:
		var key := _stream_key(idx)
		if not _steps.has(key) or (_steps[key] as Array).is_empty():
			_stream_initial[key] = _capture_state()

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
	# Validate the FULL recordable surface (#312): the inference_step signal alone is a broader
	# contract — crowd units emit it via NcnnCrowdController but carry no n_steps/reward
	# properties (those live on the shared controller), and int(null) errors on every decision.
	if not agent.has_signal("inference_step") or agent.get("n_steps") == null or agent.get("reward") == null:
		push_error("ReplayRecorder.attach_agent: '%s' must expose inference_step + n_steps + reward (crowd units are driven by the shared NcnnCrowdController — attach a wrapper controller instead)." % agent.name)
		return
	var iid := agent.get_instance_id()
	# Stream key carries the instance id (#309): same-named agents (the same subscene instanced
	# under different parents) must never share a buffer, an episode index, or a file ring.
	_agent_state[iid] = {"last_reward": 0.0, "last_n_steps": -1, "cadence": 0,
		"name": String(agent.name), "key": "inf_%s_%d" % [String(agent.name), iid]}
	agent.inference_step.connect(_on_inference_step.bind(agent))
	if _game == null:
		_game = get_node_or_null(game_path)
	if _initial_state.is_empty():
		_snapshot_initial_state()

func _on_inference_step(payload: Dictionary, agent: Node) -> void:
	var state: Dictionary = _agent_state[agent.get_instance_id()]
	var key: String = state["key"]
	var n_steps := int(agent.get("n_steps"))
	if n_steps < int(state["last_n_steps"]):
		# The controller reset since the last decision — the buffered episode is complete.
		_finish_episode(key, {"mode": "inference", "agent": String(state["name"]),
			"action_repeat": _observed_cadence(state)})
		state["last_reward"] = 0.0
	elif int(state["last_n_steps"]) >= 0:
		# Mid-episode: the n_steps delta between consecutive decisions IS the decision cadence
		# (#311) — record the observed value instead of trusting a hand-synced export that a
		# deploy-time `action_repeat=` override silently desyncs.
		state["cadence"] = n_steps - int(state["last_n_steps"])
	var reward_now := float(agent.get("reward"))
	# Plain delta of the accumulator — a decrease is a legitimate per-step penalty (#308); resets
	# are detected exactly by the n_steps wrap above, which zeroes last_reward.
	var delta := reward_now - float(state["last_reward"])
	if not _steps.has(key) or (_steps[key] as Array).is_empty():
		_steps[key] = _steps.get(key, [])
		_stream_initial[key] = _capture_state()  # episode start (#310)
	_steps[key].append({"action": payload.get("action", {}), "reward": delta})
	state["last_reward"] = reward_now
	state["last_n_steps"] = n_steps

func _observed_cadence(state: Dictionary) -> int:
	return int(state["cadence"]) if int(state["cadence"]) > 0 else inference_action_repeat

## Flush any buffered inference steps as final (partial) episodes — call before quitting a
## recording session so the tail isn't lost (training episodes flush on their done flags).
func flush_inference_episodes() -> void:
	for state in _agent_state.values():
		var key: String = state["key"]
		if _steps.has(key) and not (_steps[key] as Array).is_empty():
			_finish_episode(key, {"mode": "inference", "agent": String(state["name"]),
				"action_repeat": _observed_cadence(state), "partial": true})

# --- Shared internals ---

func _stream_key(idx: int) -> String:
	return "train_%d" % idx

func _snapshot_initial_state() -> void:
	_initial_state = _capture_state()

func _capture_state() -> Dictionary:
	if _game != null and _game.has_method("get_replay_state"):
		return _game.get_replay_state()
	if not _warned_no_state:
		_warned_no_state = true
		push_warning("ReplayRecorder: game has no get_replay_state() — replays start from the scene's default reset.")
	return {}

func _finish_episode(key: String, extra_meta: Dictionary) -> void:
	var steps: Array = _steps.get(key, [])
	if steps.is_empty():
		return
	var scene_path := ""
	if get_tree() != null and get_tree().current_scene != null:
		scene_path = String(get_tree().current_scene.scene_file_path)
	var meta := {"scene": scene_path, "recorded_at": Time.get_datetime_string_from_system()}
	meta.merge(extra_meta, true)
	# Per-stream snapshot taken at THIS episode's start (#310); the scene-start snapshot only
	# backstops a stream whose first episode began before any capture point ran.
	var ep := ReplayFormat.make_episode(meta, _stream_initial.get(key, _initial_state), steps)
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
