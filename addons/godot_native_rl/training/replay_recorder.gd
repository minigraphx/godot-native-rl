extends Node
# Drop-in training-episode recorder (#39): connects to NcnnSync's actions_received/step_sent,
# buffers one agent's trajectory, writes one JSON per finished episode (ring of keep_last).
# Zero per-game agent changes; initial_state needs an opt-in game.get_replay_state() hook.
# v1 records TRAINING mode, one agent (agent_index); inference-time + multi-agent capture are
# tracked follow-ups. Spec: docs/superpowers/specs/2026-06-12-episode-replay-design.md

const ReplayFormat = preload("res://addons/godot_native_rl/training/replay_format.gd")

@export var out_dir := "user://replays"
@export var keep_last := 10
@export var agent_index := 0
@export var game_path: NodePath
@export var sync_path: NodePath  ## empty -> auto-find a sibling exposing the replay signals

var _game: Node
var _episode_index := 0
var _saved_paths: Array = []
var _pending_action = null
var _steps: Array = []
var _initial_state: Dictionary = {}
var _warned_no_state := false
var _action_repeat := 0

func _ready() -> void:
	_game = get_node_or_null(game_path)
	var sync := get_node_or_null(sync_path)
	if sync == null and get_parent() != null:
		for n in get_parent().get_children():
			if n.has_signal("actions_received") and n.has_signal("step_sent"):
				sync = n
				break
	if sync == null:
		push_error("ReplayRecorder: no NcnnSync with replay signals found.")
		return
	attach_sync(sync)
	_snapshot_initial_state()

# Split out so tests can attach a stub emitter directly.
func attach_sync(sync: Node) -> void:
	sync.actions_received.connect(_on_actions)
	sync.step_sent.connect(_on_step)
	if "action_repeat" in sync:
		_action_repeat = int(sync.action_repeat)

func _snapshot_initial_state() -> void:
	if _game != null and _game.has_method("get_replay_state"):
		_initial_state = _game.get_replay_state()
		return
	if not _warned_no_state:
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
	var scene_path := ""
	if get_tree() != null and get_tree().current_scene != null:
		scene_path = String(get_tree().current_scene.scene_file_path)
	var meta := {"scene": scene_path, "agent_index": agent_index, "action_repeat": _action_repeat,
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
