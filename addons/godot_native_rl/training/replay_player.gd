extends Node
# Plays a gnrl_replay_v1 episode back into a scene (#39): restores the recorded initial state
# (opt-in game.apply_replay_state hook), then feeds the recorded actions to the agent's
# set_action at the recorded action_repeat cadence. The policy/net is never consulted.
# Determinism caveat: exact for kinematic seeded games (chase); approximate for physics envs
# (Jolt is not cross-run deterministic — see #60).
# Spec: docs/superpowers/specs/2026-06-12-episode-replay-design.md

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
	if _agent == null:
		_agent = get_node_or_null(agent_path)
	if _game == null:
		_game = get_node_or_null(game_path)
	if autoplay:
		play()

func set_nodes_for_test(agent: Node, game: Node) -> void:
	_agent = agent
	_game = game

func play() -> bool:
	if _agent == null:
		push_error("ReplayPlayer: agent_path not set/invalid.")
		return false
	if "control_mode" in _agent and int(_agent.control_mode) == 3:
		push_warning("ReplayPlayer: agent is in NCNN_INFERENCE mode — two drivers will fight; use a non-policy mode.")
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

func is_playing() -> bool:
	return _playing

func _physics_process(_delta: float) -> void:
	if not _playing:
		return
	step_frame()

# One playback frame; public so headless tests can drive it without a ticking physics loop.
func step_frame() -> void:
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
