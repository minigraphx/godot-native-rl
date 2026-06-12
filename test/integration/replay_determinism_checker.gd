extends Node
# Replay determinism acceptance test (#39): drive the REAL chase game+agent with a scripted
# action sequence (recording it through the real ReplayFormat), then replay the saved episode
# into the SAME (reset) scene via the real ReplayPlayer and assert the final game state EXACTLY
# matches the recorded end state. Chase is kinematic — replay must be bit-exact.

const RF = preload("res://addons/godot_native_rl/training/replay_format.gd")
const Player = preload("res://addons/godot_native_rl/training/replay_player.gd")

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames := 120

const OUT := "user://replay_determinism_episode.json"

var _game
var _agent
var _phase := 0  # 0 = record, 1 = replay
var _frame := 0
var _steps: Array = []
var _initial_state: Dictionary = {}
var _recorded_end: Dictionary = {}
var _record_mid: Dictionary = {}
var _replay_mid: Dictionary = {}
var _player

# Vacuous-pass guard: the symmetric action sweep returns the agent home, so end==start could
# also mean "no actions were ever applied". Require real mid-trajectory motion in BOTH phases.
# (Exact mid-vs-mid equality is deliberately NOT asserted: action delivery is offset by one
# tick between the phases — the stay-tail makes the END exact, mids can differ by <= one move.)
func _moved(mid: Dictionary) -> bool:
	if mid.is_empty():
		return false
	var dx: float = absf(float(mid["agent_x"]) - float(_initial_state["agent_x"]))
	var dy: float = absf(float(mid["agent_y"]) - float(_initial_state["agent_y"]))
	return dx + dy > 10.0

var _pinned := false

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("missing game/agent")

func _scripted_action(i: int) -> Dictionary:
	# Sweep the 4 movement directions for 100 frames, then a 20-frame "stay" (move=0) tail:
	# the agent keeps applying its LAST action for a tick or two around phase boundaries
	# (agent _physics_process runs before this checker's), so ending on "stay" makes the final
	# state insensitive to that bounded drift while the moving portion stays fully exercised.
	if i >= 100:
		return {"move": 0}
	return {"move": 1 + (i / 10) % 4}

func _physics_process(_delta: float) -> void:
	if _game == null:
		return
	# Pin the start state on the FIRST physics tick — NOT in _ready: children _ready before the
	# parent, so ChaseGame._ready()'s reset_positions() would randomize the start AFTER an
	# _ready-time pin (the bug the first run of this very test caught).
	if not _pinned:
		_pinned = true
		_game.apply_replay_state({"agent_x": 100.0, "agent_y": 100.0, "target_x": 800.0, "target_y": 500.0, "catches": 0})
		_agent.set_action({"move": 0})
		_initial_state = _game.get_replay_state()
		return
	match _phase:
		0:
			if _frame < frames:
				if _frame == 50:
					_record_mid = _game.get_replay_state()
				var a := _scripted_action(_frame)
				_agent.set_action(a)
				_steps.append({"action": a, "reward": 0.0})
				_frame += 1
				return
			# Recording done: persist + capture end state, then reset and start replay.
			_recorded_end = _game.get_replay_state()
			var ep := RF.make_episode({"action_repeat": 1}, _initial_state, _steps)
			var f := FileAccess.open(OUT, FileAccess.WRITE)
			f.store_string(RF.to_json(ep))
			f.close()
			# Scramble the scene state so the replay must restore it.
			_game.apply_replay_state({"agent_x": 500.0, "agent_y": 50.0, "target_x": 50.0, "target_y": 50.0, "catches": 7})
			_agent.set_action({"move": 0})
			_player = Player.new()
			_player.autoplay = false
			_player.replay_path = OUT
			_player.set_nodes_for_test(_agent, _game)
			add_child(_player)
			if not _player.play():
				_fail("player failed to load the recorded episode")
				return
			_frame = 0
			_phase = 1
		1:
			if _player.is_playing():
				if _frame == 50:
					_replay_mid = _game.get_replay_state()
				_frame += 1
				if _frame > frames + 10:
					_fail("replay did not finish in time")
				return
			if not _moved(_record_mid):
				_fail("recording phase never moved the agent (vacuous test)")
				return
			if not _moved(_replay_mid):
				_fail("replay phase never moved the agent (actions not delivered)")
				return
			var end: Dictionary = _game.get_replay_state()
			for k in _recorded_end:
				if typeof(_recorded_end[k]) == TYPE_FLOAT:
					if absf(float(end[k]) - float(_recorded_end[k])) > 1e-6:
						_fail("state mismatch at '%s': recorded %s vs replayed %s" % [k, _recorded_end[k], end[k]])
						return
				elif end[k] != _recorded_end[k]:
					_fail("state mismatch at '%s': recorded %s vs replayed %s" % [k, _recorded_end[k], end[k]])
					return
			print("REPLAY DETERMINISM PASSED (%d frames reproduced exactly: %s)" % [frames, JSON.stringify(end)])
			get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("REPLAY DETERMINISM FAILED: %s" % reason)
	get_tree().quit(1)
