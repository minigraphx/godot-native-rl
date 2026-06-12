class_name BallBalanceAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd"

const ACTION_KEY := "tilt"
const ACTION_COUNT := 2
const OBS_SIZE := 8  # tilt(2) + ball rel pos(3) + ball velocity(3), Unity 3DBall parity

@export var game_path: NodePath
# Unity's scheme scaled to our per-physics-frame accumulation (action_repeat spreads one
# decision over several frames): small alive reward each frame, one fall penalty at terminal.
@export var alive_reward := 0.01
@export var fall_penalty := 1.0

var _game
var _tilt := Vector2.ZERO

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("BallBalanceAgent: game_path not set — producing zero observations.")

func clamp_input(v: float) -> float:
	return clampf(v, -1.0, 1.0)

func get_action_space() -> Dictionary:
	# One continuous size-2 key; PPO means are unbounded -> clamp game-side (fly_by precedent).
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "continuous"}}

func get_obs() -> Dictionary:
	if _game == null:
		var z: Array = []
		z.resize(OBS_SIZE)
		z.fill(0.0)
		return {"obs": z}
	return {"obs": _game.get_obs_array()}

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	var raw: Array = action[ACTION_KEY]
	_tilt = Vector2(clamp_input(float(raw[0])), clamp_input(float(raw[1])))

func stored_tilt_for_test() -> Vector2:
	return _tilt

func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return
	_game.apply_tilt(_tilt, delta)
	reward += alive_reward
	# Fall = terminal: signal done so the trainer gets episode boundaries (quadruped lesson —
	# the core's reset_after timeout alone never fires when episodes end early).
	if _game.is_fallen():
		reward -= fall_penalty
		done = true
		needs_reset = true
	if needs_reset:
		needs_reset = false
		_game.reset_episode()
		reset()
		# Do NOT zero_reward(): the bridge reads reward+done together THEN zeroes (hide&seek
		# contract) — zeroing here would wipe the fall penalty before the trainer sees it.
