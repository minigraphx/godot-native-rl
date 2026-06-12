class_name GridWorldAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

const ACTION_KEY := "move"
const ACTION_COUNT := 5  # stay + 4 directions (chase-matching shape)
const GOAL_OBS_SIZE := 2

@export var game_path: NodePath
@export var grid_sensor_path: NodePath  ## GridSensor2D centered on the agent (the #48 showcase)
@export var goal_reward := 1.0
@export var pit_penalty := 1.0
@export var step_penalty := 0.01

var _game
var _sensor
var _action_index := 0

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	_sensor = get_node_or_null(grid_sensor_path)
	if _game == null:
		push_warning("GridWorldAgent: game_path not set — producing zero observations.")

func _expected_obs_size() -> int:
	var grid: int = _sensor.obs_size() if _sensor != null else 0
	return grid + GOAL_OBS_SIZE

func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "discrete"}}

func get_obs() -> Dictionary:
	if _game == null:
		return {"obs": [0.0, 0.0]}
	var grid_obs: Array = _sensor.get_observation() if _sensor != null else []
	return {"obs": grid_obs + _game.goal_obs()}

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	var idx := int(action[ACTION_KEY])
	assert(idx >= 0 and idx < ACTION_COUNT, "GridWorldAgent: action %d out of range" % idx)
	_action_index = idx

func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return
	# One delivered action = one cell step: apply then consume, so the remaining frames of the
	# action_repeat window are no-ops instead of repeating the move.
	if _action_index != 0:
		_game.move_agent(_action_index)
		_action_index = 0
	reward -= step_penalty
	if _game.at_goal():
		reward += goal_reward
	elif _game.at_pit():
		reward -= pit_penalty
	if _game.resolve_terminal():
		done = true
		needs_reset = true
	if needs_reset:
		needs_reset = false
		reset()
		# Do NOT zero_reward() here: the bridge reads reward+done together THEN zeroes (the
		# hide&seek-documented contract). Zeroing now would wipe the terminal +-1 before the
		# trainer ever sees it — the main learning signal of this env.
