class_name RoverAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd"

const ACTION_KEY := "move"
const ACTION_COUNT := 4
const GOAL_OBS_SIZE := 3
const DEFAULT_RAY_COUNT := 5
const RewardBuilderScript = preload("res://addons/godot_native_rl/reward/reward_builder.gd")
# RewardAdapterScript is inherited from the controller — do not redeclare.

@export var game_path: NodePath
@export var sensor_path: NodePath
@export var goal_bonus := 1.0
@export var step_penalty := 0.005
@export var collision_penalty := 0.25

var _game
var _sensor
var _action_index := 0

# --- Pure helpers (unit-tested) ---
func action_index_to_motion(idx: int, move_speed: float, turn_speed: float) -> Dictionary:
	match idx:
		1: return {"forward": move_speed, "yaw": 0.0}
		2: return {"forward": 0.0, "yaw": -turn_speed}
		3: return {"forward": 0.0, "yaw": turn_speed}
		_: return {"forward": 0.0, "yaw": 0.0}

func compute_goal_obs(bearing: float, dist: float, max_dist: float) -> Array:
	var norm := clampf(dist / max_dist, 0.0, 1.0) if max_dist > 0.0 else 0.0
	return [sin(bearing), cos(bearing), norm]

func compose_obs(ray_obs: Array, goal_obs: Array) -> Array:
	return ray_obs + goal_obs

func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "discrete"}}

# --- Runtime (obs composition, reward wiring, step loop) ---

func _expected_obs_size() -> int:
	var ray_count: int = _sensor.obs_size() if _sensor != null else DEFAULT_RAY_COUNT
	return ray_count + GOAL_OBS_SIZE

func _zero_obs() -> Array:
	var z: Array = []
	z.resize(_expected_obs_size())
	z.fill(0.0)
	return z

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	_sensor = get_node_or_null(sensor_path)
	if _game == null:
		push_warning("RoverAgent: game_path not set or invalid — producing zero observations.")
		return
	reward_source = RewardBuilderScript.new() \
		.add_progress_shaping(_game.distance, _game.max_distance, ["goal_reached"]) \
		.add_event_bonus("goal_reached", goal_bonus) \
		.add_event_bonus("bumped", -collision_penalty) \
		.add_step_penalty(step_penalty) \
		.build()
	var goal_adapter := RewardAdapterScript.new()
	add_child(goal_adapter)
	goal_adapter.on_signal_event(_game, "goal_reached", "goal_reached")
	var bump_adapter := RewardAdapterScript.new()
	add_child(bump_adapter)
	bump_adapter.on_signal_event(_game, "bumped", "bumped")

func get_obs() -> Dictionary:
	if _game == null or _sensor == null:
		return {"obs": _zero_obs()}
	var ray_obs: Array = _sensor.get_observation()
	var bearing: float = _game.bearing_to(_game.get_agent_pos(), _game.get_agent_yaw(), _game.get_goal_pos())
	var goal_obs := compute_goal_obs(bearing, _game.distance(), _game.max_distance())
	return {"obs": compose_obs(ray_obs, goal_obs)}

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	var idx := int(action[ACTION_KEY])
	assert(idx >= 0 and idx < ACTION_COUNT, "RoverAgent: action index %d out of range [0, %d)" % [idx, ACTION_COUNT])
	_action_index = idx

func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return
	var motion := action_index_to_motion(_action_index, _game.move_speed, _game.turn_speed)
	_game.move_agent(motion["forward"], motion["yaw"], delta)
	# Accumulate reward against the CURRENT goal BEFORE relocating (matches the chase pattern).
	accumulate_reward()
	if _game.distance() < _game.goal_radius:
		_game.relocate_goal()
	if needs_reset:
		needs_reset = false
		_game.reset_positions()
		reset()
		zero_reward()
		if reward_source != null:
			reward_source.reset()
