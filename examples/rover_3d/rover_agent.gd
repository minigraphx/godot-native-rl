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
