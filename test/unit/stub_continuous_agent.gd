# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

func get_obs() -> Dictionary:
	return {"obs": [0.0, 0.0]}

func get_reward() -> float:
	return reward

func get_action_space() -> Dictionary:
	return {"steer": {"size": 2, "action_type": "continuous"}}

var last_action = null

func set_action(action) -> void:
	last_action = action
