extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"
# Minimal agent for recurrent controller tests: 5-wide obs, single discrete "move" of size 4.

var obs_to_return := PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0])
var last_action = null

func get_obs() -> Dictionary:
	return {"obs": obs_to_return}

func get_action_space() -> Dictionary:
	return {"move": {"size": 4, "action_type": "discrete"}}

func set_action(action) -> void:
	last_action = action

func get_reward() -> float:
	return 0.0
