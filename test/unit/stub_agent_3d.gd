extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd"

func get_obs() -> Dictionary:
	return {"obs": [0.0, 1.0, 2.0]}

func get_reward() -> float:
	return reward

func get_action_space() -> Dictionary:
	return {"move": {"size": 3, "action_type": "discrete"}}

func set_action(_action) -> void:
	pass
