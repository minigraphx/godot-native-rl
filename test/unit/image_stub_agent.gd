# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

var image_to_return: Image = null
var last_action = null

func get_inference_image() -> Image:
	return image_to_return

func get_obs() -> Dictionary:
	return {"obs": [0.0]}

func get_action_space() -> Dictionary:
	return {"move": {"size": 4, "action_type": "discrete"}}

func set_action(action) -> void:
	last_action = action
