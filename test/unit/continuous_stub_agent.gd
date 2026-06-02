# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

var last_action = null

func get_obs() -> Dictionary:
	return {"obs": [0.0, 0.0, 0.0, 0.0]}

# Mixed space: a discrete key followed by a squashed continuous key.
func get_action_space() -> Dictionary:
	return {
		"fire": {"size": 2, "action_type": "discrete"},
		"steer": {"size": 2, "action_type": "continuous", "squash": true},
	}

func set_action(action) -> void:
	last_action = action
