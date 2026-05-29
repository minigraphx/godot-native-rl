extends NcnnAIController2D

func get_obs() -> Dictionary:
	return {"obs": [0.0, 0.0, 1.0, 0.0, 0.5]}

func get_reward() -> float:
	return reward

func get_action_space() -> Dictionary:
	return {"move": {"size": 5, "action_type": "discrete"}}

func set_action(_action) -> void:
	pass
