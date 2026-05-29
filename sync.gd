class_name NcnnSync
extends Node

enum ControlModes { HUMAN, TRAINING }

var agents_training: Array[Node] = []
var _action_space: Dictionary = {}
var _obs_space: Dictionary = {}

# --- Pure message builders (unit-tested) ---

func build_env_info_message() -> Dictionary:
	return {
		"type": "env_info",
		"observation_space": _obs_space,
		"action_space": _action_space,
		"n_agents": agents_training.size(),
	}

func build_step_message(obs: Array, reward: Array, done: Array) -> Dictionary:
	return {"type": "step", "obs": obs, "reward": reward, "done": done}

func build_reset_message(obs: Array) -> Dictionary:
	return {"type": "reset", "obs": obs}

func extract_action_dict(action_array: Array, action_space: Dictionary) -> Dictionary:
	var index := 0
	var result := {}
	for key in action_space.keys():
		var size: int = action_space[key]["size"]
		if action_space[key]["action_type"] == "discrete":
			result[key] = roundi(action_array[index])
		else:
			result[key] = action_array.slice(index, index + size)
		index += size
	return result
