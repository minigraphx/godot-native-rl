class_name CrowdMath
# Pure, node-agnostic helpers for batched crowd inference. No Net, no scene references — kept
# separate from CrowdController so the gather/validate logic is unit-testable in isolation.

# Gather each agent's flat observation vector into an Array of PackedFloat32Array, in the given
# order. Each agent must implement get_obs() -> {"obs": <numeric Array/PackedFloat32Array>}.
static func gather_obs(agents: Array) -> Array:
	var inputs: Array = []
	for agent in agents:
		var obs_dict: Dictionary = agent.get_obs()
		if not ("obs" in obs_dict):
			# Deploy-path helper: fail loud but keep index alignment. An empty obs vector becomes an
			# empty input -> empty output slot -> skipped by the controller (assert would be stripped
			# in exported release builds, silently feeding a degenerate vector).
			push_error("CrowdMath.gather_obs: agent get_obs() missing 'obs' key; using empty obs for that slot.")
			inputs.append(PackedFloat32Array())
			continue
		inputs.append(PackedFloat32Array(obs_dict["obs"]))
	return inputs

# An inference output slot is usable iff it is non-empty (run_inference_batch leaves a failed
# agent's slot as an empty PackedFloat32Array).
static func output_usable(output: PackedFloat32Array) -> bool:
	return not output.is_empty()
