class_name ActionDecode
extends RefCounted

# Pure decoder for the deploy-side inference path. Turns a raw policy output vector into a
# godot_rl action dict by slicing the output into one contiguous segment per action_space key
# (insertion order):
#   discrete   -> argmax over the next `size` values            -> int in [0, size)
#   discrete (stochastic) -> sample from softmax(values) via rng when deterministic=false
#   continuous -> the next `size` values, optionally tanh-squashed (per-key "squash": true)
#   continuous (stochastic) -> mean + std·N(0,1) via rng when deterministic=false and an
#                 action_dist {"std": [...]} sidecar is supplied (PPO DiagGaussian); std is applied
#                 positionally across the continuous dims, then the optional tanh squash.
#                 -> Array[float]  (godot_rl continuous convention is [-1, 1], so tanh suffices)
# The total consumed length must equal output.size(); a mismatch (train/deploy shape error) or an
# unknown action_type -> push_error + {} (the empty-dict sentinel the controller checks).
# Mirrors NcnnSync.extract_action_dict's key-walk and a standard policy head's output layout.

const InferenceMath = preload("res://addons/godot_native_rl/controllers/inference_math.gd")

static func decode_actions(output: PackedFloat32Array, action_space: Dictionary, deterministic: bool = true, rng: RandomNumberGenerator = null, action_dist: Dictionary = {}) -> Dictionary:
	var result := {}
	var index := 0
	var cont_index := 0  # advances per continuous value consumed; indexes action_dist["std"] positionally
	for key in action_space.keys():
		var entry: Dictionary = action_space[key]
		var size: int = entry["size"]
		var action_type: String = entry["action_type"]
		if size <= 0:
			push_error("ActionDecode.decode_actions: action key '%s' has non-positive size %d." % [key, size])
			return {}
		if index + size > output.size():
			push_error("ActionDecode.decode_actions: output too short for key '%s' (need %d at offset %d, have %d)." % [key, size, index, output.size()])
			return {}
		var segment: PackedFloat32Array = output.slice(index, index + size)
		if action_type == "discrete":
			if deterministic:
				result[key] = InferenceMath.argmax(segment)
			else:
				var probs := InferenceMath.softmax(segment)
				# rng=null falls back to Godot's global RNG; pass an explicit RNG for reproducible eval.
				var u: float = rng.randf() if rng != null else randf()
				result[key] = InferenceMath.sample_categorical(probs, u)
		elif action_type == "continuous":
			var squash: bool = entry.get("squash", false)
			# Stochastic continuous (PPO DiagGaussian): sample mean + std·N(0,1), then optional tanh.
			# Only when non-deterministic AND a std sidecar is present; else the mean (unchanged).
			var has_std: bool = (not deterministic) and action_dist.has("std")
			var std: PackedFloat32Array = action_dist["std"] if has_std else PackedFloat32Array()
			var values: Array = []
			for v in segment:
				var x: float = v
				if has_std and cont_index < std.size():
					# rng=null falls back to Godot's global RNG (mirrors the discrete branch).
					var z: float = rng.randfn(0.0, 1.0) if rng != null else randfn(0.0, 1.0)
					x = v + std[cont_index] * z
				values.append(tanh(x) if squash else x)
				cont_index += 1
			result[key] = values
		else:
			push_error("ActionDecode.decode_actions: unknown action_type '%s' for key '%s'." % [action_type, key])
			return {}
		index += size
	if index != output.size():
		push_error("ActionDecode.decode_actions: output length %d exceeds action_space total %d." % [output.size(), index])
		return {}
	return result
