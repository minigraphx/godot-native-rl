class_name ActionDecode
extends RefCounted

# Pure decoder for the deploy-side inference path. Turns a raw policy output vector into a
# godot_rl action dict by slicing the output into one contiguous segment per action_space key
# (insertion order):
#   discrete   -> argmax over the next `size` values            -> int in [0, size)
#   continuous -> the next `size` values, optionally tanh-squashed (per-key "squash": true)
#                 -> Array[float]  (godot_rl continuous convention is [-1, 1], so tanh suffices)
# The total consumed length must equal output.size(); a mismatch (train/deploy shape error) or an
# unknown action_type -> push_error + {} (the empty-dict sentinel the controller checks).
# Mirrors NcnnSync.extract_action_dict's key-walk and a standard policy head's output layout.

const InferenceMath = preload("res://addons/godot_native_rl/controllers/inference_math.gd")

static func decode_actions(output: PackedFloat32Array, action_space: Dictionary) -> Dictionary:
	var result := {}
	var index := 0
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
			result[key] = InferenceMath.argmax(segment)
		elif action_type == "continuous":
			var squash: bool = entry.get("squash", false)
			var values: Array = []
			for v in segment:
				values.append(tanh(v) if squash else v)
			result[key] = values
		else:
			push_error("ActionDecode.decode_actions: unknown action_type '%s' for key '%s'." % [action_type, key])
			return {}
		index += size
	if index != output.size():
		push_error("ActionDecode.decode_actions: output length %d exceeds action_space total %d." % [output.size(), index])
		return {}
	return result
