class_name RelativePositionSensor2D
extends "res://addons/godot_native_rl/sensors/i_sensor_2d.gd"

# Egocentric relative-position observation for a set of target nodes, matching godot_rl's
# PositionSensor2D. Each target in `objects_to_observe` contributes a slot (see
# RelativePositionMath): the normalized clamped offset (default) or a unit direction + distance
# (use_separate_direction). Per-axis include_x/y toggles drop axes. obs_size() is fixed by the
# config and target count, so the policy input width is stable; a freed/invalid target zero-fills
# its slot rather than shrinking the vector.

const RelativePositionMath = preload("res://addons/godot_native_rl/sensors/relative_position_math.gd")

## Targets to observe, in order; each contributes a slot. Freed/invalid entries zero-fill.
@export var objects_to_observe: Array[Node2D]
## Include the relative x component in each slot.
@export var include_x: bool = true
## Include the relative y component in each slot.
@export var include_y: bool = true
## Distance normalizer. Obs values are normalized so 0 is closest and 1 is at/over this distance.
@export_range(0.01, 20000.0) var max_distance: float = 1.0
## false: emit the normalized clamped offset. true: emit unit direction + a distance scalar.
@export var use_separate_direction: bool = false

var _warned_invalid := false

func obs_size() -> int:
	return objects_to_observe.size() * RelativePositionMath.per_target_size(use_separate_direction, include_x, include_y, false)

func get_observation() -> Array:
	# World transform when in the tree; local transform fallback when detached (unit tests).
	var sensor_xform := global_transform if is_inside_tree() else transform
	var sensor_rotation := sensor_xform.get_rotation()
	var out: Array = []
	var any_invalid := false
	for obj in objects_to_observe:
		var world_offset := Vector2.ZERO
		if is_instance_valid(obj):
			var target_pos := obj.global_position if obj.is_inside_tree() else obj.position
			world_offset = target_pos - sensor_xform.origin
		else:
			any_invalid = true
		out.append_array(RelativePositionMath.encode_2d(world_offset, sensor_rotation, max_distance, use_separate_direction, include_x, include_y))
	if any_invalid and not _warned_invalid:
		push_error("RelativePositionSensor2D: one or more objects_to_observe are invalid; their slots are zero-filled.")
		_warned_invalid = true
	elif not any_invalid:
		_warned_invalid = false
	return out
