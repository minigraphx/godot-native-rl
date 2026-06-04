class_name RelativePositionSensor3D
extends "res://addons/godot_native_rl/sensors/i_sensor_3d.gd"

# 3D egocentric relative-position observation for a set of target nodes, matching godot_rl's
# PositionSensor3D. Each target in `objects_to_observe` contributes a slot (see
# RelativePositionMath): the normalized clamped offset (default) or a unit direction (local frame,
# forward = -Z) + distance (use_separate_direction). Per-axis include_x/y/z toggles drop axes.
# obs_size() is fixed by config + target count; a freed/invalid target zero-fills its slot.

const RelativePositionMath = preload("res://addons/godot_native_rl/sensors/relative_position_math.gd")

## Targets to observe, in order; each contributes a slot. Freed/invalid entries zero-fill.
@export var objects_to_observe: Array[Node3D]
## Include the relative x component in each slot.
@export var include_x: bool = true
## Include the relative y component in each slot.
@export var include_y: bool = true
## Include the relative z component in each slot.
@export var include_z: bool = true
## Distance normalizer. Obs values are normalized so 0 is closest and 1 is at/over this distance.
@export_range(0.01, 2500.0) var max_distance: float = 1.0
## false: emit the normalized clamped offset. true: emit unit direction + a distance scalar.
@export var use_separate_direction: bool = false

var _warned_invalid := false

func obs_size() -> int:
	return objects_to_observe.size() * RelativePositionMath.per_target_size(use_separate_direction, include_x, include_y, include_z)

func get_observation() -> Array:
	var sensor_xform := global_transform if is_inside_tree() else transform
	var out: Array = []
	var any_invalid := false
	for obj in objects_to_observe:
		var world_offset := Vector3.ZERO
		if is_instance_valid(obj):
			var target_pos := obj.global_position if obj.is_inside_tree() else obj.position
			world_offset = target_pos - sensor_xform.origin
		else:
			any_invalid = true
		out.append_array(RelativePositionMath.encode_3d(world_offset, sensor_xform.basis, max_distance, use_separate_direction, include_x, include_y, include_z))
	if any_invalid and not _warned_invalid:
		push_error("RelativePositionSensor3D: one or more objects_to_observe are invalid; their slots are zero-filled.")
		_warned_invalid = true
	elif not any_invalid:
		_warned_invalid = false
	return out
