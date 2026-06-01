class_name RelativePositionSensor3D
extends Node3D

# Egocentric relative-position observation for a target node (3D): a unit direction in the
# sensor's local frame + a clipped, normalized distance. See RelativePositionMath.encode_3d.
# Mirrors RelativePositionSensor2D and the raycast sensors: pure math core + thin node
# wrapper, with target resolution isolated behind set_target_for_test for headless testing.

const RelativePositionMath = preload("res://addons/godot_native_rl/sensors/relative_position_math.gd")

@export var target_path: NodePath
@export var max_distance: float = 50.0

# Test seam: a target node injected directly, bypassing target_path resolution.
var _target_override: Node3D = null
var _warned_no_target := false

func set_target_for_test(node: Node3D) -> void:
	_target_override = node

func obs_size() -> int:
	return 4

func get_observation() -> Array:
	var target: Node3D = _target_override if _target_override != null else get_node_or_null(target_path) as Node3D
	if target == null:
		if not _warned_no_target:
			push_error("RelativePositionSensor3D: target_path resolves to no Node3D; returning zeros.")
			_warned_no_target = true
		var zeros := []
		zeros.resize(obs_size())
		zeros.fill(0.0)
		return zeros
	_warned_no_target = false
	# World transform when in the tree; local transform fallback when detached (unit tests).
	var sensor_xform := global_transform if is_inside_tree() else transform
	var target_pos := target.global_position if target.is_inside_tree() else target.position
	var world_offset := target_pos - sensor_xform.origin
	return RelativePositionMath.encode_3d(world_offset, sensor_xform.basis, max_distance)
