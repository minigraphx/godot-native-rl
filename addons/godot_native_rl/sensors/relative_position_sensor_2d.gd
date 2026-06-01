class_name RelativePositionSensor2D
extends Node2D

# Egocentric relative-position observation for a target node: a unit direction (in the
# sensor's local frame) + a clipped, normalized distance. See RelativePositionMath.encode_2d.
# Mirrors the raycast sensors: pure math core + thin node wrapper, with target resolution
# isolated behind set_target_for_test so the full observation path is headless-testable.
# Composition into an agent's get_obs() is manual: call get_observation() and concatenate;
# obs_size() declares the contributed size.

const RelativePositionMath = preload("res://addons/godot_native_rl/sensors/relative_position_math.gd")

@export var target_path: NodePath
@export var max_distance: float = 1000.0

# Test seam: a target node injected directly, bypassing target_path resolution (which needs
# tree membership). When null, target_path is resolved via get_node_or_null.
var _target_override: Node2D = null
var _warned_no_target := false

func set_target_for_test(node: Node2D) -> void:
	_target_override = node

func obs_size() -> int:
	return 3

func get_observation() -> Array:
	var target: Node2D = _target_override if _target_override != null else get_node_or_null(target_path) as Node2D
	if target == null:
		if not _warned_no_target:
			push_error("RelativePositionSensor2D: target_path resolves to no Node2D; returning zeros.")
			_warned_no_target = true
		var zeros := []
		zeros.resize(obs_size())
		zeros.fill(0.0)
		return zeros
	_warned_no_target = false
	# Use the world transform when in the tree; fall back to the local transform when detached
	# (e.g. unit tests) so the path resolves without tree-dependent global_position errors.
	var sensor_xform := global_transform if is_inside_tree() else transform
	var target_pos := target.global_position if target.is_inside_tree() else target.position
	var world_offset := target_pos - sensor_xform.origin
	return RelativePositionMath.encode_2d(world_offset, sensor_xform.get_rotation(), max_distance)
