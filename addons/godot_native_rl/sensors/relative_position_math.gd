class_name RelativePositionMath
extends RefCounted

# Pure, stateless helpers for relative-position sensors. No physics, no node state — fully
# unit-testable headlessly. Mirrors godot_rl's PositionSensor2D/3D encoding exactly.
#
# Each target contributes an EGOCENTRIC slot (world offset rotated into the sensor's local
# frame), in one of two modes:
#   use_separate_direction = false (DEFAULT): the normalized, clamped offset components
#     scaled = local.limit_length(max_distance) / max_distance   (no separate distance scalar)
#   use_separate_direction = true: the unit direction components, then a clamped normalized
#     distance scalar appended last.
# Per-axis include_x/y(/z) toggles drop axes from the output. Guards: max_distance <= 0 -> the
# slot is all zeros; a zero offset -> zeros.

# Number of floats one target contributes for a given config.
static func per_target_size(use_separate_direction: bool, include_x: bool, include_y: bool, include_z: bool = false) -> int:
	var n := 0
	if include_x:
		n += 1
	if include_y:
		n += 1
	if include_z:
		n += 1
	if use_separate_direction:
		n += 1
	return n

static func _zeros(n: int) -> Array:
	var out: Array = []
	out.resize(n)
	out.fill(0.0)
	return out

# world_offset: target_pos - sensor_pos, in world space.
# sensor_rotation: the sensor node's world rotation (radians).
static func encode_2d(world_offset: Vector2, sensor_rotation: float, max_distance: float, use_separate_direction: bool, include_x: bool, include_y: bool) -> Array:
	if max_distance <= 0.0:
		return _zeros(per_target_size(use_separate_direction, include_x, include_y, false))
	var local := world_offset.rotated(-sensor_rotation)
	var out: Array = []
	if use_separate_direction:
		var dir := local.normalized()
		var dist := minf(local.length() / max_distance, 1.0)
		if include_x:
			out.append(dir.x)
		if include_y:
			out.append(dir.y)
		out.append(dist)
	else:
		var scaled := local.limit_length(max_distance) / max_distance
		if include_x:
			out.append(scaled.x)
		if include_y:
			out.append(scaled.y)
	return out

# world_offset: target_pos - sensor_pos, in world space.
# sensor_basis: the sensor node's world-transform basis.
static func encode_3d(world_offset: Vector3, sensor_basis: Basis, max_distance: float, use_separate_direction: bool, include_x: bool, include_y: bool, include_z: bool) -> Array:
	if max_distance <= 0.0:
		return _zeros(per_target_size(use_separate_direction, include_x, include_y, include_z))
	var local := sensor_basis.inverse() * world_offset
	var out: Array = []
	if use_separate_direction:
		var dir := local.normalized()
		var dist := minf(local.length() / max_distance, 1.0)
		if include_x:
			out.append(dir.x)
		if include_y:
			out.append(dir.y)
		if include_z:
			out.append(dir.z)
		out.append(dist)
	else:
		var scaled := local.limit_length(max_distance) / max_distance
		if include_x:
			out.append(scaled.x)
		if include_y:
			out.append(scaled.y)
		if include_z:
			out.append(scaled.z)
	return out
