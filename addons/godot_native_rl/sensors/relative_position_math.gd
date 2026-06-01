class_name RelativePositionMath
extends RefCounted

# Pure, stateless helpers for relative-position sensors. No physics, no node state — fully
# unit-testable headlessly. Output is an EGOCENTRIC unit direction (the offset rotated into
# the sensor's local frame, then normalized) followed by a clipped, normalized distance:
# dist_norm = clamp(offset_length / max_distance, 0, 1). The direction is unit-length so
# bearing and distance are decoupled signals. Guards: a zero offset -> zero direction; a
# non-positive max_distance -> dist_norm 0.

static func _dist_norm(offset_length: float, max_distance: float) -> float:
	if max_distance <= 0.0:
		return 0.0
	return clampf(offset_length / max_distance, 0.0, 1.0)

# world_offset: target_pos - sensor_pos, in world space.
# sensor_rotation: the sensor node's world rotation (radians).
# Returns [dir_x, dir_y, dist_norm].
static func encode_2d(world_offset: Vector2, sensor_rotation: float, max_distance: float) -> Array:
	var local := world_offset.rotated(-sensor_rotation)
	var dir := local.normalized()  # Vector2.ZERO when local is zero-length
	return [dir.x, dir.y, _dist_norm(world_offset.length(), max_distance)]
