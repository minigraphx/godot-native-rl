extends RefCounted
# Pure NavMesh sensor math (#20): turns a NavigationServer path into a flat observation — navigable
# **closeness** (path length along the polyline, not straight-line) plus the unit direction to the
# next waypoint. No engine query here (NavMeshSensor2D/3D inject the path), so it's headless-unit-
# testable. A NavigationServer path is [origin, waypoint_1, ..., target]; an empty/size-1 path (no
# map / no regions) encodes to zeros. The direction can be rotated into the sensor's local frame
# (egocentric). is_reachable() distinguishes a true reach from a partial path to the closest point.

## Total length along the polyline of points (each must support distance_to); 0 for < 2 points.
static func path_length(points: Array) -> float:
	var total := 0.0
	for i in range(1, points.size()):
		total += points[i - 1].distance_to(points[i])
	return total

## godot_rl-style closeness: 1 at the target, linearly →0 at/over max_distance. max_distance <= 0 → 0.
static func closeness(path_len: float, max_distance: float) -> float:
	if max_distance <= 0.0:
		return 0.0
	return clampf(1.0 - path_len / max_distance, 0.0, 1.0)

## Encode a 2D path to [closeness, dir_x, dir_y]. `path` is the NavigationServer polyline; `from`
## is the agent position (== path[0] in practice). `sensor_rotation` (radians) rotates the
## direction into the sensor's local frame (egocentric, like RelativePositionSensor); 0 = world
## frame. Unreachable (path < 2 points) → [0, 0, 0].
static func encode_2d(from: Vector2, path: Array, max_distance: float, sensor_rotation: float = 0.0) -> Array:
	if path.size() < 2:
		return [0.0, 0.0, 0.0]
	var dir: Vector2 = Vector2(path[1]) - from
	dir = dir.normalized() if dir.length() > 0.0 else Vector2.ZERO
	if sensor_rotation != 0.0:
		dir = dir.rotated(-sensor_rotation)
	return [closeness(path_length(path), max_distance), dir.x, dir.y]

## Encode a 3D path to [closeness, dir_x, dir_y, dir_z]. `sensor_basis` rotates the direction into
## the sensor's local frame (egocentric); IDENTITY = world frame. Unreachable → [0, 0, 0, 0].
static func encode_3d(from: Vector3, path: Array, max_distance: float, sensor_basis: Basis = Basis.IDENTITY) -> Array:
	if path.size() < 2:
		return [0.0, 0.0, 0.0, 0.0]
	var dir: Vector3 = Vector3(path[1]) - from
	dir = dir.normalized() if dir.length() > 0.0 else Vector3.ZERO
	dir = sensor_basis.inverse() * dir  # world direction -> sensor-local (identity = world frame)
	return [closeness(path_length(path), max_distance), dir.x, dir.y, dir.z]

## Whether the path actually reaches `to`: NavigationServer returns a *partial* path to the closest
## reachable point when the target is on a disconnected island, so a short path whose last point is
## far from `to` means "walled off", not "near". True when the last point is within `tolerance`.
static func is_reachable(path: Array, to, tolerance: float) -> bool:
	return path.size() >= 2 and path[path.size() - 1].distance_to(to) <= tolerance
