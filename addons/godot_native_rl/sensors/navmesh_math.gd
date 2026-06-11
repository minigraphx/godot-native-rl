extends RefCounted
# Pure NavMesh sensor math (#20): turns a NavigationServer path into a flat observation — navigable
# **closeness** (path length along the polyline, not straight-line) plus the unit direction to the
# next waypoint. No engine query here (NavMeshSensor2D/3D inject the path), so it's headless-unit-
# testable. A NavigationServer path is [origin, waypoint_1, ..., target]; an empty/size-1 path means
# unreachable and encodes to zeros.

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

## Encode a 2D path to [closeness, dir_x, dir_y]. `path` is the NavigationServer polyline; `from` is
## the agent position (== path[0] in practice). Unreachable (path < 2 points) → [0, 0, 0].
static func encode_2d(from: Vector2, path: Array, max_distance: float) -> Array:
	if path.size() < 2:
		return [0.0, 0.0, 0.0]
	var dir: Vector2 = Vector2(path[1]) - from
	dir = dir.normalized() if dir.length() > 0.0 else Vector2.ZERO
	return [closeness(path_length(path), max_distance), dir.x, dir.y]

## Encode a 3D path to [closeness, dir_x, dir_y, dir_z]. Unreachable → [0, 0, 0, 0].
static func encode_3d(from: Vector3, path: Array, max_distance: float) -> Array:
	if path.size() < 2:
		return [0.0, 0.0, 0.0, 0.0]
	var dir: Vector3 = Vector3(path[1]) - from
	dir = dir.normalized() if dir.length() > 0.0 else Vector3.ZERO
	return [closeness(path_length(path), max_distance), dir.x, dir.y, dir.z]
