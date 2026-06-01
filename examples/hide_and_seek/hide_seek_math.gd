class_name HideSeekMath
extends RefCounted

# Pure, stateless helpers for the 2D hide & seek example: analytic segment/ray vs
# axis-aligned-rect geometry (no physics world, so the whole obs path is headless-unit-testable),
# plus observation assembly and the role-signed step reward. Walls are Array[Rect2] in game-local
# coordinates (tile-offset-safe for ParallelArena2D).

const _EPS := 1e-9

# Segment a->b vs an axis-aligned rect (Liang-Barsky slab clip). True if they touch/overlap,
# including when an endpoint is inside the rect.
static func segment_intersects_rect(a: Vector2, b: Vector2, rect: Rect2) -> bool:
	var d := b - a
	var tmin := 0.0
	var tmax := 1.0
	for axis in range(2):
		var da: float = d[axis]
		var a_axis: float = a[axis]
		var lo: float = rect.position[axis]
		var hi: float = rect.position[axis] + rect.size[axis]
		if absf(da) < _EPS:
			if a_axis < lo or a_axis > hi:
				return false
		else:
			var t1 := (lo - a_axis) / da
			var t2 := (hi - a_axis) / da
			if t1 > t2:
				var tmp := t1
				t1 = t2
				t2 = tmp
			tmin = maxf(tmin, t1)
			tmax = minf(tmax, t2)
			if tmin > tmax:
				return false
	return true

# True if the segment a->b crosses ANY wall (i.e. line of sight is blocked).
static func segment_blocked(a: Vector2, b: Vector2, walls: Array) -> bool:
	for rect in walls:
		if segment_intersects_rect(a, b, rect):
			return true
	return false

static func point_in_walls(p: Vector2, walls: Array) -> bool:
	for rect in walls:
		if (rect as Rect2).has_point(p):
			return true
	return false

# Nearest hit distance of a unit-direction ray from origin against a rect within max_dist,
# or -1.0 on a miss. Origin inside the rect returns 0.0.
static func ray_rect_distance(origin: Vector2, dir: Vector2, max_dist: float, rect: Rect2) -> float:
	var tmin := 0.0
	var tmax := max_dist
	for axis in range(2):
		var dd: float = dir[axis]
		var o: float = origin[axis]
		var lo: float = rect.position[axis]
		var hi: float = rect.position[axis] + rect.size[axis]
		if absf(dd) < _EPS:
			if o < lo or o > hi:
				return -1.0
		else:
			var t1 := (lo - o) / dd
			var t2 := (hi - o) / dd
			if t1 > t2:
				var tmp := t1
				t1 = t2
				t2 = tmp
			tmin = maxf(tmin, t1)
			tmax = minf(tmax, t2)
			if tmin > tmax:
				return -1.0
	return tmin
