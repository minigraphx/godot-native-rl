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

const RaycastMath = preload("res://addons/godot_native_rl/sensors/raycast_math.gd")

# N unit directions evenly spaced over the full circle (no duplicated endpoint), starting at +X.
static func ray_directions_surround(n: int) -> Array:
	var dirs := []
	if n < 1:
		return dirs
	for i in range(n):
		dirs.append(Vector2.from_angle(TAU * float(i) / float(n)))
	return dirs

# Per-ray "closeness" of the nearest wall along each direction (miss -> 0.0, near -> ~1.0).
static func wall_ray_closeness(origin: Vector2, dirs: Array, max_dist: float, walls: Array) -> Array:
	var out := []
	for dir in dirs:
		var best := -1.0
		for rect in walls:
			var d := ray_rect_distance(origin, dir, max_dist, rect)
			if d >= 0.0 and (best < 0.0 or d < best):
				best = d
		out.append(RaycastMath.closeness(best, max_dist) if best >= 0.0 else 0.0)
	return out

# Line-of-sight-gated opponent encoding: [dir_x, dir_y, dist_norm, visible].
# Occluded (a wall on the segment) -> [0, 0, 0, 0].
static func encode_opponent(self_pos: Vector2, opp_pos: Vector2, walls: Array, max_dist: float) -> Array:
	if segment_blocked(self_pos, opp_pos, walls):
		return [0.0, 0.0, 0.0, 0.0]
	var offset := opp_pos - self_pos
	var dist := offset.length()
	var dir := offset.normalized() if dist > 0.0 else Vector2.ZERO
	var dist_norm := clampf(dist / max_dist, 0.0, 1.0) if max_dist > 0.0 else 0.0
	return [dir.x, dir.y, dist_norm, 1.0]

static func role_flag(is_seeker: bool) -> float:
	return 1.0 if is_seeker else 0.0

# Own position normalized to [-1, 1] per axis (center of arena -> [0, 0]).
static func own_pos_obs(pos: Vector2, arena_size: Vector2) -> Array:
	var x := (pos.x / arena_size.x - 0.5) * 2.0 if arena_size.x > 0.0 else 0.0
	var y := (pos.y / arena_size.y - 0.5) * 2.0 if arena_size.y > 0.0 else 0.0
	return [x, y]

# Role-signed reward: +1 (seeker) / -1 (hider) per step when the seeker has LOS to the hider,
# reversed when blocked; plus a role-signed catch bonus on the frame of capture.
static func step_reward(is_seeker: bool, has_los: bool, caught: bool, catch_bonus: float) -> float:
	var s := 1.0 if is_seeker else -1.0
	var r := s * (1.0 if has_los else -1.0)
	if caught:
		r += s * catch_bonus
	return r

static func assemble_obs(own_obs: Array, wall_obs: Array, opp_obs: Array, role: float) -> Array:
	return own_obs + wall_obs + opp_obs + [role]
