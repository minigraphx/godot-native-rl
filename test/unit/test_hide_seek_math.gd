extends SceneTree

const Harness = preload("res://test/harness.gd")
const HideSeekMath = preload("res://examples/hide_and_seek/hide_seek_math.gd")

func _approx(h: Harness, out: Array, expected: Array, label: String) -> void:
	var ok := out.size() == expected.size()
	for i in range(mini(out.size(), expected.size())):
		if absf(float(out[i]) - float(expected[i])) > 1e-5:
			ok = false
	h.assert_true(ok, "%s (got %s, want %s)" % [label, str(out), str(expected)])

func _initialize() -> void:
	var h := Harness.new()
	var wall := Rect2(40, 40, 20, 20)  # x:[40,60], y:[40,60]

	# --- segment vs rect / segment_blocked ---
	# Horizontal segment passing straight through the wall.
	h.assert_true(HideSeekMath.segment_intersects_rect(Vector2(0, 50), Vector2(100, 50), wall), "segment through rect intersects")
	# Segment well above the wall — clear.
	h.assert_true(not HideSeekMath.segment_intersects_rect(Vector2(0, 10), Vector2(100, 10), wall), "segment above rect clears")
	# A wall on the line of sight blocks; a wall to the side does not.
	h.assert_true(HideSeekMath.segment_blocked(Vector2(0, 50), Vector2(100, 50), [wall]), "wall on segment blocks LOS")
	h.assert_true(not HideSeekMath.segment_blocked(Vector2(0, 10), Vector2(100, 10), [wall]), "wall beside segment does not block")
	h.assert_true(not HideSeekMath.segment_blocked(Vector2(0, 50), Vector2(100, 50), []), "no walls -> never blocked")

	# --- point_in_walls ---
	h.assert_true(HideSeekMath.point_in_walls(Vector2(50, 50), [wall]), "point inside wall")
	h.assert_true(not HideSeekMath.point_in_walls(Vector2(0, 0), [wall]), "point outside wall")

	# --- ray vs rect distance (dir is unit; returns nearest hit distance or -1) ---
	# Ray from origin (0,50) heading +X hits the wall's near face at x=40 -> distance 40.
	h.assert_true(absf(HideSeekMath.ray_rect_distance(Vector2(0, 50), Vector2(1, 0), 100.0, wall) - 40.0) < 1e-4, "ray hits near face at 40")
	# Ray heading -X (away) never hits -> -1.
	h.assert_true(HideSeekMath.ray_rect_distance(Vector2(0, 50), Vector2(-1, 0), 100.0, wall) < 0.0, "ray away misses (-1)")
	# Wall beyond max_dist -> miss.
	h.assert_true(HideSeekMath.ray_rect_distance(Vector2(0, 50), Vector2(1, 0), 30.0, wall) < 0.0, "wall beyond max_dist misses")
	# Ray origin inside the rect -> distance 0 (documented contract).
	h.assert_eq(HideSeekMath.ray_rect_distance(Vector2(50, 50), Vector2(1, 0), 100.0, wall), 0.0, "ray origin inside rect -> 0")
	# Diagonal ray exercises both slab axes: from (0,0) heading +45deg hits the wall's near corner region.
	h.assert_true(HideSeekMath.ray_rect_distance(Vector2(0, 0), Vector2(1, 1).normalized(), 200.0, wall) > 0.0, "diagonal ray hits the wall")

	# --- surround ray directions ---
	var dirs: Array = HideSeekMath.ray_directions_surround(4)
	h.assert_eq(dirs.size(), 4, "surround makes 4 dirs")
	h.assert_true(absf((dirs[0] as Vector2).angle() - 0.0) < 1e-5, "first surround dir points +X")

	# --- wall_ray_closeness: a ray straight at a near wall reads high closeness; clear rays read 0 ---
	var blk := Rect2(40, 40, 20, 20)
	var close: Array = HideSeekMath.wall_ray_closeness(Vector2(0, 50), [Vector2(1, 0)], 100.0, [blk])
	h.assert_true(close[0] > 0.5 and close[0] <= 1.0, "ray at near wall -> high closeness")
	var far: Array = HideSeekMath.wall_ray_closeness(Vector2(0, 50), [Vector2(-1, 0)], 100.0, [blk])
	h.assert_eq(far[0], 0.0, "ray away from wall -> 0 closeness")

	# --- encode_opponent: visible -> dir+dist+1; occluded -> zeros ---
	var vis: Array = HideSeekMath.encode_opponent(Vector2(0, 0), Vector2(100, 0), [], 200.0)
	_approx(h, vis, [1.0, 0.0, 0.5, 1.0], "opponent visible -> [dir, dist_norm, 1]")
	var occ: Array = HideSeekMath.encode_opponent(Vector2(0, 50), Vector2(100, 50), [blk], 200.0)
	_approx(h, occ, [0.0, 0.0, 0.0, 0.0], "opponent occluded -> zeros")

	# --- role flag + own position obs ---
	h.assert_eq(HideSeekMath.role_flag(true), 1.0, "seeker role flag 1")
	h.assert_eq(HideSeekMath.role_flag(false), 0.0, "hider role flag 0")
	_approx(h, HideSeekMath.own_pos_obs(Vector2(500, 300), Vector2(1000, 600)), [0.0, 0.0], "center -> [0,0]")

	# --- step_reward: sign flips by role; catch adds bonus only when caught ---
	h.assert_eq(HideSeekMath.step_reward(true, true, false, 5.0), 1.0, "seeker sees -> +1")
	h.assert_eq(HideSeekMath.step_reward(true, false, false, 5.0), -1.0, "seeker blind -> -1")
	h.assert_eq(HideSeekMath.step_reward(false, true, false, 5.0), -1.0, "hider seen -> -1")
	h.assert_eq(HideSeekMath.step_reward(false, false, false, 5.0), 1.0, "hider hidden -> +1")
	h.assert_eq(HideSeekMath.step_reward(true, true, true, 5.0), 6.0, "seeker catch -> +1+bonus")
	h.assert_eq(HideSeekMath.step_reward(false, true, true, 5.0), -6.0, "hider caught -> -1-bonus")

	# --- assemble_obs concatenates own + wall + opp + [role] in order ---
	var obs: Array = HideSeekMath.assemble_obs([0.1, 0.2], [0.3, 0.4], [0.5, 0.6, 0.7, 1.0], 1.0)
	h.assert_eq(obs.size(), 9, "assembled obs length = 2+2+4+1")
	h.assert_eq(float(obs[8]), 1.0, "role flag is last")

	h.finish(self)
