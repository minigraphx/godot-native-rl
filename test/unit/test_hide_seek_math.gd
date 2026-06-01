extends SceneTree

const Harness = preload("res://test/harness.gd")
const HideSeekMath = preload("res://examples/hide_and_seek/hide_seek_math.gd")

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

	h.finish(self)
