extends SceneTree

const Harness = preload("res://test/harness.gd")
const RaycastMath = preload("res://sensors/raycast_math.gd")

func _initialize() -> void:
	var h := Harness.new()

	# --- closeness ---
	h.assert_eq(RaycastMath.closeness(-1.0, 100.0), 0.0, "miss (negative distance) -> 0")
	h.assert_eq(RaycastMath.closeness(0.0, 100.0), 1.0, "hit at origin -> 1")
	h.assert_eq(RaycastMath.closeness(50.0, 100.0), 0.5, "half distance -> 0.5")
	h.assert_eq(RaycastMath.closeness(100.0, 100.0), 0.0, "hit at max range -> 0")
	h.assert_eq(RaycastMath.closeness(200.0, 100.0), 0.0, "beyond range clamps to 0")
	h.assert_eq(RaycastMath.closeness(50.0, 0.0), 0.0, "zero ray_length guard -> 0")
	h.assert_eq(RaycastMath.closeness(50.0, -5.0), 0.0, "negative ray_length guard -> 0")

	# --- ray_directions_2d ---
	h.assert_eq(RaycastMath.ray_directions_2d(0, 90.0, 0.0).size(), 0, "n_rays 0 -> empty")
	h.assert_eq(RaycastMath.ray_directions_2d(-3, 90.0, 0.0).size(), 0, "n_rays negative -> empty")

	var single: Array = RaycastMath.ray_directions_2d(1, 90.0, 0.0)
	h.assert_eq(single.size(), 1, "single ray -> 1 dir")
	h.assert_true((single[0] - Vector2(1.0, 0.0)).length() < 1e-5, "single ray at forward 0 points +X")

	var fan: Array = RaycastMath.ray_directions_2d(3, 90.0, 0.0)
	h.assert_eq(fan.size(), 3, "n_rays 3 -> 3 dirs")
	h.assert_true((fan[0] - Vector2.from_angle(-PI / 4.0)).length() < 1e-5, "fan start at forward - cone/2")
	h.assert_true((fan[1] - Vector2(1.0, 0.0)).length() < 1e-5, "fan middle at forward")
	h.assert_true((fan[2] - Vector2.from_angle(PI / 4.0)).length() < 1e-5, "fan end at forward + cone/2")

	var rotated: Array = RaycastMath.ray_directions_2d(1, 90.0, PI / 2.0)
	h.assert_true((rotated[0] - Vector2(0.0, 1.0)).length() < 1e-5, "forward PI/2 points +Y")

	var unit_ok := true
	for d in RaycastMath.ray_directions_2d(7, 120.0, 0.3):
		if absf(d.length() - 1.0) > 1e-5:
			unit_ok = false
	h.assert_true(unit_ok, "all 2D dirs are unit length")

	h.finish(self)
