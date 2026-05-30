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

	h.finish(self)
