extends SceneTree
# Unit tests for the FlyBy follow camera's pure positioning helper (#227). follow_position trails
# the plane from behind (opposite its forward -Z) and above, using a horizontal heading so the
# camera never rolls/pitches. Visual framing is verified by screenshots; this guards the math.

const Harness = preload("res://test/harness.gd")
const CamScript = preload("res://examples/fly_by/fly_by_camera.gd")

func _initialize() -> void:
	var h := Harness.new()
	var cam = CamScript.new()
	cam.distance = 14.0
	cam.height = 5.0

	# Plane at origin, identity basis → forward is -Z, so the camera sits behind (+Z) and above.
	var p0 := cam.follow_position(Transform3D(Basis(), Vector3.ZERO))
	h.assert_true(p0.is_equal_approx(Vector3(0, 5, 14)), "behind+above at origin: %s" % p0)

	# Translated plane (still facing -Z): offset carries through.
	var p1 := cam.follow_position(Transform3D(Basis(), Vector3(10, 2, -5)))
	h.assert_true(p1.is_equal_approx(Vector3(10, 7, 9)), "offset carries: %s" % p1)

	# Yawed 90° about +Y: forward becomes -X, so the camera trails toward +X, still +height up.
	var yawed := Basis(Vector3.UP, PI / 2.0)
	var p2 := cam.follow_position(Transform3D(yawed, Vector3.ZERO))
	h.assert_true(p2.is_equal_approx(Vector3(14, 5, 0)), "yaw trails behind heading: %s" % p2)

	# Pure pitch/roll must NOT lift or drop the camera height (heading is flattened to horizontal):
	# pitch the plane nose-down 45° — camera height stays exactly `height`, never dives with the nose.
	var pitched := Basis(Vector3.RIGHT, -PI / 4.0)
	var p3 := cam.follow_position(Transform3D(pitched, Vector3.ZERO))
	h.assert_true(absf(p3.y - 5.0) < 1e-4, "height stays level under pitch: y=%f" % p3.y)

	cam.free()
	h.finish(self)
