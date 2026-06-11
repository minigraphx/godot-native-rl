extends SceneTree
# Pure tests for NavMeshMath (#20): path-length closeness + next-waypoint direction encoding.

const Harness = preload("res://test/harness.gd")
const NavMeshMath = preload("res://addons/godot_native_rl/sensors/navmesh_math.gd")

func _initialize() -> void:
	var h := Harness.new()

	# path_length: sum of segment lengths; an L-shaped path is longer than the straight line.
	h.assert_eq(NavMeshMath.path_length([]), 0.0, "empty path length 0")
	h.assert_eq(NavMeshMath.path_length([Vector2(5, 5)]), 0.0, "single-point path length 0")
	h.assert_eq(NavMeshMath.path_length([Vector2(0, 0), Vector2(3, 0), Vector2(3, 4)]), 7.0,
		"L-path 3 + 4 = 7")

	# closeness: 1 at target, 0 at/over max_distance, clamped.
	h.assert_eq(NavMeshMath.closeness(0.0, 10.0), 1.0, "closeness at target = 1")
	h.assert_eq(NavMeshMath.closeness(5.0, 10.0), 0.5, "closeness half-way = 0.5")
	h.assert_eq(NavMeshMath.closeness(20.0, 10.0), 0.0, "closeness beyond max clamps to 0")
	h.assert_eq(NavMeshMath.closeness(5.0, 0.0), 0.0, "closeness with max<=0 = 0")

	# encode_2d: closeness from PATH length (around a corner), direction to the NEXT waypoint.
	# from=(0,0); path corners (0,0)->(0,10)->(10,10): path length 20, next waypoint (0,10) -> dir +Y.
	var e := NavMeshMath.encode_2d(Vector2(0, 0),
		[Vector2(0, 0), Vector2(0, 10), Vector2(10, 10)], 40.0)
	h.assert_eq(e.size(), 3, "encode_2d width 3")
	h.assert_eq(e[0], 0.5, "encode_2d closeness from path length (20/40 -> 0.5)")
	h.assert_eq(Vector2(e[1], e[2]), Vector2(0, 1), "encode_2d direction to next waypoint (+Y)")

	# Unreachable (empty / single-point) -> zeros.
	h.assert_eq(NavMeshMath.encode_2d(Vector2(1, 1), [], 10.0), [0.0, 0.0, 0.0],
		"encode_2d empty path -> zeros")
	h.assert_eq(NavMeshMath.encode_2d(Vector2(1, 1), [Vector2(1, 1)], 10.0), [0.0, 0.0, 0.0],
		"encode_2d single-point path -> zeros")

	# Degenerate: next waypoint == from -> zero direction (no NaN from normalizing a zero vector).
	var ez := NavMeshMath.encode_2d(Vector2(2, 2), [Vector2(2, 2), Vector2(2, 2)], 10.0)
	h.assert_eq(Vector2(ez[1], ez[2]), Vector2.ZERO, "encode_2d coincident waypoint -> zero dir")

	# encode_3d: [closeness, x, y, z]; direction to next waypoint along +Z.
	var e3 := NavMeshMath.encode_3d(Vector3(0, 0, 0),
		[Vector3(0, 0, 0), Vector3(0, 0, 5), Vector3(5, 0, 5)], 10.0)
	h.assert_eq(e3.size(), 4, "encode_3d width 4")
	h.assert_eq(e3[0], 0.0, "encode_3d closeness (path 10 / max 10 -> 0)")
	h.assert_eq(Vector3(e3[1], e3[2], e3[3]), Vector3(0, 0, 1), "encode_3d direction +Z")
	h.assert_eq(NavMeshMath.encode_3d(Vector3.ZERO, [], 10.0), [0.0, 0.0, 0.0, 0.0],
		"encode_3d empty path -> zeros")

	# #168: egocentric direction — rotate the world direction into the sensor's local frame.
	# World dir +Y; a sensor rotated +90° (PI/2) sees it rotated by -90° -> +X.
	var ego := NavMeshMath.encode_2d(Vector2(0, 0), [Vector2(0, 0), Vector2(0, 10)], 40.0, PI / 2)
	h.assert_true(Vector2(ego[1], ego[2]).is_equal_approx(Vector2(1, 0)),
		"encode_2d egocentric rotates dir into the sensor frame")
	# A non-identity 3D basis rotates the direction; identity leaves it world-frame (default).
	var basis_y := Basis(Vector3.UP, PI / 2)  # 90° about Y
	var ego3 := NavMeshMath.encode_3d(Vector3.ZERO, [Vector3(0, 0, 0), Vector3(0, 0, 1)], 10.0, basis_y)
	h.assert_true(Vector3(ego3[1], ego3[2], ego3[3]).is_equal_approx(basis_y.inverse() * Vector3(0, 0, 1)),
		"encode_3d egocentric applies the inverse basis")

	# #168: is_reachable — last path point near `to` is reached; a partial path (far end) is not.
	h.assert_true(NavMeshMath.is_reachable([Vector2(0, 0), Vector2(10, 10)], Vector2(10, 10), 1.0),
		"path ending at target is reachable")
	h.assert_true(not NavMeshMath.is_reachable([Vector2(0, 0), Vector2(2, 2)], Vector2(50, 50), 1.0),
		"partial path far from target is not reachable")
	h.assert_true(not NavMeshMath.is_reachable([], Vector2(1, 1), 1.0), "empty path is not reachable")
	h.assert_true(NavMeshMath.is_reachable([Vector3.ZERO, Vector3(5, 0, 0)], Vector3(5, 0, 0.5), 1.0),
		"3D reachable within tolerance")

	h.finish(self)
