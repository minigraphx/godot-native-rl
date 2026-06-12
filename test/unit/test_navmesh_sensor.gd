extends SceneTree
# NavMeshSensor2D/3D (#20): exercises the full get_observation() path with an injected path provider
# (set_path_fn_for_test), so no baked NavigationServer map is needed — same approach as the raycast
# sensor tests. Confirms obs_size, the encoded observation, the from/to passed to the query, and the
# invalid-target zero-fill.

const Harness = preload("res://test/harness.gd")
const NavMeshSensor2D = preload("res://addons/godot_native_rl/sensors/navmesh_sensor_2d.gd")
const NavMeshSensor3D = preload("res://addons/godot_native_rl/sensors/navmesh_sensor_3d.gd")

func _initialize() -> void:
	var h := Harness.new()

	# --- 2D ---
	var s := NavMeshSensor2D.new()
	s.position = Vector2(0, 0)
	s.max_distance = 40.0
	var tgt := Node2D.new()
	tgt.position = Vector2(10, 10)
	s.target = tgt
	# Record the query args; return an L-path around a corner (length 20, next waypoint +Y).
	var seen := {"from": null, "to": null}
	s.set_path_fn_for_test(func(from, to):
		seen["from"] = from
		seen["to"] = to
		return PackedVector2Array([Vector2(0, 0), Vector2(0, 10), Vector2(10, 10)]))

	h.assert_eq(s.obs_size(), 3, "2D obs_size 3")
	var obs := s.get_observation()
	h.assert_eq(obs.size(), 3, "2D observation width 3")
	h.assert_eq(obs[0], 0.5, "2D closeness from path length (20/40)")
	h.assert_eq(Vector2(obs[1], obs[2]), Vector2(0, 1), "2D direction to next waypoint")
	h.assert_eq(seen["from"], Vector2(0, 0), "2D query 'from' is the sensor position")
	h.assert_eq(seen["to"], Vector2(10, 10), "2D query 'to' is the target position")

	# Unreachable: provider returns an empty path -> zero-filled.
	s.set_path_fn_for_test(func(_from, _to): return PackedVector2Array())
	h.assert_eq(s.get_observation(), [0.0, 0.0, 0.0], "2D unreachable -> zeros")

	# Invalid target -> zero-filled, no query.
	tgt.free()
	h.assert_eq(s.get_observation(), [0.0, 0.0, 0.0], "2D freed target -> zeros")
	s.free()

	# --- 3D ---
	var s3 := NavMeshSensor3D.new()
	s3.position = Vector3.ZERO
	s3.max_distance = 10.0
	var tgt3 := Node3D.new()
	tgt3.position = Vector3(5, 0, 5)
	s3.target = tgt3
	s3.set_path_fn_for_test(func(_from, _to):
		return PackedVector3Array([Vector3(0, 0, 0), Vector3(0, 0, 5), Vector3(5, 0, 5)]))

	h.assert_eq(s3.obs_size(), 4, "3D obs_size 4")
	var obs3 := s3.get_observation()
	h.assert_eq(obs3.size(), 4, "3D observation width 4")
	h.assert_eq(obs3[0], 0.0, "3D closeness (path 10 / max 10 -> 0)")
	h.assert_eq(Vector3(obs3[1], obs3[2], obs3[3]), Vector3(0, 0, 1), "3D direction +Z")
	tgt3.free()
	s3.free()

	# --- #168: egocentric frame + opt-out ---
	var se := NavMeshSensor2D.new()
	se.position = Vector2.ZERO
	se.rotation = PI / 2  # facing "down"; egocentric direction is rotated by -PI/2
	se.max_distance = 40.0
	var te := Node2D.new()
	te.position = Vector2(0, 10)
	se.target = te
	se.set_path_fn_for_test(func(_f, _t): return PackedVector2Array([Vector2(0, 0), Vector2(0, 10)]))
	var oe := se.get_observation()  # egocentric default true: world +Y -> local +X
	h.assert_true(Vector2(oe[1], oe[2]).is_equal_approx(Vector2(1, 0)),
		"egocentric (default) rotates the direction into the sensor frame")
	se.egocentric = false
	var ow := se.get_observation()  # world frame: +Y
	h.assert_true(Vector2(ow[1], ow[2]).is_equal_approx(Vector2(0, 1)),
		"egocentric=false emits the world-frame direction")
	te.free()
	se.free()

	# --- #168: require_reachable gates a partial path (disconnected island) ---
	var sr := NavMeshSensor2D.new()
	sr.position = Vector2.ZERO
	sr.max_distance = 100.0
	sr.require_reachable = true
	sr.reachable_tolerance = 1.0
	var tr := Node2D.new()
	tr.position = Vector2(90, 0)  # "walled off"
	sr.target = tr
	# Partial path: ends at (5,0), nowhere near the (90,0) target -> reads as unreachable.
	sr.set_path_fn_for_test(func(_f, _t): return PackedVector2Array([Vector2(0, 0), Vector2(5, 0)]))
	h.assert_eq(sr.get_observation(), [0.0, 0.0, 0.0],
		"require_reachable: partial path to a walled-off target zero-fills")
	# A path that actually reaches the target encodes normally.
	sr.set_path_fn_for_test(func(_f, _t): return PackedVector2Array([Vector2(0, 0), Vector2(90, 0)]))
	h.assert_true(sr.get_observation()[0] > 0.0, "require_reachable: a reaching path encodes closeness")
	tr.free()
	sr.free()

	h.finish(self)
