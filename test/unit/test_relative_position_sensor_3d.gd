extends SceneTree

const Harness = preload("res://test/harness.gd")
const RelativePositionSensor3D = preload("res://addons/godot_native_rl/sensors/relative_position_sensor_3d.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = RelativePositionSensor3D.new()
	s.max_distance = 100.0
	s.position = Vector3.ZERO
	s.rotation = Vector3.ZERO

	var t1 := Node3D.new()
	t1.position = Vector3(0, 0, -50)   # half distance, forward (-Z)
	var t2 := Node3D.new()
	t2.position = Vector3(100, 0, 0)   # at max distance, +X
	var targets: Array[Node3D] = [t1, t2]
	s.objects_to_observe = targets

	# Default mode = non-separate, include x+y+z -> 3 floats/target.
	h.assert_eq(s.obs_size(), 6, "two targets, default mode -> obs_size 6")
	var obs: Array = s.get_observation()
	h.assert_eq(obs.size(), 6, "obs length == obs_size")
	# slot0 forward half: (0,0,-0.5); slot1 +X max: (1,0,0)
	h.assert_true(absf(obs[2] + 0.5) < 1e-5 and absf(obs[0]) < 1e-5 and absf(obs[1]) < 1e-5, "slot0 = (0,0,-0.5)")
	h.assert_true(absf(obs[3] - 1.0) < 1e-5 and absf(obs[4]) < 1e-5 and absf(obs[5]) < 1e-5, "slot1 = (1,0,0)")

	# Separate mode -> 4 floats/target -> obs_size 8.
	s.use_separate_direction = true
	h.assert_eq(s.obs_size(), 8, "separate mode two targets -> 8")
	var obs2: Array = s.get_observation()
	h.assert_eq(obs2.size(), 8, "separate obs length 8")
	# slot0 separate: dir (0,0,-1) + dist 0.5
	h.assert_true(absf(obs2[2] + 1.0) < 1e-5 and absf(obs2[3] - 0.5) < 1e-5, "slot0 separate -> (...,-1,0.5)")

	# Yaw +90deg about Y rotates the forward target (-Z) to local +X (separate mode).
	s.rotation = Vector3(0.0, PI / 2.0, 0.0)
	var obs_rot: Array = s.get_observation()
	h.assert_true(absf(obs_rot[0] - 1.0) < 1e-5, "sensor yaw rotates slot0 dir to local +X")
	s.rotation = Vector3.ZERO

	# Back to default mode; free a target -> slot zero-fills, sibling intact, length unchanged.
	s.use_separate_direction = false
	t2.free()
	var obs3: Array = s.get_observation()
	h.assert_eq(obs3.size(), 6, "freed slot keeps length 6")
	h.assert_true(absf(obs3[2] + 0.5) < 1e-5, "sibling slot0 intact")
	h.assert_true(absf(obs3[3]) < 1e-6 and absf(obs3[4]) < 1e-6 and absf(obs3[5]) < 1e-6, "freed slot1 zero-filled")

	t1.free()
	s.free()

	# Empty targets -> obs_size 0, empty obs.
	var s2 = RelativePositionSensor3D.new()
	h.assert_eq(s2.obs_size(), 0, "no targets -> obs_size 0")
	h.assert_eq(s2.get_observation().size(), 0, "no targets -> empty obs")
	s2.free()

	h.finish(self)
