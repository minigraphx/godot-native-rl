extends SceneTree

const Harness = preload("res://test/harness.gd")
const RelativePositionSensor3D = preload("res://addons/godot_native_rl/sensors/relative_position_sensor_3d.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = RelativePositionSensor3D.new()
	s.max_distance = 100.0

	# obs_size is fixed at 4
	h.assert_eq(s.obs_size(), 4, "obs_size == 4")

	# Target along -Z (forward) of an unrotated, origin sensor -> [0,0,-1,0.1]
	var target := Node3D.new()
	target.position = Vector3(0, 0, -10)
	s.set_target_for_test(target)
	s.position = Vector3.ZERO
	s.rotation = Vector3.ZERO
	var obs: Array = s.get_observation()
	h.assert_eq(obs.size(), 4, "observation length == obs_size")
	h.assert_true(absf(obs[0]) < 1e-5 and absf(obs[1]) < 1e-5 and absf(obs[2] + 1.0) < 1e-5 and absf(obs[3] - 0.1) < 1e-5, "target forward -> [0,0,-1,0.1]")

	# Sensor yawed +90deg about Y rotates the forward target to local +X
	s.rotation = Vector3(0.0, PI / 2.0, 0.0)
	var obs_rot: Array = s.get_observation()
	h.assert_true(absf(obs_rot[0] - 1.0) < 1e-5 and absf(obs_rot[1]) < 1e-5 and absf(obs_rot[2]) < 1e-5, "sensor yaw rotates direction")

	target.free()

	# No target -> zero-filled array of obs_size (no crash)
	var s2 = RelativePositionSensor3D.new()
	var obs_none: Array = s2.get_observation()
	h.assert_eq(obs_none.size(), 4, "no target -> length 4")
	var all_zero := true
	for v in obs_none:
		if absf(v) > 1e-6:
			all_zero = false
	h.assert_true(all_zero, "no target -> zeros")

	s.free()
	s2.free()
	h.finish(self)
