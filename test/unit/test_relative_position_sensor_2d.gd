extends SceneTree

const Harness = preload("res://test/harness.gd")
const RelativePositionSensor2D = preload("res://addons/godot_native_rl/sensors/relative_position_sensor_2d.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = RelativePositionSensor2D.new()
	s.max_distance = 100.0

	# obs_size is fixed at 3
	h.assert_eq(s.obs_size(), 3, "obs_size == 3")

	# Target ahead (+X) of an unrotated, origin sensor -> [1, 0, 0.1]
	var target := Node2D.new()
	target.position = Vector2(10, 0)
	s.set_target_for_test(target)
	s.position = Vector2.ZERO
	s.rotation = 0.0
	var obs: Array = s.get_observation()
	h.assert_eq(obs.size(), 3, "observation length == obs_size")
	h.assert_true(absf(obs[0] - 1.0) < 1e-5 and absf(obs[1]) < 1e-5 and absf(obs[2] - 0.1) < 1e-5, "target ahead -> [1,0,0.1]")

	# Rotating the sensor +90deg rotates the egocentric direction to (0,-1)
	s.rotation = PI / 2.0
	var obs_rot: Array = s.get_observation()
	h.assert_true(absf(obs_rot[0]) < 1e-5 and absf(obs_rot[1] + 1.0) < 1e-5, "sensor rotation rotates direction")

	target.free()

	# No target -> zero-filled array of obs_size (no crash)
	var s2 = RelativePositionSensor2D.new()
	var obs_none: Array = s2.get_observation()
	h.assert_eq(obs_none.size(), 3, "no target -> length 3")
	var all_zero := true
	for v in obs_none:
		if absf(v) > 1e-6:
			all_zero = false
	h.assert_true(all_zero, "no target -> zeros")

	s.free()
	s2.free()
	h.finish(self)
