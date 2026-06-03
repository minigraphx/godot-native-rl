extends SceneTree

const Harness = preload("res://test/harness.gd")
const ISensor2D = preload("res://addons/godot_native_rl/sensors/i_sensor_2d.gd")
const ISensor3D = preload("res://addons/godot_native_rl/sensors/i_sensor_3d.gd")

func _initialize() -> void:
	var h := Harness.new()

	var s2 = ISensor2D.new()
	h.assert_eq(s2.get_observation(), [], "ISensor2D base get_observation -> []")
	h.assert_eq(s2.obs_size(), 0, "ISensor2D base obs_size -> 0")
	h.assert_true(s2 is Node2D, "ISensor2D is a Node2D")
	s2.free()

	var s3 = ISensor3D.new()
	h.assert_eq(s3.get_observation(), [], "ISensor3D base get_observation -> []")
	h.assert_eq(s3.obs_size(), 0, "ISensor3D base obs_size -> 0")
	h.assert_true(s3 is Node3D, "ISensor3D is a Node3D")
	s3.free()

	h.finish(self)
