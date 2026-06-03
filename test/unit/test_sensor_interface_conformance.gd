extends SceneTree

const Harness = preload("res://test/harness.gd")
const ISensor2D = preload("res://addons/godot_native_rl/sensors/i_sensor_2d.gd")
const ISensor3D = preload("res://addons/godot_native_rl/sensors/i_sensor_3d.gd")
const RaycastSensor2D = preload("res://addons/godot_native_rl/sensors/raycast_sensor_2d.gd")
const RaycastSensor3D = preload("res://addons/godot_native_rl/sensors/raycast_sensor_3d.gd")
const RelativePositionSensor2D = preload("res://addons/godot_native_rl/sensors/relative_position_sensor_2d.gd")
const RelativePositionSensor3D = preload("res://addons/godot_native_rl/sensors/relative_position_sensor_3d.gd")
const GridSensor2D = preload("res://addons/godot_native_rl/sensors/grid_sensor_2d.gd")
const GridSensor3D = preload("res://addons/godot_native_rl/sensors/grid_sensor_3d.gd")

func _initialize() -> void:
	var h := Harness.new()

	var rs2 = RaycastSensor2D.new()
	h.assert_true(rs2 is ISensor2D, "RaycastSensor2D is ISensor2D")
	rs2.free()

	var rp2 = RelativePositionSensor2D.new()
	h.assert_true(rp2 is ISensor2D, "RelativePositionSensor2D is ISensor2D")
	rp2.free()

	var gs2 = GridSensor2D.new()
	h.assert_true(gs2 is ISensor2D, "GridSensor2D is ISensor2D")
	gs2.free()

	var rs3 = RaycastSensor3D.new()
	h.assert_true(rs3 is ISensor3D, "RaycastSensor3D is ISensor3D")
	rs3.free()

	var rp3 = RelativePositionSensor3D.new()
	h.assert_true(rp3 is ISensor3D, "RelativePositionSensor3D is ISensor3D")
	rp3.free()

	var gs3 = GridSensor3D.new()
	h.assert_true(gs3 is ISensor3D, "GridSensor3D is ISensor3D")
	gs3.free()

	h.finish(self)
