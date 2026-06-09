extends SceneTree

const Harness = preload("res://test/harness.gd")
const Controller2D = preload("res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd")
const Controller3D = preload("res://addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd")

func _initialize() -> void:
	var h := Harness.new()

	var c2 = Controller2D.new()
	h.assert_true(c2.has_signal("inference_step"), "2D controller declares inference_step signal")
	c2.free()

	var c3 = Controller3D.new()
	h.assert_true(c3.has_signal("inference_step"), "3D controller declares inference_step signal")
	c3.free()

	h.finish(self)
