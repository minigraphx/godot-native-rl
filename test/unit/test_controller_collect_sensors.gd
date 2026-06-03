extends SceneTree

const Harness = preload("res://test/harness.gd")
const NcnnAIController2D = preload("res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd")
const NcnnAIController3D = preload("res://addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd")

class MockSensor extends Node:
	var _obs: Array = []
	func setup(obs: Array) -> void:
		_obs = obs
	func get_observation() -> Array:
		return _obs
	func obs_size() -> int:
		return _obs.size()

func _mock(obs: Array) -> MockSensor:
	var s := MockSensor.new()
	s.setup(obs)
	return s

func _initialize() -> void:
	var h := Harness.new()

	var c2 = NcnnAIController2D.new()
	c2.add_child(_mock([1.0, 2.0]))
	c2.add_child(_mock([3.0]))
	h.assert_eq(c2.collect_sensors(), [1.0, 2.0, 3.0], "2D controller concatenates child sensors")
	c2.free()

	var c3 = NcnnAIController3D.new()
	c3.add_child(_mock([7.0]))
	h.assert_eq(c3.collect_sensors(), [7.0], "3D controller concatenates child sensors")
	c3.free()

	h.finish(self)
