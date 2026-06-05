extends SceneTree

const Harness = preload("res://test/harness.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")
const ObsHistoryBuffer = preload("res://addons/godot_native_rl/sensors/obs_history_buffer.gd")

class MockSensor extends Node:
	var _obs: Array = []
	func setup(o: Array) -> void:
		_obs = o
	func get_observation() -> Array:
		return _obs
	func obs_size() -> int:
		return _obs.size()

func _initialize() -> void:
	var h := Harness.new()

	# A real wrapper holding an inner sensor: collect_sensors returns ONLY the wrapper's stacked
	# output, never the inner sensor's raw obs separately.
	var root := Node.new()
	var wrap = ObsHistoryBuffer.new()
	wrap.history_length = 2
	var inner := MockSensor.new()
	inner.setup([1.0, 2.0])
	wrap.add_child(inner)
	root.add_child(wrap)
	# history_length 2 * inner size 2 = 4; one push -> [0,0,1,2].
	h.assert_eq(NcnnControllerCore.collect_sensors(root), [0.0, 0.0, 1.0, 2.0], "wrapper collected as leaf, inner not double-counted")
	root.free()

	h.finish(self)
