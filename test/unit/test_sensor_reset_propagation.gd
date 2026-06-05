extends SceneTree

const Harness = preload("res://test/harness.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")

class ResettableSensor extends Node:
	var reset_calls: int = 0
	func get_observation() -> Array:
		return [0.0]
	func obs_size() -> int:
		return 1
	func reset() -> void:
		reset_calls += 1

class PlainSensor extends Node:
	func get_observation() -> Array:
		return [0.0]
	func obs_size() -> int:
		return 1

func _initialize() -> void:
	var h := Harness.new()

	# collect_sensors_nodes returns the leaf sensor NODES (not their obs).
	var root := Node.new()
	var a := ResettableSensor.new()
	var b := PlainSensor.new()
	root.add_child(a)
	root.add_child(b)
	var nodes: Array = NcnnControllerCore.collect_sensors_nodes(root)
	h.assert_eq(nodes.size(), 2, "two sensor nodes discovered")
	h.assert_true(nodes.has(a) and nodes.has(b), "both nodes present")

	# Calling reset on those with a reset() method increments only the resettable one.
	for n in nodes:
		if n.has_method("reset"):
			n.reset()
	h.assert_eq(a.reset_calls, 1, "resettable sensor reset once")
	root.free()

	h.finish(self)
