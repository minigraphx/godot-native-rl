extends SceneTree

const Harness = preload("res://test/harness.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")

# Mock flat sensor: has both get_observation() and obs_size() (duck-typed match).
class MockSensor extends Node:
	var _obs: Array = []
	func setup(obs: Array) -> void:
		_obs = obs
	func get_observation() -> Array:
		return _obs
	func obs_size() -> int:
		return _obs.size()

# Mock image-like sensor: has get_observation() but NO obs_size() (must be skipped).
class MockCamera extends Node:
	func get_observation() -> String:
		return "deadbeef"

func _make_sensor(obs: Array) -> MockSensor:
	var s := MockSensor.new()
	s.setup(obs)
	return s

func _initialize() -> void:
	var h := Harness.new()

	# Empty tree -> []
	var empty_root := Node.new()
	h.assert_eq(NcnnControllerCore.collect_sensors(empty_root), [], "empty tree -> []")
	empty_root.free()

	# Single sensor -> its obs
	var root1 := Node.new()
	root1.add_child(_make_sensor([1.0, 2.0]))
	h.assert_eq(NcnnControllerCore.collect_sensors(root1), [1.0, 2.0], "single sensor -> its obs")
	root1.free()

	# Multiple + nested + camera-like skip + plain node ignored.
	# Tree (insertion order): sensorA[1,2], pivot{ sensorB[3] }, camera, plain, sensorC[4,5]
	var root := Node.new()
	root.add_child(_make_sensor([1.0, 2.0]))          # sensorA
	var pivot := Node.new()
	pivot.add_child(_make_sensor([3.0]))              # sensorB (nested)
	root.add_child(pivot)
	root.add_child(MockCamera.new())                 # skipped (no obs_size)
	root.add_child(Node.new())                        # plain node, ignored
	root.add_child(_make_sensor([4.0, 5.0]))          # sensorC
	var obs: Array = NcnnControllerCore.collect_sensors(root)
	h.assert_eq(obs, [1.0, 2.0, 3.0, 4.0, 5.0], "depth-first tree order, camera+plain skipped")
	root.free()

	# Pre-order is load-bearing: a sensor that is ITSELF the parent of another sensor.
	# Depth-first PRE-order yields parent-before-child [10, 11]; a post-order traversal
	# would yield [11, 10], so this assertion distinguishes the two strategies.
	var po_root := Node.new()
	var parent_sensor := _make_sensor([10.0])
	parent_sensor.add_child(_make_sensor([11.0]))     # sensor nested under a sensor
	po_root.add_child(parent_sensor)
	po_root.add_child(_make_sensor([12.0]))
	h.assert_eq(NcnnControllerCore.collect_sensors(po_root), [10.0, 11.0, 12.0], "pre-order: parent sensor before its child sensor")
	po_root.free()

	h.finish(self)
