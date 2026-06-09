extends SceneTree
# CrowdController wiring with a fake runner (no native Net): proves gather -> batch ->
# decode -> scatter, including skipping an agent whose output slot is empty.

const Harness = preload("res://test/harness.gd")
const CrowdController = preload("res://addons/godot_native_rl/controllers/crowd_controller.gd")

# Fake runner: records the inputs it was given and returns canned per-agent logits. The second
# slot is empty to exercise the skip path.
class FakeRunner:
	var last_inputs: Array = []
	var last_threads: int = -999
	func is_model_loaded() -> bool:
		return true
	func run_inference_batch(inputs: Array, num_threads: int) -> Array:
		last_inputs = inputs
		last_threads = num_threads
		# Agent 0: argmax index 2. Agent 1: empty (failed slot).
		return [PackedFloat32Array([0.1, 0.2, 0.9, 0.0, 0.1]), PackedFloat32Array()]

# Fake crowd agent: 5-dim obs, 5-way discrete action, records the action it received.
class FakeAgent:
	extends Node
	var received := {"set": false, "index": -1}
	func get_obs() -> Dictionary:
		return {"obs": [0.0, 0.0, 0.0, 0.0, 0.0]}
	func get_action_space() -> Dictionary:
		return {"move": {"size": 5, "action_type": "discrete"}}
	func set_action(action) -> void:
		received["set"] = true
		received["index"] = int(action["move"])

func _initialize() -> void:
	var h := Harness.new()

	var controller = CrowdController.new()
	root.add_child(controller)
	var a0 := FakeAgent.new()
	var a1 := FakeAgent.new()
	controller.add_child(a0)
	controller.add_child(a1)
	controller.num_threads = 4
	controller.set_runner_for_test(FakeRunner.new())
	controller.register_agents()

	h.assert_eq(controller.agent_count(), 2, "both child agents registered")

	controller.decide()

	# Agent 0 gets argmax (index 2); agent 1's empty slot is skipped (no action set).
	h.assert_true(a0.received["set"], "agent 0 received an action")
	h.assert_eq(a0.received["index"], 2, "agent 0 action is argmax index 2")
	h.assert_true(not a1.received["set"], "agent 1 (empty slot) was skipped")

	controller.free()
	h.finish(self)
