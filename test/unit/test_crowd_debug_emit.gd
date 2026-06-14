extends SceneTree
# #232: CrowdController emits a per-unit inference_step payload through each agent node so the
# PolicyDebugOverlay works on batched-crowd scenes. Proves the payload shape matches the single-agent
# controllers and that the controller-level identity (policy/model) travels inside the payload (the
# per-unit nodes don't expose those props). The empty-output slot must NOT emit.

const Harness = preload("res://test/harness.gd")
const CrowdController = preload("res://addons/godot_native_rl/controllers/crowd_controller.gd")

# Fake runner: agent 0 gets argmax index 2; agent 1's slot is empty (failed) -> must be skipped.
class FakeRunner:
	func is_model_loaded() -> bool:
		return true
	func run_inference_batch(_inputs: Array, _num_threads: int) -> Array:
		return [PackedFloat32Array([0.1, 0.2, 0.9, 0.0, 0.1]), PackedFloat32Array()]

# Fake crowd unit: declares the signal (so the overlay would discover it) + records its payload.
class FakeAgent:
	extends Node
	signal inference_step(debug: Dictionary)
	var captured := {"hit": false, "payload": {}}
	func _init() -> void:
		inference_step.connect(func(debug): captured["hit"] = true; captured["payload"] = debug)
	func get_obs() -> Dictionary:
		return {"obs": [0.1, 0.2, 0.3, 0.4, 0.5]}
	func get_action_space() -> Dictionary:
		return {"move": {"size": 5, "action_type": "discrete"}}
	func set_action(_action) -> void:
		pass

func _initialize() -> void:
	var h := Harness.new()

	var controller = CrowdController.new()
	controller.policy_name = "crowd_policy"
	controller.model_param_path = "res://examples/chase_the_target/models/chase_the_target.ncnn.param"
	root.add_child(controller)
	var a0 := FakeAgent.new()
	a0.name = "Unit0"
	var a1 := FakeAgent.new()
	a1.name = "Unit1"
	controller.add_child(a0)
	controller.add_child(a1)
	controller.set_runner_for_test(FakeRunner.new())
	controller.register_agents()

	controller.decide()

	# Unit 0: payload emitted with the full shape.
	h.assert_true(a0.captured["hit"], "unit 0 emitted inference_step")
	var p: Dictionary = a0.captured["payload"]
	h.assert_eq(p.get("agent_name", ""), "Unit0", "payload agent_name is the unit name")
	h.assert_eq(PackedFloat32Array(p.get("obs", [])), PackedFloat32Array([0.1, 0.2, 0.3, 0.4, 0.5]), "payload obs is this unit's obs")
	h.assert_eq(PackedFloat32Array(p.get("logits", [])), PackedFloat32Array([0.1, 0.2, 0.9, 0.0, 0.1]), "payload raw logits")
	h.assert_eq(int(p.get("action", {}).get("move", -1)), 2, "payload decoded action (argmax index 2)")
	h.assert_true(p.get("action_space", {}).has("move"), "payload action_space present")
	h.assert_true((p.get("obs_image", {}) as Dictionary).is_empty(), "payload obs_image empty (vector obs)")

	# Identity travels in the payload (the unit node has no policy/model props of its own).
	var identity: Dictionary = p.get("identity", {})
	h.assert_eq(str(identity.get("policy_name", "")), "crowd_policy", "payload identity carries the controller policy_name")
	h.assert_eq(str(identity.get("model", "")), "chase_the_target.ncnn.param", "payload identity carries the model basename")

	# Unit 1: empty output slot -> skipped, no emission.
	h.assert_true(not a1.captured["hit"], "unit 1 (empty slot) did not emit")

	controller.free()
	h.finish(self)
