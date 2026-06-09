extends SceneTree

const Harness = preload("res://test/harness.gd")
const Controller2D = preload("res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd")
const Controller3D = preload("res://addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")

# Minimal fake runner: always "loaded", returns a fixed logit vector.
class FakeRunner:
	extends RefCounted
	var out := PackedFloat32Array([2.0, 0.0, -1.0])
	func is_model_loaded() -> bool:
		return true
	func run_inference(_v: PackedFloat32Array) -> PackedFloat32Array:
		return out

# Minimal fake agent Node that declares the signal and the controller contract.
class FakeAgent:
	extends Node2D
	signal inference_step(debug: Dictionary)
	var last_action := {}
	func get_inference_image() -> Image:
		return null
	func get_obs() -> Dictionary:
		return {"obs": [0.5, -0.5]}
	func get_action_space() -> Dictionary:
		return {"move": {"size": 3, "action_type": "discrete"}}
	func set_action(action) -> void:
		last_action = action

func _initialize() -> void:
	var h := Harness.new()

	# --- signal existence on the real controllers ---
	var c2 = Controller2D.new()
	h.assert_true(c2.has_signal("inference_step"), "2D controller declares inference_step signal")
	c2.free()
	var c3 = Controller3D.new()
	h.assert_true(c3.has_signal("inference_step"), "3D controller declares inference_step signal")
	c3.free()

	# --- core emits the payload through the agent ---
	var captured := {"hit": false, "payload": {}}
	var agent := FakeAgent.new()
	agent.name = "Bot"
	agent.inference_step.connect(func(debug): captured["hit"] = true; captured["payload"] = debug)
	var core := NcnnControllerCore.new()
	core.choose_and_apply_action(agent, FakeRunner.new())

	h.assert_true(captured["hit"], "core emitted inference_step")
	var p: Dictionary = captured["payload"]
	h.assert_eq(p.get("agent_name", ""), "Bot", "payload agent_name")
	h.assert_eq(PackedFloat32Array(p.get("obs", [])), PackedFloat32Array([0.5, -0.5]), "payload obs vector")
	h.assert_eq(PackedFloat32Array(p.get("logits", [])), PackedFloat32Array([2.0, 0.0, -1.0]), "payload raw logits")
	h.assert_eq(int(p.get("action", {}).get("move", -1)), 0, "payload decoded action (argmax of logits)")
	h.assert_true(p.get("action_space", {}).has("move"), "payload action_space present")
	h.assert_true((p.get("obs_image", {}) as Dictionary).is_empty(), "payload obs_image empty on float path")
	agent.free()

	h.finish(self)
