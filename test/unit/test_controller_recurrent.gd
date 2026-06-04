extends SceneTree
# NcnnControllerCore recurrent path: feeds zero state first, feeds back returned state next frame,
# decodes the action_output blob, and re-zeroes on reset().

const Harness = preload("res://test/harness.gd")
const Core = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")
const RecurrentState = preload("res://addons/godot_native_rl/controllers/recurrent_state.gd")
const Stub = preload("res://test/unit/recurrent_stub_agent.gd")

# Records inputs; returns action with argmax==2 and a state that is the input state + 1.
class FakeMultiRunner:
	var loaded := true
	var last_state_in := PackedFloat32Array()
	var last_state_in2 := PackedFloat32Array()
	func is_model_loaded() -> bool:
		return loaded
	func run_inference_multi(inputs: Array, output_names: PackedStringArray) -> Dictionary:
		for spec in inputs:
			if spec["name"] == "in1":
				last_state_in = spec["data"]
			if spec["name"] == "in2":
				last_state_in2 = spec["data"]
		var result := {}
		result["out0"] = PackedFloat32Array([0.0, 0.0, 0.9, 0.0])  # argmax == 2
		# Echo each state input + 1 to its paired output.
		for spec in inputs:
			if spec["name"] == "in1":
				var nxt := PackedFloat32Array(spec["data"])
				for i in nxt.size():
					nxt[i] += 1.0
				result["out1"] = nxt
			if spec["name"] == "in2":
				var nxt2 := PackedFloat32Array(spec["data"])
				for i in nxt2.size():
					nxt2[i] += 1.0
				result["out2"] = nxt2
		return result

# Single-IO fake mimicking NcnnRunner.run_inference (the non-recurrent path).
class FakeSingleRunner:
	var loaded := true
	var called := false
	func is_model_loaded() -> bool:
		return loaded
	func run_inference(_input) -> PackedFloat32Array:
		called = true
		return PackedFloat32Array([0.0, 0.0, 0.9, 0.0])  # argmax == 2

func _contract() -> Dictionary:
	return RecurrentState.to_typed({
		"obs_input": "in0", "obs_shape": [5], "action_output": "out0",
		"state_pairs": [
			{"in": "in1", "out": "out1", "shape": [8]},
			{"in": "in2", "out": "out2", "shape": [8]},
		],
	})

func _initialize() -> void:
	var h := Harness.new()

	var core = Core.new()
	core.recurrent_contract = _contract()
	core.init_recurrent_state()
	h.assert_eq((core.recurrent_state["in1"] as PackedFloat32Array).size(), 8, "state zero-init sized")

	var agent = Stub.new()
	var runner := FakeMultiRunner.new()

	# Frame 1: state fed in must be all zeros.
	core.choose_and_apply_action(agent, runner)
	h.assert_eq(agent.last_action, {"move": 2}, "recurrent action decoded from out0")
	h.assert_true(absf(runner.last_state_in[0]) < 1e-9, "frame 1 feeds zero state")
	h.assert_true(absf((core.recurrent_state["in1"] as PackedFloat32Array)[0] - 1.0) < 1e-9, "state advanced to out1")
	h.assert_true(absf(runner.last_state_in2[0]) < 1e-9, "frame 1 feeds zero state for second pair")
	h.assert_true(absf((core.recurrent_state["in2"] as PackedFloat32Array)[0] - 1.0) < 1e-9, "second pair state advanced to out2")

	# Frame 2: state fed in must be the advanced state (== 1.0).
	core.choose_and_apply_action(agent, runner)
	h.assert_true(absf(runner.last_state_in[0] - 1.0) < 1e-9, "frame 2 feeds back advanced state")
	h.assert_true(absf(runner.last_state_in2[0] - 1.0) < 1e-9, "frame 2 feeds back advanced state for second pair")

	# reset() re-zeroes.
	core.reset()
	core.choose_and_apply_action(agent, runner)
	h.assert_true(absf(runner.last_state_in[0]) < 1e-9, "reset() re-zeroes recurrent state")

	# Non-recurrent core routes to run_inference (NOT run_inference_multi) — empty contract.
	var plain = Core.new()
	h.assert_true(plain.recurrent_contract.is_empty(), "default core is non-recurrent")
	var plain_agent = Stub.new()
	var plain_runner := FakeSingleRunner.new()
	plain.choose_and_apply_action(plain_agent, plain_runner)
	h.assert_true(plain_runner.called, "non-recurrent core calls run_inference")
	h.assert_eq(plain_agent.last_action, {"move": 2}, "non-recurrent action decoded from run_inference")
	plain_agent.free()

	agent.free()
	h.finish(self)
