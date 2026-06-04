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

# Returns a valid action but a WRONG-sized state blob (out1: 3 floats, sidecar declares 8), to
# exercise the returned-state size guard.
class BadStateRunner:
	var loaded := true
	func is_model_loaded() -> bool:
		return loaded
	func run_inference_multi(_inputs: Array, _output_names: PackedStringArray) -> Dictionary:
		return {
			"out0": PackedFloat32Array([0.0, 0.0, 0.9, 0.0]),
			"out1": PackedFloat32Array([1.0, 2.0, 3.0]),  # wrong: sidecar shape product is 8
			"out2": PackedFloat32Array([0, 0, 0, 0, 0, 0, 0, 0]),
		}

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

	# Controller wrapper loads the real sidecar and exposes reset_recurrent_state().
	var wrapped = Stub.new()
	wrapped.set_recurrent_contract_for_test("res://models/synthetic_lstm.recurrent.json")
	h.assert_true(not wrapped._core.recurrent_contract.is_empty(), "wrapper loads recurrent sidecar")
	h.assert_eq((wrapped._core.recurrent_state["in1"] as PackedFloat32Array).size(), 8, "wrapper zero-inits state")
	wrapped._core.recurrent_state["in1"] = PackedFloat32Array([9,9,9,9,9,9,9,9])
	wrapped.reset_recurrent_state()
	h.assert_true(absf((wrapped._core.recurrent_state["in1"] as PackedFloat32Array)[0]) < 1e-9, "reset_recurrent_state zeroes")
	wrapped.free()

	# Hardening: obs size that mismatches the sidecar obs_shape -> action skipped (no garbage action).
	var bad_obs_core = Core.new()
	bad_obs_core.recurrent_contract = _contract()
	bad_obs_core.init_recurrent_state()
	var bad_obs_agent = Stub.new()
	bad_obs_agent.obs_to_return = PackedFloat32Array([1.0, 2.0, 3.0])  # size 3, sidecar obs_shape is [5]
	bad_obs_core.choose_and_apply_action(bad_obs_agent, FakeMultiRunner.new())
	h.assert_eq(bad_obs_agent.last_action, null, "obs/obs_shape size mismatch skips action")
	bad_obs_agent.free()

	# Hardening: a returned state blob of the wrong size -> action skipped AND state not advanced
	# (caught this frame, not deferred to the next with a misdirected C++ error).
	var bad_state_core = Core.new()
	bad_state_core.recurrent_contract = _contract()
	bad_state_core.init_recurrent_state()
	var bad_state_agent = Stub.new()
	bad_state_core.choose_and_apply_action(bad_state_agent, BadStateRunner.new())
	h.assert_eq(bad_state_agent.last_action, null, "wrong-sized returned state skips action")
	h.assert_true(absf((bad_state_core.recurrent_state["in1"] as PackedFloat32Array)[0]) < 1e-9, "state not advanced on wrong-sized blob")
	bad_state_agent.free()

	h.finish(self)
