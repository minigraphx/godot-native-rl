extends SceneTree
# Unit test for the pure recurrent-contract helper: validate(), to_typed(), zero_state().

const Harness = preload("res://test/harness.gd")
const RecurrentState = preload("res://addons/godot_native_rl/controllers/recurrent_state.gd")

func _valid() -> Dictionary:
	return {
		"obs_input": "in0", "obs_shape": [5], "action_output": "out0",
		"state_pairs": [
			{"in": "in1", "out": "out1", "shape": [8]},
			{"in": "in2", "out": "out2", "shape": [8]},
		],
	}

func _initialize() -> void:
	var h := Harness.new()

	h.assert_true(RecurrentState.validate(_valid()), "well-formed contract validates")
	h.assert_true(not RecurrentState.validate({}), "empty dict invalid")

	var no_obs := _valid()
	no_obs.erase("obs_input")
	h.assert_true(not RecurrentState.validate(no_obs), "missing obs_input invalid")

	var empty_pairs := _valid()
	empty_pairs["state_pairs"] = []
	h.assert_true(not RecurrentState.validate(empty_pairs), "empty state_pairs invalid")

	var bad_pair := _valid()
	bad_pair["state_pairs"] = [{"in": "in1", "out": "out1"}]  # no shape
	h.assert_true(not RecurrentState.validate(bad_pair), "pair without shape invalid")

	var bad_shape := _valid()
	bad_shape["state_pairs"] = [{"in": "in1", "out": "out1", "shape": [0]}]
	h.assert_true(not RecurrentState.validate(bad_shape), "non-positive shape dim invalid")

	var bad_name := _valid()
	bad_name["obs_input"] = &"in0"  # StringName, not String -> must be rejected
	h.assert_true(not RecurrentState.validate(bad_name), "StringName obs_input rejected")

	var typed: Dictionary = RecurrentState.to_typed(_valid())
	h.assert_eq(typed["obs_input"], "in0", "to_typed keeps obs_input")
	h.assert_true(typed["obs_shape"] is PackedInt32Array, "obs_shape typed to PackedInt32Array")
	h.assert_eq((typed["state_pairs"] as Array).size(), 2, "two state pairs typed")

	var zero: Dictionary = RecurrentState.zero_state(typed)
	h.assert_eq(zero.size(), 2, "zero_state has one entry per pair")
	h.assert_eq((zero["in1"] as PackedFloat32Array).size(), 8, "in1 zero vector sized from shape product")
	h.assert_true(absf((zero["in1"] as PackedFloat32Array)[3]) < 1e-9, "zero_state is zero-filled")

	h.assert_eq(RecurrentState.shape_product(PackedInt32Array([2, 4])), 8, "shape_product multiplies multi-dim")
	h.assert_eq(RecurrentState.shape_product(PackedInt32Array([8])), 8, "shape_product of 1-D shape")

	h.finish(self)
