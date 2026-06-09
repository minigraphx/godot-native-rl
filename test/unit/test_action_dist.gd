extends SceneTree

const Harness = preload("res://test/harness.gd")
const ActionDist = preload("res://addons/godot_native_rl/controllers/action_dist.gd")

func _initialize() -> void:
	var h := Harness.new()

	# validate: accepts a well-formed std array.
	h.assert_true(ActionDist.validate({"std": [0.3, 0.5]}), "validate accepts std array")
	# validate: accepts matching action_dim.
	h.assert_true(ActionDist.validate({"std": [0.3, 0.5], "action_dim": 2}),
		"validate accepts matching action_dim")
	# validate: rejects mismatched action_dim.
	h.assert_true(not ActionDist.validate({"std": [0.3, 0.5], "action_dim": 3}),
		"validate rejects action_dim mismatch")
	# validate: rejects empty std.
	h.assert_true(not ActionDist.validate({"std": []}), "validate rejects empty std")
	# validate: rejects missing std key.
	h.assert_true(not ActionDist.validate({"action_dim": 2}), "validate rejects missing std")
	# validate: rejects non-numeric std element.
	h.assert_true(not ActionDist.validate({"std": [0.3, "x"]}), "validate rejects non-numeric std")

	# to_typed: coerces JSON array into PackedFloat32Array.
	var typed := ActionDist.to_typed({"std": [0.25, 0.75]})
	h.assert_true(typed["std"] is PackedFloat32Array, "to_typed coerces to PackedFloat32Array")
	h.assert_true(absf(typed["std"][0] - 0.25) < 1e-6 and absf(typed["std"][1] - 0.75) < 1e-6,
		"to_typed preserves values")

	# to_typed: invalid -> {}.
	h.assert_eq(ActionDist.to_typed({"std": []}), {}, "to_typed invalid -> {}")

	# continuous_action_dim: sums size over continuous keys only (discrete keys contribute 0).
	h.assert_eq(ActionDist.continuous_action_dim(
		{"steer": {"size": 2, "action_type": "continuous"}}), 2,
		"continuous_action_dim sums a single continuous key")
	h.assert_eq(ActionDist.continuous_action_dim({
		"fire": {"size": 3, "action_type": "discrete"},
		"steer": {"size": 2, "action_type": "continuous"},
		"throttle": {"size": 1, "action_type": "continuous"}}), 3,
		"continuous_action_dim ignores discrete keys, sums continuous")
	h.assert_eq(ActionDist.continuous_action_dim(
		{"move": {"size": 5, "action_type": "discrete"}}), 0,
		"continuous_action_dim is 0 for a discrete-only space")

	h.finish(self)
