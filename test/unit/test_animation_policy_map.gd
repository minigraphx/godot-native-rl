extends SceneTree
# Pure tests for AnimationPolicyMap (#22): the action -> blend-parameter routing math.

const Harness = preload("res://test/harness.gd")
const AnimationPolicyMap = preload("res://addons/godot_native_rl/controllers/animation_policy_map.gd")

func _initialize() -> void:
	var h := Harness.new()

	# affine_clamp(): affine then clamp.
	h.assert_eq(AnimationPolicyMap.affine_clamp(0.5, 2.0, 0.0, -INF, INF), 1.0, "remap scale")
	h.assert_eq(AnimationPolicyMap.affine_clamp(0.5, 1.0, 0.25, -INF, INF), 0.75, "remap offset")
	h.assert_eq(AnimationPolicyMap.affine_clamp(-1.0, 1.0, 0.0, 0.0, 1.0), 0.0, "remap clamps to min")
	h.assert_eq(AnimationPolicyMap.affine_clamp(10.0, 1.0, 0.0, 0.0, 1.0), 1.0, "remap clamps to max")
	# [-1,1] -> [0,1] via scale 0.5 offset 0.5 (the common tanh->blend case).
	h.assert_eq(AnimationPolicyMap.affine_clamp(-1.0, 0.5, 0.5, 0.0, 1.0), 0.0, "[-1,1]->[0,1] low")
	h.assert_eq(AnimationPolicyMap.affine_clamp(1.0, 0.5, 0.5, 0.0, 1.0), 1.0, "[-1,1]->[0,1] high")

	# resolve(): multiple entries to distinct params.
	var m := AnimationPolicyMap.new()
	m.add_mapping(0, "parameters/speed/blend_amount", 0.5, 0.5, 0.0, 1.0)
	m.add_mapping(1, "parameters/turn/blend_position", 180.0)
	h.assert_eq(m.mapping_count(), 2, "two mappings registered")
	var out := m.resolve(PackedFloat32Array([0.0, -0.5]))
	h.assert_eq(out["parameters/speed/blend_amount"], 0.5, "entry 0 remapped (0 -> 0.5)")
	h.assert_eq(out["parameters/turn/blend_position"], -90.0, "entry 1 scaled (-0.5*180 -> -90)")
	h.assert_eq(out.size(), 2, "resolve writes both params")

	# Out-of-range / empty action: those entries are skipped, not errored.
	var m2 := AnimationPolicyMap.new()
	m2.add_mapping(0, "a")
	m2.add_mapping(5, "b")  # index beyond the action vector
	var out2 := m2.resolve(PackedFloat32Array([0.3]))
	h.assert_eq(out2.size(), 1, "only the in-range entry resolves")
	h.assert_eq(out2["a"], 0.3, "in-range entry value")
	h.assert_true(not out2.has("b"), "out-of-range entry skipped")
	h.assert_eq(m2.resolve(PackedFloat32Array()).size(), 0, "empty action -> no writes")

	# Negative index is also skipped.
	var m3 := AnimationPolicyMap.new()
	m3.add_mapping(-1, "neg")
	h.assert_eq(m3.resolve(PackedFloat32Array([1.0])).size(), 0, "negative index skipped")

	h.finish(self)
