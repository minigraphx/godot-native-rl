extends SceneTree
# Integration test for AnimationPolicyAdapter (#22): apply() must write the resolved blend-param
# values onto the target via set(). A stub records the writes through _set so the test needs no
# configured AnimationTree (AnimationPlayer, state machine, etc.).

const Harness = preload("res://test/harness.gd")
const AnimationPolicyAdapter = preload("res://addons/godot_native_rl/controllers/animation_policy_adapter.gd")
const AnimationPolicyMap = preload("res://addons/godot_native_rl/controllers/animation_policy_map.gd")

# Captures set("parameters/...", value) writes: Godot routes unknown property writes through _set.
class TreeStub:
	extends Node
	var writes: Dictionary = {}
	func _set(property: StringName, value: Variant) -> bool:
		writes[String(property)] = value
		return true

func _initialize() -> void:
	var h := Harness.new()

	var stub := TreeStub.new()
	var map := AnimationPolicyMap.new()
	map.add_mapping(0, "parameters/locomotion/blend_amount", 0.5, 0.5, 0.0, 1.0)  # [-1,1]->[0,1]
	map.add_mapping(1, "parameters/lean/blend_position", 1.0)

	var adapter := AnimationPolicyAdapter.new()
	adapter.setup_for_test(stub, map)

	adapter.apply(PackedFloat32Array([1.0, -0.25]))
	h.assert_eq(stub.writes.get("parameters/locomotion/blend_amount", -1.0), 1.0,
		"action[0]=1 -> blend_amount 1.0 (remapped)")
	h.assert_eq(stub.writes.get("parameters/lean/blend_position", 99.0), -0.25,
		"action[1] passes through to lean")
	h.assert_eq(stub.writes.size(), 2, "both blend params written")

	# A second apply overwrites with fresh values (drives animation each frame).
	adapter.apply(PackedFloat32Array([-1.0, 0.5]))
	h.assert_eq(stub.writes["parameters/locomotion/blend_amount"], 0.0, "blend_amount updated to 0.0")
	h.assert_eq(stub.writes["parameters/lean/blend_position"], 0.5, "lean updated to 0.5")

	# add_mapping on the adapter (sugar over the map) also routes.
	var stub2 := TreeStub.new()
	var adapter2 := AnimationPolicyAdapter.new()
	adapter2.setup_for_test(stub2, AnimationPolicyMap.new())
	adapter2.add_mapping(0, "parameters/x")
	adapter2.apply(PackedFloat32Array([0.7]))
	h.assert_eq(stub2.writes.get("parameters/x", -1.0), 0.7, "adapter.add_mapping routes")

	# #164: a freed AnimationTree must not be dereferenced — apply() is a safe no-op (is_instance_valid).
	var stub3 := TreeStub.new()
	var adapter3 := AnimationPolicyAdapter.new()
	adapter3.setup_for_test(stub3, AnimationPolicyMap.new())
	adapter3.add_mapping(0, "parameters/y")
	stub3.free()  # free the tree out from under the still-alive adapter
	adapter3.apply(PackedFloat32Array([0.5]))  # must not crash
	h.assert_true(true, "apply() on a freed tree is a safe no-op (no crash)")

	stub.free()
	stub2.free()
	adapter.free()
	adapter2.free()
	adapter3.free()
	h.finish(self)
