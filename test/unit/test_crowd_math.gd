extends SceneTree
# Pure helpers for the crowd inference path: gather obs vectors from agents, and decide which
# output slots are usable. No Net, no nodes beyond duck-typed stubs.

const Harness = preload("res://test/harness.gd")
const CrowdMath = preload("res://addons/godot_native_rl/controllers/crowd_math.gd")

class FakeAgent:
	var _obs: Array
	func _init(o: Array) -> void:
		_obs = o
	func get_obs() -> Dictionary:
		return {"obs": _obs}

func _initialize() -> void:
	var h := Harness.new()

	var agents := [FakeAgent.new([1.0, 2.0]), FakeAgent.new([3.0, 4.0])]
	var inputs := CrowdMath.gather_obs(agents)
	h.assert_eq(inputs.size(), 2, "one input per agent")
	h.assert_true(inputs[0] is PackedFloat32Array, "input is PackedFloat32Array")
	h.assert_eq(Array(inputs[1]), [3.0, 4.0], "second agent's obs gathered in order")

	# A non-empty output slot is usable; an empty one is not.
	h.assert_true(CrowdMath.output_usable(PackedFloat32Array([0.1, 0.2])), "non-empty output usable")
	h.assert_true(not CrowdMath.output_usable(PackedFloat32Array()), "empty output not usable")

	h.finish(self)
