extends SceneTree

const Harness = preload("res://test/harness.gd")
const InferenceMath = preload("res://addons/godot_native_rl/controllers/inference_math.gd")

func _initialize() -> void:
	var h := Harness.new()

	h.assert_eq(InferenceMath.argmax(PackedFloat32Array([0.1, 0.9, 0.2, 0.0])), 1, "argmax picks max index")
	h.assert_eq(InferenceMath.argmax(PackedFloat32Array([5.0])), 0, "argmax single element")
	# Tie -> first index wins (strict > comparison).
	h.assert_eq(InferenceMath.argmax(PackedFloat32Array([0.5, 0.5, 0.1])), 0, "argmax tie -> first")
	h.assert_eq(InferenceMath.argmax(PackedFloat32Array([-3.0, -1.0, -2.0])), 1, "argmax negative values")
	# Empty -> -1 sentinel (matches run_discrete_action error contract).
	h.assert_eq(InferenceMath.argmax(PackedFloat32Array()), -1, "argmax empty -> -1")

	h.finish(self)
