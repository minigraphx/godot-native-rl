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

	# --- softmax: stable, sums to 1, uniform-in -> uniform-out ---
	var sm := InferenceMath.softmax(PackedFloat32Array([0.0, 0.0]))
	h.assert_true(absf(sm[0] - 0.5) < 1e-6 and absf(sm[1] - 0.5) < 1e-6, "softmax uniform -> [0.5,0.5]")
	var sm2 := InferenceMath.softmax(PackedFloat32Array([1.0, 2.0, 3.0]))
	var ssum := sm2[0] + sm2[1] + sm2[2]
	h.assert_true(absf(ssum - 1.0) < 1e-6, "softmax sums to 1")
	h.assert_true(sm2[2] > sm2[1] and sm2[1] > sm2[0], "softmax monotone in logits")
	var sm3 := InferenceMath.softmax(PackedFloat32Array([1000.0, 1001.0]))
	h.assert_true(is_finite(sm3[0]) and is_finite(sm3[1]) and absf(sm3[0] + sm3[1] - 1.0) < 1e-6,
		"softmax stable for large logits")
	h.assert_eq(InferenceMath.softmax(PackedFloat32Array()).size(), 0, "softmax empty -> empty")

	h.finish(self)
