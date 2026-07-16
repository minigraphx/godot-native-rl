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
	h.assert_true(absf(InferenceMath.softmax(PackedFloat32Array([5.0]))[0] - 1.0) < 1e-6, "softmax single -> [1.0]")

	# --- sample_categorical: inverse-CDF over a prob vector with a uniform draw u in [0,1) ---
	var p := PackedFloat32Array([0.3, 0.7])
	h.assert_eq(InferenceMath.sample_categorical(p, 0.0), 0, "u=0 -> first bucket")
	h.assert_eq(InferenceMath.sample_categorical(p, 0.29), 0, "u just below boundary -> 0")
	h.assert_eq(InferenceMath.sample_categorical(p, 0.5), 1, "u past boundary -> 1")
	h.assert_eq(InferenceMath.sample_categorical(p, 0.999), 1, "u near 1 -> last")
	h.assert_eq(InferenceMath.sample_categorical(p, 1.5), 1, "u>=1 clamps to last index")
	h.assert_eq(InferenceMath.sample_categorical(PackedFloat32Array([0.0, 0.0, 1.0]), 0.0),
		2, "one-hot at end -> that index for u=0")
	h.assert_eq(InferenceMath.sample_categorical(PackedFloat32Array([1.0, 0.0]), 0.5),
		0, "one-hot at start -> index 0")
	h.assert_eq(InferenceMath.sample_categorical(PackedFloat32Array(), 0.5), -1, "empty -> -1")
	h.assert_eq(InferenceMath.sample_categorical(p, -0.5), 0, "u<0 -> first bucket")

	# apply_action_mask (#385): masked slots -> -6e4 sentinel; fresh array; edge fallbacks
	var m := InferenceMath.apply_action_mask(PackedFloat32Array([1.0, 2.0, 3.0, 4.0, 5.0]), [1, 0, 1, 0, 1])
	h.assert_eq(m[0], 1.0, "unmasked slot 0 preserved")
	h.assert_true(m[1] <= -6.0e4, "masked slot 1 -> sentinel")
	h.assert_eq(m[2], 3.0, "unmasked slot 2 preserved")
	h.assert_true(m[3] <= -6.0e4, "masked slot 3 -> sentinel")
	# argmax over the masked logits never returns a masked index
	h.assert_eq(InferenceMath.argmax(m), 4, "argmax skips masked slots")
	# size mismatch -> unmasked passthrough (loud)
	var mm := InferenceMath.apply_action_mask(PackedFloat32Array([1.0, 2.0]), [1, 0, 1])
	h.assert_eq(mm[1], 2.0, "size-mismatch returns logits unmasked")
	# all-masked -> unmasked passthrough (loud), argmax still well-defined
	var am := InferenceMath.apply_action_mask(PackedFloat32Array([1.0, 9.0]), [0, 0])
	h.assert_eq(InferenceMath.argmax(am), 1, "all-masked falls back to unmasked argmax")

	h.finish(self)
