extends SceneTree

const Harness = preload("res://test/harness.gd")
const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")

func _initialize() -> void:
	var h := Harness.new()

	# Single discrete key: argmax over the whole output (== today's behavior).
	var disc := {"move": {"size": 4, "action_type": "discrete"}}
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.1, 0.9, 0.2, 0.0]), disc),
		{"move": 1}, "single discrete -> argmax index")

	# Discrete tie -> first index wins (matches InferenceMath.argmax).
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.5, 0.5, 0.1]),
		{"move": {"size": 3, "action_type": "discrete"}}),
		{"move": 0}, "discrete tie -> first index")

	# Multi-discrete: two keys, each argmax over its own segment.
	var multi := {"a": {"size": 2, "action_type": "discrete"}, "b": {"size": 3, "action_type": "discrete"}}
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.2, 0.8, 0.1, 0.0, 0.9]), multi),
		{"a": 1, "b": 2}, "multi-discrete -> per-segment argmax")

	# Continuous, no squash: raw mean values passed through.
	var cont := {"steer": {"size": 2, "action_type": "continuous"}}
	var r1 := ActionDecode.decode_actions(PackedFloat32Array([0.25, -0.5]), cont)
	h.assert_eq(r1.size(), 1, "continuous returns one key")
	h.assert_true(r1.has("steer") and r1["steer"].size() == 2, "continuous segment length 2")
	h.assert_true(absf(r1["steer"][0] - 0.25) < 1e-6 and absf(r1["steer"][1] - (-0.5)) < 1e-6,
		"continuous no-squash -> raw values")

	# Continuous, squash: tanh applied per element.
	var cont_sq := {"steer": {"size": 2, "action_type": "continuous", "squash": true}}
	var r2 := ActionDecode.decode_actions(PackedFloat32Array([0.25, -0.5]), cont_sq)
	h.assert_true(absf(r2["steer"][0] - tanh(0.25)) < 1e-6 and absf(r2["steer"][1] - tanh(-0.5)) < 1e-6,
		"continuous squash -> tanh values")

	# Mixed space: discrete then continuous, in insertion order.
	var mixed := {"fire": {"size": 2, "action_type": "discrete"}, "steer": {"size": 2, "action_type": "continuous"}}
	var r3 := ActionDecode.decode_actions(PackedFloat32Array([0.1, 0.9, 0.3, -0.3]), mixed)
	h.assert_eq(r3["fire"], 1, "mixed: discrete decoded")
	h.assert_true(absf(r3["steer"][0] - 0.3) < 1e-6 and absf(r3["steer"][1] - (-0.3)) < 1e-6,
		"mixed: continuous decoded")

	# Shape mismatch (output too short) -> {} sentinel.
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.1, 0.2]), multi),
		{}, "output too short -> {}")

	# Shape mismatch (output too long) -> {} sentinel.
	# This case has fewer values than the single key needs, so it trips the per-key
	# "too short" guard rather than the trailing over-length check.
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.1, 0.9, 0.5]), disc),
		{}, "output too long -> {}")

	# Genuine over-length: every key segment fits, but there are trailing extra values,
	# so the post-loop `index != output.size()` branch fires (not the per-key guard).
	var small := {"move": {"size": 2, "action_type": "discrete"}}
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.1, 0.9, 0.5]), small),
		{}, "trailing extra values -> {} (over-length branch)")

	# Unknown action_type -> {} sentinel.
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.1, 0.2]),
		{"x": {"size": 2, "action_type": "bogus"}}),
		{}, "unknown action_type -> {}")

	# Non-positive size -> {} sentinel (degenerate action space; fail fast, don't leak -1).
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.1, 0.2]),
		{"x": {"size": 0, "action_type": "discrete"}}),
		{}, "size 0 -> {}")

	# --- Stochastic discrete sampling (deterministic_inference = false) ---
	# Default deterministic path is unchanged (regression guard).
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.1, 0.9, 0.2, 0.0]), disc, true),
		{"move": 1}, "deterministic=true still argmax")

	# Peaked logits -> the peak is returned no matter the seed (softmax ~ one-hot).
	var peaked := {"move": {"size": 3, "action_type": "discrete"}}
	for s in [1, 7, 99]:
		var rng_peak := RandomNumberGenerator.new()
		rng_peak.seed = s
		h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.0, 12.0, 0.0]), peaked, false, rng_peak),
			{"move": 1}, "stochastic peaked logits -> peak index (seed %d)" % s)

	# Reproducibility: same seed + same logits -> identical sampled sequence.
	var rng_a := RandomNumberGenerator.new(); rng_a.seed = 42
	var rng_b := RandomNumberGenerator.new(); rng_b.seed = 42
	var seq_a: Array = []
	var seq_b: Array = []
	for i in range(20):
		seq_a.append(ActionDecode.decode_actions(PackedFloat32Array([0.0, 1.0, 0.0]), peaked, false, rng_a)["move"])
		seq_b.append(ActionDecode.decode_actions(PackedFloat32Array([0.0, 1.0, 0.0]), peaked, false, rng_b)["move"])
	h.assert_eq(seq_a, seq_b, "same seed -> identical sampled sequence")

	# Histogram: logits [0,2,0] -> softmax ~ [0.106, 0.787, 0.106]. Seeded RNG -> deterministic run.
	var rng_h := RandomNumberGenerator.new(); rng_h.seed = 123
	var counts := [0, 0, 0]
	var draws := 3000
	for i in range(draws):
		var idx: int = ActionDecode.decode_actions(PackedFloat32Array([0.0, 2.0, 0.0]), peaked, false, rng_h)["move"]
		counts[idx] += 1
	var frac1 := float(counts[1]) / draws
	h.assert_true(frac1 > 0.72 and frac1 < 0.84, "stochastic histogram: class 1 ~ 0.79 (got %f)" % frac1)
	h.assert_true(counts[0] > 0 and counts[2] > 0, "stochastic histogram: tails non-empty")

	# Multi-discrete stochastic: each key sampled independently (peaked -> deterministic).
	var rng_m := RandomNumberGenerator.new(); rng_m.seed = 5
	var multi_peak := {"a": {"size": 2, "action_type": "discrete"}, "b": {"size": 3, "action_type": "discrete"}}
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([12.0, 0.0, 0.0, 0.0, 12.0]), multi_peak, false, rng_m),
		{"a": 0, "b": 2}, "multi-discrete stochastic: per-key peak")

	# Continuous unaffected by the stochastic flag (still mean / tanh).
	var rng_c := RandomNumberGenerator.new(); rng_c.seed = 9
	var r_cont := ActionDecode.decode_actions(PackedFloat32Array([0.25, -0.5]), cont, false, rng_c)
	h.assert_true(absf(r_cont["steer"][0] - 0.25) < 1e-6 and absf(r_cont["steer"][1] - (-0.5)) < 1e-6,
		"continuous unaffected by deterministic=false")

	h.finish(self)
