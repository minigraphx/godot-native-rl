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

	h.finish(self)
