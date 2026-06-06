extends SceneTree
# Obs/action contract tests for ball_chase_agent.gd, plus the parity-critical
# no-double-tanh guard: SAC's actor already squashed, so deploy decode must pass raw.

const Harness = preload("res://test/harness.gd")
const AgentScript = preload("res://examples/ball_chase/ball_chase_agent.gd")
const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")

func _initialize() -> void:
	var h := Harness.new()
	var agent = AgentScript.new()

	# Action space: continuous, size 2, and (critical) NO squash key.
	var space := agent.get_action_space()
	h.assert_true(space.has("move"), "action key 'move'")
	h.assert_eq(space["move"]["action_type"], "continuous", "continuous action type")
	h.assert_eq(space["move"]["size"], 2, "action size 2")
	h.assert_true(not space["move"].get("squash", false), "no squash (SAC actor already tanh'd)")

	# compute_obs: 5 dims [pos.x_n, pos.y_n, dir.x, dir.y, dist_n]
	var obs := agent.compute_obs(Vector2(500, 300), Vector2(500, 0), Vector2(1000, 600))
	h.assert_eq(obs.size(), 5, "obs has 5 dims")
	h.assert_true(absf(obs[0] - 0.0) < 1e-4, "pos.x normalized to 0 at center")
	h.assert_true(obs[3] < 0.0, "dir.y points up toward target above")

	# set_action maps the continuous array to a thrust vector, clamped to [-1,1] on both bounds.
	agent.set_action({"move": [2.0, -2.0]})   # both out of range -> clamps to +1 / -1
	h.assert_eq(agent.get_thrust_for_test(), Vector2(1.0, -1.0), "thrust clamped to [-1,1] (both bounds)")
	agent.set_action({"move": [0.5, -0.5]})   # in range -> passed through unchanged
	h.assert_eq(agent.get_thrust_for_test(), Vector2(0.5, -0.5), "in-range thrust unchanged")

	# NO-DOUBLE-TANH GUARD: decoding a raw policy output against THIS action_space must
	# return the raw values (no tanh applied), because squash is absent/false.
	var raw := PackedFloat32Array([0.5, -0.5])
	var decoded := ActionDecode.decode_actions(raw, space)
	h.assert_true(absf(decoded["move"][0] - 0.5) < 1e-6, "decode passes raw value 0 (no tanh)")
	h.assert_true(absf(decoded["move"][1] - (-0.5)) < 1e-6, "decode passes raw value 1 (no tanh)")

	agent.free()
	h.finish(self)
