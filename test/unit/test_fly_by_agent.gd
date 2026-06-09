extends SceneTree
# Obs/action contract for fly_by_agent.gd: 8-dim obs, {pitch,turn} continuous, clamp to [-1,1],
# and (parity-critical) NO squash key — PPO mean is unbounded, we clamp game-side instead.

const Harness = preload("res://test/harness.gd")
const AgentScript = preload("res://examples/fly_by/fly_by_agent.gd")
const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")

func _initialize() -> void:
	var h := Harness.new()
	var agent = AgentScript.new()

	# Action space: two continuous size-1 keys, no squash.
	var space := agent.get_action_space()
	h.assert_true(space.has("pitch") and space.has("turn"), "action keys pitch+turn")
	h.assert_eq(space["pitch"]["action_type"], "continuous", "pitch continuous")
	h.assert_eq(space["pitch"]["size"], 1, "pitch size 1")
	h.assert_eq(space["turn"]["size"], 1, "turn size 1")
	h.assert_true(not space["pitch"].get("squash", false), "no squash on pitch")

	# set_action clamps both inputs to [-1,1].
	agent.set_action({"pitch": [2.0], "turn": [-2.0]})
	h.assert_true(absf(agent.get_pitch_for_test() - 1.0) < 1e-6, "pitch clamped to +1")
	h.assert_true(absf(agent.get_turn_for_test() - (-1.0)) < 1e-6, "turn clamped to -1")
	agent.set_action({"pitch": [0.4], "turn": [-0.3]})
	h.assert_true(absf(agent.get_pitch_for_test() - 0.4) < 1e-6, "in-range pitch unchanged")

	# Decoding a raw 2-elem mean against this space returns raw values (no tanh; squash absent).
	var decoded := ActionDecode.decode_actions(PackedFloat32Array([0.5, -0.5]), space)
	h.assert_true(absf(decoded["pitch"][0] - 0.5) < 1e-6, "decode passes raw pitch")
	h.assert_true(absf(decoded["turn"][0] - (-0.5)) < 1e-6, "decode passes raw turn")

	agent.free()
	h.finish(self)
