extends SceneTree
# Unit tests for the gnrl_replay_v1 episode format (#39).

const Harness = preload("res://test/harness.gd")
const RF = preload("res://addons/godot_native_rl/training/replay_format.gd")

func _initialize() -> void:
	var h = Harness.new()

	var steps := [
		{"action": {"move": 1}, "reward": 0.5},
		{"action": {"move": 2}, "reward": -0.25},
		{"action": {"move": 0}, "reward": 1.0},
	]
	var ep := RF.make_episode({"scene": "res://x.tscn", "agent_index": 0, "action_repeat": 4},
		{"agent_x": 10.0}, steps)
	h.assert_eq(ep["format"], "gnrl_replay_v1", "format tag")
	h.assert_eq(ep["meta"]["n_steps"], 3, "n_steps derived")
	h.assert_true(absf(float(ep["meta"]["total_reward"]) - 1.25) < 1e-9, "total_reward summed")
	h.assert_true(RF.validate(ep), "valid episode validates")

	# JSON round-trip preserves the essentials.
	var back := RF.from_json(RF.to_json(ep))
	h.assert_true(RF.validate(back), "round-trip validates")
	h.assert_eq(int(back["meta"]["action_repeat"]), 4, "meta survives")
	h.assert_eq(back["steps"].size(), 3, "steps survive")
	h.assert_eq(int(back["steps"][1]["action"]["move"]), 2, "action content survives")
	h.assert_eq(float(back["initial_state"]["agent_x"]), 10.0, "initial_state survives")

	# Rejections (each pushes an error — intentional).
	h.assert_true(not RF.validate({"format": "other", "meta": {}, "steps": []}), "wrong format rejected")
	h.assert_true(not RF.validate({"format": "gnrl_replay_v1", "meta": {}}), "missing steps rejected")
	h.assert_true(not RF.validate({"format": "gnrl_replay_v1", "meta": {}, "steps": [42]}), "non-dict step rejected")
	h.assert_true(not RF.validate({"format": "gnrl_replay_v1", "meta": {}, "steps": [{"reward": 1.0}]}), "step without action rejected")
	h.assert_eq(RF.from_json("{bad").size(), 0, "bad json -> empty dict")

	h.finish(self)
