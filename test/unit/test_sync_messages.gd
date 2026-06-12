extends SceneTree

const Harness = preload("res://test/harness.gd")
const SyncScript = preload("res://addons/godot_native_rl/sync.gd")
const PolicyStub = preload("res://test/unit/policy_name_stub.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s := SyncScript.new()

	var step = s.build_step_message([[0.1]], [1.0], [false], [{"is_success": true}])
	h.assert_eq(step["type"], "step", "step type")
	h.assert_eq(step["reward"], [1.0], "step reward")
	h.assert_eq(step["done"], [false], "step done")
	h.assert_eq(step["info"], [{"is_success": true}], "step info")

	var reset = s.build_reset_message([[0.2]])
	h.assert_eq(reset["type"], "reset", "reset type")

	var d = s.extract_action_dict([3.0], {"move": {"size": 5, "action_type": "discrete"}})
	h.assert_eq(d["move"], 3, "discrete action index")

	var c = s.extract_action_dict([0.5, -0.5], {"move": {"size": 2, "action_type": "continuous"}})
	h.assert_eq(c["move"], [0.5, -0.5], "continuous action vector")

	# agent_policy_names: one entry per training agent, in order. Default (multi_policy=false)
	# reads policy_name; policy_group is ignored.
	var a := PolicyStub.new()
	a.policy_group = "seeker"  # ignored while multi_policy is false
	var b := PolicyStub.new()
	b.policy_name = "hider"
	s.agents_training = [a, b]
	var info = s.build_env_info_message()
	h.assert_eq(info["type"], "env_info", "env_info type")
	h.assert_eq(info["n_agents"], 2, "env_info n_agents")
	h.assert_eq(info["agent_policy_names"], ["shared_policy", "hider"], "agent_policy_names (shared)")

	# multi_policy=true -> the distinct policy_group is honored (#73).
	s.multi_policy = true
	var info2 = s.build_env_info_message()
	h.assert_eq(info2["agent_policy_names"], ["seeker", "hider"], "agent_policy_names (multi_policy)")

	a.free()
	b.free()
	s.free()
	h.finish(self)
