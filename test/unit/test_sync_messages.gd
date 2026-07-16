extends SceneTree

const Harness = preload("res://test/harness.gd")
const SyncScript = preload("res://addons/godot_native_rl/sync.gd")
const PolicyStub = preload("res://test/unit/policy_name_stub.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s := SyncScript.new()

	var step = s.build_step_message([[0.1]], [1.0], [false], [false], [{"is_success": true}])
	h.assert_eq(step["truncated"], [false], "step message carries the additive truncated array (#12)")
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

	# --- #379 wire-level regression: a horizon end must reach the wire as truncated=[true], even
	# though the agent calls reset() in the SAME frame the horizon fires (before the Sync reads the
	# step). HorizonAgentStub wraps a REAL NcnnControllerCore and reproduces an example agent's
	# _physics_process at the horizon (step() then in-frame reset(), see chase_agent.gd:137). The
	# Sync's read functions run in their documented order (truncated BEFORE done — done's read
	# clears the pair). If reset() cleared truncated, truncs would be [false] and the #12 split would
	# silently no-op on every horizon end.
	var HorizonAgentStub = preload("res://test/unit/horizon_agent_stub.gd")
	var hz := HorizonAgentStub.new()
	hz.drive_horizon_then_reset(1)
	var s2 := SyncScript.new()
	s2.agents_training = [hz]
	var truncs = s2._get_truncated_from_agents()  # must be read first
	var dones = s2._get_done_from_agents()
	h.assert_eq(truncs, [true], "#379: horizon truncated=[true] reaches the wire despite the in-frame reset()")
	h.assert_eq(dones, [true], "#379: horizon done=[true] on the wire (0.8.2 semantics unchanged)")
	h.assert_true(not hz.get_truncated(), "#379: the paired set_done_false cleared truncated after the read")
	h.assert_true(not hz.get_done(), "#379: done cleared after the read (no leak into next episode)")
	hz.free()
	s2.free()

	# #385: additive action_mask field — present iff an agent implements get_action_mask()
	var MaskStub = preload("res://test/unit/action_mask_stub.gd")  # returns {"move":[1,0,1,1,1]}
	var ma := MaskStub.new()
	s.agents_training = [ma]
	var masks := s._get_action_masks_from_agents()
	h.assert_eq(masks, [{"move": [1, 0, 1, 1, 1]}], "#385: collects per-agent action_mask")
	var step_m := s.build_step_message([[0.1]], [1.0], [false], [false], [{}], masks)
	h.assert_eq(step_m["action_mask"], [{"move": [1, 0, 1, 1, 1]}], "#385: step message carries action_mask when present")
	var step_none := s.build_step_message([[0.1]], [1.0], [false], [false], [{}], [])
	h.assert_true(not step_none.has("action_mask"), "#385: action_mask key omitted when empty (backward-compatible)")
	# reset message mirrors the same additive contract (MaskablePPO needs a mask for the first obs)
	var reset_m := s.build_reset_message([[0.2]], masks)
	h.assert_eq(reset_m["action_mask"], [{"move": [1, 0, 1, 1, 1]}], "#385: reset message carries action_mask when present")
	var reset_none := s.build_reset_message([[0.2]])
	h.assert_true(not reset_none.has("action_mask"), "#385: reset action_mask key omitted when empty (backward-compatible)")
	# duck-typed: an agent without get_action_mask() contributes {}
	var plain := Node.new()
	s.agents_training = [ma, plain]
	h.assert_eq(s._get_action_masks_from_agents(), [{"move": [1, 0, 1, 1, 1]}, {}], "#385: agents without get_action_mask() report {}")
	plain.free()
	ma.free()

	a.free()
	b.free()
	s.free()
	h.finish(self)
