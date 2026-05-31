extends SceneTree

const Harness = preload("res://test/harness.gd")
const RoverAgentScript = preload("res://examples/rover_3d/rover_agent.gd")

func _initialize() -> void:
	var h := Harness.new()
	var a = RoverAgentScript.new()

	# action_index_to_motion: 0 idle, 1 forward, 2 turn-left (+yaw toward -X), 3 turn-right (-yaw toward +X)
	h.assert_eq(a.action_index_to_motion(0, 6.0, 2.5), {"forward": 0.0, "yaw": 0.0}, "idle")
	h.assert_eq(a.action_index_to_motion(1, 6.0, 2.5), {"forward": 6.0, "yaw": 0.0}, "forward")
	h.assert_eq(a.action_index_to_motion(2, 6.0, 2.5), {"forward": 0.0, "yaw": 2.5}, "turn left (+yaw toward -X)")
	h.assert_eq(a.action_index_to_motion(3, 6.0, 2.5), {"forward": 0.0, "yaw": -2.5}, "turn right (-yaw toward +X)")

	# compute_goal_obs(bearing, dist, max_dist) -> [sin, cos, clamped distance]
	var goal_obs: Array = a.compute_goal_obs(0.0, 10.0, 40.0)
	h.assert_eq(goal_obs.size(), 3, "goal obs has 3 elements")
	h.assert_true(absf(goal_obs[0] - 0.0) < 1e-6, "sin(0)=0")
	h.assert_true(absf(goal_obs[1] - 1.0) < 1e-6, "cos(0)=1")
	h.assert_true(absf(goal_obs[2] - 0.25) < 1e-6, "distance normalized 10/40")
	var far: Array = a.compute_goal_obs(PI / 2.0, 100.0, 40.0)
	h.assert_true(absf(far[0] - 1.0) < 1e-6, "sin(PI/2)=1")
	h.assert_true(absf(far[2] - 1.0) < 1e-6, "distance clamps to 1.0")

	# compose_obs concatenates rays + goal in order
	var composed: Array = a.compose_obs([0.1, 0.2, 0.3, 0.4, 0.5], [0.0, 1.0, 0.25])
	h.assert_eq(composed.size(), 8, "composed obs length = rays(5) + goal(3)")
	h.assert_true(absf(composed[0] - 0.1) < 1e-6, "rays come first")
	h.assert_true(absf(composed[5] - 0.0) < 1e-6, "goal obs appended after rays")

	# action space
	h.assert_eq(a.get_action_space(), {"move": {"size": 4, "action_type": "discrete"}}, "action space")

	# get_obs with no game/sensor -> a correctly-sized zero vector (no crash)
	var obs_dict: Dictionary = a.get_obs()
	h.assert_true("obs" in obs_dict, "get_obs returns an obs key")
	h.assert_eq(obs_dict["obs"].size(), 8, "fallback obs size = default rays(5) + goal(3)")
	var any_nonzero := false
	for v in obs_dict["obs"]:
		if absf(v) > 1e-9:
			any_nonzero = true
	h.assert_true(not any_nonzero, "fallback obs is all zeros")

	# set_action stores a valid discrete index
	a.set_action({"move": 3})
	h.assert_eq(a._action_index, 3, "set_action stores the index")

	a.free()
	h.finish(self)
