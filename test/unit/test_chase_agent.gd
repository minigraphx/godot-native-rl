extends SceneTree

const Harness = preload("res://test/harness.gd")
const ChaseAgentScript = preload("res://examples/chase_the_target/chase_agent.gd")

func _initialize() -> void:
	var h := Harness.new()
	var a = ChaseAgentScript.new()
	var arena := Vector2(1000, 600)

	var obs: Array = a.compute_obs(Vector2(500, 300), Vector2(1000, 300), arena)
	h.assert_eq(obs.size(), 5, "obs has 5 elements")
	h.assert_true(absf(obs[0] - 0.0) < 0.001, "obs[0] centered x ~0")
	h.assert_true(absf(obs[1] - 0.0) < 0.001, "obs[1] centered y ~0")
	h.assert_true(absf(obs[2] - 1.0) < 0.001, "obs[2] unit dir x = +1 (target right)")
	h.assert_true(absf(obs[3] - 0.0) < 0.001, "obs[3] unit dir y = 0")
	h.assert_true(obs[4] > 0.0 and obs[4] <= 1.0, "obs[4] normalized distance in (0,1]")

	var obs0: Array = a.compute_obs(Vector2(10, 10), Vector2(10, 10), arena)
	h.assert_eq(obs0[2], 0.0, "obs[2] dir x = 0 at zero distance")
	h.assert_eq(obs0[3], 0.0, "obs[3] dir y = 0 at zero distance")
	var obs_far: Array = a.compute_obs(Vector2(0, 0), Vector2(1000, 600), arena)
	h.assert_true(absf(obs_far[4] - 1.0) < 0.001, "obs[4] saturates to 1.0 at max separation")

	h.assert_eq(a.action_index_to_velocity(0, 300.0), Vector2(0, 0), "idle")
	h.assert_eq(a.action_index_to_velocity(1, 300.0), Vector2(0, -300), "up")
	h.assert_eq(a.action_index_to_velocity(2, 300.0), Vector2(0, 300), "down")
	h.assert_eq(a.action_index_to_velocity(3, 300.0), Vector2(-300, 0), "left")
	h.assert_eq(a.action_index_to_velocity(4, 300.0), Vector2(300, 0), "right")

	a.step_penalty = 0.001
	a.touch_bonus = 1.0
	var r_closer: float = a.compute_step_reward(100.0, 60.0, 1000.0, false)
	h.assert_true(r_closer > 0.0, "moving closer yields positive reward")
	var r_touch: float = a.compute_step_reward(60.0, 30.0, 1000.0, true)
	h.assert_true(r_touch > 1.0, "touch adds bonus on top of progress")
	var r_farther: float = a.compute_step_reward(60.0, 90.0, 1000.0, false)
	h.assert_true(r_farther < 0.0, "moving away yields negative reward")

	h.assert_eq(a.get_action_space(), {"move": {"size": 5, "action_type": "discrete"}}, "action space")

	a.free()
	h.finish(self)
