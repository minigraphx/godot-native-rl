extends SceneTree

const Harness = preload("res://test/harness.gd")
const Stub = preload("res://test/unit/stub_agent.gd")

func _initialize() -> void:
	var h := Harness.new()
	var a := Stub.new()

	# get_obs_space is derived from get_obs (5 floats -> box of size [5])
	var space = a.get_obs_space()
	h.assert_eq(space["obs"]["size"], [5], "obs_space size")
	h.assert_eq(space["obs"]["space"], "box", "obs_space type")

	# zero_reward resets accumulated reward
	a.reward = 5.0
	a.zero_reward()
	h.assert_eq(a.reward, 0.0, "zero_reward")

	# set_done_false clears done
	a.done = true
	a.set_done_false()
	h.assert_eq(a.get_done(), false, "set_done_false")

	# reset_after boundary: exactly reset_after steps must NOT trigger; one more must.
	a.reset_after = 3
	a.reset()
	for _i in range(3):
		a._physics_process(0.0)
	h.assert_true(not a.needs_reset, "no reset at exactly reset_after steps")
	a._physics_process(0.0)
	h.assert_true(a.needs_reset, "needs_reset after exceeding reset_after steps")

	a.free()
	h.finish(self)
