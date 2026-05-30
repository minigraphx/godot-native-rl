extends SceneTree

const Harness = preload("res://test/harness.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")

class StubSource:
	var amount: float
	var last_ctx = null
	func _init(a: float) -> void:
		amount = a
	func evaluate(ctx) -> float:
		last_ctx = ctx
		return amount

class StubAdapter:
	var amount: float
	func _init(a: float) -> void:
		amount = a
	func drain() -> float:
		return amount

func _initialize() -> void:
	var h := Harness.new()
	var c = NcnnControllerCore.new()

	# step() / reset_after threshold (godot_rl: done once n_steps > reset_after)
	c.step(3)
	h.assert_eq(c.n_steps, 1, "step increments n_steps")
	h.assert_true(not c.done, "not done before threshold")
	c.step(3)
	c.step(3)
	h.assert_true(not c.done, "not done at n_steps == reset_after")
	c.step(3)
	h.assert_eq(c.n_steps, 4, "n_steps past threshold")
	h.assert_true(c.done, "done once n_steps > reset_after")
	h.assert_true(c.needs_reset, "needs_reset set past threshold")

	# reset()
	c.reset()
	h.assert_eq(c.n_steps, 0, "reset zeroes n_steps")
	h.assert_true(not c.needs_reset, "reset clears needs_reset")

	# reset_if_done()
	c.done = true
	c.n_steps = 5
	c.reset_if_done()
	h.assert_eq(c.n_steps, 0, "reset_if_done resets when done")

	# done helpers
	c.done = true
	h.assert_true(c.get_done(), "get_done reflects done")
	c.set_done_false()
	h.assert_true(not c.get_done(), "set_done_false clears done")

	# heuristic
	c.set_heuristic("noop")
	h.assert_eq(c.heuristic, "noop", "set_heuristic stores value")

	# zero_reward
	c.reward = 5.0
	c.zero_reward()
	h.assert_eq(c.reward, 0.0, "zero_reward clears reward")

	# accumulate(): reward_source + adapters, ctx passed through
	c.reward = 0.0
	var src := StubSource.new(1.5)
	c.reward_source = src
	var ctx := RefCounted.new()
	c.accumulate([StubAdapter.new(0.25), StubAdapter.new(0.1)], ctx)
	h.assert_true(absf(c.reward - 1.85) < 1e-6, "accumulate sums reward_source + adapters")
	h.assert_eq(src.last_ctx, ctx, "accumulate passes ctx to reward_source.evaluate")

	# accumulate() with null reward_source: adapters only
	c.reward = 0.0
	c.reward_source = null
	c.accumulate([StubAdapter.new(0.5)], ctx)
	h.assert_true(absf(c.reward - 0.5) < 1e-6, "accumulate works with null reward_source")

	# obs_space_from_obs() static
	var space := NcnnControllerCore.obs_space_from_obs({"obs": [0.0, 0.0, 0.0]})
	h.assert_eq(space, {"obs": {"size": [3], "space": "box"}}, "obs_space_from_obs shape")

	h.finish(self)
