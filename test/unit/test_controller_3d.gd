extends SceneTree

const Harness = preload("res://test/harness.gd")
const StubAgent3D = preload("res://test/unit/stub_agent_3d.gd")

class StubSource:
	var amount: float
	func _init(a: float) -> void:
		amount = a
	func evaluate(_ctx) -> float:
		return amount

func _initialize() -> void:
	var h := Harness.new()
	var a = StubAgent3D.new()

	# obs_space derived from the stub's get_obs via the shared core helper
	h.assert_eq(a.get_obs_space(), {"obs": {"size": [3], "space": "box"}}, "3D get_obs_space shape")

	# forwarding properties read/write through to the core
	a.done = true
	h.assert_true(a.get_done(), "3D done forwards to core (get_done)")
	a.set_done_false()
	h.assert_true(not a.done, "3D set_done_false clears forwarded done")

	a.reward = 2.0
	h.assert_true(absf(a.reward - 2.0) < 1e-6, "3D reward forwards")
	a.zero_reward()
	h.assert_eq(a.reward, 0.0, "3D zero_reward via core")

	a.needs_reset = true
	a.reset()
	h.assert_true(not a.needs_reset, "3D reset clears needs_reset via core")

	# accumulate_reward delegates to core (no adapters collected without _ready)
	a.reward = 0.0
	a.reward_source = StubSource.new(0.75)
	a.accumulate_reward()
	h.assert_true(absf(a.reward - 0.75) < 1e-6, "3D accumulate_reward sums reward_source via core")

	# step threshold flows through the 3D wrapper into core.step(reset_after)
	a.reset_after = 3
	a.reset()
	a.set_done_false()
	a._physics_process(0.0)
	a._physics_process(0.0)
	a._physics_process(0.0)
	h.assert_true(not a.done, "3D not done at exactly reset_after steps")
	a._physics_process(0.0)
	h.assert_true(a.done, "3D done once n_steps > reset_after")
	h.assert_true(a.needs_reset, "3D needs_reset set past threshold")

	a.free()
	h.finish(self)
