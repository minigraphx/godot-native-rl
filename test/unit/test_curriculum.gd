extends SceneTree
# Unit tests for the pure Curriculum stage/promotion logic (no scene dependencies).

const Harness = preload("res://test/harness.gd")
const Curriculum = preload("res://addons/godot_native_rl/training/curriculum.gd")

func _stages() -> Array:
	return [
		{"name": "easy", "params": {"touch_radius": 120.0},
			"promote": {"metric": "mean_reward", "threshold": 5.0, "window": 4, "min_episodes": 3}},
		{"name": "mid", "params": {"touch_radius": 80.0},
			"promote": {"metric": "success_rate", "threshold": 0.75, "window": 4, "min_episodes": 4}},
		{"name": "hard", "params": {"touch_radius": 40.0}},
	]

func _initialize() -> void:
	var h = Harness.new()

	var c = Curriculum.new()
	h.assert_true(c.set_stages(_stages()), "valid stages accepted")
	h.assert_eq(c.stage_count(), 3, "3 stages")
	h.assert_eq(c.stage_index(), 0, "starts at stage 0")
	h.assert_eq(c.stage_name(), "easy", "stage 0 name")
	h.assert_eq(c.current_params()["touch_radius"], 120.0, "stage 0 params")
	h.assert_true(not c.is_final(), "not final at 0")

	# min_episodes gate: 2 great episodes are not enough (needs 3)
	c.record_episode(10.0, true)
	c.record_episode(10.0, true)
	h.assert_true(not c.should_promote(), "min_episodes gate holds")
	c.record_episode(10.0, true)
	h.assert_true(c.should_promote(), "mean_reward 10 >= 5 over 3 eps promotes")

	# advance clears the window
	h.assert_true(c.advance(), "advance to stage 1")
	h.assert_eq(c.stage_index(), 1, "now stage 1")
	h.assert_true(not c.should_promote(), "fresh window after advance")

	# success_rate metric: 3/4 successes = 0.75 >= 0.75
	c.record_episode(0.0, true)
	c.record_episode(0.0, true)
	c.record_episode(0.0, false)
	c.record_episode(0.0, true)
	h.assert_true(c.should_promote(), "success_rate 0.75 promotes")
	h.assert_true(c.advance(), "advance to final")
	h.assert_true(c.is_final(), "final stage reached")

	# final stage: never promotes, advance returns false
	c.record_episode(100.0, true)
	c.record_episode(100.0, true)
	c.record_episode(100.0, true)
	c.record_episode(100.0, true)
	h.assert_true(not c.should_promote(), "final stage never promotes")
	h.assert_true(not c.advance(), "advance refuses past final")

	# rolling window: low episodes push the good ones out
	var c2 = Curriculum.new()
	c2.set_stages(_stages())
	for i in range(4):
		c2.record_episode(10.0, true)
	h.assert_true(c2.should_promote(), "window full of 10s promotes")
	for i in range(4):
		c2.record_episode(0.0, false)
	h.assert_true(not c2.should_promote(), "window rolled to 0s: no promote")

	# set_stage jump + bounds
	h.assert_true(c2.set_stage(2), "set_stage in range")
	h.assert_eq(c2.stage_index(), 2, "jumped to 2")
	h.assert_true(not c2.set_stage(7), "set_stage out of range refused")
	h.assert_eq(c2.stage_index(), 2, "index unchanged after refusal")

	# malformed stages fail loud (return false)
	var bad = Curriculum.new()
	h.assert_true(not bad.set_stages([]), "empty stages rejected")
	h.assert_true(not bad.set_stages([{"params": {}}]), "missing name rejected")
	h.assert_true(not bad.set_stages([{"name": "x"}]), "missing params rejected")
	h.assert_true(not bad.set_stages([
		{"name": "a", "params": {}, "promote": {"metric": "bogus", "threshold": 1.0, "window": 2, "min_episodes": 1}},
		{"name": "b", "params": {}},
	]), "unknown metric rejected")

	h.finish(self)
