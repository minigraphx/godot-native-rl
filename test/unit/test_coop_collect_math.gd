extends SceneTree

# Headless unit tests for the pure Cooperative Collect helpers (MA-POCA scaffold, #30).

const Harness = preload("res://test/harness.gd")
const M = preload("res://examples/coop_collect/coop_collect_math.gd")

func _approx(h, got: Array, want: Array, msg: String) -> void:
	h.assert_eq(got.size(), want.size(), msg + " (size)")
	for i in range(got.size()):
		h.assert_true(absf(float(got[i]) - float(want[i])) < 1e-5, "%s [%d] got %f want %f" % [msg, i, got[i], want[i]])

func _initialize() -> void:
	var h := Harness.new()

	# own_pos_obs: normalized by arena.
	_approx(h, M.own_pos_obs(Vector2(500, 300), Vector2(1000, 600)), [0.5, 0.5], "own_pos_obs centre")

	# rel_obs: normalized + clamped.
	_approx(h, M.rel_obs(Vector2(0, 0), Vector2(50, -100), 100.0), [0.5, -1.0], "rel_obs clamp")

	# nearest_agent_dist.
	h.assert_true(absf(M.nearest_agent_dist(Vector2(0, 0), [Vector2(3, 4), Vector2(10, 0)]) - 5.0) < 1e-5, "nearest_agent_dist")
	h.assert_true(is_inf(M.nearest_agent_dist(Vector2(0, 0), [])), "nearest_agent_dist no agents -> INF")

	# collect_step: an in-range uncollected item is collected; already-collected skipped.
	var items := [Vector2(0, 0), Vector2(100, 0), Vector2(500, 500)]
	var collected := [false, false, false]
	var newly := M.collect_step(items, collected, [Vector2(10, 0)], 40.0)  # only item 0 within 40
	h.assert_eq(newly, 1, "collect_step newly=1")
	h.assert_eq(collected, [true, false, false], "collect_step flags")
	# Re-running with the same agent doesn't re-collect item 0.
	h.assert_eq(M.collect_step(items, collected, [Vector2(10, 0)], 40.0), 0, "collect_step no double-collect")

	# Two agents collecting different items in one step.
	var c2 := [false, false, false]
	h.assert_eq(M.collect_step(items, c2, [Vector2(0, 0), Vector2(100, 0)], 20.0), 2, "two agents collect two items")

	# team_step_reward: value per item minus time penalty.
	h.assert_true(absf(M.team_step_reward(2, 1.0, 0.01) - 1.99) < 1e-5, "team_step_reward")
	h.assert_true(absf(M.team_step_reward(0, 1.0, 0.01) - (-0.01)) < 1e-5, "team_step_reward idle penalty")

	# all_collected.
	h.assert_true(M.all_collected([true, true]), "all_collected true")
	h.assert_true(not M.all_collected([true, false]), "all_collected false")

	# item_block: rel + flag.
	_approx(h, M.item_block(Vector2(0, 0), Vector2(100, 0), false, 100.0), [1.0, 0.0, 0.0], "item_block uncollected")
	_approx(h, M.item_block(Vector2(0, 0), Vector2(100, 0), true, 100.0), [1.0, 0.0, 1.0], "item_block collected")

	# assemble_obs: own(2) + teammate(2) + 2 items(3 each) = 10.
	var obs := M.assemble_obs([0.5, 0.5], [0.1, -0.1], [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
	h.assert_eq(obs.size(), 10, "assemble_obs length 4 + 3*2")
	_approx(h, obs, [0.5, 0.5, 0.1, -0.1, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], "assemble_obs order")

	h.finish(self)
