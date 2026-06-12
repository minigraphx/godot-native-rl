extends SceneTree
# Unit tests for the GridWorld example (#48): pure cell math + game state + agent contract.
# GridSensor2D's query path is covered by the integration smoke (needs a live physics space).

const Harness = preload("res://test/harness.gd")
const Game = preload("res://examples/gridworld/gridworld_game.gd")

func _initialize() -> void:
	var h = Harness.new()

	# --- Pure helpers ---
	h.assert_eq(Game.cell_to_pos(Vector2i(0, 0), 40.0), Vector2(20, 20), "cell 0,0 center")
	h.assert_eq(Game.cell_to_pos(Vector2i(2, 1), 40.0), Vector2(100, 60), "cell 2,1 center")

	var cells := Vector2i(8, 8)
	h.assert_eq(Game.step_cell(Vector2i(4, 4), 0, cells), Vector2i(4, 4), "action 0 = stay")
	h.assert_eq(Game.step_cell(Vector2i(4, 4), 1, cells), Vector2i(4, 3), "action 1 = up")
	h.assert_eq(Game.step_cell(Vector2i(4, 4), 2, cells), Vector2i(4, 5), "action 2 = down")
	h.assert_eq(Game.step_cell(Vector2i(4, 4), 3, cells), Vector2i(3, 4), "action 3 = left")
	h.assert_eq(Game.step_cell(Vector2i(4, 4), 4, cells), Vector2i(5, 4), "action 4 = right")
	h.assert_eq(Game.step_cell(Vector2i(0, 0), 1, cells), Vector2i(0, 0), "clamped at top")
	h.assert_eq(Game.step_cell(Vector2i(7, 7), 4, cells), Vector2i(7, 7), "clamped at right")

	var gv := Game.goal_vector(Vector2i(0, 0), Vector2i(4, 2), cells)
	h.assert_eq(gv.size(), 2, "goal vector 2-dim")
	h.assert_eq(gv[0], 0.5, "goal dx normalized")
	h.assert_eq(gv[1], 0.25, "goal dy normalized")

	# --- Game state (no physics needed for cell logic) ---
	var game = Game.new()
	get_root().add_child(game)
	game.set_state_for_test(Vector2i(1, 1), Vector2i(2, 1), [Vector2i(1, 2)])
	h.assert_true(not game.at_goal(), "not at goal initially")
	game.move_agent(4)  # right -> onto goal
	h.assert_true(game.at_goal(), "reached goal")
	h.assert_true(game.resolve_terminal(), "terminal resolves")
	h.assert_eq(game.goals_reached, 1, "goal counted")

	game.set_state_for_test(Vector2i(1, 1), Vector2i(5, 5), [Vector2i(1, 2)])
	game.move_agent(2)  # down -> onto pit
	h.assert_true(game.at_pit(), "hit pit")
	h.assert_true(game.resolve_terminal(), "pit terminal resolves")
	h.assert_eq(game.pits_hit, 1, "pit counted")

	# Seeded resets never overlap goal/pits/agent.
	game.seed_rng(42)
	for i in range(30):
		game.reset_episode()
		h.assert_true(not game.at_goal() and not game.at_pit(), "spawn %d clear of terminals" % i)

	h.finish(self)
