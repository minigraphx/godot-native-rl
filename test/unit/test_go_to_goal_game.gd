extends SceneTree

# Headless pure-logic unit tests for the GoToGoal example (#386 — the GoalSensor worked example):
# the game's signaled-goal roll, per-goal reach counters, and the two scored events. NO physics /
# scene: everything is driven through the set_state_for_test() seam + resolve_touches() (like
# gridworld_game.gd:125), so the game is exercised as a bare Node2D that never enters the tree.

const Harness = preload("res://test/harness.gd")
const GoToGoalGameScript = preload("res://examples/go_to_goal/go_to_goal_game.gd")

func _initialize() -> void:
	var h := Harness.new()

	var game = GoToGoalGameScript.new()
	game.seed_rng(1234)
	game.reset_positions()

	# --- K = 3 targets; the signaled goal id is always a valid index ---
	h.assert_eq(game.goals_reached_by_id.size(), 3, "per-goal counter has one slot per target (K=3)")
	var g0: int = game.get_current_goal()
	h.assert_true(g0 >= 0 and g0 < 3, "get_current_goal() in [0, K)")
	h.assert_eq(game.goals_reached, 0, "no goals reached before any resolution")
	h.assert_eq(game.wrong_touches, 0, "no wrong touches before any resolution")

	# --- seeded determinism: two games seeded identically lay out + roll the same ---
	var game_b = GoToGoalGameScript.new()
	game_b.seed_rng(1234)
	game_b.reset_positions()
	h.assert_eq(game_b.get_current_goal(), g0, "seeded roll_goal deterministic across instances")
	h.assert_eq(game_b.get_target_pos(0), game.get_target_pos(0), "seeded target layout deterministic")

	# --- reaching the SIGNALED target scores a goal, credits its id, and re-rolls the goal ---
	var signaled: int = 1
	var targets := [Vector2(100, 100), Vector2(500, 300), Vector2(900, 500)]
	game.set_state_for_test(targets[signaled], targets, signaled)
	h.assert_eq(game.get_current_goal(), signaled, "set_state_for_test installs the goal")
	h.assert_true(game.signaled_distance() < 1e-6, "agent placed ON the signaled target (distance 0)")
	var fired: bool = game.resolve_touches()
	h.assert_true(fired, "resolve_touches reports the scored reach")
	h.assert_eq(game.goals_reached, 1, "reaching the signaled target increments goals_reached")
	h.assert_eq(int(game.goals_reached_by_id[signaled]), 1, "per-goal counter credits the reached id")
	var new_goal: int = game.get_current_goal()
	h.assert_true(new_goal >= 0 and new_goal < 3, "current_goal re-rolled into [0, K)")
	# targets relocated on a reach (all moved off their old spots)
	h.assert_true(game.get_target_pos(signaled) != targets[signaled], "reach relocates the targets")

	# --- touching a WRONG target penalizes only: wrong_touches++ , goals_reached unchanged ---
	var game2 = GoToGoalGameScript.new()
	game2.seed_rng(7)
	var t2 := [Vector2(100, 100), Vector2(500, 300), Vector2(900, 500)]
	# signaled goal = 0, but the agent sits on target 2 (a wrong target)
	game2.set_state_for_test(t2[2], t2, 0)
	var goals_before: int = game2.goals_reached
	var wrong_pos_before: Vector2 = game2.get_target_pos(2)
	var fired2: bool = game2.resolve_touches()
	h.assert_true(fired2, "resolve_touches reports the wrong-target contact")
	h.assert_eq(game2.wrong_touches, 1, "wrong touch increments wrong_touches")
	h.assert_eq(game2.goals_reached, goals_before, "wrong touch does NOT increment goals_reached")
	h.assert_eq(int(game2.goals_reached_by_id[0]), 0, "wrong touch does NOT credit any goal id")
	h.assert_eq(game2.get_current_goal(), 0, "wrong touch does NOT re-roll the goal")
	h.assert_true(game2.get_target_pos(2) != wrong_pos_before, "wrong touch relocates only that target")

	# --- no contact: agent far from every target -> nothing fires ---
	var game3 = GoToGoalGameScript.new()
	game3.seed_rng(3)
	var t3 := [Vector2(50, 50), Vector2(500, 300), Vector2(950, 550)]
	game3.set_state_for_test(Vector2(500, 50), t3, 1)  # far from all
	var fired3: bool = game3.resolve_touches()
	h.assert_true(not fired3, "no contact -> resolve_touches returns false")
	h.assert_eq(game3.goals_reached, 0, "no contact -> no goal")
	h.assert_eq(game3.wrong_touches, 0, "no contact -> no wrong touch")

	# --- max_distance is the arena diagonal (reward shaping bound) ---
	h.assert_true(absf(game.max_distance() - game.arena_size.length()) < 1e-6,
		"max_distance = arena diagonal")

	game.free()
	game_b.free()
	game2.free()
	game3.free()
	h.finish(self)
