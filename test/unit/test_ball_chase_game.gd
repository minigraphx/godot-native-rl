extends SceneTree
# Pure-helper tests for ball_chase_game.gd (continuous chase env).

const Harness = preload("res://test/harness.gd")
const GameScript = preload("res://examples/ball_chase/ball_chase_game.gd")

func _initialize() -> void:
	var h := Harness.new()

	var game := GameScript.new()
	game.arena_size = Vector2(1000, 600)

	# clamp_to_bounds keeps positions inside the arena
	h.assert_eq(game.clamp_to_bounds(Vector2(-50, 700)), Vector2(0, 600), "clamp to arena bounds")
	h.assert_eq(game.clamp_to_bounds(Vector2(500, 300)), Vector2(500, 300), "in-bounds unchanged")

	# max_distance is the arena diagonal
	h.assert_true(absf(game.max_distance() - Vector2(1000, 600).length()) < 1e-4, "max_distance = diagonal")

	# move_agent integrates continuous thrust * delta and clamps
	game.set_agent_pos_for_test(Vector2(500, 300))
	game.move_agent(Vector2(100, 0), 0.5)   # +50 x
	h.assert_eq(game.get_agent_pos(), Vector2(550, 300), "move integrates thrust*delta")

	# relocate_target increments reaches and emits target_caught
	game.set_target_pos_for_test(Vector2(10, 10))
	var caught := [false]
	game.target_caught.connect(func(): caught[0] = true)
	var before: int = game.reaches
	game.relocate_target()
	h.assert_eq(game.reaches, before + 1, "relocate_target increments reaches")
	h.assert_true(caught[0], "relocate_target emits target_caught")

	game.free()
	h.finish(self)
