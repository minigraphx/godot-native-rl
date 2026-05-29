extends SceneTree

const Harness = preload("res://test/harness.gd")
const ChaseGameScript = preload("res://examples/chase_the_target/chase_game.gd")

func _initialize() -> void:
	var h := Harness.new()
	var g := ChaseGameScript.new()
	g.arena_size = Vector2(1000, 600)

	h.assert_eq(g.clamp_to_bounds(Vector2(-50, 700)), Vector2(0, 600), "clamp low/high")
	h.assert_eq(g.clamp_to_bounds(Vector2(500, 300)), Vector2(500, 300), "clamp inside unchanged")

	h.assert_true(absf(g.max_distance() - Vector2(1000, 600).length()) < 0.001, "max_distance diagonal")

	g.seed_rng(123)
	var all_in_bounds := true
	for _i in range(200):
		var p: Vector2 = g.random_position()
		if p.x < 0.0 or p.x > 1000.0 or p.y < 0.0 or p.y > 600.0:
			all_in_bounds = false
	h.assert_true(all_in_bounds, "random_position within bounds")

	g.free()
	h.finish(self)
