extends SceneTree

const Harness = preload("res://test/harness.gd")
const ChaseGameScript = preload("res://examples/chase_the_target/chase_game.gd")

func _initialize() -> void:
	var h := Harness.new()
	var g := ChaseGameScript.new()
	g.arena_size = Vector2(1000, 600)
	h.assert_eq(g.catches, 0, "catches starts at 0")
	g.relocate_target()
	g.relocate_target()
	h.assert_eq(g.catches, 2, "relocate_target increments catches")
	g.free()
	h.finish(self)
