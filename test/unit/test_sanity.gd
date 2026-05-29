extends SceneTree

const Harness = preload("res://test/harness.gd")

func _initialize() -> void:
	var h := Harness.new()
	h.assert_eq(1 + 1, 2, "harness arithmetic")
	h.assert_eq([1, 2], [1, 2], "harness array compare")
	h.assert_true(1 < 2, "harness assert_true")
	h.finish(self)
