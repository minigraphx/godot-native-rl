extends SceneTree

const Harness = preload("res://test/harness.gd")
const GoalMath = preload("res://addons/godot_native_rl/sensors/goal_math.gd")

func _initialize() -> void:
	var h := Harness.new()

	h.assert_eq(GoalMath.one_hot(0, 3), [1.0, 0.0, 0.0], "one-hot at 0")
	h.assert_eq(GoalMath.one_hot(2, 3), [0.0, 0.0, 1.0], "one-hot at 2")
	h.assert_eq(GoalMath.one_hot(1, 3, true), [0.0, 0.0, 0.0], "blind -> all zeros")
	h.assert_eq(GoalMath.one_hot(5, 3), [0.0, 0.0, 0.0], "out-of-range -> all zeros")
	h.assert_eq(GoalMath.one_hot(-1, 3), [0.0, 0.0, 0.0], "negative -> all zeros")

	h.finish(self)
