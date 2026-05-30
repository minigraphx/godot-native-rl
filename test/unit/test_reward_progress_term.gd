extends SceneTree

const Harness = preload("res://test/harness.gd")
const ProgressShapingTerm = preload("res://reward/terms/progress_shaping_term.gd")

# Mutable holder so a Callable can read a changing "distance".
class Holder:
	extends RefCounted
	var value := 0.0
	func dist() -> float:
		return value

func _initialize() -> void:
	var h := Harness.new()
	var holder := Holder.new()

	# Baseline primed at construction = 10.0, scale = fixed 100.0.
	holder.value = 10.0
	var term := ProgressShapingTerm.new(holder.dist, 100.0, ["caught"])

	# Move closer: prev=10, cur=8 -> (10-8)/100 = 0.02
	holder.value = 8.0
	h.assert_eq(term.evaluate(null), 0.02, "progress toward target is positive")

	# Move further: prev=8, cur=9 -> (8-9)/100 = -0.01
	holder.value = 9.0
	h.assert_eq(term.evaluate(null), -0.01, "moving away is negative")

	# Rebase on event: value jumps to 50 (new target), event resets baseline to 50.
	holder.value = 50.0
	term.on_event("caught")
	# Next step prev should be 50, not 9 -> (50-48)/100 = 0.02 (the jump is NOT scored).
	holder.value = 48.0
	h.assert_eq(term.evaluate(null), 0.02, "rebase prevents scoring the relocate jump")

	# Non-matching event does not rebase.
	holder.value = 40.0
	term.on_event("unrelated")
	# prev is 48 (from previous evaluate), cur 40 -> (48-40)/100 = 0.08
	h.assert_eq(term.evaluate(null), 0.08, "non-matching event does not rebase")

	# reset() rebases baseline to current value (40) -> next step (40-30)/100 = 0.10
	term.reset()
	holder.value = 30.0
	h.assert_eq(term.evaluate(null), 0.10, "reset rebases baseline")

	# Callable scale is supported.
	holder.value = 10.0
	var scale_holder := Holder.new()
	scale_holder.value = 200.0
	var term2 := ProgressShapingTerm.new(holder.dist, scale_holder.dist, [])
	holder.value = 8.0
	h.assert_eq(term2.evaluate(null), 0.01, "callable scale: (10-8)/200 = 0.01")

	h.finish(self)
