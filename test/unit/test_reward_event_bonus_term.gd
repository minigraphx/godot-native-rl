extends SceneTree

const Harness = preload("res://test/harness.gd")
const EventBonusTerm = preload("res://addons/godot_native_rl/reward/terms/event_bonus_term.gd")

func _initialize() -> void:
	var h := Harness.new()

	var term := EventBonusTerm.new("caught", 1.0)

	# No event yet -> contributes nothing.
	h.assert_eq(term.evaluate(null), 0.0, "no bonus before event")

	# A non-matching event does nothing.
	term.on_event("other")
	h.assert_eq(term.evaluate(null), 0.0, "non-matching event pays nothing")

	# Matching event -> paid on the NEXT evaluate, exactly once.
	term.on_event("caught")
	h.assert_eq(term.evaluate(null), 1.0, "matching event pays bonus once")
	h.assert_eq(term.evaluate(null), 0.0, "bonus not paid twice")

	# Two events before an evaluate accumulate.
	term.on_event("caught")
	term.on_event("caught")
	h.assert_eq(term.evaluate(null), 2.0, "two events accumulate")

	# reset() clears a pending (queued-but-unpaid) bonus.
	term.on_event("caught")
	term.reset()
	h.assert_eq(term.evaluate(null), 0.0, "reset clears pending bonus")

	h.finish(self)
