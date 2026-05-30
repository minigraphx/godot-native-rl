extends SceneTree

const Harness = preload("res://test/harness.gd")
const RewardTerm = preload("res://reward/terms/reward_term.gd")
const StepPenaltyTerm = preload("res://reward/terms/step_penalty_term.gd")
const AliveBonusTerm = preload("res://reward/terms/alive_bonus_term.gd")

func _initialize() -> void:
	var h := Harness.new()

	var penalty := StepPenaltyTerm.new(0.001)
	h.assert_eq(penalty.evaluate(null), -0.001, "step penalty is negative amount")
	h.assert_eq(penalty.evaluate(null), -0.001, "step penalty is constant across steps")

	var alive := AliveBonusTerm.new(0.01)
	h.assert_eq(alive.evaluate(null), 0.01, "alive bonus is positive amount")

	# Base no-op hooks must not crash and must contribute nothing.
	penalty.on_event("anything")
	penalty.reset()
	h.assert_eq(penalty.evaluate(null), -0.001, "on_event/reset do not change a constant term")

	h.finish(self)
