extends SceneTree

const Harness = preload("res://test/harness.gd")
const Reward = preload("res://addons/godot_native_rl/reward/reward.gd")
const StepPenaltyTerm = preload("res://addons/godot_native_rl/reward/terms/step_penalty_term.gd")
const AliveBonusTerm = preload("res://addons/godot_native_rl/reward/terms/alive_bonus_term.gd")
const EventBonusTerm = preload("res://addons/godot_native_rl/reward/terms/event_bonus_term.gd")

func _initialize() -> void:
	var h := Harness.new()

	var penalty := StepPenaltyTerm.new(0.001)
	var alive := AliveBonusTerm.new(0.01)
	var bonus := EventBonusTerm.new("caught", 1.0)
	var reward := Reward.new([penalty, alive, bonus])

	# Sum of terms: -0.001 + 0.01 + 0.0 = 0.009
	h.assert_eq(reward.evaluate(null), 0.009, "evaluate sums all terms")

	# trigger_event routes to every term's on_event -> bonus paid next evaluate.
	reward.trigger_event("caught")
	h.assert_eq(reward.evaluate(null), 1.009, "event bonus added on next evaluate")
	h.assert_eq(reward.evaluate(null), 0.009, "bonus not repeated")

	# reset routes to every term's reset -> clears the pending bonus.
	reward.trigger_event("caught")
	reward.reset()
	h.assert_eq(reward.evaluate(null), 0.009, "reset clears pending event bonus")

	h.finish(self)
