class_name RewardBuilder
extends RefCounted

const ProgressShapingTerm = preload("res://reward/terms/progress_shaping_term.gd")
const EventBonusTerm = preload("res://reward/terms/event_bonus_term.gd")
const StepPenaltyTerm = preload("res://reward/terms/step_penalty_term.gd")
const AliveBonusTerm = preload("res://reward/terms/alive_bonus_term.gd")
const RewardScript = preload("res://reward/reward.gd")

var _terms: Array

func _init(terms: Array = []) -> void:
	_terms = terms

func term_count() -> int:
	return _terms.size()

# Copy-on-write: return a NEW builder with `term` appended; leave self unchanged.
func _with(term) -> RefCounted:
	var next := _terms.duplicate()
	next.append(term)
	return get_script().new(next)

func add_progress_shaping(value_fn: Callable, scale, rebase_on: Array = []) -> RefCounted:
	return _with(ProgressShapingTerm.new(value_fn, scale, rebase_on))

func add_event_bonus(event_name: String, amount: float) -> RefCounted:
	return _with(EventBonusTerm.new(event_name, amount))

func add_step_penalty(amount: float) -> RefCounted:
	return _with(StepPenaltyTerm.new(amount))

func add_alive_bonus(amount: float) -> RefCounted:
	return _with(AliveBonusTerm.new(amount))

func build() -> RefCounted:
	return RewardScript.new(_terms.duplicate())
