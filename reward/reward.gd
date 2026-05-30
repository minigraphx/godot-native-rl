class_name Reward
extends RefCounted

var _terms: Array   # Array of RewardTerm

func _init(terms: Array) -> void:
	_terms = terms

func evaluate(ctx) -> float:
	var total := 0.0
	for term in _terms:
		total += term.evaluate(ctx)
	return total

func trigger_event(event_name: String) -> void:
	for term in _terms:
		term.on_event(event_name)

func reset() -> void:
	for term in _terms:
		term.reset()
