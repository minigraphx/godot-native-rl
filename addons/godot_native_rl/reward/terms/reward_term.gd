class_name RewardTerm
extends RefCounted

# Sum-of-terms reward model. Each term contributes a scalar per step via evaluate(),
# may react immediately to a named event via on_event(), and may clear transient
# state at episode boundaries via reset(). All three default to no-ops.
func evaluate(_ctx) -> float:
	return 0.0

func on_event(_event_name: String) -> void:
	pass

func reset() -> void:
	pass
