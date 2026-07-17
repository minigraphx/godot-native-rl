extends RefCounted
# Pure goal-signal encoding (#386). one_hot(goal_id, num_goals) -> num_goals floats, 1.0 at goal_id.
# `blind` or an out-of-range goal_id -> all zeros (the goal-blind ablation + the invalid fallback).
static func one_hot(goal_id: int, num_goals: int, blind: bool = false) -> Array:
	var out: Array = []
	out.resize(num_goals)
	out.fill(0.0)
	if not blind and goal_id >= 0 and goal_id < num_goals:
		out[goal_id] = 1.0
	return out
