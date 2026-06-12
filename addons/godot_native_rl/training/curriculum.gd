extends RefCounted
# Pure curriculum stage/promotion logic — no scene or node dependencies.
# Stages: ordered Array of { "name": String, "params": Dictionary,
#   "promote": { "metric": "mean_reward"|"success_rate", "threshold": float,
#                "window": int, "min_episodes": int } }   (final stage: no "promote")
# Promotion: over a rolling window of the last `window` episodes, promote when at least
# `min_episodes` have been recorded AND the metric clears `threshold`.
# Spec: docs/superpowers/specs/2026-06-12-curriculum-learning-design.md (#28)

const METRICS := ["mean_reward", "success_rate"]

var _stages: Array = []
var _index := 0
var _rewards: Array = []    # rolling window (parallel arrays)
var _successes: Array = []

func set_stages(stages: Array) -> bool:
	if stages.is_empty():
		push_error("Curriculum: stages must be a non-empty Array.")
		return false
	for s in stages:
		if not (s is Dictionary) or not s.has("name") or not s.has("params"):
			push_error("Curriculum: every stage needs 'name' and 'params'.")
			return false
		if s.has("promote"):
			var p = s["promote"]
			if not (p is Dictionary) or not METRICS.has(p.get("metric", "")) \
					or not p.has("threshold") or not p.has("window") or not p.has("min_episodes"):
				push_error("Curriculum: stage '%s' has a malformed 'promote' block." % s["name"])
				return false
	_stages = stages
	_index = 0
	_clear_window()
	return true

func stage_count() -> int:
	return _stages.size()

func stage_index() -> int:
	return _index

func stage_name() -> String:
	return str(_stages[_index]["name"]) if _index < _stages.size() else ""

func current_params() -> Dictionary:
	return _stages[_index]["params"] if _index < _stages.size() else {}

func is_final() -> bool:
	return _index >= _stages.size() - 1

func record_episode(reward: float, success: bool) -> void:
	var window := _window_size()
	_rewards.append(reward)
	_successes.append(success)
	while _rewards.size() > window:
		_rewards.pop_front()
		_successes.pop_front()

func should_promote() -> bool:
	if is_final() or not _stages[_index].has("promote"):
		return false
	var p: Dictionary = _stages[_index]["promote"]
	if _rewards.size() < int(p["min_episodes"]):
		return false
	match str(p["metric"]):
		"mean_reward":
			return _mean(_rewards) >= float(p["threshold"])
		"success_rate":
			return _success_rate() >= float(p["threshold"])
	return false

func advance() -> bool:
	if is_final():
		return false
	_index += 1
	_clear_window()
	return true

func set_stage(i: int) -> bool:
	if i < 0 or i >= _stages.size():
		push_warning("Curriculum: set_stage(%d) out of range [0, %d)." % [i, _stages.size()])
		return false
	_index = i
	_clear_window()
	return true

func _window_size() -> int:
	if _index < _stages.size() and _stages[_index].has("promote"):
		return int(_stages[_index]["promote"]["window"])
	return 1

func _clear_window() -> void:
	_rewards.clear()
	_successes.clear()

func _mean(xs: Array) -> float:
	if xs.is_empty():
		return 0.0
	var s := 0.0
	for x in xs:
		s += float(x)
	return s / xs.size()

func _success_rate() -> float:
	if _successes.is_empty():
		return 0.0
	var n := 0
	for v in _successes:
		if v:
			n += 1
	return float(n) / _successes.size()
