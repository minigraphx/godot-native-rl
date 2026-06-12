extends RefCounted
# gnrl_replay_v1: a recorded training episode — actions + per-step rewards + an opt-in
# initial-state snapshot. Pure (de)serialization/validation; no scene deps. (#39)
# Spec: docs/superpowers/specs/2026-06-12-episode-replay-design.md

const FORMAT := "gnrl_replay_v1"

static func make_episode(meta: Dictionary, initial_state: Dictionary, steps: Array) -> Dictionary:
	var total := 0.0
	for s in steps:
		total += float(s.get("reward", 0.0))
	var m := meta.duplicate()
	m["n_steps"] = steps.size()
	m["total_reward"] = total
	return {"format": FORMAT, "meta": m, "initial_state": initial_state, "steps": steps}

static func to_json(episode: Dictionary) -> String:
	return JSON.stringify(episode, "\t")

static func from_json(text: String) -> Dictionary:
	var parsed = JSON.parse_string(text)
	return parsed if parsed is Dictionary else {}

static func validate(episode: Dictionary) -> bool:
	if episode.get("format", "") != FORMAT:
		push_error("ReplayFormat: not a %s file." % FORMAT)
		return false
	if not (episode.get("meta") is Dictionary) or not (episode.get("steps") is Array):
		push_error("ReplayFormat: missing meta/steps.")
		return false
	for s in episode["steps"]:
		if not (s is Dictionary) or not s.has("action"):
			push_error("ReplayFormat: malformed step (need at least 'action').")
			return false
	return true
