extends RefCounted
# Opponent pool + ELO ledger for league self-play (#29). Pure logic: file I/O stays in the
# SelfPlayManager node. Ledger shape:
# {"members": {"<name>": {"rating": float, "games": int}}, "learner_rating": float}
# Spec: docs/superpowers/specs/2026-06-12-competitive-selfplay-design.md

const Elo = preload("res://addons/godot_native_rl/training/elo.gd")
const DEFAULT_RATING := 1200.0

var _members: Dictionary = {}      # name -> {"rating": float, "games": int}
var _learner_rating := DEFAULT_RATING

func learner_rating() -> float:
	return _learner_rating

func members() -> Array:
	return _members.keys()

func member_rating(name: String) -> float:
	return float(_members.get(name, {}).get("rating", -1.0))

func is_empty() -> bool:
	return _members.is_empty()

# New snapshots enter at the current learner rating (league convention: a frozen copy starts
# where the learner left off).
func add_member(name: String) -> void:
	_members[name] = {"rating": _learner_rating, "games": 0}

func pick_opponent(rng: RandomNumberGenerator, mode := "uniform") -> String:
	if _members.is_empty():
		return ""
	var names: Array = _members.keys()
	match mode:
		"latest":
			return names.back()
		_:
			return names[rng.randi_range(0, names.size() - 1)]

func record_match(member_name: String, learner_won: bool, draw := false, k := 32.0) -> bool:
	if not _members.has(member_name):
		push_error("OpponentPool: unknown member '%s'." % member_name)
		return false
	var score := 0.5 if draw else (1.0 if learner_won else 0.0)
	var pair := Elo.update_pair(_learner_rating, member_rating(member_name), score, k)
	_learner_rating = pair[0]
	_members[member_name]["rating"] = pair[1]
	_members[member_name]["games"] = int(_members[member_name]["games"]) + 1
	return true

func ledger_to_json() -> String:
	return JSON.stringify({"members": _members, "learner_rating": _learner_rating}, "\t")

func load_ledger(json_text: String) -> bool:
	var parsed = JSON.parse_string(json_text)
	if not (parsed is Dictionary) or not (parsed.get("members") is Dictionary):
		push_error("OpponentPool: malformed ledger JSON.")
		return false
	_members = parsed["members"]
	_learner_rating = float(parsed.get("learner_rating", DEFAULT_RATING))
	return true
