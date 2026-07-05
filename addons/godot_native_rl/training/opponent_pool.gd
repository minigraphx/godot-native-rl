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

# ELO scale on which a matchup stops being informative: 200 points ~ 76% expected score, so the
# proximity kernel has largely decayed by ~2 sigma (400 points ~ 91%).
const PROXIMITY_SIGMA := 200.0

func pick_opponent(rng: RandomNumberGenerator, mode := "uniform") -> String:
	if _members.is_empty():
		return ""
	var names: Array = _members.keys()
	match mode:
		"latest":
			return names.back()
		"elo_proximity":
			# League matchmaking (#190): spar members rated NEAR the learner — far-weaker
			# opponents teach nothing and far-stronger ones give pure-loss signal. Weighted
			# draw, never a hard cutoff, so the whole pool stays reachable.
			var ratings: Array = []
			for n in names:
				ratings.append(member_rating(n))
			return weighted_pick(rng, names, proximity_weights(ratings, _learner_rating))
		_:
			return names[rng.randi_range(0, names.size() - 1)]

## Gaussian kernel on rating distance to the learner: w_i = exp(-((r_i - learner)/sigma)^2).
static func proximity_weights(ratings: Array, learner: float, sigma := PROXIMITY_SIGMA) -> Array:
	var weights: Array = []
	for r in ratings:
		var d := (float(r) - learner) / maxf(sigma, 1e-6)
		weights.append(exp(-d * d))
	return weights

## One weighted draw over names. Falls back to uniform when the total weight vanishes (every
## member many sigma away) — matchmaking must never dead-end the league.
static func weighted_pick(rng: RandomNumberGenerator, names: Array, weights: Array) -> String:
	if names.is_empty():
		return ""
	var total := 0.0
	for w in weights:
		total += float(w)
	if total <= 1e-12:
		return names[rng.randi_range(0, names.size() - 1)]
	var u := rng.randf() * total
	var acc := 0.0
	for i in range(names.size()):
		acc += float(weights[i])
		if u <= acc:
			return names[i]
	return names.back()

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
	# Validate member entries too (#306): a hand-edited/partial ledger with a non-Dictionary
	# member errors at PICK time under elo_proximity (script error mid-pick, opponent swapping
	# bricked); a missing rating silently starves that member (proximity weight ~2e-16, never
	# picked, never repaired). Same fail-loud posture as the outer-shape check above.
	var members: Dictionary = parsed["members"]
	for member_name in members:
		var entry = members[member_name]
		var rating = entry.get("rating") if entry is Dictionary else null
		if not (rating is float or rating is int):
			push_error("OpponentPool: malformed ledger member '%s' (%s) — rejecting the ledger." % [member_name, str(entry)])
			return false
	_members = members
	_learner_rating = float(parsed.get("learner_rating", DEFAULT_RATING))
	return true
