class_name RewardRecipe
extends RefCounted

# Interpreter for declarative JSON reward RECIPES (#62): builds a live `Reward` from a recipe over
# the shipped RewardBuilder vocabulary, WITHOUT any code execution. A recipe names value-functions
# and signals; every name is validated against the game's `get_reward_affordances()` manifest AND
# resolved on the game, so an LLM (or a hand-written recipe) cannot reference a hook that doesn't
# exist — an unknown name fails loud and builds nothing. This is the game-side half of the
# Eureka-style LLM-reward-design tool; the Python side emits recipes over the same schema.
#
# Recipe shape: {"terms": [ {type, ...}, ... ]}. Term types mirror RewardBuilder.add_* 1:1:
#   progress_shaping: value_fn (str), scale_fn (str) OR scale (num), rebase_on (list[str])
#   event_bonus:      signal (str), amount (num)
#   step_penalty:     amount (num)
#   alive_bonus:      amount (num)
#
# Design: pure + static (no node state); the agent applies the result. Reference by full path
# (extends here is unused) — see the reward/ module conventions.

const RewardBuilderScript = preload("res://addons/godot_native_rl/reward/reward_builder.gd")

const _TERM_TYPES := ["progress_shaping", "event_bonus", "step_penalty", "alive_bonus"]


## Build a Reward from `recipe` for `game`, validated against `affordances` (the game's
## get_reward_affordances() manifest). Returns {"reward": Reward, "signal_events":
## [[emitter, signal_name, event_name], ...]} on success, or {} (fail loud) on any invalid or
## unknown-hook recipe. The caller sets `reward_source = result.reward` and wires each
## signal_events entry through a RewardAdapter (adapter.on_signal_event(emitter, signal, event)).
static func build(game, recipe: Dictionary, affordances: Dictionary) -> Dictionary:
	var errs := validate(recipe, affordances, game)
	if not errs.is_empty():
		push_error("RewardRecipe: invalid recipe — %s" % "; ".join(errs))
		return {}
	var b = RewardBuilderScript.new()
	var wired := {}  # set of signal names needing a RewardAdapter wiring (event bonuses + rebases)
	for term in recipe["terms"]:
		match String(term["type"]):
			"progress_shaping":
				var vfn := Callable(game, String(term["value_fn"]))
				var scale = Callable(game, String(term["scale_fn"])) if term.has("scale_fn") \
					else float(term.get("scale", 1.0))
				var rebase: Array = term.get("rebase_on", [])
				b = b.add_progress_shaping(vfn, scale, rebase)
				for s in rebase:
					wired[String(s)] = true
			"event_bonus":
				b = b.add_event_bonus(String(term["signal"]), float(term["amount"]))
				wired[String(term["signal"])] = true
			"step_penalty":
				b = b.add_step_penalty(float(term["amount"]))
			"alive_bonus":
				b = b.add_alive_bonus(float(term["amount"]))
	# One wiring per unique signal, event_name == signal_name: that single trigger drives BOTH the
	# event bonus and any progress-shaping rebase keyed on the same signal (the chase pattern).
	var events: Array = []
	for s in wired:
		events.append([game, s, s])
	return {"reward": b.build(), "signal_events": events}


## Validate `recipe` against `affordances` and (defensively) the live `game`. Returns a list of
## human-readable problems; empty means valid. A name must be BOTH in the manifest (the recipe's
## declared vocabulary) AND resolvable on the game (runtime truth).
static func validate(recipe: Dictionary, affordances: Dictionary, game) -> Array:
	var errs: Array = []
	if not recipe.has("terms") or not (recipe["terms"] is Array):
		return ["recipe has no 'terms' array"]
	var value_fns: Dictionary = affordances.get("value_functions", {})
	var signals: Dictionary = affordances.get("signals", {})
	var i := 0
	for term in recipe["terms"]:
		var where := "term[%d]" % i
		i += 1
		if not (term is Dictionary) or not term.has("type"):
			errs.append("%s: not a typed term" % where)
			continue
		var t := String(term["type"])
		if not (t in _TERM_TYPES):
			errs.append("%s: unknown type '%s' (know %s)" % [where, t, str(_TERM_TYPES)])
			continue
		match t:
			"progress_shaping":
				_check_value_fn(term.get("value_fn"), value_fns, game, where, "value_fn", errs)
				if term.has("scale_fn"):
					_check_value_fn(term.get("scale_fn"), value_fns, game, where, "scale_fn", errs)
				elif not (term.get("scale") is float or term.get("scale") is int):
					errs.append("%s: needs a scale_fn (str) or scale (num)" % where)
				for s in term.get("rebase_on", []):
					_check_signal(s, signals, game, where, "rebase_on", errs)
			"event_bonus":
				_check_signal(term.get("signal"), signals, game, where, "signal", errs)
				_check_amount(term.get("amount"), where, errs)
			"step_penalty", "alive_bonus":
				_check_amount(term.get("amount"), where, errs)
	return errs


static func _check_value_fn(name, manifest: Dictionary, game, where: String, field: String, errs: Array) -> void:
	if not (name is String):
		errs.append("%s.%s: missing/not a string" % [where, field])
		return
	if not manifest.has(name):
		errs.append("%s.%s '%s' is not a declared value-function (manifest: %s)" % [where, field, name, str(manifest.keys())])
	elif game != null and not game.has_method(name):
		errs.append("%s.%s '%s' is in the manifest but the game has no such method" % [where, field, name])


static func _check_signal(name, manifest: Dictionary, game, where: String, field: String, errs: Array) -> void:
	if not (name is String):
		errs.append("%s.%s: missing/not a string" % [where, field])
		return
	if not manifest.has(name):
		errs.append("%s.%s '%s' is not a declared signal (manifest: %s)" % [where, field, name, str(manifest.keys())])
	elif game != null and not game.has_signal(name):
		errs.append("%s.%s '%s' is in the manifest but the game has no such signal" % [where, field, name])


static func _check_amount(amount, where: String, errs: Array) -> void:
	if not (amount is float or amount is int):
		errs.append("%s: amount missing/not a number" % where)
