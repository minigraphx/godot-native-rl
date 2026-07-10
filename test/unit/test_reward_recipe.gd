extends SceneTree

# RewardRecipe interpreter (#62): the game-side half of LLM reward design. The load-bearing test is
# the DOGFOOD — the committed chase recipe must build a Reward whose per-step + event-driven return
# matches the hand-coded ChaseAgent builder exactly, over a scripted (value, event) sequence. If
# that holds, a recipe is a faithful serialization of the builder, so the LLM search space == the
# builder's expressive range (the whole premise of Option B). Plus: unknown-hook rejection, and a
# manifest-vs-committed-JSON drift guard.

const Harness = preload("res://test/harness.gd")
const RewardRecipe = preload("res://addons/godot_native_rl/reward/reward_recipe.gd")
const RewardBuilderScript = preload("res://addons/godot_native_rl/reward/reward_builder.gd")

# Minimal stand-in for ChaseGame's reward surface: a distance value-fn we drive by hand, a
# constant normalizer, and the target_caught signal. Mirrors the real affordance manifest.
class StubGame:
	extends Node
	signal target_caught
	var dist := 100.0
	func distance() -> float: return dist
	func max_distance() -> float: return 1000.0
	func get_reward_affordances() -> Dictionary:
		return {"value_functions": {"distance": "d", "max_distance": "n"},
			"signals": {"target_caught": "caught"}}

func _initialize() -> void:
	var h := Harness.new()
	var game := StubGame.new()
	get_root().add_child(game)
	var aff: Dictionary = game.get_reward_affordances()

	# --- The committed chase recipe reproduces ChaseAgent's hand-coded reward, step for step. ---
	var recipe := {"terms": [
		{"type": "progress_shaping", "value_fn": "distance", "scale_fn": "max_distance", "rebase_on": ["target_caught"]},
		{"type": "event_bonus", "signal": "target_caught", "amount": 1.0},
		{"type": "step_penalty", "amount": 0.001},
	]}
	var built: Dictionary = RewardRecipe.build(game, recipe, aff)
	h.assert_true(built.has("reward"), "recipe builds a reward")
	# The signal to wire is the single unique target_caught (drives both bonus + rebase).
	h.assert_eq(built["signal_events"].size(), 1, "one signal wiring (target_caught)")
	h.assert_eq(built["signal_events"][0], [game, "target_caught", "target_caught"], "wiring shape")

	# Hand-coded reference (exactly ChaseAgent's builder).
	var reference = RewardBuilderScript.new() \
		.add_progress_shaping(game.distance, game.max_distance, ["target_caught"]) \
		.add_event_bonus("target_caught", 1.0) \
		.add_step_penalty(0.001) \
		.build()
	var recipe_reward = built["reward"]
	reference.reset()
	recipe_reward.reset()

	# Drive an identical scripted sequence through both: move closer, a catch (event), move again.
	var script := [
		{"dist": 90.0},                    # -10 px progress
		{"dist": 60.0},                    # -30 px progress
		{"dist": 55.0, "event": "target_caught"},  # catch: bonus + rebase; net of that step
		{"dist": 40.0},                    # progress from the rebased baseline
		{"dist": 80.0},                    # negative progress (moved away)
	]
	var worst := 0.0
	for step in script:
		game.dist = step["dist"]
		if step.has("event"):
			reference.trigger_event(step["event"])
			recipe_reward.trigger_event(step["event"])
		var r_ref: float = reference.evaluate(null)
		var r_rec: float = recipe_reward.evaluate(null)
		worst = maxf(worst, absf(r_ref - r_rec))
	h.assert_true(worst < 1e-9, "recipe reward == hand-coded builder over the scripted episode (worst |err| %f)" % worst)

	# --- Unknown hook rejected loud (LLM can't hallucinate a value-fn) ---
	var bad := {"terms": [{"type": "progress_shaping", "value_fn": "velocity", "scale_fn": "max_distance"}]}
	h.assert_eq(RewardRecipe.build(game, bad, aff), {}, "unknown value_fn 'velocity' -> {} (loud error above)")
	var bad2 := {"terms": [{"type": "event_bonus", "signal": "exploded", "amount": 1.0}]}
	h.assert_eq(RewardRecipe.build(game, bad2, aff), {}, "unknown signal 'exploded' -> {}")
	var bad3 := {"terms": [{"type": "curiosity", "amount": 1.0}]}
	h.assert_eq(RewardRecipe.build(game, bad3, aff), {}, "unknown term type -> {}")
	h.assert_true(RewardRecipe.validate({"foo": 1}, aff, game).size() > 0, "recipe without 'terms' is invalid")

	# --- A numeric scale (not a value_fn) is accepted ---
	var num_scale := {"terms": [{"type": "progress_shaping", "value_fn": "distance", "scale": 1000.0}]}
	h.assert_true(RewardRecipe.build(game, num_scale, aff).has("reward"), "numeric scale accepted")

	# --- Manifest drift guard: the REAL ChaseGame exposes exactly the committed JSON ---
	var chase_scene: PackedScene = load("res://examples/chase_the_target/chase_world.tscn")
	var world: Node = chase_scene.instantiate()
	get_root().add_child(world)
	await process_frame
	var committed_text := FileAccess.get_file_as_string("res://examples/chase_the_target/chase_reward_affordances.json")
	var committed: Dictionary = JSON.parse_string(committed_text)
	h.assert_eq(world.get_reward_affordances(), committed,
		"ChaseGame.get_reward_affordances() equals the committed chase_reward_affordances.json (no drift)")

	# The committed recipe file validates against the committed manifest.
	var recipe_text := FileAccess.get_file_as_string("res://examples/chase_the_target/chase_reward_recipe.json")
	var committed_recipe: Dictionary = JSON.parse_string(recipe_text)
	h.assert_eq(RewardRecipe.validate(committed_recipe, committed, world), [],
		"committed chase_reward_recipe.json validates against the committed manifest")

	h.finish(self)
