extends Node
# Shared skeleton for the trained-net behavioral checkers (#335): node resolution, the seeded
# post-seed RE-ROLL (the #298/#300 gotcha — the game's _ready rolls its FIRST layout from the
# unseeded RNG in tree order, so seeding without re-rolling leaves the run's phase random — now
# fixed in ONE place instead of retro-patched across copies), the model-loaded guard, and the
# pass/fail plumbing. Subclasses override _label() / _passed() / _report(); everything else is
# this file. Checkers with genuinely bespoke flow (multi-agent LOS, anti-livelock timeouts,
# streak criteria) stay standalone.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 1800
@export var game_seed := -1  ## >= 0 pins the game RNG (reproducible run) + re-rolls the layout

var _game
var _agent
var _frames := 0

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("could not resolve game/agent nodes")
	elif game_seed >= 0 and _game.has_method("seed_rng"):
		_game.seed_rng(game_seed)
		_re_roll()

func _re_roll() -> void:
	if _game.has_method("reset_positions"):
		_game.reset_positions()
	elif _game.has_method("reset_episode"):
		_game.reset_episode()

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null:
		return
	if _agent._ncnn_runner == null or not _agent._ncnn_runner.is_model_loaded():
		_fail("trained ncnn model not loaded")
		return
	_frames += 1
	if _frames >= frames_to_run:
		if _passed():
			print("%s PASSED (%s)" % [_label(), _report()])
			get_tree().quit(0)
		else:
			_fail(_report())

func _fail(reason: String) -> void:
	printerr("%s FAILED: %s" % [_label(), reason])
	get_tree().quit(1)

# --- Overridden per checker ---

func _label() -> String:
	return "TRAINED CHECK"

func _passed() -> bool:
	return false

func _report() -> String:
	return ""
