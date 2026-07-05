extends Node
# Headless check for the seek_target example (#38): runs the TRAINED (in-engine CMA-ES) agent
# under ncnn inference and asserts it actually learned BOTH halves of the task — reaching goals
# (a random policy almost never clears the threshold) while keeping hazard contact below a cap
# (a goal-only greedy policy trained without the penalty plows through the hazard more often).

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 1800
@export var min_goals := 5
@export var max_hazard_hits := 6
@export var game_seed := -1  ## >= 0 pins the game RNG (reproducible run)

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
		# The game's _ready already rolled the FIRST layout from the unseeded RNG (tree order:
		# world before checker) — re-roll from the seeded stream or the run is nondeterministic
		# despite the seed (the ES-eval gotcha, see trained_chase_checker).
		_game.reset_positions()

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null:
		return
	if _agent._ncnn_runner == null or not _agent._ncnn_runner.is_model_loaded():
		_fail("trained ncnn model not loaded")
		return
	_frames += 1
	if _frames >= frames_to_run:
		if _game.goals_reached >= min_goals and _game.hazard_hits <= max_hazard_hits:
			print("TRAINED SEEK PASSED (%d goals, %d hazard hits in %d frames)"
				% [_game.goals_reached, _game.hazard_hits, _frames])
			get_tree().quit(0)
		else:
			_fail("%d goals (need >= %d) with %d hazard hits (cap %d) in %d frames"
				% [_game.goals_reached, min_goals, _game.hazard_hits, max_hazard_hits, _frames])

func _fail(reason: String) -> void:
	printerr("TRAINED SEEK FAILED: %s" % reason)
	get_tree().quit(1)
