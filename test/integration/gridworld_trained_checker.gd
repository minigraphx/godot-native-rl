extends Node
# Trained GridWorld behavioral check (#48): under ncnn inference the agent must keep reaching
# goals — and not by luck: goals must clearly outnumber pit hits (rover pattern, adapted to
# the pit hazard). Random play on 8x8 with 3 pits hits pits about as often as goals.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 1800
@export var min_goals := 5
@export var max_pit_ratio := 0.5  ## pits_hit must stay under half the goals reached

var _game
var _agent
var _frames := 0

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("missing game/agent")
		return
	_game.seed_rng(3)

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null:
		return
	if _agent._ncnn_runner == null or not _agent._ncnn_runner.is_model_loaded():
		_fail("ncnn model not loaded")
		return
	_frames += 1
	if _frames >= frames_to_run:
		var goals: int = _game.goals_reached
		var pits: int = _game.pits_hit
		if goals < min_goals:
			_fail("only %d goals in %d frames (need %d)" % [goals, _frames, min_goals])
			return
		if float(pits) > float(goals) * max_pit_ratio:
			_fail("%d pits vs %d goals — not avoiding hazards" % [pits, goals])
			return
		print("TRAINED GRIDWORLD PASSED (%d goals, %d pits in %d frames)" % [goals, pits, _frames])
		get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("TRAINED GRIDWORLD FAILED: %s" % reason)
	get_tree().quit(1)
