extends Node
# Headless check: runs the TRAINED agent under ncnn inference and asserts it actually
# catches the target at least `min_catches` times within `frames_to_run` physics frames.
# A random/untrained policy almost never reaches this threshold, so this verifies learning.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 1800
@export var min_catches := 5

var _game
var _agent
var _frames := 0

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("could not resolve game/agent nodes")

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null:
		return
	if _agent._ncnn_runner == null or not _agent._ncnn_runner.is_model_loaded():
		_fail("trained ncnn model not loaded")
		return
	_frames += 1
	if _frames >= frames_to_run:
		if _game.catches >= min_catches:
			print("TRAINED CHASE PASSED (%d catches in %d frames)" % [_game.catches, _frames])
			get_tree().quit(0)
		else:
			_fail("only %d catches in %d frames (need %d) — agent did not learn to chase" % [_game.catches, _frames, min_catches])

func _fail(reason: String) -> void:
	printerr("TRAINED CHASE FAILED: %s" % reason)
	get_tree().quit(1)
