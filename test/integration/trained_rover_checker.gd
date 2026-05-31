extends Node
# Drives the rover scene under ncnn inference and asserts the trained policy actually
# reaches goals (behavioral regression guard), then quits with an exit code.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 1800
@export var min_reaches := 3

var _game
var _agent
var _frames := 0

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("missing game/agent")

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null:
		return
	if _agent._ncnn_runner == null or not _agent._ncnn_runner.is_model_loaded():
		_fail("ncnn model not loaded")
		return
	# NcnnSync._inference_process drives infer_and_act (action_repeat-gated); we just observe.
	if _frames >= frames_to_run:
		if _game.reaches >= min_reaches:
			print("TRAINED ROVER PASSED (%d goals reached in %d frames)" % [_game.reaches, _frames])
			get_tree().quit(0)
		else:
			_fail("only %d goals in %d frames (need %d)" % [_game.reaches, _frames, min_reaches])
		return
	_frames += 1

func _fail(reason: String) -> void:
	printerr("TRAINED ROVER FAILED: %s" % reason)
	get_tree().quit(1)
