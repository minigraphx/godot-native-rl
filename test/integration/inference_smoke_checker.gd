extends Node
# Headless smoke test: runs the chase scene under ncnn inference for a fixed number
# of physics frames, asserting the agent stays in bounds and produces valid discrete
# actions, then quits with an exit code.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 240
@export var action_count := 5

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
		_fail("agent ncnn model not loaded — inference is dead")
		return
	# Positions are clamped by ChaseGame.move_agent; this guards against a clamp regression.
	var pos = _game.get_agent_pos()
	if pos.x < 0.0 or pos.x > _game.arena_size.x or pos.y < 0.0 or pos.y > _game.arena_size.y:
		_fail("agent left arena bounds: %s" % pos)
		return
	if _agent._action_index < 0 or _agent._action_index >= action_count:
		_fail("invalid action index: %d" % _agent._action_index)
		return
	_frames += 1
	if _frames >= frames_to_run:
		print("INFERENCE SMOKE PASSED (%d frames)" % _frames)
		get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("INFERENCE SMOKE FAILED: %s" % reason)
	get_tree().quit(1)
