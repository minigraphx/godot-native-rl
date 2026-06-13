extends Node
# Behavioral regression for the trained MA-POCA policy (#30 M2): two agents, each running the SAME
# shared actor under ncnn inference, must cooperatively collect most of the items within a frame
# budget — the "the centralized critic produced a competent cooperative policy" check. A random
# policy collects ~0-1 items in this budget (it pays the step penalty and wanders), so the bar
# separates a learned policy from noise. Both agents drive their own body from local obs; the env's
# shared team reward is what the critic learned to assign credit for.

@export var game_path: NodePath
@export var agent_a_path: NodePath
@export var agent_b_path: NodePath
@export var frames_to_run := 1200
@export var min_collected := 3   ## of item_count (default 4) within the budget

var _game
var _a
var _b
var _frames := 0
var _best := 0

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_a = get_node_or_null(agent_a_path)
	_b = get_node_or_null(agent_b_path)
	if _game == null or _a == null or _b == null:
		_fail("could not resolve game/agent nodes")

func _count_collected() -> int:
	var n := 0
	for c in _game.collected():
		if c:
			n += 1
	return n

func _physics_process(_delta: float) -> void:
	if _game == null or _a == null or _b == null:
		return
	if _a._ncnn_runner == null or not _a._ncnn_runner.is_model_loaded():
		_fail("trained ncnn model not loaded")
		return
	# Track the best within-episode collection (the world resets between episodes).
	_best = maxi(_best, _count_collected())
	_frames += 1
	if _frames >= frames_to_run:
		if _best >= min_collected:
			print("TRAINED COOP MA-POCA PASSED (%d/%d items collected in one episode within %d frames)"
				% [_best, _game.item_count, _frames])
			get_tree().quit(0)
		else:
			_fail("only %d items collected (need %d) in %d frames — team did not learn to collect"
				% [_best, min_collected, _frames])

func _fail(reason: String) -> void:
	printerr("TRAINED COOP MA-POCA FAILED: %s" % reason)
	get_tree().quit(1)
