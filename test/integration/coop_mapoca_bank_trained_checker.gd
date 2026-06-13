extends Node
# Behavioral regression for the trained MA-POCA posthumous-credit policy (#30 M3): two agents on the
# shared early-finish actor must, in one episode, BOTH collect most of the items AND have at least
# one agent BANK out (enter the bank zone and leave after contributing). That the team still collects
# under masking shows learning survived the posthumous-credit masking; that an agent banks shows the
# early-finish mechanic is actually exercised (not trivially avoided). A random policy does neither.

@export var game_path: NodePath
@export var agent_a_path: NodePath
@export var agent_b_path: NodePath
@export var frames_to_run := 1500
@export var min_collected := 3   ## of item_count (default 4)

var _game
var _a
var _b
var _frames := 0
var _best_collected := 0
var _saw_bank := false

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
	_best_collected = maxi(_best_collected, _count_collected())
	for banked in _game.banked():
		if banked:
			_saw_bank = true
	_frames += 1
	if _frames >= frames_to_run:
		if _best_collected >= min_collected and _saw_bank:
			print("TRAINED COOP MA-POCA BANK PASSED (%d/%d items collected, an agent banked)"
				% [_best_collected, _game.item_count])
			get_tree().quit(0)
		else:
			_fail("collected %d/%d (need %d), saw_bank=%s — posthumous-credit policy underperformed"
				% [_best_collected, _game.item_count, min_collected, str(_saw_bank)])

func _fail(reason: String) -> void:
	printerr("TRAINED COOP MA-POCA BANK FAILED: %s" % reason)
	get_tree().quit(1)
