extends Node
# Headless curriculum smoke: drives the REAL chase game + agent + CurriculumController through
# fake episode results and asserts stage promotions actually mutate the game's difficulty params
# and surface in the agent's info field. No trainer/socket.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var curriculum_path: NodePath

var _game
var _agent
var _ctrl
var _stage_signals := 0

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	_ctrl = get_node_or_null(curriculum_path)
	if _game == null or _agent == null or _ctrl == null:
		_fail("could not resolve game/agent/curriculum nodes")
		return
	_ctrl.stage_changed.connect(func(_i, _n, _p): _stage_signals += 1)
	# Defer so every node's _ready (incl. initial stage application) has run.
	call_deferred("_run")

func _run() -> void:
	# Initial stage-0 params applied to the real game.
	if absf(_game.touch_radius - 120.0) > 1e-6:
		_fail("stage 0 touch_radius not applied (got %f)" % _game.touch_radius)
		return
	# 20 strong episodes -> promote to mid (threshold 8.0, window/min 20).
	for i in range(20):
		_ctrl.record_episode(10.0, true)
	if absf(_game.touch_radius - 80.0) > 1e-6:
		_fail("stage 1 touch_radius not applied (got %f)" % _game.touch_radius)
		return
	if absf(_game.arena_size.x - 750.0) > 1e-6:
		_fail("stage 1 arena_size.x not applied (got %f)" % _game.arena_size.x)
		return
	# 20 more -> final stage.
	for i in range(20):
		_ctrl.record_episode(10.0, true)
	if absf(_game.touch_radius - 40.0) > 1e-6:
		_fail("stage 2 touch_radius not applied (got %f)" % _game.touch_radius)
		return
	if _stage_signals != 2:
		_fail("expected 2 stage_changed signals, got %d" % _stage_signals)
		return
	var info: Dictionary = _agent.get_info()
	if int(info.get("curriculum_stage", -1)) != 2:
		_fail("agent info curriculum_stage != 2 (got %s)" % str(info))
		return
	# Final stage holds: more episodes change nothing.
	for i in range(20):
		_ctrl.record_episode(10.0, true)
	if _stage_signals != 2:
		_fail("promotion past final stage")
		return
	print("CURRICULUM SMOKE PASSED (2 promotions, params + info verified)")
	get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("CURRICULUM SMOKE FAILED: %s" % reason)
	get_tree().quit(1)
