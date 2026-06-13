extends Node
# Behavioral regression for #60 M2: under ncnn inference on the RACE-stage track, the trained
# policy must advance forward AND clear hurdles. Clears are computed from the best single-run
# torso reach vs the track layout (independent of the agent's reward bookkeeping).

@export var game_path: NodePath
@export var agent_path: NodePath
@export var track_path: NodePath
@export var frames_to_run := 1200
@export var min_forward := 8.0   ## torso must reach at least this far (+Z) in some episode
@export var min_cleared := 1     ## and pass at least this many hurdles

var _game
var _agent
var _track
var _frames := 0
var _max_forward := 0.0

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	_track = get_node_or_null(track_path)
	if _game == null or _agent == null or _track == null:
		_fail("missing game/agent/track")

func _cleared() -> int:
	var n := 0
	for z in _track.zs():
		if _max_forward > float(z) + _track.hurdle_depth / 2.0:
			n += 1
	return n

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null or _track == null:
		return
	if _agent._ncnn_runner == null or not _agent._ncnn_runner.is_model_loaded():
		_fail("ncnn model not loaded")
		return
	_max_forward = maxf(_max_forward, _game.torso_pos().z)
	_frames += 1
	if _frames >= frames_to_run:
		var cleared := _cleared()
		print("DIAG: max_forward=%.2f  hurdles_cleared=%d/%d" % [_max_forward, cleared, _track.zs().size()])
		if _max_forward >= min_forward and cleared >= min_cleared:
			print("TRAINED QUADRUPED HURDLES PASSED (forward %.2fm, %d hurdles cleared)" % [_max_forward, cleared])
			get_tree().quit(0)
		else:
			_fail("forward %.2fm (need %.1f), cleared %d (need %d)" % [_max_forward, min_forward, cleared, min_cleared])

func _fail(reason: String) -> void:
	printerr("TRAINED QUADRUPED HURDLES FAILED: %s" % reason)
	get_tree().quit(1)
