extends Node
# Trained 3DBall behavioral check (#47): under ncnn inference the ball must stay balanced —
# a trained policy keeps it up essentially forever, an untrained one drops it in ~100 frames.
# The agent resets episodes on falls, so we count FALL EVENTS (off-edge cycles, since the agent
# consumes is_fallen() in the same frame) and require at most max_falls across the run.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 1800
@export var max_falls := 2  ## generous: cross-platform Jolt variance (quadruped lesson)

var _game
var _agent
var _frames := 0
var _was_off := false
var _falls := 0

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("missing game/agent")
		return
	_game.seed_rng(11)

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null:
		return
	if _agent._ncnn_runner == null or not _agent._ncnn_runner.is_model_loaded():
		_fail("ncnn model not loaded")
		return
	var rel: Vector3 = _game.relative_ball_pos()
	var off: bool = absf(rel.x) > _game.platform_half_extent or absf(rel.z) > _game.platform_half_extent or rel.y < 0.0
	if off and not _was_off:
		_falls += 1
	_was_off = off
	_frames += 1
	if _frames >= frames_to_run:
		if _falls > max_falls:
			_fail("%d falls in %d frames (max %d) — policy not balancing" % [_falls, _frames, max_falls])
			return
		print("TRAINED BALL BALANCE PASSED (%d frames, %d falls)" % [_frames, _falls])
		get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("TRAINED BALL BALANCE FAILED: %s" % reason)
	get_tree().quit(1)
