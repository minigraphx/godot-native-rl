extends Node
# Trained 3DBall behavioral check (#47): under ncnn inference the ball must stay balanced —
# a trained policy keeps it up for long stretches, an untrained one drops it in ~100 frames.
#
# Criterion is the LONGEST upright STREAK, not the total fall count (#214). Jolt is run-to-run
# nondeterministic and 3DBall is a delicate balance task, so on Godot 4.6.3 the 4.5-trained policy
# occasionally hits a state it can't recover from and racks up many falls in one unlucky run — a
# total-falls bar flakes there. A genuinely competent policy still holds the ball for a long
# continuous stretch at least once even on a divergent run; an untrained one never holds past
# ~150 frames. So we require the best continuous-upright streak to clear min_streak. Falls are
# still counted, for the diagnostic message.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 1800
@export var min_streak := 600  ## longest continuous-upright run must reach this (1/3 of the run)

var _game
var _agent
var _frames := 0
var _was_off := false
var _falls := 0
var _streak := 0
var _best_streak := 0

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
	if off:
		if not _was_off:
			_falls += 1
		_streak = 0
	else:
		_streak += 1
		_best_streak = maxi(_best_streak, _streak)
	_was_off = off
	_frames += 1
	if _frames >= frames_to_run:
		if _best_streak < min_streak:
			_fail("longest upright streak %d frames (need %d), %d falls — policy not balancing"
				% [_best_streak, min_streak, _falls])
			return
		print("TRAINED BALL BALANCE PASSED (%d frames, longest streak %d, %d falls)"
			% [_frames, _best_streak, _falls])
		get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("TRAINED BALL BALANCE FAILED: %s" % reason)
	get_tree().quit(1)
