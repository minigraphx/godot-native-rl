extends Node
# Drives the quadruped under ncnn inference and asserts the trained policy actually moves the
# creature FORWARD (toward the finish at +Z) — the behavioral "it walks" regression. NcnnSync
# (control_mode=2) drives infer_and_act; we just observe the torso's max forward reach across the
# run (the agent resets the world on each fall, so we track the best single-episode advance).

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 1200
@export var min_forward := 3.0   ## torso must reach at least this far in +Z in some episode

var _game
var _agent
var _frames := 0
var _max_forward := 0.0
var _max_radial := 0.0
var _min_dist := 1e20
var _vel_sum := 0.0

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
	var p: Vector3 = _game.torso_pos()
	_max_forward = maxf(_max_forward, p.z)
	_max_radial = maxf(_max_radial, Vector2(p.x, p.z).length())
	_min_dist = minf(_min_dist, _game.distance())
	_vel_sum += _game.forward_velocity()
	_frames += 1
	if _frames >= frames_to_run:
		var mean_vel := _vel_sum / float(_frames)
		print("DIAG: max_forward=%.2f  max_radial=%.2f  min_dist=%.2f  mean_fwd_vel=%.3f" % [_max_forward, _max_radial, _min_dist, mean_vel])
		if _max_forward >= min_forward:
			print("TRAINED QUADRUPED PASSED (max forward = %.2f m in %d frames)" % [_max_forward, _frames])
			get_tree().quit(0)
		else:
			_fail("max forward only %.2f m in %d frames (need %.2f)" % [_max_forward, _frames, min_forward])

func _fail(reason: String) -> void:
	printerr("TRAINED QUADRUPED FAILED: %s" % reason)
	get_tree().quit(1)
