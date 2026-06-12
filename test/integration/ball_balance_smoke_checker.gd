extends Node
# 3DBall headless physics smoke (#47): random tilt actions through the real platform+ball under
# Jolt. Asserts obs shape/finiteness AND that the termination path works — under random actions
# the ball must fall and the episode must reset (ball back near spawn height) at least once.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 600

var _game
var _agent
var _frames := 0
var _was_off_edge := false
var _fall_reset_cycles := 0

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("could not resolve game/agent")
		return
	_game.seed_rng(7)

func _physics_process(_delta: float) -> void:
	if _game == null:
		return
	# Constant full tilt: the ball MUST roll off deterministically. The agent (earlier in tree
	# order) detects each fall and resets the episode within the same frame, so this checker
	# can't observe is_fallen() directly — instead it detects the off-edge -> recentered cycle
	# the reset produces (rel.x beyond the half-extent, then back near the center).
	_agent.set_action({"tilt": [1.0, 1.0]})
	var obs: Dictionary = _agent.get_obs()
	if obs["obs"].size() != 8:
		_fail("obs size != 8 (got %d)" % obs["obs"].size())
		return
	for v in obs["obs"]:
		if not is_finite(v):
			_fail("non-finite obs value")
			return
	var rel: Vector3 = _game.relative_ball_pos()
	if absf(rel.x) > _game.platform_half_extent or absf(rel.z) > _game.platform_half_extent:
		_was_off_edge = true
	elif _was_off_edge and absf(rel.x) < 1.0 and absf(rel.z) < 1.0:
		_was_off_edge = false
		_fall_reset_cycles += 1
	_frames += 1
	if _frames >= frames_to_run:
		if _fall_reset_cycles == 0:
			_fail("no fall->reset cycle observed in %d frames (termination path untested)" % _frames)
			return
		print("BALL BALANCE SMOKE PASSED (%d frames, %d fall->reset cycles)" % [_frames, _fall_reset_cycles])
		get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("BALL BALANCE SMOKE FAILED: %s" % reason)
	get_tree().quit(1)
