extends Node
# Headless smoke: drives the quadruped world with random continuous actions through the real
# code-built rig + Jolt physics for a fixed number of frames, asserting the observation is the
# expected shape and stays finite and the creature doesn't explode to infinity. This is the real
# gate on the builder + game + agent + physics wiring (the unit tests use synchronous fallbacks).

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 240
@export var expected_obs_size := 29
@export var sane_bound := 1000.0  ## torso must stay within this radius (catches physics blow-ups)

var _game
var _agent
var _frames := 0
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	_rng.seed = 4242
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("could not resolve game/agent nodes")

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null:
		return
	var a: Array = []
	for i in range(8):
		a.append(_rng.randf_range(-1.0, 1.0))
	_agent.set_action({"motors": a})

	var obs = _agent.get_obs()
	if not ("obs" in obs) or obs["obs"].size() != expected_obs_size:
		_fail("bad obs shape: %d" % (obs["obs"].size() if "obs" in obs else -1))
		return
	for v in obs["obs"]:
		if not is_finite(v):
			_fail("non-finite observation value")
			return

	var p = _game.torso_pos()
	if not (is_finite(p.x) and is_finite(p.y) and is_finite(p.z)) or p.length() > sane_bound:
		_fail("torso left sane bounds (physics blow-up): %s" % p)
		return

	_frames += 1
	if _frames >= frames_to_run:
		print("QUADRUPED SMOKE PASSED (%d frames)" % _frames)
		get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("QUADRUPED SMOKE FAILED: %s" % reason)
	get_tree().quit(1)
