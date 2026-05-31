extends Node
# Headless smoke test: drives the rover scene for a fixed number of physics frames with
# random actions, exercising the real RaycastSensor3D physics queries + observation pipeline
# + movement/blocking, asserting invariants, then quitting with an exit code.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 180
@export var expected_obs_size := 8
@export var action_count := 4

var _game
var _agent
var _frames := 0
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	_rng.seed = 12345
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("could not resolve game/agent nodes")

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null:
		return
	# Drive a random discrete action so the rover moves and the ray fan sweeps the obstacles.
	_agent.set_action({"move": _rng.randi_range(0, action_count - 1)})
	var obs_dict = _agent.get_obs()
	if not ("obs" in obs_dict) or obs_dict["obs"].size() != expected_obs_size:
		_fail("bad obs shape: %s" % obs_dict)
		return
	for v in obs_dict["obs"]:
		if not is_finite(v):
			_fail("non-finite observation value")
			return
	var p = _game.get_agent_pos()
	if p.x < 0.0 or p.x > _game.arena_size.x or p.z < 0.0 or p.z > _game.arena_size.y:
		_fail("rover left arena bounds: %s" % p)
		return
	_frames += 1
	if _frames >= frames_to_run:
		print("ROVER SMOKE PASSED (%d frames)" % _frames)
		get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("ROVER SMOKE FAILED: %s" % reason)
	get_tree().quit(1)
