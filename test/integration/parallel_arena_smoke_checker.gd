extends Node
# Headless smoke test: an arena tiling N rover worlds in one physics space. Asserts the
# arena spawned exactly N agents, every agent produces a finite obs vector of the expected
# size, and the spawned worlds sit at distinct tile origins >= spacing apart (isolation).
# Drives random actions each frame like the rover smoke, then quits with an exit code.

@export var arena_path: NodePath
@export var frames_to_run := 120
@export var expected_count := 4
@export var expected_obs_size := 8
@export var action_count := 4
@export var spacing := 200.0

var _arena
var _agents: Array = []
var _frames := 0
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	_rng.seed = 4321
	_arena = get_node_or_null(arena_path)
	if _arena == null:
		_fail("could not resolve arena node")
		return
	_agents = get_tree().get_nodes_in_group("AGENT")
	if _agents.size() != expected_count:
		_fail("expected %d agents, got %d" % [expected_count, _agents.size()])
		return
	# Isolation/tiling: spawned world origins are pairwise >= spacing apart.
	var worlds: Array = _arena.get_children()
	for i in range(worlds.size()):
		for j in range(i + 1, worlds.size()):
			var d: float = worlds[i].global_position.distance_to(worlds[j].global_position)
			if d < spacing - 0.001:
				_fail("worlds %d,%d only %.1f apart (need >= %.1f)" % [i, j, d, spacing])
				return

func _physics_process(_delta: float) -> void:
	if _arena == null:
		return
	for agent in _agents:
		agent.set_action({"move": _rng.randi_range(0, action_count - 1)})
		var obs_dict = agent.get_obs()
		if not ("obs" in obs_dict) or obs_dict["obs"].size() != expected_obs_size:
			_fail("bad obs shape from %s: %s" % [agent.name, obs_dict])
			return
		for v in obs_dict["obs"]:
			if not is_finite(v):
				_fail("non-finite observation from %s" % agent.name)
				return
	_frames += 1
	if _frames >= frames_to_run:
		print("PARALLEL ARENA SMOKE PASSED (%d agents, %d frames)" % [_agents.size(), _frames])
		get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("PARALLEL ARENA SMOKE FAILED: %s" % reason)
	get_tree().quit(1)
