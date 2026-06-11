extends Node
# Headless smoke test: an arena tiling N worlds in one space (3D rover or 2D ball_chase).
# Asserts the arena spawned exactly N agents, every agent produces a finite obs vector of the
# expected size, that observations change over the run (agents are actually driven), and the
# spawned worlds sit at distinct tile origins >= spacing apart
# (isolation). Drives random actions each frame — discrete ints by default,
# a continuous [-1,1]^n array when continuous_action_size > 0 — then quits with an exit code.

@export var arena_path: NodePath
@export var frames_to_run := 120
@export var expected_count := 4
@export var expected_obs_size := 8
@export var action_count := 4
@export var continuous_action_size := 0  ## >0: send a continuous [-1,1] array of this size instead of a discrete int
@export var spacing := 200.0

var _arena
var _agents: Array = []
var _frames := 0
var _rng := RandomNumberGenerator.new()
var _first_obs: Array = []      # per-agent first-frame obs snapshot
var _obs_changed := false       # any agent's obs differed from its snapshot at any frame

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
	for i in range(_agents.size()):
		var agent: Node = _agents[i]
		agent.set_action({"move": _random_action()})
		var obs_dict = agent.get_obs()
		if not ("obs" in obs_dict) or obs_dict["obs"].size() != expected_obs_size:
			_fail("bad obs shape from %s: %s" % [agent.name, obs_dict])
			return
		for v in obs_dict["obs"]:
			if not is_finite(v):
				_fail("non-finite observation from %s" % agent.name)
				return
		if _first_obs.size() <= i:
			_first_obs.append(obs_dict["obs"].duplicate())
		elif not _obs_changed and obs_dict["obs"] != _first_obs[i]:
			_obs_changed = true
	_frames += 1
	if _frames >= frames_to_run:
		if not _obs_changed:
			_fail("observations never changed over %d frames — agents not actually driven" % _frames)
			return
		print("PARALLEL ARENA SMOKE PASSED (%d agents, %d frames)" % [_agents.size(), _frames])
		get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("PARALLEL ARENA SMOKE FAILED: %s" % reason)
	get_tree().quit(1)

func _random_action() -> Variant:
	if continuous_action_size > 0:
		var a: Array = []
		for _i in range(continuous_action_size):
			a.append(_rng.randf_range(-1.0, 1.0))
		return a
	return _rng.randi_range(0, action_count - 1)
