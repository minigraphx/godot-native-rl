extends Node
# GridWorld headless smoke (#48): exercises the REAL GridSensor2D physics queries (the example's
# whole point) + both terminal paths. Pins a known layout, walks the agent onto the goal and a
# pit via scripted actions, and asserts the sensor sees the right layers along the way.

@export var game_path: NodePath
@export var agent_path: NodePath

var _game
var _agent
var _frame := 0

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("missing game/agent")
		return
	_game.seed_rng(5)

func _physics_process(_delta: float) -> void:
	if _game == null:
		return
	_frame += 1
	match _frame:
		2:
			# Pin: agent at (2,2); goal right-adjacent; pit below-adjacent.
			_game.set_state_for_test(Vector2i(2, 2), Vector2i(3, 2), [Vector2i(2, 3), Vector2i(6, 6), Vector2i(7, 1)])
		4:
			# Sensor must see goal layer (right) and pit layer (below) in the 5x5 window.
			var obs: Dictionary = _agent.get_obs()
			var o: Array = obs["obs"]
			if o.size() != 52:
				_fail("obs size != 52 (got %d)" % o.size())
				return
			var grid_sum := 0.0
			for i in range(50):
				grid_sum += float(o[i])
			if grid_sum < 2.0:
				_fail("GridSensor2D saw fewer than 2 occupied cells (sum=%f) — query path broken?" % grid_sum)
				return
		6:
			_agent.set_action({"move": 4})  # right -> goal
		8:
			if _game.goals_reached != 1:
				_fail("goal terminal not resolved (goals=%d)" % _game.goals_reached)
				return
			# Re-pin for the pit walk.
			_game.set_state_for_test(Vector2i(2, 2), Vector2i(7, 7), [Vector2i(2, 3), Vector2i(6, 6), Vector2i(0, 0)])
		10:
			_agent.set_action({"move": 2})  # down -> pit
		12:
			if _game.pits_hit != 1:
				_fail("pit terminal not resolved (pits=%d)" % _game.pits_hit)
				return
			print("GRIDWORLD SMOKE PASSED (sensor query + both terminals)")
			get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("GRIDWORLD SMOKE FAILED: %s" % reason)
	get_tree().quit(1)
