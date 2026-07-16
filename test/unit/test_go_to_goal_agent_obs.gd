extends SceneTree

# Obs-width contract for the GoToGoal agent (#386): the ENTIRE observation is sensor-composed via
# collect_sensors() — a 3-target RelativePositionSensor2D (separate mode: unit dir 2 + dist 1 per
# target = 9) plus a GoalSensor (num_goals = 3 one-hot). No hand-coded features, so the pinned width
# is 9 + 3 = 12. A drift in either child sensor's config breaks this without touching the agent.

const Harness = preload("res://test/harness.gd")
const GoToGoalGame = preload("res://examples/go_to_goal/go_to_goal_game.gd")
const GoToGoalAgent = preload("res://examples/go_to_goal/go_to_goal_agent.gd")
const RelativePositionSensor2D = preload("res://addons/godot_native_rl/sensors/relative_position_sensor_2d.gd")
const GoalSensor = preload("res://addons/godot_native_rl/sensors/goal_sensor.gd")

func _initialize() -> void:
	var h := Harness.new()

	var game := GoToGoalGame.new()
	game.name = "Game"
	get_root().add_child(game)

	var agent := GoToGoalAgent.new()
	agent.game_path = NodePath("../Game")

	# 3-target RelativePositionSensor2D in separate mode -> 3 * (dir 2 + dist 1) = 9 floats.
	var sensor := RelativePositionSensor2D.new()
	sensor.max_distance = 1000.0
	sensor.use_separate_direction = true
	var t0 := Node2D.new(); t0.position = Vector2(100, 0)
	var t1 := Node2D.new(); t1.position = Vector2(0, 200)
	var t2 := Node2D.new(); t2.position = Vector2(300, 300)
	var targets: Array[Node2D] = [t0, t1, t2]
	sensor.objects_to_observe = targets
	get_root().add_child(t0); get_root().add_child(t1); get_root().add_child(t2)
	agent.add_child(sensor)

	# GoalSensor: num_goals = 3 one-hot -> 3 floats.
	var goal_sensor := GoalSensor.new()
	goal_sensor.num_goals = 3
	agent.add_child(goal_sensor)

	get_root().add_child(agent)
	await process_frame  # let _ready run on the agent + sensors

	var obs: Array = agent.get_obs()["obs"]
	h.assert_eq(obs.size(), 12, "sensor-composed obs is 12 wide (9 RelPos + 3 goal)")

	agent.free(); game.free()
	t0.free(); t1.free(); t2.free()

	h.finish(self)
