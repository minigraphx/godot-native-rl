extends SceneTree

const Harness = preload("res://test/harness.gd")
const GoalSensor = preload("res://addons/godot_native_rl/sensors/goal_sensor.gd")

func _initialize() -> void:
	var h := Harness.new()

	# push fallback (no source): set_goal drives the one-hot
	var gs := GoalSensor.new()
	gs.num_goals = 3
	h.assert_eq(gs.obs_size(), 3, "obs_size == num_goals (config-fixed)")
	gs.set_goal(1)
	h.assert_eq(gs.get_observation(), [0.0, 1.0, 0.0], "push fallback one-hot")
	gs.goal_blind = true
	h.assert_eq(gs.get_observation(), [0.0, 0.0, 0.0], "goal_blind zeroes the channel")
	gs.free()
	# pull from a source node (get_current_goal)
	var src := preload("res://test/unit/goal_source_stub.gd").new()  # returns 2
	get_root().add_child(src)
	var gs2 := GoalSensor.new()
	gs2.num_goals = 3
	get_root().add_child(gs2)
	gs2.goal_source_path = gs2.get_path_to(src)
	h.assert_eq(gs2.get_observation(), [0.0, 0.0, 1.0], "pulls current goal from source node")
	gs2.free(); src.free()

	h.finish(self)
