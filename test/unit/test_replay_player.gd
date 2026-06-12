extends SceneTree
# ReplayPlayer (#39): cadence-correct action delivery, initial-state restore, finish signal,
# warning paths. Driven via step_frame() (SceneTree scripts get no _physics_process).

const Harness = preload("res://test/harness.gd")
const Player = preload("res://addons/godot_native_rl/training/replay_player.gd")
const RF = preload("res://addons/godot_native_rl/training/replay_format.gd")

class StubAgent:
	extends Node
	var actions: Array = []
	func set_action(a) -> void:
		actions.append(a)

class StubGame:
	extends Node
	var applied: Array = []
	func apply_replay_state(state: Dictionary) -> void:
		applied.append(state)

const PATH := "user://replay_player_test_episode.json"

func _write_fixture() -> void:
	var steps := [
		{"action": {"move": 1}, "reward": 1.0},
		{"action": {"move": 2}, "reward": 2.0},
		{"action": {"move": 3}, "reward": 3.0},
	]
	var ep := RF.make_episode({"action_repeat": 2}, {"agent_x": 42.0}, steps)
	var f := FileAccess.open(PATH, FileAccess.WRITE)
	f.store_string(RF.to_json(ep))
	f.close()

var _finished_total := -1.0

func _initialize() -> void:
	var h = Harness.new()
	_write_fixture()

	var agent := StubAgent.new()
	get_root().add_child(agent)
	var game := StubGame.new()
	get_root().add_child(game)
	var player = Player.new()
	get_root().add_child(player)
	player.autoplay = false
	player.replay_path = PATH
	player.set_nodes_for_test(agent, game)
	player.replay_finished.connect(func(t): _finished_total = t)

	h.assert_true(player.play(), "play() loads the episode")
	h.assert_eq(game.applied.size(), 1, "initial state applied")
	h.assert_eq(float(game.applied[0]["agent_x"]), 42.0, "initial state content")

	# cadence 2: actions land on frames 0/2/4; finish check happens on frame 6.
	for i in range(7):
		player.step_frame()
	h.assert_eq(agent.actions.size(), 3, "all 3 actions delivered")
	h.assert_eq(int(agent.actions[0]["move"]), 1, "action order 1")
	h.assert_eq(int(agent.actions[2]["move"]), 3, "action order 3")
	h.assert_true(not player.is_playing(), "stopped after last step")
	h.assert_true(absf(_finished_total - 6.0) < 1e-9, "finished with recorded total")

	# Bad path fails loud.
	var p2 = Player.new()
	get_root().add_child(p2)
	p2.autoplay = false
	p2.replay_path = "user://does_not_exist.json"
	p2.set_nodes_for_test(agent, game)
	h.assert_true(not p2.play(), "missing file refused")

	h.finish(self)
