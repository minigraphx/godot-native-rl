extends SceneTree
# Agent contract test: action-space shape, obs size with a real game, and that set_action
# stores a clamped 8-vector. No trainer/socket — pure node wiring, no physics stepping.

const Harness = preload("res://test/harness.gd")
const Game = preload("res://examples/quadruped_walk/quadruped_game.gd")
const Agent = preload("res://examples/quadruped_walk/quadruped_agent.gd")

func _initialize() -> void:
	var h = Harness.new()
	var finish := Marker3D.new()
	finish.position = Vector3(0, 0, 40)
	get_root().add_child(finish)
	var game = Game.new()
	get_root().add_child(game)
	game.set_finish(finish)
	game.build_now()

	var agent = Agent.new()
	agent.set_game(game)
	get_root().add_child(agent)   # added AFTER set_game so _ready sees the game

	var space = agent.get_action_space()
	h.assert_true("motors" in space, "action key 'motors'")
	h.assert_eq(space["motors"]["size"], 8, "8 continuous actions")
	h.assert_eq(space["motors"]["action_type"], "continuous", "continuous type")

	var obs = agent.get_obs()
	h.assert_eq(obs["obs"].size(), agent.expected_obs_size(), "obs matches expected size")
	h.assert_eq(agent.expected_obs_size(), 8 + 8 + 3 + 3 + 3 + 4, "expected obs = 29")
	for v in obs["obs"]:
		h.assert_true(is_finite(v), "obs value finite")

	agent.set_action({"motors": [2.0, -2.0, 0,0,0,0,0,0]})
	var stored = agent.stored_action_for_test()
	h.assert_eq(stored.size(), 8, "stored action has 8 entries")
	h.assert_eq(stored[0], 1.0, "action[0] clamped to 1")
	h.assert_eq(stored[1], -1.0, "action[1] clamped to -1")

	h.finish(self)
