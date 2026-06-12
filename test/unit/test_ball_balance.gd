extends SceneTree
# Unit tests for the 3DBall example (#47): pure helpers + agent contract. No physics stepping
# (SceneTree scripts get no _physics_process — the integration smoke covers live physics).

const Harness = preload("res://test/harness.gd")
const Game = preload("res://examples/3dball/ball_balance_game.gd")
const Agent = preload("res://examples/3dball/ball_balance_agent.gd")

func _initialize() -> void:
	var h = Harness.new()

	# --- Pure helpers ---
	h.assert_eq(Game.clamp_tilt(0.5, 0.35), 0.35, "tilt clamped high")
	h.assert_eq(Game.clamp_tilt(-0.5, 0.35), -0.35, "tilt clamped low")
	h.assert_eq(Game.clamp_tilt(0.1, 0.35), 0.1, "tilt passthrough")

	h.assert_true(Game.fallen(Vector3(0, -1.5, 0), 2.5, 1.0), "below platform = fallen")
	h.assert_true(Game.fallen(Vector3(3.6, 0.5, 0), 2.5, 1.0), "off edge x = fallen")
	h.assert_true(Game.fallen(Vector3(0, 0.5, -3.6), 2.5, 1.0), "off edge z = fallen")
	h.assert_true(not Game.fallen(Vector3(1.0, 0.7, -1.0), 2.5, 1.0), "on platform = not fallen")

	var obs := Game.assemble_obs(Vector2(0.1, -0.2), Vector3(1, 2, 3), Vector3(4, 5, 6))
	h.assert_eq(obs.size(), 8, "obs is 8-dim (Unity 3DBall parity)")
	h.assert_eq(obs[0], 0.1, "obs[0] tilt x")
	h.assert_eq(obs[1], -0.2, "obs[1] tilt z")
	h.assert_eq(obs[2], 1.0, "obs[2] rel pos x")
	h.assert_eq(obs[7], 6.0, "obs[7] vel z")

	# --- Agent contract (real game, no stepping) ---
	var game = Game.new()
	get_root().add_child(game)
	var agent = Agent.new()
	agent.set("game_path", NodePath(""))
	get_root().add_child(agent)
	agent._game = game

	var space: Dictionary = agent.get_action_space()
	h.assert_true("tilt" in space, "action key 'tilt'")
	h.assert_eq(space["tilt"]["size"], 2, "2 continuous actions")
	h.assert_eq(space["tilt"]["action_type"], "continuous", "continuous type")

	var aobs: Dictionary = agent.get_obs()
	h.assert_eq(aobs["obs"].size(), 8, "agent obs is 8-dim")
	for v in aobs["obs"]:
		h.assert_true(is_finite(v), "obs value finite")

	agent.set_action({"tilt": [2.0, -2.0]})
	h.assert_eq(agent.stored_tilt_for_test(), Vector2(1.0, -1.0), "actions clamped to [-1,1]")

	h.finish(self)
