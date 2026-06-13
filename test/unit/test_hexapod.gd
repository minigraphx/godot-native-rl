extends SceneTree
# Unit test for the #60 M3 hexapod morphology: the 6-leg rig builds with the right structure, and
# the shared (leg-count-agnostic) game + agent report hexapod dims (12 joints, 6 feet, 39-dim obs,
# 12 motors). Reuses the quadruped game/agent — only the builder differs.

const Harness = preload("res://test/harness.gd")
const Builder = preload("res://examples/quadruped_walk/hexapod_builder.gd")
const Game = preload("res://examples/quadruped_walk/hexapod_game.gd")
const Agent = preload("res://examples/quadruped_walk/quadruped_agent.gd")

func _initialize() -> void:
	var h = Harness.new()

	# --- builder structure ---
	var holder := Node3D.new()
	get_root().add_child(holder)
	var rig: Dictionary = Builder.build(holder)
	h.assert_eq(rig["joints"].size(), 12, "12 hinge joints (6 legs x 2)")
	h.assert_eq(rig["uppers"].size(), 6, "6 upper segments")
	h.assert_eq(rig["lowers"].size(), 6, "6 lower segments")
	h.assert_eq(rig["feet"].size(), 6, "6 feet")
	h.assert_true(rig["torso"] is RigidBody3D, "torso is a RigidBody3D")

	# --- game reports hexapod dims via the shared leg-count-agnostic surface ---
	var finish := Marker3D.new()
	finish.position = Vector3(0, 0, 40)
	get_root().add_child(finish)
	var game = Game.new()
	get_root().add_child(game)
	game.set_finish(finish)
	game.build_now()
	h.assert_eq(game.joint_count(), 12, "game: 12 joints")
	h.assert_eq(game.joint_angles().size(), 12, "game: 12 joint angles")
	h.assert_eq(game.joint_velocities().size(), 12, "game: 12 joint velocities")
	h.assert_eq(game.foot_contacts().size(), 6, "game: 6 foot contacts")

	# --- shared agent reports hexapod action/obs dims ---
	var agent = Agent.new()
	get_root().add_child(agent)
	agent.set_game(game)
	h.assert_eq(agent.get_action_space()["motors"]["size"], 12, "agent: 12 motors")
	h.assert_eq(agent.expected_obs_size(), 39, "agent: 39-dim obs (5*6 + 9)")

	h.finish(self)
