extends SceneTree
# Runtime-surface test for QuadrupedGame. Builds the rig and asserts the accessor surface
# synchronously (no physics stepping — a SceneTree script does not get _physics_process;
# stepping is covered by the Task 6 integration smoke). Also checks reset keeps things sane.

const Harness = preload("res://test/harness.gd")
const Game = preload("res://examples/quadruped_walk/quadruped_game.gd")

func _initialize() -> void:
	var h = Harness.new()

	var finish := Marker3D.new()
	finish.position = Vector3(0, 0, 40)
	get_root().add_child(finish)

	var game = Game.new()
	get_root().add_child(game)
	game.set_finish(finish)
	game.build_now()

	h.assert_eq(game.joint_count(), 8, "8 joints")
	h.assert_eq(game.foot_contacts().size(), 4, "4 foot contact flags")
	h.assert_eq(game.joint_angles().size(), 8, "8 joint angles")
	h.assert_eq(game.joint_velocities().size(), 8, "8 joint velocities")
	h.assert_true(game.distance() > 0.0, "distance to finish positive")
	h.assert_true(is_finite(game.upright()), "upright finite")
	h.assert_eq(game.dir_to_finish().size(), 3, "dir_to_finish is 3-dim")

	# All accessor outputs are finite.
	for v in game.joint_angles():
		h.assert_true(is_finite(v), "joint angle finite")
	for v in game.joint_velocities():
		h.assert_true(is_finite(v), "joint velocity finite")

	# reset_positions keeps the creature in a sane state.
	game.reset_positions()
	h.assert_true(game.distance() > 0.0, "distance positive after reset")
	h.assert_true(is_finite(game.upright()), "upright finite after reset")

	h.finish(self)
