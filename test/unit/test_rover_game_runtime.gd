extends SceneTree

const Harness = preload("res://test/harness.gd")
const RoverGameScript = preload("res://examples/rover_3d/rover_game.gd")

func _initialize() -> void:
	var h := Harness.new()
	var g = RoverGameScript.new()
	g.arena_size = Vector2(40.0, 40.0)

	# Inject a body + goal (Node3D, not added to tree — only local position/rotation used).
	var body := Node3D.new()
	var goal := Node3D.new()
	g._agent_body = body
	g._goal = goal
	g.obstacles = [{"center": Vector3(20.0, 0.0, 10.0), "half_extent": Vector3(2.0, 1.0, 2.0)}]

	# Unobstructed forward move from (20,0,20) facing -Z advances toward -Z and clamps in bounds.
	body.position = Vector3(20.0, 0.0, 20.0)
	body.rotation.y = 0.0
	g.move_agent(g.move_speed, 0.0, 0.1)
	h.assert_true(body.position.z < 20.0, "forward move advances toward -Z")
	h.assert_true(body.position.x > -0.001 and body.position.x < 40.001, "stays in X bounds")

	# A move that would enter the obstacle is blocked (position held) and emits `bumped`.
	var bumps := [0]
	g.bumped.connect(func() -> void: bumps[0] += 1)
	body.position = Vector3(20.0, 0.0, 13.0)  # just below the obstacle (z=10, half 2 => blocks z in [8,12])
	body.rotation.y = 0.0  # facing -Z (decreasing z) -> moves toward the obstacle
	var held := body.position
	g.move_agent(g.move_speed, 0.0, 0.5)  # large step pushes into z<=12
	h.assert_eq(body.position, held, "blocked move holds position")
	h.assert_eq(bumps[0], 1, "blocked move emits bumped once")

	# relocate_goal increments reaches and emits goal_reached
	var reached := [0]
	g.goal_reached.connect(func() -> void: reached[0] += 1)
	g.seed_rng(7)
	g.relocate_goal()
	g.relocate_goal()
	h.assert_eq(g.reaches, 2, "relocate_goal increments reaches")
	h.assert_eq(reached[0], 2, "relocate_goal emits goal_reached each call")

	# reset_positions keeps body + goal in free, in-bounds cells
	g.reset_positions()
	h.assert_true(not g.is_blocked(body.position, g.obstacles), "reset body not blocked")
	h.assert_true(not g.is_blocked(goal.position, g.obstacles), "reset goal not blocked")

	body.free()
	goal.free()
	g.free()
	h.finish(self)
