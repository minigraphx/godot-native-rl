extends SceneTree

const Harness = preload("res://test/harness.gd")
const RoverGameScript = preload("res://examples/rover_3d/rover_game.gd")

func _initialize() -> void:
	var h := Harness.new()
	var g = RoverGameScript.new()
	g.arena_size = Vector2(40.0, 40.0)

	# clamp_to_bounds (X/Z clamped, Y preserved)
	h.assert_eq(g.clamp_to_bounds(Vector3(-5.0, 1.0, 50.0)), Vector3(0.0, 1.0, 40.0), "clamp low/high")
	h.assert_eq(g.clamp_to_bounds(Vector3(20.0, 0.0, 20.0)), Vector3(20.0, 0.0, 20.0), "clamp inside unchanged")

	# is_blocked vs an obstacle AABB centered (12,_,12) half (2,_,2)
	var obs := [{"center": Vector3(12.0, 0.0, 12.0), "half_extent": Vector3(2.0, 1.0, 2.0)}]
	h.assert_true(g.is_blocked(Vector3(12.5, 0.0, 11.0), obs), "inside obstacle -> blocked")
	h.assert_true(not g.is_blocked(Vector3(20.0, 0.0, 20.0), obs), "free cell -> not blocked")
	h.assert_true(not g.is_blocked(Vector3(15.0, 0.0, 12.0), obs), "just outside half-extent -> not blocked")

	# max_distance is the XZ diagonal
	h.assert_true(absf(g.max_distance() - Vector2(40.0, 40.0).length()) < 0.001, "max_distance diagonal")

	# bearing_to: ahead(-Z)->0, +X->-PI/2, -X->+PI/2, behind(+Z)->+/-PI
	h.assert_true(absf(g.bearing_to(Vector3.ZERO, 0.0, Vector3(0.0, 0.0, -5.0))) < 1e-5, "goal ahead -> bearing 0")
	h.assert_true(absf(g.bearing_to(Vector3.ZERO, 0.0, Vector3(5.0, 0.0, 0.0)) - (-PI / 2.0)) < 1e-5, "goal +X -> -PI/2")
	h.assert_true(absf(g.bearing_to(Vector3.ZERO, 0.0, Vector3(-5.0, 0.0, 0.0)) - (PI / 2.0)) < 1e-5, "goal -X -> +PI/2")
	h.assert_true(absf(absf(g.bearing_to(Vector3.ZERO, 0.0, Vector3(0.0, 0.0, 5.0))) - PI) < 1e-5, "goal behind -> +/-PI")
	# bearing is heading-relative: facing +X-ish cancels a +X goal
	h.assert_true(absf(g.bearing_to(Vector3.ZERO, -PI / 2.0, Vector3(5.0, 0.0, 0.0))) < 1e-5, "goal +X while facing +X -> 0")

	# seeded RNG determinism + random_free_position avoids obstacles & stays in bounds
	g.seed_rng(123)
	var rng := RandomNumberGenerator.new()
	rng.seed = 123
	var all_ok := true
	for _i in range(200):
		var p: Vector3 = g.random_free_position(rng, obs)
		if p.x < 0.0 or p.x > 40.0 or p.z < 0.0 or p.z > 40.0 or g.is_blocked(p, obs):
			all_ok = false
	h.assert_true(all_ok, "random_free_position in-bounds and not blocked")

	g.free()
	h.finish(self)
