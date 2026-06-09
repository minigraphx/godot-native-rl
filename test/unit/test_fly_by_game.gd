extends SceneTree
# Pure-helper tests for fly_by_game.gd: 8-dim plane-local obs, basis advance, goal advance, bounds.

const Harness = preload("res://test/harness.gd")
const GameScript = preload("res://examples/fly_by/fly_by_game.gd")

func _initialize() -> void:
	var h := Harness.new()
	var game = GameScript.new()

	# compute_obs: 8 dims [goal_dir.xyz, goal_dist/50, next_dir.xyz, next_dist/50] in plane-local frame.
	# Plane at origin, identity basis (forward = -Z). Goal straight ahead at -Z*10.
	var xform := Transform3D(Basis(), Vector3.ZERO)
	var obs := game.compute_obs(xform, Vector3(0, 0, -10), Vector3(10, 0, 0))
	h.assert_eq(obs.size(), 8, "obs has 8 dims")
	# Goal dead ahead -> local dir is -Z (0,0,-1).
	h.assert_true(absf(obs[2] - (-1.0)) < 1e-4, "goal dir local -Z (forward)")
	h.assert_true(absf(obs[3] - (10.0 / 50.0)) < 1e-4, "goal dist normalized by 50")
	# Direction components are unit-length (normalized).
	var glen := sqrt(obs[0]*obs[0] + obs[1]*obs[1] + obs[2]*obs[2])
	h.assert_true(absf(glen - 1.0) < 1e-4, "goal dir is unit length")

	# advance_basis: positive turn rotates around UP; result stays orthonormal.
	var b := game.advance_basis(Basis(), 0.0, 1.0, 2.0, 2.0, 0.5)
	h.assert_true(absf(b.determinant() - 1.0) < 1e-4, "advanced basis orthonormal (det 1)")
	# Pure turn (no pitch) keeps the Y axis pointing up.
	h.assert_true(b.y.dot(Vector3.UP) > 0.99, "turn-only keeps up-axis up")

	# out_of_bounds: inside the half-extent box is false, outside is true.
	h.assert_true(not game.out_of_bounds(Vector3(10, 5, -10), Vector3(50, 50, 50)), "inside bounds")
	h.assert_true(game.out_of_bounds(Vector3(60, 0, 0), Vector3(50, 50, 50)), "outside bounds (x)")
	h.assert_true(game.out_of_bounds(Vector3(0, 0, 51), Vector3(50, 50, 50)), "outside bounds (z)")

	# next_goal_index wraps around the ring.
	h.assert_eq(game.next_goal_index(0, 4), 1, "next after 0 is 1")
	h.assert_eq(game.next_goal_index(3, 4), 0, "next after last wraps to 0")

	# clamp_to_bounds: inside is unchanged; outside is pulled onto the centered box.
	h.assert_eq(game.clamp_to_bounds(Vector3(10, 5, -10), Vector3(50, 50, 50)), Vector3(10, 5, -10),
		"clamp leaves in-bounds unchanged")
	h.assert_eq(game.clamp_to_bounds(Vector3(60, -70, 80), Vector3(50, 40, 50)), Vector3(50, -40, 50),
		"clamp pulls out-of-bounds onto the box")

	game.free()
	h.finish(self)
