extends SceneTree

# Headless unit tests for the drifting-target pure helpers (on-device fine-tuning env, #131).

const Harness = preload("res://test/harness.gd")
const DriftGame = preload("res://examples/chase_the_target/chase_drift_game.gd")

func _initialize() -> void:
	var h := Harness.new()
	var arena := Vector2(1000, 600)

	# Plain step: moves along dir by speed*delta, direction unchanged.
	var s: Array = DriftGame.drift_step(Vector2(500, 300), Vector2.RIGHT, 100.0, 0.1, arena)
	h.assert_eq(s[0], Vector2(510, 300), "plain step advances by speed*delta")
	h.assert_eq(s[1], Vector2.RIGHT, "plain step keeps direction")

	# Right-wall reflection: overshoot mirrors back inside, x-direction flips.
	var r: Array = DriftGame.drift_step(Vector2(995, 300), Vector2.RIGHT, 100.0, 0.1, arena)
	h.assert_eq(r[0], Vector2(995, 300), "overshoot of 5 mirrors to 5 inside the wall")
	h.assert_eq(r[1], Vector2.LEFT, "x-direction flips at the right wall")

	# Top-wall reflection with a diagonal direction: only y flips.
	var t: Array = DriftGame.drift_step(Vector2(500, 2), Vector2(1, -1).normalized(), 100.0, 0.1, arena)
	h.assert_true((t[0] as Vector2).y > 0.0, "y mirrors back inside at the top wall")
	h.assert_true((t[1] as Vector2).y > 0.0 and (t[1] as Vector2).x > 0.0, "only y-direction flips on a diagonal")

	# Corner: both components flip.
	var c: Array = DriftGame.drift_step(Vector2(998, 2), Vector2(1, -1).normalized(), 100.0, 0.1, arena)
	h.assert_eq(c[1], Vector2(-1, 1).normalized(), "both components flip in a corner")

	# Degenerate giant step stays inside the arena (guarded double-clamp).
	var g: Array = DriftGame.drift_step(Vector2(500, 300), Vector2.RIGHT, 100000.0, 1.0, arena)
	var gp: Vector2 = g[0]
	h.assert_true(gp.x >= 0.0 and gp.x <= arena.x, "giant step clamped inside")

	# drift_dir_from: unit-length, deterministic under seed.
	var rng := RandomNumberGenerator.new()
	rng.seed = 11
	var d1: Vector2 = DriftGame.drift_dir_from(rng)
	h.assert_true(absf(d1.length() - 1.0) < 1e-5, "drift dir is unit length")
	var rng2 := RandomNumberGenerator.new()
	rng2.seed = 11
	h.assert_eq(DriftGame.drift_dir_from(rng2), d1, "drift dir deterministic under seed")

	# flee_dir: pure away at jitter 0; unit length; degenerate fallbacks.
	var away: Vector2 = DriftGame.flee_dir(Vector2(600, 300), Vector2(500, 300), Vector2.UP, 0.0)
	h.assert_eq(away, Vector2.RIGHT, "jitter 0 -> pure away-vector")
	var blended: Vector2 = DriftGame.flee_dir(Vector2(600, 300), Vector2(500, 300), Vector2.UP, 0.5)
	h.assert_true(absf(blended.length() - 1.0) < 1e-5, "blended flee dir is unit length")
	h.assert_true(blended.x > 0.0 and blended.y < 0.0, "blend points between away and jitter")
	h.assert_eq(DriftGame.flee_dir(Vector2(500, 300), Vector2(500, 300), Vector2.UP, 0.3),
		Vector2.UP, "agent on target -> jitter fallback")
	# Opposing jitter exactly cancels the away component at jitter 0.5 -> away fallback.
	h.assert_eq(DriftGame.flee_dir(Vector2(600, 300), Vector2(500, 300), Vector2.LEFT, 0.5),
		Vector2.RIGHT, "cancelling blend -> away fallback")

	h.finish(self)
