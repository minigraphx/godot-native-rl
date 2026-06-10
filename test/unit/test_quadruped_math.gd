extends SceneTree
# Pure-helper unit tests for QuadrupedMath. Run headless via the project harness.

const Harness = preload("res://test/harness.gd")
const QM = preload("res://examples/quadruped_walk/quadruped_math.gd")

func _initialize() -> void:
	var h = Harness.new()

	# clamp_action keeps values in [-1, 1]
	h.assert_eq(QM.clamp_action(2.5), 1.0, "clamp_action upper")
	h.assert_eq(QM.clamp_action(-2.5), -1.0, "clamp_action lower")
	h.assert_eq(QM.clamp_action(0.3), 0.3, "clamp_action passthrough")

	# action_to_motor_velocity scales a clamped action by max_speed
	h.assert_eq(QM.action_to_motor_velocity(1.0, 6.0), 6.0, "motor vel full+")
	h.assert_eq(QM.action_to_motor_velocity(-1.0, 6.0), -6.0, "motor vel full-")
	h.assert_eq(QM.action_to_motor_velocity(2.0, 6.0), 6.0, "motor vel clamps before scaling")

	# upright_dot: world-up vs body-up basis column
	h.assert_eq(QM.upright_dot(Vector3.UP), 1.0, "upright fully up")
	h.assert_eq(QM.upright_dot(Vector3.DOWN), -1.0, "upright upside down")
	h.assert_true(absf(QM.upright_dot(Vector3.RIGHT)) < 1e-6, "upright sideways ~0")

	# progress_delta: positive when distance shrinks
	h.assert_eq(QM.progress_delta(10.0, 8.0), 2.0, "progress closed 2")
	h.assert_eq(QM.progress_delta(8.0, 10.0), -2.0, "progress regressed")

	# compose_obs concatenates in the documented order and reports its own size
	# body_up = (7,1,0): with 2 angles + 2 velocities ahead, the up-vector occupies indices 4,5,6
	# in natural x,y,z order — so obs[4]=x=7, obs[5]=y=1.
	var obs: Array = QM.compose_obs([0.1, 0.2], [1.0, 2.0], Vector3(7, 1, 0), Vector3(3, 0, 0), [0.5, 0.5, 0.5], [1.0, 0.0, 1.0, 0.0])
	h.assert_eq(obs.size(), 2 + 2 + 3 + 3 + 3 + 4, "compose_obs total size")
	h.assert_eq(obs[0], 0.1, "compose_obs first joint angle")
	h.assert_eq(obs[4], 7.0, "compose_obs up-vector x at index 4")
	h.assert_eq(obs[5], 1.0, "compose_obs up-vector y at index 5")

	h.finish(self)
