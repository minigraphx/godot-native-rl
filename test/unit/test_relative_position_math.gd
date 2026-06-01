extends SceneTree

const Harness = preload("res://test/harness.gd")
const RelativePositionMath = preload("res://addons/godot_native_rl/sensors/relative_position_math.gd")

func _approx(h: Harness, out: Array, expected: Array, label: String) -> void:
	var ok := out.size() == expected.size()
	for i in range(mini(out.size(), expected.size())):
		if absf(float(out[i]) - float(expected[i])) > 1e-5:
			ok = false
	h.assert_true(ok, "%s (got %s, want %s)" % [label, str(out), str(expected)])

func _initialize() -> void:
	var h := Harness.new()

	# Target straight ahead (+X), unrotated sensor -> dir (1,0), dist 10/100
	_approx(h, RelativePositionMath.encode_2d(Vector2(10, 0), 0.0, 100.0), [1.0, 0.0, 0.1], "2d ahead, no rotation")

	# Sensor yawed +90deg: a world +X target reads as local (0,-1)
	_approx(h, RelativePositionMath.encode_2d(Vector2(10, 0), PI / 2.0, 100.0), [0.0, -1.0, 0.1], "2d rotation rotates direction")

	# Distance is rotation-invariant and clips at max_distance
	_approx(h, RelativePositionMath.encode_2d(Vector2(200, 0), 0.0, 100.0), [1.0, 0.0, 1.0], "2d distance clips to 1")
	_approx(h, RelativePositionMath.encode_2d(Vector2(50, 0), 0.0, 100.0), [1.0, 0.0, 0.5], "2d half distance -> 0.5")

	# Zero offset -> zero direction + zero distance
	_approx(h, RelativePositionMath.encode_2d(Vector2.ZERO, 0.0, 100.0), [0.0, 0.0, 0.0], "2d zero offset")

	# max_distance <= 0 -> dist_norm guarded to 0 (direction still valid)
	_approx(h, RelativePositionMath.encode_2d(Vector2(10, 0), 0.0, 0.0), [1.0, 0.0, 0.0], "2d max_distance 0 guard")

	# Direction is unit length for a non-axis-aligned offset
	var out: Array = RelativePositionMath.encode_2d(Vector2(3, 4), 0.0, 100.0)
	h.assert_true(absf(Vector2(out[0], out[1]).length() - 1.0) < 1e-5, "2d direction is unit length")

	h.finish(self)
