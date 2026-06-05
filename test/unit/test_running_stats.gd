extends SceneTree

const Harness = preload("res://test/harness.gd")
const RunningStats = preload("res://addons/godot_native_rl/sensors/running_stats.gd")

func _approx(a: Array, b: Array, eps: float) -> bool:
	if a.size() != b.size():
		return false
	for i in a.size():
		if absf(float(a[i]) - float(b[i])) > eps:
			return false
	return true

func _initialize() -> void:
	var h := Harness.new()

	var s := RunningStats.new()

	# Samples for two features: [1,10], [2,20], [3,30].
	s.update([1.0, 10.0])
	s.update([2.0, 20.0])
	s.update([3.0, 30.0])

	h.assert_eq(s.count, 3, "count == 3")
	h.assert_true(_approx(s.mean, [2.0, 20.0], 1e-9), "mean == [2,20]")
	# Population variance: feature0 = 2/3, feature1 = 200/3.
	h.assert_true(_approx(s.variance(), [2.0 / 3.0, 200.0 / 3.0], 1e-9), "variance matches naive reference")

	# Round-trip through dict.
	var d := s.to_dict()
	var s2 := RunningStats.new()
	s2.from_dict(d)
	h.assert_eq(s2.count, 3, "round-trip count")
	h.assert_true(_approx(s2.mean, [2.0, 20.0], 1e-9), "round-trip mean")
	h.assert_true(_approx(s2.variance(), [2.0 / 3.0, 200.0 / 3.0], 1e-9), "round-trip variance")

	# Zero-count variance is all zeros (no div-by-zero).
	var empty := RunningStats.new()
	h.assert_eq(empty.variance(), [], "empty -> empty variance")

	h.finish(self)
