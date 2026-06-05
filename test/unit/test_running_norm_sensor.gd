extends SceneTree

const Harness = preload("res://test/harness.gd")
const RunningNormSensor = preload("res://addons/godot_native_rl/sensors/running_norm_sensor.gd")

class FakeInner extends Node:
	var _obs: Array = [0.0]
	func set_obs(o: Array) -> void:
		_obs = o
	func get_observation() -> Array:
		return _obs
	func obs_size() -> int:
		return 1

func _initialize() -> void:
	var h := Harness.new()

	var wrap = RunningNormSensor.new()
	wrap.clip_obs = 10.0
	var inner := FakeInner.new()
	wrap.add_child(inner)

	h.assert_eq(wrap.obs_size(), 1, "obs_size passthrough == 1")

	# First sample: after update count=1, mean=x, var=0 -> (x-mean)/sqrt(eps) == 0.
	inner.set_obs([5.0])
	var o1: Array = wrap.get_observation()
	h.assert_true(absf(o1[0]) < 1e-3, "first sample normalizes to ~0")

	# Feed more samples; output stays clipped within [-clip, clip].
	for v in [10.0, -10.0, 100.0, -100.0]:
		inner.set_obs([v])
		var o: Array = wrap.get_observation()
		h.assert_true(o[0] <= 10.0 + 1e-6 and o[0] >= -10.0 - 1e-6, "clipped within bounds (%s)" % v)

	# Freeze: stats stop updating. Capture count, then read again with update_stats=false.
	var frozen_count: int = wrap.stats_count()
	wrap.update_stats = false
	inner.set_obs([42.0])
	wrap.get_observation()
	h.assert_eq(wrap.stats_count(), frozen_count, "frozen -> count unchanged")

	# Sidecar save -> load round-trip reproduces normalization.
	var path := "user://test_running_norm_stats.json"
	wrap.save_stats(path)
	var wrap2 = RunningNormSensor.new()
	wrap2.stats_path = path
	wrap2.update_stats = false
	var inner2 := FakeInner.new()
	wrap2.add_child(inner2)
	wrap2._ready()  # trigger the load explicitly in the headless test
	inner.set_obs([7.0])
	inner2.set_obs([7.0])
	wrap.update_stats = false
	var a: Array = wrap.get_observation()
	var b: Array = wrap2.get_observation()
	h.assert_true(absf(a[0] - b[0]) < 1e-5, "loaded stats reproduce normalization")

	wrap.free()
	wrap2.free()
	h.finish(self)
