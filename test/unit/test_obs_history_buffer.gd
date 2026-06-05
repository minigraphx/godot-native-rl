extends SceneTree

const Harness = preload("res://test/harness.gd")
const ObsHistoryBuffer = preload("res://addons/godot_native_rl/sensors/obs_history_buffer.gd")

# Fake inner sensor: returns whatever obs we set; obs_size fixed at 2.
class FakeInner extends Node:
	var _obs: Array = [0.0, 0.0]
	func set_obs(o: Array) -> void:
		_obs = o
	func get_observation() -> Array:
		return _obs
	func obs_size() -> int:
		return 2

func _initialize() -> void:
	var h := Harness.new()

	var wrap = ObsHistoryBuffer.new()
	wrap.history_length = 3
	var inner := FakeInner.new()
	wrap.add_child(inner)

	# obs_size is stable from frame 1: N * inner.obs_size().
	h.assert_eq(wrap.obs_size(), 6, "obs_size = 3 * 2")

	# First read: window zero-filled except newest frame.
	inner.set_obs([1.0, 2.0])
	h.assert_eq(wrap.get_observation(), [0.0, 0.0, 0.0, 0.0, 1.0, 2.0], "warm-up zero-fill, newest last")

	# Second read: older frame slides forward.
	inner.set_obs([3.0, 4.0])
	h.assert_eq(wrap.get_observation(), [0.0, 0.0, 1.0, 2.0, 3.0, 4.0], "second frame")

	# Fill + evict.
	inner.set_obs([5.0, 6.0])
	h.assert_eq(wrap.get_observation(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "filled")
	inner.set_obs([7.0, 8.0])
	h.assert_eq(wrap.get_observation(), [3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "evict oldest")

	# reset() re-zeros the window.
	wrap.reset()
	inner.set_obs([9.0, 9.0])
	h.assert_eq(wrap.get_observation(), [0.0, 0.0, 0.0, 0.0, 9.0, 9.0], "reset clears window")

	wrap.free()

	# No inner child -> obs_size 0, empty obs (fail-loud error printed, no crash).
	var lonely = ObsHistoryBuffer.new()
	h.assert_eq(lonely.obs_size(), 0, "no inner -> obs_size 0")
	h.assert_eq(lonely.get_observation(), [], "no inner -> empty obs")
	lonely.free()

	h.finish(self)
