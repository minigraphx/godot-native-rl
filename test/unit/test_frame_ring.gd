extends SceneTree

const Harness = preload("res://test/harness.gd")
const FrameRing = preload("res://addons/godot_native_rl/sensors/frame_ring.gd")

func _initialize() -> void:
	var h := Harness.new()

	# frame_size 2, length 3 -> flat() is 6 zeros before any push.
	var r := FrameRing.new(2, 3)
	h.assert_eq(r.flat(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "fresh ring -> all zeros")

	# One push -> newest is last, older slots still zero.
	r.push([1.0, 2.0])
	h.assert_eq(r.flat(), [0.0, 0.0, 0.0, 0.0, 1.0, 2.0], "one push -> newest last, zero-filled front")

	# Fill exactly to length, oldest-first newest-last.
	r.push([3.0, 4.0])
	r.push([5.0, 6.0])
	h.assert_eq(r.flat(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "filled -> oldest-first order")

	# Overflow evicts the oldest.
	r.push([7.0, 8.0])
	h.assert_eq(r.flat(), [3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "overflow evicts oldest")

	# clear() re-zeros.
	r.clear()
	h.assert_eq(r.flat(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "clear -> all zeros")

	# A push of the wrong frame size is rejected (no mutation, error path).
	r.push([1.0, 2.0])
	r.push([99.0])  # wrong size -> ignored
	h.assert_eq(r.flat(), [0.0, 0.0, 0.0, 0.0, 1.0, 2.0], "wrong-size push ignored")

	h.finish(self)
