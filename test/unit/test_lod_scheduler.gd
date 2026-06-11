extends SceneTree
# Pure tests for the LOD scheduler (#21): the frame cadence that decides when the deliberative net
# runs. No ncnn dependency.

const Harness = preload("res://test/harness.gd")
const LodScheduler = preload("res://addons/godot_native_rl/controllers/lod_scheduler.gd")

func _collect(interval: int, frames: int, changes: Array = []) -> Array:
	var s := LodScheduler.new(interval)
	var out: Array = []
	for i in range(frames):
		out.append(s.tick(i in changes))
	return out

func _initialize() -> void:
	var h := Harness.new()

	# interval=1 -> deliberative every frame.
	h.assert_eq(_collect(1, 4), [true, true, true, true], "interval 1 runs every frame")

	# interval=3 -> deliberative on frame 0, then every 3rd (0,3,6...).
	h.assert_eq(_collect(3, 7), [true, false, false, true, false, false, true],
		"interval 3 runs on frame 0 then every 3rd")

	# A state change forces the deliberative net and re-phases the cadence.
	h.assert_eq(_collect(3, 6, [2]),
		[true, false, true, false, false, true],
		"state change at frame 2 forces deliberative and resets the period")

	# interval < 1 clamps to 1.
	var s0 := LodScheduler.new(0)
	h.assert_eq(s0.get_interval(), 1, "interval 0 clamps to 1")
	var sneg := LodScheduler.new(-5)
	h.assert_eq(sneg.get_interval(), 1, "negative interval clamps to 1")

	# reset() makes the next tick deliberative again (mid-period).
	var s := LodScheduler.new(4)
	s.tick()        # frame 0 -> true (due)
	s.tick()        # frame 1 -> false
	s.reset()
	h.assert_true(s.tick(), "reset() forces the next tick to deliberate")

	# set_interval changes cadence live.
	var s2 := LodScheduler.new(2)
	h.assert_true(s2.tick(), "frame 0 due")
	h.assert_true(not s2.tick(), "frame 1 not due (interval 2)")
	h.assert_true(s2.tick(), "frame 2 due")

	h.finish(self)
