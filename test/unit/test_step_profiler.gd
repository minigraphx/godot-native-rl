extends SceneTree

const Harness = preload("res://test/harness.gd")
const StepProfiler = preload("res://addons/godot_native_rl/net/step_profiler.gd")

func _initialize() -> void:
	var h := Harness.new()

	var p = StepProfiler.new()
	p.record("collect_obs", 100)
	p.record("serialize_send", 300)
	p.record("collect_obs", 100)  # same phase accumulates -> 200
	p.step_done()
	p.step_done()

	h.assert_eq(p.get_phase_usec("collect_obs"), 200, "phase accumulates across records")
	h.assert_eq(p.get_phase_usec("serialize_send"), 300, "second phase tracked")
	h.assert_eq(p.total_usec(), 500, "total is the sum of phases")
	h.assert_eq(p.get_steps(), 2, "step_done counts steps")
	h.assert_true(absf(p.phase_percentage("collect_obs") - 40.0) < 1e-6, "collect_obs is 200/500 = 40%")
	h.assert_true(absf(p.phase_percentage("serialize_send") - 60.0) < 1e-6, "serialize_send is 60%")

	var report := p.format_report()
	h.assert_true(report.find("[step-profile]") != -1, "report lines are tagged for grep")
	h.assert_true(report.find("collect_obs") != -1 and report.find("serialize_send") != -1,
		"report lists each phase")

	# Unknown phase reads as zero, never errors.
	h.assert_eq(p.get_phase_usec("missing"), 0, "unknown phase -> 0 usec")

	# Empty profiler must not divide by zero anywhere.
	var empty = StepProfiler.new()
	h.assert_eq(empty.total_usec(), 0, "empty total is 0")
	h.assert_true(absf(empty.phase_percentage("x") - 0.0) < 1e-6, "empty percentage is 0, no div-by-zero")
	var _r := empty.format_report()  # must not crash with zero steps/total

	h.finish(self)
