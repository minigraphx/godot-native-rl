extends SceneTree

# Headless unit tests for RunSpeed's cmdline parsing (#290/#324): the strict positive-int
# override parse. GDScript's int() prefix-parses ("1e3" -> 1) and turns garbage into 0, which
# downstream becomes a modulo-by-zero (SIGFPE in release) or a truncated overnight run.

const Harness = preload("res://test/harness.gd")
const RunSpeed = preload("res://addons/godot_native_rl/training/run_speed.gd")

func _initialize() -> void:
	var h := Harness.new()

	h.assert_eq(RunSpeed.parse_positive_int({}, "action_repeat", 8), 8, "absent key -> fallback")
	h.assert_eq(RunSpeed.parse_positive_int({"action_repeat": "12"}, "action_repeat", 8), 12, "valid int parses")
	h.assert_eq(RunSpeed.parse_positive_int({"g": "0"}, "g", 5), -1, "zero rejected (modulo-by-zero guard)")
	h.assert_eq(RunSpeed.parse_positive_int({"g": "-3"}, "g", 5), -1, "negative rejected")
	h.assert_eq(RunSpeed.parse_positive_int({"g": "1e3"}, "g", 5), -1, "scientific notation rejected (int() would prefix-parse to 1)")
	h.assert_eq(RunSpeed.parse_positive_int({"g": "garbage"}, "g", 5), -1, "non-numeric rejected (int() would give 0)")

	# parse_int_any (#332 — env_seed): any sign OK, garbage errors + keeps the fallback.
	h.assert_eq(RunSpeed.parse_int_any({}, "env_seed", 1), 1, "int_any absent -> fallback")
	h.assert_eq(RunSpeed.parse_int_any({"env_seed": "-7"}, "env_seed", 1), -7, "int_any negative accepted")
	h.assert_eq(RunSpeed.parse_int_any({"env_seed": "junk"}, "env_seed", 1), 1, "int_any garbage -> fallback (loud)")

	# parse_positive_float (#332 — speedup): garbage/zero rejected with the -1.0 sentinel.
	h.assert_eq(RunSpeed.parse_positive_float({}, "speedup", 2.0), 2.0, "pos_float absent -> fallback")
	h.assert_eq(RunSpeed.parse_positive_float({"speedup": "12.5"}, "speedup", 2.0), 12.5, "pos_float parses")
	h.assert_eq(RunSpeed.parse_positive_float({"speedup": "0"}, "speedup", 2.0), -1.0, "pos_float zero rejected")
	h.assert_eq(RunSpeed.parse_positive_float({"speedup": "junk"}, "speedup", 2.0), -1.0, "pos_float garbage rejected")

	# parse_float_any (#332 — timeouts, '<= 0 disables' convention): explicit 0/negative pass
	# through; only non-numeric falls back (loud).
	h.assert_eq(RunSpeed.parse_float_any({"read_timeout": "0"}, "read_timeout", 60.0), 0.0, "float_any explicit 0 = disable passes through")
	h.assert_eq(RunSpeed.parse_float_any({"read_timeout": "junk"}, "read_timeout", 60.0), 60.0, "float_any garbage -> fallback (loud)")

	h.finish(self)
