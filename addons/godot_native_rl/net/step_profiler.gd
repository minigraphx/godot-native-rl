extends RefCounted

# Accumulates wall-time per named step phase during training so throughput_compare.sh can show
# WHERE a training step's time goes (sim/obs vs JSON serialize vs the socket round-trip), rather
# than just total samples/sec. Pure + unit-tested; NcnnSync feeds it timestamps only when the
# `profile=true` cmdline flag is set, so it's zero-cost otherwise. Times are microseconds.

var _phase_usec := {}   # phase name -> accumulated usec
var _order: Array = []   # phase names in first-seen order (stable report layout)
var _steps := 0

func record(phase: String, usec: int) -> void:
	if not _phase_usec.has(phase):
		_phase_usec[phase] = 0
		_order.append(phase)
	_phase_usec[phase] += usec

func step_done() -> void:
	_steps += 1

func get_steps() -> int:
	return _steps

func get_phase_usec(phase: String) -> int:
	return _phase_usec.get(phase, 0)

func total_usec() -> int:
	var total := 0
	for phase in _order:
		total += _phase_usec[phase]
	return total

func phase_percentage(phase: String) -> float:
	var total := total_usec()
	if total <= 0:
		return 0.0
	return 100.0 * float(get_phase_usec(phase)) / float(total)

# Human-readable report; every line is tagged `[step-profile]` so a wrapping script can grep it.
func format_report() -> String:
	var total := total_usec()
	var lines: Array = []
	lines.append("[step-profile] steps=%d total=%.1fms" % [_steps, total / 1000.0])
	for phase in _order:
		var usec: int = _phase_usec[phase]
		var per_step: float = (float(usec) / float(_steps)) if _steps > 0 else 0.0
		lines.append("[step-profile]   %-16s %9.1fms  %5.1f%%  %8.1fus/step" % [
			phase, usec / 1000.0, phase_percentage(phase), per_step])
	return "\n".join(lines)
