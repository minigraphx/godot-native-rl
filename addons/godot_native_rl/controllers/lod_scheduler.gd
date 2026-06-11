extends RefCounted
# Pure level-of-detail scheduler (#21): decides, per frame, whether the expensive "deliberative"
# policy should run this frame — otherwise the cheap "reflex" policy carries the frame. The
# deliberative net runs on the first frame, then every `interval` frames, or immediately on a
# significant state change. No engine/ncnn dependency, so it's unit-testable in isolation; the
# NcnnLODRunner node wires two NcnnRunners through it.

var _interval: int = 1
var _since: int = 0  # frames since the last deliberative run; starts "due" so frame 0 deliberates

func _init(interval: int = 1) -> void:
	set_interval(interval)
	reset()

## interval N: the deliberative net runs every N frames (clamped to >= 1). N=1 = every frame.
func set_interval(interval: int) -> void:
	_interval = maxi(interval, 1)

func get_interval() -> int:
	return _interval

## Reset to "deliberative due next tick" — call on episode reset so a fresh episode gets an
## accurate decision on its first frame.
func reset() -> void:
	_since = _interval

## Advance one frame. Returns true if the deliberative net should run this frame: when a
## significant state change is signalled, or `interval` frames have elapsed since the last run.
func tick(state_changed: bool = false) -> bool:
	if state_changed or _since >= _interval:
		_since = 1
		return true
	_since += 1
	return false
