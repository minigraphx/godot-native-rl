extends RefCounted

# Pure fixed-size ring of float frames for ObsHistoryBuffer (frame-stacking, #17).
# Holds `length` frames of `frame_size` floats each. Newest frame is emitted LAST by flat();
# slots not yet written read as zeros. No scene-tree / geometry dependencies — unit-testable.

var _frame_size: int
var _length: int
var _frames: Array = []  # Array of Array[float], length == _length, oldest at index 0

func _init(frame_size: int, length: int) -> void:
	_frame_size = max(0, frame_size)
	_length = max(0, length)
	clear()

func clear() -> void:
	_frames = []
	for i in _length:
		_frames.append(_zero_frame())

func push(frame: Array) -> void:
	if frame.size() != _frame_size:
		push_error("FrameRing.push: frame size %d != expected %d; ignored." % [frame.size(), _frame_size])
		return
	_frames.pop_front()              # drop oldest
	_frames.append(frame.duplicate())  # newest at the end (immutable copy)

func flat() -> Array:
	var out: Array = []
	for f in _frames:
		out.append_array(f)
	return out

func _zero_frame() -> Array:
	var z: Array = []
	for i in _frame_size:
		z.append(0.0)
	return z
