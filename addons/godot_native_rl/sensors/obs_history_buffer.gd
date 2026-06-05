extends Node

# Frame-stacking sensor wrapper (#17). Wraps exactly ONE inner flat-float sensor (a child with
# get_observation()/obs_size()) and emits the last `history_length` observations concatenated,
# oldest-first newest-last, zero-filled before the window is full.
#
# Dimension-agnostic ON PURPOSE: it only touches the inner sensor's flat float Array, never
# geometry, so there is no _2d/_3d split (unlike the geometry sensors). The inner child carries
# the dimensionality.
#
# Discovery: collect_sensors treats any obs-producing node as a leaf, so this wrapper is collected
# (not its inner child) — no double-count. get_observation() advances the ring; reset() clears it.

const FrameRing = preload("res://addons/godot_native_rl/sensors/frame_ring.gd")

## Number of past observations to stack (window length N).
@export var history_length: int = 4

var _ring  # FrameRing, lazily built once the inner sensor's size is known.
var _warned_no_inner := false

func obs_size() -> int:
	var inner := _find_inner()
	if inner == null:
		return 0
	return history_length * inner.obs_size()

func get_observation() -> Array:
	var inner := _find_inner()
	if inner == null:
		return []
	_ensure_ring(inner.obs_size())
	_ring.push(inner.get_observation())
	return _ring.flat()

func reset() -> void:
	if _ring != null:
		_ring.clear()

func _ensure_ring(frame_size: int) -> void:
	if _ring == null:
		_ring = FrameRing.new(frame_size, history_length)

func _find_inner() -> Node:
	var found: Node = null
	for child in get_children():
		if child.has_method("get_observation") and child.has_method("obs_size"):
			if found != null:
				_warn_inner("ObsHistoryBuffer: more than one inner sensor child; expected exactly one.")
				return null
			found = child
	if found == null:
		_warn_inner("ObsHistoryBuffer: no inner sensor child (need a child with get_observation()/obs_size()).")
	return found

func _warn_inner(msg: String) -> void:
	if not _warned_no_inner:
		push_error(msg)
		_warned_no_inner = true
