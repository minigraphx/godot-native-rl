extends Node

# Online observation-normalization sensor wrapper (#18). Wraps exactly ONE inner flat-float sensor
# (a child with get_observation()/obs_size()) and normalizes its output with a running mean/variance
# (Welford), matching SB3 VecNormalize: (x - mean) / sqrt(var + epsilon), clipped to +/- clip_obs.
#
# update-then-normalize each step (matches VecNormalize). Set update_stats=false to FREEZE at deploy.
# Persist with save_stats(path); set stats_path to load frozen stats on _ready(). Stats persist
# ACROSS episodes (no reset() — deliberately absent from sensor reset propagation).
#
# Dimension-agnostic ON PURPOSE (only touches the inner flat float Array): no _2d/_3d split.

const RunningStats = preload("res://addons/godot_native_rl/sensors/running_stats.gd")

## When true, the running stats are updated each step (training). Set false to freeze (deploy).
@export var update_stats: bool = true
## Numerical floor, matches SB3 VecNormalize default.
@export var epsilon: float = 1e-8
## Normalized values are clipped to [-clip_obs, +clip_obs], matches SB3 VecNormalize default.
@export var clip_obs: float = 10.0
## Optional path to a stats sidecar JSON; if it exists, loaded on _ready().
@export var stats_path: String = ""

var _stats  # RunningStats
var _warned_no_inner := false

func _ready() -> void:
	_stats = RunningStats.new()
	if stats_path != "" and FileAccess.file_exists(stats_path):
		var f := FileAccess.open(stats_path, FileAccess.READ)
		if f != null:
			var parsed = JSON.parse_string(f.get_as_text())
			f.close()
			if typeof(parsed) == TYPE_DICTIONARY:
				_stats.from_dict(parsed)
			else:
				push_error("RunningNormSensor: stats_path %s is not a JSON object." % stats_path)

func obs_size() -> int:
	var inner := _find_inner()
	return 0 if inner == null else inner.obs_size()

func get_observation() -> Array:
	var inner := _find_inner()
	if inner == null:
		return []
	_ensure_stats()
	var x: Array = inner.get_observation()
	if update_stats:
		_stats.update(x)
	return _normalize(x)

func save_stats(path: String) -> void:
	_ensure_stats()
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		push_error("RunningNormSensor.save_stats: cannot open %s for write." % path)
		return
	f.store_string(JSON.stringify(_stats.to_dict()))
	f.close()

func stats_count() -> int:
	_ensure_stats()
	return _stats.count

func _normalize(x: Array) -> Array:
	var mean: Array = _stats.mean
	var var_arr: Array = _stats.variance()
	var out: Array = []
	for i in x.size():
		var m: float = mean[i] if i < mean.size() else 0.0
		var v: float = var_arr[i] if i < var_arr.size() else 0.0
		var z := (float(x[i]) - m) / sqrt(v + epsilon)
		out.append(clampf(z, -clip_obs, clip_obs))
	return out

func _ensure_stats() -> void:
	if _stats == null:
		_stats = RunningStats.new()

func _find_inner() -> Node:
	var found: Node = null
	for child in get_children():
		if child.has_method("get_observation") and child.has_method("obs_size"):
			if found != null:
				_warn_inner("RunningNormSensor: more than one inner sensor child; expected exactly one.")
				return null
			found = child
	if found == null:
		_warn_inner("RunningNormSensor: no inner sensor child (need a child with get_observation()/obs_size()).")
	return found

func _warn_inner(msg: String) -> void:
	if not _warned_no_inner:
		push_error(msg)
		_warned_no_inner = true
