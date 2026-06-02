class_name ObsNormalizer
extends Node

# Thin loader node for FROZEN SB3 VecNormalize observation statistics. Loads
# {"mean":[...], "var":[...], "epsilon":..., "clip_obs":...} from a JSON file (produced by
# scripts/export_vecnormalize_stats.py) and applies the pinned normalization via ObsNormalize.
# Mirrors the sensors: pure math core + thin node wrapper, with a set_stats_for_test seam so the
# full path is headless-testable without a file. Composition is MANUAL — call normalize(obs) in
# get_obs() before inference (no controller change). References the pure core by path-based
# preload (bare class_name is unreliable headless — see CLAUDE.md).

const ObsNormalize = preload("res://addons/godot_native_rl/obs/obs_normalize.gd")

const DEFAULT_EPSILON := 1e-8   # SB3 VecNormalize default
const DEFAULT_CLIP := 10.0      # SB3 VecNormalize default

@export var stats_path: String = ""

var _mean: Array = []
var _var: Array = []
var _epsilon: float = DEFAULT_EPSILON
var _clip: float = DEFAULT_CLIP
var _loaded: bool = false
var _warned_not_loaded := false

func is_loaded() -> bool:
	return _loaded

# Declared observation size (the stats length), for obs-space declaration. 0 until loaded.
func obs_size() -> int:
	return _mean.size()

# Test seam: inject frozen stats directly, bypassing file IO. Mirrors the sensors'
# set_target_for_test. Validates lengths and fails loud on mismatch (does not load).
func set_stats_for_test(mean: Array, var_arr: Array, epsilon: float, clip: float) -> void:
	if mean.size() != var_arr.size():
		push_error("ObsNormalizer.set_stats_for_test: mean/var length mismatch (%d vs %d)." % [mean.size(), var_arr.size()])
		return
	_apply_stats(mean.duplicate(), var_arr.duplicate(), epsilon, clip)

# Load + validate frozen stats from the JSON at stats_path. Returns false (+ push_error) on any
# failure (missing file, bad JSON, missing/mismatched mean/var); never throws. Frozen after load.
func load_stats() -> bool:
	if stats_path.is_empty():
		push_error("ObsNormalizer.load_stats: stats_path is empty.")
		return false
	if not FileAccess.file_exists(stats_path):
		push_error("ObsNormalizer.load_stats: file not found: %s" % stats_path)
		return false
	var text := FileAccess.get_file_as_string(stats_path)
	if text.is_empty():
		push_error("ObsNormalizer.load_stats: empty or unreadable file: %s" % stats_path)
		return false
	var parsed: Variant = JSON.parse_string(text)
	if typeof(parsed) != TYPE_DICTIONARY:
		push_error("ObsNormalizer.load_stats: JSON root is not an object: %s" % stats_path)
		return false
	var data: Dictionary = parsed
	if not (data.has("mean") and data.has("var")):
		push_error("ObsNormalizer.load_stats: JSON missing 'mean' and/or 'var': %s" % stats_path)
		return false
	var mean: Array = data["mean"]
	var var_arr: Array = data["var"]
	if mean.size() != var_arr.size() or mean.is_empty():
		push_error("ObsNormalizer.load_stats: invalid stats (mean=%d, var=%d): %s" % [mean.size(), var_arr.size(), stats_path])
		return false
	var epsilon: float = float(data.get("epsilon", DEFAULT_EPSILON))
	var clip: float = float(data.get("clip_obs", DEFAULT_CLIP))
	_apply_stats(mean, var_arr, epsilon, clip)
	return true

func _apply_stats(mean: Array, var_arr: Array, epsilon: float, clip: float) -> void:
	_mean = mean
	_var = var_arr
	_epsilon = epsilon
	_clip = clip
	_loaded = true
	_warned_not_loaded = false

# Apply the frozen normalization. Returns a NEW Array (immutable). If no stats are loaded, returns
# obs unchanged (one-time push_error) so a misconfigured node never crashes the inference loop.
func normalize(obs: Array) -> Array:
	if not _loaded:
		if not _warned_not_loaded:
			push_error("ObsNormalizer.normalize: no stats loaded (call load_stats/set_stats_for_test); returning obs unchanged.")
			_warned_not_loaded = true
		return obs.duplicate()
	return ObsNormalize.normalize(obs, _mean, _var, _epsilon, _clip)
