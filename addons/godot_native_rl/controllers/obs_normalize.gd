class_name ObsNormalize
extends RefCounted

# Pure deploy-side replay of SB3 VecNormalize observation normalization. The pre-inference analogue
# of action_decode.gd (the post-inference transform). VecNormalize keeps its running mean/var in a
# separate .pkl (not in the policy network), so a converted ncnn model has lost them; this replays
# the exact transform between get_obs() and run_inference():
#   normalized[i] = clamp((obs[i] - mean[i]) / sqrt(var[i] + epsilon), -clip_obs, +clip_obs)
# Size mismatch (a train/deploy shape error) -> empty array (controller skips the action; no silent
# garbage forward pass). Mirrors stable_baselines3.common.vec_env.VecNormalize._normalize_obs.

static func normalize(obs: PackedFloat32Array, mean: PackedFloat32Array,
		var_: PackedFloat32Array, epsilon: float, clip_obs: float) -> PackedFloat32Array:
	if obs.size() != mean.size() or obs.size() != var_.size():
		push_error("ObsNormalize.normalize: size mismatch (obs %d, mean %d, var %d)." % [
			obs.size(), mean.size(), var_.size()])
		return PackedFloat32Array()
	var out := PackedFloat32Array()
	out.resize(obs.size())
	for i in obs.size():
		var z: float = (obs[i] - mean[i]) / sqrt(var_[i] + epsilon)
		out[i] = clampf(z, -clip_obs, clip_obs)
	return out

# True iff a JSON-decoded stats dict is well-formed for normalize(): mean+var present as equal,
# non-empty numeric arrays, plus epsilon and clip_obs keys. Checked at load so a bad fixture fails
# loudly up front, not at the first inference frame.
static func validate(stats: Dictionary) -> bool:
	if not (stats.has("mean") and stats.has("var") and stats.has("epsilon") and stats.has("clip_obs")):
		return false
	var mean = stats["mean"]
	var var_ = stats["var"]
	if not (mean is Array or mean is PackedFloat32Array):
		return false
	if not (var_ is Array or var_ is PackedFloat32Array):
		return false
	if mean.size() == 0 or mean.size() != var_.size():
		return false
	return true

# Coerce a validated JSON stats dict into typed PackedFloat32Arrays + floats once, so the per-frame
# hot path doesn't re-coerce. Returns {} (and push_error) if invalid.
static func to_typed(stats: Dictionary) -> Dictionary:
	if not validate(stats):
		push_error("ObsNormalize.to_typed: invalid stats dictionary.")
		return {}
	return {
		"mean": PackedFloat32Array(stats["mean"]),
		"var": PackedFloat32Array(stats["var"]),
		"epsilon": float(stats["epsilon"]),
		"clip_obs": float(stats["clip_obs"]),
	}
