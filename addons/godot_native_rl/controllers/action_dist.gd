class_name ActionDist
extends RefCounted

# Pure deploy-side loader for the continuous-action DiagGaussian std sidecar. The post-inference
# analogue of obs_normalize.gd: SB3 PPO's std is a state-independent learned parameter
# (policy.log_std) that is never in the network output, so an ncnn-converted policy has lost it.
# export_action_dist.py writes std = exp(log_std) as a flat JSON ({"std": [...], "action_dim": N});
# this validates + coerces it. The actual Gaussian draw lives in action_decode.gd's continuous
# branch (it already has the per-segment mean + rng). std is applied positionally across the
# continuous action dims. PPO continuous only; SAC's state-dependent std is out of scope.

# True iff a JSON-decoded dict is well-formed: a non-empty numeric `std` array, and (if present)
# an `action_dim` equal to std.size(). Checked at load so a bad fixture fails loudly up front.
static func validate(stats: Dictionary) -> bool:
	if not stats.has("std"):
		return false
	var std = stats["std"]
	if not (std is Array or std is PackedFloat32Array):
		return false
	if std.size() == 0:
		return false
	for v in std:
		if not (v is float or v is int):
			return false
	if stats.has("action_dim"):
		var action_dim = stats["action_dim"]
		if not (action_dim is int or action_dim is float) or int(action_dim) != std.size():
			return false
	return true

# Coerce a validated JSON dict into a typed PackedFloat32Array once, so the per-frame hot path
# doesn't re-coerce. Returns {} (and push_error) if invalid.
static func to_typed(stats: Dictionary) -> Dictionary:
	if not validate(stats):
		push_error("ActionDist.to_typed: invalid stats dictionary.")
		return {}
	return {"std": PackedFloat32Array(stats["std"])}

# Total continuous action dimensions in an action_space dict — the count the sidecar's `std`
# must match. Discrete keys contribute nothing (their stochasticity is softmax over logits, not a
# sidecar). Used by the controllers to cross-check std.size() at load and fail loud on mismatch,
# so a sidecar exported from the wrong checkpoint never silently samples only some dims.
static func continuous_action_dim(action_space: Dictionary) -> int:
	var n := 0
	for key in action_space.keys():
		var entry: Dictionary = action_space[key]
		if entry.get("action_type", "") == "continuous":
			n += int(entry.get("size", 0))
	return n
