class_name ObsNormalize
extends RefCounted

# Pure, stateless replay of SB3 VecNormalize's FROZEN observation normalization. No node, file,
# or tree state — fully headless-unit-testable. This is the deploy-side mitigation for the #1
# silent failure called out in docs/ncnn_vs_onnx.md: a policy trained with VecNormalize learns
# against running mean/std that live in the wrapper (not the network, not get_obs()), so at deploy
# the frozen stats must be replayed game-side or the network silently receives un-normalized obs.
#
# Pinned formula (must match the Python exporter's parity test exactly):
#   norm[i] = clip( (obs[i] - mean[i]) / sqrt(var[i] + epsilon), -clip, +clip )
# SB3 defaults: epsilon = 1e-8, clip_obs = 10.0. Stats are frozen (never updated here).
#
# Immutable: returns a NEW Array; never mutates obs or the stat arrays.

# obs / mean / var_arr: parallel float arrays of equal length.
# epsilon: variance floor (SB3 default 1e-8). clip: |output| bound (SB3 default 10.0; <= 0 disables).
# Guards (fail loud, never silently swallow): on a length mismatch the obs is returned unchanged
# (loudly, via push_error) so the inference loop keeps a stable shape while the misconfiguration is
# surfaced; a non-positive var+epsilon passes the centered value through (no divide) to avoid NaN/inf.
static func normalize(obs: Array, mean: Array, var_arr: Array, epsilon: float, clip: float) -> Array:
	var n := obs.size()
	if mean.size() != n or var_arr.size() != n:
		push_error("ObsNormalize.normalize: stat length mismatch (obs=%d, mean=%d, var=%d); returning obs unchanged." % [n, mean.size(), var_arr.size()])
		return obs.duplicate()
	var do_clip := clip > 0.0
	var out := []
	out.resize(n)
	for i in range(n):
		var centered := float(obs[i]) - float(mean[i])
		var denom_sq := float(var_arr[i]) + epsilon
		var value := centered
		if denom_sq > 0.0:
			value = centered / sqrt(denom_sq)
		else:
			push_error("ObsNormalize.normalize: var[%d] + epsilon <= 0 (%f); passing centered value through." % [i, denom_sq])
		if do_clip:
			value = clampf(value, -clip, clip)
		out[i] = value
	return out
