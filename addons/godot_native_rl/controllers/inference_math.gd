class_name InferenceMath
extends RefCounted

# Pure helpers for the deploy-side inference path. argmax mirrors the C++
# run_discrete_action selection: first index wins on ties; empty input -> -1
# (the error sentinel) so callers handle the image and float paths uniformly.
static func argmax(values: PackedFloat32Array) -> int:
	if values.is_empty():
		return -1
	var best_index := 0
	var best_value := values[0]
	for i in range(1, values.size()):
		if values[i] > best_value:
			best_value = values[i]
			best_index = i
	return best_index

# Numerically stable softmax: subtract the max logit before exp so large logits don't overflow.
# Empty input -> empty output. A degenerate non-positive sum (e.g. all -inf logits) -> a uniform
# distribution rather than an unusable zero vector. Builds a fresh array (no in-place mutation).
static func softmax(logits: PackedFloat32Array) -> PackedFloat32Array:
	if logits.is_empty():
		return PackedFloat32Array()
	var max_logit := logits[0]
	for v in logits:
		if v > max_logit:
			max_logit = v
	var exps := PackedFloat32Array()
	exps.resize(logits.size())
	var total := 0.0
	for i in range(logits.size()):
		var e := exp(logits[i] - max_logit)
		exps[i] = e
		total += e
	var probs := PackedFloat32Array()
	probs.resize(logits.size())
	if total <= 0.0:
		probs.fill(1.0 / logits.size())
		return probs
	for i in range(exps.size()):
		probs[i] = exps[i] / total
	return probs

# Inverse-CDF categorical sample. `u` is a uniform draw expected in [0, 1): walk the cumulative
# sum and return the first index whose running total exceeds u. Float drift or u >= total clamps
# to the last index (never out of range). Empty input -> -1 (same sentinel as argmax). Leading
# zero-probability buckets are skipped (u < cumulative stays false while cumulative is 0).
# A u below the first cumulative bound (including negative u) yields index 0.
static func sample_categorical(probs: PackedFloat32Array, u: float) -> int:
	if probs.is_empty():
		return -1
	var cumulative := 0.0
	for i in range(probs.size()):
		cumulative += probs[i]
		if u < cumulative:
			return i
	return probs.size() - 1
