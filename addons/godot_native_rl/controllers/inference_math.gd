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
# Empty input -> empty output. A degenerate zero sum returns the (zero) exps rather than dividing.
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
	if total <= 0.0:
		return exps
	for i in range(exps.size()):
		exps[i] = exps[i] / total
	return exps
