class_name RecurrentState
extends RefCounted

# Pure deploy-side helper for recurrent (LSTM/GRU) policies — the state-management analogue of
# obs_normalize.gd. Parses + validates a <model>.recurrent.json sidecar describing which blobs
# carry hidden state across frames, and produces zero-initialized state. The controller reads ALL
# shapes/names from here, so nothing about the recurrent contract is hardcoded.
#
# Sidecar schema:
#   { "obs_input": "in0", "obs_shape": [5], "action_output": "out0",
#     "state_pairs": [ { "in": "in1", "out": "out1", "shape": [8] }, ... ] }

static func _is_positive_int_array(v) -> bool:
	if not (v is Array or v is PackedInt32Array) or v.size() == 0:
		return false
	for x in v:
		if not (x is int or x is float) or int(x) <= 0:
			return false
	return true

# A shape the recurrent deploy path can feed to ncnn as a flat Mat(w): 1-D, positive. Recurrent
# deploy is the flat-vector path — obs and each hidden-state tensor are 1-D — because the controller
# builds Mat(w) with w == element count. A 2-D shape like [1,8] would silently set w=1 (ncnn LSTM
# reads hidden_size from w) and corrupt inference, so reject multi-dim shapes loudly at load.
static func _is_1d_positive_int_array(v) -> bool:
	return _is_positive_int_array(v) and v.size() == 1

# True iff a JSON-decoded contract is well-formed. Checked at load so a bad fixture fails loudly
# up front, not at the first inference frame (or, worse, as silently wrong inference).
static func validate(contract: Dictionary) -> bool:
	if not (contract.has("obs_input") and contract.has("obs_shape")
			and contract.has("action_output") and contract.has("state_pairs")):
		return false
	if not (contract["obs_input"] is String) or not (contract["action_output"] is String):
		return false
	if not _is_1d_positive_int_array(contract["obs_shape"]):
		return false
	if not (contract["state_pairs"] is Array):
		return false
	var pairs: Array = contract["state_pairs"]
	if pairs.size() == 0:
		return false
	# Blob names must be unique within inputs and within outputs: a duplicate input slot, or an
	# action_output colliding with a state output, silently corrupts the carried state (one slot
	# overwrites another, or action logits get written into hidden state).
	var input_names := {contract["obs_input"]: true}
	var output_names := {contract["action_output"]: true}
	for pair in pairs:
		if not (pair is Dictionary):
			return false
		if not (pair.has("in") and pair.has("out") and pair.has("shape")):
			return false
		if not (pair["in"] is String) or not (pair["out"] is String):
			return false
		if not _is_1d_positive_int_array(pair["shape"]):
			return false
		if input_names.has(pair["in"]) or output_names.has(pair["out"]):
			return false
		input_names[pair["in"]] = true
		output_names[pair["out"]] = true
	return true

# Coerce a validated contract into typed arrays once, so the per-frame hot path doesn't re-coerce.
# Returns {} (and push_error) if invalid.
static func to_typed(contract: Dictionary) -> Dictionary:
	if not validate(contract):
		push_error("RecurrentState.to_typed: invalid recurrent contract.")
		return {}
	var pairs: Array = []
	for pair in contract["state_pairs"]:
		pairs.append({
			"in": String(pair["in"]),
			"out": String(pair["out"]),
			"shape": PackedInt32Array(pair["shape"]),
		})
	return {
		"obs_input": String(contract["obs_input"]),
		"obs_shape": PackedInt32Array(contract["obs_shape"]),
		"action_output": String(contract["action_output"]),
		"state_pairs": pairs,
	}

# Product of a shape's dimensions (element count).
static func shape_product(shape: PackedInt32Array) -> int:
	var n := 1
	for d in shape:
		n *= d
	return n

# Zero-initialized state: { pair.in: PackedFloat32Array(zeros, len == product(pair.shape)) }.
# Caller must pass a to_typed(...) result (relies on the typed "state_pairs" + "shape" entries).
static func zero_state(typed_contract: Dictionary) -> Dictionary:
	var state: Dictionary = {}
	for pair in typed_contract["state_pairs"]:
		var vec := PackedFloat32Array()
		vec.resize(shape_product(pair["shape"]))  # resize zero-fills
		state[pair["in"]] = vec
	return state
