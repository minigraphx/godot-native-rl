class_name NcnnControllerCore
extends RefCounted

# Node-agnostic episode + reward state machine shared by NcnnAIController2D/3D.
# Holds no Node references; reset_after is passed into step() by the wrapper so the
# wrapper stays the single source of truth for that exported value.

const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")
const ObsNormalize = preload("res://addons/godot_native_rl/controllers/obs_normalize.gd")
const RecurrentState = preload("res://addons/godot_native_rl/controllers/recurrent_state.gd")

var done: bool = false
var reward: float = 0.0
var n_steps: int = 0
var needs_reset: bool = false
var heuristic: String = "human"
var reward_source = null
var obs_norm_stats: Dictionary = {}

# Recurrent (LSTM/GRU) deploy: a typed contract (from RecurrentState.to_typed of a
# <model>.recurrent.json sidecar). Empty -> feed-forward (current behavior, zero overhead).
var recurrent_contract: Dictionary = {}
var recurrent_state: Dictionary = {}  # blob_name -> PackedFloat32Array (carried across frames)

func init_recurrent_state() -> void:
	if recurrent_contract.is_empty():
		recurrent_state = {}
		return
	recurrent_state = RecurrentState.zero_state(recurrent_contract)

# Stochastic deploy: when false, discrete actions are sampled from softmax(logits) via `rng`
# instead of argmax. Continuous actions are unaffected (mean). Set by the controller wrappers.
var deterministic_inference: bool = true
var rng: RandomNumberGenerator = RandomNumberGenerator.new()

# seed_value < 0 -> randomize each run; >= 0 -> fixed seed for reproducible stochastic eval.
func setup_rng(seed_value: int) -> void:
	if seed_value < 0:
		rng.randomize()
	else:
		rng.seed = seed_value

func step(reset_after: int) -> void:
	n_steps += 1
	if n_steps > reset_after:
		# Signal episode termination (godot_rl convention): the trainer reads `done`,
		# which gives proper episode boundaries and reward statistics.
		needs_reset = true
		done = true

func reset() -> void:
	n_steps = 0
	needs_reset = false
	init_recurrent_state()  # recurrent policies must not carry memory across episodes

func reset_if_done() -> void:
	if done:
		reset()

func zero_reward() -> void:
	reward = 0.0

func set_done_false() -> void:
	done = false

func get_done() -> bool:
	return done

func set_heuristic(h: String) -> void:
	heuristic = h

func accumulate(adapters: Array, ctx) -> void:
	if reward_source != null:
		reward += reward_source.evaluate(ctx)
	for adapter in adapters:
		reward += adapter.drain()

# Run native ncnn inference and apply the decoded action(s) to the agent. Uses the image path
# when the agent supplies a live frame (get_inference_image()), else the float-vector path.
# The raw output is decoded against agent.get_action_space() via ActionDecode, so discrete,
# continuous, multi-discrete, and multiple simultaneous action keys all deploy. No-op when the
# runner is missing/unloaded. The agent Node is passed in, never stored (core stays node-agnostic).
func choose_and_apply_action(agent, runner) -> void:
	if runner == null or not runner.is_model_loaded():
		return
	var output: PackedFloat32Array
	var img: Image = agent.get_inference_image()
	if img != null:
		output = runner.run_inference_image(img, true)
	else:
		var obs_dict: Dictionary = agent.get_obs()
		assert("obs" in obs_dict, "get_obs() must return a dictionary with an 'obs' key")
		var obs_vec := PackedFloat32Array(obs_dict["obs"])
		if not obs_norm_stats.is_empty():
			obs_vec = ObsNormalize.normalize(obs_vec, obs_norm_stats["mean"], obs_norm_stats["var"],
				obs_norm_stats["epsilon"], obs_norm_stats["clip_obs"])
			if obs_vec.is_empty():
				push_error("NcnnControllerCore.choose_and_apply_action: obs normalization failed (size mismatch); skipping action.")
				return
		if recurrent_contract.is_empty():
			output = runner.run_inference(obs_vec)
		else:
			output = _run_recurrent_and_advance(runner, obs_vec)
	var action: Dictionary = ActionDecode.decode_actions(output, agent.get_action_space(), deterministic_inference, rng)
	if action.is_empty():
		push_error("NcnnControllerCore.choose_and_apply_action: action decode failed (empty/mismatched output); skipping action.")
		return
	agent.set_action(action)

# Build the godot_rl observation_space from a sample get_obs() dict. Numeric-vector values
# become {"size": [len], "space": "box"}. String values are image (hex) obs whose shape can't
# be inferred from the value — the agent merges those from the sensor's get_obs_space_entry().
static func obs_space_from_obs(obs: Dictionary) -> Dictionary:
	var space := {}
	for key in obs.keys():
		var value = obs[key]
		if value is String:
			continue
		space[key] = {"size": [value.size()], "space": "box"}
	return space

# Recursively gather flat sensors under `root` (duck-typed) in stable scene-tree order and
# concatenate their observations into one flat Array. Nodes without obs_size() (e.g.
# CameraSensor, which returns a hex String under its own obs key) are skipped — compose
# those manually. Depth-first pre-order over get_children() -> deterministic obs layout.
static func collect_sensors(root: Node) -> Array:
	var out: Array = []
	_gather_sensor_obs(root, out)
	return out

static func _gather_sensor_obs(node: Node, out: Array) -> void:
	for child in node.get_children():
		if child.has_method("get_observation") and child.has_method("obs_size"):
			out.append_array(child.get_observation())
		_gather_sensor_obs(child, out)

# Recurrent inference: feed obs + the carried state blobs into run_inference_multi, store the
# returned next-state blobs for the following frame, and return the action_output blob to decode.
# On any failure (empty result, or a result missing an expected blob) it push_errors and returns
# an empty array WITHOUT advancing state, so the caller skips set_action.
func _run_recurrent_and_advance(runner, obs_vec: PackedFloat32Array) -> PackedFloat32Array:
	if recurrent_state.is_empty():
		init_recurrent_state()
	var inputs: Array = [{
		"name": recurrent_contract["obs_input"],
		"data": obs_vec,
		"shape": recurrent_contract["obs_shape"],
	}]
	var out_names := PackedStringArray([recurrent_contract["action_output"]])
	for pair in recurrent_contract["state_pairs"]:
		inputs.append({"name": pair["in"], "data": recurrent_state[pair["in"]], "shape": pair["shape"]})
		out_names.append(pair["out"])
	var result: Dictionary = runner.run_inference_multi(inputs, out_names)
	if result.is_empty():
		push_error("NcnnControllerCore: recurrent inference returned empty result.")
		return PackedFloat32Array()
	for name in out_names:
		if not result.has(name):
			push_error("NcnnControllerCore: recurrent result missing expected blob '%s'; skipping action." % name)
			return PackedFloat32Array()
	for pair in recurrent_contract["state_pairs"]:
		recurrent_state[pair["in"]] = result[pair["out"]]
	return result[recurrent_contract["action_output"]]
