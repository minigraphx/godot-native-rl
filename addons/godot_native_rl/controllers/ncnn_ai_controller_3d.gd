class_name NcnnAIController3D
extends Node3D

const RewardAdapterScript = preload("res://addons/godot_native_rl/reward/reward_adapter.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")
const ObsNormalize = preload("res://addons/godot_native_rl/controllers/obs_normalize.gd")
const ActionDist = preload("res://addons/godot_native_rl/controllers/action_dist.gd")
const RecurrentState = preload("res://addons/godot_native_rl/controllers/recurrent_state.gd")

# Emitted once per inference decision with an immutable debug payload consumed by
# PolicyDebugOverlay. Keys: agent_name:String, obs:PackedFloat32Array (normalized vector fed to
# the net; [] on the image path), obs_image:Dictionary ({"w","h","c"} or {}), logits:PackedFloat32Array
# (raw network output, pre-decode), action_space:Dictionary, action:Dictionary (decoded),
# deterministic:bool. Inert when no listener is connected.
signal inference_step(debug: Dictionary)

enum ControlModes { INHERIT_FROM_SYNC, HUMAN, TRAINING, NCNN_INFERENCE }
@export var control_mode: ControlModes = ControlModes.INHERIT_FROM_SYNC  # read/written by NcnnSync
@export var reset_after := 1000
@export_file("*.param") var model_param_path: String = ""
@export_file("*.bin") var model_bin_path: String = ""
@export var input_blob_name: String = "in0"
@export var output_blob_name: String = "out0"
@export_file("*.json") var obs_norm_stats_path: String = ""
@export_file("*.json") var action_dist_stats_path: String = ""  # continuous DiagGaussian std sidecar
@export_file("*.json") var recurrent_stats_path: String = ""  # LSTM/GRU deploy: <model>.recurrent.json
@export var policy_name: String = "shared_policy"  # multi-policy routing (PettingZoo/RLlib)
@export var policy_group: String = ""  # distinct training identity, honored only when Sync.multi_policy=true (#73)
@export var deterministic_inference: bool = true  # false -> sample stochastically: discrete from softmax(logits), continuous DiagGaussian when an action_dist sidecar is set
@export var inference_seed: int = -1  # -1 = randomize each run; >= 0 = fixed seed (reproducible eval)

var _core := NcnnControllerCore.new()
var _ncnn_runner = null
var _reward_adapters: Array = []

# --- Forwarding properties: preserve the historical public state API (subclasses + NcnnSync) ---
var done: bool:
	get:
		return _core.done
	set(value):
		_core.done = value
var reward: float:
	get:
		return _core.reward
	set(value):
		_core.reward = value
var n_steps: int:
	get:
		return _core.n_steps
	set(value):
		_core.n_steps = value
var needs_reset: bool:
	get:
		return _core.needs_reset
	set(value):
		_core.needs_reset = value
var heuristic: String:
	get:
		return _core.heuristic
	set(value):
		_core.heuristic = value
var reward_source:
	get:
		return _core.reward_source
	set(value):
		_core.reward_source = value

func _ready() -> void:
	add_to_group("AGENT")
	collect_reward_adapters()
	if control_mode == ControlModes.NCNN_INFERENCE:
		_setup_ncnn_runner()
		_load_obs_norm_stats()
		_load_action_dist_stats()
		_load_recurrent_stats()
		_core.deterministic_inference = deterministic_inference
		_core.setup_rng(inference_seed)

func _setup_ncnn_runner() -> void:
	if model_param_path.is_empty() or model_bin_path.is_empty():
		push_error("NcnnAIController3D: NCNN_INFERENCE mode requires model_param_path and model_bin_path.")
		return
	_ncnn_runner = NcnnRunner.new()
	_ncnn_runner.input_blob_name = input_blob_name
	_ncnn_runner.output_blob_name = output_blob_name
	add_child(_ncnn_runner)
	var param_bytes := FileAccess.get_file_as_bytes(model_param_path)
	var bin_bytes := FileAccess.get_file_as_bytes(model_bin_path)
	if param_bytes.is_empty() or bin_bytes.is_empty():
		push_error("NcnnAIController3D: cannot read model files '%s' / '%s'." % [model_param_path, model_bin_path])
		_ncnn_runner.queue_free()
		_ncnn_runner = null
		return
	if not _ncnn_runner.load_model_from_buffers(param_bytes, bin_bytes):
		push_error("NcnnAIController3D: failed to load ncnn model.")
		_ncnn_runner.queue_free()
		_ncnn_runner = null

func set_ncnn_runner_for_test(runner) -> void:
	_ncnn_runner = runner

# Hot-swap the deployed policy at runtime (NCNN_INFERENCE only): reloads the ncnn model in place,
# in the same scene, without recreating the controller or runner. Used by demos/tooling to compare
# policies live (e.g. trained vs untrained) — the same scene, a different .ncnn pair, different
# behaviour, no recompile and no Python. Returns true on success; on failure it push_errors and
# leaves the previously loaded model active (returns false). Carried recurrent state, if any, is
# zeroed since the new policy may have a different memory shape.
func swap_model(param_path: String, bin_path: String) -> bool:
	if control_mode != ControlModes.NCNN_INFERENCE:
		push_error("NcnnAIController3D.swap_model: only valid in NCNN_INFERENCE mode.")
		return false
	if param_path.is_empty() or bin_path.is_empty():
		push_error("NcnnAIController3D.swap_model: param_path and bin_path must be non-empty.")
		return false
	if _ncnn_runner == null:
		# Runner not yet created (e.g. initial load failed) — set up fresh from the new paths.
		model_param_path = param_path
		model_bin_path = bin_path
		_setup_ncnn_runner()
		return _ncnn_runner != null
	var param_bytes := FileAccess.get_file_as_bytes(param_path)
	var bin_bytes := FileAccess.get_file_as_bytes(bin_path)
	if param_bytes.is_empty() or bin_bytes.is_empty():
		push_error("NcnnAIController3D.swap_model: cannot read model files '%s' / '%s'." % [param_path, bin_path])
		return false
	if not _ncnn_runner.load_model_from_buffers(param_bytes, bin_bytes):
		push_error("NcnnAIController3D.swap_model: failed to load ncnn model '%s'." % param_path)
		return false
	model_param_path = param_path
	model_bin_path = bin_path
	_core.init_recurrent_state()  # clear stale hidden state across a policy swap (no-op if feed-forward)
	return true

func _load_obs_norm_stats() -> void:
	if obs_norm_stats_path.is_empty():
		return
	var f := FileAccess.open(obs_norm_stats_path, FileAccess.READ)
	if f == null:
		push_error("NcnnAIController3D: cannot open obs_norm_stats_path '%s'." % obs_norm_stats_path)
		return
	var text := f.get_as_text()
	f.close()
	var parsed = JSON.parse_string(text)
	if not (parsed is Dictionary) or not ObsNormalize.validate(parsed):
		push_error("NcnnAIController3D: invalid obs-norm stats JSON at '%s'." % obs_norm_stats_path)
		return
	_core.obs_norm_stats = ObsNormalize.to_typed(parsed)

func set_obs_norm_stats_for_test(stats: Dictionary) -> void:
	_core.obs_norm_stats = stats

func _load_action_dist_stats() -> void:
	if action_dist_stats_path.is_empty():
		return
	var f := FileAccess.open(action_dist_stats_path, FileAccess.READ)
	if f == null:
		push_error("NcnnAIController3D: cannot open action_dist_stats_path '%s'." % action_dist_stats_path)
		return
	var text := f.get_as_text()
	f.close()
	var parsed = JSON.parse_string(text)
	if not (parsed is Dictionary) or not ActionDist.validate(parsed):
		push_error("NcnnAIController3D: invalid action-dist stats JSON at '%s'." % action_dist_stats_path)
		return
	var typed := ActionDist.to_typed(parsed)
	# Fail loud if std length doesn't match the policy's continuous action dims — a sidecar from
	# the wrong checkpoint would otherwise silently sample only some dims (action_decode falls back
	# to the mean for any dim beyond std.size()).
	var cont_dim := ActionDist.continuous_action_dim(get_action_space())
	if typed["std"].size() != cont_dim:
		push_error("NcnnAIController3D: action-dist std has %d entries but the action space has %d continuous dim(s) at '%s'. Re-export the sidecar from the matching checkpoint." % [typed["std"].size(), cont_dim, action_dist_stats_path])
		return
	_core.action_dist_stats = typed

func set_action_dist_for_test(stats: Dictionary) -> void:
	_core.action_dist_stats = stats

func _load_recurrent_stats() -> void:
	if recurrent_stats_path.is_empty():
		return
	var f := FileAccess.open(recurrent_stats_path, FileAccess.READ)
	if f == null:
		push_error("NcnnAIController3D: cannot open recurrent_stats_path '%s'." % recurrent_stats_path)
		return
	var text := f.get_as_text()
	f.close()
	var parsed = JSON.parse_string(text)
	if not (parsed is Dictionary) or not RecurrentState.validate(parsed):
		push_error("NcnnAIController3D: invalid recurrent contract JSON at '%s'." % recurrent_stats_path)
		return
	_core.recurrent_contract = RecurrentState.to_typed(parsed)
	_core.init_recurrent_state()

func set_recurrent_contract_for_test(path: String) -> void:
	recurrent_stats_path = path
	_load_recurrent_stats()

# Public: zero the recurrent hidden state. Call at episode boundaries when the game manages its
# own lifecycle without routing through reset(). No-op for feed-forward policies.
func reset_recurrent_state() -> void:
	_core.init_recurrent_state()

func set_stochastic_for_test(deterministic: bool, seed_value: int) -> void:
	_core.deterministic_inference = deterministic
	_core.setup_rng(seed_value)

func infer_and_act() -> void:
	_core.choose_and_apply_action(self, _ncnn_runner)

# --- Abstract: implemented by the concrete agent ---
func get_obs() -> Dictionary:
	assert(false, "get_obs must be implemented by the agent extending NcnnAIController3D")
	return {"obs": []}

func get_reward() -> float:
	assert(false, "get_reward must be implemented by the agent extending NcnnAIController3D")
	return 0.0

func get_action_space() -> Dictionary:
	assert(false, "get_action_space must be implemented by the agent extending NcnnAIController3D")
	return {}

func set_action(_action) -> void:
	assert(false, "set_action must be implemented by the agent extending NcnnAIController3D")

# Return the flat action array for expert-demo recording (godot_rl get_action() parity).
# A scripted-expert or human-controlled agent overrides this: it decides the action,
# applies it (so the avatar moves via the agent's own _physics_process), and returns the
# flat array recorded into the demo file. Default asserts — only required when recording.
func get_action() -> Array:
	assert(false, "get_action must be implemented by the agent to record expert demos")
	return []

# Override in an image agent to return the live frame for native inference, e.g.
# `return _camera.get_image()`. Non-null routes infer_and_act through run_inference_image.
func get_inference_image() -> Image:
	return null

# Optional per-agent info (godot_rl reads response.get("info", ...)); default empty.
# Agents may override to return e.g. {"is_success": true}.
func get_info() -> Dictionary:
	return {}

# --- Concrete contract methods used by NcnnSync (delegate to the shared core) ---
func get_obs_space() -> Dictionary:
	return NcnnControllerCore.obs_space_from_obs(get_obs())

# Convenience: concatenate all child flat-sensor observations (recursive, tree order).
# Agents can write `return {"obs": collect_sensors()}` in get_obs().
func collect_sensors() -> Array:
	return NcnnControllerCore.collect_sensors(self)

func reset() -> void:
	_core.reset()
	for sensor in NcnnControllerCore.collect_sensors_nodes(self):
		if sensor.has_method("reset"):
			sensor.reset()

func reset_if_done() -> void:
	_core.reset_if_done()

func set_heuristic(h: String) -> void:
	_core.set_heuristic(h)

func get_done() -> bool:
	return _core.get_done()

func set_done_false() -> void:
	_core.set_done_false()

func zero_reward() -> void:
	_core.zero_reward()

func collect_reward_adapters() -> void:
	_reward_adapters.clear()
	for child in get_children():
		if child is RewardAdapterScript:
			_reward_adapters.append(child)

func accumulate_reward() -> void:
	_core.accumulate(_reward_adapters, self)

func _physics_process(_delta) -> void:
	_core.step(reset_after)
