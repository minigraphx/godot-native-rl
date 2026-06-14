class_name NcnnCrowdController
extends Node
# Drives a crowd of shared-policy agents with ONE shared NcnnRunner. Each decision gathers every
# child agent's obs, runs a single batched (thread-parallel) forward pass via run_inference_batch,
# decodes each agent's output against its own action space, and scatters set_action() back.
#
# Agents are the controller's children (stable get_children() order -> reproducible batch index ->
# agent mapping). An agent is anything implementing get_obs()/get_action_space()/set_action().
# ncnn has no CPU batch dim: run_inference_batch loops the passes across threads (same FLOPs as N
# single calls, far less dispatch overhead + one shared Net). See the design spec.

const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")
const CrowdMath = preload("res://addons/godot_native_rl/controllers/crowd_math.gd")

@export_file("*.param") var model_param_path: String = ""
@export_file("*.bin") var model_bin_path: String = ""
@export var input_blob_name: String = "in0"
@export var output_blob_name: String = "out0"
@export var num_threads: int = -1  # -1 = hardware_concurrency; 1 = serial; N = cap workers at N
@export var deterministic_inference: bool = true
@export var inference_seed: int = -1
@export var policy_name: String = "shared_policy"  # shown in the Policy Debugger header (all units share it)

var _runner = null
var _agents: Array = []
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	if inference_seed < 0:
		_rng.randomize()
	else:
		_rng.seed = inference_seed
	_setup_runner()
	register_agents()

func _setup_runner() -> void:
	if _runner != null:
		return  # idempotent: never create a second runner (e.g. if called after _ready already ran it)
	if model_param_path.is_empty() or model_bin_path.is_empty():
		push_error("NcnnCrowdController: model_param_path and model_bin_path are required.")
		return
	_runner = NcnnRunner.new()
	_runner.input_blob_name = input_blob_name
	_runner.output_blob_name = output_blob_name
	add_child(_runner)
	var param_bytes := FileAccess.get_file_as_bytes(model_param_path)
	var bin_bytes := FileAccess.get_file_as_bytes(model_bin_path)
	if param_bytes.is_empty() or bin_bytes.is_empty():
		push_error("NcnnCrowdController: cannot read model files '%s' / '%s'." % [model_param_path, model_bin_path])
		_runner.queue_free()
		_runner = null
		return
	if not _runner.load_model_from_buffers(param_bytes, bin_bytes):
		push_error("NcnnCrowdController: failed to load ncnn model.")
		_runner.queue_free()
		_runner = null

func set_runner_for_test(runner) -> void:
	_runner = runner

# Discover crowd agents: direct children implementing the duck-typed agent contract, in
# get_children() (scene-tree) order. The shared NcnnRunner child is skipped (no get_obs()).
func register_agents() -> void:
	_agents.clear()
	for child in get_children():
		if child.has_method("get_obs") and child.has_method("get_action_space") and child.has_method("set_action"):
			_agents.append(child)

func agent_count() -> int:
	return _agents.size()

# One batched decision for the whole crowd. No-op if the runner is missing/unloaded or the crowd is
# empty. An agent whose output slot came back empty (failed inference) is skipped (left on its last
# action) rather than fed a bad decode.
func decide() -> void:
	if _runner == null or not _runner.is_model_loaded() or _agents.is_empty():
		return
	var inputs := CrowdMath.gather_obs(_agents)
	var outputs: Array = _runner.run_inference_batch(inputs, num_threads)
	if outputs.size() != _agents.size():
		push_error("NcnnCrowdController: batch returned %d outputs for %d agents; skipping frame." % [outputs.size(), _agents.size()])
		return
	for i in _agents.size():
		var output: PackedFloat32Array = outputs[i]
		if not CrowdMath.output_usable(output):
			continue
		var agent = _agents[i]
		var action_space: Dictionary = agent.get_action_space()
		var action: Dictionary = ActionDecode.decode_actions(output, action_space, deterministic_inference, _rng, {})
		if action.is_empty():
			push_error("NcnnCrowdController: action decode failed for agent %d; skipping." % i)
			continue
		_emit_debug(agent, inputs[i], output, action, action_space)
		agent.set_action(action)

# Emit the per-unit Policy Debugger payload through the agent node (the unit owns the signal; the
# shared controller drives inference for all of them). Mirrors NcnnControllerCore._emit_debug so the
# existing PolicyDebugOverlay renders crowd units unchanged: it auto-discovers each unit by the
# inference_step signal and keys payloads by the unit's instance id. The identity (policy/model/det)
# lives on the controller, not the unit, so it travels inside the payload's "identity" key — the
# overlay prefers that over probing the emitter node. Inert when a unit doesn't declare the signal
# (only the small Dictionary is built, at decision cadence). (#232)
func _emit_debug(agent, obs_vec: PackedFloat32Array, logits: PackedFloat32Array, action: Dictionary, action_space: Dictionary) -> void:
	if not agent.has_signal("inference_step"):
		return
	agent.emit_signal("inference_step", {
		"agent_name": String(agent.name),
		"obs": obs_vec,
		"obs_image": {},
		"logits": logits,
		"action_space": action_space,
		"action": action,
		"deterministic": deterministic_inference,
		"identity": {
			"policy_name": policy_name,
			"model": model_param_path.get_file() if not model_param_path.is_empty() else "?",
			"deterministic": deterministic_inference,
			"seed": inference_seed,
		},
	})

func _physics_process(_delta: float) -> void:
	decide()
