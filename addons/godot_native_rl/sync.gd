class_name NcnnSync
extends Node

enum ControlModes { HUMAN, TRAINING, NCNN_INFERENCE }

var agents_training: Array[Node] = []
var _action_space: Dictionary = {}
var _obs_space: Dictionary = {}

# --- Pure message builders (unit-tested) ---

func build_env_info_message() -> Dictionary:
	return {
		"type": "env_info",
		"observation_space": _obs_space,
		"action_space": _action_space,
		"n_agents": agents_training.size(),
	}

func build_step_message(obs: Array, reward: Array, done: Array) -> Dictionary:
	return {"type": "step", "obs": obs, "reward": reward, "done": done}

func build_reset_message(obs: Array) -> Dictionary:
	return {"type": "reset", "obs": obs}

# Mirrors godot_rl_agents' `_extract_action_dict` verbatim (incl. `index += size`
# for discrete). Used only by the ncnn inference path (added in Part 2). The exact
# discrete encoding for multi-key action spaces is validated against the godot-rl
# package in Part 2; do not change this without checking protocol parity first.
func extract_action_dict(action_array: Array, action_space: Dictionary) -> Dictionary:
	var index := 0
	var result := {}
	for key in action_space.keys():
		var size: int = action_space[key]["size"]
		if action_space[key]["action_type"] == "discrete":
			result[key] = roundi(action_array[index])
		else:
			result[key] = action_array.slice(index, index + size)
		index += size
	return result

# --- Configuration ---
@export var control_mode: ControlModes = ControlModes.TRAINING
@export_range(1, 10, 1, "or_greater") var action_repeat := 8
@export_range(0, 10, 0.1, "or_greater") var speed_up := 1.0

const MAJOR_VERSION := "0"
const MINOR_VERSION := "7"
const DEFAULT_PORT := "11008"
const DEFAULT_SEED := "1"

var stream: StreamPeerTCP = null
var connected := false
var all_agents: Array = []
var agents_heuristic: Array = []
var agents_inference: Array = []
var need_to_send_obs := false
var args = null
var initialized := false
var just_reset := false
var n_action_steps := 0

func _ready() -> void:
	# The Sync node must keep ticking while the SceneTree is paused.
	process_mode = Node.PROCESS_MODE_ALWAYS
	await get_tree().root.ready
	get_tree().set_pause(true)
	_initialize()
	await get_tree().create_timer(1.0).timeout
	get_tree().set_pause(false)

func _initialize() -> void:
	_get_agents()
	args = _get_args()
	Engine.physics_ticks_per_second = int(_get_speedup() * 60)
	Engine.time_scale = _get_speedup() * 1.0
	_set_heuristic("human", all_agents)
	_initialize_training_agents()
	_set_seed()
	_set_action_repeat()
	initialized = true

func _initialize_training_agents() -> void:
	if agents_training.size() > 0:
		_obs_space = agents_training[0].get_obs_space()
		_action_space = agents_training[0].get_action_space()
		connected = connect_to_server()
		if connected:
			_set_heuristic("model", agents_training)
			_handshake()
			_send_env_info()
		else:
			push_warning("NcnnSync: couldn't connect to Python server; using human controls. Start training with `gdrl`.")

func _physics_process(_delta) -> void:
	if n_action_steps % action_repeat != 0:
		n_action_steps += 1
		return
	n_action_steps += 1
	# Each process guards on its own agent list; only the active mode's list is populated.
	_training_process()
	_inference_process()
	_heuristic_process()

func _training_process() -> void:
	if not connected:
		return
	get_tree().set_pause(true)
	if just_reset:
		just_reset = false
		var obs := _get_obs_from_agents(agents_training)
		_send_dict_as_json_message(build_reset_message(obs))
		get_tree().set_pause(false)
		return
	if need_to_send_obs:
		need_to_send_obs = false
		var reward_arr := _get_reward_from_agents()
		var done_arr := _get_done_from_agents()
		var obs := _get_obs_from_agents(agents_training)
		_send_dict_as_json_message(build_step_message(obs, reward_arr, done_arr))
	handle_message()

func _heuristic_process() -> void:
	if agents_heuristic.size() > 0:
		_reset_agents_if_done(agents_heuristic)

func _inference_process() -> void:
	for agent in agents_inference:
		agent.infer_and_act()
	_reset_agents_if_done(agents_inference)

func _get_agents() -> void:
	all_agents = get_tree().get_nodes_in_group("AGENT")
	for agent in all_agents:
		if agent.control_mode == agent.ControlModes.INHERIT_FROM_SYNC:
			match control_mode:
				ControlModes.TRAINING:
					agent.control_mode = agent.ControlModes.TRAINING
				ControlModes.NCNN_INFERENCE:
					agent.control_mode = agent.ControlModes.NCNN_INFERENCE
				_:
					agent.control_mode = agent.ControlModes.HUMAN
		if agent.control_mode == agent.ControlModes.TRAINING:
			agents_training.append(agent)
		elif agent.control_mode == agent.ControlModes.NCNN_INFERENCE:
			agents_inference.append(agent)
		elif agent.control_mode == agent.ControlModes.HUMAN:
			agents_heuristic.append(agent)

func _set_heuristic(h, agents: Array) -> void:
	for agent in agents:
		agent.set_heuristic(h)

func _handshake() -> void:
	var json_dict = _get_dict_json_message()
	assert(json_dict["type"] == "handshake")
	if json_dict.get("major_version") != MAJOR_VERSION:
		push_warning("NcnnSync: major version mismatch (got %s, expected %s)" % [json_dict.get("major_version"), MAJOR_VERSION])
	if json_dict.get("minor_version") != MINOR_VERSION:
		push_warning("NcnnSync: minor version mismatch (got %s, expected %s)" % [json_dict.get("minor_version"), MINOR_VERSION])

func _send_env_info() -> void:
	var json_dict = _get_dict_json_message()
	assert(json_dict["type"] == "env_info")
	_send_dict_as_json_message(build_env_info_message())

func _get_dict_json_message():
	while stream.get_available_bytes() == 0:
		stream.poll()
		if stream.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			print("NcnnSync: server disconnected, closing")
			get_tree().quit()
			return null
		OS.delay_usec(10)
	var message = stream.get_string()
	return JSON.parse_string(message)

func _send_dict_as_json_message(dict) -> void:
	stream.put_string(JSON.stringify(dict, "", false))

func connect_to_server() -> bool:
	OS.delay_msec(1000)
	stream = StreamPeerTCP.new()
	var err := stream.connect_to_host("127.0.0.1", _get_port())
	if err != OK:
		return false
	stream.set_no_delay(true)
	stream.poll()
	while stream.get_status() < StreamPeerTCP.STATUS_CONNECTED:
		stream.poll()
	return stream.get_status() == StreamPeerTCP.STATUS_CONNECTED

func handle_message() -> bool:
	var message = _get_dict_json_message()
	if message == null:
		return false
	match message["type"]:
		"close":
			get_tree().quit()
			get_tree().set_pause(false)
			return true
		"reset":
			_reset_agents()
			just_reset = true
			get_tree().set_pause(false)
			return true
		"call":
			var returns := _call_method_on_agents(message["method"])
			_send_dict_as_json_message({"type": "call", "returns": returns})
			return handle_message()
		"action":
			_set_agent_actions(message["action"], agents_training)
			need_to_send_obs = true
			get_tree().set_pause(false)
			return true
	push_warning("NcnnSync: unhandled message type %s" % message["type"])
	return false

func _call_method_on_agents(method) -> Array:
	var returns := []
	for agent in all_agents:
		returns.append(agent.call(method))
	return returns

func _reset_agents_if_done(agents: Array) -> void:
	for agent in agents:
		if agent.get_done():
			agent.set_done_false()

func _reset_agents() -> void:
	for agent in all_agents:
		agent.needs_reset = true

func _get_obs_from_agents(agents: Array) -> Array:
	var obs := []
	for agent in agents:
		obs.append(agent.get_obs())
	return obs

func _get_reward_from_agents() -> Array:
	var rewards := []
	for agent in agents_training:
		rewards.append(agent.get_reward())
		agent.zero_reward()
	return rewards

func _get_done_from_agents() -> Array:
	var dones := []
	for agent in agents_training:
		var d = agent.get_done()
		if d:
			agent.set_done_false()
		dones.append(d)
	return dones

func _set_agent_actions(actions, agents: Array) -> void:
	for i in range(actions.size()):
		agents[i].set_action(actions[i])

func _get_args() -> Dictionary:
	var arguments := {}
	for argument in OS.get_cmdline_args():
		if argument.find("=") > -1:
			var kv := argument.split("=")
			arguments[kv[0].lstrip("--")] = kv[1]
	return arguments

func _get_speedup() -> float:
	return args.get("speedup", str(speed_up)).to_float()

func _get_port() -> int:
	return args.get("port", DEFAULT_PORT).to_int()

func _set_seed() -> void:
	seed(args.get("env_seed", DEFAULT_SEED).to_int())

func _set_action_repeat() -> void:
	action_repeat = args.get("action_repeat", str(action_repeat)).to_int()
