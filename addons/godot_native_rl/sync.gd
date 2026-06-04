class_name NcnnSync
extends Node

enum ControlModes { HUMAN, TRAINING, NCNN_INFERENCE, RECORD_EXPERT_DEMOS }
const SocketTimeout = preload("res://addons/godot_native_rl/net/socket_timeout.gd")
const PolicyNames = preload("res://addons/godot_native_rl/policy_names.gd")
const StepProfiler = preload("res://addons/godot_native_rl/net/step_profiler.gd")
const DemoRecorder = preload("res://addons/godot_native_rl/training/demo_recorder.gd")

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
		"agent_policy_names": PolicyNames.policy_names_from_agents(agents_training),
	}

func build_step_message(obs: Array, reward: Array, done: Array, info: Array) -> Dictionary:
	return {"type": "step", "obs": obs, "reward": reward, "done": done, "info": info}

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
# Socket timeouts (seconds). <= 0 disables the timeout (waits forever).
# read_timeout default 60s matches godot_rl's DEFAULT_TIMEOUT.
@export var connect_timeout_sec := 10.0
@export var read_timeout_sec := 60.0

# --- Expert-demo recording (control_mode == RECORD_EXPERT_DEMOS) ---
@export_global_file("*.json") var expert_demo_save_path: String = ""
@export_enum("gnrl_v1", "godot_rl") var demo_format: String = "gnrl_v1"  # default native; "godot_rl" = legacy/interop
# InputMap action that pops the last recorded episode (undo a bad demo). Acted on only if mapped.
@export var remove_last_episode_action: StringName = &"remove_last_demo_episode"
# Headless bound: after this many recorded actions, save + quit. 0 = unlimited (editor/human play).
@export var max_record_steps: int = 0

# godot_rl WIRE-PROTOCOL version (the handshake `major_version`/`minor_version`), NOT the
# godot_rl_agents pip PACKAGE version. These track godot_env.py's MAJOR_VERSION/MINOR_VERSION
# (currently "0"/"7" in package v0.8.2) and must match it to avoid handshake mismatch warnings.
# Bump these only when godot_rl bumps its protocol version — never to follow the package version.
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
# Opt-in step-phase profiler (cmdline `profile=true`); null = disabled (zero overhead).
var _profiler = null
var _profile_interval := 1000
var _recorder = null
var _record_agent = null
var _record_action_space: Dictionary = {}
var _demos_saved := false

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
	if args.get("profile", "false") == "true":
		_profiler = StepProfiler.new()
	Engine.physics_ticks_per_second = int(_get_speedup() * 60)
	Engine.time_scale = _get_speedup() * 1.0
	_set_heuristic("human", all_agents)
	if control_mode == ControlModes.RECORD_EXPERT_DEMOS:
		_initialize_demo_recording()
	else:
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

func _initialize_demo_recording() -> void:
	# godot_rl parity: a single agent is recorded. RECORD mode is OFFLINE — it never opens
	# the TCP socket, so the "training scene without a trainer hangs" gotcha does not apply.
	assert(all_agents.size() == 1,
		"RECORD_EXPERT_DEMOS records a single agent (got %d)" % all_agents.size())
	_record_agent = all_agents[0]
	# The agent was routed to agents_heuristic by _get_agents(); take it back so
	# _heuristic_process() doesn't also reset it — the agent's own _physics_process does.
	agents_heuristic.erase(_record_agent)
	_record_action_space = _record_agent.get_action_space()
	_recorder = DemoRecorder.new()

func _demo_record_process() -> void:
	if _recorder == null:
		return
	var obs_dict: Dictionary = _record_agent.get_obs()
	if not obs_dict.has("obs"):
		push_error("NcnnSync: recording agent get_obs() has no 'obs' key; image-obs demo recording is unsupported.")
		return
	var obs: Array = obs_dict["obs"]
	var action: Array = _record_agent.get_action()
	var done: bool = _record_agent.get_done()
	_recorder.record_step(obs, action, done)
	if done:
		_record_agent.set_done_false()
	if _remove_last_episode_pressed():
		_recorder.remove_last_episode()
	if max_record_steps > 0 and _recorder.step_count() >= max_record_steps:
		save_expert_demos()
		get_tree().quit()

func _remove_last_episode_pressed() -> bool:
	if String(remove_last_episode_action).is_empty():
		return false
	if not InputMap.has_action(remove_last_episode_action):
		return false
	return Input.is_action_just_pressed(remove_last_episode_action)

func save_expert_demos() -> void:
	if _recorder == null:
		return
	if expert_demo_save_path.is_empty():
		push_error("NcnnSync: expert_demo_save_path is empty; cannot save demos.")
		return
	var abs_path := ProjectSettings.globalize_path(expert_demo_save_path)
	DirAccess.make_dir_recursive_absolute(abs_path.get_base_dir())
	var f := FileAccess.open(abs_path, FileAccess.WRITE)
	if f == null:
		push_error("NcnnSync: cannot open expert_demo_save_path '%s'." % expert_demo_save_path)
		return
	f.store_line(_recorder.to_json(demo_format, _record_action_space))
	f.close()
	_demos_saved = true

func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		if _recorder != null and not _demos_saved and not expert_demo_save_path.is_empty():
			save_expert_demos()

func _physics_process(_delta) -> void:
	if n_action_steps % action_repeat != 0:
		n_action_steps += 1
		return
	n_action_steps += 1
	# Each process guards on its own agent list; only the active mode's list is populated.
	_training_process()
	_inference_process()
	_heuristic_process()
	_demo_record_process()

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
	var t_phase_start := Time.get_ticks_usec()
	var did_send := false
	if need_to_send_obs:
		need_to_send_obs = false
		var reward_arr := _get_reward_from_agents()
		var done_arr := _get_done_from_agents()
		var obs := _get_obs_from_agents(agents_training)
		var info_arr := _get_info_from_agents()
		var t_obs_done := Time.get_ticks_usec()
		_send_dict_as_json_message(build_step_message(obs, reward_arr, done_arr, info_arr))
		var t_sent := Time.get_ticks_usec()
		did_send = true
		if _profiler != null:
			_profiler.record("collect_obs", t_obs_done - t_phase_start)
			_profiler.record("serialize_send", t_sent - t_obs_done)
	var t_wait_start := Time.get_ticks_usec()
	handle_message()
	if _profiler != null and did_send:
		_profiler.record("await_action", Time.get_ticks_usec() - t_wait_start)
		_profiler.step_done()
		if _profiler.get_steps() % _profile_interval == 0:
			print(_profiler.format_report())

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
	if json_dict == null:  # read timeout / disconnect during startup — quit already queued
		return
	assert(json_dict["type"] == "handshake")
	if json_dict.get("major_version") != MAJOR_VERSION:
		push_warning("NcnnSync: major version mismatch (got %s, expected %s)" % [json_dict.get("major_version"), MAJOR_VERSION])
	if json_dict.get("minor_version") != MINOR_VERSION:
		push_warning("NcnnSync: minor version mismatch (got %s, expected %s)" % [json_dict.get("minor_version"), MINOR_VERSION])

func _send_env_info() -> void:
	var json_dict = _get_dict_json_message()
	if json_dict == null:  # read timeout / disconnect during startup — quit already queued
		return
	assert(json_dict["type"] == "env_info")
	_send_dict_as_json_message(build_env_info_message())

func _get_dict_json_message():
	var timeout_ms := _get_read_timeout_ms()
	var deadline := SocketTimeout.deadline_after(Time.get_ticks_msec(), timeout_ms)
	while stream.get_available_bytes() == 0:
		stream.poll()
		if stream.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			print("NcnnSync: server disconnected, closing")
			get_tree().quit()
			return null
		if SocketTimeout.is_expired(deadline, Time.get_ticks_msec()):
			push_error("NcnnSync: read timed out after %.1fs (no data from trainer); closing cleanly." % (timeout_ms / 1000.0))
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
	var timeout_ms := _get_connect_timeout_ms()
	var deadline := SocketTimeout.deadline_after(Time.get_ticks_msec(), timeout_ms)
	while stream.get_status() < StreamPeerTCP.STATUS_CONNECTED:
		stream.poll()
		if SocketTimeout.is_expired(deadline, Time.get_ticks_msec()):
			push_warning("NcnnSync: connect timed out after %.1fs on port %d; falling back to human controls." % [timeout_ms / 1000.0, _get_port()])
			stream.disconnect_from_host()
			return false
		OS.delay_msec(1)
	return stream.get_status() == StreamPeerTCP.STATUS_CONNECTED

func handle_message() -> bool:
	var message = _get_dict_json_message()
	if message == null:
		return false
	match message["type"]:
		"close":
			if _profiler != null:
				print(_profiler.format_report())
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

func _get_info_from_agents() -> Array:
	var infos := []
	for agent in agents_training:
		infos.append(agent.get_info())
	return infos

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

func _get_connect_timeout_ms() -> int:
	return int(args.get("connect_timeout", str(connect_timeout_sec)).to_float() * 1000.0)

func _get_read_timeout_ms() -> int:
	return int(args.get("read_timeout", str(read_timeout_sec)).to_float() * 1000.0)
