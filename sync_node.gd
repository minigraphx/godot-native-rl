class_name SyncNode
extends Node

signal batch_applied(agent_count: int)
signal batch_failed(error_message: String)

@export var training_enabled: bool = true
@export var tick_in_physics_process: bool = true
@export var agent_group_name: String = "ncnn_training_agents"
@export var observation_method_name: String = "collect_observation"
@export var action_method_name: String = "apply_training_action"
@export var tcp_client_path: NodePath
@export var auto_connect_tcp: bool = true
@export_range(1, 60000, 1) var request_timeout_ms: int = 200
@export var include_agent_paths_in_metadata: bool = true

var _tcp_client: TcpClientBridge
var _step_running: bool = false

func _ready() -> void:
	_resolve_tcp_client()
	if _tcp_client == null:
		return

	if auto_connect_tcp and not _tcp_client.is_connected_to_server():
		_tcp_client.connect_to_server()

func _process(_delta: float) -> void:
	if not tick_in_physics_process:
		_run_training_step()

func _physics_process(_delta: float) -> void:
	if tick_in_physics_process:
		_run_training_step()

func _run_training_step() -> void:
	if not training_enabled or _step_running:
		return
	if _tcp_client == null:
		_resolve_tcp_client()
		if _tcp_client == null:
			return

	_step_running = true

	var agents := _discover_training_agents()
	if agents.is_empty():
		_step_running = false
		return

	var ordered_agents: Array[Node] = []
	var observations_batch: Array = []
	for agent in agents:
		if not agent.has_method(observation_method_name) or not agent.has_method(action_method_name):
			continue

		var raw_observation: Variant = agent.call(observation_method_name)
		var normalized_observation: Variant = _normalize_observation(raw_observation)
		if normalized_observation == null:
			push_warning("SyncNode: skipping agent '%s' because observation is not Array/PackedFloat32Array" % agent.name)
			continue

		ordered_agents.append(agent)
		observations_batch.append(normalized_observation)

	if ordered_agents.is_empty():
		_step_running = false
		return

	var metadata := {
		"agent_count": ordered_agents.size(),
		"frame": Engine.get_frames_drawn(),
	}
	if include_agent_paths_in_metadata:
		var agent_paths: Array[String] = []
		for agent in ordered_agents:
			agent_paths.append(str(agent.get_path()))
		metadata["agent_paths"] = agent_paths

	var response := _tcp_client.request_actions(observations_batch, metadata, request_timeout_ms)
	if not bool(response.get("ok", false)):
		var error_message := str(response.get("error", "unknown error"))
		emit_signal("batch_failed", error_message)
		_step_running = false
		return

	var actions_variant: Variant = response.get("actions", null)
	if typeof(actions_variant) != TYPE_ARRAY:
		emit_signal("batch_failed", "response 'actions' is not an Array")
		_step_running = false
		return

	var actions: Array = actions_variant
	var apply_count := mini(ordered_agents.size(), actions.size())
	if apply_count != ordered_agents.size():
		push_warning("SyncNode: received %d actions for %d agents" % [actions.size(), ordered_agents.size()])

	for i in range(apply_count):
		ordered_agents[i].call(action_method_name, actions[i])

	emit_signal("batch_applied", apply_count)
	_step_running = false

func _resolve_tcp_client() -> void:
	if tcp_client_path.is_empty():
		push_error("SyncNode: tcp_client_path is empty")
		_tcp_client = null
		return

	var node := get_node_or_null(tcp_client_path)
	if node == null:
		push_error("SyncNode: no node found at tcp_client_path")
		_tcp_client = null
		return

	var typed_node := node as TcpClientBridge
	if typed_node == null:
		push_error("SyncNode: node at tcp_client_path is not TcpClientBridge")
		_tcp_client = null
		return

	_tcp_client = typed_node

func _discover_training_agents() -> Array[Node]:
	var candidates := get_tree().get_nodes_in_group(agent_group_name)
	var agents: Array[Node] = []
	for candidate in candidates:
		if candidate is Node:
			agents.append(candidate as Node)
	return agents

func _normalize_observation(raw_observation: Variant) -> Variant:
	if raw_observation is PackedFloat32Array:
		return Array(raw_observation)
	if raw_observation is Array:
		return raw_observation
	return null
