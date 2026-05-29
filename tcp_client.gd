class_name TcpClientBridge
extends Node

signal connected(host: String, port: int)
signal disconnected()
signal request_failed(request_id: int, error_message: String)
signal response_received(request_id: int, actions: Array)

@export var server_host: String = "127.0.0.1"
@export_range(1, 65535, 1) var server_port: int = 11008
@export_range(1, 60000, 1) var connect_timeout_ms: int = 3000
@export_range(1, 60000, 1) var default_request_timeout_ms: int = 200
@export var reconnect_on_disconnect: bool = true
@export_range(1, 20, 1) var reconnect_attempts: int = 3
@export_range(1, 2000, 1) var reconnect_backoff_ms: int = 100

var _tcp := StreamPeerTCP.new()
var _recv_buffer: String = ""
var _pending_messages: Array[Dictionary] = []
var _next_request_id: int = 1

func connect_to_server() -> bool:
	if is_connected_to_server():
		return true

	_tcp.disconnect_from_host()
	_recv_buffer = ""
	_pending_messages.clear()

	var connect_err := _tcp.connect_to_host(server_host, server_port)
	if connect_err != OK and connect_err != ERR_BUSY:
		push_error("TcpClientBridge.connect_to_server: connect_to_host failed with code %d" % connect_err)
		return false

	var deadline := Time.get_ticks_msec() + connect_timeout_ms
	while Time.get_ticks_msec() <= deadline:
		_tcp.poll()
		var status := _tcp.get_status()
		if status == StreamPeerTCP.STATUS_CONNECTED:
			emit_signal("connected", server_host, server_port)
			return true
		if status == StreamPeerTCP.STATUS_ERROR:
			break
		OS.delay_msec(5)

	push_error("TcpClientBridge.connect_to_server: timed out connecting to %s:%d" % [server_host, server_port])
	_tcp.disconnect_from_host()
	return false

func disconnect_from_server() -> void:
	var was_connected := is_connected_to_server()
	_tcp.disconnect_from_host()
	_recv_buffer = ""
	_pending_messages.clear()
	if was_connected:
		emit_signal("disconnected")

func is_connected_to_server() -> bool:
	return _tcp.get_status() == StreamPeerTCP.STATUS_CONNECTED

func request_actions(observations_batch: Array, metadata: Dictionary = {}, timeout_ms: int = -1) -> Dictionary:
	var request_id := _next_request_id
	_next_request_id += 1

	if observations_batch.is_empty():
		var empty_error := "observations_batch is empty"
		emit_signal("request_failed", request_id, empty_error)
		return _error_result(request_id, empty_error)

	if not _ensure_connected():
		var connect_error := "not connected to training server"
		emit_signal("request_failed", request_id, connect_error)
		return _error_result(request_id, connect_error)

	var payload := {
		"type": "action_request",
		"request_id": request_id,
		"observations": observations_batch,
		"metadata": metadata,
	}

	var line := JSON.stringify(payload) + "\n"
	var send_err := _tcp.put_data(line.to_utf8_buffer())
	if send_err != OK:
		var send_error := "failed to send request (error %d)" % send_err
		emit_signal("request_failed", request_id, send_error)
		return _error_result(request_id, send_error)

	return _wait_for_response(request_id, timeout_ms)

func _ensure_connected() -> bool:
	if is_connected_to_server():
		return true

	if not reconnect_on_disconnect:
		return false

	for attempt in reconnect_attempts:
		if connect_to_server():
			return true
		if attempt < reconnect_attempts - 1:
			OS.delay_msec(reconnect_backoff_ms)

	return false

func _wait_for_response(request_id: int, timeout_ms: int) -> Dictionary:
	var effective_timeout := timeout_ms
	if effective_timeout <= 0:
		effective_timeout = default_request_timeout_ms

	var deadline := Time.get_ticks_msec() + effective_timeout
	while Time.get_ticks_msec() <= deadline:
		_tcp.poll()

		if _tcp.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			var disconnected_error := "connection lost while waiting for response"
			emit_signal("request_failed", request_id, disconnected_error)
			return _error_result(request_id, disconnected_error)

		_drain_socket_messages()
		var response := _take_response_for_request(request_id)
		if not response.is_empty():
			return _response_to_result(request_id, response)

		OS.delay_msec(1)

	var timeout_error := "timed out waiting for response"
	emit_signal("request_failed", request_id, timeout_error)
	return _error_result(request_id, timeout_error)

func _drain_socket_messages() -> void:
	while _tcp.get_available_bytes() > 0:
		var read_result := _tcp.get_data(_tcp.get_available_bytes())
		var read_err: int = read_result[0]
		if read_err != OK:
			return

		var read_bytes: PackedByteArray = read_result[1]
		if read_bytes.is_empty():
			return

		_recv_buffer += read_bytes.get_string_from_utf8()
		_extract_json_lines()

func _extract_json_lines() -> void:
	while true:
		var newline_index := _recv_buffer.find("\n")
		if newline_index == -1:
			return

		var line := _recv_buffer.substr(0, newline_index).strip_edges()
		_recv_buffer = _recv_buffer.substr(newline_index + 1)
		if line.is_empty():
			continue

		var parsed: Variant = JSON.parse_string(line)
		if typeof(parsed) != TYPE_DICTIONARY:
			push_warning("TcpClientBridge: ignoring non-dictionary JSON response line")
			continue

		_pending_messages.append(parsed as Dictionary)

func _take_response_for_request(request_id: int) -> Dictionary:
	for i in range(_pending_messages.size()):
		var message := _pending_messages[i]
		if message.get("request_id", request_id) == request_id:
			_pending_messages.remove_at(i)
			return message

	return {}

func _response_to_result(request_id: int, response: Dictionary) -> Dictionary:
	var response_ok := bool(response.get("ok", true))
	if not response_ok:
		var server_error := str(response.get("error", "server returned error"))
		emit_signal("request_failed", request_id, server_error)
		return _error_result(request_id, server_error)

	var actions_variant: Variant = response.get("actions", null)
	if typeof(actions_variant) != TYPE_ARRAY:
		var format_error := "response missing 'actions' Array"
		emit_signal("request_failed", request_id, format_error)
		return _error_result(request_id, format_error)

	var actions: Array = actions_variant
	emit_signal("response_received", request_id, actions)
	return {
		"ok": true,
		"request_id": request_id,
		"actions": actions,
	}

func _error_result(request_id: int, error_message: String) -> Dictionary:
	return {
		"ok": false,
		"request_id": request_id,
		"error": error_message,
		"actions": [],
	}
