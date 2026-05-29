class_name NcnnAIController2D
extends Node2D

enum ControlModes { INHERIT_FROM_SYNC, HUMAN, TRAINING }
@export var control_mode: ControlModes = ControlModes.INHERIT_FROM_SYNC  # read by NcnnSync
@export var reset_after := 1000

var heuristic := "human"
var done := false
var reward := 0.0
var n_steps := 0
var needs_reset := false

func _ready() -> void:
	add_to_group("AGENT")

# --- Abstract: implemented by the concrete agent ---
func get_obs() -> Dictionary:
	assert(false, "get_obs must be implemented by the agent extending NcnnAIController2D")
	return {"obs": []}

func get_reward() -> float:
	assert(false, "get_reward must be implemented by the agent extending NcnnAIController2D")
	return 0.0

func get_action_space() -> Dictionary:
	assert(false, "get_action_space must be implemented by the agent extending NcnnAIController2D")
	return {}

func set_action(_action) -> void:
	assert(false, "set_action must be implemented by the agent extending NcnnAIController2D")

# --- Concrete contract methods used by NcnnSync ---
func get_obs_space() -> Dictionary:
	var obs := get_obs()
	return {"obs": {"size": [obs["obs"].size()], "space": "box"}}

func reset() -> void:
	n_steps = 0
	needs_reset = false

func reset_if_done() -> void:
	if done:
		reset()

func set_heuristic(h: String) -> void:
	heuristic = h

func get_done() -> bool:
	return done

func set_done_false() -> void:
	done = false

func zero_reward() -> void:
	reward = 0.0

func _physics_process(_delta) -> void:
	n_steps += 1
	if n_steps > reset_after:
		needs_reset = true
