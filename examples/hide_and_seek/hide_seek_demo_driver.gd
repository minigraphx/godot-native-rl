extends Node
# Drives both hide & seek agents with random actions so the play scene shows movement + occlusion
# without a trainer. Manual-inspection only (not used in CI).

@export var action_count := 5

var _agents: Array = []
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	_rng.seed = 7
	_agents = get_tree().get_nodes_in_group("AGENT")

func _physics_process(_delta: float) -> void:
	for agent in _agents:
		agent.set_action({"move": _rng.randi_range(0, action_count - 1)})
