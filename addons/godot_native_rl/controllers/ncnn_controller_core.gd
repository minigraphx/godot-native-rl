class_name NcnnControllerCore
extends RefCounted

# Node-agnostic episode + reward state machine shared by NcnnAIController2D/3D.
# Holds no Node references; reset_after is passed into step() by the wrapper so the
# wrapper stays the single source of truth for that exported value.

var done: bool = false
var reward: float = 0.0
var n_steps: int = 0
var needs_reset: bool = false
var heuristic: String = "human"
var reward_source = null

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

static func obs_space_from_obs(obs: Dictionary) -> Dictionary:
	return {"obs": {"size": [obs["obs"].size()], "space": "box"}}
