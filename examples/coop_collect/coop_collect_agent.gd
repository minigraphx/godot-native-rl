class_name CoopCollectAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

# One agent of a cooperative team (MA-POCA scaffold, #30). Identical policy/obs/action shape for
# every team member (parameter sharing), and every agent reads the SAME shared team reward from the
# game — so this trains as a cooperative baseline today via shared-policy PPO, and is the env the
# MA-POCA centralized critic (M2) will plug into. 5 discrete actions (stay + 4 cardinal moves).

const ACTION_KEY := "move"
const ACTION_COUNT := 5
const M = preload("res://examples/coop_collect/coop_collect_math.gd")

@export var game_path: NodePath
@export var agent_index := 0          ## 0 or 1 — which team member this is

var _game  # CoopCollectGame (duck-typed to avoid class_name scope issues headless)
var _action_index := 0

# --- Pure helpers ---
func action_to_velocity(idx: int, speed: float) -> Vector2:
	match idx:
		1: return Vector2(0.0, -speed)
		2: return Vector2(0.0, speed)
		3: return Vector2(-speed, 0.0)
		4: return Vector2(speed, 0.0)
		_: return Vector2.ZERO

func _obs_size() -> int:
	# own(2) + teammate(2) + item_count * (rel 2 + flag 1)
	var n_items: int = _game.item_count if _game != null else 4
	return 4 + n_items * 3

func _zero_obs() -> Array:
	var z: Array = []
	z.resize(_obs_size())
	z.fill(0.0)
	return z

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("CoopCollectAgent: game_path not set or invalid — producing zero observations.")

# --- godot_rl contract ---
func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "discrete"}}

func get_obs() -> Dictionary:
	if _game == null:
		return {"obs": _zero_obs()}
	var self_pos: Vector2 = _game.agent_pos(agent_index)
	var own: Array = M.own_pos_obs(self_pos, _game.arena_size)
	var teammate: Array = M.rel_obs(self_pos, _game.teammate_pos(agent_index), _game.item_norm)
	var items: Array = _game.items()
	var collected: Array = _game.collected()
	var blocks: Array = []
	for i in range(items.size()):
		blocks.append(M.item_block(self_pos, items[i], collected[i], _game.item_norm))
	return {"obs": M.assemble_obs(own, teammate, blocks)}

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	var idx := int(action[ACTION_KEY])
	assert(idx >= 0 and idx < ACTION_COUNT, "CoopCollectAgent: action index %d out of range [0, %d)" % [idx, ACTION_COUNT])
	_action_index = idx

# --- Runtime step ---
func _physics_process(delta: float) -> void:
	super._physics_process(delta)  # core.step (reset_after acts only as a safety net)
	if _game == null:
		return
	_game.set_agent_velocity(agent_index, action_to_velocity(_action_index, _game.move_speed))
	# Shared team reward: every agent adds the SAME per-frame value the game computed this frame
	# (game runs at priority -10, before us). Accumulates across the action_repeat window like the
	# other examples.
	reward += _game.team_reward()
	var terminal: bool = _game.is_terminal()
	if terminal or needs_reset:
		if terminal:
			done = true
		needs_reset = false
		reset()
		# One agent drives the world reset so both observe a consistent fresh episode.
		if agent_index == 0:
			_game.request_reset()
